"""Training loop: SGD + cosine schedule + label smoothing + mixup/cutmix + EMA.
训练主循环：SGD + 余弦退火 + 标签平滑 + mixup/cutmix + EMA。

Everything is explicit (no external training frameworks) so the report can
cite exactly what was run. / 全部显式实现（不依赖外部训练框架），便于报告精确引用。
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import CifarNumpy, load_train, make_loader  # noqa: E402
from models import build_model, check_budget  # noqa: E402


class EMA:
    """Exponential moving average of weights, evaluated in eval mode at test time.
    权重的指数滑动平均，测试时在 eval 模式下使用。"""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.decay
        for s, p in zip(self.shadow.state_dict().values(), model.state_dict().values()):
            if s.dtype.is_floating_point:
                s.mul_(d).add_(p.detach(), alpha=1 - d)
            else:
                s.copy_(p)


def lr_at(step: int, total: int, base_lr: float, warmup: int) -> float:
    """Linear warmup then cosine decay to zero. / 线性预热后余弦退火至 0。"""
    if step < warmup:
        return base_lr * step / max(1, warmup)
    t = (step - warmup) / max(1, total - warmup)
    return 0.5 * base_lr * (1 + math.cos(math.pi * t))


def rand_bbox(h: int, w: int, lam: float) -> tuple[int, int, int, int]:
    """Random box for CutMix. / CutMix 的随机框。"""
    cut_rat = math.sqrt(1.0 - lam)
    cut_h, cut_w = int(h * cut_rat), int(w * cut_rat)
    cy, cx = np.random.randint(h), np.random.randint(w)
    y1, y2 = np.clip([cy - cut_h // 2, cy + cut_h // 2], 0, h)
    x1, x2 = np.clip([cx - cut_w // 2, cx + cut_w // 2], 0, w)
    return int(y1), int(y2), int(x1), int(x2)


def mix_batch(x: torch.Tensor, y: torch.Tensor, mode: str, alpha: float):
    """Return mixed inputs and the (target_a, target_b, lam) triple.
    返回混合后的输入与 (target_a, target_b, lam) 三元组。"""
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    perm = torch.randperm(x.size(0), device=x.device)
    if mode == "mixup":
        return lam * x + (1 - lam) * x[perm], y, y[perm], lam
    if mode == "cutmix":
        y1, y2, x1, x2 = rand_bbox(x.size(2), x.size(3), lam)
        x[:, :, y1:y2, x1:x2] = x[perm, :, y1:y2, x1:x2]
        lam = 1 - ((y2 - y1) * (x2 - x1) / (x.size(2) * x.size(3)))
        return x, y, y[perm], lam
    return x, y, y, 1.0


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device) -> float:
    """Top-1 accuracy on a loader. / 在给定 loader 上的 Top-1 准确率。"""
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(1, total)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--model", default="preact16x4")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--mix", default="both", choices=["none", "mixup", "cutmix", "both"])
    ap.add_argument("--mix-alpha", type=float, default=1.0)
    ap.add_argument("--mixup-alpha", type=float, default=0.2, help="beta alpha for MixUp / MixUp 的 beta 参数")
    ap.add_argument("--cutmix-alpha", type=float, default=1.0, help="beta alpha for CutMix / CutMix 的 beta 参数")
    ap.add_argument("--aug", default="strong", choices=["basic", "strong"])
    ap.add_argument("--randaug-m", type=int, default=9, help="RandAugment magnitude / RandAugment 强度")
    ap.add_argument("--erase-p", type=float, default=0.25, help="RandomErasing probability / 随机擦除概率")
    ap.add_argument("--full-data", action="store_true", help="train on all 50k, no holdout / 全量训练不留验证集")
    ap.add_argument("--mix-always", action="store_true", help="mix every step (teammate recipe) / 每步都混合")
    ap.add_argument("--ema-dynamic", action="store_true", help="decay=min(0.999,(1+step)/(10+step)) / 动态 EMA")
    ap.add_argument("--eval-ood", default="", help="optional .npy dir with cifar-10.1 data/labels for OOD monitoring / OOD 监控")
    ap.add_argument("--drop", type=float, default=0.0)
    ap.add_argument("--ema-decay", type=float, default=0.999)
    ap.add_argument("--out", default="runs/run")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--resume", action="store_true",
                    help="resume from <out>/last.pt / 从 <out>/last.pt 续训")
    ap.add_argument("--limit-batches", type=int, default=0, help="debug: cap batches per epoch / 调试：限制每 epoch 批数")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("mps" if (args.device == "auto" and torch.backends.mps.is_available())
                          else ("cuda" if (args.device == "auto" and torch.cuda.is_available()) else "cpu"))
    if args.device != "auto":
        device = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)

    # The validation split is fixed at data-seed 0 so that every run (and therefore
    # every ensemble member) is comparable on the same 2,500 images; --seed only
    # changes initialisation and augmentation randomness.
    # 验证集固定用 seed=0 划分，保证所有实验/集成成员在同一批 2500 张上可比；
    # --seed 只改变初始化与增强随机性。
    if args.full_data:
        # Full-data run: train on all 50,000 images; validation falls back to the
        # same 2,500 held-out images (now seen in training) purely for monitoring.
        # 全量训练：用全部 5 万张；验证集退回同一批 2500 张（已被训练见过），仅用于监控。
        from data import build_cache
        tr_x, tr_y = build_cache(args.root, "train")
        va_x, va_y = tr_x[:2500], tr_y[:2500]
        print("[data] FULL-DATA mode: 50000 train images (val is in-sample)", flush=True)
    else:
        tr_x, tr_y, va_x, va_y = load_train(args.root, seed=0)
    train_ds = CifarNumpy(tr_x, tr_y, train=True, aug=args.aug, randaug_m=args.randaug_m,
                          erase_p=args.erase_p)
    val_ds = CifarNumpy(va_x, va_y, train=False)
    train_ld = make_loader(train_ds, args.batch_size, True, args.workers)
    val_ld = make_loader(val_ds, 512, False, args.workers)

    # A resumed run must rebuild the SAME architecture that wrote the checkpoint,
    # otherwise state_dict shapes mismatch and resume fails.
    # 续训必须重建与 checkpoint 完全一致的结构，否则 state_dict 形状不匹配会直接失败。
    ckpt_path = os.path.join(args.out, "last.pt")
    resume_ck = None
    if args.resume and os.path.exists(ckpt_path):
        resume_ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        saved = resume_ck.get("args", {})
        for key in ("model", "drop", "aug", "mix", "mix_alpha", "ema_decay", "label_smoothing"):
            if key in saved:
                setattr(args, key, saved[key])
        print(f"[resume] architecture from checkpoint: {args.model} drop={args.drop} aug={args.aug} mix={args.mix}", flush=True)

    model = build_model(args.model, drop=args.drop).to(device)
    n = check_budget(model)  # hard 5M cap / 硬性 5M 限制
    print(f"[model] {args.model} params={n:,} device={device} train={len(tr_y)} val={len(va_y)}", flush=True)

    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                          weight_decay=args.weight_decay, nesterov=True)
    crit = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    ema = EMA(model, args.ema_decay)

    total_steps = max(1, len(train_ld) * args.epochs)
    step = 0
    best = 0.0
    history = []
    start_epoch = 0

    # Resume support: restore weights, EMA, optimizer, step and history.
    # 续训支持：恢复权重、EMA、优化器、步数与历史记录。
    if resume_ck is not None:
        ck = resume_ck
        model.load_state_dict(ck["model"])
        if "ema" in ck:
            ema.shadow.load_state_dict(ck["ema"])
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
        start_epoch = int(ck.get("epoch", 0))
        step = int(ck.get("step", start_epoch * len(train_ld)))
        best = float(ck.get("best", 0.0))
        hist_path = os.path.join(args.out, "history.json")
        if os.path.exists(hist_path):
            with open(hist_path) as fh:
                history = json.load(fh)
        print(f"[resume] from {ckpt_path} at epoch {start_epoch} step {step} best {best:.4f}", flush=True)

    t0 = time.time()
    for epoch in range(start_epoch, args.epochs):
        model.train()
        run_loss, seen = 0.0, 0
        for bi, (x, y) in enumerate(train_ld):
            if args.limit_batches and bi >= args.limit_batches:
                break
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            for g in opt.param_groups:
                g["lr"] = lr_at(step, total_steps, args.lr, args.warmup * max(1, len(train_ld)))

            if args.mix == "mixup":
                xb, ya, yb, lam = mix_batch(x, y, "mixup", args.mixup_alpha)
            elif args.mix == "cutmix":
                xb, ya, yb, lam = mix_batch(x, y, "cutmix", args.cutmix_alpha)
            elif args.mix == "both" and (args.mix_always or np.random.rand() < 0.5):
                mode = "mixup" if np.random.rand() < 0.5 else "cutmix"
                alpha = args.mixup_alpha if mode == "mixup" else args.cutmix_alpha
                xb, ya, yb, lam = mix_batch(x, y, mode, alpha)
            else:
                xb, ya, yb, lam = x, y, y, 1.0

            out = model(xb)
            loss = lam * crit(out, ya) + (1 - lam) * crit(out, yb) if lam < 1.0 else crit(out, ya)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if args.ema_dynamic:
                ema.decay = min(0.999, (1 + step) / (10 + step))
            ema.update(model)
            run_loss += loss.item() * y.numel()
            seen += y.numel()
            step += 1

        msg = f"[ep {epoch+1:03d}/{args.epochs}] loss={run_loss/max(1,seen):.4f} lr={opt.param_groups[0]['lr']:.4f} {time.time()-t0:.0f}s"
        # Validation every 5 epochs (and on the last one) to keep epochs fast.
        # 每 5 个 epoch（及最后一个）验证一次，保持训练速度。
        if (epoch + 1) % 5 == 0 or epoch + 1 == args.epochs:
            acc_raw = evaluate(model, val_ld, device)
            acc_ema = evaluate(ema.shadow, val_ld, device)
            msg += f" val={acc_raw:.4f} val_ema={acc_ema:.4f}"
            best = max(best, acc_raw, acc_ema)
            history.append({"epoch": epoch + 1, "loss": run_loss / max(1, seen),
                            "val": acc_raw, "val_ema": acc_ema})
            with open(os.path.join(args.out, "history.json"), "w") as fh:
                json.dump(history, fh, indent=1)
        print(msg, flush=True)
        torch.save({"model": model.state_dict(), "ema": ema.shadow.state_dict(), "args": vars(args),
                    "opt": opt.state_dict(), "step": step, "epoch": epoch + 1, "best": best},
                   os.path.join(args.out, "last.pt"))

    torch.save({"model": model.state_dict(), "ema": ema.shadow.state_dict(), "args": vars(args),
                "best": best}, os.path.join(args.out, "final.pt"))
    print(f"[done] best_val={best:.4f} params={n:,} elapsed={(time.time()-t0)/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
