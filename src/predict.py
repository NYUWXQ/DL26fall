"""Inference: multi-view TTA, optional ensemble, Kaggle submission CSV.
推理：多视角 TTA、可选集成、生成 Kaggle 提交 CSV。

Design notes / 设计说明
- Views: plain, horizontal flip, and 5 shifted crops (reflect pad 2) => 10 views.
  视角：原图、水平翻转，以及 reflect pad 2 后的 5 个平移裁剪 => 共 10 个视角。
- Softmax probabilities are averaged across views and checkpoints (ensembling).
  各视角与各 checkpoint 的 softmax 概率取平均（集成）。
- The CSV keeps sample_submission.csv row order and writes id,label.
  CSV 保持 sample_submission.csv 的行顺序，列名为 id,label。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import CifarNumpy, MEAN, STD, load_test  # noqa: E402
from models import build_model, check_budget  # noqa: E402

VIEWS = [(0, 0), (0, 4), (4, 0), (4, 4)]


@torch.no_grad()
def view_probs(model: torch.nn.Module, x: torch.Tensor, mode: str) -> torch.Tensor:
    """Average softmax over TTA views. / 对 TTA 各视角的 softmax 取平均。"""
    views = [x]
    if mode in ("translate", "translate_flip"):
        padded = F.pad(x, (2, 2, 2, 2), mode="reflect")
        views += [padded[:, :, y:y + 32, xx:xx + 32] for y, xx in VIEWS]
    if mode in ("flip", "translate_flip"):
        views = views + [v.flip(-1) for v in views]
    probs = [model(v).float().softmax(1) for v in views]
    return torch.stack(probs).mean(0)


def _n_train_rows(root: str) -> int:
    """Number of training rows in the manifest (cheap CSV header count).
    清单里的训练样本行数（只数 CSV 行，不解码图片）。"""
    path = os.path.join(root, "train_labels.csv")
    if not os.path.exists(path):
        return 0
    with open(path, newline="") as fh:
        return max(0, sum(1 for _ in csv.DictReader(fh)))


def ema_converged(ck: Dict, root: str, threshold: float = 0.01) -> Tuple[bool, float]:
    """Has the EMA shadow actually forgotten the random initialisation?
    EMA 影子权重是否已经真正忘掉随机初值？

    train.py does not store the step count, so we recompute it exactly the way
    train.py would: n_train rows minus the 2500-image validation split,
    floor-divided by the batch size (the loader drops the last partial batch).
    train.py 没有保存步数，因此我们按 train.py 的方式精确重算：
    训练行数减去 2500 张验证集，再按批大小整除（loader 会丢弃最后的不完整批）。

    Returns (converged, residual_init_weight). / 返回（是否收敛，残差初值权重）。
    """
    args = ck.get("args", {}) or {}
    decay = float(args.get("ema_decay", 0.999))
    epoch = int(ck.get("epoch", 0) or 0)
    batch_size = int(args.get("batch_size", 0) or 0)
    n_train = _n_train_rows(root) - 2500  # data.load_train default n_val=2500
    if n_train <= 0 or batch_size <= 0 or epoch <= 0:
        return (False, 1.0)
    steps = epoch * (n_train // batch_size)
    residual = decay ** steps
    return (residual < threshold, residual)


def load_ckpt(path: str, device: torch.device, which: str = "auto", root: str = ""):
    """Load a checkpoint saved by train.py and return (model, meta).
    加载 train.py 保存的 checkpoint，返回 (model, meta)。

    "auto" does NOT blindly prefer EMA. With decay=0.9995 the EMA shadow still
    holds a large fraction of the random initialisation early in training
    (0.9995 ** steps), so EMA can be far worse than the raw weights before the
    EMA has caught up. Measured on this run at epoch 21: EMA 18.8% vs raw
    80.9% validation accuracy.
    "auto" 不会盲目偏好 EMA。decay=0.9995 时，早期 EMA 影子权重仍含很大比例的
    随机初值（0.9995 ** steps），在 EMA 追上之前可能远差于原始权重。
    本次运行 epoch 21 实测：EMA 18.8%，raw 80.9%。

    The EMA shadow is only preferred once it has actually converged, i.e. when
    the residual initialisation weight has decayed below ~1%.
    只有当残差初值权重衰减到约 1% 以下（EMA 确实收敛）时才优先使用 EMA。
    """
    ck = torch.load(path, map_location=device, weights_only=False)
    args = ck.get("args", {})
    name = args.get("model", "preact22x4")
    model = build_model(name, drop=args.get("drop", 0.0)).to(device)
    residual = 1.0
    if which == "ema":
        sd, tag = ck["ema"], "ema"
    elif which == "raw":
        sd, tag = ck["model"], "raw"
    else:  # auto
        converged, residual = ema_converged(ck, root or ".")
        if "ema" in ck and converged:
            sd, tag = ck["ema"], "ema"
        else:
            sd, tag = ck["model"], "raw"
    model.load_state_dict(sd)
    model.eval()
    return model, {"path": path, "weights": tag, "model": name, "params": check_budget(model),
                   "ema_residual_init_weight": round(residual, 6)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--ckpt", nargs="+", required=True, help="checkpoints; more than one = ensemble / 多个则集成")
    ap.add_argument("--weights", default="auto", choices=["auto", "ema", "raw"])
    ap.add_argument("--tta", default="translate_flip", choices=["plain", "flip", "translate", "translate_flip"])
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--out", default="submissions/submission.csv")
    ap.add_argument("--probs-out", default="", help="optional .npy of averaged probabilities / 可选的概率矩阵")
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    x, ids = load_test(args.root)
    ds = CifarNumpy(x, np.zeros(len(x), dtype=np.int64), train=False)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    acc = np.zeros((len(x), 10), dtype=np.float64)
    metas = []
    for path in args.ckpt:
        model, meta = load_ckpt(path, device, args.weights, root=args.root)
        metas.append(meta)
        print(f"[ckpt] {meta['path']} weights={meta['weights']} params={meta['params']:,} "
              f"ema_residual_init={meta['ema_residual_init_weight']}", flush=True)
        offset = 0
        for xb, _ in loader:
            xb = xb.to(device)
            # MPS has no float64, so widen to float64 on the CPU, not on device.
            # MPS 不支持 float64，因此在 CPU 上扩展为 float64，而不是在设备上。
            p = view_probs(model, xb, args.tta).float().cpu().numpy().astype(np.float64)
            acc[offset:offset + len(p)] += p
            offset += len(p)
        del model
    probs = acc / len(args.ckpt)
    pred = probs.argmax(1)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(os.path.join(args.root, "sample_submission.csv"), newline="") as fh:
        order = [r["id"] for r in csv.DictReader(fh)]
    pos = {i: k for k, i in enumerate(ids)}
    with open(args.out, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["id", "label"])
        for i in order:
            wr.writerow([i, int(pred[pos[i]])])

    if args.probs_out:
        np.save(args.probs_out, probs)
    meta_path = os.path.splitext(args.out)[0] + ".json"
    with open(meta_path, "w") as fh:
        json.dump({"ckpts": metas, "tta": args.tta, "pred_hist": np.bincount(pred, minlength=10).tolist(),
                   "mean_max_prob": float(probs.max(1).mean()),
                   "sha256": hashlib.sha256(open(args.out, "rb").read()).hexdigest()}, fh, indent=1)
    print(f"[done] {args.out} rows={len(order)} mean_max_prob={probs.max(1).mean():.4f}")
    print(f"[done] label histogram {np.bincount(pred, minlength=10).tolist()}")


if __name__ == "__main__":
    main()
