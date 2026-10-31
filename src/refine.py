"""Inference-configuration comparison: plain / flip / 10-view TTA / BN recalibration.
推理配置对比：plain / flip / 10-view TTA / BN 重校准。

Scope / 范围
------------
* Reads checkpoints from ``runs/`` and NEVER writes inside ``runs/``.
  从 ``runs/`` 读取 checkpoint，绝不向 ``runs/`` 写入任何内容。
* Does not touch ``src/train.py``; it only re-uses the public helpers
  ``data.load_train`` / ``models.build_model`` / ``predict.view_probs``.
  不修改 ``src/train.py``，只复用 data / models / predict 的公开函数。
* The validation split is byte-identical to the one ``train.py`` uses
  (``seed=0``, ``n_val=2500``), so the numbers below are directly
  comparable with ``runs/*/history.json``.
  验证集划分与 train.py 完全一致（seed=0、n_val=2500），
  因此本文件的数字可与 runs/*/history.json 直接对比。

Everything printed here is measured, never assumed.
这里输出的每个数字都是实测的，不是推测的。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import tempfile
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import CifarNumpy, load_test, load_train, make_loader  # noqa: E402
from models import build_model, check_budget  # noqa: E402
from predict import view_probs  # noqa: E402  (same TTA semantics as the submission path)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# TTA modes under test / 待比较的 TTA 模式
#   plain        : 1 view          / 单视角
#   flip         : 2 views         / 原图 + 水平翻转
#   translate    : 5 views         / 原图 + 4 个平移裁剪
#   translate_flip: 10 views       / 上面 5 个视角再各自翻转
TTA_MODES = ["plain", "flip", "translate", "translate_flip"]


# --------------------------------------------------------------------------
# checkpoint handling / 检查点处理
# --------------------------------------------------------------------------
def snapshot_ckpt(path: str, tries: int = 6) -> str:
    """Copy a checkpoint that a training job may be writing right now.
    把可能正被训练进程写入的 checkpoint 先复制成一份快照。

    The training loop overwrites ``last.pt`` every epoch, so a naive read can
    hit a half-written file. We copy, then validate by loading the copy.
    训练循环每个 epoch 都会覆写 last.pt，直接读可能拿到写了一半的文件。
    这里先复制再校验，失败则重试。
    """
    last_err: Optional[Exception] = None
    for attempt in range(tries):
        fd, tmp = tempfile.mkstemp(suffix=".pt", prefix="refine_ckpt_")
        os.close(fd)
        try:
            shutil.copyfile(path, tmp)
            torch.load(tmp, map_location="cpu", weights_only=False)
            return tmp
        except Exception as exc:  # noqa: BLE001 - report and retry
            last_err = exc
            try:
                os.unlink(tmp)
            except OSError:
                pass
            time.sleep(1.0)
    raise RuntimeError(f"could not obtain a readable snapshot of {path}: {last_err}")


def load_model_from_state(ck: Dict, device: torch.device, which: str,
                          ckpt_path: str) -> Tuple[torch.nn.Module, Dict]:
    """Build a model from an already-loaded checkpoint dict.
    从已加载的 checkpoint 字典构建模型。

    All weight variants must come from the SAME checkpoint dict, otherwise the
    comparison mixes different training epochs. The training job rewrites
    `last.pt` every epoch, so re-reading the file per variant is not safe.
    所有权重变体必须来自同一个 checkpoint 字典，否则比较会混入不同 epoch。
    训练进程每个 epoch 都会重写 last.pt，逐变体重读文件并不安全。
    """
    args = ck.get("args", {}) or {}
    name = args.get("model", "preact22x4")
    model = build_model(name, drop=args.get("drop", 0.0)).to(device)
    if which == "ema" and "ema" in ck:
        sd, tag = ck["ema"], "ema"
    else:
        sd, tag = ck["model"], "raw"
    model.load_state_dict(sd)
    model.eval()
    meta = {
        "path": ckpt_path,
        "weights": tag,
        "model": name,
        "params": check_budget(model),
        "epoch": ck.get("epoch"),
        "train_args": args,
    }
    return model, meta


def load_ckpt_dict(ckpt_path: str) -> Dict:
    """Load a checkpoint into memory via a validated snapshot. / 通过校验过的快照把 checkpoint 读进内存。"""
    tmp = snapshot_ckpt(ckpt_path)
    try:
        return torch.load(tmp, map_location="cpu", weights_only=False)
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def load_model(ckpt_path: str, device: torch.device, which: str = "ema") -> Tuple[torch.nn.Module, Dict]:
    """Load a checkpoint and return (model, meta). / 加载 checkpoint，返回 (模型, 元信息)。"""
    ck = load_ckpt_dict(ckpt_path)
    return load_model_from_state(ck, device, which, ckpt_path)


def pick_default_ckpt(root: str) -> str:
    """Pick the most recently written checkpoint under runs/ (excluding smoke tests).
    选出 runs/ 下最近写入的 checkpoint（排除 smoke 调试跑）。"""
    runs = os.path.join(root, "runs")
    cands: List[Tuple[float, str]] = []
    for dirpath, _dirnames, filenames in os.walk(runs):
        base = os.path.basename(dirpath)
        if base in {"smoke", "bench"}:
            continue
        for fn in filenames:
            if fn in {"last.pt", "final.pt"}:
                p = os.path.join(dirpath, fn)
                cands.append((os.path.getmtime(p), p))
    if not cands:
        raise FileNotFoundError(f"no checkpoint found under {runs}")
    cands.sort()
    return cands[-1][1]


# --------------------------------------------------------------------------
# BatchNorm recalibration / BatchNorm 统计量重校准
# --------------------------------------------------------------------------
@torch.no_grad()
def bn_recalibrate(model: torch.nn.Module, loader, device: torch.device,
                   max_batches: int = 0) -> int:
    """Re-estimate BatchNorm running statistics with fresh forward passes.
    用新的前向传播重新估计 BatchNorm 的运行统计量。

    How / 做法
    -----------
    1. reset_running_stats() + momentum=None  => cumulative moving average.
       重置统计量并把 momentum 设为 None，从而使用累积滑动平均。
    2. model.train() makes BN use batch statistics (the usual re-estimation
       recipe); dropout is inactive here because drop=0.0.
       train() 模式下 BN 使用批统计量（标准做法）；本项目 drop=0.0，无 dropout 副作用。
    3. The forward pass must use the SAME preprocessing as evaluation.
       前向必须使用与评估完全一致的预处理。

    Returns the number of batches used. / 返回使用的批数。
    """
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.reset_running_stats()
            m.momentum = None  # cumulative moving average / 累积滑动平均
    model.train()
    used = 0
    for i, (xb, _y) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        model(xb.to(device))
        used += 1
    model.eval()
    return used


# --------------------------------------------------------------------------
# metrics / 指标
# --------------------------------------------------------------------------
def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson 95% interval for a binomial proportion. / 二项比例的 Wilson 95% 区间。

    The Wilson interval always contains the point estimate in its interior, so
    clamping to it only repairs floating-point noise at the boundaries
    (e.g. k == n should give an upper bound of exactly 1.0).
    Wilson 区间数学上总是包含点估计，因此这里的钳制只是修复边界处的浮点误差
    （例如 k == n 时上界应恰为 1.0）。
    """
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    lo = max(0.0, (centre - half) / d)
    hi = min(1.0, (centre + half) / d)
    lo = min(lo, p)
    hi = max(hi, p)
    return (lo, hi)


def ece(probs: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    """Expected calibration error over equal-width confidence bins.
    等宽置信度分箱下的期望标定误差。"""
    conf = probs.max(1)
    correct = (probs.argmax(1) == y).astype(np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if not m.any():
            continue
        total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def nll(probs: np.ndarray, y: np.ndarray) -> float:
    """Mean negative log-likelihood of the true class. / 真实类别的平均负对数似然。"""
    p = np.clip(probs[np.arange(len(y)), y], 1e-12, 1.0)
    return float(-np.log(p).mean())


def fit_temperature(probs: np.ndarray, y: np.ndarray,
                   t_min: float = 0.05, t_max: float = 20.0) -> float:
    """Fit a single temperature on (probs, y) by minimising NLL.
    用 NLL 最小化拟合一个温度参数 T（对数概率先除以 T 再 softmax）。

    The search is bounded to [t_min, t_max]. Without a bound the optimiser
    can run away towards T -> 0 when the inputs are already very peaked,
    because dividing an already-saturated log-probability vector by a tiny
    temperature keeps the cross-entropy at its limit while the gradient
    keeps shrinking. The bound keeps the fitted value interpretable.
    搜索范围被限制在 [t_min, t_max]。若不加限制，当输入概率已极度尖锐时，
    优化器可能朝 T -> 0 逃逸：把已饱和的对数概率除以极小温度后交叉熵不再下降，
    而梯度持续变小。有界后拟合值才可解释。
    """
    logits = torch.from_numpy(np.log(np.clip(probs, 1e-12, 1.0))).float()
    yt = torch.from_numpy(y).long()
    # Parameterise T directly and project it back into range after each step.
    # 直接参数化 T，每步后投影回合法区间。
    t = torch.ones(1, requires_grad=True)
    opt = torch.optim.LBFGS([t], lr=0.1, max_iter=100,
                            tolerance_grad=1e-9, tolerance_change=1e-12)

    def closure():
        opt.zero_grad()
        tt = t.clamp(t_min, t_max)
        loss = torch.nn.functional.cross_entropy(logits / tt, yt)
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        return float(t.clamp(t_min, t_max).item())


def mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided McNemar test on discordant pairs.
    对不一致配对做精确二项检验（McNemar 精确法）。

    b = cases where A is right and B is wrong; c = the opposite.
    b = A 对 B 错的样本数；c = A 错 B 对的样本数。
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


# --------------------------------------------------------------------------
# pixel-level train/test comparison / 像素级 train/test 对比
# --------------------------------------------------------------------------
def pixel_shift_report(tr_x: np.ndarray, tr_y: np.ndarray,
                        te_x: np.ndarray) -> Dict:
    """Compare the raw pixel statistics of the train and test images.
    对比 train 与 test 图片的原始像素统计量。

    A brightness gap between the two sets is the cheapest possible test for a
    train/test domain shift: if the competition's PNGs were rendered or
    encoded differently from the training ones, the first-order statistics
    would differ.
    两套图片的亮度差是检验 train/test 域偏移最廉价的方法：若竞赛 PNG 的渲染或
    编码方式与训练集不同，一阶统计量就会不同。

    To tell a real shift from class-mix noise we also compare the observed gap
    against the sampling error expected from drawing 2000 images with a
    different class mix.
    为了区分「真实偏移」与「类别比例抽样噪声」，我们会把实测差值与「2000 张
    按不同类别比例抽样」所期望的抽样误差做对比。
    """
    ft = tr_x.astype(np.float64) / 255.0
    fe = te_x.astype(np.float64) / 255.0
    tr_mean, te_mean = float(ft.mean()), float(fe.mean())
    tr_std, te_std = float(ft.std()), float(fe.std())

    # Expected sampling error of the mean under a random class mix.
    # 随机类别比例下均值的期望抽样误差。
    class_means = np.array([ft[tr_y == c].mean() for c in range(10)])
    fracs = np.array([(tr_y == c).mean() for c in range(10)])
    mix_err = float(np.sqrt(np.sum((fracs ** 2) * (class_means.std() ** 2)) / len(fe)))

    gap = te_mean - tr_mean
    return {
        "train_mean": tr_mean,
        "test_mean": te_mean,
        "mean_gap": gap,
        "mean_gap_levels": gap * 255.0,
        "train_std": tr_std,
        "test_std": te_std,
        "std_gap": te_std - tr_std,
        "class_mix_sampling_error": mix_err,
        "gap_over_sampling_error": abs(gap) / mix_err if mix_err > 0 else float("inf"),
        "per_class_mean": {int(c): float(class_means[c]) for c in range(10)},
    }


# --------------------------------------------------------------------------
# evaluation helpers / 评估辅助
# --------------------------------------------------------------------------
@torch.no_grad()
def probs_for(model: torch.nn.Module, loader, device: torch.device, mode: str) -> np.ndarray:
    """Average softmax probabilities over the requested TTA views.
    按指定 TTA 视角平均 softmax 概率。"""
    out = []
    for xb, _y in loader:
        # MPS has no float64: cast to float32 on device, widen on CPU.
        # MPS 不支持 float64：设备上转 float32，回到 CPU 再扩展为 float64。
        p = view_probs(model, xb.to(device), mode).float().cpu().numpy()
        out.append(p.astype(np.float64))
    return np.concatenate(out, axis=0)


def describe(name: str, weights: str, tta: str, bn: str, probs: np.ndarray, y: np.ndarray) -> Dict:
    """Assemble one comparison row. / 组装一行对比结果。"""
    pred = probs.argmax(1)
    correct = int((pred == y).sum())
    n = int(len(y))
    lo, hi = wilson_ci(correct, n)
    return {
        "config": name,
        "weights": weights,
        "tta": tta,
        "bn_recal": bn,
        "acc": correct / n,
        "acc_correct": correct,
        "n": n,
        "acc_ci95": [lo, hi],
        "nll": nll(probs, y),
        "ece": ece(probs, y),
        "disagree_vs_plain": None,   # filled in later against the plain baseline
        "mcnemar_p_vs_plain": None,  # 同上
        "mean_max_prob": float(probs.max(1).mean()),
        "_probs": probs,
        "_pred": pred,
    }


def compare_to_baseline(rows: List[Dict], baseline: Dict, y: np.ndarray) -> None:
    """Fill the disagreement / McNemar columns against the plain baseline, in place.
    就地填充与 plain 基线的分歧比例与 McNemar 检验结果。"""
    for r in rows:
        if r["config"] == baseline["config"]:
            r["disagree_vs_plain"] = 0.0
            r["mcnemar_p_vs_plain"] = 1.0
            continue
        b = int(((r["_pred"] == y) & (baseline["_pred"] != y)).sum())
        c = int(((r["_pred"] != y) & (baseline["_pred"] == y)).sum())
        r["disagree_vs_plain"] = float((r["_pred"] != baseline["_pred"]).mean())
        r["mcnemar_p_vs_plain"] = mcnemar_exact(b, c)


def fmt_row(r: Dict, baseline_acc: float) -> str:
    """One markdown table row. / 一行 markdown 表格。"""
    ci = r["acc_ci95"]
    dis = "—" if r["disagree_vs_plain"] is None else f"{r['disagree_vs_plain'] * 100:.2f}%"
    p = "—" if r["mcnemar_p_vs_plain"] is None else f"{r['mcnemar_p_vs_plain']:.4f}"
    n_img = r.get("bn_images", 0)
    img = "—" if not n_img else str(n_img)
    return (f"| {r['config']} | {r['weights']} | {r['tta']} | {r['bn_recal']} | {img} | "
            f"{r['acc']:.4f} | [{ci[0]:.4f}, {ci[1]:.4f}] | {r['acc'] - baseline_acc:+.4f} | "
            f"{r['nll']:.4f} | {r['ece']:.4f} | {dis} | {p} |")


# --------------------------------------------------------------------------
# main / 主流程
# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Compare inference configurations on the val split.")
    ap.add_argument("--root", default=REPO_ROOT)
    ap.add_argument("--ckpt", default="auto", help="'auto' = newest checkpoint under runs/ / 自动挑选")
    ap.add_argument("--weights", default="both", choices=["ema", "raw", "both"])
    ap.add_argument("--bn-source", default="train,train_matched,test",
                    help="comma list / 逗号列表: train,train_matched,test,val,none")
    ap.add_argument("--bn-recal", default="none",
                    choices=["none", "train", "train_matched", "test", "val"],
                    help="single BN recalibration source to feature in the report; "
                         "'test' is TRANSDUCTIVE / 单一 BN 重校准来源；'test' 属于转导式")
    ap.add_argument("--bn-samples", type=int, default=10000,
                    help="images used for BN re-estimation / BN 重校准使用的图片数")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-val", type=int, default=2500)
    ap.add_argument("--batch-size", type=int, default=500)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out-md", default=os.path.join(REPO_ROOT, "analysis", "refine.md"))
    ap.add_argument("--out-json", default=os.path.join(REPO_ROOT, "analysis", "refine.json"))
    ap.add_argument("--quick", action="store_true", help="small subset, for smoke testing / 冒烟测试")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    device = torch.device("mps" if (args.device == "auto" and torch.backends.mps.is_available())
                          else ("cuda" if (args.device == "auto" and torch.cuda.is_available()) else "cpu"))

    ckpt = pick_default_ckpt(root) if args.ckpt == "auto" else os.path.abspath(args.ckpt)
    print(f"[refine] checkpoint={ckpt}", flush=True)
    print(f"[refine] device={device}", flush=True)

    # ---- data: identical split to train.py / 与 train.py 完全一致的划分
    # The validation split is PINNED to seed=0 and is deliberately NOT tied to
    # --seed. train.py now hard-codes load_train(seed=0) so that every ensemble
    # member is measured on the SAME 2500 images; if refine.py followed
    # --seed, a `--seed 1` run would silently score a different 2500 images
    # and the numbers would not be comparable with the training curves.
    # 验证集划分固定为 seed=0，且刻意不与 --seed 绑定。train.py 已硬编码
    # load_train(seed=0)，使各集成成员都在同一批 2500 张图上评估；若 refine.py
    # 跟随 --seed，`--seed 1` 就会静默换一批 2500 张图，数字将无法与训练曲线比较。
    split_seed = 0
    tr_x, tr_y, va_x, va_y = load_train(root, seed=split_seed, n_val=args.n_val)
    if args.quick:
        tr_x, tr_y = tr_x[:2000], tr_y[:2000]
        va_x, va_y = va_x[:500], va_y[:500]
    val_ld = make_loader(CifarNumpy(va_x, va_y, train=False), args.batch_size, False, 0)
    print(f"[refine] val={len(va_y)} train={len(tr_y)} "
          f"(split seed={split_seed} PINNED, n_val={args.n_val})", flush=True)

    # ---- pixel-level train/test comparison / 像素级 train/test 对比
    # Uses the FULL train set (not the reduced one) so the class-mix estimate
    # is meaningful. / 使用完整 train 集（而非缩减版），使类别比例估计有意义。
    from data import build_cache as _build_cache  # local import to keep the top clean
    full_tr_x, full_tr_y = _build_cache(root, "train")
    full_te_x, _ = _build_cache(root, "test", need_labels=False)
    pixel_report = pixel_shift_report(full_tr_x, full_tr_y, full_te_x)
    print(f"[refine] pixel shift: train mean={pixel_report['train_mean']:.4f} "
          f"test mean={pixel_report['test_mean']:.4f} "
          f"gap={pixel_report['mean_gap_levels']:+.2f}/255 "
          f"({pixel_report['gap_over_sampling_error']:.1f}x the class-mix sampling error)", flush=True)

    # BN re-estimation loaders (no augmentation) / BN 重校准用（无增强）
    # The "test" source has only N_test images (2000). To tell "domain shift"
    # apart from "simply more data", we also build a size-matched train subset
    # with exactly the same number of images.
    # test 只有 2000 张。为了把「域偏移」和「单纯数据更多」区分开，
    # 额外构造一个张数完全相同的 train 子集作为对照。
    te_x, _ids = load_test(root)
    n_test = len(te_x)
    bn_loaders: Dict[str, object] = {}
    bn_counts: Dict[str, int] = {}
    for src in [s.strip() for s in args.bn_source.split(",") if s.strip()]:
        if src == "none":
            continue
        if src == "train":
            n = min(args.bn_samples, len(tr_x))
            bn_loaders["train"] = make_loader(
                CifarNumpy(tr_x[:n], tr_y[:n], train=False), args.batch_size, False, 0)
            bn_counts["train"] = n
            print(f"[refine] BN source train: {n} images", flush=True)
        elif src == "train_matched":
            # same number of images as the test set -> isolates domain from volume
            # 与测试集等张数 -> 把「域」与「量」分开
            n = min(n_test, len(tr_x))
            bn_loaders["train_matched"] = make_loader(
                CifarNumpy(tr_x[:n], tr_y[:n], train=False), args.batch_size, False, 0)
            bn_counts["train_matched"] = n
            print(f"[refine] BN source train_matched: {n} images (size-matched to test)", flush=True)
        elif src == "val":
            n = min(args.bn_samples, len(va_x))
            bn_loaders["val"] = make_loader(
                CifarNumpy(va_x[:n], va_y[:n], train=False), args.batch_size, False, 0)
            bn_counts["val"] = n
            print(f"[refine] BN source val (leaky, diagnostic only): {n} images", flush=True)
        elif src == "test":
            n = min(args.bn_samples, len(te_x))
            bn_loaders["test"] = make_loader(
                CifarNumpy(te_x[:n], np.zeros(n, dtype=np.int64), train=False), args.batch_size, False, 0)
            bn_counts["test"] = n
            print(f"[refine] BN source test (transductive): {n} images", flush=True)

    weight_modes = ["ema", "raw"] if args.weights == "both" else [args.weights]
    rows: List[Dict] = []

    # ---- explicit transductive warning / 显式转导式警告
    # Re-estimating BatchNorm statistics on the unlabeled test images is a
    # TRANSDUCTIVE procedure: the model sees the test inputs (without labels)
    # before predicting them. It is allowed in a Kaggle competition, but it is
    # NOT a purely supervised result and must be disclosed as such in the report.
    # 在无标签测试图上重估 BN 统计量属于「转导式」过程：模型在预测前就见过测试输入
    # （无标签）。Kaggle 竞赛允许，但它不是纯监督结果，报告必须如实披露。
    if args.bn_recal == "test":
        print("", flush=True)
        print("=" * 78, flush=True)
        print("[refine] WARNING: --bn-recal test is TRANSDUCTIVE.", flush=True)
        print("[refine] BatchNorm statistics are re-estimated on the unlabeled TEST images,", flush=True)
        print("[refine] i.e. the model sees the test inputs before predicting them.", flush=True)
        print("[refine] This is NOT a purely supervised result. If it is used in the final", flush=True)
        print("[refine] submission, the report MUST state it explicitly.", flush=True)
        print("[refine] 警告：--bn-recal test 属于转导式。BN 统计量在无标签测试图上重估，", flush=True)
        print("[refine] 即模型在预测前已见过测试输入。这不是纯监督结果；若用于最终提交，", flush=True)
        print("[refine] 报告必须明确写出这一点。", flush=True)
        print("=" * 78, flush=True)
        print("", flush=True)
    elif args.bn_recal != "none":
        print(f"[refine] BN recalibration source: {args.bn_recal} "
              f"(supervised: statistics come from labelled train images)", flush=True)

    # Take ONE snapshot of the checkpoint and use it for every weight variant.
    # 只取一次 checkpoint 快照，所有权重变体共用它——否则训练进程在两次读取之间
    # 覆写 last.pt，会把不同 epoch 的结果混进同一张表。
    ck_state = load_ckpt_dict(ckpt)
    print(f"[refine] snapshot epoch={ck_state.get('epoch')} "
          f"model={ck_state.get('args', {}).get('model')}", flush=True)

    for w in weight_modes:
        model, meta = load_model_from_state(ck_state, device, w, ckpt)
        print(f"[refine] loaded weights={w} model={meta['model']} params={meta['params']:,} "
              f"epoch={meta['epoch']}", flush=True)

        # ---- Phase 1: TTA modes, no BN recalibration
        for mode in TTA_MODES:
            t0 = time.time()
            p = probs_for(model, val_ld, device, mode)
            name = f"{w}.{mode}"
            r = describe(name, w, mode, "none", p, va_y)
            r["seconds"] = round(time.time() - t0, 1)
            r["bn_images"] = 0
            rows.append(r)
            print(f"[refine] {name:24s} acc={r['acc']:.4f} nll={r['nll']:.4f} ece={r['ece']:.4f} "
                  f"({r['seconds']}s)", flush=True)

        # ---- Phase 2: BN recalibration (applied on top of the same weights)
        for src, loader in bn_loaders.items():
            used = bn_recalibrate(model, loader, device)
            print(f"[refine] BN recalibrated from {src} ({used} batches)", flush=True)
            for mode in ("plain", "translate_flip"):
                t0 = time.time()
                p = probs_for(model, val_ld, device, mode)
                name = f"{w}.{mode}+bn[{src}]"
                r = describe(name, w, mode, src, p, va_y)
                r["seconds"] = round(time.time() - t0, 1)
                r["bn_images"] = bn_counts.get(src, 0)
                rows.append(r)
                print(f"[refine] {name:24s} acc={r['acc']:.4f} nll={r['nll']:.4f} "
                      f"ece={r['ece']:.4f} ({r['seconds']}s)", flush=True)
            # restore the original statistics for the next weight source
            model, meta = load_model_from_state(ck_state, device, w, ckpt)

    # ---- baseline for the comparison columns / 对比列的基线
    # The baseline is the configuration a normal submission would use:
    # plain (no TTA, no BN recalibration), best of the two weight modes.
    # 基线 = 常规提交所用配置：plain（无 TTA、无 BN 重校准），在两种权重里取更优者。
    plain_rows = [r for r in rows if r["bn_recal"] == "none" and r["tta"] == "plain"]
    baseline = max(plain_rows, key=lambda r: r["acc"]) if plain_rows else rows[0]
    compare_to_baseline(rows, baseline, va_y)
    base_acc = baseline["acc"]
    print(f"[refine] baseline for comparison: {baseline['config']} acc={base_acc:.4f}", flush=True)

    # ---- best configuration / 最优配置
    scored = sorted(rows, key=lambda r: (r["acc"], -r["nll"]), reverse=True)
    best = scored[0]

    # ---- temperature scaling diagnostic / 温度标定诊断（拟合与评估分开）
    temp_info = None
    if best is not None and len(va_y) >= 200:
        half = len(va_y) // 2
        idx_a, idx_b = np.arange(half), np.arange(half, len(va_y))
        t = fit_temperature(best["_probs"][idx_a], va_y[idx_a])
        logits = np.log(np.clip(best["_probs"][idx_b], 1e-12, 1.0)) / t
        p_b = np.exp(logits - logits.max(1, keepdims=True))
        p_b /= p_b.sum(1, keepdims=True)
        temp_info = {
            "fitted_on": "val first half",
            "evaluated_on": "val second half",
            "temperature": t,
            "acc_before": float((best["_probs"][idx_b].argmax(1) == va_y[idx_b]).mean()),
            "acc_after": float((p_b.argmax(1) == va_y[idx_b]).mean()),
            "nll_before": nll(best["_probs"][idx_b], va_y[idx_b]),
            "nll_after": nll(p_b, va_y[idx_b]),
        }

    # ---- write outputs / 输出
    os.makedirs(os.path.dirname(os.path.abspath(args.out_md)), exist_ok=True)
    header = [
        "# Inference-configuration comparison (`src/refine.py`)",
        "# 推理配置对比（src/refine.py）",
        "",
        f"- Checkpoint / 检查点：`{os.path.relpath(ckpt, root)}`",
        f"- Weights model / 结构：`{meta['model']}`，params={meta['params']:,}",
        f"- Checkpoint epoch / 检查点 epoch：{meta['epoch']}",
        f"- Validation split / 验证集：seed={split_seed} (PINNED, identical to `train.py`), "
        f"n_val={args.n_val} / 验证集划分固定为 seed=0，与 train.py 一致",
        f"- Generated / 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S %Z')}",
        f"- Device / 设备：{device}",
        f"- `--bn-recal` / 报告重点关注的 BN 重校准来源：`{args.bn_recal}`"
        + ("  ⚠️ **TRANSDUCTIVE — 见下方警告**" if args.bn_recal == "test" else ""),
        "",
    ]
    table_head = [
        "| Config | Weights | TTA | BN recal | BN imgs | Acc | 95% CI | Δacc vs baseline | NLL | ECE | Disagree vs baseline | McNemar p |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    body = [fmt_row(r, base_acc) for r in rows]
    tail = []

    # ---- transductive banner / 转导式警示横幅
    if args.bn_recal == "test":
        tail += [
            "> [!WARNING]",
            "> **The `bn[test]` rows are TRANSDUCTIVE.** BatchNorm statistics are re-estimated "
            "on the unlabeled test images, so the model sees the test inputs before predicting "
            "them. This is allowed in a Kaggle competition but it is **not** a purely supervised "
            "result. Any use of these numbers in the report or in the final submission must be "
            "disclosed explicitly, and the claimed accuracy must not be presented as a clean "
            "held-out evaluation.",
            ">",
            "> **`bn[test]` 各行属于转导式结果。** BN 统计量在无标签测试图上重估，即模型在预测前"
            "已见过测试输入。Kaggle 竞赛允许，但这**不是**纯监督结果。报告中引用这些数字、"
            "或把它用于最终提交，都必须明确披露，不得当作干净的留出集评估。",
            "",
        ]

    tail += ["## Best configuration / 最优配置", "",
            f"**{best['config']}** — weights={best['weights']}, TTA={best['tta']}, "
            f"BN recal={best['bn_recal']}, acc={best['acc']:.4f} "
            f"(95% CI [{best['acc_ci95'][0]:.4f}, {best['acc_ci95'][1]:.4f}]), "
            f"NLL={best['nll']:.4f}, ECE={best['ece']:.4f}",
            "",
            f"Baseline / 基线：`{baseline['config']}` acc={base_acc:.4f}. "
            f"Δacc = {best['acc'] - base_acc:+.4f}.",
            ""]

    # ---- domain-shift control: is the BN gain from the domain or from volume?
    # ---- 域偏移对照：BN 的提升来自「域」还是来自「样本量」？
    dom_lines: List[str] = []
    plain_by_w = {r["weights"]: r for r in rows
                  if r["tta"] == "plain" and r["bn_recal"] == "none"}
    for w, base_row in plain_by_w.items():
        m = next((r for r in rows if r["weights"] == w and r["bn_recal"] == "train_matched"
                  and r["tta"] == "plain"), None)
        t = next((r for r in rows if r["weights"] == w and r["bn_recal"] == "test"
                  and r["tta"] == "plain"), None)
        if not (m and t):
            continue
        d_matched = m["acc"] - base_row["acc"]
        d_test = t["acc"] - base_row["acc"]
        dom_lines += [
            f"- weights={w}: baseline {base_row['acc']:.4f} -> "
            f"train_matched ({m['bn_images']} imgs) {m['acc']:.4f} ({d_matched:+.4f}) -> "
            f"test ({t['bn_images']} imgs) {t['acc']:.4f} ({d_test:+.4f})",
        ]
    if dom_lines:
        tail += ["## Domain-shift control / 域偏移对照", "",
                 "Same image count for train and test sources, so any extra gain from the test "
                 "images is attributable to the domain, not to the sample volume.",
                 "train 与 test 来源的图像张数相同，因此 test 上多出来的收益只能归因于「域」，而非样本量。",
                 ""] + dom_lines + [""]

    # ---- pixel-level train/test comparison / 像素级 train/test 对比
    if pixel_report is not None:
        pr = pixel_report
        tail += [
            "## Pixel-level train/test comparison / 像素级 train/test 对比",
            "",
            f"- Mean pixel value: train {pr['train_mean']:.4f} vs test {pr['test_mean']:.4f} "
            f"(gap {pr['mean_gap']:+.4f} = {pr['mean_gap_levels']:+.2f} / 255 levels)",
            f"  像素均值：train {pr['train_mean']:.4f} vs test {pr['test_mean']:.4f} "
            f"（差 {pr['mean_gap']:+.4f}，即 {pr['mean_gap_levels']:+.2f} / 255 级）",
            f"- Std pixel value: train {pr['train_std']:.4f} vs test {pr['test_std']:.4f} "
            f"(gap {pr['std_gap']:+.4f})",
            f"  像素标准差：train {pr['train_std']:.4f} vs test {pr['test_std']:.4f} "
            f"（差 {pr['std_gap']:+.4f}）",
            f"- Class-mix sampling error on the mean: {pr['class_mix_sampling_error']:.4f}; "
            f"the observed gap is {pr['gap_over_sampling_error']:.1f}x that error.",
            f"  按类别比例抽样造成的均值误差约 {pr['class_mix_sampling_error']:.4f}；"
            f"实测差值是该误差的 {pr['gap_over_sampling_error']:.1f} 倍。",
            "",
        ]
    if temp_info:
        tail += [
            "## Temperature-scaling diagnostic / 温度标定诊断",
            "",
            f"- Fitted on / 拟合集：{temp_info['fitted_on']}，evaluated on / 评估集：{temp_info['evaluated_on']}",
            f"- T = {temp_info['temperature']:.4f}",
            f"- Acc {temp_info['acc_before']:.4f} -> {temp_info['acc_after']:.4f}；"
            f"NLL {temp_info['nll_before']:.4f} -> {temp_info['nll_after']:.4f}",
            "",
        ]
    tail += [
        "## Notes / 说明",
        "",
        f"- The McNemar column is an exact two-sided test on the discordant pairs against the "
        f"baseline `{baseline['config']}`; p > 0.05 means the difference is not distinguishable "
        "from noise on this validation set.",
        f"  McNemar 列为与基线 {baseline['config']} 的精确二项检验；p > 0.05 表示该差异在验证集上无法与噪声区分。",
        "- `bn[test]` is transductive: BN statistics are re-estimated on the unlabeled test images. "
        "This is legal in a Kaggle competition but must be disclosed in the report.",
        "  bn[test] 属于转导式适配：在无标签测试图上重估 BN 统计量。Kaggle 竞赛允许，但报告必须披露。",
        "- `bn[val]` is leaky (it uses the validation labels' images) and is only a diagnostic ceiling.",
        "  bn[val] 使用了验证集，仅作诊断上限，不可作为最终方案。",
        "- `bn[train_matched]` uses exactly as many images as the test set, so the comparison "
        "against `bn[test]` isolates the domain effect from the sample-volume effect.",
        "  bn[train_matched] 与测试集张数完全相同，因此与 bn[test] 的对比能把「域效应」与「样本量效应」分开。",
        "- All rows come from ONE snapshot of the checkpoint, so every weight variant is measured "
        "at the same training epoch. / 所有行都来自同一次 checkpoint 快照，因此各权重变体处于同一训练 epoch。",
        "- No number here is copied from anywhere else; all are measured by this script.",
        "  此处没有抄来的数字，全部由本脚本实测。",
        "",
    ]
    with open(args.out_md, "w") as fh:
        fh.write("\n".join(header + table_head + body + tail))

    payload = {
        "checkpoint": os.path.relpath(ckpt, root),
        "checkpoint_epoch": meta["epoch"],
        "model": meta["model"],
        "params": meta["params"],
        "val": {"split_seed": split_seed, "n_val": args.n_val, "actual": int(len(va_y)),
                "note": "split seed is pinned to 0, independent of --seed"},
        "device": str(device),
        "generated": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "rows": [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows],
        "best": {k: v for k, v in best.items() if not k.startswith("_")},
        "pixel_shift": pixel_report,
        "temperature": temp_info,
    }
    with open(args.out_json, "w") as fh:
        json.dump(payload, fh, indent=1)
    np.save(os.path.splitext(args.out_json)[0] + "_probs_best.npy", best["_probs"].astype(np.float32))

    print(f"[refine] best={best['config']} acc={best['acc']:.4f}", flush=True)
    print(f"[refine] wrote {args.out_md} and {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
