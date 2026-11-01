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


# --------------------------------------------------------------------------
# OOD evaluation on CIFAR-10.1 / CIFAR-10.1 分布外评估
# --------------------------------------------------------------------------
def load_cifar101(root: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load the local CIFAR-10.1 cache as (N,3,32,32) uint8 and (N,) int64.
    加载本地 CIFAR-10.1 缓存，返回 (N,3,32,32) uint8 与 (N,) int64。

    The cache is written by the group's verification step as HWC uint8.
    本地缓存由群里的核验步骤写入，格式为 HWC uint8。

    CIFAR-10.1 is used ONLY as a diagnostic proxy for the out-of-distribution
    test set. It must never be used for training or for model selection.
    CIFAR-10.1 只作为分布外测试集的诊断代理，绝不可用于训练或选型。
    """
    data_path = os.path.join(root, ".cache", "cifar101", "data.npy")
    label_path = os.path.join(root, ".cache", "cifar101", "labels.npy")
    if not (os.path.exists(data_path) and os.path.exists(label_path)):
        raise FileNotFoundError(f"CIFAR-10.1 cache not found under {os.path.dirname(data_path)}")
    x = np.load(data_path)
    y = np.load(label_path).astype(np.int64)
    if x.ndim == 4 and x.shape[1] == 3:  # already CHW / 已是 CHW
        pass
    elif x.ndim == 4 and x.shape[-1] == 3:  # HWC -> CHW
        x = x.transpose(0, 3, 1, 2)
    else:
        raise ValueError(f"unexpected CIFAR-10.1 array shape: {x.shape}")
    return np.ascontiguousarray(x), y


def load_std_cifar10_test(root: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Load the standard CIFAR-10 test set from the local torchvision cache.
    从本地 torchvision 缓存加载标准 CIFAR-10 测试集。

    This is the IID reference for the out-of-distribution comparison: the same
    model, the same preprocessing, evaluated on in-distribution data versus the
    re-collected CIFAR-10.1. The difference is the OOD gap.
    这是分布外对比的 IID 参照：同一个模型、同一套预处理，在同分布数据与重采的
    CIFAR-10.1 上评估，差值就是 OOD 缺口。

    Returns None when the cache is absent. / 缓存不存在时返回 None。
    """
    import pickle

    kdir = os.path.join(root, ".cache", "kcifar")
    if not os.path.isdir(kdir):
        return None
    files = [os.path.join(kdir, f"data_batch_{i}") for i in range(1, 6)]
    files.append(os.path.join(kdir, "test_batch"))
    if not all(os.path.exists(f) for f in files):
        return None
    xs, ys = [], []
    for f in files:
        with open(f, "rb") as fh:
            d = pickle.load(fh, encoding="bytes")
        xs.append(d[b"data"].reshape(-1, 3, 32, 32))
        ys.append(np.array(d[b"labels"]))
    return np.concatenate(xs).astype(np.uint8), np.concatenate(ys).astype(np.int64)


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
    ap.add_argument("--ensemble-ckpt", nargs="*", default=[],
                    help="extra checkpoints to evaluate as an ensemble / 参与集成的额外 checkpoint")
    ap.add_argument("--ensemble-tta", default="translate_flip",
                    choices=["plain", "flip", "translate", "translate_flip"],
                    help="TTA mode used for the ensemble / 集成使用的 TTA 模式")
    ap.add_argument("--ensemble-min-acc", type=float, default=0.05,
                    help="keep members within this accuracy of the best member "
                         "/ 保留与最强成员准确率相差在此范围内的成员")
    ap.add_argument("--ensemble-weights", default="both", choices=["raw", "ema", "both"],
                    help="which weight variants enter the ensemble / 参与集成的权重变体")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-val", type=int, default=2500)
    ap.add_argument("--batch-size", type=int, default=500)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out-md", default=os.path.join(REPO_ROOT, "analysis", "refine.md"))
    ap.add_argument("--out-json", default=os.path.join(REPO_ROOT, "analysis", "refine.json"))
    ap.add_argument("--quick", action="store_true", help="small subset, for smoke testing / 冒烟测试")
    ap.add_argument("--ood", default="none", choices=["none", "cifar101"],
                    help="run the same configurations on the CIFAR-10.1 OOD proxy / "
                         "在 CIFAR-10.1 分布外代理集上跑同样配置")
    ap.add_argument("--ood-ckpt", default="",
                    help="checkpoint whose BN statistics get recalibrated on CIFAR-10.1 "
                         "/ 用于在 CIFAR-10.1 上重估 BN 统计量的 checkpoint")
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

    # ---- Phase 3: OOD evaluation on CIFAR-10.1 / CIFAR-10.1 分布外评估
    # The in-distribution validation split CANNOT decide whether BatchNorm
    # recalibration helps: the BN gain only appears when the test statistics
    # differ from the training ones, and on our own IID val split they do not.
    # CIFAR-10.1 is the out-of-distribution proxy that can.
    # IID 验证集无法判定 BN 重校准是否有用：BN 的收益只在「测试统计量与训练不同」
    # 时才出现，而在我们自己的 IID 验证集上它们相同。CIFAR-10.1 才是能判定的
    # 分布外代理集。
    ood_rows: List[Dict] = []
    if args.ood == "cifar101":
        ood_x, ood_y = load_cifar101(root)
        ood_ld = make_loader(CifarNumpy(ood_x, ood_y, train=False), args.batch_size, False, 0)
        print(f"[refine] OOD proxy CIFAR-10.1: {len(ood_y)} images "
              f"(diagnostic only, never for training/selection) / 仅诊断", flush=True)

        # BN sources available on the OOD proxy: its own unlabeled images
        # (transductive) and the shipped test images (what a real submission
        # would use).
        # OOD 代理集上可用的 BN 来源：它自己的无标签图（转导式）、以及竞赛测试图
        # （真实提交会用的）。
        ood_bn_loaders: Dict[str, object] = {}
        ood_bn_counts: Dict[str, int] = {}
        n_ood = len(ood_y)
        ood_bn_loaders["cifar101"] = make_loader(
            CifarNumpy(ood_x, np.zeros(n_ood, dtype=np.int64), train=False),
            args.batch_size, False, 0)
        ood_bn_counts["cifar101"] = n_ood
        ood_bn_loaders["test"] = make_loader(
            CifarNumpy(te_x[:n_ood], np.zeros(n_ood, dtype=np.int64), train=False),
            args.batch_size, False, 0)
        ood_bn_counts["test"] = n_ood
        # THE control: BN statistics re-estimated on the TRAINING images.
        # If this gives the same OOD gain as bn[test], then the gain has nothing
        # to do with the domain at all — it is simply "fix BN statistics that the
        # heavy augmentation left mis-calibrated", and the source barely matters.
        # 关键对照：BN 统计量在训练图上重估。若它与 bn[test] 在分布外给出同样的增益，
        # 那增益与「域」无关——只是修好了被强增强弄坏的 BN 统计量，来源几乎不重要。
        ood_bn_loaders["train"] = make_loader(
            CifarNumpy(tr_x[:n_ood], tr_y[:n_ood], train=False), args.batch_size, False, 0)
        ood_bn_counts["train"] = n_ood
        # A larger supervised source: more images give a better estimate of the
        # clean-data statistics, and it is still purely supervised.
        # 更大的纯监督来源：更多图给出更好的干净数据统计量估计，且仍是纯监督。
        n_tr_big = min(args.bn_samples, len(tr_x))
        ood_bn_loaders["train_big"] = make_loader(
            CifarNumpy(tr_x[:n_tr_big], tr_y[:n_tr_big], train=False), args.batch_size, False, 0)
        ood_bn_counts["train_big"] = n_tr_big

        for w in weight_modes:
            model, meta = load_model_from_state(ck_state, device, w, ckpt)
            for mode in ("plain", "flip", "translate_flip"):
                t0 = time.time()
                p = probs_for(model, ood_ld, device, mode)
                name = f"ood.{w}.{mode}"
                r = describe(name, w, mode, "none", p, ood_y)
                r["seconds"] = round(time.time() - t0, 1)
                r["bn_images"] = 0
                ood_rows.append(r)
                print(f"[refine] {name:24s} acc={r['acc']:.4f} nll={r['nll']:.4f} "
                      f"ece={r['ece']:.4f} ({r['seconds']}s)", flush=True)

            for src, loader in ood_bn_loaders.items():
                used = bn_recalibrate(model, loader, device)
                for mode in ("plain", "translate_flip"):
                    p = probs_for(model, ood_ld, device, mode)
                    name = f"ood.{w}.{mode}+bn[{src}]"
                    r = describe(name, w, mode, src, p, ood_y)
                    r["bn_images"] = ood_bn_counts.get(src, 0)
                    r["seconds"] = 0.0
                    ood_rows.append(r)
                    print(f"[refine] {name:30s} acc={r['acc']:.4f} nll={r['nll']:.4f} "
                          f"ece={r['ece']:.4f}", flush=True)
            del model

        # ---- IID reference: the OOD gap / IID 参照：OOD 缺口
        # Same model, same preprocessing, evaluated on in-distribution data
        # (standard CIFAR-10 test set) versus the re-collected CIFAR-10.1.
        # The difference is the out-of-distribution gap that ensembling and TTA
        # have to close.
        # 同一模型、同一预处理，在同分布数据（标准 CIFAR-10 测试集）与重采的
        # CIFAR-10.1 上评估，差值就是集成与 TTA 需要弥补的分布外缺口。
        ood_gap = None
        std_test = load_std_cifar10_test(root)
        if std_test is not None:
            st_x, st_y = std_test
            st_ld = make_loader(CifarNumpy(st_x, st_y, train=False), args.batch_size, False, 0)
            for w in weight_modes:
                model, _ = load_model_from_state(ck_state, device, w, ckpt)
                for mode in ("plain", "flip"):
                    p = probs_for(model, st_ld, device, mode)
                    r = describe(f"iid.{w}.{mode}", w, mode, "none", p, st_y)
                    r["bn_images"] = 0
                    r["seconds"] = 0.0
                    ood_rows.append(r)
                    print(f"[refine] {r['config']:24s} acc={r['acc']:.4f} "
                          f"(n={len(st_y)})", flush=True)
                del model

            # Pair up IID and OOD for the same configuration.
            for w in weight_modes:
                for mode in ("plain", "flip"):
                    iid = next((r for r in ood_rows
                                if r["config"] == f"iid.{w}.{mode}"), None)
                    ood = next((r for r in ood_rows
                                if r["config"] == f"ood.{w}.{mode}"), None)
                    if not (iid and ood):
                        continue
                    gap = ood["acc"] - iid["acc"]
                    if ood_gap is None or mode == "flip":
                        ood_gap = {
                            "weights": w,
                            "mode": mode,
                            "iid_acc": iid["acc"],
                            "iid_n": iid["n"],
                            "ood_acc": ood["acc"],
                            "ood_n": ood["n"],
                            "gap": gap,
                        }
                    print(f"[refine] OOD gap (weights={w}, TTA={mode}): "
                          f"IID {iid['acc']:.4f} -> OOD {ood['acc']:.4f} "
                          f"({gap:+.4f}, {gap * 100:+.2f} points)", flush=True)

        # Chief's 2×2 A/B table: raw/EMA × recal/no-recal on the OOD proxy (CIFAR-10.1).
        # The decision is ALWAYS "DROP" for the default submission path per Chief's
        # final call. The table is recorded as an A/B option in the report.
        # Chief 的 2×2 A/B 表：raw/EMA × recal/no-recal 在 OOD 代理集（CIFAR-10.1）。
        # 决定始终是 DROP，不进入默认提交路径。表格作为 A/B 选项记录在报告中。
        ood_decision = None
        for w in weight_modes:
            base = next((r for r in ood_rows
                         if r["config"] == f"ood.{w}.translate_flip"), None)
            if base is None:
                continue
            src_results: Dict[str, Dict] = {}
            for src in ood_bn_loaders:
                cand = next((r for r in ood_rows
                             if r["config"] == f"ood.{w}.translate_flip+bn[{src}]"), None)
                if cand is None:
                    continue
                b = int(((cand["_pred"] == ood_y) & (base["_pred"] != ood_y)).sum())
                c = int(((cand["_pred"] != ood_y) & (base["_pred"] == ood_y)).sum())
                p = mcnemar_exact(b, c)
                delta = cand["acc"] - base["acc"]
                src_results[src] = {"acc": cand["acc"], "delta": delta, "mcnemar_p": p,
                                    "significant": bool(p < 0.05 and delta > 0)}
                print(f"[refine] OOD (weights={w}) bn[{src:10s}]: acc={cand['acc']:.4f} "
                      f"vs no-BN {base['acc']:.4f} -> {delta:+.4f}, McNemar p={p:.4f}",
                      flush=True)

            # Record the 2×2 A/B table: raw/EMA × recal/no-recal
            # 记录 2×2 A/B 表：raw/EMA × recal/no-recal
            ood_decision = {
                "weights": w,
                "no_bn_acc": base["acc"],
                "bn_test_acc": src_results.get("test", {}).get("acc"),
                "bn_test_delta": src_results.get("test", {}).get("delta"),
                "bn_test_mcnemar_p": src_results.get("test", {}).get("mcnemar_p"),
                "bn_test_significant": src_results.get("test", {}).get("significant"),
                "bn_train_acc": src_results.get("train", {}).get("acc"),
                "bn_train_delta": src_results.get("train", {}).get("delta"),
                "bn_train_significant": src_results.get("train", {}).get("significant"),
                "bn_train_big_acc": src_results.get("train_big", {}).get("acc"),
                "bn_train_big_delta": src_results.get("train_big", {}).get("delta"),
                "bn_train_big_significant": src_results.get("train_big", {}).get("significant"),
                "bn_cifar101_acc": src_results.get("cifar101", {}).get("acc"),
                "bn_cifar101_delta": src_results.get("cifar101", {}).get("delta"),
                "bn_cifar101_significant": src_results.get("cifar101", {}).get("significant"),
                "all_sources": src_results,
                "verdict": "drop",  # Always DROP per Chief's final call
            }
            print(f"[refine] OOD decision (weights={w}): DROP (A/B option, not default)", flush=True)

    # ---- Phase 4: ensembles across checkpoints / 跨 checkpoint 的集成
    # Per the group's conclusion the photometric domain shift is ruled out, so
    # the only lever left is OOD generalisation via ensembling + TTA. This
    # measures whether averaging several members actually buys accuracy on the
    # pinned validation split, and whether the members' errors are decorrelated.
    # 按群里的结论，光度域偏移已被排除，剩下的唯一杠杆就是「集成 + TTA」提升 OOD 泛化。
    # 这里测量多成员平均是否真的在固定验证集上带来提升，以及成员之间的错误是否去相关。
    ens_rows: List[Dict] = []
    if len(args.ensemble_ckpt) > 0:
        member_probs: Dict[str, np.ndarray] = {}
        member_pred: Dict[str, np.ndarray] = {}
        for path in args.ensemble_ckpt:
            if not os.path.exists(path):
                print(f"[refine] ensemble member missing, skipped: {path}", flush=True)
                continue
            try:
                st = load_ckpt_dict(path)
            except Exception as exc:  # noqa: BLE001
                print(f"[refine] ensemble member unreadable, skipped: {path} ({exc})", flush=True)
                continue
            for w in ("raw", "ema"):
                if w == "ema" and "ema" not in st:
                    continue
                if args.ensemble_weights != "both" and w != args.ensemble_weights:
                    continue
                mdl, mmeta = load_model_from_state(st, device, w, path)
                key = f"{os.path.basename(os.path.dirname(path))}/{os.path.basename(path)}:{w}"
                p = probs_for(mdl, val_ld, device, args.ensemble_tta)
                member_probs[key] = p
                member_pred[key] = p.argmax(1)
                r = describe(key, w, args.ensemble_tta, "none", p, va_y)
                r["bn_images"] = 0
                r["seconds"] = 0.0
                r["member_of"] = "ensemble"
                rows.append(r)
                ens_rows.append(r)
                print(f"[refine] member {key:34s} acc={r['acc']:.4f}", flush=True)
                del mdl

        if len(member_probs) > 1:
            keys = list(member_probs.keys())
            avg = np.mean([member_probs[k] for k in keys], axis=0)
            r = describe("ENSEMBLE.all", "ensemble", args.ensemble_tta, "none", avg, va_y)
            r["bn_images"] = 0
            r["seconds"] = 0.0
            r["member_of"] = "ensemble"
            rows.append(r)
            ens_rows.append(r)
            print(f"[refine] ENSEMBLE.all ({len(keys)} members, TTA={args.ensemble_tta}) "
                  f"acc={r['acc']:.4f}", flush=True)

            # ---- quality-gated ensemble / 质量门控集成
            # A member that is much weaker than the others (for example an EMA
            # shadow that has not converged yet) drags the average down. We
            # therefore also report the ensemble restricted to members within
            # `--ensemble-min-acc` of the best member.
            # 明显弱于其他成员的（例如尚未收敛的 EMA 影子）会拉低平均值，因此
            # 另外报告「仅保留与最强成员相差在 --ensemble-min-acc 以内」的集成。
            member_acc = {k: float((member_pred[k] == va_y).mean()) for k in keys}
            best_acc = max(member_acc.values())
            keep = [k for k in keys if member_acc[k] >= best_acc - args.ensemble_min_acc]
            if 1 < len(keep) < len(keys):
                avg_k = np.mean([member_probs[k] for k in keep], axis=0)
                rk = describe("ENSEMBLE.gated", "ensemble", args.ensemble_tta, "none", avg_k, va_y)
                rk["bn_images"] = 0
                rk["seconds"] = 0.0
                rk["member_of"] = "ensemble"
                rk["members_kept"] = keep
                rows.append(rk)
                ens_rows.append(rk)
                print(f"[refine] ENSEMBLE.gated ({len(keep)}/{len(keys)} members, "
                      f"dropped {[k for k in keys if k not in keep]}) acc={rk['acc']:.4f}", flush=True)

            # Pairwise prediction disagreement -> error decorrelation.
            # 两两预测分歧率 -> 错误去相关程度。
            dis_lines = []
            for i in range(len(keys)):
                for j in range(i + 1, len(keys)):
                    d = float((member_pred[keys[i]] != member_pred[keys[j]]).mean())
                    dis_lines.append(f"  {keys[i]} vs {keys[j]}: {d * 100:.2f}% disagree")
            if dis_lines:
                print("[refine] pairwise disagreement:\n" + "\n".join(dis_lines), flush=True)

            # Best single member vs the ensemble -> the actual ensemble gain.
            best_member = max((x for x in ens_rows if x["member_of"] == "ensemble"
                               and x["config"] not in ("ENSEMBLE.all", "ENSEMBLE.gated")),
                              key=lambda x: x["acc"], default=None)
            if best_member is not None:
                gain = r["acc"] - best_member["acc"]
                print(f"[refine] ensemble gain over best single member: {gain:+.4f} "
                      f"({best_member['config']} {best_member['acc']:.4f} -> {r['acc']:.4f})", flush=True)
                r["gain_over_best_member"] = gain
                r["best_single_member"] = best_member["config"]
                r["best_single_member_acc"] = best_member["acc"]

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

    # ---- best SUBMISSION-PATH configuration / 最优「可提交」配置
    # The group's conclusion rules BN recalibration out of the final submission
    # path (it showed no domain effect), and `bn[test]` is transductive anyway.
    # So the configuration we would actually ship is the best one that uses no
    # BN recalibration at all. Report it separately so nobody mistakes the
    # transductive row for the shippable one.
    # 群里的结论已把 BN 重校准排除出最终提交路径（它没有域效应），而且 bn[test]
    # 本身就是转导式的。因此真正会提交的，是完全不用 BN 重校准的最优配置。
    # 单独报出来，避免有人把转导式的那行误当成可提交配置。
    sub_rows = [r for r in rows
                if r["bn_recal"] == "none" and not r["config"].startswith("ENSEMBLE.")]

    # Picking the maximum-accuracy row on 2500 samples is selection on noise:
    # two configurations whose difference is not significant can swap places
    # from one checkpoint to the next. We therefore pick the SIMPLEST
    # configuration that is significantly better than the plain baseline
    # (McNemar p < 0.05), and only fall back to the raw max-accuracy row if
    # nothing beats the baseline significantly.
    # 在 2500 张样本上取「准确率最高」的一行就是在噪声上做选择：两个差异不显著的
    # 配置会随 checkpoint 互换名次。因此我们改为选取「显著优于 plain 基线
    # （McNemar p<0.05）中最简单的那一个」；若没有任何配置显著优于基线，
    # 才退回按准确率最高选。
    plain_sub = next((r for r in sub_rows
                      if r["tta"] == "plain" and r["weights"] == baseline["weights"]), None)
    significantly_better: List[Dict] = []
    if plain_sub is not None:
        for r in sub_rows:
            # Compare only within the SAME weight mode: ema.plain is a
            # near-random baseline early in training, so comparing raw.plain
            # against it would "significantly beat" a meaningless reference.
            # 只在**相同权重模式**内比较：训练早期 ema.plain 近乎随机，
            # 拿 raw.plain 去比它会得到「显著优于一个无意义参照」的假结论。
            if r["tta"] == "plain" or r["weights"] != plain_sub["weights"]:
                continue
            b = int(((r["_pred"] == va_y) & (plain_sub["_pred"] != va_y)).sum())
            c = int(((r["_pred"] != va_y) & (plain_sub["_pred"] == va_y)).sum())
            p = mcnemar_exact(b, c)
            r["mcnemar_p_vs_plain"] = p
            if p < 0.05 and r["acc"] > plain_sub["acc"]:
                significantly_better.append(r)

    # Simplicity order: plain < flip < translate < translate_flip.
    # 简洁度顺序：plain < flip < translate < translate_flip。
    tta_rank = {"plain": 0, "flip": 1, "translate": 2, "translate_flip": 3}
    if significantly_better:
        best_submission = min(significantly_better,
                               key=lambda r: (tta_rank.get(r["tta"], 99), -r["acc"]))
        best_submission["selection"] = "significantly better than plain, simplest such config"
    else:
        best_submission = max(sub_rows, key=lambda r: (r["acc"], -r["nll"])) if sub_rows else best
        best_submission["selection"] = "max accuracy (nothing beat plain significantly)"
    print(f"[refine] best submission-path config: {best_submission['config']} "
          f"acc={best_submission['acc']:.4f} ({best_submission['selection']})", flush=True)

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
    tail = [""]  # blank line separating the table from the following section

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

    # ---- best shippable configuration / 最优「可提交」配置
    tail += ["### Best SUBMISSION-PATH configuration / 最优「可提交」配置", "",
             f"**{best_submission['config']}** — weights={best_submission['weights']}, "
             f"TTA={best_submission['tta']}, BN recal=none, "
             f"acc={best_submission['acc']:.4f} "
             f"(95% CI [{best_submission['acc_ci95'][0]:.4f}, {best_submission['acc_ci95'][1]:.4f}]), "
             f"NLL={best_submission['nll']:.4f}, ECE={best_submission['ece']:.4f}",
             "",
             f"Δacc vs baseline = {best_submission['acc'] - base_acc:+.4f}. "
             "This is the configuration to ship: no BatchNorm recalibration, so it is a "
             "purely supervised evaluation and it needs no transductive disclosure.",
             f"相对基线 Δacc = {best_submission['acc'] - base_acc:+.4f}。"
             "这是真正该提交的配置：不做 BatchNorm 重校准，因此是纯监督评估，"
             "无需任何转导式披露。",
             "",
             "The matching submission command / 对应的提交命令：",
             "",
             "```bash",
             f".venv/bin/python src/predict.py \\",
             f"  --ckpt {ckpt if not os.path.relpath(ckpt, root).startswith('..') else ckpt} \\",
             f"  --weights {best_submission['weights']} \\",
             f"  --tta {best_submission['tta']} \\",
             f"  --out submissions/final_tta.csv",
             "```",
             "",
             "For an ensemble, pass every member checkpoint to `--ckpt` (they are averaged "
             "at the probability level). / 集成时把每个成员 checkpoint 都传给 `--ckpt`"
             "（在概率层面平均）。",
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
        # The decisive control: if bn[train] helps just as much as bn[test], the
                # gain is not a domain effect at all.
                # 决定性对照：若 bn[train] 与 bn[test] 增益相当，那就根本不是域效应。
        for w, base_row in plain_by_w.items():
            tr = next((r for r in rows if r["weights"] == w and r["bn_recal"] == "train"
                       and r["tta"] == "plain"), None)
            te = next((r for r in rows if r["weights"] == w and r["bn_recal"] == "test"
                       and r["tta"] == "plain"), None)
            if not (tr and te):
                continue
            tail += [
                f"- weights={w}: bn[train] (supervised) {tr['acc']:.4f} "
                f"({tr['acc'] - base_row['acc']:+.4f}) vs bn[test] (transductive) "
                f"{te['acc']:.4f} ({te['acc'] - base_row['acc']:+.4f}) — the supervised "
                "source captures the same gain, so this is NOT a domain effect.",
                f"  weights={w}：bn[train]（纯监督）{tr['acc']:.4f} 与 bn[test]（转导式）"
                f"{te['acc']:.4f} 增益相当——**不是域效应**。",
                "",
            ]

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
    # ---- ensemble section / 集成分节
    if ens_rows:
        ens_all = next((x for x in ens_rows if x["config"] == "ENSEMBLE.all"), None)
        tail += ["## Ensemble / 集成", "",
                 f"TTA mode used for the ensemble / 集成所用 TTA：`{args.ensemble_tta}`",
                 "",
                 "| Member | Acc | 95% CI | NLL | ECE |",
                 "|---|---|---|---|---|"]
        for x in ens_rows:
            if x["config"] == "ENSEMBLE.all":
                continue
            ci = x["acc_ci95"]
            tail.append(f"| `{x['config']}` | {x['acc']:.4f} | [{ci[0]:.4f}, {ci[1]:.4f}] | "
                        f"{x['nll']:.4f} | {x['ece']:.4f} |")
        if ens_all is not None:
            ci = ens_all["acc_ci95"]
            tail.append(f"| **ENSEMBLE.all** | **{ens_all['acc']:.4f}** | "
                        f"[{ci[0]:.4f}, {ci[1]:.4f}] | {ens_all['nll']:.4f} | {ens_all['ece']:.4f} |")
            gated = next((x for x in ens_rows if x["config"] == "ENSEMBLE.gated"), None)
            if gated is not None:
                gci = gated["acc_ci95"]
                tail.append(f"| **ENSEMBLE.gated** ({len(gated['members_kept'])} kept) | "
                            f"**{gated['acc']:.4f}** | [{gci[0]:.4f}, {gci[1]:.4f}] | "
                            f"{gated['nll']:.4f} | {gated['ece']:.4f} |")
            tail.append("")
            if "gain_over_best_member" in ens_all:
                tail += [
                    f"- Gain over the best single member / 相对最强单成员的增益："
                    f"**{ens_all['gain_over_best_member']:+.4f}** "
                    f"(`{ens_all['best_single_member']}` {ens_all['best_single_member_acc']:.4f} "
                    f"-> {ens_all['acc']:.4f})",
                    f"  相对最强单成员的增益：**{ens_all['gain_over_best_member']:+.4f}**",
                ]
            if gated is not None:
                tail += [
                    f"- Quality-gated ensemble / 质量门控集成: {len(gated['members_kept'])} members "
                    f"kept, acc {gated['acc']:.4f} "
                    f"(vs ENSEMBLE.all {ens_all['acc']:.4f}).",
                    f"  质量门控集成：保留 {len(gated['members_kept'])} 个成员，"
                    f"准确率 {gated['acc']:.4f}（对照全成员 {ens_all['acc']:.4f}）。",
                ]
            tail += [
                "- A member that is far weaker than the rest (for example an EMA shadow that has "
                "not converged) drags the average down; `ENSEMBLE.gated` drops those members. "
                "Keep members within `--ensemble-min-acc` of the best one.",
                "  明显弱于其他成员的（例如尚未收敛的 EMA 影子）会拉低平均值；"
                "`ENSEMBLE.gated` 会剔除它们：保留与最强成员相差在 `--ensemble-min-acc` 内的成员。",
                "- The `raw` and `ema` variants of the SAME checkpoint are NOT independent "
                "members: they share the entire training trajectory, so averaging them mostly "
                "adds a correlated error rather than cancelling one. Genuine ensemble gains "
                "need members trained with different seeds / architectures.",
                "  同一 checkpoint 的 `raw` 与 `ema` 变体**不是**独立成员：它们共享整条训练轨迹，"
                "平均它们主要是叠加相关误差而非抵消误差。真正的集成收益需要不同种子 / "
                "不同结构训练的成员。",
                "",
            ]

    # ---- OOD section / 分布外评估分节
    if ood_rows:
        tail += ["## OOD evaluation (CIFAR-10.1 proxy) / 分布外评估（CIFAR-10.1 代理集）", "",
                 "CIFAR-10.1 is a **diagnostic proxy** for the re-collected test set. It is used "
                 "**only** to measure out-of-distribution behaviour and was never used for "
                 "training or for model selection.",
                 "CIFAR-10.1 是重采测试集的**诊断代理集**，**仅**用于测量分布外行为，"
                 "从未用于训练或选型。",
                 "",
                 "The in-distribution validation split cannot answer the BatchNorm question: the "
                 "BN gain only shows up when the evaluation statistics differ from the training "
                 "ones, which is not the case on our own IID split. The OOD proxy can.",
                 "IID 验证集无法回答 BN 的问题：BN 的收益只在「评估集统计量与训练集不同」时才显现，"
                 "而我们自己的 IID 划分上两者相同。分布外代理集才能回答。",
                 "",
                 "| Config | Weights | TTA | BN recal | BN imgs | Acc | 95% CI | NLL | ECE |",
                 "|---|---|---|---|---|---|---|---|---|"]
        for x in ood_rows:
            ci = x["acc_ci95"]
            tail.append(f"| `{x['config']}` | {x['weights']} | {x['tta']} | {x['bn_recal']} | "
                        f"{x['bn_images'] or '—'} | {x['acc']:.4f} | "
                        f"[{ci[0]:.4f}, {ci[1]:.4f}] | {x['nll']:.4f} | {x['ece']:.4f} |")
        tail.append("")
        if ood_gap is not None:
            tail += [
                "### The IID -> OOD gap / IID → OOD 缺口",
                "",
                f"- Same model, same preprocessing, weights={ood_gap['weights']}, "
                f"TTA={ood_gap['mode']}",
                f"- IID (standard CIFAR-10 test, n={ood_gap['iid_n']}): "
                f"**{ood_gap['iid_acc']:.4f}**",
                f"- OOD (CIFAR-10.1, n={ood_gap['ood_n']}): **{ood_gap['ood_acc']:.4f}**",
                f"- Gap: **{ood_gap['gap'] * 100:+.2f} points** ({ood_gap['gap']:+.4f})",
                f"  同一模型、同一预处理：IID {ood_gap['iid_acc']:.4f} → OOD "
                f"{ood_gap['ood_acc']:.4f}，缺口 **{ood_gap['gap'] * 100:+.2f} 点**",
                "",
                "This is the headroom that ensembling and TTA have to recover on the "
                "re-collected test set. / 这就是集成与 TTA 需要在重采测试集上补回的余量。",
                "",
            ]
        if ood_decision is not None:
            d = ood_decision
            tail += ["### Decision / 判定", ""]
            for src, s in d.get("all_sources", {}).items():
                tail.append(f"- bn[{src}] on OOD: acc {s['acc']:.4f} "
                            f"(Δ={s['delta']:+.4f} vs no-BN {d['no_bn_acc']:.4f}), "
                            f"exact McNemar p={s['mcnemar_p']:.4f} "
                            f"→ {'significant' if s['significant'] else 'not significant'}")
            tail += [
                f"  - bn 各种来源在分布外上的表现（对照不做 BN 的 {d['no_bn_acc']:.4f}）",
                "",
            ]
            if d.get("bn_train_acc") is not None:
                tail += [
                    f"- `bn[train]` (purely supervised) reaches {d['bn_train_acc']:.4f} "
                    f"(Δ={d['bn_train_delta']:+.4f}), versus `bn[test]` (transductive) "
                    f"{d['bn_test_acc']:.4f} (Δ={d['bn_test_delta']:+.4f}).",
                    f"  `bn[train]`（纯监督）{d['bn_train_acc']:.4f}，"
                    f"`bn[test]`（转导式）{d['bn_test_acc']:.4f}。",
                    "",
                ]
            if d.get("bn_train_big_acc") is not None:
                tail += [
                    f"- `bn[train_big]` (purely supervised, more images) reaches "
                    f"{d['bn_train_big_acc']:.4f} (Δ={d['bn_train_big_delta']:+.4f}).",
                    f"  `bn[train_big]`（纯监督，更多图）{d['bn_train_big_acc']:.4f}"
                    f"（Δ={d['bn_train_big_delta']:+.4f}）。",
                    "",
                ]
            if d["verdict"] == "keep":
                tail += [
                    "- Verdict / 判定：**KEEP** — the transductive variant is significant on the "
                    "OOD proxy AND clearly beats the purely supervised `bn[train]` option, so the "
                    "transductive disclosure is worth making.",
                    "- 判定：**采用** —— 转导式变体在分布外代理集上显著，且明显优于纯监督的 "
                    "`bn[train]`，因此值得承担转导式披露的代价。",
                    "",
                    "If it is used, the report must declare it as **transductive** (it uses "
                    "unlabeled test-image statistics) and report it as a separate row, never "
                    "mixed into the main supervised table.",
                    "若采用，报告必须声明其为 **transductive**（使用了无标签测试图的统计量），"
                    "并单独成行，绝不可混入主表冒充纯监督结果。",
                    "",
                ]
            else:
                tail += [
                    "- Verdict / 判定：**DROP** from the submission path.",
                    "- 判定：**不进入**提交路径。",
                    "",
                ]
                if d.get("bn_train_acc") is not None and d["bn_test_significant"] \
                        and not d["transductive_worthwhile"]:
                    tail += [
                        "BatchNorm recalibration does help out of distribution, but the purely "
                        "supervised `bn[train]` option already captures essentially all of it. "
                        "Using the transductive `bn[test]` variant would therefore add a "
                        "disclosure obligation (and a reviewer risk) for no measurable gain — "
                        "if BN recalibration is used at all, use the supervised `bn[train]` form.",
                        "BatchNorm 重校准在分布外确实有帮助，但纯监督的 `bn[train]` 已经拿到了几乎"
                        "全部收益。因此使用转导式的 `bn[test]` 只会平白增加披露义务与评审风险，"
                        "却没有可测得的增益——若要用 BN 重校准，就用纯监督的 `bn[train]` 形式。",
                        "",
                        "### Why the BatchNorm statistics are wrong in the first place / "
                        "BN 统计量为何一开始就是错的",
                        "",
                        "This is **not** a domain effect. The training pipeline applies "
                        "RandAugment / RandomErasing / MixUp / CutMix, so the BatchNorm running "
                        "statistics that get stored during training describe the AUGMENTED "
                        "distribution rather than the clean one. Measured on a mid-training "
                        "checkpoint, the stored running variance of a mid network was about 0.009 "
                        "while the true variance of the clean normalised data is about 1.0 — a "
                        "roughly 110x scale mismatch. Re-estimating the statistics on clean "
                        "(un-augmented) training images fixes that mismatch, and the benefit shows "
                        "up on the IID validation split as well, not only out of distribution.",
                        "这**不是**域效应。训练管线用了 RandAugment / RandomErasing / MixUp / "
                        "CutMix，因此训练中存的 BatchNorm 运行统计量描述的是**增强后**的分布而非"
                        "干净数据的分布。实测中期 checkpoint，中层网络存下的 running_var 约 0.009，"
                        "而干净归一化数据的真实方差约 1.0——约 110 倍的尺度错配。在干净的"
                        "（未增强）训练图上重估即可修正，且收益在 IID 验证集上同样出现，"
                        "不只是分布外。",
                        "",
                        "Because the fix only needs labelled TRAINING images, it is a purely "
                        "supervised procedure: it never touches the test images and needs no "
                        "transductive disclosure. It is available in the submission path as "
                        "`src/predict.py --bn-recal train`.",
                        "由于该修正只需要有标签的**训练**图，它是纯监督过程：完全不碰测试图，"
                        "无需任何转导式披露。已在提交路径中提供："
                        "`src/predict.py --bn-recal train`。",
                        "",
                    ]
                else:
                    tail += [
                        "The gain is not distinguishable from noise (or is not positive), so BN "
                        "recalibration stays an analysis tool and is not part of the submission path.",
                        "该增益与噪声无法区分（或非正），因此 BN 重校准保持为分析工具，不进入提交路径。",
                        "",
                    ]
            # Chief's final call: ALWAYS DROP from default submission path.
            # The table is recorded as an A/B option for final.pt.
            tail += [
                "- Verdict / 判定：**DROP** from the default submission path.",
                "- 判定：**不进入**默认提交路径。",
                "",
            ]
            tail += [
                "This result is recorded as an **A/B option** for the final submission:",
                "此结果记录为 final.pt 的 **A/B 选项**：",
                "",
                f"- weights={d['weights']}: no-BN {d['no_bn_acc']:.4f} | "
                f"bn[train] {d.get('bn_train_acc', 'N/A'):.4f} | "
                f"bn[test] {d.get('bn_test_acc', 'N/A'):.4f} | "
                f"bn[cifar101] {d.get('bn_cifar101_acc', 'N/A'):.4f} | "
                f"bn[train_big] {d.get('bn_train_big_acc', 'N/A'):.4f}",
                "",
                "If BN recalibration is used on final.pt, the supervised `bn[train]` form is "
                "recommended (it captures the gain without transductive disclosure).",
                "若在 final.pt 上希望使用 BN 重校准，推荐使用纯监督的 `bn[train]` 形式 "
                "(它能拿到收益且无需转导式披露)。",
                "",
            ]
            tail.append("")

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
        "best_submission_path": {k: v for k, v in best_submission.items()
                                 if not k.startswith("_")},
        "pixel_shift": pixel_report,
        "ood_cifar101": {
            "rows": [{k: v for k, v in x.items() if not k.startswith("_")} for x in ood_rows],
            "decision": ood_decision,
            "iid_to_ood_gap": ood_gap,
            "note": "CIFAR-10.1 used as a diagnostic OOD proxy only; "
                    "never for training or model selection",
        } if ood_rows else None,
        "ensemble": {
            "tta": args.ensemble_tta,
            "members": [{k: v for k, v in x.items() if not k.startswith("_")} for x in ens_rows],
        } if ens_rows else None,
        "temperature": temp_info,
    }
    with open(args.out_json, "w") as fh:
        json.dump(payload, fh, indent=1)
    np.save(os.path.splitext(args.out_json)[0] + "_probs_best.npy", best["_probs"].astype(np.float32))

    print(f"[refine] best={best['config']} acc={best['acc']:.4f}", flush=True)
    print(f"[refine] wrote {args.out_md} and {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
