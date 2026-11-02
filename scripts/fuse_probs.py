"""Fuse probability matrices into candidate submissions, with optional hard-label votes.

Why a tool: during the competition the fusion rules were rewritten inline several
times (family pools, vote weights against a hard-label submission). Each rewrite is
an opportunity to mis-order rows or mis-weight a pool. This script makes the rule
explicit, checks all shapes, and prints the disagreement counts we use to reason
about how much a candidate differs from a known-scoring submission.

把概率矩阵融合成候选提交，可选加入硬标签投票。
为什么做成工具：比赛期间融合规则被内联重写过多次（家族池、对硬标签提交的投票权重），
每次重写都可能错行或错配权重。这个脚本把规则显式化、校验形状、并打印我们用来
判断候选差异量的分歧计数。

Usage / 用法:
  .venv/bin/python scripts/fuse_probs.py \
      --probs analysis/hpc_probs/eval_all_v3.npz \
      --keys kd_s10,kd_s11,kd_s12,kd_s13,full_s0,full_s1,full_s2,full_s3,arch_preact22x4_s0,arch_preact22x4_s1,arch_preact22x4_s2,arch_preact22x4_s3 \
      --hard submissions/../submission.csv --hard-weight 4 \
      --reference submissions/ens12_pure.csv \
      --out submissions/ens12_his_k4.csv
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np

N_EXPECTED = 2000
N_CLASSES = 10


def load_hard(path: str) -> np.ndarray:
    """Load a submission CSV as one-hot votes. / 把提交 CSV 读成 one-hot 票。"""
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    labels = np.array([int(r["label"]) for r in rows], dtype=np.int64)
    if len(labels) != N_EXPECTED or labels.min() < 0 or labels.max() >= N_CLASSES:
        raise SystemExit(f"{path}: expected {N_EXPECTED} labels in 0..{N_CLASSES-1}")
    return np.eye(N_CLASSES)[labels]


def load_probs(path: str, keys: list[str] | None) -> dict[str, np.ndarray]:
    """Load one npz of probability matrices. / 载入一个概率矩阵 npz。"""
    d = np.load(path)
    available = [k for k in d.files if k != "n"]
    use = available if keys is None else keys
    missing = [k for k in use if k not in available]
    if missing:
        raise SystemExit(f"{path}: missing keys {missing}; available {available}")
    out = {}
    for k in use:
        p = d[k]
        if p.shape != (N_EXPECTED, N_CLASSES):
            raise SystemExit(f"{k}: expected {(N_EXPECTED, N_CLASSES)}, got {p.shape}")
        out[k] = p.astype(np.float64)
    return out


def write_csv(path: str, score: np.ndarray) -> np.ndarray:
    """Write argmax predictions in sample order and return them. / 按样本顺序写出预测。"""
    pred = score.argmax(1)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "label"])
        for i, lab in enumerate(pred):
            w.writerow([i, int(lab)])
    return pred


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", nargs="+", required=True)
    ap.add_argument("--keys", default="", help="comma list; default all / 逗号分隔，默认全部")
    ap.add_argument("--hard", default="", help="submission CSV to use as hard-label votes / 硬标签票来源")
    ap.add_argument("--hard-weight", type=float, default=0.0,
                    help="how many model-votes the hard labels count as / 硬标签折算成几个模型票")
    ap.add_argument("--reference", default="", help="CSV to report disagreement against / 对照 CSV")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    keys = [k.strip() for k in args.keys.split(",") if k.strip()] or None
    pooled: dict[str, np.ndarray] = {}
    for path in args.probs:
        pooled.update(load_probs(path, keys))

    # Sum of probabilities = one vote-mass per model. / 概率求和：每个模型一票质量。
    score = sum(pooled.values())
    print(f"[fuse] pooled {len(pooled)} models: {list(pooled)}")
    print(f"[fuse] our argmax mean confidence = {(score / len(pooled)).max(1).mean():.4f}")

    if args.hard and args.hard_weight:
        hard = load_hard(args.hard)
        score = score + args.hard_weight * hard
        print(f"[fuse] added hard-label votes with weight {args.hard_weight}")

    pred = write_csv(args.out, score)
    print(f"[fuse] wrote {args.out} (rows={len(pred)})")

    if args.reference and os.path.exists(args.reference):
        with open(args.reference, newline="") as fh:
            ref = np.array([int(r["label"]) for r in csv.DictReader(fh)])
        print(f"[fuse] differs from {args.reference} on {(pred != ref).sum()} / {len(pred)} rows")
    if args.hard and args.hard_weight:
        hard_labels = load_hard(args.hard).argmax(1)
        print(f"[fuse] differs from hard-label source on {(pred != hard_labels).sum()} / {len(pred)} rows")


if __name__ == "__main__":
    main()
