"""Train/test domain-shift probe: can a linear classifier separate train from test?

Pure numpy + scipy, no torch.  Loads the uint8 image caches, builds a balanced
train/test *domain* label (0 = competition train, 1 = competition test), fits an
L2-regularised logistic regression on flattened pixels, and reports accuracy /
AUC / per-channel weight summary.  A number near 0.5 = same distribution;
significantly above 0.5 = measurable domain shift.

用法: python analysis/domain_probe.py [--n-train 5000] [--seed 0]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from PIL import Image
from scipy.optimize import minimize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_split(split: str) -> np.ndarray:
    """(N,3,32,32) uint8 for one split, in manifest order. / 按清单顺序读取某个切分。"""
    cache = os.path.join(ROOT, ".cache", f"{split}_x_probe.npy")
    if os.path.exists(cache):
        return np.load(cache)
    if split == "train":
        manifest, imgdir = "train_labels.csv", "train_images"
    else:
        manifest, imgdir = "test.csv", "test_images"
    import csv
    rows = list(csv.DictReader(open(os.path.join(ROOT, manifest))))
    out = np.zeros((len(rows), 3, 32, 32), dtype=np.uint8)
    for i, r in enumerate(rows):
        out[i] = np.asarray(Image.open(os.path.join(ROOT, imgdir, r["filename"])).convert("RGB"),
                            dtype=np.uint8).transpose(2, 0, 1)
    np.save(cache, out)
    return out


def standardise(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu, sd = X.reshape(X.shape[0], -1).mean(0), X.reshape(X.shape[0], -1).std(0) + 1e-6
    return (X.reshape(X.shape[0], -1) - mu) / sd, mu, sd


def logistic_loss(w: np.ndarray, X: np.ndarray, y: np.ndarray, lam: float) -> float:
    z = X @ w
    z = np.clip(z, -30, 30)
    ll = np.where(y == 1, -np.log1p(np.exp(-z)), -z - np.log1p(np.exp(-z)))
    return ll.mean() + 0.5 * lam * (w[1:] ** 2).sum()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tr = load_split("train")
    te = load_split("test")
    rng = np.random.RandomState(args.seed)
    idx = rng.permutation(len(tr))[: args.n_train]

    X = np.concatenate([tr[idx].reshape(len(idx), -1), te.reshape(len(te), -1)], 0).astype(np.float32)
    y = np.concatenate([np.zeros(len(idx), dtype=np.int64), np.ones(len(te), dtype=np.int64)])
    # shuffle so optimiser sees mixed batches
    perm = rng.permutation(len(y))
    X, y = X[perm], y[perm]

    Xs, mu, sd = standardise(X)
    n, d = Xs.shape
    lam = 1e-3
    w0 = np.zeros(d)
    res = minimize(logistic_loss, w0, args=(Xs, y, lam), method="L-BFGS-B",
                   options={"maxiter": 300, "disp": False})
    w = res.x
    z = np.clip(Xs @ w, -30, 30)
    prob = 1.0 / (1.0 + np.exp(-z))
    pred = (prob >= 0.5).astype(np.int64)
    acc = (pred == y).mean()
    tp = ((pred == 1) & (y == 1)).sum(); fp = ((pred == 1) & (y == 0)).sum()
    fn = ((pred == 0) & (y == 1)).sum(); tn = ((pred == 0) & (y == 0)).sum()
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    auc = ((prob > 0.5) & (y == 1)).sum()  # placeholder, replaced below
    # proper AUC via rank statistic
    pos = prob[y == 1]; neg = prob[y == 0]
    auc = float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())

    print(f"[domain] n_train={args.n_train} n_test={len(te)} dim={d} lambda={lam}")
    print(f"[domain] loss={res.fun:.5f}  acc={acc:.4f}  precision={prec:.4f}  recall={rec:.4f}  AUC={auc:.4f}")
    print(f"[domain] confusion tn={tn} fp={fp} fn={fn} tp={tp}")
    print(f"[domain] baseline (predict majority class={y.mean():.3f}) acc={max(y.mean(),1-y.mean()):.4f}")
    # weight summary per channel (first 1024 entries = channel 0, etc.)
    wv = w[1:].reshape(3, 32, 32)
    print(f"[domain] weight L2 per channel: R={np.linalg.norm(wv[0]):.4f} G={np.linalg.norm(wv[1]):.4f} B={np.linalg.norm(wv[2]):.4f}")
    print(f"[domain] bias={w[0]:.4f}")


if __name__ == "__main__":
    main()