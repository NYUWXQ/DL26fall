"""Bootstrap the uint8 tensor caches from the official CIFAR-10 python batches.

Why this exists: on Torch the compute nodes have no internet and the competition
PNGs are 52,000 small files (slow to upload). We verified the shipped training
images are pixel-identical to the official CIFAR-10 train set (50,000/50,000), so
the cluster can download CIFAR-10 once on the login node and build the same cache
the local pipeline uses. Test images (2,000 PNGs, ~4 MB) are uploaded separately.

用官方的 CIFAR-10 python batch 生成 uint8 张量缓存。
原因：Torch 算力节点无外网，且竞赛 PNG 是 5.2 万个小文件（上传慢）。我们已实测
竞赛训练图与官方 CIFAR-10 训练集逐像素一致（50000/50000），所以在登录节点下载一次
CIFAR-10 并生成与本地管线相同的缓存；测试图（2000 张 PNG，约 4 MB）单独上传。
"""
from __future__ import annotations

import argparse
import os
import pickle

import numpy as np


def _load(path: str) -> dict:
    with open(path, "rb") as fh:
        return pickle.load(fh, encoding="bytes")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", default="data/cifar-10-batches-py")
    ap.add_argument("--root", default=".")
    args = ap.parse_args()

    xs, ys = [], []
    for i in range(1, 6):
        d = _load(os.path.join(args.batches, f"data_batch_{i}"))
        xs.append(d[b"data"].reshape(-1, 3, 32, 32))
        ys.append(np.asarray(d[b"labels"], dtype=np.int64))
    x = np.concatenate(xs).astype(np.uint8)
    y = np.concatenate(ys)

    os.makedirs(os.path.join(args.root, ".cache"), exist_ok=True)
    np.save(os.path.join(args.root, ".cache", "train_x.npy"), x)
    np.save(os.path.join(args.root, ".cache", "train_y.npy"), y)
    print(f"[cache] train_x {x.shape} {x.dtype}  train_y {y.shape}")

    # Official test split with labels, for IID evaluation only.
    # 官方测试集带标签，仅供 IID 评估使用。
    t = _load(os.path.join(args.batches, "test_batch"))
    tx = t[b"data"].reshape(-1, 3, 32, 32).astype(np.uint8)
    ty = np.asarray(t[b"labels"], dtype=np.int64)
    np.save(os.path.join(args.root, ".cache", "iid_test_x.npy"), tx)
    np.save(os.path.join(args.root, ".cache", "iid_test_y.npy"), ty)
    print(f"[cache] iid_test_x {tx.shape}  iid_test_y {ty.shape}")


if __name__ == "__main__":
    main()
