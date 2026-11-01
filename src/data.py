"""Data pipeline for the CIFAR-10 variant competition.
CIFAR-10 变体竞赛的数据管线。

Images are shipped as 32x32 RGB PNG files plus CSV manifests, so we
materialise them once into a uint8 .npy cache (fast epochs afterwards).
图片以 32x32 RGB PNG + CSV 清单形式提供，故一次性缓存为 uint8 .npy，后续 epoch 读取很快。
"""
from __future__ import annotations

import csv
import os
from typing import Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

# CIFAR-10 statistics / CIFAR-10 的均值与标准差
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def _read_csv(path: str) -> list[dict]:
    """Read a manifest CSV. / 读取清单 CSV。"""
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _cache_paths(root: str, split: str) -> Tuple[str, str]:
    """Cache file pair for one split. / 某个切分对应的两个缓存文件路径。"""
    base = os.path.join(root, ".cache", split)
    return base + "_x.npy", base + "_y.npy"


def build_cache(root: str, split: str, need_labels: bool = True) -> Tuple[np.ndarray, np.ndarray | None]:
    """Load PNGs into (N,3,32,32) uint8 (+ (N,) int64 labels) and cache to disk.
    把 PNG 读成 (N,3,32,32) uint8（并可选读取 (N,) int64 标签），缓存到磁盘。"""
    x_path, y_path = _cache_paths(root, split)
    if os.path.exists(x_path) and (not need_labels or os.path.exists(y_path)):
        x = np.load(x_path, mmap_mode=None)
        y = np.load(y_path) if need_labels and os.path.exists(y_path) else None
        return x, y

    if split == "train":
        image_dir = os.path.join(root, "train_images")
        rows = _read_csv(os.path.join(root, "train_labels.csv"))
    else:
        image_dir = os.path.join(root, "test_images")
        rows = _read_csv(os.path.join(root, "test.csv"))

    x = np.zeros((len(rows), 3, 32, 32), dtype=np.uint8)
    y = np.zeros((len(rows),), dtype=np.int64)
    for i, row in enumerate(rows):
        arr = np.asarray(Image.open(os.path.join(image_dir, row["filename"])).convert("RGB"), dtype=np.uint8)
        # PNG is HWC; torch wants CHW. / PNG 是 HWC，torch 需要 CHW。
        x[i] = arr.transpose(2, 0, 1)
        if need_labels:
            y[i] = int(row["label"])
    os.makedirs(os.path.dirname(x_path), exist_ok=True)
    np.save(x_path, x)
    if need_labels:
        np.save(y_path, y)
    return x, (y if need_labels else None)


class CifarNumpy(Dataset):
    """Tensor view over an in-memory uint8 array with optional augmentation.
    内存 uint8 数组的 Tensor 视图，可选数据增强。"""

    def __init__(self, x: np.ndarray, y: np.ndarray, train: bool, aug: str = "basic",
                 randaug_n: int = 2, randaug_m: int = 9, erase_p: float = 0.25):
        self.x, self.y = x, y
        self.train = train
        if train:
            ops = [T.RandomCrop(32, padding=4), T.RandomHorizontalFlip()]
            # "basic" = flip+crop only; "strong" adds RandAugment + random erasing.
            # basic = 仅翻转+裁剪；strong 追加 RandAugment 与随机擦除。
            if aug == "strong":
                ops.append(T.RandAugment(num_ops=randaug_n, magnitude=randaug_m))
            ops.append(T.ToTensor())
            if aug == "strong":
                ops.append(T.RandomErasing(p=erase_p, scale=(0.02, 0.2)))
            ops.append(T.Normalize(MEAN, STD))
            self.tf = T.Compose(ops)
        else:
            self.tf = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int):
        img = Image.fromarray(self.x[i].transpose(1, 2, 0))
        return self.tf(img), int(self.y[i])


def load_train(root: str, seed: int = 0, n_val: int = 2500) -> Tuple[np.ndarray, ...]:
    """Load the training cache and split off a deterministic validation set.
    加载训练缓存，并划出确定性的验证集。"""
    x, y = build_cache(root, "train")
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(y))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    return x[tr_idx], y[tr_idx], x[val_idx], y[val_idx]


def load_test(root: str) -> Tuple[np.ndarray, list[str]]:
    """Load the test cache plus the ordered ids for submission.
    加载测试集缓存与有序 id 列表（用于提交）。"""
    x, _ = build_cache(root, "test", need_labels=False)
    ids = [r["id"] for r in _read_csv(os.path.join(root, "test.csv"))]
    return x, ids


def make_loader(ds: Dataset, batch_size: int, shuffle: bool, workers: int = 0) -> torch.utils.data.DataLoader:
    """DataLoader helper. / DataLoader 封装。"""
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                                       num_workers=workers, drop_last=shuffle,
                                       persistent_workers=workers > 0)
