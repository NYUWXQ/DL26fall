"""
From-Scratch Data Loading, CIFAR-10 AutoAugment, and CSV + Image Folder Pipeline (Try 5).

Key Upgrades in Try 5:
  1. Official CIFAR-10 AutoAugment (`T.AutoAugment(T.AutoAugmentPolicy.CIFAR10)`) + RandomCrop + Flip + Cutout.
  2. Default `val_split=0.0` option to train on 100% of the 50,000 competition images (`train_labels.csv` + `train_images/`)
     while using a 2,500-image clean subset for fast checkpoint tracking.
  3. Automatic path discovery across `/kaggle/input/`, `./train_images/`, and `/content/`.
"""

import glob
import pickle
from pathlib import Path
from typing import Tuple, List, Optional, Callable, Union

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as T

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

CIFAR10_CLASSES = [f"class_{i}" for i in range(10)]


def get_cifar10_transforms(
    enable_autoaugment: bool = True,
    enable_cutout: bool = True,
) -> Tuple[T.Compose, T.Compose]:
    """
    Returns (train_transform, eval_transform) for 32x32 CIFAR-10 images.
    Includes reflection-padded random cropping, random horizontal flipping,
    official CIFAR-10 AutoAugment, channel normalization, and Cutout (RandomErasing).
    """
    train_ops = [
        T.RandomCrop(32, padding=4, padding_mode="reflect"),
        T.RandomHorizontalFlip(p=0.5),
    ]
    if enable_autoaugment:
        train_ops.append(T.AutoAugment(T.AutoAugmentPolicy.CIFAR10))
    train_ops.extend([
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    if enable_cutout:
        train_ops.append(T.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0.0))

    train_transform = T.Compose(train_ops)
    eval_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    return train_transform, eval_transform


class CIFARCSVDataset(Dataset):
    """
    Custom PyTorch Dataset that links a CSV metadata file (`train_labels.csv` or `test.csv`)
    with a folder of PNG images (`train_images/` or `test_images/`).
    """

    def __init__(
        self,
        data_source: Union[str, Path, pd.DataFrame],
        img_dir: Union[str, Path],
        transform: Optional[Callable] = None,
        is_test: bool = False,
    ) -> None:
        super().__init__()
        if isinstance(data_source, pd.DataFrame):
            self.df = data_source.reset_index(drop=True).copy()
        else:
            csv_path = Path(data_source)
            if not csv_path.exists():
                raise FileNotFoundError(f"CSV file not found: {csv_path}")
            self.df = pd.read_csv(csv_path)

        self.img_dir = Path(img_dir)
        if not self.img_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.img_dir}")

        self.transform = transform
        self.is_test = is_test
        self.classes = CIFAR10_CLASSES

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_image_path(self, row: pd.Series) -> Path:
        if "filename" in row and pd.notna(row["filename"]):
            fname = str(row["filename"]).strip()
        else:
            img_id = int(row["id"])
            fname = f"test_{img_id:04d}.png" if self.is_test else f"train_{img_id:05d}.png"
        return self.img_dir / fname

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = self._resolve_image_path(row)
        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        if self.is_test:
            img_id = int(row["id"])
            return img_id, image

        label = int(row["label"])
        return image, label


def stratified_train_val_split(
    df: pd.DataFrame,
    val_split: float = 0.0,
    seed: int = 2026,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    If `val_split <= 0.0` (Try 5 default), trains on 100% of the data (all 50,000 images)
    and uses a 2,500-image sample with clean transforms for fast checkpoint validation.
    Otherwise performs a stratified train/validation split across all 10 classes.
    """
    if val_split <= 0.0:
        train_df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        n_val = min(2500, len(df))
        val_df = df.sample(n=n_val, random_state=seed).sort_values("id").reset_index(drop=True)
        return train_df, val_df

    rng = np.random.default_rng(seed)
    train_indices = []
    val_indices = []

    for _, group in df.groupby("label"):
        idx_array = group.index.to_numpy().copy()
        rng.shuffle(idx_array)
        n_val = max(1, int(round(len(idx_array) * val_split)))
        val_indices.extend(idx_array[:n_val].tolist())
        train_indices.extend(idx_array[n_val:].tolist())

    train_df = df.loc[train_indices].sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_df = df.loc[val_indices].sort_values("id").reset_index(drop=True)
    return train_df, val_df


def build_csv_folder_loaders(
    train_csv: Union[str, Path] = "./train_labels.csv",
    train_img_dir: Union[str, Path] = "./train_images",
    val_split: float = 0.0,
    batch_size: int = 128,
    eval_batch_size: int = 256,
    enable_autoaugment: bool = True,
    enable_cutout: bool = True,
    seed: int = 2026,
) -> Tuple[Dataset, Dataset, DataLoader, DataLoader]:
    """
    Builds training and validation DataLoaders from `train_labels.csv` (`id,filename,label`)
    and `train_images/` using CIFAR-10 AutoAugment + Cutout.
    """
    full_df = pd.read_csv(train_csv)
    train_df, val_df = stratified_train_val_split(full_df, val_split=val_split, seed=seed)
    train_transform, eval_transform = get_cifar10_transforms(
        enable_autoaugment=enable_autoaugment,
        enable_cutout=enable_cutout,
    )

    train_set = CIFARCSVDataset(
        data_source=train_df,
        img_dir=train_img_dir,
        transform=train_transform,
        is_test=False,
    )
    val_set = CIFARCSVDataset(
        data_source=val_df,
        img_dir=train_img_dir,
        transform=eval_transform,
        is_test=False,
    )

    use_cuda = torch.cuda.is_available()
    workers = 2 if use_cuda else 0

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=use_cuda,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=use_cuda,
    )
    return train_set, val_set, train_loader, val_loader


def build_cifar10_WarningFree_loaders(
    data_root: str = "./data",
    batch_size: int = 128,
    eval_batch_size: int = 256,
    enable_autoaugment: bool = True,
    enable_cutout: bool = True,
    train_csv: Optional[str] = None,
    train_img_dir: Optional[str] = None,
    val_split: float = 0.0,
    seed: int = 2026,
) -> Tuple[Dataset, Dataset, DataLoader, DataLoader]:
    """
    Automatically locates `train_labels.csv` and `train_00000.png` across explicit CLI paths,
    `/kaggle/input/`, `./train_images/`, or `/content/`. Falls back to `torchvision` only if
    no competition files exist.
    """
    root_path = Path(data_root)
    candidate_csvs = [Path(train_csv)] if train_csv else []
    candidate_csvs.extend([
        Path("./train_labels.csv"),
        root_path / "train_labels.csv",
        *[Path(p) for p in glob.glob("/kaggle/input/**/train_labels.csv", recursive=True)],
        *[Path(p) for p in glob.glob("/content/**/train_labels.csv", recursive=True)],
    ])

    candidate_dirs = []
    if train_img_dir and (Path(train_img_dir) / "train_00000.png").exists():
        candidate_dirs.append(Path(train_img_dir))
    for p_str in (
        glob.glob("./**/train_00000.png", recursive=True)
        + glob.glob("/kaggle/input/**/train_00000.png", recursive=True)
        + glob.glob("/content/**/train_00000.png", recursive=True)
    ):
        candidate_dirs.append(Path(p_str).parent)

    resolved_csv = next((p for p in candidate_csvs if p.exists() and p.is_file()), None)
    resolved_dir = next((p for p in candidate_dirs if p.exists() and p.is_dir()), None)

    if resolved_csv is not None and resolved_dir is not None:
        mode_str = "100% full data" if val_split <= 0.0 else f"val_split={val_split:.0%}"
        print(
            f"[Dataset] Loading competition images from '{resolved_dir}' "
            f"with labels from '{resolved_csv}' ({mode_str}, AutoAugment={enable_autoaugment})"
        )
        return build_csv_folder_loaders(
            train_csv=resolved_csv,
            train_img_dir=resolved_dir,
            val_split=val_split,
            batch_size=batch_size,
            eval_batch_size=eval_batch_size,
            enable_autoaugment=enable_autoaugment,
            enable_cutout=enable_cutout,
            seed=seed,
        )

    print(f"[Dataset] Competition 'train_images/' not found; using torchvision CIFAR-10 in '{data_root}'")
    train_transform, eval_transform = get_cifar10_transforms(
        enable_autoaugment=enable_autoaugment,
        enable_cutout=enable_cutout,
    )

    train_set = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=train_transform
    )
    test_set = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=eval_transform
    )

    use_cuda = torch.cuda.is_available()
    workers = 2 if use_cuda else 0

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=use_cuda,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=use_cuda,
    )
    return train_set, test_set, train_loader, test_loader


def build_kaggle_test_loader(
    test_csv: Union[str, Path] = "./test.csv",
    test_img_dir: Union[str, Path] = "./test_images",
    batch_size: int = 256,
) -> Tuple[CIFARCSVDataset, DataLoader]:
    """
    Builds an evaluation DataLoader for the unlabeled competition `test_images/` folder
    and `test.csv` (`id,filename`).
    """
    _, eval_transform = get_cifar10_transforms(enable_autoaugment=False, enable_cutout=False)
    test_dataset = CIFARCSVDataset(
        data_source=test_csv,
        img_dir=test_img_dir,
        transform=eval_transform,
        is_test=True,
    )
    use_cuda = torch.cuda.is_available()
    workers = 2 if use_cuda else 0
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=use_cuda,
    )
    return test_dataset, test_loader


def load_kaggle_evaluation_tensors(
    test_data_path: str,
    sample_submission_path: Optional[str] = None,
    test_csv_path: Optional[str] = "./test.csv",
) -> Tuple[List[int], torch.Tensor]:
    """
    Loads unlabeled evaluation images for the Kaggle competition from either a folder of PNGs
    (`test_images/`) paired with `test.csv` (`id,filename`), or a serialized file (`.pkl`, `.npy`, `.npz`, `.pt`).
    """
    path = Path(test_data_path)
    if not path.exists():
        raise FileNotFoundError(f"Evaluation path not found: {path}")

    _, eval_transform = get_cifar10_transforms(enable_autoaugment=False, enable_cutout=False)

    if path.is_dir() or path.suffix.lower() == ".csv":
        if path.is_dir():
            img_dir = path
            csv_candidates = [
                Path(test_csv_path) if test_csv_path else None,
                path.parent / "test.csv",
                Path("./test.csv"),
            ]
            csv_file = next((c for c in csv_candidates if c is not None and c.exists()), None)
        else:
            csv_file = path
            img_dir = path.parent / "test_images"

        if csv_file is not None and csv_file.exists():
            dataset = CIFARCSVDataset(
                data_source=csv_file,
                img_dir=img_dir,
                transform=eval_transform,
                is_test=True,
            )
            ids: List[int] = []
            tensors_list: List[torch.Tensor] = []
            for img_id, img_tensor in dataset:
                ids.append(int(img_id))
                tensors_list.append(img_tensor)
            return ids, torch.stack(tensors_list)
        else:
            png_files = sorted(img_dir.glob("*.png"))
            if not png_files:
                raise FileNotFoundError(f"No .png images found in {img_dir}")
            ids = list(range(len(png_files)))
            tensors_list = [
                eval_transform(Image.open(p).convert("RGB")) for p in png_files
            ]
            return ids, torch.stack(tensors_list)

    if path.suffix in {".pkl", ".pickle"}:
        with path.open("rb") as f:
            payload = pickle.load(f, encoding="bytes")
        if isinstance(payload, dict):
            raw_imgs = payload.get(b"data", payload.get("data"))
            ids = payload.get(b"ids", payload.get(b"id", payload.get("id", payload.get("ids"))))
        else:
            raw_imgs = payload
            ids = None
    elif path.suffix == ".npy":
        raw_imgs = np.load(path)
        ids = None
    elif path.suffix == ".npz":
        archive = np.load(path)
        raw_imgs = archive["data"] if "data" in archive else archive[archive.files[0]]
        ids = archive["id"] if "id" in archive else (archive["ids"] if "ids" in archive else None)
    elif path.suffix in {".pt", ".pth"}:
        obj = torch.load(path, map_location="cpu", weights_only=False)
        raw_imgs = obj["data"] if isinstance(obj, dict) and "data" in obj else obj
        ids = obj.get("id", obj.get("ids")) if isinstance(obj, dict) else None
    else:
        raise ValueError(f"Unsupported file extension: {path.suffix}")

    if isinstance(raw_imgs, torch.Tensor):
        raw_imgs = raw_imgs.cpu().numpy()

    if raw_imgs.ndim == 2 and raw_imgs.shape[1] == 3072:
        raw_imgs = raw_imgs.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    elif raw_imgs.ndim == 4 and raw_imgs.shape[1] == 3:
        raw_imgs = raw_imgs.transpose(0, 2, 3, 1)

    if sample_submission_path and Path(sample_submission_path).exists():
        sub_df = pd.read_csv(sample_submission_path)
        ids = sub_df["id"].astype(int).tolist()
    elif ids is None:
        ids = list(range(len(raw_imgs)))
    else:
        ids = [int(x) for x in ids]

    tensors = torch.stack([eval_transform(img) for img in raw_imgs])
    return ids, tensors
