"""
From-Scratch Data Loading, Augmentation, and CSV + Image Folder Pipeline for CIFAR-10.

Supports:
  1. Custom Folder + CSV format:
       - `train_labels.csv` (`id,filename,label`, 50,000 rows: `train_00000.png` .. `train_49999.png`)
         paired with the `train_images/` directory (with stratified train/validation split).
       - `test.csv` (`id,filename`, 2,000 rows: `test_0000.png` .. `test_1999.png`)
         paired with the `test_images/` directory for competition inference.
  2. Automatic fallback to `torchvision.datasets.CIFAR10` if `train_images/` is not present locally.
  3. Legacy array/pickle evaluation files (`.pkl`, `.npy`, `.npz`, `.pt`).
"""

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

CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def get_cifar10_transforms(enable_cutout: bool = True) -> Tuple[T.Compose, T.Compose]:
    """
    Returns (train_transform, eval_transform) for 32x32 CIFAR-10 images.
    Includes reflection-padded random cropping, random horizontal flipping,
    channel normalization, and optional Cutout (RandomErasing).
    """
    train_ops = [
        T.RandomCrop(32, padding=4, padding_mode="reflect"),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ]
    if enable_cutout:
        train_ops.append(T.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0.0))

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

    Expected CSV schemas:
      - Training/Validation (`is_test=False`): columns `id,filename,label`
      - Competition Test (`is_test=True`):     columns `id,filename`
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
    val_split: float = 0.1,
    seed: int = 2026,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits `train_labels.csv` (`id,filename,label`) into stratified training and validation
    DataFrames preserving exact class proportions across all 10 classes.
    """
    if val_split <= 0.0:
        return df.reset_index(drop=True), df.reset_index(drop=True)

    rng = np.random.default_rng(seed)
    train_indices = []
    val_indices = []

    for label_val, group in df.groupby("label"):
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
    val_split: float = 0.1,
    batch_size: int = 128,
    eval_batch_size: int = 256,
    enable_cutout: bool = True,
    seed: int = 2026,
) -> Tuple[Dataset, Dataset, DataLoader, DataLoader]:
    """
    Builds training and validation DataLoaders directly from `train_labels.csv` (`id,filename,label`)
    and the `train_images/` folder using a stratified validation split.
    """
    full_df = pd.read_csv(train_csv)
    train_df, val_df = stratified_train_val_split(full_df, val_split=val_split, seed=seed)
    train_transform, eval_transform = get_cifar10_transforms(enable_cutout=enable_cutout)

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
    enable_cutout: bool = True,
    train_csv: Optional[str] = None,
    train_img_dir: Optional[str] = None,
    val_split: float = 0.1,
    seed: int = 2026,
) -> Tuple[Dataset, Dataset, DataLoader, DataLoader]:
    """
    Builds training and evaluation DataLoaders for CIFAR-10.
    - First checks if `train_labels.csv` and `train_images/` are provided or exist in
      the working directory / `data_root`. If found, loads directly from the PNG folder + CSV.
    - Otherwise falls back cleanly to `torchvision.datasets.CIFAR10`.
    """
    root_path = Path(data_root)
    candidate_csvs = [Path(train_csv)] if train_csv else [Path("./train_labels.csv"), root_path / "train_labels.csv"]
    candidate_dirs = [Path(train_img_dir)] if train_img_dir else [Path("./train_images"), root_path / "train_images"]

    resolved_csv = next((p for p in candidate_csvs if p.exists() and p.is_file()), None)
    resolved_dir = next((p for p in candidate_dirs if p.exists() and p.is_dir()), None)

    if resolved_csv is not None and resolved_dir is not None:
        print(
            f"[Dataset] Loading custom CIFAR-10 images from '{resolved_dir}' "
            f"with labels from '{resolved_csv}' (val_split={val_split:.0%})"
        )
        return build_csv_folder_loaders(
            train_csv=resolved_csv,
            train_img_dir=resolved_dir,
            val_split=val_split,
            batch_size=batch_size,
            eval_batch_size=eval_batch_size,
            enable_cutout=enable_cutout,
            seed=seed,
        )

    print(f"[Dataset] Custom 'train_images/' folder not found; using torchvision CIFAR-10 in '{data_root}'")
    train_transform, eval_transform = get_cifar10_transforms(enable_cutout=enable_cutout)

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
    _, eval_transform = get_cifar10_transforms(enable_cutout=False)
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
    Loads unlabeled evaluation images for the Kaggle competition from either:
      1. A folder of PNG images (`test_images/`) paired with `test.csv` (`id,filename`), OR
      2. A serialized tensor/array file (`.pkl`, `.npy`, `.npz`, `.pt`).
    Aligns image IDs with `test.csv` / `sample_submission.csv` (`id,label`).
    """
    path = Path(test_data_path)
    if not path.exists():
        raise FileNotFoundError(f"Evaluation path not found: {path}")

    _, eval_transform = get_cifar10_transforms(enable_cutout=False)

    # Case 1: `test_data_path` is a directory of PNG images (`test_images/`) or `test.csv`
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
            # Fallback: sort PNG files in `test_images/` by numeric ID
            png_files = sorted(img_dir.glob("*.png"))
            if not png_files:
                raise FileNotFoundError(f"No .png images found in {img_dir}")
            ids = list(range(len(png_files)))
            tensors_list = [
                eval_transform(Image.open(p).convert("RGB")) for p in png_files
            ]
            return ids, torch.stack(tensors_list)

    # Case 2: Single-file tensor/array archive (.pkl, .npy, .npz, .pt)
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
