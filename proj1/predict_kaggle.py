#!/usr/bin/env python3
"""
Standalone CLI Script to Generate Kaggle Competition `submission.csv` (`id,label`).

Supports:
  1. Primary Folder + CSV workflow:
       - `--test-dir ./test_images` (folder containing `test_0000.png` .. `test_1999.png`)
       - `--test-csv ./test.csv`    (CSV with columns `id,filename`)
  2. Legacy single-file archive workflow (`--test-file` for `.pkl`, `.npy`, `.npz`, `.pt`).

Matches the exact required schema of `sample_submission.csv`:
  - Two columns: `id,label` (lowercase)
  - 2,000 evaluation image rows (`id` from 0 to 1999, `label` integer in 0..9)

Usage (Folder + CSV):
  python3 predict_kaggle.py \
      --checkpoint ./outputs/ultra_sub5m_best.pth \
      --test-dir ./test_images \
      --test-csv ./test.csv \
      --output submission.csv
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from configs.model_configs import ResNetSpec, get_architecture_registry
from src.dataset import build_kaggle_test_loader, load_kaggle_evaluation_tensors
from src.model import CustomResNet, count_trainable_parameters


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Kaggle submission.csv (id,label)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained .pth checkpoint")
    parser.add_argument("--test-dir", type=str, default="./test_images", help="Path to test_images/ folder")
    parser.add_argument("--test-csv", type=str, default="./test.csv", help="Path to test.csv (id,filename)")
    parser.add_argument("--test-file", type=str, default=None, help="Optional legacy test file (.pkl/.npy/.npz/.pt)")
    parser.add_argument("--sample-submission", type=str, default="./sample_submission.csv", help="Optional sample_submission.csv")
    parser.add_argument("--output", type=str, default="submission.csv", help="Output CSV path")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    if "spec" in ckpt:
        spec = ResNetSpec(**ckpt["spec"])
    elif "config" in ckpt:
        spec = ResNetSpec(**ckpt["config"])
    else:
        spec = get_architecture_registry()["ultra_sub5m"]

    model = CustomResNet(spec).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    params = count_trainable_parameters(model)
    assert params <= 5_000_000, f"Model exceeds 5M parameters: {params:,}"
    print(f"Loaded {spec.name} ({params:,} parameters) from {args.checkpoint}")

    test_dir_path = Path(args.test_dir)
    test_csv_path = Path(args.test_csv)

    # Route 1: Folder of PNG images (`test_images/`) + `test.csv` (`id,filename`)
    if args.test_file is None and test_dir_path.exists() and test_csv_path.exists():
        print(f"Running inference on folder '{test_dir_path}' using metadata '{test_csv_path}'...")
        _, test_loader = build_kaggle_test_loader(
            test_csv=test_csv_path,
            test_img_dir=test_dir_path,
            batch_size=args.batch_size,
        )
        ids = []
        predictions = []
        for batch_ids, images in test_loader:
            images = images.to(device)
            # Horizontal-flip Test-Time Augmentation (TTA)
            logits = 0.5 * (model(images) + model(torch.flip(images, dims=[3])))
            preds = logits.argmax(dim=1).cpu().tolist()
            ids.extend([int(x) for x in batch_ids])
            predictions.extend([int(p) for p in preds])
    else:
        # Route 2: Fallback to `--test-file` or `load_kaggle_evaluation_tensors`
        target_path = args.test_file if args.test_file else str(test_dir_path)
        ids, tensors = load_kaggle_evaluation_tensors(
            test_data_path=target_path,
            sample_submission_path=args.sample_submission,
            test_csv_path=args.test_csv,
        )
        predictions = []
        for start in range(0, len(tensors), args.batch_size):
            batch = tensors[start : start + args.batch_size].to(device)
            logits = 0.5 * (model(batch) + model(torch.flip(batch, dims=[3])))
            predictions.extend(logits.argmax(dim=1).cpu().tolist())

    sub_df = pd.DataFrame({
        "id": np.asarray(ids, dtype=int),
        "label": np.asarray(predictions, dtype=int),
    })

    # Verify alignment with sample_submission.csv if present
    sample_sub_path = Path(args.sample_submission)
    if sample_sub_path.exists():
        expected_ids = pd.read_csv(sample_sub_path)["id"].astype(int).tolist()
        assert len(sub_df) == len(expected_ids), (
            f"Row count mismatch: got {len(sub_df)}, expected {len(expected_ids)}"
        )
        assert sub_df["id"].tolist() == expected_ids, "ID ordering does not match sample_submission.csv!"

    # Strict schema validation matching sample_submission.csv
    assert list(sub_df.columns) == ["id", "label"], "CSV header must be exactly: id,label"
    assert len(sub_df) == len(ids), "Prediction count mismatch"
    assert sub_df["label"].isin(range(10)).all(), "Labels must be integers in 0..9"

    sub_df.to_csv(args.output, index=False)
    print(f"Successfully wrote {len(sub_df)} predictions to {args.output} (columns: id,label)")
    print(sub_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()