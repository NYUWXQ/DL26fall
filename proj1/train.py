#!/usr/bin/env python3
"""
Main CLI Entrypoint to Train & Evaluate Try 5 (`ultra_sub5m`, 4,977,834 Parameters) on CIFAR-10.

Key Upgrades in Try 5:
  - Architecture: `ultra_sub5m` (`C=[64, 144, 208, 280]`, `B=[2, 2, 2, 2]`, 4,977,834 parameters — 99.56% of 5M cap)
  - Augmentation: CIFAR-10 AutoAugment (`T.AutoAugmentPolicy.CIFAR10`) + RandomCrop + Flip + Cutout
  - Regularization: Mixup (`alpha=0.3` on 50% of batches) + Label Smoothing (`0.05`)
  - Data Utilization: Trains on 100% of competition training images (`--val-split 0.0`, all 50,000 images) for 220 epochs

Usage:
  python3 train.py \
      --arch ultra_sub5m \
      --train-img-dir ./train_images \
      --train-csv ./train_labels.csv \
      --test-img-dir ./test_images \
      --test-csv ./test.csv \
      --epochs 220 --batch-size 128
"""

import argparse
import glob
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from configs.model_configs import get_architecture_registry
from src.dataset import (
    CIFAR10_CLASSES,
    build_cifar10_WarningFree_loaders,
    build_kaggle_test_loader,
)
from src.model import CustomResNet, audit_model_parameters
from src.trainer import train_and_evaluate, evaluate_per_class_accuracy
from src.visualize import (
    plot_training_curves,
    plot_per_class_accuracy,
    plot_ablation_comparison,
)


def seed_everything(seed: int = 2026) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


@torch.no_grad()
def generate_competition_submission(
    model: CustomResNet,
    test_csv: Path,
    test_img_dir: Path,
    output_csv: Path,
    batch_size: int,
    device: torch.device,
) -> None:
    """
    Runs horizontal-flip TTA inference on `test_images/` + `test.csv` (`id,filename`)
    and writes `submission.csv` (`id,label`).
    """
    model.eval()
    _, kaggle_loader = build_kaggle_test_loader(
        test_csv=test_csv,
        test_img_dir=test_img_dir,
        batch_size=batch_size,
    )

    all_ids = []
    all_preds = []
    for img_ids, images in kaggle_loader:
        images = images.to(device)
        logits = 0.5 * (model(images) + model(torch.flip(images, dims=[3])))
        preds = logits.argmax(dim=1).cpu().tolist()
        all_ids.extend([int(i) for i in img_ids])
        all_preds.extend([int(p) for p in preds])

    sub_df = pd.DataFrame({"id": all_ids, "label": all_preds})
    assert list(sub_df.columns) == ["id", "label"], "CSV header must be exactly: id,label"
    sub_df.to_csv(output_csv, index=False)
    print(f"[Submission] Wrote {len(sub_df)} predictions to '{output_csv}' (columns: id,label)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Try 5 Ultra Sub-5M Parameter ResNet on CIFAR-10")
    parser.add_argument("--arch", type=str, default="ultra_sub5m", choices=list(get_architecture_registry().keys()))
    parser.add_argument("--data-dir", type=str, default="./data", help="Fallback directory for torchvision CIFAR-10")
    parser.add_argument("--train-csv", type=str, default="./train_labels.csv", help="Path to train_labels.csv (id,filename,label)")
    parser.add_argument("--train-img-dir", type=str, default="./train_images", help="Path to train_images/ directory")
    parser.add_argument("--val-split", type=float, default=0.0, help="0.0 trains on 100% of data (50,000 images); >0 holds out validation split")
    parser.add_argument("--test-csv", type=str, default="./test.csv", help="Path to test.csv (id,filename)")
    parser.add_argument("--test-img-dir", type=str, default="./test_images", help="Path to test_images/ directory")
    parser.add_argument("--submission-out", type=str, default="./submission.csv", help="Output path for Kaggle submission.csv")
    parser.add_argument("--out-dir", type=str, default="./outputs")
    parser.add_argument("--epochs", type=int, default=220)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--mixup-alpha", type=float, default=0.3)
    parser.add_argument("--mixup-prob", type=float, default=0.5)
    parser.add_argument("--run-ablation", action="store_true", help="Train all candidate sub-5M architectures")
    parser.add_argument("--ablation-epochs", type=int, default=25)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    registry = get_architecture_registry()
    _, test_set, train_loader, test_loader = build_cifar10_WarningFree_loaders(
        data_root=args.data_dir,
        batch_size=args.batch_size,
        enable_autoaugment=True,
        enable_cutout=True,
        train_csv=args.train_csv,
        train_img_dir=args.train_img_dir,
        val_split=args.val_split,
        seed=args.seed,
    )

    if args.run_ablation:
        ablation_rows = []
        for name, spec in registry.items():
            seed_everything(args.seed)
            model = CustomResNet(spec).to(device)
            res = train_and_evaluate(
                model=model,
                train_loader=train_loader,
                test_loader=test_loader,
                epochs=args.ablation_epochs,
                peak_lr=args.lr,
                weight_decay=args.weight_decay,
                label_smoothing=args.label_smoothing,
                output_dir=out_dir,
                device=device,
                mixup_alpha=args.mixup_alpha,
                mixup_prob=args.mixup_prob,
            )
            ablation_rows.append({
                "model": name,
                "stages_N": len(spec.channels),
                "channels_C": spec.channels,
                "blocks_B": spec.blocks_per_stage,
                "skip_K": spec.skip_kernels,
                "pool_P": spec.pool_size,
                "parameters": res["parameters"],
                "best_test_acc": res["best_test_acc"],
                "best_test_acc_tta": res["best_test_acc_tta"],
                "training_seconds": res["training_seconds"],
            })
        (out_dir / "ablation_summary.json").write_text(json.dumps(ablation_rows, indent=2))
        plot_ablation_comparison(ablation_rows, out_dir)

    # Train selected main model (default: ultra_sub5m, 4,977,834 parameters)
    seed_everything(args.seed)
    main_spec = registry[args.arch]
    main_model = CustomResNet(main_spec).to(device)
    audit_model_parameters(main_model, max_budget=5_000_000, run_summary=True)

    result = train_and_evaluate(
        model=main_model,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=args.epochs,
        peak_lr=args.lr,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        output_dir=out_dir,
        device=device,
        mixup_alpha=args.mixup_alpha,
        mixup_prob=args.mixup_prob,
    )

    class_names = getattr(test_set, "classes", CIFAR10_CLASSES)
    per_class = evaluate_per_class_accuracy(main_model, test_loader, class_names, device)
    result["per_class_accuracy"] = per_class
    (out_dir / f"{main_spec.name}_summary.json").write_text(json.dumps(result, indent=2))

    plot_training_curves(result, out_dir)
    plot_per_class_accuracy(per_class, result["best_test_acc_tta"], main_spec.name, out_dir)

    # Automatically locate test.csv and test_images/ and generate submission.csv
    test_csv_candidates = [
        Path(args.test_csv),
        *[Path(p) for p in glob.glob("/kaggle/input/**/test.csv", recursive=True)],
        *[Path(p) for p in glob.glob("/content/**/test.csv", recursive=True)],
    ]
    test_png_candidates = []
    if (Path(args.test_img_dir) / "test_0000.png").exists():
        test_png_candidates.append(Path(args.test_img_dir))
    for p_str in (
        glob.glob("./**/test_0000.png", recursive=True)
        + glob.glob("/kaggle/input/**/test_0000.png", recursive=True)
        + glob.glob("/content/**/test_0000.png", recursive=True)
    ):
        test_png_candidates.append(Path(p_str).parent)

    resolved_test_csv = next((p for p in test_csv_candidates if p.exists() and p.is_file()), None)
    resolved_test_dir = next((p for p in test_png_candidates if p.exists() and p.is_dir()), None)

    if resolved_test_csv is not None and resolved_test_dir is not None:
        generate_competition_submission(
            model=main_model,
            test_csv=resolved_test_csv,
            test_img_dir=resolved_test_dir,
            output_csv=Path(args.submission_out),
            batch_size=256,
            device=device,
        )


if __name__ == "__main__":
    main()
