#!/usr/bin/env python3
"""
Main CLI Entrypoint to Train & Evaluate Custom Sub-5M Parameter ResNets on CIFAR-10.

Supports loading directly from the custom competition folders & CSVs:
  - `train_images/` + `train_labels.csv` (`id,filename,label`, 50,000 images)
  - `test_images/`  + `test.csv` (`id,filename`, 2,000 images) -> outputs `submission.csv` (`id,label`)
Or automatically falls back to `torchvision.datasets.CIFAR10` if `train_images/` is not present.

Examples:
  1. Train Champion <5M ResNet on `train_images/` + `train_labels.csv` and generate `submission.csv`:
       python3 train.py \
           --arch champion_sub5m \
           --train-img-dir ./train_images \
           --train-csv ./train_labels.csv \
           --test-img-dir ./test_images \
           --test-csv ./test.csv \
           --epochs 100 --batch-size 128

  2. Run the 4-architecture ablation study (comparing N, B_i, C_i, K_i, P):
       python3 train.py --run-ablation --ablation-epochs 25
"""

import argparse
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
    parser = argparse.ArgumentParser(description="Train Custom Sub-5M Parameter ResNet on CIFAR-10")
    parser.add_argument("--arch", type=str, default="champion_sub5m", choices=list(get_architecture_registry().keys()))
    parser.add_argument("--data-dir", type=str, default="./data", help="Fallback directory for torchvision CIFAR-10")
    parser.add_argument("--train-csv", type=str, default="./train_labels.csv", help="Path to train_labels.csv (id,filename,label)")
    parser.add_argument("--train-img-dir", type=str, default="./train_images", help="Path to train_images/ directory")
    parser.add_argument("--val-split", type=float, default=0.1, help="Stratified validation split ratio when using train_labels.csv")
    parser.add_argument("--test-csv", type=str, default="./test.csv", help="Path to test.csv (id,filename)")
    parser.add_argument("--test-img-dir", type=str, default="./test_images", help="Path to test_images/ directory")
    parser.add_argument("--submission-out", type=str, default="./submission.csv", help="Output path for Kaggle submission.csv")
    parser.add_argument("--out-dir", type=str, default="./outputs")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--run-ablation", action="store_true", help="Train all 4 candidate sub-5M architectures")
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

    # Train selected main model (default: champion_sub5m, 4,917,450 parameters)
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
    )

    class_names = getattr(test_set, "classes", CIFAR10_CLASSES)
    per_class = evaluate_per_class_accuracy(main_model, test_loader, class_names, device)
    result["per_class_accuracy"] = per_class
    (out_dir / f"{main_spec.name}_summary.json").write_text(json.dumps(result, indent=2))

    plot_training_curves(result, out_dir)
    plot_per_class_accuracy(per_class, result["best_test_acc_tta"], main_spec.name, out_dir)

    # Automatically generate Kaggle submission.csv if test.csv and test_images/ exist
    test_csv_path = Path(args.test_csv)
    test_img_dir = Path(args.test_img_dir)
    if test_csv_path.exists() and test_img_dir.exists():
        generate_competition_submission(
            model=main_model,
            test_csv=test_csv_path,
            test_img_dir=test_img_dir,
            output_csv=Path(args.submission_out),
            batch_size=256,
            device=device,
        )


if __name__ == "__main__":
    main()
