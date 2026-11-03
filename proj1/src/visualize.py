"""
Plotting Utilities for Deep Learning Project 1 Report & Notebook Verification.
"""

from pathlib import Path
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import numpy as np


def plot_training_curves(result: Dict[str, Any], out_dir: Path) -> Path:
    """Generates a 2-panel figure of Cross-Entropy Loss and Accuracy vs. Epoch."""
    out_dir.mkdir(parents=True, exist_ok=True)
    history = result["history"]
    name = result["spec"]["name"]
    params = result["parameters"]
    best_acc = result["best_test_acc"]
    epochs = [h["epoch"] for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    axes[0].plot(epochs, [h["train_loss"] for h in history], label="Train Loss", color="#1f77b4", lw=2)
    axes[0].plot(epochs, [h["test_loss"] for h in history], label="Test Loss", color="#d62728", lw=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].set_title(f"{name}: Loss Curves ({params:,} params)")
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend()

    axes[1].plot(epochs, [h["train_acc"] for h in history], label="Train Accuracy (%)", color="#1f77b4", lw=2)
    axes[1].plot(epochs, [h["test_acc"] for h in history], label=f"Test Accuracy (Best: {best_acc:.2f}%)", color="#2ca02c", lw=2)
    axes[1].axhline(90.0, color="gray", linestyle=":", label="90% Target Threshold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title(f"{name}: Accuracy Progression")
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend()

    fig.tight_layout()
    save_path = out_dir / f"{name}_training_curves.png"
    fig.savefig(save_path, dpi=160)
    plt.close(fig)
    return save_path


def plot_per_class_accuracy(
    per_class_acc: Dict[str, float], overall_acc: float, model_name: str, out_dir: Path
) -> Path:
    """Generates a bar chart of per-class CIFAR-10 test accuracies."""
    out_dir.mkdir(parents=True, exist_ok=True)
    classes = list(per_class_acc.keys())
    accs = list(per_class_acc.values())

    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    bars = ax.bar(classes, accs, color="#2b5c8f")
    ax.axhline(overall_acc, color="#d95f02", linestyle="--", label=f"Overall Test Acc ({overall_acc:.2f}%)")
    ax.set_ylim(max(0.0, min(accs) - 10.0), 100.0)
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title(f"{model_name}: Per-Class CIFAR-10 Test Accuracy")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend()

    for b in bars:
        h = b.get_height()
        ax.annotate(
            f"{h:.1f}%",
            xy=(b.get_x() + b.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )

    fig.tight_layout()
    save_path = out_dir / f"{model_name}_per_class_accuracy.png"
    fig.savefig(save_path, dpi=160)
    plt.close(fig)
    return save_path


def plot_ablation_comparison(ablation_rows: List[Dict[str, Any]], out_dir: Path) -> Path:
    """Generates a bar chart comparing test accuracy and parameter counts across ablation models."""
    out_dir.mkdir(parents=True, exist_ok=True)
    names = [r["model"] for r in ablation_rows]
    std_accs = [r["best_test_acc"] for r in ablation_rows]
    tta_accs = [r["best_test_acc_tta"] for r in ablation_rows]

    x = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    b1 = ax.bar(x - width / 2, std_accs, width, label="Standard Test Acc (%)", color="#2b5c8f")
    b2 = ax.bar(x + width / 2, tta_accs, width, label="TTA Test Acc (%)", color="#2ca02c")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=12)
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Architectural Ablation Across Sub-5M Parameter ResNet Designs")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend()

    for b in list(b1) + list(b2):
        h = b.get_height()
        ax.annotate(
            f"{h:.2f}%",
            xy=(b.get_x() + b.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    save_path = out_dir / "ablation_comparison.png"
    fig.savefig(save_path, dpi=160)
    plt.close(fig)
    return save_path
