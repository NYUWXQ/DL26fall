"""
From-Scratch Training, Evaluation, Test-Time Augmentation (TTA), and Checkpointing.
"""

import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Any, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.model import CustomResNet, audit_model_parameters


@torch.no_grad()
def evaluate_network(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    apply_tta: bool = False,
) -> Tuple[float, float]:
    """
    Evaluates model loss and top-1 accuracy (%) on a DataLoader.
    If apply_tta=True, averages logits across original and horizontally mirrored inputs.
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        if apply_tta:
            flipped_logits = model(torch.flip(images, dims=[3]))
            logits = 0.5 * (logits + flipped_logits)

        loss = loss_fn(logits, labels)
        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    return total_loss / total_samples, 100.0 * total_correct / total_samples


@torch.no_grad()
def evaluate_per_class_accuracy(
    model: nn.Module,
    loader: DataLoader,
    class_names: List[str],
    device: torch.device,
) -> Dict[str, float]:
    """Computes per-class test accuracy (%) using horizontal-flip TTA."""
    model.eval()
    n_classes = len(class_names)
    hits = [0] * n_classes
    counts = [0] * n_classes

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = 0.5 * (model(images) + model(torch.flip(images, dims=[3])))
        preds = logits.argmax(dim=1)
        for y_true, y_pred in zip(labels, preds):
            c = int(y_true.item())
            counts[c] += 1
            if int(y_pred.item()) == c:
                hits[c] += 1

    return {
        class_names[i]: round(100.0 * hits[i] / max(1, counts[i]), 2)
        for i in range(n_classes)
    }


def train_and_evaluate(
    model: CustomResNet,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int,
    peak_lr: float,
    weight_decay: float,
    label_smoothing: float,
    output_dir: Path,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Trains CustomResNet from random initialization using Nesterov SGD, decoupled
    weight decay, linear warmup + cosine annealing schedule, and mixed precision.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    param_count = audit_model_parameters(model, max_budget=5_000_000, run_summary=False)

    train_loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    eval_loss_fn = nn.CrossEntropyLoss()

    # Apply weight decay only to >=2D convolutional/linear weights (exclude BN & biases)
    matrix_params = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
    vector_params = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
    optimizer = optim.SGD(
        [
            {"params": matrix_params, "weight_decay": weight_decay},
            {"params": vector_params, "weight_decay": 0.0},
        ],
        lr=peak_lr,
        momentum=0.9,
        nesterov=True,
    )

    warmup_epochs = min(5, max(1, epochs // 10))

    def lr_schedule_fn(epoch_idx: int) -> float:
        if epoch_idx < warmup_epochs:
            return float(epoch_idx + 1) / float(warmup_epochs)
        progress = float(epoch_idx - warmup_epochs) / float(max(1, epochs - warmup_epochs))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_schedule_fn)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history = []
    best_test_acc = 0.0
    best_ckpt_path = output_dir / f"{model.spec.name}_best.pth"
    t0 = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0
        current_lr = optimizer.param_groups[0]["lr"]

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = model(images)
                    loss = train_loss_fn(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(images)
                loss = train_loss_fn(logits, labels)
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * labels.size(0)
            running_correct += (logits.argmax(dim=1) == labels).sum().item()
            running_total += labels.size(0)

        scheduler.step()
        train_loss = running_loss / running_total
        train_acc = 100.0 * running_correct / running_total
        test_loss, test_acc = evaluate_network(model, test_loader, eval_loss_fn, device, apply_tta=False)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "spec": model.spec.to_dict(),
                    "epoch": epoch,
                    "best_test_acc": best_test_acc,
                    "trainable_parameters": param_count,
                },
                best_ckpt_path,
            )

        history.append({
            "epoch": epoch,
            "lr": round(current_lr, 6),
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 2),
            "test_loss": round(test_loss, 4),
            "test_acc": round(test_acc, 2),
            "best_test_acc": round(best_test_acc, 2),
        })

        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(
                f"[{model.spec.name}] Epoch {epoch:03d}/{epochs:03d} | LR: {current_lr:.4f} | "
                f"Train Loss: {train_loss:.4f} ({train_acc:.2f}%) | "
                f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}% (Best: {best_test_acc:.2f}%)"
            )

    elapsed_sec = time.perf_counter() - t0

    # Reload best checkpoint and evaluate both standard and TTA test accuracy
    ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    _, final_std_acc = evaluate_network(model, test_loader, eval_loss_fn, device, apply_tta=False)
    _, final_tta_acc = evaluate_network(model, test_loader, eval_loss_fn, device, apply_tta=True)

    print("\n" + "=" * 76)
    print(f"FINAL RESULTS — {model.spec.name}")
    print(f"  Total Trainable Parameters : {param_count:,} (<= 5,000,000 limit)")
    print(f"  Final Test Accuracy (Std)  : {final_std_acc:.2f}%")
    print(f"  Final Test Accuracy (TTA)  : {final_tta_acc:.2f}%")
    print(f"  Total Training Time        : {elapsed_sec:.1f}s ({elapsed_sec / 60.0:.2f} min)")
    print("=" * 76 + "\n")

    return {
        "spec": model.spec.to_dict(),
        "parameters": param_count,
        "best_test_acc": round(final_std_acc, 2),
        "best_test_acc_tta": round(final_tta_acc, 2),
        "training_seconds": round(elapsed_sec, 2),
        "history": history,
        "best_checkpoint": str(best_ckpt_path),
    }
