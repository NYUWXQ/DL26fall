"""Unit tests for the submission path (`src/predict.py`).
提交路径（src/predict.py）的单元测试。

These cover two real bugs that were found and fixed:
1. `.double()` on a MPS tensor raised TypeError -> the whole submission crashed.
2. `--weights auto` blindly preferred the EMA shadow, which early in training is
   still mostly the random initialisation.
这两个真实缺陷的回归测试：
1. 对 MPS 张量调用 .double() 会抛 TypeError，导致整个提交崩溃。
2. --weights auto 盲目偏好 EMA 影子权重，而它在训练早期仍mostly是随机初值。

Run / 运行：
    .venv/bin/python -m pytest tests/test_predict.py -q
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from predict import ema_converged, load_ckpt  # noqa: E402


# --------------------------------------------------------------------------
# ema_converged / EMA 收敛判定
# --------------------------------------------------------------------------
def _write_manifest(root: str, n_rows: int) -> None:
    """Write a fake train_labels.csv with n_rows rows. / 写一个 n_rows 行的假清单。"""
    with open(os.path.join(root, "train_labels.csv"), "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["filename", "label"])
        for i in range(n_rows):
            wr.writerow([f"{i}.png", i % 10])


def test_ema_is_not_converged_early_in_training(tmp_path):
    """With decay=0.9995 the EMA is still far from converged at epoch 20.
    decay=0.9995 时，第 20 个 epoch 的 EMA 远未收敛。"""
    _write_manifest(str(tmp_path), 50000)
    ck = {"epoch": 20, "args": {"ema_decay": 0.9995, "batch_size": 512}}
    converged, residual = ema_converged(ck, str(tmp_path))
    assert converged is False
    assert residual > 0.1


def test_ema_is_converged_after_enough_epochs(tmp_path):
    """After 300 epochs the residual initialisation weight is negligible.
    300 个 epoch 后，残差初值权重可忽略。"""
    _write_manifest(str(tmp_path), 50000)
    ck = {"epoch": 300, "args": {"ema_decay": 0.9995, "batch_size": 512}}
    converged, residual = ema_converged(ck, str(tmp_path))
    assert converged is True
    assert residual < 0.01


def test_ema_converged_is_safe_when_metadata_is_missing(tmp_path):
    """Missing epoch or batch size must not crash, and must not claim convergence.
    缺少 epoch 或 batch_size 不得崩溃，也不得声称已收敛。"""
    _write_manifest(str(tmp_path), 50000)
    ck = {"epoch": 0, "args": {"ema_decay": 0.9995}}
    converged, residual = ema_converged(ck, str(tmp_path))
    assert converged is False
    assert residual == 1.0


def test_ema_converged_is_safe_with_missing_manifest(tmp_path):
    """A missing manifest must not crash. / 清单缺失不得崩溃。"""
    ck = {"epoch": 100, "args": {"ema_decay": 0.9995, "batch_size": 512}}
    converged, residual = ema_converged(ck, str(tmp_path))
    assert converged is False


# --------------------------------------------------------------------------
# load_ckpt weight selection / 权重选择
# --------------------------------------------------------------------------
def _make_ckpt(path: str, epoch: int, ema_decay: float = 0.9995,
               batch_size: int = 512) -> None:
    """Save a checkpoint whose EMA and raw weights are clearly distinguishable.
    保存一个 EMA 与 raw 权重明显可区分的 checkpoint。"""
    from models import build_model

    model = build_model("preact16x4")
    ema = build_model("preact16x4")
    with torch.no_grad():
        for p in ema.parameters():
            p.fill_(0.0)  # a clearly different (and useless) EMA shadow
    torch.save({"model": model.state_dict(), "ema": ema.state_dict(),
                "epoch": epoch,
                "args": {"model": "preact16x4", "ema_decay": ema_decay,
                         "batch_size": batch_size, "drop": 0.0}}, path)


def test_auto_picks_raw_while_the_ema_is_unconverged(tmp_path):
    """Early in training, auto must use the raw weights, not the EMA shadow.
    训练早期，auto 必须用 raw 权重，而不是 EMA 影子权重。"""
    _write_manifest(str(tmp_path), 50000)
    ckpt = str(tmp_path / "last.pt")
    _make_ckpt(ckpt, epoch=20)
    _model, meta = load_ckpt(ckpt, torch.device("cpu"), which="auto", root=str(tmp_path))
    assert meta["weights"] == "raw"


def test_auto_picks_ema_once_it_has_converged(tmp_path):
    """After enough epochs, auto must use the EMA shadow. / 收敛之后，auto 必须用 EMA。"""
    _write_manifest(str(tmp_path), 50000)
    ckpt = str(tmp_path / "last.pt")
    _make_ckpt(ckpt, epoch=300)
    _model, meta = load_ckpt(ckpt, torch.device("cpu"), which="auto", root=str(tmp_path))
    assert meta["weights"] == "ema"


def test_explicit_ema_is_always_honoured(tmp_path):
    """An explicit --weights ema must use the EMA even when unconverged.
    显式指定 ema 时，即使未收敛也必须使用 EMA。"""
    _write_manifest(str(tmp_path), 50000)
    ckpt = str(tmp_path / "last.pt")
    _make_ckpt(ckpt, epoch=5)
    _model, meta = load_ckpt(ckpt, torch.device("cpu"), which="ema", root=str(tmp_path))
    assert meta["weights"] == "ema"


def test_explicit_raw_is_always_honoured(tmp_path):
    """An explicit --weights raw must use the raw weights. / 显式指定 raw 必须用 raw。"""
    _write_manifest(str(tmp_path), 50000)
    ckpt = str(tmp_path / "last.pt")
    _make_ckpt(ckpt, epoch=300)
    _model, meta = load_ckpt(ckpt, torch.device("cpu"), which="raw", root=str(tmp_path))
    assert meta["weights"] == "raw"


# --------------------------------------------------------------------------
# the MPS float64 crash / MPS float64 崩溃回归
# --------------------------------------------------------------------------
@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS / 需要 MPS")
def test_probabilities_can_be_widened_to_float64_on_cpu():
    """The submission path must be able to reach float64 without touching MPS float64.
    提交路径必须能在不触碰 MPS float64 的前提下得到 float64。"""
    from predict import view_probs

    model = torch.nn.Linear(4, 3).to("mps")
    x = torch.randn(8, 4, device="mps")
    probs = view_probs(model, x, "plain")
    widened = probs.float().cpu().numpy().astype(np.float64)
    assert widened.dtype == np.float64
    assert np.isfinite(widened).all()


def test_probabilities_widened_on_cpu_match_the_direct_computation():
    """Widening on the CPU must not change the numbers. / 在 CPU 上扩展不得改变数值。"""
    from predict import view_probs

    torch.manual_seed(0)
    model = torch.nn.Linear(4, 3)
    x = torch.randn(8, 4)
    probs = view_probs(model, x, "plain")
    widened = probs.float().cpu().numpy().astype(np.float64)
    with torch.no_grad():
        expected = model(x).float().softmax(1).numpy()
    assert np.allclose(widened, expected, atol=1e-6)
