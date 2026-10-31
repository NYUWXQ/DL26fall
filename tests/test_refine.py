"""Unit tests for the inference-configuration comparison helpers.
推理配置对比辅助函数的单元测试。

Run / 运行：
    .venv/bin/python -m pytest tests/test_refine.py -q
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from refine import (  # noqa: E402
    compare_to_baseline,
    describe,
    ece,
    fit_temperature,
    fmt_row,
    mcnemar_exact,
    nll,
    wilson_ci,
)


# --------------------------------------------------------------------------
# wilson_ci / Wilson 区间
# --------------------------------------------------------------------------
def test_wilson_ci_brackets_the_point_estimate():
    """The interval must contain the observed proportion. / 区间必须包含观测比例。"""
    for k, n in [(0, 100), (50, 100), (97, 100), (100, 100), (1, 2500)]:
        lo, hi = wilson_ci(k, n)
        assert 0.0 <= lo <= k / n <= hi <= 1.0


def test_wilson_ci_is_wider_for_small_samples():
    """Smaller samples must give wider intervals. / 样本越小区间越宽。"""
    lo_small, hi_small = wilson_ci(50, 100)
    lo_big, hi_big = wilson_ci(5000, 10000)
    assert (hi_small - lo_small) > (hi_big - lo_big)


def test_wilson_ci_handles_zero_samples():
    """Empty sample must not crash. / 空样本不得崩溃。"""
    assert wilson_ci(0, 0) == (0.0, 0.0)


# --------------------------------------------------------------------------
# ece / 标定误差
# --------------------------------------------------------------------------
def test_ece_is_zero_for_perfectly_calibrated_perfect_confidence():
    """100% confidence with 100% accuracy gives zero ECE. / 置信度与准确率同为 100% 时 ECE 为 0。"""
    probs = np.zeros((10, 10))
    probs[:, 0] = 1.0
    y = np.zeros(10, dtype=np.int64)
    assert ece(probs, y) < 1e-9


def test_ece_is_one_for_maximally_overconfident_wrong_predictions():
    """Certain and wrong on every sample gives ECE = 1. / 全错且极度自信时 ECE = 1。"""
    probs = np.zeros((10, 10))
    probs[:, 0] = 1.0
    y = np.ones(10, dtype=np.int64)  # class 1, but the model is sure it is class 0
    assert ece(probs, y) == pytest.approx(1.0, abs=1e-6)


def test_ece_grows_as_confidence_outruns_accuracy():
    """More confidence with the same accuracy must not lower ECE. / 同样的准确率下，更自信应带来更高 ECE。"""
    y = np.array([0, 1, 0, 1], dtype=np.int64)
    calm = np.full((4, 2), 0.5)
    bold = np.full((4, 2), 0.5)
    calm[:, 0] = 0.55
    calm[:, 1] = 0.45
    bold[:, 0] = 0.99
    bold[:, 1] = 0.01
    assert ece(bold, y) > ece(calm, y)


# --------------------------------------------------------------------------
# nll / 负对数似然
# --------------------------------------------------------------------------
def test_nll_matches_manual_computation():
    """NLL must equal the mean of -log p(true). / NLL 必须等于 -log p(真值) 的均值。"""
    probs = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
    y = np.array([0, 1], dtype=np.int64)
    expected = -np.log(np.array([0.7, 0.8])).mean()
    assert nll(probs, y) == pytest.approx(expected)


def test_nll_is_finite_for_zero_probability():
    """A zero probability must be clipped, not produce -inf. / 概率为 0 需裁剪，不得产生 -inf。"""
    probs = np.array([[1.0, 0.0, 0.0]])
    y = np.array([2], dtype=np.int64)
    val = nll(probs, y)
    assert np.isfinite(val)


def test_nll_prefers_the_confident_correct_predictor():
    """A peaked correct distribution must beat a flat one. / 尖锐且正确的预测应优于平坦预测。"""
    y = np.array([0, 1], dtype=np.int64)
    peaked = np.array([[0.99, 0.005, 0.005], [0.005, 0.99, 0.005]])
    flat = np.full((2, 3), 1 / 3)
    assert nll(peaked, y) < nll(flat, y)


# --------------------------------------------------------------------------
# mcnemar_exact / McNemar 精确检验
# --------------------------------------------------------------------------
def test_mcnemar_identical_predictions_give_p_one():
    """No discordant pairs means no evidence of a difference. / 无不一致配对即无差异证据。"""
    assert mcnemar_exact(0, 0) == 1.0


def test_mcnemar_strongly_significant_for_lopsided_pairs():
    """10 vs 0 discordant pairs must be significant. / 10 比 0 的不一致配对必须显著。"""
    p = mcnemar_exact(10, 0)
    assert p == pytest.approx(2 * (0.5 ** 10), rel=1e-9)
    assert p < 0.01


def test_mcnemar_is_symmetric_in_b_and_c():
    """Swapping the two arms must not change the p-value. / 交换两臂不得改变 p 值。"""
    assert mcnemar_exact(7, 3) == pytest.approx(mcnemar_exact(3, 7))


def test_mcnemar_is_one_for_balanced_pairs():
    """A perfectly balanced split carries no evidence. / 完全均衡则无证据。"""
    assert mcnemar_exact(5, 5) == pytest.approx(1.0)


# --------------------------------------------------------------------------
# fit_temperature / 温度标定
# --------------------------------------------------------------------------
def test_fit_temperature_is_bounded_and_plausible():
    """The fitted temperature must stay inside the documented bounds.
    拟合出的温度必须落在文档规定的区间内。"""
    rng = np.random.RandomState(0)
    n = 4000
    y = rng.randint(0, 10, n)
    probs = rng.rand(n, 10)
    probs[np.arange(n), y] += 3.0  # a confident (over-confident) predictor
    probs /= probs.sum(1, keepdims=True)
    t = fit_temperature(probs, y)
    assert 0.05 <= t <= 20.0


def test_fit_temperature_flattens_overconfident_predictions():
    """A predictor that is wrong 40% of the time yet always 99% confident
    must be given T > 1 (which flattens it).
    一个「错 40% 但永远 99% 自信」的预测器必须得到 T > 1（即被压平）。"""
    rng = np.random.RandomState(1)
    n = 2000
    y = rng.randint(0, 10, n)
    wrong = rng.rand(n) < 0.4
    target = np.where(wrong, (y + 1) % 10, y)
    probs = np.full((n, 10), 0.01)
    probs[np.arange(n), target] = 0.99
    probs /= probs.sum(1, keepdims=True)
    t = fit_temperature(probs, y)
    assert t > 1.0


def test_fit_temperature_sharpens_underconfident_predictions():
    """A predictor that is always right but only 62% confident must be
    given T < 1 (which sharpens it).
    一个「永远正确但只有 62% 自信」的预测器必须得到 T < 1（即被锐化）。"""
    rng = np.random.RandomState(1)
    n = 2000
    y = rng.randint(0, 10, n)
    probs = rng.rand(n, 10)
    probs[np.arange(n), y] += 5.0
    probs /= probs.sum(1, keepdims=True)
    t = fit_temperature(probs, y)
    assert t < 1.0


def test_fit_temperature_reduces_nll():
    """The fitted temperature must actually lower the NLL on its fitting set.
    拟合出的温度必须真的降低拟合集上的 NLL。"""
    rng = np.random.RandomState(4)
    n = 3000
    y = rng.randint(0, 10, n)
    wrong = rng.rand(n) < 0.35
    target = np.where(wrong, (y + 1) % 10, y)
    probs = np.full((n, 10), 0.01)
    probs[np.arange(n), target] = 0.99
    probs /= probs.sum(1, keepdims=True)
    t = fit_temperature(probs, y)
    logits = np.log(np.clip(probs, 1e-12, 1.0)) / t
    scaled = np.exp(logits - logits.max(1, keepdims=True))
    scaled /= scaled.sum(1, keepdims=True)
    assert nll(scaled, y) < nll(probs, y)


def test_fit_temperature_respects_its_bounds():
    """The fitted temperature must stay inside [t_min, t_max].
    拟合出的温度必须落在 [t_min, t_max] 内。"""
    rng = np.random.RandomState(3)
    n = 500
    probs = rng.rand(n, 10)
    probs /= probs.sum(1, keepdims=True)
    y = rng.randint(0, 10, n)
    t = fit_temperature(probs, y, t_min=0.5, t_max=2.0)
    assert 0.5 <= t <= 2.0


def test_fit_temperature_never_returns_non_positive():
    """The optimiser must not fall out of the valid domain. / 优化器不得跳出有效域。"""
    rng = np.random.RandomState(2)
    probs = rng.rand(200, 10)
    probs /= probs.sum(1, keepdims=True)
    y = rng.randint(0, 10, 200)
    t = fit_temperature(probs, y)
    assert t > 0 and np.isfinite(t)


# --------------------------------------------------------------------------
# describe / compare_to_baseline / fmt_row
# --------------------------------------------------------------------------
def _toy_rows():
    """Build two toy rows sharing a label vector. / 构造两行玩具数据，共享标签。

    Row 0 of probs_b is deliberately wrong, so b and a disagree on exactly
    one of four samples. / probs_b 的第 0 行故意写错，因此 b 与 a 恰好在 4 个样本中
    有 1 个不一致。
    """
    y = np.array([0, 1, 2, 3], dtype=np.int64)
    probs_a = np.zeros((4, 4))
    probs_a[np.arange(4), y] = 1.0
    probs_b = np.zeros((4, 4))
    probs_b[np.arange(4), y] = 0.6
    probs_b[0, 0] = 0.4  # row 0 now favours class 1 -> one wrong prediction
    probs_b[0, 1] = 0.6
    a = describe("a.plain", "ema", "plain", "none", probs_a, y)
    b = describe("b.plain", "raw", "plain", "none", probs_b, y)
    return a, b, y


def test_describe_reports_accuracy_and_ci():
    """Accuracy must match the predictions and the CI must bracket it. / 准确率需与预测一致，区间需包含它。"""
    a, _b, y = _toy_rows()
    assert a["acc"] == 1.0
    assert a["acc_correct"] == 4
    assert a["n"] == len(y)
    assert a["acc_ci95"][0] <= 1.0 <= a["acc_ci95"][1]


def test_compare_to_baseline_fills_disagreement_and_p():
    """The disagreement column must be filled in against the baseline. / 分歧列必须相对基线填好。"""
    a, b, y = _toy_rows()
    compare_to_baseline([a, b], a, y)
    assert a["disagree_vs_plain"] == 0.0
    assert a["mcnemar_p_vs_plain"] == 1.0
    assert b["disagree_vs_plain"] == pytest.approx(0.25)
    assert b["mcnemar_p_vs_plain"] is not None
    assert 0.0 < b["mcnemar_p_vs_plain"] <= 1.0


def test_fmt_row_renders_every_column():
    """The markdown row must have one cell per column and start/end with a pipe.
    markdown 行必须每列一个单元格，且以竖线开头结尾。"""
    a, b, y = _toy_rows()
    compare_to_baseline([a, b], a, y)
    row = fmt_row(b, a["acc"])
    assert row.startswith("|") and row.endswith("|")
    cells = row.split("|")[1:-1]
    assert len(cells) == 12  # config, weights, tta, bn_recal, bn_imgs, acc, ci, delta, nll, ece, disagree, p
    assert "b.plain" in row


def test_fmt_row_shows_bn_image_count():
    """The BN image-count cell must show the number of images used.
    BN 图像张数列必须显示实际使用的张数。"""
    a, b, y = _toy_rows()
    compare_to_baseline([a, b], a, y)
    b["bn_images"] = 2000
    row = fmt_row(b, a["acc"])
    assert "2000" in row.split("|")[5]


def test_fmt_row_handles_missing_comparison_columns():
    """Unfilled comparison columns must render as a dash, not crash. / 未填的比较列应渲染为破折号。"""
    a, _b, _y = _toy_rows()
    row = fmt_row(a, a["acc"])
    assert "—" in row


# --------------------------------------------------------------------------
# BN recalibration helper / BN 重校准
# --------------------------------------------------------------------------
def test_bn_recalibration_updates_running_stats():
    """After re-estimation the running stats must no longer be the initial ones.
    重估后运行统计量不应再是初始值。"""
    from refine import bn_recalibrate

    model = torch.nn.Sequential(torch.nn.Conv2d(3, 8, 3, padding=1), torch.nn.BatchNorm2d(8))
    before = model[1].running_mean.clone()
    x = torch.randn(64, 3, 32, 32) * 3.0 + 1.0
    loader = [(x, torch.zeros(64, dtype=torch.long))]
    bn_recalibrate(model, loader, torch.device("cpu"))
    assert not torch.allclose(before, model[1].running_mean)


def test_bn_recalibration_leaves_the_model_in_eval_mode():
    """The model must be back in eval mode after re-estimation. / 重估后模型必须回到 eval 模式。"""
    from refine import bn_recalibrate

    model = torch.nn.Sequential(torch.nn.Conv2d(3, 8, 3, padding=1), torch.nn.BatchNorm2d(8))
    x = torch.randn(16, 3, 32, 32)
    loader = [(x, torch.zeros(16, dtype=torch.long))]
    bn_recalibrate(model, loader, torch.device("cpu"))
    assert not model.training


# --------------------------------------------------------------------------
# checkpoint snapshot / 检查点快照
# --------------------------------------------------------------------------
def test_snapshot_ckpt_returns_a_readable_copy(tmp_path):
    """The snapshot must be loadable and must not be the original file. / 快照必须可加载且不是原文件。"""
    from refine import snapshot_ckpt

    src = tmp_path / "last.pt"
    torch.save({"model": {"a": torch.zeros(2)}, "args": {"model": "preact16x4"}}, src)
    snap = snapshot_ckpt(str(src))
    try:
        ck = torch.load(snap, map_location="cpu", weights_only=False)
        assert ck["args"]["model"] == "preact16x4"
        assert os.path.abspath(snap) != os.path.abspath(str(src))
    finally:
        os.unlink(snap)


def test_snapshot_ckpt_does_not_modify_the_source(tmp_path):
    """Reading must leave the source checkpoint byte-identical. / 读取不得改动源检查点。"""
    from refine import snapshot_ckpt

    src = tmp_path / "last.pt"
    torch.save({"model": {"a": torch.zeros(2)}, "args": {}}, src)
    digest_before = open(src, "rb").read()
    snap = snapshot_ckpt(str(src))
    os.unlink(snap)
    assert open(src, "rb").read() == digest_before


def test_snapshot_ckpt_retries_then_fails_loudly(tmp_path):
    """A corrupt file must raise, not silently return garbage. / 损坏文件必须报错，不得静默返回垃圾。"""
    from refine import snapshot_ckpt

    src = tmp_path / "broken.pt"
    src.write_bytes(b"this is not a torch checkpoint")
    with pytest.raises(RuntimeError):
        snapshot_ckpt(str(src), tries=2)


# --------------------------------------------------------------------------
# the pinned validation split / 固定的验证集划分
# --------------------------------------------------------------------------
def test_validation_split_is_pinned_to_seed_zero():
    """refine.py must evaluate on the SAME 2500 images as train.py, whatever
    --seed is given. train.py hard-codes load_train(seed=0) so that ensemble
    members stay comparable; refine.py must do the same.
    refine.py 必须在与 train.py 相同的 2500 张图上评估，无论 --seed 是多少。
    train.py 已硬编码 load_train(seed=0) 以保证集成成员可比；refine.py 必须一致。"""
    from data import load_train

    root = os.path.join(ROOT)
    if not os.path.exists(os.path.join(root, "train_labels.csv")):
        pytest.skip("training data not present / 训练数据不存在")

    _tx, _ty, va_x_a, va_y_a = load_train(root, seed=0, n_val=2500)
    # A different seed would give a DIFFERENT split if refine followed --seed.
    _tx2, _ty2, va_x_b, va_y_b = load_train(root, seed=1, n_val=2500)
    assert not np.array_equal(va_y_a, va_y_b), "the two splits should differ / 两种划分应不同"

    # The pinned split is the seed=0 one.
    # 固定的划分就是 seed=0 那一份。
    assert va_y_a.shape == (2500,)
    assert len(np.unique(va_y_a)) == 10


def test_refine_source_pins_the_split_seed():
    """Guard against someone re-tying the split to --seed.
    防止有人把划分重新绑回 --seed。"""
    src = open(os.path.join(ROOT, "src", "refine.py")).read()
    assert "split_seed = 0" in src, "refine.py must pin the split seed to 0"
    assert "load_train(root, seed=split_seed" in src, \
        "refine.py must load the split with the pinned seed"
    assert "load_train(root, seed=args.seed" not in src, \
        "refine.py must NOT follow --seed for the validation split"


# --------------------------------------------------------------------------
# pixel_shift_report / 像素偏移报告
# --------------------------------------------------------------------------
def test_pixel_shift_report_detects_a_brightness_difference():
    """A brightness difference between the two sets must be reported and must
    be large relative to the class-mix sampling error.
    两套图片的亮度差必须被报告，且相对类别抽样误差显著。"""
    from refine import pixel_shift_report

    rng = np.random.RandomState(0)
    n = 400
    y = rng.randint(0, 10, n)
    tr = rng.randint(0, 256, (n, 3, 8, 8), dtype=np.uint8)
    te = np.clip(tr.astype(np.int64) + 20, 0, 255).astype(np.uint8)  # brighter copy
    rep = pixel_shift_report(tr, y, te)
    assert rep["mean_gap"] > 0.05
    assert rep["gap_over_sampling_error"] > 5
    assert rep["mean_gap_levels"] == pytest.approx(rep["mean_gap"] * 255, rel=1e-6)


def test_pixel_shift_report_is_quiet_for_identical_sets():
    """Identical sets must show no gap. / 完全相同的两套图不得报告偏移。"""
    from refine import pixel_shift_report

    rng = np.random.RandomState(1)
    n = 200
    y = rng.randint(0, 10, n)
    x = rng.randint(0, 256, (n, 3, 8, 8), dtype=np.uint8)
    rep = pixel_shift_report(x, y, x.copy())
    assert rep["mean_gap"] == pytest.approx(0.0, abs=1e-9)
    assert rep["std_gap"] == pytest.approx(0.0, abs=1e-9)
