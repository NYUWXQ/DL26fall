# Inference-configuration comparison (`src/refine.py`)
# 推理配置对比（src/refine.py）

- Checkpoint / 检查点：`runs/wrn22x4_s0/last.pt`
- Weights model / 结构：`preact22x4`，params=4,326,986
- Checkpoint epoch / 检查点 epoch：37
- Validation split / 验证集：seed=0, n_val=2500 (identical to `train.py`) / 与 train.py 一致
- Generated / 生成时间：2026-10-31 18:29:16 EDT
- Device / 设备：mps

| Config | Weights | TTA | BN recal | BN imgs | Acc | 95% CI | Δacc vs baseline | NLL | ECE | Disagree vs baseline | McNemar p |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ema.plain | ema | plain | none | — | 0.1832 | [0.1685, 0.1988] | -0.6404 | 2.2483 | 0.0323 | 80.40% | 0.0000 |
| ema.flip | ema | flip | none | — | 0.1816 | [0.1670, 0.1972] | -0.6420 | 2.2481 | 0.0337 | 80.52% | 0.0000 |
| ema.translate | ema | translate | none | — | 0.1804 | [0.1658, 0.1960] | -0.6432 | 2.2514 | 0.0365 | 80.64% | 0.0000 |
| ema.translate_flip | ema | translate_flip | none | — | 0.1816 | [0.1670, 0.1972] | -0.6420 | 2.2513 | 0.0338 | 80.48% | 0.0000 |
| ema.plain+bn[train] | ema | plain | train | 10000 | 0.8572 | [0.8429, 0.8704] | +0.0336 | 0.4600 | 0.0700 | 19.08% | 0.0000 |
| ema.translate_flip+bn[train] | ema | translate_flip | train | 10000 | 0.8580 | [0.8438, 0.8711] | +0.0344 | 0.4719 | 0.0927 | 19.24% | 0.0000 |
| ema.plain+bn[train_matched] | ema | plain | train_matched | 2000 | 0.8604 | [0.8463, 0.8734] | +0.0368 | 0.4575 | 0.0721 | 18.84% | 0.0000 |
| ema.translate_flip+bn[train_matched] | ema | translate_flip | train_matched | 2000 | 0.8600 | [0.8458, 0.8731] | +0.0364 | 0.4694 | 0.0932 | 19.20% | 0.0000 |
| ema.plain+bn[test] | ema | plain | test | 2000 | 0.8632 | [0.8492, 0.8761] | +0.0396 | 0.4491 | 0.0761 | 18.80% | 0.0000 |
| ema.translate_flip+bn[test] | ema | translate_flip | test | 2000 | 0.8648 | [0.8508, 0.8776] | +0.0412 | 0.4627 | 0.0975 | 19.28% | 0.0000 |
| raw.plain | raw | plain | none | — | 0.8236 | [0.8082, 0.8380] | +0.0000 | 0.7035 | 0.1726 | 0.00% | 1.0000 |
| raw.flip | raw | flip | none | — | 0.8236 | [0.8082, 0.8380] | +0.0000 | 0.6876 | 0.1830 | 6.60% | 1.0000 |
| raw.translate | raw | translate | none | — | 0.8240 | [0.8086, 0.8384] | +0.0004 | 0.7143 | 0.1882 | 6.28% | 1.0000 |
| raw.translate_flip | raw | translate_flip | none | — | 0.8224 | [0.8069, 0.8369] | -0.0012 | 0.7001 | 0.1944 | 9.32% | 0.8838 |
| raw.plain+bn[train] | raw | plain | train | 10000 | 0.8564 | [0.8421, 0.8696] | +0.0328 | 0.5501 | 0.1534 | 15.92% | 0.0000 |
| raw.translate_flip+bn[train] | raw | translate_flip | train | 10000 | 0.8668 | [0.8529, 0.8796] | +0.0432 | 0.5609 | 0.1853 | 16.72% | 0.0000 |
| raw.plain+bn[train_matched] | raw | plain | train_matched | 2000 | 0.8576 | [0.8434, 0.8708] | +0.0340 | 0.5479 | 0.1525 | 16.20% | 0.0000 |
| raw.translate_flip+bn[train_matched] | raw | translate_flip | train_matched | 2000 | 0.8668 | [0.8529, 0.8796] | +0.0432 | 0.5589 | 0.1841 | 16.92% | 0.0000 |
| raw.plain+bn[test] | raw | plain | test | 2000 | 0.8596 | [0.8454, 0.8727] | +0.0360 | 0.6203 | 0.2096 | 15.56% | 0.0000 |
| raw.translate_flip+bn[test] | raw | translate_flip | test | 2000 | 0.8688 | [0.8550, 0.8815] | +0.0452 | 0.6312 | 0.2403 | 16.60% | 0.0000 |

## Best configuration / 最优配置

**raw.translate_flip+bn[test]** — weights=raw, TTA=translate_flip, BN recal=test, acc=0.8688 (95% CI [0.8550, 0.8815]), NLL=0.6312, ECE=0.2403

Baseline / 基线：`raw.plain` acc=0.8236. Δacc = +0.0452.

## Domain-shift control / 域偏移对照

Same image count for train and test sources, so any extra gain from the test images is attributable to the domain, not to the sample volume.
train 与 test 来源的图像张数相同，因此 test 上多出来的收益只能归因于「域」，而非样本量。

- weights=ema: baseline 0.1832 -> train_matched (2000 imgs) 0.8604 (+0.6772) -> test (2000 imgs) 0.8632 (+0.6800)
- weights=raw: baseline 0.8236 -> train_matched (2000 imgs) 0.8576 (+0.0340) -> test (2000 imgs) 0.8596 (+0.0360)

## Pixel-level train/test comparison / 像素级 train/test 对比

- Mean pixel value: train 0.4734 vs test 0.4907 (gap +0.0173 = +4.41 / 255 levels)
  像素均值：train 0.4734 vs test 0.4907 （差 +0.0173，即 +4.41 / 255 级）
- Std pixel value: train 0.2516 vs test 0.2683 (gap +0.0167)
  像素标准差：train 0.2516 vs test 0.2683 （差 +0.0167）
- Class-mix sampling error on the mean: 0.0003; the observed gap is 63.1x that error.
  按类别比例抽样造成的均值误差约 0.0003；实测差值是该误差的 63.1 倍。

## Temperature-scaling diagnostic / 温度标定诊断

- Fitted on / 拟合集：val first half，evaluated on / 评估集：val second half
- T = 0.5096
- Acc 0.8704 -> 0.8704；NLL 0.6264 -> 0.3914

## Notes / 说明

- The McNemar column is an exact two-sided test on the discordant pairs against the baseline `raw.plain`; p > 0.05 means the difference is not distinguishable from noise on this validation set.
  McNemar 列为与基线 raw.plain 的精确二项检验；p > 0.05 表示该差异在验证集上无法与噪声区分。
- `bn[test]` is transductive: BN statistics are re-estimated on the unlabeled test images. This is legal in a Kaggle competition but must be disclosed in the report.
  bn[test] 属于转导式适配：在无标签测试图上重估 BN 统计量。Kaggle 竞赛允许，但报告必须披露。
- `bn[val]` is leaky (it uses the validation labels' images) and is only a diagnostic ceiling.
  bn[val] 使用了验证集，仅作诊断上限，不可作为最终方案。
- `bn[train_matched]` uses exactly as many images as the test set, so the comparison against `bn[test]` isolates the domain effect from the sample-volume effect.
  bn[train_matched] 与测试集张数完全相同，因此与 bn[test] 的对比能把「域效应」与「样本量效应」分开。
- All rows come from ONE snapshot of the checkpoint, so every weight variant is measured at the same training epoch. / 所有行都来自同一次 checkpoint 快照，因此各权重变体处于同一训练 epoch。
- No number here is copied from anywhere else; all are measured by this script.
  此处没有抄来的数字，全部由本脚本实测。
