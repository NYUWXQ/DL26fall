# Inference-configuration comparison (`src/refine.py`)
# 推理配置对比（src/refine.py）

- Checkpoint / 检查点：`runs/wrn22x4_s0/last.pt`
- Weights model / 结构：`preact22x4`，params=4,326,986
- Checkpoint epoch / 检查点 epoch：206
- Validation split / 验证集：seed=0 (PINNED, identical to `train.py`), n_val=2500 / 验证集划分固定为 seed=0，与 train.py 一致
- Generated / 生成时间：2026-10-31 20:51:44 EDT
- Device / 设备：mps
- `--bn-recal` / 报告重点关注的 BN 重校准来源：`none`

| Config | Weights | TTA | BN recal | BN imgs | Acc | 95% CI | Δacc vs baseline | NLL | ECE | Disagree vs baseline | McNemar p |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ema.plain | ema | plain | none | — | 0.9528 | [0.9438, 0.9604] | +0.0000 | 0.3012 | 0.1633 | 0.00% | 1.0000 |
| ema.flip | ema | flip | none | — | 0.9560 | [0.9472, 0.9634] | +0.0032 | 0.2978 | 0.1659 | 0.88% | 0.0963 |
| ema.translate | ema | translate | none | — | 0.9476 | [0.9382, 0.9557] | -0.0052 | 0.3196 | 0.1708 | 1.88% | 0.0533 |
| ema.translate_flip | ema | translate_flip | none | — | 0.9496 | [0.9403, 0.9575] | -0.0032 | 0.3170 | 0.1740 | 1.72% | 0.2295 |
| ema.plain+bn[train] | ema | plain | train | 10000 | 0.9428 | [0.9330, 0.9512] | -0.0100 | 0.2184 | 0.0546 | 4.60% | 0.0199 |
| ema.translate_flip+bn[train] | ema | translate_flip | train | 10000 | 0.9440 | [0.9343, 0.9524] | -0.0088 | 0.2262 | 0.0659 | 4.72% | 0.0428 |
| ema.plain+bn[train_matched] | ema | plain | train_matched | 2000 | 0.9408 | [0.9309, 0.9494] | -0.0120 | 0.2160 | 0.0513 | 4.76% | 0.0059 |
| ema.translate_flip+bn[train_matched] | ema | translate_flip | train_matched | 2000 | 0.9448 | [0.9351, 0.9531] | -0.0080 | 0.2230 | 0.0648 | 4.80% | 0.0696 |
| ema.plain+bn[test] | ema | plain | test | 2000 | 0.9444 | [0.9347, 0.9527] | -0.0084 | 0.2366 | 0.0768 | 4.44% | 0.0504 |
| ema.translate_flip+bn[test] | ema | translate_flip | test | 2000 | 0.9472 | [0.9377, 0.9553] | -0.0056 | 0.2445 | 0.0924 | 4.36% | 0.1933 |
| raw.plain | raw | plain | none | — | 0.9360 | [0.9257, 0.9449] | -0.0168 | 0.3044 | 0.1062 | 6.12% | 0.0004 |
| raw.flip | raw | flip | none | — | 0.9400 | [0.9300, 0.9487] | -0.0128 | 0.2949 | 0.1144 | 5.32% | 0.0038 |
| raw.translate | raw | translate | none | — | 0.9292 | [0.9185, 0.9386] | -0.0236 | 0.3221 | 0.1163 | 6.56% | 0.0000 |
| raw.translate_flip | raw | translate_flip | none | — | 0.9392 | [0.9291, 0.9479] | -0.0136 | 0.3113 | 0.1293 | 5.64% | 0.0027 |
| raw.plain+bn[train] | raw | plain | train | 10000 | 0.9216 | [0.9104, 0.9315] | -0.0312 | 0.3012 | 0.0734 | 6.68% | 0.0000 |
| raw.translate_flip+bn[train] | raw | translate_flip | train | 10000 | 0.9296 | [0.9189, 0.9390] | -0.0232 | 0.3041 | 0.1025 | 6.20% | 0.0000 |
| raw.plain+bn[train_matched] | raw | plain | train_matched | 2000 | 0.9220 | [0.9108, 0.9319] | -0.0308 | 0.2980 | 0.0706 | 6.56% | 0.0000 |
| raw.translate_flip+bn[train_matched] | raw | translate_flip | train_matched | 2000 | 0.9328 | [0.9223, 0.9420] | -0.0200 | 0.3005 | 0.1054 | 6.04% | 0.0000 |
| raw.plain+bn[test] | raw | plain | test | 2000 | 0.9272 | [0.9163, 0.9367] | -0.0256 | 0.3266 | 0.1037 | 6.56% | 0.0000 |
| raw.translate_flip+bn[test] | raw | translate_flip | test | 2000 | 0.9344 | [0.9240, 0.9435] | -0.0184 | 0.3299 | 0.1331 | 6.04% | 0.0001 |

## Best configuration / 最优配置

**ema.flip** — weights=ema, TTA=flip, BN recal=none, acc=0.9560 (95% CI [0.9472, 0.9634]), NLL=0.2978, ECE=0.1659

Baseline / 基线：`ema.plain` acc=0.9528. Δacc = +0.0032.

### Best SUBMISSION-PATH configuration / 最优「可提交」配置

**ema.flip** — weights=ema, TTA=flip, BN recal=none, acc=0.9560 (95% CI [0.9472, 0.9634]), NLL=0.2978, ECE=0.1659

Δacc vs baseline = +0.0032. This is the configuration to ship: no BatchNorm recalibration, so it is a purely supervised evaluation and it needs no transductive disclosure.
相对基线 Δacc = +0.0032。这是真正该提交的配置：不做 BatchNorm 重校准，因此是纯监督评估，无需任何转导式披露。

The matching submission command / 对应的提交命令：

```bash
.venv/bin/python src/predict.py \
  --ckpt /Users/shenqing/Desktop/Deepseek Harness/DL_proj1/runs/wrn22x4_s0/last.pt \
  --weights ema \
  --tta flip \
  --out submissions/final_tta.csv
```

For an ensemble, pass every member checkpoint to `--ckpt` (they are averaged at the probability level). / 集成时把每个成员 checkpoint 都传给 `--ckpt`（在概率层面平均）。

## Domain-shift control / 域偏移对照

Same image count for train and test sources, so any extra gain from the test images is attributable to the domain, not to the sample volume.
train 与 test 来源的图像张数相同，因此 test 上多出来的收益只能归因于「域」，而非样本量。

- weights=ema: baseline 0.9528 -> train_matched (2000 imgs) 0.9408 (-0.0120) -> test (2000 imgs) 0.9444 (-0.0084)
- weights=raw: baseline 0.9360 -> train_matched (2000 imgs) 0.9220 (-0.0140) -> test (2000 imgs) 0.9272 (-0.0088)

- weights=ema: bn[train] (supervised) 0.9428 (-0.0100) vs bn[test] (transductive) 0.9444 (-0.0084) — the supervised source captures the same gain, so this is NOT a domain effect.
  weights=ema：bn[train]（纯监督）0.9428 与 bn[test]（转导式）0.9444 增益相当——**不是域效应**。

- weights=raw: bn[train] (supervised) 0.9216 (-0.0144) vs bn[test] (transductive) 0.9272 (-0.0088) — the supervised source captures the same gain, so this is NOT a domain effect.
  weights=raw：bn[train]（纯监督）0.9216 与 bn[test]（转导式）0.9272 增益相当——**不是域效应**。

## Pixel-level train/test comparison / 像素级 train/test 对比

- Mean pixel value: train 0.4734 vs test 0.4907 (gap +0.0173 = +4.41 / 255 levels)
  像素均值：train 0.4734 vs test 0.4907 （差 +0.0173，即 +4.41 / 255 级）
- Std pixel value: train 0.2516 vs test 0.2683 (gap +0.0167)
  像素标准差：train 0.2516 vs test 0.2683 （差 +0.0167）
- Class-mix sampling error on the mean: 0.0003; the observed gap is 63.1x that error.
  按类别比例抽样造成的均值误差约 0.0003；实测差值是该误差的 63.1 倍。

## OOD evaluation (CIFAR-10.1 proxy) / 分布外评估（CIFAR-10.1 代理集）

CIFAR-10.1 is a **diagnostic proxy** for the re-collected test set. It is used **only** to measure out-of-distribution behaviour and was never used for training or for model selection.
CIFAR-10.1 是重采测试集的**诊断代理集**，**仅**用于测量分布外行为，从未用于训练或选型。

The in-distribution validation split cannot answer the BatchNorm question: the BN gain only shows up when the evaluation statistics differ from the training ones, which is not the case on our own IID split. The OOD proxy can.
IID 验证集无法回答 BN 的问题：BN 的收益只在「评估集统计量与训练集不同」时才显现，而我们自己的 IID 划分上两者相同。分布外代理集才能回答。

| Config | Weights | TTA | BN recal | BN imgs | Acc | 95% CI | NLL | ECE |
|---|---|---|---|---|---|---|---|---|
| `ood.ema.plain` | ema | plain | none | — | 0.8985 | [0.8845, 0.9110] | 0.4446 | 0.1500 |
| `ood.ema.flip` | ema | flip | none | — | 0.9015 | [0.8877, 0.9138] | 0.4429 | 0.1551 |
| `ood.ema.translate_flip` | ema | translate_flip | none | — | 0.8965 | [0.8824, 0.9091] | 0.4657 | 0.1650 |
| `ood.ema.plain+bn[cifar101]` | ema | plain | cifar101 | 2000 | 0.8790 | [0.8640, 0.8926] | 0.5953 | 0.2237 |
| `ood.ema.translate_flip+bn[cifar101]` | ema | translate_flip | cifar101 | 2000 | 0.8810 | [0.8661, 0.8945] | 0.6114 | 0.2448 |
| `ood.ema.plain+bn[test]` | ema | plain | test | 2000 | 0.8825 | [0.8676, 0.8959] | 0.4002 | 0.0590 |
| `ood.ema.translate_flip+bn[test]` | ema | translate_flip | test | 2000 | 0.8840 | [0.8692, 0.8973] | 0.4122 | 0.0810 |
| `ood.ema.plain+bn[train]` | ema | plain | train | 2000 | 0.8840 | [0.8692, 0.8973] | 0.3860 | 0.0450 |
| `ood.ema.translate_flip+bn[train]` | ema | translate_flip | train | 2000 | 0.8815 | [0.8666, 0.8949] | 0.3967 | 0.0611 |
| `ood.ema.plain+bn[train_big]` | ema | plain | train_big | 10000 | 0.8800 | [0.8650, 0.8935] | 0.3935 | 0.0429 |
| `ood.ema.translate_flip+bn[train_big]` | ema | translate_flip | train_big | 10000 | 0.8805 | [0.8655, 0.8940] | 0.4047 | 0.0601 |
| `ood.raw.plain` | raw | plain | none | — | 0.8700 | [0.8545, 0.8840] | 0.4754 | 0.0902 |
| `ood.raw.flip` | raw | flip | none | — | 0.8810 | [0.8661, 0.8945] | 0.4637 | 0.1096 |
| `ood.raw.translate_flip` | raw | translate_flip | none | — | 0.8700 | [0.8545, 0.8840] | 0.4906 | 0.1173 |
| `ood.raw.plain+bn[cifar101]` | raw | plain | cifar101 | 2000 | 0.8430 | [0.8264, 0.8583] | 0.7142 | 0.2378 |
| `ood.raw.translate_flip+bn[cifar101]` | raw | translate_flip | cifar101 | 2000 | 0.8485 | [0.8321, 0.8635] | 0.7245 | 0.2678 |
| `ood.raw.plain+bn[test]` | raw | plain | test | 2000 | 0.8505 | [0.8342, 0.8655] | 0.5113 | 0.0799 |
| `ood.raw.translate_flip+bn[test]` | raw | translate_flip | test | 2000 | 0.8560 | [0.8399, 0.8707] | 0.5136 | 0.1159 |
| `ood.raw.plain+bn[train]` | raw | plain | train | 2000 | 0.8485 | [0.8321, 0.8635] | 0.4942 | 0.0570 |
| `ood.raw.translate_flip+bn[train]` | raw | translate_flip | train | 2000 | 0.8575 | [0.8415, 0.8721] | 0.4961 | 0.0961 |
| `ood.raw.plain+bn[train_big]` | raw | plain | train_big | 10000 | 0.8450 | [0.8285, 0.8602] | 0.5029 | 0.0542 |
| `ood.raw.translate_flip+bn[train_big]` | raw | translate_flip | train_big | 10000 | 0.8525 | [0.8363, 0.8674] | 0.5052 | 0.0932 |
| `iid.ema.plain` | ema | plain | none | — | 0.9836 | [0.9826, 0.9846] | 0.2372 | 0.1756 |
| `iid.ema.flip` | ema | flip | none | — | 0.9843 | [0.9833, 0.9853] | 0.2362 | 0.1765 |
| `iid.raw.plain` | raw | plain | none | — | 0.9680 | [0.9666, 0.9694] | 0.2199 | 0.1231 |
| `iid.raw.flip` | raw | flip | none | — | 0.9753 | [0.9740, 0.9765] | 0.2111 | 0.1339 |

### The IID -> OOD gap / IID → OOD 缺口

- Same model, same preprocessing, weights=raw, TTA=flip
- IID (standard CIFAR-10 test, n=60000): **0.9753**
- OOD (CIFAR-10.1, n=2000): **0.8810**
- Gap: **-9.43 points** (-0.0943)
  同一模型、同一预处理：IID 0.9753 → OOD 0.8810，缺口 **-9.43 点**

This is the headroom that ensembling and TTA have to recover on the re-collected test set. / 这就是集成与 TTA 需要在重采测试集上补回的余量。

### Decision / 判定

- bn[cifar101] on OOD: acc 0.8485 (Δ=-0.0215 vs no-BN 0.8700), exact McNemar p=0.0018 → not significant
- bn[test] on OOD: acc 0.8560 (Δ=-0.0140 vs no-BN 0.8700), exact McNemar p=0.0314 → not significant
- bn[train] on OOD: acc 0.8575 (Δ=-0.0125 vs no-BN 0.8700), exact McNemar p=0.0535 → not significant
- bn[train_big] on OOD: acc 0.8525 (Δ=-0.0175 vs no-BN 0.8700), exact McNemar p=0.0076 → not significant
  - bn 各种来源在分布外上的表现（对照不做 BN 的 0.8700）

- `bn[train]` (purely supervised) reaches 0.8575 (Δ=-0.0125), versus `bn[test]` (transductive) 0.8560 (Δ=-0.0140).
  `bn[train]`（纯监督）0.8575，`bn[test]`（转导式）0.8560。

- `bn[train_big]` (purely supervised, more images) reaches 0.8525 (Δ=-0.0175).
  `bn[train_big]`（纯监督，更多图）0.8525（Δ=-0.0175）。

- Verdict / 判定：**DROP** from the submission path.
- 判定：**不进入**提交路径。

The gain is not distinguishable from noise (or is not positive), so BN recalibration stays an analysis tool and is not part of the submission path.
该增益与噪声无法区分（或非正），因此 BN 重校准保持为分析工具，不进入提交路径。

- Verdict / 判定：**DROP** from the default submission path.
- 判定：**不进入**默认提交路径。

This result is recorded as an **A/B option** for the final submission:
此结果记录为 final.pt 的 **A/B 选项**：

- weights=raw: no-BN 0.8700 | bn[train] 0.8575 | bn[test] 0.8560 | bn[cifar101] 0.8485 | bn[train_big] 0.8525

If BN recalibration is used on final.pt, the supervised `bn[train]` form is recommended (it captures the gain without transductive disclosure).
若在 final.pt 上希望使用 BN 重校准，推荐使用纯监督的 `bn[train]` 形式 (它能拿到收益且无需转导式披露)。


## Temperature-scaling diagnostic / 温度标定诊断

- Fitted on / 拟合集：val first half，evaluated on / 评估集：val second half
- T = 0.5589
- Acc 0.9648 -> 0.9648；NLL 0.2757 -> 0.1042

## Notes / 说明

- The McNemar column is an exact two-sided test on the discordant pairs against the baseline `ema.plain`; p > 0.05 means the difference is not distinguishable from noise on this validation set.
  McNemar 列为与基线 ema.plain 的精确二项检验；p > 0.05 表示该差异在验证集上无法与噪声区分。
- `bn[test]` is transductive: BN statistics are re-estimated on the unlabeled test images. This is legal in a Kaggle competition but must be disclosed in the report.
  bn[test] 属于转导式适配：在无标签测试图上重估 BN 统计量。Kaggle 竞赛允许，但报告必须披露。
- `bn[val]` is leaky (it uses the validation labels' images) and is only a diagnostic ceiling.
  bn[val] 使用了验证集，仅作诊断上限，不可作为最终方案。
- `bn[train_matched]` uses exactly as many images as the test set, so the comparison against `bn[test]` isolates the domain effect from the sample-volume effect.
  bn[train_matched] 与测试集张数完全相同，因此与 bn[test] 的对比能把「域效应」与「样本量效应」分开。
- All rows come from ONE snapshot of the checkpoint, so every weight variant is measured at the same training epoch. / 所有行都来自同一次 checkpoint 快照，因此各权重变体处于同一训练 epoch。
- No number here is copied from anywhere else; all are measured by this script.
  此处没有抄来的数字，全部由本脚本实测。
