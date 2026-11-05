# Project 1 — CIFAR-10 ResNet Challenge (DL 26 Fall)

CIFAR-10 variant competition: train a residual network **from scratch** with **≤ 5,000,000 parameters**,
no pretrained weights, no external datasets.
CIFAR-10 变体竞赛：从零训练残差网络，参数量 ≤5M，禁用预训练权重与外部数据集。

- Competition: https://www.kaggle.com/competitions/cifar-10-res-net-architecture-challenge-dl
- Assignment: [`project1.pdf`](project1.pdf)

## Submitted report and executed evidence / 提交的报告与执行证据

| What | Where |
|---|---|
| Submitted report source (AAAI 2027 style, 5 pages) | [`report/latex/main.tex`](report/latex/main.tex) + [`project1.bib`](report/latex/project1.bib) + [`Figures/`](report/latex/Figures) |
| How to build it | [`report/latex/README.md`](report/latex/README.md) |
| Executed notebook (required evidence) | [`notebooks/results.ipynb`](notebooks/results.ipynb) — loads a checkpoint, prints **4,326,986** trainable parameters, measures **0.9750** on the 10,000 labelled CIFAR-10 test images (0.9732 plain), plots the confusion matrix and training curves |
| Every claim and its provenance | [`docs/REPORT_FACTS.md`](docs/REPORT_FACTS.md) — including the corrections we made to earlier drafts |
| HPC runbook | [`docs/HPC.md`](docs/HPC.md) |

The submitted PDF is produced from `report/latex/` on Overleaf (pdflatex + bibtex, twice). The older
Markdown draft in `report/report.md` is kept as a design record only; where it disagrees with the LaTeX
source, the LaTeX source wins.

## Leaderboard results / 榜单结果

Every row is a real Kaggle return, not a local estimate. `Params` is the trainable parameter count of a
single member as printed by `python src/models.py`; the `Log` column is the raw CLI output under
`submissions/` that the number comes from.

| Configuration | Kind | Params | Public LB | Submission | Log |
|---|---|---:|---:|---|---|
| Pre-act. WRN-22, 300 ep, 47.5k split, flip TTA | single, compliant | 4,326,986 | 0.879 | `final_ema_flip.csv` | `log_20261101_190203` |
| Same, ten-view TTA | single, compliant | 4,326,986 | 0.873 | `final_ema_10view.csv` | `log_20261101_190203` |
| Faithful recipe, WRN-16 w=84, 300 ep, full 50k | single, compliant | 4,780,366 | 0.885 | `e4_faithful.csv` | `log_20261101_190203` |
| Faithful recipe, 1200 ep, full 50k | single, compliant | 4,780,366 | 0.884 | `full_s0_1200ep.csv` | `log_20261101_190210` |
| Distilled from a 600-epoch teacher | single, compliant | 4,780,366 | 0.888 | `kd_s11_single.csv` | `log_20261101_190210` |
| SE-ResNet-164 w=16, 600 ep, run 1 | single, compliant | 2,488,378 | 0.894 | `se_s0_single.csv` | `log_20261103_190208` |
| SE-ResNet-164 w=16, 600 ep, run 2 | single, compliant | 2,488,378 | 0.899 | `se_s1_single.csv` | `log_20261103_190208` |
| SE-ResNet-164 **w=22**, 600 ep, run 1 | single, compliant | 4,684,888 | **0.902** | `se_w22_s0_single.csv` | `log_20261104_210025` |
| SE-ResNet-164 **w=22**, 600 ep, run 2 | single, compliant | 4,684,888 | 0.896 | `se_w22_s1_single.csv` | `log_20261104_210025` |
| Two-run average of the two lines above | 2 models, **summed 4,976,756** (boundary reading) | — | **0.902** | `se_s0s1_avg2.csv` | `log_20261103_190208` |
| Two runs of the faithful recipe | 2-model ensemble, ablation | — | 0.886 | `ens2_e4e5.csv` | `log_20261101_190203` |
| Four distillation runs | 4-model ensemble, ablation | — | 0.891 | `ens_kd600_4.csv` | `log_20261101_190210` |
| Four students (×3 weight) + two local runs | 6 networks, ablation | — | 0.894 | `kd4x3+e1+e4.csv` | `log_20261101_190541` |
| 4×WRN-22 + 8×WRN-16, equal weight | 12-model ensemble, ablation | — | **0.903** | `ens12_pure.csv` | `log_20261102_190203` |
| Twelve members + external hard-label votes (k=2, k=4) | 12 models + external, ablation | — | 0.901 | `ens12_his_k2.csv`, `ens12_his_k4.csv` | `log_20261102_190203` |

## Compliance position / 合规口径

`the total number of trainable parameters does not exceed 5 million`, checked with `torchsummary` on
*your model*. Read literally, a twelve-member ensemble carries between 52M and 57M trainable parameters,
so **our deliverable is a single network under the cap** and every ensemble above is labelled an ablation.
The one boundary case is the two-run SE average: 2 × 2,488,378 = **4,976,756**, still under five million
under the same literal reading. It is named as a two-model average, never presented as a single model.

## Layout / 目录

| Path | What |
|---|---|
| `src/models.py` | Pre-activation wide ResNet, parameter-budget guard / 预激活宽残差网络与参数守卫 |
| `src/data.py` | PNG → uint8 tensor cache, augmentation / PNG → uint8 张量缓存与数据增强 |
| `src/train.py` | SGD + cosine + label smoothing + MixUp/CutMix + EMA / 训练主循环 |
| `src/predict.py` | flip / 10-view TTA + checkpoint ensembling → submission CSV |
| `src/refine.py` | Inference-configuration comparison / 推理配置对比 |
| `scripts/fuse_probs.py` | Shape-checked probability fusion, optional hard-label votes / 概率融合 |
| `scripts/submit_candidates.sh` | Quota-safe submission wrapper that logs every score / 提交与记分 |
| `scripts/build_report_pdf.py` | Markdown draft → PDF (page-count guard) / Markdown 草案转 PDF |
| `scripts/burst_se_w22.sh` | Width-22 SE control experiment on the course cluster / 宽度对照实验 |
| `slurm/*.sbatch` | NYU Slurm templates for the course allocation / Slurm 模板 |
| `analysis/` | Leaderboard-noise and domain-shift analysis / 噪声与域偏移分析 |
| `proj1/` | Teammates' SE-ResNet/RICAP pipeline (cited in the report) / 队友原始管线 |
| `report/latex/` | The submitted report source / 提交报告源文件 |

## Quick start / 快速开始

```bash
python3 -m venv .venv && .venv/bin/pip install torch torchvision
.venv/bin/python src/models.py                       # parameter budget table / 参数预算表
.venv/bin/python src/train.py --model preact22x4 --epochs 300 --batch-size 512 \
    --lr 0.25 --aug strong --mix both --out runs/wrn22x4_s0 --workers 4
.venv/bin/python src/predict.py --ckpt runs/wrn22x4_s0/final.pt --out submissions/submission.csv
kaggle competitions submit -c cifar-10-res-net-architecture-challenge-dl -f submissions/submission.csv -m "run note"
```

## Constraint checks / 约束自检

```bash
.venv/bin/python src/models.py            # every model prints params and ok=True/False
grep -rn "pretrained=True\|from_pretrained\|torch.hub" src/   # must be empty
```

## What the test set is / 测试集性质（已审计）

The shipped training images are the official CIFAR-10 training set, all 50,000 of them, matched
pixel-for-pixel by MD5. The 2,000 test images are **not**: they match neither the official CIFAR-10 test
split nor CIFAR-10.1 at any of the eight dihedral transforms (zero exact matches; 18 chance hits under
horizontal flip). A controlled experiment rules out a photometric explanation — applying the measured
test-set brightness/contrast difference to a held-out split moved accuracy by +0.2 points, while a blur
negative control dropped it 57.6 points.

So local validation (~0.975) and the leaderboard (0.879–0.903) are two different measurements of two
different pools, and the repository keeps them apart. We make **no claim** about label noise: an earlier
`LB = (1-p)·a + p/10` fit that suggested ~7% was withdrawn because it conflicts with the pixel-level
evidence. See `analysis/lb_noise.md` for the analysis and `docs/REPORT_FACTS.md` for the current position.
