# Project 1 — CIFAR-10 ResNet Challenge (DL 26 Fall)

CIFAR-10 variant competition: train a residual network **from scratch** with **≤ 5,000,000 parameters**,
no pretrained weights, no external datasets.

本仓库：CIFAR-10 变体竞赛。要求从零训练残差网络，参数量 ≤5M，禁用预训练权重与外部数据集。

- Competition: https://www.kaggle.com/competitions/cifar-10-res-net-architecture-challenge-dl
- Report: [`report/report.pdf`](report/report.md) (≤ 8 pages, English)
- HPC guide: [`docs/HPC.md`](docs/HPC.md) · notebook [`notebooks/hpc_train.ipynb`](notebooks/hpc_train.ipynb)

## Layout / 目录

| Path | What |
|---|---|
| `src/models.py` | Pre-activation wide ResNet, parameter-budget guard / 预激活宽残差网络与参数守卫 |
| `src/data.py` | PNG → uint8 tensor cache, augmentation / PNG → uint8 张量缓存与数据增强 |
| `src/train.py` | SGD + cosine + label smoothing + MixUp/CutMix + EMA / 训练主循环 |
| `src/predict.py` | 10-view TTA + checkpoint ensembling → submission CSV / TTA 与集成推理 |
| `src/refine.py` | Inference-configuration comparison / 推理配置对比 |
| `slurm/train.sbatch` | NYU Torch Slurm template / Torch 作业模板 |
| `notebooks/hpc_train.ipynb` | Step-by-step HPC runbook / HPC 操作手册 |
| `analysis/` | Leaderboard-noise and domain-shift analysis / 噪声与域偏移分析 |
| `report/` | English report sources / 英文报告源文件 |
| `proj1/` | Teammates' original pipeline (cited in the report) / 队友原始管线 |

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
.venv/bin/python src/models.py            # all models print params and ok=True
grep -rn "pretrained=True\|from_pretrained\|torch.hub" src/   # must be empty
```

## Notes on the test set / 关于测试集

Local validation reaches ~97% while the public leaderboard saturates near 0.909 for every
team. Fitting `LB = (1-p)·a + p/10` gives an irreducible label-noise rate of roughly 7%,
so improvements must come from true accuracy and ensembling rather than architecture novelty.
See [`analysis/lb_noise.md`](analysis/lb_noise.md).

本地验证可达 ~97%，但公开榜所有队伍都挤在 0.909 附近。用 `LB = (1-p)·a + p/10` 拟合得到
约 7% 的不可约标签噪声，因此提升要靠真实精度与集成，而不是换结构。
