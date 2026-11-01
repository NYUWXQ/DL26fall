#!/bin/bash
# Single-GPU training job for the NYU GCP burst cluster. / burst 集群单卡训练作业。
# Submitted by scripts/burst_bootstrap.sh; do not call directly.
# 由 scripts/burst_bootstrap.sh 提交，不要直接调用。
#
#SBATCH --job-name=dl26train
#SBATCH --output=%x_%j.log
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24GB
#SBATCH --time=02:00:00

set -euo pipefail

ROOT="${ROOT:-$HOME/dl26}"
ENV="${ENV:-$HOME/envs/torch}"
SEED="${SEED:-0}"
EPOCHS="${EPOCHS:-300}"

cd "$ROOT"
source "$ENV/bin/activate"

echo "[job] host=$(hostname) seed=$SEED epochs=$EPOCHS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# Faithful reproduction of the pipeline that scored 0.906 on the leaderboard.
# 忠实复现那个在榜上拿到 0.906 的配方。
python -u src/train.py \
  --model preact16x5.25 --seed "$SEED" \
  --batch-size 128 --lr 0.1 --weight-decay 5e-4 \
  --randaug-m 6 --erase-p 0.0 --aug strong \
  --mix both --mix-always --mixup-alpha 0.2 --cutmix-alpha 1.0 \
  --label-smoothing 0.0 --ema-dynamic \
  --full-data --epochs "$EPOCHS" --workers 8 \
  --out "runs/full_s$SEED"
echo "[job] done seed=$SEED"
