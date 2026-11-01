#!/bin/bash
# Distillation job on the burst cluster. / burst 集群上的蒸馏作业。
# Student: 300-epoch run distilled from a 600-epoch teacher (KD T=2, alpha=0.75),
# which is the configuration the teammate's 0.906 submission used.
# 学生网络：从 600-epoch teacher 蒸馏 300 epoch（T=2, alpha=0.75），与队友 0.906 配置一致。
#
#SBATCH --job-name=dl26kd
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
TEACHER="${TEACHER:-runs/full_s0/final.pt}"
EPOCHS="${EPOCHS:-300}"

cd "$ROOT"
source "$ENV/bin/activate"
echo "[kdjob] host=$(hostname) seed=$SEED teacher=$TEACHER"
nvidia-smi --query-gpu=name --format=csv,noheader || true

python -u src/train.py \
  --model preact16x5.25 --seed "$SEED" \
  --batch-size 128 --lr 0.1 --weight-decay 5e-4 \
  --randaug-m 6 --erase-p 0.0 --aug strong \
  --mix both --mix-always --mixup-alpha 0.2 --cutmix-alpha 1.0 \
  --label-smoothing 0.0 --ema-dynamic \
  --teacher "$TEACHER" --kd-t 2.0 --kd-alpha 0.75 \
  --full-data --epochs "$EPOCHS" --workers 8 \
  --out "runs/kd_s$SEED"
echo "[kdjob] done seed=$SEED"
