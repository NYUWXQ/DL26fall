#!/bin/bash
# Parameterised architecture training job for the burst cluster.
# burst 集群上的可配置结构训练作业（由 scripts/burst_bootstrap_arch.sh 提交）。
#
#SBATCH --job-name=dl26arch
#SBATCH --output=%x_%j.log
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24GB
#SBATCH --time=04:00:00

set -euo pipefail
ROOT="${ROOT:-$HOME/dl26}"
ENV="${ENV:-$HOME/envs/torch}"
SEED="${SEED:-0}"
EPOCHS="${EPOCHS:-600}"
MODEL="${MODEL:-preact22x4}"
TAG="${MODEL//./_}"

cd "$ROOT"
source "$ENV/bin/activate"
echo "[archjob] host=$(hostname) model=$MODEL seed=$SEED epochs=$EPOCHS"
nvidia-smi --query-gpu=name --format=csv,noheader || true

# Same faithful recipe that scored 0.885-0.888; only the architecture changes.
# 与拿到 0.885-0.888 的忠实配方一致，只换结构。
python -u src/train.py \
  --model "$MODEL" --seed "$SEED" \
  --batch-size 128 --lr 0.1 --weight-decay 5e-4 \
  --randaug-m 6 --erase-p 0.0 --aug strong \
  --mix both --mix-always --mixup-alpha 0.2 --cutmix-alpha 1.0 \
  --label-smoothing 0.0 --ema-dynamic \
  --full-data --epochs "$EPOCHS" --workers 8 \
  --out "runs/arch_${TAG}_s$SEED"
echo "[archjob] done $MODEL seed=$SEED"
