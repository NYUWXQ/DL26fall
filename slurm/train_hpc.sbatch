#!/bin/bash
# Faithful-recipe training on NYU Torch (GPU). / 在 NYU Torch GPU 上跑忠实配方。
#
# Matches the teammate pipeline that scored 0.906 on the leaderboard:
#   WRN-16 width 84 (4.78M params) / bs128 / lr0.1 / wd5e-4 / RandAugment m6 / no erasing
#   mixup 0.2 + cutmix 1 mixed every step / no label smoothing / EMA 0.999 with warmup
#   full 50k training data / 300 epochs
#
# Submit / 提交:
#   sbatch --export=ALL,NAME=s0,SEED=0 slurm/train_hpc.sbatch
#   sbatch --export=ALL,NAME=s1,SEED=1 slurm/train_hpc.sbatch

#SBATCH --job-name=dl26
#SBATCH --output=/scratch/qs2283/dl26fall/logs/%x_%j.log
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=03:00:00
#SBATCH --account=users

set -euo pipefail

CODE=/scratch/qs2283/dl26fall/code
DATA=/scratch/qs2283/dl26fall/data
ENV=/scratch/qs2283/envs/torch

NAME="${NAME:-s0}"
SEED="${SEED:-0}"
EPOCHS="${EPOCHS:-300}"
BATCH="${BATCH:-128}"
LR="${LR:-0.1}"

source "$ENV/bin/activate"
cd "$CODE"

# Build the local-format tensor cache from the official CIFAR-10 batches once.
# 一次性从官方 CIFAR-10 batch 生成本地格式的缓存。
if [ ! -f .cache/train_x.npy ]; then
  python scripts/bootstrap_cache_from_cifar10.py --batches "$DATA/cifar-10-batches-py" --root .
fi
[ -e test_images ] || ln -sfn "$DATA/test_images" test_images

echo "[job] host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) name=$NAME seed=$SEED"
python -u src/train.py \
  --model preact16x5.25 --seed "$SEED" \
  --batch-size "$BATCH" --lr "$LR" --weight-decay 5e-4 \
  --randaug-m 6 --erase-p 0.0 --aug strong \
  --mix both --mix-always --mixup-alpha 0.2 --cutmix-alpha 1.0 \
  --label-smoothing 0.0 --ema-dynamic \
  --full-data --epochs "$EPOCHS" --workers 8 \
  --out "runs/full_$NAME"
echo "[job] done $NAME"
