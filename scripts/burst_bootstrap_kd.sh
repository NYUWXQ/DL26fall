#!/bin/bash
# Submit distillation jobs on the burst cluster, using the 600-epoch run as teacher.
# 用 600-epoch 的模型当 teacher，在 burst 集群提交蒸馏作业。
# Usage / 用法:  bash ~/dl26/scripts/burst_bootstrap_kd.sh
#   TEACHER=runs/full_s0/final.pt SEEDS="0 1 2 3" bash scripts/burst_bootstrap_kd.sh
set -euo pipefail
ACCOUNT="${ACCOUNT:-cs_gy_6923-2026fa}"
PARTITION="${PARTITION:-c12m85-a100-1}"
SEEDS="${SEEDS:-0 1 2 3}"
TEACHER="${TEACHER:-runs/full_s0/final.pt}"
EPOCHS="${EPOCHS:-300}"
ROOT="${ROOT:-$HOME/dl26}"
ENV="${ENV:-$HOME/envs/torch}"
cd "$ROOT"
if [ ! -f "$TEACHER" ]; then
  echo "[kd] teacher not found: $TEACHER"; echo "[kd] available:"; ls -1 runs/full_s*/final.pt 2>/dev/null || true
  exit 1
fi
for s in $SEEDS; do
  sbatch --account="$ACCOUNT" --partition="$PARTITION" --gres=gpu:1 \
         --export=ALL,SEED="$s",ROOT="$ROOT",ENV="$ENV",EPOCHS="$EPOCHS",TEACHER="$TEACHER" \
         slurm/train_burst_kd.sbatch
done
echo "[kd] submitted; checkpoints land in $ROOT/runs/kd_s<seed>/final.pt"
