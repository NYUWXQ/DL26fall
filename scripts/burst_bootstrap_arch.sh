#!/bin/bash
# Submit a training batch for an arbitrary architecture (diversity run).
# 按任意结构投递训练批次（多样性实验用）。
#
# Why: on the competition test set the only lever that reproduced was architectural
# diversity (+0.3 LB per added family), while extra epochs (1200 vs 300) gave nothing.
# 为什么：在竞赛测试集上唯一复现出来的杠杆是架构多样性（每加一族 +0.3），
# 而单纯加 epoch（1200 vs 300）没有收益。
#
# Usage / 用法:
#   MODEL=preact22x4 SEEDS="0 1 2 3" EPOCHS=600 bash scripts/burst_bootstrap_arch.sh
#   MODEL=preact28x2 SEEDS="0 1"     EPOCHS=600 bash scripts/burst_bootstrap_arch.sh
set -euo pipefail
ACCOUNT="${ACCOUNT:-cs_gy_6923-2026fa}"
PARTITION="${PARTITION:-c12m85-a100-1}"
MODEL="${MODEL:-preact22x4}"
SEEDS="${SEEDS:-0 1 2 3}"
EPOCHS="${EPOCHS:-600}"
ROOT="${ROOT:-$HOME/dl26}"
ENV="${ENV:-$HOME/envs/torch}"

cd "$ROOT"
source "$ENV/bin/activate"
# Parameter budget guard before spending GPU hours. / 花 GPU 之前先过参数守卫。
python -c "
import sys; sys.path.insert(0,'src')
from models import build_model, n_params
n = n_params(build_model('$MODEL'))
print(f'[arch] $MODEL params={n:,} budget_ok={n<=5_000_000}')
assert n <= 5_000_000, 'over the 5M cap'
"

for s in $SEEDS; do
  sbatch --account="$ACCOUNT" --partition="$PARTITION" --gres=gpu:1 \
         --export=ALL,SEED="$s",ROOT="$ROOT",ENV="$ENV",EPOCHS="$EPOCHS",MODEL="$MODEL" \
         slurm/train_burst_arch.sbatch
done
echo "[arch] submitted $MODEL seeds='$SEEDS' epochs=$EPOCHS -> runs/arch_<model>_s<seed>/final.pt"
