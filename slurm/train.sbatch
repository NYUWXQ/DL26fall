#!/bin/bash
# NYU Torch Slurm template for this project. / 本项目在 NYU Torch 上的 Slurm 模板。
#
# Usage / 用法:
#   sbatch slurm/train.sbatch                      # train preact22x4 for 300 epochs
#   sbatch --export=ALL,EPOCHS=600,LR=0.25 slurm/train.sbatch
#
# Before first use, replace the two placeholders below.
# 首次使用前替换下面两个占位符。
#   ACCOUNT: run `my_slurm_accounts` on Torch to get it (e.g. torch_pr_123456_abcde)
#   OVERLAY: your conda+torch overlay created per docs/HPC.md

#SBATCH --job-name=dl26-cifar
#SBATCH --output=runs/hpc_%j.log
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=06:00:00
#SBATCH --account=torch_pr_XXXXXX_XXXXX
# Preemptible: allows borrowing idle stakeholder GPUs, auto-requeued on preemption.
# 抢占模式：可用别人空闲的 GPU，被抢占后自动重新排队。
#SBATCH --comment="preemption=yes;requeue=true"
# Do NOT set --partition on Torch: QoS handles placement (official docs).
# 在 Torch 上不要手写 --partition，资源由 QoS 分配（官方文档要求）。

set -euo pipefail

ACCOUNT="${ACCOUNT:-torch_pr_XXXXXX_XXXXX}"
OVERLAY="${OVERLAY:-/scratch/$USER/dl26fall/pytorch.ext3}"
SIF=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
CODE="${CODE:-/scratch/$USER/dl26fall/code}"
EPOCHS="${EPOCHS:-300}"
LR="${LR:-0.25}"
BATCH="${BATCH:-512}"

cd "$CODE"
mkdir -p runs

# Resume if a checkpoint exists, else fresh start. / 有 checkpoint 则续训，否则从头训练。
EXTRA=""
if [ -f "runs/wrn22x4_s0/last.pt" ]; then
  EXTRA="--resume"
fi

echo "[sbatch] node=$(hostname) gpu=$(nvidia-smi -L | head -1) epochs=$EPOCHS lr=$LR batch=$BATCH resume=${EXTRA:-no}"

# --nv exposes the GPU to the container; :ro keeps the overlay immutable at runtime.
# --nv 把 GPU 透传给容器；:ro 让 overlay 运行时只读。
singularity exec --nv --overlay "${OVERLAY}:ro" "$SIF" /bin/bash -c "
  source /ext3/env.sh
  python -u src/train.py \
      --root $CODE --model preact22x4 --epochs $EPOCHS --batch-size $BATCH \
      --lr $LR --aug strong --mix both --ema-decay 0.9995 \
      --out runs/wrn22x4_s0 --workers 8 $EXTRA
"
