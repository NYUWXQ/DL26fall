#!/bin/bash
# End-to-end bootstrap on the NYU GCP burst cluster (course allocation).
# 在 NYU GCP burst 集群（课程分配）上的一键引导。
#
# What it does / 做什么:
#   1. creates a venv with CUDA torch           建 CUDA torch 的 venv
#   2. downloads CIFAR-10 (train images are pixel-identical to the competition set,
#      verified 50000/50000, so nothing has to be uploaded)
#      下载 CIFAR-10（竞赛训练图与官方逐像素一致，已验证 50000/50000，因此无需上传）
#   3. builds the uint8 tensor cache the pipeline expects / 生成管线需要的 uint8 缓存
#   4. submits N single-GPU training jobs with the faithful recipe
#      按忠实配方提交 N 个单卡训练作业
#
# Usage (inside an OOD JupyterLab terminal) / 用法（在 OOD JupyterLab 的终端里）:
#   git clone -b chief/local-run https://github.com/NYUWXQ/DL26fall.git ~/dl26
#   bash ~/dl26/scripts/burst_bootstrap.sh
#
# Override defaults with env vars / 用环境变量覆盖默认值:
#   ACCOUNT=cs_gy_6923-2026fa PARTITION=c12m85-a100-1 SEEDS="0 1 2 3" bash scripts/burst_bootstrap.sh

set -euo pipefail

ACCOUNT="${ACCOUNT:-cs_gy_6923-2026fa}"
PARTITION="${PARTITION:-c12m85-a100-1}"
SEEDS="${SEEDS:-0 1 2 3}"
EPOCHS="${EPOCHS:-300}"
ROOT="${ROOT:-$HOME/dl26}"
ENV="${ENV:-$HOME/envs/torch}"

echo "[bootstrap] account=$ACCOUNT partition=$PARTITION seeds='$SEEDS' root=$ROOT"
cd "$ROOT"

# 1) Python environment with CUDA torch. / 带 CUDA torch 的 Python 环境。
if [ ! -x "$ENV/bin/python" ]; then
  echo "[bootstrap] creating venv $ENV"
  python3 -m venv "$ENV"
fi
"$ENV/bin/python" -m pip install -q --upgrade pip
"$ENV/bin/python" -c "import torch" 2>/dev/null || {
  echo "[bootstrap] installing torch (cu121) — this takes a few minutes"
  "$ENV/bin/python" -m pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu121
}
"$ENV/bin/python" -c "import torch;print('[bootstrap] torch',torch.__version__,'cuda-build',torch.version.cuda)"

# 2) CIFAR-10 (the competition train images are byte-identical to it). / 数据。
mkdir -p data logs
if [ ! -d data/cifar-10-batches-py ]; then
  echo "[bootstrap] downloading CIFAR-10 (~163 MB)"
  curl -sL -o data/cifar-10-python.tar.gz https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
  tar -xzf data/cifar-10-python.tar.gz -C data
fi

# 3) Tensor cache. / 张量缓存。
"$ENV/bin/python" scripts/bootstrap_cache_from_cifar10.py --batches data/cifar-10-batches-py --root .

# 4) Submit one job per seed. / 每个 seed 投一个作业。
for s in $SEEDS; do
  sbatch --account="$ACCOUNT" --partition="$PARTITION" --gres=gpu:1 \
         --export=ALL,SEED="$s",ROOT="$ROOT",ENV="$ENV",EPOCHS="$EPOCHS" \
         slurm/train_burst.sbatch
done

echo
echo "[bootstrap] submitted. Watch with:  squeue -u \$USER"
echo "[bootstrap] logs land in:        $ROOT/logs/"
echo "[bootstrap] checkpoints land in: $ROOT/runs/full_s<seed>/final.pt"
