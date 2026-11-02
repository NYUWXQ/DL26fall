#!/bin/bash
# SE-ResNet-164/RICAP training + probability export on the burst cluster.
# burst 集群上的 SE-ResNet-164/RICAP 训练与概率导出。
# Submitted by scripts/burst_bootstrap_se.sh; do not call directly.
#
#SBATCH --job-name=dl26se
#SBATCH --output=%x_%j.log
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=12:00:00

set -euo pipefail
ROOT="${ROOT:-$HOME/dl26}"
ENV="${ENV:-$HOME/envs/torch}"
SEED="${SEED:-0}"
EPOCHS="${EPOCHS:-1800}"
SMOKE="${SMOKE:-0}"
NAME="${NAME:-se_s$SEED}"

cd "$ROOT/proj1"
source "$ENV/bin/activate"
echo "[sejob] host=$(hostname) name=$NAME seed=$SEED epochs=$EPOCHS smoke=$SMOKE"
nvidia-smi --query-gpu=name --format=csv,noheader || true

# The reference recipe from proj1/RESULTS.md. / 队友 RESULTS.md 里的参考配方。
COMMON=(--name "$NAME" --device cuda:0 --architecture se --depth 164 --width 16
        --autoaugment --ricap 0.3 --random-erasing 0.5 --cutout 0
        --lr 0.1 --weight-decay 0.0001 --warmup-epochs 60 --min-lr 0.0001
        --batch-size 128 --workers 8 --full-data)

if [ "$SMOKE" = "1" ]; then
  # One epoch of 20 batches: proves the pipeline runs end to end in ~10 minutes.
  # 1 epoch 只跑 20 批：10 分钟内证明整条管线能跑通。
  python -u train.py "${COMMON[@]}" --epochs 1 --max-batches 20
  echo "[sejob] smoke OK"
  exit 0
fi

python -u train.py "${COMMON[@]}" --epochs "$EPOCHS"

# Export the test-set probability matrix so it can be fused with our 17 models.
# 导出测试集概率矩阵，供与我们的 17 个模型融合。
CKPT="$ROOT/proj1/runs/$NAME/final.pt"
[ -f "$CKPT" ] || CKPT="$ROOT/proj1/runs/$NAME/best.pt"
mkdir -p "$ROOT/proj1/runs/$NAME/probs"
python -u predict.py "$CKPT" --data "$ROOT/data/competition" \
  --out "$ROOT/proj1/runs/$NAME/probs/probs.csv" --tta flip --precision fp32
echo "[sejob] done $NAME; probabilities in runs/$NAME/probs/"
