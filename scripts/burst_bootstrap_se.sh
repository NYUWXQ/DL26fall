#!/bin/bash
# Train Xinquan's SE-ResNet-164/RICAP pipeline on the burst cluster and export
# test-set probabilities, so the team's best single model can join the ensemble.
# 在 burst 集群上训练队友的 SE-ResNet-164/RICAP 管线并导出测试集概率，
# 让全队最好的单模型能进入我们的集成。
#
# Reference recipe (proj1/RESULTS.md, 98.10% on the 5k holdout):
#   2,488,378 params / 1800 epochs / AutoAugment + RICAP(beta 0.3) + RandomErasing(0.5)
#   SGD-Nesterov, lr 0.1 -> 1e-4 cosine, 60-epoch warmup, wd 1e-4
#
# Usage / 用法:
#   SEEDS="0 1 2 3" EPOCHS=1800 bash scripts/burst_bootstrap_se.sh
#   SMOKE=1 bash scripts/burst_bootstrap_se.sh          # 15-min sanity job / 冒烟测试
set -euo pipefail
ACCOUNT="${ACCOUNT:-cs_gy_6923-2026fa}"
PARTITION="${PARTITION:-c12m85-a100-1}"
SEEDS="${SEEDS:-0 1 2 3}"
EPOCHS="${EPOCHS:-1800}"
ROOT="${ROOT:-$HOME/dl26}"
ENV="${ENV:-$HOME/envs/torch}"
SMOKE="${SMOKE:-0}"

cd "$ROOT"

# torchvision wants <root>/cifar-10-batches-py and the competition images want a
# test.csv next to test_images/. Wire both once. / 把数据路径接好。
mkdir -p "$ROOT/data/competition"
[ -e "$ROOT/proj1/data" ] || ln -sfn "$ROOT/data" "$ROOT/proj1/data"
[ -e "$ROOT/data/competition/test.csv" ] || ln -sfn "$ROOT/test.csv" "$ROOT/data/competition/test.csv"
[ -e "$ROOT/data/competition/test_images" ] || ln -sfn "$ROOT/test_images" "$ROOT/data/competition/test_images"

# Parameter-budget guard (hard 5M cap) before spending GPU hours.
# 花 GPU 之前先过 5M 参数守卫。
"$ENV/bin/python" -c "
import sys; sys.path.insert(0, '$ROOT/proj1')
from model import build_model
m = build_model({'architecture': 'se', 'depth': 164, 'width': 16})
n = sum(p.numel() for p in m.parameters())
print(f'[se] params={n:,} budget_ok={n <= 5_000_000}')
assert n <= 5_000_000
"

if [ "$SMOKE" = "1" ]; then
  sbatch --account="$ACCOUNT" --partition="$PARTITION" --gres=gpu:1 --time=00:20:00 \
         --export=ALL,SEED=0,ROOT="$ROOT",ENV="$ENV",EPOCHS=1,SMOKE=1,NAME=se_smoke \
         slurm/train_burst_se.sbatch
  echo "[se] smoke job submitted; check logs before launching the real batch"
  exit 0
fi

for s in $SEEDS; do
  sbatch --account="$ACCOUNT" --partition="$PARTITION" --gres=gpu:1 --time=12:00:00 \
         --export=ALL,SEED="$s",ROOT="$ROOT",ENV="$ENV",EPOCHS="$EPOCHS",SMOKE=0,NAME="se_s$s" \
         slurm/train_burst_se.sbatch
done
echo "[se] submitted $SEEDS epochs=$EPOCHS"
echo "[se] checkpoints: $ROOT/proj1/runs/se_s<seed>/"
echo "[se] probabilities: $ROOT/proj1/runs/se_s<seed>/probs/probs.probabilities.npy"
