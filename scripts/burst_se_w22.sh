#!/bin/bash
# Width-22 SE-ResNet control experiment, run inside an already-allocated GPU session.
# 在已开好的 GPU session 里跑 width-22 SE-ResNet 受控实验。
#
# Why this script: the only confound-free comparison we still owe the report is
# "same recipe, same 600 epochs, wider network". Everything else (architecture,
# augmentation, schedule, full-data training) is copied verbatim from the width-16
# runs that returned 0.894 and 0.899, so the difference isolates width.
# 为什么写这个脚本：报告还欠一个无混杂的对比——同样的配方、同样的 600 epoch，只把网络加宽。
# 架构、增强、调度、全量训练都与 0.894 / 0.899 那两条 width-16 完全一致，差出来的只可能是宽度。
#
# The training loop does not expose a seed switch (it pins torch.manual_seed(42)),
# so the two runs differ only through non-deterministic kernels and different nodes.
# That is exactly the run-to-run variance our existing pair measured, and we report
# it as such rather than pretending it is seed control.
# 训练器没有 seed 开关（固定 manual_seed(42)），两条跑只靠非确定性核与不同节点产生差异，
# 这正是现有那对模型测到的 run-to-run 方差，报告里按这个口径写，不冒充 seed 对照。
#
# Usage / 用法 (inside the OOD session terminal):
#   bash ~/dl26/proj1/scripts/burst_se_w22.sh                 # first 8h window
#   RESUME=1 bash ~/dl26/proj1/scripts/burst_se_w22.sh        # next window, continue
#   WAIT=1 bash ~/dl26/proj1/scripts/burst_se_w22.sh          # block, then export probs
#   SMOKE=1 bash ...                                          # 20 batches, prove the pipeline
#
# The processes die when the session ends; `last.pt` is written every epoch, so the
# next session resumes from it. / session 结束进程会被杀；每 epoch 落 last.pt，下次 resume。
set -euo pipefail

ROOT="${ROOT:-$HOME/dl26}"
ENV="${ENV:-$HOME/envs/torch}"
EPOCHS="${EPOCHS:-600}"
WIDTH="${WIDTH:-22}"
SEEDS="${SEEDS:-0 1}"
RESUME="${RESUME:-0}"
WAIT="${WAIT:-0}"
SMOKE="${SMOKE:-0}"

cd "$ROOT"

# Wire the data paths the trainer expects. / 接好训练器需要的数据路径。
mkdir -p "$ROOT/data/competition"
[ -e "$ROOT/proj1/data" ] || ln -sfn "$ROOT/data" "$ROOT/proj1/data"
[ -e "$ROOT/data/competition/test.csv" ] || ln -sfn "$ROOT/test.csv" "$ROOT/data/competition/test.csv"
[ -e "$ROOT/data/competition/test_images" ] || ln -sfn "$ROOT/test_images" "$ROOT/data/competition/test_images"

source "$ENV/bin/activate"
cd "$ROOT/proj1"

# Parameter-budget guard before any GPU hour is spent. / 花 GPU 前先过 5M 守卫。
python - "$WIDTH" <<'PY'
import sys
sys.path.insert(0, '.')
from model import build_model
width = int(sys.argv[1])
model = build_model({'architecture': 'se', 'depth': 164, 'width': width})
params = sum(p.numel() for p in model.parameters())
print(f'[se-w{width}] params={params:,} budget_ok={params <= 5_000_000}')
assert params <= 5_000_000, 'width exceeds the 5M cap'
PY

RESUME_FLAG=""
[ "$RESUME" = "1" ] && RESUME_FLAG="--resume"

if [ "$SMOKE" = "1" ]; then
  # One epoch of 20 batches: proves data, augmentations and checkpointing end to end.
  CUDA_VISIBLE_DEVICES=0 python -u train.py \
    --name "se_w${WIDTH}_smoke" --device cuda:0 --architecture se --depth 164 --width "$WIDTH" \
    --autoaugment --ricap 0.3 --random-erasing 0.5 --cutout 0 \
    --lr 0.1 --weight-decay 0.0001 --warmup-epochs 60 --min-lr 0.0001 \
    --batch-size 128 --workers 8 --full-data --epochs 1 --max-batches 20
  echo "[se-w${WIDTH}] smoke OK"
  exit 0
fi

INDEX=0
for SEED in $SEEDS; do
  NAME="se_w${WIDTH}_s${SEED}"
  LOG="$ROOT/${NAME}.log"
  echo "[launch] $NAME on GPU $INDEX -> $LOG"
  CUDA_VISIBLE_DEVICES=$INDEX nohup python -u train.py \
    --name "$NAME" --device cuda:0 --architecture se --depth 164 --width "$WIDTH" \
    --autoaugment --ricap 0.3 --random-erasing 0.5 --cutout 0 \
    --lr 0.1 --weight-decay 0.0001 --warmup-epochs 60 --min-lr 0.0001 \
    --batch-size 128 --workers 8 --full-data --epochs "$EPOCHS" $RESUME_FLAG \
    > "$LOG" 2>&1 &
  echo "[launch] pid $!"
  INDEX=$((INDEX + 1))
done

echo "[launch] follow with: tail -f $ROOT/se_w${WIDTH}_s0.log"

if [ "$WAIT" = "1" ]; then
  wait
  # Export FP32 flip-TTA probabilities for each finished run. / 导出 FP32 flip-TTA 概率。
  for SEED in $SEEDS; do
    NAME="se_w${WIDTH}_s${SEED}"
    CKPT="$ROOT/proj1/runs/$NAME/final.pt"
    [ -f "$CKPT" ] || CKPT="$ROOT/proj1/runs/$NAME/best.pt"
    mkdir -p "$ROOT/proj1/runs/$NAME/probs"
    python -u predict.py "$CKPT" --data "$ROOT/data/competition" \
      --out "$ROOT/proj1/runs/$NAME/probs/probs.csv" --tta flip --precision fp32
  done
  echo "[se-w${WIDTH}] probabilities exported under runs/se_w${WIDTH}_s*/probs/"
fi
