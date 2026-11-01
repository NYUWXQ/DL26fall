#!/bin/bash
# After the faithful-recipe run finishes, start a second seed for ensembling.
# 忠实配方跑完后接着跑第二个 seed，供明天做集成。
# 集成需要错误去相关，所以同配方不同 seed 是最省事的多样性来源。
set -u
cd "$(dirname "$0")/.."
while pgrep -f "wrn16x5.25_full --" > /dev/null 2>&1; do sleep 120; done
echo "[queue2] $(date '+%F %T') E4 finished, starting E5 (seed 1)"
.venv/bin/python -u src/train.py --model preact16x5.25 --seed 1 \
  --batch-size 128 --lr 0.1 --weight-decay 5e-4 \
  --randaug-m 6 --erase-p 0.0 --aug strong \
  --mix both --mix-always --mixup-alpha 0.2 --cutmix-alpha 1.0 \
  --label-smoothing 0.0 --ema-dynamic \
  --full-data --epochs 300 --workers 4 \
  --out runs/wrn16x5.25_full_s1
echo "[queue2] $(date '+%F %T') E5 done"
