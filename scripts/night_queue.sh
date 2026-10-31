#!/bin/bash
# Overnight experiment queue: keeps the Mac busy right after the current run ends.
# 夜间实验队列：当前训练一结束就自动接着跑，机器不空转。
#
# Why these three: ensembling diverse models is the only lever that reliably helps
# on an out-of-distribution test set, so we want 2-3 members with different
# capacity/seed rather than one slightly better model.
# 为什么要这三个：OOD 测试集上只有「集成多样模型」能稳定见效，所以宁要 2-3 个
# 不同容量/种子的成员，也不要单个稍微更强的大模型。
#
# Usage / 用法:  nohup bash scripts/night_queue.sh > runs/night_queue.log 2>&1 &

set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python

# 1) Wait for the in-flight run (wrn22x4_s0) to disappear.
# 1) 等当前训练（wrn22x4_s0）退出。
while pgrep -f "train.py .*--out runs/wrn22x4_s0" > /dev/null 2>&1; do sleep 60; done
echo "[queue] $(date '+%F %T') wrn22x4_s0 finished, starting E2"

# 2) E2: different capacity (2.78M) + different seed => decorrelated errors.
# 2) E2：不同容量（2.78M）+ 不同种子 => 错误去相关。
$PY -u src/train.py --model preact16x4 --seed 1 --epochs 300 --batch-size 512 \
    --lr 0.25 --aug strong --mix both --ema-decay 0.999 --out runs/wrn16x4_s1 --workers 4
echo "[queue] $(date '+%F %T') E2 done, starting E3"

# 3) E3: smallest/fastest member, still useful in an ensemble.
# 3) E3：最小最快的成员，集成里仍然有用。
$PY -u src/train.py --model preact28x2 --seed 2 --epochs 300 --batch-size 512 \
    --lr 0.25 --aug strong --mix both --ema-decay 0.999 --out runs/wrn28x2_s2 --workers 4
echo "[queue] $(date '+%F %T') E3 done, queue empty"
