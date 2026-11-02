#!/bin/bash
# Submit prepared candidate CSVs in a fixed order and log every score.
# 按固定顺序提交候选 CSV，并记录每一次的分数。
#
# Why a script: the daily Kaggle quota resets at 00:00 UTC, so the submissions
# happen at an odd local hour; a fixed order + logged results makes the run
# reproducible and prevents losing a slot to a typo or a half-written file.
# 为什么要脚本：Kaggle 每日额度在 UTC 00:00 重置，提交常落在奇怪的时间点；
# 固定顺序 + 落盘日志让这一步可复现，也避免因为路径打错白丢一次额度。
#
# Usage / 用法:
#   bash scripts/submit_candidates.sh submissions/a.csv submissions/b.csv ...
set -u
cd "$(dirname "$0")/.."

COMP=cifar-10-res-net-architecture-challenge-dl
LOG="submissions/log_$(date +%Y%m%d_%H%M%S).txt"
mkdir -p submissions
: > "$LOG"

for f in "$@"; do
  if [ ! -f "$f" ]; then
    echo "[skip] missing $f" | tee -a "$LOG"
    continue
  fi
  # Basic guard: 2000 data rows + correct header. / 基本校验：2000 数据行 + 正确表头。
  rows=$(wc -l < "$f")
  # csv.writer emits CRLF, so strip the carriage return before comparing.
  # csv.writer 输出 CRLF，比较前先去掉回车。
  header=$(head -1 "$f" | tr -d '\r')
  if [ "$rows" -ne 2001 ] || [ "$header" != "id,label" ]; then
    echo "[skip] $f looks malformed (rows=$rows)" | tee -a "$LOG"
    continue
  fi
  echo "=== $(date '+%F %T %Z') submit $f ===" | tee -a "$LOG"
  kaggle competitions submit -c "$COMP" -f "$f" -m "auto: $(basename "$f")" 2>&1 | tail -3 | tee -a "$LOG"
  # Scoring is asynchronous; give it a moment so the next line shows the score.
  # 评分是异步的，稍等再读，才能在同一份日志里看到分数。
  sleep 60
done

echo "=== results ===" | tee -a "$LOG"
kaggle competitions submissions -c "$COMP" 2>&1 | head -10 | tee -a "$LOG"
echo "[done] log: $LOG"
