#!/usr/bin/env bash
# Run the final inference-configuration comparison on `final.pt`.
# 在 `final.pt` 上跑最终的推理配置对比。
#
# Why a script instead of a one-off command:
#   The training run rewrites `runs/*/last.pt` every epoch. The comparison must
#   be run on the FINAL checkpoint, and it must take ONE snapshot so that every
#   weight variant is measured at the same epoch.
#   训练进程每个 epoch 都会覆写 last.pt。对比必须在最终 checkpoint 上跑，
#   且必须只取一次快照，使各权重变体处于同一 epoch。
#
# Usage / 用法:
#   bash scripts/run_refine_final.sh [checkpoint]
#
# The `--bn-recal` value is deliberately `none`: per the group's conclusion the
# BatchNorm recalibration showed NO domain effect (the controlled experiment
# attributed at most 0.2 points to the photometric shift, against a 6.9-point
# gap), so BN recalibration is not part of the final submission path. The
# transductive variant is available for diagnostics via `--bn-recal test`.
# `--bn-recal` 刻意设为 `none`：BN 重校准未显示任何域效应，因此不进入最终提交
# 路径。转导式版本仅供诊断（`--bn-recal test`）。

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CKPT="${1:-runs/wrn22x4_s0/final.pt}"
PY="$ROOT/.venv/bin/python"

if [[ ! -f "$CKPT" ]]; then
  echo "[run_refine_final] checkpoint not found: $CKPT" >&2
  echo "[run_refine_final] 当前 checkpoint 不存在：$CKPT" >&2
  exit 1
fi

echo "[run_refine_final] checkpoint: $CKPT"
"$PY" src/refine.py \
  --ckpt "$CKPT" \
  --weights both \
  --bn-source train,train_matched,test \
  --bn-recal none \
  --out-md "$ROOT/analysis/refine.md" \
  --out-json "$ROOT/analysis/refine.json"

echo "[run_refine_final] done -> analysis/refine.md, analysis/refine.json"
