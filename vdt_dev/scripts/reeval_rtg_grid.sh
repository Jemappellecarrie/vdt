#!/usr/bin/env bash

set -eu

PYTHON_BIN="${PYTHON_BIN:-python}"
TARGET="${1:-vdt_dev/outputs}"
SELECTOR="${SELECTOR:-best}"
RTG_GRID="${RTG_GRID:-7200,3600,1800,720}"
EPISODES="${EPISODES:-5}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
REEVAL_DEBUG="${REEVAL_DEBUG:-false}"

CHECKPOINT_PATH="$("$PYTHON_BIN" -m vdt_dev.analysis.checkpoint_select "$TARGET" --selector "$SELECTOR")"

COMMAND=(
  "$PYTHON_BIN" -m vdt_dev.runner
  --job reeval_rtg_grid
  --checkpoint-path "$CHECKPOINT_PATH"
  --checkpoint-selector "$SELECTOR"
  --reeval-rtg-grid "$RTG_GRID"
  --reeval-num-episodes "$EPISODES"
  --reeval-debug "$REEVAL_DEBUG"
)

if [[ -n "$OUTPUT_DIR" ]]; then
  COMMAND+=(--output-dir "$OUTPUT_DIR")
fi

printf '%s\n' "${COMMAND[*]}"
"${COMMAND[@]}"
