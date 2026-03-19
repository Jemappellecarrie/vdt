#!/usr/bin/env bash

set -eu

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_ROOT="${1:-vdt_dev/outputs}"
DESTINATION_DIR="${2:-vdt_dev/outputs/aggregated}"

COMMAND=(
  "$PYTHON_BIN" -m vdt_dev.analysis.aggregate
  "$OUTPUT_ROOT"
  --destination-dir "$DESTINATION_DIR"
)

printf '%s\n' "${COMMAND[*]}"
"${COMMAND[@]}"
