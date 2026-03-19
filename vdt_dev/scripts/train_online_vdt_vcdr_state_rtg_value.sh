#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

OUTPUT_DIR="${1:-vdt_dev/outputs/hopper_medium_v2_vcdr_state_rtg_value_online}"

python -m vdt_dev.runner \
  --config vdt_dev/configs/vdt_vcdr/hopper_medium_v2_vcdr_state_rtg_value_online.yaml \
  --output-dir "$OUTPUT_DIR"
