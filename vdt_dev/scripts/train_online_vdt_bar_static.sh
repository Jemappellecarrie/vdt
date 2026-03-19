#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

OUTPUT_DIR="${1:-vdt_dev/outputs/hopper_medium_v2_bar_online}"

python -m vdt_dev.runner \
  --config vdt_dev/configs/vdt_bar/hopper_medium_v2_bar_online.yaml \
  --output-dir "$OUTPUT_DIR"
