#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

OUTPUT_DIR="${1:-vdt_dev/outputs/hopper_medium_v2_vanilla_dev}"

python -m vdt_dev.runner \
  --config vdt_dev/configs/vdt_vanilla/hopper_medium_v2_vanilla_dev.yaml \
  --output-dir "$OUTPUT_DIR"
