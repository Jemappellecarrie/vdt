#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

OUTPUT_DIR="${1:-vdt_dev/outputs/hopper_medium_v2_bar_step1}"

python -m vdt_dev.runner \
  --config vdt_dev/configs/vdt_bar/hopper_medium_v2.yaml \
  --output-dir "$OUTPUT_DIR"

# 4-GPU template for the current repo style:
# The upstream codebase is single-process and does not provide DDP/torchrun hooks.
# A practical template today is to launch four independent seeds, one per GPU.
#
# CUDA_VISIBLE_DEVICES=0 python -m vdt_dev.runner --config vdt_dev/configs/vdt_bar/hopper_medium_v2.yaml --seed 123 --output-dir vdt_dev/outputs/hopper_medium_v2_bar_seed123 &
# CUDA_VISIBLE_DEVICES=1 python -m vdt_dev.runner --config vdt_dev/configs/vdt_bar/hopper_medium_v2.yaml --seed 124 --output-dir vdt_dev/outputs/hopper_medium_v2_bar_seed124 &
# CUDA_VISIBLE_DEVICES=2 python -m vdt_dev.runner --config vdt_dev/configs/vdt_bar/hopper_medium_v2.yaml --seed 125 --output-dir vdt_dev/outputs/hopper_medium_v2_bar_seed125 &
# CUDA_VISIBLE_DEVICES=3 python -m vdt_dev.runner --config vdt_dev/configs/vdt_bar/hopper_medium_v2.yaml --seed 126 --output-dir vdt_dev/outputs/hopper_medium_v2_bar_seed126 &
# wait

