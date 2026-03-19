#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

BASE_OUTPUT_DIR="${1:-vdt_dev/outputs/hopper_querymode_sweep_4gpu}"

launch_run() {
  local gpu="$1"
  local config_path="$2"
  local run_output_dir="$3"

  mkdir -p "$run_output_dir"
  local cmd=(
    python -m vdt_dev.runner
    --config "$config_path"
    --output-dir "$run_output_dir"
  )

  printf 'Launching on GPU %s:' "$gpu"
  printf ' CUDA_VISIBLE_DEVICES=%q' "$gpu"
  printf ' %q' "${cmd[@]}"
  printf '\n'

  CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}" \
    >"$run_output_dir/launcher_stdout.log" \
    2>"$run_output_dir/launcher_stderr.log" &
}

launch_run 0 "vdt_dev/configs/vdt_bar/hopper_medium_v2.yaml" "$BASE_OUTPUT_DIR/gpu0_bar_static"
launch_run 1 "vdt_dev/configs/vdt_vcdr/hopper_medium_v2_vcdr_state.yaml" "$BASE_OUTPUT_DIR/gpu1_vcdr_state"
launch_run 2 "vdt_dev/configs/vdt_vcdr/hopper_medium_v2_vcdr_state_rtg.yaml" "$BASE_OUTPUT_DIR/gpu2_vcdr_state_rtg"
launch_run 3 "vdt_dev/configs/vdt_vcdr/hopper_medium_v2_vcdr_state_rtg_value.yaml" "$BASE_OUTPUT_DIR/gpu3_vcdr_state_rtg_value"

wait
