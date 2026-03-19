#!/usr/bin/env bash

set -u

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_ROOT="vdt_dev/outputs/depth_sweep_4gpu"
ENV_NAME="hopper-medium-v2"
SEED="0"
LAYERS_STATIC=("6" "12")
LAYERS_VALUE=("6" "12")
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --env-name)
      ENV_NAME="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    *)
      OUTPUT_ROOT="$1"
      shift
      ;;
  esac
done

mkdir -p "$OUTPUT_ROOT"
PIDS=()
RUN_NAMES=()
EXIT_CODES=()

launch_run() {
  local gpu_id="$1"
  local run_name="$2"
  local config_path="$3"
  local query_mode="$4"
  local num_layers="$5"
  local run_output_dir="$OUTPUT_ROOT/$run_name"
  local command=(
    "$PYTHON_BIN" -m vdt_dev.runner
    --config "$config_path"
    --output-dir "$run_output_dir"
    --env-name "$ENV_NAME"
    --seed "$SEED"
    --exp-name "hopper_depth_sweep"
    --experiment-family "depth_sweep"
    --preset-name "hopper_depth_sweep"
    --attnres-query-mode "$query_mode"
    --n-layer "$num_layers"
    --skip-if-complete true
    --resume-if-exists true
    --structured-logging true
  )

  printf 'GPU %s :: %s\n' "$gpu_id" "CUDA_VISIBLE_DEVICES=$gpu_id ${command[*]}"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    return
  fi

  mkdir -p "$run_output_dir"
  CUDA_VISIBLE_DEVICES="$gpu_id" "${command[@]}" \
    >"$run_output_dir/launcher_stdout.log" \
    2>"$run_output_dir/launcher_stderr.log" &
  PIDS+=("$!")
  RUN_NAMES+=("$run_name")
}

launch_run 0 "gpu0_static_L${LAYERS_STATIC[0]}" "vdt_dev/configs/vdt_bar/hopper_medium_v2.yaml" "static" "${LAYERS_STATIC[0]}"
launch_run 1 "gpu1_static_L${LAYERS_STATIC[1]}" "vdt_dev/configs/vdt_bar/hopper_medium_v2.yaml" "static" "${LAYERS_STATIC[1]}"
launch_run 2 "gpu2_state_rtg_value_L${LAYERS_VALUE[0]}" "vdt_dev/configs/vdt_vcdr/hopper_medium_v2_vcdr_state_rtg_value.yaml" "state_rtg_value" "${LAYERS_VALUE[0]}"
launch_run 3 "gpu3_state_rtg_value_L${LAYERS_VALUE[1]}" "vdt_dev/configs/vdt_vcdr/hopper_medium_v2_vcdr_state_rtg_value.yaml" "state_rtg_value" "${LAYERS_VALUE[1]}"

if [[ "$DRY_RUN" -eq 1 ]]; then
  printf '\nEdit LAYERS_STATIC/LAYERS_VALUE in this script or pass custom commands if you want an 18-layer pass.\n'
  exit 0
fi

FAILED=0
for index in "${!PIDS[@]}"; do
  pid="${PIDS[$index]}"
  run_name="${RUN_NAMES[$index]}"
  if wait "$pid"; then
    EXIT_CODES+=(0)
    printf 'SUCCESS :: %s\n' "$run_name"
  else
    EXIT_CODES+=($?)
    FAILED=1
    printf 'FAILED :: %s\n' "$run_name"
  fi
done

printf '\nSummary\n'
for index in "${!RUN_NAMES[@]}"; do
  printf '  %s -> exit %s\n' "${RUN_NAMES[$index]}" "${EXIT_CODES[$index]}"
done

exit "$FAILED"
