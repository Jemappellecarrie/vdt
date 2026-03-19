#!/usr/bin/env bash

set -u

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_ROOT="vdt_dev/outputs/online_compare_4gpu"
ENV_NAME="hopper-medium-v2"
SEED="0"
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
  local model_variant="$4"
  local query_mode="$5"
  local run_output_dir="$OUTPUT_ROOT/$run_name"
  local command=(
    "$PYTHON_BIN" -m vdt_dev.runner
    --config "$config_path"
    --output-dir "$run_output_dir"
    --env-name "$ENV_NAME"
    --seed "$SEED"
    --exp-name "hopper_online_compare"
    --experiment-family "online_compare"
    --preset-name "hopper_online_compare"
    --model-variant "$model_variant"
    --attnres-query-mode "$query_mode"
    --skip-if-complete true
    --resume-if-exists true
    --structured-logging true
    --compute-report true
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

launch_run 0 "gpu0_vanilla_dev_online" "vdt_dev/configs/vdt_vanilla/hopper_medium_v2_vanilla_dev_online.yaml" "vanilla_dev" "static"
launch_run 1 "gpu1_bar_online" "vdt_dev/configs/vdt_bar/hopper_medium_v2_bar_online.yaml" "bar" "static"
launch_run 2 "gpu2_vcdr_state_rtg_value_online" "vdt_dev/configs/vdt_vcdr/hopper_medium_v2_vcdr_state_rtg_value_online.yaml" "vcdr" "state_rtg_value"

if [[ "$DRY_RUN" -eq 1 ]]; then
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
