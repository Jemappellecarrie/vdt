#!/usr/bin/env bash

set -u

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_ROOT="vdt_dev/outputs/seed_sweep_4gpu"
ENV_NAME="hopper-medium-v2"
QUERY_MODE="state_rtg_value"
CONFIG_PATH="vdt_dev/configs/vdt_vcdr/hopper_medium_v2_vcdr_state_rtg_value.yaml"
SEEDS=("0" "1" "2" "3")
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
    --query-mode)
      QUERY_MODE="$2"
      shift 2
      ;;
    --config-path)
      CONFIG_PATH="$2"
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
  local seed="$2"
  local run_name="gpu${gpu_id}_seed${seed}"
  local run_output_dir="$OUTPUT_ROOT/$run_name"
  local command=(
    "$PYTHON_BIN" -m vdt_dev.runner
    --config "$CONFIG_PATH"
    --output-dir "$run_output_dir"
    --env-name "$ENV_NAME"
    --seed "$seed"
    --exp-name "seed_sweep"
    --experiment-family "seed_sweep"
    --preset-name "seed_sweep"
    --attnres-query-mode "$QUERY_MODE"
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

for index in "${!SEEDS[@]}"; do
  launch_run "$index" "${SEEDS[$index]}"
done

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
