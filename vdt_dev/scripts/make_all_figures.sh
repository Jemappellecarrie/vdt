#!/usr/bin/env bash

set -eu

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_ROOT="${1:-vdt_dev/outputs}"
AGGREGATE_DIR="${2:-vdt_dev/outputs/aggregated}"
FIGURE_DIR="${3:-vdt_dev/paper_figures}"
RUN_DIR="${RUN_DIR:-}"

"$PYTHON_BIN" -m vdt_dev.analysis.aggregate "$OUTPUT_ROOT" --destination-dir "$AGGREGATE_DIR"
"$PYTHON_BIN" -m vdt_dev.analysis.plot_final_bars --summary-csv "$AGGREGATE_DIR/seed_aggregated_summary.csv" --output-dir "$FIGURE_DIR"

if [[ -f "$AGGREGATE_DIR/seed_aggregated_summary.csv" ]] && grep -q "best_online_eval_return_normalized_mean" "$AGGREGATE_DIR/seed_aggregated_summary.csv"; then
  "$PYTHON_BIN" -m vdt_dev.analysis.plot_online_improvement --summary-csv "$AGGREGATE_DIR/seed_aggregated_summary.csv" --output-dir "$FIGURE_DIR"
fi

if [[ -n "$RUN_DIR" ]]; then
  "$PYTHON_BIN" -m vdt_dev.analysis.plot_learning_curves "$RUN_DIR" --output-dir "$FIGURE_DIR"
  "$PYTHON_BIN" -m vdt_dev.analysis.plot_routing_heatmap "$RUN_DIR" --output-dir "$FIGURE_DIR"
  "$PYTHON_BIN" -m vdt_dev.analysis.plot_layer_norms "$RUN_DIR" --output-dir "$FIGURE_DIR"
  "$PYTHON_BIN" -m vdt_dev.analysis.plot_query_norms "$RUN_DIR" --output-dir "$FIGURE_DIR"
  "$PYTHON_BIN" -m vdt_dev.analysis.plot_depth_source_usage "$RUN_DIR" --output-dir "$FIGURE_DIR"
  "$PYTHON_BIN" -m vdt_dev.analysis.plot_rtg_alignment "$RUN_DIR" --output-dir "$FIGURE_DIR"
fi
