# Step 3 Changelog

## Scope

Step 3 adds research infrastructure for the existing VDT / VDT-BAR / VDT-VCDR family without changing the underlying algorithmic objectives.

## Added

- [analysis/schema.py](/Users/zoe/vdt/VDT/vdt_dev/analysis/schema.py)
  - Typed run, metric, debug, and RTG-grid schema with JSON / JSONL helpers.
- [analysis/manifest.py](/Users/zoe/vdt/VDT/vdt_dev/analysis/manifest.py)
  - Run manifest creation, update, discovery, and legacy snapshot fallback.
- [analysis/checkpoint_select.py](/Users/zoe/vdt/VDT/vdt_dev/analysis/checkpoint_select.py)
  - `latest`, `best`, and explicit checkpoint resolution.
- [analysis/collect.py](/Users/zoe/vdt/VDT/vdt_dev/analysis/collect.py)
  - Metric-log summarization and debug artifact reduction.
- [analysis/aggregate.py](/Users/zoe/vdt/VDT/vdt_dev/analysis/aggregate.py)
  - Run discovery and grouped summary export.
- [analysis/plot_learning_curves.py](/Users/zoe/vdt/VDT/vdt_dev/analysis/plot_learning_curves.py)
- [analysis/plot_final_bars.py](/Users/zoe/vdt/VDT/vdt_dev/analysis/plot_final_bars.py)
- [analysis/plot_rtg_alignment.py](/Users/zoe/vdt/VDT/vdt_dev/analysis/plot_rtg_alignment.py)
- [analysis/plot_routing_heatmap.py](/Users/zoe/vdt/VDT/vdt_dev/analysis/plot_routing_heatmap.py)
- [analysis/plot_layer_norms.py](/Users/zoe/vdt/VDT/vdt_dev/analysis/plot_layer_norms.py)
- [analysis/plot_query_norms.py](/Users/zoe/vdt/VDT/vdt_dev/analysis/plot_query_norms.py)
- [analysis/plot_depth_source_usage.py](/Users/zoe/vdt/VDT/vdt_dev/analysis/plot_depth_source_usage.py)
- [analysis/export_tables.py](/Users/zoe/vdt/VDT/vdt_dev/analysis/export_tables.py)
  - Plotting and table-export suite using matplotlib only.
- [experiments/matrix.py](/Users/zoe/vdt/VDT/vdt_dev/experiments/matrix.py)
- [experiments/presets.py](/Users/zoe/vdt/VDT/vdt_dev/experiments/presets.py)
- [experiments/registry.py](/Users/zoe/vdt/VDT/vdt_dev/experiments/registry.py)
  - Preset expansion and planned-manifest generation.
- [configs/experiments](/Users/zoe/vdt/VDT/vdt_dev/configs/experiments)
  - Query-mode, depth, env-sweep, RTG-grid, and sparse-template preset metadata.
- [scripts/launch_querymode_sweep_4gpu.sh](/Users/zoe/vdt/VDT/vdt_dev/scripts/launch_querymode_sweep_4gpu.sh)
- [scripts/launch_depth_sweep_4gpu.sh](/Users/zoe/vdt/VDT/vdt_dev/scripts/launch_depth_sweep_4gpu.sh)
- [scripts/launch_seed_sweep_4gpu.sh](/Users/zoe/vdt/VDT/vdt_dev/scripts/launch_seed_sweep_4gpu.sh)
- [scripts/reeval_rtg_grid.sh](/Users/zoe/vdt/VDT/vdt_dev/scripts/reeval_rtg_grid.sh)
- [scripts/aggregate_results.sh](/Users/zoe/vdt/VDT/vdt_dev/scripts/aggregate_results.sh)
- [scripts/make_all_figures.sh](/Users/zoe/vdt/VDT/vdt_dev/scripts/make_all_figures.sh)
  - Sweep, reevaluation, aggregation, and figure wrappers.
- [README_STEP3.md](/Users/zoe/vdt/VDT/vdt_dev/README_STEP3.md)
  - Step 3 usage guide.

## Updated

- [runner.py](/Users/zoe/vdt/VDT/vdt_dev/runner.py)
  - Structured output layout, manifests, best/latest checkpoints, resumable checkpoints, eval logs, and RTG-grid reevaluation job.
- [iql.py](/Users/zoe/vdt/VDT/vdt_dev/iql.py)
  - Extra scalar logging for Step 3 summaries.
- [utils/debug_hooks.py](/Users/zoe/vdt/VDT/vdt_dev/utils/debug_hooks.py)
  - Summary-friendly routing events in addition to raw tensors.
- [README.md](/Users/zoe/vdt/VDT/vdt_dev/README.md)
  - Step 3 overview and links.

## Tests

- [tests/test_results_schema.py](/Users/zoe/vdt/VDT/vdt_dev/tests/test_results_schema.py)
- [tests/test_manifest_discovery.py](/Users/zoe/vdt/VDT/vdt_dev/tests/test_manifest_discovery.py)
- [tests/test_aggregate_metrics.py](/Users/zoe/vdt/VDT/vdt_dev/tests/test_aggregate_metrics.py)
- [tests/test_plot_scripts_smoke.py](/Users/zoe/vdt/VDT/vdt_dev/tests/test_plot_scripts_smoke.py)
- [tests/test_checkpoint_select.py](/Users/zoe/vdt/VDT/vdt_dev/tests/test_checkpoint_select.py)
- [tests/test_rtg_grid_eval_config.py](/Users/zoe/vdt/VDT/vdt_dev/tests/test_rtg_grid_eval_config.py)

## Preserved

- Step 1 static BAR behavior
- Step 2 query-conditioning behavior
- additive query fusion
- detached `V(s_t)` boundary for `state_rtg_value`
- original actor / critic loss structure
- original sampling algorithm
