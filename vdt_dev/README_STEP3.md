# Step 3 / Step 4 Experiment Stack

This document tracks the experiment infrastructure that began in Step 3 and is now extended by Step 4.

## Step 3 Foundation

Step 3 turned `vdt_dev/` into a full experiment platform for the existing model family:

- stable run manifests and structured logs
- consistent output directories
- `latest.pt` and `best.pt` checkpoint handling
- resume-if-exists and skip-if-complete hooks
- preset expansion for query-mode / depth / env / seed sweeps
- RTG-grid reevaluation
- routing / norm debug summaries
- aggregation into run-level and seed-level tables
- matplotlib-only figures
- CSV / JSON / Markdown / LaTeX export

## Step 4 Extensions

Step 4 keeps the same algorithmic semantics and adds three targeted upgrades:

### 1. Same-Stack Vanilla Baseline

- `model_variant=vanilla_dev`
- implemented via [models/vdt_vanilla_policy.py](/Users/zoe/vdt/VDT/vdt_dev/models/vdt_vanilla_policy.py)
- same runner, logging, checkpoints, manifests, RTG-grid, and aggregation as BAR / VCDR
- standard transformer residual only

### 2. Compute-Aware Depth Infrastructure

- per-run `analysis/compute_summary.json`
- compute fields propagated into `analysis/summary.json` and aggregated CSVs
- fields include:
  - parameter counts
  - approximate FLOP proxies
  - mean train-step time
  - throughput
  - peak GPU memory
  - wall-clock time
- matched-budget preset generation via [experiments/budget.py](/Users/zoe/vdt/VDT/vdt_dev/experiments/budget.py)

### 3. Stronger Runnable Setting

- Step 4 chooses the online-tuning path rather than Maze2D / AntMaze fallback
- the runtime already existed in [runner.py](/Users/zoe/vdt/VDT/vdt_dev/runner.py) through `online_finetune`
- Step 4 adds explicit online configs / scripts / presets / plots so it becomes a first-class experiment line

## Presets

Step 3 presets still exist:

- `hopper_querymode_sweep`
- `hopper_depth_sweep`
- `gym_env_sweep`
- `seed_sweep`
- `maze2d_template`
- `antmaze_template`

Step 4 adds:

- `hopper_same_stack_compare`
- `hopper_same_stack_depth_sweep`
- `hopper_matched_budget_depth_sweep`
- `gym_same_stack_sweep`
- `hopper_online_compare`

Inspect or expand any preset:

```bash
python -m vdt_dev.experiments.registry --preset hopper_same_stack_compare --format commands
python -m vdt_dev.experiments.registry --preset hopper_matched_budget_depth_sweep --format json
python -m vdt_dev.experiments.registry --preset hopper_online_compare --format commands
```

## Scripts

Single-run scripts:

- `train_vdt_vanilla_dev.sh`
- `train_vdt_bar.sh`
- `train_vdt_vcdr_state.sh`
- `train_vdt_vcdr_state_rtg.sh`
- `train_vdt_vcdr_state_rtg_value.sh`
- `train_online_vdt_vanilla_dev.sh`
- `train_online_vdt_bar_static.sh`
- `train_online_vdt_vcdr_state_rtg_value.sh`

Launcher scripts:

- `launch_same_stack_compare_4gpu.sh`
- `launch_querymode_sweep_4gpu.sh`
- `launch_depth_sweep_4gpu.sh`
- `launch_seed_sweep_4gpu.sh`
- `launch_online_compare_4gpu.sh`

## Aggregation and Plotting

Aggregate:

```bash
python -m vdt_dev.analysis.aggregate vdt_dev/outputs --destination-dir vdt_dev/outputs/aggregated
```

Core figures:

```bash
python -m vdt_dev.analysis.plot_final_bars --summary-csv vdt_dev/outputs/aggregated/seed_aggregated_summary.csv
python -m vdt_dev.analysis.plot_online_improvement --summary-csv vdt_dev/outputs/aggregated/seed_aggregated_summary.csv
python -m vdt_dev.analysis.plot_rtg_alignment vdt_dev/outputs/<run>
```

`make_all_figures.sh` now auto-adds the online before/after bar plot when aggregated online metrics are present.

## Output Layout

```text
vdt_dev/outputs/<run_name>/
  metadata/
    config_snapshot.yaml
    command.txt
    git_hash.txt
    run_manifest.json
  checkpoints/
    latest.pt
    best.pt
    offline_step_XXXXX.pt
    online_step_XXXXX.pt
  logs/
    train_metrics.jsonl
    eval_metrics.jsonl
    debug_metrics.jsonl
    rtg_grid_metrics.jsonl
  debug/
    routing/
    norms/
    query/
    eval_rtg_grid/
  analysis/
    summary.json
    compute_summary.json
    plots/
    tables/
```

## Current Boundaries

Still intentionally not implemented:

- advantage-conditioned routing
- uncertainty-conditioned routing
- Q-conditioned routing
- Q / V objective changes
- actor-loss redesign
- Maze2D runtime support
- AntMaze runtime support

Maze2D / AntMaze remain metadata templates only because the current `runner.get_env_metadata()` runtime path supports Gym medium hopper / walker2d / halfcheetah only.
