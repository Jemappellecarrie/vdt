# VDT Dev Stack

`vdt_dev/` is the research-safe experiment stack built on top of the preserved root VDT baseline.

Current staged scope:

- Step 1: `VDT-BAR`
- Step 2: `VDT-VCDR`
- Step 3: manifests / aggregation / plotting / RTG-grid reevaluation
- Step 4: same-stack `vanilla_dev` baseline, compute-aware depth reporting, and dev-stack online tuning entrypoints

The original root baseline remains untouched and is still the separate paper-reproduction reference.

## What Step 4 Adds

- a same-stack `vanilla_dev` policy inside `vdt_dev/`
- explicit `model_variant` tracking: `vanilla_dev`, `bar`, `vcdr`
- structured compute reports:
  - parameter count
  - trainable parameter count
  - approximate FLOP proxies
  - step time / throughput
  - peak GPU memory
  - wall-clock runtime
- matched-budget depth sweep preset generation
- Hopper same-stack offline-to-online comparison configs and launchers
- offline/online-aware aggregation and before-vs-after plotting

Step 4 still does not add:

- advantage-conditioned routing
- uncertainty-conditioned routing
- Q-conditioned routing
- Q / V objective changes
- actor-loss redesign
- sampling / RTG semantics changes
- runnable Maze2D / AntMaze paths

## Main Files

- [runner.py](/Users/zoe/vdt/VDT/vdt_dev/runner.py)
  - same-stack train / eval / RTG-grid / online-tuning runtime
- [models/vdt_vanilla_policy.py](/Users/zoe/vdt/VDT/vdt_dev/models/vdt_vanilla_policy.py)
  - same-stack vanilla dev baseline
- [model_variants.py](/Users/zoe/vdt/VDT/vdt_dev/model_variants.py)
  - explicit variant resolution for manifests and aggregation
- [utils/compute.py](/Users/zoe/vdt/VDT/vdt_dev/utils/compute.py)
  - compute accounting and FLOP proxies
- [experiments/presets.py](/Users/zoe/vdt/VDT/vdt_dev/experiments/presets.py)
  - Step 4 preset expansion including same-stack / matched-budget / online compare
- [analysis/aggregate.py](/Users/zoe/vdt/VDT/vdt_dev/analysis/aggregate.py)
  - run discovery, compute-aware aggregation, seed tables
- [analysis/plot_online_improvement.py](/Users/zoe/vdt/VDT/vdt_dev/analysis/plot_online_improvement.py)
  - offline-vs-online before/after bars

## Common Commands

Vanilla dev baseline:

```bash
bash vdt_dev/scripts/train_vdt_vanilla_dev.sh
```

Same-stack 3-way comparison:

```bash
bash vdt_dev/scripts/launch_same_stack_compare_4gpu.sh --dry-run
bash vdt_dev/scripts/launch_same_stack_compare_4gpu.sh
```

Matched-budget depth sweep command expansion:

```bash
python -m vdt_dev.experiments.registry --preset hopper_matched_budget_depth_sweep --format commands
```

Online comparison:

```bash
bash vdt_dev/scripts/launch_online_compare_4gpu.sh --dry-run
bash vdt_dev/scripts/launch_online_compare_4gpu.sh
```

Aggregate and figures:

```bash
bash vdt_dev/scripts/aggregate_results.sh vdt_dev/outputs vdt_dev/outputs/aggregated
bash vdt_dev/scripts/make_all_figures.sh vdt_dev/outputs vdt_dev/outputs/aggregated vdt_dev/paper_figures
```

## Docs

- [README_STEP3.md](/Users/zoe/vdt/VDT/vdt_dev/README_STEP3.md)
  - Step 3 foundation plus Step 4 additions
- [EXPERIMENT_STAGE_SUMMARY.md](/Users/zoe/vdt/VDT/vdt_dev/EXPERIMENT_STAGE_SUMMARY.md)
  - Chinese stage summary and experiment priorities
- [EXPERIMENT_COMMANDS_STAGE.md](/Users/zoe/vdt/VDT/vdt_dev/EXPERIMENT_COMMANDS_STAGE.md)
  - Chinese command sheet with runnable training / aggregation / plotting commands

## Tests

Run the full dev suite:

```bash
pytest vdt_dev/tests -q
```

Focused Step 4 checks:

```bash
pytest \
  vdt_dev/tests/test_vanilla_dev_policy.py \
  vdt_dev/tests/test_compute_reporting.py \
  vdt_dev/tests/test_experiment_presets_step4.py -q
```
