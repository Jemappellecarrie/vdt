# 阶段性实验命令清单

本文档只列当前 repo 中真实存在、并且与当前 Step 4 代码状态一致的命令。优先使用现成脚本；当 sweep 组合较多且仓库里没有专门 launcher 时，使用 `python -m vdt_dev.experiments.registry --format commands` 生成真实的 `python -m vdt_dev.runner ...` 命令文件再执行。

## 1. 运行前检查

### 1.1 全量回归测试

```bash
pytest vdt_dev/tests -q
```

预期产物：

- 终端显示全部通过

### 1.2 查看 Step 4 presets 的真实展开结果

```bash
python -m vdt_dev.experiments.registry --preset hopper_same_stack_compare --format commands
python -m vdt_dev.experiments.registry --preset hopper_same_stack_depth_sweep --format commands
python -m vdt_dev.experiments.registry --preset hopper_matched_budget_depth_sweep --format commands
python -m vdt_dev.experiments.registry --preset hopper_online_compare --format commands
```

预期产物：

- 终端打印真实的 `python -m vdt_dev.runner ...` 命令

### 1.3 若需要预写 planned manifests

```bash
python -m vdt_dev.experiments.registry \
  --preset hopper_matched_budget_depth_sweep \
  --write-planned-manifests \
  --format commands
```

预期产物：

- 对应 output root 下每个 run 的 `metadata/run_manifest.json`

## 2. 实验总览命令表

| 实验项 | 推荐入口 | 主要输出目录 | 主要产物 |
| --- | --- | --- | --- |
| P0 回归 / sanity | `pytest vdt_dev/tests -q` | 无 | regression 信号 |
| P1 same-stack attribution | `bash vdt_dev/scripts/launch_same_stack_compare_4gpu.sh` | `vdt_dev/outputs/same_stack_compare_4gpu` | vanilla / BAR / VCDR 三条线 |
| P2 query-mode sweep | `bash vdt_dev/scripts/launch_querymode_sweep_4gpu.sh` | `vdt_dev/outputs/querymode_sweep_4gpu` | static / state / state_rtg / state_rtg_value |
| P3 same-stack equal-width depth | registry 命令文件 | `vdt_dev/outputs/hopper_same_stack_depth_sweep` | 3 变体 x 3 深度 |
| P4 matched-budget depth | registry 命令文件 | `vdt_dev/outputs/hopper_matched_budget_depth_sweep` | compute-aware depth 对照 |
| P5 seed sweep | `bash vdt_dev/scripts/launch_seed_sweep_4gpu.sh` | 自定义 `vdt_dev/outputs/seed_sweep_*` | 多 seed 稳定性 |
| P6 Gym same-stack env sweep | registry 命令文件 | `vdt_dev/outputs/gym_same_stack_sweep` | Hopper / Walker2d / HalfCheetah |
| P7 online compare | `bash vdt_dev/scripts/launch_online_compare_4gpu.sh` | `vdt_dev/outputs/online_compare_4gpu` | offline-to-online before/after |
| P8 RTG-grid | `bash vdt_dev/scripts/reeval_rtg_grid.sh` | run root 或自定义 reeval root | alignment curves |
| P9 机制分析 | `python -m vdt_dev.analysis.plot_*` | `vdt_dev/paper_figures/*` | routing / norm / alignment 图 |

## 3. P0：回归 / sanity / dry-run

### 3.1 全量测试

```bash
pytest vdt_dev/tests -q
```

### 3.2 Vanilla dev sanity

```bash
bash vdt_dev/scripts/train_vdt_vanilla_dev_sanity.sh
```

输出目录：

- `vdt_dev/outputs/hopper_medium_v2_vanilla_dev_sanity`

### 3.3 BAR sanity

```bash
bash vdt_dev/scripts/train_vdt_bar_sanity.sh
```

输出目录：

- `vdt_dev/outputs/hopper_medium_v2_bar_sanity`

### 3.4 新旧 launcher dry-run

```bash
bash vdt_dev/scripts/launch_same_stack_compare_4gpu.sh --dry-run
bash vdt_dev/scripts/launch_querymode_sweep_4gpu.sh --dry-run
bash vdt_dev/scripts/launch_depth_sweep_4gpu.sh --dry-run
bash vdt_dev/scripts/launch_seed_sweep_4gpu.sh --dry-run
bash vdt_dev/scripts/launch_online_compare_4gpu.sh --dry-run
```

## 4. P1：Hopper same-stack attribution compare

### 4.1 训练

```bash
bash vdt_dev/scripts/launch_same_stack_compare_4gpu.sh \
  --output-root vdt_dev/outputs/same_stack_compare_4gpu
```

输出目录：

- `vdt_dev/outputs/same_stack_compare_4gpu/gpu0_vanilla_dev_static`
- `vdt_dev/outputs/same_stack_compare_4gpu/gpu1_bar_static`
- `vdt_dev/outputs/same_stack_compare_4gpu/gpu2_vcdr_state_rtg_value`

### 4.2 聚合

```bash
bash vdt_dev/scripts/aggregate_results.sh \
  vdt_dev/outputs/same_stack_compare_4gpu \
  vdt_dev/outputs/aggregated/same_stack_compare_4gpu
```

### 4.3 主结果图与表

```bash
python -m vdt_dev.analysis.plot_final_bars \
  --summary-csv vdt_dev/outputs/aggregated/same_stack_compare_4gpu/seed_aggregated_summary.csv \
  --output-dir vdt_dev/paper_figures/same_stack_compare

python -m vdt_dev.analysis.export_tables \
  --summary-csv vdt_dev/outputs/aggregated/same_stack_compare_4gpu/seed_aggregated_summary.csv \
  --output-dir vdt_dev/paper_tables/same_stack_compare
```

预期产物：

- `seed_aggregated_summary.csv`
- `best_per_group_summary.csv`
- `paper_figures/same_stack_compare/final_bars.*`

## 5. P2：Hopper query-mode sweep

### 5.1 训练

```bash
bash vdt_dev/scripts/launch_querymode_sweep_4gpu.sh \
  --output-root vdt_dev/outputs/querymode_sweep_4gpu
```

### 5.2 聚合

```bash
bash vdt_dev/scripts/aggregate_results.sh \
  vdt_dev/outputs/querymode_sweep_4gpu \
  vdt_dev/outputs/aggregated/querymode_sweep_4gpu
```

### 5.3 出图与表格

```bash
python -m vdt_dev.analysis.plot_final_bars \
  --summary-csv vdt_dev/outputs/aggregated/querymode_sweep_4gpu/seed_aggregated_summary.csv \
  --output-dir vdt_dev/paper_figures/querymode_sweep

python -m vdt_dev.analysis.export_tables \
  --summary-csv vdt_dev/outputs/aggregated/querymode_sweep_4gpu/seed_aggregated_summary.csv \
  --output-dir vdt_dev/paper_tables/querymode_sweep
```

## 6. P3：Hopper same-stack equal-width depth sweep

仓库当前没有专门的 same-stack depth 4-GPU launcher，因此推荐先由 registry 生成真实命令文件，再直接执行。

### 6.1 生成命令文件

```bash
mkdir -p vdt_dev/outputs/command_lists
python -m vdt_dev.experiments.registry \
  --preset hopper_same_stack_depth_sweep \
  --output-root vdt_dev/outputs \
  --write-planned-manifests \
  --format commands \
  > vdt_dev/outputs/command_lists/hopper_same_stack_depth_sweep.sh
```

### 6.2 执行命令文件

```bash
bash vdt_dev/outputs/command_lists/hopper_same_stack_depth_sweep.sh
```

主要输出目录：

- `vdt_dev/outputs/hopper_same_stack_depth_sweep/hopper_medium_v2__vanilla_dev__static__L6__seed0`
- `vdt_dev/outputs/hopper_same_stack_depth_sweep/hopper_medium_v2__vanilla_dev__static__L12__seed0`
- `vdt_dev/outputs/hopper_same_stack_depth_sweep/hopper_medium_v2__vanilla_dev__static__L18__seed0`
- `vdt_dev/outputs/hopper_same_stack_depth_sweep/hopper_medium_v2__bar__static__L6__seed0`
- `vdt_dev/outputs/hopper_same_stack_depth_sweep/hopper_medium_v2__bar__static__L12__seed0`
- `vdt_dev/outputs/hopper_same_stack_depth_sweep/hopper_medium_v2__bar__static__L18__seed0`
- `vdt_dev/outputs/hopper_same_stack_depth_sweep/hopper_medium_v2__vcdr__state_rtg_value__L6__seed0`
- `vdt_dev/outputs/hopper_same_stack_depth_sweep/hopper_medium_v2__vcdr__state_rtg_value__L12__seed0`
- `vdt_dev/outputs/hopper_same_stack_depth_sweep/hopper_medium_v2__vcdr__state_rtg_value__L18__seed0`

### 6.3 聚合

```bash
python -m vdt_dev.analysis.aggregate \
  vdt_dev/outputs/hopper_same_stack_depth_sweep \
  --destination-dir vdt_dev/outputs/aggregated/hopper_same_stack_depth_sweep
```

### 6.4 性能图与 compute 图

```bash
python -m vdt_dev.analysis.plot_final_bars \
  --summary-csv vdt_dev/outputs/aggregated/hopper_same_stack_depth_sweep/seed_aggregated_summary.csv \
  --output-dir vdt_dev/paper_figures/hopper_same_stack_depth_sweep \
  --output-stem performance_bars

python -m vdt_dev.analysis.plot_final_bars \
  --summary-csv vdt_dev/outputs/aggregated/hopper_same_stack_depth_sweep/seed_aggregated_summary.csv \
  --metric-key total_param_count_mean \
  --output-dir vdt_dev/paper_figures/hopper_same_stack_depth_compute \
  --output-stem total_param_count_bars

python -m vdt_dev.analysis.plot_final_bars \
  --summary-csv vdt_dev/outputs/aggregated/hopper_same_stack_depth_sweep/seed_aggregated_summary.csv \
  --metric-key approx_total_train_step_flops_mean \
  --output-dir vdt_dev/paper_figures/hopper_same_stack_depth_compute \
  --output-stem approx_train_flops_bars

python -m vdt_dev.analysis.export_tables \
  --summary-csv vdt_dev/outputs/aggregated/hopper_same_stack_depth_sweep/seed_aggregated_summary.csv \
  --output-dir vdt_dev/paper_tables/hopper_same_stack_depth_sweep
```

## 7. P4：Hopper matched-budget depth sweep

matched-budget preset 会自动写入实际 `embed_dim` / `n_head` / `budget_actual_gap_pct`。

### 7.1 生成命令文件

```bash
mkdir -p vdt_dev/outputs/command_lists
python -m vdt_dev.experiments.registry \
  --preset hopper_matched_budget_depth_sweep \
  --output-root vdt_dev/outputs \
  --write-planned-manifests \
  --format commands \
  > vdt_dev/outputs/command_lists/hopper_matched_budget_depth_sweep.sh
```

### 7.2 执行命令文件

```bash
bash vdt_dev/outputs/command_lists/hopper_matched_budget_depth_sweep.sh
```

主要输出目录示例：

- `vdt_dev/outputs/hopper_matched_budget_depth_sweep/hopper_medium_v2__vanilla_dev__static__L12__seed0__E176_H1`
- `vdt_dev/outputs/hopper_matched_budget_depth_sweep/hopper_medium_v2__bar__static__L18__seed0__E144_H1`
- `vdt_dev/outputs/hopper_matched_budget_depth_sweep/hopper_medium_v2__vcdr__state_rtg_value__L18__seed0__E144_H1`

### 7.3 聚合与 compute-aware 表格

```bash
python -m vdt_dev.analysis.aggregate \
  vdt_dev/outputs/hopper_matched_budget_depth_sweep \
  --destination-dir vdt_dev/outputs/aggregated/hopper_matched_budget_depth_sweep

python -m vdt_dev.analysis.export_tables \
  --summary-csv vdt_dev/outputs/aggregated/hopper_matched_budget_depth_sweep/seed_aggregated_summary.csv \
  --output-dir vdt_dev/paper_tables/hopper_matched_budget_depth_sweep

python -m vdt_dev.analysis.plot_final_bars \
  --summary-csv vdt_dev/outputs/aggregated/hopper_matched_budget_depth_sweep/seed_aggregated_summary.csv \
  --output-dir vdt_dev/paper_figures/hopper_matched_budget_depth_sweep \
  --output-stem performance_bars
```

重点关注字段：

- `budget_target_params_mean`
- `budget_actual_gap_pct_mean`
- `total_param_count_mean`
- `approx_total_train_step_flops_mean`

## 8. P5：关键结果 seed sweep

### 8.1 先做 `vcdr(state_rtg_value)` 的 4-seed sweep

```bash
bash vdt_dev/scripts/launch_seed_sweep_4gpu.sh \
  --output-root vdt_dev/outputs/seed_sweep_vcdr_state_rtg_value_L6 \
  --env-name hopper-medium-v2 \
  --query-mode state_rtg_value \
  --config-path vdt_dev/configs/vdt_vcdr/hopper_medium_v2_vcdr_state_rtg_value.yaml
```

### 8.2 如需同栈 vanilla_dev seed sweep

```bash
bash vdt_dev/scripts/launch_seed_sweep_4gpu.sh \
  --output-root vdt_dev/outputs/seed_sweep_vanilla_dev_L6 \
  --env-name hopper-medium-v2 \
  --query-mode static \
  --config-path vdt_dev/configs/vdt_vanilla/hopper_medium_v2_vanilla_dev.yaml
```

### 8.3 聚合

```bash
python -m vdt_dev.analysis.aggregate \
  vdt_dev/outputs/seed_sweep_vcdr_state_rtg_value_L6 \
  vdt_dev/outputs/seed_sweep_vanilla_dev_L6 \
  --destination-dir vdt_dev/outputs/aggregated/seed_sweeps_stage4
```

## 9. P6：Gym same-stack env sweep

说明：

- 当前用 Hopper base config 覆写 `--env-name` 是安全的
- 原因是 [runner.py](/Users/zoe/vdt/VDT/vdt_dev/runner.py) 会按 `env_name` 动态重算 env targets、reward scale、observation / action dimensions
- 训练命令仍会写入各自独立 output dir，不会污染 Hopper 结果

### 9.1 生成命令文件

```bash
mkdir -p vdt_dev/outputs/command_lists
python -m vdt_dev.experiments.registry \
  --preset gym_same_stack_sweep \
  --output-root vdt_dev/outputs \
  --write-planned-manifests \
  --format commands \
  > vdt_dev/outputs/command_lists/gym_same_stack_sweep.sh
```

### 9.2 执行命令文件

```bash
bash vdt_dev/outputs/command_lists/gym_same_stack_sweep.sh
```

### 9.3 聚合

```bash
python -m vdt_dev.analysis.aggregate \
  vdt_dev/outputs/gym_same_stack_sweep \
  --destination-dir vdt_dev/outputs/aggregated/gym_same_stack_sweep

python -m vdt_dev.analysis.export_tables \
  --summary-csv vdt_dev/outputs/aggregated/gym_same_stack_sweep/seed_aggregated_summary.csv \
  --output-dir vdt_dev/paper_tables/gym_same_stack_sweep
```

## 10. P7：Hopper online compare

### 10.1 训练

```bash
bash vdt_dev/scripts/launch_online_compare_4gpu.sh \
  --output-root vdt_dev/outputs/online_compare_4gpu
```

输出目录：

- `vdt_dev/outputs/online_compare_4gpu/gpu0_vanilla_dev_online`
- `vdt_dev/outputs/online_compare_4gpu/gpu1_bar_online`
- `vdt_dev/outputs/online_compare_4gpu/gpu2_vcdr_state_rtg_value_online`

### 10.2 聚合

```bash
python -m vdt_dev.analysis.aggregate \
  vdt_dev/outputs/online_compare_4gpu \
  --destination-dir vdt_dev/outputs/aggregated/online_compare_4gpu
```

### 10.3 offline / online before-vs-after 图与表

```bash
python -m vdt_dev.analysis.plot_online_improvement \
  --summary-csv vdt_dev/outputs/aggregated/online_compare_4gpu/seed_aggregated_summary.csv \
  --output-dir vdt_dev/paper_figures/online_compare

python -m vdt_dev.analysis.plot_final_bars \
  --summary-csv vdt_dev/outputs/aggregated/online_compare_4gpu/seed_aggregated_summary.csv \
  --metric-key best_online_eval_return_normalized_mean \
  --output-dir vdt_dev/paper_figures/online_compare \
  --output-stem best_online_bars

python -m vdt_dev.analysis.export_tables \
  --summary-csv vdt_dev/outputs/aggregated/online_compare_4gpu/seed_aggregated_summary.csv \
  --output-dir vdt_dev/paper_tables/online_compare
```

## 11. P8：RTG-grid reevaluation

### 11.1 对某个 best checkpoint 所在 run 做 RTG-grid

```bash
RTG_GRID=7200,3600,1800,720 \
EPISODES=5 \
bash vdt_dev/scripts/reeval_rtg_grid.sh \
  vdt_dev/outputs/same_stack_compare_4gpu/gpu2_vcdr_state_rtg_value
```

### 11.2 直接 runner 入口

```bash
python -m vdt_dev.runner \
  --job reeval_rtg_grid \
  --checkpoint-path vdt_dev/outputs/same_stack_compare_4gpu/gpu2_vcdr_state_rtg_value/checkpoints/best.pt \
  --output-dir vdt_dev/outputs/rtg_grid_vcdr_same_stack \
  --reeval-rtg-grid 7200,3600,1800,720 \
  --reeval-num-episodes 5
```

### 11.3 出图

```bash
python -m vdt_dev.analysis.plot_rtg_alignment \
  vdt_dev/outputs/rtg_grid_vcdr_same_stack \
  --output-dir vdt_dev/paper_figures/rtg_alignment
```

## 12. P9：机制分析图

以下命令建议对 `state_rtg_value` 的 best run 和 `bar(static)` 对照 run 各跑一套。

```bash
python -m vdt_dev.analysis.plot_routing_heatmap \
  vdt_dev/outputs/querymode_sweep_4gpu/gpu3_vcdr_state_rtg_value \
  --output-dir vdt_dev/paper_figures/mechanism_vcdr

python -m vdt_dev.analysis.plot_layer_norms \
  vdt_dev/outputs/querymode_sweep_4gpu/gpu3_vcdr_state_rtg_value \
  --output-dir vdt_dev/paper_figures/mechanism_vcdr

python -m vdt_dev.analysis.plot_query_norms \
  vdt_dev/outputs/querymode_sweep_4gpu/gpu3_vcdr_state_rtg_value \
  --output-dir vdt_dev/paper_figures/mechanism_vcdr

python -m vdt_dev.analysis.plot_depth_source_usage \
  vdt_dev/outputs/querymode_sweep_4gpu/gpu3_vcdr_state_rtg_value \
  --output-dir vdt_dev/paper_figures/mechanism_vcdr
```

## 13. 收尾与复用命令

### 13.1 找 `best` checkpoint

```bash
python -m vdt_dev.analysis.checkpoint_select \
  vdt_dev/outputs/same_stack_compare_4gpu/gpu2_vcdr_state_rtg_value \
  --selector best
```

### 13.2 只重跑 eval

```bash
python -m vdt_dev.runner \
  --job eval \
  --checkpoint-path vdt_dev/outputs/same_stack_compare_4gpu/gpu2_vcdr_state_rtg_value/checkpoints/best.pt \
  --output-dir vdt_dev/outputs/eval_only_vcdr_same_stack
```

### 13.3 重新聚合多个 output roots

```bash
python -m vdt_dev.analysis.aggregate \
  vdt_dev/outputs/same_stack_compare_4gpu \
  vdt_dev/outputs/querymode_sweep_4gpu \
  vdt_dev/outputs/hopper_same_stack_depth_sweep \
  vdt_dev/outputs/hopper_matched_budget_depth_sweep \
  vdt_dev/outputs/online_compare_4gpu \
  --destination-dir vdt_dev/outputs/aggregated/stage4_master
```

### 13.4 一键从 output root 生成聚合图

```bash
bash vdt_dev/scripts/make_all_figures.sh \
  vdt_dev/outputs \
  vdt_dev/outputs/aggregated \
  vdt_dev/paper_figures
```

说明：

- `make_all_figures.sh` 现在会在聚合表里存在 online 指标时自动补画 `online_improvement.*`

### 13.5 从聚合结果导出 paper-ready 表

```bash
python -m vdt_dev.analysis.export_tables \
  --summary-csv vdt_dev/outputs/aggregated/stage4_master/seed_aggregated_summary.csv \
  --output-dir vdt_dev/paper_tables/stage4_master
```
