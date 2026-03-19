# 阶段性实验总结与执行计划

本文档基于当前 repo 的真实代码状态撰写，覆盖原始 root baseline、`vdt_dev/` 下的 Step 1 / Step 2 / Step 3，以及本次已经落地的 Step 4 三项升级。本文档的目标不是提出新算法，而是把当前代码已经支持的研究问题、实验优先级、结果产物和后续论文映射整理成一套可直接执行的计划。

配套命令清单见：

- [EXPERIMENT_COMMANDS_STAGE.md](/Users/zoe/vdt/VDT/vdt_dev/EXPERIMENT_COMMANDS_STAGE.md)

## 1. 项目阶段性代码总结

### 1.1 原始 baseline

root 目录下的原始 VDT baseline 仍保持不动，关键入口与实现仍在：

- [main.py](/Users/zoe/vdt/VDT/main.py)
- [decision_transformer.py](/Users/zoe/vdt/VDT/decision_transformer.py)
- [trajectory_gpt2.py](/Users/zoe/vdt/VDT/trajectory_gpt2.py)
- [src/iql.py](/Users/zoe/vdt/VDT/src/iql.py)
- [src/util.py](/Users/zoe/vdt/VDT/src/util.py)
- [src/value_functions.py](/Users/zoe/vdt/VDT/src/value_functions.py)

这条线继续承担“原论文 baseline 复现”的角色，不与 dev stack 混跑。

### 1.2 Step 1：VDT-BAR

Step 1 已在 `vdt_dev/` 中实现：

- 核心实现： [models/block_attn_res.py](/Users/zoe/vdt/VDT/vdt_dev/models/block_attn_res.py)
- actor 封装： [models/vdt_bar_policy.py](/Users/zoe/vdt/VDT/vdt_dev/models/vdt_bar_policy.py)

真实改动边界：

- 只改 actor backbone 的 residual aggregation
- 引入 static-query Block Attention Residuals
- 不改 Q/V 网络
- 不改 actor loss
- 不改 value-guided sampling 主语义

### 1.3 Step 2：VDT-VCDR

Step 2 在 Step 1 基础上加入 dynamic query family：

- conditioner： [models/query_conditioner.py](/Users/zoe/vdt/VDT/vdt_dev/models/query_conditioner.py)
- policy 仍复用 [models/vdt_bar_policy.py](/Users/zoe/vdt/VDT/vdt_dev/models/vdt_bar_policy.py)

代码真实支持的 query modes 为：

- `static`
- `state`
- `state_rtg`
- `state_rtg_value`

真实实现边界：

- query fusion 只有 additive：`q = w + delta_q`
- `state_rtg_value` 只使用 detached scalar `V(s_t)`
- 没有 advantage-conditioned routing
- 没有 uncertainty-conditioned routing
- 没有 Q-conditioned routing
- 没有改 Q/V 目标函数
- 没有改 sampling 语义

### 1.4 Step 3：实验与分析基础设施

Step 3 已经把 `vdt_dev/` 变成完整实验平台：

- 主入口： [runner.py](/Users/zoe/vdt/VDT/vdt_dev/runner.py)
- manifest / schema： [analysis/manifest.py](/Users/zoe/vdt/VDT/vdt_dev/analysis/manifest.py)、[analysis/schema.py](/Users/zoe/vdt/VDT/vdt_dev/analysis/schema.py)
- 聚合： [analysis/collect.py](/Users/zoe/vdt/VDT/vdt_dev/analysis/collect.py)、[analysis/aggregate.py](/Users/zoe/vdt/VDT/vdt_dev/analysis/aggregate.py)
- best/latest 选择： [analysis/checkpoint_select.py](/Users/zoe/vdt/VDT/vdt_dev/analysis/checkpoint_select.py)
- 出图： [analysis/plot_final_bars.py](/Users/zoe/vdt/VDT/vdt_dev/analysis/plot_final_bars.py)、[analysis/plot_rtg_alignment.py](/Users/zoe/vdt/VDT/vdt_dev/analysis/plot_rtg_alignment.py)、[analysis/plot_routing_heatmap.py](/Users/zoe/vdt/VDT/vdt_dev/analysis/plot_routing_heatmap.py)、[analysis/plot_layer_norms.py](/Users/zoe/vdt/VDT/vdt_dev/analysis/plot_layer_norms.py)、[analysis/plot_query_norms.py](/Users/zoe/vdt/VDT/vdt_dev/analysis/plot_query_norms.py)、[analysis/plot_depth_source_usage.py](/Users/zoe/vdt/VDT/vdt_dev/analysis/plot_depth_source_usage.py)
- experiment preset / registry： [experiments/presets.py](/Users/zoe/vdt/VDT/vdt_dev/experiments/presets.py)、[experiments/registry.py](/Users/zoe/vdt/VDT/vdt_dev/experiments/registry.py)、[experiments/matrix.py](/Users/zoe/vdt/VDT/vdt_dev/experiments/matrix.py)

Step 3 真实已支持：

- query-mode sweep
- depth sweep
- seed sweep
- Gym env sweep
- RTG-grid reevaluation
- routing / norm / source usage 机制分析
- paper-ready aggregation / tables / figures

### 1.5 Step 4：本次新增的三项升级

#### Upgrade A：same-stack vanilla baseline

本次新增了 dev 内部的 vanilla 对照线：

- 实现： [models/vdt_vanilla_policy.py](/Users/zoe/vdt/VDT/vdt_dev/models/vdt_vanilla_policy.py)
- 变体解析： [model_variants.py](/Users/zoe/vdt/VDT/vdt_dev/model_variants.py)

现在 `vdt_dev` 内部正式区分三类 `model_variant`：

- `vanilla_dev`
- `bar`
- `vcdr`

这意味着 paper 主线可以在完全同栈条件下比较：

- 同 runner
- 同 logging
- 同 output schema
- 同 aggregation / plotting / RTG-grid

唯一差异只剩 actor backbone 是否使用：

- 标准 residual
- BAR static residual routing
- VCDR dynamic routing

#### Upgrade B：compute reporting + matched-budget depth sweep

本次新增：

- compute 工具： [utils/compute.py](/Users/zoe/vdt/VDT/vdt_dev/utils/compute.py)
- matched-budget preset 生成： [experiments/budget.py](/Users/zoe/vdt/VDT/vdt_dev/experiments/budget.py)

当前每个 run 都可产出 `analysis/compute_summary.json`，并被自动并入：

- `analysis/summary.json`
- `run_summary.csv`
- `seed_aggregated_summary.csv`

当前结构化 compute 字段包括：

- `policy_param_count`
- `policy_trainable_param_count`
- `qf_param_count`
- `vf_param_count`
- `total_param_count`
- `approx_policy_forward_flops`
- `approx_total_train_step_flops`
- `mean_train_step_time_sec`
- `mean_steps_per_sec`
- `mean_policy_tokens_per_sec`
- `peak_gpu_memory_mb`
- `total_wall_clock_sec`
- `offline_wall_clock_sec`
- `online_wall_clock_sec`

同时新增 matched-budget depth preset：

- `hopper_matched_budget_depth_sweep`

当前实现方式是“近似参数预算匹配”：

- 以 6-layer 宽度 `embed_dim=256, n_head=4` 为 reference
- 对每个深度搜索 `embed_dim` / `n_head`
- 使参数量尽量逼近 reference
- manifest / compute summary / 聚合表会记录：
  - `budget_mode`
  - `budget_reference`
  - `budget_target_params`
  - `budget_tolerance_pct`
  - `budget_actual_gap_pct`

需要明确写在论文中的 caveat：

- 这是“matched-parameter”而不是精确 matched-FLOP
- 深度越大时，最优解可能会降到 `n_head=1`
- 因此它适合作为更严格的结构性对照，但仍需配合 compute table 一起解释

#### Upgrade C：更强 setting 选择 online tuning

这一步我选择了 online tuning，而不是 Maze2D / AntMaze fallback。原因是：

- `runner.py` 中原本就已经有 offline-to-online 路径，语义沿用原始 VDT/IQL 栈
- 不需要改 Q/V 目标函数
- 不需要改 sampling 语义
- 比补 sparse-reward runtime 更低风险、更贴近 VDT narrative

本次新增的 online 入口包括：

- configs：
  - [configs/vdt_vanilla/hopper_medium_v2_vanilla_dev_online.yaml](/Users/zoe/vdt/VDT/vdt_dev/configs/vdt_vanilla/hopper_medium_v2_vanilla_dev_online.yaml)
  - [configs/vdt_bar/hopper_medium_v2_bar_online.yaml](/Users/zoe/vdt/VDT/vdt_dev/configs/vdt_bar/hopper_medium_v2_bar_online.yaml)
  - [configs/vdt_vcdr/hopper_medium_v2_vcdr_state_rtg_value_online.yaml](/Users/zoe/vdt/VDT/vdt_dev/configs/vdt_vcdr/hopper_medium_v2_vcdr_state_rtg_value_online.yaml)
- scripts：
  - [scripts/train_online_vdt_vanilla_dev.sh](/Users/zoe/vdt/VDT/vdt_dev/scripts/train_online_vdt_vanilla_dev.sh)
  - [scripts/train_online_vdt_bar_static.sh](/Users/zoe/vdt/VDT/vdt_dev/scripts/train_online_vdt_bar_static.sh)
  - [scripts/train_online_vdt_vcdr_state_rtg_value.sh](/Users/zoe/vdt/VDT/vdt_dev/scripts/train_online_vdt_vcdr_state_rtg_value.sh)
  - [scripts/launch_online_compare_4gpu.sh](/Users/zoe/vdt/VDT/vdt_dev/scripts/launch_online_compare_4gpu.sh)
- 聚合 / 出图：
  - [analysis/collect.py](/Users/zoe/vdt/VDT/vdt_dev/analysis/collect.py) 现在会区分 offline / online best return
  - [analysis/plot_online_improvement.py](/Users/zoe/vdt/VDT/vdt_dev/analysis/plot_online_improvement.py) 会画 before/after online bar 图

### 1.6 当前支持的能力边界

当前直接可跑的主线能力：

- same-stack vanilla / BAR / VCDR offline training
- same-stack vanilla / BAR / VCDR Hopper online tuning
- query-mode sweep
- equal-width depth sweep
- matched-budget depth sweep
- seed sweep
- Gym env sweep
- RTG-grid reevaluation
- routing / norm / source usage 机制分析

当前还没实现的方向：

- advantage-conditioned routing
- uncertainty-conditioned routing
- Q-conditioned routing
- Maze2D runnable path
- AntMaze runnable path
- exact matched-FLOP sweep

## 2. 当前真实支持的实验入口与分析能力

### 2.1 Launcher scripts

当前 repo 内已有并可直接使用的 launcher / wrapper scripts：

- [scripts/train_vdt_vanilla_dev.sh](/Users/zoe/vdt/VDT/vdt_dev/scripts/train_vdt_vanilla_dev.sh)
- [scripts/train_vdt_vanilla_dev_sanity.sh](/Users/zoe/vdt/VDT/vdt_dev/scripts/train_vdt_vanilla_dev_sanity.sh)
- [scripts/train_vdt_bar.sh](/Users/zoe/vdt/VDT/vdt_dev/scripts/train_vdt_bar.sh)
- [scripts/train_vdt_bar_sanity.sh](/Users/zoe/vdt/VDT/vdt_dev/scripts/train_vdt_bar_sanity.sh)
- [scripts/train_vdt_vcdr_state.sh](/Users/zoe/vdt/VDT/vdt_dev/scripts/train_vdt_vcdr_state.sh)
- [scripts/train_vdt_vcdr_state_rtg.sh](/Users/zoe/vdt/VDT/vdt_dev/scripts/train_vdt_vcdr_state_rtg.sh)
- [scripts/train_vdt_vcdr_state_rtg_value.sh](/Users/zoe/vdt/VDT/vdt_dev/scripts/train_vdt_vcdr_state_rtg_value.sh)
- [scripts/launch_same_stack_compare_4gpu.sh](/Users/zoe/vdt/VDT/vdt_dev/scripts/launch_same_stack_compare_4gpu.sh)
- [scripts/launch_querymode_sweep_4gpu.sh](/Users/zoe/vdt/VDT/vdt_dev/scripts/launch_querymode_sweep_4gpu.sh)
- [scripts/launch_depth_sweep_4gpu.sh](/Users/zoe/vdt/VDT/vdt_dev/scripts/launch_depth_sweep_4gpu.sh)
- [scripts/launch_seed_sweep_4gpu.sh](/Users/zoe/vdt/VDT/vdt_dev/scripts/launch_seed_sweep_4gpu.sh)
- [scripts/launch_online_compare_4gpu.sh](/Users/zoe/vdt/VDT/vdt_dev/scripts/launch_online_compare_4gpu.sh)
- [scripts/reeval_rtg_grid.sh](/Users/zoe/vdt/VDT/vdt_dev/scripts/reeval_rtg_grid.sh)
- [scripts/aggregate_results.sh](/Users/zoe/vdt/VDT/vdt_dev/scripts/aggregate_results.sh)
- [scripts/make_all_figures.sh](/Users/zoe/vdt/VDT/vdt_dev/scripts/make_all_figures.sh)

### 2.2 Experiment presets

当前 preset 分成两层理解：

第一层，真正驱动命令展开的是：

- [experiments/presets.py](/Users/zoe/vdt/VDT/vdt_dev/experiments/presets.py)
- [experiments/registry.py](/Users/zoe/vdt/VDT/vdt_dev/experiments/registry.py)

第二层，`configs/experiments/*.yaml` 是 metadata mirror，便于文档索引。

当前真实 preset 包括：

- `hopper_same_stack_compare`
- `hopper_querymode_sweep`
- `hopper_depth_sweep`
- `hopper_same_stack_depth_sweep`
- `hopper_matched_budget_depth_sweep`
- `gym_env_sweep`
- `gym_same_stack_sweep`
- `seed_sweep`
- `hopper_online_compare`
- `maze2d_template`
- `antmaze_template`

### 2.3 Aggregation / plotting / reevaluation

当前真实入口：

- 聚合：`python -m vdt_dev.analysis.aggregate`
- 表格：`python -m vdt_dev.analysis.export_tables`
- final bars：`python -m vdt_dev.analysis.plot_final_bars`
- online before/after：`python -m vdt_dev.analysis.plot_online_improvement`
- RTG alignment：`python -m vdt_dev.analysis.plot_rtg_alignment`
- routing heatmap：`python -m vdt_dev.analysis.plot_routing_heatmap`
- hidden/output norms：`python -m vdt_dev.analysis.plot_layer_norms`
- static/delta/fused query norms：`python -m vdt_dev.analysis.plot_query_norms`
- source usage / source distance：`python -m vdt_dev.analysis.plot_depth_source_usage`
- RTG-grid reevaluation：`python -m vdt_dev.runner --job reeval_rtg_grid ...`

### 2.4 Best checkpoint / latest checkpoint / RTG-grid reevaluation 入口

真实入口如下：

- best/latest checkpoint 选择：
  - `python -m vdt_dev.analysis.checkpoint_select <run_dir> --selector latest`
  - `python -m vdt_dev.analysis.checkpoint_select <run_dir> --selector best`
- eval：
  - `python -m vdt_dev.runner --job eval --checkpoint-path ...`
- RTG-grid：
  - `python -m vdt_dev.runner --job reeval_rtg_grid --checkpoint-path ... --reeval-rtg-grid ...`
  - `bash vdt_dev/scripts/reeval_rtg_grid.sh <run_dir>`

### 2.5 Debug artifacts 与 output 目录结构

当前 debug artifacts 可保存：

- `logs/debug_metrics.jsonl`
- `debug/routing/*.pt`
- `debug/eval_rtg_grid/*.pt`
- `debug/eval_rtg_grid/*_summary.json`

当前可提取机制指标包括：

- routing entropy
- source usage
- source distance
- hidden norm
- output norm
- static / delta / fused query norms
- `pre_attn` vs `pre_mlp` site 标签

当前标准输出结构：

```text
vdt_dev/outputs/<run_name>/
  metadata/
  checkpoints/
  logs/
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

### 2.6 当前 tests 覆盖面

当前 tests 能验证：

- BAR / VCDR forward/backward
- dynamic query modes
- detached value stopgrad
- smoke train step
- manifest / schema / discovery
- aggregation / plotting smoke
- checkpoint select
- RTG-grid config parsing
- eval / reeval 新输出目录 metadata 初始化
- vanilla_dev policy smoke
- compute summary 生成与聚合
- Step 4 preset expansion

## 3. 当前最重要的研究问题

### Q1. same-stack vanilla_dev 相比 BAR / VCDR 的真实收益是什么

这是 Step 4 最关键的问题。现在终于可以在完全同栈下比较：

- `vanilla_dev`
- `bar`
- `vcdr`

如果这一步没有清晰收益，后面所有“routing 带来收益”的归因都会变弱。

### Q2. dynamic routing 是否真的优于 static BAR

这个问题继续由 query-mode sweep 回答，但现在必须放在 same-stack attribution 之后解释：

- vanilla_dev 是否已经足够
- BAR 是否先于 VCDR 带来收益
- VCDR 的提升究竟是“动态 query”还是只是“引入 routing 模块”

### Q3. deeper VDT 是否在 compute-aware 条件下仍更受益于 BAR / VCDR

现在 depth claim 必须拆成两层：

- equal-width trend：看现象趋势
- matched-budget structural sweep：看在参数量大致控制下，结论是否仍成立

### Q4. online tuning 下同栈三条线谁更能把 offline 表示转化为在线增益

这一步把项目从“Gym offline architecture tweak”推进到更接近 VDT narrative 的 setting：

- offline best
- online best
- online minus offline gain

### Q5. RTG alignment 与 routing mechanism 是否支持我们的解释

要回答的不只是“谁分数高”，还包括：

- `state_rtg_value` 是否更对齐 RTG target
- online 后 alignment 是否继续改善
- BAR / VCDR 是否在更深层激活更有选择性的 residual routing

## 4. 实验总优先级排序

- `P0`：回归 / sanity / launcher dry-run / schema correctness
- `P1`：Hopper same-stack attribution compare
- `P2`：Hopper query-mode sweep
- `P3`：Hopper same-stack equal-width depth sweep + compute report
- `P4`：Hopper matched-budget depth sweep
- `P5`：关键结果 seed sweep
- `P6`：Gym same-stack env sweep
- `P7`：Hopper online compare
- `P8`：RTG-grid reevaluation
- `P9`：机制分析图
- `P10`：Maze2D / AntMaze template 后续储备

## 5. 实验清单

### 5.1 E0：回归与接口检查

- 实验级别：`P0`
- 研究问题：Step 4 是否保持 Step 1 / 2 / 3 不回归，同时把 vanilla / compute / online 路径接通
- 核心假设：新增路径不会破坏既有 BAR/VCDR/runtime/analysis 语义
- 实验设计
  - 比较对象：单元测试、sanity run、launcher dry-run
  - 固定变量：Hopper tiny sanity config
  - 因变量：tests 是否通过、manifest 是否落盘、compute summary 是否生成
- 预期结果：所有测试通过；sanity run 生成标准 output 结构
- 失败信号：manifest 缺失、compute summary 缺失、online/offline phase 字段丢失
- 结果产物
  - `metadata/run_manifest.json`
  - `analysis/summary.json`
  - `analysis/compute_summary.json`
- 后续分析动作：只有通过后再开大 sweep

### 5.2 E1：Hopper same-stack attribution compare

- 实验级别：`P1`
- 研究问题：在完全同栈条件下，BAR 与 VCDR 是否相对 vanilla_dev 有净收益
- 核心假设：`bar > vanilla_dev`，`vcdr(state_rtg_value) >= bar`
- 实验设计
  - 比较对象：`vanilla_dev` vs `bar(static)` vs `vcdr(state_rtg_value)`
  - 固定变量：`hopper-medium-v2`，`L=6`，同 runner，同 logging，同 eval
  - 自变量：`model_variant`
  - 因变量：`best_eval_return_normalized`、compute report、best checkpoint
  - seeds：先 `1 seed`
- 预期结果：至少 BAR 或 VCDR 中一条线相对 vanilla_dev 有清晰提升
- 失败信号：三条线几乎重合；VCDR 不优于 BAR
- 结果产物
  - 三个 run 的 checkpoint / logs / summary / compute summary
  - 聚合表与 main result bar
- 后续分析动作：选出后续 seed sweep 和 online compare 的优先线

### 5.3 E2：Hopper query-mode sweep

- 实验级别：`P2`
- 研究问题：dynamic query family 内部，哪种 conditioning 最有效
- 核心假设：`state_rtg_value >= state_rtg >= state`，且 static BAR 是重要对照
- 实验设计
  - 比较对象：`bar(static)`、`vcdr(state)`、`vcdr(state_rtg)`、`vcdr(state_rtg_value)`
  - 固定变量：`hopper-medium-v2`，`L=6`
  - 自变量：`query_mode`
  - 因变量：`best_eval_return_normalized`
- 预期结果：`state_rtg_value` 最强，说明 detached scalar value signal 有额外帮助
- 失败信号：`state_rtg_value` 不优于 `state_rtg`
- 结果产物：query-mode 主结果图、seed 表、最佳 checkpoint
- 后续分析动作：把最强 dynamic mode 带入深度和 online 主线

### 5.4 E3：Hopper same-stack equal-width depth sweep

- 实验级别：`P3`
- 研究问题：在不控预算的 equal-width 条件下，更深网络是否更受益于 BAR / VCDR
- 核心假设：随着深度增加，routing 系列相对 vanilla_dev 的优势更明显
- 实验设计
  - 比较对象：`vanilla_dev`、`bar(static)`、`vcdr(state_rtg_value)`
  - 固定变量：`hopper-medium-v2`，`embed_dim=256`，`n_head=4`
  - 自变量：`n_layer in {6,12,18}`
  - 因变量：
    - `best_eval_return_normalized`
    - `total_param_count`
    - `approx_total_train_step_flops`
    - `peak_gpu_memory_mb`
    - `total_wall_clock_sec`
- 预期结果：先看到清晰趋势，但需要带 compute caveat 解读
- 失败信号：深度增加没有收益，或收益完全可由参数量解释
- 结果产物：equal-width depth 主图、compute table
- 后续分析动作：再进入 matched-budget sweep

### 5.5 E4：Hopper matched-budget depth sweep

- 实验级别：`P4`
- 研究问题：在参数预算近似控制后，depth claim 是否仍成立
- 核心假设：routing 方法对深度的利用不完全依赖于“更宽更多参数”
- 实验设计
  - 比较对象：`vanilla_dev`、`bar(static)`、`vcdr(state_rtg_value)`
  - 固定变量：target 参数量来自各变体 6-layer reference
  - 自变量：`n_layer in {6,12,18}`，同时自动调整 `embed_dim` / `n_head`
  - 因变量：
    - `best_eval_return_normalized`
    - `budget_actual_gap_pct`
    - `total_param_count`
    - `approx_total_train_step_flops`
  - 说明：这是近似 matched-parameter，不是精确 matched-FLOP
- 预期结果：VCDR 至少在 12/18 层维持相对优势；若优势消失，则 depth claim 要降级为 trend claim
- 失败信号：matched-budget 后所有增益消失
- 结果产物：matched-budget depth 表格、结构性结论、compute-aware 附录表
- 后续分析动作：决定论文中 depth 结论的措辞强弱

### 5.6 E5：关键结果 seed sweep

- 实验级别：`P5`
- 研究问题：最重要的离线结论是否稳定
- 核心假设：same-stack 最优线和 query-mode 最优线在多 seed 下仍稳定
- 实验设计
  - 比较对象：优先做 `vcdr(state_rtg_value)`；其次补 `vanilla_dev` 或 `bar`
  - 固定变量：`hopper-medium-v2`，优先 `L=6`
  - 自变量：`seed in {0,1,2,3}`
  - 因变量：seed mean/std
- 预期结果：最强线的均值和排名稳定
- 失败信号：seed std 很大、排名频繁翻转
- 结果产物：seed aggregation summary、误差条图
- 后续分析动作：决定主文是否能上 seed-aggregated 主表

### 5.7 E6：Gym same-stack env sweep

- 实验级别：`P6`
- 研究问题：收益是否跨 Hopper / Walker2d / HalfCheetah 泛化
- 核心假设：趋势至少在常见 Gym medium 环境中可复现
- 实验设计
  - 比较对象：`vanilla_dev`、`bar(static)`、`vcdr(state_rtg_value)`
  - envs：
    - `hopper-medium-v2`
    - `walker2d-medium-v2`
    - `halfcheetah-medium-v2`
  - 固定变量：`L=6`
  - 自变量：`env_name`、`model_variant`
  - 因变量：`best_eval_return_normalized`
  - 说明：当前用 Hopper config 覆写 `--env-name` 是安全的，因为 runner 会按 `env_name` 动态重算 env targets、reward scale、state/action dimensions
- 预期结果：至少在 2/3 环境中看到 consistent ranking
- 失败信号：只有 Hopper 有效，其余环境失效
- 结果产物：跨环境主表、env sweep bar 图
- 后续分析动作：决定论文 main table 的环境覆盖范围

### 5.8 E7：Hopper online compare

- 实验级别：`P7`
- 研究问题：same-stack 三条线在 offline-to-online setting 下谁更能把 offline 表示转成在线增益
- 核心假设：`vcdr(state_rtg_value)` 的 online gain 大于 `bar`，`bar` 不低于 `vanilla_dev`
- 实验设计
  - 比较对象：`vanilla_dev`、`bar(static)`、`vcdr(state_rtg_value)`
  - 固定变量：`hopper-medium-v2`，同 online runner
  - 自变量：`model_variant`
  - 因变量：
    - `best_offline_eval_return_normalized`
    - `best_online_eval_return_normalized`
    - `online_minus_offline_best_return`
    - online compute metrics
- 预期结果：至少一条 routing 线在 online gain 上占优
- 失败信号：online 增益全部很小，或 vanilla_dev 反而最佳
- 结果产物：online compare bar 图、before/after 图、最佳 checkpoint
- 后续分析动作：决定论文是否把 online tuning 放入 main result 还是 appendix

### 5.9 E8：RTG-grid reevaluation

- 实验级别：`P8`
- 研究问题：target RTG 与 achieved return alignment 是否因 BAR / VCDR 改善；online 之后是否继续改善
- 核心假设：`vcdr(state_rtg_value)` alignment 更好，且 online 之后 alignment 曲线更贴近 target
- 实验设计
  - 比较对象：至少 `vanilla_dev`、`bar(static)`、`vcdr(state_rtg_value)`
  - checkpoint：优先 `best.pt`
  - 因变量：`rtg_alignment_error`、alignment curve
- 预期结果：VCDR 在高 RTG 区间更稳定
- 失败信号：alignment 与回报提升不一致
- 结果产物：alignment curve、alignment table
- 后续分析动作：连接到 VDT narrative 中的 RTG controllability

### 5.10 E9：机制分析

- 实验级别：`P9`
- 研究问题：routing 机制是否支持“越深越需要选择性 residual aggregation”的解释
- 核心假设：
  - 更强模型会呈现更清晰的 routing entropy 变化
  - `state_rtg_value` 的 query norm / source usage 模式更有结构
- 实验设计
  - 比较对象：重点放 `bar(static)` vs `vcdr(state_rtg_value)`，必要时加 `vanilla_dev` 作为无 routing 对照
  - 因变量：
    - `routing_entropy_mean`
    - `source_usage_by_layer`
    - `source_distance_mean`
    - `hidden_norm_by_layer`
    - `output_norm_by_layer`
    - `static_query_norm_mean`
    - `delta_query_norm_mean`
    - `fused_query_norm_mean`
    - `pre_attn` vs `pre_mlp` site pattern
- 预期结果：更强模型在 deeper layers 中呈现更可解释的 routing 选择性
- 失败信号：机制图完全噪声化、与性能结论相反
- 结果产物：heatmap / norm / source usage 图
- 后续分析动作：服务 mechanism figure 与 appendix

### 5.11 E10：Maze2D / AntMaze 模板

- 实验级别：`P10`
- 研究问题：作为未来稀疏奖励扩展接口是否保留
- 当前状态：仍是 template only
- 原因：`runner.get_env_metadata()` 目前只支持 Hopper / Walker2d / HalfCheetah
- 结论：本阶段不进入主线，不纳入 paper 主结论

## 6. 推荐的实验推进顺序

推荐按下面顺序推进，而不是把所有实验一次性铺开：

1. 先跑 `P0` 回归与 sanity，确认 Step 4 没破坏已有行为
2. 再跑 `P1` Hopper same-stack attribution compare，先回答“同栈归因值不值得继续”
3. 然后跑 `P2` query-mode sweep，确定最强 dynamic mode 仍是 `state_rtg_value`
4. 接着跑 `P3` equal-width same-stack depth sweep，配合 compute table 看趋势
5. 再跑 `P4` matched-budget depth sweep，决定 depth claim 能否写成更强结论
6. 对最重要的离线结论做 `P5` seed sweep
7. 再做 `P6` Gym same-stack env sweep，检查跨环境泛化
8. 之后再上 `P7` Hopper online compare，把项目推进到更贴近 VDT narrative 的 setting
9. 再对关键 best checkpoints 做 `P8` RTG-grid reevaluation
10. 最后补 `P9` 机制分析图
11. `P10` Maze2D / AntMaze 保持模板，不进入当前主线

## 7. 当前最可能成功 / 最可能失败的实验项

### 7.1 最可能最快看到趋势的实验

- `P1` Hopper same-stack attribution compare
- `P2` Hopper query-mode sweep
- `P7` Hopper online compare

原因：

- Hopper 是当前 configs / scripts / env metadata 最成熟的主路径
- `state_rtg_value` 是现有实现里最强的 VCDR 候选
- online tuning 现在不是“新算法”，而是把现有 runner 中的同语义路径正式化

### 7.2 更像确认性实验的项目

- `P5` seed sweep
- `P6` Gym same-stack env sweep
- `P8` RTG-grid reevaluation

这些实验主要用来把趋势升级成更可靠的论文证据，而不是第一次发现趋势。

### 7.3 风险较大、应后置的项目

- `P4` matched-budget depth sweep
- `P9` 机制分析

风险点：

- matched-budget 目前是 approximate matched-parameter，不是 exact matched-FLOP
- 18-layer 可能需要降到较小 `embed_dim` / `n_head`
- 机制图常常需要在已有 clear winner 后再解释，否则容易“图好看但结论弱”

## 8. 最终论文素材映射

- `P1` same-stack attribution compare
  - 对应：main result 主归因表
- `P2` query-mode sweep
  - 对应：routing family ablation
- `P3` equal-width depth sweep
  - 对应：depth trend figure
- `P4` matched-budget depth sweep
  - 对应：更严谨的 structural depth claim / appendix compute-aware table
- `P5` seed sweep
  - 对应：主表误差条 / appendix seed robustness
- `P6` Gym env sweep
  - 对应：cross-env main table
- `P7` online compare
  - 对应：stronger-setting main figure 或 appendix major result
- `P8` RTG-grid reevaluation
  - 对应：alignment figure
- `P9` 机制分析
  - 对应：mechanism figure + appendix
- `P10` Maze2D / AntMaze template
  - 对应：future work，不进入当前主文结论
