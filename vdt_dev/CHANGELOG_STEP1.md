# Step 1 Changelog

## Goal

Build a runnable VDT-BAR dev branch that only changes actor residual aggregation while keeping original VDT value-guidance logic intact.

## New Files

- [runner.py](/Users/zoe/vdt/VDT/vdt_dev/runner.py)
  - Dev train/eval entrypoint.
  - Copies the root training orchestration into a separate branch-friendly runner.
  - Adds YAML config loading, CLI overrides, output directories, metadata snapshots, and consolidated checkpoints.

- [models/block_attn_res.py](/Users/zoe/vdt/VDT/vdt_dev/models/block_attn_res.py)
  - Reusable static-query Block Attention Residual router.
  - Maintains embedding source, completed BAR block summaries, and current partial block summary.
  - Applies RMSNorm to routing keys and source-axis softmax.

- [models/vdt_bar_policy.py](/Users/zoe/vdt/VDT/vdt_dev/models/vdt_bar_policy.py)
  - Dev actor/policy for Step 1.
  - Preserves Decision Transformer style tokenization and autoregressive action prediction.
  - Adds pre-attn and pre-mlp BAR hooks while keeping the external forward signature aligned with the original actor.

- [utils/debug_hooks.py](/Users/zoe/vdt/VDT/vdt_dev/utils/debug_hooks.py)
  - Optional routing-weight / entropy / norm recorder for later mechanism analysis.

- [configs/vdt_bar/hopper_medium_v2.yaml](/Users/zoe/vdt/VDT/vdt_dev/configs/vdt_bar/hopper_medium_v2.yaml)
  - Main Step 1 config for `hopper-medium-v2`.

- [configs/vdt_bar/hopper_medium_v2_sanity.yaml](/Users/zoe/vdt/VDT/vdt_dev/configs/vdt_bar/hopper_medium_v2_sanity.yaml)
  - Very small config for smoke verification.

- [scripts/train_vdt_bar_sanity.sh](/Users/zoe/vdt/VDT/vdt_dev/scripts/train_vdt_bar_sanity.sh)
  - One-command sanity run.

- [scripts/train_vdt_bar.sh](/Users/zoe/vdt/VDT/vdt_dev/scripts/train_vdt_bar.sh)
  - Main Step 1 training script.
  - Includes a 4-GPU multi-seed template because the current repo has no DDP launcher.

- [scripts/eval_vdt_bar.sh](/Users/zoe/vdt/VDT/vdt_dev/scripts/eval_vdt_bar.sh)
  - Checkpoint evaluation wrapper.

- [tests/conftest.py](/Users/zoe/vdt/VDT/vdt_dev/tests/conftest.py)
  - Test-only import stubs for environments without the full RL stack.

- [tests/test_block_attn_res_shapes.py](/Users/zoe/vdt/VDT/vdt_dev/tests/test_block_attn_res_shapes.py)
  - Shape, normalization, and non-divisible BAR block partition tests.

- [tests/test_vdt_bar_forward.py](/Users/zoe/vdt/VDT/vdt_dev/tests/test_vdt_bar_forward.py)
  - Actor forward and `get_action` tests.

- [tests/test_vdt_bar_backward.py](/Users/zoe/vdt/VDT/vdt_dev/tests/test_vdt_bar_backward.py)
  - Backprop test confirming gradients hit BAR queries and actor weights.

- [tests/test_smoke_train_step.py](/Users/zoe/vdt/VDT/vdt_dev/tests/test_smoke_train_step.py)
  - Lightweight IQL update smoke test to catch NaN/Inf regressions.

## Reused Original Modules

- [src/iql.py](/Users/zoe/vdt/VDT/src/iql.py)
  - Q/V update logic, actor loss, and policy optimization.

- [src/value_functions.py](/Users/zoe/vdt/VDT/src/value_functions.py)
  - `TwinQ` and `ValueFunction`.

- [src/util.py](/Users/zoe/vdt/VDT/src/util.py)
  - Evaluation and online rollout helpers.

- [replay_buffer.py](/Users/zoe/vdt/VDT/replay_buffer.py)
  - Replay handling for online finetuning.

## Intentionally Not Changed

- Root baseline entrypoint [main.py](/Users/zoe/vdt/VDT/main.py)
- Root actor implementation [decision_transformer.py](/Users/zoe/vdt/VDT/decision_transformer.py)
- Root transformer backbone [trajectory_gpt2.py](/Users/zoe/vdt/VDT/trajectory_gpt2.py)
- Q/V training logic
- Sampling algorithm
- Dynamic or critic-conditioned routing

## Practical Notes

- `runner.py` was added because the root `main.py` is monolithic and hardcodes the actor choice, plus it mutates CUDA env vars at import time. A separate dev runner was the smallest safe way to add Step 1 without destabilizing the baseline path.
- The dev actor uses a local PyTorch transformer implementation instead of importing the root `trajectory_gpt2.py` directly, mainly so tests can run without a hard dependency on `transformers`.
