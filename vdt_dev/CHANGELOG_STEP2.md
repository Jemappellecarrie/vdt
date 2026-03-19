# Step 2 Changelog

## Goal

Extend Step 1 `VDT-BAR` into Step 2 `VDT-VCDR` by adding query-conditioned depth routing while preserving:

- the original VDT Q/V objectives
- the original actor optimization logic
- the original RTG-guided sampling/evaluation algorithm
- Step 1 static BAR behavior

## Files Added

- `models/query_conditioner.py`
  - Adds `RoutingContextExtractor` and `QueryConditioner`.
  - Extracts timestep-aligned state / RTG / detached value context from the real DT token layout.
  - Produces per-layer, per-route-site query deltas with zero-initialized output heads.

- `iql.py`
  - Dev-only IQL wrapper that keeps the original loss formulas but threads detached `V(s_t)` into the policy forward path when `state_rtg_value` is active.

- `value_conditioning.py`
  - Detached value-feature helper.
  - Rollout adapter so `src.util.evaluate_policy` and `src.util.vec_evaluate_episode_rtg` can stay unchanged.

- `configs/vdt_vcdr/hopper_medium_v2_vcdr_state.yaml`
  - Main Step 2 config for `state` query mode.

- `configs/vdt_vcdr/hopper_medium_v2_vcdr_state_rtg.yaml`
  - Main Step 2 config for `state_rtg` query mode.

- `configs/vdt_vcdr/hopper_medium_v2_vcdr_state_rtg_value.yaml`
  - Main Step 2 config for detached `state_rtg_value` routing.

- `scripts/train_vdt_vcdr_state.sh`
  - One-command Step 2 state-conditioned run.

- `scripts/train_vdt_vcdr_state_rtg.sh`
  - One-command Step 2 state+RTG run.

- `scripts/train_vdt_vcdr_state_rtg_value.sh`
  - One-command Step 2 state+RTG+value run.

- `scripts/launch_hopper_querymode_sweep_4gpu.sh`
  - Four independent single-GPU runs for quick query-mode comparisons.

- `tests/test_query_conditioner_shapes.py`
  - Query-conditioner mode and broadcast tests.

- `tests/test_block_attn_res_dynamic_query.py`
  - Static + dynamic BAR query path tests.

- `tests/test_vcdr_forward_modes.py`
  - Forward coverage for all Step 2 query modes.

- `tests/test_value_stopgrad.py`
  - Verifies routing does not backprop into value features / value net.

## Files Modified

- `models/block_attn_res.py`
  - Keeps Step 1 static routing intact.
  - Adds optional additive dynamic query deltas.
  - Records static query, dynamic delta, and fused query for debugging.

- `models/vdt_bar_policy.py`
  - Extends the existing Step 1 dev actor instead of creating a parallel policy stack.
  - Adds token-layout-aware routing context extraction through the new conditioner.
  - Supports `static`, `state`, `state_rtg`, and `state_rtg_value`.

- `utils/debug_hooks.py`
  - Adds dynamic-query, static-query, and fused-query norm logging plus per-mode summaries.

- `runner.py`
  - Adds Step 2 config fields.
  - Switches the dev runner to the Step 2-capable dev IQL wrapper.
  - Uses the value-conditioning adapter for evaluation and online rollout paths.

- `tests/test_smoke_train_step.py`
  - Adds a Step 2 `state_rtg_value` smoke train step.

- `README.md`
  - Documents Step 2 scope, configs, scripts, and the 4-GPU sweep workflow.

- `__init__.py`
  - Updates the package description to reflect Step 1 + Step 2 coverage.

## Step 2 Boundary

Implemented:

- additive dynamic query routing
- timestep/token context extraction from `(rtg, state, action)` token layout
- detached scalar `V(s_t)` routing input
- backward-compatible static BAR mode
- dev-only wrappers for training/eval plumbing

Not implemented:

- Q-conditioned routing
- advantage-conditioned routing
- uncertainty-conditioned routing
- multi-head depth routing
- block-summary redesign
- sampling algorithm changes
- Q/V loss changes

## Known Risks / Technical Debt

- `vdt_dev/iql.py` mirrors the original `src.iql` update logic instead of sharing a common abstraction, so future upstream changes would need to be ported manually.
- The rollout path intentionally preserves the original `src.util` behavior, including any quirks in how state tensors are prepared during planning.
- The query conditioner currently assumes the Step 1 DT token layout stays `(rtg, state, action)` per timestep.

## Recommended Step 3 Direction

- Add new conditioner inputs as optional extensions to `RoutingContextExtractor`, not as policy rewrites.
- Start with `state_rtg_adv` or uncertainty features as extra context channels reusing the same additive `q = w + delta_q` path.
- Keep detached critic-derived features opt-in and explicit so actor/value optimization boundaries remain obvious.
