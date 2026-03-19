from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional

import torch


ENV_DIMENSIONS: Dict[str, tuple[int, int]] = {
    "hopper-medium-v2": (11, 3),
    "walker2d-medium-v2": (17, 6),
    "halfcheetah-medium-v2": (17, 6),
}


def count_parameters(module: torch.nn.Module, *, trainable_only: bool = False) -> int:
    return int(
        sum(
            parameter.numel()
            for parameter in module.parameters()
            if (parameter.requires_grad or not trainable_only)
        )
    )


def env_dimensions(env_name: str) -> tuple[int, int]:
    if env_name not in ENV_DIMENSIONS:
        raise ValueError(
            f"Unsupported env `{env_name}` for dimension lookup. "
            f"Available: {sorted(ENV_DIMENSIONS)}"
        )
    return ENV_DIMENSIONS[env_name]


def _linear_flops(batch_tokens: int, in_dim: int, out_dim: int) -> int:
    return int(2 * batch_tokens * in_dim * out_dim)


def _mlp_forward_flops(batch_tokens: int, dims: Iterable[int]) -> int:
    dims_list = list(dims)
    total = 0
    for in_dim, out_dim in zip(dims_list[:-1], dims_list[1:]):
        total += _linear_flops(batch_tokens, in_dim, out_dim)
    return total


def estimate_policy_forward_flops(
    *,
    batch_size: int,
    context_len: int,
    hidden_size: int,
    num_layers: int,
    num_heads: int,
    state_dim: int,
    act_dim: int,
    use_attnres: bool,
    query_mode: str,
    conditioner_hidden_dim: int = 128,
    num_blocks: int = 8,
    use_pre_attn: bool = True,
    use_pre_mlp: bool = True,
) -> int:
    """
    Rough FLOP proxy for one actor forward on a DT batch.

    This is intentionally approximate. It focuses on the dominant transformer and
    routing linear algebra so we can compare runs on a common proxy rather than claim
    exact hardware-level FLOPs.
    """

    seq_len = context_len
    stacked_tokens = batch_size * seq_len * 3
    inner_dim = 4 * hidden_size
    total = 0

    # Input projections for (rtg, state, action) tokens plus output heads.
    total += _linear_flops(batch_size * seq_len, 1, hidden_size)
    total += _linear_flops(batch_size * seq_len, state_dim, hidden_size)
    total += _linear_flops(batch_size * seq_len, act_dim, hidden_size)
    total += _linear_flops(batch_size * seq_len, hidden_size, state_dim)
    total += _linear_flops(batch_size * seq_len, hidden_size, act_dim)

    for _ in range(num_layers):
        # Attention projections: q/k/v + output.
        total += 3 * _linear_flops(stacked_tokens, hidden_size, hidden_size)
        total += _linear_flops(stacked_tokens, hidden_size, hidden_size)

        # Attention score + value mixing.
        total += int(4 * batch_size * (stacked_tokens // batch_size) ** 2 * hidden_size)

        # Feed-forward block.
        total += _linear_flops(stacked_tokens, hidden_size, inner_dim)
        total += _linear_flops(stacked_tokens, inner_dim, hidden_size)

    if use_attnres:
        active_site_count = (int(use_pre_attn) + int(use_pre_mlp)) * num_layers
        average_sources = max(2, 1 + max(1, num_blocks) // 2)
        total += int(4 * stacked_tokens * hidden_size * average_sources * active_site_count)

        if query_mode != "static":
            if query_mode == "state":
                context_dim = hidden_size
            elif query_mode == "state_rtg":
                context_dim = 2 * hidden_size
            else:
                context_dim = (2 * hidden_size) + 1
            per_site_conditioner = (
                _linear_flops(stacked_tokens, context_dim, conditioner_hidden_dim)
                + _linear_flops(stacked_tokens, conditioner_hidden_dim, conditioner_hidden_dim)
                + _linear_flops(stacked_tokens, conditioner_hidden_dim, hidden_size)
            )
            total += active_site_count * per_site_conditioner

    return int(total)


def estimate_twinq_forward_flops(
    *,
    batch_size: int,
    state_dim: int,
    act_dim: int,
    hidden_dim: int,
    n_hidden: int,
) -> int:
    dims = [state_dim + act_dim, *([hidden_dim] * n_hidden), 1]
    return 2 * _mlp_forward_flops(batch_size, dims)


def estimate_value_forward_flops(
    *,
    batch_size: int,
    state_dim: int,
    hidden_dim: int,
    n_hidden: int,
) -> int:
    dims = [state_dim, *([hidden_dim] * n_hidden), 1]
    return _mlp_forward_flops(batch_size, dims)


def build_static_compute_report(
    *,
    config: Any,
    policy: torch.nn.Module,
    qf: torch.nn.Module,
    vf: torch.nn.Module,
    state_dim: int,
    act_dim: int,
) -> Dict[str, Any]:
    policy_param_count = count_parameters(policy)
    qf_param_count = count_parameters(qf)
    vf_param_count = count_parameters(vf)

    policy_trainable_param_count = count_parameters(policy, trainable_only=True)
    qf_trainable_param_count = count_parameters(qf, trainable_only=True)
    vf_trainable_param_count = count_parameters(vf, trainable_only=True)

    policy_forward_flops = estimate_policy_forward_flops(
        batch_size=config.batch_size,
        context_len=config.K,
        hidden_size=config.embed_dim,
        num_layers=config.n_layer,
        num_heads=config.n_head,
        state_dim=state_dim,
        act_dim=act_dim,
        use_attnres=config.use_attnres,
        query_mode=config.attnres_query_mode,
        conditioner_hidden_dim=config.attnres_conditioner_hidden_dim,
        num_blocks=config.attnres_num_blocks,
        use_pre_attn=config.attnres_apply_pre_attn,
        use_pre_mlp=config.attnres_apply_pre_mlp,
    )
    twinq_forward_flops = estimate_twinq_forward_flops(
        batch_size=config.batch_size,
        state_dim=state_dim,
        act_dim=act_dim,
        hidden_dim=config.hidden_dim,
        n_hidden=config.n_hidden,
    )
    value_forward_flops = estimate_value_forward_flops(
        batch_size=config.batch_size,
        state_dim=state_dim,
        hidden_dim=config.hidden_dim,
        n_hidden=config.n_hidden,
    )

    approx_policy_update_flops = int((3 * policy_forward_flops) + twinq_forward_flops)
    approx_critic_minibatch_flops = int((4 * twinq_forward_flops) + (4 * value_forward_flops))
    approx_total_train_step_flops = int((100 * approx_critic_minibatch_flops) + approx_policy_update_flops)

    return {
        "policy_param_count": policy_param_count,
        "policy_trainable_param_count": policy_trainable_param_count,
        "qf_param_count": qf_param_count,
        "qf_trainable_param_count": qf_trainable_param_count,
        "vf_param_count": vf_param_count,
        "vf_trainable_param_count": vf_trainable_param_count,
        "total_param_count": policy_param_count + qf_param_count + vf_param_count,
        "total_trainable_param_count": (
            policy_trainable_param_count + qf_trainable_param_count + vf_trainable_param_count
        ),
        "policy_tokens_per_step": int(config.batch_size * config.K * 3),
        "approx_policy_forward_flops": policy_forward_flops,
        "approx_policy_update_flops": approx_policy_update_flops,
        "approx_critic_minibatch_flops": approx_critic_minibatch_flops,
        "approx_total_train_step_flops": approx_total_train_step_flops,
    }


@dataclass
class ComputeTracker:
    base_report: Dict[str, Any]
    tokens_per_step: int
    step_times_by_phase: Dict[str, list[float]] = field(default_factory=dict)
    peak_gpu_memory_mb: Optional[float] = None

    def record_train_step(self, *, phase: str, step_time_sec: float) -> Dict[str, float]:
        phase_times = self.step_times_by_phase.setdefault(phase, [])
        phase_times.append(float(step_time_sec))
        steps_per_sec = 0.0 if step_time_sec <= 0 else (1.0 / step_time_sec)
        tokens_per_sec = 0.0 if step_time_sec <= 0 else (self.tokens_per_step / step_time_sec)
        if torch.cuda.is_available():
            peak_bytes = float(torch.cuda.max_memory_allocated())
            self.peak_gpu_memory_mb = max(
                0.0 if self.peak_gpu_memory_mb is None else self.peak_gpu_memory_mb,
                peak_bytes / (1024.0 * 1024.0),
            )
        return {
            "train_step_time_sec": float(step_time_sec),
            "steps_per_sec": float(steps_per_sec),
            "policy_tokens_per_sec": float(tokens_per_sec),
        }

    def finalize(
        self,
        *,
        total_wall_clock_sec: float,
        offline_wall_clock_sec: float = 0.0,
        online_wall_clock_sec: float = 0.0,
        eval_wall_clock_sec: float = 0.0,
        reeval_wall_clock_sec: float = 0.0,
    ) -> Dict[str, Any]:
        report = dict(self.base_report)
        report["total_wall_clock_sec"] = float(total_wall_clock_sec)
        report["offline_wall_clock_sec"] = float(offline_wall_clock_sec)
        report["online_wall_clock_sec"] = float(online_wall_clock_sec)
        report["eval_wall_clock_sec"] = float(eval_wall_clock_sec)
        report["reeval_wall_clock_sec"] = float(reeval_wall_clock_sec)
        report["peak_gpu_memory_mb"] = self.peak_gpu_memory_mb
        for phase, values in self.step_times_by_phase.items():
            if not values:
                continue
            mean_step_time = float(sum(values) / len(values))
            report[f"{phase}_mean_train_step_time_sec"] = mean_step_time
            report[f"{phase}_mean_steps_per_sec"] = 0.0 if mean_step_time <= 0 else float(1.0 / mean_step_time)
            report[f"{phase}_mean_policy_tokens_per_sec"] = (
                0.0 if mean_step_time <= 0 else float(self.tokens_per_step / mean_step_time)
            )
        if self.step_times_by_phase:
            all_values = [value for values in self.step_times_by_phase.values() for value in values]
            if all_values:
                mean_step_time = float(sum(all_values) / len(all_values))
                report["mean_train_step_time_sec"] = mean_step_time
                report["mean_steps_per_sec"] = 0.0 if mean_step_time <= 0 else float(1.0 / mean_step_time)
                report["mean_policy_tokens_per_sec"] = (
                    0.0 if mean_step_time <= 0 else float(self.tokens_per_step / mean_step_time)
                )
        return report
