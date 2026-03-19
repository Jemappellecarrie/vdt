from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from vdt_dev.models.vdt_bar_policy import VDTBARPolicy
from vdt_dev.models.vdt_vanilla_policy import VDTVanillaPolicy
from vdt_dev.utils.compute import count_parameters, env_dimensions


SEARCH_EMBED_DIMS: tuple[int, ...] = tuple(range(96, 321, 16))
SEARCH_HEADS: tuple[int, ...] = (1, 2, 4, 8)


@dataclass(frozen=True)
class MatchedBudgetSpec:
    model_variant: str
    num_layers: int
    embed_dim: int
    n_head: int
    parameter_count: int
    target_parameter_count: int
    parameter_gap_pct: float


def _build_policy_for_budget(
    *,
    env_name: str,
    model_variant: str,
    num_layers: int,
    embed_dim: int,
    n_head: int,
) -> VDTBARPolicy:
    state_dim, act_dim = env_dimensions(env_name)
    common_kwargs = dict(
        state_dim=state_dim,
        act_dim=act_dim,
        hidden_size=embed_dim,
        action_range=[-1.0, 1.0],
        max_length=20,
        max_ep_len=1000,
        n_layer=num_layers,
        n_head=n_head,
        n_positions=1024,
    )
    if model_variant == "vanilla_dev":
        return VDTVanillaPolicy(**common_kwargs)
    if model_variant == "bar":
        return VDTBARPolicy(
            **common_kwargs,
            use_attnres=True,
            attnres_query_mode="static",
            attnres_num_blocks=8,
        )
    if model_variant == "vcdr":
        return VDTBARPolicy(
            **common_kwargs,
            use_attnres=True,
            attnres_query_mode="state_rtg_value",
            attnres_conditioner_hidden_dim=128,
            attnres_num_blocks=8,
        )
    raise ValueError(f"Unsupported model_variant `{model_variant}`.")


def parameter_count_for_variant(
    *,
    env_name: str,
    model_variant: str,
    num_layers: int,
    embed_dim: int,
    n_head: int,
) -> int:
    policy = _build_policy_for_budget(
        env_name=env_name,
        model_variant=model_variant,
        num_layers=num_layers,
        embed_dim=embed_dim,
        n_head=n_head,
    )
    return count_parameters(policy)


def match_parameter_budget_for_variant(
    *,
    env_name: str,
    model_variant: str,
    num_layers: int,
    target_parameter_count: int,
) -> MatchedBudgetSpec:
    best_candidate: MatchedBudgetSpec | None = None
    for embed_dim in SEARCH_EMBED_DIMS:
        for n_head in SEARCH_HEADS:
            if embed_dim % n_head != 0:
                continue
            parameter_count = parameter_count_for_variant(
                env_name=env_name,
                model_variant=model_variant,
                num_layers=num_layers,
                embed_dim=embed_dim,
                n_head=n_head,
            )
            gap_pct = abs(parameter_count - target_parameter_count) / float(target_parameter_count)
            candidate = MatchedBudgetSpec(
                model_variant=model_variant,
                num_layers=num_layers,
                embed_dim=embed_dim,
                n_head=n_head,
                parameter_count=parameter_count,
                target_parameter_count=target_parameter_count,
                parameter_gap_pct=gap_pct * 100.0,
            )
            if best_candidate is None:
                best_candidate = candidate
                continue
            if candidate.parameter_gap_pct < best_candidate.parameter_gap_pct:
                best_candidate = candidate
                continue
            if (
                candidate.parameter_gap_pct == best_candidate.parameter_gap_pct
                and candidate.embed_dim > best_candidate.embed_dim
            ):
                best_candidate = candidate
    if best_candidate is None:
        raise RuntimeError("Failed to find a matched-budget candidate.")
    return best_candidate


def build_matched_budget_specs(
    *,
    env_name: str,
    model_variant: str,
    num_layers: Sequence[int],
    reference_layers: int = 6,
    reference_embed_dim: int = 256,
    reference_n_head: int = 4,
) -> list[MatchedBudgetSpec]:
    target_parameter_count = parameter_count_for_variant(
        env_name=env_name,
        model_variant=model_variant,
        num_layers=reference_layers,
        embed_dim=reference_embed_dim,
        n_head=reference_n_head,
    )
    specs: list[MatchedBudgetSpec] = []
    for depth in num_layers:
        if depth == reference_layers:
            specs.append(
                MatchedBudgetSpec(
                    model_variant=model_variant,
                    num_layers=depth,
                    embed_dim=reference_embed_dim,
                    n_head=reference_n_head,
                    parameter_count=target_parameter_count,
                    target_parameter_count=target_parameter_count,
                    parameter_gap_pct=0.0,
                )
            )
            continue
        specs.append(
            match_parameter_budget_for_variant(
                env_name=env_name,
                model_variant=model_variant,
                num_layers=depth,
                target_parameter_count=target_parameter_count,
            )
        )
    return specs
