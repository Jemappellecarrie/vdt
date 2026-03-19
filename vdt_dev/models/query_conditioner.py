from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


QUERY_MODES: Tuple[str, ...] = (
    "static",
    "state",
    "state_rtg",
    "state_rtg_value",
)


@dataclass(frozen=True)
class RoutingContext:
    """
    Timestep-aligned context extracted from the DT token stream.

    Step 2 keeps Step 1's learned static query as the routing base and adds a dynamic
    delta on top. That additive form is intentionally less disruptive than replacing the
    base query outright, because the static router remains a stable default when the
    dynamic path is near zero at initialization.
    """

    mode: str
    rtg_tokens: torch.Tensor
    state_tokens: torch.Tensor
    action_tokens: torch.Tensor
    timestep_context: torch.Tensor
    token_context: torch.Tensor
    value_features: Optional[torch.Tensor] = None

    def component_norms(self) -> Dict[str, torch.Tensor]:
        summary: Dict[str, torch.Tensor] = {
            "state_context_norm": self.state_tokens.norm(dim=-1),
        }
        if self.mode in {"state_rtg", "state_rtg_value"}:
            summary["rtg_context_norm"] = self.rtg_tokens.norm(dim=-1)
        if self.value_features is not None:
            summary["value_context_abs"] = self.value_features.abs().squeeze(-1)
        return summary


class RoutingContextExtractor:
    """
    Extract timestep context from the Step 1 DT token layout.

    The dev actor stacks tokens as ``(rtg_t, state_t, action_t)`` for each timestep.
    Step 2 first builds a timestep-aligned context and then broadcasts it back to token
    positions so BAR can consume token-aware query deltas without changing its
    source-axis residual mixing semantics.
    """

    TOKENS_PER_TIMESTEP = 3
    RTG_TOKEN_INDEX = 0
    STATE_TOKEN_INDEX = 1
    ACTION_TOKEN_INDEX = 2

    def split_stacked_tokens(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if hidden_states.ndim != 3:
            raise ValueError(
                f"Expected hidden_states shape [B, T, D], got {hidden_states.shape}."
            )
        batch_size, total_tokens, hidden_dim = hidden_states.shape
        if total_tokens % self.TOKENS_PER_TIMESTEP != 0:
            raise ValueError(
                "The dev DT token layout expects the sequence length to be divisible by "
                f"{self.TOKENS_PER_TIMESTEP}, got {total_tokens}."
            )

        num_timesteps = total_tokens // self.TOKENS_PER_TIMESTEP
        reshaped = hidden_states.reshape(
            batch_size, num_timesteps, self.TOKENS_PER_TIMESTEP, hidden_dim
        )
        return (
            reshaped[:, :, self.RTG_TOKEN_INDEX],
            reshaped[:, :, self.STATE_TOKEN_INDEX],
            reshaped[:, :, self.ACTION_TOKEN_INDEX],
        )

    @staticmethod
    def broadcast_timestep_context(timestep_context: torch.Tensor) -> torch.Tensor:
        if timestep_context.ndim != 3:
            raise ValueError(
                "Expected timestep_context shape [B, K, C], "
                f"got {timestep_context.shape}."
            )
        batch_size, num_timesteps, context_dim = timestep_context.shape
        return (
            timestep_context.unsqueeze(2)
            .expand(batch_size, num_timesteps, 3, context_dim)
            .reshape(batch_size, num_timesteps * 3, context_dim)
        )

    def extract(
        self,
        hidden_states: torch.Tensor,
        *,
        query_mode: str,
        value_features: Optional[torch.Tensor] = None,
    ) -> RoutingContext:
        if query_mode not in QUERY_MODES:
            raise ValueError(
                f"Unsupported query_mode `{query_mode}`. Expected one of {QUERY_MODES}."
            )

        rtg_tokens, state_tokens, action_tokens = self.split_stacked_tokens(hidden_states)
        normalized_value_features: Optional[torch.Tensor] = None

        if query_mode == "static":
            timestep_context = state_tokens.new_zeros(state_tokens.shape)
        elif query_mode == "state":
            timestep_context = state_tokens
        elif query_mode == "state_rtg":
            timestep_context = torch.cat((state_tokens, rtg_tokens), dim=-1)
        else:
            if value_features is None:
                raise ValueError(
                    "`state_rtg_value` routing requires detached value features with "
                    "shape [B, K] or [B, K, 1]."
                )
            if value_features.device != hidden_states.device:
                raise ValueError(
                    "value_features must be on the same device as hidden_states: "
                    f"{value_features.device} vs {hidden_states.device}."
                )
            if value_features.ndim == 2:
                normalized_value_features = value_features.unsqueeze(-1)
            elif value_features.ndim == 3 and value_features.shape[-1] == 1:
                normalized_value_features = value_features
            else:
                raise ValueError(
                    "`value_features` must have shape [B, K] or [B, K, 1], "
                    f"got {value_features.shape}."
                )
            if normalized_value_features.shape[:2] != state_tokens.shape[:2]:
                raise ValueError(
                    "value_features must align with timestep count. Expected "
                    f"{state_tokens.shape[:2]}, got {normalized_value_features.shape[:2]}."
                )
            normalized_value_features = normalized_value_features.to(
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            timestep_context = torch.cat(
                (state_tokens, rtg_tokens, normalized_value_features),
                dim=-1,
            )

        token_context = self.broadcast_timestep_context(timestep_context)
        return RoutingContext(
            mode=query_mode,
            rtg_tokens=rtg_tokens,
            state_tokens=state_tokens,
            action_tokens=action_tokens,
            timestep_context=timestep_context,
            token_context=token_context,
            value_features=normalized_value_features,
        )


class QueryConditioner(nn.Module):
    """
    Builds per-layer BAR query deltas from DT token context.

    Step 2 only conditions on state, RTG, and detached ``V(s_t)``. We intentionally do
    not use Q, advantage, or uncertainty here so we can answer the narrower research
    question first: does decision-context routing improve on top of Step 1's static BAR
    without changing the original VDT learning algorithm?
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        num_transformer_layers: int,
        apply_pre_attn: bool = True,
        apply_pre_mlp: bool = True,
        query_mode: str = "static",
        query_fusion: str = "additive",
        conditioner_hidden_dim: int = 128,
        use_value_stopgrad: bool = True,
        value_feature_mode: str = "scalar_v",
    ) -> None:
        super().__init__()
        if query_mode not in QUERY_MODES:
            raise ValueError(
                f"Unsupported query_mode `{query_mode}`. Expected one of {QUERY_MODES}."
            )
        if query_fusion != "additive":
            raise NotImplementedError(
                "Step 2 only supports additive query fusion: q = w + delta_q."
            )
        if value_feature_mode != "scalar_v":
            raise NotImplementedError(
                "Step 2 only supports scalar detached V(s_t) routing features."
            )
        if not apply_pre_attn and not apply_pre_mlp:
            raise ValueError("At least one routing site must be active.")

        self.hidden_dim = hidden_dim
        self.num_transformer_layers = num_transformer_layers
        self.query_mode = query_mode
        self.query_fusion = query_fusion
        self.conditioner_hidden_dim = conditioner_hidden_dim
        self.use_value_stopgrad = use_value_stopgrad
        self.value_feature_mode = value_feature_mode
        self.context_extractor = RoutingContextExtractor()
        self.active_site_kinds = tuple(
            site_kind
            for site_kind, is_active in (
                ("pre_attn", apply_pre_attn),
                ("pre_mlp", apply_pre_mlp),
            )
            if is_active
        )

        if self.query_mode == "state":
            context_dim = hidden_dim
        elif self.query_mode == "state_rtg":
            context_dim = 2 * hidden_dim
        elif self.query_mode == "state_rtg_value":
            context_dim = (2 * hidden_dim) + 1
        else:
            context_dim = hidden_dim

        if self.uses_dynamic_queries:
            self.conditioner = nn.Sequential(
                nn.Linear(context_dim, conditioner_hidden_dim),
                nn.SiLU(),
                nn.Linear(conditioner_hidden_dim, conditioner_hidden_dim),
                nn.SiLU(),
            )
            self.output_heads = nn.ModuleDict(
                {
                    site_kind: nn.ModuleList(
                        [
                            nn.Linear(conditioner_hidden_dim, hidden_dim)
                            for _ in range(num_transformer_layers)
                        ]
                    )
                    for site_kind in self.active_site_kinds
                }
            )
        else:
            self.conditioner = None
            self.output_heads = nn.ModuleDict()

        self.reset_parameters()

    @property
    def uses_dynamic_queries(self) -> bool:
        return self.query_mode != "static"

    @property
    def requires_value_features(self) -> bool:
        return self.query_mode == "state_rtg_value"

    def reset_parameters(self) -> None:
        if self.conditioner is not None:
            for module in self.conditioner:
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, mean=0.0, std=0.02)
                    nn.init.zeros_(module.bias)

        # Zero-init keeps the dynamic branch near zero so Step 2 begins from the Step 1
        # static router and only learns to deviate once the context signals become
        # useful. That is especially important for value-conditioned routing because the
        # detached scalar V(s_t) should not destabilize the actor early in training.
        for heads in self.output_heads.values():
            for head in heads:
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)

    def build_context(
        self,
        hidden_states: torch.Tensor,
        *,
        value_features: Optional[torch.Tensor] = None,
    ) -> RoutingContext:
        if value_features is not None and self.use_value_stopgrad:
            value_features = value_features.detach()
        return self.context_extractor.extract(
            hidden_states,
            query_mode=self.query_mode,
            value_features=value_features,
        )

    def get_query_delta(
        self,
        hidden_states: torch.Tensor,
        *,
        layer_index: int,
        site_kind: str,
        value_features: Optional[torch.Tensor] = None,
        token_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, RoutingContext]:
        if site_kind not in self.active_site_kinds:
            raise KeyError(f"Routing site `{site_kind}` is not active in the conditioner.")
        if not 0 <= layer_index < self.num_transformer_layers:
            raise IndexError(
                f"layer_index must be in [0, {self.num_transformer_layers}), got {layer_index}."
            )

        context = self.build_context(hidden_states, value_features=value_features)
        if self.conditioner is None:
            query_delta = hidden_states.new_zeros(hidden_states.shape)
        else:
            conditioned_context = self.conditioner(context.token_context)
            query_delta = self.output_heads[site_kind][layer_index](conditioned_context)

        if token_mask is not None:
            if token_mask.shape != hidden_states.shape[:2]:
                raise ValueError(
                    "token_mask must have shape [B, T] aligned with hidden_states, "
                    f"got {token_mask.shape} for hidden_states {hidden_states.shape}."
                )
            query_delta = query_delta * token_mask.unsqueeze(-1).to(dtype=query_delta.dtype)

        return query_delta, context
