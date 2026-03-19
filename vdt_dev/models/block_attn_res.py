from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """A lightweight RMSNorm used only on routing keys."""

    def __init__(self, hidden_dim: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.weight


class IdentityNorm(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


@dataclass
class RoutingState:
    embedding_source: torch.Tensor
    completed_blocks: List[torch.Tensor] = field(default_factory=list)
    partial_block_sum: torch.Tensor | None = None
    partial_block_count: int = 0
    current_block_index: int = 0


@dataclass
class RoutingOutput:
    routed_hidden: torch.Tensor
    weights: torch.Tensor
    entropy: torch.Tensor
    static_query: torch.Tensor
    query_delta: torch.Tensor | None
    fused_query: torch.Tensor
    source_names: Tuple[str, ...]
    source_values: torch.Tensor
    global_site_index: int
    block_index: int
    site_kind: str
    layer_index: int


class BlockAttentionResidual(nn.Module):
    """
    Static-query Block Attention Residual routing over transformer depth.

    The router runs over the sequence of active residual sites in depth order. In the
    default Step 1 setup, those sites are ``pre-attn`` then ``pre-mlp`` for every
    transformer block. A shared routing state keeps:

    - source 0: the token embedding / initial hidden representation
    - completed BAR block summaries
    - the running partial summary for the current BAR block
    """

    VALID_SITE_KINDS: Tuple[str, ...] = ("pre_attn", "pre_mlp")

    def __init__(
        self,
        hidden_dim: int,
        num_transformer_layers: int,
        num_blocks: int = 8,
        *,
        apply_pre_attn: bool = True,
        apply_pre_mlp: bool = True,
        query_mode: str = "static",
        use_rmsnorm: bool = True,
        zero_init_query: bool = True,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if query_mode not in {"static", "state", "state_rtg", "state_rtg_value"}:
            raise ValueError(
                "Unsupported query_mode. Expected one of "
                "('static', 'state', 'state_rtg', 'state_rtg_value'), "
                f"got `{query_mode}`."
            )
        # Step 1 intentionally uses static queries instead of input-dependent queries so
        # we isolate the effect of depth-wise residual selection itself before adding
        # any state/RTG/value-conditioned routing behavior in later research steps.
        if not apply_pre_attn and not apply_pre_mlp:
            raise ValueError("At least one residual routing site must be active.")
        if num_transformer_layers <= 0:
            raise ValueError("num_transformer_layers must be positive.")

        self.hidden_dim = hidden_dim
        self.num_transformer_layers = num_transformer_layers
        self.apply_pre_attn = apply_pre_attn
        self.apply_pre_mlp = apply_pre_mlp
        self.query_mode = query_mode
        self.use_rmsnorm = use_rmsnorm
        self.zero_init_query = zero_init_query
        self.key_norm = RMSNorm(hidden_dim, eps=eps) if use_rmsnorm else IdentityNorm()

        self.site_order = self._build_site_order()
        self.site_index: Dict[Tuple[str, int], int] = {
            site: idx for idx, site in enumerate(self.site_order)
        }
        self.num_sites = len(self.site_order)
        self.num_blocks = min(max(1, num_blocks), self.num_sites)
        self.block_sizes = self._compute_block_sizes(self.num_sites, self.num_blocks)
        self.site_to_block = self._compute_site_to_block(self.block_sizes)

        query_params: Dict[str, nn.Parameter] = {}
        if apply_pre_attn:
            query_params["pre_attn"] = nn.Parameter(
                torch.empty(num_transformer_layers, hidden_dim)
            )
        if apply_pre_mlp:
            query_params["pre_mlp"] = nn.Parameter(
                torch.empty(num_transformer_layers, hidden_dim)
            )
        self.queries = nn.ParameterDict(query_params)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for query in self.queries.values():
            # Zero-init keeps the initial routing logits equal across all sources, so
            # the model starts from a non-disruptive uniform residual mixture instead of
            # injecting arbitrary depth preferences before learning begins.
            if self.zero_init_query:
                nn.init.zeros_(query)
            else:
                nn.init.normal_(query, mean=0.0, std=0.02)

    def initialize_state(self, embedding_source: torch.Tensor) -> RoutingState:
        if embedding_source.ndim != 3:
            raise ValueError(
                f"Expected embedding source to have shape [B, T, D], got {embedding_source.shape}."
            )
        if embedding_source.shape[-1] != self.hidden_dim:
            raise ValueError(
                "Embedding hidden size does not match router hidden size: "
                f"{embedding_source.shape[-1]} vs {self.hidden_dim}."
            )
        return RoutingState(embedding_source=embedding_source)

    def finalize_state(self, state: RoutingState) -> RoutingState:
        if state.partial_block_count > 0 and state.partial_block_sum is not None:
            state.completed_blocks.append(
                state.partial_block_sum / float(state.partial_block_count)
            )
            state.partial_block_sum = None
            state.partial_block_count = 0
            state.current_block_index = min(
                state.current_block_index + 1, self.num_blocks
            )
        return state

    def route(
        self,
        site_kind: str,
        layer_index: int,
        state: RoutingState,
        query_delta: Optional[torch.Tensor] = None,
    ) -> RoutingOutput:
        global_site_index = self._site_to_global_index(site_kind, layer_index)
        self._advance_state_to_site(global_site_index, state)

        sources, source_names = self._gather_sources(state)
        source_values = torch.stack(sources, dim=-2)

        # RMSNorm on keys keeps routing logits from being dominated by source magnitude.
        # That matters here because completed blocks and partial block summaries can have
        # different scales even when their directions are comparable.
        normalized_keys = self.key_norm(source_values)
        static_query = self.queries[site_kind][layer_index]
        prepared_query_delta = self._prepare_query_delta(
            query_delta,
            batch_size=source_values.shape[0],
            seq_len=source_values.shape[1],
            device=source_values.device,
            dtype=source_values.dtype,
        )

        # Backward compatibility matters here: if no dynamic delta is provided, this is
        # exactly the Step 1 static query path. Step 2 only adds token-aware deltas on
        # top of the same static base query instead of replacing that learned prior.
        fused_query = static_query.view(1, 1, self.hidden_dim).expand(
            source_values.shape[0],
            source_values.shape[1],
            self.hidden_dim,
        )
        if prepared_query_delta is not None:
            fused_query = fused_query + prepared_query_delta

        logits = (normalized_keys * fused_query.unsqueeze(-2)).sum(dim=-1)
        weights = torch.softmax(logits, dim=-1)
        routed_hidden = (weights.unsqueeze(-1) * source_values).sum(dim=-2)
        safe_weights = weights.clamp_min(1e-12)
        entropy = -(safe_weights * safe_weights.log()).sum(dim=-1)

        return RoutingOutput(
            routed_hidden=routed_hidden,
            weights=weights,
            entropy=entropy,
            static_query=static_query,
            query_delta=prepared_query_delta,
            fused_query=fused_query,
            source_names=tuple(source_names),
            source_values=source_values,
            global_site_index=global_site_index,
            block_index=self.site_to_block[global_site_index],
            site_kind=site_kind,
            layer_index=layer_index,
        )

    def update(
        self,
        site_kind: str,
        layer_index: int,
        state: RoutingState,
        representation: torch.Tensor,
    ) -> RoutingState:
        if representation.ndim != 3:
            raise ValueError(
                f"Expected routed representation shape [B, T, D], got {representation.shape}."
            )
        if representation.shape != state.embedding_source.shape:
            raise ValueError(
                "Residual update shape mismatch: "
                f"{representation.shape} vs {state.embedding_source.shape}."
            )

        global_site_index = self._site_to_global_index(site_kind, layer_index)
        self._advance_state_to_site(global_site_index, state)

        if state.partial_block_sum is None:
            state.partial_block_sum = representation
        else:
            state.partial_block_sum = state.partial_block_sum + representation
        state.partial_block_count += 1

        if global_site_index == self.num_sites - 1:
            self.finalize_state(state)
        return state

    def _build_site_order(self) -> List[Tuple[str, int]]:
        site_order: List[Tuple[str, int]] = []
        for layer_index in range(self.num_transformer_layers):
            if self.apply_pre_attn:
                site_order.append(("pre_attn", layer_index))
            if self.apply_pre_mlp:
                site_order.append(("pre_mlp", layer_index))
        return site_order

    @staticmethod
    def _compute_block_sizes(num_sites: int, num_blocks: int) -> List[int]:
        base = num_sites // num_blocks
        remainder = num_sites % num_blocks
        return [base + (1 if block_idx < remainder else 0) for block_idx in range(num_blocks)]

    @staticmethod
    def _compute_site_to_block(block_sizes: Sequence[int]) -> List[int]:
        site_to_block: List[int] = []
        for block_index, block_size in enumerate(block_sizes):
            site_to_block.extend([block_index] * block_size)
        return site_to_block

    def _site_to_global_index(self, site_kind: str, layer_index: int) -> int:
        if site_kind not in self.queries:
            raise KeyError(f"Routing site `{site_kind}` is not active in this router.")
        if not 0 <= layer_index < self.num_transformer_layers:
            raise IndexError(
                f"layer_index must be in [0, {self.num_transformer_layers}), got {layer_index}."
            )
        return self.site_index[(site_kind, layer_index)]

    def _advance_state_to_site(self, global_site_index: int, state: RoutingState) -> None:
        target_block_index = self.site_to_block[global_site_index]
        while state.current_block_index < target_block_index:
            if state.partial_block_sum is None or state.partial_block_count == 0:
                raise RuntimeError(
                    "Attempted to advance to a new BAR block before the current block "
                    "received any routed representations."
                )
            state.completed_blocks.append(
                state.partial_block_sum / float(state.partial_block_count)
            )
            state.partial_block_sum = None
            state.partial_block_count = 0
            state.current_block_index += 1

    def _gather_sources(self, state: RoutingState) -> Tuple[List[torch.Tensor], List[str]]:
        sources: List[torch.Tensor] = [state.embedding_source]
        source_names: List[str] = ["embedding"]

        for block_index, block_representation in enumerate(state.completed_blocks):
            sources.append(block_representation)
            source_names.append(f"block_{block_index}")

        if state.partial_block_sum is not None and state.partial_block_count > 0:
            sources.append(state.partial_block_sum / float(state.partial_block_count))
            source_names.append("partial_block")

        return sources, source_names

    def _prepare_query_delta(
        self,
        query_delta: Optional[torch.Tensor],
        *,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if query_delta is None:
            return None
        if query_delta.device != device:
            raise ValueError(
                "query_delta must be on the same device as routing sources: "
                f"{query_delta.device} vs {device}."
            )
        if query_delta.dtype != dtype:
            raise ValueError(
                "query_delta must have the same dtype as routing sources: "
                f"{query_delta.dtype} vs {dtype}."
            )

        if query_delta.ndim == 4:
            expected_shape = (batch_size, seq_len, 1, self.hidden_dim)
            if query_delta.shape != expected_shape:
                raise ValueError(
                    f"Expected query_delta shape {expected_shape}, got {query_delta.shape}."
                )
            return query_delta.squeeze(-2)

        if query_delta.ndim != 3:
            raise ValueError(
                "query_delta must have shape [B, T, D] or [B, T, 1, D], "
                f"got {query_delta.shape}."
            )

        expected_shape = (batch_size, seq_len, self.hidden_dim)
        if query_delta.shape != expected_shape:
            raise ValueError(
                f"Expected query_delta shape {expected_shape}, got {query_delta.shape}."
            )
        return query_delta
