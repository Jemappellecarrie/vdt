from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from vdt_dev.models.block_attn_res import BlockAttentionResidual, RMSNorm
from vdt_dev.models.query_conditioner import QueryConditioner, RoutingContext
from vdt_dev.utils.debug_hooks import RoutingDebugHook


ACT2FN: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "relu": F.relu,
    "gelu": F.gelu,
    "silu": F.silu,
    "mish": F.mish,
}


class Conv1D(nn.Module):
    """
    Linear layer with GPT-2 style parameter layout.

    This matches the local `trajectory_gpt2.py` implementation closely enough that we
    can keep the actor backbone structure without importing `transformers` in tests.
    """

    def __init__(self, out_features: int, in_features: int) -> None:
        super().__init__()
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output_shape = x.shape[:-1] + (self.out_features,)
        x = torch.addmm(self.bias, x.reshape(-1, x.shape[-1]), self.weight)
        return x.reshape(output_shape)


class TrajectoryModel(nn.Module):
    def __init__(self, state_dim: int, act_dim: int, max_length: Optional[int] = None) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.max_length = max_length

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[None, None, None]:
        return None, None, None

    def get_action(self, states: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, **kwargs: object) -> torch.Tensor:
        return torch.zeros_like(actions[-1])


@dataclass
class VDTBARTransformerConfig:
    n_embd: int
    n_head: int
    n_layer: int
    n_ctx: int
    n_inner: Optional[int] = None
    activation_function: str = "relu"
    resid_pdrop: float = 0.1
    attn_pdrop: float = 0.1
    embd_pdrop: float = 0.1
    layer_norm_epsilon: float = 1e-5
    initializer_range: float = 0.02


class CausalSelfAttention(nn.Module):
    def __init__(self, hidden_size: int, n_ctx: int, config: VDTBARTransformerConfig) -> None:
        super().__init__()
        if hidden_size % config.n_head != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by n_head ({config.n_head})."
            )

        self.hidden_size = hidden_size
        self.n_head = config.n_head
        self.head_dim = hidden_size // config.n_head
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((n_ctx, n_ctx), dtype=torch.bool)).view(1, 1, n_ctx, n_ctx),
            persistent=False,
        )
        self.masked_bias = -1e4

        self.c_attn = Conv1D(3 * hidden_size, hidden_size)
        self.c_proj = Conv1D(hidden_size, hidden_size)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        new_shape = x.shape[:-1] + (self.n_head, self.head_dim)
        return x.view(*new_shape).permute(0, 2, 1, 3)

    def merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1, 3).contiguous()
        new_shape = x.shape[:-2] + (x.shape[-2] * x.shape[-1],)
        return x.view(*new_shape)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        if seq_len > self.bias.shape[-1]:
            raise ValueError(
                f"Sequence length {seq_len} exceeds configured context {self.bias.shape[-1]}."
            )

        query, key, value = self.c_attn(hidden_states).split(self.hidden_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key)
        value = self.split_heads(value)

        attention_scores = torch.matmul(query, key.transpose(-2, -1))
        attention_scores = attention_scores / math.sqrt(self.head_dim)

        causal_mask = self.bias[:, :, :seq_len, :seq_len]
        attention_scores = attention_scores.masked_fill(~causal_mask, self.masked_bias)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = torch.softmax(attention_scores, dim=-1)
        attention_probs = self.attn_dropout(attention_probs)

        context = torch.matmul(attention_probs, value)
        context = self.merge_heads(context)
        context = self.c_proj(context)
        return self.resid_dropout(context)


class FeedForward(nn.Module):
    def __init__(self, inner_dim: int, config: VDTBARTransformerConfig) -> None:
        super().__init__()
        hidden_size = config.n_embd
        self.c_fc = Conv1D(inner_dim, hidden_size)
        self.c_proj = Conv1D(hidden_size, inner_dim)
        if config.activation_function not in ACT2FN:
            raise KeyError(
                f"Unsupported activation `{config.activation_function}`. "
                f"Available: {sorted(ACT2FN)}"
            )
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.c_fc(x))
        x = self.c_proj(x)
        return self.dropout(x)


class TransformerBlock(nn.Module):
    def __init__(self, config: VDTBARTransformerConfig) -> None:
        super().__init__()
        inner_dim = config.n_inner if config.n_inner is not None else 4 * config.n_embd
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = CausalSelfAttention(config.n_embd, config.n_ctx, config)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = FeedForward(inner_dim, config)


class VDTBARTransformer(nn.Module):
    """
    Minimal GPT-style transformer with optional Block Attention Residual routing.

    The BAR depth partition is applied over the active residual sites:

    - if both routes are enabled, depth = ``2 * n_layer`` with order
      ``pre_attn(0), pre_mlp(0), pre_attn(1), pre_mlp(1), ...``
    - if only one route is enabled, depth = ``n_layer``
    """

    def __init__(
        self,
        config: VDTBARTransformerConfig,
        *,
        use_attnres: bool = False,
        attnres_variant: str = "block",
        attnres_num_blocks: int = 8,
        attnres_apply_pre_attn: bool = True,
        attnres_apply_pre_mlp: bool = True,
        attnres_query_mode: str = "static",
        attnres_query_fusion: str = "additive",
        attnres_conditioner_hidden_dim: int = 128,
        attnres_use_value_stopgrad: bool = True,
        attnres_value_feature_mode: str = "scalar_v",
        attnres_use_rmsnorm: bool = True,
        attnres_zero_init_query: bool = True,
        attnres_debug: bool = False,
        debug_hook: Optional[RoutingDebugHook] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.use_attnres = use_attnres
        self.attnres_variant = attnres_variant

        if use_attnres and attnres_variant != "block":
            raise NotImplementedError("Step 1 only implements the `block` AttnRes variant.")

        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        self.debug_hook = debug_hook or RoutingDebugHook(enabled=attnres_debug)
        if use_attnres:
            self.routing = BlockAttentionResidual(
                hidden_dim=config.n_embd,
                num_transformer_layers=config.n_layer,
                num_blocks=attnres_num_blocks,
                apply_pre_attn=attnres_apply_pre_attn,
                apply_pre_mlp=attnres_apply_pre_mlp,
                query_mode=attnres_query_mode,
                use_rmsnorm=attnres_use_rmsnorm,
                zero_init_query=attnres_zero_init_query,
            )
            self.query_conditioner = QueryConditioner(
                hidden_dim=config.n_embd,
                num_transformer_layers=config.n_layer,
                apply_pre_attn=attnres_apply_pre_attn,
                apply_pre_mlp=attnres_apply_pre_mlp,
                query_mode=attnres_query_mode,
                query_fusion=attnres_query_fusion,
                conditioner_hidden_dim=attnres_conditioner_hidden_dim,
                use_value_stopgrad=attnres_use_value_stopgrad,
                value_feature_mode=attnres_value_feature_mode,
            )
        else:
            self.routing = None
            self.query_conditioner = None

        self.apply(self._init_weights)
        if self.query_conditioner is not None:
            self.query_conditioner.reset_parameters()

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding, Conv1D)):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.LayerNorm, RMSNorm)):
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    @staticmethod
    def _prepare_attention_mask(
        attention_mask: Optional[torch.Tensor],
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if attention_mask is None:
            return None
        if attention_mask.ndim != 2:
            raise ValueError(
                f"Expected attention_mask shape [B, T], got {attention_mask.shape}."
            )
        mask = attention_mask[:, None, None, :].to(dtype=dtype)
        return (1.0 - mask) * -10000.0

    def save_debug_tensors(self, output_path: str | Path) -> None:
        self.debug_hook.save(output_path)

    def forward(
        self,
        *,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        routing_values: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if inputs_embeds.ndim != 3:
            raise ValueError(
                f"Expected inputs_embeds shape [B, T, D], got {inputs_embeds.shape}."
            )

        hidden_states = self.drop(inputs_embeds)
        prepared_attention_mask = self._prepare_attention_mask(
            attention_mask, dtype=hidden_states.dtype
        )
        routing_token_mask = attention_mask if attention_mask is not None else None
        routing_state = self.routing.initialize_state(hidden_states) if self.routing is not None else None

        for layer_index, block in enumerate(self.h):
            attn_base = hidden_states
            attn_context: Optional[RoutingContext]
            if self.routing is not None and self.routing.apply_pre_attn:
                attn_query_delta, attn_context = self.query_conditioner.get_query_delta(
                    hidden_states,
                    layer_index=layer_index,
                    site_kind="pre_attn",
                    value_features=routing_values,
                    token_mask=routing_token_mask,
                )
                attn_routing = self.routing.route(
                    "pre_attn",
                    layer_index,
                    routing_state,
                    query_delta=attn_query_delta,
                )
                attn_base = attn_routing.routed_hidden
            else:
                attn_routing = None
                attn_context = None

            attn_output = block.attn(
                block.ln_1(attn_base),
                attention_mask=prepared_attention_mask,
            )
            hidden_states = attn_base + attn_output

            if self.routing is not None and self.routing.apply_pre_attn:
                routing_state = self.routing.update(
                    "pre_attn", layer_index, routing_state, hidden_states
                )
                self.debug_hook.record(
                    f"layer_{layer_index:02d}.pre_attn",
                    weights=attn_routing.weights,
                    entropy=attn_routing.entropy,
                    hidden=attn_base,
                    output=hidden_states,
                    source_names=attn_routing.source_names,
                    dynamic_query_delta=attn_routing.query_delta,
                    static_query=attn_routing.static_query,
                    fused_query=attn_routing.fused_query,
                    query_mode=self.query_conditioner.query_mode,
                    mode_summary=attn_context.component_norms() if attn_context is not None else None,
                )

            mlp_base = hidden_states
            mlp_context: Optional[RoutingContext]
            if self.routing is not None and self.routing.apply_pre_mlp:
                mlp_query_delta, mlp_context = self.query_conditioner.get_query_delta(
                    hidden_states,
                    layer_index=layer_index,
                    site_kind="pre_mlp",
                    value_features=routing_values,
                    token_mask=routing_token_mask,
                )
                mlp_routing = self.routing.route(
                    "pre_mlp",
                    layer_index,
                    routing_state,
                    query_delta=mlp_query_delta,
                )
                mlp_base = mlp_routing.routed_hidden
            else:
                mlp_routing = None
                mlp_context = None

            feed_forward_hidden = block.mlp(block.ln_2(mlp_base))
            hidden_states = mlp_base + feed_forward_hidden

            if self.routing is not None and self.routing.apply_pre_mlp:
                routing_state = self.routing.update(
                    "pre_mlp", layer_index, routing_state, hidden_states
                )
                self.debug_hook.record(
                    f"layer_{layer_index:02d}.pre_mlp",
                    weights=mlp_routing.weights,
                    entropy=mlp_routing.entropy,
                    hidden=mlp_base,
                    output=hidden_states,
                    source_names=mlp_routing.source_names,
                    dynamic_query_delta=mlp_routing.query_delta,
                    static_query=mlp_routing.static_query,
                    fused_query=mlp_routing.fused_query,
                    query_mode=self.query_conditioner.query_mode,
                    mode_summary=mlp_context.component_norms() if mlp_context is not None else None,
                )

        hidden_states = self.ln_f(hidden_states)
        return {"last_hidden_state": hidden_states}


class VDTBARPolicy(TrajectoryModel):
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        hidden_size: int,
        action_range: tuple[float, float] | list[float],
        *,
        max_length: Optional[int] = None,
        max_ep_len: int = 4096,
        action_tanh: bool = True,
        scale: float = 1.0,
        stochastic_policy: bool = False,
        ordering: int = 0,
        init_temperature: float = 0.1,
        target_entropy: Optional[float] = None,
        n_layer: int = 6,
        n_head: int = 4,
        n_inner: Optional[int] = None,
        activation_function: str = "relu",
        n_positions: int = 1024,
        resid_pdrop: float = 0.1,
        attn_pdrop: float = 0.1,
        embd_pdrop: Optional[float] = None,
        use_attnres: bool = False,
        attnres_variant: str = "block",
        attnres_num_blocks: int = 8,
        attnres_apply_pre_attn: bool = True,
        attnres_apply_pre_mlp: bool = True,
        attnres_query_mode: str = "static",
        attnres_query_fusion: str = "additive",
        attnres_conditioner_hidden_dim: int = 128,
        attnres_use_value_stopgrad: bool = True,
        attnres_value_feature_mode: str = "scalar_v",
        attnres_use_rmsnorm: bool = True,
        attnres_zero_init_query: bool = True,
        attnres_debug: bool = False,
    ) -> None:
        super().__init__(state_dim, act_dim, max_length=max_length)
        if stochastic_policy:
            raise NotImplementedError(
                "The Step 1 dev policy keeps the deterministic actor path used by the current repo."
            )

        self.hidden_size = hidden_size
        self.scale = scale
        self.rtg_no_q = False
        self.infer_no_q = False
        self.ordering = ordering
        self.action_range = tuple(float(x) for x in action_range)
        self.stochastic_policy = stochastic_policy
        self.log_temperature = torch.tensor(np.log(init_temperature))
        self.target_entropy = target_entropy
        self.use_attnres = use_attnres
        self.attnres_query_mode = attnres_query_mode
        self.attnres_use_value_stopgrad = attnres_use_value_stopgrad

        transformer_config = VDTBARTransformerConfig(
            n_embd=hidden_size,
            n_head=n_head,
            n_layer=n_layer,
            n_ctx=n_positions,
            n_inner=n_inner if n_inner is not None else 4 * hidden_size,
            activation_function=activation_function,
            resid_pdrop=resid_pdrop,
            attn_pdrop=attn_pdrop,
            embd_pdrop=resid_pdrop if embd_pdrop is None else embd_pdrop,
        )
        self.transformer = VDTBARTransformer(
            transformer_config,
            use_attnres=use_attnres,
            attnres_variant=attnres_variant,
            attnres_num_blocks=attnres_num_blocks,
            attnres_apply_pre_attn=attnres_apply_pre_attn,
            attnres_apply_pre_mlp=attnres_apply_pre_mlp,
            attnres_query_mode=attnres_query_mode,
            attnres_query_fusion=attnres_query_fusion,
            attnres_conditioner_hidden_dim=attnres_conditioner_hidden_dim,
            attnres_use_value_stopgrad=attnres_use_value_stopgrad,
            attnres_value_feature_mode=attnres_value_feature_mode,
            attnres_use_rmsnorm=attnres_use_rmsnorm,
            attnres_zero_init_query=attnres_zero_init_query,
            attnres_debug=attnres_debug,
        )

        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_ordering = nn.Embedding(max_ep_len, hidden_size)
        self.embed_return = nn.Linear(1, hidden_size)
        self.embed_rewards = nn.Linear(1, hidden_size)
        self.embed_state = nn.Linear(self.state_dim, hidden_size)
        self.embed_action = nn.Linear(self.act_dim, hidden_size)
        self.embed_ln = nn.LayerNorm(hidden_size)

        self.predict_state = nn.Linear(hidden_size, self.state_dim)
        action_head: list[nn.Module] = [nn.Linear(hidden_size, self.act_dim)]
        if action_tanh:
            action_head.append(nn.Tanh())
        self.predict_action = nn.Sequential(*action_head)
        self.predict_rewards = nn.Linear(hidden_size, 1)

        self.apply(self._init_policy_weights)

    def _init_policy_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def temperature(self) -> Optional[torch.Tensor]:
        return self.log_temperature.exp() if self.stochastic_policy else None

    def save_debug_tensors(self, output_path: str | Path) -> None:
        self.transformer.save_debug_tensors(output_path)

    @property
    def requires_value_routing(self) -> bool:
        return self.use_attnres and self.attnres_query_mode == "state_rtg_value"

    def _prepare_routing_values(
        self,
        routing_values: Optional[torch.Tensor],
        *,
        batch_size: int,
        seq_length: int,
        attention_mask: Optional[torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if routing_values is None:
            if self.requires_value_routing:
                raise ValueError(
                    "`routing_values` are required when attnres_query_mode=state_rtg_value."
                )
            return None

        if routing_values.ndim == 1:
            if batch_size != 1 or routing_values.shape[0] != seq_length:
                raise ValueError(
                    "`routing_values` with shape [K] is only valid for single-batch "
                    f"action selection, got batch_size={batch_size}, seq_length={seq_length}, "
                    f"shape={routing_values.shape}."
                )
            prepared = routing_values.view(1, seq_length, 1)
        elif routing_values.ndim == 2:
            if routing_values.shape == (batch_size, seq_length):
                prepared = routing_values.unsqueeze(-1)
            elif batch_size == 1 and routing_values.shape == (seq_length, 1):
                prepared = routing_values.unsqueeze(0)
            else:
                raise ValueError(
                    "`routing_values` must have shape [B, K], [B, K, 1], [K], or [K, 1], "
                    f"got {routing_values.shape}."
                )
        elif routing_values.ndim == 3 and routing_values.shape == (batch_size, seq_length, 1):
            prepared = routing_values
        else:
            raise ValueError(
                "`routing_values` must have shape [B, K], [B, K, 1], [K], or [K, 1], "
                f"got {routing_values.shape}."
            )

        prepared = prepared.to(device=device, dtype=dtype)
        if self.attnres_use_value_stopgrad:
            prepared = prepared.detach()
        if attention_mask is not None:
            prepared = prepared * attention_mask.unsqueeze(-1).to(dtype=prepared.dtype)
        return prepared

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        returns_to_go: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        routing_values: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        del rewards, targets

        batch_size, seq_length = states.shape[:2]
        if returns_to_go is None:
            raise ValueError("returns_to_go must be provided.")
        if timesteps is None:
            raise ValueError("timesteps must be provided.")
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length), dtype=torch.long, device=states.device
            )
        routing_values = self._prepare_routing_values(
            routing_values,
            batch_size=batch_size,
            seq_length=seq_length,
            attention_mask=attention_mask,
            device=states.device,
            dtype=states.dtype,
        )

        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        if self.ordering:
            order_embeddings = self.embed_ordering(timesteps)
            state_embeddings = state_embeddings + order_embeddings
            action_embeddings = action_embeddings + order_embeddings
            returns_embeddings = returns_embeddings + order_embeddings

        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings

        stacked_inputs = (
            torch.stack((returns_embeddings, state_embeddings, action_embeddings), dim=1)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, 3 * seq_length, self.hidden_size)
        )
        stacked_inputs = self.embed_ln(stacked_inputs)
        stacked_attention_mask = (
            torch.stack((attention_mask, attention_mask, attention_mask), dim=1)
            .permute(0, 2, 1)
            .reshape(batch_size, 3 * seq_length)
        )

        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
            routing_values=routing_values,
        )
        hidden_states = transformer_outputs["last_hidden_state"]
        hidden_states = hidden_states.reshape(batch_size, seq_length, 3, self.hidden_size)
        hidden_states = hidden_states.permute(0, 2, 1, 3)

        action_preds = self.predict_action(hidden_states[:, 1])
        state_preds = self.predict_state(hidden_states[:, 2])
        reward_preds = None
        return state_preds, action_preds, reward_preds

    def get_action(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None,
        routing_values: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> tuple[Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
        del kwargs

        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1, self.act_dim)
        if returns_to_go is None or timesteps is None:
            raise ValueError("returns_to_go and timesteps are required for autoregressive action selection.")
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)
        routing_values = self._prepare_routing_values(
            routing_values,
            batch_size=1,
            seq_length=states.shape[1],
            attention_mask=None,
            device=states.device,
            dtype=states.dtype,
        )

        if self.max_length is not None:
            states = states[:, -self.max_length :]
            actions = actions[:, -self.max_length :]
            returns_to_go = returns_to_go[:, -self.max_length :]
            timesteps = timesteps[:, -self.max_length :]
            if routing_values is not None:
                routing_values = routing_values[:, -self.max_length :]

            attention_mask = torch.cat(
                [
                    torch.zeros(self.max_length - states.shape[1], device=states.device),
                    torch.ones(states.shape[1], device=states.device),
                ]
            ).reshape(1, -1)
            attention_mask = attention_mask.to(dtype=torch.long, device=states.device)

            states = torch.cat(
                [
                    torch.zeros(
                        (states.shape[0], self.max_length - states.shape[1], self.state_dim),
                        device=states.device,
                    ),
                    states,
                ],
                dim=1,
            ).to(dtype=torch.float32)
            actions = torch.cat(
                [
                    torch.zeros(
                        (actions.shape[0], self.max_length - actions.shape[1], self.act_dim),
                        device=actions.device,
                    ),
                    actions,
                ],
                dim=1,
            ).to(dtype=torch.float32)
            returns_to_go = torch.cat(
                [
                    torch.zeros(
                        (returns_to_go.shape[0], self.max_length - returns_to_go.shape[1], 1),
                        device=returns_to_go.device,
                    ),
                    returns_to_go,
                ],
                dim=1,
            ).to(dtype=torch.float32)
            if routing_values is not None:
                routing_values = torch.cat(
                    [
                        torch.zeros(
                            (
                                routing_values.shape[0],
                                self.max_length - routing_values.shape[1],
                                1,
                            ),
                            device=routing_values.device,
                        ),
                        routing_values,
                    ],
                    dim=1,
                ).to(dtype=torch.float32)
            timesteps = torch.cat(
                [
                    torch.zeros(
                        (timesteps.shape[0], self.max_length - timesteps.shape[1]),
                        device=timesteps.device,
                    ),
                    timesteps,
                ],
                dim=1,
            ).to(dtype=torch.long)
        else:
            attention_mask = None

        _, action_preds, _ = self.forward(
            states=states,
            actions=actions,
            returns_to_go=returns_to_go,
            timesteps=timesteps,
            attention_mask=attention_mask,
            routing_values=routing_values,
        )
        return None, action_preds[0, -1], None

    def clamp_action(self, action: torch.Tensor) -> torch.Tensor:
        return action.clamp(*self.action_range)


class VDTVCDRPolicy(VDTBARPolicy):
    """Step 2 alias for the dev policy with value-conditioned depth routing support."""
