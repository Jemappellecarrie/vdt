from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


def compute_detached_routing_values(
    policy: nn.Module,
    value_function: Optional[nn.Module],
    states: torch.Tensor,
) -> Optional[torch.Tensor]:
    if not getattr(policy, "requires_value_routing", False):
        return None
    if value_function is None:
        raise ValueError(
            "A value function is required when attnres_query_mode=state_rtg_value."
        )
    if states.ndim == 2:
        batched_states = states.unsqueeze(0)
        squeeze_output = True
    elif states.ndim == 3:
        batched_states = states
        squeeze_output = False
    else:
        raise ValueError(
            "states must have shape [T, state_dim] or [B, T, state_dim], "
            f"got {states.shape}."
        )

    flattened_states = batched_states.reshape(-1, batched_states.shape[-1])
    value_parameter = next(value_function.parameters(), None)
    if value_parameter is not None:
        flattened_states = flattened_states.to(
            device=value_parameter.device,
            dtype=value_parameter.dtype,
        )

    with torch.no_grad():
        values = value_function(flattened_states).reshape(
            batched_states.shape[0], batched_states.shape[1], 1
        )
    values = values.detach()
    if squeeze_output:
        return values.squeeze(0)
    return values


class ValueConditionedPolicyAdapter(nn.Module):
    """
    Thin eval/rollout adapter that injects detached V(s_t) into the dev policy.

    This keeps the original sampling utilities in `src.util` unchanged: they still call
    `get_action` exactly the same way, while the adapter handles the extra routing input
    needed by Step 2's `state_rtg_value` mode.
    """

    def __init__(
        self,
        policy: nn.Module,
        value_function: Optional[nn.Module],
    ) -> None:
        super().__init__()
        self.policy = policy
        self.value_function = value_function
        self.action_range = policy.action_range

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        returns_to_go: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        routing_values = compute_detached_routing_values(
            self.policy,
            self.value_function,
            states,
        )
        return self.policy(
            states,
            actions,
            rewards=rewards,
            targets=targets,
            returns_to_go=returns_to_go,
            timesteps=timesteps,
            attention_mask=attention_mask,
            routing_values=routing_values,
        )

    def get_action(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> tuple[Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
        routing_values = compute_detached_routing_values(
            self.policy,
            self.value_function,
            states,
        )
        return self.policy.get_action(
            states=states,
            actions=actions,
            returns_to_go=returns_to_go,
            timesteps=timesteps,
            routing_values=routing_values,
            **kwargs,
        )
