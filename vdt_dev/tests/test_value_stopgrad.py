from __future__ import annotations

import torch
import torch.nn as nn

from vdt_dev.models.vdt_bar_policy import VDTBARPolicy
from vdt_dev.value_conditioning import compute_detached_routing_values


def test_value_conditioned_routing_does_not_backprop_into_value_features() -> None:
    policy = VDTBARPolicy(
        state_dim=11,
        act_dim=3,
        hidden_size=32,
        action_range=[-1.0, 1.0],
        max_length=4,
        max_ep_len=16,
        n_layer=2,
        n_head=4,
        n_positions=32,
        use_attnres=True,
        attnres_num_blocks=3,
        attnres_query_mode="state_rtg_value",
        attnres_use_value_stopgrad=True,
    )
    value_net = nn.Sequential(
        nn.Linear(11, 32),
        nn.Tanh(),
        nn.Linear(32, 1),
    )

    states = torch.randn(2, 4, 11)
    actions = torch.randn(2, 4, 3)
    returns_to_go = torch.randn(2, 4, 1)
    timesteps = torch.arange(4).repeat(2, 1)
    attention_mask = torch.ones(2, 4, dtype=torch.long)
    routing_values = value_net(states.reshape(-1, 11)).reshape(2, 4, 1)
    routing_values.retain_grad()

    state_preds, action_preds, _ = policy(
        states,
        actions,
        returns_to_go=returns_to_go,
        timesteps=timesteps,
        attention_mask=attention_mask,
        routing_values=routing_values,
    )
    loss = state_preds.pow(2).mean() + action_preds.pow(2).mean()
    loss.backward()

    assert routing_values.grad is None
    for parameter in value_net.parameters():
        assert parameter.grad is None

    conditioner_grad = (
        policy.transformer.query_conditioner.output_heads["pre_attn"][0].weight.grad
    )
    assert conditioner_grad is not None
    assert torch.isfinite(conditioner_grad).all()


def test_detached_routing_values_cast_float64_states_to_value_dtype() -> None:
    policy = VDTBARPolicy(
        state_dim=11,
        act_dim=3,
        hidden_size=32,
        action_range=[-1.0, 1.0],
        max_length=4,
        max_ep_len=16,
        n_layer=2,
        n_head=4,
        n_positions=32,
        use_attnres=True,
        attnres_num_blocks=3,
        attnres_query_mode="state_rtg_value",
        attnres_use_value_stopgrad=True,
    )
    value_net = nn.Sequential(
        nn.Linear(11, 32),
        nn.Tanh(),
        nn.Linear(32, 1),
    )

    states = torch.randn(4, 11, dtype=torch.float64)
    routing_values = compute_detached_routing_values(policy, value_net, states)

    assert routing_values is not None
    assert routing_values.dtype == torch.float32
    assert routing_values.shape == (4, 1)
