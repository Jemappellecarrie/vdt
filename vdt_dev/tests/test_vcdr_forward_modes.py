from __future__ import annotations

import pytest
import torch

from vdt_dev.models.vdt_bar_policy import VDTBARPolicy


def build_inputs(batch_size: int = 2, seq_len: int = 4) -> tuple[torch.Tensor, ...]:
    states = torch.randn(batch_size, seq_len, 11)
    actions = torch.randn(batch_size, seq_len, 3)
    returns_to_go = torch.randn(batch_size, seq_len, 1)
    timesteps = torch.arange(seq_len).repeat(batch_size, 1)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    routing_values = torch.randn(batch_size, seq_len, 1)
    return states, actions, returns_to_go, timesteps, attention_mask, routing_values


@pytest.mark.parametrize(
    "query_mode",
    ["static", "state", "state_rtg", "state_rtg_value"],
)
def test_vcdr_forward_modes_run(query_mode: str) -> None:
    (
        states,
        actions,
        returns_to_go,
        timesteps,
        attention_mask,
        routing_values,
    ) = build_inputs()
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
        attnres_query_mode=query_mode,
    )

    state_preds, action_preds, reward_preds = policy(
        states,
        actions,
        returns_to_go=returns_to_go,
        timesteps=timesteps,
        attention_mask=attention_mask,
        routing_values=routing_values if query_mode == "state_rtg_value" else None,
    )

    assert state_preds.shape == (2, 4, 11)
    assert action_preds.shape == (2, 4, 3)
    assert reward_preds is None
