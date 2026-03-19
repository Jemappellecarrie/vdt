from __future__ import annotations

import torch

from vdt_dev.models.vdt_bar_policy import VDTBARPolicy


def build_inputs(batch_size: int = 2, seq_len: int = 4) -> tuple[torch.Tensor, ...]:
    states = torch.randn(batch_size, seq_len, 11)
    actions = torch.randn(batch_size, seq_len, 3)
    returns_to_go = torch.randn(batch_size, seq_len, 1)
    timesteps = torch.arange(seq_len).repeat(batch_size, 1)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    return states, actions, returns_to_go, timesteps, attention_mask


def test_vdt_bar_forward_matches_expected_output_shapes() -> None:
    states, actions, returns_to_go, timesteps, attention_mask = build_inputs()
    baseline_like = VDTBARPolicy(
        state_dim=11,
        act_dim=3,
        hidden_size=32,
        action_range=[-1.0, 1.0],
        max_length=4,
        max_ep_len=16,
        n_layer=2,
        n_head=4,
        n_positions=32,
        use_attnres=False,
    )
    vdt_bar = VDTBARPolicy(
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
    )

    baseline_outputs = baseline_like(
        states,
        actions,
        returns_to_go=returns_to_go,
        timesteps=timesteps,
        attention_mask=attention_mask,
    )
    bar_outputs = vdt_bar(
        states,
        actions,
        returns_to_go=returns_to_go,
        timesteps=timesteps,
        attention_mask=attention_mask,
    )

    assert baseline_outputs[0].shape == bar_outputs[0].shape == (2, 4, 11)
    assert baseline_outputs[1].shape == bar_outputs[1].shape == (2, 4, 3)
    assert bar_outputs[2] is None


def test_vdt_bar_get_action_runs_end_to_end() -> None:
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
        attnres_num_blocks=2,
    )

    states = torch.randn(3, 11)
    actions = torch.randn(3, 3)
    returns_to_go = torch.randn(3, 1)
    timesteps = torch.arange(3)

    _, action, _ = policy.get_action(
        states=states,
        actions=actions,
        returns_to_go=returns_to_go,
        timesteps=timesteps,
    )
    assert action.shape == (3,)

