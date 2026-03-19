from __future__ import annotations

import torch

from vdt_dev.models.vdt_bar_policy import VDTBARPolicy


def test_vdt_bar_backward_propagates_to_queries_and_actor_weights() -> None:
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
    )

    states = torch.randn(2, 4, 11)
    actions = torch.randn(2, 4, 3)
    returns_to_go = torch.randn(2, 4, 1)
    timesteps = torch.arange(4).repeat(2, 1)
    attention_mask = torch.ones(2, 4, dtype=torch.long)

    state_preds, action_preds, _ = policy(
        states,
        actions,
        returns_to_go=returns_to_go,
        timesteps=timesteps,
        attention_mask=attention_mask,
    )
    loss = state_preds.pow(2).mean() + action_preds.pow(2).mean()
    loss.backward()

    pre_attn_grad = policy.transformer.routing.queries["pre_attn"].grad
    assert pre_attn_grad is not None
    assert torch.isfinite(pre_attn_grad).all()
    assert pre_attn_grad.abs().sum().item() > 0.0

    embed_grad = policy.embed_state.weight.grad
    assert embed_grad is not None
    assert torch.isfinite(embed_grad).all()
    assert embed_grad.abs().sum().item() > 0.0

