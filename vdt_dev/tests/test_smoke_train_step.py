from __future__ import annotations

import math

import torch

from src.value_functions import TwinQ, ValueFunction
from vdt_dev.models.vdt_bar_policy import VDTBARPolicy


def asymmetric_l2_loss(u: torch.Tensor, tau: float) -> torch.Tensor:
    return torch.mean(torch.abs(tau - (u < 0).float()) * u.pow(2))


def manual_sgd_step(module: torch.nn.Module, lr: float) -> None:
    with torch.no_grad():
        for parameter in module.parameters():
            if parameter.grad is None:
                continue
            parameter.add_(parameter.grad, alpha=-lr)
            parameter.grad = None


def test_smoke_train_step_runs_without_nan_or_inf() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 4
    seq_len = 4
    state_dim = 11
    act_dim = 3
    lr = 1e-3

    dataset = {
        "observations": torch.randn(32, state_dim, device=device),
        "actions": torch.randn(32, act_dim, device=device),
        "next_observations": torch.randn(32, state_dim, device=device),
        "rewards": torch.randn(32, device=device),
        "terminals": torch.zeros(32, device=device),
    }

    dt_batch = (
        torch.randn(batch_size, seq_len, state_dim, device=device),
        torch.randn(batch_size, seq_len, act_dim, device=device),
        torch.randn(batch_size, seq_len, 1, device=device),
        torch.randn(batch_size, seq_len, act_dim, device=device),
        torch.zeros(batch_size, seq_len, 1, dtype=torch.long, device=device),
        torch.randn(batch_size, seq_len + 1, 1, device=device),
        torch.arange(seq_len, device=device).repeat(batch_size, 1),
        torch.ones(batch_size, seq_len, device=device),
        torch.randn(batch_size, seq_len, state_dim, device=device),
        torch.zeros(batch_size, seq_len, 1, dtype=torch.bool, device=device),
    )

    policy = VDTBARPolicy(
        state_dim=state_dim,
        act_dim=act_dim,
        hidden_size=32,
        action_range=[-1.0, 1.0],
        max_length=seq_len,
        max_ep_len=16,
        n_layer=2,
        n_head=4,
        n_positions=32,
        use_attnres=True,
        attnres_num_blocks=3,
    ).to(device)
    qf = TwinQ(state_dim, act_dim, hidden_dim=64, n_hidden=2).to(device)
    vf = ValueFunction(state_dim, hidden_dim=64, n_hidden=2).to(device)

    observations = dataset["observations"][:batch_size]
    actions = dataset["actions"][:batch_size]
    next_observations = dataset["next_observations"][:batch_size]
    rewards = dataset["rewards"][:batch_size]
    terminals = dataset["terminals"][:batch_size]

    target_q = qf(observations, actions).detach()
    next_v = vf(next_observations)
    v_loss = asymmetric_l2_loss(target_q - vf(observations), tau=0.7)
    v_loss.backward()
    manual_sgd_step(vf, lr=lr)

    targets = rewards + (1.0 - terminals.float()) * 0.99 * next_v.detach()
    q1, q2 = qf.both(observations, actions)
    q_loss = torch.nn.functional.mse_loss(q1, targets) + torch.nn.functional.mse_loss(q2, targets)
    q_loss.backward()
    manual_sgd_step(qf, lr=lr)

    states, action_inputs, _, action_target, _, rtg, timesteps, attention_mask, _, _ = dt_batch
    state_preds, action_preds, _ = policy(
        states,
        action_inputs,
        returns_to_go=rtg[:, :-1],
        timesteps=timesteps,
        attention_mask=attention_mask,
    )
    action_dim = action_preds.shape[-1]
    action_preds_flat = action_preds.reshape(-1, action_dim)[attention_mask.reshape(-1) > 0]
    action_target_flat = action_target.reshape(-1, action_dim)[attention_mask.reshape(-1) > 0]
    state_loss = ((state_preds[:, :-1] - states[:, 1:]).pow(2))[attention_mask[:, :-1] > 0].mean()
    bc_loss = torch.nn.functional.mse_loss(action_preds_flat, action_target_flat) + state_loss

    actor_states = states.reshape(-1, state_dim)[attention_mask.reshape(-1) > 0]
    q_actor_1, q_actor_2 = qf.both(actor_states, action_preds_flat)
    actor_q_loss = -q_actor_1.mean() / q_actor_2.abs().mean().detach()
    actor_loss = bc_loss + actor_q_loss
    actor_loss.backward()
    manual_sgd_step(policy, lr=lr)

    logs = {
        "v_loss": float(v_loss.detach().cpu()),
        "q_loss": float(q_loss.detach().cpu()),
        "actor_loss": float(actor_loss.detach().cpu()),
    }
    for value in logs.values():
        assert math.isfinite(value)


def test_state_rtg_value_smoke_train_step_runs_without_nan_or_inf() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 4
    seq_len = 4
    state_dim = 11
    act_dim = 3
    lr = 1e-3

    dataset = {
        "observations": torch.randn(32, state_dim, device=device),
        "actions": torch.randn(32, act_dim, device=device),
        "next_observations": torch.randn(32, state_dim, device=device),
        "rewards": torch.randn(32, device=device),
        "terminals": torch.zeros(32, device=device),
    }

    dt_batch = (
        torch.randn(batch_size, seq_len, state_dim, device=device),
        torch.randn(batch_size, seq_len, act_dim, device=device),
        torch.randn(batch_size, seq_len, 1, device=device),
        torch.randn(batch_size, seq_len, act_dim, device=device),
        torch.zeros(batch_size, seq_len, 1, dtype=torch.long, device=device),
        torch.randn(batch_size, seq_len + 1, 1, device=device),
        torch.arange(seq_len, device=device).repeat(batch_size, 1),
        torch.ones(batch_size, seq_len, device=device),
        torch.randn(batch_size, seq_len, state_dim, device=device),
        torch.zeros(batch_size, seq_len, 1, dtype=torch.bool, device=device),
    )

    policy = VDTBARPolicy(
        state_dim=state_dim,
        act_dim=act_dim,
        hidden_size=32,
        action_range=[-1.0, 1.0],
        max_length=seq_len,
        max_ep_len=16,
        n_layer=2,
        n_head=4,
        n_positions=32,
        use_attnres=True,
        attnres_num_blocks=3,
        attnres_query_mode="state_rtg_value",
    ).to(device)
    qf = TwinQ(state_dim, act_dim, hidden_dim=64, n_hidden=2).to(device)
    vf = ValueFunction(state_dim, hidden_dim=64, n_hidden=2).to(device)

    observations = dataset["observations"][:batch_size]
    actions = dataset["actions"][:batch_size]
    next_observations = dataset["next_observations"][:batch_size]
    rewards = dataset["rewards"][:batch_size]
    terminals = dataset["terminals"][:batch_size]

    target_q = qf(observations, actions).detach()
    next_v = vf(next_observations)
    v_loss = asymmetric_l2_loss(target_q - vf(observations), tau=0.7)
    v_loss.backward()
    manual_sgd_step(vf, lr=lr)

    targets = rewards + (1.0 - terminals.float()) * 0.99 * next_v.detach()
    q1, q2 = qf.both(observations, actions)
    q_loss = torch.nn.functional.mse_loss(q1, targets) + torch.nn.functional.mse_loss(q2, targets)
    q_loss.backward()
    manual_sgd_step(qf, lr=lr)

    states, action_inputs, _, action_target, _, rtg, timesteps, attention_mask, _, _ = dt_batch
    with torch.no_grad():
        routing_values = vf(states.reshape(-1, state_dim)).reshape(batch_size, seq_len, 1)
    state_preds, action_preds, _ = policy(
        states,
        action_inputs,
        returns_to_go=rtg[:, :-1],
        timesteps=timesteps,
        attention_mask=attention_mask,
        routing_values=routing_values,
    )
    action_dim = action_preds.shape[-1]
    action_preds_flat = action_preds.reshape(-1, action_dim)[attention_mask.reshape(-1) > 0]
    action_target_flat = action_target.reshape(-1, action_dim)[attention_mask.reshape(-1) > 0]
    state_loss = ((state_preds[:, :-1] - states[:, 1:]).pow(2))[attention_mask[:, :-1] > 0].mean()
    bc_loss = torch.nn.functional.mse_loss(action_preds_flat, action_target_flat) + state_loss

    actor_states = states.reshape(-1, state_dim)[attention_mask.reshape(-1) > 0]
    q_actor_1, q_actor_2 = qf.both(actor_states, action_preds_flat)
    actor_q_loss = -q_actor_1.mean() / q_actor_2.abs().mean().detach()
    actor_loss = bc_loss + actor_q_loss
    actor_loss.backward()
    manual_sgd_step(policy, lr=lr)

    logs = {
        "v_loss": float(v_loss.detach().cpu()),
        "q_loss": float(q_loss.detach().cpu()),
        "actor_loss": float(actor_loss.detach().cpu()),
    }
    for value in logs.values():
        assert math.isfinite(value)
