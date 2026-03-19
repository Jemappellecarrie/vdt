from __future__ import annotations

import numpy as np
import torch

from src.util import vec_evaluate_episode_rtg


class DummyPolicy(torch.nn.Module):
    def __init__(self, act_dim: int) -> None:
        super().__init__()
        self.act_dim = act_dim
        self.action_range = [-1.0, 1.0]

    def get_action(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor | None = None,
        timesteps: torch.Tensor | None = None,
        **_: object,
    ) -> tuple[None, torch.Tensor, None]:
        del states, actions, returns_to_go, timesteps
        return None, torch.full((self.act_dim,), 0.25, dtype=torch.float32), None


class DummySingleEnv:
    def __init__(self, state_dim: int, act_dim: int) -> None:
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.last_action_shape: tuple[int, ...] | None = None

    def reset(self) -> np.ndarray:
        return np.zeros(self.state_dim, dtype=np.float32)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict[str, np.ndarray]]:
        self.last_action_shape = tuple(action.shape)
        if self.last_action_shape != (self.act_dim,):
            raise AssertionError(f"expected single-env action shape {(self.act_dim,)}, got {self.last_action_shape}")
        next_state = np.ones(self.state_dim, dtype=np.float32)
        return next_state, 1.0, True, {}


def test_vec_evaluate_episode_rtg_squeezes_single_env_actions() -> None:
    state_dim = 11
    act_dim = 3
    env = DummySingleEnv(state_dim=state_dim, act_dim=act_dim)
    policy = DummyPolicy(act_dim=act_dim)

    returns, lengths, trajectories = vec_evaluate_episode_rtg(
        env,
        state_dim=state_dim,
        act_dim=act_dim,
        model=policy,
        max_ep_len=4,
        scale=1000.0,
        state_mean=np.zeros(state_dim, dtype=np.float32),
        state_std=np.ones(state_dim, dtype=np.float32),
        target_return=[7.2],
        device=torch.device("cpu"),
    )

    assert env.last_action_shape == (act_dim,)
    assert returns.shape == (1,)
    assert lengths.shape == (1,)
    assert int(lengths[0]) == 1
    assert len(trajectories) == 1
    assert trajectories[0]["actions"].shape == (1, act_dim)
