from __future__ import annotations

from pathlib import Path

import torch

from vdt_dev.models.vdt_vanilla_policy import VDTVanillaPolicy
from vdt_dev.runner import ExperimentConfig, build_manifest, build_policy, ensure_output_layout


def _build_inputs(batch_size: int = 2, seq_len: int = 4) -> tuple[torch.Tensor, ...]:
    states = torch.randn(batch_size, seq_len, 11)
    actions = torch.randn(batch_size, seq_len, 3)
    returns_to_go = torch.randn(batch_size, seq_len, 1)
    timesteps = torch.arange(seq_len).repeat(batch_size, 1)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    return states, actions, returns_to_go, timesteps, attention_mask


def test_vanilla_dev_policy_forward_backward_smoke() -> None:
    states, actions, returns_to_go, timesteps, attention_mask = _build_inputs()
    policy = VDTVanillaPolicy(
        state_dim=11,
        act_dim=3,
        hidden_size=32,
        action_range=[-1.0, 1.0],
        max_length=4,
        max_ep_len=16,
        n_layer=2,
        n_head=4,
        n_positions=32,
    )

    state_preds, action_preds, _ = policy(
        states,
        actions,
        returns_to_go=returns_to_go,
        timesteps=timesteps,
        attention_mask=attention_mask,
    )
    loss = state_preds.pow(2).mean() + action_preds.pow(2).mean()
    loss.backward()

    assert state_preds.shape == (2, 4, 11)
    assert action_preds.shape == (2, 4, 3)
    assert any(parameter.grad is not None for parameter in policy.parameters())


def test_runner_build_policy_and_manifest_support_vanilla_dev(tmp_path: Path) -> None:
    config = ExperimentConfig(
        output_dir=str(tmp_path / "vanilla_run"),
        model_variant="vanilla_dev",
        use_attnres=False,
        attnres_query_mode="static",
        embed_dim=32,
        n_layer=2,
        n_head=4,
        K=4,
    )
    policy = build_policy(
        config,
        obs_dim=11,
        act_dim=3,
        max_ep_len=1000,
        action_range=[-1.0, 1.0],
        scale=1000.0,
        target_entropy=-3.0,
    )
    paths = ensure_output_layout(config)
    manifest = build_manifest(config, paths, git_hash="test-git-hash")

    assert isinstance(policy, VDTVanillaPolicy)
    assert manifest.model_variant == "vanilla_dev"
    assert "vanilla_dev" in manifest.run_id
    assert manifest.artifacts.compute_summary_path is not None
