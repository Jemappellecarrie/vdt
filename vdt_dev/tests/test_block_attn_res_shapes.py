from __future__ import annotations

import torch

from vdt_dev.models.block_attn_res import BlockAttentionResidual


def _run_router_sequence(
    *,
    batch_size: int,
    seq_len: int,
    hidden_dim: int,
    num_layers: int,
    num_blocks: int,
) -> None:
    router = BlockAttentionResidual(
        hidden_dim=hidden_dim,
        num_transformer_layers=num_layers,
        num_blocks=num_blocks,
        apply_pre_attn=True,
        apply_pre_mlp=True,
    )
    embedding = torch.randn(batch_size, seq_len, hidden_dim)
    state = router.initialize_state(embedding)

    for layer_index in range(num_layers):
        attn_output = router.route("pre_attn", layer_index, state)
        assert attn_output.routed_hidden.shape == (batch_size, seq_len, hidden_dim)
        assert attn_output.weights.shape[:2] == (batch_size, seq_len)
        torch.testing.assert_close(
            attn_output.weights.sum(dim=-1),
            torch.ones(batch_size, seq_len),
            atol=1e-6,
            rtol=1e-6,
        )
        state = router.update("pre_attn", layer_index, state, torch.randn_like(attn_output.routed_hidden))

        mlp_output = router.route("pre_mlp", layer_index, state)
        assert mlp_output.routed_hidden.shape == (batch_size, seq_len, hidden_dim)
        torch.testing.assert_close(
            mlp_output.weights.sum(dim=-1),
            torch.ones(batch_size, seq_len),
            atol=1e-6,
            rtol=1e-6,
        )
        state = router.update("pre_mlp", layer_index, state, torch.randn_like(mlp_output.routed_hidden))


def test_block_attn_res_shape_and_softmax_normalization() -> None:
    _run_router_sequence(batch_size=2, seq_len=5, hidden_dim=16, num_layers=3, num_blocks=8)


def test_block_attn_res_supports_multiple_tensor_shapes() -> None:
    for batch_size, seq_len, hidden_dim, num_layers, num_blocks in [
        (1, 3, 8, 2, 2),
        (4, 7, 32, 4, 3),
        (3, 9, 24, 5, 5),
    ]:
        _run_router_sequence(
            batch_size=batch_size,
            seq_len=seq_len,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_blocks=num_blocks,
        )


def test_block_attn_res_non_divisible_block_partition_runs() -> None:
    router = BlockAttentionResidual(
        hidden_dim=12,
        num_transformer_layers=5,
        num_blocks=4,
        apply_pre_attn=True,
        apply_pre_mlp=True,
    )
    state = router.initialize_state(torch.randn(2, 6, 12))
    for layer_index in range(5):
        routed = router.route("pre_attn", layer_index, state)
        state = router.update("pre_attn", layer_index, state, torch.randn_like(routed.routed_hidden))
        routed = router.route("pre_mlp", layer_index, state)
        state = router.update("pre_mlp", layer_index, state, torch.randn_like(routed.routed_hidden))

    assert len(state.completed_blocks) == router.num_blocks

