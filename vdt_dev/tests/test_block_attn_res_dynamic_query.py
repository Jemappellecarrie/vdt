from __future__ import annotations

import torch

from vdt_dev.models.block_attn_res import BlockAttentionResidual


def test_block_attn_res_static_route_path_still_runs() -> None:
    router = BlockAttentionResidual(
        hidden_dim=8,
        num_transformer_layers=2,
        num_blocks=2,
        apply_pre_attn=True,
        apply_pre_mlp=True,
    )
    state = router.initialize_state(torch.randn(2, 6, 8))

    output = router.route("pre_attn", 0, state)

    assert output.routed_hidden.shape == (2, 6, 8)
    assert output.query_delta is None
    torch.testing.assert_close(
        output.fused_query,
        output.static_query.view(1, 1, 8).expand(2, 6, 8),
    )


def test_block_attn_res_adds_dynamic_query_delta() -> None:
    router = BlockAttentionResidual(
        hidden_dim=8,
        num_transformer_layers=2,
        num_blocks=2,
        apply_pre_attn=True,
        apply_pre_mlp=True,
        query_mode="state",
    )
    state = router.initialize_state(torch.randn(2, 6, 8))
    query_delta = torch.randn(2, 6, 8)

    output = router.route("pre_attn", 0, state, query_delta=query_delta)

    assert output.routed_hidden.shape == (2, 6, 8)
    assert output.weights.shape[:2] == (2, 6)
    torch.testing.assert_close(
        output.fused_query,
        output.static_query.view(1, 1, 8).expand(2, 6, 8) + query_delta,
    )
