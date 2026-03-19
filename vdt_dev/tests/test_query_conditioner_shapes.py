from __future__ import annotations

import pytest
import torch

from vdt_dev.models.query_conditioner import QueryConditioner, RoutingContextExtractor


@pytest.mark.parametrize(
    ("query_mode", "expected_context_dim"),
    [
        ("static", 16),
        ("state", 16),
        ("state_rtg", 32),
        ("state_rtg_value", 33),
    ],
)
def test_query_conditioner_shapes_for_all_modes(
    query_mode: str,
    expected_context_dim: int,
) -> None:
    hidden_states = torch.randn(2, 12, 16)
    value_features = torch.randn(2, 4, 1)
    conditioner = QueryConditioner(
        hidden_dim=16,
        num_transformer_layers=3,
        query_mode=query_mode,
    )

    query_delta, context = conditioner.get_query_delta(
        hidden_states,
        layer_index=1,
        site_kind="pre_attn",
        value_features=value_features if query_mode == "state_rtg_value" else None,
    )

    assert query_delta.shape == hidden_states.shape
    assert context.timestep_context.shape == (2, 4, expected_context_dim)
    assert context.token_context.shape == (2, 12, expected_context_dim)
    if query_mode == "static":
        torch.testing.assert_close(query_delta, torch.zeros_like(query_delta))


def test_timestep_context_broadcast_matches_token_layout() -> None:
    timestep_context = torch.tensor(
        [[[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]],
    )

    broadcast = RoutingContextExtractor.broadcast_timestep_context(timestep_context)

    expected = torch.tensor(
        [
            [
                [1.0, 10.0],
                [1.0, 10.0],
                [1.0, 10.0],
                [2.0, 20.0],
                [2.0, 20.0],
                [2.0, 20.0],
                [3.0, 30.0],
                [3.0, 30.0],
                [3.0, 30.0],
            ]
        ]
    )
    torch.testing.assert_close(broadcast, expected)
