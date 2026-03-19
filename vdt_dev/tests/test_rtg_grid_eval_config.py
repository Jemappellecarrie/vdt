from __future__ import annotations

from vdt_dev.runner import ExperimentConfig, _parse_rtg_grid


def test_rtg_grid_config_parsing() -> None:
    config = ExperimentConfig(reeval_rtg_grid="7200, 3600,1800")
    assert _parse_rtg_grid(config.reeval_rtg_grid) == [7200.0, 3600.0, 1800.0]
