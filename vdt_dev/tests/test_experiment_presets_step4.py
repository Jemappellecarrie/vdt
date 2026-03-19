from __future__ import annotations

from pathlib import Path

from vdt_dev.experiments.presets import build_preset_run_specs


def test_same_stack_compare_preset_includes_vanilla_bar_and_vcdr(tmp_path: Path) -> None:
    run_specs = build_preset_run_specs(
        "hopper_same_stack_compare",
        output_root=str(tmp_path),
    )

    assert len(run_specs) == 3
    variants = {run_spec.overrides["model_variant"] for run_spec in run_specs}
    assert variants == {"vanilla_dev", "bar", "vcdr"}


def test_matched_budget_depth_sweep_expands_all_variants_and_depths(tmp_path: Path) -> None:
    run_specs = build_preset_run_specs(
        "hopper_matched_budget_depth_sweep",
        output_root=str(tmp_path),
    )

    assert len(run_specs) == 9
    for run_spec in run_specs:
        assert run_spec.overrides["budget_mode"] == "matched_params"
        assert run_spec.overrides["budget_target_params"] > 0
        assert run_spec.overrides["budget_tolerance_pct"] == 10.0
        assert run_spec.overrides["budget_actual_gap_pct"] >= 0.0
        assert run_spec.overrides["embed_dim"] > 0
        assert run_spec.overrides["n_head"] > 0


def test_online_compare_preset_uses_online_configs(tmp_path: Path) -> None:
    run_specs = build_preset_run_specs(
        "hopper_online_compare",
        output_root=str(tmp_path),
    )

    assert len(run_specs) == 3
    assert all("_online.yaml" in run_spec.config_path for run_spec in run_specs)
