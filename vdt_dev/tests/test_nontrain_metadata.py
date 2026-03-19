from __future__ import annotations

from pathlib import Path

from vdt_dev.analysis.manifest import load_run_manifest, write_run_manifest
from vdt_dev.analysis.schema import RunManifest
from vdt_dev.runner import ExperimentConfig, ensure_nontrain_run_metadata, ensure_output_layout


def test_ensure_nontrain_run_metadata_creates_manifest_for_fresh_output_dir(tmp_path: Path) -> None:
    output_dir = tmp_path / "fresh_eval_output"
    config = ExperimentConfig(output_dir=str(output_dir), job="eval", env_name="hopper-medium-v2")
    paths = ensure_output_layout(config)

    created = ensure_nontrain_run_metadata(config, paths)

    assert created is True
    manifest = load_run_manifest(output_dir)
    assert manifest is not None
    assert manifest.job == "eval"
    assert manifest.output_dir == str(output_dir)


def test_ensure_nontrain_run_metadata_does_not_overwrite_existing_manifest(tmp_path: Path) -> None:
    output_dir = tmp_path / "existing_run"
    (output_dir / "metadata").mkdir(parents=True)
    write_run_manifest(
        output_dir / "metadata/run_manifest.json",
        RunManifest(
            run_id="existing",
            exp_name="train_run",
            env_name="hopper-medium-v2",
            output_dir=str(output_dir),
            job="train",
            status="completed",
        ),
    )

    config = ExperimentConfig(output_dir=str(output_dir), job="reeval_rtg_grid", env_name="hopper-medium-v2")
    paths = ensure_output_layout(config)
    created = ensure_nontrain_run_metadata(config, paths)

    assert created is False
    manifest = load_run_manifest(output_dir)
    assert manifest is not None
    assert manifest.run_id == "existing"
    assert manifest.job == "train"
