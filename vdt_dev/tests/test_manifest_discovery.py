from __future__ import annotations

from pathlib import Path

import yaml

from vdt_dev.analysis.manifest import discover_run_dirs, load_run_manifest, write_run_manifest
from vdt_dev.analysis.schema import RunManifest


def test_manifest_discovery_finds_manifest_and_snapshot_runs(tmp_path: Path) -> None:
    manifest_run = tmp_path / "run_with_manifest"
    snapshot_run = tmp_path / "run_with_snapshot"
    (manifest_run / "metadata").mkdir(parents=True)
    (snapshot_run / "metadata").mkdir(parents=True)

    write_run_manifest(
        manifest_run / "metadata/run_manifest.json",
        RunManifest(
            run_id="manifest-run",
            exp_name="exp",
            env_name="hopper-medium-v2",
            output_dir=str(manifest_run),
        ),
    )
    with (snapshot_run / "metadata/config_snapshot.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            {
                "env_name": "walker2d-medium-v2",
                "seed": 2,
                "attnres_query_mode": "state",
                "n_layer": 6,
            },
            handle,
        )

    discovered = discover_run_dirs([tmp_path])
    assert manifest_run in discovered
    assert snapshot_run in discovered

    snapshot_manifest = load_run_manifest(snapshot_run)
    assert snapshot_manifest is not None
    assert snapshot_manifest.query_mode == "state"
    assert snapshot_manifest.env_name == "walker2d-medium-v2"
