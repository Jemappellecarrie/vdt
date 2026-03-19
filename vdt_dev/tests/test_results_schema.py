from __future__ import annotations

from pathlib import Path

from vdt_dev.analysis.schema import (
    ArtifactPaths,
    MetricRecord,
    RunManifest,
    append_jsonl,
    load_dataclass_json,
    read_jsonl,
    write_dataclass_json,
)


def test_results_schema_roundtrip_and_backward_compatible_defaults(tmp_path: Path) -> None:
    manifest = RunManifest(
        run_id="run-123",
        exp_name="example",
        env_name="hopper-medium-v2",
        seed=1,
        query_mode="state_rtg_value",
        num_layers=6,
        hidden_size=256,
        num_heads=4,
        context_len=20,
        num_blocks=8,
        train_steps=1000,
        eval_interval=50,
        output_dir=str(tmp_path),
        artifacts=ArtifactPaths(latest_checkpoint_path="latest.pt"),
    )
    manifest_path = tmp_path / "manifest.json"
    write_dataclass_json(manifest_path, manifest)

    loaded = load_dataclass_json(manifest_path, RunManifest)
    assert loaded.run_id == manifest.run_id
    assert loaded.query_mode == "state_rtg_value"
    assert loaded.artifacts.latest_checkpoint_path == "latest.pt"

    legacy_path = tmp_path / "legacy_manifest.json"
    legacy_path.write_text('{"run_id": "legacy", "env_name": "hopper-medium-v2"}\n', encoding="utf-8")
    legacy_loaded = load_dataclass_json(legacy_path, RunManifest)
    assert legacy_loaded.run_id == "legacy"
    assert legacy_loaded.query_mode == "static"
    assert legacy_loaded.num_layers == 0


def test_metric_record_jsonl_roundtrip(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.jsonl"
    record = MetricRecord(
        step=10,
        phase="offline",
        split="train",
        train_actor_loss=1.5,
        train_q_loss=0.5,
        train_v_loss=0.25,
    )
    append_jsonl(metrics_path, record.__dict__)
    rows = read_jsonl(metrics_path)
    assert rows[0]["step"] == 10
    assert rows[0]["train_actor_loss"] == 1.5
