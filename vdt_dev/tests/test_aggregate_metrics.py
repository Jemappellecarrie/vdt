from __future__ import annotations

from pathlib import Path

from vdt_dev.analysis.aggregate import aggregate_output_roots
from vdt_dev.analysis.schema import RunManifest, append_jsonl, write_dataclass_json


def _make_fake_run(root: Path, run_name: str, seed: int, best_value: float) -> Path:
    run_dir = root / run_name
    (run_dir / "metadata").mkdir(parents=True)
    (run_dir / "logs").mkdir(parents=True)
    (run_dir / "analysis").mkdir(parents=True)
    write_dataclass_json(
        run_dir / "metadata/run_manifest.json",
        RunManifest(
            run_id=run_name,
            exp_name="exp",
            experiment_family="query_mode_sweep",
            env_name="hopper-medium-v2",
            seed=seed,
            query_mode="state_rtg_value",
            num_layers=6,
            output_dir=str(run_dir),
            status="completed",
        ),
    )
    append_jsonl(
        run_dir / "logs/eval_metrics.jsonl",
        {
            "step": 100,
            "phase": "offline",
            "split": "eval",
            "eval_return_normalized": best_value,
            "eval_return_raw": best_value * 10.0,
            "eval_length": 1000.0,
            "checkpoint_path": str(run_dir / "checkpoints/best.pt"),
        },
    )
    append_jsonl(
        run_dir / "logs/debug_metrics.jsonl",
        {
            "step": 100,
            "phase": "offline",
            "split": "eval",
            "routing_entropy_mean": 0.5 + seed,
        },
    )
    return run_dir


def test_aggregate_metrics_computes_mean_std_and_count(tmp_path: Path) -> None:
    _make_fake_run(tmp_path, "run_seed0", seed=0, best_value=80.0)
    _make_fake_run(tmp_path, "run_seed1", seed=1, best_value=100.0)

    summary = aggregate_output_roots([tmp_path], destination_dir=tmp_path / "agg")
    assert summary["run_count"] == 2
    group = summary["groups"][0]
    assert group["count"] == 2
    assert abs(group["best_eval_return_normalized_mean"] - 90.0) < 1e-6
    assert abs(group["best_eval_return_normalized_std"] - 10.0) < 1e-6
