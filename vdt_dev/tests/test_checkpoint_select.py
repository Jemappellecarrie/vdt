from __future__ import annotations

from pathlib import Path

from vdt_dev.analysis.checkpoint_select import resolve_checkpoint_path
from vdt_dev.analysis.schema import RunManifest, append_jsonl, write_dataclass_json


def test_checkpoint_select_resolves_latest_and_best(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "metadata").mkdir(parents=True)
    (run_dir / "logs").mkdir(parents=True)
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "checkpoints/latest.pt").write_text("latest", encoding="utf-8")
    (run_dir / "checkpoints/best.pt").write_text("best", encoding="utf-8")
    write_dataclass_json(
        run_dir / "metadata/run_manifest.json",
        RunManifest(
            run_id="run",
            env_name="hopper-medium-v2",
            output_dir=str(run_dir),
        ),
    )
    append_jsonl(
        run_dir / "logs/eval_metrics.jsonl",
        {
            "step": 10,
            "eval_return_normalized": 50.0,
            "checkpoint_path": str(run_dir / "checkpoints/best.pt"),
        },
    )

    assert resolve_checkpoint_path(run_dir, selector="latest") == run_dir / "checkpoints/latest.pt"
    assert resolve_checkpoint_path(run_dir, selector="best") == run_dir / "checkpoints/best.pt"
