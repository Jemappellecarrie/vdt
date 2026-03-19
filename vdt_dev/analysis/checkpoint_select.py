from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from vdt_dev.analysis.collect import summarize_metrics_logs
from vdt_dev.analysis.manifest import load_run_manifest


def _coerce_run_dir(path_or_run_dir: Path | str) -> Path:
    input_path = Path(path_or_run_dir)
    if input_path.is_file():
        return input_path.parent.parent if input_path.parent.name == "checkpoints" else input_path.parent
    return input_path


def resolve_checkpoint_path(
    path_or_run_dir: Path | str,
    *,
    selector: str = "latest",
    explicit_checkpoint: Optional[str] = None,
    metric_name: str = "eval_return_normalized",
) -> Path:
    if explicit_checkpoint:
        checkpoint_path = Path(explicit_checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint `{checkpoint_path}` does not exist.")
        return checkpoint_path

    candidate_path = Path(path_or_run_dir)
    if candidate_path.is_file():
        return candidate_path

    run_dir = _coerce_run_dir(candidate_path)
    manifest = load_run_manifest(run_dir)
    checkpoints_dir = run_dir / "checkpoints"

    if selector == "latest":
        latest_path = checkpoints_dir / "latest.pt"
        if latest_path.exists():
            return latest_path
        if manifest and manifest.artifacts.latest_checkpoint_path:
            latest_from_manifest = Path(manifest.artifacts.latest_checkpoint_path)
            if latest_from_manifest.exists():
                return latest_from_manifest
    elif selector == "best":
        best_path = checkpoints_dir / "best.pt"
        if best_path.exists():
            return best_path
        if manifest and manifest.artifacts.best_checkpoint_path:
            best_from_manifest = Path(manifest.artifacts.best_checkpoint_path)
            if best_from_manifest.exists():
                return best_from_manifest
        summary = summarize_metrics_logs(run_dir)
        best_checkpoint = summary.get("best_checkpoint_path")
        if best_checkpoint:
            checkpoint_path = Path(best_checkpoint)
            if checkpoint_path.exists():
                return checkpoint_path
    else:
        explicit_path = checkpoints_dir / selector
        if explicit_path.exists():
            return explicit_path
        selector_path = Path(selector)
        if selector_path.exists():
            return selector_path

    raise FileNotFoundError(
        f"Could not resolve selector `{selector}` under run directory `{run_dir}` "
        f"using metric `{metric_name}`."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Resolve a run checkpoint path.")
    parser.add_argument("path", type=str, help="Run directory, manifest path, or checkpoint path.")
    parser.add_argument(
        "--selector",
        type=str,
        default="latest",
        help="Checkpoint selector: latest, best, checkpoint filename, or explicit path.",
    )
    parser.add_argument(
        "--metric-name",
        type=str,
        default="eval_return_normalized",
        help="Metric used when resolving the best checkpoint.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print(resolve_checkpoint_path(args.path, selector=args.selector, metric_name=args.metric_name))


if __name__ == "__main__":
    main()
