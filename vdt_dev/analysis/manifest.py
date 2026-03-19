from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional

import yaml

from vdt_dev.analysis.schema import (
    ArtifactPaths,
    RunManifest,
    dataclass_from_dict,
    read_json,
    write_dataclass_json,
)
from vdt_dev.model_variants import resolve_model_variant


MANIFEST_RELATIVE_PATH = Path("metadata/run_manifest.json")
CONFIG_SNAPSHOT_RELATIVE_PATH = Path("metadata/config_snapshot.yaml")


def build_run_id(
    *,
    exp_name: str,
    model_variant: str | None = None,
    env_name: str,
    query_mode: str,
    seed: int,
    num_layers: int,
    output_dir: str,
) -> str:
    if exp_name:
        base = exp_name
    else:
        base = Path(output_dir).name
    if model_variant:
        return (
            f"{base}__{model_variant}__{env_name}__{query_mode}"
            f"__L{num_layers}__seed{seed}"
        )
    return f"{base}__{env_name}__{query_mode}__L{num_layers}__seed{seed}"


def manifest_path_for_run_dir(run_dir: Path | str) -> Path:
    return Path(run_dir) / MANIFEST_RELATIVE_PATH


def config_snapshot_path_for_run_dir(run_dir: Path | str) -> Path:
    return Path(run_dir) / CONFIG_SNAPSHOT_RELATIVE_PATH


def write_run_manifest(path: Path | str, manifest: RunManifest) -> Path:
    return write_dataclass_json(path, manifest)


def _load_config_snapshot(run_dir: Path) -> Dict[str, Any]:
    snapshot_path = config_snapshot_path_for_run_dir(run_dir)
    if not snapshot_path.exists():
        return {}
    with snapshot_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _manifest_from_snapshot(run_dir: Path) -> Optional[RunManifest]:
    config = _load_config_snapshot(run_dir)
    if not config:
        return None
    snapshot_path = config_snapshot_path_for_run_dir(run_dir)
    query_mode = config.get("attnres_query_mode", "static")
    model_variant = resolve_model_variant(
        explicit=config.get("model_variant"),
        use_attnres=bool(config.get("use_attnres", False)),
        query_mode=query_mode,
    )
    exp_name = config.get("exp_name", "") or Path(run_dir).name
    return RunManifest(
        run_id=build_run_id(
            exp_name=exp_name,
            model_variant=model_variant,
            env_name=config.get("env_name", ""),
            query_mode=query_mode,
            seed=int(config.get("seed", 0)),
            num_layers=int(config.get("n_layer", 0)),
            output_dir=str(run_dir),
        ),
        exp_name=exp_name,
        experiment_family=config.get("experiment_family", ""),
        preset_name=config.get("preset_name", ""),
        env_name=config.get("env_name", ""),
        seed=int(config.get("seed", 0)),
        model_variant=model_variant,
        query_mode=query_mode,
        attnres_variant=config.get("attnres_variant", "block"),
        num_layers=int(config.get("n_layer", 0)),
        hidden_size=int(config.get("embed_dim", config.get("hidden_dim", 0))),
        num_heads=int(config.get("n_head", 0)),
        context_len=int(config.get("K", 0)),
        use_pre_attn=bool(config.get("attnres_apply_pre_attn", True)),
        use_pre_mlp=bool(config.get("attnres_apply_pre_mlp", True)),
        num_blocks=int(config.get("attnres_num_blocks", 0)),
        train_steps=int(config.get("n_steps", 0)),
        eval_interval=int(config.get("eval_interval", config.get("eval_period", 0))),
        git_hash="",
        config_path=str(snapshot_path),
        output_dir=str(run_dir),
        job=config.get("job", "train"),
        status="legacy",
        command="",
        artifacts=ArtifactPaths(
            latest_checkpoint_path=str(run_dir / "checkpoints/latest.pt")
            if (run_dir / "checkpoints/latest.pt").exists()
            else None,
            best_checkpoint_path=str(run_dir / "checkpoints/best.pt")
            if (run_dir / "checkpoints/best.pt").exists()
            else None,
            metrics_jsonl_path=str(run_dir / "logs/train_metrics.jsonl")
            if (run_dir / "logs/train_metrics.jsonl").exists()
            else None,
            eval_metrics_jsonl_path=str(run_dir / "logs/eval_metrics.jsonl")
            if (run_dir / "logs/eval_metrics.jsonl").exists()
            else None,
            debug_metrics_jsonl_path=str(run_dir / "logs/debug_metrics.jsonl")
            if (run_dir / "logs/debug_metrics.jsonl").exists()
            else None,
            compute_summary_path=str(run_dir / "analysis/compute_summary.json")
            if (run_dir / "analysis/compute_summary.json").exists()
            else None,
        ),
        notes={"legacy_snapshot_only": True},
    )


def load_run_manifest(path_or_run_dir: Path | str) -> Optional[RunManifest]:
    path = Path(path_or_run_dir)
    if path.is_dir():
        manifest_path = manifest_path_for_run_dir(path)
        if manifest_path.exists():
            payload = read_json(manifest_path)
            return dataclass_from_dict(RunManifest, payload)
        return _manifest_from_snapshot(path)
    if path.exists():
        payload = read_json(path)
        return dataclass_from_dict(RunManifest, payload)
    return None


def discover_run_dirs(output_roots: Iterable[Path | str]) -> list[Path]:
    discovered: set[Path] = set()
    for root in output_roots:
        root_path = Path(root)
        if not root_path.exists():
            continue
        if manifest_path_for_run_dir(root_path).exists() or config_snapshot_path_for_run_dir(root_path).exists():
            discovered.add(root_path)
            continue
        for manifest_path in root_path.rglob(str(MANIFEST_RELATIVE_PATH)):
            discovered.add(manifest_path.parent.parent)
        for snapshot_path in root_path.rglob(str(CONFIG_SNAPSHOT_RELATIVE_PATH)):
            discovered.add(snapshot_path.parent.parent)
    return sorted(discovered)


def iter_run_manifests(output_roots: Iterable[Path | str]) -> Iterator[tuple[Path, RunManifest]]:
    for run_dir in discover_run_dirs(output_roots):
        manifest = load_run_manifest(run_dir)
        if manifest is not None:
            yield run_dir, manifest


def update_manifest(
    run_dir: Path | str,
    *,
    status: Optional[str] = None,
    finished_at: Optional[str] = None,
    resume_count: Optional[int] = None,
    artifacts: Optional[Dict[str, Any]] = None,
    notes: Optional[Dict[str, Any]] = None,
) -> RunManifest:
    run_path = Path(run_dir)
    manifest = load_run_manifest(run_path)
    if manifest is None:
        raise FileNotFoundError(f"No manifest or config snapshot found under {run_path}.")
    if status is not None:
        manifest.status = status
    if finished_at is not None:
        manifest.finished_at = finished_at
    if resume_count is not None:
        manifest.resume_count = resume_count
    if artifacts:
        artifact_state = asdict(manifest.artifacts)
        artifact_state.update(artifacts)
        manifest.artifacts = ArtifactPaths(**artifact_state)
    if notes:
        manifest.notes.update(notes)
    write_run_manifest(manifest_path_for_run_dir(run_path), manifest)
    return manifest
