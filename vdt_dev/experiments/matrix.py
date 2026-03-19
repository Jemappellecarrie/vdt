from __future__ import annotations

import itertools
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from vdt_dev.analysis.manifest import build_run_id, manifest_path_for_run_dir, write_run_manifest
from vdt_dev.analysis.schema import ArtifactPaths, RunManifest


@dataclass
class RunSpec:
    name: str
    config_path: str
    output_dir: str
    overrides: Dict[str, Any] = field(default_factory=dict)
    gpu_id: int | None = None


def expand_matrix(
    axes: Dict[str, Sequence[Any]],
    *,
    shared_overrides: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    shared = dict(shared_overrides or {})
    axis_names = sorted(axes)
    values = [axes[name] for name in axis_names]
    expanded: List[Dict[str, Any]] = []
    for combo in itertools.product(*values):
        record = dict(shared)
        for name, value in zip(axis_names, combo):
            record[name] = value
        expanded.append(record)
    return expanded


def _cli_key(key: str) -> str:
    return f"--{key.replace('_', '-')}"


def run_spec_to_command(run_spec: RunSpec) -> str:
    command_parts = [
        "python",
        "-m",
        "vdt_dev.runner",
        "--config",
        run_spec.config_path,
        "--output-dir",
        run_spec.output_dir,
    ]
    for key, value in run_spec.overrides.items():
        if value is None:
            continue
        command_parts.extend([_cli_key(key), str(value)])
    if run_spec.gpu_id is not None:
        return f"CUDA_VISIBLE_DEVICES={run_spec.gpu_id} " + " ".join(
            shlex.quote(part) for part in command_parts
        )
    return " ".join(shlex.quote(part) for part in command_parts)


def planned_manifest_for_run_spec(run_spec: RunSpec) -> RunManifest:
    exp_name = str(run_spec.overrides.get("exp_name", run_spec.name))
    env_name = str(run_spec.overrides.get("env_name", ""))
    model_variant = str(run_spec.overrides.get("model_variant", ""))
    query_mode = str(run_spec.overrides.get("attnres_query_mode", "static"))
    seed = int(run_spec.overrides.get("seed", 0))
    num_layers = int(run_spec.overrides.get("n_layer", 0))
    return RunManifest(
        run_id=build_run_id(
            exp_name=exp_name,
            model_variant=model_variant or None,
            env_name=env_name,
            query_mode=query_mode,
            seed=seed,
            num_layers=num_layers,
            output_dir=run_spec.output_dir,
        ),
        exp_name=exp_name,
        experiment_family=str(run_spec.overrides.get("experiment_family", "")),
        preset_name=str(run_spec.overrides.get("preset_name", "")),
        env_name=env_name,
        seed=seed,
        model_variant=model_variant,
        query_mode=query_mode,
        attnres_variant=str(run_spec.overrides.get("attnres_variant", "block")),
        num_layers=num_layers,
        hidden_size=int(run_spec.overrides.get("embed_dim", run_spec.overrides.get("hidden_dim", 0))),
        num_heads=int(run_spec.overrides.get("n_head", 0)),
        context_len=int(run_spec.overrides.get("K", 0)),
        use_pre_attn=bool(run_spec.overrides.get("attnres_apply_pre_attn", True)),
        use_pre_mlp=bool(run_spec.overrides.get("attnres_apply_pre_mlp", True)),
        num_blocks=int(run_spec.overrides.get("attnres_num_blocks", 0)),
        train_steps=int(run_spec.overrides.get("n_steps", 0)),
        eval_interval=int(run_spec.overrides.get("eval_interval", run_spec.overrides.get("eval_period", 0))),
        config_path=run_spec.config_path,
        output_dir=run_spec.output_dir,
        status="planned",
        command=run_spec_to_command(run_spec),
        artifacts=ArtifactPaths(),
        notes={"planned_only": True},
    )


def write_planned_manifest(run_spec: RunSpec) -> Path:
    manifest = planned_manifest_for_run_spec(run_spec)
    run_dir = Path(run_spec.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_path_for_run_dir(run_dir)
    write_run_manifest(manifest_path, manifest)
    return manifest_path
