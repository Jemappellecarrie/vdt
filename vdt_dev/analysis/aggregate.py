from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

from vdt_dev.analysis.collect import collect_run_summary
from vdt_dev.analysis.manifest import iter_run_manifests
from vdt_dev.analysis.schema import write_json


DEFAULT_GROUP_FIELDS = ["env_name", "model_variant", "query_mode", "num_layers", "experiment_family"]


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (float, int, np.floating, np.integer)):
        value_float = float(value)
        if math.isfinite(value_float):
            return value_float
    return None


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _group_key(row: Dict[str, Any], group_fields: Sequence[str]) -> tuple[Any, ...]:
    return tuple(row.get(field) for field in group_fields)


def _stats_for_rows(rows: Sequence[Dict[str, Any]], metric_fields: Sequence[str]) -> Dict[str, Any]:
    stats: Dict[str, Any] = {"count": len(rows)}
    for metric_field in metric_fields:
        values = [
            _safe_float(row.get(metric_field))
            for row in rows
            if _safe_float(row.get(metric_field)) is not None
        ]
        if not values:
            continue
        values_np = np.asarray(values, dtype=np.float32)
        stats[f"{metric_field}_mean"] = float(values_np.mean())
        stats[f"{metric_field}_std"] = float(values_np.std(ddof=0))
        stats[f"{metric_field}_min"] = float(values_np.min())
        stats[f"{metric_field}_max"] = float(values_np.max())
    return stats


def aggregate_output_roots(
    output_roots: Iterable[Path | str],
    *,
    destination_dir: Path | str,
    group_fields: Sequence[str] = DEFAULT_GROUP_FIELDS,
) -> Dict[str, Any]:
    destination = Path(destination_dir)
    destination.mkdir(parents=True, exist_ok=True)

    run_rows: List[Dict[str, Any]] = []
    for run_dir, manifest in iter_run_manifests(output_roots):
        summary = collect_run_summary(run_dir, write_summary=True)
        row: Dict[str, Any] = {
            "run_dir": str(run_dir),
            "run_id": manifest.run_id,
            "exp_name": manifest.exp_name,
            "experiment_family": manifest.experiment_family,
            "preset_name": manifest.preset_name,
            "env_name": manifest.env_name,
            "seed": manifest.seed,
            "model_variant": manifest.model_variant,
            "query_mode": manifest.query_mode,
            "attnres_variant": manifest.attnres_variant,
            "num_layers": manifest.num_layers,
            "hidden_size": manifest.hidden_size,
            "num_heads": manifest.num_heads,
            "context_len": manifest.context_len,
            "use_pre_attn": manifest.use_pre_attn,
            "use_pre_mlp": manifest.use_pre_mlp,
            "num_blocks": manifest.num_blocks,
            "train_steps": manifest.train_steps,
            "eval_interval": manifest.eval_interval,
            "git_hash": manifest.git_hash,
            "config_path": manifest.config_path,
            "output_dir": manifest.output_dir,
            "status": manifest.status,
            "budget_mode": manifest.notes.get("budget_mode", ""),
            "budget_reference": manifest.notes.get("budget_reference", ""),
            "budget_target_params": manifest.notes.get("budget_target_params"),
            "budget_tolerance_pct": manifest.notes.get("budget_tolerance_pct"),
            "budget_actual_gap_pct": manifest.notes.get("budget_actual_gap_pct"),
        }
        row.update(summary)
        run_rows.append(row)

    metric_fields = [
        "final_eval_return_normalized",
        "best_eval_return_normalized",
        "best_offline_eval_return_normalized",
        "best_online_eval_return_normalized",
        "online_minus_offline_best_return",
        "final_eval_return_raw",
        "rtg_alignment_mean",
        "routing_entropy_mean",
        "static_query_norm_mean",
        "delta_query_norm_mean",
        "fused_query_norm_mean",
        "source_distance_mean",
        "policy_param_count",
        "total_param_count",
        "budget_target_params",
        "budget_actual_gap_pct",
        "approx_policy_forward_flops",
        "approx_total_train_step_flops",
        "mean_train_step_time_sec",
        "mean_steps_per_sec",
        "mean_policy_tokens_per_sec",
        "peak_gpu_memory_mb",
        "total_wall_clock_sec",
        "offline_wall_clock_sec",
        "online_wall_clock_sec",
    ]

    grouped_rows: List[Dict[str, Any]] = []
    best_per_group_rows: List[Dict[str, Any]] = []
    rows_by_group: Dict[tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for row in run_rows:
        rows_by_group[_group_key(row, group_fields)].append(row)

    for group_key, rows in sorted(rows_by_group.items(), key=lambda item: item[0]):
        group_row = {field: value for field, value in zip(group_fields, group_key)}
        group_row.update(_stats_for_rows(rows, metric_fields))
        grouped_rows.append(group_row)

        sortable_rows = [
            row
            for row in rows
            if _safe_float(row.get("best_eval_return_normalized")) is not None
        ]
        if sortable_rows:
            best_row = max(
                sortable_rows,
                key=lambda row: float(row["best_eval_return_normalized"]),
            )
            best_per_group_rows.append(
                {
                    **group_row,
                    "best_run_id": best_row.get("run_id"),
                    "best_run_dir": best_row.get("run_dir"),
                    "best_eval_return_normalized": best_row.get("best_eval_return_normalized"),
                    "best_checkpoint_path": best_row.get("best_checkpoint_path"),
                }
            )

    run_summary_path = destination / "run_summary.csv"
    seed_summary_path = destination / "seed_aggregated_summary.csv"
    best_summary_path = destination / "best_per_group_summary.csv"
    json_summary_path = destination / "summary.json"

    _write_csv(run_summary_path, run_rows)
    _write_csv(seed_summary_path, grouped_rows)
    _write_csv(best_summary_path, best_per_group_rows)
    summary_payload = {
        "output_roots": [str(Path(root)) for root in output_roots],
        "run_summary_csv": str(run_summary_path),
        "seed_aggregated_summary_csv": str(seed_summary_path),
        "best_per_group_summary_csv": str(best_summary_path),
        "run_count": len(run_rows),
        "group_count": len(grouped_rows),
        "group_fields": list(group_fields),
        "runs": run_rows,
        "groups": grouped_rows,
        "best_per_group": best_per_group_rows,
    }
    write_json(json_summary_path, summary_payload)
    return summary_payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate Step 3 outputs.")
    parser.add_argument(
        "output_roots",
        nargs="+",
        type=str,
        help="One or more run roots or output directories to scan.",
    )
    parser.add_argument(
        "--destination-dir",
        type=str,
        default="vdt_dev/outputs/aggregated",
        help="Directory where CSV/JSON summaries will be written.",
    )
    parser.add_argument(
        "--group-by",
        type=str,
        default=",".join(DEFAULT_GROUP_FIELDS),
        help="Comma-separated list of grouping fields.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    group_fields = [field.strip() for field in args.group_by.split(",") if field.strip()]
    summary = aggregate_output_roots(
        args.output_roots,
        destination_dir=args.destination_dir,
        group_fields=group_fields or DEFAULT_GROUP_FIELDS,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
