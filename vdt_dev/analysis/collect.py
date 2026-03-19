from __future__ import annotations

import argparse
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from vdt_dev.analysis.manifest import load_run_manifest, update_manifest
from vdt_dev.analysis.schema import (
    DebugSummaryRecord,
    append_jsonl,
    read_json,
    read_jsonl,
    write_dataclass_json,
    write_json,
)


LAYER_NAME_PATTERN = re.compile(r"layer_(?P<layer>\d+)\.(?P<site>[a-z_]+)")


def _mean_or_none(values: Sequence[float]) -> Optional[float]:
    finite_values = [value for value in values if value is not None and math.isfinite(value)]
    if not finite_values:
        return None
    return float(np.mean(finite_values))


def _layer_order(name: str) -> tuple[int, str]:
    match = LAYER_NAME_PATTERN.match(name)
    if not match:
        return (10**9, name)
    return (int(match.group("layer")), match.group("site"))


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (float, int, np.floating, np.integer)):
        value_float = float(value)
        if math.isfinite(value_float):
            return value_float
        return None
    return None


def _source_distance_for_names(source_names: Sequence[str], weights_mean: Sequence[float]) -> float:
    indices: List[float] = []
    for source_name in source_names:
        if source_name == "embedding":
            indices.append(0.0)
        elif source_name == "partial_block":
            indices.append(float(len(source_names) - 1))
        elif source_name.startswith("block_"):
            try:
                indices.append(float(int(source_name.split("_", 1)[1]) + 1))
            except ValueError:
                indices.append(float(len(indices)))
        else:
            indices.append(float(len(indices)))
    return float(np.dot(np.asarray(indices, dtype=np.float32), np.asarray(weights_mean, dtype=np.float32)))


def summarize_debug_payload(
    payload: Dict[str, Any],
    *,
    step: int,
    phase: str,
    split: str,
    artifact_path: Optional[str] = None,
) -> DebugSummaryRecord:
    if not payload:
        return DebugSummaryRecord(step=step, phase=phase, split=split, artifact_path=artifact_path)

    per_layer: Dict[str, Dict[str, Any]] = {}
    events = payload.get("events", [])

    if events:
        grouped_events: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for event in events:
            grouped_events[event["name"]].append(event)
        for name, name_events in grouped_events.items():
            per_layer[name] = {
                "routing_entropy_mean": _mean_or_none(
                    [_safe_float(event.get("entropy_mean")) for event in name_events]
                ),
                "static_query_norm_mean": _mean_or_none(
                    [_safe_float(event.get("static_query_norm_mean")) for event in name_events]
                ),
                "delta_query_norm_mean": _mean_or_none(
                    [_safe_float(event.get("dynamic_query_delta_norm_mean")) for event in name_events]
                ),
                "fused_query_norm_mean": _mean_or_none(
                    [_safe_float(event.get("fused_query_norm_mean")) for event in name_events]
                ),
                "hidden_norm_mean": _mean_or_none(
                    [_safe_float(event.get("hidden_norm_mean")) for event in name_events]
                ),
                "output_norm_mean": _mean_or_none(
                    [_safe_float(event.get("output_norm_mean")) for event in name_events]
                ),
                "source_names": name_events[-1].get("source_names", []),
                "source_usage": name_events[-1].get("weights_mean", []),
            }
    else:
        source_names_map = payload.get("source_names", {})
        for name, entropy_tensor in payload.get("routing_entropy", {}).items():
            weights = payload.get("routing_weights", {}).get(name)
            hidden_norm = payload.get("hidden_norm", {}).get(name)
            output_norm = payload.get("output_norm", {}).get(name)
            dynamic_norm = payload.get("dynamic_query_delta_norm", {}).get(name)
            static_norm = payload.get("static_query_norm", {}).get(name)
            fused_norm = payload.get("fused_query_norm", {}).get(name)
            per_layer[name] = {
                "routing_entropy_mean": _safe_float(torch.as_tensor(entropy_tensor).float().mean().item()),
                "static_query_norm_mean": _safe_float(torch.as_tensor(static_norm).float().mean().item())
                if static_norm is not None
                else None,
                "delta_query_norm_mean": _safe_float(torch.as_tensor(dynamic_norm).float().mean().item())
                if dynamic_norm is not None
                else None,
                "fused_query_norm_mean": _safe_float(torch.as_tensor(fused_norm).float().mean().item())
                if fused_norm is not None
                else None,
                "hidden_norm_mean": _safe_float(torch.as_tensor(hidden_norm).float().mean().item())
                if hidden_norm is not None
                else None,
                "output_norm_mean": _safe_float(torch.as_tensor(output_norm).float().mean().item())
                if output_norm is not None
                else None,
                "source_names": source_names_map.get(name, []),
                "source_usage": torch.as_tensor(weights).float().mean(dim=(0, 1)).tolist()
                if weights is not None
                else [],
            }

    ordered_names = sorted(per_layer, key=_layer_order)
    routing_entropy_by_layer: List[float] = []
    hidden_norm_by_layer: List[float] = []
    output_norm_by_layer: List[float] = []
    source_usage_by_layer: List[Dict[str, Any]] = []
    source_distance_by_layer: List[float] = []

    for name in ordered_names:
        layer_payload = per_layer[name]
        if layer_payload["routing_entropy_mean"] is not None:
            routing_entropy_by_layer.append(layer_payload["routing_entropy_mean"])
        if layer_payload["hidden_norm_mean"] is not None:
            hidden_norm_by_layer.append(layer_payload["hidden_norm_mean"])
        if layer_payload["output_norm_mean"] is not None:
            output_norm_by_layer.append(layer_payload["output_norm_mean"])
        source_names = list(layer_payload.get("source_names") or [])
        source_usage = list(layer_payload.get("source_usage") or [])
        if source_names and source_usage:
            source_usage_by_layer.append(
                {
                    "layer_name": name,
                    "source_names": source_names,
                    "weights_mean": source_usage,
                }
            )
            source_distance_by_layer.append(
                _source_distance_for_names(source_names, source_usage)
            )

    record = DebugSummaryRecord(
        step=step,
        phase=phase,
        split=split,
        routing_entropy_mean=_mean_or_none(routing_entropy_by_layer),
        routing_entropy_by_layer=routing_entropy_by_layer or None,
        static_query_norm_mean=_mean_or_none(
            [
                per_layer[name]["static_query_norm_mean"]
                for name in ordered_names
                if per_layer[name]["static_query_norm_mean"] is not None
            ]
        ),
        delta_query_norm_mean=_mean_or_none(
            [
                per_layer[name]["delta_query_norm_mean"]
                for name in ordered_names
                if per_layer[name]["delta_query_norm_mean"] is not None
            ]
        ),
        fused_query_norm_mean=_mean_or_none(
            [
                per_layer[name]["fused_query_norm_mean"]
                for name in ordered_names
                if per_layer[name]["fused_query_norm_mean"] is not None
            ]
        ),
        hidden_norm_by_layer=hidden_norm_by_layer or None,
        output_norm_by_layer=output_norm_by_layer or None,
        source_usage_by_layer=source_usage_by_layer or None,
        source_distance_mean=_mean_or_none(source_distance_by_layer),
        source_distance_by_layer=source_distance_by_layer or None,
        artifact_path=artifact_path,
    )
    return record


def _load_summary_if_present(run_dir: Path) -> Dict[str, Any]:
    summary_path = run_dir / "analysis/summary.json"
    return read_json(summary_path, default={}) or {}


def _best_eval_record_for_phase(
    eval_records: Sequence[Dict[str, Any]],
    *,
    phase: str,
) -> Optional[Dict[str, Any]]:
    phase_records = [
        record
        for record in eval_records
        if record.get("phase") == phase and _safe_float(record.get("eval_return_normalized")) is not None
    ]
    if not phase_records:
        return None
    return max(phase_records, key=lambda record: float(record["eval_return_normalized"]))


def _final_eval_record_for_phase(
    eval_records: Sequence[Dict[str, Any]],
    *,
    phase: str,
) -> Optional[Dict[str, Any]]:
    phase_records = [record for record in eval_records if record.get("phase") == phase]
    if not phase_records:
        return None
    return phase_records[-1]


def _canonicalize_metric(record: Dict[str, Any]) -> Dict[str, Any]:
    canonical = dict(record)
    if "Actor Loss" in record and "train_actor_loss" not in canonical:
        canonical["train_actor_loss"] = record["Actor Loss"]
    if "Value Loss" in record and "train_v_loss" not in canonical:
        canonical["train_v_loss"] = record["Value Loss"]
    if "Q Loss" in record and "train_q_loss" not in canonical:
        canonical["train_q_loss"] = record["Q Loss"]
    if "target_normalized_score" in record and "eval_return_normalized" not in canonical:
        canonical["eval_return_normalized"] = record["target_normalized_score"]
    if "target_length_mean" in record and "eval_length" not in canonical:
        canonical["eval_length"] = record["target_length_mean"]
    if "eval_return_raw_mean" in record and "eval_return_raw" not in canonical:
        canonical["eval_return_raw"] = record["eval_return_raw_mean"]
    return canonical


def summarize_metrics_logs(run_dir: Path | str) -> Dict[str, Any]:
    root = Path(run_dir)
    train_records = [_canonicalize_metric(record) for record in read_jsonl(root / "logs/train_metrics.jsonl")]
    eval_records = [_canonicalize_metric(record) for record in read_jsonl(root / "logs/eval_metrics.jsonl")]
    debug_records = [_canonicalize_metric(record) for record in read_jsonl(root / "logs/debug_metrics.jsonl")]
    rtg_records = [_canonicalize_metric(record) for record in read_jsonl(root / "logs/rtg_grid_metrics.jsonl")]
    compute_summary = read_json(root / "analysis/compute_summary.json", default={}) or {}

    summary: Dict[str, Any] = {
        "run_dir": str(root),
        "num_train_records": len(train_records),
        "num_eval_records": len(eval_records),
        "num_debug_records": len(debug_records),
        "num_rtg_grid_records": len(rtg_records),
    }

    if train_records:
        final_train = train_records[-1]
        summary["final_train_step"] = final_train.get("step")
        summary["final_train_actor_loss"] = _safe_float(final_train.get("train_actor_loss"))
        summary["final_train_q_loss"] = _safe_float(final_train.get("train_q_loss"))
        summary["final_train_v_loss"] = _safe_float(final_train.get("train_v_loss"))

    if eval_records:
        final_eval = eval_records[-1]
        summary["final_eval_phase"] = final_eval.get("phase")
        summary["final_eval_step"] = final_eval.get("step")
        summary["final_eval_return_raw"] = _safe_float(final_eval.get("eval_return_raw"))
        summary["final_eval_return_normalized"] = _safe_float(final_eval.get("eval_return_normalized"))
        summary["final_eval_length"] = _safe_float(final_eval.get("eval_length"))
        best_eval = max(
            (
                record
                for record in eval_records
                if _safe_float(record.get("eval_return_normalized")) is not None
            ),
            key=lambda record: float(record["eval_return_normalized"]),
            default=None,
        )
        if best_eval is not None:
            summary["best_eval_return_normalized"] = float(best_eval["eval_return_normalized"])
            summary["best_eval_step"] = best_eval.get("step")
            summary["best_checkpoint_path"] = best_eval.get("checkpoint_path")

        final_offline_eval = _final_eval_record_for_phase(eval_records, phase="offline")
        final_online_eval = _final_eval_record_for_phase(eval_records, phase="online")
        best_offline_eval = _best_eval_record_for_phase(eval_records, phase="offline")
        best_online_eval = _best_eval_record_for_phase(eval_records, phase="online")
        if final_offline_eval is not None:
            summary["final_offline_eval_return_normalized"] = _safe_float(
                final_offline_eval.get("eval_return_normalized")
            )
        if final_online_eval is not None:
            summary["final_online_eval_return_normalized"] = _safe_float(
                final_online_eval.get("eval_return_normalized")
            )
        if best_offline_eval is not None:
            summary["best_offline_eval_return_normalized"] = _safe_float(
                best_offline_eval.get("eval_return_normalized")
            )
        if best_online_eval is not None:
            summary["best_online_eval_return_normalized"] = _safe_float(
                best_online_eval.get("eval_return_normalized")
            )
        if (
            _safe_float(summary.get("best_offline_eval_return_normalized")) is not None
            and _safe_float(summary.get("best_online_eval_return_normalized")) is not None
        ):
            summary["online_minus_offline_best_return"] = float(
                summary["best_online_eval_return_normalized"]
                - summary["best_offline_eval_return_normalized"]
            )

    if debug_records:
        final_debug = debug_records[-1]
        summary["routing_entropy_mean"] = _safe_float(final_debug.get("routing_entropy_mean"))
        summary["static_query_norm_mean"] = _safe_float(final_debug.get("static_query_norm_mean"))
        summary["delta_query_norm_mean"] = _safe_float(final_debug.get("delta_query_norm_mean"))
        summary["fused_query_norm_mean"] = _safe_float(final_debug.get("fused_query_norm_mean"))
        summary["source_distance_mean"] = _safe_float(final_debug.get("source_distance_mean"))
        summary["routing_entropy_by_layer"] = final_debug.get("routing_entropy_by_layer")
        summary["hidden_norm_by_layer"] = final_debug.get("hidden_norm_by_layer")
        summary["output_norm_by_layer"] = final_debug.get("output_norm_by_layer")
        summary["source_usage_by_layer"] = final_debug.get("source_usage_by_layer")

    if rtg_records:
        summary["rtg_alignment_mean"] = _mean_or_none(
            [
                _safe_float(record.get("rtg_alignment_error"))
                for record in rtg_records
                if _safe_float(record.get("rtg_alignment_error")) is not None
            ]
        )
        summary["rtg_grid_targets"] = sorted(
            {
                float(record["rtg_target"])
                for record in rtg_records
                if _safe_float(record.get("rtg_target")) is not None
            }
        )

    if compute_summary:
        summary.update(compute_summary)

    return summary


def collect_run_summary(run_dir: Path | str, *, write_summary: bool = True) -> Dict[str, Any]:
    root = Path(run_dir)
    manifest = load_run_manifest(root)
    summary = summarize_metrics_logs(root)
    summary["manifest"] = manifest.__dict__ if manifest is not None else {}
    if write_summary:
        write_json(root / "analysis/summary.json", summary)
        if manifest is not None:
            update_manifest(
                root,
                artifacts={"aggregated_csv_path": str(root / "analysis/summary.json")},
                notes={"summary_collected": True},
            )
    return summary


def append_debug_summary(
    output_path: Path | str,
    payload: Dict[str, Any],
    *,
    step: int,
    phase: str,
    split: str,
    artifact_path: Optional[str] = None,
) -> DebugSummaryRecord:
    record = summarize_debug_payload(
        payload,
        step=step,
        phase=phase,
        split=split,
        artifact_path=artifact_path,
    )
    append_jsonl(output_path, record.__dict__)
    return record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect a Step 3 run summary.")
    parser.add_argument("run_dir", type=str, help="Run directory to summarize.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = collect_run_summary(args.run_dir, write_summary=True)
    write_json(Path(args.run_dir) / "analysis/summary.json", summary)
    print(Path(args.run_dir) / "analysis/summary.json")


if __name__ == "__main__":
    main()
