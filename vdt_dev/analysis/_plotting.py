from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from vdt_dev.analysis.aggregate import aggregate_output_roots
from vdt_dev.analysis.collect import collect_run_summary
from vdt_dev.analysis.manifest import iter_run_manifests
from vdt_dev.analysis.schema import read_json, read_jsonl


def load_csv_rows(path: Path | str) -> List[Dict[str, Any]]:
    input_path = Path(path)
    if not input_path.exists():
        return []
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def coerce_float(value: Any) -> float | None:
    if value in {None, "", "nan", "None"}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def maybe_prepare_aggregate(
    output_roots: Sequence[str] | None,
    *,
    destination_dir: Path | str,
) -> Dict[str, Any]:
    if not output_roots:
        return {}
    return aggregate_output_roots(output_roots, destination_dir=destination_dir)


def ensure_figure_dir(output_dir: Path | str) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_figure(fig: plt.Figure, output_dir: Path | str, stem: str) -> tuple[Path, Path]:
    figure_dir = ensure_figure_dir(output_dir)
    png_path = figure_dir / f"{stem}.png"
    pdf_path = figure_dir / f"{stem}.pdf"
    fig.tight_layout()
    fig.savefig(png_path, dpi=200)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, pdf_path


def default_style() -> None:
    plt.style.use("default")
    matplotlib.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.titlesize": 13,
        }
    )


def group_rows(rows: Sequence[Dict[str, Any]], key_field: str) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        key = str(row.get(key_field, "unknown"))
        grouped.setdefault(key, []).append(row)
    return grouped


def load_run_metric_series(run_dir: Path | str, metric_file_name: str) -> List[Dict[str, Any]]:
    return read_jsonl(Path(run_dir) / "logs" / metric_file_name)


def load_run_summaries(output_roots: Iterable[Path | str]) -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []
    for run_dir, manifest in iter_run_manifests(output_roots):
        summary = collect_run_summary(run_dir, write_summary=True)
        summary.update(
            {
                "run_dir": str(run_dir),
                "run_id": manifest.run_id,
                "env_name": manifest.env_name,
                "model_variant": manifest.model_variant,
                "query_mode": manifest.query_mode,
                "num_layers": manifest.num_layers,
                "seed": manifest.seed,
                "experiment_family": manifest.experiment_family,
            }
        )
        summaries.append(summary)
    return summaries


def load_debug_records_from_run(run_dir: Path | str) -> List[Dict[str, Any]]:
    return read_jsonl(Path(run_dir) / "logs/debug_metrics.jsonl")


def load_rtg_grid_records_from_run(run_dir: Path | str) -> List[Dict[str, Any]]:
    return read_jsonl(Path(run_dir) / "logs/rtg_grid_metrics.jsonl")


def parse_layer_array(value: Any) -> List[float]:
    if value is None:
        return []
    if isinstance(value, list):
        return [float(item) for item in value]
    if isinstance(value, str):
        payload = read_json(value) if Path(value).exists() else None
        if isinstance(payload, list):
            return [float(item) for item in payload]
        stripped = value.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            import json

            return [float(item) for item in json.loads(stripped)]
    return []
