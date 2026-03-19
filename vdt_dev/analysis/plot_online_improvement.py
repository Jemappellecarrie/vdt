from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from vdt_dev.analysis._plotting import (
    coerce_float,
    default_style,
    load_csv_rows,
    maybe_prepare_aggregate,
    save_figure,
)


def _load_rows(summary_csv: str | None, output_roots: List[str] | None, aggregate_dir: str) -> List[Dict[str, Any]]:
    if summary_csv:
        return load_csv_rows(summary_csv)
    summary = maybe_prepare_aggregate(output_roots, destination_dir=aggregate_dir)
    return load_csv_rows(summary["seed_aggregated_summary_csv"]) if summary else []


def _row_label(row: Dict[str, Any]) -> str:
    return (
        f'{row.get("env_name", "?")}:'
        f'{row.get("model_variant", row.get("query_mode", "?"))}:'
        f'{row.get("query_mode", "?")}:'
        f'L{row.get("num_layers", "?")}'
    )


def plot_online_improvement(
    *,
    summary_csv: str | None = None,
    output_roots: List[str] | None = None,
    aggregate_dir: str = "vdt_dev/outputs/aggregated",
    output_dir: str = "vdt_dev/paper_figures",
) -> tuple[Path, Path]:
    import matplotlib.pyplot as plt

    rows = _load_rows(summary_csv, output_roots, aggregate_dir)
    default_style()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    labels: list[str] = []
    offline_means: list[float] = []
    online_means: list[float] = []

    for row in rows:
        offline_value = coerce_float(row.get("best_offline_eval_return_normalized_mean"))
        online_value = coerce_float(row.get("best_online_eval_return_normalized_mean"))
        if offline_value is None and online_value is None:
            continue
        labels.append(_row_label(row))
        offline_means.append(0.0 if offline_value is None else offline_value)
        online_means.append(0.0 if online_value is None else online_value)

    x = np.arange(len(labels))
    width = 0.36
    ax.bar(x - (width / 2.0), offline_means, width=width, label="Best Offline")
    ax.bar(x + (width / 2.0), online_means, width=width, label="Best Online")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Normalized Return")
    ax.set_title("Offline-to-Online Comparison")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    return save_figure(fig, output_dir, "online_improvement")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot offline-to-online before/after bars.")
    parser.add_argument("--summary-csv", type=str, default=None)
    parser.add_argument("--output-root", action="append", dest="output_roots", default=None)
    parser.add_argument("--aggregate-dir", type=str, default="vdt_dev/outputs/aggregated")
    parser.add_argument("--output-dir", type=str, default="vdt_dev/paper_figures")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    plot_online_improvement(
        summary_csv=args.summary_csv,
        output_roots=args.output_roots,
        aggregate_dir=args.aggregate_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
