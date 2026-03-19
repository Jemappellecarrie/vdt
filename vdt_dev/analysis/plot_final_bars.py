from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from vdt_dev.analysis._plotting import coerce_float, default_style, load_csv_rows, maybe_prepare_aggregate, save_figure


def _load_rows(summary_csv: str | None, output_roots: List[str] | None, aggregate_dir: str) -> List[Dict[str, Any]]:
    if summary_csv:
        return load_csv_rows(summary_csv)
    summary = maybe_prepare_aggregate(output_roots, destination_dir=aggregate_dir)
    return load_csv_rows(summary["seed_aggregated_summary_csv"]) if summary else []


def plot_final_bars(
    *,
    summary_csv: str | None = None,
    output_roots: List[str] | None = None,
    aggregate_dir: str = "vdt_dev/outputs/aggregated",
    output_dir: str = "vdt_dev/paper_figures",
    metric_key: str = "best_eval_return_normalized_mean",
    output_stem: str = "final_bars",
) -> tuple[Path, Path]:
    import matplotlib.pyplot as plt

    rows = _load_rows(summary_csv, output_roots, aggregate_dir)
    default_style()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    labels = []
    means = []
    stds = []
    for row in rows:
        model_variant = row.get("model_variant", "")
        label_parts = [str(row.get("env_name", "?"))]
        if model_variant not in {None, ""}:
            label_parts.append(str(model_variant))
        label_parts.append(str(row.get("query_mode", "?")))
        label_parts.append(f'L{row.get("num_layers", "?")}')
        label = ":".join(label_parts)
        mean = coerce_float(row.get(metric_key))
        std = coerce_float(row.get(metric_key.replace("_mean", "_std")))
        if mean is None:
            continue
        labels.append(label)
        means.append(mean)
        stds.append(0.0 if std is None else std)
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel(metric_key)
    ax.set_title("Final Performance")
    ax.grid(axis="y", alpha=0.25)
    return save_figure(fig, output_dir, output_stem)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot Step 3 final-performance bars.")
    parser.add_argument("--summary-csv", type=str, default=None)
    parser.add_argument("--output-root", action="append", dest="output_roots", default=None)
    parser.add_argument("--aggregate-dir", type=str, default="vdt_dev/outputs/aggregated")
    parser.add_argument("--output-dir", type=str, default="vdt_dev/paper_figures")
    parser.add_argument("--metric-key", type=str, default="best_eval_return_normalized_mean")
    parser.add_argument("--output-stem", type=str, default="final_bars")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    plot_final_bars(
        summary_csv=args.summary_csv,
        output_roots=args.output_roots,
        aggregate_dir=args.aggregate_dir,
        output_dir=args.output_dir,
        metric_key=args.metric_key,
        output_stem=args.output_stem,
    )


if __name__ == "__main__":
    main()
