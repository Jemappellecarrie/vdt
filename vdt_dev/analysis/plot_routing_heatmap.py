from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from vdt_dev.analysis._plotting import default_style, load_debug_records_from_run, save_figure


def _latest_source_usage(debug_records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not debug_records:
        return []
    latest = debug_records[-1]
    usage = latest.get("source_usage_by_layer")
    if isinstance(usage, str):
        return json.loads(usage)
    return usage or []


def plot_routing_heatmap(run_dir: str, *, output_dir: str) -> tuple[Path, Path]:
    import matplotlib.pyplot as plt

    debug_records = load_debug_records_from_run(run_dir)
    source_usage = _latest_source_usage(debug_records)
    default_style()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if not source_usage:
        ax.text(0.5, 0.5, "No routing summaries found", ha="center", va="center")
        ax.set_axis_off()
        return save_figure(fig, output_dir, f"{Path(run_dir).name}_routing_heatmap")

    max_sources = max(len(layer["weights_mean"]) for layer in source_usage)
    heatmap = np.full((len(source_usage), max_sources), np.nan, dtype=np.float32)
    y_labels: List[str] = []
    for layer_index, layer in enumerate(source_usage):
        weights = np.asarray(layer["weights_mean"], dtype=np.float32)
        heatmap[layer_index, : len(weights)] = weights
        y_labels.append(layer.get("layer_name", f"layer_{layer_index:02d}"))

    image = ax.imshow(heatmap, aspect="auto", interpolation="nearest")
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("Source index")
    ax.set_title("Routing Source Usage")
    fig.colorbar(image, ax=ax, shrink=0.9)
    return save_figure(fig, output_dir, f"{Path(run_dir).name}_routing_heatmap")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot routing heatmaps from debug summaries.")
    parser.add_argument("run_dir", type=str)
    parser.add_argument("--output-dir", type=str, default="vdt_dev/paper_figures")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    plot_routing_heatmap(args.run_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
