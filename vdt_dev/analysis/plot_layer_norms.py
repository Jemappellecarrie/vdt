from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from vdt_dev.analysis._plotting import default_style, load_debug_records_from_run, parse_layer_array, save_figure


def plot_layer_norms(run_dir: str, *, output_dir: str) -> tuple[Path, Path]:
    import matplotlib.pyplot as plt

    debug_records = load_debug_records_from_run(run_dir)
    latest = debug_records[-1] if debug_records else {}
    hidden_norms = parse_layer_array(latest.get("hidden_norm_by_layer"))
    output_norms = parse_layer_array(latest.get("output_norm_by_layer"))
    default_style()
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    if hidden_norms:
        ax.plot(np.arange(len(hidden_norms)), hidden_norms, marker="o", label="Hidden norm")
    if output_norms:
        ax.plot(np.arange(len(output_norms)), output_norms, marker="s", label="Output norm")
    ax.set_xlabel("Layer / route site index")
    ax.set_ylabel("Norm")
    ax.set_title("Layerwise Norms")
    ax.grid(alpha=0.25)
    ax.legend()
    return save_figure(fig, output_dir, f"{Path(run_dir).name}_layer_norms")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot layer norm summaries.")
    parser.add_argument("run_dir", type=str)
    parser.add_argument("--output-dir", type=str, default="vdt_dev/paper_figures")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    plot_layer_norms(args.run_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
