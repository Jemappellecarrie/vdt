from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from vdt_dev.analysis._plotting import default_style, load_debug_records_from_run, parse_layer_array, save_figure


def plot_depth_source_usage(run_dir: str, *, output_dir: str) -> tuple[Path, Path]:
    import matplotlib.pyplot as plt

    debug_records = load_debug_records_from_run(run_dir)
    latest = debug_records[-1] if debug_records else {}
    source_distance = parse_layer_array(latest.get("source_distance_by_layer"))
    default_style()
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    if source_distance:
        ax.plot(np.arange(len(source_distance)), source_distance, marker="o")
    ax.set_xlabel("Layer / route site index")
    ax.set_ylabel("Effective depth usage")
    ax.set_title("Depth Source Usage")
    ax.grid(alpha=0.25)
    return save_figure(fig, output_dir, f"{Path(run_dir).name}_depth_source_usage")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot depth source usage.")
    parser.add_argument("run_dir", type=str)
    parser.add_argument("--output-dir", type=str, default="vdt_dev/paper_figures")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    plot_depth_source_usage(args.run_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
