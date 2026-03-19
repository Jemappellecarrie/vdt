from __future__ import annotations

import argparse
from pathlib import Path

from vdt_dev.analysis._plotting import coerce_float, default_style, load_debug_records_from_run, save_figure


def plot_query_norms(run_dir: str, *, output_dir: str) -> tuple[Path, Path]:
    import matplotlib.pyplot as plt

    debug_records = load_debug_records_from_run(run_dir)
    latest = debug_records[-1] if debug_records else {}
    labels = ["static", "delta", "fused"]
    values = [
        coerce_float(latest.get("static_query_norm_mean")) or 0.0,
        coerce_float(latest.get("delta_query_norm_mean")) or 0.0,
        coerce_float(latest.get("fused_query_norm_mean")) or 0.0,
    ]
    default_style()
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.bar(labels, values)
    ax.set_ylabel("Norm")
    ax.set_title("Query Norm Summary")
    ax.grid(axis="y", alpha=0.25)
    return save_figure(fig, output_dir, f"{Path(run_dir).name}_query_norms")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot query norm summaries.")
    parser.add_argument("run_dir", type=str)
    parser.add_argument("--output-dir", type=str, default="vdt_dev/paper_figures")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    plot_query_norms(args.run_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
