from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from vdt_dev.analysis._plotting import default_style, load_rtg_grid_records_from_run, save_figure


def plot_rtg_alignment(run_dirs: List[str], *, output_dir: str) -> tuple[Path, Path]:
    import matplotlib.pyplot as plt

    default_style()
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for run_dir in run_dirs:
        records = load_rtg_grid_records_from_run(run_dir)
        targets = np.asarray([float(record["rtg_target"]) for record in records], dtype=np.float32)
        achieved = np.asarray([float(record["rtg_achieved"]) for record in records], dtype=np.float32)
        if len(targets) == 0:
            continue
        unique_targets = sorted(set(targets.tolist()))
        mean_achieved = [
            float(np.mean(achieved[targets == target])) for target in unique_targets
        ]
        ax.plot(unique_targets, mean_achieved, marker="o", label=Path(run_dir).name)
    ax.set_xlabel("Target RTG")
    ax.set_ylabel("Achieved Return")
    ax.set_title("RTG Alignment")
    ax.grid(alpha=0.25)
    ax.legend()
    return save_figure(fig, output_dir, "rtg_alignment")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot RTG alignment curves.")
    parser.add_argument("run_dirs", nargs="+", type=str)
    parser.add_argument("--output-dir", type=str, default="vdt_dev/paper_figures")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    plot_rtg_alignment(args.run_dirs, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
