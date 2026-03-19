from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from vdt_dev.analysis._plotting import (
    coerce_float,
    default_style,
    load_run_metric_series,
    save_figure,
)


def _series_label(run_dir: str) -> str:
    return Path(run_dir).name


def plot_learning_curves(
    run_dirs: List[str],
    *,
    output_dir: str,
    metric_key: str = "eval_return_normalized",
    train_metric_key: str = "train_actor_loss",
) -> tuple[Path, Path]:
    import matplotlib.pyplot as plt

    default_style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for run_dir in run_dirs:
        eval_records = load_run_metric_series(run_dir, "eval_metrics.jsonl")
        train_records = load_run_metric_series(run_dir, "train_metrics.jsonl")
        eval_steps = [coerce_float(record.get("step")) for record in eval_records]
        eval_values = [coerce_float(record.get(metric_key)) for record in eval_records]
        train_steps = [coerce_float(record.get("step")) for record in train_records]
        train_values = [coerce_float(record.get(train_metric_key)) for record in train_records]

        axes[0].plot(
            [step for step, value in zip(eval_steps, eval_values) if step is not None and value is not None],
            [value for value in eval_values if value is not None],
            label=_series_label(run_dir),
        )
        axes[1].plot(
            [step for step, value in zip(train_steps, train_values) if step is not None and value is not None],
            [value for value in train_values if value is not None],
            label=_series_label(run_dir),
        )

    axes[0].set_title("Eval Learning Curve")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel(metric_key)
    axes[1].set_title("Train Learning Curve")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel(train_metric_key)
    for axis in axes:
        axis.grid(alpha=0.25)
    axes[0].legend()
    return save_figure(fig, output_dir, "learning_curves")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot Step 3 learning curves.")
    parser.add_argument("run_dirs", nargs="+", type=str, help="Run directories to plot.")
    parser.add_argument("--output-dir", type=str, default="vdt_dev/paper_figures")
    parser.add_argument("--metric-key", type=str, default="eval_return_normalized")
    parser.add_argument("--train-metric-key", type=str, default="train_actor_loss")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    plot_learning_curves(
        args.run_dirs,
        output_dir=args.output_dir,
        metric_key=args.metric_key,
        train_metric_key=args.train_metric_key,
    )


if __name__ == "__main__":
    main()
