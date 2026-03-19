from __future__ import annotations

from pathlib import Path

from vdt_dev.analysis.plot_depth_source_usage import plot_depth_source_usage
from vdt_dev.analysis.plot_final_bars import plot_final_bars
from vdt_dev.analysis.plot_layer_norms import plot_layer_norms
from vdt_dev.analysis.plot_learning_curves import plot_learning_curves
from vdt_dev.analysis.plot_online_improvement import plot_online_improvement
from vdt_dev.analysis.plot_query_norms import plot_query_norms
from vdt_dev.analysis.plot_routing_heatmap import plot_routing_heatmap
from vdt_dev.analysis.plot_rtg_alignment import plot_rtg_alignment
from vdt_dev.analysis.schema import append_jsonl


def _prepare_fake_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    (run_dir / "logs").mkdir(parents=True)
    append_jsonl(run_dir / "logs/train_metrics.jsonl", {"step": 1, "train_actor_loss": 2.0})
    append_jsonl(run_dir / "logs/train_metrics.jsonl", {"step": 2, "train_actor_loss": 1.0})
    append_jsonl(run_dir / "logs/eval_metrics.jsonl", {"step": 1, "eval_return_normalized": 10.0})
    append_jsonl(run_dir / "logs/eval_metrics.jsonl", {"step": 2, "eval_return_normalized": 20.0})
    append_jsonl(
        run_dir / "logs/debug_metrics.jsonl",
        {
            "step": 2,
            "hidden_norm_by_layer": [1.0, 1.5],
            "output_norm_by_layer": [1.2, 1.7],
            "static_query_norm_mean": 0.5,
            "delta_query_norm_mean": 0.25,
            "fused_query_norm_mean": 0.75,
            "source_distance_by_layer": [0.1, 0.3],
            "source_usage_by_layer": [
                {"layer_name": "layer_00.pre_attn", "weights_mean": [0.7, 0.3]},
                {"layer_name": "layer_01.pre_attn", "weights_mean": [0.2, 0.8]},
            ],
        },
    )
    append_jsonl(run_dir / "logs/rtg_grid_metrics.jsonl", {"rtg_target": 1000.0, "rtg_achieved": 900.0})
    append_jsonl(run_dir / "logs/rtg_grid_metrics.jsonl", {"rtg_target": 2000.0, "rtg_achieved": 1800.0})
    return run_dir


def test_plot_scripts_create_files(tmp_path: Path) -> None:
    run_dir = _prepare_fake_run(tmp_path)
    summary_csv = tmp_path / "summary.csv"
    summary_csv.write_text(
        (
            "env_name,model_variant,query_mode,num_layers,"
            "best_eval_return_normalized_mean,best_eval_return_normalized_std,"
            "best_offline_eval_return_normalized_mean,best_online_eval_return_normalized_mean\n"
            "hopper-medium-v2,vcdr,state_rtg_value,6,20.0,1.0,18.0,22.0\n"
        ),
        encoding="utf-8",
    )
    figure_dir = tmp_path / "figures"

    plot_learning_curves([str(run_dir)], output_dir=str(figure_dir))
    plot_final_bars(summary_csv=str(summary_csv), output_dir=str(figure_dir))
    plot_online_improvement(summary_csv=str(summary_csv), output_dir=str(figure_dir))
    plot_rtg_alignment([str(run_dir)], output_dir=str(figure_dir))
    plot_routing_heatmap(str(run_dir), output_dir=str(figure_dir))
    plot_layer_norms(str(run_dir), output_dir=str(figure_dir))
    plot_query_norms(str(run_dir), output_dir=str(figure_dir))
    plot_depth_source_usage(str(run_dir), output_dir=str(figure_dir))

    assert any(path.suffix == ".png" for path in figure_dir.iterdir())
    assert any(path.suffix == ".pdf" for path in figure_dir.iterdir())
