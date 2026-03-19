from __future__ import annotations

from pathlib import Path

from src.value_functions import TwinQ, ValueFunction
from vdt_dev.analysis.aggregate import aggregate_output_roots
from vdt_dev.analysis.collect import collect_run_summary
from vdt_dev.analysis.schema import RunManifest, append_jsonl, write_dataclass_json, write_json
from vdt_dev.models.vdt_vanilla_policy import VDTVanillaPolicy
from vdt_dev.runner import ExperimentConfig, build_compute_tracker


def test_compute_tracker_generates_machine_readable_summary() -> None:
    config = ExperimentConfig(
        batch_size=4,
        K=4,
        embed_dim=32,
        n_layer=2,
        n_head=4,
        hidden_dim=64,
        n_hidden=2,
        model_variant="vanilla_dev",
        use_attnres=False,
        attnres_query_mode="static",
    )
    policy = VDTVanillaPolicy(
        state_dim=11,
        act_dim=3,
        hidden_size=32,
        action_range=[-1.0, 1.0],
        max_length=4,
        max_ep_len=16,
        n_layer=2,
        n_head=4,
        n_positions=32,
    )
    qf = TwinQ(11, 3, hidden_dim=64, n_hidden=2)
    vf = ValueFunction(11, hidden_dim=64, n_hidden=2)

    tracker = build_compute_tracker(
        config,
        policy=policy,
        qf=qf,
        vf=vf,
        state_dim=11,
        act_dim=3,
    )
    assert tracker is not None
    offline_step = tracker.record_train_step(phase="offline", step_time_sec=0.5)
    online_step = tracker.record_train_step(phase="online", step_time_sec=0.25)
    summary = tracker.finalize(
        total_wall_clock_sec=3.0,
        offline_wall_clock_sec=1.0,
        online_wall_clock_sec=2.0,
    )

    assert offline_step["train_step_time_sec"] == 0.5
    assert online_step["steps_per_sec"] == 4.0
    assert summary["policy_param_count"] > 0
    assert summary["approx_total_train_step_flops"] > 0
    assert summary["total_wall_clock_sec"] == 3.0
    assert summary["offline_mean_train_step_time_sec"] == 0.5
    assert summary["online_mean_train_step_time_sec"] == 0.25


def test_collect_and_aggregate_include_compute_and_online_fields(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "metadata").mkdir(parents=True)
    (run_dir / "logs").mkdir(parents=True)
    (run_dir / "analysis").mkdir(parents=True)

    write_dataclass_json(
        run_dir / "metadata/run_manifest.json",
        RunManifest(
            run_id="run",
            exp_name="exp",
            experiment_family="online_compare",
            env_name="hopper-medium-v2",
            seed=0,
            model_variant="vanilla_dev",
            query_mode="static",
            num_layers=6,
            output_dir=str(run_dir),
            status="completed",
            notes={
                "budget_mode": "matched_params",
                "budget_reference": "L6_E256_H4",
                "budget_target_params": 12345,
                "budget_tolerance_pct": 10.0,
                "budget_actual_gap_pct": 2.5,
            },
        ),
    )
    append_jsonl(
        run_dir / "logs/eval_metrics.jsonl",
        {
            "step": 100,
            "phase": "offline",
            "split": "eval",
            "eval_return_normalized": 80.0,
            "checkpoint_path": str(run_dir / "checkpoints/offline_best.pt"),
        },
    )
    append_jsonl(
        run_dir / "logs/eval_metrics.jsonl",
        {
            "step": 200,
            "phase": "online",
            "split": "eval",
            "eval_return_normalized": 95.0,
            "checkpoint_path": str(run_dir / "checkpoints/online_best.pt"),
        },
    )
    write_json(
        run_dir / "analysis/compute_summary.json",
        {
            "policy_param_count": 111,
            "total_param_count": 222,
            "approx_policy_forward_flops": 333,
            "approx_total_train_step_flops": 444,
            "mean_train_step_time_sec": 0.5,
            "mean_steps_per_sec": 2.0,
            "mean_policy_tokens_per_sec": 96.0,
            "peak_gpu_memory_mb": 256.0,
            "total_wall_clock_sec": 50.0,
            "offline_wall_clock_sec": 20.0,
            "online_wall_clock_sec": 30.0,
        },
    )

    summary = collect_run_summary(run_dir, write_summary=True)
    aggregate = aggregate_output_roots([tmp_path], destination_dir=tmp_path / "agg")
    group = aggregate["groups"][0]

    assert summary["best_offline_eval_return_normalized"] == 80.0
    assert summary["best_online_eval_return_normalized"] == 95.0
    assert summary["online_minus_offline_best_return"] == 15.0
    assert summary["policy_param_count"] == 111
    assert group["policy_param_count_mean"] == 111.0
    assert group["best_online_eval_return_normalized_mean"] == 95.0
