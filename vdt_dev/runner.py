from __future__ import annotations

import argparse
import json
import math
import random
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from time import perf_counter
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union, get_args, get_origin, get_type_hints

import numpy as np
import torch
import yaml
from tqdm import trange

from vdt_dev.analysis.collect import collect_run_summary, summarize_debug_payload
from vdt_dev.analysis.manifest import (
    build_run_id,
    load_run_manifest,
    manifest_path_for_run_dir,
    update_manifest,
    write_run_manifest,
)
from vdt_dev.analysis.schema import ArtifactPaths, RunManifest, append_jsonl, write_json
from vdt_dev.models.vdt_bar_policy import VDTBARPolicy
from vdt_dev.models.vdt_vanilla_policy import VDTVanillaPolicy
from vdt_dev.model_variants import resolve_model_variant
from vdt_dev.utils.compute import ComputeTracker, build_static_compute_report


ESSENTIAL_KEYS = {
    "observations",
    "actions",
    "rewards",
    "next_observations",
    "terminals",
    "timeouts",
}

ESSENTIAL_KEYS_NO_TIMEOUT = {
    "observations",
    "actions",
    "rewards",
    "next_observations",
    "terminals",
}

DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class ExperimentConfig:
    config: Optional[str] = None
    job: str = "train"
    exp_name: str = ""
    experiment_family: str = ""
    preset_name: str = ""
    model_variant: str = ""
    env_name: str = "hopper-medium-v2"
    seed: int = 123
    discount: float = 0.99
    hidden_dim: int = 256
    n_hidden: int = 2
    n_steps: int = 10000
    batch_size: int = 2048
    learning_rate: float = 3e-4
    alpha: float = 0.005
    tau: float = 0.7
    beta: float = 3.0
    eval_period: int = 100
    n_eval_episodes: int = 5
    ordering: int = 0
    init_temperature: float = 0.1
    online_finetune: bool = True
    replay_size: int = 1000
    num_online_rollouts: int = 1
    eval_interval: int = 50
    max_online_iters: int = 25000
    K: int = 20
    pct_traj: float = 1.0
    embed_dim: int = 256
    n_layer: int = 6
    n_head: int = 4
    activation_function: str = "relu"
    dropout: float = 0.1
    reward_tune: str = "no"
    mode: str = "normal"
    output_dir: str = "vdt_dev/outputs/default"
    checkpoint_every_eval: bool = True
    save_debug_tensors: bool = False
    checkpoint_path: Optional[str] = None
    checkpoint_selector: str = "latest"
    use_attnres: bool = False
    attnres_variant: str = "block"
    attnres_num_blocks: int = 8
    attnres_block_size: Optional[int] = None
    attnres_apply_pre_attn: bool = True
    attnres_apply_pre_mlp: bool = True
    attnres_query_mode: str = "static"
    attnres_query_fusion: str = "additive"
    attnres_conditioner_hidden_dim: int = 128
    attnres_use_value_stopgrad: bool = True
    attnres_value_feature_mode: str = "scalar_v"
    attnres_use_rmsnorm: bool = True
    attnres_zero_init_query: bool = True
    attnres_debug: bool = False
    structured_logging: bool = True
    save_best_checkpoint: bool = True
    primary_metric_name: str = "eval_return_normalized"
    primary_metric_mode: str = "max"
    compute_report: bool = True
    budget_mode: str = "none"
    budget_reference: str = ""
    budget_target_params: int = 0
    budget_tolerance_pct: float = 0.0
    budget_actual_gap_pct: float = 0.0
    skip_if_complete: bool = False
    resume_if_exists: bool = False
    resume_from: Optional[str] = None
    save_resume_buffer: bool = False
    debug_capture_mode: str = "off"
    debug_capture_train: bool = False
    debug_capture_eval: bool = True
    debug_capture_interval: int = 1
    debug_max_artifacts: int = 8
    reeval_rtg_grid: Optional[str] = None
    reeval_num_episodes: int = 5
    reeval_debug: bool = False


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse boolean value from `{value}`.")


def current_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def split_into_trajectories(dataset: Dict[str, np.ndarray]) -> list[Dict[str, np.ndarray]]:
    trajectories: list[Dict[str, np.ndarray]] = []
    current_traj = {key: [] for key in ESSENTIAL_KEYS}
    for index in range(len(dataset["observations"])):
        for key in ESSENTIAL_KEYS:
            current_traj[key].append(dataset[key][index])
        if dataset["terminals"][index] or dataset["timeouts"][index]:
            trajectories.append({key: np.array(value) for key, value in current_traj.items()})
            current_traj = {key: [] for key in ESSENTIAL_KEYS}

    if current_traj["observations"]:
        trajectories.append({key: np.array(value) for key, value in current_traj.items()})
    return trajectories


def process_qlearning_dataset(
    trajectories: list[Dict[str, np.ndarray]],
    *,
    include_timeouts: bool = True,
) -> Dict[str, np.ndarray]:
    keys = ESSENTIAL_KEYS if include_timeouts else ESSENTIAL_KEYS_NO_TIMEOUT
    total_len = sum(len(traj["observations"]) for traj in trajectories)
    output: Dict[str, np.ndarray] = {}
    for key in keys:
        dtype = np.bool_ if key in {"terminals", "timeouts"} else np.float32
        output[key] = np.empty(
            (total_len, *trajectories[0][key].shape[1:]),
            dtype=dtype,
        )

    index = 0
    for traj in trajectories:
        traj_len = len(traj["observations"])
        for key in keys:
            output[key][index : index + traj_len] = traj[key]
        index += traj_len
    return output


def get_env_metadata(env_name: str) -> tuple[int, list[int], float]:
    if "hopper" in env_name:
        return 1000, [72000, 36000, 18000, 7200, 3600, 1800, 720], 1000.0
    if "halfcheetah" in env_name:
        return 1000, [12000, 9000, 6000], 1000.0
    if "walker2d" in env_name:
        return 1000, [5000, 4000, 2500], 1000.0
    raise ValueError(f"Unsupported environment `{env_name}` for the current repo.")


def add_next_observations(raw_dataset: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    if "next_observations" in raw_dataset:
        return raw_dataset

    observations = raw_dataset["observations"]
    terminals = raw_dataset["terminals"]
    timeouts = raw_dataset["timeouts"]
    next_observations = np.empty_like(observations)

    for index in range(len(observations) - 1):
        if terminals[index] or timeouts[index]:
            next_observations[index] = observations[index]
        else:
            next_observations[index] = observations[index + 1]
    next_observations[-1] = observations[-1]

    raw_dataset["next_observations"] = next_observations
    return raw_dataset


def compute_dataset_statistics(
    trajectories: list[Dict[str, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    states, next_observations, traj_lens, returns = [], [], [], []
    for traj in trajectories:
        states.append(traj["observations"])
        next_observations.append(traj["next_observations"])
        traj_lens.append(len(traj["observations"]))
        returns.append(traj["rewards"].sum())

    states_np = np.concatenate(states, axis=0)
    next_obs_np = np.concatenate(next_observations, axis=0)
    traj_lens_np = np.array(traj_lens)
    returns_np = np.array(returns)

    state_mean = np.mean(states_np, axis=0)
    state_std = np.std(states_np, axis=0) + 1e-6
    next_obs_mean = np.mean(next_obs_np, axis=0)
    next_obs_std = np.std(next_obs_np, axis=0) + 1e-6
    return state_mean, state_std, next_obs_mean, next_obs_std, traj_lens_np, returns_np


def normalize_rewards(dataset: Dict[str, np.ndarray], env_name: str, scale: float) -> None:
    if any(name in env_name for name in ("halfcheetah", "hopper", "walker2d")):
        dataset["rewards"] /= scale
    elif "antmaze" in env_name:
        dataset["rewards"] -= 1.0


def select_top_trajectories(
    trajectories: list[Dict[str, np.ndarray]],
    traj_lens: np.ndarray,
    returns: np.ndarray,
    pct_traj: float,
) -> tuple[np.ndarray, np.ndarray]:
    num_timesteps = max(int(pct_traj * int(traj_lens.sum())), 1)
    sorted_inds = np.argsort(returns)
    num_trajectories = 1
    timesteps = traj_lens[sorted_inds[-1]]
    index = len(trajectories) - 2
    while index >= 0 and timesteps + traj_lens[sorted_inds[index]] <= num_timesteps:
        timesteps += traj_lens[sorted_inds[index]]
        num_trajectories += 1
        index -= 1

    sorted_inds = sorted_inds[-num_trajectories:]
    p_sample = traj_lens[sorted_inds] / np.sum(traj_lens[sorted_inds])
    return sorted_inds, p_sample


def torchify(x: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(x)
    if tensor.dtype is torch.float64:
        tensor = tensor.float()
    return tensor.to(device=DEFAULT_DEVICE)


def discount_cumsum(x: np.ndarray, gamma: float) -> np.ndarray:
    output = np.zeros_like(x)
    output[-1] = x[-1]
    for index in reversed(range(x.shape[0] - 1)):
        output[index] = x[index] + gamma * output[index + 1]
    return output


def build_offline_batch_fn(
    trajectories: list[Dict[str, np.ndarray]],
    sorted_inds: np.ndarray,
    p_sample: np.ndarray,
    *,
    batch_size: int,
    max_len: int,
    state_dim: int,
    act_dim: int,
    max_ep_len: int,
    scale: float,
    state_mean: np.ndarray,
    state_std: np.ndarray,
    next_observations_mean: np.ndarray,
    next_observations_std: np.ndarray,
) -> Callable[[], tuple[torch.Tensor, ...]]:
    num_trajectories = len(sorted_inds)

    def get_batch() -> tuple[torch.Tensor, ...]:
        batch_inds = np.random.choice(
            np.arange(num_trajectories),
            size=batch_size,
            replace=True,
            p=p_sample,
        )

        s, a, r, d, rtg = [], [], [], [], []
        timesteps, mask, target_a, next_obs, terminals = [], [], [], [], []

        for batch_index in range(batch_size):
            traj = trajectories[int(sorted_inds[batch_inds[batch_index]])]
            start_index = random.randint(0, traj["rewards"].shape[0] - 1)

            s.append(traj["observations"][start_index : start_index + max_len].reshape(1, -1, state_dim))
            next_obs.append(traj["next_observations"][start_index : start_index + max_len].reshape(1, -1, state_dim))
            a.append(traj["actions"][start_index : start_index + max_len].reshape(1, -1, act_dim))
            target_a.append(traj["actions"][start_index : start_index + max_len].reshape(1, -1, act_dim))
            d.append(traj["terminals"][start_index : start_index + max_len].reshape(1, -1, 1))
            terminals.append(traj["terminals"][start_index : start_index + max_len].reshape(1, -1, 1))

            timestep = np.arange(start_index, start_index + s[-1].shape[1]).reshape(1, -1)
            timestep[timestep >= max_ep_len] = max_ep_len - 1
            timesteps.append(timestep)

            traj_rewards = traj["rewards"]
            r.append(traj_rewards[start_index : start_index + max_len].reshape(1, -1, 1))
            rtg_slice = discount_cumsum(traj_rewards[start_index:], gamma=1.0)[: s[-1].shape[1] + 1]
            rtg.append(rtg_slice.reshape(1, -1, 1))
            if rtg[-1].shape[1] <= s[-1].shape[1]:
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            tlen = s[-1].shape[1]
            s[-1] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1)
            s[-1] = (s[-1] - state_mean) / state_std
            next_obs[-1] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), next_obs[-1]], axis=1)
            next_obs[-1] = (next_obs[-1] - next_observations_mean) / next_observations_std
            a[-1] = np.concatenate([np.zeros((1, max_len - tlen, act_dim)), a[-1]], axis=1)
            r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
            target_a[-1] = np.concatenate([np.zeros((1, max_len - tlen, act_dim)), target_a[-1]], axis=1)
            d[-1] = np.concatenate([np.ones((1, max_len - tlen, 1)), d[-1]], axis=1)
            terminals[-1] = np.concatenate([np.ones((1, max_len - tlen, 1)), terminals[-1]], axis=1)
            rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1) / scale
            timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))

        device = DEFAULT_DEVICE
        return (
            torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device),
            torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device),
            torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device),
            torch.from_numpy(np.concatenate(target_a, axis=0)).to(dtype=torch.float32, device=device),
            torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=device),
            torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device),
            torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device),
            torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device),
            torch.from_numpy(np.concatenate(next_obs, axis=0)).to(dtype=torch.float32, device=device),
            torch.from_numpy(np.concatenate(terminals, axis=0)).to(dtype=torch.bool, device=device),
        )

    return get_batch


def get_batch_online(
    trajectories: list[Dict[str, np.ndarray]],
    *,
    batch_size: int,
    max_len: int,
    pct_traj: float,
    state_dim: int,
    act_dim: int,
    max_ep_len: int,
    scale: float,
) -> tuple[torch.Tensor, ...]:
    state_mean, state_std, next_obs_mean, next_obs_std, traj_lens, returns = compute_dataset_statistics(trajectories)
    sorted_inds, p_sample = select_top_trajectories(trajectories, traj_lens, returns, pct_traj)
    batch_fn = build_offline_batch_fn(
        trajectories,
        sorted_inds,
        p_sample,
        batch_size=batch_size,
        max_len=max_len,
        state_dim=state_dim,
        act_dim=act_dim,
        max_ep_len=max_ep_len,
        scale=scale,
        state_mean=state_mean,
        state_std=state_std,
        next_observations_mean=next_obs_mean,
        next_observations_std=next_obs_std,
    )
    return batch_fn()


def resolve_attnres_num_blocks(config: ExperimentConfig) -> int:
    active_sites = int(config.attnres_apply_pre_attn) + int(config.attnres_apply_pre_mlp)
    total_sites = max(1, active_sites * config.n_layer)
    if config.attnres_block_size is not None:
        return max(1, math.ceil(total_sites / config.attnres_block_size))
    return min(max(1, config.attnres_num_blocks), total_sites)


def build_policy(
    config: ExperimentConfig,
    *,
    obs_dim: int,
    act_dim: int,
    max_ep_len: int,
    action_range: list[float],
    scale: float,
    target_entropy: float,
) -> VDTBARPolicy:
    effective_blocks = resolve_attnres_num_blocks(config)
    resolved_variant = resolve_model_variant(
        explicit=config.model_variant or None,
        use_attnres=config.use_attnres,
        query_mode=config.attnres_query_mode,
    )
    policy_cls = VDTVanillaPolicy if resolved_variant == "vanilla_dev" else VDTBARPolicy
    return policy_cls(
        state_dim=obs_dim,
        act_dim=act_dim,
        max_length=config.K,
        max_ep_len=max_ep_len,
        hidden_size=config.embed_dim,
        action_range=action_range,
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_inner=4 * config.embed_dim,
        activation_function=config.activation_function,
        n_positions=1024,
        resid_pdrop=config.dropout,
        attn_pdrop=config.dropout,
        scale=scale,
        stochastic_policy=False,
        ordering=config.ordering,
        init_temperature=config.init_temperature,
        target_entropy=target_entropy,
        use_attnres=config.use_attnres,
        attnres_variant=config.attnres_variant,
        attnres_num_blocks=effective_blocks,
        attnres_apply_pre_attn=config.attnres_apply_pre_attn,
        attnres_apply_pre_mlp=config.attnres_apply_pre_mlp,
        attnres_query_mode=config.attnres_query_mode,
        attnres_query_fusion=config.attnres_query_fusion,
        attnres_conditioner_hidden_dim=config.attnres_conditioner_hidden_dim,
        attnres_use_value_stopgrad=config.attnres_use_value_stopgrad,
        attnres_value_feature_mode=config.attnres_value_feature_mode,
        attnres_use_rmsnorm=config.attnres_use_rmsnorm,
        attnres_zero_init_query=config.attnres_zero_init_query,
        attnres_debug=config.attnres_debug,
    )


def ensure_output_layout(config: ExperimentConfig) -> Dict[str, Path]:
    root = Path(config.output_dir)
    checkpoints = root / "checkpoints"
    logs = root / "logs"
    metadata = root / "metadata"
    debug = root / "debug"
    debug_routing = debug / "routing"
    debug_norms = debug / "norms"
    debug_query = debug / "query"
    debug_eval_rtg_grid = debug / "eval_rtg_grid"
    analysis = root / "analysis"
    analysis_plots = analysis / "plots"
    analysis_tables = analysis / "tables"
    for path in (
        root,
        checkpoints,
        logs,
        metadata,
        debug,
        debug_routing,
        debug_norms,
        debug_query,
        debug_eval_rtg_grid,
        analysis,
        analysis_plots,
        analysis_tables,
    ):
        path.mkdir(parents=True, exist_ok=True)
    return {
        "root": root,
        "checkpoints": checkpoints,
        "logs": logs,
        "metadata": metadata,
        "debug": debug,
        "debug_routing": debug_routing,
        "debug_norms": debug_norms,
        "debug_query": debug_query,
        "debug_eval_rtg_grid": debug_eval_rtg_grid,
        "analysis": analysis,
        "analysis_plots": analysis_plots,
        "analysis_tables": analysis_tables,
    }


def get_git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            cwd=Path(__file__).resolve().parents[1],
        ).strip()
    except Exception:
        return "unavailable"


def build_manifest(config: ExperimentConfig, paths: Dict[str, Path], git_hash: str) -> RunManifest:
    exp_name = config.exp_name or paths["root"].name
    model_variant = resolve_model_variant(
        explicit=config.model_variant or None,
        use_attnres=config.use_attnres,
        query_mode=config.attnres_query_mode,
    )
    return RunManifest(
        run_id=build_run_id(
            exp_name=exp_name,
            model_variant=model_variant,
            env_name=config.env_name,
            query_mode=config.attnres_query_mode,
            seed=config.seed,
            num_layers=config.n_layer,
            output_dir=str(paths["root"]),
        ),
        exp_name=exp_name,
        experiment_family=config.experiment_family,
        preset_name=config.preset_name,
        env_name=config.env_name,
        seed=config.seed,
        model_variant=model_variant,
        query_mode=config.attnres_query_mode,
        attnres_variant=config.attnres_variant,
        num_layers=config.n_layer,
        hidden_size=config.embed_dim,
        num_heads=config.n_head,
        context_len=config.K,
        use_pre_attn=config.attnres_apply_pre_attn,
        use_pre_mlp=config.attnres_apply_pre_mlp,
        num_blocks=resolve_attnres_num_blocks(config),
        train_steps=config.n_steps + (config.max_online_iters if config.online_finetune else 0),
        eval_interval=config.eval_interval,
        git_hash=git_hash,
        config_path=config.config or str(paths["metadata"] / "config_snapshot.yaml"),
        output_dir=str(paths["root"]),
        job=config.job,
        status="running",
        primary_metric_name=config.primary_metric_name,
        primary_metric_mode=config.primary_metric_mode,
        started_at=current_timestamp(),
        command=" ".join(shlex.quote(arg) for arg in sys.argv),
        artifacts=ArtifactPaths(
            checkpoint_path=str(paths["checkpoints"] / "latest.pt"),
            best_checkpoint_path=str(paths["checkpoints"] / "best.pt"),
            latest_checkpoint_path=str(paths["checkpoints"] / "latest.pt"),
            metrics_jsonl_path=str(paths["logs"] / "train_metrics.jsonl"),
            eval_metrics_jsonl_path=str(paths["logs"] / "eval_metrics.jsonl"),
            debug_metrics_jsonl_path=str(paths["logs"] / "debug_metrics.jsonl"),
            rtg_grid_jsonl_path=str(paths["logs"] / "rtg_grid_metrics.jsonl"),
            plot_output_path=str(paths["analysis_plots"]),
            aggregated_csv_path=str(paths["analysis"] / "summary.json"),
            compute_summary_path=str(paths["analysis"] / "compute_summary.json"),
        ),
        notes={
            "budget_mode": config.budget_mode,
            "budget_reference": config.budget_reference,
            "budget_target_params": config.budget_target_params,
            "budget_tolerance_pct": config.budget_tolerance_pct,
            "budget_actual_gap_pct": config.budget_actual_gap_pct,
        },
    )


def build_compute_tracker(
    config: ExperimentConfig,
    *,
    policy: Any,
    qf: Any,
    vf: Any,
    state_dim: int,
    act_dim: int,
) -> Optional[ComputeTracker]:
    if not config.compute_report:
        return None
    base_report = build_static_compute_report(
        config=config,
        policy=policy,
        qf=qf,
        vf=vf,
        state_dim=state_dim,
        act_dim=act_dim,
    )
    base_report.update(
        {
            "env_name": config.env_name,
            "model_variant": resolve_model_variant(
                explicit=config.model_variant or None,
                use_attnres=config.use_attnres,
                query_mode=config.attnres_query_mode,
            ),
            "query_mode": config.attnres_query_mode,
            "num_layers": config.n_layer,
            "budget_mode": config.budget_mode,
            "budget_reference": config.budget_reference,
            "budget_target_params": config.budget_target_params,
            "budget_tolerance_pct": config.budget_tolerance_pct,
            "budget_actual_gap_pct": config.budget_actual_gap_pct,
        }
    )
    return ComputeTracker(
        base_report=base_report,
        tokens_per_step=int(config.batch_size * config.K * 3),
    )


def save_run_metadata(
    config: ExperimentConfig,
    paths: Dict[str, Path],
    *,
    resume_count: int = 0,
) -> RunManifest:
    config_path = paths["metadata"] / "config_snapshot.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(asdict(config), handle, sort_keys=False)

    command_path = paths["metadata"] / "command.txt"
    command_path.write_text(" ".join(shlex.quote(arg) for arg in sys.argv), encoding="utf-8")

    git_hash = get_git_hash()
    git_hash_path = paths["metadata"] / "git_hash.txt"
    git_hash_path.write_text(git_hash + "\n", encoding="utf-8")

    manifest = build_manifest(config, paths, git_hash)
    existing_manifest = load_run_manifest(paths["root"])
    if existing_manifest is not None and existing_manifest.started_at:
        manifest.started_at = existing_manifest.started_at
    manifest.resume_count = resume_count
    write_run_manifest(manifest_path_for_run_dir(paths["root"]), manifest)
    return manifest


def should_skip_run(config: ExperimentConfig, paths: Dict[str, Path]) -> bool:
    if not config.skip_if_complete:
        return False
    manifest = load_run_manifest(paths["root"])
    if manifest is not None and manifest.status == "completed":
        print(f"Skipping completed run at {paths['root']}")
        return True
    return False


def ensure_nontrain_run_metadata(
    config: ExperimentConfig,
    paths: Dict[str, Path],
) -> bool:
    """
    Eval / reeval jobs may write into a fresh output directory that has no manifest yet.

    In that case we snapshot the resolved config so Step 3 discovery / aggregation can
    treat the directory as a first-class run root. If the target directory is the
    original training run root, we leave its metadata untouched.
    """

    if load_run_manifest(paths["root"]) is not None:
        return False
    save_run_metadata(config, paths, resume_count=0)
    return True


def save_checkpoint(
    checkpoint_path: Path,
    *,
    step: int,
    phase: str,
    offline_step: int,
    online_step: int,
    best_metric_value: Optional[float],
    policy: VDTBARPolicy,
    qf: Any,
    vf: Any,
    iql: Any,
    config: ExperimentConfig,
    state_mean: np.ndarray,
    state_std: np.ndarray,
    max_ep_len: int,
    scale: float,
    replay_buffer_trajectories: Optional[list[Dict[str, np.ndarray]]] = None,
) -> None:
    payload = {
        "step": step,
        "phase": phase,
        "policy": policy.state_dict(),
        "qf": qf.state_dict(),
        "vf": vf.state_dict(),
        "q_target": iql.q_target.state_dict(),
        "config": asdict(config),
        "state_mean": state_mean,
        "state_std": state_std,
        "max_ep_len": max_ep_len,
        "scale": scale,
        "optimizers": {
            "vf": iql.v_optimizer.state_dict(),
            "policy": iql.policy_optimizer.state_dict(),
            "qf1": iql.qf1_optimizer.state_dict(),
            "qf2": iql.qf2_optimizer.state_dict(),
            "policy_lr_schedule": iql.policy_lr_schedule.state_dict(),
        },
        "train_state": {
            "phase": phase,
            "offline_step": offline_step,
            "online_step": online_step,
            "best_metric_value": best_metric_value,
        },
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    if replay_buffer_trajectories is not None:
        payload["replay_buffer_trajectories"] = replay_buffer_trajectories
    torch.save(payload, checkpoint_path)


def restore_rng_state(payload: Dict[str, Any]) -> None:
    rng_state = payload.get("rng_state")
    if not rng_state:
        return
    try:
        random.setstate(rng_state["python"])
        np.random.set_state(rng_state["numpy"])
        torch.set_rng_state(rng_state["torch"])
        if torch.cuda.is_available() and rng_state.get("cuda") is not None:
            torch.cuda.set_rng_state_all(rng_state["cuda"])
    except Exception:
        pass


def maybe_load_resume_state(
    config: ExperimentConfig,
    paths: Dict[str, Path],
    *,
    policy: VDTBARPolicy,
    qf: Any,
    vf: Any,
    iql: Any,
) -> Dict[str, Any]:
    if config.resume_from is not None:
        checkpoint_path = Path(config.resume_from)
    elif config.resume_if_exists:
        checkpoint_path = paths["checkpoints"] / "latest.pt"
    else:
        return {
            "offline_step": 0,
            "online_step": 0,
            "phase": "offline",
            "best_metric_value": None,
            "resume_count": 0,
            "replay_buffer_trajectories": None,
        }

    if not checkpoint_path.exists():
        return {
            "offline_step": 0,
            "online_step": 0,
            "phase": "offline",
            "best_metric_value": None,
            "resume_count": 0,
            "replay_buffer_trajectories": None,
        }

    payload = torch.load(checkpoint_path, map_location=DEFAULT_DEVICE)
    policy.load_state_dict(payload["policy"])
    qf.load_state_dict(payload["qf"])
    vf.load_state_dict(payload["vf"])
    if "q_target" in payload:
        iql.q_target.load_state_dict(payload["q_target"])
    optimizers = payload.get("optimizers", {})
    if optimizers:
        if "vf" in optimizers:
            iql.v_optimizer.load_state_dict(optimizers["vf"])
        if "policy" in optimizers:
            iql.policy_optimizer.load_state_dict(optimizers["policy"])
        if "qf1" in optimizers:
            iql.qf1_optimizer.load_state_dict(optimizers["qf1"])
        if "qf2" in optimizers:
            iql.qf2_optimizer.load_state_dict(optimizers["qf2"])
        schedule_state = optimizers.get("policy_lr_schedule")
        if schedule_state is not None:
            iql.policy_lr_schedule.load_state_dict(schedule_state)
    restore_rng_state(payload)

    manifest = load_run_manifest(paths["root"])
    return {
        "offline_step": int(payload.get("train_state", {}).get("offline_step", payload.get("step", 0))),
        "online_step": int(payload.get("train_state", {}).get("online_step", 0)),
        "phase": payload.get("train_state", {}).get("phase", payload.get("phase", "offline")),
        "best_metric_value": payload.get("train_state", {}).get("best_metric_value"),
        "resume_count": 0 if manifest is None else manifest.resume_count + 1,
        "replay_buffer_trajectories": payload.get("replay_buffer_trajectories"),
    }


def _primary_metric_is_better(
    candidate: Optional[float],
    best_value: Optional[float],
    mode: str,
) -> bool:
    if candidate is None:
        return False
    if best_value is None:
        return True
    if mode == "min":
        return candidate < best_value
    return candidate > best_value


def _float_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        value_float = float(value)
    except (TypeError, ValueError):
        return None
    if math.isfinite(value_float):
        return value_float
    return None


def _parse_rtg_grid(grid: Optional[str]) -> list[float]:
    if grid is None or grid == "":
        return []
    return [float(item.strip()) for item in grid.split(",") if item.strip()]


def _clear_debug_hook(policy: VDTBARPolicy) -> None:
    transformer = getattr(policy, "transformer", None)
    debug_hook = getattr(transformer, "debug_hook", None)
    if debug_hook is not None:
        debug_hook.clear()


def _debug_enabled(config: ExperimentConfig, split: str, step: int) -> bool:
    if not (config.attnres_debug and config.use_attnres):
        return False
    if split == "train" and not config.debug_capture_train:
        return False
    if split in {"eval", "rtg_grid"} and not (config.debug_capture_eval or config.reeval_debug):
        return False
    if config.debug_capture_mode == "off" and not config.save_debug_tensors:
        return False
    if config.debug_capture_interval <= 0:
        return False
    return step % config.debug_capture_interval == 0


def _write_debug_summary(
    config: ExperimentConfig,
    paths: Dict[str, Path],
    policy: VDTBARPolicy,
    *,
    step: int,
    phase: str,
    split: str,
    artifact_stem: str,
    artifacts_written: int,
) -> tuple[Optional[Dict[str, Any]], int]:
    if not (config.attnres_debug and config.use_attnres):
        return None, artifacts_written
    payload = policy.transformer.debug_hook.state_dict()
    if not payload.get("routing_weights") and not payload.get("events"):
        _clear_debug_hook(policy)
        return None, artifacts_written

    artifact_path: Optional[Path] = None
    should_save_artifact = (
        config.save_debug_tensors or config.debug_capture_mode == "full"
    ) and artifacts_written < config.debug_max_artifacts
    if should_save_artifact:
        if split == "rtg_grid":
            artifact_path = paths["debug_eval_rtg_grid"] / f"{artifact_stem}.pt"
        else:
            artifact_path = paths["debug_routing"] / f"{artifact_stem}.pt"
        policy.save_debug_tensors(artifact_path)
        artifacts_written += 1

    summary_record = summarize_debug_payload(
        payload,
        step=step,
        phase=phase,
        split=split,
        artifact_path=str(artifact_path) if artifact_path is not None else None,
    )
    append_jsonl(paths["logs"] / "debug_metrics.jsonl", asdict(summary_record))
    if split == "rtg_grid":
        summary_json_path = paths["debug_eval_rtg_grid"] / f"{artifact_stem}_summary.json"
        write_json(summary_json_path, asdict(summary_record))
    _clear_debug_hook(policy)
    return asdict(summary_record), artifacts_written


def build_train_record(
    step: int,
    phase: str,
    outputs: Dict[str, Any],
    *,
    compute_metrics: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    record = {
        "step": step,
        "phase": phase,
        "split": "train",
        "train_actor_loss": _float_or_none(outputs.get("Actor Loss")),
        "train_q_loss": _float_or_none(outputs.get("Q Loss", outputs.get("QL Loss"))),
        "train_v_loss": _float_or_none(outputs.get("Value Loss")),
    }
    record.update(outputs)
    if compute_metrics:
        record.update(compute_metrics)
    return record


def build_eval_record(
    *,
    step: int,
    phase: str,
    outputs: Dict[str, Any],
    checkpoint_path: str,
    best_eval_return_normalized: Optional[float],
) -> Dict[str, Any]:
    record = {
        "step": step,
        "phase": phase,
        "split": "eval",
        "eval_return_raw": _float_or_none(outputs.get("eval_return_raw_mean")),
        "eval_return_raw_std": _float_or_none(outputs.get("eval_return_raw_std")),
        "eval_return_normalized": _float_or_none(outputs.get("eval_return_normalized")),
        "eval_length": _float_or_none(outputs.get("eval_length")),
        "best_eval_return_normalized": best_eval_return_normalized,
        "checkpoint_path": checkpoint_path,
    }
    record.update(outputs)
    return record


def evaluate_policy_detailed(
    env: Any,
    qf: Any,
    policy: Any,
    env_targets: list[int],
    num_eval_episodes: int,
    scale: float,
    state_dim: int,
    act_dim: int,
    max_ep_len: int,
    mode: str,
    state_mean: np.ndarray,
    state_std: np.ndarray,
) -> Dict[str, Any]:
    from src.util import evaluate_episode_rtg

    returns, lengths = [], []
    for _ in range(num_eval_episodes):
        with torch.no_grad():
            episode_return, episode_length = evaluate_episode_rtg(
                env,
                state_dim,
                act_dim,
                policy,
                qf,
                max_ep_len=max_ep_len,
                scale=scale,
                target_return=[target / scale for target in env_targets],
                state_mean=state_mean,
                state_std=state_std,
                mode=mode,
            )
        returns.append(float(episode_return))
        lengths.append(float(episode_length))

    mean_return = float(np.mean(returns))
    normalized = (
        float(env.get_normalized_score(mean_return) * 100.0)
        if hasattr(env, "get_normalized_score")
        else None
    )
    return {
        "eval_return_raw_mean": mean_return,
        "eval_return_raw_std": float(np.std(returns)),
        "eval_length": float(np.mean(lengths)),
        "eval_length_std": float(np.std(lengths)),
        "eval_return_normalized": normalized,
        "target_normalized_score": normalized,
        "target_length_mean": float(np.mean(lengths)),
    }


def save_eval_checkpoint_bundle(
    config: ExperimentConfig,
    paths: Dict[str, Path],
    *,
    step: int,
    phase: str,
    offline_step: int,
    online_step: int,
    best_metric_value: Optional[float],
    policy: VDTBARPolicy,
    qf: Any,
    vf: Any,
    iql: Any,
    state_mean: np.ndarray,
    state_std: np.ndarray,
    max_ep_len: int,
    scale: float,
    replay_buffer_trajectories: Optional[list[Dict[str, np.ndarray]]] = None,
) -> Path:
    checkpoint_path = paths["checkpoints"] / f"{phase}_step_{step:05d}.pt"
    if config.checkpoint_every_eval:
        save_checkpoint(
            checkpoint_path,
            step=step,
            phase=phase,
            offline_step=offline_step,
            online_step=online_step,
            best_metric_value=best_metric_value,
            policy=policy,
            qf=qf,
            vf=vf,
            iql=iql,
            config=config,
            state_mean=state_mean,
            state_std=state_std,
            max_ep_len=max_ep_len,
            scale=scale,
            replay_buffer_trajectories=replay_buffer_trajectories,
        )
    latest_path = paths["checkpoints"] / "latest.pt"
    save_checkpoint(
        latest_path,
        step=step,
        phase=phase,
        offline_step=offline_step,
        online_step=online_step,
        best_metric_value=best_metric_value,
        policy=policy,
        qf=qf,
        vf=vf,
        iql=iql,
        config=config,
        state_mean=state_mean,
        state_std=state_std,
        max_ep_len=max_ep_len,
        scale=scale,
        replay_buffer_trajectories=replay_buffer_trajectories,
    )
    return checkpoint_path if config.checkpoint_every_eval else latest_path


def maybe_save_best_checkpoint(
    config: ExperimentConfig,
    paths: Dict[str, Path],
    *,
    metric_value: Optional[float],
    best_metric_value: Optional[float],
    step: int,
    phase: str,
    offline_step: int,
    online_step: int,
    policy: VDTBARPolicy,
    qf: Any,
    vf: Any,
    iql: Any,
    state_mean: np.ndarray,
    state_std: np.ndarray,
    max_ep_len: int,
    scale: float,
    replay_buffer_trajectories: Optional[list[Dict[str, np.ndarray]]] = None,
) -> Optional[float]:
    if not config.save_best_checkpoint:
        return best_metric_value
    if not _primary_metric_is_better(metric_value, best_metric_value, config.primary_metric_mode):
        return best_metric_value
    save_checkpoint(
        paths["checkpoints"] / "best.pt",
        step=step,
        phase=phase,
        offline_step=offline_step,
        online_step=online_step,
        best_metric_value=metric_value,
        policy=policy,
        qf=qf,
        vf=vf,
        iql=iql,
        config=config,
        state_mean=state_mean,
        state_std=state_std,
        max_ep_len=max_ep_len,
        scale=scale,
        replay_buffer_trajectories=replay_buffer_trajectories,
    )
    return metric_value


def train(config: ExperimentConfig) -> None:
    import gym
    from replay_buffer import ReplayBuffer
    from src.util import set_seed, vec_evaluate_episode_rtg
    from src.value_functions import TwinQ, ValueFunction
    from vdt_dev.iql import DevImplicitQLearning
    from vdt_dev.value_conditioning import ValueConditionedPolicyAdapter

    torch.set_num_threads(1)
    max_ep_len, env_targets, scale = get_env_metadata(config.env_name)
    paths = ensure_output_layout(config)
    if should_skip_run(config, paths):
        return

    env = gym.make(config.env_name)
    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    raw_dataset = add_next_observations(env.get_dataset())
    filtered_dataset = {key: raw_dataset[key] for key in ESSENTIAL_KEYS}
    trajectories = split_into_trajectories(filtered_dataset)

    (
        state_mean,
        state_std,
        next_obs_mean,
        next_obs_std,
        traj_lens,
        returns,
    ) = compute_dataset_statistics(trajectories)
    num_timesteps = int(traj_lens.sum())

    print("=" * 50)
    print(f"Starting dev experiment: {config.env_name}")
    print(f"{len(traj_lens)} trajectories, {num_timesteps} timesteps found")
    print(f"Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}")
    print(f"Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}")

    sorted_inds, p_sample = select_top_trajectories(trajectories, traj_lens, returns, config.pct_traj)
    offline_trajs = [trajectories[index] for index in sorted_inds]

    dataset_np = process_qlearning_dataset(trajectories)
    normalize_rewards(dataset_np, config.env_name, scale)
    dataset_np["observations"] = (dataset_np["observations"] - state_mean) / state_std
    dataset = {key: torchify(value) for key, value in dataset_np.items() if key != "timeouts"}

    action_range = [
        float(env.action_space.low.min()) + 1e-6,
        float(env.action_space.high.max()) - 1e-6,
    ]
    set_seed(config.seed, env=env)
    target_entropy = -act_dim
    policy = build_policy(
        config,
        obs_dim=state_dim,
        act_dim=act_dim,
        max_ep_len=max_ep_len,
        action_range=action_range,
        scale=scale,
        target_entropy=target_entropy,
    )
    qf = TwinQ(state_dim, act_dim, hidden_dim=config.hidden_dim, n_hidden=config.n_hidden)
    vf = ValueFunction(state_dim, hidden_dim=config.hidden_dim, n_hidden=config.n_hidden)
    iql = DevImplicitQLearning(
        qf=qf,
        vf=vf,
        policy=policy,
        optimizer_factory=lambda params: torch.optim.Adam(params, lr=config.learning_rate),
        max_steps=config.n_steps,
        tau=config.tau,
        beta=config.beta,
        alpha=config.alpha,
        discount=config.discount,
        learning_rate=config.learning_rate,
    )
    eval_policy = ValueConditionedPolicyAdapter(policy, vf)
    compute_tracker = build_compute_tracker(
        config,
        policy=policy,
        qf=qf,
        vf=vf,
        state_dim=state_dim,
        act_dim=act_dim,
    )
    if config.compute_report and torch.cuda.is_available():
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

    resume_state = maybe_load_resume_state(
        config,
        paths,
        policy=policy,
        qf=qf,
        vf=vf,
        iql=iql,
    )
    save_run_metadata(config, paths, resume_count=resume_state["resume_count"])

    batch_fn = build_offline_batch_fn(
        trajectories,
        sorted_inds,
        p_sample,
        batch_size=config.batch_size,
        max_len=config.K,
        state_dim=state_dim,
        act_dim=act_dim,
        max_ep_len=max_ep_len,
        scale=scale,
        state_mean=state_mean,
        state_std=state_std,
        next_observations_mean=next_obs_mean,
        next_observations_std=next_obs_std,
    )

    train_log = paths["logs"] / "train_metrics.jsonl"
    eval_log = paths["logs"] / "eval_metrics.jsonl"
    best_metric_value = _float_or_none(resume_state["best_metric_value"])
    debug_artifacts_written = 0
    train_start_time = perf_counter()
    offline_phase_start_time = perf_counter()
    offline_wall_clock_sec = 0.0
    online_wall_clock_sec = 0.0

    offline_start = int(resume_state["offline_step"]) if resume_state["phase"] == "offline" else config.n_steps
    online_start = int(resume_state["online_step"]) if resume_state["phase"] == "online" else 0

    for step in trange(offline_start, config.n_steps, initial=offline_start, total=config.n_steps):
        if config.attnres_debug:
            _clear_debug_hook(policy)
        step_start_time = perf_counter()
        outputs = iql.update(dataset, config.batch_size, DT_batch=batch_fn())
        step_compute_metrics = (
            None
            if compute_tracker is None
            else compute_tracker.record_train_step(
                phase="offline",
                step_time_sec=perf_counter() - step_start_time,
            )
        )
        train_record = build_train_record(
            step + 1,
            "offline",
            outputs,
            compute_metrics=step_compute_metrics,
        )
        append_jsonl(train_log, train_record)

        if _debug_enabled(config, "train", step + 1):
            _, debug_artifacts_written = _write_debug_summary(
                config,
                paths,
                policy,
                step=step + 1,
                phase="offline",
                split="train",
                artifact_stem=f"offline_step_{step + 1:05d}_train",
                artifacts_written=debug_artifacts_written,
            )
        elif config.attnres_debug:
            _clear_debug_hook(policy)

        is_eval_step = (step + 1) % config.eval_period == 0 or step == config.n_steps - 1
        if not is_eval_step:
            continue

        if config.attnres_debug:
            _clear_debug_hook(policy)
        eval_outputs = evaluate_policy_detailed(
            env,
            qf,
            eval_policy,
            env_targets,
            config.n_eval_episodes,
            scale,
            state_dim,
            act_dim,
            max_ep_len,
            config.mode,
            state_mean,
            state_std,
        )
        metric_value = _float_or_none(eval_outputs.get(config.primary_metric_name))
        best_metric_value = maybe_save_best_checkpoint(
            config,
            paths,
            metric_value=metric_value,
            best_metric_value=best_metric_value,
            step=step + 1,
            phase="offline",
            offline_step=step + 1,
            online_step=0,
            policy=policy,
            qf=qf,
            vf=vf,
            iql=iql,
            state_mean=state_mean,
            state_std=state_std,
            max_ep_len=max_ep_len,
            scale=scale,
        )
        checkpoint_path = save_eval_checkpoint_bundle(
            config,
            paths,
            step=step + 1,
            phase="offline",
            offline_step=step + 1,
            online_step=0,
            best_metric_value=best_metric_value,
            policy=policy,
            qf=qf,
            vf=vf,
            iql=iql,
            state_mean=state_mean,
            state_std=state_std,
            max_ep_len=max_ep_len,
            scale=scale,
        )
        eval_record = build_eval_record(
            step=step + 1,
            phase="offline",
            outputs=eval_outputs,
            checkpoint_path=str(checkpoint_path),
            best_eval_return_normalized=best_metric_value,
        )
        append_jsonl(eval_log, eval_record)
        if _debug_enabled(config, "eval", step + 1):
            _, debug_artifacts_written = _write_debug_summary(
                config,
                paths,
                policy,
                step=step + 1,
                phase="offline",
                split="eval",
                artifact_stem=f"offline_step_{step + 1:05d}_eval",
                artifacts_written=debug_artifacts_written,
            )
        elif config.attnres_debug:
            _clear_debug_hook(policy)
        update_manifest(
            paths["root"],
            artifacts={
                "latest_checkpoint_path": str(paths["checkpoints"] / "latest.pt"),
                "best_checkpoint_path": str(paths["checkpoints"] / "best.pt")
                if (paths["checkpoints"] / "best.pt").exists()
                else None,
            },
            notes={"best_eval_return_normalized": best_metric_value},
        )

    replay_buffer_seed = resume_state.get("replay_buffer_trajectories")
    replay_buffer = ReplayBuffer(
        config.replay_size,
        replay_buffer_seed if replay_buffer_seed is not None else offline_trajs,
    )
    offline_wall_clock_sec = perf_counter() - offline_phase_start_time
    if not config.online_finetune:
        update_manifest(paths["root"], status="completed", finished_at=current_timestamp())
        if compute_tracker is not None:
            write_json(
                paths["analysis"] / "compute_summary.json",
                compute_tracker.finalize(
                    total_wall_clock_sec=perf_counter() - train_start_time,
                    offline_wall_clock_sec=offline_wall_clock_sec,
                ),
            )
        collect_run_summary(paths["root"], write_summary=True)
        return

    def augment_trajectories() -> None:
        with torch.no_grad():
            returns_np, lengths_np, new_trajs = vec_evaluate_episode_rtg(
                env,
                state_dim,
                act_dim,
                eval_policy,
                max_ep_len=max_ep_len,
                scale=scale,
                target_return=7200 / scale,
                mode="normal",
                state_mean=state_mean,
                state_std=state_std,
                device=DEFAULT_DEVICE,
                use_mean=False,
            )
        replay_buffer.add_new_trajs(new_trajs)
        print("aug_traj/return", np.mean(returns_np), "\naug_traj/length", np.mean(lengths_np))

    online_phase_start_time = perf_counter()
    for online_iter in trange(
        online_start,
        config.max_online_iters,
        initial=online_start,
        total=config.max_online_iters,
    ):
        augment_trajectories()
        online_dataset_np = process_qlearning_dataset(replay_buffer.trajectories, include_timeouts=False)
        normalize_rewards(online_dataset_np, config.env_name, scale)
        online_dataset_np["observations"] = (online_dataset_np["observations"] - state_mean) / state_std
        online_dataset = {
            key: torchify(value)
            for key, value in online_dataset_np.items()
            if key != "timeouts"
        }
        if config.attnres_debug:
            _clear_debug_hook(policy)
        step_start_time = perf_counter()
        outputs = iql.update(
            online_dataset,
            config.batch_size,
            DT_batch=get_batch_online(
                replay_buffer.trajectories,
                batch_size=config.batch_size,
                max_len=config.K,
                pct_traj=config.pct_traj,
                state_dim=state_dim,
                act_dim=act_dim,
                max_ep_len=max_ep_len,
                scale=scale,
            ),
        )
        step_compute_metrics = (
            None
            if compute_tracker is None
            else compute_tracker.record_train_step(
                phase="online",
                step_time_sec=perf_counter() - step_start_time,
            )
        )
        append_jsonl(
            train_log,
            build_train_record(
                online_iter + 1,
                "online",
                outputs,
                compute_metrics=step_compute_metrics,
            ),
        )
        if _debug_enabled(config, "train", online_iter + 1):
            _, debug_artifacts_written = _write_debug_summary(
                config,
                paths,
                policy,
                step=online_iter + 1,
                phase="online",
                split="train",
                artifact_stem=f"online_step_{online_iter + 1:05d}_train",
                artifacts_written=debug_artifacts_written,
            )
        elif config.attnres_debug:
            _clear_debug_hook(policy)

        is_eval_step = online_iter % config.eval_interval == 0 or online_iter == config.max_online_iters - 1
        if not is_eval_step:
            continue

        if config.attnres_debug:
            _clear_debug_hook(policy)
        eval_outputs = evaluate_policy_detailed(
            env,
            qf,
            eval_policy,
            env_targets,
            config.n_eval_episodes,
            scale,
            state_dim,
            act_dim,
            max_ep_len,
            config.mode,
            state_mean,
            state_std,
        )
        metric_value = _float_or_none(eval_outputs.get(config.primary_metric_name))
        best_metric_value = maybe_save_best_checkpoint(
            config,
            paths,
            metric_value=metric_value,
            best_metric_value=best_metric_value,
            step=online_iter + 1,
            phase="online",
            offline_step=config.n_steps,
            online_step=online_iter + 1,
            policy=policy,
            qf=qf,
            vf=vf,
            iql=iql,
            state_mean=state_mean,
            state_std=state_std,
            max_ep_len=max_ep_len,
            scale=scale,
            replay_buffer_trajectories=(
                replay_buffer.trajectories if config.save_resume_buffer else None
            ),
        )
        checkpoint_path = save_eval_checkpoint_bundle(
            config,
            paths,
            step=online_iter + 1,
            phase="online",
            offline_step=config.n_steps,
            online_step=online_iter + 1,
            best_metric_value=best_metric_value,
            policy=policy,
            qf=qf,
            vf=vf,
            iql=iql,
            state_mean=state_mean,
            state_std=state_std,
            max_ep_len=max_ep_len,
            scale=scale,
            replay_buffer_trajectories=(
                replay_buffer.trajectories if config.save_resume_buffer else None
            ),
        )
        append_jsonl(
            eval_log,
            build_eval_record(
                step=online_iter + 1,
                phase="online",
                outputs=eval_outputs,
                checkpoint_path=str(checkpoint_path),
                best_eval_return_normalized=best_metric_value,
            ),
        )
        if _debug_enabled(config, "eval", online_iter + 1):
            _, debug_artifacts_written = _write_debug_summary(
                config,
                paths,
                policy,
                step=online_iter + 1,
                phase="online",
                split="eval",
                artifact_stem=f"online_step_{online_iter + 1:05d}_eval",
                artifacts_written=debug_artifacts_written,
            )
        elif config.attnres_debug:
            _clear_debug_hook(policy)
        update_manifest(
            paths["root"],
            artifacts={
                "latest_checkpoint_path": str(paths["checkpoints"] / "latest.pt"),
                "best_checkpoint_path": str(paths["checkpoints"] / "best.pt")
                if (paths["checkpoints"] / "best.pt").exists()
                else None,
            },
            notes={"best_eval_return_normalized": best_metric_value},
        )

    online_wall_clock_sec = perf_counter() - online_phase_start_time
    if compute_tracker is not None:
        write_json(
            paths["analysis"] / "compute_summary.json",
            compute_tracker.finalize(
                total_wall_clock_sec=perf_counter() - train_start_time,
                offline_wall_clock_sec=offline_wall_clock_sec,
                online_wall_clock_sec=online_wall_clock_sec,
            ),
        )
    update_manifest(paths["root"], status="completed", finished_at=current_timestamp())
    collect_run_summary(paths["root"], write_summary=True)


def load_checkpoint_config(config: ExperimentConfig) -> tuple[ExperimentConfig, Dict[str, Any]]:
    if config.checkpoint_path is None:
        raise ValueError("`checkpoint_path` is required for evaluation.")
    payload = torch.load(config.checkpoint_path, map_location=DEFAULT_DEVICE)
    checkpoint_config = ExperimentConfig(**payload["config"])
    checkpoint_config.job = config.job
    checkpoint_config.checkpoint_path = config.checkpoint_path
    checkpoint_config.output_dir = (
        config.output_dir
        if config.output_dir != ExperimentConfig().output_dir
        else str(Path(config.checkpoint_path).resolve().parents[1])
    )
    if config.n_eval_episodes != ExperimentConfig().n_eval_episodes:
        checkpoint_config.n_eval_episodes = config.n_eval_episodes
    if config.reeval_rtg_grid is not None:
        checkpoint_config.reeval_rtg_grid = config.reeval_rtg_grid
    if config.reeval_num_episodes != ExperimentConfig().reeval_num_episodes:
        checkpoint_config.reeval_num_episodes = config.reeval_num_episodes
    if config.reeval_debug:
        checkpoint_config.reeval_debug = True
        checkpoint_config.attnres_debug = True
    if config.attnres_debug:
        checkpoint_config.attnres_debug = True
    return checkpoint_config, payload


def evaluate(config: ExperimentConfig) -> None:
    import gym
    from src.value_functions import TwinQ, ValueFunction
    from vdt_dev.value_conditioning import ValueConditionedPolicyAdapter

    run_config, payload = load_checkpoint_config(config)
    max_ep_len, env_targets, scale = get_env_metadata(run_config.env_name)
    env = gym.make(run_config.env_name)
    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    action_range = [
        float(env.action_space.low.min()) + 1e-6,
        float(env.action_space.high.max()) - 1e-6,
    ]
    target_entropy = -act_dim

    policy = build_policy(
        run_config,
        obs_dim=state_dim,
        act_dim=act_dim,
        max_ep_len=max_ep_len,
        action_range=action_range,
        scale=scale,
        target_entropy=target_entropy,
    )
    qf = TwinQ(state_dim, act_dim, hidden_dim=run_config.hidden_dim, n_hidden=run_config.n_hidden)
    vf = ValueFunction(state_dim, hidden_dim=run_config.hidden_dim, n_hidden=run_config.n_hidden)

    policy.load_state_dict(payload["policy"])
    qf.load_state_dict(payload["qf"])
    vf.load_state_dict(payload["vf"])
    eval_policy = ValueConditionedPolicyAdapter(policy, vf)
    compute_tracker = build_compute_tracker(
        run_config,
        policy=policy,
        qf=qf,
        vf=vf,
        state_dim=state_dim,
        act_dim=act_dim,
    )
    eval_start_time = perf_counter()
    if run_config.compute_report and torch.cuda.is_available():
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

    state_mean = payload["state_mean"]
    state_std = payload["state_std"]
    paths = ensure_output_layout(run_config)
    created_metadata = ensure_nontrain_run_metadata(run_config, paths)
    eval_outputs = evaluate_policy_detailed(
        env,
        qf,
        eval_policy,
        env_targets,
        run_config.n_eval_episodes,
        payload["scale"],
        state_dim,
        act_dim,
        payload["max_ep_len"],
        run_config.mode,
        state_mean,
        state_std,
    )
    checkpoint_path = run_config.checkpoint_path or ""
    append_jsonl(
        paths["logs"] / "eval_metrics.jsonl",
        build_eval_record(
            step=int(payload.get("step", 0)),
            phase="eval_only",
            outputs=eval_outputs,
            checkpoint_path=checkpoint_path,
            best_eval_return_normalized=_float_or_none(eval_outputs.get("eval_return_normalized")),
        ),
    )
    if _debug_enabled(run_config, "eval", int(payload.get("step", 0))):
        _write_debug_summary(
            run_config,
            paths,
            policy,
            step=int(payload.get("step", 0)),
            phase="eval_only",
            split="eval",
            artifact_stem=f"eval_only_step_{int(payload.get('step', 0)):05d}",
            artifacts_written=0,
        )
    if compute_tracker is not None:
        write_json(
            paths["analysis"] / "compute_summary.json",
            compute_tracker.finalize(
                total_wall_clock_sec=perf_counter() - eval_start_time,
                eval_wall_clock_sec=perf_counter() - eval_start_time,
            ),
        )
    collect_run_summary(paths["root"], write_summary=True)
    if created_metadata:
        update_manifest(paths["root"], status="completed", finished_at=current_timestamp())
    print(json.dumps(eval_outputs, indent=2, sort_keys=True))


def reeval_rtg_grid(config: ExperimentConfig) -> None:
    import gym
    from src.util import evaluate_episode_rtg
    from src.value_functions import TwinQ, ValueFunction
    from vdt_dev.value_conditioning import ValueConditionedPolicyAdapter

    run_config, payload = load_checkpoint_config(config)
    rtg_grid = _parse_rtg_grid(run_config.reeval_rtg_grid)
    if not rtg_grid:
        raise ValueError("`reeval_rtg_grid` must be a comma-separated list of raw RTG targets.")

    max_ep_len, _env_targets, scale = get_env_metadata(run_config.env_name)
    env = gym.make(run_config.env_name)
    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    action_range = [
        float(env.action_space.low.min()) + 1e-6,
        float(env.action_space.high.max()) - 1e-6,
    ]
    target_entropy = -act_dim

    policy = build_policy(
        run_config,
        obs_dim=state_dim,
        act_dim=act_dim,
        max_ep_len=max_ep_len,
        action_range=action_range,
        scale=scale,
        target_entropy=target_entropy,
    )
    qf = TwinQ(state_dim, act_dim, hidden_dim=run_config.hidden_dim, n_hidden=run_config.n_hidden)
    vf = ValueFunction(state_dim, hidden_dim=run_config.hidden_dim, n_hidden=run_config.n_hidden)

    policy.load_state_dict(payload["policy"])
    qf.load_state_dict(payload["qf"])
    vf.load_state_dict(payload["vf"])
    eval_policy = ValueConditionedPolicyAdapter(policy, vf)
    compute_tracker = build_compute_tracker(
        run_config,
        policy=policy,
        qf=qf,
        vf=vf,
        state_dim=state_dim,
        act_dim=act_dim,
    )
    reeval_start_time = perf_counter()
    if run_config.compute_report and torch.cuda.is_available():
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
    state_mean = payload["state_mean"]
    state_std = payload["state_std"]
    paths = ensure_output_layout(run_config)
    created_metadata = ensure_nontrain_run_metadata(run_config, paths)
    debug_artifacts_written = 0

    for target in rtg_grid:
        for episode_index in range(run_config.reeval_num_episodes):
            if run_config.attnres_debug:
                _clear_debug_hook(policy)
            with torch.no_grad():
                achieved_return, episode_length = evaluate_episode_rtg(
                    env,
                    state_dim,
                    act_dim,
                    eval_policy,
                    qf,
                    max_ep_len=payload["max_ep_len"],
                    scale=payload["scale"],
                    target_return=[target / scale],
                    state_mean=state_mean,
                    state_std=state_std,
                    mode=run_config.mode,
                )
            normalized = (
                float(env.get_normalized_score(float(achieved_return)) * 100.0)
                if hasattr(env, "get_normalized_score")
                else None
            )
            artifact_stem = f"rtg_{int(target)}_episode_{episode_index:03d}"
            debug_summary, debug_artifacts_written = _write_debug_summary(
                run_config,
                paths,
                policy,
                step=int(payload.get("step", 0)),
                phase="reeval_rtg_grid",
                split="rtg_grid",
                artifact_stem=artifact_stem,
                artifacts_written=debug_artifacts_written,
            ) if run_config.reeval_debug or _debug_enabled(run_config, "rtg_grid", int(payload.get("step", 0))) else (None, debug_artifacts_written)

            record = {
                "checkpoint_path": run_config.checkpoint_path,
                "selector": run_config.checkpoint_selector,
                "env_name": run_config.env_name,
                "seed": run_config.seed,
                "query_mode": run_config.attnres_query_mode,
                "num_layers": run_config.n_layer,
                "episode_index": episode_index,
                "rtg_target": float(target),
                "rtg_achieved": float(achieved_return),
                "rtg_alignment_error": float(abs(float(achieved_return) - float(target))),
                "rtg_alignment_signed_error": float(float(achieved_return) - float(target)),
                "eval_return_normalized": normalized,
                "eval_length": float(episode_length),
                "debug_summary": debug_summary,
            }
            append_jsonl(paths["logs"] / "rtg_grid_metrics.jsonl", record)

    if compute_tracker is not None:
        write_json(
            paths["analysis"] / "compute_summary.json",
            compute_tracker.finalize(
                total_wall_clock_sec=perf_counter() - reeval_start_time,
                reeval_wall_clock_sec=perf_counter() - reeval_start_time,
            ),
        )
    collect_run_summary(paths["root"], write_summary=True)
    if created_metadata:
        update_manifest(paths["root"], status="completed", finished_at=current_timestamp())
    print(json.dumps({"rtg_grid": rtg_grid, "output_dir": str(paths["root"])}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="VDT dev runner for Step 1 BAR, Step 2 VCDR, and Step 3 analysis infrastructure."
    )
    parser.add_argument("--config", type=str, default=None, help="YAML config path.")
    parser.add_argument(
        "--job",
        type=str,
        choices=("train", "eval", "reeval_rtg_grid"),
        default=None,
    )

    type_hints = get_type_hints(ExperimentConfig)
    for field in fields(ExperimentConfig):
        if field.name in {"config", "job"}:
            continue
        arg_name = f"--{field.name.replace('_', '-')}"
        resolved_type = type_hints[field.name]
        origin = get_origin(resolved_type)
        if origin is Union:
            args = [arg for arg in get_args(resolved_type) if arg is not type(None)]
            if len(args) == 1:
                resolved_type = args[0]
        if resolved_type == bool:
            parser.add_argument(arg_name, type=str2bool, default=None)
        else:
            parser.add_argument(arg_name, type=resolved_type, default=None)
    return parser


def load_config(parser: argparse.ArgumentParser) -> ExperimentConfig:
    args = parser.parse_args()
    config_dict = asdict(ExperimentConfig())
    if args.config is not None:
        with open(args.config, "r", encoding="utf-8") as handle:
            yaml_config = yaml.safe_load(handle) or {}
        config_dict.update(yaml_config)

    for key, value in vars(args).items():
        if value is not None:
            config_dict[key] = value
    return ExperimentConfig(**config_dict)


def main() -> None:
    parser = build_parser()
    config = load_config(parser)
    if config.job == "eval":
        evaluate(config)
    elif config.job == "reeval_rtg_grid":
        reeval_rtg_grid(config)
    else:
        train(config)


if __name__ == "__main__":
    main()
