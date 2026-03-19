from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from vdt_dev.experiments.budget import build_matched_budget_specs
from vdt_dev.experiments.matrix import RunSpec


CONFIG_VANILLA = "vdt_dev/configs/vdt_vanilla/hopper_medium_v2_vanilla_dev.yaml"
CONFIG_VANILLA_SANITY = "vdt_dev/configs/vdt_vanilla/hopper_medium_v2_vanilla_dev_sanity.yaml"
CONFIG_VANILLA_ONLINE = "vdt_dev/configs/vdt_vanilla/hopper_medium_v2_vanilla_dev_online.yaml"

CONFIG_BAR = "vdt_dev/configs/vdt_bar/hopper_medium_v2.yaml"
CONFIG_BAR_SANITY = "vdt_dev/configs/vdt_bar/hopper_medium_v2_sanity.yaml"
CONFIG_BAR_ONLINE = "vdt_dev/configs/vdt_bar/hopper_medium_v2_bar_online.yaml"

CONFIG_VCDR_STATE = "vdt_dev/configs/vdt_vcdr/hopper_medium_v2_vcdr_state.yaml"
CONFIG_VCDR_STATE_RTG = "vdt_dev/configs/vdt_vcdr/hopper_medium_v2_vcdr_state_rtg.yaml"
CONFIG_VCDR_STATE_RTG_VALUE = "vdt_dev/configs/vdt_vcdr/hopper_medium_v2_vcdr_state_rtg_value.yaml"
CONFIG_VCDR_STATE_RTG_VALUE_ONLINE = (
    "vdt_dev/configs/vdt_vcdr/hopper_medium_v2_vcdr_state_rtg_value_online.yaml"
)

DEFAULT_QUERY_MODE_BY_VARIANT = {
    "vanilla_dev": "static",
    "bar": "static",
    "vcdr": "state_rtg_value",
}

DEFAULT_CORE_VARIANTS = ("vanilla_dev", "bar", "vcdr")
DEFAULT_QUERY_SWEEP_MODES = ("static", "state", "state_rtg", "state_rtg_value")
SUPPORTED_GYM_ENVS = (
    "hopper-medium-v2",
    "walker2d-medium-v2",
    "halfcheetah-medium-v2",
)


@dataclass(frozen=True)
class PresetDefinition:
    name: str
    family: str
    description: str


@dataclass(frozen=True)
class ModelSpec:
    model_variant: str
    query_mode: str
    config_path: str
    online: bool = False


def available_presets() -> Dict[str, PresetDefinition]:
    return {
        "hopper_same_stack_compare": PresetDefinition(
            name="hopper_same_stack_compare",
            family="same_stack_compare",
            description=(
                "Hopper same-stack attribution compare over vanilla_dev, BAR static, "
                "and VCDR state_rtg_value at 6 layers."
            ),
        ),
        "hopper_querymode_sweep": PresetDefinition(
            name="hopper_querymode_sweep",
            family="query_mode_sweep",
            description="Hopper medium query-mode sweep over static/state/state_rtg/state_rtg_value at 6 layers.",
        ),
        "hopper_depth_sweep": PresetDefinition(
            name="hopper_depth_sweep",
            family="depth_sweep",
            description=(
                "Equal-width Hopper depth sweep over BAR static and VCDR state_rtg_value "
                "at 6/12/18 layers."
            ),
        ),
        "hopper_same_stack_depth_sweep": PresetDefinition(
            name="hopper_same_stack_depth_sweep",
            family="same_stack_depth_sweep",
            description=(
                "Equal-width Hopper depth sweep over vanilla_dev, BAR static, "
                "and VCDR state_rtg_value at 6/12/18 layers."
            ),
        ),
        "hopper_matched_budget_depth_sweep": PresetDefinition(
            name="hopper_matched_budget_depth_sweep",
            family="matched_budget_depth_sweep",
            description=(
                "Approximate matched-parameter Hopper depth sweep over vanilla_dev, BAR static, "
                "and VCDR state_rtg_value."
            ),
        ),
        "gym_env_sweep": PresetDefinition(
            name="gym_env_sweep",
            family="env_sweep",
            description="Gym medium env sweep over hopper/walker2d/halfcheetah for static and state_rtg_value.",
        ),
        "gym_same_stack_sweep": PresetDefinition(
            name="gym_same_stack_sweep",
            family="same_stack_env_sweep",
            description=(
                "Gym medium env sweep over hopper/walker2d/halfcheetah for vanilla_dev, "
                "BAR static, and VCDR state_rtg_value."
            ),
        ),
        "seed_sweep": PresetDefinition(
            name="seed_sweep",
            family="seed_sweep",
            description="Configurable seed sweep for a fixed env and model/query configuration.",
        ),
        "hopper_online_compare": PresetDefinition(
            name="hopper_online_compare",
            family="online_compare",
            description=(
                "Hopper offline-to-online compare over vanilla_dev, BAR static, "
                "and VCDR state_rtg_value."
            ),
        ),
        "maze2d_template": PresetDefinition(
            name="maze2d_template",
            family="future_sparse_reward",
            description="Sparse-reward template for future Maze2D experiments.",
        ),
        "antmaze_template": PresetDefinition(
            name="antmaze_template",
            family="future_sparse_reward",
            description="Sparse-reward template for future AntMaze experiments.",
        ),
    }


def _infer_variant_for_query_mode(query_mode: str) -> str:
    return "bar" if query_mode == "static" else "vcdr"


def _resolve_config_path(
    *,
    model_variant: str,
    query_mode: str,
    online: bool = False,
) -> str:
    if model_variant == "vanilla_dev":
        if query_mode != "static":
            raise ValueError("`vanilla_dev` only supports the static query path.")
        return CONFIG_VANILLA_ONLINE if online else CONFIG_VANILLA
    if model_variant == "bar":
        if query_mode != "static":
            raise ValueError("`bar` only supports the static query path.")
        return CONFIG_BAR_ONLINE if online else CONFIG_BAR
    if model_variant == "vcdr":
        if query_mode == "state":
            if online:
                raise ValueError("Online configs currently exist only for `vcdr` state_rtg_value.")
            return CONFIG_VCDR_STATE
        if query_mode == "state_rtg":
            if online:
                raise ValueError("Online configs currently exist only for `vcdr` state_rtg_value.")
            return CONFIG_VCDR_STATE_RTG
        if query_mode == "state_rtg_value":
            return CONFIG_VCDR_STATE_RTG_VALUE_ONLINE if online else CONFIG_VCDR_STATE_RTG_VALUE
        raise ValueError(f"Unsupported VCDR query mode `{query_mode}`.")
    raise ValueError(f"Unsupported model_variant `{model_variant}`.")


def _build_model_specs(
    *,
    model_variants: Sequence[str] | None = None,
    query_modes: Sequence[str] | None = None,
    online: bool = False,
) -> List[ModelSpec]:
    variants = list(model_variants or ())
    modes = list(query_modes or ())
    if not variants and not modes:
        variants = list(DEFAULT_CORE_VARIANTS)

    model_specs: list[ModelSpec] = []
    seen: set[tuple[str, str, bool]] = set()

    if variants and modes:
        candidate_pairs = ((variant, query_mode) for variant in variants for query_mode in modes)
    elif variants:
        candidate_pairs = ((variant, DEFAULT_QUERY_MODE_BY_VARIANT[variant]) for variant in variants)
    else:
        candidate_pairs = ((_infer_variant_for_query_mode(query_mode), query_mode) for query_mode in modes)

    for model_variant, query_mode in candidate_pairs:
        key = (model_variant, query_mode, online)
        if key in seen:
            continue
        try:
            config_path = _resolve_config_path(
                model_variant=model_variant,
                query_mode=query_mode,
                online=online,
            )
        except ValueError:
            continue
        seen.add(key)
        model_specs.append(
            ModelSpec(
                model_variant=model_variant,
                query_mode=query_mode,
                config_path=config_path,
                online=online,
            )
        )

    if not model_specs:
        raise ValueError("No runnable model/query combinations were produced for this preset request.")
    return model_specs


def _shared_overrides(
    *,
    preset_name: str,
    family: str,
    env_name: str,
    seed: int,
    model_variant: str,
    query_mode: str,
    num_layers: int,
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    overrides = {
        "exp_name": preset_name,
        "experiment_family": family,
        "preset_name": preset_name,
        "env_name": env_name,
        "seed": seed,
        "model_variant": model_variant,
        "attnres_query_mode": query_mode,
        "n_layer": num_layers,
        "structured_logging": True,
        "compute_report": True,
        "save_best_checkpoint": True,
        "resume_if_exists": True,
        "skip_if_complete": True,
    }
    if extra:
        overrides.update(extra)
    return overrides


def _run_name(
    env_name: str,
    model_variant: str,
    query_mode: str,
    num_layers: int,
    seed: int,
    *,
    tag: str | None = None,
) -> str:
    base = (
        env_name.replace("-", "_")
        + f"__{model_variant}__{query_mode}__L{num_layers}__seed{seed}"
    )
    if tag:
        return base + f"__{tag}"
    return base


def _build_run_spec(
    *,
    preset_name: str,
    family: str,
    output_root: str,
    env_name: str,
    model_variant: str,
    query_mode: str,
    config_path: str,
    num_layers: int,
    seed: int,
    extra_overrides: Dict[str, Any] | None = None,
    run_tag: str | None = None,
) -> RunSpec:
    name = _run_name(
        env_name,
        model_variant,
        query_mode,
        num_layers,
        seed,
        tag=run_tag,
    )
    output_dir = str(Path(output_root) / preset_name / name)
    return RunSpec(
        name=name,
        config_path=config_path,
        output_dir=output_dir,
        overrides=_shared_overrides(
            preset_name=preset_name,
            family=family,
            env_name=env_name,
            seed=seed,
            model_variant=model_variant,
            query_mode=query_mode,
            num_layers=num_layers,
            extra=extra_overrides,
        ),
    )


def _expand_runs(
    *,
    preset_name: str,
    family: str,
    output_root: str,
    envs: Sequence[str],
    seeds: Sequence[int],
    model_specs: Sequence[ModelSpec],
    depths: Sequence[int],
) -> List[RunSpec]:
    return [
        _build_run_spec(
            preset_name=preset_name,
            family=family,
            output_root=output_root,
            env_name=env_name,
            model_variant=model_spec.model_variant,
            query_mode=model_spec.query_mode,
            config_path=model_spec.config_path,
            num_layers=depth,
            seed=seed,
        )
        for env_name in envs
        for model_spec in model_specs
        for depth in depths
        for seed in seeds
    ]


def _expand_matched_budget_runs(
    *,
    preset_name: str,
    family: str,
    output_root: str,
    env_name: str,
    seeds: Sequence[int],
    model_variants: Sequence[str],
    depths: Sequence[int],
) -> List[RunSpec]:
    run_specs: list[RunSpec] = []
    budget_reference = "L6_E256_H4"
    for model_variant in model_variants:
        matched_specs = build_matched_budget_specs(
            env_name=env_name,
            model_variant=model_variant,
            num_layers=depths,
        )
        query_mode = DEFAULT_QUERY_MODE_BY_VARIANT[model_variant]
        config_path = _resolve_config_path(
            model_variant=model_variant,
            query_mode=query_mode,
            online=False,
        )
        for matched_spec in matched_specs:
            extra_overrides = {
                "embed_dim": matched_spec.embed_dim,
                "n_head": matched_spec.n_head,
                "budget_mode": "matched_params",
                "budget_reference": budget_reference,
                "budget_target_params": matched_spec.target_parameter_count,
                "budget_tolerance_pct": 10.0,
                "budget_actual_gap_pct": matched_spec.parameter_gap_pct,
            }
            tag = f"E{matched_spec.embed_dim}_H{matched_spec.n_head}"
            for seed in seeds:
                run_specs.append(
                    _build_run_spec(
                        preset_name=preset_name,
                        family=family,
                        output_root=output_root,
                        env_name=env_name,
                        model_variant=model_variant,
                        query_mode=query_mode,
                        config_path=config_path,
                        num_layers=matched_spec.num_layers,
                        seed=seed,
                        extra_overrides=extra_overrides,
                        run_tag=tag,
                    )
                )
    return run_specs


def _normalize_envs(env_name: str, *, use_all_gym_envs: bool) -> List[str]:
    if use_all_gym_envs and env_name == "hopper-medium-v2":
        return list(SUPPORTED_GYM_ENVS)
    return [env_name]


def build_preset_run_specs(
    preset_name: str,
    *,
    output_root: str = "vdt_dev/outputs",
    env_name: str = "hopper-medium-v2",
    seeds: Sequence[int] = (0,),
    query_modes: Sequence[str] | None = None,
    num_layers: Sequence[int] | None = None,
    model_variants: Sequence[str] | None = None,
) -> List[RunSpec]:
    preset = available_presets()[preset_name]

    if preset_name == "hopper_same_stack_compare":
        return _expand_runs(
            preset_name=preset_name,
            family=preset.family,
            output_root=output_root,
            envs=[env_name],
            seeds=seeds,
            model_specs=_build_model_specs(model_variants=model_variants or DEFAULT_CORE_VARIANTS),
            depths=list(num_layers or (6,)),
        )

    if preset_name == "hopper_querymode_sweep":
        return _expand_runs(
            preset_name=preset_name,
            family=preset.family,
            output_root=output_root,
            envs=[env_name],
            seeds=seeds,
            model_specs=_build_model_specs(query_modes=query_modes or DEFAULT_QUERY_SWEEP_MODES),
            depths=list(num_layers or (6,)),
        )

    if preset_name == "hopper_depth_sweep":
        return _expand_runs(
            preset_name=preset_name,
            family=preset.family,
            output_root=output_root,
            envs=[env_name],
            seeds=seeds,
            model_specs=_build_model_specs(query_modes=query_modes or ("static", "state_rtg_value")),
            depths=list(num_layers or (6, 12, 18)),
        )

    if preset_name == "hopper_same_stack_depth_sweep":
        return _expand_runs(
            preset_name=preset_name,
            family=preset.family,
            output_root=output_root,
            envs=[env_name],
            seeds=seeds,
            model_specs=_build_model_specs(model_variants=model_variants or DEFAULT_CORE_VARIANTS),
            depths=list(num_layers or (6, 12, 18)),
        )

    if preset_name == "hopper_matched_budget_depth_sweep":
        return _expand_matched_budget_runs(
            preset_name=preset_name,
            family=preset.family,
            output_root=output_root,
            env_name=env_name,
            seeds=seeds,
            model_variants=list(model_variants or DEFAULT_CORE_VARIANTS),
            depths=list(num_layers or (6, 12, 18)),
        )

    if preset_name == "gym_env_sweep":
        return _expand_runs(
            preset_name=preset_name,
            family=preset.family,
            output_root=output_root,
            envs=_normalize_envs(env_name, use_all_gym_envs=True),
            seeds=seeds,
            model_specs=_build_model_specs(query_modes=query_modes or ("static", "state_rtg_value")),
            depths=list(num_layers or (6,)),
        )

    if preset_name == "gym_same_stack_sweep":
        return _expand_runs(
            preset_name=preset_name,
            family=preset.family,
            output_root=output_root,
            envs=_normalize_envs(env_name, use_all_gym_envs=True),
            seeds=seeds,
            model_specs=_build_model_specs(model_variants=model_variants or DEFAULT_CORE_VARIANTS),
            depths=list(num_layers or (6,)),
        )

    if preset_name == "seed_sweep":
        model_specs = _build_model_specs(
            model_variants=model_variants,
            query_modes=query_modes or ("state_rtg_value",),
        )
        return _expand_runs(
            preset_name=preset_name,
            family=preset.family,
            output_root=output_root,
            envs=[env_name],
            seeds=seeds,
            model_specs=model_specs,
            depths=list(num_layers or (6,)),
        )

    if preset_name == "hopper_online_compare":
        return _expand_runs(
            preset_name=preset_name,
            family=preset.family,
            output_root=output_root,
            envs=[env_name],
            seeds=seeds,
            model_specs=_build_model_specs(
                model_variants=model_variants or DEFAULT_CORE_VARIANTS,
                online=True,
            ),
            depths=list(num_layers or (6,)),
        )

    if preset_name in {"maze2d_template", "antmaze_template"}:
        template_env = "maze2d-medium-v1" if preset_name == "maze2d_template" else "antmaze-umaze-v2"
        return [
            _build_run_spec(
                preset_name=preset_name,
                family=preset.family,
                output_root=output_root,
                env_name=template_env,
                model_variant="vcdr",
                query_mode="state_rtg_value",
                config_path=CONFIG_VCDR_STATE_RTG_VALUE,
                num_layers=6,
                seed=seed,
            )
            for seed in seeds
        ]
    raise KeyError(f"Unknown preset `{preset_name}`.")
