from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from vdt_dev.experiments.matrix import run_spec_to_command, write_planned_manifest
from vdt_dev.experiments.presets import available_presets, build_preset_run_specs


def list_preset_names() -> List[str]:
    return sorted(available_presets())


def get_preset(name: str):
    presets = available_presets()
    if name not in presets:
        raise KeyError(f"Unknown preset `{name}`. Available presets: {sorted(presets)}")
    return presets[name]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect or expand Step 4 experiment presets.")
    parser.add_argument("--preset", type=str, required=True, choices=list_preset_names())
    parser.add_argument("--output-root", type=str, default="vdt_dev/outputs")
    parser.add_argument("--env-name", type=str, default="hopper-medium-v2")
    parser.add_argument("--seeds", type=str, default="0")
    parser.add_argument("--model-variants", type=str, default=None)
    parser.add_argument("--query-modes", type=str, default=None)
    parser.add_argument("--num-layers", type=str, default=None)
    parser.add_argument("--write-planned-manifests", action="store_true")
    parser.add_argument("--format", type=str, choices=("commands", "json"), default="commands")
    return parser


def _parse_int_csv(value: str | None) -> tuple[int, ...] | None:
    if value is None or value == "":
        return None
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _parse_str_csv(value: str | None) -> tuple[str, ...] | None:
    if value is None or value == "":
        return None
    return tuple(item.strip() for item in value.split(",") if item.strip())


def main() -> None:
    args = build_parser().parse_args()
    run_specs = build_preset_run_specs(
        args.preset,
        output_root=args.output_root,
        env_name=args.env_name,
        seeds=_parse_int_csv(args.seeds) or (0,),
        model_variants=_parse_str_csv(args.model_variants),
        query_modes=_parse_str_csv(args.query_modes),
        num_layers=_parse_int_csv(args.num_layers),
    )

    if args.write_planned_manifests:
        for run_spec in run_specs:
            write_planned_manifest(run_spec)

    if args.format == "json":
        print(
            json.dumps(
                [
                    {
                        "name": run_spec.name,
                        "config_path": run_spec.config_path,
                        "output_dir": run_spec.output_dir,
                        "overrides": run_spec.overrides,
                        "command": run_spec_to_command(run_spec),
                    }
                    for run_spec in run_specs
                ],
                indent=2,
                sort_keys=True,
            )
        )
    else:
        for run_spec in run_specs:
            print(run_spec_to_command(run_spec))


if __name__ == "__main__":
    main()
