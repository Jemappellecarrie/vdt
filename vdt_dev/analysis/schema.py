from __future__ import annotations

import json
from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Type, TypeVar, Union, get_args, get_origin, get_type_hints


JSONDict = Dict[str, Any]
T = TypeVar("T")
SCHEMA_VERSION = "step4.v1"


@dataclass
class ArtifactPaths:
    checkpoint_path: Optional[str] = None
    best_checkpoint_path: Optional[str] = None
    latest_checkpoint_path: Optional[str] = None
    metrics_jsonl_path: Optional[str] = None
    eval_metrics_jsonl_path: Optional[str] = None
    debug_metrics_jsonl_path: Optional[str] = None
    rtg_grid_jsonl_path: Optional[str] = None
    debug_artifact_path: Optional[str] = None
    plot_output_path: Optional[str] = None
    aggregated_csv_path: Optional[str] = None
    compute_summary_path: Optional[str] = None


@dataclass
class RunManifest:
    schema_version: str = SCHEMA_VERSION
    run_id: str = ""
    exp_name: str = ""
    experiment_family: str = ""
    preset_name: str = ""
    env_name: str = ""
    seed: int = 0
    model_variant: str = ""
    query_mode: str = "static"
    attnres_variant: str = "block"
    num_layers: int = 0
    hidden_size: int = 0
    num_heads: int = 0
    context_len: int = 0
    use_pre_attn: bool = True
    use_pre_mlp: bool = True
    num_blocks: int = 0
    train_steps: int = 0
    eval_interval: int = 0
    git_hash: str = ""
    config_path: str = ""
    output_dir: str = ""
    job: str = "train"
    status: str = "created"
    primary_metric_name: str = "eval_return_normalized"
    primary_metric_mode: str = "max"
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    resume_count: int = 0
    command: str = ""
    artifacts: ArtifactPaths = field(default_factory=ArtifactPaths)
    notes: JSONDict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.notes is None:
            self.notes = {}


@dataclass
class MetricRecord:
    step: int
    phase: str
    split: str
    epoch: Optional[float] = None
    train_actor_loss: Optional[float] = None
    train_q_loss: Optional[float] = None
    train_v_loss: Optional[float] = None
    eval_return_raw: Optional[float] = None
    eval_return_normalized: Optional[float] = None
    eval_length: Optional[float] = None
    best_eval_return_normalized: Optional[float] = None
    checkpoint_path: Optional[str] = None
    rtg_target: Optional[float] = None
    rtg_achieved: Optional[float] = None
    rtg_alignment_error: Optional[float] = None
    routing_entropy_mean: Optional[float] = None
    routing_entropy_by_layer: Optional[List[float]] = None
    static_query_norm_mean: Optional[float] = None
    delta_query_norm_mean: Optional[float] = None
    fused_query_norm_mean: Optional[float] = None
    hidden_norm_by_layer: Optional[List[float]] = None
    output_norm_by_layer: Optional[List[float]] = None
    source_usage_by_layer: Optional[List[JSONDict]] = None
    source_distance_mean: Optional[float] = None
    extras: JSONDict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.extras is None:
            self.extras = {}


@dataclass
class DebugSummaryRecord:
    step: int
    phase: str
    split: str
    routing_entropy_mean: Optional[float] = None
    routing_entropy_by_layer: Optional[List[float]] = None
    static_query_norm_mean: Optional[float] = None
    delta_query_norm_mean: Optional[float] = None
    fused_query_norm_mean: Optional[float] = None
    hidden_norm_by_layer: Optional[List[float]] = None
    output_norm_by_layer: Optional[List[float]] = None
    source_usage_by_layer: Optional[List[JSONDict]] = None
    source_distance_mean: Optional[float] = None
    source_distance_by_layer: Optional[List[float]] = None
    artifact_path: Optional[str] = None
    extras: JSONDict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.extras is None:
            self.extras = {}


@dataclass
class RTGGridRecord:
    checkpoint_path: str
    selector: str
    env_name: str
    seed: int
    query_mode: str
    num_layers: int
    episode_index: int
    rtg_target: float
    rtg_achieved: float
    rtg_alignment_error: float
    eval_return_normalized: Optional[float] = None
    eval_length: Optional[float] = None
    debug_summary_path: Optional[str] = None
    extras: JSONDict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.extras is None:
            self.extras = {}


def _json_default(value: Any) -> Any:
    if is_dataclass(value):
        return dataclass_to_dict(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable.")


def _coerce_optional_type(value: Any, annotation: Any) -> Any:
    origin = get_origin(annotation)
    if origin is Union:
        non_none = [item for item in get_args(annotation) if item is not type(None)]
        if len(non_none) == 1:
            return _coerce_optional_type(value, non_none[0])
    if value is None:
        return None
    if annotation in {Any, object}:
        return value
    if origin in {list, List, Sequence, Iterable}:
        item_type = get_args(annotation)[0] if get_args(annotation) else Any
        return [_coerce_optional_type(item, item_type) for item in value]
    if origin in {dict, Dict}:
        key_type, value_type = get_args(annotation) if get_args(annotation) else (Any, Any)
        return {
            _coerce_optional_type(key, key_type): _coerce_optional_type(item, value_type)
            for key, item in value.items()
        }
    if hasattr(annotation, "__dataclass_fields__"):
        return dataclass_from_dict(annotation, value)
    if annotation is Path:
        return Path(value)
    return value


def dataclass_to_dict(value: Any) -> Any:
    if is_dataclass(value):
        output: JSONDict = {}
        for field_info in fields(value):
            output[field_info.name] = dataclass_to_dict(getattr(value, field_info.name))
        return output
    if isinstance(value, dict):
        return {key: dataclass_to_dict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [dataclass_to_dict(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def dataclass_from_dict(dataclass_type: Type[T], payload: Optional[JSONDict]) -> T:
    if payload is None:
        payload = {}
    type_hints = get_type_hints(dataclass_type)
    kwargs: JSONDict = {}
    for field_info in fields(dataclass_type):
        annotation = type_hints.get(field_info.name, field_info.type)
        if field_info.name in payload:
            kwargs[field_info.name] = _coerce_optional_type(
                payload[field_info.name],
                annotation,
            )
        elif field_info.default is not MISSING:
            kwargs[field_info.name] = field_info.default
        elif field_info.default_factory is not MISSING:  # type: ignore[attr-defined]
            kwargs[field_info.name] = field_info.default_factory()  # type: ignore[misc]
    return dataclass_type(**kwargs)


def ensure_parent(path: Path | str) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def read_json(path: Path | str, default: Optional[Any] = None) -> Any:
    input_path = Path(path)
    if not input_path.exists():
        return default
    with input_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path | str, payload: Any, *, indent: int = 2) -> Path:
    output_path = ensure_parent(path)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=indent, sort_keys=True, default=_json_default)
        handle.write("\n")
    return output_path


def read_jsonl(path: Path | str) -> List[JSONDict]:
    input_path = Path(path)
    if not input_path.exists():
        return []
    records: List[JSONDict] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def append_jsonl(path: Path | str, payload: Any) -> Path:
    output_path = ensure_parent(path)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=_json_default) + "\n")
    return output_path


def write_dataclass_json(path: Path | str, instance: Any) -> Path:
    return write_json(path, dataclass_to_dict(instance))


def load_dataclass_json(path: Path | str, dataclass_type: Type[T], *, default: Optional[T] = None) -> T:
    payload = read_json(path)
    if payload is None:
        if default is None:
            return dataclass_type()  # type: ignore[call-arg]
        return default
    return dataclass_from_dict(dataclass_type, payload)
