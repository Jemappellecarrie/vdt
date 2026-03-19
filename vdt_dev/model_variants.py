from __future__ import annotations


MODEL_VARIANTS = ("vanilla_dev", "bar", "vcdr")


def resolve_model_variant(
    *,
    explicit: str | None = None,
    use_attnres: bool,
    query_mode: str,
) -> str:
    if explicit:
        normalized = explicit.strip()
        if normalized not in MODEL_VARIANTS:
            raise ValueError(
                f"Unsupported model_variant `{explicit}`. Expected one of {MODEL_VARIANTS}."
            )
        return normalized
    if not use_attnres:
        return "vanilla_dev"
    if query_mode == "static":
        return "bar"
    return "vcdr"
