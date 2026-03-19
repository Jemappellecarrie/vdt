from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch


class RoutingDebugHook:
    """
    Collects lightweight routing diagnostics for later analysis.

    The hook is intentionally inert when disabled so the Step 1 path stays fast by
    default. When enabled, it stores CPU tensors that can be written with ``torch.save``.
    """

    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled
        self.clear()

    def clear(self) -> None:
        self.records: Dict[str, Dict[str, Any]] = {
            "routing_weights": {},
            "routing_entropy": {},
            "hidden_norm": {},
            "output_norm": {},
            "dynamic_query_delta_norm": {},
            "static_query_norm": {},
            "fused_query_norm": {},
            "query_mode": {},
            "mode_summaries": {},
            "source_names": {},
            "events": [],
        }

    def record(
        self,
        name: str,
        *,
        weights: torch.Tensor,
        entropy: torch.Tensor,
        hidden: torch.Tensor,
        output: torch.Tensor,
        source_names: tuple[str, ...],
        dynamic_query_delta: torch.Tensor | None = None,
        static_query: torch.Tensor | None = None,
        fused_query: torch.Tensor | None = None,
        query_mode: str | None = None,
        mode_summary: Dict[str, torch.Tensor] | None = None,
    ) -> None:
        if not self.enabled:
            return

        hidden_norm = hidden.detach().norm(dim=-1).cpu()
        self.records["routing_weights"][name] = weights.detach().cpu()
        self.records["routing_entropy"][name] = entropy.detach().cpu()
        self.records["hidden_norm"][name] = hidden_norm
        self.records["output_norm"][name] = output.detach().norm(dim=-1).cpu()
        if dynamic_query_delta is None:
            self.records["dynamic_query_delta_norm"][name] = torch.zeros_like(hidden_norm)
        else:
            self.records["dynamic_query_delta_norm"][name] = (
                dynamic_query_delta.detach().norm(dim=-1).cpu()
            )
        if static_query is None:
            self.records["static_query_norm"][name] = torch.tensor(0.0)
        else:
            self.records["static_query_norm"][name] = static_query.detach().norm().cpu()
        if fused_query is None:
            self.records["fused_query_norm"][name] = torch.zeros_like(hidden_norm)
        else:
            self.records["fused_query_norm"][name] = fused_query.detach().norm(dim=-1).cpu()
        if query_mode is not None:
            self.records["query_mode"][name] = query_mode
        if mode_summary is not None:
            self.records["mode_summaries"][name] = {
                key: value.detach().cpu() for key, value in mode_summary.items()
            }
        self.records["source_names"][name] = source_names
        self.records["events"].append(
            {
                "name": name,
                "source_names": list(source_names),
                "weights_mean": weights.detach().float().mean(dim=(0, 1)).cpu().tolist(),
                "entropy_mean": float(entropy.detach().float().mean().cpu().item()),
                "hidden_norm_mean": float(hidden_norm.detach().float().mean().cpu().item()),
                "output_norm_mean": float(output.detach().norm(dim=-1).detach().float().mean().cpu().item()),
                "dynamic_query_delta_norm_mean": float(
                    0.0
                    if dynamic_query_delta is None
                    else dynamic_query_delta.detach().norm(dim=-1).float().mean().cpu().item()
                ),
                "static_query_norm_mean": float(
                    0.0
                    if static_query is None
                    else static_query.detach().float().norm().cpu().item()
                ),
                "fused_query_norm_mean": float(
                    0.0
                    if fused_query is None
                    else fused_query.detach().norm(dim=-1).float().mean().cpu().item()
                ),
                "query_mode": query_mode,
                "mode_summary": {
                    key: float(value.detach().float().mean().cpu().item())
                    for key, value in (mode_summary or {}).items()
                },
            }
        )

    def state_dict(self) -> Dict[str, Dict[str, Any]]:
        return self.records

    def save(self, path: str | Path) -> None:
        if not self.enabled:
            return
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.records, output_path)
