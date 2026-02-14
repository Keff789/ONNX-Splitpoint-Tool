"""Shared GUI state models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class GuiState:
    """Cross-panel UI state container."""

    current_model_path: Optional[str] = None
    model_type: str = "onnx"
    analysis_params: Dict[str, Any] = field(default_factory=dict)
    llm_params: Dict[str, Any] = field(default_factory=dict)
    hardware_selection: Dict[str, Any] = field(default_factory=dict)
    export_flags: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalysisResult:
    """Central analysis outputs consumed by table/plots/actions."""

    candidates: List[int] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    plot_data: Dict[str, Any] = field(default_factory=dict)
    memory_estimate: Optional[Dict[int, Dict[str, Any]]] = None


@dataclass
class SelectedCandidate:
    """Current table/plot candidate selection."""

    boundary_id: int
    semantic_label: str
    cut_tensors: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)


__all__ = ["GuiState", "AnalysisResult", "SelectedCandidate"]
