"""Central schema for analysis-related GUI parameters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Literal, Optional, Sequence


Visibility = Literal["cv", "llm", "both"]
Scope = Literal["analysis", "llm"]


@dataclass(frozen=True)
class AnalysisParamSpec:
    key: str
    label: str
    param_type: Literal["int", "float", "str", "bool", "choice"]
    default: Any
    section: Literal["candidate", "scoring", "shape", "llm"]
    tooltip: str = ""
    visibility: Visibility = "both"
    advanced: bool = False
    scope: Scope = "analysis"
    var_name: str = ""
    options: Optional[Sequence[str]] = None


ANALYSIS_PARAM_SPECS: tuple[AnalysisParamSpec, ...] = (
    AnalysisParamSpec("topk", "Top-k", "int", "10", "candidate", scope="analysis", var_name="var_topk"),
    AnalysisParamSpec("min_gap", "Min gap", "int", "2", "candidate", scope="analysis", var_name="var_min_gap"),
    AnalysisParamSpec("min_compute_pct", "Min compute each side (%)", "float", "1", "candidate", scope="analysis", var_name="var_min_compute"),
    AnalysisParamSpec("batch_override", "Batch override", "int", "", "candidate", scope="analysis", var_name="var_batch", advanced=True),
    AnalysisParamSpec("assume_bpe", "Assume act bytes/elt", "int", "", "candidate", scope="analysis", var_name="var_bpe", advanced=True),
    AnalysisParamSpec("unknown_tensor_proxy_mb", "Unknown MB/tensor", "float", "2.0", "shape", scope="analysis", var_name="var_unknown_mb"),
    AnalysisParamSpec("exclude_trivial", "Exclude trivial ops", "bool", True, "candidate", scope="analysis", var_name="var_exclude_trivial"),
    AnalysisParamSpec("only_single_tensor", "Only one crossing tensor", "bool", False, "candidate", scope="analysis", var_name="var_only_one"),
    AnalysisParamSpec("strict_boundary", "Strict boundary", "bool", True, "shape", scope="analysis", var_name="var_strict_boundary"),
    AnalysisParamSpec("rank", "Ranking", "choice", "score", "scoring", scope="analysis", var_name="var_rank", options=("cut", "score", "latency")),
    AnalysisParamSpec("enable", "Enable LLM presets", "bool", False, "llm", scope="llm", var_name="var_llm_enable", visibility="llm"),
    AnalysisParamSpec("preset", "LLM preset", "choice", "Standard", "llm", scope="llm", var_name="var_llm_preset", visibility="llm", options=("Standard", "Latency Critical (Chat)", "Throughput/RAG", "Custom")),
    AnalysisParamSpec("mode", "LLM mode", "choice", "decode", "llm", scope="llm", var_name="var_llm_mode", visibility="llm", options=("decode", "prefill")),
    AnalysisParamSpec("prefill", "Prefill length", "int", "512", "llm", scope="llm", var_name="var_llm_prefill", visibility="llm"),
    AnalysisParamSpec("decode", "Decode past length", "int", "2048", "llm", scope="llm", var_name="var_llm_decode", visibility="llm"),
    AnalysisParamSpec("use_ort_symbolic", "Use ORT symbolic inference", "bool", True, "llm", scope="llm", var_name="var_llm_use_ort_symbolic", visibility="llm", advanced=True),
)


def iter_specs(scope: Scope | None = None) -> Iterable[AnalysisParamSpec]:
    for spec in ANALYSIS_PARAM_SPECS:
        if scope is None or spec.scope == scope:
            yield spec
