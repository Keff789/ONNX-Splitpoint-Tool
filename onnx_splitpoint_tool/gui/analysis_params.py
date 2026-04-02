"""Central schema for analysis-related GUI parameters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Literal, Optional, Sequence

from ..core_params import mapping_for_key


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
    validation: str = ""
    deprecated: bool = False
    deprecated_note: str = ""


ANALYSIS_PARAM_SPECS: tuple[AnalysisParamSpec, ...] = (
    AnalysisParamSpec("topk", "Top-k", "int", "10", "candidate", scope="analysis", var_name="var_topk"),
    AnalysisParamSpec("min_gap", "Min gap", "int", "2", "candidate", scope="analysis", var_name="var_min_gap"),
    AnalysisParamSpec("min_compute_pct", "Min compute each side (%)", "float", "1", "candidate", scope="analysis", var_name="var_min_compute"),
    AnalysisParamSpec("batch_override", "Batch override", "int", "", "candidate", scope="analysis", var_name="var_batch", advanced=True),
    AnalysisParamSpec("assume_bpe", "Assume act bytes/elt", "int", "", "candidate", scope="analysis", var_name="var_bpe", advanced=True),
    AnalysisParamSpec("unknown_tensor_proxy_mb", "Unknown MB/tensor", "float", "2.0", "shape", scope="analysis", var_name="var_unknown_mb"),
    AnalysisParamSpec(
        "exclude_trivial",
        "Exclude trivial ops",
        "bool",
        True,
        "candidate",
        scope="analysis",
        var_name="var_exclude_trivial",
        tooltip="Drop boundaries next to reshape/transpose-like utility ops.",
    ),
    AnalysisParamSpec(
        "only_single_tensor",
        "Only one crossing tensor",
        "bool",
        False,
        "candidate",
        scope="analysis",
        var_name="var_only_one",
        tooltip="Keep only boundaries with exactly one crossing activation tensor.",
    ),
    AnalysisParamSpec(
        "strict_boundary",
        "Strict boundary",
        "bool",
        True,
        "shape",
        scope="analysis",
        var_name="var_strict_boundary",
        tooltip="Require Part2 dependencies to be limited to allowed split inputs.",
    ),
    AnalysisParamSpec(
        "show_top_tensors",
        "Show top tensors",
        "int",
        "3",
        "candidate",
        scope="analysis",
        var_name="var_show_top_tensors",
        tooltip="Show largest crossing tensors as child rows per boundary (0 disables).",
    ),
    AnalysisParamSpec(
        "prune_skip_block",
        "Skip/Block pruning",
        "bool",
        True,
        "candidate",
        scope="analysis",
        var_name="var_prune_skip_block",
        tooltip="Avoid candidates inside detected residual/skip blocks.",
    ),
    AnalysisParamSpec(
        "skip_min_span",
        "Min skip span (ops)",
        "int",
        "8",
        "candidate",
        scope="analysis",
        var_name="var_skip_min_span",
        tooltip="Minimum skip span in ops to treat a path as skip/residual.",
    ),
    AnalysisParamSpec(
        "skip_allow_last_n",
        "Allow last N inside",
        "int",
        "0",
        "candidate",
        scope="analysis",
        var_name="var_skip_allow_last_n",
        tooltip="Allow splits in the last N ops before skip merge.",
    ),
    AnalysisParamSpec(
        "cluster_best_per_region",
        "Best per region",
        "bool",
        True,
        "candidate",
        scope="analysis",
        var_name="var_cluster_best_region",
        tooltip="Keep only best-scoring candidate per region/bin.",
    ),
    AnalysisParamSpec(
        "cluster_region_ops",
        "Region (ops)",
        "str",
        "auto",
        "candidate",
        scope="analysis",
        var_name="var_cluster_region_ops",
        tooltip="Region size in ops for clustering; use 'auto' for heuristic.",
    ),
    AnalysisParamSpec(
        "cluster_mode",
        "Cluster mode",
        "choice",
        "Auto",
        "candidate",
        scope="analysis",
        var_name="var_cluster_mode",
        options=("Auto", "Uniform", "Semantic (LLM)"),
        tooltip="Candidate clustering strategy: Auto, Uniform, or Semantic.",
    ),
    AnalysisParamSpec("rank", "Ranking", "choice", "score", "scoring", scope="analysis", var_name="var_rank", options=("cut", "score", "latency")),
    AnalysisParamSpec("log_comm", "log10(1+comm)", "bool", True, "scoring", scope="analysis", var_name="var_log_comm", tooltip="Use log-scaled communication term when computing score rank.", advanced=True),
    AnalysisParamSpec("w_comm", "w_comm", "float", "1.0", "scoring", scope="analysis", var_name="var_w_comm", tooltip="Weight of communication term in score ranking.", advanced=True),
    AnalysisParamSpec("w_imb", "w_imb", "float", "3.0", "scoring", scope="analysis", var_name="var_w_imb", tooltip="Weight of imbalance term in score ranking.", advanced=True),
    AnalysisParamSpec("w_tensors", "w_tensors", "float", "0.2", "scoring", scope="analysis", var_name="var_w_tensors", tooltip="Weight of crossing-tensor-count term in score ranking.", advanced=True),
    AnalysisParamSpec("show_pareto_front", "Show Pareto front", "bool", True, "scoring", scope="analysis", var_name="var_show_pareto", tooltip="Overlay Pareto front in comm-vs-imbalance plot.", advanced=True),
    AnalysisParamSpec("enable", "Enable LLM presets", "bool", False, "llm", scope="llm", var_name="var_llm_enable", visibility="llm"),
    AnalysisParamSpec("preset", "LLM preset", "choice", "Standard", "llm", scope="llm", var_name="var_llm_preset", visibility="llm", options=("Standard", "Latency Critical (Chat)", "Throughput/RAG", "Custom")),
    AnalysisParamSpec("mode", "LLM mode", "choice", "decode", "llm", scope="llm", var_name="var_llm_mode", visibility="llm", options=("decode", "prefill")),
    AnalysisParamSpec("prefill", "Prefill length", "int", "512", "llm", scope="llm", var_name="var_llm_prefill", visibility="llm"),
    AnalysisParamSpec("decode", "Decode past length", "int", "2048", "llm", scope="llm", var_name="var_llm_decode", visibility="llm"),
    AnalysisParamSpec("use_ort_symbolic", "Use ORT symbolic inference", "bool", True, "llm", scope="llm", var_name="var_llm_use_ort_symbolic", visibility="llm", advanced=True),
)


def _enrich_from_core_mapping(spec: AnalysisParamSpec) -> AnalysisParamSpec:
    mapping = mapping_for_key(spec.key)
    if mapping is None:
        return AnalysisParamSpec(
            key=spec.key,
            label=spec.label,
            param_type=spec.param_type,
            default=spec.default,
            section=spec.section,
            tooltip=spec.tooltip,
            visibility=spec.visibility,
            advanced=spec.advanced,
            scope=spec.scope,
            var_name=spec.var_name,
            options=spec.options,
            validation="not mapped to core Params (deprecated)",
            deprecated=True,
            deprecated_note="Legacy option: not part of core Params mapping",
        )

    return AnalysisParamSpec(
        key=spec.key,
        label=spec.label,
        param_type=spec.param_type,
        default=spec.default,
        section=spec.section,
        tooltip=spec.tooltip,
        visibility=mapping.visibility,
        advanced=spec.advanced,
        scope=spec.scope,
        var_name=spec.var_name,
        options=spec.options,
        validation=mapping.validation,
        deprecated=bool(mapping.deprecated),
        deprecated_note=str(mapping.deprecated_note or ""),
    )


ANALYSIS_PARAM_SPECS = tuple(_enrich_from_core_mapping(spec) for spec in ANALYSIS_PARAM_SPECS)


def iter_specs(scope: Scope | None = None) -> Iterable[AnalysisParamSpec]:
    for spec in ANALYSIS_PARAM_SPECS:
        if scope is None or spec.scope == scope:
            yield spec
