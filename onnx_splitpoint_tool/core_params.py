"""Core parameter schema and GUI->Params mapping metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional, Sequence

Visibility = Literal["cv", "llm", "both"]


@dataclass(frozen=True)
class Params:
    topk: int
    min_gap: int
    min_compute_pct: float

    batch_override: Optional[int]
    llm_enable: bool
    llm_preset: str
    llm_mode: str
    llm_prefill_len: int
    llm_decode_past_len: int
    llm_use_ort_symbolic: bool
    assume_bpe: Optional[int]
    unknown_tensor_proxy_mb: float

    cluster_best_per_region: bool
    cluster_mode: str
    cluster_region_ops: int
    exclude_trivial: bool
    only_single_tensor: bool
    strict_boundary: bool

    prune_skip_block: bool
    skip_min_span: int
    skip_allow_last_n: int

    ranking: str
    log_comm: bool
    w_comm: float
    w_imb: float
    w_tensors: float
    show_pareto_front: bool

    link_model: str
    bw_value: Optional[float]
    bw_unit: str
    gops_left: Optional[float]
    gops_right: Optional[float]
    overhead_ms: float

    link_energy_pj_per_byte: Optional[float]
    link_mtu_payload_bytes: Optional[int]
    link_per_packet_overhead_ms: Optional[float]
    link_per_packet_overhead_bytes: Optional[int]

    energy_pj_per_flop_left: Optional[float]
    energy_pj_per_flop_right: Optional[float]

    link_max_latency_ms: Optional[float]
    link_max_energy_mJ: Optional[float]
    link_max_bytes: Optional[int]

    max_peak_act_left: Optional[float]
    max_peak_act_left_unit: str
    max_peak_act_right: Optional[float]
    max_peak_act_right_unit: str

    hailo_check: bool
    hailo_hw_arch: str
    hailo_max_checks: int
    hailo_fixup: bool
    hailo_keep_artifacts: bool
    hailo_target: str
    hailo_backend: str
    hailo_wsl_distro: Optional[str]
    hailo_wsl_venv_activate: str
    hailo_wsl_timeout_s: int

    show_top_tensors: int


@dataclass(frozen=True)
class GuiParamMapping:
    gui_field: str
    params_key: str
    default: Any
    validation: str
    visibility: Visibility
    deprecated: bool = False
    deprecated_note: str = ""


ANALYSIS_GUI_PARAM_MAP: tuple[GuiParamMapping, ...] = (
    GuiParamMapping("var_topk", "topk", "10", "int > 0", "both"),
    GuiParamMapping("var_min_gap", "min_gap", "2", "int >= 0", "both"),
    GuiParamMapping("var_min_compute", "min_compute_pct", "1", "float >= 0", "both"),
    GuiParamMapping("var_batch", "batch_override", "", "optional int", "both"),
    GuiParamMapping("var_bpe", "assume_bpe", "", "optional int", "both"),
    GuiParamMapping("var_unknown_mb", "unknown_tensor_proxy_mb", "2.0", "float >= 0", "both"),
    GuiParamMapping("var_exclude_trivial", "exclude_trivial", True, "bool", "both"),
    GuiParamMapping("var_only_one", "only_single_tensor", False, "bool", "both"),
    GuiParamMapping("var_strict_boundary", "strict_boundary", True, "bool", "both"),
    GuiParamMapping("var_rank", "rank", "score", "choice: cut|score|latency", "both"),
    GuiParamMapping("var_llm_enable", "enable", False, "bool", "llm"),
    GuiParamMapping("var_llm_preset", "preset", "Standard", "choice", "llm"),
    GuiParamMapping("var_llm_mode", "mode", "decode", "choice: decode|prefill", "llm"),
    GuiParamMapping("var_llm_prefill", "prefill", "512", "int > 0", "llm"),
    GuiParamMapping("var_llm_decode", "decode", "2048", "int >= 0", "llm"),
    GuiParamMapping("var_llm_use_ort_symbolic", "use_ort_symbolic", True, "bool", "llm"),
)


def mapping_for_key(param_key: str) -> Optional[GuiParamMapping]:
    for item in ANALYSIS_GUI_PARAM_MAP:
        if item.params_key == param_key:
            return item
    return None


def mapped_param_keys() -> Sequence[str]:
    return tuple(item.params_key for item in ANALYSIS_GUI_PARAM_MAP if not item.deprecated)
