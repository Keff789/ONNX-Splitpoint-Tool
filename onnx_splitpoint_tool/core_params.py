"""Core parameter schema and GUI->Params mapping metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Sequence

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
    cluster_region_ops: Optional[int]
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
    GuiParamMapping("var_show_top_tensors", "show_top_tensors", "3", "int >= 0", "both"),
    GuiParamMapping("var_prune_skip_block", "prune_skip_block", True, "bool", "both"),
    GuiParamMapping("var_skip_min_span", "skip_min_span", "8", "int >= 0", "both"),
    GuiParamMapping("var_skip_allow_last_n", "skip_allow_last_n", "0", "int >= 0", "both"),
    GuiParamMapping("var_cluster_best_region", "cluster_best_per_region", True, "bool", "both"),
    GuiParamMapping("var_cluster_region_ops", "cluster_region_ops", "auto", "int >= 0 or auto", "both"),
    GuiParamMapping("var_cluster_mode", "cluster_mode", "Auto", "choice: auto|uniform|semantic", "both"),
    GuiParamMapping("var_rank", "rank", "score", "choice: cut|score|latency", "both"),
    GuiParamMapping("var_log_comm", "log_comm", True, "bool", "both"),
    GuiParamMapping("var_w_comm", "w_comm", "1.0", "float", "both"),
    GuiParamMapping("var_w_imb", "w_imb", "3.0", "float", "both"),
    GuiParamMapping("var_w_tensors", "w_tensors", "0.2", "float", "both"),
    GuiParamMapping("var_show_pareto", "show_pareto_front", True, "bool", "both"),
    GuiParamMapping("var_llm_enable", "enable", False, "bool", "llm"),
    GuiParamMapping("var_llm_preset", "preset", "Standard", "choice", "llm"),
    GuiParamMapping("var_llm_mode", "mode", "decode", "choice: decode|prefill", "llm"),
    GuiParamMapping("var_llm_prefill", "prefill", "512", "int > 0", "llm"),
    GuiParamMapping("var_llm_decode", "decode", "2048", "int >= 0", "llm"),
    GuiParamMapping("var_llm_use_ort_symbolic", "use_ort_symbolic", True, "bool", "llm"),
)

# Canonical key names expected by Params.
_PARAM_KEY_ALIASES: Dict[str, str] = {
    "rank": "ranking",
    "enable": "llm_enable",
    "preset": "llm_preset",
    "mode": "llm_mode",
    "prefill": "llm_prefill_len",
    "decode": "llm_decode_past_len",
    "use_ort_symbolic": "llm_use_ort_symbolic",
}

REQUIRED_MINIMUM_CV_KEYS: frozenset[str] = frozenset(
    {
        "topk",
        "min_gap",
        "min_compute_pct",
        "ranking",
        "unknown_tensor_proxy_mb",
        "cluster_mode",
        "cluster_region_ops",
        "w_comm",
        "w_imb",
        "w_tensors",
    }
)

REQUIRED_MINIMUM_LLM_KEYS: frozenset[str] = frozenset(
    {
        "llm_enable",
        "llm_preset",
        "llm_mode",
        "llm_prefill_len",
        "llm_decode_past_len",
        "llm_use_ort_symbolic",
    }
)


def mapping_for_key(param_key: str) -> Optional[GuiParamMapping]:
    for item in ANALYSIS_GUI_PARAM_MAP:
        if item.params_key == param_key:
            return item
    return None


def mapped_param_keys() -> Sequence[str]:
    return tuple(item.params_key for item in ANALYSIS_GUI_PARAM_MAP if not item.deprecated)


def exposed_param_keys() -> frozenset[str]:
    keys = set()
    for item in ANALYSIS_GUI_PARAM_MAP:
        if item.deprecated:
            continue
        canonical = _PARAM_KEY_ALIASES.get(item.params_key, item.params_key)
        keys.add(canonical)
    return frozenset(keys)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value or "").strip().lower()
    return text in {"1", "true", "yes", "on"}


def _coerce_by_validation(value: Any, validation: str) -> Any:
    rule = str(validation or "").lower()
    if "bool" in rule:
        return _as_bool(value)
    # Keep scalar entries as strings for downstream numeric parsing/validation.
    return value


def gui_state_to_params_dict(analysis_params: Dict[str, Any], llm_params: Dict[str, Any]) -> Dict[str, Any]:
    """Single source of truth for schema-mapped GUI state -> canonical param values."""

    out: Dict[str, Any] = {}
    ap = dict(analysis_params or {})
    llm = dict(llm_params or {})

    for mapping in ANALYSIS_GUI_PARAM_MAP:
        if mapping.deprecated:
            continue
        source = llm if mapping.visibility == "llm" else ap
        value = source.get(mapping.params_key, mapping.default)
        canonical_key = _PARAM_KEY_ALIASES.get(mapping.params_key, mapping.params_key)
        out[canonical_key] = _coerce_by_validation(value, mapping.validation)

    return out


def params_dict_to_gui_state(param_values: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Inverse mapping for preset/state application (canonical keys -> gui_state scopes)."""

    values = dict(param_values or {})
    analysis_out: Dict[str, Any] = {}
    llm_out: Dict[str, Any] = {}
    reverse_aliases = {v: k for k, v in _PARAM_KEY_ALIASES.items()}

    for mapping in ANALYSIS_GUI_PARAM_MAP:
        if mapping.deprecated:
            continue
        canonical = _PARAM_KEY_ALIASES.get(mapping.params_key, mapping.params_key)
        if canonical in values:
            source_key = reverse_aliases.get(canonical, mapping.params_key)
            target = llm_out if mapping.visibility == "llm" else analysis_out
            target[source_key] = values[canonical]

    return {"analysis": analysis_out, "llm": llm_out}
