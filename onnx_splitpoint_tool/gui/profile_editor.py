"""Evaluation Profile editor for the GUI.

The formal workflow consumes YAML files.  This editor deliberately writes the
same schema that the CLI/runner already use, so the GUI stays a thin, friendly
front-end instead of becoming a second configuration system.
"""

from __future__ import annotations

import copy
import os
import re
import json
import hashlib
from datetime import datetime, timezone
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import Any, Callable, Iterable, Mapping, Optional

import yaml

from ..config_values import parse_config_bool, validate_profile_config_booleans
from ..campaign import (
    EVALUATED_MATRIX_CLAIM_SCOPE,
    RANKING_GENERALIZATION_CLAIM_SCOPE,
    build_campaign_readiness,
    create_holdout_registry,
    readiness_markdown,
)
from ..protocol_freeze import (
    CONFIRMATORY_HOLDOUT_ROLE,
    is_confirmatory_holdout,
    normalize_evaluation_role,
)
from ..execution_plan import build_effective_execution_plan, execution_plan_text
from ..hailo_timeout_policy import parse_hailo_timeout_seconds
from ..dataset_provisioning import default_registry_path, load_registry, registry_status
from .dataset_dialogs import open_dataset_provisioning_dialog
from ..run_modes import (
    apply_run_mode,
    default_run_modes_path,
    get_run_mode,
    infer_run_mode,
    load_run_modes_config,
    mode_summary,
    native_energy_requested,
    native_full_backends_from_run_profiles,
    normalize_mode_id,
    run_mode_profile_brief,
)
from .run_mode_editor import build_run_modes_panel
from ..workflow.hardware_matrix import default_hardware_setups_file
from ..validation.official_coco import pycocotools_status
from ..benchmark.evaluation_profiles import (
    evaluation_profile_default_root,
    load_evaluation_profile,
    save_evaluation_profile_yaml,
    validate_evaluation_profile_payload,
)
from .widgets.tooltip import attach_tooltip


ModelRow = dict[str, Any]

MODEL_USAGE_DEVELOPMENT = "Development"
MODEL_USAGE_HOLDOUT = "Hold-out"

_HOLDOUT_GENERALIZATION_SCOPES = {
    "model_family_holdout",
    "within_family_transfer",
}
_DEVELOPMENT_GENERALIZATION_SCOPES = {
    "development",
    "stress_test",
}
_AUDIT_UNIVERSE_MODES = {
    "all_feasible",
    "deterministic_audit",
    "audit_universe",
}
_MODEL_UNIVERSE_MODES = _AUDIT_UNIVERSE_MODES | {"declared_shortlist"}


def _norm_id(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = Path(text).stem if any(ch in text for ch in "/\\.") else text
    text = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()
    return text or "model"


def _split_csv(value: Any) -> list[str]:
    out: list[str] = []
    for item in str(value or "").replace(";", ",").split(","):
        item = item.strip()
        if item:
            out.append(item)
    return out


def _normalize_hardware_setup_id(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    # Recover from previously double-serialized list strings such as
    # "['[\"orin_nx_hailo8_01\"]']".
    m = re.search(r"orin_nx_[A-Za-z0-9_]+", text)
    if m:
        return m.group(0)
    text = text.strip("[](){} \t\r\n\"'")
    return text.strip()

def _normalize_hardware_setup_ids(values: Any) -> list[str]:
    out: list[str] = []
    raw = values if isinstance(values, (list, tuple, set)) and not isinstance(values, (str, bytes, bytearray)) else _split_csv(values)
    for item in raw:
        sid = _normalize_hardware_setup_id(item)
        if sid and sid not in out:
            out.append(sid)
    return out


def _shape_from_text(value: Any) -> list[int]:
    text = str(value or "").strip()
    if not text:
        return []
    parts = re.split(r"[xX,;\s]+", text)
    shape: list[int] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        try:
            shape.append(int(part))
        except Exception:
            return []
    return shape if len(shape) >= 4 else []


def _shape_to_text(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        try:
            return "x".join(str(int(x)) for x in value)
        except Exception:
            return ""
    return str(value or "")


def _as_bool_var(master: tk.Misc, value: bool) -> tk.BooleanVar:
    return tk.BooleanVar(master=master, value=bool(value))


def _host_display_to_id(value: Any) -> str:
    text = str(value or "").strip()
    if "—" in text:
        return text.split("—", 1)[0].strip()
    return text


def _normalize_model_usage(value: Any, *, strict: bool = False) -> str:
    """Return the two-value label used by the compact model editor.

    The public profile contract historically used both ``holdout`` and
    ``confirmatory_holdout``.  The GUI deliberately presents one unambiguous
    choice while continuing to accept both spellings on load.
    """

    token = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if token in {
        "holdout",
        "hold_out",
        "confirmatory_holdout",
        "confirmatory",
        "final_holdout",
    }:
        return MODEL_USAGE_HOLDOUT
    if token in {"", "development", "develop", "dev", "screening"}:
        return MODEL_USAGE_DEVELOPMENT
    if strict:
        raise ValueError("Verwendung muss Development oder Hold-out sein.")
    return MODEL_USAGE_DEVELOPMENT


def _model_usage_from_row(row: Mapping[str, Any] | None) -> str:
    """Infer the compact usage label from current or legacy model metadata."""

    data = dict(row or {}) if isinstance(row, Mapping) else {}
    role = normalize_evaluation_role(data.get("evaluation_role"))
    if role == CONFIRMATORY_HOLDOUT_ROLE:
        return MODEL_USAGE_HOLDOUT
    if role == "development":
        return MODEL_USAGE_DEVELOPMENT
    scope = str(data.get("generalization_scope") or "").strip().lower().replace("-", "_")
    if scope in _HOLDOUT_GENERALIZATION_SCOPES:
        return MODEL_USAGE_HOLDOUT
    return MODEL_USAGE_DEVELOPMENT


def _derive_simple_model_contract(
    row: Mapping[str, Any] | None,
    *,
    usage: Any,
    selection_strategy: Any = "stratified_windows",
    default_validation_tier: Any = "screening",
    development_family_ids: Iterable[str] = (),
    preserve_legacy: bool = True,
) -> ModelRow:
    """Materialise hidden scientific fields for the compact model form.

    Audit size, minimum and seed intentionally do not originate here.  New
    rows inherit those values from ``selection_policy`` at runtime.  Existing
    nested values are retained only when an unchanged legacy contract is
    loaded, which keeps old profiles roundtrip-compatible.
    """

    result: ModelRow = copy.deepcopy(dict(row or {})) if isinstance(row, Mapping) else {}
    chosen_usage = _normalize_model_usage(usage, strict=True)
    previous_usage = _model_usage_from_row(result)
    has_legacy_contract = any(
        key in result
        for key in (
            "evaluation_role",
            "generalization_scope",
            "validation_tier",
            "candidate_universe",
            "candidate_universe_complete",
        )
    )
    preserve_existing = bool(
        preserve_legacy and has_legacy_contract and previous_usage == chosen_usage
    )

    family_id = str(result.get("family_id") or result.get("family") or "").strip()
    if family_id and not str(result.get("family_id") or "").strip():
        result["family_id"] = family_id
    development_families = {
        str(value or "").strip()
        for value in development_family_ids
        if str(value or "").strip()
    }
    existing_scope = str(result.get("generalization_scope") or "").strip().lower().replace("-", "_")
    existing_tier = str(result.get("validation_tier") or "").strip().lower()
    existing_universe = (
        copy.deepcopy(dict(result.get("candidate_universe") or {}))
        if isinstance(result.get("candidate_universe"), Mapping)
        else {}
    )
    existing_mode = str(existing_universe.get("mode") or "").strip().lower().replace("-", "_")
    strategy = str(selection_strategy or "stratified_windows").strip().lower().replace("-", "_")
    development_audit = strategy in {
        "score_independent_audit",
        "ranking_audit",
        "deterministic_audit",
    }

    if chosen_usage == MODEL_USAGE_HOLDOUT:
        result["evaluation_role"] = CONFIRMATORY_HOLDOUT_ROLE
        result["generalization_scope"] = (
            existing_scope
            if preserve_existing and existing_scope in _HOLDOUT_GENERALIZATION_SCOPES
            else (
                "within_family_transfer"
                if family_id and family_id in development_families
                else "model_family_holdout"
            )
        )
        result["validation_tier"] = "final"
        universe_mode = (
            existing_mode
            if preserve_existing and existing_mode in _AUDIT_UNIVERSE_MODES
            else "deterministic_audit"
        )
        preserve_universe_extras = bool(
            preserve_existing
            and existing_mode == universe_mode
            and not development_audit
        )
        universe = existing_universe if preserve_universe_extras else {}
        universe["mode"] = universe_mode
        result["candidate_universe"] = universe
    else:
        result["evaluation_role"] = "development"
        result["generalization_scope"] = (
            existing_scope
            if preserve_existing and existing_scope in _DEVELOPMENT_GENERALIZATION_SCOPES
            else "development"
        )
        tier = existing_tier if preserve_existing else ""
        if tier not in {"screening", "final"}:
            tier = str(default_validation_tier or "screening").strip().lower()
        result["validation_tier"] = tier if tier in {"screening", "final"} else "screening"

        if development_audit:
            universe_mode = "deterministic_audit"
        elif strategy == "all_feasible":
            universe_mode = "all_feasible"
        elif strategy == "candidate_universe" and preserve_existing and existing_mode in _MODEL_UNIVERSE_MODES:
            universe_mode = existing_mode
        else:
            universe_mode = "declared_shortlist"
        preserve_universe_extras = bool(
            preserve_existing
            and existing_mode == universe_mode
            and not development_audit
        )
        universe = existing_universe if preserve_universe_extras else {}
        universe["mode"] = universe_mode
        result["candidate_universe"] = universe
        result.pop("holdout_group", None)
        result.pop("unseen_attestation", None)

    # Completeness is established by the generated, hashed universe/freeze
    # artefact.  An editor-authored False is an explicit veto in the runner,
    # so new/edited rows omit the field.  Preserve a genuinely untouched
    # legacy declaration, except for an old audit False that caused valid
    # score-independent universes to remain unauditable.
    if not preserve_existing or (
        development_audit
        and result.get("candidate_universe_complete") is False
    ):
        result.pop("candidate_universe_complete", None)

    return result


class EvaluationProfileEditor(tk.Toplevel):
    """Simplified evaluation-profile builder.

    A per-run profile contains only the run mode, models, candidate selection,
    logical hardware targets and the Native/Energy switches.  Detailed build,
    validation, reporting, hold-out and reproducibility settings are resolved
    from the centrally editable Tool Config run-mode registry.
    """

    def __init__(
        self,
        master: tk.Misc,
        *,
        app: Any | None = None,
        profile_var: tk.StringVar | None = None,
        combo: ttk.Combobox | None = None,
        on_saved: Callable[[Path], None] | None = None,
    ) -> None:
        super().__init__(master)
        self.title("Evaluation Profile erstellen / bearbeiten")
        self.geometry("1220x860")
        self.minsize(1040, 760)
        self.resizable(True, True)
        try:
            self.state("zoomed")
        except Exception:
            try:
                self.attributes("-zoomed", True)
            except Exception:
                pass
        self.transient(master.winfo_toplevel())

        self.app = app
        self.profile_var = profile_var
        self.combo = combo
        self.on_saved = on_saved
        self.models: list[ModelRow] = []
        # The compact reserve text field edits membership only.  Keep the
        # structured rows alongside it so a load/save cycle cannot discard an
        # explicit ``enabled: false`` flag or the scientific role/family data.
        self._reserve_model_entries: list[Any] = []
        # ``measurement_campaign`` is a frozen scientific contract rather than
        # a simplified GUI control.  Preserve it verbatim until the editor has
        # dedicated controls for every field in that block.
        self._measurement_campaign_passthrough: dict[str, Any] | None = None
        # Cache-verification profiles are immutable safety contracts.  The
        # simplified editor has no controls for their exact row attestation or
        # variants, so preserve those blocks verbatim on a load/save roundtrip.
        self._execution_guard_passthrough: dict[str, Any] | None = None
        self._cache_verify_native_passthrough: dict[str, Any] | None = None
        self._cache_verify_forced_cases_passthrough: dict[str, Any] | None = None
        self._protocol_freeze_passthrough: dict[str, Any] | None = None
        self._require_protocol_freeze_passthrough: bool = False
        self._prediction_freeze_enabled_passthrough: bool | None = None
        self._ranking_validation_passthrough: dict[str, Any] | None = None
        self._selected_model_index: int | None = None

        # General profile fields.
        self.var_path = tk.StringVar(self, value="")
        self.var_name = tk.StringVar(self, value="custom_splitpoint_eval_v1")
        self.var_purpose = tk.StringVar(self, value="Custom evaluation profile for splitpoint prediction and benchmark validation.")
        self.var_models_root_hint = tk.StringVar(self, value="")
        self.var_reserve = tk.StringVar(self, value="")

        # Candidate selection.
        self.var_cases = tk.IntVar(self, value=5)
        self.var_shortlist = tk.IntVar(self, value=10)
        self.var_min_gap = tk.IntVar(self, value=2)
        self.var_pool = tk.StringVar(self, value="auto")
        self.var_selection_strategy = tk.StringVar(self, value="stratified_windows")
        from ..backend_backfill import DEFAULT_BACKFILL
        self.var_backend_backfill = tk.BooleanVar(self, value=True)
        self.var_backfill_limits = {key: tk.IntVar(self, value=value)
            for key, value in DEFAULT_BACKFILL.items() if key not in {'enabled', 'technical_output_contract_version'}}
        self.backend_output_contract_version = DEFAULT_BACKFILL['technical_output_contract_version']

        self.var_audit_size = tk.IntVar(self, value=20)
        self.var_audit_min_valid = tk.IntVar(self, value=10)
        self.var_audit_seed = tk.IntVar(self, value=20260710)
        self.var_require_single_part2_input = tk.BooleanVar(self, value=False)
        self.var_keep_partial_hailo = tk.BooleanVar(self, value=True)
        self.var_report_blocked = tk.BooleanVar(self, value=True)
        self.var_full_preflight = tk.StringVar(self, value="skip")

        # Model edit fields.
        self.var_model_id = tk.StringVar(self, value="")
        self.var_model_task = tk.StringVar(self, value="auto")
        self.var_model_onnx = tk.StringVar(self, value="")
        self.var_model_family = tk.StringVar(self, value="")
        self.var_model_family_id = tk.StringVar(self, value="")
        self.var_model_generalization_scope = tk.StringVar(self, value="development")
        self.var_model_source = tk.StringVar(self, value="")
        self.var_model_shape = tk.StringVar(self, value="")
        self.var_model_dataset = tk.StringVar(self, value="")
        self.var_model_subset = tk.StringVar(self, value="")
        self.var_model_note = tk.StringVar(self, value="")

        # Model role / hold-out registration fields.
        self.var_model_usage = tk.StringVar(self, value=MODEL_USAGE_DEVELOPMENT)
        self.var_model_role = tk.StringVar(self, value="development")
        self.var_model_validation_tier = tk.StringVar(self, value="screening")
        self.var_model_sha256 = tk.StringVar(self, value="")
        self.var_model_holdout_group = tk.StringVar(self, value="")
        self.var_model_universe_mode = tk.StringVar(self, value="declared_shortlist")
        self.var_model_universe_complete = tk.BooleanVar(self, value=False)
        self.var_model_unseen = tk.BooleanVar(self, value=False)
        self.var_model_attested_by = tk.StringVar(self, value="")
        self.var_model_attested_at = tk.StringVar(self, value="")
        self.var_model_attestation_note = tk.StringVar(self, value="")

        # Run profiles / targets.
        self.var_run_cpu = tk.BooleanVar(self, value=False)
        self.var_run_cuda = tk.BooleanVar(self, value=True)
        self.var_run_trt = tk.BooleanVar(self, value=True)
        self.var_run_hailo = tk.BooleanVar(self, value=True)
        self.var_run_hailo_to_trt = tk.BooleanVar(self, value=True)
        self.var_run_trt_to_hailo = tk.BooleanVar(self, value=False)
        self.var_run_hailo10 = tk.BooleanVar(self, value=False)
        self.var_run_hailo10_to_trt = tk.BooleanVar(self, value=False)
        self.var_run_deepx = tk.BooleanVar(self, value=False)
        self.var_run_deepx_to_trt = tk.BooleanVar(self, value=False)
        self.var_run_trt_to_deepx = tk.BooleanVar(self, value=False)

        # Validation and report settings.
        self.var_validation_mode = tk.StringVar(self, value="summary_only")
        self.var_classification_metrics = tk.StringVar(self, value="top1, top5, logit_cosine_similarity")
        self.var_detection_metrics = tk.StringVar(self, value="semantic_proxy_match_ratio, semantic_proxy_mean_iou, mini_coco_ap50")
        self.var_reference_mode = tk.StringVar(self, value="auto")
        self.var_backend_drift_reference = tk.StringVar(self, value="cpu_full")
        self.var_require_explicit_validation = tk.BooleanVar(self, value=False)
        self.var_require_task_metrics = tk.BooleanVar(self, value=False)
        self.var_report_decisions = tk.BooleanVar(self, value=True)
        self.var_report_backend_drift = tk.BooleanVar(self, value=True)
        self.var_report_task_quality = tk.BooleanVar(self, value=True)

        # Final campaign / task-quality / ranking-reporting configuration.
        self.var_campaign_id = tk.StringVar(self, value="thesis_final_campaign_v1")
        self.var_claim_scope = tk.StringVar(self, value=EVALUATED_MATRIX_CLAIM_SCOPE)
        self.var_campaign_mode = tk.StringVar(self, value="development")
        self.var_campaign_enforcement = tk.StringVar(self, value="warn")
        self.var_campaign_frozen = tk.BooleanVar(self, value=False)
        self.var_dataset_registry = tk.StringVar(self, value=str(default_registry_path()))
        self.var_auto_bind_dataset_registry = tk.BooleanVar(self, value=True)
        self.var_manifest_cls_calib = tk.StringVar(self, value="")
        self.var_manifest_cls_val = tk.StringVar(self, value="")
        self.var_manifest_det_calib = tk.StringVar(self, value="")
        self.var_manifest_det_val = tk.StringVar(self, value="")
        self.var_pipeline_contract_manifest = tk.StringVar(self, value="")
        self.var_holdout_registry = tk.StringVar(self, value="campaign_inputs/holdout_registry.json")
        self.var_ranking_model_bundle = tk.StringVar(self, value="")
        self.var_energy_calibration_manifest = tk.StringVar(self, value="")
        self.var_require_fitted_stage_time = tk.BooleanVar(self, value=False)
        self.var_require_native_handover = tk.BooleanVar(self, value=False)
        self.var_require_campaign_freeze = tk.BooleanVar(self, value=False)
        self.var_require_prediction_approval = tk.BooleanVar(self, value=False)
        self.var_require_prediction_signature = tk.BooleanVar(self, value=False)
        self.var_prediction_public_key = tk.StringVar(self, value="")

        self.var_quality_profile_id = tk.StringVar(self, value="task_quality_v1")
        self.var_quality_frozen = tk.BooleanVar(self, value=False)
        self.var_quality_tier = tk.StringVar(self, value="screening")
        self.var_quality_canonical_ref = tk.StringVar(self, value="canonical_full_onnx")
        self.var_quality_cls_margin_pp = tk.DoubleVar(self, value=1.0)
        self.var_quality_cls_top5_margin_pp = tk.DoubleVar(self, value=1.0)
        self.var_quality_det_margin_ap = tk.DoubleVar(self, value=1.0)
        self.var_quality_det_ap50_margin = tk.DoubleVar(self, value=1.0)
        self.var_quality_det_ap75_margin = tk.DoubleVar(self, value=1.0)
        self.var_quality_confidence = tk.DoubleVar(self, value=0.95)
        self.var_quality_bootstrap = tk.IntVar(self, value=5000)
        self.var_quality_seed = tk.IntVar(self, value=20260710)
        self.var_quality_execution_location = tk.StringVar(self, value="central_management")
        self.var_quality_workers = tk.IntVar(self, value=4)

        self.var_ranking_enabled = tk.BooleanVar(self, value=True)
        self.var_ranking_require_frozen = tk.BooleanVar(self, value=True)
        self.var_ranking_require_complete = tk.BooleanVar(self, value=True)
        self.var_ranking_k_values = tk.StringVar(self, value="1, 3, 5")
        self.var_ranking_q_values = tk.StringVar(self, value="1, 3")
        self.var_ranking_primary_k = tk.IntVar(self, value=5)
        self.var_ranking_min_corr = tk.IntVar(self, value=3)
        self.var_ranking_epsilon_pct = tk.DoubleVar(self, value=1.0)
        self.var_ranking_bootstrap = tk.IntVar(self, value=5000)
        self.var_ranking_seed = tk.IntVar(self, value=20260710)

        self.var_official_coco_enabled = tk.BooleanVar(self, value=False)
        self.var_official_coco_required = tk.BooleanVar(self, value=False)
        self.var_official_coco_annotations = tk.StringVar(self, value="")
        self.var_official_coco_remote_annotations = tk.StringVar(self, value="")
        self.var_official_coco_archive_tensors = tk.BooleanVar(self, value=True)
        self.var_official_coco_max_dets = tk.StringVar(self, value="1, 10, 100")
        self.var_generate_thesis_tex = tk.BooleanVar(self, value=True)
        self.var_generate_thesis_figures = tk.BooleanVar(self, value=True)
        self.var_cleanup_legacy_reports = tk.BooleanVar(self, value=True)
        self.var_include_campaign_readiness = tk.BooleanVar(self, value=True)

        # Workflow/runtime defaults stored in YAML.
        self.var_workflow_execution_mode = tk.StringVar(self, value="generate_and_run")
        self.var_workflow_skip_runtime = tk.BooleanVar(self, value=False)
        self.var_workflow_no_model_hash = tk.BooleanVar(self, value=True)
        self.var_workflow_include_reserve = tk.BooleanVar(self, value=False)
        self.var_parallel_remote_setups = tk.BooleanVar(self, value=True)
        self.var_max_parallel_setups = tk.IntVar(self, value=3)
        self.var_max_parallel_uploads = tk.IntVar(self, value=1)
        self.var_powercalc_workers = tk.IntVar(self, value=1)
        self.var_benchmark_provider = tk.StringVar(self, value="auto")
        self.var_benchmark_warmup = tk.IntVar(self, value=1)
        self.var_benchmark_runs = tk.IntVar(self, value=3)
        self.var_benchmark_timeout = tk.IntVar(self, value=0)
        self.var_energy_enabled = tk.BooleanVar(self, value=False)
        self.var_energy_final_all_splits = tk.BooleanVar(self, value=False)
        self.var_energy_final_skip_cpu_ort = tk.BooleanVar(self, value=True)
        self.var_energy_repeat_override = tk.IntVar(self, value=0)
        self.var_energy_scope = tk.StringVar(self, value="row_variant")
        self.var_energy_phases = tk.StringVar(self, value="latency, streaming")
        self.var_energy_target_policy = tk.StringVar(self, value="canonical_only")
        self.var_energy_skip_backends = tk.StringVar(self, value="ort_cpu, ort_cuda")
        self.var_energy_include_run_ids = tk.StringVar(self, value="")
        self.var_energy_exclude_run_ids = tk.StringVar(self, value="")
        self.var_energy_heartbeat_s = tk.IntVar(self, value=60)
        self.var_energy_max_targets_per_run_id = tk.IntVar(self, value=0)
        self.var_energy_max_work_units_per_window = tk.IntVar(self, value=0)
        self.var_energy_max_window_duration_s = tk.IntVar(self, value=0)
        self.var_energy_timeout_s_per_window = tk.IntVar(self, value=0)
        self.var_energy_sizing_probe_max_work_units = tk.IntVar(self, value=256)
        self.var_energy_include_raw_parquet = tk.BooleanVar(self, value=False)
        self.var_energy_strict = tk.BooleanVar(self, value=False)
        self.var_energy_estimate = tk.StringVar(self, value="Energy: disabled")

        # Native producer fastpath defaults (strict supported-only execution mode).
        self.var_native_enabled = tk.BooleanVar(self, value=False)
        self.var_native_backends = tk.StringVar(self, value="hailo8, hailo10h, deepx")
        self.var_native_case_policy = tk.StringVar(self, value="all_accepted")
        self.var_native_precision = tk.StringVar(self, value="uint8_cast_fp16")
        self.var_native_frames = tk.IntVar(self, value=100)
        self.var_native_warmup = tk.IntVar(self, value=10)
        self.var_native_repetitions = tk.IntVar(self, value=1)
        self.var_native_queue_depth = tk.IntVar(self, value=3)
        self.var_native_inflight = tk.IntVar(self, value=8)
        self.var_native_full_baselines = tk.BooleanVar(self, value=True)
        self.var_native_validation_enabled = tk.BooleanVar(self, value=True)
        self.var_native_energy_enabled = tk.BooleanVar(self, value=False)
        self.var_native_energy_mode = tk.StringVar(self, value="plan")
        # Native energy is duration-based.  The value is normally left at 0,
        # meaning: use Tool Config / energy_defaults.native_measurement_duration_s.
        self.var_native_energy_duration_s = tk.IntVar(self, value=0)
        self.var_window_method_probe_enabled = tk.BooleanVar(self, value=True)
        self.var_window_method_probe_repeats = tk.IntVar(self, value=3)
        self.var_window_method_probe_include_raw = tk.BooleanVar(self, value=True)
        self.var_window_method_probe_strict = tk.BooleanVar(self, value=True)

        # Advanced defaults.
        self.var_prep_mode = tk.StringVar(self, value="current")
        self.var_hailo_build_mode = tk.StringVar(self, value="reuse_and_build_missing")
        self.var_hailo_arch = tk.StringVar(self, value="hailo8")
        self.var_hailo_timeout = tk.IntVar(self, value=3600)
        self.var_hailo_targets = tk.StringVar(self, value="hailo8")
        self.var_hailo_build_full = tk.BooleanVar(self, value=True)
        self.var_hailo_build_part1 = tk.BooleanVar(self, value=True)
        self.var_hailo_build_part2 = tk.BooleanVar(self, value=True)
        self.var_hailo_preset = tk.StringVar(self, value="quick")
        self.var_hailo_opt_level = tk.IntVar(self, value=0)
        self.var_hailo_calib_count = tk.IntVar(self, value=16)
        self.var_hailo_calib_batch = tk.IntVar(self, value=8)
        self.var_hailo_calib_dir = tk.StringVar(self, value="")
        self.var_hailo_force_build = tk.BooleanVar(self, value=False)
        self.var_hailo_keep_artifacts = tk.BooleanVar(self, value=True)
        self.var_hw_smoke_mode = tk.StringVar(self, value="summary_only")
        self.var_remote_enabled = tk.BooleanVar(self, value=False)
        self.var_remote_host_id = tk.StringVar(self, value="")
        self.var_remote_host = tk.StringVar(self, value="")
        self.var_remote_user = tk.StringVar(self, value="")
        self.var_remote_port = tk.IntVar(self, value=22)
        self.var_remote_base = tk.StringVar(self, value="~/splitpoint_runs")
        self.var_remote_provider = tk.StringVar(self, value="auto")
        self.var_remote_iters = tk.IntVar(self, value=50)
        self.var_remote_warmup = tk.IntVar(self, value=10)
        self.var_remote_timeout = tk.IntVar(self, value=0)
        self.var_remote_venv = tk.StringVar(self, value="")
        self.var_remote_transfer_mode = tk.StringVar(self, value="bundle")
        self.var_remote_reuse_bundle = tk.BooleanVar(self, value=True)
        self.var_remote_resume = tk.BooleanVar(self, value=True)
        self.var_remote_ssh_extra_args = tk.StringVar(self, value="")

        # Accelerator build-environment registry values written into YAML.
        self.var_deepx_root = tk.StringVar(self, value="~/dx-all-suite")
        self.var_deepx_compiler_venv = tk.StringVar(self, value="~/dx-all-suite/dx-compiler/venv-dx-compiler-local")
        self.var_deepx_compiler_overlay = tk.StringVar(self, value="")
        self.var_deepx_legacy_notice = tk.StringVar(self, value="")
        self.var_deepx_cache_dir = tk.StringVar(self, value="~/Models/BackendArtifacts/deepx")
        # New classification profiles use the verified ImageNet normalization
        # contract.  Loading an older profile without this field explicitly
        # selects current_scale_only below, preserving its legacy behaviour.
        self.var_deepx_classification_preprocessing = tk.StringVar(
            self, value="imagenet_mean_std"
        )
        def _refresh_deepx_legacy_notice(*_args: Any) -> None:
            self.var_deepx_legacy_notice.set("Legacy/A-B-Diagnose: current_scale_only; für reguläre Claims imagenet_mean_std wählen." if self.var_deepx_classification_preprocessing.get() == "current_scale_only" else "")
        self.var_deepx_classification_preprocessing.trace_add("write", _refresh_deepx_legacy_notice)
        self.var_hardware_selected_setups = tk.StringVar(self, value="")
        self.var_hardware_selected_groups = tk.StringVar(self, value="")
        self.var_hardware_setups_file = tk.StringVar(self, value="")

        # Per-accelerator remote setup fields. These make the three-Orin-NX
        # matrix editable in the GUI instead of requiring hand-written YAML.
        def _b(default: bool = False) -> tk.BooleanVar:
            return tk.BooleanVar(self, value=default)
        def _s(default: str = "") -> tk.StringVar:
            return tk.StringVar(self, value=default)
        self.var_hw_h8_enabled = _b(False)
        self.var_hw_h10_enabled = _b(False)
        self.var_hw_dx_enabled = _b(False)
        self.var_hw_h8_host_id = _s("")
        self.var_hw_h10_host_id = _s("")
        self.var_hw_dx_host_id = _s("")
        self.var_hw_h8_base = _s("~/splitpoint_runs")
        self.var_hw_h10_base = _s("~/splitpoint_runs")
        self.var_hw_dx_base = _s("~/splitpoint_runs")
        self.var_hw_h8_venv = _s("~/hailo_py/bin/activate")
        self.var_hw_h10_venv = _s("~/hailo_py/bin/activate")
        self.var_hw_dx_venv = _s("~/venvs/deepx-runtime/bin/activate")
        self.var_hw_h8_provider = _s("hailo8")
        self.var_hw_h10_provider = _s("hailo10")
        self.var_hw_dx_provider = _s("deepx_m1")

        # v60n: the Evaluation Profile only exposes the high-level run mode.
        # All detailed build/validation/reporting values are edited centrally in
        # Tool Config and materialised into the saved profile/run snapshot.
        try:
            _run_modes_cfg = load_run_modes_config()
            _default_run_mode = normalize_mode_id(_run_modes_cfg.get("default_mode") or "standard")
        except Exception:
            _default_run_mode = "standard"
        self.var_run_mode_id = tk.StringVar(self, value=_default_run_mode)
        self.var_run_mode_follow_tool_config = tk.BooleanVar(self, value=True)
        self.var_run_mode_summary = tk.StringVar(self, value="")
        self._loading_profile = False

        self._build_ui()
        try:
            self.var_run_mode_id.trace_add("write", self._on_run_mode_changed)
            self.var_native_enabled.trace_add("write", self._update_run_summary)
            self.var_energy_enabled.trace_add("write", self._update_run_summary)
            self.var_backend_backfill.trace_add('write', self._update_run_summary)
            for variable in self.var_backfill_limits.values():
                variable.trace_add('write', self._update_run_summary)
        except Exception:
            pass
        for _var in [
            self.var_energy_enabled, self.var_energy_final_all_splits, self.var_energy_repeat_override, self.var_energy_scope, self.var_energy_phases,
            self.var_energy_target_policy, self.var_energy_skip_backends, self.var_energy_include_run_ids,
            self.var_energy_exclude_run_ids, self.var_cases, self.var_benchmark_runs, self.var_benchmark_warmup,
            self.var_run_cpu, self.var_run_cuda, self.var_run_trt, self.var_run_hailo, self.var_run_hailo_to_trt,
            self.var_run_trt_to_hailo, self.var_run_hailo10, self.var_run_hailo10_to_trt, self.var_run_deepx,
            self.var_run_deepx_to_trt, self.var_run_trt_to_deepx,
        ]:
            try:
                _var.trace_add("write", self._update_energy_estimate)
            except Exception:
                pass
        try:
            self.var_energy_final_all_splits.trace_add("write", self._apply_final_all_split_energy_checkbox)
            self.var_energy_final_skip_cpu_ort.trace_add("write", self._apply_final_all_split_energy_checkbox)
        except Exception:
            pass
        self._try_load_initial_profile()
        self._update_energy_estimate()
        self._update_run_summary()
        self._install_fallback_tooltips()

    # ------------------------------------------------------------------
    # UI construction helpers
    # ------------------------------------------------------------------

    def _tip(self, widget: tk.Widget, text: str) -> tk.Widget:
        try:
            attach_tooltip(widget, text, delay_ms=350, wraplength=460)
            setattr(widget, "_splitpoint_has_tooltip", True)
        except Exception:
            pass
        return widget

    def _label(self, master: tk.Misc, text: str, tooltip: str = "") -> ttk.Label:
        lbl = ttk.Label(master, text=text)
        if tooltip:
            self._tip(lbl, tooltip)
        return lbl

    def _entry(self, master: tk.Misc, var: tk.Variable, tooltip: str = "", **kw: Any) -> ttk.Entry:
        w = ttk.Entry(master, textvariable=var, **kw)
        if tooltip:
            self._tip(w, tooltip)
        return w

    def _combo(self, master: tk.Misc, var: tk.Variable, values: Iterable[str], tooltip: str = "", **kw: Any) -> ttk.Combobox:
        w = ttk.Combobox(master, textvariable=var, values=list(values), **kw)
        if tooltip:
            self._tip(w, tooltip)
        return w

    def _spin(self, master: tk.Misc, var: tk.Variable, tooltip: str = "", **kw: Any) -> ttk.Spinbox:
        w = ttk.Spinbox(master, textvariable=var, **kw)
        if tooltip:
            self._tip(w, tooltip)
        return w

    def _check(self, master: tk.Misc, text: str, var: tk.BooleanVar, tooltip: str = "", **kw: Any) -> ttk.Checkbutton:
        w = ttk.Checkbutton(master, text=text, variable=var, **kw)
        if tooltip:
            self._tip(w, tooltip)
        return w

    def _button(self, master: tk.Misc, text: str, command: Callable[[], Any] | None, tooltip: str = "") -> ttk.Button:
        w = ttk.Button(master, text=text, command=command if callable(command) else (lambda: None))
        if tooltip:
            self._tip(w, tooltip)
        return w

    def _remote_host_values(self) -> list[str]:
        try:
            vals = list(self.app._remote_hosts_values_for_combo()) if self.app is not None else []
        except Exception:
            vals = []
        return vals

    def _remote_host_config_by_id(self, host_id: str) -> Mapping[str, Any] | None:
        hid = _host_display_to_id(host_id)
        if not hid or self.app is None:
            return None
        try:
            svc = getattr(self.app, "_remote_service", None)
            hosts = getattr(self.app, "remote_hosts", []) or []
            if svc is not None:
                host = svc.get_selected_host(hosts, hid)
                if host is not None:
                    try:
                        return dict(host.to_dict())
                    except Exception:
                        return {
                            "id": str(getattr(host, "id", "") or hid),
                            "label": str(getattr(host, "label", "") or hid),
                            "host": str(getattr(host, "host", "") or ""),
                            "user": str(getattr(host, "user", "") or ""),
                            "port": int(getattr(host, "port", 22) or 22),
                            "remote_base_dir": str(getattr(host, "remote_base_dir", "~/splitpoint_runs") or "~/splitpoint_runs"),
                            "ssh_extra_args": str(getattr(host, "ssh_extra_args", "") or ""),
                        }
        except Exception:
            return None
        return None

    def _apply_remote_host_config_to_fields(self, payload: Mapping[str, Any] | None) -> None:
        if not isinstance(payload, Mapping):
            return
        self.var_remote_host_id.set(str(payload.get("id") or payload.get("label") or self.var_remote_host_id.get() or ""))
        self.var_remote_host.set(str(payload.get("host") or ""))
        self.var_remote_user.set(str(payload.get("user") or ""))
        try:
            self.var_remote_port.set(int(payload.get("port") or 22))
        except Exception:
            self.var_remote_port.set(22)
        self.var_remote_base.set(str(payload.get("remote_base_dir") or "~/splitpoint_runs"))
        self.var_remote_ssh_extra_args.set(str(payload.get("ssh_extra_args") or ""))

    def _install_fallback_tooltips(self) -> None:
        """Attach a plain fallback tooltip to editable controls without one.

        The editor uses many compact Entry/Combobox/Spinbox widgets.  v49i
        guarantees that hovering over the actual field, not only its label, gives
        at least a useful explanation.
        """
        generic = (
            "Dieses Feld beschreibt nur die Auswahl für den konkreten Run. "
            "Tiefe Build-, Quality-, Reporting- und Reproducibility-Einstellungen kommen aus Tool Config → Run modes."
        )

        def walk(widget: tk.Widget) -> None:
            for child in list(widget.winfo_children()):
                try:
                    cls = child.winfo_class()
                except Exception:
                    cls = ""
                if cls in {"TEntry", "TCombobox", "TSpinbox", "TCheckbutton", "TButton", "Treeview"} and not getattr(child, "_splitpoint_has_tooltip", False):
                    self._tip(child, generic)
                walk(child)

        try:
            walk(self)
        except Exception:
            pass

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        intro = ttk.Label(
            self,
            text=(
                "Ein Evaluation Profile enthält nur noch die Entscheidungen für den konkreten Run: "
                "Run-Modus, Modelle, Kandidatenauswahl, Zielpfade, Native Runner und Energie. "
                "Build-, Dataset-, Task-Quality-, Hold-out-, Reporting- und Reproduzierbarkeitsdetails "
                "werden zentral unter Tool Config → Run modes gepflegt."
            ),
            wraplength=1120,
            justify="left",
        )
        intro.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 8))

        nb = ttk.Notebook(self)
        nb.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 8))
        self.nb = nb

        general = ttk.Frame(nb)
        models = ttk.Frame(nb)
        targets = ttk.Frame(nb)
        summary = ttk.Frame(nb)
        nb.add(general, text="1. Run")
        nb.add(models, text="2. Modelle")
        nb.add(targets, text="3. Ziele & Kandidaten")
        nb.add(summary, text="4. Zusammenfassung")

        self._build_run_tab(general)
        self._build_models_tab_simple(models)
        self._build_targets_tab_simple(targets)
        self._build_run_summary_tab(summary)

        footer = ttk.Frame(self)
        footer.grid(row=2, column=0, sticky="ew", padx=12, pady=(0, 12))
        footer.columnconfigure(0, weight=1)
        self.status_var = tk.StringVar(self, value="Bereit.")
        ttk.Label(footer, textvariable=self.status_var).grid(row=0, column=0, sticky="w")
        ttk.Button(footer, text="Aktuelles Profil laden", command=self._load_from_dialog_or_current).grid(row=0, column=1, padx=(8, 0))
        ttk.Button(footer, text="Effektive YAML", command=self._preview_yaml).grid(row=0, column=2, padx=(8, 0))
        ttk.Button(footer, text="Validieren", command=self._validate_current).grid(row=0, column=3, padx=(8, 0))
        ttk.Button(footer, text="Speichern als…", command=lambda: self._save(use_after=False)).grid(row=0, column=4, padx=(8, 0))
        ttk.Button(footer, text="Speichern & verwenden", command=lambda: self._save(use_after=True)).grid(row=0, column=5, padx=(8, 0))
        ttk.Button(footer, text="Schließen", command=self.destroy).grid(row=0, column=6, padx=(8, 0))

    def _build_run_tab(self, tab: ttk.Frame) -> None:
        tab.columnconfigure(1, weight=1)
        tab.rowconfigure(6, weight=1)
        row = 0
        self._label(tab, "Profil-Datei:", "Pfad der kompakten Evaluation-Profile-YAML.").grid(row=row, column=0, sticky="e", padx=(8, 6), pady=(8, 4))
        ttk.Entry(tab, textvariable=self.var_path).grid(row=row, column=1, sticky="ew", pady=(8, 4))
        ttk.Button(tab, text="Durchsuchen…", command=self._choose_profile_path).grid(row=row, column=2, padx=8, pady=(8, 4))
        row += 1
        self._label(tab, "Name / Profile ID:", "Stabile ID des Runprofils.").grid(row=row, column=0, sticky="e", padx=(8, 6), pady=4)
        ttk.Entry(tab, textvariable=self.var_name).grid(row=row, column=1, columnspan=2, sticky="ew", padx=(0, 8), pady=4)
        row += 1
        self._label(tab, "Beschreibung:", "Kurze Beschreibung des geplanten Runs.").grid(row=row, column=0, sticky="e", padx=(8, 6), pady=4)
        ttk.Entry(tab, textvariable=self.var_purpose).grid(row=row, column=1, columnspan=2, sticky="ew", padx=(0, 8), pady=4)
        row += 1
        self._label(tab, "Models root:", "Optionaler Suchpfad für Modelle ohne absoluten ONNX-Pfad.").grid(row=row, column=0, sticky="e", padx=(8, 6), pady=4)
        ttk.Entry(tab, textvariable=self.var_models_root_hint).grid(row=row, column=1, sticky="ew", pady=4)
        ttk.Button(tab, text="Ordner…", command=lambda: self._browse_dir_var(self.var_models_root_hint)).grid(row=row, column=2, padx=8, pady=4)
        row += 1

        mode_box = ttk.LabelFrame(tab, text="Run mode")
        mode_box.grid(row=row, column=0, columnspan=3, sticky="ew", padx=8, pady=(10, 6))
        mode_box.columnconfigure(1, weight=1)
        ttk.Label(mode_box, text="Mode:").grid(row=0, column=0, sticky="e", padx=(8, 6), pady=8)
        combo = ttk.Combobox(mode_box, textvariable=self.var_run_mode_id, values=["smoke", "standard", "final"], state="readonly", width=18)
        combo.grid(row=0, column=1, sticky="w", pady=8)
        self._tip(combo, "Smoke = schnellster Funktionstest; Standard = ausgewogene Development-Evaluation; Final Quality = derselbe Ausführungspfad mit größerem Validierungs- und Bootstrap-Budget.")
        ttk.Button(mode_box, text="Run modes konfigurieren…", command=self._open_run_mode_settings).grid(row=0, column=2, sticky="e", padx=8, pady=8)
        ttk.Label(mode_box, textvariable=self.var_run_mode_summary, foreground="#444", wraplength=1050, justify="left").grid(row=1, column=0, columnspan=3, sticky="ew", padx=8, pady=(0, 8))
        row += 1

        switches = ttk.LabelFrame(tab, text="Optionale Run-Schalter")
        switches.grid(row=row, column=0, columnspan=3, sticky="ew", padx=8, pady=6)
        self._check(switches, "Native Runner ausführen", self.var_native_enabled, "Verwendet die Native FIFO/Producer-Pfade zusätzlich zum Generic Runner. Native- und Energy-Einstellungen bleiben beim Wechsel des Run-Modus unverändert.").grid(row=0, column=0, sticky="w", padx=8, pady=8)
        self._check(switches, "Native Systemenergie messen", self.var_energy_enabled, "Misst Energie im Native Runner. Physisch erfasste Rohenergie bleibt bei zeilenlokalen Quality-Problemen erhalten, wird aber als nicht qualifiziert markiert. Generic-Runner-Energie ist in EvalRuns standardmäßig deaktiviert.", command=self._on_energy_master_toggle).grid(row=0, column=1, sticky="w", padx=16, pady=8)
        ttk.Label(switches, text="Native Energy wird für physisch ausführbare Native-Zeilen erfasst; bei Quality-Problemen bleibt Rohenergie als nicht qualifiziert erhalten. Generic Energy bleibt aus.", foreground="#666").grid(row=0, column=2, sticky="w", padx=8, pady=8)
        row += 1

        info = tk.Text(tab, height=10, wrap="word", bg=self.cget("background"), relief="flat")
        info.grid(row=row, column=0, columnspan=3, sticky="nsew", padx=8, pady=(8, 8))
        info.insert(
            "1.0",
            "Run-Modi:\n"
            "• smoke: minimaler Build-/Validierungsaufwand für schnelle Verdrahtungs-, Remote- und Contract-Tests.\n"
            "• standard: ausgewogene Development-Evaluation mit moderatem Calibration-/Bootstrap-Aufwand.\n"
            "• final: Standard-Ausführung mit 5.000 Validierungsbildern je Task und 5.000 Quality-Bootstrap-Wiederholungen.\n"
            "  Strikte Kampagnen, vollständige Manifest-Prüfung und Canaries werden dadurch nicht automatisch aktiviert.\n\n"
            "Hardware-Hosts, SSH, Runtime-Venvs und u.RECS-Adressen werden automatisch aus Tool Config → Hardware Run Profiles abgeleitet. "
            "Der konkrete Modus-Snapshot wird in jedem Run archiviert, damit spätere Änderungen an Tool Config alte Ergebnisse nicht verändern.",
        )
        info.configure(state="disabled")

    def _build_models_tab_simple(self, tab: ttk.Frame) -> None:
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(0, weight=1)
        columns = ("id", "task", "usage", "onnx", "family", "shape", "note")
        tree = ttk.Treeview(tab, columns=columns, show="headings", height=10)
        self.model_tree = tree
        headings = {
            "id": "ID",
            "task": "Task",
            "usage": "Verwendung",
            "onnx": "ONNX",
            "family": "Modellfamilie",
            "shape": "Input shape",
            "note": "Note",
        }
        widths = {
            "id": 125,
            "task": 95,
            "usage": 105,
            "onnx": 330,
            "family": 130,
            "shape": 110,
            "note": 260,
        }
        for col in columns:
            tree.heading(col, text=headings[col])
            tree.column(col, width=widths[col], anchor="w")
        tree.grid(row=0, column=0, columnspan=4, sticky="nsew", padx=8, pady=(8, 6))
        scrollbar = ttk.Scrollbar(tab, orient="vertical", command=tree.yview)
        scrollbar.grid(row=0, column=4, sticky="ns", pady=(8, 6))
        tree.configure(yscrollcommand=scrollbar.set)
        tree.bind("<<TreeviewSelect>>", self._on_model_selected, add="+")

        form = ttk.LabelFrame(tab, text="Modell")
        form.grid(row=1, column=0, columnspan=5, sticky="ew", padx=8, pady=(0, 8))
        for col in (1, 3, 5):
            form.columnconfigure(col, weight=1)
        ttk.Label(form, text="ID:").grid(row=0, column=0, sticky="e", padx=(8, 6), pady=6)
        ttk.Entry(form, textvariable=self.var_model_id).grid(row=0, column=1, sticky="ew", pady=6)
        ttk.Label(form, text="Task:").grid(row=0, column=2, sticky="e", padx=(12, 6), pady=6)
        ttk.Combobox(form, textvariable=self.var_model_task, values=["auto", "classification", "detection"], state="readonly", width=15).grid(row=0, column=3, sticky="w", pady=6)
        ttk.Label(form, text="Verwendung:").grid(row=0, column=4, sticky="e", padx=(12, 6), pady=6)
        ttk.Combobox(
            form,
            textvariable=self.var_model_usage,
            values=[MODEL_USAGE_DEVELOPMENT, MODEL_USAGE_HOLDOUT],
            state="readonly",
            width=15,
        ).grid(row=0, column=5, sticky="w", padx=(0, 8), pady=6)

        ttk.Label(form, text="ONNX:").grid(row=1, column=0, sticky="e", padx=(8, 6), pady=6)
        ttk.Entry(form, textvariable=self.var_model_onnx).grid(row=1, column=1, columnspan=4, sticky="ew", pady=6)
        ttk.Button(form, text="ONNX…", command=self._browse_model_onnx).grid(row=1, column=5, sticky="w", padx=(6, 8), pady=6)

        ttk.Label(form, text="Modellfamilie:").grid(row=2, column=0, sticky="e", padx=(8, 6), pady=6)
        ttk.Entry(form, textvariable=self.var_model_family).grid(row=2, column=1, sticky="ew", pady=6)
        ttk.Label(form, text="Input shape:").grid(row=2, column=2, sticky="e", padx=(12, 6), pady=6)
        ttk.Entry(form, textvariable=self.var_model_shape).grid(row=2, column=3, sticky="ew", pady=6)
        ttk.Label(form, text="Note:").grid(row=2, column=4, sticky="e", padx=(12, 6), pady=6)
        ttk.Entry(form, textvariable=self.var_model_note).grid(row=2, column=5, sticky="ew", padx=(0, 8), pady=6)

        buttons = ttk.Frame(tab)
        buttons.grid(row=2, column=0, columnspan=5, sticky="ew", padx=8, pady=(0, 8))
        ttk.Button(buttons, text="Aktuelles GUI-Modell hinzufügen", command=self._add_current_gui_model).pack(side=tk.LEFT)
        ttk.Button(buttons, text="Modell setzen", command=self._upsert_model).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(buttons, text="Neu", command=self._clear_model_fields).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(buttons, text="Entfernen", command=self._remove_selected_model).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Label(
            buttons,
            text=(
                "Role, Generalization, Validation tier und Candidate universe werden aus "
                "Development/Hold-out konsistent abgeleitet. Auditgröße, Minimum und Seed "
                "werden global unter Ziele & Kandidaten gesetzt."
            ),
            foreground="#666",
        ).pack(side=tk.LEFT, padx=(18, 0))

    def _build_targets_tab_simple(self, tab: ttk.Frame) -> None:
        tab.columnconfigure(0, weight=1)
        candidates = ttk.LabelFrame(tab, text="Candidate selection")
        candidates.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 6))
        for c in (1, 3, 5): candidates.columnconfigure(c, weight=1)
        ttk.Label(candidates, text="Cases/model:").grid(row=0, column=0, sticky="e", padx=(8, 6), pady=8)
        ttk.Spinbox(candidates, textvariable=self.var_cases, from_=1, to=10000, width=8).grid(row=0, column=1, sticky="w", pady=8)
        ttk.Label(candidates, text="Shortlist:").grid(row=0, column=2, sticky="e", padx=(12, 6), pady=8)
        ttk.Spinbox(candidates, textvariable=self.var_shortlist, from_=1, to=10000, width=8).grid(row=0, column=3, sticky="w", pady=8)
        ttk.Label(candidates, text="Strategy:").grid(row=0, column=4, sticky="e", padx=(12, 6), pady=8)
        ttk.Combobox(
            candidates,
            textvariable=self.var_selection_strategy,
            values=[
                "score_independent_audit",
                "stratified_windows",
                "predicted_topk",
                "candidate_universe",
                "all_feasible",
                "deterministic_audit",
            ],
            state="readonly",
            width=24,
        ).grid(row=0, column=5, sticky="w", padx=(0, 8), pady=8)

        ttk.Label(candidates, text="Audit size:").grid(row=1, column=0, sticky="e", padx=(8, 6), pady=(0, 8))
        ttk.Spinbox(candidates, textvariable=self.var_audit_size, from_=1, to=99999, width=8).grid(row=1, column=1, sticky="w", pady=(0, 8))
        ttk.Label(candidates, text="Minimum valid:").grid(row=1, column=2, sticky="e", padx=(12, 6), pady=(0, 8))
        ttk.Spinbox(candidates, textvariable=self.var_audit_min_valid, from_=1, to=99999, width=8).grid(row=1, column=3, sticky="w", pady=(0, 8))
        ttk.Label(candidates, text="Audit seed:").grid(row=1, column=4, sticky="e", padx=(12, 6), pady=(0, 8))
        ttk.Spinbox(candidates, textvariable=self.var_audit_seed, from_=0, to=2147483647, width=12).grid(row=1, column=5, sticky="w", padx=(0, 8), pady=(0, 8))

        ttk.Label(candidates, text="Min graph gap:").grid(row=2, column=0, sticky="e", padx=(8, 6), pady=(0, 8))
        ttk.Spinbox(candidates, textvariable=self.var_min_gap, from_=0, to=10000, width=8).grid(row=2, column=1, sticky="w", pady=(0, 8))
        ttk.Label(candidates, text="Search pool:").grid(row=2, column=2, sticky="e", padx=(12, 6), pady=(0, 8))
        ttk.Entry(candidates, textvariable=self.var_pool, width=12).grid(row=2, column=3, sticky="w", pady=(0, 8))
        single_part2 = ttk.Checkbutton(
            candidates,
            text="Part-2 input count = 1",
            variable=self.var_require_single_part2_input,
            command=self._update_run_summary,
        )
        single_part2.grid(row=2, column=4, columnspan=2, sticky="w", padx=(12, 8), pady=(0, 8))
        attach_tooltip(
            single_part2,
            "Filtert den Auswahlpool vor Shortlist und Stratifikation auf Part-2-Modelle mit genau einem externen Eingang.\n"
            "Der Generic Runner selbst bleibt für mehrere Ein- und Ausgänge ausgelegt.",
        )
        ttk.Label(
            candidates,
            text="Der Filter kann die YOLO-Auswahl stark verkleinern; geeignete Fälle werden aus dem restlichen Pool nachgefüllt.",
            foreground="#666",
        ).grid(row=3, column=0, columnspan=6, sticky="w", padx=(8, 8), pady=(0, 8))

        ttk.Checkbutton(candidates, text="Backendweise nachrücken gemäß angezeigter Policy",
            variable=self.var_backend_backfill, command=self._update_run_summary).grid(row=4, column=0, columnspan=6, sticky="w", padx=8)
        labels = {'max_candidates_per_backend': 'Kandidaten je Backend', 'max_cold_builds': 'Kaltbuilds je Modell',
                  'max_hailo_part1_builds': 'Hailo Part1 Starts', 'max_trt_part2_builds': 'TRT Part2 Starts',
                  'hailo_build_timeout_s': 'Hailo Sekunden', 'trt_build_timeout_s': 'TRT Sekunden'}
        for index, (key, variable) in enumerate(self.var_backfill_limits.items()):
            row, column = 5 + index // 3, (index % 3) * 2
            ttk.Label(candidates, text=labels[key]).grid(row=row, column=column, sticky="e", padx=8)
            ttk.Spinbox(candidates, textvariable=variable, from_=0, to=99999, width=8,
                command=self._update_run_summary).grid(row=row, column=column+1, sticky="w")

        targets = ttk.LabelFrame(tab, text="Hardware run profiles")
        targets.grid(row=1, column=0, sticky="ew", padx=8, pady=6)
        checks = [
            ("ORT CPU", self.var_run_cpu), ("ORT CUDA", self.var_run_cuda), ("TensorRT Full", self.var_run_trt),
            ("Hailo-8 Full", self.var_run_hailo), ("Hailo-8 → TensorRT", self.var_run_hailo_to_trt), ("TensorRT → Hailo-8", self.var_run_trt_to_hailo),
            ("Hailo-10H Full", self.var_run_hailo10), ("Hailo-10H → TensorRT", self.var_run_hailo10_to_trt),
            ("DEEPX Full", self.var_run_deepx), ("DEEPX → TensorRT", self.var_run_deepx_to_trt), ("TensorRT → DEEPX", self.var_run_trt_to_deepx),
        ]
        for i, (text, var) in enumerate(checks):
            ttk.Checkbutton(targets, text=text, variable=var, command=self._update_run_summary).grid(row=i // 3, column=i % 3, sticky="w", padx=12, pady=7)
        ttk.Label(targets, text="DeepX CLS preprocessing:").grid(
            row=4, column=0, sticky="w", padx=12, pady=(8, 4)
        )
        deepx_preprocessing = ttk.Combobox(
            targets,
            textvariable=self.var_deepx_classification_preprocessing,
            values=["imagenet_mean_std", "current_scale_only"],
            state="readonly",
            width=24,
        )
        self.deepx_legacy_notice_label = ttk.Label(
            targets, textvariable=self.var_deepx_legacy_notice,
            foreground="#a04b00", wraplength=1050, justify="left",
        )
        self.deepx_legacy_notice_label.grid(
            row=5, column=0, columnspan=3, sticky="ew", padx=12, pady=(2, 6)
        )
        ttk.Label(targets, text="DeepX compiler overlay (optional):").grid(row=6, column=0, sticky="w", padx=12)
        ttk.Entry(targets, textvariable=self.var_deepx_compiler_overlay, width=55).grid(row=6, column=1, columnspan=2, sticky="ew", padx=12)
        deepx_preprocessing.grid(
            row=4, column=1, sticky="w", padx=12, pady=(8, 4)
        )
        attach_tooltip(
            deepx_preprocessing,
            "ImageNet mean/std is the default for newly created profiles. "
            "current_scale_only preserves legacy profiles that did not name "
            "this DeepX classification preprocessing axis.",
        )
        self.hardware_profile_explanation_label = ttk.Label(targets, text=(
            "Remote Setup, SSH, Runtime-Venv, Build-Environment und u.RECS-Kanal werden aus dem zentralen "
            f"Hardware Run Profile abgeleitet ({default_hardware_setups_file()})."
        ), foreground="#555", wraplength=1050, justify="left")
        self.hardware_profile_explanation_label.grid(
            row=7, column=0, columnspan=3, sticky="ew", padx=12, pady=(4, 10)
        )

    def _build_run_summary_tab(self, tab: ttk.Frame) -> None:
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(0, weight=1)
        self.run_summary_text = scrolledtext.ScrolledText(tab, wrap="word", font=("TkFixedFont", 10))
        self.run_summary_text.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        self.run_summary_text.configure(state="disabled")
        actions = ttk.Frame(tab)
        actions.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 8))
        ttk.Button(actions, text="Run modes konfigurieren…", command=self._open_run_mode_settings).pack(side=tk.LEFT)
        ttk.Button(actions, text="Zusammenfassung aktualisieren", command=self._update_run_summary).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(actions, text="Effektive YAML anzeigen", command=self._preview_yaml).pack(side=tk.LEFT, padx=(8, 0))

    def _open_run_mode_settings(self) -> None:
        win = tk.Toplevel(self)
        win.title("Tool Config — Run modes")
        win.geometry("1180x820")
        win.minsize(960, 680)
        frame = ttk.Frame(win)
        frame.pack(fill="both", expand=True)
        build_run_modes_panel(frame, app=self.app)
        ttk.Button(win, text="Schließen", command=lambda: (win.destroy(), self._on_run_mode_changed())).pack(pady=(0, 8))

    def _on_energy_master_toggle(self) -> None:
        enabled = bool(self.var_energy_enabled.get())
        self.var_native_energy_enabled.set(bool(enabled and self.var_native_enabled.get()))
        self.var_native_energy_mode.set("measure" if enabled and self.var_native_enabled.get() else "plan")
        self._update_run_summary()

    def _on_run_mode_changed(self, *_args: Any) -> None:
        mode_id = normalize_mode_id(self.var_run_mode_id.get())
        if self.var_run_mode_id.get() != mode_id:
            self.var_run_mode_id.set(mode_id)
            return
        try:
            follow = parse_config_bool(
                self.var_run_mode_follow_tool_config.get(), field="execution_preset.follow_tool_config"
            )
            if not follow:
                previous = getattr(self, "_loaded_run_mode_preset", {})
                if normalize_mode_id(previous.get("id")) != mode_id or not previous.get("snapshot"):
                    raise ValueError("run_mode_snapshot_missing:execution_preset.snapshot")
                mode = copy.deepcopy(previous["snapshot"])
                self.var_run_mode_summary.set(f"{mode_id} / gebundener Profilsnapshot")
            else:
                config = load_run_modes_config()
                mode = get_run_mode(mode_id, config)
                self.var_run_mode_summary.set(mode_summary(mode_id, config))
            if not self._loading_profile:
                previous = getattr(self, "_native_budget_widget_baseline", {})
                current_mode = (mode.get("runtime") or {}).get("native") or {}
                for field in ("frames", "warmup", "repetitions"):
                    var = getattr(self, "var_native_" + field, None)
                    if var is not None and (field not in previous or int(var.get()) == previous[field]):
                        var.set(int(current_mode.get(field, 1)))
                self._native_budget_widget_baseline = {
                    k: int(current_mode.get(k, 1)) for k in ("frames", "warmup", "repetitions")}
                quality_defaults = mode.get("quality") if isinstance(mode.get("quality"), Mapping) else {}
                hold_cfg = mode.get("holdout") if isinstance(mode.get("holdout"), Mapping) else {}
                quality_location = str(
                    (quality_defaults or {}).get("execution_location")
                    or "central_management"
                )
                self.var_quality_execution_location.set(quality_location)
                self.var_quality_workers.set(
                    max(
                        1,
                        int((quality_defaults or {}).get("workers") or 4),
                    )
                )
                # Audit size, minimum-valid count and seed are explicit
                # selection decisions in the Evaluation Profile.  A run-mode
                # change may refresh execution/quality defaults, but must not
                # silently replace those profile-owned values with the
                # central mode's hold-out defaults.
                if self._selected_model_index is None:
                    quality_cfg = mode.get("quality") if isinstance(mode.get("quality"), Mapping) else {}
                    self.var_model_validation_tier.set(
                        str((quality_cfg or {}).get("dataset_tier") or "screening")
                    )
                    is_holdout_draft = (
                        _normalize_model_usage(self.var_model_usage.get())
                        == MODEL_USAGE_HOLDOUT
                    )
                    self.var_model_universe_mode.set(
                        str((hold_cfg or {}).get("candidate_universe") or "deterministic_audit")
                        if is_holdout_draft
                        else "declared_shortlist"
                    )
                    self.var_model_universe_complete.set(
                        bool((hold_cfg or {}).get("require_complete_candidate_universe", False))
                        if is_holdout_draft
                        else False
                    )
        except Exception as exc:
            self.var_run_mode_summary.set(f"Run-mode configuration could not be loaded: {type(exc).__name__}: {exc}")
        self._update_run_summary()

    def _update_run_summary(self, *_args: Any) -> None:
        summary = "Effektive Moduskonfiguration noch nicht verfügbar."
        targets = [str(row.get("id") or "") for row in self._run_profiles()]
        model_lines = []
        for row in self.models:
            model_lines.append(f"  - {row.get('id','?')} · task={row.get('task','auto')} · role={row.get('evaluation_role','development')}")
        try:
            ds = registry_status(str(default_registry_path()), verify_manifests=False)
            ds_ready = bool(ds.get("ready_for_final_profile") or ds.get("ok"))
            dataset_line = f"Dataset registry: {'ready' if ds_ready else 'incomplete'} · {default_registry_path()}"
        except Exception:
            dataset_line = f"Dataset registry: {default_registry_path()}"
        try:
            effective_profile = self._build_payload()
            summary = run_mode_profile_brief(effective_profile)
            _plan = build_effective_execution_plan(effective_profile)
            _plan_text = execution_plan_text(_plan)
        except Exception as _plan_exc:
            _plan_text = f"Execution plan not yet available: {type(_plan_exc).__name__}: {_plan_exc}"
            summary = _plan_text
        text = (
            f"RUN MODE\n{summary}\n\n"
            f"RUN OVERRIDES\n  Native Runner: {'enabled' if self.var_native_enabled.get() else 'disabled'}\n"
            f"  Energy measurement: {'enabled' if self.var_energy_enabled.get() else 'disabled'}\n\n"
            f"MODELS ({len(self.models)})\n"
            + ("\n".join(model_lines) if model_lines else "  - none")
            + "\n\n"
            + f"BACKEND-NACHRÜCKEN: {self.var_backend_backfill.get()} · {'Build + bewiesener Scorekollaps' if self.backend_output_contract_version == 1 else 'nur Buildausschlüsse'} · Outputvertrag v{self.backend_output_contract_version} · { {key: value.get() for key, value in self.var_backfill_limits.items()} }\n\n"
            + f"CANDIDATES\n  cases/model={self.var_cases.get()} · shortlist={self.var_shortlist.get()} · strategy={self.var_selection_strategy.get()} · audit={self.var_audit_size.get()}/{self.var_audit_min_valid.get()} · Part-2 inputs=1={'on' if self.var_require_single_part2_input.get() else 'off'}\n\n"
            + f"RUN PROFILES ({len(targets)})\n  "
            + (", ".join(targets) if targets else "none")
            + "\n\n"
            + f"DEEPX CLASSIFICATION PREPROCESSING\n  {self.var_deepx_classification_preprocessing.get()}\n\n"
            + f"CENTRAL CONFIGURATION\n  {dataset_line}\n  Hardware registry: {default_hardware_setups_file()}\n"
            + f"  Run-mode registry: {default_run_modes_path()}\n\n"
            + "EFFECTIVE EXECUTION PLAN\n" + _plan_text + "\n\n"
            + "The saved YAML contains a resolved mode snapshot for reproducibility. Detailed fields are not edited per run."
        )
        self.var_run_mode_summary.set(summary)
        box = getattr(self, "run_summary_text", None)
        if box is not None:
            try:
                box.configure(state="normal")
                box.delete("1.0", "end")
                box.insert("1.0", text)
                box.configure(state="disabled")
            except Exception:
                pass

    def _build_general_tab(self, tab: ttk.Frame) -> None:
        tab.columnconfigure(1, weight=1)
        row = 0
        self._label(tab, "Profil-Datei:", "Pfad der YAML-Datei. Eingebaute Profile werden nicht überschrieben; speichere eigene Profile als Datei.").grid(row=row, column=0, sticky="w", padx=(8, 6), pady=8)
        ttk.Entry(tab, textvariable=self.var_path).grid(row=row, column=1, sticky="ew", padx=(0, 8), pady=8)
        ttk.Button(tab, text="Durchsuchen…", command=self._choose_profile_path).grid(row=row, column=2, padx=(0, 8), pady=8)
        row += 1
        self._label(tab, "Name / Profile ID:", "Stabile ID des Profils, z. B. thesis_final_eval_v1. Diese ID erscheint später im run_manifest.json.").grid(row=row, column=0, sticky="w", padx=(8, 6), pady=(0, 8))
        ttk.Entry(tab, textvariable=self.var_name).grid(row=row, column=1, columnspan=2, sticky="ew", padx=(0, 8), pady=(0, 8))
        row += 1
        self._label(tab, "Beschreibung:", "Kurze Beschreibung, warum dieses Profil existiert und was es evaluiert.").grid(row=row, column=0, sticky="nw", padx=(8, 6), pady=(0, 8))
        ttk.Entry(tab, textvariable=self.var_purpose).grid(row=row, column=1, columnspan=2, sticky="ew", padx=(0, 8), pady=(0, 8))
        row += 1
        self._label(tab, "Models root hint:", "Optionaler Hinweis auf den Ordner mit ONNX-Modellen. Der Workflow-Tab kann zusätzlich einen Models root setzen.").grid(row=row, column=0, sticky="w", padx=(8, 6), pady=(0, 8))
        ttk.Entry(tab, textvariable=self.var_models_root_hint).grid(row=row, column=1, sticky="ew", padx=(0, 8), pady=(0, 8))
        ttk.Button(tab, text="Ordner…", command=lambda: self._browse_dir_var(self.var_models_root_hint)).grid(row=row, column=2, padx=(0, 8), pady=(0, 8))
        row += 1
        self._label(tab, "Reserve-Modelle:", "Optionale Reserve-Modelle als kommagetrennte IDs. Sie laufen nur mit Include reserve.").grid(row=row, column=0, sticky="w", padx=(8, 6), pady=(0, 8))
        ttk.Entry(tab, textvariable=self.var_reserve).grid(row=row, column=1, columnspan=2, sticky="ew", padx=(0, 8), pady=(0, 8))
        row += 1
        help_text = tk.Text(tab, height=12, wrap="word", bg=self.cget("background"), relief="flat")
        help_text.grid(row=row, column=0, columnspan=3, sticky="nsew", padx=8, pady=(8, 8))
        help_text.insert(
            "1.0",
            "Normaler Ablauf:\n"
            "1) Modelle im Tab 'Modelle' eintragen.\n"
            "2) Ziele/Kandidaten wählen.\n"
            "3) Speichern & verwenden.\n"
            "4) Im Workflow-Tab Start Evaluation Workflow drücken.\n\n"
            "Wichtig: Run name / Run ID ist kein Pflichtfeld. Leer lassen = automatischer Name aus Profile ID + Zeitstempel. "
            "Das Results Bundle ist das erzeugte Run-Verzeichnis unter EvaluationRuns.\n",
        )
        help_text.configure(state="disabled")
        tab.rowconfigure(row, weight=1)

    def _build_models_tab(self, tab: ttk.Frame) -> None:
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(0, weight=1)
        columns = ("id", "task", "role", "scope", "tier", "universe", "onnx", "family_id")
        tree = ttk.Treeview(tab, columns=columns, show="headings", height=9)
        self.model_tree = tree
        headings = {
            "id": "Model ID",
            "task": "Task",
            "role": "Evaluation role",
            "scope": "Generalization scope",
            "tier": "Validation tier",
            "universe": "Candidate universe",
            "onnx": "ONNX path",
            "family_id": "Family ID",
        }
        widths = {"id": 120, "task": 90, "role": 100, "scope": 150, "tier": 90, "universe": 135, "onnx": 260, "family_id": 105}
        for col in columns:
            tree.heading(col, text=headings[col])
            tree.column(col, width=widths[col], anchor="w")
        tree.grid(row=0, column=0, columnspan=4, sticky="nsew", padx=8, pady=(8, 6))
        scrollbar = ttk.Scrollbar(tab, orient="vertical", command=tree.yview)
        scrollbar.grid(row=0, column=4, sticky="ns", pady=(8, 6))
        tree.configure(yscrollcommand=scrollbar.set)
        tree.bind("<<TreeviewSelect>>", self._on_model_selected, add="+")

        form = ttk.LabelFrame(tab, text="Modelleintrag")
        form.grid(row=1, column=0, columnspan=5, sticky="ew", padx=8, pady=(0, 8))
        for col in (1, 3, 5):
            form.columnconfigure(col, weight=1)
        self._label(form, "ID:", "Stabile Model-ID, z. B. resnet50 oder yolo26m. Wenn leer, wird sie aus dem ONNX-Dateinamen abgeleitet.").grid(row=0, column=0, sticky="w", padx=(8, 6), pady=6)
        ttk.Entry(form, textvariable=self.var_model_id).grid(row=0, column=1, sticky="ew", padx=(0, 8), pady=6)
        self._label(form, "Task:", "classification, detection oder auto.").grid(row=0, column=2, sticky="w", padx=(0, 6), pady=6)
        ttk.Combobox(form, textvariable=self.var_model_task, values=["auto", "classification", "detection"], state="readonly", width=14).grid(row=0, column=3, sticky="w", padx=(0, 8), pady=6)
        self._label(form, "Family:", "Optional: Modellfamilie wie resnet, yolo11, yolo26, mobilenet.").grid(row=0, column=4, sticky="w", padx=(0, 6), pady=6)
        ttk.Entry(form, textvariable=self.var_model_family).grid(row=0, column=5, sticky="ew", padx=(0, 8), pady=6)

        self._label(form, "ONNX:", "Optionaler direkter ONNX-Pfad. Wenn leer, sucht der Workflow im Models folder nach der ID.").grid(row=1, column=0, sticky="w", padx=(8, 6), pady=6)
        ttk.Entry(form, textvariable=self.var_model_onnx).grid(row=1, column=1, columnspan=4, sticky="ew", padx=(0, 8), pady=6)
        ttk.Button(form, text="ONNX…", command=self._browse_model_onnx).grid(row=1, column=5, sticky="w", padx=(0, 8), pady=6)

        self._label(form, "Input shape:", "Optional, z. B. 1x3x224x224 oder 1x3x640x640.").grid(row=2, column=0, sticky="w", padx=(8, 6), pady=6)
        ttk.Entry(form, textvariable=self.var_model_shape).grid(row=2, column=1, sticky="ew", padx=(0, 8), pady=6)
        self._label(form, "Dataset:", "Semantisches Dataset, z. B. imagenet_val oder coco2017_val.").grid(row=2, column=2, sticky="w", padx=(0, 6), pady=6)
        ttk.Entry(form, textvariable=self.var_model_dataset).grid(row=2, column=3, sticky="ew", padx=(0, 8), pady=6)
        self._label(form, "Preset:", "Validation preset/development subset, z. B. imagenet_val_mini_200 oder coco_50.").grid(row=2, column=4, sticky="w", padx=(0, 6), pady=6)
        ttk.Entry(form, textvariable=self.var_model_subset).grid(row=2, column=5, sticky="ew", padx=(0, 8), pady=6)

        self._label(form, "Source:", "Optional: torchvision, ultralytics, custom, ...").grid(row=3, column=0, sticky="w", padx=(8, 6), pady=6)
        ttk.Entry(form, textvariable=self.var_model_source).grid(row=3, column=1, sticky="ew", padx=(0, 8), pady=6)
        self._label(form, "Family ID:", "Stabile wissenschaftliche Familien-ID, z. B. yolo26 oder regnet. Entscheidend für Family-Hold-outs.").grid(row=3, column=2, sticky="w", padx=(0, 6), pady=6)
        ttk.Entry(form, textvariable=self.var_model_family_id).grid(row=3, column=3, sticky="ew", padx=(0, 8), pady=6)
        self._label(form, "Generalization:", "development, model_family_holdout oder within_family_transfer.").grid(row=3, column=4, sticky="w", padx=(0, 6), pady=6)
        ttk.Combobox(form, textvariable=self.var_model_generalization_scope, values=["development", "model_family_holdout", "within_family_transfer", "stress_test"], state="readonly", width=22).grid(row=3, column=5, sticky="w", padx=(0, 8), pady=6)

        self._label(form, "Evaluation role:", "development rows may fit models; holdout rows are never used for fitting.").grid(row=4, column=0, sticky="w", padx=(8,6), pady=4)
        ttk.Combobox(form, textvariable=self.var_model_role, values=["development", CONFIRMATORY_HOLDOUT_ROLE], state="readonly", width=22).grid(row=4, column=1, sticky="w", pady=4)
        self._label(form, "Validation tier:", "screening or final dataset gate.").grid(row=4, column=2, sticky="w", padx=(0,6), pady=4)
        ttk.Combobox(form, textvariable=self.var_model_validation_tier, values=["screening","final"], state="readonly", width=12).grid(row=4, column=3, sticky="w", pady=4)
        self._label(form, "Model SHA-256:", "Required for final hold-outs; computed from the exact ONNX file.").grid(row=4, column=4, sticky="w", padx=(0,6), pady=4)
        hrow=ttk.Frame(form); hrow.grid(row=4,column=5,sticky="ew",padx=(0,8),pady=4); hrow.columnconfigure(0,weight=1)
        ttk.Entry(hrow,textvariable=self.var_model_sha256).grid(row=0,column=0,sticky="ew")
        ttk.Button(hrow,text="Hash",command=self._hash_current_model).grid(row=0,column=1,padx=(6,0))

        self._label(form, "Hold-out group:", "e.g. classification_holdout or detection_holdout.").grid(row=5,column=0,sticky="w",padx=(8,6),pady=4)
        ttk.Entry(form,textvariable=self.var_model_holdout_group).grid(row=5,column=1,sticky="ew",pady=4)
        self._label(form, "Candidate universe:", "all_feasible for exhaustive hold-out; deterministic_audit for score-independent bounded audit.").grid(row=5,column=2,sticky="w",padx=(0,6),pady=4)
        ttk.Combobox(form,textvariable=self.var_model_universe_mode,values=["all_feasible","deterministic_audit","audit_universe","declared_shortlist"],state="readonly",width=20).grid(row=5,column=3,sticky="w",pady=4)
        ttk.Checkbutton(form,text="Universe complete/declaratively frozen",variable=self.var_model_universe_complete).grid(row=5,column=4,columnspan=2,sticky="w",padx=(8,8),pady=4)

        self._label(form,"Audit budget:","Audit size, minimum valid candidates and seed are configured globally on the Targets tab.").grid(row=6,column=0,columnspan=2,sticky="w",padx=(8,6),pady=4)
        ttk.Checkbutton(form,text="Genuinely unseen",variable=self.var_model_unseen).grid(row=6,column=2,sticky="w",padx=(0,6),pady=4)
        self._label(form,"Attested by / at:","Named prospective attestation; set before opening hold-out measurements.").grid(row=6,column=3,sticky="e",padx=(0,6),pady=4)
        atf=ttk.Frame(form); atf.grid(row=6,column=4,columnspan=2,sticky="ew",padx=(0,8),pady=4); atf.columnconfigure(0,weight=1); atf.columnconfigure(1,weight=1)
        ttk.Entry(atf,textvariable=self.var_model_attested_by).grid(row=0,column=0,sticky="ew"); ttk.Entry(atf,textvariable=self.var_model_attested_at).grid(row=0,column=1,sticky="ew",padx=(6,0)); ttk.Button(atf,text="Now",command=self._attest_model_now).grid(row=0,column=2,padx=(6,0))
        self._label(form,"Attestation note:","Explain why this model remained untouched before the freeze.").grid(row=7,column=0,sticky="w",padx=(8,6),pady=(4,6))
        ttk.Entry(form,textvariable=self.var_model_attestation_note).grid(row=7,column=1,columnspan=5,sticky="ew",padx=(0,8),pady=(4,6))
        self._label(form, "Model note:", "Kurze Begründung, warum das Modell im Profil ist.").grid(row=8, column=0, sticky="w", padx=(8, 6), pady=(4, 6))
        ttk.Entry(form, textvariable=self.var_model_note).grid(row=8, column=1, columnspan=5, sticky="ew", padx=(0, 8), pady=(4, 6))

        btns = ttk.Frame(tab)
        btns.grid(row=2, column=0, columnspan=5, sticky="ew", padx=8, pady=(0, 8))
        ttk.Button(btns, text="Hinzufügen / Aktualisieren", command=self._upsert_model).pack(side=tk.LEFT)
        ttk.Button(btns, text="Entfernen", command=self._remove_selected_model).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(btns, text="Aus aktuellem GUI-Modell", command=self._add_current_gui_model).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(btns, text="Felder leeren", command=self._clear_model_fields).pack(side=tk.LEFT, padx=(8, 0))

    def _build_targets_tab(self, tab: ttk.Frame) -> None:
        tab.columnconfigure(1, weight=1)
        sel = ttk.LabelFrame(tab, text="Kandidatenauswahl")
        sel.grid(row=0, column=0, sticky="ew", padx=8, pady=8)
        for c in (1, 3):
            sel.columnconfigure(c, weight=1)
        self._label(sel, "Accepted cases:", "Wie viele Splitfälle pro Modell final benchmarkfähig erzeugt werden sollen.").grid(row=0, column=0, sticky="w", padx=(8, 6), pady=6)
        ttk.Spinbox(sel, from_=1, to=999, textvariable=self.var_cases, width=8).grid(row=0, column=1, sticky="w", padx=(0, 12), pady=6)
        self._label(sel, "Candidate shortlist:", "Wie viele vorhergesagte Top-Kandidaten aus prediction.json in die engere Wahl kommen.").grid(row=0, column=2, sticky="w", padx=(0, 6), pady=6)
        ttk.Spinbox(sel, from_=1, to=999, textvariable=self.var_shortlist, width=8).grid(row=0, column=3, sticky="w", padx=(0, 8), pady=6)
        self._label(sel, "Min gap:", "Mindestabstand zwischen Splitkandidaten, um fast identische Cuts zu vermeiden.").grid(row=1, column=0, sticky="w", padx=(8, 6), pady=6)
        ttk.Spinbox(sel, from_=0, to=999, textvariable=self.var_min_gap, width=8).grid(row=1, column=1, sticky="w", padx=(0, 12), pady=6)
        self._label(sel, "Search pool:", "auto oder Zahl. Größer = mehr Kandidaten werden betrachtet, aber teurer.").grid(row=1, column=2, sticky="w", padx=(0, 6), pady=6)
        ttk.Entry(sel, textvariable=self.var_pool, width=10).grid(row=1, column=3, sticky="w", padx=(0, 8), pady=6)
        ttk.Checkbutton(sel, text="Partial Hailo cases behalten", variable=self.var_keep_partial_hailo).grid(row=2, column=0, columnspan=2, sticky="w", padx=(8, 12), pady=6)
        ttk.Checkbutton(sel, text="Blocked/infeasible Fälle berichten", variable=self.var_report_blocked).grid(row=2, column=2, columnspan=2, sticky="w", padx=(0, 8), pady=6)
        self._label(sel, "Selection strategy:", "score_independent_audit = 20 deterministische, scoreblinde Auditfälle plus normale Predictor-Shortlist; stratified_windows und predicted_topk bleiben Entwicklungsoptionen.").grid(row=3, column=0, sticky="w", padx=(8, 6), pady=6)
        ttk.Combobox(sel, textvariable=self.var_selection_strategy, values=["score_independent_audit", "stratified_windows", "predicted_topk"], state="readonly", width=24).grid(row=3, column=1, sticky="w", padx=(0, 8), pady=6)
        self._label(sel, "Full-Hailo preflight:", "enabled = Full-Hailo Parser/Preflight früher prüfen; skip = schneller, Hailo-Build/Reuse entscheidet später.").grid(row=3, column=2, sticky="w", padx=(0, 6), pady=6)
        ttk.Combobox(sel, textvariable=self.var_full_preflight, values=["enabled", "skip"], state="readonly", width=10).grid(row=3, column=3, sticky="w", padx=(0, 8), pady=6)
        self._label(sel, "Audit size / min valid:", "Auditumfang und Mindestzahl quality-freigegebener Kandidaten je Ranking-Stratum.").grid(row=4, column=0, sticky="w", padx=(8, 6), pady=6)
        audit_counts = ttk.Frame(sel)
        audit_counts.grid(row=4, column=1, sticky="w", padx=(0, 8), pady=6)
        ttk.Spinbox(audit_counts, from_=1, to=999, textvariable=self.var_audit_size, width=6).pack(side=tk.LEFT)
        ttk.Label(audit_counts, text=" / ").pack(side=tk.LEFT)
        ttk.Spinbox(audit_counts, from_=1, to=999, textvariable=self.var_audit_min_valid, width=6).pack(side=tk.LEFT)
        self._label(sel, "Audit seed:", "Reproduzierbarer Seed; die Auswahl verwendet keine Predictor-Scores und keine Messwerte.").grid(row=4, column=2, sticky="w", padx=(0, 6), pady=6)
        ttk.Entry(sel, textvariable=self.var_audit_seed, width=12).grid(row=4, column=3, sticky="w", padx=(0, 8), pady=6)

        runs = ttk.LabelFrame(tab, text="Hardware-/Run-Profile")
        runs.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 8))
        ttk.Checkbutton(runs, text="CPU ORT reference", variable=self.var_run_cpu).grid(row=0, column=0, sticky="w", padx=8, pady=6)
        ttk.Checkbutton(runs, text="CUDA ORT full/split", variable=self.var_run_cuda).grid(row=0, column=1, sticky="w", padx=8, pady=6)
        ttk.Checkbutton(runs, text="TensorRT full/split", variable=self.var_run_trt).grid(row=0, column=2, sticky="w", padx=8, pady=6)
        ttk.Checkbutton(runs, text="Hailo8 full/split", variable=self.var_run_hailo).grid(row=1, column=0, sticky="w", padx=8, pady=6)
        ttk.Checkbutton(runs, text="Hailo8 → TensorRT", variable=self.var_run_hailo_to_trt).grid(row=1, column=1, sticky="w", padx=8, pady=6)
        ttk.Checkbutton(runs, text="TensorRT → Hailo8", variable=self.var_run_trt_to_hailo).grid(row=1, column=2, sticky="w", padx=8, pady=6)
        ttk.Checkbutton(runs, text="Hailo10 full/split", variable=self.var_run_hailo10).grid(row=2, column=0, sticky="w", padx=8, pady=6)
        ttk.Checkbutton(runs, text="Hailo10 → TensorRT", variable=self.var_run_hailo10_to_trt).grid(row=2, column=1, sticky="w", padx=8, pady=6)
        ttk.Checkbutton(runs, text="DeepX DX-M1 full", variable=self.var_run_deepx).grid(row=3, column=0, sticky="w", padx=8, pady=6)
        ttk.Checkbutton(runs, text="DeepX DX-M1 → TensorRT", variable=self.var_run_deepx_to_trt).grid(row=3, column=1, sticky="w", padx=8, pady=6)
        ttk.Checkbutton(runs, text="TensorRT → DeepX DX-M1", variable=self.var_run_trt_to_deepx).grid(row=3, column=2, sticky="w", padx=8, pady=6)
        self._label(
            runs,
            "DeepX CLS preprocessing:",
            "imagenet_mean_std ist der Default für neue Profile; "
            "current_scale_only erhält das Verhalten älterer Profile.",
        ).grid(row=4, column=0, sticky="w", padx=8, pady=6)
        ttk.Combobox(
            runs,
            textvariable=self.var_deepx_classification_preprocessing,
            values=["imagenet_mean_std", "current_scale_only"],
            state="readonly",
            width=24,
        ).grid(row=4, column=1, sticky="w", padx=8, pady=6)
        self._tip(runs, "Diese Auswahl erzeugt die run_profiles im YAML. Der formale Workflow nutzt daraus später die Benchmark-/Hailo-/Remote-Planung. DeepX nutzt die Build-Umgebungen aus dem Hardware-Tab / build_environments.")

        hwsel = ttk.LabelFrame(tab, text="Remote-Hardware-Setups aus Tool Config")
        hwsel.grid(row=2, column=0, sticky="ew", padx=8, pady=(0, 8))
        for c in range(3):
            hwsel.columnconfigure(c, weight=1)
        ttk.Label(
            hwsel,
            text=(
                "Remote-Host, Runtime-venv und Base-Dir werden zentral im Tool-Config-Tab pro Hardware-Setup gepflegt. "
                "Dieses Profil wählt nur noch aus, welche Setups für EvaluationRuns verwendet werden."
            ),
            foreground="#555",
            wraplength=940,
            justify="left",
        ).grid(row=0, column=0, columnspan=3, sticky="ew", padx=8, pady=(8, 4))

        def _central_setup_card(col: int, setup_id: str, title: str, enabled: tk.BooleanVar, run_flag: tk.BooleanVar) -> None:
            card = ttk.LabelFrame(hwsel, text=title)
            card.grid(row=1, column=col, sticky="nsew", padx=(8 if col == 0 else 4, 8 if col == 2 else 4), pady=8)
            card.columnconfigure(0, weight=1)
            self._check(card, "Setup verwenden", enabled, f"{setup_id} aus Tool Config in die Hardware-Matrix des Profils aufnehmen.").grid(row=0, column=0, sticky="w", padx=8, pady=(8, 2))
            ttk.Label(card, text=setup_id, foreground="#555").grid(row=1, column=0, sticky="w", padx=8, pady=(0, 2))
            summary = "Configure remote host/venv in Tool Config."
            try:
                if self.app is not None and hasattr(self.app, "_hardware_setup_summary"):
                    summary = self.app._hardware_setup_summary(setup_id)
            except Exception:
                pass
            ttk.Label(card, text=summary, wraplength=280, foreground="#555", justify="left").grid(row=2, column=0, sticky="ew", padx=8, pady=(0, 8))
            def _sync_from_run(*_args: object) -> None:
                if bool(run_flag.get()):
                    enabled.set(True)
            try:
                run_flag.trace_add("write", _sync_from_run)
            except Exception:
                pass

        _central_setup_card(0, "orin_nx_hailo8_01", "Orin NX + Hailo-8", self.var_hw_h8_enabled, self.var_run_hailo)
        _central_setup_card(1, "orin_nx_hailo10_01", "Orin NX + Hailo-10", self.var_hw_h10_enabled, self.var_run_hailo10)
        _central_setup_card(2, "orin_nx_deepx_m1_01", "Orin NX + DeepX DX-M1", self.var_hw_dx_enabled, self.var_run_deepx)

        legacy = ttk.LabelFrame(tab, text="Advanced: externe Hardware-Registry")
        legacy.grid(row=3, column=0, sticky="ew", padx=8, pady=(0, 8))
        legacy.columnconfigure(1, weight=1)
        self._label(legacy, "Selected setups:", "Optionaler manueller Override. Normalerweise leer lassen; die Auswahl oben schreibt zentrale Setup-IDs ins Profil.").grid(row=0, column=0, sticky="w", padx=(8, 6), pady=6)
        self._entry(legacy, self.var_hardware_selected_setups, "Optionaler Override; sonst aus den Karten ableiten.").grid(row=0, column=1, sticky="ew", padx=(0, 8), pady=6)
        self._label(legacy, "Selected groups:", "Optionale Gruppen aus hardware_setups.yaml.").grid(row=1, column=0, sticky="w", padx=(8, 6), pady=6)
        self._entry(legacy, self.var_hardware_selected_groups, "Optional, z. B. all_accelerators.").grid(row=1, column=1, sticky="ew", padx=(0, 8), pady=6)
        self._label(legacy, "Registry file:", "Optionaler Pfad zu hardware_setups.yaml. Leer = ~/.onnx_splitpoint_tool/hardware_setups.yaml.").grid(row=2, column=0, sticky="w", padx=(8, 6), pady=6)
        self._entry(legacy, self.var_hardware_setups_file, "Optional, normalerweise leer lassen.").grid(row=2, column=1, sticky="ew", padx=(0, 8), pady=6)

    def _build_validation_tab(self, tab: ttk.Frame) -> None:
        tab.columnconfigure(1, weight=1)
        self._label(tab, "Validation mode:", "summary_only = Ergebnisse zusammenfassen; strict = Validierungslücken härter markieren; disabled = Validierungsstufe überspringen.").grid(row=0, column=0, sticky="w", padx=(8, 6), pady=8)
        ttk.Combobox(tab, textvariable=self.var_validation_mode, values=["summary_only", "strict", "disabled"], state="readonly", width=14).grid(row=0, column=1, sticky="w", padx=(0, 8), pady=8)
        self._label(tab, "Reference mode:", "auto ist normalerweise richtig. Der Workflow wählt CPU/full oder passende Baseline als Referenz.").grid(row=1, column=0, sticky="w", padx=(8, 6), pady=(0, 8))
        ttk.Entry(tab, textvariable=self.var_reference_mode).grid(row=1, column=1, sticky="ew", padx=(0, 8), pady=(0, 8))
        self._label(tab, "Backend drift reference:", "Referenz für Backend-Drift, meistens cpu_full.").grid(row=2, column=0, sticky="w", padx=(8, 6), pady=(0, 8))
        ttk.Entry(tab, textvariable=self.var_backend_drift_reference).grid(row=2, column=1, sticky="ew", padx=(0, 8), pady=(0, 8))
        self._label(tab, "Classification metrics:", "Kommagetrennte Metriken. Beispiele: top1, top5, logit_cosine_similarity.").grid(row=3, column=0, sticky="w", padx=(8, 6), pady=(0, 8))
        ttk.Entry(tab, textvariable=self.var_classification_metrics).grid(row=3, column=1, sticky="ew", padx=(0, 8), pady=(0, 8))
        self._label(tab, "Detection metrics:", "Kommagetrennte Metriken. Beispiele: semantic_proxy_match_ratio, semantic_proxy_mean_iou, mini_coco_ap50.").grid(row=4, column=0, sticky="w", padx=(8, 6), pady=(0, 8))
        ttk.Entry(tab, textvariable=self.var_detection_metrics).grid(row=4, column=1, sticky="ew", padx=(0, 8), pady=(0, 8))
        val_checks = ttk.Frame(tab)
        val_checks.grid(row=5, column=1, sticky="w", padx=(0, 8), pady=(0, 8))
        ttk.Checkbutton(val_checks, text="Explicit validation evidence erforderlich", variable=self.var_require_explicit_validation).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Checkbutton(val_checks, text="Task-Metriken erforderlich", variable=self.var_require_task_metrics).pack(side=tk.LEFT)
        self._tip(val_checks, "Für finale YOLO-/Detection-Claims aktivieren: Zeilen ohne echte Task-Metriken werden dann nicht als ausreichend validiert behandelt.")
        rep = ttk.LabelFrame(tab, text="Report-Blöcke")
        rep.grid(row=6, column=0, columnspan=2, sticky="ew", padx=8, pady=8)
        ttk.Checkbutton(rep, text="Decision summary", variable=self.var_report_decisions).grid(row=0, column=0, sticky="w", padx=8, pady=6)
        ttk.Checkbutton(rep, text="Backend drift", variable=self.var_report_backend_drift).grid(row=0, column=1, sticky="w", padx=8, pady=6)
        ttk.Checkbutton(rep, text="Task-specific quality", variable=self.var_report_task_quality).grid(row=0, column=2, sticky="w", padx=8, pady=6)


    def _build_campaign_data_tab(self, tab: ttk.Frame) -> None:
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(0, weight=1)
        nb = ttk.Notebook(tab)
        nb.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        basics = ttk.Frame(nb); manifests = ttk.Frame(nb); quality = ttk.Frame(nb)
        nb.add(basics, text="Validation basics")
        nb.add(manifests, text="Dataset manifests")
        nb.add(quality, text="Task-quality gate")
        self._build_validation_tab(basics)

        manifests.columnconfigure(1, weight=1); manifests.columnconfigure(3, weight=1)
        ttk.Label(manifests, text=(
            "Finale Calibration/Validation-Datensätze werden über content-addressed Manifeste gebunden. "
            "Die Pfade können aus Tool Config → Final campaign data übernommen oder explizit eingetragen werden."
        ), foreground="#555", wraplength=1050, justify="left").grid(row=0, column=0, columnspan=4, sticky="ew", padx=8, pady=(8,4))
        self._label(manifests, "Dataset registry:", "Globale Registry aus Tool Config.").grid(row=1, column=0, sticky="e", padx=(8,6), pady=4)
        self._entry(manifests, self.var_dataset_registry, "Pfad zu dataset_registry.json.").grid(row=1, column=1, columnspan=2, sticky="ew", pady=4)
        self._button(manifests, "Aus Tool Config übernehmen", self._load_dataset_registry_into_profile, "Lädt die vier finalen Manifestpfade und COCO-Annotations aus der globalen Registry.").grid(row=1, column=3, sticky="w", padx=(6,8), pady=4)
        rows = [
            ("Classification calibration:", self.var_manifest_cls_calib, "Classification validation:", self.var_manifest_cls_val),
            ("Detection calibration:", self.var_manifest_det_calib, "Detection validation:", self.var_manifest_det_val),
            ("Pipeline contract:", self.var_pipeline_contract_manifest, "Energy calibration:", self.var_energy_calibration_manifest),
        ]
        for r, (l1,v1,l2,v2) in enumerate(rows, start=2):
            self._label(manifests, l1, "Manifest/contract used by campaign preflight.").grid(row=r, column=0, sticky="e", padx=(8,6), pady=4)
            self._entry(manifests, v1, "File path; relative paths are resolved against the profile file.").grid(row=r, column=1, sticky="ew", pady=4)
            self._label(manifests, l2, "Manifest/contract used by campaign preflight.").grid(row=r, column=2, sticky="e", padx=(12,6), pady=4)
            self._entry(manifests, v2, "File path; relative paths are resolved against the profile file.").grid(row=r, column=3, sticky="ew", padx=(0,8), pady=4)
        btns = ttk.Frame(manifests); btns.grid(row=5, column=0, columnspan=4, sticky="w", padx=8, pady=(6,8))
        self._button(btns, "Registry status", self._show_dataset_registry_status, "Shows final dataset and manifest readiness.").pack(side=tk.LEFT)
        self._button(btns, "Tool Config öffnen", self._open_tool_config_dataset_dialog, "Opens the provisioning/import dialog from Tool Config.").pack(side=tk.LEFT, padx=(8,0))
        self._check(
            btns,
            "Fehlende Manifestpfade beim Speichern/Ausführen automatisch aus Registry binden",
            self.var_auto_bind_dataset_registry,
            "Explicit profile paths remain authoritative; only empty fields are filled.",
        ).pack(side=tk.LEFT, padx=(16,0))

        quality.columnconfigure(1, weight=1); quality.columnconfigure(3, weight=1)
        self._label(quality, "Profile ID:", "Versionierte ID des Task-Quality-Profiles.").grid(row=0, column=0, sticky="e", padx=(8,6), pady=(8,4))
        self._entry(quality, self.var_quality_profile_id, "Stable policy ID stored in every result row.").grid(row=0, column=1, sticky="ew", pady=(8,4))
        self._check(quality, "Vor finaler Kampagne eingefroren", self.var_quality_frozen, "Must be true for final-mode evidence.").grid(row=0, column=2, columnspan=2, sticky="w", padx=(12,8), pady=(8,4))
        self._label(quality, "Dataset tier:", "screening rows are never final ranking evidence; final enables non-inferiority eligibility.").grid(row=1, column=0, sticky="e", padx=(8,6), pady=4)
        self._combo(quality, self.var_quality_tier, ["screening","final"], "Task-quality tier.", state="readonly", width=14).grid(row=1, column=1, sticky="w", pady=4)
        self._label(quality, "Canonical reference:", "Normally canonical_full_onnx.").grid(row=1, column=2, sticky="e", padx=(12,6), pady=4)
        self._entry(quality, self.var_quality_canonical_ref, "Reference row for total quality budget.").grid(row=1, column=3, sticky="ew", padx=(0,8), pady=4)
        margins = ttk.LabelFrame(quality, text="Non-inferiority margins (absolute points)")
        margins.grid(row=2, column=0, columnspan=4, sticky="ew", padx=8, pady=8)
        for c in (1,3,5): margins.columnconfigure(c, weight=1)
        items = [
            ("Top-1 pp", self.var_quality_cls_margin_pp), ("Top-5 pp", self.var_quality_cls_top5_margin_pp),
            ("COCO AP points", self.var_quality_det_margin_ap), ("AP50 points", self.var_quality_det_ap50_margin),
            ("AP75 points", self.var_quality_det_ap75_margin),
        ]
        for i,(label,var) in enumerate(items):
            r=i//3; c=(i%3)*2
            ttk.Label(margins,text=label+":").grid(row=r,column=c,sticky="e",padx=(8,6),pady=4)
            self._spin(margins,var,"Absolute percentage/AP points; GUI converts to 0..1 metric units.",from_=0,to=100,increment=0.1,width=9).grid(row=r,column=c+1,sticky="w",pady=4)
        stats = ttk.LabelFrame(quality, text="Paired bootstrap decision")
        stats.grid(row=3, column=0, columnspan=4, sticky="ew", padx=8, pady=(0,8))
        ttk.Label(stats,text="Confidence:").grid(row=0,column=0,sticky="e",padx=(8,6),pady=6)
        self._spin(stats,self.var_quality_confidence,"One-sided confidence level.",from_=0.5,to=0.9999,increment=0.01,width=8).grid(row=0,column=1,sticky="w",pady=6)
        ttk.Label(stats,text="Bootstrap repetitions:").grid(row=0,column=2,sticky="e",padx=(12,6),pady=6)
        self._spin(stats,self.var_quality_bootstrap,"5000 recommended for final tables.",from_=100,to=1000000,width=10).grid(row=0,column=3,sticky="w",pady=6)
        ttk.Label(stats,text="Seed:").grid(row=0,column=4,sticky="e",padx=(12,6),pady=6)
        self._spin(stats,self.var_quality_seed,"Frozen bootstrap seed.",from_=0,to=2147483647,width=12).grid(row=0,column=5,sticky="w",padx=(0,8),pady=6)
        ttk.Label(stats,text="Execution:").grid(row=1,column=0,sticky="e",padx=(8,6),pady=(0,6))
        self._combo(
            stats,
            self.var_quality_execution_location,
            ["local", "central_management"],
            "Final campaigns evaluate paired uncertainty on the management node.",
            state="readonly",
            width=20,
        ).grid(row=1,column=1,columnspan=2,sticky="w",pady=(0,6))
        ttk.Label(stats,text="Workers:").grid(row=1,column=3,sticky="e",padx=(12,6),pady=(0,6))
        self._spin(
            stats,
            self.var_quality_workers,
            "Management-side process workers; Final defaults to four.",
            from_=1,
            to=64,
            width=8,
        ).grid(row=1,column=4,sticky="w",pady=(0,6))

    def _build_holdout_reporting_tab(self, tab: ttk.Frame) -> None:
        tab.columnconfigure(0, weight=1); tab.rowconfigure(0, weight=1)
        nb = ttk.Notebook(tab); nb.grid(row=0,column=0,sticky="nsew",padx=8,pady=8)
        campaign = ttk.Frame(nb); ranking = ttk.Frame(nb); reporting = ttk.Frame(nb)
        nb.add(campaign,text="Campaign & hold-outs"); nb.add(ranking,text="Ranking validation"); nb.add(reporting,text="Reporting & official COCO")

        campaign.columnconfigure(1,weight=1); campaign.columnconfigure(3,weight=1)
        self._label(campaign,"Campaign ID:","Stable ID for preflight/freeze archives.").grid(row=0,column=0,sticky="e",padx=(8,6),pady=(8,4))
        self._entry(campaign,self.var_campaign_id,"Campaign identifier.").grid(row=0,column=1,sticky="ew",pady=(8,4))
        self._label(campaign,"Mode / enforcement:","final+strict blocks missing contracts; development+warn is safe for test runs.").grid(row=0,column=2,sticky="e",padx=(12,6),pady=(8,4))
        modef=ttk.Frame(campaign); modef.grid(row=0,column=3,sticky="w",padx=(0,8),pady=(8,4))
        self._combo(modef,self.var_campaign_mode,["development","screening","final"],"Campaign mode.",state="readonly",width=13).pack(side=tk.LEFT)
        ttk.Label(modef,text=" / ").pack(side=tk.LEFT)
        self._combo(modef,self.var_campaign_enforcement,["warn","strict"],"Preflight enforcement.",state="readonly",width=9).pack(side=tk.LEFT)
        self._check(campaign,"Vor finaler Kampagne eingefroren",self.var_campaign_frozen,"Declares that profile inputs were frozen prospectively.").grid(row=1,column=1,sticky="w",pady=4)
        self._label(campaign,"Claim scope:","evaluated_matrix limits conclusions to measured workloads/hardware; ranking_generalization enables the strict prospective hold-out contract.").grid(row=1,column=2,sticky="e",padx=(12,6),pady=4)
        self._combo(campaign,self.var_claim_scope,[EVALUATED_MATRIX_CLAIM_SCOPE,RANKING_GENERALIZATION_CLAIM_SCOPE],"Scientific scope of the final claim.",state="readonly",width=24).grid(row=1,column=3,sticky="w",padx=(0,8),pady=4)
        self._label(campaign,"Hold-out registry:","Generated from model rows whose role is holdout.").grid(row=2,column=0,sticky="e",padx=(8,6),pady=4)
        self._entry(campaign,self.var_holdout_registry,"Output/input path for holdout registry.").grid(row=2,column=1,sticky="ew",pady=4)
        self._label(campaign,"Ranking model bundle:","Fitted only on development rows.").grid(row=2,column=2,sticky="e",padx=(12,6),pady=4)
        self._entry(campaign,self.var_ranking_model_bundle,"Frozen stage-time/handover model bundle.").grid(row=2,column=3,sticky="ew",padx=(0,8),pady=4)
        req=ttk.LabelFrame(campaign,text="Final-mode requirements"); req.grid(row=3,column=0,columnspan=4,sticky="ew",padx=8,pady=8)
        checks=[("Fitted stage-time",self.var_require_fitted_stage_time),("Native handover model",self.var_require_native_handover),("Campaign freeze",self.var_require_campaign_freeze),("Prediction approval",self.var_require_prediction_approval),("Cryptographic signature",self.var_require_prediction_signature)]
        for i,(label,var) in enumerate(checks): self._check(req,label,var,"Required preflight condition.").grid(row=i//3,column=i%3,sticky="w",padx=8,pady=5)
        ttk.Label(req,text="Public key:").grid(row=2,column=0,sticky="e",padx=(8,6),pady=5)
        self._entry(req,self.var_prediction_public_key,"PEM public key used to verify freeze signatures.").grid(row=2,column=1,columnspan=2,sticky="ew",padx=(0,8),pady=5)
        buttons=ttk.Frame(campaign); buttons.grid(row=4,column=0,columnspan=4,sticky="w",padx=8,pady=(0,8))
        self._button(buttons,"Hold-out registry erzeugen",self._create_holdout_registry_from_current,"Saves a temporary/current profile and creates a content-addressed holdout_registry.json.").pack(side=tk.LEFT)
        self._button(buttons,"Campaign readiness prüfen",self._campaign_readiness_preview,"Runs the same final campaign preflight used by the CLI.").pack(side=tk.LEFT,padx=(8,0))

        ranking.columnconfigure(1,weight=1); ranking.columnconfigure(3,weight=1)
        self._check(ranking,"Ranking validation enabled",self.var_ranking_enabled,"Emit hold-out MAE/MAPE, correlations and shortlist metrics.").grid(row=0,column=0,columnspan=2,sticky="w",padx=8,pady=(8,4))
        self._check(ranking,"Frozen predictions required",self.var_ranking_require_frozen,"Prospective predictions_frozen artefact required.").grid(row=0,column=2,sticky="w",padx=8,pady=(8,4))
        self._check(ranking,"Complete/audit universe required",self.var_ranking_require_complete,"Blocks top-k claims without declared score-independent universe.").grid(row=0,column=3,sticky="w",padx=8,pady=(8,4))
        fields=[("k values",self.var_ranking_k_values),("elite q values",self.var_ranking_q_values)]
        for i,(label,var) in enumerate(fields, start=1):
            ttk.Label(ranking,text=label+":").grid(row=i,column=0,sticky="e",padx=(8,6),pady=4); self._entry(ranking,var,"Comma-separated integers.").grid(row=i,column=1,sticky="ew",pady=4)
        ttk.Label(ranking,text="Primary k:").grid(row=1,column=2,sticky="e",padx=(12,6),pady=4); self._spin(ranking,self.var_ranking_primary_k,"Primary shortlist size.",from_=1,to=999,width=8).grid(row=1,column=3,sticky="w",pady=4)
        ttk.Label(ranking,text="Min candidates for correlation:").grid(row=2,column=2,sticky="e",padx=(12,6),pady=4); self._spin(ranking,self.var_ranking_min_corr,"Minimum rows for Spearman/Kendall.",from_=2,to=999,width=8).grid(row=2,column=3,sticky="w",pady=4)
        ttk.Label(ranking,text="Near-optimal epsilon [%]:").grid(row=3,column=0,sticky="e",padx=(8,6),pady=4); self._spin(ranking,self.var_ranking_epsilon_pct,"Relative practical-equivalence band.",from_=0,to=100,increment=0.1,width=9).grid(row=3,column=1,sticky="w",pady=4)
        ttk.Label(ranking,text="Bootstrap / seed:").grid(row=3,column=2,sticky="e",padx=(12,6),pady=4)
        rf=ttk.Frame(ranking); rf.grid(row=3,column=3,sticky="w",pady=4); self._spin(rf,self.var_ranking_bootstrap,"Macro bootstrap repetitions.",from_=100,to=1000000,width=9).pack(side=tk.LEFT); ttk.Label(rf,text=" / ").pack(side=tk.LEFT); self._spin(rf,self.var_ranking_seed,"Bootstrap seed.",from_=0,to=2147483647,width=11).pack(side=tk.LEFT)
        ttk.Label(ranking,text="Methods: Cut Bytes only; Weighted Score; Cycle Time without handover; Cycle Time with runner/direction-specific handover.",foreground="#555",wraplength=1000).grid(row=4,column=0,columnspan=4,sticky="ew",padx=8,pady=(8,8))

        reporting.columnconfigure(1,weight=1); reporting.columnconfigure(3,weight=1)
        tex=ttk.LabelFrame(reporting,text="Canonical report outputs"); tex.grid(row=0,column=0,columnspan=4,sticky="ew",padx=8,pady=8)
        opts=[("Thesis-ready TeX tables",self.var_generate_thesis_tex),("Thesis-ready PDF/PNG figures",self.var_generate_thesis_figures),("Legacy reports cleanup",self.var_cleanup_legacy_reports),("Campaign readiness block",self.var_include_campaign_readiness)]
        for i,(label,var) in enumerate(opts): self._check(tex,label,var,"Canonical scientific report option.").grid(row=i//2,column=i%2,sticky="w",padx=8,pady=5)
        coco=ttk.LabelFrame(reporting,text="Official COCOeval artefact"); coco.grid(row=1,column=0,columnspan=4,sticky="ew",padx=8,pady=(0,8)); coco.columnconfigure(1,weight=1); coco.columnconfigure(3,weight=1)
        self._check(coco,"Enable pycocotools verification",self.var_official_coco_enabled,"Runs official COCOeval for detection rows in addition to paired-bootstrap gating.").grid(row=0,column=0,sticky="w",padx=8,pady=6)
        self._check(coco,"Required for final",self.var_official_coco_required,"Final detector run fails/blocks if pycocotools or annotations are missing.").grid(row=0,column=1,sticky="w",padx=8,pady=6)
        self._check(coco,"Archive precision/recall tensors",self.var_official_coco_archive_tensors,"Stores compressed COCOeval precision/recall/score arrays.").grid(row=0,column=2,columnspan=2,sticky="w",padx=8,pady=6)
        ttk.Label(coco,text="Local annotations:").grid(row=1,column=0,sticky="e",padx=(8,6),pady=4); self._entry(coco,self.var_official_coco_annotations,"Local instances_val2017.json used for local runs/provenance.").grid(row=1,column=1,sticky="ew",pady=4)
        self._button(coco,"JSON…",lambda:self._browse_file_var(self.var_official_coco_annotations,[("JSON","*.json")]),"Select instances_val2017.json.").grid(row=1,column=2,padx=6,pady=4)
        ttk.Label(coco,text="Remote annotations:").grid(row=2,column=0,sticky="e",padx=(8,6),pady=4); self._entry(coco,self.var_official_coco_remote_annotations,"Path visible on remote benchmark targets.").grid(row=2,column=1,columnspan=3,sticky="ew",padx=(0,8),pady=4)
        ttk.Label(coco,text="maxDets:").grid(row=3,column=0,sticky="e",padx=(8,6),pady=4); self._entry(coco,self.var_official_coco_max_dets,"Usually 1,10,100.",width=20).grid(row=3,column=1,sticky="w",pady=4)
        self._button(coco,"pycocotools status",self._show_pycocotools_status,"Checks the GUI environment; remote targets are checked during the run.").grid(row=3,column=2,sticky="w",padx=6,pady=4)
        ttk.Label(coco,text="The internal paired-bootstrap COCO-style metric remains the non-inferiority gate. The official COCOeval result is archived as independent final-claim evidence.",foreground="#666",wraplength=1000).grid(row=4,column=0,columnspan=4,sticky="ew",padx=8,pady=(4,8))

    def _build_advanced_tab(self, runtime_tab: ttk.Frame, build_tab: ttk.Frame) -> None:
        """Workflow/runtime defaults written into the YAML.

        v49i deliberately uses the Evaluation Profile as the single source of
        truth for execution mode, Hailo policy and remote execution.  The main
        Workflow tab no longer duplicates these fields.
        """
        for tab in (runtime_tab, build_tab):
            tab.columnconfigure(1, weight=1)
            tab.columnconfigure(3, weight=1)
        row = 0

        workflow = ttk.LabelFrame(runtime_tab, text="Workflow execution")
        workflow.grid(row=row, column=0, columnspan=4, sticky="ew", padx=8, pady=8)
        workflow.columnconfigure(1, weight=1)
        workflow.columnconfigure(3, weight=1)
        exec_tip = (
            "contracts_only schreibt nur Verträge/Pläne. generate_benchmarksets nutzt den bestehenden BenchmarkSet-Generator "
            "als Source of Truth für Split-Export/Hailo-HEF-Builds. generate_and_run startet danach die normale Suite lokal oder remote."
        )
        self._label(workflow, "Execution mode:", exec_tip).grid(row=0, column=0, sticky="w", padx=(8, 6), pady=6)
        cb_exec = self._combo(workflow, self.var_workflow_execution_mode, ["contracts_only", "generate_benchmarksets", "generate_and_run", "legacy_benchmarkset"], exec_tip, state="readonly", width=24)
        cb_exec.grid(row=0, column=1, sticky="w", padx=(0, 12), pady=6)

        def _exec_mode_changed(_event: object | None = None) -> None:
            mode = str(self.var_workflow_execution_mode.get() or "")
            if mode == "generate_and_run":
                self.var_workflow_skip_runtime.set(False)
            elif mode == "contracts_only":
                self.var_workflow_skip_runtime.set(True)

        cb_exec.bind("<<ComboboxSelected>>", _exec_mode_changed, add="+")
        self._check(workflow, "Runtime benchmarks überspringen", self.var_workflow_skip_runtime, "Aktiv = kein echter Benchmarkstart. Für reine Plan-/Suite-Erzeugung aktiv lassen; für echte Messung deaktivieren.").grid(row=0, column=2, sticky="w", padx=(0, 12), pady=6)
        self._check(workflow, "No model hash", self.var_workflow_no_model_hash, "Beschleunigt Smoke-Läufe, indem große ONNX-Dateien nicht gehasht werden. Für finale Thesis-Läufe eher deaktivieren.").grid(row=0, column=3, sticky="w", padx=(0, 8), pady=6)
        self._check(workflow, "Reserve-Modelle einbeziehen", self.var_workflow_include_reserve, "Reserve-Modelle aus dem Profil ebenfalls ausführen.").grid(row=1, column=1, sticky="w", padx=(0, 12), pady=(0, 6))
        self._check(workflow, "Parallel remote setups", self.var_parallel_remote_setups, "Unabhängige Orin/u.RECS-Setups parallel ausführen; innerhalb eines Setups seriell.").grid(row=2, column=0, sticky="w", padx=(8, 12), pady=(0, 6))
        pframe = ttk.Frame(workflow)
        pframe.grid(row=2, column=1, columnspan=3, sticky="w", padx=(0, 8), pady=(0, 6))
        ttk.Label(pframe, text="setups/uploads/powercalc:").pack(side=tk.LEFT, padx=(0, 6))
        self._spin(pframe, self.var_max_parallel_setups, "Maximale parallele Hardware-Setup-Worker. 3 passt für Hailo8/Hailo10/DeepX.", from_=1, to=16, width=5).pack(side=tk.LEFT)
        ttk.Label(pframe, text=" / ").pack(side=tk.LEFT)
        self._spin(pframe, self.var_max_parallel_uploads, "Maximale parallele Bundle-Uploads; 1 ist konservativ und vermeidet Netz-/I/O-Spitzen.", from_=0, to=16, width=5).pack(side=tk.LEFT)
        ttk.Label(pframe, text=" / ").pack(side=tk.LEFT)
        self._spin(pframe, self.var_powercalc_workers, "Maximale parallele power_calculations-Prozesse; 1 ist stabiler für I/O.", from_=0, to=16, width=5).pack(side=tk.LEFT)

        bench = ttk.LabelFrame(runtime_tab, text="Benchmark runtime defaults")
        row += 1
        bench.grid(row=row, column=0, columnspan=4, sticky="ew", padx=8, pady=(0, 8))
        bench.columnconfigure(1, weight=1)
        bench.columnconfigure(3, weight=1)
        self._label(bench, "Provider:", "Backend für lokale Suite-Ausführung; auto nutzt die Suite-/Profil-Defaults.").grid(row=0, column=0, sticky="w", padx=(8, 6), pady=6)
        self._combo(bench, self.var_benchmark_provider, ["auto", "cpu", "cuda", "tensorrt"], "Lokaler Benchmark-Provider.", state="normal", width=14).grid(row=0, column=1, sticky="w", padx=(0, 12), pady=6)
        self._label(bench, "Warmup / runs / timeout:", "Lokale Aufwärmiterationen, Messläufe und Timeout in Sekunden. Timeout 0 = kein Zusatzlimit.").grid(row=0, column=2, sticky="w", padx=(0, 6), pady=6)
        small = ttk.Frame(bench)
        small.grid(row=0, column=3, sticky="w", padx=(0, 8), pady=6)
        self._spin(small, self.var_benchmark_warmup, "Lokale Warmup-Iterationen.", from_=0, to=999999, width=7).pack(side=tk.LEFT)
        ttk.Label(small, text=" / ").pack(side=tk.LEFT)
        self._spin(small, self.var_benchmark_runs, "Lokale Messläufe.", from_=1, to=999999, width=7).pack(side=tk.LEFT)
        ttk.Label(small, text=" / ").pack(side=tk.LEFT)
        self._spin(small, self.var_benchmark_timeout, "Lokaler Timeout in Sekunden; 0 deaktiviert.", from_=0, to=999999, width=7).pack(side=tk.LEFT)

        energy = ttk.LabelFrame(runtime_tab, text="Energy measurement")
        row += 1
        energy.grid(row=row, column=0, columnspan=4, sticky="ew", padx=8, pady=(0, 8))
        energy.columnconfigure(1, weight=1)
        energy.columnconfigure(3, weight=1)
        self._check(energy, "u.RECS Energy im Eval-Run messen", self.var_energy_enabled, "Aktiviert u.RECS Energiemessung für Remote-Dispatches im Evaluation Workflow. u.RECS IPs kommen aus Tool Config → Hardware Setups.").grid(row=0, column=0, columnspan=2, sticky="w", padx=(8, 12), pady=6)
        self._label(energy, "Scope:", "row_variant misst canonical full rows und composed split rows; dispatch ist nur ein Legacy-/Debug-Fallback.").grid(row=0, column=2, sticky="w", padx=(0, 6), pady=6)
        self._combo(energy, self.var_energy_scope, ["row_variant", "dispatch"], "Energy scope.", state="readonly", width=14).grid(row=0, column=3, sticky="w", padx=(0, 8), pady=6)

        self._label(energy, "Policy:", "canonical_only misst Full-Baselines und echte heterogene Splits; deepx_only misst nur DeepX-Run-IDs; all misst alles; manual nutzt Include-Liste.").grid(row=1, column=0, sticky="w", padx=(8, 6), pady=(0, 6))
        self._combo(energy, self.var_energy_target_policy, ["canonical_only", "best_valid_only", "best_plus_predicted", "deepx_only", "all", "manual"], "Energy target policy.", state="readonly", width=16).grid(row=1, column=1, sticky="w", padx=(0, 12), pady=(0, 6))
        self._label(energy, "Phases:", "Kommagetrennt: latency, streaming. Leer = beide.").grid(row=1, column=2, sticky="w", padx=(0, 6), pady=(0, 6))
        self._combo(energy, self.var_energy_phases, ["latency, streaming", "latency", "streaming"], "Energy phases preset; editable for advanced comma-separated values.", state="normal", width=22).grid(row=1, column=3, sticky="ew", padx=(0, 8), pady=(0, 6))

        self._label(energy, "Repeats override:", "0 = Benchmark/Remote-Repeats verwenden. >0 = u.RECS-Fenster-Repeats überschreiben.").grid(row=2, column=0, sticky="w", padx=(8, 6), pady=(0, 6))
        self._spin(energy, self.var_energy_repeat_override, "Energy repeat override; 0 nutzt Benchmark-Repeats.", from_=0, to=99, width=7).grid(row=2, column=1, sticky="w", padx=(0, 12), pady=(0, 6))
        self._label(energy, "Heartbeat (s):", "Status-Update-Intervall während langer u.RECS-Fenster.").grid(row=2, column=2, sticky="w", padx=(0, 6), pady=(0, 6))
        self._spin(energy, self.var_energy_heartbeat_s, "Heartbeat in Sekunden; 60 ist ein sinnvoller Standard.", from_=10, to=3600, width=7).grid(row=2, column=3, sticky="w", padx=(0, 8), pady=(0, 6))

        self._label(energy, "Skip run IDs:", "Kommagetrennt. Default: ort_cpu, ort_cuda; vermeidet sehr lange Diagnose-Energy-Messungen.").grid(row=3, column=0, sticky="w", padx=(8, 6), pady=(0, 6))
        self._combo(energy, self.var_energy_skip_backends, ["ort_cpu, ort_cuda", "ort_cpu", "ort_cuda", ""], "Skip run-id preset; editable comma-separated list.", state="normal", width=28).grid(row=3, column=1, sticky="ew", padx=(0, 12), pady=(0, 6))
        self._label(energy, "Include IDs:", "Nur bei Policy manual: kommagetrennte run_ids, z.B. ort_tensorrt, deepx_m1_full.").grid(row=3, column=2, sticky="w", padx=(0, 6), pady=(0, 6))
        self._combo(energy, self.var_energy_include_run_ids, ["", "ort_tensorrt, deepx_m1_full, deepx_m1_to_tensorrt", "deepx_m1_full, deepx_m1_to_tensorrt", "ort_tensorrt"], "Manual include run-id preset; editable comma-separated list.", state="normal", width=28).grid(row=3, column=3, sticky="ew", padx=(0, 8), pady=(0, 6))

        self._label(energy, "Exclude IDs:", "Zusätzliche Deny-Liste für Energy, unabhängig von Policy.").grid(row=4, column=0, sticky="w", padx=(8, 6), pady=(0, 6))
        self._combo(energy, self.var_energy_exclude_run_ids, ["", "ort_cpu, ort_cuda", "ort_cpu", "ort_cuda"], "Additional exclude run-id preset; editable comma-separated list.", state="normal", width=28).grid(row=4, column=1, sticky="ew", padx=(0, 12), pady=(0, 6))
        opts = ttk.Frame(energy)
        opts.grid(row=4, column=2, columnspan=2, sticky="w", padx=(0, 8), pady=(0, 6))
        self._check(opts, "Final: alle Split-Energy", self.var_energy_final_all_splits, "Thesis-/Final-Modus: setzt Energy=an, Scope=row_variant, Policy=all, Max targets/run=0 und Strict=an. CPU ORT kann über die nächste Checkbox bewusst ausgespart werden.").pack(side=tk.LEFT, padx=(0, 12))
        self._check(opts, "CPU ORT überspringen", self.var_energy_final_skip_cpu_ort, "Empfohlen für Final-Energy: CPU-ORT Split-Diagnostik nicht per u.RECS messen, weil sie sehr lange dauert und nicht accelerator-relevant ist. TensorRT/CUDA/Hailo/DeepX bleiben enthalten.").pack(side=tk.LEFT, padx=(0, 12))
        self._check(opts, "Strict", self.var_energy_strict, "Wenn aktiv, wird fehlende u.RECS-Evidence als Fehler behandelt. Für Smoke-Runs meist aus lassen.").pack(side=tk.LEFT, padx=(0, 12))
        self._check(opts, "Raw parquet ins Debug-Pack", self.var_energy_include_raw_parquet, "Kann sehr groß werden. Standard: aus.").pack(side=tk.LEFT)

        self._label(energy, "Max targets/run:", "0 = alle Split-Targets messen; 1/2 begrenzt lange YOLO/DeepX-Energy-Runs.").grid(row=5, column=0, sticky="w", padx=(8, 6), pady=(0, 6))
        self._spin(energy, self.var_energy_max_targets_per_run_id, "Energy target cap pro run_id; 0=alle Targets messen. best_valid_only nutzt typischerweise 1, best_plus_predicted typischerweise bis zu 2.", from_=0, to=999, width=7).grid(row=5, column=1, sticky="w", padx=(0, 12), pady=(0, 6))
        self._label(energy, "Max WU/window:", "Hard cap für Work Units pro u.RECS-Fenster; 0 = aus.").grid(row=5, column=2, sticky="w", padx=(0, 6), pady=(0, 6))
        self._spin(energy, self.var_energy_max_work_units_per_window, "Hard cap für Work Units pro u.RECS-Fenster; 0 = aus.", from_=0, to=9999999, width=9).grid(row=5, column=3, sticky="w", padx=(0, 8), pady=(0, 6))

        self._label(energy, "Max window s:", "Schätzt/cappt aktive Benchmarkdauer pro Energy-Fenster; 0 = aus.").grid(row=6, column=0, sticky="w", padx=(8, 6), pady=(0, 6))
        self._spin(energy, self.var_energy_max_window_duration_s, "Max aktive Benchmarkdauer je u.RECS-Fenster; 0 = aus.", from_=0, to=999999, width=7).grid(row=6, column=1, sticky="w", padx=(0, 12), pady=(0, 6))
        self._label(energy, "Timeout/window s:", "Hard timeout für Collector-Fenster; 0 = automatisch.").grid(row=6, column=2, sticky="w", padx=(0, 6), pady=(0, 6))
        self._spin(energy, self.var_energy_timeout_s_per_window, "Hard timeout für Collector-Fenster; 0 = automatisch.", from_=0, to=999999, width=9).grid(row=6, column=3, sticky="w", padx=(0, 8), pady=(0, 6))

        self._label(energy, "Sizing pilot WU:", "Max Work Units für den kurzen Sizing-Pilot vor Auto-Scaling.").grid(row=7, column=0, sticky="w", padx=(8, 6), pady=(0, 6))
        self._spin(energy, self.var_energy_sizing_probe_max_work_units, "Sizing pilot cap; kleinere Werte debuggen schneller.", from_=1, to=999999, width=7).grid(row=7, column=1, sticky="w", padx=(0, 12), pady=(0, 6))

        est = ttk.Label(energy, textvariable=self.var_energy_estimate, foreground="#555", wraplength=1050, justify="left")
        est.grid(row=8, column=0, columnspan=4, sticky="ew", padx=8, pady=(2, 8))
        self._tip(est, "Schätzung: targets × phases × repeats. Row-Ziele hängen von accepted cases und gewählter Policy ab; echte Laufzeit hängt vom Modell ab.")

        native = ttk.LabelFrame(runtime_tab, text="Native producer fastpath")
        row += 1
        native.grid(row=row, column=0, columnspan=4, sticky="ew", padx=8, pady=(0, 8))
        native.columnconfigure(1, weight=1)
        native.columnconfigure(3, weight=1)
        self._check(
            native,
            "Native Runner im EvalRun ausführen",
            self.var_native_enabled,
            "Aktiviert die strikte Native-Fastpath-Stage nach dem generischen EvalRun. Unterstützte Contracts werden nativ gemessen; nicht unterstützte Cases werden als unsupported reportet.",
        ).grid(row=0, column=0, columnspan=2, sticky="w", padx=(8, 12), pady=6)
        self._label(native, "Backends:", "Kommagetrennt: hailo8, hailo10h, deepx. Diese Backends nutzen die Remote-Setups aus Tool Config.").grid(row=0, column=2, sticky="w", padx=(0, 6), pady=6)
        self._entry(native, self.var_native_backends, "Native-Producer-Backends, z. B. hailo8,hailo10h,deepx", width=28).grid(row=0, column=3, sticky="ew", padx=(0, 8), pady=6)

        self._label(native, "Case policy:", "all_accepted versucht alle erzeugten b*/Cases und markiert unsupported Contracts transparent. case_map_only nutzt nur explizite Case-Maps.").grid(row=1, column=0, sticky="w", padx=(8, 6), pady=(0, 6))
        self._combo(native, self.var_native_case_policy, ["all_accepted", "case_map_only", "preferred_then_backfill"], "Native-Case-Auswahl.", state="readonly", width=24).grid(row=1, column=1, sticky="w", padx=(0, 12), pady=(0, 6))
        self._label(native, "Precision:", "Native TensorRT Boundary-Bridge. uint8_cast_fp16 nutzt uint8 Boundary und Cast innerhalb TensorRT.").grid(row=1, column=2, sticky="w", padx=(0, 6), pady=(0, 6))
        self._combo(native, self.var_native_precision, ["uint8_cast_fp16", "fp16", "fp32"], "Native TensorRT Precision/Boundary-Bridge.", state="normal", width=18).grid(row=1, column=3, sticky="w", padx=(0, 8), pady=(0, 6))

        self._label(native, "Frames / Warmup / Wiederholungen:", "Standard 100/10/1; Final Quality 1000/100/3. Energie-Replikate werden separat aufgelöst.").grid(row=2, column=0, sticky="w", padx=(8, 6), pady=(0, 6))
        nrun = ttk.Frame(native)
        nrun.grid(row=2, column=1, sticky="w", padx=(0, 12), pady=(0, 6))
        self._spin(nrun, self.var_native_frames, "Native-Producer Frames.", from_=1, to=9999999, width=8).pack(side=tk.LEFT)
        ttk.Label(nrun, text=" / ").pack(side=tk.LEFT)
        self._spin(nrun, self.var_native_warmup, "Native-Producer Warmup Frames.", from_=0, to=9999999, width=8).pack(side=tk.LEFT)
        self._spin(nrun, self.var_native_repetitions, "Unabhängige Performancewiederholungen; bei n=1 kein Wiederholungs-CI.", from_=1, to=999, width=5).pack(side=tk.LEFT)
        self._label(native, "Queue / inflight:", "Queue depth für FIFO und Hailo10-InferModel inflight-Jobs.").grid(row=2, column=2, sticky="w", padx=(0, 6), pady=(0, 6))
        nq = ttk.Frame(native)
        nq.grid(row=2, column=3, sticky="w", padx=(0, 8), pady=(0, 6))
        self._spin(nq, self.var_native_queue_depth, "Native FIFO queue depth.", from_=1, to=9999, width=6).pack(side=tk.LEFT)
        ttk.Label(nq, text=" / ").pack(side=tk.LEFT)
        self._spin(nq, self.var_native_inflight, "Hailo10 InferModel inflight jobs.", from_=1, to=9999, width=6).pack(side=tk.LEFT)

        native_opts = ttk.Frame(native)
        native_opts.grid(row=3, column=0, columnspan=4, sticky="w", padx=8, pady=(0, 6))
        self._check(native_opts, "Native Full-Baselines", self.var_native_full_baselines, "Misst native Full-Baselines separat: native_full_tensorrt, native_full_hailo8, native_full_hailo10h, native_full_deepx. Kein FIFO für Full-Modelle.").pack(side=tk.LEFT, padx=(0, 16))
        self._check(native_opts, "Native Validation/Visuals", self.var_native_validation_enabled, "Erzeugt Native Output-Dumps und gemeinsame Tensor-/Task-Validation samt visuellen Artefakten.").pack(side=tk.LEFT, padx=(0, 16))
        self._check(native_opts, "Native Energy", self.var_native_energy_enabled, "Erzeugt oder misst u.RECS-Energy für erfolgreiche Native-Producer-Rows. Für Smokes zunächst mode=plan nutzen.").pack(side=tk.LEFT, padx=(0, 12))
        ttk.Label(native_opts, text="mode:").pack(side=tk.LEFT)
        self._combo(native_opts, self.var_native_energy_mode, ["plan", "measure"], "plan erzeugt Kommandos, measure führt sie direkt aus.", state="readonly", width=9).pack(side=tk.LEFT, padx=(4, 16))
        ttk.Label(native_opts, text="energy duration (s):").pack(side=tk.LEFT)
        self._spin(native_opts, self.var_native_energy_duration_s, "Native-Energy-Messdauer in Sekunden. 0 = ToolConfig energy_defaults.native_measurement_duration_s.", from_=0, to=999999, width=8).pack(side=tk.LEFT, padx=(4, 0))

        probe_opts = ttk.Frame(native)
        probe_opts.grid(row=4, column=0, columnspan=4, sticky="w", padx=8, pady=(0, 6))
        self._check(probe_opts, "Window A/B probe", self.var_window_method_probe_enabled, "Separater Screening-Test auf einem erfolgreichen Native-Kommando; lockert keine Full-Paarung und wird nie in Energy-Claims importiert.").pack(side=tk.LEFT, padx=(0, 12))
        ttk.Label(probe_opts, text="repeats:").pack(side=tk.LEFT)
        self._spin(probe_opts, self.var_window_method_probe_repeats, "Mindestens 3 unabhängige Rohspuren sind entscheidungsfähig; 1–2 werden klar als non-decision-capable markiert.", from_=1, to=99, width=5).pack(side=tk.LEFT, padx=(4, 12))
        self._check(probe_opts, "raw Parquet in Debug Pack", self.var_window_method_probe_include_raw, "Nimmt die hashgebundenen Probe-Rohspuren in den Debug Pack auf.").pack(side=tk.LEFT, padx=(0, 12))
        self._check(probe_opts, "strict", self.var_window_method_probe_strict, "Markiert null oder unvollständige Probe-Messungen als Validierungsfehler. Workflow-blockierend ist das nur im expliziten Final-Modus; Smoke und Standard behalten eine Warnung.").pack(side=tk.LEFT)

        native_note = ttk.Label(
            native,
            text=(
                "Native Producer ist eine zusätzliche strikte Ausführungsart. "
                "Der generische Runner bleibt für Kandidatensuche/Validation; Native misst nur unterstützte IO-Contracts und reportet Fallback-Gründe."
            ),
            foreground="#666",
            wraplength=1050,
            justify="left",
        )
        native_note.grid(row=5, column=0, columnspan=4, sticky="ew", padx=8, pady=(0, 8))
        self._tip(native_note, "Für den finalen Thesis-Run: Native Runner an, Full-Baselines an, Energy erst nach einem erfolgreichen plan-Smoke auf measure stellen.")

        build_row = 0
        hailo = ttk.LabelFrame(build_tab, text="Hailo / hardware policy")
        hailo.grid(row=build_row, column=0, columnspan=4, sticky="ew", padx=8, pady=8)
        hailo.columnconfigure(1, weight=1)
        hailo.columnconfigure(3, weight=1)
        self._label(hailo, "Model preparation:", "current = Profil nutzt vorhandene ONNX-Dateien; screen_yolo_full_hailo = YOLO-Full-Hailo/Raw-Head-Screening als Workflow-Kontext.").grid(row=0, column=0, sticky="w", padx=(8, 6), pady=6)
        self._combo(hailo, self.var_prep_mode, ["current", "screen_yolo_full_hailo"], "Model-Preparation-Strategie im Profil.", state="readonly", width=24).grid(row=0, column=1, sticky="w", padx=(0, 12), pady=6)
        self._label(hailo, "Hailo build mode:", "reuse_only verwendet nur vorhandene HEFs. reuse_and_build_missing/auto verwenden vorhandene HEFs und schreiben fehlende Builds als auszuführende Queue. request/local/venv/wsl markieren konkrete Build-Pfade.").grid(row=1, column=0, sticky="w", padx=(8, 6), pady=6)
        self._combo(hailo, self.var_hailo_build_mode, ["auto", "reuse_only", "reuse_and_build_missing", "request", "local", "venv", "wsl"], "Hailo-Build-/Reuse-Policy. Für echte Runs meist reuse_and_build_missing oder auto; reuse_only bleibt rein nicht-destruktiv.", state="readonly", width=24).grid(row=1, column=1, sticky="w", padx=(0, 12), pady=6)
        self._label(hailo, "Hailo arch / timeout:", "Zielarchitektur und Build-/Service-Timeout in Sekunden.").grid(row=1, column=2, sticky="w", padx=(0, 6), pady=6)
        hsmall = ttk.Frame(hailo)
        hsmall.grid(row=1, column=3, sticky="w", padx=(0, 8), pady=6)
        self._entry(hsmall, self.var_hailo_arch, "Meist hailo8.", width=12).pack(side=tk.LEFT)
        ttk.Label(hsmall, text=" / ").pack(side=tk.LEFT)
        self._spin(hsmall, self.var_hailo_timeout, "Timeout für Hailo-Build-/Service-Requests.", from_=0, to=999999, width=10).pack(side=tk.LEFT)
        self._label(hailo, "Hardware smoke:", "summary_only schreibt Bereitschaft/Reuse/Remote-Status; strict markiert fehlende Hardware härter; disabled überspringt den Smoke-Block.").grid(row=2, column=0, sticky="w", padx=(8, 6), pady=(0, 6))
        self._combo(hailo, self.var_hw_smoke_mode, ["summary_only", "strict", "disabled"], "Hardware-Smoke-Policy.", state="readonly", width=16).grid(row=2, column=1, sticky="w", padx=(0, 8), pady=(0, 6))
        self._button(hailo, "Aus Benchmark/Hardware übernehmen", self._apply_current_split_export_hailo_settings, "Übernimmt Hailo-Ziele aus der Benchmark-Auswahl und Build-/Kalibrierdefaults aus Hardware in dieses YAML-Profil.").grid(row=2, column=2, columnspan=2, sticky="w", padx=(0, 8), pady=(0, 6))

        self._label(hailo, "Targets / Build:", "Hailo-Targets und welche HEF-Artefakte der Evaluation-Run erwarten/aufbauen soll. Diese Werte kommen aus dem Profil, nicht mehr aus dem Workflow-Haupttab.").grid(row=3, column=0, sticky="w", padx=(8, 6), pady=(0, 6))
        htargets = ttk.Frame(hailo)
        htargets.grid(row=3, column=1, columnspan=3, sticky="w", padx=(0, 8), pady=(0, 6))
        self._entry(htargets, self.var_hailo_targets, "Kommagetrennte Hailo-Targets, z. B. hailo8 oder hailo8,hailo10n.", width=18).pack(side=tk.LEFT)
        self._check(htargets, "full", self.var_hailo_build_full, "Full-Hailo-Baseline als HEF erwarten/aufbauen.").pack(side=tk.LEFT, padx=(12, 0))
        self._check(htargets, "part1", self.var_hailo_build_part1, "Part1-HEFs für Splitfälle erwarten/aufbauen.").pack(side=tk.LEFT, padx=(8, 0))
        self._check(htargets, "part2", self.var_hailo_build_part2, "Part2-HEFs für Splitfälle erwarten/aufbauen, sofern policy/Output-Contract das erlaubt.").pack(side=tk.LEFT, padx=(8, 0))
        self._check(htargets, "Force rebuild", self.var_hailo_force_build, "Vorhandene HEFs nicht bevorzugen; für finale Rebuilds/Diagnose.").pack(side=tk.LEFT, padx=(12, 0))
        self._check(htargets, "Keep HARs", self.var_hailo_keep_artifacts, "Zwischenartefakte/HARs behalten, wenn ein Build-Service sie erzeugt.").pack(side=tk.LEFT, padx=(8, 0))

        self._label(hailo, "Preset / opt / calib:", "Hailo-Compiler-/Kalibrierparameter, die in der Build Queue und im Run Manifest dokumentiert werden.").grid(row=4, column=0, sticky="w", padx=(8, 6), pady=(0, 6))
        hcalib = ttk.Frame(hailo)
        hcalib.grid(row=4, column=1, columnspan=3, sticky="ew", padx=(0, 8), pady=(0, 6))
        hcalib.columnconfigure(6, weight=1)
        self._combo(hcalib, self.var_hailo_preset, ["quick", "performance", "balanced", "accuracy", "custom"], "Hailo-Build-Preset aus Hardware bzw. YAML.", state="normal", width=13).grid(row=0, column=0, sticky="w")
        ttk.Label(hcalib, text=" / ").grid(row=0, column=1)
        self._spin(hcalib, self.var_hailo_opt_level, "Hailo Optimierungslevel.", from_=0, to=5, width=5).grid(row=0, column=2, sticky="w")
        ttk.Label(hcalib, text=" / ").grid(row=0, column=3)
        self._spin(hcalib, self.var_hailo_calib_count, "Anzahl Kalibrierbilder.", from_=0, to=999999, width=7).grid(row=0, column=4, sticky="w")
        ttk.Label(hcalib, text=" / ").grid(row=0, column=5)
        self._spin(hcalib, self.var_hailo_calib_batch, "Kalibrier-Batchgröße.", from_=1, to=999999, width=7).grid(row=0, column=6, sticky="w")
        self._entry(hcalib, self.var_hailo_calib_dir, "Kalibrierordner, z. B. /homes/kmika/hailomz/data/coco. Leer = Hardware-/Service-Default.").grid(row=0, column=7, sticky="ew", padx=(10, 0))
        self._button(hcalib, "Ordner…", lambda: self._browse_dir_var(self.var_hailo_calib_dir), "Kalibrierordner wählen.").grid(row=0, column=8, padx=(8, 0))

        dfc_note = ttk.Label(
            hailo,
            text="DFC-Compiler-Umgebungen werden zentral im Hardware-Tab verwaltet: DFC env status / Open wheels / Install/Repair DFC.",
            foreground="#666",
            wraplength=920,
            justify="left",
        )
        dfc_note.grid(row=5, column=0, columnspan=4, sticky="ew", padx=8, pady=(0, 8))
        self._tip(dfc_note, "Der Profile Editor schreibt nur die Hailo-Policy ins YAML. Installation/Repair der lokalen DFC-Compiler-venvs bleibt im Hardware-Tab, damit es keine doppelte UI gibt.")

        # Legacy single-remote profile editing was intentionally removed from the
        # visible profile editor. Remote execution is now selected through
        # hardware.selected_setups and configured centrally in Tool Config
        # (~/.onnx_splitpoint_tool/hardware_setups.yaml). Keep the variables and
        # loader helpers for backwards compatibility, but avoid presenting or
        # saving a second remote host configuration from this dialog.
        self.var_remote_enabled.set(False)

        note = ttk.Label(
            build_tab,
            text=(
                "Hinweis: Remote-Hosts und Runtime-venvs werden nicht mehr im Profil gespeichert. "
                "Das Profil wählt nur noch Hardware-Setup-IDs; Host/User/venv/base-dir kommen zentral aus Tool Config → Accelerator envs."
            ),
            foreground="#666",
            wraplength=940,
            justify="left",
        )
        build_row += 1
        note.grid(row=build_row, column=0, columnspan=4, sticky="ew", padx=8, pady=(0, 8))
        self._tip(note, "Der einfache Workflow-Pfad bleibt: Profil erstellen/auswählen → Start → Results Bundle öffnen.")


    def _active_profile_run_ids(self) -> list[str]:
        rows: list[str] = []
        if bool(self.var_run_cpu.get()): rows.append("ort_cpu")
        if bool(self.var_run_cuda.get()): rows.append("ort_cuda")
        if bool(self.var_run_trt.get()): rows.append("ort_tensorrt")
        if bool(self.var_run_hailo.get()): rows.append("hailo8")
        if bool(self.var_run_hailo_to_trt.get()): rows.append("hailo8_to_trt")
        if bool(self.var_run_trt_to_hailo.get()): rows.append("trt_to_hailo8")
        if bool(self.var_run_hailo10.get()): rows.append("hailo10")
        if bool(self.var_run_hailo10_to_trt.get()): rows.append("hailo10_to_tensorrt")
        if bool(self.var_run_deepx.get()): rows.append("deepx_m1_full")
        if bool(self.var_run_deepx_to_trt.get()): rows.append("deepx_m1_to_tensorrt")
        if bool(self.var_run_trt_to_deepx.get()): rows.append("tensorrt_to_deepx_m1")
        return rows

    def _apply_final_all_split_energy_checkbox(self, *_args: object) -> None:
        """Apply the GUI thesis/final energy preset.

        Users should not need to remember the CLI combination
        ``--energy-target-policy all --energy-max-targets-per-run-id 0``.
        When the checkbox is active the profile becomes the source of truth and
        the workflow later hard-overrides policy/skips/strict from YAML.
        """
        try:
            if not bool(self.var_energy_final_all_splits.get()):
                self._update_energy_estimate()
                return
            self.var_energy_enabled.set(True)
            self.var_energy_scope.set("row_variant")
            self.var_energy_target_policy.set("all")
            self.var_energy_skip_backends.set("ort_cpu" if bool(self.var_energy_final_skip_cpu_ort.get()) else "")
            self.var_energy_exclude_run_ids.set("")
            self.var_energy_max_targets_per_run_id.set(0)
            self.var_energy_strict.set(True)
            if not str(self.var_energy_phases.get() or "").strip():
                self.var_energy_phases.set("latency, streaming")
        except Exception:
            pass
        self._update_energy_estimate()

    def _update_energy_estimate(self, *_args: object) -> None:
        try:
            model_count = max(1, len(self.models) or 1)
            cases = max(1, int(self.var_cases.get() or 1))
            phases = _split_csv(self.var_energy_phases.get()) or ["latency", "streaming"]
            bench_repeats = 1
            try:
                # Evaluation remote dispatch currently uses remote_repeats=1 unless overridden by CLI;
                # energy repeat_override=0 follows that effective repeat count.
                bench_repeats = max(1, int(self._app_var_value("var_remote_repeats", 1) or 1))
            except Exception:
                bench_repeats = 1
            repeats = max(1, int(self.var_energy_repeat_override.get() or 0) or bench_repeats)
            run_ids = self._active_profile_run_ids()
            final_all = bool(self.var_energy_final_all_splits.get())
            policy = "all" if final_all else str(self.var_energy_target_policy.get() or "canonical_only").strip().lower()
            skip = set(_split_csv(self.var_energy_skip_backends.get()))
            if final_all and bool(self.var_energy_final_skip_cpu_ort.get()):
                skip.update({"ort_cpu", "cpu_ort", "cpu"})
            inc = set(_split_csv(self.var_energy_include_run_ids.get()))
            exc = (set() if final_all else set(_split_csv(self.var_energy_exclude_run_ids.get()))) | skip
            measured: list[str] = []
            for rid in run_ids:
                low = rid.lower()
                ok = True
                if policy == "manual":
                    ok = low in {x.lower() for x in inc}
                elif policy == "deepx_only":
                    ok = "deepx" in low or "dx_m1" in low
                elif policy in {"canonical_only", "best_valid_only", "best_plus_predicted"}:
                    ok = not (low.startswith("ort_cpu") or low.startswith("ort_cuda"))
                elif policy == "all":
                    ok = True
                if low in {x.lower() for x in exc}:
                    ok = False
                if ok:
                    measured.append(rid)
            # Canonical full baselines are one target per model; heterogeneous/split rows are roughly per accepted case.
            target_count = 0
            for rid in measured:
                low = rid.lower()
                if low in {"ort_tensorrt", "deepx_m1_full", "hailo8", "hailo10", "ort_cpu", "ort_cuda"}:
                    target_count += 1
                else:
                    cap = max(0, int(self.var_energy_max_targets_per_run_id.get() or 0))
                    target_count += min(cases, cap) if cap > 0 else cases
            windows = model_count * target_count * len(phases) * repeats if bool(self.var_energy_enabled.get()) else 0
            runtime_runs = max(1, int(self.var_benchmark_runs.get() or 1))
            runtime_warmup = max(0, int(self.var_benchmark_warmup.get() or 0))
            final_prefix = "FINAL all-split" if bool(self.var_energy_final_all_splits.get()) else ""
            if final_prefix and bool(self.var_energy_final_skip_cpu_ort.get()):
                final_prefix += " (CPU ORT skip)"
            final_prefix = (final_prefix + " · ") if final_prefix else ""
            text = (
                f"Energy {final_prefix}{'enabled' if bool(self.var_energy_enabled.get()) else 'disabled'} · "
                f"models={model_count}, run_ids={len(run_ids)}, measured_run_ids={len(measured)}, cases≈{cases}, "
                f"targets≈{model_count * target_count}, phases={len(phases)}, repeats={repeats} → u.RECS windows≈{windows}. "
                f"Runtime loop per dispatch: warmup={runtime_warmup}, runs={runtime_runs}. "
                f"Caps: max_targets/run={int(self.var_energy_max_targets_per_run_id.get() or 0)}, max_wu/window={int(self.var_energy_max_work_units_per_window.get() or 0) or 'off'}, max_window_s={int(self.var_energy_max_window_duration_s.get() or 0) or 'off'}. "
                "u.RECS windows use their own min-active-duration scaling; long YOLO windows can dominate runtime."
            )
            self.var_energy_estimate.set(text)
        except Exception:
            try:
                self.var_energy_estimate.set("Energy estimate unavailable")
            except Exception:
                pass

    def _app_var_value(self, name: str, default: Any = "") -> Any:
        """Read a Tk variable from the main app without depending on a specific panel."""
        try:
            obj = getattr(self.app, name, None) if self.app is not None else None
            if obj is not None and hasattr(obj, "get"):
                return obj.get()
        except Exception:
            pass
        return default

    def _apply_current_benchmark_tab_remote_defaults(self, *, enable_remote: bool = False) -> None:
        """Copy runtime defaults from the existing Benchmark tab into this profile.

        The central host list intentionally remains the source of truth for SSH
        endpoints.  The Benchmark tab still owns user-friendly runtime defaults
        such as the remote venv (for example ``~/hailo_py/bin/activate``),
        transfer mode and iteration count.  When the selected host changes in
        the Profile Editor, mirror those values so the YAML profile can be run
        without the user typing the same paths again.
        """
        if enable_remote:
            self.var_remote_enabled.set(True)
        def _set_str(var: tk.StringVar, app_name: str, default: str = "", *, overwrite_empty: bool = False) -> None:
            try:
                value = str(self._app_var_value(app_name, default) or "").strip()
                if value or overwrite_empty:
                    var.set(value)
            except Exception:
                pass
        def _set_int(var: tk.IntVar, app_name: str, default: int) -> None:
            try:
                value = self._app_var_value(app_name, default)
                if value is not None and str(value).strip() != "":
                    var.set(int(float(str(value).strip())))
            except Exception:
                pass
        def _set_bool(var: tk.BooleanVar, app_name: str, default: bool) -> None:
            try:
                value = self._app_var_value(app_name, default)
                if isinstance(value, str):
                    var.set(value.strip().lower() in {"1", "true", "yes", "y", "on"})
                else:
                    var.set(bool(value))
            except Exception:
                pass

        _set_str(self.var_remote_base, "var_remote_base", str(self.var_remote_base.get() or "~/splitpoint_runs"))
        _set_str(self.var_remote_venv, "var_remote_venv", str(self.var_remote_venv.get() or ""))
        _set_str(self.var_remote_transfer_mode, "var_remote_transfer_mode", str(self.var_remote_transfer_mode.get() or "bundle"))
        _set_str(self.var_remote_provider, "var_remote_provider", str(self.var_remote_provider.get() or "auto"))
        _set_str(self.var_remote_ssh_extra_args, "var_remote_ssh_extra_args", str(self.var_remote_ssh_extra_args.get() or ""))
        _set_int(self.var_remote_warmup, "var_remote_warmup", int(self.var_remote_warmup.get() or 10))
        _set_int(self.var_remote_iters, "var_remote_iters", int(self.var_remote_iters.get() or 50))
        _set_int(self.var_remote_timeout, "var_remote_timeout", int(self.var_remote_timeout.get() or 0))
        _set_bool(self.var_remote_reuse_bundle, "var_remote_reuse_bundle", bool(self.var_remote_reuse_bundle.get()))
        _set_bool(self.var_remote_resume, "var_remote_resume", bool(self.var_remote_resume.get()))

    def _apply_current_split_export_hailo_settings(self) -> None:
        """Copy Benchmark target selection + Hardware Hailo build defaults into this profile.

        The Evaluation Workflow is YAML-first, but the GUI should remain the
        source for normal users. This button derives Hailo targets from the
        Benchmark accelerator checkboxes and copies calibration/build defaults
        from the Hardware variables.
        """
        targets: list[str] = []
        try:
            if bool(self._app_var_value("var_bench_acc_hailo8", False)):
                targets.append(str(self._app_var_value("var_hailo_hef_hailo8_hw_arch", "hailo8") or "hailo8").strip() or "hailo8")
            if bool(self._app_var_value("var_bench_acc_hailo10", False)):
                targets.append(str(self._app_var_value("var_hailo_hef_hailo10_hw_arch", "hailo10h") or "hailo10h").strip() or "hailo10h")
            if not targets:
                if bool(self._app_var_value("var_hailo_hef_hailo8_enable", True)):
                    targets.append(str(self._app_var_value("var_hailo_hef_hailo8_hw_arch", "hailo8") or "hailo8").strip() or "hailo8")
                if bool(self._app_var_value("var_hailo_hef_hailo10_enable", False)):
                    targets.append(str(self._app_var_value("var_hailo_hef_hailo10_hw_arch", "hailo10h") or "hailo10h").strip() or "hailo10h")
        except Exception:
            targets = []
        if targets:
            self.var_hailo_targets.set(", ".join(dict.fromkeys(targets)))
            self.var_hailo_arch.set(targets[0])
        try:
            mode = str(self._app_var_value("var_hailo_bench_preset", "End-to-end compare") or "").strip().lower()
            if mode.startswith("end"):
                self.var_hailo_build_full.set(True); self.var_hailo_build_part1.set(True); self.var_hailo_build_part2.set(True)
            elif mode.startswith("split"):
                self.var_hailo_build_full.set(False); self.var_hailo_build_part1.set(True); self.var_hailo_build_part2.set(True)
            elif mode.startswith("every"):
                self.var_hailo_build_full.set(True); self.var_hailo_build_part1.set(True); self.var_hailo_build_part2.set(True)
            else:
                self.var_hailo_build_full.set(bool(self._app_var_value("var_hailo_bench_custom_full", self.var_hailo_build_full.get())))
                self.var_hailo_build_part1.set(bool(self._app_var_value("var_hailo_bench_custom_part1", self.var_hailo_build_part1.get()) or self._app_var_value("var_hailo_bench_custom_composed", False)))
                self.var_hailo_build_part2.set(bool(self._app_var_value("var_hailo_bench_custom_part2", self.var_hailo_build_part2.get()) or self._app_var_value("var_hailo_bench_custom_composed", False)))
        except Exception:
            pass
        def _set_int(var: tk.IntVar, name: str, default: int) -> None:
            try:
                var.set(int(float(str(self._app_var_value(name, default) or default).strip())))
            except Exception:
                var.set(default)
        _set_int(self.var_hailo_opt_level, "var_hailo_hef_opt_level", int(self.var_hailo_opt_level.get() or 0))
        _set_int(self.var_hailo_calib_count, "var_hailo_hef_calib_count", int(self.var_hailo_calib_count.get() or 16))
        _set_int(self.var_hailo_calib_batch, "var_hailo_hef_calib_batch_size", int(self.var_hailo_calib_batch.get() or 8))
        try:
            self.var_hailo_calib_dir.set(str(self._app_var_value("var_hailo_hef_calib_dir", self.var_hailo_calib_dir.get()) or ""))
            self.var_hailo_preset.set(str(self._app_var_value("var_hailo_hef_preset", self.var_hailo_preset.get() or "quick") or "quick"))
            self.var_hailo_force_build.set(parse_config_bool(
                self._app_var_value("var_hailo_hef_force", self.var_hailo_force_build.get()),
                field="hailo_build.force_build",
            ))
            self.var_hailo_keep_artifacts.set(bool(self._app_var_value("var_hailo_hef_keep_artifacts", self.var_hailo_keep_artifacts.get())))
        except ValueError as exc:
            messagebox.showerror("Hailo-Einstellungen", str(exc), parent=self)
            return
        except Exception:
            pass
        self.status_var.set("Hailo-Einstellungen aus Benchmark/Hardware ins Profil übernommen.")

    # ------------------------------------------------------------------
    # File/profile loading
    # ------------------------------------------------------------------

    def _choose_profile_path(self) -> None:
        path = filedialog.askopenfilename(
            parent=self,
            title="Evaluation Profile YAML wählen",
            filetypes=[("YAML", "*.yaml *.yml"), ("All files", "*.*")],
        )
        if path:
            self.var_path.set(path)
            self._load_profile(path)

    def _browse_dir_var(self, var: tk.StringVar) -> None:
        path = filedialog.askdirectory(parent=self, title="Ordner wählen")
        if path:
            var.set(path)

    def _browse_model_onnx(self) -> None:
        path = filedialog.askopenfilename(parent=self, title="ONNX-Modell wählen", filetypes=[("ONNX", "*.onnx"), ("All files", "*.*")])
        if path:
            self.var_model_onnx.set(path)
            if not self.var_model_id.get().strip():
                self.var_model_id.set(_norm_id(path))
            self._infer_model_defaults_from_path(path)

    def _try_load_initial_profile(self) -> None:
        request = ""
        try:
            request = str(self.profile_var.get() if self.profile_var is not None else "").strip()
        except Exception:
            request = ""
        if request:
            loaded = self._load_profile(request, silent=True)
            if loaded:
                return
        # Start with the current GUI model if there is one. This makes the dialog
        # immediately useful for single-model evaluation profiles.
        self._add_current_gui_model(silent=True)
        # Also preselect the central Benchmark-tab remote host when one exists.
        # This mirrors the normal Benchmark tab path: if the user already picked
        # a target host and venv there, a newly created Evaluation Profile should
        # use the same remote runtime instead of silently creating a build-only
        # YAML.
        try:
            selected_id = str(getattr(getattr(self.app, "var_remote_host_id", None), "get", lambda: "")() or "").strip() if self.app is not None else ""
            if selected_id:
                self.var_remote_host_id.set(selected_id)
                self._apply_remote_host_config_to_fields(self._remote_host_config_by_id(selected_id))
                self._apply_current_benchmark_tab_remote_defaults(enable_remote=True)
                self.var_workflow_execution_mode.set("generate_and_run")
                self.var_workflow_skip_runtime.set(False)
        except Exception:
            pass

    def _load_from_dialog_or_current(self) -> None:
        current = str(self.var_path.get() or "").strip() or str(self.profile_var.get() if self.profile_var is not None else "").strip()
        if current:
            if self._load_profile(current):
                return
        self._choose_profile_path()

    def _load_profile(self, request: str, *, silent: bool = False) -> bool:
        try:
            loaded = load_evaluation_profile(request, validate=True)
            if loaded is None or isinstance(loaded, tuple):
                raise ValueError(f"Profile not found: {request}")
            self._apply_payload(dict(loaded.raw_profile or {}), path=str(loaded.profile_path))
            self.status_var.set(f"Geladen: {loaded.profile_id}")
            return True
        except Exception as exc:
            if not silent:
                messagebox.showerror("Evaluation Profile", f"Profil konnte nicht geladen werden:\n\n{type(exc).__name__}: {exc}", parent=self)
            return False

    # ------------------------------------------------------------------
    # Model table
    # ------------------------------------------------------------------

    def _refresh_model_tree(self) -> None:
        tree = self.model_tree
        for item in tree.get_children():
            tree.delete(item)
        columns = tuple(str(x) for x in tree.cget("columns"))
        for idx, row in enumerate(self.models):
            values_by_column = {
                "id": row.get("id", ""),
                "task": row.get("task", ""),
                "usage": _model_usage_from_row(row),
                "role": row.get("evaluation_role", "development"),
                "scope": row.get("generalization_scope", ""),
                "tier": row.get("validation_tier", "screening"),
                "universe": ((row.get("candidate_universe") or {}).get("mode") if isinstance(row.get("candidate_universe"), Mapping) else ""),
                "onnx": row.get("onnx") or row.get("path") or row.get("model_path") or "",
                "family": row.get("family") or row.get("family_id") or "",
                "family_id": row.get("family_id", ""),
                "shape": _shape_to_text(row.get("input_shape")),
                "note": row.get("note", ""),
            }
            tree.insert("", "end", iid=str(idx), values=tuple(values_by_column.get(col, "") for col in columns))

    def _on_model_selected(self, _event: object | None = None) -> None:
        sel = self.model_tree.selection()
        if not sel:
            self._selected_model_index = None
            return
        try:
            idx = int(sel[0])
        except Exception:
            self._selected_model_index = None
            return
        if 0 <= idx < len(self.models):
            self._selected_model_index = idx
            self._fill_model_fields(self.models[idx])

    def _fill_model_fields(self, row: Mapping[str, Any]) -> None:
        self.var_model_id.set(str(row.get("id") or ""))
        self.var_model_task.set(str(row.get("task") or "auto"))
        self.var_model_onnx.set(str(row.get("onnx") or row.get("path") or row.get("model_path") or ""))
        self.var_model_family.set(str(row.get("family") or row.get("family_id") or ""))
        self.var_model_family_id.set(str(row.get("family_id") or ""))
        self.var_model_generalization_scope.set(str(row.get("generalization_scope") or "development"))
        self.var_model_source.set(str(row.get("source") or ""))
        self.var_model_shape.set(_shape_to_text(row.get("input_shape")))
        self.var_model_dataset.set(str(row.get("semantic_dataset") or ""))
        self.var_model_subset.set(str(row.get("development_subset") or ""))
        self.var_model_note.set(str(row.get("note") or ""))
        usage = _model_usage_from_row(row)
        self.var_model_usage.set(usage)
        self.var_model_role.set(
            CONFIRMATORY_HOLDOUT_ROLE
            if usage == MODEL_USAGE_HOLDOUT
            else "development"
        )
        self.var_model_validation_tier.set(str(row.get("validation_tier") or "screening"))
        self.var_model_sha256.set(str(row.get("model_sha256") or ""))
        self.var_model_holdout_group.set(str(row.get("holdout_group") or ""))
        universe = row.get("candidate_universe") if isinstance(row.get("candidate_universe"), Mapping) else {}
        self.var_model_universe_mode.set(str((universe or {}).get("mode") or ("all_feasible" if row.get("candidate_universe_complete") else "declared_shortlist")))
        self.var_model_universe_complete.set(bool(row.get("candidate_universe_complete", False)))
        att = row.get("unseen_attestation") if isinstance(row.get("unseen_attestation"), Mapping) else {}
        self.var_model_unseen.set(bool((att or {}).get("unseen", False)))
        self.var_model_attested_by.set(str((att or {}).get("attested_by") or ""))
        self.var_model_attested_at.set(str((att or {}).get("attested_at") or ""))
        self.var_model_attestation_note.set(str((att or {}).get("note") or ""))

    def _clear_model_fields(self) -> None:
        self._selected_model_index = None
        for var in (
            self.var_model_id,
            self.var_model_onnx,
            self.var_model_family,
            self.var_model_family_id,
            self.var_model_source,
            self.var_model_shape,
            self.var_model_dataset,
            self.var_model_subset,
            self.var_model_note, self.var_model_sha256, self.var_model_holdout_group,
            self.var_model_attested_by, self.var_model_attested_at, self.var_model_attestation_note,
        ):
            var.set("")
        self.var_model_task.set("auto")
        self.var_model_usage.set(MODEL_USAGE_DEVELOPMENT)
        self.var_model_role.set("development")
        self.var_model_generalization_scope.set("development")
        self.var_model_validation_tier.set("screening")
        self.var_model_universe_mode.set("declared_shortlist")
        self.var_model_universe_complete.set(False)
        self.var_model_unseen.set(False)
        try:
            self.model_tree.selection_remove(self.model_tree.selection())
        except Exception:
            pass

    def _current_gui_model_path(self) -> str:
        try:
            state_path = getattr(getattr(self.app, "gui_state", None), "current_model_path", None)
        except Exception:
            state_path = None
        raw = state_path or getattr(self.app, "model_path", None) if self.app is not None else ""
        return str(raw or "").strip()

    def _add_current_gui_model(self, *, silent: bool = False) -> None:
        path = self._current_gui_model_path()
        if not path:
            if not silent:
                messagebox.showinfo("Evaluation Profile", "In der GUI ist aktuell kein ONNX-Modell geladen.", parent=self)
            return
        self.var_model_onnx.set(path)
        self.var_model_id.set(_norm_id(path))
        self._infer_model_defaults_from_path(path)
        self._upsert_model(silent=silent)

    def _infer_model_defaults_from_path(self, path: str) -> None:
        name = _norm_id(path)
        if not self.var_model_task.get().strip() or self.var_model_task.get() == "auto":
            self.var_model_task.set("detection" if name.startswith("yolo") else "classification" if any(tok in name for tok in ("resnet", "mobilenet", "efficientnet", "regnet", "convnext")) else "auto")
        if not self.var_model_family.get().strip():
            fam = "yolo26" if name.startswith("yolo26") else "yolo11" if name.startswith("yolo11") else name.split("_")[0]
            self.var_model_family.set(fam)
        if not self.var_model_family_id.get().strip():
            self.var_model_family_id.set(str(self.var_model_family.get() or name).strip())
        if not self.var_model_shape.get().strip():
            self.var_model_shape.set("1x3x640x640" if self.var_model_task.get() == "detection" else "1x3x224x224")
        if not self.var_model_dataset.get().strip():
            self.var_model_dataset.set("coco2017_val" if self.var_model_task.get() == "detection" else "imagenet_val")
        if not self.var_model_subset.get().strip():
            self.var_model_subset.set("coco_50" if self.var_model_task.get() == "detection" else "imagenet_val_mini_200")
        if not self.var_model_source.get().strip():
            self.var_model_source.set("ultralytics" if self.var_model_task.get() == "detection" else "torchvision")

    def _default_model_validation_tier(self) -> str:
        try:
            mode_id = normalize_mode_id(self.var_run_mode_id.get())
            mode_cfg = get_run_mode(mode_id)
            quality_cfg = mode_cfg.get("quality") if isinstance(mode_cfg.get("quality"), Mapping) else {}
            tier = str((quality_cfg or {}).get("dataset_tier") or "screening").strip().lower()
        except Exception:
            tier = str(self.var_quality_tier.get() or "screening").strip().lower()
        return tier if tier in {"screening", "final"} else "screening"

    def _development_family_ids(self, *, exclude_index: int | None = None) -> set[str]:
        families: set[str] = set()
        for idx, row in enumerate(self.models):
            if exclude_index is not None and idx == exclude_index:
                continue
            if _model_usage_from_row(row) != MODEL_USAGE_DEVELOPMENT:
                continue
            family_id = str(row.get("family_id") or row.get("family") or "").strip()
            if family_id:
                families.add(family_id)
        return families

    def _normalized_models_for_payload(self) -> list[ModelRow]:
        development_families = {
            str(row.get("family_id") or row.get("family") or "").strip()
            for row in self.models
            if _model_usage_from_row(row) == MODEL_USAGE_DEVELOPMENT
            and str(row.get("family_id") or row.get("family") or "").strip()
        }
        return [
            _derive_simple_model_contract(
                row,
                usage=_model_usage_from_row(row),
                selection_strategy=self.var_selection_strategy.get(),
                default_validation_tier=self._default_model_validation_tier(),
                development_family_ids=development_families,
                preserve_legacy=True,
            )
            for row in self.models
        ]

    def _model_from_fields(self) -> ModelRow:
        onnx_path = str(self.var_model_onnx.get() or "").strip()
        mid = str(self.var_model_id.get() or "").strip() or _norm_id(onnx_path)
        if not mid:
            raise ValueError("Model ID fehlt. Trage eine ID ein oder wähle eine ONNX-Datei.")

        base_index = self._selected_model_index
        if base_index is None or not (0 <= base_index < len(self.models)):
            base_index = next(
                (
                    idx
                    for idx, item in enumerate(self.models)
                    if str(item.get("id") or "") == mid
                ),
                None,
            )
        base: ModelRow = (
            copy.deepcopy(self.models[base_index])
            if base_index is not None and 0 <= base_index < len(self.models)
            else {}
        )
        if not base and onnx_path:
            # Directly typed paths follow the same deterministic defaults as
            # the file picker/current-GUI-model actions.
            self._infer_model_defaults_from_path(onnx_path)
        row: ModelRow = copy.deepcopy(base)
        row["id"] = mid
        task = str(self.var_model_task.get() or "auto").strip() or "auto"
        row["task"] = task

        family = str(self.var_model_family.get() or "").strip()
        shape_text = str(self.var_model_shape.get() or "").strip()
        note = str(self.var_model_note.get() or "").strip()
        usage = _normalize_model_usage(self.var_model_usage.get(), strict=True)
        previous_family = str(base.get("family") or base.get("family_id") or "").strip()
        previous_path = str(base.get("onnx") or base.get("path") or base.get("model_path") or "").strip()
        visible_unchanged = bool(base) and all(
            (
                str(base.get("id") or "") == mid,
                str(base.get("task") or "auto") == task,
                previous_path == onnx_path,
                previous_family == family,
                _shape_to_text(base.get("input_shape")) == shape_text,
                str(base.get("note") or "").strip() == note,
                _model_usage_from_row(base) == usage,
            )
        )
        if family:
            row["family"] = family
            if not str(base.get("family_id") or "").strip() or family != previous_family:
                row["family_id"] = family
        else:
            row.pop("family", None)
            row.pop("family_id", None)

        if shape_text:
            shape = _shape_from_text(shape_text)
            if not shape or any(int(value) <= 0 for value in shape):
                raise ValueError("Input shape muss mindestens vier positive Dimensionen enthalten, z. B. 1x3x224x224.")
            row["input_shape"] = shape
        else:
            row.pop("input_shape", None)

        if note:
            row["note"] = note
        else:
            row.pop("note", None)

        if onnx_path:
            path_keys = [key for key in ("onnx", "path", "model_path") if key in row]
            for key in path_keys or ["onnx"]:
                row[key] = onnx_path
        else:
            for key in ("onnx", "path", "model_path"):
                row.pop(key, None)

        internal_defaults = {
            "classification": {
                "source": "torchvision",
                "semantic_dataset": "imagenet_val",
                "development_subset": "imagenet_val_mini_200",
            },
            "detection": {
                "source": "ultralytics",
                "semantic_dataset": "coco2017_val",
                "development_subset": "coco_50",
            },
        }.get(task, {})
        for key, hidden_var in (
            ("source", self.var_model_source),
            ("semantic_dataset", self.var_model_dataset),
            ("development_subset", self.var_model_subset),
        ):
            if str(row.get(key) or "").strip():
                continue
            value = str(hidden_var.get() or internal_defaults.get(key) or "").strip()
            if value:
                row[key] = value

        row = _derive_simple_model_contract(
            row,
            usage=usage,
            selection_strategy=self.var_selection_strategy.get(),
            default_validation_tier=self._default_model_validation_tier(),
            development_family_ids=self._development_family_ids(exclude_index=base_index),
            preserve_legacy=visible_unchanged,
        )
        role = str(row.get("evaluation_role") or "development")
        universe = row.get("candidate_universe") if isinstance(row.get("candidate_universe"), Mapping) else {}
        self.var_model_role.set(role)
        self.var_model_generalization_scope.set(str(row.get("generalization_scope") or "development"))
        self.var_model_validation_tier.set(str(row.get("validation_tier") or "screening"))
        self.var_model_universe_mode.set(str((universe or {}).get("mode") or "declared_shortlist"))
        self.var_model_universe_complete.set(bool(row.get("candidate_universe_complete", False)))

        mode_id = normalize_mode_id(self.var_run_mode_id.get())
        try:
            mode_cfg = get_run_mode(mode_id)
        except Exception:
            mode_cfg = {}
        campaign_cfg = mode_cfg.get("campaign") if isinstance(mode_cfg.get("campaign"), Mapping) else {}
        reproducibility_cfg = mode_cfg.get("reproducibility") if isinstance(mode_cfg.get("reproducibility"), Mapping) else {}
        strict_final_campaign = bool(
            str(self.var_campaign_mode.get() or "").strip().lower() == "final"
            or str((campaign_cfg or {}).get("mode") or "").strip().lower() == "final"
            or (reproducibility_cfg or {}).get("verify_model_content") is True
        )
        model_hash = str(row.get("model_sha256") or self.var_model_sha256.get() or "").strip()
        if strict_final_campaign and not model_hash and onnx_path:
            try:
                path_obj = Path(onnx_path).expanduser()
                if path_obj.is_file():
                    h = hashlib.sha256()
                    with path_obj.open("rb") as fh:
                        for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b""):
                            h.update(chunk)
                    model_hash = "sha256:" + h.hexdigest()
                    self.var_model_sha256.set(model_hash)
            except Exception:
                model_hash = ""
        if model_hash:
            row["model_sha256"] = model_hash
        if is_confirmatory_holdout(role):
            group = str(row.get("holdout_group") or self.var_model_holdout_group.get() or "").strip()
            row["holdout_group"] = group or f"{task}_holdout"
            if not isinstance(row.get("unseen_attestation"), Mapping):
                attestation = {
                    "unseen": bool(self.var_model_unseen.get()),
                    "attested_by": str(self.var_model_attested_by.get() or "").strip(),
                    "attested_at": str(self.var_model_attested_at.get() or "").strip(),
                    "note": str(self.var_model_attestation_note.get() or "").strip(),
                }
                if any(
                    bool(value)
                    for value in (
                        attestation["unseen"],
                        attestation["attested_by"],
                        attestation["attested_at"],
                        attestation["note"],
                    )
                ):
                    row["unseen_attestation"] = attestation
        return row

    def _commit_current_model_edit_if_needed(self) -> None:
        """Persist the currently edited model row before save/preview/validate.

        Users often edit an ONNX path and then press "Speichern & verwenden"
        without clicking "Modell setzen".  The Profile Editor should behave like
        a form editor, not silently save the old row.  Empty edit fields are
        ignored; invalid non-empty edits still raise the normal validation error.
        """
        fields = [
            self.var_model_id.get(),
            self.var_model_onnx.get(),
            self.var_model_family.get(),
            self.var_model_shape.get(),
            self.var_model_note.get(),
        ]
        task = str(self.var_model_task.get() or "").strip().lower()
        if task and task != "auto":
            fields.append(task)
        if not any(str(x or "").strip() for x in fields):
            return
        row = self._model_from_fields()
        idx = self._selected_model_index
        if idx is None:
            for i, old in enumerate(self.models):
                if str(old.get("id") or "") == str(row.get("id") or ""):
                    idx = i
                    break
        if idx is None:
            self.models.append(row)
            idx = len(self.models) - 1
        else:
            self.models[idx] = row
        self._selected_model_index = idx
        self._refresh_model_tree()
        try:
            self.model_tree.selection_set(str(idx))
            self.model_tree.see(str(idx))
        except Exception:
            pass

    def _upsert_model(self, *, silent: bool = False) -> None:
        try:
            row = self._model_from_fields()
        except Exception as exc:
            if not silent:
                messagebox.showwarning("Evaluation Profile", str(exc), parent=self)
            return
        idx = self._selected_model_index
        if idx is None:
            for i, old in enumerate(self.models):
                if str(old.get("id") or "") == str(row.get("id") or ""):
                    idx = i
                    break
        if idx is None:
            self.models.append(row)
            idx = len(self.models) - 1
        else:
            self.models[idx] = row
        self._selected_model_index = idx
        self._refresh_model_tree()
        try:
            self.model_tree.selection_set(str(idx))
            self.model_tree.see(str(idx))
        except Exception:
            pass
        self.status_var.set(f"Modell gesetzt: {row.get('id')}")

    def _remove_selected_model(self) -> None:
        idx = self._selected_model_index
        if idx is None:
            sel = self.model_tree.selection()
            try:
                idx = int(sel[0]) if sel else None
            except Exception:
                idx = None
        if idx is None or not (0 <= idx < len(self.models)):
            return
        removed = self.models.pop(idx)
        self._selected_model_index = None
        self._refresh_model_tree()
        self._clear_model_fields()
        self.status_var.set(f"Entfernt: {removed.get('id')}")

    def _browse_file_var(self, var: tk.StringVar, filetypes: list[tuple[str, str]]) -> None:
        current = str(var.get() or "").strip()
        initialdir = str(Path(current).expanduser().parent) if current else str(Path.home())
        value = filedialog.askopenfilename(parent=self, initialdir=initialdir, filetypes=filetypes)
        if value:
            var.set(value)

    def _hash_current_model(self) -> None:
        path = Path(str(self.var_model_onnx.get() or "").strip()).expanduser()
        if not path.is_file():
            messagebox.showwarning("Model hash", "Bitte zuerst eine vorhandene ONNX-Datei auswählen.", parent=self)
            return
        h = hashlib.sha256()
        try:
            with path.open("rb") as f:
                while True:
                    chunk = f.read(4 * 1024 * 1024)
                    if not chunk:
                        break
                    h.update(chunk)
            self.var_model_sha256.set("sha256:" + h.hexdigest())
            self.status_var.set(f"SHA-256 berechnet: {path.name}")
        except Exception as exc:
            messagebox.showerror("Model hash", f"Hash konnte nicht berechnet werden:\n\n{type(exc).__name__}: {exc}", parent=self)

    def _attest_model_now(self) -> None:
        self.var_model_unseen.set(True)
        self.var_model_attested_at.set(datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"))
        if not str(self.var_model_attestation_note.get() or "").strip():
            self.var_model_attestation_note.set("Selected before the corresponding benchmark results were inspected.")

    def _load_dataset_registry_into_profile(self, *, fill_only_missing: bool = False, silent: bool = False) -> bool:
        try:
            registry_path = str(self.var_dataset_registry.get() or default_registry_path()).strip()
            registry = load_registry(registry_path)
            manifests = dict(registry.get("manifests") or {})
            pairs = [
                (self.var_manifest_cls_calib, "classification_calibration"),
                (self.var_manifest_cls_val, "classification_validation"),
                (self.var_manifest_det_calib, "detection_calibration"),
                (self.var_manifest_det_val, "detection_validation"),
            ]
            for variable, key in pairs:
                current = str(variable.get() or "").strip()
                value = str(manifests.get(key) or "").strip()
                if value and (not fill_only_missing or not current):
                    variable.set(value)
            datasets = dict(registry.get("datasets") or {})
            annotation = str((datasets.get("coco2017_validation") or {}).get("annotations") or "")
            if annotation and (not fill_only_missing or not str(self.var_official_coco_annotations.get() or "").strip()):
                self.var_official_coco_annotations.set(annotation)
            if not silent:
                self.status_var.set("Finale Dataset-Manifeste aus der Tool-Registry übernommen.")
            return True
        except Exception as exc:
            if not silent:
                messagebox.showerror("Dataset registry", f"Registry konnte nicht übernommen werden:\n\n{type(exc).__name__}: {exc}", parent=self)
            return False

    def _show_dataset_registry_status(self) -> None:
        try:
            payload = registry_status(str(self.var_dataset_registry.get() or "").strip() or None, verify_manifests=False)
            text = json.dumps(payload, indent=2, ensure_ascii=False, default=str)
        except Exception as exc:
            text = json.dumps({"status": "error", "error": f"{type(exc).__name__}: {exc}"}, indent=2)
        win = tk.Toplevel(self)
        win.title("Final dataset registry status")
        win.geometry("920x680")
        box = scrolledtext.ScrolledText(win, wrap="word")
        box.pack(fill="both", expand=True, padx=8, pady=8)
        box.insert("1.0", text)
        box.configure(state="disabled")

    def _open_tool_config_dataset_dialog(self) -> None:
        def _updated() -> None:
            app_registry = getattr(self.app, "var_final_dataset_registry", None) if self.app is not None else None
            try:
                if app_registry is not None and str(app_registry.get() or "").strip():
                    self.var_dataset_registry.set(str(app_registry.get()).strip())
            except Exception:
                pass
            self._load_dataset_registry_into_profile()
        open_dataset_provisioning_dialog(self, app=self.app, on_updated=_updated)

    def _profile_context_path(self) -> Path:
        current = str(self.var_path.get() or "").strip()
        if current:
            return Path(current).expanduser().resolve()
        return self._suggest_save_path().expanduser().resolve()

    def _create_holdout_registry_from_current(self) -> None:
        temp_profile: Path | None = None
        try:
            payload = self._build_payload()
            profile_path = self._profile_context_path()
            base = profile_path.parent
            base.mkdir(parents=True, exist_ok=True)
            output_raw = str(self.var_holdout_registry.get() or "campaign_inputs/holdout_registry.json").strip()
            output = Path(output_raw).expanduser()
            if not output.is_absolute():
                output = base / output
            output.parent.mkdir(parents=True, exist_ok=True)
            temp_profile = output.parent / ".profile_editor_holdout_source.yaml"
            temp_profile.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
            result = create_holdout_registry(profile=temp_profile, output=output)
            try:
                shown = str(result.resolve().relative_to(base.resolve()))
            except Exception:
                shown = str(result.resolve())
            self.var_holdout_registry.set(shown)
            messagebox.showinfo("Hold-out registry", f"Registry erzeugt:\n\n{result}", parent=self)
        except Exception as exc:
            messagebox.showerror("Hold-out registry", f"Registry konnte nicht erzeugt werden:\n\n{type(exc).__name__}: {exc}", parent=self)
        finally:
            if temp_profile is not None:
                try:
                    temp_profile.unlink(missing_ok=True)
                except Exception:
                    pass

    def _campaign_readiness_preview(self) -> None:
        temp_profile: Path | None = None
        try:
            payload = self._build_payload()
            profile_path = self._profile_context_path()
            profile_path.parent.mkdir(parents=True, exist_ok=True)
            temp_profile = profile_path.parent / ".profile_editor_readiness.yaml"
            temp_profile.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
            report = build_campaign_readiness(payload, profile_path=temp_profile)
            text = readiness_markdown(report) + "\n\n```json\n" + json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n```\n"
        except Exception as exc:
            text = f"# Campaign readiness failed\n\n`{type(exc).__name__}: {exc}`\n"
        finally:
            if temp_profile is not None:
                try:
                    temp_profile.unlink(missing_ok=True)
                except Exception:
                    pass
        win = tk.Toplevel(self)
        win.title("Campaign readiness")
        win.geometry("1060x760")
        box = scrolledtext.ScrolledText(win, wrap="word")
        box.pack(fill="both", expand=True, padx=8, pady=8)
        box.insert("1.0", text)
        box.configure(state="disabled")

    def _show_pycocotools_status(self) -> None:
        status = pycocotools_status()
        messagebox.showinfo("pycocotools", json.dumps(status, indent=2, ensure_ascii=False), parent=self)

    # ------------------------------------------------------------------
    # Payload generation/loading
    # ------------------------------------------------------------------


    def _selected_hardware_setup_ids(self) -> list[str]:
        """Return central Tool Config setup ids selected by this profile.

        Older versions embedded host/user/venv fields directly in the profile.
        The central hardware registry is now the source of truth, so the
        profile stores setup ids only.
        """
        selected = _normalize_hardware_setup_ids(self.var_hardware_selected_setups.get())
        def _add(sid: str) -> None:
            if sid and sid not in selected:
                selected.append(sid)
        use_h8 = bool(self.var_run_hailo.get() or self.var_run_hailo_to_trt.get() or self.var_run_trt_to_hailo.get() or self.var_hw_h8_enabled.get())
        use_h10 = bool(self.var_run_hailo10.get() or self.var_run_hailo10_to_trt.get() or self.var_hw_h10_enabled.get())
        use_dx = bool(self.var_run_deepx.get() or self.var_run_deepx_to_trt.get() or self.var_run_trt_to_deepx.get() or self.var_hw_dx_enabled.get())
        if use_h8:
            self.var_hw_h8_enabled.set(True)
            _add("orin_nx_hailo8_01")
        if use_h10:
            self.var_hw_h10_enabled.set(True)
            _add("orin_nx_hailo10_01")
        if use_dx:
            self.var_hw_dx_enabled.set(True)
            _add("orin_nx_deepx_m1_01")
        return selected

    def _hardware_setup_entries(self) -> list[dict[str, Any]]:
        """Deprecated compatibility hook.

        Profiles no longer embed remote runtime details.  The workflow resolves
        hardware.selected_setups through ~/.onnx_splitpoint_tool/hardware_setups.yaml.
        """
        return []

    def _build_environment_entries(self) -> list[dict[str, Any]]:
        """Build YAML build_environment entries for Hailo DFC and DeepX DX-COM."""
        hailo_cache = "~/Models/BackendArtifacts/hailo"
        entries: list[dict[str, Any]] = [
            {
                "id": "hailo8_dfc_managed",
                "kind": "hailo8_dfc",
                "host": "local",
                "shell": "bash",
                "workdir": "~/.onnx_splitpoint_tool/hailo/builds/hailo8",
                "venv_activate": "source ~/.onnx_splitpoint_tool/hailo/venv_hailo8/bin/activate",
                "cache_dir": hailo_cache,
            },
            {
                "id": "hailo10_dfc_managed",
                "kind": "hailo10_dfc",
                "host": "local",
                "shell": "bash",
                "workdir": "~/.onnx_splitpoint_tool/hailo/builds/hailo10",
                "venv_activate": "source ~/.onnx_splitpoint_tool/hailo/venv_hailo10/bin/activate",
                "cache_dir": hailo_cache,
            },
        ]
        dx_root = str(self.var_deepx_root.get() or "").strip()
        dx_venv = str(self.var_deepx_compiler_venv.get() or "").strip()
        dx_cache = str(self.var_deepx_cache_dir.get() or "~/Models/BackendArtifacts/deepx").strip() or "~/Models/BackendArtifacts/deepx"
        if dx_root or dx_venv or self.var_run_deepx.get() or self.var_run_deepx_to_trt.get() or self.var_run_trt_to_deepx.get():
            if not dx_root:
                dx_root = "/home/jasmin/dx-all-suite"
            if not dx_venv:
                dx_venv = str(Path(dx_root) / "dx-compiler" / "venv-dx-compiler-local")
            entries.append({
                "id": "deepx_dxcom_x86",
                "kind": "deepx_dxcom",
                "host": "local",
                "shell": "bash",
                "dx_all_suite_root": dx_root,
                "compiler_overlay": str(self.var_deepx_compiler_overlay.get() or "").strip(),
                "venv_activate": "source " + dx_venv.rstrip("/") + "/bin/activate",
                "cache_dir": dx_cache,
            })
        return entries

    def _run_profiles(self) -> list[dict[str, Any]]:
        """Return only logical run profiles. Physical setups are resolved centrally."""
        out: list[dict[str, Any]] = []
        if self.var_run_cpu.get():
            out.append({"id": "ort_cpu", "type": "same_backend_reference", "full": "cpu", "stage1": "cpu", "stage2": "cpu", "required": True})
        if self.var_run_cuda.get():
            out.append({"id": "ort_cuda", "type": "same_backend_reference", "full": "cuda", "stage1": "cuda", "stage2": "cuda", "required": True})
        if self.var_run_trt.get():
            out.append({"id": "ort_tensorrt", "type": "same_backend_reference", "full": "tensorrt", "stage1": "tensorrt", "stage2": "tensorrt", "required": True})
        if self.var_run_hailo.get():
            out.append({"id": "hailo8", "type": "same_backend_reference", "full": "hailo8", "stage1": "hailo8", "stage2": "hailo8", "required": False, "treat_preflight_block_as_result": True})
        if self.var_run_hailo_to_trt.get():
            out.append({"id": "hailo8_to_trt", "type": "mixed_backend", "full_reference": "cpu_full", "stage1": "hailo8", "stage2": "tensorrt", "required": False, "treat_preflight_block_as_result": True})
        if self.var_run_trt_to_hailo.get():
            out.append({"id": "trt_to_hailo8", "type": "mixed_backend", "full_reference": "cpu_full", "stage1": "tensorrt", "stage2": "hailo8", "required": False, "treat_preflight_block_as_result": True, "activation_calibration_source": "activation_proxy_cache"})
        if self.var_run_hailo10.get():
            out.append({"id": "hailo10", "type": "same_backend_reference", "full": "hailo10", "stage1": "hailo10", "stage2": "hailo10", "required": False, "treat_preflight_block_as_result": True})
        if self.var_run_hailo10_to_trt.get():
            out.append({"id": "hailo10_to_tensorrt", "type": "mixed_backend", "full_reference": "cpu_full", "stage1": "hailo10", "stage2": "tensorrt", "required": False, "treat_preflight_block_as_result": True})
        if self.var_run_deepx.get():
            out.append({"id": "deepx_m1_full", "type": "same_backend_reference", "full": "deepx_m1", "stage1": "deepx_m1", "stage2": "deepx_m1", "required": False})
        if self.var_run_deepx_to_trt.get():
            out.append({"id": "deepx_m1_to_tensorrt", "type": "mixed_backend", "full_reference": "cpu_full", "stage1": "deepx_m1", "stage2": "tensorrt", "required": False})
        if self.var_run_trt_to_deepx.get():
            out.append({"id": "tensorrt_to_deepx_m1", "type": "mixed_backend", "full_reference": "cpu_full", "stage1": "tensorrt", "stage2": "deepx_m1", "required": False, "activation_calibration_source": "activation_proxy_cache"})
        return out

    def _build_payload(self) -> dict[str, Any]:
        self._commit_current_model_edit_if_needed()
        self.models = self._normalized_models_for_payload()
        if bool(self.var_auto_bind_dataset_registry.get()):
            self._load_dataset_registry_into_profile(fill_only_missing=True, silent=True)
        name = str(self.var_name.get() or "").strip()
        if not name:
            raise ValueError("Profile ID / Name fehlt.")
        if not self.models:
            raise ValueError("Mindestens ein Modell muss im Profil stehen.")
        run_profiles = self._run_profiles()
        if not run_profiles:
            raise ValueError("Mindestens ein Hardware-/Run-Profil muss ausgewählt sein.")
        pool_raw = str(self.var_pool.get() or "auto").strip() or "auto"
        try:
            pool: int | str = int(pool_raw)
        except Exception:
            pool = "auto"
        execution_mode = str(self.var_workflow_execution_mode.get() or "generate_and_run").strip() or "generate_and_run"
        skip_runtime = bool(self.var_workflow_skip_runtime.get())
        if execution_mode == "generate_and_run":
            skip_runtime = False
            self.var_workflow_skip_runtime.set(False)

        payload: dict[str, Any] = {
            "name": name,
            "purpose": str(self.var_purpose.get() or "").strip(),
            "selection_policy": {
                "backend_backfill": {"enabled": bool(self.var_backend_backfill.get()),
                    "technical_output_contract_version": self.backend_output_contract_version,
                    **{key: int(value.get()) for key, value in self.var_backfill_limits.items()}},
                "max_accepted_cases_per_model": max(1, int(self.var_cases.get() or 1)),
                "preferred_shortlist": max(1, int(self.var_shortlist.get() or 1)),
                "min_gap": max(0, int(self.var_min_gap.get() or 0)),
                "candidate_search_pool": pool,
                "require_single_part2_input": bool(self.var_require_single_part2_input.get()),
                "selection_strategy": str(self.var_selection_strategy.get() or "stratified_windows"),
                "score_independent_audit_enabled": str(self.var_selection_strategy.get() or "").strip() == "score_independent_audit",
                "audit_candidate_universe": "deterministic_audit",
                "audit_size": max(1, int(self.var_audit_size.get() or 20)),
                "minimum_valid_audit_candidates": max(1, int(self.var_audit_min_valid.get() or 10)),
                "audit_seed": int(self.var_audit_seed.get() or 20260710),
                "report_blocked_configs": bool(self.var_report_blocked.get()),
                "report_plan_adjustments": True,
                "keep_partial_hailo_cases": bool(self.var_keep_partial_hailo.get()),
                "full_model_hailo_preflight_policy": str(self.var_full_preflight.get() or "skip"),
            },
            "model_suite": {
                "primary": [copy.deepcopy(dict(row)) for row in self.models],
            },
            "run_profiles": run_profiles,
            "hardware": {
                "selected_setups": self._selected_hardware_setup_ids(),
                "selected_groups": _split_csv(self.var_hardware_selected_groups.get()),
                "setups_file": str(self.var_hardware_setups_file.get() or "").strip(),
            },
            "validation": {
                "split_fidelity_reference_mode": str(self.var_reference_mode.get() or "auto").strip() or "auto",
                "classification_metrics": _split_csv(self.var_classification_metrics.get()),
                "detection_metrics": _split_csv(self.var_detection_metrics.get()),
                "backend_drift_reference": str(self.var_backend_drift_reference.get() or "cpu_full").strip() or "cpu_full",
                "report_blocked_and_infeasible": bool(self.var_report_blocked.get()),
                "mode": str(self.var_validation_mode.get() or "summary_only"),
                "require_explicit": bool(self.var_require_explicit_validation.get()),
                "require_task_metrics": bool(self.var_require_task_metrics.get()),
            },
            "campaign": {
                "id": str(self.var_campaign_id.get() or name).strip() or name,
                "claim_scope": str(self.var_claim_scope.get() or EVALUATED_MATRIX_CLAIM_SCOPE).strip() or EVALUATED_MATRIX_CLAIM_SCOPE,
                "mode": str(self.var_campaign_mode.get() or "development").strip() or "development",
                "enforcement": str(self.var_campaign_enforcement.get() or "warn").strip() or "warn",
                "frozen_before_final_campaign": bool(self.var_campaign_frozen.get()),
                "dataset_registry": str(self.var_dataset_registry.get() or default_registry_path()).strip(),
                "auto_bind_dataset_registry": bool(self.var_auto_bind_dataset_registry.get()),
                "dataset_manifests": {
                    "classification": {
                        "calibration": str(self.var_manifest_cls_calib.get() or "").strip(),
                        "validation": str(self.var_manifest_cls_val.get() or "").strip(),
                    },
                    "detection": {
                        "calibration": str(self.var_manifest_det_calib.get() or "").strip(),
                        "validation": str(self.var_manifest_det_val.get() or "").strip(),
                    },
                },
                "pipeline_contract_manifest": str(self.var_pipeline_contract_manifest.get() or "").strip(),
                "holdout_registry": str(self.var_holdout_registry.get() or "").strip(),
                "ranking_model_bundle": str(self.var_ranking_model_bundle.get() or "").strip(),
                "energy_calibration_manifest": str(self.var_energy_calibration_manifest.get() or "").strip(),
                "require_fitted_stage_time": bool(self.var_require_fitted_stage_time.get()),
                "require_native_handover_model": bool(self.var_require_native_handover.get()),
                "require_campaign_freeze": bool(self.var_require_campaign_freeze.get()),
                "require_prediction_freeze_approval": bool(self.var_require_prediction_approval.get()),
                "require_cryptographic_prediction_signature": bool(self.var_require_prediction_signature.get()),
                "prediction_freeze_public_key": str(self.var_prediction_public_key.get() or "").strip(),
            },
            "quality_gate": {
                "schema": "onnx-splitpoint/task-quality-policy",
                "schema_version": 3,
                "name": str(self.var_quality_profile_id.get() or "task_quality_v2").strip() or "task_quality_v2",
                "profile_id": str(self.var_quality_profile_id.get() or "task_quality_v2").strip() or "task_quality_v2",
                "frozen_before_final_campaign": bool(self.var_quality_frozen.get()),
                "dataset_tier": str(self.var_quality_tier.get() or "screening").strip() or "screening",
                "canonical_reference": str(self.var_quality_canonical_ref.get() or "canonical_full_onnx").strip() or "canonical_full_onnx",
                "classification": {
                    "primary_metric": "top1_accuracy",
                    "non_inferiority_margin": max(0.0, float(self.var_quality_cls_margin_pp.get() or 0.0)) / 100.0,
                    "guardrails": {"top5_accuracy_margin": max(0.0, float(self.var_quality_cls_top5_margin_pp.get() or 0.0)) / 100.0},
                },
                "detection": {
                    "primary_metric": "coco_ap_50_95",
                    "non_inferiority_margin": max(0.0, float(self.var_quality_det_margin_ap.get() or 0.0)) / 100.0,
                    "guardrails": {
                        "ap50_margin": max(0.0, float(self.var_quality_det_ap50_margin.get() or 0.0)) / 100.0,
                        "ap75_margin": max(0.0, float(self.var_quality_det_ap75_margin.get() or 0.0)) / 100.0,
                    },
                },
                "statistics": {
                    "method": "paired_bootstrap",
                    "confidence_level": min(0.999999, max(0.5, float(self.var_quality_confidence.get() or 0.95))),
                    "bootstrap_repetitions": max(100, int(self.var_quality_bootstrap.get() or 5000)),
                    "seed": int(self.var_quality_seed.get() or 20260710),
                    "decision": "lower_one_sided_bound",
                    "execution_location": str(self.var_quality_execution_location.get() or "central_management").strip() or "central_management",
                    "workers": max(1, int(self.var_quality_workers.get() or 4)),
                },
                "native_contract": {
                    "detection_self_reference_min_match_ratio": 0.80,
                    "detection_self_reference": {
                        "policy_id": "class_aware_iou50_postnms_v2",
                        "min_reference_match_ratio": 0.80,
                        "min_mean_matched_iou": 0.85,
                        "iou_threshold": 0.50,
                        "confidence_threshold": 0.25,
                        "denominator": "reference_detections",
                        "class_aware": True,
                    },
                    "numerical_similarity_required_for_claim": True,
                    "self_reference_counts_as_task_accuracy": False,
                },
                "screening_eligible_for_ranking": False,
                "legacy_point_estimate_eligible_for_ranking": False,
            },
            "ranking_validation": {
                "enabled": bool(self.var_ranking_enabled.get()),
                "holdout_unit": "model_direction_runner",
                "candidate_universe": "model_declared",
                "require_frozen_predictions": bool(self.var_ranking_require_frozen.get()),
                "require_complete_candidate_universe": bool(self.var_ranking_require_complete.get()),
                "k_values": [max(1, int(x)) for x in _split_csv(self.var_ranking_k_values.get())] or [1, 3, 5],
                "elite_q_values": [max(1, int(x)) for x in _split_csv(self.var_ranking_q_values.get())] or [1, 3],
                "primary_k": max(1, int(self.var_ranking_primary_k.get() or 5)),
                "minimum_candidates_for_correlation": max(2, int(self.var_ranking_min_corr.get() or 3)),
                "near_optimal_relative_epsilon": max(0.0, float(self.var_ranking_epsilon_pct.get() or 0.0)) / 100.0,
                "bootstrap_repetitions": max(100, int(self.var_ranking_bootstrap.get() or 5000)),
                "bootstrap_seed": int(self.var_ranking_seed.get() or 20260710),
                "targets": ["pipeline_cycle_ms"],
                "methods": ["cut_bytes_only", "weighted_score", "cycle_time_no_handover", "cycle_time_with_handover", "onnx_real_boundary_hardware_aware"],
                "weighted_score": {"w_comm": 1.0, "w_imb": 3.0, "w_tensors": 0.2, "log_comm": True},
                "cycle_time_no_handover": {"backend_throughput_gops": {}, "stage_time_models": {}},
                "cycle_time_with_handover": {"backend_throughput_gops": {}, "stage_time_models": {}, "handover_models": {"generic": {}, "native_fifo": {}}},
            },
            "official_coco_evaluation": {
                "enabled": bool(self.var_official_coco_enabled.get()),
                "required_for_final": bool(self.var_official_coco_required.get()),
                "annotations": str(self.var_official_coco_annotations.get() or "").strip(),
                "remote_annotations": str(self.var_official_coco_remote_annotations.get() or "").strip(),
                "iou_type": "bbox",
                "archive_predictions": True,
                "archive_eval_tensors": bool(self.var_official_coco_archive_tensors.get()),
                "max_detections": [max(1, int(x)) for x in _split_csv(self.var_official_coco_max_dets.get())] or [1, 10, 100],
                "require_pycocotools": bool(self.var_official_coco_required.get()),
                "note": "Official pycocotools verification is archived in addition to the paired-bootstrap quality gate.",
            },
            "reporting": {
                "include_decision_summary": bool(self.var_report_decisions.get()),
                "include_backend_drift_block": bool(self.var_report_backend_drift.get()),
                "include_task_specific_quality_block": bool(self.var_report_task_quality.get()),
                "aggregate_by_model": True,
                "aggregate_by_run_profile": True,
                "canonical_scientific_report": True,
                "cleanup_legacy_reports": bool(self.var_cleanup_legacy_reports.get()),
                "generate_thesis_tex": bool(self.var_generate_thesis_tex.get()),
                "generate_thesis_figures": bool(self.var_generate_thesis_figures.get()),
                "include_campaign_readiness": bool(self.var_include_campaign_readiness.get()),
            },
            "workflow": {
                "execution_mode": execution_mode,
                "skip_runtime_benchmarks": skip_runtime,
                "no_model_hash": bool(self.var_workflow_no_model_hash.get()),
                "include_reserve": bool(self.var_workflow_include_reserve.get()),
                "parallel_remote_setups": bool(self.var_parallel_remote_setups.get()),
                "max_parallel_setups": max(1, int(self.var_max_parallel_setups.get() or 3)),
                "max_parallel_uploads": max(0, int(self.var_max_parallel_uploads.get() or 1)),
                "powercalc_workers": max(0, int(self.var_powercalc_workers.get() or 1)),
                "artifact_cache_preflight": {
                    "enabled": True,
                    "default_expectation": "unspecified",
                    "block_on_unexpected_cold_builds": False,
                },
            },
            "benchmark_execution": {
                "provider": str(self.var_benchmark_provider.get() or "auto"),
                "warmup": max(0, int(self.var_benchmark_warmup.get() or 0)),
                "runs": max(1, int(self.var_benchmark_runs.get() or 1)),
                "timeout_s": max(0, int(self.var_benchmark_timeout.get() or 0)),
                "backend": "auto",
            },
            "energy": {
                "enabled": False,
                "generic_enabled": False,
                "measurement_path": "native_only",
                "requested_native_energy": bool(self.var_energy_enabled.get()),
                "final_all_split_energy": bool(self.var_energy_final_all_splits.get()),
                "final_energy_skip_cpu_ort": bool(self.var_energy_final_skip_cpu_ort.get()) if bool(self.var_energy_final_all_splits.get()) else False,
                "scope": "row_variant" if bool(self.var_energy_final_all_splits.get()) else str(self.var_energy_scope.get() or "row_variant"),
                "repeat_override": max(0, int(self.var_energy_repeat_override.get() or 0)),
                "phases": _split_csv(self.var_energy_phases.get()) or ["latency", "streaming"],
                "target_policy": "all" if bool(self.var_energy_final_all_splits.get()) else str(self.var_energy_target_policy.get() or "canonical_only"),
                "skip_backends": (["ort_cpu"] if bool(self.var_energy_final_all_splits.get()) and bool(self.var_energy_final_skip_cpu_ort.get()) else ([] if bool(self.var_energy_final_all_splits.get()) else _split_csv(self.var_energy_skip_backends.get()))),
                "include_run_ids": _split_csv(self.var_energy_include_run_ids.get()),
                "exclude_run_ids": [] if bool(self.var_energy_final_all_splits.get()) else _split_csv(self.var_energy_exclude_run_ids.get()),
                "heartbeat_s": max(10, int(self.var_energy_heartbeat_s.get() or 60)),
                "max_targets_per_run_id": 0 if bool(self.var_energy_final_all_splits.get()) else max(0, int(self.var_energy_max_targets_per_run_id.get() or 0)),
                "max_work_units_per_window": max(0, int(self.var_energy_max_work_units_per_window.get() or 0)),
                "max_window_duration_s": max(0, int(self.var_energy_max_window_duration_s.get() or 0)),
                "timeout_s_per_window": max(0, int(self.var_energy_timeout_s_per_window.get() or 0)),
                "sizing_probe_max_work_units": max(1, int(self.var_energy_sizing_probe_max_work_units.get() or 256)),
                "include_raw_parquet_in_debug_pack": bool(self.var_energy_include_raw_parquet.get()),
                "strict": bool(self.var_energy_strict.get() or self.var_energy_final_all_splits.get()),
            },
            "native_producers": {
                "enabled": bool(self.var_native_enabled.get()),
                "backends": _split_csv(self.var_native_backends.get()) or ["hailo8", "hailo10h", "deepx"],
                "case_policy": str(self.var_native_case_policy.get() or "all_accepted"),
                "precision": str(self.var_native_precision.get() or "uint8_cast_fp16"),
                "frames": max(1, int(self.var_native_frames.get() or 100)),
                "warmup": max(0, int(self.var_native_warmup.get())),
                "repetitions": max(1, int(self.var_native_repetitions.get())),
                "queue_depth": max(1, int(self.var_native_queue_depth.get() or 3)),
                "inflight": max(1, int(self.var_native_inflight.get() or 8)),
                "hailo_format": "uint8",
                "remote_root": "/home/nx/native_fifo_evalsets",
                "remote_tool_dir": "/home/nx/ONNX-Splitpoint-Tool",
                "build_missing_engines": getattr(self, "_loaded_native_build_missing_engines", True),
                "copy_benchmarksets": True,
                "strict_supported_only": True,
                "full_baselines": {
                    "enabled": bool(self.var_native_enabled.get() and native_full_backends_from_run_profiles(run_profiles)),
                    "backends": native_full_backends_from_run_profiles(run_profiles),
                    "source": "selected_full_run_profiles",
                    "auto_enabled": True,
                },
                "validation": {"enabled": bool(self.var_native_validation_enabled.get()), "mode": "dump_and_visual", "topk": 5},
                "dump_outputs": bool(self.var_native_validation_enabled.get()),
                "energy": {
                    "enabled": bool(self.var_native_enabled.get() and self.var_energy_enabled.get()),
                    "mode": "measure" if bool(self.var_native_enabled.get() and self.var_energy_enabled.get()) else "plan",
                    "duration_s": int(self.var_native_energy_duration_s.get() or 0),
                    "include_split_rows": True,
                    "include_full_baselines": True,
                    "window_method_validation_probe": {
                        "enabled": bool(self.var_window_method_probe_enabled.get()),
                        "repeats": max(1, int(self.var_window_method_probe_repeats.get() or 3)),
                        "minimum_decision_repeats": 3,
                        "include_raw_parquet": bool(self.var_window_method_probe_include_raw.get()),
                        "strict": bool(self.var_window_method_probe_strict.get()),
                        "screening_only": True,
                        "eligible_for_energy_results_import": False,
                        "eligible_for_scientific_claim": False,
                    },
                },
                "remotes": {
                    "hailo8": {"ssh": "nx@192.168.0.104"},
                    "hailo10h": {"ssh": "nx@192.168.0.145", "env": "export PYTHONNOUSERSITE=1; source ~/venvs/hailo10/bin/activate"},
                    "deepx": {"ssh": "nx@192.168.0.102", "env": "source ~/venvs/deepx-runtime/bin/activate"},
                },
            },
            "model_preparation": {
                "mode": str(self.var_prep_mode.get() or "current"),
                "note": "Generated by Evaluation Profile editor.",
            },
            "hailo_build": {
                "mode": str(self.var_hailo_build_mode.get() or "reuse_and_build_missing"),
                "hw_arch": str(self.var_hailo_arch.get() or "hailo8"),
                "targets": _split_csv(self.var_hailo_targets.get()) or [str(self.var_hailo_arch.get() or "hailo8")],
                "timeout_s": parse_hailo_timeout_seconds(
                    self.var_hailo_timeout.get(),
                    default=0,
                    label="hailo_build.timeout_s",
                ),
                "build_full": bool(self.var_hailo_build_full.get()),
                "build_part1": bool(self.var_hailo_build_part1.get()),
                "build_part2": bool(self.var_hailo_build_part2.get()),
                "preset": str(self.var_hailo_preset.get() or "quick"),
                "optimization_level": max(0, int(self.var_hailo_opt_level.get() or 0)),
                "calib_dir": str(self.var_hailo_calib_dir.get() or "").strip(),
                "calib_count": max(0, int(self.var_hailo_calib_count.get() or 0)),
                "calib_batch_size": max(1, int(self.var_hailo_calib_batch.get() or 1)),
                "force_build": parse_config_bool(self.var_hailo_force_build.get(), field="hailo_build.force_build"),
                "keep_artifacts": bool(self.var_hailo_keep_artifacts.get()),
                "source": "evaluation_profile_yaml",
            },
            "hardware_smoke": {
                "mode": str(self.var_hw_smoke_mode.get() or "summary_only"),
                "require_remote_for_hailo": False,
                "note": "Generated by Evaluation Profile editor.",
            },
            "build_environments": self._build_environment_entries(),
            "deepx_build": {
                "mode": "reuse_and_build_missing" if (self.var_run_deepx.get() or self.var_run_deepx_to_trt.get() or self.var_run_trt_to_deepx.get()) else "reuse_only",
                "target": "deepx_m1",
                "dx_all_suite_root": str(self.var_deepx_root.get() or "~/dx-all-suite").strip(),
                "compiler_overlay": str(self.var_deepx_compiler_overlay.get() or "").strip(),
                "compiler_venv": str(self.var_deepx_compiler_venv.get() or "~/dx-all-suite/dx-compiler/venv-dx-compiler-local").strip(),
                "cache_dir": str(self.var_deepx_cache_dir.get() or "~/Models/BackendArtifacts/deepx").strip(),
                "classification_preprocessing": str(
                    self.var_deepx_classification_preprocessing.get()
                    or "imagenet_mean_std"
                ).strip(),
                "calib_dir": str(self.var_hailo_calib_dir.get() or "").strip(),
                "calib_count": max(0, int(self.var_hailo_calib_count.get() or 0)),
                "calibration_method": "ema",
                "opt_level": 0,
                "force_build": False,
            },
            "implementation_note": "Generated by the ONNX Splitpoint Tool Evaluation Profile editor; edit YAML directly only for unsupported advanced fields.",
        }
        reserve_ids = _split_csv(self.var_reserve.get())
        if reserve_ids:
            preserved_by_id: dict[str, Any] = {}
            for item in self._reserve_model_entries:
                item_id = str(item.get("id") if isinstance(item, Mapping) else item).strip()
                if item_id and item_id not in preserved_by_id:
                    preserved_by_id[item_id] = item
            payload["model_suite"]["reserve"] = [
                copy.deepcopy(preserved_by_id.get(item_id, item_id))
                for item_id in reserve_ids
            ]
        if isinstance(self._measurement_campaign_passthrough, Mapping):
            payload["measurement_campaign"] = copy.deepcopy(dict(self._measurement_campaign_passthrough))
        if isinstance(self._execution_guard_passthrough, Mapping):
            payload["execution_guard"] = copy.deepcopy(
                dict(self._execution_guard_passthrough)
            )
            if isinstance(self._cache_verify_native_passthrough, Mapping):
                payload["native_producers"] = copy.deepcopy(
                    dict(self._cache_verify_native_passthrough)
                )
            if isinstance(
                self._cache_verify_forced_cases_passthrough, Mapping
            ):
                payload.setdefault("selection_policy", {})["forced_cases"] = (
                    copy.deepcopy(
                        dict(self._cache_verify_forced_cases_passthrough)
                    )
                )
        if isinstance(getattr(self, "_protocol_freeze_passthrough", None), Mapping):
            payload["campaign"]["protocol_freeze"] = copy.deepcopy(dict(self._protocol_freeze_passthrough))
            payload["campaign"]["require_protocol_freeze"] = bool(
                getattr(self, "_require_protocol_freeze_passthrough", False)
            )
        prediction_freeze_enabled = getattr(
            self, "_prediction_freeze_enabled_passthrough", None
        )
        if prediction_freeze_enabled is not None:
            payload["campaign"]["prediction_freeze_enabled"] = bool(
                prediction_freeze_enabled
            )
        if isinstance(
            getattr(self, "_ranking_validation_passthrough", None), Mapping
        ):
            ranking_validation = copy.deepcopy(
                dict(self._ranking_validation_passthrough)
            )
            ranking_validation.update(dict(payload.get("ranking_validation") or {}))
            payload["ranking_validation"] = ranking_validation
        root_hint = str(self.var_models_root_hint.get() or "").strip()
        if root_hint:
            payload["models_root_hint"] = root_hint
        legacy_remote_enabled = bool(self.var_remote_enabled.get()) and str(os.environ.get("ONNX_SPLITPOINT_PROFILE_LEGACY_REMOTE", "")).strip().lower() in {"1", "true", "yes", "on"}
        if legacy_remote_enabled:
            host_id = _host_display_to_id(self.var_remote_host_id.get())
            host_copy = self._remote_host_config_by_id(host_id) if host_id else None
            if host_copy is None and str(self.var_remote_host.get() or "").strip():
                host_copy = {
                    "id": host_id or _norm_id(self.var_remote_host.get()) or "workflow_remote",
                    "label": host_id or str(self.var_remote_host.get() or "").strip(),
                    "host": str(self.var_remote_host.get() or "").strip(),
                    "user": str(self.var_remote_user.get() or "").strip(),
                    "port": max(1, int(self.var_remote_port.get() or 22)),
                    "remote_base_dir": str(self.var_remote_base.get() or "~/splitpoint_runs").strip() or "~/splitpoint_runs",
                    "ssh_extra_args": str(self.var_remote_ssh_extra_args.get() or "").strip(),
                }
            if host_copy:
                self._apply_remote_host_config_to_fields(host_copy)
            payload["remote_execution"] = {
                "enabled": True,
                "host_id": host_id or str((host_copy or {}).get("id") or ""),
                "hosts": [dict(host_copy)] if isinstance(host_copy, Mapping) else [],
                "host": str(self.var_remote_host.get() or "").strip(),
                "user": str(self.var_remote_user.get() or "").strip(),
                "port": max(1, int(self.var_remote_port.get() or 22)),
                "remote_base_dir": str(self.var_remote_base.get() or "~/splitpoint_runs").strip() or "~/splitpoint_runs",
                "ssh_extra_args": str(self.var_remote_ssh_extra_args.get() or "").strip(),
                "provider": str(self.var_remote_provider.get() or "auto"),
                "warmup": max(0, int(self.var_remote_warmup.get() or 0)),
                "iters": max(1, int(self.var_remote_iters.get() or 1)),
                "timeout_s": max(0, int(self.var_remote_timeout.get() or 0)),
                "remote_venv": str(self.var_remote_venv.get() or "").strip(),
                "transfer_mode": str(self.var_remote_transfer_mode.get() or "bundle"),
                "reuse_bundle": bool(self.var_remote_reuse_bundle.get()),
                "resume": bool(self.var_remote_resume.get()),
            }
        payload["execution_preset"] = {
            "id": normalize_mode_id(self.var_run_mode_id.get()),
            "follow_tool_config": parse_config_bool(
                self.var_run_mode_follow_tool_config.get(), field="execution_preset.follow_tool_config"
            ),
            "overrides": {
                "native_enabled": bool(self.var_native_enabled.get()),
                "energy_enabled": bool(self.var_energy_enabled.get()),
            },
        }
        previous_preset = getattr(self, "_loaded_run_mode_preset", {})
        for key in ("snapshot", "native_budget_sources"):
            if key in previous_preset:
                payload["execution_preset"][key] = copy.deepcopy(previous_preset[key])
        baseline = getattr(self, "_native_budget_widget_baseline", {})
        explicit = dict((previous_preset.get("overrides") or {}).get("native_performance") or {})
        for field in ("frames", "warmup", "repetitions"):
            value = payload["native_producers"][field]
            if field in baseline and value != baseline[field]:
                explicit[field] = value
            elif field in baseline:
                explicit.pop(field, None)
                # The widgets just followed a different mode; mark their
                # baseline as that mode so the resolver does not infer an override.
                payload["execution_preset"].setdefault("snapshot", {}).setdefault("runtime", {}).setdefault("native", {})[field] = value
                payload["execution_preset"].setdefault("native_budget_sources", {})[field] = "tool_config"
        if explicit:
            payload["execution_preset"]["overrides"]["native_performance"] = explicit
        for key, value in getattr(self, "_native_energy_budget_passthrough", {}).items():
            payload["native_producers"]["energy"][key] = copy.deepcopy(value)
        loaded_compute = getattr(self, "_loaded_hailo_compute_by_family", None)
        if loaded_compute is not None:
            # This field is edited in Tool Config (or explicitly in YAML).
            # A profile preview/save must not silently erase that choice.
            payload["hailo_build"]["compute_by_family"] = copy.deepcopy(loaded_compute)
            payload["execution_preset"]["build_provenance"] = copy.deepcopy(previous_preset.get("build_provenance") or {})
        if getattr(self, "_loaded_workflow_stop_after", None):
            payload.setdefault("workflow", {})["stop_after"] = self._loaded_workflow_stop_after
        # Cache/build admission is an explicit profile contract, not a mode
        # effort default. An editor roundtrip must retain a warm-only scope.
        for location, value in getattr(self, "_loaded_cache_admission", {}).items():
            target = payload if location == "profile" else payload["workflow"]
            target["artifact_cache_preflight"] = copy.deepcopy(value)
        if normalize_mode_id(previous_preset.get("id")) == payload["execution_preset"]["id"]:
            # An unchanged archived choice is still that bound snapshot. A
            # preview/save must not resolve it through today's registry.
            for key in ("config_path", "config_sha256"):
                if key in previous_preset:
                    payload["execution_preset"][key] = copy.deepcopy(previous_preset[key])
            if not payload["execution_preset"]["follow_tool_config"]:
                for key in ("snapshot", "snapshot_sha256"):
                    if key in previous_preset:
                        payload["execution_preset"][key] = copy.deepcopy(previous_preset[key])
        # These two controls are deliberate profile-level quality settings.
        # Preserve them across run-mode materialisation instead of silently
        # replacing a loaded/custom contract with the current mode defaults.
        quality_execution_location = str(
            self.var_quality_execution_location.get() or "central_management"
        ).strip() or "central_management"
        quality_workers = max(1, int(self.var_quality_workers.get() or 4))
        deepx_classification_preprocessing = str(
            self.var_deepx_classification_preprocessing.get()
            or "imagenet_mean_std"
        ).strip().lower()
        if deepx_classification_preprocessing not in {
            "current_scale_only", "imagenet_mean_std",
        }:
            raise ValueError(
                "DeepX classification preprocessing must be "
                "current_scale_only or imagenet_mean_std."
            )
        claim_scope = str(
            self.var_claim_scope.get() or EVALUATED_MATRIX_CLAIM_SCOPE
        ).strip() or EVALUATED_MATRIX_CLAIM_SCOPE
        payload, _mode_audit = apply_run_mode(
            payload,
            mode_id=normalize_mode_id(self.var_run_mode_id.get()),
            follow_tool_config=payload["execution_preset"]["follow_tool_config"],
        )
        effective_deepx_build = dict(payload.get("deepx_build") or {})
        effective_deepx_build["classification_preprocessing"] = (
            deepx_classification_preprocessing
        )
        payload["deepx_build"] = effective_deepx_build
        effective_quality = dict(payload.get("quality_gate") or {})
        effective_statistics = dict(effective_quality.get("statistics") or {})
        effective_statistics["execution_location"] = quality_execution_location
        effective_statistics["workers"] = quality_workers
        effective_quality["statistics"] = effective_statistics
        payload["quality_gate"] = effective_quality
        # Run-mode projection intentionally owns execution defaults, not the
        # separately amended prospective protocol. Restore that immutable block
        # after applying the mode so an editor roundtrip cannot drop the seal.
        campaign = dict(payload.get("campaign") or {})
        campaign["claim_scope"] = claim_scope
        if isinstance(getattr(self, "_protocol_freeze_passthrough", None), Mapping):
            campaign["protocol_freeze"] = copy.deepcopy(dict(self._protocol_freeze_passthrough))
            campaign["require_protocol_freeze"] = bool(
                getattr(self, "_require_protocol_freeze_passthrough", False)
            )
        payload["campaign"] = campaign
        return validate_evaluation_profile_payload(payload, source="profile editor")

    def _apply_payload(self, payload: Mapping[str, Any], *, path: str = "") -> None:
        validate_profile_config_booleans(payload)
        self._loading_profile = True
        data = dict(payload or {})
        self._loaded_cache_admission = {
            location: copy.deepcopy(value["artifact_cache_preflight"])
            for location, value in (("profile", data), ("workflow", data.get("workflow") or {}))
            if isinstance(value.get("artifact_cache_preflight"), Mapping)
        }
        preset = data.get("execution_preset") if isinstance(data.get("execution_preset"), Mapping) else {}
        self._loaded_run_mode_preset = copy.deepcopy(preset)
        self.var_run_mode_id.set(normalize_mode_id((preset or {}).get("id") or infer_run_mode(data)))
        self.var_run_mode_follow_tool_config.set(parse_config_bool(
            (preset or {}).get("follow_tool_config", True), field="execution_preset.follow_tool_config"
        ))
        self.var_path.set(path)
        self.var_name.set(str(data.get("name") or "custom_splitpoint_eval_v1"))
        self.var_purpose.set(str(data.get("purpose") or ""))
        self.var_models_root_hint.set(str(data.get("models_root_hint") or ""))
        sel = dict(data.get("selection_policy") or {})
        from ..backend_backfill import DEFAULT_BACKFILL
        backfill = sel.get('backend_backfill') or {'enabled': False}
        self.backend_output_contract_version = backfill.get('technical_output_contract_version', 0)
        self.var_backend_backfill.set(bool(backfill.get('enabled', False)))
        for key, variable in self.var_backfill_limits.items():
            variable.set(backfill.get(key, DEFAULT_BACKFILL[key]))

        self.var_cases.set(int(sel.get("max_accepted_cases_per_model") or 5))
        self.var_shortlist.set(int(sel.get("preferred_shortlist") or 10))
        self.var_min_gap.set(
            int(sel["min_gap"] if sel.get("min_gap") is not None else 2)
        )
        self.var_pool.set(str(sel.get("candidate_search_pool") if sel.get("candidate_search_pool") is not None else "auto"))
        self.var_require_single_part2_input.set(bool(sel.get("require_single_part2_input", False)))
        ranking_for_selection = (
            dict(data.get("ranking_validation") or {})
            if isinstance(data.get("ranking_validation"), Mapping)
            else {}
        )
        strategy = str(sel.get("selection_strategy") or sel.get("coverage_strategy") or "").strip()
        if not strategy and bool(sel.get("score_independent_audit_enabled", False)):
            strategy = "score_independent_audit"
        self.var_selection_strategy.set(strategy or "stratified_windows")
        audit_size = sel.get("audit_size")
        if audit_size is None:
            audit_size = ranking_for_selection.get("audit_size")
        self.var_audit_size.set(int(audit_size if audit_size is not None else 20))
        audit_min_valid = sel.get("minimum_valid_audit_candidates")
        if audit_min_valid is None:
            audit_min_valid = ranking_for_selection.get(
                "minimum_valid_audit_candidates"
            )
        self.var_audit_min_valid.set(
            int(audit_min_valid if audit_min_valid is not None else 10)
        )
        audit_seed = sel.get("audit_seed")
        if audit_seed is None:
            audit_seed = ranking_for_selection.get("audit_seed")
        if audit_seed is None:
            audit_seed = ranking_for_selection.get("bootstrap_seed")
        self.var_audit_seed.set(
            int(audit_seed if audit_seed is not None else 20260710)
        )
        self.var_keep_partial_hailo.set(bool(sel.get("keep_partial_hailo_cases", True)))
        self.var_report_blocked.set(bool(sel.get("report_blocked_configs", True)))
        self.var_full_preflight.set(str(sel.get("full_model_hailo_preflight_policy") or "skip"))

        suite = dict(data.get("model_suite") or {})
        primary_raw: list[ModelRow] = []
        for item in list(suite.get("primary") or []):
            if isinstance(item, Mapping):
                primary_raw.append(copy.deepcopy(dict(item)))
            else:
                primary_raw.append({"id": str(item)})
        development_families = {
            str(row.get("family_id") or row.get("family") or "").strip()
            for row in primary_raw
            if _model_usage_from_row(row) == MODEL_USAGE_DEVELOPMENT
            and str(row.get("family_id") or row.get("family") or "").strip()
        }
        quality_for_models = (
            dict(data.get("quality_gate") or {})
            if isinstance(data.get("quality_gate"), Mapping)
            else {}
        )
        default_model_tier = str(quality_for_models.get("dataset_tier") or "screening")
        self.models = [
            _derive_simple_model_contract(
                row,
                usage=_model_usage_from_row(row),
                selection_strategy=self.var_selection_strategy.get(),
                default_validation_tier=default_model_tier,
                development_family_ids=development_families,
                preserve_legacy=True,
            )
            for row in primary_raw
        ]
        reserve = []
        for item in list(suite.get("reserve") or []):
            if isinstance(item, Mapping):
                row = dict(item)
                if row.get("evaluation_role") not in (None, ""):
                    row["evaluation_role"] = normalize_evaluation_role(row.get("evaluation_role"), strict=True)
                reserve.append(row)
            else:
                reserve.append(item)
        self._reserve_model_entries = copy.deepcopy(reserve)
        self.var_reserve.set(", ".join(str(x.get("id") if isinstance(x, Mapping) else x) for x in reserve if str(x).strip()))
        measurement_campaign = data.get("measurement_campaign")
        self._measurement_campaign_passthrough = (
            copy.deepcopy(dict(measurement_campaign))
            if isinstance(measurement_campaign, Mapping)
            else None
        )
        execution_guard = data.get("execution_guard")
        self._execution_guard_passthrough = (
            copy.deepcopy(dict(execution_guard))
            if isinstance(execution_guard, Mapping)
            and str(execution_guard.get("mode") or "").strip().lower().replace("-", "_")
            == "cache_verify_only"
            else None
        )
        self._cache_verify_native_passthrough = (
            copy.deepcopy(dict(data.get("native_producers") or {}))
            if self._execution_guard_passthrough is not None
            and isinstance(data.get("native_producers"), Mapping)
            else None
        )
        self._cache_verify_forced_cases_passthrough = (
            copy.deepcopy(dict(sel.get("forced_cases") or {}))
            if self._execution_guard_passthrough is not None
            and isinstance(sel.get("forced_cases"), Mapping)
            else None
        )
        campaign_for_protocol = data.get("campaign") if isinstance(data.get("campaign"), Mapping) else {}
        protocol_for_passthrough = campaign_for_protocol.get("protocol_freeze") if isinstance(campaign_for_protocol.get("protocol_freeze"), Mapping) else None
        self._protocol_freeze_passthrough = copy.deepcopy(dict(protocol_for_passthrough)) if protocol_for_passthrough is not None else None
        self._require_protocol_freeze_passthrough = bool(campaign_for_protocol.get("require_protocol_freeze", False))
        self._prediction_freeze_enabled_passthrough = (
            bool(campaign_for_protocol.get("prediction_freeze_enabled"))
            if "prediction_freeze_enabled" in campaign_for_protocol
            else None
        )
        ranking_validation = data.get("ranking_validation")
        self._ranking_validation_passthrough = (
            copy.deepcopy(dict(ranking_validation))
            if isinstance(ranking_validation, Mapping)
            else None
        )
        self._selected_model_index = 0 if self.models else None
        self._refresh_model_tree()
        if self.models:
            try:
                self.model_tree.selection_set("0")
                self.model_tree.see("0")
            except Exception:
                pass
            self._fill_model_fields(self.models[0])
        else:
            self._clear_model_fields()

        run_ids = {str(r.get("id") or "") for r in list(data.get("run_profiles") or []) if isinstance(r, Mapping)}
        self.var_run_cpu.set("ort_cpu" in run_ids)
        self.var_run_cuda.set("ort_cuda" in run_ids)
        self.var_run_trt.set("ort_tensorrt" in run_ids)
        self.var_run_hailo.set("hailo8" in run_ids)
        self.var_run_hailo_to_trt.set("hailo8_to_trt" in run_ids)
        self.var_run_trt_to_hailo.set("trt_to_hailo8" in run_ids)
        self.var_run_hailo10.set("hailo10" in run_ids or "hailo10_full" in run_ids)
        self.var_run_hailo10_to_trt.set("hailo10_to_tensorrt" in run_ids or "hailo10_to_trt" in run_ids)
        self.var_run_deepx.set("deepx_m1_full" in run_ids or "deepx_m1" in run_ids)
        self.var_run_deepx_to_trt.set("deepx_m1_to_tensorrt" in run_ids or "deepx_m1_to_trt" in run_ids)
        self.var_run_trt_to_deepx.set("tensorrt_to_deepx_m1" in run_ids or "trt_to_deepx_m1" in run_ids)

        benvs = data.get("build_environments") or []
        if isinstance(benvs, list):
            for env in benvs:
                if not isinstance(env, Mapping):
                    continue
                if "deepx" in str(env.get("kind") or "").lower():
                    self.var_deepx_root.set(str(env.get("dx_all_suite_root") or self.var_deepx_root.get()))
                    venv_act = str(env.get("venv_activate") or "")
                    m = re.search(r"source\s+(.+?)(?:/bin/activate)?$", venv_act.strip())
                    if m:
                        self.var_deepx_compiler_venv.set(m.group(1).rstrip("/"))
                    self.var_deepx_cache_dir.set(str(env.get("cache_dir") or self.var_deepx_cache_dir.get()))

        val = dict(data.get("validation") or {})
        self.var_validation_mode.set(str(val.get("mode") or "summary_only"))
        self.var_classification_metrics.set(", ".join(str(x) for x in list(val.get("classification_metrics") or [])))
        self.var_detection_metrics.set(", ".join(str(x) for x in list(val.get("detection_metrics") or [])))
        self.var_reference_mode.set(str(val.get("split_fidelity_reference_mode") or "auto"))
        self.var_backend_drift_reference.set(str(val.get("backend_drift_reference") or "cpu_full"))
        self.var_require_explicit_validation.set(bool(val.get("require_explicit", False)))
        self.var_require_task_metrics.set(bool(val.get("require_task_metrics", False)))
        rep = dict(data.get("reporting") or {})
        self.var_report_decisions.set(bool(rep.get("include_decision_summary", True)))
        self.var_report_backend_drift.set(bool(rep.get("include_backend_drift_block", True)))
        self.var_report_task_quality.set(bool(rep.get("include_task_specific_quality_block", True)))
        self.var_generate_thesis_tex.set(bool(rep.get("generate_thesis_tex", True)))
        self.var_generate_thesis_figures.set(bool(rep.get("generate_thesis_figures", True)))
        self.var_cleanup_legacy_reports.set(bool(rep.get("cleanup_legacy_reports", True)))
        self.var_include_campaign_readiness.set(bool(rep.get("include_campaign_readiness", True)))

        campaign = dict(data.get("campaign") or {})
        self.var_campaign_id.set(str(campaign.get("id") or campaign.get("campaign_id") or data.get("name") or "thesis_final_campaign_v1"))
        self.var_claim_scope.set(str(campaign.get("claim_scope") or EVALUATED_MATRIX_CLAIM_SCOPE))
        self.var_campaign_mode.set(str(campaign.get("mode") or "development"))
        self.var_campaign_enforcement.set(str(campaign.get("enforcement") or "warn"))
        self.var_campaign_frozen.set(bool(campaign.get("frozen_before_final_campaign", False)))
        manifests = campaign.get("dataset_manifests") if isinstance(campaign.get("dataset_manifests"), Mapping) else {}
        cls_m = manifests.get("classification") if isinstance(manifests.get("classification"), Mapping) else {}
        det_m = manifests.get("detection") if isinstance(manifests.get("detection"), Mapping) else {}
        self.var_manifest_cls_calib.set(str((cls_m or {}).get("calibration") or ""))
        self.var_manifest_cls_val.set(str((cls_m or {}).get("validation") or ""))
        self.var_manifest_det_calib.set(str((det_m or {}).get("calibration") or ""))
        self.var_manifest_det_val.set(str((det_m or {}).get("validation") or ""))
        self.var_pipeline_contract_manifest.set(str(campaign.get("pipeline_contract_manifest") or ""))
        self.var_holdout_registry.set(str(campaign.get("holdout_registry") or "campaign_inputs/holdout_registry.json"))
        self.var_ranking_model_bundle.set(str(campaign.get("ranking_model_bundle") or ""))
        self.var_energy_calibration_manifest.set(str(campaign.get("energy_calibration_manifest") or ""))
        self.var_require_fitted_stage_time.set(bool(campaign.get("require_fitted_stage_time", False)))
        self.var_require_native_handover.set(bool(campaign.get("require_native_handover_model", False)))
        self.var_require_campaign_freeze.set(bool(campaign.get("require_campaign_freeze", False)))
        self.var_require_prediction_approval.set(bool(campaign.get("require_prediction_freeze_approval", False)))
        self.var_require_prediction_signature.set(bool(campaign.get("require_cryptographic_prediction_signature", False)))
        self.var_prediction_public_key.set(str(campaign.get("prediction_freeze_public_key") or ""))

        qgate = dict(data.get("quality_gate") or {})
        self.var_quality_profile_id.set(str(qgate.get("profile_id") or qgate.get("name") or "task_quality_v1"))
        self.var_quality_frozen.set(bool(qgate.get("frozen_before_final_campaign", False)))
        self.var_quality_tier.set(str(qgate.get("dataset_tier") or "screening"))
        self.var_quality_canonical_ref.set(str(qgate.get("canonical_reference") or "canonical_full_onnx"))
        qcls = qgate.get("classification") if isinstance(qgate.get("classification"), Mapping) else {}
        qcls_guard = qcls.get("guardrails") if isinstance(qcls.get("guardrails"), Mapping) else {}
        qdet = qgate.get("detection") if isinstance(qgate.get("detection"), Mapping) else {}
        qdet_guard = qdet.get("guardrails") if isinstance(qdet.get("guardrails"), Mapping) else {}
        self.var_quality_cls_margin_pp.set(float((qcls or {}).get("non_inferiority_margin", 0.01) or 0.0) * 100.0)
        self.var_quality_cls_top5_margin_pp.set(float((qcls_guard or {}).get("top5_accuracy_margin", 0.01) or 0.0) * 100.0)
        self.var_quality_det_margin_ap.set(float((qdet or {}).get("non_inferiority_margin", 0.01) or 0.0) * 100.0)
        self.var_quality_det_ap50_margin.set(float((qdet_guard or {}).get("ap50_margin", 0.01) or 0.0) * 100.0)
        self.var_quality_det_ap75_margin.set(float((qdet_guard or {}).get("ap75_margin", 0.01) or 0.0) * 100.0)
        qstats = qgate.get("statistics") if isinstance(qgate.get("statistics"), Mapping) else {}
        self.var_quality_confidence.set(float((qstats or {}).get("confidence_level", 0.95) or 0.95))
        self.var_quality_bootstrap.set(int((qstats or {}).get("bootstrap_repetitions", 5000) or 5000))
        self.var_quality_seed.set(int((qstats or {}).get("seed", 20260710) or 20260710))
        # v2.64 runs the semantic CPU reference and paired uncertainty on the
        # management node in Smoke, Standard and Final.  An explicitly stored
        # local policy remains loadable as a user override.
        default_quality_location = "central_management"
        quality_location = str((qstats or {}).get("execution_location") or default_quality_location).strip()
        if quality_location not in {"local", "central_management"}:
            quality_location = default_quality_location
        self.var_quality_execution_location.set(quality_location)
        default_quality_workers = 4 if quality_location == "central_management" else 1
        self.var_quality_workers.set(max(1, int((qstats or {}).get("workers") or default_quality_workers)))

        ranking = dict(data.get("ranking_validation") or {})
        self.var_ranking_enabled.set(bool(ranking.get("enabled", True)))
        self.var_ranking_require_frozen.set(bool(ranking.get("require_frozen_predictions", True)))
        self.var_ranking_require_complete.set(bool(ranking.get("require_complete_candidate_universe", True)))
        self.var_ranking_k_values.set(", ".join(str(x) for x in list(ranking.get("k_values") or [1, 3, 5])))
        self.var_ranking_q_values.set(", ".join(str(x) for x in list(ranking.get("elite_q_values") or [1, 3])))
        self.var_ranking_primary_k.set(int(ranking.get("primary_k") or 5))
        self.var_ranking_min_corr.set(int(ranking.get("minimum_candidates_for_correlation") or 3))
        self.var_ranking_epsilon_pct.set(float(ranking.get("near_optimal_relative_epsilon") or 0.01) * 100.0)
        self.var_ranking_bootstrap.set(int(ranking.get("bootstrap_repetitions") or 5000))
        self.var_ranking_seed.set(int(ranking.get("bootstrap_seed") or 20260710))

        official = dict(data.get("official_coco_evaluation") or {})
        self.var_official_coco_enabled.set(bool(official.get("enabled", False)))
        self.var_official_coco_required.set(bool(official.get("required_for_final", False)))
        self.var_official_coco_annotations.set(str(official.get("annotations") or ""))
        self.var_official_coco_remote_annotations.set(str(official.get("remote_annotations") or ""))
        self.var_official_coco_archive_tensors.set(bool(official.get("archive_eval_tensors", True)))
        self.var_official_coco_max_dets.set(", ".join(str(x) for x in list(official.get("max_detections") or [1, 10, 100])))
        profile_registry = str(campaign.get("dataset_registry") or "").strip()
        try:
            app_registry = getattr(self.app, "var_final_dataset_registry", None) if self.app is not None else None
            fallback_registry = str(app_registry.get() or default_registry_path()) if app_registry is not None else str(default_registry_path())
        except Exception:
            fallback_registry = str(default_registry_path())
        self.var_dataset_registry.set(profile_registry or fallback_registry)
        self.var_auto_bind_dataset_registry.set(bool(campaign.get("auto_bind_dataset_registry", True)))

        wf = dict(data.get("workflow") or data.get("evaluation_workflow") or {})
        self.var_workflow_execution_mode.set(str(wf.get("execution_mode") or "generate_and_run"))
        self.var_workflow_skip_runtime.set(bool(wf.get("skip_runtime_benchmarks", wf.get("skip_benchmarks", self.var_workflow_execution_mode.get() != "generate_and_run"))))
        self.var_workflow_no_model_hash.set(bool(wf.get("no_model_hash", True)))
        self.var_workflow_include_reserve.set(bool(wf.get("include_reserve", False)))
        pr = dict(data.get("parallel_remote") or data.get("parallel_execution") or {}) if isinstance(data.get("parallel_remote") or data.get("parallel_execution"), Mapping) else {}
        self.var_parallel_remote_setups.set(bool(wf.get("parallel_remote_setups", pr.get("enabled", pr.get("parallel_remote_setups", True)))))
        self.var_max_parallel_setups.set(int(wf.get("max_parallel_setups", pr.get("max_parallel_setups", 3)) or 3))
        self.var_max_parallel_uploads.set(int(wf.get("max_parallel_uploads", pr.get("max_parallel_uploads", 1)) or 1))
        self.var_powercalc_workers.set(int(wf.get("powercalc_workers", pr.get("powercalc_workers", 1)) or 1))
        bex = dict(data.get("benchmark_execution") or data.get("runtime_benchmark") or {})
        self.var_benchmark_provider.set(str(bex.get("provider") or bex.get("benchmark_provider") or "auto"))
        self.var_benchmark_warmup.set(int(bex.get("warmup") or bex.get("benchmark_warmup") or 1))
        self.var_benchmark_runs.set(int(bex.get("runs") or bex.get("iters") or bex.get("benchmark_runs") or 3))
        self.var_benchmark_timeout.set(int(bex.get("timeout_s") or bex.get("timeout") or 0))

        energy = dict(data.get("energy") or {})
        _native_energy_master = native_energy_requested(data)
        self.var_energy_enabled.set(bool(_native_energy_master))
        final_all_energy = bool(energy.get("final_all_split_energy") or energy.get("all_split_energy") or energy.get("require_complete_split_energy"))
        if not final_all_energy:
            try:
                final_all_energy = (
                    bool(energy.get("enabled", False))
                    and str(energy.get("target_policy") or energy.get("policy") or "").strip().lower() == "all"
                    and int(energy.get("max_targets_per_run_id", energy.get("max_targets", 0)) or 0) == 0
                    and all(x.lower().replace("-", "_") in {"ort_cpu", "cpu_ort", "cpu"} for x in _split_csv(", ".join(str(x) for x in (energy.get("skip_backends") or energy.get("skip_run_ids") or energy.get("exclude_backends") or []) if str(x).strip()) if isinstance((energy.get("skip_backends") or energy.get("skip_run_ids") or energy.get("exclude_backends") or []), (list, tuple)) else str(energy.get("skip_backends") or energy.get("skip_run_ids") or energy.get("exclude_backends") or "")))
                    and bool(energy.get("strict", False))
                )
            except Exception:
                final_all_energy = False
        self.var_energy_final_all_splits.set(bool(final_all_energy))
        self.var_energy_final_skip_cpu_ort.set(bool(energy.get("final_energy_skip_cpu_ort", energy.get("skip_cpu_ort_in_final_energy", True))))
        self.var_energy_scope.set(str(energy.get("scope") or "row_variant"))
        self.var_energy_repeat_override.set(int(energy.get("repeat_override") or energy.get("repeats") or 0))
        self.var_energy_target_policy.set(str(energy.get("target_policy") or energy.get("policy") or "canonical_only"))
        def _csv_value(value: object) -> str:
            if isinstance(value, (list, tuple)):
                return ", ".join(str(x) for x in value if str(x).strip())
            return str(value or "")
        self.var_energy_skip_backends.set(_csv_value(energy.get("skip_backends") or energy.get("skip_run_ids") or energy.get("exclude_backends") or ["ort_cpu", "ort_cuda"]))
        self.var_energy_include_run_ids.set(_csv_value(energy.get("include_run_ids") or energy.get("allow_run_ids") or energy.get("include_backends") or []))
        self.var_energy_exclude_run_ids.set(_csv_value(energy.get("exclude_run_ids") or energy.get("deny_run_ids") or []))
        self.var_energy_heartbeat_s.set(int(energy.get("heartbeat_s") or 60))
        self.var_energy_max_targets_per_run_id.set(int(energy.get("max_targets_per_run_id", energy.get("max_targets", 0)) or 0))
        self.var_energy_max_work_units_per_window.set(int(energy.get("max_work_units_per_window", 0) or 0))
        self.var_energy_max_window_duration_s.set(int(energy.get("max_window_duration_s", energy.get("max_active_s", 0)) or 0))
        self.var_energy_timeout_s_per_window.set(int(energy.get("timeout_s_per_window", energy.get("window_timeout_s", 0)) or 0))
        self.var_energy_sizing_probe_max_work_units.set(int(energy.get("sizing_probe_max_work_units", 256) or 256))
        self.var_energy_include_raw_parquet.set(bool(energy.get("include_raw_parquet_in_debug_pack", False)))
        self.var_energy_strict.set(bool(energy.get("strict", False)))
        phases = energy.get("phases") or ["latency", "streaming"]
        self.var_energy_phases.set(", ".join(str(x) for x in phases) if isinstance(phases, list) else str(phases or "latency, streaming"))
        if bool(self.var_energy_final_all_splits.get()):
            self._apply_final_all_split_energy_checkbox()
        else:
            self._update_energy_estimate()

        native = dict(data.get("native_producers") or {})
        from ..native_execution_contract import resolve_native_execution_contract
        contract = resolve_native_execution_contract(data)
        self._native_budget_widget_baseline = {
            k: int(((preset.get("snapshot") or {}).get("runtime") or {}).get("native", {}).get(k, contract[k]))
            for k in ("frames", "warmup", "repetitions")}
        self._native_energy_budget_passthrough = {
            k: copy.deepcopy(v) for k, v in (native.get("energy") or {}).items()
            if k in {"task_budget", "task_budget_source"}}
        self.var_native_enabled.set(bool(native.get("enabled", False)))
        self._loaded_native_build_missing_engines = native.get("build_missing_engines", True)
        self.var_native_backends.set(_csv_value(native.get("backends") or ["hailo8", "hailo10h", "deepx"]))
        self.var_native_case_policy.set(str(native.get("case_policy") or "all_accepted"))
        self.var_native_precision.set(str(native.get("precision") or "uint8_cast_fp16"))
        self.var_native_frames.set(contract["frames"])
        self.var_native_warmup.set(contract["warmup"])
        self.var_native_repetitions.set(contract["repetitions"])
        self.var_native_queue_depth.set(int(native.get("queue_depth") or 3))
        self.var_native_inflight.set(int(native.get("inflight") or 8))
        fcfg = native.get("full_baselines") if isinstance(native.get("full_baselines"), Mapping) else {}
        _selected_full_backends = native_full_backends_from_run_profiles(data.get("run_profiles") or [])
        self.var_native_full_baselines.set(bool(self.var_native_enabled.get() and (_selected_full_backends or (fcfg or {}).get("enabled", False))))
        vcfg = native.get("validation") if isinstance(native.get("validation"), Mapping) else {}
        self.var_native_validation_enabled.set(bool((vcfg or {}).get("enabled", native.get("dump_outputs", True))))
        ecfg = native.get("energy") if isinstance(native.get("energy"), Mapping) else {}
        self.var_native_energy_enabled.set(bool(self.var_native_enabled.get() and _native_energy_master))
        self.var_native_energy_mode.set("measure" if self.var_native_energy_enabled.get() else "plan")
        self.var_native_energy_duration_s.set(int((ecfg or {}).get("duration_s") or (ecfg or {}).get("duration") or 0))
        probe_cfg = (ecfg or {}).get("window_method_validation_probe") if isinstance((ecfg or {}).get("window_method_validation_probe"), Mapping) else {}
        self.var_window_method_probe_enabled.set(bool(probe_cfg.get("enabled", True)))
        self.var_window_method_probe_repeats.set(max(1, int(probe_cfg.get("repeats") or 3)))
        self.var_window_method_probe_include_raw.set(bool(probe_cfg.get("include_raw_parquet", True)))
        self.var_window_method_probe_strict.set(bool(probe_cfg.get("strict", True)))

        prep = dict(data.get("model_preparation") or {})
        self.var_prep_mode.set(str(prep.get("mode") or "current"))
        hb = dict(data.get("hailo_build") or {})
        self._loaded_hailo_compute_by_family = copy.deepcopy(hb.get("compute_by_family"))
        self._loaded_workflow_stop_after = (data.get("workflow") or {}).get("stop_after")
        self.var_hailo_build_mode.set(str(hb.get("mode") or "reuse_and_build_missing"))
        self.var_hailo_arch.set(str(hb.get("hw_arch") or "hailo8"))
        self.var_hailo_timeout.set(parse_hailo_timeout_seconds(
            hb.get("timeout_s"),
            default=3600,
            label="hailo_build.timeout_s",
        ))
        targets = hb.get("targets")
        if isinstance(targets, (list, tuple)):
            self.var_hailo_targets.set(", ".join(str(x) for x in targets if str(x).strip()))
        else:
            self.var_hailo_targets.set(str(targets or hb.get("hw_arch") or "hailo8"))
        self.var_hailo_build_full.set(bool(hb.get("build_full", True)))
        self.var_hailo_build_part1.set(bool(hb.get("build_part1", True)))
        self.var_hailo_build_part2.set(bool(hb.get("build_part2", True)))
        self.var_hailo_preset.set(str(hb.get("preset") or "quick"))
        self.var_hailo_opt_level.set(int(hb.get("optimization_level") if hb.get("optimization_level") is not None else 0))
        self.var_hailo_calib_dir.set(str(hb.get("calib_dir") or ""))
        self.var_hailo_calib_count.set(int(hb.get("calib_count") if hb.get("calib_count") is not None else 16))
        self.var_hailo_calib_batch.set(int(hb.get("calib_batch_size") if hb.get("calib_batch_size") is not None else 8))
        self.var_hailo_force_build.set(parse_config_bool(hb.get("force_build", False), field="hailo_build.force_build"))
        self.var_hailo_keep_artifacts.set(bool(hb.get("keep_artifacts", True)))
        hw = dict(data.get("hardware_smoke") or {})
        self.var_hw_smoke_mode.set(str(hw.get("mode") or "summary_only"))
        db = dict(data.get("deepx_build") or {})
        self.var_deepx_classification_preprocessing.set(
            str(
                db.get("classification_preprocessing")
                if db.get("classification_preprocessing") is not None
                else "current_scale_only"
            )
        )
        self.var_deepx_compiler_overlay.set(str(db.get("compiler_overlay") or ""))
        if db:
            self.var_deepx_root.set(str(db.get("dx_all_suite_root") or self.var_deepx_root.get()))
            self.var_deepx_compiler_venv.set(str(db.get("compiler_venv") or self.var_deepx_compiler_venv.get()))
            self.var_deepx_cache_dir.set(str(db.get("cache_dir") or self.var_deepx_cache_dir.get()))


        # Per-accelerator hardware matrix selection from central Tool Config registry.
        hw_section = dict(data.get("hardware") or {})
        selected_setups = _normalize_hardware_setup_ids(hw_section.get("selected_setups"))
        self.var_hardware_selected_setups.set(", ".join(selected_setups))
        self.var_hardware_selected_groups.set(", ".join(_split_csv(hw_section.get("selected_groups"))))
        self.var_hardware_setups_file.set(str(hw_section.get("setups_file") or ""))
        if "orin_nx_hailo8_01" in selected_setups:
            self.var_hw_h8_enabled.set(True)
        if "orin_nx_hailo10_01" in selected_setups:
            self.var_hw_h10_enabled.set(True)
        if "orin_nx_deepx_m1_01" in selected_setups:
            self.var_hw_dx_enabled.set(True)

        # Backwards-compat: older profiles embedded complete hardware_targets.
        for _ht in list(data.get("hardware_targets") or data.get("hardware_setups") or []):
            if not isinstance(_ht, Mapping):
                continue
            accel = str(_ht.get("accelerator") or _ht.get("backend") or "").strip().lower()
            remote_exec = _ht.get("remote_execution") if isinstance(_ht.get("remote_execution"), Mapping) else {}
            def _apply_card(enabled_var: tk.BooleanVar, host_var: tk.StringVar, base_var: tk.StringVar, venv_var: tk.StringVar, provider_var: tk.StringVar) -> None:
                enabled_var.set(bool(remote_exec.get("enabled", True)))
                host_var.set(str(remote_exec.get("host_id") or remote_exec.get("id") or ""))
                base_var.set(str(remote_exec.get("remote_base_dir") or remote_exec.get("base_dir") or base_var.get()))
                venv_var.set(str(remote_exec.get("remote_venv") or remote_exec.get("venv") or venv_var.get()))
                provider_var.set(str(remote_exec.get("provider") or provider_var.get()))
            if "hailo8" in accel:
                _apply_card(self.var_hw_h8_enabled, self.var_hw_h8_host_id, self.var_hw_h8_base, self.var_hw_h8_venv, self.var_hw_h8_provider)
            elif "hailo10" in accel:
                _apply_card(self.var_hw_h10_enabled, self.var_hw_h10_host_id, self.var_hw_h10_base, self.var_hw_h10_venv, self.var_hw_h10_provider)
            elif "deepx" in accel or "dx" in accel:
                _apply_card(self.var_hw_dx_enabled, self.var_hw_dx_host_id, self.var_hw_dx_base, self.var_hw_dx_venv, self.var_hw_dx_provider)

        remote = dict(data.get("remote_execution") or data.get("remote") or {})
        # v55e: legacy remote_execution may exist in older profiles, but new
        # profiles should not duplicate remote settings. Hardware setup cards
        # on tab 3 are the source of truth for remote execution.
        self.var_remote_enabled.set(False)
        host_id = str(remote.get("host_id") or remote.get("id") or "").strip()
        inline_hosts = remote.get("hosts") or remote.get("remote_hosts") or []
        selected_host = None
        if isinstance(inline_hosts, list):
            for item in inline_hosts:
                if not isinstance(item, Mapping):
                    continue
                hid = str(item.get("id") or item.get("label") or "").strip()
                if not host_id or hid == host_id:
                    selected_host = dict(item)
                    host_id = hid or host_id
                    break
        self.var_remote_host_id.set(host_id)
        if selected_host is None and host_id:
            selected_host = self._remote_host_config_by_id(host_id)
        if selected_host:
            self._apply_remote_host_config_to_fields(selected_host)
        else:
            self.var_remote_host.set(str(remote.get("host") or ""))
            self.var_remote_user.set(str(remote.get("user") or ""))
            self.var_remote_port.set(int(remote.get("port") or 22))
            self.var_remote_base.set(str(remote.get("remote_base_dir") or "~/splitpoint_runs"))
            self.var_remote_ssh_extra_args.set(str(remote.get("ssh_extra_args") or ""))
        self.var_remote_provider.set(str(remote.get("provider") or "auto"))
        self.var_remote_warmup.set(int(remote.get("warmup") or 10))
        self.var_remote_iters.set(int(remote.get("iters") or 50))
        self.var_remote_timeout.set(int(remote.get("timeout_s") or 0))
        self.var_remote_venv.set(str(remote.get("remote_venv") or remote.get("venv") or ""))
        self.var_remote_transfer_mode.set(str(remote.get("transfer_mode") or "bundle"))
        self.var_remote_reuse_bundle.set(bool(remote.get("reuse_bundle", True)))
        self.var_remote_resume.set(bool(remote.get("resume", True)))
        if self.var_remote_enabled.get() and not str(self.var_remote_venv.get() or "").strip():
            # Mirror the Benchmark tab's remote venv if the YAML was saved before this field existed.
            self._apply_current_benchmark_tab_remote_defaults(enable_remote=True)
        self._loading_profile = False
        self._update_run_summary()

    # ------------------------------------------------------------------
    # Validation/save/preview
    # ------------------------------------------------------------------

    def _suggest_save_path(self) -> Path:
        current = str(self.var_path.get() or "").strip()
        if current:
            p = Path(current).expanduser()
            try:
                builtin_root = evaluation_profile_default_root().resolve()
                p.resolve().relative_to(builtin_root)
                # Built-in profiles should not be overwritten from the GUI.
            except Exception:
                if p.suffix.lower() in {".yaml", ".yml"}:
                    return p
        name = _norm_id(self.var_name.get()) or "custom_evaluation_profile_v1"
        try:
            base = Path.cwd() / "profiles"
        except Exception:
            base = Path.home() / "profiles"
        return base / f"{name}.yaml"

    def _validate_current(self) -> None:
        try:
            payload = self._build_payload()
            messagebox.showinfo("Evaluation Profile", f"Profil ist gültig.\n\nModels: {len(payload['model_suite']['primary'])}\nRun profiles: {len(payload['run_profiles'])}", parent=self)
            self.status_var.set("Profil ist gültig.")
        except Exception as exc:
            messagebox.showerror("Evaluation Profile", f"Profil ist nicht gültig:\n\n{type(exc).__name__}: {exc}", parent=self)
            self.status_var.set("Profil ist nicht gültig.")

    def _preview_yaml(self) -> None:
        try:
            payload = self._build_payload()
            text = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)
        except Exception as exc:
            messagebox.showerror("Evaluation Profile", f"YAML konnte nicht erzeugt werden:\n\n{type(exc).__name__}: {exc}", parent=self)
            return
        win = tk.Toplevel(self)
        win.title("Evaluation Profile YAML Vorschau")
        win.geometry("900x680")
        win.columnconfigure(0, weight=1)
        win.rowconfigure(0, weight=1)
        txt = scrolledtext.ScrolledText(win, wrap="none")
        txt.grid(row=0, column=0, sticky="nsew")
        txt.insert("1.0", text)
        txt.configure(state="disabled")
        ttk.Button(win, text="Schließen", command=win.destroy).grid(row=1, column=0, sticky="e", padx=8, pady=8)

    def _save(self, *, use_after: bool) -> None:
        try:
            payload = self._build_payload()
        except Exception as exc:
            messagebox.showerror("Evaluation Profile", f"Profil kann nicht gespeichert werden:\n\n{type(exc).__name__}: {exc}", parent=self)
            return
        suggested = self._suggest_save_path()
        path = filedialog.asksaveasfilename(
            parent=self,
            title="Evaluation Profile speichern",
            initialdir=str(suggested.parent),
            initialfile=suggested.name,
            defaultextension=".yaml",
            filetypes=[("YAML", "*.yaml *.yml"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            saved = save_evaluation_profile_yaml(path, payload, validate=True)
        except Exception as exc:
            messagebox.showerror("Evaluation Profile", f"Speichern fehlgeschlagen:\n\n{type(exc).__name__}: {exc}", parent=self)
            return
        self.var_path.set(str(saved))
        self.status_var.set(f"Gespeichert: {saved}")
        if use_after:
            self._use_saved_path(saved)
            self.status_var.set(f"Gespeichert und verwendet: {saved}")
            try:
                self.after(150, self.destroy)
            except Exception:
                pass
        else:
            self.status_var.set(f"Gespeichert: {saved}")

    def _use_saved_path(self, path: Path) -> None:
        text = str(path)
        try:
            if self.profile_var is not None:
                self.profile_var.set(text)
        except Exception:
            pass
        try:
            if self.combo is not None:
                values = list(self.combo.cget("values") or [])
                if text not in values:
                    values.insert(0, text)
                    self.combo.configure(values=values)
        except Exception:
            pass
        if self.on_saved is not None:
            try:
                self.on_saved(path)
            except Exception:
                pass
        try:
            if self.app is not None and hasattr(self.app, "_persist_settings"):
                self.app._persist_settings()
        except Exception:
            pass


def open_profile_editor(
    parent: tk.Misc,
    *,
    app: Any | None = None,
    profile_var: tk.StringVar | None = None,
    combo: ttk.Combobox | None = None,
    on_saved: Callable[[Path], None] | None = None,
) -> EvaluationProfileEditor:
    """Open the profile editor dialog and return it."""
    win = EvaluationProfileEditor(parent, app=app, profile_var=profile_var, combo=combo, on_saved=on_saved)
    try:
        win.lift()
        win.focus_force()
    except Exception:
        pass
    return win


def open_evaluation_profile_editor(
    parent: tk.Misc,
    *,
    app: Any | None = None,
    profile_var: tk.StringVar | None = None,
    profile_combo: ttk.Combobox | None = None,
    models_root_var: tk.StringVar | None = None,
) -> EvaluationProfileEditor:
    """Compatibility wrapper used by the Evaluation Workflow tab.

    `models_root_var` is optional; when a saved profile contains a
    `models_root_hint` and the main-tab field is still empty, the hint is copied
    over so the user does not have to repeat the same path manually.
    """

    def _on_saved(path: Path) -> None:
        if models_root_var is None:
            return
        try:
            current = str(models_root_var.get() or "").strip()
        except Exception:
            current = ""
        if current:
            return
        try:
            loaded = load_evaluation_profile(str(path), validate=False)
            hint = str((loaded.raw_profile or {}).get("models_root_hint") or "").strip() if loaded is not None else ""
        except Exception:
            hint = ""
        if hint:
            try:
                models_root_var.set(hint)
            except Exception:
                pass

    return open_profile_editor(parent, app=app, profile_var=profile_var, combo=profile_combo, on_saved=_on_saved)

# v60m: master-energy UI binding and concise integrity-mode hint.
from .v60m_ui_patch import install_profile_editor_class_patches as _v60m_install_profile_editor_patches
_v60m_install_profile_editor_patches(globals())

# v60m: development profiles keep screening validation unless final validation
# was explicitly selected; calibration manifests may still auto-bind.
from onnx_splitpoint_tool.v60m_policy import install_dataset_binding_guards as _v60m_install_dataset_binding_guards
_v60m_install_dataset_binding_guards(globals())

# v60m: the top-level energy switch is authoritative for all native energy paths.
from onnx_splitpoint_tool.v60m_policy import install_energy_object_guards as _v60m_install_energy_guards
_v60m_install_energy_guards(globals())


# v60z profile-editor normalisation wrappers
# Keep the simplified GUI authoritative while preserving legacy YAML compatibility.
def _v60z_profile_path_from_editor(instance, args, kwargs):
    from pathlib import Path
    candidates = []
    for value in list(args) + list(kwargs.values()):
        if isinstance(value, (str, Path)):
            candidates.append(value)
    for name in ("profile_path", "current_profile_path", "_profile_path", "path", "filename"):
        value = getattr(instance, name, None)
        if hasattr(value, "get"):
            try: value = value.get()
            except Exception: value = None
        if value: candidates.append(value)
    for value in candidates:
        try:
            p = Path(value).expanduser()
            if p.suffix.lower() in {".yaml", ".yml"} and p.exists():
                return p
        except Exception:
            pass
    return None


def _v60z_editor_bool(instance, include_tokens):
    for name, value in vars(instance).items():
        token = name.lower()
        if not all(t in token for t in include_tokens):
            continue
        if hasattr(value, "get"):
            try: return bool(value.get())
            except Exception: pass
    return None


def _v60z_normalise_saved_editor_profile(instance, args, kwargs):
    from onnx_splitpoint_tool.native_full_quality import (
        load_and_normalise_yaml, save_normalised_yaml, apply_native_energy_state,
        normalise_evaluation_profile,
    )
    p = _v60z_profile_path_from_editor(instance, args, kwargs)
    if p is None:
        return
    try:
        data = load_and_normalise_yaml(p)
        native_energy = _v60z_editor_bool(instance, ("energy",))
        # Prefer an explicitly named native-energy variable if present.
        explicit = _v60z_editor_bool(instance, ("native", "energy"))
        if explicit is not None:
            native_energy = explicit
        if native_energy is not None:
            apply_native_energy_state(data, native_energy)
        normalise_evaluation_profile(data)
        save_normalised_yaml(p, data)
    except Exception:
        # The editor's original error handling remains authoritative.  Runtime
        # profile resolution normalises the profile again before execution.
        return


def _v60z_install_profile_editor_wrappers():
    import inspect
    from onnx_splitpoint_tool.native_full_quality import normalise_evaluation_profile, resolve_native_energy_enabled
    for obj in list(globals().values()):
        if not inspect.isclass(obj) or obj.__module__ != __name__:
            continue
        for name, original in list(vars(obj).items()):
            if not callable(original) or getattr(original, "_v60z_wrapped", False):
                continue
            lname = name.lower()
            if any(t in lname for t in ("build_profile", "collect_profile", "profile_dict", "to_profile")):
                def make_builder(fn):
                    def wrapper(self, *args, **kwargs):
                        result = fn(self, *args, **kwargs)
                        if isinstance(result, dict):
                            normalise_evaluation_profile(result)
                        return result
                    wrapper._v60z_wrapped = True
                    wrapper.__name__ = getattr(fn, "__name__", "wrapped")
                    return wrapper
                setattr(obj, name, make_builder(original))
            elif "save" in lname and "profile" in lname:
                def make_save(fn):
                    def wrapper(self, *args, **kwargs):
                        result = fn(self, *args, **kwargs)
                        _v60z_normalise_saved_editor_profile(self, args, kwargs)
                        return result
                    wrapper._v60z_wrapped = True
                    wrapper.__name__ = getattr(fn, "__name__", "wrapped")
                    return wrapper
                setattr(obj, name, make_save(original))
            elif "load" in lname and "profile" in lname:
                def make_load(fn):
                    def wrapper(self, *args, **kwargs):
                        result = fn(self, *args, **kwargs)
                        # Reflect the resolved Native Energy value into all
                        # clearly named native-energy Tk variables.
                        try:
                            p = _v60z_profile_path_from_editor(self, args, kwargs)
                            if p:
                                from onnx_splitpoint_tool.native_full_quality import load_and_normalise_yaml
                                state = resolve_native_energy_enabled(load_and_normalise_yaml(p))
                                for attr, var in vars(self).items():
                                    low = attr.lower()
                                    if "native" in low and "energy" in low and hasattr(var, "set"):
                                        var.set(state)
                        except Exception:
                            pass
                        return result
                    wrapper._v60z_wrapped = True
                    wrapper.__name__ = getattr(fn, "__name__", "wrapped")
                    return wrapper
                setattr(obj, name, make_load(original))

_v60z_install_profile_editor_wrappers()
