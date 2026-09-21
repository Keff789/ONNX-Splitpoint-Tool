from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .artifacts import now_iso, relpath
from .evidence_status import project_native_evidence_status
from .result_context import bind_benchmark_source_context
from .logical_measurement import (
    annotate_logical_measurements,
    canonical_runtime_precision_identity,
)
from ..ranking_methods import canonical_backend, canonical_direction, direction_parts
from ..energy.comparison import resolve_energy_comparison

try:
    from onnx_splitpoint_tool.validation.accuracy_gates import apply_accuracy_gate_to_row
except Exception:  # keep report generation robust if optional gate module is unavailable
    def apply_accuracy_gate_to_row(row, policy=None):
        return row
from onnx_splitpoint_tool.validation.accuracy_gates import apply_gate_fields

REQUIRED_NORMALIZED_FIELDS = [
    "model_id",
    "case_id",
    "backend",
    "variant",
    "part1_latency_ms",
    "part2_latency_ms",
    "transfer_latency_ms",
    "total_latency_ms",
    "compile_ok",
    "runtime_ok",
    "validation_ok",
    "error_class",
    "endpoint_mode",
    "measurement_endpoint",
    "quality_endpoint",
    # v49i validation evidence fields. They are optional for producers, but
    # required in the normalized contract so reports can reason about validity
    # without parsing backend-specific logs.
    "max_abs_error",
    "mean_abs_error",
    "cosine_similarity",
    "output_shape_match",
    "full_latency_ms",
    "component_measurement_status",
    "measured_variants",
    "skipped_variants",
    "variant_status",
    "variant_errors",
    "stage1_provider",
    "stage2_provider",
    "full_provider",
    # v51: distinguish real full baselines from full-reference timings
    # embedded in mixed split rows. This avoids reporting hailo8_to_tensorrt
    # as a best full backend and lets Hailo full/raw-head E2E measurements be
    # treated as real full rows.
    "full_backend",
    "full_latency_source",
    "full_e2e_latency_ms",
    "full_host_tail_latency_ms",
    "full_raw_accelerator_latency_ms",
    "hailo_runtime_kind",
    "hailo_full_runtime_ok",
    "hailo_composed_runtime_ok",
    "deepx_runtime_kind",
    "deepx_full_runtime_ok",
    "deepx_composed_runtime_ok",
    # v55d reporting polish: keep final-pass and Stage2-contract evidence
    # in the normalized contract so dashboards do not have to re-open backend
    # validation_report.json files.
    "final_pass",
    "final_pass_all",
    "semantic_validation_ok",
    "semantic_validation_passed",
    "stage2_accel_calibration_status",
    "stage2_calibration_source",
    "stage2_calibration_trust_level",
    "stage2_calibration_producer_exact",
    "deepx_stage2_contract_status",
    "deepx_stage2_contract_pass",
    "deepx_stage2_contract_probe_samples",
    "deepx_stage2_contract_selected_pass_ratio",
    "deepx_stage2_contract_selected_mean_score",
    "deepx_stage2_contract_selected_top1_match_ratio",
    "deepx_stage2_contract_rejected_candidate_count",
    # v55p: raw-head / host-tail and feature-tensor contract evidence.
    "raw_head_contract_present",
    "raw_head_contract_status",
    "host_tail_required",
    "host_tail_available",
    "stage2_input_contract_kind",
    "stage2_feature_tensor_contract_status",
    "deepx_stage1_semantic_override",
    "semantic_validation_metric_gate",
    # v55r: latency-vs-throughput pipeline metrics are first-class report fields.
    "split_latency_e2e_ms",
    "pipeline_model",
    "pipeline_stage1_lane_ms",
    "pipeline_stage2_lane_ms",
    "pipeline_transfer_est_ms",
    "pipeline_cycle_no_transfer_ms",
    "pipeline_cycle_with_transfer_ms",
    "pipeline_cycle_selected_ms",
    "pipeline_fps_selected",
    "pipeline_speedup_full_over_cycle",
    "pipeline_latency_penalty_vs_full",
    "pipeline_cycle_source",
    "pipeline_note",
    # v57m: keep the small set of decision metrics separated by
    # throughput semantics. Full-backend throughput is not the same as a
    # heterogeneous two-stage pipeline; same-backend composed rows remain
    # diagnostics.
    "throughput_kind",
    "pipeline_applicable",
    "full_backend_throughput_fps",
    "heterogeneous_pipeline_fps",
    "same_backend_composed_fps",
    "backend_tool_fps",
    "throughput_primary_fps",
    # v58d: detection score/confidence semantics. Raw logits or unnormalised
    # scores must not be reported as calibrated 0..1 confidence.
    "score_semantics",
    "raw_score_mean",
    "confidence_mean_clipped_0_1",
    "throughput_primary_metric",
    "throughput_primary_source",
    # v60r: exact Run-Mode validation cardinality is part of the row contract.
    "validation_cardinality_contract",
]


# v59j thesis/reporting contract additions. These fields separate semantic E2E
# validity from interface-contract and strict boundary-numeric evidence and keep
# row-level Energy provenance explicit.
for _field in [
    "semantic_e2e_pass",
    "interface_contract_pass",
    "strict_boundary_numeric_pass",
    "visual_validation_ok",
    "validation_claim_level",
    "validation_claim_label",
    "interface_contract_status",
    "strict_boundary_numeric_status",
    "interface_check_status",
    "interface_check_pass",
    "interface_stage1_status",
    "interface_stage1_pass",
    "interface_stage1_max_abs",
    "interface_stage1_mean_abs_mean",
    "interface_stage1_num_compared_tensors",
    "raw_head_drift_status",
    "raw_head_drift_pass",
    "raw_head_drift_max_abs",
    "raw_head_drift_mean_abs_mean",
    "validation_level_summary",
    "energy_row_level_source",
    "energy_coverage_status",
    "energy_target_status",
    "energy_target_phase_count",
    "energy_target_window_count",
    "energy_target_valid_window_count",
    "energy_target_valid_window_ratio",
    "energy_target_id",
    "energy_target_ok",
    "energy_target_error_summary",
    "energy_merge_source",
    # v59ej strict task/ranking gate fields.
    "buildable",
    "runtime_executable",
    "contract_consistent",
    "task_valid",
    "accuracy_gate_pass",
    "eligible_for_ranking",
    "ranking_eligible",
    "gate_status",
    "validation_verdict",
    "accuracy_gate_reason",
    "accuracy_reference_source",
    "accuracy_gate_policy",
    # v60i: preserve the runtime task-quality contract and policy provenance
    # end-to-end instead of flattening it away during result ingestion.
    "task",
    "benchmark_task_requested",
    "benchmark_task_used",
    "validation_dataset_task",
    "task_quality_policy",
    "task_quality_policy_sha256",
    "task_quality_gate",
    "task_quality_gates_by_variant",
    "quality_request_identities_by_variant",
    "row_eligibility",
    "accuracy_gate_policy_sha256",
    "runtime_quality_gate_policy_sha256",
    "accuracy_gate_policy_match",
]:
    if _field not in REQUIRED_NORMALIZED_FIELDS:
        REQUIRED_NORMALIZED_FIELDS.append(_field)

ERROR_CLASS_MAP = {
    "parse": "parse_failed",
    "compile": "compile_failed",
    "build": "compile_failed",
    "runtime": "runtime_failed",
    "run": "runtime_failed",
    "validation": "validation_failed",
    "validate": "validation_failed",
    "timeout": "timeout",
    "unsupported": "unsupported_op",
    "resource": "resource_infeasible",
    "memory": "resource_infeasible",
}




def _as_list(value: Any) -> list[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    return []


def _task_quality_policy_v60i(row: Mapping[str, Any], gate: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return the exact runtime policy embedded in a raw result row.

    The runtime gate is authoritative for what was executed.  The scientific
    reporter later compares this embedded policy with the current profile and
    marks any mismatch rather than silently recomputing a different decision.
    """
    direct = row.get("task_quality_policy")
    if isinstance(direct, Mapping) and direct:
        return dict(direct)
    if isinstance(gate, Mapping):
        nested = gate.get("policy")
        if isinstance(nested, Mapping) and nested:
            return dict(nested)
    return {}


def _select_task_quality_gate_v60i(row: Mapping[str, Any], variant: str | None = None) -> dict[str, Any]:
    """Select the task-quality gate that belongs to one normalized row.

    A validation report can contain gates for ``full``, ``part1``, ``part2``
    and ``composed``.  Earlier reporting copied none of them and therefore
    turned real pass/fail/inconclusive decisions into ``quality_unavailable``.
    """
    requested = str(variant or row.get("primary_variant") or row.get("variant") or "").strip().lower()
    aliases = {
        "split": "composed",
        "complete": "composed",
        "pipeline": "composed",
        "full_backend": "full",
    }
    requested = aliases.get(requested, requested)
    gates = row.get("task_quality_gates_by_variant")
    if isinstance(gates, Mapping):
        order: list[str] = []
        if requested:
            order.append(requested)
        primary = str(row.get("primary_variant") or "").strip().lower()
        if primary:
            order.append(aliases.get(primary, primary))
        if requested == "composed":
            order.extend(["split", "primary"])
        elif requested == "full":
            order.extend(["full_model", "reference"])
        for key in order:
            value = gates.get(key)
            if isinstance(value, Mapping) and value:
                return dict(value)
    direct = row.get("task_quality_gate")
    if isinstance(direct, Mapping) and direct:
        direct_variant = aliases.get(str(direct.get("variant") or "").strip().lower(), str(direct.get("variant") or "").strip().lower())
        if not requested or not direct_variant or direct_variant == requested:
            return dict(direct)
    return {}

def _read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _read_csv(path: Path) -> list[dict[str, Any]]:
    try:
        # BenchmarkSet CSVs may contain large embedded JSON/log fields.
        # Python defaults to 128 KiB per field and otherwise raises
        # ``field larger than field limit``; older workflow versions silently
        # treated those files as empty.
        try:
            csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
        except Exception:
            try:
                csv.field_size_limit(2**31 - 1)
            except Exception:
                pass
        with path.open("r", newline="", encoding="utf-8") as f:
            return [dict(r) for r in csv.DictReader(f)]
    except Exception:
        return []


def _float_or_none(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        x = float(value)
        if math.isfinite(x):
            return x
    except Exception:
        return None
    return None


def _bool_or_none(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value is None or value == "":
        return None
    s = str(value).strip().lower()
    if s in {"1", "true", "yes", "y", "ok", "pass", "passed", "success"}:
        return True
    if s in {"0", "false", "no", "n", "fail", "failed", "error"}:
        return False
    return None


def _first(row: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for k in keys:
        if k in row and row.get(k) not in (None, ""):
            return row.get(k)
    return None


def _case_id(row: Mapping[str, Any]) -> str:
    value = _first(row, ["case_id", "case_dir", "folder", "boundary_id", "boundary", "split_index"])
    if value is None:
        return ""
    s = str(value).strip()
    if s.startswith("b"):
        return s
    try:
        return f"b{int(float(s)):03d}"
    except Exception:
        return s


def _canon_backend_token(value: Any) -> str:
    s = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if not s:
        return ""
    if "deepx" in s or "dx_m1" in s or "dxm1" in s:
        return "deepx_m1"
    if "hailo10" in s:
        return "hailo10"
    if "hailo8" in s or s == "hailo":
        return "hailo8"
    if s in {"cpu", "ort_cpu", "cpu_ort"} or "cpuexecutionprovider" in s:
        return "cpu_ort"
    if s in {"cuda", "gpu", "ort_cuda", "cuda_ort"} or "cudaexecutionprovider" in s:
        return "cuda_ort"
    if s in {"trt", "tensor_rt", "tensorrt"} or "tensorrtexecutionprovider" in s:
        return "tensorrt"
    return s


def _is_mixed_backend_token(value: Any) -> bool:
    s = _canon_backend_token(value)
    return "_to_" in s


def _first_token(*values: Any) -> str:
    for value in values:
        s = _canon_backend_token(value)
        if s:
            return s
    return ""


def _backend(row: Mapping[str, Any], tag: str = "") -> str:
    stage1 = str(row.get("stage1_provider") or row.get("stage1") or "").strip()
    stage2 = str(row.get("stage2_provider") or row.get("stage2") or "").strip()
    provider = str(row.get("provider") or row.get("backend") or "").strip()
    run_id = str(row.get("run_id") or tag or "").strip()
    stage1_c = _canon_backend_token(stage1)
    stage2_c = _canon_backend_token(stage2)
    provider_c = _canon_backend_token(provider)
    run_id_c = _canon_backend_token(run_id)
    if stage1_c and stage2_c and stage1_c != stage2_c:
        return f"{stage1_c}_to_{stage2_c}"
    if provider_c:
        return provider_c
    if stage1_c:
        return stage1_c
    if run_id_c:
        return run_id_c
    return "unknown"


def _variant(row: Mapping[str, Any]) -> str:
    primary = str(row.get("primary_variant") or row.get("variant") or "").strip().lower()
    if primary == "full":
        return "full"
    if primary in {"split", "composed", "part1", "part2"}:
        return "split"

    stage1 = _canon_backend_token(row.get("stage1_provider") or row.get("stage1"))
    stage2 = _canon_backend_token(row.get("stage2_provider") or row.get("stage2"))
    has_split_context = bool(stage1 or stage2)
    has_split_timing = any(
        _float_or_none(_first(row, keys)) is not None
        for keys in (
            ["composed_mean_ms", "sum_parts_ms", "throughput_latency_mean_ms", "throughput_cycle_est_ms"],
            ["part1_latency_ms", "part1_mean_ms", "throughput_stage1_mean_ms"],
            ["part2_latency_ms", "part2_mean_ms", "throughput_stage2_mean_ms"],
        )
    )
    # v51: mixed rows may contain full-reference timings.  They are still split
    # rows unless the runner explicitly selected primary_variant=full.
    if has_split_context or has_split_timing:
        return "split"
    if _float_or_none(_first(row, ["full_mean_ms", "full_e2e_mean_ms", "full_latency_ms"])) is not None:
        return "full"
    return "split"


def _error_class(row: Mapping[str, Any]) -> str:
    explicit = str(row.get("error_class") or "").strip()
    if explicit:
        return explicit
    text = " ".join(str(row.get(k) or "") for k in ("error", "status", "message", "reason", "fail_reason"))
    low = text.lower()
    for needle, klass in ERROR_CLASS_MAP.items():
        if needle in low:
            return klass
    ok = _bool_or_none(_first(row, ["ok", "runtime_ok", "eps_pass"]))
    return "" if ok is not False else "runtime_failed"


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _logical_run_id_from_source_v269f(
    row: Mapping[str, Any], source_path: Path, tag: str,
) -> tuple[str, str]:
    """Keep the logical plan run ID when a partial report names only a backend.

    A completed suite normally writes the logical ID (for example
    ``ort_tensorrt``) into its result row.  A validation report rescued after a
    timeout can instead contain only the implementation backend
    (``tensorrt``).  The enclosing ``results_<logical-run>`` directory or the
    management-side ``benchmark_results_<logical-run>_auto`` filename is the
    authoritative dispatch identity in that case.  Preserve the raw value for
    diagnostics, but never infer a logical ID from an arbitrary substring.
    """

    raw = str(
        _first(row, ["run_id", "benchmark_run_id", "run_profile_id"]) or ""
    ).strip()
    path_candidates: list[str] = []
    for part in reversed(source_path.parts):
        lowered = str(part).strip().lower()
        if lowered.startswith("results_") and len(lowered) > len("results_"):
            path_candidates.append(str(part)[len("results_"):].strip())
            break
    name = source_path.name
    match = re.fullmatch(
        r"benchmark_results_(.+?)(?:_auto)?\.(?:json|csv)",
        name,
        flags=re.IGNORECASE,
    )
    if match:
        path_candidates.append(str(match.group(1)).strip())

    logical = next((candidate for candidate in path_candidates if candidate), "")
    if not logical:
        tag_l = str(tag or "").strip()
        if tag_l and tag_l.lower() not in {
            "benchmark_results", "validation_report", "results", "report",
        }:
            logical = re.sub(r"_auto$", "", tag_l, flags=re.IGNORECASE)

    if not logical:
        return raw or str(tag or "").strip(), raw

    # A case-local runner is allowed to call itself by the concrete backend;
    # the dispatch path still identifies the logical experiment profile.
    raw_token = raw.lower().replace("-", "_")
    logical_token = logical.lower().replace("-", "_")
    backend_aliases = {
        "tensorrt", "trt", "onnxruntime", "ort", "hailo", "hailo8",
        "hailo10", "hailo10h", "deepx", "deepx_m1", "cpu", "cuda",
    }
    if not raw or raw_token == logical_token or raw_token in backend_aliases:
        return logical, raw
    return raw, raw


_FULL_FROZEN_SHA_FIELDS_V269D = {
    "preprocessing_contract_sha256",
    "endpoint_contract_hash",
    "source_model_sha256",
    "runtime_artifact_sha256",
}


def _identity_sha_v269d(value: Any) -> tuple[str, bool]:
    text = str(value or "").strip().lower()
    if text.startswith("sha256:"):
        text = text[7:]
    if not text:
        return "", True
    return text, bool(
        len(text) == 64 and all(char in "0123456789abcdef" for char in text)
    )


def _quality_requests_for_variant_v269d(
    row: Mapping[str, Any], variant: str,
) -> list[Mapping[str, Any]]:
    """Return every embedded request that claims to describe ``variant``."""
    key = str(variant or "").strip().lower()
    requests: list[Mapping[str, Any]] = []

    direct = row.get("task_quality_input_requests_by_variant")
    if isinstance(direct, Mapping) and isinstance(direct.get(key), Mapping):
        requests.append(_mapping(direct.get(key)))

    gates = row.get("task_quality_gates_by_variant")
    if isinstance(gates, Mapping) and isinstance(gates.get(key), Mapping):
        request = _mapping(gates.get(key)).get("quality_input_request")
        if isinstance(request, Mapping):
            requests.append(request)

    selected_variant = str(
        row.get("primary_variant") or row.get("variant") or ""
    ).strip().lower()
    if selected_variant == key:
        task_gate = row.get("task_quality_gate")
        if isinstance(task_gate, Mapping) and isinstance(task_gate.get("quality_input_request"), Mapping):
            requests.append(_mapping(task_gate.get("quality_input_request")))
        quality_input = row.get("quality_input_request")
        if isinstance(quality_input, Mapping):
            requests.append(quality_input)

    # Do not let mirrored copies multiply evidence.  Canonical JSON gives a
    # stable exact identity while retaining genuinely conflicting requests.
    unique: dict[str, Mapping[str, Any]] = {}
    for request in requests:
        try:
            key_json = json.dumps(dict(request), sort_keys=True, separators=(",", ":"), default=str)
        except Exception:
            key_json = repr(dict(request))
        unique.setdefault(key_json, request)
    return list(unique.values())


def _frozen_identity_evidence_v269d(
    row: Mapping[str, Any], variant: str,
) -> dict[str, Any]:
    """Collect variant-scoped identities without choosing a conflicting value."""
    variant_key = str(variant or "").strip().lower()
    selected_variant = str(
        row.get("primary_variant") or row.get("variant") or ""
    ).strip().lower()
    values: dict[str, set[str]] = {
        "preprocessing_contract_sha256": set(),
        "endpoint_contract_hash": set(),
        "source_model_sha256": set(),
        "runtime_artifact_sha256": set(),
        "runtime_precision_identity": set(),
    }
    invalid: dict[str, set[str]] = {key: set() for key in values}
    runtime_precision_raw: set[str] = set()

    def add(field: str, value: Any) -> None:
        if value in (None, ""):
            return
        if field in _FULL_FROZEN_SHA_FIELDS_V269D:
            normalized, valid = _identity_sha_v269d(value)
            if normalized:
                values[field].add(normalized)
                if not valid:
                    invalid[field].add(normalized)
            return
        if field == "runtime_precision_identity":
            try:
                runtime_precision_raw.add(json.dumps(
                    value, sort_keys=True, separators=(",", ":"),
                    ensure_ascii=False, default=str,
                ) if isinstance(value, Mapping) else str(value).strip())
            except Exception:
                runtime_precision_raw.add(str(value).strip())
            structured_deepx = isinstance(value, Mapping) or str(
                value or ""
            ).strip().lower().startswith((
                "deepx_dxnn_sha256:", "{",
            ))
            if not structured_deepx:
                normalized = str(value or "").strip().lower().replace(
                    " ", ""
                )
                if normalized:
                    values[field].add(normalized)
                return
            canonical, error = canonical_runtime_precision_identity(value)
            if canonical:
                values[field].add(canonical)
            if error:
                invalid[field].add(error)
            return
        normalized = str(value or "").strip().lower().replace(" ", "")
        if normalized:
            values[field].add(normalized)

    # Full-specific fields are safe on mixed split rows.  Generic producer
    # fields are used only when this row's primary measurement is the requested
    # variant, preventing a split endpoint/precision from contaminating its Full
    # companion row.
    if variant_key == "full":
        direct_aliases = {
            "preprocessing_contract_sha256": (
                "full_preprocessing_contract_sha256", "full_preprocess_contract_sha256",
            ),
            "endpoint_contract_hash": (
                "full_endpoint_contract_hash", "full_output_endpoint_contract_sha256",
            ),
            "source_model_sha256": (
                "full_model_sha256", "source_model_sha256", "model_sha256",
            ),
            "runtime_artifact_sha256": (
                "full_runtime_artifact_sha256", "full_artifact_sha256",
                "full_engine_sha256", "full_hef_sha256", "full_dxnn_sha256",
            ),
            "runtime_precision_identity": (
                "full_runtime_precision_identity", "full_runtime_precision",
                "full_execution_precision",
            ),
        }
    else:
        direct_aliases = {
            "preprocessing_contract_sha256": ("preprocessing_contract_sha256",),
            "endpoint_contract_hash": ("endpoint_contract_hash",),
            "source_model_sha256": ("source_model_sha256", "model_sha256"),
            "runtime_artifact_sha256": ("runtime_artifact_sha256", "artifact_sha256"),
            "runtime_precision_identity": (
                "runtime_precision_identity", "execution_precision",
            ),
        }
    if selected_variant == variant_key:
        direct_aliases = {
            **direct_aliases,
            "preprocessing_contract_sha256": (
                *direct_aliases["preprocessing_contract_sha256"],
                "preprocessing_contract_sha256", "preprocess_contract_sha256",
            ),
            "endpoint_contract_hash": (
                *direct_aliases["endpoint_contract_hash"], "endpoint_contract_hash",
            ),
            "runtime_artifact_sha256": (
                *direct_aliases["runtime_artifact_sha256"],
                "runtime_artifact_sha256", "artifact_sha256",
            ),
            "runtime_precision_identity": (
                *direct_aliases["runtime_precision_identity"],
                "runtime_precision_identity", "execution_precision",
            ),
        }
    for field, aliases in direct_aliases.items():
        for alias in aliases:
            add(field, row.get(alias))

    if selected_variant == variant_key:
        attestation = _mapping(row.get("output_endpoint_attestation"))
        add("endpoint_contract_hash", attestation.get("endpoint_contract_hash"))
        producer = _mapping(row.get("candidate_execution_contract"))
        if producer:
            add("preprocessing_contract_sha256", producer.get("preprocessing_contract_sha256"))
            add("endpoint_contract_hash", producer.get("endpoint_contract_hash"))
            add("runtime_precision_identity", producer.get("runtime_precision_identity"))
            model = _mapping(producer.get("model"))
            add("source_model_sha256", model.get("source_onnx_sha256"))
            add("runtime_artifact_sha256", model.get("runtime_artifact_sha256"))

    for request in _quality_requests_for_variant_v269d(row, variant_key):
        producer = _mapping(request.get("producer_identity"))
        contract = _mapping(request.get("quality_contract"))
        for block in (request, producer, contract):
            add("preprocessing_contract_sha256", block.get("preprocessing_contract_sha256"))
            add("endpoint_contract_hash", block.get("endpoint_contract_hash"))
            add("runtime_precision_identity", block.get("runtime_precision_identity"))
            add("runtime_artifact_sha256", block.get("runtime_artifact_sha256"))
            add("source_model_sha256", block.get("source_model_sha256"))
        model = _mapping(producer.get("model"))
        add("source_model_sha256", model.get("source_onnx_sha256"))
        add("runtime_artifact_sha256", model.get("runtime_artifact_sha256"))
        preprocessing = _mapping(producer.get("preprocessing"))
        endpoint = _mapping(producer.get("endpoint"))
        precision = _mapping(producer.get("precision"))
        add("preprocessing_contract_sha256", preprocessing.get("sha256"))
        add("endpoint_contract_hash", endpoint.get("sha256"))
        add("runtime_precision_identity", precision.get("identity"))

    # Legacy BenchmarkSet rows predate the explicit preprocessing-contract SHA,
    # but do record the effective prepared-feed policy.  Hash only semantic
    # preprocessing fields (not warmup/loop counts) and only when no authoritative
    # SHA exists.  This makes a raw-vs-norm Full baseline disagreement visible
    # instead of silently selecting the first split case.
    if not values["preprocessing_contract_sha256"]:
        policy = _mapping(row.get("benchmark_input_policy"))
        semantic_policy = {
            key: policy.get(key)
            for key in (
                "image_scale", "preprocess_mode", "normalization",
                "color_space", "layout", "letterbox", "letterbox_pad_value",
                "input_dtype", "input_shape",
            )
            if policy.get(key) not in (None, "")
        }
        if semantic_policy:
            encoded = json.dumps(
                semantic_policy, sort_keys=True, separators=(",", ":"),
                ensure_ascii=True,
            ).encode("utf-8")
            add(
                "preprocessing_contract_sha256",
                hashlib.sha256(encoded).hexdigest(),
            )

    return {
        "schema": "onnx-splitpoint/frozen-result-identity-evidence",
        "schema_version": 1,
        "variant": variant_key,
        "values": {key: sorted(items) for key, items in values.items() if items},
        "invalid": {key: sorted(items) for key, items in invalid.items() if items},
        "runtime_precision_identity_raw_representations": sorted(
            value for value in runtime_precision_raw if value
        ),
        "runtime_precision_identity_canonical_candidates": sorted(
            values["runtime_precision_identity"]
        ),
        "runtime_precision_identity_resolution_status": (
            "conflict"
            if invalid["runtime_precision_identity"]
            or len(values["runtime_precision_identity"]) > 1
            else "canonical"
            if len(values["runtime_precision_identity"]) == 1
            else "missing"
        ),
    }


def _unique_frozen_identity_v269d(
    evidence: Mapping[str, Any], field: str,
) -> str:
    values = _mapping(evidence.get("values")).get(field)
    candidates = [str(value) for value in _as_list(values) if str(value)]
    invalid = _mapping(evidence.get("invalid")).get(field)
    if len(set(candidates)) == 1 and not _as_list(invalid):
        return candidates[0]
    return ""


def _frozen_identity_or_legacy_v269d(
    evidence: Mapping[str, Any], field: str, legacy: Any,
) -> str:
    values = _as_list(_mapping(evidence.get("values")).get(field))
    invalid = _as_list(_mapping(evidence.get("invalid")).get(field))
    if values or invalid:
        return _unique_frozen_identity_v269d(evidence, field)
    return str(legacy or "").strip()


def _nested(mapping: Mapping[str, Any], *keys: str) -> Mapping[str, Any]:
    cur: Any = mapping
    for key in keys:
        if not isinstance(cur, Mapping):
            return {}
        cur = cur.get(key)
    return cur if isinstance(cur, Mapping) else {}


def _deepx_contract_block(row: Mapping[str, Any]) -> Mapping[str, Any]:
    # direct normalized/template top-level block
    direct = row.get("deepx_stage2_contract")
    if isinstance(direct, Mapping):
        return direct
    # validation_report layout
    block = _nested(row, "interface_checks", "checks", "deepx_stage2_contract")
    if block:
        return block
    return {}


def _stage2_gate_block(row: Mapping[str, Any]) -> Mapping[str, Any]:
    direct = row.get("stage2_accelerator_calibration_gate")
    if isinstance(direct, Mapping):
        return direct
    block = _nested(row, "deployment_contract", "stage2_accelerator_calibration_gate")
    if block:
        return block
    return {}


def _backend_tokens(row: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(row.get("backend_tokens"))


def _provider_from_stage(stage: Any) -> str:
    if isinstance(stage, Mapping):
        ty = str(stage.get("type") or stage.get("kind") or "").strip().lower()
        if ty == "hailo":
            return _canon_backend_token(stage.get("hw_arch") or stage.get("arch") or stage.get("id") or "hailo8")
        if ty in {"onnxruntime", "ort"}:
            return _canon_backend_token(stage.get("provider") or "")
    return _canon_backend_token(stage)


def _full_backend(row: Mapping[str, Any], tag: str = "") -> str:
    """Backend that produced the full-model baseline timing.

    Legacy rows can contain a full-reference timing inside a split/matrix run.
    Use explicit full-provider metadata when available and never return a mixed
    token such as ``hailo8_to_tensorrt`` as a full backend.
    """
    tokens = _backend_tokens(row)
    run_cfg = _mapping(row.get("run_cfg"))
    explicit_values = (
        row.get("full_provider"),
        row.get("full_backend"),
        tokens.get("full"),
        tokens.get("provider"),
        row.get("provider"),
        run_cfg.get("full_provider"),
        run_cfg.get("provider"),
        row.get("backend_drift_provider"),
        row.get("full_session_providers_in_use"),
        row.get("full_session_providers_requested"),
    )
    for value in explicit_values:
        b = _canon_backend_token(value)
        if b and not _is_mixed_backend_token(b) and b not in {"auto", "default"}:
            return b

    # Same-backend rows can use the stage backend as the full backend.
    stage1 = _stage_backend(row, "stage1", "stage1")
    stage2 = _stage_backend(row, "stage2", "stage2")
    if stage1 and stage2 and stage1 == stage2:
        return stage1

    primary = str(row.get("primary_variant") or row.get("variant") or "").strip().lower()
    backend = _backend(row, tag=tag)
    if primary == "full" and backend and not _is_mixed_backend_token(backend) and backend not in {"auto", "unknown"}:
        return backend
    if not stage1 and not stage2 and backend and not _is_mixed_backend_token(backend) and backend not in {"auto", "unknown"}:
        return backend
    return ""


def _stage_backend(row: Mapping[str, Any], stage_key: str, token_key: str) -> str:
    tokens = _backend_tokens(row)
    for value in (
        row.get(f"{stage_key}_provider"),
        tokens.get(token_key),
        _mapping(row.get("run_cfg")).get(f"{stage_key}_provider"),
        row.get(stage_key),
        _provider_from_stage(_mapping(row.get("plan_run")).get(stage_key)),
    ):
        b = _canon_backend_token(value)
        if b:
            return b
    return ""


def _deployment_contract_for_variant(row: Mapping[str, Any], variant: str) -> Mapping[str, Any]:
    """Select deployment metadata for the measured output variant.

    v2.63 attached a report-global Hailo Full raw-head contract to every row in
    the same BenchmarkSet report.  Prefer the variant-scoped map emitted by the
    generated runner so decoded CPU/TensorRT/composed rows cannot inherit that
    unrelated contract.
    """
    key = str(variant or "").strip().lower()
    direct = row.get("deployment_contracts_by_variant")
    if isinstance(direct, Mapping) and isinstance(direct.get(key), Mapping):
        return _mapping(direct.get(key))
    deployment = _mapping(row.get("deployment_contract"))
    nested = deployment.get("contracts_by_variant")
    if isinstance(nested, Mapping) and isinstance(nested.get(key), Mapping):
        return _mapping(nested.get(key))
    summary = row.get("deployment_contract_summary")
    if isinstance(summary, Mapping):
        summary_variant = str(summary.get("variant") or "").strip().lower()
        if not summary_variant or summary_variant == key:
            return _mapping(summary)
    return {}


def _full_endpoint_mode(row: Mapping[str, Any]) -> str:
    scoped = _deployment_contract_for_variant(row, "full")
    for key in ("endpoint_mode", "full_endpoint_mode"):
        if scoped.get(key) not in (None, ""):
            return str(scoped.get(key)).strip()
    explicit = _first(row, ["full_endpoint_mode", "endpoint_mode"])
    if explicit not in (None, ""):
        return str(explicit).strip()
    timings = _mapping(row.get("timings"))
    full_blk = _mapping(timings.get("full"))
    if full_blk.get("endpoint_mode") not in (None, ""):
        return str(full_blk.get("endpoint_mode")).strip()
    deploy = _mapping(row.get("deployment_contract"))
    if deploy.get("full_endpoint_mode") not in (None, ""):
        return str(deploy.get("full_endpoint_mode")).strip()
    return "decoded_or_native"


def _best_full_latency(row: Mapping[str, Any]) -> tuple[Optional[float], Optional[float], Optional[float], str]:
    """Return (total_full_latency, raw_full_latency, e2e_latency, status).

    For Hailo raw-head full baselines, ``full_mean_ms`` is only the accelerator
    prefix.  The thesis/fair baseline is the raw-head HEF plus host-tail E2E
    timing when available.  Keep both values in the normalized row so reports can
    show accelerator-only and complete E2E latency separately.
    """
    # A successful run_model diagnostic must never become a prepared-feed
    # Full measurement after that required path failed (observed in 2.79.26).
    if (
        str(row.get("performance_benchmark_source") or "")
        == "explicit_deepx_contract_required"
        and str(row.get("latency_semantics") or "") == "diagnostic_only"
    ):
        return None, None, None, "none"
    timings = _mapping(row.get("timings"))
    full_block = _mapping(timings.get("full"))
    # Prefer the dedicated full-variant block.  Split validation reports may
    # carry top-level latency fields for their primary composed variant; using
    # those for TensorRT Full fabricated duplicate split latencies.
    raw = _float_or_none(_first(full_block, ["mean_ms", "latency_ms", "total_latency_ms"]))
    if raw is None:
        raw = _float_or_none(_first(row, ["full_mean_ms", "full_latency_ms"]))
    e2e = _float_or_none(_first(full_block, ["e2e_mean_ms", "e2e_latency_ms"]))
    if e2e is None:
        e2e = _float_or_none(_first(row, ["full_e2e_latency_ms", "full_e2e_mean_ms", "full_e2e_ms"]))
    endpoint = _full_endpoint_mode(row).lower()
    if e2e is not None and ("raw" in endpoint or "head" in endpoint or _bool_or_none(_mapping(row.get("deployment_contract")).get("full_raw_detection_head")) is True):
        return e2e, raw, e2e, "full_raw_head_e2e"
    if raw is not None:
        return raw, raw, e2e, "full"
    if e2e is not None:
        return e2e, raw, e2e, "full_e2e"
    return None, raw, e2e, "none"


def _variant_status_ok(row: Mapping[str, Any], name: str) -> bool:
    statuses = _mapping(row.get("variant_status"))
    measured = {str(x).strip().lower() for x in _as_list(row.get("measured_variants"))}
    status = str(statuses.get(name) or "").strip().lower()
    return status in {"ok", "measured", "success", "passed"} or str(name).lower() in measured


def _validation_ok_for_variant(row: Mapping[str, Any], name: str) -> Optional[bool]:
    comps = _mapping(row.get("comparisons"))
    comp = _mapping(comps.get(name))
    if comp:
        status = str(comp.get("status") or "").lower()
        if status in {"ok", "skipped_layout_mismatch"}:
            passed = comp.get("passed")
            if passed is not None:
                return _bool_or_none(passed)
        if status in {"error", "failed"}:
            return False
    if name == str(row.get("primary_variant") or "").strip().lower():
        return _bool_or_none(_first(row, ["validation_ok", "final_pass", "eps_pass"]))
    return _bool_or_none(_first(row, ["validation_ok", "eps_pass"]))




def _pipeline_block(row: Mapping[str, Any]) -> Mapping[str, Any]:
    direct = row.get("pipeline_timing_model")
    if isinstance(direct, Mapping):
        return direct
    direct = row.get("pipeline")
    if isinstance(direct, Mapping):
        return direct
    direct = row.get("throughput")
    if isinstance(direct, Mapping):
        return direct
    return {}


def _pipeline_first(row: Mapping[str, Any], block: Mapping[str, Any], row_keys: Sequence[str], block_keys: Sequence[str]) -> Optional[float]:
    v = _float_or_none(_first(row, row_keys))
    if v is not None:
        return v
    for key in block_keys:
        v = _float_or_none(block.get(key))
        if v is not None:
            return v
    return None


def _norm_backend_name(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _is_cross_backend_split_name(backend: str, stage1: str = "", stage2: str = "") -> bool:
    b = _norm_backend_name(backend)
    if "_to_" in b or "->" in str(backend or "") or "→" in str(backend or ""):
        return True
    s1 = _norm_backend_name(stage1)
    s2 = _norm_backend_name(stage2)
    return bool(s1 and s2 and s1 != s2)


def _throughput_taxonomy(*, variant: str, backend: str, stage1: str = "", stage2: str = "") -> tuple[str, bool, str]:
    """Return (kind, pipeline_applicable, primary_metric)."""
    v = str(variant or "").lower()
    if v == "full":
        return "full_backend", False, "full_backend_throughput_fps"
    if v == "split" and _is_cross_backend_split_name(backend, stage1, stage2):
        return "heterogeneous_pipeline", True, "heterogeneous_pipeline_fps"
    if v == "split":
        return "same_backend_split_diagnostic", False, "same_backend_composed_fps"
    return "component_or_diagnostic", False, "diagnostic_fps"




# v58i: preserve u.RECS row-level energy fields when benchmark_results rows
# are normalized.  Earlier normalize_benchmark_row selected only canonical
# timing/validation fields, so Eval energy merged into benchmark_results_*.json
# was lost again when normalized_results.json was built.  Energy belongs only
# to the matching target variant: full-energy must not be copied to the
# composed/split companion row and composed-energy must not be copied to the
# full companion row.
def _energy_target_matches_variant_v58i(row: Mapping[str, Any], normalized_variant: str) -> bool:
    target = str(row.get("energy_target_variant") or row.get("target_variant") or row.get("energy_variant") or "").strip().lower()
    if not target:
        return True
    variant = str(normalized_variant or "").strip().lower()
    if variant == target:
        return True
    if variant == "split" and target in {"split", "composed", "complete"}:
        return True
    if variant == "composed" and target in {"split", "composed", "complete"}:
        return True
    if variant == "full" and target == "full":
        return True
    return False




def _derive_energy_relpath_from_abs_v58n(path_value: Any, old_rel: Any = None) -> str:
    """Derive a portable run-relative path from an absolute Energy artifact path."""
    rel = str(old_rel or "").strip()
    if rel and rel not in {"energy_aggregate.json", "energy_summary.json"} and "/" in rel.replace("\\", "/"):
        return rel.replace("\\", "/")
    s = str(path_value or "").replace("\\", "/")
    if not s:
        return rel
    idx = s.find("/models/")
    if idx >= 0:
        return s[idx + 1:]
    idx = s.find("/reports/")
    if idx >= 0:
        return s[idx + 1:]
    return rel or Path(s).name


def _normalize_full_case_id_for_dedupe_v58n(row: Mapping[str, Any]) -> str:
    case = str(row.get("case_id") or "").strip().lower()
    variant = str(row.get("variant") or "").strip().lower()
    primary_variant = str(row.get("primary_variant") or "").strip().lower()
    backend = _canon_backend_token(row.get("backend") or row.get("run_id") or row.get("provider") or "")
    full_provider = _canon_backend_token(row.get("full_provider") or row.get("full_backend") or "")
    # A Full model is independent of the split boundary that happened to host
    # its legacy companion timing.  Canonicalize it before grouping; the source
    # boundary is retained separately by the dedupe step for auditability.
    if variant == "full" or primary_variant == "full":
        return "full"
    if not case and (variant == "full" or full_provider or backend in {"deepx_m1", "tensorrt", "ort_tensorrt", "cpu_ort", "ort_cpu", "cuda_ort", "ort_cuda"}):
        return "full"
    return case

def _copy_energy_fields_v58i(out: dict[str, Any], row: Mapping[str, Any], *, normalized_variant: str) -> dict[str, Any]:
    if not _energy_target_matches_variant_v58i(row, normalized_variant):
        return out
    for k, v in dict(row or {}).items():
        lk = str(k).lower()
        if not (
            "energy" in lk
            or "urecs" in lk
            or "power" in lk
            or "work_units" in lk
            or "fps_per_watt" in lk
            or "j_per_frame" in lk
            or lk in {"avg_power_w", "weighted_avg_power_w"}
        ):
            continue
        if v in (None, "", [], {}):
            continue
        out[k] = v
    # Normalize common aliases for dashboard/report code.
    if out.get("energy_measurement_scope") in (None, "") and row.get("energy_measurement_scope") not in (None, ""):
        out["energy_measurement_scope"] = row.get("energy_measurement_scope")
    if out.get("energy_enabled") in (None, "") and any("energy" in str(k).lower() for k in row.keys()):
        out["energy_enabled"] = True
    if out.get("energy_aggregate_path") not in (None, ""):
        out["energy_aggregate_relpath"] = _derive_energy_relpath_from_abs_v58n(out.get("energy_aggregate_path"), out.get("energy_aggregate_relpath"))
    if out.get("energy_summary_path") not in (None, ""):
        out["energy_summary_relpath"] = _derive_energy_relpath_from_abs_v58n(out.get("energy_summary_path"), out.get("energy_summary_relpath"))
    return out




# v59j: thesis-safe validation-level labelling and row-level Energy merge helpers.
def _copy_interface_fields_v59j(out: dict[str, Any], row: Mapping[str, Any]) -> dict[str, Any]:
    iface = _mapping(row.get("interface_checks") or row.get("interface_check") or row.get("split_interface_checks"))
    checks = _mapping(iface.get("checks"))
    stage1 = _mapping(iface.get("stage1_vs_ort") or iface.get("stage1_boundary_vs_ort") or iface.get("boundary_vs_ort") or checks.get("stage1_vs_ort") or checks.get("stage1_boundary_vs_ort") or checks.get("boundary_vs_ort") or row.get("stage1_vs_ort"))
    raw_head = _mapping(row.get("raw_head_drift") or iface.get("raw_head_drift") or checks.get("raw_head_drift") or iface.get("raw_head_vs_ort") or checks.get("raw_head_vs_ort"))

    status = _first(row, ["interface_check_status", "interface_contract_status"])
    if status in (None, "") and iface:
        status = iface.get("status") or iface.get("interface_check_status")
    out["interface_check_status"] = status
    direct_pass = _bool_or_none(_first(row, ["interface_check_pass", "interface_contract_pass"]))
    if direct_pass is None and iface:
        direct_pass = _bool_or_none(iface.get("pass") if iface.get("pass") is not None else iface.get("passed"))
    out["interface_check_pass"] = direct_pass
    out["interface_gate_override"] = iface.get("gate_override")
    out["interface_gate_blocking"] = _bool_or_none(iface.get("gate_blocking"))

    out["interface_stage1_status"] = _first(row, ["interface_stage1_status", "stage1_vs_ort_status"]) or stage1.get("status")
    out["interface_stage1_pass"] = (_bool_or_none(_first(row, ["interface_stage1_pass", "stage1_vs_ort_pass"]))
                                      if _first(row, ["interface_stage1_pass", "stage1_vs_ort_pass"]) not in (None, "")
                                      else _bool_or_none(stage1.get("pass") if stage1.get("pass") is not None else stage1.get("passed")))
    out["interface_stage1_max_abs"] = _float_or_none(_first(row, ["interface_stage1_max_abs", "stage1_vs_ort_max_abs"])) or _float_or_none(stage1.get("max_abs") or stage1.get("max_abs_error"))
    out["interface_stage1_mean_abs_mean"] = _float_or_none(_first(row, ["interface_stage1_mean_abs_mean", "stage1_vs_ort_mean_abs_mean"])) or _float_or_none(stage1.get("mean_abs_mean") or stage1.get("mean_abs") or stage1.get("mean_abs_error"))
    out["interface_stage1_num_compared_tensors"] = _float_or_none(_first(row, ["interface_stage1_num_compared_tensors", "stage1_vs_ort_num_compared_tensors"])) or _float_or_none(stage1.get("num_compared_tensors") or stage1.get("compared_tensors"))
    out["interface_stage1_structural_pass"] = _bool_or_none(
        stage1.get("structural_pass")
    )
    out["interface_stage1_mapping_pass"] = _bool_or_none(
        stage1.get("mapping_pass")
    )
    out["interface_stage1_shape_pass"] = _bool_or_none(
        stage1.get("shape_pass")
    )
    out["interface_stage1_numeric_pass"] = _bool_or_none(
        stage1.get("numeric_pass")
    )

    out["raw_head_drift_status"] = _first(row, ["raw_head_drift_status"]) or raw_head.get("status")
    out["raw_head_drift_pass"] = (_bool_or_none(_first(row, ["raw_head_drift_pass"]))
                                  if _first(row, ["raw_head_drift_pass"]) not in (None, "")
                                  else _bool_or_none(raw_head.get("pass") if raw_head.get("pass") is not None else raw_head.get("passed")))
    out["raw_head_drift_max_abs"] = _float_or_none(_first(row, ["raw_head_drift_max_abs"])) or _float_or_none(raw_head.get("max_abs") or raw_head.get("max_abs_error"))
    out["raw_head_drift_mean_abs_mean"] = _float_or_none(_first(row, ["raw_head_drift_mean_abs_mean"])) or _float_or_none(raw_head.get("mean_abs_mean") or raw_head.get("mean_abs") or raw_head.get("mean_abs_error"))
    return out


def _is_heterogeneous_split_row_v59j(row: Mapping[str, Any]) -> bool:
    if str(row.get("variant") or "").strip().lower() != "split":
        return False
    backend = str(row.get("backend") or row.get("run_id") or "").strip().lower().replace("-", "_")
    if "_to_" in backend or "->" in backend or "→" in backend:
        return True
    s1 = _canon_backend_token(row.get("stage1_provider") or row.get("stage1") or "")
    s2 = _canon_backend_token(row.get("stage2_provider") or row.get("stage2") or "")
    return bool(s1 and s2 and s1 != s2)


def _apply_validation_level_fields_v59j(row: dict[str, Any]) -> dict[str, Any]:
    variant = str(row.get("variant") or "").strip().lower()
    hetero = _is_heterogeneous_split_row_v59j(row)
    runtime_ok = _bool_or_none(row.get("runtime_ok"))
    semantic = _bool_or_none(row.get("semantic_e2e_pass"))
    if semantic is None:
        for key in ("semantic_validation_ok", "semantic_validation_passed", "final_pass_all", "final_pass", "validation_ok"):
            semantic = _bool_or_none(row.get(key))
            if semantic is not None:
                break
    if runtime_ok is False:
        semantic = False if semantic is not True else semantic
    row["semantic_e2e_pass"] = semantic

    # Interface contract is a shape/layout/protocol claim. For full rows it is not
    # applicable; for heterogeneous rows it may be available, warning, or unknown.
    iface_pass = _bool_or_none(row.get("interface_check_pass"))
    iface_status = str(row.get("interface_check_status") or "").strip()
    structural = [
        _bool_or_none(row.get(key)) for key in (
            "interface_stage1_structural_pass", "interface_stage1_mapping_pass",
            "interface_stage1_shape_pass",
        )
    ]
    if hetero and all(value is not None for value in structural):
        # Legacy interface_check_pass combines numeric and structural results.
        # Explicit mapping/shape evidence owns the structural axis only.
        iface_pass = all(structural)
        iface_status = "structural_interface_pass" if iface_pass else "structural_interface_failed"
    proxy_override = bool(
        str(row.get("runtime_contract_decision") or "").strip().lower()
        == "pass"
        and str(row.get("interface_gate_override") or "").strip().lower()
        == "primary_cpu_full_proxy_passed"
        and row.get("interface_gate_blocking") is False
        and row.get("interface_stage1_structural_pass") is True
    )
    if proxy_override:
        # The strict cut numerics remain false below, while the structural I/O
        # contract is admissible through the exactly bound end-to-end proxy.
        iface_pass = True
        iface_status = (
            iface_status or "warning_primary_cpu_full_proxy_override"
        )
    if iface_pass is None and row.get("interface_stage1_pass") not in (None, ""):
        # Stage1-vs-ORT failure is a strict numeric drift, but also marks that the
        # interface checker ran. Keep warning statuses distinct below.
        iface_status = iface_status or str(row.get("interface_stage1_status") or "")
    if iface_pass is None and row.get("deepx_stage2_contract_pass") not in (None, ""):
        iface_pass = _bool_or_none(row.get("deepx_stage2_contract_pass"))
        iface_status = iface_status or str(row.get("deepx_stage2_contract_status") or "deepx_stage2_contract")
    if iface_pass is None and row.get("stage2_feature_tensor_contract_status") not in (None, ""):
        st = str(row.get("stage2_feature_tensor_contract_status") or "").strip().lower()
        iface_status = iface_status or st
        if st in {"ok", "pass", "passed", "available", "trusted", "producer_exact", "feature_tensor_contract_ok"}:
            iface_pass = True
        elif any(tok in st for tok in ("fail", "reject", "mismatch", "missing", "error")):
            iface_pass = False
    if variant == "full":
        row["interface_contract_pass"] = None
        row["interface_contract_status"] = "not_applicable_full_backend"
    elif not hetero:
        # Numerical agreement is a separate axis. Use actual structure evidence
        # even for same-backend splits; execution alone does not prove the cut.
        if all(value is not None for value in structural):
            iface_pass = all(structural)
        elif iface_pass is None and semantic is True:
            # A successful comparison still carries its legacy positive
            # structure evidence; a numeric mismatch does not prove breakage.
            iface_pass = True
        row["interface_contract_pass"] = iface_pass
        row["interface_contract_status"] = iface_status or (
            "same_backend_contract" if iface_pass is not None else "boundary_check_unavailable"
        )
    else:
        if iface_pass is None:
            row["interface_contract_pass"] = None
            row["interface_contract_status"] = iface_status or "boundary_check_unavailable"
        else:
            row["interface_contract_pass"] = iface_pass
            if iface_pass is False and not iface_status:
                iface_status = "interface_warning_or_failure"
            row["interface_contract_status"] = iface_status or "interface_contract_pass"

    strict_pass = _bool_or_none(row.get("strict_boundary_numeric_pass"))
    strict_status = str(row.get("strict_boundary_numeric_status") or "").strip()
    if strict_pass is None and row.get("interface_stage1_pass") not in (None, ""):
        strict_pass = _bool_or_none(row.get("interface_stage1_pass"))
        strict_status = strict_status or str(row.get("interface_stage1_status") or "stage1_vs_ort")
    if strict_pass is None and row.get("raw_head_drift_pass") not in (None, ""):
        strict_pass = _bool_or_none(row.get("raw_head_drift_pass"))
        strict_status = strict_status or str(row.get("raw_head_drift_status") or "raw_head_drift")
    if strict_pass is None and not hetero and variant == "split":
        # Same-backend split diagnostics are the place where strict numeric
        # equivalence is meaningful. Use explicit normalized numeric evidence if
        # present, otherwise label it as derived from the existing same-backend gate.
        shape = _bool_or_none(row.get("output_shape_match"))
        max_abs = _float_or_none(row.get("max_abs_error"))
        mean_abs = _float_or_none(row.get("mean_abs_error"))
        cos = _float_or_none(row.get("cosine_similarity"))
        if shape is False:
            strict_pass = False
            strict_status = "output_shape_mismatch"
        elif max_abs is not None or mean_abs is not None or cos is not None:
            strict_pass = bool((shape is not False) and (max_abs is None or max_abs <= 1e-3) and (mean_abs is None or mean_abs <= 1e-4) and (cos is None or cos >= 0.999))
            strict_status = "same_backend_numeric_metrics"
        elif semantic is not None:
            strict_pass = semantic
            strict_status = "same_backend_semantic_gate_no_raw_boundary_dump"
    if variant == "full":
        row["strict_boundary_numeric_pass"] = None
        row["strict_boundary_numeric_status"] = "not_applicable_full_backend"
    elif strict_pass is None:
        row["strict_boundary_numeric_pass"] = None
        row["strict_boundary_numeric_status"] = strict_status or ("not_required_heterogeneous_backend" if hetero else "boundary_check_unavailable")
    else:
        row["strict_boundary_numeric_pass"] = strict_pass
        row["strict_boundary_numeric_status"] = strict_status or ("strict_numeric_pass" if strict_pass else "strict_numeric_warning")

    visual = _bool_or_none(row.get("visual_validation_ok"))
    row["visual_validation_ok"] = visual

    if semantic is False:
        level = "failed"
        label = "validation failed"
    elif variant == "full":
        level = "full_semantic"
        label = "full semantic pass" if semantic is True else "full unvalidated"
    elif not hetero:
        if row.get("strict_boundary_numeric_pass") is True:
            level = "strict_numeric"
            label = "strict numeric pass"
        else:
            level = "same_backend_semantic"
            label = "same-backend semantic pass" if semantic is True else "same-backend unvalidated"
    else:
        if row.get("interface_contract_pass") is False:
            level = "semantic_with_interface_warning"
            label = "semantic pass; interface warning" if semantic is True else "interface warning"
        elif row.get("interface_contract_pass") is None:
            level = "semantic_without_boundary_dump"
            label = "semantic pass; boundary check unavailable" if semantic is True else "boundary check unavailable"
        elif row.get("strict_boundary_numeric_pass") is False:
            level = "semantic_with_boundary_numeric_warning"
            label = "semantic pass; boundary numeric warning" if semantic is True else "boundary numeric warning"
        else:
            level = "semantic_e2e"
            label = "semantic pass" if semantic is True else "semantic unvalidated"
    if visual is False and semantic is not False:
        label = f"{label}; visual warning" if label else "visual warning"
    row["validation_claim_level"] = level
    row["validation_claim_label"] = label
    row["validation_level_summary"] = label

    # v59ej: add explicit task/accuracy/ranking eligibility gates.  This makes
    # dashboards/rankings gate by declared task validity instead of raw speed.
    task = "detection" if (row.get("mini_coco_ap50_primary") is not None or "yolo" in str(row.get("model_id") or "").lower()) else "classification" if row.get("classification_top1") is not None or row.get("classification_top5") is not None else ""
    apply_gate_fields(row, task=task, source="normalized_results")
    return row


def _strip_energy_fields_v59j(row: dict[str, Any]) -> dict[str, Any]:
    for k in list(row.keys()):
        lk = str(k).lower()
        if (
            "energy" in lk
            or "urecs" in lk
            or "power" in lk
            or "work_units" in lk
            or "fps_per_watt" in lk
            or "j_per_frame" in lk
            or lk in {"avg_power_w", "weighted_avg_power_w"}
        ):
            row.pop(k, None)
    return row


def _row_has_row_level_energy_v59j(row: Mapping[str, Any]) -> bool:
    """Return True only for usable row-attributed u.RECS energy.

    v59k tightens v59j: a target id/path alone is diagnostic provenance, not a
    valid energy measurement.  Partial/failed target aggregates and rows without
    finite energy metrics are reported, but they are not counted as energy-
    complete.
    """
    source = str(row.get("energy_row_level_source") or "").strip().lower()
    coverage = str(row.get("energy_coverage_status") or "").strip().lower()
    if source in {"dispatch_only", "dispatch_only_no_row_energy", "target_aggregate_unusable"}:
        return False
    if coverage in {"dispatch_only_no_row_energy", "target_aggregate_partial", "target_aggregate_failed", "missing_row_level_energy", "missing_or_not_measured"}:
        return False

    explicit_row_metric = any(
        _energy_finite_positive_v59k(row.get(k)) is not None
        for k in (
            "row_energy_streaming_j_per_frame",
            "row_energy_latency_j_per_inference",
            "row_energy_j_per_frame",
            "row_energy_j_per_inference",
        )
    )
    if explicit_row_metric:
        return True

    target_hint = False
    for key in ("energy_target_id", "energy_target_case", "target_id", "target_case"):
        if row.get(key) not in (None, ""):
            target_hint = True
            break
    if not target_hint:
        for key in ("energy_aggregate_relpath", "energy_aggregate_path"):
            val = str(row.get(key) or "").replace("\\", "/")
            if val and "/targets/" in val:
                target_hint = True
                break

    if target_hint:
        return _row_energy_target_ok_v59k(row)

    # Legacy non-target row-scope fields are accepted only when they contain a
    # finite energy/power metric and explicitly claim row scope.
    if _bool_or_none(row.get("energy_row_scope")) is True:
        return _row_has_usable_energy_metric_v59k(row)
    return False

def _energy_norm_token_v59j(value: Any) -> str:
    s = str(value or "").strip().lower()
    s = s.replace("→", "_to_").replace("->", "_to_").replace("/", "_")
    s = s.replace("-", "_").replace(" ", "_").replace("__", "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")


def _energy_backend_aliases_v59j(value: Any) -> set[str]:
    s = _energy_norm_token_v59j(value)
    if not s:
        return set()
    aliases = {s}
    repls = []
    if "tensorrt" in s:
        repls.append(s.replace("tensorrt", "trt"))
    if "trt" in s:
        repls.append(s.replace("trt", "tensorrt"))
    if "deepx_m1" in s:
        repls.append(s.replace("deepx_m1", "deepx"))
        repls.append(s.replace("deepx_m1", "dx_m1"))
    if "dx_m1" in s:
        repls.append(s.replace("dx_m1", "deepx_m1"))
    if s.startswith("ort_"):
        base = s.replace("ort_", "", 1)
        repls.append(base)
        repls.append(f"{base}_ort")
    if s.endswith("_ort"):
        base = s[:-4]
        repls.append(base)
        repls.append(f"ort_{base}")
    if s.endswith("_full"):
        repls.append(s[:-5])
    aliases.update(x for x in repls if x)
    expanded = set(aliases)
    for a in list(aliases):
        if "tensorrt" in a:
            expanded.add(a.replace("tensorrt", "trt"))
        if "trt" in a:
            expanded.add(a.replace("trt", "tensorrt"))
        if a == "tensorrt":
            expanded.update({"trt", "ort_tensorrt"})
        if a == "trt":
            expanded.update({"tensorrt", "ort_tensorrt"})
        if a == "deepx":
            expanded.update({"deepx_m1", "deepx_m1_full"})
        if a == "hailo":
            expanded.add("hailo8")
    return {x for x in expanded if x}


def _row_backend_aliases_v59j(row: Mapping[str, Any]) -> set[str]:
    vals = [row.get(k) for k in ("backend", "run_id", "source_tag", "provider", "full_backend", "full_provider")]
    s1 = _energy_norm_token_v59j(row.get("stage1_provider") or row.get("stage1") or "")
    s2 = _energy_norm_token_v59j(row.get("stage2_provider") or row.get("stage2") or "")
    if s1 and s2:
        vals.append(f"{s1}_to_{s2}")
    out: set[str] = set()
    for v in vals:
        out.update(_energy_backend_aliases_v59j(v))
    return out


def _energy_backend_matches_row_v59j(row: Mapping[str, Any], run_id: Any) -> bool:
    # v59an: for composed/split heterogeneous rows require the target aggregate
    # to match the row backend/run_id itself, not merely the stage2 provider.
    # Without this guard, an ort_tensorrt target can accidentally attach to
    # hailo10_to_tensorrt or deepx_to_tensorrt rows because both contain
    # "tensorrt" as stage2_provider.
    rv = str(row.get("variant") or "").strip().lower()
    rb = _energy_norm_token_v59j(row.get("backend") or row.get("run_id") or row.get("source_tag") or "")
    run_aliases = _energy_backend_aliases_v59j(run_id)
    if rv in {"split", "composed"} and "_to_" in rb:
        strict: set[str] = set()
        for k in ("backend", "run_id", "source_tag"):
            strict.update(_energy_backend_aliases_v59j(row.get(k)))
        return bool(strict & run_aliases)
    return bool(_row_backend_aliases_v59j(row) & run_aliases)


def _energy_target_variant_matches_row_v59j(row: Mapping[str, Any], target_variant: Any) -> bool:
    tv = str(target_variant or "").strip().lower()
    rv = str(row.get("variant") or "").strip().lower()
    if not tv:
        return True
    if tv == rv:
        return True
    if tv in {"composed", "split", "complete"} and rv in {"split", "composed"}:
        return True
    if tv == "full" and rv == "full":
        return True
    return False


def _energy_target_case_matches_row_v59j(row: Mapping[str, Any], target_case: Any, applies_all: bool) -> bool:
    if applies_all:
        return True
    tc = str(target_case or "").strip()
    rc = str(row.get("case_id") or "").strip()
    if not tc or tc.lower() == "all":
        return True
    if tc.lower() == "full":
        return rc.lower() in {"", "full"}
    return tc == rc



def _energy_status_token_v59k(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_")


def _energy_phase_ok_v59k(phase: Mapping[str, Any]) -> bool:
    """True only when a phase produced usable u.RECS post-processing output.

    Failed duration/sizing probes often still carry phase_repeat_count=1.  Those
    counters describe a requested window, not a measured energy window, and must
    not make a row look energy-complete.
    """
    if not isinstance(phase, Mapping) or not phase:
        return False
    if phase.get("ok") is False:
        return False
    status = _energy_status_token_v59k(phase.get("status"))
    if status in {"failed", "fail", "error", "partial", "skipped", "skip", "duration_probe_failed", "no_results"}:
        return False
    if phase.get("ok") is True or status in {"ok", "success", "complete"}:
        return True
    # Legacy phase files may not contain an explicit status.  Accept them only
    # if they carry an actual energy/power metric.
    return any(_float_or_none(phase.get(k)) is not None for k in (
        "avg_energy_total_j", "sum_energy_total_j", "energy_total_j",
        "avg_power_w", "weighted_avg_power_w",
        "avg_energy_per_inference_j", "avg_energy_per_work_unit_j",
        "avg_energy_per_pipeline_frame_j",
    ))


def _energy_aggregate_ok_v59k(agg: Mapping[str, Any]) -> bool:
    if not isinstance(agg, Mapping) or not agg:
        return False
    if agg.get("ok") is False:
        return False
    status = _energy_status_token_v59k(agg.get("status"))
    if status in {"failed", "fail", "error", "partial", "skipped", "skip", "duration_probe_failed", "no_results"}:
        return False
    phases = [p for p in (agg.get("phases") if isinstance(agg.get("phases"), list) else []) if isinstance(p, Mapping)]
    if phases:
        return any(_energy_phase_ok_v59k(p) for p in phases)
    return agg.get("ok") is True or status in {"ok", "success", "complete"}


def _energy_finite_positive_v59k(value: Any) -> Optional[float]:
    v = _float_or_none(value)
    if v is None:
        return None
    try:
        if v > 0:
            return float(v)
    except Exception:
        return None
    return None


def _energy_aggregate_has_metric_v59k(agg: Mapping[str, Any]) -> bool:
    if not isinstance(agg, Mapping) or not agg:
        return False
    for key in ("sum_energy_total_j", "energy_total_j", "target_energy_total_j", "avg_power_w", "avg_power_w_weighted"):
        if _energy_finite_positive_v59k(agg.get(key)) is not None:
            return True
    phases = [p for p in (agg.get("phases") if isinstance(agg.get("phases"), list) else []) if isinstance(p, Mapping)]
    for phase in phases:
        if not _energy_phase_ok_v59k(phase):
            continue
        for key in (
            "avg_energy_total_j", "sum_energy_total_j", "energy_total_j",
            "avg_power_w", "weighted_avg_power_w",
            "avg_energy_per_inference_j", "avg_energy_per_work_unit_j",
            "avg_energy_per_pipeline_frame_j",
        ):
            if _energy_finite_positive_v59k(phase.get(key)) is not None:
                return True
    return False


def _energy_aggregate_is_usable_v59k(agg: Mapping[str, Any]) -> bool:
    return _energy_aggregate_ok_v59k(agg) and _energy_aggregate_has_metric_v59k(agg)


def _row_has_usable_energy_metric_v59k(row: Mapping[str, Any]) -> bool:
    for key in (
        "row_energy_streaming_j_per_frame",
        "row_energy_latency_j_per_inference",
        "row_energy_j_per_frame",
        "row_energy_j_per_inference",
        "energy_streaming_j_per_frame",
        "energy_per_inference_j",
        "energy_streaming_avg_power_w",
        "energy_latency_avg_power_w",
        "energy_work_units_per_j",
        "energy_total_j",
    ):
        if _energy_finite_positive_v59k(row.get(key)) is not None:
            return True
    return False


def _row_energy_target_ok_v59k(row: Mapping[str, Any]) -> bool:
    status = _energy_status_token_v59k(row.get("energy_target_status"))
    if status in {"", "unknown"}:
        # Legacy row-scoped energy may not have target_status.  In that case the
        # existence of a usable metric is enough.
        return _row_has_usable_energy_metric_v59k(row)
    if status not in {"ok", "success", "complete"}:
        return False
    return _row_has_usable_energy_metric_v59k(row)

def _energy_phase_by_name_v59j(agg: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    phases = agg.get("phases") if isinstance(agg.get("phases"), list) else []
    name_l = str(name or "").lower()
    for phase in phases:
        if not isinstance(phase, Mapping):
            continue
        p = str(phase.get("phase") or phase.get("name") or "").strip().lower()
        if p == name_l:
            return phase
    for phase in phases:
        if not isinstance(phase, Mapping):
            continue
        p = str(phase.get("phase") or phase.get("name") or "").strip().lower()
        if name_l in p:
            return phase
    return {}


def _energy_phase_value_v59j(phase: Mapping[str, Any], *keys: str) -> Optional[float]:
    for key in keys:
        v = _float_or_none(phase.get(key))
        if v is not None:
            return v
    return None


def _energy_phase_windows_v59j(phase: Mapping[str, Any]) -> tuple[int, int]:
    # A u.RECS Energy "window" is a measured collector/workload run, not the
    # number of requested windows. Failed duration probes often retain
    # phase_repeat_count=1; ignore those so partial targets are not counted as
    # valid row-level energy.
    if not _energy_phase_ok_v59k(phase):
        return 0, 0
    wc = _energy_phase_value_v59j(phase, "energy_window_count", "window_count", "run_count", "valid_postprocessed_runs")
    vc = _energy_phase_value_v59j(phase, "valid_energy_window_count", "valid_window_count", "valid_postprocessed_runs", "valid_runs", "run_count")
    try:
        wi = int(wc or 0)
    except Exception:
        wi = 0
    try:
        vi = int(vc or 0)
    except Exception:
        vi = 0
    if wi <= 0 and _row_has_usable_energy_metric_v59k(phase):
        wi = 1
    if vi <= 0 and wi > 0 and _row_has_usable_energy_metric_v59k(phase):
        vi = wi
    return wi, vi


def _selected_reference_fps_v59j(row: Mapping[str, Any]) -> Optional[float]:
    for key in ("throughput_primary_fps", "heterogeneous_pipeline_fps", "full_backend_throughput_fps", "pipeline_fps_selected", "same_backend_composed_fps"):
        v = _float_or_none(row.get(key))
        if v is not None and v > 0:
            return v
    lat = _float_or_none(row.get("total_latency_ms") or row.get("split_latency_e2e_ms") or row.get("full_latency_ms"))
    if lat is not None and lat > 0:
        return 1000.0 / lat
    return None


def _energy_payload_from_target_aggregate_v59j(agg: Mapping[str, Any], agg_path: Path, run_root: Path) -> dict[str, Any]:
    latency = _energy_phase_by_name_v59j(agg, "latency")
    streaming = _energy_phase_by_name_v59j(agg, "streaming")
    phases = [p for p in (agg.get("phases") if isinstance(agg.get("phases"), list) else []) if isinstance(p, Mapping)]
    try:
        rel = str(agg_path.relative_to(run_root))
    except Exception:
        rel = _derive_energy_relpath_from_abs_v58n(str(agg_path))
    lat_wc, lat_vc = _energy_phase_windows_v59j(latency)
    stream_wc, stream_vc = _energy_phase_windows_v59j(streaming)
    total_wc = sum(_energy_phase_windows_v59j(p)[0] for p in phases) or lat_wc or stream_wc
    total_vc = sum(_energy_phase_windows_v59j(p)[1] for p in phases) or lat_vc or stream_vc
    lat_j = _energy_phase_value_v59j(latency, "avg_energy_per_inference_j", "avg_energy_per_work_unit_j", "energy_per_inference_j", "energy_j_per_inference")
    stream_j = _energy_phase_value_v59j(streaming, "avg_energy_per_pipeline_frame_j", "avg_energy_per_work_unit_j", "avg_energy_per_inference_j", "energy_j_per_frame")
    stream_power = _energy_phase_value_v59j(streaming, "avg_power_w", "weighted_avg_power_w")
    lat_power = _energy_phase_value_v59j(latency, "avg_power_w", "weighted_avg_power_w")
    total_j = _float_or_none(agg.get("sum_energy_total_j"))
    if total_j is None:
        total_j = sum(float(_energy_phase_value_v59j(p, "sum_energy_total_j", "energy_total_j") or 0.0) for p in phases)
    agg_status = str(agg.get("status") or ("ok" if agg.get("ok") is True else "unknown"))
    usable = _energy_aggregate_is_usable_v59k(agg)

    def _repeat_stat(phase: Mapping[str, Any], metric_names: tuple[str, ...], field: str) -> Any:
        block = phase.get("repeat_statistics") if isinstance(phase.get("repeat_statistics"), Mapping) else {}
        for metric in metric_names:
            row = block.get(metric) if isinstance(block.get(metric), Mapping) else {}
            if row.get(field) is not None:
                return row.get(field)
        return None

    lat_j_std = _repeat_stat(latency, ("energy_per_work_unit_j", "energy_per_inference_j"), "sample_stddev")
    lat_j_lo = _repeat_stat(latency, ("energy_per_work_unit_j", "energy_per_inference_j"), "ci_low")
    lat_j_hi = _repeat_stat(latency, ("energy_per_work_unit_j", "energy_per_inference_j"), "ci_high")
    lat_n = _repeat_stat(latency, ("energy_per_work_unit_j", "energy_per_inference_j"), "n")
    stream_j_std = _repeat_stat(streaming, ("energy_per_pipeline_frame_j", "energy_per_work_unit_j"), "sample_stddev")
    stream_j_lo = _repeat_stat(streaming, ("energy_per_pipeline_frame_j", "energy_per_work_unit_j"), "ci_low")
    stream_j_hi = _repeat_stat(streaming, ("energy_per_pipeline_frame_j", "energy_per_work_unit_j"), "ci_high")
    stream_n = _repeat_stat(streaming, ("energy_per_pipeline_frame_j", "energy_per_work_unit_j"), "n")
    stream_power_std = _repeat_stat(streaming, ("avg_power_w",), "sample_stddev")
    stream_power_lo = _repeat_stat(streaming, ("avg_power_w",), "ci_low")
    stream_power_hi = _repeat_stat(streaming, ("avg_power_w",), "ci_high")
    coverage = "target_aggregate_merged" if usable else ("target_aggregate_partial" if _energy_status_token_v59k(agg_status) == "partial" else "target_aggregate_failed")
    payload = {
        "energy_enabled": True,
        "energy_source": "urecs_target_aggregate",
        "energy_row_scope": bool(usable),
        "energy_row_level_source": "target_aggregate_row_scope" if usable else "target_aggregate_unusable",
        "energy_merge_source": "normalized_results_target_aggregate_fallback_v59k",
        "energy_coverage_status": coverage,
        "energy_measurement_scope": str(agg.get("energy_measurement_scope") or "command_energy"),
        "energy_physical_scope": str(agg.get("energy_physical_scope") or streaming.get("energy_physical_scope") or latency.get("energy_physical_scope") or ""),
        "energy_window_label": str(agg.get("energy_window_label") or streaming.get("energy_window_label") or latency.get("energy_window_label") or "command"),
        "energy_confidence_level": agg.get("confidence_level") or streaming.get("confidence_level") or latency.get("confidence_level"),
        "energy_target_status": agg_status,
        "energy_target_ok": bool(usable),
        "energy_target_case": str(agg.get("energy_target_case") or agg.get("target_case") or ""),
        "energy_target_variant": str(agg.get("energy_target_variant") or agg.get("target_variant") or ""),
        "energy_applies_to_all_cases": bool(agg.get("energy_applies_to_all_cases")),
        "energy_target_id": str(agg.get("target_id") or ""),
        "energy_target_error_summary": "; ".join(str(p.get("phase") or p.get("name") or "phase") + ":" + str(p.get("error") or p.get("status") or "failed") for p in phases if isinstance(p, Mapping) and not _energy_phase_ok_v59k(p))[:500],
        "energy_target_phase_count": len(phases),
        "energy_target_window_count": total_wc,
        "energy_target_valid_window_count": total_vc,
        "energy_window_count": total_wc,
        "energy_valid_window_count": total_vc,
        "energy_target_valid_window_ratio": (float(total_vc) / float(total_wc)) if total_wc else None,
        "energy_aggregate_path": str(agg_path),
        "energy_aggregate_relpath": rel,
        "energy_total_j": total_j,
        "row_energy_latency_j_per_inference": lat_j,
        "row_energy_latency_j_per_inference_sample_stddev": lat_j_std,
        "row_energy_latency_j_per_inference_ci_low": lat_j_lo,
        "row_energy_latency_j_per_inference_ci_high": lat_j_hi,
        "energy_latency_repeat_n": lat_n or latency.get("valid_postprocessed_runs"),
        "energy_per_inference_j": lat_j,
        "row_host_normalized_energy_latency_j_per_inference_est": _energy_phase_value_v59j(latency, "avg_host_normalized_energy_per_work_unit_est_j", "host_normalized_energy_per_work_unit_est_j"),
        "energy_latency_avg_power_w": lat_power,
        "row_energy_streaming_j_per_frame": stream_j,
        "row_energy_streaming_j_per_frame_sample_stddev": stream_j_std,
        "row_energy_streaming_j_per_frame_ci_low": stream_j_lo,
        "row_energy_streaming_j_per_frame_ci_high": stream_j_hi,
        "energy_streaming_repeat_n": stream_n or streaming.get("valid_postprocessed_runs"),
        "energy_streaming_j_per_frame": stream_j,
        "energy_streaming_avg_power_w": stream_power,
        "row_host_normalized_energy_streaming_j_per_frame_est": _energy_phase_value_v59j(streaming, "avg_host_normalized_energy_per_work_unit_est_j", "host_normalized_energy_per_work_unit_est_j"),
        "host_normalized_energy_per_work_est_j_sample_stddev": _repeat_stat(streaming or latency, ("host_normalized_energy_per_work_unit_est_j",), "sample_stddev"),
        "host_normalized_energy_per_work_est_j_ci_low": _repeat_stat(streaming or latency, ("host_normalized_energy_per_work_unit_est_j",), "ci_low"),
        "host_normalized_energy_per_work_est_j_ci_high": _repeat_stat(streaming or latency, ("host_normalized_energy_per_work_unit_est_j",), "ci_high"),
        "host_normalized_streaming_avg_power_est_w": _energy_phase_value_v59j(streaming, "avg_host_normalized_average_power_est_w", "host_normalized_average_power_est_w"),
        "energy_streaming_avg_power_w_sample_stddev": stream_power_std,
        "energy_streaming_avg_power_w_ci_low": stream_power_lo,
        "energy_streaming_avg_power_w_ci_high": stream_power_hi,
        "avg_power_w": stream_power if stream_power is not None else lat_power,
        "energy_work_units_per_j": _energy_phase_value_v59j(streaming, "avg_work_units_per_j", "work_units_per_j") or _energy_phase_value_v59j(latency, "avg_work_units_per_j", "work_units_per_j"),
    }
    normalization = streaming or latency
    for key in (
        "host_normalization_role", "host_normalization_source_run_id",
        "host_normalization_target_variant", "host_normalization_identity_verified",
        "accelerator_idle_correction_requested", "accelerator_idle_correction_applied",
        "accelerator_idle_correction_statuses", "accelerator_idle_w_applied",
        "accelerator_idle_calibration_verified", "accelerator_idle_calibration_status",
        "accelerator_idle_calibration_binding_path", "accelerator_idle_calibration_binding_sha256",
        "accelerator_idle_calibration_evidence", "accelerator_idle_calibrated_at",
        "energy_efficiency_claim_eligible",
    ):
        if normalization.get(key) is not None:
            payload[key] = normalization.get(key)
    payload.update(resolve_energy_comparison(payload))
    return {k: v for k, v in payload.items() if v not in (None, "", [], {})}


def _attach_energy_payload_to_row_v59j(row: dict[str, Any], payload: Mapping[str, Any], *, overwrite: bool = False) -> bool:
    changed = False
    path_like = {"energy_aggregate_path", "energy_aggregate_relpath"}
    provenance_like = {
        "energy_source",
        "energy_row_scope",
        "energy_row_level_source",
        "energy_merge_source",
        "energy_coverage_status",
        "energy_target_status",
        "energy_target_case",
        "energy_target_variant",
        "energy_applies_to_all_cases",
        "energy_target_id",
        "energy_target_ok",
        "energy_target_error_summary",
    }
    for k, v in payload.items():
        if v in (None, "", [], {}):
            continue
        cur = row.get(k)
        count_like = k in {"energy_target_window_count", "energy_target_valid_window_count", "energy_window_count", "energy_valid_window_count", "energy_target_phase_count"}
        should_overwrite = bool(overwrite)
        if k in path_like:
            cur_s = str(cur or "").replace("\\", "/")
            val_s = str(v or "").replace("\\", "/")
            if "/targets/" in val_s and "/targets/" not in cur_s:
                should_overwrite = True
        elif k in provenance_like:
            # Once a target aggregate matches a row, its provenance is more
            # specific than dispatch-only data copied from a validation report.
            should_overwrite = True
        elif count_like:
            # v59k: target aggregates are authoritative for their own window/phase
            # counts.  A previous reporting pass may have copied optimistic counts
            # from phase_repeat_count even when the replay target later failed; allow
            # a strict target payload to overwrite those stale values with 0.
            should_overwrite = True
        if should_overwrite or cur in (None, "", [], {}):
            if cur != v:
                row[k] = v
                changed = True
    pwr = _float_or_none(row.get("energy_streaming_avg_power_w") or row.get("avg_power_w"))
    fps = _selected_reference_fps_v59j(row)
    if pwr is not None and fps is not None and fps > 0:
        jpf = float(pwr) / float(fps)
        for k in ("energy_streaming_j_per_frame_from_selected_fps", "energy_j_per_frame_from_selected_fps"):
            if row.get(k) in (None, "") or overwrite:
                row[k] = jpf
                changed = True
        if row.get("energy_streaming_fps_per_watt_from_selected_fps") in (None, "") and pwr > 0:
            row["energy_streaming_fps_per_watt_from_selected_fps"] = float(fps) / float(pwr)
            changed = True
    return changed


def _target_aggregate_matches_row_v59j(row: Mapping[str, Any], agg: Mapping[str, Any]) -> bool:
    run_id = agg.get("run_id") or agg.get("backend") or ""
    if not _energy_backend_matches_row_v59j(row, run_id):
        return False
    tv = agg.get("energy_target_variant") or agg.get("target_variant")
    if not _energy_target_variant_matches_row_v59j(row, tv):
        return False
    tc = agg.get("energy_target_case") or agg.get("target_case")
    applies_all = bool(agg.get("energy_applies_to_all_cases"))
    if not _energy_target_case_matches_row_v59j(row, tc, applies_all):
        return False
    return True


def _augment_results_with_target_energy_v59j(results: Sequence[Mapping[str, Any]], *, model_id: str, run_root: Optional[Path]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    out = [dict(r or {}) for r in results]
    for row in out:
        _apply_validation_level_fields_v59j(row)
        if _row_has_row_level_energy_v59j(row):
            row.setdefault("energy_coverage_status", "row_level_energy_present")
            row.setdefault("energy_row_level_source", "benchmark_result_row_scope")
        elif row.get("energy_measurement_scope") not in (None, ""):
            row.setdefault("energy_coverage_status", "dispatch_only_no_row_energy")
            row.setdefault("energy_row_level_source", "")
    summary = {
        "schema": "onnx-splitpoint/energy-row-merge-summary",
        "schema_version": 1,
        "model_id": model_id,
        "status": "no_run_root",
        "target_aggregate_count": 0,
        "matched_target_aggregate_count": 0,
        "matched_row_count": 0,
        "split_row_count": len([r for r in out if str(r.get("variant") or "").lower() == "split"]),
        "split_row_with_energy_count": len([r for r in out if str(r.get("variant") or "").lower() == "split" and _row_has_row_level_energy_v59j(r)]),
        "missing_split_energy_count": 0,
        "unmatched_targets": [],
    }
    if run_root is None:
        return out, summary
    base = Path(run_root) / "models" / model_id / "benchmark_results" / "energy"
    if not base.is_dir():
        summary["status"] = "no_energy_target_directory"
    else:
        summary["status"] = "ok"
        for agg_path in sorted(base.glob("**/energy_aggregate.json")):
            # Only row-scope/target aggregates are candidates. Dispatch-level
            # aggregates stay diagnostic-only and must not be copied to rows.
            if any(part in {"latency", "streaming", "probe"} or part.startswith("run_") for part in agg_path.parts):
                continue
            agg = _read_json(agg_path, default={}) or {}
            if not isinstance(agg, Mapping):
                continue
            is_target = (
                "targets" in agg_path.parts
                or agg.get("energy_target_case") is not None
                or agg.get("target_case") is not None
                or agg.get("energy_target_variant") is not None
                or agg.get("target_variant") is not None
                or agg.get("target_id") is not None
            )
            if not is_target:
                continue
            summary["target_aggregate_count"] += 1
            matches = [row for row in out if _target_aggregate_matches_row_v59j(row, agg)]
            if not matches:
                try:
                    rel = str(agg_path.relative_to(run_root))
                except Exception:
                    rel = str(agg_path)
                summary.setdefault("unmatched_targets", []).append({
                    "run_id": str(agg.get("run_id") or ""),
                    "case_id": str(agg.get("energy_target_case") or ""),
                    "variant": str(agg.get("energy_target_variant") or ""),
                    "path": rel,
                })
                continue
            summary["matched_target_aggregate_count"] += 1
            payload = _energy_payload_from_target_aggregate_v59j(agg, agg_path, Path(run_root))
            for row in matches:
                # Do not override an already row-scoped benchmark-result energy
                # merge; fill missing aliases/provenance only.
                before = _row_has_row_level_energy_v59j(row)
                changed = _attach_energy_payload_to_row_v59j(row, payload, overwrite=False)
                if changed or not before:
                    summary["matched_row_count"] += 1
    for row in out:
        if _row_has_row_level_energy_v59j(row):
            row.setdefault("energy_coverage_status", "row_level_energy_present")
            row.setdefault("energy_row_level_source", "benchmark_result_row_scope")
        else:
            if str(row.get("variant") or "").lower() == "split":
                row.setdefault("energy_coverage_status", "missing_row_level_energy")
            else:
                row.setdefault("energy_coverage_status", "missing_or_not_measured")
        _apply_validation_level_fields_v59j(row)
    split_rows = [r for r in out if str(r.get("variant") or "").lower() == "split"]
    summary["split_row_count"] = len(split_rows)
    summary["split_row_with_energy_count"] = len([r for r in split_rows if _row_has_row_level_energy_v59j(r)])
    summary["missing_split_energy_count"] = len([r for r in split_rows if not _row_has_row_level_energy_v59j(r)])
    if summary["missing_split_energy_count"]:
        summary["status"] = "partial" if summary.get("target_aggregate_count") else summary.get("status", "missing")
        summary["missing_split_energy_examples"] = [
            {"backend": r.get("backend", ""), "case_id": r.get("case_id", ""), "variant": r.get("variant", "")}
            for r in split_rows if not _row_has_row_level_energy_v59j(r)
        ][:20]

    # v59o: policy-aware counters for thesis energy completeness.  The legacy
    # counters above intentionally describe *all* split rows.  Final-energy runs
    # can exclude CPU ORT and should only gate complete, valid split rows.
    profile_energy: Mapping[str, Any] = {}
    try:
        import yaml  # optional dependency in the workflow environment
        payload = yaml.safe_load((Path(run_root) / "profile.yaml").read_text(encoding="utf-8")) if run_root is not None and (Path(run_root) / "profile.yaml").is_file() else {}
        if isinstance(payload, Mapping) and isinstance(payload.get("energy"), Mapping):
            profile_energy = payload.get("energy") or {}
    except Exception:
        profile_energy = {}

    generic_not_requested = bool(
        profile_energy.get("generic_enabled") is False
        or (profile_energy.get("enabled") is False
            and profile_energy.get("measurement_path") not in {"generic", "native_and_generic"})
    )
    if generic_not_requested:
        for row in out:
            if not _row_has_row_level_energy_v59j(row):
                row["energy_coverage_status"] = "not_requested"
                row["energy_enabled"] = False
        summary["missing_split_energy_count"] = 0
        summary["missing_split_energy_examples"] = []
        summary["status"] = "not_requested"

    def _truthy_policy_v59o(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        return str(value).strip().lower() in {"1", "true", "yes", "y", "on", "strict"}

    final_all = _truthy_policy_v59o(profile_energy.get("final_all_split_energy") or profile_energy.get("all_split_energy") or profile_energy.get("require_complete_split_energy"))
    skip_cpu = False
    if final_all:
        include_cpu = any(_truthy_policy_v59o(profile_energy.get(k)) for k in ("include_cpu_ort_in_final_energy", "measure_cpu_ort_in_final_energy", "final_energy_include_cpu_ort"))
        if not include_cpu:
            if any(k in profile_energy for k in ("final_energy_skip_cpu_ort", "skip_cpu_ort_in_final_energy", "exclude_cpu_ort_in_final_energy")):
                skip_cpu = any(_truthy_policy_v59o(profile_energy.get(k)) for k in ("final_energy_skip_cpu_ort", "skip_cpu_ort_in_final_energy", "exclude_cpu_ort_in_final_energy"))
            else:
                skip_cpu = True
    excluded = {"cpu", "cpu_ort", "ort_cpu"} if skip_cpu else set()
    raw_skip = profile_energy.get("skip_backends")
    if isinstance(raw_skip, str):
        excluded.update(x.strip().lower().replace("-", "_") for x in raw_skip.split(",") if x.strip())
    elif isinstance(raw_skip, Sequence) and not isinstance(raw_skip, (str, bytes, bytearray)):
        excluded.update(str(x).strip().lower().replace("-", "_") for x in raw_skip if str(x).strip())

    def _row_backend_policy_id(row: Mapping[str, Any]) -> str:
        return str(row.get("backend") or row.get("run_id") or row.get("full_backend") or "").strip().lower().replace("-", "_").replace(" ", "_")

    def _row_final_pass_policy(row: Mapping[str, Any]) -> Optional[bool]:
        for key in ("final_pass_all", "final_pass", "validation_ok", "semantic_validation_ok", "semantic_validation_passed"):
            v = _bool_or_none(row.get(key))
            if v is not None:
                return v
        return None

    complete_policy_rows = [
        r for r in split_rows
        if _has_complete_split_latency_v58q(r)
        and _row_final_pass_policy(r) is not False
        and str(r.get("quality") or "").strip().lower() != "invalid"
        and _row_backend_policy_id(r) not in excluded
    ]
    excluded_complete_rows = [
        r for r in split_rows
        if _has_complete_split_latency_v58q(r)
        and _row_final_pass_policy(r) is not False
        and str(r.get("quality") or "").strip().lower() != "invalid"
        and _row_backend_policy_id(r) in excluded
    ]
    missing_complete_policy_rows = [r for r in complete_policy_rows if not _row_has_row_level_energy_v59j(r)]
    summary["schema_version"] = 2
    summary["energy_policy_final_all_split_energy"] = bool(final_all)
    summary["energy_policy_excluded_backend_ids"] = sorted(excluded)
    summary["complete_split_row_count_under_policy"] = len(complete_policy_rows)
    summary["complete_split_row_with_energy_count_under_policy"] = len([r for r in complete_policy_rows if _row_has_row_level_energy_v59j(r)])
    summary["complete_split_row_excluded_from_energy_count"] = len(excluded_complete_rows)
    summary["missing_complete_split_energy_count_under_policy"] = len(missing_complete_policy_rows)
    summary["missing_complete_split_energy_examples_under_policy"] = [
        {"backend": r.get("backend", ""), "case_id": r.get("case_id", ""), "variant": r.get("variant", ""), "coverage": r.get("energy_coverage_status", "")}
        for r in missing_complete_policy_rows
    ][:20]
    summary["status_under_energy_policy"] = "complete" if not missing_complete_policy_rows else "partial"
    if generic_not_requested:
        summary["missing_complete_split_energy_count_under_policy"] = 0
        summary["missing_complete_split_energy_examples_under_policy"] = []
        summary["status_under_energy_policy"] = "not_requested"
    if final_all and not missing_complete_policy_rows:
        summary["status"] = "complete_under_energy_policy"
    return out, summary

def _identity_setup_ids_v27550(row: Mapping[str, Any]) -> list[str]:
    """Resolve setup identities carried by exact quality-request records.

    Current benchmark rows can keep the physical setup only inside the
    per-variant quality input request. Losing that value before de-duplication
    collapses otherwise distinct setup-local observations. Inspect only the
    known identity containers and accept a nested value only when it resolves
    to one unambiguous setup.
    """

    setup_ids: set[str] = set()

    def add(value: Any) -> None:
        text = str(value or "").strip()
        if text:
            setup_ids.add(text)

    def add_mapping(mapping: Any) -> None:
        if not isinstance(mapping, Mapping):
            return
        for key in (
            "setup_id", "source_setup_id", "measurement_setup_id",
            "hardware_setup_id",
        ):
            add(mapping.get(key))
        for key in ("setup_ids", "quality_source_setup_ids"):
            values = mapping.get(key)
            if isinstance(values, Sequence) and not isinstance(
                values, (str, bytes, bytearray),
            ):
                for value in values:
                    add(value)

    add_mapping(row)
    add_mapping(row.get("producer_identity"))
    add_mapping(row.get("quality_contract"))

    identity_sets = row.get("quality_request_identities_by_variant")
    if isinstance(identity_sets, Mapping):
        for identity in identity_sets.values():
            add_mapping(identity)

    gates: list[Mapping[str, Any]] = []
    direct_gate = row.get("task_quality_gate")
    if isinstance(direct_gate, Mapping):
        gates.append(direct_gate)
    per_variant = row.get("task_quality_gates_by_variant")
    if isinstance(per_variant, Mapping):
        gates.extend(
            value for value in per_variant.values()
            if isinstance(value, Mapping)
        )
    for gate in gates:
        add_mapping(gate)
        quality_request = gate.get("quality_input_request")
        if not isinstance(quality_request, Mapping):
            continue
        add_mapping(quality_request)
        for key in (
            "producer_identity", "quality_contract", "request", "request_record",
        ):
            nested = quality_request.get(key)
            add_mapping(nested)
            if isinstance(nested, Mapping):
                add_mapping(nested.get("producer_identity"))
                add_mapping(nested.get("quality_contract"))
    return sorted(setup_ids, key=lambda value: value.lower())


def _resolved_setup_id_v27550(row: Mapping[str, Any]) -> str:
    variant = _dedupe_variant_key_v58g(row)
    embedded_values: set[str] = set()
    identities = row.get("quality_request_identities_by_variant")
    selected_identity = (
        identities.get(variant)
        if isinstance(identities, Mapping) and variant else None
    )
    if isinstance(selected_identity, Mapping):
        # A selected identity that already records drift is not a trustworthy
        # setup source.  Keep the observation fail-closed for later reporting.
        if selected_identity.get("identity_valid") is False or bool(
            selected_identity.get("identity_errors")
        ):
            return ""
        for value in (
            selected_identity.get("setup_id"),
            *list(selected_identity.get("setup_ids") or []),
        ):
            text = str(value or "").strip()
            if text:
                embedded_values.add(text)

    # During first-pass normalization the compact identity does not exist yet;
    # inspect only the exact selected request (split aliases composed), never
    # requests for Full or another component variant.
    for request in _quality_requests_for_variant_v269d(row, variant):
        blocks: list[Mapping[str, Any]] = [request]
        for key in (
            "request_payload", "request_record", "payload", "request",
            "producer_identity", "quality_contract",
        ):
            nested = request.get(key)
            if isinstance(nested, Mapping):
                blocks.append(nested)
        for block in blocks:
            for key in (
                "setup_id", "source_setup_id", "measurement_setup_id",
                "hardware_setup_id",
            ):
                text = str(block.get(key) or "").strip()
                if text:
                    embedded_values.add(text)

    explicit_values = {
        str(row.get(key) or "").strip()
        for key in (
            "setup_id", "source_setup_id", "measurement_setup_id",
            "hardware_setup_id",
        )
        if str(row.get(key) or "").strip()
    }
    if len(embedded_values) > 1 or len(explicit_values) > 1:
        return ""
    if embedded_values and explicit_values and embedded_values != explicit_values:
        return ""
    if len(embedded_values) == 1:
        return next(iter(embedded_values))
    if len(explicit_values) == 1:
        return next(iter(explicit_values))

    # Historical rows without a variant projection retain the old fallback,
    # but only when the complete evidence surface names exactly one setup.
    setup_ids = _identity_setup_ids_v27550(row)
    return setup_ids[0] if len(setup_ids) == 1 else ""


def _source_row_sha256_v2797(row: Mapping[str, Any]) -> str:
    """Hash the exact pre-projection row used by a normalized result.

    A Full measurement may physically live in a selected split-case container.
    ``source_case_id`` identifies that container; this digest prevents a timing
    from an unrelated row in the same file from satisfying the projection.
    """

    encoded = json.dumps(
        dict(row), sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _measurement_endpoint_for_variant_v27927(
    row: Mapping[str, Any], variant: str,
) -> str:
    """Read physical endpoint evidence for exactly one measured variant.

    Quality endpoint declarations and expected scope never describe a timed
    interval.  In particular a Full companion must not inherit its split's
    p2_output endpoint.  Conflicting producer declarations remain unbound.
    """
    variant = str(variant).strip().lower()
    if variant == "split":
        variant = "composed"
    selected = str(row.get("primary_variant") or _variant(row)).lower()
    if selected == "split":
        selected = "composed"
    values = [
        _mapping(row.get("measurement_endpoints_by_variant")).get(variant),
        _mapping(_mapping(row.get("timings")).get(variant)).get("measurement_endpoint"),
        row.get(f"{variant}_measurement_endpoint"),
    ]
    if variant == selected:
        values.append(row.get("measurement_endpoint"))
    explicit = {str(value).strip() for value in values if value not in (None, "")}
    return next(iter(explicit)) if len(explicit) == 1 else ""


def _generic_completion_projection(
    raw: Mapping[str, Any], normalized: Mapping[str, Any], completion: Mapping[str, Any],
) -> dict[str, Any]:
    """Project measured completion without overwriting contradictory evidence."""
    from ..runners.task_completion import completion_projection

    backend = str(normalized.get("backend") or "")
    producer = (
        "generic_deepx_full" if "deepx" in backend else
        "generic_hailo_full" if "hailo" in backend else "generic_ort_full"
    )
    projected = completion_projection(
        completion,
        task=str(normalized.get("task") or raw.get("benchmark_task_used") or ""),
        producer=producer,
    )
    # A report can contain both a split measurement and a Full baseline. Its
    # primary completion fields describe only the primary variant; explicit
    # Full deployment fields always describe the baseline.
    same_variant = _dedupe_variant_key_v58g(raw) == _dedupe_variant_key_v58g(normalized)
    declarations = [raw] if same_variant else []
    if str(normalized.get("variant") or "") == "full":
        declarations.append(_deployment_contract_for_variant(raw, "full"))
        alias = raw.get("generic_full_completion_evidence")
        if alias is not None and alias != completion:
            raise ValueError("generic_completion_evidence_alias_conflict")
        bound_hash = _mapping(raw.get("deployment_contract")).get("full_primary_host_tail_sha256")
        if bound_hash not in (None, "") and bound_hash != _mapping(completion.get("postprocess_contract")).get("contract_sha256"):
            raise ValueError("generic_completion_host_tail_binding_conflict")
    expected = {key: projected[key] for key in (
        "postprocess_included", "postprocess_completion_verified",
        "postprocess_completed_frames",
    )}
    # Legacy availability=False means N/A for classification and integrated
    # outputs. It is a conflict only where the producer requires a host tail.
    if (normalized.get("host_tail_required") is True
            or _mapping(completion.get("postprocess_contract")).get("source_contract_family") == "raw_head"):
        expected.update(host_tail_available=True, host_postprocessing_available=True)
    for declaration in declarations:
        for key, value in expected.items():
            declared = declaration.get(key)
            if declared in (None, ""):
                continue
            if type(declared) is not type(value) or declared != value:
                raise ValueError(f"generic_completion_projection_conflict:{key}")
    return projected


def _project_completion_or_record_error(raw, normalized, completion, *, record_errors):
    """Retain a rejected variant for diagnostics without admitting its metrics.

    Strict imports still raise. The workflow records the original cause on the
    affected variant so independent measurements and the raw source survive.
    """
    try:
        return _generic_completion_projection(raw, normalized, completion)
    except ValueError as exc:
        if not record_errors:
            raise
        return {
            "normalization_error": f"{type(exc).__name__}: {exc}",
            "error_class": "generic_completion_invalid",
            "runtime_executable": False,
            "structural_contract_pass": False,
            "structural_contract_reason": str(exc),
            "measurement_valid": False,
        }


def normalize_benchmark_row(row: Mapping[str, Any], *, model_id: str, source_path: Path, tag: str = "", record_completion_errors: bool = False) -> dict[str, Any]:
    endpoint_contract_complete_explicit = (
        row.get("endpoint_contract_complete_explicit")
        if isinstance(row.get("endpoint_contract_complete_explicit"), bool)
        else isinstance(row.get("endpoint_contract_complete"), bool)
    )
    part1 = _float_or_none(_first(row, ["part1_latency_ms", "part1_mean_ms", "throughput_stage1_mean_ms"]))
    part2 = _float_or_none(_first(row, ["part2_latency_ms", "part2_mean_ms", "throughput_stage2_mean_ms"]))
    composed = _float_or_none(_first(row, ["total_latency_ms", "latency_ms", "mean_ms", "avg_ms", "composed_mean_ms", "throughput_latency_mean_ms", "throughput_cycle_est_ms", "sum_parts_ms"]))
    pipe = _pipeline_block(row)
    # Prefer explicit pipeline single-sample latency for split rows when total_latency_ms is absent.
    if composed is None:
        composed = _pipeline_first(row, pipe, ["split_latency_e2e_ms"], ["single_sample_latency_ms", "sequential_composed_latency_ms"])
    full_total, full_raw, full_e2e, full_status = _best_full_latency(row)
    variant = _variant(row)
    source_case_id = _case_id(row)
    source_row_sha256 = _source_row_sha256_v2797(row)
    frozen_identity_evidence = _frozen_identity_evidence_v269d(row, variant)
    frozen_preprocessing_sha = _unique_frozen_identity_v269d(
        frozen_identity_evidence, "preprocessing_contract_sha256",
    )
    frozen_source_model_sha = _unique_frozen_identity_v269d(
        frozen_identity_evidence, "source_model_sha256",
    )
    frozen_runtime_artifact_sha = _unique_frozen_identity_v269d(
        frozen_identity_evidence, "runtime_artifact_sha256",
    )
    frozen_runtime_precision = _unique_frozen_identity_v269d(
        frozen_identity_evidence, "runtime_precision_identity",
    )
    selected_quality_gate = _select_task_quality_gate_v60i(row, str(row.get("primary_variant") or variant))
    embedded_quality_policy = _task_quality_policy_v60i(row, selected_quality_gate)

    # v49q/v51: do not backfill a split total from full-model latency. Legacy
    # BenchmarkSet rows often contain full TensorRT/CPU timings alongside a
    # requested Hailo->TRT composed run. When Hailo part1/composed was skipped,
    # using full_mean_ms as total_latency_ms would fabricate a split result.
    if variant == "full":
        total = full_total if full_total is not None else composed
    else:
        total = composed

    transfer = _float_or_none(_first(row, ["transfer_latency_ms", "overhead_ms", "cut_transfer_ms"]))
    transfer_source = "explicit" if transfer is not None else ""
    transfer_derived_invalid = False
    transfer_raw = transfer
    if transfer is None and total is not None and part1 is not None and part2 is not None:
        transfer_raw = total - part1 - part2
        transfer_source = "derived_total_minus_parts"
        if transfer_raw < -1e-6:
            # Negative transfer is a derived-metric inconsistency, not a physical
            # negative communication time. Keep the raw value for debugging but
            # clamp the normalized transfer to zero so downstream speedups do not
            # silently benefit from an impossible value.
            transfer_derived_invalid = True
            transfer = 0.0
        else:
            transfer = max(0.0, transfer_raw)

    measured_variants = list(row.get("measured_variants") or []) if isinstance(row.get("measured_variants"), list) else []
    skipped_variants = list(row.get("skipped_variants") or []) if isinstance(row.get("skipped_variants"), list) else []
    variant_status = dict(row.get("variant_status") or {}) if isinstance(row.get("variant_status"), Mapping) else {}
    variant_errors = dict(row.get("variant_errors") or {}) if isinstance(row.get("variant_errors"), Mapping) else {}
    component_status = "none"
    if variant == "full" and total is not None:
        component_status = full_status if full_status != "none" else "full"
    elif variant == "split" and composed is not None:
        component_status = "composed"
    elif variant == "split" and part1 is not None and part2 is not None:
        component_status = "split_parts"
    elif variant == "split" and part1 is not None:
        component_status = "part1_only"
    elif variant == "split" and part2 is not None:
        component_status = "part2_only"
    runtime_contract_decision = str(
        row.get("runtime_contract_decision") or ""
    ).strip().lower()
    if runtime_contract_decision == "pass":
        validation_ok = True
    elif runtime_contract_decision == "fail":
        validation_ok = False
    elif runtime_contract_decision == "inconclusive":
        validation_ok = None
    else:
        validation_ok = _bool_or_none(_first(row, [
            "validation_ok",
            "backend_drift_dataset_pass",
            "backend_drift_single_pass",
            "validation_dataset_primary_pass",
            "eps_pass",
        ]))
    runtime_ok = _bool_or_none(_first(row, ["runtime_ok", "ok", "eps_pass"]))
    if runtime_ok is None and total is not None:
        runtime_ok = True
    compile_ok = _bool_or_none(_first(row, ["compile_ok", "build_pass", "hailo_compile_ok", "part1_hailo_compile_ok", "part2_hailo_compile_ok"]))
    contract_variant = str(
        row.get("primary_variant")
        or ("full" if variant == "full" else "composed")
    ).strip().lower()
    scoped_deployment_contract = _deployment_contract_for_variant(row, contract_variant)
    endpoint_mode = str(
        scoped_deployment_contract.get("endpoint_mode")
        or _first(row, ["endpoint_mode", "full_endpoint_mode", "part2_endpoint_mode"])
        or _full_endpoint_mode(row)
        or "decoded_or_native"
    ).strip() or "decoded_or_native"
    stage1_backend = _stage_backend(row, "stage1", "stage1")
    stage2_backend = _stage_backend(row, "stage2", "stage2")
    full_backend = _full_backend(row, tag=tag)
    backend = _backend(row, tag=tag)
    source_run_id, reported_run_id = _logical_run_id_from_source_v269f(
        row, source_path, tag,
    )
    source_primary_variant = str(
        row.get("primary_variant") or row.get("variant") or variant
    ).strip().lower()
    if variant == "full" and full_backend:
        backend = full_backend

    final_pass = _bool_or_none(_first(row, ["final_pass", "final_pass_all", "passed", "eps_pass"]))
    final_pass_all = _bool_or_none(_first(row, ["final_pass_all", "final_pass", "passed", "eps_pass"]))
    semantic_validation_ok_value = _bool_or_none(_first(row, [
        "semantic_validation_ok",
        "semantic_validation_passed_all",
        "semantic_validation_passed",
        "semantic_pass",
        "semantic_passed",
    ]))
    semantic_validation_passed_value = _bool_or_none(_first(row, [
        "semantic_validation_passed_all",
        "semantic_validation_passed",
        "semantic_validation_ok",
        "semantic_pass",
        "semantic_passed",
    ]))
    if semantic_validation_ok_value is None and final_pass is True and validation_ok is True:
        semantic_validation_ok_value = True
    if semantic_validation_ok_value is None:
        for _sv_key in ("semantic_validation_passed_all", "semantic_validation_passed", "final_pass_all", "final_pass", "validation_ok"):
            _sv_val = _bool_or_none(row.get(_sv_key))
            if _sv_val is not None:
                semantic_validation_ok_value = _sv_val
                break
    if semantic_validation_passed_value is None:
        semantic_validation_passed_value = semantic_validation_ok_value
    # v55d: DeepX full semantic rows sometimes had runtime_ok+validation_ok
    # but lacked final_pass.  Treat validated full rows as final-pass rows.
    if final_pass is None and variant == "full" and backend.startswith("deepx") and runtime_ok is not False and validation_ok is True:
        final_pass = True
    if final_pass_all is None and variant == "full" and backend.startswith("deepx") and runtime_ok is not False and validation_ok is True:
        final_pass_all = True

    gate_block = _stage2_gate_block(row)
    dx_contract = _deepx_contract_block(row)

    split_latency_e2e = _pipeline_first(row, pipe, ["split_latency_e2e_ms"], ["single_sample_latency_ms", "sequential_composed_latency_ms"])
    if split_latency_e2e is None and variant == "split":
        split_latency_e2e = total
    pipeline_stage1_lane = _pipeline_first(row, pipe, ["pipeline_stage1_lane_ms"], ["stage1_ms", "stage1_lane_ms"])
    pipeline_stage2_lane = _pipeline_first(row, pipe, ["pipeline_stage2_lane_ms"], ["stage2_ms", "stage2_lane_ms"])
    pipeline_transfer = _pipeline_first(row, pipe, ["pipeline_transfer_est_ms"], ["transfer_or_mapping_ms", "transfer_ms"])
    pipeline_cycle_no_transfer = _pipeline_first(row, pipe, ["pipeline_cycle_no_transfer_ms"], ["optimistic_cycle_ms", "cycle_no_transfer_ms"])
    pipeline_cycle_with_transfer = _pipeline_first(row, pipe, ["pipeline_cycle_with_transfer_ms"], ["conservative_cycle_ms", "cycle_with_transfer_ms"])
    pipeline_cycle_selected = _pipeline_first(row, pipe, ["pipeline_cycle_selected_ms"], ["effective_cycle_ms", "measured_cycle_ms", "cycle_ms"])
    # v56q: never treat a backend tool's raw `fps` field as pipeline FPS.  For
    # DeepX run_model, `fps` may represent DXRT internal throughput and can be
    # much higher than 1000/full_latency_ms.  Pipeline-FPS is only valid when it
    # comes from explicit pipeline/throughput fields or can be derived from a
    # selected cycle time.  For full rows, the conservative steady-state cycle is
    # the full/e2e latency unless a real streaming cycle was measured.
    pipeline_fps_selected = _pipeline_first(row, pipe, ["pipeline_fps_selected"], ["effective_fps", "measured_fps"])
    if variant == "full" and pipeline_cycle_selected in (None, 0):
        pipeline_cycle_selected = full_total or total
    if pipeline_fps_selected is None and pipeline_cycle_selected not in (None, 0):
        try:
            pipeline_fps_selected = 1000.0 / float(pipeline_cycle_selected)
        except Exception:
            pipeline_fps_selected = None
    pipeline_speedup = _pipeline_first(row, pipe, ["pipeline_speedup_full_over_cycle"], ["speedup_full_over_cycle", "pipeline_speedup_full_over_cycle"])
    if pipeline_speedup is None and full_total not in (None, 0) and pipeline_cycle_selected not in (None, 0):
        try:
            pipeline_speedup = float(full_total) / float(pipeline_cycle_selected)
        except Exception:
            pipeline_speedup = None
    pipeline_latency_penalty = _pipeline_first(row, pipe, ["pipeline_latency_penalty_vs_full"], ["latency_penalty_vs_full", "pipeline_latency_penalty_vs_full"])
    if pipeline_latency_penalty is None and full_total not in (None, 0) and split_latency_e2e not in (None, 0):
        try:
            pipeline_latency_penalty = float(split_latency_e2e) / float(full_total)
        except Exception:
            pipeline_latency_penalty = None
    pipeline_cycle_source = str(row.get("pipeline_cycle_source") or pipe.get("mode") or ("measured" if pipe.get("measured_cycle_ms") else "estimated") if pipe else "")
    pipeline_note = str(row.get("pipeline_note") or pipe.get("note") or "")

    throughput_kind, pipeline_applicable, throughput_primary_metric = _throughput_taxonomy(
        variant=variant, backend=backend, stage1=stage1_backend, stage2=stage2_backend
    )
    # v58m: full-backend throughput must come from full latency, not from a
    # same-backend composed/pipeline diagnostic column that may coexist in mixed
    # BenchmarkSet rows.  This keeps TensorRT/DeepX full baselines comparable to
    # the repeated-feed full-backend benchmark semantics.
    _full_fps_from_latency = None
    if throughput_kind == "full_backend":
        _full_lat_for_fps = full_total if full_total not in (None, 0) else total
        if _full_lat_for_fps not in (None, 0):
            try:
                _full_fps_from_latency = 1000.0 / float(_full_lat_for_fps)
            except Exception:
                _full_fps_from_latency = None
    full_backend_throughput_fps = _full_fps_from_latency if throughput_kind == "full_backend" else None
    heterogeneous_pipeline_fps = pipeline_fps_selected if throughput_kind == "heterogeneous_pipeline" else None
    same_backend_composed_fps = pipeline_fps_selected if throughput_kind == "same_backend_split_diagnostic" else None
    backend_tool_fps = _float_or_none(_first(row, ["dxrt_tool_fps", "backend_tool_fps", "fps"]))
    if throughput_kind == "full_backend":
        throughput_primary_fps = full_backend_throughput_fps
        throughput_primary_source = "full_latency_fps"
    elif throughput_kind == "heterogeneous_pipeline":
        throughput_primary_fps = heterogeneous_pipeline_fps
        throughput_primary_source = "heterogeneous_pipeline_cycle_or_streaming"
    elif throughput_kind == "same_backend_split_diagnostic":
        throughput_primary_fps = same_backend_composed_fps
        throughput_primary_source = "same_backend_split_diagnostic"
    else:
        throughput_primary_fps = None
        throughput_primary_source = "diagnostic_or_component"

    out: dict[str, Any] = {
        "schema": "onnx-splitpoint/normalized-benchmark-result",
        "schema_version": 2,
        "model_id": model_id,
        "case_id": "full" if variant == "full" else source_case_id,
        "source_case_id": (
            source_case_id
            if variant == "full" and source_case_id.lower() not in {"", "full"}
            else ""
        ),
        "backend": backend,
        "variant": variant,
        "precision": str(_first(row, ["precision", "execution_precision", "trt_precision", "native_precision"]) or "").strip(),
        "execution_precision": _frozen_identity_or_legacy_v269d(
            frozen_identity_evidence,
            "runtime_precision_identity",
            _first(row, ["execution_precision", "runtime_precision_identity"]),
        ),
        "runtime_precision_identity": frozen_runtime_precision,
        # Keep the exact runtime-input and repetition identities in the
        # normalized measurement contract.  Scientific reports are a
        # projection of these rows; dropping either identity here would let a
        # report row for another numeric feed or repetition aggregate satisfy
        # the same model/backend/case identity.
        "runtime_numeric_input_identity": (
            dict(row.get("runtime_numeric_input_identity") or {})
            if isinstance(row.get("runtime_numeric_input_identity"), Mapping)
            else {}
        ),
        "runtime_numeric_input_sha256": str(
            row.get("runtime_numeric_input_sha256") or ""
        ).strip().lower().removeprefix("sha256:"),
        "repetition_index": _first(
            row,
            ("repetition_index", "process_local_repetition_index", "repeat_idx", "repeat_index"),
        ),
        "repetition_id": row.get("repetition_id"),
        "repetition_count_requested": _first(
            row,
            ("repetition_count_requested", "repetitions_requested", "performance_repeat_count_requested"),
        ),
        "repetition_count_attempted": _first(
            row,
            ("repetition_count_attempted", "repetitions_attempted", "performance_repeat_count_attempted"),
        ),
        "repetition_count_valid": _first(
            row,
            ("repetition_count_valid", "repetitions_completed", "performance_repeat_count_valid"),
        ),
        "repetition_status": row.get("repetition_status"),
        "repetition_aggregation": row.get("repetition_aggregation"),
        "repetition_runtime_scope": row.get("repetition_runtime_scope"),
        "repetition_independence_verified": row.get(
            "repetition_independence_verified"
        ),
        "preprocessing_contract_sha256": frozen_preprocessing_sha,
        "source_model_sha256": frozen_source_model_sha,
        "runtime_artifact_sha256": frozen_runtime_artifact_sha,
        "frozen_identity_evidence": frozen_identity_evidence,
        "setup_id": _resolved_setup_id_v27550(row),
        "comparison_backend": str(_first(row, ["comparison_backend"]) or "").strip(),
        "stage": str(_first(row, ["stage"]) or "").strip(),
        "endpoint_contract_complete": _first(row, ["endpoint_contract_complete"]) is True,
        # Preserve whether the source producer actually declared completeness.
        # The Boolean field above remains for compatibility, where a missing
        # value historically became False; the provenance bit prevents that
        # default from being mistaken for a contradictory attestation during
        # exact Generic/Native identity projection.
        "endpoint_contract_complete_explicit": (
            endpoint_contract_complete_explicit
        ),
        "endpoint_contract_hash": _frozen_identity_or_legacy_v269d(
            frozen_identity_evidence,
            "endpoint_contract_hash",
            _first(row, ["endpoint_contract_hash"]),
        ).lower(),
        "output_endpoint_attestation": dict(row.get("output_endpoint_attestation") or {}) if isinstance(row.get("output_endpoint_attestation"), Mapping) else {},
        "task": str(_first(row, ["task", "benchmark_task_used", "benchmark_task_requested", "validation_dataset_task"]) or selected_quality_gate.get("task") or "").strip().lower(),
        "benchmark_task_requested": _first(row, ["benchmark_task_requested"]),
        "benchmark_task_used": _first(row, ["benchmark_task_used"]),
        "validation_dataset_task": _first(row, ["validation_dataset_task"]),
        "task_quality_policy": embedded_quality_policy,
        "task_quality_policy_sha256": _first(row, ["task_quality_policy_sha256"]),
        "task_quality_gate": selected_quality_gate,
        "task_quality_gates_by_variant": dict(row.get("task_quality_gates_by_variant") or {}) if isinstance(row.get("task_quality_gates_by_variant"), Mapping) else {},
        "validation_cardinality_contract": dict(row.get("validation_cardinality_contract") or {}) if isinstance(row.get("validation_cardinality_contract"), Mapping) else {},
        "row_eligibility": dict(row.get("row_eligibility") or {}) if isinstance(row.get("row_eligibility"), Mapping) else {},
        "runtime_contract_decision": runtime_contract_decision,
        "runtime_contract_reason_codes": [
            str(value) for value in list(
                row.get("runtime_contract_reason_codes") or []
            ) if str(value)
        ],
        "quantized_interface_gate_decision": (
            dict(row.get("quantized_interface_gate_decision") or {})
            if isinstance(row.get("quantized_interface_gate_decision"), Mapping)
            else {}
        ),
        "native_stage2_binding_evidence": (
            dict(row.get("native_stage2_binding_evidence") or {})
            if isinstance(row.get("native_stage2_binding_evidence"), Mapping)
            else {}
        ),
        "part1_latency_ms": part1,
        "part2_latency_ms": part2,
        "transfer_latency_ms": transfer,
        "transfer_latency_raw_ms": transfer_raw,
        # v55s: this is a derived composition residual, not a measured
        # communication latency. Negative residuals are diagnostic only.
        "composition_residual_ms": transfer_raw if transfer_source == "derived_total_minus_parts" else None,
        "composition_residual_invalid": bool(transfer_derived_invalid),
        "transfer_latency_source": transfer_source,
        "transfer_latency_invalid": bool(transfer_derived_invalid),
        "derived_metric_invalid": bool(transfer_derived_invalid),
        "total_latency_ms": total,
        "split_latency_e2e_ms": split_latency_e2e,
        "pipeline_model": str(row.get("pipeline_model") or pipe.get("interleaving_model") or pipe.get("model") or "interleaved_two_stage" if (variant == "split" and (pipeline_cycle_selected is not None or pipeline_fps_selected is not None)) else ""),
        "pipeline_stage1_lane_ms": pipeline_stage1_lane,
        "pipeline_stage2_lane_ms": pipeline_stage2_lane,
        "pipeline_transfer_est_ms": pipeline_transfer,
        "pipeline_cycle_no_transfer_ms": pipeline_cycle_no_transfer,
        "pipeline_cycle_with_transfer_ms": pipeline_cycle_with_transfer,
        "pipeline_cycle_selected_ms": pipeline_cycle_selected,
        "pipeline_fps_selected": pipeline_fps_selected,
        "pipeline_speedup_full_over_cycle": pipeline_speedup,
        "pipeline_latency_penalty_vs_full": pipeline_latency_penalty,
        "pipeline_cycle_source": pipeline_cycle_source,
        "pipeline_note": pipeline_note,
        "throughput_kind": throughput_kind,
        "pipeline_applicable": bool(pipeline_applicable),
        "full_backend_throughput_fps": full_backend_throughput_fps,
        "heterogeneous_pipeline_fps": heterogeneous_pipeline_fps,
        "same_backend_composed_fps": same_backend_composed_fps,
        "backend_tool_fps": backend_tool_fps,
        "backend_tool_fps_semantics": _first(row, ["backend_tool_fps_semantics", "dxrt_tool_fps_semantics"]),
        "dxrt_tool_fps": _float_or_none(_first(row, ["dxrt_tool_fps", "run_model_fps", "fps"])),
        "dxrt_tool_fps_semantics": _first(row, ["dxrt_tool_fps_semantics"]),
        "performance_benchmark_source": _first(row, ["performance_benchmark_source", "benchmark_source"]),
        "prepared_feed_image_source": _first(row, ["prepared_feed_image_source"]),
        "prepared_feed_sequence_policy": _first(row, ["prepared_feed_sequence_policy", "benchmark_input_mode"]),
        "prepared_feed_unique_input_images": _float_or_none(_first(row, ["prepared_feed_unique_input_images", "benchmark_unique_dataset_images"])),
        "primary_latency_source": _first(row, ["performance_benchmark_source", "benchmark_source", "latency_source"]),
        "primary_latency_semantics": ("prepared_feed_dx_engine_latency" if str(_first(row, ["performance_benchmark_source", "benchmark_source"]) or "").strip() == "dx_engine_prepared_feed" else None),
        "throughput_primary_fps": throughput_primary_fps,
        "throughput_primary_metric": throughput_primary_metric,
        "throughput_primary_source": throughput_primary_source,
        "full_latency_ms": full_total,
        "full_raw_head_latency_ms": full_raw,
        "full_e2e_latency_ms": full_e2e,
        "diagnostic_full_e2e_latency_ms": (
            _float_or_none(_first(row, [
                "diagnostic_full_e2e_latency_ms", "full_e2e_latency_ms", "full_e2e_mean_ms",
            ]))
            if str(row.get("latency_semantics") or "") == "diagnostic_only"
            else _float_or_none(row.get("diagnostic_full_e2e_latency_ms"))
        ),
        "compile_ok": compile_ok,
        "runtime_ok": runtime_ok,
        "runner_returncode": row.get("runner_returncode", row.get("_runner_rc")),
        "runner_terminal_failure": row.get("runner_terminal_failure"),
        "runner_signal_name": row.get("runner_signal_name"),
        "validation_ok": validation_ok,
        "final_pass": final_pass,
        "final_pass_all": final_pass_all,
        "semantic_validation_ok": semantic_validation_ok_value,
        "semantic_validation_passed": semantic_validation_passed_value,
        "error_class": _error_class(row),
        "endpoint_mode": endpoint_mode,
        "measurement_endpoint": _measurement_endpoint_for_variant_v27927(
            row, contract_variant,
        ),
        "full_measurement_endpoint": _measurement_endpoint_for_variant_v27927(
            row, "full",
        ),
        "measurement_endpoints_by_variant": dict(
            _mapping(row.get("measurement_endpoints_by_variant"))
        ),
        "quality_endpoint": str(
            _first(row, ["quality_endpoint"]) or ""
        ).strip(),
        "source_path": str(source_path),
        "source_row_sha256": source_row_sha256,
        "source_tag": tag,
        # Stable producer identity for the central-quality join.  Backend is
        # not sufficient here: e.g. Hailo Full and Hailo-to-TRT Full can share
        # tokens while representing different quality requests.
        "run_id": source_run_id,
        "reported_run_id": reported_run_id,
        "component_measurement_status": component_status,
        "primary_variant": str(row.get("primary_variant") or ""),
        "full_source_run_id": source_run_id if variant == "full" else "",
        "full_source_primary_variant": source_primary_variant if variant == "full" else "",
        "full_source_stage1_provider": stage1_backend if variant == "full" else "",
        "full_source_stage2_provider": stage2_backend if variant == "full" else "",
        "full_source_owner_declared": bool(
            variant == "full"
            and row.get("full_baseline_owner") is True
            and row.get("full_measurement_requested_in_case") is True
        ),
        "full_source_owner_run_id": (
            str(row.get("full_baseline_owner_run_id") or "").strip().lower()
            if variant == "full" else ""
        ),
        "full_source_owner_backend": (
            str(row.get("full_baseline_owner_backend") or "").strip().lower()
            if variant == "full" else ""
        ),
        "measured_variants": measured_variants,
        "skipped_variants": skipped_variants,
        "variant_status": variant_status,
        "variant_errors": variant_errors,
        # A genuine Full measurement has no split boundary.  Legacy producers
        # sometimes retain the setup's stage labels on the Full row; preserve
        # their source case separately, but do not let those labels leak into
        # the canonical result identity or report.
        "stage1_provider": "" if variant == "full" else stage1_backend,
        "stage2_provider": "" if variant == "full" else stage2_backend,
        "full_provider": full_backend,
        "full_backend": full_backend,
        "full_latency_source": full_status,
        "full_host_tail_latency_ms": _float_or_none(_first(row, ["full_host_tail_latency_ms", "full_host_tail_mean_ms"])),
        "full_raw_accelerator_latency_ms": full_raw,
        "hailo_runtime_kind": ("full" if full_backend.startswith("hailo") and variant == "full" and total is not None else ("composed" if "hailo" in backend and variant == "split" and composed is not None else ("part1" if stage1_backend.startswith("hailo") and part1 is not None else ("part2" if stage2_backend.startswith("hailo") and part2 is not None else "none")))),
        "hailo_full_runtime_ok": bool(full_backend.startswith("hailo") and variant == "full" and total is not None and runtime_ok is not False),
        "hailo_composed_runtime_ok": bool("hailo" in backend and variant == "split" and composed is not None and runtime_ok is not False),
        "deepx_runtime_kind": ("full" if full_backend.startswith("deepx") and variant == "full" and total is not None else ("composed" if "deepx" in backend and variant == "split" and composed is not None else "none")),
        "deepx_full_runtime_ok": bool(full_backend.startswith("deepx") and variant == "full" and total is not None and runtime_ok is not False),
        "deepx_composed_runtime_ok": bool("deepx" in backend and variant == "split" and composed is not None and runtime_ok is not False),
        "stage2_accel_calibration_required": _bool_or_none(_first(row, ["stage2_accel_calibration_required"])) if _first(row, ["stage2_accel_calibration_required"]) not in (None, "") else gate_block.get("required"),
        "stage2_accel_calibration_available": _bool_or_none(_first(row, ["stage2_accel_calibration_available"])) if _first(row, ["stage2_accel_calibration_available"]) not in (None, "") else gate_block.get("available"),
        "stage2_accel_calibration_status": _first(row, ["stage2_accel_calibration_status"]) or gate_block.get("status"),
        "stage2_calibration_source": _first(row, ["stage2_calibration_source"]) or gate_block.get("source"),
        "stage2_calibration_trust_level": _first(row, ["stage2_calibration_trust_level"]) or gate_block.get("trust_level"),
        "stage2_calibration_producer_exact": _bool_or_none(_first(row, ["stage2_calibration_producer_exact"])) if _first(row, ["stage2_calibration_producer_exact"]) not in (None, "") else gate_block.get("producer_exact"),
        "stage2_calibration_sample_count": _float_or_none(_first(row, ["stage2_calibration_sample_count"])) or _float_or_none(gate_block.get("sample_count")),
        "deepx_stage2_contract_status": _first(row, ["deepx_stage2_contract_status"]) or dx_contract.get("status"),
        "deepx_stage2_contract_pass": _bool_or_none(_first(row, ["deepx_stage2_contract_pass"])) if _first(row, ["deepx_stage2_contract_pass"]) not in (None, "") else dx_contract.get("pass"),
        "deepx_stage2_contract_probe_samples": _float_or_none(_first(row, ["deepx_stage2_contract_probe_samples"])) or _float_or_none(dx_contract.get("probe_sample_count")),
        "deepx_stage2_contract_selected_pass_ratio": _float_or_none(_first(row, ["deepx_stage2_contract_selected_pass_ratio"])) or _float_or_none(dx_contract.get("selected_pass_ratio")),
        "deepx_stage2_contract_selected_mean_score": _float_or_none(_first(row, ["deepx_stage2_contract_selected_mean_score"])) or _float_or_none(dx_contract.get("selected_mean_score")),
        "deepx_stage2_contract_selected_top1_match_ratio": _float_or_none(_first(row, ["deepx_stage2_contract_selected_top1_match_ratio"])) or _float_or_none(dx_contract.get("selected_top1_match_ratio")),
        "deepx_stage2_contract_rejected_candidate_count": _float_or_none(_first(row, ["deepx_stage2_contract_rejected_candidate_count"])) or _float_or_none(dx_contract.get("rejected_candidate_count")),
        "raw_boundary": row.get("boundary"),
        "prediction_rank": row.get("prediction_rank") or row.get("rank") or row.get("source_rank"),
        "predicted_total_latency_ms": _float_or_none(_first(row, ["predicted_total_latency_ms", "pred_total_ms", "latency_pred_ms"])),
        "predicted_transfer_latency_ms": _float_or_none(_first(row, ["predicted_transfer_latency_ms", "pred_transfer_ms"])),
        "mini_coco_ap50_primary": _float_or_none(_first(row, ["mini_coco_ap50_primary", "map_light", "ap50", "coco_ap50", "validation_map_light"])),
        "classification_top1": _float_or_none(_first(row, ["classification_top1", "classification_top1_accuracy", "top1", "mini_classification_eval_primary_top1", "validation_dataset_primary_top1_accuracy"])),
        "classification_top5": _float_or_none(_first(row, ["classification_top5", "classification_top5_accuracy", "top5", "mini_classification_eval_primary_top5", "validation_dataset_primary_top5_accuracy"])),
        # v49i validation binding: keep numeric/task validation evidence in the
        # normalized contract so the validation stage does not have to parse
        # backend-specific result files or text logs.
        "max_abs_error": _float_or_none(_first(row, [
            "max_abs_error", "max_abs_diff", "max_absolute_error",
            "backend_drift_dataset_max_abs_error", "backend_drift_single_max_abs_error",
            "eps_max_abs", "output_max_abs_error",
        ])),
        "mean_abs_error": _float_or_none(_first(row, [
            "mean_abs_error", "mean_abs_diff", "mean_absolute_error",
            "backend_drift_dataset_mean_abs_error", "backend_drift_single_mean_abs_error",
            "eps_mean_abs", "output_mean_abs_error",
        ])),
        "cosine_similarity": _float_or_none(_first(row, [
            "cosine_similarity", "cosine", "cos_sim",
            "backend_drift_dataset_cosine_similarity", "backend_drift_single_cosine_similarity",
            "output_cosine_similarity",
        ])),
        "output_shape_match": _bool_or_none(_first(row, [
            "output_shape_match", "shape_match", "outputs_shape_match",
            "backend_drift_dataset_shape_match", "backend_drift_single_shape_match",
        ])),
        "detection_count": _float_or_none(_first(row, ["detection_count", "num_detections", "detections", "n_detections"])),
        "confidence_mean": _float_or_none(_first(row, ["confidence_mean", "mean_confidence", "avg_confidence"])),
        "raw_score_mean": None,
        "confidence_mean_clipped_0_1": None,
        "score_semantics": _first(row, ["score_semantics", "confidence_semantics"]),
        "detection_count_delta": _float_or_none(_first(row, ["detection_count_delta", "num_detections_delta", "detections_delta"])),
        "confidence_mean_abs_delta": _float_or_none(_first(row, ["confidence_mean_abs_delta", "confidence_delta", "score_mean_abs_delta"])),
        "raw_head_decode_ok": _bool_or_none(_first(row, ["raw_head_decode_ok", "yolo_raw_head_decode_ok", "host_tail_decode_ok"])),
        "raw_head_contract_present": _bool_or_none(
            scoped_deployment_contract.get("raw_head_contract_present")
            if scoped_deployment_contract else _first(row, ["raw_head_contract_present"])
        ),
        "raw_head_contract_status": (
            scoped_deployment_contract.get("raw_head_contract_status")
            if scoped_deployment_contract else _first(row, ["raw_head_contract_status"])
        ),
        "host_tail_required": _bool_or_none(
            scoped_deployment_contract.get("host_tail_required")
            if scoped_deployment_contract else _first(row, ["host_tail_required"])
        ),
        "host_tail_available": _bool_or_none(
            scoped_deployment_contract.get("host_tail_available")
            if scoped_deployment_contract else _first(row, ["host_tail_available"])
        ),
        "stage2_input_contract_kind": (
            scoped_deployment_contract.get("stage2_input_contract_kind")
            if scoped_deployment_contract else _first(row, ["stage2_input_contract_kind"])
        ),
        "stage2_feature_tensor_contract_status": _first(row, ["stage2_feature_tensor_contract_status", "stage2_accel_calibration_status"]),
        "deepx_stage1_semantic_override": _bool_or_none(_first(row, ["deepx_stage1_semantic_override"])),
        "semantic_validation_metric_gate": _first(row, ["semantic_validation_metric_gate"]),
        "nms_ok": _bool_or_none(row.get("nms_ok")),
    }
    completion = row.get("generic_completion_evidence") or (row.get("generic_full_completion_evidence") if str(out.get("variant") or "") == "full" else None)
    if completion:
        out.update(_project_completion_or_record_error(
            row, out, completion, record_errors=record_completion_errors))
    if row.get("generic_full_completion_evidence"):
        out["generic_full_completion_evidence"] = row["generic_full_completion_evidence"]
    # v59j: preserve interface/drift evidence before deriving validation claim labels.
    # Variant-aware exclusion reconciliation must see explicit starts and
    # completions even when the primary row reports only a component timing.
    from ..native_job_identity import generic_runtime_observation_fields
    out.update(generic_runtime_observation_fields(row))
    _copy_interface_fields_v59j(out, row)

    # v58i: keep row-level u.RECS energy metrics in normalized_results.
    _copy_energy_fields_v58i(out, row, normalized_variant=variant)

    # v58d: normalise detection confidence semantics.  Values outside [0,1]
    # are raw/unnormalised scores, not probabilities.  Keep the raw value and
    # provide a clipped diagnostic field, but let dashboards prefer the semantic
    # label instead of calling it calibrated confidence.
    try:
        cm = _float_or_none(out.get("confidence_mean"))
        if cm is not None:
            if 0.0 <= float(cm) <= 1.0:
                out["confidence_mean_clipped_0_1"] = float(cm)
                out["score_semantics"] = out.get("score_semantics") or "sigmoid_confidence"
            else:
                out["raw_score_mean"] = float(cm)
                out["confidence_mean_clipped_0_1"] = max(0.0, min(1.0, float(cm)))
                out["score_semantics"] = out.get("score_semantics") or "raw_or_unnormalized_score"
    except Exception:
        pass
    if variant == "full" and backend in {"tensorrt", "cpu_ort", "cuda_ort", "ort_tensorrt", "ort_cpu", "ort_cuda"} and runtime_ok is not False:
        out["interface_contract_pass"] = True
        out["contract_consistent"] = True
        out["contract_gate_reason"] = "full_backend_no_split_boundary"
    # Preserve the actual producer declaration for the same variant. It is
    # provenance, not an inferred complete endpoint or a successful quality gate.
    if _dedupe_variant_key_v58g(row) == _dedupe_variant_key_v58g(out):
        for key in ("candidate_execution_contract", "candidate_execution_contract_sha256"):
            if row.get(key) not in (None, "", {}):
                out[key] = row[key]
    # Project the exact per-variant quality request while the original runner
    # payload is still available.  The request contains large per-sample
    # records, so normalized benchmark rows intentionally keep only the compact
    # identity projection.  Performing this before de-duplication also prevents
    # two physical setups from collapsing merely because ``setup_id`` was
    # carried only by ``composed_request.json``.
    identity_source = dict(out)
    # Keep an explicitly declared setup long enough to compare it with the
    # selected embedded request.  ``_resolved_setup_id_v27550`` deliberately
    # returns empty on disagreement, but the compact identity must still record
    # that disagreement rather than losing its reason.
    for setup_key in (
        "setup_id", "source_setup_id", "measurement_setup_id",
        "hardware_setup_id",
    ):
        if row.get(setup_key) not in (None, ""):
            identity_source[setup_key] = row.get(setup_key)
    # ``normalize_benchmark_row`` historically materialises a missing endpoint
    # completeness field as False.  That compatibility default is not an
    # explicit declaration and must not conflict with a True value attested by
    # the exact request.
    if (
        endpoint_contract_complete_explicit
        and isinstance(row.get("endpoint_contract_complete"), bool)
    ):
        identity_source["endpoint_contract_complete"] = row.get(
            "endpoint_contract_complete"
        )
    else:
        identity_source.pop("endpoint_contract_complete", None)
    request_map = row.get("task_quality_input_requests_by_variant")
    if isinstance(request_map, Mapping):
        identity_source["task_quality_input_requests_by_variant"] = dict(
            request_map
        )
    _apply_quality_source_identity_v265(identity_source)
    for identity_key in (
        "quality_source_run_id",
        "quality_source_setup_ids",
        "quality_source_variant",
        "quality_request_identities_by_variant",
    ):
        if identity_key in identity_source:
            out[identity_key] = identity_source[identity_key]
    selected_identity = (
        identity_source.get("quality_request_identities_by_variant", {}).get(
            _dedupe_variant_key_v58g(identity_source)
        )
        if isinstance(
            identity_source.get("quality_request_identities_by_variant"),
            Mapping,
        ) else None
    )
    if isinstance(selected_identity, Mapping):
        out["quality_identity_valid"] = bool(
            selected_identity.get("identity_valid") is True
        )
        out["quality_identity_errors"] = list(
            selected_identity.get("identity_errors") or []
        )
        selected_setup = str(selected_identity.get("setup_id") or "").strip()
        if selected_identity.get("identity_valid") is True:
            if selected_setup:
                out["setup_id"] = selected_setup
            for projected_key in (
                "task", "stage", "producer_backend", "comparison_backend",
                "endpoint_contract_hash", "runtime_precision_identity",
                "runtime_numeric_input_identity",
                "runtime_numeric_input_sha256",
                "runtime_input_encoding_identity",
                "runtime_input_encoding_sha256",
                "runtime_numeric_input_identity_status",
            ):
                projected_value = selected_identity.get(projected_key)
                if projected_value not in (None, ""):
                    out[projected_key] = projected_value
            if isinstance(
                selected_identity.get("endpoint_contract_complete"), bool,
            ):
                out["endpoint_contract_complete"] = selected_identity.get(
                    "endpoint_contract_complete"
                )
            if isinstance(
                selected_identity.get("output_endpoint_attestation"), Mapping,
            ):
                out["output_endpoint_attestation"] = dict(
                    selected_identity.get("output_endpoint_attestation") or {}
                )
    # Evaluate structure after the existing exact per-variant identity join.
    # Otherwise its temporary absent-endpoint default becomes a persistent
    # structural failure even when that same runner's request attests it.
    _apply_validation_level_fields_v59j(out)
    try:
        apply_accuracy_gate_to_row(out, embedded_quality_policy or None)
    except Exception:
        pass
    return out


def _quality_evidence_only_payload(value: Mapping[str, Any]) -> bool:
    """Return True only for an explicitly non-performance quality artifact.

    Setup-local TensorRT Full quality companions intentionally execute the
    semantic dataset without creating a benchmark measurement.  Treat their
    explicit marker as a hard ingestion boundary even if a stale/malicious
    report also contains timing-shaped fields.  Embedded supplemental quality
    requests on a normal benchmark row do not set these top-level markers and
    therefore remain available for ordinary row enrichment.
    """

    run_cfg = value.get("run_cfg") if isinstance(value.get("run_cfg"), Mapping) else {}
    producer = (
        value.get("producer_identity")
        if isinstance(value.get("producer_identity"), Mapping) else {}
    )
    role = str(
        value.get("execution_role") or producer.get("execution_role") or ""
    ).strip().lower()
    explicit = bool(
        value.get("quality_evidence_only") is True
        or run_cfg.get("quality_evidence_only") is True
        or str(value.get("schema") or "").strip().lower()
        == "onnx-splitpoint/tensorrt-full-quality-evidence-report"
        or role == "full_quality_only"
    )
    if not explicit:
        return False
    performance_flag = value.get("performance_claims_emitted")
    if performance_flag is None:
        performance_flag = producer.get("performance_claims_emitted")
    return performance_flag is not True


def expand_normalized_benchmark_rows(row: Mapping[str, Any], *, model_id: str, source_path: Path, tag: str = "", record_completion_errors: bool = False) -> list[dict[str, Any]]:
    """Normalize one BenchmarkSet-suite row into one or more contract rows.

    Legacy validation reports can contain both a full-model timing and a split
    timing in the same JSON.  For thesis reports these are different facts: a
    full baseline row and a candidate split row.  v51 therefore materializes a
    companion ``variant=full`` row whenever full timing exists, while keeping the
    primary row for the requested candidate/split variant.
    """
    if _quality_evidence_only_payload(row):
        return []
    primary = normalize_benchmark_row(row, model_id=model_id, source_path=source_path, tag=tag, record_completion_errors=record_completion_errors)
    rows: list[dict[str, Any]] = [primary]
    full_total, full_raw, full_e2e, full_status = _best_full_latency(row)
    if full_total is None:
        return rows

    full_backend = _full_backend(row, tag=tag)
    full_source_case_id = str(
        primary.get("source_case_id") or primary.get("case_id") or _case_id(row)
    ).strip().lower()
    full_identity_evidence = _frozen_identity_evidence_v269d(row, "full")
    full_preprocessing_sha = _unique_frozen_identity_v269d(
        full_identity_evidence, "preprocessing_contract_sha256",
    )
    full_endpoint_sha = _unique_frozen_identity_v269d(
        full_identity_evidence, "endpoint_contract_hash",
    )
    full_source_model_sha = _unique_frozen_identity_v269d(
        full_identity_evidence, "source_model_sha256",
    )
    full_runtime_artifact_sha = _unique_frozen_identity_v269d(
        full_identity_evidence, "runtime_artifact_sha256",
    )
    full_runtime_precision = _unique_frozen_identity_v269d(
        full_identity_evidence, "runtime_precision_identity",
    )
    raw_primary_variant = str(
        row.get("primary_variant") or row.get("variant") or ""
    ).strip().lower()
    full_row = dict(primary)
    if raw_primary_variant != "full" and primary.get("normalization_error"):
        # A rejected primary completion is not evidence about the independent
        # Full timer. Its own declarations are checked below against raw input.
        for field in ("normalization_error", "runtime_executable", "measurement_valid"):
            full_row.pop(field, None)
    if raw_primary_variant != "full":
        # These are derived for the primary split, not evidence about Full.
        for field in ("structural_contract_pass", "structural_contract_status", "structural_contract_reason"):
            full_row.pop(field, None)
    full_row.update({
        "case_id": "full",
        "source_case_id": (
            full_source_case_id if full_source_case_id not in {"", "full"} else ""
        ),
        "source_case_row_sha256": str(
            primary.get("source_row_sha256") or ""
        ),
        "full_source_run_id": str(primary.get("run_id") or tag).strip(),
        "full_source_primary_variant": raw_primary_variant,
        "full_source_stage1_provider": _stage_backend(row, "stage1", "stage1"),
        "full_source_stage2_provider": _stage_backend(row, "stage2", "stage2"),
        "full_source_owner_declared": bool(
            row.get("full_baseline_owner") is True
            and row.get("full_measurement_requested_in_case") is True
        ),
        "full_source_owner_run_id": str(
            row.get("full_baseline_owner_run_id") or ""
        ).strip().lower(),
        "full_source_owner_backend": str(
            row.get("full_baseline_owner_backend") or ""
        ).strip().lower(),
        "backend": full_backend,
        "variant": "full",
        "precision": str(
            row.get("full_precision")
            or (row.get("precision") if raw_primary_variant == "full" else "")
            or ""
        ).strip(),
        "execution_precision": full_runtime_precision,
        "runtime_precision_identity": full_runtime_precision,
        "preprocessing_contract_sha256": full_preprocessing_sha,
        "endpoint_contract_hash": full_endpoint_sha,
        "source_model_sha256": full_source_model_sha,
        "runtime_artifact_sha256": full_runtime_artifact_sha,
        "frozen_identity_evidence": full_identity_evidence,
        "endpoint_contract_complete": bool(
            row.get("full_endpoint_contract_complete") is True
            or (raw_primary_variant == "full" and row.get("endpoint_contract_complete") is True)
        ),
        "output_endpoint_attestation": (
            dict(row.get("full_output_endpoint_attestation") or {})
            if isinstance(row.get("full_output_endpoint_attestation"), Mapping)
            else dict(row.get("output_endpoint_attestation") or {})
            if raw_primary_variant == "full" and isinstance(row.get("output_endpoint_attestation"), Mapping)
            else {}
        ),
        "stage1_provider": "",
        "stage2_provider": "",
        "full_provider": full_backend,
        "full_backend": full_backend,
        "full_latency_source": full_status,
        "full_host_tail_latency_ms": _float_or_none(_first(row, ["full_host_tail_latency_ms", "full_host_tail_mean_ms"])),
        "full_raw_accelerator_latency_ms": full_raw,
        "hailo_runtime_kind": "full" if full_backend.startswith("hailo") and full_total is not None else "none",
        "hailo_full_runtime_ok": bool(full_backend.startswith("hailo") and full_total is not None and _variant_status_ok(row, "full")),
        "deepx_runtime_kind": "full" if full_backend.startswith("deepx") and full_total is not None else "none",
        "deepx_full_runtime_ok": bool(full_backend.startswith("deepx") and full_total is not None and _variant_status_ok(row, "full")),
        "deepx_composed_runtime_ok": False,
        "hailo_composed_runtime_ok": False,
        "part1_latency_ms": None,
        "part2_latency_ms": None,
        "transfer_latency_ms": None,
        "total_latency_ms": full_total,
        "full_latency_ms": full_total,
        "full_raw_head_latency_ms": full_raw,
        "full_e2e_latency_ms": full_e2e,
        "component_measurement_status": full_status if full_status != "none" else "full",
        "endpoint_mode": _full_endpoint_mode(row),
        "measurement_endpoint": _measurement_endpoint_for_variant_v27927(row, "full"),
        "setup_id": _resolved_setup_id_v27550({
            **dict(row), "variant": "full", "primary_variant": "full",
        }),
        "runtime_ok": True if full_total is not None and _variant_status_ok(row, "full") else primary.get("runtime_ok"),
        "validation_ok": _validation_ok_for_variant(row, "full"),
        "final_pass": (_validation_ok_for_variant(row, "full") if _validation_ok_for_variant(row, "full") is not None else (True if _variant_status_ok(row, "full") else None)),
        "final_pass_all": (_validation_ok_for_variant(row, "full") if _validation_ok_for_variant(row, "full") is not None else (True if _variant_status_ok(row, "full") else None)),
        "error_class": "" if _variant_status_ok(row, "full") else primary.get("error_class", ""),
        "throughput_kind": "full_backend",
        "pipeline_applicable": False,
        "pipeline_model": "full_backend_repeated_feed",
        "pipeline_applicable": False,
        # Compatibility: older report code looked at pipeline_* even for full rows.
        # For full companion rows this is explicitly full-backend throughput, not a
        # heterogeneous pipeline.  The taxonomy fields below are authoritative.
        "pipeline_cycle_selected_ms": full_total,
        "pipeline_fps_selected": (1000.0 / float(full_total)) if full_total not in (None, 0) else None,
        "pipeline_cycle_source": "full_latency_fps_compat",
        "pipeline_note": "Full backend repeated-feed throughput; not a heterogeneous split pipeline.",
        "full_backend_throughput_fps": (1000.0 / float(full_total)) if full_total not in (None, 0) else None,
        "heterogeneous_pipeline_fps": None,
        "same_backend_composed_fps": None,
        "backend_tool_fps": _float_or_none(_first(row, ["dxrt_tool_fps", "backend_tool_fps", "fps"])),
        "throughput_primary_fps": (1000.0 / float(full_total)) if full_total not in (None, 0) else None,
        "throughput_primary_metric": "full_backend_throughput_fps",
        "throughput_primary_source": "full_latency_fps",
        "measurement_note": "companion full-baseline row extracted from a BenchmarkSet validation report",
    })
    full_deployment_contract = _deployment_contract_for_variant(row, "full")
    if full_deployment_contract:
        full_row.update({
            "endpoint_mode": str(
                full_deployment_contract.get("endpoint_mode")
                or full_deployment_contract.get("full_endpoint_mode")
                or "decoded_or_native"
            ),
            "raw_head_contract_present": _bool_or_none(
                full_deployment_contract.get("raw_head_contract_present")
            ),
            "raw_head_contract_status": full_deployment_contract.get("raw_head_contract_status"),
            "host_tail_required": _bool_or_none(full_deployment_contract.get("host_tail_required")),
            "host_tail_available": _bool_or_none(full_deployment_contract.get("host_tail_available")),
            "stage2_input_contract_kind": full_deployment_contract.get("stage2_input_contract_kind"),
        })
    full_has_raw_head = _bool_or_none(full_row.get("raw_head_contract_present")) is True
    # v60o: full companion rows must not inherit any split/pipeline timing or
    # boundary-contract fields from the primary candidate row.  Those stale
    # fields previously made TensorRT Full display the split latency and a false
    # contract failure.
    for _split_only_key in (
        "pipeline_stage1_lane_ms", "pipeline_stage2_lane_ms",
        "pipeline_transfer_est_ms", "pipeline_cycle_no_transfer_ms",
        "pipeline_cycle_with_transfer_ms", "pipeline_overlap_efficiency",
        "pipeline_latency_penalty_ms", "pipeline_residual_overhead_ms",
        "pipeline_residual_overhead_percent", "split_latency_e2e_ms",
        "composed_latency_ms", "composed_mean_ms", "handover_latency_ms",
        "boundary_transfer_latency_ms", "stage1_output_bytes",
        "interface_stage1_max_abs", "interface_stage1_mean_abs_mean",
        "interface_stage1_num_compared_tensors", "strict_boundary_numeric_pass",
        "strict_boundary_numeric_status", "raw_head_drift_pass",
        "raw_head_drift_status", "raw_head_drift_max_abs",
        "raw_head_drift_mean_abs_mean",
    ):
        full_row[_split_only_key] = None
    full_row["latency_ms"] = full_total
    full_row["latency_source"] = full_status or "full_timing"
    if not full_has_raw_head:
        full_row["interface_check_pass"] = True
        full_row["interface_check_status"] = "not_applicable_full_backend"
        full_row["interface_contract_pass"] = True
        full_row["interface_contract_status"] = "not_applicable_full_backend"
        full_row["contract_consistent"] = True
        full_row["contract_gate_reason"] = "full_backend_no_split_boundary"
    if not full_has_raw_head and full_backend in {"tensorrt", "cpu_ort", "cuda_ort", "ort_tensorrt", "ort_cpu", "ort_cuda"} and full_row.get("runtime_ok") is not False:
        # A full GPU/ORT baseline has no split-boundary handoff contract.  Its
        # admissibility is governed by runtime and dataset task-quality gates.
        full_row["interface_contract_pass"] = True
        full_row["contract_consistent"] = True
        full_row["contract_gate_reason"] = "full_backend_no_split_boundary"
    # v58i/v59j: copy energy only when the raw row actually targets the full variant.
    # full_row starts as a copy of primary, so strip split-target energy before
    # adding full-target context to avoid advertising composed Energy as full Energy.
    if not _energy_target_matches_variant_v58i(row, "full"):
        _strip_energy_fields_v59j(full_row)
    _copy_energy_fields_v58i(full_row, row, normalized_variant="full")
    full_quality_gate = _select_task_quality_gate_v60i(row, "full")
    full_quality_policy = _task_quality_policy_v60i(row, full_quality_gate)
    full_row["task_quality_gate"] = full_quality_gate
    full_row["task_quality_policy"] = full_quality_policy
    full_row["task_quality_gates_by_variant"] = dict(row.get("task_quality_gates_by_variant") or {}) if isinstance(row.get("task_quality_gates_by_variant"), Mapping) else {}
    full_row["validation_cardinality_contract"] = dict(row.get("validation_cardinality_contract") or {}) if isinstance(row.get("validation_cardinality_contract"), Mapping) else {}
    full_row["row_eligibility"] = dict(row.get("row_eligibility") or {}) if isinstance(row.get("row_eligibility"), Mapping) else {}
    if full_quality_gate.get("task"):
        full_row["task"] = str(full_quality_gate.get("task") or "").lower()
    _apply_validation_level_fields_v59j(full_row)
    completion = row.get("generic_full_completion_evidence") or row.get("generic_completion_evidence")
    if completion:
        full_row.update(_project_completion_or_record_error(
            row, full_row, completion, record_errors=record_completion_errors))
    try:
        apply_accuracy_gate_to_row(full_row, full_quality_policy or None)
    except Exception:
        pass
    # Accuracy gating must not reintroduce a split-boundary requirement for a
    # decoded full backend row. A genuine raw-head Full row keeps the decoder/
    # host-tail contract verdict produced by the accuracy gate.
    if not full_has_raw_head:
        full_row["interface_check_pass"] = True
        full_row["interface_check_status"] = "not_applicable_full_backend"
        full_row["interface_contract_pass"] = True
        full_row["interface_contract_status"] = "not_applicable_full_backend"
        full_row["contract_consistent"] = True
        full_row["contract_gate_reason"] = "full_backend_no_split_boundary"

    # Do not duplicate a row that was already primarily a full baseline with the
    # same backend/latency.
    if not (
        str(primary.get("variant") or "").lower() == "full"
        and str(primary.get("backend") or "") == str(full_row.get("backend") or "")
        and primary.get("total_latency_ms") == full_row.get("total_latency_ms")
    ):
        rows.append(full_row)
    return rows



# v58f: Compact per-case validation_report.json files in remote diagnostics are
# enrichment/fallback result sources.  Flatten the essential timings/status and
# validation metrics into the same shape as benchmark_results rows; source-aware
# dedupe later merges them into primary rows when present.
def _validation_report_to_row(report: Mapping[str, Any], path: Path) -> dict[str, Any]:
    row: dict[str, Any] = {}
    run_cfg = report.get("run_cfg") if isinstance(report.get("run_cfg"), Mapping) else {}
    timings = report.get("timings") if isinstance(report.get("timings"), Mapping) else {}
    variant_status = report.get("variant_status") if isinstance(report.get("variant_status"), Mapping) else {}
    case = str(report.get("case_id") or report.get("case_dir") or path.parent.parent.name if path.parent.name.startswith("results_") else report.get("case_id") or "").strip()
    if case and not case.startswith("b"):
        # common path layout: .../b011/results_backend/validation_report.json
        for part in path.parts:
            if part.startswith("b") and part[1:].isdigit():
                case = part
    row["case_id"] = case
    backend = str(run_cfg.get("provider") or "").strip()
    if not backend and path.parent.name.startswith("results_"):
        backend = path.parent.name.replace("results_", "", 1)
    row["provider"] = backend
    row["backend"] = backend
    row["run_id"] = backend
    row["stage1_provider"] = run_cfg.get("stage1_provider") or run_cfg.get("stage1_backend") or ""
    row["stage2_provider"] = run_cfg.get("stage2_provider") or run_cfg.get("stage2_backend") or ""
    row["full_provider"] = run_cfg.get("full_provider") or run_cfg.get("provider") or backend
    primary_variant = str(report.get("primary_variant") or ("composed" if "composed" in timings else "full")).strip().lower()
    row["primary_variant"] = primary_variant
    row["variant"] = "split" if primary_variant in {"composed", "part1", "part2"} else "full"
    for key in (
        "setup_id", "source_setup_id", "measurement_setup_id", "hardware_setup_id",
        "measurement_endpoint", "full_measurement_endpoint", "quality_endpoint",
        "measurement_endpoints_by_variant", "task_quality_input_requests_by_variant",
    ):
        if report.get(key) not in (None, ""):
            row[key] = report[key]
    # The per-variant timings carry their own endpoint proof and must survive
    # the compact validation-report ingestion path unchanged.
    row["timings"] = dict(timings)
    for vname, prefix in (("full", "full"), ("part1", "part1"), ("part2", "part2"), ("composed", "composed")):
        blk = timings.get(vname) if isinstance(timings.get(vname), Mapping) else {}
        mean = _float_or_none(blk.get("mean_ms") or blk.get("latency_ms"))
        if mean is None:
            continue
        if vname == "full":
            row.setdefault("full_mean_ms", mean)
            row.setdefault("full_latency_ms", mean)
            row.setdefault("total_latency_ms", mean if primary_variant == "full" else row.get("total_latency_ms"))
        elif vname == "composed":
            row.setdefault("composed_mean_ms", mean)
            row.setdefault("total_latency_ms", mean)
        else:
            row.setdefault(f"{prefix}_mean_ms", mean)
            row.setdefault(f"{prefix}_latency_ms", mean)
    # Pipeline timing model, if present.
    pipe = report.get("pipeline_timing_model") if isinstance(report.get("pipeline_timing_model"), Mapping) else {}
    for dst, keys in {
        "pipeline_cycle_selected_ms": ("effective_cycle_ms", "measured_cycle_ms", "cycle_ms"),
        "pipeline_fps_selected": ("effective_fps", "measured_fps", "fps"),
        "pipeline_stage1_lane_ms": ("stage1_lane_ms", "stage1_ms"),
        "pipeline_stage2_lane_ms": ("stage2_lane_ms", "stage2_ms"),
        "pipeline_transfer_est_ms": ("transfer_ms", "transfer_or_mapping_ms"),
    }.items():
        for k in keys:
            val = _float_or_none(pipe.get(k))
            if val is not None:
                row[dst] = val
                break
    # Task validation metrics.
    mini_cls = report.get("mini_classification_eval") if isinstance(report.get("mini_classification_eval"), Mapping) else {}
    cls_vars = mini_cls.get("variants") if isinstance(mini_cls.get("variants"), Mapping) else {}
    cls_blk = cls_vars.get(primary_variant) if isinstance(cls_vars.get(primary_variant), Mapping) else (cls_vars.get("composed") if isinstance(cls_vars.get("composed"), Mapping) else (cls_vars.get("full") if isinstance(cls_vars.get("full"), Mapping) else {}))
    if cls_blk:
        row["classification_top1"] = cls_blk.get("top1_accuracy")
        row["classification_top5"] = cls_blk.get("top5_accuracy")
        row["classification_top1_agreement"] = cls_blk.get("top1_agreement")
        row["classification_top5_agreement"] = cls_blk.get("top5_agreement")
    mini_coco = report.get("mini_coco_ap50") if isinstance(report.get("mini_coco_ap50"), Mapping) else {}
    if mini_coco:
        row["mini_coco_ap50_primary"] = mini_coco.get("ap50") or mini_coco.get("map_light")
        row["detection_count"] = mini_coco.get("detection_count") or mini_coco.get("detection_count_mean")
        row["confidence_mean"] = mini_coco.get("confidence_mean") or mini_coco.get("mean_confidence")
    # v60i: retain the exact runtime task, gate, policy and row-eligibility
    # structures.  Scientific reporting compares their policy hash with the
    # configured profile instead of treating them as missing aggregate metrics.
    for _k in ("benchmark_task_requested", "benchmark_task_used", "validation_dataset_task", "task_quality_policy_sha256"):
        if report.get(_k) not in (None, ""):
            row[_k] = report.get(_k)
    row["task"] = str(
        report.get("benchmark_task_used")
        or report.get("benchmark_task_requested")
        or report.get("validation_dataset_task")
        or run_cfg.get("benchmark_task")
        or run_cfg.get("task")
        or ""
    ).strip().lower()
    if isinstance(report.get("task_quality_policy"), Mapping):
        row["task_quality_policy"] = dict(report.get("task_quality_policy") or {})
    if isinstance(report.get("task_quality_gate"), Mapping):
        row["task_quality_gate"] = dict(report.get("task_quality_gate") or {})
    if isinstance(report.get("task_quality_gates_by_variant"), Mapping):
        row["task_quality_gates_by_variant"] = dict(report.get("task_quality_gates_by_variant") or {})
    if isinstance(report.get("validation_cardinality_contract"), Mapping):
        row["validation_cardinality_contract"] = dict(report.get("validation_cardinality_contract") or {})
    if isinstance(report.get("row_eligibility"), Mapping):
        row["row_eligibility"] = dict(report.get("row_eligibility") or {})
    for _contract_key in (
        "deployment_contract", "deployment_contract_summary",
        "deployment_contracts_by_variant",
    ):
        if isinstance(report.get(_contract_key), Mapping):
            row[_contract_key] = dict(report.get(_contract_key) or {})
    scoped_report_contract = _deployment_contract_for_variant(row, primary_variant)
    if scoped_report_contract:
        row["endpoint_mode"] = scoped_report_contract.get("endpoint_mode")
        for _field in (
            "raw_head_contract_present", "raw_head_contract_status",
            "host_tail_required", "host_tail_available",
            "stage2_input_contract_kind",
        ):
            if _field in scoped_report_contract:
                row[_field] = scoped_report_contract.get(_field)
    selected_gate = _select_task_quality_gate_v60i(row, primary_variant)
    if selected_gate:
        row["task_quality_gate"] = selected_gate
        if not row.get("task") and selected_gate.get("task"):
            row["task"] = str(selected_gate.get("task") or "").lower()
        if not row.get("task_quality_policy") and isinstance(selected_gate.get("policy"), Mapping):
            row["task_quality_policy"] = dict(selected_gate.get("policy") or {})

    # Status evidence.  Newer runners keep runtime/contract admissibility
    # tri-state so deferred semantic evidence is not flattened to a false
    # validation result.  Prefer that explicit decision over legacy booleans.
    row["runtime_ok"] = True if any(isinstance(timings.get(k), Mapping) for k in timings.keys()) else None
    runtime_contract_decision = str(
        report.get("runtime_contract_decision") or ""
    ).strip().lower()
    if runtime_contract_decision == "pass":
        row["validation_ok"] = True
    elif runtime_contract_decision == "fail":
        row["validation_ok"] = False
    elif runtime_contract_decision == "inconclusive":
        row["validation_ok"] = None
    else:
        row["validation_ok"] = report.get("final_pass_all") if report.get("final_pass_all") is not None else report.get("final_pass")
    if runtime_contract_decision:
        row["runtime_contract_decision"] = runtime_contract_decision
        row["runtime_contract_reason_codes"] = [
            str(value) for value in list(
                report.get("runtime_contract_reason_codes") or []
            ) if str(value)
        ]
    row["semantic_validation_ok"] = report.get("semantic_validation_passed_all") if report.get("semantic_validation_passed_all") is not None else report.get("semantic_validation_passed")
    row["semantic_validation_passed"] = row.get("semantic_validation_ok")
    row["final_pass"] = report.get("final_pass")
    row["final_pass_all"] = report.get("final_pass_all")
    # v59j: keep backend interface/drift evidence from compact validation reports
    # so normalized reporting can distinguish strict numeric pass from semantic
    # pass with an interface warning/unavailable boundary dump.
    for _k in ("interface_checks", "interface_check", "split_interface_checks", "raw_head_drift"):
        if isinstance(report.get(_k), Mapping):
            row[_k] = report.get(_k)
    for _k in (
        "quantized_interface_gate_decision",
        "native_stage2_binding_evidence",
    ):
        if isinstance(report.get(_k), Mapping):
            row[_k] = dict(report.get(_k) or {})
    for _k in (
        "interface_check_status", "interface_check_pass", "interface_stage1_status", "interface_stage1_pass",
        "interface_stage1_max_abs", "interface_stage1_mean_abs_mean", "interface_stage1_num_compared_tensors",
        "raw_head_drift_status", "raw_head_drift_pass", "raw_head_drift_max_abs", "raw_head_drift_mean_abs_mean",
    ):
        if report.get(_k) not in (None, ""):
            row[_k] = report.get(_k)
    row["source_validation_report"] = True
    row["source_path"] = str(path)
    row["variant_status"] = variant_status
    return row


def _rows_from_json(path: Path, *, diagnostic_summary: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    # Only the production ingestion caller opts into companion publication.
    # Historical/read-only consumers must never mutate their source run.
    observed_before = None
    if diagnostic_summary is not None:
        from .compact_runtime_diagnostics import source_observation, write_runtime_companion
        try:
            observed_before = source_observation(path)
        except OSError:
            pass
    payload = _read_json(path, default=None)
    if diagnostic_summary is not None and payload is not None:
        diagnostic_summary.update(write_runtime_companion(path, payload, observed_before=observed_before))
    if isinstance(payload, list):
        return [dict(x) for x in payload if isinstance(x, Mapping)]
    if isinstance(payload, Mapping):
        if _quality_evidence_only_payload(payload):
            return []
        if path.name == "validation_report.json" and (payload.get("timings") or payload.get("variant_status") or payload.get("run_cfg")):
            return [_validation_report_to_row(payload, path)]
        for key in ("results", "benchmark_results", "measurements", "runs", "rows", "cases", "planned_runs"):
            value = payload.get(key)
            if isinstance(value, list):
                return [dict(x) for x in value if isinstance(x, Mapping)]
    return []


def discover_result_files(paths: Iterable[str | Path]) -> list[Path]:
    files: list[Path] = []
    seen: set[str] = set()
    for raw in paths:
        if not raw:
            continue
        p = Path(raw).expanduser()
        candidates: list[Path] = []
        if p.is_file():
            candidates.append(p)
        elif p.is_dir():
            search_dirs = [p]
            for name in ("results", "generated_suite", "generated_suite/results", "suite", "suite/results", "benchmark_results", "remote", "remote/results"):
                q = p / name
                if q.is_dir():
                    search_dirs.append(q)
            # Include one nested level for benchmark executor outputs without
            # walking potentially huge model/data trees recursively.
            for child in p.iterdir():
                if child.is_dir() and child.name in {"results", "generated_suite", "suite", "benchmark_results"}:
                    search_dirs.append(child)
            seen_dirs: set[str] = set()
            for q in search_dirs:
                try:
                    key_dir = str(q.resolve())
                except Exception:
                    key_dir = str(q)
                if key_dir in seen_dirs:
                    continue
                seen_dirs.add(key_dir)
                for pat in (
                    "benchmark_results_*.json",
                    "benchmark_results_*.csv",
                    "benchmark_results.json",
                    "results.json",
                    "summary.csv",
                    "benchmark_summary.csv",
                    # v60i: v42/v47 pipeline summaries are diagnostic legacy
                    # artefacts.  They duplicate canonical benchmark/validation
                    # rows and are deliberately excluded from scientific ingestion.
                    # status matrices contain ok/missing flags but no timings; keep
                    # them diagnostic-only, not normalized result sources.
                ):
                    candidates.extend(sorted(q.glob(pat)))
                # v58f: nested remote diagnostics can hold the decisive per-case
                # validation reports.  Source-aware dedupe treats them as
                # enrichment/fallback rows, not duplicate measurements.
                for relroot in ("remote_diagnostics", "case_reports", "lean_bundle", "results"):
                    rr = q / relroot
                    if rr.is_dir():
                        candidates.extend(sorted(rr.rglob("validation_report.json")))
        for c in candidates:
            key = str(c.resolve())
            if key not in seen and c.is_file():
                seen.add(key)
                files.append(c)
    return files




def _has_complete_split_latency_v58q(row: Mapping[str, Any]) -> bool:
    if str(row.get("variant") or "").strip().lower() != "split":
        return False
    for k in ("total_latency_ms", "split_latency_e2e_ms", "composed_mean_ms", "composed_latency_ms"):
        if _float_or_none(row.get(k)) is not None:
            return True
    p1 = _float_or_none(row.get("part1_latency_ms") or row.get("part1_mean_ms"))
    p2 = _float_or_none(row.get("part2_latency_ms") or row.get("part2_mean_ms"))
    return p1 is not None and p2 is not None

def _drop_spurious_full_placeholders_v58q(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Remove empty-case Full/backend placeholder rows that duplicate canonical rows.

    Some remote result sources emit helper rows with case_id='' for a full backend
    and a synthetic split variant.  They are useful while merging raw files but
    should not appear as measured rows in normalized_results/reports.
    """
    full_keys = {
        (str(r.get("model_id") or "").strip().lower(), str(r.get("backend") or "").strip().lower())
        for r in rows
        if str(r.get("variant") or "").strip().lower() == "full"
        and str(r.get("case_id") or "").strip().lower() == "full"
        and _bool_or_none(r.get("final_pass")) is not None
    }
    out: list[dict[str, Any]] = []
    for raw in rows:
        r = dict(raw or {})
        case = str(r.get("case_id") or "").strip()
        var = str(r.get("variant") or "").strip().lower()
        key = (str(r.get("model_id") or "").strip().lower(), str(r.get("backend") or "").strip().lower())
        if not case and key in full_keys:
            if var == "split" and not _has_complete_split_latency_v58q(r):
                continue
            if var == "full" and _bool_or_none(r.get("final_pass")) is None:
                continue
        out.append(r)
    return out

def _dedupe_variant_key_v58g(row: Mapping[str, Any]) -> str:
    primary = str(row.get("primary_variant") or "").strip().lower()
    variant = str(row.get("variant") or "").strip().lower()
    if variant in {"", "split", "complete"} and primary in {"composed", "part1", "part2", "full"}:
        return primary
    if variant == "split":
        return "composed"
    return variant or primary or ""


def _quality_run_token_v265(value: Any) -> str:
    token = str(value or "").strip().lower().replace("-", "_")
    for prefix in ("benchmark_results_", "results_"):
        if token.startswith(prefix):
            token = token[len(prefix):]
    if token.endswith("_auto"):
        token = token[:-5]
    # Runner folders use the short ``trt`` spelling while normalized backend
    # labels often use ``tensorrt``.  This is an explicit alias, not substring
    # matching.
    token = token.replace("_to_tensorrt", "_to_trt")
    return {
        "deepx_m1_to_trt": "deepx_to_trt",
        "hailo10h_to_trt": "hailo10_to_trt",
    }.get(token, token)


def _quality_source_setup_ids_v265(row: Mapping[str, Any], source_run_id: str) -> list[str]:
    """Recover setup IDs only from paths for the exact producer run."""

    setup_ids: set[str] = set()
    paths = list(_as_list(row.get("source_paths")))
    if row.get("source_path"):
        paths.append(row.get("source_path"))
    for raw_path in paths:
        parts = list(Path(str(raw_path)).parts)
        lowered = [part.strip().lower() for part in parts]
        result_runs = {
            _quality_run_token_v265(part)
            for part in lowered
            if part.startswith("results_")
        }
        if source_run_id not in result_runs:
            continue
        for index, part in enumerate(lowered[:-1]):
            if part != "remote_diagnostics":
                continue
            candidate = str(parts[index + 1]).strip()
            if candidate.lower() not in {"", "case_reports", "lean_bundle", "results"}:
                setup_ids.add(candidate)
    return sorted(setup_ids)


def _apply_quality_source_identity_v265(row: dict[str, Any]) -> None:
    source_run_id = _quality_run_token_v265(
        row.get("run_id") or row.get("source_tag") or row.get("backend")
    )
    row["quality_source_run_id"] = source_run_id
    explicit_setup = str(row.get("source_setup_id") or row.get("setup_id") or "").strip()
    setup_ids = _quality_source_setup_ids_v265(row, source_run_id)
    if explicit_setup and explicit_setup not in setup_ids:
        setup_ids.append(explicit_setup)
    nested_setup_ids = _identity_setup_ids_v27550(row)
    for setup_id in nested_setup_ids:
        if setup_id not in setup_ids:
            setup_ids.append(setup_id)
    if not str(row.get("setup_id") or "").strip() and len(set(setup_ids)) == 1:
        row["setup_id"] = setup_ids[0]
    row["quality_source_setup_ids"] = sorted(setup_ids)
    row["quality_source_variant"] = _dedupe_variant_key_v58g(row)
    # The request descriptor is canonical scientific evidence embedded in the
    # benchmark result.  v2.75.49 stores the authoritative fields directly in
    # ``composed_request.json`` (setup/backend/precision plus a nested endpoint
    # contract), rather than below ``producer_identity``.  Project that exact
    # shape per variant and retain conflicts instead of selecting the first
    # convenient alias.
    existing_identities = row.get("quality_request_identities_by_variant")
    identities: dict[str, dict[str, Any]] = {
        str(key or "").strip().lower(): dict(value)
        for key, value in (
            existing_identities.items()
            if isinstance(existing_identities, Mapping) else []
        )
        if str(key or "").strip() and isinstance(value, Mapping)
    }
    variants: set[str] = set()
    for container_key in (
        "task_quality_input_requests_by_variant",
        "task_quality_gates_by_variant",
    ):
        container = row.get(container_key)
        if isinstance(container, Mapping):
            variants.update(
                str(value or "").strip().lower()
                for value in container
                if str(value or "").strip()
            )
    selected_variant = _dedupe_variant_key_v58g(row)
    if selected_variant:
        variants.add(selected_variant)
    variants.update(identities)

    def _request_blocks(request: Mapping[str, Any]) -> list[Mapping[str, Any]]:
        blocks: list[Mapping[str, Any]] = [request]
        for key in ("request_payload", "request_record", "payload"):
            nested = request.get(key)
            if isinstance(nested, Mapping):
                blocks.append(nested)
        nested_request = request.get("request")
        if isinstance(nested_request, Mapping) and any(
            key in nested_request
            for key in (
                "setup_id", "backend", "source_run_id", "endpoint_contract",
                "endpoint_contract_hash", "runtime_precision_identity",
            )
        ):
            blocks.append(nested_request)
        return blocks

    def _scalar(
        values: Iterable[Any], *, field: str, errors: list[str],
        normalize: Any = None,
    ) -> str:
        normalized: list[str] = []
        for value in values:
            if value in (None, ""):
                continue
            text = str(value).strip()
            if not text:
                continue
            text = normalize(text) if callable(normalize) else text.lower()
            if text not in normalized:
                normalized.append(text)
        if len(normalized) > 1:
            errors.append(f"conflicting_{field}:" + "|".join(sorted(normalized)))
            return ""
        return normalized[0] if normalized else ""

    def _mapping_value(
        values: Iterable[Any], *, field: str, errors: list[str],
    ) -> dict[str, Any]:
        unique: dict[str, dict[str, Any]] = {}
        for value in values:
            if not isinstance(value, Mapping) or not value:
                continue
            payload = dict(value)
            encoded = json.dumps(
                payload, sort_keys=True, separators=(",", ":"), default=str,
            )
            unique.setdefault(encoded, payload)
        if len(unique) > 1:
            errors.append(f"conflicting_{field}")
            return {}
        return next(iter(unique.values())) if unique else {}

    def _runtime_precision(values: Iterable[Any], errors: list[str]) -> str:
        # A DXNN digest token and the original structured DXNN contract are
        # equivalent only when the existing precision validator agrees. Never
        # stringify a mapping before checking it: that creates false conflicts
        # and loses malformed-contract errors.
        identities: set[str] = set()
        invalid = False
        for value in values:
            if value in (None, ""):
                continue
            canonical, error = canonical_runtime_precision_identity(value)
            if error:
                invalid = True
                errors.append("invalid_embedded_runtime_precision_identity:" + error)
            elif canonical:
                identities.add(canonical)
        if len(identities) > 1:
            errors.append("conflicting_embedded_runtime_precision_identity:" + "|".join(sorted(identities)))
        return next(iter(identities)) if len(identities) == 1 and not invalid else ""

    for variant in sorted(variants):
        requests = _quality_requests_for_variant_v269d(row, variant)
        if not requests:
            # The normalization pass already projected the exact request.
            # Preserve that compact evidence through later de-duplication;
            # never rebuild a variant identity from an unrelated variant.
            continue
        blocks = [
            block
            for request in requests
            for block in _request_blocks(request)
        ]
        producers = [
            _mapping(block.get("producer_identity")) for block in blocks
        ]
        contracts = [
            _mapping(block.get("quality_contract")) for block in blocks
        ]
        preprocessing_contracts = [
            _mapping(contract.get("preprocessing")) for contract in contracts
        ]
        prepared_input_evidence = [
            _mapping(contract.get("prepared_input_evidence"))
            for contract in contracts
        ] + [
            _mapping(block.get("prepared_input_evidence")) for block in blocks
        ]
        endpoints = [
            _mapping(block.get("endpoint_contract")) for block in blocks
        ]
        attestations = [
            value
            for endpoint in endpoints
            for value in (endpoint.get("output_endpoint_attestation"),)
        ] + [block.get("output_endpoint_attestation") for block in blocks]
        errors: list[str] = []

        embedded_direction = _scalar(
            [
                value
                for block in blocks
                for value in (
                    block.get("direction"), block.get("backend"),
                    block.get("source_run_id"),
                )
            ],
            field="embedded_direction",
            errors=errors,
            normalize=canonical_direction,
        )
        source_run = _scalar(
            [
                value
                for block in blocks
                for value in (block.get("source_run_id"), block.get("backend"))
            ] + [
                value
                for producer in producers
                for value in (producer.get("source_run_id"), producer.get("run_id"))
            ],
            field="embedded_source_run_id",
            errors=errors,
            normalize=_quality_run_token_v265,
        )
        setup_id = _scalar(
            [
                value
                for block in blocks
                for value in (
                    block.get("setup_id"), block.get("source_setup_id"),
                    block.get("measurement_setup_id"),
                )
            ] + [
                value
                for mapping in (*producers, *contracts)
                for value in (
                    mapping.get("setup_id"), mapping.get("source_setup_id"),
                )
            ],
            field="embedded_setup_id",
            errors=errors,
        )
        task = _scalar(
            [block.get("task") for block in blocks]
            + [endpoint.get("task") for endpoint in endpoints],
            field="embedded_task", errors=errors,
        )
        stage = _scalar(
            [block.get("stage") for block in blocks]
            + [endpoint.get("stage") for endpoint in endpoints]
            + [
                _mapping(endpoint.get("producer_endpoint_identity")).get("stage")
                for endpoint in endpoints
            ]
            + [
                _mapping(value).get("stage") for value in attestations
            ],
            field="embedded_stage", errors=errors,
        )
        endpoint_hash = _scalar(
            [block.get("endpoint_contract_hash") for block in blocks]
            + [endpoint.get("endpoint_contract_hash") for endpoint in endpoints]
            + [
                producer.get("endpoint_contract_hash") for producer in producers
            ]
            + [
                contract.get("endpoint_contract_hash") for contract in contracts
            ]
            + [
                _mapping(value).get("endpoint_contract_hash")
                for value in attestations
            ],
            field="embedded_endpoint_contract_hash", errors=errors,
            normalize=lambda value: value.lower().removeprefix("sha256:"),
        )
        precision = _runtime_precision(
            [
                value
                for block in blocks
                for value in (
                    block.get("runtime_precision_identity"),
                    block.get("execution_precision"), block.get("precision"),
                )
            ] + [
                value
                for mapping in (*producers, *contracts)
                for value in (
                    mapping.get("runtime_precision_identity"),
                    mapping.get("execution_precision"), mapping.get("precision"),
                )
            ],
            errors=errors,
        )
        # Numeric-input identity was not written at one fixed level by all
        # producer generations. Preserve only exact mappings already present
        # in the request/evidence chain. In particular, a preprocessing
        # ``runtime_input_encoding`` describes how an image is encoded but is
        # not a complete ``runtime_numeric_input_identity``; keeping the two
        # surfaces distinct avoids manufacturing stronger evidence on replay.
        numeric_identity = _mapping_value(
            [block.get("runtime_numeric_input_identity") for block in blocks]
            + [
                mapping.get("runtime_numeric_input_identity")
                for mapping in (*producers, *contracts, *prepared_input_evidence)
            ],
            field="embedded_runtime_numeric_input_identity", errors=errors,
        )
        numeric_sha = _scalar(
            [block.get("runtime_numeric_input_sha256") for block in blocks]
            + [
                mapping.get("runtime_numeric_input_sha256")
                for mapping in (*producers, *contracts, *prepared_input_evidence)
            ],
            field="embedded_runtime_numeric_input_sha256", errors=errors,
            normalize=lambda value: value.lower().removeprefix("sha256:"),
        )
        input_encoding = _mapping_value(
            [block.get("runtime_input_encoding") for block in blocks]
            + [
                mapping.get("runtime_input_encoding")
                for mapping in (*contracts, *preprocessing_contracts)
            ],
            field="embedded_runtime_input_encoding", errors=errors,
        )
        input_encoding_sha = _scalar(
            [block.get("runtime_input_encoding_sha256") for block in blocks]
            + [
                mapping.get("runtime_input_encoding_sha256")
                for mapping in (*contracts, *preprocessing_contracts)
            ],
            field="embedded_runtime_input_encoding_sha256", errors=errors,
            normalize=lambda value: value.lower().removeprefix("sha256:"),
        )
        endpoint_complete_values = {
            value
            for value in (
                [block.get("endpoint_contract_complete") for block in blocks]
                + [endpoint.get("endpoint_contract_complete") for endpoint in endpoints]
            )
            if isinstance(value, bool)
        }
        if len(endpoint_complete_values) > 1:
            errors.append("conflicting_embedded_endpoint_contract_complete")
            endpoint_complete: bool | None = None
        else:
            endpoint_complete = (
                next(iter(endpoint_complete_values))
                if endpoint_complete_values else None
            )
        attestation = _mapping_value(
            attestations, field="embedded_output_endpoint_attestation",
            errors=errors,
        )

        # Row-level fields describe only the selected measurement variant.
        # Compare them with the embedded request and fail closed on drift.
        if variant == selected_variant:
            explicit_setup = str(
                row.get("setup_id") or row.get("source_setup_id") or ""
            ).strip().lower()
            if not explicit_setup and len(set(setup_ids)) == 1:
                explicit_setup = str(setup_ids[0]).strip().lower()
            comparisons = (
                ("setup_id", explicit_setup, setup_id),
                (
                    "direction",
                    canonical_direction(
                        row.get("direction") or row.get("run_id")
                        or row.get("quality_source_run_id")
                        or row.get("backend") or "",
                    ),
                    embedded_direction,
                ),
                (
                    "runtime_precision_identity",
                    _runtime_precision([
                        row.get("runtime_precision_identity")
                        or row.get("execution_precision") or row.get("precision")
                        or ""
                    ], errors),
                    precision,
                ),
                (
                    "endpoint_contract_hash",
                    str(row.get("endpoint_contract_hash") or "")
                    .strip().lower().removeprefix("sha256:"),
                    endpoint_hash,
                ),
            )
            for field, explicit, embedded in comparisons:
                if explicit and embedded and explicit != embedded:
                    errors.append(
                        f"explicit_embedded_{field}_conflict:{explicit}!={embedded}"
                    )
            if not setup_id:
                setup_id = explicit_setup
            if not embedded_direction:
                embedded_direction = comparisons[1][1]
            if not precision:
                precision = comparisons[2][1]
            if not endpoint_hash:
                endpoint_hash = comparisons[3][1]

            explicit_complete = row.get("endpoint_contract_complete")
            if (
                isinstance(explicit_complete, bool)
                and endpoint_complete is not None
                and explicit_complete is not endpoint_complete
            ):
                errors.append(
                    "explicit_embedded_endpoint_contract_complete_conflict"
                )
            if endpoint_complete is None and isinstance(explicit_complete, bool):
                endpoint_complete = explicit_complete
            explicit_attestation = row.get("output_endpoint_attestation")
            if isinstance(explicit_attestation, Mapping) and explicit_attestation:
                if attestation and json.dumps(
                    dict(explicit_attestation), sort_keys=True,
                    separators=(",", ":"), default=str,
                ) != json.dumps(
                    attestation, sort_keys=True, separators=(",", ":"),
                    default=str,
                ):
                    errors.append(
                        "explicit_embedded_output_endpoint_attestation_conflict"
                    )
                elif not attestation:
                    attestation = dict(explicit_attestation)

            explicit_numeric = (
                dict(row.get("runtime_numeric_input_identity") or {})
                if isinstance(row.get("runtime_numeric_input_identity"), Mapping)
                else {}
            )
            explicit_numeric_sha = str(
                row.get("runtime_numeric_input_sha256") or ""
            ).strip().lower().removeprefix("sha256:")
            if explicit_numeric and numeric_identity and json.dumps(
                explicit_numeric, sort_keys=True, separators=(",", ":"),
                default=str,
            ) != json.dumps(
                numeric_identity, sort_keys=True, separators=(",", ":"),
                default=str,
            ):
                errors.append(
                    "explicit_embedded_runtime_numeric_input_identity_conflict"
                )
            elif explicit_numeric and not numeric_identity:
                numeric_identity = explicit_numeric
            if (
                explicit_numeric_sha and numeric_sha
                and explicit_numeric_sha != numeric_sha
            ):
                errors.append(
                    "explicit_embedded_runtime_numeric_input_sha256_conflict:"
                    f"{explicit_numeric_sha}!={numeric_sha}"
                )
            elif explicit_numeric_sha and not numeric_sha:
                numeric_sha = explicit_numeric_sha

        if any("runtime_numeric_input" in error for error in errors):
            numeric_status = "conflict"
        elif numeric_identity and numeric_sha:
            numeric_status = "complete"
        elif numeric_identity or numeric_sha:
            numeric_status = "partial_explicit_numeric_identity"
        elif input_encoding or input_encoding_sha:
            numeric_status = "runtime_input_encoding_only"
        else:
            numeric_status = "unavailable"

        direction = embedded_direction or canonical_direction(source_run)
        stage1, stage2 = direction_parts(direction)
        producer_backend = _scalar(
            [
                value
                for block in blocks
                for value in (block.get("producer_backend"),)
            ] + [
                value
                for producer in producers
                for value in (producer.get("producer_backend"), producer.get("backend"))
            ],
            field="embedded_producer_backend", errors=errors,
            normalize=canonical_backend,
        ) or stage1
        comparison_backend = _scalar(
            [block.get("comparison_backend") for block in blocks]
            + [producer.get("comparison_backend") for producer in producers],
            field="embedded_comparison_backend", errors=errors,
            normalize=canonical_backend,
        )
        if not comparison_backend:
            accelerators = [
                value for value in (stage1, stage2)
                if value.startswith(("hailo", "deepx"))
            ]
            comparison_backend = accelerators[0] if len(accelerators) == 1 else ""

        request_sha_values: list[Any] = []
        for request in requests:
            request_record = request.get("request")
            if isinstance(request_record, Mapping):
                request_sha_values.append(request_record.get("sha256"))
            request_sha_values.append(request.get("source_request_sha256"))
        request_sha = _scalar(
            request_sha_values, field="source_request_sha256", errors=errors,
            normalize=lambda value: value.lower().removeprefix("sha256:"),
        )
        identities[variant] = {
            "schema": "onnx-splitpoint/central-quality-request-identity",
            "schema_version": 3,
            "model_id": str(row.get("model_id") or "").strip().lower(),
            "case_id": str(row.get("case_id") or "").strip().lower(),
            "source_run_id": source_run or source_run_id,
            "setup_id": setup_id,
            "setup_ids": [setup_id] if setup_id else [],
            "variant": variant,
            "task": task or str(row.get("task") or "").strip().lower(),
            "direction": direction,
            "producer_backend": producer_backend,
            "comparison_backend": comparison_backend,
            "stage": stage,
            "source_request_sha256": request_sha,
            "endpoint_contract_complete": endpoint_complete,
            "endpoint_contract_hash": endpoint_hash,
            "output_endpoint_attestation": attestation,
            "runtime_precision_identity": precision,
            "runtime_numeric_input_identity": numeric_identity,
            "runtime_numeric_input_sha256": numeric_sha,
            "runtime_input_encoding_identity": input_encoding,
            "runtime_input_encoding_sha256": input_encoding_sha,
            "runtime_numeric_input_identity_status": numeric_status,
            "identity_valid": not errors,
            "identity_errors": sorted(set(errors)),
        }
    row["quality_request_identities_by_variant"] = identities


def _full_source_case_ids_v269d(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    values: set[str] = set()
    for row in rows:
        for value in _as_list(row.get("full_source_case_ids")):
            token = str(value or "").strip().lower()
            if token and token != "full":
                values.add(token)
        for key in ("source_case_id", "original_case_id"):
            token = str(row.get(key) or "").strip().lower()
            if token and token != "full":
                values.add(token)
        token = str(row.get("case_id") or "").strip().lower()
        if token and token != "full":
            values.add(token)
    return sorted(values)


def _full_source_role_v269d(row: Mapping[str, Any]) -> str:
    """Classify whether a row is the planned owner of a Full measurement."""
    run_cfg = _mapping(row.get("run_cfg"))
    if row.get("full_source_owner_declared") is True or run_cfg.get("_full_baseline_owner") is True:
        backend = _canon_backend_token(
            row.get("backend") or row.get("full_provider") or row.get("full_backend") or ""
        )
        declared_backend = _canon_backend_token(
            row.get("full_source_owner_backend")
            or run_cfg.get("_full_baseline_owner_backend") or ""
        )
        source_run_id = _full_source_run_id_v269d(row)
        declared_run_id = str(
            row.get("full_source_owner_run_id")
            or run_cfg.get("_full_baseline_owner_run_id") or ""
        ).strip().lower()
        if (
            backend and declared_backend == backend
            and source_run_id and declared_run_id == source_run_id
        ):
            return "declared_owner"
        return "declared_owner_conflict"
    primary = str(
        row.get("full_source_primary_variant") or row.get("primary_variant") or ""
    ).strip().lower()
    if primary == "full":
        return "primary_full"
    backend = _canon_backend_token(
        row.get("backend") or row.get("full_provider") or row.get("full_backend") or ""
    )
    stage1 = _canon_backend_token(
        row.get("full_source_stage1_provider") or ""
    )
    stage2 = _canon_backend_token(
        row.get("full_source_stage2_provider") or ""
    )
    if backend and stage1 == backend and stage2 == backend:
        return "same_backend_reference"
    return "incidental_companion"


def _full_source_run_id_v269d(row: Mapping[str, Any]) -> str:
    return str(
        row.get("full_source_run_id") or row.get("run_id") or row.get("source_tag") or ""
    ).strip().lower()


def _full_mirror_evidence_v269d(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return claim-relevant evidence used to prove two rows are mirrors."""
    evidence: dict[str, Any] = {}
    for field in (
        "preprocessing_contract_sha256", "endpoint_contract_hash",
        "source_model_sha256", "runtime_artifact_sha256",
        "runtime_precision_identity", "execution_precision",
        "endpoint_contract_complete", "task", "task_quality_policy_sha256",
        "runtime_ok", "validation_ok", "final_pass", "final_pass_all",
        "total_latency_ms", "full_latency_ms", "full_e2e_latency_ms",
        "full_raw_head_latency_ms", "full_raw_accelerator_latency_ms",
        "full_host_tail_latency_ms", "full_backend_throughput_fps",
        "full_source_owner_declared", "full_source_owner_run_id",
        "full_source_owner_backend", "full_source_primary_variant",
        "full_source_stage1_provider", "full_source_stage2_provider",
    ):
        value = row.get(field)
        if value not in (None, "", [], {}):
            evidence[field] = value
    frozen = _mapping(row.get("frozen_identity_evidence"))
    if frozen:
        evidence["frozen_identity_evidence"] = dict(frozen)
    attestation = _mapping(row.get("output_endpoint_attestation"))
    for field in ("endpoint_contract_hash", "stage", "status", "attested"):
        value = attestation.get(field)
        if value not in (None, "", [], {}):
            evidence[f"output_endpoint_attestation.{field}"] = value
    quality_identities = _mapping(row.get("quality_request_identities_by_variant"))
    if quality_identities:
        evidence["quality_request_identities_by_variant"] = dict(quality_identities)
    return evidence


def _full_mirror_conflicts_v269d(
    canonical: Mapping[str, Any], mirror: Mapping[str, Any],
) -> list[str]:
    left = _full_mirror_evidence_v269d(canonical)
    right = _full_mirror_evidence_v269d(mirror)
    conflicts: list[str] = []
    for field in sorted(set(left) & set(right)):
        a = left[field]
        b = right[field]
        if isinstance(a, (int, float)) and not isinstance(a, bool) and isinstance(b, (int, float)) and not isinstance(b, bool):
            if not math.isclose(float(a), float(b), rel_tol=1.0e-9, abs_tol=1.0e-9):
                conflicts.append(field)
            continue
        if isinstance(a, Mapping) or isinstance(b, Mapping) or isinstance(a, list) or isinstance(b, list):
            try:
                a_text = json.dumps(a, sort_keys=True, separators=(",", ":"), default=str)
                b_text = json.dumps(b, sort_keys=True, separators=(",", ":"), default=str)
            except Exception:
                a_text, b_text = repr(a), repr(b)
            if a_text != b_text:
                conflicts.append(field)
            continue
        if str(a).strip().lower() != str(b).strip().lower():
            conflicts.append(field)
    return conflicts


def _select_full_source_rows_v269d(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select the owner rows and collapse JSON/CSV mirrors of one sample."""
    role_priority = {
        "declared_owner_conflict": -1,
        "declared_owner": 0,
        "primary_full": 1,
        "same_backend_reference": 2,
        "incidental_companion": 9,
    }
    annotated = [
        (role_priority[_full_source_role_v269d(row)], _full_source_role_v269d(row), dict(row))
        for row in rows
    ]
    selected_priority = min((item[0] for item in annotated), default=99)
    candidates = [item for item in annotated if item[0] == selected_priority]

    # The same physical row can be present in canonical JSON, CSV and a
    # validation-report mirror.  Aggregate one observation per owner run/case,
    # preferring the canonical benchmark JSON source.
    candidates_by_sample: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for _priority, _role, row in candidates:
        run_id = _full_source_run_id_v269d(row)
        case_id = str(
            row.get("source_case_id") or row.get("original_case_id")
            or row.get("case_id") or "full"
        ).strip().lower()
        candidates_by_sample.setdefault((run_id, case_id), []).append(row)

    selected: list[dict[str, Any]] = []
    duplicate_conflicts: list[dict[str, Any]] = []
    collapsed_mirror_count = 0
    for (run_id, case_id), sample_rows in sorted(candidates_by_sample.items()):
        ordered = sorted(sample_rows, key=_source_priority_for_dedupe_v58f)
        best_priority = _source_priority_for_dedupe_v58f(ordered[0])
        best = [
            row for row in ordered
            if _source_priority_for_dedupe_v58f(row) == best_priority
        ]
        reasons: set[str] = set()
        if len(best) != 1:
            reasons.add("multiple_equal_priority_representations")
        canonical = best[0]
        for mirror in ordered[1:]:
            for field in _full_mirror_conflicts_v269d(canonical, mirror):
                reasons.add(f"evidence_mismatch:{field}")
        if reasons:
            # Retain every representation so the downstream frozen-identity
            # gate can also expose endpoint/model/artifact disagreements.
            selected.extend(ordered)
            duplicate_conflicts.append({
                "source_run_id": run_id,
                "source_case_id": case_id,
                "row_count": len(ordered),
                "reasons": sorted(reasons),
                "source_paths": sorted({
                    str(row.get("source_path") or "") for row in ordered
                    if str(row.get("source_path") or "")
                }),
            })
        else:
            selected.append(canonical)
            collapsed_mirror_count += max(0, len(ordered) - 1)
    selected.sort(key=lambda row: (
        _full_source_run_id_v269d(row),
        str(row.get("source_case_id") or row.get("case_id") or ""),
        _source_priority_for_dedupe_v58f(row),
    ))
    selected_roles = sorted({_full_source_role_v269d(row) for row in selected})
    selected_run_ids = sorted({
        _full_source_run_id_v269d(row) for row in selected
        if _full_source_run_id_v269d(row)
    })
    all_run_ids = sorted({
        _full_source_run_id_v269d(row) for row in rows
        if _full_source_run_id_v269d(row)
    })
    ignored_run_ids = sorted(set(all_run_ids) - set(selected_run_ids))
    selected_role = selected_roles[0] if len(selected_roles) == 1 else "conflict"
    complete = bool(
        selected
        and len(selected_run_ids) == 1
        and selected_role in {"declared_owner", "primary_full", "same_backend_reference"}
        and not duplicate_conflicts
    )
    status = (
        "selected" if complete
        else "source_conflict" if (
            len(selected_run_ids) > 1 or len(selected_roles) > 1
            or selected_role == "declared_owner_conflict" or duplicate_conflicts
        )
        else "fallback_incidental" if selected_role == "incidental_companion"
        else "unavailable"
    )
    return selected, {
        "status": status,
        "complete": complete,
        "selected_role": selected_role,
        "selected_priority": selected_priority,
        "selected_run_ids": selected_run_ids,
        "all_run_ids": all_run_ids,
        "ignored_run_ids": ignored_run_ids,
        "selected_row_count": len(selected),
        "candidate_row_count": len(rows),
        "mirror_row_count": collapsed_mirror_count,
        "duplicate_conflicts": duplicate_conflicts,
    }


def _invalidate_full_source_v269d(row: dict[str, Any], reason: str) -> None:
    row["row_conflict"] = "full_baseline_source_identity_conflict"
    row["contract_gate_reason"] = reason
    row["error_class"] = "full_baseline_source_identity_conflict"
    row["contract_consistent"] = False
    row["structural_contract_pass"] = False
    row["structural_contract_status"] = "failed"
    row["structural_contract_reason"] = reason
    for key in (
        "validation_ok", "final_pass", "final_pass_all",
        "semantic_validation_ok", "semantic_validation_passed",
        "semantic_e2e_pass", "task_valid", "accuracy_gate_pass",
        "eligible_for_ranking", "ranking_eligible", "performance_eligible",
        "energy_eligible",
    ):
        row[key] = False
    row["gate_status"] = "failed"
    row["ranking_exclusion_reason"] = reason
    eligibility = dict(row.get("row_eligibility") or {}) if isinstance(row.get("row_eligibility"), Mapping) else {}
    eligibility.update({
        "io_contract_pass": False,
        "task_quality_pass": False,
        "ranking_eligible": False,
        "performance_eligible": False,
        "energy_eligible": False,
        "pareto_eligible": False,
        "exclusion_reason": reason,
    })
    row["row_eligibility"] = eligibility


def _apply_full_source_selection_v269d(
    row: dict[str, Any], selected: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    row["full_baseline_source_status"] = str(metadata.get("status") or "unavailable")
    row["full_baseline_source_complete"] = metadata.get("complete") is True
    row["full_baseline_source_role"] = str(metadata.get("selected_role") or "")
    row["full_baseline_source_run_ids"] = list(metadata.get("selected_run_ids") or [])
    row["full_baseline_all_source_run_ids"] = list(metadata.get("all_run_ids") or [])
    row["full_baseline_ignored_source_run_ids"] = list(metadata.get("ignored_run_ids") or [])
    row["full_baseline_selected_source_row_count"] = int(metadata.get("selected_row_count") or 0)
    row["full_baseline_candidate_source_row_count"] = int(metadata.get("candidate_row_count") or 0)
    row["full_baseline_mirror_row_count"] = int(metadata.get("mirror_row_count") or 0)
    row["full_baseline_duplicate_source_conflicts"] = list(
        metadata.get("duplicate_conflicts") or []
    )
    row["full_baseline_duplicate_source_conflict_count"] = len(
        row["full_baseline_duplicate_source_conflicts"]
    )

    samples: list[dict[str, Any]] = []
    for source in selected:
        latency = _float_or_none(source.get("total_latency_ms"))
        if latency is None:
            continue
        samples.append({
            "source_run_id": _full_source_run_id_v269d(source),
            "source_case_id": str(
                source.get("source_case_id") or source.get("case_id") or "full"
            ).strip().lower(),
            "source_path": str(source.get("source_path") or ""),
            "latency_ms": latency,
        })
    samples.sort(key=lambda sample: (
        sample["source_run_id"], sample["source_case_id"], sample["source_path"],
    ))
    row["full_baseline_latency_samples"] = samples
    row["full_baseline_latency_sample_count"] = len(samples)
    if samples:
        values = [float(sample["latency_ms"]) for sample in samples]
        mean_latency = sum(values) / len(values)
        std_latency = (
            math.sqrt(sum((value - mean_latency) ** 2 for value in values) / (len(values) - 1))
            if len(values) > 1 else 0.0
        )
        row["full_baseline_latency_aggregate"] = "arithmetic_mean_of_owner_case_containers"
        row["full_baseline_latency_mean_ms"] = mean_latency
        row["full_baseline_latency_std_between_cases_ms"] = std_latency
        row["total_latency_ms"] = mean_latency
        row["full_latency_ms"] = mean_latency
        for field in (
            "full_e2e_latency_ms", "full_raw_head_latency_ms",
            "full_raw_accelerator_latency_ms", "full_host_tail_latency_ms",
        ):
            field_values = [
                value for value in (
                    _float_or_none(source.get(field)) for source in selected
                )
                if value is not None
            ]
            if field_values:
                row[field] = sum(field_values) / len(field_values)
        if mean_latency > 0.0:
            fps = 1000.0 / mean_latency
            row["full_backend_throughput_fps"] = fps
            if str(row.get("throughput_kind") or "").strip().lower() == "full_backend":
                row["throughput_primary_fps"] = fps

    if metadata.get("complete") is not True:
        reason = "full_baseline_authoritative_source_" + str(
            metadata.get("status") or "unavailable"
        )
        _invalidate_full_source_v269d(row, reason)
    return row


def _full_identity_group_evidence_v269d(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    values: dict[str, set[str]] = {
        "preprocessing_contract_sha256": set(),
        "endpoint_contract_hash": set(),
        "source_model_sha256": set(),
        "runtime_artifact_sha256": set(),
        "runtime_precision_identity": set(),
    }
    invalid: dict[str, set[str]] = {key: set() for key in values}

    def add(field: str, value: Any) -> None:
        if value in (None, ""):
            return
        if field in _FULL_FROZEN_SHA_FIELDS_V269D:
            normalized, valid = _identity_sha_v269d(value)
            if normalized:
                values[field].add(normalized)
                if not valid:
                    invalid[field].add(normalized)
            return
        normalized = str(value or "").strip().lower().replace(" ", "")
        if normalized:
            values[field].add(normalized)

    for row in rows:
        evidence = _mapping(row.get("frozen_identity_evidence"))
        for field, candidates in _mapping(evidence.get("values")).items():
            if field not in values:
                continue
            for candidate in _as_list(candidates):
                add(field, candidate)
        for field, candidates in _mapping(evidence.get("invalid")).items():
            if field not in invalid:
                continue
            for candidate in _as_list(candidates):
                normalized = str(candidate or "").strip().lower()
                if normalized:
                    invalid[field].add(normalized)

        # Support already-normalized/manual rows that predate the evidence map.
        for field, aliases in {
            "preprocessing_contract_sha256": ("preprocessing_contract_sha256",),
            "endpoint_contract_hash": ("endpoint_contract_hash",),
            "source_model_sha256": ("source_model_sha256", "model_sha256"),
            "runtime_artifact_sha256": ("runtime_artifact_sha256",),
            "runtime_precision_identity": (
                "runtime_precision_identity", "execution_precision",
            ),
        }.items():
            for alias in aliases:
                add(field, row.get(alias))
        attestation = _mapping(row.get("output_endpoint_attestation"))
        add("endpoint_contract_hash", attestation.get("endpoint_contract_hash"))

    return (
        {key: sorted(items) for key, items in values.items() if items},
        {key: sorted(items) for key, items in invalid.items() if items},
    )


def _apply_full_identity_gate_v269d(
    row: dict[str, Any], group: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    values, invalid = _full_identity_group_evidence_v269d(group)
    conflicts = sorted({
        field for field, candidates in values.items() if len(candidates) > 1
    } | set(invalid))
    row["case_id"] = "full"
    row["full_source_case_ids"] = _full_source_case_ids_v269d(group)
    row["full_baseline_frozen_identity_values"] = values
    row["full_baseline_frozen_identity_invalid"] = invalid
    row["full_baseline_frozen_identity_conflicts"] = conflicts
    required = {
        "preprocessing_contract_sha256", "endpoint_contract_hash",
        "runtime_artifact_sha256", "runtime_precision_identity",
    }
    row["full_baseline_frozen_identity_complete"] = bool(
        required.issubset(values) and not conflicts
    )
    row["full_baseline_frozen_identity_status"] = (
        "conflict" if conflicts else "consistent" if values else "unavailable"
    )
    for field, candidates in values.items():
        if len(candidates) == 1 and field not in invalid:
            row[field] = candidates[0]

    if not conflicts:
        return row

    reason = "full_baseline_frozen_identity_conflict:" + ",".join(conflicts)
    row["row_conflict"] = "full_baseline_frozen_identity_conflict"
    row["contract_gate_reason"] = reason
    row["error_class"] = "contract_identity_conflict"
    row["contract_consistent"] = False
    row["structural_contract_pass"] = False
    row["structural_contract_status"] = "failed"
    row["structural_contract_reason"] = reason
    row["validation_ok"] = False
    row["final_pass"] = False
    row["final_pass_all"] = False
    row["semantic_validation_ok"] = False
    row["semantic_validation_passed"] = False
    row["semantic_e2e_pass"] = False
    row["task_valid"] = False
    row["accuracy_gate_pass"] = False
    row["eligible_for_ranking"] = False
    row["ranking_eligible"] = False
    row["performance_eligible"] = False
    row["energy_eligible"] = False
    row["gate_status"] = "failed"
    row["ranking_exclusion_reason"] = reason
    eligibility = dict(row.get("row_eligibility") or {}) if isinstance(row.get("row_eligibility"), Mapping) else {}
    eligibility.update({
        "io_contract_pass": False,
        "task_quality_pass": False,
        "ranking_eligible": False,
        "performance_eligible": False,
        "energy_eligible": False,
        "pareto_eligible": False,
        "exclusion_reason": reason,
    })
    row["row_eligibility"] = eligibility
    return row


def _row_identity_for_dedupe_v58f(row: Mapping[str, Any]) -> tuple[str, ...]:
    # v58g: use canonical backend/variant identities so compact
    # validation_report.json rows enrich primary benchmark rows instead of
    # becoming duplicate split rows.  Validation reports often say
    # variant=split/provider=deepx while benchmark_results rows use
    # variant=composed/backend=deepx_m1_to_tensorrt.
    variant_key = _dedupe_variant_key_v58g(row)
    is_full = variant_key == "full"
    backend_key = (
        _canon_backend_token(
            row.get("backend") or row.get("full_provider")
            or row.get("full_backend") or row.get("provider") or ""
        )
        if is_full
        else _backend(
            row, tag=str(row.get("source_tag") or row.get("run_id") or ""),
        )
    )
    if not backend_key or backend_key == "unknown":
        backend_key = _canon_backend_token(row.get("backend") or row.get("run_id") or row.get("provider") or "")
    return (
        str(row.get("model_id") or "").strip().lower(),
        _normalize_full_case_id_for_dedupe_v58n(row),
        str(backend_key or "").strip().lower(),
        variant_key,
        _resolved_setup_id_v27550(row).strip().lower(),
        "" if is_full else _canon_backend_token(row.get("stage1_provider") or row.get("stage1") or ""),
        "" if is_full else _canon_backend_token(row.get("stage2_provider") or row.get("stage2") or ""),
        _canon_backend_token(row.get("full_provider") or row.get("full_backend") or ""),
        # Raw model outputs and completed detections are different measured
        # intervals even on the same model/backend/case and must not merge.
        str(row.get("measurement_endpoint") or "").strip().lower(),
    )


def _source_priority_for_dedupe_v58f(row: Mapping[str, Any]) -> int:
    sp = str(row.get("source_path") or "").replace("\\", "/").lower()
    name = Path(sp).name
    is_validation = bool(row.get("source_validation_report")) or name == "validation_report.json"
    is_lean = "/lean_bundle/" in sp
    if name.startswith("benchmark_results_") and name.endswith(".json") and not is_validation:
        return 0
    if name == "benchmark_results.json":
        return 1
    if is_validation and not is_lean:
        return 2
    if is_validation and is_lean:
        return 3
    if name.startswith("benchmark_results_") and name.endswith(".csv"):
        return 4
    if name.endswith(".json"):
        return 5
    if name.endswith(".csv"):
        return 6
    return 9


def _merge_result_rows_for_dedupe_v58f(primary: dict[str, Any], extra: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(primary or {})
    extra = dict(extra or {})
    protected = {
        "part1_latency_ms", "part2_latency_ms", "transfer_latency_ms", "transfer_latency_raw_ms",
        "total_latency_ms", "split_latency_e2e_ms", "pipeline_cycle_selected_ms",
        "pipeline_fps_selected", "full_latency_ms", "full_e2e_latency_ms",
        "full_backend_throughput_fps", "heterogeneous_pipeline_fps", "same_backend_composed_fps",
        "throughput_primary_fps", "throughput_primary_metric", "throughput_primary_source",
    }
    enrichment = {
        "classification_top1", "classification_top5", "mini_coco_ap50_primary", "semantic_validation_ok",
        "semantic_validation_passed", "semantic_validation_passed_all", "final_pass", "final_pass_all",
        "validation_ok", "raw_head_contract_present", "raw_head_contract_status", "host_tail_required",
        "host_tail_available", "stage2_input_contract_kind", "stage2_feature_tensor_contract_status",
        "deepx_stage2_contract_status", "deepx_stage2_contract_pass", "deepx_stage2_contract_probe_samples",
        "deepx_stage2_contract_selected_pass_ratio", "stage2_calibration_source", "stage2_calibration_trust_level",
        "score_semantics", "confidence_score_semantics", "raw_score_mean", "confidence_mean_clipped_0_1",
        # v60i runtime task-quality provenance from validation_report.json.
        "task", "benchmark_task_requested", "benchmark_task_used", "validation_dataset_task",
        "task_quality_policy", "task_quality_policy_sha256", "task_quality_gate",
        "task_quality_gates_by_variant", "quality_request_identities_by_variant",
        "validation_cardinality_contract", "row_eligibility",
    }
    srcs = list(_as_list(out.get("source_paths")))
    for sp in (out.get("source_path"), extra.get("source_path")):
        if sp and str(sp) not in srcs:
            srcs.append(str(sp))
    if srcs:
        out["source_paths"] = srcs
    for k, v in extra.items():
        if k in protected or k in {"schema", "schema_version"}:
            continue
        if v in (None, "", [], {}):
            continue
        if out.get(k) in (None, "", [], {}) or (k in enrichment and bool(extra.get("source_validation_report"))):
            # Do not hide primary runner failures with later validation reports.
            if bool(extra.get("source_validation_report")) and k in {"runtime_ok", "validation_ok", "final_pass", "final_pass_all"}:
                a = _bool_or_none(out.get(k))
                b = _bool_or_none(v)
                if a is False and b is True:
                    out["row_conflict"] = "auto_failed_but_validation_report_exists"
                    out["row_conflict_validation_source"] = str(extra.get("source_path") or "")
                    continue
            out[k] = v
    if out.get("semantic_validation_ok") in (None, ""):
        for k in ("semantic_validation_passed_all", "semantic_validation_passed", "validation_ok", "final_pass_all", "final_pass"):
            val = _bool_or_none(out.get(k))
            if val is not None:
                out["semantic_validation_ok"] = val
                break
    if out.get("semantic_validation_passed") in (None, "") and out.get("semantic_validation_ok") not in (None, ""):
        out["semantic_validation_passed"] = out.get("semantic_validation_ok")
    return out


def _dedupe_normalized_rows_v58f(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in rows:
        rr = dict(row or {})
        groups.setdefault(_row_identity_for_dedupe_v58f(rr), []).append(rr)
    out: list[dict[str, Any]] = []
    for key, items in groups.items():
        items = sorted(items, key=_source_priority_for_dedupe_v58f)
        is_full = len(key) > 3 and key[3] == "full"
        source_metadata: Mapping[str, Any] = {}
        merge_items = items
        if is_full:
            selected_items, source_metadata = _select_full_source_rows_v269d(items)
            if selected_items:
                merge_items = selected_items
        primary = dict(merge_items[0])
        primary["dedupe_group_size"] = len(items)
        primary["dedupe_selected_group_size"] = len(merge_items)
        primary["row_identity"] = "|".join(key)
        primary["ingestion_source_priority"] = _source_priority_for_dedupe_v58f(primary)
        for extra in merge_items[1:]:
            primary = _merge_result_rows_for_dedupe_v58f(primary, extra)
        if is_full:
            primary = _apply_full_identity_gate_v269d(primary, merge_items)
            primary = _apply_full_source_selection_v269d(
                primary, merge_items, source_metadata,
            )
        _apply_validation_level_fields_v59j(primary)
        _apply_quality_source_identity_v265(primary)
        out.append(primary)
    out.sort(key=lambda r: (str(r.get("model_id") or ""), str(r.get("backend") or ""), str(r.get("case_id") or ""), str(r.get("variant") or "")))
    return out


def _csv_mirrors_json_row_v27927(
    csv_row: Mapping[str, Any], json_row: Mapping[str, Any], *, tag: str,
) -> bool:
    """Identify a lossy CSV view of the same row in its sibling JSON file.

    Matching only a filename suppresses CSV-only cases/repetitions.  Match the
    measured variant, concrete identity and every shared scalar instead.  JSON
    retains nested setup/quality identity that CSV cannot faithfully encode.
    """
    if (
        not _case_id(csv_row)
        or _case_id(csv_row) != _case_id(json_row)
        or _variant(csv_row) != _variant(json_row)
        or _backend(csv_row, tag=tag) != _backend(json_row, tag=tag)
    ):
        return False
    for key in (
        "setup_id", "source_setup_id", "measurement_setup_id", "hardware_setup_id",
    ):
        csv_setup = str(csv_row.get(key) or "").strip()
        if csv_setup and csv_setup != _resolved_setup_id_v27550(json_row):
            return False
    comparable = 0
    for key, value in json_row.items():
        if key not in csv_row or isinstance(value, (Mapping, list, tuple)):
            continue
        other = csv_row.get(key)
        if value in (None, "") and other in (None, ""):
            continue
        if isinstance(value, bool):
            equal = _bool_or_none(other) is value
        elif isinstance(value, (float, int)):
            number = _float_or_none(other)
            equal = number is not None and number == value
        else:
            equal = str(value if value is not None else "") == str(other or "")
        if not equal:
            return False
        comparable += 1
    # Extra physical fields on CSV must not be silently discarded because an
    # older JSON producer omitted them.
    for key in (
        "measurement_endpoint", "full_measurement_endpoint", "quality_endpoint",
        "runtime_precision_identity", "execution_precision", "precision",
        "repetition_index", "process_local_repetition_index", "repeat_idx", "repeat_index",
    ):
        if csv_row.get(key) not in (None, "") and json_row.get(key) in (None, ""):
            return False
    return comparable > 1


def normalize_benchmark_files(*, model_id: str, source_paths: Iterable[str | Path], source_contexts: Sequence[Mapping[str, Any]] = (), write_diagnostic_summaries: bool = False, record_completion_errors: bool = False) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    normalized: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    discovered = discover_result_files(source_paths)
    summary_outcomes: dict[Path, dict[str, Any]] = {}
    json_rows_by_path = {}
    for path in discovered:
        if path.suffix.lower() == ".json":
            outcome: dict[str, Any] | None = {} if write_diagnostic_summaries else None
            json_rows_by_path[path] = _rows_from_json(path, diagnostic_summary=outcome)
            if outcome is not None:
                summary_outcomes[path] = outcome
    for path in discovered:
        if path.name == "validation_report.json" and path.parent.name.startswith("results_"):
            tag = path.parent.name.replace("results_", "", 1)
        else:
            tag = path.stem.replace("benchmark_results_", 1) if False else path.stem.replace("benchmark_results_", "", 1)
        rows = json_rows_by_path[path] if path.suffix.lower() == ".json" else _read_csv(path)
        source = {"path": str(path), "tag": tag, "row_count": len(rows)}
        if path in summary_outcomes:
            source["diagnostic_summary"] = summary_outcomes[path]
        sources.append(source)
        if path.suffix.lower() == ".csv":
            sibling_rows = json_rows_by_path.get(path.with_suffix(".json"), [])
            original_count = len(rows)
            rows = [
                row for row in rows
                if sum(
                    _csv_mirrors_json_row_v27927(row, candidate, tag=tag)
                    for candidate in sibling_rows
                ) != 1
            ]
            source["suppressed_csv_mirror_count"] = original_count - len(rows)
        for row in rows:
            for nrow in expand_normalized_benchmark_rows(row, model_id=model_id, source_path=path, tag=tag, record_completion_errors=record_completion_errors):
                nrow.setdefault("source_path", str(path))
                nrow.setdefault("source_tag", tag)
                if nrow.get("normalization_error"):
                    source.setdefault("normalization_errors", []).append({
                        "variant": nrow.get("variant"),
                        "case_id": nrow.get("case_id"),
                        "source_row_sha256": nrow.get("source_row_sha256"),
                        "reason": nrow["normalization_error"],
                    })
                normalized.append(bind_benchmark_source_context(nrow, source_contexts))
    canonical = _drop_spurious_full_placeholders_v58q(
        _dedupe_normalized_rows_v58f(normalized)
    )
    return annotate_logical_measurements(canonical), sources


def _native_evidence_status_v272(
    run_root: Optional[Path],
    explicit: Optional[Mapping[str, Any]],
) -> dict[str, Any]:
    if isinstance(explicit, Mapping):
        return dict(explicit)
    if run_root is None:
        return {}
    try:
        payload = json.loads(
            (
                Path(run_root)
                / "reports"
                / "native_evidence_status.json"
            ).read_text(encoding="utf-8")
        )
    except (OSError, ValueError, TypeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def build_normalized_results_payload(
    *,
    model_id: str,
    results: Sequence[Mapping[str, Any]],
    sources: Sequence[Mapping[str, Any]],
    run_root: Optional[Path] = None,
    native_evidence_status: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    source_records: list[dict[str, Any]] = []
    for src in sources:
        rec = dict(src or {})
        if run_root is not None and rec.get("path"):
            rec["path_rel"] = relpath(str(rec.get("path")), run_root)
        source_records.append(rec)
    enriched_results, energy_row_merge_summary = _augment_results_with_target_energy_v59j(results, model_id=model_id, run_root=run_root)
    native_evidence = _native_evidence_status_v272(
        run_root, native_evidence_status,
    )
    return {
        "schema": "onnx-splitpoint/normalized-benchmark-results",
        "schema_version": 2,
        "model_id": model_id,
        "created_at": now_iso(),
        "results": enriched_results,
        "result_count": len(enriched_results),
        "status": "measured" if enriched_results else "pending_benchmark_execution",
        "sources": source_records,
        "required_result_fields": list(REQUIRED_NORMALIZED_FIELDS),
        "energy_row_merge_summary": energy_row_merge_summary,
        "native_evidence_status": native_evidence,
        "native_evidence_summary": project_native_evidence_status(
            native_evidence
        ),
    }
