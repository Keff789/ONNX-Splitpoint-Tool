from __future__ import annotations

"""Hardware-smoke binding for formal Evaluation Workflow runs.

v49i makes hardware readiness visible without fabricating measurements or
launching expensive Hailo/DFC work.  It records local runtime probes, Hailo HEF
reuse/build status, remote dispatch status, and whether normalized runtime
measurements actually exist.
"""

import importlib.util
import shutil
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .artifacts import now_iso, read_json, relpath, write_csv, write_json, write_text
from .full_only_quality_canary import resolve_full_only_quality_canary


def _as_list(value: Any) -> List[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    return []


def _canon_backend(value: Any) -> str:
    s = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if "deepx" in s or "dx_m1" in s or "dxm1" in s:
        return "deepx_m1"
    if "hailo10" in s:
        return "hailo10"
    if "hailo8" in s or s == "hailo" or "hailo" in s:
        return "hailo8"
    if "trt" in s or "tensorrt" in s:
        return "tensorrt"
    if "cuda" in s or s == "gpu":
        return "cuda_ort"
    if "cpu" in s:
        return "cpu_ort"
    return s


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def _which(names: Sequence[str]) -> Dict[str, str]:
    return {name: (shutil.which(name) or "") for name in names}


def _int(value: Any, default: int = 0) -> int:
    try:
        if value is None or str(value).strip() == "":
            return int(default)
        return int(float(str(value).strip()))
    except Exception:
        return int(default)


def _bool_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "ok", "pass", "passed", "success"}




def _variant_ok(row: Mapping[str, Any], name: str) -> bool:
    variant_status = row.get("variant_status") if isinstance(row.get("variant_status"), Mapping) else {}
    measured_variants = {str(x).lower() for x in _as_list(row.get("measured_variants"))}
    return str(variant_status.get(name) or "").lower() in {"ok", "measured", "success", "passed"} or name in measured_variants


def _row_has_hailo_full_runtime_evidence(row: Mapping[str, Any]) -> bool:
    backend = str(row.get("backend") or "").lower()
    fullp = str(row.get("full_provider") or row.get("full_backend") or "").lower()
    if not (backend.startswith("hailo") or fullp.startswith("hailo")):
        return False
    if row.get("total_latency_ms") not in (None, "") and str(row.get("variant") or "").lower() == "full":
        return _variant_ok(row, "full") or _bool_true(row.get("runtime_ok")) or _bool_true(row.get("hailo_full_runtime_ok"))
    if row.get("full_e2e_latency_ms") not in (None, "") or row.get("full_latency_ms") not in (None, ""):
        return _variant_ok(row, "full") or _bool_true(row.get("hailo_full_runtime_ok"))
    return False


def _row_has_hailo_composed_runtime_evidence(row: Mapping[str, Any]) -> bool:
    backend = str(row.get("backend") or "").lower()
    if "hailo" not in backend:
        return False
    if row.get("total_latency_ms") in (None, ""):
        return False
    if str(row.get("variant") or "").lower() != "split":
        return False
    stage1 = str(row.get("stage1_provider") or "").lower()
    stage2 = str(row.get("stage2_provider") or "").lower()
    if not (stage1.startswith("hailo") or stage2.startswith("hailo") or "hailo" in backend):
        return False
    return _variant_ok(row, "composed") or _bool_true(row.get("hailo_composed_runtime_ok")) or _bool_true(row.get("runtime_ok"))


def _row_has_hailo_runtime_evidence(row: Mapping[str, Any]) -> bool:
    """Return true only for rows that actually executed a Hailo runtime stage."""
    if _row_has_hailo_full_runtime_evidence(row) or _row_has_hailo_composed_runtime_evidence(row):
        return True
    backend = str(row.get("backend") or "").lower()
    if "hailo" not in backend:
        return False
    stage1 = str(row.get("stage1_provider") or "").lower()
    stage2 = str(row.get("stage2_provider") or "").lower()
    if stage1.startswith("hailo") and row.get("part1_latency_ms") not in (None, "") and _variant_ok(row, "part1"):
        return True
    if stage2.startswith("hailo") and row.get("part2_latency_ms") not in (None, "") and _variant_ok(row, "part2"):
        return True
    component_status = str(row.get("component_measurement_status") or "").lower()
    if backend in {"hailo", "hailo8", "hailo10"} and _bool_true(row.get("runtime_ok")) and component_status in {"full", "full_raw_head_e2e", "composed", "split_parts"}:
        return True
    return False

def _status_text(payload: Mapping[str, Any], fallback: str = "unknown") -> str:
    return str(payload.get("status") or payload.get("reason") or fallback).strip() or fallback


def _status_rank(status: str) -> int:
    s = str(status or "").lower()
    if s in {"ok", "passed", "hardware_verified", "artifacts_ready_pending_runtime", "not_applicable_no_hardware_target", "not_applicable", "not_requested", "disabled_by_workflow_option"}:
        return 0
    if s in {"partial", "pending", "pending_execution", "pending_configuration", "pending_hailo_artifacts_or_remote_execution", "pending_hardware_execution"}:
        return 1
    if s in {"failed", "failed_to_dispatch", "timeout"}:
        return 2
    return 1


def _overall(checks: Sequence[Mapping[str, Any]], default: str) -> str:
    if default in {"disabled_by_workflow_option", "not_applicable_no_hardware_target", "hardware_verified"}:
        return default
    worst = max((_status_rank(str(c.get("status") or "")) for c in checks), default=1)
    if worst >= 2:
        return "failed"
    if worst == 1:
        return default if str(default).startswith("pending") else "partial"
    return default


def _verified_full_only_quality_evidence(
    *,
    profile_payload: Optional[Mapping[str, Any]],
    normalized: Mapping[str, Any],
    validation: Mapping[str, Any],
    remote_matrix: Mapping[str, Any],
) -> Dict[str, Any]:
    """Verify the Quality-only exception without making a runtime claim.

    Hardware smoke normally requires normalized performance rows.  A sealed
    Full-only Quality Canary deliberately has none, so its separate exact
    Central Quality evidence is accepted only when every archived contract
    agrees.  Any missing, stale or ambiguous field falls through to the
    ordinary fail-closed hardware-smoke path.
    """

    contract = resolve_full_only_quality_canary(
        dict(profile_payload or {}), plan_rows=None,
    )
    contract_identities = [
        dict(row) for row in _as_list(
            contract.get("expected_full_quality_identities")
        ) if isinstance(row, Mapping)
    ]
    expected_contract_count = len(contract_identities)

    def quality_identity(row: Mapping[str, Any]) -> tuple[Any, ...]:
        return (
            str(
                row.get("source_run_id") or row.get("run_id") or ""
            ).strip(),
            str(row.get("setup_id") or "").strip(),
            _canon_backend(row.get("backend")),
            str(row.get("variant") or "").strip().lower(),
            str(row.get("execution_role") or "").strip().lower(),
            row.get("performance_claims_emitted"),
        )

    contract_identity_keys = [
        quality_identity(row) for row in contract_identities
    ]
    normalized_rows = [
        row for row in _as_list(normalized.get("results"))
        if isinstance(row, Mapping)
    ]
    normalized_expected = _int(
        normalized.get("expected_full_quality_count"), 0,
    )
    normalized_count = _int(normalized.get("quality_evidence_count"), 0)
    validation_expected = _int(
        validation.get("expected_full_quality_count"), 0,
    )
    validation_count = _int(validation.get("quality_evidence_count"), 0)
    remote_expected = _int(
        remote_matrix.get("expected_full_quality_count"), 0,
    )
    remote_count = _int(remote_matrix.get("quality_evidence_count"), 0)
    dispatches = [
        dict(row) for row in _as_list(remote_matrix.get("dispatches"))
        if isinstance(row, Mapping)
    ]
    dispatches_exact = bool(dispatches)
    dispatched_identity_keys: List[tuple[Any, ...]] = []
    for dispatch in dispatches:
        metrics = (
            dict(dispatch.get("metrics") or {})
            if isinstance(dispatch.get("metrics"), Mapping) else {}
        )
        identities = [
            row for row in _as_list(
                dispatch.get("expected_full_quality_identities")
            ) if isinstance(row, Mapping)
        ]
        dispatched_identity_keys.extend(
            quality_identity(row) for row in identities
        )
        dispatches_exact = bool(
            dispatches_exact
            and str(dispatch.get("status") or "").strip().lower()
            in {"ok", "success", "completed"}
            and identities
            and str(metrics.get("quality_evidence_status") or "")
            .strip().lower() == "verified_exact"
            and not _as_list(metrics.get("quality_evidence_errors"))
            and _int(metrics.get("quality_evidence_count"), 0)
            == len(identities)
        )
    dispatches_exact = bool(
        dispatches_exact
        and len(dispatched_identity_keys) == expected_contract_count
        and sorted(dispatched_identity_keys) == sorted(contract_identity_keys)
    )

    verified = bool(
        contract.get("enabled") is True
        and contract.get("ok") is True
        and str(contract.get("execution_scope") or "").strip().lower()
        == "full_only"
        and contract.get("performance_claims_emitted") is False
        and expected_contract_count > 0
        and not normalized_rows
        and _int(normalized.get("result_count"), 0) == 0
        and normalized.get("performance_matrix_applicable") is False
        and normalized.get("quality_evidence_only_complete") is True
        and str(normalized.get("status") or "").strip().lower()
        == "quality_evidence_only_complete"
        and normalized.get("matrix_complete") is True
        and normalized_count == normalized_expected == expected_contract_count
        and str(validation.get("status") or "").strip().lower()
        == "not_applicable_quality_evidence_only"
        and validation.get("performance_matrix_applicable") is False
        and validation.get("quality_evidence_only_complete") is True
        and validation_count == validation_expected == expected_contract_count
        and str(remote_matrix.get("status") or "").strip().lower()
        in {"ok", "success", "completed"}
        and remote_matrix.get("matrix_complete") is True
        and remote_matrix.get("performance_matrix_applicable") is False
        and remote_matrix.get("quality_evidence_only_complete") is True
        and remote_count == remote_expected == expected_contract_count
        and dispatches_exact
    )
    return {
        "verified": verified,
        "expected_count": expected_contract_count,
        "quality_evidence_count": normalized_count,
        "dispatch_count": len(dispatches),
        "contract_status": str(contract.get("status") or ""),
    }


def materialize_hardware_smoke_status(
    *,
    run_dir: str | Path,
    model_id: str,
    targets: Sequence[Any],
    options: Any,
    profile_payload: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    root = Path(run_dir)
    model_dir = root / "models" / str(model_id)
    hardware_dir = model_dir / "hardware"
    val_dir = model_dir / "validation"
    bset_dir = model_dir / "benchmark_set"
    bres_dir = model_dir / "benchmark_results"
    for d in (hardware_dir, val_dir, bres_dir):
        d.mkdir(parents=True, exist_ok=True)

    mode = str(getattr(options, "hardware_smoke_mode", "summary_only") or "summary_only").strip().lower()
    if bool(getattr(options, "skip_hardware_smoke", False)) or mode in {"disabled", "off", "none"}:
        summary = {
            "schema": "onnx-splitpoint/hardware-smoke-summary",
            "schema_version": 1,
            "created_at": now_iso(),
            "model_id": model_id,
            "targets": list(targets),
            "status": "disabled",
            "hardware_verified": False,
            "hardware_smoke_mode": mode,
            "measured_hardware_result_count": 0,
            "runtime_ok_hardware_result_count": 0,
            "notes": ["Hardware smoke disabled by workflow option."],
        }
        p_json = write_json(hardware_dir / "hardware_smoke_status.json", summary)
        p_md = write_text(hardware_dir / "hardware_smoke_status.md", _md(summary))
        p_val = write_json(val_dir / "hardware_smoke_summary.json", summary)
        p_bres = write_json(bres_dir / "hardware_smoke_report.json", summary)
        return {
            "artifacts": {
                "hardware_smoke_status_json": p_json,
                "hardware_smoke_status_md": p_md,
                "validation_hardware_smoke_summary_json": p_val,
                "benchmark_results_hardware_smoke_report_json": p_bres,
            },
            "metrics": {"hardware_smoke_status": "disabled", "hardware_verified": False},
            "status": "skipped",
            "message": "Hardware smoke disabled by workflow option.",
        }

    model_manifest = read_json(model_dir / "model_manifest.json", default={}) or {}
    mode = str(getattr(options, "hardware_smoke_mode", "summary_only") or "summary_only").strip().lower()
    if mode == "disabled" or bool(getattr(options, "skip_hardware_smoke", False)):
        summary = {
            "schema": "onnx-splitpoint/hardware-smoke-summary",
            "schema_version": 1,
            "created_at": now_iso(),
            "model_id": model_id,
            "targets": list(targets),
            "status": "disabled",
            "hardware_verified": False,
            "measured_hardware_result_count": 0,
            "runtime_ok_hardware_result_count": 0,
            "hailo_result_count": 0,
            "remote_result_count": 0,
            "hailo": {"target_present": any(_canon_backend(t).startswith("hailo") for t in targets), "artifact_status": "not_evaluated_disabled"},
            "remote": {"status": "not_evaluated_disabled", "dispatched": False},
            "checks": [],
            "notes": ["Hardware smoke disabled by workflow option."],
        }
        p_json = write_json(hardware_dir / "hardware_smoke_status.json", summary)
        p_md = write_text(hardware_dir / "hardware_smoke_status.md", _md(summary))
        p_val = write_json(val_dir / "hardware_smoke_summary.json", summary)
        p_bres = write_json(bres_dir / "hardware_smoke_report.json", summary)
        return {
            "artifacts": {
                "hardware_smoke_status_json": p_json,
                "hardware_smoke_status_md": p_md,
                "validation_hardware_smoke_summary_json": p_val,
                "benchmark_results_hardware_smoke_report_json": p_bres,
            },
            "metrics": {"hardware_smoke_status": "disabled", "hardware_verified": False, "measured_hardware_result_count": 0, "runtime_ok_hardware_result_count": 0, "remote_status": "not_evaluated_disabled"},
            "status": "skipped",
            "message": "Hardware smoke disabled by workflow option.",
        }
    normalized = read_json(bres_dir / "normalized_results.json", default={}) or {}
    rows = [dict(x or {}) for x in _as_list(normalized.get("results")) if isinstance(x, Mapping)]
    validation = read_json(val_dir / "validation_summary.json", default={}) or {}
    hailo_status = read_json(bset_dir / "hailo_artifact_status.json", default={}) or {}
    hailo_plan = read_json(bset_dir / "hailo_artifact_service_plan.json", default={}) or {}
    hailo_queue = read_json(bset_dir / "hailo_build_queue.json", default={}) or {}
    hailo_dispatch = read_json(bset_dir / "hailo_service_dispatch.json", default={}) or {}
    # v58f: DeepX artefact status lives under benchmark_set/deepx/ in the
    # current suite layout.  Keep the legacy root paths as fallback only.
    deepx_status_path = bset_dir / "deepx" / "deepx_artifact_status.json"
    deepx_env_path = bset_dir / "deepx" / "deepx_environment_status.json"
    if not deepx_status_path.exists():
        deepx_status_path = bset_dir / "deepx_artifact_status.json"
    if not deepx_env_path.exists():
        deepx_env_path = bset_dir / "deepx_environment_status.json"
    deepx_status = read_json(deepx_status_path, default={}) or {}
    deepx_env = read_json(deepx_env_path, default={}) or {}
    local_exec = read_json(bres_dir / "benchmark_executor_status.json", default={}) or {}
    remote_exec = read_json(bres_dir / "remote_benchmark_status.json", default={}) or {}
    remote_matrix = read_json(
        bres_dir / "remote_hardware_matrix_status.json", default={},
    ) or {}

    quality_evidence = _verified_full_only_quality_evidence(
        profile_payload=profile_payload,
        normalized=normalized,
        validation=validation,
        remote_matrix=remote_matrix,
    )
    if quality_evidence.get("verified") is True:
        expected_count = int(quality_evidence.get("expected_count") or 0)
        quality_count = int(
            quality_evidence.get("quality_evidence_count") or 0
        )
        checks = [
            {
                "id": "full_only_quality_canary_contract",
                "status": "ok",
                "evidence": "profile.yaml",
                "detail": "sealed Full-only Quality Canary profile resolved",
            },
            {
                "id": "result_ingestion",
                "status": "not_applicable",
                "evidence": relpath(
                    bres_dir / "normalized_results.json", root,
                ),
                "detail": "Generic/performance result rows are not applicable",
            },
            {
                "id": "hardware_runtime_result",
                "status": "not_applicable",
                "evidence": relpath(
                    bres_dir / "normalized_results.json", root,
                ),
                "detail": "No hardware-performance claim is emitted",
            },
            {
                "id": "quality_evidence",
                "status": "ok",
                "evidence": relpath(
                    bres_dir / "remote_hardware_matrix_status.json", root,
                ),
                "detail": (
                    f"exact Full quality identities={quality_count}/"
                    f"{expected_count}"
                ),
            },
            {
                "id": "remote_dispatch",
                "status": "ok",
                "evidence": relpath(
                    bres_dir / "remote_hardware_matrix_status.json", root,
                ),
                "detail": (
                    "remote matrix status ok; all quality dispatches "
                    "verified_exact"
                ),
            },
            {
                "id": "validation_binding",
                "status": "not_applicable",
                "evidence": relpath(
                    val_dir / "validation_summary.json", root,
                ),
                "detail": (
                    "Generic validation is N/A; Central Quality evidence "
                    "is exact"
                ),
            },
            {
                "id": "hailo_runtime_result",
                "status": "not_applicable",
                "evidence": "",
                "detail": "Quality-evidence-only projection",
            },
            {
                "id": "deepx_runtime_result",
                "status": "not_applicable",
                "evidence": "",
                "detail": "Quality-evidence-only projection",
            },
        ]
        summary = {
            "schema": "onnx-splitpoint/hardware-smoke-summary",
            "schema_version": 1,
            "created_at": now_iso(),
            "model_id": model_id,
            "targets": list(targets),
            "status": "not_applicable_quality_evidence_only",
            "hardware_smoke_mode": mode,
            "performance_matrix_applicable": False,
            "hardware_verified": False,
            "quality_evidence_verified": True,
            "quality_evidence_count": quality_count,
            "expected_full_quality_count": expected_count,
            "normalized_result_count": 0,
            "measured_hardware_result_count": 0,
            "runtime_ok_hardware_result_count": 0,
            "hailo_result_count": 0,
            "hailo_runtime_ok_result_count": 0,
            "hailo_runtime_verified": False,
            "hailo_full_runtime_verified": False,
            "hailo_composed_runtime_verified": False,
            "deepx_result_count": 0,
            "deepx_runtime_ok_result_count": 0,
            "deepx_runtime_verified": False,
            "deepx_full_runtime_verified": False,
            "deepx_composed_runtime_verified": False,
            "remote_result_count": 0,
            "remote": {
                "status": str(remote_matrix.get("status") or ""),
                "requested": True,
                "dispatched": True,
                "dispatch_count": int(
                    quality_evidence.get("dispatch_count") or 0
                ),
                "quality_evidence_verified": True,
                "status_path": relpath(
                    bres_dir / "remote_hardware_matrix_status.json", root,
                ),
            },
            "hailo": {
                "target_present": any(
                    _canon_backend(target).startswith("hailo")
                    for target in targets
                ),
                "runtime_verified": False,
                "status": "not_applicable_quality_evidence_only",
            },
            "deepx": {
                "target_present": any(
                    _canon_backend(target).startswith("deepx")
                    for target in targets
                ),
                "runtime_verified": False,
                "status": "not_applicable_quality_evidence_only",
            },
            "checks": checks,
            "notes": [
                "The Full-only Quality Canary intentionally emits no "
                "Generic or performance measurements.",
                "Exact Central Quality evidence completes the workflow "
                "without asserting hardware-performance verification.",
            ],
        }
        p_json = write_json(
            hardware_dir / "hardware_smoke_status.json", summary,
        )
        p_md = write_text(
            hardware_dir / "hardware_smoke_status.md", _md(summary),
        )
        p_checks = write_csv(
            hardware_dir / "hardware_smoke_checks.csv", checks,
            ["id", "status", "evidence", "detail"],
        )
        p_val = write_json(
            val_dir / "hardware_smoke_summary.json", summary,
        )
        p_bres = write_json(
            bres_dir / "hardware_smoke_report.json", summary,
        )
        return {
            "artifacts": {
                "hardware_smoke_status_json": p_json,
                "hardware_smoke_status_md": p_md,
                "hardware_smoke_checks_csv": p_checks,
                "validation_hardware_smoke_summary_json": p_val,
                "benchmark_results_hardware_smoke_report_json": p_bres,
            },
            "metrics": {
                "hardware_smoke_status": (
                    "not_applicable_quality_evidence_only"
                ),
                "hardware_verified": False,
                "quality_evidence_verified": True,
                "quality_evidence_count": quality_count,
                "expected_full_quality_count": expected_count,
                "performance_matrix_applicable": False,
                "measured_hardware_result_count": 0,
                "runtime_ok_hardware_result_count": 0,
            },
            "status": "ok",
            "message": (
                "Full-only Quality evidence verified exactly; hardware "
                "performance smoke is not applicable."
            ),
        }

    canon_targets = [_canon_backend(t) for t in targets]
    target_has_hailo = any(t.startswith("hailo") for t in canon_targets)
    target_has_deepx = any(t.startswith("deepx") or t == "dx_m1" for t in canon_targets)
    result_count = int(normalized.get("result_count") or len(rows) or 0)
    detected_hef_count = _int(hailo_status.get("detected_hef_count"), len(_as_list(hailo_plan.get("detected_hefs"))))
    ready_full = _int(hailo_status.get("ready_full_baselines"), 0)
    pending_full = _int(hailo_status.get("pending_full_baselines"), 0)
    ready_case = _int(hailo_status.get("ready_case_hefs"), 0)
    pending_case = _int(hailo_status.get("pending_case_hefs"), 0)
    queue_count = _int(hailo_queue.get("queue_count"), pending_full + pending_case)
    expected_unsupported = _int(hailo_status.get("expected_unsupported_count"), len(_as_list(hailo_plan.get("expected_unsupported"))))

    measured_hardware_rows = [r for r in rows if any(tok in str(r.get("backend") or "").lower() for tok in ("hailo", "deepx", "dx_m1", "cuda", "trt", "tensorrt"))]
    runtime_ok_hardware_rows = [r for r in measured_hardware_rows if _bool_true(r.get("runtime_ok"))]
    hailo_rows = [r for r in rows if "hailo" in str(r.get("backend") or "").lower()]
    # A row may mention hailo8_to_tensorrt but only contain TensorRT full/part2
    # timings when the Hailo runtime is unavailable.  Require explicit evidence
    # that a Hailo full/part/composed variant actually ran.
    hailo_runtime_rows = [r for r in hailo_rows if _row_has_hailo_runtime_evidence(r)]
    hailo_full_runtime_rows = [r for r in hailo_rows if _row_has_hailo_full_runtime_evidence(r)]
    hailo_composed_runtime_rows = [r for r in hailo_rows if _row_has_hailo_composed_runtime_evidence(r)]
    hailo_component_only_rows = [r for r in hailo_rows if not _row_has_hailo_runtime_evidence(r)]
    deepx_rows = [r for r in rows if any(tok in str(r.get("backend") or "").lower() for tok in ("deepx", "dx_m1"))]
    deepx_full_runtime_rows = [r for r in deepx_rows if str(r.get("variant") or "").lower() == "full" and r.get("total_latency_ms") not in (None, "") and _bool_true(r.get("runtime_ok"))]
    deepx_composed_runtime_rows = [r for r in deepx_rows if str(r.get("variant") or "").lower() == "split" and r.get("total_latency_ms") not in (None, "") and _bool_true(r.get("runtime_ok"))]
    deepx_runtime_rows = deepx_full_runtime_rows + deepx_composed_runtime_rows
    remote_rows = [r for r in rows if "remote" in str(r.get("source_path") or "").lower()]

    no_remote = bool(getattr(options, "no_remote", False))
    execution_backend = str(getattr(options, "benchmark_execution_backend", "auto") or "auto")
    # v55i: Evaluation profiles now use the central hardware setup registry,
    # not the old single-remote fields.  Treat hardware_setup dispatch reports
    # and measured remote/hardware rows as host evidence so hardware-smoke does
    # not stay partial merely because legacy remote_host is empty.
    selected_setups = []
    try:
        hw = (profile_payload or {}).get("hardware") if isinstance(profile_payload, Mapping) else {}
        if isinstance(hw, Mapping):
            selected_setups = list(hw.get("selected_setups") or [])
    except Exception:
        selected_setups = []
    remote_dispatches = []
    if isinstance(remote_exec, Mapping):
        for key in ("dispatches", "remote_dispatches", "hardware_dispatches", "runs"):
            val = remote_exec.get(key)
            if isinstance(val, list):
                remote_dispatches.extend([x for x in val if isinstance(x, Mapping)])
    registry_host_configured = bool(selected_setups)
    remote_exec_dispatched = bool(
        isinstance(remote_exec, Mapping)
        and (remote_exec.get("dispatched") or remote_exec.get("remote_dispatched") or remote_exec.get("run_group_dir") or remote_dispatches)
    )
    remote_host_configured = bool(
        getattr(options, "remote_host", "")
        or getattr(options, "remote_host_json", "")
        or getattr(options, "remote_hosts_file", "")
        or registry_host_configured
        or remote_exec_dispatched
        or remote_rows
    )
    remote_requested = (not no_remote) and (execution_backend == "remote" or target_has_hailo or target_has_deepx or remote_host_configured or bool(remote_exec))
    remote_status = _status_text(remote_exec, "not_requested" if no_remote else ("pending" if remote_requested else "not_applicable"))
    # If normalized runtime rows exist but the remote status file is missing/stale,
    # prefer the runtime evidence over legacy pending_configuration.
    if remote_requested and remote_status in {"pending", "not_requested", "not_applicable", "unknown"} and (runtime_ok_hardware_rows or remote_rows):
        remote_status = "completed_from_runtime_evidence"
    local_status = _status_text(local_exec, "not_requested")
    smoke_mode = str(getattr(options, "hardware_smoke_mode", "summary_only") or "summary_only").strip().lower()
    if bool(getattr(options, "skip_hardware_smoke", False)):
        smoke_mode = "disabled"

    local_probes = {
        "onnx_module": _module_available("onnx"),
        "onnxruntime_module": _module_available("onnxruntime"),
        "hailo_platform_module": _module_available("hailo_platform"),
        "hailort_module": _module_available("hailort"),
        "hailo_tools": _which(["hailortcli", "hailort", "hailo", "hailomz"]),
        "deepx_module": _module_available("dx_engine"),
        "deepx_tools": _which(["dxrt-cli", "parse_model", "run_model", "dxcom", "dx_com"]),
    }

    contradictory_quality_only_projection = bool(
        normalized.get("performance_matrix_applicable") is False
        or normalized.get("quality_evidence_only_complete") is True
        or str(normalized.get("status") or "").strip().lower().startswith(
            "quality_evidence_only_"
        )
    )

    if smoke_mode in {"disabled", "off", "none"}:
        base_status = "disabled_by_workflow_option"
        hardware_verified = False
    elif contradictory_quality_only_projection:
        # A claimed Quality-only projection that failed the exact exception
        # verifier above must not fall back to green performance evidence.  A
        # leaked runtime row is itself a contract contradiction.
        base_status = "pending_hardware_execution"
        hardware_verified = False
    elif runtime_ok_hardware_rows:
        base_status = "hardware_verified"
        hardware_verified = True
    elif target_has_deepx and (deepx_runtime_rows):
        base_status = "hardware_verified"
        hardware_verified = True
    elif target_has_hailo and (detected_hef_count or ready_full or ready_case):
        base_status = "artifacts_ready_pending_runtime"
        hardware_verified = False
    elif target_has_deepx and bool(deepx_status.get("ready")):
        base_status = "artifacts_ready_pending_runtime"
        hardware_verified = False
    elif target_has_hailo and (pending_full + pending_case + queue_count) > 0:
        base_status = "pending_hailo_artifacts_or_remote_execution"
        hardware_verified = False
    elif target_has_hailo or target_has_deepx:
        base_status = "pending_hardware_execution"
        hardware_verified = False
    else:
        base_status = "not_applicable_no_hardware_target"
        hardware_verified = False

    checks = [
        {"id": "model_file", "status": "ok" if bool(model_manifest.get("resolved")) else "pending", "evidence": relpath(model_dir / "model_manifest.json", root), "detail": "resolved ONNX path"},
        {"id": "result_ingestion", "status": "ok" if rows else "pending_execution", "evidence": relpath(bres_dir / "normalized_results.json", root), "detail": f"normalized_results={len(rows)}"},
        {"id": "hardware_runtime_result", "status": "ok" if runtime_ok_hardware_rows else ("pending_execution" if measured_hardware_rows or target_has_hailo else "not_applicable"), "evidence": relpath(bres_dir / "normalized_results.json", root), "detail": f"hardware_rows={len(measured_hardware_rows)}, runtime_ok={len(runtime_ok_hardware_rows)}"},
        {"id": "hailo_runtime_result", "status": "ok" if hailo_runtime_rows else ("pending_execution" if target_has_hailo else "not_applicable"), "evidence": relpath(bres_dir / "normalized_results.json", root), "detail": f"hailo_rows={len(hailo_rows)}, hailo_runtime_ok={len(hailo_runtime_rows)}, full={len(hailo_full_runtime_rows)}, composed={len(hailo_composed_runtime_rows)}, component_only={len(hailo_component_only_rows)}"},
        {"id": "hailo_full_runtime_result", "status": "ok" if hailo_full_runtime_rows else ("pending_execution" if target_has_hailo else "not_applicable"), "evidence": relpath(bres_dir / "normalized_results.json", root), "detail": f"hailo_full_runtime_ok={len(hailo_full_runtime_rows)}"},
        {"id": "hailo_composed_runtime_result", "status": "ok" if hailo_composed_runtime_rows else ("pending_execution" if target_has_hailo else "not_applicable"), "evidence": relpath(bres_dir / "normalized_results.json", root), "detail": f"hailo_composed_runtime_ok={len(hailo_composed_runtime_rows)}"},
        {"id": "hailo_hef_reuse", "status": "ok" if (ready_full or ready_case or detected_hef_count) else ("pending" if target_has_hailo else "not_applicable"), "evidence": relpath(bset_dir / "hailo_artifact_status.json", root), "detail": f"detected_hefs={detected_hef_count}, ready_full={ready_full}, ready_case={ready_case}"},
        {"id": "hailo_pending_queue", "status": "ok" if (not target_has_hailo or (pending_full + pending_case) == 0) else "pending", "evidence": relpath(bset_dir / "hailo_build_queue.json", root), "detail": f"pending_full={pending_full}, pending_case={pending_case}, queue={queue_count}"},
        {"id": "deepx_artifact_ready", "status": "ok" if bool(deepx_status.get("ready")) else ("pending" if target_has_deepx else "not_applicable"), "evidence": relpath(deepx_status_path, root), "detail": f"status={deepx_status.get('status','')}, compiler_ready={deepx_env.get('compiler_ready','')}, runtime_ready={deepx_env.get('runtime_ready','')}"},
        {"id": "deepx_runtime_result", "status": "ok" if deepx_runtime_rows else ("pending_execution" if target_has_deepx else "not_applicable"), "evidence": relpath(bres_dir / "normalized_results.json", root), "detail": f"deepx_rows={len(deepx_rows)}, full={len(deepx_full_runtime_rows)}, composed={len(deepx_composed_runtime_rows)}"},
        {"id": "remote_dispatch", "status": "ok" if remote_status in {"ok", "completed", "success", "measured", "completed_from_runtime_evidence"} else ("not_requested" if no_remote else ("pending_configuration" if remote_requested and not remote_host_configured else "pending")), "evidence": relpath(bres_dir / "remote_benchmark_status.json", root), "detail": f"remote_status={remote_status}, host_configured={remote_host_configured}, selected_setups={len(selected_setups)}, dispatches={len(remote_dispatches)}"},
        {"id": "validation_binding", "status": "ok" if _int(validation.get("validated_result_count"), 0) and not _int(validation.get("invalid_result_count"), 0) else ("pending_execution" if not rows else "partial"), "evidence": relpath(val_dir / "validation_summary.json", root), "detail": f"validated={validation.get('validated_result_count', 0)}, invalid={validation.get('invalid_result_count', 0)}"},
        {"id": "local_runtime_modules", "status": "ok" if local_probes["onnx_module"] else "pending", "evidence": "importlib/shutil probes", "detail": str(local_probes)},
    ]
    status = _overall(checks, base_status)

    summary = {
        "schema": "onnx-splitpoint/hardware-smoke-summary",
        "schema_version": 1,
        "created_at": now_iso(),
        "model_id": model_id,
        "targets": list(targets),
        "status": status,
        "hardware_smoke_mode": smoke_mode,
        "hardware_verified": hardware_verified,
        "hailo_runtime_verified": bool(hailo_runtime_rows),
        "hailo_full_runtime_verified": bool(hailo_full_runtime_rows),
        "hailo_composed_runtime_verified": bool(hailo_composed_runtime_rows),
        "deepx_runtime_verified": bool(deepx_runtime_rows),
        "deepx_full_runtime_verified": bool(deepx_full_runtime_rows),
        "deepx_composed_runtime_verified": bool(deepx_composed_runtime_rows),
        "normalized_result_count": result_count,
        "measured_hardware_result_count": len(measured_hardware_rows),
        "runtime_ok_hardware_result_count": len(runtime_ok_hardware_rows),
        "hailo_result_count": len(hailo_rows),
        "hailo_runtime_ok_result_count": len(hailo_runtime_rows),
        "hailo_full_runtime_ok_result_count": len(hailo_full_runtime_rows),
        "hailo_composed_runtime_ok_result_count": len(hailo_composed_runtime_rows),
        "hailo_component_only_result_count": len(hailo_component_only_rows),
        "deepx_result_count": len(deepx_rows),
        "deepx_runtime_ok_result_count": len(deepx_runtime_rows),
        "deepx_full_runtime_ok_result_count": len(deepx_full_runtime_rows),
        "deepx_composed_runtime_ok_result_count": len(deepx_composed_runtime_rows),
        "remote_result_count": len(remote_rows),
        "local_probes": local_probes,
        "hailo": {
            "target_present": target_has_hailo,
            "artifact_status": hailo_status.get("status", hailo_plan.get("status", "unknown")),
            "service_dispatch_status": hailo_dispatch.get("status", "unknown"),
            "service_dispatch_reason": hailo_dispatch.get("reason", ""),
            "detected_hef_count": detected_hef_count,
            "ready_full_baselines": ready_full,
            "pending_full_baselines": pending_full,
            "ready_case_hefs": ready_case,
            "pending_case_hefs": pending_case,
            "queue_count": queue_count,
            "expected_unsupported_count": expected_unsupported,
            "runtime_verified": bool(hailo_runtime_rows),
            "full_runtime_verified": bool(hailo_full_runtime_rows),
            "composed_runtime_verified": bool(hailo_composed_runtime_rows),
            "runtime_ok_result_count": len(hailo_runtime_rows),
            "full_runtime_ok_result_count": len(hailo_full_runtime_rows),
            "composed_runtime_ok_result_count": len(hailo_composed_runtime_rows),
            "component_only_result_count": len(hailo_component_only_rows),
            "status_path": relpath(bset_dir / "hailo_artifact_status.json", root),
            "service_plan_path": relpath(bset_dir / "hailo_artifact_service_plan.json", root),
            "dispatch_path": relpath(bset_dir / "hailo_service_dispatch.json", root),
        },
        "deepx": {
            "target_present": target_has_deepx,
            "artifact_status": deepx_status.get("status", "unknown"),
            "environment_status": deepx_env.get("status", "unknown"),
            "compiler_ready": bool(deepx_env.get("compiler_ready")),
            "runtime_ready": bool(deepx_env.get("runtime_ready")),
            "artifact_ready": bool(deepx_status.get("ready")),
            "runtime_verified": bool(deepx_runtime_rows),
            "full_runtime_verified": bool(deepx_full_runtime_rows),
            "composed_runtime_verified": bool(deepx_composed_runtime_rows),
            "runtime_ok_result_count": len(deepx_runtime_rows),
            "full_runtime_ok_result_count": len(deepx_full_runtime_rows),
            "composed_runtime_ok_result_count": len(deepx_composed_runtime_rows),
            "status_path": relpath(deepx_status_path, root),
            "environment_path": relpath(deepx_env_path, root),
        },
        "remote": {
            "status": remote_status,
            "reason": remote_exec.get("reason", local_exec.get("reason", "")) if isinstance(remote_exec, Mapping) else "",
            "requested": remote_requested,
            "dispatched": bool(remote_exec.get("dispatched") or remote_exec.get("remote_dispatched") or remote_status in {"ok", "completed", "success"}),
            "host_configured": remote_host_configured,
            "selected_setup_count": len(selected_setups),
            "dispatch_count": len(remote_dispatches),
            "runtime_evidence_fallback": remote_status == "completed_from_runtime_evidence",
            "status_path": relpath(bres_dir / "remote_benchmark_status.json", root),
            "executor_status": local_status,
            "executor_status_path": relpath(bres_dir / "benchmark_executor_status.json", root),
        },
        "hailo_reuse": {
            "service_status": hailo_status.get("status", hailo_plan.get("status", "")),
            "detected_hef_existing_count": detected_hef_count,
            "queue_count": queue_count,
        },
        "remote_connectivity": {
            "status": remote_status,
            "message": remote_exec.get("reason", local_exec.get("reason", "")) if isinstance(remote_exec, Mapping) else "",
            "requested": remote_requested,
        },
        "summary": {
            "detected_hef_existing_count": detected_hef_count,
            "ready_declared_hailo_contracts": ready_full,
            "target_count": len(canon_targets),
        },
        "checks": checks,
        "notes": [
            "Hardware smoke is evidence-based and never marks hardware_verified without runtime rows.",
            "v49m separates generic hardware_verified from hailo_runtime_verified; CUDA/TensorRT success does not imply that Hailo executed.",
            "Hailo raw-head/full-HEF reuse readiness is separate from decoded-output validation semantics.",
            "v55i treats central hardware_setup dispatches as remote host configuration; legacy remote_host fields are no longer required.",
        ],
    }

    p_json = write_json(hardware_dir / "hardware_smoke_status.json", summary)
    p_md = write_text(hardware_dir / "hardware_smoke_status.md", _md(summary))
    p_checks = write_csv(hardware_dir / "hardware_smoke_checks.csv", checks, ["id", "status", "evidence", "detail"])
    # Compatibility/convenience mirrors used by aggregate/report views and old GUI snippets.
    p_val = write_json(val_dir / "hardware_smoke_summary.json", summary)
    p_bres = write_json(bres_dir / "hardware_smoke_report.json", summary)
    stage_status = "ok" if status in {"hardware_verified", "artifacts_ready_pending_runtime", "not_applicable_no_hardware_target", "disabled_by_workflow_option"} else "partial"
    return {
        "artifacts": {
            "hardware_smoke_status_json": p_json,
            "hardware_smoke_status_md": p_md,
            "hardware_smoke_checks_csv": p_checks,
            "validation_hardware_smoke_summary_json": p_val,
            "benchmark_results_hardware_smoke_report_json": p_bres,
        },
        "metrics": {
            "hardware_smoke_status": status,
            "hardware_verified": hardware_verified,
            "hailo_runtime_verified": bool(hailo_runtime_rows),
            "hailo_full_runtime_verified": bool(hailo_full_runtime_rows),
            "hailo_composed_runtime_verified": bool(hailo_composed_runtime_rows),
            "deepx_runtime_verified": bool(deepx_runtime_rows),
            "deepx_full_runtime_verified": bool(deepx_full_runtime_rows),
            "deepx_composed_runtime_verified": bool(deepx_composed_runtime_rows),
        "deepx_runtime_verified": bool(deepx_runtime_rows),
        "deepx_full_runtime_verified": bool(deepx_full_runtime_rows),
        "deepx_composed_runtime_verified": bool(deepx_composed_runtime_rows),
            "measured_hardware_result_count": len(measured_hardware_rows),
            "runtime_ok_hardware_result_count": len(runtime_ok_hardware_rows),
            "hailo_runtime_ok_result_count": len(hailo_runtime_rows),
            "hailo_full_runtime_ok_result_count": len(hailo_full_runtime_rows),
            "hailo_composed_runtime_ok_result_count": len(hailo_composed_runtime_rows),
            "deepx_runtime_ok_result_count": len(deepx_runtime_rows),
            "deepx_full_runtime_ok_result_count": len(deepx_full_runtime_rows),
            "deepx_composed_runtime_ok_result_count": len(deepx_composed_runtime_rows),
            "hailo_detected_hefs": detected_hef_count,
            "hailo_pending_artifacts": pending_full + pending_case,
            "remote_status": remote_status,
        },
        "status": "skipped" if status == "disabled_by_workflow_option" else stage_status,
        "message": "Hardware smoke summary materialized from Hailo/remote status and normalized runtime evidence.",
    }


def materialize_hardware_smoke_binding(
    *,
    run_dir: str | Path,
    model_id: str,
    targets: Sequence[Any] = (),
    options: Any = None,
    profile_payload: Optional[Mapping[str, Any]] = None,
    log: Any = None,
) -> Dict[str, Any]:
    return materialize_hardware_smoke_status(
        run_dir=run_dir,
        model_id=model_id,
        targets=targets,
        options=options,
        profile_payload=profile_payload,
    )


def _md(summary: Mapping[str, Any]) -> str:
    hailo = summary.get("hailo") if isinstance(summary.get("hailo"), Mapping) else {}
    remote = summary.get("remote") if isinstance(summary.get("remote"), Mapping) else {}
    lines = [
        "# Hardware Smoke Summary",
        "",
        f"Model: `{summary.get('model_id')}`",
        f"Status: `{summary.get('status')}`",
        f"Hardware verified: `{summary.get('hardware_verified')}`",
        f"Hailo runtime verified: `{summary.get('hailo_runtime_verified')}`",
        f"Hailo full runtime verified: `{summary.get('hailo_full_runtime_verified')}`",
        f"Hailo composed runtime verified: `{summary.get('hailo_composed_runtime_verified')}`",
        f"Smoke mode: `{summary.get('hardware_smoke_mode')}`",
        "",
        "## Hailo",
        "",
        f"- Target present: {hailo.get('target_present')}",
        f"- Artifact status: `{hailo.get('artifact_status')}`",
        f"- Detected HEFs: {hailo.get('detected_hef_count')}",
        f"- Ready full baselines: {hailo.get('ready_full_baselines')}",
        f"- Pending full baselines: {hailo.get('pending_full_baselines')}",
        f"- Ready case HEFs: {hailo.get('ready_case_hefs')}",
        f"- Pending case HEFs: {hailo.get('pending_case_hefs')}",
        f"- Expected unsupported records: {hailo.get('expected_unsupported_count')}",
        f"- Full runtime rows: {hailo.get('full_runtime_ok_result_count')}",
        f"- Composed runtime rows: {hailo.get('composed_runtime_ok_result_count')}",
        f"- Hailo runtime verified: {hailo.get('runtime_verified')}",
        f"- Hailo runtime OK rows: {hailo.get('runtime_ok_result_count')}",
        f"- Hailo component-only rows: {hailo.get('component_only_result_count')}",
        "",
        "## Remote",
        "",
        f"- Remote status: `{remote.get('status')}`",
        f"- Reason: `{remote.get('reason')}`",
        f"- Requested: {remote.get('requested')}",
        f"- Dispatched: {remote.get('dispatched')}",
        f"- Host configured: {remote.get('host_configured')}",
        "",
        "This file is a cheap readiness/status smoke. It does not launch DFC builds and does not fabricate latency measurements.",
        "",
    ]
    return "\n".join(lines)
