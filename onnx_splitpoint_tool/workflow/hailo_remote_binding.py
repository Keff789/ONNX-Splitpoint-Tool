from __future__ import annotations

"""Hailo artifact/reuse service plan for the formal Evaluation Workflow.

The formal runner should not hide Hailo work inside logs or rebuild expensive
HEFs blindly.  This module inspects the generated/imported suite after the
backend-artifact decision stage, records which Hailo artifacts are already
available, which full/case HEFs are still pending, and which YOLO raw-head cases
must be treated as host-tail contracts rather than decoded ONNX-equivalent
outputs.
"""

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from ..native_output_endpoint import load_authoritative_output_contract
from ..config_values import parse_config_bool
from .artifacts import now_iso, read_json, relpath, write_json, write_text


def _as_list(value: Any) -> List[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    return []


_MISSING_HAILO_BUILD_TARGETS = object()


def _hailo_build_targets_from_options(options: Any) -> List[str]:
    """Keep an explicit empty target list distinct from an absent option."""

    raw = getattr(
        options, "hailo_build_targets", _MISSING_HAILO_BUILD_TARGETS
    )
    if raw is _MISSING_HAILO_BUILD_TARGETS:
        fallback = _option_str(options, "hailo_hw_arch", "hailo8").strip()
        return [fallback or "hailo8"]
    if isinstance(raw, (list, tuple)):
        return [str(value).strip() for value in raw if str(value).strip()]
    return [
        value.strip()
        for value in str(raw or "").replace(";", ",").split(",")
        if value.strip()
    ]


def _canon_backend(value: Any) -> str:
    s = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if s in {"cpu", "cpu_ort", "ort_cpu"}:
        return "cpu_ort"
    if s in {"cuda", "cuda_ort", "ort_cuda", "gpu"}:
        return "cuda_ort"
    if "trt" in s or "tensorrt" in s:
        return "tensorrt"
    if "hailo10h" in s:
        return "hailo10h"
    if "hailo10p" in s:
        return "hailo10p"
    if "hailo10" in s:
        return "hailo10h"
    if "hailo8l" in s:
        return "hailo8l"
    if "hailo8r" in s:
        return "hailo8r"
    if "hailo8" in s or "hailo" in s:
        return "hailo8"
    return s


def _resolve_suite_dir(run_root: Path, model_dir: Path, benchmark_set_contract: Mapping[str, Any]) -> Path:
    materialized = bool(benchmark_set_contract.get("materialized"))
    scope = str(benchmark_set_contract.get("materialization_scope") or "").lower()
    if (not materialized) and (
        "contract_only" in scope
        or benchmark_set_contract.get("source_of_truth_for_real_runs") == "legacy_benchmarkset_generator"
        or benchmark_set_contract.get("legacy_benchmarkset_required")
        or benchmark_set_contract.get("direct_suite_disabled")
    ):
        return (model_dir / "benchmark_set").resolve()
    raw = str(
        benchmark_set_contract.get("legacy_suite_dir")
        or benchmark_set_contract.get("suite_dir")
        or benchmark_set_contract.get("benchmark_suite_dir")
        or benchmark_set_contract.get("generated_suite_dir")  # compatibility with v49c-v49m runs only
        or ""
    ).strip()
    candidates: List[Path] = []
    if raw:
        p = Path(raw).expanduser()
        if p.is_absolute():
            candidates.append(p)
        else:
            candidates.extend([run_root / p, model_dir / p, model_dir / "benchmark_set" / p])
    candidates.extend([
        model_dir / "benchmark_set" / "legacy_suite",
    ])
    # The old direct suite/generated_suite are not scanned in normal v49p
    # workflows. They are only available through the explicit debug path that
    # records a concrete suite_dir.
    for cand in candidates:
        if cand.is_dir():
            return cand.resolve()
    return (model_dir / "benchmark_set").resolve()


def _infer_case_id(path: Path) -> str:
    for part in path.parts:
        low = part.lower()
        if len(low) >= 2 and low[0] == "b" and low[1:].isdigit():
            return part
    return "full" if "full" in {p.lower() for p in path.parts} else ""


def _infer_variant(path: Path) -> str:
    low = [p.lower() for p in path.parts]
    for key in ("full", "part1", "part2", "composed"):
        if key in low:
            return key
    return "unknown"


def _infer_backend(path: Path) -> str:
    """Infer the Hailo architecture without letting a generic ``hailo``
    directory mask a more specific child such as ``hailo10``.

    Generated suites commonly use ``.../hailo/hailo10/...``.  The old
    first-match implementation canonicalised the parent ``hailo`` to
    ``hailo8`` and consequently queued already-built Hailo-10 artifacts as
    missing.
    """
    candidates = [_canon_backend(part) for part in path.parts]
    priority = ("hailo10h", "hailo10p", "hailo8l", "hailo8r", "hailo8")
    for backend in priority:
        if backend in candidates:
            return backend
    return "hailo8"


def _contract_for(contracts: Sequence[Mapping[str, Any]], backend: str, variant: str = "full") -> Dict[str, Any]:
    b = _canon_backend(backend)
    for raw in contracts:
        if not isinstance(raw, Mapping):
            continue
        if _canon_backend(raw.get("backend")) == b and str(raw.get("variant") or variant) == variant:
            return dict(raw)
    for raw in contracts:
        if isinstance(raw, Mapping) and _canon_backend(raw.get("backend")) == b:
            return dict(raw)
    return {}


def _discover_hefs(
    suite_dir: Path, run_root: Path, contracts: Sequence[Mapping[str, Any]],
    *, model_id: str = "", task: str = "",
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not suite_dir.is_dir():
        return out
    seen: set[str] = set()
    for hef in sorted(suite_dir.rglob("compiled.hef")):
        if ".hailo-generations" in hef.relative_to(suite_dir).parts:
            continue
        if not hef.is_file():
            continue
        key = str(hef.resolve())
        if key in seen:
            continue
        seen.add(key)
        variant = _infer_variant(hef)
        backend = _infer_backend(hef)
        case_id = _infer_case_id(hef)
        contract = _contract_for(contracts, backend, "full" if variant == "full" else "split")
        endpoint_mode = str(contract.get("endpoint_mode") or "decoded")
        if variant == "full":
            # Adjacent per-artifact JSON is diagnostic metadata, not the
            # authoritative suite decision.  Resolve only the exact root row;
            # task/model/backend conflicts intentionally produce no stage.
            contract = load_authoritative_output_contract(
                suite_dir,
                backend=backend,
                model_id=str(model_id or ""),
                variant="full",
                task=str(task or ""),
            )
            endpoint_mode = str(
                contract.get("stage")
                or contract.get("endpoint_mode")
                or "unknown"
            )
        out.append({
            "backend": backend,
            "hw_arch": backend,
            "case_id": case_id,
            "variant": variant,
            "stage": variant,
            "path": str(hef),
            "path_rel": relpath(hef, run_root),
            "size_bytes": hef.stat().st_size if hef.is_file() else None,
            "endpoint_mode": endpoint_mode,
            "output_contract": contract,
        })
    return out


def _has_hef(hefs: Sequence[Mapping[str, Any]], *, backend: str, variant: str, case_id: str = "") -> Optional[Mapping[str, Any]]:
    b = _canon_backend(backend)
    v = str(variant or "").lower()
    c = str(case_id or "").strip()
    for raw in hefs:
        if not isinstance(raw, Mapping):
            continue
        if _canon_backend(raw.get("backend") or raw.get("hw_arch")) != b:
            continue
        if str(raw.get("variant") or raw.get("stage") or "").lower() != v:
            continue
        if c and str(raw.get("case_id") or "") not in {c, ""}:
            continue
        return raw
    return None


def _expected_unsupported_records(*, contracts: Sequence[Mapping[str, Any]], case_requests: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for req in case_requests:
        if not isinstance(req, Mapping):
            continue
        backend = _canon_backend(req.get("backend"))
        if not backend.startswith("hailo"):
            continue
        if not (bool(req.get("part2_expected_unsupported")) or bool(req.get("expected_unsupported"))):
            continue
        cid = str(req.get("case_id") or req.get("case_dir") or "")
        key = (cid, backend, "part2")
        if key in seen:
            continue
        seen.add(key)
        records.append({
            "case_id": cid,
            "case_dir": str(req.get("case_dir") or cid),
            "split_index": req.get("split_index"),
            "backend": backend,
            "stage": "part2",
            "expected_unsupported_reason": str(
                req.get("expected_unsupported_reason")
                or "decoded YOLO detection tail is parser-blocking on Hailo unless the split uses a raw-head/host-tail endpoint contract"
            ),
            "error_class": "unsupported_op",
            "workflow_action": "do_not_rebuild_decoded_tail_blindly; use the ready raw-head/host-tail or Hailo-part1-to-host pipeline",
        })
    return records



def _hailo_attempt_status(payload: Mapping[str, Any]) -> str:
    from .artifact_cache_preflight import known_negative_build_evidence
    negative = known_negative_build_evidence(payload)
    phase = str(payload.get("last_active_stage") or payload.get("compiler_phase") or payload.get("last_stage") or "")
    if negative:
        return "known_infeasible"
    if phase == "build_evidence_lookup":
        return "not_started_evidence_unconfirmed"
    if phase == "local_dfc_workspace_preflight":
        return "not_started_workspace_blocked"
    semantic = str(payload.get("semantic_status") or "").strip().lower()
    if bool(payload.get("ok")) or semantic in {"success", "succeeded", "completed"}:
        return "succeeded"
    failure = str(payload.get("failure_kind") or "").strip().lower()
    if (
        bool(payload.get("timed_out"))
        or semantic in {"timeout", "timed_out"}
        or failure == "timeout"
    ):
        return "attempted_timeout"
    if semantic == "unsupported" or str(payload.get("unsupported_reason") or "").strip():
        return "attempted_unsupported"
    if failure in {"deferred_cold_full_cache_miss", "cache_miss_deferred"}:
        return "deferred_cache_miss"
    if bool(payload.get("skipped")):
        return "attempted_skipped"
    return "attempted_failed"


def _attempt_stage_candidates(payload: Mapping[str, Any], path: Path) -> set[str]:
    # full_or_unspecified is an old endpoint placeholder, not a Full identity.
    # Concrete producer stage and output path remain authoritative.
    stages = set()
    endpoint = str(payload.get("endpoint") or "").strip().lower()
    endpoint_stage = {"decoded_full": "full", "raw_head_fallback": "full",
                      "split_part1": "part1", "split_part2": "part2"}.get(endpoint)
    if endpoint_stage:
        stages.add(endpoint_stage)
    for source in (payload, payload.get("build_evidence_context") or {},
                   (payload.get("details") or {}).get("build_evidence_context") or {}):
        if isinstance(source, Mapping) and source.get("stage") in {"full", "part1", "part2"}:
            stages.add(str(source["stage"]))
    inferred = _infer_variant(path)
    if inferred in {"full", "part1", "part2"}:
        stages.add(inferred)
    if not stages:
        parts = [str(part).lower() for part in path.parts]
        stages.update(stage for stage in ("full", "part1", "part2") if stage in parts)
    return stages


def _attempt_stage_from_payload(payload: Mapping[str, Any], path: Path) -> tuple[str, str]:
    stages = _attempt_stage_candidates(payload, path)
    if len(stages) != 1:
        return "unknown", "unknown"
    stage = next(iter(stages))
    return ("full" if stage == "full" else "split"), stage


def _attempt_case_from_payload(payload: Mapping[str, Any], path: Path, stage: str) -> str:
    for value in (payload.get("case_id"), payload.get("case")):
        token = str(value or "").strip()
        if token:
            return token
    inferred = _infer_case_id(path)
    if inferred:
        return inferred
    for part in path.parts:
        token = str(part)
        if token.lower().startswith("b") and token[1:].isdigit():
            return token
    return "full" if stage == "full" else ""


def _immutable_attempt_row(path: Path, payload: Mapping[str, Any], run_root: Path) -> Dict[str, Any]:
    variant, stage = _attempt_stage_from_payload(payload, path)
    backend = _canon_backend(payload.get("hw_arch") or payload.get("backend") or _infer_backend(path))
    case_id = _attempt_case_from_payload(payload, path, stage)
    ended = payload.get("ended_at_epoch_s")
    try:
        ended_epoch_s = float(ended)
    except Exception:
        ended_epoch_s = float(path.stat().st_mtime)
    return {
        "backend": backend,
        "variant": variant,
        "stage": stage,
        "case_id": case_id,
        "endpoint": str(payload.get("endpoint") or ""),
        "status": _hailo_attempt_status(payload),
        "ok": str(payload.get("semantic_status") or "").lower() in {"success", "succeeded", "completed"},
        "skipped": bool(payload.get("skipped")),
        "timed_out": bool(payload.get("timed_out")) or str(payload.get("semantic_status") or "").lower() == "timeout",
        "timeout_kind": payload.get("timeout_kind"),
        "timeout_policy": dict(payload.get("timeout_policy") or {}),
        "last_stage": payload.get("last_active_stage") or payload.get("compiler_phase"),
        "compiler_phase": payload.get("compiler_phase") or payload.get("last_active_stage"),
        "failure_kind": payload.get("failure_kind"),
        "error_class": payload.get("error_class"),
        "unsupported_reason": payload.get("unsupported_reason"),
        "elapsed_s": payload.get("elapsed_s", payload.get("duration_s")),
        "returncode": payload.get("returncode"),
        "error": payload.get("error"),
        "stdout_tail": payload.get("stdout_tail"),
        "stderr_tail": payload.get("stderr_tail"),
        "source_onnx": payload.get("source_onnx"),
        "source_onnx_sha256": payload.get("source_onnx_sha256"),
        "compiler_onnx": payload.get("compiler_onnx"),
        "compiler_onnx_sha256": payload.get("compiler_onnx_sha256"),
        "start_nodes": list(payload.get("start_nodes") or []),
        "end_nodes": list(payload.get("end_nodes") or []),
        "attempt_id": payload.get("attempt_id"),
        "started_at_epoch_s": payload.get("started_at_epoch_s"),
        "ended_at_epoch_s": ended_epoch_s,
        "invocation_status": payload.get("invocation_status"),
        "semantic_status": payload.get("semantic_status"),
        "result_path": relpath(path, run_root),
        "receipt_kind": "immutable_attempt_receipt",
        "identity_conflicts": sorted(_attempt_stage_candidates(payload, path)) if stage == "unknown" else [],
        "identity_status": "conflicting_or_unresolved" if stage == "unknown" else "resolved",
        "compiler_dispatch_count": (payload.get("details") or {}).get("compiler_dispatch_count",
            0 if _hailo_attempt_status(payload).startswith("not_started") or _hailo_attempt_status(payload) == "known_infeasible" else None),
    }


def _legacy_attempt_row(path: Path, payload: Mapping[str, Any], run_root: Path) -> Dict[str, Any]:
    variant = _infer_variant(path)
    stage = "full" if variant == "full" else variant
    return {
        "backend": _infer_backend(path),
        "variant": "full" if stage == "full" else "split",
        "stage": stage,
        "case_id": _infer_case_id(path) or ("full" if stage == "full" else ""),
        "endpoint": "",
        "status": _hailo_attempt_status(payload),
        "ok": bool(payload.get("ok")),
        "skipped": bool(payload.get("skipped")),
        "timed_out": bool(payload.get("timed_out")),
        "timeout_kind": payload.get("timeout_kind"),
        "timeout_policy": {},
        "last_stage": payload.get("last_stage"),
        "compiler_phase": payload.get("last_stage"),
        "failure_kind": payload.get("failure_kind"),
        "error_class": payload.get("failure_kind") or payload.get("timeout_kind"),
        "unsupported_reason": payload.get("unsupported_reason"),
        "elapsed_s": payload.get("elapsed_s"),
        "returncode": payload.get("returncode"),
        "error": payload.get("error"),
        "cache_info": payload.get("calib_info") if isinstance(payload.get("calib_info"), Mapping) else {},
        "result_path": relpath(path, run_root),
        "debug_log": payload.get("debug_log"),
        "started_at_epoch_s": None,
        "ended_at_epoch_s": float(path.stat().st_mtime),
        "invocation_status": "returned",
        "semantic_status": (
            "success" if bool(payload.get("ok")) else
            "timeout" if bool(payload.get("timed_out")) else
            "unsupported" if str(payload.get("unsupported_reason") or "").strip() else
            "failed"
        ),
        "receipt_kind": "legacy_build_result",
    }


def _collect_hailo_build_attempts(suite_dir: Path, run_root: Path) -> List[Dict[str, Any]]:
    """Collect immutable attempts and fall back to legacy result files.

    Every immutable receipt is retained, so a decoded parser failure followed
    by a raw-head timeout remains visible.  The corresponding legacy result in
    the same output directory is suppressed to avoid double-counting one
    invocation.
    """
    attempts: List[Dict[str, Any]] = []
    if not Path(suite_dir).is_dir():
        return attempts
    receipt_outdirs: set[Path] = set()
    for path in sorted(Path(suite_dir).rglob("hailo_attempt_receipts/attempt_*.json")):
        # Start and heartbeat receipts are deliberately retained on disk, but
        # they are not terminal compiler outcomes and must not enter the
        # attempt ledger or suppress a legacy terminal result.
        if path.name.endswith(".started.json") or path.name.endswith(".heartbeat.json"):
            continue
        try:
            payload = read_json(path, default={}) or {}
            if not isinstance(payload, Mapping):
                continue
            if int(payload.get("schema_version") or 1) >= 2 and payload.get("terminal") is not True:
                continue
            attempts.append(_immutable_attempt_row(path, payload, run_root))
            receipt_outdirs.add(path.parent.parent.resolve())
        except Exception:
            continue
    for path in sorted(Path(suite_dir).rglob("hailo_hef_build_result.json")):
        try:
            if path.parent.resolve() in receipt_outdirs:
                continue
            payload = read_json(path, default={}) or {}
            if not isinstance(payload, Mapping):
                continue
            attempts.append(_legacy_attempt_row(path, payload, run_root))
        except Exception:
            continue
    attempts.sort(key=lambda row: (
        float(row.get("ended_at_epoch_s") or 0.0),
        str(row.get("attempt_id") or row.get("result_path") or ""),
    ))
    return attempts


def _attempt_for(attempts: Sequence[Mapping[str, Any]], *, backend: str, variant: str, case_id: str = "") -> Dict[str, Any]:
    backend = _canon_backend(backend)
    variant = str(variant or "").lower()
    case_id = str(case_id or "")
    candidates: List[Mapping[str, Any]] = []
    for row in attempts:
        if _canon_backend(row.get("backend")) != backend:
            continue
        row_variant = str(row.get("variant") or "").lower()
        row_stage = str(row.get("stage") or "").lower()
        if variant == "full":
            if row_variant != "full" and row_stage != "full":
                continue
        elif row_stage != variant and row_variant != variant:
            continue
        rid = str(row.get("case_id") or "")
        if case_id and rid and rid != case_id:
            continue
        candidates.append(row)
    if not candidates:
        return {}
    # The actual final attempt is authoritative even when it failed.  A prior
    # success must not mask a later parser/fallback timeout or cancellation.
    return dict(max(candidates, key=lambda row: (
        float(row.get("ended_at_epoch_s") or 0.0),
        float(row.get("started_at_epoch_s") or 0.0),
        str(row.get("attempt_id") or row.get("result_path") or ""),
    )))

def materialize_hailo_artifact_service_plan(
    *,
    run_dir: str | Path,
    model_id: str,
    targets: Sequence[str],
    full_baseline_plan: Mapping[str, Any],
    output_contracts: Mapping[str, Any],
    benchmark_set_contract: Mapping[str, Any],
    backend_artifact_decisions: Mapping[str, Any],
    build_backend_artifacts_request: Mapping[str, Any],
    no_remote: bool,
    execution_mode: str,
    hailo_full_requested: bool = True,
) -> Dict[str, Any]:
    run_root = Path(run_dir)
    model_dir = run_root / "models" / str(model_id)
    base = model_dir / "benchmark_set"
    suite_dir = _resolve_suite_dir(run_root, model_dir, benchmark_set_contract)
    contracts = [dict(x) for x in _as_list(output_contracts.get("contracts")) if isinstance(x, Mapping)]
    baselines = [dict(x) for x in _as_list(full_baseline_plan.get("baselines")) if isinstance(x, Mapping)]
    case_requests = [dict(x) for x in _as_list(backend_artifact_decisions.get("case_build_requests")) if isinstance(x, Mapping)]
    task = str(
        full_baseline_plan.get("task")
        or output_contracts.get("task")
        or next((row.get("task") for row in contracts if row.get("task")), "")
        or ""
    ).strip().lower()
    detected_hefs = _discover_hefs(
        suite_dir, run_root, contracts, model_id=model_id, task=task,
    )
    build_attempts = _collect_hailo_build_attempts(suite_dir, run_root)

    full_requests: List[Dict[str, Any]] = []
    for raw in baselines:
        backend = _canon_backend(raw.get("backend"))
        if not backend.startswith("hailo"):
            continue
        contract = _contract_for(contracts, backend, "full")
        requested = not (
            raw.get("requested") is False
            or contract.get("requested") is False
            or not hailo_full_requested
        )
        hit = (
            _has_hef(
                detected_hefs, backend=backend, variant="full", case_id="full",
            )
            or _has_hef(detected_hefs, backend=backend, variant="full")
        ) if requested else {}
        endpoint_mode = str(raw.get("endpoint_mode") or contract.get("endpoint_mode") or "decoded")
        attempt = (
            _attempt_for(
                build_attempts, backend=backend, variant="full", case_id="full",
            )
            if requested else {}
        )
        status = (
            "not_requested_by_profile"
            if not requested
            else (
                "ready_reused" if hit
                else str(
                    attempt.get("status")
                    or "pending_dfc_build_or_preparation"
                )
            )
        )
        if requested and raw.get("artifact_path") and not hit and not attempt:
            status = "source_artifact_missing_or_not_copied"
        full_requests.append({
            "model_id": model_id,
            "backend": backend,
            "variant": "full",
            "endpoint_mode": endpoint_mode,
            "postprocessing_required": bool(contract.get("postprocessing_required") or endpoint_mode == "raw_detection_head"),
            "host_tail_required": bool(contract.get("host_tail_required") or endpoint_mode == "raw_detection_head"),
            "source_artifact_path": raw.get("artifact_path", ""),
            "requested": requested,
            "status": status,
            "ready": bool(hit),
            "hef_path": hit.get("path_rel") if isinstance(hit, Mapping) else "",
            "output_contract": contract,
            "build_policy": "reuse_existing_prepared_or_compiled_artifact_before_rebuild",
            "build_attempt": attempt,
            "attempted": bool(attempt),
        })

    case_plan: List[Dict[str, Any]] = []
    for raw in case_requests:
        backend = _canon_backend(raw.get("backend"))
        if not backend.startswith("hailo"):
            continue
        cid = str(raw.get("case_id") or "")
        for stage_key in ("part1", "part2"):
            if not bool(raw.get(f"build_{stage_key}")):
                continue
            hit = _has_hef(detected_hefs, backend=backend, variant=stage_key, case_id=cid)
            attempt = _attempt_for(build_attempts, backend=backend, variant=stage_key, case_id=cid)
            status = "ready_existing_hef" if hit else str(attempt.get("status") or "pending_dfc_build")
            case_plan.append({
                "model_id": model_id,
                "case_id": cid,
                "backend": backend,
                "variant": "split",
                "stage": stage_key,
                "split_index": raw.get("split_index"),
                "status": status,
                "ready": bool(hit),
                "hef_path": hit.get("path_rel") if isinstance(hit, Mapping) else "",
                "requires_heavy_service": True,
                "remote_runtime_can_consume": bool(hit),
                "build_attempt": attempt,
                "attempted": bool(attempt),
            })

    expected_unsupported = _expected_unsupported_records(contracts=contracts, case_requests=case_requests)
    requested_full = [r for r in full_requests if r.get("requested") is not False]
    not_requested_full = [r for r in full_requests if r.get("requested") is False]
    missing_full = [r for r in requested_full if not r.get("ready")]
    missing_cases = [r for r in case_plan if not r.get("ready")]
    remote_candidate = (not bool(no_remote)) and str(execution_mode or "").lower() == "generate_and_run"
    target_has_hailo = any(_canon_backend(t).startswith("hailo") for t in targets)
    if not full_requests and not case_plan and not target_has_hailo:
        status = "not_applicable"
    elif not missing_full and not missing_cases:
        status = "ready_from_reuse"
    elif remote_candidate:
        status = "pending_remote_or_hailo_service"
    else:
        status = "pending_manual_or_advanced_hailo_build"

    service_plan = {
        "schema": "onnx-splitpoint/hailo-artifact-service-plan",
        "schema_version": 1,
        "created_at": now_iso(),
        "model_id": model_id,
        "suite_dir": relpath(suite_dir, run_root),
        "targets": list(targets),
        "execution_mode": str(execution_mode or ""),
        "no_remote": bool(no_remote),
        "status": status,
        "detected_hefs": detected_hefs,
        "build_attempts": build_attempts,
        "build_attempt_count": len(build_attempts),
        "full_baseline_requests": full_requests,
        "requested_full_baseline_count": len(requested_full),
        "not_requested_full_baseline_count": len(not_requested_full),
        "case_hef_requests": case_plan,
        "expected_unsupported": expected_unsupported,
        "missing_full_baseline_count": len(missing_full),
        "missing_case_hef_count": len(missing_cases),
        "reuse_policy": "never rebuild expensive Hailo HEFs blindly; prefer prepared/full raw-head HEFs and resume-safe suite artifacts",
        "service_binding_note": "The formal runner records Hailo DFC work as structured service requests. Actual DFC builds remain delegated to existing benchmark/preparation services or remote benchmark execution.",
        "source_build_request_path": relpath(base / "build_backend_artifacts_request.json", run_root),
        "source_backend_decisions_path": relpath(base / "backend_artifact_decisions.json", run_root),
        "source_build_backend_artifacts_request": dict(build_backend_artifacts_request or {}),
    }
    status_payload = {
        "schema": "onnx-splitpoint/hailo-artifact-status",
        "schema_version": 1,
        "created_at": now_iso(),
        "model_id": model_id,
        "suite_dir": relpath(suite_dir, run_root),
        "status": status,
        "detected_hef_count": len(detected_hefs),
        "ready_full_baselines": sum(1 for r in requested_full if r.get("ready")),
        "pending_full_baselines": len(missing_full),
        "not_requested_full_baselines": len(not_requested_full),
        "ready_case_hefs": sum(1 for r in case_plan if r.get("ready")),
        "pending_case_hefs": len(missing_cases),
        "expected_unsupported_count": len(expected_unsupported),
        "build_attempt_count": len(build_attempts),
        "attempted_timeout_count": sum(1 for row in build_attempts if row.get("status") == "attempted_timeout"),
        "deferred_cache_miss_count": sum(1 for row in build_attempts if row.get("status") == "deferred_cache_miss"),
    }

    p_plan = write_json(base / "hailo_artifact_service_plan.json", service_plan)
    p_status = write_json(base / "hailo_artifact_status.json", status_payload)
    p_expected = write_json(base / "hailo_expected_unsupported.json", {
        "schema": "onnx-splitpoint/hailo-expected-unsupported",
        "schema_version": 1,
        "created_at": now_iso(),
        "model_id": model_id,
        "records": expected_unsupported,
        "record_count": len(expected_unsupported),
    })
    p_suite_manifest = write_json(suite_dir / "hailo_artifacts_manifest.json", {
        "schema": "onnx-splitpoint/hailo-artifacts-manifest",
        "schema_version": 1,
        "created_at": now_iso(),
        "model_id": model_id,
        "detected_hefs": detected_hefs,
        "build_attempts": build_attempts,
        "full_baseline_requests": full_requests,
        "case_hef_requests": case_plan,
        "status": status,
    })
    p_note = write_text(base / "hailo_service_binding.md", _hailo_service_binding_markdown(service_plan))
    return {
        "artifacts": {
            "hailo_artifact_service_plan_json": p_plan,
            "hailo_artifact_status_json": p_status,
            "hailo_expected_unsupported_json": p_expected,
            "suite_hailo_artifacts_manifest_json": p_suite_manifest,
            "hailo_service_binding_md": p_note,
        },
        "metrics": {
            "detected_hefs": len(detected_hefs),
            "full_ready": sum(1 for r in requested_full if r.get("ready")),
            "full_pending": len(missing_full),
            "full_not_requested": len(not_requested_full),
            "case_hefs_ready": sum(1 for r in case_plan if r.get("ready")),
            "case_hefs_pending": len(missing_cases),
            "expected_unsupported": len(expected_unsupported),
            "service_status": status,
        },
        "status": "ok" if status in {"ready_from_reuse", "not_applicable"} else "partial",
        "message": "Hailo HEF reuse/build service plan recorded; existing HEFs were detected and missing heavy builds are structured.",
    }



def _hailo_service_binding_markdown(plan: Mapping[str, Any]) -> str:
    lines = [
        "# Hailo Artifact Service Binding",
        "",
        f"Model: `{plan.get('model_id')}`",
        f"Status: `{plan.get('status')}`",
        f"Suite: `{plan.get('suite_dir')}`",
        "",
        "This file is generated by the formal Evaluation Workflow. It is explicit about what is reused and what still requires a heavy Hailo DFC/preparation/remote service.",
        "",
        "## Reuse/build policy",
        "",
        "- Reuse prepared/full Hailo HEFs before rebuilding.",
        "- Preserve raw-head output contracts for YOLO-style detection models.",
        "- Do not treat raw-head Hailo outputs as decoded ONNX outputs.",
        "- Do not fabricate benchmark measurements; remote/local execution must produce result files.",
        "",
        "## Summary",
        "",
        f"- Detected HEFs: {len(_as_list(plan.get('detected_hefs')))}",
        f"- Full baseline contracts: {len(_as_list(plan.get('full_baseline_requests')))}",
        f"- Requested Full baselines: {int(plan.get('requested_full_baseline_count') or 0)}",
        f"- Full baselines not requested by profile: {int(plan.get('not_requested_full_baseline_count') or 0)}",
        f"- Case HEF requests: {len(_as_list(plan.get('case_hef_requests')))}",
        f"- Expected unsupported records: {len(_as_list(plan.get('expected_unsupported')))}",
        "",
    ]
    return "\n".join(lines)


def _option_bool(options: Any, name: str, default: bool = False) -> bool:
    try:
        return bool(getattr(options, name, default))
    except Exception:
        return default


def _option_str(options: Any, name: str, default: str = "") -> str:
    try:
        return str(getattr(options, name, default) or "")
    except Exception:
        return default


def _remote_host_configured(options: Any) -> bool:
    if _option_str(options, "remote_host_json").strip():
        return True
    if _option_str(options, "remote_host").strip():
        return True
    if _option_str(options, "remote_hosts_file").strip():
        return True
    if _option_str(options, "remote_host_id").strip():
        return True
    return False


def _build_queue_from_plan(plan: Mapping[str, Any]) -> List[Dict[str, Any]]:
    queue: List[Dict[str, Any]] = []
    for raw in _as_list(plan.get("full_baseline_requests")):
        if (
            not isinstance(raw, Mapping)
            or raw.get("ready")
            or raw.get("requested") is False
        ):
            continue
        queue.append({
            "model_id": plan.get("model_id", ""),
            "case_id": "full",
            "backend": _canon_backend(raw.get("backend")),
            "variant": "full",
            "stage": "full",
            "endpoint_mode": raw.get("endpoint_mode", "decoded"),
            "postprocessing_required": bool(raw.get("postprocessing_required")),
            "host_tail_required": bool(raw.get("host_tail_required")),
            "source_artifact_path": raw.get("source_artifact_path", ""),
            "status": raw.get("status", "pending_dfc_build_or_preparation"),
            "service": "hailo_dfc_or_preparation",
        })
    for raw in _as_list(plan.get("case_hef_requests")):
        if not isinstance(raw, Mapping) or raw.get("ready"):
            continue
        queue.append({
            "model_id": plan.get("model_id", ""),
            "case_id": raw.get("case_id", ""),
            "backend": _canon_backend(raw.get("backend")),
            "variant": raw.get("variant", "split"),
            "stage": raw.get("stage", ""),
            "split_index": raw.get("split_index", ""),
            "status": raw.get("status", "pending_dfc_build"),
            "service": "hailo_dfc_or_remote_benchmark_service",
        })
    return queue


def _collect_hailo_build_attempts_binding(base: Path) -> List[Dict[str, Any]]:
    # ``base`` is below the run root in the normal workflow.  Relative paths
    # are not required by the binding queue, so use the model benchmark-set as
    # a stable root when the exact run root is unavailable here.
    return _collect_hailo_build_attempts(Path(base), Path(base))

def _merge_attempts_into_queue(queue: List[Dict[str, Any]], attempts: Sequence[Mapping[str, Any]]) -> None:
    for item in queue:
        backend = _canon_backend(item.get("backend"))
        stage = str(item.get("stage") or "")
        case_id = str(item.get("case_id") or "")
        matches = [row for row in attempts if _canon_backend(row.get("backend")) == backend and str(row.get("stage") or "") == stage]
        if case_id and case_id != "full":
            matches = [row for row in matches if str(row.get("case_id") or "") == case_id]
        if not matches:
            continue
        attempt = matches[-1]
        item["status"] = attempt.get("status")
        item["attempted"] = True
        item["attempt"] = dict(attempt)
        for key in ("timed_out", "last_stage", "failure_kind", "error", "elapsed_s", "cache_hit", "cache_key"):
            item[key] = attempt.get(key)


def materialize_hailo_artifact_service_binding(
    *,
    run_dir: str | Path,
    model_id: str,
    options: Any,
    targets: Sequence[str],
    benchmark_set_contract: Mapping[str, Any],
    benchmark_plan: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Compatibility entry point used by the formal workflow runner.

    It extends the generic backend-artifact decision stage with v49e-v49i Hailo
    service artifacts: a reuse/build plan, a queue, and an explicit dispatch
    record. Heavy DFC/remote work is still delegated to the existing generator,
    preparation, and RemoteBenchmarkService paths; this function makes the
    handoff structured and resume-safe.
    """

    root = Path(run_dir)
    model_dir = root / "models" / str(model_id)
    base = model_dir / "benchmark_set"
    full_plan = read_json(model_dir / "full_baselines" / "full_baseline_plan.json", default={}) or {}
    output_contracts = read_json(model_dir / "full_baselines" / "output_contracts.json", default={}) or {}
    backend_decisions = read_json(base / "backend_artifact_decisions.json", default={}) or {}
    build_request = read_json(base / "build_backend_artifacts_request.json", default={}) or {}
    execution_mode = _option_str(options, "execution_mode", "contracts_only").strip().lower().replace("-", "_") or "contracts_only"
    no_remote = _option_bool(options, "no_remote", False)
    hailo_full_requested = _option_bool(options, "hailo_build_full", True)
    plan_result = materialize_hailo_artifact_service_plan(
        run_dir=root,
        model_id=model_id,
        targets=targets,
        full_baseline_plan=full_plan,
        output_contracts=output_contracts,
        benchmark_set_contract=benchmark_set_contract,
        backend_artifact_decisions=backend_decisions,
        build_backend_artifacts_request=build_request,
        no_remote=no_remote,
        execution_mode=execution_mode,
        hailo_full_requested=hailo_full_requested,
    )
    artifacts: Dict[str, Path] = dict(plan_result.get("artifacts") or {})
    metrics: Dict[str, Any] = dict(plan_result.get("metrics") or {})
    service_plan_path = artifacts.get("hailo_artifact_service_plan_json") or (base / "hailo_artifact_service_plan.json")
    service_plan = read_json(service_plan_path, default={}) or {}
    queue = _build_queue_from_plan(service_plan)
    build_attempts = _collect_hailo_build_attempts_binding(base)
    _merge_attempts_into_queue(queue, build_attempts)
    detected_hefs = _as_list(service_plan.get("detected_hefs"))
    remote_host_ready = _remote_host_configured(options)
    deferred_statuses = {"deferred_cache_miss", "deferred_cold_build"}
    deferred_queue = [item for item in queue if str(item.get("status") or "") in deferred_statuses]
    actionable_queue = [item for item in queue if item not in deferred_queue]
    has_pending = bool(actionable_queue)
    if not any(_canon_backend(t).startswith("hailo") for t in targets or []):
        dispatch_status = "not_applicable"
        dispatch_reason = "profile_has_no_hailo_target"
    elif not has_pending:
        dispatch_status = "not_required"
        dispatch_reason = (
            "smoke_cold_builds_deferred_by_policy" if deferred_queue
            else "all_requested_hailo_artifacts_ready_or_no_hailo_builds_requested"
        )
    elif no_remote:
        dispatch_status = "not_dispatched"
        dispatch_reason = "remote_disabled; queue waits for Advanced/Diagnostics Hailo build or imported suite artifacts"
    elif execution_mode != "generate_and_run":
        dispatch_status = "not_dispatched"
        dispatch_reason = "execution_mode_does_not_request_runtime_or_remote_execution"
    elif remote_host_ready:
        dispatch_status = "ready_for_remote_execution"
        dispatch_reason = "RemoteBenchmarkService can consume benchmark_set.json during run_benchmarks stage"
    else:
        dispatch_status = "pending_remote_configuration"
        dispatch_reason = "Hailo work exists but no remote host is configured"

    build_mode = _option_str(options, "hailo_build_mode", "reuse_only").strip().lower().replace("-", "_") or "reuse_only"
    if build_mode == "reuse_build_missing":
        build_mode = "reuse_and_build_missing"
    if build_mode not in {
        "auto", "reuse_only", "reuse_and_build_missing", "request",
        "local", "venv", "wsl", "cache_verify_only",
    }:
        build_mode = "reuse_only"
    build_targets = _hailo_build_targets_from_options(options)
    build_settings = {
        "mode": build_mode,
        "hw_arch": _option_str(options, "hailo_hw_arch", "hailo8"),
        "targets": build_targets,
        "backend": _option_str(options, "hailo_build_backend", "auto"),
        "timeout_s": int(float(_option_str(options, "hailo_build_timeout_s", "3600") or 0)),
        "build_full": hailo_full_requested,
        "build_part1": _option_bool(options, "hailo_build_part1", True),
        "build_part2": _option_bool(options, "hailo_build_part2", True),
        "preset": _option_str(options, "hailo_preset", "quick"),
        "optimization_level": int(float(_option_str(options, "hailo_optimization_level", "0") or 0)),
        "calib_dir": _option_str(options, "hailo_calib_dir", ""),
        "calib_count": int(float(_option_str(options, "hailo_calib_count", "16") or 0)),
        "calib_batch_size": int(float(_option_str(options, "hailo_calib_batch_size", "8") or 1)),
        "force_build": parse_config_bool(getattr(options, "hailo_force_build", False), field="options.hailo_force_build"),
        "keep_artifacts": _option_bool(options, "hailo_keep_artifacts", False),
        "source": "evaluation_profile_yaml_or_cli",
    }
    reuse_plan = {
        "schema": "onnx-splitpoint/hailo-build-reuse-plan",
        "schema_version": 1,
        "created_at": now_iso(),
        "model_id": model_id,
        "workflow_version": "v51a-profile-persistence-runtime-gates",
        "source_service_plan": relpath(service_plan_path, root),
        "status": service_plan.get("status", "unknown"),
        "reuse_policy": service_plan.get("reuse_policy", "reuse_existing_prepared_or_compiled_artifact_before_rebuild"),
        "detected_hefs": detected_hefs,
        "detected_hef_count": len(detected_hefs),
        "queue_count": len(queue),
        "benchmark_plan_summary": {
            "run_count": len(_as_list((benchmark_plan or {}).get("runs"))) if isinstance(benchmark_plan, Mapping) else 0,
            "case_count": len(_as_list((benchmark_plan or {}).get("cases"))) if isinstance(benchmark_plan, Mapping) else 0,
        },
        "build_settings": build_settings,
    }
    queue_payload = {
        "schema": "onnx-splitpoint/hailo-build-queue",
        "schema_version": 1,
        "created_at": now_iso(),
        "model_id": model_id,
        "queue": queue,
        "queue_count": len(queue),
        "build_attempts": build_attempts,
        "build_attempt_count": len(build_attempts),
        "ready_hef_count": len(detected_hefs),
        "expected_unsupported": _as_list(service_plan.get("expected_unsupported")),
        "build_settings": build_settings,
    }
    dispatch_payload = {
        "schema": "onnx-splitpoint/hailo-service-dispatch",
        "schema_version": 1,
        "created_at": now_iso(),
        "model_id": model_id,
        "status": dispatch_status,
        "reason": dispatch_reason,
        "execution_mode": execution_mode,
        "no_remote": no_remote,
        "remote_host_configured": remote_host_ready,
        "queue_count": len(queue),
        "actionable_queue_count": len(actionable_queue),
        "deferred_queue_count": len(deferred_queue),
        "service": "RemoteBenchmarkService" if dispatch_status == "ready_for_remote_execution" else "manual_or_advanced_hailo_service",
        "next_stage": "run_benchmarks" if dispatch_status == "ready_for_remote_execution" else "wait_for_artifacts_or_configure_remote",
    }
    if not queue:
        build_service_status = "not_required"
        build_service_reason = "all_hailo_artifacts_ready_or_no_hailo_targets"
    elif build_mode in {"reuse_only", "cache_verify_only"}:
        build_service_status = "queued_for_manual_or_remote_service"
        build_service_reason = (
            "cache_verify_only records an exact cache miss and forbids every "
            "compiler dispatch"
            if build_mode == "cache_verify_only"
            else "reuse_only mode records missing HEFs but intentionally does not request/build them"
        )
    elif build_mode in {"auto", "reuse_and_build_missing"}:
        build_service_status = "queued_missing_builds_for_configured_service"
        build_service_reason = "existing HEFs will be reused and missing HEFs are queued for the configured Hailo/remote build path; no fake build result is written"
    elif build_mode == "request":
        build_service_status = "requested_not_started"
        build_service_reason = "build requests are materialized for an external Hailo/DFC service"
    elif build_mode in {"local", "venv", "wsl"}:
        build_service_status = "ready_for_configured_build_service"
        build_service_reason = f"{build_mode} build mode selected; v49i records dispatch contract but does not fabricate build results"
    else:
        build_service_status = "unknown_build_mode"
        build_service_reason = f"unsupported hailo_build_mode={build_mode}"
    non_dispatch = [row for row in build_attempts if str(row.get("status") or "").startswith("not_started")
                    or row.get("status") == "known_infeasible"]
    failed_attempts = [row for row in build_attempts if not row.get("ok") and row not in non_dispatch]
    if failed_attempts:
        statuses = {str(row.get("status") or "") for row in failed_attempts}
        if "attempted_timeout" in statuses:
            build_service_status = "attempted_timeout"
            build_service_reason = "one or more Hailo builds were attempted and reached their hard timeout"
        elif "deferred_cache_miss" in statuses:
            build_service_status = "deferred_cache_miss"
            build_service_reason = "Smoke cache_or_defer policy skipped an expensive Hailo Full cache miss"
        else:
            build_service_status = "attempted_failed"
            build_service_reason = "one or more Hailo builds were attempted and failed"

    elif non_dispatch:
        if any(row.get("status") == "not_started_workspace_blocked" for row in non_dispatch):
            build_service_status = "not_started_workspace_blocked"
            build_service_reason = "Required cold jobs did not start: DFC workspace admission blocked them"
        elif all(row.get("status") == "known_infeasible" for row in non_dispatch):
            build_service_status = "known_infeasible"
            build_service_reason = "Exact known compiler exclusions reused; no compiler dispatch"
        else:
            build_service_status = "not_started_evidence_unconfirmed"
            build_service_reason = "Build evidence lookup incomplete; no compiler dispatch attested"

    build_service_dispatch = {
        "schema": "onnx-splitpoint/hailo-build-service-dispatch",
        "schema_version": 1,
        "created_at": now_iso(),
        "model_id": model_id,
        "status": build_service_status,
        "reason": build_service_reason,
        "hailo_build_mode": build_mode,
        "build_settings": build_settings,
        "queue_count": len(queue),
        "actionable_queue_count": len(actionable_queue),
        "deferred_queue_count": len(deferred_queue),
        "queue_path": "hailo_build_queue.json",
        "non_destructive": True,
    }
    build_service_status_payload = {
        "schema": "onnx-splitpoint/hailo-build-service-status",
        "schema_version": 1,
        "created_at": now_iso(),
        "model_id": model_id,
        "status": build_service_status,
        "reason": build_service_reason,
        "queue_count": len(queue),
        "actionable_queue_count": len(actionable_queue),
        "deferred_queue_count": len(deferred_queue),
        "executed": bool(build_attempts),
        "produced_hef_count": sum(1 for row in build_attempts if row.get("ok")),
        "build_attempt_count": len(build_attempts),
        "build_attempts": build_attempts,
        "build_settings": build_settings,
        "note": "v49i binds the Hailo build queue to a service contract. Actual DFC/local/WSL execution should update this file with produced HEFs.",
    }
    p_attempts = write_json(base / "hailo_build_attempts.json", {
        "schema": "onnx-splitpoint/hailo-build-attempts", "schema_version": 1,
        "created_at": now_iso(), "model_id": model_id, "attempts": build_attempts,
        "attempt_count": len(build_attempts),
    })
    p_reuse = write_json(base / "hailo_build_reuse_plan.json", reuse_plan)
    p_queue = write_json(base / "hailo_build_queue.json", queue_payload)
    p_dispatch = write_json(base / "hailo_service_dispatch.json", dispatch_payload)
    p_build_dispatch = write_json(base / "hailo_build_service_dispatch.json", build_service_dispatch)
    p_build_status = write_json(base / "hailo_build_service_status.json", build_service_status_payload)
    artifacts.update({
        "hailo_build_attempts_json": p_attempts,
        "hailo_build_reuse_plan_json": p_reuse,
        "hailo_build_queue_json": p_queue,
        "hailo_service_dispatch_json": p_dispatch,
        "hailo_build_service_dispatch_json": p_build_dispatch,
        "hailo_build_service_status_json": p_build_status,
    })
    metrics.update({
        "build_queue_count": len(queue),
        "actionable_build_queue_count": len(actionable_queue),
        "deferred_build_queue_count": len(deferred_queue),
        "remote_host_configured": remote_host_ready,
        "dispatch_status": dispatch_status,
        "build_service_status": build_service_status,
        "build_mode": build_mode,
    })
    status = str(plan_result.get("status") or "partial")
    if deferred_queue and not actionable_queue:
        status = "ok"
    elif dispatch_status in {"pending_remote_configuration", "not_dispatched", "ready_for_remote_execution"} and has_pending:
        status = "partial"
    message = str(plan_result.get("message") or "Hailo artifact service plan recorded.")
    message += f" Hailo build queue: {len(queue)} item(s); dispatch status: {dispatch_status}."
    return {"artifacts": artifacts, "metrics": metrics, "status": status, "message": message}
