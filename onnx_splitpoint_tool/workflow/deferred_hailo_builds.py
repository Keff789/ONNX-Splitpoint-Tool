"""Cache-only preparation and exact selected-artifact build continuation.

The normal generator still owns split ONNX creation and the build parameters.
Its callable is wrapped only until the campaign's final cache matrix is saved.
Continuation consumes those calls; it never runs candidate selection again.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Optional

from .artifacts import sha256_file, write_json
from .artifact_cache_preflight import known_negative_build_evidence


REQUEST_NAME = "deferred_hailo_build.json"
DEEPX_PART1_REQUEST = "deferred_deepx_part1_build.json"
DEEPX_PART2_REQUEST = "deferred_deepx_part2_build.json"
_BUILD_ENV = (
    "ONNX_SPLITPOINT_HAILO_CALIBRATION_STORAGE",
    "ONNX_SPLITPOINT_HAILO_CALIB_CAP_MB",
    "ONNX_SPLITPOINT_HAILO_CALIB_MANIFEST",
    "ONNX_SPLITPOINT_HAILO_CACHE_ENABLED",
    "ONNX_SPLITPOINT_HAILO_CACHE_ROOT",
    "ONNX_SPLITPOINT_HAILO_CACHE_INTEGRITY",
    "ONNX_SPLITPOINT_ACTIVATION_PROXY_STRICT",
    "SPLITPOINT_ACTIVATION_PROXY_STRICT",
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected object in {path}")
    return value


def _result_negative_evidence(result: Any) -> dict[str, Any]:
    return known_negative_build_evidence({
        "details": getattr(result, "details", None),
        "calib_info": getattr(result, "calib_info", None),
    })


def _probe_classification(result: Any) -> tuple[str, str]:
    details = getattr(result, "details", None)
    details = dict(details) if isinstance(details, Mapping) else {}
    calib = getattr(result, "calib_info", None)
    calib = dict(calib) if isinstance(calib, Mapping) else {}
    unsupported = str(getattr(result, "unsupported_reason", "") or "")
    failure = str(getattr(result, "failure_kind", "") or "")
    reason = str(calib.get("reason") or details.get("reason") or unsupported or failure or getattr(result, "error", "") or "cache_probe_unavailable")
    if getattr(result, "ok", False):
        return "HIT", "exact_cache_hit"
    negative = _result_negative_evidence(result)
    if negative:
        return "KNOWN_INFEASIBLE", str(negative.get("reason") or negative["state"])
    if details.get("compiler_identity_available") is False or unsupported in {
        "compiler_identity_unavailable", "cache_verify_only_force_conflict",
    }:
        return "UNKNOWN", reason
    confirmed = failure in {"cache_miss_blocked", "cache_miss", "deferred_cold_full_cache_miss"} or unsupported in {"cache_only_policy", "cache_verify_only_policy"}
    return ("MISS" if confirmed else "UNKNOWN"), reason


def _completed_job_observation(
    request: Mapping[str, Any], result: Any, *, model_id: str,
) -> dict[str, Any]:
    """Link the saved preflight to observed dispatch without inventing proof."""
    args = request.get("kwargs") or {}
    details = getattr(result, "details", None) or {}
    calib = getattr(result, "calib_info", None) or {}
    details = details if isinstance(details, Mapping) else {}
    calib = calib if isinstance(calib, Mapping) else {}
    context = details.get("compiler_context") or calib.get("compiler_context") or {}
    context = context if isinstance(context, Mapping) else {}
    cache_hit = details.get("cache_hit", calib.get("cache_hit"))
    count = details.get("compiler_dispatch_count", calib.get("compiler_dispatch_count"))
    # Explicit cache evidence is enough to establish zero compiler starts.
    # A generic successful return is not enough to infer a cold compilation.
    if count is None and cache_hit is True:
        count = 0
    if type(count) is not int or count < 0:
        count = None
    net_name = str(args.get("net_name") or "")
    binding = args.get("build_evidence_context") or {}
    binding = binding if isinstance(binding, Mapping) else {}
    stage = str(binding.get("stage") or Path(str(args.get("outdir") or "")).name)
    stage = stage if stage in {"full", "part1", "part2"} else "unknown"
    boundary = binding.get("boundary")
    boundary_text = f"b{boundary:03d}" if type(boundary) is int and boundary >= 0 else "full" if stage == "full" else "unknown"
    ok = bool(getattr(result, "ok", False))
    return {
        "model_id": str(binding.get("model_id") or model_id),
        "backend": str(args.get("hw_arch") or ""),
        "boundary": boundary_text,
        "identity_source": "build_evidence_context" if binding else "legacy_output_stage_only",
        "stage": stage,
        "net_name": net_name,
        "source_onnx": request.get("source_onnx"),
        "end_node_names": args.get("end_node_names"),
        "compute_by_family": args.get("compute_by_family") or {},
        "preflight_decision": request.get("cache_probe_status", "UNKNOWN"),
        "preflight_reason": request.get("cache_probe_reason", ""),
        "cache_hit": cache_hit if type(cache_hit) is bool else None,
        "compiler_dispatch_count": count,
        "model_build_status": "reused" if ok and cache_hit is True else "completed" if ok else "failed",
        "compiler_context": dict(context),
        "primary_failure_reason": str(getattr(result, "failure_kind", "") or ""),
        "workspace_preflight": details.get("workspace_preflight"),
        "workspace_preflight_path": details.get("workspace_preflight_path"),
        "exception_type": details.get("exception_type"),
        "exception_details": details.get("exception_details"),
        "hef_path": getattr(result, "hef_path", None),
        "receipt_path": calib.get("build_receipt_path") or details.get("build_receipt_path"),
        "result_json_path": getattr(result, "result_json_path", None),
        "next_required_action": "runtime_binding_and_quality" if ok else "inspect_original_failure",
    }


def _request_context(request: Mapping[str, Any], path: Path, *, model_id: str,
                     backend: str = "", stage: str = "", boundary: Any = None,
                     collection: bool = False) -> dict[str, Any]:
    """Use producer/request bindings; never infer a split from a network name."""
    args = request.get("kwargs") or {}
    binding = args.get("build_evidence_context") or {}
    binding = binding if isinstance(binding, Mapping) else {}
    raw_boundary = binding.get("boundary", boundary)
    role = str(binding.get("stage") or stage or "unknown")
    case = ("full" if role == "full" else
            f"b{raw_boundary:03d}" if type(raw_boundary) is int and raw_boundary >= 0
            else str(raw_boundary or "unknown"))
    return {
        "request": str(path), "model_id": str(binding.get("model_id") or model_id),
        "backend": str(backend or args.get("hw_arch") or "unknown"),
        "boundary": "selected_cases" if collection else case, "stage": role,
        "job_kind": "collection" if collection else "artifact",
        "identity_source": "bound_collection_request" if collection else
                           "build_evidence_context" if binding else "producer_request_context",
        "selected": True,
    }


def project_deferred_build_readiness(deferred_result: Mapping[str, Any], *,
                                     selected_requests: Any = None) -> dict[str, Any]:
    """Project existing current request observations, without dispatch or I/O.

    Backend counters remain owned by native_build_summary. Artifact job counts
    exclude collective DeepX requests, whose underlying cardinality is unknown.
    Repeated presentations of one request are observations of one logical job.
    """
    from ..native_job_identity import native_comparison
    jobs = deferred_result.get("jobs") or []
    contexts = (deferred_result.get("selected_requests") if selected_requests is None
                else selected_requests)
    if contexts is None:
        contexts = jobs
    observations = {}
    for job in jobs:
        if isinstance(job, Mapping) and job.get("request"):
            observations.setdefault(str(job["request"]), []).append(dict(job))
    projected = []
    seen = set()
    for context in contexts:
        if not isinstance(context, Mapping):
            context = {"status": "unknown", "identity_source": "invalid_request_context"}
        if context.get("selected") is False or context.get("required") is False or context.get("status") == "not_applicable":
            continue
        path = str(context.get("request") or "")
        key = path or (str(context.get("model_id")), str(context.get("backend")),
                       str(context.get("boundary")), str(context.get("stage")))
        if key in seen:
            continue
        seen.add(key)
        trail = observations.get(path, [])
        current = trail[-1] if trail else {}
        state = str(current.get("status") or "unknown").lower()
        if state == "not_applicable":
            continue
        row = {**dict(context), **current}
        # Bound identities may not be overwritten by a conflicting observation.
        conflicts = [field for field in ("model_id", "boundary", "stage")
                     if context.get(field) and current.get(field) and context[field] != current[field]]
        family = native_comparison(context.get("backend") or current.get("backend"))
        if current.get("backend") and context.get("backend") and native_comparison(current["backend"]) != family:
            conflicts.append("backend")
        row.update({field: context[field] for field in ("model_id", "boundary", "stage") if context.get(field)})
        row["backend"] = family
        boundary = str(row.get("boundary") or "")
        stage = str(row.get("stage") or "")
        boundary_ok = (boundary == "selected_cases" and stage == "part1"
                       if row.get("job_kind") == "collection" else
                       boundary == "full" if stage == "full" else
                       stage in {"part1", "part2"} and re.fullmatch(r"b[0-9]+", boundary) is not None)
        identity_ok = (family in {"hailo8", "hailo10h", "deepx"} and bool(row.get("model_id"))
                       and boundary_ok and not conflicts)
        negative = known_negative_build_evidence(current)
        available = state == "completed" and identity_ok
        if conflicts:
            reason = "deferred_build_identity_conflict:" + ",".join(conflicts)
        elif not identity_ok:
            reason = "deferred_build_identity_unresolved"
        elif state == "known_infeasible" and not negative:
            reason = "deferred_negative_evidence_unconfirmed"
        else:
            reason = str(current.get("primary_failure_reason") or current.get("error")
                         or negative.get("reason") or ("" if available else "deferred_build_" + state))
        count = current.get("compiler_dispatch_count")
        if count is None and current.get("cache_hit") is True:
            count = 0
        row.update(status=state, artifact_ready=available,
                   readiness="ready" if available else "not_executable" if state == "known_infeasible" and negative and identity_ok else "blocked" if state == "failed" and identity_ok else "unconfirmed",
                   primary_failure_reason=reason, failure_stage="build_backend_artifacts",
                   upstream_evidence_path=path,
                   compiler_dispatch_count=count if type(count) is int and count >= 0 else None,
                   identity_conflicts=conflicts)
        if len(trail) > 1:
            row["previous_observations"] = trail[:-1]
        projected.append(row)
    blocked = [row for row in projected if not row["artifact_ready"]]
    artifacts = [row for row in projected if row.get("job_kind", "artifact") == "artifact"]
    return {
        "status": "partial" if blocked else "ok", "jobs": projected, "blocked_jobs": blocked,
        "required_artifact_job_count": len(artifacts),
        "ready_artifact_job_count": sum(row["artifact_ready"] for row in artifacts),
        "failed_artifact_job_count": sum(row["status"] == "failed" for row in artifacts),
        "known_infeasible_artifact_job_count": sum(row["readiness"] == "not_executable" for row in artifacts),
        "unconfirmed_artifact_job_count": sum(row["readiness"] == "unconfirmed" for row in artifacts),
        "required_collection_request_count": len(projected) - len(artifacts),
        "blocked_collection_request_count": sum(row.get("job_kind") == "collection" for row in blocked),
        "count_unit": "bound artifact requests; collective requests counted separately",
        "backend_states": {family: "partial" if any(not row["artifact_ready"] for row in projected if row["backend"] == family) else "ok"
                           for family in sorted({row["backend"] for row in projected})},
    }


def _record_previous_observation(request: dict[str, Any]) -> None:
    previous = request.get("build_observation")
    if isinstance(previous, Mapping):
        request.setdefault("previous_build_observations", []).append({
            "status": request.get("status"), "error": request.get("build_error", ""),
            **dict(previous),
        })


def _call_selected_builder(build_fn: Callable[..., Any], source: str, args: Mapping[str, Any]) -> Any:
    # Only this known job-local preparation error is converted to a result.
    # Policy, scope, source mutation, cancel and unknown exceptions propagate.
    from ..hailo_compiler_context import CompilerContextError
    try:
        return build_fn(source, **args)
    except CompilerContextError as exc:
        return SimpleNamespace(ok=False, error=f"{type(exc).__name__}: {exc}",
                               failure_kind="compiler_context_error", calib_info={},
                               details={"exception_type": type(exc).__name__, "exception_details": dict(exc.details),
                                        "compiler_dispatch_count": None})


def cache_preflight_builder(builder: Callable[..., Any]) -> Callable[..., Any]:
    """Probe the real builder's exact cache path and persist its later call."""
    def probe(model_path: Any, **kwargs: Any) -> Any:
        source = Path(model_path).resolve(strict=True)
        output = Path(kwargs["outdir"]).resolve()
        output.mkdir(parents=True, exist_ok=True)
        saved_kwargs = {
            key: value for key, value in kwargs.items()
            if key != "on_log" and not callable(value)
        }
        saved_kwargs["outdir"] = str(output)
        request = {
            "schema": "onnx-splitpoint/deferred-hailo-build/v1",
            "status": "probing",
            "source_onnx": str(source),
            "source_onnx_sha256": sha256_file(source),
            "kwargs": saved_kwargs,
            "environment": {key: os.environ.get(key) for key in _BUILD_ENV},
        }
        # Preserve the original build policy in the request.  A force-build
        # profile still probes without forcing optimization before the barrier.
        write_json(output / REQUEST_NAME, request)
        probe_kwargs = dict(kwargs, cache_only=True, force=False)
        result = builder(str(source), **probe_kwargs)
        negative = _result_negative_evidence(result)
        request["status"] = "known_infeasible" if negative else "pending"
        # Persist the exact decision from this probe, not just failure text.
        # A later run with a changed recipe/endpoints creates a fresh request.
        request["build_evidence"] = negative
        request["cache_probe_details"] = dict(getattr(result, "details", None) or {})
        request["cache_probe_ok"] = bool(getattr(result, "ok", False))
        request["cache_probe_error"] = str(getattr(result, "error", "") or "")
        request["cache_probe_status"], request["cache_probe_reason"] = _probe_classification(result)
        write_json(output / REQUEST_NAME, request)
        return result
    return probe


def selection_preflight_builder(
    builder: Callable[..., Any], *, model_id: str,
    profile_payload: Mapping[str, Any], log: Optional[Callable[[str], None]] = None,
) -> Callable[..., Any]:
    """Report compiler-dependent Gate-A candidates before each cold dispatch.

    This scope is explicitly provisional: Gate-A must test candidate artifacts
    before it can name a feasible final boundary.  The final campaign matrix
    still runs after that controller freezes its accepted selection.
    """
    from ..cache_verify_policy import CacheVerifyPolicyError, cache_verify_guard
    from .artifact_cache_preflight import resolve_artifact_cache_preflight_policy, _expectation_for
    policy = resolve_artifact_cache_preflight_policy(profile_payload)
    def probe(source: Any, **kwargs: Any) -> Any:
        output = Path(kwargs["outdir"])
        output.mkdir(parents=True, exist_ok=True)
        probe_error = None
        try:
            result = builder(source, **dict(kwargs, cache_only=True, force=False))
        except Exception as exc:
            probe_error = exc
            result = SimpleNamespace(ok=False, failure_kind="cache_probe_failed", error=f"{type(exc).__name__}: {exc}")
        status, reason = _probe_classification(result)
        if status != "KNOWN_INFEASIBLE" and kwargs.get("force") and not kwargs.get("cache_only"):
            status = "MISS"
            reason = "force_build_requested"
        match = re.search(r"_b(\d+)$", str(kwargs.get("net_name") or ""))
        boundary = f"b{int(match.group(1)):03d}" if match else "full"
        arch = str(kwargs.get("hw_arch") or "")
        role = "hailo10_hef" if "hailo10" in arch else "hailo8_hef"
        stage = str((kwargs.get('build_evidence_context') or {}).get('stage') or 'part1')
        expectation = _expectation_for(policy, model_id=model_id, role=role, item_id=f"{boundary}:{stage}")
        blocked = bool(
            (cache_verify_guard(profile_payload) and status not in {"HIT", "KNOWN_INFEASIBLE"})
            or policy.get("block_on_unexpected_cold_builds")
            and (status == "UNKNOWN" or status == "MISS" and expectation == "warm")
        )
        report = {
            "schema": "onnx-splitpoint/selection-probe-cache-preflight/v1",
            "scope": "selection_probe", "model_id": model_id,
            "boundary": boundary, "backend": arch, "stage": stage,
            "status": status, "expectation": expectation,
            "reason": reason,
            "expected_cold_build": status == "MISS",
            "compiler_dispatch_allowed": not blocked and status not in {"HIT", "KNOWN_INFEASIBLE"},
            "build_evidence": _result_negative_evidence(result),
        }
        write_json(output / "selection_probe_cache_preflight.json", report)
        if callable(log):
            log(f"[cache-preflight] scope=selection_probe model={model_id} boundary={boundary} backend={arch} status={status} reason={report['reason']} expected_cold_build={status == 'MISS'} compiler_dispatch_allowed={report['compiler_dispatch_allowed']}")
        if blocked:
            raise CacheVerifyPolicyError(
                f"cache_preflight_selection_blocked: model={model_id} boundary={boundary} backend={arch} status={status} reason={report['reason']}"
            )
        if probe_error is not None:
            raise probe_error
        if kwargs.get("cache_only") or status in {"HIT", "KNOWN_INFEASIBLE"}:
            return result
        return builder(source, **kwargs)
    return probe


def defer_deepx_part1_build(**kwargs: Any) -> dict[str, Any]:
    """Keep the existing DeepX materializer's exact arguments for continuation."""
    output = Path(kwargs["out_dir"])
    saved = {key: value for key, value in kwargs.items() if not callable(value)}
    saved["out_dir"] = str(output)
    suite = _read(output / "benchmark_set.json")
    sources = {}
    selected = []
    for case in suite.get("cases", []):
        folder = str(case.get("case_dir") or case.get("folder") or "")
        if 'selected_case_dirs' in kwargs and folder not in kwargs['selected_case_dirs']:
            continue
        selected.append(folder)
        manifest = _read(output / folder / str(case.get("manifest") or "split_manifest.json"))
        source = Path(str(manifest.get("part1_model") or manifest.get("part1") or manifest.get("part1_path") or ""))
        if not source.is_absolute():
            source = output / folder / source
        if source.is_file():
            sources[str(source.resolve())] = sha256_file(source)
    saved["selected_case_dirs"] = selected
    write_json(output / DEEPX_PART1_REQUEST, {"status": "pending", "kwargs": saved, "source_onnx_sha256": sources})
    return {"status": "deferred_until_cache_preflight", "compiler_dispatched": False}


def probe_deferred_deepx_part1_cache(suite: Path, *, log: Any = None) -> dict[str, Any]:
    """Resolve the selected Part1 receipt before dependent TRT cache probes.

    Use the generator's saved exact request and leave its build continuation
    pending. No compiler/SDK probe is needed for a verified reuse hit.
    """
    path = suite / DEEPX_PART1_REQUEST
    if not path.is_file():
        return {}
    request = _read(path)
    args = dict(request['kwargs'])
    selected = {str(c.get('case_dir') or c.get('folder') or '')
                for c in _read(suite / 'benchmark_set.json').get('cases', [])}
    args['selected_case_dirs'] = sorted(selected.intersection(args.get('selected_case_dirs', selected)))
    try:
        if Path(args['out_dir']).resolve() != suite.resolve():
            raise RuntimeError('Deferred DeepX Part1 suite identity mismatch')
        for source_path, digest in request.get('source_onnx_sha256', {}).items():
            source = Path(source_path)
            if not source.is_file() or sha256_file(source) != digest:
                raise RuntimeError(f'Selected DeepX ONNX changed before cache preflight: {source}')
    except Exception as exc:
        for folder in args['selected_case_dirs']:
            write_json(suite / folder / 'deepx/deepx_m1/part1/deepx_part1_artifact_status.json', {
                'ok': False, 'status': 'failed', 'error': str(exc),
                'cache_lookup': {'outcome': 'UNKNOWN', 'reason': 'native_part1_identity_unavailable'},
            })
        raise
    args.update(out_dir=suite, force_build=False, cache_only=True)
    if callable(log):
        args['log'] = lambda msg, **_kw: log(str(msg))
    from ..gui.benchmark_workflow import _materialize_manual_deepx_part1_artifacts
    return _materialize_manual_deepx_part1_artifacts(**args)


def _finalize_deepx(suite: Path, selected: set[str], log: Any, cancel_event: Any) -> list[dict[str, Any]]:
    jobs = []
    p1_request = suite / DEEPX_PART1_REQUEST
    if p1_request.is_file():
        request = _read(p1_request)
        if request.get("status") in {"pending", "failed", "completed"}:
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("Deferred DeepX build cancelled before dispatch")
            from ..gui.benchmark_workflow import _materialize_manual_deepx_part1_artifacts
            args = dict(request["kwargs"])
            args["out_dir"] = Path(args["out_dir"])
            args["selected_case_dirs"] = sorted(selected.intersection(args.get("selected_case_dirs", selected)))
            if request.get("status") == "completed":
                args["force_build"] = False
            for source_path, digest in request.get("source_onnx_sha256", {}).items():
                source = Path(source_path)
                if not source.is_file() or sha256_file(source) != digest:
                    raise RuntimeError(f"Selected DeepX ONNX changed after cache preflight: {source}")
            if callable(log):
                args["log"] = lambda msg, **_kw: log(str(msg))
            from ..backend_backfill import call_with_build_budget
            result = call_with_build_budget(_materialize_manual_deepx_part1_artifacts,
                state_path=suite / 'generation_state.json', stage='part1', **args)
            request["status"] = "completed" if result.get("status") in {"ok", "not_selected"} and not result.get("failed_count") else "failed"
            request["result"] = result
            write_json(p1_request, request)
            write_json(suite.parent / "deepx_part1_artifact_status.json", result)
        jobs.append({**_request_context(request, p1_request, model_id=suite.parent.parent.name,
                                       backend="deepx", stage="part1", collection=True),
                     "status": request["status"], "result": request.get("result") or {},
                     "error": str((request.get("result") or {}).get("error") or (request.get("result") or {}).get("reason") or "")})
    for path in sorted(suite.glob(f"*/{DEEPX_PART2_REQUEST}")):
        if path.parent.name not in selected:
            continue
        request = _read(path)
        previous_dxnn = Path(str(request.get("result", {}).get("dxnn_path") or ""))
        if request.get("status") == "completed" and previous_dxnn.is_file() and request.get("artifact_sha256") == sha256_file(previous_dxnn):
            jobs.append({**_request_context(request, path, model_id=suite.parent.parent.name,
                                           backend="deepx", stage="part2", boundary=path.parent.name),
                         "status": "completed", "resumed": True})
            continue
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("Deferred DeepX build cancelled before dispatch")
        args = dict(request["kwargs"])
        source = Path(args["part2_onnx"])
        if not source.is_file() or sha256_file(source) != request["source_onnx_sha256"]:
            raise RuntimeError(f"Selected DeepX ONNX changed after cache preflight: {source}")
        from ..deepx.activation_proxy import compile_deepx_stage2_from_activation_proxy
        from ..backend_backfill import call_with_build_budget
        result = call_with_build_budget(compile_deepx_stage2_from_activation_proxy,
            state_path=suite / 'generation_state.json', stage='part2', **args)
        manifest_path = path.parent / "split_manifest.json"
        manifest = _read(manifest_path)
        deepx = manifest.setdefault("deepx", {})
        proxy = deepx.setdefault("stage2_calibration", {})
        proxy.pop("build_deferred_until_cache_preflight", None)
        deepx["stage2_dxnn_build"] = dict(result)
        write_json(path.parent / "deepx" / "deepx_m1" / "part2" / "deepx_part2_artifact_status.json", result)
        dxnn_rel = os.path.relpath(result["dxnn_path"], path.parent).replace("\\", "/") if result.get("dxnn_path") else ""
        cfg_rel = os.path.relpath(result["config_path"], path.parent).replace("\\", "/") if result.get("config_path") else ""
        deepx["part2_dxnn"] = {
            "ok": bool(result.get("ok")), "status": "ok" if result.get("ok") else str(result.get("status") or "failed"),
            "dxnn_path": dxnn_rel, "config_path": cfg_rel,
            "activation_manifest": str(proxy.get("manifest_rel") or ""),
            "experimental": True, "source": proxy.get("source"),
            "trust_level": proxy.get("trust_level"), "producer_exact": proxy.get("producer_exact"),
        }
        if result.get("ok"):
            manifest["deepx_part2_dxnn"] = dxnn_rel
            manifest["deepx_part2_activation_manifest"] = str(proxy.get("manifest_rel") or "")
        write_json(manifest_path, manifest)
        request["status"] = "completed" if result.get("ok") else "failed"
        request["result"] = dict(result)
        if result.get("ok") and result.get("dxnn_path"):
            request["artifact_sha256"] = sha256_file(Path(result["dxnn_path"]))
        write_json(path, request)
        jobs.append({**_request_context(request, path, model_id=suite.parent.parent.name,
                                       backend="deepx", stage="part2", boundary=path.parent.name),
                     "status": request["status"], "result": dict(result),
                     "error": str(result.get("error") or result.get("reason") or "")})
    return jobs


def _case_manifest(suite: Path, output: Path) -> Optional[Path]:
    parent = output.parent.parent.parent
    if parent != suite and parent.parent == suite:
        return parent / "split_manifest.json"
    return None


def _update_build_metadata(suite: Path, request: Mapping[str, Any], result: Any) -> None:
    from ..benchmark.services import BenchmarkGenerationService

    args = dict(request["kwargs"])
    output = Path(args["outdir"])
    variant = output.name
    arch = str(args["hw_arch"])
    manifest = _case_manifest(suite, output)
    destination = manifest or suite / "benchmark_set.json"
    payload = _read(destination)
    from ..benchmark.services import _canonical_hailo_evidence_arch
    hefs = payload.setdefault("hailo", {}).setdefault("hefs", {})
    arch_key = arch if arch in hefs else _canonical_hailo_evidence_arch(arch)
    meta = hefs.setdefault(arch_key, {})
    summary = BenchmarkGenerationService().compact_hailo_build_summary(result)
    summary.update({
        "source_onnx_path": str(request["source_onnx"]),
        "compiler_onnx_path": str(getattr(result, "fixed_onnx_path", None) or request["source_onnx"]),
        "build_receipt_path": str(output / "hailo_hef_build_receipt.json"),
    })
    meta[f"{variant}_build"] = summary
    for suffix in ("pending_preflight", "deferred", "deferred_reason", "cold_build_request"):
        meta.pop(f"{variant}_{suffix}", None)
    if bool(getattr(result, "ok", False)):
        hef = Path(getattr(result, "hef_path", None) or output / "compiled.hef")
        if not hef.is_file():
            raise RuntimeError(f"Hailo builder reported success without HEF: {hef}")
        meta[variant] = os.path.relpath(hef, destination.parent).replace("\\", "/")
        meta.pop(f"{variant}_error", None)
    else:
        meta.pop(variant, None)
        meta[f"{variant}_error"] = str(getattr(result, "error", "") or "Hailo build failed")
    if variant == "full":
        meta["full_required"] = True
        if request.get("full_output_contract"):
            meta["full_output_contract"] = dict(request["full_output_contract"])
            meta["full_end_node_names"] = list(args.get("end_node_names") or [])
            meta["full_endpoint_mode"] = "raw_detection_head"
            meta["full_true_full_model"] = False
    payload.setdefault("hailo", {}).pop("builds_deferred_until_cache_preflight", None)
    write_json(destination, payload)


def _refresh_suite_mirrors(suite: Path, formal: Path) -> None:
    from ..benchmark.services import (
        BenchmarkGenerationService,
        build_case_hailo_variant_availability,
        case_hailo_backend_terminal_states,
    )

    payload = _read(suite / "benchmark_set.json")
    suite_hefs = payload.get("hailo", {}).get("hefs", {})
    for case in payload.get("cases", []):
        folder = str(case.get("case_dir") or case.get("folder") or "")
        manifest = suite / folder / str(case.get("manifest") or "split_manifest.json")
        if not manifest.is_file():
            continue
        data = _read(manifest)
        hailo = data.setdefault("hailo", {})
        hefs = hailo.setdefault("hefs", {})
        # Full artifacts live at suite scope; the case uses relative aliases.
        for arch, full in suite_hefs.items():
            target = hefs.setdefault(arch, {})
            for key, value in full.items():
                if key.startswith("full"):
                    target[key] = value
            if full.get("full"):
                target["full"] = os.path.relpath(suite / full["full"], manifest.parent).replace("\\", "/")
        availability = build_case_hailo_variant_availability(suite_hefs, hefs)
        terminal = case_hailo_backend_terminal_states(payload.get("plan", {}).get("runs", []), availability)
        hailo["case_variant_availability"] = availability
        hailo["backend_terminal_states"] = terminal
        case["hailo_case_variant_availability"] = availability
        case["hailo_backend_terminal_states"] = terminal
        case["hailo_compile"] = BenchmarkGenerationService().case_hailo_compile_from_hefs(hefs)
        write_json(manifest, data)
    write_json(suite / "benchmark_set.json", payload)
    state_path = suite / "generation_state.json"
    if state_path.is_file():
        state = _read(state_path)
        state["cases"] = payload.get("cases", [])
        state["suite_hailo_hefs"] = suite_hefs
        write_json(state_path, state)
        if state.get('backend_backfill'):
            from ..backend_backfill import bind_plan_cases
            payload['backend_backfill'] = state['backend_backfill']
            write_json(suite / 'benchmark_set.json', payload)
            write_json(formal / 'backend_selection.json', state['backend_backfill'])
            for directory in (suite, formal):
                plan_path = directory / 'benchmark_plan.json'
                if plan_path.is_file():
                    plan = _read(plan_path)
                    bind_plan_cases(plan, state['backend_backfill'])
                    write_json(plan_path, plan)
    for name in ("benchmark_set.json", "generation_decisions.json"):
        path = formal / name
        if path.is_file():
            mirror = _read(path)
            key = "accepted_cases" if name == "generation_decisions.json" else "cases"
            mirror[key] = payload.get("cases", [])
            if payload.get('backend_backfill'):
                mirror['backend_backfill'] = payload['backend_backfill']
            mirror["hailo_build_continuation_completed"] = True
            write_json(path, mirror)


def _refresh_saved_negative_probe(
    *, suite: Path, request_path: Path, request: dict[str, Any],
    build_fn: Callable[..., Any], log: Any = None, process_registry: Any = None,
    before_cache_preflight: bool = False,
) -> Any:
    """Revalidate a historical negative against today's backend identity.

    The saved builder source remains the selected split contract. Compiler,
    calibration, context and rescued positive artifacts are inspected anew by
    the ordinary backend, with compiler dispatch forbidden for this probe.
    """
    source = Path(request["source_onnx"])
    if not source.is_file() or sha256_file(source) != request["source_onnx_sha256"]:
        raise RuntimeError(f"Selected Hailo ONNX changed after cache preflight: {source}")
    args = dict(request["kwargs"], cache_only=True, force=False)
    if callable(log):
        args["on_log"] = lambda stream, line: log(str(line))
    old_env = {key: os.environ.get(key) for key in _BUILD_ENV}
    try:
        for key, value in request.get("environment", {}).items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = str(value)
        from ..process_control import bind_process_registry, current_process_registry
        with bind_process_registry(process_registry or current_process_registry()):
            try:
                result = build_fn(str(source), **args)
            except Exception as exc:
                # This remains a visible UNKNOWN probe; neither a transient
                # probe error nor unavailable identity licenses a cold build.
                result = SimpleNamespace(ok=False, failure_kind="cache_probe_failed",
                                         error=f"{type(exc).__name__}: {exc}", details={})
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    previous = known_negative_build_evidence(request)
    negative = _result_negative_evidence(result)
    status, reason = _probe_classification(result)
    if previous:
        request["previous_negative_build_evidence"] = previous
    request.update({
        "status": "known_infeasible" if negative else "completed" if getattr(result, "ok", False) else "pending",
        "build_evidence": negative,
        "cache_probe_details": dict(getattr(result, "details", None) or {}),
        "cache_probe_ok": bool(getattr(result, "ok", False)),
        "cache_probe_error": str(getattr(result, "error", "") or ""),
        "cache_probe_status": status,
        "cache_probe_reason": reason,
        # A changed decision found after the final matrix must be reported in
        # a new matrix before a later invocation may run its cold compiler.
        "cache_preflight_refresh_required": bool(
            not before_cache_preflight and status not in {"HIT", "KNOWN_INFEASIBLE"}
        ),
    })
    write_json(request_path, request)
    _update_build_metadata(suite, request, result)
    if callable(log):
        log(f"[cache-preflight] refreshed negative evidence: model={args.get('net_name')} backend={args.get('hw_arch')} status={status} reason={reason} compiler_dispatch_allowed=false")
    return result


def refresh_deferred_hailo_negative_probes(
    *, model_dir: Path, log: Optional[Callable[[str], None]] = None,
    process_registry: Any = None, cancel_event: Any = None,
    build_fn: Optional[Callable[..., Any]] = None,
) -> dict[str, Any]:
    """Refresh saved negative decisions before the final campaign matrix.

    This is a cache-only pass on resume. A rescued HEF becomes HIT; a changed
    compiler/recipe/calibration becomes its actual current MISS/UNKNOWN and is
    visible in the matrix before the existing continuation can build it.
    """
    formal = Path(model_dir) / "benchmark_set"
    suite = formal / "legacy_suite"
    if not (suite / "benchmark_set.json").is_file():
        return {"status": "not_applicable", "jobs": []}
    payload = _read(suite / "benchmark_set.json")
    selected = {str(case.get("case_dir") or case.get("folder") or "")
                for case in payload.get("cases", []) if isinstance(case, Mapping)}
    jobs = []
    for request_path in sorted(suite.rglob(REQUEST_NAME)):
        request = _read(request_path)
        if request.get("status") != "known_infeasible" and not request.get("cache_preflight_refresh_required"):
            continue
        output = Path(request["kwargs"]["outdir"])
        manifest = _case_manifest(suite, output)
        if manifest is not None and manifest.parent.name not in selected:
            continue
        if output.parent.parent.parent != suite and manifest is None:
            continue
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("Deferred Hailo negative refresh cancelled before probe")
        if build_fn is None:
            from ..gui.benchmark_workflow import resolve_hailo_benchmark_helpers
            build_fn = resolve_hailo_benchmark_helpers(need_build=True, need_part2=False).hailo_build_hef_fn
        if build_fn is None:
            raise RuntimeError("Deferred Hailo negative refresh has no cache probe builder")
        _refresh_saved_negative_probe(
            suite=suite, request_path=request_path, request=request, build_fn=build_fn,
            log=log, process_registry=process_registry, before_cache_preflight=True,
        )
        jobs.append({"request": str(request_path), "status": request["cache_probe_status"],
                     "reason": request["cache_probe_reason"], "compiler_dispatched": False})
    if jobs:
        _refresh_suite_mirrors(suite, formal)
    return {"status": "refreshed" if jobs else "not_applicable", "jobs": jobs}


def finalize_deferred_hailo_builds(
    *, model_dir: Path, run_dir: Path, profile_payload: Mapping[str, Any],
    log: Optional[Callable[[str], None]] = None, process_registry: Any = None,
    cancel_event: Any = None, build_fn: Optional[Callable[..., Any]] = None,
) -> dict[str, Any]:
    """Execute persisted calls only for the frozen suite's selected cases.

    The runner invokes this after saving/logging the complete cache matrix.
    Source changes block dispatch; rejected/unselected case requests are ignored.
    """
    from ..cache_verify_policy import cache_verify_guard

    formal = Path(model_dir) / "benchmark_set"
    suite = formal / "legacy_suite"
    if not (suite / "benchmark_set.json").is_file():
        return {"status": "not_applicable", "jobs": []}
    requests = sorted(suite.rglob(REQUEST_NAME))
    if not requests and not (suite / DEEPX_PART1_REQUEST).is_file() and not list(suite.glob(f"*/{DEEPX_PART2_REQUEST}")):
        return {"status": "not_applicable", "jobs": []}
    cache_verify = bool(cache_verify_guard(profile_payload))
    from .artifact_cache_preflight import resolve_artifact_cache_preflight_policy, _expectation_for
    cache_policy = resolve_artifact_cache_preflight_policy(profile_payload)
    payload = _read(suite / "benchmark_set.json")
    selected = {
        str(case.get("case_dir") or case.get("folder") or "")
        for case in payload.get("cases", []) if isinstance(case, Mapping)
    }
    if requests and build_fn is None:
        from ..gui.benchmark_workflow import resolve_hailo_benchmark_helpers
        build_fn = resolve_hailo_benchmark_helpers(need_build=True, need_part2=False).hailo_build_hef_fn
    if requests and build_fn is None:
        raise RuntimeError("Deferred Hailo build continuation has no builder")
    from ..build_dispatch_policy import bind_profile_hailo_builder, require_productive_force_off
    require_productive_force_off(profile_payload)
    if build_fn is not None:
        build_fn = bind_profile_hailo_builder(build_fn, profile_payload)
    jobs = []
    selected_requests = []
    for request_path in requests:
        request = _read(request_path)
        if request.get("selected") is False or request.get("required") is False or request.get("status") == "not_applicable":
            continue
        args = dict(request["kwargs"])
        if cache_verify:
            args.update(cache_only=True, force=False)
        output = Path(args["outdir"])
        case_manifest = _case_manifest(suite, output)
        if case_manifest is not None and case_manifest.parent.name not in selected:
            continue
        if output.resolve() != request_path.parent.resolve() or not output.resolve().is_relative_to(suite.resolve()):
            raise ValueError(f"deferred_request_output_scope_invalid:{request_path}")
        if output.parent.parent.parent != suite and case_manifest is None:
            raise ValueError(f"deferred_request_stage_scope_invalid:{request_path}")
        context = _request_context(request, request_path, model_id=Path(model_dir).name)
        cache_role = "hailo10_hef" if "hailo10" in context["backend"] else "hailo8_hef"
        strict_warm = bool(cache_policy.get("enabled", True) and cache_policy.get("block_on_unexpected_cold_builds") and
            _expectation_for(cache_policy, model_id=context["model_id"], role=cache_role,
                item_id=f"{context['boundary']}:{context['stage']}") == "warm")
        if strict_warm:
            # The barrier's HIT may disappear before continuation. Preserve
            # the frozen warm contract at the actual builder boundary too.
            args.update(cache_only=True, force=False)
        binding = args.get("build_evidence_context") or {}
        # A logical suite ID is authoritative; legacy suites may not declare
        # one and need not share their directory basename with the model name.
        suite_model_id = str(payload.get("model_id") or payload.get("model_name") or "")
        if (suite_model_id and binding.get("model_id") and str(binding["model_id"]) != suite_model_id
                or binding.get("stage") and binding["stage"] != output.name
                or case_manifest is not None and context["boundary"] != "unknown"
                and context["boundary"] != case_manifest.parent.name):
            raise ValueError(f"deferred_request_identity_scope_invalid:{request_path}")
        selected_requests.append(context)
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("Deferred Hailo builds cancelled before dispatch")
        source = Path(request["source_onnx"])
        if not source.is_file() or sha256_file(source) != request["source_onnx_sha256"]:
            raise RuntimeError(f"Selected Hailo ONNX changed after cache preflight: {source}")
        if request.get("cache_preflight_refresh_required"):
            raise RuntimeError(
                f"cache_preflight_refresh_required: selected Hailo identity changed after the final matrix: {args.get('net_name')} {args.get('hw_arch')}"
            )
        if request.get("status") == "known_infeasible":
            result = _refresh_saved_negative_probe(
                suite=suite, request_path=request_path, request=request, build_fn=build_fn,
                log=log, process_registry=process_registry,
            )
            if request["cache_probe_status"] in {"HIT", "KNOWN_INFEASIBLE"}:
                observation = _completed_job_observation(request, result, model_id=Path(model_dir).name)
                jobs.append({**context, **observation, "status": request["status"],
                             "build_evidence": request["build_evidence"], "compiler_dispatched": False,
                             "error": str(getattr(result, "error", "") or "")})
                continue
            _refresh_suite_mirrors(suite, formal)
            if cache_verify:
                from ..cache_verify_policy import CacheVerifyPolicyError
                raise CacheVerifyPolicyError(
                    f"cache_miss_blocked[hailo_dfc]: negative evidence changed to {request['cache_probe_status']}; cache_preflight_refresh_required: {args.get('net_name')} {args.get('hw_arch')}"
                )
            raise RuntimeError(
                f"cache_preflight_refresh_required: negative evidence changed to {request['cache_probe_status']} after final matrix: {args.get('net_name')} {args.get('hw_arch')}"
            )
        if request.get("status") == "completed":
            # Re-enter the receipt-aware cache lookup after resume.  A missing
            # or damaged artifact must not inherit an old success flag.
            args["force"] = False
        if callable(log):
            log(f"[cache-preflight] selected Hailo continuation: {args.get('net_name')} {args.get('hw_arch')}")
            args["on_log"] = lambda stream, line: log(str(line))
        # The captured cache_only flag encodes actual policy (e.g. Smoke Full
        # cache_or_defer).  Global deferral is applied only by the wrapper.
        old_env = {key: os.environ.get(key) for key in _BUILD_ENV}
        attempt_observations: list[dict[str, Any]] = []
        try:
            for key, value in request.get("environment", {}).items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = str(value)
            from ..process_control import bind_process_registry, current_process_registry
            with bind_process_registry(process_registry or current_process_registry()):
                result = _call_selected_builder(build_fn, str(source), args)
            attempt_observations.append(_completed_job_observation(dict(request, kwargs=args), result, model_id=Path(model_dir).name))
            if (output.name == "full" and not getattr(result, "ok", False)
                    and not args.get("end_node_names") and not args.get("cache_only")
                    and getattr(result, "failure_kind", "") not in {
                        "local_dfc_workspace_insufficient", "local_dfc_workspace_unresolved",
                        "local_dfc_workspace_exhausted"}):
                from ..benchmark.model_preparation import infer_yolo_raw_detection_head_end_nodes
                raw_nodes = infer_yolo_raw_detection_head_end_nodes(str(source), "")
                if raw_nodes:
                    # Preserve the generator's decoded -> raw-head fallback.
                    # Its changed endpoint identity gets a fresh exact probe
                    # and visible decision before the fallback compiler call.
                    retry = dict(args, end_node_names=list(raw_nodes), force=False)
                    probe = _call_selected_builder(build_fn, str(source), dict(retry, cache_only=True))
                    attempt_observations.append(_completed_job_observation(dict(request, kwargs=retry), probe, model_id=Path(model_dir).name))
                    if callable(log):
                        fallback_status, fallback_reason = _probe_classification(probe)
                        log(f"[cache-preflight] full raw-head fallback {args.get('hw_arch')}: status={fallback_status}; reason={fallback_reason}; trigger=decoded_full_build_failed; end_nodes={list(raw_nodes)}")
                    if getattr(probe, "ok", False) or _result_negative_evidence(probe):
                        result = probe
                    else:
                        result = _call_selected_builder(build_fn, str(source), retry)
                        attempt_observations.append(_completed_job_observation(dict(request, kwargs=retry), result, model_id=Path(model_dir).name))
                    request["kwargs"] = {key: value for key, value in retry.items() if key != "on_log"}
                    request["full_output_contract"] = {
                        "mode": "yolo26_one2one_raw_head" if any("one2one_cv" in str(node) for node in raw_nodes) else "raw_detection_head",
                        "requires_external_postprocess": True,
                        "description": "YOLO raw detection-head tensors; decode/NMS remains outside the HEF.",
                        "end_node_names": list(raw_nodes),
                    }
        finally:
            for key, value in old_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
        _update_build_metadata(suite, request, result)
        negative = _result_negative_evidence(result)
        _record_previous_observation(request)
        request["status"] = "completed" if bool(getattr(result, "ok", False)) else "known_infeasible" if negative else "failed"
        request["build_evidence"] = negative
        request["build_error"] = str(getattr(result, "error", "") or "")
        request["build_observation"] = _completed_job_observation(
            request, result, model_id=Path(model_dir).name,
        )
        dispatch_counts = [row.get("compiler_dispatch_count") for row in attempt_observations]
        request["build_observation"]["compiler_dispatch_count"] = (
            sum(dispatch_counts) if dispatch_counts and all(type(value) is int for value in dispatch_counts) else None
        )
        request["build_observation"]["attempt_observations"] = attempt_observations
        write_json(request_path, request)
        jobs.append({**context, **request["build_observation"], "status": request["status"],
                     "error": request["build_error"], "build_evidence": negative,
                     "previous_build_observations": request.get("previous_build_observations", [])})
        if callable(log):
            observed = request["build_observation"]
            log(f"[build-result] model={observed['model_id']} boundary={observed['boundary']} backend={observed['backend']} stage={observed['stage']} preflight={observed['preflight_decision']} cache_hit={observed['cache_hit']} compiler_dispatch_count={observed['compiler_dispatch_count']} status={request['status']}")
        if cache_verify and request["status"] not in {"completed", "known_infeasible"}:
            from ..cache_verify_policy import CacheVerifyPolicyError
            raise CacheVerifyPolicyError(
                f"cache_miss_blocked[hailo_dfc]: deferred selected artifact {args.get('net_name')} {args.get('hw_arch')} could not restore its exact cache entry: {request['build_error']}"
            )
    from ..process_control import bind_process_registry, current_process_registry
    from ..cache_verify_policy import bind_artifact_policy
    with bind_artifact_policy(profile_payload), bind_process_registry(process_registry or current_process_registry()):
        deepx_jobs = _finalize_deepx(suite, selected, log, cancel_event)
        jobs.extend(deepx_jobs)
        selected_requests.extend({key: job[key] for key in
            ("request", "model_id", "backend", "boundary", "stage", "job_kind", "identity_source", "selected") if key in job}
            for job in deepx_jobs)
    _refresh_suite_mirrors(suite, formal)
    result = {"status": "partial" if any(job["status"] == "failed" for job in jobs) else "completed",
              "jobs": jobs, "selected_requests": selected_requests}
    result["readiness"] = project_deferred_build_readiness(result)
    return result
