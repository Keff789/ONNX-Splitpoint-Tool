from __future__ import annotations

from ..config_values import parse_config_bool, validate_profile_config_booleans

from onnx_splitpoint_tool.native_full_quality import (
    enabled_run_profiles,
    normalise_evaluation_profile,
    resolve_native_split_plan,
)
"""Evaluation Workflow binding for the existing BenchmarkSet generator.

The formal workflow must not reimplement the benchmark pipeline.  This module is
an adapter around the mature BenchmarkGenerationOrchestrationService used by the
Benchmark tab: split export, Hailo HEF build/reuse, YOLO raw-head policies,
benchmark_set.json, benchmark_suite.py and generated case folders remain the
same source of truth.  The Evaluation Workflow only wraps that suite in a
resumable EvaluationRuns bundle and records the provenance.
"""

import concurrent.futures
import json
import os
import re
import shutil
import copy
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from .artifacts import (
    now_iso,
    relpath,
    sha256_file,
    sha256_json,
    sha256_payload,
    slugify,
    write_csv,
    write_json,
    write_text,
)
from .dataset_binding import calibration_manifest_for_task, validation_manifest_for_task, manifest_dataset_root
from .artifact_registry_binding import register_benchmark_set_artifacts
from ..cache_verify_policy import (
    CacheVerifyPolicyError,
    cache_miss_blocked_message,
    cache_verify_guard,
)
from ..process_control import current_process_registry
from ..hailo_timeout_policy import parse_hailo_timeout_seconds
from .deferred_hailo_builds import cache_preflight_builder, finalize_deferred_hailo_builds, defer_deepx_part1_build, selection_preflight_builder
from ..management_reference import (
    bind_management_cpu_reference_runs,
    finalize_management_cpu_reference_plan_aliases,
    management_cpu_reference_required,
    profile_has_explicit_cpu_reference,
)
from .full_only_quality_canary import (
    project_full_only_quality_plan_rows,
    resolve_full_only_quality_canary,
)


_DEEPX_PREFETCH_CANCEL_GRACE_S = 8.0


def _hailo_feasibility_stop_workflow_v2783(
    outcome: Any,
    control: Mapping[str, Any],
) -> bool:
    terminal = str(outcome or "").strip()
    if terminal == "EVIDENCE_CONFLICT":
        return True
    return bool(
        terminal == "CANARY_BUDGET_EXHAUSTED"
        and control.get("stop_workflow_on_exhaustion", True)
    )


def _benchmark_service_log_adapter(
    log: Callable[[str], None],
) -> Callable[..., None]:
    """Adapt the workflow's text logger to service-style logging callbacks.

    Benchmark services may attach standard logging metadata such as ``level``.
    The workflow logger deliberately remains text-only, so metadata is accepted
    at this boundary and left for richer service loggers to interpret.
    """

    def _callback(message: Any, *_args: Any, **_kwargs: Any) -> None:
        log(str(message))

    return _callback


def _make_hailo_feasibility_evidence_lookup_v2783(
    control: Mapping[str, Any],
    *,
    suite_root: Path,
    log: Optional[Callable[[str], None]] = None,
) -> Optional[Callable[[Mapping[str, Any]], Mapping[str, Any]]]:
    """Bind the optional exact ledger to the Gate-A controller.

    The old EvaluationRun remains read-only.  Positive evidence is admitted
    only after the evidence module copies and re-verifies the HEF, build
    receipt, and compiler ONNX into the current case directory.
    """

    index_path = str(control.get("evidence_index_path") or "").strip()
    artifact_root = str(control.get("evidence_artifact_root") or "").strip()
    if not index_path and not artifact_root:
        return None
    if not index_path or not artifact_root:
        raise ValueError(
            "Hailo feasibility evidence index and artifact root must be bound together"
        )
    from ..build_evidence import (
        ARTIFACT_PASS,
        _normalize_hw_arch,
        _open_directory_nofollow,
        _path_inside,
        boundary_endpoint_contract_sha256,
        canonical_build_key_from_hailo_v3_payload,
        load_build_evidence_index,
        lookup_build_evidence,
        materialize_verified_artifact,
    )

    # Load and self-verify once before any compiler decision.  Artifact bytes
    # remain verified lazily by each positive lookup/materialization.
    evidence_index = load_build_evidence_index(index_path)
    trusted_suite_root = Path(
        os.path.abspath(
            os.path.normpath(os.fspath(Path(suite_root).expanduser()))
        )
    )
    trusted_root_fd = _open_directory_nofollow(
        trusted_suite_root, label="feasibility.suite_root"
    )
    os.close(trusted_root_fd)

    def _lookup(request: Mapping[str, Any]) -> Mapping[str, Any]:
        if not isinstance(request, Mapping):
            return {
                "exact": False,
                "outcome": "MISS",
                "reason": "invalid_lookup_request",
            }
        if str(request.get("required_variant") or "") != "part1":
            return {
                "exact": True,
                "outcome": "EVIDENCE_CONFLICT",
                "reason": "required_variant_mismatch",
            }
        request_boundary = request.get("boundary")
        if type(request_boundary) is not int or int(request_boundary) < 0:
            return {
                "exact": True,
                "outcome": "EVIDENCE_CONFLICT",
                "reason": "invalid_boundary",
            }
        cache_probe = (
            request.get("cache_probe")
            if isinstance(request.get("cache_probe"), Mapping)
            else {}
        )
        context = (
            request.get("evidence_context")
            if isinstance(request.get("evidence_context"), Mapping)
            else {}
        )
        cache_payload = cache_probe.get("cache_payload_v3")
        cache_key = str(cache_probe.get("cache_key_v3") or "").strip()
        if not isinstance(cache_payload, Mapping) or not cache_key:
            return {
                "exact": False,
                "outcome": "MISS",
                "reason": "exact_v3_cache_contract_unavailable",
            }
        builder_sha = str(
            context.get("builder_source_onnx_sha256") or ""
        ).strip()
        full_sha = str(context.get("full_source_onnx_sha256") or "").strip()
        split_manifest = (
            context.get("split_manifest")
            if isinstance(context.get("split_manifest"), Mapping)
            else {}
        )
        if (
            type(split_manifest.get("boundary")) is not int
            or int(split_manifest.get("boundary")) != int(request_boundary)
        ):
            return {
                "exact": True,
                "outcome": "EVIDENCE_CONFLICT",
                "reason": "split_manifest_boundary_mismatch",
            }
        endpoint_sha = boundary_endpoint_contract_sha256(
            stage="part1",
            cache_payload=cache_payload,
            split_manifest=split_manifest,
        )
        exact_key = canonical_build_key_from_hailo_v3_payload(
            cache_payload,
            builder_source_onnx_sha256=builder_sha,
            full_source_onnx_sha256=full_sha,
            boundary_endpoint_contract_sha256=endpoint_sha,
            expected_cache_key=cache_key,
        )
        target = str(request.get("target") or "").strip()
        normalized_target = _normalize_hw_arch(target)
        key_target = _normalize_hw_arch(exact_key.get("hw_arch"))
        if key_target != normalized_target:
            return {
                "exact": True,
                "outcome": "EVIDENCE_CONFLICT",
                "reason": "canonical_key_target_mismatch",
            }
        decision = lookup_build_evidence(
            evidence_index,
            exact_key,
            artifact_root=artifact_root,
        )
        projected = decision.as_dict()
        if str(decision.status or "").upper() == "CONFLICT":
            return {
                **projected,
                "exact": True,
                "outcome": "EVIDENCE_CONFLICT",
                "origin": dict(decision.evidence_origin or {}),
            }
        if decision.status != "HIT" or not bool(decision.reusable):
            return {
                **projected,
                "exact": False,
                "outcome": "MISS",
                "origin": dict(decision.evidence_origin or {}),
            }
        if decision.state != ARTIFACT_PASS:
            return {
                **projected,
                "exact": True,
                "outcome": str(decision.state or ""),
                "origin": dict(decision.evidence_origin or {}),
            }

        materialization = (
            request.get("materialization")
            if isinstance(request.get("materialization"), Mapping)
            else {}
        )
        raw_case_dir = str(materialization.get("case_dir") or "").strip()
        raw_case_path = Path(raw_case_dir).expanduser()
        if (
            not raw_case_dir
            or not raw_case_path.is_absolute()
            or any(part == ".." for part in raw_case_path.parts)
        ):
            return {
                **projected,
                "exact": True,
                "outcome": "EVIDENCE_CONFLICT",
                "reason": "unsafe_materialization_case_dir",
            }
        case_dir = Path(os.path.normpath(os.fspath(raw_case_path)))
        if (
            case_dir.parent != trusted_suite_root
            or not _path_inside(case_dir, trusted_suite_root)
        ):
            return {
                **projected,
                "exact": True,
                "outcome": "EVIDENCE_CONFLICT",
                "reason": "materialization_case_outside_active_suite",
            }
        folder_match = re.fullmatch(r"b0*(\d+)", case_dir.name)
        if (
            folder_match is None
            or int(folder_match.group(1)) != int(request_boundary)
        ):
            return {
                **projected,
                "exact": True,
                "outcome": "EVIDENCE_CONFLICT",
                "reason": "materialization_boundary_mismatch",
            }
        case_descriptor = _open_directory_nofollow(
            case_dir, label="feasibility.case_dir"
        )
        os.close(case_descriptor)
        destination = case_dir / "hailo" / target / "part1"
        if not _path_inside(destination, case_dir):
            raise ValueError("evidence materialization escaped case_dir")
        materialized = materialize_verified_artifact(
            decision,
            destination,
            exact_key,
            artifact_root=artifact_root,
        )
        materialized_target = _normalize_hw_arch(
            materialized.get("hw_arch")
        )
        if (
            materialized_target != normalized_target
            or materialized_target != key_target
        ):
            raise ValueError("materialized evidence target mismatch")
        hef_path = Path(str(materialized.get("hef_path") or "")).expanduser()
        if (
            not hef_path.is_absolute()
            or not _path_inside(hef_path, destination)
            or not hef_path.is_file()
            or hef_path.is_symlink()
        ):
            raise ValueError("unsafe materialized evidence HEF path")
        target_outcome = {
            "hw_arch": target,
            "target_output": {
                "part1": os.path.relpath(hef_path, case_dir).replace("\\", "/"),
                "part1_build": {
                    "ok": True,
                    "skipped": False,
                    "timed_out": False,
                    "cache_hit": True,
                    "cache_source": "exact_build_evidence",
                    "cache_key": str(materialized.get("cache_key") or cache_key),
                    "failure_kind": None,
                    "error": None,
                    "artifact_hash": str(
                        (
                            (materialized.get("details") or {}).get(
                                "exact_build_evidence"
                            )
                            or {}
                        ).get("artifact_sha256")
                        or ""
                    ),
                    "details": dict(materialized.get("details") or {}),
                },
            },
            "errors": [],
            "failure_records": [],
            "first_rejection": None,
            "row_per_cut_hints": [],
            "diagnostics": [],
            "full_metadata": {},
        }
        if callable(log):
            log(
                f"[build-evidence] materialized exact {target} Part1 "
                f"artifact for b{int(request.get('boundary') or 0)}"
            )
        return {
            **projected,
            "exact": True,
            "outcome": ARTIFACT_PASS,
            "origin": dict(decision.evidence_origin or {}),
            "target_outcome": target_outcome,
            "materialized": materialized,
        }

    return _lookup


def _read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _read_gate_resume_state_v2783(path: Path) -> Mapping[str, Any]:
    """Strict no-follow loader for Gate-A control-plane state."""

    from ..build_evidence import _read_regular_nofollow, _strict_json

    observed = _read_regular_nofollow(
        path,
        label="hailo_feasibility_resume_state",
        collect=True,
        size_limit=64 * 1024 * 1024,
    )
    payload = _strict_json(
        observed.data or b"", label="hailo_feasibility_resume_state"
    )
    if not isinstance(payload, Mapping):
        raise ValueError("Hailo feasibility resume state must be a JSON object")
    if not isinstance(payload.get("hailo_feasibility_state"), Mapping):
        raise ValueError(
            "Hailo feasibility resume state is missing hailo_feasibility_state"
        )
    return dict(payload)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return int(default)
        return int(float(str(value).strip()))
    except Exception:
        return int(default)


def _runner_int_or_none(value: Any) -> Optional[int]:
    """Mirror ``WorkflowRunner._int_or_none`` without float-string coercion."""

    try:
        return int(value)
    except Exception:
        return None


def _mode_task_item_count(profile_payload: Mapping[str, Any], task: str, *, kind: str, fallback: int) -> int:
    """Resolve task-specific run-mode budgets without exposing them per run."""
    preset = profile_payload.get("execution_preset") if isinstance(profile_payload, Mapping) else {}
    effective = preset.get("effective") if isinstance(preset, Mapping) and isinstance(preset.get("effective"), Mapping) else {}
    values = effective.get(kind) if isinstance(effective, Mapping) and isinstance(effective.get(kind), Mapping) else {}
    task_key = "detection" if str(task or "").lower() == "detection" else "classification"
    try:
        value = int(values.get(task_key))
        if kind == "calibration_items":
            return max(1, value)
        return max(0, value)
    except Exception:
        return int(fallback)


def _safe_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in {"1", "true", "yes", "y", "on", "enabled", "enable"}:
        return True
    if s in {"0", "false", "no", "n", "off", "disabled", "disable"}:
        return False
    return bool(default)


def _selection_policy_from_profile(profile_payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Mirror the runner's resolved selection-policy aliases."""

    if isinstance(profile_payload.get("selection_policy"), Mapping):
        return dict(profile_payload.get("selection_policy") or {})
    if isinstance(profile_payload.get("benchmark"), Mapping):
        return dict(profile_payload.get("benchmark") or {})
    if isinstance(profile_payload.get("benchmark_candidates"), Mapping):
        raw = dict(profile_payload.get("benchmark_candidates") or {})
        aliases = {
            "accepted_cases": "max_accepted_cases_per_model",
            "candidate_shortlist": "preferred_shortlist",
            "min_gap": "min_gap",
            "candidate_search_pool": "candidate_search_pool",
            "selection_strategy": "selection_strategy",
        }
        resolved = {
            destination: raw.get(source)
            for source, destination in aliases.items()
            if source in raw
        }
        return resolved or raw
    return {}


def _selection_strategy_from_profile(
    profile_payload: Mapping[str, Any],
    selection_policy: Mapping[str, Any],
) -> str:
    """Mirror the runner's analysis override and selection-policy fallback."""

    analysis = (
        profile_payload.get("analysis")
        if isinstance(profile_payload, Mapping)
        else None
    )
    if isinstance(analysis, Mapping) and analysis.get("selection_strategy"):
        return str(analysis.get("selection_strategy"))
    return str(
        selection_policy.get("selection_strategy")
        or selection_policy.get("coverage_strategy")
        or "stratified_windows"
    )


def _normalise_selection_strategy(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_")


def _first_policy_int(
    policy: Mapping[str, Any],
    *keys: str,
    default: int,
) -> int:
    """Mirror ``WorkflowRunner._int_value`` for resolved policy authority."""

    for key in keys:
        value = policy.get(key)
        try:
            if value is not None and str(value) != "":
                return int(value)
        except Exception:
            continue
    return int(default)


def _native_split_requires_single_part2_input(
    native_config: Mapping[str, Any] | None,
) -> bool:
    """Mirror the runner's current Native split-boundary capability."""

    cfg = native_config if isinstance(native_config, Mapping) else {}
    if "split_backends" in cfg:
        raw = cfg.get("split_backends")
        if isinstance(raw, str):
            return bool(raw.strip())
        if isinstance(raw, Sequence) and not isinstance(
            raw, (bytes, bytearray)
        ):
            return any(str(value or "").strip() for value in raw)
        return False
    return bool(
        cfg.get("enabled")
        or cfg.get("run")
        or cfg.get("enabled_in_evalrun")
    )


def _native_capability_config(
    profile_payload: Mapping[str, Any],
    options: Any,
) -> Dict[str, Any]:
    """Resolve the subset of Native config that governed runner selection."""

    resolved: Dict[str, Any] = {}
    current = profile_payload.get("native_producers")
    if isinstance(current, Mapping):
        resolved.update(dict(current))
    legacy = profile_payload.get("native_fifo")
    if isinstance(legacy, Mapping):
        resolved.update({
            key: value
            for key, value in dict(legacy).items()
            if key not in resolved
        })
    if bool(getattr(options, "native_producer_enabled", False)):
        resolved["enabled"] = True
    explicit_backends = (
        list(getattr(options, "native_producer_backends") or [])
        if getattr(options, "native_producer_backends", None)
        else None
    )
    if explicit_backends is not None:
        resolved["backends"] = list(explicit_backends)
    split_profile = dict(profile_payload)
    split_profile["native_producers"] = dict(resolved)
    split_plan = resolve_native_split_plan(
        split_profile,
        explicit_backends=explicit_backends,
    )
    # Presence, including an empty list, is authoritative.  Full-only logical
    # matrices must not inherit the configured adapter inventory as an implicit
    # Split request.
    resolved["split_backends"] = list(split_plan.selected_split_backends)
    resolved["split_selection_source"] = split_plan.source
    return resolved


def _native_full_requested(native_cfg: Mapping[str, Any]) -> bool:
    full = (
        native_cfg.get("full_baselines")
        if isinstance(native_cfg.get("full_baselines"), Mapping)
        else {}
    )
    return bool(
        _safe_bool(native_cfg.get("enabled"), False)
        and _safe_bool(full.get("enabled"), False)
    )

def _profile_activation_proxy_strict(profile_payload: Mapping[str, Any]) -> bool:
    """Return True only when an evaluation profile explicitly requests strict proxy.

    GUI Tool-Config strict mode is useful for manual debugging, but for broad
    evaluation campaigns it can accidentally reject every split if a local CUDA
    proxy falls back to CPU.  Evaluation profiles therefore opt into strict mode
    explicitly via one of these keys.
    """
    if not isinstance(profile_payload, Mapping):
        return False
    candidates = []
    for key in ("activation_proxy_strict", "strict_activation_proxy"):
        if key in profile_payload:
            candidates.append(profile_payload.get(key))
    ap = profile_payload.get("activation_proxy")
    if isinstance(ap, Mapping):
        for key in ("strict", "strict_proxy", "fail_on_fallback"):
            if key in ap:
                candidates.append(ap.get(key))
    deepx = profile_payload.get("deepx_build")
    if isinstance(deepx, Mapping):
        for key in ("activation_proxy_strict", "strict_activation_proxy", "fail_on_proxy_fallback"):
            if key in deepx:
                candidates.append(deepx.get(key))
    for value in candidates:
        if _safe_bool(value, False):
            return True
    return False


def _effective_str(value: Any) -> str:
    """Return a non-empty override string, treating auto/none/null as unset."""
    s = str(value or "").strip()
    return "" if s.lower() in {"", "auto", "none", "null", "default"} else s


def _task_gated_flags(task: str, *, mini_coco: bool = False, mini_cls: bool = False) -> tuple[bool, bool]:
    task_l = str(task or "auto").strip().lower()
    if task_l == "classification":
        return False, bool(mini_cls)
    if task_l == "detection":
        return bool(mini_coco), False
    return False, False


def _normalize_target_token(value: Any) -> str:
    s = str(value or "").strip().lower().replace("-", "_")
    aliases = {
        "cpu": "cpu_ort",
        "ort_cpu": "cpu_ort",
        "cuda": "cuda_ort",
        "ort_cuda": "cuda_ort",
        "trt": "tensorrt",
        "tensor_rt": "tensorrt",
        "ort_tensorrt": "tensorrt",
        "hailo8_to_trt": "hailo8_to_tensorrt",
        "trt_to_hailo8": "tensorrt_to_hailo8",
        "dx_m1": "deepx_m1",
        "deepx": "deepx_m1",
        "deepx_dx_m1": "deepx_m1",
        "deepx_m1_to_trt": "deepx_m1_to_tensorrt",
        "trt_to_deepx_m1": "tensorrt_to_deepx_m1",
        "tensorrt_to_dx_m1": "tensorrt_to_deepx_m1",
    }
    return aliases.get(s, s)


def _profile_run_profiles(profile_payload: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    rps = profile_payload.get("run_profiles") if isinstance(profile_payload, Mapping) else None
    return list(enabled_run_profiles(rps))


def _profile_targets(profile_payload: Mapping[str, Any], fallback_targets: Sequence[str]) -> List[str]:
    out: List[str] = []
    raw_targets = profile_payload.get("targets") if isinstance(profile_payload, Mapping) else None
    if isinstance(raw_targets, list):
        for raw in raw_targets:
            t = _normalize_target_token(raw)
            if t and t not in out:
                out.append(t)
    for rp in _profile_run_profiles(profile_payload):
        rid = _normalize_target_token(rp.get("id") or rp.get("name") or "")
        st1 = _normalize_target_token(rp.get("stage1") or "")
        st2 = _normalize_target_token(rp.get("stage2") or "")
        full = _normalize_target_token(rp.get("full") or rp.get("full_reference") or "")
        for token in (rid, st1, st2, full):
            if token and token not in out:
                out.append(token)
    for raw in list(fallback_targets or []):
        t = _normalize_target_token(raw)
        if t and t not in out:
            out.append(t)
    return out or ["cpu_ort", "cuda_ort", "tensorrt", "hailo8", "hailo8_to_tensorrt"]


def _bind_management_reference_targets_v27519(
    profile_payload: Mapping[str, Any],
    targets: Sequence[str],
    *,
    cache_verify_enabled: bool,
) -> tuple[list[str], bool, bool]:
    """Classify the internal CPU recipe without changing logical targets."""

    bound = list(targets)
    explicit = profile_has_explicit_cpu_reference(profile_payload, bound)
    required = bool(
        management_cpu_reference_required(profile_payload)
        and not cache_verify_enabled
    )
    return bound, explicit, required


def _bind_management_reference_cpu_switch_v27519(
    switches: Mapping[str, Any],
    *,
    required: bool,
    cache_verify_enabled: bool,
) -> Dict[str, bool]:
    """Bind the internal recipe at the generator boundary, not target UX."""

    bound = {str(key): bool(value) for key, value in dict(switches).items()}
    if cache_verify_enabled:
        bound["acc_cpu"] = False
    elif required:
        bound["acc_cpu"] = True
    return bound




def _profile_requests_deepx(profile_payload: Mapping[str, Any]) -> tuple[bool, bool, bool]:
    full = False
    to_trt = False
    to_deepx = False
    for rp in _profile_run_profiles(profile_payload):
        rid = _normalize_target_token(rp.get("id") or "")
        st1 = _normalize_target_token(rp.get("stage1") or "")
        st2 = _normalize_target_token(rp.get("stage2") or "")
        full_tok = _normalize_target_token(rp.get("full") or "")
        if rid == "deepx_m1" or rid == "deepx_m1_full" or full_tok == "deepx_m1":
            full = True
        if rid == "deepx_m1_to_tensorrt" or (st1 == "deepx_m1" and st2 == "tensorrt"):
            to_trt = True
        if rid == "tensorrt_to_deepx_m1" or (st1 == "tensorrt" and st2 == "deepx_m1"):
            to_deepx = True
    targets = profile_payload.get("targets") if isinstance(profile_payload, Mapping) else None
    if isinstance(targets, list):
        toks = {_normalize_target_token(x) for x in targets}
        full = full or "deepx_m1" in toks
        to_trt = to_trt or "deepx_m1_to_tensorrt" in toks
        to_deepx = to_deepx or "tensorrt_to_deepx_m1" in toks
    return full, to_trt, to_deepx


def _profile_build_scheduler_config(profile_payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the resolved scheduler mapping from either supported profile shape."""
    scheduler = (
        profile_payload.get("build_scheduler")
        if isinstance(profile_payload.get("build_scheduler"), Mapping)
        else None
    )
    if isinstance(scheduler, Mapping):
        return dict(scheduler)
    build_cfg = (
        profile_payload.get("build")
        if isinstance(profile_payload.get("build"), Mapping)
        else {}
    )
    nested = build_cfg.get("scheduler") if isinstance(build_cfg, Mapping) else None
    return dict(nested) if isinstance(nested, Mapping) else {}


def _hailo_full_hef_policy_v2772(build_full: bool) -> str:
    """Project the explicit Full-build switch into suite orchestration."""
    return "end" if bool(build_full) else "skip"


def _profile_requests_hailo_pair(profile_payload: Mapping[str, Any], targets: Sequence[str]) -> bool:
    tokens = [_normalize_target_token(value) for value in list(targets or [])]
    tokens.extend(
        _normalize_target_token(row.get("id") or "")
        for row in _profile_run_profiles(profile_payload)
    )
    return any("hailo8" in token for token in tokens) and any(
        "hailo10" in token for token in tokens
    )


def _v60s_start_deepx_prefetch(
    *,
    run_dir: Path,
    model_id: str,
    model_path: str,
    model_row: Mapping[str, Any],
    task: str,
    profile_payload: Mapping[str, Any],
    targets: Sequence[str],
    suite_dir: Path,
    log: Callable[[str], None],
    hailo_backend: str = "",
    process_registry: Any = None,
    cancel_event: Any = None,
    _allow_hailo_pair_defer: bool = True,
) -> Optional[dict[str, Any]]:
    """Start the independent DeepX-full build while Hailo generation runs.

    The prefetch is deliberately best-effort.  The regular backend-artifact
    stage remains authoritative and can reuse the exact artifact/cache entry
    produced here.  The handle must be joined before the suite is packaged.
    """
    want_full, _want_to_trt, _want_to_deepx = _profile_requests_deepx(profile_payload)
    scheduler_cfg = _profile_build_scheduler_config(profile_payload)
    if not want_full or not bool(scheduler_cfg.get("enabled", True)) or not bool(scheduler_cfg.get("prefetch_deepx_full", True)):
        return None
    hailo_pair_decision: Dict[str, Any] = {}
    if _allow_hailo_pair_defer and _profile_requests_hailo_pair(profile_payload, targets):
        # Use the exact same resolved resource/backend gate as the Hailo build
        # controller.  Merely requesting both targets must not postpone DeepX
        # when the pair will actually execute serially.
        from ..benchmark.services import (
            _hailo_pair_parallel_decision_v27550,
            _physical_hailo_targets_for_build,
            _v60s_build_scheduler_config,
        )

        preset_cfg = (
            profile_payload.get("execution_preset")
            if isinstance(profile_payload.get("execution_preset"), Mapping)
            else {}
        )
        resolved_scheduler = _v60s_build_scheduler_config(
            scheduler_cfg,
            mode=str(preset_cfg.get("id") or "standard"),
        )
        hailo_pair_decision = _hailo_pair_parallel_decision_v27550(
            _physical_hailo_targets_for_build(targets),
            resolved_scheduler,
            backend=hailo_backend,
        )
    if bool(hailo_pair_decision.get("effective")):
        # The historic prefetch owns a separate resource pool.  Until Hailo and
        # DeepX share one scheduler, running it beside the two-compiler Hailo
        # pair can overcommit the controller.  Defer it until Hailo generation
        # has joined; the regular artifact stage still reuses the result.
        log(
            "[build-scheduler] DeepX full prefetch deferred until after the "
            "requested Hailo-8/Hailo-10 build pair"
        )
        return {
            "deferred": True,
            "request": {
                "run_dir": run_dir,
                "model_id": model_id,
                "model_path": model_path,
                "model_row": dict(model_row or {}),
                "task": task,
                "profile_payload": profile_payload,
                "targets": list(targets or []),
                "suite_dir": suite_dir,
                "hailo_backend": hailo_backend,
            },
        }
    if bool(hailo_pair_decision.get("requested")):
        log(
            "[build-scheduler] DeepX full prefetch remains active because "
            "the Hailo pair is serial: "
            f"{hailo_pair_decision.get('reason') or 'parallel_gate_not_met'}"
        )
    try:
        from ..build_scheduler import BuildScheduler, BuildTaskSpec, scheduler_config_from_mapping
        from .deepx_build_binding import materialize_deepx_build_binding

        resolved = scheduler_config_from_mapping(scheduler_cfg, mode=str((profile_payload.get("execution_preset") or {}).get("id") or "standard"))
        weights = dict((resolved.get("weights") or {}).get("deepx") or {})
        scheduler = BuildScheduler(
            max_workers=1,
            cpu_tokens=int(resolved.get("cpu_tokens") or 0) or None,
            ram_mb=int(resolved.get("ram_mb") or 0),
            family_limits={"deepx": int((resolved.get("family_limits") or {}).get("deepx") or 1)},
            log=log,
        )
        spec = BuildTaskSpec(
            name=f"deepx-full-prefetch:{model_id}",
            family="deepx",
            cpu_tokens=max(1, int(weights.get("cpu_tokens") or 2)),
            ram_mb=max(0, int(weights.get("ram_mb") or 4096)),
            metadata={"model_id": model_id, "task": task, "target": "deepx_m1"},
        )
        row = dict(model_row or {})
        row.setdefault("id", model_id)
        row.setdefault("task", task)
        future = scheduler.submit(
            spec,
            materialize_deepx_build_binding,
            run_dir=run_dir,
            model_id=model_id,
            model_path=model_path,
            row=row,
            profile_payload=profile_payload,
            targets=list(targets or ["deepx_m1"]),
            benchmark_set_contract={"legacy_suite_dir": str(suite_dir)},
            log=log,
            process_registry=process_registry,
            cancel_event=cancel_event,
        )
        log(f"[build-scheduler] DeepX full prefetch submitted for {model_id}")
        return {"scheduler": scheduler, "future": future, "started": True}
    except Exception as exc:
        log(f"[build-scheduler] DeepX full prefetch unavailable: {type(exc).__name__}: {exc}")
        return None


def _v60s_finish_deepx_prefetch(
    handle: Optional[dict[str, Any]],
    *,
    suite_dir: Path,
    log: Callable[[str], None],
    process_registry: Any = None,
    cancel_event: Any = None,
) -> dict[str, Any]:
    if not handle:
        return {"enabled": False, "status": "not_requested"}
    if bool(handle.get("deferred")):
        request = dict(handle.get("request") or {})
        cancelled = bool(
            (cancel_event is not None and cancel_event.is_set())
            or (
                process_registry is not None
                and getattr(process_registry, "cancelled", False)
            )
        )
        if cancelled:
            report = {
                "enabled": True,
                "status": "cancelled_before_deferred_start",
                "cancelled": True,
                "deferred_for_hailo_pair": True,
            }
            log(
                "[build-scheduler] deferred DeepX full prefetch not started "
                "because cancellation is already active"
            )
            try:
                write_json(suite_dir / "deepx_prefetch_v60s.json", report)
            except Exception:
                pass
            return report
        log("[build-scheduler] starting deferred DeepX full prefetch after Hailo pair join")
        started = _v60s_start_deepx_prefetch(
            **request,
            log=log,
            process_registry=process_registry,
            cancel_event=cancel_event,
            _allow_hailo_pair_defer=False,
        )
        report = _v60s_finish_deepx_prefetch(
            started,
            suite_dir=suite_dir,
            log=log,
            process_registry=process_registry,
            cancel_event=cancel_event,
        )
        report["deferred_for_hailo_pair"] = True
        return report
    scheduler = handle.get("scheduler")
    future = handle.get("future")
    report: dict[str, Any]
    cancellation_seen = False
    try:
        result = None
        cancel_deadline: float | None = None
        while future is not None and not future.done():
            cancelled = bool(
                (cancel_event is not None and cancel_event.is_set())
                or (
                    process_registry is not None
                    and getattr(process_registry, "cancelled", False)
                )
            )
            if cancelled:
                cancellation_seen = True
                future.cancel()
                if cancel_deadline is None:
                    cancel_deadline = (
                        time.monotonic() + _DEEPX_PREFETCH_CANCEL_GRACE_S
                    )
                if time.monotonic() >= cancel_deadline:
                    break
            try:
                result = future.result(timeout=0.1)
            except concurrent.futures.TimeoutError:
                continue
        if future is not None and result is None and future.done():
            result = future.result(timeout=0.0)
        if future is not None and not future.done():
            uncertainty_registry = process_registry or current_process_registry()
            uncertainty_recorder = getattr(
                uncertainty_registry,
                "record_cleanup_uncertainty",
                None,
            )
            if callable(uncertainty_recorder):
                uncertainty_recorder(
                    label="deepx-prefetch-worker",
                    detail=(
                        "DeepX prefetch worker remained after bounded "
                        "cancellation grace"
                    ),
                )
            report = {
                "enabled": True,
                "status": "cancelled_cleanup_unresolved",
                "cancelled": True,
                "cancel_grace_s": _DEEPX_PREFETCH_CANCEL_GRACE_S,
            }
            log(
                "[build-scheduler] DeepX full prefetch did not exit within "
                "the bounded cancellation grace period"
            )
        else:
            report = {"enabled": True, "status": str((result or {}).get("status") or "ok"), "result": result}
            log(f"[build-scheduler] DeepX full prefetch finished status={report['status']}")
    except concurrent.futures.CancelledError:
        report = {"enabled": True, "status": "cancelled"}
        log("[build-scheduler] DeepX full prefetch cancelled")
    except Exception as exc:
        report = {"enabled": True, "status": "failed_non_blocking", "error": f"{type(exc).__name__}: {exc}"}
        log(f"[build-scheduler] DeepX full prefetch failed non-blockingly: {type(exc).__name__}: {exc}")
    finally:
        try:
            if scheduler is not None:
                # A running compiler observes the shared cancel event and the
                # sticky ProcessTreeRegistry.  Do not let executor shutdown
                # reintroduce an unbounded wait after that bounded hand-off.
                scheduler.shutdown(not cancellation_seen)
        except Exception:
            pass
    try:
        write_json(suite_dir / "deepx_prefetch_v60s.json", report)
    except Exception:
        pass
    return report


def _append_deepx_runs_to_plan(plan_payload: Mapping[str, Any], profile_payload: Mapping[str, Any]) -> dict[str, Any]:
    plan = dict(plan_payload or {})
    runs = [dict(r) for r in list(plan.get("runs") or plan.get("planned_runs") or []) if isinstance(r, Mapping)]
    existing = {str(r.get("id") or r.get("name") or "") for r in runs}
    want_full, want_to_trt, want_to_deepx = _profile_requests_deepx(profile_payload)
    if want_full and "deepx_m1_full" not in existing:
        runs.append({
            "id": "deepx_m1_full",
            "type": "deepx",
            "backend": "deepx_m1",
            "variant": "full",
            "artifact_kind": "dxnn",
            "dxnn_path": "deepx/deepx_m1/full/model.dxnn",
            "contract_path": "deepx/deepx_m1/full/output_contract.json",
            "required": False,
        })
    if want_to_trt and "deepx_m1_to_tensorrt" not in existing:
        runs.append({
            "id": "deepx_m1_to_tensorrt",
            "type": "matrix",
            "backend": "deepx_m1_to_tensorrt",
            "provider": "tensorrt",
            "variants": ["part1", "part2", "composed"],
            "stage1": {"type": "deepx", "target": "deepx_m1", "backend": "deepx_m1", "artifact_kind": "dxnn"},
            "stage2": {"type": "onnxruntime", "provider": "tensorrt"},
            "required": False,
        })
    if want_to_deepx and "tensorrt_to_deepx_m1" not in existing:
        runs.append({
            "id": "tensorrt_to_deepx_m1",
            "type": "matrix",
            "backend": "tensorrt_to_deepx_m1",
            "provider": "tensorrt",
            "variants": ["part1", "part2", "composed"],
            "stage1": {"type": "onnxruntime", "provider": "tensorrt"},
            "stage2": {"type": "deepx", "target": "deepx_m1", "backend": "deepx_m1", "artifact_kind": "dxnn"},
            "activation_calibration_source": "activation_proxy_cache",
            "experimental": True,
            "required": False,
        })
    plan["runs"] = runs
    plan["planned_runs"] = runs
    return plan


_DEFERRED_HAILO_FULL_FAILURE_KINDS = {
    "deferred_cold_full_cache_miss",
    "deferred_cold_build",
    "cache_only_miss",
}


def _canonical_hailo_target(value: Any) -> str:
    token = str(value or "").strip().lower().replace("-", "").replace("_", "")
    if "hailo10" in token or token in {"h10", "hailo10h", "hailo10p"}:
        return "hailo10"
    if "hailo8" in token or token in {"h8", "hailo"}:
        return "hailo8"
    return ""


def _discover_deferred_hailo_full_builds(suite_dir: Path) -> List[Dict[str, Any]]:
    """Return intentional Smoke cold-build deferrals for Hailo Full baselines.

    The legacy generator writes a structured ``hailo_hef_build_result.json`` and,
    for cache-only Smoke misses, a ``cold_build_request.json``.  Those files are
    the authoritative source for deciding that the corresponding *Full* run must
    not be dispatched or counted as a missing required matrix row.
    """
    out: List[Dict[str, Any]] = []
    hailo_root = Path(suite_dir) / "hailo"
    if not hailo_root.is_dir():
        return out
    for arch_dir in sorted(p for p in hailo_root.iterdir() if p.is_dir()):
        target = _canonical_hailo_target(arch_dir.name)
        if not target:
            continue
        full_dir = arch_dir / "full"
        result_path = full_dir / "hailo_hef_build_result.json"
        request_path = full_dir / "cold_build_request.json"
        payload = _read_json(result_path, default={}) or {}
        failure_kind = str(payload.get("failure_kind") or "").strip().lower()
        unsupported_reason = str(payload.get("unsupported_reason") or "").strip().lower()
        deferred = (
            failure_kind in _DEFERRED_HAILO_FULL_FAILURE_KINDS
            or (request_path.is_file() and unsupported_reason in {"cache_only_policy", "cache_only", "smoke_cache_only"})
        )
        if not deferred:
            continue
        calib = payload.get("calib_info") if isinstance(payload.get("calib_info"), Mapping) else {}
        details = payload.get("details") if isinstance(payload.get("details"), Mapping) else {}
        out.append({
            "target": target,
            "status": "deferred_cold_build",
            "reason": failure_kind or "deferred_cold_full_cache_miss",
            "unsupported_reason": unsupported_reason or "cache_only_policy",
            "result_path": str(result_path),
            "request_path": str(request_path) if request_path.is_file() else "",
            "cache_key": str((calib or {}).get("cache_key") or (details or {}).get("cache_key") or ""),
            "message": str(payload.get("error") or "Smoke cold-build policy deferred a Hailo Full cache miss."),
        })
    return out


def _is_matching_hailo_full_run(row: Mapping[str, Any], target: str) -> bool:
    """Identify the standalone Hailo Full run, never a split/matrix run."""
    rid = str(row.get("id") or row.get("name") or "").strip().lower()
    rtype = str(row.get("type") or row.get("kind") or "").strip().lower()
    backend = str(row.get("backend") or row.get("provider") or row.get("target") or "").strip().lower()
    variants = [str(x).strip().lower() for x in list(row.get("variants") or [])]
    variant = str(row.get("variant") or "").strip().lower()
    text = " ".join((rid, rtype, backend, variant, " ".join(variants)))
    if "to_trt" in text or "to_tensorrt" in text or rtype in {"matrix", "pipeline", "split"}:
        return False
    if rtype not in {"hailo", "hailort", "accelerator"} and not (variant == "full" or variants == ["full"]):
        return False
    return _canonical_hailo_target(text) == _canonical_hailo_target(target)


def _apply_deferred_hailo_full_builds(
    plan_payload: Mapping[str, Any],
    deferred_builds: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Annotate intentionally deferred Full runs in both plan aliases.

    A deferred run remains in the plan for auditability, but dispatch and matrix
    completeness code can now unambiguously exclude it.  Split rows for the same
    accelerator are left untouched.
    """
    plan = dict(plan_payload or {})
    runs = [dict(r) for r in list(plan.get("runs") or plan.get("planned_runs") or []) if isinstance(r, Mapping)]
    annotations: List[Dict[str, Any]] = []
    for build in deferred_builds:
        target = _canonical_hailo_target(build.get("target"))
        if not target:
            continue
        matched = False
        for row in runs:
            if not _is_matching_hailo_full_run(row, target):
                continue
            row.update({
                "deferred": True,
                "required": False,
                "deferred_reason": str(build.get("reason") or "deferred_cold_build"),
                "deferred_by": "smoke_cache_or_defer",
                "deferred_artifact": "hailo_full_hef",
                "deferred_target": target,
                "deferred_request_path": str(build.get("request_path") or ""),
                "deferred_cache_key": str(build.get("cache_key") or ""),
            })
            matched = True
        annotations.append({**dict(build), "plan_run_matched": matched})
    plan["runs"] = runs
    plan["planned_runs"] = [dict(r) for r in runs]
    if annotations:
        plan["deferred_full_baselines"] = annotations
        plan["deferred_run_ids"] = [
            str(r.get("id") or r.get("name") or "")
            for r in runs
            if bool(r.get("deferred"))
        ]
    return plan


def _apply_declared_run_variants(
    runs: Sequence[Mapping[str, Any]], profile_payload: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    """Carry explicit profile scope into the real generated suite.

    The generator defaults come from accelerator switches. A frozen profile
    with an explicit Full-only reference must not grow generic split engines
    at that handoff. Unspecified variants retain the ordinary generator defaults.
    """
    from .required_run_scope import canonical_run_id
    declared = {
        canonical_run_id(row.get('id')): list(row['variants'])
        for row in _profile_run_profiles(profile_payload)
        if isinstance(row.get('variants'), list)
        and canonical_run_id(row.get('id')) == 'ort_tensorrt'
    }
    projected = []
    for row in runs:
        result = dict(row)
        variants = declared.get(canonical_run_id(row.get('id')))
        if variants is not None:
            result['variants'] = list(variants)
        projected.append(result)
    return projected


def _infer_run_switches(profile_payload: Mapping[str, Any], targets: Sequence[str]) -> Dict[str, bool]:
    toks = {_normalize_target_token(t) for t in list(targets or [])}
    rps = _profile_run_profiles(profile_payload)
    for rp in rps:
        rid = _normalize_target_token(rp.get("id") or "")
        st1 = _normalize_target_token(rp.get("stage1") or "")
        st2 = _normalize_target_token(rp.get("stage2") or "")
        full = _normalize_target_token(rp.get("full") or "")
        toks.update(t for t in (rid, st1, st2, full) if t)
    text = " ".join(sorted(toks))
    matrix_hailo_to_trt = (
        any(
            _normalize_target_token(rp.get("stage1")).startswith("hailo")
            and _normalize_target_token(rp.get("stage2")) == "tensorrt"
            for rp in rps
        )
        or any("hailo" in tok and ("_to_tensorrt" in tok or "_to_trt" in tok) for tok in toks)
    )
    matrix_trt_to_hailo = (
        any(
            _normalize_target_token(rp.get("stage1")) == "tensorrt"
            and _normalize_target_token(rp.get("stage2")).startswith("hailo")
            for rp in rps
        )
        or any((tok.startswith("tensorrt_to_hailo") or tok.startswith("trt_to_hailo")) for tok in toks)
    )
    matrix_deepx_to_trt = (
        any(
            _normalize_target_token(rp.get("stage1")) in {"deepx", "deepx_m1", "dx_m1", "dxm1"}
            and _normalize_target_token(rp.get("stage2")) == "tensorrt"
            for rp in rps
        )
        or any("deepx" in tok and ("_to_tensorrt" in tok or "_to_trt" in tok) for tok in toks)
    )
    matrix_trt_to_deepx = (
        any(
            _normalize_target_token(rp.get("stage1")) == "tensorrt"
            and _normalize_target_token(rp.get("stage2")) in {"deepx", "deepx_m1", "dx_m1", "dxm1"}
            for rp in rps
        )
        or any((tok.startswith("tensorrt_to_deepx") or tok.startswith("trt_to_deepx")) for tok in toks)
    )
    hailo_full_requested = any(_normalize_target_token(rp.get("full")).startswith("hailo") for rp in rps)
    deepx_full_requested = any(_normalize_target_token(rp.get("full")) in {"deepx", "deepx_m1", "dx_m1", "dxm1"} for rp in rps)
    return {
        "acc_cpu": ("cpu_ort" in toks or "cpu" in text),
        "acc_cuda": ("cuda_ort" in toks or "cuda" in text),
        "acc_trt": ("tensorrt" in toks or "trt" in text),
        "acc_h8": ("hailo8" in text),
        "acc_h10": ("hailo10" in text),
        "acc_deepx": ("deepx" in text or "dx_m1" in text or "dxm1" in text),
        "hailo_full_requested": hailo_full_requested,
        "hailo_part1_requested": matrix_hailo_to_trt,
        "hailo_part2_requested": matrix_trt_to_hailo,
        "deepx_full_requested": deepx_full_requested,
        "deepx_part1_requested": matrix_deepx_to_trt,
        "deepx_part2_requested": matrix_trt_to_deepx,
        "matrix_hailo_to_trt": matrix_hailo_to_trt,
        "matrix_trt_to_hailo": matrix_trt_to_hailo,
        "matrix_deepx_to_trt": matrix_deepx_to_trt,
        "matrix_trt_to_deepx": matrix_trt_to_deepx,
    }


def _model_task(row: Mapping[str, Any], model_id: str) -> str:
    task = str(row.get("task") or "").strip().lower()
    if task in {"classification", "classify", "cls"}:
        return "classification"
    if task in {"detection", "detect", "det", "object_detection"}:
        return "detection"
    blob = " ".join(str(row.get(k) or "") for k in ("id", "family", "model_family", "resolved_path", "onnx")) + " " + model_id
    low = blob.lower()
    if any(tok in low for tok in ("yolo", "ssd", "detect", "detr")):
        return "detection"
    if any(tok in low for tok in ("resnet", "mobilenet", "regnet", "efficientnet", "vit", "swin", "convnext", "classifier", "classification", "imagenet", "imagenette")):
        return "classification"
    return "auto"


def _validation_defaults(profile_payload: Mapping[str, Any], row: Mapping[str, Any], model_id: str) -> Dict[str, Any]:
    validation = profile_payload.get("validation") if isinstance(profile_payload.get("validation"), Mapping) else {}
    task = _model_task(row, model_id)
    preset = str(row.get("validation_preset") or row.get("development_subset") or row.get("semantic_dataset") or "").strip()
    quality = profile_payload.get("quality_gate") if isinstance(profile_payload.get("quality_gate"), Mapping) else {}
    validation_tier = str(row.get("validation_tier") or (quality or {}).get("dataset_tier") or "screening").strip().lower()

    # v60o: the run mode, not a legacy per-model development_subset, decides
    # whether the registered ImageNet/COCO source is used.  Standard uses the
    # final registry with a bounded 500-item screening sample; Final uses its
    # configured budget (or the complete manifest when the budget is zero).
    preset_cfg = profile_payload.get("execution_preset") if isinstance(profile_payload.get("execution_preset"), Mapping) else {}
    snap = preset_cfg.get("snapshot") if isinstance(preset_cfg.get("snapshot"), Mapping) else {}
    snap_data = snap.get("data") if isinstance(snap.get("data"), Mapping) else {}
    use_registry = bool(snap_data.get("use_final_dataset_registry"))
    if not use_registry:
        campaign = profile_payload.get("campaign") if isinstance(profile_payload.get("campaign"), Mapping) else {}
        use_registry = bool(campaign.get("auto_bind_dataset_registry") and validation_manifest_for_task(profile_payload, task))

    manifest = validation_manifest_for_task(profile_payload, task) if use_registry else ""
    manifest_root = manifest_dataset_root(manifest) if manifest else ""
    manifest_count = 0
    if manifest:
        try:
            manifest_count = int((_read_json(Path(manifest).expanduser(), default={}) or {}).get("item_count") or 0)
        except Exception:
            manifest_count = 0

    mode_budget = _mode_task_item_count(
        profile_payload, task, kind="validation_items",
        fallback=(200 if task == "classification" else 50),
    )
    validation_execution = profile_payload.get("validation_execution") if isinstance(profile_payload.get("validation_execution"), Mapping) else {}
    validation_limits = validation_execution.get("max_items") if isinstance(validation_execution.get("max_items"), Mapping) else {}
    # The materialised run mode is authoritative even when its value is zero
    # (zero means the complete registered manifest in Final mode).
    budget_authoritative = bool(
        isinstance(profile_payload.get("execution_preset"), Mapping)
        and task in validation_limits
    )
    # 0 means complete manifest.  Otherwise the run-mode budget is authoritative
    # and old model-level values cannot silently force COCO-50/Imagenette-50.
    if manifest_root:
        validation_images = manifest_root
        max_images = manifest_count if mode_budget <= 0 and manifest_count > 0 else mode_budget
        source_kind = "content_addressed_manifest"
    else:
        validation_images = _effective_str(validation.get("validation_images") or validation.get("images")) or preset
        if not validation_images:
            validation_images = "imagenette_val_mini_200" if task == "classification" else ""
        legacy_default = 200 if task == "classification" else 50
        explicit_max = validation.get("validation_max_images") or validation.get("max_images") or row.get("validation_max_images")
        max_images = _safe_int(explicit_max or mode_budget or legacy_default, legacy_default)
        source_kind = "profile_or_screening_preset"

    detection_metrics = [str(x) for x in list(validation.get("detection_metrics") or [])]
    classification_metrics = [str(x) for x in list(validation.get("classification_metrics") or [])]
    req_mini_coco = bool("mini_coco_ap50" in detection_metrics or validation.get("mini_coco_ap50", False))
    req_mini_cls = bool(classification_metrics or validation.get("mini_classification_eval", False) or task == "classification")
    mini_coco, mini_cls = _task_gated_flags(task, mini_coco=req_mini_coco, mini_cls=req_mini_cls)
    return {
        "image_scale": str(validation.get("image_scale") or "auto"),
        "validation_images": validation_images,
        "validation_max_images": max(0, int(max_images or 0)),
        "validation_reference_mode": str(validation.get("split_fidelity_reference_mode") or validation.get("validation_reference_mode") or "auto"),
        "mini_coco_ap50": mini_coco,
        "benchmark_task": task,
        "mini_classification_eval": mini_cls,
        "validation_tier": validation_tier,
        "validation_manifest": manifest,
        "validation_source_kind": source_kind,
        "validation_items_requested": int(mode_budget),
        "validation_manifest_item_count": int(manifest_count),
        "validation_budget_authoritative": bool(budget_authoritative),
    }


def _boundary_from_candidate(candidate: Mapping[str, Any]) -> Optional[int]:
    for key in ("split_index", "boundary", "boundary_index"):
        if key in candidate:
            try:
                return int(float(str(candidate.get(key))))
            except Exception:
                continue
    return None


def _boundary_from_case_id(value: Any) -> Optional[int]:
    """Return the numeric boundary encoded by canonical ids such as ``b052``."""

    match = re.fullmatch(r"[bB]?0*(\d+)", str(value or "").strip())
    if match is None:
        return None
    return int(match.group(1))


def _candidate_boundaries(candidate_plan: Mapping[str, Any], prediction: Mapping[str, Any]) -> Tuple[List[int], List[int]]:
    ranked: List[int] = []
    pool: List[int] = []
    for cand in list(candidate_plan.get("selected_candidates") or []):
        if isinstance(cand, Mapping):
            b = _boundary_from_candidate(cand)
            if b is not None and b not in ranked:
                ranked.append(int(b))
            if b is not None and b not in pool:
                pool.append(int(b))
    for cand in list(prediction.get("candidates") or []):
        if isinstance(cand, Mapping):
            b = _boundary_from_candidate(cand)
            if b is not None and b not in pool:
                pool.append(int(b))
    if not ranked:
        ranked = list(pool[: max(1, min(5, len(pool)))])
    return ranked, pool or list(ranked)


def _cache_verify_candidate_scope(
    *,
    profile_payload: Mapping[str, Any],
    model_id: str,
    candidate_plan: Mapping[str, Any],
    ranked_candidates: Sequence[int],
    candidate_search_pool: Sequence[int],
    requested: int,
) -> tuple[Dict[str, Any], list[str], list[int], list[int], int]:
    """Constrain a guarded generator to its exact attested case set."""

    guard = cache_verify_guard(profile_payload)
    if not guard:
        return (
            {}, [], list(ranked_candidates), list(candidate_search_pool),
            int(requested),
        )
    expected_plan = (
        guard.get("expected_plan")
        if isinstance(guard.get("expected_plan"), Mapping)
        else {}
    )
    forced_case_map = (
        expected_plan.get("forced_case_map")
        if isinstance(expected_plan.get("forced_case_map"), Mapping)
        else {}
    )
    exact_cases = [
        str(value).strip()
        for value in list(forced_case_map.get(model_id) or [])
        if str(value or "").strip()
    ]
    exact_boundaries = [
        boundary
        for boundary in (
            _boundary_from_case_id(value) for value in exact_cases
        )
        if boundary is not None
    ]
    selected_boundaries = [
        boundary
        for boundary in (
            _boundary_from_candidate(candidate)
            for candidate in list(
                candidate_plan.get("selected_candidates") or []
            )
            if isinstance(candidate, Mapping)
        )
        if boundary is not None
    ]
    if (
        not exact_boundaries
        or len(exact_boundaries) != len(exact_cases)
        or len(selected_boundaries) != len(exact_boundaries)
        or sorted(set(selected_boundaries))
        != sorted(set(exact_boundaries))
    ):
        raise CacheVerifyPolicyError(
            "cache_verify_only exact-case handoff failed before generation: "
            f"expected={exact_cases!r} "
            f"selected_boundaries={sorted(set(selected_boundaries))!r}"
        )
    constrained = list(dict.fromkeys(exact_boundaries))
    return (
        dict(guard), exact_cases, constrained, list(constrained),
        len(constrained),
    )


def _analysis_prediction_metrics(analysis_payload: Optional[Mapping[str, Any]], boundary: int) -> Dict[str, Any]:
    if not isinstance(analysis_payload, Mapping):
        return {}
    out: Dict[str, Any] = {}

    def at(seq: Any, idx: int) -> Any:
        try:
            if isinstance(seq, Mapping):
                return seq.get(idx, seq.get(str(idx)))
            if isinstance(seq, Sequence) and not isinstance(seq, (str, bytes)) and 0 <= idx < len(seq):
                return seq[idx]
        except Exception:
            return None
        return None

    for src_key, dst_key in (
        ("costs_bytes", "cut_bytes"),
        ("crossing_counts_known", "crossing_tensors_known"),
        ("crossing_counts_all", "crossing_tensors_all"),
        ("unknown_crossing_counts", "unknown_crossing_tensors"),
        ("flops_left_prefix", "flops_left"),
        ("params_left_prefix", "params_left"),
    ):
        val = at(analysis_payload.get(src_key), int(boundary))
        if val not in (None, ""):
            try:
                out[dst_key] = float(val)
            except Exception:
                out[dst_key] = val
    try:
        if "cut_bytes" in out:
            out["cut_mib"] = float(out["cut_bytes"]) / (1024.0 ** 2)
    except Exception:
        pass
    return out


def _hailo_parse_entry_for_boundary(analysis_payload: Optional[Mapping[str, Any]], boundary: int) -> Optional[Dict[str, Any]]:
    if not isinstance(analysis_payload, Mapping):
        return None
    summary = analysis_payload.get("hailo_check") if isinstance(analysis_payload.get("hailo_check"), Mapping) else None
    results = analysis_payload.get("hailo_check_results") if isinstance(analysis_payload.get("hailo_check_results"), Mapping) else None
    entry = None
    if isinstance(results, Mapping):
        raw = results.get(int(boundary), results.get(str(boundary)))
        if isinstance(raw, Mapping):
            entry = dict(raw)
    if entry is None and summary is None:
        return None
    out: Dict[str, Any] = {}
    if isinstance(summary, Mapping):
        out.update(dict(summary))
    if isinstance(entry, Mapping):
        out.update(dict(entry))
    return out or None


def _hailo_parse_scalar_fields(entry: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if not isinstance(entry, Mapping):
        return {}
    out: Dict[str, Any] = {}
    for key in ("ok", "accepted_by", "policy", "target", "backend", "hw_arch", "error", "strategy"):
        if key in entry:
            out[key] = entry.get(key)
    return out


def _prepare_full_baseline_info(model_path: Path) -> Tuple[Dict[str, Any], List[str], str]:
    baseline: Dict[str, Any] = {}
    end_nodes: List[str] = []
    endpoint_mode = ""
    try:
        from ..benchmark.model_preparation import (
            find_latest_preparation_full_hailo_baseline,
            load_preparation_full_hailo_baseline,
            load_preparation_full_hailo_end_nodes,
        )
        try:
            baseline = dict(load_preparation_full_hailo_baseline(model_path) or {})
        except Exception:
            baseline = {}
        if not bool(baseline.get("ok")):
            try:
                latest = dict(find_latest_preparation_full_hailo_baseline(model_path) or {})
            except Exception:
                latest = {}
            if latest:
                baseline = latest
        try:
            end_info = dict(load_preparation_full_hailo_end_nodes(model_path) or {})
            end_nodes = [str(x).strip() for x in list(end_info.get("end_node_names") or []) if str(x).strip()]
            endpoint_mode = str(end_info.get("strategy") or "")
        except Exception:
            pass
    except Exception:
        baseline = {}
    if not end_nodes:
        end_nodes = [str(x).strip() for x in list(baseline.get("end_node_names") or []) if str(x).strip()]
    if not endpoint_mode:
        endpoint_mode = str(baseline.get("endpoint_mode") or ("raw_detection_head" if end_nodes else ""))
    return baseline, end_nodes, endpoint_mode


def _copy_small_suite_summary(suite_dir: Path, formal_bdir: Path, run_dir: Path) -> Dict[str, Path]:
    artifacts: Dict[str, Path] = {}
    mapping = {
        "benchmark_set.json": "legacy_benchmark_set.json",
        "benchmark_plan.json": "legacy_benchmark_plan.json",
        "generation_state.json": "legacy_generation_state.json",
        "benchmark_generation.log": "legacy_benchmark_generation.log",
        "README.md": "legacy_README.md",
    }
    for src_name, dst_name in mapping.items():
        src = suite_dir / src_name
        if src.is_file():
            dst = formal_bdir / dst_name
            try:
                shutil.copy2(src, dst)
                artifacts[dst_name.replace(".", "_")] = dst
            except Exception:
                pass
    return artifacts


def _cases_from_suite_payload(payload: Mapping[str, Any]) -> List[Dict[str, Any]]:
    raw = payload.get("cases") or payload.get("accepted_cases") or []
    return [dict(x) for x in list(raw or []) if isinstance(x, Mapping)]


def _rejected_from_generation_state(payload: Mapping[str, Any]) -> List[Dict[str, Any]]:
    raw = payload.get("discarded_case_entries") or payload.get("discarded_cases") or payload.get("rejected_cases") or []
    return [dict(x) for x in list(raw or []) if isinstance(x, Mapping)]


def _generation_case_identity(candidate: Mapping[str, Any], fallback_index: int = 0) -> tuple[str, Optional[int]]:
    boundary = _boundary_from_candidate(candidate)
    case_id = str(
        candidate.get("case_id")
        or candidate.get("case")
        or candidate.get("folder")
        or candidate.get("case_dir")
        or candidate.get("id")
        or ""
    ).strip()
    numeric_case_boundary: Optional[int] = None
    case_token = case_id.lower()
    if case_token.startswith("b") and case_token[1:].isdigit():
        numeric_case_boundary = int(case_token[1:])
        if boundary is not None and int(boundary) != numeric_case_boundary:
            raise ValueError(
                "Generator case identity is contradictory: "
                f"case_id={case_id!r} boundary={boundary!r}."
            )
        boundary = numeric_case_boundary if boundary is None else int(boundary)
        case_id = f"b{int(boundary):03d}"
    if not case_id and boundary is not None:
        case_id = f"b{int(boundary):03d}"
    if not case_id and fallback_index:
        case_id = f"candidate_{int(fallback_index):04d}"
    return case_id, boundary


def _is_frozen_predeclared_execution_union(candidate_plan: Mapping[str, Any]) -> bool:
    """Return whether generation must consume an exact, prospectively frozen union.

    ``requested_cases`` retains its historical meaning in a final candidate
    plan: it limits the predictor-selected deployment shortlist.  A
    score-independent audit expands that shortlist with a prospectively frozen
    audit universe.  The benchmark generator must therefore not reuse
    ``requested_cases`` as a limit for the expanded execution union.
    """

    scope = str(candidate_plan.get("candidate_universe_scope") or "").strip().lower()
    if scope != "predeclared_audit_universe":
        return False

    mode = str(candidate_plan.get("candidate_universe_mode") or "").strip().lower()
    if mode not in {"all_feasible", "deterministic_audit"}:
        raise ValueError(
            "Frozen predeclared execution union has an unsupported "
            f"candidate_universe_mode={mode!r}."
        )
    if candidate_plan.get("score_independent") is not True:
        raise ValueError(
            "Frozen predeclared execution union is not explicitly marked "
            "score_independent=true."
        )
    if candidate_plan.get("selection_uses_predictions") is not False:
        raise ValueError(
            "Frozen predeclared execution union must explicitly record "
            "selection_uses_predictions=false."
        )
    if candidate_plan.get("selection_uses_measurements") is not False:
        raise ValueError(
            "Frozen predeclared execution union must explicitly record "
            "selection_uses_measurements=false."
        )
    return True


def _require_complete_hailo_matrix_for_candidate_plan(
    candidate_plan: Mapping[str, Any],
) -> bool:
    """Backend exclusions never discard another executable path of a case."""
    return False


def _deduplicated_generation_identities(
    candidates: Sequence[Mapping[str, Any]],
    *,
    label: str,
    allow_exact_duplicates: bool = True,
) -> List[tuple[str, int]]:
    """Canonicalize and stably deduplicate candidate identities.

    Audit/deployment overlap is expected and is executed exactly once.  Any
    other case-id/boundary collision is ambiguous and is rejected rather than
    silently choosing one identity.
    """

    identities: List[tuple[str, int]] = []
    case_to_boundary: Dict[str, int] = {}
    boundary_to_case: Dict[int, str] = {}
    for index, candidate in enumerate(list(candidates or []), start=1):
        if not isinstance(candidate, Mapping):
            raise ValueError(f"{label} contains a non-object candidate at position {index}.")
        case_id, boundary = _generation_case_identity(candidate, index)
        if not case_id or boundary is None:
            raise ValueError(
                f"{label} candidate at position {index} is missing a canonical "
                "case_id/boundary identity."
            )
        boundary = int(boundary)
        previous_boundary = case_to_boundary.get(case_id)
        previous_case = boundary_to_case.get(boundary)
        if previous_boundary is not None and previous_boundary != boundary:
            raise ValueError(
                f"{label} maps case_id={case_id!r} to conflicting boundaries "
                f"{previous_boundary} and {boundary}."
            )
        if previous_case is not None and previous_case != case_id:
            raise ValueError(
                f"{label} maps boundary={boundary} to conflicting case ids "
                f"{previous_case!r} and {case_id!r}."
            )
        case_to_boundary[case_id] = boundary
        boundary_to_case[boundary] = case_id
        identity = (case_id, boundary)
        if identity in identities:
            if not allow_exact_duplicates:
                raise ValueError(
                    f"{label} contains duplicate canonical identity "
                    f"case_id={case_id!r} boundary={boundary}."
                )
            continue
        identities.append(identity)
    return identities


def _predeclared_execution_union_identities(
    candidate_plan: Mapping[str, Any],
) -> List[tuple[str, int]]:
    selected_rows = list(candidate_plan.get("selected_candidates") or [])
    selected = _deduplicated_generation_identities(
        selected_rows,
        label="selected_candidates",
        allow_exact_duplicates=False,
    )
    if not selected:
        raise ValueError("Frozen predeclared execution union is empty.")

    # The long-form audit/deployment lists are independent evidence of the
    # intended union.  The ordinary frozen order is audit then deployment.  An
    # explicit forced deployment anchor may instead be placed first so an exact
    # cached feasibility artifact is consumed before the unchanged audit.  The
    # order policy is part of the sealed plan; it never adds a candidate.
    audit_rows = list(candidate_plan.get("audit_candidates") or [])
    deployment_rows = list(candidate_plan.get("deployment_shortlist") or [])
    if audit_rows or deployment_rows:
        audit = _deduplicated_generation_identities(
            audit_rows,
            label="audit_candidates",
            allow_exact_duplicates=False,
        )
        deployment = _deduplicated_generation_identities(
            deployment_rows,
            label="deployment_shortlist",
            allow_exact_duplicates=False,
        )
        order_policy = str(
            candidate_plan.get("execution_union_order_policy")
            or "audit_then_deployment"
        ).strip()
        if order_policy == "audit_then_deployment":
            ordered_roles = audit + deployment
        elif order_policy == "forced_deployment_then_audit":
            if not deployment:
                raise ValueError(
                    "Frozen forced-deployment execution order has no "
                    "deployment_shortlist candidates."
                )
            ordered_roles = deployment + audit
        else:
            raise ValueError(
                "Frozen predeclared execution union has an unsupported "
                f"execution_union_order_policy={order_policy!r}."
            )
        declared: List[tuple[str, int]] = []
        for identity in ordered_roles:
            if identity not in declared:
                declared.append(identity)
        if selected != declared:
            raise ValueError(
                "Frozen predeclared execution union does not match the "
                "deterministically deduplicated audit/deployment union under "
                f"{order_policy!r}: "
                f"selected={[case_id for case_id, _ in selected]!r} "
                f"declared={[case_id for case_id, _ in declared]!r}."
            )
        expected_roles: Dict[tuple[str, int], set[str]] = {
            identity: set() for identity in declared
        }
        for identity in audit:
            expected_roles[identity].add("audit")
        for identity in deployment:
            expected_roles[identity].add("deployment_shortlist")
        for index, row in enumerate(selected_rows, start=1):
            identity = _generation_case_identity(row, index)
            canonical_identity = (identity[0], int(identity[1]))
            roles = row.get("candidate_execution_roles")
            if isinstance(roles, str):
                role_set = {
                    value.strip()
                    for value in roles.replace(";", ",").split(",")
                    if value.strip()
                }
            elif isinstance(roles, Sequence) and not isinstance(
                roles, (str, bytes, bytearray)
            ):
                role_set = {
                    str(value).strip()
                    for value in roles
                    if str(value or "").strip()
                }
            else:
                role_set = set()
            if role_set != expected_roles[canonical_identity]:
                raise ValueError(
                    "Frozen selected_candidates role projection is not the "
                    "exact merged audit/deployment union for "
                    f"{canonical_identity[0]}."
                )
    return selected


def _verify_frozen_predeclared_candidate_bindings(
    candidate_plan: Mapping[str, Any],
    prediction: Mapping[str, Any],
    *,
    candidate_universe_path: str | Path | None,
    prediction_path: str | Path | None,
    selection_input_path: str | Path | None,
    model_path: str | Path | None,
    prediction_freeze_manifest_path: str | Path | None,
    expected_selection_policy: Mapping[str, Any] | None,
    expected_selection_strategy: str | None,
    expected_native_single_part2_input: bool | None,
) -> List[tuple[str, int]]:
    """Verify the archived identity chain before an audit union is executable."""

    universe_path = (
        Path(candidate_universe_path)
        if candidate_universe_path is not None
        else Path()
    )
    if not universe_path.is_file():
        raise ValueError(
            "Frozen predeclared execution union is missing its authoritative "
            "candidate_universe_manifest.json binding."
        )
    candidate_universe = _read_json(universe_path, default={}) or {}
    if not isinstance(candidate_universe, Mapping) or not candidate_universe:
        raise ValueError(
            "Frozen predeclared execution union candidate-universe manifest "
            "is not valid JSON."
        )
    universe = dict(candidate_universe)
    stored_universe_sha256 = str(universe.get("universe_sha256") or "").strip()
    computed_universe_sha256 = sha256_json({
        key: value
        for key, value in universe.items()
        if key not in {"universe_sha256", "created_at"}
    })
    plan_universe_sha256 = str(
        candidate_plan.get("candidate_universe_sha256") or ""
    ).strip()
    if (
        not stored_universe_sha256
        or stored_universe_sha256 != computed_universe_sha256
        or plan_universe_sha256 != stored_universe_sha256
    ):
        raise ValueError(
            "Frozen predeclared execution union has a stale or tampered "
            "candidate-universe binding: "
            f"plan={plan_universe_sha256!r} stored={stored_universe_sha256!r} "
            f"computed={computed_universe_sha256!r}."
        )

    model_id = str(candidate_plan.get("model_id") or "").strip()
    prediction_model_id = str(prediction.get("model_id") or "").strip()
    universe_model_id = str(universe.get("model_id") or "").strip()
    if not model_id or model_id != prediction_model_id or model_id != universe_model_id:
        raise ValueError(
            "Frozen predeclared execution union model binding mismatch: "
            f"plan={model_id!r} prediction={prediction_model_id!r} "
            f"universe={universe_model_id!r}."
        )

    prediction_artifact_id = str(prediction.get("artifact_id") or "").strip()
    expected_prediction_artifact_id = (
        f"{slugify('prediction')}_{slugify(model_id)}_"
        f"{sha256_payload(list(prediction.get('candidates') or []))[:12]}"
    )
    source_prediction_artifact_id = str(
        candidate_plan.get("source_prediction_artifact_id") or ""
    ).strip()
    if (
        not prediction_artifact_id
        or prediction_artifact_id != expected_prediction_artifact_id
        or source_prediction_artifact_id != prediction_artifact_id
    ):
        raise ValueError(
            "Frozen predeclared execution union prediction artifact binding "
            "mismatch: "
            f"plan={source_prediction_artifact_id!r} "
            f"prediction={prediction_artifact_id!r} "
            f"computed={expected_prediction_artifact_id!r}."
        )

    archived_prediction_path = (
        Path(prediction_path) if prediction_path is not None else Path()
    )
    if not archived_prediction_path.is_file():
        raise ValueError(
            "Frozen predeclared execution union is missing its archived "
            "prediction.json binding."
        )
    archived_prediction = _read_json(archived_prediction_path, default={}) or {}
    if not isinstance(archived_prediction, Mapping) or dict(archived_prediction) != dict(prediction):
        raise ValueError(
            "Frozen predeclared execution union runtime prediction payload "
            "does not match the archived prediction.json."
        )
    prediction_sha256 = sha256_file(archived_prediction_path) or ""
    expected_prediction_sha256 = str(
        universe.get("source_prediction_sha256") or ""
    ).strip()
    if (
        not prediction_sha256
        or not expected_prediction_sha256
        or str(prediction_sha256).strip() != expected_prediction_sha256
    ):
        raise ValueError(
            "Frozen predeclared execution union prediction content binding "
            "mismatch: "
            f"manifest={expected_prediction_sha256!r} "
            f"actual={str(prediction_sha256 or '')!r}."
        )

    if (
        str(candidate_plan.get("candidate_universe_mode") or "").strip().lower()
        != str(universe.get("mode") or "").strip().lower()
        or str(candidate_plan.get("candidate_universe_scope") or "").strip().lower()
        != str(universe.get("claim_scope") or "").strip().lower()
    ):
        raise ValueError(
            "Frozen predeclared execution union mode/scope does not match its "
            "candidate-universe manifest."
        )

    universe_rows = list(universe.get("candidates") or [])
    universe_identities = _deduplicated_generation_identities(
        universe_rows,
        label="candidate_universe_manifest.candidates",
        allow_exact_duplicates=False,
    )
    universe_by_identity: Dict[tuple[str, int], Dict[str, Any]] = {}
    model_sha256 = str(universe.get("model_sha256") or "").strip()
    if not model_sha256:
        raise ValueError(
            "Frozen predeclared execution union manifest is missing model_sha256."
        )

    actual_model_path = Path(model_path) if model_path is not None else Path()
    actual_model_sha256 = (
        sha256_file(actual_model_path) or ""
        if actual_model_path.is_file()
        else ""
    )
    freeze_path = (
        Path(prediction_freeze_manifest_path)
        if prediction_freeze_manifest_path is not None
        else Path()
    )
    freeze = _read_json(freeze_path, default={}) or {} if freeze_path.is_file() else {}
    freeze_binding_ok = bool(
        isinstance(freeze, Mapping)
        and str(freeze.get("model_sha256") or "") == model_sha256
        and str(freeze.get("prediction_sha256") or "") == prediction_sha256
        and str(freeze.get("candidate_universe_sha256") or "")
        == stored_universe_sha256
    )
    if actual_model_sha256:
        if actual_model_sha256 != model_sha256:
            raise ValueError(
                "Frozen candidate-universe model_sha256 does not match the "
                "actual ONNX model file: "
                f"manifest={model_sha256!r} actual={actual_model_sha256!r}."
            )
    elif not freeze_binding_ok:
        raise ValueError(
            "Frozen candidate-universe model identity is not anchored to an "
            "actual model file or a coherent prediction-freeze manifest."
        )
    if freeze_path.is_file() and not freeze_binding_ok:
        raise ValueError(
            "Frozen prediction-freeze manifest does not coherently bind the "
            "model, prediction and candidate universe."
        )

    # Reuse the candidate identity contract used when the prospective universe
    # is created.  The manifest model hash is its authoritative identity
    # context; score/rank/measurement fields are intentionally excluded.
    from ..campaign import stable_candidate_identity

    for index, (row, identity) in enumerate(
        zip(universe_rows, universe_identities),
        start=1,
    ):
        canonical = stable_candidate_identity(
            model_id,
            {**dict(row), "model_sha256": model_sha256},
            index,
        )
        stored_candidate_id = str(row.get("candidate_id") or "").strip()
        stored_candidate_sha256 = str(
            row.get("candidate_identity_sha256") or ""
        ).strip()
        if (
            stored_candidate_id != canonical["candidate_id"]
            or stored_candidate_sha256
            != canonical["candidate_identity_sha256"]
        ):
            raise ValueError(
                "Frozen candidate-universe identity hash mismatch for "
                f"{identity[0]}: candidate_id={stored_candidate_id!r} "
                f"candidate_identity_sha256={stored_candidate_sha256!r}."
            )
        universe_by_identity[identity] = dict(row)

    manifest_selected_case_ids = [
        str(value).strip()
        for value in list(universe.get("selected_case_ids") or [])
        if str(value or "").strip()
    ]
    manifest_selected_identities = [
        identity
        for identity in universe_identities
        if identity[0] in set(manifest_selected_case_ids)
    ]
    manifest_selected_by_case = {
        identity[0]: identity for identity in manifest_selected_identities
    }
    manifest_selected_ordered = [
        manifest_selected_by_case[case_id]
        for case_id in manifest_selected_case_ids
        if case_id in manifest_selected_by_case
    ]
    if len(manifest_selected_ordered) != len(manifest_selected_case_ids):
        raise ValueError(
            "Frozen candidate-universe selected_case_ids do not resolve to "
            "canonical manifest candidates."
        )
    selected_candidate_ids = [
        str(universe_by_identity[identity].get("candidate_id") or "")
        for identity in manifest_selected_ordered
    ]
    if (
        list(universe.get("selected_candidate_ids") or [])
        != selected_candidate_ids
        or str(universe.get("selected_candidate_identity_sha256") or "")
        != sha256_json(selected_candidate_ids)
    ):
        raise ValueError(
            "Frozen candidate-universe selected identity binding is invalid."
        )
    feasible_candidate_ids = [
        str(universe_by_identity[identity].get("candidate_id") or "")
        for identity in universe_identities
    ]
    if (
        str(universe.get("feasible_candidate_identity_sha256") or "")
        != sha256_json(feasible_candidate_ids)
        or int(universe.get("feasible_candidate_count") or -1)
        != len(universe_identities)
        or int(universe.get("selected_candidate_count") or -1)
        != len(manifest_selected_ordered)
    ):
        raise ValueError(
            "Frozen candidate-universe feasible identity/count binding is invalid."
        )

    prediction_identities = _deduplicated_generation_identities(
        list(prediction.get("candidates") or []),
        label="prediction.candidates",
        allow_exact_duplicates=False,
    )
    if (
        len(prediction_identities) != len(universe_identities)
        or set(prediction_identities) != set(universe_identities)
    ):
        only_prediction = [
            case_id
            for case_id, boundary in prediction_identities
            if (case_id, boundary) not in set(universe_identities)
        ]
        only_universe = [
            case_id
            for case_id, boundary in universe_identities
            if (case_id, boundary) not in set(prediction_identities)
        ]
        raise ValueError(
            "Frozen candidate universe and prediction candidates are not an "
            "exact case_id+boundary bijection: "
            f"prediction_only={only_prediction!r} "
            f"universe_only={only_universe!r}."
        )
    prediction_identity_set = set(prediction_identities)

    def _verify_plan_rows(key: str) -> List[tuple[str, int]]:
        rows = list(candidate_plan.get(key) or [])
        identities = _deduplicated_generation_identities(
            rows,
            label=key,
            allow_exact_duplicates=False,
        )
        for index, row in enumerate(rows, start=1):
            case_id, boundary = _generation_case_identity(row, index)
            if not case_id or boundary is None:
                raise ValueError(
                    f"Frozen {key} candidate at position {index} has no "
                    "canonical identity."
                )
            identity = (case_id, int(boundary))
            canonical_row = universe_by_identity.get(identity)
            if canonical_row is None:
                raise ValueError(
                    f"Frozen {key} candidate {identity[0]!r} is absent from "
                    "candidate_universe_manifest.candidates."
                )
            if identity not in prediction_identity_set:
                raise ValueError(
                    f"Frozen {key} candidate {identity[0]!r} has no exact "
                    "case_id+boundary match in prediction.candidates."
                )
            if (
                str(row.get("candidate_id") or "").strip()
                != str(canonical_row.get("candidate_id") or "").strip()
                or str(row.get("candidate_identity_sha256") or "").strip()
                != str(canonical_row.get("candidate_identity_sha256") or "").strip()
            ):
                raise ValueError(
                    f"Frozen {key} candidate identity hash mismatch for "
                    f"{identity[0]}."
                )
        return identities

    audit_identities = _verify_plan_rows("audit_candidates")
    deployment_identities = _verify_plan_rows("deployment_shortlist")
    selected_identities = _verify_plan_rows("selected_candidates")
    if audit_identities != manifest_selected_ordered:
        raise ValueError(
            "Frozen audit_candidates do not match the candidate-universe "
            "selected_case_ids in their prospective order."
        )

    if not isinstance(expected_selection_policy, Mapping):
        raise ValueError(
            "Frozen candidate selection is missing the resolved evaluation "
            "profile selection policy."
        )
    resolved_policy = dict(expected_selection_policy)
    expected_requested_cases = max(
        1,
        _first_policy_int(
            resolved_policy,
            "max_accepted_cases_per_model",
            "accepted_cases",
            default=5,
        ),
    )
    from .selection_request_counts import requested_cases_for_model
    expected_requested_cases = requested_cases_for_model(
        resolved_policy, model_id, default=expected_requested_cases, audit_scope=True)
    expected_min_gap = max(
        0,
        _first_policy_int(resolved_policy, "min_gap", default=2),
    )
    if expected_selection_strategy is None:
        raise ValueError(
            "Frozen candidate selection is missing the resolved evaluation "
            "profile selection strategy."
        )
    expected_raw_strategy = _normalise_selection_strategy(
        expected_selection_strategy
    )
    if not expected_raw_strategy:
        raise ValueError(
            "Frozen candidate selection has no resolved profile strategy."
        )

    prediction_requested_cases = _runner_int_or_none(
        prediction.get("requested_cases")
    )
    if prediction_requested_cases != expected_requested_cases:
        raise ValueError(
            "Frozen prediction requested_cases does not match the resolved "
            "evaluation profile: "
            f"prediction={prediction_requested_cases} "
            f"profile={expected_requested_cases}."
        )
    prediction_min_gap = _runner_int_or_none(prediction.get("min_gap"))
    if prediction_min_gap != expected_min_gap:
        raise ValueError(
            "Frozen prediction min_gap does not match the resolved "
            "evaluation profile: "
            f"prediction={prediction_min_gap} profile={expected_min_gap}."
        )
    prediction_strategy = _normalise_selection_strategy(
        prediction.get("selection_strategy")
    )
    if prediction_strategy != expected_raw_strategy:
        raise ValueError(
            "Frozen prediction selection_strategy does not match the "
            "resolved evaluation profile: "
            f"prediction={prediction_strategy!r} "
            f"profile={expected_raw_strategy!r}."
        )

    requested_cases = _runner_int_or_none(candidate_plan.get("requested_cases"))
    if requested_cases != expected_requested_cases:
        raise ValueError(
            "Frozen candidate-plan requested_cases does not match the "
            "resolved evaluation profile: "
            f"plan={requested_cases} profile={expected_requested_cases}."
        )
    if len(deployment_identities) > requested_cases:
        raise ValueError(
            "Frozen deployment_shortlist exceeds the resolved "
            "requested_cases limit: "
            f"deployment={len(deployment_identities)} "
            f"requested_cases={requested_cases}."
        )
    plan_min_gap = _runner_int_or_none(candidate_plan.get("min_gap"))
    if plan_min_gap != expected_min_gap:
        raise ValueError(
            "Frozen candidate-plan min_gap does not match the resolved "
            "evaluation profile: "
            f"plan={plan_min_gap} profile={expected_min_gap}."
        )
    plan_strategy = _normalise_selection_strategy(
        candidate_plan.get("selection_strategy")
    )
    if plan_strategy != "score_independent_audit":
        raise ValueError(
            "Frozen candidate-plan selection_strategy must normalize to "
            "score_independent_audit."
        )
    deployment_rows = list(candidate_plan.get("deployment_shortlist") or [])
    deployment_ranks = [
        _runner_int_or_none(row.get("deployment_rank"))
        for row in deployment_rows
    ]
    if deployment_ranks != list(range(1, len(deployment_rows) + 1)):
        raise ValueError(
            "Frozen deployment_shortlist deployment_rank values must be "
            "unique, sequential and match list order."
        )

    selection_path = (
        Path(selection_input_path)
        if selection_input_path is not None
        else Path()
    )
    selection_input = _read_json(selection_path, default={}) or {}
    if not selection_path.is_file() or not isinstance(selection_input, Mapping):
        raise ValueError(
            "Frozen deployment_shortlist is missing its authoritative "
            "selection_input.json."
        )
    if (
        str(selection_input.get("model_id") or "") != model_id
        or str(selection_input.get("source_prediction_artifact_id") or "")
        != prediction_artifact_id
        or str(selection_input.get("candidate_universe_sha256") or "")
        != stored_universe_sha256
        or _runner_int_or_none(selection_input.get("requested_cases"))
        != requested_cases
        or _runner_int_or_none(
            selection_input.get("deployment_shortlist_count")
        )
        != len(deployment_identities)
    ):
        raise ValueError(
            "Frozen deployment_shortlist selection_input binding is "
            "inconsistent with plan/prediction/universe evidence."
        )
    min_gap = _runner_int_or_none(selection_input.get("min_gap"))
    if min_gap != expected_min_gap:
        raise ValueError(
            "Frozen selection_input min_gap does not match the resolved "
            "evaluation profile: "
            f"selection_input={min_gap} profile={expected_min_gap}."
        )
    selection_strategy = _normalise_selection_strategy(
        selection_input.get("selection_strategy")
    )
    if selection_strategy != expected_raw_strategy:
        raise ValueError(
            "Frozen selection_input selection_strategy does not match the "
            "resolved evaluation profile: "
            f"selection_input={selection_strategy!r} "
            f"profile={expected_raw_strategy!r}."
        )

    profile_policy = (
        selection_input.get("profile_selection_policy")
        if isinstance(selection_input.get("profile_selection_policy"), Mapping)
        else {}
    )
    if (
        dict(profile_policy) != resolved_policy
    ):
        raise ValueError(
            "Frozen selection_input profile_selection_policy does not match "
            "the resolved evaluation profile."
        )
    forced_raw = (
        profile_policy.get("forced_cases")
        or profile_policy.get("fixed_cases")
        or profile_policy.get("case_map")
        or {}
    )
    forced_case_ids: List[str] = []
    if isinstance(forced_raw, Mapping):
        forced_values = forced_raw.get(model_id) or []
        if isinstance(forced_values, str):
            forced_case_ids = [
                value.strip()
                for value in forced_values.replace(";", ",").split(",")
                if value.strip()
            ]
        elif isinstance(forced_values, Sequence) and not isinstance(
            forced_values, (str, bytes, bytearray)
        ):
            forced_case_ids = [
                str(value).strip()
                for value in forced_values
                if str(value or "").strip()
            ]
    elif isinstance(forced_raw, str) and forced_raw.strip():
        try:
            parsed_forced = json.loads(forced_raw)
        except Exception as exc:
            raise ValueError(
                "Frozen selection policy forced_cases string is not valid "
                "JSON."
            ) from exc
        forced_values = (
            parsed_forced.get(model_id)
            if isinstance(parsed_forced, Mapping)
            else []
        )
        if isinstance(forced_values, str):
            forced_case_ids = [
                value.strip()
                for value in forced_values.replace(";", ",").split(",")
                if value.strip()
            ]
        elif isinstance(forced_values, Sequence) and not isinstance(
            forced_values, (str, bytes, bytearray)
        ):
            forced_case_ids = [
                str(value).strip()
                for value in forced_values
                if str(value or "").strip()
            ]
    normalized_forced_case_ids: List[str] = []
    for raw_case_id in forced_case_ids:
        token = str(raw_case_id or "").strip().lower()
        if not re.fullmatch(r"b\d+", token):
            raise ValueError(
                "Frozen audit forced_cases contains a non-canonical case "
                f"identity: {raw_case_id!r}."
            )
        case_id = f"b{int(token[1:]):03d}"
        if case_id in normalized_forced_case_ids:
            raise ValueError(
                "Frozen audit forced_cases contains a duplicate case "
                f"identity: {case_id!r}."
            )
        normalized_forced_case_ids.append(case_id)
    forced_case_ids = normalized_forced_case_ids
    if len(forced_case_ids) > requested_cases:
        raise ValueError(
            "Frozen audit forced_cases exceeds the resolved deployment "
            f"shortlist limit: forced={len(forced_case_ids)} "
            f"requested_cases={requested_cases}."
        )

    expected_order_policy = (
        "forced_deployment_then_audit"
        if forced_case_ids
        else "audit_then_deployment"
    )
    plan_order_policy = str(
        candidate_plan.get("execution_union_order_policy")
        or "audit_then_deployment"
    ).strip()
    selection_order_policy = str(
        selection_input.get("execution_union_order_policy")
        or "audit_then_deployment"
    ).strip()
    if (
        plan_order_policy != expected_order_policy
        or selection_order_policy != expected_order_policy
    ):
        raise ValueError(
            "Frozen audit execution-union order does not match the resolved "
            "forced-case policy: "
            f"expected={expected_order_policy!r} "
            f"plan={plan_order_policy!r} "
            f"selection_input={selection_order_policy!r}."
        )
    prediction_rows = list(prediction.get("candidates") or [])
    prediction_by_case = {
        identity[0]: (dict(row), identity)
        for row, identity in zip(
            prediction_rows,
            prediction_identities,
        )
    }
    if expected_native_single_part2_input is None:
        raise ValueError(
            "Frozen candidate selection is missing the resolved Native "
            "split-boundary capability."
        )
    requested_single_part2_input = bool(
        resolved_policy.get("require_single_part2_input", False)
    )
    native_single_part2_input = bool(expected_native_single_part2_input)
    effective_single_part2_input = requested_single_part2_input
    expected_capability_flags = {
        "require_single_part2_input": requested_single_part2_input,
        "requested_require_single_part2_input": (
            requested_single_part2_input
        ),
        "native_split_requires_single_part2_input": (
            native_single_part2_input
        ),
        "effective_require_single_part2_input": (
            effective_single_part2_input
        ),
    }
    for evidence_name, evidence in (
        ("prediction", prediction),
        ("candidate-plan", candidate_plan),
        ("selection_input", selection_input),
    ):
        mismatched_flags = {
            key: {
                "stored": evidence.get(key),
                "expected": expected,
            }
            for key, expected in expected_capability_flags.items()
            if evidence.get(key) is not expected
        }
        if mismatched_flags:
            raise ValueError(
                f"Frozen {evidence_name} Part-2 capability flags do not "
                "match the resolved profile/native configuration: "
                f"{mismatched_flags!r}."
            )

    eligible_prediction_rows = [
        (dict(row), identity)
        for row, identity in zip(prediction_rows, prediction_identities)
        if (
            not effective_single_part2_input
            or _runner_int_or_none(row.get("part2_input_count")) == 1
        )
    ]
    if effective_single_part2_input and len(eligible_prediction_rows) != len(
        prediction_rows
    ):
        ineligible_case_ids = [
            identity[0]
            for row, identity in zip(prediction_rows, prediction_identities)
            if _runner_int_or_none(row.get("part2_input_count")) != 1
        ]
        raise ValueError(
            "Frozen prediction.candidates contains candidates outside the "
            "resolved single-Part2-input capability, although the runner "
            "applies that filter before freezing prediction/universe: "
            f"{ineligible_case_ids!r}."
        )
    if _runner_int_or_none(
        selection_input.get("eligible_candidate_count")
    ) != len(eligible_prediction_rows):
        raise ValueError(
            "Frozen selection_input eligible_candidate_count does not match "
            "the resolved Part-2 capability filter."
        )
    eligible_by_case = {
        identity[0]: (row, identity)
        for row, identity in eligible_prediction_rows
    }
    eligible_identity_set = {
        identity for _row, identity in eligible_prediction_rows
    }
    for projection_name, projection in (
        ("audit_candidates", audit_identities),
        ("deployment_shortlist", deployment_identities),
        ("selected_candidates", selected_identities),
    ):
        ineligible = [
            case_id
            for case_id, boundary in projection
            if (case_id, boundary) not in eligible_identity_set
        ]
        if ineligible:
            raise ValueError(
                f"Frozen {projection_name} contains candidates outside the "
                "resolved single-Part2-input eligible prediction set: "
                f"{ineligible!r}."
            )
    if forced_case_ids:
        missing_forced_case_ids = [
            case_id
            for case_id in forced_case_ids
            if case_id not in eligible_by_case
        ]
        if missing_forced_case_ids:
            raise ValueError(
                "Frozen audit forced_cases is absent from the resolved "
                "capability-eligible prediction set: "
                f"{missing_forced_case_ids!r}."
            )
        deployment_source = [
            eligible_by_case[case_id]
            for case_id in forced_case_ids
        ]
    else:
        deployment_source = list(eligible_prediction_rows)
    expected_deployment: List[tuple[str, int]] = []
    previous_boundary: Optional[int] = None
    for _prediction_row, identity in deployment_source:
        if len(expected_deployment) >= requested_cases:
            break
        boundary = int(identity[1])
        if (
            previous_boundary is not None
            and abs(boundary - previous_boundary) < min_gap
        ):
            continue
        expected_deployment.append(identity)
        previous_boundary = boundary
    if forced_case_ids and [
        case_id for case_id, _boundary in expected_deployment
    ] != forced_case_ids:
        raise ValueError(
            "Frozen audit forced_cases cannot be materialized exactly under "
            "the resolved deployment min-gap policy: "
            f"forced={forced_case_ids!r} "
            "materialized="
            f"{[case_id for case_id, _boundary in expected_deployment]!r}."
        )
    if deployment_identities != expected_deployment:
        raise ValueError(
            "Frozen deployment_shortlist does not match the authoritative "
            "deterministic prediction/forced-case order: "
            f"expected={[case_id for case_id, _ in expected_deployment]!r} "
            f"actual={[case_id for case_id, _ in deployment_identities]!r}."
        )
    for deployment_row, identity in zip(deployment_rows, deployment_identities):
        prediction_row = prediction_by_case[identity[0]][0]
        prediction_rank = _runner_int_or_none(prediction_row.get("rank"))
        if (
            prediction_rank is None
            or _runner_int_or_none(deployment_row.get("source_rank"))
            != prediction_rank
            or _runner_int_or_none(deployment_row.get("rank"))
            != prediction_rank
        ):
            raise ValueError(
                "Frozen deployment_shortlist source_rank/rank does not match "
                f"prediction.candidates for {identity[0]}."
            )

    selected_rows = list(candidate_plan.get("selected_candidates") or [])
    candidate_plan_artifact_id = str(
        candidate_plan.get("artifact_id") or ""
    ).strip()
    expected_candidate_plan_artifact_id = (
        f"{slugify('candidate_plan')}_{slugify(model_id)}_"
        f"{sha256_payload(selected_rows)[:12]}"
    )
    if candidate_plan_artifact_id != expected_candidate_plan_artifact_id:
        raise ValueError(
            "Frozen candidate-plan artifact binding mismatch: "
            f"stored={candidate_plan_artifact_id!r} "
            f"computed={expected_candidate_plan_artifact_id!r}."
        )

    return _predeclared_execution_union_identities(candidate_plan)


def _resolve_generation_candidate_scope(
    candidate_plan: Mapping[str, Any],
    prediction: Mapping[str, Any],
    *,
    candidate_universe_path: str | Path | None = None,
    prediction_path: str | Path | None = None,
    selection_input_path: str | Path | None = None,
    model_path: str | Path | None = None,
    prediction_freeze_manifest_path: str | Path | None = None,
    expected_selection_policy: Mapping[str, Any] | None = None,
    expected_selection_strategy: str | None = None,
    expected_native_single_part2_input: bool | None = None,
) -> tuple[List[int], List[int], int, bool]:
    """Resolve the exact generator count/order/pool for one candidate plan."""

    ranked_candidates, candidate_search_pool = _candidate_boundaries(
        candidate_plan,
        prediction,
    )
    requested = _safe_int(
        candidate_plan.get("requested_cases"),
        max(1, len(ranked_candidates)),
    )
    if requested <= 0:
        requested = max(1, len(ranked_candidates))

    frozen_union = _is_frozen_predeclared_execution_union(candidate_plan)
    if not frozen_union:
        identities = _deduplicated_generation_identities(
            list(candidate_plan.get("selected_candidates") or []),
            label="selected_candidates", allow_exact_duplicates=False,
        )
        prediction_ids = set(_deduplicated_generation_identities(
            list(prediction.get("candidates") or []), label="prediction.candidates",
        ))
        if any(identity not in prediction_ids for identity in identities):
            raise ValueError("Selected Generic case is missing from prediction.candidates.")
        if any(case != f"b{boundary:03d}" for case, boundary in identities):
            raise ValueError("Selected Generic scope contains a non-canonical case identity.")
        policy = dict(expected_selection_policy or {})
        forced = policy.get("forced_cases") or policy.get("fixed_cases") or policy.get("case_map") or {}
        if isinstance(forced, str):
            try:
                forced = json.loads(forced)
            except Exception as exc:
                raise ValueError("Forced candidate scope is not valid JSON.") from exc
        model = str(candidate_plan.get("model_id") or "")
        declared = (forced.get(model) or []) if isinstance(forced, Mapping) else []
        if not model and isinstance(forced, Mapping) and len(forced) == 1:
            declared = next(iter(forced.values())) or []
        if isinstance(declared, str):
            declared = [value.strip() for value in declared.replace(";", ",").split(",") if value.strip()]
        if declared:
            if any(not re.fullmatch(r"b\d+", str(value).strip().lower()) for value in declared):
                raise ValueError("Forced candidate scope contains a non-canonical declared case identity.")
            declared = [f"b{int(str(value).strip()[1:]):03d}" for value in declared]
            if len(set(declared)) != len(declared):
                raise ValueError("Forced candidate scope contains a duplicate declared case identity.")
            # Only the user's explicit Single-Tensor constraint can remove a
            # declared forced case, with its existing prediction evidence.
            excluded = set()
            if policy.get("require_single_part2_input") is True:
                for row in prediction.get("policy_excluded_candidates") or []:
                    if (isinstance(row, Mapping)
                            and row.get("exclude_source") == "requested_selection_policy"
                            and row.get("exclude_reason") == "part2_input_count_not_one"
                            and _runner_int_or_none(row.get("part2_input_count")) != 1):
                        excluded.add(_generation_case_identity(row)[0])
            expected = [case for case in declared if case not in excluded]
            if [case for case, _boundary in identities] != expected:
                raise ValueError("Forced candidate scope selected_candidates do not match the declared cases/order.")
        # The existing plan is the selected cohort, even when fewer than the
        # requested number are suitable. Backend results cannot add cases.
        boundaries = [boundary for _case, boundary in identities]
        return list(boundaries), list(boundaries), len(boundaries), True

    identities = _verify_frozen_predeclared_candidate_bindings(
        candidate_plan,
        prediction,
        candidate_universe_path=candidate_universe_path,
        prediction_path=prediction_path,
        selection_input_path=selection_input_path,
        model_path=model_path,
        prediction_freeze_manifest_path=prediction_freeze_manifest_path,
        expected_selection_policy=expected_selection_policy,
        expected_selection_strategy=expected_selection_strategy,
        expected_native_single_part2_input=(
            expected_native_single_part2_input
        ),
    )
    exact_boundaries = [boundary for _case_id, boundary in identities]
    prediction_boundaries = {
        int(boundary)
        for boundary in (
            _boundary_from_candidate(candidate)
            for candidate in list(prediction.get("candidates") or [])
            if isinstance(candidate, Mapping)
        )
        if boundary is not None
    }
    missing_from_prediction = [
        case_id
        for case_id, boundary in identities
        if boundary not in prediction_boundaries
    ]
    if missing_from_prediction:
        raise ValueError(
            "Frozen predeclared execution union contains candidates absent "
            f"from prediction.candidates: {missing_from_prediction!r}."
        )

    # The exact union is both shortlist and search pool.  This prevents the
    # mature generator's normal replacement policy from introducing a boundary
    # that was not prospectively frozen.
    return exact_boundaries, list(exact_boundaries), len(exact_boundaries), True


def _post_build_audit_minimum_projection(
    candidate_plan: Mapping[str, Any],
    accepted_identities: Sequence[tuple[str, int]],
    *,
    pending_direct_fallback: bool = False,
) -> Dict[str, Any]:
    """Describe the frozen audit's post-build minimum without backfilling.

    The minimum applies to prospectively declared audit candidates, not to the
    deployment-only rows that may also be present in the execution union.
    Generator rejection is an observed outcome.  It can require a later,
    explicitly scoped repair/resume, but it must never trigger an adaptive
    replacement after the audit selection has been frozen.
    """
    accepted_set = {
        (str(case_id), int(boundary))
        for case_id, boundary in accepted_identities
    }
    audit_identities: List[tuple[str, int]] = []
    for index, row in enumerate(
        list(candidate_plan.get("audit_candidates") or []),
        start=1,
    ):
        if not isinstance(row, Mapping):
            continue
        case_id, boundary = _generation_case_identity(row, index)
        if case_id and boundary is not None:
            identity = (case_id, int(boundary))
            if identity not in audit_identities:
                audit_identities.append(identity)

    minimum_raw = candidate_plan.get("minimum_valid_audit_candidates")
    minimum_required = (
        max(1, _safe_int(minimum_raw, 1))
        if minimum_raw is not None
        else None
    )
    materialized_audit_count = sum(
        1 for identity in audit_identities if identity in accepted_set
    )
    base = {
        "post_build_minimum_required": minimum_required,
        "post_build_audit_candidate_count": len(audit_identities),
        "post_build_audit_materialized_count": materialized_audit_count,
        "automatic_post_build_backfill_performed": False,
        "automatic_post_build_backfill_allowed": False,
    }
    if pending_direct_fallback:
        base.update({
            "post_build_minimum_status": "pending_direct_fallback",
            "execution_status": "pending_direct_fallback",
            "repair_required": False,
            "repair_status": "not_evaluated_pending_direct_fallback",
        })
        return base
    if minimum_required is None:
        base.update({
            "post_build_minimum_status": "not_configured",
            "execution_status": "ready",
            "repair_required": False,
            "repair_status": "not_required",
        })
        return base
    if materialized_audit_count >= minimum_required:
        base.update({
            "post_build_minimum_status": "met",
            "execution_status": "ready",
            "repair_required": False,
            "repair_status": "not_required",
        })
        return base
    base.update({
        "post_build_minimum_status": "shortfall",
        "execution_status": "insufficient_materialized_audit_candidates",
        "repair_required": True,
        "repair_status": "required_no_automatic_backfill",
        "repair_reason": (
            "post_build_audit_minimum_shortfall:"
            f"{materialized_audit_count}<{minimum_required}"
        ),
    })
    return base


def reconcile_candidate_plan_after_generation(
    candidate_plan: Mapping[str, Any],
    *,
    prediction: Mapping[str, Any],
    accepted_cases: Sequence[Mapping[str, Any]],
    rejected_cases: Sequence[Mapping[str, Any]],
    generation_summary: Mapping[str, Any] | None = None,
    backend_selection_state: Mapping[str, Any] | None = None,
    candidate_universe_path: str | Path | None = None,
    prediction_path: str | Path | None = None,
    selection_input_path: str | Path | None = None,
    model_path: str | Path | None = None,
    prediction_freeze_manifest_path: str | Path | None = None,
    expected_selection_policy: Mapping[str, Any] | None = None,
    expected_selection_strategy: str | None = None,
    expected_native_single_part2_input: bool | None = None,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Keep selected cases fixed and record the generated accepted/rejected subset.

    The existing plan remains authoritative for IDs and order. Backend or
    Native exclusions cannot replace cases; the existing reconciliation trace
    describes which selected cases were materialized.
    """

    original = copy.deepcopy(dict(candidate_plan or {}))
    original_selected = [
        copy.deepcopy(dict(row))
        for row in list(original.get("selected_candidates") or [])
        if isinstance(row, Mapping)
    ]
    accepted = [copy.deepcopy(dict(row)) for row in list(accepted_cases or []) if isinstance(row, Mapping)]
    rejected = [copy.deepcopy(dict(row)) for row in list(rejected_cases or []) if isinstance(row, Mapping)]

    frozen_union = _is_frozen_predeclared_execution_union(original)
    frozen_identities = (
        _verify_frozen_predeclared_candidate_bindings(
            original,
            prediction,
            candidate_universe_path=candidate_universe_path,
            prediction_path=prediction_path,
            selection_input_path=selection_input_path,
            model_path=model_path,
            prediction_freeze_manifest_path=prediction_freeze_manifest_path,
            expected_selection_policy=expected_selection_policy,
            expected_selection_strategy=expected_selection_strategy,
            expected_native_single_part2_input=(
                expected_native_single_part2_input
            ),
        )
        if frozen_union
        else _deduplicated_generation_identities(
            original_selected, label="selected_candidates", allow_exact_duplicates=False,
        )
    )
    requested_cases = len(frozen_identities)
    if not frozen_union and backend_selection_state and backend_selection_state.get('enabled'):
        raise ValueError("Selected Generic cohort cannot enable backend backfill.")
    if len(accepted) > requested_cases:
        raise ValueError(
            "Generator returned more accepted cases than the authoritative "
            f"candidate plan selected ({len(accepted)} > {requested_cases})."
        )
    accepted_identity_rows: List[tuple[str, int]] = []
    seen_case_ids: set[str] = set()
    seen_boundaries: set[int] = set()
    for idx, accepted_row in enumerate(accepted, start=1):
        case_id, boundary = _generation_case_identity(accepted_row, idx)
        if not case_id or boundary is None:
            raise ValueError(
                "Generator accepted case is missing a canonical case_id/boundary identity."
            )
        if case_id in seen_case_ids or int(boundary) in seen_boundaries:
            raise ValueError(
                "Generator returned duplicate accepted case identities: "
                f"case_id={case_id!r} boundary={boundary!r}."
            )
        seen_case_ids.add(case_id)
        seen_boundaries.add(int(boundary))
        accepted_identity_rows.append((case_id, int(boundary)))

    if frozen_union or original_selected:
        # Do not return before this attempt ledger has proved that every
        # planned candidate was explicitly classified.  In particular,
        # accepted=0 is a valid precursor to the backend-agnostic direct
        # fallback only when the generator recorded the whole frozen union as
        # rejected/unsupported; a partial/global abort must remain fail-closed.
        expected_ids = [case_id for case_id, _boundary in frozen_identities]
        expected_set = set(frozen_identities)

        rejected_identity_rows: List[tuple[str, int]] = []
        rejected_case_ids_seen: set[str] = set()
        rejected_boundaries_seen: set[int] = set()
        for idx, rejected_row in enumerate(rejected, start=1):
            case_id, boundary = _generation_case_identity(rejected_row, idx)
            if not case_id or boundary is None:
                raise ValueError(
                    "Generator rejected case is missing a canonical "
                    "case_id/boundary identity."
                )
            canonical_boundary = int(boundary)
            if (
                case_id in rejected_case_ids_seen
                or canonical_boundary in rejected_boundaries_seen
            ):
                raise ValueError(
                    "Generator returned duplicate rejected case identities: "
                    f"case_id={case_id!r} boundary={canonical_boundary!r}."
                )
            if case_id in seen_case_ids or canonical_boundary in seen_boundaries:
                raise ValueError(
                    "Generator classified a frozen candidate as both accepted "
                    "and rejected: "
                    f"case_id={case_id!r} boundary={canonical_boundary!r}."
                )
            rejected_case_ids_seen.add(case_id)
            rejected_boundaries_seen.add(canonical_boundary)
            rejected_identity_rows.append((case_id, canonical_boundary))

        accepted_set = set(accepted_identity_rows)
        rejected_set = set(rejected_identity_rows)
        classified_set = accepted_set | rejected_set
        unexpected_identities = [
            identity
            for identity in accepted_identity_rows + rejected_identity_rows
            if identity not in expected_set
        ]
        missing_identities = [
            identity
            for identity in frozen_identities
            if identity not in classified_set
        ]
        if missing_identities or unexpected_identities:
            raise ValueError(
                "Frozen predeclared execution union was not fully materialized "
                "or explicitly rejected: "
                f"expected={expected_ids!r} "
                f"accepted={[case_id for case_id, _ in accepted_identity_rows]!r} "
                f"rejected={[case_id for case_id, _ in rejected_identity_rows]!r} "
                f"missing={[case_id for case_id, _ in missing_identities]!r} "
                f"unexpected={[case_id for case_id, _ in unexpected_identities]!r}."
            )

        # generation_state stores accepted and rejected rows separately.  Each
        # list must therefore be the exact corresponding subsequence of the
        # prospectively frozen attempt order.  Together with the disjoint/full
        # partition above this proves the global deterministic attempt ledger
        # without permitting a replacement outside the frozen union.
        expected_accepted_order = [
            identity for identity in frozen_identities if identity in accepted_set
        ]
        expected_rejected_order = [
            identity for identity in frozen_identities if identity in rejected_set
        ]
        if accepted_identity_rows != expected_accepted_order:
            raise ValueError(
                "Frozen accepted cases do not follow the declared deterministic "
                "attempt order: "
                f"expected={[case_id for case_id, _ in expected_accepted_order]!r} "
                f"actual={[case_id for case_id, _ in accepted_identity_rows]!r}."
            )
        if rejected_identity_rows != expected_rejected_order:
            raise ValueError(
                "Frozen rejected cases do not follow the declared deterministic "
                "attempt order: "
                f"expected={[case_id for case_id, _ in expected_rejected_order]!r} "
                f"actual={[case_id for case_id, _ in rejected_identity_rows]!r}."
            )

        accepted_ids_exact = [
            case_id for case_id, _boundary in accepted_identity_rows
        ]
        rejected_ids_exact = [
            case_id for case_id, _boundary in rejected_identity_rows
        ]
        attempt_ledger = [
            {
                "case_id": case_id,
                "boundary": boundary,
                "generation_status": (
                    "accepted" if (case_id, boundary) in accepted_set else "rejected"
                ),
            }
            for case_id, boundary in frozen_identities
        ]
        if not frozen_union:
            return original, {
                "status": ("no_accepted_cases_pending_direct_fallback" if not accepted
                           else "selected_generic_cases_materialized" if not rejected
                           else "selected_generic_cases_with_rejections"),
                "changed": False,
                "pre_generation_case_ids": expected_ids,
                "final_case_ids": expected_ids,
                "accepted_case_ids": accepted_ids_exact,
                "rejected_case_ids": rejected_ids_exact,
                "attempted_case_ids": expected_ids,
                "attempt_ledger": attempt_ledger,
                "removed_case_ids": [], "backfilled_case_ids": [],
                "materialized_count": len(accepted_ids_exact),
                "rejected_count": len(rejected_ids_exact),
                "claim_eligible": not rejected,
            }
        if not accepted:
            trace = {
                "status": "frozen_union_no_accepted_cases_pending_direct_fallback",
                "changed": False,
                "pre_generation_case_ids": expected_ids,
                "final_case_ids": [],
                "accepted_case_ids": [],
                "rejected_case_ids": rejected_ids_exact,
                "attempted_case_ids": expected_ids,
                "attempt_ledger": attempt_ledger,
                "removed_case_ids": [],
                "backfilled_case_ids": [],
                "execution_union_count": len(expected_ids),
                "materialized_count": 0,
                "rejected_count": len(rejected_ids_exact),
                "claim_eligible": False,
            }
            trace.update(
                _post_build_audit_minimum_projection(
                    original,
                    [],
                    pending_direct_fallback=True,
                )
            )
            return original, trace

        status = (
            "matches_frozen_predeclared_execution_union"
            if not rejected_identity_rows
            else "frozen_predeclared_execution_union_with_recorded_rejections"
        )
        trace = {
            "status": status,
            "changed": False,
            "pre_generation_case_ids": expected_ids,
            "final_case_ids": accepted_ids_exact,
            "accepted_case_ids": accepted_ids_exact,
            "rejected_case_ids": rejected_ids_exact,
            "attempted_case_ids": expected_ids,
            "attempt_ledger": attempt_ledger,
            "removed_case_ids": [],
            "backfilled_case_ids": [],
            "execution_union_count": len(expected_ids),
            "materialized_count": len(accepted_ids_exact),
            "rejected_count": len(rejected_ids_exact),
            "claim_eligible": True,
        }
        minimum_projection = _post_build_audit_minimum_projection(
            original,
            accepted_identity_rows,
        )
        trace.update(minimum_projection)
        if minimum_projection.get("post_build_minimum_status") == "shortfall":
            trace["claim_eligible"] = False
        return original, trace

    return original, {
        "status": "empty_selected_generic_cohort", "changed": False,
        "pre_generation_case_ids": [], "final_case_ids": [],
        "removed_case_ids": [], "backfilled_case_ids": [],
    }


def write_reconciled_candidate_plan_mirrors(
    model_dir: Path,
    candidate_plan: Mapping[str, Any],
) -> Dict[str, Path]:
    """Project one authoritative post-generation plan identically to both paths."""

    analysis_path = write_json(Path(model_dir) / "analysis" / "final_candidate_plan.json", candidate_plan)
    benchmark_path = write_json(Path(model_dir) / "benchmark_set" / "final_candidate_plan.json", candidate_plan)
    return {
        "final_candidate_plan_json": analysis_path,
        "benchmark_final_candidate_plan_json": benchmark_path,
    }


@dataclass
class LegacyBenchmarkSetResult:
    artifacts: Dict[str, Path]
    metrics: Dict[str, Any]
    message: str
    status: str
    suite_dir: Path


def materialize_legacy_benchmark_set(
    *,
    model_id: str,
    model_path: str,
    model_dir: Path,
    run_dir: Path,
    profile_id: str,
    run_id: str,
    prediction: Mapping[str, Any],
    candidate_plan: Mapping[str, Any],
    targets: Sequence[str],
    profile_payload: Mapping[str, Any],
    row: Mapping[str, Any],
    options: Any,
    log: Optional[Callable[[str], None]] = None,
    process_registry: Any = None,
    cancel_event: Any = None,
) -> LegacyBenchmarkSetResult:
    """Generate a BenchmarkSet by delegating to the existing service stack.

    The generated suite lives in ``models/<model>/benchmark_set/legacy_suite``.
    Formal files in ``models/<model>/benchmark_set`` point to it so the later
    run_benchmarks stage can use the same benchmark_suite.py and remote service
    as the Benchmark tab.
    """
    validate_profile_config_booleans(profile_payload)
    hailo_profile_force = parse_config_bool(
        profile_payload.get("hailo_build", {}).get("force_build", False),
        field="hailo_build.force_build",
    )
    hailo_option_force = parse_config_bool(
        getattr(options, "hailo_force_build", False), field="options.hailo_force_build",
    )

    def _log(msg: str) -> None:
        if callable(log):
            try:
                log(f"[benchmarkset:{model_id}] {msg}")
            except Exception:
                pass

    active_process_registry = process_registry or current_process_registry()

    quality_canary = resolve_full_only_quality_canary(
        profile_payload,
        plan_rows=[
            dict(run) for run in enabled_run_profiles(
                profile_payload.get("run_profiles")
            ) if isinstance(run, Mapping)
        ],
    )
    if quality_canary.get("enabled") and quality_canary.get("ok") is not True:
        raise ValueError(
            "full_only_quality_canary_invalid:"
            + ",".join(
                str(value) for value in quality_canary.get("errors") or []
            )
        )

    def _cancel_requested() -> bool:
        try:
            if cancel_event is not None and cancel_event.is_set():
                return True
        except Exception:
            pass
        return bool(
            active_process_registry is not None
            and getattr(active_process_registry, "cancelled", False)
        )

    from ..benchmark.services import (
        BenchmarkGenerationExecutionCallbacks,
        BenchmarkGenerationExecutionConfig,
        BenchmarkGenerationExecutionService,
        BenchmarkGenerationOrchestrationConfig,
        BenchmarkGenerationOrchestrationService,
        BenchmarkGenerationService,
        _hailo_feasibility_file_sha256_v2783,
        _revalidate_hailo_feasibility_anchor_v2783,
        _validate_hailo_feasibility_resume_state_v2783,
        normalize_hailo_feasibility_control,
        normalize_full_hef_policy,
        normalize_hailo_full_model_preflight_policy,
    )
    from ..gui.benchmark_workflow import resolve_hailo_benchmark_helpers, resolve_tool_core_version, _materialize_manual_deepx_part1_artifacts
    from ..gui.controller import write_benchmark_suite_script
    from ..resources_utils import copy_resource_tree

    p_model = Path(str(model_path or "")).expanduser()
    formal_bdir = model_dir / "benchmark_set"
    # The BenchmarkSet path is the source of truth in v49p.  Remove stale
    # reduced/direct-suite folders from v49c-v49m so downstream stages do not
    # accidentally execute the old mini-suite instead of the mature generator.
    stale_direct_suite = formal_bdir / "generated_suite"
    if stale_direct_suite.exists() and not bool(getattr(options, "resume", False)):
        shutil.rmtree(stale_direct_suite, ignore_errors=True)
    stale_suite = formal_bdir / "suite"
    if stale_suite.exists() and not bool(getattr(options, "resume", False)):
        shutil.rmtree(stale_suite, ignore_errors=True)
    suite_dir = formal_bdir / "legacy_suite"
    if suite_dir.exists() and not bool(getattr(options, "resume", False)):
        shutil.rmtree(suite_dir)
    suite_dir.mkdir(parents=True, exist_ok=True)
    bench_log_path = suite_dir / "benchmark_generation.log"
    archived_prediction_path = model_dir / "analysis" / "prediction.json"
    candidate_universe_manifest_path = (
        model_dir / "analysis" / "candidate_universe_manifest.json"
    )
    selection_input_path = model_dir / "analysis" / "selection_input.json"
    prediction_freeze_manifest_path = (
        model_dir / "analysis" / "prediction_freeze_manifest.json"
    )
    policy = _selection_policy_from_profile(profile_payload)
    expected_selection_strategy = _selection_strategy_from_profile(
        profile_payload,
        policy,
    )
    native_cfg = _native_capability_config(profile_payload, options)
    native_single_part2_input = _native_split_requires_single_part2_input(
        native_cfg
    )

    (
        ranked_candidates,
        candidate_search_pool,
        requested,
        exact_candidate_scope,
    ) = _resolve_generation_candidate_scope(
        candidate_plan,
        prediction,
        candidate_universe_path=candidate_universe_manifest_path,
        prediction_path=archived_prediction_path,
        selection_input_path=selection_input_path,
        model_path=p_model,
        prediction_freeze_manifest_path=prediction_freeze_manifest_path,
        expected_selection_policy=policy,
        expected_selection_strategy=expected_selection_strategy,
        expected_native_single_part2_input=native_single_part2_input,
    )
    (
        cache_guard,
        cache_verify_exact_cases,
        ranked_candidates,
        candidate_search_pool,
        requested,
    ) = _cache_verify_candidate_scope(
        profile_payload=profile_payload,
        model_id=model_id,
        candidate_plan=candidate_plan,
        ranked_candidates=ranked_candidates,
        candidate_search_pool=candidate_search_pool,
        requested=requested,
    )
    if cache_guard:
        # The general BenchmarkSet generator normally backfills from the full
        # prediction pool until it has enough accepted cases.  A cache canary
        # instead inspects only the attested cases and terminates on an exact
        # miss.
        _log(
            "[cache-verify] exact candidate scope="
            f"{cache_verify_exact_cases}; alternate candidate search disabled"
        )
    frozen_predeclared_union = _is_frozen_predeclared_execution_union(
        candidate_plan
    )
    if frozen_predeclared_union:
        exact_boundaries = [
            boundary
            for _case_id, boundary in _predeclared_execution_union_identities(
                candidate_plan
            )
        ]
        if (
            list(ranked_candidates) != exact_boundaries
            or list(candidate_search_pool) != exact_boundaries
            or requested != len(exact_boundaries)
        ):
            raise ValueError(
                "Frozen predeclared execution union was altered by a secondary "
                "generation scope policy."
            )
    elif exact_candidate_scope:
        exact_boundaries = [
            int(boundary)
            for index, row in enumerate(
                list(candidate_plan.get("selected_candidates") or []),
                start=1,
            )
            if isinstance(row, Mapping)
            for _case_id, boundary in [_generation_case_identity(row, index)]
            if boundary is not None
        ]
        if (
            not exact_boundaries
            or list(ranked_candidates) != exact_boundaries
            or list(candidate_search_pool) != exact_boundaries
            or requested < len(exact_boundaries)
        ):
            raise ValueError(
                "Forced candidate scope was altered by a secondary "
                "generation scope policy."
            )
    require_single_part2_input = _safe_bool(
        policy.get("require_single_part2_input"), False
    )
    gap = _safe_int(policy.get("min_gap"), _safe_int(candidate_plan.get("min_gap"), 0))
    selection_strategy = str(policy.get("selection_strategy") or policy.get("coverage_strategy") or candidate_plan.get("selection_strategy") or "").strip().lower().replace("-", "_")
    preserve_stratified_order = selection_strategy in {"stratified", "stratified_windows", "windowed", "coverage_windows"}
    preserve_exact_scope_order = bool(exact_candidate_scope)
    if preserve_exact_scope_order:
        # The freeze already applied selection/capability policy.  Reapplying
        # min-gap in the legacy case loop can silently drop adjacent frozen
        # audit boundaries (for example b118/b119).
        gap = 0
    base = p_model.stem or model_id
    pad_source = candidate_search_pool or ranked_candidates or [0]
    pad = max(3, len(str(max([abs(int(x)) for x in pad_source] or [0]))))

    _log(f"delegating generation to existing BenchmarkGenerationOrchestrationService; requested={requested}, ranked={ranked_candidates[:10]}, pool={len(candidate_search_pool)}")

    try:
        from ..core_analysis import analyze_model
        analysis_payload = analyze_model(str(p_model), min_gap=0)
    except Exception as exc:
        raise RuntimeError(f"legacy generator requires a parseable ONNX model: {type(exc).__name__}: {exc}") from exc
    model = analysis_payload.get("model") if isinstance(analysis_payload, Mapping) else None
    nodes = analysis_payload.get("nodes") if isinstance(analysis_payload, Mapping) else None
    order = analysis_payload.get("order") if isinstance(analysis_payload, Mapping) else None
    if model is None or not isinstance(nodes, list) or not isinstance(order, list):
        raise RuntimeError("legacy generator analysis payload does not contain model/nodes/order")

    targets_eff, explicit_cpu_reference, central_cpu_reference = (
        _bind_management_reference_targets_v27519(
            profile_payload,
            _profile_targets(profile_payload, targets),
            cache_verify_enabled=bool(cache_guard),
        )
    )
    switches = _infer_run_switches(profile_payload, targets_eff)
    # The management reference is an internal producer recipe, never a logical
    # deployment target.  Force it at the generator switch boundary, and force
    # it off for the exact cache canary even when legacy ``cpu_full`` text is
    # present in a mixed profile.
    switches = _bind_management_reference_cpu_switch_v27519(
        switches,
        required=central_cpu_reference,
        cache_verify_enabled=bool(cache_guard),
    )
    validation_defaults = _validation_defaults(profile_payload, row, model_id)

    # v55l: Evaluation runs must use the same tool-wide calibration defaults as
    # the manual Benchmark tab.  An empty hailo_build.calib_dir used to fall
    # through to the Hailo helper which generated random calibration tensors;
    # that made Hailo full/split rows run but fail semantically.  Resolve a
    # concrete image directory here so HEF builds use Imagenette-500 for
    # classifiers and COCO-200 for detectors unless the profile explicitly
    # overrides calib_dir.
    def _looks_like_image_dir_local(path: Path) -> bool:
        try:
            if not path.exists() or not path.is_dir():
                return False
            exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.npy', '.npz'}
            return any(p.is_file() and p.suffix.lower() in exts for p in path.rglob('*'))
        except Exception:
            return False

    def _preset_image_dir_local(name: str, *, task: str = '') -> str:
        raw = str(name or '').strip().lower().replace('-', '_')
        root = Path(os.environ.get('ONNX_SPLITPOINT_TOOL_VALIDATION_DATASETS') or (Path.home() / '.onnx_splitpoint_tool' / 'validation_datasets')).expanduser()
        candidates: list[Path] = []
        if raw in {'', 'auto', 'classification_calib', 'imagenette_val_mini_500', 'imagenette500', 'imagenet_val_mini_500', 'imagenet500'}:
            candidates.extend([root / 'classification' / 'imagenette_val_mini_500' / 'images', root / 'classification' / 'imagenette_val_mini_500'])
        if raw in {'imagenette_val_mini_200', 'imagenette200', 'imagenet_val_mini_200', 'imagenet200'}:
            candidates.extend([root / 'classification' / 'imagenette_val_mini_200' / 'images', root / 'classification' / 'imagenette_val_mini_200'])
        if raw in {'', 'auto', 'detection_calib', 'coco_200', 'coco200', 'coco_200_data'}:
            candidates.append(root / 'detection' / 'coco_200_data')
        if raw in {'coco_50', 'coco50', 'coco_50_data'}:
            candidates.append(root / 'detection' / 'coco_50_data')
        if name and raw not in {'auto'}:
            candidates.insert(0, Path(os.path.expanduser(str(name))))
        # Task-specific order: if no explicit name was given, do not let the
        # generic candidate order pick classification images for detection.
        if not str(name or '').strip() or raw == 'auto':
            if str(task or '').lower() == 'detection':
                candidates = [root / 'detection' / 'coco_200_data', root / 'detection' / 'coco_50_data']
            elif str(task or '').lower() == 'classification':
                candidates = [root / 'classification' / 'imagenette_val_mini_500' / 'images', root / 'classification' / 'imagenette_val_mini_500', root / 'classification' / 'imagenette_val_mini_200' / 'images', root / 'classification' / 'imagenette_val_mini_200']
        for cand in candidates:
            if cand.is_file() and cand.name == 'manifest.json':
                img = cand.parent / 'images'
                if _looks_like_image_dir_local(img):
                    return str(img.resolve())
                if _looks_like_image_dir_local(cand.parent):
                    return str(cand.parent.resolve())
            if _looks_like_image_dir_local(cand):
                img = cand / 'images'
                if _looks_like_image_dir_local(img):
                    return str(img.resolve())
                return str(cand.resolve())
        return ''

    task_hint_for_calib = str(validation_defaults.get('benchmark_task') or _model_task(row, model_id) or 'auto').lower()
    explicit_hailo_calib = str((profile_payload.get('hailo_build') if isinstance(profile_payload.get('hailo_build'), Mapping) else {}).get('calib_dir') or getattr(options, 'hailo_calib_dir', '') or '').strip()
    calibration_manifest = calibration_manifest_for_task(profile_payload, task_hint_for_calib)
    manifest_calib_dir = manifest_dataset_root(calibration_manifest) if calibration_manifest else ''
    if explicit_hailo_calib:
        effective_hailo_calib_dir = _preset_image_dir_local(explicit_hailo_calib, task=task_hint_for_calib)
        calibration_source_kind = 'explicit_profile_or_cli'
    elif manifest_calib_dir:
        effective_hailo_calib_dir = _preset_image_dir_local(manifest_calib_dir, task=task_hint_for_calib)
        calibration_source_kind = 'content_addressed_manifest'
    else:
        effective_hailo_calib_dir = _preset_image_dir_local('', task=task_hint_for_calib)
        calibration_source_kind = 'legacy_screening_fallback' if effective_hailo_calib_dir else 'unresolved'
    if effective_hailo_calib_dir:
        _log(
            f"[dataset-binding] effective accelerator calibration dir={effective_hailo_calib_dir} "
            f"task={task_hint_for_calib} source={calibration_source_kind} "
            f"manifest={calibration_manifest or '-'}"
        )
    elif explicit_hailo_calib:
        _log(f"[dataset-binding] warning: explicit Hailo calibration dir not usable: {explicit_hailo_calib}")
    else:
        _log(f"[dataset-binding] warning: no calibration directory resolved for task={task_hint_for_calib}; accelerator build may fall back to random calibration")

    generation_service = BenchmarkGenerationService()
    validate_profile_config_booleans(profile_payload)
    hailo_build_cfg = profile_payload.get("hailo_build") if isinstance(profile_payload.get("hailo_build"), Mapping) else {}
    hailo_force_requested = hailo_profile_force or hailo_option_force
    hailo_feasibility_control = normalize_hailo_feasibility_control(
        hailo_build_cfg.get("feasibility_control")
        if isinstance(hailo_build_cfg, Mapping)
        else None
    )
    from ..backend_backfill import backfill_policy, bind_plan_cases, selected_cases_for_backend, selection_contracts
    backend_backfill_policy = ({} if frozen_predeclared_union or exact_candidate_scope or cache_guard
                               else backfill_policy(profile_payload))
    defer_hailo_builds = bool(
        hailo_build_cfg.get("defer_until_cache_preflight")
        and not cache_verify_guard(profile_payload)
    )
    defer_deepx_builds = defer_hailo_builds
    selection_probe_preflight = bool(
        defer_hailo_builds and (hailo_feasibility_control.get("enabled") or backend_backfill_policy)
    )
    if selection_probe_preflight:
        # Gate-A's accepted boundary depends on measured compiler feasibility.
        # Its own cache probes receive an explicit provisional preflight before
        # cold dispatch; the complete final matrix follows the frozen result.
        defer_hailo_builds = False
    os.environ["ONNX_SPLITPOINT_HAILO_CALIBRATION_STORAGE"] = str(hailo_build_cfg.get("calibration_storage") or "auto")
    native_full_requested = _native_full_requested(native_cfg)
    configured_build_full = _safe_bool(hailo_build_cfg.get("build_full"), bool(getattr(options, "hailo_build_full", True)))
    configured_build_part1 = _safe_bool(hailo_build_cfg.get("build_part1"), bool(getattr(options, "hailo_build_part1", True)))
    configured_build_part2 = _safe_bool(hailo_build_cfg.get("build_part2"), bool(getattr(options, "hailo_build_part2", True)))
    if bool(hailo_feasibility_control.get("enabled")) and (
        configured_build_full
        or not configured_build_part1
        or configured_build_part2
    ):
        raise ValueError(
            "Hailo8-first Gate-A requires build_full=false, "
            "build_part1=true, build_part2=false"
        )
    # Only build the Hailo directions that the selected logical hardware paths
    # can execute.  Earlier versions built Part2 even for Hailo->TensorRT-only
    # profiles, which added substantial compiler time without producing a row.
    build_full = bool(configured_build_full and (switches.get("hailo_full_requested") or (native_full_requested and (switches.get("acc_h8") or switches.get("acc_h10")))))
    build_part1 = bool(configured_build_part1 and switches.get("hailo_part1_requested"))
    build_part2 = bool(configured_build_part2 and switches.get("hailo_part2_requested"))
    full_hef_policy = _hailo_full_hef_policy_v2772(build_full)
    _log(
        f"[build-plan] Hailo full={build_full} part1={build_part1} part2={build_part2} "
        f"(native_full={native_full_requested}, h2trt={bool(switches.get('matrix_hailo_to_trt'))}, trt2h={bool(switches.get('matrix_trt_to_hailo'))})"
    )
    # Request the same variants the Benchmark tab uses for complete suites.
    # Composed must be true whenever a heterogeneous Hailo->host/TRT pipeline is
    # requested; otherwise the runner can only measure isolated components.
    build_composed = bool(build_part1 or build_part2 or switches.get("matrix_hailo_to_trt") or switches.get("matrix_trt_to_hailo") or switches.get("matrix_deepx_to_trt") or switches.get("matrix_trt_to_deepx"))
    # v60r: once a Run Mode is materialised, its task budget is the source of
    # truth.  Do not let the old WorkflowOptions/GUI default of 50 images win.
    if bool(validation_defaults.get("validation_budget_authoritative")):
        run_plan_validation_max = _safe_int(validation_defaults.get("validation_max_images"), 0)
    else:
        run_plan_validation_max = _safe_int(
            getattr(options, "remote_validation_max_images", 0)
            if int(getattr(options, "remote_validation_max_images", 0) or 0) > 0
            else validation_defaults.get("validation_max_images", 0),
            0,
        )

    if bool(validation_defaults.get("validation_budget_authoritative")):
        run_plan_validation_images = str(validation_defaults.get("validation_images") or "")
    else:
        run_plan_validation_images = str(_effective_str(getattr(options, "remote_validation_images", "")) or validation_defaults.get("validation_images") or "")

    run_plan = generation_service.build_run_plan(
        acc_cpu=bool(switches.get("acc_cpu")),
        acc_cuda=bool(switches.get("acc_cuda")),
        acc_trt=bool(switches.get("acc_trt")),
        acc_h8=bool(switches.get("acc_h8")),
        acc_h10=bool(switches.get("acc_h10")),
        acc_deepx=bool(switches.get("acc_deepx")),
        hailo8_hw=str(getattr(options, "hailo_hw_arch", "hailo8") or "hailo8"),
        hailo10_hw="hailo10" if bool(switches.get("acc_h10")) else "",
        image_scale=str(validation_defaults.get("image_scale") or "auto"),
        validation_images=run_plan_validation_images,
        validation_max_images=run_plan_validation_max,
        validation_reference_mode=str(_effective_str(getattr(options, "remote_validation_reference_mode", "")) or validation_defaults.get("validation_reference_mode") or "auto"),
        mini_coco_ap50=bool(getattr(options, "remote_mini_coco_ap50", False) or validation_defaults.get("mini_coco_ap50")),
        benchmark_task=str(_effective_str(getattr(options, "remote_benchmark_task", "")) or validation_defaults.get("benchmark_task") or "auto"),
        mini_classification_eval=bool(getattr(options, "remote_mini_classification_eval", False) or validation_defaults.get("mini_classification_eval")),
        hailo_preset="Custom",
        hailo_custom_full=bool(build_full),
        hailo_custom_composed=bool(build_composed),
        hailo_custom_part1=bool(build_part1),
        hailo_custom_part2=bool(build_part2),
        matrix_trt_to_hailo=bool(switches.get("matrix_trt_to_hailo")),
        matrix_hailo_to_trt=bool(switches.get("matrix_hailo_to_trt")),
        matrix_deepx_to_trt=bool(switches.get("matrix_deepx_to_trt")),
        matrix_trt_to_deepx=bool(switches.get("matrix_trt_to_deepx")),
        full_hef_policy=full_hef_policy,
        cache_verify_only=bool(cache_verify_guard(profile_payload)),
        cache_verify_hailo_variants=[
            variant
            for variant, enabled in (
                ("full", build_full),
                ("part1", build_part1),
                ("part2", build_part2),
                ("composed", build_composed),
            )
            if enabled
        ],
    )
    run_plan.bench_plan_runs = _apply_declared_run_variants(
        run_plan.bench_plan_runs, profile_payload,
    )
    # v60q: carry the authoritative content-addressed validation manifest into
    # every executable run.  ``validation_images`` is the dataset root, while
    # the manifest lets suite generation select 16/12 Smoke items without
    # traversing or copying the complete ImageNet/COCO population.
    for _run in list(run_plan.bench_plan_runs or []):
        if not isinstance(_run, dict):
            continue
        if validation_defaults.get("validation_manifest"):
            _run["validation_manifest"] = str(validation_defaults.get("validation_manifest") or "")
        _run["validation_source_kind"] = str(validation_defaults.get("validation_source_kind") or "")
        _run["validation_items_requested"] = int(validation_defaults.get("validation_items_requested") or 0)
        _run["validation_budget_authoritative"] = bool(validation_defaults.get("validation_budget_authoritative"))
        if bool(validation_defaults.get("validation_budget_authoritative")):
            _run["validation_max_images"] = int(validation_defaults.get("validation_max_images") or 0)
    if central_cpu_reference:
        run_plan.bench_plan_runs = bind_management_cpu_reference_runs(
            [
                dict(run)
                for run in list(run_plan.bench_plan_runs or [])
                if isinstance(run, Mapping)
            ],
            automatic=not explicit_cpu_reference,
            require_existing=True,
        )
    run_plan.bench_plan_runs = project_full_only_quality_plan_rows(
        profile_payload,
        [
            dict(run) for run in list(run_plan.bench_plan_runs or [])
            if isinstance(run, Mapping)
        ],
    )

    if backend_backfill_policy:
        from .required_run_scope import authoritative_run_descriptors, merge_authoritative_run_descriptors
        physical_scope = _read_json(run_dir / 'required_run_scope.json', {}) or {}
        descriptors = authoritative_run_descriptors(physical_scope, model_id=model_id)
        if descriptors:
            run_plan.bench_plan_runs = merge_authoritative_run_descriptors(
                {'runs': run_plan.bench_plan_runs}, descriptors)['runs']
        elif physical_scope.get('identity_mode') == 'physical_strict_v3':
            raise ValueError('backend_backfill_physical_descriptors_missing')

    # One missing Hailo backend is a backend-local result. Every selected
    # Generic case retains its independently executable paths.
    require_complete_hailo_matrix_per_case = (
        _require_complete_hailo_matrix_for_candidate_plan(candidate_plan)
    )
    if frozen_predeclared_union:
        _log(
            "formal score-independent audit: retaining backend-partial Hailo "
            "cases; per-backend terminal states remain authoritative"
        )

    try:
        plan = generation_service.prepare_generation_plan(
            dict(analysis_payload),
            list(ranked_candidates),
            list(candidate_search_pool),
            requested,
            strict_boundary=False,
            hailo_selected=bool(run_plan.hailo_selected),
            outlook_top_n=max(12, requested),
        )
        if preserve_stratified_order or preserve_exact_scope_order:
            # v59s: for evaluation campaigns, candidate diversity is deliberate.
            # Keep the workflow-selected early/mid/late windows as the suite input;
            # use prepare_generation_plan only for Hailo outlook diagnostics.
            if preserve_exact_scope_order:
                scope_label = (
                    "frozen predeclared execution union"
                    if frozen_predeclared_union
                    else "selected Generic candidate scope"
                )
                _log(
                    f"{scope_label}: preserving exact candidate order/scope; "
                    "Hailo rerank is diagnostic only"
                )
            else:
                _log(f"selection_strategy={selection_strategy}: preserving stratified candidate order; Hailo rerank is diagnostic only")
        else:
            ranked_candidates = list(plan.ranked_candidates)
            candidate_search_pool = list(plan.candidate_search_pool)
        hailo_compile_rank_meta = dict(plan.hailo_compile_rank_meta or {})
        hailo_outlook_summary = plan.hailo_outlook_summary
    except Exception as exc:
        _log(f"prepare_generation_plan fallback: {type(exc).__name__}: {exc}")
        hailo_compile_rank_meta = {}
        hailo_outlook_summary = None

    resume_requested = bool(getattr(options, "resume", False))
    resume_state_hint: Optional[Mapping[str, Any]] = None
    if resume_requested:
        resume_path = suite_dir / "generation_state.json"
        if bool(hailo_feasibility_control.get("enabled")):
            resume_state_hint = dict(_read_gate_resume_state_v2783(resume_path))
            source_sha = _hailo_feasibility_file_sha256_v2783(p_model)
            persisted_gate = resume_state_hint.get("hailo_feasibility_state")
            if not isinstance(persisted_gate, Mapping) or str(
                persisted_gate.get("full_source_onnx_sha256") or ""
            ) != source_sha:
                raise ValueError(
                    "Hailo feasibility resume full source identity changed"
                )
            portable_model = suite_dir / "models" / p_model.name
            if (
                _hailo_feasibility_file_sha256_v2783(portable_model)
                != source_sha
            ):
                raise ValueError(
                    "Hailo feasibility portable full model identity changed"
                )
            resume_state_hint["_strict_hailo_feasibility_resume"] = True
        else:
            raw_resume = _read_json(resume_path, default={})
            resume_state_hint = (
                dict(raw_resume) if isinstance(raw_resume, Mapping) else {}
            )

    runtime = generation_service.start_generation_runtime(
        out_dir=suite_dir,
        bench_log_path=bench_log_path,
        requested_cases=requested,
        ranked_candidates=ranked_candidates,
        candidate_search_pool=candidate_search_pool,
        hef_full_policy=full_hef_policy,
        model_name=base,
        model_source=str(p_model),
        require_single_part2_input=require_single_part2_input,
        resume_generation=resume_requested,
        resume_state_hint=resume_state_hint,
    )
    full_model_dst = generation_service.copy_portable_full_model(runtime, str(p_model), log_cb=lambda msg, **_kw: _log(str(msg)))
    _deepx_prefetch_handle = _v60s_start_deepx_prefetch(
        run_dir=run_dir,
        model_id=model_id,
        model_path=str(full_model_dst or p_model),
        model_row=row,
        task=task_hint_for_calib,
        profile_payload=profile_payload,
        targets=targets_eff,
        suite_dir=suite_dir,
        log=_log,
        hailo_backend=str(
            getattr(options, "hailo_build_backend", "auto") or "auto"
        ),
        process_registry=active_process_registry,
        cancel_event=cancel_event,
    )

    build_mode = str(hailo_build_cfg.get("mode") or getattr(options, "hailo_build_mode", "reuse_and_build_missing") or "reuse_and_build_missing").strip().lower().replace("-", "_")
    if build_mode == "reuse_build_missing":
        build_mode = "reuse_and_build_missing"
    # Reuse still needs the normal receipt-checked cache resolver. Disabling
    # that helper also hid existing Full/Part1 HEFs from every new suite.
    hailo_cache_only = build_mode in {"cache_verify_only", "reuse_only"}
    active_build_mode = build_mode not in {"disabled", "skip"}
    preset = profile_payload.get("execution_preset") if isinstance(profile_payload.get("execution_preset"), Mapping) else {}
    run_mode_id = str(preset.get("id") or "").strip().lower()
    cold_full_policy = str(hailo_build_cfg.get("full_baseline_cold_build_policy") or ("cache_or_defer" if run_mode_id == "smoke" else "build_missing")).strip().lower().replace("-", "_")
    native_full_cfg = native_cfg.get("full_baselines") if isinstance(native_cfg.get("full_baselines"), Mapping) else {}
    native_full_backends = {str(x).strip().lower() for x in list(native_full_cfg.get("backends") or []) if str(x).strip()}
    native_full_by_producer = native_full_cfg.get("backends_by_producer") if isinstance(native_full_cfg.get("backends_by_producer"), Mapping) else {}
    native_hailo_full_required = bool(
        native_cfg.get("enabled")
        and native_full_cfg.get("enabled", bool(native_full_cfg))
        and (
            {"hailo8", "hailo10", "hailo10h"}.intersection(native_full_backends)
            or any(
                backend in {"hailo8", "hailo10", "hailo10h"}
                for rows in native_full_by_producer.values()
                for backend in [str(x).strip().lower() for x in list(rows or [])]
            )
        )
    )
    if run_mode_id == "smoke" and native_hailo_full_required and not hailo_cache_only:
        # A visible Full checkbox plus Native Runner means the user explicitly
        # requested a real Native Full baseline.  Do not silently defer the HEF
        # and then produce a missing Native-Full row.  Use the dedicated cold
        # timeout while retaining cache reuse whenever an exact artefact exists.
        cold_full_policy = "build_missing"
        _log("[build-policy] explicit Native Full selection overrides Smoke cache_or_defer; missing Hailo Full HEFs will be built")
    hailo_full_cache_only = bool(
        hailo_cache_only
        or (
            run_mode_id == "smoke"
            and cold_full_policy in {"cache_or_defer", "reuse_only", "cache_only"}
            and not hailo_force_requested
        )
    )
    hailo_normal_timeout_s = parse_hailo_timeout_seconds(
        hailo_build_cfg.get(
            "timeout_s", getattr(options, "hailo_build_timeout_s", 3600)
        ),
        default=3600,
        label="hailo_build.timeout_s",
    )
    hailo_full_timeout_explicit = "cold_build_timeout_s" in hailo_build_cfg
    hailo_full_timeout_s = (
        parse_hailo_timeout_seconds(
            hailo_build_cfg.get("cold_build_timeout_s"),
            default=hailo_normal_timeout_s,
            label="hailo_build.cold_build_timeout_s",
        )
        if hailo_full_timeout_explicit
        else hailo_normal_timeout_s
    )
    if run_mode_id == "smoke" and bool(run_plan.hef_full) and not hailo_full_cache_only:
        _log(f"[build-policy] explicit Smoke cold Full build enabled; full_timeout_s={hailo_full_timeout_s}")
    elif bool(run_plan.hef_full) and hailo_full_timeout_explicit and hailo_full_timeout_s <= 0:
        _log(
            "[build-policy] Hailo Full hard timeout disabled by explicit profile; "
            "heartbeat and cooperative/manual abort remain enabled"
        )
    if hailo_cache_only:
        _log(
            f"[build-policy] {build_mode} permits exact cache restoration "
            "only; every miss terminates before DFC dispatch"
        )
    elif hailo_full_cache_only:
        _log("[build-policy] Smoke Hailo Full uses cache_or_defer; exact cache misses are reported without a long compiler run")
    helpers = resolve_hailo_benchmark_helpers(
        need_build=bool(active_build_mode and run_plan.hef_targets and (run_plan.hef_full or run_plan.hef_part1 or run_plan.hef_part2)),
        need_part2=bool(run_plan.hef_targets and run_plan.hef_part2),
    )
    if not active_build_mode and run_plan.hef_targets:
        helpers.hailo_build_unavailable = f"Hailo build mode is {build_mode}; artifact lookup/build is disabled."
    if helpers.hailo_build_unavailable:
        _log(helpers.hailo_build_unavailable)
    if helpers.hailo_part2_import_error:
        _log(f"hailo part2 precheck unavailable: {helpers.hailo_part2_import_error}")
    hailo_builder = helpers.hailo_build_hef_fn
    if selection_probe_preflight and hailo_builder is not None:
        hailo_builder = selection_preflight_builder(
            hailo_builder, model_id=model_id, profile_payload=profile_payload, log=_log
        )
    if defer_hailo_builds and hailo_builder is not None:
        hailo_builder = cache_preflight_builder(hailo_builder)
        _log("[cache-preflight] preparing selected split ONNX and exact cache probes; compiler dispatch follows the complete campaign matrix")
    if hailo_builder is not None:
        from ..build_dispatch_policy import bind_profile_hailo_builder
        # Outside the preflight wrapper so its persisted exact call includes
        # the frozen family selection, including across scheduler threads.
        hailo_builder = bind_profile_hailo_builder(hailo_builder, profile_payload)

    if cache_verify_guard(profile_payload):
        prep_baseline, full_end_nodes, full_endpoint_mode = {}, [], ""
    else:
        prep_baseline, full_end_nodes, full_endpoint_mode = (
            _prepare_full_baseline_info(p_model)
        )
    if bool(prep_baseline.get("ok")):
        _log(f"prepared full-Hailo baseline available: {prep_baseline.get('hef_path') or prep_baseline.get('artifact_path') or prep_baseline.get('compiled_hef')}")

    def _persist_generation_state(status: str = "running", current_boundary: Optional[int] = None) -> None:
        runtime.persist(status=status, current_boundary=current_boundary)

    def _queue_put(_item: tuple) -> None:
        return None

    execution_service = BenchmarkGenerationExecutionService(generation_service)
    _legacy_calib_fallback = _safe_int(hailo_build_cfg.get("calib_count", getattr(options, "hailo_calib_count", 64)), 64)
    _task_calib_count = _mode_task_item_count(
        profile_payload, task_hint_for_calib, kind="calibration_items", fallback=_legacy_calib_fallback
    )
    hailo_feasibility_evidence_lookup = (
        _make_hailo_feasibility_evidence_lookup_v2783(
            hailo_feasibility_control,
            suite_root=suite_dir,
            log=_log,
        )
        if bool(hailo_feasibility_control.get("enabled"))
        else None
    )
    from .hardware_matrix import normalize_hardware_targets
    execution_cfg = BenchmarkGenerationExecutionConfig(
        runtime=runtime,
        backend_backfill_policy=backend_backfill_policy,
        hailo_metadata_store_root=str((profile_payload.get('artifact_store') or {}).get('root') or ''),
        hailo_metadata_targets=(normalize_hardware_targets(profile_payload)
            if (backend_backfill_policy or {}).get('technical_output_contract_version') == 1
            and any(row['backend'] == 'hailo10h' and row['stage'] == 'part1'
                    for row in selection_contracts(run_plan.bench_plan_runs)) else []),
        target_cases=int(requested),
        gap=int(gap),
        ranked_candidates=list(ranked_candidates),
        candidate_search_pool=list(candidate_search_pool),
        out_dir=suite_dir,
        base=base,
        pad=int(pad),
        strict_boundary=False,
        model=model,
        nodes=nodes,
        order=order,
        analysis_payload=analysis_payload,
        analysis_candidates=[dict(x) for x in list(prediction.get("candidates") or []) if isinstance(x, Mapping)],
        require_single_part2_input=require_single_part2_input,
        bench_plan_runs=list(run_plan.bench_plan_runs),
        benchmark_task=task_hint_for_calib,
        runner_target="auto",
        do_ctx_full=False,
        do_ctx_cutflow=False,
        ctx_hops=2,
        llm_style=False,
        value_bytes_map=analysis_payload.get("value_bytes") if isinstance(analysis_payload, Mapping) else None,
        full_model_src=str(p_model),
        full_model_dst=str(full_model_dst),
        tool_gui_version="workflow-legacy-adapter",
        tool_core_version=resolve_tool_core_version(),
        hailo_compile_rank_meta=dict(hailo_compile_rank_meta),
        hef_targets=list(run_plan.hef_targets),
        hef_part1=bool(run_plan.hef_part1),
        hef_part2=bool(run_plan.hef_part2),
        hef_backend=str(getattr(options, "hailo_build_backend", "auto") or "auto"),
        hef_fixup=True,
        hef_opt_level=_safe_int(hailo_build_cfg.get("optimization_level", getattr(options, "hailo_optimization_level", 1)), 1),
        hef_calib_dir=(effective_hailo_calib_dir or (str(hailo_build_cfg.get("calib_dir", getattr(options, "hailo_calib_dir", "")) or "") or None)),
        hef_calib_count=_task_calib_count,
        hef_calib_bs=_safe_int(hailo_build_cfg.get("calib_batch_size", getattr(options, "hailo_calib_batch_size", 8)), 8),
        hef_force=hailo_force_requested,
        hef_keep=_safe_bool(hailo_build_cfg.get("keep_artifacts"), bool(getattr(options, "hailo_keep_artifacts", False))),
        hef_wsl_distro=None,
        hef_wsl_venv="auto",
        hef_timeout_s=hailo_normal_timeout_s,
        hailo_full_cache_only=hailo_full_cache_only,
        hailo_cache_only=hailo_cache_only,
        defer_hailo_builds=defer_hailo_builds,
        defer_deepx_builds=defer_deepx_builds,
        hailo_full_timeout_s=hailo_full_timeout_s,
        hailo_full_timeout_explicit=hailo_full_timeout_explicit,
        hailo_run_mode=run_mode_id,
        build_scheduler_config=_profile_build_scheduler_config(profile_payload),
        hailo_build_hef_fn=hailo_builder,
        hailo_build_unavailable=helpers.hailo_build_unavailable,
        hailo_parse_check_fn=helpers.hailo_parse_check_fn,
        hailo_feasibility_control=dict(hailo_feasibility_control),
        hailo_feasibility_evidence_lookup=(
            hailo_feasibility_evidence_lookup
        ),
        hailo_full_end_node_names=list(full_end_nodes or []),
        hailo_full_endpoint_mode=str(full_endpoint_mode or ""),
        hailo_full_output_contract=(dict(prep_baseline.get("output_contract") or {}) if isinstance(prep_baseline.get("output_contract"), Mapping) else None),
        hailo_part2_precheck_fn=helpers.hailo_part2_precheck_fn,
        hailo_part2_precheck_error_fn=helpers.hailo_part2_precheck_error_fn,
        hailo_part2_parser_precheck_fn=helpers.hailo_part2_parser_precheck_fn,
        hailo_part2_parser_precheck_error_fn=helpers.hailo_part2_parser_precheck_error_fn,
        hailo_part2_enable_suggested_endnode_fallback=True,
        hailo_salvage_enable=not hailo_cache_only and not backend_backfill_policy,
        should_cancel=_cancel_requested,
    )
    callbacks = BenchmarkGenerationExecutionCallbacks(
        log=_benchmark_service_log_adapter(_log),
        queue_put=_queue_put,
        persist_state=lambda **kwargs: _persist_generation_state(**kwargs),
        publish_hailo_diagnostics=lambda label, result, log_cb=None: _log(f"hailo diagnostics: {label}"),
        predicted_metrics_for_boundary=lambda analysis, boundary: _analysis_prediction_metrics(analysis, int(boundary)),
        hailo_parse_entry_for_boundary=lambda analysis, boundary: _hailo_parse_entry_for_boundary(analysis, int(boundary)),
        hailo_parse_scalar_fields=lambda entry: _hailo_parse_scalar_fields(entry),
    )

    orchestration_service = BenchmarkGenerationOrchestrationService(generation_service, execution_service)
    # v55j: do not let a stale GUI/session strict-proxy flag make evaluation
    # campaigns reject all candidates.  Strict activation-proxy mode is only
    # honored when the evaluation profile explicitly opts into it.
    _old_proxy_strict = os.environ.get("ONNX_SPLITPOINT_ACTIVATION_PROXY_STRICT")
    _old_proxy_strict_alt = os.environ.get("SPLITPOINT_ACTIVATION_PROXY_STRICT")
    _env_keys_v60o = [
        "ONNX_SPLITPOINT_HAILO_CALIBRATION_STORAGE",
        "ONNX_SPLITPOINT_HAILO_CALIB_CAP_MB",
        "ONNX_SPLITPOINT_HAILO_CALIB_MANIFEST",
        "ONNX_SPLITPOINT_HAILO_CACHE_ENABLED",
        "ONNX_SPLITPOINT_HAILO_CACHE_ROOT",
        "ONNX_SPLITPOINT_HAILO_CACHE_INTEGRITY",
    ]
    _old_env_v60o = {key: os.environ.get(key) for key in _env_keys_v60o}
    os.environ["ONNX_SPLITPOINT_HAILO_CALIBRATION_STORAGE"] = str(hailo_build_cfg.get("calibration_storage") or "memory")
    os.environ["ONNX_SPLITPOINT_HAILO_CALIB_CAP_MB"] = str(_safe_int(hailo_build_cfg.get("calibration_memory_cap_mb"), 256))
    if calibration_manifest:
        os.environ["ONNX_SPLITPOINT_HAILO_CALIB_MANIFEST"] = str(calibration_manifest)
    else:
        os.environ.pop("ONNX_SPLITPOINT_HAILO_CALIB_MANIFEST", None)
    os.environ["ONNX_SPLITPOINT_HAILO_CACHE_ENABLED"] = "1" if _safe_bool(hailo_build_cfg.get("cache_enabled"), True) else "0"
    os.environ["ONNX_SPLITPOINT_HAILO_CACHE_ROOT"] = str(Path(str(hailo_build_cfg.get("cache_root") or "~/.cache/onnx_splitpoint/hailo_hef")).expanduser())
    os.environ["ONNX_SPLITPOINT_HAILO_CACHE_INTEGRITY"] = str(hailo_build_cfg.get("cache_integrity") or "relaxed")
    _log(
        f"[build-cache] Hailo calibration storage={os.environ['ONNX_SPLITPOINT_HAILO_CALIBRATION_STORAGE']} "
        f"requested={_task_calib_count} cache={'on' if os.environ['ONNX_SPLITPOINT_HAILO_CACHE_ENABLED']=='1' else 'off'}"
    )
    _eval_strict_proxy = _profile_activation_proxy_strict(profile_payload)
    if _eval_strict_proxy:
        os.environ["ONNX_SPLITPOINT_ACTIVATION_PROXY_STRICT"] = "1"
        _log("activation proxy strict mode explicitly enabled by evaluation profile")
    else:
        os.environ["ONNX_SPLITPOINT_ACTIVATION_PROXY_STRICT"] = "0"
        os.environ["SPLITPOINT_ACTIVATION_PROXY_STRICT"] = "0"
        _log("activation proxy strict mode disabled for evaluation campaign unless explicitly requested")

    cfg = BenchmarkGenerationOrchestrationConfig(
        runtime=runtime,
        execution_cfg=execution_cfg,
        execution_callbacks=callbacks,
        target_cases=int(requested),
        preferred_shortlist_original=list(ranked_candidates),
        ranked_candidates=list(ranked_candidates),
        candidate_search_pool=list(candidate_search_pool),
        out_dir=suite_dir,
        base=base,
        pad=int(pad),
        full_model_src=str(p_model),
        full_model_dst=str(full_model_dst),
        analysis_payload=analysis_payload,
        analysis_params_payload={
            "objective": "workflow_legacy_adapter",
            "topk": len(candidate_search_pool),
            "min_gap": gap,
            "require_single_part2_input": require_single_part2_input,
        },
        require_single_part2_input=require_single_part2_input,
        system_spec_payload=None,
        bench_log_path=str(bench_log_path),
        bench_plan_runs=list(run_plan.bench_plan_runs),
        hef_targets=list(run_plan.hef_targets),
        hef_full=bool(run_plan.hef_full),
        hef_part1=bool(run_plan.hef_part1),
        hef_part2=bool(run_plan.hef_part2),
        hef_backend=str(getattr(options, "hailo_build_backend", "auto") or "auto"),
        hef_fixup=True,
        hef_opt_level=_safe_int(hailo_build_cfg.get("optimization_level", getattr(options, "hailo_optimization_level", 1)), 1),
        hef_calib_dir=(effective_hailo_calib_dir or (str(hailo_build_cfg.get("calib_dir", getattr(options, "hailo_calib_dir", "")) or "") or None)),
        hef_calib_count=_task_calib_count,
        hef_calib_bs=_safe_int(hailo_build_cfg.get("calib_batch_size", getattr(options, "hailo_calib_batch_size", 8)), 8),
        hef_force=hailo_force_requested,
        hef_keep=_safe_bool(hailo_build_cfg.get("keep_artifacts"), bool(getattr(options, "hailo_keep_artifacts", False))),
        hef_wsl_distro=None,
        hef_wsl_venv="auto",
        hef_timeout_s=hailo_normal_timeout_s,
        hailo_full_cache_only=hailo_full_cache_only,
        hailo_cache_only=hailo_cache_only,
        defer_hailo_builds=defer_hailo_builds,
        defer_deepx_builds=defer_deepx_builds,
        hailo_full_timeout_s=hailo_full_timeout_s,
        hailo_full_timeout_explicit=hailo_full_timeout_explicit,
        hailo_run_mode=run_mode_id,
        full_hef_policy=normalize_full_hef_policy(full_hef_policy),
        full_model_preflight_policy=("skip" if defer_hailo_builds or hailo_cache_only else normalize_hailo_full_model_preflight_policy((policy or {}).get("full_model_hailo_preflight_policy") or "enabled")),
        hailo_full_end_node_names=list(full_end_nodes or []),
        hailo_full_endpoint_mode=str(full_endpoint_mode or ""),
        hailo_full_output_contract=(dict(prep_baseline.get("output_contract") or {}) if isinstance(prep_baseline.get("output_contract"), Mapping) else None),
        prepared_full_hailo_baseline=dict(prep_baseline or {}),
        hailo_build_hef_fn=hailo_builder,
        hailo_parse_check_fn=helpers.hailo_parse_check_fn,
        hailo_build_unavailable=helpers.hailo_build_unavailable,
        hailo_part2_precheck_fn=helpers.hailo_part2_precheck_fn,
        hailo_part2_precheck_error_fn=helpers.hailo_part2_precheck_error_fn,
        hailo_part2_parser_precheck_fn=helpers.hailo_part2_parser_precheck_fn,
        hailo_part2_parser_precheck_error_fn=helpers.hailo_part2_parser_precheck_error_fn,
        resume_generation=bool(getattr(options, "resume", False)),
        resume_report_summary_lines=[],
        hailo_selected=bool(run_plan.hailo_selected),
        hailo_outlook_summary=hailo_outlook_summary,
        write_harness_script=lambda dst_dir, bench_json_name="benchmark_set.json": write_benchmark_suite_script(dst_dir, bench_json_name=bench_json_name),
        copy_schema_tree=lambda: copy_resource_tree("resources", "schemas", dest=suite_dir / "schemas"),
        tool_gui_version="workflow-legacy-adapter",
        tool_core_version=resolve_tool_core_version(),
        evaluation_profile_meta={"profile_id": profile_id, "source": "evaluation_workflow"},
        benchmark_objective="workflow_legacy_adapter",
        require_complete_hailo_matrix_per_case=(
            require_complete_hailo_matrix_per_case
        ),
        should_cancel=_cancel_requested,
    )

    try:
        orch = orchestration_service.run(cfg)
    finally:
        if _old_proxy_strict is None:
            os.environ.pop("ONNX_SPLITPOINT_ACTIVATION_PROXY_STRICT", None)
        else:
            os.environ["ONNX_SPLITPOINT_ACTIVATION_PROXY_STRICT"] = _old_proxy_strict
        if _old_proxy_strict_alt is None:
            os.environ.pop("SPLITPOINT_ACTIVATION_PROXY_STRICT", None)
        else:
            os.environ["SPLITPOINT_ACTIVATION_PROXY_STRICT"] = _old_proxy_strict_alt
        for _key, _value in _old_env_v60o.items():
            if _value is None:
                os.environ.pop(_key, None)
            else:
                os.environ[_key] = _value
        try:
            runtime.close()
        except Exception:
            pass

    _deepx_prefetch_report = _v60s_finish_deepx_prefetch(
        _deepx_prefetch_handle,
        suite_dir=suite_dir,
        log=_log,
        process_registry=active_process_registry,
        cancel_event=cancel_event,
    )

    suite_payload = _read_json(suite_dir / "benchmark_set.json", default={}) or {}
    plan_payload = _read_json(suite_dir / "benchmark_plan.json", default={}) or {}
    gate_enabled = bool(hailo_feasibility_control.get("enabled"))
    if gate_enabled:
        state_payload = dict(
            _read_gate_resume_state_v2783(
                suite_dir / "generation_state.json"
            )
        )
    else:
        state_payload = _read_json(
            suite_dir / "generation_state.json", default={}
        ) or {}
    accepted = _cases_from_suite_payload(suite_payload) or list(getattr(orch, "bench_payload", {}).get("cases") or [])
    rejected = _rejected_from_generation_state(state_payload)
    generation_summary = dict(getattr(orch, "summary_data", {}) or {})
    feasibility_state = (
        dict(state_payload.get("hailo_feasibility_state") or {})
        if isinstance(state_payload.get("hailo_feasibility_state"), Mapping)
        else {}
    )
    feasibility_invalid_reason = ""
    if gate_enabled:
        try:
            feasibility_state = _validate_hailo_feasibility_resume_state_v2783(
                feasibility_state,
                control=hailo_feasibility_control,
                candidate_order=candidate_search_pool,
                targets=run_plan.hef_targets,
                backend=str(
                    getattr(options, "hailo_build_backend", "auto") or "auto"
                ),
                full_source_onnx_sha256=(
                    _hailo_feasibility_file_sha256_v2783(p_model)
                ),
            )
        except Exception as exc:
            feasibility_invalid_reason = (
                f"invalid_gate_state:{type(exc).__name__}:{exc}"
            )
            feasibility_state = dict(feasibility_state)
            feasibility_state["outcome"] = "EVIDENCE_CONFLICT"
            feasibility_state["exhaustion_reason"] = (
                "invalid_or_corrupt_terminal_gate_state"
            )
    feasibility_outcome = str(feasibility_state.get("outcome") or "").strip()
    if gate_enabled and feasibility_outcome == "ANCHOR_FOUND":
        accepted_boundaries = {
            int(row.get("boundary"))
            for row in accepted
            if isinstance(row, Mapping) and type(row.get("boundary")) is int
        }
        completed_values = {
            int(value)
            for value in list(state_payload.get("completed_boundaries") or [])
            if type(value) is int
        }
        if not _revalidate_hailo_feasibility_anchor_v2783(
            feasibility_state,
            out_dir=suite_dir,
            cases=[row for row in accepted if isinstance(row, Mapping)],
            completed_boundaries=completed_values,
            accepted_boundaries=accepted_boundaries,
            targets=run_plan.hef_targets,
        ):
            feasibility_invalid_reason = "anchor_material_revalidation_failed"
            feasibility_state["outcome"] = "EVIDENCE_CONFLICT"
            feasibility_state["exhaustion_reason"] = (
                "anchor_material_revalidation_failed"
            )
            feasibility_outcome = "EVIDENCE_CONFLICT"
    if gate_enabled and feasibility_outcome not in {
        "ANCHOR_FOUND",
        "CANARY_BUDGET_EXHAUSTED",
        "EVIDENCE_CONFLICT",
    }:
        feasibility_invalid_reason = (
            feasibility_invalid_reason or "nonterminal_gate_state_after_generation"
        )
        feasibility_state["outcome"] = "EVIDENCE_CONFLICT"
        feasibility_state["exhaustion_reason"] = (
            "nonterminal_gate_state_after_generation"
        )
        feasibility_outcome = "EVIDENCE_CONFLICT"
    feasibility_exhausted = gate_enabled and feasibility_outcome in {
        "CANARY_BUDGET_EXHAUSTED",
        "EVIDENCE_CONFLICT",
    }
    feasibility_stop_workflow = _hailo_feasibility_stop_workflow_v2783(
        feasibility_outcome,
        hailo_feasibility_control,
    )
    feasibility_receipt_path: Optional[Path] = None
    if gate_enabled:
        feasibility_receipt_path = write_json(
            formal_bdir / "hailo_feasibility_receipt.json",
            {
                "schema": "onnx-splitpoint/hailo8-first-feasibility-receipt/v1",
                "schema_version": 1,
                "model_id": model_id,
                "profile_id": profile_id,
                "run_id": run_id,
                "outcome": feasibility_outcome or "CANARY_BUDGET_EXHAUSTED",
                "fallback_allowed": False,
                "stop_workflow": feasibility_stop_workflow,
                "candidate_order": list(candidate_search_pool),
                "candidate_order_sha256": str(
                    feasibility_state.get("candidate_order_sha256") or ""
                ),
                "state": feasibility_state,
                "validation_error": feasibility_invalid_reason,
                "created_at": now_iso(),
            },
        )
    if cache_guard and not accepted:
        rejection_details = [
            str(
                row.get("detail")
                or row.get("error")
                or row.get("reason")
                or ""
            ).strip()
            for row in rejected
            if isinstance(row, Mapping)
        ]
        rejection_details = [value for value in rejection_details if value]
        detail = (
            rejection_details[0]
            if rejection_details
            else "the exact attested case produced no accepted BenchmarkSet row"
        )
        blocked_message = cache_miss_blocked_message(
            "hailo_dfc",
            f"exact cases {cache_verify_exact_cases!r} for {model_id} "
            f"could not restore a receipt-bound Part1 HEF ({detail})",
        )
        write_json(formal_bdir / "cache_verify_generation_result.json", {
            "schema": "onnx-splitpoint/cache-verify-generation-result",
            "schema_version": 1,
            "status": "cache_miss_blocked",
            "model_id": model_id,
            "expected_cases": list(cache_verify_exact_cases),
            "attempted_boundaries": list(candidate_search_pool),
            "compiler_dispatch_allowed": False,
            "fallback_allowed": False,
            "error": blocked_message,
            "rejected_cases": rejected,
            "created_at": now_iso(),
        })
        _log(f"[cache-verify] {blocked_message}")
        raise CacheVerifyPolicyError(blocked_message)
    consumed_candidate_plan_artifact_id = str(candidate_plan.get("artifact_id") or "")
    candidate_plan, candidate_plan_reconciliation = reconcile_candidate_plan_after_generation(
        candidate_plan,
        prediction=prediction,
        accepted_cases=accepted,
        rejected_cases=rejected,
        generation_summary=generation_summary,
        backend_selection_state=state_payload.get('backend_backfill'),
        candidate_universe_path=candidate_universe_manifest_path,
        prediction_path=archived_prediction_path,
        selection_input_path=selection_input_path,
        model_path=p_model,
        prediction_freeze_manifest_path=prediction_freeze_manifest_path,
        expected_selection_policy=policy,
        expected_selection_strategy=expected_selection_strategy,
        expected_native_single_part2_input=native_single_part2_input,
    )
    reconciled_plan_paths: Dict[str, Path] = {}
    if bool(candidate_plan_reconciliation.get("changed")):
        reconciled_plan_paths = write_reconciled_candidate_plan_mirrors(
            model_dir,
            candidate_plan,
        )
        _log(
            "authoritative candidate plan reconciled after generator backfill: "
            f"before={candidate_plan_reconciliation.get('pre_generation_case_ids')} "
            f"after={candidate_plan_reconciliation.get('final_case_ids')}"
        )

    # v55k: do not let a Hailo/DFC build failure veto all other backends.
    # The legacy BenchmarkSet generator treats Hailo feasibility as authoritative
    # for case acceptance.  In mixed Hailo/DeepX/TensorRT evaluation campaigns
    # this is too strict: if Hailo part builds fail because of a host DFC/CUDA
    # issue, TensorRT/ORT/DeepX-full evidence is still useful and should be
    # materialized.  When the mature legacy generator returns zero accepted
    # cases but the formal candidate plan has selected split points, fall back to
    # the backend-agnostic direct ONNX split exporter.  Hailo rows will later be
    # marked unavailable/missing-artifact if no HEF exists, but non-Hailo rows can
    # still be benchmarked.
    try:
        _selected_for_fallback = [dict(x) for x in list(candidate_plan.get("selected_candidates") or candidate_plan.get("candidates") or []) if isinstance(x, Mapping)]
    except Exception:
        _selected_for_fallback = []
    if not accepted and _selected_for_fallback and not gate_enabled and not backend_backfill_policy:
        try:
            from .generator_binding import materialize_suite_from_candidate_plan
            _log(
                "legacy generator accepted=0; falling back to backend-agnostic "
                "direct split materialization so non-Hailo runs remain executable"
            )
            direct_res = materialize_suite_from_candidate_plan(
                model_id=model_id,
                model_path=model_path,
                model_dir=model_dir,
                run_dir=run_dir,
                profile_id=profile_id,
                run_id=run_id,
                prediction=prediction,
                candidate_plan=candidate_plan,
                targets=targets_eff,
                full_baseline_plan={},
                output_contracts={},
                profile_payload=profile_payload,
                model_entry=row,
                dry_run=False,
                log=log,
            )
            if _is_frozen_predeclared_execution_union(candidate_plan):
                direct_identities = [
                    identity
                    for index, case in enumerate(
                        list(direct_res.accepted_cases or []),
                        start=1,
                    )
                    if isinstance(case, Mapping)
                    for identity in [_generation_case_identity(case, index)]
                    if identity[0] and identity[1] is not None
                ]
                direct_minimum = _post_build_audit_minimum_projection(
                    candidate_plan,
                    [
                        (case_id, int(boundary))
                        for case_id, boundary in direct_identities
                        if boundary is not None
                    ],
                )
                candidate_plan_reconciliation.update(direct_minimum)
                candidate_plan_reconciliation.update({
                    "status": (
                        "frozen_union_direct_fallback_materialized"
                        if direct_identities
                        else "frozen_union_direct_fallback_no_materialized_cases"
                    ),
                    "execution_status": direct_minimum.get(
                        "execution_status"
                    ),
                    "direct_fallback_materialized_count": len(
                        direct_identities
                    ),
                    "direct_fallback_is_backfill": False,
                    "final_case_ids": [
                        case_id for case_id, _boundary in direct_identities
                    ],
                    "claim_eligible": (
                        bool(direct_identities)
                        and direct_minimum.get("post_build_minimum_status")
                        != "shortfall"
                    ),
                })
                if not direct_identities:
                    candidate_plan_reconciliation.update({
                        "execution_status": "direct_fallback_no_materialized_cases",
                        "repair_required": True,
                        "repair_status": "required_no_automatic_backfill",
                        "repair_reason": "direct_fallback_materialized_zero_cases",
                    })
            # Keep a small breadcrumb from the failed legacy attempt next to the
            # direct-generation artifacts for auditability.
            fallback_evidence_path: Optional[Path] = None
            try:
                fallback_evidence_path = write_json(model_dir / "benchmark_set" / "legacy_hailo_veto_fallback.json", {
                    "schema": "onnx-splitpoint/legacy-hailo-veto-fallback",
                    "schema_version": 1,
                    "created_at": now_iso(),
                    "reason": "legacy_generator_accepted_zero_cases",
                    "legacy_suite_dir": relpath(suite_dir, run_dir),
                    "legacy_rejected_count": len(rejected),
                    "direct_suite_dir": relpath(direct_res.suite_dir or (model_dir / "benchmark_set" / "generated_suite"), run_dir),
                    "direct_accepted_count": len(direct_res.accepted_cases),
                    "direct_rejected_count": len(direct_res.rejected_cases),
                    "candidate_plan_reconciliation": dict(
                        candidate_plan_reconciliation
                    ),
                    "note": "Hailo feasibility/build failures no longer veto TensorRT/ORT/DeepX-full benchmark materialization.",
                })
            except Exception:
                pass
            direct_artifacts = dict(direct_res.artifacts)
            if fallback_evidence_path is not None:
                direct_artifacts["legacy_hailo_veto_fallback_json"] = (
                    fallback_evidence_path
                )
            return LegacyBenchmarkSetResult(
                artifacts=direct_artifacts,
                metrics={
                    **dict(direct_res.metrics),
                    "fallback_from_legacy_hailo_veto": True,
                    "legacy_rejected_cases": len(rejected),
                    "candidate_plan_reconciliation": dict(
                        candidate_plan_reconciliation
                    ),
                    "candidate_plan_claim_eligible_after_legacy_generation": bool(
                        candidate_plan_reconciliation.get("claim_eligible")
                    ),
                },
                message=(direct_res.message or "Direct split suite materialized after legacy Hailo veto."),
                status=direct_res.status,
                suite_dir=direct_res.suite_dir or (model_dir / "benchmark_set" / "generated_suite"),
            )
        except Exception as exc:
            _log(f"direct fallback after accepted=0 failed: {type(exc).__name__}: {exc}")
            if _is_frozen_predeclared_execution_union(candidate_plan):
                failed_projection = _post_build_audit_minimum_projection(
                    candidate_plan,
                    [],
                )
                candidate_plan_reconciliation.update(failed_projection)
                candidate_plan_reconciliation.update({
                    "status": "frozen_union_direct_fallback_failed",
                    "execution_status": "direct_fallback_failed",
                    "repair_required": True,
                    "repair_status": "required_no_automatic_backfill",
                    "repair_reason": (
                        "direct_fallback_failed:"
                        f"{type(exc).__name__}:{exc}"
                    ),
                    "claim_eligible": False,
                })

    artifacts = _copy_small_suite_summary(suite_dir, formal_bdir, run_dir)
    artifacts.update(reconciled_plan_paths)
    if feasibility_receipt_path is not None:
        artifacts["hailo_feasibility_receipt_json"] = (
            feasibility_receipt_path
        )
    p_input = write_json(formal_bdir / "generator_input.json", {
        "schema": "onnx-splitpoint/benchmark-generator-input",
        "schema_version": 3,
        "mode": "legacy_benchmarkset_source_of_truth",
        "model_id": model_id,
        "profile_id": profile_id,
        "run_id": run_id,
        "model_path": str(p_model),
        "source_candidate_plan_path": "../analysis/final_candidate_plan.json",
        "consumed_candidate_plan_artifact_id": consumed_candidate_plan_artifact_id,
        "resolved_candidate_plan_artifact_id": candidate_plan.get("artifact_id", ""),
        "candidate_plan_reconciliation": candidate_plan_reconciliation,
        "candidate_plan_is_authoritative": True,
        "ranked_candidates": ranked_candidates,
        "candidate_search_pool": candidate_search_pool,
        "target_cases": requested,
        "hailo_build_mode": build_mode,
        "hailo_calibration_dir_effective": effective_hailo_calib_dir,
        "accelerator_calibration_source_kind": calibration_source_kind,
        "accelerator_calibration_manifest": calibration_manifest,
        "hailo_calibration_task_hint": task_hint_for_calib,
        "legacy_suite_dir": relpath(suite_dir, run_dir),
        "note": "The Evaluation Workflow delegates build/split/HEF generation to the existing BenchmarkSet generator; this file records the handoff only.",
        "created_at": now_iso(),
    })
    p_decisions = write_json(formal_bdir / "generation_decisions.json", {
        "schema": "onnx-splitpoint/benchmark-generation-decisions",
        "schema_version": 4,
        "mode": "legacy_benchmarkset_source_of_truth",
        "model_id": model_id,
        "profile_id": profile_id,
        "run_id": run_id,
        "source_prediction_artifact_id": prediction.get("artifact_id", ""),
        "source_candidate_plan_artifact_id": candidate_plan.get("artifact_id", ""),
        "consumed_candidate_plan_artifact_id": consumed_candidate_plan_artifact_id,
        "candidate_plan_is_authoritative": True,
        "legacy_suite_dir": relpath(suite_dir, run_dir),
        "accepted_cases": accepted,
        "rejected_cases": rejected,
        "benchmark_generation_status": getattr(orch, "final_status", ""),
        "benchmark_generation_message": getattr(orch, "final_msg", ""),
        "benchmark_generation_summary": generation_summary,
        "hailo_feasibility_outcome": feasibility_outcome,
        "fallback_allowed": False if bool(
            hailo_feasibility_control.get("enabled")
        ) else True,
        "policy_promotions": list(candidate_plan.get("policy_promotions") or []),
        "policy_backfills": list(candidate_plan.get("policy_backfills") or []),
        "candidate_plan_reconciliation": candidate_plan_reconciliation,
        "created_at": now_iso(),
    })
    backend_state = runtime.generation_state.get('backend_backfill') or {}
    if backend_state:
        bind_plan_cases(plan_payload, backend_state)
        write_json(suite_dir / 'benchmark_plan.json', plan_payload)
        suite_contract = _read_json(suite_dir / 'benchmark_set.json', {})
        suite_contract['backend_backfill'] = backend_state
        write_json(suite_dir / 'benchmark_set.json', suite_contract)
        write_json(formal_bdir / 'backend_selection.json', backend_state)
    p_contract = write_json(formal_bdir / "benchmark_set.json", {
        "schema": "onnx-splitpoint/benchmark-set-contract",
        "schema_version": 4,
        "mode": "legacy_benchmarkset_source_of_truth",
        "model_id": model_id,
        "profile_id": profile_id,
        "run_id": run_id,
        "source": "existing_benchmark_generation_orchestration_service",
        "source_candidate_plan_artifact_id": candidate_plan.get("artifact_id", ""),
        "consumed_candidate_plan_artifact_id": consumed_candidate_plan_artifact_id,
        "candidate_plan_reconciliation": candidate_plan_reconciliation,
        "candidate_plan_is_authoritative": True,
        "suite_dir": relpath(suite_dir, run_dir),
        "legacy_suite_dir": relpath(suite_dir, run_dir),
        "legacy_suite_benchmark_set": relpath(suite_dir / "benchmark_set.json", run_dir),
        "benchmark_plan": relpath(suite_dir / "benchmark_plan.json", run_dir) if (suite_dir / "benchmark_plan.json").is_file() else "",
        "benchmark_suite_py": relpath(suite_dir / "benchmark_suite.py", run_dir) if (suite_dir / "benchmark_suite.py").is_file() else "",
        "materialized": bool(accepted),
        "materialization_scope": "legacy_benchmarkset_full_pipeline",
        "cases": accepted,
        "backend_backfill": backend_state,
        "rejected_cases": rejected,
        "planned_runs": list(plan_payload.get("runs") or plan_payload.get("planned_runs") or []),
        "status": (
            "failed"
            if feasibility_exhausted
            else ("ok" if accepted else "partial")
        ),
        "hailo_feasibility_outcome": feasibility_outcome,
        "fallback_allowed": False if bool(
            hailo_feasibility_control.get("enabled")
        ) else True,
    })
    # Mirror benchmark_plan.json at the formal level so existing runner/report
    # code does not need to know whether the suite is legacy or direct.
    if isinstance(plan_payload, Mapping) and plan_payload:
        plan_copy = dict(plan_payload)
    else:
        plan_copy = {"runs": list((suite_payload or {}).get("planned_runs") or []), "planned_runs": list((suite_payload or {}).get("planned_runs") or [])}
    plan_copy = _append_deepx_runs_to_plan(plan_copy, profile_payload)
    deferred_full_builds = _discover_deferred_hailo_full_builds(suite_dir)
    plan_copy = _apply_deferred_hailo_full_builds(plan_copy, deferred_full_builds)
    if central_cpu_reference:
        # Post-generation invariant: the executable suite and every formal
        # mirror below derive from one, and only one, producer-owned CPU
        # recipe.  Fail here rather than discovering the omission after remote
        # accelerator work has already started.
        bound_cpu_runs = bind_management_cpu_reference_runs(
            [
                dict(run)
                for run in list(
                    plan_copy.get("runs")
                    or plan_copy.get("planned_runs")
                    or []
                )
                if isinstance(run, Mapping)
            ],
            automatic=not explicit_cpu_reference,
            require_existing=True,
        )
        plan_copy["runs"] = bound_cpu_runs
        plan_copy["planned_runs"] = [
            dict(run) for run in bound_cpu_runs
        ]
        plan_copy["management_cpu_reference_invariant"] = {
            "status": "verified",
            "recipe_count": 1,
            "execution_location": "central_management",
            "performance_dispatch_allowed": False,
        }
    if deferred_full_builds:
        _log(
            "[build-policy] deferred Smoke Full baselines removed from dispatch/completeness: "
            + ", ".join(str(row.get("target") or "") for row in deferred_full_builds)
        )
        try:
            write_json(suite_dir / "deferred_full_baselines.json", {
                "schema": "onnx-splitpoint/deferred-full-baselines",
                "schema_version": 1,
                "created_at": now_iso(),
                "mode": "smoke_cache_or_defer",
                "rows": deferred_full_builds,
                "deferred_run_ids": list(plan_copy.get("deferred_run_ids") or []),
            })
        except Exception:
            pass
    scientific_freeze_artifacts = {}
    analysis_dir = model_dir / "analysis"
    for freeze_name in (
        "prediction.json",
        "predictions_frozen.csv",
        "holdout_predictions_frozen.csv",
        "ranking_predictions_frozen.csv",
        "holdout_ranking_predictions_frozen.csv",
        "prediction_freeze_manifest.json",
        "prediction_freeze_conflict.json",
        "candidate_universe_manifest.json",
        "candidate_universe.csv",
        "prediction_freeze_approval.json",
        "prediction_freeze_approval.json.sha256",
        "prediction_freeze_approval.json.sig",
        "prediction_freeze_public_key.pem",
        "prediction_freeze_approval_verification.json",
    ):
        source = analysis_dir / freeze_name
        if not source.is_file():
            continue
        destination = suite_dir / freeze_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        scientific_freeze_artifacts[freeze_name] = freeze_name
    plan_copy.update({
        "mode": "legacy_benchmarkset_source_of_truth",
        "legacy_suite_dir": relpath(suite_dir, run_dir),
        "suite_dir": relpath(suite_dir, run_dir),
        "quality_gate": dict(profile_payload.get("quality_gate") or {}) if isinstance(profile_payload.get("quality_gate"), Mapping) else {},
        "official_coco_evaluation": dict(profile_payload.get("official_coco_evaluation") or {}) if isinstance(profile_payload.get("official_coco_evaluation"), Mapping) else {},
        "ranking_validation": dict(profile_payload.get("ranking_validation") or {}) if isinstance(profile_payload.get("ranking_validation"), Mapping) else {},
        "campaign": dict(profile_payload.get("campaign") or {}) if isinstance(profile_payload.get("campaign"), Mapping) else {},
        "model_suite": {"primary": [dict(row or {"id": model_id})], "reserve": []},
        "prediction_freeze_artifacts": scientific_freeze_artifacts,
    })
    # Keep the executable Benchmark suite plan in sync with the formal mirror so
    # RemoteBenchmarkService can execute DeepX full runs from benchmark_suite.py.
    try:
        write_json(suite_dir / "benchmark_plan.json", plan_copy)
    except Exception:
        pass
    # The formal BenchmarkSet contract must expose the same active/deferred run
    # matrix as the executable plan; otherwise pack/report consumers can still
    # treat an intentional Smoke deferral as a missing required result.
    try:
        _contract_payload = _read_json(p_contract, default={}) or {}
        _contract_payload["planned_runs"] = [dict(r) for r in list(plan_copy.get("runs") or []) if isinstance(r, Mapping)]
        _contract_payload["deferred_full_baselines"] = list(plan_copy.get("deferred_full_baselines") or [])
        _contract_payload["deferred_run_ids"] = list(plan_copy.get("deferred_run_ids") or [])
        write_json(p_contract, _contract_payload)
    except Exception:
        pass

    # v55l: Evaluation Workflow must materialize the same per-split DeepX Part1
    # DXNN artifacts as the manual Benchmark tab when DeepX→TensorRT is planned.
    # Without these, remote runs fail with "Missing DXNN for DeepX stage1/part1".
    try:
        from ..deepx.env_status import profile_compiler_configuration
        deepx_build_cfg = profile_compiler_configuration(profile_payload)
        # v60r: the task-specific content-addressed manifest is authoritative.
        # A generic legacy DeepX path must never route a detector to Imagenette
        # (or a classifier to COCO) before the cache key is computed.
        task_specific_keys = (
            ("detection_calib_dir", "detection_calibration_dir")
            if task_hint_for_calib == "detection"
            else ("classification_calib_dir", "classification_calibration_dir")
        )
        task_specific_cfg = next(
            (str(deepx_build_cfg.get(key) or "").strip() for key in task_specific_keys if str(deepx_build_cfg.get(key) or "").strip()),
            "",
        )
        generic_cfg = str(deepx_build_cfg.get("calib_dir") or deepx_build_cfg.get("calibration_dir") or "").strip()
        generic_low = generic_cfg.lower()
        generic_compatible = bool(generic_cfg) and not (
            (task_hint_for_calib == "detection" and any(tok in generic_low for tok in ("imagenet", "imagenette", "classification")))
            or (task_hint_for_calib == "classification" and any(tok in generic_low for tok in ("coco", "detection")))
        )
        deepx_calib_fallback = (
            effective_hailo_calib_dir
            or task_specific_cfg
            or (generic_cfg if generic_compatible else "")
            or _preset_image_dir_local('', task=task_hint_for_calib)
        )
        deepx_materializer = defer_deepx_part1_build if defer_deepx_builds else _materialize_manual_deepx_part1_artifacts
        from ..backend_backfill import call_with_build_budget
        deepx_part1_status = call_with_build_budget(deepx_materializer,
            state=backend_state, persist=runtime.persist, stage='part1',
            selected_case_dirs=selected_cases_for_backend(backend_state, 'deepx', [str(c.get('folder') or c.get('case_dir')) for c in accepted]),
            out_dir=suite_dir,
            bench_plan_runs=list(plan_copy.get("runs") or plan_copy.get("planned_runs") or []),
            validation_images=str(validation_defaults.get("validation_images") or ""),
            fallback_calib_dir=deepx_calib_fallback,
            calibration_num=_mode_task_item_count(
                profile_payload,
                task_hint_for_calib,
                kind="calibration_items",
                fallback=_safe_int(deepx_build_cfg.get("calib_count") or getattr(options, "deepx_calib_count", 0) or validation_defaults.get("validation_max_images") or 100, 100),
            ),
            task_hint=task_hint_for_calib,
            calibration_manifest=str(calibration_manifest or ""),
            strict_calibration_source=bool(calibration_manifest or effective_hailo_calib_dir),
            classification_preprocessing=str(
                deepx_build_cfg.get("classification_preprocessing")
                or "imagenet_mean_std"
            ),
            build_config=deepx_build_cfg,
            profile_payload=profile_payload,
            force_build=parse_config_bool(deepx_build_cfg.get("force_build", False), field="deepx_build.force_build"),
            log=lambda msg, **_kw: _log(str(msg)),
        )
        try:
            write_json(formal_bdir / "deepx_part1_artifact_status.json", deepx_part1_status)
        except Exception:
            pass
        if bool((deepx_part1_status or {}).get("selected")):
            _log(f"[deepx] eval Part1 DXNN artifacts ready status={(deepx_part1_status or {}).get('status')} ok={(deepx_part1_status or {}).get('ok_count', 0)} failed={(deepx_part1_status or {}).get('failed_count', 0)}")
    except Exception as exc:
        _log(f"[deepx] eval Part1 DXNN artifact materialization failed: {type(exc).__name__}: {exc}")

    if backend_state:
        bind_plan_cases(plan_copy, backend_state)
        write_json(suite_dir / 'benchmark_plan.json', plan_copy)
    p_plan = write_json(formal_bdir / "benchmark_plan.json", plan_copy)
    cpu_reference_invariant = finalize_management_cpu_reference_plan_aliases(
        executable_plan_path=suite_dir / "benchmark_plan.json",
        formal_plan_path=p_plan,
        profile=profile_payload,
        cache_verify_enabled=bool(cache_guard),
        automatic=not explicit_cpu_reference,
        require_existing=True,
        benchmark_set_paths=(suite_dir / "benchmark_set.json", p_contract),
    )
    p_cases = write_csv(
        formal_bdir / "benchmark_cases.csv",
        [
            {
                "model_id": model_id,
                "case_id": c.get("case_id") or c.get("folder") or c.get("id") or "",
                "split_index": c.get("split_index", c.get("boundary", "")),
                "prediction_rank": c.get("prediction_rank", c.get("rank", "")),
                "generation_status": "accepted",
                "source": "legacy_benchmarkset",
            }
            for c in accepted
        ],
        ["model_id", "case_id", "split_index", "prediction_rank", "generation_status", "source"],
    )
    p_rejected = write_csv(
        formal_bdir / "rejected_cases.csv",
        [
            {
                "model_id": model_id,
                "case_id": c.get("case_id") or c.get("folder") or c.get("id") or "",
                "split_index": c.get("split_index", c.get("boundary", "")),
                "reject_reason": c.get("reason") or c.get("reject_reason") or c.get("detail") or "",
                "error_class": c.get("error_class") or c.get("issue_kind") or "",
            }
            for c in rejected
        ],
        ["model_id", "case_id", "split_index", "reject_reason", "error_class"],
    )
    p_binding = write_json(formal_bdir / "legacy_benchmarkset_binding.json", {
        "schema": "onnx-splitpoint/legacy-benchmarkset-binding",
        "schema_version": 1,
        "model_id": model_id,
        "profile_id": profile_id,
        "run_id": run_id,
        "suite_dir": relpath(suite_dir, run_dir),
        "benchmark_set_json": relpath(suite_dir / "benchmark_set.json", run_dir) if (suite_dir / "benchmark_set.json").is_file() else "",
        "benchmark_plan_json": relpath(suite_dir / "benchmark_plan.json", run_dir) if (suite_dir / "benchmark_plan.json").is_file() else "",
        "benchmark_suite_py": relpath(suite_dir / "benchmark_suite.py", run_dir) if (suite_dir / "benchmark_suite.py").is_file() else "",
        "benchmark_generation_log": relpath(bench_log_path, run_dir),
        "final_status": getattr(orch, "final_status", ""),
        "final_message": getattr(orch, "final_msg", ""),
        "accepted_case_count": len(accepted),
        "rejected_case_count": len(rejected),
        "created_at": now_iso(),
    })
    artifacts.update({
        "generator_input_json": p_input,
        "generation_decisions_json": p_decisions,
        "benchmark_set_json": p_contract,
        "benchmark_plan_json": p_plan,
        "benchmark_cases_csv": p_cases,
        "rejected_cases_csv": p_rejected,
        "legacy_benchmarkset_binding_json": p_binding,
    })
    if (suite_dir / "benchmark_suite.py").is_file():
        artifacts["benchmark_suite_py"] = suite_dir / "benchmark_suite.py"
    if (suite_dir / "benchmark_set.json").is_file():
        artifacts["legacy_suite_benchmark_set_json"] = suite_dir / "benchmark_set.json"
    if (suite_dir / "benchmark_plan.json").is_file():
        artifacts["legacy_suite_benchmark_plan_json"] = suite_dir / "benchmark_plan.json"
    if (suite_dir / "deepx_prefetch_v60s.json").is_file():
        artifacts["deepx_prefetch_v60s_json"] = suite_dir / "deepx_prefetch_v60s.json"

    # v60t: publish all successfully materialised compiler artifacts only after
    # the authoritative BenchmarkSet is complete.  This closes the final v60s
    # integration gap: the Artifact Library could index legacy caches manually,
    # but newly generated HEF/DXNN files were not registered by the normal
    # Evaluation Workflow path.
    artifact_registry_report: Mapping[str, Any] = {}
    try:
        artifact_registry_report = register_benchmark_set_artifacts(
            suite_dir=suite_dir,
            model_id=model_id,
            task=task_hint_for_calib,
            profile=profile_payload,
            source_run=run_id,
        )
        registry_path = suite_dir / "artifact_registry_registration.json"
        if registry_path.is_file():
            artifacts["artifact_registry_registration_json"] = registry_path
        _log(
            f"[artifact-store] post-generation registration: "
            f"successful={int((artifact_registry_report or {}).get('successful') or 0)} "
            f"failed={int((artifact_registry_report or {}).get('failed') or 0)}"
        )
    except Exception as exc:
        artifact_registry_report = {"enabled": True, "successful": 0, "failed": 1, "error": f"{type(exc).__name__}: {exc}"}
        _log(f"[artifact-store] post-generation registration failed: {type(exc).__name__}: {exc}")

    final_status = str(getattr(orch, "final_status", "") or "").strip().lower()
    status = (
        "failed"
        if feasibility_exhausted
        else (
            "ok"
            if accepted and final_status in {"ok", "warn", ""}
            else "partial"
        )
    )
    msg = "BenchmarkSet generator produced the authoritative suite."
    if feasibility_exhausted:
        msg = (
            f"{feasibility_outcome}: Hailo8-first Gate-A did not admit a "
            "common Part1 anchor; backend-agnostic fallback and the unchanged "
            "B5 are blocked."
        )
    if final_status and final_status not in {"ok", "warn"}:
        msg += f" Generator status={final_status}."
    _log(f"benchmark suite ready: accepted={len(accepted)} rejected={len(rejected)} suite={suite_dir}")
    return LegacyBenchmarkSetResult(
        artifacts=artifacts,
        metrics={
            "binding_mode": "legacy_benchmarkset_source_of_truth",
            "legacy_suite_dir": relpath(suite_dir, run_dir),
            "accepted_cases": len(accepted),
            "rejected_cases": len(rejected),
            "target_cases": requested,
            "candidate_pool_size": len(candidate_search_pool),
            "planned_run_count": len(list(plan_payload.get("runs") or plan_payload.get("planned_runs") or [])) if isinstance(plan_payload, Mapping) else 0,
            "management_cpu_reference_invariant": cpu_reference_invariant,
            "hailo_build_mode": build_mode,
            "benchmark_generation_status": final_status,
            "benchmark_generation_log": relpath(bench_log_path, run_dir),
            "terminal_reason": (
                feasibility_outcome
                if feasibility_exhausted
                else feasibility_outcome
            ),
            "fallback_allowed": False if bool(
                hailo_feasibility_control.get("enabled")
            ) else True,
            "stop_workflow": feasibility_stop_workflow,
            "hailo_feasibility_outcome": feasibility_outcome,
            "artifact_registry_registered": int((artifact_registry_report or {}).get("successful") or 0),
            "artifact_registry_failed": int((artifact_registry_report or {}).get("failed") or 0),
        },
        message=msg,
        status=status,
        suite_dir=suite_dir,
    )
