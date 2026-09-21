#!/usr/bin/env python3
"""Build a unified native-producer report for Hailo8, Hailo10 and DeepX.

This is the thesis-facing collector for the native producer execution mode. It
can operate on one staged root, or recursively collect analysis_tables and
native_pipeline result files from several copied backend roots. It intentionally
keeps the native producer as a strict execution mode: rows are marked ok only
when a native E2E result exists and is ok. Probe-only / unsupported rows remain
explicit.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, os, random, re, statistics, sys
from pathlib import Path
from typing import Any, Iterable, Mapping


def _resolve_tool_root(script_path: str | Path) -> Path:
    """Resolve both ``scripts/`` and packaged ``resources/remote_scripts/``."""
    resolved = Path(script_path).resolve()
    candidates = [resolved.parents[1]]
    if len(resolved.parents) > 3:
        candidates.append(resolved.parents[3])
    relative_contract = (
        Path("onnx_splitpoint_tool") / "resources" / "validation"
        / "hailo10_yolo26_claim_exclusions_v272.json"
    )
    for candidate in candidates:
        if (candidate / relative_contract).is_file():
            return candidate
    # Preserve the source-script layout as a fail-closed missing-file path.
    return candidates[0]


_TOOL_ROOT = _resolve_tool_root(__file__)
if str(_TOOL_ROOT) not in sys.path:
    # A directly executed source script otherwise resolves imports against an
    # unrelated installed package before the checked source tree.
    sys.path.insert(0, str(_TOOL_ROOT))
_DEFAULT_DETECTION_EXCLUSIONS = (
    _TOOL_ROOT / "onnx_splitpoint_tool" / "resources" / "validation"
    / "hailo10_yolo26_claim_exclusions_v272.json"
)
_DEFAULT_DETECTION_EXCLUSIONS_SHA256 = (
    "5303970c1d84150202b93abc4cda2d0196a9613678a150a711f944dbcd4c1fda"
)

try:
    from onnx_splitpoint_tool.quality_service import (
        _validate_candidate_execution_contract as _validate_quality_producer_contract,
    )
except Exception:  # pragma: no cover - copied collector must fail closed
    _validate_quality_producer_contract = None

try:
    from onnx_splitpoint_tool.native_command_contract import (
        verify_native_command_contract as _strict_verify_native_command_contract,
    )
except Exception:  # pragma: no cover - copied standalone tools must fail closed
    _strict_verify_native_command_contract = None

try:
    from onnx_splitpoint_tool.native_detection_postprocess import (
        build_completed_detection_endpoint_attestation as _build_completed_detection_endpoint_attestation,
        build_normalized_detection_endpoint_attestation as _build_normalized_detection_endpoint_attestation,
        verify_completed_detection_comparison_endpoint_contract as _verify_completed_detection_comparison_endpoint_contract,
        verify_detection_completion_execution_attestation as _verify_detection_completion_execution_attestation,
        verify_detection_completion_execution_contract as _verify_detection_completion_execution_contract,
        verify_frozen_decoded_nms_normalization_contract as _verify_frozen_decoded_nms_normalization_contract,
    )
except Exception:  # pragma: no cover - copied collector must fail closed
    _build_completed_detection_endpoint_attestation = None
    _build_normalized_detection_endpoint_attestation = None
    _verify_completed_detection_comparison_endpoint_contract = None
    _verify_detection_completion_execution_attestation = None
    _verify_detection_completion_execution_contract = None
    _verify_frozen_decoded_nms_normalization_contract = None

try:
    from onnx_splitpoint_tool.validation.host_postprocess import (
        apply_host_postprocess_aliases as _apply_host_postprocess_aliases,
    )
except Exception:  # pragma: no cover - copied collector must fail closed
    _apply_host_postprocess_aliases = None

try:
    from onnx_splitpoint_tool.native_split_quality import (
        bind_quality_to_native_split as _bind_quality_to_native_split,
    )
except Exception:  # pragma: no cover - copied standalone tools must fail closed
    _bind_quality_to_native_split = None

try:
    from onnx_splitpoint_tool.native_split_quality_authority import (
        apply_native_split_quality_authority as _apply_split_quality_authority,
        canonical_native_split_backend as _canonical_native_split_backend,
        is_native_split_backend as _is_native_split_backend,
        resolve_native_split_quality_authority as _resolve_split_quality_authority,
    )
except Exception:  # pragma: no cover - copied standalone tools must fail closed
    _apply_split_quality_authority = None
    _canonical_native_split_backend = None
    _is_native_split_backend = None
    _resolve_split_quality_authority = None

try:
    from onnx_splitpoint_tool.native_detection_diagnostics import (
        verify_prospective_detection_exclusion_set as
        _verify_prospective_detection_exclusion_set,
    )
except Exception:  # pragma: no cover - standalone must fail closed
    _verify_prospective_detection_exclusion_set = None

try:
    from scripts.native_producer_energy_plan import (
        _verify_full_command_contract as _strict_verify_full_command_contract,
    )
except Exception:  # pragma: no cover - standalone remote script layout
    try:
        from native_producer_energy_plan import (  # type: ignore
            _verify_full_command_contract as _strict_verify_full_command_contract,
        )
    except Exception:  # pragma: no cover - missing verifier is not claimable
        _strict_verify_full_command_contract = None


def _load_json(p: Path | str | None) -> Any:
    try:
        pp = Path(p) if p else None
        if pp and pp.is_file():
            return json.loads(pp.read_text(encoding='utf-8'))
    except Exception:
        return None
    return None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


_REMOTE_EXECUTION_CONTEXT_SCHEMA = (
    'onnx-splitpoint/native-final-report-remote-execution-context'
)
_REMOTE_EXECUTION_CONTEXT_VERSION = 1


def _canonical_remote_context_path(value: Any) -> str:
    raw = str(value or '').strip()
    path = Path(raw)
    if not raw or not path.is_absolute() or '..' in path.parts:
        return ''
    normalized = os.path.normpath(raw)
    return raw if normalized == raw else ''


def _parse_remote_execution_context_allowlist(
    raw_values: Iterable[Any],
) -> list[dict[str, str]]:
    """Parse the external setup/root/tool allowlist used by DeepX reports."""
    records: list[dict[str, str]] = []
    for raw in raw_values:
        try:
            parsed = json.loads(str(raw)) if not isinstance(raw, Mapping) else raw
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError('remote execution context JSON is invalid') from exc
        values = parsed if isinstance(parsed, list) else [parsed]
        for value in values:
            if not isinstance(value, Mapping):
                raise ValueError('remote execution context must be an object')
            if (
                value.get('schema') != _REMOTE_EXECUTION_CONTEXT_SCHEMA
                or int(value.get('schema_version') or 0)
                != _REMOTE_EXECUTION_CONTEXT_VERSION
            ):
                raise ValueError('remote execution context schema is invalid')
            setup_id = str(value.get('setup_id') or '').strip()
            remote_root = _canonical_remote_context_path(
                value.get('remote_root')
            )
            remote_tool_dir = _canonical_remote_context_path(
                value.get('remote_tool_dir')
            )
            if not setup_id or not remote_root or not remote_tool_dir:
                raise ValueError('remote execution context identity is invalid')
            record = {
                'setup_id': setup_id,
                'remote_root': remote_root,
                'remote_tool_dir': remote_tool_dir,
            }
            same_scope = [
                item for item in records
                if item['setup_id'] == setup_id
                and item['remote_root'] == remote_root
            ]
            if same_scope and any(
                item['remote_tool_dir'] != remote_tool_dir
                for item in same_scope
            ):
                raise ValueError(
                    'remote execution context tool-dir conflicts for setup/root'
                )
            if record not in records:
                records.append(record)
    return records


def _remote_execution_context_for_performance_row(
    row: Mapping[str, Any], contract: Mapping[str, Any],
    allowlist: Iterable[Mapping[str, Any]],
) -> tuple[dict[str, str], str]:
    setup_id = str(row.get('setup_id') or '').strip()
    declared_root = _canonical_remote_context_path(contract.get('root'))
    matches = [
        dict(item) for item in allowlist
        if str(item.get('setup_id') or '') == setup_id
        and str(item.get('remote_root') or '') == declared_root
    ]
    if not setup_id or not declared_root or not matches:
        return {}, 'full_command_contract_deepx_remote_execution_context_missing'
    if len(matches) != 1:
        return {}, 'full_command_contract_deepx_remote_execution_context_ambiguous'
    return matches[0], 'remote_execution_context_verified_exact'


def _num(x: Any) -> Any:
    try:
        if x is None or x == '':
            return None
        return float(x)
    except Exception:
        return None


def _finite_num(value: Any) -> float | None:
    """Parse one finite number without accepting booleans or NaN/Inf."""
    if isinstance(value, bool):
        return None
    parsed = _num(value)
    if parsed is None:
        return None
    parsed = float(parsed)
    return parsed if math.isfinite(parsed) else None


def _strict_bool(value: Any) -> bool | None:
    """Accept only JSON/Python booleans; aliases are malformed evidence."""
    return value if isinstance(value, bool) else None


def _first_explicit_bool(
    sources: Iterable[Mapping[str, Any]], fields: Iterable[str],
) -> bool | None:
    """Return the first explicit boolean, failing closed on malformed values."""
    for source in sources:
        if not isinstance(source, Mapping):
            continue
        for field in fields:
            if field not in source or source.get(field) in (None, ''):
                continue
            parsed = _strict_bool(source.get(field))
            return parsed if parsed is not None else False
    return None


def _strict_nonnegative_int(value: Any) -> int | None:
    """Return an exact non-negative integer; reject truncation and booleans."""
    if isinstance(value, bool) or value in (None, ''):
        return None
    try:
        parsed = float(value)
    except Exception:
        return None
    if not math.isfinite(parsed) or parsed < 0 or not parsed.is_integer():
        return None
    return int(parsed)


def _percentile(values: Iterable[float], q: float) -> float | None:
    vals = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    pos = max(0.0, min(1.0, float(q))) * (len(vals) - 1)
    lo = int(math.floor(pos)); hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    frac = pos - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def _bootstrap_median_ci(values: Iterable[float], seed_key: str) -> tuple[float | None, float | None, float | None]:
    vals = [float(value) for value in values if math.isfinite(float(value))]
    if not vals:
        return None, None, None
    median = float(statistics.median(vals))
    if len(vals) < 2:
        return median, None, None
    rng = random.Random(int(hashlib.sha256(seed_key.encode("utf-8")).hexdigest()[:16], 16))
    n = len(vals)
    boot = [float(statistics.median(vals[rng.randrange(n)] for _ in range(n))) for _ in range(4000)]
    return median, _percentile(boot, 0.025), _percentile(boot, 0.975)


# Keep the original prerequisite/count types across reader defaults.  This is a
# diagnostic projection of existing observations, not new runtime evidence.
_PREREQUISITE_OBSERVATION_FIELDS = (
    'backend', 'producer_backend', 'model', 'model_id', 'case', 'case_id',
    'setup_id', 'measurement_setup_id', 'comparison_backend', 'precision',
    'planned_native_identity', 'status', 'prerequisite_status',
    'failure_stage', 'primary_failure_reason', 'upstream_evidence_path',
    'build_exclusion', 'upstream_build_observation', 'disposition',
    'repetition_count_attempted', 'performance_repeat_count_attempted',
    'repetition_count_valid', 'performance_repeat_count_valid', 'performance_repeat_n',
    'repetitions_completed', 'completed_frames', 'frames_completed', 'completed_work_units',
    'postprocess_completed_frames', 'repetition_index',
    'repetition_records', 'repetition_evidence', 'performance_repetitions',
    'fps_repetition_samples', 'latency_mean_repetition_samples_ms',
    'fps_makespan', 'fps_median', 'latency_mean_ms', 'latency_median_ms',
    'latency_p50_ms', 'latency_p95_ms', 'completion_interval_mean_ms',
    'handoff_ms', 'p1_ms', 'p2_run_ms', 'p1_thread_ms', 'p2_thread_ms',
    'runtime_success', 'runtime_started', 'execution_started', 'measurement_started',
    'ok', 'result_ok', 'identity_conflicts', 'identity_status',
    'output_endpoint_id', 'endpoint_contract_hash', 'performance_endpoint',
)


def _prerequisite_observations(*sources: Mapping[str, Any]) -> list[dict[str, Any]]:
    if not any(isinstance(source, Mapping) and (
        source.get('prerequisite_status') == 'blocked'
        or source.get('status') in {'blocked', 'blocked_upstream_quality', 'excluded_known_build'}
    ) for source in sources):
        return []
    return [{field: source[field] for field in _PREREQUISITE_OBSERVATION_FIELDS
             if field in source} for source in sources if isinstance(source, Mapping) and source]


def _diagnostic_fields(row: dict[str, Any], result: dict[str, Any], *, fallback: str = "", ok: bool = False) -> dict[str, Any]:
    """Return diagnostics without manufacturing a failure for successful rows.

    Earlier collectors always inserted the backend fallback string even when the
    native command returned ``ok=true``.  This made successful rows appear to
    have failed in the concise report and workflow log.  A fallback is now used
    only for an actual runtime failure.
    """
    steps = row.get("steps") if isinstance(row.get("steps"), list) else []
    last_step = steps[-1] if steps and isinstance(steps[-1], dict) else {}
    explicit_failure = str(
        row.get("failure_reason") or result.get("failure_reason") or
        row.get("reason") or result.get("error") or ""
    ).strip()
    failure = "" if ok else (explicit_failure or str(fallback or ""))
    explicit_error = str(
        row.get("error") or result.get("error") or
        (last_step.get("stderr_tail") if not ok else "") or
        (last_step.get("stdout_tail") if not ok else "") or ""
    ).strip()
    return {
        **{field: source[field] for source in (result, row) for field in (
            "failure_stage", "primary_failure_reason", "upstream_evidence_path",
            "prerequisite_status", "identity_conflicts", "child_observation", "execution_mode",
            "build_exclusion", "upstream_build_observation", "disposition",
        ) if field in source},
        "prerequisite_observations": _prerequisite_observations(row, result),
        "failure_reason": failure,
        "status_detail": "ok" if ok else str(row.get("status_detail") or result.get("status_detail") or failure),
        "error": "" if ok and not explicit_failure else explicit_error,
        "timed_out": bool(row.get("timed_out") or result.get("timed_out") or last_step.get("timed_out") or last_step.get("timeout")),
        "returncode": row.get("returncode") if row.get("returncode") is not None else last_step.get("rc"),
        "stdout_tail": str(row.get("stdout_tail") or result.get("stdout_tail") or last_step.get("stdout_tail") or ""),
        "stderr_tail": str(row.get("stderr_tail") or result.get("stderr_tail") or last_step.get("stderr_tail") or ""),
    }


def _repeat_fields(*sources: dict[str, Any]) -> dict[str, Any]:
    """Ingest the shared Split/Full repetition contract without losing raw evidence."""
    def pick(*keys: str) -> Any:
        for source in sources:
            if not isinstance(source, dict):
                continue
            for key in keys:
                value = source.get(key)
                if value not in (None, "", []):
                    return value
        return None

    raw = pick("repetition_records", "repetition_evidence", "performance_repetitions", "repeat_measurements")
    if isinstance(raw, dict):
        raw = raw.get("rows") or raw.get("repetitions") or raw.get("records") or []
    raw = list(raw) if isinstance(raw, list) else []
    samples = pick("fps_repetition_samples", "performance_fps_samples", "repeat_fps_samples")
    samples = _as_numeric_list(samples)
    if not samples:
        samples = [
            float(value) for value in (
                _num(record.get("fps_makespan") or record.get("fps"))
                for record in raw if isinstance(record, dict) and bool(record.get("ok", True))
            ) if value is not None
        ]
    median = _num(pick("fps_median", "fps_makespan_median", "performance_fps_median"))
    if median is None and samples:
        median = float(statistics.median(samples))
    latency_samples = _as_numeric_list(pick("latency_mean_repetition_samples_ms"))
    if not latency_samples:
        latency_samples = [
            float(value) for value in (
                _num(record.get("latency_median_ms") or record.get("latency_mean_ms"))
                for record in raw if isinstance(record, dict) and bool(record.get("ok", True))
            ) if value is not None
        ]
    attempted_keys = ('repetition_count_attempted', 'repetitions_completed', 'performance_repeat_count_attempted')
    explicit_attempts = [source[key] for source in sources if isinstance(source, Mapping)
                         for key in attempted_keys if key in source]
    if explicit_attempts:
        attempted = explicit_attempts[0]
        if type(attempted) is not int or attempted < 0:
            attempted = None
    else:
        # Actual legacy samples imply attempts; an empty diagnostic does not.
        attempted = max(len(raw), len(samples)) or (
            1 if _finite_num(pick('fps_makespan', 'fps')) is not None else None)
    return {
        "fps_makespan": median if median is not None else _num(pick("fps_makespan", "fps")),
        "fps_median": median,
        "fps_ci95_low": _num(pick("fps_ci95_low", "fps_makespan_ci95_low", "performance_fps_ci95_low")),
        "fps_ci95_high": _num(pick("fps_ci95_high", "fps_makespan_ci95_high", "performance_fps_ci95_high")),
        "request_latency": pick("request_latency") or {},
        "latency_mean_ms": _num(pick("latency_median_ms", "latency_mean_ms")),
        "latency_median_ms": _num(pick("latency_median_ms", "latency_mean_ms")),
        "latency_p50_ms": _num(pick("latency_p50_ms")),
        "latency_p95_ms": _num(pick("latency_p95_ms")),
        "latency_ci95_low_ms": _num(pick("latency_ci95_low_ms")),
        "latency_ci95_high_ms": _num(pick("latency_ci95_high_ms")),
        "latency_semantics": str(pick("latency_semantics") or ""),
        "repetition_count_requested": int(_num(pick("repetition_count_requested", "repetitions_requested", "performance_repeat_count_requested")) or max(1, len(raw), len(samples))),
        "repetition_count_attempted": attempted,
        "repetition_count_valid": int(_num(pick("repetition_count_valid", "repetitions_completed", "performance_repeat_count_valid", "performance_repeat_n")) or len(samples)),
        "repetition_status": str(pick("repetition_status", "performance_repetition_status") or ("complete" if samples else "")),
        "repetition_aggregation": str(pick("repetition_aggregation", "performance_repetition_aggregation") or ""),
        "repetition_runtime_scope": str(pick("repetition_runtime_scope") or ""),
        "repetition_independence_verified": pick("repetition_independence_verified"),
        "fps_repetition_samples": samples,
        "latency_mean_repetition_samples_ms": latency_samples,
        "repetition_records": raw,
        "repetition_evidence": raw,
        "performance_repetitions": raw,
    }


def _claim_contract_fields(*sources: dict[str, Any]) -> dict[str, Any]:
    # Callers pass the summary first and the selected raw endpoint last. Keep
    # its selection as one tuple, including missing values; never splice a
    # repetition out of multiple reports or infer one from list order/mtime.
    selection_keys = (
        "semantic_evidence_repetition_index",
        "semantic_evidence_repetition_id",
        "semantic_evidence_runtime_instance_id",
    )
    selection_source = next((
        source for source in reversed(sources)
        if isinstance(source, Mapping) and (
            any(key in source for key in selection_keys)
            or isinstance(source.get("completed_task_endpoint_attestation"), Mapping)
            or isinstance(source.get("completion_execution_attestation"), Mapping)
        )
    ), {})
    selection = {key: selection_source.get(key) for key in selection_keys}
    def pick(key: str) -> Any:
        for source in sources:
            if isinstance(source, dict) and source.get(key) not in (None, ""):
                return source.get(key)
        return None
    def conflict(*keys: str) -> bool:
        observed: set[str] = set()
        for source in sources:
            if not isinstance(source, Mapping):
                continue
            for key in keys:
                value = source.get(key)
                if value in (None, ""):
                    continue
                try:
                    token = json.dumps(
                        value,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                except (TypeError, ValueError):
                    token = repr(value)
                observed.add(token)
        return len(observed) > 1

    completed_v2_projection_conflicts = [
        label
        for label, keys in (
            (
                "completion_execution_attestation",
                (
                    "completion_execution_attestation",
                    "completed_task_endpoint_attestation",
                ),
            ),
            (
                "completion_execution_contract",
                ("completion_execution_contract",),
            ),
            (
                "completion_execution_contract_sha256",
                ("completion_execution_contract_sha256",),
            ),
            (
                "completion_artifact_sha256",
                ("completion_artifact_sha256",),
            ),
            (
                "completion_schema_sha256",
                ("completion_schema_sha256",),
            ),
            (
                "completion_content_sha256",
                ("completion_content_sha256",),
            ),
            (
                "completion_invocation_sha256",
                ("completion_invocation_sha256",),
            ),
            (
                "completion_relation_sha256",
                ("completion_relation_sha256",),
            ),
            (
                "completed_task_endpoint_contract",
                ("completed_task_endpoint_contract",),
            ),
            (
                "completed_task_endpoint_contract_hash",
                ("completed_task_endpoint_contract_hash",),
            ),
            (
                "completed_task_output_endpoint_id",
                ("completed_task_output_endpoint_id",),
            ),
            (
                "completed_task_comparison_endpoint_contract",
                ("completed_task_comparison_endpoint_contract",),
            ),
            (
                "completed_task_comparison_endpoint_contract_hash",
                ("completed_task_comparison_endpoint_contract_hash",),
            ),
            (
                "completed_task_comparison_output_endpoint_id",
                ("completed_task_comparison_output_endpoint_id",),
            ),
            (
                "completed_work_units",
                ("completed_work_units", "completed_frames"),
            ),
            (
                "completion_observation_relation",
                ("completion_observation_relation",),
            ),
            (
                "completion_exact_result_claim_bound",
                ("completion_exact_result_claim_bound",),
            ),
            (
                "completed_task_result_artifact",
                ("completed_task_result_artifact",),
            ),
            (
                "completed_task_result_artifact_path",
                ("completed_task_result_artifact_path",),
            ),
            (
                "completed_task_result_artifact_saved",
                ("completed_task_result_artifact_saved",),
            ),
            (
                "completed_task_result_artifact_sha256",
                ("completed_task_result_artifact_sha256",),
            ),
            (
                "completed_task_result_artifact_file_sha256",
                ("completed_task_result_artifact_file_sha256",),
            ),
        )
        if conflict(*keys)
    ]
    completed_v2_projection_conflicts.extend(
        key for key in selection_keys if conflict(key)
    )
    raw_source_scope = pick("source_e2e_scope")
    evaluated_scope = pick("e2e_scope")
    source_scope = str(
        raw_source_scope
        if raw_source_scope is not None
        else evaluated_scope or ""
    ).strip()
    normalized_scope = {
        "accelerator_only": "accelerator_output_endpoint",
        "accelerator_output": "accelerator_output_endpoint",
    }.get(
        str(evaluated_scope or "").strip(),
        str(evaluated_scope or "").strip()
        or source_scope
        or "unavailable",
    )
    completed_attestation = pick("completed_task_endpoint_attestation")
    completed_attestation = (
        dict(completed_attestation)
        if isinstance(completed_attestation, Mapping)
        else None
    )
    completed_attested = _strict_bool(
        pick("completed_task_endpoint_attested")
    )
    if completed_attested is None and completed_attestation is not None:
        completed_attested = _strict_bool(
            completed_attestation.get("attested")
        )
    completed_attestation_status = str(
        pick("completed_task_endpoint_attestation_status")
        or (
            completed_attestation.get("status")
            if completed_attestation is not None else ""
        )
        or ""
    )
    completion_mode = str(
        pick("completed_task_completion_mode") or ""
    ).strip()
    completion_execution_attestation = pick(
        "completion_execution_attestation"
    )
    completion_execution_attestation = (
        dict(completion_execution_attestation)
        if isinstance(completion_execution_attestation, Mapping)
        else None
    )
    completion_execution_contract = pick(
        "completion_execution_contract"
    )
    if not isinstance(completion_execution_contract, Mapping):
        completion_execution_contract = None
        for source in sources:
            if not isinstance(source, Mapping):
                continue
            native_command = source.get("native_command_contract")
            runtime_options = (
                native_command.get("runtime_options")
                if isinstance(native_command, Mapping)
                and isinstance(
                    native_command.get("runtime_options"), Mapping,
                )
                else {}
            )
            candidate = runtime_options.get(
                "completion_execution_contract"
            )
            if isinstance(candidate, Mapping):
                completion_execution_contract = dict(candidate)
                break
    else:
        completion_execution_contract = dict(
            completion_execution_contract
        )

    # v2.72.0 Hailo-8 wrote one known pre-canonical mode name.  Both that
    # legacy shape and the canonical v2 producer store the cryptographic
    # object under ``completion_execution_attestation``; downstream consumers
    # also require the semantically identical completed-endpoint alias.  Only
    # create the aliases after full cryptographic verification.  Invalid
    # evidence is still projected under the v2 mode so the canonical consumer
    # rejects it instead of falling back to a permissive legacy raw-head gate.
    if completion_mode == "completed_detection_execution_contract":
        completion_mode = "detection_completion_execution_v1"
    if completion_mode == "detection_completion_execution_v1":
        supplied_attestation = (
            completed_attestation
            if completed_attestation is not None
            else completion_execution_attestation
        )
        try:
            if (
                _verify_detection_completion_execution_contract is None
                or _verify_detection_completion_execution_attestation is None
            ):
                raise RuntimeError(
                    "completion_execution_verifier_unavailable"
                )
            verified_execution = (
                _verify_detection_completion_execution_contract(
                    completion_execution_contract
                )
            )
            verified_attestation = (
                _verify_detection_completion_execution_attestation(
                    supplied_attestation,
                    execution_contract=verified_execution,
                    expected_observation_relation=(
                        "same_hotloop_sentinel"
                    ),
                )
            )
        except Exception:
            # Preserve the exact supplied objects.  Canonical verification in
            # host_postprocess will return the stable fail-closed status.
            pass
        else:
            completion_execution_contract = dict(verified_execution)
            completed_attestation = dict(verified_attestation)
            completion_execution_attestation = dict(
                verified_attestation
            )
            completed_attested = True
            completed_attestation_status = "passed"
    task_quality_policy_sha256 = str(
        pick("task_quality_policy_sha256") or ""
    )
    runtime_quality_gate_policy_sha256 = str(
        pick("runtime_quality_gate_policy_sha256") or ""
    )
    # The runtime name is a diagnostic compatibility alias, not an
    # independent claim binding.  Project the canonical task-policy digest
    # into that alias when an older producer omitted it, while preserving any
    # explicitly supplied value so a conflicting duplicate still fails closed
    # in the downstream validator.
    if (
        not runtime_quality_gate_policy_sha256
        and task_quality_policy_sha256
    ):
        runtime_quality_gate_policy_sha256 = task_quality_policy_sha256
    return {
        "task": str(pick("task") or ""),
        **selection,
        "stage": str(pick("stage") or ""),
        "output_format": str(pick("output_format") or ""),
        "contract_family": str(pick("contract_family") or ""),
        "contract_source": str(pick("contract_source") or ""),
        "endpoint_contract_complete": pick("endpoint_contract_complete"),
        "endpoint_contract_hash": str(
            pick("endpoint_contract_hash") or ""
        ),
        "output_endpoint_attestation": (
            dict(pick("output_endpoint_attestation"))
            if isinstance(pick("output_endpoint_attestation"), Mapping)
            else None
        ),
        "accelerator_output_stage": str(
            pick("accelerator_output_stage") or ""
        ),
        "accelerator_output_contract_family": str(
            pick("accelerator_output_contract_family") or ""
        ),
        "accelerator_endpoint_contract_hash": str(
            pick("accelerator_endpoint_contract_hash") or ""
        ),
        "accelerator_output_endpoint_attestation": (
            dict(pick("accelerator_output_endpoint_attestation"))
            if isinstance(
                pick("accelerator_output_endpoint_attestation"), Mapping,
            )
            else None
        ),
        "output_endpoint_id": str(pick("output_endpoint_id") or ""),
        "physical_output_endpoint_id": str(
            pick("physical_output_endpoint_id")
            or pick("output_endpoint_id")
            or ""
        ),
        "comparison_output_endpoint_id": str(
            pick("comparison_output_endpoint_id") or ""
        ),
        "output_endpoint_match": pick("output_endpoint_match"),
        "comparison_endpoint_match": pick("comparison_endpoint_match"),
        "output_endpoint_comparison_stratum": pick(
            "output_endpoint_comparison_stratum"
        ),
        "comparison_stratum_explicit": pick(
            "comparison_stratum_explicit"
        ),
        "claim_ok": pick("claim_ok"),
        "claim_ok_source": pick("claim_ok_source"),
        "claim_structural_gate_pass": pick(
            "claim_structural_gate_pass"
        ),
        "claim_structural_gate_reason": str(
            pick("claim_structural_gate_reason") or ""
        ),
        "claim_ok_structural_clamped": pick(
            "claim_ok_structural_clamped"
        ),
        "semantic_ok": pick("semantic_ok"),
        "contract_consistent": pick("contract_consistent"),
        "structural_contract_pass": pick("structural_contract_pass"),
        "structural_contract_status": str(
            pick("structural_contract_status") or ""
        ),
        "structural_contract_reason": str(
            pick("structural_contract_reason") or ""
        ),
        "numerical_similarity_pass": pick("numerical_similarity_pass"),
        "numerical_similarity_status": str(
            pick("numerical_similarity_status") or ""
        ),
        "numerical_similarity_reason": str(
            pick("numerical_similarity_reason") or ""
        ),
        "numerical_similarity_scope": str(
            pick("numerical_similarity_scope") or ""
        ),
        "numerical_similarity_metric": str(
            pick("numerical_similarity_metric") or ""
        ),
        "numerical_similarity_value": pick("numerical_similarity_value"),
        "numerical_similarity_threshold": pick(
            "numerical_similarity_threshold"
        ),
        "numerical_similarity_mean_iou": pick(
            "numerical_similarity_mean_iou"
        ),
        "numerical_similarity_mean_iou_threshold": pick(
            "numerical_similarity_mean_iou_threshold"
        ),
        "numerical_similarity_policy_id": str(
            pick("numerical_similarity_policy_id") or ""
        ),
        "task_quality_pass": pick("task_quality_pass"),
        "task_quality_status": str(pick("task_quality_status") or ""),
        "task_quality_reason": str(pick("task_quality_reason") or ""),
        "source_e2e_scope": source_scope,
        "e2e_scope": normalized_scope,
        "e2e_claim_eligible": (
            pick("e2e_claim_eligible")
            if pick("e2e_claim_eligible") is not None
            else pick("claim_eligible_e2e")
        ),
        "e2e_contract_reason": str(
            pick("e2e_contract_reason") or ""
        ),
        "comparison_endpoint_stratum": str(
            pick("comparison_endpoint_stratum") or ""
        ),
        "measurement_concurrency": pick("measurement_concurrency"),
        "requires_host_decode_nms": pick("requires_host_decode_nms"),
        "postprocess_included": pick("postprocess_included"),
        "postprocess_location": str(pick("postprocess_location") or ""),
        "host_postprocess_frozen": pick("host_postprocess_frozen"),
        "host_postprocessing_available": pick(
            "host_postprocessing_available"
        ),
        "host_tail_available": pick("host_tail_available"),
        "host_postprocess_required": pick("host_postprocess_required"),
        "host_tail_required": pick("host_tail_required"),
        "host_postprocessing_evidence_status": str(
            pick("host_postprocessing_evidence_status") or ""
        ),
        "host_postprocessing_evidence_source": str(
            pick("host_postprocessing_evidence_source") or ""
        ),
        "host_postprocessing_legacy_alias_conflict": pick(
            "host_postprocessing_legacy_alias_conflict"
        ),
        "decoder_contract_pass": pick("decoder_contract_pass"),
        "nms_ok": pick("nms_ok"),
        "decoder_id": str(pick("decoder_id") or ""),
        "postprocess_completed_frames": pick(
            "postprocess_completed_frames"
        ),
        "postprocess_completion_verified": pick(
            "postprocess_completion_verified"
        ),
        "frozen_host_postprocess_contract": (
            dict(pick("frozen_host_postprocess_contract"))
            if isinstance(pick("frozen_host_postprocess_contract"), Mapping)
            else None
        ),
        "frozen_host_postprocess_contract_sha256": str(
            pick("frozen_host_postprocess_contract_sha256")
            or pick("frozen_postprocess_contract_sha256")
            or ""
        ),
        "frozen_host_postprocess_result": (
            dict(pick("frozen_host_postprocess_result"))
            if isinstance(pick("frozen_host_postprocess_result"), Mapping)
            else None
        ),
        "completed_task_stage": str(
            pick("completed_task_stage") or ""
        ),
        "completed_task_contract_family": str(
            pick("completed_task_contract_family") or ""
        ),
        "completed_task_endpoint_contract": (
            dict(pick("completed_task_endpoint_contract"))
            if isinstance(pick("completed_task_endpoint_contract"), Mapping)
            else None
        ),
        "completed_task_endpoint_contract_hash": str(
            pick("completed_task_endpoint_contract_hash") or ""
        ),
        "completed_task_output_endpoint_id": str(
            pick("completed_task_output_endpoint_id") or ""
        ),
        "completed_task_comparison_endpoint_contract": (
            dict(pick("completed_task_comparison_endpoint_contract"))
            if isinstance(
                pick("completed_task_comparison_endpoint_contract"),
                Mapping,
            )
            else None
        ),
        "completed_task_comparison_endpoint_contract_hash": str(
            pick("completed_task_comparison_endpoint_contract_hash") or ""
        ),
        "completed_task_comparison_output_endpoint_id": str(
            pick("completed_task_comparison_output_endpoint_id") or ""
        ),
        "completed_task_completion_mode": completion_mode,
        "completed_task_endpoint_attested": completed_attested,
        "completed_task_endpoint_attestation": completed_attestation,
        "completed_task_endpoint_attestation_status":
            completed_attestation_status,
        "completed_task_result_artifact": (
            dict(pick("completed_task_result_artifact"))
            if isinstance(
                pick("completed_task_result_artifact"), Mapping,
            )
            else None
        ),
        "completed_task_result_artifact_path": str(
            pick("completed_task_result_artifact_path") or ""
        ),
        "completed_task_result_artifact_saved": _strict_bool(
            pick("completed_task_result_artifact_saved")
        ),
        "completed_task_result_artifact_sha256": str(
            pick("completed_task_result_artifact_sha256") or ""
        ),
        "completed_task_result_artifact_file_sha256": str(
            pick("completed_task_result_artifact_file_sha256") or ""
        ),
        "completion_execution_contract": (
            completion_execution_contract
        ),
        "completion_execution_contract_sha256": str(
            pick("completion_execution_contract_sha256")
            or (
                completion_execution_contract.get("contract_sha256")
                if isinstance(
                    completion_execution_contract, Mapping,
                )
                else ""
            )
            or ""
        ),
        "completion_execution_attestation": (
            completion_execution_attestation
        ),
        "completed_work_units": pick("completed_work_units"),
        "completed_frames": pick("completed_frames"),
        "completion_observation_relation": str(
            pick("completion_observation_relation") or ""
        ),
        "completion_exact_result_claim_bound": pick(
            "completion_exact_result_claim_bound"
        ),
        "completion_artifact_sha256": str(
            pick("completion_artifact_sha256") or ""
        ),
        "completion_schema_sha256": str(
            pick("completion_schema_sha256") or ""
        ),
        "completion_content_sha256": str(
            pick("completion_content_sha256") or ""
        ),
        "completion_invocation_sha256": str(
            pick("completion_invocation_sha256") or ""
        ),
        "completion_relation_sha256": str(
            pick("completion_relation_sha256") or ""
        ),
        "completed_v2_projection_conflicts": (
            completed_v2_projection_conflicts
        ),
        "precision_quality_verified": pick("precision_quality_verified"),
        "precision_quality_binding_verified": pick(
            "precision_quality_binding_verified"
        ),
        "task_quality_observation_valid": pick(
            "task_quality_observation_valid"
        ),
        "quality_claim_result_verified": pick(
            "quality_claim_result_verified"
        ),
        "model_sha256": str(pick("model_sha256") or pick("source_onnx_sha256") or ""),
        "validation_dataset_sha256": str(pick("validation_dataset_sha256") or pick("dataset_sha256") or ""),
        "validation_dataset_image_ids_sha256": str(
            pick("validation_dataset_image_ids_sha256")
            or pick("validation_image_ids_sha256")
            or pick("dataset_image_ids_sha256")
            or pick("image_ids_sha256") or ""
        ),
        "validation_dataset_ground_truth_sha256": str(
            pick("validation_dataset_ground_truth_sha256")
            or pick("validation_ground_truth_sha256")
            or pick("dataset_ground_truth_sha256")
            or pick("ground_truth_sha256") or ""
        ),
        "source_request_sha256": str(pick("source_request_sha256") or ""),
        "accuracy_gate_policy_sha256": str(pick("accuracy_gate_policy_sha256") or ""),
        "task_quality_policy_sha256": task_quality_policy_sha256,
        "runtime_quality_gate_policy_sha256": (
            runtime_quality_gate_policy_sha256
        ),
        "quality_contract_sha256": str(pick("quality_contract_sha256") or ""),
        "preprocessing_contract_sha256": str(pick("preprocessing_contract_sha256") or ""),
        "decoder_contract_sha256": str(pick("decoder_contract_sha256") or ""),
        "nms_contract_sha256": str(pick("nms_contract_sha256") or ""),
        "quality_record_endpoint_contract_sha256": str(
            pick("quality_record_endpoint_contract_sha256") or ""
        ),
    }


def _native_identity_fields(*sources: dict[str, Any]) -> dict[str, Any]:
    """Prefer documented planned identity; command fallback remains explicit."""
    from onnx_splitpoint_tool.native_job_identity import (
        attach_identity_without_conflicts, native_comparison,
    )
    planned = next((item['planned_native_identity'] for item in sources
                    if isinstance(item, Mapping) and isinstance(item.get('planned_native_identity'), Mapping)), None)
    if planned is not None:
        merged: dict[str, Any] = {}
        for source in sources:
            if not isinstance(source, Mapping): continue
            attached = attach_identity_without_conflicts(planned, source)
            if attached.get('identity_conflicts'):
                return {key: attached[key] for key in (
                    'planned_native_identity', 'setup_id', 'comparison_backend',
                    'identity_conflicts', 'failure_reason', 'status', 'ok',
                ) if key in attached}
            merged.update({key: attached[key] for key in (
                'planned_native_identity', 'setup_id', 'comparison_backend',
            ) if key in attached})
        return merged
    setup_id = ''; comparison_backend = ''
    for source in sources:
        if not isinstance(source, Mapping): continue
        contract = source.get('native_command_contract') or {}
        if not setup_id:
            setup_id = str(source.get('setup_id') or contract.get('setup_id') or '')
        if not comparison_backend:
            comparison_backend = native_comparison(source.get('comparison_backend') or contract.get('comparison_backend') or '')
    return {'setup_id': setup_id, 'comparison_backend': comparison_backend}


def _verified_native_producer_impl(*sources: dict[str, Any]) -> str:
    """Return only one producer implementation bound by verified row evidence."""
    if _strict_verify_native_command_contract is None:
        return ''
    verified_values: list[str] = []
    explicit_values: list[str] = []
    saw_contract = False
    for source in sources:
        if not isinstance(source, dict):
            continue
        explicit = str(source.get('producer_impl') or '').strip()
        if explicit:
            explicit_values.append(explicit)
        raw = source.get('native_command_contract')
        if not isinstance(raw, Mapping):
            continue
        saw_contract = True
        verified, _status = _strict_verify_native_command_contract(raw)
        if verified is None:
            return ''
        declared = str(verified.get('contract_sha256') or '').strip().lower()
        for field in (
            'native_command_contract_sha256', 'workload_contract_sha256',
        ):
            reference = str(source.get(field) or '').strip().lower()
            if reference and reference != declared:
                return ''
        options = verified.get('runtime_options')
        implementation = str(
            options.get('producer_impl') if isinstance(options, Mapping)
            else ''
        ).strip()
        if not implementation:
            return ''
        verified_values.append(implementation)
    if not saw_contract or not verified_values:
        return ''
    identities = {*verified_values, *explicit_values}
    return verified_values[0] if len(identities) == 1 else ''


def _split_quality_fields(*sources: dict[str,Any]) -> dict[str,Any]:
    """Carry split-QF evidence without silently choosing a conflicting copy."""
    expanded=[]
    for source in sources:
        if not isinstance(source,dict): continue
        expanded.append(source)
        command=source.get('native_command_contract')
        if isinstance(command,dict): expanded.append(command)
    fields=(
        'eval_run_id','source_run_id','native_split_quality_binding',
        'native_split_quality_binding_sha256','native_split_quality_eval_run_id',
        'native_split_quality_source_run_id','source_request_sha256',
        'native_split_quality_source_request_sha256',
        'native_split_quality_central_result_sha256',
        'native_split_quality_selection_sha256',
        'native_split_quality_cache_verify_source_binding_sha256',
        'native_split_quality_cache_verify_replay_sha256',
        'native_split_quality_consumer_attestation',
        'native_split_quality_consumer_status','workload_contract_sha256',
        'performance_claims_emitted','execution_role',
    )
    quality_markers=(
        'native_split_quality_binding','native_split_quality_binding_sha256',
        'native_split_quality_eval_run_id','native_split_quality_source_run_id',
        'native_split_quality_source_request_sha256',
        'native_split_quality_central_result_sha256',
        'native_split_quality_selection_sha256',
        'native_split_quality_cache_verify_source_binding_sha256',
        'native_split_quality_cache_verify_replay_sha256',
        'native_split_quality_consumer_attestation',
        'native_split_quality_consumer_status',
    )
    # Determine whether the row belongs to the Quality-first path *before*
    # duplicate evidence is collapsed.  Otherwise two conflicting copies can
    # make every marker disappear from ``result`` and silently downgrade a
    # Quality-first row to the legacy path.
    marked=any(
        source.get('native_split_quality_required') is True
        or any(source.get(field) not in (None,'',{}) for field in quality_markers)
        for source in expanded
    )
    result={}; conflicts=[]
    for field in fields:
        values=[source.get(field) for source in expanded if source.get(field) not in (None,'',{})]
        if not values: continue
        if field in {'source_run_id', 'native_split_quality_source_run_id'}:
            canonical=[
                (
                    _canonical_native_split_backend(value)
                    if _canonical_native_split_backend is not None
                    else str(value or '').strip().lower().replace('-', '_')
                    .replace('_to_tensorrt', '_to_trt')
                    .replace('hailo10_to_trt', 'hailo10h_to_trt')
                    .replace('deepx_m1_to_trt', 'deepx_to_trt')
                )
                for value in values
            ]
        else:
            canonical=[json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False) if isinstance(value,(dict,list)) else str(value) for value in values]
        if len(set(canonical)) != 1:
            conflicts.append(field); continue
        result[field]=canonical[0] if field in {
            'source_run_id', 'native_split_quality_source_run_id',
        } else values[0]
    result['native_split_quality_required']=marked
    result['native_split_quality_provenance_conflict']=bool(conflicts)
    result['native_split_quality_provenance_conflict_fields']=conflicts
    if marked and conflicts:
        result['native_split_quality_consumer_status']='conflicting_duplicate_evidence'
    return result


def _native_split_authority(outdir: Path) -> dict[str, Any]:
    eval_root = outdir.parent if outdir.name == 'reports' else outdir
    for candidate in (outdir, *outdir.parents):
        if (candidate / 'run_manifest.json').is_file():
            eval_root = candidate
            break
    if _resolve_split_quality_authority is None:
        return {
            'schema': 'onnx-splitpoint/native-split-quality-authority',
            'schema_version': 1,
            'mode': 'invalid',
            'valid': False,
            'native_split_quality_required': True,
            'workflow_version': '',
            'run_id': '',
            'errors': ['native_split_quality_authority_verifier_unavailable'],
        }
    return _resolve_split_quality_authority(
        run_manifest_path=eval_root / 'run_manifest.json',
        stage_path=eval_root / 'reports' / 'native_producer_stage.json',
    )


def _apply_native_split_authority_to_rows(
    rows: Iterable[dict[str, Any]], authority: Mapping[str, Any],
) -> None:
    for row in rows:
        split_backend = (
            bool(_is_native_split_backend(row.get('backend')))
            if _is_native_split_backend is not None else
            str(row.get('backend') or '').strip().lower()
            in {'hailo8_to_trt', 'hailo10h_to_trt', 'deepx_to_trt'}
        )
        if not split_backend:
            continue
        if _apply_split_quality_authority is None:
            row['native_split_quality_required'] = True
            row['native_split_quality_binding_required'] = True
            row['native_split_quality_authority_valid'] = False
            row['native_split_quality_authority_errors'] = [
                'native_split_quality_authority_verifier_unavailable'
            ]
        else:
            _apply_split_quality_authority(row, authority)
        if row.get('native_split_quality_required') is True:
            binding = row.get('native_split_quality_binding')
            binding = binding if isinstance(binding, Mapping) else {}
            receipt = binding.get('central_quality_selection')
            receipt = receipt if isinstance(receipt, Mapping) else {}
            expected_eval = str(authority.get('run_id') or '').strip()
            eval_values = (
                str(row.get('eval_run_id') or '').strip(),
                str(row.get('native_split_quality_eval_run_id') or '').strip(),
                str(binding.get('eval_run_id') or '').strip(),
                str(receipt.get('eval_run_id') or '').strip(),
            )
            if (
                authority.get('valid') is not True
                or not expected_eval
                or any(value != expected_eval for value in eval_values)
            ):
                row['native_split_semantic_binding_required'] = True
                row['native_split_semantic_binding_valid'] = False
                row['native_split_semantic_binding_status'] = (
                    'authority_eval_run_id_mismatch'
                )
                row['native_split_final_portable_binding_valid'] = False
                row['native_split_final_portable_binding_status'] = (
                    'native_split_quality_authority_eval_run_id_mismatch'
                )
        if row.get('native_split_quality_required') is True and (
            row.get('native_split_semantic_binding_required') is not True
        ):
            row['native_split_semantic_binding_required'] = True
            row['native_split_semantic_binding_valid'] = False
            row['native_split_semantic_binding_status'] = (
                'authority_required_quality_binding_missing'
            )
            row['native_split_final_portable_binding_valid'] = False
            row['native_split_final_portable_binding_status'] = (
                'authority_required_quality_binding_missing'
            )


def _split_output_contract(
    result_path: Path | None,
    *sources: dict[str, Any],
    fallback_relpaths: Iterable[str] = (),
) -> dict[str, Any]:
    """Load the measured Split output endpoint from its dump manifest.

    Runtime result files often carry only a remote absolute manifest path.  A
    copied evaluation root therefore also checks backend-specific paths next to
    the locally collected result.  Endpoint metadata is deliberately *not*
    inferred from tensor shapes or from a model name: an absent/incomplete
    manifest remains an unverified endpoint and fails the comparison gate.
    """
    manifest_path, resolution = _resolve_split_semantic_manifest(
        result_path, sources, kind="output", fallback_relpaths=fallback_relpaths,
    )
    manifest = _load_strict_json_object(manifest_path) if manifest_path else None
    manifest = manifest if isinstance(manifest, dict) else {}
    task = str(manifest.get("task") or "").strip()
    output_format = str(manifest.get("output_format") or "").strip()
    contract_family = str(manifest.get("contract_family") or "").strip()
    stage = str(manifest.get("stage") or "").strip()
    endpoint_contract_hash = str(manifest.get("endpoint_contract_hash") or "").strip().lower()
    complete = bool(
        manifest.get("endpoint_contract_complete") is True
        and resolution["manifest_resolution_status"] in ("manifest_verified", "legacy_manifest_found")
        and task and stage and output_format and contract_family
        and len(endpoint_contract_hash) == 64
        and all(ch in "0123456789abcdef" for ch in endpoint_contract_hash)
    )
    output_manifest_sha256 = ""
    if manifest_path and manifest_path.is_file():
        digest = hashlib.sha256()
        with manifest_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        output_manifest_sha256 = digest.hexdigest()
    semantic_binding = _verify_split_semantic_artifacts(
        result_path=result_path, manifest_path=manifest_path,
        manifest_sha256=output_manifest_sha256, sources=sources,
    )
    if resolution["manifest_resolution_status"] not in (
        "manifest_verified", "legacy_manifest_found", "missing"
    ):
        semantic_binding.update({
            "native_split_semantic_binding_valid": False,
            "native_split_semantic_binding_status": resolution["manifest_resolution_status"],
        })
    return {
        "output_dump_manifest": str(manifest_path) if manifest_path else "",
        "native_output_manifest": str(manifest_path) if manifest_path else "",
        **resolution,
        "output_contract_manifest_status": (
            "content_invalid" if resolution["manifest_resolution_status"] not in (
                "manifest_verified", "legacy_manifest_found", "missing"
            ) else
            "explicit_complete" if complete else
            "explicit_incomplete" if manifest_path else
            "missing"
        ),
        "task": task,
        "stage": stage,
        "output_format": output_format,
        "contract_family": contract_family,
        "contract_source": str(manifest.get("contract_source") or "").strip(),
        "endpoint_contract_complete": complete,
        "endpoint_contract_hash": endpoint_contract_hash,
        "output_manifest_sha256": output_manifest_sha256,
        "output_endpoint_attestation": manifest.get("output_endpoint_attestation")
        if isinstance(manifest.get("output_endpoint_attestation"), dict) else {},
        **semantic_binding,
    }


def _hash_file(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''): digest.update(chunk)
    return digest.hexdigest()


def _load_strict_json_object(path: Path) -> dict[str,Any] | None:
    duplicate=False
    def _object(pairs: list[tuple[str,Any]]) -> dict[str,Any]:
        nonlocal duplicate
        value: dict[str,Any]={}
        for key,item in pairs:
            if key in value: duplicate=True
            value[key]=item
        return value
    try:
        parsed=json.loads(path.read_text(encoding='utf-8'),object_pairs_hook=_object)
    except Exception:
        return None
    return parsed if isinstance(parsed,dict) and not duplicate else None


def _resolve_split_semantic_manifest(
    result_path: Path | None, sources: Iterable[Mapping[str, Any]], *,
    kind: str, fallback_relpaths: Iterable[str] = (),
) -> tuple[Path | None, dict[str, Any]]:
    """Resolve one job's declared role without searching other runs or roles.

    Remote absolute paths remain provenance. Only their exact command-bound
    artifact is mapped into this collected job directory; sealed JSON is never
    edited. Manifest existence is deliberately weaker than payload validity.
    """
    source_list = [item for item in sources if isinstance(item, Mapping)]
    commands = [item["native_command_contract"] for item in source_list
                if isinstance(item.get("native_command_contract"), Mapping)]
    command = commands[0] if commands else {}
    options = command.get("runtime_options") or {}
    attests = [item["native_split_quality_consumer_attestation"] for item in source_list
               if isinstance(item.get("native_split_quality_consumer_attestation"), Mapping)]
    contexts = [*source_list, *commands, *attests]
    roles = {str(item.get("measurement_endpoint") or "") for item in [*contexts, options]
             if isinstance(item, Mapping) and item.get("measurement_endpoint")}
    role = next(iter(roles)) if len(roles) == 1 else ""
    info = {"manifest_expected_role": role or "legacy_flat",
            "manifest_resolution_source": "", "manifest_resolution_status": "missing",
            "resolved_local_path": ""}
    def fail(reason: str):
        info["manifest_resolution_status"] = reason
        return None, info
    if len(roles) > 1 or (role and role not in ("completed_task", "raw_model_outputs", "model_outputs")):
        return fail("manifest_endpoint_role_conflict")
    if commands and any(dict(item) != dict(command) for item in commands[1:]):
        return fail("manifest_command_context_conflict")
    if role == "model_outputs":
        # Hailo8 classification records this explicit endpoint in its flat
        # job directory. It is not an alias for either detection endpoint.
        tasks = {str(item.get("task")) for item in [*contexts, options]
                 if isinstance(item, Mapping) and item.get("task")}
        if (not command or not isinstance(options, Mapping)
                or command.get("backend") != "hailo8_to_trt"
                or options.get("task") != "classification"
                or tasks != {"classification"}):
            return fail("manifest_endpoint_role_conflict")
        if _strict_verify_native_command_contract is None:
            return fail("manifest_command_verification_unavailable")
        verified, _status = _strict_verify_native_command_contract(command)
        if verified is None:
            return fail("manifest_command_contract_invalid")
    for names in (("model", "model_id"), ("case", "case_id"), ("setup_id",),
                  ("comparison_backend",), ("precision",), ("eval_run_id", "native_split_quality_eval_run_id")):
        values = {str(item.get(name)) for item in contexts for name in names
                  if item.get(name) not in (None, "")}
        if len(values) > 1:
            return fail("manifest_job_identity_conflict:" + names[0])
    if result_path is None:
        return fail("missing")
    root = Path(result_path).parent.resolve()
    if root.name in ("completed_task", "raw_model_outputs") and root.parent.name == "endpoints":
        if role and role != root.name:
            return fail("manifest_endpoint_role_conflict")
        role = role or root.name
        info["manifest_expected_role"] = role
        root = root.parent.parent
    # A normal collected native_pipeline path itself binds case and precision.
    if root.parent.parent.parent.name == "native_pipeline":
        if command.get("case") and str(command["case"]) != root.parent.parent.name:
            return fail("manifest_job_path_case_mismatch")
        if command.get("precision") and str(command["precision"]) != root.name:
            return fail("manifest_job_path_precision_mismatch")
        if command.get("model") and str(command["model"]) != root.parents[4].name:
            return fail("manifest_job_path_model_mismatch")
    artifact = (command.get("artifacts") or {}).get("semantic_" + kind + "_manifest") or {}
    expected_path = str(artifact.get("path") or "")
    expected_hashes = {str(item).lower() for item in [artifact.get("sha256"),
        *[att.get("semantic_" + kind + "_manifest_sha256") for att in attests]] if item}
    if len(expected_hashes) > 1:
        return fail("semantic_" + kind + "_manifest_expected_hash_conflict")
    expected_sha = next(iter(expected_hashes), "")
    if role == "model_outputs" and (
        not expected_path or "endpoints" in Path(expected_path).parts
        or len(expected_sha) != 64
        or any(ch not in "0123456789abcdef" for ch in expected_sha)
    ):
        return fail("manifest_classification_flat_binding_required")
    fields = (("native_fifo_output_manifest", "native_output_manifest", "output_manifest", "output_dump_manifest", "native_split_semantic_output_manifest")
              if kind == "output" else ("native_fifo_boundary_manifest", "boundary_manifest", "native_split_semantic_boundary_manifest"))
    raw_paths = [str(item.get(field)) for item in source_list for field in fields if item.get(field)]
    if expected_path:
        raw_paths.append(expected_path)
    candidates: dict[Path, str] = {}
    invalid = ""
    def add(path: Path, origin: str):
        nonlocal invalid
        if ".." in path.parts:
            invalid = "manifest_path_traversal"
            return
        resolved = path.resolve()
        try:
            rel = resolved.relative_to(root)
        except ValueError:
            invalid = "manifest_path_outside_collection_root"
            return
        if "endpoints" in rel.parts:
            index = rel.parts.index("endpoints")
            if (role == "model_outputs" or index + 1 >= len(rel.parts)
                    or not role or rel.parts[index + 1] != role):
                invalid = "manifest_endpoint_role_conflict"
                return
        if path.is_file():
            candidates.setdefault(resolved, origin)
    for raw in raw_paths:
        path = Path(raw)
        if ".." in path.parts:
            return fail("manifest_path_traversal")
        if not path.is_absolute():
            add(root / path, "explicit_local_relative")
        elif path.is_relative_to(root):
            add(path, "explicit_local")
        elif expected_path and raw == expected_path:
            # The exact recorded command artifact maps by its job-relative
            # role suffix, not by recursively matching a basename.
            parts = path.parts
            if "endpoints" in parts:
                i = parts.index("endpoints")
                add(root.joinpath(*parts[i:]), "command_remote_role_mapping")
            else:
                add(root / path.parent.name / path.name, "command_remote_legacy_mapping")
        elif not expected_path:
            # Unbound historical remote paths are not opened locally.
            continue
        else:
            return fail("manifest_remote_path_context_mismatch")
    if role and role != "model_outputs":
        names = (("native_fifo_outputs/native_fifo_output_manifest.json", "native_fifo_outputs/native_fifo_outputs_manifest.json", "native_outputs/native_outputs_manifest.json")
                 if kind == "output" else ("native_fifo_boundary/native_fifo_boundary_manifest.json", "native_boundary/native_fifo_boundary_manifest.json"))
        for name in names:
            add(root / "endpoints" / role / name, "known_role_local")
    for rel in fallback_relpaths:
        add(root / rel, "legacy_flat")
    if invalid:
        return fail(invalid)
    if len(candidates) > 1:
        return fail("manifest_multiple_local_sources")
    if not candidates:
        return fail("missing")
    path, origin = next(iter(candidates.items()))
    info.update(manifest_resolution_source=origin, resolved_local_path=str(path))
    manifest = _load_strict_json_object(path)
    if manifest is None:
        info["manifest_resolution_status"] = "manifest_invalid_json_or_duplicate_keys"
        return path, info
    if expected_sha and _hash_file(path) != expected_sha:
        info["manifest_resolution_status"] = "semantic_" + kind + "_manifest_sha256_mismatch"
        return path, info
    if role == "model_outputs":
        preprocess = manifest.get("preprocess")
        manifest_task = (manifest.get("task") if kind == "output"
                         else preprocess.get("task") if isinstance(preprocess, Mapping) else None)
        if manifest_task != "classification":
            return fail("manifest_endpoint_role_conflict")
    info["manifest_resolution_status"] = "manifest_verified" if expected_sha else "legacy_manifest_found"
    return path, info


def _verify_manifest_payload_files(path: Path | None) -> tuple[bool,str]:
    if path is None or not path.is_file(): return False,'manifest_missing'
    payload=_load_strict_json_object(path)
    if payload is None: return False,'manifest_invalid_json_or_duplicate_keys'
    rows=payload.get('payload_artifacts')
    if not isinstance(rows,list) or not rows: return False,'payload_artifacts_missing'
    root=path.parent.resolve()
    for index,row in enumerate(rows):
        if not isinstance(row,dict): return False,f'payload_artifact_{index}_invalid'
        original=Path(str(row.get('path') or ''))
        if not str(row.get('path') or '') or '..' in original.parts:
            return False,f'payload_artifact_{index}_path_invalid'
        # Absolute paths in a sealed remote manifest are mapped only into its
        # own manifest directory. Never consume an unrelated existing remote
        # path from the local machine.
        actual=root/original.name if original.is_absolute() else root/original
        if not actual.resolve().is_relative_to(root):
            return False,f'payload_artifact_{index}_path_outside_manifest_root'
        if not actual.is_file(): return False,f'payload_artifact_{index}_missing'
        if type(row.get('size_bytes')) is not int or actual.stat().st_size != row['size_bytes']:
            return False,f'payload_artifact_{index}_size_mismatch'
        if _hash_file(actual) != str(row.get('sha256') or '').lower(): return False,f'payload_artifact_{index}_sha256_mismatch'
    canonical=json.dumps(rows,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode('utf-8')
    if hashlib.sha256(canonical).hexdigest() != str(payload.get('payload_artifacts_sha256') or '').lower():
        return False,'payload_artifacts_set_sha256_mismatch'
    return True,'payload_files_rehashed'


def _verify_split_semantic_artifacts(
    *, result_path: Path | None, manifest_path: Path | None,
    manifest_sha256: str, sources: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    result = _verify_split_semantic_artifacts_impl(
        result_path=result_path, manifest_path=manifest_path,
        manifest_sha256=manifest_sha256, sources=sources,
    )
    if result.get('native_split_semantic_binding_required') is True and result.get('native_split_semantic_binding_valid') is not True:
        result.setdefault('native_split_final_portable_binding_valid', False)
        result.setdefault('native_split_final_portable_binding_status', result.get('native_split_semantic_binding_status', 'unverified'))
    return result


def _verify_split_semantic_artifacts_impl(
    *, result_path: Path | None, manifest_path: Path | None,
    manifest_sha256: str, sources: Iterable[dict[str,Any]],
) -> dict[str,Any]:
    source_list=[source for source in sources if isinstance(source,dict)]
    commands=[]; attestations=[]; bindings=[]; binding_marked=False; boundary_candidates=[]
    for source in source_list:
        command=source.get('native_command_contract')
        if isinstance(command,dict):
            commands.append(command)
            command_binding=command.get('native_split_quality_binding')
            if isinstance(command_binding,dict) and command_binding: bindings.append(command_binding)
        binding=source.get('native_split_quality_binding')
        if isinstance(binding,dict) and binding: bindings.append(binding)
        att=source.get('native_split_quality_consumer_attestation')
        if isinstance(att,dict) and att: attestations.append(att)
        binding_marked=binding_marked or bool(source.get('native_split_quality_binding_sha256') or source.get('native_split_quality_binding'))
        boundary_candidates.append(source.get('native_fifo_boundary_manifest') or source.get('boundary_manifest'))
    if not binding_marked:
        return {'native_split_semantic_binding_required':False}
    if not commands or any(command != commands[0] for command in commands[1:]):
        return {'native_split_semantic_binding_required':True,'native_split_semantic_binding_valid':False,'native_split_semantic_binding_status':'command_contract_missing_or_conflicting'}
    if not attestations or any(att != attestations[0] for att in attestations[1:]):
        return {'native_split_semantic_binding_required':True,'native_split_semantic_binding_valid':False,'native_split_semantic_binding_status':'consumer_attestation_missing_or_conflicting'}
    if not bindings or any(binding != bindings[0] for binding in bindings[1:]):
        return {'native_split_semantic_binding_required':True,'native_split_semantic_binding_valid':False,'native_split_semantic_binding_status':'quality_binding_missing_or_conflicting','native_split_final_portable_binding_valid':False,'native_split_final_portable_binding_status':'quality_binding_missing_or_conflicting'}
    command=commands[0]; attestation=attestations[0]
    binding=bindings[0]
    selection=binding.get('preselection') if isinstance(binding.get('preselection'),dict) else {}
    def _identity_value(*names: str) -> Any:
        for source in (*source_list,command,selection):
            for name in names:
                value=source.get(name) if isinstance(source,dict) else None
                if value not in (None,''): return value
        return ''
    selected_hashes={
        'source_request_sha256':str(binding.get('source_request_sha256') or '').strip().lower(),
        'native_split_quality_source_request_sha256':str(binding.get('source_request_sha256') or '').strip().lower(),
        'native_split_quality_central_result_sha256':str(binding.get('central_result_sha256') or '').strip().lower(),
        'native_split_quality_selection_sha256':str(binding.get('central_quality_selection_sha256') or '').strip().lower(),
    }
    selected_identity_sources=[*source_list,command,attestation]
    for field,expected in selected_hashes.items():
        values=[
            str(source.get(field) or '').strip().lower()
            for source in selected_identity_sources
            if isinstance(source,dict) and source.get(field) not in (None,'')
        ]
        if not expected or not values or any(value != expected for value in values):
            return {
                'native_split_semantic_binding_required':True,
                'native_split_semantic_binding_valid':False,
                'native_split_semantic_binding_status':(
                    f'{field}_missing_or_conflicting'
                ),
                'native_split_final_portable_binding_valid':False,
                'native_split_final_portable_binding_status':(
                    f'native_split_quality_{field}_missing_or_conflicting'
                ),
            }
    portable_row={
        # ``backend`` in producer analysis tables names the Part-1 device
        # family (``deepx``/``hailo10h``), not the sealed Split pipeline.
        # The portable Quality join must use the pipeline source identity.
        'backend':(
            _canonical_native_split_backend(
                _identity_value('source_run_id')
            )
            if (
                _canonical_native_split_backend is not None
                and _identity_value('source_run_id')
            )
            else str(
                command.get('backend')
                or selection.get('backend')
                or _identity_value('backend')
                or ''
            )
        ),
        'model':_identity_value('model','model_id'),
        'case':_identity_value('case','case_id'),
        'precision':_identity_value('runtime_precision_identity','execution_precision','precision'),
        'setup_id':_identity_value('setup_id'),
        'task':_identity_value('task'),
        'comparison_backend':_identity_value('comparison_backend'),
        'native_command_contract':command,
        'native_command_contract_sha256':str(command.get('contract_sha256') or ''),
        'native_split_quality_binding_sha256':str(binding.get('binding_sha256') or ''),
        'native_split_quality_eval_run_id':str(binding.get('eval_run_id') or ''),
        'native_split_quality_source_run_id':(
            _canonical_native_split_backend(binding.get('source_run_id'))
            if _canonical_native_split_backend is not None
            else str(binding.get('source_run_id') or '')
        ),
        'eval_run_id':str(binding.get('eval_run_id') or ''),
        **selected_hashes,
        'native_split_quality_consumer_attestation':attestation,
    }
    artifacts=command.get('artifacts') if isinstance(command.get('artifacts'),dict) else {}
    expected_output=artifacts.get('semantic_output_manifest') if isinstance(artifacts.get('semantic_output_manifest'),dict) else {}
    if not manifest_sha256 or manifest_sha256 != str(expected_output.get('sha256') or '').lower() or manifest_sha256 != str(attestation.get('semantic_output_manifest_sha256') or '').lower():
        return {'native_split_semantic_binding_required':True,'native_split_semantic_binding_valid':False,'native_split_semantic_binding_status':'semantic_output_manifest_sha256_mismatch'}
    output_payload_ok,output_payload_status=_verify_manifest_payload_files(manifest_path)
    if not output_payload_ok:
        return {'native_split_semantic_binding_required':True,'native_split_semantic_binding_valid':False,'native_split_semantic_binding_status':output_payload_status}
    boundary_path, boundary_resolution = _resolve_split_semantic_manifest(
        result_path or (manifest_path.parent / '__collected_result__.json' if manifest_path else None),
        source_list, kind="boundary", fallback_relpaths=(
            'native_fifo_boundary/native_fifo_boundary_manifest.json',
            'native_boundary/native_fifo_boundary_manifest.json',
        ),
    )
    if boundary_resolution['manifest_resolution_status'] not in (
        'missing', 'manifest_verified', 'legacy_manifest_found'
    ):
        return {'native_split_semantic_binding_required':True,
                'native_split_semantic_binding_valid':False,
                'native_split_semantic_binding_status':boundary_resolution['manifest_resolution_status']}
    expected_boundary=artifacts.get('semantic_boundary_manifest') if isinstance(artifacts.get('semantic_boundary_manifest'),dict) else {}
    expected_boundary_sha=str(expected_boundary.get('sha256') or '').lower()
    if expected_boundary_sha:
        if boundary_path is None:
            return {'native_split_semantic_binding_required':True,'native_split_semantic_binding_valid':False,'native_split_semantic_binding_status':'semantic_boundary_manifest_missing'}
        boundary_sha=_hash_file(boundary_path)
        if boundary_sha != expected_boundary_sha or boundary_sha != str(attestation.get('semantic_boundary_manifest_sha256') or '').lower():
            return {'native_split_semantic_binding_required':True,'native_split_semantic_binding_valid':False,'native_split_semantic_binding_status':'semantic_boundary_manifest_sha256_mismatch'}
        boundary_payload_ok,boundary_payload_status=_verify_manifest_payload_files(boundary_path)
        if not boundary_payload_ok:
            return {'native_split_semantic_binding_required':True,'native_split_semantic_binding_valid':False,'native_split_semantic_binding_status':boundary_payload_status}
    if _bind_quality_to_native_split is None:
        portable_binding=None; portable_status='portable_binding_verifier_unavailable'
    else:
        portable_binding,portable_status=_bind_quality_to_native_split(
            native_row=portable_row,quality_binding=binding,
            verification_mode='portable',
        )
    if portable_binding is None:
        return {
            'native_split_semantic_binding_required':True,
            'native_split_semantic_binding_valid':False,
            'native_split_semantic_binding_status':'final_portable_binding_invalid',
            'native_split_final_portable_binding_valid':False,
            'native_split_final_portable_binding_status':portable_status,
        }
    return {
        'native_split_semantic_binding_required':True,
        'native_split_semantic_binding_valid':True,
        'native_split_semantic_binding_status':'sealed_manifest_and_payload_bytes_rehashed',
        'native_split_final_portable_binding_valid':True,
        'native_split_final_portable_binding_status':portable_status,
        'native_split_semantic_output_manifest_sha256':manifest_sha256,
        'native_split_semantic_boundary_manifest_sha256':_hash_file(boundary_path) if boundary_path else '',
        'native_split_semantic_output_manifest':str(manifest_path) if manifest_path else '',
        'native_split_semantic_boundary_manifest':str(boundary_path) if boundary_path else '',
    }


def _first_existing(paths: Iterable[Path | str | None]) -> Path | None:
    for p in paths:
        if not p:
            continue
        pp = Path(p).expanduser()
        if pp.is_file():
            return pp
    return None


def _find_eval_roots(root: Path, recursive: bool) -> list[Path]:
    root = root.expanduser().resolve()
    roots = [root]
    if recursive:
        for p in root.rglob('analysis_tables'):
            rr = p.parent
            if rr not in roots:
                roots.append(rr)
    # Also include direct children named like backend copies when they have analysis_tables
    for child in root.iterdir() if root.is_dir() else []:
        if child.is_dir() and (child / 'analysis_tables').exists() and child not in roots:
            roots.append(child)
    # stable order, longest paths last to keep human grouping easy
    return sorted(set(roots), key=lambda x: str(x))


from onnx_splitpoint_tool.native_rate_endpoints import rate_endpoint_fields

def _rows_from_native_fifo_runner(
    root: Path, *, include_direct_fallback: bool = True,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    analysis_dir = root / 'analysis_tables'
    summary_files = sorted(analysis_dir.glob('native_fifo_eval_runner*.json')) if analysis_dir.is_dir() else []
    if analysis_dir.is_dir():
        for blocked_path in sorted(analysis_dir.glob('native_hailo8_producer_e2e_eval__prerequisites.json')):
            blocked = _load_strict_json_object(blocked_path) or {}
            for observation in blocked.get('rows') or []:
                if not isinstance(observation, dict) or not isinstance(observation.get('planned_native_identity'), dict):
                    continue
                out.append({**observation, 'report': '', 'source_root': str(root),
                            'analysis_summary': str(blocked_path),
                            'execution_mode': 'native_split', 'producer_impl': 'prerequisite_blocked'})
    for p in summary_files:
        data = _load_json(p) or {}
        for r in data.get('rows', []) or []:
            model = str(r.get('model','')); case = str(r.get('case') or r.get('case_id') or ''); precision = str(r.get('precision') or data.get('precision') or 'uint8_cast_fp16')
            bs = root / model / 'benchmark_set'
            rp = _first_existing([
                r.get('report'), r.get('native_fifo_result'),
                bs / 'native_pipeline' / case / 'hailo_to_trt' / precision / 'native_fifo_results.json',
            ])
            j = _load_json(rp) or {}
            result_sha256 = _sha256_file(rp) if rp is not None else ''
            result_size_bytes = (
                int(rp.stat().st_size) if rp is not None else 0
            )
            quality_first = r.get('native_split_quality_required') is True
            if quality_first:
                ok = bool(
                    r.get('result_ok') is True
                    and j.get('ok') is True
                    and r.get('child_result_fresh') is True
                    and str(r.get('status') or '') == 'ok'
                    and int(r.get('returncode') or 0) == 0
                    and r.get('timed_out') is False
                    and result_sha256
                    and str(
                        r.get('native_fifo_result_sha256') or ''
                    ).strip().lower() == result_sha256
                    and int(
                        r.get('native_fifo_result_size_bytes') or 0
                    ) == result_size_bytes
                )
            else:
                ok = bool(r.get('ok') or r.get('result_ok') or j.get('ok'))
            runtime_success = _first_explicit_bool(
                (r, j), ('runtime_success',),
            )
            if runtime_success is None:
                runtime_success = ok
            diag = _diagnostic_fields(r, j, fallback='hailo8_native_fifo_failed_or_missing', ok=ok)
            endpoint_results = (
                dict(j.get('endpoint_results') or {})
                if isinstance(j.get('endpoint_results'), dict) else {}
            )
            raw_endpoint = (
                dict(endpoint_results.get('raw_model_outputs') or {})
                if isinstance(endpoint_results.get('raw_model_outputs'), dict)
                else {}
            )
            completed_endpoint = (
                dict(endpoint_results.get('completed_task') or {})
                if isinstance(endpoint_results.get('completed_task'), dict)
                else {}
            )
            primary_endpoint = completed_endpoint or j
            application_endpoint = completed_endpoint or j
            producer_impl = (
                'hailo8_cpp_vstreams_fifo'
                if raw_endpoint
                else _verified_native_producer_impl(r, j)
            )
            out.append({
                'backend': 'hailo8_to_trt', 'producer_impl': producer_impl,
                'model': model, 'case': case, 'precision': precision,
                'status': 'ok' if ok else str(r.get('status') or r.get('reason') or j.get('error') or 'failed'),
                'ok': ok,
                'runtime_success': runtime_success,
                'performance_endpoint': 'raw_model_outputs' if raw_endpoint else str(r.get('performance_endpoint') or ''),
                'primary_performance_endpoint': j.get('primary_performance_endpoint') or ('raw_model_outputs' if raw_endpoint else ''),
                'application_performance_endpoint': j.get('application_performance_endpoint') or ('completed_task' if completed_endpoint else ''),
                'energy_performance_endpoint': j.get('energy_performance_endpoint') or ('completed_task' if completed_endpoint else ''),
                'fps_makespan': _num(r.get('fps_makespan') or r.get('fps') or primary_endpoint.get('fps_makespan')),
                'fps_median': _num(r.get('fps_median') or primary_endpoint.get('fps_median') or primary_endpoint.get('fps_makespan')),
                'fps_ci95_low': _num(r.get('fps_ci95_low') or primary_endpoint.get('fps_ci95_low')),
                'fps_ci95_high': _num(r.get('fps_ci95_high') or primary_endpoint.get('fps_ci95_high')),
                'paper_fps': _num(r.get('paper_fps') or r.get('paper_equivalent_fps') or primary_endpoint.get('paper_equivalent_fps')),
                'handoff_ms': _num(r.get('handoff_ms') or primary_endpoint.get('handoff_ms')),
                'p1_ms': _num(primary_endpoint.get('p1_ms')), 'p2_run_ms': _num(primary_endpoint.get('p2_run_ms')),
                'p1_thread_ms': _num(primary_endpoint.get('p1_thread_ms')), 'p2_thread_ms': _num(primary_endpoint.get('p2_thread_ms')),
                'raw_model_outputs_fps_makespan': _num(raw_endpoint.get('fps_makespan')) if raw_endpoint else None,
                'raw_model_outputs_fps_median': _num(raw_endpoint.get('fps_median') or raw_endpoint.get('fps_makespan')) if raw_endpoint else None,
                'raw_model_outputs_preprocess_ms': _num(raw_endpoint.get('preprocess_ms')) if raw_endpoint else None,
                'raw_model_outputs_p1_ms': _num(raw_endpoint.get('p1_ms')) if raw_endpoint else None,
                'raw_model_outputs_handoff_ms': _num(raw_endpoint.get('handoff_ms')) if raw_endpoint else None,
                'raw_model_outputs_p2_run_ms': _num(raw_endpoint.get('p2_run_ms')) if raw_endpoint else None,
                'completed_task_fps_makespan': _num(completed_endpoint.get('fps_makespan')) if completed_endpoint else None,
                'completed_task_fps_median': _num(completed_endpoint.get('fps_median') or completed_endpoint.get('fps_makespan')) if completed_endpoint else None,
                'completed_task_completion_tail_ms': _num(completed_endpoint.get('completion_tail_ms')) if completed_endpoint else None,
                'completed_task_p2_run_ms': _num(completed_endpoint.get('p2_run_ms')) if completed_endpoint else None,
                'endpoint_relation': j.get('endpoint_relation') or {},
                'endpoint_relation_verified': j.get('endpoint_relation_verified'),
                'frames': primary_endpoint.get('frames'), 'warmup': primary_endpoint.get('warmup'), 'inflight': '',
                'producer_ready': '', 'consumer_ready': '',
                'input_image': j.get('input_image') or r.get('input_image') or '',
                'input_image_source': j.get('input_image_source') or r.get('input_image_source') or '',
                'input_image_sha256': j.get('input_image_sha256') or r.get('input_image_sha256') or '',
                'native_command_contract': j.get('native_command_contract') or r.get('native_command_contract') or {},
                'native_command_contract_sha256': (
                    (j.get('native_command_contract') or r.get('native_command_contract') or {}).get('contract_sha256', '')
                    if isinstance(j.get('native_command_contract') or r.get('native_command_contract') or {}, dict) else ''
                ),
                'native_fifo_result_sha256': result_sha256,
                'native_fifo_result_size_bytes': result_size_bytes,
                'child_result_fresh': r.get('child_result_fresh') is True,
                'report': str(rp) if rp else '', 'source_root': str(root),
                'analysis_summary': str(p),
                **_native_identity_fields(r, j),
                **_split_quality_fields(r, j),
                **_claim_contract_fields(r, j),
                **_split_output_contract(
                    rp, r, j,
                    fallback_relpaths=(
                        'native_fifo_outputs/native_fifo_output_manifest.json',
                        'native_fifo_outputs/native_fifo_outputs_manifest.json',
                        'native_outputs/native_outputs_manifest.json',
                    ),
                ),
                **_repeat_fields(r, completed_endpoint or j),
                **rate_endpoint_fields(j),
                **diag,
                'note': (
                    f'native {producer_impl} FIFO E2E measured'
                    if ok and producer_impl
                    else 'native FIFO producer provenance unverified'
                    if ok
                    else (
                        diag.get('failure_reason')
                        or 'Hailo8 native FIFO failed or missing'
                    )
                ),
            })

    if include_direct_fallback:
        runner_tokens = {
            token for row in out
            for token in _native_fifo_authority_tokens(row)
        }
        for fallback in _rows_from_native_fifo_direct_fallback(root):
            if runner_tokens.intersection(
                _native_fifo_authority_tokens(fallback)
            ):
                continue
            out.append(fallback)
    return out


def _rows_from_native_fifo_direct_fallback(
    root: Path,
) -> list[dict[str, Any]]:
    """Recover raw Hailo-8 results only when no runner summary survived."""
    out: list[dict[str, Any]] = []
    for rp in sorted(
        root.rglob(
            'native_pipeline/*/hailo_to_trt/*/native_fifo_results.json'
        ),
        key=lambda path: str(path),
    ):
        j = _load_json(rp) or {}
        case = rp.parents[2].name if len(rp.parents) >= 3 else ''
        precision = rp.parents[0].name if len(rp.parents) >= 1 else 'uint8_cast_fp16'
        model = ''
        for par in rp.parents:
            if par.name == 'benchmark_set' and par.parent.name:
                model = par.parent.name; break
        ok = bool(j.get('ok'))
        runtime_success = _first_explicit_bool(
            (j,), ('runtime_success',),
        )
        if runtime_success is None:
            runtime_success = ok
        diag = _diagnostic_fields({}, j, fallback='hailo8_native_fifo_failed_or_missing', ok=ok)
        producer_impl = _verified_native_producer_impl(j)
        out.append({
            'backend': 'hailo8_to_trt', 'producer_impl': producer_impl,
            'model': model, 'case': case, 'precision': precision,
            'status': 'ok' if ok else str(j.get('error') or 'failed'), 'ok': ok,
            'runtime_success': runtime_success,
            'fps_makespan': _num(j.get('fps_makespan')), 'paper_fps': _num(j.get('paper_equivalent_fps')),
            'handoff_ms': _num(j.get('handoff_ms')), 'p1_ms': _num(j.get('p1_ms')), 'p2_run_ms': _num(j.get('p2_run_ms')),
            'p1_thread_ms': _num(j.get('p1_thread_ms')), 'p2_thread_ms': _num(j.get('p2_thread_ms')),
            'frames': j.get('frames'), 'warmup': j.get('warmup'), 'inflight': '', 'producer_ready': '', 'consumer_ready': '',
            'input_image': j.get('input_image') or '', 'input_image_source': j.get('input_image_source') or '', 'input_image_sha256': j.get('input_image_sha256') or '',
            'native_command_contract': j.get('native_command_contract') or {},
            'native_command_contract_sha256': (
                (j.get('native_command_contract') or {}).get('contract_sha256', '')
                if isinstance(j.get('native_command_contract'), dict) else ''
            ),
            'report': str(rp), 'source_root': str(root),
            **_native_identity_fields(j),
            **_split_quality_fields(j),
            **_claim_contract_fields(j),
            **_split_output_contract(
                rp, j,
                fallback_relpaths=(
                    'native_fifo_outputs/native_fifo_output_manifest.json',
                    'native_fifo_outputs/native_fifo_outputs_manifest.json',
                    'native_outputs/native_outputs_manifest.json',
                ),
            ),
            **_repeat_fields(j),
            **diag,
            'note': 'native C++ FIFO E2E measured' if ok else (diag.get('failure_reason') or 'Hailo8 native FIFO failed or missing'),
        })
    return out


def _native_fifo_authority_tokens(
    row: Mapping[str, Any],
) -> set[tuple[str, ...]]:
    """Return exact file/runtime tokens for one collected Hailo-8 row.

    Paths identify the same locally materialized result.  UUID-bound runtime
    evidence also identifies a copied mirror of that result without relying on
    its container path.  Coarse model/case keys are intentionally excluded so
    independent repetitions remain independent.
    """
    tokens: set[tuple[str, ...]] = set()
    report = str(row.get('report') or '').strip()
    if report:
        try:
            path = Path(report).expanduser().resolve(strict=True)
        except (OSError, RuntimeError):
            pass
        else:
            if path.is_file():
                tokens.add(('result_path', str(path)))

    evidence = _source_repetition_evidence_identity(dict(row))
    if evidence and evidence[0] in {'records', 'runtime'}:
        tokens.add((
            'runtime_evidence',
            *_measurement_identity(dict(row)),
            *evidence,
        ))
    return tokens


def _rows_from_native_fifo_roots(
    roots: Iterable[Path],
) -> list[dict[str, Any]]:
    """Collect canonical runner rows first, then uncovered legacy results.

    A runner row is authoritative even when it records a failed execution.  A
    raw ``native_fifo_results.json`` may therefore recover a lost summary, but
    can never replace or launder an existing runner decision.
    """
    unique_roots = list(dict.fromkeys(Path(root).resolve() for root in roots))
    runner_rows: list[dict[str, Any]] = []
    for root in unique_roots:
        runner_rows.extend(
            _rows_from_native_fifo_runner(
                root, include_direct_fallback=False,
            )
        )

    authoritative_tokens = {
        token for row in runner_rows
        for token in _native_fifo_authority_tokens(row)
    }
    fallback_rows: list[dict[str, Any]] = []
    fallback_tokens_seen: set[tuple[str, ...]] = set()
    for root in unique_roots:
        for fallback in _rows_from_native_fifo_direct_fallback(root):
            tokens = _native_fifo_authority_tokens(fallback)
            if tokens.intersection(authoritative_tokens):
                continue
            if tokens and tokens.intersection(fallback_tokens_seen):
                continue
            fallback_rows.append(fallback)
            fallback_tokens_seen.update(tokens)
    return [*runner_rows, *fallback_rows]


def _rows_from_hailo10(root: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    analysis_dir = root / 'analysis_tables'
    for p in (sorted(analysis_dir.glob('native_hailo10h_producer_e2e_eval*.json')) if analysis_dir.is_dir() else []):
        data = _load_json(p) or {}
        for r in data.get('rows', []) or []:
            model = str(r.get('model','')); case = str(r.get('case') or r.get('case_id') or ''); precision = str(r.get('precision') or 'uint8_cast_fp16')
            bs = root / model / 'benchmark_set'
            rp = _first_existing([r.get('report'), bs / 'native_pipeline' / case / 'hailo10h_to_trt' / precision / 'hailo10_native_fifo_e2e_results.json'])
            j = _load_json(rp) or {}
            ok = bool(r.get('ok') or j.get('ok'))
            runtime_success = _first_explicit_bool(
                (r, j), ('runtime_success',),
            )
            if runtime_success is None:
                runtime_success = ok
            diag = _diagnostic_fields(r, j, fallback='hailo10_native_e2e_failed_or_missing', ok=ok)
            out.append({
                'backend':'hailo10h_to_trt','producer_impl':j.get('producer_impl') or 'hailo10_python_infermodel_async_fifo',
                'model':model,'case':case,'precision':precision,'status':'ok' if ok else str(r.get('status') or j.get('error') or 'failed'),'ok':ok,
                'runtime_success':runtime_success,
                'fps_makespan':_num(r.get('fps_makespan') or j.get('fps_makespan')),'paper_fps':_num(r.get('paper_fps') or j.get('paper_equivalent_fps')),
                'handoff_ms':_num(r.get('handoff_ms') or j.get('handoff_ms')),'p1_ms':_num(j.get('p1_ms')),'p2_run_ms':_num(j.get('p2_run_ms')),
                'p1_thread_ms':_num(j.get('p1_thread_ms')),'p2_thread_ms':_num(j.get('p2_thread_ms')),'frames':j.get('frames'),'warmup':j.get('warmup'),'inflight':j.get('inflight'),
                'input_image':r.get('input_image') or j.get('input_image') or '', 'input_image_source':r.get('input_image_source') or j.get('input_image_source') or '', 'input_image_sha256':r.get('input_image_sha256') or j.get('input_image_sha256') or '',
                'native_command_contract':j.get('native_command_contract') or r.get('native_command_contract') or {},
                'native_command_contract_sha256':(
                    j.get('native_command_contract_sha256') or r.get('native_command_contract_sha256')
                    or ((j.get('native_command_contract') or r.get('native_command_contract') or {}).get('contract_sha256', '')
                        if isinstance(j.get('native_command_contract') or r.get('native_command_contract') or {}, dict) else '')
                ),
                'producer_ready':'','consumer_ready':'','report':str(rp) if rp else '','source_root':str(root),'analysis_summary':str(p),
                **_native_identity_fields(r, j),
                **_split_quality_fields(r, j),
                **_claim_contract_fields(r, j),
                **_split_output_contract(
                    rp, r, j,
                    fallback_relpaths=('native_outputs/native_outputs_manifest.json',),
                ),
                **_repeat_fields(r, j),
                **diag,
                'note':'Hailo10 InferModel async FIFO E2E measured' if ok else (diag.get('failure_reason') or 'Hailo10 E2E failed or missing'),
            })
    return out


def _rows_from_deepx(root: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    analysis_dir = root / 'analysis_tables'
    e2e_files = sorted(analysis_dir.glob('native_deepx_producer_e2e_eval*.json')) if analysis_dir.is_dir() else []
    if e2e_files:
        for e2e in e2e_files:
            data = _load_json(e2e) or {}
            for r in data.get('rows', []) or []:
                model=str(r.get('model','')); case=str(r.get('case') or r.get('case_id') or ''); precision=str(r.get('precision') or 'uint8_cast_fp16')
                bs=root/model/'benchmark_set'; rp=_first_existing([r.get('report'),bs/'native_pipeline'/case/'deepx_to_trt'/precision/'deepx_native_fifo_e2e_results.json',bs/'native_pipeline'/case/'deepx_to_trt'/precision/'deepx_native_fifo_e2e_status.json',bs/'native_pipeline'/case/'deepx_to_trt'/precision/'deepx_native_fifo_probe.json'])
                j=_load_json(rp) or {}
                explicit_ok = _first_explicit_bool((j, r), ('ok',))
                ok = explicit_ok is True
                producer_ready = _first_explicit_bool(
                    (j, r), ('producer_ready',),
                )
                consumer_ready = _first_explicit_bool(
                    (j, r), ('consumer_ready',),
                )
                buildable = _first_explicit_bool(
                    (j, r), ('buildable', 'build_ok', 'compile_ok'),
                )
                runtime_executable = _first_explicit_bool(
                    (j, r),
                    ('runtime_executable', 'runtime_ok', 'run_ok', 'result_ok'),
                )
                if buildable is None:
                    buildable = ok
                if runtime_executable is None:
                    runtime_executable = ok
                runtime_success = _first_explicit_bool(
                    (r, j), ('runtime_success',),
                )
                if runtime_success is None:
                    runtime_success = runtime_executable
                native_impl=j.get('native_fifo_e2e_implemented')
                status='ok' if ok else ('e2e_not_implemented' if native_impl is False else str(r.get('status') or j.get('error') or 'failed'))
                diag=_diagnostic_fields(r, j, fallback='deepx_native_e2e_failed_or_missing', ok=ok)
                out.append({'backend':'deepx_to_trt','producer_impl':'deepx_python_dx_engine_fifo' if ok else 'deepx_scaffold','model':model,'case':case,'precision':precision,'status':status,'ok':ok,
                    'runtime_success':runtime_success,
                    'fps_makespan':_num(r.get('fps_makespan') or j.get('fps_makespan')),'paper_fps':_num(r.get('paper_fps') or j.get('paper_equivalent_fps')),'handoff_ms':_num(r.get('handoff_ms') or j.get('handoff_ms')),
                    'p1_ms':_num(j.get('p1_ms') or j.get('deepx_run_ms')),'p2_run_ms':_num(j.get('p2_run_ms')),'p1_thread_ms':_num(j.get('p1_thread_ms')),'p2_thread_ms':_num(j.get('p2_thread_ms')),
                    'frames':j.get('frames'),'warmup':j.get('warmup'),'inflight':'','producer_ready':producer_ready,'consumer_ready':consumer_ready,
                    'buildable':buildable,'runtime_executable':runtime_executable,
                    'input_image':r.get('input_image') or j.get('input_image') or '', 'input_image_source':r.get('input_image_source') or j.get('input_image_source') or '', 'input_image_sha256':r.get('input_image_sha256') or j.get('input_image_sha256') or '',
                    'native_command_contract':j.get('native_command_contract') or r.get('native_command_contract') or {},
                    'native_command_contract_sha256':(
                        j.get('native_command_contract_sha256') or r.get('native_command_contract_sha256')
                        or ((j.get('native_command_contract') or r.get('native_command_contract') or {}).get('contract_sha256', '')
                            if isinstance(j.get('native_command_contract') or r.get('native_command_contract') or {}, dict) else '')
                    ),
                    'report':str(rp) if rp else '','source_root':str(root),'analysis_summary':str(e2e),
                    **_native_identity_fields(r, j),
                    **_split_quality_fields(r, j),
                    **_claim_contract_fields(r, j),
                    **_split_output_contract(
                        rp, r, j,
                        fallback_relpaths=('native_outputs/native_outputs_manifest.json',),
                    ),
                    **_repeat_fields(r, j),**diag,'note':'DeepX dx_engine FIFO E2E measured' if ok else (diag.get('failure_reason') or j.get('next_action') or 'DeepX status row')})
    else:
        pdata=_load_json(analysis_dir/'native_deepx_producer_probe_eval.json') or {}
        for r in pdata.get('rows',[]) or []:
            producer_ready = _first_explicit_bool(
                (r,), ('producer_ready',),
            )
            consumer_ready = _first_explicit_bool(
                (r,), ('consumer_ready',),
            )
            out.append({'backend':'deepx_to_trt','producer_impl':'deepx_probe_only','model':r.get('model',''),'case':r.get('case',''),'precision':r.get('precision') or 'uint8_cast_fp16',
                'status':'probe_ready' if producer_ready is True and consumer_ready is True else 'probe_incomplete','ok':False,'producer_ready':producer_ready,'consumer_ready':consumer_ready,
                'buildable':False,'runtime_executable':False,
                'fps_makespan':None,'paper_fps':None,'handoff_ms':None,'p1_ms':None,'p2_run_ms':None,'p1_thread_ms':None,'p2_thread_ms':None,'frames':'','warmup':'','inflight':'','report':r.get('report',''),'source_root':str(root),'note':'DeepX producer/consumer probe only; E2E adapter required'})
    return out



def _rows_from_native_full(root: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    p = root / 'analysis_tables' / 'native_full_baseline_eval.json'
    data = _load_json(p) or {}
    for r in data.get('rows', []) or []:
        # A setup-local Central-Quality companion is evidence, never a timing
        # observation.  Even a malformed companion carrying plausible FPS
        # fields must not enter the performance matrix.
        if (
            r.get('quality_evidence_only') is True
            or str(r.get('execution_role') or '').strip().lower()
            == 'full_quality_only'
            or r.get('performance_claims_emitted') is False
        ):
            continue
        backend = str(r.get('backend') or '')
        model = str(r.get('model') or '')
        ok = bool(r.get('ok'))
        runtime_success = _first_explicit_bool(
            (r,), ('runtime_success', 'runtime_executable'),
        )
        if runtime_success is None:
            runtime_success = ok
        measured_fps = _num(
            r.get('fps_makespan') or r.get('pipeline_fps_selected')
            or r.get('full_backend_throughput_fps') or r.get('throughput_primary_fps')
            or r.get('FPS') or r.get('throughput_qps')
        )
        legacy_derived_fps = _num(r.get('prepared_feed_fps_from_mean_latency'))
        if backend == 'native_full_deepx' and r.get('outer_makespan_verified') is not True:
            # Preserve old reciprocal-latency values for diagnostics, but never
            # relabel them as a measured outer-makespan throughput.
            legacy_derived_fps = legacy_derived_fps or measured_fps
            measured_fps = None
        source_contract = _claim_contract_fields(r)
        source_completed_endpoint = (
            dict(r.get("completed_task_endpoint_attestation"))
            if isinstance(
                r.get("completed_task_endpoint_attestation"), Mapping,
            )
            else {}
        )
        completed_endpoint: dict[str, Any] = dict(source_completed_endpoint)
        completed_endpoint_status = str(
            r.get("completed_task_endpoint_attestation_status") or ""
        ) or str(
            source_completed_endpoint.get("status") or ""
        ) or ("unavailable" if completed_endpoint else "not_required")
        completed_endpoint_projection_status = (
            "source_attestation_preserved"
            if completed_endpoint else "not_required"
        )
        completed_endpoint_conflict = False
        # Reconstruct a completed endpoint only for a proven frozen host tail.
        # ``postprocess_included`` alone can describe accelerator-side NMS and
        # must never be relabelled as host postprocessing.
        if (
            str(r.get("task") or "").strip().lower() == "detection"
            and r.get("host_postprocess_frozen") is True
        ):
            completed_endpoint_status = "unavailable"
            completed_endpoint_projection_status = "reconstruction_pending"
            if _build_completed_detection_endpoint_attestation is not None:
                try:
                    reconstructed_endpoint = (
                        _build_completed_detection_endpoint_attestation(
                            r.get("frozen_host_postprocess_contract") or {},
                            r.get("frozen_host_postprocess_result") or {},
                            completed_frames=int(r.get("completed_frames") or 0),
                            postprocess_completed_frames=int(
                                r.get("postprocess_completed_frames") or 0
                            ),
                            source_endpoint_contract_hash=str(
                                r.get("endpoint_contract_hash") or ""
                            ),
                        )
                    )
                    source_legacy_projection = dict(
                        source_completed_endpoint,
                    )
                    reconstructed_legacy_projection = dict(
                        reconstructed_endpoint,
                    )
                    for projection in (
                        source_legacy_projection,
                        reconstructed_legacy_projection,
                    ):
                        for field in (
                            "completed_task_comparison_endpoint_contract",
                            "completed_task_comparison_endpoint_contract_hash",
                            "completed_task_comparison_output_endpoint_id",
                            "completed_task_completion_mode",
                        ):
                            projection.pop(field, None)
                    if (
                        source_completed_endpoint
                        and json.dumps(
                            source_legacy_projection,
                            sort_keys=True,
                            separators=(',', ':'),
                            ensure_ascii=False,
                        )
                        != json.dumps(
                            reconstructed_legacy_projection,
                            sort_keys=True,
                            separators=(',', ':'),
                            ensure_ascii=False,
                        )
                    ):
                        completed_endpoint_conflict = True
                        completed_endpoint = source_completed_endpoint
                        completed_endpoint_status = (
                            "conflict:source_and_reconstructed_attestation_mismatch"
                        )
                        completed_endpoint_projection_status = (
                            "conflict:source_and_reconstructed_attestation_mismatch"
                        )
                    else:
                        completed_endpoint = reconstructed_endpoint
                        completed_endpoint_status = str(
                            completed_endpoint.get("status") or ""
                        )
                        completed_endpoint_projection_status = (
                            "passed_existing_attestation_verified"
                            if source_completed_endpoint else
                            "passed_reconstructed_attestation"
                        )
                except Exception as exc:
                    completed_endpoint_status = (
                        f"failed:{type(exc).__name__}:{exc}"
                    )
                    completed_endpoint_projection_status = (
                        completed_endpoint_status
                    )
        direct_projection: dict[str, Any] = {}
        if (
            str(
                completed_endpoint.get('completed_task_completion_mode')
                or r.get('completed_task_completion_mode')
                or ''
            ).strip()
            == 'integrated_accelerator_plus_frozen_normalization'
        ):
            direct_projection = _strict_direct_completion_projection(
                r, completed_endpoint,
            )
            if not direct_projection:
                completed_endpoint_conflict = True
                completed_endpoint_status = (
                    'failed:strict_direct_bn6_completion_projection_invalid'
                )
                completed_endpoint_projection_status = (
                    completed_endpoint_status
                )
        out.append({
            'backend': backend,
            'producer_impl': r.get('producer_impl') or backend,
            'model': model,
            'case': 'full',
            'precision': r.get('precision') or r.get('trt_precision') or '',
            'execution_mode': 'native_full_baseline',
            **{key: r[key] for key in (
                'planned_native_identity', 'deepx_classification_admission',
                'diagnostic_only', 'counts_as_benchmark',
                'scientific_claim_exclusion_reason',
            ) if key in r},
            'setup_id': r.get('setup_id') or '',
            'comparison_backend': r.get('comparison_backend') or '',
            **source_contract,
            'status': 'ok' if ok else str(r.get('status') or 'failed'),
            'ok': ok,
            'runtime_success': runtime_success,
            'fps_makespan': measured_fps,
            'legacy_reciprocal_latency_fps': legacy_derived_fps,
            'fps_median': _num(r.get('fps_median') or r.get('fps_makespan')),
            'fps_ci95_low': _num(r.get('fps_ci95_low')),
            'fps_ci95_high': _num(r.get('fps_ci95_high')),
            'latency_mean_ms': _num(r.get('latency_mean_ms')),
            'latency_median_ms': _num(r.get('latency_median_ms') or r.get('latency_mean_ms')),
            'latency_p50_ms': _num(r.get('latency_p50_ms')),
            'latency_p95_ms': _num(r.get('latency_p95_ms')),
            'latency_ci95_low_ms': _num(r.get('latency_ci95_low_ms')),
            'latency_ci95_high_ms': _num(r.get('latency_ci95_high_ms')),
            'latency_semantics': r.get('latency_semantics') or '',
            'completion_interval_mean_ms': _num(r.get('completion_interval_mean_ms')),
            'paper_fps': _num(r.get('paper_fps') or r.get('fps_makespan')),
            'handoff_ms': None,
            'p1_ms': None,
            'p2_run_ms': None,
            'p1_thread_ms': None,
            'p2_thread_ms': None,
            'frames': r.get('frames') or data.get('frames') or '',
            'warmup': r.get('warmup') or data.get('warmup') or '',
            'inflight': r.get('inflight') or '',
            'producer_ready': '',
            'consumer_ready': '',
            'input_image': r.get('input_image') or '',
            'input_image_source': r.get('input_image_source') or '',
            'input_image_sha256': r.get('input_image_sha256') or '',
            'model_sha256': r.get('model_sha256') or r.get('source_onnx_sha256') or '',
            'hailo_hef_build_receipt_status': r.get(
                'hailo_hef_build_receipt_status'
            ) or '',
            'hailo_hef_build_receipt_path': r.get(
                'hailo_hef_build_receipt_path'
            ) or '',
            'hailo_hef_build_receipt_file_sha256': r.get(
                'hailo_hef_build_receipt_file_sha256'
            ) or '',
            'hailo_hef_build_receipt_sha256': r.get(
                'hailo_hef_build_receipt_sha256'
            ) or '',
            'hailo_hef_source_onnx_path': r.get(
                'hailo_hef_source_onnx_path'
            ) or '',
            'hailo_hef_compiler_onnx_sha256': r.get(
                'hailo_hef_compiler_onnx_sha256'
            ) or '',
            'hailo_hef_preprocessing_contract': (
                dict(r.get('hailo_hef_preprocessing_contract') or {})
                if isinstance(
                    r.get('hailo_hef_preprocessing_contract'), Mapping,
                ) else {}
            ),
            'hailo_hef_preprocessing_contract_sha256': r.get(
                'hailo_hef_preprocessing_contract_sha256'
            ) or '',
            'validation_dataset_sha256': r.get('validation_dataset_sha256') or r.get('dataset_sha256') or '',
            'validation_dataset_image_ids_sha256': r.get(
                'validation_dataset_image_ids_sha256'
            ) or '',
            'validation_dataset_ground_truth_sha256': r.get(
                'validation_dataset_ground_truth_sha256'
            ) or '',
            'source_request_sha256': r.get('source_request_sha256') or '',
            # Keep the canonical projection produced by
            # ``_claim_contract_fields``.  Reading the raw row again here used
            # to erase its diagnostic runtime-policy alias when older native
            # producers supplied only ``task_quality_policy_sha256``.
            'task_quality_policy_sha256': source_contract.get(
                'task_quality_policy_sha256'
            ) or '',
            'runtime_quality_gate_policy_sha256': source_contract.get(
                'runtime_quality_gate_policy_sha256'
            ) or '',
            'input_manifest': r.get('input_manifest') or '',
            'runtime_input_dtype': r.get('runtime_input_dtype') or '',
            'runtime_input_shape': r.get('runtime_input_shape') or [],
            'runtime_input_layout': r.get('runtime_input_layout') or '',
            'runtime_preprocess_mode': r.get('runtime_preprocess_mode') or '',
            'runtime_normalization': r.get('runtime_normalization') or {},
            'runtime_color_space': r.get('runtime_color_space') or '',
            'runtime_preprocessing_identity': (
                dict(r.get('runtime_preprocessing_identity') or {})
                if isinstance(
                    r.get('runtime_preprocessing_identity'), Mapping,
                ) else {}
            ),
            'runtime_preprocessing_sha256': r.get(
                'runtime_preprocessing_sha256'
            ) or '',
            'runtime_numeric_input_identity': (
                dict(r.get('runtime_numeric_input_identity') or {})
                if isinstance(
                    r.get('runtime_numeric_input_identity'), Mapping,
                ) else {}
            ),
            'runtime_numeric_input_sha256': r.get(
                'runtime_numeric_input_sha256'
            ) or '',
            # Never infer this from the legacy comparison precision.  The Full
            # runner must attest the actual engine/runtime precision.
            'engine_precision': r.get('engine_precision') or '',
            'output_dump_manifest': r.get('output_dump_manifest') or r.get('native_output_manifest') or '',
            'native_output_manifest': r.get('native_output_manifest') or r.get('output_dump_manifest') or '',
            'semantic_dump_status': r.get('semantic_dump_status') or '',
            'semantic_dump_failure_reason': r.get('semantic_dump_failure_reason') or '',
            'semantic_dump_status_detail': r.get('semantic_dump_status_detail') or '',
            'task': r.get('task') or '',
            'stage': r.get('stage') or '',
            'output_format': r.get('output_format') or '',
            'contract_family': r.get('contract_family') or '',
            'contract_source': r.get('contract_source') or '',
            'endpoint_contract_complete': _strict_bool(
                r.get('endpoint_contract_complete')
            ),
            'endpoint_contract_hash': r.get('endpoint_contract_hash') or '',
            'output_manifest_sha256': r.get('output_manifest_sha256') or '',
            'output_endpoint_attestation': r.get('output_endpoint_attestation')
            if isinstance(r.get('output_endpoint_attestation'), dict) else {},
            # Keep the accelerator tensor endpoint and the completed task
            # endpoint distinct.  A raw dump remains raw even when frozen
            # decode/NMS was included once per measured frame.
            'accelerator_output_stage': r.get('stage') or '',
            'accelerator_output_contract_family': r.get('contract_family') or '',
            'accelerator_endpoint_contract_hash': r.get('endpoint_contract_hash') or '',
            'accelerator_output_endpoint_attestation': r.get('output_endpoint_attestation')
            if isinstance(r.get('output_endpoint_attestation'), dict) else {},
            'source_e2e_scope': str(r.get('e2e_scope') or ''),
            'e2e_scope': source_contract.get('e2e_scope'),
            'e2e_claim_eligible': source_contract.get(
                'e2e_claim_eligible'
            ),
            'e2e_contract_reason': source_contract.get(
                'e2e_contract_reason'
            ),
            'comparison_endpoint_stratum': source_contract.get(
                'comparison_endpoint_stratum'
            ),
            'measurement_concurrency': source_contract.get(
                'measurement_concurrency'
            ),
            'requires_host_decode_nms': source_contract.get(
                'requires_host_decode_nms'
            ),
            'postprocess_location': source_contract.get(
                'postprocess_location'
            ),
            'host_postprocess_frozen': _strict_bool(
                r.get('host_postprocess_frozen')
            ),
            'postprocess_included': _strict_bool(
                r.get('postprocess_included')
            ),
            'postprocess_completed_frames': (
                r.get('postprocess_completed_frames')
                if r.get('postprocess_completed_frames') is not None
                else None
            ),
            'postprocess_completion_verified': _strict_bool(
                r.get('postprocess_completion_verified')
            ),
            'completed_task_result_artifact_verification_status': (
                r.get(
                    'completed_task_result_artifact_verification_status'
                ) or ''
            ),
            'direct_source_endpoint_binding_verified': _strict_bool(
                r.get('direct_source_endpoint_binding_verified')
            ),
            'direct_source_endpoint_binding_status': r.get(
                'direct_source_endpoint_binding_status'
            ) or '',
            'frozen_host_postprocess_contract': r.get('frozen_host_postprocess_contract')
            if isinstance(r.get('frozen_host_postprocess_contract'), dict) else None,
            'frozen_host_postprocess_contract_sha256': r.get('frozen_host_postprocess_contract_sha256') or '',
            'frozen_host_postprocess_result': r.get('frozen_host_postprocess_result')
            if isinstance(r.get('frozen_host_postprocess_result'), dict) else None,
            'normalization_frozen': bool(direct_projection),
            'frozen_decoded_nms_normalization_contract': (
                dict(direct_projection.get('contract') or {})
                if direct_projection else None
            ),
            'frozen_decoded_nms_normalization_contract_sha256': str(
                direct_projection.get('contract_sha256') or ''
            ),
            'frozen_decoded_nms_normalization_result': (
                dict(direct_projection.get('result') or {})
                if direct_projection else None
            ),
            'direct_bn6_completion_projection_status': str(
                direct_projection.get('status')
                or (
                    'invalid'
                    if completed_endpoint_conflict
                    and str(
                        completed_endpoint.get(
                            'completed_task_completion_mode'
                        )
                        or r.get('completed_task_completion_mode')
                        or ''
                    ).strip()
                    == (
                        'integrated_accelerator_plus_frozen_normalization'
                    )
                    else 'not_applicable'
                )
            ),
            'completed_task_stage': str(
                r.get('completed_task_stage')
                or completed_endpoint.get('stage')
                or ''
            ),
            'completed_task_contract_family': (
                str(r.get('completed_task_contract_family') or '')
                or (
                    'decoded_nms'
                    if completed_endpoint.get('attested') is True else ''
                )
            ),
            'completed_task_endpoint_contract': (
                r.get('completed_task_endpoint_contract')
                if isinstance(
                    r.get('completed_task_endpoint_contract'), dict,
                )
                else completed_endpoint.get('completed_endpoint_contract')
                if isinstance(
                    completed_endpoint.get('completed_endpoint_contract'),
                    dict,
                )
                else None
            ),
            'completed_task_endpoint_contract_hash': str(
                r.get('completed_task_endpoint_contract_hash')
                or completed_endpoint.get('endpoint_contract_hash')
                or ''
            ),
            'completed_task_output_endpoint_id': str(
                r.get('completed_task_output_endpoint_id')
                or completed_endpoint.get('output_endpoint_id')
                or ''
            ),
            'completed_task_comparison_endpoint_contract': (
                r.get('completed_task_comparison_endpoint_contract')
                if isinstance(
                    r.get('completed_task_comparison_endpoint_contract'),
                    dict,
                )
                else completed_endpoint.get(
                    'completed_task_comparison_endpoint_contract'
                )
                if isinstance(
                    completed_endpoint.get(
                        'completed_task_comparison_endpoint_contract'
                    ),
                    dict,
                )
                else None
            ),
            'completed_task_comparison_endpoint_contract_hash': str(
                r.get(
                    'completed_task_comparison_endpoint_contract_hash'
                )
                or completed_endpoint.get(
                    'completed_task_comparison_endpoint_contract_hash'
                )
                or ''
            ),
            'completed_task_comparison_output_endpoint_id': str(
                r.get('completed_task_comparison_output_endpoint_id')
                or completed_endpoint.get(
                    'completed_task_comparison_output_endpoint_id'
                )
                or ''
            ),
            'completed_task_completion_mode': str(
                r.get('completed_task_completion_mode')
                or completed_endpoint.get('completed_task_completion_mode')
                or ''
            ),
            'completed_task_endpoint_attested': (
                False
                if completed_endpoint_conflict
                else _strict_bool(
                    r.get('completed_task_endpoint_attested')
                )
                if r.get('completed_task_endpoint_attested') is not None
                else _strict_bool(completed_endpoint.get('attested'))
            ),
            'completed_task_endpoint_attestation': (
                completed_endpoint if completed_endpoint else None
            ),
            'completed_task_endpoint_attestation_status': completed_endpoint_status,
            # Preserve the same-hotloop canonical result and its on-disk
            # binding.  The validator must verify this file; an embedded
            # payload alone is not proof of what the measured runner saved.
            'completed_task_result_artifact': (
                dict(r.get('completed_task_result_artifact') or {})
                if isinstance(
                    r.get('completed_task_result_artifact'), Mapping,
                )
                else None
            ),
            'completed_task_result_artifact_path': str(
                r.get('completed_task_result_artifact_path') or ''
            ),
            'completed_task_result_artifact_saved': _strict_bool(
                r.get('completed_task_result_artifact_saved')
            ),
            'completed_task_result_artifact_sha256': str(
                r.get('completed_task_result_artifact_sha256') or ''
            ),
            'completed_task_result_artifact_file_sha256': str(
                r.get(
                    'completed_task_result_artifact_file_sha256'
                ) or ''
            ),
            'completed_task_endpoint_projection_status':
                completed_endpoint_projection_status,
            'report': r.get('report') or str(p),
            'source_root': str(root),
            **{field: r[field] for field in (
                'failure_stage', 'primary_failure_reason', 'upstream_evidence_path',
                'identity_conflicts', 'child_observation', 'performance_claim_eligible',
                'failure_class', 'upstream_stage', 'upstream_binding_set_error',
                'prerequisite_status', 'native_full_binding_transfer_attempted',
                'build_exclusion', 'upstream_build_observation', 'disposition',
            ) if field in r},
            'prerequisite_observations': _prerequisite_observations(r),
            'primary_repetition_failure': dict(r.get('primary_repetition_failure') or {}),
            'primary_repetition_failure_reason': r.get('primary_repetition_failure_reason') or '',
            'primary_repetition_status_detail': r.get('primary_repetition_status_detail') or '',
            'primary_repetition_error': r.get('primary_repetition_error') or '',
            'original_full_error': r.get('original_full_error') or '',
            'original_full_result_file': r.get('original_full_result_file') or '',
            'original_full_failure_context_file': r.get('original_full_failure_context_file') or '',
            'preparation_count_attempted': r.get('preparation_count_attempted'),
            'failure_reason': r.get('failure_reason') or '', 
            'status_detail': r.get('status_detail') or '',
            'error': r.get('error') or '',
            'timed_out': r.get('timed_out'),
            'returncode': r.get('returncode'),
            'stdout_tail': r.get('stdout_tail') or '',
            'stderr_tail': r.get('stderr_tail') or '',
            'result_source': r.get('result_source') or '',
            'fps_source': r.get('fps_source') or '',
            'performance_benchmark_source': r.get('performance_benchmark_source') or '',
            'performance_input_contract_mode': r.get('performance_input_contract_mode') or '',
            'outer_makespan_verified': r.get('outer_makespan_verified'),
            'comparison_precision': r.get('comparison_precision') or '',
            'legacy_comparison_precision': r.get('legacy_comparison_precision') or '',
            'execution_precision': r.get('execution_precision') or '',
            'full_runtime_precision': r.get('full_runtime_precision') or '',
            'runtime_precision_source': r.get('runtime_precision_source') or '',
            'repetition_count_requested': r.get('repetition_count_requested') or 1,
            'repetition_count_attempted': r.get('repetition_count_attempted', 1),
            'repetition_count_valid': r.get('repetition_count_valid') or (1 if ok else 0),
            'repetition_status': r.get('repetition_status') or ('complete' if ok else 'partial'),
            'repetition_aggregation': r.get('repetition_aggregation') or 'single_observation',
            'repetition_runtime_scope': r.get('repetition_runtime_scope') or '',
            'repetition_independence_verified': r.get('repetition_independence_verified'),
            'fps_repetition_samples': r.get('fps_repetition_samples') or ([r.get('fps_makespan')] if r.get('fps_makespan') is not None else []),
            'latency_mean_repetition_samples_ms': _repeat_fields(r).get('latency_mean_repetition_samples_ms') or (
                [r.get('latency_mean_ms')] if r.get('latency_mean_ms') is not None else []
            ),
            'request_latency': r.get('request_latency') or {},
            'repetition_records': r.get('repetition_records') or [],
            'performance_repetitions': r.get('performance_repetitions') or r.get('repetition_records') or [],
            'claim_ok': r.get('claim_ok'),
            'semantic_ok': r.get('semantic_ok'),
            'contract_consistent': r.get('contract_consistent'),
            'precision_quality_verified': r.get('precision_quality_verified'),
            'precision_quality_binding_verified': r.get(
                'precision_quality_binding_verified'
            ),
            'task_quality_observation_valid': r.get(
                'task_quality_observation_valid'
            ),
            'quality_claim_result_verified': r.get(
                'quality_claim_result_verified'
            ),
            'quality_request_binding': (
                dict(r.get('quality_request_binding') or {})
                if isinstance(r.get('quality_request_binding'), Mapping)
                else {}
            ),
            'quality_request_binding_status': r.get(
                'quality_request_binding_status'
            ) or '',
            'quality_request_binding_sha256': r.get(
                'quality_request_binding_sha256'
            ) or '',
            'quality_request_binding_set_sha256': r.get(
                'quality_request_binding_set_sha256'
            ) or '',
            'central_quality_result_sha256': r.get(
                'central_quality_result_sha256'
            ) or '',
            'full_command_contract': r.get('full_command_contract') or {},
            'quality_first_producer_identity': r.get('quality_first_producer_identity') or (
                (r.get('full_command_contract') or {}).get('quality_first_producer_identity', {})
                if isinstance(r.get('full_command_contract'), dict) else {}
            ),
            'quality_first_producer_identity_sha256': r.get('quality_first_producer_identity_sha256') or (
                (r.get('full_command_contract') or {}).get('quality_first_producer_identity_sha256', '')
                if isinstance(r.get('full_command_contract'), dict) else ''
            ),
            'full_command_contract_sha256': r.get('full_command_contract_sha256') or (
                (r.get('full_command_contract') or {}).get('contract_sha256', '')
                if isinstance(r.get('full_command_contract'), dict) else ''
            ),
            'steps': r.get('steps') or [],
            'note': 'native full baseline measured' if ok else (r.get('failure_reason') or r.get('status_detail') or r.get('error') or 'native full baseline failed'),
        })
        # Preserve adapter timing and its series even when this report is
        # exported without access to the collected source_root.
        out[-1].update({key: r[key] for key in (
            'measurement_endpoint', 'measurement_boundary', 'measured_duration_s',
            'makespan_ms', 'completed_frames', 'completed_work_units',
            'completed_work_units_status', 'historical_rate', 'fps_ci95_method', 'request_latency',
        ) if key in r})
    return out

def _measurement_identity(row: dict[str, Any]) -> tuple[str, ...]:
    if isinstance(row.get('planned_native_identity'), Mapping):
        from onnx_splitpoint_tool.native_job_identity import native_identity_key
        return ('planned_native_job', *native_identity_key(row['planned_native_identity']))
    is_full = (
        str(row.get('execution_mode') or '') == 'native_full_baseline'
        or str(row.get('backend') or '').startswith('native_full_')
    )
    return (
        str(row.get('backend') or ''), str(row.get('producer_impl') or ''),
        str(row.get('model') or ''), 'full' if is_full else str(row.get('case') or ''),
        str(row.get('precision') or ''), str(row.get('setup_id') or ''),
        str(row.get('comparison_backend') or '') if is_full else '',
        str(row.get('execution_precision') or row.get('full_runtime_precision') or ''),
        str(row.get('task') or ''), str(row.get('output_format') or ''),
        str(row.get('contract_family') or ''),
    )


def _repetition_claim_identity(row: Mapping[str, Any]) -> tuple[str, ...]:
    """Identity which must remain byte-for-byte stable across repetitions."""
    command_sha = _strict_sha256_token(
        row.get('full_command_contract_sha256')
        if _is_full_row(dict(row)) else row.get('native_command_contract_sha256')
    )
    producer_sha = _strict_sha256_token(
        row.get('quality_first_producer_identity_sha256')
    )
    return (
        command_sha,
        producer_sha,
        _strict_sha256_token(row.get('model_sha256') or row.get('source_onnx_sha256')),
        _strict_sha256_token(row.get('endpoint_contract_hash')),
        _runtime_precision_identity(dict(row)),
        _strict_sha256_token(row.get('validation_dataset_sha256')),
        _strict_sha256_token(row.get('preprocessing_contract_sha256')),
        _strict_sha256_token(row.get('quality_contract_sha256')),
        _strict_sha256_token(row.get('task_quality_policy_sha256')),
        str(row.get('e2e_scope') or '').strip().lower(),
        str(row.get('comparison_endpoint_stratum') or '').strip().lower(),
        str(row.get('measurement_concurrency') or '').strip().lower(),
        str(row.get('completed_task_stage') or '').strip().lower(),
        str(row.get('completed_task_contract_family') or '').strip().lower(),
        _strict_sha256_token(
            row.get('completed_task_endpoint_contract_hash')
        ),
        str(row.get('completed_task_output_endpoint_id') or '').strip(),
        _strict_sha256_token(
            row.get('frozen_host_postprocess_contract_sha256')
        ),
    )


def _as_numeric_list(value: Any) -> list[float]:
    if not isinstance(value, list):
        return []
    return [parsed for item in value if (parsed := _finite_num(item)) is not None]


def _source_repetition_evidence_identity(row: dict[str, Any]) -> tuple[str, ...]:
    """Identify one collected repetition source without using mirror paths.

    Recursive report collection can discover the same JSON payload through a
    producer root and through its benchmark-set child.  The two collector rows
    then have different ``analysis_summary``/``source_root`` values even though
    they contain the same runtime repetition.  Treating those paths as part of
    the evidence identity doubled Hailo-8's requested/attempted/valid counts.

    Runtime/repetition UUIDs are authoritative when present.  This deliberately
    still keeps two independent invocations that both use local repetition
    index one (and may even name the same report path).
    """
    raw_records = row.get('repetition_records')
    records = [record for record in raw_records if isinstance(record, dict)] \
        if isinstance(raw_records, list) else []
    record_ids: list[str] = []
    for record in records:
        repeat_id = str(
            record.get('repetition_id')
            or record.get('repetition_uuid')
            or ''
        ).strip()
        runtime_id = str(record.get('runtime_instance_id') or '').strip()
        workload_hash = str(
            record.get('workload_contract_sha256')
            or record.get('native_command_contract_sha256')
            or record.get('full_command_contract_sha256')
            or ''
        ).strip().lower()
        if repeat_id or runtime_id:
            record_ids.append('|'.join((repeat_id, runtime_id, workload_hash)))
    if record_ids:
        return ('records', *sorted(set(record_ids)))

    repeat_id = str(
        row.get('repetition_id')
        or row.get('repetition_uuid')
        or ''
    ).strip()
    runtime_id = str(row.get('runtime_instance_id') or '').strip()
    if repeat_id or runtime_id:
        return ('runtime', repeat_id, runtime_id)

    # With no UUID evidence, one concrete report path is one observation.  An
    # analysis table/root is only a fallback; it must not make a second mirror
    # of that same report look independent.
    report = str(row.get('report') or '').strip()
    if report:
        return ('report', report)
    analysis = str(row.get('analysis_summary') or '').strip()
    if analysis:
        return ('analysis', analysis)
    return ('source_root', str(row.get('source_root') or '').strip())


def _explicit_unstarted_prerequisite(group: list[dict[str, Any]]) -> bool:
    """Require bound zero-attempt evidence, never a reader's numeric default.

    Missing claim seals are expected before launch.  Planned job conflicts and
    any actual start/measurement observation still veto this narrow projection.
    """
    from onnx_splitpoint_tool.native_job_identity import (
        FIELDS, attach_identity_without_conflicts, planned_native_identity,
    )
    endpoint_values: dict[str, set[str]] = {}
    for row in group:
        planned = row.get('planned_native_identity')
        if not isinstance(planned, Mapping):
            return False
        identity = planned_native_identity(planned)
        required = FIELDS[:5] if identity['backend'].startswith('native_full_') else FIELDS
        if any(not identity[field] for field in required):
            return False
        if (row.get('status') not in {'blocked', 'blocked_upstream_quality', 'excluded_known_build'}
                or row.get('prerequisite_status') != 'blocked'
                or not all(str(row.get(field) or '').strip() for field in (
                    'primary_failure_reason', 'failure_stage', 'upstream_evidence_path'))):
            return False
        originals = row.get('prerequisite_observations')
        originals = originals if isinstance(originals, list) and originals else [row]
        # Only original explicit integers prove that no attempt was dispatched.
        if not any(type(source.get('repetition_count_attempted')) is int
                   and source['repetition_count_attempted'] == 0
                   and source.get('prerequisite_status') == 'blocked'
                   for source in originals if isinstance(source, Mapping)):
            return False
        observations = [row, *originals]
        child = row.get('child_observation')
        if isinstance(child, Mapping) and child:
            observations.append(child)
        for source in observations:
            if not isinstance(source, Mapping):
                return False
            if source.get('status') not in (None, '', 'blocked', 'blocked_upstream_quality', 'excluded_known_build'):
                return False
            if (source.get('identity_conflicts')
                    or source.get('identity_status') in {'identity_conflict', 'identity_ambiguous', 'identity_unresolved'}
                    or attach_identity_without_conflicts(planned, source).get('identity_conflicts')):
                return False
            other_planned = source.get('planned_native_identity')
            if isinstance(other_planned, Mapping) and planned_native_identity(other_planned) != identity:
                return False
            if any(field in source and (type(source[field]) is not int or source[field] != 0)
                   for field in ('repetition_count_attempted', 'performance_repeat_count_attempted')):
                return False
            if any(source.get(field) is True for field in (
                'ok', 'result_ok', 'runtime_success', 'runtime_started',
                'execution_started', 'measurement_started',
            )):
                return False
            if any(source.get(field) not in (None, '', 0) for field in (
                'repetition_count_valid', 'repetitions_completed', 'completed_frames',
                'frames_completed', 'completed_work_units', 'postprocess_completed_frames',
                'performance_repeat_count_valid', 'performance_repeat_n',
            )):
                return False
            if source.get('repetition_index') not in (None, ''):
                return False
            if any(source.get(field) for field in (
                'repetition_records', 'repetition_evidence', 'performance_repetitions',
                'fps_repetition_samples', 'latency_mean_repetition_samples_ms',
            )):
                return False
            if any(source.get(field) not in (None, '') for field in (
                'fps_makespan', 'fps_median', 'latency_mean_ms', 'latency_median_ms',
                'latency_p50_ms', 'latency_p95_ms', 'completion_interval_mean_ms',
                'handoff_ms', 'p1_ms', 'p2_run_ms', 'p1_thread_ms', 'p2_thread_ms',
            )):
                return False
            for field in ('output_endpoint_id', 'endpoint_contract_hash', 'performance_endpoint'):
                if source.get(field) not in (None, ''):
                    endpoint_values.setdefault(field, set()).add(str(source[field]))
        if any(len(values) > 1 for values in endpoint_values.values()):
            return False
    return bool(group)


def _aggregate_repetitions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate repeated evidence by median; never select the fastest row."""
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    evidence_seen: dict[tuple[tuple[str, ...], tuple[str, ...]], dict[str, Any]] = {}
    for row in rows:
        key = _measurement_identity(row)
        evidence_key = (key, _source_repetition_evidence_identity(row))
        if evidence_key in evidence_seen:
            previous = evidence_seen[evidence_key]
            # Mirrors may share a report path.  A conflicting prerequisite or
            # a newer start at that path is not an identical source mirror.
            compare_fields = (*_PREREQUISITE_OBSERVATION_FIELDS, 'prerequisite_observations', 'child_observation')
            same_prerequisite = all(previous.get(field) == row.get(field) for field in compare_fields)
            if not _prerequisite_observations(previous, row) or same_prerequisite:
                continue
        evidence_seen[evidence_key] = row
        groups.setdefault(key, []).append(row)

    out: list[dict[str, Any]] = []
    for key, group in groups.items():
        claim_identities = [_repetition_claim_identity(row) for row in group]
        claim_identity_drift = len(set(claim_identities)) != 1
        is_full_group = any(_is_full_row(row) for row in group)
        is_trt_full_group = any(
            str(row.get('backend') or '').strip().lower()
            == 'native_full_tensorrt' for row in group
        )
        claim_identity_incomplete = bool(
            is_full_group and any(not identity[0] for identity in claim_identities)
        )
        if is_trt_full_group:
            claim_identity_incomplete = bool(
                claim_identity_incomplete
                or any(not identity[1] for identity in claim_identities)
            )
        if not claim_identity_drift and _explicit_unstarted_prerequisite(group):
            representative = dict(group[0])
            representative.update(
                ok=False, runtime_success=False,
                repetition_status='not_started', repetition_aggregation='',
                aggregation_applied=False,
                repetition_count_requested=max(
                    int(_num(row.get('repetition_count_requested')) or 1) for row in group),
                repetition_count_attempted=0, repetition_count_valid=0,
                failure_reason=representative['primary_failure_reason'],
                repetition_claim_identity_status='incomplete' if claim_identity_incomplete else 'exact',
                repetition_claim_identity_drift=False,
                repetition_claim_identity_incomplete=claim_identity_incomplete,
                performance_claim_eligible=False,
                native_job_observations=[dict(row) for row in group],
                source_reports=list(dict.fromkeys(str(row['report']) for row in group if row.get('report'))),
                source_roots=sorted({str(row.get('source_root') or '') for row in group}),
            )
            for field in ('fps_makespan', 'fps_median', 'fps_ci95_low', 'fps_ci95_high',
                          'latency_mean_ms', 'latency_median_ms', 'latency_p50_ms', 'latency_p95_ms',
                          'latency_ci95_low_ms', 'latency_ci95_high_ms'):
                representative[field] = None
            for field in ('repetition_records', 'repetition_evidence', 'performance_repetitions',
                          'fps_repetition_samples', 'latency_mean_repetition_samples_ms'):
                representative[field] = []
            out.append(representative)
            continue
        valid_rows = [
            row for row in group
            if _strict_bool(row.get('ok')) is True
            and (_finite_num(row.get('fps_makespan')) or 0.0) > 0.0
        ]
        representative = dict(valid_rows[0] if valid_rows else group[0])
        fps_samples: list[float] = []
        latency_samples: list[float] = []
        raw_records: list[dict[str, Any]] = []
        attempted = 0
        attempted_unknown = False
        requested = 0
        for row in group:
            explicit_fps = _as_numeric_list(row.get('fps_repetition_samples'))
            row_fps = _finite_num(row.get('fps_makespan'))
            fps_samples.extend(
                explicit_fps
                or ([row_fps] if _strict_bool(row.get('ok')) is True and row_fps is not None and row_fps > 0 else [])
            )
            explicit_latency = _as_numeric_list(row.get('latency_mean_repetition_samples_ms'))
            row_latency = _finite_num(row.get('latency_mean_ms'))
            latency_samples.extend(
                explicit_latency
                or ([row_latency] if _strict_bool(row.get('ok')) is True and row_latency is not None else [])
            )
            embedded_records = row.get('repetition_records')
            if isinstance(embedded_records, list):
                raw_records.extend(dict(record) for record in embedded_records if isinstance(record, dict))
            elif row.get('repetition_index') not in (None, ''):
                raw_records.append({
                    'request_latency': row.get('request_latency'),
                    'repetition_id': row.get('repetition_id'),
                    'runtime_instance_id': row.get('runtime_instance_id'),
                    'repetition_index': row.get('repetition_index'),
                    'ok': row.get('ok'),
                    'status': row.get('status'),
                    'fps_makespan': row.get('fps_makespan'),
                    'latency_mean_ms': row.get('latency_mean_ms'),
                    'repetition_runtime_scope': row.get('repetition_runtime_scope'),
                    **{field: row[field] for field in (
                        'failure_stage', 'failure_reason', 'primary_failure_reason',
                        'error', 'status_detail', 'identity_conflicts',
                    ) if field in row},
                })
            row_attempted = row.get('repetition_count_attempted')
            if type(row_attempted) is int and row_attempted >= 0:
                attempted += row_attempted
            elif 'repetition_count_attempted' not in row and (embedded_records or explicit_fps or (row.get('ok') is True and row_fps is not None)):
                attempted += max(1, len(embedded_records or []), len(explicit_fps))
            else:
                attempted_unknown = True
            requested += int(_num(row.get('repetition_count_requested')) or 1)
        # Each outer workflow invocation can legitimately start its local
        # repetition counter at one.  Keep that value for audit, but assign
        # canonical aggregate indices and preserve the runner UUID/runtime id
        # as the actual independence evidence.
        normalized_records: list[dict[str, Any]] = []
        seen_record_ids: set[tuple[str, str, str]] = set()
        for record in raw_records:
            local_index = record.get('repetition_index')
            repeat_id = _repeat_record_id(record)
            runtime_id = str(record.get('runtime_instance_id') or '').strip()
            report = str(record.get('report') or '').strip()
            identity = (repeat_id, runtime_id, report)
            if identity in seen_record_ids:
                continue
            seen_record_ids.add(identity)
            normalized = dict(record)
            normalized['source_repetition_index'] = local_index
            normalized['repetition_index'] = len(normalized_records) + 1
            if not str(normalized.get('repetition_id') or '').strip():
                normalized['repetition_id'] = (
                    f"aggregate:{len(normalized_records) + 1}:"
                    f"{runtime_id or report}"
                )
            normalized_records.append(normalized)
        raw_records = normalized_records
        # When raw UUID-bound records are available they are the authoritative
        # sample vector.  Do not retain a duplicated or stale summary vector.
        record_fps = [
            value for record in raw_records
            if _strict_bool(record.get('ok')) is True
            and (value := _finite_num(record.get('fps_makespan'))) is not None
            and value > 0
        ]
        if raw_records:
            fps_samples = record_fps
        record_latency = [
            value for record in raw_records
            if _strict_bool(record.get('ok')) is True
            and (value := _finite_num(
                record.get('latency_median_ms')
                or record.get('latency_mean_ms')
            )) is not None
        ]
        if raw_records and record_latency:
            latency_samples = record_latency
        seed = '|'.join(key)
        fps_median, fps_low, fps_high = _bootstrap_median_ci(fps_samples, seed + '|fps')
        lat_median, lat_low, lat_high = _bootstrap_median_ci(latency_samples, seed + '|latency')
        complete = bool(
            fps_samples
            and len(fps_samples) == requested
            and all(_strict_bool(row.get('ok')) is True for row in group)
            and not claim_identity_drift
            and not claim_identity_incomplete
        )
        runtime_success = bool(
            group
            and all(
                _strict_bool(row.get('runtime_success')) is True
                for row in group
            )
        )
        representative.update({
            'native_job_observations': [{field: row[field] for field in (
                'ok', 'status', 'failure_stage', 'failure_reason', 'primary_failure_reason',
                'error', 'status_detail', 'identity_conflicts', 'report',
                'repetition_count_attempted', 'repetition_count_valid',
            ) if field in row} for row in group],
            'ok': complete,
            'runtime_success': runtime_success,
            'status': 'ok' if complete else 'partial_repetitions',
            'fps_makespan': fps_median,
            'fps_median': fps_median,
            'fps_ci95_low': fps_low,
            'fps_ci95_high': fps_high,
            'latency_mean_ms': lat_median,
            'latency_median_ms': lat_median,
            'latency_ci95_low_ms': lat_low,
            'latency_ci95_high_ms': lat_high,
            'repetition_count_requested': requested,
            'repetition_count_attempted': None if attempted_unknown else attempted,
            'repetition_count_valid': len(fps_samples),
            'repetition_status': 'complete' if complete else 'partial',
            'repetition_aggregation': 'median_with_deterministic_percentile_bootstrap_ci95',
            'fps_repetition_samples': fps_samples,
            'latency_mean_repetition_samples_ms': latency_samples,
            'repetition_records': raw_records,
            'repetition_evidence': raw_records,
            'performance_repetitions': raw_records,
            'repetition_claim_identity_status': (
                'drift' if claim_identity_drift else
                'incomplete' if claim_identity_incomplete else 'exact'
            ),
            'repetition_claim_identity_drift': claim_identity_drift,
            'repetition_claim_identity_incomplete': claim_identity_incomplete,
            'repetition_claim_identities': [list(identity) for identity in claim_identities],
            'source_reports': list(dict.fromkeys(
                str(row.get('report') or '') for row in group
                if str(row.get('report') or '')
            )),
            'source_roots': sorted({str(row.get('source_root') or '') for row in group}),
        })
        if not complete:
            # Every candidate here belongs to the exact grouped measurement
            # identity; never search an unrelated setup for an older message.
            failed = next((r for r in group if r.get('primary_repetition_failure_reason')), None)
            if failed is None:
                failed = next((r for r in group if r.get('primary_failure_reason')), None)
            if failed is None:
                failed = next((r for r in group if r.get('semantic_dump_failure_reason')), None)
            if failed is not None:
                for name in ('primary_repetition_failure', 'primary_repetition_failure_reason',
                             'primary_repetition_status_detail', 'primary_repetition_error',
                             'semantic_dump_failure_reason', 'semantic_dump_status',
                             'primary_failure_reason', 'failure_stage', 'upstream_evidence_path',
                             'identity_conflicts', 'child_observation', 'error', 'status_detail',
                             'build_exclusion', 'upstream_build_observation', 'disposition'):
                    if name in failed:
                        representative[name] = failed[name]
                # A current preparation failure may predate the common primary
                # alias. Promote its preserved cause, never an aggregate gate.
                representative['primary_failure_reason'] = str(
                    failed.get('primary_repetition_failure_reason')
                    or failed.get('primary_failure_reason')
                    or failed.get('semantic_dump_failure_reason')
                    or ''
                )
            reasons = sorted({str(row.get('failure_reason') or '') for row in group if str(row.get('failure_reason') or '')})
            identity_reason = (
                'native_repetition_claim_identity_drift'
                if claim_identity_drift else
                'native_repetition_claim_identity_incomplete'
                if claim_identity_incomplete else
                'native_repetition_set_incomplete'
            )
            representative['failure_reason'] = identity_reason + (':' + ','.join(reasons) if reasons else '')
        out.append(representative)
    return out


def _dedupe(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compatibility alias: deduplication is now statistical aggregation."""
    return _aggregate_repetitions(rows)


def _strict_sha256_token(value: Any) -> str:
    """Canonicalise one bare or singly ``sha256:``-prefixed digest."""
    text = str(value or '').strip().lower()
    if text.startswith('sha256:'):
        text = text[len('sha256:'):]
    return text if len(text) == 64 and all(ch in '0123456789abcdef' for ch in text) else ''


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
    ).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


_TRT_QUALITY_PRODUCER_SCHEMA = (
    'onnx-splitpoint/tensorrt-central-quality-producer-identity'
)


def _validated_quality_first_producer(
    value: Any, *, task: str, role: str,
) -> tuple[dict[str, Any], str, list[str]]:
    if _validate_quality_producer_contract is None:
        return {}, '', [f'{role}:validator_unavailable']
    if not isinstance(value, Mapping):
        return {}, '', [f'{role}:producer_missing']
    try:
        validated, producer_sha = _validate_quality_producer_contract(
            dict(value), role=role, task=str(task or '').strip().lower(),
        )
    except Exception as exc:
        token = re.sub(
            r'[^a-z0-9_]+', '_', f'{type(exc).__name__}_{exc}'.lower(),
        ).strip('_')
        return {}, '', [f'{role}:producer_invalid:{token[:240]}']
    producer = dict(validated)
    producer_sha = _strict_sha256_token(producer_sha)
    errors: list[str] = []
    if not producer_sha:
        errors.append(f'{role}:producer_sha256_invalid')
    if (
        producer.get('schema') != _TRT_QUALITY_PRODUCER_SCHEMA
        or producer.get('execution_role') != 'full_quality_only'
        or producer.get('backend') != 'native_tensorrt'
        or producer.get('variant') != 'full'
        or producer.get('case_id') != 'full'
        or producer.get('source_run_id') != 'native_full_tensorrt'
        or producer.get('performance_claims_emitted') is not False
    ):
        errors.append(f'{role}:quality_only_scope_invalid')
    if not _strict_sha256_token(
        producer.get('engine_build_receipt_file_sha256')
    ):
        errors.append(f'{role}:receipt_file_sha256_invalid')
    return (producer if not errors else {}), (producer_sha if not errors else ''), errors


def _artifact_exact(
    observed: Any, expected: Any, *, expected_sha: Any = None,
) -> bool:
    if not isinstance(observed, Mapping) or not isinstance(expected, Mapping):
        return False
    try:
        observed_size = int(observed.get('size_bytes'))
        expected_size = int(expected.get('size_bytes'))
    except (TypeError, ValueError):
        return False
    canonical_expected_sha = _strict_sha256_token(
        expected.get('sha256') if expected_sha is None else expected_sha
    )
    return bool(
        str(observed.get('path') or '').strip()
        == str(expected.get('path') or '').strip()
        and _strict_sha256_token(observed.get('sha256')) == canonical_expected_sha
        and canonical_expected_sha
        and observed_size > 0 and observed_size == expected_size
    )


def _performance_quality_first_binding(
    row: Mapping[str, Any],
) -> tuple[dict[str, Any], str, list[str]]:
    """Verify the exact Quality-FIRST identity sealed into Native Full TRT."""
    errors: list[str] = []
    raw_contract = row.get('full_command_contract')
    if not isinstance(raw_contract, Mapping):
        return {}, '', ['performance:full_command_contract_missing']
    contract = dict(raw_contract)
    declared_contract_sha = _strict_sha256_token(contract.get('contract_sha256'))
    body = dict(contract); body.pop('contract_sha256', None)
    if not declared_contract_sha or _canonical_json_sha256(body) != declared_contract_sha:
        errors.append('performance:full_command_contract_sha256_mismatch')
    if _strict_sha256_token(row.get('full_command_contract_sha256')) != declared_contract_sha:
        errors.append('performance:full_command_contract_duplicate_mismatch')
    producer, producer_sha, producer_errors = _validated_quality_first_producer(
        contract.get('quality_first_producer_identity'),
        task=str(row.get('task') or ''), role='performance full command contract',
    )
    errors.extend(producer_errors)
    if (
        _strict_sha256_token(contract.get('quality_first_producer_identity_sha256'))
        != producer_sha
        or _strict_sha256_token(row.get('quality_first_producer_identity_sha256'))
        != producer_sha
    ):
        errors.append('performance:quality_first_producer_sha256_mismatch')
    direct = row.get('quality_first_producer_identity')
    if direct not in (None, {}) and direct != producer:
        errors.append('performance:quality_first_producer_duplicate_mismatch')
    if producer:
        if (
            str(row.get('backend') or '') != 'native_full_tensorrt'
            or str(row.get('case') or row.get('case_id') or '').strip().lower() != 'full'
            or str(row.get('model') or row.get('model_id') or '').strip().lower()
            != str(producer.get('model_id') or '').strip().lower()
            or str(row.get('setup_id') or '').strip().lower()
            != str(producer.get('setup_id') or '').strip().lower()
            or _runtime_precision_identity(dict(row))
            != str(producer.get('runtime_precision_identity') or '')
        ):
            errors.append('performance:producer_scope_or_precision_mismatch')
        artifacts = contract.get('artifacts') if isinstance(contract.get('artifacts'), Mapping) else {}
        workload = contract.get('energy_workload') if isinstance(contract.get('energy_workload'), Mapping) else {}
        receipt_binding = producer.get('engine_build_receipt') if isinstance(producer.get('engine_build_receipt'), Mapping) else {}
        receipt_artifact = artifacts.get(
            str(workload.get('engine_build_receipt_artifact') or '')
        )
        bindings = (
            ('build_onnx', artifacts.get(str(workload.get('source_model_artifact') or '')), producer.get('build_onnx'), None),
            ('engine', artifacts.get(str(workload.get('engine_artifact') or '')), producer.get('engine'), None),
            ('trtexec', artifacts.get(str(workload.get('trtexec_artifact') or '')), producer.get('trtexec'), None),
            (
                'engine_build_receipt_file',
                receipt_artifact,
                receipt_binding,
                producer.get('engine_build_receipt_file_sha256'),
            ),
        )
        for name, observed, expected, expected_sha in bindings:
            if not _artifact_exact(observed, expected, expected_sha=expected_sha):
                errors.append(f'performance:{name}_artifact_mismatch')
        if dict(contract.get('trt_engine_build_receipt') or {}) != dict(
            receipt_binding.get('receipt') or {}
        ):
            errors.append('performance:engine_build_receipt_content_mismatch')
        receipt = receipt_binding.get('receipt') if isinstance(receipt_binding.get('receipt'), Mapping) else {}
        receipt_duplicates = (
            ('engine_build_receipt_path', str(contract.get('engine_build_receipt_path') or '').strip(), str(receipt_binding.get('path') or '').strip()),
            ('engine_build_receipt_sha256', _strict_sha256_token(contract.get('engine_build_receipt_sha256')), _strict_sha256_token(receipt_binding.get('sha256'))),
            ('engine_build_receipt_file_sha256', _strict_sha256_token(contract.get('engine_build_receipt_file_sha256')), _strict_sha256_token(producer.get('engine_build_receipt_file_sha256'))),
            ('trt_engine_build_receipt_sha256', _strict_sha256_token(contract.get('trt_engine_build_receipt_sha256')), _strict_sha256_token(receipt.get('receipt_sha256'))),
        )
        for field_name, observed, expected in receipt_duplicates:
            if not observed or observed != expected:
                errors.append(f'performance:{field_name}_mismatch')
        try:
            receipt_size_match = (
                int(contract.get('engine_build_receipt_size_bytes'))
                == int(receipt_binding.get('size_bytes')) > 0
            )
        except (TypeError, ValueError):
            receipt_size_match = False
        if not receipt_size_match:
            errors.append('performance:engine_build_receipt_size_bytes_mismatch')
        try:
            receipt_file_size_match = (
                int(contract.get('engine_build_receipt_file_size_bytes'))
                == int((receipt_artifact or {}).get('file_size_bytes')) > 0
            )
        except (TypeError, ValueError):
            receipt_file_size_match = False
        if not receipt_file_size_match:
            errors.append('performance:engine_build_receipt_file_size_bytes_mismatch')
        if _strict_sha256_token(contract.get('source_model_sha256')) != _strict_sha256_token(
            (producer.get('source_onnx') or {}).get('sha256')
        ):
            errors.append('performance:source_model_sha256_mismatch')
    return (producer if not errors else {}), (producer_sha if not errors else ''), list(dict.fromkeys(errors))


def _validation_quality_first_binding(
    row: Mapping[str, Any],
) -> tuple[dict[str, Any], str, list[str]]:
    producer, producer_sha, errors = _validated_quality_first_producer(
        row.get('quality_first_producer_identity'),
        task=str(row.get('task') or ''), role='native validation summary',
    )
    if (
        _strict_sha256_token(row.get('quality_first_producer_identity_sha256'))
        != producer_sha
        or row.get('quality_first_semantic_dump_binding_valid') is not True
        or str(row.get('quality_first_binding_status') or '')
        != 'central_native_exact_identity_match'
    ):
        errors.append('validation:quality_first_binding_not_exact')
    return (producer if not errors else {}), (producer_sha if not errors else ''), list(dict.fromkeys(errors))


def _verified_performance_command_contract(
    row: dict[str, Any],
    *, remote_execution_contexts: Iterable[Mapping[str, Any]] = (),
) -> tuple[str, list[str]]:
    """Verify the producer contract before trusting its top-level digest.

    The top-level ``*_command_contract_sha256`` field is only a reference.  It
    must not be able to self-assert an arbitrary or merely hash-consistent
    nested payload.  The same backend-aware validators used to admit energy
    workloads are therefore applied here as a scientific-claim boundary.
    """
    full = _is_full_row(row)
    field = 'full_command_contract' if full else 'native_command_contract'
    raw = row.get(field)
    errors: list[str] = []
    if not isinstance(raw, dict):
        return '', [f'command_contract_sha256:{field}_missing']
    top_level, top_errors = _quality_binding_evidence(row)
    errors.extend(top_errors)
    model_sha = top_level.get('model_sha256', '')
    expected_identity = {
        'backend': row.get('backend'),
        'model': row.get('model') or row.get('model_id'),
        'case': row.get('case') or row.get('case_id') or ('full' if full else ''),
        'precision': row.get('precision'),
        'setup_id': row.get('setup_id'),
        'comparison_backend': row.get('comparison_backend'),
        'model_sha256': model_sha,
        'source_model_sha256': model_sha,
    }
    verified: dict[str, Any] | None = None
    status = ''
    if full:
        backend = str(row.get('backend') or '').strip().lower()
        if backend == 'native_full_deepx':
            remote_context, remote_context_status = (
                _remote_execution_context_for_performance_row(
                    row, raw, remote_execution_contexts,
                )
            )
            if remote_context:
                expected_identity.update({
                    'remote_root': remote_context['remote_root'],
                    'remote_tool_dir': remote_context['remote_tool_dir'],
                })
            else:
                status = remote_context_status
        if _strict_verify_full_command_contract is None:
            status = 'full_command_contract_strict_verifier_unavailable'
        elif not status:
            verified, status = _strict_verify_full_command_contract(
                raw, expected_identity=expected_identity,
            )
    elif _strict_verify_native_command_contract is None:
        status = 'native_command_contract_strict_verifier_unavailable'
    else:
        verified, status = _strict_verify_native_command_contract(
            raw, expected_identity=expected_identity,
        )
    if verified is None:
        errors.append(
            f'command_contract_sha256:{field}_strict_validation_failed:{status or "unknown"}'
        )

    declared = _strict_sha256_token(
        (verified or raw).get('contract_sha256')
        if isinstance(verified or raw, dict) else ''
    )
    if not declared:
        errors.append(f'command_contract_sha256:{field}_canonical_hash_mismatch')

    backend = str(row.get('backend') or '').strip().lower()
    if backend == 'native_full_tensorrt':
        if not model_sha:
            errors.append('command_contract_sha256:tensorrt_performance_model_sha256_missing')
        source_model_sha = _strict_sha256_token(
            (verified or {}).get('source_model_sha256')
        )
        if not source_model_sha or source_model_sha != model_sha:
            errors.append('command_contract_sha256:tensorrt_source_model_sha256_mismatch')
        full_contract = row.get('full_command_contract')
        has_quality_first_marker = bool(
            row.get('quality_first_producer_identity_sha256')
            or (
                isinstance(full_contract, Mapping)
                and (
                    full_contract.get('quality_first_producer_identity')
                    or full_contract.get('quality_first_producer_identity_sha256')
                )
            )
        )
        if has_quality_first_marker:
            producer, producer_sha, producer_errors = _performance_quality_first_binding(row)
            errors.extend(
                f'command_contract_sha256:{reason}' for reason in producer_errors
            )
            if producer and producer_sha:
                row['quality_first_producer_identity'] = producer
                row['quality_first_producer_identity_sha256'] = producer_sha

    top_digest = top_level.get('command_contract_sha256', '')
    if declared and top_digest and declared != top_digest:
        errors.append('command_contract_sha256:nested_top_level_mismatch')
    return (declared if not errors else ''), list(dict.fromkeys(errors))


def _explicit_output_endpoint(row: dict[str, Any]) -> str:
    """Return only a fully attested, hash-bound runtime endpoint identity."""
    task = str(row.get('task') or '').strip().lower()
    stage = str(row.get('stage') or '').strip().lower()
    contract_hash = _strict_sha256_token(row.get('endpoint_contract_hash'))
    attestation = row.get('output_endpoint_attestation')
    if (
        task not in {'classification', 'detection'} or not stage
        or row.get('endpoint_contract_complete') is not True
        or not contract_hash
        or not isinstance(attestation, dict)
    ):
        return ''
    if attestation.get('attested') is not True:
        return ''
    if str(attestation.get('status') or '').strip().lower() != 'passed':
        return ''
    attested_hash = _strict_sha256_token(attestation.get('endpoint_contract_hash'))
    if not attested_hash or attested_hash != contract_hash:
        return ''
    allowed_stages = {
        'classification': {'classification_logits', 'classification_probabilities'},
        'detection': {'raw_head', 'decoded_pre_nms', 'decoded_nms'},
    }
    if stage not in allowed_stages[task]:
        return ''
    attested_stage = str(attestation.get('stage') or '').strip().lower()
    attested_endpoint = str(attestation.get('endpoint') or '').strip().lower()
    if attested_stage != stage or attested_endpoint != stage:
        return ''
    attested_task = str(attestation.get('task') or '').strip().lower()
    if attested_task and attested_task != task:
        return ''
    contract_family = str(row.get('contract_family') or '').strip().lower()
    if contract_family and contract_family != stage:
        return ''
    expected = f"{task}:{stage}:{contract_hash}"
    for raw_explicit in (
        row.get('output_endpoint_id'), attestation.get('output_endpoint_id'),
    ):
        if raw_explicit not in (None, '') and str(raw_explicit).strip().lower() != expected:
            return ''
    return expected


def _strict_direct_completion_projection(
    row: Mapping[str, Any], completion: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify and project the Direct-BN6 contract sealed in a Full command."""
    if (
        _verify_frozen_decoded_nms_normalization_contract is None
        or _build_normalized_detection_endpoint_attestation is None
        or not isinstance(completion, Mapping)
    ):
        return {}
    full_contract = row.get('full_command_contract')
    if not isinstance(full_contract, Mapping):
        return {}
    full_contract_sha = _strict_sha256_token(
        full_contract.get('contract_sha256')
    )
    full_body = dict(full_contract)
    full_body.pop('contract_sha256', None)
    duplicate_full_sha = _strict_sha256_token(
        row.get('full_command_contract_sha256')
    )
    if (
        not full_contract_sha
        or _canonical_json_sha256(full_body) != full_contract_sha
        or (
            duplicate_full_sha
            and duplicate_full_sha != full_contract_sha
        )
    ):
        return {}
    workload = full_contract.get('energy_workload')
    raw_direct = (
        workload.get('frozen_decoded_nms_normalization_contract')
        if isinstance(workload, Mapping) else None
    )
    if not isinstance(raw_direct, Mapping):
        return {}
    try:
        verified_direct = (
            _verify_frozen_decoded_nms_normalization_contract(
                raw_direct,
            )
        )
        result = completion.get(
            'frozen_decoded_nms_normalization_result'
        )
        if not isinstance(result, Mapping):
            return {}
        expected_completion = (
            _build_normalized_detection_endpoint_attestation(
                verified_direct,
                result,
                completed_frames=completion.get('completed_frames'),
                postprocess_completed_frames=completion.get(
                    'postprocess_completed_frames'
                ),
            )
        )
    except Exception:
        return {}
    direct_sha = _strict_sha256_token(
        verified_direct.get('contract_sha256')
    )
    source_hash = _strict_sha256_token(
        verified_direct.get('source_endpoint_contract_hash')
    )
    source_id = str(
        verified_direct.get('source_output_endpoint_id') or ''
    ).strip()
    source_signature = verified_direct.get(
        'source_output_tensor_signature'
    )
    source_attestation = row.get('output_endpoint_attestation')
    declared_source_contract = (
        source_attestation.get('declared_contract')
        if isinstance(source_attestation, Mapping) else None
    )
    row_source_id = str(row.get('output_endpoint_id') or '').strip()
    if (
        dict(expected_completion) != dict(completion)
        or not direct_sha
        or not source_hash
        or source_id != f'detection:decoded_nms:{source_hash}'
        or not isinstance(source_signature, Mapping)
        or not isinstance(source_attestation, Mapping)
        or source_attestation.get('schema')
        != 'onnx-splitpoint/runtime-output-endpoint-attestation'
        or source_attestation.get('schema_version') != 3
        or isinstance(source_attestation.get('schema_version'), bool)
        or source_attestation.get('attested') is not True
        or str(source_attestation.get('status') or '').strip().lower()
        != 'passed'
        or str(source_attestation.get('endpoint') or '').strip().lower()
        != 'decoded_nms'
        or str(source_attestation.get('stage') or '').strip().lower()
        != 'decoded_nms'
        or source_attestation.get(
            'values_decoded_xyxy_score_class'
        ) is not True
        or source_attestation.get('declaration_attested') is not True
        or _strict_sha256_token(
            source_attestation.get('endpoint_contract_hash')
        ) != source_hash
        or dict(source_attestation.get('tensor_signature') or {})
        != dict(source_signature)
        or not isinstance(declared_source_contract, Mapping)
        or str(declared_source_contract.get('model_id') or '').strip()
        != str(verified_direct.get('model_id') or '').strip()
        or declared_source_contract.get('source_coordinate_space')
        != 'model_input_letterbox_xyxy_pixels'
        or _canonical_json_sha256(source_attestation)
        != _strict_sha256_token(
            verified_direct.get(
                'source_output_endpoint_attestation_sha256'
            )
        )
        or _strict_sha256_token(row.get('endpoint_contract_hash'))
        != source_hash
        or (row_source_id and row_source_id != source_id)
        or _strict_sha256_token(
            workload.get(
                'frozen_decoded_nms_normalization_contract_sha256'
            )
        ) != direct_sha
        or _strict_sha256_token(
            workload.get('source_endpoint_contract_hash')
        ) != source_hash
        or str(workload.get('source_output_endpoint_id') or '').strip()
        != source_id
        or dict(workload.get('source_output_tensor_signature') or {})
        != dict(source_signature)
        or _strict_sha256_token(
            workload.get(
                'source_output_endpoint_attestation_sha256'
            )
        ) != _strict_sha256_token(
            verified_direct.get(
                'source_output_endpoint_attestation_sha256'
            )
        )
        or dict(
            workload.get('completed_task_endpoint_attestation') or {}
        ) != dict(completion)
        or _strict_sha256_token(
            completion.get(
                'frozen_decoded_nms_normalization_contract_sha256'
            )
        ) != direct_sha
    ):
        return {}
    return {
        'contract': dict(verified_direct),
        'contract_sha256': direct_sha,
        'result': dict(result),
        'status': 'strict_direct_bn6_completion_verified',
    }


def _explicit_completed_task_comparison_endpoint(
    row: dict[str, Any],
) -> str:
    """Return the semantic completed endpoint, never the physical raw hash."""
    if _verify_completed_detection_comparison_endpoint_contract is None:
        return ''
    if str(row.get('task') or '').strip().lower() != 'detection':
        return ''
    if str(row.get('completed_task_stage') or '').strip().lower() != 'decoded_nms':
        return ''
    if (
        str(row.get('completed_task_contract_family') or '').strip().lower()
        != 'decoded_nms'
    ):
        return ''
    if _strict_bool(row.get('completed_task_endpoint_attested')) is not True:
        return ''
    completion = row.get('completed_task_endpoint_attestation')
    if (
        not isinstance(completion, Mapping)
        or completion.get('attested') is not True
        or str(completion.get('status') or '').strip().lower() != 'passed'
    ):
        return ''
    contract = row.get('completed_task_comparison_endpoint_contract')
    completion_mode = str(
        row.get('completed_task_completion_mode') or ''
    ).strip()
    nested_completion_mode = str(
        completion.get('completed_task_completion_mode') or ''
    ).strip()
    frozen_contract: Mapping[str, Any] | None = None
    direct_contract: Mapping[str, Any] | None = None
    if completion_mode == 'native_three_stage_fast_oracle_outside_timing':
        try:
            from scripts.native_producer_energy_plan import _verified_fast_energy_completion
        except ImportError:
            from native_producer_energy_plan import _verified_fast_energy_completion
        try:
            _att, comparison = _verified_fast_energy_completion(
                row, None, row.get('native_command_contract') or {},
            )
        except Exception:
            return ''
        return str(comparison['output_endpoint_id'])
    if completion_mode == 'detection_completion_execution_v1':
        if (
            _verify_detection_completion_execution_contract is None
            or _verify_detection_completion_execution_attestation is None
        ):
            return ''
        native_command = row.get('native_command_contract')
        runtime_options = (
            native_command.get('runtime_options')
            if isinstance(native_command, Mapping) else {}
        )
        raw_execution = (
            row.get('completion_execution_contract')
            if isinstance(
                row.get('completion_execution_contract'), Mapping,
            )
            else runtime_options.get('completion_execution_contract')
            if isinstance(runtime_options, Mapping)
            else None
        )
        try:
            verified_execution = (
                _verify_detection_completion_execution_contract(
                    raw_execution
                )
            )
            verified_attestation = (
                _verify_detection_completion_execution_attestation(
                    completion,
                    execution_contract=verified_execution,
                    expected_observation_relation=(
                        'same_hotloop_sentinel'
                    ),
                )
            )
        except Exception:
            return ''
        source_endpoint = dict(
            verified_execution.get('source_endpoint') or {}
        )
        completed_endpoint = dict(
            verified_execution.get('completed_endpoint_contract') or {}
        )
        execution_comparison = dict(
            verified_execution.get('comparison_endpoint_contract') or {}
        )
        if (
            dict(verified_attestation) != dict(completion)
            or verified_attestation.get(
                'exact_result_claim_bound'
            ) is not True
            or completed_endpoint
            != dict(
                row.get('completed_task_endpoint_contract') or {}
            )
            or execution_comparison != dict(contract or {})
            or source_endpoint.get('endpoint_contract_hash')
            != _strict_sha256_token(row.get('endpoint_contract_hash'))
            or (
                str(row.get('output_endpoint_id') or '').strip()
                and source_endpoint.get('output_endpoint_id')
                != str(row.get('output_endpoint_id') or '').strip()
            )
        ):
            return ''
    elif completion_mode == 'frozen_host_tail':
        raw_frozen = row.get('frozen_host_postprocess_contract')
        if not isinstance(raw_frozen, Mapping):
            return ''
        frozen_contract = raw_frozen
    elif (
        completion_mode
        == 'integrated_accelerator_plus_frozen_normalization'
    ):
        if _verify_frozen_decoded_nms_normalization_contract is None:
            return ''
        full_contract = row.get('full_command_contract')
        workload = (
            full_contract.get('energy_workload')
            if isinstance(full_contract, Mapping) else None
        )
        raw_direct = (
            workload.get('frozen_decoded_nms_normalization_contract')
            if isinstance(workload, Mapping) else None
        )
        if not isinstance(raw_direct, Mapping):
            return ''
        try:
            verified_direct = (
                _verify_frozen_decoded_nms_normalization_contract(
                    raw_direct,
                )
            )
        except Exception:
            return ''
        direct_sha = _strict_sha256_token(
            verified_direct.get('contract_sha256')
        )
        source_hash = _strict_sha256_token(
            verified_direct.get('source_endpoint_contract_hash')
        )
        source_id = str(
            verified_direct.get('source_output_endpoint_id') or ''
        ).strip()
        source_signature = verified_direct.get(
            'source_output_tensor_signature'
        )
        source_attestation = row.get('output_endpoint_attestation')
        declared_source_contract = (
            source_attestation.get('declared_contract')
            if isinstance(source_attestation, Mapping) else None
        )
        row_source_id = str(row.get('output_endpoint_id') or '').strip()
        if (
            not direct_sha
            or not source_hash
            or source_id != f'detection:decoded_nms:{source_hash}'
            or not isinstance(source_signature, Mapping)
            or not isinstance(source_attestation, Mapping)
            or source_attestation.get('schema')
            != 'onnx-splitpoint/runtime-output-endpoint-attestation'
            or source_attestation.get('schema_version') != 3
            or isinstance(source_attestation.get('schema_version'), bool)
            or source_attestation.get('attested') is not True
            or str(source_attestation.get('status') or '').strip().lower()
            != 'passed'
            or str(source_attestation.get('endpoint') or '').strip().lower()
            != 'decoded_nms'
            or str(source_attestation.get('stage') or '').strip().lower()
            != 'decoded_nms'
            or source_attestation.get(
                'values_decoded_xyxy_score_class'
            ) is not True
            or source_attestation.get('declaration_attested') is not True
            or _strict_sha256_token(
                source_attestation.get('endpoint_contract_hash')
            ) != source_hash
            or dict(source_attestation.get('tensor_signature') or {})
            != dict(source_signature)
            or not isinstance(declared_source_contract, Mapping)
            or str(declared_source_contract.get('model_id') or '').strip()
            != str(verified_direct.get('model_id') or '').strip()
            or declared_source_contract.get('source_coordinate_space')
            != 'model_input_letterbox_xyxy_pixels'
            or _canonical_json_sha256(source_attestation)
            != _strict_sha256_token(
                verified_direct.get(
                    'source_output_endpoint_attestation_sha256'
                )
            )
            or _strict_sha256_token(row.get('endpoint_contract_hash'))
            != source_hash
            or (row_source_id and row_source_id != source_id)
            or _strict_sha256_token(
                workload.get(
                    'frozen_decoded_nms_normalization_contract_sha256'
                )
            ) != direct_sha
            or _strict_sha256_token(
                workload.get('source_endpoint_contract_hash')
            ) != source_hash
            or str(workload.get('source_output_endpoint_id') or '').strip()
            != source_id
            or dict(workload.get('source_output_tensor_signature') or {})
            != dict(source_signature)
            or _strict_sha256_token(
                workload.get(
                    'source_output_endpoint_attestation_sha256'
                )
            ) != _strict_sha256_token(
                verified_direct.get(
                    'source_output_endpoint_attestation_sha256'
                )
            )
            or dict(
                workload.get('completed_task_endpoint_attestation') or {}
            ) != dict(completion)
            or _strict_sha256_token(
                completion.get(
                    'frozen_decoded_nms_normalization_contract_sha256'
                )
            ) != direct_sha
        ):
            return ''
        direct_projection = _strict_direct_completion_projection(
            row, completion,
        )
        if not direct_projection:
            return ''
        direct_contract = direct_projection['contract']
    else:
        return ''
    try:
        verified = _verify_completed_detection_comparison_endpoint_contract(
            contract,
            frozen_contract=frozen_contract,
            direct_normalization_contract=direct_contract,
        )
    except Exception:
        return ''
    declared_hash = _strict_sha256_token(
        row.get('completed_task_comparison_endpoint_contract_hash')
    )
    declared_id = str(
        row.get('completed_task_comparison_output_endpoint_id') or ''
    ).strip()
    nested_contract = completion.get(
        'completed_task_comparison_endpoint_contract'
    )
    nested_hash = _strict_sha256_token(
        completion.get(
            'completed_task_comparison_endpoint_contract_hash'
        )
    )
    nested_id = str(
        completion.get(
            'completed_task_comparison_output_endpoint_id'
        ) or ''
    ).strip()
    if (
        declared_hash != verified['endpoint_contract_hash']
        or declared_id != verified['output_endpoint_id']
        or not isinstance(nested_contract, Mapping)
        or dict(nested_contract) != dict(verified)
        or nested_hash != declared_hash
        or nested_id != declared_id
        or nested_completion_mode != completion_mode
    ):
        return ''
    return declared_id


def _comparison_output_endpoint(row: dict[str, Any]) -> str:
    completed = _explicit_completed_task_comparison_endpoint(row)
    if str(row.get('task') or '').strip().lower() == 'detection':
        return completed
    return _explicit_output_endpoint(row)


# Source fields copied by older tests/collectors.  The actual join below uses a
# canonical runtime precision instead of the legacy Full comparison precision.
_QUALITY_IDENTITY_FIELDS = (
    'backend', 'model', 'case', 'precision', 'execution_precision',
    'full_runtime_precision', 'setup_id', 'comparison_backend', 'task', 'stage',
    'endpoint_contract_complete', 'endpoint_contract_hash',
    'output_endpoint_attestation', 'output_endpoint_id',
)

_QUALITY_IDENTITY_NAMES = (
    'backend', 'model', 'case', 'runtime_precision', 'setup_id',
    'comparison_backend', 'output_endpoint_id',
)

# Diagnostic-only subset used to explain a failed exact join.  Runtime
# precision and the physical output endpoint deliberately stay out of this
# key so a candidate from the same workload/hardware row can expose which of
# those two strict identity axes drifted.  This index must never be used to
# attach evidence or relax the seven-axis join.
_QUALITY_NEAREST_BASE_AXIS_NAMES = (
    'backend', 'model', 'case', 'setup_id', 'comparison_backend',
)
_QUALITY_NEAREST_BASE_AXIS_INDEXES = tuple(
    _QUALITY_IDENTITY_NAMES.index(name)
    for name in _QUALITY_NEAREST_BASE_AXIS_NAMES
)
_QUALITY_NEAREST_CANDIDATE_LIMIT = 8

_SUPPORTED_QUALITY_SUMMARY_SCHEMA_VERSIONS = frozenset({5, 6, 7, 8, 9, 10})

_QUALITY_SUMMARY_V10_AXIS_COUNTS = (
    (
        'precision_quality_binding_verified',
        'precision_quality_binding_verified_count',
    ),
    (
        'task_quality_observation_valid',
        'task_quality_observation_valid_count',
    ),
    (
        'quality_claim_result_verified',
        'quality_claim_result_verified_count',
    ),
)


def _quality_summary_schema_valid(summary: Any) -> bool:
    """Validate the summary envelope and every version-specific contract."""
    if not isinstance(summary, dict):
        return False
    schema_version = summary.get('schema_version')
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version not in _SUPPORTED_QUALITY_SUMMARY_SCHEMA_VERSIONS
        or summary.get('schema')
        != 'onnx-splitpoint/native-producer-validation-summary'
        or str(summary.get('status') or '').strip().lower() != 'complete'
        or not isinstance(summary.get('rows'), list)
    ):
        return False
    rows = summary['rows']
    if _strict_nonnegative_int(summary.get('row_count')) != len(rows):
        return False
    if schema_version < 10:
        return True

    # Schema 10 introduced three independent evidence axes.  A payload that
    # merely relabels an older row as v10 must remain invalid and fail closed.
    for row in rows:
        if not isinstance(row, Mapping):
            return False
        if any(
            not isinstance(row.get(axis_name), bool)
            for axis_name, _ in _QUALITY_SUMMARY_V10_AXIS_COUNTS
        ):
            return False
    for axis_name, count_name in _QUALITY_SUMMARY_V10_AXIS_COUNTS:
        declared_count = _strict_nonnegative_int(summary.get(count_name))
        observed_count = sum(row.get(axis_name) is True for row in rows)
        if declared_count is None or declared_count != observed_count:
            return False
    return True

# Fields evaluated or normalized by the validation summary must remain attached
# to the exact performance identity.  This includes explicit False/zero values;
# checking truthiness here would silently turn failed or empty evidence into
# "unknown" in the Combined and Scientific reports.
_QUALITY_REPORT_FIELDS = (
    'claim_ok',
    'claim_ok_source',
    'claim_structural_gate_pass',
    'claim_structural_gate_reason',
    'claim_ok_structural_clamped',
    'semantic_ok',
    'contract_consistent',
    'structural_contract_pass',
    'structural_contract_status',
    'structural_contract_reason',
    'numerical_similarity_pass',
    'numerical_similarity_status',
    'numerical_similarity_reason',
    'numerical_similarity_scope',
    'numerical_similarity_metric',
    'numerical_similarity_value',
    'numerical_similarity_threshold',
    'numerical_similarity_mean_iou',
    'numerical_similarity_mean_iou_threshold',
    'numerical_similarity_policy_id',
    'task_quality_pass',
    'task_quality_status',
    'task_quality_reason',
    'source_e2e_scope',
    'e2e_scope',
    'e2e_claim_eligible',
    'e2e_contract_reason',
    'comparison_endpoint_stratum',
    'measurement_concurrency',
    'requires_host_decode_nms',
    'postprocess_included',
    'postprocess_location',
    'host_postprocess_frozen',
    'host_postprocessing_available',
    'host_tail_available',
    'host_postprocess_required',
    'host_tail_required',
    'host_postprocessing_evidence_status',
    'host_postprocessing_evidence_source',
    'host_postprocessing_legacy_alias_conflict',
    'decoder_contract_pass',
    'nms_ok',
    'decoder_id',
    'postprocess_completed_frames',
    'postprocess_completion_verified',
    'frozen_host_postprocess_contract',
    'frozen_host_postprocess_contract_sha256',
    'frozen_host_postprocess_result',
    'task',
    'stage',
    'output_format',
    'contract_family',
    'contract_source',
    'endpoint_contract_complete',
    'endpoint_contract_hash',
    'output_endpoint_attestation',
    'accelerator_output_stage',
    'accelerator_output_contract_family',
    'accelerator_endpoint_contract_hash',
    'accelerator_output_endpoint_attestation',
    'output_endpoint_id',
    'physical_output_endpoint_id',
    'comparison_output_endpoint_id',
    'output_endpoint_match',
    'comparison_endpoint_match',
    'output_endpoint_comparison_stratum',
    'comparison_stratum_explicit',
    'completed_task_stage',
    'completed_task_contract_family',
    'completed_task_endpoint_contract',
    'completed_task_endpoint_contract_hash',
    'completed_task_output_endpoint_id',
    'completed_task_comparison_endpoint_contract',
    'completed_task_comparison_endpoint_contract_hash',
    'completed_task_comparison_output_endpoint_id',
    'completed_task_completion_mode',
    'completed_task_endpoint_attested',
    'completed_task_endpoint_attestation',
    'completed_task_endpoint_attestation_status',
    'completed_task_endpoint_projection_status',
)

_PHYSICAL_EXECUTION_ATTESTATION_FIELDS = (
    'stage',
    'output_format',
    'contract_family',
    'contract_source',
    'endpoint_contract_complete',
    'endpoint_contract_hash',
    'output_endpoint_attestation',
    'output_endpoint_id',
    'physical_output_endpoint_id',
)

_QUALITY_BINDING_ALIASES: dict[str, tuple[str, ...]] = {
    'command_contract_sha256': (
        'full_command_contract_sha256', 'native_command_contract_sha256',
        'source_contract_sha256',
    ),
    'model_sha256': (
        'model_sha256', 'source_onnx_sha256', 'onnx_sha256',
    ),
    'runtime_input_sha256': (
        'runtime_input_sha256', 'input_image_sha256',
        'prepared_feed_source_image_sha256', 'validation_input_or_image_sha256',
    ),
    'validation_dataset_sha256': (
        'validation_dataset_sha256', 'dataset_sha256', 'dataset_fingerprint',
    ),
    'validation_dataset_image_ids_sha256': (
        'validation_dataset_image_ids_sha256', 'validation_image_ids_sha256',
        'dataset_image_ids_sha256', 'image_ids_sha256',
    ),
    'validation_dataset_ground_truth_sha256': (
        'validation_dataset_ground_truth_sha256', 'validation_ground_truth_sha256',
        'dataset_ground_truth_sha256', 'ground_truth_sha256',
    ),
    'accuracy_gate_policy_sha256': (
        'accuracy_gate_policy_sha256',
    ),
    'task_quality_policy_sha256': (
        'task_quality_policy_sha256',
    ),
    'source_request_sha256': (
        'source_request_sha256', 'quality_source_sha256',
        'native_split_quality_source_request_sha256',
    ),
    'native_split_quality_central_result_sha256': (
        'native_split_quality_central_result_sha256',
    ),
    'native_split_quality_selection_sha256': (
        'native_split_quality_selection_sha256',
    ),
    'quality_contract_sha256': (
        'quality_contract_sha256',
    ),
    'preprocessing_contract_sha256': (
        'preprocessing_contract_sha256',
    ),
    'decoder_contract_sha256': (
        'decoder_contract_sha256',
    ),
    'nms_contract_sha256': (
        'nms_contract_sha256',
    ),
    'quality_record_endpoint_contract_sha256': (
        'quality_record_endpoint_contract_sha256',
    ),
    'output_manifest_sha256': (
        'output_manifest_sha256', 'dump_manifest_sha256',
    ),
}


def _is_full_row(row: dict[str, Any]) -> bool:
    return bool(
        str(row.get('execution_mode') or '') == 'native_full_baseline'
        or str(row.get('backend') or '').startswith('native_full_')
        or str(row.get('case') or '').strip().lower() == 'full'
    )


def _runtime_precision_identity(row: dict[str, Any]) -> str:
    """Return actual runtime precision; never reuse Full comparison precision."""
    backend = str(row.get('backend') or '').strip().lower()
    scalar = {'fp16', 'fp32', 'int8'}
    def valid(value: Any) -> str:
        normalized = str(value or '').strip().lower().replace(' ', '')
        if normalized in scalar or re.fullmatch(
            r'(?:uint8_cast|uint8_dequant|float32_layout)_(?:fp16|fp32|int8)',
            normalized,
        ):
            return normalized
        artifact = re.fullmatch(
            r'(deepx_dxnn_sha256|hailo_hef_sha256):([0-9a-f]{64})',
            normalized,
        )
        if artifact:
            prefix = artifact.group(1)
            if prefix == 'deepx_dxnn_sha256' and 'deepx' in backend:
                return normalized
            if prefix == 'hailo_hef_sha256' and 'hailo' in backend:
                return normalized
        return ''

    explicit_identity = row.get('runtime_precision_identity')
    explicit = valid(explicit_identity)
    if explicit_identity not in (None, '') and not explicit:
        return ''
    if _is_full_row(row):
        primary_value = row.get('full_runtime_precision') or row.get('execution_precision')
    else:
        primary_value = row.get('execution_precision') or row.get('precision')
    primary = valid(primary_value)
    if primary_value not in (None, ''):
        return primary
    if explicit:
        return explicit
    command = row.get('full_command_contract') if isinstance(row.get('full_command_contract'), dict) else {}
    artifacts = command.get('artifacts') if isinstance(command.get('artifacts'), dict) else {}
    artifact_name = ''
    token_prefix = ''
    if backend == 'native_full_deepx':
        artifact_name, token_prefix = 'dxnn', 'deepx_dxnn_sha256:'
    elif backend in {'native_full_hailo8', 'native_full_hailo10h'}:
        artifact_name, token_prefix = 'hef', 'hailo_hef_sha256:'
    artifact = artifacts.get(artifact_name) if isinstance(artifacts.get(artifact_name), dict) else {}
    digest = str(artifact.get('sha256') or '').strip().lower()
    if token_prefix and len(digest) == 64 and all(ch in '0123456789abcdef' for ch in digest):
        return token_prefix + digest
    return ''


def _quality_binding_evidence(
    row: dict[str, Any],
) -> tuple[dict[str, str], list[str]]:
    """Return canonical, conflict-free provenance plus explicit error fields.

    Every populated alias participates.  Thus a valid canonical value cannot
    hide a malformed or contradictory legacy duplicate in the same row.
    """
    values: dict[str, str] = {}
    errors: list[str] = []
    for canonical, aliases in _QUALITY_BINDING_ALIASES.items():
        observed: set[str] = set()
        invalid = False
        for alias in aliases:
            if row.get(alias) not in (None, ''):
                normalized = _strict_sha256_token(row.get(alias))
                if not normalized:
                    invalid = True
                else:
                    observed.add(normalized)
        if invalid:
            errors.append(f'{canonical}:malformed')
        if len(observed) > 1:
            errors.append(f'{canonical}:conflict')
        values[canonical] = next(iter(observed)) if len(observed) == 1 and not invalid else ''
    return values, errors


def _quality_binding_values(row: dict[str, Any]) -> dict[str, str]:
    """Compatibility wrapper used by older callers and diagnostics."""
    return _quality_binding_evidence(row)[0]


def _required_quality_bindings(task: str) -> tuple[str, ...]:
    required = [
        'source_request_sha256', 'model_sha256',
        'validation_dataset_sha256',
        'validation_dataset_image_ids_sha256',
        'validation_dataset_ground_truth_sha256',
        'accuracy_gate_policy_sha256', 'task_quality_policy_sha256',
        'quality_contract_sha256', 'preprocessing_contract_sha256',
        'quality_record_endpoint_contract_sha256',
    ]
    if str(task or '').strip().lower() == 'detection':
        required.extend(('decoder_contract_sha256', 'nms_contract_sha256'))
    return tuple(required)


def _quality_identity(row: dict[str, Any]) -> tuple[str, ...]:
    """Return the exact, frozen identity used for quality evidence joins."""
    return (
        str(row.get('backend') or '').strip().lower(),
        str(row.get('model') or '').strip(),
        str(row.get('case') or '').strip().lower(),
        _runtime_precision_identity(row),
        str(row.get('setup_id') or '').strip().lower(),
        str(row.get('comparison_backend') or '').strip().lower(),
        _explicit_output_endpoint(row),
    )


def _quality_nearest_base_identity(
    identity: tuple[str, ...],
) -> tuple[str, ...]:
    """Return the stable five-axis key used only for join diagnostics."""
    return tuple(identity[index] for index in _QUALITY_NEAREST_BASE_AXIS_INDEXES)


def _quality_nearest_candidate_diagnostics(
    identity: tuple[str, ...],
    candidates: list[tuple[int, tuple[str, ...]]],
) -> tuple[int, list[dict[str, Any]]]:
    """Describe same-workload candidates without participating in the join."""
    diagnostics: list[dict[str, Any]] = []
    for row_index, candidate_identity in sorted(
        candidates, key=lambda item: item[0],
    )[:_QUALITY_NEAREST_CANDIDATE_LIMIT]:
        differing_axes = {
            name: {
                'performance': identity[index],
                'quality': candidate_identity[index],
            }
            for index, name in enumerate(_QUALITY_IDENTITY_NAMES)
            if identity[index] != candidate_identity[index]
        }
        diagnostics.append({
            'quality_row_index': row_index,
            'quality_identity': dict(zip(
                _QUALITY_IDENTITY_NAMES, candidate_identity,
            )),
            'differing_axes': differing_axes,
        })
    return len(candidates), diagnostics


def _report_evidence_present(value: Any) -> bool:
    return value not in (None, "", {}, [])


def _report_evidence_equal(left: Any, right: Any) -> bool:
    if isinstance(left, (Mapping, list, tuple)) or isinstance(
        right, (Mapping, list, tuple),
    ):
        try:
            return json.dumps(
                left,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ) == json.dumps(
                right,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
        except (TypeError, ValueError):
            return False
    return left == right


def _attach_quality_evidence(
    rows: list[dict[str, Any]],
    quality_summary: dict[str, Any] | None,
    *,
    quality_summary_source: str = '',
    quality_summary_status: str = '',
    remote_execution_contexts: Iterable[Mapping[str, Any]] = (),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Attach explicit validation rows by an exact, fail-closed identity join.

    Runtime booleans are diagnostics, not a dataset-quality attestation.  A
    variant is verified only by one supported complete validation row with the
    same runtime precision, hardware/comparison identity, attested endpoint and
    every available frozen provenance hash.
    """
    schema_ok = _quality_summary_schema_valid(quality_summary)
    payload_ok = schema_ok and quality_summary_status in {'', 'loaded'}
    if not quality_summary_status:
        quality_summary_status = 'loaded' if payload_ok else ('not_provided' if quality_summary is None else 'invalid')
    index: dict[tuple[str, ...], list[tuple[int, dict[str, Any]]]] = {}
    nearest_index: dict[
        tuple[str, ...], list[tuple[int, tuple[str, ...]]]
    ] = {}
    if payload_ok:
        for row_index, quality_row in enumerate(quality_summary.get('rows') or []):
            if not isinstance(quality_row, dict):
                continue
            quality_identity = _quality_identity(quality_row)
            index.setdefault(quality_identity, []).append((row_index, quality_row))
            nearest_index.setdefault(
                _quality_nearest_base_identity(quality_identity), [],
            ).append((row_index, quality_identity))

    counts = {'verified': 0, 'failed': 0, 'missing': 0, 'ambiguous': 0}
    for row in rows:
        identity = _quality_identity(row)
        matches = index.get(identity, []) if payload_ok else []
        nearest_candidate_count = 0
        nearest_candidates: list[dict[str, Any]] = []
        if payload_ok and not matches:
            (
                nearest_candidate_count, nearest_candidates,
            ) = _quality_nearest_candidate_diagnostics(
                identity,
                nearest_index.get(
                    _quality_nearest_base_identity(identity), [],
                ),
            )
        if not payload_ok:
            match_status = 'summary_not_provided' if quality_summary_status == 'not_provided' else 'summary_invalid_or_unreadable'
        elif not matches:
            match_status = 'no_exact_identity_match'
        elif len(matches) > 1:
            match_status = 'ambiguous_exact_identity_match'
        else:
            match_status = 'exact_identity_match'

        quality_row = matches[0][1] if len(matches) == 1 else {}
        physical_evidence_conflicts: list[str] = []
        physical_evidence_fill: dict[str, Any] = {}
        if len(matches) == 1:
            for field in _PHYSICAL_EXECUTION_ATTESTATION_FIELDS:
                if field not in quality_row:
                    continue
                quality_value = quality_row.get(field)
                performance_value = row.get(field)
                if not _report_evidence_present(quality_value):
                    continue
                if not _report_evidence_present(performance_value):
                    physical_evidence_fill[field] = quality_value
                elif not _report_evidence_equal(
                    performance_value, quality_value,
                ):
                    physical_evidence_conflicts.append(field)
        performance_bindings, performance_binding_errors = _quality_binding_evidence(row)
        _nested_command_digest, nested_command_errors = (
            _verified_performance_command_contract(
                row,
                remote_execution_contexts=remote_execution_contexts,
            )
        )
        performance_binding_errors = list(dict.fromkeys(
            [*performance_binding_errors, *nested_command_errors]
        ))
        quality_bindings, quality_binding_errors = _quality_binding_evidence(quality_row)
        is_trt_full = (
            str(row.get('backend') or '').strip().lower()
            == 'native_full_tensorrt'
        )
        full_contract = row.get('full_command_contract')
        has_quality_first_marker = bool(
            row.get('quality_first_producer_identity_sha256')
            or quality_row.get('quality_first_producer_identity_sha256')
            or (
                isinstance(full_contract, Mapping)
                and (
                    full_contract.get('quality_first_producer_identity')
                    or full_contract.get('quality_first_producer_identity_sha256')
                )
            )
        )
        strict_trt_quality_first = bool(
            is_trt_full
            and (
                int((quality_summary or {}).get('schema_version') or 0) >= 7
                or has_quality_first_marker
            )
        )
        legacy_trt_quality_first_unbound = bool(
            is_trt_full and not strict_trt_quality_first
        )
        performance_producer: dict[str, Any] = {}
        performance_producer_sha = ''
        quality_producer: dict[str, Any] = {}
        quality_producer_sha = ''
        if strict_trt_quality_first:
            (
                performance_producer, performance_producer_sha,
                performance_producer_errors,
            ) = _performance_quality_first_binding(row)
            quality_producer, quality_producer_sha, quality_producer_errors = (
                _validation_quality_first_binding(quality_row)
            )
            performance_binding_errors.extend(
                f'quality_first:{reason}' for reason in performance_producer_errors
            )
            quality_binding_errors.extend(
                f'quality_first:{reason}' for reason in quality_producer_errors
            )
        binding_mismatches = [
            name for name, expected in performance_bindings.items()
            if expected and quality_bindings.get(name) != expected
        ]
        if strict_trt_quality_first and (
            not performance_producer_sha
            or not quality_producer_sha
            or performance_producer_sha != quality_producer_sha
            or performance_producer != quality_producer
        ):
            binding_mismatches.append('quality_first_producer_identity')
        required_bindings = _required_quality_bindings(
            str(quality_row.get('task') or row.get('task') or '')
        )
        quality_required_bindings = (
            'command_contract_sha256', *required_bindings,
        )
        performance_required_bindings = ('command_contract_sha256',)
        split_quality_required = bool(
            row.get('native_split_quality_required') is True
            or quality_row.get('native_split_quality_required') is True
        )
        if split_quality_required:
            split_selection_bindings = (
                'source_request_sha256',
                'native_split_quality_central_result_sha256',
                'native_split_quality_selection_sha256',
            )
            quality_required_bindings = (
                *quality_required_bindings, *split_selection_bindings,
            )
            performance_required_bindings = (
                *performance_required_bindings, *split_selection_bindings,
            )
        missing_quality_bindings = [
            name for name in quality_required_bindings
            if not quality_bindings.get(name)
        ]
        missing_performance_bindings = [
            name for name in performance_required_bindings
            if not performance_bindings.get(name)
        ]
        if strict_trt_quality_first:
            if not performance_producer_sha:
                missing_performance_bindings.append(
                    'quality_first_producer_identity_sha256'
                )
            if not quality_producer_sha:
                missing_quality_bindings.append(
                    'quality_first_producer_identity_sha256'
                )
        quality_policy_values = {
            quality_bindings.get(name) for name in (
                'accuracy_gate_policy_sha256', 'task_quality_policy_sha256',
            ) if quality_bindings.get(name)
        }
        policy_consistent = len(quality_policy_values) == 1
        identity_complete = all(identity)
        if len(matches) == 1 and not identity_complete:
            match_status = 'exact_base_identity_incomplete'
        elif len(matches) == 1 and (
            performance_binding_errors or quality_binding_errors
        ):
            match_status = 'exact_identity_provenance_malformed_or_conflicting'
        elif len(matches) == 1 and (
            missing_quality_bindings or missing_performance_bindings
        ):
            match_status = 'exact_identity_provenance_incomplete'
        elif len(matches) == 1 and not policy_consistent:
            match_status = 'exact_identity_quality_policy_mismatch'
        elif len(matches) == 1 and binding_mismatches:
            match_status = 'exact_identity_provenance_mismatch'
        elif len(matches) == 1 and physical_evidence_conflicts:
            match_status = 'exact_identity_physical_evidence_conflict'
        binding_verified = bool(
            len(matches) == 1
            and identity_complete
            and not performance_binding_errors
            and not quality_binding_errors
            and not missing_quality_bindings
            and not missing_performance_bindings
            and policy_consistent
            and not binding_mismatches
            and not physical_evidence_conflicts
            and quality_row.get('central_quality_evidence_verified') is True
            and quality_row.get('accuracy_gate_policy_match') is True
        )
        accuracy_gate_pass = quality_row.get('accuracy_gate_pass')
        task_quality_observation_valid = bool(
            binding_verified
            and isinstance(accuracy_gate_pass, bool)
            and isinstance(quality_row.get('task_valid'), bool)
        )
        legacy_quality_evidence_verified = bool(
            binding_verified and task_quality_observation_valid
        )
        claim_result_verified = bool(
            binding_verified
            and task_quality_observation_valid
            and quality_row.get('ok') is True
            and quality_row.get('claim_ok') is True
            and quality_row.get('contract_consistent') is True
            and quality_row.get('task_valid') is True
            and accuracy_gate_pass is True
            and quality_row.get('eligible_for_ranking') is True
        )
        if binding_verified:
            counts['verified'] += 1
        elif len(matches) > 1:
            counts['ambiguous'] += 1
        elif not matches:
            counts['missing'] += 1
        else:
            counts['failed'] += 1

        row.update({
            'quality_summary_source': quality_summary_source,
            'quality_summary_status': quality_summary_status,
            'quality_summary_schema_valid': schema_ok,
            'quality_identity': dict(zip(_QUALITY_IDENTITY_NAMES, identity)),
            'quality_runtime_precision': identity[3],
            'quality_identity_complete': identity_complete,
            'quality_binding_expected': performance_bindings,
            'quality_binding_observed': quality_bindings,
            'quality_binding_required': {
                'performance': list(performance_required_bindings),
                'quality': list(quality_required_bindings),
            },
            'quality_binding_missing': {
                'performance': missing_performance_bindings,
                'quality': missing_quality_bindings,
            },
            'quality_binding_errors': {
                'performance': performance_binding_errors,
                'quality': quality_binding_errors,
            },
            'quality_policy_binding_consistent': policy_consistent,
            'quality_binding_mismatches': binding_mismatches,
            'quality_physical_evidence_conflict': bool(
                physical_evidence_conflicts
            ),
            'quality_physical_evidence_conflict_fields': list(
                physical_evidence_conflicts
            ),
            'quality_first_producer_identity_sha256': performance_producer_sha,
            'quality_first_producer_identity_match': bool(
                strict_trt_quality_first
                and performance_producer_sha
                and performance_producer_sha == quality_producer_sha
                and performance_producer == quality_producer
            ) if strict_trt_quality_first else None,
            'quality_first_binding_status': (
                'exact_identity_match' if strict_trt_quality_first
                and performance_producer_sha
                and performance_producer_sha == quality_producer_sha
                and performance_producer == quality_producer
                else 'legacy_unbound_nonclaimable'
                if legacy_trt_quality_first_unbound else ''
            ),
            'quality_match_status': match_status,
            'quality_match_count': len(matches),
            'quality_row_index': matches[0][0] if len(matches) == 1 else None,
            'quality_nearest_candidate_count': nearest_candidate_count,
            'quality_nearest_candidates': nearest_candidates,
            'quality_contract_consistent': quality_row.get('contract_consistent'),
            'quality_central_evidence_verified': quality_row.get('central_quality_evidence_verified'),
            'quality_task_valid': quality_row.get('task_valid'),
            'quality_accuracy_gate_pass': quality_row.get('accuracy_gate_pass'),
            'quality_eligible_for_ranking': quality_row.get('eligible_for_ranking'),
            'quality_gate_status': quality_row.get('gate_status'),
            'quality_accuracy_gate_reason': quality_row.get('accuracy_gate_reason'),
            'quality_ranking_exclusion_reason': quality_row.get('ranking_exclusion_reason'),
            # v2.72 separates exact binding, a well-typed observation and its
            # positive result.  Legacy verification fields keep their stricter
            # fail-closed projection so malformed observations cannot be
            # uplifted merely because their provenance happens to bind.
            'quality_evidence_verified': legacy_quality_evidence_verified,
            'precision_quality_verified': legacy_quality_evidence_verified,
            'precision_quality_binding_verified': binding_verified,
            'task_quality_observation_valid': task_quality_observation_valid,
            'quality_claim_result_verified': claim_result_verified,
        })
        if len(matches) == 1:
            row.update(physical_evidence_fill)
            for report_field in _QUALITY_REPORT_FIELDS:
                if report_field in _PHYSICAL_EXECUTION_ATTESTATION_FIELDS:
                    continue
                if report_field not in quality_row:
                    continue
                quality_value = quality_row.get(report_field)
                # Empty compatibility placeholders must not erase richer
                # physical evidence already carried by the timing row.
                # Explicit False and zero remain present decisions.
                if (
                    not _report_evidence_present(quality_value)
                    and _report_evidence_present(row.get(report_field))
                ):
                    continue
                row[report_field] = quality_value
            if physical_evidence_conflicts:
                row.update({
                    'claim_ok': False,
                    'contract_consistent': False,
                    'structural_contract_pass': False,
                    'structural_contract_status': 'failed',
                    'structural_contract_reason': (
                        'quality_physical_evidence_conflict:'
                        + ','.join(physical_evidence_conflicts)
                    ),
                    'quality_evidence_verified': False,
                    'precision_quality_verified': False,
                    'precision_quality_binding_verified': False,
                    'task_quality_observation_valid': False,
                    'quality_claim_result_verified': False,
                })
        if binding_verified:
            row['endpoint_contract_hash'] = identity[6].rsplit(':', 1)[-1]
            imported_names = {
                'source_request_sha256', 'model_sha256',
                'validation_dataset_sha256',
                'validation_dataset_image_ids_sha256',
                'validation_dataset_ground_truth_sha256',
                'accuracy_gate_policy_sha256', 'task_quality_policy_sha256',
                'runtime_quality_gate_policy_sha256',
                'quality_contract_sha256', 'preprocessing_contract_sha256',
                'decoder_contract_sha256', 'nms_contract_sha256',
                'quality_record_endpoint_contract_sha256',
            }
            for name in imported_names:
                if quality_bindings.get(name):
                    row[name] = quality_bindings[name]
            if strict_trt_quality_first:
                row['quality_first_producer_identity'] = performance_producer

    metadata = {
        'source': quality_summary_source,
        'status': quality_summary_status,
        'schema': quality_summary.get('schema') if payload_ok else None,
        'schema_version': quality_summary.get('schema_version') if payload_ok else None,
        'schema_valid': schema_ok,
        'quality_row_count': sum(len(matches) for matches in index.values()),
        'unique_identity_count': len(index),
        'duplicate_identity_count': sum(1 for matches in index.values() if len(matches) > 1),
        'performance_row_count': len(rows),
        **{f'{key}_performance_row_count': value for key, value in counts.items()},
    }
    return rows, metadata


def _quality_precision_verified(row: dict[str, Any]) -> bool:
    # Never infer this from the runtime row.  It becomes true only through the
    # exact validation-summary join in ``_attach_quality_evidence``.
    binding_verified = (
        row.get('precision_quality_binding_verified') is True
        if 'precision_quality_binding_verified' in row
        else row.get('quality_evidence_verified') is True
    )
    if str(row.get('backend') or '').strip().lower() == 'native_full_tensorrt':
        # v5/v6 summaries remain readable for diagnostics, but without the
        # Quality-FIRST producer chain they cannot release a scientific claim.
        binding_status = str(row.get('quality_first_binding_status') or '')
        if binding_status == 'legacy_unbound_nonclaimable':
            return False
        if binding_status:
            return bool(
                binding_verified
                and row.get('quality_first_producer_identity_match') is True
            )
    return binding_verified


_CLAIMABLE_REPETITION_RUNTIME_SCOPES = {
    'fresh_runtime_per_repetition',
    'fresh_process_per_repetition',
    'fresh_hailo_vstreams_trt_completion_runtime_per_repetition',
}


def _repeat_record_id(record: dict[str, Any]) -> str:
    for key in ('repetition_id', 'repetition_uuid', 'repetition_index', 'measurement_repetition_index'):
        value = record.get(key)
        if value not in (None, '') and not isinstance(value, bool):
            return str(value).strip()
    return ''


def _validate_repeat_claim_evidence(row: dict[str, Any]) -> tuple[bool, list[str]]:
    """Recompute repeat statistics from unique raw, fresh-runtime records.

    Summary counts and pre-computed confidence intervals are never sufficient
    evidence.  This gate validates the raw records, rejects best-of selection,
    and replaces the thesis-facing median/CI with a deterministic recomputation.
    """
    reasons: list[str] = []
    if row.get('repetition_claim_identity_drift') is True:
        reasons.append('repetition_claim_identity_drift')
    if row.get('repetition_claim_identity_incomplete') is True:
        reasons.append('repetition_claim_identity_incomplete')
    records_value = row.get('repetition_records')
    if not isinstance(records_value, list) or not records_value:
        records: list[dict[str, Any]] = []
        reasons.append('repetition_raw_records_missing')
    elif any(not isinstance(record, dict) for record in records_value):
        records = [record for record in records_value if isinstance(record, dict)]
        reasons.append('repetition_raw_record_schema_invalid')
    else:
        records = [dict(record) for record in records_value]

    requested = _strict_nonnegative_int(row.get('repetition_count_requested'))
    attempted = _strict_nonnegative_int(row.get('repetition_count_attempted'))
    valid = _strict_nonnegative_int(row.get('repetition_count_valid'))
    if requested is None or attempted is None or valid is None:
        reasons.append('repetition_counts_missing_or_invalid')
    elif not (requested == attempted == valid == len(records) and valid >= 3):
        reasons.append('repetition_counts_not_exact_or_below_three')

    if str(row.get('repetition_status') or '').strip().lower() != 'complete':
        reasons.append('repetition_status_not_complete')
    aggregation = str(row.get('repetition_aggregation') or '').strip().lower()
    known_median_contract = bool(
        aggregation.startswith('median_')
        or aggregation == 'median'
        or aggregation.startswith('median-with-')
    )
    unsafe_selection = bool(
        any(token in aggregation for token in ('fastest', 'minimum', 'maximum', 'best-of', 'best_of'))
        and aggregation != 'median_never_best_of'
    )
    if not aggregation or not known_median_contract or unsafe_selection:
        reasons.append('repetition_aggregation_not_median_or_best_of')

    runtime_scope = str(row.get('repetition_runtime_scope') or '').strip().lower()
    independence_verified = _strict_bool(row.get('repetition_independence_verified'))
    if runtime_scope not in _CLAIMABLE_REPETITION_RUNTIME_SCOPES:
        reasons.append('repetition_runtime_scope_not_independent')
    if independence_verified is not True:
        reasons.append('repetition_independence_not_verified')

    repeat_ids = [_repeat_record_id(record) for record in records]
    if any(not repeat_id for repeat_id in repeat_ids):
        reasons.append('repetition_id_missing')
    elif len(set(repeat_ids)) != len(repeat_ids):
        reasons.append('repetition_id_duplicate')

    runtime_ids = [str(record.get('runtime_instance_id') or '').strip() for record in records]
    if any(not runtime_id for runtime_id in runtime_ids):
        reasons.append('repetition_runtime_instance_id_missing')
    elif len(set(runtime_ids)) != len(runtime_ids):
        reasons.append('repetition_runtime_instance_id_duplicate')

    work_units: list[int] = []
    workload_hashes: list[str] = []
    for record in records:
        units_value = next((
            record.get(key) for key in ('completed_work_units', 'completed_frames', 'frames')
            if record.get(key) not in (None, '')
        ), None)
        units = _strict_nonnegative_int(units_value)
        if units is None or units <= 0:
            reasons.append('repetition_completed_work_units_missing_or_nonpositive')
        else:
            work_units.append(units)
        workload_hash = str(
            record.get('workload_contract_sha256')
            or record.get('native_command_contract_sha256')
            or record.get('full_command_contract_sha256')
            or ''
        ).strip().lower()
        if workload_hash.startswith('sha256:'):
            workload_hash = workload_hash.split(':', 1)[1]
        if len(workload_hash) != 64 or any(ch not in '0123456789abcdef' for ch in workload_hash):
            reasons.append('repetition_workload_contract_hash_missing_or_invalid')
        else:
            workload_hashes.append(workload_hash)
    if work_units and len(work_units) == len(records) and len(set(work_units)) != 1:
        reasons.append('repetition_completed_work_units_mismatch')
    if workload_hashes and len(workload_hashes) == len(records) and len(set(workload_hashes)) != 1:
        reasons.append('repetition_workload_contract_mismatch')

    fps_values: list[float] = []
    for record in records:
        if _strict_bool(record.get('ok')) is not True:
            reasons.append('repetition_record_not_ok')
            continue
        status = str(record.get('status') or '').strip().lower()
        if status and status not in {'ok', 'complete', 'passed', 'pass'}:
            reasons.append('repetition_record_status_not_ok')
        fps = _finite_num(record.get('fps_makespan'))
        if fps is None or fps <= 0:
            reasons.append('repetition_record_fps_nonfinite_or_nonpositive')
        else:
            fps_values.append(fps)

    published_samples = row.get('fps_repetition_samples')
    if not isinstance(published_samples, list) or len(published_samples) != len(records):
        reasons.append('repetition_sample_vector_missing_or_wrong_length')
    else:
        parsed_samples = [_finite_num(value) for value in published_samples]
        if any(value is None or value <= 0 for value in parsed_samples):
            reasons.append('repetition_sample_vector_nonfinite_or_nonpositive')
        elif len(fps_values) == len(parsed_samples) and any(
            not math.isclose(float(observed), expected, rel_tol=1e-12, abs_tol=1e-12)
            for observed, expected in zip(parsed_samples, fps_values)
        ):
            reasons.append('repetition_sample_vector_not_bound_to_raw_records')

    reported_median = _finite_num(row.get('fps_median'))
    reported_low = _finite_num(row.get('fps_ci95_low'))
    reported_high = _finite_num(row.get('fps_ci95_high'))
    if reported_median is None or reported_low is None or reported_high is None:
        reasons.append('repetition_reported_median_or_ci_missing')
    elif not (0 < reported_low <= reported_median <= reported_high):
        reasons.append('repetition_reported_ci_invalid')

    if len(fps_values) == len(records) and fps_values:
        seed = '|'.join(_measurement_identity(row)) + '|claim-repeat-fps'
        median, ci_low, ci_high = _bootstrap_median_ci(fps_values, seed)
        row['fps_median_reported'] = reported_median
        row['fps_ci95_low_reported'] = reported_low
        row['fps_ci95_high_reported'] = reported_high
        row['fps_makespan'] = median
        row['fps_median'] = median
        row['fps_ci95_low'] = ci_low
        row['fps_ci95_high'] = ci_high
        row['repetition_statistics_source'] = 'recomputed_from_unique_raw_repeat_records'
        row['repetition_statistics_ci_method'] = 'deterministic_percentile_bootstrap_median_4000'
        if reported_median is not None and median is not None and not math.isclose(
            reported_median, median, rel_tol=1e-9, abs_tol=1e-12,
        ):
            reasons.append('repetition_reported_median_mismatch')
    else:
        row['repetition_statistics_source'] = 'unavailable_invalid_raw_repeat_records'

    # Preserve a stable, unique reason set without hiding repeated diagnostics.
    reasons = list(dict.fromkeys(reasons))
    row['repeat_claim_exclusion_reasons'] = reasons
    return not reasons, reasons


def _apply_comparison_claim_gates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fail closed for endpoint, precision-quality and repeat comparability."""
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in rows:
        is_full = (
            str(row.get('execution_mode') or '') == 'native_full_baseline'
            or str(row.get('backend') or '').startswith('native_full_')
        )
        if is_full:
            # A Full vendor baseline is paired only with the Full TensorRT
            # baseline executed for the same physical setup/comparison target.
            # Grouping every Full backend by model allowed one unrelated raw
            # Hailo endpoint to invalidate an otherwise valid DeepX/TRT pair.
            key = (
                str(row.get('model') or ''), 'full', 'native_full_baseline',
                str(row.get('setup_id') or ''),
                str(row.get('comparison_backend') or ''),
            )
        else:
            # Split rows intentionally retain the common model/case stratum so
            # endpoint equality remains a cross-backend property.
            key = (
                str(row.get('model') or ''), str(row.get('case') or ''),
                'native_split',
            )
        groups.setdefault(key, []).append(row)
    for key, group in groups.items():
        endpoints = sorted({
            endpoint for row in group
            for endpoint in [_comparison_output_endpoint(row)]
            if endpoint and _strict_bool(row.get('ok')) is True
        })
        participating_backends = {
            str(row.get('backend') or '')
            for row in group if _strict_bool(row.get('ok')) is True
        }
        endpoint_backends = {
            str(row.get('backend') or '')
            for row in group
            if _strict_bool(row.get('ok')) is True and _comparison_output_endpoint(row)
        }
        endpoint_match = bool(
            len(endpoints) == 1
            and len(participating_backends) >= 2
            and endpoint_backends == participating_backends
        )
        for row in group:
            endpoint = _comparison_output_endpoint(row)
            physical_endpoint = _explicit_output_endpoint(row)
            is_full = (
                str(row.get('execution_mode') or '') == 'native_full_baseline'
                or str(row.get('backend') or '').startswith('native_full_')
            )
            precision = _runtime_precision_identity(row)
            precision_explicit = bool(precision and precision not in {'unknown', 'unavailable', 'auto'})
            quality_verified = _quality_precision_verified(row)
            repeat_gate, repeat_reasons = _validate_repeat_claim_evidence(row)
            comparison_stratum_explicit = bool(
                not is_full
                or (
                    str(row.get('setup_id') or '').strip()
                    and str(row.get('comparison_backend') or '').strip()
                )
            )
            row_endpoint_match = bool(endpoint and endpoint_match and comparison_stratum_explicit)
            reasons: list[str] = []
            if _strict_bool(row.get('ok')) is not True:
                reasons.append('runtime_or_repetition_failed')
            if not endpoint:
                reasons.append('output_endpoint_missing')
                if not endpoint_match:
                    reasons.append('output_endpoint_not_common_across_backends')
            elif not row_endpoint_match:
                reasons.append('output_endpoint_not_common_across_backends')
            if not comparison_stratum_explicit:
                reasons.append('comparison_stratum_identity_missing')
            if not precision_explicit:
                reasons.append('runtime_precision_missing')
            if not quality_verified:
                reasons.append('precision_variant_quality_not_verified')
            elif row.get('task_quality_observation_valid') is not True:
                reasons.append('task_quality_observation_invalid')
            elif row.get('quality_accuracy_gate_pass') is not True:
                reasons.append('accuracy_gate_failed')
            if row.get('native_split_quality_required') is True:
                if row.get('native_split_quality_authority_valid') is not True:
                    reasons.append('native_split_quality_authority_invalid')
                if row.get('native_split_quality_provenance_conflict') is True:
                    reasons.append('native_split_quality_provenance_conflict')
                if str(row.get('native_split_quality_consumer_status') or '') != 'exact_quality_native_engine_command_and_boundary_match':
                    reasons.append('native_split_quality_consumer_attestation_missing_or_invalid')
                if row.get('native_split_final_portable_binding_valid') is not True:
                    reasons.append('native_split_quality_final_portable_binding_invalid')
                if row.get('native_split_semantic_binding_valid') is not True:
                    reasons.append('native_split_semantic_manifest_or_payload_binding_invalid')
            if (
                row.get('native_split_quality_authority_mode') == 'legacy'
                or row.get('native_split_quality_legacy_status')
                == 'historical_diagnostic_only'
            ):
                reasons.append('native_split_legacy_historical_diagnostic_only')
            if str(row.get('backend') or '') == 'native_full_deepx' and _strict_bool(row.get('outer_makespan_verified')) is not True:
                reasons.append('deepx_outer_makespan_not_verified')
            if not repeat_gate:
                reasons.append('independent_repetitions_below_three')
                reasons.extend(repeat_reasons)
            row['physical_output_endpoint_id'] = physical_endpoint
            row['comparison_output_endpoint_id'] = endpoint
            row['output_endpoint_id'] = physical_endpoint
            row['output_endpoint_candidates'] = endpoints
            row['output_endpoint_match'] = row_endpoint_match
            row['comparison_endpoint_match'] = row_endpoint_match
            row['output_endpoint_comparison_stratum'] = list(key)
            row['comparison_stratum_explicit'] = comparison_stratum_explicit
            row['runtime_precision_explicit'] = precision_explicit
            row['runtime_precision_identity'] = precision
            row['precision_quality_verified'] = quality_verified
            row['precision_quality_binding_verified'] = quality_verified
            row['repeat_claim_gate_pass'] = repeat_gate
            row['performance_claim_eligible'] = not reasons
            row['performance_claim_exclusion_reasons'] = reasons
    return rows



def _safe_component(value: Any) -> str:
    text = str(value or "unknown").strip() or "unknown"
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in text)[:120]


def _csv_cell(value: Any) -> Any:
    if isinstance(value, (Mapping, list, tuple)):
        return json.dumps(
            value,
            sort_keys=True,
            separators=(',', ':'),
            ensure_ascii=False,
        )
    return "" if value is None else value


def _load_detection_claim_exclusions(
    path_value: Any,
    expected_sha256: Any,
) -> tuple[dict[tuple[str, str, str], dict[str, Any]], str]:
    token = str(path_value or "").strip()
    if not token:
        return {}, "not_provided"
    path = Path(token).expanduser().resolve()
    wanted = str(expected_sha256 or "").strip().lower()
    if (
        not path.is_file()
        or len(wanted) != 64
        or _hash_file(path) != wanted
        or _verify_prospective_detection_exclusion_set is None
    ):
        return {}, "file_or_sha256_invalid"
    raw = _load_strict_json_object(path)
    try:
        verified = _verify_prospective_detection_exclusion_set(raw)
    except Exception as exc:
        return {}, f"contract_invalid:{type(exc).__name__}"
    entries: dict[tuple[str, str, str], dict[str, Any]] = {}
    for raw_entry in list(verified.get("entries") or []):
        entry = dict(raw_entry)
        key = (
            str(entry.get("setup_id") or ""),
            str(entry.get("backend") or ""),
            str(entry.get("model_id") or ""),
        )
        if not all(key) or key in entries:
            return {}, "identity_ambiguous"
        entries[key] = entry
    return entries, "verified"


def _apply_detection_claim_exclusions(
    rows: list[dict[str, Any]],
    exclusions: Mapping[
        tuple[str, str, str], Mapping[str, Any]
    ],
) -> list[dict[str, Any]]:
    aliases = {
        "hailo10_to_trt": "hailo10h_to_trt",
        "hailo10h_to_tensorrt": "hailo10h_to_trt",
    }
    for row in rows:
        backend = str(row.get("backend") or "").strip().lower()
        model = str(
            row.get("model") or row.get("model_id") or ""
        ).strip().lower()
        key = (
            str(row.get("setup_id") or "").strip(),
            aliases.get(backend, backend),
            model,
        )
        entry = exclusions.get(key)
        if not isinstance(entry, Mapping):
            continue
        reason = str(entry.get("reason") or "")
        reasons = [
            str(value)
            for value in list(
                row.get("performance_claim_exclusion_reasons") or []
            )
            if str(value)
        ]
        if reason and reason not in reasons:
            reasons.append(reason)
        row.update({
            "prospective_detection_claim_exclusion": True,
            "prospective_detection_claim_exclusion_reason": reason,
            "detection_diagnostic_sha256": str(
                entry.get("diagnostic_sha256") or ""
            ),
            "detection_claim_exclusion_entry_sha256": str(
                entry.get("entry_sha256") or ""
            ),
            "performance_claim_eligible": False,
            "performance_claim_exclusion_reasons": reasons,
        })
    return rows


def _strict_nonnegative_count(
    source: Mapping[str, Any], field: str,
) -> int | None:
    value = source.get(field)
    return value if type(value) is int and value >= 0 else None


def _expected_matrix_rows(
    matrix: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    if not isinstance(matrix, Mapping):
        return []
    declared = matrix.get("expected_rows")
    if isinstance(declared, list):
        return [dict(row) for row in declared if isinstance(row, Mapping)]
    # v2 matrices store a lossless expected partition: every expected row is
    # either present or missing.  ``failed`` and ``successful`` are subsets of
    # ``present`` and must not be concatenated here.
    return [
        dict(row)
        for key in ("present_expected_rows", "missing_expected_rows")
        for row in list(matrix.get(key) or [])
        if isinstance(row, Mapping)
    ]


def _scoped_expected_matrix(
    matrix: Mapping[str, Any] | None, *, model: str,
) -> dict[str, Any] | None:
    if not isinstance(matrix, Mapping):
        return None
    expected_rows = [
        row for row in _expected_matrix_rows(matrix)
        if str(row.get("model") or "unknown") == model
    ]
    present = [
        dict(row) for row in list(matrix.get("present_expected_rows") or [])
        if isinstance(row, Mapping)
        and str(row.get("model") or "unknown") == model
    ]
    successful = [
        dict(row) for row in list(matrix.get("successful_expected_rows") or [])
        if isinstance(row, Mapping)
        and str(row.get("model") or "unknown") == model
    ]
    failed = [
        dict(row) for row in list(matrix.get("failed_expected_rows") or [])
        if isinstance(row, Mapping)
        and str(row.get("model") or "unknown") == model
    ]
    missing = [
        dict(row) for row in list(matrix.get("missing_expected_rows") or [])
        if isinstance(row, Mapping)
        and str(row.get("model") or "unknown") == model
    ]
    from onnx_splitpoint_tool.native_job_identity import project_known_build_exclusions
    return project_known_build_exclusions({
        "expected_row_count": len(expected_rows),
        "present_expected_row_count": len(present),
        "successful_expected_row_count": len(successful),
        "failed_expected_row_count": len(failed),
        "missing_expected_row_count": len(missing),
        "present_expected_rows": present,
        "successful_expected_rows": successful,
        "failed_expected_rows": failed,
        "missing_expected_rows": missing,
    })


def _evidence_summary(
    rows: list[dict[str, Any]],
    expected_matrix: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    from onnx_splitpoint_tool.native_job_identity import known_build_exclusion, project_known_build_exclusions
    if isinstance(expected_matrix, Mapping):
        expected_matrix = project_known_build_exclusions(expected_matrix)
    excluded_rows = [r for r in rows if known_build_exclusion(r)]
    ok_rows = [r for r in rows if bool(r.get("ok"))]
    failed_rows = [r for r in rows if not bool(r.get("ok")) and not known_build_exclusion(r)]
    full_rows = [
        r for r in rows
        if str(r.get("execution_mode") or "") == "native_full_baseline"
        or str(r.get("backend") or "").startswith("native_full_")
    ]
    result = {
        "evidence_status": "empty",
        "row_count": len(rows),
        "ok_count": len(ok_rows),
        "failed_or_unsupported_count": len(failed_rows),
        "native_full_row_count": len(full_rows),
    }
    if not isinstance(expected_matrix, Mapping):
        status = (
            "complete" if rows and not failed_rows
            else "partial" if rows else "empty"
        )
        result["evidence_status"] = status
        return result

    expected = _strict_nonnegative_count(
        expected_matrix, "expected_row_count",
    )
    present = _strict_nonnegative_count(
        expected_matrix, "present_expected_row_count",
    )
    successful = _strict_nonnegative_count(
        expected_matrix, "successful_expected_row_count",
    )
    failed = _strict_nonnegative_count(
        expected_matrix, "failed_expected_row_count",
    )
    missing = _strict_nonnegative_count(
        expected_matrix, "missing_expected_row_count",
    )
    excluded = _strict_nonnegative_count(expected_matrix, "excluded_expected_row_count")
    counts = (expected, present, successful, failed, missing, excluded)
    counts_consistent = bool(
        all(value is not None for value in counts)
        and expected == present + missing
        and present == successful + failed + excluded
        and present == len(rows)
        and successful == len(ok_rows)
        and failed == len(failed_rows)
        and excluded == len(excluded_rows)
    )
    matrix_complete = bool(
        counts_consistent
        and expected is not None and expected > 0
        and present == expected
        and successful == expected
        and failed == 0
        and missing == 0
    )
    # Never allow a stale compatibility boolean to override contradictory
    # counts or the observed report rows.  It may only make an otherwise exact
    # matrix fail closed.
    for completeness_field in (
        "row_presence_complete", "execution_success_complete",
        "matrix_complete",
    ):
        if (
            completeness_field in expected_matrix
            and expected_matrix.get(completeness_field) is not True
        ):
            matrix_complete = False
    technical_complete = bool(counts_consistent and expected_matrix.get("technical_execution_complete"))
    result.update({
        "evidence_status": (
            "complete_with_exclusions" if technical_complete and excluded
            else "complete" if matrix_complete
            else "partial" if (expected or rows) else "empty"
        ),
        "matrix_bound": True,
        "matrix_counts_consistent": counts_consistent,
        "matrix_complete": matrix_complete,
        "expected_row_count": expected,
        "present_expected_row_count": present,
        "successful_expected_row_count": successful,
        "failed_expected_row_count": failed,
        "missing_expected_row_count": missing,
        "excluded_expected_row_count": excluded,
        "technical_execution_complete": technical_complete,
        "failed_or_unsupported_count": (
            (failed or 0) + (missing or 0)
            if failed is not None and missing is not None
            else len(failed_rows)
        ),
    })
    return result


def _write_model_scoped_reports(
    outdir: Path, rows: list[dict[str, Any]], fields: list[str],
    expected_matrix: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    outputs: dict[str, str] = {}
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row.get("model") or "unknown"), []).append(row)
    root = outdir / "native_models"
    for model, model_rows in sorted(groups.items()):
        model_dir = root / _safe_component(model)
        model_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "schema": "onnx-splitpoint/native-producer-model-summary",
            "schema_version": 2,
            "model": model,
            "rows": model_rows,
            **_evidence_summary(
                model_rows,
                _scoped_expected_matrix(expected_matrix, model=model),
            ),
        }
        json_path = model_dir / "native_producer_summary.json"
        csv_path = model_dir / "native_producer_summary.csv"
        md_path = model_dir / "native_producer_summary.md"
        json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for row in model_rows:
                w.writerow({_key: _csv_cell(row.get(_key)) for _key in fields})
        lines = [
            f"# Native producer summary — {model}",
            "",
            f"Evidence status: **{summary['evidence_status']}**",
            (
                "Matrix coverage: "
                f"**{summary.get('present_expected_row_count')} / "
                f"{summary.get('expected_row_count')}** "
                "present; "
                f"**{summary.get('successful_expected_row_count')}** "
                "successful; "
                f"**{summary.get('failed_expected_row_count')}** failed; "
                f"**{summary.get('excluded_expected_row_count', 0)}** known build exclusions; "
                f"**{summary.get('missing_expected_row_count')}** missing"
                if summary.get("matrix_bound") is True else ""
            ),
            "",
            "| backend | case | precision | E2E scope | completed endpoint | host postprocess | status | ok | Completed Task FPS / CI95 | P2 FPS / CI95 | note |",
            "|---|---|---|---|---|---|---|---:|---:|---:|---|",
        ]
        for row in model_rows:
            fps = "unavailable" if row.get("fps_makespan") is None else f"{float(row.get('fps_makespan')):.3f} [{row.get('fps_ci95_low')}, {row.get('fps_ci95_high')}]"
            p2 = f"{row.get('p2_output_fps')} [{row.get('p2_output_fps_ci95_low')}, {row.get('p2_output_fps_ci95_high')}]"
            lines.append(
                f"| {row.get('backend','')} | {row.get('case','')} | "
                f"{row.get('precision','')} | {row.get('e2e_scope','')} | "
                f"{row.get('completed_task_stage','')}:"
                f"{row.get('completed_task_comparison_output_endpoint_id') or row.get('completed_task_output_endpoint_id','')} | "
                f"{row.get('postprocess_location','')}:"
                f"{row.get('host_postprocessing_available', row.get('host_postprocess_frozen',''))} | "
                f"{row.get('status','')}{' / not_started' if row.get('repetition_status') == 'not_started' else ''} | {row.get('ok')} | {fps} | {p2} | "
                f"{row.get('note','')} |"
            )
        md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        outputs[model] = str(json_path)
    return outputs

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', action='append', required=True, help='Staged native_fifo_evalsets root. Can be passed multiple times.')
    ap.add_argument('--recursive', action='store_true', help='Recursively collect nested backend roots that contain analysis_tables.')
    ap.add_argument('--out-dir', default='', help='Defaults to first <root>/analysis_tables')
    ap.add_argument(
        '--expected-matrix', default='',
        help=(
            'Optional native_expected_matrix.json. When supplied, JSON and '
            'Markdown evidence status are bound to its expected denominator '
            'instead of only the rows that happened to be collected.'
        ),
    )
    ap.add_argument(
        '--quality-summary', default='',
        help=(
            'Optional native_producer_validation_summary.json. Precision variants '
            'are claim-eligible only after an exact identity match to an explicitly '
            'passing validation row. Without this file the gate fails closed.'
        ),
    )
    ap.add_argument(
        '--remote-execution-context-json', action='append', default=[],
        help=(
            'Repeatable schema-bound JSON allowlist record with setup_id, '
            'the exact remote run root and remote tool directory. DeepX Full '
            'claim verification fails closed without one exact setup/root match.'
        ),
    )
    ap.add_argument(
        '--detection-claim-exclusions',
        default=(
            str(_DEFAULT_DETECTION_EXCLUSIONS)
            if _DEFAULT_DETECTION_EXCLUSIONS.is_file() else ''
        ),
    )
    ap.add_argument(
        '--detection-claim-exclusions-sha256',
        default=(
            _DEFAULT_DETECTION_EXCLUSIONS_SHA256
            if _DEFAULT_DETECTION_EXCLUSIONS.is_file() else ''
        ),
    )
    ns = ap.parse_args()
    expected_matrix: dict[str, Any] | None = None
    if ns.expected_matrix:
        expected_matrix_path = Path(ns.expected_matrix).expanduser().resolve()
        expected_matrix = _load_strict_json_object(expected_matrix_path)
        if not isinstance(expected_matrix, dict):
            ap.error(
                'invalid --expected-matrix: expected a readable JSON object'
            )
    try:
        remote_execution_contexts = (
            _parse_remote_execution_context_allowlist(
                ns.remote_execution_context_json
            )
        )
    except ValueError as exc:
        ap.error(str(exc))
    detection_exclusions, detection_exclusion_status = (
        _load_detection_claim_exclusions(
            ns.detection_claim_exclusions,
            ns.detection_claim_exclusions_sha256,
        )
    )
    if (
        ns.detection_claim_exclusions
        and detection_exclusion_status != 'verified'
    ):
        ap.error(
            'invalid --detection-claim-exclusions: '
            + detection_exclusion_status
        )
    base = Path(ns.root[0]).expanduser().resolve()
    outdir = Path(ns.out_dir).expanduser().resolve() if ns.out_dir else base / 'analysis_tables'
    outdir.mkdir(parents=True, exist_ok=True)
    split_quality_authority = _native_split_authority(outdir)
    all_roots: list[Path] = []
    for r in ns.root:
        all_roots.extend(_find_eval_roots(Path(r), ns.recursive))
    all_roots = list(dict.fromkeys(all_roots))
    rows: list[dict[str, Any]] = _rows_from_native_fifo_roots(all_roots)
    for root in all_roots:
        rows += _rows_from_hailo10(root)
        rows += _rows_from_deepx(root)
        rows += _rows_from_native_full(root)
    if _apply_host_postprocess_aliases is not None:
        for row in rows:
            _apply_host_postprocess_aliases(row)
    _apply_native_split_authority_to_rows(rows, split_quality_authority)
    quality_source = str(Path(ns.quality_summary).expanduser().resolve()) if ns.quality_summary else ''
    quality_payload: dict[str, Any] | None = None
    quality_status = 'not_provided'
    if ns.quality_summary:
        quality_path = Path(ns.quality_summary).expanduser()
        if not quality_path.is_file():
            quality_status = 'missing'
        else:
            loaded = _load_strict_json_object(quality_path)
            if isinstance(loaded, dict) and isinstance(loaded.get('rows'), list):
                quality_payload = loaded
                quality_status = 'loaded'
            else:
                quality_status = 'invalid'
    from onnx_splitpoint_tool.native_job_identity import complete_historical_identity
    planned_contexts = []
    if isinstance(expected_matrix, Mapping):
        planned_contexts = list(expected_matrix.get('expected_native_rows') or expected_matrix.get('expected_rows') or [])
    if not planned_contexts:
        stage = _load_strict_json_object(outdir / 'native_producer_stage.json') or {}
        planned_contexts = list(stage.get('expected_native_rows') or [])
    if planned_contexts:
        rows = [complete_historical_identity(row, planned_contexts) for row in rows]
    aggregated_rows = _aggregate_repetitions(rows)
    _apply_native_split_authority_to_rows(
        aggregated_rows, split_quality_authority,
    )
    rows, quality_evidence = _attach_quality_evidence(
        aggregated_rows,
        quality_payload,
        quality_summary_source=quality_source,
        quality_summary_status=quality_status,
        remote_execution_contexts=remote_execution_contexts,
    )
    if _apply_host_postprocess_aliases is not None:
        for row in rows:
            _apply_host_postprocess_aliases(row)
    rows = _apply_comparison_claim_gates(rows)
    rows = _apply_detection_claim_exclusions(
        rows, detection_exclusions,
    )
    from onnx_splitpoint_tool.native_rate_endpoints import report_rate_fields
    for row in rows:
        row.update(report_rate_fields(row))
        if row.get('completed_task_fps') is None:
            row['performance_claim_eligible'] = False
            row.setdefault('performance_claim_exclusion_reasons', []).append(row['completed_task_fps_unavailable_reason'])

    rows = sorted(rows, key=lambda r:(str(r.get('backend','')), str(r.get('model','')), str(r.get('case',''))))
    fields = ['backend','producer_impl','model','case','precision','setup_id','comparison_backend','execution_mode','status','ok','runtime_success','buildable','runtime_executable','fps_makespan','fps_median','fps_ci95_low','fps_ci95_high','legacy_reciprocal_latency_fps','latency_mean_ms','latency_median_ms','latency_p50_ms','latency_p95_ms','latency_ci95_low_ms','latency_ci95_high_ms','latency_semantics','completion_interval_mean_ms','outer_makespan_verified','repetition_count_requested','repetition_count_attempted','repetition_count_valid','repetition_status','repetition_aggregation','repetition_claim_identity_status','repetition_claim_identity_drift','repetition_claim_identity_incomplete','output_endpoint_id','physical_output_endpoint_id','comparison_output_endpoint_id','output_endpoint_match','comparison_endpoint_match','output_endpoint_comparison_stratum','comparison_stratum_explicit','runtime_precision_explicit','precision_quality_verified','precision_quality_binding_verified','task_quality_observation_valid','quality_claim_result_verified','quality_evidence_verified','quality_match_status','quality_match_count','quality_summary_status','quality_contract_consistent','quality_central_evidence_verified','quality_task_valid','quality_accuracy_gate_pass','quality_eligible_for_ranking','quality_gate_status','quality_accuracy_gate_reason','quality_ranking_exclusion_reason','structural_contract_pass','structural_contract_status','structural_contract_reason','numerical_similarity_pass','numerical_similarity_status','numerical_similarity_reason','numerical_similarity_scope','numerical_similarity_metric','numerical_similarity_value','numerical_similarity_threshold','numerical_similarity_mean_iou','numerical_similarity_mean_iou_threshold','numerical_similarity_policy_id','task_quality_pass','task_quality_status','task_quality_reason','quality_first_producer_identity_sha256','quality_first_producer_identity_match','quality_request_binding_status','quality_request_binding_sha256','quality_request_binding_set_sha256','central_quality_result_sha256','source_request_sha256','model_sha256','hailo_hef_build_receipt_status','hailo_hef_build_receipt_path','hailo_hef_build_receipt_file_sha256','hailo_hef_build_receipt_sha256','hailo_hef_source_onnx_path','hailo_hef_compiler_onnx_sha256','hailo_hef_preprocessing_contract','hailo_hef_preprocessing_contract_sha256','validation_dataset_sha256','validation_dataset_image_ids_sha256','validation_dataset_ground_truth_sha256','accuracy_gate_policy_sha256','task_quality_policy_sha256','runtime_quality_gate_policy_sha256','quality_contract_sha256','preprocessing_contract_sha256','decoder_contract_sha256','nms_contract_sha256','quality_record_endpoint_contract_sha256','repeat_claim_gate_pass','performance_claim_eligible','performance_claim_exclusion_reasons','paper_fps','handoff_ms','p1_ms','p2_run_ms','p1_thread_ms','p2_thread_ms','frames','warmup','inflight','producer_ready','consumer_ready','input_image','input_image_source','input_image_sha256','input_manifest','runtime_input_dtype','runtime_input_shape','runtime_input_layout','runtime_preprocess_mode','runtime_normalization','runtime_color_space','runtime_preprocessing_identity','runtime_preprocessing_sha256','runtime_numeric_input_identity','runtime_numeric_input_sha256','engine_precision','native_command_contract_sha256','output_dump_manifest','native_output_manifest','output_contract_manifest_status','semantic_dump_status','semantic_dump_failure_reason','task','stage','output_format','contract_family','contract_source','endpoint_contract_complete','endpoint_contract_hash','output_endpoint_attestation','accelerator_output_stage','accelerator_output_contract_family','accelerator_endpoint_contract_hash','accelerator_output_endpoint_attestation','source_e2e_scope','e2e_scope','e2e_claim_eligible','e2e_contract_reason','comparison_endpoint_stratum','measurement_concurrency','requires_host_decode_nms','postprocess_included','postprocess_location','host_postprocess_frozen','host_postprocessing_available','host_tail_available','host_postprocess_required','host_tail_required','host_postprocessing_evidence_status','host_postprocessing_evidence_source','host_postprocessing_legacy_alias_conflict','decoder_contract_pass','nms_ok','decoder_id','postprocess_completed_frames','postprocess_completion_verified','completed_task_result_artifact_verification_status','direct_source_endpoint_binding_verified','direct_source_endpoint_binding_status','frozen_host_postprocess_contract','frozen_host_postprocess_contract_sha256','frozen_host_postprocess_result','normalization_frozen','frozen_decoded_nms_normalization_contract','frozen_decoded_nms_normalization_contract_sha256','frozen_decoded_nms_normalization_result','direct_bn6_completion_projection_status','completed_task_stage','completed_task_contract_family','completed_task_endpoint_contract','completed_task_endpoint_contract_hash','completed_task_output_endpoint_id','completed_task_comparison_endpoint_contract','completed_task_comparison_endpoint_contract_hash','completed_task_comparison_output_endpoint_id','completed_task_completion_mode','completed_task_endpoint_attested','completed_task_endpoint_attestation','completed_task_endpoint_attestation_status','completed_task_result_artifact','completed_task_result_artifact_path','completed_task_result_artifact_saved','completed_task_result_artifact_sha256','completed_task_result_artifact_file_sha256','completed_task_endpoint_projection_status','performance_benchmark_source','performance_input_contract_mode','comparison_precision','legacy_comparison_precision','execution_precision','full_runtime_precision','runtime_precision_source','full_command_contract_sha256','failure_reason','status_detail','error','timed_out','returncode','fps_source','result_source','stdout_tail','stderr_tail','note','report','source_root']
    from onnx_splitpoint_tool.native_rate_endpoints import rate_endpoint_fields
    fields.extend(key for key in rate_endpoint_fields({}) if key not in fields)

    fields.extend((
        'prerequisite_status', 'failure_stage', 'primary_failure_reason',
        'upstream_evidence_path', 'aggregation_applied', 'disposition', 'build_exclusion',
        'quality_nearest_candidate_count',
        'quality_nearest_candidates',
    ))
    for claim_field in reversed((
        'claim_ok', 'claim_ok_source', 'claim_structural_gate_pass',
        'claim_structural_gate_reason', 'claim_ok_structural_clamped',
        'semantic_ok', 'contract_consistent',
        'quality_physical_evidence_conflict',
        'quality_physical_evidence_conflict_fields',
        'prospective_detection_claim_exclusion',
        'prospective_detection_claim_exclusion_reason',
        'detection_diagnostic_sha256',
        'detection_claim_exclusion_entry_sha256',
    )):
        if claim_field not in fields:
            fields.insert(fields.index('fps_makespan'), claim_field)
    model_reports = _write_model_scoped_reports(
        outdir, rows, fields, expected_matrix,
    )
    data = {
        'schema': 'onnx-splitpoint/native-producer-combined-summary',
        'schema_version': 7,
        'roots': [str(x) for x in all_roots],
        'quality_evidence': quality_evidence,
        'native_split_quality_authority': split_quality_authority,
        'detection_claim_exclusions_status': (
            detection_exclusion_status
        ),
        'rows': rows,
        'model_reports': model_reports,
        **_evidence_summary(rows, expected_matrix),
    }
    jsonp = outdir / 'native_producer_combined_summary.json'
    csvp = outdir / 'native_producer_combined_summary.csv'
    mdp = outdir / 'native_producer_combined_summary.md'
    jsonp.write_text(json.dumps(data, indent=2), encoding='utf-8')
    with csvp.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in rows:
            w.writerow({_key: _csv_cell(r.get(_key)) for _key in fields})
    lines = [
        '# Native producer combined summary', '',
        f"Evidence status: **{data['evidence_status']}**",
    ]
    if data.get('matrix_bound') is True:
        lines.extend([
            (
                'Matrix coverage: '
                f"**{data.get('present_expected_row_count')} / "
                f"{data.get('expected_row_count')}** present; "
                f"**{data.get('successful_expected_row_count')}** "
                'successful; '
                f"**{data.get('failed_expected_row_count')}** failed; "
                f"**{data.get('excluded_expected_row_count', 0)}** known build exclusions; "
                f"**{data.get('missing_expected_row_count')}** missing"
            ),
        ])
    lines.extend([
        '', 'Roots:', *[f'- `{x}`' for x in all_roots], '',
        '| backend | impl | model | case | precision | E2E scope | completed endpoint | host postprocess | structure | numerical | task quality | status | ok | Completed Task FPS / CI95 | P2 FPS / CI95 | handoff ms | note |',
        '|---|---|---|---|---|---|---|---|---:|---:|---:|---|---:|---:|---:|---:|---|',
    ])
    for r in rows:
        fps = 'unavailable' if r.get('fps_makespan') is None else f"{float(r.get('fps_makespan')):.3f} [{r.get('fps_ci95_low')}, {r.get('fps_ci95_high')}]"
        p2 = f"{r.get('p2_output_fps')} [{r.get('p2_output_fps_ci95_low')}, {r.get('p2_output_fps_ci95_high')}]"
        h = '' if r.get('handoff_ms') is None else f"{float(r.get('handoff_ms')):.3f}"
        lines.append(
            f"| {r.get('backend','')} | {r.get('producer_impl','')} | "
            f"{r.get('model','')} | {r.get('case','')} | "
            f"{r.get('precision','')} | {r.get('e2e_scope','')} | "
            f"{r.get('completed_task_stage','')}:"
            f"{r.get('completed_task_comparison_output_endpoint_id') or r.get('completed_task_output_endpoint_id','')} | "
            f"{r.get('postprocess_location','')}:"
            f"{r.get('host_postprocessing_available', r.get('host_postprocess_frozen',''))} | "
            f"{r.get('structural_contract_pass','')} | "
            f"{r.get('numerical_similarity_pass','')} | "
            f"{r.get('task_quality_pass','')} | {r.get('status','')}{' / not_started' if r.get('repetition_status') == 'not_started' else ''} | "
            f"{r.get('ok')} | {fps} | {p2} | {h} | {r.get('note','')} |"
        )
    mdp.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({'ok': True, 'rows': len(rows), 'ok_count': data['ok_count'], 'evidence_status': data['evidence_status'], 'quality_evidence': quality_evidence, 'model_reports': model_reports, 'json': str(jsonp), 'csv': str(csvp), 'md': str(mdp)}, indent=2))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
