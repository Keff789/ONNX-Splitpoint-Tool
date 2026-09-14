#!/usr/bin/env python3
"""Run u.RECS energy measurement for native producer/full rows from a summary.

This is intentionally thin around existing native_producer_energy_plan.py and
energy_measurement_cli.py.  It creates local command files and executes selected
rows sequentially on the energy host.
"""
from __future__ import annotations
import argparse, hashlib, json, math, re, subprocess, sys, time, shlex, os, posixpath, uuid, signal
from pathlib import Path
from typing import Any, Mapping


_AGGREGATE_MTIME_TOLERANCE_NS = 1_000_000_000
_ACTIVE_MANAGED_STAGE_CONTEXT: dict[str, Any] = {}

_NONCLAIMABLE_SCREENING_FIELDS = {
    'screening_only': True,
    'screening_energy': True,
    'diagnostic_only': True,
    'energy_evidence_tier': 'screening',
    'energy_tier': 'screening',
    'claim_ok': False,
    'semantic_claim_ok': False,
    'claim_eligible': False,
    'eligible_for_energy_results_import': False,
    'eligible_for_scientific_claim': False,
    'energy_claim_eligible': False,
    'strict_requested': False,
    'strict_failure': False,
}

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
# Script entry points are often launched from the GUI with a working directory
# that is not the project root.  Make the in-tree package importable without
# relying on an editable installation.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from onnx_splitpoint_tool.resume_preparation import (  # noqa: E402
    prepare_resume_measurement_cohort as _prepare_resume_measurement_cohort,
)
from onnx_splitpoint_tool.native_energy_quality_admission import (  # noqa: E402
    verify_sealed_energy_quality_admission as _verify_energy_quality_admission,
    energy_quality_reason_projection,
)
from onnx_splitpoint_tool.workflow.checkpoints import (  # noqa: E402
    AtomicRowJournal,
    atomic_write_json as _checkpoint_atomic_write_json,
    atomic_write_text as _checkpoint_atomic_write_text,
    canonical_json_sha256 as _checkpoint_json_sha256,
    load_stage_checkpoint as _load_stage_checkpoint,
    write_stage_checkpoint as _write_stage_checkpoint,
)


class _ManagedEnergyCancelled(BaseException):
    def __init__(self, signum: int = 0, reason: str = "cancelled") -> None:
        super().__init__(reason)
        self.signum = int(signum or 0)
        self.reason = str(reason or "cancelled")


def _install_managed_cancel_handlers():
    previous: dict[int, Any] = {}
    delivered = {"signum": 0}

    def _handler(signum, _frame):
        if delivered["signum"]:
            return
        delivered["signum"] = int(signum)
        raise _ManagedEnergyCancelled(
            int(signum), f"signal:{signal.Signals(signum).name}"
        )

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.getsignal(signum)
        signal.signal(signum, _handler)
    return previous


def _restore_managed_cancel_handlers(previous: Mapping[int, Any]) -> None:
    for signum, handler in previous.items():
        signal.signal(int(signum), handler)


def _script_path(name: str) -> Path:
    source = ROOT / 'scripts' / name
    return source if source.is_file() else SCRIPT_DIR / name


def _clamp_screening_result(value: Any) -> Any:
    """Mark every result-bearing layer as Screening-only and non-claimable."""
    if not isinstance(value, dict):
        return value
    clamped = dict(value)
    clamped.update(_NONCLAIMABLE_SCREENING_FIELDS)
    for key in ('row', 'run', 'energy_aggregate'):
        child = clamped.get(key)
        if isinstance(child, dict):
            clamped[key] = _clamp_screening_result(child)
    return clamped

def _strict_json_text(text: str, *, label: str) -> Any:
    def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(
                    f'duplicate JSON object key in {label}: {key!r}'
                )
            value[key] = item
        return value

    return json.loads(text, object_pairs_hook=_object)


def _load_json(p: Path) -> Any:
    try:
        return _strict_json_text(p.read_text(encoding='utf-8'), label=str(p))
    except Exception:
        return None


def _measurement_start_observation(
    output_dir: str | Path,
    *,
    aggregate_verified: bool,
) -> dict[str, Any]:
    """Project physical-start counters from the measurement's own evidence."""
    root = Path(output_dir)
    summary = _load_json(root / 'energy_summary.json')
    aggregate_runs = (
        summary.get('runs') if isinstance(summary, dict) else None
    )
    if not isinstance(aggregate_runs, list):
        aggregate_runs = []
    runs_by_index: dict[int, dict[str, Any]] = {}
    unindexed_runs: list[dict[str, Any]] = []
    for position, row in enumerate(aggregate_runs):
        if not isinstance(row, dict):
            continue
        run_index = row.get('run_index')
        if isinstance(run_index, int) and not isinstance(run_index, bool):
            runs_by_index[run_index] = dict(row)
        else:
            unindexed_runs.append(dict(row))
    run_summary_file_count = 0
    workload_timing_start_indexes: set[int] = set()
    for run_dir in sorted(root.glob('run_*')):
        if run_dir.is_symlink() or not run_dir.is_dir():
            continue
        match = re.fullmatch(r'run_([0-9]+)', run_dir.name)
        if match is None:
            continue
        run_index = int(match.group(1))
        run_summary = _load_json(run_dir / 'energy_summary.json')
        if isinstance(run_summary, dict):
            run_summary_file_count += 1
            merged = dict(runs_by_index.get(run_index) or {})
            # The repeat-local summary is the last writer before an abrupt
            # wrapper failure and therefore carries authoritative physical
            # start evidence that may never reach the aggregate.
            merged.update(run_summary)
            merged['run_index'] = run_index
            runs_by_index[run_index] = merged
        timing_path = run_dir / 'workload_timing.txt'
        if timing_path.is_symlink() or not timing_path.is_file():
            continue
        try:
            timing_values = dict(
                line.partition('=')[::2]
                for line in timing_path.read_text(
                    encoding='utf-8', errors='replace',
                ).splitlines()
                if '=' in line
            )
            if int(str(timing_values.get('start_ns') or '0').strip()) > 0:
                workload_timing_start_indexes.add(run_index)
        except (OSError, TypeError, ValueError):
            pass
    runs = [
        runs_by_index[index] for index in sorted(runs_by_index)
    ] + unindexed_runs
    collector_count = sum(
        1 for row in runs
        if isinstance(row, dict) and row.get('collector_started') is True
    )
    workload_started_indexes: set[int] = set(
        workload_timing_start_indexes
    )
    unindexed_workload_count = 0
    for row in runs:
        if not isinstance(row, dict):
            continue
        timing = row.get('workload_timing')
        timing_started = bool(
            isinstance(timing, dict)
            and isinstance(timing.get('start_ns'), int)
            and not isinstance(timing.get('start_ns'), bool)
            and timing.get('start_ns') > 0
        )
        workload_started = bool(
            row.get('workload_started') is True
            or timing_started
            or row.get('workload_command_rc') is not None
        )
        if not workload_started:
            continue
        run_index = row.get('run_index')
        if isinstance(run_index, int) and not isinstance(run_index, bool):
            workload_started_indexes.add(run_index)
        else:
            unindexed_workload_count += 1
    workload_count = (
        len(workload_started_indexes) + unindexed_workload_count
    )
    physical_started = bool(
        collector_count or workload_count or aggregate_verified
    )
    return {
        'measurement_started': physical_started,
        'collector_started_repeat_count': collector_count,
        'workload_started_repeat_count': workload_count,
        'energy_summary_present': isinstance(summary, dict),
        'aggregate_run_count': len(aggregate_runs),
        'repeat_summary_file_count': run_summary_file_count,
        'workload_timing_start_count': len(
            workload_timing_start_indexes
        ),
        'aggregate_verified_fallback': bool(
            aggregate_verified and not runs
        ),
    }


def _zero_measurement_start_observation() -> dict[str, Any]:
    """Return the canonical evidence shape for a row that never started."""

    return {
        'measurement_started': False,
        'collector_started_repeat_count': 0,
        'workload_started_repeat_count': 0,
        'energy_summary_present': False,
        'aggregate_run_count': 0,
        'repeat_summary_file_count': 0,
        'workload_timing_start_count': 0,
        'aggregate_verified_fallback': False,
    }


def _energy_global_infrastructure_failure(
    result: Mapping[str, Any] | None,
    output_dir: str | Path | None = None,
    *,
    error: Any = "",
) -> dict[str, str]:
    """Return a narrow, structured stop reason shared by remaining rows.

    Ordinary workload/vendor/Quality failures intentionally do not match.  The
    allow-list covers only SSH authentication/connectivity, an unavailable
    platform lock, and collector/u.RECS initialization or transport failures.
    """

    result_payload = dict(result or {})
    if (
        result_payload.get('rc') == 0
        and result_payload.get('energy_aggregate_verified') is True
        and isinstance(result_payload.get('energy_aggregate'), Mapping)
        and result_payload['energy_aggregate'].get('ok') is True
    ):
        return {}
    payloads: list[Any] = [result_payload, str(error or "")]
    root = Path(str(output_dir or "")) if str(output_dir or "") else None
    if root is not None and root.is_dir():
        for path in (
            root / 'energy_aggregate.json',
            root / 'energy_summary.json',
        ):
            value = _load_json(path)
            if isinstance(value, Mapping):
                payloads.append(dict(value))
        for pattern in (
            'run_*/energy_summary.json',
            'run_*/preflight/preflight_evidence.json',
        ):
            for path in sorted(root.glob(pattern)):
                value = _load_json(path)
                if isinstance(value, Mapping):
                    payloads.append(dict(value))
        for pattern in (
            'run_*/collector_stdout.log',
            'run_*/collector_stderr.log',
            'run_*/workload_stdout.log',
            'run_*/workload_stderr.log',
            'run_*/preflight/preflight_stdout.log',
            'run_*/preflight/preflight_stderr.log',
        ):
            for path in sorted(root.glob(pattern)):
                try:
                    payloads.append(path.read_text(
                        encoding='utf-8', errors='replace',
                    )[-65536:])
                except OSError:
                    pass

    structured: list[tuple[str, str, str]] = []
    diagnostics: list[str] = []

    def _visit(value: Any, depth: int = 0) -> None:
        if depth > 8:
            return
        if isinstance(value, Mapping):
            code = str(value.get('global_infrastructure_failure') or '').strip()
            if code:
                structured.append((
                    str(value.get('global_infrastructure_category') or '').strip(),
                    code,
                    str(value.get('global_infrastructure_detail') or '').strip(),
                ))
            for key, item in value.items():
                if key in {
                    'cmd', 'collector_cmd', 'measure_command', 'argv',
                    'command', 'native_command_contract',
                }:
                    continue
                _visit(item, depth + 1)
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                _visit(item, depth + 1)
            return
        text = ' '.join(str(value or '').split())
        if text:
            diagnostics.append(text[-2000:])

    for value in payloads:
        _visit(value)
    if structured:
        category, code, detail = structured[0]
        return {
            'category': category or 'native_energy_infrastructure',
            'code': code,
            'detail': detail or code,
        }

    def _match(
        tokens: tuple[str, ...],
        *,
        anchors: tuple[str, ...] = (),
    ) -> str:
        for text in diagnostics:
            lowered = text.lower()
            if (
                any(token in lowered for token in tokens)
                and (
                    not anchors
                    or any(anchor in lowered for anchor in anchors)
                )
            ):
                return text[-800:]
        return ''

    detail = _match(
        (
            'permission denied (publickey', 'authentication failed',
            'host key verification failed', 'no supported authentication',
        ),
        anchors=(
            'ssh:', 'publickey', 'known_hosts', 'host key verification',
            'port 22', 'no supported authentication',
        ),
    )
    if detail:
        return {
            'category': 'global_remote_infrastructure',
            'code': 'remote_authentication_failed',
            'detail': detail,
        }
    detail = _match(
        (
            'connection refused', 'connection timed out', 'no route to host',
            'network is unreachable', 'could not resolve hostname',
            'temporary failure in name resolution',
        ),
        anchors=(
            'ssh:', 'connect to host', 'port 22',
            'could not resolve hostname', 'known_hosts',
        ),
    )
    if detail:
        return {
            'category': 'global_remote_infrastructure',
            'code': 'remote_connection_failed',
            'detail': detail,
        }
    detail = _match((
        'platform_lock unavailable', 'platform lock unavailable',
        'platform_lock busy', 'platform lock busy',
        'failed to acquire platform lock',
        'platform lock acquisition failed', 'campaign lock unavailable',
    ))
    if detail:
        return {
            'category': 'platform_lock',
            'code': 'platform_lock_unavailable',
            'detail': detail,
        }
    detail = _match((
        'urecs-data-collector not found',
        'collector_process_start_failed',
        'collector executable not found',
    ))
    if detail:
        return {
            'category': 'collector_initialization',
            'code': 'collector_initialization_failed',
            'detail': detail,
        }
    return {}


def _canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
    ).encode('utf-8')).hexdigest()


def _strict_sha256(value: Any) -> str:
    token = str(value or '').strip().lower()
    if token.startswith('sha256:'):
        token = token[7:]
    return token if re.fullmatch(r'[0-9a-f]{64}', token) else ''


def _verify_split_energy_quality_binding(
    row: dict[str, Any],
    *,
    admission_scope: str = "",
) -> tuple[str, str]:
    """Verify the plan-local Quality-first seal before a measurement."""
    runtime_observation = (
        admission_scope == "native_runtime_observation"
    )
    backend = str(row.get("backend") or "").strip().lower()
    if (
        runtime_observation
        and not backend.startswith("native_full_")
        and row.get("native_split_energy_binding_valid") is False
    ):
        raw = row.get("native_split_energy_quality_binding")
        if not isinstance(raw, dict):
            raise ValueError(
                "native_split_runtime_observation_binding_missing"
            )
        evidence = dict(raw)
        declared = _strict_sha256(
            evidence.pop("evidence_sha256", "")
        )
        if (
            not declared
            or _canonical_json_sha256(evidence) != declared
            or _strict_sha256(
                row.get(
                    "native_split_energy_quality_binding_sha256"
                )
            ) != declared
        ):
            raise ValueError(
                "native_split_runtime_observation_binding_sha256_mismatch"
            )
        if (
            evidence.get("schema")
            != (
                "onnx-splitpoint/"
                "native-split-energy-runtime-observation"
            )
            or evidence.get("schema_version") != 1
            or evidence.get(
                "native_split_quality_required"
            ) is not True
            or evidence.get(
                "native_split_energy_binding_valid"
            ) is not False
            or not str(
                evidence.get(
                    "native_split_energy_binding_status"
                ) or ""
            ).strip()
            or row.get(
                "native_split_quality_required"
            ) is not True
            or row.get(
                "native_split_energy_binding_valid"
            ) is not False
            or str(
                row.get(
                    "native_split_energy_binding_status"
                ) or ""
            ).strip()
            != str(
                evidence.get(
                    "native_split_energy_binding_status"
                ) or ""
            ).strip()
        ):
            raise ValueError(
                "native_split_runtime_observation_binding_invalid"
            )
        command_sha = _strict_sha256(
            evidence.get("native_command_contract_sha256")
        )
        if (
            not command_sha
            or command_sha
            != _strict_sha256(
                row.get("successful_command_contract_sha256")
            )
        ):
            raise ValueError(
                "native_split_runtime_observation_command_contract_drift"
            )
        return (
            command_sha,
            "sealed_runtime_observation_binding_verified",
        )
    if row.get('native_split_quality_required') is not True:
        return '', 'not_required'
    raw = row.get('native_split_energy_quality_binding')
    if not isinstance(raw, dict):
        raise ValueError('native_split_energy_quality_binding_missing')
    evidence = dict(raw)
    declared = _strict_sha256(evidence.pop('evidence_sha256', ''))
    if not declared or _canonical_json_sha256(evidence) != declared:
        raise ValueError('native_split_energy_quality_binding_sha256_mismatch')
    evidence['evidence_sha256'] = declared
    if (
        evidence.get('schema') != 'onnx-splitpoint/native-split-energy-quality-binding'
        or evidence.get('schema_version') != 1
        or evidence.get('native_split_quality_required') is not True
        or evidence.get('native_split_energy_binding_valid') is not True
    ):
        raise ValueError('native_split_energy_quality_binding_schema_or_status_invalid')
    duplicate_fields = (
        'native_split_quality_binding_sha256',
        'native_split_quality_preselection_sha256',
        'native_split_quality_source_request_sha256',
        'native_split_quality_central_result_sha256',
        'native_split_quality_selection_sha256',
        'native_split_quality_eval_run_id',
        'native_split_quality_source_run_id',
        'native_split_quality_consumer_attestation_sha256',
        'native_split_quality_authority_workflow_version',
        'native_split_quality_authority_run_id',
        'native_split_semantic_output_manifest_sha256',
        'native_split_semantic_boundary_manifest_sha256',
    )
    for field in duplicate_fields:
        observed = str(row.get(field) or '').strip()
        expected = str(evidence.get(field) or '').strip()
        if not observed or observed != expected:
            raise ValueError(f'native_split_energy_quality_binding_{field}_drift')
    command_sha = _strict_sha256(evidence.get('native_command_contract_sha256'))
    if (
        not command_sha
        or _strict_sha256(row.get('successful_command_contract_sha256')) != command_sha
    ):
        raise ValueError('native_split_energy_command_contract_sha256_drift')
    if _strict_sha256(row.get('native_split_energy_quality_binding_sha256')) != declared:
        raise ValueError('native_split_energy_quality_binding_duplicate_sha256_drift')
    if _strict_sha256(row.get('source_request_sha256')) != _strict_sha256(
        evidence.get('native_split_quality_source_request_sha256')
    ):
        raise ValueError('native_split_energy_generic_source_request_sha256_drift')
    attestation = row.get('native_split_quality_consumer_attestation')
    if not isinstance(attestation, dict):
        raise ValueError('native_split_quality_consumer_attestation_missing')
    attestation_body = dict(attestation)
    attestation_sha = _strict_sha256(attestation_body.pop('attestation_sha256', ''))
    if (
        not attestation_sha
        or _canonical_json_sha256(attestation_body) != attestation_sha
        or attestation_sha != _strict_sha256(
            evidence.get('native_split_quality_consumer_attestation_sha256')
        )
    ):
        raise ValueError('native_split_quality_consumer_attestation_sha256_drift')
    for field in (
        'native_split_quality_source_request_sha256',
        'native_split_quality_central_result_sha256',
        'native_split_quality_selection_sha256',
    ):
        if _strict_sha256(attestation.get(field)) != _strict_sha256(evidence.get(field)):
            raise ValueError(f'native_split_quality_consumer_attestation_{field}_drift')
    return command_sha, 'sealed_quality_binding_verified'


def _load_stream_command():
    """Load progress support from the package or the synchronized sibling."""
    try:
        from onnx_splitpoint_tool.native_progress import stream_command as helper
        return helper
    except Exception:
        import importlib.util
        sibling = SCRIPT_DIR / "native_progress.py"
        if not sibling.is_file():
            raise
        spec = importlib.util.spec_from_file_location("onnx_splitpoint_native_progress_standalone", sibling)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load native progress helper: {sibling}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module.stream_command


def _run(cmd: list[str], timeout: int|None=None, label: str="energy") -> dict[str,Any]:
    stream_command = _load_stream_command()
    progress_root = Path(os.environ.get("ONNX_SPLITPOINT_NATIVE_PROGRESS_DIR", "")) if os.environ.get("ONNX_SPLITPOINT_NATIVE_PROGRESS_DIR") else None
    return stream_command(
        cmd, timeout=timeout, label=label,
        heartbeat_s=float(os.environ.get("ONNX_SPLITPOINT_NATIVE_HEARTBEAT_S", "15")),
        progress_jsonl=(progress_root / "native_progress.jsonl" if progress_root else None),
        progress_json=(progress_root / "native_progress.json" if progress_root else None),
    )

def _safe(s: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]+','_',s).strip('_') or 'row'

def _single_command_option_parts(
    parts: list[str], name: str,
) -> tuple[str, int, bool]:
    """Return one unambiguous top-level shell option.

    A quoted workload command is one token after :func:`shlex.split`, so options
    inside that token are deliberately ignored.  Duplicate outer options and
    option-looking values fail closed instead of relying on argparse's
    last-value-wins behaviour.
    """
    prefix = f"{name}="
    matches: list[tuple[int, bool]] = []
    option_end = parts.index("--") if "--" in parts else len(parts)
    for index, part in enumerate(parts[:option_end]):
        if part == name:
            matches.append((index, False))
        elif part.startswith(prefix):
            matches.append((index, True))
    if len(matches) != 1:
        raise ValueError(
            f"{name} must occur exactly once at top level (found {len(matches)})"
        )
    index, inline = matches[0]
    if inline:
        value = parts[index][len(prefix):]
    else:
        if index + 1 >= len(parts):
            raise ValueError(f"{name} has no value")
        value = parts[index + 1]
        if value.startswith("--"):
            raise ValueError(f"{name} has an option-looking value: {value}")
    if not value:
        raise ValueError(f"{name} has no value")
    return value, index, inline


def _command_option(command: str, name: str) -> str:
    """Return exactly one top-level ``--name value``/``--name=value`` option."""
    value, _index, _inline = _single_command_option_parts(shlex.split(command), name)
    return value


def _replace_command_option(parts: list[str], name: str, value: str) -> list[str]:
    updated = list(parts)
    _old, index, inline = _single_command_option_parts(updated, name)
    if inline:
        updated[index] = f"{name}={value}"
    else:
        updated[index + 1] = value
    return updated


def _strict_positive_int(value: Any, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{field} must be a positive JSON integer")
    return value


def _strict_nonnegative_int(value: Any, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field} must be a non-negative JSON integer")
    return value


def _positive_cli_int(value: str, field: str) -> int:
    try:
        parsed = int(value)
    except Exception as exc:
        raise ValueError(f"{field} is not an integer: {value}") from exc
    if parsed <= 0 or str(parsed) != str(value).strip():
        raise ValueError(f"{field} must be a canonical positive integer: {value}")
    return parsed


def _resolved_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def _canonical_energy_identity(
    row: Mapping[str, Any],
) -> tuple[str, str, str, str, str, str]:
    from onnx_splitpoint_tool.native_job_identity import native_identity_key
    identity = native_identity_key(row)
    backend = identity[0]
    required = identity[:5] if backend.startswith("native_full_") else identity
    if any(not value for value in required):
        raise ValueError("energy result identity is incomplete")
    return identity


def _result_ledger_from_results(
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    ledger: list[dict[str, Any]] = []
    for result in results:
        row = (
            result.get("row")
            if isinstance(result.get("row"), Mapping)
            else {}
        )
        identity = _canonical_energy_identity(row)
        observation = (
            result.get("run", {}).get("measurement_start_observation")
            if isinstance(result.get("run"), Mapping)
            else {}
        )
        started = bool(
            isinstance(observation, Mapping)
            and observation.get("measurement_started") is True
        )
        status = (
            "success" if result.get("ok") is True
            else "dry_run" if result.get("dry_run") is True
            else "blocked_before_measurement"
            if result.get("skipped") else "failed"
        )
        ledger.append({
            **dict(zip(
                (
                    "backend", "model", "case", "setup_id",
                    "comparison_backend", "precision",
                ),
                identity,
            )),
            "status": status,
            "reason": str(
                result.get("error")
                or result.get("energy_not_started_reason")
                or result.get("blocked_reason")
                or result.get("skipped")
                or result.get("global_infrastructure_failure")
                or result.get("failure_category")
                or ""
            ),
            "measurement_started": started,
            "ok": result.get("ok"),
        })
    return ledger


def _result_ledger_valid(
    ledger: list[dict[str, Any]],
    expected_rows: list[dict[str, Any]],
) -> bool:
    try:
        ledger_identities = [
            _canonical_energy_identity(row) for row in ledger
        ]
        expected_identities = [
            _canonical_energy_identity(row) for row in expected_rows
        ]
    except ValueError:
        return False
    return bool(
        len(ledger) == len(expected_rows)
        and len(ledger_identities) == len(set(ledger_identities))
        and set(ledger_identities) == set(expected_identities)
        and all(
            str(row.get("status") or "") in {
                "success", "failed", "dry_run",
                "blocked_before_measurement",
            }
            and isinstance(row.get("measurement_started"), bool)
            for row in ledger
        )
    )


def _fresh_execution_output_dir(
    base_dir: str | Path,
    *,
    allowed_root: str | Path,
    attempt_id: str | None = None,
) -> tuple[Path, str]:
    """Atomically reserve an empty, execution-unique collector directory."""
    base = _resolved_path(base_dir)
    root = _resolved_path(allowed_root)
    if base != root and root not in base.parents:
        raise ValueError(f"planned measurement output escapes EvaluationRun: {base}")
    base.mkdir(parents=True, exist_ok=True)
    token = str(attempt_id or uuid.uuid4().hex).strip()
    if not token or re.fullmatch(r"[A-Za-z0-9_.-]+", token) is None:
        raise ValueError("measurement execution attempt id is invalid")
    target = base / f"attempt_{token}"
    if target.exists():
        raise FileExistsError(f"measurement execution target already exists: {target}")
    target.mkdir(mode=0o700, parents=False, exist_ok=False)
    if any(target.iterdir()):
        raise RuntimeError(f"fresh measurement execution target is not empty: {target}")
    return target.resolve(), token


def _prepare_measurement_execution(
    row: dict[str, Any],
    plan_payload: dict[str, Any],
    *,
    allowed_root: str | Path,
    attempt_id: str | None = None,
    validate_only: bool = False,
) -> dict[str, Any]:
    """Validate the frozen plan and optionally materialize one fresh attempt."""
    _canonical_energy_identity(row)
    command = str(row.get("measure_command") or "").strip()
    if not command:
        raise ValueError("measure_command is missing")
    try:
        parts = shlex.split(command)
    except ValueError as exc:
        raise ValueError(f"measure_command shell parsing failed: {exc}") from exc

    command_out, _out_index, _out_inline = _single_command_option_parts(parts, "--out")
    command_runs_raw, _runs_index, _runs_inline = _single_command_option_parts(parts, "--runs")
    command_setup, _setup_index, _setup_inline = _single_command_option_parts(parts, "--setup-id")
    command_run_id, _run_id_index, _run_id_inline = _single_command_option_parts(parts, "--run-id")
    command_runs = _positive_cli_int(command_runs_raw, "--runs")
    (
        quality_admission_sha,
        quality_admission_status,
    ) = _verify_energy_quality_admission(
        row,
        required=bool(
            plan_payload.get(
                "energy_quality_admission_required"
            ) is True
        ),
    )
    verified_admission = row.get("energy_quality_admission")
    admission_scope = (
        str(verified_admission.get("admission_scope") or "")
        if isinstance(verified_admission, dict)
        else ""
    )
    split_command_sha, split_binding_status = (
        _verify_split_energy_quality_binding(
            row,
            admission_scope=admission_scope,
        )
    )
    if split_command_sha:
        preflight_contract_sha = _strict_sha256(_command_option(
            command, '--preflight-expected-command-contract-sha256',
        ))
        if preflight_contract_sha != split_command_sha:
            raise ValueError(
                'measure_command_preflight_command_contract_sha256_drift'
            )

    planned_out_raw = str(row.get("measurement_output_base_dir") or "").strip()
    compatibility_out_raw = str(row.get("measurement_output_dir") or "").strip()
    if not planned_out_raw or not compatibility_out_raw:
        raise ValueError("planned measurement output binding is missing")
    planned_out = _resolved_path(planned_out_raw)
    compatibility_out = _resolved_path(compatibility_out_raw)
    command_out_path = _resolved_path(command_out)
    if not (planned_out == compatibility_out == command_out_path):
        raise ValueError(
            "plan/row/command measurement output mismatch: "
            f"base={planned_out} row={compatibility_out} command={command_out_path}"
        )
    allowed_output_root = _resolved_path(allowed_root)
    if (
        planned_out != allowed_output_root
        and allowed_output_root not in planned_out.parents
    ):
        raise ValueError(
            f"planned measurement output escapes EvaluationRun: {planned_out}"
        )

    row_plan_attempt = str(row.get("measurement_plan_attempt_id") or "").strip()
    payload_plan_attempt = str(plan_payload.get("measurement_plan_attempt_id") or "").strip()
    if not row_plan_attempt or row_plan_attempt != payload_plan_attempt:
        raise ValueError("row and plan measurement attempt identities do not match")

    row_runs = _strict_positive_int(
        row.get("measurement_requested_repeats"),
        "row.measurement_requested_repeats",
    )
    plan_runs = _strict_positive_int(
        plan_payload.get("energy_runs_per_row"), "plan.energy_runs_per_row",
    )
    if not (command_runs == row_runs == plan_runs):
        raise ValueError(
            "plan/row/command repeat mismatch: "
            f"plan={plan_runs} row={row_runs} command={command_runs}"
        )
    repeat_contract_fields = (
        "energy_profile_requested_runs_per_row",
        "energy_effective_runs_per_row",
        "energy_repeat_expansion_applied",
        "energy_repeat_expansion_reason",
    )
    row_repeat_contract_fields = (
        "measurement_profile_requested_repeats",
        "measurement_effective_repeats",
        "measurement_repeat_expansion_applied",
        "measurement_repeat_expansion_reason",
    )
    repeat_contract_present = any(
        field in plan_payload for field in repeat_contract_fields
    ) or any(field in row for field in row_repeat_contract_fields)
    expected_effective_runs: int | None = None
    if repeat_contract_present:
        if (
            not all(
                field in plan_payload
                for field in repeat_contract_fields
            )
            or not all(
                field in row
                for field in row_repeat_contract_fields
            )
        ):
            raise ValueError(
                "v2.72.1 repeat contract is incomplete"
            )
        profile_runs = _strict_positive_int(
            plan_payload[
                "energy_profile_requested_runs_per_row"
            ],
            "plan.energy_profile_requested_runs_per_row",
        )
        effective_runs = _strict_positive_int(
            plan_payload["energy_effective_runs_per_row"],
            "plan.energy_effective_runs_per_row",
        )
        row_profile_runs = _strict_positive_int(
            row["measurement_profile_requested_repeats"],
            "row.measurement_profile_requested_repeats",
        )
        row_effective_runs = _strict_positive_int(
            row["measurement_effective_repeats"],
            "row.measurement_effective_repeats",
        )
        plan_expanded = plan_payload[
            "energy_repeat_expansion_applied"
        ]
        row_expanded = row[
            "measurement_repeat_expansion_applied"
        ]
        if type(plan_expanded) is not bool or type(row_expanded) is not bool:
            raise ValueError(
                "v2.72.1 repeat expansion flags must be booleans"
            )
        expected_expanded = effective_runs != profile_runs
        expected_reason = (
            "frozen_window_method_ab_minimum"
            if expected_expanded else "none"
        )
        if not (
            profile_runs == row_profile_runs
            and effective_runs == row_effective_runs
            and effective_runs == plan_runs
            and effective_runs == row_runs
            and effective_runs == command_runs
            and plan_expanded == row_expanded == expected_expanded
            and str(
                plan_payload["energy_repeat_expansion_reason"]
            ) == expected_reason
            and str(
                row["measurement_repeat_expansion_reason"]
            ) == expected_reason
        ):
            raise ValueError(
                "v2.72.1 requested/effective repeat contract drift"
            )
        expected_effective_runs = effective_runs

    row_setup = str(row.get("setup_id") or "").strip()
    planned_setup = str(row.get("measurement_setup_id") or "").strip()
    if not row_setup or not (row_setup == planned_setup == command_setup):
        raise ValueError(
            "plan/row/command setup identity mismatch: "
            f"row={row_setup!r} planned={planned_setup!r} command={command_setup!r}"
        )
    planned_run_id = str(row.get("measurement_run_id") or "").strip()
    if not planned_run_id or planned_run_id != command_run_id:
        raise ValueError(
            "plan/command run identity mismatch: "
            f"planned={planned_run_id!r} command={command_run_id!r}"
        )

    if validate_only:
        return {
            "validated": True,
            "quality_admission_sha256": quality_admission_sha,
            "quality_admission_status": quality_admission_status,
            "split_quality_binding_status": split_binding_status,
            "expected_runs": command_runs,
            "expected_effective_runs": expected_effective_runs,
            "expected_setup_id": command_setup,
            "expected_run_id": command_run_id,
            "expected_command_contract_sha256": split_command_sha,
        }

    actual_out, execution_attempt_id = _fresh_execution_output_dir(
        planned_out, allowed_root=allowed_root, attempt_id=attempt_id,
    )
    actual_parts = _replace_command_option(parts, "--out", str(actual_out))
    actual_command = shlex.join(actual_parts)
    if _resolved_path(_command_option(actual_command, "--out")) != actual_out:
        raise RuntimeError("materialized command --out does not match execution attempt")

    runtime_row = dict(row)
    runtime_row.update({
        "measurement_planned_output_dir": str(planned_out),
        "measurement_output_dir": str(actual_out),
        "measurement_execution_attempt_id": execution_attempt_id,
        "measurement_output_materialized_at_execution": True,
        "measure_command_planned": command,
        "measure_command": actual_command,
    })
    return {
        "argv": actual_parts,
        "command": actual_command,
        "row": runtime_row,
        "output_dir": actual_out,
        "expected_runs": command_runs,
        "expected_effective_runs": expected_effective_runs,
        "expected_setup_id": command_setup,
        "expected_run_id": command_run_id,
        "expected_command_contract_sha256": split_command_sha,
        "split_quality_binding_status": split_binding_status,
        "energy_quality_admission_sha256": (
            quality_admission_sha
        ),
        "energy_quality_admission_status": (
            quality_admission_status
        ),
        "execution_attempt_id": execution_attempt_id,
    }


def _selected_energy_repeat_run_dir(
    run: Mapping[str, Any], *, measurement_root: Path, logical_index: int,
    declared_measurement_root: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Verify the collector's existing choice; never search or select by value.

    ``declared_measurement_root`` is only supplied by explicit offline archive
    projection. It maps exactly one original measurement root to the extracted
    root; the collector JSON and its hashes remain unchanged.
    """
    def fail(reason: str) -> None:
        raise ValueError(reason)

    def integer(value: Any) -> bool:
        return type(value) is int and value >= 0

    root = Path(measurement_root).resolve()
    declared = Path(declared_measurement_root or root)
    if not declared.is_absolute() or '..' in declared.parts:
        fail('energy_repeat_declared_measurement_root_invalid')
    if not isinstance(run, Mapping) or not integer(logical_index):
        fail('energy_repeat_index_invalid')
    if not integer(run.get('run_index')) or run['run_index'] != logical_index:
        fail('energy_repeat_logical_index_mismatch')
    keys = (
        'logical_repeat_index', 'selected_repeat_attempt_index',
        'repeat_attempt_count', 'repeat_attempt_history',
        'repeat_retry_attempted', 'repeat_retry_recovered',
    )
    retry_hints = any(key in run for key in keys) or any(
        'repeat_retry_attempts' in str(run.get(key) or '')
        for key in ('storage_dir', 'run_directory')
    )

    def checked_path(value: Any, expected_relative: Path) -> Path:
        if not isinstance(value, str) or not value.strip():
            fail('energy_repeat_path_missing_or_invalid')
        supplied = Path(value)
        if '..' in supplied.parts:
            fail('energy_repeat_path_outside_measurement')
        logical = supplied if supplied.is_absolute() else declared / supplied
        if not logical.is_relative_to(declared):
            fail('energy_repeat_path_outside_measurement')
        if logical.relative_to(declared) != expected_relative:
            fail('energy_repeat_path_selection_mismatch')
        physical = root / expected_relative
        resolved = physical.resolve()
        if not resolved.is_relative_to(root):
            fail('energy_repeat_path_outside_measurement')
        # A symlink to another attempt in the same job is also a wrong choice.
        if resolved != physical:
            fail('energy_repeat_path_symlink_or_selection_mismatch')
        return physical

    selected_index = 0
    if retry_hints:
        required = keys[:4]
        if any(key not in run for key in required):
            fail('energy_repeat_selection_incomplete')
        if (not integer(run['logical_repeat_index'])
            or run['logical_repeat_index'] != logical_index
            or not integer(run['selected_repeat_attempt_index'])
            or not integer(run['repeat_attempt_count'])
            or run['repeat_attempt_count'] < 1):
            fail('energy_repeat_selection_indices_invalid')
        history = run['repeat_attempt_history']
        if (not isinstance(history, list) or not history
            or len(history) != run['repeat_attempt_count']
            or any(not isinstance(item, Mapping) for item in history)):
            fail('energy_repeat_selection_history_invalid')
        if (any(not integer(item.get('attempt_index')) for item in history)
            or [item['attempt_index'] for item in history] != list(range(len(history)))
            or any(type(item.get('selected')) is not bool for item in history)):
            fail('energy_repeat_selection_history_invalid')
        selected = [item for item in history if item['selected'] is True]
        if len(selected) != 1:
            fail('energy_repeat_selection_ambiguous')
        selected_index = run['selected_repeat_attempt_index']
        # The collector stops at the first accepted retry. A later failed
        # attempt must never authorize falling back to an earlier good value.
        if (selected[0]['attempt_index'] != selected_index
            or selected_index != len(history) - 1):
            fail('energy_repeat_selection_index_conflict')
        for field, expected in (
            ('repeat_retry_attempted', len(history) > 1),
            ('repeat_retry_recovered', selected_index > 0),
        ):
            if field in run and (type(run[field]) is not bool or run[field] is not expected):
                fail('energy_repeat_selection_recovery_conflict')
        for item in history:
            attempt = item['attempt_index']
            relative = (Path(f'run_{logical_index:03d}') if attempt == 0 else
                        Path('repeat_retry_attempts') / f'repeat_{logical_index:03d}'
                        / f'attempt_{attempt:02d}' / 'run_000')
            checked_path(item.get('run_directory'), relative)
            if 'aggregate_path' in item:
                if attempt == 0:
                    expected_aggregate = Path('energy_aggregate.json')
                else:
                    expected_aggregate = relative.parent / 'energy_aggregate.json'
                checked_path(item['aggregate_path'], expected_aggregate)
            if ('accepted_for_logical_repeat' in item and
                (type(item['accepted_for_logical_repeat']) is not bool or
                 item['accepted_for_logical_repeat'] is not item['selected'])):
                fail('energy_repeat_selection_acceptance_conflict')
        selected_entry = selected[0]
        if (selected_entry.get('retry_reasons') not in (None, [])
            or selected_entry.get('final_energy_gate_status') not in (None, '', 'pass')
            or selected_entry.get('cancelled') is True
            or run.get('final_energy_gate_status') not in (None, '', 'pass')
            or run.get('cancelled') is True
            or run.get('global_infrastructure_failure')):
            fail('energy_repeat_selected_attempt_not_accepted')
    relative = (Path(f'run_{logical_index:03d}') if selected_index == 0 else
                Path('repeat_retry_attempts') / f'repeat_{logical_index:03d}'
                / f'attempt_{selected_index:02d}' / 'run_000')
    directory = checked_path(str(declared / relative), relative)
    if 'run_directory' in run:
        checked_path(run['run_directory'], relative)
    if 'storage_dir' in run:
        checked_path(run['storage_dir'], relative / 'collector_storage')
    # Embedded evidence may carry physical file references. They must name
    # this same selected attempt even when their recorded bytes are embedded.
    diagnostics = run.get('collector_and_workload_log_diagnostics')
    if isinstance(diagnostics, Mapping):
        for name in ('workload_stdout', 'workload_stderr', 'collector_stdout', 'collector_stderr'):
            log = diagnostics.get(name)
            if isinstance(log, Mapping) and 'path' in log:
                checked_path(log['path'], relative / (name + '.log'))
    for field, path_key, filename in (
        ('workload_timing', 'path', 'workload_timing.txt'),
        ('runtime_work_unit_evidence', 'evidence_path', 'workload_stdout.log'),
    ):
        evidence = run.get(field)
        if isinstance(evidence, Mapping) and path_key in evidence:
            checked_path(evidence[path_key], relative / filename)
    preflight = run.get('preflight_evidence')
    if isinstance(preflight, Mapping):
        if preflight.get('cancelled') is True:
            fail('energy_repeat_selected_preflight_cancelled')
        if 'repeat_index' in preflight:
            physical_index = 0 if selected_index else logical_index
            if not integer(preflight['repeat_index']) or preflight['repeat_index'] != physical_index:
                fail('energy_repeat_selected_preflight_index_mismatch')
        for field, filename in (
            ('command_path', 'preflight_command.sh'), ('stdout_path', 'preflight_stdout.log'),
            ('stderr_path', 'preflight_stderr.log'), ('attestation_path', 'preflight_attestation.json'),
        ):
            if field in preflight:
                checked_path(preflight[field], relative / 'preflight' / filename)
    if not directory.is_dir():
        fail('energy_repeat_selected_directory_missing')
    stdout = directory / 'workload_stdout.log'
    if not stdout.is_file():
        fail('energy_repeat_selected_stdout_missing')
    checked_path(str(declared / relative / stdout.name), relative / stdout.name)
    return directory, {
        'logical_repeat_index': logical_index,
        'selected_repeat_attempt_index': selected_index,
        'selected_repeat_run_directory': str(directory),
        'selected_repeat_declared_run_directory': str(declared / relative),
        'repeat_selection_provenance': 'collector_explicit_selection' if retry_hints else 'legacy_initial_without_retry_hints',
    }


def _verify_fresh_fast_energy_completion(
    run: Mapping[str, Any], planned_row: Mapping[str, Any], *, run_dir: Path,
) -> tuple[dict[str, Any] | None, str]:
    """Join one fresh runtime publication to its nonce, exact count and window."""
    try:
        from scripts.native_producer_energy_plan import _verified_fast_energy_completion
    except ImportError:
        from native_producer_energy_plan import _verified_fast_energy_completion
    path = run_dir / "workload_stdout.log"
    try:
        if not path.is_file() or not path.resolve().is_relative_to(run_dir.resolve()):
            return None, "energy_completion_stdout_missing_or_outside_run"
        stat_before = path.stat()
        if stat_before.st_size > 8 * 1024 * 1024:
            return None, "energy_completion_stdout_exceeds_bound"
        raw = path.read_bytes()
        stat_after = path.stat()
        if ((stat_before.st_dev, stat_before.st_ino, stat_before.st_size,
             stat_before.st_mtime_ns, stat_before.st_ctime_ns) !=
            (stat_after.st_dev, stat_after.st_ino, stat_after.st_size,
             stat_after.st_mtime_ns, stat_after.st_ctime_ns)):
            return None, "energy_completion_stdout_changed_during_import"
        if len(raw) > 8 * 1024 * 1024:
            return None, "energy_completion_stdout_exceeds_bound"
        captured = (run.get("collector_and_workload_log_diagnostics") or {}).get("workload_stdout") or {}
        if (captured.get("available") is not True
            or type(captured.get("bytes")) is not int or captured["bytes"] != len(raw)
            or captured.get("sha256") != hashlib.sha256(raw).hexdigest()):
            return None, "energy_completion_captured_stdout_hash_mismatch"
        prefix = "__SPLITPOINT_ENERGY_COMPLETION__="
        markers = [line[len(prefix):] for line in raw.decode("utf-8").splitlines() if line.startswith(prefix)]
        if len(markers) != 1:
            return None, "energy_completion_fresh_marker_missing_or_ambiguous"
        fresh = _strict_json_text(markers[0], label="fresh energy completion")
        if not isinstance(fresh, dict):
            return None, "energy_completion_fresh_marker_invalid"
        preflight = run.get("preflight_evidence") or {}
        if (preflight.get("ok") is not True or not preflight.get("nonce")
            or fresh.get("energy_preflight_nonce") != preflight["nonce"]):
            return None, "energy_completion_preflight_nonce_mismatch"
        command = planned_row.get("native_command_contract") or {}
        if fresh.get("source_contract_sha256") != command.get("contract_sha256"):
            return None, "energy_completion_source_command_mismatch"
        work = run.get("runtime_work_unit_evidence") or {}
        timing = run.get("workload_timing") or {}
        if (work.get("exact") is not True or type(work.get("count")) is not int
            or work["count"] <= 0 or work["count"] != fresh.get("completed_work_units")
            or type(fresh.get("warmup")) is not int or fresh["warmup"] != 0):
            return None, "energy_completion_count_or_warmup_invalid"
        if (timing.get("status") != "ok" or timing.get("rc") != 0
            or type(timing.get("start_ns")) is not int or type(timing.get("end_ns")) is not int
            or timing["start_ns"] <= 0 or timing["end_ns"] <= timing["start_ns"]
            or fresh.get("energy_completion_observation_scope") != "fresh_energy_invocation"
            or fresh.get("energy_completion_window_source") != "collector_command_marker_window"):
            return None, "energy_completion_own_measurement_window_missing"
        att, comparison = _verified_fast_energy_completion(
            fresh, None, command, energy_observation=True,
        )
        planned_comparison = planned_row.get("completed_task_comparison_output_endpoint_id")
        if planned_comparison and comparison["output_endpoint_id"] != planned_comparison:
            return None, "energy_completion_comparison_endpoint_mismatch"
        return {
            "status": "fresh_energy_completion_nonce_count_and_window_verified",
            "completion_execution_attestation": att,
            "comparison_output_endpoint_id": comparison["output_endpoint_id"],
            "completed_work_units": work["count"],
            "preflight_nonce": preflight["nonce"],
            "workload_timing": dict(timing),
            "stdout_sha256": hashlib.sha256(raw).hexdigest(),
            "observation_relation": "postflight_oracle_sentinel",
            "sentinel_location": "outside_fifo_performance_timing_inside_energy_command_window",
        }, "verified"
    except Exception as exc:
        return None, "energy_completion_invalid:" + str(exc)


def _attach_energy_aggregate(
    result: dict[str, Any],
    command: str,
    measurement_output_dir: str | Path | None = None,
    *,
    expected_runs: int | None = None,
    expected_effective_runs: int | None = None,
    expected_run_id: str | None = None,
    expected_setup_id: str | None = None,
    expected_command_contract_sha256: str | None = None,
    execution_started_ns: int | None = None,
    aggregate_absent_before_execution: bool | None = None,
    expected_native_row: Mapping[str, Any] | None = None,
    measurement_path_mapping: tuple[str | Path, str | Path] | None = None,
    archived_aggregate_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Embed the compact collector aggregate before external paths disappear.

    The streaming helper intentionally retains only a bounded output tail. A
    verbose three-repeat A/B run can therefore truncate the beginning of the
    printed aggregate. The authoritative compact JSON already exists below the
    CLI ``--out`` directory, so bind it to the EvaluationRun result directly.
    Raw captures remain external and are not copied.
    """
    binding_errors: list[str] = []
    completion_errors: list[str] = []
    fresh_completion_evidence: list[dict[str, Any]] = []
    # Reimports must not retain evidence from a prior successful projection.
    result.pop('fresh_energy_completion_evidence', None)
    result.pop('energy_repeat_selection_evidence', None)
    selection_evidence: list[dict[str, Any]] = []

    def reject(reason: str) -> None:
        if reason not in binding_errors:
            binding_errors.append(reason)

    def incomplete(reason: str) -> None:
        if reason not in completion_errors:
            completion_errors.append(reason)

    aggregate_path: Path | None = None
    aggregate: dict[str, Any] | None = None
    raw = b''
    try:
        command_out = _resolved_path(_command_option(command, '--out'))
        command_runs = _positive_cli_int(_command_option(command, '--runs'), '--runs')
        command_run_id = _command_option(command, '--run-id')
        command_setup_id = _command_option(command, '--setup-id')
        declared_out = str(measurement_output_dir or '').strip()
        if not declared_out:
            reject('declared_measurement_output_missing')
            out_dir = command_out
        else:
            out_dir = _resolved_path(declared_out)
            if out_dir != command_out:
                reject('declared_output_does_not_match_command_out')

        if expected_runs is None:
            expected_runs = command_runs
        if type(expected_runs) is not int or expected_runs <= 0:
            reject('expected_repeat_count_missing_or_invalid')
        elif command_runs != expected_runs:
            reject(
                f'command_repeat_count_{command_runs}_does_not_match_expected_{expected_runs}'
            )
        expected_run_id = str(expected_run_id or command_run_id)
        expected_setup_id = str(expected_setup_id or command_setup_id)
        if command_run_id != expected_run_id:
            reject('command_run_id_does_not_match_expected')
        if command_setup_id != expected_setup_id:
            reject('command_setup_id_does_not_match_expected')
        if type(execution_started_ns) is not int or execution_started_ns <= 0:
            reject('execution_start_time_missing_or_invalid')
        if aggregate_absent_before_execution is not True:
            reject('aggregate_absence_before_execution_not_verified')
        if type(result.get('rc')) is not int:
            incomplete('measurement_process_rc_invalid')
        elif result['rc'] != 0:
            incomplete('measurement_process_rc_nonzero')

        # Do not inspect or embed a file from a path whose plan binding already
        # failed.  This is the critical stale-output barrier.
        if binding_errors:
            raise ValueError(';'.join(binding_errors))

        declared_out_dir = out_dir
        if measurement_path_mapping is not None:
            original_root, extracted_root = map(Path, measurement_path_mapping)
            if (not original_root.is_absolute() or '..' in original_root.parts
                or original_root != declared_out_dir or not extracted_root.is_absolute()
                or '..' in extracted_root.parts or extracted_root.resolve() != extracted_root):
                reject('archive_measurement_root_mapping_invalid')
                raise ValueError(';'.join(binding_errors))
            out_dir = extracted_root
            result['energy_archive_path_projection'] = {
                'original_measurement_root': str(original_root),
                'extracted_measurement_root': str(extracted_root),
                'original_bytes_modified': False,
            }
        aggregate_path = out_dir / 'energy_aggregate.json'
        if (aggregate_path.resolve() != aggregate_path
            or not aggregate_path.resolve().is_relative_to(out_dir)):
            reject('aggregate_path_symlink_or_outside_measurement')
            raise ValueError(';'.join(binding_errors))
        if not aggregate_path.is_file():
            reject('aggregate_missing')
            raise FileNotFoundError(str(aggregate_path))
        stat_before = aggregate_path.stat()
        authoritative_mtime_ns = stat_before.st_mtime_ns
        archived_sha = None
        if measurement_path_mapping is not None:
            # Deterministic archives normalize filesystem timestamps. The
            # original timestamp is admissible only together with its existing
            # bound byte digest from the historical checkpoint/result.
            provenance = archived_aggregate_provenance
            if (not isinstance(provenance, Mapping)
                or type(provenance.get('mtime_ns')) is not int
                or provenance['mtime_ns'] <= 0
                or not _strict_sha256(provenance.get('sha256'))):
                reject('archive_original_aggregate_provenance_missing')
                raise ValueError(';'.join(binding_errors))
            authoritative_mtime_ns = provenance['mtime_ns']
            archived_sha = provenance['sha256']
            result['energy_archive_path_projection'].update({
                'original_aggregate_mtime_ns': authoritative_mtime_ns,
                'extracted_aggregate_mtime_ns': stat_before.st_mtime_ns,
                'timestamp_source': 'historical_bound_result',
            })
        if (
            authoritative_mtime_ns + _AGGREGATE_MTIME_TOLERANCE_NS
            < int(execution_started_ns)
        ):
            reject('aggregate_predates_measurement_execution')
        raw = aggregate_path.read_bytes()
        stat_after = aggregate_path.stat()
        if archived_sha is not None and hashlib.sha256(raw).hexdigest() != archived_sha:
            reject('archive_original_aggregate_hash_mismatch')
        if (
            stat_before.st_mtime_ns != stat_after.st_mtime_ns
            or stat_before.st_size != stat_after.st_size
            or stat_before.st_ino != stat_after.st_ino
            or stat_before.st_dev != stat_after.st_dev
            or stat_before.st_ctime_ns != stat_after.st_ctime_ns
        ):
            reject('aggregate_changed_during_import')
        try:
            parsed = _strict_json_text(
                raw.decode('utf-8'), label=str(aggregate_path),
            )
        except Exception:
            reject('aggregate_invalid_json')
            parsed = None
        if not isinstance(parsed, dict) or not parsed:
            reject('aggregate_missing_or_invalid_object')
        else:
            aggregate = dict(parsed)

        if aggregate is not None:
            if 'ok' not in aggregate:
                reject('aggregate_ok_missing')
            elif aggregate.get('ok') is not True:
                incomplete('aggregate_ok_not_exactly_true')
            if 'repeat_contract_complete' not in aggregate:
                reject('repeat_contract_complete_missing')
            elif aggregate.get('repeat_contract_complete') is not True:
                incomplete('repeat_contract_complete_not_exactly_true')
            if aggregate.get('scientific_primary_method') != 'command_marker_window':
                reject('aggregate_scientific_primary_method_not_command_marker_window')
            if aggregate.get('scientific_primary_energy_status') != 'available':
                incomplete('aggregate_scientific_primary_energy_status_not_available')
            if str(aggregate.get('run_id') or '') != expected_run_id:
                reject('aggregate_run_id_mismatch')
            if str(aggregate.get('setup_id') or '') != expected_setup_id:
                reject('aggregate_setup_id_mismatch')
            expected_contract_sha = _strict_sha256(
                expected_command_contract_sha256
            )
            if expected_contract_sha:
                if aggregate.get('preflight_requested') is not True:
                    reject('aggregate_split_preflight_not_requested')
                if _strict_sha256(
                    aggregate.get('preflight_expected_command_contract_sha256')
                ) != expected_contract_sha:
                    reject('aggregate_split_command_contract_sha256_mismatch')
            aggregate_out_raw = str(aggregate.get('out_dir') or '').strip()
            if not aggregate_out_raw:
                reject('aggregate_out_dir_missing')
            elif _resolved_path(aggregate_out_raw) != declared_out_dir:
                reject('aggregate_out_dir_mismatch')

            # ``--runs`` is the collector request.  A/B may legitimately raise
            # it to its frozen minimum, so bind the command to the runtime
            # contract's requested count and all materialized/result counts to
            # that contract's effective count.
            ab_runtime = aggregate.get('energy_window_method_ab')
            if not isinstance(ab_runtime, dict):
                reject('aggregate_energy_window_method_ab_missing')
                effective_runs = None
            else:
                if ab_runtime.get('scientific_primary_method') != 'command_marker_window':
                    reject(
                        'aggregate_ab_scientific_primary_method_not_command_marker_window'
                    )
                try:
                    aggregate_requested_runs = _strict_positive_int(
                        ab_runtime.get('requested_run_count'),
                        'aggregate.energy_window_method_ab.requested_run_count',
                    )
                except ValueError:
                    reject('aggregate_requested_run_count_missing_or_invalid')
                else:
                    if aggregate_requested_runs != expected_runs:
                        reject(
                            'aggregate_requested_run_count_'
                            f'{aggregate_requested_runs}_does_not_match_planned_{expected_runs}'
                        )
                try:
                    effective_runs = _strict_positive_int(
                        ab_runtime.get('effective_run_count'),
                        'aggregate.energy_window_method_ab.effective_run_count',
                    )
                except ValueError:
                    reject('aggregate_effective_run_count_missing_or_invalid')
                    effective_runs = None
                else:
                    if (
                        expected_effective_runs is not None
                        and effective_runs != expected_effective_runs
                    ):
                        reject(
                            'aggregate_effective_run_count_'
                            f'{effective_runs}_does_not_match_frozen_'
                            f'{expected_effective_runs}'
                        )
            if expected_contract_sha:
                verified_preflights = aggregate.get('preflight_verified_run_count')
                if (
                    isinstance(verified_preflights, bool)
                    or not isinstance(verified_preflights, int)
                    or effective_runs is None
                    or verified_preflights != effective_runs
                ):
                    reject('aggregate_split_preflight_verified_run_count_mismatch')

            materialized_repeat_fields = (
                'run_count',
                'requested_valid_repeat_count',
                'materialized_logical_repeat_count',
            )
            repeat_values: dict[str, int] = {}
            for field in materialized_repeat_fields:
                try:
                    repeat_values[field] = _strict_positive_int(
                        aggregate.get(field), f'aggregate.{field}',
                    )
                except ValueError:
                    reject(f'aggregate_{field}_missing_or_invalid')
            for field, value in repeat_values.items():
                if effective_runs is None or value != effective_runs:
                    reject(
                        f'aggregate_{field}_{value}_does_not_match_effective_{effective_runs}'
                    )
            for field in (
                'valid_postprocessed_runs',
                'scientific_primary_valid_run_count',
            ):
                try:
                    value = _strict_nonnegative_int(
                        aggregate.get(field), f'aggregate.{field}',
                    )
                except ValueError:
                    reject(f'aggregate_{field}_missing_or_invalid')
                    continue
                repeat_values[field] = value
                if effective_runs is None or value > effective_runs:
                    reject(
                        f'aggregate_{field}_{value}_exceeds_effective_{effective_runs}'
                    )
                elif value < effective_runs:
                    incomplete(
                        f'aggregate_{field}_{value}_does_not_match_effective_{effective_runs}'
                    )

            runs = aggregate.get('runs')
            if not isinstance(runs, list):
                reject('aggregate_runs_missing_or_not_list')
            elif effective_runs is None or len(runs) != effective_runs:
                reject(
                    f'aggregate_runs_length_{len(runs)}_does_not_match_effective_{effective_runs}'
                )
            else:
                indices = [
                    item.get('run_index') if isinstance(item, dict) else None
                    for item in runs
                ]
                if any(type(value) is not int for value in indices) or indices != list(range(effective_runs)):
                    reject('aggregate_run_indices_not_exact_contiguous_sequence')
                fast_completion = bool(
                    isinstance(expected_native_row, Mapping)
                    and expected_native_row.get('completed_task_completion_mode')
                    == 'native_three_stage_fast_oracle_outside_timing')
                for index, run in enumerate(runs):
                    has_selection = isinstance(run, Mapping) and any(
                        key in run for key in ('logical_repeat_index', 'selected_repeat_attempt_index',
                                               'repeat_attempt_history', 'repeat_attempt_count',
                                               'repeat_retry_attempted', 'repeat_retry_recovered'))
                    if fast_completion or has_selection:
                        try:
                            run_directory, selection = _selected_energy_repeat_run_dir(
                                run, measurement_root=out_dir, logical_index=index,
                                declared_measurement_root=declared_out_dir,
                            )
                        except ValueError as exc:
                            incomplete(f'energy_repeat_{index}_' + str(exc))
                            continue
                        selection_evidence.append(selection)
                        if not fast_completion:
                            continue
                        evidence, reason = _verify_fresh_fast_energy_completion(
                            run, expected_native_row, run_dir=run_directory,
                        )
                        if evidence is None:
                            incomplete(f'energy_repeat_{index}_' + reason)
                        else:
                            fresh_completion_evidence.append({'run_index': index, **selection, **evidence})


            statistics_block = aggregate.get('scientific_primary_energy_statistics')
            energy_stats = (
                statistics_block.get('energy_j')
                if isinstance(statistics_block, dict) else None
            )
            if not isinstance(energy_stats, dict):
                reject('scientific_primary_energy_statistics_missing')
            else:
                try:
                    statistics_n = _strict_nonnegative_int(
                        energy_stats.get('n'),
                        'aggregate.scientific_primary_energy_statistics.energy_j.n',
                    )
                except ValueError:
                    reject('scientific_primary_energy_statistics_n_missing_or_invalid')
                else:
                    if effective_runs is None or statistics_n > effective_runs:
                        reject(
                            'scientific_primary_energy_statistics_n_'
                            f'{statistics_n}_exceeds_effective_{effective_runs}'
                        )
                    elif statistics_n < effective_runs:
                        incomplete(
                            'scientific_primary_energy_statistics_n_'
                            f'{statistics_n}_does_not_match_effective_{effective_runs}'
                        )

            result['energy_aggregate_requested_repeat_count'] = repeat_values.get(
                'requested_valid_repeat_count'
            )
            result['energy_aggregate_valid_repeat_count'] = repeat_values.get(
                'valid_postprocessed_runs'
            )
            result['energy_aggregate_materialized_repeat_count'] = repeat_values.get(
                'materialized_logical_repeat_count'
            )
            result['energy_aggregate_planned_request_repeat_count'] = expected_runs
            result[
                'energy_aggregate_planned_effective_repeat_count'
            ] = expected_effective_runs
            result['energy_aggregate_effective_repeat_count'] = effective_runs
            result['energy_aggregate_mtime_ns'] = authoritative_mtime_ns
            result['energy_execution_started_ns'] = execution_started_ns
            result['energy_aggregate_absent_before_execution'] = aggregate_absent_before_execution
            result['energy_aggregate_mtime_tolerance_ns'] = (
                _AGGREGATE_MTIME_TOLERANCE_NS
            )

        if binding_errors:
            raise ValueError(';'.join(binding_errors))

        assert aggregate is not None and aggregate_path is not None
        result['energy_aggregate'] = aggregate
        if selection_evidence:
            result['energy_repeat_selection_evidence'] = selection_evidence
        if fresh_completion_evidence:
            result['fresh_energy_completion_evidence'] = fresh_completion_evidence
        result['energy_aggregate_path'] = str(aggregate_path.resolve())
        result['energy_aggregate_sha256'] = hashlib.sha256(raw).hexdigest()
        result['energy_aggregate_embedded'] = True
        result['energy_aggregate_bound'] = True
        result['energy_aggregate_complete'] = not completion_errors
        result['energy_aggregate_verified'] = not completion_errors
        result['energy_aggregate_import_status'] = (
            'verified' if not completion_errors else 'bound_incomplete'
        )
        result['energy_aggregate_completion_errors'] = list(completion_errors)
    except Exception as exc:
        result.pop('energy_aggregate', None)
        result.pop('energy_aggregate_path', None)
        result.pop('energy_aggregate_sha256', None)
        result['energy_aggregate_embedded'] = False
        result['energy_aggregate_bound'] = False
        result['energy_aggregate_complete'] = False
        result['energy_aggregate_verified'] = False
        result['energy_aggregate_validation_errors'] = binding_errors or [
            f'aggregate_import_exception_{type(exc).__name__}'
        ]
        result['energy_aggregate_completion_errors'] = list(completion_errors)
        result['energy_aggregate_import_status'] = 'rejected_fail_closed'
        result['energy_aggregate_embed_error'] = f'{type(exc).__name__}: {exc}'
        if aggregate_path is not None:
            result['energy_aggregate_rejected_path'] = str(aggregate_path.resolve())
    else:
        result['energy_aggregate_validation_errors'] = []
    return result


_RESUME_ROW_CONTRACT_FIELDS = (
    'backend',
    'model',
    'case',
    'setup_id',
    'comparison_backend',
    'successful_command_contract_sha256',
    'completed_task_completion_mode',
    'native_command_contract',
    'energy_completion_requires_fresh_observation',
    'command_contract_file_sha256',
    'native_split_energy_quality_binding_sha256',
    'native_split_quality_binding_sha256',
    'native_split_quality_preselection_sha256',
    'native_split_quality_source_request_sha256',
    'native_split_quality_central_result_sha256',
    'native_split_quality_selection_sha256',
    'native_split_quality_consumer_attestation_sha256',
    'native_split_semantic_output_manifest_sha256',
    'native_split_semantic_boundary_manifest_sha256',
    'source_request_sha256',
    'model_sha256',
    'physical_output_endpoint_id',
    'physical_endpoint_contract_hash',
    'comparison_output_endpoint_id',
    'comparison_endpoint_contract_hash',
    'completion_pairing_eligible',
    'measurement_requested_repeats',
    'measurement_profile_requested_repeats',
    'measurement_effective_repeats',
    'measurement_repeat_expansion_applied',
    'measurement_repeat_expansion_reason',
    'duration_s',
    'target_duration_s',
    'window_method_ab',
    'energy_scope',
    'energy_window',
    'energy_evidence_tier',
    'energy_tier',
    'screening_only',
    'screening_energy',
    'measure_all_runtime_successful',
    'measurement_admission_policy',
    'smoke_diagnostic',
    'require_runtime_work_units',
    'require_command_window_alignment',
    'energy_calibration_sha256',
    'pipeline_contract_sha256',
    'pipeline_preprocessing_sha256',
    'pipeline_decoder_sha256',
    'pipeline_nms_sha256',
    'prepared_feed_task',
    'prepared_feed_preprocess_mode',
    'prepared_feed_letterbox_pad_value',
    'prepared_feed_source_image_sha256',
)

# P0.3 freezes measurement identity independently from Quality, Semantic,
# pairing, endpoint-claim, and reporting annotations.  Legacy rows continue to
# use the wider contract above; only rows carrying the explicit P0.3 admission
# record use this technical subset.
_P03_RESUME_ROW_CONTRACT_FIELDS = (
    'backend',
    'model',
    'case',
    'setup_id',
    'successful_command_contract_sha256',
    'completed_task_completion_mode',
    'native_command_contract',
    'energy_completion_requires_fresh_observation',
    'command_contract_file_sha256',
    'measurement_requested_repeats',
    'measurement_profile_requested_repeats',
    'measurement_effective_repeats',
    'measurement_repeat_expansion_applied',
    'measurement_repeat_expansion_reason',
    'duration_s',
    'target_duration_s',
    'window_method_ab',
    'energy_scope',
    'energy_window',
    'require_runtime_work_units',
    'require_command_window_alignment',
    'energy_calibration_sha256',
    'prepared_feed_preprocess_mode',
    'prepared_feed_letterbox_pad_value',
    'prepared_feed_source_image_sha256',
)

_RESUME_PLAN_CONTRACT_FIELDS = (
    'schema',
    'schema_version',
    'energy_runs_per_row',
    'energy_profile_requested_runs_per_row',
    'energy_effective_runs_per_row',
    'energy_repeat_expansion_applied',
    'energy_repeat_expansion_reason',
    'window_method_ab',
    'require_runtime_work_units',
    'require_command_window_alignment',
    'physical_scope',
    'window_label',
    'calibration_sha256',
    'pipeline_contract_sha256',
    'model_hash_map_sha256',
    'energy_evidence_tier',
    'energy_tier',
    'screening_only',
    'screening_energy',
    'measure_all_runtime_successful',
    'measurement_admission_policy',
    'smoke_diagnostic',
    'diagnostic_only',
    'hardware_setups_file',
    'hardware_setups_file_sha256',
)

_P03_RESUME_PLAN_CONTRACT_FIELDS = (
    'schema',
    'schema_version',
    'energy_runs_per_row',
    'energy_profile_requested_runs_per_row',
    'energy_effective_runs_per_row',
    'energy_repeat_expansion_applied',
    'energy_repeat_expansion_reason',
    'window_method_ab',
    'require_runtime_work_units',
    'require_command_window_alignment',
    'physical_scope',
    'window_label',
    'calibration_sha256',
    'hardware_setups_file',
    'hardware_setups_file_sha256',
)

_RESUME_EXECUTION_CONTEXT_SCHEMA = (
    'onnx-splitpoint/native-energy-execution-context'
)
_RESUME_EXECUTION_CONTEXT_OPTIONS = (
    ('remote_root', '--remote-root', 'remote_path'),
    ('remote_tool_dir', '--remote-tool-dir', 'remote_path'),
    ('hailo8_ssh', '--hailo8-ssh', 'ssh'),
    ('hailo10_ssh', '--hailo10-ssh', 'ssh'),
    ('deepx_ssh', '--deepx-ssh', 'ssh'),
    ('hailo8_env', '--hailo8-env', 'shell'),
    ('hailo10_env', '--hailo10-env', 'shell'),
    ('deepx_env', '--deepx-env', 'shell'),
    ('engine_build_python', '--engine-build-python', 'token'),
)
_RESUME_REMOTE_COMMAND_FIELDS = (
    'command_file',
    'preflight_command_file',
)
_RESUME_REMOTE_PATH_FIELDS = (
    'remote_command_contract_file',
    'preflight_runtime_attestation_path_template',
)
_RESUME_COMMAND_MAX_BYTES = 2 * 1024 * 1024


def _normalize_resume_execution_context_value(
    field: str, value: Any, kind: str,
) -> Any:
    if not isinstance(value, str):
        raise ValueError(
            f'resume execution context {field} must be a string'
        )
    if '\x00' in value:
        raise ValueError(
            f'resume execution context {field} contains NUL'
        )
    if kind == 'remote_path':
        stripped = value.strip()
        if not stripped or not stripped.startswith('/'):
            raise ValueError(
                f'resume execution context {field} must be an absolute '
                'remote path'
            )
        return posixpath.normpath(stripped)
    if kind == 'shell':
        try:
            return tuple(shlex.split(value, posix=True))
        except ValueError as exc:
            raise ValueError(
                f'resume execution context {field} is not valid shell text: '
                f'{exc}'
            ) from exc
    return value.strip()


def _validated_resume_execution_context(
    value: Any, *, label: str,
) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError(f'{label} has no execution_context object')
    if (
        value.get('schema') != _RESUME_EXECUTION_CONTEXT_SCHEMA
        or value.get('schema_version') != 1
    ):
        raise ValueError(f'{label} execution_context schema is invalid')
    context: dict[str, str] = {}
    for field, _option, kind in _RESUME_EXECUTION_CONTEXT_OPTIONS:
        raw = value.get(field)
        _normalize_resume_execution_context_value(field, raw, kind)
        assert isinstance(raw, str)
        context[field] = raw
    return context


def _single_argv_option_value(
    argv: list[Any], option: str, *, label: str,
) -> str:
    parts = [str(value) for value in argv]
    prefix = f'{option}='
    matches: list[tuple[int, bool]] = []
    option_end = parts.index('--') if '--' in parts else len(parts)
    for index, part in enumerate(parts[:option_end]):
        if part == option:
            matches.append((index, False))
        elif part.startswith(prefix):
            matches.append((index, True))
    if len(matches) != 1:
        raise ValueError(
            f'{label} must contain {option} exactly once '
            f'(found {len(matches)})'
        )
    index, inline = matches[0]
    if inline:
        return parts[index][len(prefix):]
    if index + 1 >= option_end:
        raise ValueError(f'{label} has no value for {option}')
    return parts[index + 1]


def _execution_context_from_plan_command(
    existing_report: dict[str, Any],
) -> dict[str, str] | None:
    plan_result = existing_report.get('plan')
    if not isinstance(plan_result, dict):
        return None
    command = plan_result.get('cmd')
    if not isinstance(command, list):
        return None
    context: dict[str, str] = {}
    for field, option, kind in _RESUME_EXECUTION_CONTEXT_OPTIONS:
        raw = _single_argv_option_value(
            command, option, label='canonical plan command',
        )
        _normalize_resume_execution_context_value(field, raw, kind)
        context[field] = raw
    return context


def _execution_contexts_equal(
    left: dict[str, str], right: dict[str, str],
) -> tuple[bool, str]:
    for field, _option, kind in _RESUME_EXECUTION_CONTEXT_OPTIONS:
        left_value = _normalize_resume_execution_context_value(
            field, left.get(field), kind,
        )
        right_value = _normalize_resume_execution_context_value(
            field, right.get(field), kind,
        )
        if left_value != right_value:
            return False, field
    return True, ''


def _canonical_resume_execution_context(
    existing_report: dict[str, Any],
) -> dict[str, str]:
    old_plan = existing_report.get('plan_payload')
    if not isinstance(old_plan, dict):
        raise ValueError('existing energy result has no frozen plan_payload')
    payload_context: dict[str, str] | None = None
    if old_plan.get('execution_context') is not None:
        payload_context = _validated_resume_execution_context(
            old_plan.get('execution_context'), label='canonical plan payload',
        )
    command_context = _execution_context_from_plan_command(existing_report)
    if payload_context is None and command_context is None:
        raise ValueError(
            'canonical energy plan has no frozen execution context'
        )
    if payload_context is not None and command_context is not None:
        equal, field = _execution_contexts_equal(
            payload_context, command_context,
        )
        if not equal:
            raise ValueError(
                f'canonical execution context disagrees with plan command: '
                f'{field}'
            )
    return dict(payload_context or command_context or {})


def _explicit_option_present(argv: list[str], option: str) -> bool:
    prefix = f'{option}='
    count = sum(
        1 for value in argv
        if value == option or value.startswith(prefix)
    )
    if count > 1:
        raise ValueError(
            f'resume execution context option occurs more than once: {option}'
        )
    return count == 1


def _bind_resume_execution_context(
    namespace: argparse.Namespace,
    argv: list[str],
    canonical_context: dict[str, str],
) -> None:
    for field, option, kind in _RESUME_EXECUTION_CONTEXT_OPTIONS:
        canonical_value = canonical_context[field]
        if _explicit_option_present(argv, option):
            observed_value = getattr(namespace, field)
            if _normalize_resume_execution_context_value(
                field, observed_value, kind,
            ) != _normalize_resume_execution_context_value(
                field, canonical_value, kind,
            ):
                raise ValueError(
                    f'resume execution context drift for {option}'
                )
        # Use the canonical spelling even when an explicitly supplied value was
        # semantically equivalent.  This makes generated remote commands
        # reproducible and prevents shell-quoting drift.
        setattr(namespace, field, canonical_value)


def _read_resume_command_file(path_value: Any, *, label: str) -> str:
    path = Path(str(path_value or '')).expanduser()
    if not str(path_value or '').strip():
        raise ValueError(f'{label} path is missing')
    if path.is_symlink() or not path.is_file():
        raise ValueError(f'{label} is missing or not a regular file: {path}')
    raw = path.read_bytes()
    if len(raw) > _RESUME_COMMAND_MAX_BYTES:
        raise ValueError(f'{label} exceeds the maximum allowed size')
    try:
        return raw.decode('utf-8')
    except UnicodeDecodeError as exc:
        raise ValueError(f'{label} is not UTF-8') from exc


def _normalized_resume_remote_command(
    row: dict[str, Any], field: str,
) -> bytes:
    selector = _resume_selector_text(_resume_row_identity(row))
    text = _read_resume_command_file(
        row.get(field), label=f'{field} for {selector}',
    )
    local_contract = str(row.get('command_contract_file') or '').strip()
    if field == 'preflight_command_file':
        if not local_contract:
            raise ValueError(
                f'command_contract_file is missing for {selector}'
            )
        replacements = sorted(
            {local_contract, shlex.quote(local_contract)},
            key=len,
            reverse=True,
        )
        replaced = False
        for token in replacements:
            if token and token in text:
                text = text.replace(
                    token, '__LOCAL_COMMAND_CONTRACT_FILE__',
                )
                replaced = True
        if not replaced:
            raise ValueError(
                f'preflight command is not bound to its local contract for '
                f'{selector}'
            )
    normalized_lines = [
        line.rstrip() for line in text.replace('\r\n', '\n').replace(
            '\r', '\n',
        ).split('\n')
    ]
    while normalized_lines and not normalized_lines[-1]:
        normalized_lines.pop()
    return ('\n'.join(normalized_lines) + '\n').encode('utf-8')


def _validate_resume_measure_command_binding(
    row: dict[str, Any], selector: str,
) -> None:
    command = str(row.get('measure_command') or '').strip()
    if not command:
        raise ValueError(f'measure_command is missing for {selector}')
    try:
        parts = shlex.split(command)
    except ValueError as exc:
        raise ValueError(
            f'measure_command is not valid shell text for {selector}: {exc}'
        ) from exc
    for option, field in (
        ('--command-file', 'command_file'),
        ('--preflight-command-file', 'preflight_command_file'),
    ):
        observed, _index, _inline = _single_command_option_parts(
            parts, option,
        )
        expected = str(row.get(field) or '').strip()
        if (
            not expected
            or _resolved_path(observed) != _resolved_path(expected)
        ):
            raise ValueError(
                f'measure_command {option} is not bound to {field} for '
                f'{selector}'
            )
    observed_attestation, _index, _inline = _single_command_option_parts(
        parts, '--preflight-runtime-attestation-path',
    )
    expected_attestation = str(
        row.get('preflight_runtime_attestation_path_template') or ''
    ).strip()
    if (
        not expected_attestation
        or observed_attestation != expected_attestation
    ):
        raise ValueError(
            'measure_command preflight attestation path is not bound for '
            f'{selector}'
        )


def _validate_resume_remote_commands(
    old_row: dict[str, Any],
    new_row: dict[str, Any],
    identity: tuple[str, str, str, str],
) -> None:
    selector = _resume_selector_text(identity)
    try:
        _validate_resume_measure_command_binding(old_row, selector)
        _validate_resume_measure_command_binding(new_row, selector)
    except ValueError as exc:
        raise ValueError(
            f'resume remote command drift for {selector}: {exc}'
        ) from exc
    for field in _RESUME_REMOTE_PATH_FIELDS:
        old_value = str(old_row.get(field) or '').strip()
        new_value = str(new_row.get(field) or '').strip()
        if not old_value or old_value != new_value:
            raise ValueError(
                f'resume remote command drift for {selector}: {field}'
            )
    for field in _RESUME_REMOTE_COMMAND_FIELDS:
        try:
            old_command = _normalized_resume_remote_command(old_row, field)
            new_command = _normalized_resume_remote_command(new_row, field)
        except ValueError as exc:
            raise ValueError(
                f'resume remote command drift for {selector}: {exc}'
            ) from exc
        if not hashlib.sha256(old_command).digest() == hashlib.sha256(
            new_command
        ).digest():
            raise ValueError(
                f'resume remote command drift for {selector}: {field}'
            )


def _resume_row_identity(row: dict[str, Any]) -> tuple[str, str, str, str]:
    identity = tuple(
        str(row.get(field) or '').strip()
        for field in ('backend', 'model', 'case', 'setup_id')
    )
    if any(not value for value in identity):
        raise ValueError(
            'resume row identity requires backend, model, case, and setup_id'
        )
    return identity  # type: ignore[return-value]


def _parse_resume_selector(value: str) -> tuple[str, str, str, str]:
    parts = tuple(part.strip() for part in str(value or '').split('|'))
    if len(parts) != 4 or any(not part for part in parts):
        raise ValueError(
            '--only-row must be exactly backend|model|case|setup_id'
        )
    return parts  # type: ignore[return-value]


def _resume_selector_text(identity: tuple[str, str, str, str]) -> str:
    return '|'.join(identity)


def _p03_resume_admission(
    row: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Return the normalized P0.3 technical truth table, if authentic."""

    admission = row.get('native_energy_planner_admission')
    if not isinstance(admission, Mapping):
        return None
    if not (
        admission.get('schema')
        == 'onnx-splitpoint/native-energy-planner-admission'
        and admission.get('schema_version') == 1
        and admission.get('selected') is True
        and admission.get('runtime_success') is True
        and admission.get('energy_command_preflight_ok') is True
        and row.get('runtime_success') is True
        and row.get('energy_command_preflight_ok') is True
    ):
        return None
    full_baseline = admission.get('full_baseline')
    split_has_part2 = admission.get('split_has_valid_part2_input')
    if not (
        (full_baseline is True and split_has_part2 is False)
        or (full_baseline is False and split_has_part2 is True)
    ):
        return None
    if (
        row.get('full_baseline') is not full_baseline
        or row.get('split_has_valid_part2_input') is not split_has_part2
    ):
        return None
    return {
        'schema': admission.get('schema'),
        'schema_version': admission.get('schema_version'),
        'selected': True,
        'runtime_success': True,
        'energy_command_preflight_ok': True,
        'full_baseline': full_baseline,
        'split_has_valid_part2_input': split_has_part2,
    }


def _resume_contract_sha256(
    row: dict[str, Any], plan_payload: dict[str, Any],
) -> str:
    p03_admission = _p03_resume_admission(row)
    if p03_admission is not None:
        payload = {
            'row': {
                field: row.get(field)
                for field in _P03_RESUME_ROW_CONTRACT_FIELDS
            },
            'technical_admission': p03_admission,
            'plan': {
                field: plan_payload.get(field)
                for field in _P03_RESUME_PLAN_CONTRACT_FIELDS
            },
        }
        return _canonical_json_sha256(payload)
    payload = {
        'row': {
            field: row.get(field)
            for field in _RESUME_ROW_CONTRACT_FIELDS
        },
        'plan': {
            field: plan_payload.get(field)
            for field in _RESUME_PLAN_CONTRACT_FIELDS
        },
    }
    return _canonical_json_sha256(payload)


def _managed_row_identity(row: Mapping[str, Any]) -> str:
    return "|".join(_canonical_energy_identity(row))


def _managed_journal_contract(
    rows: list[dict[str, Any]], plan_payload: dict[str, Any],
) -> tuple[str, list[str], list[str]]:
    identities = [_managed_row_identity(row) for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("managed energy journal row identities are not unique")
    row_hashes = [
        _resume_contract_sha256(row, plan_payload) for row in rows
    ]
    stable_plan = {
        field: plan_payload.get(field)
        for field in _RESUME_PLAN_CONTRACT_FIELDS
    }
    plan_hash = _checkpoint_json_sha256({
        "schema": "onnx-splitpoint/managed-energy-journal-contract",
        "schema_version": 1,
        "plan": stable_plan,
        "ordered_rows": [
            {"identity": identity, "contract_sha256": contract_hash}
            for identity, contract_hash in zip(identities, row_hashes)
        ],
    })
    return plan_hash, identities, row_hashes


def _managed_energy_invocation_hash(
    namespace: argparse.Namespace,
) -> str:
    """Freeze every execution-affecting CLI value before planning starts."""

    excluded = {
        'resume_checkpoint',
        'resume_existing',
        'list_resume_candidates',
        'only_row',
        'expected_selected_rows',
    }
    options = {
        key: value
        for key, value in sorted(vars(namespace).items())
        if key not in excluded
    }

    def _bound_file(value: Any) -> dict[str, Any]:
        raw = str(value or '').strip()
        if not raw:
            return {'path': '', 'sha256': ''}
        path = Path(raw).expanduser().resolve()
        return {
            'path': str(path),
            'sha256': (
                hashlib.sha256(path.read_bytes()).hexdigest()
                if path.is_file() and not path.is_symlink() else ''
            ),
        }

    return _checkpoint_json_sha256({
        'schema': 'onnx-splitpoint/managed-energy-invocation',
        'schema_version': 1,
        'options': options,
        'summary': _bound_file(namespace.summary),
        'validation_summary': _bound_file(namespace.validation_summary),
        'hardware_registry': _bound_file(namespace.hardware_setups_file),
        'runner_sha256': hashlib.sha256(
            Path(__file__).resolve().read_bytes()
        ).hexdigest(),
    })


def _energy_quality_result_projection(
    row: Mapping[str, Any],
    *,
    measurement_started: bool = False,
    raw_energy_collected: bool = False,
) -> dict[str, Any]:
    """Project verified Quality state without claiming uncollected raw Energy."""

    admission = (
        row.get('energy_quality_admission')
        if isinstance(row.get('energy_quality_admission'), Mapping)
        else {}
    )
    qualified = bool(
        str(admission.get('admission_scope') or '') == 'native_energy'
        and admission.get('central_quality_evidence_verified') is True
        and admission.get('precision_quality_binding_verified') is True
        and admission.get('task_quality_observation_valid') is True
        and admission.get('accuracy_gate_pass') is True
        and admission.get('quality_provenance_complete') is True
        and admission.get('quality_claim_result_verified') is True
        and admission.get('diagnostic_only') is False
        and admission.get('claim_comparable') is True
        and admission.get('energy_claim_eligible') is True
    )
    if not measurement_started:
        return {
            **energy_quality_reason_projection(admission),
            'energy_quality_qualified': False,
            'energy_quality_status': 'energy_not_collected',
            'native_energy_after_technical_error': (
                'not_collected_measurement_not_started'
            ),
        }
    if not raw_energy_collected:
        return {
            **energy_quality_reason_projection(admission),
            'energy_quality_qualified': False,
            'energy_quality_status': (
                'collection_started_raw_energy_unavailable'
            ),
            'native_energy_after_technical_error': (
                'collection_started_raw_energy_unavailable'
            ),
        }
    return {
            **energy_quality_reason_projection(admission),
        'energy_quality_qualified': qualified,
        'energy_quality_status': str(
            'quality_qualified'
            if qualified else 'raw_energy_quality_not_qualified'
        ),
        'native_energy_after_technical_error': str(
            'not_applicable_quality_qualified'
            if qualified else 'collect_raw_quality_unqualified'
        ),
    }


def _raw_energy_collected(result: Mapping[str, Any] | None) -> bool:
    """Return true only when the child bound usable raw/derived Energy data."""

    payload = dict(result or {})
    aggregate = payload.get('energy_aggregate')
    runs = aggregate.get('runs') if isinstance(aggregate, Mapping) else []
    if not isinstance(runs, list):
        return False
    for run in runs:
        if not isinstance(run, Mapping):
            continue
        work_units = run.get('runtime_work_unit_evidence')
        exact_work_units = bool(
            isinstance(work_units, Mapping)
            and work_units.get('exact') is True
            and isinstance(work_units.get('count'), int)
            and not isinstance(work_units.get('count'), bool)
            and work_units.get('count') > 0
        )
        workload_success = bool(
            run.get('workload_command_rc') == 0
            or exact_work_units
            or payload.get('energy_aggregate_verified') is True
        )
        if not workload_success:
            continue
        if any(
            isinstance(run.get(field), (int, float))
            and not isinstance(run.get(field), bool)
            and math.isfinite(float(run[field]))
            and float(run[field]) >= 0.0
            for field in (
                'scientific_primary_energy_j', 'energy_total_j',
                'raw_energy_total_j',
            )
        ):
            return True
        sizes = run.get('parquet_file_sizes')
        if isinstance(sizes, Mapping) and any(
            isinstance(value, int) and not isinstance(value, bool)
            and value >= 2048
            for value in sizes.values()
        ):
            return True
    return False


def _reimport_energy_checkpoint(
    checkpoint: Mapping[str, Any], *,
    measurement_path_mapping: tuple[str | Path, str | Path] | None = None,
) -> dict[str, Any]:
    """Replay only the original bound import, without any device invocation.

    v2.81 terminal checkpoints retain the process result under ``result.run``;
    v2.82 also journals it before import. Missing historical process provenance
    is unknown, never a fabricated rc=0 or a new invocation timestamp.
    """
    if 'checkpoint_sha256' in checkpoint:
        unsealed = dict(checkpoint)
        recorded = unsealed.pop('checkpoint_sha256')
        if not recorded or _checkpoint_json_sha256(unsealed) != recorded:
            raise ValueError('historical_checkpoint_digest_mismatch')
    execution = checkpoint.get('execution')
    if not isinstance(execution, Mapping):
        raise ValueError('historical_execution_provenance_missing')
    row = execution.get('runtime_row')
    if not isinstance(row, Mapping):
        raise ValueError('historical_runtime_row_missing')
    for field in ('expected_runs', 'expected_effective_runs'):
        if type(execution.get(field)) is not int or execution[field] <= 0:
            raise ValueError('historical_expected_repeat_contract_missing_or_invalid:' + field)
    for field in ('expected_setup_id', 'expected_run_id', 'output_dir'):
        if not isinstance(execution.get(field), str) or not execution[field]:
            raise ValueError('historical_expected_identity_missing:' + field)
    command = execution.get('command')
    if not isinstance(command, str) or not command:
        raise ValueError('historical_measurement_command_missing')
    previous = checkpoint.get('result')
    previous = previous if isinstance(previous, Mapping) else {}
    prior_run = previous.get('run')
    prior_run = prior_run if isinstance(prior_run, Mapping) else {}
    process = execution.get('process_result')
    if not isinstance(process, Mapping):
        process = prior_run
    if type(process.get('rc')) is not int:
        raise ValueError('historical_process_result_missing_or_invalid')
    if isinstance(execution.get('process_result'), Mapping) and 'rc' in prior_run:
        if type(prior_run['rc']) is not int or prior_run['rc'] != process['rc']:
            raise ValueError('historical_process_result_conflict')
    if isinstance(previous.get('row'), Mapping):
        prior_row = previous['row']
        for key in ('backend', 'model', 'case', 'setup_id', 'measurement_output_dir',
                    'native_command_contract', 'completed_task_completion_mode'):
            if key in prior_row and prior_row[key] != row.get(key):
                raise ValueError('historical_runtime_row_conflict:' + key)
    argv = execution.get('argv')
    if not isinstance(argv, list) or any(not isinstance(part, str) for part in argv):
        raise ValueError('historical_measurement_argv_missing')
    if shlex.split(command) != argv:
        raise ValueError('historical_measurement_command_argv_conflict')
    if 'cmd' in process and process['cmd'] != argv:
        raise ValueError('historical_process_command_conflict')
    # The start and prior absence facts are forwarded exactly as stored. The
    # full importer remains responsible for all freshness and binding gates.
    recovered = {
        'cmd': list(argv), 'rc': process['rc'],
        'elapsed_s': process.get('elapsed_s'),
        'recovered_after_parent_import_gap': True,
        'execution_attempt_id': str(execution.get('execution_attempt_id') or ''),
        'historical_process_result_source': ('execution.process_result'
            if isinstance(execution.get('process_result'), Mapping) else 'result.run'),
    }
    if (process.get('cancelled') is True or process.get('timed_out') is True
        or checkpoint.get('state') == 'cancelled'):
        # A cancellation cannot be made successful by a complete stale file.
        raise ValueError('historical_execution_cancelled_or_timed_out')
    recovered = _attach_energy_aggregate(
        recovered, command, execution.get('output_dir'),
        expected_runs=execution.get('expected_runs'),
        expected_effective_runs=execution.get('expected_effective_runs'),
        expected_run_id=execution.get('expected_run_id'),
        expected_setup_id=execution.get('expected_setup_id'),
        expected_command_contract_sha256=execution.get('expected_command_contract_sha256'),
        execution_started_ns=execution.get('execution_started_ns'),
        expected_native_row=row,
        aggregate_absent_before_execution=execution.get('aggregate_absent_before_execution'),
        measurement_path_mapping=measurement_path_mapping,
        archived_aggregate_provenance=({
            'mtime_ns': prior_run.get('energy_aggregate_mtime_ns'),
            'sha256': prior_run.get('energy_aggregate_sha256'),
        } if measurement_path_mapping is not None else None),
    )
    recorded_aggregate_sha = prior_run.get('energy_aggregate_sha256')
    if (recorded_aggregate_sha is not None
        and recovered.get('energy_aggregate_sha256') is not None
        and recorded_aggregate_sha != recovered['energy_aggregate_sha256']):
        recovered.update({
            'energy_aggregate_verified': False, 'energy_aggregate_complete': False,
            'energy_aggregate_import_status': 'rejected_fail_closed',
            'energy_aggregate_validation_errors': [
                *recovered.get('energy_aggregate_validation_errors', []),
                'historical_bound_aggregate_hash_mismatch',
            ],
        })
    # This is a derived Screening/1-s import, not a new acquisition or claim.
    recovered.update(_NONCLAIMABLE_SCREENING_FIELDS)
    return {
        'row': dict(row), 'run': recovered,
        'ok': recovered.get('energy_aggregate_verified') is True,
        'original_import_decision': {
            key: prior_run.get(key) for key in (
                'energy_aggregate_import_status', 'energy_aggregate_verified',
                'energy_aggregate_validation_errors', 'energy_aggregate_completion_errors',
            ) if key in prior_run
        },
        'derived_read_only_reimport': True,
        'new_measurement_started': False,
        **_NONCLAIMABLE_SCREENING_FIELDS,
    }


def _recover_managed_running_result(
    checkpoint: Mapping[str, Any],
) -> dict[str, Any] | None:
    execution = checkpoint.get('execution')
    if not isinstance(execution, Mapping):
        return None
    output_dir = Path(str(execution.get('output_dir') or ''))
    if not output_dir.is_dir() or not (output_dir / 'energy_aggregate.json').is_file():
        return None
    runtime_row = execution.get('runtime_row')
    if not isinstance(runtime_row, Mapping):
        return None
    command = str(execution.get('command') or '')
    if not command:
        return None
    try:
        recovered = _reimport_energy_checkpoint(checkpoint)['run']
    except ValueError:
        return None
    if recovered.get('energy_aggregate_verified') is not True:
        return None
    observation = _measurement_start_observation(
        output_dir,
        aggregate_verified=(
            recovered.get('energy_aggregate_verified') is True
        ),
    )
    recovered['measurement_start_observation'] = observation
    quality_projection = _energy_quality_result_projection(
        runtime_row,
        measurement_started=(
            observation.get('measurement_started') is True
        ),
        raw_energy_collected=_raw_energy_collected(recovered),
    )
    recovered.update(quality_projection)
    return {
        'row': dict(runtime_row),
        'ok': bool(
            recovered.get('rc') == 0
            and recovered.get('energy_aggregate_verified') is True
        ),
        'run': recovered,
        **quality_projection,
        'recovered_after_parent_import_gap': True,
    }


def _energy_result_reusable(item: Any) -> bool:
    if not isinstance(item, dict) or item.get('ok') is not True:
        return False
    run = item.get('run')
    if not isinstance(run, dict):
        return False
    aggregate = run.get('energy_aggregate')
    return bool(
        run.get('rc') == 0
        and run.get('energy_aggregate_embedded') is True
        and run.get('energy_aggregate_verified') is True
        and isinstance(aggregate, dict)
        and aggregate.get('ok') is True
        and aggregate.get('repeat_contract_complete') is True
    )


def _index_resume_rows(
    rows: list[Any], *, label: str, result_rows: bool,
) -> tuple[
    dict[tuple[str, str, str, str], dict[str, Any]],
    dict[tuple[str, str, str, str], int],
]:
    indexed: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    positions: dict[tuple[str, str, str, str], int] = {}
    for index, item in enumerate(rows):
        if not isinstance(item, dict):
            raise ValueError(f'{label} row {index} is not an object')
        row = item.get('row') if result_rows else item
        if not isinstance(row, dict):
            raise ValueError(f'{label} row {index} has no row object')
        identity = _resume_row_identity(row)
        if identity in indexed:
            raise ValueError(
                f'{label} contains duplicate resume identity '
                f'{_resume_selector_text(identity)}'
            )
        indexed[identity] = item
        positions[identity] = index
    return indexed, positions


def _resume_candidates(existing_report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = existing_report.get('rows')
    if not isinstance(rows, list):
        raise ValueError('existing energy result has no rows list')
    if not rows:
        raise ValueError(
            'existing energy result is empty; it is neither reusable nor an '
            'explicit row-level resume source'
        )
    candidates: list[dict[str, Any]] = []
    indexed, _positions = _index_resume_rows(
        rows, label='existing result', result_rows=True,
    )
    for identity, item in indexed.items():
        if _energy_result_reusable(item):
            continue
        run = item.get('run') if isinstance(item.get('run'), dict) else {}
        candidates.append({
            'selector': _resume_selector_text(identity),
            'backend': identity[0],
            'model': identity[1],
            'case': identity[2],
            'setup_id': identity[3],
            'previous_ok': item.get('ok') is True,
            'previous_rc': run.get('rc'),
            'energy_aggregate_bound': run.get('energy_aggregate_bound') is True,
            'energy_aggregate_complete': run.get('energy_aggregate_complete') is True,
            'energy_aggregate_verified': run.get('energy_aggregate_verified') is True,
        })
    return candidates


def _validate_resume_selection(
    existing_report: dict[str, Any],
    new_plan_payload: dict[str, Any],
    selectors: list[tuple[str, str, str, str]],
) -> dict[str, Any]:
    existing_rows = existing_report.get('rows')
    new_rows = new_plan_payload.get('rows')
    old_plan = existing_report.get('plan_payload')
    if not isinstance(existing_rows, list):
        raise ValueError('existing energy result has no rows list')
    if not isinstance(new_rows, list):
        raise ValueError('fresh resume plan has no rows list')
    if not isinstance(old_plan, dict):
        raise ValueError('existing energy result has no frozen plan_payload')
    old_execution_context = _canonical_resume_execution_context(
        existing_report,
    )
    new_execution_context = _validated_resume_execution_context(
        new_plan_payload.get('execution_context'),
        label='fresh resume plan',
    )
    context_equal, context_field = _execution_contexts_equal(
        old_execution_context, new_execution_context,
    )
    if not context_equal:
        raise ValueError(
            f'resume execution context drift in fresh plan: {context_field}'
        )
    old_index, old_positions = _index_resume_rows(
        existing_rows, label='existing result', result_rows=True,
    )
    new_index, _new_positions = _index_resume_rows(
        new_rows, label='fresh plan', result_rows=False,
    )
    if set(old_index) != set(new_index):
        missing = sorted(_resume_selector_text(key) for key in set(old_index) - set(new_index))
        added = sorted(_resume_selector_text(key) for key in set(new_index) - set(old_index))
        raise ValueError(
            f'resume plan row set drift: missing={missing} added={added}'
        )
    for identity in sorted(old_index):
        old_row = old_index[identity].get('row')
        new_row = new_index[identity]
        assert isinstance(old_row, dict)
        old_sha = _resume_contract_sha256(old_row, old_plan)
        new_sha = _resume_contract_sha256(new_row, new_plan_payload)
        if old_sha != new_sha:
            raise ValueError(
                'resume contract drift for '
                f'{_resume_selector_text(identity)}: {old_sha} != {new_sha}'
            )
        _validate_resume_remote_commands(old_row, new_row, identity)
    if len(set(selectors)) != len(selectors):
        raise ValueError('duplicate --only-row selectors are not allowed')
    selected_rows: list[dict[str, Any]] = []
    selected_positions: dict[tuple[str, str, str, str], int] = {}
    for identity in selectors:
        if identity not in new_index:
            raise ValueError(
                f'--only-row did not match the fresh plan: '
                f'{_resume_selector_text(identity)}'
            )
        if _energy_result_reusable(old_index[identity]):
            raise ValueError(
                f'--only-row already has a complete verified result: '
                f'{_resume_selector_text(identity)}'
            )
        selected_rows.append(new_index[identity])
        selected_positions[identity] = old_positions[identity]
    selector_set = set(selectors)
    unselected_incomplete = [
        _resume_selector_text(identity)
        for identity, item in old_index.items()
        if identity not in selector_set and not _energy_result_reusable(item)
    ]
    if unselected_incomplete:
        raise ValueError(
            'every incomplete existing row must be selected explicitly; '
            f'unselected={sorted(unselected_incomplete)}'
        )
    return {
        'rows': selected_rows,
        'positions': selected_positions,
        'existing_rows': list(existing_rows),
        'old_plan_payload': old_plan,
    }


def _atomic_write_text(path: Path, text: str) -> None:
    _checkpoint_atomic_write_text(path, text)


def _atomic_write_json(path: Path, value: Any) -> None:
    _checkpoint_atomic_write_json(path, value)


def _archive_resume_state(
    out: Path, *, resume_attempt_id: str,
) -> tuple[Path, str]:
    history = out / 'resume_history' / f'resume_{resume_attempt_id}'
    history.mkdir(parents=True, exist_ok=False)
    manifest: dict[str, Any] = {
        'schema': 'onnx-splitpoint/native-energy-resume-history',
        'schema_version': 1,
        'resume_attempt_id': resume_attempt_id,
        'files': [],
    }
    candidates = (
        out / 'native_producer_energy_results.json',
        out / 'native_producer_energy_results.md',
        out / 'plan' / 'native_producer_energy_plan.json',
        out / 'plan' / 'native_producer_energy_plan.md',
    )
    prior_result_sha = ''
    for source in candidates:
        if not source.exists():
            continue
        if source.is_symlink() or not source.is_file():
            raise ValueError(f'resume archive source is not a regular file: {source}')
        raw = source.read_bytes()
        relative = source.relative_to(out)
        target = history / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        digest = hashlib.sha256(raw).hexdigest()
        if source.name == 'native_producer_energy_results.json':
            prior_result_sha = digest
        manifest['files'].append({
            'source': str(relative),
            'archived': str(target.relative_to(out)),
            'size': len(raw),
            'sha256': digest,
        })
    if not prior_result_sha:
        raise ValueError('existing canonical energy result could not be archived')
    _atomic_write_json(history / 'resume_history_manifest.json', manifest)
    return history, prior_result_sha


def _measurement_preflight_allows(
    plan_payload: Mapping[str, Any],
) -> bool:
    """Gate collectors on replay safety, never on scientific coverage."""

    preflight = (
        dict(plan_payload.get('preflight') or {})
        if isinstance(plan_payload, Mapping) else {}
    )
    return bool(
        str(plan_payload.get('preflight_status') or '') == 'passed'
        and str(preflight.get('status') or '') == 'passed'
        and preflight.get('ok') is True
        and preflight.get('measurement_start_allowed') is True
        and preflight.get(
            'technical_measurement_contract_valid'
        ) is True
    )


def _main_impl() -> int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--summary', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument(
        '--hardware-setups-file', default='',
        help='Exact hardware registry propagated to every claim-bearing measurement.',
    )
    ap.add_argument('--validation-summary', default='', help='Native semantic validation summary archived as a post-hoc annotation.')
    ap.add_argument('--hailo8-ssh', default='')
    ap.add_argument('--hailo10-ssh', default='')
    ap.add_argument('--deepx-ssh', default='')
    ap.add_argument('--hailo8-env', default='')
    ap.add_argument('--hailo10-env', default='')
    ap.add_argument('--deepx-env', default='')
    ap.add_argument('--engine-build-python', default='auto')
    ap.add_argument('--remote-tool-dir', default='/home/nx/ONNX-Splitpoint-Tool')
    ap.add_argument('--remote-root', default='/home/nx/native_fifo_evalsets')
    ap.add_argument('--duration-s', type=float, default=0.0, help='Time-based workload duration. If 0, ToolConfig native_energy_duration_s is used.')
    ap.add_argument('--frames', type=int, default=0, help='Legacy override. Normally derived from FPS*duration_s by the plan script.')
    ap.add_argument('--warmup', type=int, default=0, help='Legacy command warmup. Native Energy defaults to 0.')
    ap.add_argument('--timeout', type=int, default=900)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--runs', type=int, default=1, help='Independent u.RECS repetitions per energy row.')
    ap.add_argument('--require-runtime-work-units', action='store_true')
    ap.add_argument('--require-command-window-alignment', action='store_true')
    ap.add_argument('--physical-scope', default='')
    ap.add_argument('--window-label', default='command')
    ap.add_argument('--window-method-ab-json', default='', help='Frozen top-level energy.window_method_ab object propagated to every measurement row.')
    ap.add_argument('--calibration-manifest', default='')
    ap.add_argument('--calibration-sha256', default='')
    ap.add_argument('--pipeline-contract-manifest', default='')
    ap.add_argument('--pipeline-contract-sha256', default='')
    ap.add_argument('--model-hash-map', default='')
    ap.add_argument('--model-hash-map-sha256', default='')
    ap.add_argument('--allow-unpaired', action='store_true', help='Diagnostic mode: measure split/full rows even when a complete setup-local pair is unavailable.')
    ap.add_argument(
        '--measure-all-runtime-successful',
        action='store_true',
        help=(
            'Compatibility alias for the invariant Native Energy admission: '
            'all runtime-successful, technically constructible rows.'
        ),
    )
    ap.add_argument('--smoke-diagnostic', action='store_true', help='Smoke-only diagnostic admission with all claim eligibility forced false.')
    ap.add_argument(
        '--screening-energy', action='store_true',
        help=(
            'Development/Screening measurement tier for technically admitted '
            'rows; Quality/Semantics/pairing are post-hoc and every result is '
            'forced non-claimable.'
        ),
    )
    ap.add_argument(
        '--final-all-split-energy', action='store_true',
        help=(
            'Measure every technically verified Split observation; exactly '
            'bound negative accuracy observations remain non-claimable.'
        ),
    )
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument(
        '--resume-existing', action='store_true',
        help=(
            'Resume only explicitly selected incomplete rows from an existing '
            'native_producer_energy_results.json and atomically merge them.'
        ),
    )
    ap.add_argument(
        '--resume-checkpoint', action='store_true',
        help=(
            'Same-EvaluationRun resume from the exact atomic row journal. '
            'Completed/failed rows remain terminal; cancelled/not-started '
            'rows continue in frozen plan order.'
        ),
    )
    ap.add_argument(
        '--only-row', action='append', default=[],
        metavar='BACKEND|MODEL|CASE|SETUP_ID',
        help='Exact incomplete row to rerun; repeat for every intended row.',
    )
    ap.add_argument(
        '--expected-selected-rows', type=int, default=0,
        help='Fail closed unless exactly this many unique rows are selected.',
    )
    ap.add_argument(
        '--list-resume-candidates', action='store_true',
        help='Read-only listing of incomplete exact row selectors.',
    )
    ns=ap.parse_args()
    if str(ns.hardware_setups_file or '').strip():
        registry_input = Path(ns.hardware_setups_file).expanduser()
        if registry_input.is_symlink():
            ap.error('--hardware-setups-file must not be a symlink')
        try:
            registry_path = registry_input.resolve(strict=True)
        except OSError as exc:
            ap.error(f'--hardware-setups-file is not readable: {exc}')
        if not registry_path.is_file():
            ap.error('--hardware-setups-file must be a regular file')
        ns.hardware_setups_file = str(registry_path)
    if ns.screening_energy and ns.smoke_diagnostic:
        ap.error('--screening-energy and --smoke-diagnostic are mutually exclusive')
    if ns.final_all_split_energy and (
        ns.screening_energy or ns.smoke_diagnostic
    ):
        ap.error(
            '--final-all-split-energy cannot be combined with a diagnostic '
            'or screening tier'
        )
    if ns.screening_energy and (
        ns.require_runtime_work_units
        or ns.require_command_window_alignment
    ):
        ap.error(
            '--screening-energy cannot be combined with Final-only runtime '
            'work-unit or command-window requirements'
        )
    # Normalize before hashing or checkpoint recovery: the compatibility flag
    # may not change the managed invocation contract in P0.3.
    ns.measure_all_runtime_successful = True
    try:
        resume_selectors = [
            _parse_resume_selector(value) for value in ns.only_row
        ]
    except ValueError as exc:
        ap.error(str(exc))
    if ns.list_resume_candidates and ns.resume_existing:
        ap.error('--list-resume-candidates and --resume-existing are exclusive')
    if ns.resume_checkpoint and (ns.resume_existing or ns.list_resume_candidates):
        ap.error(
            '--resume-checkpoint is exclusive with historical result resume modes'
        )
    if ns.list_resume_candidates and resume_selectors:
        ap.error('--list-resume-candidates does not accept --only-row')
    if resume_selectors and not ns.resume_existing:
        ap.error('--only-row requires --resume-existing')
    if ns.expected_selected_rows and not ns.resume_existing:
        ap.error('--expected-selected-rows requires --resume-existing')
    if ns.resume_existing:
        if not resume_selectors:
            ap.error('--resume-existing requires at least one --only-row')
        if ns.expected_selected_rows <= 0:
            ap.error('--resume-existing requires --expected-selected-rows > 0')
        if ns.expected_selected_rows != len(resume_selectors):
            ap.error(
                '--expected-selected-rows must equal the number of --only-row '
                'arguments'
            )
        if len(set(resume_selectors)) != len(resume_selectors):
            ap.error('duplicate --only-row selectors are not allowed')
        if ns.dry_run:
            ap.error(
                '--resume-existing cannot be combined with --dry-run; use '
                '--list-resume-candidates for a read-only preview'
            )
        if ns.limit:
            ap.error('--resume-existing cannot be combined with --limit')
    if ns.resume_checkpoint and ns.dry_run:
        ap.error('--resume-checkpoint cannot be combined with --dry-run')
    smoke_claim_clamp = {
        'smoke_diagnostic': True,
        'diagnostic_only': True,
        'claim_ok': False,
        'semantic_claim_ok': False,
        'claim_eligible': False,
        'eligible_for_energy_results_import': False,
        'eligible_for_scientific_claim': False,
        'energy_claim_eligible': False,
    } if ns.smoke_diagnostic else {}
    screening_claim_clamp = (
        dict(_NONCLAIMABLE_SCREENING_FIELDS)
        if ns.screening_energy else {}
    )
    artifact_claim_clamp = {
        **smoke_claim_clamp,
        **screening_claim_clamp,
    }
    out=Path(ns.out_dir).expanduser().resolve()
    canonical_result_path = out/'native_producer_energy_results.json'
    managed_journal_root = out/'checkpoints'/'native_energy'
    managed_stage_checkpoint_path = out/'stages'/'native_energy'/'stage_result.json'
    managed_invocation_hash = _managed_energy_invocation_hash(ns)
    managed_journal: AtomicRowJournal | None = None
    managed_plan_hash = ''
    managed_identities: list[str] = []
    managed_row_hashes: list[str] = []
    global _ACTIVE_MANAGED_STAGE_CONTEXT
    _ACTIVE_MANAGED_STAGE_CONTEXT = (
        {
            'out': out,
            'checkpoint': managed_stage_checkpoint_path,
            'journal': managed_journal_root,
            'input_hash': managed_invocation_hash,
            'resume_checkpoint': bool(ns.resume_checkpoint),
        }
        if not ns.resume_existing and not ns.list_resume_candidates else {}
    )

    def _terminal_managed_failure(
        *, status: str, return_code: int, artifacts: list[Path] | None = None,
    ) -> None:
        if not _ACTIVE_MANAGED_STAGE_CONTEXT:
            return
        if managed_journal is not None:
            managed_journal.mark_run_state('failed')
        durable_artifacts = [
            path for path in list(artifacts or [])
            if path.is_file() and not path.is_symlink()
        ]
        journal_manifest = managed_journal_root / 'journal.json'
        if journal_manifest.is_file() and not journal_manifest.is_symlink():
            durable_artifacts.append(journal_manifest)
        _write_stage_checkpoint(
            managed_stage_checkpoint_path,
            stage='native_energy',
            state='failed',
            complete=True,
            input_hash=managed_invocation_hash,
            run_root=out,
            artifacts=durable_artifacts,
            details={
                'status': str(status or 'native_energy_failed'),
                'return_code': int(return_code),
                'resume_checkpoint': bool(ns.resume_checkpoint),
            },
            error=str(status or 'native_energy_failed'),
        )
    if (
        (ns.resume_existing or ns.list_resume_candidates)
        and (
            managed_journal_root.exists()
            or managed_stage_checkpoint_path.exists()
        )
    ):
        ap.error(
            'historical result resume cannot be mixed with the managed '
            'same-run Energy journal'
        )
    if ns.resume_checkpoint:
        if not managed_stage_checkpoint_path.is_file():
            ap.error(
                'managed Energy stage checkpoint is missing; this run cannot '
                'use same-run Resume'
            )
        managed_stage, managed_stage_reason = _load_stage_checkpoint(
            managed_stage_checkpoint_path,
            stage='native_energy',
            input_hash=managed_invocation_hash,
            run_root=out,
        )
        if managed_stage is None:
            ap.error(
                'managed Energy invocation/checkpoint drift: '
                f'{managed_stage_reason}'
            )
        if (
            str(managed_stage.get('state') or '') in {'completed', 'failed'}
            and managed_stage.get('complete') is True
        ):
            return_code = int(
                (managed_stage.get('details') or {}).get('return_code')
                or (
                    0 if managed_stage.get('state') == 'completed' else 2
                )
            )
            try:
                print(json.dumps({
                    'ok': str(managed_stage.get('state')) == 'completed',
                    'state': managed_stage.get('state'),
                    'complete': True,
                    'resume_checkpoint': True,
                    'terminal_stage_reused': True,
                    'json': str(canonical_result_path),
                }, indent=2), flush=True)
            except BrokenPipeError:
                return 130
            return return_code
    existing_report: dict[str, Any] | None = None
    existing_result_raw = b''
    canonical_execution_context: dict[str, str] | None = None
    if ns.list_resume_candidates or ns.resume_existing:
        if (
            canonical_result_path.is_symlink()
            or not canonical_result_path.is_file()
        ):
            ap.error(
                'existing canonical energy result is missing or not a regular '
                f'file: {canonical_result_path}'
            )
        existing_result_raw = canonical_result_path.read_bytes()
        try:
            parsed_existing = _strict_json_text(
                existing_result_raw.decode('utf-8'),
                label=str(canonical_result_path),
            )
        except Exception as exc:
            ap.error(f'existing canonical energy result is invalid: {exc}')
        if not isinstance(parsed_existing, dict):
            ap.error('existing canonical energy result is not an object')
        existing_report = dict(parsed_existing)
    if ns.resume_existing:
        assert existing_report is not None
        try:
            canonical_execution_context = (
                _canonical_resume_execution_context(existing_report)
            )
            _bind_resume_execution_context(
                ns, list(sys.argv[1:]), canonical_execution_context,
            )
        except ValueError as exc:
            ap.error(str(exc))
    if ns.list_resume_candidates:
        try:
            candidates = _resume_candidates(existing_report or {})
        except ValueError as exc:
            ap.error(str(exc))
        print(json.dumps({
            'ok': True,
            'read_only': True,
            'candidate_count': len(candidates),
            'candidates': candidates,
        }, indent=2))
        return 0

    out.mkdir(parents=True, exist_ok=True)
    resume_attempt_id = uuid.uuid4().hex if ns.resume_existing else ''
    resume_attempt_root = (
        out/'resume_attempts'/f'resume_{resume_attempt_id}'
        if ns.resume_existing else None
    )
    plan_dir = (
        resume_attempt_root/'plan'
        if resume_attempt_root is not None else out/'plan'
    )
    if not ns.resume_existing and not ns.resume_checkpoint:
        if (
            managed_journal_root.exists()
            or managed_stage_checkpoint_path.exists()
        ):
            ap.error(
                'an atomic Native Energy journal already exists; use the '
                'Evaluation Workflow resume path or a fresh run id'
            )
        # A restarted stage must never expose a terminal result or progress file
        # from its predecessor while the new execution is still running.
        for stale_path in (
            canonical_result_path,
            out/'native_producer_energy_results.md',
            out/'native_producer_energy_results.partial.json',
            out/'plan'/'native_producer_energy_plan.json',
            out/'plan'/'native_producer_energy_plan.md',
        ):
            try:
                stale_path.unlink()
            except FileNotFoundError:
                pass
    if not ns.resume_existing:
        _write_stage_checkpoint(
            managed_stage_checkpoint_path,
            stage='native_energy',
            state='running',
            complete=False,
            input_hash=managed_invocation_hash,
            run_root=out,
            details={
                'resume_checkpoint': bool(ns.resume_checkpoint),
                'phase': 'planning',
            },
        )
    plan_cmd=[sys.executable, '-u', str(_script_path('native_producer_energy_plan.py')), '--summary', ns.summary, '--out-dir', str(plan_dir), '--hailo8-ssh', ns.hailo8_ssh, '--hailo10-ssh', ns.hailo10_ssh, '--deepx-ssh', ns.deepx_ssh, '--hailo8-env', ns.hailo8_env, '--hailo10-env', ns.hailo10_env, '--deepx-env', ns.deepx_env, '--engine-build-python', ns.engine_build_python, '--remote-tool-dir', ns.remote_tool_dir, '--remote-root', ns.remote_root, '--duration-s', str(float(ns.duration_s or 0.0)), '--frames', str(ns.frames), '--warmup', str(ns.warmup), '--timeout', str(ns.timeout), '--runs', str(max(1, int(ns.runs or 1))), '--physical-scope', ns.physical_scope, '--window-label', ns.window_label, '--calibration-manifest', ns.calibration_manifest, '--calibration-sha256', ns.calibration_sha256, '--pipeline-contract-manifest', ns.pipeline_contract_manifest, '--pipeline-contract-sha256', ns.pipeline_contract_sha256, '--model-hash-map', ns.model_hash_map, '--model-hash-map-sha256', ns.model_hash_map_sha256]
    if str(ns.hardware_setups_file or '').strip():
        plan_cmd += ['--hardware-setups-file', str(ns.hardware_setups_file)]
    if str(ns.window_method_ab_json or '').strip():
        plan_cmd += ['--window-method-ab-json', str(ns.window_method_ab_json)]
    if ns.require_runtime_work_units:
        plan_cmd.append('--require-runtime-work-units')
    if ns.require_command_window_alignment:
        plan_cmd.append('--require-command-window-alignment')
    if ns.allow_unpaired:
        plan_cmd.append('--allow-unpaired')
    if ns.measure_all_runtime_successful:
        plan_cmd.append('--measure-all-runtime-successful')
    if ns.smoke_diagnostic:
        plan_cmd.append('--smoke-diagnostic')
    if ns.screening_energy:
        plan_cmd.append('--screening-energy')
    if ns.final_all_split_energy:
        plan_cmd.append('--final-all-split-energy')
    if ns.validation_summary:
        plan_cmd += ['--validation-summary', ns.validation_summary]
    print(f"[native-energy] PLAN_START summary={ns.summary}", flush=True)
    pr=_run(plan_cmd, timeout=120, label="energy_plan")
    plan_payload=_load_json(plan_dir/'native_producer_energy_plan.json') or {}
    if ns.resume_existing:
        assert canonical_execution_context is not None
        plan_payload = dict(plan_payload)
        plan_payload['execution_context'] = {
            'schema': _RESUME_EXECUTION_CONTEXT_SCHEMA,
            'schema_version': 1,
            **canonical_execution_context,
        }
    rows=plan_payload.get('rows', []) if isinstance(plan_payload, dict) else []
    print(f"[native-energy] PLAN_END rc={pr.get('rc')} rows={len(rows)}", flush=True)
    if pr.get('rc') != 0:
        report={'ok':False,'complete':False,'status':'native_energy_plan_failed','plan':pr,'rows':[], **artifact_claim_clamp,
                'preflight_status':'native_energy_plan_failed',
                'preflight':dict(plan_payload.get('preflight') or {}),
                'error':'native_energy_plan_failed'}
        failure_path = (
            resume_attempt_root/'resume_failure.json'
            if resume_attempt_root is not None
            else canonical_result_path
        )
        _atomic_write_json(failure_path, report)
        _terminal_managed_failure(
            status='native_energy_plan_failed',
            return_code=2,
            artifacts=[failure_path],
        )
        print(json.dumps({'ok':False,'rows':0,'error':'native_energy_plan_failed'}, indent=2), flush=True)
        return 2
    if not ns.resume_existing and rows:
        managed_rows = [dict(row) for row in rows]
        (
            managed_plan_hash,
            managed_identities,
            managed_row_hashes,
        ) = _managed_journal_contract(managed_rows, plan_payload)
        # Publish the exact journal binding before opening an interrupted
        # journal.  A signal may arrive while Resume is recovering a previously
        # running row; the outer cancellation boundary then has enough immutable
        # context to reload the authoritative row files and cancel only that row.
        _ACTIVE_MANAGED_STAGE_CONTEXT.update({
            'managed_plan_hash': managed_plan_hash,
            'managed_identities': list(managed_identities),
            'managed_row_hashes': list(managed_row_hashes),
        })
        if (
            ns.resume_checkpoint
            and (managed_journal_root / 'journal.json').is_file()
        ):
            managed_journal = AtomicRowJournal.open_for_resume(
                managed_journal_root,
                plan_hash=managed_plan_hash,
                identities=managed_identities,
                row_contract_hashes=managed_row_hashes,
            )
        else:
            managed_journal = AtomicRowJournal.create(
                managed_journal_root,
                plan_hash=managed_plan_hash,
                rows=managed_rows,
                identities=managed_identities,
                row_contract_hashes=managed_row_hashes,
            )
    preflight = (
        dict(plan_payload.get('preflight') or {})
        if isinstance(plan_payload, dict) else {}
    )
    preflight_allows_measurement = _measurement_preflight_allows(
        plan_payload
    )
    if rows and not preflight_allows_measurement:
        blocked_reason = str(
            preflight.get('blocked_reason')
            or preflight.get('status')
            or plan_payload.get('preflight_status')
            or 'energy_plan_preflight_not_passed'
        )
        blocked_rows = [
            {
                'row': dict(row),
                'ok': False,
                'skipped': 'blocked_before_measurement',
                'blocked_reason': blocked_reason,
                'measurement_started': False,
            }
            for row in rows if isinstance(row, dict)
        ]
        result_ledger = [
            {
                'backend': str(row.get('backend') or ''),
                'model': str(row.get('model') or ''),
                'case': str(row.get('case') or ''),
                'setup_id': str(row.get('setup_id') or ''),
                'comparison_backend': str(
                    row.get('comparison_backend') or ''
                ),
                'precision': str(row.get('precision') or ''),
                'status': 'blocked_before_measurement',
                'reason': blocked_reason,
                'measurement_started': False,
            }
            for row in rows if isinstance(row, dict)
        ]
        report = {
            'ok': False,
            'complete': False,
            'status': 'blocked_energy_plan_preflight',
            'blocked_reason': blocked_reason,
            'requested': True,
            'dry_run': bool(ns.dry_run),
            'plan': pr,
            'plan_payload': plan_payload,
            'preflight_status': str(
                plan_payload.get('preflight_status') or ''
            ),
            'preflight': preflight,
            'rows': blocked_rows,
            'result_ledger': result_ledger,
            'result_ledger_valid': _result_ledger_valid(
                result_ledger, rows,
            ),
            'measurement_wrapper_started_count': 0,
            'started_measurement_count': 0,
            'collector_started_repeat_count': 0,
            'workload_started_repeat_count': 0,
            **artifact_claim_clamp,
        }
        result_path = (
            resume_attempt_root/'resume_failure.json'
            if resume_attempt_root is not None
            else canonical_result_path
        )
        markdown_path = (
            resume_attempt_root/'resume_failure.md'
            if resume_attempt_root is not None
            else out/'native_producer_energy_results.md'
        )
        _atomic_write_json(result_path, report)
        _atomic_write_text(
            markdown_path,
            '# Native producer energy run\n\n'
            'The sealed plan preflight blocked the complete cohort; '
            'no measurement wrapper, collector, or workload was started.\n',
        )
        _terminal_managed_failure(
            status='blocked_energy_plan_preflight',
            return_code=4,
            artifacts=[result_path, markdown_path],
        )
        print(json.dumps({
            'ok': False,
            'complete': False,
            'rows': len(rows),
            'status': report['status'],
            'started_measurement_count': 0,
            'collector_started_repeat_count': 0,
            'workload_started_repeat_count': 0,
        }, indent=2), flush=True)
        return 4
    if not rows:
        empty_plan_status = str(
            plan_payload.get('preflight_status')
            or (
                'blocked_no_runtime_constructible_rows'
                if ns.measure_all_runtime_successful
                else 'blocked_no_complete_pairs'
            )
        )
        empty_blocked_reason = (
            'no_runtime_constructible_native_rows'
            if empty_plan_status
            == 'blocked_no_runtime_constructible_rows'
            else 'no_complete_setup_local_native_energy_pairs'
        )
        report={
            'ok': False, 'complete': False, 'status': empty_plan_status,
            'blocked_reason': empty_blocked_reason,
            'requested': True, 'started_measurement_count': 0,
            'dry_run': bool(ns.dry_run), 'plan': pr, 'plan_payload': plan_payload, 'rows': [],
            'preflight_status': empty_plan_status,
            'preflight': dict(plan_payload.get('preflight') or {}),
            **artifact_claim_clamp,
        }
        result_path = (
            resume_attempt_root/'resume_failure.json'
            if resume_attempt_root is not None
            else canonical_result_path
        )
        markdown_path = (
            resume_attempt_root/'resume_failure.md'
            if resume_attempt_root is not None
            else out/'native_producer_energy_results.md'
        )
        _atomic_write_json(result_path, report)
        _atomic_write_text(
            markdown_path,
            '# Native producer energy run\n\n'
            + (
                'No runtime-successful, technically constructible Native row '
                'was available; no physical measurement was started.\n'
                if empty_plan_status
                == 'blocked_no_runtime_constructible_rows'
                else
                'No complete setup-local Split + Vendor Full + TensorRT Full '
                'pair was available; no physical measurement was started.\n'
            ),
        )
        _terminal_managed_failure(
            status=empty_plan_status,
            return_code=3,
            artifacts=[result_path, markdown_path],
        )
        print(json.dumps({
            'ok': False,
            'complete': False,
            'rows': 0,
            'status': empty_plan_status,
        }, indent=2), flush=True)
        return 3

    cohort_validation_errors: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        try:
            _prepare_measurement_execution(
                row,
                plan_payload,
                allowed_root=out,
                validate_only=True,
            )
        except Exception as exc:
            cohort_validation_errors.append({
                'row_index': row_index,
                'backend': str(row.get('backend') or ''),
                'model': str(row.get('model') or ''),
                'case': str(row.get('case') or ''),
                'error': f'{type(exc).__name__}: {exc}',
            })
    if cohort_validation_errors:
        blocked_rows = [
            {
                'row': dict(row),
                'ok': False,
                'skipped': 'blocked_before_measurement',
                'blocked_reason': (
                    'energy_plan_cohort_contract_invalid'
                ),
                'measurement_started': False,
            }
            for row in rows
        ]
        result_ledger = [
            {
                'backend': str(row.get('backend') or ''),
                'model': str(row.get('model') or ''),
                'case': str(row.get('case') or ''),
                'setup_id': str(row.get('setup_id') or ''),
                'comparison_backend': str(
                    row.get('comparison_backend') or ''
                ),
                'precision': str(row.get('precision') or ''),
                'status': 'blocked_before_measurement',
                'reason': 'energy_plan_cohort_contract_invalid',
                'measurement_started': False,
            }
            for row in rows
        ]
        report = {
            'ok': False,
            'complete': False,
            'status': 'blocked_energy_plan_cohort_invalid',
            'blocked_reason': 'energy_plan_cohort_contract_invalid',
            'requested': True,
            'dry_run': bool(ns.dry_run),
            'plan': pr,
            'plan_payload': plan_payload,
            'preflight_status': str(
                plan_payload.get('preflight_status') or ''
            ),
            'preflight': preflight,
            'cohort_validation_errors': cohort_validation_errors,
            'rows': blocked_rows,
            'result_ledger': result_ledger,
            'result_ledger_valid': _result_ledger_valid(
                result_ledger, rows,
            ),
            'measurement_wrapper_started_count': 0,
            'started_measurement_count': 0,
            'collector_started_repeat_count': 0,
            'workload_started_repeat_count': 0,
            **artifact_claim_clamp,
        }
        result_path = (
            resume_attempt_root/'resume_failure.json'
            if resume_attempt_root is not None
            else canonical_result_path
        )
        markdown_path = (
            resume_attempt_root/'resume_failure.md'
            if resume_attempt_root is not None
            else out/'native_producer_energy_results.md'
        )
        _atomic_write_json(result_path, report)
        _atomic_write_text(
            markdown_path,
            '# Native producer energy run\n\n'
            'The complete plan cohort failed immutable contract validation; '
            'no measurement wrapper, collector, or workload was started.\n',
        )
        _terminal_managed_failure(
            status='blocked_energy_plan_cohort_invalid',
            return_code=5,
            artifacts=[result_path, markdown_path],
        )
        print(json.dumps({
            'ok': False,
            'complete': False,
            'rows': len(rows),
            'status': report['status'],
            'cohort_validation_error_count': len(
                cohort_validation_errors
            ),
            'started_measurement_count': 0,
            'collector_started_repeat_count': 0,
            'workload_started_repeat_count': 0,
        }, indent=2), flush=True)
        return 5

    if not ns.resume_existing:
        if managed_journal is None:
            raise ValueError(
                'managed Energy rows exist without an atomic row journal'
            )
        managed_journal.mark_run_state('running')
        _write_stage_checkpoint(
            managed_stage_checkpoint_path,
            stage='native_energy',
            state='running',
            complete=False,
            input_hash=managed_invocation_hash,
            run_root=out,
            details={
                'planned_row_count': len(managed_rows),
                'managed_plan_hash': managed_plan_hash,
                'resume_checkpoint': bool(ns.resume_checkpoint),
                'phase': 'measuring',
            },
        )
    resume_positions: dict[tuple[str, str, str, str], int] = {}
    resume_history_dir: Path | None = None
    prior_result_sha256 = ''
    if ns.resume_existing:
        assert existing_report is not None
        try:
            resume_state = _validate_resume_selection(
                existing_report, plan_payload, resume_selectors,
            )
            rows = list(resume_state['rows'])
            resume_positions = dict(resume_state['positions'])
            results = list(resume_state['existing_rows'])
            if len(rows) != ns.expected_selected_rows:
                raise ValueError(
                    f'fresh plan selected {len(rows)} rows, expected '
                    f'{ns.expected_selected_rows}'
                )
            assert resume_attempt_root is not None
            resume_history_dir, prior_result_sha256 = _archive_resume_state(
                out, resume_attempt_id=resume_attempt_id,
            )
        except Exception as exc:
            assert resume_attempt_root is not None
            _atomic_write_json(resume_attempt_root/'resume_failure.json', {
                'ok': False,
                'status': 'resume_selection_or_contract_failed',
                'error': f'{type(exc).__name__}: {exc}',
                'selected': [
                    _resume_selector_text(value) for value in resume_selectors
                ],
                'started_measurement_count': 0,
            })
            print(json.dumps({
                'ok': False,
                'status': 'resume_selection_or_contract_failed',
                'error': f'{type(exc).__name__}: {exc}',
                'started_measurement_count': 0,
            }, indent=2), flush=True)
            return 2
    else:
        results=[]
    managed_results_by_index: list[dict[str, Any] | None] = [
        None for _row in rows
    ]
    if managed_journal is not None:
        for checkpoint in managed_journal.rows:
            index = int(checkpoint.get('index', -1))
            if str(checkpoint.get('state') or '') not in {'completed', 'failed'}:
                continue
            result = checkpoint.get('result')
            if index < 0 or index >= len(managed_results_by_index) or not isinstance(result, Mapping):
                raise ValueError('managed energy journal terminal result is invalid')
            managed_results_by_index[index] = dict(result)
        results = [
            item for item in managed_results_by_index if item is not None
        ]

        # A process may stop after the measurement child has durably published
        # its aggregate but before this parent imports the result.  Row files
        # are authoritative: recover that exact aggregate before deciding that
        # the interrupted row needs another attempt.
        interrupted_indexes = [
            int(checkpoint.get('index', -1))
            for checkpoint in managed_journal.rows
            if str(checkpoint.get('state') or '') in {'running', 'cancelled'}
        ]
        if sum(
            1 for checkpoint in managed_journal.rows
            if str(checkpoint.get('state') or '') == 'running'
        ) > 1:
            raise ValueError(
                'managed energy journal contains more than one running row'
            )
        for row_index in interrupted_indexes:
            checkpoint = managed_journal.rows[row_index]
            recovered = _recover_managed_running_result(checkpoint)
            if recovered is not None:
                managed_journal.recover_terminal(
                    row_index,
                    result=recovered,
                    reason='recovered_after_parent_import_gap',
                )
                managed_results_by_index[row_index] = dict(recovered)
            elif str(checkpoint.get('state') or '') == 'running':
                managed_journal.transition(
                    row_index,
                    'cancelled',
                    reason='interrupted_before_terminal_row_commit',
                )
        results = [
            item for item in managed_results_by_index if item is not None
        ]
    resume_preparation: dict[str, Any] | None = None
    if ns.resume_existing:
        assert resume_attempt_root is not None
        assert canonical_execution_context is not None
        resume_preparation = _prepare_resume_measurement_cohort(
            rows,
            attempt_dir=resume_attempt_root,
            plan_root=plan_dir,
            summary_path=ns.summary,
            canonical_execution_context=canonical_execution_context,
            resume_attempt_id=resume_attempt_id,
            timeout_s=float(ns.timeout),
            preflight_max_age_s=300.0,
        )
        if (
            resume_preparation.get('ok') is not True
            or resume_preparation.get('measurement_wrapper_allowed') is not True
        ):
            failure = {
                'ok': False,
                'complete': False,
                'status': 'resume_preparation_or_cohort_preflight_failed',
                'requested': True,
                'resume_existing': True,
                'resume_attempt_id': resume_attempt_id,
                'resume_selected_rows': [
                    _resume_selector_text(value)
                    for value in resume_selectors
                ],
                'resume_selected_row_count': len(resume_selectors),
                'resume_merge_published': False,
                'resume_previous_results_sha256': prior_result_sha256,
                'resume_history_dir': (
                    str(resume_history_dir)
                    if resume_history_dir is not None else ''
                ),
                'resume_preparation': resume_preparation,
                'measurement_wrapper_started_count': 0,
                'started_measurement_count': 0,
                'collector_started_repeat_count': 0,
                'workload_started_repeat_count': 0,
            }
            _atomic_write_json(
                resume_attempt_root/'resume_failure.json',
                failure,
            )
            print(json.dumps({
                'ok': False,
                'status': failure['status'],
                'resume_selected_rows': len(resume_selectors),
                'resume_merge_published': False,
                'measurement_wrapper_started_count': 0,
                'started_measurement_count': 0,
                'collector_started_repeat_count': 0,
                'workload_started_repeat_count': 0,
                'resume_preparation_report': resume_preparation.get(
                    'report_path'
                ),
            }, indent=2), flush=True)
            return 2
    count=0
    started_measurement_count=0
    measurement_wrapper_started_count=0
    collector_started_repeat_count=0
    workload_started_repeat_count=0
    global_infrastructure_stop: dict[str, Any] = {}
    if managed_journal is not None:
        measurement_wrapper_started_count = sum(
            int(
                isinstance(checkpoint.get('execution'), Mapping)
                and checkpoint['execution'].get(
                    'measurement_wrapper_started'
                ) is True
            )
            for checkpoint in managed_journal.rows
        )
        for item in results:
            run = item.get('run') if isinstance(item, Mapping) else {}
            observation = (
                run.get('measurement_start_observation')
                if isinstance(run, Mapping) else {}
            )
            if not isinstance(observation, Mapping):
                continue
            started_measurement_count += int(
                observation.get('measurement_started') is True
            )
            collector_started_repeat_count += int(
                observation.get('collector_started_repeat_count') or 0
            )
            workload_started_repeat_count += int(
                observation.get('workload_started_repeat_count') or 0
            )
    partial_path = (
        resume_attempt_root/'native_producer_energy_results.partial.json'
        if resume_attempt_root is not None
        else out/'native_producer_energy_results.partial.json'
    )

    def _record_result(
        item: dict[str, Any], *, row_index: int | None = None,
    ) -> None:
        nonlocal results
        if managed_journal is not None:
            if row_index is None:
                raise ValueError('managed energy result has no row index')
            terminal_state = (
                'completed'
                if item.get('ok') is True or item.get('dry_run') is True
                else 'failed'
            )
            managed_journal.transition(
                row_index,
                terminal_state,
                result=item,
                reason=str(
                    item.get('error')
                    or item.get('energy_not_started_reason')
                    or item.get('blocked_reason')
                    or item.get('skipped')
                    or item.get('failure_category')
                    or ''
                ),
            )
            managed_results_by_index[row_index] = dict(item)
            results = [
                value for value in managed_results_by_index
                if value is not None
            ]
            return
        if not ns.resume_existing:
            results.append(item)
            return
        row = item.get('row')
        if not isinstance(row, dict):
            raise ValueError('resume result has no row identity')
        identity = _resume_row_identity(row)
        if identity not in resume_positions:
            raise ValueError(
                f'resume produced an unselected row: '
                f'{_resume_selector_text(identity)}'
            )
        results[resume_positions[identity]] = item

    def _write_partial():
        visible_rows = (
            [_clamp_screening_result(item) for item in results]
            if ns.screening_energy else results
        )
        partial={
            'ok': all(bool(x.get('ok')) for x in results),
            'complete': False,
            'plan':pr,
            'plan_payload': plan_payload,
            'preflight_status': str(plan_payload.get('preflight_status') or ''),
            'preflight': dict(plan_payload.get('preflight') or {}),
            'rows':visible_rows,
            'planned_row_count': total,
            'row_journal': (
                str(managed_journal.manifest_path)
                if managed_journal is not None else ''
            ),
            'row_states': (
                [
                    {
                        'index': int(value.get('index', -1)),
                        'identity': str(value.get('identity') or ''),
                        'state': str(value.get('state') or ''),
                        'attempt_count': int(value.get('attempt_count') or 0),
                    }
                    for value in managed_journal.rows
                ]
                if managed_journal is not None else []
            ),
            'resume_existing': bool(ns.resume_existing),
            'resume_attempt_id': resume_attempt_id,
            'selected_row_count': len(rows) if ns.resume_existing else None,
            'resume_preparation': resume_preparation,
            'measurement_wrapper_started_count': (
                measurement_wrapper_started_count
            ),
            'started_measurement_count': started_measurement_count,
            'collector_started_repeat_count': (
                collector_started_repeat_count
            ),
            'workload_started_repeat_count': (
                workload_started_repeat_count
            ),
            'global_infrastructure_stop': dict(
                global_infrastructure_stop
            ),
            'energy_not_started_categories': list(
                global_infrastructure_stop.get('categories') or []
            ),
            'energy_not_started_reason': str(
                global_infrastructure_stop.get('reason') or ''
            ),
            **artifact_claim_clamp,
        }
        _atomic_write_json(partial_path, partial)
    def _cancel_managed_row(
        row_index: int, *, reason: str,
    ) -> None:
        nonlocal managed_journal, results
        if managed_journal is None:
            return
        # A signal can arrive after the row-file replace but before the
        # in-memory list/derived manifest refresh.  Reload the authoritative
        # row files before deciding whether the active row needs cancellation
        # or already reached a terminal commit.
        managed_journal = AtomicRowJournal.open_for_resume(
            managed_journal_root,
            plan_hash=managed_plan_hash,
            identities=managed_identities,
            row_contract_hashes=managed_row_hashes,
        )
        checkpoint = managed_journal.rows[row_index]
        state = str(checkpoint.get('state') or '')
        if state == 'running':
            recovered = _recover_managed_running_result(checkpoint)
            if recovered is not None:
                managed_journal.recover_terminal(
                    row_index,
                    result=recovered,
                    reason='recovered_after_parent_import_gap',
                )
                managed_results_by_index[row_index] = dict(recovered)
            else:
                managed_journal.transition(
                    row_index, 'cancelled', reason=reason,
                )
        elif state in {'completed', 'failed'}:
            result = checkpoint.get('result')
            if isinstance(result, Mapping):
                managed_results_by_index[row_index] = dict(result)
        results = [
            value for value in managed_results_by_index
            if value is not None
        ]

    def _recount_managed_activity() -> None:
        nonlocal measurement_wrapper_started_count
        nonlocal started_measurement_count
        nonlocal collector_started_repeat_count
        nonlocal workload_started_repeat_count
        if managed_journal is None:
            return
        measurement_wrapper_started_count = sum(
            int(
                isinstance(checkpoint.get('execution'), Mapping)
                and checkpoint['execution'].get(
                    'measurement_wrapper_started'
                ) is True
            )
            for checkpoint in managed_journal.rows
        )
        started_measurement_count = 0
        collector_started_repeat_count = 0
        workload_started_repeat_count = 0
        for checkpoint in managed_journal.rows:
            result = checkpoint.get('result')
            run = result.get('run') if isinstance(result, Mapping) else {}
            observation = (
                run.get('measurement_start_observation')
                if isinstance(run, Mapping) else None
            )
            if not isinstance(observation, Mapping):
                execution = checkpoint.get('execution')
                output_dir = (
                    execution.get('output_dir')
                    if isinstance(execution, Mapping) else ''
                )
                if not str(output_dir or ''):
                    continue
                observation = _measurement_start_observation(
                    str(output_dir), aggregate_verified=False,
                )
            started_measurement_count += int(
                observation.get('measurement_started') is True
            )
            collector_started_repeat_count += int(
                observation.get('collector_started_repeat_count') or 0
            )
            workload_started_repeat_count += int(
                observation.get('workload_started_repeat_count') or 0
            )

    total=len(rows)

    def _set_global_infrastructure_stop(
        failure: Mapping[str, Any], *, row_index: int,
        trigger_measurement_started: bool = False,
    ) -> None:
        nonlocal global_infrastructure_stop
        category = str(
            failure.get('category') or 'native_energy_infrastructure'
        )
        code = str(
            failure.get('code') or 'global_energy_infrastructure_failed'
        )
        detail = ' '.join(str(
            failure.get('detail') or code
        ).split())[:1200]
        reason = (
            'Remaining Energy rows not started: global infrastructure '
            f'blocked by {code}: {detail}'
            if trigger_measurement_started else
            'Energy requested, not started: global infrastructure blocked by '
            f'{code}: {detail}'
        )
        global_infrastructure_stop = {
            'category': category,
            'categories': [category],
            'code': code,
            'detail': detail,
            'reason': reason,
            'trigger_row_index': int(row_index),
        }

    def _block_remaining_rows(*, after_index: int) -> None:
        if not global_infrastructure_stop:
            return
        for blocked_index in range(after_index + 1, total):
            if managed_journal is not None:
                state = str(
                    managed_journal.rows[blocked_index].get('state') or ''
                )
                if state in {'completed', 'failed'}:
                    continue
                if state in {'not_started', 'cancelled'}:
                    managed_journal.transition(
                        blocked_index,
                        'running',
                        execution={
                            'phase': 'blocked_global_infrastructure',
                            'row_index': blocked_index,
                            'measurement_wrapper_started': False,
                        },
                    )
            blocked_row = dict(rows[blocked_index])
            observation = _zero_measurement_start_observation()
            quality_projection = _energy_quality_result_projection(
                blocked_row,
                measurement_started=False,
                raw_energy_collected=False,
            )
            _record_result({
                'row': blocked_row,
                'ok': False,
                'skipped': 'global_infrastructure_stop',
                'blocked_reason': global_infrastructure_stop['code'],
                'failure_category': 'global_energy_infrastructure',
                'global_infrastructure_failure': (
                    global_infrastructure_stop['code']
                ),
                'global_infrastructure_category': (
                    global_infrastructure_stop['category']
                ),
                'energy_not_started_categories': list(
                    global_infrastructure_stop['categories']
                ),
                'energy_not_started_reason': (
                    global_infrastructure_stop['reason']
                ),
                'run': {
                    'measurement_start_observation': observation,
                },
                **quality_projection,
            }, row_index=(
                blocked_index if managed_journal is not None else None
            ))

    # A crash may occur after the trigger row was durably committed but before
    # the in-memory stop was projected to later rows.  Reconstruct that stop
    # from the row journal before any remaining measurement can start.
    if managed_journal is not None:
        for prior_index, prior_result in enumerate(managed_results_by_index):
            if not isinstance(prior_result, Mapping):
                continue
            prior_execution = managed_journal.rows[prior_index].get(
                'execution'
            )
            prior_output_dir = (
                prior_execution.get('output_dir')
                if isinstance(prior_execution, Mapping) else None
            )
            prior_failure = _energy_global_infrastructure_failure(
                prior_result, prior_output_dir,
            )
            if not prior_failure:
                continue
            prior_run = prior_result.get('run')
            prior_observation = (
                prior_run.get('measurement_start_observation')
                if isinstance(prior_run, Mapping) else {}
            )
            _set_global_infrastructure_stop(
                prior_failure,
                row_index=prior_index,
                trigger_measurement_started=bool(
                    isinstance(prior_observation, Mapping)
                    and prior_observation.get('measurement_started') is True
                ),
            )
            _block_remaining_rows(after_index=prior_index)
            break

    run_cancelled = False
    cancel_reason = ''
    active_row_index: int | None = None
    try:
        for row_index, r in enumerate(rows):
            active_row_index = row_index
            if managed_journal is not None:
                current_state = str(
                    managed_journal.rows[row_index].get('state') or ''
                )
                if current_state in {'completed', 'failed'}:
                    continue
                if current_state not in {'not_started', 'cancelled'}:
                    raise ValueError(
                        f'energy row {row_index} is not resumable: '
                        f'{current_state}'
                    )
                managed_journal.transition(
                    row_index,
                    'running',
                    execution={
                        'phase': 'preparing',
                        'row_index': row_index,
                        'measurement_wrapper_started': False,
                    },
                )

            cmd=str(r.get('measure_command') or '')
            count+=1
            if not cmd or r.get('skipped'):
                observation = _zero_measurement_start_observation()
                _record_result({
                    'row': r,
                    'ok': False,
                    'skipped': r.get('skipped') or 'no command',
                    'run': {
                        'measurement_start_observation': observation,
                    },
                    **_energy_quality_result_projection(
                        r,
                        measurement_started=False,
                        raw_energy_collected=False,
                    ),
                }, row_index=row_index if managed_journal is not None else None)
                _write_partial()
                continue
            if ns.limit and count>ns.limit:
                observation = _zero_measurement_start_observation()
                _record_result(
                    {
                        'row': r, 'ok': False, 'skipped': 'limit',
                        'run': {
                            'measurement_start_observation': observation,
                        },
                        **_energy_quality_result_projection(
                            r,
                            measurement_started=False,
                            raw_energy_collected=False,
                        ),
                    },
                    row_index=row_index if managed_journal is not None else None,
                )
                _write_partial()
                continue
            if ns.dry_run:
                observation = _zero_measurement_start_observation()
                _record_result(
                    {
                        'row': r, 'ok': None, 'dry_run': True,
                        'command': cmd,
                        'run': {
                            'measurement_start_observation': observation,
                        },
                    },
                    row_index=row_index if managed_journal is not None else None,
                )
                _write_partial()
                continue
            runtime_row = dict(r)
            execution_output_dir: str | Path | None = None
            activity_recorded = False
            wrapper_started = False
            row_global_infrastructure_failure: dict[str, str] = {}
            try:
                execution = _prepare_measurement_execution(
                    r, plan_payload, allowed_root=out,
                )
                runtime_row = dict(execution['row'])
                execution_output_dir = execution['output_dir']
                aggregate_absent_before_execution = not (
                    Path(execution['output_dir']) / 'energy_aggregate.json'
                ).exists()
                if not aggregate_absent_before_execution:
                    raise FileExistsError(
                        'fresh measurement target already contains '
                        'energy_aggregate.json'
                    )
                execution_started_ns = time.time_ns()
                runtime_row['measurement_execution_started_ns'] = (
                    execution_started_ns
                )
                if managed_journal is not None:
                    managed_journal.update_running(
                        row_index,
                        execution={
                            'phase': 'measurement_child_starting',
                            'row_index': row_index,
                            'measurement_wrapper_started': False,
                            'runtime_row': runtime_row,
                            'output_dir': str(execution['output_dir']),
                            'command': str(execution['command']),
                            'argv': list(map(str, execution['argv'])),
                            'expected_runs': int(execution['expected_runs']),
                            'expected_effective_runs': int(
                                execution['expected_effective_runs']
                            ),
                            'expected_run_id': str(
                                execution['expected_run_id']
                            ),
                            'expected_setup_id': str(
                                execution['expected_setup_id']
                            ),
                            'expected_command_contract_sha256': str(
                                execution[
                                    'expected_command_contract_sha256'
                                ]
                            ),
                            'execution_started_ns': execution_started_ns,
                            'aggregate_absent_before_execution': (
                                aggregate_absent_before_execution
                            ),
                            'execution_attempt_id': str(
                                execution.get('execution_attempt_id') or ''
                            ),
                        },
                    )
                print(
                    f"[native-energy] ROW_START index={row_index + 1} "
                    f"total={total} backend={r.get('backend','')} "
                    f"model={r.get('model','')} case={r.get('case','')} "
                    f"attempt={execution.get('execution_attempt_id','')}",
                    flush=True,
                )
                rr=_run(
                    execution['argv'], timeout=ns.timeout+300,
                    label=(
                        f"energy:{row_index + 1}/{total}:"
                        f"{r.get('backend','')}:{r.get('model','')}"
                    ),
                )
                if (
                    rr.get('cancelled') is True
                    or rr.get('rc') == 130
                    or rr.get('returncode') == 130
                ):
                    raise _ManagedEnergyCancelled(
                        0, 'measurement_child_cancelled'
                    )
                # A normal return proves that stream_command crossed its Popen
                # boundary.  Pre-Popen/import exceptions leave this counter and
                # journal marker false.
                measurement_wrapper_started_count += 1
                wrapper_started = True
                if managed_journal is not None:
                    journal_execution = dict(
                        managed_journal.rows[row_index].get('execution') or {}
                    )
                    journal_execution.update({
                        'phase': 'measurement_child_finished',
                        'measurement_wrapper_started': True,
                        'process_result': {key: rr[key] for key in (
                            'rc', 'cmd', 'elapsed_s', 'cancelled', 'timed_out'
                        ) if key in rr},
                    })
                    managed_journal.update_running(
                        row_index, execution=journal_execution,
                    )
                rr=_attach_energy_aggregate(
                    rr,
                    execution['command'],
                    execution['output_dir'],
                    expected_runs=execution['expected_runs'],
                    expected_effective_runs=execution[
                        'expected_effective_runs'
                    ],
                    expected_run_id=execution['expected_run_id'],
                    expected_setup_id=execution['expected_setup_id'],
                    expected_command_contract_sha256=execution[
                        'expected_command_contract_sha256'
                    ],
                    execution_started_ns=execution_started_ns,
                    expected_native_row=runtime_row,
                    aggregate_absent_before_execution=(
                        aggregate_absent_before_execution
                    ),
                )
                activity = _measurement_start_observation(
                    execution['output_dir'],
                    aggregate_verified=(
                        rr.get('energy_aggregate_verified') is True
                    ),
                )
                activity_recorded = True
                started_measurement_count += int(
                    activity['measurement_started']
                )
                collector_started_repeat_count += int(
                    activity['collector_started_repeat_count']
                )
                workload_started_repeat_count += int(
                    activity['workload_started_repeat_count']
                )
                rr['measurement_start_observation'] = activity
                quality_projection = _energy_quality_result_projection(
                    runtime_row,
                    measurement_started=(
                        activity.get('measurement_started') is True
                    ),
                    raw_energy_collected=_raw_energy_collected(rr),
                )
                rr.update(quality_projection)
                row_global_infrastructure_failure = (
                    _energy_global_infrastructure_failure(
                        rr, execution['output_dir'],
                    )
                )
                if row_global_infrastructure_failure:
                    rr['global_infrastructure_failure'] = str(
                        row_global_infrastructure_failure.get('code') or ''
                    )
                    rr['global_infrastructure_category'] = str(
                        row_global_infrastructure_failure.get('category') or ''
                    )
                    rr['global_infrastructure_detail'] = str(
                        row_global_infrastructure_failure.get('detail') or ''
                    )
                if execution['expected_command_contract_sha256']:
                    for field in (
                        'native_split_energy_quality_binding_sha256',
                        'native_split_quality_binding_sha256',
                        'native_split_quality_preselection_sha256',
                        'native_split_quality_source_request_sha256',
                        'native_split_quality_central_result_sha256',
                        'native_split_quality_selection_sha256',
                        'source_request_sha256',
                        'native_split_quality_eval_run_id',
                        'native_split_quality_source_run_id',
                        'native_split_quality_consumer_attestation_sha256',
                        'native_split_quality_authority_workflow_version',
                        'native_split_quality_authority_run_id',
                        'native_split_semantic_output_manifest_sha256',
                        'native_split_semantic_boundary_manifest_sha256',
                    ):
                        rr[field] = runtime_row.get(field)
                    rr['native_split_energy_command_contract_sha256'] = (
                        execution['expected_command_contract_sha256']
                    )
                    rr['native_split_energy_quality_binding_status'] = (
                        execution['split_quality_binding_status']
                    )
                _record_result({
                    'row': runtime_row,
                    'ok': (
                        rr.get('rc') == 0
                        and rr.get('energy_aggregate_verified') is True
                        and not row_global_infrastructure_failure
                    ),
                    'run': rr,
                    'failure_category': (
                        'global_energy_infrastructure'
                        if row_global_infrastructure_failure else ''
                    ),
                    'global_infrastructure_failure': str(
                        row_global_infrastructure_failure.get('code') or ''
                    ),
                    'global_infrastructure_category': str(
                        row_global_infrastructure_failure.get('category') or ''
                    ),
                    **quality_projection,
                }, row_index=row_index if managed_journal is not None else None)
                print(
                    f"[native-energy] ROW_END index={row_index + 1} "
                    f"total={total} rc={rr.get('rc')} "
                    f"elapsed={rr.get('elapsed_s',0):.1f}s",
                    flush=True,
                )
            except (_ManagedEnergyCancelled, BrokenPipeError) as exc:
                cancel_reason = (
                    exc.reason
                    if isinstance(exc, _ManagedEnergyCancelled)
                    else 'parent_pipe_closed'
                )
                _cancel_managed_row(row_index, reason=cancel_reason)
                run_cancelled = True
                break
            except subprocess.TimeoutExpired as exc:
                activity = (
                    _measurement_start_observation(
                        execution_output_dir,
                        aggregate_verified=False,
                    )
                    if execution_output_dir is not None
                    else _zero_measurement_start_observation()
                )
                activity_recorded = True
                started_measurement_count += int(
                    activity['measurement_started']
                )
                collector_started_repeat_count += int(
                    activity['collector_started_repeat_count']
                )
                workload_started_repeat_count += int(
                    activity['workload_started_repeat_count']
                )
                error_text = f'TimeoutExpired: {exc}'
                row_global_infrastructure_failure = (
                    _energy_global_infrastructure_failure(
                        {}, execution_output_dir, error=error_text,
                    )
                )
                quality_projection = _energy_quality_result_projection(
                    runtime_row,
                    measurement_started=(
                        activity['measurement_started'] is True
                    ),
                    raw_energy_collected=False,
                )
                _record_result({
                    'row':runtime_row,
                    'ok':False,
                    'error': error_text,
                    'failure_category': (
                        'global_energy_infrastructure'
                        if row_global_infrastructure_failure
                        else 'measurement_timeout'
                    ),
                    'global_infrastructure_failure': str(
                        row_global_infrastructure_failure.get('code') or ''
                    ),
                    'global_infrastructure_category': str(
                        row_global_infrastructure_failure.get('category') or ''
                    ),
                    'run': {
                        'measurement_start_observation': activity,
                    },
                    **quality_projection,
                }, row_index=row_index if managed_journal is not None else None)
            except Exception as exc:
                activity = (
                    _measurement_start_observation(
                        execution_output_dir,
                        aggregate_verified=False,
                    )
                    if execution_output_dir is not None
                    else _zero_measurement_start_observation()
                )
                activity_recorded = True
                started_measurement_count += int(
                    activity['measurement_started']
                )
                collector_started_repeat_count += int(
                    activity['collector_started_repeat_count']
                )
                workload_started_repeat_count += int(
                    activity['workload_started_repeat_count']
                )
                error_text = f'{type(exc).__name__}: {exc}'
                row_global_infrastructure_failure = (
                    _energy_global_infrastructure_failure(
                        {}, execution_output_dir, error=error_text,
                    )
                )
                quality_projection = _energy_quality_result_projection(
                    runtime_row,
                    measurement_started=(
                        activity['measurement_started'] is True
                    ),
                    raw_energy_collected=False,
                )
                _record_result({
                    'row': runtime_row,
                    'ok': False,
                    'error': error_text,
                    'failure_category': (
                        'global_energy_infrastructure'
                        if row_global_infrastructure_failure
                        else 'measurement_execution_contract_failed'
                    ),
                    'global_infrastructure_failure': str(
                        row_global_infrastructure_failure.get('code') or ''
                    ),
                    'global_infrastructure_category': str(
                        row_global_infrastructure_failure.get('category') or ''
                    ),
                    'run': {
                        'measurement_start_observation': activity,
                    },
                    **quality_projection,
                }, row_index=row_index if managed_journal is not None else None)
            finally:
                if (
                    execution_output_dir is not None
                    and wrapper_started
                    and not activity_recorded
                ):
                    activity = _measurement_start_observation(
                        execution_output_dir,
                        aggregate_verified=False,
                    )
                    started_measurement_count += int(
                        activity['measurement_started']
                    )
                    collector_started_repeat_count += int(
                        activity['collector_started_repeat_count']
                    )
                    workload_started_repeat_count += int(
                        activity['workload_started_repeat_count']
                    )
            if row_global_infrastructure_failure:
                _set_global_infrastructure_stop(
                    row_global_infrastructure_failure,
                    row_index=row_index,
                    trigger_measurement_started=bool(
                        activity_recorded
                        and activity.get('measurement_started') is True
                    ),
                )
                _block_remaining_rows(after_index=row_index)
                _write_partial()
                break
            _write_partial()
    except (_ManagedEnergyCancelled, BrokenPipeError) as exc:
        cancel_reason = (
            exc.reason
            if isinstance(exc, _ManagedEnergyCancelled)
            else 'parent_pipe_closed'
        )
        if active_row_index is not None:
            _cancel_managed_row(
                active_row_index, reason=cancel_reason,
            )
        run_cancelled = True
    _recount_managed_activity()
    if run_cancelled:
        if managed_journal is not None:
            managed_journal.mark_run_state('cancelled')
        _write_partial()
        if managed_journal is not None:
            row_counts = {
                state: sum(
                    1 for checkpoint in managed_journal.rows
                    if str(checkpoint.get('state') or '') == state
                )
                for state in (
                    'not_started', 'running', 'completed', 'failed',
                    'cancelled',
                )
            }
            _write_stage_checkpoint(
                managed_stage_checkpoint_path,
                stage='native_energy',
                state='cancelled',
                complete=False,
                input_hash=managed_invocation_hash,
                run_root=out,
                artifacts=[managed_journal.manifest_path, partial_path],
                details={
                    'planned_row_count': total,
                    'row_counts': row_counts,
                    'resume_checkpoint': bool(ns.resume_checkpoint),
                    'return_code': 130,
                },
                error=cancel_reason,
            )
        return 130
    if managed_journal is not None:
        if not managed_journal.complete:
            managed_journal.mark_run_state('failed')
            _write_partial()
            raise ValueError(
                'managed energy row journal is incomplete after execution'
            )
        managed_journal.mark_run_state('completed')
    zero_started = bool(not ns.dry_run and started_measurement_count == 0)
    if ns.screening_energy:
        if ns.resume_existing:
            for position in resume_positions.values():
                results[position] = _clamp_screening_result(results[position])
        else:
            results = [_clamp_screening_result(item) for item in results]
    report_ok = (
        True if ns.dry_run
        else bool(started_measurement_count) and all(bool(x.get('ok')) for x in results)
    )
    resume_selected_ok = bool(
        ns.resume_existing
        and resume_positions
        and all(
            bool(results[position].get('ok'))
            for position in resume_positions.values()
        )
    )
    resume_selected_position_set = set(resume_positions.values())
    report=dict(existing_report or {}) if ns.resume_existing else {}
    result_ledger = _result_ledger_from_results(results)
    report.update({
        'ok': report_ok,
        'complete': False if zero_started else True,
        'state': 'completed' if report_ok or ns.dry_run else 'failed',
        'status': (
            'failed_global_energy_infrastructure'
            if global_infrastructure_stop
            else 'blocked_zero_measurements_started' if zero_started
            else 'dry_run' if ns.dry_run
            else 'completed' if report_ok
            else 'failed_measurement_or_aggregate_contract'
        ),
        'blocked_reason': (
            str(global_infrastructure_stop.get('code') or '')
            if global_infrastructure_stop
            else 'no_runnable_measurement_command' if zero_started else ''
        ),
        'global_infrastructure_stop': dict(global_infrastructure_stop),
        'energy_not_started_categories': list(
            global_infrastructure_stop.get('categories') or []
        ),
        'energy_not_started_reason': str(
            global_infrastructure_stop.get('reason') or ''
        ),
        'requested': True,
        'historical_diagnostic_only': bool(
            plan_payload.get('historical_diagnostic_only')
        ),
        'screening_only': bool(ns.screening_energy),
        'screening_energy': bool(ns.screening_energy),
        'measure_all_runtime_successful': bool(
            ns.measure_all_runtime_successful
        ),
        'measurement_admission_policy': str(
            plan_payload.get("measurement_admission_policy") or ""
        ),
        'energy_evidence_tier': (
            'screening' if ns.screening_energy else
            'smoke_diagnostic' if ns.smoke_diagnostic else 'final_claim'
        ),
        'energy_tier': (
            'screening' if ns.screening_energy else
            'smoke_diagnostic' if ns.smoke_diagnostic else 'final_claim'
        ),
        'smoke_diagnostic': bool(ns.smoke_diagnostic),
        'diagnostic_only': bool(ns.screening_energy or ns.smoke_diagnostic),
        'claim_eligible': False
        if ns.screening_energy or ns.smoke_diagnostic else None,
        'eligible_for_energy_results_import': False
        if ns.screening_energy or ns.smoke_diagnostic
        or plan_payload.get('historical_diagnostic_only') is True else None,
        'eligible_for_scientific_claim': False
        if ns.screening_energy or ns.smoke_diagnostic
        or plan_payload.get('historical_diagnostic_only') is True else None,
        'energy_claim_eligible': False
        if ns.screening_energy or ns.smoke_diagnostic else None,
        'claim_ok': False
        if ns.screening_energy or ns.smoke_diagnostic else None,
        'semantic_claim_ok': False
        if ns.screening_energy or ns.smoke_diagnostic else None,
        'scientific_claim_exclusion_reasons': list(
            plan_payload.get('scientific_claim_exclusion_reasons') or []
        ),
        'preflight_status': str(plan_payload.get('preflight_status') or ''),
        'preflight': dict(plan_payload.get('preflight') or {}),
        'resume_preparation': resume_preparation,
        'measurement_wrapper_started_count': (
            measurement_wrapper_started_count
        ),
        'started_measurement_count': started_measurement_count,
        'collector_started_repeat_count': (
            collector_started_repeat_count
        ),
        'workload_started_repeat_count': (
            workload_started_repeat_count
        ),
        'dry_run': bool(ns.dry_run), 'plan':pr, 'plan_payload':plan_payload,
        'rows':results,
        'planned_row_count': total,
        'row_journal': (
            str(managed_journal.manifest_path)
            if managed_journal is not None else ''
        ),
        'row_states': (
            [
                {
                    'index': int(value.get('index', -1)),
                    'identity': str(value.get('identity') or ''),
                    'state': str(value.get('state') or ''),
                    'attempt_count': int(value.get('attempt_count') or 0),
                }
                for value in managed_journal.rows
            ]
            if managed_journal is not None else []
        ),
        'decision_complete': (
            managed_journal.complete
            if managed_journal is not None
            else False if zero_started else True
        ),
        'result_ledger': result_ledger,
        'result_ledger_valid': _result_ledger_valid(
            result_ledger,
            [
                dict(item.get('row') or {})
                for item in results
                if isinstance(item, Mapping)
            ],
        ),
        'resume_existing': bool(ns.resume_existing),
        'resume_attempt_id': resume_attempt_id,
        'resume_selected_rows': (
            [_resume_selector_text(value) for value in resume_selectors]
            if ns.resume_existing else []
        ),
        'resume_selected_row_count': (
            len(resume_selectors) if ns.resume_existing else 0
        ),
        'resume_reused_row_count': (
            sum(
                1
                for position, item in enumerate(results)
                if position not in resume_selected_position_set
                and _energy_result_reusable(item)
            )
            if ns.resume_existing else 0
        ),
        'resume_previous_results_sha256': prior_result_sha256,
        'resume_history_dir': (
            str(resume_history_dir) if resume_history_dir is not None else ''
        ),
        'resume_repeat_policy': (
            'fresh_complete_series_per_selected_row_no_repeat_splicing'
            if ns.resume_existing else ''
        ),
        'resume_selected_rows_ok': (
            resume_selected_ok if ns.resume_existing else None
        ),
        'resume_merge_published': (
            resume_selected_ok if ns.resume_existing else None
        ),
    })
    if ns.resume_existing and not resume_selected_ok:
        report['status'] = 'resume_selected_measurement_failed_no_merge'
    md=['# Native producer energy run','','| backend | model | case | ok | note |','|---|---|---|---:|---|']
    for x in results:
        r=x.get('row',{})
        md.append(f"| {r.get('backend','')} | {r.get('model','')} | {r.get('case','')} | {x.get('ok')} | {x.get('energy_not_started_reason') or x.get('error') or x.get('skipped') or ''} |")
    published_json_path = canonical_result_path
    published_md_path = out/'native_producer_energy_results.md'
    if ns.resume_existing and resume_selected_ok:
        _atomic_write_json(canonical_result_path, report)
        _atomic_write_text(
            out/'native_producer_energy_results.md', '\n'.join(md),
        )
    elif ns.resume_existing:
        assert resume_attempt_root is not None
        published_json_path = (
            resume_attempt_root/'resume_failed_results_no_merge.json'
        )
        published_md_path = (
            resume_attempt_root/'resume_failed_results_no_merge.md'
        )
        _atomic_write_json(published_json_path, report)
        _atomic_write_text(published_md_path, '\n'.join(md))
    else:
        _atomic_write_json(canonical_result_path, report)
        _atomic_write_text(
            out/'native_producer_energy_results.md', '\n'.join(md),
        )
    if managed_journal is not None:
        row_counts = {
            state: sum(
                1 for checkpoint in managed_journal.rows
                if str(checkpoint.get('state') or '') == state
            )
            for state in (
                'not_started', 'running', 'completed', 'failed',
                'cancelled',
            )
        }
        _write_stage_checkpoint(
            managed_stage_checkpoint_path,
            stage='native_energy',
            state='completed' if report_ok or ns.dry_run else 'failed',
            complete=True,
            input_hash=managed_invocation_hash,
            run_root=out,
            artifacts=[
                managed_journal.manifest_path,
                canonical_result_path,
                out/'native_producer_energy_results.md',
            ],
            details={
                'planned_row_count': total,
                'row_counts': row_counts,
                'report_ok': bool(report_ok),
                'resume_checkpoint': bool(ns.resume_checkpoint),
                'managed_plan_hash': managed_plan_hash,
                'return_code': (
                    0 if report_ok
                    else 3 if zero_started
                    else 2
                ),
            },
            error=(
                '' if report_ok or ns.dry_run
                else str(report.get('status') or 'energy_stage_failed')
            ),
        )
    if report_ok or (ns.resume_existing and resume_selected_ok):
        try:
            partial_path.unlink()
        except FileNotFoundError:
            pass
    try:
        print(json.dumps({
            'ok': report['ok'],
            'rows': len(results),
            'json': str(published_json_path),
            'md': str(published_md_path),
            'resume_existing': bool(ns.resume_existing),
            'resume_checkpoint': bool(ns.resume_checkpoint),
            'resume_selected_rows': (
                len(resume_selectors) if ns.resume_existing else 0
            ),
            'resume_artifact_rehydration_ok': (
                bool(resume_preparation.get('ok'))
                if isinstance(resume_preparation, dict) else None
            ),
            'resume_cohort_preflight_ok': (
                bool(
                    isinstance(
                        resume_preparation.get('cohort_preflight'), dict
                    )
                    and resume_preparation['cohort_preflight'].get('ok') is True
                )
                if isinstance(resume_preparation, dict) else None
            ),
            'measurement_wrapper_started_count': (
                measurement_wrapper_started_count
            ),
            'started_measurement_count': started_measurement_count,
            'collector_started_repeat_count': (
                collector_started_repeat_count
            ),
            'workload_started_repeat_count': (
                workload_started_repeat_count
            ),
            'resume_merge_published': report.get('resume_merge_published'),
        }, indent=2), flush=True)
    except BrokenPipeError:
        # The report and stage checkpoint are already durable.  A later parent
        # can import them without rerunning completed Energy rows.
        return 130
    return 0 if report['ok'] else (3 if zero_started else 2)


def main() -> int:
    """Install one cancellation boundary around planning and row execution."""

    global _ACTIVE_MANAGED_STAGE_CONTEXT
    previous_signal_handlers = _install_managed_cancel_handlers()
    try:
        return _main_impl()
    except (_ManagedEnergyCancelled, BrokenPipeError) as exc:
        context = dict(_ACTIVE_MANAGED_STAGE_CONTEXT)
        if context:
            out = Path(context['out'])
            checkpoint_path = Path(context['checkpoint'])
            input_hash = str(context['input_hash'])
            existing, _reason = _load_stage_checkpoint(
                checkpoint_path,
                stage='native_energy',
                input_hash=input_hash,
                run_root=out,
            )
            if not (
                isinstance(existing, Mapping)
                and existing.get('complete') is True
                and str(existing.get('state') or '')
                in {'completed', 'failed'}
            ):
                reason = (
                    exc.reason
                    if isinstance(exc, _ManagedEnergyCancelled)
                    else 'parent_pipe_closed'
                )
                journal_cancel_error = ''
                journal_path = Path(context['journal'])
                managed_plan_hash = str(
                    context.get('managed_plan_hash') or ''
                )
                managed_identities = list(
                    context.get('managed_identities') or []
                )
                managed_row_hashes = list(
                    context.get('managed_row_hashes') or []
                )
                if (
                    (journal_path / 'journal.json').is_file()
                    and managed_plan_hash
                    and managed_identities
                    and managed_row_hashes
                ):
                    try:
                        journal = AtomicRowJournal.open_for_resume(
                            journal_path,
                            plan_hash=managed_plan_hash,
                            identities=managed_identities,
                            row_contract_hashes=managed_row_hashes,
                        )
                        running_indexes = [
                            int(row.get('index', -1))
                            for row in journal.rows
                            if str(row.get('state') or '') == 'running'
                        ]
                        if len(running_indexes) > 1:
                            raise ValueError(
                                'managed Energy cancellation found more than '
                                'one running row'
                            )
                        if running_indexes:
                            journal.transition(
                                running_indexes[0],
                                'cancelled',
                                reason=reason,
                            )
                        journal.mark_run_state('cancelled')
                    except Exception as journal_exc:
                        journal_cancel_error = (
                            f'{type(journal_exc).__name__}: {journal_exc}'
                        )
                artifacts: list[Path] = []
                for candidate in (
                    Path(context['journal']) / 'journal.json',
                    out / 'native_producer_energy_results.partial.json',
                ):
                    if candidate.is_file() and not candidate.is_symlink():
                        artifacts.append(candidate)
                _write_stage_checkpoint(
                    checkpoint_path,
                    stage='native_energy',
                    state='cancelled',
                    complete=False,
                    input_hash=input_hash,
                    run_root=out,
                    artifacts=artifacts,
                    details={
                        'return_code': 130,
                        'resume_checkpoint': bool(
                            context.get('resume_checkpoint')
                        ),
                        'journal_cancel_error': journal_cancel_error,
                    },
                    error=reason,
                )
        return 130
    finally:
        _restore_managed_cancel_handlers(previous_signal_handlers)
        _ACTIVE_MANAGED_STAGE_CONTEXT = {}


if __name__=='__main__': raise SystemExit(main())
