"""Pairing-independent u.RECS window-method screening probe.

The probe deliberately lives outside ``native_energy_measurements``.  It uses
one successful representative native command, acquires one raw trace per
repeat, and compares marker-v2 cropping with the historical flank/duration
window on exactly those bytes.  Nothing written here is eligible for Native
Energy, scientific-report, or claim-table ingestion.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

from .native_command_contract import (
    NATIVE_COMMAND_CONTRACT_SCHEMA,
    split_energy_runtime_argv,
    verify_native_command_contract,
)
from .process_control import ProcessTreeRegistry
from .remote.process_lease import (
    REMOTE_LEASE_OUTER_CLEANUP_MARGIN_S,
    cancel_journaled_remote_processes_from_environment,
    journaled_ssh_wrapper_argv,
)


SCHEMA = "onnx-splitpoint/window-method-validation-probe"
MIN_DECISION_REPEATS = 3
PARQUET_MAGIC = b"PAR1"


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _valid_parquet_container(path: Path) -> tuple[bool, str]:
    """Reject truncated collector placeholders before they become evidence."""
    try:
        size = path.stat().st_size
        if size < 12:
            return False, "too_small_for_parquet_footer"
        with path.open("rb") as handle:
            head = handle.read(4)
            handle.seek(-4, 2)
            tail = handle.read(4)
        if head != PARQUET_MAGIC or tail != PARQUET_MAGIC:
            return False, "parquet_magic_or_footer_missing"
        return True, "valid_parquet_container"
    except Exception as exc:
        return False, f"parquet_validation_failed:{type(exc).__name__}"


def _resolve(raw: Any, *, relative_to: Path) -> Path | None:
    text = str(raw or "").strip()
    if not text:
        return None
    path = Path(text).expanduser()
    return path if path.is_absolute() else (relative_to / path).resolve()


def _relative_path_or_absolute(path: Path, root: Path) -> str:
    """Python-3.8-compatible equivalent of Path.is_relative_to + relative_to."""
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path.resolve())


def _script(name: str) -> Path:
    project_root = Path(__file__).resolve().parents[1]
    source = project_root / "scripts" / name
    if source.is_file():
        return source
    packaged = Path(__file__).resolve().parent / "resources" / "remote_scripts" / name
    if packaged.is_file():
        return packaged
    raise FileNotFoundError(name)


def _run(command: Sequence[str], *, timeout: float | None = None) -> dict[str, Any]:
    argv = [str(item) for item in command]
    broker_timeout_s = (
        max(
            0.1,
            float(timeout)
            - min(
                REMOTE_LEASE_OUTER_CLEANUP_MARGIN_S,
                max(1.0, float(timeout) * 0.10),
            ),
        )
        if timeout is not None else None
    )
    argv = journaled_ssh_wrapper_argv(
        argv,
        label="window-probe-command",
        env=os.environ,
        timeout_s=broker_timeout_s,
    )
    registry = ProcessTreeRegistry()
    proc: subprocess.Popen[str] | None = None

    def _bounded_collect_after_stop() -> tuple[str, str]:
        assert proc is not None
        try:
            stdout, stderr = proc.communicate(timeout=2.0)
            return str(stdout or ""), str(stderr or "")
        except subprocess.TimeoutExpired as exc:
            registry.terminate_registered(proc, grace_s=0.0)
            try:
                stdout, stderr = proc.communicate(timeout=0.5)
                return str(stdout or ""), str(stderr or "")
            except subprocess.TimeoutExpired as final_exc:
                stdout = final_exc.output if final_exc.output is not None else exc.output
                stderr = final_exc.stderr if final_exc.stderr is not None else exc.stderr
                return str(stdout or ""), str(stderr or "")

    try:
        popen_kwargs: dict[str, Any] = {
            "text": True,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
        }
        if os.name == "posix":
            popen_kwargs["start_new_session"] = True
        elif hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP"):  # pragma: no cover
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        proc = subprocess.Popen(
            argv,
            **popen_kwargs,
        )
        registry.register(proc, label="window-method-validation-child")
        deadline = (
            time.monotonic() + max(0.0, float(timeout))
            if timeout is not None else None
        )
        while True:
            wait_s = 0.1
            if deadline is not None:
                remaining_s = deadline - time.monotonic()
                if remaining_s <= 0.0:
                    cancel_journaled_remote_processes_from_environment(
                        env=os.environ,
                        grace_s=3.0,
                    )
                    registry.terminate_registered(proc, grace_s=2.0)
                    stdout, stderr = _bounded_collect_after_stop()
                    return {
                        "rc": 124,
                        "error": f"TimeoutExpired after {timeout}s",
                        "stdout_tail": str(stdout or "")[-4000:],
                        "stderr_tail": str(stderr or "")[-4000:],
                    }
                wait_s = min(wait_s, remaining_s)
            try:
                stdout, stderr = proc.communicate(timeout=wait_s)
                break
            except subprocess.TimeoutExpired:
                continue
        if stdout:
            print(stdout, end="" if stdout.endswith("\n") else "\n", flush=True)
        if stderr:
            print(stderr, end="" if stderr.endswith("\n") else "\n", file=sys.stderr, flush=True)
        return {
            "rc": int(proc.returncode or 0),
            "stdout_tail": str(stdout or "")[-4000:],
            "stderr_tail": str(stderr or "")[-4000:],
        }
    except BaseException:
        if proc is not None and proc.poll() is None:
            try:
                cancel_journaled_remote_processes_from_environment(
                    env=os.environ,
                    grace_s=3.0,
                )
            finally:
                registry.terminate_registered(proc, grace_s=2.0)
        raise
    finally:
        if proc is not None:
            registry.unregister(proc)


def _option(parts: list[str], flag: str, value: str) -> list[str]:
    """Replace one single-valued CLI option without retaining stale values."""
    out = list(parts)
    while flag in out:
        index = out.index(flag)
        del out[index:index + (2 if index + 1 < len(out) else 1)]
    out.extend([flag, str(value)])
    return out


def _source_row_for_target(
    summary: Mapping[str, Any], target: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, str]:
    fields = ("backend", "model", "case", "precision")
    identity = tuple(str(target.get(field) or "").strip().lower() for field in fields)
    matches = []
    for raw in list(summary.get("rows") or []):
        if not isinstance(raw, Mapping) or raw.get("ok") is not True:
            continue
        row = dict(raw)
        row_identity = tuple(str(row.get(field) or "").strip().lower() for field in fields)
        if row_identity == identity:
            matches.append(row)
    if len(matches) != 1:
        return None, "successful_source_row_missing" if not matches else "successful_source_row_ambiguous"
    return matches[0], "exact_identity_match"


def _verified_native_contract(
    row: Mapping[str, Any], target: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, str]:
    contract, status = verify_native_command_contract(
        row.get("native_command_contract"), expected_identity=target,
    )
    if contract is None:
        return None, status
    options = contract.get("runtime_options")
    backend = str(contract.get("backend") or "").strip().lower()
    required_options = {
        "hailo8_to_trt": ("warmup", "queue_depth", "hailo_format", "letterbox_pad_value"),
        "hailo10h_to_trt": ("warmup", "queue_depth", "inflight", "producer_impl"),
        "deepx_to_trt": ("warmup", "queue_depth"),
    }.get(backend)
    if required_options is None:
        return None, "native_command_contract_probe_backend_unsupported"
    if any(options.get(field) in (None, "") for field in required_options):
        return None, "native_command_contract_required_runtime_option_missing"
    return contract, status


def _remote_endpoint(
    contract: Mapping[str, Any], ns: argparse.Namespace,
) -> tuple[str, str]:
    backend = str(contract.get("backend") or "").strip().lower()
    endpoints = {
        "hailo8_to_trt": (str(ns.hailo8_ssh or ""), str(ns.hailo8_env or "")),
        "hailo10h_to_trt": (str(ns.hailo10_ssh or ""), str(ns.hailo10_env or "")),
        "deepx_to_trt": (str(ns.deepx_ssh or ""), str(ns.deepx_env or "")),
    }
    if backend not in endpoints:
        raise ValueError(f"unsupported split replay backend: {backend or '<missing>'}")
    ssh, remote_env = endpoints[backend]
    if not ssh.strip():
        raise ValueError(f"SSH endpoint missing for exact {backend} replay")
    return ssh.strip(), remote_env.strip()


def _hailo8_replay_command(
    *,
    contract: Mapping[str, Any],
    ns: argparse.Namespace,
    attempt_id: str,
    repeat_index: int,
    reconnect_attempt: int,
    command_file: Path,
) -> dict[str, Any]:
    """Materialise a successful split contract with only duration/output changes."""
    options = dict(contract.get("runtime_options") or {})
    boundary = dict(contract.get("boundary_contract") or {})
    model = str(contract.get("model") or Path(str(contract["benchmark_set"])).parent.name)
    expected_bs = f"{str(ns.remote_root).rstrip('/')}/{model}/benchmark_set"
    if str(contract["benchmark_set"]).rstrip("/") != expected_bs.rstrip("/"):
        raise ValueError(
            "archived benchmark_set does not match the current staged native root: "
            f"archived={contract['benchmark_set']} expected={expected_bs}"
        )
    ssh, remote_env = _remote_endpoint(contract, ns)
    remote_probe = (
        Path(str(contract["benchmark_set"])) / "window_method_validation_probe"
        / attempt_id / f"repeat_{repeat_index:03d}"
        / f"attempt_{reconnect_attempt:02d}"
        / "capture___ONNX_SPLITPOINT_PREFLIGHT_NONCE__"
    ).as_posix()
    python_executable = str(contract["python_executable"])
    # The energy plan always attaches a per-repeat, nonce-bound preflight.  The
    # workload template must therefore carry the same literal placeholders as
    # the plan-generated command.  ``successful_runtime_argv`` is the right
    # command for a normal benchmark replay, but it deliberately has no energy
    # attestation arguments and consequently caused the collector to reject the
    # probe before the first sample was acquired.
    child = split_energy_runtime_argv(
        contract,
        duration_s=float(ns.duration_s),
        fresh_output_root=remote_probe,
        remote_tool_dir=str(ns.remote_tool_dir),
        preflight_attestation_path="__ONNX_SPLITPOINT_PREFLIGHT_ATTESTATION__",
        preflight_nonce="__ONNX_SPLITPOINT_PREFLIGHT_NONCE__",
    )
    wrapper = f"{str(ns.remote_tool_dir).rstrip('/')}/scripts/run_and_report_work_units.py"
    remote_parts = [python_executable, wrapper, "--", *child]
    # The runner must create this attempt directory itself.  A pre-existing
    # location is evidence of stale/reused output and aborts before workload
    # execution.
    remote_body = (
        "cd " + shlex.quote(str(ns.remote_tool_dir))
        + " && test ! -e " + shlex.quote(remote_probe)
        + " && " + shlex.join(remote_parts)
    )
    if remote_env:
        remote_body = remote_env + " && " + remote_body
    ssh_command = [
        "ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
        ssh, remote_body,
    ]
    leased_ssh_command = journaled_ssh_wrapper_argv(
        ssh_command,
        label=f"window-probe-repeat-{repeat_index}-attempt-{reconnect_attempt}",
        env=os.environ,
        timeout_s=max(
            0.1,
            # Older programmatic callers construct the Namespace directly and
            # therefore do not pass through argparse's 900-second default.
            float(getattr(ns, "timeout", 900))
            - REMOTE_LEASE_OUTER_CLEANUP_MARGIN_S,
        ),
    )
    command_file.parent.mkdir(parents=True, exist_ok=True)
    command_file.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        + shlex.join(leased_ssh_command) + "\n",
        encoding="utf-8",
    )
    command_file.chmod(0o755)
    return {
        "command_file": str(command_file),
        "command_file_sha256": _sha256(command_file),
        "remote_output_root": remote_probe,
        "source_contract_sha256": str(contract["contract_sha256"]),
        "runtime_contract_preserved": {
            "image": str(contract["input_image"]),
            "image_sha256": str(contract["input_image_sha256"]),
            "queue_depth": int(options["queue_depth"]),
            "source_warmup": int(options["warmup"]),
            "effective_energy_warmup": 0,
            "boundary_layout": str(boundary["boundary_layout_effective"]),
            "precision": str(contract["precision"]),
            "backend_specific_runtime_options": options,
            "preflight_attested": True,
            "preflight_nonce_template_present": True,
        },
        "controlled_changes": {
            "duration_s": float(ns.duration_s),
            "fresh_remote_output_root": remote_probe,
            "reconnect_attempt": int(reconnect_attempt),
            "preexisting_remote_output_rejected": True,
            "build_disabled_during_measurement": True,
        },
        "ssh_command": ssh_command,
        "lease_broker_command": leased_ssh_command,
    }


def _hailo8_remote_preflight(
    contract: Mapping[str, Any], ns: argparse.Namespace,
) -> list[str]:
    """Verify the exact staged files immediately before each collector attempt."""
    remote_runner = (
        f"{str(ns.remote_tool_dir).rstrip('/')}/"
        f"{str(contract['runner']).lstrip('/')}"
    )
    files = {
        remote_runner: str(contract["runner_sha256"]),
        str(contract["input_image"]): str(contract["input_image_sha256"]),
    }
    for artifact in dict(contract.get("artifacts") or {}).values():
        if isinstance(artifact, Mapping):
            path = str(artifact.get("path") or "").strip()
            digest = str(artifact.get("sha256") or "").strip()
            if path and digest:
                files[path] = digest
    code = (
        "import hashlib,json,pathlib,sys;"
        f"files=json.loads({json.dumps(files, separators=(',', ':'))!r});"
        "bad=[];"
        "[(bad.append({'path':p,'reason':'missing'}) if not pathlib.Path(p).is_file() "
        "else bad.append({'path':p,'reason':'sha256_mismatch'}) if hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()!=h else None) for p,h in files.items()];"
        "print(json.dumps({'ok':not bad,'files':len(files),'failures':bad}));"
        "sys.exit(0 if not bad else 9)"
    )
    python_executable = str(contract["python_executable"])
    ssh, remote_env = _remote_endpoint(contract, ns)
    remote_body = "cd " + shlex.quote(str(ns.remote_tool_dir)) + " && " + shlex.join(
        [python_executable, "-c", code]
    )
    if remote_env:
        remote_body = remote_env + " && " + remote_body
    return [
        "ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
        ssh, remote_body,
    ]
def _write_report(out: Path, payload: Mapping[str, Any]) -> Path:
    path = out / "window_method_validation_probe.json"
    path.write_text(json.dumps(dict(payload), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _write_probe_artifact_index(out: Path) -> Path:
    path = out / "artifact_index.json"
    files: list[dict[str, Any]] = []
    for source in sorted(out.rglob("*")):
        if not source.is_file() or source == path:
            continue
        rel = source.relative_to(out).as_posix()
        files.append({
            "path": str(source.resolve()),
            "relative_path": rel,
            "size_bytes": source.stat().st_size,
            "sha256": _sha256(source),
            "artifact_kind": (
                "raw_trace" if source.suffix.lower() == ".parquet"
                else "window_method_A/B_comparison" if source.name == "window_method_comparison.json"
                else "probe_evidence"
            ),
        })
    payload = {
        "schema": "onnx-splitpoint/window-method-validation-probe-artifact-index",
        "schema_version": 1,
        "screening_only": True,
        "eligible_for_energy_results_import": False,
        "file_count": len(files),
        "raw_parquet_count": sum(1 for row in files if row["artifact_kind"] == "raw_trace"),
        "comparison_json_count": sum(1 for row in files if row["artifact_kind"] == "window_method_A/B_comparison"),
        "files": files,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _base_report(*, repeats: int, include_raw: bool, strict: bool) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "schema_version": 2,
        "requested": True,
        "screening_only": True,
        "diagnostic_only": True,
        "eligible_for_energy_results_import": False,
        "eligible_for_scientific_report_import": False,
        "eligible_for_scientific_claim": False,
        "affects_native_energy_pairing": False,
        "affects_primary_energy_result": False,
        "affects_final_energy_gate": False,
        "scientific_method_decision": "frozen_command_marker_primary_chapter4_shadow",
        "scientific_primary_method_frozen": True,
        "scientific_primary_method": "command_marker_window",
        "scientific_shadow_method": "chapter4_legacy_window",
        "primary_method": "command_marker_window",
        "shadow_method": "chapter4_legacy_window",
        "shadow_role": "same_trace_sensitivity_only",
        "shadow_affects_primary_result": False,
        "shadow_affects_final_gate": False,
        "primary_implementation": "collector_sample_marker_crop",
        "shadow_implementation": "historical_power_edge_detection_with_estimated_duration",
        "comparison_method": "chapter4_legacy_window",
        "same_raw_trace_required": True,
        "raw_parquet_required": True,
        "raw_parquet_debug_pack_requested": bool(include_raw),
        "minimum_sensitivity_repeats": MIN_DECISION_REPEATS,
        # Compatibility name used by pre-v2.67 probe readers.  This now means
        # decision-capable sensitivity evidence, not an open method choice.
        "minimum_decision_repeats": MIN_DECISION_REPEATS,
        "requested_repeat_count": int(repeats),
        "strict_requested_validation": bool(strict),
    }


def _repeat_evidence(measurement: Path, aggregate: Mapping[str, Any]) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    runs = list(aggregate.get("runs") or []) if isinstance(aggregate, Mapping) else []
    for index, run in enumerate(runs):
        row = dict(run) if isinstance(run, Mapping) else {}
        run_index = int(row.get("run_index") if row.get("run_index") is not None else index)
        explicit_run_dir = _resolve(row.get("_probe_run_dir"), relative_to=measurement)
        run_dir = explicit_run_dir or (measurement / f"run_{run_index:03d}")
        comparison_path = run_dir / "window_method_comparison.json"
        comparison = _json(comparison_path)
        raw = comparison.get("raw_trace") if isinstance(comparison.get("raw_trace"), Mapping) else {}
        trace = _resolve(raw.get("trace_path"), relative_to=comparison_path.parent)
        trace_hash = _sha256(trace) if trace is not None and trace.is_file() else ""
        declared_hash_values = [
            str(raw.get("request_sha256") or "").lower(),
            str(raw.get("sha256_before_legacy_postprocess") or "").lower(),
            str(raw.get("sha256_after_legacy_postprocess") or "").lower(),
        ]
        parquet_files = sorted((run_dir / "collector_storage").glob("*.parquet"))
        parquet = []
        for path in parquet_files:
            if not path.is_file():
                continue
            valid_container, validation_status = _valid_parquet_container(path)
            parquet.append({
                "path": str(path.resolve()),
                "relative_path": _relative_path_or_absolute(path, measurement),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "valid_parquet_container": valid_container,
                "parquet_validation_status": validation_status,
            })
        valid_parquet = [item for item in parquet if item["valid_parquet_container"] is True]
        exact_parquet_trace_match = bool(
            len(parquet_files) == 1
            and len(valid_parquet) == 1
            and trace is not None
            and trace.is_file()
            and parquet_files[0].resolve() == trace.resolve()
        )
        complete_hash_chain = bool(
            trace_hash
            and len(declared_hash_values) == 3
            and all(value == trace_hash for value in declared_hash_values)
        )
        same_trace = bool(
            comparison.get("same_raw_trace_verified") is True
            and exact_parquet_trace_match
            and complete_hash_chain
            and valid_parquet[0].get("sha256") == trace_hash
        )
        evidence.append({
            "repeat_index": run_index,
            "collector_started": row.get("started_at") is not None,
            "collector_rc": row.get("collector_rc"),
            "workload_command_rc": row.get("workload_command_rc"),
            "comparison_path": str(comparison_path.resolve()),
            "comparison_relative_path": comparison_path.relative_to(measurement.parent).as_posix(),
            "comparison_sha256": _sha256(comparison_path) if comparison_path.is_file() else "",
            "comparison_status": comparison.get("status") or "missing",
            "same_raw_trace_verified": same_trace,
            "trace_path": str(trace.resolve()) if trace is not None and trace.is_file() else "",
            "trace_sha256": trace_hash,
            "request_before_after_hash_chain_complete": complete_hash_chain,
            "raw_parquet": parquet,
            "raw_parquet_count": len(parquet),
            "raw_parquet_valid_count": len(valid_parquet),
            "raw_parquet_exactly_one": len(parquet) == 1 and len(valid_parquet) == 1,
            "raw_parquet_matches_comparison_trace_path": exact_parquet_trace_match,
            "raw_parquet_present": exact_parquet_trace_match,
            "attempt_history": list(row.get("_probe_attempt_history") or []),
            "legacy_minus_command_window": comparison.get("legacy_minus_command_window"),
            "command_window": comparison.get("command_window"),
            "legacy_window": comparison.get("legacy_window"),
        })
    return evidence


def _attempt_failure_text(process: Mapping[str, Any], run_dir: Path | None) -> str:
    values = [
        str(process.get("error") or ""),
        str(process.get("stdout_tail") or ""),
        str(process.get("stderr_tail") or ""),
    ]
    if run_dir is not None:
        for name in (
            "collector_stderr.log", "collector_stdout.log",
            "workload_stderr.log", "workload_stdout.log",
        ):
            path = run_dir / name
            if path.is_file():
                values.append(path.read_text(encoding="utf-8", errors="replace")[-8000:])
    return "\n".join(values)


def _retryable_first_sample_barrier(text: str) -> bool:
    low = str(text or "").lower()
    return bool(
        "first-sample barrier failed" in low
        or ("channel is empty" in low and "sending half is closed" in low)
    )


_OUTER_ACQUISITION_RETRY_ERRORS = frozenset({
    # These are transport/acquisition-integrity failures emitted by the
    # marker-v2 collector.  Contract/schema/identity failures are deliberately
    # absent: retrying those would only hide a deterministic software error.
    "marker_dropped_samples_nonzero",
    "marker_trace_does_not_cover_window",
})


def _outer_acquisition_retry_reasons(
    process: Mapping[str, Any], row: Mapping[str, Any], run_dir: Path | None,
) -> list[str]:
    """Return the narrow allow-list for one fresh outer acquisition retry.

    The inner collector remains in exact-run mode with zero retries.  A caller
    may start one *new* collector process only for a transient stream-integrity
    failure.  Every failed physical attempt remains archived separately.
    """
    if bool(row.get("cancelled")) or str(row.get("status") or "").strip().lower() in {
        "cancelled", "preflight_failed",
    }:
        return []

    request = row.get("command_window_request")
    request = dict(request) if isinstance(request, Mapping) else {}
    errors = {
        str(value).strip().lower()
        for value in list(request.get("errors") or [])
        if str(value).strip()
    }
    # Some early collector exits write the rejection beside the aggregate but
    # do not manage to copy it into the one-row aggregate.  Read that structured
    # evidence as a fallback; do not infer marker failures from generic text.
    if run_dir is not None:
        rejection = _json(run_dir / "command_window_request_rejection.json")
        errors.update(
            str(value).strip().lower()
            for value in list(rejection.get("errors") or [])
            if str(value).strip()
        )

    reasons = sorted(errors & _OUTER_ACQUISITION_RETRY_ERRORS)
    if _retryable_first_sample_barrier(_attempt_failure_text(process, run_dir)):
        reasons.append("first_sample_barrier_invalid")
    return list(dict.fromkeys(reasons))


def _single_run_from_aggregate(out_dir: Path) -> tuple[dict[str, Any], Path | None]:
    aggregate = _json(out_dir / "energy_aggregate.json")
    runs = [dict(row) for row in list(aggregate.get("runs") or []) if isinstance(row, Mapping)]
    if len(runs) != 1:
        return {}, None
    row = runs[0]
    raw_dir = _resolve(row.get("storage_dir"), relative_to=out_dir)
    run_dir = raw_dir.parent if raw_dir is not None else out_dir / "run_000"
    return row, run_dir


from .energy.task_budget import add_campaign_budget_arguments, campaign_budget_forward_args


def run_probe(ns: argparse.Namespace) -> tuple[dict[str, Any], int]:
    out = Path(ns.out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    attempt_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        + "_" + uuid.uuid4().hex[:12]
    )
    # Every invocation receives a fresh output tree.  Existing failed or
    # completed traces remain immutable evidence and can never be mistaken for
    # a repeat of the current invocation.
    attempt = out / "attempts" / attempt_id
    attempt.mkdir(parents=True, exist_ok=False)
    repeats = max(1, int(ns.repeats or MIN_DECISION_REPEATS))
    report = _base_report(
        repeats=repeats,
        include_raw=bool(ns.include_raw_parquet),
        strict=bool(ns.strict),
    )
    report.update({
        "attempt_id": attempt_id,
        "attempt_directory": str(attempt),
        "fresh_attempt_directory": True,
        "stale_measurement_reuse_allowed": False,
        "report_path": str(out / "window_method_validation_probe.json"),
    })

    plan_dir = attempt / "plan"
    plan_cmd = [
        sys.executable, "-u", str(_script("native_producer_energy_plan.py")),
        "--summary", str(Path(ns.summary).expanduser().resolve()),
        "--out-dir", str(plan_dir),
        "--screening-window-probe", "--allow-unpaired",
        "--remote-tool-dir", ns.remote_tool_dir,
        "--remote-root", ns.remote_root,
        "--hailo8-ssh", ns.hailo8_ssh,
        "--hailo10-ssh", ns.hailo10_ssh,
        "--deepx-ssh", ns.deepx_ssh,
        "--hailo8-env", ns.hailo8_env,
        "--hailo10-env", ns.hailo10_env,
        "--deepx-env", ns.deepx_env,
        "--engine-build-python", ns.engine_build_python,
        "--duration-s", str(float(ns.duration_s)),
        "--timeout", str(int(ns.timeout)),
        "--runs", str(repeats),
        "--physical-scope", ns.physical_scope,
        "--window-label", ns.window_label,
        "--calibration-manifest", ns.calibration_manifest,
        "--calibration-sha256", ns.calibration_sha256,
    ]
    plan_cmd += campaign_budget_forward_args(ns)
    if getattr(ns, "hardware_setups_file", ""):
        plan_cmd += ["--hardware-setups-file", ns.hardware_setups_file]
    if ns.validation_summary:
        plan_cmd += ["--validation-summary", ns.validation_summary]
    plan_run = _run(plan_cmd, timeout=180)
    plan_path = plan_dir / "native_producer_energy_plan.json"
    plan = _json(plan_path)
    rows = [dict(row) for row in list(plan.get("rows") or []) if isinstance(row, Mapping)]
    runnable = [row for row in rows if str(row.get("measure_command") or "").strip()]
    report.update({
        "plan_process": plan_run,
        "plan_path": str(plan_path),
        "plan_sha256": _sha256(plan_path) if plan_path.is_file() else "",
        "representative_target_count": len(runnable),
        "representative_target": runnable[0] if runnable else None,
        "pairing_policy": "screening_probe_one_successful_native_target_no_pairing",
    })
    if plan_run.get("rc") != 0 or len(runnable) != 1:
        report.update({
            "ok": False,
            "complete": False,
            "status": "blocked_no_successful_representative_native_target",
            "blocked_reason": (
                "probe_plan_failed" if plan_run.get("rc") != 0
                else "probe_requires_exactly_one_runnable_native_target"
            ),
            "started_repeat_count": 0,
            "successful_comparison_count": 0,
            "decision_capable": False,
            "per_repeat": [],
        })
        _write_report(out, report)
        _write_probe_artifact_index(out)
        return report, 3 if ns.strict else 0

    source_summary_path = Path(ns.summary).expanduser().resolve()
    source_summary = _json(source_summary_path)
    source_row, source_status = _source_row_for_target(source_summary, runnable[0])
    contract, contract_status = (
        _verified_native_contract(source_row, runnable[0])
        if source_row is not None else (None, source_status)
    )
    report["exact_native_command_binding"] = {
        "source_summary": str(source_summary_path),
        "source_summary_sha256": _sha256(source_summary_path) if source_summary_path.is_file() else "",
        "source_row_status": source_status,
        "contract_status": contract_status,
        "contract_sha256": str((contract or {}).get("contract_sha256") or ""),
        "plan_generated_command_executed": False,
        "duration_and_fresh_output_paths_are_the_only_controlled_runtime_changes": True,
    }
    if contract is None:
        report.update({
            "ok": False,
            "complete": False,
            "status": "blocked_exact_native_command_contract_unavailable",
            "blocked_reason": contract_status,
            "started_repeat_count": 0,
            "successful_comparison_count": 0,
            "decision_capable": False,
            "per_repeat": [],
        })
        _write_report(out, report)
        _write_probe_artifact_index(out)
        return report, 3 if ns.strict else 0

    base_measure_cmd = shlex.split(str(runnable[0]["measure_command"]))
    measurement = attempt / "measurement"
    measurement.mkdir(parents=True, exist_ok=False)
    aggregate_runs: list[dict[str, Any]] = []
    measurement_processes: list[dict[str, Any]] = []
    repeat_bindings: list[dict[str, Any]] = []
    materialized_remote_output_roots: set[str] = set()
    abort_reason = ""
    requested_reconnect_retries = max(0, int(ns.max_reconnect_retries))
    # The exact-run collector itself still performs zero retries.  The probe
    # owns one bounded *outer* recovery capture for a narrow acquisition-
    # integrity allow-list, using a fresh local directory and remote root.
    effective_reconnect_retries = min(1, requested_reconnect_retries)
    if getattr(ns, "campaign_budget_file", None):
        effective_reconnect_retries = min(effective_reconnect_retries, ns.campaign_max_retries)
    reconnect_backoff_s = max(0.0, min(60.0, float(ns.reconnect_backoff_s)))
    outer_retry_attempt_count = 0
    outer_retry_recovered_count = 0

    for repeat_index in range(repeats):
        attempt_history: list[dict[str, Any]] = []
        final_row: dict[str, Any] = {}
        repeat_success = False
        max_attempts = 1 + effective_reconnect_retries
        for reconnect_attempt in range(max_attempts):
            backoff_record: dict[str, Any] | None = None
            if reconnect_attempt > 0:
                started_backoff = time.monotonic()
                if reconnect_backoff_s > 0:
                    time.sleep(reconnect_backoff_s)
                backoff_record = {
                    "requested_seconds": reconnect_backoff_s,
                    "elapsed_seconds": max(0.0, time.monotonic() - started_backoff),
                    "fresh_collector_process_required": True,
                }
            repeat_root = measurement / f"repeat_{repeat_index:03d}"
            attempt_out = repeat_root / f"attempt_{reconnect_attempt:02d}"
            command_file = repeat_root / f"workload_contract_attempt_{reconnect_attempt:02d}.sh"
            try:
                binding = _hailo8_replay_command(
                    contract=contract, ns=ns, attempt_id=attempt_id,
                    repeat_index=repeat_index, reconnect_attempt=reconnect_attempt,
                    command_file=command_file,
                )
            except Exception as exc:
                final_row = {
                    "run_index": repeat_index,
                    "_probe_run_dir": str(attempt_out / "run_000"),
                    "_probe_attempt_history": attempt_history,
                    "probe_command_binding_failed": True,
                    "probe_command_binding_error": f"{type(exc).__name__}: {exc}",
                }
                abort_reason = "exact_native_command_materialization_failed"
                break
            repeat_bindings.append({
                "repeat_index": repeat_index,
                "attempt_index": reconnect_attempt,
                **binding,
            })
            remote_output_root = str(binding.get("remote_output_root") or "")
            if not remote_output_root or remote_output_root in materialized_remote_output_roots:
                final_row = {
                    "run_index": repeat_index,
                    "_probe_run_dir": str(attempt_out / "run_000"),
                    "_probe_attempt_history": attempt_history,
                    "probe_command_binding_failed": True,
                    "probe_command_binding_error": "remote_output_root_missing_or_reused",
                }
                abort_reason = "remote_output_root_uniqueness_contract_failed"
                break
            materialized_remote_output_roots.add(remote_output_root)
            if getattr(ns, "campaign_budget_file", None):
                remote_preflight = {"rc": 0, "status": "deferred_to_reserved_measurement_chain"}
            else:
                remote_preflight = _run(_hailo8_remote_preflight(contract, ns), timeout=180)
            if remote_preflight.get("rc") != 0:
                attempt_history.append({
                    "attempt_index": reconnect_attempt,
                    "remote_preflight": remote_preflight,
                    "measurement_started": False,
                    "retryable": False,
                })
                final_row = {
                    "run_index": repeat_index,
                    "_probe_run_dir": str(attempt_out / "run_000"),
                    "_probe_attempt_history": attempt_history,
                    "probe_preflight_failed": True,
                }
                abort_reason = "remote_workload_contract_preflight_failed"
                break

            measure_cmd = list(base_measure_cmd)
            measure_cmd = _option(measure_cmd, "--runs", "1")
            if "--exact-run-count" not in measure_cmd:
                measure_cmd.append("--exact-run-count")
            measure_cmd = _option(
                measure_cmd, "--invalid-repeat-max-retries", "0"
            )
            measure_cmd = _option(measure_cmd, "--out", str(attempt_out))
            measure_cmd = _option(measure_cmd, "--command-file", str(command_file))
            measure_cmd = _option(
                measure_cmd, "--run-id",
                f"window_probe_{attempt_id}_r{repeat_index:03d}_a{reconnect_attempt:02d}",
            )
            if getattr(ns, "campaign_budget_file", None):
                row_id = base_measure_cmd[base_measure_cmd.index("--campaign-row-id") + 1]
                measure_cmd = _option(measure_cmd, "--campaign-row-id", "window_probe:" + row_id)
                measure_cmd = _option(measure_cmd, "--task-logical-repeat", f"repeat:{repeat_index}")
                measure_cmd = _option(measure_cmd, "--preflight-prepare-command", shlex.join(_hailo8_remote_preflight(contract, ns)))
                measure_cmd = _option(measure_cmd, "--preflight-timeout-s", "180")
            process = _run(measure_cmd, timeout=float(ns.timeout) + 600.0)
            row, run_dir = _single_run_from_aggregate(attempt_out)
            if run_dir is None:
                run_dir = attempt_out / "run_000"
            if getattr(ns, "campaign_budget_file", None):
                remote_preflight = row.get("preflight_prepare_result") or {"status": "NOT_RUN", "reason": "campaign_budget_or_preflight_blocked"}
            row.update({
                "run_index": repeat_index,
                "_probe_run_dir": str(run_dir),
            })
            provisional = _repeat_evidence(measurement, {"runs": [row]})
            evidence = provisional[0] if provisional else {}
            repeat_success = bool(
                evidence.get("comparison_status") == "ok"
                and evidence.get("same_raw_trace_verified") is True
                and evidence.get("raw_parquet_present") is True
            )
            retry_reasons = (
                [] if repeat_success
                else _outer_acquisition_retry_reasons(process, row, run_dir)
            )
            retryable = bool(
                not repeat_success
                and retry_reasons
                and reconnect_attempt + 1 < max_attempts
            )
            history_row = {
                "attempt_index": reconnect_attempt,
                "remote_preflight": remote_preflight,
                "measurement_process": process,
                "measurement_directory": str(attempt_out),
                "run_directory": str(run_dir),
                "comparison_complete": repeat_success,
                "acquisition_integrity_retry_reasons": retry_reasons,
                "retryable_acquisition_integrity_failure": retryable,
                "retry_backoff_before_attempt": backoff_record,
                # Preserve the complete one-row aggregate for every physical
                # attempt.  Only the final successful attempt may represent the
                # logical repeat in the screening aggregate below.
                "physical_attempt_result": dict(row),
            }
            attempt_history.append(history_row)
            measurement_processes.append({
                "repeat_index": repeat_index,
                "attempt_index": reconnect_attempt,
                **process,
            })
            final_row = row
            if repeat_success:
                if reconnect_attempt > 0:
                    outer_retry_recovered_count += 1
                break
            if retryable:
                outer_retry_attempt_count += 1
                continue
            abort_reason = (
                "acquisition_integrity_retry_exhausted:"
                + ",".join(retry_reasons)
                if retry_reasons
                else "non_retryable_collector_workload_or_comparison_failure"
            )
            break

        final_row["_probe_attempt_history"] = attempt_history
        aggregate_runs.append(final_row)
        if not repeat_success:
            # Later independent repeats are not scientifically useful after a
            # failed one and must not blindly reconnect to the same firmware.
            break

    aggregate_path = measurement / "energy_aggregate.json"
    aggregate = {
        "schema": "onnx-splitpoint/window-method-validation-probe-repeat-aggregate",
        "schema_version": 1,
        "requested_repeat_count": repeats,
        "started_repeat_count": len(aggregate_runs),
        "fail_fast": True,
        "requested_reconnect_retries": requested_reconnect_retries,
        "max_reconnect_retries": effective_reconnect_retries,
        "max_physical_collector_attempts_per_outer_repeat": (
            1 + effective_reconnect_retries
        ),
        "valid_capture_count_required_per_outer_repeat": 1,
        "reconnect_retry_clamped_to_one": requested_reconnect_retries > 1,
        "reconnect_retry_suppressed_by_exact_repeat_contract": False,
        "reconnect_backoff_s": reconnect_backoff_s,
        "outer_retry_attempt_count": outer_retry_attempt_count,
        "outer_retry_recovered_count": outer_retry_recovered_count,
        "abort_reason": abort_reason,
        "runs": aggregate_runs,
    }
    aggregate_path.write_text(json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8")
    aggregate_path = measurement / "energy_aggregate.json"
    per_repeat = _repeat_evidence(measurement, aggregate)
    started = sum(1 for row in per_repeat if row.get("collector_started"))
    successful = sum(
        1 for row in per_repeat
        if row.get("comparison_status") == "ok"
        and row.get("same_raw_trace_verified") is True
        and row.get("raw_parquet_present") is True
    )
    raw_complete = bool(per_repeat) and all(row.get("raw_parquet_present") is True for row in per_repeat)
    complete = bool(
        started == repeats
        and len(per_repeat) == repeats
        and successful == repeats
        and raw_complete
    )
    decision_capable = bool(complete and repeats >= MIN_DECISION_REPEATS)
    if complete and decision_capable:
        status = "screening_complete_sensitivity_summary_capable"
    elif complete:
        status = "screening_complete_sensitivity_only_too_few_repeats"
    elif started == 0:
        status = "blocked_zero_measurements_started"
    else:
        status = "incomplete_probe_measurement_or_comparison_failed"
    report.update({
        "ok": complete,
        "complete": complete,
        "status": status,
        "blocked_reason": (
            "zero_measurements_started" if started == 0
            else "" if complete
            else "one_or_more_repeats_lack_a_hash_verified_A/B_comparison_and_raw_trace"
        ),
        "decision_capable": decision_capable,
        "sensitivity_summary_capable": decision_capable,
        "method_selection_pending": False,
        "method_selection_affected": False,
        "decision_capability_reason": (
            "sensitivity_complete_with_minimum_repeats" if decision_capable
            else "fewer_than_three_repeats" if complete
            else "probe_incomplete"
        ),
        "measurement_processes": measurement_processes,
        "exact_replay_bindings": repeat_bindings,
        "repeat_execution_policy": {
            "mode": "sequential_exact_repeat_bounded_outer_acquisition_retry",
            "repeat_control": "caller_managed_exact",
            "collector_runs_per_outer_repeat": 1,
            "collector_runs_per_physical_attempt": 1,
            "max_physical_collector_attempts_per_outer_repeat": (
                1 + effective_reconnect_retries
            ),
            "valid_capture_count_required_per_outer_repeat": 1,
            "collector_invalid_repeat_retries": 0,
            "unique_remote_output_root_required": True,
            "unique_remote_output_root_count": len(materialized_remote_output_roots),
            "requested_reconnect_retries": requested_reconnect_retries,
            "max_reconnect_retries": effective_reconnect_retries,
            "reconnect_retry_clamped_to_one": requested_reconnect_retries > 1,
            "reconnect_retry_suppressed_by_exact_repeat_contract": False,
            "reconnect_backoff_s": reconnect_backoff_s,
            "outer_retry_attempt_count": outer_retry_attempt_count,
            "outer_retry_recovered_count": outer_retry_recovered_count,
            "outer_retry_failure_allowlist": sorted(
                _OUTER_ACQUISITION_RETRY_ERRORS | {"first_sample_barrier_invalid"}
            ),
            "abort_reason": abort_reason,
            "skipped_repeat_count": max(0, repeats - len(per_repeat)),
            "blind_followup_repeats_allowed": False,
        },
        "measurement_directory": str(measurement),
        "measurement_aggregate_path": str(aggregate_path),
        "measurement_aggregate_sha256": _sha256(aggregate_path) if aggregate_path.is_file() else "",
        "started_repeat_count": started,
        "materialized_repeat_count": len(per_repeat),
        "skipped_repeat_count": max(0, repeats - len(per_repeat)),
        "successful_comparison_count": successful,
        "raw_trace_evidence_complete": raw_complete,
        "per_repeat": per_repeat,
    })
    _write_report(out, report)
    _write_probe_artifact_index(out)
    if not complete and ns.strict:
        return report, 4
    return report, 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_campaign_budget_arguments(parser)
    parser.add_argument("--hardware-setups-file", default="")
    parser.add_argument("--summary", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--validation-summary", default="")
    parser.add_argument("--hailo8-ssh", default="")
    parser.add_argument("--hailo10-ssh", default="")
    parser.add_argument("--deepx-ssh", default="")
    parser.add_argument("--hailo8-env", default="")
    parser.add_argument("--hailo10-env", default="")
    parser.add_argument("--deepx-env", default="")
    parser.add_argument("--engine-build-python", default="auto")
    parser.add_argument("--remote-tool-dir", default="/home/nx/ONNX-Splitpoint-Tool")
    parser.add_argument("--remote-root", default="/home/nx/native_fifo_evalsets")
    parser.add_argument("--duration-s", type=float, default=60.0)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--repeats", type=int, default=MIN_DECISION_REPEATS)
    parser.add_argument("--max-reconnect-retries", type=int, default=0, help="Maximum fresh outer collector retries per logical repeat (clamped to 1). Inner exact-run retries remain disabled.")
    parser.add_argument("--reconnect-backoff-s", type=float, default=5.0, help="Delay before the one permitted fresh outer collector retry (clamped to 0..60 seconds).")
    parser.add_argument("--physical-scope", default="")
    parser.add_argument("--window-label", default="command")
    parser.add_argument("--calibration-manifest", default="")
    parser.add_argument("--calibration-sha256", default="")
    strict = parser.add_mutually_exclusive_group()
    strict.add_argument("--strict", dest="strict", action="store_true")
    strict.add_argument("--no-strict", dest="strict", action="store_false")
    raw = parser.add_mutually_exclusive_group()
    raw.add_argument("--include-raw-parquet", dest="include_raw_parquet", action="store_true")
    raw.add_argument("--no-include-raw-parquet", dest="include_raw_parquet", action="store_false")
    parser.set_defaults(strict=True, include_raw_parquet=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    report, rc = run_probe(build_parser().parse_args(argv))
    print(json.dumps({
        "ok": report.get("ok"),
        "complete": report.get("complete"),
        "status": report.get("status"),
        "decision_capable": report.get("decision_capable"),
        "started_repeat_count": report.get("started_repeat_count"),
        "successful_comparison_count": report.get("successful_comparison_count"),
        "report": str(report.get("report_path") or "window_method_validation_probe.json"),
    }, indent=2), flush=True)
    return int(rc)


if __name__ == "__main__":
    raise SystemExit(main())
