from __future__ import annotations

import json
import csv
import functools
import hashlib
import os
import random
import re
import shlex
import tarfile
import time
import threading
import textwrap
import posixpath
import traceback
import math
from datetime import datetime, timezone
from dataclasses import dataclass, field
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, List, Sequence

from onnx_splitpoint_tool.cache_verify_policy import (
    cache_miss_blocked_message,
    compiler_dispatch_forbidden,
)
from onnx_splitpoint_tool.filesystem_admission import require_write_target
from onnx_splitpoint_tool.energy.comparison import resolve_energy_comparison

HAILO_DISCOVERY_SHELL = '\nHAILO_PY=""\nHAILO_SITE=""\n_hailo_has_py_bindings() {\n  _py="$1"\n  [ -n "$_py" ] && [ -x "$_py" ] || return 1\n  "$_py" - <<\'HAILOPY\' >/dev/null 2>&1\ntry:\n    import hailo_platform  # noqa: F401\nexcept Exception:\n    try:\n        import hailort  # noqa: F401\n    except Exception:\n        raise SystemExit(1)\nraise SystemExit(0)\nHAILOPY\n}\n_hailo_site_dirs() {\n  _py="$1"\n  "$_py" - <<\'HAILOSITEPY\' 2>/dev/null || true\nimport os, site, sysconfig\npaths=[]\ngetsp=getattr(site, \'getsitepackages\', None)\nif getsp:\n    try: paths.extend(getsp() or [])\n    except Exception: pass\ntry:\n    p=site.getusersitepackages()\n    if p: paths.append(p)\nexcept Exception:\n    pass\nfor key in (\'purelib\',\'platlib\'):\n    try:\n        p=sysconfig.get_paths().get(key)\n        if p: paths.append(p)\n    except Exception:\n        pass\nout=[]\nfor p in paths:\n    if p and os.path.isdir(p) and p not in out:\n        out.append(p)\nprint(\':\'.join(out))\nHAILOSITEPY\n}\nif [ "$WANT_HAILO" = "1" ]; then\n  _CANDIDATE_PYS="${ENV_PY:-} ${SYS_PY:-} $HOME/hailo_py/bin/python3 $HOME/hailo_py/bin/python $HOME/hailo_venv/bin/python3 $HOME/hailo_venv/bin/python $HOME/hailort_venv/bin/python3 $HOME/hailort_venv/bin/python $HOME/.venvs/hailo/bin/python3 $HOME/.venvs/hailo/bin/python $HOME/.venvs/hailort/bin/python3 $HOME/.venvs/hailort/bin/python"\n  for _c in $_CANDIDATE_PYS; do\n    if _hailo_has_py_bindings "$_c"; then HAILO_PY="$_c"; break; fi\n  done\n  if [ -z "$HAILO_PY" ]; then\n    for _c in $(find "$HOME" /opt -maxdepth 4 \\( -path \'*/bin/python\' -o -path \'*/bin/python3\' \\) 2>/dev/null | grep -Ei \'hailo|hailort|hailomz\' | head -n 40); do\n      if _hailo_has_py_bindings "$_c"; then HAILO_PY="$_c"; break; fi\n    done\n  fi\n  if [ -n "$HAILO_PY" ]; then\n    HAILO_SITE="$(_hailo_site_dirs "$HAILO_PY")"\n    if [ -n "$HAILO_SITE" ]; then\n      if [ -n "${SPLITPOINT_EXTRA_SITES:-}" ]; then\n        export SPLITPOINT_EXTRA_SITES="${SPLITPOINT_EXTRA_SITES}:$HAILO_SITE"\n      else\n        export SPLITPOINT_EXTRA_SITES="$HAILO_SITE"\n      fi\n    fi\n  fi\nfi\nexport HAILO_PY HAILO_SITE\nexport PRECHECK_HAILO_PY="$HAILO_PY"\nexport PRECHECK_HAILO_SITE="$HAILO_SITE"\n'

DEEPX_DISCOVERY_SHELL = '\nDEEPX_PY=""\nDEEPX_SITE=""\n_deepx_has_py_bindings() {\n  _py="$1"\n  [ -n "$_py" ] && [ -x "$_py" ] || return 1\n  "$_py" - <<\'DEEPXPY\' >/dev/null 2>&1\ntry:\n    import dx_engine  # noqa: F401\nexcept Exception:\n    raise SystemExit(1)\nraise SystemExit(0)\nDEEPXPY\n}\n_deepx_site_dirs() {\n  _py="$1"\n  "$_py" - <<\'DEEPXSITEPY\' 2>/dev/null || true\nimport os, site, sysconfig\npaths=[]\ngetsp=getattr(site, \'getsitepackages\', None)\nif getsp:\n    try: paths.extend(getsp() or [])\n    except Exception: pass\ntry:\n    p=site.getusersitepackages()\n    if p: paths.append(p)\nexcept Exception:\n    pass\nfor key in (\'purelib\',\'platlib\'):\n    try:\n        p=sysconfig.get_paths().get(key)\n        if p: paths.append(p)\n    except Exception:\n        pass\nout=[]\nfor p in paths:\n    if p and os.path.isdir(p) and p not in out:\n        out.append(p)\nprint(\':\'.join(out))\nDEEPXSITEPY\n}\nif [ "${WANT_DEEPX:-0}" = "1" ]; then\n  _CANDIDATE_PYS="${ENV_PY:-} ${SYS_PY:-} $HOME/venvs/deepx-runtime/bin/python3 $HOME/venvs/deepx-runtime/bin/python $HOME/venvs/deepx-yolo/bin/python3 $HOME/venvs/deepx-yolo/bin/python $HOME/venvs/splitpoint-deepx/bin/python3 $HOME/venvs/splitpoint-deepx/bin/python $HOME/deepx-yolo/bin/python3 $HOME/deepx-yolo/bin/python $HOME/.venvs/deepx/bin/python3 $HOME/.venvs/deepx/bin/python $HOME/.venvs/deepx-yolo/bin/python3 $HOME/.venvs/deepx-yolo/bin/python"\n  for _c in $_CANDIDATE_PYS; do\n    if _deepx_has_py_bindings "$_c"; then DEEPX_PY="$_c"; break; fi\n  done\n  if [ -z "$DEEPX_PY" ]; then\n    for _c in $(find "$HOME" /opt -maxdepth 5 \\( -path \'*/bin/python\' -o -path \'*/bin/python3\' \\) 2>/dev/null | grep -Ei \'deepx|dxrt|deepx-runtime|splitpoint-deepx|deepx-yolo\' | head -n 40); do\n      if _deepx_has_py_bindings "$_c"; then DEEPX_PY="$_c"; break; fi\n    done\n  fi\n  if [ -n "$DEEPX_PY" ]; then\n    DEEPX_SITE="$(_deepx_site_dirs "$DEEPX_PY")"\n    if [ -n "$DEEPX_SITE" ]; then\n      if [ -n "${SPLITPOINT_EXTRA_SITES:-}" ]; then\n        export SPLITPOINT_EXTRA_SITES="${SPLITPOINT_EXTRA_SITES}:$DEEPX_SITE"\n      else\n        export SPLITPOINT_EXTRA_SITES="$DEEPX_SITE"\n      fi\n    fi\n    if [ -z "${ENV_PY:-}" ] || [ "${ENV_PY:-}" = "${SYS_PY:-}" ]; then\n      ENV_PY="$DEEPX_PY"\n    fi\n  fi\nfi\nexport DEEPX_PY DEEPX_SITE ENV_PY\nexport PRECHECK_DEEPX_PY="$DEEPX_PY"\nexport PRECHECK_DEEPX_SITE="$DEEPX_SITE"\n'


from onnx_splitpoint_tool.remote.bundle import BundleCancelled, build_suite_bundle, remote_minimal_bundle_patterns
from onnx_splitpoint_tool.benchmark.results_bundle import create_results_bundle_from_results_dir
from onnx_splitpoint_tool.log_utils import sanitize_log
from onnx_splitpoint_tool.remote.ssh_transport import HostConfig as RemoteHost
from onnx_splitpoint_tool.remote.ssh_transport import SSHTransport
from onnx_splitpoint_tool.remote.process_lease import (
    REMOTE_LEASE_OUTER_CLEANUP_MARGIN_S,
    RemoteProcessLeaseRegistry,
    RemoteProcessLeaseJournalError,
    journaled_ssh_wrapper_argv,
    resolve_remote_process_lease_scope,
)
from onnx_splitpoint_tool.benchmark.suite_refresh import (
    refresh_suite_harness,
    normalize_benchmark_task,
    normalize_mini_classification_eval,
    normalize_semantic_validation_request,
)


RUN_STATUS_SCHEMA_VERSION = 1
TERMINAL_REMOTE_CLEANUP_RC = 70
_TERMINAL_REMOTE_STORAGE_TOKENS = (
    "no space left on device",
    "errno 28",
    "read-only file system",
    "errno 30",
    "disk quota exceeded",
)

_REMOTE_CONNECTIVITY_FAILURE_TOKENS = (
    "no route to host",
    "network is unreachable",
    "connection timed out",
    "timeout after",
    "connection refused",
    "could not resolve hostname",
    "temporary failure in name resolution",
    "name or service not known",
)

_REMOTE_HOST_STORAGE_LOCKS_GUARD = threading.Lock()
_REMOTE_HOST_STORAGE_LOCKS: dict[tuple[str, str], threading.Lock] = {}


def _is_terminal_remote_storage_error(value: object) -> bool:
    text = str(value or "").lower()
    return any(token in text for token in _TERMINAL_REMOTE_STORAGE_TOKENS)


def _is_remote_connectivity_failure(value: object) -> bool:
    """Return whether a pre-mutation SSH probe could not reach its host.

    This classifier is intentionally narrow.  It is used only before a leased
    command or remote mutation has started, where an unreachable host proves
    that there is no remote benchmark process to collect or clean up.  Once a
    leased operation starts, the exact-cleanup/quarantine contract remains the
    sole authority.
    """

    text = str(value or "").lower()
    return any(token in text for token in _REMOTE_CONNECTIVITY_FAILURE_TOKENS)


def _remote_failure_rc(value: object, *, default: int = 1) -> int:
    """Recover an SSH-style return code from a preserved exception string."""

    match = re.search(r"\brc\s*=\s*(\d+)\b", str(value or ""), re.IGNORECASE)
    if match is None:
        return int(default)
    try:
        return int(match.group(1))
    except (TypeError, ValueError):
        return int(default)


def _remote_preflight_failure(rc: int, output: object) -> RuntimeError:
    """Keep preflight stderr in the exception used by terminal classification."""

    tail = str(output or "").strip()[-4000:]
    category = (
        "terminal_remote_storage_failure"
        if _is_terminal_remote_storage_error(tail)
        else "remote_runtime_preflight_failure"
    )
    return RuntimeError(
        f"{category}: Remote preflight failed (rc={int(rc)}): "
        + (tail or "no remote output")
    )


def _remote_command_failure_prefix(*, stage: str, output: object) -> str:
    """Mark capacity/retention rejection as terminal storage admission.

    Retention deliberately exits 75 when safe pruning cannot establish its
    bounded contract.  Its inventory need not contain an ENOSPC token, so the
    stage itself is authoritative.  The outer failure path recognizes this
    prefix and suppresses all result collection, packaging and SCP writes.
    """

    if stage == "remote_trt_cache_retention" or _is_terminal_remote_storage_error(output):
        return "remote_storage_preflight_failed: "
    return ""


def _terminal_remote_storage_stage_payload(
    *, stage: str, rc: int | None, output: object,
) -> dict[str, Any]:
    """Classify storage exhaustion from any remote stage, including rc=0.

    Several best-effort collection commands intentionally use ``|| true``.
    Their return code therefore cannot be the admission signal; stderr tokens
    are authoritative and must stop every later remote write/download step.
    """

    lines = [line for line in str(output or "").splitlines() if line.strip()]
    if not _is_terminal_remote_storage_error("\n".join(lines)):
        return {}
    return _terminal_remote_failure_payload(
        remote_rc=rc,
        recent_remote_lines=lines,
        fallback_error=f"Remote storage failure during {stage} (rc={rc})",
        occurred_at=_utc_now_iso(),
        force_terminal=True,
        failure_kind="terminal_remote_storage_failure",
    )


def _serialized_remote_storage_by_host(function):
    """Serialize complete remote runs per physical host.

    A read-only statvfs probe cannot reserve blocks.  Holding this lock for the
    complete remote run prevents two logical workers in this process from both
    admitting the same free blocks and later overbooking them.  Different
    physical hosts remain fully parallel.
    """

    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        host = kwargs.get("host")
        host_name = str(getattr(host, "host", "") or "").strip().lower()
        host_user = str(getattr(host, "user", "") or "").strip().lower()
        key = (host_name, host_user)
        with _REMOTE_HOST_STORAGE_LOCKS_GUARD:
            lock = _REMOTE_HOST_STORAGE_LOCKS.setdefault(key, threading.Lock())
        cancel_event = kwargs.get("cancel_event")
        while not lock.acquire(timeout=0.25):
            if cancel_event is not None and cancel_event.is_set():
                raise BundleCancelled("cancelled while waiting for remote storage admission")
        try:
            return function(*args, **kwargs)
        finally:
            lock.release()

    return wrapped


def _terminal_remote_failure_payload(
    *,
    remote_rc: int | None,
    recent_remote_lines: list[str],
    fallback_error: str = "",
    occurred_at: str = "",
    force_terminal: bool = False,
    failure_kind: str = "",
) -> dict[str, Any]:
    """Preserve the first remote error when exact lease cleanup is unproven.

    ``SSHTransport`` deliberately returns rc=70 when a failed remote command
    could not be cleaned up exactly and the shared lease registry was poisoned.
    That is a workflow-wide terminal infrastructure failure.  Keep the remote
    stderr intact here so upper layers can report it once instead of replacing
    it with dozens of downstream "missing" errors.
    """

    if int(remote_rc or 0) != TERMINAL_REMOTE_CLEANUP_RC and not force_terminal:
        return {}

    synthetic_markers = (
        "cleanup unproven",
        "lease session poisoned",
        "remote benchmark failed (rc=70)",
    )
    context = [
        str(line).rstrip()
        for line in list(recent_remote_lines or [])[-50:]
        if str(line).strip()
    ]
    original_lines = [
        line for line in context
        if not any(marker in line.lower() for marker in synthetic_markers)
    ]
    preferred_tokens = (
        "no space left on device",
        "errno 28",
        "error",
        "exception",
        "failed",
    )
    primary_error = ""
    for token in preferred_tokens:
        for line in original_lines:
            if token in line.lower():
                primary_error = line.strip()
                break
        if primary_error:
            break
    if not primary_error and original_lines:
        primary_error = original_lines[-1].strip()
    if not primary_error:
        primary_error = str(fallback_error or "").strip()
    if not primary_error:
        primary_error = (
            "Remote execution failed and exact cleanup could not be proven "
            f"(rc={TERMINAL_REMOTE_CLEANUP_RC})."
        )

    return {
        "schema": "onnx-splitpoint/terminal-remote-failure",
        "schema_version": 1,
        "terminal_remote_failure": True,
        "failure_kind": str(
            failure_kind or "terminal_remote_execution_failure"
        ),
        "remote_rc": remote_rc,
        "primary_error": primary_error,
        "primary_error_context": original_lines,
        "primary_error_at": str(occurred_at or _utc_now_iso()),
    }


class _TerminalRemoteCollectionSuppressed(RuntimeError):
    """Internal control signal: the poisoned lease forbids more remote I/O."""


def _journaled_energy_ssh_command(
    transport: SSHTransport,
    remote_process_registry: RemoteProcessLeaseRegistry | None,
    remote_command: str,
    *,
    label: str,
    timeout_s: float | None,
) -> tuple[str, dict[str, str] | None]:
    """Build one repeat-safe energy SSH command under the parent run lease.

    The returned command invokes the journal broker anew every time u.RECS or
    a duration probe executes its command script.  Thus each retry/repeat gets
    a fresh exact remote operation instead of reusing a stale lease token.
    """

    raw_argv = transport._ssh_cmd(remote_command)
    lease_env: dict[str, str] | None = None
    if remote_process_registry is not None:
        if remote_process_registry.configured_scope is None:
            if getattr(transport, "_remote_lease_scope", None) is not None:
                raise RemoteProcessLeaseJournalError(
                    "remote energy execution requires a configured lease journal"
                )
        else:
            lease_env = remote_process_registry.journal_environment()
    elif getattr(transport, "_remote_lease_scope", None) is not None:
        raise RemoteProcessLeaseJournalError(
            "remote energy execution cannot cross a process boundary without "
            "the parent lease registry journal"
        )
    broker_timeout_s: float | None = None
    if timeout_s is not None:
        outer_timeout_s = max(0.1, float(timeout_s))
        # Reserve the bounded cleanup budget plus scheduler/teardown slack.
        cleanup_budget_s = min(
            REMOTE_LEASE_OUTER_CLEANUP_MARGIN_S,
            max(0.0, outer_timeout_s - 0.1),
        )
        broker_timeout_s = max(0.1, outer_timeout_s - cleanup_budget_s)
    wrapped_argv = journaled_ssh_wrapper_argv(
        raw_argv,
        label=label,
        env=lease_env,
        timeout_s=broker_timeout_s,
    )
    return " ".join(shlex.quote(str(value)) for value in wrapped_argv), lease_env
RUN_RESULTS_SCHEMA_VERSION = 1
_CASE_DIR_RE = re.compile(r"^b\d+$")

_UPLOAD_SEMAPHORES: dict[int, threading.Semaphore] = {}
_UPLOAD_SEMAPHORES_LOCK = threading.Lock()
_SUITE_REFRESH_LOCKS: dict[str, threading.RLock] = {}
_SUITE_REFRESH_LOCKS_GUARD = threading.Lock()


def _suite_refresh_lock(suite_dir: Path) -> threading.RLock:
    key = str(Path(suite_dir).expanduser().resolve())
    with _SUITE_REFRESH_LOCKS_GUARD:
        lock = _SUITE_REFRESH_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _SUITE_REFRESH_LOCKS[key] = lock
        return lock


@contextmanager
def _exclusive_suite_refresh_guard(
    suite_dir: Path,
    *,
    log: Optional[Callable[[str], None]] = None,
    cancel_event: Any = None,
):
    """Serialize in-place suite refreshes before parallel setup packaging.

    All physical setup workers share one generated suite.  The harness refresh
    can rewrite ``benchmark_plan.json`` and materialise the run-mode validation
    subset.  Running it concurrently allowed one worker to package while another
    was still modifying the suite, producing different archive hashes for the
    same model.  The guard is process-local because concurrent setup workers live
    in one evaluation process; independent EvaluationRuns use different suite
    directories.
    """
    lock = _suite_refresh_lock(suite_dir)
    started = time.monotonic()
    announced = False
    while True:
        if cancel_event is not None and getattr(cancel_event, "is_set", lambda: False)():
            raise RuntimeError("cancelled while waiting for shared suite refresh")
        if lock.acquire(timeout=0.25):
            break
        if not announced and time.monotonic() - started >= 1.0:
            if callable(log):
                log("[suite-refresh] waiting for sibling setup to finish shared suite refresh")
            announced = True
    try:
        if callable(log):
            elapsed = time.monotonic() - started
            if elapsed >= 1.0:
                log(f"[suite-refresh] shared refresh lock acquired after {elapsed:.1f}s")
        yield
    finally:
        lock.release()


def _remote_upload_semaphore() -> threading.Semaphore | None:
    """Global per-process upload limiter for parallel remote setup runs.

    Multiple independent Orin/u.RECS setups may execute benchmarks in parallel,
    but uploading 3 huge suite bundles at once is wasteful and can make the GUI
    look hung.  The default keeps uploads serial while allowing execution/energy
    windows to overlap across setups.  Set ONNX_SPLITPOINT_MAX_PARALLEL_UPLOADS=0
    to disable the limiter or >1 to allow more concurrent uploads.
    """
    try:
        limit = int(float(str(os.environ.get("ONNX_SPLITPOINT_MAX_PARALLEL_UPLOADS", "1") or "1")))
    except Exception:
        limit = 1
    if limit <= 0:
        return None
    with _UPLOAD_SEMAPHORES_LOCK:
        sem = _UPLOAD_SEMAPHORES.get(limit)
        if sem is None:
            sem = threading.Semaphore(limit)
            _UPLOAD_SEMAPHORES[limit] = sem
        return sem


def _utc_now_iso() -> str:
    """UTC timestamp in ISO-8601 with 'Z'."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with open(path, "w", encoding="utf-8"):
            pass


def _read_tail_lines(path: Path, *, max_lines: int = 100, max_bytes: int = 64_000) -> list[str]:
    """Read last N lines from a text file with a hard byte cap.

    Returns [] if the file doesn't exist.
    """
    if not path.exists():
        return []
    try:
        data = path.read_bytes()
        if len(data) > max_bytes:
            data = data[-max_bytes:]
        text = data.decode("utf-8", errors="replace")
        lines = text.splitlines()
        if len(lines) > max_lines:
            lines = lines[-max_lines:]
        return lines
    except Exception:
        return []


def _detect_useful_results(results_dir: Path) -> bool:
    """Return true only for structured, non-empty resumable result evidence.

    Log files, empty JSON objects and image/PDF renderings are diagnostics, not
    checkpoints.  Accepting any non-empty file here caused failed setup stubs to
    be reopened and overwritten by a later automatic resume.
    """
    if not results_dir.is_dir():
        return False

    json_candidates: list[Path] = []
    json_candidates.extend(results_dir.glob("benchmark_results_*.json"))
    json_candidates.extend(results_dir.glob("validation_report.json"))
    json_candidates.extend(results_dir.glob("*/results_*/validation_report.json"))
    for path in sorted(set(json_candidates)):
        if path.is_symlink() or not path.is_file():
            continue
        payload = _read_json_dict(path)
        if not payload:
            continue
        rows = payload.get("results")
        if isinstance(rows, list) and any(isinstance(row, dict) and row for row in rows):
            return True
        validation_rows = payload.get("rows")
        if isinstance(validation_rows, list) and any(
            isinstance(row, dict) and row for row in validation_rows
        ):
            return True
        status = str(payload.get("status") or "").strip().lower()
        if status in {"ok", "pass", "passed", "partial", "failed"} and any(
            key in payload
            for key in (
                "validated_result_count",
                "validation_results",
                "adapter_counts",
                "metrics",
            )
        ):
            return True
    return False


def _assert_generated_runner_is_self_consistent(path: Path) -> None:
    """Reject generated runner scripts that are obviously stale or broken.

    This catches partial refreshes before we upload a large bundle to the remote host.
    The checks stay intentionally static/lightweight so they do not require importing
    heavyweight runtime dependencies such as onnxruntime or Hailo bindings.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as e:
        raise RuntimeError(f"Could not read generated runner for self-check: {path}: {e}") from e

    try:
        compile(text, str(path), "exec")
    except SyntaxError as e:
        raise RuntimeError(f"{path} failed syntax self-check: {e}") from e

    if "class HailoInferModelSession" in text:
        from ..split_export_runners import assert_generated_hailo_layout_current
        assert_generated_hailo_layout_current(path)

    required_helpers = (
        "_maybe_cast_for_onnx_input",
        "_shape_from_ort_input",
    )
    missing: list[str] = []
    for helper_name in required_helpers:
        referenced = helper_name in text
        defined = f"def {helper_name}(" in text
        if referenced and not defined:
            missing.append(helper_name)

    if missing:
        raise RuntimeError(
            f"{path} references helper(s) {', '.join(missing)} but does not define them. "
            "Refusing to package a stale or broken runner."
        )

    suite_runtime_import = text.find(
        "from splitpoint_runners.native_split_quality_runtime import"
    )
    suite_path_bootstrap = text.find(
        "\n_maybe_add_suite_runtime_to_syspath()\n"
    )
    if suite_runtime_import >= 0 and (
        suite_path_bootstrap < 0 or suite_path_bootstrap > suite_runtime_import
    ):
        raise RuntimeError(
            f"{path} imports the suite-owned quality runtime before the suite root "
            "is added to sys.path. Refusing to package a remotely unimportable runner."
        )

    module_requirements = {
        "re": r"\bre\.(?:search|match|sub|compile|fullmatch|findall|finditer)\b",
    }
    for module_name, usage_pattern in module_requirements.items():
        uses_module = re.search(usage_pattern, text) is not None
        has_import = re.search(rf"^\s*(?:import\s+{module_name}\b|from\s+{module_name}\s+import\b)", text, flags=re.M) is not None
        if uses_module and not has_import:
            raise RuntimeError(
                f"{path} references module '{module_name}' helpers but does not import '{module_name}'. "
                "Refusing to package a stale or broken runner."
            )



def _remote_preflight_stage_token(stage: Any) -> Optional[str]:
    """Resolve a benchmark-plan stage config into a provider token for remote preflight.

    Returns either an ORT provider token (cpu/cuda/tensorrt/auto) or a Hailo
    hardware token (hailo8/hailo8l/...). The helper mirrors the benchmark-suite
    side stage resolution so the remote launcher can correctly decide whether it
    needs Hailo python bindings even for matrix runs that encode backends in
    nested ``stage1``/``stage2`` dicts instead of legacy flat ``stage1_provider``
    fields.
    """
    if not isinstance(stage, dict):
        return None
    ty = str(stage.get("type") or stage.get("kind") or "").strip().lower()
    if ty == "hailo":
        hw = stage.get("hw_arch") or stage.get("arch") or stage.get("id")
        hw = str(hw or "").strip().lower()
        return hw or "hailo"
    if ty in {"deepx", "deepx_m1", "dx_m1"}:
        return "deepx_m1"
    if ty in {"onnxruntime", "ort"}:
        provider = str(stage.get("provider") or "").strip().lower()
        return provider or None
    return None


def _update_remote_preflight_wants_from_token(token: Optional[str], *, wants: dict[str, bool]) -> None:
    tok = str(token or "").strip().lower()
    if not tok:
        return
    if tok.startswith("hailo"):
        # v59af: Hailo runtime itself does not require ONNXRuntime.  Older
        # preflight logic required ORT for every Hailo run because validation or
        # ORT fallback may use it.  That made Hailo10/Hailo->native-TRT runs
        # fail on offline hosts that have HailoRT+TensorRT but no onnxruntime.
        # ORT is now requested only by explicit ORT stages/runs or by
        # TensorRT stages when the runtime mode is ORT-based.
        wants["hailo"] = True
    elif tok in {"deepx", "deepx_m1", "dx_m1"}:
        wants["deepx"] = True
    elif tok == "cuda":
        wants["cuda"] = True
        wants["onnxruntime"] = True
    elif tok == "tensorrt":
        wants["tensorrt"] = True
        wants["cuda"] = True
        wants["onnxruntime"] = True
    elif tok in {"cpu", "cpu_ort", "ort", "onnxruntime", "auto"}:
        wants["onnxruntime"] = True


def _scan_remote_preflight_requirements_from_plan(plan: dict[str, Any]) -> dict[str, bool]:
    """Infer whether the remote benchmark run needs Hailo / CUDA / TensorRT.

    Supports both legacy flat provider fields and the current nested stage
    dictionaries used by matrix runs such as ``hailo8_to_trt``.
    """
    wants = {"hailo": False, "cuda": False, "tensorrt": False, "deepx": False, "onnxruntime": False}
    for run in plan.get("runs", []):
        if not isinstance(run, dict):
            continue
        for k in ("provider", "full_provider", "stage1_provider", "stage2_provider"):
            _update_remote_preflight_wants_from_token(run.get(k), wants=wants)
        for stage_key in ("stage1", "stage2"):
            _update_remote_preflight_wants_from_token(
                _remote_preflight_stage_token(run.get(stage_key)),
                wants=wants,
            )
        run_type = str(run.get("type") or "").strip().lower()
        if run_type == "hailo":
            # v59af: a pure Hailo full run can execute without onnxruntime.
            wants["hailo"] = True
            _update_remote_preflight_wants_from_token(run.get("hw_arch"), wants=wants)
        elif run_type in {"deepx", "deepx_m1", "dx_m1"}:
            wants["deepx"] = True
        elif run_type in {"onnxruntime", "ort"}:
            wants["onnxruntime"] = True
        elif run_type in {"matrix", "split"}:
            # Matrix/split rows are handled through their explicit stage1/stage2
            # dictionaries.  Do not require ORT solely because the row type is
            # generic.
            pass
    return wants


def _extract_run_ids_from_add_args(add_args: object) -> list[str]:
    """Parse single or comma-separated benchmark-suite run filters."""
    txt = str(add_args or "").strip()
    if not txt:
        return []
    try:
        parts = shlex.split(txt)
    except Exception:
        parts = txt.split()
    out: list[str] = []
    for i, tok in enumerate(parts):
        value = ""
        if tok in {"--run-id", "--run-ids"} and i + 1 < len(parts):
            value = str(parts[i + 1])
        elif tok.startswith("--run-id=") or tok.startswith("--run-ids="):
            value = tok.split("=", 1)[1]
        if value:
            for rid in value.split(","):
                rid = rid.strip()
                if rid and rid not in out:
                    out.append(rid)
    return out


def _extract_run_id_from_add_args(add_args: object) -> str:
    rows = _extract_run_ids_from_add_args(add_args)
    return rows[0] if len(rows) == 1 else (",".join(rows) if rows else "")


def _extract_trt_runtime_mode_from_add_args(
    add_args: str | None, default: str = "native_preferred",
) -> str:
    """Resolve either shipped spelling of the TensorRT runtime selector.

    Empty means the generated v59ac+ suite default applies (native_preferred).
    """
    txt = str(add_args or "").strip()
    if not txt:
        return str(default or "native_preferred").strip().lower()
    try:
        parts = shlex.split(txt)
    except Exception:
        parts = txt.split()
    for i, tok in enumerate(parts):
        if tok in {"--trt-runtime", "--trt-runtime-mode"} and i + 1 < len(parts):
            return str(parts[i + 1]).strip().lower()
        if tok.startswith("--trt-runtime=") or tok.startswith("--trt-runtime-mode="):
            return tok.split("=", 1)[1].strip().lower()
    return str(default or "native_preferred").strip().lower()


def _native_trt_can_avoid_ort_for_run(run_id: str | None, trt_runtime_mode: str) -> bool:
    """Return True if a TensorRT-labelled run can be valid without onnxruntime.

    v59ac introduced native TensorRT engines.  Old preflight logic still treated
    every TensorRT-labelled run as ORT-TRT and therefore failed on systems where
    TensorRT Python/trtexec is installed in system Python but onnxruntime is not.
    For native/native_preferred runs we should only require TensorRT/CUDA libs;
    ORT is useful for fallback/reference, but it is not a hard preflight blocker.
    """
    rid = str(run_id or "").strip().lower()
    mode = str(trt_runtime_mode or "native_preferred").strip().lower()
    if mode in {"ort", "ort_trt", "onnxruntime", "onnxruntime_trt"}:
        return False
    if mode not in {"native", "native_preferred", "auto", ""}:
        return False
    if not rid:
        return False
    return (
        rid in {"ort_tensorrt", "tensorrt", "trt", "native_tensorrt", "native_trt"}
        or rid.endswith("_to_trt")
        or rid.endswith("_to_tensorrt")
    )


def _pure_hailo_run_can_avoid_ort(run_id: str | None) -> bool:
    rid = str(run_id or "").strip().lower()
    if not rid:
        return False
    if "_to_" in rid:
        return False
    return rid.startswith("hailo")


def _filter_benchmark_plan_for_run_id(plan: dict[str, Any], run_id: str) -> dict[str, Any]:
    """Return a copy of *plan* containing only the selected run row.

    If the run id cannot be found, the original plan is returned.  This keeps
    legacy behavior for manual/all-runs launches while preventing run-scoped
    dispatches from inheriting unrelated backend requirements.
    """
    rid = str(run_id or "").strip()
    if not rid or not isinstance(plan, dict):
        return plan
    runs = plan.get("runs") or []
    if not isinstance(runs, list):
        return plan
    selected = []
    for raw in runs:
        if not isinstance(raw, dict):
            continue
        cand = str(raw.get("id") or raw.get("name") or raw.get("run_id") or "").strip()
        if cand == rid:
            selected.append(raw)
    if not selected:
        return plan
    out = dict(plan)
    out["runs"] = selected
    # Keep only matrix rows that explicitly reference the selected id, if the
    # matrix uses run-id fields.  If it does not, keep it empty for preflight.
    matrix = []
    for raw in plan.get("matrix") or []:
        if not isinstance(raw, dict):
            continue
        vals = {str(raw.get(k) or "").strip() for k in ("id", "run_id", "source_run", "target_run", "stage1_run", "stage2_run")}
        if rid in vals:
            matrix.append(raw)
    out["matrix"] = matrix
    return out


def _filter_benchmark_plan_for_run_ids(plan: dict[str, Any], run_ids: list[str]) -> dict[str, Any]:
    wanted = {str(x).strip() for x in run_ids if str(x).strip()}
    if not wanted:
        return dict(plan or {})
    out = dict(plan or {})
    out["runs"] = [r for r in list(plan.get("runs") or []) if isinstance(r, dict) and str(r.get("id") or r.get("name") or "").strip() in wanted]
    return out


def _mixed_quality_only_run_statuses(
    *,
    plan: Mapping[str, Any],
    suite_status: Mapping[str, Any],
    explicit_quality_only_run_ids: list[str],
    expected_eval_run_id: str,
    expected_model_id: str,
    expected_setup_id: str,
    expected_endpoint_id: str,
) -> dict[str, str]:
    """Validate quality-only terminal outcomes inside a mixed suite.

    A Hailo setup executes normal Hailo performance rows and, in the same
    suite, a setup-local TensorRT Full *quality-only* companion.  The latter
    intentionally has no ``benchmark_results_*.json``.  The old rowless
    recognizer required the *entire* suite to be rowless, so it labelled this
    successful companion ``failed`` even though its sealed quality report was
    present.  This branch is deliberately limited to scheduler-declared
    ``--quality-only-run-ids`` and verifies the complete report identity.
    """

    wanted = {
        str(value or "").strip().lower().replace("-", "_")
        for value in explicit_quality_only_run_ids
        if str(value or "").strip()
    }
    eval_run_id = str(expected_eval_run_id or "").strip()
    model_id = str(expected_model_id or "").strip()
    setup_id = str(expected_setup_id or "").strip()
    endpoint_id = str(expected_endpoint_id or "").strip()
    if not wanted or not all((eval_run_id, model_id, setup_id)):
        return {}

    runs_by_id = {
        str(row.get("id") or row.get("name") or "")
        .strip().lower().replace("-", "_"): dict(row)
        for row in list(plan.get("runs") or [])
        if isinstance(row, Mapping)
        and str(row.get("id") or row.get("name") or "").strip()
    }
    if not wanted <= set(runs_by_id):
        return {}

    def _backend(value: Any) -> str:
        if isinstance(value, Mapping):
            value = (
                value.get("backend") or value.get("provider")
                or value.get("hw_arch") or value.get("type") or ""
            )
        token = str(value or "").strip().lower().replace("-", "_")
        return {
            "trt": "tensorrt",
            "ort_tensorrt": "tensorrt",
            "native_tensorrt": "tensorrt",
            "native_full_tensorrt": "tensorrt",
        }.get(token, token)

    declared_identity_by_run: dict[str, tuple[str, str]] = {}
    legacy_trt_runs: set[str] = set()
    for run_id in sorted(wanted):
        run = runs_by_id[run_id]
        variants = [
            str(value or "").strip().lower()
            for value in list(run.get("variants") or [])
            if str(value or "").strip()
        ]
        endpoints = [
            str(value or "").strip()
            for value in list(run.get("quality_canary_endpoint_ids") or [])
            if str(value or "").strip()
        ]
        sources = [
            str(value or "").strip()
            for value in list(run.get("quality_canary_source_run_ids") or [])
            if str(value or "").strip()
        ]
        setups = [
            str(value or "").strip()
            for value in list(run.get("quality_canary_setup_ids") or [])
            if str(value or "").strip()
        ]
        declared = bool(
            run.get("quality_evidence_only") is True
            and str(run.get("execution_scope") or "").strip().lower()
            == "full_only"
            and str(run.get("execution_role") or "").strip().lower()
            == "full_quality_only"
            and run.get("performance_claims_emitted") is False
            and str(run.get("variant") or "").strip().lower() == "full"
            and variants == ["full"]
            and len(endpoints) == len(sources) == 1
            and setups == [setup_id]
        )
        stage1 = _backend(run.get("stage1"))
        stage2 = _backend(run.get("stage2"))
        full_backend = _backend(
            run.get("backend") or run.get("provider") or run.get("type")
        )
        legacy_trt = bool(
            run_id == "ort_tensorrt"
            and full_backend in {"tensorrt", "onnxruntime"}
            and _backend(run.get("provider")) == "tensorrt"
            and stage1 in {"", "tensorrt"}
            and stage2 in {"", "tensorrt"}
            and endpoint_id
        )
        if declared:
            declared_identity_by_run[run_id] = (sources[0], endpoints[0])
        elif legacy_trt:
            legacy_trt_runs.add(run_id)
        else:
            return {}

    reports = [
        dict(row)
        for row in list(suite_status.get("quality_evidence_reports") or [])
        if isinstance(row, Mapping)
    ]
    try:
        declared_report_count = int(
            suite_status.get("quality_evidence_report_count")
        )
    except (TypeError, ValueError):
        return {}
    failed_ids = {
        str(row.get("run_id") or "").strip().lower().replace("-", "_")
        for row in list(suite_status.get("failed_runs") or [])
        if isinstance(row, Mapping)
    }
    if (
        suite_status.get("any_quality_evidence") is not True
        or declared_report_count != len(reports)
        or wanted & failed_ids
    ):
        return {}

    def _canonical_sha256(value: Mapping[str, Any]) -> str:
        encoded = json.dumps(
            dict(value), sort_keys=True, separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    observed: set[str] = set()
    allowed_schemas = {
        "onnx-splitpoint/full-only-quality-evidence-report",
        "onnx-splitpoint/native-full-tensorrt-quality-evidence-report",
    }
    for report in reports:
        request = (
            report.get("request")
            if isinstance(report.get("request"), Mapping)
            else report.get("quality_input_request")
            if isinstance(report.get("quality_input_request"), Mapping)
            else None
        )
        if not isinstance(request, Mapping):
            continue
        identity = request.get("full_only_plan_identity")
        if not isinstance(identity, Mapping):
            continue
        identity = dict(identity)
        source_run_id = str(identity.get("source_run_id") or "").strip()
        canary_id = str(identity.get("quality_canary_id") or "").strip()
        request_sha = str(
            request.get("full_only_plan_identity_sha256") or ""
        ).strip().lower().removeprefix("sha256:")
        try:
            report_count = int(report.get("record_count"))
            request_count = int(request.get("record_count"))
        except (TypeError, ValueError):
            continue
        common_valid = bool(
            str(report.get("schema") or "") in allowed_schemas
            and report.get("schema_version") == 1
            and report.get("quality_evidence_only") is True
            and report.get("performance_claims_emitted") is False
            and str(report.get("eval_run_id") or "").strip() == eval_run_id
            and str(report.get("model_id") or "").strip() == model_id
            and str(report.get("setup_id") or "").strip() == setup_id
            and str(report.get("source_run_id") or "").strip()
            == source_run_id
            and str(report.get("variant") or "").strip().lower() == "full"
            and str(report.get("execution_role") or "").strip().lower()
            == "full_quality_only"
            and report_count > 0
            and request_count == report_count
            and request.get("full_only_plan_identity_required") is True
            and request_sha == _canonical_sha256(identity)
            and str(request.get("quality_canary_id") or "").strip()
            == canary_id
            and str(request.get("eval_run_id") or "").strip() == eval_run_id
            and request.get("performance_claims_emitted") is False
            and identity.get("schema")
            == "onnx-splitpoint/full-only-quality-request-identity"
            and identity.get("schema_version") == 1
            and str(identity.get("eval_run_id") or "").strip() == eval_run_id
            and str(identity.get("model_id") or "").strip() == model_id
            and str(identity.get("setup_id") or "").strip() == setup_id
            and str(identity.get("variant") or "").strip().lower() == "full"
            and str(identity.get("execution_role") or "").strip().lower()
            == "full_quality_only"
            and identity.get("performance_claims_emitted") is False
        )
        if not common_valid:
            continue

        matched = ""
        for run_id, expected in declared_identity_by_run.items():
            if run_id not in observed and expected == (source_run_id, canary_id):
                matched = run_id
                break
        if not matched:
            for run_id in sorted(legacy_trt_runs - observed):
                if (
                    canary_id == endpoint_id
                    and source_run_id in {run_id, "native_full_tensorrt"}
                    and _backend(report.get("backend")) == "tensorrt"
                    and _backend(identity.get("backend")) == "tensorrt"
                ):
                    matched = run_id
                    break
        if matched:
            observed.add(matched)

    if observed != wanted:
        return {}
    return {
        run_id: "quality_evidence_only_complete"
        for run_id in sorted(observed)
    }


def _rowless_full_only_quality_run_statuses(
    *,
    plan: Mapping[str, Any],
    suite_status: Mapping[str, Any],
    final_status: str,
    benchmark_result_count: int,
    expected_eval_run_id: str,
    expected_model_id: str,
    expected_setup_id: str,
    explicit_quality_only_run_ids: Optional[list[str]] = None,
    expected_endpoint_id: str = "",
) -> dict[str, str]:
    """Recognize an exact rowless Full-only Quality suite completion.

    ``run_results.json`` historically labelled every planned run without a
    ``benchmark_results_*.json`` file as failed.  Full-only Quality producers
    intentionally emit no benchmark file, so that label is wrong only for the
    tightly sealed case below.  Any incomplete marker, stale physical identity,
    duplicate evidence item or ordinary performance run remains fail-closed.
    """

    if str(final_status or "").strip().lower() != "ok":
        return {}
    if explicit_quality_only_run_ids:
        return _mixed_quality_only_run_statuses(
            plan=plan,
            suite_status=suite_status,
            explicit_quality_only_run_ids=list(explicit_quality_only_run_ids),
            expected_eval_run_id=expected_eval_run_id,
            expected_model_id=expected_model_id,
            expected_setup_id=expected_setup_id,
            expected_endpoint_id=expected_endpoint_id,
        )
    if int(benchmark_result_count or 0) != 0:
        return {}
    eval_run_id = str(expected_eval_run_id or "").strip()
    model_id = str(expected_model_id or "").strip()
    setup_id = str(expected_setup_id or "").strip()
    if not all((eval_run_id, model_id, setup_id)):
        return {}

    runs = [
        dict(row) for row in list(plan.get("runs") or [])
        if isinstance(row, Mapping)
    ]
    if not runs:
        return {}

    expected_by_identity: dict[tuple[str, str, str], str] = {}
    for run in runs:
        run_id = str(
            run.get("id") or run.get("name") or run.get("run_id") or ""
        ).strip()
        variants = [
            str(value or "").strip().lower()
            for value in list(run.get("variants") or [])
            if str(value or "").strip()
        ]
        endpoint_ids = [
            str(value or "").strip()
            for value in list(run.get("quality_canary_endpoint_ids") or [])
            if str(value or "").strip()
        ]
        setup_ids = [
            str(value or "").strip()
            for value in list(run.get("quality_canary_setup_ids") or [])
            if str(value or "").strip()
        ]
        source_run_ids = [
            str(value or "").strip()
            for value in list(run.get("quality_canary_source_run_ids") or [])
            if str(value or "").strip()
        ]
        exact_plan_row = bool(
            run_id
            and run.get("quality_evidence_only") is True
            and str(run.get("execution_scope") or "").strip().lower()
            == "full_only"
            and str(run.get("execution_role") or "").strip().lower()
            == "full_quality_only"
            and run.get("performance_claims_emitted") is False
            and str(run.get("variant") or "").strip().lower() == "full"
            and variants == ["full"]
            and str(run.get("case_id") or "").strip().lower() == "full"
            and len(endpoint_ids) == 1
            and setup_ids == [setup_id]
            and len(source_run_ids) == 1
        )
        if not exact_plan_row:
            return {}
        identity_key = (source_run_ids[0], setup_id, endpoint_ids[0])
        if identity_key in expected_by_identity:
            return {}
        expected_by_identity[identity_key] = run_id

    reports = [
        dict(row) for row in list(
            suite_status.get("quality_evidence_reports") or []
        ) if isinstance(row, Mapping)
    ]
    try:
        report_count = int(suite_status.get("quality_evidence_report_count"))
        total_runs = int(suite_status.get("total_runs"))
    except (TypeError, ValueError):
        return {}
    if not (
        suite_status.get("any_rows") is False
        and suite_status.get("any_quality_evidence") is True
        and suite_status.get("performance_claims_emitted") is False
        and list(suite_status.get("failed_runs") or []) == []
        and report_count == total_runs == len(runs) == len(reports)
    ):
        return {}

    def _canonical_sha256(value: Mapping[str, Any]) -> str:
        encoded = json.dumps(
            dict(value), sort_keys=True, separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _backend(value: Any) -> str:
        token = str(value or "").strip().lower().replace("-", "_")
        return {
            "trt": "tensorrt",
            "ort_tensorrt": "tensorrt",
            "native_tensorrt": "tensorrt",
            "native_full_tensorrt": "tensorrt",
            "deepx": "deepx_m1",
            "dx_m1": "deepx_m1",
        }.get(token, token)

    observed_run_ids: set[str] = set()
    allowed_report_schemas = {
        "onnx-splitpoint/full-only-quality-evidence-report",
        "onnx-splitpoint/native-full-tensorrt-quality-evidence-report",
    }
    for report in reports:
        request = (
            report.get("request")
            if isinstance(report.get("request"), Mapping)
            else report.get("quality_input_request")
            if isinstance(report.get("quality_input_request"), Mapping)
            else None
        )
        if not isinstance(request, Mapping):
            return {}
        identity = request.get("full_only_plan_identity")
        if not isinstance(identity, Mapping):
            return {}
        identity = dict(identity)
        source_run_id = str(identity.get("source_run_id") or "").strip()
        canary_id = str(identity.get("quality_canary_id") or "").strip()
        identity_setup_id = str(identity.get("setup_id") or "").strip()
        identity_key = (source_run_id, identity_setup_id, canary_id)
        run_id = expected_by_identity.get(identity_key)
        try:
            report_record_count = int(report.get("record_count"))
            request_record_count = int(request.get("record_count"))
        except (TypeError, ValueError):
            return {}
        exact_identity = bool(
            identity.get("schema")
            == "onnx-splitpoint/full-only-quality-request-identity"
            and type(identity.get("schema_version")) is int
            and identity.get("schema_version") == 1
            and str(identity.get("eval_run_id") or "").strip()
            == eval_run_id
            and str(identity.get("model_id") or "").strip() == model_id
            and identity_setup_id == setup_id
            and source_run_id
            and canary_id
            and str(identity.get("variant") or "").strip().lower()
            == "full"
            and str(identity.get("execution_role") or "").strip().lower()
            == "full_quality_only"
            and identity.get("performance_claims_emitted") is False
            and _backend(identity.get("backend"))
            == _backend(report.get("backend"))
        )
        request_identity_sha = str(
            request.get("full_only_plan_identity_sha256") or ""
        ).strip().lower().removeprefix("sha256:")
        exact_report = bool(
            run_id
            and run_id not in observed_run_ids
            and str(report.get("schema") or "") in allowed_report_schemas
            and type(report.get("schema_version")) is int
            and report.get("schema_version") == 1
            and report.get("quality_evidence_only") is True
            and report.get("performance_claims_emitted") is False
            and str(report.get("eval_run_id") or "").strip() == eval_run_id
            and str(report.get("model_id") or "").strip() == model_id
            and str(report.get("setup_id") or "").strip() == setup_id
            and str(report.get("source_run_id") or "").strip()
            == source_run_id
            and str(report.get("variant") or "").strip().lower() == "full"
            and str(report.get("execution_role") or "").strip().lower()
            == "full_quality_only"
            and report_record_count > 0
            and request_record_count == report_record_count
            and request.get("full_only_plan_identity_required") is True
            and request_identity_sha == _canonical_sha256(identity)
            and str(request.get("quality_canary_id") or "").strip()
            == canary_id
            and str(request.get("eval_run_id") or "").strip()
            == eval_run_id
            and request.get("performance_claims_emitted") is False
        )
        if not exact_identity or not exact_report:
            return {}
        quality_ids = [
            str(value or "").strip()
            for value in list(report.get("quality_ids") or [])
            if str(value or "").strip()
        ]
        if quality_ids and quality_ids != [canary_id]:
            return {}
        observed_run_ids.add(str(run_id))

    if observed_run_ids != set(expected_by_identity.values()):
        return {}
    return {
        run_id: "quality_evidence_only_complete"
        for run_id in sorted(observed_run_ids)
    }


def _remote_hailo_auto_env_shell(*, log_prefix: str = "preflight") -> list[str]:
    """Shell lines that auto-discover a Hailo Python environment on the remote host.

    The benchmark tab often had a Hailo venv configured while the formal
    Evaluation Workflow YAML left ``remote_venv`` empty.  In that case the
    remote runner used system Python: CUDA/TensorRT worked, but full Hailo rows
    were skipped with ``No module named hailort``.  v51 probes common Hailo venv
    / setup locations and exposes their site-packages through
    SPLITPOINT_EXTRA_SITES, while still keeping RUN_PY on the system interpreter
    when that is the one with TensorRT/CUDA providers.
    """
    return [
        'if [ "${WANT_HAILO:-0}" = "1" ]; then',
        '  _splitpoint_hailo_env_ok=0',
        '  if [ -n "${ENV_PY:-}" ] && [ -x "${ENV_PY:-}" ]; then',
        "    if \"$ENV_PY\" - <<'PY' >/dev/null 2>&1",
        'try:',
        '    import hailo_platform  # noqa: F401',
        'except Exception:',
        '    import hailort  # noqa: F401',
        'PY',
        '    then _splitpoint_hailo_env_ok=1; fi',
        '  fi',
        '  if [ "$_splitpoint_hailo_env_ok" != "1" ]; then',
        '    for cand in "$HOME/hailo_py/bin/activate" "$HOME/hailo_env/bin/activate" "$HOME/hailort/bin/activate" "$HOME/.venvs/hailo/bin/activate" "$HOME/.venv/hailo/bin/activate" "$HOME/venvs/hailo/bin/activate" "$HOME/hailo/bin/activate" /opt/hailo*/setup_env.sh /opt/hailo/*/setup_env.sh /opt/hailo/*/*/setup_env.sh; do',
        '      [ -f "$cand" ] || continue',
        '      # shellcheck disable=SC1090',
        '      . "$cand" >/dev/null 2>&1 || true',
        '      _cand_py=$(command -v python3 || command -v python || true)',
        "      if [ -n \"$_cand_py\" ] && \"$_cand_py\" - <<'PY' >/dev/null 2>&1",
        'try:',
        '    import hailo_platform  # noqa: F401',
        'except Exception:',
        '    import hailort  # noqa: F401',
        'PY',
        '      then ENV_PY="$_cand_py"; export SPLITPOINT_AUTO_HAILO_ENV="$cand"; echo "[' + log_prefix + '] auto-hailo-env: $cand"; break; fi',
        '    done',
        '  fi',
        'fi',
    ]


# ------------------------------
# Tarball extraction (safe)
# ------------------------------

def _extract_tarball(tar_path: Path, out_dir: Path, *, log: Callable[[str], None] | None = None) -> None:
    """Extract a .tar/.tar.gz file into *out_dir* safely.

    Protects against path traversal (e.g. entries like '../../etc/passwd').
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    mode = "r:*"
    if tar_path.name.endswith(".tar.gz") or tar_path.name.endswith(".tgz"):
        mode = "r:gz"

    if log is not None:
        log(f"[extract] {tar_path} -> {out_dir} ({mode})")

    with tarfile.open(tar_path, mode) as tf:
        base = out_dir.resolve()
        for member in tf.getmembers():
            member_path = (out_dir / member.name).resolve()
            if not str(member_path).startswith(str(base)):
                raise RuntimeError(f"Unsafe path in tarball: {member.name}")
        tf.extractall(out_dir)


def init_local_run_artifacts(local_run_dir: Path, run_meta: dict[str, Any]) -> None:
    """Create the full local run folder structure and minimal files.

    IMPORTANT: Must be called before any remote/SSH activity so that debugging artifacts
    exist even if connection/setup fails.
    """
    logs_dir = local_run_dir / "logs"
    results_dir = local_run_dir / "results"
    logs_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Always present, even if empty.
    _touch(logs_dir / "runner.log")
    _touch(logs_dir / "stdout.txt")
    _touch(logs_dir / "stderr.txt")

    # Write meta + placeholder status/results immediately.
    _write_json(local_run_dir / "run_meta.json", run_meta)

    _write_json(
        local_run_dir / "run_status.json",
        {
            "schema_version": RUN_STATUS_SCHEMA_VERSION,
            "status": "failed",  # will be finalized at the end
            "started_at": run_meta.get("started_at"),
            "ended_at": None,
            "remote_rc": None,
            "fail_reason": {
                "message": "Run started but not finalized.",
                "exception": None,
                "stderr_tail": [],
                "stdout_tail": [],
            },
        },
    )

    _write_json(
        local_run_dir / "run_results.json",
        {
            "schema_version": RUN_RESULTS_SCHEMA_VERSION,
            "host": run_meta.get("host"),
            "objective": run_meta.get("objective", "latency"),
            "planned_runs": [],
            "timings": {"init_build_ms": None, "runtime_ms": None},
            "artifacts": {},
        },
    )


def _finalize_run_status(
    local_run_dir: Path,
    *,
    status: str,
    started_at: str,
    ended_at: str,
    remote_rc: int | None,
    fail_message: str | None = None,
    exception_text: str | None = None,
    stdout_tail: list[str] | None = None,
    stderr_tail: list[str] | None = None,
    extra_fail_reason: Optional[dict[str, Any]] = None,
) -> None:
    """Write final run_status.json."""
    if stdout_tail is None:
        stdout_tail = []
    if stderr_tail is None:
        stderr_tail = []
    payload: dict[str, Any] = {
        "schema_version": RUN_STATUS_SCHEMA_VERSION,
        "status": status,
        "started_at": started_at,
        "ended_at": ended_at,
        "remote_rc": remote_rc,
    }
    if status != "ok":
        fail_reason: dict[str, Any] = {
            "message": fail_message or "Run failed.",
            "exception": exception_text,
            "stderr_tail": stderr_tail,
            "stdout_tail": stdout_tail,
        }
        if isinstance(extra_fail_reason, dict) and extra_fail_reason:
            fail_reason.update(extra_fail_reason)
        payload["fail_reason"] = fail_reason
    else:
        payload["fail_reason"] = None
    _write_json(local_run_dir / "run_status.json", payload)


def _finalize_run_results(
    local_run_dir: Path,
    *,
    host: dict[str, Any],
    objective: str,
    planned_runs: list[dict[str, Any]],
    artifact_index: dict[str, Any],
) -> None:
    payload: dict[str, Any] = {
        "schema_version": RUN_RESULTS_SCHEMA_VERSION,
        "host": host,
        "objective": objective,
        "planned_runs": planned_runs,
        "timings": {"init_build_ms": None, "runtime_ms": None},
        "artifacts": artifact_index,
    }
    _write_json(local_run_dir / "run_results.json", payload)


@dataclass
class RemoteBenchmarkArgs:
    # Runtime compiler admission only; never part of the engine cache identity.
    trt_build_guard: dict[str, Any] = field(default_factory=dict)

    # NOTE: provider='auto' means "run the embedded plan".
    provider: str = "auto"
    # Optional: shell snippet or path to a venv activate script that should run before the suite.
    # Examples: "~/hailo_py/bin/activate" or "source /opt/hailo/setup_env.sh"
    remote_venv: str = ""
    repeats: int = 1
    warmup: int = 10
    iters: int = 100
    add_args: str = ""

    # Total timeout for the remote benchmark command.
    timeout_s: Optional[int] = 7200

    # Transfer mode for the suite:
    # - bundle: tar.gz (fast for many small files, supports caching)
    # - direct: scp -r (useful for debugging; may be slower and copies everything)
    transfer_mode: str = "bundle"  # 'bundle' | 'direct'

    # Only relevant for transfer_mode='bundle'
    reuse_bundle: bool = True
    # Resume a previous partial run for the same suite/host/settings when possible.
    resume: bool = True

    # Optional streaming/interleaving parameters for heterogeneous pipelines.
    throughput_frames: int = 24
    throughput_warmup_frames: int = 6
    throughput_queue_depth: int = 2
    phase_runs: int = 5

    # Optional local u.RECS energy wrapper metadata.  The remote runner itself
    # ignores these fields; GUI/workflow orchestration uses them to decide
    # whether to wrap benchmark-remote CLI calls in urecs-data-collector.
    energy_enabled: bool = False
    energy_setup_id: str = ""
    # Exact hardware registry selected by the caller. Claim-bearing energy
    # must keep this provenance through preflight and every collector window.
    energy_registry_path: str = ""
    # Populated only by the shared, fresh pre-transport admission.  Keeping
    # the digest on the immutable dispatch args lets automatic Resume reject
    # an otherwise identical partial run that used a different registry
    # snapshot (including a changed calibration binding at the same path).
    energy_registry_snapshot_sha256: str = ""
    energy_run_count: int = 0
    # Optional local output root for u.RECS energy artifacts.  GUI/manual
    # dispatch may set this so EnergyMeasurements are written next to the
    # selected working directory/results instead of the package folder.
    energy_output_root: str = ""
    # Optional fixed collector duration.  Normally left unset so the energy
    # wrapper performs a duration probe for each measured phase/window.
    energy_known_duration_s: Optional[float] = None
    energy_scope: str = "row_variant"
    energy_phases: list[str] = field(default_factory=list)
    energy_strict: bool = False
    energy_heartbeat_s: int = 60
    energy_target_policy: str = "canonical_only"
    energy_max_targets_per_run_id: int = 0
    energy_max_work_units_per_window: int = 0
    energy_max_window_duration_s: int = 0
    energy_timeout_s_per_window: int = 0
    energy_sizing_probe_max_work_units: int = 256
    energy_confidence_level: float = 0.95
    energy_physical_scope: str | None = None
    energy_window_label: str | None = None
    energy_window_ab_enabled: bool = False
    energy_window_ab_baseline_method: str = "chapter4_baseline"
    energy_window_ab_candidate_method: str = "candidate_v263"
    energy_window_ab_same_raw_capture: bool = True
    energy_window_ab_mode: str = "shadow"
    energy_window_ab_auto_switch: bool = False
    energy_window_ab_smoke_repeats: int = 3
    energy_window_ab_include_raw_parquet: bool = True
    energy_window_ab_requires_picoscope: bool = False
    # When enabled, the row-variant target blocks are shuffled with a recorded
    # deterministic seed before measurement.  Repeats within one target remain
    # contiguous because the u.RECS collector owns their raw-trace bundle.
    energy_randomize_target_order: bool = False
    energy_randomization_seed: int = 20260710

    # Cleanup policy for remote run directories.  By default successful runs are
    # removed from the remote NX after their results have been copied locally.
    # Failed/partial runs are kept for debugging unless explicitly requested.
    cleanup_remote_after_download: bool = True
    cleanup_remote_on_partial: bool = False

    # Optional semantic validation source. Detection can fall back to the suite-embedded
    # COCO-50 dataset; classification requires a labeled dataset path from the user.
    validation_images: str = ""
    validation_max_images: int = 0
    validation_budget_authoritative: bool = False
    validation_reference_mode: str = "auto"
    mini_coco_ap50: bool = False
    benchmark_task: str = "auto"
    mini_classification_eval: bool = False

    # Immutable setup-local TensorRT Full-quality identity.  The Evaluation
    # Workflow supplies these values; a remote host must not infer a physical
    # setup ID from a logical run profile.
    quality_evidence_eval_id: str = ""
    quality_evidence_model_id: str = ""
    quality_evidence_setup_id: str = ""
    quality_evidence_endpoint_id: str = ""


def _remote_trt_build_guard_shell(args: RemoteBenchmarkArgs) -> str:
    """Carry the admitted warm/cold policy through the actual remote shell."""
    guard = dict(getattr(args, "trt_build_guard", {}) or {})
    if guard:
        guard["model_id"] = str(guard.get("model_id") or getattr(args, "quality_evidence_model_id", "") or "")
        guard["setup_id"] = str(getattr(args, "quality_evidence_setup_id", "") or "")
        if not guard["model_id"]:
            raise ValueError("strict TensorRT build guard requires model identity")
    payload = json.dumps(guard, sort_keys=True, separators=(",", ":")) if guard else ""
    # Clear inherited shell configuration on ordinary runs as well.
    return "export ONNX_SPLITPOINT_TRT_BUILD_GUARD=" + shlex.quote(payload)


def _bind_energy_ab_runtime_args(defaults: Any, args: RemoteBenchmarkArgs) -> Any:
    """Apply the frozen profile A/B contract to the local collector defaults."""
    defaults.window_ab_enabled = bool(getattr(args, "energy_window_ab_enabled", False))
    defaults.window_ab_baseline_method = str(
        getattr(args, "energy_window_ab_baseline_method", "chapter4_baseline")
        or "chapter4_baseline"
    )
    defaults.window_ab_candidate_method = str(
        getattr(args, "energy_window_ab_candidate_method", "candidate_v263")
        or "candidate_v263"
    )
    defaults.window_ab_same_raw_capture = bool(
        getattr(args, "energy_window_ab_same_raw_capture", True)
    )
    defaults.window_ab_mode = str(getattr(args, "energy_window_ab_mode", "shadow") or "shadow")
    defaults.window_ab_auto_switch = bool(getattr(args, "energy_window_ab_auto_switch", False))
    defaults.window_ab_smoke_repeats = max(
        1,
        int(getattr(args, "energy_window_ab_smoke_repeats", 3) or 3),
    )
    defaults.window_ab_include_raw_parquet = bool(
        getattr(args, "energy_window_ab_include_raw_parquet", True)
    )
    defaults.window_ab_requires_picoscope = bool(
        getattr(args, "energy_window_ab_requires_picoscope", False)
    )
    defaults.compare_legacy_window = bool(defaults.window_ab_enabled)
    defaults.keep_raw_parquet = bool(
        getattr(defaults, "keep_raw_parquet", True)
        or defaults.window_ab_include_raw_parquet
    )
    return defaults


@dataclass
class SuiteProgress:
    run_id: str
    i: int
    n: int
    pct: float


def parse_benchmark_suite_progress(line: str) -> SuiteProgress | None:
    """Parse progress lines produced by benchmark_suite.py.

    Expected pattern:
        "[{run_id}] [{i}/{n}] Running ..."

    This is intentionally simple and tolerant.
    """

    line = line.strip()
    if not line.startswith("["):
        return None

    try:
        run_id = line.split("]", 1)[0].lstrip("[")
        # second bracket block
        rest = line.split("]", 2)[1]
        if "[" not in rest:
            return None
        part = rest.split("[", 1)[1].split("]", 1)[0]
        if "/" not in part:
            return None
        i_s, n_s = part.split("/", 1)
        i = int(i_s)
        n = int(n_s)
        if n <= 0:
            return None
        pct = max(0.0, min(1.0, i / n))
        return SuiteProgress(run_id=run_id, i=i, n=n, pct=pct)
    except Exception:
        return None


def _iter_suite_case_dirs(suite_dir: Path) -> list[Path]:
    """Return benchmark case directories in a suite.

    We intentionally only count directories that look like real benchmark cases
    (``bXXX/`` with a ``split_manifest.json``). This avoids inflating timeout
    estimates with auxiliary folders.
    """

    case_dirs: list[Path] = []
    try:
        for child in sorted(suite_dir.iterdir()):
            if not child.is_dir():
                continue
            if _CASE_DIR_RE.match(child.name) is None:
                continue
            if not (child / "split_manifest.json").exists():
                continue
            case_dirs.append(child)
    except Exception:
        return []
    return case_dirs


def _load_plan_runs_from_suite(suite_dir: Path) -> list[dict[str, Any]]:
    """Best-effort load of ``benchmark_plan.json`` runs from a suite."""

    plan_path = suite_dir / "benchmark_plan.json"
    if not plan_path.exists():
        return []
    try:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    runs = plan.get("runs") if isinstance(plan, dict) else None
    if not isinstance(runs, list):
        return []
    return [r for r in runs if isinstance(r, dict)]


def _stable_file_sha256(path: Path) -> str:
    # A path/size/mtime memo can be fooled by a same-size replacement whose
    # timestamp is restored.  The persistent TensorRT namespace claims exact
    # graph-byte identity, so hash the bytes on every key construction.
    return _streaming_sha256(path)


def _portable_suite_plan_value(value: Any, suite_dir: Path) -> Any:
    volatile_keys = {
        "created_at", "generated_at", "started_at", "finished_at",
        "ended_at", "updated_at", "suite_dir", "legacy_suite_dir",
        "evaluation_run_dir", "evaluation_run_id", "output_root",
        "generation_log",
        # Compiler wall time is diagnostic-only and changes between otherwise
        # identical benchmark-set generations.  It must not select a new
        # persistent TensorRT namespace.
        "elapsed_s",
    }
    if isinstance(value, dict):
        return {
            str(key): _portable_suite_plan_value(item, suite_dir)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if str(key) not in volatile_keys
        }
    if isinstance(value, list):
        return [_portable_suite_plan_value(item, suite_dir) for item in value]
    if isinstance(value, str):
        try:
            candidate = Path(value).expanduser()
            if candidate.is_absolute():
                resolved = candidate.resolve(strict=False)
                root = suite_dir.resolve(strict=True)
                if resolved == root or resolved.is_relative_to(root):
                    return "suite://" + resolved.relative_to(root).as_posix()
                # Management-host absolute roots are transport provenance, not
                # TensorRT engine semantics.  Preserve the path role/name while
                # removing volatile home/EvaluationRun prefixes.
                return "external://" + candidate.name
        except (OSError, RuntimeError, ValueError):
            pass
    return value


def _stable_suite_cache_key(suite_dir: Path) -> str:
    """Semantic, path-portable key for the persistent TensorRT cache.

    Run timestamps and EvaluationRun-local paths are excluded, while exact
    ONNX bytes and the effective benchmark plan remain bound.  Thus repeated
    runs reuse one cache namespace without allowing a changed graph to collide.
    """

    suite_dir = Path(suite_dir).expanduser().resolve(strict=True)
    plan = _read_json_dict(suite_dir / "benchmark_plan.json") or {}
    benchmark_set = _read_json_dict(suite_dir / "benchmark_set.json") or {}
    primary = (
        ((plan.get("model_suite") or {}).get("primary") or [])
        if isinstance(plan.get("model_suite"), dict)
        else []
    )
    primary_first = primary[0] if primary and isinstance(primary[0], dict) else {}
    model_hint = str(
        primary_first.get("id")
        or benchmark_set.get("model_name")
        or suite_dir.parent.parent.name
        or suite_dir.name
    )
    model_hint = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(model_hint or "model"))[:80]
    payload: dict[str, Any] = {
        "schema": "onnx-splitpoint/semantic-remote-cache-key",
        "schema_version": 2,
        "benchmark_plan": _portable_suite_plan_value(plan, suite_dir),
        "benchmark_set": _portable_suite_plan_value(benchmark_set, suite_dir),
        "onnx_files": [],
    }
    for path in sorted(suite_dir.rglob("*.onnx")):
        if path.is_symlink() or not path.is_file():
            continue
        payload["onnx_files"].append({
            "path": path.relative_to(suite_dir).as_posix(),
            "size_bytes": int(path.stat().st_size),
            "sha256": _stable_file_sha256(path),
        })
    h = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    )
    digest = h.hexdigest()[:16]
    return f"{model_hint}-{digest}"


def _argument_value(add_args: object, option: str, default: str) -> str:
    """Return one benchmark-suite option without interpreting unrelated CLI state."""

    text = str(add_args or "").strip()
    if not text:
        return str(default)
    try:
        parts = shlex.split(text)
    except Exception:
        parts = text.split()
    for index, token in enumerate(parts):
        if token == option and index + 1 < len(parts):
            return str(parts[index + 1]).strip()
        if token.startswith(option + "="):
            return str(token.split("=", 1)[1]).strip()
    return str(default)


def _canonical_model_onnx_identities(suite_dir: Path) -> list[dict[str, Any]]:
    """Return the canonical Full-model content identities for a TRT namespace.

    Split graphs are deliberately not part of the namespace key: their existing
    leaf paths already contain the exact ONNX SHA-256.  Keying the parent by all
    candidate graphs made an added/removed split invalidate every overlapping
    engine.  The Full model still keeps unrelated model families apart.
    """

    suite_dir = Path(suite_dir).expanduser().resolve(strict=True)
    benchmark_set = _read_json_dict(suite_dir / "benchmark_set.json") or {}
    candidates: list[Path] = []
    model_value = benchmark_set.get("model")
    if isinstance(model_value, str) and model_value.strip():
        model_path = Path(model_value).expanduser()
        if not model_path.is_absolute():
            model_path = suite_dir / model_path
        try:
            resolved = model_path.resolve(strict=True)
            if resolved.is_file() and resolved.suffix.lower() == ".onnx":
                candidates.append(resolved)
        except (OSError, RuntimeError):
            pass
    models_dir = suite_dir / "models"
    if models_dir.is_dir() and not models_dir.is_symlink():
        for path in sorted(models_dir.glob("*.onnx")):
            if path.is_file() and not path.is_symlink():
                candidates.append(path.resolve(strict=True))
    if not candidates:
        # Compatibility for old single-model suites which kept Full next to the
        # runner.  Do not include candidate Part1/Part2 graphs in this fallback.
        for path in sorted(suite_dir.glob("*.onnx")):
            name = path.name.lower()
            if (
                path.is_file() and not path.is_symlink()
                and "part1" not in name and "part2" not in name
            ):
                candidates.append(path.resolve(strict=True))
    unique: dict[tuple[int, str], dict[str, Any]] = {}
    for path in candidates:
        identity = {
            "size_bytes": int(path.stat().st_size),
            "sha256": _stable_file_sha256(path),
        }
        unique[(identity["size_bytes"], identity["sha256"])] = identity
    if not unique:
        raise RuntimeError(
            f"Cannot construct TensorRT engine cache key: canonical Full ONNX missing in {suite_dir}"
        )
    return [unique[key] for key in sorted(unique)]


def _canonical_model_onnx_sources(suite_dir: Path) -> list[Path]:
    """Return the exact Full ONNX files used by the persistent TRT key.

    Keep this path resolver in lockstep with
    :func:`_canonical_model_onnx_identities`.  The cache preflight needs the
    source bytes as well as their digest, but must not invent a second model
    identity rule.
    """

    suite_dir = Path(suite_dir).expanduser().resolve(strict=True)
    benchmark_set = _read_json_dict(suite_dir / "benchmark_set.json") or {}
    candidates: list[Path] = []
    model_value = benchmark_set.get("model")
    if isinstance(model_value, str) and model_value.strip():
        model_path = Path(model_value).expanduser()
        if not model_path.is_absolute():
            model_path = suite_dir / model_path
        try:
            resolved = model_path.resolve(strict=True)
            if resolved.is_file() and resolved.suffix.lower() == ".onnx":
                candidates.append(resolved)
        except (OSError, RuntimeError):
            pass
    models_dir = suite_dir / "models"
    if models_dir.is_dir() and not models_dir.is_symlink():
        for path in sorted(models_dir.glob("*.onnx")):
            if path.is_file() and not path.is_symlink():
                candidates.append(path.resolve(strict=True))
    if not candidates:
        for path in sorted(suite_dir.glob("*.onnx")):
            name = path.name.lower()
            if (
                path.is_file() and not path.is_symlink()
                and "part1" not in name and "part2" not in name
            ):
                candidates.append(path.resolve(strict=True))
    unique: dict[tuple[int, str], Path] = {}
    for path in candidates:
        key = (int(path.stat().st_size), _stable_file_sha256(path))
        unique.setdefault(key, path)
    if not unique:
        raise RuntimeError(
            f"Cannot resolve TensorRT Full source: canonical Full ONNX missing in {suite_dir}"
        )
    return [unique[key] for key in sorted(unique)]


def _trt_backend_token(value: Any) -> str:
    if isinstance(value, Mapping):
        for key in ("provider", "backend", "type", "id", "name"):
            token = _trt_backend_token(value.get(key))
            if token:
                return token
        return ""
    token = str(value or "").strip().lower().replace("-", "_")
    if token in {"trt", "tensorrt", "ort_tensorrt"}:
        return "tensorrt"
    return token


def _trt_preflight_run_requirements(
    suite_dir: Path,
    *,
    active_run_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Resolve only the TensorRT artifacts consumed by active plan rows.

    Explicit Full-only references consume one Full engine. Ordinary TRT rows
    without variants execute Full, Part1, Part2 and composed in the suite
    runner. Their ordinary Part2 engine is distinct from a vendor Native
    bridge engine even when both consume the same selected boundary.
    """

    suite = Path(suite_dir).expanduser().resolve(strict=True)
    plan = _read_json_dict(suite / "benchmark_plan.json") or {}
    contract = _read_json_dict(suite / "benchmark_set.json") or {}
    rows = [
        dict(row) for row in list(plan.get("runs") or [])
        if isinstance(row, Mapping)
    ]
    selected = {
        str(value or "").strip() for value in list(active_run_ids or [])
        if str(value or "").strip()
    }
    if active_run_ids is not None:
        rows = [
            row for row in rows
            if str(
                row.get("id") or row.get("run_id") or row.get("name") or ""
            ).strip() in selected
        ]

    case_ids = [path.name for path in _iter_suite_case_dirs(suite)]
    if not case_ids:
        for raw in list(contract.get("cases") or []):
            if isinstance(raw, Mapping):
                case_id = str(
                    raw.get("id") or raw.get("case_id") or raw.get("case") or ""
                ).strip()
                if not case_id:
                    try:
                        case_id = f"b{int(raw.get('boundary')):03d}"
                    except (TypeError, ValueError):
                        case_id = ""
            else:
                case_id = str(raw or "").strip()
            if case_id and case_id not in case_ids:
                case_ids.append(case_id)

    full_run_ids: list[str] = []
    p1_cases: list[str] = []
    p1_run_ids_by_case: dict[str, list[str]] = {}
    p2_cases: list[str] = []
    p2_run_ids_by_case: dict[str, list[str]] = {}
    generic_p2_cases: list[str] = []
    generic_p2_run_ids_by_case: dict[str, list[str]] = {}
    for row in rows:
        if row.get("enabled") is False or bool(row.get("deferred")):
            continue
        status = str(row.get("status") or row.get("build_status") or "").lower()
        if status in {"disabled", "deferred", "not_selected"}:
            continue
        run_id = str(
            row.get("id") or row.get("run_id") or row.get("name") or ""
        ).strip()
        run_type = str(row.get("type") or "").strip().lower().replace("-", "_")
        variants = {
            str(value or "").strip().lower()
            for value in list(row.get("variants") or [])
            if str(value or "").strip()
        }

        explicit_full = row.get("full")
        full_backend = _trt_backend_token(explicit_full)
        full_reference = bool(
            full_backend == "tensorrt"
            or (
                run_type in {"onnxruntime", "same_backend_reference", "full"}
                and _trt_backend_token(
                    row.get("full_provider") or row.get("provider")
                    or row.get("backend")
                ) == "tensorrt"
            )
            or (
                not run_type
                and run_id.lower().replace("-", "_")
                in {"ort_tensorrt", "tensorrt", "trt"}
            )
        )
        if full_reference and (not variants or "full" in variants) and run_id not in full_run_ids:
            full_run_ids.append(run_id or "tensorrt_full")

        # A provider=tensorrt matrix row does not by itself prove the direction;
        # only the explicitly frozen stage2 contract is authoritative.
        stage2_declared = row.get("stage2") not in (None, "", {})
        stage1_declared = row.get("stage1") not in (None, "", {})
        run_id_token = run_id.lower().replace("-", "_")
        generic_trt = bool(
            run_type in {"", "onnxruntime", "ort", "same_backend_reference", "matrix", "split"}
            and (run_type not in {"matrix", "split"} or stage1_declared and stage2_declared)
            and _trt_backend_token(row.get("provider") or row.get("backend")
                or row.get("stage1") or run_id) == "tensorrt"
            and (not stage1_declared or _trt_backend_token(row.get("stage1")) == "tensorrt")
            and (not stage2_declared or _trt_backend_token(row.get("stage2")) == "tensorrt")
        )
        if generic_trt and (not variants or "full" in variants) and run_id not in full_run_ids:
            full_run_ids.append(run_id or "tensorrt_full")
        generic_part2 = generic_trt and (not variants or bool(variants & {"part2", "composed"}))
        stage1_trt = (
            generic_trt
            or
            (
                run_type == "matrix"
                and _trt_backend_token(row.get("stage1")) == "tensorrt"
            )
            or (
                not stage1_declared and not run_type
                and (run_id_token.startswith("trt_to_")
                     or run_id_token.startswith("tensorrt_to_"))
            )
        ) and (not variants or bool(variants & {"part1", "composed"}))
        stage2_trt = (
            (
                run_type == "matrix"
                and _trt_backend_token(row.get("stage2")) == "tensorrt"
            )
            or (
                not stage2_declared
                and not run_type
                and (
                    run_id_token.endswith("_to_trt")
                    or run_id_token.endswith("_to_tensorrt")
                )
            )
        ) and not generic_trt and (not variants or bool(variants & {"part2", "composed"}))
        if not stage1_trt and not stage2_trt and not generic_part2:
            continue
        raw_cases: list[Any] = []
        for key in ("case_id", "case", "cases", "case_ids", "selected_cases"):
            value = row.get(key)
            if value in (None, "", [], {}):
                continue
            if isinstance(value, Sequence) and not isinstance(
                value, (str, bytes, bytearray)
            ):
                raw_cases.extend(value)
            else:
                raw_cases.append(value)
        selected_cases: list[str] = []
        for raw in raw_cases:
            if isinstance(raw, Mapping):
                token = str(
                    raw.get("id") or raw.get("case_id") or raw.get("case") or ""
                ).strip()
            else:
                token = str(raw or "").strip()
            # Benchmark-set case directories use the canonical ``bNNN``
            # spelling even when a scheduler row serializes the boundary as
            # an integer (or as an unpadded b-token).  Normalize before the
            # requirement becomes an artifact/item identity.
            if token.isdigit():
                token = f"b{int(token):03d}"
            elif (
                token.lower().startswith("b")
                and token[1:].isdigit()
            ):
                token = f"b{int(token[1:]):03d}"
            if token and token not in selected_cases:
                selected_cases.append(token)
        if not selected_cases:
            selected_cases = list(case_ids)
        for case_id in selected_cases:
            for required, cases, run_ids_by_case in (
                (stage1_trt, p1_cases, p1_run_ids_by_case),
                (stage2_trt, p2_cases, p2_run_ids_by_case),
            ):
                if not required:
                    continue
                if case_id not in cases:
                    cases.append(case_id)
                run_ids_by_case.setdefault(case_id, [])
                if run_id and run_id not in run_ids_by_case[case_id]:
                    run_ids_by_case[case_id].append(run_id)
            if generic_part2:
                if case_id not in generic_p2_cases:
                    generic_p2_cases.append(case_id)
                generic_p2_run_ids_by_case.setdefault(case_id, [])
                if run_id and run_id not in generic_p2_run_ids_by_case[case_id]:
                    generic_p2_run_ids_by_case[case_id].append(run_id)
    return {
        "full_required": bool(full_run_ids),
        "full_run_ids": full_run_ids,
        "p1_cases": p1_cases,
        "p1_run_ids_by_case": p1_run_ids_by_case,
        "p2_cases": p2_cases,
        "p2_run_ids_by_case": p2_run_ids_by_case,
        "generic_p2_cases": generic_p2_cases,
        "generic_p2_run_ids_by_case": generic_p2_run_ids_by_case,
    }


def _trt_engine_runtime_contract(
    suite_dir: Path,
    *,
    args: "RemoteBenchmarkArgs | None" = None,
    active_run_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Project only build-affecting runtime values onto an explicit allowlist."""

    plan = _read_json_dict(Path(suite_dir) / "benchmark_plan.json") or {}
    runs = plan.get("runs") if isinstance(plan.get("runs"), list) else []
    run_filters = (
        [str(value).strip() for value in list(active_run_ids or []) if str(value).strip()]
        if active_run_ids is not None
        else _extract_run_ids_from_add_args(getattr(args, "add_args", ""))
        if args is not None else []
    )
    if active_run_ids is not None or run_filters:
        selected = set(run_filters)
        runs = [
            run for run in runs
            if isinstance(run, dict)
            and str(run.get("id") or run.get("name") or "") in selected
        ]
    trt_runs = [run for run in runs if isinstance(run, dict) and _run_mentions_tensorrt(run)]

    def allowed_values(*keys: str) -> list[Any]:
        values: list[Any] = []
        for run in trt_runs:
            for key in keys:
                value = run.get(key)
                if value not in (None, "", [], {}):
                    values.append(value)
        encoded = {
            json.dumps(value, sort_keys=True, separators=(",", ":"), default=str): value
            for value in values
        }
        return [encoded[key] for key in sorted(encoded)]

    add_args = getattr(args, "add_args", "") if args is not None else ""
    plan_precisions = allowed_values("native_trt_precision", "precision")
    plan_workspaces = allowed_values("native_trt_workspace_mb", "workspace_mb")
    plan_runtimes = allowed_values("trt_runtime", "trt_runtime_mode")
    precision_default = str(plan_precisions[0]) if len(plan_precisions) == 1 else "fp16"
    workspace_default = str(plan_workspaces[0]) if len(plan_workspaces) == 1 else "4096"
    runtime_default = (
        str(plan_runtimes[0]) if len(plan_runtimes) == 1 else "native_preferred"
    )
    precision = _argument_value(add_args, "--native-trt-precision", precision_default)
    workspace_text = _argument_value(add_args, "--native-trt-workspace-mb", workspace_default)
    trt_runtime = _extract_trt_runtime_mode_from_add_args(
        add_args, runtime_default,
    )
    try:
        workspace_mb: int | str = int(workspace_text)
    except (TypeError, ValueError):
        workspace_mb = str(workspace_text)
    return {
        "native_trt_precision": str(precision or "fp16").lower(),
        "native_trt_workspace_mb": workspace_mb,
        "trt_runtime": str(trt_runtime or "native_preferred").lower(),
        "plan_precision_overrides": plan_precisions,
        "plan_workspace_overrides": plan_workspaces,
        "plan_runtime_overrides": plan_runtimes,
        # Changes to command construction must explicitly bump this ABI token.
        "command_builder_abi": "native-trt-command-v1",
        "engine_receipt_schema_version": 1,
        "shape_policy": "static-omit-dynamic-explicit-v1",
    }


def _trt_engine_builder_abi_contract(builder_abi: Mapping[str, Any] | None) -> dict[str, Any]:
    """Project a remote probe onto stable, engine-semantic ABI fields only.

    ``trtexec --version`` output is retained by the diagnostic probe, but is
    deliberately absent here: NVIDIA prefixes that output with wall-clock log
    timestamps on some releases.  Paths, probe return codes and GPU UUIDs are
    likewise diagnostic rather than engine-format inputs.
    """

    raw = dict(builder_abi or {})

    def normalized_text(value: object) -> str:
        return re.sub(r"\s+", " ", str(value or "")).strip()

    gpu_targets: list[dict[str, str]] = []
    structured_targets = raw.get("gpu_targets")
    if isinstance(structured_targets, list):
        for target in structured_targets:
            if not isinstance(target, Mapping):
                continue
            normalized = {
                "name": normalized_text(target.get("name")),
                "compute_capability": normalized_text(
                    target.get("compute_capability") or target.get("compute_cap")
                ),
                "driver_version": normalized_text(target.get("driver_version")),
            }
            if normalized["name"]:
                normalized["index"] = normalized_text(target.get("index"))
                gpu_targets.append(normalized)
    if not gpu_targets:
        # Backward-compatible projection for pre-contract probes/tests.  The
        # historical CSV is name,uuid[,compute_cap],driver; UUID is omitted.
        legacy_rows = raw.get("gpu_identity")
        if isinstance(legacy_rows, list):
            for index, row in enumerate(legacy_rows):
                parts = [normalized_text(value) for value in str(row).split(",")]
                if len(parts) >= 4:
                    name, compute_capability, driver_version = parts[0], parts[-2], parts[-1]
                elif len(parts) >= 3:
                    name, compute_capability, driver_version = parts[0], "", parts[-1]
                else:
                    continue
                gpu_targets.append({
                    "index": str(index),
                    "name": name,
                    "compute_capability": compute_capability,
                    "driver_version": driver_version,
                })

    # TensorRT serializes an engine for the device selected by the CUDA process,
    # not for every accelerator installed in the host.  Keying the complete
    # nvidia-smi inventory made an unrelated secondary GPU invalidate the cache
    # and, more importantly, did not distinguish selecting GPU A from GPU B on
    # an otherwise unchanged multi-GPU machine.  The remote probe therefore
    # publishes the effective CUDA-visible device zero explicitly.  Older test
    # fixtures/probes are accepted by selecting their declared index (default
    # zero) from the diagnostic inventory, but new probes always carry the
    # stronger ``selected_gpu_target`` object.
    selected_raw = raw.get("selected_gpu_target")
    selected = dict(selected_raw) if isinstance(selected_raw, Mapping) else {}
    if not selected and gpu_targets:
        requested_index = normalized_text(raw.get("selected_gpu_index") or "0")
        selected = dict(
            next(
                (target for target in gpu_targets if target.get("index") == requested_index),
                gpu_targets[0],
            )
        )
        selected.setdefault("visible_device_index", "0")
        selected.setdefault("physical_device_index", selected.get("index", requested_index))

    selected_gpu_target = {
        "visible_device_index": normalized_text(
            selected.get("visible_device_index") or "0"
        ),
        "name": normalized_text(selected.get("name")),
        "compute_capability": normalized_text(
            selected.get("compute_capability") or selected.get("compute_cap")
        ),
        "driver_version": normalized_text(selected.get("driver_version")),
        "driver_api_version": normalized_text(selected.get("driver_api_version")),
    }

    libraries: dict[tuple[str, int, str], dict[str, Any]] = {}
    raw_libraries = raw.get("linked_runtime_libraries")
    if isinstance(raw_libraries, list):
        for library in raw_libraries:
            if not isinstance(library, Mapping):
                continue
            name = normalized_text(library.get("name"))
            digest = str(library.get("sha256") or "").strip().lower()
            try:
                size_bytes = int(library.get("size_bytes") or 0)
            except (TypeError, ValueError):
                size_bytes = 0
            if name and re.fullmatch(r"[0-9a-f]{64}", digest) and size_bytes >= 0:
                libraries[(name, size_bytes, digest)] = {
                    "name": name,
                    "size_bytes": size_bytes,
                    "sha256": digest,
                }

    try:
        trtexec_size = int(raw.get("trtexec_size_bytes") or 0)
    except (TypeError, ValueError):
        trtexec_size = 0
    return {
        "schema": "onnx-splitpoint/trt-engine-builder-abi-contract",
        "schema_version": 2,
        "trtexec_binary": {
            "size_bytes": trtexec_size,
            "sha256": str(raw.get("trtexec_sha256") or "").strip().lower(),
        },
        "selected_gpu_target": selected_gpu_target,
        "linked_runtime_libraries": [libraries[key] for key in sorted(libraries)],
    }


def _stable_trt_engine_cache_key(
    suite_dir: Path,
    *,
    args: "RemoteBenchmarkArgs | None" = None,
    builder_abi: dict[str, Any] | None = None,
    active_run_ids: Sequence[str] | None = None,
) -> str:
    """Content/ABI key for persistent TensorRT engines, independent of a run.

    The stricter suite key remains the resume/transport identity.  This key is
    intentionally smaller: Full-model bytes select the namespace, exact split
    bytes select existing leaf paths, and only engine-building runtime/ABI data
    can invalidate both.  Profiles, campaigns, tool release labels and vendor
    compiler diagnostics never enter this payload.
    """

    suite_dir = Path(suite_dir).expanduser().resolve(strict=True)
    plan = _read_json_dict(suite_dir / "benchmark_plan.json") or {}
    benchmark_set = _read_json_dict(suite_dir / "benchmark_set.json") or {}
    primary = (
        ((plan.get("model_suite") or {}).get("primary") or [])
        if isinstance(plan.get("model_suite"), dict)
        else []
    )
    primary_first = primary[0] if primary and isinstance(primary[0], dict) else {}
    model_hint = str(
        primary_first.get("id")
        or benchmark_set.get("model_name")
        or suite_dir.parent.parent.name
        or suite_dir.name
    )
    model_hint = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(model_hint or "model"))[:80]
    payload = {
        "schema": "onnx-splitpoint/trt-engine-cache-key",
        "schema_version": 2,
        "canonical_full_onnx": _canonical_model_onnx_identities(suite_dir),
        "runtime_contract": _trt_engine_runtime_contract(
            suite_dir, args=args, active_run_ids=active_run_ids,
        ),
        "builder_abi": _trt_engine_builder_abi_contract(builder_abi),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:16]
    return f"{model_hint}-{digest}"


def _trt_persistent_engine_relative_dir(
    *,
    role: str,
    source_onnx_sha256: str,
    precision: str,
    case_id: str = "",
) -> Path:
    """Return the v2.79.20 persistent TensorRT leaf policy.

    The namespace selected by :func:`_stable_trt_engine_cache_key` is already
    bound to the canonical Full-model bytes and the builder/runtime ABI.  A
    Full engine therefore must not inherit a selected split boundary.  Split
    engines, in contrast, remain explicitly case-scoped.  This helper only
    maps already-established identities onto directories; it introduces no
    additional digest or evidence contract.
    """

    role_token = str(role or "").strip().lower()
    if role_token not in {"full", "part1", "part2"}:
        raise ValueError(f"unsupported TensorRT engine role: {role!r}")
    digest = str(source_onnx_sha256 or "").strip().lower()
    if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError("source_onnx_sha256 must be a lowercase SHA-256 digest")
    precision_token = str(precision or "").strip().lower()
    if re.fullmatch(r"[a-z0-9_.-]{1,80}", precision_token) is None:
        raise ValueError(f"unsafe TensorRT precision token: {precision!r}")
    if role_token == "full":
        return Path("full") / digest / precision_token
    case_token = str(case_id or "").strip()
    if re.fullmatch(r"b[0-9]+", case_token, flags=re.IGNORECASE) is None:
        raise ValueError(f"split TensorRT engine requires a case id: {case_id!r}")
    return Path("splits") / case_token / role_token / digest / precision_token


def _trt_persistent_cache_layout(
    *,
    namespace_root: str,
    canonical_full_onnx: list[dict[str, Any]] | tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    """Describe the model-level Full and case-level split cache layout."""

    root = str(namespace_root or "").rstrip("/")
    full_identities = sorted({
        str(row.get("sha256") or "").strip().lower()
        for row in canonical_full_onnx
        if isinstance(row, Mapping)
        and re.fullmatch(
            r"[0-9a-f]{64}", str(row.get("sha256") or "").strip().lower()
        )
    })
    return {
        "schema": "onnx-splitpoint/trt-persistent-cache-layout",
        "schema_version": 1,
        "namespace_root": root,
        "full_root": f"{root}/full",
        "splits_root": f"{root}/splits",
        "canonical_full_onnx_sha256": full_identities,
        "full_scope": "model_content_runtime_builder_abi",
        "split_scope": "case_and_source_content",
        "full_leaf_template": "full/<source_onnx_sha256>/<precision>",
        "split_leaf_template": (
            "splits/<case_id>/<part1_or_part2>/<source_onnx_sha256>/<precision>"
        ),
    }


def _remote_trt_activation_shell(remote_venv: str | None) -> str:
    """Return a narrowly admitted activation command for a read-only probe."""

    value = str(remote_venv or "").strip()
    if not value:
        return ""
    try:
        parts = shlex.split(value)
    except ValueError as exc:
        raise ValueError("remote TensorRT environment activation is invalid") from exc
    if len(parts) == 1:
        path = parts[0]
    elif len(parts) == 2 and parts[0] in {"source", "."}:
        path = parts[1]
    else:
        raise ValueError(
            "remote TensorRT ABI probe only accepts an activation-script path "
            "or `source <path>`"
        )
    if not path or any(char in path for char in ("\x00", "\n", "\r")):
        raise ValueError("remote TensorRT activation path is unsafe")
    if path.startswith("~/"):
        quoted_path = '"$HOME"/' + shlex.quote(path[2:])
    else:
        quoted_path = shlex.quote(path)
    return f"source {quoted_path}"


def _remote_trt_builder_abi(
    transport: SSHTransport, *, remote_venv: str = "",
) -> dict[str, Any]:
    """Probe the runtime environment's TensorRT/GPU ABI without mutating it."""

    script = r'''import csv, ctypes, ctypes.util, hashlib, io, json, os, pathlib, re, shutil, subprocess

def file_sha(path):
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

trtexec = None
for candidate in (
    os.environ.get("TRTEXEC", ""),
    "trtexec",
    "/usr/src/tensorrt/bin/trtexec",
    "/usr/bin/trtexec",
    "/usr/local/bin/trtexec",
):
    if not candidate:
        continue
    try:
        if os.sep in candidate:
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                trtexec = candidate
                break
        else:
            discovered = shutil.which(candidate)
            if discovered:
                trtexec = discovered
                break
    except Exception:
        continue
if not trtexec:
    raise SystemExit("trtexec_not_found")
trtexec = str(pathlib.Path(trtexec).resolve())

def capture(argv, timeout=30):
    try:
        result = subprocess.run(argv, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, timeout=timeout)
        return int(result.returncode), str(result.stdout or "").strip()
    except Exception as exc:
        return 255, "%s:%s" % (type(exc).__name__, exc)

version_rc, version_text = capture([trtexec, "--version"])

# CUDA's driver API is the authoritative fallback on Jetson, where nvidia-smi
# is commonly absent.  It is also useful on workstations because visible device
# zero is the device selected by an otherwise unqualified TensorRT invocation.
# All calls below are queries; no context, allocation or output file is created.
cuda_driver_probe = {
    "ok": False,
    "source": "cuda_driver_api_visible_device_0",
    "errors": [],
}
selected_gpu_target = {}
cuda_driver_library_path = ""

def loaded_cuda_driver_path():
    try:
        for line in pathlib.Path("/proc/self/maps").read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if not fields:
                continue
            candidate = fields[-1]
            if "/libcuda.so" not in candidate:
                continue
            path = pathlib.Path(candidate)
            if path.is_file():
                return str(path.resolve(strict=True))
    except Exception:
        pass
    return ""

cuda_candidates = []
try:
    discovered = ctypes.util.find_library("cuda")
    if discovered:
        cuda_candidates.append(discovered)
except Exception as exc:
    cuda_driver_probe["errors"].append("find_library:%s" % type(exc).__name__)
cuda_candidates.extend([
    "libcuda.so.1",
    "/usr/lib/aarch64-linux-gnu/tegra/libcuda.so.1",
    "/usr/lib/aarch64-linux-gnu/libcuda.so.1",
    "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
])

cuda = None
for candidate in dict.fromkeys(cuda_candidates):
    try:
        cuda = ctypes.CDLL(candidate)
        break
    except Exception as exc:
        cuda_driver_probe["errors"].append(
            "%s:%s" % (pathlib.Path(candidate).name, type(exc).__name__)
        )

if cuda is not None:
    try:
        cuda.cuInit.argtypes = [ctypes.c_uint]
        cuda.cuInit.restype = ctypes.c_int
        cuda.cuDeviceGetCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
        cuda.cuDeviceGetCount.restype = ctypes.c_int
        cuda.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        cuda.cuDeviceGet.restype = ctypes.c_int
        cuda.cuDeviceGetName.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
        cuda.cuDeviceGetName.restype = ctypes.c_int
        cuda.cuDeviceComputeCapability.argtypes = [
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.c_int,
        ]
        cuda.cuDeviceComputeCapability.restype = ctypes.c_int
        cuda.cuDriverGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
        cuda.cuDriverGetVersion.restype = ctypes.c_int
        cuda.cuDeviceGetAttribute.argtypes = [
            ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int,
        ]
        cuda.cuDeviceGetAttribute.restype = ctypes.c_int

        if int(cuda.cuInit(0)) != 0:
            raise RuntimeError("cuInit_failed")
        count = ctypes.c_int()
        if int(cuda.cuDeviceGetCount(ctypes.byref(count))) != 0 or count.value <= 0:
            raise RuntimeError("cuDeviceGetCount_failed_or_empty")
        visible_index = 0
        device = ctypes.c_int()
        if int(cuda.cuDeviceGet(ctypes.byref(device), visible_index)) != 0:
            raise RuntimeError("cuDeviceGet_failed")
        name_buffer = ctypes.create_string_buffer(256)
        if int(cuda.cuDeviceGetName(name_buffer, len(name_buffer), device.value)) != 0:
            raise RuntimeError("cuDeviceGetName_failed")
        major = ctypes.c_int()
        minor = ctypes.c_int()
        if int(cuda.cuDeviceComputeCapability(
            ctypes.byref(major), ctypes.byref(minor), device.value,
        )) != 0:
            raise RuntimeError("cuDeviceComputeCapability_failed")
        driver_api_version = ctypes.c_int()
        if int(cuda.cuDriverGetVersion(ctypes.byref(driver_api_version))) != 0:
            raise RuntimeError("cuDriverGetVersion_failed")

        pci_values = {}
        for key, attribute in (("bus", 33), ("device", 34), ("domain", 50)):
            value = ctypes.c_int()
            if int(cuda.cuDeviceGetAttribute(
                ctypes.byref(value), attribute, device.value,
            )) == 0:
                pci_values[key] = int(value.value)
        pci_bus_id = ""
        if {"domain", "bus", "device"} <= set(pci_values):
            pci_bus_id = "%08x:%02x:%02x.0" % (
                pci_values["domain"], pci_values["bus"], pci_values["device"],
            )

        visible = str(os.environ.get("CUDA_VISIBLE_DEVICES") or "").strip()
        selector = visible.split(",", 1)[0].strip() if visible else ""
        selected_gpu_target = {
            "visible_device_index": str(visible_index),
            "physical_device_index": selector if selector.isdigit() else "",
            "name": re.sub(r"\s+", " ", name_buffer.value.decode("utf-8", "replace")).strip(),
            "compute_capability": "%d.%d" % (major.value, minor.value),
            "driver_version": "",
            "driver_api_version": str(driver_api_version.value),
            "pci_bus_id": pci_bus_id,
            "selection_source": "cuda_driver_api_visible_device_0",
        }
        cuda_driver_library_path = loaded_cuda_driver_path()
        cuda_driver_probe.update({
            "ok": True,
            "visible_device_count": int(count.value),
            "cuda_visible_devices": visible,
            "selected_gpu_target": dict(selected_gpu_target),
            "library_path": cuda_driver_library_path,
        })
    except Exception as exc:
        cuda_driver_probe["errors"].append("query:%s:%s" % (type(exc).__name__, exc))

gpu_rc, gpu_text = capture([
    "nvidia-smi", "--query-gpu=index,name,uuid,pci.bus_id,compute_cap,driver_version",
    "--format=csv,noheader,nounits",
])
gpu_has_compute_cap = gpu_rc == 0
if gpu_rc != 0:
    gpu_rc, gpu_text = capture([
        "nvidia-smi", "--query-gpu=index,name,uuid,pci.bus_id,driver_version",
        "--format=csv,noheader,nounits",
    ])
    gpu_has_compute_cap = False

gpu_targets = []
if gpu_rc == 0:
    try:
        for row in csv.reader(io.StringIO(gpu_text)):
            fields = [re.sub(r"\s+", " ", str(value or "")).strip() for value in row]
            if gpu_has_compute_cap and len(fields) >= 6:
                index, name, uuid, pci_bus_id, compute_capability, driver_version = fields[:6]
            elif len(fields) >= 5:
                index, name, uuid, pci_bus_id, driver_version = fields[:5]
                compute_capability = ""
            else:
                continue
            if name:
                gpu_targets.append({
                    "index": index,
                    "name": name,
                    "uuid": uuid,
                    "pci_bus_id": pci_bus_id.lower(),
                    "compute_capability": compute_capability,
                    "driver_version": driver_version,
                })
    except Exception:
        gpu_targets = []

# Enrich the CUDA-selected device with the nvidia-smi physical index and driver
# package version when available.  Inventory rows remain diagnostics only.
def normalized_pci(value):
    return re.sub(r"^0+([0-9a-f]{4}:)", r"\1", str(value or "").strip().lower())

selected_inventory_target = None
if selected_gpu_target and gpu_targets:
    selected_pci = normalized_pci(selected_gpu_target.get("pci_bus_id"))
    if selected_pci:
        selected_inventory_target = next(
            (row for row in gpu_targets if normalized_pci(row.get("pci_bus_id")) == selected_pci),
            None,
        )
    if selected_inventory_target is None:
        physical_index = str(selected_gpu_target.get("physical_device_index") or "")
        if physical_index:
            selected_inventory_target = next(
                (row for row in gpu_targets if str(row.get("index") or "") == physical_index),
                None,
            )
    if selected_inventory_target is None and len(gpu_targets) == 1:
        selected_inventory_target = gpu_targets[0]
    if selected_inventory_target is not None:
        selected_gpu_target["physical_device_index"] = str(
            selected_inventory_target.get("index") or ""
        )
        selected_gpu_target["driver_version"] = str(
            selected_inventory_target.get("driver_version") or ""
        )

# Preserve x86/nvidia-smi operation if the CUDA driver query itself is
# unavailable, while still selecting exactly one target.  Jetson succeeds via
# the driver branch above and never needs this fallback.
if not selected_gpu_target and gpu_targets:
    visible = str(os.environ.get("CUDA_VISIBLE_DEVICES") or "").strip()
    selector = visible.split(",", 1)[0].strip() if visible else "0"
    disabled_selectors = {"-1", "none", "void"}
    if selector.lower() not in disabled_selectors:
        selected_inventory_target = next(
            (
                row for row in gpu_targets
                if str(row.get("index") or "") == selector
                or str(row.get("uuid") or "") == selector
            ),
            gpu_targets[0] if not visible else None,
        )
        if selected_inventory_target is not None:
            selected_gpu_target = {
                "visible_device_index": "0",
                "physical_device_index": str(selected_inventory_target.get("index") or selector),
                "name": str(selected_inventory_target.get("name") or ""),
                "compute_capability": str(selected_inventory_target.get("compute_capability") or ""),
                "driver_version": str(selected_inventory_target.get("driver_version") or ""),
                "driver_api_version": "",
                "pci_bus_id": str(selected_inventory_target.get("pci_bus_id") or ""),
                "selection_source": "nvidia_smi_selected_index_fallback",
            }

libraries = []
ldd_rc, ldd_text = capture(["ldd", trtexec])
if ldd_rc == 0:
    for line in ldd_text.splitlines():
        match = re.search(r"=>\s+(/\S+)", line)
        if not match:
            match = re.match(r"\s*(/\S+)", line)
        if not match:
            continue
        path = pathlib.Path(match.group(1))
        name = path.name.lower()
        if not any(token in name for token in ("nvinfer", "nvonnx", "cudart", "cuda.so")):
            continue
        try:
            resolved = path.resolve(strict=True)
            libraries.append({
                "name": resolved.name,
                "size_bytes": int(resolved.stat().st_size),
                "sha256": file_sha(resolved),
            })
        except OSError:
            continue
if cuda_driver_library_path:
    try:
        driver_path = pathlib.Path(cuda_driver_library_path).resolve(strict=True)
        libraries.append({
            "name": driver_path.name,
            "size_bytes": int(driver_path.stat().st_size),
            "sha256": file_sha(driver_path),
        })
    except OSError:
        pass
libraries = list({
    (row["name"], int(row["size_bytes"]), row["sha256"]): row
    for row in libraries
}.values())
payload = {
    "schema": "onnx-splitpoint/remote-trt-builder-abi",
    "schema_version": 1,
    "trtexec_path": trtexec,
    "trtexec_sha256": file_sha(trtexec),
    "trtexec_size_bytes": int(pathlib.Path(trtexec).stat().st_size),
    "trtexec_version": version_text[-2000:],
    "trtexec_version_rc": version_rc,
    "gpu_identity": sorted(line.strip() for line in gpu_text.splitlines() if line.strip()),
    "gpu_targets": sorted(gpu_targets, key=lambda row: (row["index"], row["name"])),
    "gpu_probe_rc": gpu_rc,
    "cuda_driver_probe": cuda_driver_probe,
    "selected_gpu_target": selected_gpu_target,
    "linked_runtime_libraries": sorted(libraries, key=lambda row: (row["name"], row["sha256"])),
}
print("SPLITPOINT_TRT_BUILDER_ABI=" + json.dumps(payload, sort_keys=True, separators=(",", ":")))
'''
    activation = _remote_trt_activation_shell(remote_venv)
    command_lines = [
        "set -e",
        'SPLITPOINT_ABI_PY="$(command -v python3 || command -v python || true)"',
        'test -n "$SPLITPOINT_ABI_PY"',
    ]
    if activation:
        command_lines.append(activation)
    command_lines.extend([
        '"$SPLITPOINT_ABI_PY" - <<\'SPLITPOINT_TRT_ABI_PY\'',
        script,
        "SPLITPOINT_TRT_ABI_PY",
    ])
    command = "\n".join(command_lines)
    rc, output = transport.run_read_only(command, timeout=120)
    marker = "SPLITPOINT_TRT_BUILDER_ABI="
    if rc == 0:
        for line in reversed(str(output or "").splitlines()):
            if line.startswith(marker):
                try:
                    payload = json.loads(line[len(marker):])
                    contract = _trt_engine_builder_abi_contract(payload)
                    selected = dict(contract.get("selected_gpu_target") or {})
                    if (
                        isinstance(payload, dict)
                        and re.fullmatch(r"[0-9a-f]{64}", str(payload.get("trtexec_sha256") or ""))
                        and selected.get("name")
                        and re.fullmatch(
                            r"\d+\.\d+", str(selected.get("compute_capability") or "")
                        )
                        and (
                            selected.get("driver_version")
                            or selected.get("driver_api_version")
                        )
                        and contract.get("linked_runtime_libraries")
                    ):
                        return payload
                except Exception:
                    break
    raise RuntimeError(
        "Could not attest the remote TensorRT builder ABI before cache selection: "
        + str(output or "").strip()[-2000:]
    )


def _trt_preflight_model_id(suite_dir: Path) -> str:
    plan = _read_json_dict(Path(suite_dir) / "benchmark_plan.json") or {}
    contract = _read_json_dict(Path(suite_dir) / "benchmark_set.json") or {}
    primary = (
        ((plan.get("model_suite") or {}).get("primary") or [])
        if isinstance(plan.get("model_suite"), Mapping) else []
    )
    first = primary[0] if primary and isinstance(primary[0], Mapping) else {}
    return str(
        first.get("id") or contract.get("model_id")
        or contract.get("model_name") or Path(suite_dir).parent.parent.name
        or "model"
    ).strip()


def _trt_preflight_split_source(suite_dir: Path, case_id: str, *, role: str) -> Path:
    if role not in {"part1", "part2"}:
        raise ValueError(f"unsupported TensorRT split role: {role}")
    case_dir = Path(suite_dir) / str(case_id)
    candidates: list[Path] = []
    for pattern in (f"*_{role}_*.onnx", f"*{role}*.onnx"):
        candidates.extend(sorted(case_dir.glob(pattern)))
    unique: dict[str, Path] = {}
    for path in candidates:
        if path.is_file() and not path.is_symlink():
            resolved = path.resolve(strict=True)
            unique.setdefault(_stable_file_sha256(resolved), resolved)
    if not unique:
        raise FileNotFoundError(f"TensorRT {role} source missing for {case_id}")
    if len(unique) != 1:
        raise RuntimeError(f"TensorRT {role} source ambiguous for {case_id}")
    return next(iter(unique.values()))


def _trt_preflight_part2_source(suite_dir: Path, case_id: str) -> Path:
    return _trt_preflight_split_source(suite_dir, case_id, role="part2")


def _trt_preflight_part1_source(suite_dir: Path, case_id: str) -> Path:
    return _trt_preflight_split_source(suite_dir, case_id, role="part1")


def _trt_preflight_shape_contract(source_onnx: Path) -> dict[str, Any]:
    """Project the native TRT runner's default ONNX shape policy.

    Generated benchmark suites are valid ONNX and the controller normally has
    ``onnx`` available.  Keep a narrow fallback for diagnostic/minimal test
    environments: an omitted ``--shapes`` argument is always permitted by the
    runner's retry contract, but a non-empty shape cannot be certified without
    parsing the graph and is therefore reported as UNKNOWN by the remote
    verifier rather than guessed.
    """

    source = Path(source_onnx).expanduser().resolve(strict=True)
    try:
        import onnx  # type: ignore

        model = onnx.load(str(source), load_external_data=False)
        initializer_names = {value.name for value in model.graph.initializer}
        specs: list[tuple[str, list[int], bool]] = []
        for value in model.graph.input:
            if value.name in initializer_names:
                continue
            dims: list[int] = []
            dynamic = False
            for dim in value.type.tensor_type.shape.dim:
                if getattr(dim, "dim_value", 0) and int(dim.dim_value) > 0:
                    dims.append(int(dim.dim_value))
                else:
                    dims.append(1)
                    dynamic = True
            specs.append((str(value.name), dims, dynamic))
        if not specs:
            raise RuntimeError("onnx_graph_inputs_missing")
        all_static = all(not row[2] for row in specs)
        shape_arg = ",".join(
            f"{name}:{'x'.join(str(int(dim)) for dim in dims)}"
            for name, dims, _dynamic in specs if dims
        )
        # ``retry_without_shapes`` defaults to true in the actual native TRT
        # builder, hence empty remains compatible for a dynamic model.
        allowed = [""] if all_static else [shape_arg, ""]
        return {
            "complete": True,
            "inputs_static": all_static,
            "expected_shapes": "" if all_static else shape_arg,
            "allowed_shapes": list(dict.fromkeys(allowed)),
            "policy": "static_omit_dynamic_explicit_v1",
        }
    except Exception as exc:
        return {
            "complete": False,
            "inputs_static": None,
            "expected_shapes": "",
            "allowed_shapes": [""],
            "policy": "static_omit_dynamic_explicit_v1",
            "error": f"{type(exc).__name__}: {exc}",
        }


def _trt_preflight_source_shape_contracts(
    suite_dir: Path, *, active_run_ids: Sequence[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Map active canonical source identities to their build shape contract."""

    suite = Path(suite_dir).expanduser().resolve(strict=True)
    sources = list(_canonical_model_onnx_sources(suite))
    requirement_plan = _trt_preflight_run_requirements(
        suite, active_run_ids=active_run_ids,
    )
    for case_id in dict.fromkeys(requirement_plan["p2_cases"] + requirement_plan.get("generic_p2_cases", [])):
        sources.append(_trt_preflight_part2_source(suite, case_id))
    for case_id in requirement_plan["p1_cases"]:
        sources.append(_trt_preflight_part1_source(suite, case_id))
    contracts: dict[str, dict[str, Any]] = {}
    for source in sources:
        digest = _stable_file_sha256(source)
        contracts[digest] = _trt_preflight_shape_contract(source)
    return contracts


def _trt_preflight_native_backend(
    suite_dir: Path, *, run_ids: Sequence[str], setup_accelerator: str = "",
) -> str:
    """Resolve one exact Native producer backend for a Part-2 requirement.

    Quality-FIRST bindings are intentionally backend-specific.  Do not infer
    compatibility from the setup name when the active plan rows disagree or
    do not name a supported Native producer.
    """

    plan = _read_json_dict(Path(suite_dir) / "benchmark_plan.json") or {}
    selected = {str(value or "").strip() for value in run_ids if str(value or "").strip()}
    rows = [
        dict(row) for row in list(plan.get("runs") or [])
        if isinstance(row, Mapping)
        and str(row.get("id") or row.get("run_id") or row.get("name") or "").strip()
        in selected
    ]

    def canonical(value: Any) -> str:
        if isinstance(value, Mapping):
            for key in ("hw_arch", "accelerator", "backend", "provider", "id", "name"):
                resolved = canonical(value.get(key))
                if resolved:
                    return resolved
            return ""
        token = str(value or "").strip().lower().replace("-", "_")
        if "hailo10" in token:
            return "hailo10h_to_trt"
        if "hailo8" in token:
            return "hailo8_to_trt"
        if "deepx" in token or "dx_m1" in token:
            return "deepx_to_trt"
        return ""

    backends: set[str] = set()
    for row in rows:
        run_id = str(
            row.get("id") or row.get("run_id") or row.get("name") or ""
        )
        backend = canonical(row.get("stage1")) or canonical(run_id)
        if backend:
            backends.add(backend)
    if not backends:
        fallback = canonical(setup_accelerator)
        if fallback:
            backends.add(fallback)
    return next(iter(backends)) if len(backends) == 1 else ""



def _trt_preflight_native_policy(
    suite_dir: Path, *, model_id: str, setup_id: str,
    requirement: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Project the same quality-side variant selected by the composed runtime.

    The global native_trt_precision describes ordinary TensorRT engines. A
    vendor composed row consumes the policy's boundary bridge instead. This
    only chooses the expected recipe; the remote strict binding validator must
    still verify the source, bridge, Part1, metadata and receipt cross-links.
    """
    backend = str(requirement.get("expected_backend") or "")
    if not backend:
        return None
    plan = _read_json_dict(Path(suite_dir) / "benchmark_plan.json") or {}
    run_ids = set(requirement.get("run_ids") or [])
    rows = [row for row in list(plan.get("runs") or [])
            if isinstance(row, Mapping)
            and str(row.get("id") or row.get("run_id") or row.get("name") or "")
            in run_ids]
    # A Part2-only benchmark does not enter the composed quality call site.
    if not rows or not any(
        not row.get("variants") or "composed" in {
            str(value).strip().lower() for value in row.get("variants", [])
        } for row in rows
    ):
        return None
    if "native_split_quality_selection" in plan:
        selection = plan["native_split_quality_selection"]
        aliases = {
            "hailo8": "hailo8", "hailo8l": "hailo8", "hailo8r": "hailo8",
            "hailo10": "hailo10h", "hailo10h": "hailo10h",
            "hailo10n": "hailo10h", "hailo10p": "hailo10h",
            "deepx": "deepx", "deepx_m1": "deepx", "dx_m1": "deepx",
        }
        if not isinstance(selection, Mapping):
            raise ValueError("native_split_quality_selection_invalid")
        selected = selection.get("split_backends")
        applicable = selection.get("applicable")
        source = str(selection.get("split_selection_source") or "").strip()
        if (
            selection.get("schema") != "onnx-splitpoint/native-split-quality-selection"
            or selection.get("schema_version") != 1
            or not isinstance(applicable, bool)
            or not isinstance(selected, list) or not source
        ):
            raise ValueError("native_split_quality_selection_invalid")
        normalized = [aliases.get(str(value or "").strip().lower().replace("-", "_"), "")
                      for value in selected]
        if (any(not value for value in normalized)
            or len(normalized) != len(set(normalized))
            or bool(normalized) != applicable
            or source == "native_disabled" and (applicable or normalized)):
            raise ValueError("native_split_quality_selection_invalid")
        if backend.removesuffix("_to_trt") not in normalized:
            return None
    from ..native_split_quality import known_native_split_policy

    policy = known_native_split_policy(
        model_id=model_id, case_id=requirement.get("case_id"),
        setup_id=setup_id, backend=backend,
    )
    # Unsupported model families have no policy to project. Their existing
    # generic cache probe remains unchanged; runtime applicability is separate.
    if policy is not None and policy.get("task") != requirement.get("expected_task"):
        raise ValueError("native_split_quality_policy_task_mismatch")
    return policy


def _suite_native_quality_validator_payload(suite_dir: Path) -> dict[str, Any]:
    """Embed the suite's exact vendored strict validator for remote execution.

    The probe runs before suite upload, so the selected host cannot be assumed
    to have the current validator installed.  Source bytes are therefore sent
    over the existing read-only SSH stdin command and executed in an in-memory
    ``splitpoint_runners`` package.  No remote file is created or changed.
    """

    suite = Path(suite_dir).expanduser().resolve(strict=True)
    runner_dir = suite / "splitpoint_runners"
    requested = {
        "splitpoint_runners.native_command_contract": (
            runner_dir / "native_command_contract.py"
        ),
        "splitpoint_runners.native_split_quality": (
            runner_dir / "native_split_quality.py"
        ),
    }
    trusted_root = Path(__file__).resolve().parents[1]
    trusted = {
        "splitpoint_runners.native_command_contract": (
            trusted_root / "native_command_contract.py"
        ),
        "splitpoint_runners.native_split_quality": (
            trusted_root / "native_split_quality.py"
        ),
    }
    modules: dict[str, dict[str, Any]] = {}
    try:
        import base64
        import zlib

        for name, path in requested.items():
            if not path.is_file() or path.is_symlink():
                raise FileNotFoundError(str(path))
            raw = path.read_bytes()
            if not raw:
                raise ValueError(f"empty validator module: {path}")
            trusted_raw = trusted[name].read_bytes()
            if raw != trusted_raw:
                raise ValueError(
                    "suite validator differs from trusted installed source: "
                    f"{path}"
                )
            # Decode now as well, so a malformed suite cannot turn the remote
            # verifier into an opaque import failure.
            raw.decode("utf-8")
            modules[name] = {
                "source_zlib_base64": base64.b64encode(
                    zlib.compress(raw, level=9)
                ).decode("ascii"),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "size_bytes": len(raw),
                "suite_relative_path": str(path.relative_to(suite)),
            }
    except Exception as exc:
        return {
            "available": False,
            "implementation": (
                "splitpoint_runners.native_split_quality."
                "validate_native_split_quality_binding"
            ),
            "verification_mode": "local",
            "reason": f"{type(exc).__name__}: {exc}",
            "modules": {},
        }
    quality_sha = modules[
        "splitpoint_runners.native_split_quality"
    ]["sha256"]
    return {
        "available": True,
        "implementation": (
            "splitpoint_runners.native_split_quality."
            "validate_native_split_quality_binding"
        ),
        "verification_mode": "local",
        "validator_source_identity": (
            "splitpoint_runners.native_split_quality@sha256:" + quality_sha
        ),
        "modules": modules,
    }


def _remote_trt_cache_probe_command(payload: Mapping[str, Any]) -> str:
    """Build one self-contained, read-only remote receipt verifier."""

    import base64

    encoded = base64.b64encode(json.dumps(
        dict(payload), sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).decode("ascii")
    script = r'''import base64, hashlib, json, os, pathlib, re, sys, types, zlib

P = json.loads(base64.b64decode("__PAYLOAD__").decode("utf-8"))

def sha_file(path):
    h = hashlib.sha256()
    with pathlib.Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def canonical_sha(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")).hexdigest()

def load(path):
    try:
        value = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}

def load_owner(namespace):
    marker = pathlib.Path(namespace) / ".splitpoint_trt_cache_owner.json"
    if not marker.is_file() or marker.is_symlink():
        return {}
    payload = load(marker)
    if (
        payload.get("schema") != "onnx-splitpoint/managed-trt-cache-owner"
        or payload.get("schema_version") != 1
        or payload.get("owner") != "onnx-splitpoint-tool"
        or str(payload.get("cache_key") or "") != pathlib.Path(namespace).name
    ):
        return {}
    return payload

def within(path, root):
    try:
        pathlib.Path(path).resolve(strict=True).relative_to(
            pathlib.Path(root).resolve(strict=True)
        )
        return True
    except Exception:
        return False

_STRICT_NATIVE_QUALITY_VALIDATOR = None
_STRICT_NATIVE_QUALITY_VALIDATOR_DETAIL = None

def strict_native_quality_validator():
    """Load the suite-vendored validator into an in-memory package once."""
    global _STRICT_NATIVE_QUALITY_VALIDATOR
    global _STRICT_NATIVE_QUALITY_VALIDATOR_DETAIL
    if _STRICT_NATIVE_QUALITY_VALIDATOR_DETAIL is not None:
        return (
            _STRICT_NATIVE_QUALITY_VALIDATOR,
            _STRICT_NATIVE_QUALITY_VALIDATOR_DETAIL,
        )
    descriptor = P.get("native_quality_validator") or {}
    detail = {
        "validator_implementation": str(
            descriptor.get("implementation") or ""
        ),
        "validator_verification_mode": "local",
        "validator_source_identity": str(
            descriptor.get("validator_source_identity") or ""
        ),
        "validator_module_sha256": {},
    }
    if descriptor.get("available") is not True:
        detail["validator_error"] = str(
            descriptor.get("reason") or "suite validator unavailable"
        )
        _STRICT_NATIVE_QUALITY_VALIDATOR_DETAIL = detail
        return None, detail
    modules = descriptor.get("modules") or {}
    wanted = (
        "splitpoint_runners.native_command_contract",
        "splitpoint_runners.native_split_quality",
    )
    decoded = {}
    try:
        for name in wanted:
            row = modules.get(name) or {}
            raw = zlib.decompress(base64.b64decode(
                str(row.get("source_zlib_base64") or ""), validate=True,
            ))
            declared = str(row.get("sha256") or "").lower()
            if (
                not raw
                or len(raw) != int(row.get("size_bytes") or 0)
                or not re.fullmatch(r"[0-9a-f]{64}", declared)
                or hashlib.sha256(raw).hexdigest() != declared
            ):
                raise ValueError("validator_source_identity_mismatch:" + name)
            decoded[name] = raw.decode("utf-8")
            detail["validator_module_sha256"][name] = declared

        package = types.ModuleType("splitpoint_runners")
        package.__package__ = "splitpoint_runners"
        package.__path__ = []
        sys.modules["splitpoint_runners"] = package
        for name in wanted:
            module = types.ModuleType(name)
            module.__package__ = "splitpoint_runners"
            module.__file__ = "<read-only-cache-probe:%s>" % name
            sys.modules[name] = module
            setattr(package, name.rsplit(".", 1)[-1], module)
            exec(compile(decoded[name], module.__file__, "exec"), module.__dict__)
        validator = getattr(
            sys.modules["splitpoint_runners.native_split_quality"],
            "validate_native_split_quality_binding", None,
        )
        if not callable(validator):
            raise AttributeError("strict validator symbol unavailable")
        _STRICT_NATIVE_QUALITY_VALIDATOR = validator
        _STRICT_NATIVE_QUALITY_VALIDATOR_DETAIL = detail
        return validator, detail
    except Exception as exc:
        detail["validator_error"] = "%s:%s" % (type(exc).__name__, exc)
        _STRICT_NATIVE_QUALITY_VALIDATOR = None
        _STRICT_NATIVE_QUALITY_VALIDATOR_DETAIL = detail
        return None, detail

base = pathlib.Path(P["remote_base"]).resolve(strict=True)
cache_parent = base / "_onnx_splitpoint_cache"
managed = cache_parent / "tensorrt_managed_v27516"
current = managed / P["stable_key"]
model_prefix = str(P["stable_key"]).rsplit("-", 1)[0] + "-"
legacy_suite_key = str(P.get("legacy_suite_key") or "")
canonical_full_sha256 = {
    str(value).lower() for value in list(P.get("canonical_full_sha256") or [])
    if re.fullmatch(r"[0-9a-f]{64}", str(value).lower())
}
scan_errors = []
try:
    managed_info = os.lstat(managed)
except FileNotFoundError:
    managed_info = None
except OSError as exc:
    managed_info = None
    scan_errors.append(
        "managed_cache_root_unreadable:%s:%s" % (managed, type(exc).__name__)
    )
if managed_info is not None and (
    not pathlib.Path(managed).is_dir() or pathlib.Path(managed).is_symlink()
):
    scan_errors.append("managed_cache_root_not_plain_directory:%s" % managed)

def namespace_has_canonical_full(namespace):
    if not canonical_full_sha256:
        return False
    try:
        candidates = sorted(pathlib.Path(namespace).rglob(
            "engine_build_receipt.json"
        ))
    except OSError as exc:
        scan_errors.append(
            "canonical_full_inventory:%s:%s" % (
                pathlib.Path(namespace), type(exc).__name__,
            )
        )
        return False
    for candidate in candidates[:8192]:
        raw = load(candidate)
        if str(raw.get("source_onnx_sha256") or "").lower() in canonical_full_sha256:
            return True
    return False

roots = []
for root, namespace in ((current, "current"),):
    if root.is_dir() and not root.is_symlink():
        owner = load_owner(root)
        owner_builder_abi = str(
            owner.get("trt_builder_abi_sha256") or ""
        ).lower()
        expected_builder_abi = str(P["builder_abi_sha256"]).lower()
        owner_status = (
            "current_owner_missing"
            if not owner
            else "current_owner_builder_abi_mismatch"
            if owner_builder_abi != expected_builder_abi
            else "exact_current"
        )
        roots.append((root, namespace, owner_status))
for parent, namespace in ((managed, "managed_legacy"),):
    if not parent.is_dir() or parent.is_symlink():
        continue
    try:
        children = sorted(parent.iterdir(), key=lambda x: x.name)
    except OSError as exc:
        scan_errors.append(
            "namespace_inventory:%s:%s" % (parent, type(exc).__name__)
        )
        continue
    for child in children:
        if child == current or not child.is_dir() or child.is_symlink():
            continue
        exact_legacy_source = bool(
            legacy_suite_key and child.name == legacy_suite_key
        )
        same_model_namespace = child.name.startswith(model_prefix)
        if not exact_legacy_source and not same_model_namespace:
            continue
        owner = load_owner(child)
        owner_builder_abi = str(
            owner.get("trt_builder_abi_sha256") or ""
        ).lower()
        expected_builder_abi = str(P["builder_abi_sha256"]).lower()
        owner_stable_key = str(
            owner.get("trt_engine_cache_key") or ""
        )
        owner_status = (
            "legacy_owner_missing"
            if not owner
            else "legacy_owner_builder_abi_mismatch"
            if owner_builder_abi != expected_builder_abi
            else "legacy_namespace_not_runtime_migratable"
            if not exact_legacy_source and not namespace_has_canonical_full(child)
            else "legacy_owner_engine_key_changed"
            if owner_stable_key not in {"", str(P["stable_key"])}
            else "legacy_owner_builder_abi_verified"
        )
        roots.append((child, namespace, owner_status))

# Match runtime migration order: current, exact former suite, then lexical
# older model namespaces. Invalid duplicates never hide a later valid receipt.
roots.sort(key=lambda row: (
    0 if row[1] == "current" else 1 if row[0].name == legacy_suite_key else 2,
    row[0].name,
))
receipts = []
seen = set()
for root, namespace, owner_status in roots:
    try:
        paths = sorted(root.rglob("engine_build_receipt.json"))
    except OSError as exc:
        scan_errors.append(
            "receipt_inventory:%s:%s" % (root, type(exc).__name__)
        )
        paths = []
    for path in paths[:8192]:
        try:
            resolved = path.resolve(strict=True)
        except OSError:
            continue
        if resolved in seen or not within(resolved, root):
            continue
        seen.add(resolved)
        receipts.append((resolved, root, namespace, owner_status))

def native_binding_for(receipt_path, root, requirement, receipt):
    for parent in receipt_path.parents:
        if not within(parent, root):
            break
        candidate = parent / "native_split_quality_binding.json"
        if not candidate.is_file() or candidate.is_symlink():
            continue
        binding = load(candidate)
        validator, validator_detail = strict_native_quality_validator()
        if validator is None:
            return False, "native_binding_strict_validation_unavailable", {
                **validator_detail, "binding_path": str(candidate),
            }
        expected_identity = {
            "model": str(P.get("model_id") or ""),
            "case": str(requirement.get("case_id") or ""),
            "setup_id": str(P.get("setup_id") or ""),
            "backend": str(requirement.get("expected_backend") or ""),
            "task": str(requirement.get("expected_task") or ""),
            "precision": str(requirement.get("engine_precision") or ""),
        }
        if any(not str(value or "").strip() for value in expected_identity.values()):
            return False, "native_binding_strict_validation_unavailable", {
                **validator_detail, "binding_path": str(candidate),
                "validator_error": "expected_identity_incomplete",
                "expected_identity": expected_identity,
            }
        try:
            verified, validation_status = validator(
                binding, expected_identity=expected_identity,
                verification_mode="local",
            )
        except Exception as exc:
            return False, "native_binding_strict_validation_unavailable", {
                **validator_detail, "binding_path": str(candidate),
                "validator_error": "%s:%s" % (type(exc).__name__, exc),
                "expected_identity": expected_identity,
            }
        if not isinstance(verified, dict):
            return False, "native_binding_strict_validation_failed", {
                **validator_detail, "binding_path": str(candidate),
                "validator_status": str(validation_status or ""),
                "expected_identity": expected_identity,
            }
        binding = verified
        declared = str(binding.get("binding_sha256") or "").lower()
        artifacts = binding.get("artifacts") or {}
        source_row = artifacts.get("source_part2_onnx") or {}
        build_row = artifacts.get("build_part2_onnx") or {}
        engine_row = artifacts.get("engine") or {}
        if str(source_row.get("sha256") or "").lower() != requirement["source_sha256"]:
            return False, "source_onnx_mismatch", {}
        expected_part1 = str(requirement.get("part1_artifact_sha256") or "").lower()
        if expected_part1:
            actual_part1 = artifacts.get("part1_runtime") or {}
            if (str(actual_part1.get("sha256") or "").lower() != expected_part1
                or int(actual_part1.get("size_bytes") or 0)
                   != int(requirement.get("part1_artifact_size_bytes") or 0)):
                return False, "native_binding_part1_mismatch", {}
        expected_policy = str(requirement.get("native_policy_sha256") or "").lower()
        if expected_policy and str((binding.get("preselection") or {}).get(
            "policy_sha256") or "").lower() != expected_policy:
            return False, "native_binding_policy_mismatch", {}
        receipt_source_path = pathlib.Path(str(receipt.get("source_onnx") or ""))
        receipt_engine_path = pathlib.Path(str(receipt.get("engine") or ""))
        try:
            crosslinks_match = (
                pathlib.Path(str(build_row.get("path") or "")).resolve(strict=True)
                == receipt_source_path.resolve(strict=True)
                and pathlib.Path(str(engine_row.get("path") or "")).resolve(strict=True)
                == receipt_engine_path.resolve(strict=True)
            )
        except OSError:
            crosslinks_match = False
        if not crosslinks_match:
            return False, "native_binding_strict_validation_failed", {
                **validator_detail, "binding_path": str(candidate),
                "validator_status": "receipt_artifact_path_crosslink_mismatch",
            }
        receipt_row = artifacts.get("engine_build_receipt") or {}
        receipt_artifact = pathlib.Path(str(receipt_row.get("path") or ""))
        receipt_file_sha = str(receipt_row.get("sha256") or "").lower()
        bound_receipt_inner = str(
            binding.get("engine_build_receipt_sha256") or ""
        ).lower()
        if (
            not receipt_artifact.is_file() or receipt_artifact.is_symlink()
            or receipt_artifact.resolve() != receipt_path.resolve()
            or not within(receipt_artifact, root)
            or not re.fullmatch(r"[0-9a-f]{64}", receipt_file_sha)
            or sha_file(receipt_artifact) != receipt_file_sha
            or bound_receipt_inner
               and bound_receipt_inner
               != str(receipt.get("receipt_sha256") or "").lower()
        ):
            return False, "native_binding_strict_validation_failed", {
                **validator_detail, "binding_path": str(candidate),
                "validator_status": "receipt_path_crosslink_mismatch",
            }
        shape_contract = dict(requirement.get("shape_contract") or {})
        meta_row = artifacts.get("native_trt_meta") or {}
        if isinstance(meta_row, dict) and str(meta_row.get("path") or ""):
            meta_path = pathlib.Path(str(meta_row.get("path") or ""))
            meta_sha = str(meta_row.get("sha256") or "").lower()
            if (
                not meta_path.is_file() or meta_path.is_symlink()
                or not within(meta_path, root)
                or not re.fullmatch(r"[0-9a-f]{64}", meta_sha)
                or sha_file(meta_path) != meta_sha
            ):
                return False, "native_binding_artifact_mismatch", {}
            meta = load(meta_path)
            successful_shape = str(
                meta.get("build_success_shapes_arg")
                if meta.get("build_success_shapes_arg") is not None
                else meta.get("shapes_arg") or ""
            )
            shape_contract = {
                "complete": True,
                "expected_shapes": successful_shape,
                "allowed_shapes": [successful_shape],
                "policy": "sealed_native_trt_meta",
            }
        return True, "native_split_quality_binding_verified", {
            "binding_path": str(candidate), "binding_sha256": declared,
            "shape_contract": shape_contract,
            "source_binding": "strict_native_split_quality_binding",
            "quality_binding_addressed": True,
            "validator_status": str(validation_status or ""),
            **validator_detail,
        }
    return False, "native_binding_not_found", {}

def generic_uint8_cast_bridge_for(
    receipt_path, root, requirement, receipt,
):
    """Verify the existing generic Cast bridge relation without mutation."""
    meta_path = receipt_path.parent / "uint8_cast_bridge_meta.json"
    if not meta_path.is_file() or meta_path.is_symlink():
        return False, "generic_bridge_metadata_missing", {}
    if not within(meta_path, root):
        return False, "generic_bridge_metadata_outside_namespace", {}
    meta = load(meta_path)
    if (
        meta.get("schema") != "onnx-splitpoint/uint8-cast-bridge"
        or meta.get("schema_version") != 1
    ):
        return False, "generic_bridge_metadata_invalid", {}
    source = pathlib.Path(str(meta.get("source") or ""))
    bridge = pathlib.Path(str(meta.get("bridge") or ""))
    receipt_source = pathlib.Path(str(receipt.get("source_onnx") or ""))
    try:
        if (
            not bridge.is_file() or bridge.is_symlink()
            or bridge.resolve(strict=True) != receipt_source.resolve(strict=True)
            or not within(bridge, root)
        ):
            return False, "generic_bridge_path_mismatch", {}
    except OSError:
        return False, "generic_bridge_path_mismatch", {}
    bridge_sha = sha_file(bridge)
    if bridge_sha != str(receipt.get("source_onnx_sha256") or "").lower():
        return False, "generic_bridge_sha256_mismatch", {}
    declared_bridge_sha = str(meta.get("bridge_sha256") or "").lower()
    try:
        declared_bridge_size = int(meta.get("bridge_size_bytes"))
        actual_bridge_size = int(bridge.stat().st_size)
    except (TypeError, ValueError, OSError):
        return False, "generic_bridge_metadata_incomplete", {}
    if (
        not re.fullmatch(r"[0-9a-f]{64}", declared_bridge_sha)
        or declared_bridge_sha != bridge_sha
        or declared_bridge_size != actual_bridge_size
    ):
        return False, "generic_bridge_sha256_mismatch", {}

    expected_source_sha = str(requirement.get("source_sha256") or "").lower()
    declared_source_sha = str(meta.get("source_sha256") or "").lower()
    try:
        declared_source_size = int(meta.get("source_size_bytes"))
        expected_source_size = int(requirement.get("source_size_bytes"))
    except (TypeError, ValueError):
        return False, "generic_bridge_metadata_incomplete", {}
    if (
        not re.fullmatch(r"[0-9a-f]{64}", declared_source_sha)
        or declared_source_sha != expected_source_sha
        or declared_source_size != expected_source_size
    ):
        return False, "generic_bridge_source_mismatch", {}
    source_verification = ""
    try:
        if source.is_file() and not source.is_symlink():
            if (
                sha_file(source) != expected_source_sha
                or int(source.stat().st_size) != expected_source_size
            ):
                return False, "generic_bridge_source_mismatch", {}
            source_verification = "source_bytes_verified"
        else:
            # Newer generic bridge metadata records both identities.  The
            # original uploaded suite may already have been retired, but the
            # existing metadata still binds its controller-computed digest to
            # the receipt-sealed bridge bytes.
            source_verification = "metadata_source_and_bridge_sha256_verified"
    except OSError:
        return False, "generic_bridge_source_identity_unprovable", {}
    if (
        str(meta.get("input_dtype") or "").upper() != "UINT8"
        or str(meta.get("cast_to") or "").upper() != "FLOAT"
        or not str(meta.get("input_name") or "")
        or not str(meta.get("cast_output") or "")
    ):
        return False, "generic_bridge_contract_mismatch", {}
    try:
        if int(meta.get("replaced_uses") or 0) <= 0:
            return False, "generic_bridge_contract_mismatch", {}
    except (TypeError, ValueError):
        return False, "generic_bridge_contract_mismatch", {}
    precision_tag = str(meta.get("precision_tag") or "").lower()
    if precision_tag and precision_tag != "uint8_cast_fp16":
        return False, "generic_bridge_contract_mismatch", {}
    return True, "generic_uint8_cast_bridge_verified", {
        "generic_bridge_metadata_path": str(meta_path),
        "generic_bridge_source_verification": source_verification,
        "generic_bridge_sha256": bridge_sha,
        "generic_bridge_size_bytes": actual_bridge_size,
        "source_binding": "generic_uint8_cast_bridge",
    }

def verify(receipt_path, root, namespace, owner_status, requirement):
    if owner_status == "current_owner_missing":
        return None, "current_owner_missing", {}
    if owner_status == "current_owner_builder_abi_mismatch":
        return None, "current_owner_builder_abi_mismatch", {}
    if owner_status == "legacy_owner_missing":
        return None, "legacy_owner_missing", {}
    if owner_status == "legacy_owner_builder_abi_mismatch":
        return None, "legacy_owner_builder_abi_mismatch", {}
    if owner_status == "legacy_namespace_not_runtime_migratable":
        return None, "legacy_namespace_not_runtime_migratable", {}
    raw = load(receipt_path)
    declared = str(raw.pop("receipt_sha256", "") or "").lower()
    if (
        not re.fullmatch(r"[0-9a-f]{64}", declared)
        or canonical_sha(raw) != declared
    ):
        return None, "receipt_integrity_mismatch", {}
    raw["receipt_sha256"] = declared
    if (
        raw.get("schema") != "onnx-splitpoint/tensorrt-engine-build-receipt"
        or raw.get("schema_version") != 1
        or raw.get("build_returncode") != 0
        or raw.get("dry_run") is not False
    ):
        return None, "receipt_schema_or_status_invalid", {}
    source = pathlib.Path(str(raw.get("source_onnx") or ""))
    engine = pathlib.Path(str(raw.get("engine") or ""))
    builder = pathlib.Path(str(raw.get("trtexec") or ""))
    if not all(within(path, root) for path in (source, engine, receipt_path)):
        return None, "receipt_path_outside_namespace", {}
    if receipt_path.parent.resolve() != engine.parent.resolve():
        return None, "receipt_engine_leaf_mismatch", {}
    if not source.is_file() or source.is_symlink():
        return None, "source_onnx_missing", {}
    if not engine.is_file() or engine.is_symlink() or engine.stat().st_size <= 0:
        return None, "engine_not_found", {}
    if not builder.is_file() or builder.is_symlink():
        return None, "builder_not_found", {}
    source_sha = sha_file(source)
    engine_sha = sha_file(engine)
    builder_sha = sha_file(builder)
    if source_sha != str(raw.get("source_onnx_sha256") or "").lower():
        return None, "source_onnx_mismatch", {}
    if engine_sha != str(raw.get("engine_sha256") or "").lower():
        return None, "engine_sha256_mismatch", {}
    if (
        builder_sha != str(raw.get("trtexec_sha256") or "").lower()
        or builder_sha != str(P["trtexec_sha256"]).lower()
    ):
        return None, "builder_abi_mismatch", {}
    command = [str(value) for value in list(raw.get("command") or [])]
    if (
        not command
        or pathlib.Path(command[0]).resolve() != builder.resolve()
        or [v for v in command[1:] if v.startswith("--onnx=")]
           != ["--onnx=" + str(source.resolve())]
        or [v for v in command[1:] if v.startswith("--saveEngine=")]
           != ["--saveEngine=" + str(engine.resolve())]
    ):
        return None, "receipt_command_mismatch", {}
    unknown_build_args = []
    for value in command[1:]:
        if value.startswith((
            "--onnx=", "--saveEngine=", "--timingCacheFile=",
            "--shapes=", "--workspace=", "--memPoolSize=workspace:",
        )):
            continue
        if value in {"--fp16", "--int8", "--verbose"}:
            continue
        unknown_build_args.append(value)
    if unknown_build_args:
        return None, "receipt_unknown_build_args", {
            "unknown_build_args": unknown_build_args,
        }
    precision = str(requirement.get("engine_precision") or P["precision"]).lower()
    if precision in {
        "fp16", "uint8_cast_fp16", "uint8_dequant_fp16",
        "float32_layout_fp16",
    }:
        expected_precision_flags = ["--fp16"]
    elif precision in {"fp32", "float32"}:
        expected_precision_flags = []
    elif precision == "int8":
        expected_precision_flags = ["--int8"]
    else:
        return None, "build_contract_mismatch", {}
    actual_precision_flags = [
        value for value in command[1:] if value in {"--fp16", "--int8"}
    ]
    if actual_precision_flags != expected_precision_flags:
        return None, "build_contract_mismatch", {}
    workspace_flags = []
    for value in command[1:]:
        try:
            if value.startswith("--workspace="):
                workspace_flags.append(("workspace", int(value.split("=", 1)[1])))
            elif value.startswith("--memPoolSize=workspace:"):
                workspace_flags.append(("mempool", int(value.rsplit(":", 1)[1])))
        except ValueError:
            return None, "receipt_workspace_mismatch", {}
    expected_workspace = int(P.get("workspace_mb") or 0)
    if expected_workspace > 0:
        if len(workspace_flags) != 1 or workspace_flags[0] not in {
            ("workspace", expected_workspace),
            ("mempool", expected_workspace),
        }:
            return None, "receipt_workspace_mismatch", {}
    elif workspace_flags:
        return None, "receipt_workspace_mismatch", {}
    shape_args = [
        value.split("=", 1)[1]
        for value in command[1:] if value.startswith("--shapes=")
    ]
    if len(shape_args) > 1:
        return None, "receipt_shape_mismatch", {}
    actual_shape = shape_args[0] if shape_args else ""
    parts = [part.lower() for part in engine.parts]
    role = requirement["role"]
    special_bridge_precisions = {
        "uint8_cast_fp16", "uint8_dequant_fp16",
        "float32_layout_fp16",
    }
    if role == "trt_full":
        if "full" not in parts or not engine.name.startswith("full_"):
            return None, "role_path_mismatch", {}
        if source_sha != requirement["source_sha256"]:
            return None, "source_onnx_mismatch", {}
        bridge = {}
        shape_contract = dict(requirement.get("shape_contract") or {})
    elif role == "trt_p1":
        case_id = str(requirement.get("case_id") or "").lower()
        if case_id not in parts or "part1" not in parts:
            return None, "case_or_role_path_mismatch", {}
        # Part-1 consumes the model's image inputs directly. Part-2 Cast,
        # dequantization and quality-bound transformed inputs cannot attest it.
        if source_sha != requirement["source_sha256"]:
            return None, "source_onnx_mismatch", {}
        bridge = {"source_binding": "direct_part1_source"}
        shape_contract = dict(requirement.get("shape_contract") or {})
    else:
        case_id = str(requirement.get("case_id") or "").lower()
        if case_id not in parts or "part2" not in parts:
            return None, "case_or_role_path_mismatch", {}
        if source_sha == requirement["source_sha256"]:
            if precision in special_bridge_precisions:
                return None, "special_precision_direct_source_not_allowed", {}
            bridge = {"source_binding": "direct_part2_source"}
            shape_contract = dict(requirement.get("shape_contract") or {})
        else:
            valid = False
            reason = "source_onnx_mismatch"
            bridge = {}
            generic_reason = ""
            generic_detail = {}
            if precision == "uint8_cast_fp16":
                valid, generic_reason, generic_detail = (
                    generic_uint8_cast_bridge_for(
                        receipt_path, root, requirement, raw,
                    )
                )
                if valid:
                    reason, bridge = generic_reason, generic_detail
            if not valid and precision in special_bridge_precisions:
                native_valid, native_reason, native_detail = native_binding_for(
                    receipt_path, root, requirement, raw,
                )
                if native_valid:
                    # Quality-FIRST engines intentionally live below their
                    # binding-addressed cache tree.  The exact suite-vendored
                    # runtime validator has re-hashed all eight artifacts and
                    # verified every identity/cross-link, so this is the one
                    # non-canonical Part-2 path that is execution-equivalent.
                    valid = True
                    reason, bridge = native_reason, native_detail
                elif precision == "uint8_cast_fp16" and generic_reason:
                    reason, bridge = generic_reason, generic_detail
                else:
                    reason, bridge = native_reason, native_detail
            if not valid:
                return None, reason, bridge
            shape_contract = dict(
                bridge.pop("shape_contract", None)
                or requirement.get("shape_contract") or {}
            )
    expected_engine_name = str(
        requirement.get("expected_engine_filename") or ""
    )
    if expected_engine_name and engine.name != expected_engine_name:
        return None, "canonical_engine_filename_mismatch", {
            "expected_engine_filename": expected_engine_name,
            "actual_engine_filename": engine.name,
        }
    if namespace == "current" and not bool(
        bridge.get("quality_binding_addressed")
    ):
        expected_leaf = root / str(
            requirement.get("expected_relative_leaf") or ""
        )
        try:
            leaf_matches = (
                receipt_path.parent.resolve(strict=True)
                == expected_leaf.resolve(strict=True)
            )
        except OSError:
            leaf_matches = False
        if not leaf_matches:
            return None, "canonical_leaf_mismatch", {
                "expected_leaf": str(expected_leaf),
                "actual_leaf": str(receipt_path.parent),
            }
    # A changed namespace key is not itself an artifact incompatibility.
    # Reuse across it needs a fully known current source/build contract; an
    # unreadable ONNX's permissive empty-shape fallback cannot authorize it.
    if owner_status == "legacy_owner_engine_key_changed":
        if not bool(shape_contract.get("complete")):
            return None, "shape_contract_unavailable", {
                "shape_contract": shape_contract,
            }
        if bridge.get("quality_binding_addressed"):
            # The generic migration below cannot rebase the eight-artifact
            # native quality binding. Do not promise a migratable cache hit.
            return None, "legacy_native_binding_migration_unavailable", {}
        if not source.resolve().is_relative_to(receipt_path.parent.resolve()):
            return None, "receipt_leaf_is_not_self_contained", {}
        target_leaf = current / str(requirement.get("expected_relative_leaf") or "")
        if os.path.lexists(target_leaf):
            return None, "current_cache_leaf_blocks_migration", {
                "target_leaf": str(target_leaf),
            }
    allowed_shapes = [
        str(value) for value in list(shape_contract.get("allowed_shapes") or [])
    ]
    if actual_shape not in allowed_shapes:
        if not bool(shape_contract.get("complete")) and actual_shape:
            return None, "shape_contract_unavailable", {
                "actual_shapes": actual_shape,
                "shape_contract": shape_contract,
            }
        return None, "receipt_shape_mismatch", {
            "actual_shapes": actual_shape,
            "shape_contract": shape_contract,
        }
    return {
        "artifact_path": str(engine),
        "receipt_path": str(receipt_path),
        "identity": declared,
        "source_namespace": namespace,
        "evidence": {
            "verification_reason": "engine_build_receipt_verified",
            "source_onnx_sha256": source_sha,
            "engine_sha256": engine_sha,
            "trtexec_sha256": builder_sha,
            "precision_flags": actual_precision_flags,
            "workspace": workspace_flags[0] if workspace_flags else None,
            "shapes": actual_shape,
            "namespace_owner_status": owner_status,
            "previous_engine_cache_key": str(
                load_owner(root).get("trt_engine_cache_key") or ""
            ),
            "current_engine_cache_key": str(P["stable_key"]),
            **bridge,
        },
    }, "compatible_receipt", {}

observations = []
for requirement in P["requirements"]:
    failures = []
    hit = None
    for receipt_path, root, namespace, owner_status in receipts:
        try:
            candidate, reason, detail = verify(
                receipt_path, root, namespace, owner_status, requirement,
            )
        except Exception as exc:
            candidate = None
            reason = "candidate_validation_error"
            detail = {"error_class": type(exc).__name__}
        if candidate is not None:
            hit = candidate
            break
        failures.append({
            "receipt_path": str(receipt_path), "reason": reason,
            "namespace_owner_status": owner_status,
            "previous_engine_cache_key": str(
                load_owner(root).get("trt_engine_cache_key") or ""
            ),
            "current_engine_cache_key": str(P["stable_key"]),
            **detail,
        })
    if hit is not None:
        observations.append({
            "model_id": P["model_id"], "role": requirement["role"],
            "item_id": requirement["item_id"], "status": "HIT",
            "reason": "compatible_receipt", "setup_id": P["setup_id"],
            **hit,
        })
        continue
    reason = (
        "remote_cache_unreadable" if scan_errors
        else "legacy_candidate_not_found" if len(roots) > 1 else "not_found"
    )
    priorities = (
        "builder_abi_mismatch", "source_onnx_mismatch",
        "engine_sha256_mismatch", "receipt_integrity_mismatch",
        "native_binding_integrity_mismatch", "native_binding_receipt_mismatch",
        "native_binding_part1_mismatch", "native_binding_policy_mismatch",
        "generic_bridge_source_mismatch", "generic_bridge_sha256_mismatch",
        "generic_bridge_contract_mismatch",
        "special_precision_direct_source_not_allowed",
        "canonical_leaf_mismatch", "canonical_engine_filename_mismatch",
        "receipt_unknown_build_args", "build_contract_mismatch",
        "receipt_workspace_mismatch", "receipt_shape_mismatch",
        "shape_contract_unavailable",
        "current_owner_builder_abi_mismatch", "current_owner_missing",
        "legacy_owner_builder_abi_mismatch", "legacy_owner_missing",
        "legacy_namespace_not_runtime_migratable",
        "legacy_native_binding_migration_unavailable",
        "receipt_leaf_is_not_self_contained",
        "current_cache_leaf_blocks_migration",
        "native_binding_strict_validation_unavailable",
        "native_binding_strict_validation_failed",
        "generic_bridge_metadata_incomplete",
        "generic_bridge_metadata_missing",
        "generic_bridge_source_identity_unprovable",
    )
    if not scan_errors:
        for preferred in priorities:
            if any(row.get("reason") == preferred for row in failures):
                reason = preferred
                break
    history = managed / ".eviction_history.jsonl"
    if reason in {"not_found", "legacy_candidate_not_found"} and history.is_file():
        try:
            if any(P["stable_key"] in line for line in history.read_text(
                encoding="utf-8", errors="replace",
            ).splitlines()[-4096:]):
                reason = "evicted_by_retention"
        except OSError:
            pass
    observation_status = "UNKNOWN" if reason in {
        "shape_contract_unavailable", "current_owner_missing",
        "generic_bridge_metadata_missing",
        "generic_bridge_metadata_incomplete",
        "generic_bridge_source_identity_unprovable",
        "native_binding_strict_validation_unavailable",
        "native_binding_strict_validation_failed",
        "legacy_native_binding_migration_unavailable",
        "current_cache_leaf_blocks_migration",
        "remote_cache_unreadable",
    } else "MISS"
    observations.append({
        "model_id": P["model_id"], "role": requirement["role"],
        "item_id": requirement["item_id"], "status": observation_status,
        "reason": reason, "setup_id": P["setup_id"],
        "artifact_path": "", "receipt_path": "", "identity": "",
        "source_namespace": "", "evidence": {
            "candidate_failure_count": len(failures),
            "candidate_failures": failures[:64],
            "scan_errors": scan_errors[:64],
            "expected_source_sha256": requirement["source_sha256"],
        },
    })

result = {
    "schema": "onnx-splitpoint/remote-trt-cache-preflight",
    "schema_version": 1,
    "status": "ok",
    "setup_id": P["setup_id"],
    "model_id": P["model_id"],
    "stable_key": P["stable_key"],
    "remote_base": str(base),
    "current_namespace": str(current),
    "namespace_count": len(roots),
    "receipt_count": len(receipts),
    "observations": observations,
}
print("SPLITPOINT_REMOTE_TRT_CACHE_PREFLIGHT=" + json.dumps(
    result, sort_keys=True, separators=(",", ":"),
))
'''
    return (
        "python3 -B - <<'SPLITPOINT_REMOTE_TRT_CACHE_PREFLIGHT_PY'\n"
        + script.replace("__PAYLOAD__", encoded)
        + "\nSPLITPOINT_REMOTE_TRT_CACHE_PREFLIGHT_PY"
    )


def probe_remote_trt_artifact_cache(
    *,
    transport: SSHTransport | None,
    suite_dir: str | Path,
    setup_id: str,
    setup_accelerator: str = "",
    active_run_ids: Sequence[str] | None = None,
    args: "RemoteBenchmarkArgs | None" = None,
    timeout_s: int = 180,
    resolved_remote_base: str = "",
    builder_abi: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Probe one setup's persistent TRT cache without upload or mutation.

    The method reuses the normal model/runtime/builder identities, validates
    cached receipts and bytes on their owning host, and returns setup-scoped
    observations.  Host/config/ABI failures are explicit ``UNKNOWN`` rows;
    they are never silently converted into cache misses.
    """

    suite = Path(suite_dir).expanduser().resolve(strict=True)
    setup = str(setup_id or "").strip() or "remote_setup"
    model_id = _trt_preflight_model_id(suite)
    expected_task = _infer_suite_benchmark_task(
        suite, suite / "benchmark_set.json",
    )
    if expected_task not in {"classification", "detection"}:
        expected_task = ""
    native_quality_validator = _suite_native_quality_validator_payload(suite)
    native_quality_validator_evidence = {
        key: value for key, value in native_quality_validator.items()
        if key != "modules"
    }
    native_quality_validator_evidence["modules"] = {
        name: {
            key: value for key, value in dict(row).items()
            if key != "source_zlib_base64"
        }
        for name, row in dict(
            native_quality_validator.get("modules") or {}
        ).items()
        if isinstance(row, Mapping)
    }
    requirement_plan = _trt_preflight_run_requirements(
        suite, active_run_ids=active_run_ids,
    )
    runtime_contract = _trt_engine_runtime_contract(
        suite, args=args, active_run_ids=active_run_ids,
    )
    precision = str(
        runtime_contract.get("native_trt_precision") or "fp16"
    ).lower()
    trt_runtime = str(
        runtime_contract.get("trt_runtime") or "native_preferred"
    ).strip().lower()
    try:
        workspace_mb = int(
            runtime_contract.get("native_trt_workspace_mb") or 0
        )
    except (TypeError, ValueError):
        workspace_mb = -1
    requirements: list[dict[str, Any]] = []
    local_errors: list[str] = []
    if requirement_plan["full_required"]:
        try:
            full_sources = _canonical_model_onnx_sources(suite)
            for index, source in enumerate(full_sources):
                sha = _stable_file_sha256(source)
                suffix = "full" if len(full_sources) == 1 else f"full:{sha[:12]}"
                requirements.append({
                    "role": "trt_full", "case_id": "",
                    "item_id": f"{setup}/{suffix}",
                    "source_path": str(source),
                    "source_sha256": sha,
                    "source_size_bytes": int(source.stat().st_size),
                    "shape_contract": _trt_preflight_shape_contract(source),
                    "run_ids": list(requirement_plan["full_run_ids"]),
                })
        except Exception as exc:
            local_errors.append(f"full_source:{type(exc).__name__}:{exc}")
            requirements.append({
                "role": "trt_full", "case_id": "",
                "item_id": f"{setup}/full", "source_path": "",
                "source_sha256": "", "source_size_bytes": 0,
                "run_ids": list(requirement_plan["full_run_ids"]),
                "local_error": local_errors[-1],
            })
    for case_id in requirement_plan["p1_cases"]:
        run_ids = list(requirement_plan["p1_run_ids_by_case"].get(case_id) or [])
        requirement = {
            "role": "trt_p1", "case_id": case_id,
            "item_id": f"{setup}/{case_id}", "run_ids": run_ids,
        }
        try:
            source = _trt_preflight_part1_source(suite, case_id)
            requirement.update({
                "source_path": str(source), "source_sha256": _stable_file_sha256(source),
                "source_size_bytes": int(source.stat().st_size),
                "shape_contract": _trt_preflight_shape_contract(source),
            })
        except Exception as exc:
            local_errors.append(f"part1_source:{case_id}:{type(exc).__name__}:{exc}")
            requirement.update({
                "source_path": "", "source_sha256": "", "source_size_bytes": 0,
                "local_error": local_errors[-1],
            })
        requirements.append(requirement)
    for case_id in requirement_plan["p2_cases"]:
        try:
            source = _trt_preflight_part2_source(suite, case_id)
            sha = _stable_file_sha256(source)
            requirements.append({
                "role": "trt_p2", "case_id": case_id,
                "item_id": f"{setup}/{case_id}",
                "source_path": str(source), "source_sha256": sha,
                "source_size_bytes": int(source.stat().st_size),
                "shape_contract": _trt_preflight_shape_contract(source),
                "expected_task": expected_task,
                "expected_backend": _trt_preflight_native_backend(
                    suite,
                    run_ids=list(
                        requirement_plan["p2_run_ids_by_case"].get(case_id)
                        or []
                    ),
                    setup_accelerator=setup_accelerator,
                ),
                "run_ids": list(
                    requirement_plan["p2_run_ids_by_case"].get(case_id) or []
                ),
            })
        except Exception as exc:
            local_errors.append(
                f"part2_source:{case_id}:{type(exc).__name__}:{exc}"
            )
            requirements.append({
                "role": "trt_p2", "case_id": case_id,
                "item_id": f"{setup}/{case_id}", "source_path": "",
                "source_sha256": "", "source_size_bytes": 0,
                "expected_task": expected_task,
                "expected_backend": _trt_preflight_native_backend(
                    suite,
                    run_ids=list(
                        requirement_plan["p2_run_ids_by_case"].get(case_id)
                        or []
                    ),
                    setup_accelerator=setup_accelerator,
                ),
                "run_ids": list(
                    requirement_plan["p2_run_ids_by_case"].get(case_id) or []
                ),
                "local_error": local_errors[-1],
            })

    for case_id in requirement_plan.get("generic_p2_cases", []):
        # This is an ordinary same-backend split, not the accelerator Native
        # quality bridge. Keep separate observations so a bridge HIT cannot
        # conceal a missing fp16 Part2 engine needed by this dispatch.
        requirement = {
            "role": "trt_p2", "case_id": case_id,
            "item_id": f"{setup}/{case_id}:generic",
            "run_ids": list(requirement_plan["generic_p2_run_ids_by_case"].get(case_id) or []),
            "recipe_source": "generic_same_backend_split",
        }
        try:
            source = _trt_preflight_part2_source(suite, case_id)
            requirement.update(source_path=str(source), source_sha256=_stable_file_sha256(source),
                source_size_bytes=int(source.stat().st_size),
                shape_contract=_trt_preflight_shape_contract(source))
        except Exception as exc:
            local_errors.append(f"generic_part2_source:{case_id}:{type(exc).__name__}:{exc}")
            requirement.update(source_path="", source_sha256="", source_size_bytes=0,
                local_error=local_errors[-1])
        requirements.append(requirement)

    special_bridge_precisions = {
        "uint8_cast_fp16", "uint8_dequant_fp16", "float32_layout_fp16",
    }
    for requirement in requirements:
        source_sha = str(requirement.get("source_sha256") or "").lower()
        if re.fullmatch(r"[0-9a-f]{64}", source_sha) is None:
            continue
        role = {"trt_full": "full", "trt_p1": "part1", "trt_p2": "part2"}[requirement["role"]]
        effective_precision = (
            "fp16"
            if role == "full" and precision in special_bridge_precisions
            else precision
        )
        if role == "part2" and requirement.get("recipe_source") != "generic_same_backend_split":
            try:
                native_policy = _trt_preflight_native_policy(
                    suite, model_id=model_id, setup_id=setup,
                    requirement=requirement,
                )
            except (TypeError, ValueError) as exc:
                local_errors.append(f"native_recipe:{requirement['case_id']}:{exc}")
                requirement["local_error"] = local_errors[-1]
                native_policy = None
            if native_policy is not None:
                effective_precision = str(native_policy["precision"])
                requirement["native_policy_sha256"] = native_policy["policy_sha256"]
                requirement["recipe_source"] = "native_split_quality_policy"
                try:
                    from ..runners.native_split_quality_runtime import _find_part1
                    part1 = _find_part1(
                        suite / str(requirement["case_id"]),
                        str(requirement["expected_backend"]),
                    )
                    requirement["part1_artifact_sha256"] = _stable_file_sha256(part1)
                    requirement["part1_artifact_size_bytes"] = int(part1.stat().st_size)
                except (OSError, RuntimeError, ValueError) as exc:
                    local_errors.append(
                        f"native_part1_source:{requirement['case_id']}:{exc}"
                    )
                    requirement["local_error"] = local_errors[-1]
                    requirement["local_error_reason"] = "native_part1_identity_unavailable"
                    from ..native_job_identity import native_comparison
                    requirement["dependency_backend"] = native_comparison(requirement.get("expected_backend"))
                    requirement["dependency_boundary"] = requirement.get("case_id")
        requirement["engine_precision"] = effective_precision
        requirement["expected_relative_leaf"] = str(
            _trt_persistent_engine_relative_dir(
                role=role,
                case_id=str(requirement.get("case_id") or ""),
                source_onnx_sha256=source_sha,
                precision=effective_precision,
            )
        )
        requirement["expected_engine_filename"] = (
            f"{role}_{effective_precision}.engine"
        )

    def unknown_rows(reason: str, detail: str = "", *, selected=None) -> list[dict[str, Any]]:
        return [{
            "model_id": model_id, "role": row["role"],
            "item_id": row["item_id"], "status": "UNKNOWN",
            "reason": reason, "setup_id": setup,
            "artifact_path": "", "receipt_path": "", "identity": "",
            "source_namespace": "", "evidence": {
                "detail": detail, "setup_accelerator": setup_accelerator,
                "dependency_backend": row.get("dependency_backend", ""),
                "dependency_boundary": row.get("dependency_boundary", ""),
                "dependency_stage": "part1" if row.get("dependency_backend") else "",
                "expected_source_sha256": row.get("source_sha256") or "",
                "active_run_ids": list(row.get("run_ids") or []),
            },
        } for row in (requirements if selected is None else selected)]

    result: dict[str, Any] = {
        "schema": "onnx-splitpoint/remote-trt-cache-preflight",
        "schema_version": 1,
        "status": "not_applicable" if not requirements else "pending",
        "setup_id": setup,
        "setup_accelerator": str(setup_accelerator or ""),
        "model_id": model_id,
        "suite_dir": str(suite),
        "active_run_ids": list(active_run_ids or []),
        "runtime_contract": runtime_contract,
        "requirement_plan": requirement_plan,
        "requirements": requirements,
        "native_quality_validator": native_quality_validator_evidence,
        "local_errors": local_errors,
        "observations": [],
        "read_only": True,
        "hardware_action_performed": False,
    }
    if not requirements:
        return result
    if trt_runtime in {"ort", "ort_trt", "onnxruntime", "onnxruntime_trt"}:
        # ORT TensorRT EP owns a different cache format and key space.  A
        # valid native ``.engine`` receipt proves nothing about that cache, so
        # keep the observation epistemically UNKNOWN until an ORT-EP-specific
        # read-only validator exists.  This branch intentionally precedes all
        # SSH/ABI work.
        result.update({
            "status": "unknown",
            "reason": "ort_tensorrt_cache_probe_not_implemented",
            "observations": unknown_rows(
                "ort_tensorrt_cache_probe_not_implemented",
                "native TensorRT receipts are not ORT TensorRT EP cache evidence",
            ),
        })
        return result
    if workspace_mb < 0:
        local_errors.append("native_trt_workspace_mb_invalid")
    if workspace_mb < 0:
        result.update(status="unknown", reason="native_trt_workspace_mb_invalid",
                      observations=unknown_rows("native_trt_workspace_mb_invalid"))
        return result
    # Missing split sources/producers are request-local. An independent Full
    # reference or another split still gets its exact read-only remote probe.
    unavailable = [row for row in requirements if row.get("local_error")]
    probe_requirements = [row for row in requirements if not row.get("local_error")]
    unavailable_observations = []
    for row in unavailable:
        unavailable_observations.extend(unknown_rows(
            row.get("local_error_reason") or "local_source_identity_unavailable",
            str(row["local_error"]), selected=[row]))
    if not probe_requirements:
        result.update(status="unknown", reason="local_source_identity_unavailable",
                      observations=unavailable_observations)
        return result
    if transport is None:
        result.update({
            "status": "unknown", "reason": "remote_trt_cache_probe_unavailable",
            "error": "remote transport is not configured",
            "observations": unknown_rows(
                "remote_trt_cache_probe_unavailable",
                "remote transport is not configured",
            ),
        })
        return result
    try:
        remote_base = str(resolved_remote_base or "").rstrip("/")
        if not remote_base:
            remote_base = transport.resolve_path_read_only(
                str(getattr(transport.host, "remote_base_dir", "") or "~/splitpoint_runs"),
                timeout_s=max(15, int(timeout_s)),
            ).rstrip("/")
        effective_builder_abi = dict(builder_abi or {})
        if not effective_builder_abi:
            effective_builder_abi = _remote_trt_builder_abi(
                transport,
                remote_venv=str(getattr(args, "remote_venv", "") or ""),
            )
        builder_cache_abi = _trt_engine_builder_abi_contract(
            effective_builder_abi
        )
        builder_abi_sha256 = hashlib.sha256(json.dumps(
            builder_cache_abi,
            sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        ).encode("utf-8")).hexdigest()
        stable_key = _stable_trt_engine_cache_key(
            suite, args=args, builder_abi=effective_builder_abi,
            active_run_ids=active_run_ids,
        )
        canonical_full_sha256 = [
            _stable_file_sha256(path)
            for path in _canonical_model_onnx_sources(suite)
        ]
        payload = {
            "remote_base": remote_base,
            "stable_key": stable_key,
            "legacy_suite_key": _stable_suite_cache_key(suite),
            "canonical_full_sha256": canonical_full_sha256,
            "setup_id": setup,
            "setup_accelerator": str(setup_accelerator or ""),
            "model_id": model_id,
            "precision": precision,
            "workspace_mb": workspace_mb,
            "builder_abi_sha256": builder_abi_sha256,
            "trtexec_sha256": str(
                effective_builder_abi.get("trtexec_sha256") or ""
            ),
            "native_quality_validator": native_quality_validator,
            "requirements": probe_requirements,
        }
        rc, output = transport.run_read_only(
            _remote_trt_cache_probe_command(payload),
            timeout=max(30, int(timeout_s)),
        )
        marker = "SPLITPOINT_REMOTE_TRT_CACHE_PREFLIGHT="
        if int(rc) != 0:
            raise RuntimeError(
                f"remote_cache_probe_failed_rc={rc}:"
                + str(output or "").strip()[-2000:]
            )
        remote_payload: dict[str, Any] | None = None
        for line in reversed(str(output or "").splitlines()):
            if line.startswith(marker):
                loaded = json.loads(line[len(marker):])
                if isinstance(loaded, Mapping):
                    remote_payload = dict(loaded)
                break
        if remote_payload is None:
            raise RuntimeError("remote_cache_probe_marker_missing")
        observations = [
            dict(row) for row in list(remote_payload.get("observations") or [])
            if isinstance(row, Mapping)
        ]
        if len(observations) != len(probe_requirements):
            raise RuntimeError("remote_cache_probe_observation_count_mismatch")
        result.update({
            **remote_payload,
            "setup_accelerator": str(setup_accelerator or ""),
            "suite_dir": str(suite),
            "active_run_ids": list(active_run_ids or []),
            "requirement_plan": requirement_plan,
            "requirements": requirements,
            "local_errors": local_errors,
            "observations": observations + unavailable_observations,
            "status": "partial" if unavailable else remote_payload.get("status", "ok"),
            "builder_abi": effective_builder_abi,
            "builder_cache_abi": builder_cache_abi,
            "builder_abi_sha256": builder_abi_sha256,
            "read_only": True,
            "hardware_action_performed": False,
        })
        return result
    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}"
        result.update({
            "status": "unknown", "reason": "remote_trt_cache_probe_unavailable",
            "error": detail,
            "observations": unknown_rows(
                "remote_trt_cache_probe_unavailable", detail,
            ),
        })
        return result


def _remote_storage_reserve_bytes() -> int:
    """Safety reserve retained after a worst-case non-reflink suite clone."""

    raw = str(
        os.environ.get("ONNX_SPLITPOINT_REMOTE_STORAGE_RESERVE_BYTES", "2147483648")
        or "2147483648"
    ).strip()
    try:
        return max(512 * 1024 * 1024, int(raw))
    except (TypeError, ValueError):
        return 2 * 1024 * 1024 * 1024


def _suite_storage_footprint(suite_dir: Path) -> dict[str, int]:
    """Conservatively count source bytes and all file/directory inodes."""

    total_bytes = 0
    file_count = 0
    directory_count = 1
    for path in Path(suite_dir).rglob("*"):
        try:
            if path.is_symlink():
                file_count += 1
            elif path.is_dir():
                directory_count += 1
            elif path.is_file():
                total_bytes += int(path.stat().st_size)
                file_count += 1
        except OSError:
            # A concurrently changing suite is rejected later by packaging;
            # retain an inode in the early conservative estimate.
            file_count += 1
    return {
        "total_bytes": total_bytes,
        "file_count": file_count,
        "directory_count": directory_count,
    }


def _remote_run_capacity_requirement(suite_dir: Path) -> dict[str, int]:
    """Peak cold/warm admission bounds before the first remote mutation.

    The bound deliberately treats compression and reflinks as unavailable.  It
    additionally derives runtime output/log and TensorRT engine/build budgets
    from the selected benchmark plan instead of hiding them in one fixed
    reserve.  ``required_free_*`` remains the conservative cold contract for
    callers that do not distinguish cache state.
    """

    return _remote_run_capacity_requirement_for_args(suite_dir, args=None)


def _run_mentions_tensorrt(value: Any) -> bool:
    if isinstance(value, dict):
        return any(_run_mentions_tensorrt(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_run_mentions_tensorrt(item) for item in value)
    token = str(value or "").strip().lower()
    return bool(
        token in {"trt", "tensorrt", "ort_tensorrt"}
        or "tensorrt" in token
        or re.search(r"(?:^|[_-])trt(?:$|[_-])", token)
    )


def _remote_run_capacity_requirement_for_args(
    suite_dir: Path,
    *,
    args: "RemoteBenchmarkArgs | None",
) -> dict[str, int]:
    """Return conservative selected-plan cold/warm byte and inode budgets."""

    footprint = _suite_storage_footprint(suite_dir)
    raw = int(footprint["total_bytes"])
    entries = int(footprint["file_count"]) + int(footprint["directory_count"])
    plan = _read_json_dict(Path(suite_dir) / "benchmark_plan.json") or {}
    runs = plan.get("runs") if isinstance(plan.get("runs"), list) else []
    runs = [run for run in runs if isinstance(run, dict)]
    if args is not None:
        selected_ids = _extract_run_ids_from_add_args(getattr(args, "add_args", ""))
        if selected_ids:
            selected = set(selected_ids)
            runs = [run for run in runs if str(run.get("id") or run.get("name") or "") in selected]
    run_count = max(1, len(runs))
    case_count = max(1, len(_iter_suite_case_dirs(Path(suite_dir))))
    work_items = max(1, run_count * case_count)

    onnx_sizes: list[int] = []
    for path in Path(suite_dir).rglob("*.onnx"):
        try:
            if path.is_file() and not path.is_symlink():
                onnx_sizes.append(int(path.stat().st_size))
        except OSError:
            continue
    onnx_bytes = sum(onnx_sizes)
    largest_onnx = max(onnx_sizes, default=0)
    trt_runs = [run for run in runs if _run_mentions_tensorrt(run)]
    provider_forces_trt = bool(
        args is not None
        and _run_mentions_tensorrt(getattr(args, "provider", ""))
    )
    trt_profiles = {
        json.dumps(
            {
                "precision": run.get("precision") or run.get("native_trt_precision") or "fp16",
                "runtime": run.get("trt_runtime") or run.get("runtime") or "default",
                "provider": run.get("provider") or run.get("full_provider") or "tensorrt",
                "variants": run.get("variants") or [],
            },
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        for run in trt_runs
    }
    if provider_forces_trt:
        trt_profiles.add(json.dumps({
            "precision": "provider_default",
            "runtime": "provider_default",
            "provider": str(getattr(args, "provider", "tensorrt") or "tensorrt"),
            "variants": [],
        }, sort_keys=True, separators=(",", ":")))
    trt_profile_count = len(trt_profiles)
    trt_run_count = max(
        len(trt_runs),
        max(1, len(runs)) if provider_forces_trt else 0,
    )

    # Results include per-case scalar/quality artifacts and logs.  Eight MiB per
    # selected case/run plus a 256-MiB floor covers verbose diagnostics without
    # pretending they are part of the generic safety reserve.
    output_log_bytes = max(256 * 1024 * 1024, work_items * 8 * 1024 * 1024)
    output_log_inodes = max(2048, work_items * 128)

    if trt_profile_count:
        # Each unique TRT profile may retain engines comparable to several ONNX
        # copies.  A build can simultaneously hold source, rewritten graph,
        # temporary engine and final engine.  No compression/reflink gain is
        # assumed, and a 512-MiB floor covers small-graph builder auxiliaries.
        trt_engine_bytes = max(
            512 * 1024 * 1024,
            3 * max(onnx_bytes, largest_onnx) * trt_profile_count,
        )
        trt_build_bytes = max(512 * 1024 * 1024, 2 * largest_onnx)
        trt_inodes = max(1024, case_count * trt_profile_count * 32)
    else:
        trt_engine_bytes = 0
        trt_build_bytes = 0
        trt_inodes = 0

    transport_bytes = 2 * raw
    reserve_bytes = _remote_storage_reserve_bytes()
    warm_required = transport_bytes + output_log_bytes + reserve_bytes
    cold_required = warm_required + trt_engine_bytes + trt_build_bytes
    warm_inodes = 2 * entries + output_log_inodes + 8192
    cold_inodes = warm_inodes + trt_inodes
    return {
        **footprint,
        "plan_run_count": len(runs),
        "case_count": case_count,
        "work_item_count": work_items,
        "trt_run_count": trt_run_count,
        "trt_profile_count": trt_profile_count,
        "onnx_bytes": onnx_bytes,
        "transport_peak_bytes": transport_bytes,
        "runtime_output_log_bytes": output_log_bytes,
        "trt_engine_cache_bytes": trt_engine_bytes,
        "trt_build_scratch_bytes": trt_build_bytes,
        "safety_reserve_bytes": reserve_bytes,
        "warm_required_free_bytes": warm_required,
        "cold_required_free_bytes": cold_required,
        "warm_required_free_inodes": warm_inodes,
        "cold_required_free_inodes": cold_inodes,
        "required_free_bytes": cold_required,
        "required_free_inodes": cold_inodes,
    }


def _remote_trt_retention_limits() -> tuple[int, int]:
    """Return total-namespace and retained non-current byte limits."""

    try:
        max_namespaces = int(
            str(os.environ.get("ONNX_SPLITPOINT_REMOTE_TRT_CACHE_MAX_NAMESPACES", "6") or "6")
        )
    except (TypeError, ValueError):
        max_namespaces = 6
    try:
        max_bytes = int(
            str(os.environ.get("ONNX_SPLITPOINT_REMOTE_TRT_CACHE_MAX_BYTES", str(20 * 1024**3)) or str(20 * 1024**3))
        )
    except (TypeError, ValueError):
        max_bytes = 20 * 1024**3
    return max(1, max_namespaces), max(1024**3, max_bytes)


def _remote_trt_cache_retention_command(
    *,
    remote_base: str,
    current_key: str,
    max_namespaces: int,
    max_bytes: int,
    planned_current_growth_bytes: int = 0,
    legacy_suite_key: str = "",
    stable_engine_key: str = "",
    builder_abi_sha256: str = "",
    current_trtexec_sha256: str = "",
    native_trt_precision: str = "fp16",
    native_trt_workspace_mb: int = 4096,
    canonical_full_onnx_sha256: list[str] | tuple[str, ...] = (),
    source_shape_contracts: Mapping[str, Mapping[str, Any]] | None = None,
    preserve_existing_namespaces: bool = False,
) -> str:
    """Build a symlink-/marker-/receipt-/lock-safe TRT retention command.

    Only namespaces below the v2.75.16 managed root can ever be removed.  The
    legacy ``tensorrt`` tree and malformed/unowned managed children are merely
    inventoried.  A deletion candidate needs a valid owner marker and at least
    one self-hashed engine receipt bound to an engine inside its namespace.

    ``max_bytes`` bounds retained non-current namespaces.  The current
    namespace and ``planned_current_growth_bytes`` form the selected run's
    active working set, whose physical capacity is proved before this command.
    The total namespace count remains bounded by ``max_namespaces`` outside
    an active EvaluationWorkflow.  During a workflow all existing namespaces
    may be referenced by later quality, performance or energy stages, even
    when their compiler process has exited.  Preserve them and defer these
    retention targets; the preceding physical byte/inode admission still
    applies to every dispatch.
    """

    script = r'''import fcntl, hashlib, json, os, pathlib, re, shutil, stat, subprocess, sys, time
base = pathlib.Path(sys.argv[1])
current_key = sys.argv[2]
max_namespaces = max(1, int(sys.argv[3]))
max_bytes = max(1024 ** 3, int(sys.argv[4]))
planned_current_growth_bytes = max(0, int(sys.argv[5]))
legacy_suite_key = str(sys.argv[6] or "")
stable_engine_key = str(sys.argv[7] or "")
builder_abi_sha256 = str(sys.argv[8] or "").lower()
current_trtexec_sha256 = str(sys.argv[9] or "").lower()
native_trt_precision = str(sys.argv[10] or "fp16").lower()
native_trt_workspace_mb = max(0, int(sys.argv[11]))
preserve_existing_namespaces = sys.argv[14] == "1"
canonical_full_onnx_sha256 = {
    value.strip().lower() for value in str(sys.argv[12] or "").split(",")
    if re.fullmatch(r"[0-9a-fA-F]{64}", value.strip())
}
try:
    source_shape_contracts = json.loads(str(sys.argv[13] or "{}"))
    if not isinstance(source_shape_contracts, dict):
        source_shape_contracts = {}
except Exception:
    source_shape_contracts = {}
if re.fullmatch(r"[A-Za-z0-9_.-]{1,160}", current_key) is None:
    raise SystemExit("invalid managed TensorRT cache key")
for optional_key in (legacy_suite_key, stable_engine_key):
    if optional_key and re.fullmatch(r"[A-Za-z0-9_.-]{1,160}", optional_key) is None:
        raise SystemExit("invalid optional managed TensorRT cache key")
for digest in (builder_abi_sha256, current_trtexec_sha256):
    if digest and re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise SystemExit("invalid TensorRT ABI digest")

def lexists(path):
    return os.path.lexists(str(path))

def ensure_plain_dir(path):
    if lexists(path):
        info = os.lstat(path)
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise RuntimeError("unsafe cache directory: %s" % path)
    else:
        path.mkdir(mode=0o700)
    return path

def canonical_sha(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()

def file_sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def tree_inventory(root):
    total = 0
    files = 0
    unsafe_symlink = False
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        parent = pathlib.Path(dirpath)
        kept = []
        for name in dirnames:
            child = parent / name
            if child.is_symlink():
                unsafe_symlink = True
            else:
                kept.append(name)
        dirnames[:] = kept
        for name in filenames:
            child = parent / name
            try:
                info = os.lstat(child)
            except OSError:
                continue
            if stat.S_ISLNK(info.st_mode):
                unsafe_symlink = True
                continue
            if stat.S_ISREG(info.st_mode):
                total += int(info.st_size)
                files += 1
    return total, files, unsafe_symlink

def load_owner(namespace):
    marker = namespace / ".splitpoint_trt_cache_owner.json"
    try:
        info = os.lstat(marker)
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            return None
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("schema") != "onnx-splitpoint/managed-trt-cache-owner" or payload.get("schema_version") != 1:
        return None
    if payload.get("owner") != "onnx-splitpoint-tool" or str(payload.get("cache_key") or "") != namespace.name:
        return None
    return payload

def valid_receipt_present(namespace):
    root = namespace.resolve(strict=True)
    for dirpath, dirnames, filenames in os.walk(namespace, followlinks=False):
        parent = pathlib.Path(dirpath)
        kept = []
        for name in dirnames:
            child = parent / name
            if not child.is_symlink():
                kept.append(name)
        dirnames[:] = kept
        if "engine_build_receipt.json" not in filenames:
            continue
        receipt_path = parent / "engine_build_receipt.json"
        try:
            info = os.lstat(receipt_path)
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
                continue
            raw = json.loads(receipt_path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                continue
            declared = str(raw.get("receipt_sha256") or "").lower()
            unsigned = dict(raw)
            unsigned.pop("receipt_sha256", None)
            if len(declared) != 64 or canonical_sha(unsigned) != declared:
                continue
            if raw.get("schema") != "onnx-splitpoint/tensorrt-engine-build-receipt" or raw.get("schema_version") != 1:
                continue
            if raw.get("build_returncode") != 0 or raw.get("dry_run") is not False:
                continue
            engine = pathlib.Path(str(raw.get("engine") or "")).resolve(strict=True)
            if not engine.is_relative_to(root):
                continue
            einfo = os.lstat(engine)
            if stat.S_ISLNK(einfo.st_mode) or not stat.S_ISREG(einfo.st_mode):
                continue
            if file_sha(engine) != str(raw.get("engine_sha256") or "").lower():
                continue
            return True
        except Exception:
            continue
    return False

def acquire_inactive_lock(namespace):
    active_path = namespace / ".active.lock"
    active_flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(str(active_path), active_flags, 0o600)
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            os.close(descriptor)
            return None
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return descriptor
    except (BlockingIOError, OSError):
        try:
            os.close(descriptor)
        except (UnboundLocalError, OSError):
            pass
        return None

cache_parent = base / "_onnx_splitpoint_cache"
if not lexists(base) or os.path.islink(base) or not base.is_dir():
    raise RuntimeError("remote base is not a plain directory")
if lexists(cache_parent):
    ensure_plain_dir(cache_parent)
else:
    cache_parent.mkdir(mode=0o700)
managed = cache_parent / "tensorrt_managed_v27516"
if lexists(managed):
    ensure_plain_dir(managed)
else:
    managed.mkdir(mode=0o700)

lock_path = managed / ".retention.lock"
flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
lock_fd = os.open(str(lock_path), flags, 0o600)
candidate_lock_fds = []
try:
    if not stat.S_ISREG(os.fstat(lock_fd).st_mode):
        raise RuntimeError("retention lock is not a regular file")
    fcntl.flock(lock_fd, fcntl.LOCK_EX)
    now = time.time()
    current = managed / current_key
    current_created = not lexists(current)
    if not current_created:
        ensure_plain_dir(current)
        owner = load_owner(current)
        if owner is None:
            raise RuntimeError("current managed TensorRT namespace is unowned or malformed")
    else:
        current.mkdir(mode=0o700)
        owner = {
            "schema": "onnx-splitpoint/managed-trt-cache-owner",
            "schema_version": 1,
            "owner": "onnx-splitpoint-tool",
            "cache_key": current_key,
            "created_at_unix": now,
        }
    owner["last_used_at_unix"] = now
    owner["cache_layout_schema"] = (
        "onnx-splitpoint/trt-persistent-cache-layout"
    )
    owner["cache_layout_schema_version"] = 1
    if stable_engine_key:
        owner["trt_engine_cache_key"] = stable_engine_key
    if builder_abi_sha256:
        owner["trt_builder_abi_sha256"] = builder_abi_sha256
    marker = current / ".splitpoint_trt_cache_owner.json"
    if lexists(marker) and marker.is_symlink():
        raise RuntimeError("current owner marker is a symlink")
    marker_tmp = current / (".splitpoint_trt_cache_owner.json.tmp-%d-%d" % (os.getpid(), time.time_ns()))
    marker_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        marker_fd = os.open(str(marker_tmp), marker_flags, 0o600)
        with os.fdopen(marker_fd, "w", encoding="utf-8") as marker_handle:
            marker_handle.write(json.dumps(owner, sort_keys=True, separators=(",", ":")) + "\n")
            marker_handle.flush()
            os.fsync(marker_handle.fileno())
        os.replace(marker_tmp, marker)
    except BaseException:
        try:
            marker_tmp.unlink(missing_ok=True)
        except OSError:
            pass
        if current_created:
            try:
                if not any(current.iterdir()):
                    current.rmdir()
            except OSError:
                pass
        raise

    eviction_history_path = managed / ".eviction_history.jsonl"
    prior_eviction = None
    if lexists(eviction_history_path):
        try:
            history_info = os.lstat(eviction_history_path)
            if stat.S_ISLNK(history_info.st_mode) or not stat.S_ISREG(history_info.st_mode):
                raise RuntimeError("TensorRT eviction history is not a plain file")
            # The ledger is diagnostic, not a new trust root.  Bound reads on a
            # long-lived host and retain only the most recent matching event.
            with eviction_history_path.open("rb") as history_handle:
                history_handle.seek(max(0, int(history_info.st_size) - 1024 * 1024))
                history_text = history_handle.read().decode("utf-8", errors="replace")
            for history_line in history_text.splitlines():
                try:
                    history_row = json.loads(history_line)
                except Exception:
                    continue
                if (
                    isinstance(history_row, dict)
                    and history_row.get("namespace_key") == current_key
                ):
                    prior_eviction = history_row
        except OSError as exc:
            raise RuntimeError(
                "could not read TensorRT eviction history: %s" % type(exc).__name__
            )

    def append_eviction_history(event):
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(str(eviction_history_path), flags, 0o600)
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise RuntimeError("TensorRT eviction history is not a regular file")
            payload = (json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
            os.write(descriptor, payload)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    legacy = cache_parent / "tensorrt"
    legacy_inventory = {
        "exists": lexists(legacy), "deleted": False, "entries": [],
        "inventory_mode": (
            "shallow_existing_stable_namespace"
            if not current_created else "bootstrap_deep"
        ),
    }
    if lexists(legacy):
        try:
            info = os.lstat(legacy)
            legacy_inventory["symlink"] = stat.S_ISLNK(info.st_mode)
            if stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode):
                for child in sorted(legacy.iterdir(), key=lambda item: item.name)[:256]:
                    if current_created and child.is_dir() and not child.is_symlink():
                        size, files, unsafe = tree_inventory(child)
                    else:
                        child_info = os.lstat(child)
                        size = int(child_info.st_size) if stat.S_ISREG(child_info.st_mode) else 0
                        files = 1 if stat.S_ISREG(child_info.st_mode) else 0
                        unsafe = stat.S_ISLNK(child_info.st_mode)
                    legacy_inventory["entries"].append({"name": child.name, "bytes": size, "files": files, "unsafe_symlink": unsafe})
        except OSError as exc:
            legacy_inventory["error"] = type(exc).__name__

    migration = {
        # Compatibility is artifact-wise rather than namespace-wise.  An
        # existing stable root may be only partially populated, so every use
        # performs a cheap receipt-path inventory of compatible older roots.
        # Expensive artifact hashing/deserialization happens only for a leaf
        # which is actually absent from the current namespace.
        "requested": bool(
            canonical_full_onnx_sha256
            or (legacy_suite_key and legacy_suite_key != current_key)
        ),
        "trigger": (
            "namespace_bootstrap"
            if current_created else "partial_namespace_missing_artifact_scan"
        ),
        "target_preexisting": not current_created,
        "source_key": legacy_suite_key,
        "target_key": current_key,
        "inventoried_receipts": 0,
        "migrated_receipts": 0,
        "existing_receipts": 0,
        "partial_target_leaves": 0,
        "rejected_receipts": 0,
        "gpu_deserialization_failures": 0,
        "owner_key_changes": [],
        "candidate_failures": [],
        "status": "not_requested",
    }

    def receipt_precision_is_bound(argv, receipt_parent):
        precision = receipt_parent.name.lower()
        flags = [value for value in argv[1:] if value in {"--fp16", "--int8"}]
        if precision in {"fp16", "uint8_cast_fp16", "uint8_dequant_fp16", "float32_layout_fp16"}:
            return flags == ["--fp16"]
        if precision == "int8":
            return flags == ["--int8"]
        if precision in {"fp32", "float32"}:
            return flags == []
        return False

    def precision_family(value):
        normalized = str(value or "").strip().lower()
        if normalized in {
            "fp16", "uint8_cast_fp16", "uint8_dequant_fp16",
            "float32_layout_fp16",
        }:
            return "fp16"
        if normalized in {"fp32", "float32"}:
            return "fp32"
        if normalized == "int8":
            return "int8"
        return ""

    def canonical_source_sha_for_receipt(raw, receipt_path):
        """Resolve the original source identity behind supported bridges."""
        build_sha = str(raw.get("source_onnx_sha256") or "").lower()
        precision = receipt_path.parent.name.lower()
        if precision not in {
            "uint8_cast_fp16", "uint8_dequant_fp16",
            "float32_layout_fp16",
        }:
            return build_sha, "direct_source"
        if precision != "uint8_cast_fp16":
            # Dequant/layout identities are claim-bearing native-quality
            # contracts.  This generic migration path does not implement the
            # exact strict cross-link validator and must not adopt them based
            # on a generated ONNX receipt alone.
            return "", "special_bridge_binding_unverified"
        meta_path = receipt_path.parent / "uint8_cast_bridge_meta.json"
        try:
            meta_info = os.lstat(meta_path)
            if stat.S_ISLNK(meta_info.st_mode) or not stat.S_ISREG(meta_info.st_mode):
                return "", "generic_bridge_metadata_invalid"
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if (
                not isinstance(meta, dict)
                or meta.get("schema") != "onnx-splitpoint/uint8-cast-bridge"
                or meta.get("schema_version") != 1
            ):
                return "", "generic_bridge_metadata_invalid"
            source = pathlib.Path(str(meta.get("source") or ""))
            bridge = pathlib.Path(str(meta.get("bridge") or ""))
            build_source = pathlib.Path(str(raw.get("source_onnx") or ""))
            if (
                bridge.resolve(strict=True) != build_source.resolve(strict=True)
                or file_sha(bridge) != build_sha
            ):
                return "", "generic_bridge_artifact_mismatch"
            declared_bridge = str(meta.get("bridge_sha256") or "").lower()
            try:
                declared_bridge_size = int(meta.get("bridge_size_bytes"))
                actual_bridge_size = int(bridge.stat().st_size)
                declared_source_size = int(meta.get("source_size_bytes"))
            except (TypeError, ValueError, OSError):
                return "", "generic_bridge_metadata_incomplete"
            if (
                not re.fullmatch(r"[0-9a-f]{64}", declared_bridge)
                or declared_bridge != build_sha
                or declared_bridge_size != actual_bridge_size
            ):
                return "", "generic_bridge_artifact_mismatch"
            declared_source = str(meta.get("source_sha256") or "").lower()
            if source.is_file() and not source.is_symlink():
                original_sha = file_sha(source)
                if (
                    not re.fullmatch(r"[0-9a-f]{64}", declared_source)
                    or declared_source != original_sha
                    or declared_source_size != int(source.stat().st_size)
                ):
                    return "", "generic_bridge_source_mismatch"
            elif (
                re.fullmatch(r"[0-9a-f]{64}", declared_source)
                and declared_source_size >= 0
            ):
                original_sha = declared_source
            else:
                return "", "generic_bridge_source_identity_unprovable"
            if (
                str(meta.get("input_dtype") or "").upper() != "UINT8"
                or str(meta.get("cast_to") or "").upper() != "FLOAT"
                or not str(meta.get("input_name") or "")
                or not str(meta.get("cast_output") or "")
                or str(meta.get("precision_tag") or "").lower()
                   not in {"", "uint8_cast_fp16"}
                or int(meta.get("replaced_uses") or 0) <= 0
            ):
                return "", "generic_bridge_contract_mismatch"
            return original_sha, "generic_uint8_cast_bridge_verified"
        except Exception as exc:
            return "", "generic_bridge_validation_%s" % type(exc).__name__

    def validate_migration_receipt(receipt_path, source_namespace):
        try:
            info = os.lstat(receipt_path)
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
                return None, "receipt_not_plain_file"
            raw = json.loads(receipt_path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                return None, "receipt_not_object"
            declared = str(raw.get("receipt_sha256") or "").lower()
            unsigned = dict(raw)
            unsigned.pop("receipt_sha256", None)
            if len(declared) != 64 or canonical_sha(unsigned) != declared:
                return None, "receipt_digest_invalid"
            if (
                raw.get("schema") != "onnx-splitpoint/tensorrt-engine-build-receipt"
                or raw.get("schema_version") != 1
                or raw.get("build_returncode") != 0
                or raw.get("dry_run") is not False
            ):
                return None, "receipt_schema_or_status_invalid"
            source_root = source_namespace.resolve(strict=True)
            source_onnx = pathlib.Path(str(raw.get("source_onnx") or "")).resolve(strict=True)
            engine = pathlib.Path(str(raw.get("engine") or "")).resolve(strict=True)
            trtexec = pathlib.Path(str(raw.get("trtexec") or "")).resolve(strict=True)
            if not source_onnx.is_relative_to(source_root) or not engine.is_relative_to(source_root):
                return None, "receipt_artifact_outside_namespace"
            parent_root = receipt_path.parent.resolve(strict=True)
            if not source_onnx.is_relative_to(parent_root) or not engine.is_relative_to(parent_root):
                return None, "receipt_leaf_is_not_self_contained"
            for artifact in (source_onnx, engine, trtexec):
                artifact_info = os.lstat(artifact)
                if stat.S_ISLNK(artifact_info.st_mode) or not stat.S_ISREG(artifact_info.st_mode):
                    return None, "receipt_artifact_not_plain_file"
            expected_hashes = {
                "source_onnx_sha256": file_sha(source_onnx),
                "engine_sha256": file_sha(engine),
                "trtexec_sha256": file_sha(trtexec),
            }
            if any(str(raw.get(key) or "").lower() != value for key, value in expected_hashes.items()):
                return None, "receipt_artifact_hash_mismatch"
            if current_trtexec_sha256 and expected_hashes["trtexec_sha256"] != current_trtexec_sha256:
                return None, "receipt_builder_abi_mismatch"
            command = raw.get("command")
            argv = [str(value) for value in command] if isinstance(command, list) else []
            if not argv or pathlib.Path(argv[0]).resolve(strict=True) != trtexec:
                return None, "receipt_command_trtexec_mismatch"
            if [value for value in argv[1:] if value.startswith("--onnx=")] != [f"--onnx={source_onnx}"]:
                return None, "receipt_command_source_mismatch"
            if [value for value in argv[1:] if value.startswith("--saveEngine=")] != [f"--saveEngine={engine}"]:
                return None, "receipt_command_engine_mismatch"
            if not receipt_precision_is_bound(argv, receipt_path.parent):
                return None, "receipt_precision_mismatch"
            if precision_family(receipt_path.parent.name) != precision_family(native_trt_precision):
                return None, "receipt_runtime_precision_mismatch"
            workspace_flags = [
                value for value in argv[1:]
                if value.startswith("--workspace=")
                or value.startswith("--memPoolSize=workspace:")
            ]
            workspace_values = {
                f"--memPoolSize=workspace:{native_trt_workspace_mb}",
                f"--workspace={native_trt_workspace_mb}",
            }
            if native_trt_workspace_mb > 0:
                if len(workspace_flags) != 1 or workspace_flags[0] not in workspace_values:
                    return None, "receipt_workspace_mismatch"
            elif workspace_flags:
                return None, "receipt_workspace_mismatch"
            unknown_build_args = []
            for value in argv[1:]:
                if value.startswith((
                    "--onnx=", "--saveEngine=", "--timingCacheFile=",
                    "--shapes=", "--workspace=",
                    "--memPoolSize=workspace:",
                )) or value in {"--fp16", "--int8", "--verbose"}:
                    pass
                else:
                    unknown_build_args.append(value)
            if unknown_build_args:
                return None, "receipt_unknown_build_args"
            shape_args = [
                value.split("=", 1)[1]
                for value in argv[1:] if value.startswith("--shapes=")
            ]
            if len(shape_args) > 1:
                return None, "receipt_shape_mismatch"
            actual_shape = shape_args[0] if shape_args else ""
            canonical_source_sha, source_relation = (
                canonical_source_sha_for_receipt(raw, receipt_path)
            )
            if not canonical_source_sha:
                return None, source_relation
            shape_contract = source_shape_contracts.get(
                canonical_source_sha
            )
            owner_key = str((load_owner(source_namespace) or {}).get(
                "trt_engine_cache_key"
            ) or "")
            if source_namespace != current and owner_key not in {"", stable_engine_key}:
                # Old namespace metadata may include unrelated suite/selection
                # fields. Its individual artifact must still be requested now.
                if not current_trtexec_sha256:
                    return None, "receipt_builder_abi_unavailable"
                if not isinstance(shape_contract, dict):
                    return None, "source_not_in_current_request"
                if not bool(shape_contract.get("complete")):
                    return None, "receipt_shape_contract_unavailable"
                role = receipt_path.parent.parent.name.lower()
                if role not in {"full", "part1", "part2"}:
                    role = receipt_path.parent.parent.parent.name.lower()
                if role not in {"full", "part1", "part2"}:
                    return None, "receipt_role_path_mismatch"
                if engine.name != role + "_" + receipt_path.parent.name + ".engine":
                    return None, "canonical_engine_filename_mismatch"
                if role == "full" and canonical_source_sha not in canonical_full_onnx_sha256:
                    return None, "source_not_in_current_request"
            if isinstance(shape_contract, dict):
                allowed_shapes = {
                    str(value)
                    for value in list(shape_contract.get("allowed_shapes") or [])
                }
                if actual_shape not in allowed_shapes:
                    return None, "receipt_shape_mismatch"
            elif actual_shape:
                return None, "receipt_shape_contract_unavailable"
            return {
                "raw": raw,
                "source_onnx": source_onnx,
                "engine": engine,
                "trtexec": trtexec,
                "argv": argv,
                "canonical_source_sha256": canonical_source_sha,
                "source_relation": source_relation,
            }, "verified"
        except Exception as exc:
            return None, "receipt_validation_%s" % type(exc).__name__

    if migration["requested"]:
        migration["status"] = "source_missing_or_ineligible"
        migration["source_keys"] = []
        model_prefix = current_key.rsplit("-", 1)[0] + "-"

        def namespace_identity_compatible(source_namespace):
            source_owner = load_owner(source_namespace) if lexists(source_namespace) else None
            source_builder_abi = str((source_owner or {}).get("trt_builder_abi_sha256") or "").lower()
            return bool(
                source_owner is not None
                and source_namespace != current
                and bool(builder_abi_sha256)
                and source_builder_abi == builder_abi_sha256
            )

        def namespace_matches_canonical_full(source_namespace):
            if not canonical_full_onnx_sha256:
                return False
            for receipt_path in sorted(source_namespace.rglob("engine_build_receipt.json")):
                try:
                    raw = json.loads(receipt_path.read_text(encoding="utf-8"))
                    if (
                        isinstance(raw, dict)
                        and str(raw.get("source_onnx_sha256") or "").lower()
                        in canonical_full_onnx_sha256
                    ):
                        return True
                except Exception:
                    continue
            return False

        def migration_destination_relative(receipt_path, source_namespace):
            """Map old case-scoped leaves onto the v2.79.20 role layout."""
            fallback = receipt_path.parent.resolve(strict=True).relative_to(
                source_namespace.resolve(strict=True)
            )
            try:
                raw = json.loads(receipt_path.read_text(encoding="utf-8"))
                if not isinstance(raw, dict):
                    return fallback
                source_sha, _source_relation = canonical_source_sha_for_receipt(
                    raw, receipt_path,
                )
                if re.fullmatch(r"[0-9a-f]{64}", source_sha) is None:
                    return fallback
                precision = receipt_path.parent.name.lower()
                if re.fullmatch(r"[a-z0-9_.-]{1,80}", precision) is None:
                    return fallback
                role = receipt_path.parent.parent.name.lower()
                if role not in {"full", "part1", "part2"}:
                    # Canonical v2.79.20 leaves place the source digest between
                    # role and precision.
                    try:
                        role = receipt_path.parent.parent.parent.name.lower()
                    except IndexError:
                        role = ""
                if role == "full":
                    return pathlib.Path("full") / source_sha / precision
                if role not in {"part1", "part2"}:
                    return fallback
                case_id = next(
                    (
                        part for part in reversed(receipt_path.parts)
                        if re.fullmatch(r"b[0-9]+", part, flags=re.IGNORECASE)
                    ),
                    "",
                )
                if not case_id:
                    return fallback
                return (
                    pathlib.Path("splits") / case_id / role
                    / source_sha / precision
                )
            except Exception:
                return fallback

        source_namespaces = []
        # v2.79.19 already used the model-level namespace key, but its Full
        # leaves were still nested below the selected bNNN case.  Adopt those
        # leaves in-place before looking at other namespaces.  The global
        # retention lock already excludes another migration writer, and the
        # source and canonical destination are distinct directories.
        if any(current.rglob("engine_build_receipt.json")):
            source_namespaces.append(current)
        exact_source = managed / legacy_suite_key
        if (
            exact_source != current
            and namespace_identity_compatible(exact_source)
        ):
            source_namespaces.append(exact_source)
        for child in sorted(managed.iterdir(), key=lambda path: path.name):
            if (
                child in source_namespaces or child == current
                or not child.name.startswith(model_prefix)
                or not namespace_identity_compatible(child)
            ):
                continue
            if namespace_matches_canonical_full(child):
                source_namespaces.append(child)

        seen_destination_leaves = set()
        for source_namespace in source_namespaces:
            source_lock_fd = None
            if source_namespace != current:
                source_lock_fd = acquire_inactive_lock(source_namespace)
                if source_lock_fd is None:
                    migration["status"] = "source_active_or_lock_unsafe"
                    continue
            try:
                _source_size, _source_files, source_unsafe = tree_inventory(source_namespace)
                if source_unsafe:
                    migration["status"] = "source_contains_symlink"
                    continue
                migration["source_keys"].append(source_namespace.name)
                source_owner_key = str((load_owner(source_namespace) or {}).get(
                    "trt_engine_cache_key"
                ) or "")
                if source_namespace != current and source_owner_key not in {"", stable_engine_key}:
                    migration["owner_key_changes"].append({
                        "source_namespace": source_namespace.name,
                        "previous_engine_cache_key": source_owner_key,
                        "current_engine_cache_key": stable_engine_key,
                        "decision": "verify_individual_receipts",
                    })
                if migration["status"] not in {"verified_receipts_migrated", "source_active_or_lock_unsafe"}:
                    migration["status"] = "no_verified_receipts"
                receipt_paths = sorted(source_namespace.rglob("engine_build_receipt.json"))
                for receipt_path in receipt_paths:
                    migration["inventoried_receipts"] += 1
                    relative_leaf = migration_destination_relative(
                        receipt_path, source_namespace,
                    )
                    destination_identity = relative_leaf.as_posix()
                    if destination_identity in seen_destination_leaves:
                        continue
                    destination_leaf = current / relative_leaf
                    if lexists(destination_leaf):
                        seen_destination_leaves.add(destination_identity)
                        receipt_candidate = destination_leaf / "engine_build_receipt.json"
                        if (
                            destination_leaf.is_dir()
                            and not destination_leaf.is_symlink()
                            and receipt_candidate.is_file()
                            and not receipt_candidate.is_symlink()
                        ):
                            migration["existing_receipts"] += 1
                            if migration["status"] == "no_verified_receipts":
                                migration["status"] = (
                                    "compatible_receipts_already_present"
                                )
                        else:
                            migration["partial_target_leaves"] += 1
                        continue
                    verified, reason = validate_migration_receipt(receipt_path, source_namespace)
                    if verified is None:
                        migration["rejected_receipts"] += 1
                        if len(migration["candidate_failures"]) < 64:
                            migration["candidate_failures"].append({
                                "receipt_path": str(receipt_path), "reason": reason,
                            })
                        continue
                    source_leaf = receipt_path.parent.resolve(strict=True)
                    # A legacy receipt has no GPU field.  Deserialize its exact
                    # engine with the currently attested trtexec before copying;
                    # this is the compatibility gate for the present GPU/runtime.
                    try:
                        probe = subprocess.run(
                            [str(verified["trtexec"]), f"--loadEngine={verified['engine']}", "--skipInference"],
                            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            timeout=180,
                        )
                    except Exception:
                        probe = None
                    if probe is None or int(probe.returncode) != 0:
                        migration["gpu_deserialization_failures"] += 1
                        migration["rejected_receipts"] += 1
                        continue
                    destination_leaf.parent.mkdir(parents=True, exist_ok=True)
                    temporary_leaf = destination_leaf.parent / (
                        ".trt-migrate-%s-%d-%d" % (destination_leaf.name, os.getpid(), time.time_ns())
                    )
                    try:
                        def link_or_copy(source_path, destination_path):
                            # The receipt is rewritten with rebased paths below;
                            # hard-linking it would mutate the legacy evidence
                            # when the temporary copy is replaced in place.
                            if pathlib.Path(source_path).name in {
                                "engine_build_receipt.json", "uint8_cast_bridge_meta.json",
                            }:
                                return shutil.copy2(source_path, destination_path)
                            try:
                                os.link(source_path, destination_path)
                                return destination_path
                            except OSError:
                                return shutil.copy2(source_path, destination_path)

                        shutil.copytree(
                            source_leaf, temporary_leaf, symlinks=False,
                            copy_function=link_or_copy,
                        )
                        copied_receipt_path = temporary_leaf / "engine_build_receipt.json"
                        copied = dict(verified["raw"])

                        def rebase_path(raw_value):
                            old_path = pathlib.Path(str(raw_value)).resolve(strict=False)
                            relative = old_path.relative_to(source_leaf)
                            return str((destination_leaf / relative).resolve(strict=False))

                        copied["source_onnx"] = rebase_path(copied["source_onnx"])
                        copied["engine"] = rebase_path(copied["engine"])
                        rebased_command = []
                        source_prefix = str(source_leaf) + os.sep
                        current_prefix = str(destination_leaf.resolve(strict=False)) + os.sep
                        for value in [str(item) for item in copied.get("command") or []]:
                            rebased_command.append(value.replace(source_prefix, current_prefix))
                        copied["command"] = rebased_command
                        copied.pop("receipt_sha256", None)
                        copied["receipt_sha256"] = canonical_sha(copied)
                        copied_receipt_path.write_text(
                            json.dumps(copied, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8",
                        )
                        copied_bridge_meta_path = (
                            temporary_leaf / "uint8_cast_bridge_meta.json"
                        )
                        if copied_bridge_meta_path.is_file():
                            bridge_meta = json.loads(
                                copied_bridge_meta_path.read_text(
                                    encoding="utf-8"
                                )
                            )
                            if not isinstance(bridge_meta, dict):
                                raise RuntimeError(
                                    "migrated uint8 bridge metadata invalid"
                                )
                            for field in ("source", "bridge"):
                                old_value = pathlib.Path(
                                    str(bridge_meta.get(field) or "")
                                ).resolve(strict=False)
                                try:
                                    relative_value = old_value.relative_to(
                                        source_leaf
                                    )
                                except ValueError:
                                    continue
                                bridge_meta[field] = str(
                                    (destination_leaf / relative_value).resolve(
                                        strict=False
                                    )
                                )
                            copied_bridge_meta_path.write_text(
                                json.dumps(
                                    bridge_meta, indent=2, sort_keys=True,
                                ) + "\n",
                                encoding="utf-8",
                            )
                        copied_source = temporary_leaf / verified["source_onnx"].relative_to(source_leaf)
                        copied_engine = temporary_leaf / verified["engine"].relative_to(source_leaf)
                        if (
                            file_sha(copied_source) != str(copied.get("source_onnx_sha256") or "").lower()
                            or file_sha(copied_engine) != str(copied.get("engine_sha256") or "").lower()
                        ):
                            raise RuntimeError("migrated artifact hash mismatch")
                        os.replace(temporary_leaf, destination_leaf)
                        seen_destination_leaves.add(destination_identity)
                        migration["migrated_receipts"] += 1
                        migration["status"] = "verified_receipts_migrated"
                    except Exception:
                        migration["rejected_receipts"] += 1
                        try:
                            if lexists(temporary_leaf):
                                shutil.rmtree(temporary_leaf)
                        except OSError:
                            pass
            finally:
                if source_lock_fd is not None:
                    try:
                        fcntl.flock(source_lock_fd, fcntl.LOCK_UN)
                    finally:
                        os.close(source_lock_fd)

    # Process locks cover running commands, not references retained by the
    # enclosing EvaluationWorkflow.  A model prepared now can be measured much
    # later after six other models have used this cache.  Never let LRU remove
    # its bound Full source/engine or native quality Part1 files in between.
    # Physical free bytes/inodes were already admitted before this command;
    # deferring retention targets does not relax that storage admission.
    if preserve_existing_namespaces:
        current_bytes, _current_files, current_unsafe = tree_inventory(current)
        preserved = []
        for child in sorted(managed.iterdir(), key=lambda item: item.name):
            if child.name in {".retention.lock", ".eviction_history.jsonl", current_key}:
                continue
            info = os.lstat(child)
            if stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode):
                child_bytes, _child_files, _child_unsafe = tree_inventory(child)
            else:
                child_bytes = int(info.st_size) if stat.S_ISREG(info.st_mode) else 0
            preserved.append({
                "name": child.name,
                "reason": "active_workflow_may_reference_namespace",
                "bytes": child_bytes,
            })
        retained_bytes = sum(row["bytes"] for row in preserved)
        active_bytes = current_bytes + planned_current_growth_bytes
        result = {
            "schema": "onnx-splitpoint/remote-trt-cache-retention",
            "schema_version": 1,
            "managed_root": str(managed),
            "current_key": current_key,
            "current_root": str(current),
            "admission_ok": not current_unsafe,
            "reason": (
                "current_managed_trt_cache_contains_symlink"
                if current_unsafe else "retention_deferred_active_workflow"
            ),
            "retention_deferred": True,
            "retention_limits_exceeded": (
                len(preserved) + 1 > max_namespaces or retained_bytes > max_bytes
            ),
            "max_namespaces": max_namespaces,
            "max_bytes": max_bytes,
            "managed_count": len(preserved) + 1,
            "managed_bytes": retained_bytes + current_bytes,
            "projected_managed_bytes": retained_bytes + active_bytes,
            "planned_current_growth_bytes": planned_current_growth_bytes,
            "retained_cache_budget_bytes": max_bytes,
            "retained_noncurrent_bytes": retained_bytes,
            "current_namespace_bytes": current_bytes,
            "active_working_set_bytes": active_bytes,
            "effective_admission_max_bytes": max_bytes + active_bytes,
            "selected_plan_reserve_applied": True,
            "eligible_remaining": 0,
            "eligible_remaining_bytes": 0,
            "removed": [],
            "protected": preserved,
            "legacy_inventory": legacy_inventory,
            "legacy_receipt_migration": migration,
            "prior_eviction": prior_eviction,
            "eviction_history_path": str(eviction_history_path),
        }
        print("SPLITPOINT_TRT_RETENTION_JSON=" + json.dumps(result, sort_keys=True, separators=(",", ":")))
        raise SystemExit(75 if current_unsafe else 0)

    # Warm namespaces need exact byte/count/symlink accounting, but receipt and
    # engine hashes are only relevant when a deletion is actually necessary.
    # ``max_bytes`` is the retained, non-current cache budget.  The selected
    # run's current namespace plus its conservative planned growth is an active
    # working-set reserve; the earlier physical-capacity preflight has already
    # proved that reserve, build scratch, transport, outputs and the safety
    # margin fit on the remote filesystem.  Charging the same reserve against
    # the retained-cache budget made any selected plan larger than 20 GiB
    # mathematically inadmissible even on an otherwise empty host.
    #
    # Do one metadata-only inventory first.  If the retained non-current tree
    # and total namespace count already fit, return without scanning receipts,
    # hashing multi-GiB engines, or creating active locks in old namespaces.
    if not current_created:
        retained_unclassified = []
        light_protected = []
        light_other_bytes = 0
        light_other_count = 0
        for child in sorted(managed.iterdir(), key=lambda item: item.name):
            if child.name in {".retention.lock", ".eviction_history.jsonl", current_key}:
                continue
            try:
                info = os.lstat(child)
            except OSError:
                continue
            light_other_count += 1
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                child_bytes = int(info.st_size) if stat.S_ISREG(info.st_mode) else 0
                light_other_bytes += child_bytes
                light_protected.append({
                    "name": child.name,
                    "reason": "symlink_or_non_directory",
                    "bytes": child_bytes,
                })
                continue
            child_bytes, child_files, child_unsafe = tree_inventory(child)
            light_other_bytes += child_bytes
            child_owner = load_owner(child)
            if child_owner is None:
                light_protected.append({
                    "name": child.name,
                    "reason": "unowned_or_bad_marker",
                    "bytes": child_bytes,
                })
            elif child_unsafe:
                light_protected.append({
                    "name": child.name,
                    "reason": "symlink_in_tree",
                    "bytes": child_bytes,
                })
            else:
                retained_unclassified.append({
                    "name": child.name,
                    "bytes": child_bytes,
                    "files": child_files,
                    "reason": "within_limits_no_deletion_validation",
                })

        light_current_bytes, _light_current_files, light_current_unsafe = tree_inventory(current)
        light_managed_count = light_other_count + 1
        light_managed_bytes = light_other_bytes + light_current_bytes
        light_active_working_set_bytes = (
            light_current_bytes + planned_current_growth_bytes
        )
        light_projected_bytes = (
            light_other_bytes + light_active_working_set_bytes
        )
        light_effective_admission_max_bytes = (
            max_bytes + light_active_working_set_bytes
        )
        if (
            not light_current_unsafe
            and light_managed_count <= max_namespaces
            and light_other_bytes <= max_bytes
        ):
            result = {
                "schema": "onnx-splitpoint/remote-trt-cache-retention",
                "schema_version": 1,
                "managed_root": str(managed),
                "current_key": current_key,
                "current_root": str(current),
                "max_namespaces": max_namespaces,
                "max_bytes": max_bytes,
                "admission_ok": True,
                "managed_count": light_managed_count,
                "managed_bytes": light_managed_bytes,
                "projected_managed_bytes": light_projected_bytes,
                "planned_current_growth_bytes": planned_current_growth_bytes,
                "retained_cache_budget_bytes": max_bytes,
                "retained_noncurrent_bytes": light_other_bytes,
                "current_namespace_bytes": light_current_bytes,
                "active_working_set_bytes": light_active_working_set_bytes,
                "effective_admission_max_bytes": light_effective_admission_max_bytes,
                "selected_plan_reserve_applied": True,
                "eligible_remaining": 0,
                "eligible_remaining_bytes": 0,
                "removed": [],
                "protected": light_protected,
                "retained_unclassified": retained_unclassified,
                "retention_fast_path": "existing_stable_namespace_within_limits",
                "receipt_hash_validation_performed": False,
                "legacy_inventory": legacy_inventory,
                "legacy_receipt_migration": migration,
                "prior_eviction": prior_eviction,
            }
            print("SPLITPOINT_TRT_RETENTION_JSON=" + json.dumps(result, sort_keys=True, separators=(",", ":")))
            raise SystemExit(0)

    eligible = []
    protected = []
    for child in sorted(managed.iterdir(), key=lambda item: item.name):
        if child.name in {".retention.lock", ".eviction_history.jsonl", current_key}:
            continue
        try:
            info = os.lstat(child)
        except OSError:
            continue
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            protected.append({
                "name": child.name,
                "reason": "symlink_or_non_directory",
                "bytes": int(info.st_size) if stat.S_ISREG(info.st_mode) else 0,
            })
            continue
        child_owner = load_owner(child)
        size, files, unsafe = tree_inventory(child)
        if child.name in migration.get("source_keys", []):
            protected.append({
                "name": child.name, "reason": "legacy_migration_source", "bytes": size,
            })
            continue
        if child_owner is None:
            protected.append({"name": child.name, "reason": "unowned_or_bad_marker", "bytes": size})
            continue
        if unsafe:
            protected.append({"name": child.name, "reason": "symlink_in_tree", "bytes": size})
            continue
        if not valid_receipt_present(child):
            protected.append({"name": child.name, "reason": "no_valid_engine_receipt", "bytes": size})
            continue
        eligible.append({
            "path": child,
            "name": child.name,
            "bytes": size,
            "files": files,
            "last_used": float(child_owner.get("last_used_at_unix") or child_owner.get("created_at_unix") or 0.0),
            "builder_abi_match": bool(
                builder_abi_sha256
                and str(child_owner.get("trt_builder_abi_sha256") or "").lower()
                == builder_abi_sha256
            ),
            "model_identity_match": bool(
                stable_engine_key
                and child.name.rsplit("-", 1)[0]
                == stable_engine_key.rsplit("-", 1)[0]
            ),
        })

    # Classify every potentially deletable namespace and hold all of its
    # inactive locks before planning a single deletion.  Otherwise an active
    # newer candidate discovered late could make admission impossible only
    # after an unrelated older cache had already been destroyed.
    locked_eligible = []
    for candidate in eligible:
        if load_owner(candidate["path"]) is None or not valid_receipt_present(candidate["path"]):
            protected.append({"name": candidate["name"], "reason": "revalidation_failed", "bytes": candidate["bytes"]})
            continue
        _size, _files, unsafe = tree_inventory(candidate["path"])
        if unsafe:
            protected.append({"name": candidate["name"], "reason": "revalidation_symlink", "bytes": candidate["bytes"]})
            continue
        active_fd = acquire_inactive_lock(candidate["path"])
        if active_fd is None:
            protected.append({"name": candidate["name"], "reason": "active_or_unsafe_lock", "bytes": candidate["bytes"]})
            continue
        candidate_lock_fds.append(active_fd)
        locked_eligible.append(candidate)

    # Evict incompatible/other-model caches before a receipt-valid namespace
    # for the present model and ABI.  LRU remains the tie-breaker.  Current and
    # remotely active namespaces are hard-protected above.
    eligible = sorted(
        locked_eligible,
        key=lambda row: (
            int(bool(row.get("builder_abi_match"))),
            int(bool(row.get("model_identity_match"))),
            row["last_used"],
            row["name"],
        ),
    )
    eligible_bytes = sum(int(row["bytes"]) for row in eligible)
    protected_bytes = sum(int(row.get("bytes") or 0) for row in protected)
    current_bytes, _current_files, current_unsafe = tree_inventory(current)
    removed = []

    def immutable_floor_exceeds_limits():
        return (
            len(protected) + 1 > max_namespaces
            or protected_bytes > max_bytes
        )

    planned = []
    remaining = list(eligible)
    remaining_bytes = eligible_bytes
    if not current_unsafe and not immutable_floor_exceeds_limits():
        while remaining and (
            len(remaining) + len(protected) + 1 > max_namespaces
            or remaining_bytes + protected_bytes > max_bytes
        ):
            victim = remaining.pop(0)
            remaining_bytes -= int(victim["bytes"])
            planned.append(victim)

    plan_is_admissible = bool(
        not current_unsafe
        and len(remaining) + len(protected) + 1 <= max_namespaces
        and remaining_bytes + protected_bytes <= max_bytes
    )
    if planned and plan_is_admissible:
        invalid_planned = []
        for victim in planned:
            if load_owner(victim["path"]) is None or not valid_receipt_present(victim["path"]):
                invalid_planned.append((victim, "deletion_boundary_revalidation_failed"))
                continue
            _size, _files, unsafe = tree_inventory(victim["path"])
            if unsafe:
                invalid_planned.append((victim, "deletion_boundary_symlink"))
        if invalid_planned:
            # Abort the complete plan before deleting anything.  The acquired
            # locks remain held until the outer finally block.
            invalid_names = {row[0]["name"] for row in invalid_planned}
            eligible = [row for row in eligible if row["name"] not in invalid_names]
            eligible_bytes = sum(int(row["bytes"]) for row in eligible)
            for victim, reason in invalid_planned:
                protected.append({"name": victim["name"], "reason": reason, "bytes": victim["bytes"]})
                protected_bytes += int(victim["bytes"])
        else:
            for victim in planned:
                event = {
                    "schema": "onnx-splitpoint/trt-cache-eviction-event",
                    "schema_version": 1,
                    "evicted_at_unix": time.time(),
                    "namespace_key": victim["name"],
                    "bytes": victim["bytes"],
                    "last_used_at_unix": victim["last_used"],
                    "reason": "retained_cache_limit_lru",
                    "builder_abi_match": bool(victim.get("builder_abi_match")),
                    "model_identity_match": bool(victim.get("model_identity_match")),
                }
                append_eviction_history(event)
                shutil.rmtree(victim["path"])
                removed.append({
                    "name": victim["name"],
                    "bytes": victim["bytes"],
                    "reason": event["reason"],
                    "eviction_event": event,
                })
            eligible = remaining
            eligible_bytes = remaining_bytes

    managed_count = len(eligible) + len(protected) + 1
    managed_bytes = eligible_bytes + protected_bytes + current_bytes
    retained_noncurrent_bytes = eligible_bytes + protected_bytes
    active_working_set_bytes = current_bytes + planned_current_growth_bytes
    projected_managed_bytes = retained_noncurrent_bytes + active_working_set_bytes
    effective_admission_max_bytes = max_bytes + active_working_set_bytes
    if (
        current_unsafe
        or managed_count > max_namespaces
        or retained_noncurrent_bytes > max_bytes
    ):
        failure = {
            "schema": "onnx-splitpoint/remote-trt-cache-retention",
            "schema_version": 1,
            "admission_ok": False,
            "reason": (
                "current_managed_trt_cache_contains_symlink"
                if current_unsafe
                else (
                    "managed_trt_cache_retained_noncurrent_exceeds_limit"
                    if retained_noncurrent_bytes > max_bytes and not protected
                    else "managed_trt_cache_limits_blocked_by_preserved_migration_sources"
                    if any(row.get("reason") == "legacy_migration_source" for row in protected)
                    else "managed_trt_cache_limits_blocked_by_active_or_unowned_namespaces"
                )
            ),
            "managed_count": managed_count,
            "managed_bytes": managed_bytes,
            "projected_managed_bytes": projected_managed_bytes,
            "planned_current_growth_bytes": planned_current_growth_bytes,
            "retained_cache_budget_bytes": max_bytes,
            "retained_noncurrent_bytes": retained_noncurrent_bytes,
            "current_namespace_bytes": current_bytes,
            "active_working_set_bytes": active_working_set_bytes,
            "effective_admission_max_bytes": effective_admission_max_bytes,
            "selected_plan_reserve_applied": True,
            "max_namespaces": max_namespaces,
            "max_bytes": max_bytes,
            "protected": protected,
            "removed": removed,
            "legacy_inventory": legacy_inventory,
            "legacy_receipt_migration": migration,
            "prior_eviction": prior_eviction,
            "eviction_history_path": str(eviction_history_path),
        }
        # Roll back only the namespace created by this invocation and only while
        # it is still the empty owner-marker shell.  Existing/active/uncertain
        # data is never removed automatically.
        if current_created and not current_unsafe:
            allowed_names = {".splitpoint_trt_cache_owner.json"}
            if {item.name for item in current.iterdir()} <= allowed_names:
                marker.unlink(missing_ok=True)
                current.rmdir()
                failure["new_empty_current_rolled_back"] = True
        print("SPLITPOINT_TRT_RETENTION_JSON=" + json.dumps(failure, sort_keys=True, separators=(",", ":")))
        raise SystemExit(75)

    result = {
        "schema": "onnx-splitpoint/remote-trt-cache-retention",
        "schema_version": 1,
        "managed_root": str(managed),
        "current_key": current_key,
        "current_root": str(current),
        "max_namespaces": max_namespaces,
        "max_bytes": max_bytes,
        "admission_ok": True,
        "managed_count": managed_count,
        "managed_bytes": managed_bytes,
        "projected_managed_bytes": projected_managed_bytes,
        "planned_current_growth_bytes": planned_current_growth_bytes,
        "retained_cache_budget_bytes": max_bytes,
        "retained_noncurrent_bytes": retained_noncurrent_bytes,
        "current_namespace_bytes": current_bytes,
        "active_working_set_bytes": active_working_set_bytes,
        "effective_admission_max_bytes": effective_admission_max_bytes,
        "selected_plan_reserve_applied": True,
        "eligible_remaining": len(eligible),
        "eligible_remaining_bytes": eligible_bytes,
        "removed": removed,
        "protected": protected,
        "legacy_inventory": legacy_inventory,
        "legacy_receipt_migration": migration,
        "prior_eviction": prior_eviction,
        "eviction_history_path": str(eviction_history_path),
    }
    print("SPLITPOINT_TRT_RETENTION_JSON=" + json.dumps(result, sort_keys=True, separators=(",", ":")))
finally:
    try:
        for candidate_fd in candidate_lock_fds:
            try:
                fcntl.flock(candidate_fd, fcntl.LOCK_UN)
            finally:
                os.close(candidate_fd)
    finally:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        finally:
            os.close(lock_fd)
'''
    return (
        f"python3 - {shlex.quote(str(remote_base))} {shlex.quote(str(current_key))} "
        f"{int(max_namespaces)} {int(max_bytes)} {max(0, int(planned_current_growth_bytes))} "
        f"{shlex.quote(str(legacy_suite_key or ''))} {shlex.quote(str(stable_engine_key or ''))} "
        f"{shlex.quote(str(builder_abi_sha256 or ''))} {shlex.quote(str(current_trtexec_sha256 or ''))} "
        f"{shlex.quote(str(native_trt_precision or 'fp16'))} {max(0, int(native_trt_workspace_mb))} "
        f"{shlex.quote(','.join(sorted(str(value).lower() for value in canonical_full_onnx_sha256)))} "
        f"{shlex.quote(json.dumps(dict(source_shape_contracts or {}), sort_keys=True, separators=(',', ':')))} "
        f"{int(bool(preserve_existing_namespaces))} "
        "<<'SPLITPOINT_TRT_RETENTION_PY'\n"
        + script
        + "\nSPLITPOINT_TRT_RETENTION_PY"
    )


def _remote_storage_preflight(
    transport: SSHTransport,
    remote_path: str,
    *,
    required_free_bytes: int,
    required_free_inodes: int,
    stage: str,
) -> dict[str, Any]:
    """Read-only remote mount/capacity check.

    The probe uses ``statvfs`` and permission metadata only.  It deliberately
    does not create the old ``.splitpoint_write_test`` directory, so even a
    rejected read-only target is left byte-for-byte unchanged.
    """

    probe_script = r'''import json, os, stat, sys
requested = os.path.expanduser(sys.argv[1])
probe = requested
while not os.path.exists(probe):
    parent = os.path.dirname(probe) or os.sep
    if parent == probe:
        break
    probe = parent
if os.path.isfile(probe):
    probe = os.path.dirname(probe)
st = os.stat(probe)
fs = os.statvfs(probe)
uid = os.geteuid()
groups = set(os.getgroups()) | {os.getegid()}
if uid == st.st_uid:
    mode_ok = bool(st.st_mode & stat.S_IWUSR) and bool(st.st_mode & stat.S_IXUSR)
elif st.st_gid in groups:
    mode_ok = bool(st.st_mode & stat.S_IWGRP) and bool(st.st_mode & stat.S_IXGRP)
else:
    mode_ok = bool(st.st_mode & stat.S_IWOTH) and bool(st.st_mode & stat.S_IXOTH)
readonly = bool(getattr(os, "ST_RDONLY", 1) and fs.f_flag & getattr(os, "ST_RDONLY", 1))
payload = {
    "requested_path": requested,
    "probe_path": os.path.realpath(probe),
    "read_only_mount": readonly,
    "permission_bits_allow": mode_ok,
    "os_access_allow": os.access(probe, os.W_OK | os.X_OK),
    "free_bytes": int(fs.f_bavail) * int(fs.f_frsize or fs.f_bsize),
    "free_inodes": int(fs.f_favail),
}
print("SPLITPOINT_STORAGE_JSON=" + json.dumps(payload, sort_keys=True, separators=(",", ":")))
'''
    cmd = (
        f"python3 - {shlex.quote(str(remote_path))} "
        "<<'SPLITPOINT_STORAGE_PY'\n"
        + probe_script
        + "\nSPLITPOINT_STORAGE_PY"
    )
    rc, output = transport.run_read_only(cmd, timeout=30)
    if rc == 130:
        detail = str(output or "").strip()
        raise BundleCancelled(
            detail or "cancelled during read-only remote storage admission"
        )
    if rc != 0:
        raise RuntimeError(
            "remote_storage_preflight_failed: read-only statvfs probe could not "
            f"run at stage={stage}, rc={rc}: {str(output or '')[-2000:]}"
        )
    marker = "SPLITPOINT_STORAGE_JSON="
    encoded = ""
    for line in str(output or "").splitlines():
        if line.startswith(marker):
            encoded = line[len(marker):]
    try:
        payload = json.loads(encoded)
    except (TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "remote_storage_preflight_failed: remote probe returned no valid "
            f"contract at stage={stage}: {str(output or '')[-2000:]}"
        ) from exc
    if not isinstance(payload, dict):
        raise RuntimeError(
            f"remote_storage_preflight_failed: invalid payload at stage={stage}"
        )

    required_bytes = max(0, int(required_free_bytes))
    required_inodes = max(0, int(required_free_inodes))
    problems: list[str] = []
    if payload.get("read_only_mount") is True:
        problems.append("read_only_mount")
    if payload.get("permission_bits_allow") is not True:
        problems.append("directory_permissions_read_only")
    if payload.get("os_access_allow") is not True:
        problems.append("directory_access_denied")
    free_bytes = int(payload.get("free_bytes") or 0)
    free_inodes = int(payload.get("free_inodes") or 0)
    if free_bytes < required_bytes:
        problems.append(f"free_bytes={free_bytes}<required={required_bytes}")
    if free_inodes < required_inodes:
        problems.append(f"free_inodes={free_inodes}<required={required_inodes}")
    payload.update(
        {
            "stage": stage,
            "required_free_bytes": required_bytes,
            "required_free_inodes": required_inodes,
            "ok": not problems,
            "problems": problems,
        }
    )
    if problems:
        raise RuntimeError(
            "remote_storage_preflight_failed: refusing remote mutation; "
            f"stage={stage}, target={remote_path}, reason={';'.join(problems)}. "
            "Choose a writable remote base with sufficient free blocks/inodes; "
            "no measurement or cache directory was removed."
        )
    return payload


def _run_likely_uses_hailo(run: dict[str, Any]) -> bool:
    """Return True if a plan run likely exercises Hailo at runtime."""

    try:
        run_type = str(run.get("type") or run.get("kind") or "").strip().lower()
    except Exception:
        run_type = ""
    if run_type == "hailo":
        return True

    for key in ("provider", "full_provider", "stage1_provider", "stage2_provider"):
        try:
            tok = str(run.get(key) or "").strip().lower()
        except Exception:
            tok = ""
        if tok.startswith("hailo"):
            return True

    for key in ("stage1", "stage2"):
        st = run.get(key)
        if isinstance(st, dict):
            try:
                tok = str(st.get("type") or st.get("provider") or st.get("hw_arch") or "").strip().lower()
            except Exception:
                tok = ""
            if tok.startswith("hailo"):
                return True
        elif isinstance(st, str) and st.strip().lower().startswith("hailo"):
            return True

    return False


def estimate_remote_timeout_hint(suite_dir: Path, args: "RemoteBenchmarkArgs") -> dict[str, Any]:
    """Estimate a sane outer timeout for a remote benchmark run.

    This is intentionally heuristic. The goal is not exact runtime prediction,
    but to catch obviously too-small outer timeouts before a long remote run is
    aborted after hours of useful work.
    """

    case_count = len(_iter_suite_case_dirs(suite_dir))
    plan_runs = _load_plan_runs_from_suite(suite_dir)
    if not plan_runs:
        # Fallback: approximate a single provider run.
        plan_runs = [{"id": f"ort_{getattr(args, 'provider', 'auto')}", "type": "onnxruntime", "provider": getattr(args, 'provider', 'auto')}]

    try:
        warmup = max(0, int(getattr(args, "warmup", 0) or 0))
    except Exception:
        warmup = 0
    try:
        repeats = max(1, int(getattr(args, "repeats", 1) or 1))
    except Exception:
        repeats = 1
    try:
        iters = max(1, int(getattr(args, "iters", 1) or 1))
    except Exception:
        iters = 1

    effective_runs = max(1, repeats * iters)
    invocations_per_case_run = 4 * (warmup + effective_runs)

    per_run_lower_bounds_s: list[float] = []
    for run in plan_runs:
        uses_hailo = _run_likely_uses_hailo(run)
        # Lower-bound wall-clock heuristic:
        # - 4 timed phases per case (full / part1 / part2 / composed)
        # - even "fast" cases incur session init, validation and file IO
        phase_floor_s = 0.08 if uses_hailo else 0.05
        per_case_overhead_s = 4.0 if uses_hailo else 1.0
        per_case_run_s = float(invocations_per_case_run) * phase_floor_s + per_case_overhead_s
        per_run_lower_bounds_s.append(float(case_count) * per_case_run_s)

    lower_bound_s = float(sum(per_run_lower_bounds_s))
    recommended_timeout_s = int(max(600.0, min(172800.0, lower_bound_s * 1.5 + 600.0)))

    return {
        "heuristic": "lower_bound_cases_x_runs_x_4phases",
        "case_count": int(case_count),
        "planned_run_count": int(len(plan_runs)),
        "warmup": int(warmup),
        "repeats": int(repeats),
        "iters": int(iters),
        "effective_runs": int(effective_runs),
        "invocations_per_case_run": int(invocations_per_case_run),
        "total_phase_invocations": int(case_count * len(plan_runs) * invocations_per_case_run),
        "lower_bound_s": float(round(lower_bound_s, 3)),
        "recommended_timeout_s": int(recommended_timeout_s),
        "plan_run_ids": [str(r.get("id") or r.get("name") or "") for r in plan_runs],
        "plan_uses_hailo": bool(any(_run_likely_uses_hailo(r) for r in plan_runs)),
    }


def apply_remote_timeout_hint(requested_timeout_s: Optional[int], hint: Optional[dict[str, Any]]) -> dict[str, Any]:
    """Return an adjusted timeout decision for the remote outer timeout.

    Policy:
    - if the current timeout is disabled/None, leave it disabled
    - if the timeout equals the legacy default (7200 s) and the heuristic says
      this is too low, auto-raise to the recommendation
    - otherwise keep the user's explicit value and only emit a warning
    """

    try:
        requested = None if requested_timeout_s is None else int(requested_timeout_s)
    except Exception:
        requested = None

    if requested is not None and requested <= 0:
        requested = None

    result: dict[str, Any] = {
        "requested_timeout_s": requested,
        "effective_timeout_s": requested,
        "auto_raised": False,
        "warn_too_low": False,
        "hint": dict(hint or {}),
    }
    if requested is None or not hint:
        return result

    try:
        recommended = int(hint.get("recommended_timeout_s") or 0)
    except Exception:
        recommended = 0
    if recommended <= 0 or requested >= recommended:
        return result

    if requested == 7200:
        result["effective_timeout_s"] = recommended
        result["auto_raised"] = True
    else:
        result["warn_too_low"] = True
    return result


def _safe_local_name(s: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in s)


def _streaming_sha256(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Hash large bundles without reading hundreds of MiB into RAM."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _remote_suite_cache_paths(remote_base: str, suite_cache_key: str, bundle_hash: str) -> dict[str, str]:
    root = f"{str(remote_base).rstrip('/')}/_onnx_splitpoint_cache/suite_bundles/{suite_cache_key}/{bundle_hash}"
    return {
        "root": root,
        "bundle": f"{root}/suite_bundle.tar.gz",
        "bundle_ready": f"{root}/BUNDLE_READY",
        "suite": f"{root}/suite",
        "suite_ready": f"{root}/SUITE_READY",
        "population_lock": f"{root}/.population.lock",
    }


def _verified_uncached_suite_extract_command(
    *, remote_bundle: str, remote_suite_dir: str, bundle_hash: str,
) -> str:
    """Verify, extract and discard one per-run transport archive."""

    return (
        f"actual=$(sha256sum {shlex.quote(remote_bundle)} | awk '{{print $1}}'); "
        f'test "$actual" = {shlex.quote(bundle_hash)} && '
        f"rm -rf {shlex.quote(remote_suite_dir)} && "
        f"mkdir -p {shlex.quote(remote_suite_dir)} && "
        f"tar -xzf {shlex.quote(remote_bundle)} -C {shlex.quote(remote_suite_dir)} && "
        f"rm -f {shlex.quote(remote_bundle)}"
    )


@contextmanager
def _remote_cache_population_guard(
    *,
    transport: Any,
    lock_dir: str,
    ready_test_cmd: str,
    log: Callable[[str], None],
    cancel_event: Any = None,
    timeout_s: Optional[int] = None,
):
    """Serialize remote cache upload *and* extraction across run workers.

    Several logical run IDs may target the same physical Jetson.  All of them
    can observe a cache miss before the process-wide upload semaphore is
    acquired.  Without a remote atomic lock they then upload the same archive
    sequentially and race while extracting it.  ``mkdir`` is atomic on the
    remote filesystem, so one worker becomes the population owner while the
    others wait for both ready markers and then reuse the result.
    """
    try:
        wait_timeout = int(timeout_s or os.environ.get("ONNX_SPLITPOINT_REMOTE_CACHE_LOCK_TIMEOUT_S", "1800") or 1800)
    except Exception:
        wait_timeout = 1800
    wait_timeout = max(60, wait_timeout)
    try:
        stale_after = int(os.environ.get("ONNX_SPLITPOINT_REMOTE_CACHE_LOCK_STALE_S", "7200") or 7200)
    except Exception:
        stale_after = 7200
    stale_after = max(wait_timeout, stale_after, 600)

    started = time.monotonic()
    last_notice = -30.0
    owner = False
    qlock = shlex.quote(lock_dir)
    owner_file = shlex.quote(lock_dir.rstrip("/") + "/owner.txt")
    while True:
        if cancel_event is not None and getattr(cancel_event, "is_set", lambda: False)():
            raise RuntimeError("cancelled while waiting for remote suite-cache population lock")

        # Fast path before attempting the atomic lock.  This is important after
        # an owner has written all ready markers and released its lock: without
        # the pre-check, a late waiter could acquire the now-free lock and be
        # reported as a second population owner even though no work remains.
        rc_ready, _ = transport.run(ready_test_cmd, timeout=20)
        if rc_ready == 0:
            log("[remote][cache] sibling worker completed cache population; reusing it")
            break

        rc, _out = transport.run(f"mkdir {qlock} 2>/dev/null", timeout=20)
        if rc == 0:
            # Close the small race between the readiness pre-check and mkdir:
            # a prior owner can finish, remove its lock and expose the ready
            # markers in between those operations.  Re-check after acquiring
            # the directory and relinquish it immediately when the cache is
            # already complete.
            rc_ready_after_lock, _ = transport.run(ready_test_cmd, timeout=20)
            if rc_ready_after_lock == 0:
                transport.run(f"rm -rf {qlock}", timeout=30)
                log("[remote][cache] cache became ready while acquiring population lock; reusing it")
                break
            owner = True
            transport.run(
                f"printf '%s\n' {shlex.quote(_utc_now_iso())} > {owner_file} 2>/dev/null || true",
                timeout=20,
            )
            log(f"[remote][cache] population lock acquired: {lock_dir}")
            break

        # A sibling worker may have completed while this worker was queued.
        rc_ready, _ = transport.run(ready_test_cmd, timeout=20)
        if rc_ready == 0:
            log("[remote][cache] sibling worker completed cache population; reusing it")
            break

        # Recover only genuinely stale lock directories.  The threshold is
        # deliberately longer than the configured wait timeout so a live owner
        # doing a slow upload is never displaced.
        stale_cmd = (
            f"if [ -d {qlock} ]; then "
            f"now=$(date +%s); mt=$(stat -c %Y {qlock} 2>/dev/null || echo $now); "
            f'age=$((now-mt)); if [ "$age" -gt {int(stale_after)} ]; then '
            f"rm -rf {qlock} && echo stale_lock_removed; fi; fi"
        )
        _rc_stale, stale_out = transport.run(stale_cmd, timeout=20)
        if "stale_lock_removed" in str(stale_out or ""):
            log(f"[remote][cache][warn] removed stale population lock older than {stale_after}s")
            continue

        elapsed = time.monotonic() - started
        if elapsed >= wait_timeout:
            raise RuntimeError(
                "Timed out waiting for remote suite-cache population: "
                f"lock={lock_dir}, elapsed={elapsed:.0f}s, timeout={wait_timeout}s"
            )
        if elapsed - last_notice >= 15.0:
            log(f"[remote][cache] waiting for sibling cache population ({elapsed:.0f}s): {lock_dir}")
            last_notice = elapsed
        time.sleep(2.0)

    try:
        yield owner
    finally:
        if owner:
            transport.run(f"rm -rf {qlock}", timeout=30)
            log(f"[remote][cache] population lock released: {lock_dir}")


def _read_json_dict(path: Path) -> dict[str, Any] | None:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None




def _read_json_any(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _energy_case_ids_from_benchmark_set(path: Path) -> list[str]:
    obj = _read_json_any(Path(path))
    if not isinstance(obj, dict):
        return []
    cases = obj.get("cases")
    out: list[str] = []
    if isinstance(cases, list):
        for c in cases:
            if not isinstance(c, dict):
                continue
            raw = str(c.get("case_dir") or c.get("folder") or c.get("case_id") or c.get("boundary") or "").strip()
            if not raw:
                continue
            if raw.startswith("b"):
                cid = raw
            else:
                try:
                    cid = f"b{int(float(raw)):03d}"
                except Exception:
                    cid = raw
            if cid not in out:
                out.append(cid)
    return out


def _energy_targets_for_run_id(*, run_id: str, benchmark_set_json: Path, target_policy: str = "") -> list[dict[str, Any]]:
    rid = str(run_id or "").strip()
    low = rid.lower()
    policy = str(target_policy or "").strip().lower().replace("-", "_")
    cases = _energy_case_ids_from_benchmark_set(benchmark_set_json)
    first_case = cases[0] if cases else "full"
    # Full baselines are canonical: measure once and apply to all cases.  v59j:
    # policy=all additionally measures same-backend composed split diagnostics
    # (ORT CPU/CUDA/TensorRT) per split case so "all splits" really has row-level
    # Energy coverage instead of only a dispatch/full aggregate.
    if low in {"ort_cpu", "ort_cuda", "ort_tensorrt", "tensorrt", "deepx_m1_full", "hailo8", "hailo10"} or low.endswith("_full"):
        targets = [{
            "energy_target_case": first_case if low not in {"deepx_m1_full"} else "full",
            "energy_target_variant": "full",
            "energy_applies_to_all_cases": True,
            "canonical_full_target_case": first_case if first_case else "full",
        }]
        if policy == "all" and low in {"ort_cpu", "ort_cuda", "ort_tensorrt", "tensorrt"}:
            targets.extend({
                "energy_target_case": cid,
                "energy_target_variant": "composed",
                "energy_applies_to_all_cases": False,
                "predicted_rank": int(i + 1),
                "predicted_best": bool(i == 0),
                "same_backend_diagnostic": True,
            } for i, cid in enumerate(cases or [first_case]))
        return targets
    if "_to_" in low or "to_tensorrt" in low or "tensorrt_to" in low or "trt_to" in low:
        # Cases are stored in accepted/predicted order by the benchmark-set generator.
        # Keep the rank so Eval Energy can measure both the measured-best and
        # the predictor-best target without parsing another file.
        return [{
            "energy_target_case": cid,
            "energy_target_variant": "composed",
            "energy_applies_to_all_cases": False,
            "predicted_rank": int(i + 1),
            "predicted_best": bool(i == 0),
        } for i, cid in enumerate(cases or ["full"])]
    return [{
        "energy_target_case": first_case,
        "energy_target_variant": "full",
        "energy_applies_to_all_cases": True,
        "canonical_full_target_case": first_case,
    }]


def _energy_max_targets_per_run_id_from_args(args: Any, current_run_id: str) -> int:
    """Return max row-level energy targets for one run-id.

    Important semantics:
    - explicit 0 disables the cap and measures all selected targets;
    - None/empty means use the policy default;
    - negative values are treated like 0/all for safety.
    """
    rid = str(current_run_id or "").lower()
    policy = str(getattr(args, "energy_target_policy", "") or "").strip().lower()
    raw = getattr(args, "energy_max_targets_per_run_id", None)
    explicit = raw not in (None, "")
    if explicit:
        try:
            val = int(raw)
            return 0 if val <= 0 else val
        except Exception:
            pass
    if policy == "all":
        return 0
    if policy == "best_valid_only" and ("_to_" in rid or "to_tensorrt" in rid or "tensorrt_to" in rid or "trt_to" in rid):
        return 1
    if policy == "best_plus_predicted" and ("_to_" in rid or "to_tensorrt" in rid or "tensorrt_to" in rid or "trt_to" in rid):
        return 2
    # Full baselines are already canonical single-target.  True heterogeneous
    # split run-ids are the expensive ones.  Keep a small default for Eval runs;
    # manual Benchmark users can override to 0/all from the profile/CLI.
    if "_to_" in rid or "to_tensorrt" in rid or "tensorrt_to" in rid or "trt_to" in rid:
        return 2
    return 0

def _energy_apply_target_cap(targets: list[dict[str, Any]], *, args: Any, current_run_id: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cap = _energy_max_targets_per_run_id_from_args(args, current_run_id)
    meta = {"requested_target_count": len(targets), "max_targets_per_run_id": cap, "truncated": False}
    if cap > 0 and len(targets) > cap:
        meta.update({"truncated": True, "kept_target_count": cap, "dropped_target_count": len(targets) - cap})
        return targets[:cap], meta
    meta["kept_target_count"] = len(targets)
    meta["dropped_target_count"] = 0
    return targets, meta


def _rows_from_result_obj(obj: Any) -> list[dict[str, Any]]:
    if isinstance(obj, list):
        return [r for r in obj if isinstance(r, dict)]
    if isinstance(obj, dict):
        for key in ("results", "rows", "measurements", "records"):
            val = obj.get(key)
            if isinstance(val, list):
                return [r for r in val if isinstance(r, dict)]
        return [obj]
    return []


def _energy_find_target_row(search_root: Path, *, run_id: str, target_case: str, target_variant: str) -> dict[str, Any] | None:
    """Find the benchmark result row that corresponds to a row-scoped energy target.

    Eval energy must size u.RECS windows from the target row rather than the
    short Eval benchmark loop count.  Without this, an Eval profile with
    runs=3 creates 3-run energy windows even when Min active s is 30s.
    """
    rid = str(run_id or "").strip().lower()
    tcase = str(target_case or "").strip().lower()
    tvar = str(target_variant or "").strip().lower()
    candidates: list[dict[str, Any]] = []
    roots: list[Path] = []
    for base in (Path(search_root), Path(search_root) / "results", Path(search_root).parent, Path(search_root).parent / "results"):
        try:
            if base.exists() and base not in roots:
                roots.append(base)
        except Exception:
            pass
    seen: set[str] = set()
    for base in roots:
        for fp in list(base.rglob("benchmark_results*.json"))[:400]:
            s = str(fp)
            if s in seen:
                continue
            seen.add(s)
            stem = fp.stem.lower()
            if rid and rid not in stem:
                continue
            obj = _read_json_any(fp)
            for row in _rows_from_result_obj(obj):
                backend = str(row.get("backend") or row.get("run_id") or row.get("provider") or stem).lower()
                if rid and rid not in backend and rid not in stem:
                    continue
                case = str(row.get("case_id") or row.get("case") or row.get("case_dir") or "").strip().lower()
                if tcase and tcase != "full" and case and case != tcase:
                    continue
                variant = str(row.get("variant") or row.get("primary_variant") or "").strip().lower()
                if tvar == "full":
                    if variant and variant != "full" and not any(row.get(k) not in (None, "") for k in ("full_mean_ms", "full_e2e_mean_ms", "total_latency_ms")):
                        continue
                elif tvar == "composed":
                    if variant and variant not in {"composed", "split"}:
                        continue
                    if not any(row.get(k) not in (None, "") for k in ("composed_mean_ms", "split_latency_e2e_ms", "total_latency_ms", "pipeline_fps_selected")):
                        continue
                elif tvar and variant and variant != tvar:
                    continue
                candidates.append(row)
    if not candidates:
        return None
    def _score(row: dict[str, Any]) -> tuple[int, int]:
        s = 0
        if tvar == "full" and any(row.get(k) not in (None, "") for k in ("full_mean_ms", "full_e2e_mean_ms", "total_latency_ms")):
            s += 20
        if tvar == "composed" and any(row.get(k) not in (None, "") for k in ("composed_mean_ms", "split_latency_e2e_ms")):
            s += 20
        if str(row.get("case_id") or "").strip().lower() == tcase:
            s += 5
        if row.get("final_pass") is True or row.get("final_pass_all") is True:
            s += 2
        return (s, len(str(row)))
    return sorted(candidates, key=_score, reverse=True)[0]




def _energy_row_is_valid_for_target(row: dict[str, Any] | None) -> bool:
    if not isinstance(row, dict):
        return False
    def _truthy(k: str) -> bool:
        v = row.get(k)
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return bool(v)
        if isinstance(v, str):
            return v.strip().lower() in {"1", "true", "yes", "ok", "pass", "passed"}
        return False
    if str(row.get("deepx_stage2_contract_status") or row.get("contract_status") or "").lower() in {"all_candidates_rejected", "rejected", "contract_rejected", "native_unstable", "deepx_stage2_native_unstable"}:
        return False
    if _truthy("contract_rejected") or _truthy("runtime_contract_only"):
        return False
    if row.get("runtime_ok") is False or row.get("final_pass") is False or row.get("semantic_validation_ok") is False:
        return False
    return _truthy("final_pass") or _truthy("final_pass_all") or _truthy("semantic_validation_ok") or _truthy("semantic_validation_passed") or _truthy("validation_ok")


def _energy_rank_targets_by_results(search_root: Path, *, run_id: str, targets: list[dict[str, Any]], policy: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Rank/select row-scoped energy targets using the primary benchmark results.

    Policies:
    - best_valid_only: sort by valid/high-FPS/low-latency; the later cap usually keeps one.
    - best_plus_predicted: keep the measured-best valid target and the predictor-best
      accepted target (first benchmark-set case) if they differ.  This is useful for
      Thesis/debugging because it measures both "what the predictor chose" and
      "what actually won" without measuring all split candidates.
    """
    pol = str(policy or "").strip().lower()
    meta: dict[str, Any] = {"target_ranking_policy": pol, "ranked": False, "valid_target_count": 0}
    if not targets or pol not in {"best_valid_only", "best_plus_predicted"}:
        return targets, meta

    scored: list[tuple[tuple[int, float, float], dict[str, Any], dict[str, Any] | None]] = []
    for t in targets:
        row = _energy_find_target_row(search_root, run_id=run_id, target_case=str(t.get("energy_target_case") or ""), target_variant=str(t.get("energy_target_variant") or ""))
        valid = _energy_row_is_valid_for_target(row)
        fps = None
        lat = None
        if isinstance(row, dict):
            for k in ("heterogeneous_pipeline_fps", "pipeline_fps_selected", "full_backend_throughput_fps", "throughput_primary_fps"):
                try:
                    v = float(row.get(k))
                    if v > 0:
                        fps = v; break
                except Exception:
                    pass
            for k in ("composed_mean_ms", "split_latency_e2e_ms", "full_mean_ms", "full_e2e_mean_ms", "total_latency_ms"):
                try:
                    v = float(row.get(k))
                    if v > 0:
                        lat = v; break
                except Exception:
                    pass
        valid_score = 1 if valid else 0
        # Prefer valid rows, then higher FPS, then lower latency.
        scored.append(((valid_score, float(fps or 0.0), -float(lat or 1e9)), dict(t), row if isinstance(row, dict) else None))
    if not scored:
        return targets, meta
    scored.sort(key=lambda x: x[0], reverse=True)
    ranked = [t for _, t, _ in scored]
    valid_count = sum(1 for score, _, _ in scored if score[0] > 0)

    if pol == "best_plus_predicted":
        selected: list[dict[str, Any]] = []
        selected_cases: set[str] = set()
        # measured best valid target (or best ranked target if no valid target exists)
        if ranked:
            best = dict(ranked[0])
            best["energy_selection_reason"] = "measured_best_valid" if valid_count else "measured_best_available"
            selected.append(best)
            selected_cases.add(str(best.get("energy_target_case") or ""))
        # predictor best = first accepted benchmark-set case
        predicted = None
        for t in targets:
            try:
                if int(t.get("predicted_rank") or 0) == 1:
                    predicted = dict(t); break
            except Exception:
                pass
        if predicted is None and targets:
            predicted = dict(targets[0])
        if predicted is not None:
            pc = str(predicted.get("energy_target_case") or "")
            if pc not in selected_cases:
                predicted["energy_selection_reason"] = "predicted_best"
                selected.append(predicted)
                selected_cases.add(pc)
            else:
                # annotate the already selected target for traceability
                for t in selected:
                    if str(t.get("energy_target_case") or "") == pc:
                        t["energy_selection_reason"] = str(t.get("energy_selection_reason") or "") + "+predicted_best"
                        break
        meta.update({
            "ranked": True,
            "valid_target_count": valid_count,
            "selected_target": selected[0] if selected else None,
            "selected_target_count": len(selected),
            "selected_reasons": [t.get("energy_selection_reason") for t in selected],
        })
        return selected or ranked, meta

    meta.update({"ranked": True, "valid_target_count": valid_count, "selected_target": ranked[0] if ranked else None})
    return ranked, meta

def _energy_target_sizing_from_row(row: dict[str, Any] | None, *, target_variant: str) -> dict[str, Any]:
    if not isinstance(row, dict):
        return {}
    tvar = str(target_variant or "").strip().lower()
    def _first_float(*keys: str) -> float | None:
        for k in keys:
            try:
                v = float(row.get(k))
            except Exception:
                continue
            if 0.0 < v < 100000.0:
                return v
        return None
    out: dict[str, Any] = {}
    if tvar == "full":
        lat = _first_float("full_mean_ms", "full_e2e_mean_ms", "total_latency_ms", "latency_mean_ms", "mean_latency_ms")
        if lat:
            out["latency_ms"] = lat
            out["reference_fps"] = 1000.0 / lat
            out["reference_fps_source"] = "target_full_latency_fps"
    else:
        lat = _first_float("composed_mean_ms", "split_latency_e2e_ms", "total_latency_ms", "latency_mean_ms", "mean_latency_ms")
        if lat:
            out["latency_ms"] = lat
        fps = _first_float("pipeline_fps_selected", "heterogeneous_pipeline_fps", "pipeline_fps_measured", "throughput_fps_makespan", "pipeline_fps_with_transfer")
        if fps:
            out["reference_fps"] = fps
            out["reference_fps_source"] = "target_row_pipeline_fps"
        elif lat:
            out["reference_fps"] = 1000.0 / lat
            out["reference_fps_source"] = "target_latency_fps_fallback"
    return out


def _energy_phase_payload_from_aggregate(agg: dict[str, Any], aggregate_path: Path, target: dict[str, Any]) -> dict[str, Any]:
    def _phase(name: str) -> dict[str, Any]:
        phases = agg.get("phases")
        if isinstance(phases, list):
            for ph in phases:
                if isinstance(ph, dict) and str(ph.get("phase") or "") == name:
                    return ph
        return agg if name == "latency" else {}
    lat = _phase("latency")
    stream = _phase("streaming")
    out: dict[str, Any] = {
        "energy_enabled": True,
        "energy_source": "urecs_fast_firmware",
        "energy_measurement_scope": agg.get("energy_measurement_scope") or "command_energy",
        "energy_physical_scope": agg.get("energy_physical_scope"),
        "energy_window_label": agg.get("energy_window_label") or "command",
        "energy_confidence_level": agg.get("confidence_level"),
        "energy_row_scope": True,
        "energy_target_case": target.get("energy_target_case"),
        "energy_target_variant": target.get("energy_target_variant"),
        "energy_applies_to_all_cases": bool(target.get("energy_applies_to_all_cases")),
        "energy_aggregate_path": str(aggregate_path),
        "energy_aggregate_relpath": str(aggregate_path.name),
    }
    if lat:
        lat_work_stats = ((lat.get("repeat_statistics") or {}).get("energy_per_work_unit_j") or (lat.get("repeat_statistics") or {}).get("energy_per_inference_j") or {}) if isinstance(lat.get("repeat_statistics"), dict) else {}
        lat_host_norm_stats = ((lat.get("repeat_statistics") or {}).get("host_normalized_energy_per_work_unit_est_j") or {}) if isinstance(lat.get("repeat_statistics"), dict) else {}
        lat_power_stats = ((lat.get("repeat_statistics") or {}).get("avg_power_w") or {}) if isinstance(lat.get("repeat_statistics"), dict) else {}
        out.update({
            "energy_total_j": lat.get("avg_energy_total_j") or lat.get("energy_total_j") or lat.get("sum_energy_total_j"),
            "avg_power_w": lat.get("avg_power_w") or lat.get("weighted_avg_power_w"),
            "energy_dynamic_j": lat.get("avg_energy_dynamic_j") if lat.get("avg_energy_dynamic_j") is not None else lat.get("energy_dynamic_j"),
            "host_normalized_energy_est_j": lat.get("avg_host_normalized_energy_est_j") if lat.get("avg_host_normalized_energy_est_j") is not None else lat.get("host_normalized_energy_est_j"),
            "row_host_normalized_energy_latency_j_per_inference_est": lat.get("avg_host_normalized_energy_per_work_unit_est_j") or lat.get("host_normalized_energy_per_work_unit_est_j"),
            "host_normalized_energy_per_work_est_j_sample_stddev": lat_host_norm_stats.get("sample_stddev") or lat.get("host_normalized_energy_per_work_unit_est_j_sample_stddev"),
            "host_normalized_energy_per_work_est_j_ci_low": lat_host_norm_stats.get("ci_low") or lat.get("host_normalized_energy_per_work_unit_est_j_ci_low"),
            "host_normalized_energy_per_work_est_j_ci_high": lat_host_norm_stats.get("ci_high") or lat.get("host_normalized_energy_per_work_unit_est_j_ci_high"),
            "host_normalized_average_power_est_w": lat.get("avg_host_normalized_average_power_est_w") or lat.get("host_normalized_average_power_est_w"),
            "host_normalization_role": lat.get("host_normalization_role"),
            "accelerator_idle_correction_requested": lat.get("accelerator_idle_correction_requested"),
            "accelerator_idle_correction_applied": lat.get("accelerator_idle_correction_applied"),
            "accelerator_idle_correction_statuses": lat.get("accelerator_idle_correction_statuses"),
            "row_energy_latency_j_per_inference": lat.get("avg_energy_per_inference_j") or lat.get("energy_per_inference_j") or lat.get("avg_energy_per_work_unit_j"),
            "energy_latency_avg_power_w": lat.get("avg_power_w") or lat.get("weighted_avg_power_w"),
            "row_energy_latency_j_per_inference_sample_stddev": lat_work_stats.get("sample_stddev") or lat.get("energy_per_work_unit_j_sample_stddev"),
            "row_energy_latency_j_per_inference_ci_low": lat_work_stats.get("ci_low") or lat.get("energy_per_work_unit_j_ci_low"),
            "row_energy_latency_j_per_inference_ci_high": lat_work_stats.get("ci_high") or lat.get("energy_per_work_unit_j_ci_high"),
            "energy_latency_avg_power_w_sample_stddev": lat_power_stats.get("sample_stddev") or lat.get("avg_power_w_sample_stddev"),
            "energy_latency_avg_power_w_ci_low": lat_power_stats.get("ci_low") or lat.get("avg_power_w_ci_low"),
            "energy_latency_avg_power_w_ci_high": lat_power_stats.get("ci_high") or lat.get("avg_power_w_ci_high"),
            "energy_latency_repeat_n": lat_work_stats.get("n") or lat.get("valid_postprocessed_runs"),
        })
    if stream:
        stream_stats_block = stream.get("repeat_statistics") if isinstance(stream.get("repeat_statistics"), dict) else {}
        stream_work_stats = stream_stats_block.get("energy_per_pipeline_frame_j") or stream_stats_block.get("energy_per_work_unit_j") or {}
        stream_host_norm_stats = stream_stats_block.get("host_normalized_energy_per_work_unit_est_j") or {}
        stream_power_stats = stream_stats_block.get("avg_power_w") or {}
        out.update({
            "energy_streaming_avg_power_w": stream.get("avg_power_w") or stream.get("weighted_avg_power_w"),
            "host_normalized_streaming_avg_power_est_w": stream.get("avg_host_normalized_average_power_est_w") or stream.get("host_normalized_average_power_est_w"),
            "row_energy_streaming_j_per_frame": stream.get("avg_energy_per_pipeline_frame_j") or stream.get("avg_energy_per_work_unit_j") or stream.get("energy_per_inference_j"),
            "row_host_normalized_energy_streaming_j_per_frame_est": stream.get("avg_host_normalized_energy_per_work_unit_est_j") or stream.get("host_normalized_energy_per_work_unit_est_j"),
            "host_normalized_energy_per_work_est_j_sample_stddev": stream_host_norm_stats.get("sample_stddev") or stream.get("host_normalized_energy_per_work_unit_est_j_sample_stddev"),
            "host_normalized_energy_per_work_est_j_ci_low": stream_host_norm_stats.get("ci_low") or stream.get("host_normalized_energy_per_work_unit_est_j_ci_low"),
            "host_normalized_energy_per_work_est_j_ci_high": stream_host_norm_stats.get("ci_high") or stream.get("host_normalized_energy_per_work_unit_est_j_ci_high"),
            "energy_streaming_j_per_frame": stream.get("avg_energy_per_pipeline_frame_j") or stream.get("avg_energy_per_work_unit_j") or stream.get("energy_per_inference_j"),
            "energy_work_units_per_j": stream.get("avg_energy_work_units_per_j") or stream.get("energy_work_units_per_j"),
            "row_energy_streaming_j_per_frame_sample_stddev": stream_work_stats.get("sample_stddev") or stream.get("energy_per_pipeline_frame_j_sample_stddev") or stream.get("energy_per_work_unit_j_sample_stddev"),
            "row_energy_streaming_j_per_frame_ci_low": stream_work_stats.get("ci_low") or stream.get("energy_per_pipeline_frame_j_ci_low") or stream.get("energy_per_work_unit_j_ci_low"),
            "row_energy_streaming_j_per_frame_ci_high": stream_work_stats.get("ci_high") or stream.get("energy_per_pipeline_frame_j_ci_high") or stream.get("energy_per_work_unit_j_ci_high"),
            "energy_streaming_avg_power_w_sample_stddev": stream_power_stats.get("sample_stddev") or stream.get("avg_power_w_sample_stddev"),
            "energy_streaming_avg_power_w_ci_low": stream_power_stats.get("ci_low") or stream.get("avg_power_w_ci_low"),
            "energy_streaming_avg_power_w_ci_high": stream_power_stats.get("ci_high") or stream.get("avg_power_w_ci_high"),
            "energy_streaming_repeat_n": stream_work_stats.get("n") or stream.get("valid_postprocessed_runs"),
        })
    normalization = stream or lat
    for key in (
        "host_normalization_role",
        "host_normalization_source_run_id",
        "host_normalization_target_variant",
        "host_normalization_identity_verified",
        "accelerator_idle_correction_requested",
        "accelerator_idle_correction_applied",
        "accelerator_idle_correction_statuses",
        "accelerator_idle_w_applied",
        "accelerator_idle_calibration_verified",
        "accelerator_idle_calibration_status",
        "accelerator_idle_calibration_binding_path",
        "accelerator_idle_calibration_binding_sha256",
        "accelerator_idle_calibration_evidence",
        "accelerator_idle_calibrated_at",
        "energy_efficiency_claim_eligible",
    ):
        if normalization.get(key) is not None:
            out[key] = normalization.get(key)
    out.update(resolve_energy_comparison(out))
    return {k: v for k, v in out.items() if v is not None}


def _merge_energy_payload_into_results(results_dir: Path, energy_payload: dict[str, Any]) -> dict[str, Any]:
    """Merge row-scoped u.RECS energy payloads into benchmark_results*.json.

    v58h: evaluation remote bundles store benchmark_results*.json as a top-level
    list, while some manual paths store {"results": [...]}. Older merge code
    only handled the dict shape, so Eval Energy produced good energy artefacts
    but merged zero rows. This helper now supports both shapes and searches a
    small set of candidate result directories.
    """
    touched: list[str] = []
    rows_merged = 0
    if not isinstance(energy_payload, dict) or not energy_payload.get("row_scope"):
        return {"ok": False, "reason": "not_row_scope"}
    targets = [t for t in (energy_payload.get("target_results") or []) if isinstance(t, dict)]
    if not targets:
        return {"ok": False, "reason": "no_targets"}

    base = Path(results_dir)
    search_dirs: list[Path] = []
    for cand in (base, base / "results", base.parent, base.parent / "results"):
        try:
            if cand.is_dir() and cand not in search_dirs:
                search_dirs.append(cand)
        except Exception:
            pass
    json_files: list[Path] = []
    for d in search_dirs:
        json_files.extend(sorted(d.glob("benchmark_results*.json")))
    seen_fp: set[str] = set()
    json_files = [fp for fp in json_files if not (str(fp) in seen_fp or seen_fp.add(str(fp)))]

    def _load_rows(fp: Path) -> tuple[Any, list[dict[str, Any]], str]:
        try:
            obj = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            return None, [], "invalid"
        if isinstance(obj, list):
            return obj, [r for r in obj if isinstance(r, dict)], "list"
        if isinstance(obj, dict):
            for key in ("results", "rows", "measurements", "records"):
                val = obj.get(key)
                if isinstance(val, list):
                    return obj, [r for r in val if isinstance(r, dict)], key
        return obj, [], "unsupported"

    def _write_rows(fp: Path, obj: Any, rows: list[dict[str, Any]], mode: str) -> None:
        if mode == "list":
            _write_json(fp, rows)
        elif isinstance(obj, dict) and mode not in {"invalid", "unsupported"}:
            obj[mode] = rows
            _write_json(fp, obj)
        else:
            return
        csvp = fp.with_suffix(".csv")
        try:
            all_keys: list[str] = []
            for row in rows:
                for k in row.keys():
                    if k not in all_keys:
                        all_keys.append(k)
            with csvp.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=all_keys)
                w.writeheader()
                w.writerows(rows)
        except Exception:
            pass

    for target in targets:
        agg_path_s = str(target.get("energy_aggregate") or target.get("energy_summary") or "")
        agg_path = Path(agg_path_s).expanduser()
        if not agg_path.exists():
            continue
        agg = _read_json_dict(agg_path) or {}
        payload = _energy_phase_payload_from_aggregate(agg, agg_path, target)
        tcase = str(target.get("energy_target_case") or "")
        tvariant = str(target.get("energy_target_variant") or "").lower()
        apply_all = bool(target.get("energy_applies_to_all_cases") or target.get("applies_to_all_cases"))
        for fp in json_files:
            obj, rows, mode = _load_rows(fp)
            if not rows:
                continue
            changed = False
            for row in rows:
                case = str(row.get("case_id") or row.get("case_dir") or row.get("folder") or "")
                var = str(row.get("primary_variant") or row.get("variant") or "").lower()
                if tvariant == "composed" and var in {"split", "composed", ""}:
                    var_match = True
                else:
                    var_match = (var == tvariant) or (tvariant == "full" and (var == "full" or row.get("full_mean_ms") is not None))
                if (apply_all or not tcase or case == tcase) and var_match:
                    row.update(payload)
                    if "semantic_validation_ok" not in row:
                        for k in ("semantic_validation_passed_all", "semantic_validation_passed", "validation_ok", "final_pass_all", "final_pass"):
                            if k in row and row.get(k) not in (None, ""):
                                row["semantic_validation_ok"] = bool(row.get(k))
                                break
                    try:
                        # v58l: energy targets can be merged into mixed benchmark rows
                        # whose variant field is empty even though the target is the
                        # canonical full baseline.  In that case use the full-backend
                        # FPS, not same-backend composed/pipeline diagnostics.
                        var_l = str(row.get("variant") or row.get("primary_variant") or "").lower()
                        target_var_l = str(row.get("energy_target_variant") or tvariant or "").lower()
                        is_full_energy_row = target_var_l == "full" or (apply_all and row.get("full_mean_ms") is not None) or var_l == "full"
                        if is_full_energy_row:
                            full_fps = row.get("full_backend_throughput_fps")
                            if full_fps in (None, ""):
                                fm = row.get("full_mean_ms") or row.get("full_e2e_mean_ms") or row.get("total_latency_ms")
                                try:
                                    fm_f = float(fm)
                                    if fm_f > 0:
                                        full_fps = 1000.0 / fm_f
                                        row["full_backend_throughput_fps"] = full_fps
                                except Exception:
                                    full_fps = None
                            pfs = float(full_fps or row.get("pipeline_fps_selected") or 0.0)
                            if pfs:
                                row["energy_streaming_reference_fps"] = pfs
                                row["energy_streaming_reference_fps_source"] = "full_backend_throughput_fps"
                        else:
                            pfs = float(row.get("pipeline_fps_selected") or row.get("heterogeneous_pipeline_fps") or row.get("full_backend_throughput_fps") or 0.0)
                            if pfs:
                                row["energy_streaming_reference_fps"] = pfs
                                row["energy_streaming_reference_fps_source"] = "pipeline_fps_selected"
                        pwr = float(row.get("energy_streaming_avg_power_w") or row.get("avg_power_w") or 0.0)
                        if pfs and pwr:
                            row["energy_streaming_j_per_frame_from_selected_fps"] = pwr / pfs
                            row["energy_streaming_fps_per_watt_from_selected_fps"] = pfs / pwr
                            row["energy_per_pipeline_frame_from_selected_fps_j"] = pwr / pfs
                            row["pipeline_fps_per_watt_from_selected_fps"] = pfs / pwr
                    except Exception:
                        pass
                    rows_merged += 1
                    changed = True
            if changed:
                _write_rows(fp, obj, rows, mode)
                touched.append(str(fp))
    manifest = {
        "ok": rows_merged > 0,
        "merged_rows": rows_merged,
        "merged_files": sorted(set(touched)),
        "target_count": len(targets),
        "json_file_count": len(json_files),
        "supported_root_shapes": ["list", "dict.results", "dict.rows"],
    }
    try:
        _write_json(Path(results_dir).parent / "energy_merge_manifest.json", manifest)
    except Exception:
        pass
    return manifest

def _infer_suite_benchmark_task(suite_dir: Path, benchmark_set_json: Path | None = None) -> str:
    """Infer task for old/manual benchmark suites whose plan still says auto.

    v52f guard: older generated benchmark sets often store benchmark_task=auto
    and stale resources/validation/coco_50_data even for ResNet/MobileNet/RegNet.
    The remote runner must not convert that into COCO validation for classification.
    """
    candidates: list[Path] = []
    if benchmark_set_json is not None:
        candidates.append(Path(benchmark_set_json))
    candidates.append(Path(suite_dir) / "benchmark_set.json")
    candidates.append(Path(suite_dir) / "benchmark_plan.json")
    blobs: list[str] = []
    explicit_tasks: list[str] = []
    for fp in candidates:
        try:
            if not fp or not fp.exists():
                continue
            obj = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(obj, dict):
            for key in ("task", "benchmark_task", "model_task", "model_name", "model", "model_source"):
                val = obj.get(key)
                if val:
                    blobs.append(str(val))
                if key in {"task", "benchmark_task", "model_task"} and str(val or "").strip().lower() in {"classification", "detection"}:
                    explicit_tasks.append(str(val).strip().lower())
            model_obj = obj.get("model")
            if isinstance(model_obj, dict):
                for key in ("task", "family", "id", "name", "path", "onnx"):
                    val = model_obj.get(key)
                    if val:
                        blobs.append(str(val))
            for run in obj.get("runs") or []:
                if isinstance(run, dict):
                    for key in ("benchmark_task", "task"):
                        val = run.get(key)
                        if val and str(val).strip().lower() not in {"", "auto", "none", "null"}:
                            blobs.append(str(val))
                        if str(val or "").strip().lower() in {"classification", "detection"}:
                            explicit_tasks.append(str(val).strip().lower())
    # Explicit per-model/per-run metadata wins over path and validation-asset
    # heuristics.  This prevents a classification suite from becoming a COCO
    # run merely because a stale validation path contains "coco".
    explicit_unique = sorted(set(explicit_tasks))
    if len(explicit_unique) == 1:
        return explicit_unique[0]
    if len(explicit_unique) > 1:
        return "auto"
    # Also use suite folder name as a last-resort hint.
    blobs.append(str(Path(suite_dir).name))
    blob = " ".join(blobs).lower()
    if any(tok in blob for tok in ("yolo", "coco", "detect", "detection", "object_detection")):
        return "detection"
    if any(tok in blob for tok in ("resnet", "mobilenet", "regnet", "efficientnet", "convnext", "imagenet", "imagenette", "classification", "classify")):
        return "classification"
    return "auto"


def _find_resumable_local_run(
    *,
    local_working_dir: Path,
    suite_dir: Path,
    benchmark_set_json: Path,
    repeat_dir: str,
    host: RemoteHost,
    args: RemoteBenchmarkArgs,
) -> tuple[Path, dict[str, Any], dict[str, Any]] | None:
    """Return the newest local partial run that looks safe to resume.

    Matching is intentionally strict enough to avoid mixing unrelated runs while
    still allowing the user to increase the timeout between retries.
    """
    root = Path(local_working_dir).expanduser().resolve() / "Results" / suite_dir.name / repeat_dir
    if not root.exists() or not root.is_dir():
        return None

    wanted_bench = str(Path(benchmark_set_json).expanduser().resolve())
    wanted_suite_key = _stable_suite_cache_key(suite_dir)
    wanted_provider = str(args.provider or "auto").strip()
    wanted_add_args = str(args.add_args or "").strip()
    wanted_warmup = int(args.warmup)
    wanted_iters = int(args.iters)
    wanted_repeats = int(args.repeats)
    wanted_port = int(host.port or 22)
    wanted_energy_enabled = getattr(args, "energy_enabled", False)
    if type(wanted_energy_enabled) is not bool:
        return None
    wanted_energy_setup_id = getattr(args, "energy_setup_id", "")
    wanted_energy_registry_path = getattr(args, "energy_registry_path", "")
    wanted_energy_registry_sha256 = getattr(
        args, "energy_registry_snapshot_sha256", ""
    )
    if wanted_energy_enabled:
        if not (
            isinstance(wanted_energy_setup_id, str)
            and wanted_energy_setup_id
            and wanted_energy_setup_id == wanted_energy_setup_id.strip()
            and isinstance(wanted_energy_registry_path, str)
            and wanted_energy_registry_path
            and wanted_energy_registry_path
            == wanted_energy_registry_path.strip()
            and Path(wanted_energy_registry_path).is_absolute()
            and isinstance(wanted_energy_registry_sha256, str)
            and re.fullmatch(
                r"[0-9a-f]{64}", wanted_energy_registry_sha256
            )
            is not None
        ):
            # Energy Resume without the fresh preflight binding must never
            # fall back to the older provider/work-count-only matcher.
            return None

    candidates: list[tuple[str, Path, dict[str, Any], dict[str, Any]]] = []
    for run_dir in sorted(root.iterdir(), reverse=True):
        if run_dir.is_symlink() or not run_dir.is_dir():
            continue
        try:
            if run_dir.resolve(strict=True).parent != root:
                continue
        except (OSError, RuntimeError):
            continue
        protected = (
            run_dir / "run_meta.json",
            run_dir / "run_status.json",
            run_dir / "results",
        )
        if any(path.is_symlink() for path in protected):
            continue
        meta = _read_json_dict(run_dir / "run_meta.json")
        status = _read_json_dict(run_dir / "run_status.json")
        if not meta or not status:
            continue
        if (
            meta.get("schema_version") != 1
            or status.get("schema_version") != RUN_STATUS_SCHEMA_VERSION
            or str(meta.get("run_id") or "") != run_dir.name
            or str(meta.get("repeat_idx") or "") != repeat_dir
        ):
            continue
        if str(meta.get("benchmark_set_json") or "") != wanted_bench:
            continue
        if str(meta.get("suite_semantic_cache_key") or "") != wanted_suite_key:
            continue

        meta_host = meta.get("host")
        if not isinstance(meta_host, dict):
            continue
        if str(meta_host.get("user") or "") != str(host.user or ""):
            continue
        if str(meta_host.get("host") or "") != str(host.host or ""):
            continue
        try:
            meta_port = int(meta_host.get("port") or 22)
        except (TypeError, ValueError):
            continue
        if meta_port != wanted_port:
            continue

        meta_args = meta.get("args")
        if not isinstance(meta_args, dict):
            continue
        if str(meta_args.get("provider") or "auto").strip() != wanted_provider:
            continue
        try:
            observed_work = (
                int(meta_args.get("warmup") or 0),
                int(meta_args.get("iters") or 0),
                int(meta_args.get("repeats") or 0),
            )
        except (TypeError, ValueError):
            continue
        if observed_work != (wanted_warmup, wanted_iters, wanted_repeats):
            continue
        if str(meta_args.get("add_args") or "").strip() != wanted_add_args:
            continue
        observed_energy_enabled = meta_args.get("energy_enabled", False)
        if type(observed_energy_enabled) is not bool:
            continue
        if observed_energy_enabled is not wanted_energy_enabled:
            continue
        if wanted_energy_enabled:
            if (
                type(meta_args.get("energy_setup_id")) is not str
                or meta_args.get("energy_setup_id")
                != wanted_energy_setup_id
                or type(meta_args.get("energy_registry_path")) is not str
                or meta_args.get("energy_registry_path")
                != wanted_energy_registry_path
                or type(
                    meta_args.get("energy_registry_snapshot_sha256")
                )
                is not str
                or meta_args.get("energy_registry_snapshot_sha256")
                != wanted_energy_registry_sha256
            ):
                continue
            observed_registry_binding = meta.get(
                "energy_registry_binding"
            )
            if not isinstance(observed_registry_binding, Mapping):
                continue
            if (
                observed_registry_binding.get("schema")
                != "onnx-splitpoint/remote-energy-registry-binding"
                or observed_registry_binding.get("schema_version") != 1
                or observed_registry_binding.get("energy_enabled") is not True
                or observed_registry_binding.get("verified") is not True
                or observed_registry_binding.get("setup_id")
                != wanted_energy_setup_id
                or observed_registry_binding.get("path")
                != wanted_energy_registry_path
                or observed_registry_binding.get("snapshot_sha256")
                != wanted_energy_registry_sha256
            ):
                continue

        status_name = str(status.get("status") or "").strip().lower()
        if status_name not in {"partial", "cancelled"}:
            continue
        if not str(status.get("ended_at") or "").strip():
            continue
        try:
            require_write_target(
                run_dir,
                operation="Automatic remote benchmark resume",
                minimum_free_bytes=1024 * 1024,
                minimum_free_inodes=16,
            )
        except RuntimeError:
            # Read-only historical runs remain available for inspection but
            # must never be reopened by an automatic writer.
            continue
        if not _detect_useful_results(run_dir / "results"):
            continue
        try:
            if any(path.is_symlink() for path in (run_dir / "results").rglob("*")):
                continue
        except OSError:
            continue

        ended = str(status.get("ended_at") or meta.get("started_at") or run_dir.name)
        candidates.append((ended, run_dir, meta, status))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    _ended, run_dir, meta, status = candidates[0]
    return run_dir, meta, status


def _remote_result_collect_script(*, remote_results_dir: str, remote_suite_dir: str) -> str:
    """Build the remote, best-effort result collection script.

    Case runners write their quality requests below ``b*/results_*`` and were
    already covered by the historic per-case loop.  DeepX Full is a suite-level
    producer and writes to ``results/<run_id>/task_quality_inputs`` instead.
    Copy those small scientific inputs explicitly, preserving the run-id, so
    they reach the management quality queue without copying the complete DeepX
    work directory or compiled accelerator artefacts.
    """

    rr = str(remote_results_dir).replace('"', '\\"')
    rs = str(remote_suite_dir).replace('"', '\\"')
    return (
        "set -e; shopt -s nullglob; "
        f"suite=\"{rs}\"; "
        f"out=\"{rr}\"; "
        "mkdir -p \"$out\"; "
        # Suite-level artifacts (including suite logs).
        "for p in \"$suite\"/benchmark_results_* \"$suite\"/benchmark_summary_* "
        "\"$suite\"/benchmark_table_* \"$suite\"/benchmark_tables_* "
        "\"$suite\"/benchmark_report_* \"$suite\"/paper_figures_* "
        "\"$suite\"/benchmark_plan.json \"$suite\"/benchmark_set.json "
        "\"$suite\"/run_meta.json \"$suite\"/benchmark_suite.py "
        "\"$suite\"/scientific_reporter_v60.py \"$suite\"/scientific_report "
        "\"$suite\"/logs; do "
        "  [ -e \"$p\" ] && cp -a \"$p\" \"$out/\" || true; "
        "done; "
        # Per-case artifacts (preserve case-id to avoid collisions).
        "for cd in \"$suite\"/b*; do "
        "  [ -d \"$cd\" ] || continue; "
        "  cid=$(basename \"$cd\"); "
        "  mkdir -p \"$out/$cid\"; "
        "  for p in \"$cd\"/results_* \"$cd\"/validation_*; do "
        "    [ -e \"$p\" ] && cp -a \"$p\" \"$out/$cid/\" || true; "
        "  done; "
        "done; "
        # Suite-level Full producers, notably DeepX Full.  Only copy the
        # portable quality request/candidate directory, not all results.
        "for qdir in \"$suite\"/results/*/task_quality_inputs; do "
        "  [ -d \"$qdir\" ] || continue; "
        "  qrun=$(basename \"$(dirname \"$qdir\")\"); "
        "  mkdir -p \"$out/$qrun\"; "
        "  cp -a \"$qdir\" \"$out/$qrun/\" || true; "
        "done; "
        # DeepX Full seals the exact prepared runtime input before timing.
        # Transport only that three-file contract.  In particular, do not
        # broaden this to arbitrary *.bin files or the sibling DXNN artifact.
        "for pdir in \"$suite\"/results/*/prepared_input; do "
        "  [ -d \"$pdir\" ] || continue; "
        "  prun=$(basename \"$(dirname \"$pdir\")\"); "
        "  mkdir -p \"$out/$prun/prepared_input\"; "
        "  for name in native_full_input_manifest.json runtime_input.bin input_rgb_uint8.bin; do "
        "    [ -f \"$pdir/$name\" ] && cp -a \"$pdir/$name\" \"$out/$prun/prepared_input/$name\" || true; "
        "  done; "
        "done; "
        # Quality-FIRST TensorRT companions live outside bNNN case folders and
        # emit no benchmark row.  Flatten only their portable inputs to a
        # canonical run directory; the workflow collector adds the one
        # authoritative physical-target prefix on the management side.
        "for qdir in \"$suite\"/native_full_quality/results_native_full_tensorrt/task_quality_inputs "
        "\"$suite\"/native_full_quality/*/results_native_full_tensorrt/task_quality_inputs "
        "\"$suite\"/native_full_quality/*/*/results_native_full_tensorrt/task_quality_inputs; do "
        "  [ -d \"$qdir\" ] || continue; "
        "  mkdir -p \"$out/results_native_full_tensorrt\"; "
        "  cp -a \"$qdir\" \"$out/results_native_full_tensorrt/\" || true; "
        "done"
    )


def preflight_remote_energy_dispatch(
    *,
    host: RemoteHost,
    args: RemoteBenchmarkArgs,
    registry_path: str | Path | None = None,
    registry: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Admit an FS-energy dispatch before any workload or transport.

    The Benchmark tab first executes an unmeasured primary deployment, so its
    admission cannot live only inside the later collector path.  This shared,
    read-only gate binds the fresh registry, exact SSH endpoint, configured
    inherited method, and installed source integrity before that primary run.
    """

    from onnx_splitpoint_tool.energy.config import (
        default_registry_path,
        energy_defaults_from_registry,
        energy_setup_from_registry,
        hardware_registry_snapshot_sha256,
        load_hardware_registry,
        normalise_ssh_extra_args,
    )
    from onnx_splitpoint_tool.energy.method_manifest import (
        verify_configured_energy_method,
    )

    enabled = getattr(args, "energy_enabled", False)
    if type(enabled) is not bool:
        raise RuntimeError(
            "remote energy admission failed before transport: "
            "energy_enabled_not_literal_boolean"
        )
    if enabled is not True:
        return {
            "status": "not_requested",
            "verified": False,
            "registry": None,
            "setup": None,
            "method_verification": {},
        }

    selected_registry_input = Path(
        registry_path
        or getattr(args, "energy_registry_path", "")
        or default_registry_path()
    ).expanduser()
    selected_registry_path = selected_registry_input.resolve(strict=False)
    current_registry = (
        dict(registry)
        if registry is not None
        else load_hardware_registry(selected_registry_path)
    )
    raw_setup_id = getattr(args, "energy_setup_id", "")
    if raw_setup_id in (None, ""):
        raw_setup_id = host.id

    def _canonical_token(value: Any) -> bool:
        return bool(
            isinstance(value, str)
            and value
            and value == value.strip()
            and not any(
                character.isspace()
                or ord(character) < 32
                or ord(character) == 127
                for character in value
            )
        )

    errors: list[str] = []
    if not _canonical_token(raw_setup_id):
        errors.append("energy_setup_id_not_canonical")
        setup_id = ""
    else:
        setup_id = raw_setup_id

    # These readers are configuration-only.  Resolve even an invalid id so a
    # malformed or duplicate registry remains an authoritative preflight error.
    setup = energy_setup_from_registry(
        current_registry,
        setup_id,
        registry_path=selected_registry_input,
    )
    defaults = energy_defaults_from_registry(current_registry)
    scope_raw = getattr(args, "energy_physical_scope", None)
    scope = (
        getattr(defaults, "physical_scope", "")
        if scope_raw is None
        else scope_raw
    )
    if not (isinstance(scope, str) and scope in {"FS", "FULL_SYSTEM"}):
        errors.append("energy_physical_scope_not_exact_full_system")
    window_raw = getattr(args, "energy_window_label", None)
    window = (
        getattr(defaults, "window_label", "")
        if window_raw is None
        else window_raw
    )
    if not (isinstance(window, str) and window == "command"):
        errors.append("energy_window_label_not_exact_command")
    if not setup_id:
        errors.append("energy_setup_id_missing")
    if getattr(defaults, "registry_contract_valid", False) is not True:
        errors.extend(
            str(value)
            for value in tuple(
                getattr(defaults, "registry_contract_errors", ()) or ()
            )
        )
    if getattr(setup, "registry_contract_valid", False) is not True:
        errors.extend(
            str(value)
            for value in tuple(
                getattr(setup, "registry_contract_errors", ()) or ()
            )
        )
    if getattr(setup, "hardware_registry_provenance_valid", False) is not True:
        errors.extend(
            str(value)
            for value in tuple(
                getattr(setup, "hardware_registry_provenance_errors", ())
                or ("hardware_registry_provenance_not_verified",)
            )
        )
    try:
        fresh_registry = load_hardware_registry(selected_registry_path)
        fresh_registry_sha = hardware_registry_snapshot_sha256(fresh_registry)
    except Exception as exc:
        fresh_registry_sha = ""
        errors.append(
            "hardware_registry_fresh_reload_failed:"
            f"{type(exc).__name__}:{exc}"
        )
    if fresh_registry_sha != str(
        getattr(setup, "hardware_registry_snapshot_sha256", "") or ""
    ):
        errors.append("hardware_registry_snapshot_stale_or_mismatched")
    if getattr(setup, "jetson_identity_valid", False) is not True:
        errors.extend(
            str(value)
            for value in tuple(
                getattr(setup, "jetson_identity_errors", ()) or ()
            )
        )
    if getattr(setup, "enabled", None) is not True:
        errors.append("energy_setup_not_enabled")
    if getattr(setup, "expected_channel_bindings_valid", False) is not True:
        errors.extend(
            str(value)
            for value in tuple(
                getattr(setup, "expected_channel_binding_errors", ())
                or ("energy_expected_channel_bindings_invalid",)
            )
        )

    if not _canonical_token(host.host):
        errors.append("remote_host_address_not_canonical")
    if not _canonical_token(host.user):
        errors.append("remote_host_user_not_canonical")
    if type(host.port) is not int:
        errors.append("remote_host_port_not_exact_integer")
        host_port = 0
    else:
        host_port = host.port
        if not 1 <= host_port <= 65535:
            errors.append("remote_host_port_out_of_range")
    if not isinstance(host.ssh_extra_args, str):
        errors.append("remote_host_ssh_extra_args_not_string")
        host_ssh_args = ""
    else:
        host_ssh_args = normalise_ssh_extra_args(host.ssh_extra_args)

    expected_endpoint = (
        str(getattr(setup, "jetson_address", "")),
        str(getattr(setup, "jetson_user", "")),
        int(getattr(setup, "jetson_port", 0) or 0),
        normalise_ssh_extra_args(
            getattr(setup, "jetson_ssh_extra_args", "")
        ),
    )
    actual_endpoint = (
        host.host if isinstance(host.host, str) else "",
        host.user if isinstance(host.user, str) else "",
        host_port,
        host_ssh_args,
    )
    if actual_endpoint != expected_endpoint:
        errors.append("remote_host_energy_endpoint_mismatch")

    method: dict[str, Any] = {}
    if not errors:
        method = verify_configured_energy_method(
            setup_id,
            registry=current_registry,
        )
        verification = (
            dict(method.get("verification") or {})
            if isinstance(method.get("verification"), Mapping)
            else {}
        )
        runtime_binding_id = str(
            method.get("runtime_binding_id")
            or verification.get("runtime_binding_id")
            or ""
        )
        source = method.get("source_integrity_verification")
        if not isinstance(source, Mapping):
            admission = verification.get("configured_method_admission")
            if isinstance(admission, Mapping):
                source = admission.get(
                    "source_integrity_binding_verification"
                )
        source = dict(source) if isinstance(source, Mapping) else {}
        if (
            method.get("verified") is not True
            or str(method.get("status") or "")
            != "inherited_validated_method_verified"
            or not runtime_binding_id
            or source.get("ok") is not True
            or str(source.get("status") or "") != "verified"
        ):
            errors.append("configured_energy_method_not_verified")
            if (
                source.get("ok") is not True
                or str(source.get("status") or "") != "verified"
            ):
                errors.append(
                    "configured_energy_method_source_integrity_not_verified"
                )
            errors.extend(
                str(value)
                for value in list(method.get("configuration_errors") or [])
            )
            errors.extend(
                str(value)
                for value in list(method.get("runtime_binding_errors") or [])
            )
            errors.extend(
                str(value) for value in list(source.get("errors") or [])
            )

    if errors:
        raise RuntimeError(
            "remote energy admission failed before transport: "
            + "; ".join(dict.fromkeys(errors))
        )

    args.energy_physical_scope = scope
    args.energy_window_label = window
    args.energy_registry_path = str(selected_registry_path)
    args.energy_registry_snapshot_sha256 = str(
        getattr(setup, "hardware_registry_snapshot_sha256", "") or ""
    )
    return {
        "status": "verified",
        "verified": True,
        "registry": current_registry,
        "setup": setup,
        "defaults": defaults,
        "method_verification": method,
        "setup_id": setup_id,
        "physical_scope": scope,
        "window_label": window,
        "expected_endpoint": expected_endpoint,
    }


@_serialized_remote_storage_by_host
def run_remote_benchmark(
    *,
    host: RemoteHost,
    benchmark_set_json: Path,
    repeats_idx: str = "1",
    local_working_dir: Path,
    run_id: str,
    args: RemoteBenchmarkArgs,
    log: Callable[[str], None],
    progress: Callable[[float, str], None],
    cancel_event,
    remote_process_registry: RemoteProcessLeaseRegistry | None = None,
    workflow_session_id: str = "",
) -> dict:
    """Run a benchmark suite on a remote host (ssh/scp).

    Returns a dict for the GUI with:
        ok: bool
        local_run_dir: str
        remote_run_dir: str
        error: optional error string
    """

    if compiler_dispatch_forbidden():
        raise RuntimeError(cache_miss_blocked_message(
            "remote_generic_dispatch",
            "cache_verify_only permits only the attested Native cache-hit row; "
            "Generic benchmark-suite dispatch is disabled",
        ))

    t0 = time.time()

    benchmark_set_json = Path(benchmark_set_json).expanduser().resolve()
    suite_dir = benchmark_set_json.parent
    bench_payload = _read_json_dict(benchmark_set_json) or {}
    plan_payload = bench_payload.get('plan') if isinstance(bench_payload.get('plan'), dict) else {}
    objective_value_raw = str(plan_payload.get('objective') or bench_payload.get('objective') or 'latency').strip() or 'latency'
    objective_value = objective_value_raw.strip().lower() or 'latency'
    energy_enabled_raw = getattr(args, "energy_enabled", False)
    if not isinstance(energy_enabled_raw, bool):
        raise RuntimeError(
            "remote energy admission failed before transport: "
            "energy_enabled_not_literal_boolean"
        )
    energy_enabled_for_meta = energy_enabled_raw
    energy_registry_preflight = None
    energy_setup_preflight = None
    energy_setup_id_preflight = ""
    energy_registry_admission_verified = False
    if energy_enabled_for_meta:
        objective_value = "remote benchmark with energy"
        admission = preflight_remote_energy_dispatch(
            host=host,
            args=args,
            registry_path=(
                getattr(args, "energy_registry_path", "") or None
            ),
        )
        energy_registry_preflight = admission["registry"]
        energy_setup_preflight = admission["setup"]
        energy_setup_id_preflight = str(admission["setup_id"])
        energy_registry_admission_verified = (
            admission.get("verified") is True
        )
    timeout_hint = estimate_remote_timeout_hint(suite_dir, args)
    timeout_decision = apply_remote_timeout_hint(getattr(args, "timeout_s", None), timeout_hint)
    effective_outer_timeout_s = timeout_decision.get("effective_timeout_s")

    requested_run_id = str(run_id)
    repeat_dir = _safe_local_name(str(repeats_idx).strip() or "1")
    resume_requested = False
    resume_meta: dict[str, Any] | None = None
    local_results_root = Path(local_working_dir).expanduser().resolve() / "Results" / suite_dir.name / repeat_dir

    require_write_target(
        local_results_root,
        operation="Remote benchmark local result",
        minimum_free_bytes=16 * 1024 * 1024,
        minimum_free_inodes=128,
    )

    if bool(getattr(args, "resume", True)):
        resume_hit = _find_resumable_local_run(
            local_working_dir=Path(local_working_dir),
            suite_dir=suite_dir,
            benchmark_set_json=benchmark_set_json,
            repeat_dir=repeat_dir,
            host=host,
            args=args,
        )
        if resume_hit is not None:
            local_run_dir, prev_meta, _prev_status = resume_hit
            prev_run_id = str(prev_meta.get("run_id") or local_run_dir.name)
            run_id = prev_run_id
            resume_requested = True
            resume_meta = {
                "requested_run_id": requested_run_id,
                "resumed_run_id": prev_run_id,
                "previous_started_at": prev_meta.get("started_at"),
            }
        else:
            local_run_dir = local_results_root / run_id
    else:
        local_run_dir = local_results_root / run_id

    started_at = _utc_now_iso()
    energy_registry_path_meta = str(
        getattr(args, "energy_registry_path", "") or ""
    )
    energy_registry_snapshot_meta = str(
        getattr(args, "energy_registry_snapshot_sha256", "") or ""
    )
    energy_registry_binding = {
        "schema": "onnx-splitpoint/remote-energy-registry-binding",
        "schema_version": 1,
        "energy_enabled": energy_enabled_for_meta,
        "setup_id": str(
            energy_setup_id_preflight
            or getattr(args, "energy_setup_id", "")
            or ""
        ),
        "path": energy_registry_path_meta,
        "snapshot_sha256": energy_registry_snapshot_meta,
        "verified": bool(
            energy_registry_admission_verified
            and energy_registry_path_meta
            and Path(energy_registry_path_meta).is_absolute()
            and re.fullmatch(
                r"[0-9a-f]{64}", energy_registry_snapshot_meta
            )
            is not None
        ),
    }
    run_meta: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "repeat_idx": str(repeats_idx),
        "started_at": started_at,
        "created_at": started_at,
        "suite_dir": str(suite_dir),
        "benchmark_set_json": str(benchmark_set_json),
        "suite_semantic_cache_key": _stable_suite_cache_key(suite_dir),
        "host": {
            "id": host.id,
            "label": host.label,
            "user": host.user,
            "host": host.host,
            "port": host.port,
            "remote_base_dir": host.remote_base_dir,
        },
        "args": {
            "provider": args.provider,
            "repeats": args.repeats,
            "warmup": args.warmup,
            "iters": args.iters,
            "add_args": args.add_args,
            "timeout_s": args.timeout_s,
            "timeout_s_effective": effective_outer_timeout_s,
            "transfer_mode": args.transfer_mode,
            "reuse_bundle": args.reuse_bundle,
            "tensorrt_cache_root": remote_trt_cache_root if 'remote_trt_cache_root' in locals() else None,
            "resume": bool(getattr(args, "resume", True)),
            "energy_enabled": bool(getattr(args, "energy_enabled", False)),
            "energy_setup_id": str(getattr(args, "energy_setup_id", "") or ""),
            "energy_registry_path": energy_registry_path_meta,
            "energy_registry_snapshot_sha256": (
                energy_registry_snapshot_meta
            ),
            "energy_run_count": int(getattr(args, "energy_run_count", 1) or 1),
            "cleanup_remote_after_download": bool(getattr(args, "cleanup_remote_after_download", True)),
            "cleanup_remote_on_partial": bool(getattr(args, "cleanup_remote_on_partial", False)),
        },
        "timeout_estimate": {
            **timeout_hint,
            "auto_raised": bool(timeout_decision.get("auto_raised")),
            "warn_too_low": bool(timeout_decision.get("warn_too_low")),
        },
        "resume": {
            "enabled": bool(getattr(args, "resume", True)),
            "reused_previous_run": resume_requested,
            "requested_run_id": requested_run_id,
            "active_run_id": str(run_id),
            "details": resume_meta,
        },
        "energy_registry_binding": dict(energy_registry_binding),
        # Keep an explicit objective for future-proofing and dissertation exports.
        "objective": objective_value,
        "objective_raw": objective_value_raw,
        "objective_source": "energy_dispatch_override" if energy_enabled_for_meta else "benchmark_plan",
    }

    # Phase-0 invariant: create artifacts BEFORE any SSH work happens.
    init_local_run_artifacts(local_run_dir, run_meta)

    # Always keep a local transcript so remote-debug is not lost, even if the
    # GUI buffer is cleared or the run fails mid-way.
    _gui_log = log
    runner_log_path = local_run_dir / "logs" / "runner.log"

    def log(line: str) -> None:  # type: ignore[no-redef]
        """Log output to GUI and to a local transcript.

        We sanitize logs to improve UX:
        - replace carriage returns (\r) with newlines
        - strip ANSI escape sequences
        """

        sanitized = sanitize_log("" if line is None else str(line))
        if sanitized == "":
            return

        # sanitize_log may expand CR to multiple lines.
        out_lines = sanitized.split("\n")
        for out_line in out_lines:
            _gui_log(out_line)

        try:
            with runner_log_path.open("a", encoding="utf-8") as f:
                for out_line in out_lines:
                    f.write(out_line + "\n")
        except Exception:
            # best-effort only
            pass
    
    if resume_requested:
        try:
            log(f"[resume] Reusing previous partial run: {run_id} (requested new run id was {requested_run_id})")
            log(f"[resume] Local resume dir: {local_run_dir}")
        except Exception:
            pass

    try:
        log(
            "[timeout-estimate] "
            f"cases={timeout_hint.get('case_count')} plan_runs={timeout_hint.get('planned_run_count')} "
            f"effective_runs={timeout_hint.get('effective_runs')} total_phase_invocations≈{timeout_hint.get('total_phase_invocations')} "
            f"lower_bound≈{timeout_hint.get('lower_bound_s')}s recommended≈{timeout_hint.get('recommended_timeout_s')}s"
        )
        if bool(timeout_decision.get("auto_raised")):
            log(
                "[timeout-estimate] "
                f"auto-raised remote outer timeout from {timeout_decision.get('requested_timeout_s')}s "
                f"to {timeout_decision.get('effective_timeout_s')}s because the legacy default looked too low for this suite."
            )
        elif bool(timeout_decision.get("warn_too_low")):
            log(
                "[timeout-estimate] warning: current remote outer timeout "
                f"({timeout_decision.get('requested_timeout_s')}s) is below the heuristic recommendation "
                f"({timeout_hint.get('recommended_timeout_s')}s)."
            )
    except Exception:
        pass

    # Keep this available for error reporting (even if we fail mid-way).
    remote_run_dir: Optional[str] = None
    force_bundle_rebuild = False
    last_suite_progress: dict[str, Any] | None = None
    recent_remote_lines: list[str] = []
    bundle_stats_payload: dict[str, Any] = {}

    benchmark_task_norm = normalize_benchmark_task(getattr(args, "benchmark_task", "auto"), log=log)
    if benchmark_task_norm == "auto":
        inferred_task = _infer_suite_benchmark_task(suite_dir, benchmark_set_json)
        if inferred_task != "auto":
            benchmark_task_norm = inferred_task
            try:
                log(f"[validation] Inferred benchmark task from suite/model metadata: {benchmark_task_norm}")
            except Exception:
                pass
    mini_classification_eval_norm = normalize_mini_classification_eval(getattr(args, "mini_classification_eval", False), log=log)
    validation_budget_authoritative = bool(getattr(args, "validation_budget_authoritative", False))
    if validation_budget_authoritative:
        # Generated Evaluation Workflow plans already carry a per-run source and
        # exact budget.  Keep global CLI fields empty so benchmark_suite.py uses
        # the plan rather than a stale legacy default.
        validation_images_norm = None
        validation_max_images_norm = int(getattr(args, "validation_max_images", 0) or 0)
        validation_use_embedded = False
    else:
        validation_images_norm, validation_max_images_norm, validation_use_embedded = normalize_semantic_validation_request(
            getattr(args, "validation_images", ""),
            getattr(args, "validation_max_images", 0),
            benchmark_task=benchmark_task_norm,
            log=log,
        )
    if validation_use_embedded:
        log("[validation] Using prepared COCO-50 as the suite semantic validation source.")
    elif benchmark_task_norm == "classification" and not str(validation_images_norm or "").strip():
        log("[validation] Classification benchmark task selected without a labeled dataset; dataset semantic validation stays disabled.")

    # Best-effort: refresh runner scripts inside an existing suite.
    # Older suites may contain stale harness files; bundling should be self-healing.
    # IMPORTANT: Only touch files whose content actually changed so bundle caching
    # remains effective.
    try:
        # v60q: all setup workers operate on the same generated suite.  Refresh
        # and subset materialisation must complete once before any worker starts
        # scanning or packaging that suite.
        with _exclusive_suite_refresh_guard(suite_dir, log=log, cancel_event=cancel_event):
            refresh_stats = refresh_suite_harness(
                suite_dir,
                benchmark_set_json=benchmark_set_json,
                validation_images=None if validation_budget_authoritative else validation_images_norm,
                validation_max_images=None if validation_budget_authoritative else int(validation_max_images_norm or 0),
                validation_reference_mode=str(getattr(args, "validation_reference_mode", "auto") or "auto"),
                mini_coco_ap50=bool(getattr(args, "mini_coco_ap50", False)),
                benchmark_task=benchmark_task_norm,
                mini_classification_eval=mini_classification_eval_norm,
                log=log,
            )
        if bool(refresh_stats.get("changed")):
            force_bundle_rebuild = True
        # v52i: suite refresh may convert a bare classification preset such as
        # "imagenette_val_mini_200" into a suite-local manifest path
        # (resources/validation/classification/.../manifest.json).  Bundle
        # slimming must use that effective path, otherwise the prepared
        # Imagenette folder is wrongly removed and the remote runner falls back
        # to single-image/no-dataset validation.
        try:
            eff_val = str(refresh_stats.get("validation_images") or "").strip()
            if eff_val:
                validation_images_norm = eff_val
                validation_use_embedded = False
            eff_max = refresh_stats.get("validation_max_images")
            if eff_max is not None:
                validation_max_images_norm = int(eff_max or 0)
        except Exception:
            pass
        if bool(refresh_stats.get("requires_regeneration_for_part2_host_tail")):
            log(
                "[suite-check] This suite was generated before the Part2 host-tail artifacts were available. "
                "Remote benchmark can still run existing variants, but Hailo Part2 host-tail will stay unavailable "
                "until the benchmark set is regenerated."
            )
    except Exception as e:
        log(f'[warn] Could not refresh suite runner scripts: {e}')

    # NOTE: run_meta.json is already written by init_local_run_artifacts().

    # The benchmark result has a model/target-specific sub-run id, but remote
    # process ownership belongs to the enclosing EvaluationRun.  Reuse the
    # registry's frozen scope when present so journal identity cannot drift.
    remote_lease_scope = resolve_remote_process_lease_scope(
        registry=remote_process_registry,
        fallback_run_id=str(run_id),
        workflow_session_id=str(workflow_session_id),
    )
    transport = SSHTransport(
        host,
        cancel_event=cancel_event,
        remote_lease_scope=remote_lease_scope,
        remote_lease_registry=remote_process_registry,
    )
    remote_base_resolved_for_diag = ""
    # This boundary is authoritative for failure handling.  Read-only probes
    # before it deliberately mint no remote lease.  Once either flag becomes
    # true, the existing exact-cleanup/quarantine path must remain fail-closed.
    remote_mutation_started = False
    remote_leased_operation_started = False

    def _diag_slug(text: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text or "remote"))[:96] or "remote"

    def _write_local_remote_failure_artifact(
        *,
        kind: str,
        stage: str,
        cmd: str,
        rc: int | None,
        output: str,
        extra: dict[str, Any] | None = None,
    ) -> Path:
        # Write a local, permanent diagnostic artifact for failures before remote results exist.
        diag_dir = local_run_dir / "diagnostics"
        diag_dir.mkdir(parents=True, exist_ok=True)
        slug = _diag_slug(stage or kind)
        try:
            remote_host_payload = host.to_dict() if hasattr(host, "to_dict") else {
                "user": getattr(host, "user", None),
                "host": getattr(host, "host", None),
                "port": getattr(host, "port", None),
            }
        except Exception:
            remote_host_payload = {}
        payload = {
            "schema": "onnx-splitpoint/remote-command-failure",
            "schema_version": 1,
            "timestamp_utc": _utc_now_iso(),
            "kind": kind,
            "stage": stage,
            "command": cmd,
            "rc": rc,
            "output_tail": (output or "")[-12000:],
            "remote_host": remote_host_payload,
            "remote_base_raw": str(host.remote_base_dir or "~/splitpoint_runs"),
            "remote_base_resolved": remote_base_resolved_for_diag,
            "remote_run_dir": remote_run_dir,
            "remote_results_dir": locals().get("remote_results_dir", None),
            "extra": extra or {},
        }
        json_path = diag_dir / f"{slug}_{kind}.json"
        log_path = diag_dir / f"{slug}_{kind}.log"
        _write_json(json_path, payload)
        try:
            log_path.write_text(
                "COMMAND:\n" + str(cmd or "") + "\n\nRC:\n" + str(rc) + "\n\nOUTPUT:\n" + str(output or ""),
                encoding="utf-8",
                errors="replace",
            )
        except Exception:
            pass
        return json_path

    def _collect_remote_storage_diagnostics(stage: str, *, failed_cmd: str = "", failed_output: str = "", deep: bool = False) -> Path | None:
        # Best-effort remote storage/space diagnostics that works even when mkdir/upload failed.
        try:
            base_hint = remote_base_resolved_for_diag or (host.remote_base_dir or "~/splitpoint_runs")
            deep_lines = []
            if deep:
                deep_lines = [
                    'echo "=== splitpoint_runs size overview ==="',
                    'if [ -d "$BASE" ]; then',
                    '  du -h -d 2 "$BASE" 2>/dev/null | sort -h | tail -80 || true',
                    'else',
                    '  echo "BASE does not exist: $BASE"',
                    '  PARENT="$(dirname "$BASE")"',
                    '  du -h -d 2 "$PARENT" 2>/dev/null | sort -h | tail -80 || true',
                    'fi',
                    'echo "=== largest files under BASE/PARENT ==="',
                    'if [ -d "$BASE" ]; then',
                    "  find \"$BASE\" -type f -printf '%s %p\\n' 2>/dev/null | sort -n | tail -40 || true",
                    'else',
                    '  PARENT="$(dirname "$BASE")"',
                    "  find \"$PARENT\" -maxdepth 4 -type f -printf '%s %p\\n' 2>/dev/null | sort -n | tail -40 || true",
                    'fi',
                ]
            lines = [
                'set +e',
                f'BASE={shlex.quote(str(base_hint))}',
                f'FAILED_CMD={shlex.quote(str(failed_cmd or ""))}',
                f'FAILED_OUTPUT={shlex.quote(str((failed_output or "")[-4000:]))}',
                'echo "=== splitpoint remote storage diagnostics ==="',
                'date -Is || true',
                f'echo "stage={stage}"',
                'echo "host=$(hostname 2>/dev/null || true)"',
                'echo "user=$(id 2>/dev/null || true)"',
                'echo "pwd=$(pwd 2>/dev/null || true)"',
                'echo "HOME=$HOME"',
                'echo "BASE=$BASE"',
                'echo "FAILED_CMD=$FAILED_CMD"',
                'echo "=== failed output tail ==="',
                "printf '%s\\n' \"$FAILED_OUTPUT\"",
                'echo "=== path listing ==="',
                'ls -ld "$BASE" "$(dirname "$BASE")" "$HOME" 2>&1 || true',
                'echo "=== df -hP ==="',
                'df -hP "$BASE" "$(dirname "$BASE")" "$HOME" 2>&1 || true',
                'echo "=== df -iP ==="',
                'df -iP "$BASE" "$(dirname "$BASE")" "$HOME" 2>&1 || true',
                'echo "=== non-mutating write admission ==="',
                'python3 - "$BASE" <<\'SPLITPOINT_DIAG_PY\'',
                'import json, os, stat, sys',
                'p=os.path.expanduser(sys.argv[1])',
                'while not os.path.exists(p) and os.path.dirname(p) != p: p=os.path.dirname(p)',
                'if os.path.isfile(p): p=os.path.dirname(p)',
                's=os.stat(p); v=os.statvfs(p); uid=os.geteuid(); groups=set(os.getgroups())|{os.getegid()}',
                'mode_ok=(bool(s.st_mode & stat.S_IWUSR) and bool(s.st_mode & stat.S_IXUSR)) if uid==s.st_uid else ((bool(s.st_mode & stat.S_IWGRP) and bool(s.st_mode & stat.S_IXGRP)) if s.st_gid in groups else (bool(s.st_mode & stat.S_IWOTH) and bool(s.st_mode & stat.S_IXOTH)))',
                'print(json.dumps({"probe_path":os.path.realpath(p),"read_only_mount":bool(getattr(os,"ST_RDONLY",1) and v.f_flag & getattr(os,"ST_RDONLY",1)),"permission_bits_allow":mode_ok,"os_access_allow":os.access(p,os.W_OK|os.X_OK),"free_bytes":int(v.f_bavail)*int(v.f_frsize or v.f_bsize),"free_inodes":int(v.f_favail)},sort_keys=True))',
                'SPLITPOINT_DIAG_PY',
            ] + deep_lines
            payload = "\n".join(lines)
            cmd = "bash -s <<'SPLITPOINT_REMOTE_STORAGE_DIAG'\n" + payload + "\nSPLITPOINT_REMOTE_STORAGE_DIAG"
            # Storage diagnostics are deliberately read-only.  In particular,
            # do not publish another lease operation after ENOSPC/EROFS or a
            # rejected retention admission: the lease wrapper itself mutates
            # remote /tmp and can destroy the original failure boundary.
            rc, out = transport.run_read_only(cmd, timeout=90 if deep else 30)
            diag_dir = local_run_dir / "diagnostics"
            diag_dir.mkdir(parents=True, exist_ok=True)
            slug = _diag_slug(stage)
            log_file = diag_dir / f"{slug}_remote_storage_diagnostics.log"
            json_file = diag_dir / f"{slug}_remote_storage_diagnostics.json"
            try:
                log_file.write_text(str(out or ""), encoding="utf-8", errors="replace")
            except Exception:
                pass
            _write_json(json_file, {
                "schema": "onnx-splitpoint/remote-storage-diagnostics",
                "schema_version": 1,
                "timestamp_utc": _utc_now_iso(),
                "stage": stage,
                "rc": rc,
                "remote_base": str(base_hint),
                "output_log": str(log_file.relative_to(local_run_dir)) if log_file.exists() else None,
                "output_tail": str(out or "")[-12000:],
            })
            return json_file
        except Exception as diag_exc:
            try:
                log(f"[diag][warn] failed to collect remote storage diagnostics at stage={stage}: {diag_exc}")
            except Exception:
                pass
            return None

    def run_checked(cmd: str, *, timeout: Optional[int] = None, stage: str = "remote_command") -> str:
        nonlocal remote_leased_operation_started
        remote_leased_operation_started = True
        rc, out = transport.run(cmd, timeout=timeout)
        if rc != 0:
            diag_path = _write_local_remote_failure_artifact(kind="remote_command_failed", stage=stage, cmd=cmd, rc=rc, output=out)
            storage_diag = None
            if any(tok in str(out or "").lower() for tok in ("no space left", "disk quota", "cannot create directory", "mkdir:")) or "mkdir" in cmd:
                storage_diag = _collect_remote_storage_diagnostics(stage=f"{stage}_storage", failed_cmd=cmd, failed_output=out, deep=True)
            hint = ""
            if any(tok in str(out or "").lower() for tok in ("no space left", "disk quota")):
                base_for_hint = str(remote_base_resolved_for_diag or host.remote_base_dir or "~/splitpoint_runs")
                hint = (
                    "\nLikely remote storage problem. Check/clean the remote splitpoint base directory, e.g.:\n"
                    f"  ssh {host.user_host_pretty} 'df -h {shlex.quote(base_for_hint)}; "
                    f"du -h -d 2 {shlex.quote(base_for_hint)} 2>/dev/null | sort -h | tail -50'\n"
                )
            admission_prefix = _remote_command_failure_prefix(
                stage=stage,
                output=out,
            )
            raise RuntimeError(
                admission_prefix
                + f"Remote command failed (rc={rc}) at stage={stage}: {cmd}\n{out}\n"
                f"Local diagnostics: {diag_path}"
                + (f"\nRemote storage diagnostics: {storage_diag}" if storage_diag else "")
                + hint
            )
        return str(out or "")

    def scp_upload_checked(local_path: Path, remote_path: str, *, recursive: bool | None = None, stage: str = "scp_upload") -> None:
        nonlocal remote_mutation_started
        # scp can be very quiet for large suite bundles.  Acquire the shared
        # upload slot *before* announcing "upload started"; older logs started
        # the heartbeat while merely waiting for the slot, which made queued
        # workers look like duplicate active transfers.
        size_mb = None
        try:
            if Path(local_path).is_file():
                size_mb = Path(local_path).stat().st_size / (1024 * 1024)
        except Exception:
            size_mb = None
        try:
            hb_s = float(str(os.environ.get("ONNX_SPLITPOINT_SCP_HEARTBEAT_S", "30") or "30"))
        except Exception:
            hb_s = 30.0

        sem = _remote_upload_semaphore()
        sem_acquired = False
        stop_hb = threading.Event()
        hb_thread = None
        rc = 1
        out = ""
        try:
            if sem is not None:
                try:
                    log(
                        f"[remote][scp] waiting for upload slot stage={stage} "
                        f"(limit={int(float(str(os.environ.get('ONNX_SPLITPOINT_MAX_PARALLEL_UPLOADS', '1') or '1')))})"
                    )
                except Exception:
                    pass
                sem.acquire()
                sem_acquired = True

            try:
                if hb_s > 0 and (size_mb is None or size_mb >= 25 or bool(recursive)):
                    started = time.monotonic()
                    size_txt = f"{size_mb:.1f} MiB" if isinstance(size_mb, (int, float)) else "unknown size"
                    log(
                        f"[remote][scp] upload started stage={stage}: {Path(local_path).name} "
                        f"({size_txt}) -> {remote_path}; this may take minutes"
                    )

                    def _heartbeat() -> None:
                        while not stop_hb.wait(hb_s):
                            try:
                                elapsed = time.monotonic() - started
                                log(
                                    f"[remote][scp] still uploading stage={stage}: "
                                    f"elapsed={elapsed:.0f}s size={size_txt} target={remote_path}"
                                )
                            except Exception:
                                pass

                    hb_thread = threading.Thread(target=_heartbeat, daemon=True)
                    hb_thread.start()
            except Exception:
                hb_thread = None

            remote_mutation_started = True
            rc, out = transport.scp_upload(local_path=local_path, remote_path=remote_path, recursive=recursive)
        finally:
            try:
                stop_hb.set()
                if hb_thread is not None:
                    hb_thread.join(timeout=1.0)
            except Exception:
                pass
            try:
                if sem is not None and sem_acquired:
                    sem.release()
            except Exception:
                pass

        if rc != 0:
            diag_path = _write_local_remote_failure_artifact(
                kind="scp_upload_failed",
                stage=stage,
                cmd=f"scp_upload {local_path} -> {remote_path}",
                rc=rc,
                output=out,
            )
            storage_diag = _collect_remote_storage_diagnostics(
                stage=f"{stage}_storage",
                failed_cmd=f"scp_upload {local_path} -> {remote_path}",
                failed_output=out,
                deep=True,
            )
            raise RuntimeError(
                f"SCP upload failed (rc={rc}): {local_path} -> {remote_path}\n{out}\n"
                f"Local diagnostics: {diag_path}\nRemote storage diagnostics: {storage_diag}"
            )
        try:
            if size_mb is not None and size_mb >= 25:
                log(f"[remote][scp] upload finished stage={stage}: {Path(local_path).name} ({size_mb:.1f} MiB)")
        except Exception:
            pass

    def scp_download_checked(remote_path: str, local_path: Path, *, recursive: bool = False, stage: str = "scp_download") -> None:
        rc, out = transport.scp_download(remote_path=remote_path, local_path=local_path, recursive=recursive)
        if rc != 0:
            diag_path = _write_local_remote_failure_artifact(kind="scp_download_failed", stage=stage, cmd=f"scp_download {remote_path} -> {local_path}", rc=rc, output=out)
            raise RuntimeError(f"SCP download failed (rc={rc}): {remote_path} -> {local_path}\n{out}\nLocal diagnostics: {diag_path}")

    # Phase-0 status tracking (finalized at the end, no early returns)
    remote_base_raw = (host.remote_base_dir or "~/splitpoint_runs").rstrip("/")
    remote_run_dir = f"{remote_base_raw}/{suite_dir.name}/{run_id}/{repeat_dir}"
    remote_suite_dir = f"{remote_run_dir}/suite"
    remote_results_dir = f"{remote_run_dir}/results"
    remote_results_tar = f"{remote_run_dir}/results.tar.gz"

    bench_error: Optional[str] = None
    exception_text: Optional[str] = None
    remote_rc: Optional[int] = None
    terminal_remote_failure: dict[str, Any] = {}
    energy_result_payload: dict[str, Any] | None = None
    cancelled: bool = False
    results_downloaded: bool = False
    remote_resume_state_found: bool = False
    terminal_remote_storage_failure: bool = False
    pre_mutation_dispatch_failure: dict[str, Any] = {}

    try:
        # Resolve remote base (expands ~ and symlinks)
        remote_base = transport.resolve_path_read_only(
            remote_base_raw,
            timeout_s=max(60, int(getattr(args, "timeout_s", 0) or 0)),
        ).rstrip("/")
        remote_base_resolved_for_diag = remote_base
        # This is the first remote storage gate and performs no write probe.
        # It catches EROFS, exhausted filesystems and missing directory access
        # before mkdir/upload/extraction can create another incomplete run.
        capacity_requirement = _remote_run_capacity_requirement_for_args(
            suite_dir,
            args=args,
        )
        base_preflight = _remote_storage_preflight(
            transport,
            remote_base,
            required_free_bytes=int(capacity_requirement["required_free_bytes"]),
            required_free_inodes=int(capacity_requirement["required_free_inodes"]),
            stage="pre_remote_mutation",
        )
        log(
            "[remote][storage] pre-mutation capacity contract ok: "
            f"cold_required={int(capacity_requirement['cold_required_free_bytes']) / (1024 * 1024):.1f} MiB, "
            f"warm_required={int(capacity_requirement['warm_required_free_bytes']) / (1024 * 1024):.1f} MiB, "
            f"free={int(base_preflight.get('free_bytes') or 0) / (1024 * 1024):.1f} MiB, "
            f"inodes={int(base_preflight.get('free_inodes') or 0)}"
        )
        # Store a pre-mkdir storage snapshot locally.  If mkdir/upload fails later,
        # this file gives us df/inode/write-test context even when remote results
        # cannot be created or downloaded.
        try:
            diag_path = _collect_remote_storage_diagnostics("pre_mkdir_storage", failed_cmd="", failed_output="", deep=False)
            if diag_path:
                log(f"[diag] remote storage preflight: {diag_path}")
        except Exception:
            pass
        remote_run_dir = f"{remote_base}/{suite_dir.name}/{run_id}/{repeat_dir}"
        remote_suite_dir = f"{remote_run_dir}/suite"
        remote_results_dir = f"{remote_run_dir}/results"
        remote_results_tar = f"{remote_run_dir}/results.tar.gz"
        suite_cache_key = _stable_suite_cache_key(suite_dir)
        trt_cache_active = int(capacity_requirement["trt_run_count"]) > 0
        trt_builder_abi: dict[str, Any] = {}
        trt_builder_cache_abi: dict[str, Any] = {}
        trt_builder_abi_sha256 = ""
        trt_engine_cache_key = suite_cache_key
        trt_runtime_contract = _trt_engine_runtime_contract(suite_dir, args=args)
        canonical_trt_models: list[dict[str, Any]] = []
        if trt_cache_active:
            canonical_trt_models = _canonical_model_onnx_identities(suite_dir)
            trt_builder_abi = _remote_trt_builder_abi(
                transport,
                remote_venv=str(getattr(args, "remote_venv", "") or ""),
            )
            trt_builder_cache_abi = _trt_engine_builder_abi_contract(trt_builder_abi)
            trt_builder_abi_sha256 = hashlib.sha256(
                json.dumps(
                    trt_builder_cache_abi,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ).encode("utf-8")
            ).hexdigest()
            trt_engine_cache_key = _stable_trt_engine_cache_key(
                suite_dir,
                args=args,
                builder_abi=trt_builder_abi,
            )
        remote_trt_cache_root = (
            f"{remote_base}/_onnx_splitpoint_cache/"
            f"tensorrt_managed_v27516/{trt_engine_cache_key}"
        )

        run_meta["trt_engine_cache"] = {
            "suite_semantic_cache_key": suite_cache_key,
            "engine_cache_key": trt_engine_cache_key,
            "builder_abi_sha256": trt_builder_abi_sha256,
            "builder_abi": trt_builder_abi,
            "builder_cache_abi": trt_builder_cache_abi,
            "runtime_contract": trt_runtime_contract,
            "root": remote_trt_cache_root,
            "layout": _trt_persistent_cache_layout(
                namespace_root=remote_trt_cache_root,
                canonical_full_onnx=canonical_trt_models,
            ),
        }
        run_meta["args"]["tensorrt_cache_root"] = remote_trt_cache_root
        try:
            _write_json(local_run_dir / "run_meta.json", run_meta)
        except Exception:
            pass

        # RemoteHost.user_host is a plain string ("user@host").
        # We keep a pretty variant that includes the port for logs.
        log(f"Remote host: {host.user_host_pretty}")
        log(f"Suite (local): {suite_dir}")
        log(f"[remote] run_dir={remote_run_dir}")
        log(f"[remote][cache] TensorRT cache={remote_trt_cache_root}")
        if trt_cache_active:
            log(
                "[remote][cache] TensorRT engine identity="
                f"{trt_engine_cache_key} builder_abi={trt_builder_abi_sha256}; "
                f"suite/resume identity={suite_cache_key}"
            )
        if resume_requested:
            # ``run()`` publishes a remote lease even for this existence check.
            # From this point onward an unreachable host must therefore remain
            # on the exact-cleanup/quarantine path.
            remote_leased_operation_started = True
            rc_resume_check, _out_resume_check = transport.run(f"test -d {shlex.quote(remote_suite_dir)}", timeout=10)
            remote_resume_state_found = (rc_resume_check == 0)
            if remote_resume_state_found:
                log(f"[resume] Remote suite state found at {remote_suite_dir}; remote benchmark will continue there.")
            else:
                log(f"[resume] Remote suite state not found at {remote_suite_dir}; remote benchmark will restart from scratch but keep the same run id.")

        # Explicit phases so the UI shows where it hangs (mkdir vs untar vs run).
        progress(0.01, "Remote mkdir (run/results)")
        mkdir_cmd = f"mkdir -p {remote_run_dir} {remote_results_dir}"
        log(f"[remote] {mkdir_cmd}")
        remote_mutation_started = True
        run_checked(mkdir_cmd, stage="remote_mkdir")
        progress(0.05, "Remote dirs ready")
        if trt_cache_active:
            trt_max_namespaces, trt_max_bytes = _remote_trt_retention_limits()
            retention_active_run_ids = _extract_run_ids_from_add_args(
                getattr(args, "add_args", "")
            )
            retention_shape_contracts = _trt_preflight_source_shape_contracts(
                suite_dir,
                active_run_ids=(
                    retention_active_run_ids
                    if retention_active_run_ids else None
                ),
            )
            # Deliberately reserve the full conservative engine budget even if
            # current contains valid receipts.  A receipt count alone does not
            # prove the complete expected engine/profile set, so treating that
            # namespace as warm could understate the active working set.  This
            # reserve is protected by the physical-capacity preflight and is
            # additional to the retained non-current cache budget.
            retention_output = run_checked(
                _remote_trt_cache_retention_command(
                    remote_base=remote_base,
                    current_key=trt_engine_cache_key,
                    max_namespaces=trt_max_namespaces,
                    max_bytes=trt_max_bytes,
                    planned_current_growth_bytes=int(
                        capacity_requirement["trt_engine_cache_bytes"]
                    ),
                    legacy_suite_key=suite_cache_key,
                    stable_engine_key=trt_engine_cache_key,
                    builder_abi_sha256=trt_builder_abi_sha256,
                    current_trtexec_sha256=str(
                        trt_builder_abi.get("trtexec_sha256") or ""
                    ),
                    native_trt_precision=str(
                        trt_runtime_contract.get("native_trt_precision") or "fp16"
                    ),
                    native_trt_workspace_mb=int(
                        trt_runtime_contract.get("native_trt_workspace_mb") or 0
                    ),
                    canonical_full_onnx_sha256=[
                        str(identity.get("sha256") or "")
                        for identity in canonical_trt_models
                    ],
                    source_shape_contracts=retention_shape_contracts,
                    preserve_existing_namespaces=remote_lease_scope is not None,
                ),
                stage="remote_trt_cache_retention",
            )
            for retention_line in retention_output.splitlines():
                if retention_line.startswith("SPLITPOINT_TRT_RETENTION_JSON="):
                    retention_json = retention_line.split("=", 1)[1]
                    log("[remote][cache][retention] " + retention_json)
                    try:
                        retention_payload = json.loads(retention_json)
                    except Exception:
                        retention_payload = {}
                    if isinstance(retention_payload, dict):
                        run_meta["trt_engine_cache"]["retention"] = retention_payload
                        if retention_payload.get("retention_deferred"):
                            log(
                                "[trt-cache] RETENTION_DEFERRED "
                                "reason=active_evaluation_workflow "
                                f"preserved_namespaces={retention_payload.get('managed_count')} "
                                f"soft_limits_exceeded={bool(retention_payload.get('retention_limits_exceeded'))}; "
                                "bound model artifacts remain available for later stages"
                            )
                        migration_payload = retention_payload.get(
                            "legacy_receipt_migration"
                        )
                        if isinstance(migration_payload, dict):
                            log(
                                "[trt-cache] migration "
                                f"status={migration_payload.get('status')} "
                                f"trigger={migration_payload.get('trigger')} "
                                f"migrated={int(migration_payload.get('migrated_receipts') or 0)} "
                                f"already_present={int(migration_payload.get('existing_receipts') or 0)}"
                            )
                        prior_eviction = retention_payload.get("prior_eviction")
                        if isinstance(prior_eviction, dict):
                            log(
                                "[trt-cache] prior-eviction "
                                f"identity={prior_eviction.get('namespace_key')} "
                                f"reason={prior_eviction.get('reason')}"
                            )
                        for removed_namespace in retention_payload.get("removed") or []:
                            if isinstance(removed_namespace, dict):
                                log(
                                    "[trt-cache] EVICT "
                                    f"identity={removed_namespace.get('name')} "
                                    f"reason={removed_namespace.get('reason') or 'retained_cache_limit'} "
                                    f"bytes={int(removed_namespace.get('bytes') or 0)}"
                                )
                        try:
                            _write_json(local_run_dir / "run_meta.json", run_meta)
                        except Exception:
                            pass
        else:
            log("[remote][cache] selected plan has no TensorRT workload; managed TRT namespace is not created")
        # ---------------------------------------------------------------------
        # Remote preflight (cheap sanity check before transferring a huge bundle).
        #
        # This does *not* require the suite to be uploaded yet and helps catch:
        # - missing Python modules (onnx / onnxruntime)
        # - missing ORT ExecutionProviders (CUDA / TensorRT)
        # - Hailo runtime / device availability problems
        #
        # A JSON report is written to: <remote_results_dir>/preflight.json
        # (and will be downloaded together with the benchmark results).
        # ---------------------------------------------------------------------
        progress(0.055, "Remote preflight (python / ORT / Hailo / DeepX)")

        want_hailo = False
        want_cuda = False
        want_tensorrt = False
        want_deepx = False
        require_ort = False

        plan_path = suite_dir / "benchmark_plan.json"
        run_id_filters = _extract_run_ids_from_add_args(getattr(args, "add_args", ""))
        run_id_filter = run_id_filters[0] if len(run_id_filters) == 1 else (",".join(run_id_filters) if run_id_filters else "")
        trt_runtime_mode_filter = _extract_trt_runtime_mode_from_add_args(getattr(args, "add_args", ""))
        if plan_path.exists():
            try:
                plan = json.loads(plan_path.read_text(encoding="utf-8"))
                if run_id_filters:
                    plan = _filter_benchmark_plan_for_run_ids(plan, run_id_filters)
                    log(f"[preflight] scoped to run-ids: {','.join(run_id_filters)}")
                wants = _scan_remote_preflight_requirements_from_plan(plan)
                want_hailo = bool(wants.get("hailo"))
                want_cuda = bool(wants.get("cuda"))
                want_tensorrt = bool(wants.get("tensorrt"))
                want_deepx = bool(wants.get("deepx"))
                require_ort = bool(wants.get("onnxruntime"))
                # v59af: Native TRT does not need onnxruntime for TensorRT
                # execution.  Keep ORT mandatory for explicit ort_* run IDs,
                # but allow hailo*_to_trt native smoke/benchmark dispatches on
                # offline Hailo10 hosts with TensorRT but no ORT installed.
                add_args_txt = str(getattr(args, "add_args", "") or "")
                native_trt_requested = any(s in add_args_txt for s in ("--trt-runtime native", "--trt-runtime=native", "--trt-runtime-mode native", "--trt-runtime-mode=native"))
                native_trt_requested = native_trt_requested or any(s in add_args_txt for s in ("--trt-runtime native_preferred", "--trt-runtime=native_preferred", "--trt-runtime-mode native_preferred", "--trt-runtime-mode=native_preferred"))
                if native_trt_requested and run_id_filters and any(("_to_trt" in rid or rid in {"hailo10h", "hailo10", "hailo8"}) for rid in run_id_filters):
                    require_ort = False
            except Exception as e:
                log(f"[preflight] warning: could not parse benchmark_plan.json: {e!r}")

        provider_override = str(getattr(args, "provider", "auto") or "auto").strip().lower()
        if provider_override not in {"", "auto"}:
            # Mirror benchmark_suite.py provider filtering for preflight.
            if provider_override in {"deepx", "deepx_m1", "dx_m1"}:
                want_hailo = False
                want_cuda = False
                want_tensorrt = False
                want_deepx = True
                require_ort = False
            elif provider_override in {"tensorrt", "trt"}:
                want_tensorrt = True
                want_cuda = True
                require_ort = True
            elif provider_override in {"cuda", "cuda_ort"}:
                want_cuda = True
                require_ort = True
            elif provider_override in {"cpu", "cpu_ort", "openvino"}:
                require_ort = True

        # v59ae: Native TensorRT and Hailo-only run IDs must not be blocked by
        # a missing onnxruntime package.  The old gate was correct for ORT-TRT,
        # but too strict for native TRT and for Hailo10/Hailo8 pure rows.
        # Keep ORT mandatory for explicit CPU/CUDA ORT runs.
        if run_id_filter:
            _rid_l = str(run_id_filter).strip().lower()
            if _rid_l in {"ort_cpu", "cpu", "cpu_ort", "cuda", "cuda_ort", "ort_cuda"}:
                pass
            elif _native_trt_can_avoid_ort_for_run(run_id_filter, trt_runtime_mode_filter):
                if require_ort:
                    log(f"[preflight] native TensorRT mode ({trt_runtime_mode_filter}) for run-id={run_id_filter}: onnxruntime is optional, not a hard preflight requirement")
                require_ort = False
            elif _pure_hailo_run_can_avoid_ort(run_id_filter):
                if require_ort:
                    log(f"[preflight] pure Hailo run-id={run_id_filter}: onnxruntime is optional, not a hard preflight requirement")
                require_ort = False

        preflight_remote_path = posixpath.join(remote_results_dir, "preflight.json")

        # Build a small bash script (run remotely) that selects the same Python as the benchmark run.
        preflight_lines: List[str] = []
        preflight_lines.append("set -e")
        preflight_lines.append("SYS_PY=$(command -v python3 || command -v python || true)")
        preflight_lines.append('if [ -z "$SYS_PY" ]; then echo "[preflight] ERROR: python not found" >&2; exit 2; fi')
        preflight_lines.append("ENV_PY=''")

        if args.remote_venv:
            remote_venv_cmd = args.remote_venv.strip()
            if any(ch.isspace() for ch in remote_venv_cmd):
                preflight_lines.append(remote_venv_cmd)
            else:
                venv_path = remote_venv_cmd
                if venv_path.startswith("~/"):
                    venv_path = "$HOME/" + venv_path[2:]
                preflight_lines.append(f'if [ -f "{venv_path}" ]; then source "{venv_path}"; fi')
            preflight_lines.append("ENV_PY=$(command -v python3 || command -v python || true)")

        # Backend needs inferred from benchmark_plan.json.  DeepX-only suites do not need ORT,
        # but mixed ORT/TensorRT/DeepX suites do.
        preflight_lines.append(f"WANT_HAILO={'1' if want_hailo else '0'}")
        preflight_lines.append(f"WANT_CUDA={'1' if want_cuda else '0'}")
        preflight_lines.append(f"WANT_TRT={'1' if want_tensorrt else '0'}")
        preflight_lines.append(f"TRT_RUNTIME_MODE={shlex.quote(trt_runtime_mode_filter)}")
        preflight_lines.append(f"WANT_DEEPX={'1' if want_deepx else '0'}")
        preflight_lines.append(f"REQUIRE_ORT={'1' if require_ort else '0'}")
        preflight_lines.extend(DEEPX_DISCOVERY_SHELL.splitlines())
        preflight_lines.extend(_remote_hailo_auto_env_shell(log_prefix="preflight"))

        preflight_lines.append('RUN_PY="$SYS_PY"')
        preflight_lines.append('ENV_SITE=""')
        preflight_lines.append('SYS_EPS=""')
        preflight_lines.append('ENV_EPS=""')
        preflight_lines.append('SYS_CORE=""')
        preflight_lines.append('ENV_CORE=""')
        preflight_lines.append('SYS_DEEPX=""')
        preflight_lines.append('ENV_DEEPX=""')
        preflight_lines.append('has_cuda() { case "$1" in *CUDAExecutionProvider*) return 0;; *) return 1;; esac; }')
        preflight_lines.append('has_trt() { case "$1" in *TensorrtExecutionProvider*) return 0;; *) return 1;; esac; }')
        preflight_lines.append('SYS_CORE=$("$SYS_PY" -c "import onnx,onnxruntime; print(\\"ok\\")" 2>/dev/null || true)')
        preflight_lines.append('SYS_EPS=$("$SYS_PY" -c \'import onnxruntime as ort; print("|".join(ort.get_available_providers()))\' 2>/dev/null || true)')
        preflight_lines.append('SYS_DEEPX=$("$SYS_PY" -c "import dx_engine; print(\\"ok\\")" 2>/dev/null || true)')
        preflight_lines.append('if [ -n "$ENV_PY" ] && [ -x "$ENV_PY" ]; then')
        preflight_lines.append('  ENV_SITE=$("$ENV_PY" -c \'import site,os; ps=[]; getsp=getattr(site,"getsitepackages",None); ps.extend(getsp() if getsp else []); usp=getattr(site,"getusersitepackages",lambda:None)(); ps.append(usp); ps=[p for p in ps if p and os.path.isdir(p)]; out=[]; [out.append(p) for p in ps if p not in out]; print(":".join(out))\' 2>/dev/null || true)')
        preflight_lines.append('  ENV_CORE=$("$ENV_PY" -c "import onnx,onnxruntime; print(\\"ok\\")" 2>/dev/null || true)')
        preflight_lines.append('  ENV_EPS=$("$ENV_PY" -c \'import onnxruntime as ort; print("|".join(ort.get_available_providers()))\' 2>/dev/null || true)')
        preflight_lines.append('  ENV_DEEPX=$("$ENV_PY" -c "import dx_engine; print(\\"ok\\")" 2>/dev/null || true)')
        preflight_lines.append('  if [ "$WANT_TRT" = "1" ]; then')
        preflight_lines.append('    if has_cuda "$ENV_EPS" && has_trt "$ENV_EPS"; then RUN_PY="$ENV_PY"; elif has_cuda "$SYS_EPS" && has_trt "$SYS_EPS"; then RUN_PY="$SYS_PY"; fi')
        preflight_lines.append('  elif [ "$WANT_CUDA" = "1" ]; then')
        preflight_lines.append('    if has_cuda "$ENV_EPS"; then RUN_PY="$ENV_PY"; elif has_cuda "$SYS_EPS"; then RUN_PY="$SYS_PY"; fi')
        preflight_lines.append('  elif [ "$REQUIRE_ORT" = "1" ]; then')
        preflight_lines.append('    if [ -n "$ENV_CORE" ]; then RUN_PY="$ENV_PY"; elif [ -n "$SYS_CORE" ]; then RUN_PY="$SYS_PY"; fi')
        preflight_lines.append('  elif [ "$WANT_DEEPX" = "1" ]; then')
        preflight_lines.append('    if [ -n "$ENV_DEEPX" ]; then RUN_PY="$ENV_PY"; elif [ -n "$SYS_DEEPX" ]; then RUN_PY="$SYS_PY"; fi')
        preflight_lines.append('  else')
        preflight_lines.append('    if [ -n "$ENV_CORE" ]; then RUN_PY="$ENV_PY"; fi')
        preflight_lines.append('  fi')
        preflight_lines.append('fi')
        preflight_lines.append('if [ "$WANT_HAILO" = "1" ] && [ -n "$ENV_SITE" ]; then export SPLITPOINT_EXTRA_SITES="$ENV_SITE"; fi')
        preflight_lines.append('if [ "$WANT_DEEPX" = "1" ] && [ -n "$ENV_SITE" ]; then export SPLITPOINT_EXTRA_SITES="$ENV_SITE:${SPLITPOINT_EXTRA_SITES:-}"; fi')
        preflight_lines.extend(HAILO_DISCOVERY_SHELL.splitlines())
        preflight_lines.append('export PRECHECK_SYS_PY="$SYS_PY"')
        preflight_lines.append('export PRECHECK_ENV_PY="$ENV_PY"')
        preflight_lines.append('export PRECHECK_RUN_PY="$RUN_PY"')
        preflight_lines.append('export PRECHECK_ENV_SITE="$ENV_SITE"')
        preflight_lines.append('export PRECHECK_EXTRA_SITES="${SPLITPOINT_EXTRA_SITES:-}"')

        preflight_lines.append(f"export PRECHECK_OUT={shlex.quote(preflight_remote_path)}")
        preflight_lines.append(f"export PRECHECK_WANT_HAILO={'1' if want_hailo else '0'}")
        preflight_lines.append(f"export PRECHECK_WANT_CUDA={'1' if want_cuda else '0'}")
        preflight_lines.append(f"export PRECHECK_WANT_TRT={'1' if want_tensorrt else '0'}")
        preflight_lines.append(f"export PRECHECK_TRT_RUNTIME_MODE={shlex.quote(trt_runtime_mode_filter)}")
        preflight_lines.append(f"export PRECHECK_WANT_DEEPX={'1' if want_deepx else '0'}")
        preflight_lines.append(f"export PRECHECK_REQUIRE_ORT={'1' if require_ort else '0'}")
        preflight_lines.append('echo "[preflight] SYS_PY=$SYS_PY ENV_PY=$ENV_PY RUN_PY=$RUN_PY HAILO_PY=${HAILO_PY:-} DEEPX_PY=${DEEPX_PY:-} EXTRA_SITES=${SPLITPOINT_EXTRA_SITES:-}"')

        preflight_py = textwrap.dedent(r"""
        import glob
        import json
        import csv
        import os
        import platform
        import subprocess
        import sys
        import time

        # Optional: add extra site-packages *after* default sys.path.
        # This avoids shadowing a system onnxruntime-gpu with a cpu-only wheel from a venv.
        _extra = os.environ.get("SPLITPOINT_EXTRA_SITES") or os.environ.get("SPLITPOINT_EXTRA_SITE")
        if _extra:
            try:
                import site

                for _p in _extra.split(os.pathsep):
                    _p = (_p or "").strip()
                    if _p and os.path.isdir(_p):
                        site.addsitedir(_p)
            except Exception:
                pass

        out_path = os.environ.get("PRECHECK_OUT", "preflight.json")
        want_hailo = os.environ.get("PRECHECK_WANT_HAILO", "0") == "1"
        want_cuda = os.environ.get("PRECHECK_WANT_CUDA", "0") == "1"
        want_trt = os.environ.get("PRECHECK_WANT_TRT", "0") == "1"
        want_deepx = os.environ.get("PRECHECK_WANT_DEEPX", "0") == "1"
        require_ort = os.environ.get("PRECHECK_REQUIRE_ORT", "0") == "1"

        info = {
            "schema_version": 1,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "python": {
                "executable": sys.executable,
                "version": sys.version,
                "sys_py": os.environ.get("PRECHECK_SYS_PY"),
                "env_py": os.environ.get("PRECHECK_ENV_PY"),
                "run_py": os.environ.get("PRECHECK_RUN_PY"),
                "env_site": os.environ.get("PRECHECK_ENV_SITE"),
                "hailo_py": os.environ.get("PRECHECK_HAILO_PY"),
                "hailo_site": os.environ.get("PRECHECK_HAILO_SITE"),
                "deepx_py": os.environ.get("PRECHECK_DEEPX_PY"),
                "deepx_site": os.environ.get("PRECHECK_DEEPX_SITE"),
                "extra_sites": os.environ.get("PRECHECK_EXTRA_SITES") or os.environ.get("SPLITPOINT_EXTRA_SITES"),
            },
            "platform": {"platform": platform.platform(), "machine": platform.machine()},
            "wants": {"hailo": want_hailo, "cuda": want_cuda, "tensorrt": want_trt, "deepx": want_deepx, "requires_onnxruntime": require_ort, "trt_runtime_mode": os.environ.get("PRECHECK_TRT_RUNTIME_MODE")},
        }

        critical_missing = []

        try:
            import onnx  # type: ignore
            info["onnx"] = {"ok": True, "version": getattr(onnx, "__version__", None)}
        except Exception as e:
            info["onnx"] = {"ok": False, "error": repr(e)}
            if require_ort:
                critical_missing.append("onnx")

        try:
            import onnxruntime as ort  # type: ignore
            info["onnxruntime"] = {
                "ok": True,
                "version": getattr(ort, "__version__", None),
                "available_providers": ort.get_available_providers(),
            }
        except Exception as e:
            info["onnxruntime"] = {"ok": False, "error": repr(e)}
            if require_ort:
                critical_missing.append("onnxruntime")

        # Provider availability hints. v52e: requested CUDA/TensorRT EPs are
        # hard requirements; otherwise the runner could silently execute on CPU
        # while the result is labelled as TensorRT/CUDA.
        if info.get("onnxruntime", {}).get("ok"):
            eps = info["onnxruntime"]["available_providers"]
            warnings = []
            provider_failures = []
            if want_cuda and "CUDAExecutionProvider" not in eps:
                msg = f"CUDAExecutionProvider missing (available: {eps})"
                warnings.append(msg)
                provider_failures.append("CUDAExecutionProvider")
            if want_trt and "TensorrtExecutionProvider" not in eps:
                msg = f"TensorrtExecutionProvider missing (available: {eps})"
                warnings.append(msg)
                provider_failures.append("TensorrtExecutionProvider")
            info["onnxruntime"]["warnings"] = warnings
            info["onnxruntime"]["provider_gate_ok"] = not provider_failures
            info["onnxruntime"]["provider_gate_failures"] = provider_failures
            for ep in provider_failures:
                critical_missing.append(ep)

        # Direct TensorRT diagnostics.  This is intentionally separate from
        # ORT-TRT availability: import tensorrt/trtexec can be OK while
        # TensorrtExecutionProvider is missing, and those are different fixes.
        tensorrt_info = {}
        try:
            import tensorrt as trt  # type: ignore
            tensorrt_info["python_import_ok"] = True
            tensorrt_info["version"] = getattr(trt, "__version__", None)
            tensorrt_info["module_path"] = getattr(trt, "__file__", None)
        except Exception as e:
            tensorrt_info["python_import_ok"] = False
            tensorrt_info["python_import_error"] = repr(e)
        try:
            cp = subprocess.run(["bash", "-lc", "command -v trtexec || true"], capture_output=True, text=True, timeout=5)
            lines = (cp.stdout or "").strip().splitlines()
            tensorrt_info["trtexec"] = lines[-1].strip() if lines else None
        except Exception as e:
            tensorrt_info["trtexec_error"] = repr(e)
        info["tensorrt"] = tensorrt_info

        # v52h: ORT may list TensorRT/CUDA EPs even though the provider .so
        # cannot load at session creation time because Jetson CUDA libraries are
        # missing from the system loader path.  Catch the common cuBLAS case in
        # preflight instead of letting the benchmark silently fall back to CPU.
        cuda_libs = {}
        def _find_cuda_lib(label, soname):
            try:
                import ctypes.util
                found = ctypes.util.find_library(label)
                if found:
                    return found
            except Exception:
                pass
            patterns = [
                f"/usr/local/cuda*/targets/aarch64-linux/lib/{soname}",
                f"/usr/local/cuda*/lib64/{soname}",
                f"/usr/lib/aarch64-linux-gnu/{soname}",
                f"/usr/lib/aarch64-linux-gnu/nvidia/{soname}",
                f"/usr/lib/aarch64-linux-gnu/tegra/{soname}",
            ]
            matches = []
            for pat in patterns:
                try:
                    matches.extend(glob.glob(pat))
                except Exception:
                    pass
            return sorted(set(matches))[0] if matches else None
        cuda_libs["libcublas.so.12"] = _find_cuda_lib("cublas", "libcublas.so.12")
        cuda_libs["libcublasLt.so.12"] = _find_cuda_lib("cublasLt", "libcublasLt.so.12")
        cuda_libs["libcudart.so.12"] = _find_cuda_lib("cudart", "libcudart.so.12")
        cuda_libs["libcudnn.so.9"] = _find_cuda_lib("cudnn", "libcudnn.so.9")
        # v52j: ORT-GPU 1.23 on Jetson may list CUDA/TRT providers even when
        # the CUDA provider cannot load at session creation time.  The concrete
        # failure seen on Orin NX was libcufft.so.11, so gate it up front.
        cuda_libs["libcufft.so.11"] = _find_cuda_lib("cufft", "libcufft.so.11")
        cuda_libs["libcurand.so.10"] = _find_cuda_lib("curand", "libcurand.so.10")
        cuda_libs["libcusolver.so.11"] = _find_cuda_lib("cusolver", "libcusolver.so.11")
        cuda_libs["libcusparse.so.12"] = _find_cuda_lib("cusparse", "libcusparse.so.12")
        info["cuda_libraries"] = cuda_libs
        if want_trt:
            for _soname in ("libcudart.so.12", "libcublas.so.12", "libcublasLt.so.12", "libcudnn.so.9", "libcufft.so.11"):
                if not cuda_libs.get(_soname):
                    critical_missing.append(_soname)
        elif want_cuda:
            for _soname in ("libcudart.so.12", "libcublas.so.12", "libcublasLt.so.12", "libcufft.so.11"):
                if not cuda_libs.get(_soname):
                    critical_missing.append(_soname)

        # v52j: Do a tiny ONNXRuntime provider smoke test.  get_available_providers()
        # can be optimistic; a session can still fall back to CPU when provider
        # shared-library dependencies are missing.  This catches that before the
        # actual benchmark cases run.
        if info.get("onnxruntime", {}).get("ok") and (want_trt or want_cuda):
            provider_smoke = {"requested": None, "ok": False}
            try:
                import tempfile
                import numpy as _np  # type: ignore
                import onnx as _onnx  # type: ignore
                import onnxruntime as _ort  # type: ignore
                from onnx import TensorProto, helper  # type: ignore
                providers = ["CPUExecutionProvider"]
                if want_cuda:
                    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if want_trt:
                    providers = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
                provider_smoke["requested"] = providers
                with tempfile.TemporaryDirectory(prefix="splitpoint_ort_ep_smoke_") as _td:
                    _mp = os.path.join(_td, "ep_smoke.onnx")
                    _x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 4, 4])
                    _y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 4, 4])
                    _node = helper.make_node("Relu", ["input"], ["output"])
                    _graph = helper.make_graph([_node], "splitpoint_ep_smoke", [_x], [_y])
                    _model = helper.make_model(_graph, opset_imports=[helper.make_opsetid("", 13)])
                    _model.ir_version = 9
                    _onnx.save(_model, _mp)
                    _sess = _ort.InferenceSession(_mp, providers=providers)
                    _out = _sess.run(None, {"input": _np.ones((1, 3, 4, 4), dtype=_np.float32)})
                    provider_smoke["providers_in_use"] = list(_sess.get_providers())
                    provider_smoke["output_shape"] = list(_out[0].shape) if _out else None
                    provider_smoke["ok"] = True
                    if want_trt and "TensorrtExecutionProvider" not in provider_smoke.get("providers_in_use", []):
                        provider_smoke["ok"] = False
                        provider_smoke["error"] = "TensorrtExecutionProvider not active in smoke session"
                        critical_missing.append("onnxruntime_provider_smoke_failed:TensorrtExecutionProvider")
                    elif want_cuda and "CUDAExecutionProvider" not in provider_smoke.get("providers_in_use", []):
                        provider_smoke["ok"] = False
                        provider_smoke["error"] = "CUDAExecutionProvider not active in smoke session"
                        critical_missing.append("onnxruntime_provider_smoke_failed:CUDAExecutionProvider")
            except Exception as e:
                provider_smoke["ok"] = False
                provider_smoke["error"] = f"{type(e).__name__}: {e}"
                # Surface the concrete missing shared library in preflight output.
                _msg = str(e)
                for _soname in ("libcufft.so.11", "libcudnn.so.9", "libcublas.so.12", "libcublasLt.so.12", "libcudart.so.12"):
                    if _soname in _msg and _soname not in critical_missing:
                        critical_missing.append(_soname)
                if want_trt:
                    critical_missing.append("onnxruntime_provider_smoke_failed:TensorrtExecutionProvider")
                elif want_cuda:
                    critical_missing.append("onnxruntime_provider_smoke_failed:CUDAExecutionProvider")
            info.setdefault("onnxruntime", {})["provider_smoke"] = provider_smoke

        hailo = {"dev_nodes": sorted(glob.glob("/dev/hailo*"))}

        # Collect some system-level hints (best effort, no sudo)
        try:
            hailo["ps_hailo"] = subprocess.run(
                ["bash", "-lc", "ps aux | grep -i hailo | grep -v grep || true"],
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
        except Exception:
            pass
        try:
            hailo["ls_dev_hailo"] = subprocess.run(
                ["bash", "-lc", "ls -l /dev/hailo* 2>/dev/null || true"],
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
        except Exception:
            pass

        if want_hailo:
            try:
                # Prefer hailo_platform, but fall back to hailort if that's what is installed.
                hp = None
                try:
                    import hailo_platform as hp  # type: ignore

                    hailo["hailo_module"] = "hailo_platform"
                    hailo["hailo_import_ok"] = True
                    hailo["hailo_module_path"] = getattr(hp, "__file__", None)
                except Exception as e_platform:
                    try:
                        import hailort as hp  # type: ignore

                        hailo["hailo_module"] = "hailort"
                        hailo["hailo_import_ok"] = True
                        hailo["hailo_module_path"] = getattr(hp, "__file__", None)
                    except Exception as e_hailort:
                        hailo["hailo_module"] = None
                        hailo["hailo_import_ok"] = False
                        hailo["hailo_import_error"] = {
                            "hailo_platform": repr(e_platform),
                            "hailort": repr(e_hailort),
                        }
                        hp = None

                # Device scan (best effort)
                scan = None
                if hp is not None and hasattr(hp, "Device") and hasattr(hp.Device, "scan"):
                    try:
                        scan = hp.Device.scan()
                    except Exception as e:
                        scan = {"error": repr(e)}
                hailo["device_scan"] = scan

                # Probe VDevice allocation in a subprocess (contains potential crashes).
                # Probe VDevice allocation in a subprocess (contains potential crashes).
                # Important: the child must extend sys.path with SPLITPOINT_EXTRA_SITES too,
                # otherwise the parent can import hailo_platform while the child probe fails.
                probe_code = r'''
        import os
        import site
        import sys
        import traceback
        import math

        _extra = os.environ.get("SPLITPOINT_EXTRA_SITES") or os.environ.get("SPLITPOINT_EXTRA_SITE")
        if _extra:
            for _p in _extra.split(os.pathsep):
                _p = (_p or "").strip()
                if _p and os.path.isdir(_p):
                    site.addsitedir(_p)

        try:
            try:
                import hailo_platform as hp
            except Exception:
                import hailort as hp
        except Exception as e:
            print(f"import_failed: {e!r}", file=sys.stderr)
            sys.exit(2)

        try:
            if hasattr(hp, "VDevice"):
                v = hp.VDevice()
                if hasattr(v, "__enter__"):
                    v.__enter__()
                if hasattr(v, "__exit__"):
                    v.__exit__(None, None, None)
                for meth in ("release", "close"):
                    if hasattr(v, meth):
                        try:
                            getattr(v, meth)()
                        except Exception:
                            pass
                del v
            print("vdevice_ok")
            sys.exit(0)
        except Exception as e:
            traceback.print_exc()
            print(f"vdevice_failed: {e!r}", file=sys.stderr)
            sys.exit(3)
        '''
                proc = subprocess.run(
                    [sys.executable, "-c", probe_code],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                hailo["vdevice_probe"] = {
                    "rc": proc.returncode,
                    "stdout": (proc.stdout or "")[-4000:],
                    "stderr": (proc.stderr or "")[-4000:],
                }

                combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
                low = combined.lower()
                markers = [
                    "not enough free devices",
                    "out_of_physical_devices",
                    "hailo_out_of_physical_devices",
                    "failed to create vdevice",
                    "network_group_not_activated",
                ]
                hits = [m for m in markers if m in low]
                hailo["vdevice_probe_markers"] = hits
                hailo["vdevice_ok"] = (proc.returncode == 0 and not hits)

            except Exception as e:
                hailo["hailo_import_ok"] = False
                hailo["error"] = repr(e)

        deepx = {
            "dev_nodes": sorted(glob.glob("/dev/dxrt*")),
            "deepx_py": os.environ.get("PRECHECK_DEEPX_PY"),
            "deepx_site": os.environ.get("PRECHECK_DEEPX_SITE"),
        }
        if want_deepx:
            def _which(name: str) -> str | None:
                try:
                    cp = subprocess.run(["bash", "-lc", f"command -v {name} || true"], capture_output=True, text=True, timeout=5)
                    out = (cp.stdout or "").strip().splitlines()
                    return out[-1].strip() if out else None
                except Exception:
                    return None
            deepx["run_model"] = _which("run_model")
            deepx["parse_model"] = _which("parse_model")
            deepx["dxrt_cli"] = _which("dxrt-cli")
            try:
                import dx_engine  # type: ignore
                deepx["dx_engine_import_ok"] = True
                deepx["dx_engine_path"] = getattr(dx_engine, "__file__", None)
            except Exception as e:
                deepx["dx_engine_import_ok"] = False
                deepx["dx_engine_import_error"] = repr(e)
            if not deepx.get("run_model"):
                critical_missing.append("run_model")
        info["deepx"] = deepx

        info["hailo"] = hailo

        # write file
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(info, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[preflight] ERROR: failed to write {out_path}: {e!r}", file=sys.stderr)

        # human summary
        print(f"[preflight] wrote {out_path}")
        print(f"[preflight] python: {info['python']['executable']}")

        if info.get("onnx", {}).get("ok") and info.get("onnxruntime", {}).get("ok"):
            print(f"[preflight] onnx {info['onnx'].get('version')} / onnxruntime {info['onnxruntime'].get('version')}")

        if info.get("onnxruntime", {}).get("ok"):
            print(f"[preflight] ort eps: {info['onnxruntime']['available_providers']}")
            for w in info["onnxruntime"].get("warnings", []):
                print(f"[preflight] WARNING: {w}")

        if want_trt:
            trt_info = info.get("tensorrt") or {}
            print(f"[preflight] TensorRT Python: {trt_info.get('python_import_ok')} version={trt_info.get('version')}")
            print(f"[preflight] trtexec: {trt_info.get('trtexec') or '-'}")
            print(f"[preflight] CUDA libs: {info.get('cuda_libraries')}")
            if not trt_info.get('python_import_ok'):
                print(f"[preflight] WARNING: TensorRT Python import failed: {trt_info.get('python_import_error')}")

        if want_hailo:
            print(f"[preflight] /dev/hailo*: {hailo.get('dev_nodes')}")
            if hailo.get("ps_hailo"):
                print(f"[preflight] ps hailo:\n{hailo.get('ps_hailo')}")
            if hailo.get("hailo_import_ok"):
                if hailo.get("device_scan") is not None:
                    print(f"[preflight] Device.scan: {hailo.get('device_scan')}")
                if "vdevice_ok" in hailo:
                    print(f"[preflight] vdevice_ok: {hailo.get('vdevice_ok')} (markers={hailo.get('vdevice_probe_markers')})")
                    if not hailo.get("vdevice_ok"):
                        probe = hailo.get("vdevice_probe") or {}
                        tail = (probe.get("stderr") or probe.get("stdout") or "").strip()
                        if tail:
                            print(f"[preflight] vdevice probe tail: {tail[-400:]}")
            else:
                err = hailo.get('hailo_import_error') or hailo.get('error')
                print(f"[preflight] WARNING: hailo import failed: {err}")

        if want_deepx:
            print(f"[preflight] /dev/dxrt*: {deepx.get('dev_nodes')}")
            print(f"[preflight] run_model: {deepx.get('run_model') or '-'}")
            print(f"[preflight] dx_engine: {deepx.get('dx_engine_import_ok')}")
            if not deepx.get('dx_engine_import_ok'):
                print(f"[preflight] WARNING: dx_engine import failed: {deepx.get('dx_engine_import_error')}")

        if critical_missing:
            print(f"[preflight] ERROR: missing critical runtime components: {critical_missing}", file=sys.stderr)
            if require_ort:
                print("[preflight] Hint: this run still needs onnxruntime for the generated suite reference/validation path. For native TensorRT smoke tests use scripts/native_trt_from_benchmarkset.py; for GUI/remote benchmarks install onnxruntime in the selected remote Python env or select an ORT-capable env plus Hailo EXTRA_SITES.", file=sys.stderr)
            if any(str(x).startswith('libcublas') for x in critical_missing):
                print("[preflight] Hint: Jetson cuBLAS is missing from the loader path. Try: sudo apt install -y libcublas-12-6 libcublas-dev-12-6 && sudo ldconfig. Note: libcublasLt is usually inside libcublas-12-6; there may be no separate libcublaslt-12-6 package.", file=sys.stderr)
            if any(str(x).startswith('libcudnn') for x in critical_missing):
                print("[preflight] Hint: Jetson cuDNN is missing from the loader path. Install/reinstall the JetPack cuDNN runtime package, e.g. sudo apt install -y libcudnn9-cuda-12 libcudnn9-dev-cuda-12 || sudo apt install -y cudnn && sudo ldconfig, then rerun the preflight.", file=sys.stderr)
            if any(str(x).startswith('libcufft') for x in critical_missing):
                print("[preflight] Hint: Jetson cuFFT is missing from the loader path. Try: sudo apt install -y libcufft-12-6 libcufft-dev-12-6 && sudo ldconfig.", file=sys.stderr)
            if any(str(x).startswith('onnxruntime_provider_smoke_failed') for x in critical_missing):
                print("[preflight] Hint: ONNX Runtime listed the requested provider, but a tiny provider session could not activate it. Check CUDA/TensorRT shared libraries and LD_LIBRARY_PATH/ldconfig; CPU fallback is not accepted for TensorRT/CUDA-labelled runs.", file=sys.stderr)
            if want_deepx:
                print("[preflight] Hint: DeepX full needs run_model and usually dx_engine. Verify: source ~/venvs/deepx-runtime/bin/activate && python -c 'import dx_engine' && command -v run_model", file=sys.stderr)
            sys.exit(10)

        sys.exit(0)
        """).strip()

        try:
            compile(preflight_py, "remote_preflight.py", "exec")
        except SyntaxError as _preflight_syntax_error:
            diag_dir = local_run_dir / "diagnostics"
            diag_dir.mkdir(parents=True, exist_ok=True)
            diag_path = diag_dir / "local_preflight_template_syntax_error.json"
            _write_json(diag_path, {
                "schema": "onnx-splitpoint/local-preflight-template-syntax-error",
                "schema_version": 1,
                "error": str(_preflight_syntax_error),
                "lineno": getattr(_preflight_syntax_error, "lineno", None),
                "offset": getattr(_preflight_syntax_error, "offset", None),
                "text": getattr(_preflight_syntax_error, "text", None),
            })
            raise RuntimeError(f"Local remote-preflight template has a Python syntax error before upload: {_preflight_syntax_error}. Diagnostics: {diag_path}")

        preflight_lines.append("\"$RUN_PY\" - <<'PY'\n" + preflight_py + "\nPY")

        preflight_cmd = "bash -lc " + shlex.quote("\n".join(preflight_lines))

        rc, out = transport.run(preflight_cmd, timeout=120)
        log(out)
        if rc != 0:
            raise _remote_preflight_failure(rc, out)

        progress(0.06, "Preflight OK")


        # ----------------------------
        # Transfer suite
        # ----------------------------
        transfer_mode = (args.transfer_mode or "bundle").strip().lower()
        if transfer_mode not in ("bundle", "direct"):
            raise ValueError(f"Unknown transfer_mode: {args.transfer_mode!r}")

        if transfer_mode == "bundle":
            # Cache bundle inside suite_dir/dist
            bundle_dir = suite_dir / "dist"
            bundle_dir.mkdir(parents=True, exist_ok=True)
            bundle_path = bundle_dir / "suite_bundle.tar.gz"

            def _bundle_progress(pct: float, msg: str) -> None:
                # Map bundling into 5%..25%
                try:
                    progress(0.05 + 0.20 * float(pct), msg)
                except Exception:
                    pass
                log(f"[package] {msg}")

            log("Packaging suite (bundle)")
            if force_bundle_rebuild and bool(args.reuse_bundle):
                log('[package] Suite files were refreshed locally; the shared bundle manifest will decide whether a rebuild is actually required.')
            bundle_includes, bundle_extra_excludes = remote_minimal_bundle_patterns()
            # v60q: validation transport is exact-root, never
            # ``resources/validation/**``.  The latter pulled the complete
            # 50,000-image ImageNet tree into a 16-item Smoke bundle.
            bundle_includes = [p for p in bundle_includes if str(p) != "resources/validation/**"]

            def _validation_source_is_suite_local(src: object) -> bool:
                raw = str(src or "").strip()
                if not raw:
                    return False
                try:
                    candidate = Path(raw).expanduser()
                    if not candidate.is_absolute():
                        candidate = suite_dir / candidate
                    candidate = candidate.resolve()
                    sd = suite_dir.resolve()
                    return candidate == sd or sd in candidate.parents
                except Exception:
                    return False

            validation_is_suite_local = bool(
                validation_images_norm and _validation_source_is_suite_local(validation_images_norm)
            )
            validation_embed_root: Optional[Path] = None
            if validation_is_suite_local:
                src = Path(str(validation_images_norm)).expanduser()
                if not src.is_absolute():
                    src = suite_dir / src
                src = src.resolve()
                validation_embed_root = src.parent if src.is_file() else src
                rel_root = validation_embed_root.relative_to(suite_dir.resolve()).as_posix().strip("/")
                budget = max(0, int(validation_max_images_norm or 0))
                if budget > 0 and rel_root in {"", ".", "resources", "resources/validation"}:
                    raise RuntimeError(
                        "Refusing a broad suite-local validation root for a bounded run: "
                        f"root={validation_embed_root}, budget={budget}. Regenerate the suite with v60q."
                    )
                exact_pattern = f"{rel_root}/**" if rel_root else "**"
                if exact_pattern not in bundle_includes:
                    bundle_includes.append(exact_pattern)
                log(f"[package] Embedding only selected validation root: {rel_root or '.'}")
            elif validation_images_norm:
                log(
                    "[package] External validation source configured; no suite-local validation tree is embedded: "
                    f"{validation_images_norm}"
                )
            elif validation_use_embedded:
                # Compatibility fallback for old detection suites.  Current suites
                # are normalised above and normally provide an exact effective path.
                fallback_candidates = [
                    suite_dir / "resources" / "validation" / "detection" / "coco_50_data",
                    suite_dir / "resources" / "validation" / "coco_50_data",
                ]
                for fallback_root in fallback_candidates:
                    if fallback_root.is_dir():
                        validation_embed_root = fallback_root.resolve()
                        validation_is_suite_local = True
                        rel_root = validation_embed_root.relative_to(suite_dir.resolve()).as_posix()
                        bundle_includes.append(f"{rel_root}/**")
                        validation_images_norm = rel_root
                        log(f"[package] Embedded compatibility validation root: {rel_root}")
                        break
                if not validation_is_suite_local:
                    log("[package][warn] Embedded validation was requested, but no prepared suite-local root was found.")

            # v60q: a bounded Smoke/Standard run must transport only its
            # effective subset.  Detect an accidentally copied full dataset
            # before the expensive shared bundle scan begins.
            validation_footprint: dict[str, object] = {}
            if validation_is_suite_local and validation_embed_root is not None:
                try:
                    src = Path(str(validation_images_norm)).expanduser()
                    if not src.is_absolute():
                        src = (suite_dir / src).resolve()
                    root = validation_embed_root
                    file_count = 0
                    total_validation_bytes = 0
                    largest: list[tuple[int, str]] = []
                    if root.exists():
                        for vp in root.rglob("*"):
                            if not vp.is_file():
                                continue
                            try:
                                size = int(vp.stat().st_size)
                            except OSError:
                                size = 0
                            file_count += 1
                            total_validation_bytes += size
                            largest.append((size, str(vp.relative_to(root))))
                    largest = sorted(largest, reverse=True)[:10]
                    budget = max(0, int(validation_max_images_norm or 0))
                    max_files = max(100, budget * 4 + 50) if budget > 0 else 0
                    max_bytes = max(256 * 1024 * 1024, budget * 8 * 1024 * 1024) if budget > 0 else 0
                    validation_footprint = {
                        "source": str(src),
                        "root": str(root),
                        "budget_items": int(budget),
                        "file_count": int(file_count),
                        "total_bytes": int(total_validation_bytes),
                        "largest_files": [{"path": rel, "size": size} for size, rel in largest],
                        "max_files_guard": int(max_files),
                        "max_bytes_guard": int(max_bytes),
                    }
                    log(
                        f"[package] Validation footprint: items<= {budget or 'all'}, "
                        f"files={file_count}, size={total_validation_bytes / (1024*1024):.1f} MiB"
                    )
                    if budget > 0 and (file_count > max_files or total_validation_bytes > max_bytes):
                        raise RuntimeError(
                            "Suite-local validation payload is inconsistent with the run-mode budget: "
                            f"budget={budget}, files={file_count} (limit {max_files}), "
                            f"size={total_validation_bytes / (1024*1024):.1f} MiB "
                            f"(limit {max_bytes / (1024*1024):.1f} MiB), root={root}. "
                            "Regenerate the benchmark suite with the current tool so only the effective subset is materialised."
                        )
                except RuntimeError:
                    raise
                except Exception as exc:
                    log(f"[package][warn] Could not inspect validation footprint: {type(exc).__name__}: {exc}")
            stats = build_suite_bundle(
                suite_dir=suite_dir,
                out_path=bundle_path,
                includes=bundle_includes,
                excludes=bundle_extra_excludes,
                progress_cb=_bundle_progress,
                should_cancel=(lambda: bool(cancel_event and cancel_event.is_set())),
                reuse_if_unchanged=bool(args.reuse_bundle),
            )

            raw_mb = stats.total_bytes / (1024 * 1024) if stats.total_bytes else 0.0
            tgz_mb = stats.bundle_path.stat().st_size / (1024 * 1024)
            cache_tag = "cached" if getattr(stats, "reused", False) else "rebuilt"
            manifest_path = bundle_path.with_name(bundle_path.name + ".manifest.json")
            manifest_payload: dict[str, Any] = {}
            try:
                if manifest_path.exists():
                    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                manifest_payload = {}
            bundle_stats_payload = {
                "schema": "onnx-splitpoint/remote-suite-bundle-status",
                "schema_version": 1,
                "created_at": _utc_now_iso(),
                "suite_dir": str(suite_dir),
                "bundle_path": str(stats.bundle_path),
                "manifest_path": str(manifest_path) if manifest_path.exists() else "",
                "file_count": int(getattr(stats, "files", 0) or 0),
                "raw_total_bytes": int(getattr(stats, "total_bytes", 0) or 0),
                "tar_gz_bytes": int(stats.bundle_path.stat().st_size) if stats.bundle_path.exists() else 0,
                "raw_total_mb": round(raw_mb, 3),
                "tar_gz_mb": round(tgz_mb, 3),
                "reused": bool(getattr(stats, "reused", False)),
                "validation_assets_embedded": bool(validation_is_suite_local),
                "validation_images": validation_images_norm or "",
                "validation_footprint": validation_footprint,
                "minimal_includes": list(bundle_includes),
                "minimal_excludes": list(bundle_extra_excludes),
                "size_by_suffix": manifest_payload.get("size_by_suffix", {}),
                "largest_files": manifest_payload.get("largest_files", []),
                "slimming_policy": manifest_payload.get("slimming_policy", ""),
                "bundle_sha256": str(getattr(stats, "sha256", "") or manifest_payload.get("bundle_sha256") or ""),
                "cache_policy": manifest_payload.get("cache_policy", ""),
                "integrity_mode": manifest_payload.get("integrity_mode", ""),
                "per_file_hashes": bool(manifest_payload.get("per_file_hashes", False)),
                "gzip_level": manifest_payload.get("gzip_level"),
                "deterministic_archive": bool(manifest_payload.get("deterministic_archive", False)),
                "archive_metadata_policy": manifest_payload.get("archive_metadata_policy", ""),
            }
            try:
                _write_json(local_run_dir / "suite_bundle_status.json", bundle_stats_payload)
            except Exception:
                pass
            log(
                f"[remote] bundle: {stats.bundle_path} (raw {raw_mb:.1f} MB, tar.gz {tgz_mb:.1f} MB, files={stats.files}, {cache_tag})"
            )

            progress(0.25, "Bundle ready")

            # v60o: content-addressed remote suite cache.  The same model suite is
            # used by several run IDs on one physical setup.  Uploading 700+ MiB
            # for every Full/Split/reference row dominated earlier night runs.
            bundle_hash = str(getattr(stats, "sha256", "") or "").strip() or _streaming_sha256(stats.bundle_path)
            # The suite is transport staging, not a compiler cache.  Its raw
            # contracts carry timestamps and EvaluationRun paths, so retaining
            # both the archive and a pristine extraction accumulated about
            # four GiB per three-model run and client without useful reuse.
            # Preserve detection of the historical switch only to emit a clear
            # warning.  Bounded per-run upload is mandatory in this release;
            # the separate TensorRT cache remains persistent.
            cache_requested = str(
                os.environ.get("ONNX_SPLITPOINT_REMOTE_BUNDLE_CACHE", "0") or "0"
            ).strip().lower() not in {"0", "false", "off", "no"}
            # v2.75.16 deliberately removes the opt-in escape hatch: the
            # legacy cache has no ownership/retention contract and its bundle
            # identity still contains scientific run-local metadata.  Keep the
            # implementation only for forensic compatibility, but make it
            # unreachable until a marker/receipt/lock-safe retention design is
            # introduced.
            cache_enabled = False
            if cache_requested:
                log(
                    "[remote][cache][warn] ONNX_SPLITPOINT_REMOTE_BUNDLE_CACHE "
                    "is ignored in v2.75.16; bounded per-run transport is mandatory"
                )
            cache_paths = _remote_suite_cache_paths(remote_base, suite_cache_key, bundle_hash)
            remote_cache_dir = cache_paths["root"]
            remote_cached_bundle = cache_paths["bundle"]
            remote_cached_suite = cache_paths["suite"]
            remote_bundle_ready = cache_paths["bundle_ready"]
            remote_suite_ready = cache_paths["suite_ready"]
            remote_bundle_cache_hit = False
            remote_suite_cache_hit = False

            if not cache_enabled:
                actual_transport_bytes = (
                    int(stats.bundle_path.stat().st_size)
                    + int(getattr(stats, "total_bytes", 0) or 0)
                )
                runtime_cold_bytes = (
                    int(capacity_requirement["cold_required_free_bytes"])
                    - int(capacity_requirement["transport_peak_bytes"])
                )
                required_bytes = max(
                    int(capacity_requirement["cold_required_free_bytes"]),
                    actual_transport_bytes + runtime_cold_bytes,
                )
                _remote_storage_preflight(
                    transport,
                    remote_base,
                    required_free_bytes=required_bytes,
                    required_free_inodes=int(capacity_requirement["cold_required_free_inodes"]),
                    stage="before_uncached_suite_upload",
                )
                progress(0.27, "Uploading suite bundle")
                remote_bundle = f"{remote_run_dir}/suite_bundle.tar.gz"
                scp_upload_checked(stats.bundle_path, remote_bundle, stage="scp_upload_suite")
                remote_suite_dir = f"{remote_run_dir}/suite"
                run_checked(
                    _verified_uncached_suite_extract_command(
                        remote_bundle=remote_bundle,
                        remote_suite_dir=remote_suite_dir,
                        bundle_hash=bundle_hash,
                    ),
                    stage="remote_verify_and_untar_suite",
                )
                progress(0.40, "Suite uploaded")
                bundle_stats_payload.update({
                    "bundle_sha256": bundle_hash,
                    "remote_cache_enabled": False,
                    "remote_bundle_cache_hit": False,
                    "remote_suite_cache_hit": False,
                })
                try:
                    _write_json(local_run_dir / "suite_bundle_status.json", bundle_stats_payload)
                except Exception:
                    pass
            else:
                cache_complete_test = (
                    f"test -s {shlex.quote(remote_cached_bundle)} "
                    f"-a -f {shlex.quote(remote_bundle_ready)} "
                    f"-a -d {shlex.quote(remote_cached_suite)} "
                    f"-a -f {shlex.quote(remote_suite_ready)}"
                )
                rc_complete, _ = transport.run(cache_complete_test, timeout=20)
                cache_was_complete = rc_complete == 0
                raw_suite_bytes = int(getattr(stats, "total_bytes", 0) or 0)
                bundle_bytes = int(stats.bundle_path.stat().st_size)
                if cache_was_complete:
                    # Assume no reflink support: one full writable work clone.
                    required_bytes = raw_suite_bytes + _remote_storage_reserve_bytes()
                    required_inodes = int(getattr(stats, "files", 0) or 0) + 1024
                    capacity_stage = "before_warm_suite_cache_clone"
                else:
                    # Cold population retains archive + pristine extraction and
                    # creates a complete work clone.  Reflinks are an optional
                    # optimisation and are never part of the capacity proof.
                    required_bytes = (
                        bundle_bytes
                        + 2 * raw_suite_bytes
                        + _remote_storage_reserve_bytes()
                    )
                    required_inodes = 2 * int(getattr(stats, "files", 0) or 0) + 2048
                    capacity_stage = "before_cold_suite_cache_population"
                storage_capacity = _remote_storage_preflight(
                    transport,
                    remote_base,
                    required_free_bytes=required_bytes,
                    required_free_inodes=required_inodes,
                    stage=capacity_stage,
                )
                log(
                    "[remote][storage] capacity contract ok: "
                    f"stage={capacity_stage}, required={required_bytes / (1024 * 1024):.1f} MiB, "
                    f"free={int(storage_capacity.get('free_bytes') or 0) / (1024 * 1024):.1f} MiB"
                )
                run_checked(
                    f"mkdir -p {shlex.quote(remote_cache_dir)}",
                    stage="remote_bundle_cache_mkdir",
                )

                # v60q: the process-wide upload semaphore alone is insufficient.
                # Every logical setup worker can observe a miss before entering
                # the semaphore and then upload the same archive sequentially.
                # Hold one atomic remote lock across upload, verification and
                # pristine extraction, and re-check markers after acquiring it.
                if not cache_was_complete:
                    with _remote_cache_population_guard(
                        transport=transport,
                        lock_dir=cache_paths["population_lock"],
                        ready_test_cmd=cache_complete_test,
                        log=log,
                        cancel_event=cancel_event,
                        timeout_s=(int(effective_outer_timeout_s) if effective_outer_timeout_s else 1800),
                    ):
                        rc_cache, _ = transport.run(
                            f"test -s {shlex.quote(remote_cached_bundle)} -a -f {shlex.quote(remote_bundle_ready)}",
                            timeout=20,
                        )
                        remote_bundle_cache_hit = rc_cache == 0
                        if remote_bundle_cache_hit:
                            log(f"[remote][cache] suite bundle hit after lock/recheck: {remote_cached_bundle}")
                        else:
                            progress(0.27, "Uploading suite bundle once")
                            remote_tmp_bundle = f"{remote_cache_dir}/suite_bundle.tar.gz.part-{os.getpid()}-{threading.get_ident()}"
                            log(f"[remote][cache] suite bundle miss; population owner uploading -> {remote_cached_bundle}")
                            scp_upload_checked(stats.bundle_path, remote_tmp_bundle, stage="scp_upload_suite_cache")
                            verify_cmd = (
                                f"actual=$(sha256sum {shlex.quote(remote_tmp_bundle)} | awk '{{print $1}}'); "
                                f'test "$actual" = {shlex.quote(bundle_hash)} && '
                                f"mv -f {shlex.quote(remote_tmp_bundle)} {shlex.quote(remote_cached_bundle)} && "
                                f"touch {shlex.quote(remote_bundle_ready)}"
                            )
                            run_checked(verify_cmd, stage="remote_bundle_cache_verify")

                        rc_suite_cache, _ = transport.run(
                            f"test -d {shlex.quote(remote_cached_suite)} -a -f {shlex.quote(remote_suite_ready)}",
                            timeout=20,
                        )
                        remote_suite_cache_hit = rc_suite_cache == 0
                        if not remote_suite_cache_hit:
                            tmp_suite = f"{remote_cache_dir}/suite.part-{os.getpid()}-{threading.get_ident()}"
                            build_cache_cmd = (
                                f"rm -rf {shlex.quote(tmp_suite)} && mkdir -p {shlex.quote(tmp_suite)} && "
                                f"tar -xzf {shlex.quote(remote_cached_bundle)} -C {shlex.quote(tmp_suite)} && "
                                f"rm -rf {shlex.quote(remote_cached_suite)} && "
                                f"mv {shlex.quote(tmp_suite)} {shlex.quote(remote_cached_suite)} && "
                                f"touch {shlex.quote(remote_suite_ready)}"
                            )
                            log("[remote][cache] population owner creating pristine extracted suite cache")
                            run_checked(build_cache_cmd, stage="remote_suite_cache_extract")
                        else:
                            log(f"[remote][cache] extracted suite hit after lock/recheck: {remote_cached_suite}")
                else:
                    remote_bundle_cache_hit = True
                    remote_suite_cache_hit = True
                    log(f"[remote][cache] complete suite cache hit: {remote_cached_suite}")

                # A waiter can leave the guard because a sibling completed the
                # cache without ever becoming owner.  Verify the final contract
                # before cloning so a failed owner cannot yield a half cache.
                rc_final_cache, final_cache_out = transport.run(cache_complete_test, timeout=20)
                if rc_final_cache != 0:
                    raise RuntimeError(
                        "Remote suite-cache population finished without complete ready markers: "
                        f"{remote_cache_dir}\n{final_cache_out}"
                    )

                remote_suite_dir = f"{remote_run_dir}/suite"
                log(f"[remote] suite_dir={remote_suite_dir}")
                progress(0.36, "Cloning cached suite on remote")
                clone_cmd = (
                    f"rm -rf {shlex.quote(remote_suite_dir)} && mkdir -p {shlex.quote(remote_suite_dir)} && "
                    f"(cp -a --reflink=auto {shlex.quote(remote_cached_suite)}/. {shlex.quote(remote_suite_dir)}/ "
                    f"2>/dev/null || cp -a {shlex.quote(remote_cached_suite)}/. {shlex.quote(remote_suite_dir)}/)"
                )
                run_checked(clone_cmd, stage="remote_suite_cache_clone")
                progress(0.40, "Suite ready from cache")

                bundle_stats_payload.update({
                    "bundle_sha256": bundle_hash,
                    "remote_cache_enabled": True,
                    "remote_bundle_cache_hit": remote_bundle_cache_hit,
                    "remote_suite_cache_hit": remote_suite_cache_hit,
                    "remote_cache_initially_complete": cache_was_complete,
                    "remote_cache_population_lock": cache_paths["population_lock"],
                    "remote_cached_bundle": remote_cached_bundle,
                    "remote_cached_suite": remote_cached_suite,
                })
                try:
                    _write_json(local_run_dir / "suite_bundle_status.json", bundle_stats_payload)
                except Exception:
                    pass

        else:
            # direct copy via scp -r
            direct_capacity = _remote_run_capacity_requirement_for_args(
                suite_dir,
                args=args,
            )
            _remote_storage_preflight(
                transport,
                remote_base,
                required_free_bytes=int(direct_capacity["required_free_bytes"]),
                required_free_inodes=int(direct_capacity["required_free_inodes"]),
                stage="before_direct_suite_upload",
            )
            log("Uploading suite (direct scp -r)")
            scp_upload_checked(suite_dir, remote_run_dir + "/", recursive=True)
            remote_suite_dir = f"{remote_run_dir}/{suite_dir.name}"
            log(f"[remote] suite_dir={remote_suite_dir}")
            progress(0.40, "Suite uploaded")

        if resume_requested and not remote_resume_state_found:
            local_resume_overlay = local_run_dir / "results"
            if local_resume_overlay.is_dir() and any(local_resume_overlay.iterdir()):
                log("[resume] Uploading local partial results overlay to remote suite")
                uploaded_overlay_items = 0
                for child in sorted(local_resume_overlay.iterdir()):
                    try:
                        scp_upload_checked(child, remote_suite_dir + "/", recursive=child.is_dir())
                        uploaded_overlay_items += 1
                    except Exception as e:
                        log(f"[warn] resume overlay upload failed for {child.name}: {e}")
                if uploaded_overlay_items:
                    log(f"[resume] Uploaded {uploaded_overlay_items} local result item(s) into {remote_suite_dir}")

        # ----------------------------
        # Run benchmark on remote
        # ----------------------------
        log("Running benchmark suite on remote")

        # NOTE: benchmark_suite.py (generated into the suite folder) does not accept
        # --set/--out/--repeats/--iters. Its CLI uses:
        #   --provider {auto,cpu,cuda,tensorrt}
        #   --plan <benchmark_plan.json>
        #   --warmup <N>
        #   --runs <N>
        # It also writes outputs into the suite folder. Therefore we:
        #   1) fold repeats into --runs (effective_runs = repeats * iters)
        #   2) run the suite script
        #   3) copy produced artifacts into <remote_results_dir> afterwards.

        provider = (args.provider or "auto").strip()
        warmup = int(args.warmup)
        repeats = max(1, int(args.repeats))
        iters = int(args.iters)
        effective_runs = max(1, repeats * iters)

        if repeats != 1:
            log(f"[ui] NOTE: repeats={repeats} folded into --runs (effective runs={effective_runs})")

        # Persist the effective execution counts so the local run metadata matches
        # what we actually ask benchmark_suite.py to execute remotely.
        try:
            run_meta["effective_args"] = {
                "provider": provider,
                "warmup": warmup,
                "repeats": repeats,
                "iters": iters,
                "effective_runs": effective_runs,
                "total_invocations_per_benchmark": warmup + effective_runs,
                "throughput_frames": int(getattr(args, "throughput_frames", 24) or 0),
                "throughput_warmup_frames": int(getattr(args, "throughput_warmup_frames", 6) or 0),
                "throughput_queue_depth": int(getattr(args, "throughput_queue_depth", 2) or 1),
                "phase_runs": int(getattr(args, "phase_runs", 5) or 0),
                "validation_images": validation_images_norm or "",
                "validation_max_images": int(validation_max_images_norm or 0),
                "validation_budget_authoritative": bool(validation_budget_authoritative),
                "validation_reference_mode": str(getattr(args, "validation_reference_mode", "auto") or "auto"),
                "mini_coco_ap50": bool(getattr(args, "mini_coco_ap50", False)),
                "benchmark_task": str(benchmark_task_norm),
                "mini_classification_eval": bool(mini_classification_eval_norm),
                "trt_build_guard": dict(getattr(args, "trt_build_guard", {}) or {}),
                "quality_evidence_eval_id": str(
                    getattr(args, "quality_evidence_eval_id", "") or ""
                ),
                "quality_evidence_model_id": str(
                    getattr(args, "quality_evidence_model_id", "") or ""
                ),
                "quality_evidence_setup_id": str(
                    getattr(args, "quality_evidence_setup_id", "") or ""
                ),
                "quality_evidence_endpoint_id": str(
                    getattr(args, "quality_evidence_endpoint_id", "") or ""
                ),
            }
            _write_json(local_run_dir / "run_meta.json", run_meta)
        except Exception:
            pass

        throughput_frames = max(0, int(getattr(args, "throughput_frames", 24) or 0))
        throughput_warmup_frames = max(0, int(getattr(args, "throughput_warmup_frames", 6) or 0))
        throughput_queue_depth = max(1, int(getattr(args, "throughput_queue_depth", 2) or 1))
        phase_runs = max(0, int(getattr(args, "phase_runs", 5) or 0))

        bench_cmd = (
            f'"$RUN_PY" -u benchmark_suite.py'
            f" --provider {shlex.quote(provider)}"
            f" --plan benchmark_plan.json"
            f" --warmup {warmup}"
            f" --runs {effective_runs}"
            f" --trt-cache-root {shlex.quote(remote_trt_cache_root)}"
            f" --throughput-frames {throughput_frames}"
            f" --throughput-warmup-frames {throughput_warmup_frames}"
            f" --throughput-queue-depth {throughput_queue_depth}"
            f" --phase-runs {phase_runs}"
        )
        # v52d: benchmark_task=auto must not force COCO-50 via CLI, because that
        # overrides per-run classification plans.  Only detection gets a CLI
        # COCO fallback; auto leaves validation_images empty so benchmark_suite.py
        # uses each run's benchmark_plan fields.
        effective_validation_images = validation_images_norm or ""
        effective_validation_max_images = int(validation_max_images_norm or 0)
        validation_reference_mode = str(getattr(args, "validation_reference_mode", "auto") or "auto").strip().lower() or "auto"
        if not validation_budget_authoritative:
            bench_cmd += (
                f" --validation-images {shlex.quote(str(effective_validation_images))}"
                f" --validation-max-images {max(0, effective_validation_max_images)}"
            )
        bench_cmd += (
            f" --validation-reference-mode {shlex.quote(validation_reference_mode)}"
            f" --benchmark-task {shlex.quote(str(benchmark_task_norm))}"
        )
        quality_eval_id = str(
            getattr(args, "quality_evidence_eval_id", "") or ""
        ).strip()
        quality_model_id = str(
            getattr(args, "quality_evidence_model_id", "") or ""
        ).strip()
        quality_setup_id = str(
            getattr(args, "quality_evidence_setup_id", "") or ""
        ).strip()
        quality_endpoint_id = str(
            getattr(args, "quality_evidence_endpoint_id", "") or ""
        ).strip()
        quality_identity_values = (
            quality_eval_id, quality_model_id, quality_setup_id,
        )
        if any(quality_identity_values) and not all(quality_identity_values):
            raise RuntimeError(
                "setup-local TensorRT quality identity is partial; eval, "
                "model and physical setup IDs are all required"
            )
        if all(quality_identity_values):
            bench_cmd += (
                f" --quality-evidence-eval-id {shlex.quote(quality_eval_id)}"
                f" --quality-evidence-model-id {shlex.quote(quality_model_id)}"
                f" --quality-evidence-setup-id {shlex.quote(quality_setup_id)}"
            )
        if quality_endpoint_id:
            if not all(quality_identity_values):
                raise RuntimeError(
                    "setup-local TensorRT quality endpoint identity requires "
                    "eval, model and physical setup IDs"
                )
            bench_cmd += (
                " --quality-evidence-endpoint-id "
                + shlex.quote(quality_endpoint_id)
            )
        if benchmark_task_norm == "detection" and bool(getattr(args, "mini_coco_ap50", False)):
            bench_cmd += " --mini-coco-ap50"
        if benchmark_task_norm == "classification" and bool(mini_classification_eval_norm):
            bench_cmd += " --mini-classification-eval"
        if bool(getattr(args, "resume", True)):
            bench_cmd += " --resume"
        if args.add_args:
            # Advanced args supported by benchmark_suite.py (raw passthrough).
            bench_cmd += f" {args.add_args}"

        case_progress_re = re.compile(r"^\[(?P<run_id>[^\]]+)\]\s+\[(?P<i>\d+)/(?P<n>\d+)\]\s+Running\s+(?P<case>b\d+)\b")

        def on_line(line: str) -> None:
            nonlocal last_suite_progress
            recent_remote_lines.append(str(line))
            if len(recent_remote_lines) > 200:
                del recent_remote_lines[:-200]
            log(line)
            sp = parse_benchmark_suite_progress(line)
            if sp is not None:
                last_suite_progress = {
                    "run_id": sp.run_id,
                    "index": int(sp.i),
                    "count": int(sp.n),
                    "pct": float(round(sp.pct, 6)),
                    "line": str(line),
                }
                m_case = case_progress_re.match(str(line).strip())
                if m_case is not None:
                    last_suite_progress["case_id"] = str(m_case.group("case"))
                # Map suite progress into 40%..90%
                progress(0.40 + 0.50 * sp.pct, f"Running {sp.i}/{sp.n}")
        # Make sure we also have remote stdout/stderr files for debugging.
        remote_venv_cmd = (args.remote_venv or "").strip()
        env_snippet = ""
        if remote_venv_cmd:
            # If user provided a full shell snippet (contains whitespace), use it as-is.
            # Otherwise treat it as a path to an activate script and source it.
            if any(ch.isspace() for ch in remote_venv_cmd):
                env_snippet = remote_venv_cmd
            else:
                # Keep ~ expansion working by translating to $HOME when possible.
                if remote_venv_cmd.startswith("~/"):
                    remote_venv_cmd = "$HOME/" + remote_venv_cmd[2:]
                env_snippet = f"source {remote_venv_cmd}"

        bench_inner_lines = [
            "set -e",
            f"cd {shlex.quote(remote_suite_dir)}",
            "mkdir -p logs",
            'SYS_PY="$(command -v python3)"',
        ]
        if env_snippet:
            bench_inner_lines.append(env_snippet)

        bench_inner_lines += [
            _remote_trt_build_guard_shell(args),
            'ENV_PY="$(command -v python3)"',
            f"WANT_HAILO={'1' if want_hailo else '0'}",
            f"WANT_CUDA={'1' if want_cuda else '0'}",
            f"WANT_TRT={'1' if want_tensorrt else '0'}",
            f"WANT_DEEPX={'1' if want_deepx else '0'}",
            f"REQUIRE_ORT={'1' if require_ort else '0'}",
            *DEEPX_DISCOVERY_SHELL.splitlines(),
            *_remote_hailo_auto_env_shell(log_prefix="remote"),
            'RUN_PY="$SYS_PY"',
            'ENV_SITE=""',
            'SYS_EPS=""',
            'ENV_EPS=""',
            'SYS_CORE=""',
            'ENV_CORE=""',
            'SYS_DEEPX=""',
            'ENV_DEEPX=""',
            'has_cuda() { case "$1" in *CUDAExecutionProvider*) return 0;; *) return 1;; esac; }',
            'has_trt() { case "$1" in *TensorrtExecutionProvider*) return 0;; *) return 1;; esac; }',
            'SYS_CORE=$("$SYS_PY" -c "import onnx,onnxruntime; print(\\"ok\\")" 2>/dev/null || true)',
            'SYS_EPS=$("$SYS_PY" -c \'import onnxruntime as ort; print("|".join(ort.get_available_providers()))\' 2>/dev/null || true)',
            'SYS_DEEPX=$("$SYS_PY" -c "import dx_engine; print(\\"ok\\")" 2>/dev/null || true)',
            'if [ -n "$ENV_PY" ] && [ -x "$ENV_PY" ]; then',
            '  ENV_SITE=$("$ENV_PY" -c \'import site,os; ps=[]; getsp=getattr(site,"getsitepackages",None); ps.extend(getsp() if getsp else []); usp=getattr(site,"getusersitepackages",lambda:None)(); ps.append(usp); ps=[p for p in ps if p and os.path.isdir(p)]; out=[]; [out.append(p) for p in ps if p not in out]; print(":".join(out))\' 2>/dev/null || true)',
            '  ENV_CORE=$("$ENV_PY" -c "import onnx,onnxruntime; print(\\"ok\\")" 2>/dev/null || true)',
            '  ENV_EPS=$("$ENV_PY" -c \'import onnxruntime as ort; print("|".join(ort.get_available_providers()))\' 2>/dev/null || true)',
            '  ENV_DEEPX=$("$ENV_PY" -c "import dx_engine; print(\\"ok\\")" 2>/dev/null || true)',
            '  if [ "$WANT_TRT" = "1" ]; then',
            '    if has_cuda "$ENV_EPS" && has_trt "$ENV_EPS"; then RUN_PY="$ENV_PY"; elif has_cuda "$SYS_EPS" && has_trt "$SYS_EPS"; then RUN_PY="$SYS_PY"; fi',
            '  elif [ "$WANT_CUDA" = "1" ]; then',
            '    if has_cuda "$ENV_EPS"; then RUN_PY="$ENV_PY"; elif has_cuda "$SYS_EPS"; then RUN_PY="$SYS_PY"; fi',
            '  elif [ "$REQUIRE_ORT" = "1" ]; then',
            '    if [ -n "$ENV_CORE" ]; then RUN_PY="$ENV_PY"; elif [ -n "$SYS_CORE" ]; then RUN_PY="$SYS_PY"; fi',
            '  elif [ "$WANT_DEEPX" = "1" ]; then',
            '    if [ -n "$ENV_DEEPX" ]; then RUN_PY="$ENV_PY"; elif [ -n "$SYS_DEEPX" ]; then RUN_PY="$SYS_PY"; fi',
            '  else',
            '    if [ -n "$ENV_CORE" ]; then RUN_PY="$ENV_PY"; fi',
            '  fi',
            'fi',
            'if [ "$WANT_HAILO" = "1" ] && [ -n "$ENV_SITE" ]; then export SPLITPOINT_EXTRA_SITES="$ENV_SITE"; fi',
            'if [ "$WANT_DEEPX" = "1" ] && [ -n "$ENV_SITE" ]; then export SPLITPOINT_EXTRA_SITES="$ENV_SITE:${SPLITPOINT_EXTRA_SITES:-}"; fi',
            'echo "[remote] python: SYS_PY=${SYS_PY} ENV_PY=${ENV_PY} RUN_PY=${RUN_PY} | SYS_EPS=${SYS_EPS} | ENV_EPS=${ENV_EPS} | HAILO_PY=${HAILO_PY:-} | DEEPX_PY=${DEEPX_PY:-} | EXTRA_SITES=${SPLITPOINT_EXTRA_SITES:-}" >&2',
            f"{bench_cmd} 1> >(tee logs/stdout.txt) 2> >(tee logs/stderr.txt >&2)",
        ]
        # Insert v51 Hailo Python discovery immediately before the remote echo and command.
        if len(bench_inner_lines) >= 2:
            _bench_cmd_line = bench_inner_lines.pop()
            _bench_echo_line = bench_inner_lines.pop()
            bench_inner_lines.extend(HAILO_DISCOVERY_SHELL.splitlines())
            bench_inner_lines.append(_bench_echo_line)
            bench_inner_lines.append(_bench_cmd_line)
        bench_inner = "\n".join(bench_inner_lines)
        trt_active_lock = f"{remote_trt_cache_root}/.active.lock"

        def _with_trt_active_lock(command: str) -> str:
            if not trt_cache_active:
                return command
            return (
                f"flock -x {shlex.quote(trt_active_lock)} -c "
                + shlex.quote(command)
            )

        bench_remote_cmd = _with_trt_active_lock(
            "bash -lc " + shlex.quote(bench_inner)
        )

        def _bench_remote_cmd_for(extra_args: str = "") -> str:
            _cmd = str(bench_cmd)
            if extra_args:
                _cmd = _cmd + " " + str(extra_args).strip()
            _lines = list(bench_inner_lines)
            for _i in range(len(_lines) - 1, -1, -1):
                if "benchmark_suite.py" in _lines[_i]:
                    _lines[_i] = f"{_cmd} 1> >(tee logs/stdout.txt) 2> >(tee logs/stderr.txt >&2)"
                    break
            return _with_trt_active_lock(
                "bash -lc " + shlex.quote("\n".join(_lines))
            )

        outer_timeout_s = None
        try:
            if effective_outer_timeout_s is not None and int(effective_outer_timeout_s) > 0:
                outer_timeout_s = int(effective_outer_timeout_s)
        except Exception:
            outer_timeout_s = 7200

        energy_enabled = bool(getattr(args, "energy_enabled", False))
        energy_result_payload = None
        if energy_enabled and str(getattr(args, "energy_scope", "") or "").strip().lower() == "row_variant":
            try:
                log("[energy] row_variant mode: running primary remote benchmark before u.RECS target windows")
                remote_rc = transport.run_streaming(
                    bench_remote_cmd,
                    timeout=outer_timeout_s,
                    on_line=on_line,
                    cancel_event=cancel_event,
                )
                if remote_rc != 0:
                    log(f"[energy][abort] primary remote benchmark rc={remote_rc}; skipping row_variant energy windows")
                else:
                    from onnx_splitpoint_tool.energy.config import (
                        energy_defaults_from_registry,
                        energy_measurements_root,
                        energy_setup_from_registry,
                        load_hardware_registry,
                    )
                    from onnx_splitpoint_tool.energy.collector import (
                        run_fast_firmware_measurement,
                        run_duration_probe,
                        select_host_normalization_role,
                    )

                    assert energy_registry_preflight is not None
                    assert energy_setup_preflight is not None
                    registry = energy_registry_preflight
                    energy_defaults = _bind_energy_ab_runtime_args(
                        energy_defaults_from_registry(registry),
                        args,
                    )
                    try:
                        energy_defaults.heartbeat_s = max(10, int(getattr(args, "energy_heartbeat_s", 60) or 60))
                    except Exception:
                        pass
                    energy_setup_id = energy_setup_id_preflight
                    energy_setup = energy_setup_preflight
                    if not energy_setup.enabled or not str(energy_setup.urecs_address or "").strip():
                        log(f"[energy][skip] setup {energy_setup_id!r} is not enabled or has no u.RECS address")
                    else:
                        energy_root_base = str(getattr(args, "energy_output_root", "") or "").strip()
                        if energy_root_base:
                            energy_base = Path(energy_root_base).expanduser().resolve()
                        else:
                            energy_base = energy_measurements_root(local_working_dir) / "Benchmarks" / suite_dir.name
                        current_run_id = str(_extract_run_id_from_add_args(getattr(args, 'add_args', '')) or 'all')
                        energy_run_id = _safe_local_name(f"{run_id}_{energy_setup_id}_{current_run_id}")
                        energy_out_dir = energy_base / energy_run_id
                        energy_policy = str(getattr(args, "energy_target_policy", "") or "canonical_only").strip().lower()
                        targets_raw = _energy_targets_for_run_id(run_id=current_run_id, benchmark_set_json=benchmark_set_json, target_policy=energy_policy)
                        targets_ranked, target_rank_meta = _energy_rank_targets_by_results(local_run_dir, run_id=str(current_run_id), targets=targets_raw, policy=energy_policy)
                        targets, target_cap_meta = _energy_apply_target_cap(targets_ranked, args=args, current_run_id=current_run_id)
                        target_cap_meta.update(target_rank_meta)
                        randomize_targets = bool(getattr(args, "energy_randomize_target_order", False))
                        randomization_seed = int(getattr(args, "energy_randomization_seed", 20260710) or 20260710)
                        if randomize_targets and len(targets) > 1:
                            # Use a stable per-run seed; Python's hash() is
                            # deliberately process-randomised and is unsuitable
                            # for a reproducible measurement campaign.
                            suffix = int(hashlib.sha256(str(current_run_id).encode("utf-8")).hexdigest()[:8], 16)
                            random.Random(randomization_seed + suffix).shuffle(targets)
                        target_order_payload = {
                            "schema": "onnx-splitpoint/energy-target-order",
                            "schema_version": 1,
                            "created_at": _utc_now_iso(),
                            "run_id": str(current_run_id),
                            "mode": "deterministic_block_randomized" if randomize_targets else "fixed",
                            "seed": randomization_seed if randomize_targets else None,
                            "note": "Target blocks are randomized before acquisition; repeats inside a target remain contiguous in one raw-trace bundle.",
                            "targets": [
                                {
                                    "order": i + 1,
                                    "case": target.get("energy_target_case"),
                                    "variant": target.get("energy_target_variant"),
                                }
                                for i, target in enumerate(targets)
                            ],
                        }
                        _write_json(energy_out_dir / "energy_target_order.json", target_order_payload)
                        target_results: list[dict[str, Any]] = []
                        cap_note = ""
                        if target_cap_meta.get("truncated"):
                            cap_note = f"; capped {target_cap_meta.get('requested_target_count')} -> {target_cap_meta.get('kept_target_count')} (max_targets_per_run_id={target_cap_meta.get('max_targets_per_run_id')})"
                        if target_cap_meta.get("ranked"):
                            cap_note += f"; ranked valid_targets={target_cap_meta.get('valid_target_count')} policy={energy_policy}"
                        log(f"[energy] row_variant targets={len(targets)} for run_id={current_run_id}{cap_note}: " + ", ".join(f"{t.get('energy_target_case')}/{t.get('energy_target_variant')}" for t in targets))
                        energy_phases = [str(x).strip().lower() for x in (getattr(args, "energy_phases", None) or []) if str(x).strip()] or ["latency", "streaming"]
                        energy_repeats = max(1, int(getattr(args, "energy_run_count", 1) or 1))
                        for target in targets:
                            tcase = str(target.get("energy_target_case") or "").strip()
                            tvariant = str(target.get("energy_target_variant") or "composed").strip().lower()
                            host_normalization_role = select_host_normalization_role(
                                run_id=current_run_id,
                                target_variant=tvariant,
                            )
                            tname = _safe_local_name(f"{energy_setup_id}_{current_run_id}_{tcase}_{tvariant}")
                            tout = energy_out_dir / "targets" / tname
                            tout.mkdir(parents=True, exist_ok=True)
                            phase_payloads: list[dict[str, Any]] = []
                            phase_ok: list[bool] = []
                            # v58j: size Eval u.RECS windows from the concrete
                            # target result row, not from the short Eval loop
                            # count. This makes Eval Energy use the same
                            # Min-active policy as manual Benchmark Energy.
                            target_row = _energy_find_target_row(local_run_dir, run_id=str(current_run_id), target_case=tcase, target_variant=tvariant)
                            sizing = _energy_target_sizing_from_row(target_row, target_variant=tvariant)
                            try:
                                min_active_s = max(0.0, float(getattr(energy_defaults, "min_active_duration_s", 30.0) or 0.0))
                            except Exception:
                                min_active_s = 30.0
                            log(f"[energy] target case={tcase} variant={tvariant} phases={','.join(energy_phases)} out={tout} sizing_latency_ms={sizing.get('latency_ms')} sizing_fps={sizing.get('reference_fps')} source={sizing.get('reference_fps_source')}")
                            for phase in energy_phases:
                                # v58h: Eval energy uses the same row/variant phase split as the
                                # manual benchmark path.  Each u.RECS window measures either the
                                # latency loop or the streaming/throughput loop for exactly one
                                # case+variant target.  We intentionally append duplicate CLI
                                # options at the end; argparse keeps the last value, so these
                                # phase-specific values override the primary benchmark command.
                                if phase == "latency":
                                    phase_work_units = max(1, int(effective_runs))
                                    phase_extra_tokens = [
                                        "--runs", str(phase_work_units),
                                        "--throughput-frames", "0",
                                        "--throughput-warmup-frames", "0",
                                    ]
                                elif phase == "streaming":
                                    if tvariant == "full":
                                        phase_work_units = max(1, int(throughput_frames or effective_runs))
                                        phase_extra_tokens = [
                                            "--runs", str(phase_work_units),
                                            "--throughput-frames", "0",
                                            "--throughput-warmup-frames", "0",
                                        ]
                                    else:
                                        phase_work_units = max(1, int(throughput_frames or effective_runs))
                                        phase_extra_tokens = [
                                            "--runs", "1",
                                            "--throughput-frames", str(phase_work_units),
                                            "--throughput-warmup-frames", str(max(0, int(throughput_warmup_frames or 0))),
                                            "--throughput-queue-depth", str(max(1, int(throughput_queue_depth or 1))),
                                        ]
                                else:
                                    log(f"[energy] skipping unsupported phase={phase!r} for target {tcase}/{tvariant}")
                                    continue

                                # v58j: auto-scale Eval energy windows to the configured
                                # minimum active duration.  The displayed Eval runs (e.g. 3)
                                # remain the normal benchmark loop count; u.RECS windows get
                                # more work units only to obtain stable power/energy samples.
                                original_phase_work_units = int(phase_work_units)
                                scale_reason = ""
                                try:
                                    if min_active_s > 0 and phase == "streaming" and sizing.get("reference_fps"):
                                        phase_work_units = max(phase_work_units, int(math.ceil(min_active_s * float(sizing.get("reference_fps")))))
                                        scale_reason = f"target_reference_fps={float(sizing.get('reference_fps')):.3f}; source={sizing.get('reference_fps_source')}"
                                    elif min_active_s > 0 and sizing.get("latency_ms"):
                                        phase_work_units = max(phase_work_units, int(math.ceil((min_active_s * 1000.0) / float(sizing.get("latency_ms")))))
                                        scale_reason = f"target_latency_ms={float(sizing.get('latency_ms')):.3f}"
                                except Exception as _scale_exc:
                                    scale_reason = f"scaling_failed:{type(_scale_exc).__name__}"
                                estimated_preprobe_work_units = int(phase_work_units)
                                if phase_work_units > 50000:
                                    scale_reason += f"; capped_from={phase_work_units}"
                                    phase_work_units = 50000
                                # v58o: Do not run the expensive sizing probe with a huge
                                # analytically estimated window.  A wrong FPS estimate for YOLO
                                # could otherwise turn the probe itself into a 10+ minute run.
                                # Use a bounded pilot window first, then scale from the measured
                                # pilot duration.
                                pilot_cap = int(getattr(args, "energy_sizing_probe_max_work_units", 256) or 256)
                                if min_active_s > 0 and int(phase_work_units) > pilot_cap:
                                    scale_reason += f"; pilot_capped_from={phase_work_units}"
                                    phase_work_units = max(original_phase_work_units, pilot_cap)
                                if int(phase_work_units) != original_phase_work_units:
                                    log(f"[energy] target case={tcase} variant={tvariant} phase={phase}: sizing pilot work units {original_phase_work_units} -> {phase_work_units} for min_active={min_active_s:.1f}s (estimated_final_hint={estimated_preprobe_work_units}; {scale_reason})")
                                else:
                                    log(f"[energy] target case={tcase} variant={tvariant} phase={phase}: sizing pilot work units {phase_work_units}; min_active={min_active_s:.1f}s ({scale_reason or 'no target estimate'})")
                                # Update the CLI tokens after scaling.
                                if phase == "latency":
                                    phase_extra_tokens = ["--runs", str(int(phase_work_units)), "--throughput-frames", "0", "--throughput-warmup-frames", "0"]
                                elif phase == "streaming":
                                    if tvariant == "full":
                                        phase_extra_tokens = ["--runs", str(int(phase_work_units)), "--throughput-frames", "0", "--throughput-warmup-frames", "0"]
                                    else:
                                        phase_extra_tokens = ["--runs", "1", "--throughput-frames", str(int(phase_work_units)), "--throughput-warmup-frames", str(max(0, int(throughput_warmup_frames or 0))), "--throughput-queue-depth", str(max(1, int(throughput_queue_depth or 1)))]
                                phase_extra_tokens.extend([
                                    "--energy-measurement-only",
                                    "--energy-target-case", tcase,
                                    "--energy-target-variant", tvariant,
                                ])
                                def _replace_token_value(tokens: list[str], opt: str, value: int) -> None:
                                    try:
                                        idx = tokens.index(opt)
                                        if idx + 1 < len(tokens):
                                            tokens[idx + 1] = str(int(value))
                                    except ValueError:
                                        pass

                                phase_out = tout / phase
                                # v58j: Evaluation energy used to pass the raw GUI/profile counts
                                # (often runs=3, streaming=24) directly into u.RECS windows.  That is
                                # useful for smoke tests but too short for stable power measurements.
                                # The manual benchmark path already probes/scales windows.  Mirror that
                                # behaviour here: run a cheap duration probe for the exact target command,
                                # then scale work units to min_active_duration_s and measure the scaled
                                # command.  The final u.RECS measurement still performs its own probe so the
                                # collector duration matches the scaled command.
                                initial_phase_work_units = int(phase_work_units)
                                energy_scale_meta: dict[str, Any] = {
                                    "requested_work_units_per_window": int(initial_phase_work_units),
                                    "energy_auto_scale_policy": "duration_probe_to_min_active_s",
                                    "min_active_duration_s": getattr(energy_defaults, "min_active_duration_s", None),
                                }

                                extra = " ".join(shlex.quote(str(x)) for x in phase_extra_tokens)
                                target_cmd = _bench_remote_cmd_for(extra)
                                known_duration = getattr(args, "energy_known_duration_s", None)
                                try:
                                    min_active = float(getattr(energy_defaults, "min_active_duration_s", 0.0) or 0.0)
                                except Exception:
                                    min_active = 0.0
                                if known_duration in (None, "", 0, 0.0) and min_active > 0 and int(phase_work_units) > 0:
                                    try:
                                        probe_dir = phase_out / "sizing_probe"
                                        log(f"[energy] sizing probe target case={tcase} variant={tvariant} phase={phase} work_units={phase_work_units} min_active={min_active:g}s")
                                        try:
                                            probe_timeout_s = max(120.0, float(min_active or 30.0) * 4.0 + 60.0)
                                            if outer_timeout_s:
                                                probe_timeout_s = min(float(outer_timeout_s) + 300.0, probe_timeout_s)
                                        except Exception:
                                            probe_timeout_s = 180.0
                                        energy_command, energy_lease_env = _journaled_energy_ssh_command(
                                            transport,
                                            remote_process_registry,
                                            target_cmd,
                                            label=f"energy-sizing-{tcase}-{tvariant}-{phase}",
                                            timeout_s=probe_timeout_s,
                                        )
                                        probe = run_duration_probe(
                                            energy_command,
                                            probe_dir,
                                            timeout_s=probe_timeout_s,
                                            margin_s=0.0,
                                            cancel_event=cancel_event,
                                            subprocess_env=energy_lease_env,
                                        )
                                        energy_scale_meta["sizing_probe_timeout_s"] = probe_timeout_s
                                        energy_scale_meta["sizing_probe"] = {k: probe.get(k) for k in ("ok", "probe_rc", "probe_duration_s", "collector_duration_s", "power_estimated_duration_s")}
                                        pdur = float(probe.get("probe_duration_s") or 0.0)
                                        if probe.get("ok") and pdur > 0 and pdur < min_active:
                                            import math as _energy_math
                                            scale = max(1, int(_energy_math.ceil(min_active / max(pdur, 1e-6))))
                                            # Avoid accidental million-frame windows when a broken command returns too fast.
                                            scale = min(scale, 200)
                                            new_units = max(int(phase_work_units), int(phase_work_units) * scale)
                                            phase_work_units = int(new_units)
                                            if phase == "latency" or (phase == "streaming" and tvariant == "full"):
                                                _replace_token_value(phase_extra_tokens, "--runs", phase_work_units)
                                            elif phase == "streaming":
                                                _replace_token_value(phase_extra_tokens, "--throughput-frames", phase_work_units)
                                            extra = " ".join(shlex.quote(str(x)) for x in phase_extra_tokens)
                                            target_cmd = _bench_remote_cmd_for(extra)
                                            energy_scale_meta.update({
                                                "auto_scaled": True,
                                                "scale_factor": scale,
                                                "scaled_work_units_per_window": int(phase_work_units),
                                                "sizing_reason": "probe_duration_below_min_active",
                                            })
                                            log(f"[energy] auto-scaled target case={tcase} variant={tvariant} phase={phase}: {initial_phase_work_units} -> {phase_work_units} work units (probe={pdur:.3f}s, min_active={min_active:g}s)")
                                            # v58r: one bounded pilot can under-size very fast targets because
                                            # fixed session/command overhead dominates the pilot. Verify the
                                            # scaled command once more and rescale if the measured workload is
                                            # still far below the requested min-active duration.
                                            try:
                                                verify_dir = phase_out / "sizing_probe_verify"
                                                verify_timeout_s = max(120.0, float(min_active or 30.0) * 4.0 + 60.0)
                                                if outer_timeout_s:
                                                    verify_timeout_s = min(float(outer_timeout_s) + 300.0, verify_timeout_s)
                                                log(f"[energy] sizing verify target case={tcase} variant={tvariant} phase={phase} work_units={phase_work_units} min_active={min_active:g}s")
                                                energy_command, energy_lease_env = _journaled_energy_ssh_command(
                                                    transport,
                                                    remote_process_registry,
                                                    target_cmd,
                                                    label=f"energy-sizing-verify-{tcase}-{tvariant}-{phase}",
                                                    timeout_s=verify_timeout_s,
                                                )
                                                verify_probe = run_duration_probe(
                                                    energy_command,
                                                    verify_dir,
                                                    timeout_s=verify_timeout_s,
                                                    margin_s=0.0,
                                                    cancel_event=cancel_event,
                                                    subprocess_env=energy_lease_env,
                                                )
                                                energy_scale_meta["sizing_verify_timeout_s"] = verify_timeout_s
                                                energy_scale_meta["sizing_verify_probe"] = {k: verify_probe.get(k) for k in ("ok", "probe_rc", "probe_duration_s", "collector_duration_s", "power_estimated_duration_s")}
                                                vdur = float(verify_probe.get("probe_duration_s") or 0.0)
                                                if verify_probe.get("ok") and vdur > 0 and vdur < (0.90 * float(min_active)):
                                                    import math as _energy_math2
                                                    scale2 = max(1, int(_energy_math2.ceil((float(min_active) * 1.05) / max(vdur, 1e-6))))
                                                    scale2 = min(scale2, 50)
                                                    old_units2 = int(phase_work_units)
                                                    phase_work_units = int(max(old_units2 + 1, old_units2 * scale2))
                                                    if phase == "latency" or (phase == "streaming" and tvariant == "full"):
                                                        _replace_token_value(phase_extra_tokens, "--runs", phase_work_units)
                                                    elif phase == "streaming":
                                                        _replace_token_value(phase_extra_tokens, "--throughput-frames", phase_work_units)
                                                    extra = " ".join(shlex.quote(str(x)) for x in phase_extra_tokens)
                                                    target_cmd = _bench_remote_cmd_for(extra)
                                                    energy_scale_meta.update({
                                                        "auto_scaled_second_pass": True,
                                                        "second_pass_scale_factor": scale2,
                                                        "second_pass_from_work_units": old_units2,
                                                        "scaled_work_units_per_window": int(phase_work_units),
                                                        "sizing_reason": "verify_probe_duration_below_min_active",
                                                    })
                                                    log(f"[energy] second-pass auto-scale target case={tcase} variant={tvariant} phase={phase}: {old_units2} -> {phase_work_units} work units (verify_probe={vdur:.3f}s, min_active={min_active:g}s)")
                                            except Exception as verify_exc:
                                                energy_scale_meta["sizing_verify_error"] = f"{type(verify_exc).__name__}: {verify_exc}"
                                                log(f"[energy][warn] sizing verify failed for {tcase}/{tvariant}/{phase}: {type(verify_exc).__name__}: {verify_exc}")
                                        else:
                                            energy_scale_meta.update({"auto_scaled": False, "sizing_reason": "probe_long_enough_or_failed"})
                                    except Exception as scale_exc:
                                        energy_scale_meta.update({"auto_scaled": False, "sizing_error": f"{type(scale_exc).__name__}: {scale_exc}"})
                                        log(f"[energy][warn] sizing probe failed for {tcase}/{tvariant}/{phase}: {type(scale_exc).__name__}: {scale_exc}")

                                # v58p: optional safety caps for long Eval u.RECS windows.  These caps
                                # are profile/CLI-controlled and are meant for debug/smoke runs with
                                # very large models such as YOLO11L. They are documented in the phase
                                # payload so thesis runs can prove whether a cap was active.
                                cap_notes: list[str] = []
                                try:
                                    max_wu = max(0, int(getattr(args, "energy_max_work_units_per_window", 0) or 0))
                                except Exception:
                                    max_wu = 0
                                if max_wu > 0 and int(phase_work_units) > max_wu:
                                    cap_notes.append(f"max_work_units_per_window:{phase_work_units}->{max_wu}")
                                    phase_work_units = int(max_wu)
                                try:
                                    max_win_s = max(0, int(getattr(args, "energy_max_window_duration_s", 0) or 0))
                                except Exception:
                                    max_win_s = 0
                                if max_win_s > 0:
                                    est_wu_cap = None
                                    try:
                                        if phase == "streaming" and sizing.get("reference_fps"):
                                            est_wu_cap = int(max(1, math.floor(float(max_win_s) * float(sizing.get("reference_fps")))))
                                        elif sizing.get("latency_ms"):
                                            est_wu_cap = int(max(1, math.floor((float(max_win_s) * 1000.0) / float(sizing.get("latency_ms")))))
                                    except Exception:
                                        est_wu_cap = None
                                    if est_wu_cap is not None and int(phase_work_units) > int(est_wu_cap):
                                        cap_notes.append(f"max_window_duration_s:{phase_work_units}->{est_wu_cap}")
                                        phase_work_units = int(est_wu_cap)
                                if cap_notes:
                                    if phase == "latency" or (phase == "streaming" and tvariant == "full"):
                                        _replace_token_value(phase_extra_tokens, "--runs", phase_work_units)
                                    elif phase == "streaming":
                                        _replace_token_value(phase_extra_tokens, "--throughput-frames", phase_work_units)
                                    extra = " ".join(shlex.quote(str(x)) for x in phase_extra_tokens)
                                    target_cmd = _bench_remote_cmd_for(extra)
                                    energy_scale_meta["cap_notes"] = cap_notes
                                    energy_scale_meta["capped_work_units_per_window"] = int(phase_work_units)
                                    log(f"[energy] target case={tcase} variant={tvariant} phase={phase}: applied caps {'; '.join(cap_notes)}")
                                try:
                                    window_timeout_override = max(0, int(getattr(args, "energy_timeout_s_per_window", 0) or 0))
                                except Exception:
                                    window_timeout_override = 0
                                window_timeout_s = float(window_timeout_override) if window_timeout_override > 0 else (outer_timeout_s + 300 if outer_timeout_s else None)
                                energy_command, energy_lease_env = _journaled_energy_ssh_command(
                                    transport,
                                    remote_process_registry,
                                    target_cmd,
                                    label=f"energy-window-{tcase}-{tvariant}-{phase}",
                                    timeout_s=window_timeout_s,
                                )
                                log(f"[energy] target case={tcase} variant={tvariant} phase={phase} work_units={phase_work_units} repeats={energy_repeats} timeout_s={window_timeout_s or 'auto'}")
                                eres = run_fast_firmware_measurement(
                                    energy_command,
                                    phase_out,
                                    setup=energy_setup,
                                    defaults=energy_defaults,
                                    duration_s=known_duration,
                                    run_count=energy_repeats,
                                    timeout_s=window_timeout_s,
                                    setup_id=energy_setup_id,
                                    run_id=str(current_run_id),
                                    inference_count=max(1, int(phase_work_units)),
                                    confidence_level=float(getattr(args, "energy_confidence_level", 0.95) or 0.95),
                                    physical_scope=str(getattr(args, "energy_physical_scope", "") or "FS"),
                                    window_label=str(getattr(args, "energy_window_label", "command") or "command"),
                                    require_runtime_work_units=bool(getattr(args, "energy_strict", False)),
                                    require_command_window_alignment=bool(getattr(args, "energy_strict", False)),
                                    host_normalization_role=host_normalization_role,
                                    host_normalization_source_run_id=current_run_id,
                                    host_normalization_target_variant=tvariant,
                                    cancel_event=cancel_event,
                                    subprocess_env=energy_lease_env,
                                )
                                phase_ok.append(bool(eres.get("ok")))
                                phase_payload = dict(eres)
                                try:
                                    phase_payload.update(energy_scale_meta)
                                except Exception:
                                    pass
                                phase_payload["phase"] = phase
                                phase_payload["work_units_per_window"] = int(phase_work_units)
                                phase_payload["requested_work_units_per_window"] = int(initial_phase_work_units)
                                phase_payload["phase_repeat_count"] = int(energy_repeats)
                                phase_payload["energy_target_case"] = tcase
                                phase_payload["energy_target_variant"] = tvariant
                                phase_payload["energy_phase_out_dir"] = str(phase_out)
                                phase_payload["energy_aggregate_path"] = str(phase_out / "energy_aggregate.json")
                                phase_payloads.append(phase_payload)
                                log(f"[energy] target case={tcase} variant={tvariant} phase={phase} status={eres.get('status')} avg_power_w={eres.get('avg_power_w')}")
                            try:
                                sum_e_target = sum(float(ph.get("sum_energy_total_j") or ph.get("avg_energy_total_j") or 0.0) for ph in phase_payloads)
                            except Exception:
                                sum_e_target = None
                            target_aggregate = {
                                "schema": "onnx-splitpoint/energy-target-phase-aggregate",
                                "schema_version": 1,
                                "ok": bool(phase_payloads) and all(phase_ok),
                                "status": "ok" if phase_payloads and all(phase_ok) else ("partial" if phase_payloads else "skipped"),
                                "setup_id": energy_setup_id,
                                "run_id": str(current_run_id),
                                "row_scope": True,
                                "energy_measurement_scope": "command_energy",
                                "energy_physical_scope": str(getattr(args, "energy_physical_scope", "") or "FS").upper(),
                                "energy_window_label": str(getattr(args, "energy_window_label", "command") or "command").lower(),
                                "confidence_level": float(getattr(args, "energy_confidence_level", 0.95) or 0.95),
                                "target_order_mode": target_order_payload.get("mode"),
                                "target_order_seed": target_order_payload.get("seed"),
                                "energy_target_case": tcase,
                                "energy_target_variant": tvariant,
                                "energy_applies_to_all_cases": bool(target.get("energy_applies_to_all_cases")),
                                "target_id": tname,
                                "phase_count": len(phase_payloads),
                                "phases": phase_payloads,
                                "sum_energy_total_j": sum_e_target,
                            }
                            _write_json(tout / "energy_aggregate.json", target_aggregate)
                            _write_json(tout / "energy_summary.json", target_aggregate)
                            target_results.append({
                                **dict(target),
                                "target_id": tname,
                                "ok": bool(target_aggregate.get("ok")),
                                "status": target_aggregate.get("status"),
                                "energy_aggregate": str(tout / "energy_aggregate.json"),
                                "energy_summary": str(tout / "energy_summary.json"),
                                "avg_power_w": next((ph.get("avg_power_w") for ph in phase_payloads if ph.get("avg_power_w") is not None), None),
                                "sum_energy_total_j": sum_e_target,
                            })
                            log(f"[energy] target case={tcase} variant={tvariant} aggregate_status={target_aggregate.get('status')} phases={len(phase_payloads)}")
                        try:
                            sum_e = sum(float(t.get("sum_energy_total_j") or 0.0) for t in target_results if t.get("sum_energy_total_j") is not None)
                        except Exception:
                            sum_e = None
                        energy_result_payload = {
                            "ok": any(bool(t.get("ok")) for t in target_results),
                            "status": "ok" if target_results and all(bool(t.get("ok")) for t in target_results) else "partial",
                            "row_scope": True,
                            "schema": "onnx-splitpoint/energy-row-scope-aggregate",
                            "schema_version": 1,
                            "setup_id": energy_setup_id,
                            "run_id": current_run_id,
                            "energy_measurement_scope": "command_energy",
                            "target_count": len(target_results),
                            "target_results": target_results,
                            "sum_energy_total_j": sum_e,
                        }
                        energy_out_dir.mkdir(parents=True, exist_ok=True)
                        _write_json(energy_out_dir / "energy_aggregate.json", energy_result_payload)
                        _write_json(energy_out_dir / "energy_summary.json", energy_result_payload)
                        _write_json(local_run_dir / "energy_measurement.json", energy_result_payload)
                        try:
                            run_meta["energy"] = {
                                "enabled": True,
                                "setup_id": energy_setup_id,
                                "energy_out_dir": str(energy_out_dir),
                                "aggregate": str(energy_out_dir / "energy_aggregate.json"),
                                "ok": bool(energy_result_payload.get("ok")),
                                "row_scope": True,
                                "target_count": len(target_results),
                            }
                            _write_json(local_run_dir / "run_meta.json", run_meta)
                        except Exception:
                            pass
                energy_enabled = False
            except Exception as exc:
                log(f"[energy] row_variant energy measurement failed: {type(exc).__name__}: {exc}")
                try:
                    _write_json(local_run_dir / "energy_measurement_error.json", {"error": f"{type(exc).__name__}: {exc}", "scope": "row_variant"})
                except Exception:
                    pass
                energy_enabled = False
        if energy_enabled:
            try:
                from onnx_splitpoint_tool.energy.config import (
                    energy_defaults_from_registry,
                    energy_measurements_root,
                    energy_setup_from_registry,
                    load_hardware_registry,
                )
                from onnx_splitpoint_tool.energy.collector import run_fast_firmware_measurement, run_duration_probe

                assert energy_registry_preflight is not None
                assert energy_setup_preflight is not None
                registry = energy_registry_preflight
                energy_defaults = _bind_energy_ab_runtime_args(
                    energy_defaults_from_registry(registry),
                    args,
                )
                try:
                    energy_defaults.heartbeat_s = max(10, int(getattr(args, "energy_heartbeat_s", 60) or 60))
                except Exception:
                    pass
                energy_setup_id = energy_setup_id_preflight
                energy_setup = energy_setup_preflight
                if not energy_setup.enabled:
                    log(f"[energy] Energy measurement requested but setup {energy_setup_id!r} is not enabled; running benchmark without energy.")
                    remote_rc = transport.run_streaming(
                        bench_remote_cmd,
                        timeout=outer_timeout_s,
                        on_line=on_line,
                        cancel_event=cancel_event,
                    )
                elif not str(energy_setup.urecs_address or "").strip():
                    log(f"[energy] Energy measurement requested but setup {energy_setup_id!r} has no u.RECS address; running benchmark without energy.")
                    remote_rc = transport.run_streaming(
                        bench_remote_cmd,
                        timeout=outer_timeout_s,
                        on_line=on_line,
                        cancel_event=cancel_event,
                    )
                else:
                    energy_root_base = str(getattr(args, "energy_output_root", "") or "").strip()
                    if energy_root_base:
                        energy_base = Path(energy_root_base).expanduser().resolve()
                    else:
                        energy_base = energy_measurements_root(local_working_dir) / "Benchmarks" / suite_dir.name
                    energy_run_id = _safe_local_name(f"{run_id}_{energy_setup_id}_{_extract_run_id_from_add_args(getattr(args, 'add_args', '')) or 'all'}")
                    energy_out_dir = energy_base / energy_run_id
                    # Reproduce exactly the remote shell command that SSHTransport would run.
                    # The u.RECS collector is local; it executes this ssh command during
                    # the measured window.  Bundle/upload happened before this point, and
                    # result collection happens after this point, so the power window covers
                    # the actual remote benchmark execution instead of local scp/download.
                    energy_window_timeout_s = outer_timeout_s + 300 if outer_timeout_s else None
                    energy_command, energy_lease_env = _journaled_energy_ssh_command(
                        transport,
                        remote_process_registry,
                        bench_remote_cmd,
                        label="energy-legacy-window",
                        timeout_s=energy_window_timeout_s,
                    )
                    log(f"[energy] Measuring remote benchmark with u.RECS setup={energy_setup_id} address={energy_setup.urecs_address} out={energy_out_dir}")
                    log("[energy] Duration probe will run the remote benchmark once before the measured run unless a known duration is supplied.")
                    energy_result_payload = run_fast_firmware_measurement(
                        energy_command,
                        energy_out_dir,
                        setup=energy_setup,
                        defaults=energy_defaults,
                        duration_s=getattr(args, "energy_known_duration_s", None),
                        run_count=max(1, int(getattr(args, "energy_run_count", 1) or 1)),
                        timeout_s=energy_window_timeout_s,
                        setup_id=energy_setup_id,
                        run_id=str(_extract_run_id_from_add_args(getattr(args, 'add_args', '')) or 'all'),
                        inference_count=max(1, int(warmup + effective_runs)),
                            confidence_level=float(getattr(args, "energy_confidence_level", 0.95) or 0.95),
                            physical_scope=str(getattr(args, "energy_physical_scope", "") or "FS"),
                            window_label=str(getattr(args, "energy_window_label", "command") or "command"),
                            require_runtime_work_units=bool(getattr(args, "energy_strict", False)),
                            require_command_window_alignment=bool(getattr(args, "energy_strict", False)),
                            cancel_event=cancel_event,
                            subprocess_env=energy_lease_env,
                    )
                    _write_json(local_run_dir / "energy_measurement.json", energy_result_payload)
                    try:
                        run_meta["energy"] = {
                            "enabled": True,
                            "setup_id": energy_setup_id,
                            "energy_out_dir": str(energy_out_dir),
                            "aggregate": str(energy_out_dir / "energy_aggregate.json"),
                            "ok": bool(energy_result_payload.get("ok")),
                            "avg_power_w": energy_result_payload.get("avg_power_w"),
                            "dispatch_energy_total_j": energy_result_payload.get("dispatch_energy_total_j") or energy_result_payload.get("sum_energy_total_j") or energy_result_payload.get("avg_energy_total_j"),
                            "dispatch_avg_energy_j_per_window": energy_result_payload.get("dispatch_avg_energy_j_per_window") or energy_result_payload.get("avg_energy_total_j_per_window") or energy_result_payload.get("avg_energy_total_j"),
                            "avg_energy_total_j_semantics": energy_result_payload.get("avg_energy_total_j_semantics") or "legacy; prefer dispatch_energy_total_j and dispatch_avg_energy_j_per_window",
                            "dispatch_energy_naming_note": "dispatch_energy_total_j is total/dispatch-scale; dispatch_avg_energy_j_per_window is per u.RECS window average",
                        }
                        _write_json(local_run_dir / "run_meta.json", run_meta)
                    except Exception:
                        pass
                    for line in [
                        f"[energy] collector status={energy_result_payload.get('status')} valid_runs={energy_result_payload.get('valid_postprocessed_runs')} avg_power_w={energy_result_payload.get('avg_power_w')} energy_j={energy_result_payload.get('sum_energy_total_j') or energy_result_payload.get('avg_energy_total_j')}",
                        f"[energy] summary={energy_out_dir / 'energy_aggregate.json'}",
                    ]:
                        log(line)
                    # Collector return code refers to the acquisition process.  Treat
                    # remote benchmark as successful when the collector ran and the
                    # remote command did not visibly fail.  Result collection below will
                    # still determine whether useful benchmark artifacts exist.
                    remote_rc = 0 if bool(energy_result_payload.get("ok")) else 1
            except Exception as exc:
                log(f"[energy] Energy measurement failed before/during collection: {type(exc).__name__}: {exc}")
                try:
                    _write_json(local_run_dir / "energy_measurement_error.json", {"error": f"{type(exc).__name__}: {exc}"})
                except Exception:
                    pass
                # Fall back to a normal benchmark execution so a temporary energy
                # issue does not make manual benchmarking unusable.
                remote_rc = transport.run_streaming(
                    bench_remote_cmd,
                    timeout=outer_timeout_s,
                    on_line=on_line,
                    cancel_event=cancel_event,
                )
        else:
            remote_rc = transport.run_streaming(
                bench_remote_cmd,
                timeout=outer_timeout_s,
                on_line=on_line,
                cancel_event=cancel_event,
            )

        if remote_rc != 0:
            remote_error_text = "\n".join(recent_remote_lines).lower()
            if _is_terminal_remote_storage_error(remote_error_text):
                terminal_remote_failure = _terminal_remote_failure_payload(
                    remote_rc=remote_rc,
                    recent_remote_lines=recent_remote_lines,
                    fallback_error=f"Remote benchmark failed (rc={remote_rc})",
                    occurred_at=_utc_now_iso(),
                    force_terminal=True,
                    failure_kind="terminal_remote_storage_failure",
                )
                terminal_remote_storage_failure = True
                bench_error = str(
                    terminal_remote_failure.get("primary_error")
                    or f"Remote benchmark failed (rc={remote_rc})"
                )
                log(
                    "[primary-error] remote storage failure detected from "
                    f"runtime output (rc={remote_rc}); no collection/SCP follows: "
                    f"{bench_error}"
                )
            elif remote_rc == 124:
                timeout_label = f"{int(outer_timeout_s)}s" if outer_timeout_s is not None else "the configured limit"
                bench_error = f"Remote benchmark timed out after {timeout_label}"
            elif remote_rc == 130:
                bench_error = "Remote benchmark cancelled"
            elif remote_rc == TERMINAL_REMOTE_CLEANUP_RC:
                terminal_remote_failure = _terminal_remote_failure_payload(
                    remote_rc=remote_rc,
                    recent_remote_lines=recent_remote_lines,
                    fallback_error="Remote benchmark failed and exact cleanup could not be proven.",
                    occurred_at=_utc_now_iso(),
                )
                terminal_remote_storage_failure = True
                bench_error = str(
                    terminal_remote_failure.get("primary_error") or
                    "Remote benchmark failed and exact cleanup could not be proven."
                )
                log(
                    "[primary-error] "
                    f"{bench_error} (remote rc={TERMINAL_REMOTE_CLEANUP_RC}; "
                    "no further remote collection will be attempted)"
                )
            else:
                bench_error = f"Remote benchmark failed (rc={remote_rc})"
            if not terminal_remote_failure:
                log(f"[warn] {bench_error} (continuing to collect/download results)")

    except BundleCancelled as exc:
        cancelled = True
        bench_error = str(exc).strip() or "Cancelled"
        remote_rc = 130
        if not remote_mutation_started and not remote_leased_operation_started:
            pre_mutation_dispatch_failure = {
                "schema": "onnx-splitpoint/pre-remote-admission-failure",
                "schema_version": 1,
                "stage": "pre_remote_mutation",
                "dispatch_status": "cancelled_before_dispatch",
                "failure_kind": "remote_dispatch_cancelled_before_start",
                "remote_dispatch_failed": False,
                "remote_dispatched": False,
                "pre_remote_mutation": True,
                "remote_mutation_started": False,
                "remote_leased_operation_started": False,
                "primary_error": bench_error,
                "primary_error_at": _utc_now_iso(),
                "remote_rc": remote_rc,
            }
            log(
                "[cancel] remote dispatch cancelled before any mutation or "
                "leased operation; collection is not applicable"
            )

    except KeyboardInterrupt:
        cancelled = True
        bench_error = "Cancelled"
        remote_rc = 130
        exception_text = "KeyboardInterrupt"
        if not remote_mutation_started and not remote_leased_operation_started:
            pre_mutation_dispatch_failure = {
                "schema": "onnx-splitpoint/pre-remote-admission-failure",
                "schema_version": 1,
                "stage": "pre_remote_mutation",
                "dispatch_status": "cancelled_before_dispatch",
                "failure_kind": "remote_dispatch_cancelled_before_start",
                "remote_dispatch_failed": False,
                "remote_dispatched": False,
                "pre_remote_mutation": True,
                "remote_mutation_started": False,
                "remote_leased_operation_started": False,
                "primary_error": bench_error,
                "primary_error_at": _utc_now_iso(),
                "remote_rc": remote_rc,
            }
            log(
                "[cancel] remote dispatch interrupted before any mutation or "
                "leased operation; collection is not applicable"
            )

    except Exception as e:
        bench_error = str(e)
        exception_text = traceback.format_exc()
        error_text = f"{type(e).__name__}: {e}".lower()
        pre_mutation_failure = bool(
            not remote_mutation_started
            and not remote_leased_operation_started
        )
        parsed_pre_mutation_rc = _remote_failure_rc(bench_error)
        pre_mutation_cancelled = bool(
            pre_mutation_failure
            and (
                parsed_pre_mutation_rc == 130
                or (cancel_event is not None and cancel_event.is_set())
            )
        )
        pre_mutation_connectivity_failure = bool(
            pre_mutation_failure
            and not pre_mutation_cancelled
            and _is_remote_connectivity_failure(error_text)
        )
        terminal_remote_storage_failure = bool(
            not pre_mutation_cancelled
            and not pre_mutation_connectivity_failure
            and (
                "remote_storage_preflight_failed" in error_text
                or "terminal_remote_storage_failure" in error_text
                or _is_terminal_remote_storage_error(error_text)
            )
        )
        if pre_mutation_cancelled:
            cancelled = True
            remote_rc = 130
            pre_mutation_dispatch_failure = {
                "schema": "onnx-splitpoint/pre-remote-admission-failure",
                "schema_version": 1,
                "stage": "pre_remote_mutation",
                "dispatch_status": "cancelled_before_dispatch",
                "failure_kind": "remote_dispatch_cancelled_before_start",
                "remote_dispatch_failed": False,
                "remote_dispatched": False,
                "pre_remote_mutation": True,
                "remote_mutation_started": False,
                "remote_leased_operation_started": False,
                "primary_error": bench_error,
                "primary_error_at": _utc_now_iso(),
                "remote_rc": remote_rc,
            }
            log(
                "[cancel] remote dispatch cancelled before any mutation or "
                "leased operation; collection is not applicable"
            )
        elif pre_mutation_failure:
            remote_rc = parsed_pre_mutation_rc
            pre_mutation_failure_kind = (
                "remote_connectivity_unavailable"
                if pre_mutation_connectivity_failure
                else "terminal_remote_storage_failure"
                if terminal_remote_storage_failure
                else "pre_remote_admission_failure"
            )
            pre_mutation_dispatch_failure = {
                "schema": "onnx-splitpoint/pre-remote-admission-failure",
                "schema_version": 1,
                "stage": "pre_remote_mutation",
                "dispatch_status": "failed_to_dispatch",
                "failure_kind": pre_mutation_failure_kind,
                "remote_dispatch_failed": True,
                "remote_dispatched": False,
                "pre_remote_mutation": True,
                "remote_mutation_started": False,
                "remote_leased_operation_started": False,
                "primary_error": bench_error,
                "primary_error_at": _utc_now_iso(),
                "remote_rc": remote_rc,
            }
            log(
                "[primary-error] remote admission failed before any "
                "remote mutation or leased operation; preserving the original "
                "failure and skipping collection without minting a lease: "
                f"{bench_error}"
            )
        registry_poisoned = bool(
            remote_process_registry is not None
            and remote_process_registry.cancelled
            and not (cancel_event and cancel_event.is_set())
        )
        if registry_poisoned:
            remote_rc = TERMINAL_REMOTE_CLEANUP_RC
            terminal_remote_failure = _terminal_remote_failure_payload(
                remote_rc=remote_rc,
                recent_remote_lines=recent_remote_lines,
                fallback_error=bench_error,
                occurred_at=_utc_now_iso(),
            )
            bench_error = str(terminal_remote_failure.get("primary_error") or bench_error)
            terminal_remote_storage_failure = True
        elif terminal_remote_storage_failure:
            terminal_remote_failure = _terminal_remote_failure_payload(
                remote_rc=remote_rc,
                recent_remote_lines=[
                    *recent_remote_lines,
                    *str(bench_error or "").splitlines(),
                ],
                fallback_error=bench_error,
                occurred_at=_utc_now_iso(),
                force_terminal=True,
                failure_kind="terminal_remote_storage_failure",
            )

    # ----------------------------
    # Always-download: collect/package/download even if the run failed.
    # ----------------------------
    try:
        if pre_mutation_dispatch_failure:
            log(
                "[remote] pre-mutation dispatch did not start; skipping "
                "collect/package/SCP because no remote process or run path was created"
            )
            raise _TerminalRemoteCollectionSuppressed()
        if terminal_remote_failure or terminal_remote_storage_failure:
            log(
                "[primary-error] terminal remote failure preserved; skipping "
                "collect/package/SCP because the lease is poisoned or those "
                "steps require more remote writes"
            )
            raise _TerminalRemoteCollectionSuppressed()
        cancel_requested = bool(cancelled or (cancel_event and cancel_event.is_set()) or remote_rc == 130)
        collect_timeout = 20 if cancel_requested else 60
        pack_timeout = 60 if cancel_requested else 300
        scp_timeout = 60 if cancel_requested else (int(args.timeout_s) if args.timeout_s is not None and int(args.timeout_s) > 0 else 300)

        log("Collecting results on remote (best effort)")
        # Quote remote paths inside the script (the *whole* script is single-quoted
        # for bash -lc, so we use double quotes for paths).
        _rr = remote_results_dir
        _rs = remote_suite_dir
        collect_cmd = "bash -lc " + shlex.quote(
            _remote_result_collect_script(remote_results_dir=_rr, remote_suite_dir=_rs)
        )
        rc_collect, out_collect = transport.run(collect_cmd, timeout_s=collect_timeout)
        if rc_collect == TERMINAL_REMOTE_CLEANUP_RC:
            remote_rc = TERMINAL_REMOTE_CLEANUP_RC
            terminal_remote_failure = _terminal_remote_failure_payload(
                remote_rc=remote_rc,
                recent_remote_lines=[
                    *recent_remote_lines,
                    *str(out_collect or "").splitlines(),
                ],
                fallback_error=(
                    "Remote result collection failed and exact cleanup could "
                    "not be proven."
                ),
                occurred_at=_utc_now_iso(),
            )
            terminal_remote_storage_failure = True
            bench_error = str(
                terminal_remote_failure.get("primary_error")
                or "Remote result collection failed and exact cleanup could not be proven."
            )
            log(
                "[primary-error] result collection returned rc=70; "
                "packaging and SCP are suppressed: " + bench_error
            )
            raise _TerminalRemoteCollectionSuppressed()
        collect_storage_failure = _terminal_remote_storage_stage_payload(
            stage="remote_result_collect",
            rc=rc_collect,
            output=out_collect,
        )
        if collect_storage_failure:
            terminal_remote_failure = collect_storage_failure
            terminal_remote_storage_failure = True
            bench_error = str(
                collect_storage_failure.get("primary_error")
                or "Remote storage failure during result collection"
            )
            log(
                "[primary-error] result collection exhausted remote storage; "
                "packaging and SCP are suppressed: " + bench_error
            )
            raise _TerminalRemoteCollectionSuppressed()
        if rc_collect != 0:
            log(f"[warn] remote collect step failed (rc={rc_collect})")
            if out_collect.strip():
                log(out_collect.strip()[-2000:])

        progress(0.90, "Benchmark done")

        log("Packaging results on remote")
        pack_cmd = (
            "python3 - <<'PY'\n"
            "import os, tarfile\n"
            f"results_dir = os.path.expanduser({remote_results_dir!r})\n"
            f"out_path = os.path.expanduser({remote_results_tar!r})\n"
            "os.makedirs(results_dir, exist_ok=True)\n"
            "os.makedirs(os.path.dirname(out_path), exist_ok=True)\n"
            "with tarfile.open(out_path, 'w:gz') as tar:\n"
            "    tar.add(results_dir, arcname='results', recursive=True)\n"
            "print('OK packaged', out_path)\n"
            "PY"
        )
        rc_pack, out_pack = transport.run(pack_cmd, timeout_s=pack_timeout)
        pack_storage_failure = _terminal_remote_storage_stage_payload(
            stage="remote_result_pack",
            rc=rc_pack,
            output=out_pack,
        )
        if pack_storage_failure:
            terminal_remote_failure = pack_storage_failure
            terminal_remote_storage_failure = True
            bench_error = str(
                pack_storage_failure.get("primary_error")
                or "Remote storage failure during result packaging"
            )
            log(
                "[primary-error] result packaging exhausted remote storage; "
                "all SCP paths are suppressed: " + bench_error
            )
            raise _TerminalRemoteCollectionSuppressed()
        if rc_pack != 0:
            log(f"[warn] remote results packaging failed (rc={rc_pack})")
            if out_pack.strip():
                log(out_pack.strip()[-2000:])
        progress(0.93, "Pack results")

        if rc_pack == 0:
            log("Downloading results (scp)")
            local_tar = local_run_dir / "results_bundle.tar.gz"
            rc_dl, out_dl = transport.scp_download(remote_results_tar, str(local_tar), timeout_s=scp_timeout)
            if rc_dl != 0:
                log(f"[warn] scp download of results tar failed (rc={rc_dl})")
                if out_dl.strip():
                    log(out_dl.strip()[-2000:])
            else:
                progress(0.97, "Download results")
                try:
                    log("Extracting results locally")
                    _extract_tarball(local_tar, local_run_dir, log=log)
                    results_downloaded = True
                    try:
                        local_results_dir = local_run_dir / "results"
                        if local_results_dir.is_dir() and energy_result_payload:
                            merge_manifest = _merge_energy_payload_into_results(local_results_dir, energy_result_payload)
                            _write_json(local_run_dir / "diagnostics" / "post_energy_merge_consistency.json", {
                                "energy_merge_manifest": merge_manifest,
                                "energy_payload_summary": {
                                    "row_scope": bool(energy_result_payload.get("row_scope")),
                                    "target_count": len(energy_result_payload.get("target_results") or []),
                                },
                            })
                            log(f"[energy] post-download merge: rows={merge_manifest.get('merged_rows')} files={len(merge_manifest.get('merged_files') or [])}")
                    except Exception as merge_exc:
                        log(f"[warn] failed to merge energy metrics into downloaded results: {merge_exc}")
                    try:
                        local_results_dir = local_run_dir / "results"
                        if local_results_dir.is_dir():
                            # Repack after possible energy merge so the bundle is not stale.
                            try:
                                create_results_bundle_from_results_dir(local_results_dir, local_run_dir / "results_bundle.tar.gz", mode="full")
                            except TypeError:
                                create_results_bundle_from_results_dir(local_results_dir, local_run_dir / "results_bundle.tar.gz")
                            lean_tar = local_run_dir / "results_bundle_lean.tar.gz"
                            create_results_bundle_from_results_dir(local_results_dir, lean_tar, mode="lean")
                            log(f"[info] wrote lean results bundle: {lean_tar}")
                    except Exception as lean_exc:
                        log(f"[warn] failed to create lean results bundle: {lean_exc}")
                except Exception as e:
                    log(f"[warn] failed to extract results tar: {e}")

        if not results_downloaded:
            # Fallback: recursive download of the results directory.
            log("Downloading results directory (scp -r)")
            rc_dl2, out_dl2 = transport.scp_download(
                remote_results_dir, str(local_run_dir), recursive=True, timeout_s=scp_timeout
            )
            if rc_dl2 != 0:
                log(f"[warn] scp -r results download failed (rc={rc_dl2})")
                if out_dl2.strip():
                    log(out_dl2.strip()[-2000:])
            else:
                results_downloaded = True
                try:
                    local_results_dir = local_run_dir / "results"
                    if local_results_dir.is_dir() and energy_result_payload:
                        merge_manifest = _merge_energy_payload_into_results(local_results_dir, energy_result_payload)
                        _write_json(local_run_dir / "diagnostics" / "post_energy_merge_consistency.json", {
                            "energy_merge_manifest": merge_manifest,
                            "energy_payload_summary": {
                                "row_scope": bool(energy_result_payload.get("row_scope")),
                                "target_count": len(energy_result_payload.get("target_results") or []),
                            },
                        })
                        log(f"[energy] post-download merge: rows={merge_manifest.get('merged_rows')} files={len(merge_manifest.get('merged_files') or [])}")
                except Exception as merge_exc:
                    log(f"[warn] failed to merge energy metrics into downloaded results: {merge_exc}")
                try:
                    local_results_dir = local_run_dir / "results"
                    if local_results_dir.is_dir():
                        try:
                            create_results_bundle_from_results_dir(local_results_dir, local_run_dir / "results_bundle.tar.gz", mode="full")
                        except TypeError:
                            create_results_bundle_from_results_dir(local_results_dir, local_run_dir / "results_bundle.tar.gz")
                        lean_tar = local_run_dir / "results_bundle_lean.tar.gz"
                        create_results_bundle_from_results_dir(local_results_dir, lean_tar, mode="lean")
                        log(f"[info] wrote lean results bundle: {lean_tar}")
                except Exception as lean_exc:
                    log(f"[warn] failed to create lean results bundle: {lean_exc}")

    except _TerminalRemoteCollectionSuppressed:
        pass
    except BaseException as e:
        # Best-effort only (including KeyboardInterrupt).
        if isinstance(e, KeyboardInterrupt):
            cancelled = True
            bench_error = bench_error or "Cancelled"
            remote_rc = remote_rc or 130
            exception_text = exception_text or "KeyboardInterrupt"
        else:
            if bench_error is None:
                bench_error = str(e)
            if exception_text is None:
                exception_text = traceback.format_exc()

    # Merge remote logs into local logs folder (best effort)
    try:
        remote_logs_local = local_run_dir / "results" / "logs"
        local_logs_dir = local_run_dir / "logs"
        if remote_logs_local.is_dir():
            for name in ["stdout.txt", "stderr.txt"]:
                src = remote_logs_local / name
                if src.exists() and src.stat().st_size > 0:
                    (local_logs_dir / name).write_text(src.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
    except Exception:
        pass

    # ----------------------------
    # Finalize status + minimal stable results index
    # ----------------------------
    results_dir_local = local_run_dir / "results"
    has_results = _detect_useful_results(results_dir_local)

    if cancelled or (cancel_event and cancel_event.is_set()) or remote_rc == 130:
        final_status = "cancelled"
    elif remote_rc == 0 and bench_error is None and exception_text is None:
        final_status = "ok"
    else:
        final_status = "partial" if has_results else "failed"

    # Build tails for fail_reason
    stdout_tail = _read_tail_lines(local_run_dir / "logs" / "stdout.txt")
    stderr_tail = _read_tail_lines(local_run_dir / "logs" / "stderr.txt")
    if not stdout_tail and not stderr_tail:
        stdout_tail = _read_tail_lines(local_run_dir / "logs" / "runner.log")

    extra_fail_reason: dict[str, Any] = {}
    if isinstance(last_suite_progress, dict) and last_suite_progress:
        extra_fail_reason["last_suite_progress"] = dict(last_suite_progress)
    if recent_remote_lines:
        extra_fail_reason["recent_remote_lines"] = list(recent_remote_lines[-50:])
    if pre_mutation_dispatch_failure:
        extra_fail_reason.update(pre_mutation_dispatch_failure)
    if terminal_remote_failure:
        extra_fail_reason["terminal_remote_failure"] = True
        extra_fail_reason["primary_error"] = str(
            terminal_remote_failure.get("primary_error") or ""
        )
        extra_fail_reason["primary_error_context"] = list(
            terminal_remote_failure.get("primary_error_context") or []
        )
        extra_fail_reason["primary_error_at"] = str(
            terminal_remote_failure.get("primary_error_at") or ""
        )
        extra_fail_reason["failure_kind"] = str(
            terminal_remote_failure.get("failure_kind") or
            "terminal_remote_execution_failure"
        )
    if isinstance(timeout_hint, dict) and timeout_hint:
        extra_fail_reason["timeout_estimate"] = {
            **timeout_hint,
            "effective_timeout_s": effective_outer_timeout_s,
            "auto_raised": bool(timeout_decision.get("auto_raised")),
            "warn_too_low": bool(timeout_decision.get("warn_too_low")),
        }

    _finalize_run_status(
        local_run_dir,
        status=final_status,
        started_at=started_at,
        ended_at=_utc_now_iso(),
        remote_rc=remote_rc,
        fail_message=None if final_status == "ok" else (bench_error or "Run failed."),
        exception_text=exception_text,
        stdout_tail=stdout_tail,
        stderr_tail=stderr_tail,
        extra_fail_reason=(extra_fail_reason or None),
    )

    planned_runs: list[dict[str, Any]] = []
    artifact_index: dict[str, Any] = {}
    try:
        br_json = sorted(results_dir_local.glob("benchmark_results_*.json"))
        artifact_index["benchmark_results_json"] = [str(p.relative_to(local_run_dir)) for p in br_json]
        br_md = sorted(results_dir_local.glob("benchmark_summary_*.md"))
        if br_md:
            artifact_index["benchmark_summary_md"] = [str(p.relative_to(local_run_dir)) for p in br_md]
        br_csv = sorted(results_dir_local.glob("benchmark_results_*.csv"))
        if br_csv:
            artifact_index["benchmark_results_csv"] = [str(p.relative_to(local_run_dir)) for p in br_csv]

        plan_path = results_dir_local / "benchmark_plan.json"
        if not plan_path.exists():
            plan_path = suite_dir / "benchmark_plan.json"
        plan = json.loads(plan_path.read_text(encoding="utf-8")) if plan_path.exists() else None
        result_run_id_filters = _extract_run_ids_from_add_args(getattr(args, "add_args", ""))
        result_run_id_filter = ",".join(result_run_id_filters)
        if result_run_id_filters and isinstance(plan, dict):
            plan = _filter_benchmark_plan_for_run_ids(plan, result_run_id_filters)

        rowless_quality_statuses = _rowless_full_only_quality_run_statuses(
            plan=plan if isinstance(plan, Mapping) else {},
            suite_status=(
                _read_json_dict(
                    results_dir_local / "benchmark_suite_status.json"
                ) or {}
            ),
            final_status=final_status,
            benchmark_result_count=len(br_json),
            expected_eval_run_id=str(
                getattr(args, "quality_evidence_eval_id", "") or ""
            ),
            expected_model_id=str(
                getattr(args, "quality_evidence_model_id", "") or ""
            ),
            expected_setup_id=str(
                getattr(args, "quality_evidence_setup_id", "") or ""
            ),
            explicit_quality_only_run_ids=[
                value.strip()
                for value in _argument_value(
                    getattr(args, "add_args", ""),
                    "--quality-only-run-ids",
                    "",
                ).split(",")
                if value.strip()
            ],
            expected_endpoint_id=str(
                getattr(args, "quality_evidence_endpoint_id", "") or ""
            ),
        )

        plan_ids: set[str] = set()
        if plan and isinstance(plan, dict) and isinstance(plan.get("runs"), list):
            for r in plan["runs"]:
                if not isinstance(r, dict):
                    continue
                tag = str(r.get("id", ""))
                if not tag:
                    continue
                plan_ids.add(tag)
                backend = str(r.get("type", ""))
                if backend == "ort":
                    backend = f"ort:{r.get('provider', '')}".rstrip(":")
                elif backend == "hailo":
                    backend = f"hailo:{r.get('hw_arch', '')}".rstrip(":")
                elif not backend:
                    backend = "unknown"
                def _glob_first(pat: str) -> Optional[Path]:
                    ms = sorted(results_dir_local.glob(pat))
                    return ms[0] if ms else None

                # The suite may append suffixes (e.g. _auto) to avoid collisions.
                res_json = results_dir_local / f"benchmark_results_{tag}.json"
                if not res_json.exists():
                    alt = _glob_first(f"benchmark_results_{tag}*.json")
                    if alt is not None:
                        res_json = alt

                res_csv = results_dir_local / f"benchmark_results_{tag}.csv"
                if not res_csv.exists():
                    alt = _glob_first(f"benchmark_results_{tag}*.csv")
                    if alt is not None:
                        res_csv = alt

                res_md = results_dir_local / f"benchmark_summary_{tag}.md"
                if not res_md.exists():
                    alt = _glob_first(f"benchmark_summary_{tag}*.md")
                    if alt is not None:
                        res_md = alt

                res_tex = results_dir_local / f"benchmark_table_{tag}.tex"
                if not res_tex.exists():
                    alt = _glob_first(f"benchmark_table_{tag}*.tex")
                    if alt is not None:
                        res_tex = alt

                planned_runs.append(
                    {
                        "tag": tag,
                        "backend": backend,
                        "status": (
                            "ok" if res_json.exists()
                            else rowless_quality_statuses.get(tag)
                            or (
                                "partial"
                                if final_status == "partial" else "failed"
                            )
                        ),
                        "artifacts": {
                            "benchmark_results_json": str(res_json.relative_to(local_run_dir)) if res_json.exists() else None,
                            "benchmark_results_csv": str(res_csv.relative_to(local_run_dir)) if res_csv.exists() else None,
                            "benchmark_summary_md": str(res_md.relative_to(local_run_dir)) if res_md.exists() else None,
                            "benchmark_table_tex": str(res_tex.relative_to(local_run_dir)) if res_tex.exists() else None,
                        },
                    }
                )

        for p in br_json:
            tag = p.stem.replace("benchmark_results_", "", 1)
            base_tag = tag
            # benchmark_suite.py often appends the preset suffix, e.g.
            # benchmark_results_ort_tensorrt_auto.json.  Avoid adding a
            # duplicate "unknown" row when the scoped plan already has the
            # base run id.
            if base_tag.endswith("_auto"):
                base_tag = base_tag[:-5]
            if tag in plan_ids or base_tag in plan_ids:
                continue
            planned_runs.append(
                {
                    "tag": tag,
                    "backend": "unknown",
                    "status": "ok",
                    "artifacts": {"benchmark_results_json": str(p.relative_to(local_run_dir))},
                }
            )
    except Exception:
        pass

    _finalize_run_results(
        local_run_dir,
        host=run_meta["host"],
        objective=run_meta.get("objective", "latency"),
        planned_runs=planned_runs,
        artifact_index=artifact_index,
    )

    dt = time.time() - t0
    log(f"DONE in {dt:.1f}s. Results in: {local_run_dir}")
    progress(1.0, "Done" if final_status == "ok" else "Done (with errors)")

    final_error: Optional[str]
    if final_status == "ok":
        final_error = None
    elif final_status == "partial" and bool(getattr(args, "resume", True)):
        base_msg = bench_error or "Run ended early"
        final_error = f"{base_msg}. Partial results were collected; rerun the remote benchmark to resume the same run."
    else:
        final_error = bench_error or "Run failed"

    remote_cleanup_payload: dict[str, Any] | None = None
    try:
        want_cleanup = bool(getattr(args, "cleanup_remote_after_download", True))
        cleanup_ok_status = (final_status == "ok") or (final_status == "partial" and bool(getattr(args, "cleanup_remote_on_partial", False)))
        if want_cleanup and cleanup_ok_status and results_downloaded and str(remote_run_dir or "").strip():
            safe_remote = str(remote_run_dir).strip()
            cleanup_script = (
                "set -u\n"
                f"REMOTE_DIR={shlex.quote(safe_remote)}\n"
                "case \"$REMOTE_DIR\" in\n"
                "  *'/splitpoint_runs/'*) ;;\n"
                "  *) echo '[cleanup] refusing unsafe remote path:' \"$REMOTE_DIR\"; exit 2 ;;\n"
                "esac\n"
                "if [ -d \"$REMOTE_DIR\" ]; then\n"
                "  echo '[cleanup] before:'; du -sh \"$REMOTE_DIR\" 2>/dev/null || true\n"
                "  rm -rf -- \"$REMOTE_DIR\"\n"
                "  if [ -e \"$REMOTE_DIR\" ]; then echo '[cleanup] remove failed: path still exists'; exit 3; fi\n"
                "  echo '[cleanup] removed:' \"$REMOTE_DIR\"\n"
                "else\n"
                "  echo '[cleanup] remote path already absent:' \"$REMOTE_DIR\"\n"
                "fi\n"
            )
            rc_clean, out_clean = transport.run("bash -lc " + shlex.quote(cleanup_script), timeout_s=180)
            remote_cleanup_payload = {
                "schema": "onnx-splitpoint/remote-cleanup",
                "schema_version": 1,
                "timestamp_utc": _utc_now_iso(),
                "enabled": True,
                "attempted": True,
                "remote_run_dir": safe_remote,
                "rc": rc_clean,
                "ok": rc_clean == 0,
                "output_tail": str(out_clean or "")[-12000:],
                "policy": {
                    "cleanup_remote_after_download": want_cleanup,
                    "cleanup_remote_on_partial": bool(getattr(args, "cleanup_remote_on_partial", False)),
                    "final_status": final_status,
                    "results_downloaded": bool(results_downloaded),
                },
            }
            diag_dir = local_run_dir / "diagnostics"
            diag_dir.mkdir(parents=True, exist_ok=True)
            _write_json(diag_dir / "remote_cleanup.json", remote_cleanup_payload)
            try:
                (diag_dir / "remote_cleanup.log").write_text(str(out_clean or ""), encoding="utf-8", errors="replace")
            except Exception:
                pass
            if rc_clean == 0:
                log(f"[cleanup] removed remote run dir: {safe_remote}")
            else:
                log(f"[cleanup][warn] failed to remove remote run dir rc={rc_clean}: {str(out_clean or '')[-1000:]}")
        else:
            remote_cleanup_payload = {
                "schema": "onnx-splitpoint/remote-cleanup",
                "schema_version": 1,
                "timestamp_utc": _utc_now_iso(),
                "enabled": bool(getattr(args, "cleanup_remote_after_download", True)),
                "attempted": False,
                "remote_run_dir": str(remote_run_dir or ""),
                "reason": "disabled_or_not_successful_or_no_download",
                "policy": {
                    "final_status": final_status,
                    "results_downloaded": bool(results_downloaded),
                    "cleanup_remote_after_download": bool(getattr(args, "cleanup_remote_after_download", True)),
                    "cleanup_remote_on_partial": bool(getattr(args, "cleanup_remote_on_partial", False)),
                },
            }
    except Exception as cleanup_exc:
        remote_cleanup_payload = {
            "schema": "onnx-splitpoint/remote-cleanup",
            "schema_version": 1,
            "timestamp_utc": _utc_now_iso(),
            "enabled": bool(getattr(args, "cleanup_remote_after_download", True)),
            "attempted": True,
            "remote_run_dir": str(remote_run_dir or ""),
            "ok": False,
            "error": f"{type(cleanup_exc).__name__}: {cleanup_exc}",
        }
        try:
            diag_dir = local_run_dir / "diagnostics"
            diag_dir.mkdir(parents=True, exist_ok=True)
            _write_json(diag_dir / "remote_cleanup_error.json", remote_cleanup_payload)
        except Exception:
            pass
        log(f"[cleanup][warn] remote cleanup bookkeeping failed: {type(cleanup_exc).__name__}: {cleanup_exc}")

    return {
        "ok": final_status == "ok",
        "status": final_status,
        "dispatch_status": str(
            pre_mutation_dispatch_failure.get("dispatch_status") or ""
        ),
        "local_run_dir": str(local_run_dir),
        "remote_run_dir": remote_run_dir,
        "error": final_error,
        "remote_rc": remote_rc,
        "terminal_remote_failure": bool(terminal_remote_failure),
        "primary_failure": dict(
            terminal_remote_failure or pre_mutation_dispatch_failure
        ),
        "failure_kind": str(
            pre_mutation_dispatch_failure.get("failure_kind") or ""
        ),
        "remote_dispatch_failed": bool(
            pre_mutation_dispatch_failure.get("remote_dispatch_failed")
        ),
        "remote_dispatched": bool(
            remote_mutation_started or remote_leased_operation_started
        ),
        "pre_remote_mutation": bool(pre_mutation_dispatch_failure),
        "resumed": resume_requested,
        "requested_run_id": requested_run_id,
        "active_run_id": str(run_id),
        "bundle": bundle_stats_payload,
        "energy_summary": str((local_run_dir / "energy" / "remote_benchmark" / "energy_summary.json")) if (local_run_dir / "energy" / "remote_benchmark" / "energy_summary.json").exists() else None,
        "energy_aggregate": str((local_run_dir / "energy" / "remote_benchmark" / "energy_aggregate.json")) if (local_run_dir / "energy" / "remote_benchmark" / "energy_aggregate.json").exists() else None,
        "energy_registry_path": energy_registry_path_meta,
        "energy_registry_snapshot_sha256": (
            energy_registry_snapshot_meta
        ),
        "energy_registry_binding": dict(energy_registry_binding),
        "remote_cleanup": remote_cleanup_payload,
    }
