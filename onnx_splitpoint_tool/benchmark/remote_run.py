from __future__ import annotations

import json
import re
import shlex
import tarfile
import time
import textwrap
import posixpath
import traceback
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from onnx_splitpoint_tool.remote.bundle import BundleCancelled, build_suite_bundle, remote_minimal_bundle_patterns
from onnx_splitpoint_tool.benchmark.results_bundle import create_results_bundle_from_results_dir
from onnx_splitpoint_tool.log_utils import sanitize_log
from onnx_splitpoint_tool.remote.ssh_transport import HostConfig as RemoteHost
from onnx_splitpoint_tool.remote.ssh_transport import SSHTransport
from onnx_splitpoint_tool.benchmark.suite_refresh import refresh_suite_harness


RUN_STATUS_SCHEMA_VERSION = 1
RUN_RESULTS_SCHEMA_VERSION = 1
_CASE_DIR_RE = re.compile(r"^b\d+$")


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
    """Heuristic: consider a run 'useful' if at least one key artifact exists."""
    if not results_dir.exists():
        return False
    patterns = [
        "benchmark_results_*.json",
        "validation_report.json",
        "*/results_*/validation_report.json",
        "*/results_*/validation_report.pdf",
        "*/results_*/validation_report.png",
    ]
    for pat in patterns:
        if any(results_dir.glob(pat)):
            return True
    # fallback: any non-empty file
    for p in results_dir.rglob("*"):
        if p.is_file() and p.stat().st_size > 0:
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


def _read_json_dict(path: Path) -> dict[str, Any] | None:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


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
    wanted_provider = str(args.provider or "auto").strip()
    wanted_add_args = str(args.add_args or "").strip()
    wanted_warmup = int(args.warmup)
    wanted_iters = int(args.iters)
    wanted_repeats = int(args.repeats)
    wanted_port = int(host.port or 22)

    candidates: list[tuple[str, Path, dict[str, Any], dict[str, Any]]] = []
    for run_dir in sorted(root.iterdir(), reverse=True):
        if not run_dir.is_dir():
            continue
        meta = _read_json_dict(run_dir / "run_meta.json")
        status = _read_json_dict(run_dir / "run_status.json")
        if not meta or not status:
            continue
        if str(meta.get("benchmark_set_json") or "") != wanted_bench:
            continue

        meta_host = meta.get("host") or {}
        if str(meta_host.get("user") or "") != str(host.user or ""):
            continue
        if str(meta_host.get("host") or "") != str(host.host or ""):
            continue
        if int(meta_host.get("port") or 22) != wanted_port:
            continue

        meta_args = meta.get("args") or {}
        if str(meta_args.get("provider") or "auto").strip() != wanted_provider:
            continue
        if int(meta_args.get("warmup") or 0) != wanted_warmup:
            continue
        if int(meta_args.get("iters") or 0) != wanted_iters:
            continue
        if int(meta_args.get("repeats") or 0) != wanted_repeats:
            continue
        if str(meta_args.get("add_args") or "").strip() != wanted_add_args:
            continue

        status_name = str(status.get("status") or "").strip().lower()
        if status_name in {"ok", "running"}:
            continue
        if not _detect_useful_results(run_dir / "results"):
            continue

        ended = str(status.get("ended_at") or meta.get("started_at") or run_dir.name)
        candidates.append((ended, run_dir, meta, status))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    _ended, run_dir, meta, status = candidates[0]
    return run_dir, meta, status


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
) -> dict:
    """Run a benchmark suite on a remote host (ssh/scp).

    Returns a dict for the GUI with:
        ok: bool
        local_run_dir: str
        remote_run_dir: str
        error: optional error string
    """

    t0 = time.time()

    benchmark_set_json = Path(benchmark_set_json).expanduser().resolve()
    suite_dir = benchmark_set_json.parent
    bench_payload = _read_json_dict(benchmark_set_json) or {}
    plan_payload = bench_payload.get('plan') if isinstance(bench_payload.get('plan'), dict) else {}
    objective_value = str(plan_payload.get('objective') or bench_payload.get('objective') or 'latency').strip().lower() or 'latency'
    timeout_hint = estimate_remote_timeout_hint(suite_dir, args)
    timeout_decision = apply_remote_timeout_hint(getattr(args, "timeout_s", None), timeout_hint)
    effective_outer_timeout_s = timeout_decision.get("effective_timeout_s")

    requested_run_id = str(run_id)
    repeat_dir = _safe_local_name(str(repeats_idx).strip() or "1")
    resume_requested = False
    resume_meta: dict[str, Any] | None = None
    local_results_root = Path(local_working_dir).expanduser().resolve() / "Results" / suite_dir.name / repeat_dir

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
    run_meta: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "repeat_idx": str(repeats_idx),
        "started_at": started_at,
        "created_at": started_at,
        "suite_dir": str(suite_dir),
        "benchmark_set_json": str(benchmark_set_json),
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
            "resume": bool(getattr(args, "resume", True)),
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
        # Keep an explicit objective for future-proofing and dissertation exports.
        "objective": objective_value,
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

    # Best-effort: refresh runner scripts inside an existing suite.
    # Older suites may contain stale harness files; bundling should be self-healing.
    # IMPORTANT: Only touch files whose content actually changed so bundle caching
    # remains effective.
    try:
        refresh_stats = refresh_suite_harness(
            suite_dir,
            benchmark_set_json=benchmark_set_json,
            log=log,
        )
        if bool(refresh_stats.get("changed")):
            force_bundle_rebuild = True
    except Exception as e:
        log(f'[warn] Could not refresh suite runner scripts: {e}')

    # NOTE: run_meta.json is already written by init_local_run_artifacts().

    transport = SSHTransport(host)

    def run_checked(cmd: str, *, timeout: Optional[int] = None) -> None:
        rc, out = transport.run(cmd, timeout=timeout)
        if rc != 0:
            raise RuntimeError(f"Remote command failed (rc={rc}): {cmd}\n{out}")

    def scp_upload_checked(local_path: Path, remote_path: str, *, recursive: bool | None = None) -> None:
        rc, out = transport.scp_upload(local_path=local_path, remote_path=remote_path, recursive=recursive)
        if rc != 0:
            raise RuntimeError(f"SCP upload failed (rc={rc}): {local_path} -> {remote_path}\n{out}")

    def scp_download_checked(remote_path: str, local_path: Path, *, recursive: bool = False) -> None:
        rc, out = transport.scp_download(remote_path=remote_path, local_path=local_path, recursive=recursive)
        if rc != 0:
            raise RuntimeError(f"SCP download failed (rc={rc}): {remote_path} -> {local_path}\n{out}")

    # Phase-0 status tracking (finalized at the end, no early returns)
    remote_base_raw = (host.remote_base_dir or "~/splitpoint_runs").rstrip("/")
    remote_run_dir = f"{remote_base_raw}/{suite_dir.name}/{run_id}/{repeat_dir}"
    remote_suite_dir = f"{remote_run_dir}/suite"
    remote_results_dir = f"{remote_run_dir}/results"
    remote_results_tar = f"{remote_run_dir}/results.tar.gz"

    bench_error: Optional[str] = None
    exception_text: Optional[str] = None
    remote_rc: Optional[int] = None
    cancelled: bool = False
    results_downloaded: bool = False
    remote_resume_state_found: bool = False

    try:
        # Resolve remote base (expands ~ and symlinks)
        remote_base = transport.resolve_path(remote_base_raw, timeout_s=10).rstrip("/")
        remote_run_dir = f"{remote_base}/{suite_dir.name}/{run_id}/{repeat_dir}"
        remote_suite_dir = f"{remote_run_dir}/suite"
        remote_results_dir = f"{remote_run_dir}/results"
        remote_results_tar = f"{remote_run_dir}/results.tar.gz"
        remote_trt_cache_root = f"{remote_base}/{suite_dir.name}/_shared_trt_cache"

        # RemoteHost.user_host is a plain string ("user@host").
        # We keep a pretty variant that includes the port for logs.
        log(f"Remote host: {host.user_host_pretty}")
        log(f"Suite (local): {suite_dir}")
        log(f"[remote] run_dir={remote_run_dir}")
        if resume_requested:
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
        run_checked(mkdir_cmd)
        progress(0.05, "Remote dirs ready")
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
        progress(0.055, "Remote preflight (python / onnx / ort / hailo)")

        want_hailo = False
        want_cuda = False
        want_tensorrt = False

        plan_path = suite_dir / "benchmark_plan.json"
        if plan_path.exists():
            try:
                plan = json.loads(plan_path.read_text(encoding="utf-8"))
                for run in plan.get("runs", []):
                    if not isinstance(run, dict):
                        continue

                    for k in ("provider", "full_provider", "stage1_provider", "stage2_provider"):
                        v = run.get(k)
                        if not isinstance(v, str):
                            continue
                        tok = v.strip().lower()
                        if tok.startswith("hailo"):
                            want_hailo = True
                        elif tok == "cuda":
                            want_cuda = True
                        elif tok == "tensorrt":
                            want_tensorrt = True

                    t = run.get("type")
                    if isinstance(t, str) and t.strip().lower() == "hailo":
                        want_hailo = True
            except Exception as e:
                log(f"[preflight] warning: could not parse benchmark_plan.json: {e!r}")

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
                # Treat as shell snippet (e.g. "source /opt/hailo/setup_env.sh")
                preflight_lines.append(remote_venv_cmd)
            else:
                # Treat as path to activate script (e.g. "~/hailo_py/bin/activate").
                # Important: avoid single-quote shlex.quote here because it prevents "~/" and "$HOME" expansion.
                venv_path = remote_venv_cmd
                if venv_path.startswith("~/"):
                    venv_path = "$HOME/" + venv_path[2:]
                preflight_lines.append(f'if [ -f "{venv_path}" ]; then source "{venv_path}"; fi')
            preflight_lines.append("ENV_PY=$(command -v python3 || command -v python || true)")

        # Embed the expected provider needs into the remote script so we can pick a
        # python interpreter that actually has the required ORT EPs (CUDA/TRT live
        # in host on many setups, while Hailo lives in a venv).
        preflight_lines.append(f"WANT_HAILO={'1' if want_hailo else '0'}")
        preflight_lines.append(f"WANT_CUDA={'1' if want_cuda else '0'}")
        preflight_lines.append(f"WANT_TRT={'1' if want_tensorrt else '0'}")

        # Select python for the benchmark run:
        # - If CUDA/TRT are requested, prefer the interpreter that exposes those EPs.
        # - Otherwise, prefer ENV_PY if it has core deps.
        # Also: do NOT use PYTHONPATH to inject venv site-packages, because PYTHONPATH
        # entries are searched *before* system site-packages and may shadow onnxruntime-gpu.
        # Instead we pass the venv site-packages via SPLITPOINT_EXTRA_SITES and append it
        # using site.addsitedir() inside the runner.
        preflight_lines.append('RUN_PY="$SYS_PY"')
        preflight_lines.append('ENV_SITE=""')
        preflight_lines.append('SYS_EPS=""')
        preflight_lines.append('ENV_EPS=""')
        preflight_lines.append('has_cuda() { case "$1" in *CUDAExecutionProvider*) return 0;; *) return 1;; esac; }')
        preflight_lines.append('has_trt() { case "$1" in *TensorrtExecutionProvider*) return 0;; *) return 1;; esac; }')
        preflight_lines.append('if [ -n "$ENV_PY" ] && [ -x "$ENV_PY" ]; then')
        preflight_lines.append('  ENV_SITE=$("$ENV_PY" -c \'import site,os; ps=[]; getsp=getattr(site,"getsitepackages",None); ps.extend(getsp() if getsp else []); usp=getattr(site,"getusersitepackages",lambda:None)(); ps.append(usp); ps=[p for p in ps if p and os.path.isdir(p)]; out=[]; [out.append(p) for p in ps if p not in out]; print(":".join(out))\' 2>/dev/null || true)')
        preflight_lines.append('  SYS_EPS=$("$SYS_PY" -c \'import onnxruntime as ort; print("|".join(ort.get_available_providers()))\' 2>/dev/null || true)')
        preflight_lines.append('  ENV_EPS=$("$ENV_PY" -c \'import onnxruntime as ort; print("|".join(ort.get_available_providers()))\' 2>/dev/null || true)')
        preflight_lines.append('  if [ "$WANT_TRT" = "1" ]; then')
        preflight_lines.append('    if has_cuda "$SYS_EPS" && has_trt "$SYS_EPS"; then RUN_PY="$SYS_PY"; elif has_cuda "$ENV_EPS" && has_trt "$ENV_EPS"; then RUN_PY="$ENV_PY"; fi')
        preflight_lines.append('  elif [ "$WANT_CUDA" = "1" ]; then')
        preflight_lines.append('    if has_cuda "$SYS_EPS"; then RUN_PY="$SYS_PY"; elif has_cuda "$ENV_EPS"; then RUN_PY="$ENV_PY"; fi')
        preflight_lines.append('  else')
        preflight_lines.append('    if "$ENV_PY" -c "import onnx,onnxruntime" >/dev/null 2>&1; then RUN_PY="$ENV_PY"; fi')
        preflight_lines.append('  fi')
        preflight_lines.append('fi')
        preflight_lines.append('if [ "$WANT_HAILO" = "1" ] && [ -n "$ENV_SITE" ]; then export SPLITPOINT_EXTRA_SITES="$ENV_SITE"; fi')
        preflight_lines.append('export PRECHECK_SYS_PY="$SYS_PY"')
        preflight_lines.append('export PRECHECK_ENV_PY="$ENV_PY"')
        preflight_lines.append('export PRECHECK_RUN_PY="$RUN_PY"')
        preflight_lines.append('export PRECHECK_ENV_SITE="$ENV_SITE"')

        preflight_lines.append(f"export PRECHECK_OUT={shlex.quote(preflight_remote_path)}")
        preflight_lines.append(f"export PRECHECK_WANT_HAILO={'1' if want_hailo else '0'}")
        preflight_lines.append(f"export PRECHECK_WANT_CUDA={'1' if want_cuda else '0'}")
        preflight_lines.append(f"export PRECHECK_WANT_TRT={'1' if want_tensorrt else '0'}")
        preflight_lines.append('echo "[preflight] SYS_PY=$SYS_PY ENV_PY=$ENV_PY RUN_PY=$RUN_PY"')

        preflight_py = textwrap.dedent(r"""
        import glob
        import json
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
            },
            "platform": {"platform": platform.platform(), "machine": platform.machine()},
            "wants": {"hailo": want_hailo, "cuda": want_cuda, "tensorrt": want_trt},
        }

        critical_missing = []

        try:
            import onnx  # type: ignore
            info["onnx"] = {"ok": True, "version": getattr(onnx, "__version__", None)}
        except Exception as e:
            info["onnx"] = {"ok": False, "error": repr(e)}
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
            critical_missing.append("onnxruntime")

        # Provider availability hints
        if info.get("onnxruntime", {}).get("ok"):
            eps = info["onnxruntime"]["available_providers"]
            warnings = []
            if want_cuda and "CUDAExecutionProvider" not in eps:
                warnings.append(f"CUDAExecutionProvider missing (available: {eps})")
            if want_trt and "TensorrtExecutionProvider" not in eps:
                warnings.append(f"TensorrtExecutionProvider missing (available: {eps})")
            info["onnxruntime"]["warnings"] = warnings

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

        if critical_missing:
            print(f"[preflight] ERROR: missing critical python modules: {critical_missing}", file=sys.stderr)
            sys.exit(10)

        sys.exit(0)
        """).strip()

        preflight_lines.append("\"$RUN_PY\" - <<'PY'\n" + preflight_py + "\nPY")

        preflight_cmd = "bash -lc " + shlex.quote("\n".join(preflight_lines))

        rc, out = transport.run(preflight_cmd, timeout=120)
        log(out)
        if rc != 0:
            raise RuntimeError(f"Remote preflight failed (rc={rc}). See output above.")

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
                log('[package] Rebuilding bundle because suite files were refreshed locally.')
            bundle_includes, bundle_extra_excludes = remote_minimal_bundle_patterns()
            stats = build_suite_bundle(
                suite_dir=suite_dir,
                out_path=bundle_path,
                includes=bundle_includes,
                excludes=bundle_extra_excludes,
                progress_cb=_bundle_progress,
                should_cancel=(lambda: bool(cancel_event and cancel_event.is_set())),
                reuse_if_unchanged=bool(args.reuse_bundle) and not force_bundle_rebuild,
            )

            raw_mb = stats.total_bytes / (1024 * 1024) if stats.total_bytes else 0.0
            tgz_mb = stats.bundle_path.stat().st_size / (1024 * 1024)
            cache_tag = "cached" if getattr(stats, "reused", False) else "rebuilt"
            log(
                f"[remote] bundle: {stats.bundle_path} (raw {raw_mb:.1f} MB, tar.gz {tgz_mb:.1f} MB, files={stats.files}, {cache_tag})"
            )

            progress(0.25, "Bundle ready")

            remote_bundle_path = f"{remote_run_dir}/bundle.tar.gz"
            log("Uploading bundle (scp)")
            scp_upload_checked(stats.bundle_path, remote_bundle_path)
            progress(0.35, "Bundle uploaded")

            # Extract into run-local suite dir
            remote_suite_dir = f"{remote_run_dir}/suite"
            log(f"[remote] suite_dir={remote_suite_dir}")
            log("Extracting bundle on remote")
            progress(0.36, "Remote mkdir (suite dir)")
            mkdir_suite_cmd = f"mkdir -p {remote_suite_dir}"
            log(f"[remote] {mkdir_suite_cmd}")
            run_checked(mkdir_suite_cmd)
            progress(0.38, "Remote untar bundle")
            untar_cmd = f"tar -xzf {remote_bundle_path} -C {remote_suite_dir}"
            log(f"[remote] {untar_cmd}")
            run_checked(untar_cmd)
            progress(0.40, "Suite uploaded")

        else:
            # direct copy via scp -r
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
            }
            _write_json(local_run_dir / "run_meta.json", run_meta)
        except Exception:
            pass

        throughput_frames = max(0, int(getattr(args, "throughput_frames", 24) or 0))
        throughput_warmup_frames = max(0, int(getattr(args, "throughput_warmup_frames", 6) or 0))
        throughput_queue_depth = max(1, int(getattr(args, "throughput_queue_depth", 2) or 1))

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
        )
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
            'ENV_PY="$(command -v python3)"',
            f"WANT_HAILO={'1' if want_hailo else '0'}",
            f"WANT_CUDA={'1' if want_cuda else '0'}",
            f"WANT_TRT={'1' if want_tensorrt else '0'}",
            'RUN_PY="$SYS_PY"',
            'ENV_SITE=""',
            'SYS_EPS=""',
            'ENV_EPS=""',
            'has_cuda() { case "$1" in *CUDAExecutionProvider*) return 0;; *) return 1;; esac; }',
            'has_trt() { case "$1" in *TensorrtExecutionProvider*) return 0;; *) return 1;; esac; }',
            # Collect venv site-packages (for Hailo) without using PYTHONPATH.
            'if [ -n "$ENV_PY" ] && [ -x "$ENV_PY" ]; then',
            '  ENV_SITE=$("$ENV_PY" -c \'import site,os; ps=[]; getsp=getattr(site,"getsitepackages",None); ps.extend(getsp() if getsp else []); usp=getattr(site,"getusersitepackages",lambda:None)(); ps.append(usp); ps=[p for p in ps if p and os.path.isdir(p)]; out=[]; [out.append(p) for p in ps if p not in out]; print(":".join(out))\' 2>/dev/null || true)',
            '  SYS_EPS=$("$SYS_PY" -c \'import onnxruntime as ort; print("|".join(ort.get_available_providers()))\' 2>/dev/null || true)',
            '  ENV_EPS=$("$ENV_PY" -c \'import onnxruntime as ort; print("|".join(ort.get_available_providers()))\' 2>/dev/null || true)',
            '  if [ "$WANT_TRT" = "1" ]; then',
            '    if has_cuda "$SYS_EPS" && has_trt "$SYS_EPS"; then RUN_PY="$SYS_PY"; elif has_cuda "$ENV_EPS" && has_trt "$ENV_EPS"; then RUN_PY="$ENV_PY"; fi',
            '  elif [ "$WANT_CUDA" = "1" ]; then',
            '    if has_cuda "$SYS_EPS"; then RUN_PY="$SYS_PY"; elif has_cuda "$ENV_EPS"; then RUN_PY="$ENV_PY"; fi',
            '  else',
            '    if "$ENV_PY" -c "import onnx,onnxruntime" >/dev/null 2>&1; then RUN_PY="$ENV_PY"; fi',
            '  fi',
            'fi',
            # Make Hailo python wheels visible without shadowing system packages.
            'if [ "$WANT_HAILO" = "1" ] && [ -n "$ENV_SITE" ]; then export SPLITPOINT_EXTRA_SITES="$ENV_SITE"; fi',
            'echo "[remote] python: SYS_PY=${SYS_PY} ENV_PY=${ENV_PY} RUN_PY=${RUN_PY} | SYS_EPS=${SYS_EPS} | ENV_EPS=${ENV_EPS} | EXTRA_SITES=${SPLITPOINT_EXTRA_SITES:-}" >&2',
            f"{bench_cmd} 1> >(tee logs/stdout.txt) 2> >(tee logs/stderr.txt >&2)",
        ]
        bench_inner = "\n".join(bench_inner_lines)
        bench_remote_cmd = "bash -lc " + shlex.quote(bench_inner)

        outer_timeout_s = None
        try:
            if effective_outer_timeout_s is not None and int(effective_outer_timeout_s) > 0:
                outer_timeout_s = int(effective_outer_timeout_s)
        except Exception:
            outer_timeout_s = 7200

        remote_rc = transport.run_streaming(
            bench_remote_cmd,
            timeout=outer_timeout_s,
            on_line=on_line,
            cancel_event=cancel_event,
        )

        if remote_rc != 0:
            if remote_rc == 124:
                timeout_label = f"{int(outer_timeout_s)}s" if outer_timeout_s is not None else "the configured limit"
                bench_error = f"Remote benchmark timed out after {timeout_label}"
            elif remote_rc == 130:
                bench_error = "Remote benchmark cancelled"
            else:
                bench_error = f"Remote benchmark failed (rc={remote_rc})"
            log(f"[warn] {bench_error} (continuing to collect/download results)")

    except BundleCancelled:
        cancelled = True
        bench_error = "Cancelled"
        remote_rc = 130

    except KeyboardInterrupt:
        cancelled = True
        bench_error = "Cancelled"
        remote_rc = 130
        exception_text = "KeyboardInterrupt"

    except Exception as e:
        bench_error = str(e)
        exception_text = traceback.format_exc()

    # ----------------------------
    # Always-download: collect/package/download even if the run failed.
    # ----------------------------
    try:
        cancel_requested = bool(cancelled or (cancel_event and cancel_event.is_set()) or remote_rc == 130)
        collect_timeout = 20 if cancel_requested else 60
        pack_timeout = 60 if cancel_requested else 300
        scp_timeout = 60 if cancel_requested else (int(args.timeout_s) if args.timeout_s is not None and int(args.timeout_s) > 0 else 300)

        log("Collecting results on remote (best effort)")
        # Quote remote paths inside the script (the *whole* script is single-quoted
        # for bash -lc, so we use double quotes for paths).
        _rr = remote_results_dir.replace('"', '\\"')
        _rs = remote_suite_dir.replace('"', '\\"')
        collect_cmd = (
            "bash -lc "
            + shlex.quote(
                "set -e; shopt -s nullglob; "
                f"suite=\"{_rs}\"; "
                f"mkdir -p \"{_rr}\"; "
                # Suite-level artifacts (including suite logs)
                f"for p in \"$suite\"/benchmark_results_* \"$suite\"/benchmark_summary_* \"$suite\"/benchmark_table_* \"$suite\"/benchmark_tables_* \"$suite\"/benchmark_report_* \"$suite\"/paper_figures_* \"$suite\"/benchmark_plan.json \"$suite\"/benchmark_set.json \"$suite\"/run_meta.json \"$suite\"/benchmark_suite.py \"$suite\"/logs; do "
                f"  [ -e \"$p\" ] && cp -a \"$p\" \"{_rr}/\" || true; "
                "done; "
                # Per-case artifacts (preserve case-id to avoid collisions)
                f"for cd in \"$suite\"/b*; do "
                "  [ -d \"$cd\" ] || continue; "
                "  cid=$(basename \"$cd\"); "
                f"  mkdir -p \"{_rr}/$cid\"; "
                f"  for p in \"$cd\"/results_* \"$cd\"/validation_*; do "
                f"    [ -e \"$p\" ] && cp -a \"$p\" \"{_rr}/$cid/\" || true; "
                "  done; "
                "done"
            )
        )
        rc_collect, out_collect = transport.run(collect_cmd, timeout_s=collect_timeout)
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
                        if local_results_dir.is_dir():
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
                    if local_results_dir.is_dir():
                        lean_tar = local_run_dir / "results_bundle_lean.tar.gz"
                        create_results_bundle_from_results_dir(local_results_dir, lean_tar, mode="lean")
                        log(f"[info] wrote lean results bundle: {lean_tar}")
                except Exception as lean_exc:
                    log(f"[warn] failed to create lean results bundle: {lean_exc}")

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
                        "status": "ok" if res_json.exists() else ("partial" if final_status == "partial" else "failed"),
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
            if tag in plan_ids:
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

    return {
        "ok": final_status == "ok",
        "status": final_status,
        "local_run_dir": str(local_run_dir),
        "remote_run_dir": remote_run_dir,
        "error": final_error,
        "resumed": resume_requested,
        "requested_run_id": requested_run_id,
        "active_run_id": str(run_id),
    }
