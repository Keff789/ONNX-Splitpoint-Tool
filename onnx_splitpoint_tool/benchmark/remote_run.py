from __future__ import annotations

import json
import shlex
import tarfile
import time
import posixpath
import traceback
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from onnx_splitpoint_tool.remote.bundle import BundleCancelled, build_suite_bundle
from onnx_splitpoint_tool.remote.ssh_transport import HostConfig as RemoteHost
from onnx_splitpoint_tool.remote.ssh_transport import SSHTransport


RUN_STATUS_SCHEMA_VERSION = 1
RUN_RESULTS_SCHEMA_VERSION = 1


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
        payload["fail_reason"] = {
            "message": fail_message or "Run failed.",
            "exception": exception_text,
            "stderr_tail": stderr_tail,
            "stdout_tail": stdout_tail,
        }
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
    repeats: int = 1
    warmup: int = 10
    iters: int = 100
    add_args: str = ""

    # Total timeout for the remote benchmark command.
    timeout_s: int = 7200

    # Transfer mode for the suite:
    # - bundle: tar.gz (fast for many small files, supports caching)
    # - direct: scp -r (useful for debugging; may be slower and copies everything)
    transfer_mode: str = "bundle"  # 'bundle' | 'direct'

    # Only relevant for transfer_mode='bundle'
    reuse_bundle: bool = True


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


def _safe_local_name(s: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in s)


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

    repeat_dir = _safe_local_name(str(repeats_idx).strip() or "1")
    local_run_dir = (
        Path(local_working_dir).expanduser().resolve()
        / "Results"
        / suite_dir.name
        / repeat_dir
        / run_id
    )

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
            "transfer_mode": args.transfer_mode,
            "reuse_bundle": args.reuse_bundle,
        },
        # Keep an explicit objective for future-proofing.
        "objective": "latency",
    }

    # Phase-0 invariant: create artifacts BEFORE any SSH work happens.
    init_local_run_artifacts(local_run_dir, run_meta)

    # Always keep a local transcript so remote-debug is not lost, even if the
    # GUI buffer is cleared or the run fails mid-way.
    _gui_log = log
    runner_log_path = local_run_dir / "logs" / "runner.log"

    def log(line: str) -> None:  # type: ignore[no-redef]
        _gui_log(line)
        try:
            with runner_log_path.open("a", encoding="utf-8") as f:
                f.write(line.rstrip("\n") + "\n")
        except Exception:
            # best-effort only
            pass
    
    # Keep this available for error reporting (even if we fail mid-way).
    remote_run_dir: Optional[str] = None

    # Best-effort: refresh runner scripts inside an existing suite.
    # Older suites may contain stale runner scripts; bundling should be self-healing.
    #
    # IMPORTANT: Don't touch files if they're already up-to-date, otherwise we
    # break bundle caching (mtime changes force rebuild every run).
    try:
        import shutil
        import tempfile

        def _copy_if_changed(src: Path, dst: Path) -> bool:
            """Copy src -> dst only if bytes differ (preserves suite_bundle caching)."""
            try:
                if dst.exists() and src.read_bytes() == dst.read_bytes():
                    return False
            except Exception:
                # If comparison fails for any reason, overwrite as safest default.
                pass
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            return True

        # --- Refresh suite-level benchmark_suite.py from the latest template (best-effort). ---
        # Generate into a temp dir and only overwrite if content changed.
        try:
            from ..gui.controller import write_benchmark_suite_script

            with tempfile.TemporaryDirectory(prefix='osp_suite_refresh_') as _td:
                tmp_dir = Path(_td)
                tmp_path = Path(write_benchmark_suite_script(tmp_dir, bench_json_name='benchmark_set.json'))
                dst_path = suite_dir / 'benchmark_suite.py'
                if tmp_path.exists() and _copy_if_changed(tmp_path, dst_path):
                    log(f'[info] Refreshed benchmark_suite.py: {dst_path}')
        except Exception as e:
            log(f"[warn] Could not refresh benchmark_suite.py: {e}")

        # --- Refresh per-case ONNXRuntime runner skeletons ---
        # Runner generator function name changed over time; support both.
        try:
            from ..split_export_runners import write_runner_skeleton_onnxruntime as _write_runner_onnxruntime
        except Exception:  # pragma: no cover
            from ..split_export_runners import write_runner_onnxruntime as _write_runner_onnxruntime  # type: ignore

        def _refresh_case_runner(case_dir: Path, manifest_filename: str) -> int:
            """Regenerate ORT runner skeleton into temp dir and copy changed files into case_dir.

            Returns number of updated files.
            """
            updated = 0
            with tempfile.TemporaryDirectory(prefix='osp_runner_refresh_') as _td:
                tmp_case = Path(_td)
                try:
                    _write_runner_onnxruntime(str(tmp_case), manifest_filename=manifest_filename, target='auto')  # type: ignore[arg-type]
                except TypeError:
                    # Legacy API
                    _write_runner_onnxruntime(str(tmp_case), Path(manifest_filename), export_mode='folder')  # type: ignore[misc]

                for fname in (
                    'run_split_onnxruntime.py',
                    'run_split_onnxruntime.sh',
                    'run_split_onnxruntime.bat',
                ):
                    src = tmp_case / fname
                    if not src.exists():
                        continue
                    dst = case_dir / fname
                    if _copy_if_changed(src, dst):
                        updated += 1
            return updated

        # Discover case folders by manifest presence (robust vs relying on split_candidates.csv)
        case_manifests = sorted(suite_dir.glob('b*/split_manifest.json'))
        total_updated_files = 0
        total_updated_cases = 0
        for manifest in case_manifests:
            case_dir = manifest.parent
            if case_dir.parent != suite_dir:
                continue
            n = _refresh_case_runner(case_dir, manifest.name)
            if n:
                total_updated_cases += 1
                total_updated_files += n

        if total_updated_files:
            log(
                f"[info] Refreshed runner scripts in {total_updated_cases}/{len(case_manifests)} cases "
                f"(files updated: {total_updated_files})."
            )

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

    try:
        # Resolve remote base (expands ~ and symlinks)
        remote_base = transport.resolve_path(remote_base_raw, timeout_s=10).rstrip("/")
        remote_run_dir = f"{remote_base}/{suite_dir.name}/{run_id}/{repeat_dir}"
        remote_suite_dir = f"{remote_run_dir}/suite"
        remote_results_dir = f"{remote_run_dir}/results"
        remote_results_tar = f"{remote_run_dir}/results.tar.gz"

        # RemoteHost.user_host is a plain string ("user@host").
        # We keep a pretty variant that includes the port for logs.
        log(f"Remote host: {host.user_host_pretty}")
        log(f"Suite (local): {suite_dir}")
        log(f"[remote] run_dir={remote_run_dir}")

        # Explicit phases so the UI shows where it hangs (mkdir vs untar vs run).
        progress(0.01, "Remote mkdir (run/results)")
        mkdir_cmd = f"mkdir -p {remote_run_dir} {remote_results_dir}"
        log(f"[remote] {mkdir_cmd}")
        run_checked(mkdir_cmd)
        progress(0.05, "Remote dirs ready")

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
            stats = build_suite_bundle(
                suite_dir=suite_dir,
                out_path=bundle_path,
                progress_cb=_bundle_progress,
                should_cancel=(lambda: bool(cancel_event and cancel_event.is_set())),
                reuse_if_unchanged=bool(args.reuse_bundle),
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

        bench_cmd = (
            f"python3 -u benchmark_suite.py"
            f" --provider {shlex.quote(provider)}"
            f" --plan benchmark_plan.json"
            f" --warmup {warmup}"
            f" --runs {effective_runs}"
        )
        if args.add_args:
            # Advanced args supported by benchmark_suite.py (raw passthrough).
            bench_cmd += f" {args.add_args}"

        def on_line(line: str) -> None:
            log(line)
            sp = parse_benchmark_suite_progress(line)
            if sp is not None:
                # Map suite progress into 40%..90%
                progress(0.40 + 0.50 * sp.pct, f"Running {sp.i}/{sp.n}")

        # Make sure we also have remote stdout/stderr files for debugging.
        bench_inner = (
            f"set -e; cd {shlex.quote(remote_suite_dir)}; mkdir -p logs; "
            f"{bench_cmd} "
            "1> >(tee logs/stdout.txt) 2> >(tee logs/stderr.txt >&2)"
        )
        bench_remote_cmd = "bash -lc " + shlex.quote(bench_inner)

        remote_rc = transport.run_streaming(
            bench_remote_cmd,
            timeout=args.timeout_s,
            on_line=on_line,
            cancel_event=cancel_event,
        )

        if remote_rc != 0:
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
        scp_timeout = 60 if cancel_requested else (args.timeout_s or 300)

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

    return {
        "ok": final_status == "ok",
        "status": final_status,
        "local_run_dir": str(local_run_dir),
        "remote_run_dir": remote_run_dir,
        "error": None if final_status == "ok" else (bench_error or "Run failed"),
    }
