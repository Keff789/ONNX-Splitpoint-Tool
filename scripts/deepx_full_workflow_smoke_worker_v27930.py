#!/usr/bin/env python3
"""Bounded H2 orchestration of the normal Native-Full CLI, never an inference path."""
from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deepx_full_output_probe_worker_v27930 import read_json, write_json, sha256, validate_stage

FRAMES = 100
WARMUP = 10
CHILD_TIMEOUT = 180
SCHEMA = "onnx-splitpoint/deepx-full-normal-workflow-smoke/v1"


def _processes():
    """Same-namespace Linux identities, including host-mounted /proc views."""
    namespace = os.readlink("/proc/self/ns/pid")
    self_status = Path("/proc/self/status").read_text()
    self_ids = next((line.split()[1:] for line in self_status.splitlines() if line.startswith("NSpid:")), [])
    if not self_ids or int(self_ids[-1]) != os.getpid():
        raise RuntimeError("process_supervision_pid_namespace_unverifiable")
    host_rows = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if os.readlink(entry / "ns/pid") != namespace:
                continue
            status = (entry / "status").read_text()
            ids = next((line.split()[1:] for line in status.splitlines() if line.startswith("NSpid:")), [])
            if not ids:
                continue
            fields = (entry / "stat").read_text().rsplit(") ", 1)[1].split()
            host_rows[int(entry.name)] = {"local": int(ids[-1]), "parent_host": int(fields[1]),
                                          "start": fields[19], "state": fields[0]}
        except (FileNotFoundError, ProcessLookupError, PermissionError, IndexError, ValueError):
            continue
    result = {item["local"]: {"parent": host_rows.get(item["parent_host"], {}).get("local", 0),
                              "start": item["start"], "state": item["state"]}
              for item in host_rows.values()}
    if os.getpid() not in result:
        raise RuntimeError("process_supervision_self_identity_missing")
    return result


def _capture_owned(owner, tracked):
    current = _processes()
    anchors = {owner}
    anchors.update(pid for pid, start in tracked.items() if current.get(pid, {}).get("start") == start)
    while True:
        children = {pid for pid, item in current.items() if item["parent"] in anchors}
        if children.issubset(anchors):
            break
        anchors.update(children)
    for pid in anchors - {os.getpid()}:
        if pid in current:
            tracked[pid] = current[pid]["start"]
    return {pid: item for pid, item in current.items()
            if tracked.get(pid) == item["start"] and item["state"] != "Z"}


def _signal_owned(tracked, sig):
    current = _processes()
    for pid, start in list(tracked.items()):
        if current.get(pid, {}).get("start") != start or pid == os.getpid():
            continue
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            pass


def supervised_run(command, *, cwd, log_path, timeout=CHILD_TIMEOUT, grace=5.0):
    """Run one CLI tree; reap children even when a grandchild owns a session."""
    if os.name != "posix" or not Path("/proc/self/stat").is_file():
        raise RuntimeError("normal_workflow_smoke_requires_linux_process_supervision")
    _processes()  # Validate namespace mapping before creating any child.
    # A standalone supervisor can reap orphaned descendants instead of leaving
    # zombies or untracked workers behind after the CLI's own timeout.
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        raise OSError(ctypes.get_errno(), "cannot establish child supervision")
    cancelled = []
    previous = {}
    for sig in (signal.SIGTERM, signal.SIGINT):
        previous[sig] = signal.signal(sig, lambda signum, frame: cancelled.append(signum))
    tracked = {}
    began = time.monotonic()
    timed_out = False
    lingering = False
    proc = None
    try:
        with Path(log_path).open("w", encoding="utf-8") as log:
            proc = subprocess.Popen(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            while True:
                live = _capture_owned(proc.pid, tracked)
                # Orphaned subprocesses are now our direct children. Adopt only
                # within this standalone supervisor's own process tree.
                _capture_owned(os.getpid(), tracked)
                returncode = proc.poll()
                if cancelled or time.monotonic() - began >= timeout:
                    timed_out = not bool(cancelled)
                    break
                if returncode is not None:
                    # The pre-poll snapshot may still contain the CLI itself:
                    # it can exit between /proc observation and poll(). Only a
                    # fresh post-exit snapshot can establish live descendants.
                    lingering = bool(_capture_owned(os.getpid(), tracked))
                    break
                time.sleep(0.05)
            _capture_owned(os.getpid(), tracked)
            _signal_owned(tracked, signal.SIGTERM)
            until = time.monotonic() + grace
            while time.monotonic() < until and _capture_owned(os.getpid(), tracked):
                time.sleep(0.05)
            _signal_owned(tracked, signal.SIGKILL)
            proc.wait(timeout=5)
            # Reap adopted grandchildren after the direct child has been waited.
            until = time.monotonic() + 5
            while time.monotonic() < until:
                try:
                    pid, _ = os.waitpid(-1, os.WNOHANG)
                except ChildProcessError:
                    break
                if pid == 0:
                    if not _capture_owned(os.getpid(), tracked):
                        break
                    time.sleep(0.05)
            survivors = list(_capture_owned(os.getpid(), tracked))
            return {"returncode": proc.returncode, "timed_out": timed_out,
                    "cancelled": bool(cancelled), "lingering_children_after_cli": lingering,
                    "owned_processes_observed": len(tracked), "owned_survivors": survivors,
                    "elapsed_s": time.monotonic() - began,
                    "cleanup_complete": not survivors}
    finally:
        if proc is not None:
            _capture_owned(os.getpid(), tracked)
            _signal_owned(tracked, signal.SIGKILL)
            if proc.poll() is None:
                proc.wait(timeout=5)
            until = time.monotonic() + 5
            while time.monotonic() < until:
                try:
                    pid, _ = os.waitpid(-1, os.WNOHANG)
                except ChildProcessError:
                    break
                if pid == 0:
                    if not _capture_owned(os.getpid(), tracked):
                        break
                    time.sleep(0.05)
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def prepare_exact_input(request, work, suite, image):
    tool = work / "tool"
    sys.path.insert(0, str(tool))
    from onnx_splitpoint_tool.runners.native_full_input import (
        prepare_and_seal_deepx_native_full_input, load_sealed_deepx_native_full_input,
    )
    from onnx_splitpoint_tool import runners
    if not Path(runners.__file__).resolve().is_relative_to(tool):
        raise RuntimeError("staged_input_runtime_imported_from_other_installation")
    original = read_json(work / "original_full_input_manifest.json")
    contract = read_json(suite / "deepx/deepx_m1/full/output_contract.json")
    sealed = prepare_and_seal_deepx_native_full_input(
        image_path=image, input_contract=contract, task="detection",
        out_dir=suite / "results/deepx_m1_full/prepared_input",
        model=request["model_id"], setup_id=request["setup_id"], comparison_backend="deepx",
    )
    actual = sealed["payload"]
    fields = ("schema", "schema_version", "backend", "model", "setup_id", "comparison_backend", "task",
              "input_image_sha256", "input_dump_sha256", "input_dump_bytes", "input_shape_hwc",
              "runtime_input_name", "runtime_input_shape", "runtime_input_dtype",
              "runtime_input_sha256", "runtime_input_bytes", "runtime_input_layout",
              "runtime_color_space", "runtime_normalization", "runtime_preprocessing_sha256",
              "runtime_numeric_input_sha256")
    conflicts = [key for key in fields if key not in original or original[key] != actual[key]]
    if conflicts:
        raise ValueError("original_full_input_binding_mismatch:" + ",".join(conflicts))
    load_sealed_deepx_native_full_input(sealed["manifest_path"], image_path=image,
        input_contract=contract, task="detection", expected_model=request["model_id"],
        expected_setup_id=request["setup_id"], expected_comparison_backend="deepx")
    return {"original_full_input_binding_verified": True, "fields_compared": list(fields),
            "runtime_input_sha256": actual["runtime_input_sha256"],
            "runtime_input_bytes": actual["runtime_input_bytes"],
            "input_image_sha256": actual["input_image_sha256"]}


def summarize_native_result(data, process):
    rows = data.get("rows") or []
    if len(rows) != 1 or not isinstance(rows[0], dict):
        return {"status": "failed", "failure_reason": "normal_native_full_result_row_missing_or_ambiguous"}
    row = rows[0]
    checks = {
        "normal_cli_success": process.get("returncode") == 0 and not process.get("timed_out") and not process.get("cancelled"),
        "process_cleanup": process.get("cleanup_complete") is True and not process.get("lingering_children_after_cli"),
        "row_accepted": row.get("ok") is True,
        "one_requested": row.get("repetition_count_requested") == 1,
        "one_attempted": row.get("repetition_count_attempted") == 1,
        "one_valid": row.get("repetition_count_valid") == 1,
        "physical_endpoint": row.get("stage") == "decoded_pre_nms",
        "semantic_success": row.get("semantic_dump_status") == "ok",
        "full_completion": row.get("completed_frames") == FRAMES and row.get("postprocess_completed_frames") == FRAMES and row.get("postprocess_completion_verified") is True,
        "frozen_postprocess": row.get("host_postprocess_frozen") is True and row.get("postprocess_included") is True,
        "input_binding": row.get("semantic_performance_source_binding_verified") is True,
    }
    failed_checks = [name for name, passed in checks.items() if not passed]
    return {"status": "pass" if not failed_checks else "failed", "checks": checks,
            "failure_reason": row.get("primary_repetition_failure_reason") or row.get("failure_reason") or (
                "normal_native_full_smoke_checks_failed:" + ",".join(failed_checks) if failed_checks else ""),
            "repetition_count_requested": row.get("repetition_count_requested"),
            "repetition_count_attempted": row.get("repetition_count_attempted"),
            "repetition_count_valid": row.get("repetition_count_valid"),
            "completed_frames": row.get("completed_frames"), "postprocess_completed_frames": row.get("postprocess_completed_frames"),
            "physical_stage": row.get("stage"), "frozen_contract_sha256": row.get("frozen_host_postprocess_contract_sha256")}


def collect_suite_diagnostics(suite, output):
    target = output / "suite_diagnostics"
    for path in sorted(suite.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(suite)
        if relative.parts[0] in {"splitpoint_runners", "b003", "deepx", "resources"}:
            continue
        if path.suffix.lower() not in {".json", ".csv", ".txt", ".log", ".bin", ".md"}:
            continue
        if path.stat().st_size > 16 * 1024 * 1024:
            continue
        dest = target / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)


def run_smoke(request, work):
    if request.get("model_id") != "yolo11l" or request.get("setup_id") != "orin_nx_deepx_m1_01":
        raise ValueError("H2_is_fixed_to_yolo11l_orin_nx_deepx_m1_01")
    suite, _, image, _ = validate_stage(request, work)
    native_root = work / "native_root"
    destination = native_root / "yolo11l/benchmark_set"
    destination.parent.mkdir(parents=True)
    relative_image = image.relative_to(suite)
    shutil.move(str(suite), destination)
    suite, image = destination, destination / relative_image
    output = work / "results"
    output.mkdir(exist_ok=True)
    binding = prepare_exact_input(request, work, suite, image)
    # Use the ordinary generator to bind __BENCH_JSON__ and vendor its complete
    # runtime dependency set. This touches only this fresh diagnostic suite.
    from onnx_splitpoint_tool.gui.controller import write_benchmark_suite_script
    write_benchmark_suite_script(suite, bench_json_name="benchmark_set.json")
    write_json(output / "input_replay_binding.json", binding)
    write_json(output / "image_map.json", {"yolo11l": {"b003": str(image), "full": str(image)}})
    command = [sys.executable, "-B", str(work / "tool/scripts/native_full_baseline_eval_runner.py"),
        "--root", str(native_root), "--models", "yolo11l", "--backends", "deepx", "--frames", str(FRAMES),
        "--warmup", str(WARMUP), "--repetitions", "1", "--setup-id", request["setup_id"],
        "--comparison-backend", "deepx", "--dump-outputs", "--engine-build-python", sys.executable,
        "--timeout", "120", "--image-map", str(output / "image_map.json"), "--out-dir", str(output)]
    write_json(output / "normal_runner_command.json", {"argv": command, "diagnostic_only": True})
    process = supervised_run(command, cwd=work / "tool", log_path=output / "normal_runner_console.log")
    write_json(output / "process_supervision.json", process)
    collect_suite_diagnostics(suite, output)
    report = output / "analysis_tables/native_full_baseline_eval.json"
    data = read_json(report) if report.is_file() else {}
    result = summarize_native_result(data, process)
    result.update(schema=SCHEMA, diagnostic_only=True, counts_as_benchmark=False,
        quality_evaluated=False, energy_measured=False, compiler_invoked=False,
        normal_runner_invoked=True, source_results_modified=False,
        input_replay_binding=binding, process=process)
    write_json(output / "deepx_full_workflow_smoke.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--runtime", action="store_true")
    args = parser.parse_args(argv)
    request_path = args.request.resolve(strict=True)
    work = request_path.parent
    output = work / "results"
    output.mkdir(exist_ok=True)
    try:
        request = read_json(request_path)
        env = dict(os.environ)
        for key in ("PYTHONPATH", "PYTHONHOME"):
            env.pop(key, None)
        env.update(PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg", MPLCONFIGDIR=str(work / "matplotlib"),
                   ONNX_SPLITPOINT_ARTIFACT_POLICY="cache_verify_only")
        if not args.runtime:
            python = Path(request["runtime_venv"]).expanduser() / "bin/python"
            if not python.is_file() or not os.access(python, os.X_OK):
                raise ValueError("existing_deepx_runtime_python_missing:" + str(python))
            os.execve(str(python), [str(python), "-I", "-B", str(Path(__file__).resolve()),
                      "--request", str(request_path), "--runtime"], env)
        for key in ("PYTHONPATH", "PYTHONHOME"):
            os.environ.pop(key, None)
        os.environ.update(env)
        result = run_smoke(request, work)
        print("NORMAL_FULL_SMOKE_STATUS=" + result["status"], flush=True)
        return 0 if result["status"] == "pass" else 2
    except Exception as exc:
        result = {"schema": SCHEMA, "status": "setup_or_runtime_failed", "diagnostic_only": True,
                  "counts_as_benchmark": False, "error": f"{type(exc).__name__}: {exc}"}
        write_json(output / "deepx_full_workflow_smoke.json", result)
        (output / "worker_traceback.log").write_text(traceback.format_exc(), encoding="utf-8")
        print("NORMAL_FULL_SMOKE_ERROR=" + result["error"], file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
