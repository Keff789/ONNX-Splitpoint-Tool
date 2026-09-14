#!/usr/bin/env python3
"""Stage v2.79.30 and run one bounded normal DeepX Full diagnostic repetition."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import traceback
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import deepx_full_output_probe_v27930 as probe

SOURCE_VERSION = "2.79.30"
SOURCE_TAG = "v27930"
SOURCE_NAME = "ONNX-Splitpoint-Tool_v2.79.30_SOURCE.zip"
RESULT_NAME = "deepx_full_workflow_smoke.json"


def run_transport(command, *, timeout, check=False, capture_output=False, **kwargs):
    """Bound an SSH/SCP process group, including scp's own ssh child."""
    if capture_output:
        kwargs.update(stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    with subprocess.Popen(command, start_new_session=True, **kwargs) as process:
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except BaseException:
            for sig, grace in ((signal.SIGTERM, 5), (signal.SIGKILL, 5)):
                try:
                    os.killpg(process.pid, sig)
                except ProcessLookupError:
                    pass
                try:
                    process.communicate(timeout=grace)
                except subprocess.TimeoutExpired:
                    continue
                # Always send the final KILL to the owned group: the direct
                # process may have exited while a child ignores TERM.
            process.wait(timeout=5)
            raise
        result = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
        if check:
            result.check_returncode()
        return result


def _is_source_payload(relative):
    token = PurePosixPath(relative)
    if not token.parts or token.parts[0] not in {"onnx_splitpoint_tool", "scripts"}:
        return False
    return "__pycache__" not in token.parts and token.suffix.lower() in {".py", ".json", ".txt", ".yaml", ".yml"}


def extract_source(archive, destination):
    """Extract the packaged source dependency closure into temporary staging."""
    with zipfile.ZipFile(archive) as bundle:
        names = bundle.namelist()
        release_names = [n for n in names if n.endswith("/onnx_splitpoint_tool/release_identity.py")]
        if len(release_names) != 1:
            raise ValueError("v27930_source_archive_root_not_unique")
        prefix = release_names[0][:-len("onnx_splitpoint_tool/release_identity.py")]
        count, size = 0, 0
        for member in bundle.infolist():
            if not member.filename.startswith(prefix) or member.is_dir():
                continue
            relative = probe.safe_relative(member.filename[len(prefix):])
            if not _is_source_payload(relative):
                continue
            size += member.file_size
            count += 1
            if size > 128 * 1024 * 1024 or count > 5000:
                raise ValueError("source_runtime_payload_too_large")
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(bundle.read(member))
    return destination


def source_root_for(args, temporary):
    if args.source_root:
        source = args.source_root.expanduser().resolve(strict=True)
    else:
        archive = args.bundle_dir.expanduser() / SOURCE_NAME
        if not archive.is_file():
            raise ValueError("packaged_v27930_source_missing:" + str(archive))
        source = extract_source(archive, temporary / "source")
    identity = source / "onnx_splitpoint_tool/release_identity.py"
    if not identity.is_file() or not re.search(r'VERSION\s*=\s*["\']' + re.escape(SOURCE_VERSION) + r'["\']', identity.read_text()):
        raise ValueError("normal_workflow_smoke_requires_source:" + SOURCE_VERSION)
    return source


def smoke_staging_files(request, sources, source, run_dir):
    files = probe.staging_files(request, sources)
    # Historical top-level measurements are provenance, never candidates for a
    # new successful result. Only original configuration/code is staged.
    for relative in list(files):
        if PurePosixPath(relative).name.startswith(("benchmark_results_", "native_full_", "deepx_prepared_feed_benchmark")):
            del files[relative]
    suite = sources["suite"]
    native = run_dir / "native_producers/deepx/yolo11l/benchmark_set"
    roots = [suite, suite.parent, native]
    for relative in ("benchmark_set.json", "b003/run_split_onnxruntime.py"):
        selected = probe.choose_file([root / relative for root in roots], role="normal_workflow_" + relative.replace("/", "_"))
        files["suite/" + relative] = selected
    original = probe.choose_file([root / "results/deepx_m1_full/prepared_input/native_full_input_manifest.json" for root in roots], role="original_full_input_manifest")
    metadata = probe.read_json(original)
    if (metadata.get("model") != request["model_id"] or metadata.get("setup_id") != request["setup_id"]
            or metadata.get("backend") != "native_full_deepx" or metadata.get("comparison_backend") != "deepx"
            or metadata.get("input_image_sha256") != request["expected_image_sha256"]):
        raise ValueError("original_full_input_manifest_identity_mismatch")
    files["original_full_input_manifest.json"] = original
    for path in sorted(source.rglob("*")):
        if path.is_file() and not path.is_symlink() and _is_source_payload(str(path.relative_to(source))):
            files["tool/" + str(path.relative_to(source))] = path
    for name in ("native_full_baseline_eval_runner.py", "native_full_semantic_dump.py", "run_benchmark_suite_from_set.py"):
        remote = source / "onnx_splitpoint_tool/resources/remote_scripts" / name
        if not remote.is_file():
            raise ValueError("current_remote_runner_missing:" + str(remote))
        files["tool/scripts/" + name] = remote
    # Generated package must also be this release; preserve its real vendored
    # layout instead of importing a potentially installed historical tool.
    package = source / "onnx_splitpoint_tool"
    for relative in list(files):
        if relative.startswith("suite/splitpoint_runners/"):
            del files[relative]
    for path in sorted((package / "runners").rglob("*")):
        if path.is_file() and not path.is_symlink() and path.suffix in {".py", ".json", ".txt"}:
            files["suite/splitpoint_runners/" + str(path.relative_to(package / "runners"))] = path
    for name in ("native_output_endpoint.py", "native_detection_postprocess.py", "preprocessing_contract.py"):
        files["suite/splitpoint_runners/" + name] = package / name
    files["suite/benchmark_suite.py"] = package / "resources/templates/benchmark_suite.py.txt"
    scripts = source / "scripts"
    for name in ("deepx_full_workflow_smoke_worker_v27930.py", "deepx_full_output_probe_worker_v27930.py"):
        files[name] = scripts / name
    files.pop("deepx_full_output_probe_worker.py", None)
    total = sum(path.stat().st_size for path in files.values())
    if total > 256 * 1024 * 1024 or len(files) > 5000:
        raise ValueError("normal_workflow_stage_too_large")
    return files


def remote_dir_valid(path):
    return re.fullmatch(r"/tmp/onnx-" + re.escape(SOURCE_TAG) + r"-full-workflow-[A-Za-z0-9]{10}", path) is not None


def result_exit_code(collected, remote_code, result):
    return 0 if collected and remote_code == 0 and result.get("status") == "pass" else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path.home() / "Models/EvaluationRuns" / probe.DEFAULT_RUN)
    parser.add_argument("--bundle-dir", type=Path, default=Path.home() / "Downloads" / probe.DEFAULT_BUNDLE)
    parser.add_argument("--source-root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--local-dxnn", type=Path)
    parser.add_argument("--local-image", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path.home() / "Downloads")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args(argv)
    args.model, args.payload_dir = "yolo11l", None
    destination = args.output_dir.expanduser().resolve()
    output = None
    summary = {"diagnostic_only": True, "counts_as_benchmark": False, "status": "setup_started",
               "source_version": SOURCE_VERSION, "remote_returncode": None, "compiler_invoked": False, "source_results_modified": False}
    remote, remote_dir, result, collected = {}, "", {}, False
    if not args.plan_only:
        destination.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = Path(tempfile.mkdtemp(prefix=f"deepx_full_workflow_{SOURCE_TAG}_{stamp}_", dir=destination))
    try:
        with tempfile.TemporaryDirectory(prefix=f"onnx-{SOURCE_TAG}-H2-stage-") as temporary:
            temp = Path(temporary)
            source = source_root_for(args, temp)
            # Resolve the existing original Full artifact/image using the proven
            # collector; current source supplies the execution payload.
            payload = temp / "payload"
            payload.mkdir()
            shutil.copyfile(source / "onnx_splitpoint_tool/resources/templates/benchmark_suite.py.txt", payload / "benchmark_suite.py.txt")
            shutil.copyfile(source / "onnx_splitpoint_tool/native_detection_postprocess.py", payload / "native_detection_postprocess.py")
            args.payload_dir = payload
            request, remote, sources = probe.resolve_inputs(args)
            sources["template"] = source / "onnx_splitpoint_tool/resources/templates/benchmark_suite.py.txt"
            sources["postprocess"] = source / "onnx_splitpoint_tool/native_detection_postprocess.py"
            if request["setup_id"] != "orin_nx_deepx_m1_01":
                raise ValueError("H2_requires_orin_nx_deepx_m1_01")
            files = smoke_staging_files(request, sources, source, args.run_dir)
            plan = {"model": request["model_id"], "setup_id": request["setup_id"],
                    "frames": 100, "warmup": 10, "repetitions": 1, "diagnostic_only": True,
                    "counts_as_benchmark": False, "staged_file_count": len(files),
                    "staged_bytes": sum(p.stat().st_size for p in files.values()),
                    "remote_worker_limit_seconds": 300, "normal_runner_limit_seconds": 180,
                    "transfer_limit_seconds": 300, "ssh_executed": False}
            if args.plan_only:
                print(json.dumps(plan, indent=2))
                return 0
            probe.write_json(output / "smoke_plan.json", plan)
            probe.write_json(output / "probe_request.json", request)
            stage = temp / "stage"
            stage.mkdir()
            probe.prepare_stage(stage, request, files)
            for executable in ("ssh", "scp"):
                if not shutil.which(executable):
                    raise ValueError("required_executable_missing:" + executable)
            made = run_transport(probe.ssh_base(remote) + [f"mktemp -d /tmp/onnx-{SOURCE_TAG}-full-workflow-XXXXXXXXXX"],
                                  text=True, capture_output=True, timeout=45)
            if made.returncode:
                raise RuntimeError("remote_mktemp_failed:" + made.stderr)
            remote_dir = made.stdout.strip()
            if not remote_dir_valid(remote_dir):
                raise ValueError("remote_smoke_directory_invalid")
            summary["remote_directory"] = remote_dir
            host = "[" + remote["host"] + "]" if ":" in remote["host"] else remote["host"]
            target = remote["user"] + "@" + host + ":" + remote_dir + "/"
            print("H2: normal Native Full, 1 repetition, 100 frames +10 warmup; no compile, quality or energy.", flush=True)
            with (output / "transfer_console.log").open("w") as log:
                run_transport(probe.scp_base(remote) + ["-r", *[str(p) for p in sorted(stage.iterdir())], target],
                               stdout=log, stderr=subprocess.STDOUT, timeout=300, check=True)
            command = shlex.join(["timeout", "--signal=TERM", "--kill-after=10s", "300s", "python3", "-I", "-B",
                remote_dir + "/deepx_full_workflow_smoke_worker_v27930.py", "--request", remote_dir + "/probe_request.json"])
            with (output / "remote_console.log").open("w") as log:
                try:
                    completed = run_transport(probe.ssh_base(remote) + [command], stdout=log, stderr=subprocess.STDOUT, timeout=340)
                    summary["remote_returncode"] = completed.returncode
                except subprocess.TimeoutExpired:
                    summary["remote_returncode"] = 124
                    summary["ssh_timeout"] = True
            with (output / "collection_console.log").open("w") as log:
                run_transport(probe.scp_base(remote) + ["-r", target + "results", str(output / "results")],
                               stdout=log, stderr=subprocess.STDOUT, timeout=300, check=True)
            result = probe.read_json(output / "results" / RESULT_NAME)
            if not isinstance(result, dict) or not isinstance(result.get("status"), str):
                raise ValueError("normal_full_smoke_result_invalid")
            collected = True
            summary.update(status="collected", smoke_status=result["status"])
            print("NORMAL_FULL_SMOKE_STATUS=" + result["status"], flush=True)
    except Exception as exc:
        if args.plan_only:
            raise
        summary.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        (output / "collector_traceback.log").write_text(traceback.format_exc(), encoding="utf-8")
        print("H2_ERROR=" + summary["error"], file=sys.stderr, flush=True)
    finally:
        if output is not None:
            if collected and remote_dir_valid(remote_dir) and (result.get("process") or {}).get("cleanup_complete") is True:
                try:
                    cleanup = run_transport(probe.ssh_base(remote) + [shlex.join(["rm", "-rf", "--", remote_dir])], capture_output=True, text=True, timeout=30)
                    summary["temporary_remote_directory_removed"] = cleanup.returncode == 0
                except (OSError, subprocess.TimeoutExpired) as exc:
                    summary["cleanup_error"] = str(exc)
            summary["exit_code"] = result_exit_code(collected, summary["remote_returncode"], result)
            probe.write_json(output / "collection_summary.json", summary)
            print("DIAGNOSTIC_ZIP=" + str(probe.make_archive(output)), flush=True)
    return summary["exit_code"]


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError) as exc:
        print("STOP: " + str(exc), file=sys.stderr)
        raise SystemExit(2)
