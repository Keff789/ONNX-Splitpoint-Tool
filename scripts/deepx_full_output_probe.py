#!/usr/bin/env python3
"""Collect a short Full-DXNN output probe using an existing D run and SSH."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import zipfile

DEFAULT_RUN = "v27926_acceptance_d_yolo11l_b003_deepx_gpu_20260907_101443"
REQUEST_SCHEMA = "onnx-splitpoint/deepx-full-output-probe-request/v1"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_request(run_dir: Path, model: str = "yolo11l") -> tuple[dict, dict]:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", model):
        raise ValueError("invalid_model_id")
    results = run_dir / "models" / model / "benchmark_results"
    raw = read_json(results / "benchmark_results_deepx_m1_full_auto.json")
    rows = raw if isinstance(raw, list) else raw.get("results", raw.get("rows", []))
    rows = [row for row in rows if row.get("run_id") == "deepx_m1_full"
            and row.get("backend") == "deepx_m1" and row.get("variant") == "full"]
    if len(rows) != 1:
        raise ValueError("original_deepx_full_result_not_unique")
    row = rows[0]
    dxnn = PurePosixPath(str(row.get("dxnn_path") or ""))
    if not dxnn.is_absolute() or dxnn.parts[-4:] != ("deepx", "deepx_m1", "full", "model.dxnn"):
        raise ValueError("original_remote_dxnn_path_invalid")
    remote_suite = dxnn.parents[3]
    image = PurePosixPath(str((row.get("deepx_prepared_feed_benchmark") or {}).get("image") or ""))
    if not image.is_absolute() or not image.is_relative_to(remote_suite):
        raise ValueError("original_prepared_feed_image_missing_or_outside_suite")
    suite = run_dir / "models" / model / "benchmark_set/legacy_suite"
    contract = read_json(suite / "deepx/deepx_m1/full/output_contract.json")
    digest = str(contract.get("suite_artifact_sha256") or contract.get("artifact_sha256") or "")
    if not re.fullmatch(r"[a-f0-9]{64}", digest):
        raise ValueError("original_full_dxnn_digest_missing")
    if contract.get("model_id") != model or contract.get("endpoint_mode") != "decoded_pre_nms":
        raise ValueError("probe_requires_original_decoded_pre_nms_model_contract")
    plan = read_json(suite / "benchmark_plan.json")
    plans = [r for r in plan.get("runs", []) if r.get("id") == "deepx_m1_full"]
    if len(plans) != 1:
        raise ValueError("original_full_plan_not_unique")
    setup_id = str(plans[0].get("setup_id") or plans[0].get("expected_setup_id") or "")
    matrix = read_json(run_dir / "hardware_matrix.json")
    targets = [r for r in matrix.get("hardware_targets", []) if r.get("id") == setup_id]
    if len(targets) != 1 or targets[0].get("accelerator") != "deepx_m1":
        raise ValueError("original_deepx_setup_not_unique")
    target = targets[0]
    remote = target.get("runtime") or target.get("remote") or {}
    host, user = str(remote.get("host") or ""), str(remote.get("user") or "")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]*", host) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]*", user):
        raise ValueError("original_ssh_destination_invalid")
    port = int(remote.get("port") or 22)
    if not 1 <= port <= 65535:
        raise ValueError("original_ssh_port_invalid")
    venv = str((target.get("build_environment") or {}).get("runtime_venv") or "")
    if not venv:
        activate = shlex.split(str(remote.get("remote_venv") or ""))
        if len(activate) != 2 or activate[0] not in {"source", "."} or not activate[1].endswith("/bin/activate"):
            raise ValueError("original_runtime_venv_not_resolved")
        venv = activate[1][:-len("/bin/activate")]
    if not (venv.startswith("/") or venv.startswith("~/")) or any(c in venv for c in "\r\n\0"):
        raise ValueError("original_runtime_venv_invalid")
    request = {
        "schema": REQUEST_SCHEMA, "original_run_id": run_dir.name,
        "model_id": model, "setup_id": setup_id,
        "remote_suite": str(remote_suite), "dxnn_path": str(dxnn),
        "image_path": str(image), "expected_dxnn_sha256": digest,
        "runtime_venv": venv,
        "original_error": row.get("error_detail"),
    }
    return request, {"host": host, "user": user, "port": port}


def payload_files(payload: Path | None) -> dict[str, Path]:
    source = Path(__file__).resolve().parents[1]
    if payload is not None:
        mapping = {name: payload / name for name in (
            "deepx_full_output_probe_worker.py", "benchmark_suite.py.txt",
            "native_detection_postprocess.py")}
    else:
        mapping = {
            "deepx_full_output_probe_worker.py": Path(__file__).with_name("deepx_full_output_probe_worker.py"),
            "benchmark_suite.py.txt": source / "onnx_splitpoint_tool/resources/templates/benchmark_suite.py.txt",
            "native_detection_postprocess.py": source / "onnx_splitpoint_tool/native_detection_postprocess.py",
        }
    for name, path in mapping.items():
        if not path.is_file():
            raise ValueError(f"probe_payload_missing:{name}")
    return mapping


def ssh_base(remote: dict) -> list[str]:
    return ["ssh", "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=10",
            "-o", "ServerAliveCountMax=3", "-p", str(remote["port"]),
            remote["user"] + "@" + remote["host"]]


def scp_base(remote: dict) -> list[str]:
    return ["scp", "-q", "-o", "ConnectTimeout=15", "-P", str(remote["port"])]


def remote_temp_valid(path: str) -> bool:
    return re.fullmatch(r"/tmp/onnx-v27927-full-probe-[A-Za-z0-9]{10}", path) is not None


def make_archive(output: Path) -> Path:
    archive = output.with_suffix(".zip")
    files = [p for p in sorted(output.rglob("*")) if p.is_file() and not p.is_symlink()]
    if sum(p.stat().st_size for p in files) > 64 * 1024 * 1024:
        raise ValueError("probe_output_exceeds_64_mib")
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in files:
            bundle.write(path, str(path.relative_to(output)))
    with zipfile.ZipFile(archive) as bundle:
        bad = bundle.testzip()
        if bad:
            raise ValueError("probe_archive_crc_failed:" + bad)
    return archive


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path.home() / "Models/EvaluationRuns" / DEFAULT_RUN)
    parser.add_argument("--model", default="yolo11l")
    parser.add_argument("--payload-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path.home() / "Downloads")
    parser.add_argument("--plan-only", action="store_true", help="Resolve original artifacts and show the exact probe; no SSH or inference.")
    args = parser.parse_args(argv)
    request, remote = resolve_request(args.run_dir.expanduser().resolve(), args.model)
    payload = payload_files(args.payload_dir)
    print("MODEL=" + request["model_id"] + " SETUP=" + request["setup_id"], flush=True)
    print("IMAGE=" + request["image_path"], flush=True)
    print("DXNN=" + request["dxnn_path"], flush=True)
    print("MODE=single_input_output_diagnostic; no compile; no campaign; original files preserved", flush=True)
    if args.plan_only:
        print(json.dumps(request, indent=2))
        return 0
    for executable in ("ssh", "scp"):
        if not shutil.which(executable):
            raise ValueError("required_executable_missing:" + executable)
    destination = args.output_dir.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = Path(tempfile.mkdtemp(prefix=f"deepx_full_probe_v27927_{stamp}_", dir=destination))
    request_path = output / "probe_request.json"
    request_path.write_text(json.dumps(request, indent=2) + "\n", encoding="utf-8")
    summary = {"diagnostic_only": True, "status": "setup_started", "remote_returncode": None}
    remote_dir = ""
    collected = False
    try:
        # mktemp, copy and timeout operate solely on the newly allocated probe directory.
        made = subprocess.run(ssh_base(remote) + ["mktemp -d /tmp/onnx-v27927-full-probe-XXXXXXXXXX"],
                              text=True, capture_output=True, timeout=45, check=True)
        remote_dir = made.stdout.strip()
        if not remote_temp_valid(remote_dir):
            raise ValueError("remote_probe_directory_invalid")
        target = remote["user"] + "@" + remote["host"] + ":" + remote_dir + "/"
        subprocess.run(scp_base(remote) + [str(p) for p in payload.values()] + [str(request_path), target],
                       timeout=90, check=True)
        command = shlex.join([
            "timeout", "--signal=TERM", "--kill-after=5s", "90s", "python3", "-I", "-B",
            remote_dir + "/deepx_full_output_probe_worker.py", "--request", remote_dir + "/probe_request.json",
        ])
        print("Full-Probe läuft; maximal 90 Sekunden Remote-Laufzeit.", flush=True)
        with (output / "remote_console.log").open("w", encoding="utf-8") as log:
            completed = subprocess.run(ssh_base(remote) + [command], stdout=log,
                                       stderr=subprocess.STDOUT, timeout=125)
        summary["remote_returncode"] = completed.returncode
        subprocess.run(scp_base(remote) + ["-r", target + "results", str(output / "results")],
                       timeout=90, check=True)
        result_file = output / "results/deepx_output_value_probe.json"
        result = read_json(result_file)
        if not isinstance(result, dict) or not isinstance(result.get("status"), str):
            raise ValueError("probe_result_invalid")
        collected = True
        summary.update(status="collected", probe_status=result.get("status"), probe_error=result.get("error"))
        print("PROBE_STATUS=" + str(result.get("status")), flush=True)
        if result.get("error"):
            print("PROBE_ERROR=" + str(result["error"]), flush=True)
    except Exception as exc:
        summary.update(status="collection_failed", error=f"{type(exc).__name__}: {exc}", remote_probe_dir=remote_dir)
        print("PROBE_COLLECTION_ERROR=" + summary["error"], file=sys.stderr, flush=True)
    finally:
        if collected and remote_temp_valid(remote_dir):
            try:
                cleanup = subprocess.run(ssh_base(remote) + [shlex.join(["rm", "-rf", "--", remote_dir])],
                                         text=True, capture_output=True, timeout=30)
                summary["remote_temporary_directory_removed"] = cleanup.returncode == 0
            except (OSError, subprocess.TimeoutExpired) as exc:
                summary["remote_temporary_directory_removed"] = False
                summary["cleanup_error"] = f"{type(exc).__name__}: {exc}"
                summary["remote_probe_dir"] = remote_dir
        (output / "collection_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        archive = make_archive(output)
        print("DIAGNOSTIC_ZIP=" + str(archive), flush=True)
    return 0 if collected else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError) as exc:
        print("STOP: " + str(exc), file=sys.stderr)
        raise SystemExit(2)
