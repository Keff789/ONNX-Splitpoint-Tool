#!/usr/bin/env python3
"""Stage the existing Full DXNN locally, then run the v2.79.28 output diagnostic.

No compiler, installation, campaign or cache population is performed. The old
remote run paths are provenance only; the worker uses its new temporary suite.
Management-side dependencies: Python >=3.10 stdlib, OpenSSH ssh/scp.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import traceback
import zipfile

DEFAULT_RUN = "v27926_acceptance_d_yolo11l_b003_deepx_gpu_20260907_101443"
DEFAULT_BUNDLE = "ONNX-Splitpoint-Tool_v2.79.28_COMPLETE_DELIVERY_BUNDLE"
REQUEST_SCHEMA = "onnx-splitpoint/deepx-full-output-probe-request/v1"
STAGING_MODE = "local_files_to_fresh_remote_directory"
DXNN_RELATIVE = "deepx/deepx_m1/full/model.dxnn"
CONTRACT_RELATIVE = "deepx/deepx_m1/full/output_contract.json"
MAX_STAGE_BYTES = 256 * 1024 * 1024


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def valid_digest(value: object) -> str:
    digest = str(value or "").removeprefix("sha256:").lower()
    if not re.fullmatch(r"[a-f0-9]{64}", digest):
        raise ValueError("original_artifact_digest_invalid")
    return digest


def safe_relative(value: str) -> str:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value or any(c in value for c in "\r\n\0"):
        raise ValueError("unsafe_relative_path:" + value)
    return str(path)


def resolve_metadata(run_dir: Path, model: str = "yolo11l") -> tuple[dict, dict, dict]:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", model):
        raise ValueError("invalid_model_id")
    suite = run_dir / "models" / model / "benchmark_set/legacy_suite"
    raw = read_json(run_dir / "models" / model / "benchmark_results/benchmark_results_deepx_m1_full_auto.json")
    rows = raw if isinstance(raw, list) else raw.get("results", raw.get("rows", []))
    rows = [row for row in rows if row.get("run_id") == "deepx_m1_full"
            and row.get("backend") == "deepx_m1" and row.get("variant") == "full"]
    if len(rows) != 1:
        raise ValueError("original_deepx_full_result_not_unique")
    row = rows[0]
    old_dxnn = PurePosixPath(str(row.get("dxnn_path") or ""))
    if not old_dxnn.is_absolute() or old_dxnn.parts[-4:] != tuple(PurePosixPath(DXNN_RELATIVE).parts):
        raise ValueError("original_remote_dxnn_path_invalid")
    old_suite = old_dxnn.parents[3]  # Lexical only; NEVER resolve/stat the old remote directory.
    old_image = PurePosixPath(str((row.get("deepx_prepared_feed_benchmark") or {}).get("image") or ""))
    if not old_image.is_absolute() or not old_image.is_relative_to(old_suite):
        raise ValueError("original_prepared_feed_image_missing_or_outside_suite")
    image_relative = safe_relative(str(old_image.relative_to(old_suite)))
    contract = read_json(suite / CONTRACT_RELATIVE)
    digest = valid_digest(contract.get("suite_artifact_sha256") or contract.get("artifact_sha256"))
    if contract.get("artifact_sha256") and valid_digest(contract["artifact_sha256"]) != digest:
        raise ValueError("original_full_dxnn_contract_digest_conflict")
    if contract.get("model_id") != model or contract.get("endpoint_mode") != "decoded_pre_nms":
        raise ValueError("probe_requires_original_decoded_pre_nms_model_contract")
    plan = read_json(suite / "benchmark_plan.json")
    runs = [r for r in plan.get("runs", []) if r.get("id") == "deepx_m1_full"]
    if len(runs) != 1:
        raise ValueError("original_full_plan_not_unique")
    run = runs[0]
    if safe_relative(str(run.get("dxnn_path") or DXNN_RELATIVE)) != DXNN_RELATIVE:
        raise ValueError("original_full_plan_dxnn_path_mismatch")
    if safe_relative(str(run.get("contract_path") or CONTRACT_RELATIVE)) != CONTRACT_RELATIVE:
        raise ValueError("original_full_plan_contract_path_mismatch")
    setup_id = str(run.get("setup_id") or run.get("expected_setup_id") or "")
    targets = [r for r in read_json(run_dir / "hardware_matrix.json").get("hardware_targets", []) if r.get("id") == setup_id]
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
    # Retain the established SSH command contract; do not execute extra shell text.
    if str(remote.get("ssh_extra_args") or "").strip():
        raise ValueError("nonempty_ssh_extra_args_not_supported_by_this_probe")
    venv = str((target.get("build_environment") or {}).get("runtime_venv") or "")
    if not venv:
        activate = shlex.split(str(remote.get("remote_venv") or ""))
        if len(activate) != 2 or activate[0] not in {"source", "."} or not activate[1].endswith("/bin/activate"):
            raise ValueError("original_runtime_venv_not_resolved")
        venv = activate[1][:-len("/bin/activate")]
    if not (venv.startswith("/") or venv.startswith("~/")) or any(c in venv for c in "\r\n\0"):
        raise ValueError("original_runtime_venv_invalid")
    request = {
        "schema": REQUEST_SCHEMA, "staging_mode": STAGING_MODE,
        "probe_fix": "v27928", "original_run_id": run_dir.name,
        "model_id": model, "setup_id": setup_id,
        "original_remote_suite": str(old_suite), "original_dxnn_path": str(old_dxnn),
        "original_image_path": str(old_image), "staged_dxnn_relative": DXNN_RELATIVE,
        "staged_image_relative": image_relative, "expected_dxnn_sha256": digest,
        "runtime_venv": venv, "original_error": row.get("error_detail"),
        "model_acceptance": "not_evaluated_by_diagnostic",
    }
    return request, {"host": host, "user": user, "port": port}, {"suite": suite, "run": run, "contract": contract}


def choose_file(candidates: list[Path], *, expected_digest: str = "", role: str) -> Path:
    tried: list[str] = []
    seen: set[str] = set()
    for path in candidates:
        path = path.expanduser().absolute()
        if str(path) in seen:
            continue
        seen.add(str(path))
        if not path.is_file():
            tried.append(str(path) + " [missing]")
            continue
        if path.stat().st_size <= 0:
            tried.append(str(path) + " [empty]")
            continue
        if expected_digest and sha256(path) != expected_digest:
            tried.append(str(path) + " [sha256 mismatch]")
            continue
        return path.resolve(strict=True)
    raise ValueError(role + "_not_available; checked: " + "; ".join(tried))


def image_candidates(run_dir: Path, suite: Path, request: dict, run: dict) -> tuple[list[Path], str]:
    relative = request["staged_image_relative"]
    basename = PurePosixPath(relative).name
    candidates = [suite / relative, suite.parent / relative]
    manifests = [run_dir / "campaign/inputs/dataset_detection_validation.json"]
    if run.get("validation_manifest"):
        manifests.append(Path(run["validation_manifest"]).expanduser())
    expected: set[str] = set()
    for path in manifests:
        if not path.is_file():
            continue
        data = read_json(path)
        matches = [item for item in data.get("items", [])
                   if PurePosixPath(str(item.get("relative_path") or "")).name == basename]
        if len(matches) > 1:
            raise ValueError("original_validation_image_not_unique:" + str(path))
        for item in matches:
            if item.get("sha256"):
                expected.add(valid_digest(item["sha256"]))
            if data.get("root"):
                item_path = safe_relative(str(item["relative_path"]))
                candidates.append(Path(data["root"]).expanduser() / item_path)
    if len(expected) > 1:
        raise ValueError("original_validation_image_digest_conflict")
    return candidates, next(iter(expected), "")


def resolve_inputs(args: argparse.Namespace) -> tuple[dict, dict, dict[str, Path]]:
    run_dir = args.run_dir.expanduser().resolve(strict=True)
    request, remote, meta = resolve_metadata(run_dir, args.model)
    suite, contract = meta["suite"], meta["contract"]
    dxnn_candidates = [suite / DXNN_RELATIVE]
    dxnn_candidates.extend(Path(contract[k]) for k in ("artifact_path", "suite_artifact_path") if contract.get(k))
    # An explicit path is an alternative location, never permission to use another model.
    if args.local_dxnn:
        dxnn_candidates = [args.local_dxnn]
    dxnn = choose_file(dxnn_candidates, expected_digest=request["expected_dxnn_sha256"], role="existing_full_dxnn")
    images, image_digest = image_candidates(run_dir, suite, request, meta["run"])
    if args.local_image:
        if args.local_image.name != PurePosixPath(request["staged_image_relative"]).name:
            raise ValueError("explicit_image_must_be_the_original_image")
        images = [args.local_image]
    image = choose_file(images, expected_digest=image_digest, role="original_validation_image")
    package = suite / "splitpoint_runners"
    for name in ("__init__.py", "harness/base.py", "harness/yolo.py"):
        if not (package / name).is_file():
            raise ValueError("local_generated_runner_missing:" + str(package / name))
    here = Path(__file__).resolve().parent
    if args.payload_dir:
        payload = args.payload_dir.expanduser().resolve()
    elif (here / "benchmark_suite.py.txt").is_file():
        payload = here
    elif (args.bundle_dir.expanduser() / "probe").is_dir():
        payload = args.bundle_dir.expanduser().resolve() / "probe"
    else:
        payload = here.parent / "onnx_splitpoint_tool/resources/templates"
    template = choose_file([payload / "benchmark_suite.py.txt"], role="v27928_probe_template")
    tree = ast.parse(template.read_text(encoding="utf-8-sig"), filename=str(template))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_deepx_output_value_probe"]
    if len(functions) != 1 or len(functions[0].args.posonlyargs + functions[0].args.args) != 5:
        raise ValueError("v27928_template_probe_api_missing_or_changed")
    here = Path(__file__).resolve().parent
    worker = here / "deepx_full_output_probe_worker_v27928.py"
    postprocess = payload / "native_detection_postprocess.py"
    if not postprocess.is_file():
        postprocess = here.parent / "onnx_splitpoint_tool/native_detection_postprocess.py"
    for path in (worker, postprocess):
        if not path.is_file():
            raise ValueError("fix_payload_missing:" + str(path))
    request.update(expected_image_sha256=image_digest or sha256(image),
                   image_identity_source="original_validation_manifest" if image_digest else "local_original_relative_path_and_transfer_digest")
    sources = {"suite": suite, "dxnn": dxnn, "image": image, "package": package,
               "template": template, "worker": worker, "postprocess": postprocess}
    request["local_sources"] = {key: str(value) for key, value in sources.items()}
    return request, remote, sources


def staging_files(request: dict, sources: dict[str, Path]) -> dict[str, Path]:
    """Minimal standalone suite: original metadata/runners, one DXNN, one image."""
    suite, package = sources["suite"], sources["package"]
    files: dict[str, Path] = {}
    for path in sorted(suite.iterdir()):
        if path.is_file() and path.suffix.lower() in {".py", ".json", ".yaml", ".yml"}:
            files["suite/" + path.name] = path
    for path in sorted((suite / "deepx/deepx_m1/full").iterdir()):
        if path.is_file() and path.suffix.lower() in {".json", ".yaml", ".yml", ".txt"}:
            files["suite/deepx/deepx_m1/full/" + path.name] = path
    for path in sorted(package.rglob("*")):
        if "__pycache__" in path.parts or not path.is_file():
            continue
        if path.suffix.lower() not in {".py", ".json", ".yaml", ".yml", ".txt"}:
            continue
        if not path.resolve(strict=True).is_relative_to(package.resolve(strict=True)):
            raise ValueError("runner_file_outside_generated_package:" + str(path))
        files["suite/splitpoint_runners/" + str(path.relative_to(package))] = path
    files.update({
        "suite/" + DXNN_RELATIVE: sources["dxnn"],
        "suite/" + request["staged_image_relative"]: sources["image"],
        "suite/benchmark_suite.py": sources["template"],
        "suite/splitpoint_runners/native_detection_postprocess.py": sources["postprocess"],
        "deepx_full_output_probe_worker.py": sources["worker"],
    })
    for name in ("suite/benchmark_plan.json", "suite/" + CONTRACT_RELATIVE):
        if name not in files:
            raise ValueError("required_suite_metadata_missing:" + name)
    total = sum(p.stat().st_size for p in files.values())
    if total > MAX_STAGE_BYTES or len(files) > 1000:
        raise ValueError(f"probe_payload_too_large: files={len(files)} bytes={total}")
    return files


def prepare_stage(stage: Path, request: dict, files: dict[str, Path]) -> None:
    for relative, source in files.items():
        target = stage / safe_relative(relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)  # Real copies; no hardlinks/symlinks into original files.
    if sha256(stage / "suite" / DXNN_RELATIVE) != request["expected_dxnn_sha256"]:
        raise ValueError("staged_dxnn_sha256_mismatch")
    if sha256(stage / "suite" / request["staged_image_relative"]) != request["expected_image_sha256"]:
        raise ValueError("staged_image_sha256_mismatch")
    write_json(stage / "probe_request.json", request)


def ssh_base(remote: dict) -> list[str]:
    return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=10",
            "-o", "ServerAliveCountMax=3", "-p", str(remote["port"]), remote["user"] + "@" + remote["host"]]


def scp_base(remote: dict) -> list[str]:
    return ["scp", "-q", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-P", str(remote["port"])]


def remote_temp_valid(path: str) -> bool:
    return re.fullmatch(r"/tmp/onnx-v27928-full-probe-[A-Za-z0-9]{10}", path) is not None


def make_archive(output: Path) -> Path:
    archive = output.with_suffix(".zip")
    files = [p for p in sorted(output.rglob("*")) if p.is_file() and not p.is_symlink()]
    if sum(p.stat().st_size for p in files) > 64 * 1024 * 1024:
        raise ValueError("probe_output_exceeds_64_mib; local diagnostics preserved:" + str(output))
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in files:
            bundle.write(path, str(path.relative_to(output)))
    with zipfile.ZipFile(archive) as bundle:
        bad = bundle.testzip()
        if bad:
            raise ValueError("probe_archive_crc_failed:" + bad)
    return archive


def exit_code(*, collected: bool, remote_returncode: int | None, result: dict) -> int:
    # Transport success MUST NOT conceal a worker/setup error as before.
    return 0 if collected and remote_returncode == 0 and result.get("status") not in {None, "probe_setup_failed", "probe_timeout"} else 2


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path.home() / "Models/EvaluationRuns" / DEFAULT_RUN)
    parser.add_argument("--model", default="yolo11l")
    parser.add_argument("--bundle-dir", type=Path, default=Path.home() / "Downloads" / DEFAULT_BUNDLE)
    parser.add_argument("--payload-dir", type=Path, help="Directory containing the ORIGINAL v2.79.28 benchmark_suite.py.txt.")
    parser.add_argument("--local-dxnn", type=Path, help="Explicit existing copy; original model checksum must match.")
    parser.add_argument("--local-image", type=Path, help="Explicit copy of the SAME original validation image.")
    parser.add_argument("--output-dir", type=Path, default=Path.home() / "Downloads")
    parser.add_argument("--plan-only", action="store_true", help="Check local files and print plan; no SSH, inference or output writes.")
    args = parser.parse_args(argv)
    if args.plan_only:
        request, remote, sources = resolve_inputs(args)
        files = staging_files(request, sources)
        print(json.dumps({"request": request, "remote": remote, "staged_file_count": len(files),
                          "staged_bytes": sum(p.stat().st_size for p in files.values()), "ssh_executed": False}, indent=2))
        return 0
    destination = args.output_dir.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = Path(tempfile.mkdtemp(prefix=f"deepx_full_probe_v27928_{stamp}_", dir=destination))
    summary = {"diagnostic_only": True, "probe_fix": "v27928", "status": "setup_started",
               "remote_returncode": None, "source_results_modified": False, "compiler_invoked": False,
               "model_acceptance": "not_evaluated_by_diagnostic"}
    remote_dir, collected, result, remote = "", False, {}, {}
    try:
        request, remote, sources = resolve_inputs(args)
        files = staging_files(request, sources)
        write_json(output / "probe_request.json", request)
        for executable in ("ssh", "scp"):
            if not shutil.which(executable):
                raise ValueError("required_executable_missing:" + executable)
        print("PROBE_FIX=v27928", flush=True)
        print("MODEL=" + request["model_id"] + " SETUP=" + request["setup_id"], flush=True)
        print("LOCAL_DXNN=" + str(sources["dxnn"]), flush=True)
        print("LOCAL_IMAGE=" + str(sources["image"]), flush=True)
        print("LOCAL_PREFLIGHT=PASS; original DXNN/image verified", flush=True)
        print("MODE=single_input_output_diagnostic; no compile; no campaign; originals unchanged", flush=True)
        # The bulky one-time staging data is NOT kept in the diagnostic ZIP.
        with tempfile.TemporaryDirectory(prefix=".deepx-probe-transfer-", dir=destination) as temporary:
            stage = Path(temporary)
            prepare_stage(stage, request, files)
            with (output / "transfer_console.log").open("w", encoding="utf-8") as transfer_log:
                made = subprocess.run(ssh_base(remote) + ["mktemp -d /tmp/onnx-v27928-full-probe-XXXXXXXXXX"],
                                      text=True, capture_output=True, timeout=45)
                transfer_log.write(made.stderr or "")
                transfer_log.flush()
                if made.returncode != 0:
                    raise RuntimeError(f"remote_mktemp_failed: rc={made.returncode}; {made.stderr.strip()}")
                remote_dir = made.stdout.strip()
                if not remote_temp_valid(remote_dir):
                    raise ValueError("remote_probe_directory_invalid:" + repr(remote_dir))
                summary["remote_probe_dir"] = remote_dir
                host = "[" + remote["host"] + "]" if ":" in remote["host"] else remote["host"]
                target = remote["user"] + "@" + host + ":" + remote_dir + "/"
                print("REMOTE_PROBE_DIR=" + remote_dir, flush=True)
                print("Transfer: one DXNN, one image, runner and contracts (no full dataset).", flush=True)
                subprocess.run(scp_base(remote) + ["-r", str(stage / "suite"), str(stage / "deepx_full_output_probe_worker.py"),
                               str(stage / "probe_request.json"), target], stdout=transfer_log, stderr=subprocess.STDOUT,
                               timeout=300, check=True)
            command = shlex.join(["timeout", "--signal=TERM", "--kill-after=5s", "90s", "python3", "-I", "-B",
                                  remote_dir + "/deepx_full_output_probe_worker.py", "--request", remote_dir + "/probe_request.json"])
            print("Full-Probe läuft: 90 s Remote-Limit (+ höchstens 5 s bis KILL); Transfer zusätzlich.", flush=True)
            with (output / "remote_console.log").open("w", encoding="utf-8") as log:
                completed = subprocess.run(ssh_base(remote) + [command], stdout=log, stderr=subprocess.STDOUT, timeout=125)
            summary["remote_returncode"] = completed.returncode
            with (output / "collection_console.log").open("w", encoding="utf-8") as log:
                subprocess.run(scp_base(remote) + ["-r", target + "results", str(output / "results")],
                               stdout=log, stderr=subprocess.STDOUT, timeout=90, check=True)
        result_file = output / "results/deepx_output_value_probe.json"
        if result_file.is_file():
            result = read_json(result_file)
        elif summary["remote_returncode"] in {124, 137}:
            result = {"diagnostic_only": True, "status": "probe_timeout", "error": "bounded_remote_probe_timed_out"}
            write_json(result_file, result)
        else:
            raise ValueError("probe_result_missing; see remote_console.log")
        if not isinstance(result, dict) or not isinstance(result.get("status"), str):
            raise ValueError("probe_result_invalid")
        collected = True
        summary.update(status="collected", probe_status=result["status"], probe_error=result.get("error"))
        print("PROBE_STATUS=" + result["status"], flush=True)
        if result.get("error"):
            print("PROBE_ERROR=" + str(result["error"]), flush=True)
        normalization = result.get("decoded_pre_nms_score_normalization")
        if isinstance(normalization, dict):
            print("SCORE_NORMALIZATION=" + json.dumps(normalization, sort_keys=True), flush=True)
        if result.get("completed_detection_count") is not None:
            print("COMPLETED_DETECTIONS=" + str(result["completed_detection_count"]), flush=True)
        print("MODEL_ACCEPTANCE=NOT_EVALUATED_BY_DIAGNOSTIC", flush=True)
    except Exception as exc:
        summary.update(status="collection_failed", error=f"{type(exc).__name__}: {exc}")
        (output / "collector_traceback.log").write_text(traceback.format_exc(), encoding="utf-8")
        print("PROBE_COLLECTION_ERROR=" + summary["error"], file=sys.stderr, flush=True)
    finally:
        # Never touch the old run, persistent cache, installation, or vendor venv.
        if collected and remote_temp_valid(remote_dir):
            try:
                cleanup = subprocess.run(ssh_base(remote) + [shlex.join(["rm", "-rf", "--", remote_dir])],
                                         text=True, capture_output=True, timeout=30)
                summary["remote_temporary_directory_removed"] = cleanup.returncode == 0
                if cleanup.returncode:
                    summary["cleanup_error"] = cleanup.stderr.strip()
            except (OSError, subprocess.TimeoutExpired) as exc:
                summary["remote_temporary_directory_removed"] = False
                summary["cleanup_error"] = str(exc)
        summary["collector_exit_code"] = exit_code(collected=collected, remote_returncode=summary["remote_returncode"], result=result)
        write_json(output / "collection_summary.json", summary)
        archive = make_archive(output)
        print("DIAGNOSTIC_ZIP=" + str(archive), flush=True)
    return int(summary["collector_exit_code"])


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError, SyntaxError) as exc:
        print("STOP: " + str(exc), file=sys.stderr)
        raise SystemExit(2)
