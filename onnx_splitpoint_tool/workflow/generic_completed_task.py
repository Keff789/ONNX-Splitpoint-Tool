"""Selective Generic completed-task measurements outside a closed EvaluationRun.

This adapter owns transport and scheduling only. Runtime execution remains in
the generated Generic runner; physical ownership and SSH cleanup use the same
locks and process leases as the normal evaluation workflow.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import inspect
import hashlib
import json
import os
import re
from pathlib import Path, PurePosixPath
import shlex
import shutil
import signal
import tarfile
import threading
import time
import traceback
import uuid

from ..benchmark.schema import write_json_atomic
from ..remote.process_lease import RemoteProcessLeaseRegistry, RemoteProcessLeaseScope
from ..remote.resource_recovery_observation import _observe_processes
from ..remote.ssh_transport import HostConfig, SSHTransport
from .execution_binding import physical_dut_key
from .run_control import EvaluationRunLock


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(Path(path), value)


def case_name(row: dict) -> str:
    identity = row.get("identity", row)
    values = [identity[k] for k in ("model_id", "case_id", "setup_id", "precision")]
    for value in values:
        if not isinstance(value, str) or not value or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for c in value):
            raise ValueError("unsafe completion case identity")
    return "__".join(values)


def outside_primary(path: Path, primary: Path) -> Path:
    path = Path(path).expanduser().resolve()
    primary = Path(primary).expanduser().resolve()
    if path == primary or primary in path.parents:
        raise ValueError("completion output must be outside the closed primary run")
    return path


def bound_targets(primary: Path, rows: list[dict]) -> dict[str, dict]:
    targets = {r["id"]: r for r in read_json(primary / "run_manifest.json")["hardware_targets"]}
    wanted = {(r.get("identity") or r)["setup_id"] for r in rows}
    result = {}
    for setup_id in sorted(wanted):
        target = targets[setup_id]
        remote = target.get("remote") or target.get("runtime") or {}
        if not remote.get("host") or not remote.get("user"):
            raise ValueError("original target binding missing: " + setup_id)
        result[setup_id] = {**target, "remote": remote}
    keys = [physical_dut_key(r) for r in result.values()]
    if len(keys) != len(set(keys)):
        raise ValueError("selected setup aliases share a physical device")
    return result


def transport_for(target: dict, diagnostics: Path, *, scope=None, registry=None, cancel=None, log=print):
    remote = target["remote"]
    host = HostConfig.from_dict({**remote, "id": target["id"], "label": target.get("label", target["id"])})
    transport = SSHTransport(host, log=log, cancel_event=cancel,
                             remote_lease_scope=scope, remote_lease_registry=registry)
    transport.diagnostics_dir = diagnostics
    return transport


def observe_owners(targets: dict[str, dict], output: Path) -> dict:
    """Read-only existing owner probe, without recovery or stopping a process."""
    controller = _observe_processes([])
    result = {"observed_at": now(), "controller": controller, "setups": {}, "ok": False}
    if not controller.get("ok"):
        write_json(output / "owners.json", result)
        raise RuntimeError("controller has competing work or incomplete visibility")
    code = inspect.getsource(_observe_processes)
    command = ("python3 -B - <<'GENERIC_COMPLETION_OWNER_PROBE'\n" + code
               + "\nimport json\nprint('OWNERS=' + json.dumps(_observe_processes([])))\n"
               + "GENERIC_COMPLETION_OWNER_PROBE")
    for setup_id, target in targets.items():
        transport = transport_for(target, output / setup_id)
        rc, stdout = transport.run_read_only(command, timeout=40,
                                             env={"PYTHONDONTWRITEBYTECODE": "1"})
        records = [line[7:] for line in stdout.splitlines() if line.startswith("OWNERS=")]
        observation = json.loads(records[0]) if rc == 0 and len(records) == 1 else {
            "ok": False, "error": "owner observation failed", "rc": rc, "output": stdout[-4000:]}
        result["setups"][setup_id] = observation
        write_json(output / "owners.json", result)
        if not observation.get("ok"):
            raise RuntimeError("device has competing work or incomplete visibility: " + setup_id)
    result["ok"] = True
    write_json(output / "owners.json", result)
    return result


def capture_configuration(transport: SSHTransport, target: dict, *, primary: Path, output: Path, phase: str):
    """Run the existing non-mutating host telemetry collector over normal SSH."""
    from .. import resources_utils
    code = resources_utils.read_text("resources", "remote_scripts", "native_host_telemetry.py")
    namespace = "__generic_configuration_probe__"
    args = {"phase": phase, "backend": str(target.get("accelerator") or ""),
            "setup_id": target["id"], "run_id": primary.name, "capture_group": "generic-completed-task"}
    script = ("import json\nns={'__name__':" + repr(namespace) + "}\nexec(" + repr(code)
              + ", ns)\nprint('CONFIGURATION=' + json.dumps(ns['capture'](**" + repr(args) + ")))\n")
    rc, stdout = transport.run_read_only("python3 -B - <<'GENERIC_CONFIGURATION'\n" + script
                                         + "GENERIC_CONFIGURATION", timeout=60)
    matches = [line[len("CONFIGURATION="):] for line in stdout.splitlines() if line.startswith("CONFIGURATION=")]
    if rc != 0 or len(matches) != 1:
        raise RuntimeError("read-only configuration snapshot failed: " + target["id"])
    payload = json.loads(matches[0])
    write_json(output, payload)
    return payload


def verify_runtime_environment(transport: SSHTransport, row: dict, output: Path):
    python = row["remote_python_executable"]
    expected = row["native_command_contract"]["artifacts"]["python_executable"]["sha256"]
    modules = ["numpy", "onnx", "onnxruntime", "tensorrt", "hailo_platform", "dx_engine"]
    script = ("import sys,json,os,hashlib,importlib.util,importlib.metadata,site\nfrom pathlib import Path\n"
              "for p in os.environ.get('SPLITPOINT_EXTRA_SITES','').split(os.pathsep):\n"
              " if p and os.path.isdir(p):site.addsitedir(p)\n"
              "exe=Path(sys.executable).resolve()\n"
              "result={'executable':sys.executable,'resolved_executable':str(exe),"
              "'executable_sha256':hashlib.sha256(exe.read_bytes()).hexdigest(),'version':sys.version,"
              "'modules':{},'thread_environment':{k:os.environ.get(k) for k in "
              "['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','CUDA_VISIBLE_DEVICES']}}\n"
              "for name in " + repr(modules) + ":\n"
              " spec=importlib.util.find_spec(name)\n result['modules'][name]={'available':spec is not None,'origin':spec.origin if spec else None}\n"
              " try:result['modules'][name]['version']=importlib.metadata.version(name)\n"
              " except importlib.metadata.PackageNotFoundError:pass\n"
              "print('RUNTIME_ENVIRONMENT='+json.dumps(result))\n")
    rc, stdout = transport.run_read_only(shlex.quote(python) + " -B - <<'GENERIC_RUNTIME_ENVIRONMENT'\n"
                                         + script + "GENERIC_RUNTIME_ENVIRONMENT", timeout=40,
                                         env=row.get("remote_environment", {}))
    records = [line.split("=", 1)[1] for line in stdout.splitlines() if line.startswith("RUNTIME_ENVIRONMENT=")]
    if rc or len(records) != 1:
        raise RuntimeError("bound interpreter dependency inspection failed: " + stdout[-2000:])
    result = json.loads(records[0])
    required = ["numpy", "onnx", "onnxruntime", "tensorrt",
                "dx_engine" if row["setup_id"] == "orin_nx_deepx_m1_01" else "hailo_platform"]
    result["required_modules"] = required
    result["matches_original_interpreter"] = result["executable_sha256"] == expected
    result["missing_modules"] = [name for name in required if not result["modules"][name]["available"]]
    write_json(output, result)
    if not result["matches_original_interpreter"] or result["missing_modules"]:
        raise RuntimeError("bound runtime environment unavailable or changed: " + row["setup_id"])
    return result


def checked_remote(command: str, transport: SSHTransport, *, timeout=120):
    rc, output = transport.run(command, timeout=timeout,
                               env={"PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1"})
    if rc:
        raise RuntimeError(f"remote command failed ({rc}): {output[-6000:]}")
    return output


def prepare_runner_bundle(destination: Path) -> Path:
    """Use the product writer in a new session directory, never refresh a run."""
    from ..split_export_runners import write_runner_skeleton_onnxruntime
    from ..gui.controller import _copy_runner_lib
    destination.mkdir(parents=True, exist_ok=False)
    _copy_runner_lib(destination)
    write_runner_skeleton_onnxruntime(str(destination / "runner"))
    bundle = destination.with_suffix(".tar.gz")
    with tarfile.open(bundle, "w:gz") as archive:
        archive.add(destination / "splitpoint_runners", arcname="splitpoint_runners",
                    filter=lambda info: None if "__pycache__" in info.name or info.name.endswith(".pyc") else info)
        archive.add(destination / "runner" / "run_split_onnxruntime.py", arcname="runner/run_split_onnxruntime.py")
    return bundle


def file_sha256(path: Path) -> str:
    with Path(path).open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
        return digest.hexdigest()


def upload_new_file(transport: SSHTransport, local: Path, remote: str, *, expected_sha=""):
    """Copy into this attempt only; existing unequal destination bytes block."""
    actual = file_sha256(local)
    if expected_sha and actual != expected_sha:
        raise ValueError("staging source differs from its original binding: " + str(local))
    probe = ("from pathlib import Path\nimport hashlib,json\np=Path(" + repr(remote) + ")\n"
             "h=hashlib.sha256()\n"
             "if p.is_file():\n"
             " with p.open('rb') as f:\n"
             "  for b in iter(lambda:f.read(8388608),b''):h.update(b)\n"
             "print(json.dumps({'exists':p.exists(),'sha256':h.hexdigest() if p.is_file() else ''}))\n")
    command = "python3 -B - <<'COMPLETION_FILE_PROBE'\n" + probe + "COMPLETION_FILE_PROBE"
    rc, stdout = transport.run_read_only(command, timeout=120)
    if rc:
        raise RuntimeError("staging destination probe failed: " + stdout[-2000:])
    observation = json.loads(stdout)
    if observation["exists"]:
        if observation["sha256"] != actual:
            raise ValueError("existing staging destination differs: " + remote)
        return {"path": remote, "sha256": actual, "status": "reused_exact_bytes"}
    parent = str(PurePosixPath(remote).parent)
    temporary = remote + ".upload-" + uuid.uuid4().hex
    checked_remote("mkdir -p -- " + shlex.quote(parent), transport)
    rc, stdout = transport.scp_upload(str(local), temporary, timeout=600)
    if rc:
        raise RuntimeError("staging transfer failed: " + stdout[-2000:])
    commit = ("from pathlib import Path\nimport hashlib,os\np=Path(" + repr(temporary) + ")\n"
              "h=hashlib.sha256()\nwith p.open('rb') as f:\n"
              " for b in iter(lambda:f.read(8388608),b''):h.update(b)\n"
              "assert h.hexdigest()==" + repr(actual) + ", 'transferred artifact hash mismatch'\n"
              "os.link(p," + repr(remote) + ")\np.unlink()\n")
    checked_remote("python3 -B - <<'COMPLETION_FILE_COMMIT'\n" + commit + "COMPLETION_FILE_COMMIT", transport)
    return {"path": remote, "sha256": actual, "status": "uploaded_exact_bytes"}


def validate_plan(plan: dict, scope_csv: Path, primary: Path) -> list[dict]:
    import csv
    from ..generic_completion_plan import completion_identity
    with Path(scope_csv).open(newline="", encoding="utf-8") as stream:
        source = list(csv.DictReader(stream))
    rows = plan.get("rows")
    if not isinstance(rows, list) or len(rows) != 192 or len(source) != 192:
        raise ValueError("authorized Generic completion scope requires exactly 192 rows")
    def key(row):
        identity = row.get("identity", row)
        return completion_identity(identity)
    expected, actual = [key(r) for r in source], [key(r) for r in rows]
    if len(set(expected)) != 192 or len(set(actual)) != 192 or set(expected) != set(actual):
        raise ValueError("completion plan differs from the authorized 192 identities")
    for row in rows:
        config = row.get("runtime_config", row)
        if key(config) != key(row) or (row.get("identity") and key(row) != completion_identity(row)):
            raise ValueError("completion runtime identity differs from its authorized row")
        if any(config.get(k) != v for k, v in (("repetitions", 3), ("warmup_count", 100), ("task_count", 1000))):
            raise ValueError("completion plan work budget changed")
        if Path(config.get("original_run", "")).resolve() != primary.resolve():
            raise ValueError("completion plan original run binding changed")
        relative = PurePosixPath(row["case_output_relative"])
        if relative.is_absolute() or ".." in relative.parts or str(relative) != "/".join(
                str(row[k]) for k in ("setup_id", "model_id", "case_id")):
            raise ValueError("completion case output path differs from its identity")
        if plan.get("output_root"):
            root = outside_primary(Path(plan["output_root"]), primary)
            expected_result = (root / "results" / str(relative) / "generic_completion_results.json").resolve()
            if Path(row["result_path"]).resolve() != expected_result or root not in expected_result.parents:
                raise ValueError("completion result path escapes its external case output")
        if "runtime_config" in row:
            if config.get("execution_id") != plan.get("execution_id"):
                raise ValueError("completion execution identity changed")
            remote_root = PurePosixPath(row["remote_root"])
            remote_paths = [item["remote"] for item in row["staging_files"]]
            remote_paths.extend(p for p in (config.get("case_dir"), config.get("original_run_cfg", {}).get(
                "native_split_quality_binding")) if p)
            if not remote_root.is_absolute() or any(".." in PurePosixPath(p).parts or
                    remote_root not in PurePosixPath(p).parents for p in remote_paths):
                raise ValueError("completion staging path escapes its session")
    return rows


def runtime_configuration(row: dict, remote_root: str, execution_id: str) -> tuple[dict, list[dict]]:
    """Project original bound files into a new attempt, with unchanged content."""
    import posixpath
    local_case = Path(row["original_local_case_dir"])
    case_dir = f"{remote_root}/cases/{row['model_id']}/{row['case_id']}"
    config_dir = f"{remote_root}/configs/{row['model_id']}/{row['case_id']}"
    manifest_path = local_case / "split_manifest.json"
    manifest = read_json(manifest_path)
    staged = [{"local": str(manifest_path), "remote": case_dir + "/split_manifest.json",
               "sha256": file_sha256(manifest_path), "role": "original_manifest"}]
    for role in ("part1", "part2"):
        relative = str(manifest.get(role) or (manifest.get("models", {}).get(role) or {}).get("path") or "")
        if not relative or PurePosixPath(relative).is_absolute():
            raise ValueError("unsupported original model path: " + role)
        remote = posixpath.normpath(case_dir + "/" + relative)
        if not remote.startswith(remote_root + "/"):
            raise ValueError("original model path escapes completion workspace")
        local = (local_case / relative).resolve()
        if not local.is_file():
            raise FileNotFoundError("original Generic model metadata missing: " + str(local))
        sha = file_sha256(local)
        expected = row["expected_runtime_artifacts"].get("source_part2_onnx", {}) if role == "part2" else {}
        if expected.get("sha256") and sha != expected["sha256"]:
            raise ValueError("original Generic Part2 source binding differs")
        staged.append({"local": str(local), "remote": remote, "sha256": sha,
                       "role": role + "_onnx_metadata",
                       "existing_remote": expected.get("path")})
    original_native = read_json(Path(row["native_source"]))
    from ..native_command_contract import verify_native_command_contract
    command, command_status = verify_native_command_contract(original_native["native_command_contract"])
    if command is None or command["contract_sha256"] != row["native_command_contract_sha256"]:
        raise ValueError("original Native command changed since planning")
    binding = command["native_split_quality_binding"]
    if binding["binding_sha256"] != row["native_quality_binding_sha256"]:
        raise ValueError("original Generic/Native split artifact binding differs")
    prepared = row["input_binding"]["entries"][0]
    input_spec = {"path": prepared["remote_path"], "sha256": prepared["sha256"],
                  "shape": prepared["shape"], "dtype": prepared["dtype"],
                  "input_name": (manifest.get("orig_inputs") or [prepared["name"]])[0],
                  "format": "raw" if str(prepared["remote_path"]).endswith(".bin") else "npy"}
    reference_path = Path(row["functional_reference"]["output_manifest"])
    expected_manifest = command["artifacts"]["semantic_output_manifest"]
    if file_sha256(reference_path) != expected_manifest["sha256"]:
        raise ValueError("original functional output manifest changed")
    reference = read_json(reference_path)
    # H8's original C++ output writer records the input path and sequence; its
    # manifest SHA and input-image SHA are jointly sealed by the original
    # Native command. Other writers also include the image SHA in the manifest.
    reference_image = reference.get("input_image") or (reference.get("provenance") or {}).get("image")
    reference_image_sha = reference.get("input_image_sha256") or (reference.get("provenance") or {}).get("image_sha256")
    if (reference_image != command["input_image"] or
            command["input_image_sha256"] != row["input_binding"]["source_image_sha256"] or
            (reference_image_sha and reference_image_sha != command["input_image_sha256"])):
        raise ValueError("functional output reference uses a different original input")
    outputs = {}
    for descriptor in reference["outputs"]:
        local = (reference_path.parent / descriptor["file"]).resolve()
        if local.parent != reference_path.parent.resolve():
            raise ValueError("functional output tensor escapes original output directory")
        if file_sha256(local) != descriptor["sha256"]:
            raise ValueError("functional output tensor changed")
        remote = config_dir + "/reference/" + local.name
        staged.append({"local": str(local), "remote": remote, "sha256": descriptor["sha256"], "role": "functional_reference"})
        outputs[descriptor["name"]] = {"path": remote, "sha256": descriptor["sha256"],
            "shape": descriptor["shape"], "dtype": descriptor["dtype"], "format": "raw"}
    original_cfg = dict(row["original_run_cfg"])
    original_cfg["manifest"] = case_dir + "/split_manifest.json"
    original_cfg["native_split_quality_binding"] = config_dir + "/native_split_quality_binding.json"
    policy = json.loads(row["original_run_cfg"]["quality_gate_json"])
    limits = policy["native_contract"]["detection_self_reference"]
    runtime_options = command["runtime_options"]
    if command["backend"] == "hailo8_to_trt":
        runtime_input_format = runtime_options.get("hailo_format")
    elif command["backend"] == "hailo10h_to_trt":
        if type(runtime_options.get("quantized_inputs")) is not bool:
            raise ValueError("original Hailo10 input quantization binding missing")
        runtime_input_format = "uint8" if runtime_options["quantized_inputs"] else "float32"
    else:
        runtime_input_format = input_spec["dtype"]
    if runtime_input_format not in {"uint8", "float32"}:
        raise ValueError("original accelerator runtime input format missing")
    completion = row["completion_contract"] if row["task"] == "detection" else {
        "task": "classification", "operation": "top1_top5", "label_offset": 0}
    config = {"identity": row["identity"], "execution_id": execution_id, "original_run": row["original_run"],
              "task": row["task"], "case_dir": case_dir, "original_run_cfg": original_cfg,
              "repetitions": 3, "warmup_count": 100, "task_count": 1000,
              "queue_depth": int(original_cfg["throughput_queue_depth"]),
              "timeout_s": 3600, "measurement_boundary": "prepared_input_before_p1_to_last_completed_task",
              "input_binding": {**row["input_binding"], "prepared_input": input_spec,
                  "runtime_input_format": runtime_input_format,
                  "runtime_input_format_source": row["native_source"] + "#native_command_contract.runtime_options"},
              "completion_contract": completion,
              "expected_runtime_artifacts": {k: {"path": row["expected_runtime_artifacts"][k]["path"],
                   "sha256": row["expected_runtime_artifacts"][k]["sha256"]} for k in ("part1_runtime", "engine")},
              "functional_reference": {"input_sha256": input_spec["sha256"], "outputs": outputs,
                  "source_manifest": str(reference_path), "source_manifest_sha256": expected_manifest["sha256"],
                  "tolerances": {"match_iou_threshold": limits["iou_threshold"],
                                 "min_match_ratio": limits["min_reference_match_ratio"],
                                 "min_mean_iou": limits["min_mean_matched_iou"]}}}
    config["measurement_contract"] = row["measurement_boundary"]
    return config, [*staged, {"role": "original_split_binding", "payload": binding,
                             "remote": original_cfg["native_split_quality_binding"]}]


def stage_case(row: dict, config: dict, files: list[dict], transport: SSHTransport, local: Path) -> str:
    local.mkdir(parents=True, exist_ok=True)
    receipts = []
    for item in files:
        if "payload" in item:
            path = local / "native_split_quality_binding.json"
            write_json(path, item["payload"])
        else:
            path = Path(item["local"])
        existing = item.get("existing_remote")
        if existing:
            # Source Part2 is already attested in the immutable Native cache.
            # A session-local symlink avoids another model transfer.
            script = ("from pathlib import Path\nimport hashlib,os\nsrc=Path(" + repr(existing) + ")\n"
                      "dst=Path(" + repr(item["remote"]) + ")\nh=hashlib.sha256()\nwith src.open('rb') as f:\n"
                      " for b in iter(lambda:f.read(8388608),b''):h.update(b)\n"
                      "assert h.hexdigest()==" + repr(item["sha256"]) + ", 'existing source Part2 differs'\n"
                      "dst.parent.mkdir(parents=True,exist_ok=True)\n"
                      "if dst.is_symlink():\n assert dst.resolve()==src.resolve(), 'session source alias differs'\n"
                      "else:\n assert not dst.exists(), 'session model path already occupied'\n dst.symlink_to(src)\n")
            checked_remote("python3 -B - <<'COMPLETION_SOURCE_ALIAS'\n" + script + "COMPLETION_SOURCE_ALIAS", transport)
            receipts.append({"role": item["role"], "path": item["remote"], "source": existing, "sha256": item["sha256"], "status": "session_alias_to_exact_source"})
        else:
            receipts.append({"role": item["role"], **upload_new_file(transport, path, item["remote"], expected_sha=item.get("sha256", ""))})
    config_path = local / "generic_completion_config.json"
    write_json(config_path, config)
    remote_config = str(PurePosixPath(config["original_run_cfg"]["native_split_quality_binding"]).parent / config_path.name)
    receipts.append(upload_new_file(transport, config_path, remote_config))
    write_json(local / "staging_receipts.json", {"files": receipts, "completed_at": now()})
    return remote_config


def prepare_execution(scope_plan: Path, output: Path) -> Path:
    plan = read_json(scope_plan)
    primary = Path(plan["original_run"]).resolve()
    output = outside_primary(output, primary)
    rows = validate_plan(plan, Path(plan["scope_source"]), primary)
    execution_path = output / "plan" / "execution-plan.json"
    if execution_path.exists():
        raise FileExistsError("execution plan already exists; use its bound entry: " + str(execution_path))
    execution_id = datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]
    targets = bound_targets(primary, rows)
    for row in rows:
        target = targets[row["setup_id"]]
        runtime = original_generic_environment(row)
        row.update(remote_python_executable=runtime["python"], remote_environment=runtime["environment"],
                   generic_environment_source=runtime["source"], generic_environment_original_line=runtime["original_line"])
        remote_root = f"/home/{target['remote']['user']}/splitpoint_runs/generic_completed_task/{execution_id}"
        config, files = runtime_configuration(row, remote_root, execution_id)
        row.update(runtime_config=config, staging_files=files, remote_root=remote_root,
                   result_path=str(output / "results" / row["case_output_relative"] / "generic_completion_results.json"))
    plan.update(execution_id=execution_id, prepared_at=now(), output_root=str(output),
                scope_plan_source=str(scope_plan.resolve()), hardware_targets=targets)
    validate_plan(plan, Path(plan["scope_source"]), primary)
    write_json(execution_path, plan)
    from ..generic_completion_reporting import write_completion_reports
    write_completion_reports(plan, output / "results", output / "reports")
    return execution_path


def original_generic_environment(row: dict) -> dict:
    """Replay the original normal dispatch's interpreter and appended SDK sites.

    H8 uses system GPU Python plus its existing Hailo venv's appended sites.
    Merely using the Native C++ controller interpreter loses that SDK binding.
    """
    source = Path(row["generic_raw_source"]).resolve()
    original_results = (Path(row["original_run"]) / "models" / row["model_id"] / "benchmark_results").resolve()
    if original_results not in source.parents:
        raise ValueError("Generic runtime source is outside its original model results")
    # Successful supplemental rows retain their own dispatch alongside the
    # bound raw result. The canonical stdout can belong to an earlier failure.
    path = source.parent / ("remote_benchmark_stdout_" + row["setup_id"] + ".txt")
    matches = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if "[remote] python:" not in line:
            continue
        python = re.search(r"\bRUN_PY=([^\s|]+)", line)
        extra = re.search(r"\bEXTRA_SITES=([^\s]*)", line)
        if python and extra:
            matches.append((python.group(1), extra.group(1), line))
    if not matches or len({(p, sites) for p, sites, _ in matches}) != 1:
        raise ValueError("original Generic runtime environment is not uniquely bound: " + str(path))
    python, extra, line = matches[0]
    if not PurePosixPath(python).is_absolute() or any(not PurePosixPath(p).is_absolute() for p in extra.split(":") if p):
        raise ValueError("original Generic runtime environment path is not absolute")
    return {"python": python, "environment": {"SPLITPOINT_EXTRA_SITES": extra},
            "source": str(path), "original_line": line}


def configuration_comparison(row: dict, current: dict, current_path: Path) -> dict:
    """Compare available original snapshots; preserve unknown historical fields."""
    from ..resources.remote_scripts.native_host_telemetry import _identity
    primary = Path(row["original_run"])
    captures = read_json(primary / "reports" / "native_host_telemetry_summary.json")["captures"]
    relevant = [c for c in captures if (c.get("identity") or {}).get("setup_id") == row["setup_id"]]
    # The result path identifies the exact production attempt. Original cases
    # outside attempts use the canonical backend directory's captures.
    native_source = Path(row["native_source"])
    candidates = [c for c in relevant if Path(c["path"]).parent.parent in native_source.parents]
    if not candidates:
        command = row["native_command_contract"]
        backend = {"deepx_to_trt": "deepx", "hailo10h_to_trt": "hailo10h", "hailo8_to_trt": "hailo8"}[command["backend"]]
        source_root = primary / "native_producers" / backend
        if "attempts" in native_source.parts:
            i = native_source.parts.index("attempts")
            source_root = Path(*native_source.parts[:i + 3])
        candidates = [c for c in relevant if Path(c["path"]).parent.parent == source_root]
    current_identity = _identity(current)
    fields = ("cpu_governors", "cpu_min_max", "device_governors", "device_min_max")
    compared, differences, unknown = {}, [], []
    for key in fields:
        originals = [(c["path"], (c.get("identity") or {}).get(key)) for c in candidates]
        originals = [(path, value) for path, value in originals if value]
        current_value = json.loads(json.dumps(current_identity.get(key)))
        if not originals or not current_value:
            unknown.append(key)
            continue
        matches = all(value == current_value for _, value in originals)
        compared[key] = {"matches": matches, "current": current_value, "historical": originals}
        if not matches:
            differences.append(key)
    # nvpmodel query text contains current state as well as the selected mode.
    def mode(payload):
        return str(((payload.get("nvidia_commands") or {}).get("nvpmodel_mode") or {}).get("stdout") or "").strip()
    old_modes = [(c["path"], mode(read_json(Path(c["path"])))) for c in candidates]
    old_modes = [(p, value) for p, value in old_modes if value]
    if old_modes and mode(current):
        matched = all(value == mode(current) for _, value in old_modes)
        compared["nvpmodel_mode"] = {"matches": matched, "current": mode(current), "historical": old_modes}
        if not matched:
            differences.append("nvpmodel_mode")
    else:
        unknown.append("nvpmodel_mode")
    unknown += ["historical_thread_environment_not_separately_recorded", "thermal_state_is_observed_not_equalized"]
    return {"status": "conflict" if differences else "observed_no_known_configuration_conflict",
            "current_snapshot": str(current_path), "historical_snapshots": [c["path"] for c in candidates],
            "checks": compared, "differences": differences, "unknown_or_not_equalized": unknown,
            "note": "No tuning. Runtime/engine identity is independently checked by original command and artifact receipts."}


def download_case(transport: SSHTransport, remote_output: str, local_output: Path, attempt: int) -> dict:
    """Keep all attempt downloads, including evidence after a failed process."""
    from ..generic_completion_reporting import _read_result
    destination = local_output / "downloads" / f"attempt_{attempt:03d}"
    destination.mkdir(parents=True, exist_ok=False)
    rc, stdout = transport.scp_download(remote_output, str(destination), recursive=True, timeout=600)
    record = {"at": now(), "rc": rc, "output": stdout[-4000:], "destination": str(destination)}
    if rc:
        return record
    source_dir = destination / PurePosixPath(remote_output).name
    source = source_dir / "generic_completion_results.json"
    if source.is_file():
        payload = _read_result(source)
        # Keep the declared, immutable functional samples next to the imported
        # result so the normal parser verifies actual bytes on every reentry.
        for repetition in payload.get("repetitions", []):
            descriptors = list((repetition.get("raw_output_samples") or {}).values())
            if repetition.get("completed_result_artifact"):
                descriptors.append(repetition["completed_result_artifact"])
            for descriptor in descriptors:
                relative = Path(str(descriptor.get("path") or ""))
                if not descriptor.get("path") or relative.is_absolute() or ".." in relative.parts:
                    raise ValueError("downloaded functional output sample path invalid")
                sample = source_dir / relative
                if not sample.resolve().is_relative_to(source_dir.resolve()):
                    raise ValueError("downloaded functional output sample escapes attempt")
                if not sample.is_file():
                    # Preserve the result to let the ordinary reporter record
                    # the exact missing-output error instead of inventing reuse.
                    continue
                if file_sha256(sample) != descriptor.get("sha256"):
                    raise ValueError("downloaded functional output sample changed")
                target = local_output / relative
                if target.exists():
                    if file_sha256(target) != descriptor["sha256"]:
                        raise ValueError("existing collected functional output sample differs")
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(sample, target)
        write_json(local_output / "generic_completion_results.json", payload)
        record["result_collected"] = True
        record["remote_output_mirror"] = str(source_dir)
    else:
        record["result_collected"] = False
    return record


def run_case(row: dict, transport: SSHTransport, output: Path, index: int, *, total: int = 192,
             runner_root: str | None = None) -> None:
    from ..generic_completion_reporting import load_valid_repetitions, validate_completion_case
    config = row["runtime_config"]
    local_output = Path(row["result_path"]).parent
    local_output.mkdir(parents=True, exist_ok=True)
    valid = load_valid_repetitions(row["result_path"], row)
    completed = {r["repetition_index"] for r in valid}
    if completed == {1, 2, 3}:
        print(f"[completion] {index}/{total} {case_name(row)} reuse=3/3", flush=True)
        return
    remote_config = stage_case(row, config, row["staging_files"], transport,
                              output / "staging" / row["case_output_relative"])
    remote_output = row["remote_root"] + "/results/" + row["case_output_relative"]
    runtime_root = runner_root or row["remote_root"]
    checked_remote("mkdir -p -- " + shlex.quote(remote_output), transport)
    # A reentry may have locally collected valid work even if a previous remote
    # workspace was lost. Copy that exact parsed result into the new workspace.
    if completed:
        for repetition in valid:
            descriptors = [*(repetition.get("raw_output_samples") or {}).values(),
                           repetition["completed_result_artifact"]]
            for descriptor in descriptors:
                relative = descriptor["path"]  # checked by load_valid_repetitions
                upload_new_file(transport, local_output / relative, remote_output + "/" + relative,
                                expected_sha=descriptor["sha256"])
        upload_new_file(transport, Path(row["result_path"]), remote_output + "/generic_completion_results.json")
    attempts_path = local_output / "dispatch_attempts.json"
    attempts = read_json(attempts_path).get("attempts", []) if attempts_path.exists() else []
    for repetition in (1, 2, 3):
        if repetition in completed:
            continue
        argv = [row["remote_python_executable"], "-B", "-u",
                runtime_root + "/runner/run_split_onnxruntime.py",
                "--generic-completion-config", remote_config,
                "--generic-completion-repetition", str(repetition), "--out-dir", remote_output]
        command = "cd -- " + shlex.quote(runtime_root) + " && " + shlex.join(argv)
        attempt = {"repetition_index": repetition, "command": command, "started_at": now(), "status": "running"}
        attempts.append(attempt)
        write_json(attempts_path, {"attempts": attempts})
        print(f"[completion] {index}/{total} {case_name(row)} repetition={repetition}/3 warmup=100 tasks=1000 START", flush=True)
        logfile = local_output / f"dispatch_{len(attempts):03d}.log"
        try:
            with logfile.open("x", encoding="utf-8") as stream:
                def on_line(line):
                    stream.write(line + "\n"); stream.flush()
                    print(f"[{index}/{total} rep{repetition}/3] {line}", flush=True)
                row["dispatch_attempted"] = True
                rc = transport.run_streaming(command, on_line=on_line, timeout=7200,
                    env={**row.get("remote_environment", {}), "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1",
                         "ONNX_SPLITPOINT_ARTIFACT_POLICY": "cache_verify_only"})
            attempt.update(returncode=rc, status="exited")
        finally:
            attempt["finished_at"] = now()
            attempt["download"] = download_case(transport, remote_output, local_output, len(attempts))
            write_json(attempts_path, {"attempts": attempts})
        valid = load_valid_repetitions(row["result_path"], row)
        completed = {r["repetition_index"] for r in valid}
        if attempt.get("returncode") != 0 or repetition not in completed:
            raise RuntimeError(f"Generic case stopped: {case_name(row)} repetition={repetition}, rc={attempt.get('returncode')}")
        print(f"[completion] {index}/{total} {case_name(row)} repetition={repetition}/3 VALID", flush=True)
    final = validate_completion_case(read_json(Path(row["result_path"])), row, result_root=local_output)
    if final["status"] != "completed":
        raise RuntimeError("Generic case lacks three valid independent repetitions")


def terminal_bundle(output: Path) -> None:
    """Small diagnostics only; never include models, datasets or raw tensors."""
    destination = output / "completion-diagnostics.tar.gz"
    roots = [output / "reports", output / "plan", output / "device-preflight"]
    files = []
    for root in roots:
        if root.is_dir():
            files.extend(p for p in root.rglob("*") if p.is_file() and
                         p.suffix in {".json", ".csv", ".txt", ".log", ".md"} and p.stat().st_size <= 5 * 1024**2)
    files.extend(p for p in output.iterdir() if p.is_file() and p.suffix in {".json", ".log", ".txt", ".md"}
                 and p.stat().st_size <= 5 * 1024**2)
    with tarfile.open(destination, "w:gz") as archive:
        for path in sorted(set(files)):
            archive.add(path, arcname=str(path.relative_to(output)), recursive=False)


def write_status_report(plan: dict, summary: dict, output: Path, *, status: str, error: str = ""):
    remaining = summary["planned_cases"] - summary["status_counts"]["completed"]
    text = ["THESIS20 Generic Completed Task", f"Status: {status}",
            f"Execution: {plan['execution_id']}", f"Updated: {now()}",
            f"Original run (read only): {plan['original_run']}",
            f"Case balance: {json.dumps(summary['status_counts'], sort_keys=True)}",
            f"Valid repetitions: {summary['valid_repetitions']}/576; remaining cases: {remaining}",
            f"Actual measured completed tasks: {summary['completed_task_count']}/576000",
            f"Technical/quality/claim pairs: {summary['technical_pairs']}/{summary['quality_pairs']}/{summary['claim_pairs']}",
            "Each valid repetition contains 100 completed warmups and 1000 completed measured tasks.",
            "A single bound prepared input is repeated; this is not a 1000-image quality dataset.",
            "Original Generic, Quality, Native performance and energy results are preserved.",
            "No Quality campaign, Native performance, energy capture or model compilation is part of this entry.",
            "Development/screening evidence remains development/screening; no holdout claim is inferred.",
            "Completed-task FPS is count/makespan. Per-request latency is not derived from its reciprocal.",
            "Tables: reports/generic_completion_observations.csv, generic_completion_cross_runner_pairs.csv, generic_completion_cross_runner_groups.csv",
            "Raw repetitions: results/<setup>/<model>/<case>/; every dispatch and failed attempt is retained.",
            "Exact commands: per-case dispatch_attempts.json and dispatch_*.log.",
            "Plan/boundaries: plan/execution-plan.json; small diagnostic archive: completion-diagnostics.tar.gz."]
    if error:
        text.append("Stopped without automatic repair/restart: " + error)
    (output / "STATUS.txt").write_text("\n".join(text) + "\n", encoding="utf-8")
    observations = read_json(output / "reports" / "generic_completion_observations.json")["rows"]
    import csv
    fields = ("model_id", "case_id", "direction", "setup_id", "precision", "status",
              "valid_repetition_count", "completed_task_count", "completed_task_fps", "errors", "result_path")
    with (output / "reports" / "case-balance.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: json.dumps(row.get(key), ensure_ascii=False) if isinstance(row.get(key), (list, dict))
                         else row.get(key) for key in fields} for row in observations)


def execute_plan(plan_path: Path) -> int:
    from ..generic_completion_reporting import write_completion_reports, load_valid_repetitions, completion_pair, validate_completion_case
    plan = read_json(plan_path)
    primary = Path(plan["original_run"]).resolve()
    output = outside_primary(Path(plan["output_root"]), primary)
    rows = validate_plan(plan, Path(plan["scope_source"]), primary)
    bundle = output / "runner-bundle.tar.gz"
    if not bundle.is_file():
        raise FileNotFoundError("prepare the final tested session runner bundle before production")
    targets = bound_targets(primary, rows)
    starts = plan["start_case_keys"]
    if len(starts) != 6 or len(set(starts)) != 6 or not set(starts) <= {r["case_key"] for r in rows}:
        raise ValueError("six authorized start cases missing")
    start_rows = [next(r for r in rows if r["case_key"] == key) for key in starts]
    if {(r["setup_id"], r["task"]) for r in start_rows} != {(setup, task) for setup in targets for task in ("classification", "detection")}:
        raise ValueError("start cases must cover one classification and detection case per setup")
    ordered = start_rows + [r for r in rows if r["case_key"] not in starts]
    session = plan["execution_id"]
    cancel = threading.Event()
    scope = RemoteProcessLeaseScope(primary.name, session)
    registry = RemoteProcessLeaseRegistry()
    transports = {}
    status, error, current = "blocked", "", None
    rc = 1
    old_handlers = {}
    admitted = False
    stack = ExitStack()
    def cancelled(signum, frame):
        cancel.set()
    for signum in (signal.SIGINT, signal.SIGTERM):
        old_handlers[signum] = signal.signal(signum, cancelled)
    try:
        stack.enter_context(EvaluationRunLock(out_root=output.parent, run_dir=output,
                            owner={"run_id": primary.name, "session_id": session, "purpose": "generic_completed_task_192"}))
        physical_locks = []
        for target in targets.values():
            lock = EvaluationRunLock.for_resource(physical_dut_key(target),
                owner={"run_id": primary.name, "session_id": session, "purpose": "generic_completed_task_192"})
            stack.enter_context(lock)
            physical_locks.append(lock)
        def fence_unresolved_cleanup():
            pending = registry.active_count()
            if pending:
                for lock in physical_locks:
                    lock.commit_quarantine_fence({"reason": "generic_completion_remote_cleanup_unproven",
                        "pending_operations": pending, "journal": str(output / "remote-process-leases" / session)})
        stack.callback(fence_unresolved_cleanup)
        start_record = output / "production-start.json"
        # A previous technical stop never becomes an automatic restart.
        with start_record.open("x", encoding="utf-8") as stream:
            json.dump({"execution_id": session, "pid": os.getpid(), "ppid": os.getppid(),
                       "started_at": now(), "plan": str(plan_path), "runner_bundle": str(bundle),
                       "runner_bundle_sha256": file_sha256(bundle)}, stream, indent=2)
        admitted = True
        registry.configure_journal(scope=scope, journal_dir=output / "remote-process-leases" / session)
        observe_owners(targets, output / "production-preflight")
        for setup, target in targets.items():
            transport = transport_for(target, output / "diagnostics" / setup, scope=scope, registry=registry, cancel=cancel)
            transports[setup] = transport
            config_path = output / "production-preflight" / setup / "configuration.json"
            snapshot = capture_configuration(transport, target, primary=primary, output=config_path, phase="pre")
            representative = next(r for r in rows if r["setup_id"] == setup)
            verify_runtime_environment(transport, representative,
                output / "production-preflight" / setup / "runtime-environment.json")
            for row in rows:
                if row["setup_id"] == setup:
                    row["runtime_environment_comparison"] = configuration_comparison(row, snapshot, config_path)
            remote_root = next(r["remote_root"] for r in rows if r["setup_id"] == setup)
            upload_new_file(transport, bundle, remote_root + "/runner-bundle.tar.gz")
            checked_remote("tar -xzf " + shlex.quote(remote_root + "/runner-bundle.tar.gz")
                           + " -C " + shlex.quote(remote_root), transport)
        write_json(plan_path, plan)
        status = "start_cases_running"
        for index, row in enumerate(ordered, 1):
            current = row
            if cancel.is_set():
                raise RuntimeError("completion cancelled before next case")
            entered = time.monotonic()
            run_case(row, transports[row["setup_id"]], output, index)
            row["completion_observed_at"] = now()
            row["dispatch_wall_seconds"] = time.monotonic() - entered
            observed = validate_completion_case(read_json(Path(row["result_path"])), row,
                result_root=Path(row["result_path"]).parent)
            pair = completion_pair(observed, row)
            if row["case_key"] in starts and not pair["eligible_for_technical_transfer"]:
                write_json(output / "start-case-pairing-failure.json", pair)
                raise RuntimeError("start case has no valid completed-task pair: " + row["case_key"])
            write_json(plan_path, plan)
            summary = write_completion_reports(plan, output / "results", output / "reports")
            if index == 6:
                write_json(output / "start-cases-accepted.json", {"accepted_at": now(),
                    "case_keys": starts, "valid_repetitions": 18, "remaining_cases": 186,
                    "original_start_results_reused": True, "summary": summary})
                print("[completion] START CASES ACCEPTED 6/6, 18/18 repetitions; continuing remaining 186", flush=True)
                status = "remainder_running"
            durations = {(r["model_id"], r["setup_id"]): r["dispatch_wall_seconds"] for r in rows if r.get("dispatch_wall_seconds")}
            unresolved_groups = []
            estimate = 0.0
            for pending in ordered[index:]:
                measured = durations.get((pending["model_id"], pending["setup_id"]))
                if measured is None:
                    unresolved_groups.append((pending["model_id"], pending["setup_id"]))
                else:
                    estimate += measured
            write_json(output / "progress.json", {"updated_at": now(), "status": status,
                "last_completed_case": row["case_key"], "case_index": index, "total": 192,
                "remaining": 192-index, "measured_group_remaining_seconds": estimate,
                "groups_without_observed_duration": sorted(set(unresolved_groups)), "summary": summary})
            write_status_report(plan, summary, output, status=status)
        status, rc = "complete", 0
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
        if not admitted:
            return rc
        status = "cancelled" if cancel.is_set() else "stopped"
        if current is not None:
            current["execution_status"] = "failed" if current.get("dispatch_attempted") else "blocked"
            current["execution_error"] = error
            write_json(plan_path, plan)
        write_json(output / "technical-stop.json", {"at": now(), "error": error,
            "case_key": current.get("case_key") if current else None, "automatic_restart": False})
    finally:
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
        try:
            if admitted:
                # SSHTransport already performs exact token-bound cleanup on failures.
                # Any unresolved lease is retained and fenced, never globally killed.
                write_json(output / "cleanup-status.json", {"at": now(), "active_owned_remote_operations": registry.active_count(),
                    "registry_cancelled_or_poisoned": registry.cancelled, "foreign_processes_stopped": False})
                summary = write_completion_reports(plan, output / "results", output / "reports")
                write_status_report(plan, summary, output, status=status, error=error)
                write_json(output / "terminal-status.json", {"finished_at": now(), "status": status, "exitcode": rc,
                           "summary": summary, "error": error, "automatic_restart": False})
                terminal_bundle(output)
        finally:
            stack.close()
    return rc


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--prepare", type=Path, metavar="SCOPE_PLAN_JSON")
    actions.add_argument("--execute", type=Path, metavar="EXECUTION_PLAN_JSON")
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args(argv)
    if args.prepare:
        if args.output_root is None:
            parser.error("--prepare requires --output-root outside the primary run")
        print(prepare_execution(args.prepare, args.output_root))
        return 0
    return execute_plan(args.execute)


if __name__ == "__main__":
    raise SystemExit(main())
