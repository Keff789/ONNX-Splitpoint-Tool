"""Normal, leased deployment of the untimed Native dataset Quality capture.

Only software and frozen metadata are staged.  The exact existing Native
artifacts and dataset stay at their bound accelerator paths; no builder,
performance measurement, CPU reference inference or statistics pool is opened.
"""
from __future__ import annotations

from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path, PurePosixPath
import shlex
import shutil
import tarfile
import uuid

from ..native_dataset_quality import checked_json, file_descriptor
from ..native_command_contract import verify_native_command_contract
from ..remote_runtime_closure import (
    NATIVE_DATASET_QUALITY_PACKAGE_CLOSURE, NATIVE_DATASET_QUALITY_RESOURCE_CLOSURE,
)
from ..source_integrity import verify_installed_source_integrity
from ..remote.process_lease import RemoteProcessLeaseRegistry, RemoteProcessLeaseScope
from .execution_binding import physical_dut_key
from .generic_completed_task import observe_owners, transport_for, upload_new_file, checked_remote
from .run_control import EvaluationRunLock


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def capacity_admission(path, *, reserve_bytes, forecast_bytes):
    """Keep the declared full footprint and local reserve before staging."""
    if reserve_bytes < 0 or forecast_bytes <= 0:
        raise ValueError("Native Quality capacity budget invalid")
    free = shutil.disk_usage(path).free
    result = {"available_bytes": free, "reserve_bytes": reserve_bytes,
              "forecast_bytes": forecast_bytes, "admitted": free >= reserve_bytes + forecast_bytes}
    if not result["admitted"]:
        raise RuntimeError("Native Quality capacity admission failed: " + json.dumps(result))
    return result


def _native_process_environment(config):
    """Reuse only the process-local sites sealed by the measured Native job."""
    timed = checked_json(config['timed_fps_authority'])
    raw = timed.get('native_command_contract')
    if raw is None:
        if config.get('native', {}).get('process_local_extra_sites'):
            raise ValueError('Native Quality extra sites lack measured command authority')
        return {}
    identity = config['identity']
    command, status = verify_native_command_contract(raw, expected_identity={
        'model': identity['model_id'], 'case': identity['case_id'],
        'setup_id': identity['setup_id'], 'backend': identity['backend'],
        'precision': identity['runtime_precision_identity'],
    })
    if command is None:
        raise ValueError('Native Quality measured process environment invalid: ' + status)
    sites = command.get('runtime_options', {}).get('process_local_extra_sites') or []
    if (not isinstance(sites, list) or any(not isinstance(path, str)
            or not PurePosixPath(path).is_absolute() or ':' in path
            or any(char in path for char in ('\n', '\r', '\x00')) for path in sites)):
        raise ValueError('Native Quality measured process-local sites invalid')
    supplied = config.get('native', {}).get('process_local_extra_sites', sites)
    if supplied != sites:
        raise ValueError('Native Quality process-local sites conflict with measured command')
    return {'SPLITPOINT_EXTRA_SITES': ':'.join(sites)} if sites else {}


def prepare_native_quality_action(config_path, *, target, python, remote_root,
                                  output_root, source_freeze,
                                  reserve_bytes=24 * 1024**3, forecast_bytes=2 * 1024**3,
                                  timeout_s=7200):
    """Prepare a reviewable one-case action without contacting the accelerator."""
    out = Path(output_root).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    admission = capacity_admission(out.parent, reserve_bytes=reserve_bytes, forecast_bytes=forecast_bytes)
    original = json.loads(Path(config_path).read_text())
    identity = original["identity"]
    remote_target = target.get("remote") or target.get("runtime") or {}
    if not remote_target.get("host") or not remote_target.get("user"):
        raise ValueError("Native Quality normal target host/user missing")
    if target["id"] != identity["setup_id"] or original["native"].get("engine_build_forbidden") is not True:
        raise ValueError("Native Quality target/no-build authority differs")
    if not PurePosixPath(remote_root).is_absolute() or timeout_s <= 0:
        raise ValueError("Native Quality private remote path/timeout invalid")
    environment = _native_process_environment(original)
    # The immutable release descriptor is itself hash bound, and the staged
    # module list is rehashed before and after the actual capture.
    ready = checked_json(source_freeze)
    root = Path(__file__).resolve().parents[2]
    if Path(ready["source_export_root"]).resolve() != root:
        raise ValueError("Native Quality controller differs from its published source freeze")
    source_integrity = verify_installed_source_integrity(root)
    if source_integrity.get("ok") is not True:
        raise ValueError("Native Quality installed source integrity failed")
    out.mkdir(exist_ok=False)
    stage = out / "stage"
    stage.mkdir()
    closure = []
    inventory = [entry[0] for entry in NATIVE_DATASET_QUALITY_PACKAGE_CLOSURE]
    inventory += [entry[0] for entry in NATIVE_DATASET_QUALITY_RESOURCE_CLOSURE]
    for relative in sorted(set(inventory)):
        source = root / relative
        destination = stage / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        descriptor = file_descriptor(destination)
        descriptor["path"] = relative
        closure.append(descriptor)
    config = deepcopy(original)
    inputs = []
    for field in ("ground_truth", "prior_artifact_selection_file", "timed_fps_authority"):
        checked_json(original[field])
        source = Path(original[field]["path"])
        relative = "inputs/" + field + ".json"
        destination = stage / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        descriptor = file_descriptor(destination)
        inputs.append({**descriptor, "path": relative})
        config[field] = {**descriptor, "path": remote_root + "/" + relative}
    config["native"]["native_split_quality_binding"] = config["prior_artifact_selection_file"]["path"]
    _write(stage / "capture_config.json", config)
    closure.append({**file_descriptor(stage / "capture_config.json"), "path": "capture_config.json"})
    _write(stage / "STAGED_SOURCE_FILES.json", closure + inputs)
    archive = out / "native_quality_stage.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        for path in sorted(stage.rglob("*")):
            if path.is_file():
                bundle.add(path, arcname=path.relative_to(stage).as_posix(), recursive=False)
    relative_output = (f"models/{identity['model_id']}/benchmark_results/quality_inputs/{identity['setup_id']}"
                       f"/results/{identity['case_id']}/results_{identity['source_run_id']}/task_quality_inputs")
    remote_output = remote_root + "/results/" + relative_output
    command = shlex.join([
        "env", "-u", "ONNX_SPLITPOINT_ARTIFACT_POLICY", "ONNX_SPLITPOINT_NATIVE_TRT_BUILD=0",
        *(name + '=' + value for name, value in environment.items()),
        python, "-B", remote_root + "/scripts/native_dataset_quality.py",
        "--config", remote_root + "/capture_config.json", "--out", remote_output,
    ])
    action = {"schema": "onnx-splitpoint/native-dataset-quality-action", "schema_version": 1,
        "target": target, "python": python, "remote_root": remote_root,
        "output_root": str(out), "relative_output": relative_output,
        "original_config": file_descriptor(config_path), "source_freeze": source_freeze,
        "stage_archive": file_descriptor(archive), "source_files": closure, "input_files": inputs,
        "source_integrity": source_integrity,
        "command": command, "timeout_s": timeout_s, "capacity": admission,
        "environment": environment,
        "engine_build_allowed": False, "performance_claims_emitted": False,
        "expected_images": 5000, "identity": identity}
    _write(out / "ACTION.json", action)
    return action


def _verify_remote_stage(transport, action):
    remote = action["remote_root"]
    code = ("import json,hashlib\nfrom pathlib import Path\nr=Path(" + repr(remote) + ")\n"
            "for d in json.loads((r/'STAGED_SOURCE_FILES.json').read_text()):\n"
            " p=r/d['path'];assert p.is_file() and p.stat().st_size==d['size_bytes'],d['path']\n"
            " assert hashlib.sha256(p.read_bytes()).hexdigest()==d['sha256'],d['path']\n"
            "print('native_quality_source_files_verified')\n")
    return checked_remote(shlex.quote(action["python"]) + " -B - <<'NATIVE_QUALITY_VERIFY'\n" + code + "NATIVE_QUALITY_VERIFY", transport)


def execute_native_quality_action(action_path):
    """Execute exactly the prepared action under ordinary physical leases."""
    action = json.loads(Path(action_path).read_text())
    if action.get("schema") != "onnx-splitpoint/native-dataset-quality-action" or action.get("engine_build_allowed") is not False:
        raise ValueError("Native Quality action/no-build invalid")
    out = Path(action["output_root"])
    checked_json(action["source_freeze"])
    archive = file_descriptor(action["stage_archive"]["path"])
    if archive != action["stage_archive"]:
        raise ValueError("Native Quality prepared software archive changed")
    budget = action["capacity"]
    _write(out / "CAPACITY_BEFORE_EXECUTION.json", capacity_admission(out,
        reserve_bytes=budget["reserve_bytes"], forecast_bytes=budget["forecast_bytes"]))
    target = action["target"]
    session = "native_quality_" + uuid.uuid4().hex
    scope = RemoteProcessLeaseScope(action["identity"]["eval_run_id"], session)
    registry = RemoteProcessLeaseRegistry()
    registry.configure_journal(scope=scope, journal_dir=out / "remote-process-leases" / session)
    terminal = {"status": "failed", "session_id": session, "capture_completed": False,
                "performance_claims_emitted": False, "engine_build_allowed": False}
    with ExitStack() as stack:
        stack.enter_context(EvaluationRunLock(out_root=out, run_dir=out / "execution", owner={"purpose": "native_dataset_quality", "session_id": session}))
        lock = stack.enter_context(EvaluationRunLock.for_resource(physical_dut_key(target), owner={"purpose": "native_dataset_quality", "session_id": session}))
        try:
            owners = observe_owners({target["id"]: target}, out / "ownership_before", resource_scoped=True)
            if owners.get("ok") is not True:
                raise RuntimeError("Native Quality existing resource owner unresolved")
            transport = transport_for(target, out / "transport", scope=scope, registry=registry)
            remote = action["remote_root"]
            checked_remote("test ! -e " + shlex.quote(remote) + " && mkdir -m 700 -p -- " + shlex.quote(remote), transport)
            upload_new_file(transport, Path(archive["path"]), remote + "/stage.tar.gz", expected_sha=archive["sha256"])
            checked_remote("tar -xzf " + shlex.quote(remote + "/stage.tar.gz") + " -C " + shlex.quote(remote), transport)
            _verify_remote_stage(transport, action)
            checked_remote(shlex.join([action["python"], "-B", remote + "/scripts/native_dataset_quality.py", "--help"]), transport)
            rc, output = transport.run(action["command"], timeout=action["timeout_s"],
                env={"PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1"})
            (out / "CAPTURE_STDOUT.log").write_text(output)
            terminal["capture_returncode"] = rc
            _verify_remote_stage(transport, action)
            collected = out / "collected"
            collected.mkdir()
            collect_rc, collect_output = transport.scp_download(remote + "/results", str(collected), recursive=True, timeout=900)
            terminal.update(collection_returncode=collect_rc, collection_output=collect_output)
            if rc or collect_rc:
                raise RuntimeError("Native Quality capture or collection failed")
            request = collected / "results" / action["relative_output"] / "composed_request.json"
            payload = json.loads(request.read_text())
            producer = payload["producer_identity"]
            if producer.get("record_count") != 5000 or producer.get("additional_nms_applied") is not False:
                raise ValueError("Native Quality actual completed population invalid")
            terminal.update(status="completed", capture_completed=True, request=file_descriptor(request),
                            completed_images=5000, producer_identity_sha256=producer["producer_identity_sha256"])
        except Exception as error:
            terminal["error"] = repr(error)
        finally:
            cleanup = {"active_owned_remote_operations": registry.active_count(), "registry_cancelled": registry.cancelled,
                       "foreign_processes_stopped": False, "remote_source_preserved": action["remote_root"]}
            _write(out / "CLEANUP_STATUS.json", cleanup)
            if registry.active_count() or registry.cancelled:
                lock.commit_quarantine_fence({"reason": "native_quality_remote_cleanup_unproven", "session_id": session})
            _write(out / "TERMINAL.json", terminal)
    _write(out / "PHYSICAL_RELEASE.json", {"physical_lock_released": True, "run_lock_released": True,
        "active_owned_remote_operations": registry.active_count(), "registry_cancelled": registry.cancelled,
        "session_id": session})
    if terminal["status"] != "completed":
        raise RuntimeError(terminal.get("error", "Native Quality capture failed"))
    return terminal
