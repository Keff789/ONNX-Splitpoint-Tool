"""Normal Native Quality preparation and leased lifecycle, using local transport."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tarfile
from types import SimpleNamespace
from typing import Any

import pytest

from onnx_splitpoint_tool.workflow import native_dataset_quality as workflow
from onnx_splitpoint_tool.native_split_quality import seal_native_split_quality_binding
from test_v269f_native_split_receipt_validation import _fixture_payload
from test_v269f_native_split_receipt_validation import _native_row_for_binding


ROOT = Path(__file__).resolve().parents[1]


def _write(path: Path, value: Any) -> dict[str, Any]:
    path.write_text(json.dumps(value, sort_keys=True))
    return workflow.file_descriptor(path)


@pytest.fixture
def prepared(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    prior_payload, _ = _fixture_payload(tmp_path / "prior")
    prior = seal_native_split_quality_binding(prior_payload)
    prior_descriptor = _write(tmp_path / "prior.json", prior)
    freeze = _write(tmp_path / "freeze.json", {"source_export_root": str(ROOT), "ready": True})
    config = {"identity": {"eval_run_id": "fresh-native-capture", "model_id": "yolo26s", "case_id": "b038",
        "setup_id": "hailo8_setup", "source_run_id": "hailo8_to_trt"}, "native": {
        "engine_build_forbidden": True, "native_split_quality_binding": prior_descriptor["path"],
        "hef": prior["artifacts"]["part1_runtime"]["path"], "engine": prior["artifacts"]["engine"]["path"]},
        "ground_truth": _write(tmp_path / "ground_truth.json", {"records": []}),
        "prior_artifact_selection_file": prior_descriptor, "prior_artifact_selection": prior,
        "timed_fps_authority": _write(tmp_path / "timed.json", {"fps_makespan": 10, "case": "b038"})}
    config_path = tmp_path / "config.json"
    _write(config_path, config)
    integrity_calls = []

    def integrity(root: Path) -> dict[str, Any]:
        integrity_calls.append(root)
        return {"ok": True, "unit_test_source_guard": True}

    monkeypatch.setattr(workflow, "verify_installed_source_integrity", integrity)
    target = {"id": "hailo8_setup", "physical_host_id": "unit-hailo8-chassis",
              "remote": {"host": "192.0.2.23", "user": "nx", "port": 22}}
    kwargs = {"target": target, "python": sys.executable, "remote_root": str(tmp_path / "fake_remote"),
        "output_root": str(tmp_path / "action"), "source_freeze": freeze,
        "reserve_bytes": 0, "forecast_bytes": 1, "timeout_s": 30}
    return SimpleNamespace(config=config, path=config_path, kwargs=kwargs, prior=prior,
                           output=Path(kwargs["output_root"]), integrity_calls=integrity_calls)


def test_preparation_stages_exact_closure_and_preserves_original_selection(prepared: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ONNX_SPLITPOINT_ARTIFACT_POLICY", "cache_verify_only")
    before = prepared.path.read_bytes()
    action = workflow.prepare_native_quality_action(prepared.path, **prepared.kwargs)
    assert prepared.integrity_calls == [ROOT]
    assert prepared.path.read_bytes() == before
    assert action["engine_build_allowed"] is action["performance_claims_emitted"] is False
    command = shlex.split(action["command"])
    assert command[:4] == ["env", "-u", "ONNX_SPLITPOINT_ARTIFACT_POLICY", "ONNX_SPLITPOINT_NATIVE_TRT_BUILD=0"]
    assert "cache_verify_only" not in command
    assert command[4:6] == [sys.executable, "-B"]
    assert command[6].endswith("/scripts/native_dataset_quality.py")
    stage = prepared.output / "stage"
    staged = json.loads((stage / "capture_config.json").read_text())
    assert staged["prior_artifact_selection"] == prepared.prior
    assert staged["prior_artifact_selection"]["eval_run_id"] != staged["identity"]["eval_run_id"]
    assert staged["native"]["hef"] == prepared.config["native"]["hef"]
    assert staged["native"]["engine"] == prepared.config["native"]["engine"]
    for field in ("ground_truth", "prior_artifact_selection_file", "timed_fps_authority"):
        relative = "inputs/" + field + ".json"
        assert staged[field]["path"] == action["remote_root"] + "/" + relative
        assert staged[field]["sha256"] == prepared.config[field]["sha256"]
        assert staged[field]["size_bytes"] == prepared.config[field]["size_bytes"]
    assert staged["native"]["native_split_quality_binding"] == staged["prior_artifact_selection_file"]["path"]
    inventory = {entry[0] for entry in workflow.NATIVE_DATASET_QUALITY_PACKAGE_CLOSURE}
    inventory.update(entry[0] for entry in workflow.NATIVE_DATASET_QUALITY_RESOURCE_CLOSURE)
    assert {entry["path"] for entry in action["source_files"]} == inventory | {"capture_config.json"}
    manifest = json.loads((stage / "STAGED_SOURCE_FILES.json").read_text())
    for descriptor in manifest:
        path = stage / descriptor["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == descriptor["sha256"]
        assert path.stat().st_size == descriptor["size_bytes"]
    with tarfile.open(action["stage_archive"]["path"]) as archive:
        assert set(archive.getnames()) == {path.relative_to(stage).as_posix() for path in stage.rglob("*") if path.is_file()}


@pytest.mark.parametrize('conflict', [False, True])
def test_measured_native_process_local_sites_are_exact_and_child_scoped(prepared, conflict, monkeypatch):
    from onnx_splitpoint_tool.native_command_contract import seal_native_command_contract
    command = _native_row_for_binding(prepared.prior)['native_command_contract']
    # This focused fixture uses the normal sealed command verifier; the full
    # mixed H8 contract itself has separate existing Native verifier tests.
    command['runtime_options']['process_local_extra_sites'] = ['/opt/existing_hailo/site-packages']
    command = seal_native_command_contract(command)
    timed = {'native_command_contract': command}
    prepared.config['timed_fps_authority'] = _write(Path(prepared.config['timed_fps_authority']['path']), timed)
    prepared.config['identity'].update(backend='hailo8_to_trt', runtime_precision_identity='uint8_dequant_fp16')
    if conflict:
        prepared.config['native']['process_local_extra_sites'] = ['/opt/different/site-packages']
    _write(prepared.path, prepared.config)
    monkeypatch.delenv('SPLITPOINT_EXTRA_SITES', raising=False)
    if conflict:
        with pytest.raises(ValueError, match='sites conflict'):
            workflow.prepare_native_quality_action(prepared.path, **prepared.kwargs)
    else:
        action = workflow.prepare_native_quality_action(prepared.path, **prepared.kwargs)
        assert action['environment'] == {'SPLITPOINT_EXTRA_SITES': '/opt/existing_hailo/site-packages'}
        assert 'SPLITPOINT_EXTRA_SITES=/opt/existing_hailo/site-packages' in shlex.split(action['command'])
    assert 'SPLITPOINT_EXTRA_SITES' not in os.environ


@pytest.mark.parametrize("fault", ["wrong_setup", "missing_host", "missing_user"])
def test_preparation_rejects_malformed_target(prepared: SimpleNamespace, fault: str) -> None:
    target = deepcopy(prepared.kwargs["target"])
    if fault == "wrong_setup":
        target["id"] = "another_setup"
    else:
        target["remote"].pop("host" if fault == "missing_host" else "user")
    with pytest.raises(ValueError, match="target"):
        workflow.prepare_native_quality_action(prepared.path, **{**prepared.kwargs, "target": target})
    assert not prepared.output.exists()


def test_preparation_denies_capacity_before_staging(prepared: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(workflow.shutil, "disk_usage", lambda path: SimpleNamespace(free=100))
    with pytest.raises(RuntimeError, match="capacity admission failed"):
        workflow.prepare_native_quality_action(prepared.path, **{**prepared.kwargs, "reserve_bytes": 90, "forecast_bytes": 20})
    assert not prepared.output.exists()
    assert not prepared.integrity_calls


@pytest.mark.parametrize("fault", ["wrong_root", "invalid_source"])
def test_preparation_requires_normal_published_source_guards(prepared: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, fault: str) -> None:
    if fault == "wrong_root":
        freeze = _write(Path(prepared.kwargs["source_freeze"]["path"]), {"source_export_root": "/unrelated/product"})
        prepared.kwargs["source_freeze"] = freeze
    else:
        monkeypatch.setattr(workflow, "verify_installed_source_integrity", lambda root: {"ok": False})
    with pytest.raises(ValueError, match="source"):
        workflow.prepare_native_quality_action(prepared.path, **prepared.kwargs)
    assert not prepared.output.exists()


@pytest.fixture
def execution(prepared: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    action = workflow.prepare_native_quality_action(prepared.path, **prepared.kwargs)
    state = SimpleNamespace(events=[], owners_ok=True, capture_rc=0, active=0, cancelled=False, tamper=False,
                            capture_calls=0, verify_calls=0, fences=[], held=set(), transports=0)

    class Lock:
        def __init__(self, *, out_root: Path, run_dir: Path, owner: Any, kind="run"):
            self.kind = kind
            self.owner = owner

        @classmethod
        def for_resource(cls, resource: str, *, owner: Any):
            assert resource == workflow.physical_dut_key(action["target"]) == "dut:unit-hailo8-chassis"
            return cls(out_root=prepared.output, run_dir=prepared.output, owner=owner, kind="physical")

        def __enter__(self):
            assert self.owner["purpose"] == "native_dataset_quality"
            state.held.add(self.kind)
            state.events.append("enter_" + self.kind)
            return self

        def __exit__(self, *args: Any):
            state.held.remove(self.kind)
            state.events.append("exit_" + self.kind)

        def commit_quarantine_fence(self, value: Any):
            state.fences.append(value)

    class Registry:
        @property
        def cancelled(self):
            return state.cancelled

        def active_count(self):
            return state.active

        def configure_journal(self, *, scope: Any, journal_dir: Path):
            assert scope.run_id == action["identity"]["eval_run_id"]
            assert journal_dir.parent == prepared.output / "remote-process-leases"
            state.events.append("registry_journal")

    class Transport:
        def run(self, command: str, *, timeout: int, env=None):
            assert state.held == {"run", "physical"}
            assert env == {"PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1"}
            if command == action["command"]:
                state.capture_calls += 1
                state.events.append("capture")
                result_path = Path(action["remote_root"]) / "results" / action["relative_output"] / "composed_request.json"
                result_path.parent.mkdir(parents=True)
                _write(result_path, {"producer_identity": {"record_count": 5000, "additional_nms_applied": False,
                                                          "producer_identity_sha256": "a" * 64}})
                if state.tamper:
                    source = Path(action["remote_root"]) / "onnx_splitpoint_tool/quality_cache.py"
                    source.write_bytes(source.read_bytes() + b"\n# injected stage mutation\n")
                return state.capture_rc, "unit-only Native capture transport result\n"
            if "NATIVE_QUALITY_VERIFY" in command:
                state.verify_calls += 1
                state.events.append("verify")
            result = subprocess.run(command, shell=True, executable="/bin/bash", capture_output=True, text=True,
                env={**os.environ, **(env or {})}, timeout=timeout)
            return result.returncode, result.stdout + result.stderr

        def run_read_only(self, command: str, *, timeout: int):
            result = subprocess.run(command, shell=True, executable="/bin/bash", capture_output=True, text=True, timeout=timeout)
            return result.returncode, result.stdout + result.stderr

        def scp_upload(self, local: str, remote: str, *, timeout: int):
            shutil.copyfile(local, remote)
            return 0, "local transport upload"

        def scp_download(self, remote: str, local: str, *, recursive: bool, timeout: int):
            assert recursive is True
            state.events.append("collect")
            shutil.copytree(remote, Path(local) / Path(remote).name)
            return 0, "local transport collection"

    def owners(targets: Any, output: Path, *, resource_scoped: bool):
        assert state.held == {"run", "physical"}
        assert targets == {action["target"]["id"]: action["target"]} and resource_scoped is True
        state.events.append("owners")
        return {"ok": state.owners_ok, "unit_only_read_only_probe": True}

    def transport(target: Any, diagnostics: Path, *, scope: Any, registry: Any):
        assert target == action["target"]
        assert scope.run_id == action["identity"]["eval_run_id"] and isinstance(registry, Registry)
        state.transports += 1
        return Transport()

    actual_write = workflow._write

    def writes(path: Path, value: Any):
        state.events.append("write_" + Path(path).name)
        actual_write(path, value)

    monkeypatch.setattr(workflow, "EvaluationRunLock", Lock)
    monkeypatch.setattr(workflow, "RemoteProcessLeaseRegistry", Registry)
    monkeypatch.setattr(workflow, "observe_owners", owners)
    monkeypatch.setattr(workflow, "transport_for", transport)
    monkeypatch.setattr(workflow, "_write", writes)
    return SimpleNamespace(action=action, state=state, output=prepared.output, path=prepared.output / "ACTION.json")


def _release_receipts(execution: SimpleNamespace, *, completed: bool) -> None:
    out, state = execution.output, execution.state
    terminal = json.loads((out / "TERMINAL.json").read_text())
    cleanup = json.loads((out / "CLEANUP_STATUS.json").read_text())
    release = json.loads((out / "PHYSICAL_RELEASE.json").read_text())
    assert terminal["capture_completed"] is completed
    assert terminal["status"] == ("completed" if completed else "failed")
    assert terminal["performance_claims_emitted"] is terminal["engine_build_allowed"] is False
    assert cleanup["foreign_processes_stopped"] is False
    assert cleanup["remote_source_preserved"] == execution.action["remote_root"]
    assert release["physical_lock_released"] is release["run_lock_released"] is True
    assert not state.held
    assert state.events.index("write_TERMINAL.json") < state.events.index("exit_physical") < state.events.index("exit_run") < state.events.index("write_PHYSICAL_RELEASE.json")


def test_execution_uses_normal_physical_owners_leases_and_rehashes(execution: SimpleNamespace) -> None:
    terminal = workflow.execute_native_quality_action(execution.path)
    assert terminal["completed_images"] == 5000
    assert execution.state.capture_calls == 1 and execution.state.verify_calls == 2
    assert execution.state.events.index("owners") < execution.state.events.index("capture")
    assert execution.state.events.count("collect") == 1
    assert not execution.state.fences
    _release_receipts(execution, completed=True)


def test_unknown_owner_cannot_admit_transport_or_capture(execution: SimpleNamespace) -> None:
    execution.state.owners_ok = False
    with pytest.raises(RuntimeError, match="existing resource owner unresolved"):
        workflow.execute_native_quality_action(execution.path)
    assert execution.state.transports == execution.state.capture_calls == 0
    _release_receipts(execution, completed=False)


@pytest.mark.parametrize("fault", ["capture_failure", "source_change"])
def test_execution_failure_preserves_terminal_cleanup_and_physical_release(execution: SimpleNamespace, fault: str) -> None:
    execution.state.capture_rc = 9 if fault == "capture_failure" else 0
    execution.state.tamper = fault == "source_change"
    with pytest.raises(RuntimeError, match="capture or collection failed" if fault == "capture_failure" else "remote command failed"):
        workflow.execute_native_quality_action(execution.path)
    assert execution.state.capture_calls == 1 and execution.state.verify_calls == 2
    _release_receipts(execution, completed=False)


def test_prepared_archive_change_stops_before_leases(execution: SimpleNamespace) -> None:
    path = Path(execution.action["stage_archive"]["path"])
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="prepared software archive changed"):
        workflow.execute_native_quality_action(execution.path)
    assert not execution.state.events


def test_unresolved_owned_operations_commit_physical_quarantine(execution: SimpleNamespace) -> None:
    execution.state.active = 1
    workflow.execute_native_quality_action(execution.path)
    assert len(execution.state.fences) == 1
    assert execution.state.fences[0]["reason"] == "native_quality_remote_cleanup_unproven"
    cleanup = json.loads((execution.output / "CLEANUP_STATUS.json").read_text())
    assert cleanup["active_owned_remote_operations"] == 1 and cleanup["foreign_processes_stopped"] is False
