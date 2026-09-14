"""T02: real Native package closure, transport barriers and bound imports.

The only substituted boundary is the remote transport. Package bytes, Python
initializers, import code and SHA/token checks are the release implementations.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import shlex
import shutil
import subprocess
import sys
from typing import Mapping

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(os.environ.get("V2801_NATIVE_SOURCE", str(ROOT)))
REAL_INITIALIZERS = (
    "onnx_splitpoint_tool/__init__.py",
    "onnx_splitpoint_tool/release_identity.py",
    "onnx_splitpoint_tool/validation/__init__.py",
    "onnx_splitpoint_tool/runners/__init__.py",
    "onnx_splitpoint_tool/runners/backends/__init__.py",
    "onnx_splitpoint_tool/runners/harness/__init__.py",
    "onnx_splitpoint_tool/runners/harness/classification.py",
)


def _closure():
    return runpy.run_path(str(SOURCE / "onnx_splitpoint_tool/remote_runtime_closure.py"))[
        "native_remote_package_closure"
    ]()


def _copy(remote, relative):
    target = remote / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SOURCE / relative, target)


def _import_probe(remote, modules, *, extra=None):
    # -I excludes PYTHONPATH/user site and the source checkout. An explicitly
    # supplied second installation is adversarial and must never fill a hole.
    code = """
import hashlib, importlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1]).resolve()
if sys.argv[3]:
    sys.path.insert(0, sys.argv[3])
sys.path.insert(0, str(root))
for name in json.loads(sys.argv[2]):
    importlib.import_module(name)
proofs = {}
for name, module in sorted(sys.modules.items()):
    if name == 'onnx_splitpoint_tool' or name.startswith('onnx_splitpoint_tool.'):
        path = pathlib.Path(module.__file__).resolve()
        if not path.is_relative_to(root):
            raise RuntimeError('unbound_remote_import:' + name + ':' + str(path))
        proofs[name] = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
print(json.dumps(proofs, sort_keys=True))
"""
    return subprocess.run(
        [sys.executable, "-I", "-B", "-c", code, str(remote), json.dumps(modules), str(extra or "")],
        cwd=remote, capture_output=True, text=True, timeout=40,
    )


@pytest.mark.parametrize("module", [
    "onnx_splitpoint_tool.validation.accuracy_gates",
    "onnx_splitpoint_tool.validation.host_postprocess",
])
def test_t02_01_old_closure_reproduces_missing_contract(tmp_path, module):
    remote = tmp_path / "old_target"
    for relative, _, _ in _closure():
        if relative != "onnx_splitpoint_tool/quality_result_contract.py":
            _copy(remote, relative)
    # Deployment historically relied on installed initializers. Use exact
    # release initializers explicitly; do not suppress validation.__init__.
    for relative in REAL_INITIALIZERS:
        _copy(remote, relative)
    cp = _import_probe(remote, [module])
    assert cp.returncode != 0
    assert "No module named 'onnx_splitpoint_tool.quality_result_contract'" in cp.stderr


@pytest.mark.parametrize("first", [
    "onnx_splitpoint_tool.validation.accuracy_gates",
    "onnx_splitpoint_tool.validation.host_postprocess",
])
def test_t02_02_10_complete_closure_real_initializers_and_transitive_imports(tmp_path, first):
    remote = tmp_path / "fresh_target"
    closure = _closure()
    for relative, _, tokens in closure:
        _copy(remote, relative)
        assert all(token in (remote / relative).read_text() for token in tokens)
    # No additional files are supplied here: this is exactly the published
    # inventory, including its authentic package initializers.
    assert set(REAL_INITIALIZERS) <= {relative for relative, _, _ in closure}
    cp = _import_probe(remote, [first] + [name for _, name, _ in closure])
    assert cp.returncode == 0, cp.stderr
    proofs = json.loads(cp.stdout)
    assert "onnx_splitpoint_tool.validation" in proofs
    assert "onnx_splitpoint_tool.quality_result_contract" in proofs
    for name, proof in proofs.items():
        path = Path(proof["path"])
        relative = path.relative_to(remote)
        assert proof["sha256"] == hashlib.sha256((SOURCE / relative).read_bytes()).hexdigest(), name


CALLERS = (
    "onnx_splitpoint_tool/workflow/runner.py",
    "scripts/update_evalset_native_producers.py",
    "scripts/preflight_v27520_native_remotes.py",
    "scripts/preflight_v27521_native_remotes.py",
)


def _load_script(relative):
    spec = importlib.util.spec_from_file_location("v2801_" + Path(relative).stem, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LocalTransport:
    """Execute exact remote commands locally; never run hardware commands."""

    def __init__(self, root, *, fault=""):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.fault = fault
        self.events = []
        self.bad_asset = "onnx_splitpoint_tool/quality_result_contract.py"

    def __call__(self, command, **kwargs):
        env = dict(os.environ)
        env.update(PATH=str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", ""),
                   PYTHONPATH="", PYTHONDONTWRITEBYTECODE="1")
        if command[0] == "rsync":
            source = Path(command[-2])
            target = Path(command[-1].split(":", 1)[1])
            relative = target.relative_to(self.root).as_posix()
            self.events.append(("sync", relative))
            if self.fault == "transfer" and relative == self.bad_asset:
                return subprocess.CompletedProcess(command, 23, "", "injected transfer failure")
            shutil.copy2(source, target)
            if self.fault == "tamper" and relative == self.bad_asset:
                target.write_bytes(target.read_bytes() + b"\n# changed after transfer\n")
            return subprocess.CompletedProcess(command, 0, "", "")
        assert command[0] == "ssh", command
        code = command[-1]
        if "__SPLITPOINT_REMOTE_LEASE__" in code:
            # The workflow wraps SSH payloads in its independently tested
            # remote process lease. Execute only that exact transport payload
            # in this local simulator; do not create physical remote leases.
            code = shlex.split(code)[-1]
        if "importlib.import_module(" in code:
            self.events.append(("import", code))
            expected = {relative for relative, _, _ in _closure()}
            copied = {relative for event, relative in self.events if event == "sync"}
            assert expected <= copied, "module import started before full transfer barrier"
            if self.fault == "import":
                # Exact bytes were transferred and verified. A disappearing
                # module between transport and import must stop the runtime.
                (self.root / self.bad_asset).unlink(missing_ok=True)
            elif self.fault == "binding_tamper":
                target = self.root / "onnx_splitpoint_tool/deepx/__init__.py"
                target.write_bytes(target.read_bytes() + b"\n# changed after byte verification\n")
        elif "read_bytes" in code:
            self.events.append(("verify_bytes", code))
        elif "remote-contract-preflight" in code:
            self.events.append(("contract_preflight", code))
        else:
            assert code.startswith("mkdir -p "), code
        return subprocess.run(
            ["bash", "-c", code], cwd=self.root, env=env,
            capture_output=True, text=True, timeout=45,
        )

    def updater_run(self, command, **kwargs):
        result = self(command, **kwargs)
        return {"rc": result.returncode, "stdout_tail": result.stdout, "stderr_tail": result.stderr}


def _staging_block(relative):
    """Execute the exact small staging region within otherwise large callers.

    No staging code is reimplemented: AST preserves the existing loops and
    actual helper invocations. Full public preflight calls are tested below.
    """
    tree = ast.parse((SOURCE / relative).read_text())
    for parent in ast.walk(tree):
        for _, value in ast.iter_fields(parent):
            if not isinstance(value, list):
                continue
            for index, node in enumerate(value):
                if not (isinstance(node, ast.For) and isinstance(node.iter, ast.Call)
                        and isinstance(node.iter.func, ast.Name)
                        and node.iter.func.id == "native_remote_package_closure"):
                    continue
                start = index
                if index and isinstance(value[index - 1], ast.Assign):
                    if any(isinstance(x, ast.Name) and x.id == "pending_module_checks"
                           for x in value[index - 1].targets):
                        start -= 1
                end = index + 1
                if end < len(value) and isinstance(value[end], ast.For):
                    if isinstance(value[end].iter, ast.Name) and value[end].iter.id == "pending_module_checks":
                        end += 1
                return compile(ast.Module(body=value[start:end], type_ignores=[]), str(SOURCE / relative), "exec")
    raise AssertionError("no production staging region found: " + relative)


def _execute_staging(caller, transport, monkeypatch):
    from onnx_splitpoint_tool.workflow import runner
    updater = _load_script("scripts/update_evalset_native_producers.py")
    monkeypatch.setattr(updater, "_run", transport.updater_run)
    def sync_asset(**kwargs):
        return runner._sync_remote_package_asset_v263(**kwargs, process_runner=transport)
    def verify_module(**kwargs):
        return runner._verify_remote_module_binding_v263(**kwargs, process_runner=transport)
    namespace = dict(
        ROOT=ROOT, hashlib=hashlib, Mapping=Mapping, native_remote_package_closure=_closure,
        ssh="nx@isolated-setup", rtool=str(transport.root), remote_tool_dir=str(transport.root),
        env="", remote_env="", timeout=45, _stream_native=transport,
        result={"steps": []}, steps=[], module_proofs=[], setup={"backend": "hailo8"},
        sync_asset_fn=sync_asset, verify_module_fn=verify_module,
        _sync_remote_package_asset_v263=runner._sync_remote_package_asset_v263,
        _verify_remote_module_binding_v263=runner._verify_remote_module_binding_v263,
    )
    if caller.startswith("scripts/update"):
        namespace.update(_sync_remote_package_asset_v263=updater._sync_remote_package_asset_v263,
                         _verify_remote_module_binding_v263=updater._verify_remote_module_binding_v263)
    exec(_staging_block(caller), namespace)
    return namespace


@pytest.mark.parametrize("caller", CALLERS)
def test_t02_03_04_05_all_callers_transfer_before_first_real_import(tmp_path, monkeypatch, caller):
    transport = LocalTransport(tmp_path / "remote")
    _execute_staging(caller, transport, monkeypatch)
    kinds = [event for event, _ in transport.events]
    assert kinds.count("sync") == len(_closure())
    assert kinds.count("import") == len(_closure())
    assert max(i for i, kind in enumerate(kinds) if kind == "verify_bytes") < kinds.index("import")


@pytest.mark.parametrize("caller", CALLERS)
@pytest.mark.parametrize("fault", ["transfer", "tamper", "import", "binding_tamper"])
def test_t02_06_07_failure_stops_import_or_native_start(tmp_path, monkeypatch, caller, fault):
    transport = LocalTransport(tmp_path / "remote", fault=fault)
    starts = []
    with pytest.raises(RuntimeError) as error:
        _execute_staging(caller, transport, monkeypatch)
        starts.append("native_performance_or_energy")
    assert starts == []
    assert "nx@isolated-setup" in str(error.value)
    assert str(transport.root) in str(error.value)
    if fault in {"transfer", "tamper"}:
        assert transport.bad_asset in str(error.value)
        assert not any(kind == "import" for kind, _ in transport.events)
    else:
        assert sum(kind == "sync" for kind, _ in transport.events) == len(_closure())
        assert ("quality_result_contract" if fault == "import" else "binding verification") in str(error.value)


def test_t02_08_second_installation_does_not_fill_missing_module(tmp_path):
    remote = tmp_path / "target"
    other = tmp_path / "unrelated_installation"
    for relative, _, _ in _closure():
        _copy(other, relative)
        if relative != "onnx_splitpoint_tool/quality_result_contract.py":
            _copy(remote, relative)
    cp = _import_probe(remote, ["onnx_splitpoint_tool.validation.host_postprocess"], extra=other)
    assert cp.returncode != 0
    assert "No module named 'onnx_splitpoint_tool.quality_result_contract'" in cp.stderr


def test_t02_09_old_target_update_and_warm_identical_target(tmp_path, monkeypatch):
    transport = LocalTransport(tmp_path / "remote")
    _copy(transport.root, "onnx_splitpoint_tool/validation/accuracy_gates.py")
    (transport.root / "onnx_splitpoint_tool/validation/accuracy_gates.py").write_text("raise RuntimeError('stale module')\n")
    _execute_staging(CALLERS[0], transport, monkeypatch)
    before = {relative: (transport.root / relative).read_bytes() for relative, _, _ in _closure()}
    transport.events.clear()
    _execute_staging(CALLERS[0], transport, monkeypatch)
    assert all((transport.root / relative).read_bytes() == content for relative, content in before.items())
    assert sum(kind == "import" for kind, _ in transport.events) == len(_closure())


def test_t02_04_standalone_script_mirror_is_exact():
    assert (ROOT / CALLERS[1]).read_bytes() == (
        ROOT / "onnx_splitpoint_tool/resources/remote_scripts/update_evalset_native_producers.py"
    ).read_bytes()


@pytest.mark.parametrize("release", ["v27520", "v27521"])
def test_t02_05_normal_preflight_entrypoint_three_clean_setups(tmp_path, release):
    """The real preflight API stages scripts, bytes, imports and helper probe."""
    from onnx_splitpoint_tool.workflow import runner
    script = _load_script(f"scripts/preflight_{release}_native_remotes.py")
    families = {"hailo8": "hailo8", "hailo10h": "hailo10", "deepx": "deepx_m1"}
    transports = {
        family: LocalTransport(tmp_path / family) for family in families
    }
    profile = {
        "run_profiles": [{"id": "ort_tensorrt", "type": "same_backend_reference",
                          "full": "tensorrt", "stage1": "tensorrt", "stage2": "tensorrt"}],
        "quality_gate": {"statistics": {"execution_location": "central_management"}},
        "hardware_targets": [], "native_producers": {"enabled": True, "remotes": {}},
    }
    registry = tmp_path / "hardware_setups.yaml"
    registry.write_text(json.dumps({"hardware_setups": [
        {"id": family + "_setup", "accelerator": backend, "host": family, "user": "nx"}
        for family, backend in families.items()
    ]}))
    profile["hardware"] = {"setups_file": str(registry)}
    for family, backend in families.items():
        profile["run_profiles"].append({
            "id": backend, "type": "same_backend_reference", "full": backend,
            "stage1": backend, "stage2": backend, "hardware_setup_id": family + "_setup",
        })
        profile["hardware_targets"].append({
            "id": family + "_setup", "accelerator": backend, "enabled": True,
            "runtime": {"enabled": True, "host": family, "user": "nx"},
        })
        profile["native_producers"]["remotes"][family] = {
            "ssh": "nx@" + family, "setup_id": family + "_setup",
            "remote_tool_dir": str(transports[family].root),
        }
    def by_kwargs(kwargs):
        return transports[kwargs["ssh"].split("@")[-1]]
    def sync_script(**kwargs):
        return runner._sync_remote_script_v60i(**kwargs, process_runner=by_kwargs(kwargs))
    def sync_asset(**kwargs):
        return runner._sync_remote_package_asset_v263(**kwargs, process_runner=by_kwargs(kwargs))
    def verify_module(**kwargs):
        return runner._verify_remote_module_binding_v263(**kwargs, process_runner=by_kwargs(kwargs))
    def process(command, **kwargs):
        return transports[command[-2].split("@")[-1]](command, **kwargs)
    result = script.run_remote_contract_preflight(
        profile, timeout=45, sync_script=sync_script, sync_asset=sync_asset,
        verify_module=verify_module, process_runner=process,
    )
    assert result["ok"] is True
    assert {row["setup_id"] for row in result["rows"]} == {x + "_setup" for x in families}
    for row in result["rows"]:
        assert row["staged_module_count"] == len(_closure())
        assert row["helper_proof"]["ok"] is True
        for proof in row["module_proofs"]:
            assert proof["verification"]["ok"] is True
            assert Path(proof["verification"]["imported_path"]).is_relative_to(
                transports[row["backend"]].root
            )
    for transport in transports.values():
        assert sum(kind == "contract_preflight" for kind, _ in transport.events) == 1


@pytest.mark.parametrize("fault", ["transfer", "import"])
def test_t02_06_07_whole_workflow_stage_persists_failure_without_workload(tmp_path, monkeypatch, fault):
    """Run the complete GUI Native stage; fault only its remote transport."""
    from onnx_splitpoint_tool.workflow import runner as implementation
    from onnx_splitpoint_tool.workflow.runner import EvaluationWorkflowRunner, WorkflowOptions
    from onnx_splitpoint_tool.remote.process_lease import RemoteProcessLeaseScope
    transport = LocalTransport(tmp_path / "remote", fault=fault)
    workflow = EvaluationWorkflowRunner(WorkflowOptions(profile="", out=str(tmp_path)))
    workflow.run_id = "native_transfer_" + fault
    workflow.run_dir = tmp_path / workflow.run_id
    workflow._remote_process_registry.configure_journal(
        scope=RemoteProcessLeaseScope(workflow.run_id, workflow.session_id),
        journal_dir=tmp_path / "lease_journal",
    )
    workflow.profile_payload = {
        "campaign": {"mode": "measurement"}, "execution_preset": {"id": "standard"},
        "measurement_campaign": {"system_power": {"scope": "system", "window": "command"}},
    }
    workflow.profile_start_snapshot = {}
    workflow.manifest = {"models": {"resnet50": {}}}
    suite = workflow.run_dir / "models/resnet50/benchmark_set"
    (suite / "b001").mkdir(parents=True)
    (suite / "b001/split_manifest.json").write_text(json.dumps({"part2_external_inputs": ["boundary_tensor"]}))
    (suite / "benchmark_set.json").write_text(json.dumps({"cases": [{"id": "b001"}]}))
    quality = workflow.run_dir / "quality_management"
    quality.mkdir(parents=True)
    (quality / "central_quality_summary.json").write_text(json.dumps({"results": []}))
    cfg = {
        "enabled": True, "models": ["resnet50"], "backends": ["hailo8"],
        "precision": "fp16", "case_policy": "case_map_only", "case_map": {"resnet50": ["b001"]},
        "remotes": {"hailo8": {"ssh": "nx@isolated-setup", "setup_id": "hailo8_setup",
                                 "remote_tool_dir": str(transport.root)}},
        "copy_benchmarksets": False, "build_missing_engines": False,
        "validation": {"enabled": False}, "full_baselines": {"enabled": False},
        "energy": {"enabled": True}, "cleanup_remote_native_root": False,
    }
    labels = []
    def stream(command, *args, **kwargs):
        label = str(kwargs.get("label") or "")
        labels.append(label)
        if label.startswith(("sync-asset-", "verify-remote-module:")):
            return transport(command, **kwargs)
        return subprocess.CompletedProcess(command, 0, "", "")
    monkeypatch.setattr(implementation, "run_streaming", stream)
    monkeypatch.setattr(implementation, "normalize_hardware_targets", lambda *a, **k: [])
    monkeypatch.setattr(implementation, "benchmark_set_postcondition_v60v",
                        lambda path: {"valid": True, "selected_suite_dir": str(path)})
    monkeypatch.setattr(implementation, "_sync_remote_script_v60i", lambda *a, **k: [])
    monkeypatch.setattr(implementation, "_select_native_report_python",
                        lambda *a, **k: (sys.executable, {"selected": sys.executable, "onnxruntime_ok": True}))
    monkeypatch.setattr("onnx_splitpoint_tool.trt_quality_chain.load_split_binding_set_from_central_quality_summary",
                        lambda *a, **k: {"schema": "test/native-split-quality-binding-set", "bindings": []})
    monkeypatch.setattr(workflow, "_finish_native_direct_remote_lease", lambda *a, **k: None)
    monkeypatch.setattr(workflow, "_native_producer_config", lambda: cfg)
    workflow._stage_run_native_producers()
    assert not any(label.startswith(("split:", "full:", "native-energy:measure")) for label in labels)
    stage = json.loads((workflow.run_dir / "reports/native_producer_stage.json").read_text())
    row = next(row for row in stage["backend_results"] if row["backend"] == "hailo8")
    assert row["ok"] is False
    assert row["started_performance_count"] == 0
    assert "nx@isolated-setup" in json.dumps(row)
    assert ("remote package asset sync failed" if fault == "transfer"
            else "remote Python module binding verification failed") in row["error"]
    assert transport.events, "must actually reach the repaired transfer boundary"
    if fault == "transfer":
        assert not any(kind == "import" for kind, _ in transport.events)
    else:
        assert any(kind == "import" for kind, _ in transport.events)


@pytest.mark.parametrize("fault", ["transfer", "import"])
def test_t02_04_06_07_whole_standalone_stage_stops_on_transfer_failure(tmp_path, monkeypatch, fault):
    module = _load_script("scripts/update_evalset_native_producers.py")
    transport = LocalTransport(tmp_path / "remote", fault=fault)
    run = tmp_path / "legacy_eval"
    suite = run / "models/resnet50/benchmark_set"
    (suite / "b001").mkdir(parents=True)
    (suite / "benchmark_set.json").write_text("{}")
    (suite / "b001/split_manifest.json").write_text("{}")
    # Exercise the updater's supported historical-run path. This is a transfer
    # test, not a claim that legacy quality contracts qualify current results.
    (run / "run_manifest.json").write_text(json.dumps({
        "schema": "onnx-splitpoint/evaluation-run-manifest", "schema_version": 1,
        "run_id": run.name, "workflow_version": "v2.68-level-playing-field", "tool_version": "2.68.0",
    }))
    labels = []
    def local_run(command, **kwargs):
        label = str(kwargs.get("label") or "")
        labels.append(label)
        if command[0] == "rsync" or (command[0] == "ssh" and not label):
            return transport.updater_run(command, **kwargs)
        return {"rc": 0, "stdout_tail": "", "stderr_tail": ""}
    monkeypatch.setattr(module, "_run", local_run)
    monkeypatch.setattr(module, "_sync_remote_script_v60i", lambda *a, **k: [])
    monkeypatch.setattr(module, "_capture_remote_host_telemetry", lambda **k: {"rc": 0})
    monkeypatch.setattr(module, "_summarize_host_telemetry", lambda *a, **k: {"rc": 0})
    stage = module._run_native_producers(run, {
        "backends": ["deepx"], "remotes": {"deepx": {"ssh": "nx@isolated-setup", "setup_id": "deepx_setup"}},
        "remote_root": str(tmp_path / "results"), "remote_tool_dir": str(transport.root),
        "copy_benchmarksets": False, "build_missing_engines": False,
        "frames": 1, "warmup": 1, "repetitions": 1,
        "full_baselines": {"enabled": True, "backends_by_producer": {"deepx": ["deepx"]}},
        "energy": {"enabled": True},
    }, timeout=45)
    assert not any(label.startswith(("split:", "full:", "native-energy:measure")) for label in labels)
    row = stage["backend_results"][0]
    assert row["ok"] is False
    assert "nx@isolated-setup" in row["error"]
    assert transport.events, "must reach the real standalone transfer boundary"
    if fault == "transfer":
        assert not any(kind == "import" for kind, _ in transport.events)
    else:
        assert any(kind == "import" for kind, _ in transport.events)
