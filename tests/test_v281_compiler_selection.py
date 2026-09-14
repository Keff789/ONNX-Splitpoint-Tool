"""Compiler barrier and configuration repair; no vendor/HW acceptance claims."""
import copy
import fcntl
import json
from pathlib import Path

import pytest
import yaml

from onnx_splitpoint_tool import hailo_backend as backend
from onnx_splitpoint_tool import hailo_compiler_context as cc
from onnx_splitpoint_tool import hailo_overlay_migration as migration
from onnx_splitpoint_tool import run_modes
from onnx_splitpoint_tool.gui import run_mode_editor as gui
from onnx_splitpoint_tool.workflow.hailo_compiler_preflight import preflight_cold_hailo_contexts
from test_v27934_compiler_context import vendor
from test_v280_hailo8_compute_env import boundary


def row(status="MISS", family="hailo8", ready=False):
    return {"model_id": "yolo11l", "boundary": "b064", "item_id": "b064:part1",
            "role": "hailo8_hef" if family == "hailo8" else "hailo10_hef",
            "status": status, "compiler_dispatch_allowed": status == "MISS" and not ready,
            "runtime_artifact_available": ready}


@pytest.mark.parametrize("status,ready", [("HIT", False), ("KNOWN_INFEASIBLE", False), ("MISS", True), ("UNKNOWN", False)])
def test_reusable_or_blocked_rows_do_not_resolve_any_venv_or_gpu(tmp_path, monkeypatch, status, ready):
    monkeypatch.setattr(backend, "_resolve_managed_venv_python", lambda **_k: pytest.fail("non-cold compiler invoked"))
    result = preflight_cold_hailo_contexts({"cold_build_rows": [row(status, ready=ready)]},
        {"hailo_build": {"compute_by_family": {"hailo8": {"device": "gpu", "dependency_manifest": "/absent"}}}}, work_dir=tmp_path)
    assert result["status"] == "not_required" and result["compiler_dispatch_allowed"]


@pytest.mark.parametrize("policy", ["strict_blocked", "verify_only", "captured_cache_only"])
def test_dispatch_prohibition_does_not_probe_compiler(tmp_path, monkeypatch, policy):
    monkeypatch.setattr(backend, "_resolve_managed_venv_python", lambda **_k: pytest.fail("prohibited compiler probe"))
    report = {"cold_build_rows": [row()]}
    profile = {}
    if policy == "strict_blocked":
        report["runtime_dispatch_allowed"] = False
    elif policy == "verify_only":
        profile["execution_guard"] = {"mode": "cache_verify_only"}
    else:
        report["cold_build_rows"][0]["evidence"] = {"compiler_cache_only": True}
    result = preflight_cold_hailo_contexts(report, profile, work_dir=tmp_path)
    assert result["status"] == "not_required" and result["contexts"] == []


def test_required_cold_h8_fails_before_build_without_components(boundary, monkeypatch, tmp_path):
    monkeypatch.delenv(cc.DEPENDENCY_MANIFEST_ENV, raising=False)
    monkeypatch.setattr(backend, "_resolve_managed_venv_python", lambda **_k: ("h8", boundary["h8"] / "bin/python", "activate"))
    result = preflight_cold_hailo_contexts({"cold_build_rows": [row()]},
        {"hailo_build": {"compute_by_family": {"hailo8": {"device": "gpu"}}}}, work_dir=tmp_path / "preflight")
    assert result["status"] == "failed" and result["compiler_dispatch_allowed"] is False
    assert result["contexts"][0]["reason"] == "hailo_compiler_components_missing"
    assert result["model_build"] == "NOT_RUN"


def test_normal_workflow_barrier_stops_before_any_backend_or_runtime_dispatch(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from onnx_splitpoint_tool.workflow.runner import EvaluationWorkflowRunner
    from test_v27921_final_selection_cache_preflight import _yolo_suite, _write
    runner = object.__new__(EvaluationWorkflowRunner)
    runner._stop_requested = False
    runner.jobs = None
    runner.run_dir = tmp_path
    runner.profile_payload = {}
    runner.options = SimpleNamespace(benchmark_execution_backend="local")
    runner.outputs = {}
    runner.report_paths = []
    runner._targets = lambda: ["hailo10h"]
    runner._materialize_local_cache_preflight_inputs = lambda *_a: []
    _yolo_suite(tmp_path)
    _write(tmp_path / "models/yolo26m/benchmark_set/legacy_suite/b398/hailo/hailo10/part1/hailo_cache_miss.json", {
        "net_name": "yolo26m_part1_b398", "hw_arch": "hailo10h", "probe_outcomes": {}})
    def missing(**_kw):
        raise cc.CompilerContextError("hailo_compiler_components_missing", "fixture missing selected components")
    monkeypatch.setattr(backend, "_resolve_managed_venv_python", missing)
    logs = []
    dispatched = []
    runner._emit_log = logs.append
    def stage(_model, _name, action):
        _, details, _, status = action()
        return SimpleNamespace(status=status, details=details)
    runner._run_stage = stage
    runner._run_model = lambda _row, *, stage_names, **_kw: dispatched.extend(stage_names)
    runner._run_model_pipeline_with_cache_preflight([{"id": "yolo26m"}])
    assert runner._stop_requested
    assert "build_backend_artifacts" not in dispatched and "run_benchmarks" not in dispatched
    report = json.loads((tmp_path / "reports/hailo_compiler_preflight.json").read_text())
    assert report["status"] == "failed" and report["contexts"][0]["cold_rows"][0]["boundary"] == "b398"
    assert any("hailo_compiler_components_missing" in line for line in logs)


def test_required_cold_h8_selected_overlay_reaches_real_child_environment(boundary, monkeypatch, tmp_path):
    monkeypatch.setattr(backend, "_resolve_managed_venv_python", lambda **_k: ("h8", boundary["h8"] / "bin/python", "activate"))
    profile = {"hailo_build": {"compute_by_family": {"hailo8": {"device": "gpu", "dependency_manifest": str(boundary["manifest"])}}}}
    result = preflight_cold_hailo_contexts({"cold_build_rows": [row(), row()]}, profile, work_dir=tmp_path / "preflight")
    assert result["status"] == "pass" and len(result["contexts"]) == 1
    context = result["contexts"][0]["context"]
    assert context["dependency_manifest_source"] == "compute_by_family.hailo8"
    assert context["dependency_library_dirs"] and context["target_probe"]["output_size"] > 0
    assert not Path(context["view_root"]).exists()


def test_valid_local_gpu_venv_needs_no_overlay(vendor, monkeypatch, tmp_path):
    monkeypatch.setattr(backend, "_resolve_managed_venv_python", lambda **_k: ("fixture", vendor["python"], "activate"))
    monkeypatch.setattr(backend, "_managed_venv_child_env", lambda _p: vendor["env"])
    result = preflight_cold_hailo_contexts({"cold_build_rows": [row(family="hailo10h")]},
        {"hailo_build": {"compute_by_family": {"hailo10h": {"device": "gpu"}}}}, work_dir=tmp_path)
    assert result["status"] == "pass"
    assert result["contexts"][0]["context"]["component_source"] == "selected_venv_triton"


def test_cpu_coldbuild_never_inspects_gpu_or_deleted_overlay(vendor, monkeypatch, tmp_path):
    monkeypatch.setattr(backend, "_resolve_managed_venv_python", lambda **_k: ("fixture", vendor["python"], "activate"))
    monkeypatch.setattr(cc, "_gpu_target", lambda *_a: pytest.fail("CPU GPU probe"))
    result = preflight_cold_hailo_contexts({"cold_build_rows": [row()]},
        {"hailo_build": {"compute_by_family": {"hailo8": {"device": "cpu", "dependency_manifest": "/absent"}}}}, work_dir=tmp_path)
    assert result["status"] == "pass"


def test_pasted_gui_path_is_saved_even_without_checkbox():
    mode = {}
    path = "build.hailo.compute_by_family.hailo8.dependency_manifest"
    gui.apply_optional_manifest(mode, path, "  /pasted/overlay.json  ", False)
    assert gui._get_path(mode, path) == "/pasted/overlay.json"
    gui.apply_optional_manifest(mode, path, "", True)
    assert gui._get_path(mode, path) == ""
    gui.apply_optional_manifest(mode, path, "", False)
    assert gui._get_path(mode, path, None) is None


@pytest.mark.parametrize("entry", [{"device": "gpu", "dependency_manifest": ""}, {"device": "gpu", "dependency_manifest": "/other"}, {"device": "cpu"}])
def test_migration_preserves_explicit_none_other_and_cpu(entry):
    before = {"hailo_build": {"compute_by_family": {"hailo8": entry, "hailo10h": {"device": "gpu"}}}, "quality": {"margin": 1}}
    after, changed = migration.migrate_payload(before, "/reviewed")
    assert not changed and after == before


def test_payload_preserves_legacy_field_without_treating_it_as_runtime_selection():
    before = {"hailo_build": {"venv_activate": "/other/complete_venv/bin/activate",
               "compute_by_family": {"hailo8": {"device": "gpu"}}}}
    after, changed = migration.migrate_payload(before, "/reviewed")
    assert changed
    assert after["hailo_build"]["venv_activate"] == before["hailo_build"]["venv_activate"]


def test_migration_updates_snapshot_hash_and_preserves_nonh8():
    snapshot = {"build": {"hailo": {"compute_by_family": {"hailo8": {"device": "gpu"}, "hailo10h": {"device": "gpu"}}}}, "quality": {"bootstrap_repetitions": 5000}}
    before = {"hailo_build": copy.deepcopy(snapshot["build"]["hailo"]), "execution_preset": {"snapshot": snapshot, "snapshot_sha256": run_modes._json_hash(snapshot)}}
    after, changed = migration.migrate_payload(before, "/reviewed")
    assert after["execution_preset"]["snapshot_sha256"] == run_modes._json_hash(after["execution_preset"]["snapshot"])
    assert after["execution_preset"]["snapshot_sha256"] != before["execution_preset"]["snapshot_sha256"]
    assert after["execution_preset"]["snapshot"]["quality"] == snapshot["quality"]
    assert after["hailo_build"]["compute_by_family"]["hailo10h"] == {"device": "gpu"}
    assert migration.migrate_payload(after, "/reviewed") == (after, [])


def setup_migration(tmp_path, monkeypatch):
    tool = tmp_path / "tool"
    (tool / "profiles").mkdir(parents=True)
    binding = tool / "onnx_splitpoint_tool/resources/hailo/reviewed_overlay_v281.json"
    binding.parent.mkdir(parents=True)
    binding.write_text(json.dumps({"selection": {"path": "/reviewed", "selected_venv": "/reviewed_venv"}}))
    profile = tool / "profiles/CompleteSetDev.yaml"
    profile.write_text(yaml.safe_dump({"hailo_build": {"compute_by_family": {"hailo8": {"device": "gpu"}}}, "deepx_build": {"force_build": False}}))
    monkeypatch.setattr(migration, "reviewed_selection", lambda *_a: {"dependency_manifest": "/reviewed", "manifest_sha256": "fixture"})
    monkeypatch.setattr(backend, "_resolve_managed_venv_python", lambda **_kw:
                        ("hailo8", Path("/reviewed_venv/bin/python"), "/reviewed_venv/bin/activate"))
    return tool, profile


@pytest.mark.parametrize("entry", [{"device": "cpu"}, {"device": "gpu", "dependency_manifest": ""},
                                  {"device": "gpu", "dependency_manifest": "/other"}])
def test_no_eligible_config_does_not_inspect_historical_overlay(tmp_path, monkeypatch, entry):
    tool, profile = setup_migration(tmp_path, monkeypatch)
    profile.write_text(yaml.safe_dump({"hailo_build": {"compute_by_family": {"hailo8": entry}}}))
    before = profile.read_bytes()
    monkeypatch.setattr(migration, "reviewed_selection", lambda *_a: pytest.fail("no-op inspected overlay"))
    result = migration.migrate_configs(tool, tmp_path / "report", registry_path=tmp_path / "absent")
    assert result["status"] == "not_required" and result["dependency_manifest"] is None
    assert not result["changed_files"] and profile.read_bytes() == before
    assert not (tmp_path / "report/private_backups").exists()


@pytest.mark.parametrize("selection_source", ["family_environment", "persistent_registry"])
def test_real_dfc_manager_other_venv_noop_does_not_require_reviewed_overlay(tmp_path, monkeypatch, selection_source):
    import sys
    from onnx_splitpoint_tool.runners.backends import hailo_utils
    real_resolver = backend._resolve_managed_venv_python
    tool, profile = setup_migration(tmp_path, monkeypatch)
    monkeypatch.setattr(backend, "_resolve_managed_venv_python", real_resolver)
    alternate = tmp_path / "actual_selected_venv"
    (alternate / "bin").mkdir(parents=True)
    (alternate / "bin/python").symlink_to(sys.executable)
    (alternate / "bin/activate").write_text("# fixture selected by the real DfcManager\n")
    registry = tmp_path / "persistent_hailo_resources"
    registry.mkdir()
    activate = str(alternate / "bin/activate")
    (registry / "profiles.json").write_text(json.dumps({"profiles": {"hailo8": {
        "profile_id": "hailo8", "hw_arch_prefixes": ["hailo8"],
        "wsl_venv_activate": activate if selection_source == "persistent_registry" else "/reviewed_venv/bin/activate"}}}))
    monkeypatch.setattr(hailo_utils.DfcManager, "_resources_root", staticmethod(lambda: registry))
    monkeypatch.setattr(hailo_utils, "_DEFAULT_MANAGER", None)
    if selection_source == "family_environment":
        monkeypatch.setenv("ONNX_SPLITPOINT_HAILO_VENV_HAILO8", activate)
    else:
        monkeypatch.delenv("ONNX_SPLITPOINT_HAILO_VENV_HAILO8", raising=False)
    before = profile.read_bytes()
    registry_before = (registry / "profiles.json").read_bytes()
    monkeypatch.setattr(migration, "reviewed_selection", lambda *_a: pytest.fail("other venv inspected overlay"))
    result = migration.migrate_configs(tool, tmp_path / "report", registry_path=tmp_path / "absent")
    assert result["status"] == "not_required" and profile.read_bytes() == before
    assert result["reason"] == "selected_hailo8_venv_differs_from_reviewed"
    assert result["dfc_runtime_selection"]["selected_python"] == str(alternate / "bin/python")
    assert (registry / "profiles.json").read_bytes() == registry_before
    assert not (tmp_path / "report/private_backups").exists()


def test_actual_config_backup_apply_and_idempotence(tmp_path, monkeypatch):
    tool, profile = setup_migration(tmp_path, monkeypatch)
    original = profile.read_bytes()
    registry = tmp_path / "run_modes.yaml"
    cfg = run_modes.default_run_modes_config()
    cfg["modes"]["standard"]["build"]["hailo"]["compute_by_family"]["hailo8"] = {"device": "gpu"}
    registry.write_text(yaml.safe_dump(cfg))
    report = migration.migrate_configs(tool, tmp_path / "report", registry_path=registry)
    assert report["status"] == "pass" and len(report["changed_files"]) == 2
    assert original in [p.read_bytes() for p in (tmp_path / "report/private_backups").iterdir()]
    after = profile.read_bytes()
    again = migration.migrate_configs(tool, tmp_path / "again", registry_path=registry)
    assert again["changed_files"] == [] and profile.read_bytes() == after
    resolved, _ = run_modes.apply_run_mode(yaml.safe_load(after), config=yaml.safe_load(registry.read_bytes()))
    assert resolved["hailo_build"]["compute_by_family"]["hailo8"]["dependency_manifest"] == "/reviewed"


def test_busy_workflow_lock_causes_no_config_writes(tmp_path, monkeypatch):
    from onnx_splitpoint_tool.workflow.run_control import platform_workflow_interlock_path
    tool, profile = setup_migration(tmp_path, monkeypatch)
    original = profile.read_bytes()
    lock = platform_workflow_interlock_path()
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a+b") as held:
        fcntl.flock(held, fcntl.LOCK_SH | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            migration.migrate_configs(tool, tmp_path / "blocked", registry_path=tmp_path / "absent")
    assert profile.read_bytes() == original


def test_partial_config_write_failure_rolls_back_earlier_files(tmp_path, monkeypatch):
    tool, profile = setup_migration(tmp_path, monkeypatch)
    second = tool / "profiles/Z.yaml"
    second.write_bytes(profile.read_bytes())
    original = profile.read_bytes()
    real = migration._atomic_bytes
    def fail(path, content, mode=0o600):
        if path == second:
            raise OSError("fixture disk write failure")
        return real(path, content, mode)
    monkeypatch.setattr(migration, "_atomic_bytes", fail)
    with pytest.raises(OSError, match="fixture disk"):
        migration.migrate_configs(tool, tmp_path / "failed", registry_path=tmp_path / "absent")
    assert profile.read_bytes() == second.read_bytes() == original


def test_reviewed_manifest_and_components_are_revalidated(boundary, tmp_path):
    tool = tmp_path / "binding_tool"
    resource = tool / "onnx_splitpoint_tool/resources/hailo/reviewed_overlay_v281.json"
    resource.parent.mkdir(parents=True)
    from onnx_splitpoint_tool.hailo_dependency_plan import validated_overlay_components
    components = validated_overlay_components(family="hailo8", selected_python=boundary["h8"] / "bin/python", manifest_path=boundary["manifest"])
    resource.write_text(json.dumps({"source_evidence_sha256": "fixture", "selection": {
        "path": str(boundary["manifest"]), "manifest_sha256": migration.digest(boundary["manifest"]),
        "selected_venv": str(boundary["h8"]), "component_sha256": {k: migration.digest(Path(components[k])) for k in ("ptxas_path", "libdevice_path")}}}))
    assert migration.reviewed_selection(tool)["dependency_manifest"] == str(boundary["manifest"])
    Path(components["libdevice_path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="reviewed_component_changed"):
        migration.reviewed_selection(tool)
