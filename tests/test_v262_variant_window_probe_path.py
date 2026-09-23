from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import pytest

from onnx_splitpoint_tool.remote.process_lease import RemoteProcessLeaseScope

from onnx_splitpoint_tool.workflow.runner import EvaluationWorkflowRunner, WorkflowOptions


ROOT = Path(__file__).resolve().parents[1]


def _load_script(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_legacy_run_manifest(run_dir: Path) -> None:
    """Bind historical coordinator fixtures to their pre-2.69f policy."""
    _write_json(run_dir / "run_manifest.json", {
        "schema": "onnx-splitpoint/evaluation-run-manifest",
        "schema_version": 1,
        "run_id": run_dir.name,
        "workflow_version": "v2.62-window-validation-native-binding",
        "tool_version": "2.62.0",
    })


def _private_workflow_context(runner):
    """Real session journal and profile; only process/device leaves are faked."""
    runner.profile_payload["quality_gate"] = {
        "schema": "onnx-splitpoint/task-quality-policy", "schema_version": 3,
        "name": "r6_fixture_quality", "profile_id": "r6_fixture_quality",
    }
    _write_json(runner.run_dir / "profile.yaml", runner.profile_payload)
    from onnx_splitpoint_tool.release_identity import VERSION, BUILD_ID
    _write_json(runner.run_dir / "run_manifest.json", {
        "schema": "onnx-splitpoint/evaluation-run-manifest", "schema_version": 1,
        "run_id": runner.run_id, "workflow_version": BUILD_ID, "tool_version": VERSION,
    })
    runner._remote_process_registry.configure_journal(
        scope=RemoteProcessLeaseScope(runner.run_id, runner.session_id),
        journal_dir=runner.run_dir / "jobs/remote_process_leases" / runner.session_id,
    )


def _assert_journal_environment(runner, kwargs):
    expected = runner._remote_process_registry.journal_environment()
    assert expected
    assert all(kwargs["env"].get(k) == v for k, v in expected.items())


def _terminal_child_handoff(runner, args, *, rc):
    """Commit the real handoff contract for the controlled child process leaf."""
    from onnx_splitpoint_tool.workflow.checkpoints import native_coordinator_input_hash, write_stage_checkpoint
    from onnx_splitpoint_tool.validation.accuracy_gates import AccuracyGatePolicy
    cmd = args[0]
    config_path = Path(cmd[cmd.index("--config") + 1])
    stage_path = runner.run_dir / "reports/native_producer_stage.json"
    stage = json.loads(stage_path.read_text())
    state = "failed" if rc else "completed"
    stage.update(state=state, complete=True)
    _write_json(stage_path, stage)
    write_stage_checkpoint(
        runner.run_dir / "stages/run_native_producers/native_coordinator/stage_result.json",
        stage="native_coordinator", state=state, complete=True,
        input_hash=native_coordinator_input_hash(runner.run_dir, config_path,
            quality_gate_policy_sha256=AccuracyGatePolicy.from_mapping(runner.profile_payload["quality_gate"]).sha256()),
        run_root=runner.run_dir, artifacts=[stage_path],
        details={"return_code":rc,"stage_complete":True,"stage_state":state})


def test_variant_coordinator_keeps_strict_smoke_probe_nonblocking_with_green_baseline(
    tmp_path: Path, monkeypatch,
) -> None:
    coordinator = _load_script(
        ROOT / "scripts" / "run_evalrun_native_producer_variants.py",
        "v262_variant_coordinator_test",
    )
    run_dir = tmp_path / "eval-native-split-001"
    reports = run_dir / "reports"
    (run_dir / "native_producers" / "hailo8").mkdir(parents=True)
    _write_legacy_run_manifest(run_dir)
    suite = run_dir / "models/yolo26s/benchmark_set"
    (suite / "b038").mkdir(parents=True)
    _write_json(suite / "benchmark_set.json", {"benchmark_task": "detection", "cases": [{"id": "b038"}]})
    _write_json(suite / "b038/split_manifest.json", {"part2_external_inputs": ["boundary_tensor"]})
    _write_json(run_dir / "profile.yaml", {"quality_gate": {
        "schema": "onnx-splitpoint/task-quality-policy", "schema_version": 3,
        "name": "r6_fixture_quality", "profile_id": "r6_fixture_quality",
    }})
    config = {
        "_workflow_context": {
            "campaign": {"mode": "development"},
            "execution_preset": {"id": "smoke"},
        },
        "variants": [{"id": "resnet", "case_map": {"yolo26s": ["b038"]}}],
        "validation": {"enabled": True},
        "remotes": {"hailo8": {"ssh": "nx@host", "env": "source env", "setup_id": "hailo8_setup"}},
        "energy": {
            "enabled": False,
            "window_method_validation_probe": {
                "enabled": True,
                "repeats": 3,
                "include_raw_parquet": True,
                "strict": True,
            },
        },
    }
    from tests.test_v269f_variant_native_split_quality_first import _binding_and_summary, _write_summary
    _binding, summary = _binding_and_summary(tmp_path / "quality_fixture")
    _write_summary(run_dir, summary)
    config["precision"] = "uint8_dequant_fp16"
    config_path = tmp_path / "native.json"
    _write_json(config_path, config)
    calls: list[str] = []

    def fake_run(cmd, *, timeout=None, cwd=None, label="native-child"):
        calls.append(label)
        if label == "final_report":
            _write_json(reports / "native_producer_combined_summary.json", {
                "rows": [{
                    "backend": "hailo8_to_trt", "model": "yolo26s", "case": "b038",
                    "precision": "uint8_dequant_fp16", "setup_id": "hailo8_setup", "ok": True,
                    "fps_makespan": 100.0,
                }],
            })
            return {"rc": 0, "stdout_tail": "", "stderr_tail": ""}
        if label == "native_validation":
            _write_json(
                reports / "native_validation" / "native_producer_validation_summary.json",
                {
                    "status": "complete",
                    "technical_error_count": 0,
                    "row_count": 1,
                    "rows": [{"ok": True, "status": "technical_pass"}],
                },
            )
            return {"rc": 0, "stdout_tail": "", "stderr_tail": ""}
        if label == "window-method-validation-probe":
            assert calls.index("native_validation") < calls.index(label)
            assert "--repeats" in cmd and cmd[cmd.index("--repeats") + 1] == "3"
            assert "--include-raw-parquet" in cmd
            assert "--strict" in cmd
            out = Path(cmd[cmd.index("--out-dir") + 1])
            _write_json(out / "window_method_validation_probe.json", {
                "ok": False,
                "complete": False,
                "status": "blocked_zero_measurements_started",
                "blocked_reason": "zero_measurements_started",
                "decision_capable": False,
                "started_repeat_count": 0,
                "successful_comparison_count": 0,
            })
            return {"rc": 4, "stdout_tail": "", "stderr_tail": "probe incomplete"}
        return {"rc": 0, "stdout_tail": "", "stderr_tail": ""}

    monkeypatch.setattr(coordinator, "_run", fake_run)
    monkeypatch.setattr(
        coordinator,
        "_select_report_python",
        lambda *_args, **_kwargs: (sys.executable, {"selected": sys.executable, "onnxruntime_ok": True}),
    )
    monkeypatch.setattr(sys, "argv", [
        "run_evalrun_native_producer_variants.py",
        "--eval-run-dir", str(run_dir),
        "--config", str(config_path),
    ])

    assert coordinator.main() == 0
    assert calls.count("window-method-validation-probe") == 1
    stage = json.loads((reports / "native_producer_stage.json").read_text(encoding="utf-8"))
    probe = stage["window_method_validation_probe"]
    assert probe["status"] == "blocked_zero_measurements_started"
    assert probe["strict_validation_failure"] is True
    assert probe["workflow_blocking_requested"] is False
    assert probe["strict_failure"] is False
    assert probe["screening_only"] is True
    assert probe["eligible_for_energy_results_import"] is False
    assert stage["status"] == "ok"


def test_workflow_variant_importer_normalizes_smoke_probe_to_nonblocking(
    tmp_path: Path, monkeypatch,
) -> None:
    runner = EvaluationWorkflowRunner(WorkflowOptions(profile="", out=str(tmp_path)))
    runner.run_id = "variant_probe_run"
    runner.run_dir = tmp_path / runner.run_id
    runner.profile_payload = {
        "campaign": {"mode": "development"},
        "execution_preset": {"id": "smoke"},
        "measurement_campaign": {"system_power": {"scope": "system", "window": "command"}},
    }
    reports = runner.run_dir / "reports"
    cfg = {
        "enabled": True,
        "variants": [{"id": "resnet"}],
        "energy": {
            "enabled": False,
            "window_method_validation_probe": {
                "enabled": True, "repeats": 3,
                "include_raw_parquet": True, "strict": True,
            },
        },
    }

    def fake_streaming(*_args, **_kwargs):
        _assert_journal_environment(runner, _kwargs)
        probe = reports / "window_method_validation_probe"
        raw = probe / "measurement" / "run_000" / "collector_storage" / "trace.parquet"
        raw.parent.mkdir(parents=True, exist_ok=True)
        raw.write_bytes(b"PAR1-probe")
        _write_json(probe / "window_method_validation_probe.json", {
            "complete": False, "status": "blocked_zero_measurements_started",
        })
        _write_json(reports / "native_producer_summary.json", {"rows": [{
            "backend": "hailo8_to_trt", "model": "resnet50", "case": "b052",
            "precision": "fp16", "ok": True, "fps_makespan": 100.0,
        }]})
        _write_json(reports / "native_producer_stage.json", {
            "schema": "onnx-splitpoint/native-producer-variant-stage",
            "status": "failed",
            "window_method_validation_probe": {
                "enabled": True, "status": "blocked_zero_measurements_started",
                "strict_requested": True, "strict_failure": True,
                "include_raw_parquet": True,
            },
            "native_energy": {"enabled": False, "status": "skipped", "strict_failure": False},
        })
        _terminal_child_handoff(runner, _args, rc=2)
        return SimpleNamespace(returncode=2, stdout="", stderr="strict probe failure")

    monkeypatch.setattr("onnx_splitpoint_tool.workflow.runner.run_streaming", fake_streaming)
    _private_workflow_context(runner)
    with mock.patch.object(runner, "_native_producer_config", return_value=cfg):
        paths, details, _message, status = runner._stage_run_native_producers()

    # The child process itself returned a technical failure here, so the
    # imported stage remains partial even though the probe is non-blocking.
    assert status == "partial"
    assert details["strict_failure"] is False
    assert details["window_method_validation_probe"]["strict_validation_failure"] is True
    assert details["window_method_validation_probe"]["workflow_blocking_requested"] is False
    assert details["window_method_validation_probe"]["strict_failure"] is False
    assert any(Path(path).name == "trace.parquet" for path in paths.values())
    assert json.loads((reports / "native_producer_stage.json").read_text())["status"] == "failed"
    stage = json.loads((reports / "native_producer_parent_import.json").read_text(encoding="utf-8"))
    assert stage["status"] == "partial"
    assert stage["window_method_validation_probe_strict_failure"] is False


def test_workflow_variant_importer_keeps_incomplete_final_probe_nonblocking(
    tmp_path: Path, monkeypatch,
) -> None:
    runner = EvaluationWorkflowRunner(WorkflowOptions(profile="", out=str(tmp_path)))
    runner.run_id = "variant_final_incomplete_probe"
    runner.run_dir = tmp_path / runner.run_id
    runner.profile_payload = {
        "campaign": {"mode": "final"},
        "execution_preset": {"id": "final"},
        "measurement_campaign": {
            "system_power": {"scope": "system", "window": "command"},
        },
    }
    reports = runner.run_dir / "reports"
    cfg = {
        "enabled": True,
        "variants": [{"id": "resnet"}],
        "energy": {
            "enabled": False,
            "window_method_validation_probe": {
                "enabled": True,
                "repeats": 3,
                "include_raw_parquet": True,
                "strict": True,
            },
        },
    }

    def fake_streaming(*_args, **_kwargs):
        _assert_journal_environment(runner, _kwargs)
        _write_json(reports / "native_producer_summary.json", {"rows": [{
            "backend": "hailo8_to_trt",
            "model": "resnet50",
            "case": "b052",
            "precision": "fp16",
            "ok": True,
            "fps_makespan": 100.0,
        }]})
        # Simulate a partial/older child that records incompleteness but omits
        # the derived v2.66 failure fields.
        _write_json(reports / "native_producer_stage.json", {
            "schema": "onnx-splitpoint/native-producer-variant-stage",
            "status": "ok",
            "window_method_validation_probe": {
                "enabled": True,
                "status": "incomplete_probe_measurement_or_comparison_failed",
                "strict_requested": True,
                "complete": False,
            },
            "native_energy": {
                "enabled": False,
                "status": "skipped",
                "strict_failure": False,
            },
        })
        _terminal_child_handoff(runner, _args, rc=0)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(
        "onnx_splitpoint_tool.workflow.runner.run_streaming",
        fake_streaming,
    )
    _private_workflow_context(runner)
    with mock.patch.object(runner, "_native_producer_config", return_value=cfg):
        _paths, details, _message, status = runner._stage_run_native_producers()

    # v2.67 freezes the marker window as the scientific primary.  This child
    # completed normally; the incomplete Chapter-4 diagnostic remains visible
    # without changing the otherwise successful Native stage status.
    assert status == "ok"
    probe = details["window_method_validation_probe"]
    assert probe["strict_validation_failure"] is True
    assert probe["workflow_blocking_requested"] is False
    assert probe["workflow_blocking_scope"] == "none"
    assert probe["strict_failure"] is False
    assert details["strict_failure"] is False


def test_workflow_variant_coordinator_crash_without_state_fails_closed(
    tmp_path: Path, monkeypatch,
) -> None:
    runner = EvaluationWorkflowRunner(WorkflowOptions(profile="", out=str(tmp_path)))
    runner.run_id = "variant_coordinator_crash"
    runner.run_dir = tmp_path / runner.run_id
    runner.profile_payload = {
        "campaign": {"mode": "final"},
        "measurement_campaign": {"system_power": {"scope": "system", "window": "command"}},
    }
    cfg = {
        "enabled": True,
        "variants": [{"id": "resnet"}],
        "energy": {
            "enabled": True,
            "mode": "measure",
            "window_method_validation_probe": {
                "enabled": True,
                "repeats": 3,
                "include_raw_parquet": True,
                "strict": True,
            },
        },
    }

    calls = []
    def crash_child(*args, **kwargs):
        _assert_journal_environment(runner, kwargs)
        calls.append(kwargs["label"])
        return SimpleNamespace(returncode=9, stdout="", stderr="coordinator crashed before stage state")
    monkeypatch.setattr("onnx_splitpoint_tool.workflow.runner.run_streaming", crash_child)
    _private_workflow_context(runner)
    with mock.patch.object(runner, "_native_producer_config", return_value=cfg):
        _paths, details, _message, status = runner._stage_run_native_producers()

    # Current durable-stage contract keeps an uncommitted crash incomplete.
    # No terminal success/failure state may be invented by the parent.
    assert status == "cancelled"
    assert details["child_complete"] is False
    assert details["coordinator_handoff_complete"] is False
    assert details["variant_coordinator_rc"] == 9
    assert runner._stop_requested is True
    assert len(calls) == 1
    assert not (runner.run_dir / "reports/native_producer_stage.json").exists()
    imported = json.loads((runner.run_dir / "reports/native_producer_parent_import.json").read_text())
    assert imported["variant_coordinator_rc"] == 9
    assert imported.get("complete") is not True


@pytest.mark.parametrize("resume,checkpoint", [(False, False), (True, False), (True, True), (False, True)])
def test_standard_final_native_energy_exception_fails_closed(
    tmp_path: Path, monkeypatch, resume: bool, checkpoint: bool,
) -> None:
    runner = EvaluationWorkflowRunner(WorkflowOptions(profile="", out=str(tmp_path)))
    runner.run_id = "eval-native-split-001"
    runner.run_dir = tmp_path / runner.run_id
    runner.options.resume = resume
    if checkpoint:
        _write_json(runner.run_dir / "reports/native_energy_measurements/stages/native_energy/stage_result.json", {"state": "cancelled", "complete": False})
    runner.profile_payload = {
        "campaign": {"mode": "final"},
        "measurement_campaign": {"system_power": {"scope": "system", "window": "command"}},
    }
    registry = tmp_path / "hardware_setups.yaml"
    _write_json(registry, {"schema": "onnx-splitpoint/hardware-setups", "schema_version": 2,
        "hardware_setups": [{"id": "hailo8_setup", "accelerator": "hailo8", "enabled": True,
            "host": {"address": "fixture.invalid", "user": "fixture"}}]})
    runner.profile_payload["hardware"] = {"setups_file": str(registry), "selected_setups": ["hailo8_setup"]}
    runner.options.hardware_setups_file = str(registry)
    runner.manifest = {"models": {"yolo26s": {}}}
    suite = runner.run_dir / "models" / "yolo26s" / "benchmark_set"
    (suite / "b038").mkdir(parents=True)
    _write_json(suite / "benchmark_set.json", {"cases": [{"id": "b038"}]})
    _write_json(suite / "benchmark_plan.json", {"runs": [{"id": "split"}]})
    _write_json(suite / "b038/split_manifest.json", {"part2_external_inputs": ["boundary_tensor"]})
    (suite / "benchmark_suite.py").write_text("# test harness\n", encoding="utf-8")
    cfg = {
        "enabled": True,
        "models": ["yolo26s"],
        "backends": ["hailo8"],
        "remotes": {"hailo8": {"setup_id": "hailo8_setup", "ssh": "fixture@fixture.invalid"}},
        "copy_benchmarksets": False, "build_missing_engines": False,
        "energy": {"enabled": True, "mode": "measure"},
        "cleanup_remote_native_root": False,
    }

    from tests.test_v269f_variant_native_split_quality_first import _binding_and_summary, _write_summary
    _binding, summary = _binding_and_summary(tmp_path / "quality_fixture")
    _write_summary(runner.run_dir, summary)
    cfg["precision"] = "uint8_dequant_fp16"
    calls = []
    def child_leaf(cmd, **kwargs):
        _assert_journal_environment(runner, kwargs)
        label = kwargs["label"]; calls.append(label)
        if label == "energy:measure":
            assert ("--resume-checkpoint" in cmd) is (resume and checkpoint)
            assert "--resume-existing" not in cmd
            raise RuntimeError("r6 controlled energy child failure")
        if label == "final_report":
            _write_json(runner.run_dir / "reports/native_producer_combined_summary.json", {"rows": [{
                "backend": "hailo8_to_trt", "model": "yolo26s", "case": "b038", "precision": "uint8_dequant_fp16",
                "setup_id": "hailo8_setup", "ok": True, "fps_makespan": 100.0}]})
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr("onnx_splitpoint_tool.workflow.runner.run_streaming", child_leaf)
    monkeypatch.setattr("onnx_splitpoint_tool.workflow.runner._sync_remote_script_v60i", lambda *a, **k: [])
    monkeypatch.setattr("onnx_splitpoint_tool.workflow.runner._sync_remote_package_asset_v263", lambda *a, **k: [{"name":"fixture", "rc":0, "expected_sha256": __import__("hashlib").sha256((ROOT / k["relative_path"]).read_bytes()).hexdigest()}])
    monkeypatch.setattr("onnx_splitpoint_tool.workflow.runner._verify_remote_module_binding_v263", lambda *a, **k: {"rc":0})
    _private_workflow_context(runner)
    with mock.patch.object(runner, "_native_producer_config", return_value=cfg):
        _paths, details, _message, status = runner._stage_run_native_producers()

    assert status == "partial"
    assert details["strict_failure"] is True
    stage = json.loads(
        (runner.run_dir / "reports" / "native_producer_stage.json").read_text(encoding="utf-8")
    )
    assert stage["native_energy"]["status"] == "failed"
    assert stage["native_energy"]["strict_requested"] is True
    assert stage["native_energy"]["strict_failure"] is True
    assert stage["native_energy"]["final_energy_contract_enforced"] is True

    assert calls.count("energy:measure") == 1, calls
    assert "r6 controlled energy child failure" in stage["native_energy"]["error"]

def test_packaged_variant_helpers_resolve_siblings_without_source_scripts(tmp_path: Path) -> None:
    resource = ROOT / "onnx_splitpoint_tool" / "resources" / "remote_scripts"
    coordinator = _load_script(resource / "run_evalrun_native_producer_variants.py", "packaged_variant")
    updater = _load_script(resource / "update_evalset_native_producers.py", "packaged_updater")
    coordinator.ROOT = tmp_path / "installed_package_without_checkout_scripts"
    updater.ROOT = tmp_path / "installed_package_without_checkout_scripts"

    assert coordinator._script("update_evalset_native_producers.py") == resource / "update_evalset_native_producers.py"
    assert coordinator._script("run_window_method_validation_probe.py") == resource / "run_window_method_validation_probe.py"
    assert updater._script("run_benchmark_suite_from_set.py") == resource / "run_benchmark_suite_from_set.py"
    assert updater._script("native_producer_final_report.py") == resource / "native_producer_final_report.py"


def test_update_helper_propagates_failed_native_stage_to_exit_code(
    tmp_path: Path, monkeypatch,
) -> None:
    updater = _load_script(
        ROOT / "scripts" / "update_evalset_native_producers.py",
        "v262_update_status_test",
    )
    run_dir = tmp_path / "EvaluationRun"
    run_dir.mkdir()
    monkeypatch.setattr(
        updater, "_run_native_producers",
        lambda *_args, **_kwargs: {"status": "failed", "ok_backends": 0},
    )
    monkeypatch.setattr(sys, "argv", [
        "update_evalset_native_producers.py",
        "--eval-run-dir", str(run_dir),
        "--run-native-producers",
    ])

    assert updater.main() == 2
    summary = json.loads(
        (run_dir / "reports" / "update_evalset_native_producers.json").read_text(encoding="utf-8")
    )
    assert summary["ok"] is False
    assert summary["status"] == "failed"


def test_update_helper_syncs_self_contained_yolo_reference_probe() -> None:
    update_source = (ROOT / "scripts" / "update_evalset_native_producers.py").read_text(encoding="utf-8")
    probe_source = (ROOT / "scripts" / "native_yolo_full_self_reference_probe.py").read_text(encoding="utf-8")
    assert '"native_yolo_full_self_reference_probe.py"' in update_source
    assert '("--benchmark-set", "_input_dump_feed_from_manifest", "native_fifo_boundary")' in update_source
    assert "from split_chain_reference_contract_probe import" not in probe_source
    assert "from native_boundary_activation_compare import" not in probe_source
    assert "def _input_dump_feed_from_manifest(" in probe_source


@pytest.mark.parametrize('mode', ['missing', 'drift'])
def test_private_quality_context_rejects_absence_or_drift_before_child(tmp_path, monkeypatch, mode):
    coordinator = _load_script(ROOT / 'scripts/run_evalrun_native_producer_variants.py', 'r6_quality_negative_' + mode)
    cfg = {'validation': {'enabled': True}, 'variants': [{'id': 'one'}]}
    run = tmp_path / 'run'
    if mode == 'drift':
        policy = {'schema': 'onnx-splitpoint/task-quality-policy', 'schema_version': 3,
                  'name': 'private-profile', 'profile_id': 'private-profile'}
        _write_json(run / 'profile.yaml', {'quality_gate': policy})
        cfg['quality_gate_policy'] = {**policy, 'name': 'different', 'profile_id': 'different'}
    config = tmp_path / 'config.json'; _write_json(config, cfg)
    calls = []
    monkeypatch.setattr(coordinator, '_run', lambda *a, **k: calls.append(a))
    monkeypatch.setattr(sys, 'argv', ['coordinator', '--eval-run-dir', str(run), '--config', str(config)])
    assert coordinator.main() == 2
    assert calls == []
    assert not (run / 'stages/run_native_producers/native_coordinator/stage_result.json').exists()


def test_private_workflow_requires_real_journal_before_child(tmp_path, monkeypatch):
    runner = EvaluationWorkflowRunner(WorkflowOptions(profile='', out=str(tmp_path)))
    runner.run_id = 'missing_journal'; runner.run_dir = tmp_path / runner.run_id
    runner.profile_payload = {'campaign': {'mode': 'development'}}
    calls = []
    monkeypatch.setattr('onnx_splitpoint_tool.workflow.runner.run_streaming', lambda *a, **k: calls.append(a))
    with mock.patch.object(runner, '_native_producer_config', return_value={'enabled': True, 'variants': [{'id': 'one'}]}):
        with pytest.raises(Exception, match='journal'):
            runner._stage_run_native_producers()
    assert calls == []
