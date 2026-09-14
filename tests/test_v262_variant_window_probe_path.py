from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

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


def test_variant_coordinator_keeps_strict_smoke_probe_nonblocking_with_green_baseline(
    tmp_path: Path, monkeypatch,
) -> None:
    coordinator = _load_script(
        ROOT / "scripts" / "run_evalrun_native_producer_variants.py",
        "v262_variant_coordinator_test",
    )
    run_dir = tmp_path / "EvaluationRun"
    reports = run_dir / "reports"
    (run_dir / "native_producers" / "hailo8").mkdir(parents=True)
    _write_legacy_run_manifest(run_dir)
    config = {
        "_workflow_context": {
            "campaign": {"mode": "development"},
            "execution_preset": {"id": "smoke"},
        },
        "variants": [{"id": "resnet", "case_map": {"resnet50": ["b052"]}}],
        "validation": {"enabled": True},
        "remotes": {"hailo8": {"ssh": "nx@host", "env": "source env"}},
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
    config_path = tmp_path / "native.json"
    _write_json(config_path, config)
    calls: list[str] = []

    def fake_run(cmd, *, timeout=None, cwd=None, label="native-child"):
        calls.append(label)
        if label == "final_report":
            _write_json(reports / "native_producer_combined_summary.json", {
                "rows": [{
                    "backend": "hailo8_to_trt", "model": "resnet50", "case": "b052",
                    "precision": "fp16", "setup_id": "h8", "ok": True,
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
        return SimpleNamespace(returncode=2, stdout="", stderr="strict probe failure")

    monkeypatch.setattr("onnx_splitpoint_tool.workflow.runner.run_streaming", fake_streaming)
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
    stage = json.loads((reports / "native_producer_stage.json").read_text(encoding="utf-8"))
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
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(
        "onnx_splitpoint_tool.workflow.runner.run_streaming",
        fake_streaming,
    )
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

    monkeypatch.setattr(
        "onnx_splitpoint_tool.workflow.runner.run_streaming",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=9,
            stdout="",
            stderr="coordinator crashed before stage state",
        ),
    )
    with mock.patch.object(runner, "_native_producer_config", return_value=cfg):
        _paths, details, _message, status = runner._stage_run_native_producers()

    assert status == "failed"
    assert details["strict_failure"] is True
    assert details["window_method_validation_probe"]["strict_failure"] is False
    assert details["native_energy"]["strict_failure"] is True
    stage = json.loads(
        (runner.run_dir / "reports" / "native_producer_stage.json").read_text(encoding="utf-8")
    )
    assert stage["status"] == "failed"
    assert stage["orchestration_status"] == "failed"
    assert stage["coordinator_failed_before_strict_component_state"] is True


def test_standard_final_native_energy_exception_fails_closed(
    tmp_path: Path, monkeypatch,
) -> None:
    runner = EvaluationWorkflowRunner(WorkflowOptions(profile="", out=str(tmp_path)))
    runner.run_id = "standard_energy_exception"
    runner.run_dir = tmp_path / runner.run_id
    runner.profile_payload = {
        "campaign": {"mode": "final"},
        "measurement_campaign": {"system_power": {"scope": "system", "window": "command"}},
    }
    runner.manifest = {"models": {"resnet50": {}}}
    suite = runner.run_dir / "models" / "resnet50" / "benchmark_set"
    (suite / "b001").mkdir(parents=True)
    _write_json(suite / "benchmark_set.json", {"cases": [{"id": "b001"}]})
    _write_json(suite / "benchmark_plan.json", {"runs": [{"id": "split"}]})
    (suite / "benchmark_suite.py").write_text("# test harness\n", encoding="utf-8")
    cfg = {
        "enabled": True,
        "models": ["resnet50"],
        "backends": ["hailo8"],
        "remotes": {"hailo8": {}},
        "energy": {"enabled": True, "mode": "measure"},
        "cleanup_remote_native_root": False,
    }

    with mock.patch.object(runner, "_native_producer_config", return_value=cfg):
        _paths, details, _message, status = runner._stage_run_native_producers()

    assert status == "failed"
    assert details["strict_failure"] is True
    stage = json.loads(
        (runner.run_dir / "reports" / "native_producer_stage.json").read_text(encoding="utf-8")
    )
    assert stage["native_energy"]["status"] == "failed"
    assert stage["native_energy"]["strict_requested"] is True
    assert stage["native_energy"]["strict_failure"] is True
    assert stage["native_energy"]["final_energy_contract_enforced"] is True


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
