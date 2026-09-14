from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Mapping
from unittest import mock

import pytest

from onnx_splitpoint_tool.workflow.runner import (
    EvaluationWorkflowRunner,
    WorkflowOptions,
)
from onnx_splitpoint_tool.remote.process_lease import RemoteProcessLeaseScope


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload)), encoding="utf-8")


def _variant_runner(tmp_path: Path, *, run_mode: str) -> EvaluationWorkflowRunner:
    runner = EvaluationWorkflowRunner(WorkflowOptions(profile="", out=str(tmp_path)))
    runner.run_id = f"variant_{run_mode}"
    runner.run_dir = tmp_path / runner.run_id
    runner._remote_process_registry.configure_journal(
        scope=RemoteProcessLeaseScope(runner.run_id, runner.session_id),
        journal_dir=tmp_path / f"{runner.run_id}_remote_lease_journal",
    )
    runner.profile_payload = {
        "campaign": {"mode": "development" if run_mode == "smoke" else "measurement"},
        "execution_preset": {"id": run_mode},
    }
    return runner


def test_variant_parent_preserves_standard_child_upstream_quality_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _variant_runner(tmp_path, run_mode="standard")
    reports = runner.run_dir / "reports"
    cfg = {
        "enabled": True,
        "variants": [{"id": "resnet"}],
        "energy": {"enabled": False},
    }

    def fake_streaming(*_args: Any, **_kwargs: Any) -> SimpleNamespace:
        _write_json(reports / "native_producer_summary.json", {
            "rows": [{
                "backend": "hailo8_to_trt",
                "model": "resnet50",
                "case": "b052",
                "precision": "fp16",
                "ok": True,
            }],
        })
        _write_json(reports / "native_producer_stage.json", {
            "schema": "onnx-splitpoint/native-producer-variant-stage",
            "status": "failed",
            "failure_class": "upstream_quality_evidence",
            "failure_reason": "upstream_central_quality_binding_missing",
            "summary": {"native_validation_technical_error_count": 1},
            "native_energy": {
                "enabled": False,
                "status": "skipped",
                "strict_failure": False,
            },
        })
        return SimpleNamespace(
            returncode=2,
            stdout="",
            stderr="upstream central quality binding missing",
        )

    monkeypatch.setattr(
        "onnx_splitpoint_tool.workflow.runner.run_streaming", fake_streaming,
    )
    with mock.patch.object(runner, "_native_producer_config", return_value=cfg):
        _paths, details, _message, status = runner._stage_run_native_producers()

    assert status == "failed"
    assert details["terminal_child_failure"] is True
    assert details["orchestration_status"] == "failed"
    stage = json.loads(
        (reports / "native_producer_stage.json").read_text(encoding="utf-8")
    )
    assert stage["status"] == "failed"
    assert stage["failure_class"] == "upstream_quality_evidence"
    assert stage["failure_reason"] == "upstream_central_quality_binding_missing"


def test_variant_parent_only_normalizes_legacy_smoke_probe_failure_to_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _variant_runner(tmp_path, run_mode="smoke")
    reports = runner.run_dir / "reports"
    cfg = {
        "enabled": True,
        "variants": [{"id": "resnet"}],
        "energy": {
            "enabled": False,
            "window_method_validation_probe": {
                "enabled": True,
                "strict": True,
            },
        },
    }

    def fake_streaming(*_args: Any, **_kwargs: Any) -> SimpleNamespace:
        _write_json(reports / "native_producer_summary.json", {
            "rows": [{
                "backend": "hailo8_to_trt",
                "model": "resnet50",
                "case": "b052",
                "precision": "fp16",
                "ok": True,
            }],
        })
        # Historical children used the stage-level failure for the strict
        # screening probe although that probe was never workflow-blocking.
        _write_json(reports / "native_producer_stage.json", {
            "schema": "onnx-splitpoint/native-producer-variant-stage",
            "status": "failed",
            "window_method_validation_probe": {
                "enabled": True,
                "status": "blocked_zero_measurements_started",
                "strict_requested": True,
                "strict_failure": True,
                "complete": False,
            },
            "native_energy": {
                "enabled": False,
                "status": "skipped",
                "strict_failure": False,
            },
        })
        return SimpleNamespace(
            returncode=2,
            stdout="",
            stderr="legacy strict probe failure",
        )

    monkeypatch.setattr(
        "onnx_splitpoint_tool.workflow.runner.run_streaming", fake_streaming,
    )
    with mock.patch.object(runner, "_native_producer_config", return_value=cfg):
        _paths, details, _message, status = runner._stage_run_native_producers()

    assert status == "partial"
    assert details["terminal_child_failure"] is False
    assert details["strict_failure"] is False
    probe = details["window_method_validation_probe"]
    assert probe["strict_validation_failure"] is True
    assert probe["workflow_blocking_requested"] is False
    assert probe["strict_failure"] is False


def _direct_runner(
    tmp_path: Path,
    *,
    run_mode: str,
) -> tuple[EvaluationWorkflowRunner, Path]:
    runner = EvaluationWorkflowRunner(WorkflowOptions(profile="", out=str(tmp_path)))
    runner.run_id = f"direct_{run_mode}"
    runner.run_dir = tmp_path / runner.run_id
    runner._remote_process_registry.configure_journal(
        scope=RemoteProcessLeaseScope(runner.run_id, runner.session_id),
        journal_dir=tmp_path / f"{runner.run_id}_remote_lease_journal",
    )
    runner.profile_payload = {
        "campaign": {
            "mode": "development" if run_mode == "smoke" else "measurement",
        },
        "execution_preset": {"id": run_mode},
        "measurement_campaign": {
            "system_power": {"scope": "system", "window": "command"},
        },
    }
    runner.profile_start_snapshot = {}
    runner.manifest = {"models": {"resnet50": {}}}

    suite = runner.run_dir / "models" / "resnet50" / "benchmark_set"
    (suite / "b001").mkdir(parents=True)
    _write_json(
        suite / "b001" / "split_manifest.json",
        {"part2_external_inputs": ["boundary_tensor"]},
    )
    _write_json(suite / "benchmark_set.json", {"cases": [{"id": "b001"}]})
    _write_json(suite / "benchmark_plan.json", {"runs": [{"id": "split"}]})
    (suite / "benchmark_suite.py").write_text("# workflow boundary fixture\n", encoding="utf-8")

    reports = runner.run_dir / "reports"
    _write_json(reports / "native_producer_summary.json", {
        "row_count": 1,
        "rows": [{
            "backend": "hailo8_to_trt",
            "model": "resnet50",
            "case": "b001",
            "precision": "fp16",
            "execution_mode": "native_split",
            "ok": True,
            "fps_makespan": 100.0,
        }],
    })
    return runner, reports


def test_direct_standard_partial_quality_preflight_dispatches_diagnostic_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, reports = _direct_runner(tmp_path, run_mode="standard")
    _write_json(
        runner.run_dir / "quality_management" / "central_quality_summary.json",
        {"results": []},
    )
    cfg = {
        "enabled": True,
        "models": ["resnet50"],
        "backends": ["hailo8", "deepx"],
        "precision": "fp16",
        "case_policy": "case_map_only",
        "case_map": {"resnet50": ["b001"]},
        "remotes": {
            "hailo8": {
                "ssh": "nx@hailo8",
                "setup_id": "hailo8_setup",
            },
            "deepx": {
                "ssh": "nx@deepx",
                "setup_id": "deepx_setup",
            },
        },
        "copy_benchmarksets": False,
        "build_missing_engines": False,
        "validation": {"enabled": False},
        "energy": {"enabled": False},
        "full_baselines": {
            "enabled": True,
            "backends_by_producer": {
                "hailo8": ["hailo8", "tensorrt"],
                "deepx": ["deepx", "tensorrt"],
            },
        },
    }
    stream_calls: list[list[str]] = []
    stream_labels: list[str] = []

    def successful_streaming(
        command: list[str], *_args: Any, **_kwargs: Any,
    ) -> SimpleNamespace:
        stream_calls.append([str(item) for item in command])
        stream_labels.append(str(_kwargs.get("label") or ""))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    def producer_set(
        _summary: Path, *, setup_id: str, model_ids: list[str], **_kwargs: Any,
    ) -> dict[str, Any]:
        if setup_id == "deepx_setup":
            raise RuntimeError("intentional missing DeepX TensorRT binding")
        return {
            "schema": "test/tensorrt-quality-producer-set",
            "setup_id": setup_id,
            "producers_by_model": {
                model_id: {"model_id": model_id, "fixture": True}
                for model_id in model_ids
            },
        }

    def split_set(
        _summary: Path, *, setup_id: str,
        selections: list[dict[str, Any]], **_kwargs: Any,
    ) -> dict[str, Any]:
        if setup_id == "deepx_setup":
            raise RuntimeError("intentional missing DeepX split binding")
        return {
            "schema": "test/native-split-quality-binding-set",
            "setup_id": setup_id,
            "bindings_by_model_case_backend": {
                "|".join((
                    str(selection["model_id"]),
                    str(selection["case_id"]).lower(),
                    str(selection["backend"]).lower().replace("-", "_"),
                )): {"fixture": True}
                for selection in selections
            },
            "binding_set_sha256": "fixture",
        }

    monkeypatch.setattr(
        "onnx_splitpoint_tool.workflow.runner.run_streaming",
        successful_streaming,
    )
    monkeypatch.setattr(
        "onnx_splitpoint_tool.workflow.runner.normalize_hardware_targets",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        "onnx_splitpoint_tool.workflow.runner.benchmark_set_postcondition_v60v",
        lambda path: {"valid": True, "selected_suite_dir": str(path)},
    )
    monkeypatch.setattr(
        "onnx_splitpoint_tool.workflow.runner._sync_remote_script_v60i",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        "onnx_splitpoint_tool.workflow.runner._sync_remote_package_asset_v263",
        lambda *_args, **_kwargs: [{
            "name": "sync_fixture",
            "rc": 0,
            "expected_sha256": "a" * 64,
        }],
    )
    monkeypatch.setattr(
        "onnx_splitpoint_tool.workflow.runner._verify_remote_module_binding_v263",
        lambda *_args, **_kwargs: {
            "name": "verify_fixture",
            "rc": 0,
        },
    )
    monkeypatch.setattr(
        runner,
        "_finish_native_direct_remote_lease",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "onnx_splitpoint_tool.trt_quality_chain.load_producer_set_from_central_quality_summary",
        producer_set,
    )
    monkeypatch.setattr(
        "onnx_splitpoint_tool.trt_quality_chain.load_split_binding_set_from_central_quality_summary",
        split_set,
    )

    with mock.patch.object(runner, "_native_producer_config", return_value=cfg):
        _paths, details, _message, status = runner._stage_run_native_producers()

    stage = json.loads(
        (reports / "native_producer_stage.json").read_text(encoding="utf-8")
    )
    expected_matrix = json.loads(
        (reports / "native_expected_matrix.json").read_text(encoding="utf-8")
    )
    assert stream_calls
    assert stage["central_binding_preflight"]["runtime_blocking"] is False
    assert any(
        row.get("transfer_attempted") is True
        for row in stage["backend_results"]
    )
    assert stage["started_remote_count"] > 0
    assert stage["started_performance_count"] > 0
    assert details["started_remote_count"] == stage["started_remote_count"]
    assert (
        details["started_performance_count"]
        == stage["started_performance_count"]
    )
    assert {
        "split:hailo8:generic",
        "full:hailo8:hailo8,tensorrt",
        "split:deepx:generic",
        "full:deepx:deepx",
    }.issubset(set(stream_labels))
    assert status in {"partial", "failed"}
    assert stage["status"] == status
    # Two selected Native-Split rows remain in the configured denominator
    # even when the preflight blocks their adapters, in addition to four Full
    # baselines. The denominator must never shrink to the supported subset.
    assert expected_matrix["expected_row_count"] == 6
    assert expected_matrix["present_expected_row_count"] == 0
    assert expected_matrix["missing_expected_row_count"] == 6
    assert len(expected_matrix["missing_expected_rows"]) == 6
    assert stage["expected_matrix"]["expected_row_count"] == 6


def _run_direct_workflow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    run_mode: str,
    validation_payload: Mapping[str, Any],
    energy_enabled: bool = False,
) -> tuple[dict[str, Any], dict[str, list[str]], str]:
    runner, reports = _direct_runner(tmp_path, run_mode=run_mode)
    commands: dict[str, list[str]] = {}
    cfg = {
        "enabled": True,
        "models": ["resnet50"],
        "backends": ["hailo8"],
        "precision": "fp16",
        "case_policy": "case_map_only",
        "case_map": {"resnet50": ["b001"]},
        "remotes": {"hailo8": {"setup_id": "hailo8_setup"}},
        "validation": {"enabled": True},
        "energy": {
            "enabled": energy_enabled,
            "mode": "plan",
            "strict": True,
            "duration_s": 1.0,
            "timeout": 1,
        },
        "cleanup_remote_native_root": False,
    }

    def fake_streaming(
        command: list[str], *_args: Any, **kwargs: Any,
    ) -> SimpleNamespace:
        label = str(kwargs.get("label") or "")
        commands[label] = [str(item) for item in command]
        if label == "validation":
            _write_json(
                reports / "native_validation"
                / "native_producer_validation_summary.json",
                validation_payload,
            )
        elif label == "energy:plan":
            _write_json(
                reports / "native_energy_plan"
                / "native_producer_energy_plan.json",
                {
                    "status": "ok",
                    "rows": [],
                    "paired_missing_rows": [],
                    "diagnostic_only": True,
                    "claim_eligible": False,
                },
            )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(
        "onnx_splitpoint_tool.workflow.runner.run_streaming", fake_streaming,
    )
    monkeypatch.setattr(
        "onnx_splitpoint_tool.workflow.runner.normalize_hardware_targets",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        "onnx_splitpoint_tool.workflow.runner.benchmark_set_postcondition_v60v",
        lambda path: {"valid": True, "selected_suite_dir": str(path)},
    )
    monkeypatch.setattr(
        "onnx_splitpoint_tool.workflow.runner._select_native_report_python",
        lambda *_args, **_kwargs: (
            sys.executable,
            {"selected": sys.executable, "onnxruntime_ok": True},
        ),
    )
    monkeypatch.setattr(
        "onnx_splitpoint_tool.trt_quality_chain.load_split_binding_set_from_central_quality_summary",
        lambda *_args, **_kwargs: {
            "schema": "onnx-splitpoint/native-split-quality-binding-set",
            "bindings": [],
        },
    )
    with mock.patch.object(runner, "_native_producer_config", return_value=cfg):
        _paths, _details, _message, status = runner._stage_run_native_producers()

    stage = json.loads(
        (reports / "native_producer_stage.json").read_text(encoding="utf-8")
    )
    return stage, commands, status


@pytest.mark.parametrize("quality_failure", ["technical_error", "zero_rows"])
@pytest.mark.parametrize(
    ("run_mode", "expected_status"),
    [("standard", "failed"), ("smoke", "partial")],
)
def test_direct_quality_failure_is_hard_only_in_standard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    quality_failure: str,
    run_mode: str,
    expected_status: str,
) -> None:
    if quality_failure == "technical_error":
        validation = {
            "status": "complete",
            "row_count": 1,
            "technical_error_count": 1,
            "rows": [{"ok": False, "status": "diagnostic_technical_error"}],
        }
    else:
        validation = {
            "status": "complete",
            "row_count": 0,
            "technical_error_count": 0,
            "rows": [],
        }

    stage, _commands, status = _run_direct_workflow(
        tmp_path,
        monkeypatch,
        run_mode=run_mode,
        validation_payload=validation,
    )

    assert status == expected_status
    assert stage["status"] == expected_status
    assert stage["technical_quality_failure"] is True
    assert stage["expected_matrix"]["missing_expected_row_count"] == 0
    assert not any(
        row.get("failure_class") == "upstream_quality_evidence"
        for row in stage["backend_results"]
    )
    if run_mode == "smoke":
        assert stage["diagnostic_only"] is True
        assert stage["claim_eligible"] is False


def test_direct_smoke_energy_empty_pair_annotation_reaches_planner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    validation = {
        "status": "complete",
        "row_count": 1,
        "technical_error_count": 0,
        "rows": [{
            "backend": "hailo8_to_trt",
            "model": "resnet50",
            "case": "b001",
            "precision": "fp16",
            "ok": True,
            "central_quality_evidence_verified": True,
        }],
    }

    stage, commands, status = _run_direct_workflow(
        tmp_path,
        monkeypatch,
        run_mode="smoke",
        validation_payload=validation,
        energy_enabled=True,
    )

    assert status == "partial"
    assert "energy:plan" in commands
    preflight = stage["native_energy_preflight"]
    assert preflight["plan_viable"] is True
    assert preflight["theoretical_pair_count"] == 0
    assert (
        "no_theoretical_setup_local_energy_pair"
        in preflight["nonblocking_annotations"]
    )
    assert preflight["started_remote_count"] == 0
    assert preflight["started_performance_count"] == 0
    energy = stage["native_energy"]
    assert energy["status"] == "blocked_no_runtime_constructible_rows"
    assert (
        energy["plan_preflight_status"]
        == "blocked_no_runtime_constructible_rows"
    )
    assert energy["strict_requested"] is False
    assert energy["strict_failure"] is False
    assert energy["diagnostic_only"] is True
    assert energy["claim_eligible"] is False
    assert energy["energy_claim_eligible"] is False
    assert stage["diagnostic_only"] is True
    assert stage["claim_eligible"] is False
