from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shlex
import sys
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(
        f"test_energy_failure_isolation_{path.stem}", path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _profile(*, explicit_generic: bool) -> dict:
    from onnx_splitpoint_tool.run_modes import default_run_modes_config

    mode = default_run_modes_config()["modes"]["standard"]
    profile = {
        "name": "energy_failure_isolation",
        "model_suite": {"primary": [{
            "id": "resnet50", "task": "classification", "enabled": True,
        }]},
        "run_profiles": [{
            "id": "hailo8_to_trt", "stage1": "hailo8",
            "stage2": "tensorrt", "enabled": True,
        }],
        "execution_preset": {
            "id": "standard",
            "follow_tool_config": False,
            "snapshot": mode,
            "overrides": {
                "native_enabled": True,
                "energy_enabled": True,
            },
        },
    }
    if explicit_generic:
        profile["energy"] = {
            "requested": True,
            "requested_native_energy": True,
            "enabled": True,
            "generic_enabled": True,
            "measurement_path": "native_and_generic",
        }
    return profile


def test_invalid_model_is_local_while_valid_models_continue() -> None:
    from onnx_splitpoint_tool.workflow.runner import (
        _native_model_selection_decision,
    )

    runnable, blocking = _native_model_selection_decision(
        ["resnet50", "yolo26s"],
        {"yolov7_ultralytics": {"valid": False}},
    )
    assert runnable == ["resnet50", "yolo26s"]
    assert blocking == []


def test_no_runnable_model_remains_a_global_preflight_stop() -> None:
    from onnx_splitpoint_tool.workflow.runner import (
        _native_model_selection_decision,
    )

    runnable, blocking = _native_model_selection_decision(
        [], {"yolov7_ultralytics": {"valid": False}},
    )
    assert runnable == []
    assert blocking == ["benchmark_set_invalid:yolov7_ultralytics"]


def test_runtime_quality_error_is_explicitly_raw_and_unqualified() -> None:
    planner = _load_script("native_producer_energy_plan.py")
    runner = _load_script("run_native_producer_energy_from_summary.py")
    admission = {
        "admission_scope": "native_runtime_observation",
        "central_quality_evidence_verified": False,
        "precision_quality_binding_verified": False,
        "task_quality_observation_valid": False,
        "accuracy_gate_pass": False,
        "quality_provenance_complete": False,
        "quality_claim_result_verified": False,
        "diagnostic_only": True,
        "claim_comparable": False,
        "energy_claim_eligible": False,
    }
    expected = {
        "energy_quality_qualified": False,
        "energy_quality_status": "raw_energy_quality_not_qualified",
        "native_energy_after_technical_error": (
            "collect_raw_quality_unqualified"
        ),
    }
    assert {key: planner._energy_quality_result_fields(admission)[key] for key in expected} == expected
    projection = runner._energy_quality_result_projection(
        {"energy_quality_admission": admission},
        measurement_started=True,
        raw_energy_collected=True,
    )
    assert {key: projection[key] for key in expected} == expected
    assert projection["local_task_quality_decision"] == "unavailable"
    assert projection["energy_quality_exclusion_reasons"]["binding_or_completion"]


def test_raw_quality_label_requires_actual_energy_evidence() -> None:
    runner = _load_script("run_native_producer_energy_from_summary.py")
    admission = {
        "admission_scope": "native_runtime_observation",
        "central_quality_evidence_verified": False,
        "energy_claim_eligible": False,
    }
    not_started = runner._energy_quality_result_projection(
        {"energy_quality_admission": admission},
        measurement_started=False,
        raw_energy_collected=False,
    )
    assert not_started["native_energy_after_technical_error"] == (
        "not_collected_measurement_not_started"
    )
    assert "collect_raw" not in str(not_started)

    started_without_raw = runner._energy_quality_result_projection(
        {"energy_quality_admission": admission},
        measurement_started=True,
        raw_energy_collected=False,
    )
    assert started_without_raw["native_energy_after_technical_error"] == (
        "collection_started_raw_energy_unavailable"
    )
    assert "collect_raw" not in str(started_without_raw)

    collected = runner._energy_quality_result_projection(
        {
            "energy_quality_admission": admission,
            # Untrusted display fields cannot turn the sealed admission into a
            # qualified result.
            "energy_quality_qualified": True,
            "energy_quality_status": "quality_qualified",
        },
        measurement_started=True,
        raw_energy_collected=True,
    )
    assert {key: collected[key] for key in ("energy_quality_qualified", "energy_quality_status", "native_energy_after_technical_error")} == {
        "energy_quality_qualified": False,
        "energy_quality_status": "raw_energy_quality_not_qualified",
        "native_energy_after_technical_error": (
            "collect_raw_quality_unqualified"
        ),
    }


def test_global_energy_classifier_is_narrow_and_structured(
    tmp_path: Path,
) -> None:
    runner = _load_script("run_native_producer_energy_from_summary.py")

    assert runner._energy_global_infrastructure_failure({
        "stderr_tail": "ssh: Permission denied (publickey).",
    })["code"] == "remote_authentication_failed"
    assert runner._energy_global_infrastructure_failure({
        "stderr_tail": "ssh: connect to host nx: Connection timed out",
    })["code"] == "remote_connection_failed"
    assert runner._energy_global_infrastructure_failure({
        "error": "platform_lock unavailable",
    })["code"] == "platform_lock_unavailable"
    assert runner._energy_global_infrastructure_failure({
        "error": "urecs-data-collector not found",
    })["code"] == "collector_initialization_failed"

    aggregate = tmp_path / "energy_aggregate.json"
    aggregate.write_text(json.dumps({
        "global_infrastructure_failure": "urecs_transport_unavailable",
        "global_infrastructure_category": "collector_initialization",
        "global_infrastructure_detail": "u.RECS connection refused",
    }), encoding="utf-8")
    structured = runner._energy_global_infrastructure_failure(
        {}, tmp_path,
    )
    assert structured == {
        "category": "collector_initialization",
        "code": "urecs_transport_unavailable",
        "detail": "u.RECS connection refused",
    }

    for local_error in (
        "benchmark_set_invalid:yolov7_ultralytics",
        "technical_quality_error: preprocessing_contract",
        "vendor runtime exited with rc=7",
        "collector_rc_nonzero",
        "postprocess_failed_or_energy_missing",
    ):
        assert runner._energy_global_infrastructure_failure({
            "error": local_error,
        }) == {}


def test_collector_process_start_truth_and_failure_classifier(
    tmp_path: Path, monkeypatch,
) -> None:
    from onnx_splitpoint_tool.energy import collector

    started: list[int] = []

    def _raise_popen(*_args, **_kwargs):
        raise FileNotFoundError("collector binary missing")

    monkeypatch.setattr(collector.subprocess, "Popen", _raise_popen)
    result = collector._run_one(
        ["missing-collector"],
        cwd=None,
        stdout_path=tmp_path / "stdout.log",
        stderr_path=tmp_path / "stderr.log",
        on_process_started=lambda proc, ns: started.append(ns),
    )
    assert result["process_started"] is False
    assert result["process_started_at_unix_ns"] == 0
    assert started == []
    failure = collector._collector_global_infrastructure_failure(
        result, tmp_path,
    )
    assert failure["code"] == "collector_process_start_failed"
    callback_failure = collector._collector_global_infrastructure_failure(
        {
            "process_started": True,
            "process_start_callback_failed": True,
            "exception": "OSError",
            "stderr": "OSError: disk full",
        },
        tmp_path,
    )
    assert callback_failure["code"] == "collector_start_evidence_failed"


def test_successful_process_reports_start_only_after_popen(
    tmp_path: Path,
) -> None:
    import sys
    from onnx_splitpoint_tool.energy import collector

    observed: list[tuple[int, int]] = []
    result = collector._run_one(
        [sys.executable, "-c", "pass"],
        cwd=None,
        stdout_path=tmp_path / "stdout.log",
        stderr_path=tmp_path / "stderr.log",
        on_process_started=lambda proc, ns: observed.append((proc.pid, ns)),
    )
    assert result["rc"] == 0
    assert result["process_started"] is True
    assert result["process_started_at_unix_ns"] > 0
    assert observed == [(
        observed[0][0], result["process_started_at_unix_ns"],
    )]


def test_idle_trace_around_failed_workload_is_not_model_raw_energy() -> None:
    runner = _load_script("run_native_producer_energy_from_summary.py")
    assert runner._raw_energy_collected({
        "energy_aggregate_verified": True,
        "energy_aggregate": {
            "runs": [{
                "workload_command_rc": 0,
            }],
        },
    }) is False
    failed_workload = {
        "energy_aggregate": {
            "runs": [{
                "collector_started": True,
                "workload_command_rc": 255,
                "parquet_file_sizes": {"samples.parquet": 4096},
                "energy_total_j": 12.0,
            }],
        },
    }
    assert runner._raw_energy_collected(failed_workload) is False
    failed_workload["energy_aggregate"]["runs"][0][
        "workload_command_rc"
    ] = 0
    assert runner._raw_energy_collected(failed_workload) is True


def test_hard_collector_failure_stops_remaining_repeats(
    tmp_path: Path, monkeypatch,
) -> None:
    from onnx_splitpoint_tool.energy import collector
    from onnx_splitpoint_tool.energy.config import EnergyDefaults, EnergySetup

    monkeypatch.setattr(
        collector, "check_energy_tools",
        lambda _defaults: {
            "collector_found": True,
            "power_calculations_found": False,
        },
    )
    calls: list[list[str]] = []

    def _failed_start(cmd, **_kwargs):
        calls.append(list(cmd))
        return {
            "cmd": list(cmd), "rc": -998, "stdout": "",
            "stderr": "FileNotFoundError: collector missing",
            "exception": "FileNotFoundError", "duration_s": 0.0,
            "process_started": False,
            "process_started_at_unix_ns": 0,
        }

    monkeypatch.setattr(collector, "_run_one", _failed_start)
    result = collector.run_fast_firmware_measurement(
        command="true",
        out_dir=tmp_path / "hard_failure",
        setup=EnergySetup(
            setup_id="fake", enabled=True,
            urecs_address="127.0.0.1",
        ),
        defaults=EnergyDefaults(
            collector_binary="missing-collector",
            postprocess_with_power_calculations=False,
            compare_legacy_window=False,
            pre_duration_s=0,
            post_duration_s=0,
        ),
        duration_s=1.0,
        run_count=3,
        exact_run_count=True,
        postprocess=False,
        compare_legacy_window=False,
    )
    assert len(calls) == 1
    assert len(result["runs"]) == 1
    assert result["runs"][0]["collector_started"] is False
    assert result["global_infrastructure_failure"] == (
        "collector_process_start_failed"
    )
    assert result["unattempted_repeat_count"] == 2


def test_ordinary_collector_child_failure_remains_repeat_local(
    tmp_path: Path, monkeypatch,
) -> None:
    from onnx_splitpoint_tool.energy import collector
    from onnx_splitpoint_tool.energy.config import EnergyDefaults, EnergySetup

    monkeypatch.setattr(
        collector, "check_energy_tools",
        lambda _defaults: {
            "collector_found": True,
            "power_calculations_found": False,
        },
    )
    calls: list[list[str]] = []

    def _row_failure(cmd, **kwargs):
        calls.append(list(cmd))
        callback = kwargs.get("on_process_started")
        if callable(callback):
            callback(SimpleNamespace(pid=1234), 123456)
        return {
            "cmd": list(cmd), "rc": 7, "stdout": "",
            "stderr": "vendor runtime exited with rc=7",
            "duration_s": 0.0, "process_started": True,
            "process_started_at_unix_ns": 123456,
        }

    monkeypatch.setattr(collector, "_run_one", _row_failure)
    result = collector.run_fast_firmware_measurement(
        command="false",
        out_dir=tmp_path / "row_failure",
        setup=EnergySetup(
            setup_id="fake", enabled=True,
            urecs_address="127.0.0.1",
        ),
        defaults=EnergyDefaults(
            collector_binary="fake-collector",
            postprocess_with_power_calculations=False,
            compare_legacy_window=False,
            pre_duration_s=0,
            post_duration_s=0,
        ),
        duration_s=1.0,
        run_count=3,
        exact_run_count=True,
        postprocess=False,
        compare_legacy_window=False,
    )
    assert len(calls) == 3
    assert len(result["runs"]) == 3
    assert result["global_infrastructure_failure"] == ""
    assert result["unattempted_repeat_count"] == 0


def test_post_run_quality_veto_keeps_physical_row_energy_eligible(
    tmp_path: Path,
) -> None:
    planner = _load_script("native_producer_energy_plan.py")
    final_report = _load_script("native_producer_final_report.py")
    full_runner = _load_script("native_full_baseline_eval_runner.py")

    assert planner._runtime_successful_for_energy({
        "ok": False,
        "runtime_success": True,
        "technical_quality_error": "quality_identity_mismatch",
    }) is True
    assert planner._runtime_successful_for_energy({
        "ok": False,
        "runtime_success": False,
    }) is False
    assert planner._runtime_successful_for_energy({"ok": True}) is True

    analysis = tmp_path / "analysis_tables"
    analysis.mkdir()
    result = tmp_path / "hailo10_native_fifo_e2e_results.json"
    result.write_text(json.dumps({
        "ok": False,
        "runtime_success": True,
        "technical_quality_error": "quality_identity_mismatch",
    }), encoding="utf-8")
    (analysis / "native_hailo10h_producer_e2e_eval_test.json").write_text(
        json.dumps({"rows": [{
            "model": "resnet50", "case": "b500",
            "precision": "uint8_dequant_fp16", "ok": False,
            "runtime_success": True, "report": str(result),
        }]}),
        encoding="utf-8",
    )
    normalized = final_report._rows_from_hailo10(tmp_path)
    assert len(normalized) == 1
    assert normalized[0]["ok"] is False
    assert normalized[0]["runtime_success"] is True
    assert planner._runtime_successful_for_energy(normalized[0]) is True

    aggregated = final_report._aggregate_repetitions([
        {
            "backend": "native_full_hailo10h", "model": "resnet50",
            "case": "full", "precision": "uint8", "setup_id": "setup-1",
            "comparison_backend": "hailo10h", "ok": False,
            "runtime_success": True, "fps_makespan": 10.0,
            "report": "repeat-1.json",
        },
        {
            "backend": "native_full_hailo10h", "model": "resnet50",
            "case": "full", "precision": "uint8", "setup_id": "setup-1",
            "comparison_backend": "hailo10h", "ok": False,
            "runtime_success": True, "fps_makespan": 11.0,
            "report": "repeat-2.json",
        },
    ])
    assert len(aggregated) == 1
    assert aggregated[0]["ok"] is False
    assert aggregated[0]["runtime_success"] is True
    assert planner._runtime_successful_for_energy(aggregated[0]) is True

    full_aggregate = full_runner._aggregate_full_repetitions([
        {
            "backend": "native_full_hailo10h", "model": "resnet50",
            "case": "full", "ok": False, "runtime_success": True,
            "fps_makespan": 10.0, "runtime_instance_id": "one",
        },
        {
            "backend": "native_full_hailo10h", "model": "resnet50",
            "case": "full", "ok": False, "runtime_success": True,
            "fps_makespan": 11.0, "runtime_instance_id": "two",
        },
    ], requested=2)
    assert full_aggregate["ok"] is False
    assert full_aggregate["runtime_success"] is True


def test_failed_accuracy_gate_cannot_be_labelled_quality_qualified() -> None:
    planner = _load_script("native_producer_energy_plan.py")
    admission = {
        "admission_scope": "native_energy",
        "central_quality_evidence_verified": True,
        "precision_quality_binding_verified": True,
        "task_quality_observation_valid": True,
        "accuracy_gate_pass": False,
        "quality_provenance_complete": True,
        "quality_claim_result_verified": True,
        "diagnostic_only": True,
        "claim_comparable": False,
        "energy_claim_eligible": False,
    }
    fields = planner._energy_quality_result_fields(admission)
    assert fields["energy_quality_qualified"] is False
    assert fields["energy_quality_status"] == (
        "raw_energy_quality_not_qualified"
    )


def test_energy_executor_collects_quality_failed_row_then_independent_valid_row(
    tmp_path: Path, monkeypatch,
) -> None:
    runner = _load_script("run_native_producer_energy_from_summary.py")
    summary = tmp_path / "native-summary.json"
    summary.write_text(json.dumps({"rows": []}), encoding="utf-8")
    out = tmp_path / "native-energy"
    plan_attempt = "v27916-row-local-plan"

    unqualified_admission = {
        "admission_scope": "native_runtime_observation",
        "central_quality_evidence_verified": False,
        "precision_quality_binding_verified": False,
        "task_quality_observation_valid": False,
        "accuracy_gate_pass": False,
        "quality_provenance_complete": False,
        "quality_claim_result_verified": False,
        "diagnostic_only": True,
        "screening_comparable": False,
        "claim_comparable": False,
        "energy_claim_eligible": False,
        "runtime_observation_reason": "technical_quality_error",
    }
    qualified_admission = {
        "admission_scope": "native_energy",
        "central_quality_evidence_verified": True,
        "precision_quality_binding_verified": True,
        "task_quality_observation_valid": True,
        "accuracy_gate_pass": True,
        "quality_provenance_complete": True,
        "quality_claim_result_verified": True,
        "diagnostic_only": False,
        "screening_comparable": True,
        "claim_comparable": True,
        "energy_claim_eligible": True,
    }

    def planned_row(
        *, case: str, run_id: str, admission: dict,
    ) -> dict:
        from onnx_splitpoint_tool.native_energy_quality_admission import (
            SCHEMA,
            SCHEMA_VERSION,
            canonical_json_sha256,
        )

        base = out / "measurements" / case / plan_attempt
        command_contract_sha256 = "a" * 64
        command = shlex.join([
            sys.executable,
            "energy_measurement_cli.py",
            "measure",
            "--setup-id", "setup-a",
            "--run-id", run_id,
            "--runs", "1",
            "--out", str(base),
            "--command", "native-workload --frames 10",
        ])
        sealed_admission = {
            "schema": SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "backend": "hailo8_to_trt",
            "model": "resnet50",
            "case": case,
            "setup_id": "setup-a",
            "comparison_backend": "hailo8",
            "precision": "fp16",
            "successful_command_contract_sha256": (
                command_contract_sha256
            ),
            **admission,
        }
        admission_sha256 = canonical_json_sha256(sealed_admission)
        sealed_admission["admission_sha256"] = admission_sha256
        row = {
            "backend": "hailo8_to_trt",
            "model": "resnet50",
            "case": case,
            "setup_id": "setup-a",
            "comparison_backend": "hailo8",
            "precision": "fp16",
            "measurement_output_dir": str(base),
            "measurement_output_base_dir": str(base),
            "measurement_plan_attempt_id": plan_attempt,
            "measurement_setup_id": "setup-a",
            "measurement_run_id": run_id,
            "measurement_requested_repeats": 1,
            "measurement_profile_requested_repeats": 1,
            "measurement_effective_repeats": 1,
            "measurement_repeat_expansion_applied": False,
            "measurement_repeat_expansion_reason": "none",
            "measure_command": command,
            "successful_command_contract_sha256": (
                command_contract_sha256
            ),
            "energy_quality_admission": sealed_admission,
            "energy_quality_admission_sha256": admission_sha256,
        }
        if admission["diagnostic_only"] is True:
            row.update({
                "claim_ok": False,
                "semantic_claim_ok": False,
                "claim_eligible": False,
                "eligible_for_energy_results_import": False,
                "eligible_for_scientific_claim": False,
                "energy_claim_eligible": False,
            })
        return row

    rows = [
        planned_row(
            case="quality-failed",
            run_id="row-quality-failed",
            admission=unqualified_admission,
        ),
        planned_row(
            case="quality-valid",
            run_id="row-quality-valid",
            admission=qualified_admission,
        ),
    ]
    plan_payload = {
        "measurement_plan_attempt_id": plan_attempt,
        "energy_runs_per_row": 1,
        "energy_profile_requested_runs_per_row": 1,
        "energy_effective_runs_per_row": 1,
        "energy_repeat_expansion_applied": False,
        "energy_repeat_expansion_reason": "none",
        "preflight_status": "passed",
        "technical_measurement_contract_valid": True,
        "preflight": {
            "status": "passed",
            "ok": True,
            "measurement_start_allowed": True,
            "energy_plan_coverage_contract_valid": True,
            "technical_measurement_contract_valid": True,
        },
        "rows": rows,
    }
    measured_run_ids: list[str] = []

    def fake_run(command, timeout=None, label=""):
        del timeout
        if label == "energy_plan":
            plan_dir = Path(command[command.index("--out-dir") + 1])
            plan_dir.mkdir(parents=True, exist_ok=True)
            (plan_dir / "native_producer_energy_plan.json").write_text(
                json.dumps(plan_payload), encoding="utf-8",
            )
            return {"rc": 0, "elapsed_s": 0.01}

        command_text = shlex.join([str(value) for value in command])
        measurement_dir = Path(
            runner._command_option(command_text, "--out")
        )
        run_id = runner._command_option(command_text, "--run-id")
        measured_run_ids.append(run_id)
        measurement_dir.mkdir(parents=True, exist_ok=True)
        aggregate = {
            "ok": True,
            "status": "ok",
            "setup_id": "setup-a",
            "run_id": run_id,
            "out_dir": str(measurement_dir.resolve()),
            "run_count": 1,
            "requested_valid_repeat_count": 1,
            "materialized_logical_repeat_count": 1,
            "valid_postprocessed_runs": 1,
            "scientific_primary_valid_run_count": 1,
            "scientific_primary_method": "command_marker_window",
            "scientific_primary_energy_status": "available",
            "repeat_contract_complete": True,
            "energy_window_method_ab": {
                "requested_run_count": 1,
                "effective_run_count": 1,
                "scientific_primary_method": "command_marker_window",
            },
            "runs": [{
                "run_index": 0,
                "scientific_primary_energy_j": 10.0,
            }],
            "scientific_primary_energy_statistics": {
                "energy_j": {
                    "n": 1,
                    "mean": 10.0,
                    "ci_low": 10.0,
                    "ci_high": 10.0,
                },
            },
        }
        (measurement_dir / "energy_aggregate.json").write_text(
            json.dumps(aggregate), encoding="utf-8",
        )
        return {"rc": 0, "elapsed_s": 0.01}

    monkeypatch.setattr(runner, "_run", fake_run)
    monkeypatch.setattr(sys, "argv", [
        str(runner.__file__),
        "--summary", str(summary),
        "--out-dir", str(out),
        "--runs", "1",
    ])

    assert runner.main() == 0
    report = json.loads(
        (out / "native_producer_energy_results.json").read_text(
            encoding="utf-8"
        )
    )
    assert measured_run_ids == ["row-quality-failed", "row-quality-valid"]
    assert report["started_measurement_count"] == 2
    assert report["ok"] is True
    assert len(report["rows"]) == 2
    failed_quality, valid_quality = report["rows"]
    assert failed_quality["ok"] is True
    assert failed_quality["energy_quality_qualified"] is False
    assert failed_quality["energy_quality_status"] == (
        "raw_energy_quality_not_qualified"
    )
    assert failed_quality["native_energy_after_technical_error"] == (
        "collect_raw_quality_unqualified"
    )
    assert failed_quality["run"]["energy_aggregate_verified"] is True
    assert failed_quality["run"]["measurement_start_observation"][
        "measurement_started"
    ] is True
    assert valid_quality["ok"] is True
    assert valid_quality["energy_quality_qualified"] is True
    assert valid_quality["energy_quality_status"] == "quality_qualified"
    assert valid_quality["run"]["energy_aggregate_verified"] is True


def test_energy_executor_global_infrastructure_stops_remaining_rows(
    tmp_path: Path, monkeypatch,
) -> None:
    runner = _load_script("run_native_producer_energy_from_summary.py")
    summary = tmp_path / "native-summary.json"
    summary.write_text(json.dumps({"rows": []}), encoding="utf-8")
    out = tmp_path / "native-energy"
    plan_attempt = "v27916-global-stop-plan"

    def planned_row(model: str) -> dict:
        base = out / "measurements" / model / plan_attempt
        run_id = f"energy-{model}"
        command = shlex.join([
            sys.executable, "energy_measurement_cli.py", "measure",
            "--setup-id", "setup-a", "--run-id", run_id,
            "--runs", "1", "--out", str(base),
            "--command", "native-workload --frames 10",
        ])
        return {
            "backend": "native_full_tensorrt",
            "model": model,
            "case": "full",
            "setup_id": "setup-a",
            "comparison_backend": "hailo8",
            "measurement_output_dir": str(base),
            "measurement_output_base_dir": str(base),
            "measurement_plan_attempt_id": plan_attempt,
            "measurement_setup_id": "setup-a",
            "measurement_run_id": run_id,
            "measurement_requested_repeats": 1,
            "measurement_profile_requested_repeats": 1,
            "measurement_effective_repeats": 1,
            "measurement_repeat_expansion_applied": False,
            "measurement_repeat_expansion_reason": "none",
            "measure_command": command,
        }

    rows = [planned_row(model) for model in ("model-a", "model-b", "model-c")]
    plan_payload = {
        "measurement_plan_attempt_id": plan_attempt,
        "energy_runs_per_row": 1,
        "energy_profile_requested_runs_per_row": 1,
        "energy_effective_runs_per_row": 1,
        "energy_repeat_expansion_applied": False,
        "energy_repeat_expansion_reason": "none",
        "preflight_status": "passed",
        "technical_measurement_contract_valid": True,
        "preflight": {
            "status": "passed", "ok": True,
            "measurement_start_allowed": True,
            "energy_plan_coverage_contract_valid": True,
            "technical_measurement_contract_valid": True,
        },
        "rows": rows,
    }
    measured_run_ids: list[str] = []

    def fake_run(command, timeout=None, label=""):
        del timeout
        if label == "energy_plan":
            plan_dir = Path(command[command.index("--out-dir") + 1])
            plan_dir.mkdir(parents=True, exist_ok=True)
            (plan_dir / "native_producer_energy_plan.json").write_text(
                json.dumps(plan_payload), encoding="utf-8",
            )
            return {"rc": 0, "elapsed_s": 0.01}
        command_text = shlex.join([str(value) for value in command])
        measurement_dir = Path(
            runner._command_option(command_text, "--out")
        )
        run_id = runner._command_option(command_text, "--run-id")
        measured_run_ids.append(run_id)
        measurement_dir.mkdir(parents=True, exist_ok=True)
        (measurement_dir / "energy_summary.json").write_text(json.dumps({
            "ok": False,
            "status": "collector_initialization_failed",
            "collector_started": False,
            "workload_started": False,
            "global_infrastructure_failure": (
                "collector_executable_not_found"
            ),
            "global_infrastructure_category": "collector_initialization",
            "global_infrastructure_detail": (
                "urecs-data-collector not found"
            ),
        }), encoding="utf-8")
        return {"rc": 1, "elapsed_s": 0.01, "stdout": "", "stderr": ""}

    monkeypatch.setattr(runner, "_run", fake_run)
    monkeypatch.setattr(sys, "argv", [
        str(runner.__file__), "--summary", str(summary),
        "--out-dir", str(out), "--runs", "1",
    ])

    assert runner.main() == 3
    report = json.loads(
        (out / "native_producer_energy_results.json").read_text(
            encoding="utf-8"
        )
    )
    assert measured_run_ids == ["energy-model-a"]
    assert report["status"] == "failed_global_energy_infrastructure"
    assert report["blocked_reason"] == "collector_executable_not_found"
    assert report["started_measurement_count"] == 0
    assert report["measurement_wrapper_started_count"] == 1
    assert report["result_ledger_valid"] is True
    assert len(report["rows"]) == 3
    assert [row.get("skipped") for row in report["rows"]] == [
        None, "global_infrastructure_stop", "global_infrastructure_stop",
    ]
    assert all(
        row["run"]["measurement_start_observation"][
            "measurement_started"
        ] is False
        for row in report["rows"]
    )
    assert report["energy_not_started_reason"].startswith(
        "Energy requested, not started: global infrastructure blocked by "
    )


def test_direct_stage_invalid_model_is_local_and_valid_model_reaches_energy_plan(
    tmp_path: Path, monkeypatch,
) -> None:
    from onnx_splitpoint_tool.remote.process_lease import (
        RemoteProcessLeaseScope,
    )
    from onnx_splitpoint_tool.workflow.runner import (
        EvaluationWorkflowRunner,
        WorkflowOptions,
    )

    workflow = EvaluationWorkflowRunner(
        WorkflowOptions(profile="", out=str(tmp_path))
    )
    workflow.run_id = "direct_row_isolation"
    workflow.run_dir = tmp_path / workflow.run_id
    workflow._remote_process_registry.configure_journal(
        scope=RemoteProcessLeaseScope(workflow.run_id, workflow.session_id),
        journal_dir=tmp_path / "direct_row_isolation_remote_leases",
    )
    workflow.profile_payload = {
        "campaign": {"mode": "measurement"},
        "execution_preset": {"id": "standard"},
        "measurement_campaign": {
            "system_power": {"scope": "FS", "window": "command"},
        },
    }
    workflow.profile_start_snapshot = {}
    workflow.manifest = {
        "models": {
            "resnet50": {},
            "yolov7_ultralytics": {},
        },
    }

    valid_suite = (
        workflow.run_dir / "models" / "resnet50" / "benchmark_set"
    )
    (valid_suite / "b001").mkdir(parents=True)
    (valid_suite / "b001" / "split_manifest.json").write_text(
        json.dumps({"part2_external_inputs": ["boundary_tensor"]}),
        encoding="utf-8",
    )
    (valid_suite / "benchmark_set.json").write_text(
        json.dumps({"cases": [{"id": "b001"}]}), encoding="utf-8"
    )
    (valid_suite / "benchmark_plan.json").write_text(
        json.dumps({"runs": [{"id": "split"}]}), encoding="utf-8"
    )
    (valid_suite / "benchmark_suite.py").write_text(
        "# direct row-isolation fixture\n", encoding="utf-8"
    )
    reports = workflow.run_dir / "reports"
    reports.mkdir(parents=True)
    (reports / "native_producer_summary.json").write_text(
        json.dumps({
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
        }),
        encoding="utf-8",
    )
    quality_summary = (
        workflow.run_dir / "quality_management" /
        "central_quality_summary.json"
    )
    quality_summary.parent.mkdir(parents=True)
    quality_summary.write_text(json.dumps({"results": []}), encoding="utf-8")
    registry_path = tmp_path / "hardware_setups.yaml"
    registry_path.write_text("hardware_setups: []\n", encoding="utf-8")

    cfg = {
        "enabled": True,
        "models": ["resnet50", "yolov7_ultralytics"],
        "backends": ["hailo8"],
        "precision": "fp16",
        "case_policy": "case_map_only",
        "case_map": {
            "resnet50": ["b001"],
            "yolov7_ultralytics": ["b001"],
        },
        "remotes": {
            "hailo8": {
                "ssh": "nx@hailo8",
                "setup_id": "hailo8_setup",
            },
        },
        "copy_benchmarksets": False,
        "build_missing_engines": False,
        "validation": {"enabled": False},
        "energy": {
            "enabled": True,
            "mode": "plan",
            "strict": False,
            "duration_s": 1.0,
            "timeout": 1,
        },
        "full_baselines": {
            "enabled": True,
            "backends_by_producer": {
                "hailo8": ["hailo8", "tensorrt"],
            },
        },
        "cleanup_remote_native_root": False,
    }
    labels: list[str] = []

    def fake_streaming(command, *_args, **kwargs):
        label = str(kwargs.get("label") or "")
        labels.append(label)
        if label == "energy:plan":
            energy_out = reports / "native_energy_plan"
            energy_out.mkdir(parents=True, exist_ok=True)
            (energy_out / "native_producer_energy_plan.json").write_text(
                json.dumps({
                    "preflight_status": (
                        "blocked_no_runtime_constructible_rows"
                    ),
                    "preflight": {
                        "status": "blocked_no_runtime_constructible_rows",
                        "measurement_start_allowed": False,
                    },
                    "rows": [],
                    "paired_missing_rows": [],
                }),
                encoding="utf-8",
            )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    def benchmark_status(path: Path) -> dict:
        model = path.parent.name
        if model == "yolov7_ultralytics":
            return {
                "valid": False,
                "selected_suite_dir": str(path),
                "reasons": ["benchmark_set_missing"],
            }
        return {
            "valid": True,
            "selected_suite_dir": str(path),
            "reasons": [],
        }

    monkeypatch.setattr(
        "onnx_splitpoint_tool.workflow.runner.run_streaming",
        fake_streaming,
    )
    monkeypatch.setattr(
        "onnx_splitpoint_tool.workflow.runner.normalize_hardware_targets",
        lambda *_a, **_k: [{
            "enabled": True,
            "id": "hailo8_setup",
            "setup_source": str(registry_path),
        }],
    )
    monkeypatch.setattr(
        "onnx_splitpoint_tool.workflow.runner.benchmark_set_postcondition_v60v",
        benchmark_status,
    )
    monkeypatch.setattr(
        "onnx_splitpoint_tool.workflow.runner._sync_remote_script_v60i",
        lambda *_a, **_k: [],
    )
    monkeypatch.setattr(
        "onnx_splitpoint_tool.workflow.runner._sync_remote_package_asset_v263",
        lambda *_a, **_k: [{
            "name": "sync_fixture",
            "rc": 0,
            "expected_sha256": "a" * 64,
        }],
    )
    monkeypatch.setattr(
        "onnx_splitpoint_tool.workflow.runner._verify_remote_module_binding_v263",
        lambda *_a, **_k: {"name": "verify_fixture", "rc": 0},
    )
    monkeypatch.setattr(
        workflow,
        "_finish_native_direct_remote_lease",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        "onnx_splitpoint_tool.trt_quality_chain.load_producer_set_from_central_quality_summary",
        lambda _summary, *, setup_id, model_ids, **_kwargs: {
            "schema": "test/tensorrt-quality-producer-set",
            "setup_id": setup_id,
            "producers_by_model": {
                model_id: {"model_id": model_id, "fixture": True}
                for model_id in model_ids
            },
        },
    )
    monkeypatch.setattr(
        "onnx_splitpoint_tool.trt_quality_chain.load_split_binding_set_from_central_quality_summary",
        lambda _summary, *, setup_id, selections, **_kwargs: {
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
        },
    )
    monkeypatch.setattr(workflow, "_native_producer_config", lambda: cfg)

    _paths, details, _message, _status = (
        workflow._stage_run_native_producers()
    )
    stage = json.loads(
        (reports / "native_producer_stage.json").read_text(encoding="utf-8")
    )
    rows_by_model = {
        row["model"]: row for row in stage["model_preflight_rows"]
    }
    blocked = rows_by_model["yolov7_ultralytics"]
    ready = rows_by_model["resnet50"]

    assert stage["models"] == ["resnet50"]
    assert set(stage["excluded_models"]) == {"yolov7_ultralytics"}
    assert blocked["status"] == "blocked"
    assert blocked["energy_not_started_categories"] == ["model_preflight"]
    assert blocked["energy_not_started_reason"] == (
        "Energy requested, not started: Native preflight blocked by "
        "benchmark_set_invalid:yolov7_ultralytics"
    )
    assert ready["status"] == "ready"
    assert ready["energy_not_started_reason"] == ""
    assert "split:hailo8:generic" in labels
    assert "full:hailo8:hailo8,tensorrt" in labels
    assert "energy:plan" in labels
    assert stage["started_remote_count"] > 0
    assert stage["started_performance_count"] > 0
    assert details["started_remote_count"] == stage["started_remote_count"]
    assert (
        details["started_performance_count"]
        == stage["started_performance_count"]
    )


def test_explicit_generic_request_survives_materialization_and_dispatch() -> None:
    from onnx_splitpoint_tool.execution_plan import (
        build_effective_execution_plan,
    )
    from onnx_splitpoint_tool.run_modes import (
        apply_run_mode, default_run_modes_config,
    )
    from onnx_splitpoint_tool.workflow.execution_binding import (
        _energy_enabled_for_profile,
    )

    resolved, _audit = apply_run_mode(
        _profile(explicit_generic=True),
        config=default_run_modes_config(),
    )
    assert resolved["energy"]["generic_enabled"] is True
    assert resolved["energy"]["measurement_path"] == "native_and_generic"
    plan = build_effective_execution_plan(resolved)
    assert plan["generic_energy_enabled"] is True
    assert plan["native_energy_enabled"] is True
    assert plan["energy_measurement_path"] == "native_and_generic"
    assert plan["energy_plan_blocked"] is False
    assert _energy_enabled_for_profile(SimpleNamespace(), resolved) is True


def test_inconsistent_generic_request_is_a_visible_planning_blocker() -> None:
    from onnx_splitpoint_tool.execution_plan import (
        build_effective_execution_plan, execution_plan_text,
    )
    from onnx_splitpoint_tool.run_modes import (
        apply_run_mode, default_run_modes_config,
    )

    resolved, _audit = apply_run_mode(
        _profile(explicit_generic=False),
        config=default_run_modes_config(),
    )
    resolved["energy"].update({
        "requested": True,
        "generic_enabled": False,
        "measurement_path": "native_and_generic",
    })
    plan = build_effective_execution_plan(resolved)
    assert plan["generic_energy_enabled"] is False
    assert plan["energy_plan_blocked"] is True
    assert (
        "generic_energy_path_requested_but_generic_energy_disabled"
        in plan["energy_configuration_errors"]
    )
    assert "Energy requested, not started: planning blocked by" in str(
        plan["warnings"]
    )
    assert "planning=blocked:" in execution_plan_text(plan)
    runner_source = (
        ROOT / "onnx_splitpoint_tool/workflow/runner.py"
    ).read_text(encoding="utf-8")
    assert 'energy_not_started_categories": (' in runner_source
    assert '("failed" if energy_plan_blocked else "ok")' in runner_source


def test_gui_result_surface_exposes_native_energy_preflight_reason() -> None:
    source = (ROOT / "onnx_splitpoint_tool/gui/app.py").read_text(
        encoding="utf-8",
    )
    assert "_evaluation_energy_not_started_messages(payload)" in source
    assert 'lines.extend(["", "Energy:"])' in source
    assert "Energy requested, not started:" in source


def test_partial_quality_contracts_are_row_local_in_remote_runners() -> None:
    producer = (ROOT / "scripts/native_producer_e2e_eval_runner.py").read_text(
        encoding="utf-8",
    )
    hailo8 = (ROOT / "scripts/native_fifo_smoke_matrix.py").read_text(
        encoding="utf-8",
    )
    for source in (producer, hailo8):
        assert "row_quality_first" in source
        assert "runtime_observation_quality_unqualified" in source
        assert "collect_raw_quality_unqualified" in source


def test_hailo10_known_contract_failure_keeps_later_rows_runnable() -> None:
    source = (ROOT / "onnx_splitpoint_tool/workflow/runner.py").read_text(
        encoding="utf-8",
    )
    start = source.index(
        'result["known_contract_backend"] = "hailo10h"'
    )
    end = source.index('elif b == "deepx":', start)
    block = source[start:end]
    assert 'result.setdefault(\n                                    "known_contract_failures"' in block
    assert "remaining contracts " in block
    assert 'if infrastructure_failure:' in block
    assert "remote Hailo10H infrastructure failed" in block
    assert "remote Hailo10H known-contract runner failed" not in block


def test_generic_composed_quality_request_duplicates_existing_preprocessing_field() -> None:
    """Regression for the five TensorRT request failures in the debug pack."""

    source = (
        ROOT / "onnx_splitpoint_tool/resources/templates/"
        "run_split_onnxruntime.py.txt"
    ).read_text(encoding="utf-8")
    start = source.index("generic_composed_duplicates = {")
    end = source.index("\n            }", start)
    block = source[start:end]
    assert '"preprocessing_contract_sha256": str(' in block
    assert 'generic_producer["preprocessing_contract_sha256"]' in block


def test_remote_script_mirrors_are_byte_identical() -> None:
    for name in (
        "native_producer_energy_plan.py",
        "run_native_producer_energy_from_summary.py",
        "native_producer_e2e_eval_runner.py",
        "native_fifo_smoke_matrix.py",
        "native_producer_final_report.py",
        "native_full_baseline_eval_runner.py",
    ):
        assert (ROOT / "scripts" / name).read_bytes() == (
            ROOT / "onnx_splitpoint_tool/resources/remote_scripts" / name
        ).read_bytes()


def test_final_reporting_never_labels_unstarted_row_as_raw_energy(
    tmp_path: Path,
) -> None:
    from onnx_splitpoint_tool.native_energy_reporting import (
        collect_native_energy,
    )

    measurements = tmp_path / "reports" / "native_energy_measurements"
    measurements.mkdir(parents=True)
    (measurements / "native_producer_energy_results.json").write_text(
        json.dumps({
            "rows": [{
                "row": {
                    "backend": "hailo10h_to_trt",
                    "model": "yolov7_ultralytics",
                    "case": "b500",
                    "energy_quality_qualified": False,
                },
                "ok": False,
                "skipped": "benchmark_set_invalid:yolov7_ultralytics",
                # Stale display labels cannot overrule physical evidence.
                "energy_quality_status": "raw_energy_quality_not_qualified",
                "native_energy_after_technical_error": (
                    "collect_raw_quality_unqualified"
                ),
                "run": {
                    "measurement_start_observation": {
                        "measurement_started": False,
                        "collector_started_repeat_count": 0,
                        "workload_started_repeat_count": 0,
                    },
                },
            }],
        }),
        encoding="utf-8",
    )

    [row] = collect_native_energy(tmp_path)
    assert row["measurement_started"] is False
    assert row["raw_energy_collected"] is False
    assert row["energy_quality_qualified"] is False
    assert row["energy_quality_status"] == "energy_not_collected"
    assert row["native_energy_after_technical_error"] == (
        "not_collected_measurement_not_started"
    )
