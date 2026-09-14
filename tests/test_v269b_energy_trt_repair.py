from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(f"v269b_{path.stem}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_trt_performance_is_one_process_duration_zero_exact_iterations(tmp_path: Path) -> None:
    module = _load_script("native_trt_from_benchmarkset.py")
    warmup = module._same_process_warmup_contract(10, 0)
    command = module._engine_run_cmd(
        "trtexec", tmp_path / "model.engine", "", 123,
        warmup["warmup_ms_effective"], 0,
        [f"--exportTimes={tmp_path / 'times.json'}"],
    )
    assert "--iterations=123" in command
    assert "--duration=0" in command
    assert "--warmUp=200" in command
    assert warmup["warmup_and_measurement_same_process"] is True
    assert warmup["warmup_and_measurement_same_execution_context"] is True
    assert warmup["warmup_iterations_completed"] is None
    assert warmup["warmup_iterations_status"] == "not_countable_trtexec_internal_time_window"
    assert warmup["warmup_work_equivalence"] == "cross_backend_not_iteration_equivalent"
    assert warmup["warmup_iterations_observed"] is False
    assert warmup["warmup_budget_source"] == (
        "requested_iterations_trigger_conservative_trtexec_200ms_minimum"
    )
    source = (ROOT / "scripts" / "native_trt_from_benchmarkset.py").read_text(encoding="utf-8")
    assert "separate_exact_iteration_invocation" not in source
    assert "_run(warmup_cmd" not in source


def test_trt_energy_export_times_binds_exact_count_and_trace_span(tmp_path: Path) -> None:
    module = _load_script("native_full_baseline_eval_runner.py")
    trace = tmp_path / "times.json"
    trace.write_text(json.dumps([
        {
            "startEnqMs": 0.2, "endEnqMs": 1.0,
            "startH2dMs": 0.1, "endH2dMs": 0.2,
            "startComputeMs": 0.3, "endComputeMs": 1.1,
            "startD2hMs": 1.1, "endD2hMs": 1.2,
            "latencyMs": 1.1,
        },
        {
            "startEnqMs": 59_999.0, "endEnqMs": 60_000.0,
            "startH2dMs": 59_998.9, "endH2dMs": 59_999.0,
            "startComputeMs": 59_999.1, "endComputeMs": 60_000.1,
            "startD2hMs": 60_000.1, "endD2hMs": 60_000.2,
            "latencyMs": 1.3,
        },
    ]), encoding="utf-8")
    evidence = module._trtexec_exported_iteration_evidence(trace)
    assert evidence["status"] == "ok"
    assert evidence["completed_work_units"] == 2
    assert evidence["measured_trace_duration_s"] == pytest.approx(60.0001)


def test_energy_plan_never_derives_work_count_from_historical_fps() -> None:
    module = _load_script("native_producer_energy_plan.py")
    assert module._duration_controlled_minimum_work_units(0) == (
        1, "duration_driven_minimum_one",
    )
    assert module._duration_controlled_minimum_work_units(77) == (
        77, "explicit_legacy_minimum_override",
    )
    source = (ROOT / "scripts" / "native_producer_energy_plan.py").read_text(
        encoding="utf-8",
    )
    assert "ceil(fps * duration_s)" not in source
    assert "missing_or_zero_fps" not in source


def test_hailo_repeat_and_runtime_ids_are_unique(tmp_path: Path) -> None:
    module = _load_script("native_hailo_trt_fifo_from_benchmarkset.py")
    first = module._fresh_hailo_repetition_identity(1, tmp_path / "result.json")
    second = module._fresh_hailo_repetition_identity(1, tmp_path / "result.json")
    assert first[0].startswith("hailo8:")
    assert first[1].startswith("fresh_process:")
    assert first[0] != second[0]
    assert first[1] != second[1]


def test_hailo_independent_local_repeat_one_rows_survive_final_aggregation() -> None:
    module = _load_script("native_producer_final_report.py")
    common = {
        "backend": "hailo8",
        "producer_impl": "hailo8_fifo_trt_part2",
        "model": "resnet50",
        "case": "split_1",
        "precision": "fp16",
        "setup_id": "hailo8_setup",
        "task": "classification",
        "ok": True,
        "report": "/same/copied/report.json",
        "repetition_index": 1,
        "repetition_count_requested": 1,
        "repetition_count_attempted": 1,
    }
    rows = []
    for token, fps in (("a", 80.0), ("b", 100.0)):
        repeat_id = f"hailo8:{token}"
        runtime_id = f"fresh_process:{token}"
        rows.append({
            **common,
            "fps_makespan": fps,
            "latency_mean_ms": 1000.0 / fps,
            "repetition_id": repeat_id,
            "runtime_instance_id": runtime_id,
            "repetition_records": [{
                "repetition_index": 1,
                "repetition_id": repeat_id,
                "runtime_instance_id": runtime_id,
                "repetition_runtime_scope": "fresh_process_per_repetition",
                "ok": True,
                "fps_makespan": fps,
                "latency_mean_ms": 1000.0 / fps,
            }],
        })

    aggregated = module._aggregate_repetitions(rows)
    assert len(aggregated) == 1
    result = aggregated[0]
    assert result["fps_repetition_samples"] == [80.0, 100.0]
    assert result["fps_median"] == pytest.approx(90.0)
    assert result["repetition_count_requested"] == 2
    records = result["repetition_records"]
    assert len(records) == 2
    assert {record["repetition_id"] for record in records} == {
        "hailo8:a", "hailo8:b",
    }
    assert [record["repetition_index"] for record in records] == [1, 2]
    assert [record["source_repetition_index"] for record in records] == [1, 1]


def test_collector_reconnect_evidence_and_full_log_binding(tmp_path: Path) -> None:
    from onnx_splitpoint_tool.energy.collector import (
        _collector_log_diagnostics,
        _collector_reconnect_backoff,
    )

    for name, text in (
        ("collector_stdout.log", "collector complete\n"),
        ("collector_stderr.log", "first sample recovered\n"),
        ("workload_stdout.log", "work units=123\n"),
        ("workload_stderr.log", ""),
    ):
        (tmp_path / name).write_text(text, encoding="utf-8")
    recovery = _collector_reconnect_backoff(0.0)
    assert recovery["transport_policy"].startswith("fresh_collector_process")
    diagnostics = _collector_log_diagnostics(tmp_path)
    assert all(item["available"] is True for item in diagnostics.values())
    assert all(len(item["sha256"]) == 64 for item in diagnostics.values())
    assert diagnostics["collector_stderr"]["tail"] == "first sample recovered\n"


def test_first_sample_failure_executes_bounded_retry_and_records_backoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the real collector retry section, not only its helpers."""
    import onnx_splitpoint_tool.energy.collector as collector_module
    from onnx_splitpoint_tool.energy.collector import run_fast_firmware_measurement
    from onnx_splitpoint_tool.energy.config import EnergyDefaults, EnergySetup

    fake_collector = tmp_path / "fake_collector.py"
    fake_collector.write_text(
        """#!/usr/bin/env python3
import os
import pathlib
import subprocess
import sys

if '--help' in sys.argv:
    print('fake collector')
    raise SystemExit(0)
state = pathlib.Path(os.environ['V269B_RETRY_STATE'])
attempt = int(state.read_text() or '0') if state.exists() else 0
state.write_text(str(attempt + 1))
storage = pathlib.Path(next(value[3:] for value in sys.argv if value.startswith('-s=')))
storage.mkdir(parents=True, exist_ok=True)
if attempt == 0:
    (storage / 'samples.parquet').write_bytes(b'PAR1')
    print('First-sample barrier failed: channel is empty because sending half is closed', file=sys.stderr)
    raise SystemExit(9)
command = pathlib.Path(next(value[3:] for value in sys.argv if value.startswith('-c=')))
(storage / 'samples.parquet').write_bytes(b'PAR1' + b'x' * 4096 + b'PAR1')
raise SystemExit(subprocess.run([str(command)], check=False).returncode)
""",
        encoding="utf-8",
    )
    fake_collector.chmod(0o755)
    state = tmp_path / "collector_attempts.txt"
    monkeypatch.setenv("V269B_RETRY_STATE", str(state))

    backoff_calls: list[float] = []

    def _observed_backoff(seconds: float, *, cancel_event=None):
        backoff_calls.append(float(seconds))
        return {
            "requested_backoff_s": float(seconds),
            "elapsed_backoff_s": float(seconds),
            "cancelled": False,
            "transport_policy": (
                "fresh_collector_process_fresh_storage_reconnect_after_quiescence"
            ),
        }

    monkeypatch.setattr(
        collector_module, "_collector_reconnect_backoff", _observed_backoff,
    )
    defaults = EnergyDefaults(
        collector_binary=str(fake_collector),
        power_calculations_binary="unused",
        pre_duration_s=0,
        post_duration_s=0,
        run_count=1,
        compare_legacy_window=False,
        postprocess_with_power_calculations=False,
        invalid_repeat_max_retries=1,
        invalid_repeat_reconnect_backoff_s=0.25,
    )
    output_dir = tmp_path / "measurement"
    result = run_fast_firmware_measurement(
        "printf 'workload complete\\n'",
        output_dir,
        setup=EnergySetup(
            setup_id="retry-integration", enabled=True,
            urecs_address="127.0.0.1",
        ),
        defaults=defaults,
        duration_s=0.01,
        run_count=1,
        postprocess=False,
        compare_legacy_window=False,
    )

    assert state.read_text(encoding="utf-8") == "2"
    assert backoff_calls == [0.25]
    assert result["repeat_retry_attempt_count"] == 1
    history = result["repeat_retry_history"][0]
    assert "first_sample_barrier_invalid" in history["initial_retry_reasons"]
    failed_placeholder = (
        output_dir / "run_000" / "collector_storage" / "samples.parquet"
    )
    assert failed_placeholder.read_bytes() == b"PAR1"
    failed_summary = json.loads(
        (output_dir / "run_000" / "energy_summary.json").read_text(
            encoding="utf-8",
        )
    )
    assert failed_summary["collector_rc"] == 9
    assert failed_summary["parquet_file_sizes"][str(failed_placeholder)] == 4
    assert failed_summary["final_energy_gate_status"] == "fail"
    assert history["attempts"][0]["selected"] is False
    assert len(history["attempts"]) == 2
    retry = history["attempts"][1]
    assert retry["collector_reconnect_evidence"]["requested_backoff_s"] == 0.25
    evidence_path = (
        output_dir / "repeat_retry_attempts" / "repeat_000" / "attempt_01"
        / "collector_reconnect_evidence.json"
    )
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert evidence["trigger_reasons"] == ["first_sample_barrier_invalid"]
    assert evidence["transport_policy"].startswith("fresh_collector_process")


def test_debug_pack_bounds_energy_process_logs_above_size_limit(
    tmp_path: Path,
) -> None:
    module = _load_script("create_evaluation_debug_pack.py")
    run_dir = tmp_path / "run"
    log_path = (
        run_dir / "reports" / "native_energy_measurements" / "attempt"
        / "collector_stderr.log"
    )
    log_path.parent.mkdir(parents=True)
    log_path.write_bytes(b"diagnostic\n" * 100)
    include, reason = module._should_include(
        run_dir, log_path, 32, probe_include_raw=False,
    )
    assert include is False
    assert reason == "too large"
