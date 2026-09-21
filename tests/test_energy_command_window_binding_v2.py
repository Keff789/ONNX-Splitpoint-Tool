from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import pytest

from onnx_splitpoint_tool.energy.collector import (
    _command_window_binding,
    _prepare_command_window_request_v2,
    _window_comparison_trace_binding,
    _window_alignment,
    run_fast_firmware_measurement,
)
from onnx_splitpoint_tool.energy.config import (
    EnergyDefaults,
    EnergySetup,
    energy_defaults_from_registry,
)
from onnx_splitpoint_tool.energy.metrics import extract_power_calculation_summary


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_legacy_window_comparison_defaults_on_and_can_be_disabled_in_config() -> None:
    assert EnergyDefaults().compare_legacy_window is True
    defaults = energy_defaults_from_registry(
        {"energy_defaults": {"compare_legacy_window": False}}
    )
    assert defaults.compare_legacy_window is False


def test_window_comparison_trace_binding_requires_exact_storage_path_and_hash_chain(
    tmp_path: Path,
) -> None:
    storage = tmp_path / "collector_storage"
    storage.mkdir()
    trace = storage / "fast_firmware.parquet"
    trace.write_bytes(b"trace")
    digest = _sha(trace)
    binding = _window_comparison_trace_binding(
        storage_dir=storage,
        request_trace_path=trace,
        request_trace_sha256=digest,
        trace_sha256_before=digest,
        trace_sha256_after=digest,
    )
    assert binding["verified"] is True
    assert binding["collector_storage_parquet_count"] == 1

    (storage / "stale.parquet").write_bytes(b"stale")
    binding = _window_comparison_trace_binding(
        storage_dir=storage,
        request_trace_path=trace,
        request_trace_sha256=digest,
        trace_sha256_before=digest,
        trace_sha256_after=digest,
    )
    assert binding["verified"] is False
    assert binding["collector_storage_parquet_count"] == 2
    assert "collector_storage_parquet_count_not_exactly_one" in binding["failure_reasons"]

    (storage / "stale.parquet").unlink()
    outside = tmp_path / "outside.parquet"
    outside.write_bytes(trace.read_bytes())
    binding = _window_comparison_trace_binding(
        storage_dir=storage,
        request_trace_path=outside,
        request_trace_sha256=digest,
        trace_sha256_before=digest,
        trace_sha256_after="",
    )
    assert binding["verified"] is False
    assert binding["request_trace_matches_collector_storage_parquet"] is False
    assert binding["request_before_after_hash_chain_complete"] is False


def _binding_fixture(
    tmp_path: Path,
    *,
    dropped_samples: int = 0,
    uncertainty_samples: int = 1,
    maximum_uncertainty_samples: int = 64,
    result_duration_s: float = 0.9995,
) -> tuple[dict, dict, Path]:
    trace = tmp_path / "fast_firmware.parquet"
    trace.write_bytes(b"firmware trace" * 400)
    command_path = tmp_path / "workload_command.sh"
    command_path.write_text("#!/bin/sh\necho workload\n", encoding="utf-8")
    timing_path = tmp_path / "workload_timing.txt"
    timing_path.write_text("start_ns=1000000000\nend_ns=2000000000\nrc=0\n", encoding="utf-8")
    marker_path = tmp_path / "command_window_markers.json"
    marker = {
        "schema": "urecs-data-collector/command-window-markers",
        "schema_version": 2,
        "run_id": "run-a",
        "window_id": "window-a",
        "command": {
            "text": str(command_path),
            "argv": [str(command_path)],
            "process_id": 123,
            "process_rc": 0,
            "process_error": None,
            "start_realtime_ns": 1_000_000_000,
            "end_realtime_ns": 2_000_000_000,
            "start_monotonic_ns": 10_000,
            "end_monotonic_ns": 20_000,
            "start_clock_read_uncertainty_ns": 10,
            "end_clock_read_uncertainty_ns": 10,
        },
        "stream": {
            "source": "fast_firmware",
            "trace_path": str(trace),
            "sample_rate_hz": 2000,
            "first_sample_index": 0,
            "last_sample_index": 1999,
            "total_samples": 2500,
            "boundary_uncertainty_samples": uncertainty_samples,
            "dropped_samples": dropped_samples,
            "trace_covers_window": True,
        },
        "valid_for_final_energy": True,
        "validation_errors": [],
    }
    _write_json(marker_path, marker)
    request_path = tmp_path / "command_window_request.json"
    request = {
        "schema": "onnx-splitpoint/command-window-request",
        "schema_version": 2,
        "binding_method": "collector_sample_marker_crop",
        "run_id": "run-a",
        "window_id": "window-a",
        "source": "fast_firmware",
        "marker_path": str(marker_path),
        "marker_sha256": _sha(marker_path),
        "trace_path": str(trace),
        "trace_sha256": _sha(trace),
        "command_path": str(command_path),
        "command_sha256": _sha(command_path),
        "workload_command_path": str(command_path),
        "workload_command_sha256": _sha(command_path),
        "timing_path": str(timing_path),
        "timing_sha256": _sha(timing_path),
        "first_sample_index": 0,
        "last_sample_index": 1999,
        "sample_count": 2000,
        "interval_count": 1999,
        "sample_rate_hz": 2000,
        "duration_s": 1999 / 2000,
        "process_rc": 0,
        "command_process_id": 123,
        "process_id": 123,
        "process_error": None,
        "start_clock_read_uncertainty_ns": 10,
        "end_clock_read_uncertainty_ns": 10,
        "command_start_realtime_ns": 1_000_000_000,
        "command_end_realtime_ns": 2_000_000_000,
        "command_start_monotonic_ns": 10_000,
        "command_end_monotonic_ns": 20_000,
        "command_start_clock_read_uncertainty_ns": 10,
        "command_end_clock_read_uncertainty_ns": 10,
        "dropped_samples": dropped_samples,
        "boundary_uncertainty_samples": uncertainty_samples,
        "maximum_boundary_uncertainty_samples": maximum_uncertainty_samples,
        "trace_covers_window": True,
        "valid_for_final_energy": True,
        "total_samples": 2500,
        "index_semantics": "zero_based_inclusive_rows",
        "energy_semantics": "calibrated_input_energy_unsubtracted",
        "energy_field": "firmware_results.energy",
    }
    _write_json(request_path, request)
    result_path = tmp_path / "results.yaml"
    result_data = {
        "firmware_results": {
            "energy": 5.0,
            "duration": result_duration_s,
            "start_stop_idx": [0, 1999],
            "max_frame_energy": 0.0,
            "idle_frame_energy": 0.0,
        }
    }
    result_path.write_text(json.dumps(result_data), encoding="utf-8")
    post_path = tmp_path / "postprocessor_window_result.json"
    post = {
        "schema": "power-calculations/command-window-result",
        "schema_version": 2,
        "status": "ok",
        "run_id": "run-a",
        "window_id": "window-a",
        "source": "fast_firmware",
        "request_path": str(request_path),
        "marker_path": str(marker_path),
        "trace_path": str(trace),
        "command_path": str(command_path),
        "timing_path": str(timing_path),
        "results_path": str(result_path),
        "request_sha256": _sha(request_path),
        "marker_sha256": _sha(marker_path),
        "trace_sha256": _sha(trace),
        "command_sha256": _sha(command_path),
        "timing_sha256": _sha(timing_path),
        "first_sample_index": 0,
        "last_sample_index": 1999,
        "sample_count": 2000,
        "interval_count": 1999,
        "sample_rate_hz": 2000,
        "duration_s": result_duration_s,
        "process_rc": 0,
        "command_process_id": 123,
        "command_start_realtime_ns": 1_000_000_000,
        "command_end_realtime_ns": 2_000_000_000,
        "command_start_monotonic_ns": 10_000,
        "command_end_monotonic_ns": 20_000,
        "command_start_clock_read_uncertainty_ns": 10,
        "command_end_clock_read_uncertainty_ns": 10,
        "drop_count": dropped_samples,
        "boundary_uncertainty_samples": uncertainty_samples,
        "maximum_boundary_uncertainty_samples": maximum_uncertainty_samples,
        "trace_covers_window": True,
        "energy_semantics": "calibrated_input_energy_unsubtracted",
        "energy_field": "firmware_results.energy",
        "energy_j": 5.0,
    }
    _write_json(post_path, post)
    binding = {
        "schema": "onnx-splitpoint/command-window-binding",
        "schema_version": 2,
        "binding_method": "collector_sample_marker_crop",
        "run_id": "run-a",
        "window_id": "window-a",
        "source": "fast_firmware",
        "request_path": str(request_path),
        "request_sha256": _sha(request_path),
        "marker_path": str(marker_path),
        "marker_sha256": _sha(marker_path),
        "trace_path": str(trace),
        "trace_sha256": _sha(trace),
        "command_path": str(command_path),
        "command_sha256": _sha(command_path),
        "workload_command_path": str(command_path),
        "workload_command_sha256": _sha(command_path),
        "timing_path": str(timing_path),
        "timing_sha256": _sha(timing_path),
        "postprocessor_result_path": str(post_path),
        "postprocessor_result_sha256": _sha(post_path),
        "result_path": str(result_path),
        "result_sha256": _sha(result_path),
        "first_sample_index": 0,
        "last_sample_index": 1999,
        "sample_count": 2000,
        "interval_count": 1999,
        "sample_rate_hz": 2000,
        "duration_s": result_duration_s,
        "process_rc": 0,
        "command_process_id": 123,
        "command_start_realtime_ns": 1_000_000_000,
        "command_end_realtime_ns": 2_000_000_000,
        "command_start_monotonic_ns": 10_000,
        "command_end_monotonic_ns": 20_000,
        "command_start_clock_read_uncertainty_ns": 10,
        "command_end_clock_read_uncertainty_ns": 10,
        "dropped_samples": dropped_samples,
        "boundary_uncertainty_samples": uncertainty_samples,
        "maximum_boundary_uncertainty_samples": maximum_uncertainty_samples,
        "trace_covers_window": True,
        "energy_semantics": "calibrated_input_energy_unsubtracted",
        "energy_field": "firmware_results.energy",
        "energy_j": 5.0,
    }
    _write_json(tmp_path / "command_window_binding.json", binding)
    timing = {"start_ns": 1_000_000_000, "end_ns": 2_000_000_000, "rc": 0}
    return result_data, timing, result_path


def test_v2_sample_marker_binding_verifies_every_artifact(tmp_path: Path) -> None:
    result_data, timing, result_path = _binding_fixture(tmp_path)
    binding = _command_window_binding(tmp_path, result_data, timing, result_path=result_path)
    assert binding["status"] == "verified"
    assert binding["calibrated_input_energy_unsubtracted_verified"] is True
    assert binding["first_sample_index"] == 0
    assert binding["last_sample_index"] == 1999
    assert binding["interval_count"] == 1999
    assert binding["integrated_duration_s"] == pytest.approx(1999 / 2000)

    alignment = _window_alignment(
        {
            "power_calculations_mode": "command_marker_crop",
            "active_duration_s": 1999 / 2000,
            "workload_execution_duration_s": 1.0,
            "command_window_trace_binding": binding,
        },
        "command_window",
    )
    assert alignment["energy_window_alignment_status"] == "pass"
    assert alignment["energy_window_effective"] == "command_window"


@pytest.mark.parametrize(
    ("kwargs", "expected_status"),
    [
        ({"dropped_samples": 1}, "dropped_samples_in_command_window"),
        (
            {"uncertainty_samples": 65, "maximum_uncertainty_samples": 64},
            "boundary_uncertainty_exceeds_limit",
        ),
        ({"result_duration_s": 1.1}, "postprocessor_duration_mismatch"),
    ],
)
def test_v2_binding_rejects_invalid_window_evidence(
    tmp_path: Path, kwargs: dict, expected_status: str
) -> None:
    result_data, timing, result_path = _binding_fixture(tmp_path, **kwargs)
    binding = _command_window_binding(tmp_path, result_data, timing, result_path=result_path)
    assert binding["verified"] is False
    assert binding["status"] == expected_status


def test_v2_binding_rejects_command_tampering(tmp_path: Path) -> None:
    result_data, timing, result_path = _binding_fixture(tmp_path)
    (tmp_path / "workload_command.sh").write_text("#!/bin/sh\necho tampered\n", encoding="utf-8")
    binding = _command_window_binding(tmp_path, result_data, timing, result_path=result_path)
    assert binding["status"] == "command_sha256_mismatch"


def test_v1_alignment_tolerance_is_explicit_and_zero_is_preserved(tmp_path: Path) -> None:
    trace = tmp_path / "trace.parquet"
    trace.write_bytes(b"trace")
    result_data = {"firmware_results": {"energy": 1.0, "duration": 1.0}}
    result_path = tmp_path / "results.yaml"
    result_path.write_text("result", encoding="utf-8")
    base = {
        "binding_method": "trace_timestamp_crop",
        "trace_path": str(trace),
        "trace_sha256": _sha(trace),
        "result_path": str(result_path),
        "result_sha256": _sha(result_path),
        "energy_semantics": "raw_input_energy",
        "energy_field": "firmware_results.energy",
        "raw_input_energy_j": 1.0,
        "command_start_ns": 10,
        "command_end_ns": 20,
        "integrated_start_ns": 10,
        "integrated_end_ns": 20,
    }
    _write_json(tmp_path / "command_window_binding.json", base)
    missing = _command_window_binding(tmp_path, result_data, {"start_ns": 10, "end_ns": 20}, result_path)
    assert missing["status"] == "alignment_tolerance_ns_missing"
    base["alignment_tolerance_ns"] = 0
    _write_json(tmp_path / "command_window_binding.json", base)
    verified = _command_window_binding(tmp_path, result_data, {"start_ns": 10, "end_ns": 20}, result_path)
    assert verified["status"] == "verified"
    assert verified["alignment_tolerance_ns"] == 0


def test_oscilloscope_energy_field_preserves_nested_results_path() -> None:
    summary = extract_power_calculation_summary(
        {"oscilloscope_results": {"results": {"energy": 3.0, "duration": 1.5}}}
    )
    assert summary["source_key"] == "oscilloscope_results.results"


def _write_executable(path: Path, source: str) -> None:
    path.write_text(source, encoding="utf-8")
    path.chmod(0o755)


@pytest.mark.parametrize("startup_budget", [15.0, 30.0])
@pytest.mark.parametrize("strict_claim_flags", [True, False])
def test_fake_collector_and_postprocessor_complete_v2_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    startup_budget: float, strict_claim_flags: bool,
) -> None:
    collector = tmp_path / "fake-collector"
    powercalc = tmp_path / "fake-power-calculations"
    _write_executable(
        collector,
        """#!/usr/bin/env python3
import json, pathlib, subprocess, sys, time
if '--help' in sys.argv:
    print('fake collector')
    raise SystemExit(0)
storage = pathlib.Path(next(x.split('=', 1)[1] for x in sys.argv if x.startswith('-s=')))
command = pathlib.Path(next(x.split('=', 1)[1] for x in sys.argv if x.startswith('-c=')))
run_id = sys.argv[sys.argv.index('--run-id') + 1]
window_id = sys.argv[sys.argv.index('--window-id') + 1]
storage.mkdir(parents=True, exist_ok=True)
trace = storage / 'fast_firmware.parquet'
# Model a late 14-second command end without making the unit test sleep.
# The old inner-duration-only -d=2s cannot cover this acquisition.
capture = float(next(x.split('=', 1)[1].removesuffix('s') for x in sys.argv if x.startswith('-d=')))
assert capture >= 14, 'late_command_end_outside_capture'
trace.write_bytes(b'firmware-trace' * 400)
start_rt = time.time_ns(); start_mono = time.monotonic_ns()
proc = subprocess.run([str(command)])
end_mono = time.monotonic_ns(); end_rt = time.time_ns()
marker = {
  'schema': 'urecs-data-collector/command-window-markers', 'schema_version': 2,
  'run_id': run_id, 'window_id': window_id,
  'command': {'text': str(command), 'argv': [str(command)], 'process_id': 123,
              'process_rc': proc.returncode, 'process_error': None,
              'start_realtime_ns': start_rt, 'end_realtime_ns': end_rt,
              'start_monotonic_ns': start_mono, 'end_monotonic_ns': end_mono,
              'start_clock_read_uncertainty_ns': 10, 'end_clock_read_uncertainty_ns': 10},
  'stream': {'source': 'fast_firmware', 'trace_path': str(trace),
             'sample_rate_hz': 2000, 'first_sample_index': 0,
             'last_sample_index': 1999, 'total_samples': 2500,
             'boundary_uncertainty_samples': 1, 'dropped_samples': 0,
             'trace_covers_window': True},
  'valid_for_final_energy': proc.returncode == 0, 'validation_errors': []}
(storage / 'command_window_markers.json').write_text(json.dumps(marker))
raise SystemExit(proc.returncode)
""",
    )
    _write_executable(
        powercalc,
        """#!/usr/bin/env python3
import hashlib, json, pathlib, sys
if '--help' in sys.argv:
    print('fake power calculations')
    raise SystemExit(0)
out = pathlib.Path(next(x.split('=', 1)[1] for x in sys.argv if x.startswith('--output-path=')))
if '--command-window-request' not in sys.argv:
    assert '-c' in sys.argv
    assert '-r' in sys.argv
    assert '--estimated-duration=3' in sys.argv
    out.mkdir(parents=True, exist_ok=True)
    (out / 'results.yaml').write_text(
        'firmware_results:\\n  energy: 4.8\\n  duration: 0.95' +
        '\\n  start_stop_idx: [10, 1909]\\n  max_frame_energy: 0.0\\n  idle_frame_energy: 0.0\\n')
    raise SystemExit(0)
idx = sys.argv.index('--command-window-request')
request_path = pathlib.Path(sys.argv[idx + 1])
assert '--require-command-window' in sys.argv
request = json.loads(request_path.read_text())
out.mkdir(parents=True, exist_ok=True)
duration = request['interval_count'] / request['sample_rate_hz']
energy = 5.0
(out / 'results.yaml').write_text(
    'firmware_results:\\n  energy: 5.0\\n  duration: ' + str(duration) +
    '\\n  start_stop_idx: [0, 1999]\\n  max_frame_energy: 0.0\\n  idle_frame_energy: 0.0\\n')
sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
post = {
  'schema': 'power-calculations/command-window-result', 'schema_version': 2,
  'status': 'ok', 'run_id': request['run_id'], 'window_id': request['window_id'],
  'source': request['source'], 'request_path': str(request_path),
  'marker_path': request['marker_path'], 'trace_path': request['trace_path'],
  'command_path': request['command_path'], 'timing_path': request['timing_path'],
  'results_path': str(out / 'results.yaml'), 'request_sha256': sha(request_path),
  'marker_sha256': request['marker_sha256'], 'trace_sha256': request['trace_sha256'],
  'command_sha256': request['command_sha256'], 'timing_sha256': request['timing_sha256'],
  'first_sample_index': request['first_sample_index'],
  'last_sample_index': request['last_sample_index'], 'sample_count': request['sample_count'],
  'interval_count': request['interval_count'], 'sample_rate_hz': request['sample_rate_hz'],
  'duration_s': duration, 'process_rc': request['process_rc'],
  'command_process_id': request['process_id'],
  'command_start_realtime_ns': request['command_start_realtime_ns'],
  'command_end_realtime_ns': request['command_end_realtime_ns'],
  'command_start_monotonic_ns': request['command_start_monotonic_ns'],
  'command_end_monotonic_ns': request['command_end_monotonic_ns'],
  'command_start_clock_read_uncertainty_ns': request['start_clock_read_uncertainty_ns'],
  'command_end_clock_read_uncertainty_ns': request['end_clock_read_uncertainty_ns'],
  'drop_count': request['dropped_samples'],
  'boundary_uncertainty_samples': request['boundary_uncertainty_samples'],
  'maximum_boundary_uncertainty_samples': request['maximum_boundary_uncertainty_samples'],
  'trace_covers_window': request['trace_covers_window'],
  'energy_semantics': 'calibrated_input_energy_unsubtracted',
  'energy_field': 'firmware_results.energy', 'energy_j': energy}
(out / 'postprocessor_window_result.json').write_text(json.dumps(post))
""",
    )
    defaults = EnergyDefaults(
        collector_binary=str(collector),
        power_calculations_binary=str(powercalc),
        command_startup_budget_s=startup_budget,
        sample_rate=2000,
        pre_duration_s=0,
        post_duration_s=0,
        run_count=1,
        physical_scope="MB",
        window_label="command",
    )
    setup = EnergySetup(setup_id="fake", enabled=True, urecs_address="127.0.0.1")
    command = (
        "printf '__SPLITPOINT_WORK_UNITS__=100\\n"
        "__SPLITPOINT_WORK_UNITS_SOURCE__=completed_frames\\n"
        "__SPLITPOINT_WORK_UNITS_EXACT__=1\\n'"
    )
    output = run_fast_firmware_measurement(
        command,
        tmp_path / "measurement",
        setup=setup,
        defaults=defaults,
        duration_s=1.0,
        run_count=1,
        inference_count=100,
        physical_scope="MB",
        window_label="command",
        require_runtime_work_units=strict_claim_flags,
        require_command_window_alignment=strict_claim_flags,
    )
    assert output["ok"] is True
    assert output["energy_primary_metric"] == "calibrated_input_energy_unsubtracted"
    run = output["runs"][0]
    assert run["collector_measurement_duration_s"] == startup_budget + 2
    assert run["energy_configured_workload_duration_s"] == 1.0
    assert run["power_calculations_mode"] == "command_marker_crop"
    assert run["command_window_trace_binding"]["status"] == "verified"
    assert run["energy_window_alignment_status"] == "pass"
    assert run["final_energy_gate_status"] == "pass"
    assert run["energy_per_work_unit_j"] == pytest.approx(0.05)
    assert run["energy_total_j"] == pytest.approx(5.0)
    assert run["legacy_window_comparison_status"] == "ok"
    assert run["scientific_method_decision"] == "frozen_command_marker_primary_chapter4_shadow"
    assert run["scientific_primary_method"] == "command_marker_window"
    assert run["scientific_shadow_method"] == "chapter4_legacy_window"
    assert run["scientific_primary_method_frozen"] is True
    assert run["window_method_validation_tier"] == "screening"
    run_dir = tmp_path / "measurement" / "run_000"
    assert (run_dir / "command_window_request.json").is_file()
    assert (run_dir / "command_window_binding.json").is_file()
    power_command = (run_dir / "power_calculations_command.txt").read_text(encoding="utf-8")
    assert "--command-window-request" in power_command
    assert "--require-command-window" in power_command
    assert "--estimated-duration" not in power_command
    assert " -c " not in f" {power_command} "
    legacy_power_command = (
        run_dir / "processed_legacy_window" / "power_calculations_command.txt"
    ).read_text(encoding="utf-8")
    assert " -c " in f" {legacy_power_command} "
    assert " --estimated-duration=3 " in f" {legacy_power_command} "
    comparison = json.loads((run_dir / "window_method_comparison.json").read_text())
    assert comparison["schema"] == "onnx-splitpoint/window-method-comparison"
    assert comparison["status"] == "ok"
    assert comparison["same_raw_trace_verified"] is True
    assert comparison["same_raw_trace_verification_failure_reasons"] == []
    assert comparison["raw_trace"]["collector_storage_parquet_count"] == 1
    assert comparison["raw_trace"]["collector_storage_parquet_exactly_one"] is True
    assert comparison["raw_trace"]["request_trace_matches_collector_storage_parquet"] is True
    assert comparison["raw_trace"]["request_before_after_hash_chain_complete"] is True
    assert comparison["diagnostic_only"] is True
    assert comparison["eligible_for_final_energy"] is False
    assert comparison["affects_primary_result"] is False
    assert comparison["affects_final_gate"] is False
    assert comparison["scientific_method_decision"] == "frozen_command_marker_primary_chapter4_shadow"
    assert comparison["scientific_primary_method_frozen"] is True
    assert comparison["scientific_primary_method"] == "command_marker_window"
    assert comparison["scientific_shadow_method"] == "chapter4_legacy_window"
    assert comparison["command_marker_window"]["scientific_role"] == "primary"
    assert comparison["chapter4_legacy_window"]["scientific_role"] == "shadow_only"
    assert comparison["command_window"]["energy_j"] == pytest.approx(5.0)
    assert comparison["legacy_window"]["energy_j"] == pytest.approx(4.8)
    deltas = comparison["legacy_minus_command_window"]
    assert deltas["energy_j"]["absolute"] == pytest.approx(-0.2)
    assert deltas["energy_j"]["relative_percent"] == pytest.approx(-4.0)
    assert deltas["energy_j"]["symmetric_percent"] == pytest.approx(-4.081632653)
    assert deltas["energy_j"]["ln_command_window_over_legacy"] == pytest.approx(
        math.log(5.0 / 4.8)
    )
    assert deltas["start_sample_index"]["absolute"] == 10
    assert deltas["start_sample_index"]["milliseconds"] == pytest.approx(5.0)
    assert deltas["end_sample_index"]["absolute"] == -90
    assert deltas["end_sample_index"]["milliseconds"] == pytest.approx(-45.0)
    assert comparison["command_window"]["dropped_samples"] == 0
    assert comparison["command_window"]["boundary_uncertainty_samples"] == 1
    assert comparison["calibration_contract"][
        "identical_non_window_calibration_and_filter_settings_verified"
    ] is True
    assert comparison["calibration_contract"]["same_power_calculations_binary"] is True
    assert comparison["calibration_contract"]["power_calculations_binary_sha256"] == _sha(powercalc)
    # The same-trace sensitivity aggregate uses at least three independent
    # traces when the caller requested only one run.
    assert output["legacy_window_comparison_successful_runs"] == 3
    assert output["scientific_method_decision"] == "frozen_command_marker_primary_chapter4_shadow"
    assert output["scientific_primary_method_frozen"] is True
    assert output["scientific_primary_method"] == "command_marker_window"
    assert output["scientific_shadow_method"] == "chapter4_legacy_window"
    assert output["scientific_primary_energy_total_j"] == pytest.approx(5.0)
    assert output["chapter4_legacy_shadow_energy_total_j"] == pytest.approx(4.8)
    assert output["window_method_validation_tier"] == "screening"
    assert output["legacy_minus_command_window_energy_percent_mean"] == pytest.approx(-4.0)
    method_stats = output["window_method_comparison_statistics"]
    assert method_stats["agreement_decision"] == "not_evaluated_no_frozen_tolerance"
    assert method_stats["agreement_threshold"] is None
    assert method_stats["energy_j"]["relative_percent"]["n"] == 3
    assert method_stats["energy_j"]["relative_percent"]["mean"] == pytest.approx(-4.0)
    assert method_stats["energy_j"]["symmetric_percent"]["mean"] == pytest.approx(-4.081632653)
    assert method_stats["energy_j"]["ln_command_window_over_legacy"]["mean"] == pytest.approx(
        math.log(5.0 / 4.8)
    )
    assert method_stats["start_boundary"]["milliseconds"]["mean"] == pytest.approx(5.0)
    assert method_stats["end_boundary"]["milliseconds"]["mean"] == pytest.approx(-45.0)
    marker = json.loads((run_dir / "collector_storage" / "command_window_markers.json").read_text())
    assert Path(marker["command"]["argv"][0]).resolve() == (run_dir / "energy_command.sh").resolve()

    # The separate method-validation probe owns its outer three-repeat loop.
    # Its child collector must therefore complete exactly one same-trace A/B
    # capture without being expanded back to three internal runs.
    exact_output = run_fast_firmware_measurement(
        command,
        tmp_path / "caller_managed_measurement",
        setup=setup,
        defaults=defaults,
        duration_s=1.0,
        run_count=1,
        exact_run_count=True,
        inference_count=100,
        physical_scope="MB",
        window_label="command",
        require_runtime_work_units=True,
        require_command_window_alignment=True,
    )
    assert exact_output["ok"] is True
    assert exact_output["status"] == "ok_caller_managed_repeat_capture"
    assert exact_output["run_count"] == 1
    assert len(exact_output["runs"]) == 1
    assert exact_output["legacy_window_comparison_successful_runs"] == 1
    assert exact_output["caller_managed_repeat_capture_complete"] is True
    assert exact_output["invalid_repeat_max_retries"] == 0
    assert exact_output["invalid_repeat_max_retries_requested"] == 1
    assert exact_output["invalid_repeat_retry_suppressed_by_exact_run_count"] is True
    exact_contract = exact_output["energy_window_method_ab"]
    assert exact_contract["requested_run_count"] == 1
    assert exact_contract["effective_run_count"] == 1
    assert exact_contract["repeat_control"] == "caller_managed_exact"
    assert exact_contract["smoke_repeat_contract_attested"] is False
    assert exact_contract["status"] == "caller_managed_capture_complete"

    # v2.79.14 permits a manifest-free Full-System capture only for the exact
    # diagnostic M.2 idle-calibration purpose.  It still produces ordinary raw
    # and summary files, but it never becomes a scientific claim itself.
    idle_diagnostic = run_fast_firmware_measurement(
        command,
        tmp_path / "idle_calibration_measurement",
        setup=setup,
        defaults=defaults,
        duration_s=1.0,
        run_count=1,
        exact_run_count=True,
        compare_legacy_window=False,
        inference_count=100,
        physical_scope="FS",
        window_label="command",
        require_runtime_work_units=True,
        require_command_window_alignment=True,
        diagnostic_only=True,
        claim_exclusion_reason="m2_accelerator_idle_power_calibration",
    )
    assert idle_diagnostic["ok"] is True
    assert idle_diagnostic["full_system_scope_calibration_status"] == (
        "diagnostic_idle_calibration_not_required"
    )
    assert idle_diagnostic["diagnostic_only"] is True
    assert idle_diagnostic["energy_efficiency_claim_eligible"] is False

    ordinary_full_system = run_fast_firmware_measurement(
        command,
        tmp_path / "ordinary_full_system_measurement",
        setup=setup,
        defaults=defaults,
        duration_s=1.0,
        run_count=1,
        exact_run_count=True,
        compare_legacy_window=False,
        inference_count=100,
        physical_scope="FS",
        window_label="command",
        require_runtime_work_units=True,
        require_command_window_alignment=True,
    )
    assert ordinary_full_system["ok"] is False
    assert ordinary_full_system["full_system_scope_calibration_status"] == "missing"
    assert "full_system_calibration_not_locally_verified" in (
        ordinary_full_system["final_energy_gate_failures"][0]["reasons"]
    )

    # A broken Chapter-4 reprocessing branch remains visible as missing shadow
    # evidence, but cannot invalidate or reacquire a valid marker primary.
    monkeypatch.setattr(
        "onnx_splitpoint_tool.energy.collector._run_chapter4_shadow_comparison",
        lambda *args, **kwargs: {
            "schema": "onnx-splitpoint/window-method-comparison",
            "schema_version": 2,
            "status": "legacy_window_postprocess_failed",
            "diagnostic_only": True,
            "affects_primary_result": False,
            "affects_final_gate": False,
        },
    )
    shadow_failed = run_fast_firmware_measurement(
        command,
        tmp_path / "shadow_failed_measurement",
        setup=setup,
        defaults=defaults,
        duration_s=1.0,
        run_count=1,
        inference_count=100,
        physical_scope="MB",
        window_label="command",
        require_runtime_work_units=True,
        require_command_window_alignment=True,
    )
    assert shadow_failed["ok"] is True
    assert shadow_failed["status"] == "ok_shadow_incomplete"
    assert shadow_failed["repeat_contract_complete"] is True
    assert shadow_failed["valid_postprocessed_runs"] == 3
    assert shadow_failed["scientific_primary_energy_status"] == "available"
    assert shadow_failed["scientific_primary_energy_statistics"]["energy_j"]["n"] == 3
    assert shadow_failed["scientific_shadow_energy_status"] == "unavailable"
    assert shadow_failed["scientific_shadow_affects_primary_result"] is False
    assert shadow_failed["scientific_shadow_affects_final_gate"] is False


@pytest.mark.parametrize(
    ("failure", "expected_error"),
    [
        ("process_rc", "marker_process_rc_nonzero"),
        ("drops", "marker_dropped_samples_nonzero"),
        ("uncertainty", "marker_boundary_uncertainty_exceeds_limit"),
        ("wrong_command", "marker_command_is_not_captured_command"),
        ("invalid_final_marker", "marker_not_valid_for_final_energy"),
    ],
)
def test_invalid_marker_is_rejected_before_postprocessor(
    tmp_path: Path, failure: str, expected_error: str
) -> None:
    storage = tmp_path / "storage"
    storage.mkdir()
    trace = storage / "fast_firmware.parquet"
    trace.write_bytes(b"trace" * 1000)
    workload = tmp_path / "workload.sh"
    workload.write_text("echo workload\n", encoding="utf-8")
    captured = tmp_path / "energy_command.sh"
    captured.write_text("echo captured\n", encoding="utf-8")
    timing = tmp_path / "workload_timing.txt"
    timing.write_text("start_ns=1\nend_ns=2\nrc=0\n", encoding="utf-8")
    marker_command = workload if failure == "wrong_command" else captured
    marker = {
        "schema": "urecs-data-collector/command-window-markers",
        "schema_version": 2,
        "run_id": "run",
        "window_id": "window",
        "command": {
            "text": str(marker_command),
            "argv": [str(marker_command)],
            "process_id": 123,
            "process_rc": 1 if failure == "process_rc" else 0,
            "process_error": None,
            "start_realtime_ns": 1,
            "end_realtime_ns": 2,
            "start_monotonic_ns": 1,
            "end_monotonic_ns": 2,
            "start_clock_read_uncertainty_ns": 0,
            "end_clock_read_uncertainty_ns": 0,
        },
        "stream": {
            "source": "fast_firmware",
            "trace_path": str(trace),
            "sample_rate_hz": 2000,
            "first_sample_index": 0,
            "last_sample_index": 1999,
            "total_samples": 2000,
            "boundary_uncertainty_samples": 65 if failure == "uncertainty" else 1,
            "dropped_samples": 1 if failure == "drops" else 0,
            "trace_covers_window": True,
        },
        "valid_for_final_energy": failure != "invalid_final_marker",
        "validation_errors": [],
    }
    _write_json(storage / "command_window_markers.json", marker)
    prepared = _prepare_command_window_request_v2(
        tmp_path,
        storage,
        workload_script=workload,
        captured_command_script=captured,
        timing_path=timing,
        physical_scope="MB",
        expected_run_id="run",
        expected_window_id="window",
    )
    assert prepared["eligible_for_postprocessor"] is False
    assert expected_error in prepared["errors"]
    assert not (tmp_path / "command_window_request.json").exists()


def test_legacy_collector_never_gets_command_window_powercalc_flags(tmp_path: Path) -> None:
    # The command builder itself is exercised by the end-to-end test above.  A
    # missing marker must remain the explicit legacy/fail-closed branch.
    command = tmp_path / "workload.sh"
    command.write_text("echo legacy\n", encoding="utf-8")
    timing = tmp_path / "timing.txt"
    timing.write_text("start_ns=1\nend_ns=2\nrc=0\n", encoding="utf-8")
    prepared = _prepare_command_window_request_v2(
        tmp_path,
        tmp_path / "storage",
        workload_script=command,
        captured_command_script=command,
        timing_path=timing,
        physical_scope="MB",
        expected_run_id="legacy-run",
        expected_window_id="legacy-window",
    )
    assert prepared["status"] == "marker_missing"
    assert prepared["eligible_for_postprocessor"] is False
    assert not (tmp_path / "command_window_request.json").exists()
