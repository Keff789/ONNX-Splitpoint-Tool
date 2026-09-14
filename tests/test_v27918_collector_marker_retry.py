from __future__ import annotations

import json
from pathlib import Path

import pytest

from onnx_splitpoint_tool.energy.collector import run_fast_firmware_measurement
from onnx_splitpoint_tool.energy.config import EnergyDefaults, EnergySetup


def _write_executable(path: Path, source: str) -> None:
    path.write_text(source, encoding="utf-8")
    path.chmod(0o755)


def _fake_marker_tools(tmp_path: Path) -> tuple[Path, Path]:
    collector = tmp_path / "fake-collector"
    powercalc = tmp_path / "fake-power-calculations"
    _write_executable(
        collector,
        """#!/usr/bin/env python3
import json
import os
import pathlib
import subprocess
import sys
import time

if '--help' in sys.argv:
    print('fake collector')
    raise SystemExit(0)

state = pathlib.Path(os.environ['V27918_MARKER_RETRY_STATE'])
attempt = int(state.read_text() or '0') if state.exists() else 0
state.write_text(str(attempt + 1))
mode = os.environ['V27918_MARKER_RETRY_MODE']
kind = os.environ['V27918_MARKER_RETRY_KIND']
invalid = mode == 'always' or attempt == 0

storage = pathlib.Path(next(x.split('=', 1)[1] for x in sys.argv if x.startswith('-s=')))
command = pathlib.Path(next(x.split('=', 1)[1] for x in sys.argv if x.startswith('-c=')))
run_id = sys.argv[sys.argv.index('--run-id') + 1]
window_id = sys.argv[sys.argv.index('--window-id') + 1]
storage.mkdir(parents=True, exist_ok=True)
trace = storage / 'fast_firmware.parquet'
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
              'start_clock_read_uncertainty_ns': 10,
              'end_clock_read_uncertainty_ns': 10},
  'stream': {'source': 'fast_firmware', 'trace_path': str(trace),
             'sample_rate_hz': 2000, 'first_sample_index': 0,
             'last_sample_index': 1999, 'total_samples': 2500,
             'boundary_uncertainty_samples': 1,
             'dropped_samples': 1 if invalid and kind == 'dropped' else 0,
             'trace_covers_window': not (invalid and kind == 'coverage')},
  'valid_for_final_energy': proc.returncode == 0,
  'validation_errors': []}
(storage / 'command_window_markers.json').write_text(json.dumps(marker))
raise SystemExit(proc.returncode)
""",
    )
    _write_executable(
        powercalc,
        """#!/usr/bin/env python3
import hashlib
import json
import pathlib
import sys

if '--help' in sys.argv:
    print('fake power calculations')
    raise SystemExit(0)

assert '--command-window-request' in sys.argv
assert '--require-command-window' in sys.argv
out = pathlib.Path(next(x.split('=', 1)[1] for x in sys.argv if x.startswith('--output-path=')))
request_path = pathlib.Path(sys.argv[sys.argv.index('--command-window-request') + 1])
request = json.loads(request_path.read_text())
out.mkdir(parents=True, exist_ok=True)
duration = request['interval_count'] / request['sample_rate_hz']
(out / 'results.yaml').write_text(
    'firmware_results:\\n  energy: 5.0\\n  duration: ' + str(duration) +
    '\\n  start_stop_idx: [0, 1999]\\n  max_frame_energy: 0.0' +
    '\\n  idle_frame_energy: 0.0\\n')
sha = lambda path: hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
post = {
  'schema': 'power-calculations/command-window-result', 'schema_version': 2,
  'status': 'ok', 'run_id': request['run_id'], 'window_id': request['window_id'],
  'source': request['source'], 'request_path': str(request_path),
  'marker_path': request['marker_path'], 'trace_path': request['trace_path'],
  'command_path': request['command_path'], 'timing_path': request['timing_path'],
  'results_path': str(out / 'results.yaml'), 'request_sha256': sha(request_path),
  'marker_sha256': request['marker_sha256'],
  'trace_sha256': request['trace_sha256'],
  'command_sha256': request['command_sha256'],
  'timing_sha256': request['timing_sha256'],
  'first_sample_index': request['first_sample_index'],
  'last_sample_index': request['last_sample_index'],
  'sample_count': request['sample_count'],
  'interval_count': request['interval_count'],
  'sample_rate_hz': request['sample_rate_hz'], 'duration_s': duration,
  'process_rc': request['process_rc'], 'command_process_id': request['process_id'],
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
  'energy_field': 'firmware_results.energy', 'energy_j': 5.0}
(out / 'postprocessor_window_result.json').write_text(json.dumps(post))
""",
    )
    return collector, powercalc


def _run_marker_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    failure_kind: str,
    failure_mode: str,
) -> tuple[dict, Path]:
    collector, powercalc = _fake_marker_tools(tmp_path)
    state = tmp_path / "collector-attempt-count.txt"
    monkeypatch.setenv("V27918_MARKER_RETRY_STATE", str(state))
    monkeypatch.setenv("V27918_MARKER_RETRY_KIND", failure_kind)
    monkeypatch.setenv("V27918_MARKER_RETRY_MODE", failure_mode)
    defaults = EnergyDefaults(
        collector_binary=str(collector),
        power_calculations_binary=str(powercalc),
        sample_rate=2000,
        pre_duration_s=0,
        post_duration_s=0,
        run_count=1,
        compare_legacy_window=False,
        invalid_repeat_max_retries=1,
        invalid_repeat_reconnect_backoff_s=0,
        physical_scope="MB",
        window_label="command",
    )
    result = run_fast_firmware_measurement(
        "printf '__SPLITPOINT_WORK_UNITS__=1\\n"
        "__SPLITPOINT_WORK_UNITS_SOURCE__=completed_frames\\n"
        "__SPLITPOINT_WORK_UNITS_EXACT__=1\\n'",
        tmp_path / "measurement",
        setup=EnergySetup(
            setup_id="v27918-marker-retry", enabled=True,
            urecs_address="127.0.0.1",
        ),
        defaults=defaults,
        duration_s=0.01,
        run_count=1,
        exact_run_count=False,
        postprocess=True,
        compare_legacy_window=False,
        inference_count=1,
        physical_scope="MB",
        window_label="command",
        require_runtime_work_units=True,
        require_command_window_alignment=True,
        invalid_repeat_max_retries=1,
    )
    assert state.read_text(encoding="utf-8") == "2"
    return result, tmp_path / "measurement"


@pytest.mark.parametrize(
    ("failure_kind", "expected_reason"),
    [
        ("dropped", "marker_dropped_samples_nonzero"),
        ("coverage", "marker_trace_does_not_cover_window"),
    ],
)
def test_marker_acquisition_error_gets_one_fresh_retry_and_passes_only_after_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_kind: str,
    expected_reason: str,
) -> None:
    result, measurement = _run_marker_retry(
        tmp_path,
        monkeypatch,
        failure_kind=failure_kind,
        failure_mode="first",
    )

    assert result["ok"] is True
    assert result["status"] == "ok"
    assert result["invalid_repeat_max_retries"] == 1
    assert result["invalid_repeat_max_retries_requested"] == 1
    assert result["invalid_repeat_retry_suppressed_by_exact_run_count"] is False
    assert result["repeat_retry_attempt_count"] == 1
    assert result["repeat_retry_recovered_count"] == 1
    assert result["final_energy_gate_status"] == "pass"

    history = result["repeat_retry_history"][0]
    assert expected_reason in history["initial_retry_reasons"]
    assert history["selected_attempt_index"] == 1
    assert history["recovered"] is True
    assert len(history["attempts"]) == 2
    initial, retry = history["attempts"]
    assert initial["selected"] is False
    assert expected_reason in initial["retry_reasons"]
    assert retry["selected"] is True
    assert retry["accepted_for_logical_repeat"] is True
    assert retry["final_energy_gate_status"] == "pass"
    assert Path(initial["run_directory"]).resolve() != Path(
        retry["run_directory"]
    ).resolve()
    assert (measurement / "run_000/collector_storage/fast_firmware.parquet").is_file()
    assert (
        measurement
        / "repeat_retry_attempts/repeat_000/attempt_01/run_000"
        / "collector_storage/fast_firmware.parquet"
    ).is_file()
    failed = json.loads(
        (measurement / "run_000/energy_summary.json").read_text(encoding="utf-8")
    )
    assert expected_reason in failed["command_window_request"]["errors"]
    assert failed["final_energy_gate_status"] == "fail"
    selected = result["runs"][0]
    assert selected["selected_repeat_attempt_index"] == 1
    assert selected["final_energy_gate_status"] == "pass"


def test_persistent_marker_acquisition_error_is_not_accepted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, _measurement = _run_marker_retry(
        tmp_path,
        monkeypatch,
        failure_kind="dropped",
        failure_mode="always",
    )

    assert result["ok"] is False
    assert result["status"] == "acquisition_integrity_failed_no_valid_runs"
    assert result["repeat_retry_attempt_count"] == 1
    assert result["repeat_retry_recovered_count"] == 0
    assert result["final_energy_gate_status"] == "fail"
    assert "marker_dropped_samples_nonzero" in (
        result["acquisition_integrity_failure_reasons"]
    )

    history = result["repeat_retry_history"][0]
    assert "marker_dropped_samples_nonzero" in history["initial_retry_reasons"]
    assert history["recovered"] is False
    assert len(history["attempts"]) == 2
    assert history["attempts"][0]["selected"] is False
    retry = history["attempts"][1]
    assert retry["selected"] is False
    assert retry["accepted_for_logical_repeat"] is False
    assert retry["final_energy_gate_status"] == "fail"
    assert "marker_dropped_samples_nonzero" in retry["retry_reasons"]
    assert result["runs"][0]["repeat_retry_recovered"] is False
    assert result["runs"][0]["final_energy_gate_status"] == "fail"
