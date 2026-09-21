#!/usr/bin/env python3
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
        'firmware_results:\n  energy: 4.8\n  duration: 0.95' +
        '\n  start_stop_idx: [10, 1909]\n  max_frame_energy: 0.0\n  idle_frame_energy: 0.0\n')
    raise SystemExit(0)
idx = sys.argv.index('--command-window-request')
request_path = pathlib.Path(sys.argv[idx + 1])
assert '--require-command-window' in sys.argv
request = json.loads(request_path.read_text())
out.mkdir(parents=True, exist_ok=True)
duration = request['interval_count'] / request['sample_rate_hz']
energy = 5.0
(out / 'results.yaml').write_text(
    'firmware_results:\n  energy: 5.0\n  duration: ' + str(duration) +
    '\n  start_stop_idx: [0, 1999]\n  max_frame_energy: 0.0\n  idle_frame_energy: 0.0\n')
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
