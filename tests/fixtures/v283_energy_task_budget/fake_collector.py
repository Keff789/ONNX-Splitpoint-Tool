#!/usr/bin/env python3
import json, pathlib, subprocess, sys, time, os
if '--help' in sys.argv:
    print('fake collector')
    raise SystemExit(0)
events = pathlib.Path(os.environ['R3_FAKE_EVENTS'])
scenario = json.loads(pathlib.Path(os.environ['R3_FAKE_SCENARIO']).read_text())
old = [json.loads(line) for line in events.read_text().splitlines()] if events.exists() else []
index = sum(e['kind'] == 'collector' for e in old)
mode = scenario[min(index, len(scenario)-1)]
with events.open('a') as f: f.write(json.dumps({'kind': 'collector', 'index': index, 'mode': mode, 'pid': os.getpid()}) + '\n')
if mode in ('first_sample', 'closed'):
    print('Error: fast-firmware first-sample barrier failed: channel is empty and sending half is closed' if mode == 'first_sample' else 'channel is empty and sending half is closed', file=sys.stderr)
    raise SystemExit(1)
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
if mode == 'drop':
    marker['stream']['dropped_samples'] = 1
if mode == 'coverage':
    marker['stream']['trace_covers_window'] = False
(storage / 'command_window_markers.json').write_text(json.dumps(marker))
raise SystemExit(proc.returncode)
