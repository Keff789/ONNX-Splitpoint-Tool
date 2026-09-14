#!/usr/bin/env python3
"""Prepare/execute one private, receipt-bound MobileNet Hailo10 B500 build.

prepare performs no SDK import, GPU compute or compilation. execute uses the
normal builder and shared platform interlock. A generated HEF never implies
GPU, runtime, quality or campaign acceptance. compare consumes local real raw
outputs; compact evidence export excludes all models, HARs, images and arrays.
"""
from __future__ import annotations
import argparse
import contextlib
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from onnx_splitpoint_tool.hailo_model_diagnostics_v27934 import (
    write_json, file_identity, check_identity, prepare_request, validate_request,
    builder_arguments, private_environment, attributable_gpu_evidence,
    compare_classification_arrays,
)


@contextlib.contextmanager
def interlock():
    import fcntl
    from onnx_splitpoint_tool.workflow.run_control import platform_workflow_interlock_path
    path = platform_workflow_interlock_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('diagnostic_workflow_or_platform_operation_active') from exc
        yield
    finally:
        os.close(fd)


class GpuActivity:
    """Observe pmon samples only; this does not run any GPU kernel itself."""
    def __init__(self, directory):
        self.directory = Path(directory)
        self.proc = None
        self.thread = None
        self.samples = []
        self.status = 'not_available'

    def start(self):
        binary = shutil.which('nvidia-smi')
        if not binary:
            self.status = 'nvidia_smi_not_available'
            return
        try:
            self.proc = subprocess.Popen([binary, 'pmon', '-s', 'u', '-d', '1'],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        except OSError as exc:
            self.status = 'pmon_unavailable:' + str(exc)
            return
        self.status = 'collecting'
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self):
        from deepx_full_workflow_smoke_worker_v27930 import _capture_owned
        tracked = {}
        columns = None
        try:
            with (self.directory / 'gpu_activity.log').open('w', encoding='utf-8') as log:
                for line in self.proc.stdout:
                    stamp = time.monotonic()
                    log.write(line)
                    log.flush()
                    words = line.strip().split()
                    if words[:2] == ['#', 'gpu'] and 'pid' in words and 'sm' in words:
                        columns = words[1:]
                        continue
                    if not columns or not words or words[0] == '#' or len(words) < len(columns):
                        continue
                    values = dict(zip(columns, words))
                    try:
                        pid, gpu, sm = int(values['pid']), int(values['gpu']), float(values['sm'])
                    except (KeyError, ValueError):
                        continue
                    owned = _capture_owned(os.getpid(), tracked)
                    # A fresh namespace-correct descendant snapshot excludes
                    # unrelated GUI/workstation activity and the sampler itself.
                    self.samples.append({'pid': pid, 'gpu_index': gpu,
                        'sm_utilization_percent': sm, 'observed_monotonic_s': stamp,
                        'interval_start_monotonic_s': stamp - 1.1,
                        'owned_compiler_descendant': pid in owned and pid != self.proc.pid,
                        'process_identity': tracked.get(pid)})
        except Exception as exc:
            self.status = 'pmon_collection_error:' + type(exc).__name__ + ':' + str(exc)

    def stop(self):
        if self.proc is not None:
            if self.proc.poll() is None:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    self.proc.wait(timeout=3)
            self.thread.join(timeout=3)
            if self.thread.is_alive():
                raise RuntimeError('diagnostic_gpu_sampler_did_not_stop')
            if self.status == 'collecting':
                self.status = 'collected' if self.samples else 'no_supported_process_samples'
        write_json(self.directory / 'gpu_activity.json', {'status': self.status, 'samples': self.samples})


def result_events(result):
    for area in ('details', 'calib_info'):
        events = (result.get(area) or {}).get('phase_events')
        if isinstance(events, list):
            return events
    return []


def summarize_result(result, request, work, samples):
    """A successful call is insufficient without completed actual phases."""
    events = result_events(result)
    completed = {r.get('phase') for r in events if r.get('event') == 'completed'}
    receipt = None
    hef = result.get('hef_path')
    error = None
    if hef and Path(hef).is_file():
        if not Path(hef).resolve().is_relative_to(Path(work).resolve()):
            raise ValueError('diagnostic_builder_returned_nonprivate_artifact')
        from onnx_splitpoint_tool.hailo_backend import _load_valid_hailo_receipt
        receipt = _load_valid_hailo_receipt(Path(hef), allow_diagnostic=True)
    recipe_match = False
    if receipt:
        expected = request['cpu_build_receipt']['cache_payload']
        actual = receipt['cache_payload']
        keys = ('model_sha256', 'optimization_level', 'calibration_identity',
                'prepared_calibration_identity_sha256', 'calibration_count',
                'requested_calibration_count', 'calibration_storage',
                'calibration_memory_cap_bytes', 'calibration_batch_size',
                'extra_model_script', 'start_nodes', 'end_nodes', 'integrity',
                'preprocessing_contract', 'preprocessing_contract_sha256',
                'net_name', 'net_input_shapes', 'disable_rt_metadata_extraction', 'hailo_sdk_version')
        differences = [k for k in keys if expected.get(k) != actual.get(k)]
        for key in ('compiler_onnx_sha256', 'source_onnx_sha256'):
            if receipt.get(key) != request['compiler_onnx']['sha256']:
                differences.append('private_' + key + '_not_exact_bound_compiler_graph')
        recipe_match = not differences
    else:
        differences = ['valid_private_receipt_missing']
    calibration = result.get('calib_info') or {}
    if calibration.get('source') != request['calibration_dir'] or calibration.get('used_count') != 500:
        differences.append('actual_calibration_not_bound_B500_images')
        recipe_match = False
    if result.get('skipped'):
        status = 'model_build_not_executed'
    elif result.get('ok') is True and receipt and {'translate', 'optimize', 'compile', 'publication'} <= completed:
        status = 'pass' if recipe_match else 'recipe_mismatch'
    else:
        status = 'failed' if result.get('ok') is not True else 'build_completion_unproven'
    context = (result.get('details') or {}).get('compiler_context') or (result.get('calib_info') or {}).get('compiler_context') or {}
    selected = context.get('gpu_index', context.get('physical_gpu_index'))
    if isinstance(selected, str) and selected.isdigit():
        selected = int(selected)
    gpu = attributable_gpu_evidence(samples, events, gpu_index=selected)
    if context.get('device', context.get('compute_effective')) != 'gpu':
        gpu['status'] = 'gpu_execution_unproven'
        gpu['reason'] = 'last_effective_compiler_context_not_GPU'
    hef_validation = (result.get('details') or {}).get('hef_validation') or calibration.get('hef_validation') or {}
    return {'schema': 'hailo_model_diagnostic_result_v27934',
        'model_build_status': status, 'gpu_execution_status': gpu['status'],
        'gpu_execution_evidence': gpu, 'compute_requested': 'gpu',
        'compute_effective': context.get('device', context.get('compute_effective', 'not_available')),
        'compiler_context': context, 'phase_events': events,
        'recipe_matches_cpu_baseline': recipe_match, 'recipe_differences': differences,
        'bound_original_source_onnx': request['source_onnx'],
        'bound_cpu_compiler_onnx': request['compiler_onnx'],
        'hef_receipt_status': 'validated' if receipt else 'not_available',
        'hef_readability_status': {'passed': 'pass'}.get(hef_validation.get('status'), hef_validation.get('status', 'not_run')),
        'hef_validation': hef_validation, 'runtime_status': 'not_run',
        'quality_status': 'not_evaluated', 'raw_energy_status': 'not_run',
        'claim_eligible': False, 'regular_path_released': False,
        'cache_reuse_proven': False, 'publication': 'private_only',
        'compiler_dispatch_count': 1 if 'translate' in completed else (0 if result.get('skipped') else None),
        'error': result.get('error'), 'failure_kind': result.get('failure_kind'),
        'last_stage': result.get('last_stage'), 'timed_out': result.get('timed_out', False),
        'private_hef': file_identity(hef) if receipt else None,
        'optimization_procedure_source': 'actual_compiler.log_unmodified',
        'optimization_procedure_equivalence': 'not_evaluated', 'speedup_claim': False}


def worker(request, output):
    output = Path(output).resolve()
    validate_request(request)
    env = private_environment(request, output)
    os.environ.clear()
    os.environ.update(env)  # Only this owned worker, never GUI/launcher parent.
    os.chdir(output)
    activity = GpuActivity(output)
    build = None
    activity.start()
    try:
        from onnx_splitpoint_tool.hailo_backend import hailo_build_hef_auto
        with (output / 'compiler.log').open('w', encoding='utf-8') as log:
            def on_log(*parts):
                line = ' '.join(str(part) for part in parts)
                log.write(line + '\n')
                log.flush()
                print(line, flush=True)
            build = asdict(hailo_build_hef_auto(**builder_arguments(request, output), on_log=on_log))
        write_json(output / 'build_result.json', build)
    except BaseException as exc:
        write_json(output / 'primary_error.json', {'type': type(exc).__name__, 'error': str(exc)})
        raise
    finally:
        activity.stop()
    summary = summarize_result(build, request, output, activity.samples)
    validate_request(request)
    summary['frozen_inputs_unchanged'] = True
    write_json(output / 'summary.json', summary)
    return 0 if summary['model_build_status'] == 'pass' else 2


def export_evidence(output):
    """An explicit compact allowlist: no arbitrary build tree or raw arrays."""
    output = Path(output)
    archive = output.parent / (output.name + '_evidence.zip')
    names = ('request.json', 'summary.json', 'build_result.json', 'primary_error.json',
        'supervision.json', 'gpu_activity.json', 'gpu_activity.log', 'compiler.log',
        'worker.log', 'comparison.json', 'readability.json',
        'build/hailo_build_phases.json', 'build/hailo_compiler_context.json')
    with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as target:
        for name in names:
            file = output / name
            if file.is_file() and not file.is_symlink():
                if file.stat().st_size > 32 * 1024 * 1024:
                    raise ValueError('diagnostic_compact_log_budget_exceeded:' + name)
                target.write(file, output.name + '/' + name)
        # Exact trace paths named by the actual effective compiler context;
        # never recursively export HARs, models, calibration data or arrays.
        for metadata in ('build/hailo_compiler_context.json', 'summary.json'):
            source = output / metadata
            if not source.is_file():
                continue
            record = json.loads(source.read_text())
            context = record.get('compiler_context', record)
            trace = context.get('ptxas_trace_path')
            if not trace:
                continue
            trace = Path(trace).resolve()
            if not trace.is_relative_to(output.resolve()) or not trace.is_file() or trace.stat().st_size > 8 * 1024 * 1024:
                continue
            name = output.name + '/ptxas_invocations.jsonl'
            if name not in target.namelist():
                target.write(trace, name)
    return archive


def execute(request_file, output):
    request = json.loads(Path(request_file).read_text(encoding='utf-8'))
    validate_request(request)
    output = Path(output).expanduser().absolute()
    # A preexisting or input-containing directory is never reused.
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    write_json(output / 'request.json', request)
    summary = {'model_build_status': 'not_run', 'gpu_execution_status': 'not_run',
        'runtime_status': 'not_run', 'quality_status': 'not_evaluated', 'claim_eligible': False}
    try:
        print('MODEL_BUILD_TIMEOUT_S=' + str(request['timeout_s']), flush=True)
        with interlock():
            from deepx_full_workflow_smoke_worker_v27930 import supervised_run
            # Standalone controller: supervisor never adopts pytest/GUI children.
            process = supervised_run([sys.executable, '-B', str(Path(__file__).resolve()),
                '_worker', '--request', str(output / 'request.json'), '--output-dir', str(output)],
                cwd=output, log_path=output / 'worker.log',
                timeout=request['timeout_s'] + 90, grace=10)
            write_json(output / 'supervision.json', process)
            if (output / 'summary.json').is_file():
                summary = json.loads((output / 'summary.json').read_text())
            elif (output / 'primary_error.json').is_file():
                summary['primary_error'] = json.loads((output / 'primary_error.json').read_text())
            phase_file = output / 'build/hailo_build_phases.json'
            if phase_file.is_file() and not summary.get('phase_events'):
                summary['phase_events'] = json.loads(phase_file.read_text()).get('events', [])
            if process['timed_out'] or process['cancelled'] or not process['cleanup_complete'] or process['returncode']:
                summary.update(model_build_status='failed', supervision=process)
                if not summary.get('error'):
                    summary['error'] = (summary.get('primary_error') or {}).get('error', 'supervised_model_build_failed_or_incomplete')
            else:
                summary['supervision'] = process
            validate_request(request)
            summary['frozen_inputs_unchanged'] = True
    except BaseException as exc:
        summary.update(model_build_status='failed', error=type(exc).__name__ + ': ' + str(exc))
    finally:
        summary['g3_status'] = 'incomplete_runtime_comparison_pending'
        write_json(output / 'summary.json', summary)
        print('EVIDENCE_ZIP=' + str(export_evidence(output)), flush=True)
        print(json.dumps(summary, indent=2), flush=True)
    # This command's RC reports completion of the model-build step. G3 stays
    # explicitly incomplete until the separate real fixed16 runtime collector.
    return 0 if summary.get('model_build_status') == 'pass' and (summary.get('supervision') or {}).get('cleanup_complete') is True else 2


def compare(request, manifest):
    import numpy as np
    validate_request(request)
    data = json.loads(Path(manifest).read_text(encoding='utf-8'))
    stages = {}
    for name, row in data['stages'].items():
        with np.load(check_identity(row['arrays']), allow_pickle=False) as arrays:
            if set(arrays.files) != {'logits', 'feed'}:
                raise ValueError('diagnostic_named_logits_and_feed_arrays_required')
            stages[name] = {'ids': row['ids'], 'logits': arrays['logits'], 'feed': arrays['feed'], 'origin': row.get('origin')}
    return compare_classification_arrays(stages, labels=[r['label'] for r in request['images']], ids=[r['id'] for r in request['images']])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prepare = sub.add_parser('prepare', help='validate/freeze local CPU baseline; no hardware execution')
    for name in ('cpu-hef', 'source-onnx', 'compiler-onnx', 'calibration-dir', 'images-json', 'venv', 'request-out'):
        prepare.add_argument('--' + name, required=True, type=Path)
    prepare.add_argument('--calibration-manifest', type=Path)
    prepare.add_argument('--timeout-s', type=int, required=True)
    prepare.add_argument('--gpu-selector', default='0')
    for command in ('execute', '_worker', 'validate', 'compare'):
        item = sub.add_parser(command)
        item.add_argument('--request', required=True, type=Path)
        if command in {'execute', '_worker'}:
            item.add_argument('--output-dir', required=True, type=Path)
        if command == 'compare':
            item.add_argument('--manifest', required=True, type=Path)
            item.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == 'prepare':
        request = prepare_request(cpu_hef=args.cpu_hef, source_onnx=args.source_onnx,
            compiler_onnx=args.compiler_onnx, calibration_dir=args.calibration_dir,
            images_json=args.images_json, venv=args.venv, timeout_s=args.timeout_s,
            calibration_manifest=args.calibration_manifest)
        request['gpu_selector'] = args.gpu_selector
        if args.request_out.exists():
            raise ValueError('diagnostic_request_output_already_exists')
        write_json(args.request_out, request)
        print(json.dumps({'status': 'prepared', 'model_build_status': 'not_run',
            'gpu_execution_status': 'not_run', 'request': str(args.request_out)}, indent=2))
        return 0
    request = json.loads(args.request.read_text(encoding='utf-8'))
    if args.command == 'validate':
        validate_request(request)
        print(json.dumps({'status': 'validated', 'hardware_execution': 'NOT_RUN'}))
        return 0
    if args.command == 'execute':
        return execute(args.request, args.output_dir)
    if args.command == '_worker':
        return worker(request, args.output_dir)
    report = compare(request, args.manifest)
    write_json(args.output, report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
