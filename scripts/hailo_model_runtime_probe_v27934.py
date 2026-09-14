#!/usr/bin/env python3
"""Run the private CPU/GPU HEF pair on one configured Hailo10 setup, 16 IDs.

Uses existing SSH/SCP process supervision and production Hailo input/runtime
functions. Models, images and raw arrays remain in private local staging;
only compact results/logs are exported. No compiler or energy measurement.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from onnx_splitpoint_tool.hailo_model_diagnostics_v27934 import (
    check_identity, file_identity, validate_request, write_json, compare_classification_arrays,
)
from hailo_model_build_probe_v27934 import interlock
from deepx_full_workflow_smoke_v27930 import run_transport
import deepx_full_output_probe_v27930 as transport


def hardware_setup(registry, setup_id):
    # Read through the existing parser, without its optional migration writer.
    from onnx_splitpoint_tool.energy.config import _read_hardware_registry_unlocked, resolve_jetson_identity
    data, _ = _read_hardware_registry_unlocked(Path(registry).expanduser())
    setups = data.get('hardware_setups', [])
    rows = [row for row in setups if row.get('id') == setup_id]
    if len(rows) != 1:
        raise ValueError('diagnostic_hardware_setup_not_unique')
    setup = rows[0]
    accelerator = str(setup.get('accelerator') or setup.get('backend') or '').lower()
    if accelerator not in {'hailo10', 'hailo10h'}:
        raise ValueError('diagnostic_selected_setup_is_not_hailo10')
    identity = resolve_jetson_identity(setup)
    if identity.get('valid') is not True:
        raise ValueError('diagnostic_selected_setup_ssh_identity_invalid')
    if identity.get('ssh_extra_args'):
        raise ValueError('diagnostic_explicit_setup_ssh_extra_args_not_supported_by_bounded_transport')
    return {'host': identity['address'], 'user': identity['user'], 'port': identity['port']}


def onnx_contract(path):
    import onnx
    model = onnx.load(str(path), load_external_data=False)
    initializers = {item.name for item in model.graph.initializer}
    inputs = [item for item in model.graph.input if item.name not in initializers]
    outputs = list(model.graph.output)
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError('diagnostic_single_input_and_classification_output_required')
    def shape(value):
        return [int(dim.dim_value) if dim.dim_value > 0 else None for dim in value.type.tensor_type.shape.dim]
    source, output = shape(inputs[0]), shape(outputs[0])
    if len(source) != 4 or source[0] not in (None, 1) or source[1] != 3 or any(type(v) is not int or v <= 0 for v in source[1:]):
        raise ValueError('diagnostic_explicit_NCHW_image_contract_required')
    if len(output) != 2 or output[0] not in (None, 1) or type(output[1]) is not int or output[1] < 5:
        raise ValueError('diagnostic_explicit_NC_classification_contract_required')
    return {'input_name': inputs[0].name, 'input_shape': [1] + source[1:],
            'output_name': outputs[0].name, 'class_count': output[1]}


def prepare_runtime_request(build_directory, setup_id):
    build_directory = Path(build_directory).resolve()
    request = json.loads((build_directory / 'request.json').read_text())
    validate_request(request)
    summary = json.loads((build_directory / 'summary.json').read_text())
    if summary.get('model_build_status') != 'pass' or summary.get('recipe_matches_cpu_baseline') is not True:
        raise ValueError('diagnostic_requires_completed_matching_private_model_build')
    gpu = summary.get('private_hef')
    if not gpu:
        raise ValueError('diagnostic_private_gpu_hef_missing')
    check_identity(gpu)
    if not Path(gpu['path']).resolve().is_relative_to(build_directory):
        raise ValueError('diagnostic_gpu_hef_not_inside_original_private_build')
    from onnx_splitpoint_tool.hailo_backend import _load_valid_hailo_receipt
    receipt = _load_valid_hailo_receipt(Path(gpu['path']), allow_diagnostic=True)
    if not receipt or receipt.get('diagnostic_only') is not True:
        raise ValueError('diagnostic_private_gpu_receipt_required')
    contract = onnx_contract(check_identity(request['compiler_onnx']))
    original = onnx_contract(check_identity(request['source_onnx']))
    if contract['input_shape'] != original['input_shape'] or contract['class_count'] != original['class_count']:
        raise ValueError('diagnostic_original_and_build_graph_io_differ_requires_explicit_transition_diagnosis')
    runtime = {'schema': 'hailo_model_runtime_request_v27934', 'model': request['model'],
        'family': request['family'], 'setup_id': setup_id, **contract,
        'cpu_hef': {**request['cpu_hef'], 'path': 'cpu.hef'},
        'gpu_hef': {**gpu, 'path': 'gpu.hef'},
        'images': [{**row, 'path': f'images/{index:03d}' + Path(row['path']).suffix} for index, row in enumerate(request['images'])],
        'preprocessing_contract': request['cpu_build_receipt']['preprocessing_contract'],
        'expected_source_onnx_sha256': request['source_onnx']['sha256'],
        'expected_compiler_sha256': request['compiler_onnx']['sha256'],
        'model_build_summary': file_identity(build_directory / 'summary.json'),
        'images_usage': 'opened_development_diagnostic_not_holdout',
        'compiler_invoked': False, 'energy_invoked': False}
    return request, summary, runtime


def prepare_stage(destination, request, runtime, summary):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    for directory in ('scripts', 'onnx_splitpoint_tool'):
        for source in (ROOT / directory).rglob('*.py'):
            if '__pycache__' in source.parts:
                continue
            target = destination / 'tool' / source.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    shutil.copyfile(check_identity(request['cpu_hef']), destination / 'cpu.hef')
    shutil.copyfile(check_identity(summary['private_hef']), destination / 'gpu.hef')
    for before, staged in zip(request['images'], runtime['images']):
        target = destination / staged['path']
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(check_identity(before), target)
    write_json(destination / 'request.json', runtime)
    return destination


def float_comparison(request, runtime, results):
    import numpy as np
    import onnxruntime as ort
    ort.disable_telemetry_events()
    results = Path(results)
    result = json.loads((results / 'runtime_result.json').read_text())
    data_file = results / 'runtime_arrays.npz'
    actual_identity = file_identity(data_file)
    declared = result.get('arrays') or {}
    if (actual_identity['sha256'] != declared.get('sha256')
            or actual_identity['size_bytes'] != declared.get('size_bytes')
            or actual_identity['size_bytes'] > 128 * 1024 * 1024):
        raise ValueError('diagnostic_runtime_arrays_changed_or_over_budget')
    with np.load(data_file, allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    if (result.get('runtime_status') != 'pass' or result.get('model') != runtime['model']
            or result.get('setup_id') != runtime['setup_id'] or result.get('family') != 'hailo10h'):
        raise ValueError('diagnostic_runtime_capture_incomplete_or_wrong_scope')
    ids = [row['id'] for row in request['images']]
    if (result.get('expected_source_onnx_sha256') != runtime['expected_source_onnx_sha256']
            or result.get('expected_compiler_sha256') != runtime['expected_compiler_sha256']
            or result.get('logical_feed_equal') is not True
            or result.get('compiler_invoked') is not False
            or result.get('frozen_inputs_unchanged') is not True):
        raise ValueError('diagnostic_runtime_capture_binding_incomplete')
    for role in ('cpu_hef', 'gpu_hef'):
        stage_result = result['stages'][role]
        if stage_result.get('runtime_status') != 'pass' or stage_result.get('hef_readability_status') != 'pass':
            raise ValueError('diagnostic_runtime_stage_or_HEF_readability_incomplete:' + role)
        if stage_result.get('ids') != ids or stage_result.get('inference_count') != 16:
            raise ValueError('diagnostic_runtime_capture_wrong_image_ids')
        if stage_result['artifact_sha256'] != runtime[role]['sha256']:
            raise ValueError('diagnostic_runtime_capture_wrong_artifact:' + role)
    feed = arrays['cpu_hef_logical_feed']
    gpu_feed = arrays['gpu_hef_logical_feed']
    if feed.dtype != gpu_feed.dtype or not np.array_equal(feed, gpu_feed):
        raise ValueError('diagnostic_actual_cpu_gpu_logical_feeds_differ')
    shape = runtime['input_shape']
    if feed.shape != (16, shape[2], shape[3], shape[1]) or feed.dtype != np.float32:
        raise ValueError('diagnostic_runtime_logical_feed_contract_changed')
    stages = {}
    for role in ('cpu_hef', 'gpu_hef'):
        stages[role] = {'ids': ids, 'feed': feed, 'logits': arrays[role + '_logits'],
            'origin': {'artifact_sha256': runtime[role]['sha256'], 'setup_id': runtime['setup_id'],
                       'collector': 'existing_product_HailoBackend_fixed16_unmeasured'}}
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    for role, field in (('source_float', 'source_onnx'), ('build_float', 'compiler_onnx')):
        path = check_identity(request[field])
        contract = onnx_contract(path)
        session = ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
        outputs = []
        for image_feed in feed:
            nchw = np.ascontiguousarray(image_feed.transpose(2, 0, 1)[None])
            value = session.run([contract['output_name']], {contract['input_name']: nchw})[0]
            if value.shape != (1, runtime['class_count']):
                raise ValueError('diagnostic_float_output_axis_changed')
            outputs.append(value[0])
        logits = np.stack(outputs)
        arrays[role + '_logits'] = logits
        stages[role] = {'ids': ids, 'feed': feed, 'logits': logits,
            'origin': {'artifact_sha256': request[field]['sha256'], 'provider': 'CPUExecutionProvider',
                       'feed': 'same_actual_product_logical_feed_before_HEF_quantization'}}
    comparison = compare_classification_arrays(stages, labels=[row['label'] for row in request['images']], ids=ids)
    comparison.update(runtime_status='pass',
        hef_readability_status={role: result['stages'][role].get('hef_readability_status', 'not_available') for role in ('cpu_hef', 'gpu_hef')},
        actual_runtime_feed_status='logical_feeds_identical',
        actual_quantized_feed_equal=result.get('actual_quantized_feed_equal'),
        actual_quantized_feed_comparison_note=result.get('actual_quantized_feed_comparison_note'),
        model_build_status='separate_original_build_summary', gpu_execution_status='separate_original_build_summary')
    # Local-only output, never included in the compact evidence ZIP.
    np.savez_compressed(Path(results) / 'paired_local_arrays.npz', **arrays)
    return comparison


REMOTE_STAGING_PREFIX = '/tmp/onnx-v27934-hailo-model-'
COMPACT_EVIDENCE_FILES = (
    'plan.json', 'runtime_request.json', 'comparison.json', 'transport.log',
    'results/runtime_result.json', 'results/supervision.json', 'results/runtime.log', 'results/error.json',
)


def _owned_staging_path(path):
    return isinstance(path, str) and re.fullmatch(re.escape(REMOTE_STAGING_PREFIX) + r'[A-Za-z0-9]{10}', path) is not None


def _remember_error(report, exc, phase):
    # A cleanup, reporting or transport follow-up must not replace the cause.
    report.setdefault('error', type(exc).__name__ + ': ' + str(exc))
    report.setdefault('error_phase', phase)
    report['runtime_status'] = 'failed'


def _local_result(path):
    path = Path(path)
    if path.parent.is_symlink() or path.is_symlink() or not path.is_file() or path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError('diagnostic_local_result_missing_or_not_regular:' + path.name)
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError('diagnostic_local_result_not_object:' + path.name)
    return value


def _collect_process_evidence(output, report):
    """Only inspect the fresh results directory collected for this invocation."""
    try:
        result = _local_result(output / 'results/runtime_result.json')
    except (OSError, ValueError):
        result = {}
    detail = result.get('error')
    if result.get('runtime_status') == 'failed' and isinstance(detail, str) and detail:
        report['remote_runtime_error'] = detail
        if not report.get('error') or report['error'].startswith('ValueError: diagnostic_runtime_supervision_failed'):
            report.update(error=detail, error_phase='runtime', runtime_status='failed')
    try:
        supervision = _local_result(output / 'results/supervision.json')
        value = supervision.get('cleanup_complete')
        report['remote_process_cleanup_complete'] = value if type(value) is bool else None
        consistent = not supervision.get('owned_survivors') and not supervision.get('lingering_children_after_cli')
        if value is not True or not consistent:
            raise ValueError('diagnostic_runtime_supervision_failed:process_cleanup_unproven')
        return True
    except (OSError, ValueError) as exc:
        _remember_error(report, exc, 'collect_results')
        return False


def _cleanup_remote_staging(remote, report, log):
    status = report['remote_staging_cleanup']
    status.update(attempted=True, status='unknown')
    path = report['remote_staging_path']
    # Same owned target, same bounded SSH call, including dangling symlinks.
    command = (shlex.join(['rm', '-rf', '--', path])
               + ' && ' + shlex.join(['test', '!', '-e', path])
               + ' && ' + shlex.join(['test', '!', '-L', path]))
    try:
        result = run_transport(transport.ssh_base(remote) + [command], timeout=30,
            stdout=log, stderr=subprocess.STDOUT)
        status['returncode'] = result.returncode
        if result.returncode:
            status['status'] = 'unknown' if result.returncode == 255 else 'failed'
            raise subprocess.CalledProcessError(result.returncode, command)
        status['status'] = 'pass'
        return False
    except (Exception, KeyboardInterrupt) as exc:
        if isinstance(exc, subprocess.TimeoutExpired):
            status.update(status='unknown', timed_out=True, returncode=None)
        elif isinstance(exc, subprocess.CalledProcessError):
            status.update(status='unknown' if exc.returncode == 255 else 'failed', returncode=exc.returncode)
        else:
            status.update(status='unknown', returncode=None)
        status['error'] = type(exc).__name__ + ': ' + str(exc)
        _remember_error(report, exc, 'remote_staging_cleanup')
        return isinstance(exc, KeyboardInterrupt)


def _finish_evidence(output, report, interrupted):
    report['remote_cleanup_complete'] = (
        report['remote_process_cleanup_complete'] is True
        and report['remote_staging_cleanup']['status'] == 'pass')
    if not report['remote_cleanup_complete'] and not report.get('error'):
        _remember_error(report, ValueError('diagnostic_remote_cleanup_incomplete'), 'remote_staging_cleanup')
    report['g3_status'] = ('pass' if report.get('model_build_status') == 'pass'
        and report.get('gpu_execution_status') == 'pass' and report.get('runtime_status') == 'pass'
        and report['remote_cleanup_complete'] and not interrupted
        and all((report.get('stages') or {}).get(stage, {}).get('finite') is True
                for stage in ('source_float', 'build_float', 'cpu_hef', 'gpu_hef')) else 'incomplete')
    archive = output.parent / (output.name + '_evidence.zip')
    phase = 'comparison_report'
    temporary_archive = None
    archive_created = False
    try:
        write_json(output / 'comparison.json', report)
        phase = 'evidence_zip'
        # A failed ZIP write must never leave a final-looking archive whose
        # comparison still claims PASS. Publish only the fully closed ZIP,
        # without overwriting an archive belonging to an earlier invocation.
        fd, name = tempfile.mkstemp(prefix='.' + archive.name + '.', suffix='.tmp', dir=archive.parent)
        os.close(fd)
        temporary_archive = Path(name)
        with zipfile.ZipFile(temporary_archive, 'w', zipfile.ZIP_DEFLATED) as target:
            for name in COMPACT_EVIDENCE_FILES:
                path = output / name
                if not path.parent.is_symlink() and path.is_file() and not path.is_symlink() and path.stat().st_size <= 16 * 1024 * 1024:
                    target.write(path, output.name + '/' + name)
        os.link(temporary_archive, archive)
        archive_created = True
    except (Exception, KeyboardInterrupt) as exc:
        interrupted = interrupted or isinstance(exc, KeyboardInterrupt)
        detail = type(exc).__name__ + ': ' + str(exc)
        report.setdefault('evidence_errors', []).append({'phase': phase, 'error': detail})
        _remember_error(report, exc, phase)
        report['g3_status'] = 'incomplete'
        print('EVIDENCE_WRITE_FAILED=' + phase + ':' + detail, file=sys.stderr)
        try:
            write_json(output / 'comparison.json', report)
        except (Exception, KeyboardInterrupt) as write_exc:
            interrupted = interrupted or isinstance(write_exc, KeyboardInterrupt)
            print('COMPARISON_WRITE_FAILED=' + str(write_exc), file=sys.stderr)
    finally:
        if temporary_archive is not None:
            try:
                temporary_archive.unlink(missing_ok=True)
            except OSError as cleanup_exc:
                print('EVIDENCE_TEMP_CLEANUP_FAILED=' + str(cleanup_exc), file=sys.stderr)
    if archive_created:
        print('EVIDENCE_ZIP=' + str(archive))
    print(json.dumps(report, indent=2))
    return 130 if interrupted else (0 if report.get('runtime_status') == 'pass' and report['remote_cleanup_complete'] else 2)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir', type=Path, required=True)
    parser.add_argument('--hardware-registry', type=Path, required=True)
    parser.add_argument('--setup-id', required=True)
    parser.add_argument('--remote-python', required=True)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args(argv)
    try:
        if not args.remote_python.startswith('/') or '\n' in args.remote_python:
            raise ValueError('diagnostic_explicit_absolute_remote_python_required')
        registry = file_identity(args.hardware_registry)
        remote = hardware_setup(args.hardware_registry, args.setup_id)
        request, build_summary, runtime = prepare_runtime_request(args.build_dir, args.setup_id)
    except (Exception, KeyboardInterrupt) as exc:
        print('RUNTIME_DIAGNOSTIC_NOT_STARTED=' + type(exc).__name__ + ': ' + str(exc), file=sys.stderr)
        return 130 if isinstance(exc, KeyboardInterrupt) else 2
    plan = {'model': runtime['model'], 'setup_id': args.setup_id, 'remote_host': remote['host'],
        'remote_python': args.remote_python, 'fixed_image_ids': [row['id'] for row in request['images']],
        'model_build_status': build_summary['model_build_status'],
        'gpu_execution_status': build_summary.get('gpu_execution_status', 'not_available'),
        'runtime_status': 'not_run', 'quality_status': 'not_evaluated', 'claim_eligible': False,
        'compiler_invoked': False, 'energy_invoked': False}
    if args.plan_only:
        print(json.dumps(plan, indent=2))
        return 0
    if args.output_dir is None:
        parser.error('--output-dir is required for execution')
    output = args.output_dir.expanduser().absolute()
    try:
        output.mkdir(parents=True, exist_ok=False, mode=0o700)
    except (Exception, KeyboardInterrupt) as exc:
        print('RUNTIME_DIAGNOSTIC_NO_OWN_OUTPUT=' + type(exc).__name__ + ': ' + str(exc), file=sys.stderr)
        return 130 if isinstance(exc, KeyboardInterrupt) else 2
    destination = remote['user'] + '@' + remote['host']
    report = dict(plan)
    report.update(remote_process_cleanup_complete=None, remote_staging_path=None,
        remote_staging_cleanup={'status': 'not_created', 'attempted': False, 'returncode': None,
                                'timed_out': False, 'error': None})
    interrupted = False
    phase = 'prepare_stage'
    try:
        write_json(output / 'plan.json', plan)
        write_json(output / 'runtime_request.json', runtime)
        with interlock(), tempfile.TemporaryDirectory(prefix='hailo34_runtime_', dir=output) as local:
            stage = prepare_stage(Path(local) / 'stage', request, runtime, build_summary)
            with (output / 'transport.log').open('w', encoding='utf-8') as log:
                collection_started = False
                process_proven = False
                try:
                    phase = 'remote_staging_create'
                    made = run_transport(transport.ssh_base(remote) + ['mktemp -d ' + REMOTE_STAGING_PREFIX + 'XXXXXXXXXX'],
                        timeout=30, check=True, capture_output=True, text=True)
                    candidate = made.stdout.strip()
                    if made.returncode != 0 or not _owned_staging_path(candidate):
                        raise ValueError('diagnostic_remote_staging_directory_invalid')
                    remote_dir = candidate
                    report['remote_staging_path'] = remote_dir
                    report['remote_staging_cleanup']['status'] = 'not_attempted'
                    phase = 'upload_stage'
                    run_transport(transport.scp_base(remote) + ['-r', *map(str, stage.iterdir()), destination + ':' + remote_dir + '/'],
                        timeout=300, check=True, stdout=log, stderr=subprocess.STDOUT)
                    command = shlex.join(['timeout', '--signal=TERM', '--kill-after=10s', '360s',
                        'env', 'PYTHONNOUSERSITE=1', 'PYTHONDONTWRITEBYTECODE=1', 'ORT_DISABLE_TELEMETRY=1',
                        args.remote_python, '-B', remote_dir + '/tool/scripts/hailo_model_runtime_worker_v27934.py',
                        '--stage', remote_dir])
                    phase = 'runtime'
                    result = run_transport(transport.ssh_base(remote) + [command], timeout=390, stdout=log, stderr=subprocess.STDOUT)
                    if result.returncode:
                        _remember_error(report, ValueError('diagnostic_runtime_supervision_failed:returncode=' + str(result.returncode)), phase)
                    phase = 'collect_results'
                    collection_started = True
                    run_transport(transport.scp_base(remote) + ['-r', destination + ':' + remote_dir + '/results', str(output / 'results')],
                        timeout=120, check=True, stdout=log, stderr=subprocess.STDOUT)
                    process_proven = _collect_process_evidence(output, report)
                    if not report.get('error') and process_proven:
                        phase = 'float_comparison'
                        report.update(float_comparison(request, runtime, output / 'results'))
                        report['model_build_status'] = build_summary['model_build_status']
                        report['gpu_execution_status'] = build_summary.get('gpu_execution_status', 'not_available')
                        phase = 'input_binding'
                        check_identity(registry)
                        validate_request(request)
                        report['frozen_inputs_and_registry_unchanged'] = True
                except (Exception, KeyboardInterrupt) as exc:
                    interrupted = isinstance(exc, KeyboardInterrupt)
                    _remember_error(report, exc, phase)
                finally:
                    # Partial result transfer may already include the owned
                    # worker's bounded supervision. Never infer it from SSH RC.
                    if collection_started:
                        process_proven = _collect_process_evidence(output, report)
                    if _owned_staging_path(report['remote_staging_path']):
                        if process_proven:
                            interrupted = _cleanup_remote_staging(remote, report, log) or interrupted
                        else:
                            report['remote_staging_cleanup']['status'] = 'blocked_process_cleanup_unproven'
    except (Exception, KeyboardInterrupt) as exc:
        interrupted = interrupted or isinstance(exc, KeyboardInterrupt)
        _remember_error(report, exc, phase)
    return _finish_evidence(output, report, interrupted)


if __name__ == '__main__':
    raise SystemExit(main())
