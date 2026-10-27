"""Exact Generic completed-task energy contracts and preflight, with no inference."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from .task_completion import normalize_stage1_execution

SCHEMA = 'onnx-splitpoint/generic-completed-task-energy/v1'
PREFLIGHT_SCHEMA = 'onnx-splitpoint/energy-preflight-attestation'
NONCE_TOKEN = '__ONNX_SPLITPOINT_PREFLIGHT_NONCE__'
REPEAT_TOKEN = '__ONNX_SPLITPOINT_PREFLIGHT_REPEAT__'
ATTESTATION_TOKEN = '__ONNX_SPLITPOINT_PREFLIGHT_ATTESTATION__'
COMPLETION_MARKER = '__SPLITPOINT_GENERIC_ENERGY_COMPLETION__='
OPERATION_FIELDS = ('identity', 'task', 'input_binding', 'completion_contract',
                    'expected_runtime_artifacts', 'functional_reference', 'original_run_cfg', 'queue_depth',
                    'execution_environment')


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def operation_binding(config):
    value = {k: config[k] for k in OPERATION_FIELDS if k in config}
    value['queue_depth'] = config.get('queue_depth', 2)
    value['handoff_payload_copy'] = (config['original_run_cfg'].get('streaming_impl', 'auto') == 'generic')
    # Preserve old synchronous command hashes, while sealing any explicit policy.
    # An absent field always denotes synchronous execution with one P1 job.
    execution = normalize_stage1_execution(config.get('stage1_execution'))
    original = config['original_run_cfg']
    provider = str(original.get('stage1_provider') or original.get('provider') or '').strip().lower()
    if execution['policy'] == 'hailo_async_slots_v1' and provider not in {'hailo10', 'hailo10h'}:
        raise ValueError('Generic async energy requires the bound Hailo10H provider')
    if execution['policy'] == 'deepx_async_slots_v1' and provider != 'deepx_m1':
        raise ValueError('Generic async energy requires the bound DeepX M1 provider')
    if 'stage1_execution' in config:
        value['stage1_execution'] = execution
    return value


def execution_environment(python_executable, environment):
    names = {'PATH', 'LD_LIBRARY_PATH', 'LD_PRELOAD', 'PYTHONPATH', 'PYTHONHOME', 'NUMEXPR_NUM_THREADS',
             'SPLITPOINT_EXTRA_SITES', 'OMP_NUM_THREADS', 'OMP_THREAD_LIMIT', 'OMP_DYNAMIC', 'OMP_PROC_BIND',
             'OMP_PLACES', 'OMP_SCHEDULE', 'MKL_NUM_THREADS', 'MKL_DYNAMIC', 'OPENBLAS_NUM_THREADS',
             'GOTO_NUM_THREADS', 'CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER', 'CUDA_MODULE_LOADING',
             'CUDA_LAUNCH_BLOCKING', 'CUDA_MPS_ACTIVE_THREAD_PERCENTAGE', 'CUDA_MPS_PIPE_DIRECTORY',
             'ORT_INTRA_OP_NUM_THREADS', 'ORT_INTER_OP_NUM_THREADS', 'TF_NUM_INTRAOP_THREADS',
             'TF_NUM_INTEROP_THREADS', 'KMP_AFFINITY', 'KMP_BLOCKTIME', 'TRTEXEC',
             'ONNX_SPLITPOINT_TRT_RUNTIME', 'ONNX_SPLITPOINT_NATIVE_TRT_PRECISION',
             'ONNX_SPLITPOINT_BOUNDARY_MODE', 'ONNX_SPLITPOINT_STREAMING_IMPL',
             'ONNX_SPLITPOINT_HAILO_ACTIVATION_MODE', 'ONNX_SPLITPOINT_HAILO_QUANTIZED_IMAGE_INPUTS',
             'SPLITPOINT_HAILO_QUANTIZED_IMAGE_INPUTS'}
    return {'python_executable': python_executable,
            'environment': {k: v for k, v in environment.items() if k in names}}


def runtime_execution_identity(config):
    declared = config.get('execution_environment') or {}
    requested = declared.get('python_executable')
    executable = str(Path(sys.executable).resolve())
    if requested and str(Path(requested).absolute()) != str(Path(sys.executable).absolute()):
        raise ValueError('Generic runtime Python differs from declared execution environment')
    explicit = declared.get('environment') or {}
    if any(os.environ.get(k) != v for k, v in explicit.items()):
        raise ValueError('Generic runtime environment differs from declared execution environment')
    names = set(explicit) | {'PATH', 'LD_LIBRARY_PATH', 'LD_PRELOAD', 'PYTHONPATH', 'PYTHONHOME', 'NUMEXPR_NUM_THREADS', 'SPLITPOINT_EXTRA_SITES'}
    names.update(execution_environment(sys.executable, os.environ)['environment'])
    return {'python_executable': str(Path(sys.executable).absolute()), 'python_resolved_executable': executable,
            'python_prefix': sys.prefix, 'python_base_prefix': sys.base_prefix,
            'python_executable_sha256': file_hash(executable),
            'python_version': sys.version, 'environment': {k: os.environ.get(k) for k in sorted(names)},
            'cpu_affinity': sorted(os.sched_getaffinity(0)), 'scheduler_policy': os.sched_getscheduler(0),
            'priority': os.getpriority(os.PRIO_PROCESS, 0)}


def build_contract(config, *, source_files, runner_path, completion_runtime_mode, duration_s=60.0, execution_identity=None):
    if type(duration_s) not in (int, float) or not math.isfinite(duration_s) or duration_s < 60:
        raise ValueError('Generic scientific energy requires duration >=60 seconds')
    if not source_files or len({x['path'] for x in source_files}) != len(source_files):
        raise ValueError('Generic energy requires a unique frozen source closure')
    source_files = sorted([dict(x) for x in source_files], key=lambda x: x['path'])
    if runner_path not in {x['path'] for x in source_files}:
        raise ValueError('Generic runner missing from source closure')
    if not isinstance(completion_runtime_mode, str) or not completion_runtime_mode:
        raise ValueError('Generic energy requires the actual current performance completion mode')
    contract = {'schema': SCHEMA, 'producer': 'generic_split', 'duration_s': float(duration_s),
                'warmup_count': 0, 'task_count': None, 'repetitions': 3,
                'measurement_boundary': 'prepared_input_before_p1_to_completed_pipeline_drain',
                'count_policy': 'all_admitted_tasks_completed_after_lossless_drain',
                'source_files': source_files, 'runner_path': runner_path,
                'completion_runtime_mode': completion_runtime_mode,
                'execution_identity': execution_identity or runtime_execution_identity(config),
                'operation': operation_binding(config),
                'capture_scope': 'whole_workload_command_including_initialization_and_postflight'}
    contract['command_contract_sha256'] = canonical_hash(contract)
    return contract


def validate_config(config, *, expected_sha256=None, check_files=False):
    energy = config.get('energy_capture') or {}
    contract = energy.get('contract') or {}
    body = {k: v for k, v in contract.items() if k != 'command_contract_sha256'}
    digest = canonical_hash(body)
    if (contract.get('schema') != SCHEMA or contract.get('producer') != 'generic_split'
            or digest != contract.get('command_contract_sha256')
            or (expected_sha256 is not None and digest != expected_sha256)
            or contract.get('operation') != operation_binding(config)):
        raise ValueError('Generic energy operation/source contract mismatch')
    rebuilt = build_contract(config, source_files=contract['source_files'],
                             runner_path=contract['runner_path'], duration_s=contract['duration_s'],
                             completion_runtime_mode=contract['completion_runtime_mode'],
                             execution_identity=contract['execution_identity'])
    if rebuilt != contract:
        raise ValueError('Generic energy protocol differs from supported contract')
    for source in contract['source_files']:
        if set(source) != {'path', 'sha256'} or not Path(source['path']).is_absolute() or not re.fullmatch('[0-9a-f]{64}', str(source['sha256'])):
            raise ValueError('Generic energy source descriptor malformed')
    if check_files:
        if runtime_execution_identity(config) != contract['execution_identity']:
            raise ValueError('Generic energy actual Python/environment differs from measured performance')
        descriptors = [*contract['source_files'], *config['expected_runtime_artifacts'].values(),
                       config['input_binding']['prepared_input']]
        if config.get('admission_mode') == 'generic_only':
            from .generic_only import validate_generic_only_config
            validate_generic_only_config(config, check_files=True)
        else:
            descriptors += list(config['functional_reference']['outputs'].values())
        for descriptor in descriptors:
            if file_hash(descriptor['path']) != descriptor['sha256']:
                raise ValueError('Generic energy source/artifact changed: ' + descriptor['path'])
        if str(Path(__file__).resolve()) not in {str(Path(x['path']).resolve()) for x in contract['source_files']}:
            raise ValueError('Executing Generic energy verifier missing from frozen source closure')
    return contract


def build_preflight(config, *, nonce, expected_sha256, valid_for_s=60.0):
    if not nonce or nonce == NONCE_TOKEN or len(nonce) < 16:
        raise ValueError('Fresh collector nonce required')
    if type(valid_for_s) not in (int, float) or not 0 < valid_for_s <= 60:
        raise ValueError('Invalid preflight validity window')
    contract = validate_config(config, expected_sha256=expected_sha256, check_files=True)
    now = time.time_ns()
    result = {'schema': PREFLIGHT_SCHEMA, 'schema_version': 1, 'ok': True,
              'artifact_verification_status': 'pass', 'nonce': nonce,
              'created_at_unix_ns': now, 'expires_at_unix_ns': now + int(valid_for_s * 1e9),
              'command_contract_sha256': contract['command_contract_sha256'],
              'producer': 'generic_split', 'generic_energy_schema': SCHEMA,
              'verified_source_files': contract['source_files'], 'operation_sha256': canonical_hash(contract['operation'])}
    # Match the existing collector seal exactly.
    result['attestation_sha256'] = canonical_hash(result)
    return result


def verify_runtime_preflight(config, *, nonce, attestation_path, expected_sha256):
    contract = validate_config(config, expected_sha256=expected_sha256)
    data = json.loads(Path(attestation_path).read_text())
    body = {k: v for k, v in data.items() if k != 'attestation_sha256'}
    now = time.time_ns()
    if (data.get('schema') != PREFLIGHT_SCHEMA or data.get('schema_version') != 1
            or data.get('producer') != 'generic_split' or data.get('generic_energy_schema') != SCHEMA
            or data.get('ok') is not True or data.get('artifact_verification_status') != 'pass'
            or data.get('nonce') != nonce or len(nonce) < 16 or nonce == NONCE_TOKEN
            or data.get('command_contract_sha256') != expected_sha256
            or data.get('attestation_sha256') != canonical_hash(body)
            or data.get('operation_sha256') != canonical_hash(contract['operation'])
            or data.get('verified_source_files') != contract['source_files']
            or type(data.get('created_at_unix_ns')) is not int or type(data.get('expires_at_unix_ns')) is not int
            or not data['created_at_unix_ns'] - 5_000_000_000 <= now < data['expires_at_unix_ns']
            or not 0 < data['expires_at_unix_ns'] - data['created_at_unix_ns'] <= 60_000_000_000):
        raise ValueError('Generic energy runtime preflight missing, stale or foreign')
    return data


def validate_energy_result(result, config, *, result_root=None):
    from .task_completion import validate_split_completion_duration, _validate_split_completion_samples
    contract = validate_config(config)
    execution = normalize_stage1_execution(config.get('stage1_execution'))
    if (result.get('schema_version') != 2 or result.get('measurement_mode') != 'generic_completed_task_energy'
            or result.get('status') != 'ok' or result.get('energy_contract') != contract
            or result.get('identity') != config['identity'] or result.get('input_binding') != config['input_binding']
            or result.get('completion_contract') != config['completion_contract']
            or result.get('measurement_boundary') != 'prepared_input_before_p1_to_completed_pipeline_drain'
            or result.get('functional_reference') != config['functional_reference']
            or normalize_stage1_execution(result.get('stage1_execution')) != execution
            or len(result.get('repetitions') or []) != 1):
        raise ValueError('Generic energy result schema/operation mismatch')
    row = result['repetitions'][0]
    validate_split_completion_duration(row, duration_s=contract['duration_s'])
    if (row.get('completion_runtime_mode') != contract['completion_runtime_mode']
            or row.get('execution_identity') != contract['execution_identity']
            or row.get('runtime_artifacts') != config['expected_runtime_artifacts']
            or row.get('queue_depth') != contract['operation']['queue_depth']
            or row.get('handoff_payload_copy') != contract['operation']['handoff_payload_copy']
            or normalize_stage1_execution(row.get('stage1_execution')) != execution
            or row.get('functional_control', {}).get('status') != 'passed'
            or row.get('functional_control', {}).get('reference_binding') != config['functional_reference']
            or row.get('functional_control', {}).get('checked_outside_timer') is not True
            or not row.get('preflight_nonce') or len(row['preflight_nonce']) < 16
            or row.get('energy_command_contract_sha256') != contract['command_contract_sha256']
            or any(type(row.get(k)) is not int or row[k] <= 0 for k in ('pid', 'process_start_ticks'))
            or not row.get('boot_id') or row.get('repetition_index') not in (1, 2, 3)):
        raise ValueError('Generic energy result completion/process binding mismatch')
    if config['task'] == 'detection':
        att = row.get('completion_execution_attestation') or {}
        if att.get('completed_work_units') != row['completed_work_units']:
            raise ValueError('Generic energy postflight count mismatch')
        try:
            from ..native_three_stage import verify_fast_completion_attestation
            from ..native_detection_postprocess import verify_detection_completion_execution_attestation
        except ImportError:
            from .native_three_stage import verify_fast_completion_attestation
            from .native_detection_postprocess import verify_detection_completion_execution_attestation
        if row['completion_runtime_mode'] == 'fast_oracle_outside_timing':
            verify_fast_completion_attestation(att, execution_contract=config['completion_contract'])
        else:
            verify_detection_completion_execution_attestation(att, execution_contract=config['completion_contract'],
                expected_observation_relation='same_hotloop_sentinel')
    if result_root is not None:
        _validate_split_completion_samples(row, config, result_root)
    if config.get('admission_mode') == 'generic_only':
        from .generic_only import validate_serial_reference
        validate_serial_reference(row, config, result_root=result_root)
    return row


def completion_marker(config, repetition, *, report_path, nonce):
    from .task_completion import validate_split_completion_duration
    contract = validate_config(config, check_files=True)
    validate_split_completion_duration(repetition, duration_s=contract['duration_s'])
    if normalize_stage1_execution(repetition.get('stage1_execution')) != normalize_stage1_execution(config.get('stage1_execution')):
        raise ValueError('Generic energy stage1 execution policy mismatch')
    if repetition.get('functional_control', {}).get('status') != 'passed':
        raise ValueError('Generic energy functional control did not pass')
    if repetition.get('preflight_nonce') != nonce:
        raise ValueError('Generic energy result nonce mismatch')
    count = repetition['completed_work_units']
    return {'schema': SCHEMA, 'status': 'completed_task_energy_verified', 'preflight_nonce': nonce,
            'command_contract_sha256': contract['command_contract_sha256'],
            'execution_identity_sha256': canonical_hash(contract['execution_identity']),
            'operation_sha256': canonical_hash(contract['operation']), 'completed_work_units': count,
            'report_path': str(report_path), 'report_sha256': file_hash(report_path),
            'pid': repetition['pid'], 'process_start_ticks': repetition['process_start_ticks'],
            'boot_id': repetition['boot_id']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True); p.add_argument('--nonce', required=True)
    p.add_argument('--expected-command-contract-sha256', required=True)
    p.add_argument('--attestation-path', required=True)
    a = p.parse_args()
    result = build_preflight(json.loads(Path(a.config).read_text()), nonce=a.nonce,
                             expected_sha256=a.expected_command_contract_sha256)
    dest = Path(a.attestation_path); dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open('x') as f: json.dump(result, f, sort_keys=True)
    print('__SPLITPOINT_PREFLIGHT_ATTESTATION__=' + json.dumps(result, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
