"""Normal collector import for the Generic completed-task duration workload.

Raw collector aggregates and runtime reports are immutable source evidence.
Only exact same-operation performance may be paired with these observations.
"""
from __future__ import annotations
import json
import math
from pathlib import Path
import shlex
from .runners.generic_energy import (SCHEMA, COMPLETION_MARKER, canonical_hash, file_hash,
                                     operation_binding, validate_config, validate_energy_result)
from .runners.task_completion import valid_split_completion_repetitions
from .energy.collector import (_energy_repeat_contract_complete, _repeat_statistics,
                               _clamp_diagnostic_claim_fields)


def _positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def _acceptance_repetitions(scope):
    """Keep scientific cohorts strict; opt in explicitly to a functional check."""
    mode = scope.get('acceptance_mode', 'scientific')
    expected = 1 if mode == 'functional' else 3
    requested = scope.get('requested_repetitions', expected)
    if mode not in {'scientific', 'functional'} or type(requested) is not int or requested != expected:
        raise ValueError('Generic acceptance mode/repetition recipe differs')
    if mode == 'functional' and scope.get('requested_repetitions') != 1:
        raise ValueError('Generic functional acceptance requires one explicit requested repetition')
    return expected


def _functional_view(value):
    """Project derived flags without modifying immutable source records."""
    def generic_fields(item):
        if isinstance(item, dict):
            return {key: False if key in {'eligible_for_claim_transfer', 'primary_energy_claim_gate'}
                    else generic_fields(child) for key, child in item.items()}
        if isinstance(item, list):
            return [generic_fields(child) for child in item]
        return item
    return generic_fields(_clamp_diagnostic_claim_fields(value))


def _source(path, expected_sha, *, maximum_bytes=8 * 1024 * 1024):
    path = Path(path)
    if not path.is_file() or path.is_symlink() or path.stat().st_size > maximum_bytes:
        raise ValueError('Generic energy evidence missing, symlinked or too large')
    raw = path.read_bytes()
    import hashlib
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError('Generic energy evidence source hash mismatch')
    return raw


def import_generic_energy_capture(descriptor, config, *, expected_setup_id, expected_campaign):
    """Import a real single capture through the existing collector importer."""
    from scripts.run_native_producer_energy_from_summary import _attach_energy_aggregate, _selected_energy_repeat_run_dir
    contract = validate_config(config)
    requested_repetitions = _acceptance_repetitions(expected_campaign)
    command = descriptor['command']; receipt = descriptor['process_receipt']
    if (not isinstance(command, list) or canonical_hash(command) != descriptor['command_sha256']
            or receipt.get('argv') != command or receipt.get('command_sha256') != descriptor['command_sha256']
            or type(receipt.get('finished_ns')) is not int or type(receipt.get('started_ns')) is not int
            or receipt['finished_ns'] <= receipt['started_ns']):
        raise ValueError('Generic energy physical capture command/receipt mismatch')
    output = Path(descriptor['output_dir']).resolve()
    raw_path = output / 'energy_aggregate.json'
    if str(raw_path) != descriptor['raw_aggregate_path']:
        raise ValueError('Generic energy aggregate path does not belong to capture')
    _source(raw_path, descriptor['raw_aggregate_sha256'])
    imported = _attach_energy_aggregate({'rc': receipt['returncode']}, shlex.join(command), output,
        expected_runs=1, expected_effective_runs=1, expected_setup_id=expected_setup_id,
        expected_command_contract_sha256=contract['command_contract_sha256'],
        execution_started_ns=receipt['started_ns'], aggregate_absent_before_execution=receipt['aggregate_absent_before_execution'],
        caller_managed_single_capture=True)
    if imported.get('energy_aggregate_verified') is not True:
        raise ValueError('Normal energy collector import rejected: ' + repr(imported.get('energy_aggregate_validation_errors', []) + imported.get('energy_aggregate_completion_errors', [])))
    if imported['energy_aggregate_sha256'] != descriptor['raw_aggregate_sha256']:
        raise ValueError('Generic energy aggregate changed during normal import')
    aggregate = imported['energy_aggregate']; records = aggregate['runs']
    if len(records) != 1 or records[0].get('run_index') != 0 or not _energy_repeat_contract_complete(records, requested_count=1, require_ab=False):
        raise ValueError('Generic energy requires one valid actual primary capture')
    record = records[0]
    directory, selection = _selected_energy_repeat_run_dir(record, measurement_root=output, logical_index=0,
                                                          campaign_command=shlex.join(command))
    import fcntl
    def option(flag):
        if command.count(flag) != 1:
            raise ValueError('Generic energy campaign option missing or ambiguous: ' + flag)
        return command[command.index(flag) + 1]
    identity = config['identity']
    row_id = json.dumps(['generic_completed:' + identity['comparison_backend'], identity['model_id'],
                         identity['case_id'], identity['setup_id']], separators=(',', ':'))
    index = descriptor['repetition_index']
    logical_repeat = f'repeat:{index - 1}'
    if (index not in range(1, requested_repetitions + 1) or option('--campaign-row-id') != row_id
            or expected_campaign['campaign_row_id'] != row_id
            or option('--campaign-budget-file') != expected_campaign['campaign_budget_file']
            or option('--task-logical-repeat') != logical_repeat
            or option('--campaign-repeats') != str(requested_repetitions) or option('--campaign-max-retries') != '2'
            or option('--campaign-max-transport-failures') != '20' or '--task-budget-file' in command):
        raise ValueError('Generic energy original shared campaign binding differs')
    path = Path(expected_campaign['campaign_budget_file'])
    if not path.is_file() or path.is_symlink():
        raise ValueError('Generic energy original campaign checkpoint missing')
    with path.open() as stream:
        fcntl.flock(stream, fcntl.LOCK_SH)
        budget = json.load(stream)
    task = budget.get('tasks', {}).get(row_id) or {}
    if (budget.get('campaign') is not True or budget.get('max_transport_failures') != 20
            or task.get('limits') != {'max_chains': 3 * requested_repetitions, 'max_retries': 2, 'max_transport_failures': 20}
            or task.get('source_id') != expected_campaign['campaign_source_id']):
        raise ValueError('Generic energy actual campaign case/source differs')
    matching = [c for c in task['chains'] if c.get('logical_repeat') == logical_repeat and c.get('finished') is True and c.get('valid') is True]
    if (len(matching) != 1 or matching[0].get('run_directory') != str(directory)
            or matching[0].get('source_completion_verified') is not True):
        raise ValueError('Generic energy capture differs from accepted shared campaign chain')
    log = directory / 'workload_stdout.log'
    binding = (record.get('collector_and_workload_log_diagnostics') or {}).get('workload_stdout') or {}
    stdout = _source(log, binding.get('sha256'))
    if binding.get('available') is not True or binding.get('bytes') != len(stdout):
        raise ValueError('Generic energy captured workload stdout binding invalid')
    markers = [json.loads(line[len(COMPLETION_MARKER):]) for line in stdout.decode().splitlines() if line.startswith(COMPLETION_MARKER)]
    if len(markers) != 1:
        raise ValueError('Generic energy fresh completion marker missing or ambiguous')
    marker = markers[0]; preflight = record.get('preflight_evidence') or {}; work = record.get('runtime_work_unit_evidence') or {}
    timing = record.get('workload_timing') or {}
    if (marker.get('schema') != SCHEMA or marker.get('status') != 'completed_task_energy_verified'
            or marker.get('command_contract_sha256') != contract['command_contract_sha256']
            or marker.get('operation_sha256') != canonical_hash(contract['operation'])
            or marker.get('execution_identity_sha256') != canonical_hash(contract['execution_identity'])
            or preflight.get('ok') is not True or not preflight.get('nonce')
            or marker.get('preflight_nonce') != preflight['nonce']
            or work.get('exact') is not True or type(work.get('count')) is not int
            or work['count'] != marker.get('completed_work_units') or work['count'] <= 0
            or record.get('energy_work_units_used') != work['count']
            or timing.get('status') != 'ok' or timing.get('rc') != 0
            or type(timing.get('start_ns')) is not int or type(timing.get('end_ns')) is not int
            or timing['end_ns'] <= timing['start_ns']
            or not _positive(record.get('scientific_primary_active_duration_s'))
            or record['scientific_primary_active_duration_s'] < contract['duration_s']
            or record.get('energy_configured_workload_duration_s') != contract['duration_s']):
        raise ValueError('Generic energy nonce/count/operation/window mismatch')
    raw_result = _source(descriptor['runtime_result_path'], marker.get('report_sha256'))
    if descriptor.get('runtime_remote_result_path') != marker.get('report_path'):
        raise ValueError('Generic energy downloaded report source path mismatch')
    result = json.loads(raw_result)
    repetition = validate_energy_result(result, config, result_root=Path(descriptor['runtime_result_path']).parent)
    if any(marker.get(k) != repetition.get(k) for k in ('pid', 'process_start_ticks', 'boot_id', 'completed_work_units', 'preflight_nonce')):
        raise ValueError('Generic energy report and physical capture identity differ')
    expected_sources = {x['path']: x['sha256'] for x in contract['source_files']}
    if repetition.get('runtime_source_binding') != {'before': expected_sources, 'after': expected_sources, 'unchanged': True}:
        raise ValueError('Generic energy measured source closure missing or changed')
    energy = record.get('scientific_primary_energy_j'); count = work['count']
    if not _positive(energy):
        raise ValueError('Generic energy primary joules invalid')
    claim = (aggregate.get('energy_efficiency_claim_eligible') is True
             and record.get('energy_efficiency_claim_eligible') is True
             and aggregate.get('diagnostic_only') is not True and record.get('diagnostic_only') is not True
             and not aggregate.get('claim_exclusion_reason') and not record.get('claim_exclusion_reason'))
    observation = {'schema': SCHEMA, 'status': 'normal_collector_and_generic_completion_verified',
        'raw_aggregate_path': str(raw_path), 'raw_aggregate_sha256': descriptor['raw_aggregate_sha256'],
        'runtime_result_path': descriptor['runtime_result_path'], 'runtime_result_sha256': marker['report_sha256'],
        'physical_run_directory': str(directory), 'selection_evidence': selection,
        'preflight_nonce': preflight['nonce'], 'pid': repetition['pid'],
        'process_start_ticks': repetition['process_start_ticks'], 'boot_id': repetition['boot_id'],
        'raw_run_index': record['run_index'], 'completed_tasks': count,
        'energy_j': energy, 'energy_per_completed_task_j': energy / count,
        'completed_tasks_per_j': count / energy, 'active_duration_s': record['scientific_primary_active_duration_s'],
        'source_contract_sha256': contract['command_contract_sha256'],
        'completion_runtime_mode': repetition['completion_runtime_mode'],
        'normal_import': imported, 'primary_energy_claim_gate': claim}
    if requested_repetitions == 1:
        observation = _functional_view(observation)
        observation.update(evaluation_role='functional_acceptance', diagnostic_only=True,
                           source_flags_projected_for_functional_scope=True)
    return observation


def _admitted_performance_pair(plan, case, admission):
    """Reuse the exact normal comparison admitted by the owning product parent."""
    from .ranking_methods import canonical_direction
    identity = plan['identity']
    performance = admission.get('performance_result') or {}
    if (performance != {'path': plan['performance_result'], 'sha256': plan['performance_result_sha256']}
            or admission.get('valid_repetitions') != 3
            or admission.get('fresh_confirmation_only') is not True
            or case.get('identity') != identity):
        raise ValueError('Generic admitted performance source or identity differs')
    binding = admission.get('normal_pair_report') or {}
    report = json.loads(_source(binding.get('path', ''), binding.get('sha256')))
    selected = [pair for pair in report.get('pairs', [])
        if pair.get('generic_source_path') == plan['performance_result']
        and all(pair.get(key) == identity.get(key) for key in ('model_id', 'case_id', 'setup_id', 'precision'))
        and canonical_direction(pair.get('direction')) == canonical_direction(identity.get('direction'))]
    if (len(selected) != 1 or selected[0] != admission.get('normal_pair')
            or selected[0].get('eligible_for_technical_transfer') is not True):
        raise ValueError('Generic admitted normal comparison missing, changed or technically excluded')
    for key in ('native_cohort_proof', 'coordinator_success'):
        source = admission.get(key) or {}
        _source(source.get('path', ''), source.get('sha256'))
    return selected[0]


def _admitted_generic_only_performance(plan, case, projection, admission):
    """Retain normal standalone admission, without manufacturing a Native pair."""
    from .generic_only_reporting import generic_only_admission
    own = generic_only_admission(projection, case)
    admission = admission or {}
    binding = admission.get('normal_generic_only_report') or {}
    report = json.loads(_source(binding.get('path', ''), binding.get('sha256')))
    selected = [item for item in report.get('rows', []) if item.get('case_key') == case['case_key']]
    if (admission.get('admission_mode') != 'generic_only'
            or admission.get('performance_result') != {'path': plan['performance_result'], 'sha256': plan['performance_result_sha256']}
            or admission.get('valid_repetitions') != 3
            or own.get('eligible_for_generic_only_execution') is not True
            or selected != [own] or admission.get('normal_generic_only_admission') != own
            or admission.get('native_performance_claimed') is not False):
        raise ValueError('Generic-only energy normal source admission missing or changed')
    return own


def collect_generic_energy_case(plan, descriptors, case, *, performance_admission=None):
    """Admit three fresh captures and retain normal Generic quality gates."""
    if _acceptance_repetitions(plan) == 1:
        if performance_admission is not None:
            raise ValueError('Functional Generic acceptance cannot borrow a scientific cohort admission')
        return _collect_functional_energy_case(plan, descriptors, case)
    from .generic_completion_reporting import validate_completion_case, completion_pair
    config = plan['config']; contract = validate_config(config)
    performance_path = Path(plan['performance_result'])
    performance = json.loads(_source(performance_path, plan['performance_result_sha256']))
    if config['energy_capture'].get('performance_result_sha256') != plan['performance_result_sha256']:
        raise ValueError('Generic energy current performance source mismatch')
    performance_config = case['runtime_config']
    if operation_binding(performance_config) != contract['operation']:
        raise ValueError('Generic energy/performance operation differs')
    reps = valid_split_completion_repetitions(performance, performance_config, result_root=performance_path.parent)
    if len(reps) != 3 or performance.get('process_repetition_policy') != 'separate_processes':
        raise ValueError('Generic energy requires three actual bound performance processes')
    source_map = {x['path']: x['sha256'] for x in contract['source_files']}
    if any(r.get('runtime_source_binding') != {'before': source_map, 'after': source_map, 'unchanged': True}
           or r.get('execution_identity') != contract['execution_identity']
           or r.get('completion_runtime_mode', 'classification_top1_top5') != contract['completion_runtime_mode'] for r in reps):
        raise ValueError('Generic energy/performance source or completion mode differs')
    if sorted(d.get('repetition_index') for d in descriptors) != [1, 2, 3]:
        raise ValueError('Three explicit Generic energy capture positions required')
    calls = {c['repetition_index']: c for c in plan['calls']}
    if any(d['command'] != calls[d['repetition_index']]['argv']
           or canonical_hash(calls[d['repetition_index']]['argv']) != calls[d['repetition_index']]['command_sha256'] for d in descriptors):
        raise ValueError('Generic energy capture differs from prepared command')
    for call in calls.values():
        for kind in ('workload_command', 'preflight_command'):
            _source(call[kind + '_file'], call[kind + '_sha256'])
    observations = [import_generic_energy_capture(d, config, expected_setup_id=case['setup_id'], expected_campaign=plan) for d in descriptors]
    for field in ('raw_aggregate_path', 'raw_aggregate_sha256', 'physical_run_directory', 'preflight_nonce'):
        if len({r[field] for r in observations}) != 3:
            raise ValueError('Generic energy captures repeat a physical source: ' + field)
    if len({(r['boot_id'], r['pid'], r['process_start_ticks']) for r in observations}) != 3:
        raise ValueError('Generic energy captures share a workload process')
    statistics = {field: _repeat_statistics([r[field] for r in observations], .95) for field in
                  ('energy_j', 'energy_per_completed_task_j', 'completed_tasks_per_j', 'active_duration_s')}
    projection = validate_completion_case(performance, case, result_root=performance_path.parent)
    generic_only = config.get('admission_mode') == 'generic_only'
    own_admission = None
    if generic_only:
        own_admission = _admitted_generic_only_performance(plan, case, projection, performance_admission)
        pair = None
        quality_admitted = own_admission['quality_eligible']
    else:
        pair = (completion_pair(projection, case) if performance_admission is None else
                _admitted_performance_pair(plan, case, performance_admission))
        quality_admitted = pair.get('eligible_for_claim_transfer') is True
    claim = quality_admitted and all(r['primary_energy_claim_gate'] for r in observations)
    return {'schema': 'onnx-splitpoint/generic-completed-task-energy-observation/v1',
        'identity': config['identity'], 'status': 'completed', 'valid_repetitions': 3,
        'performance_result_sha256': plan['performance_result_sha256'],
        'source_contract_sha256': contract['command_contract_sha256'],
        'operation': contract['operation'], 'observations': observations, 'statistics': statistics,
        'energy_per_completed_task_j': statistics['energy_per_completed_task_j']['mean'],
        'completed_tasks_per_j': statistics['completed_tasks_per_j']['mean'],
        'inverse_mean_energy_per_completed_task_j': 1 / statistics['energy_per_completed_task_j']['mean'],
        'normal_report_verified': True, 'energy_efficiency_claim_eligible': claim,
        'normal_generic_performance_pair': pair,
        **({'admission_mode': 'generic_only', 'normal_generic_only_admission': own_admission,
            'native_performance_claimed': False} if generic_only else {}),
        'scientific_primary_method': 'command_marker_window',
        'raw_source_records_modified': False, 'raw_run_indices': [r['raw_run_index'] for r in observations]}


def _collect_functional_energy_case(plan, descriptors, case):
    """Validate one ordinary capture and publish no scientific cohort claims."""
    from .generic_completion_reporting import validate_completion_case
    if _acceptance_repetitions(plan) != 1 or _acceptance_repetitions(case) != 1:
        raise ValueError('Functional Generic energy/performance role differs')
    config = plan['config']; contract = validate_config(config)
    performance_path = Path(plan['performance_result'])
    performance = json.loads(_source(performance_path, plan['performance_result_sha256']))
    if (config['energy_capture'].get('performance_result_sha256') != plan['performance_result_sha256']
            or operation_binding(case['runtime_config']) != contract['operation']):
        raise ValueError('Functional Generic performance source/operation differs')
    projection = validate_completion_case(performance, case, result_root=performance_path.parent)
    repetitions = projection['repetitions']
    if (projection['status'] != 'partial' or projection['valid_repetition_count'] != 1
            or repetitions[0]['repetition_index'] != 1
            or performance.get('process_repetition_policy') != 'separate_processes'):
        raise ValueError('Functional Generic energy requires one actual validated performance process')
    source_map = {x['path']: x['sha256'] for x in contract['source_files']}
    rep = repetitions[0]
    if (rep.get('runtime_source_binding') != {'before': source_map, 'after': source_map, 'unchanged': True}
            or rep.get('execution_identity') != contract['execution_identity']
            or rep.get('completion_runtime_mode', 'classification_top1_top5') != contract['completion_runtime_mode']):
        raise ValueError('Functional Generic energy/performance source or completion mode differs')
    if ([d.get('repetition_index') for d in descriptors] != [1]
            or [c.get('repetition_index') for c in plan['calls']] != [1]):
        raise ValueError('Functional Generic energy requires one explicit actual capture')
    call = plan['calls'][0]; descriptor = descriptors[0]
    if descriptor['command'] != call['argv'] or canonical_hash(call['argv']) != call['command_sha256']:
        raise ValueError('Functional Generic capture differs from its prepared command')
    for kind in ('workload_command', 'preflight_command'):
        _source(call[kind + '_file'], call[kind + '_sha256'])
    observation = import_generic_energy_capture(descriptor, config,
        expected_setup_id=case['setup_id'], expected_campaign=plan)
    return {'schema': 'onnx-splitpoint/generic-functional-energy-observation/v1',
        'acceptance_mode': 'functional', 'requested_repetitions': 1, 'valid_repetitions': 1,
        'evaluation_role': 'functional_acceptance', 'diagnostic_only': True,
        'identity': config['identity'], 'status': 'completed', 'normal_report_verified': True,
        'performance_result_sha256': plan['performance_result_sha256'],
        'source_contract_sha256': contract['command_contract_sha256'], 'operation': contract['operation'],
        'observations': [observation], 'normal_scientific_performance_projection': projection,
        'aggregation': 'single_observation_no_cohort_statistics',
        'energy_per_completed_task_j': observation['energy_per_completed_task_j'],
        'completed_tasks_per_j': observation['completed_tasks_per_j'],
        'energy_efficiency_claim_eligible': False, 'scientific_cohort_complete': False,
        'scientific_claim_eligible': False, 'scientific_statistics_published': False,
        'raw_source_records_modified': False, 'raw_run_indices': [observation['raw_run_index']]}
