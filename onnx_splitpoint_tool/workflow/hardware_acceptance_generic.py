"""Fresh, finite Generic Completed acceptance through the normal product runner.

Historical plans supply exact inputs, artifacts and compatible references only.
Every prepared cell gets a new execution identity and normal result directory;
collector captures use the acceptance controller's existing ownership journal.
"""
from __future__ import annotations

import copy
import csv
import json
import os
from pathlib import Path
import shlex
import sys

from . import generic_completed_task as completed
from .generic_energy_parent import ResourceTransport, source_and_target


def _root(context):
    return Path(context.get('acceptance_root') or context['run_dir']).expanduser().resolve()


def _profile_budget(context):
    expected = {'expected_duration_s': 60., 'expected_repetitions': 3,
                'expected_frames': 1000, 'expected_warmup': 100}
    for name, value in expected.items():
        if context.get(name, value) != value:
            raise ValueError('Generic Completed requires the regular 100/1000/3/60s profile: ' + name)
    profile = context.get('profile') or context.get('source_profile') or {}
    if isinstance(profile, dict):
        native = profile.get('native_producers') or profile.get('native_producer') or {}
        if isinstance(native, dict) and (native.get('energy') or {}).get('task_budget'):
            from ..energy.task_budget import campaign_budget_policy
            policy = campaign_budget_policy(native)
            if policy.get('max_retries') != 2 or policy.get('max_transport_failures') != 20:
                raise ValueError('Generic Completed regular energy retry policy differs')


def _source(cell):
    binding = cell.get('generic_source_plan') or cell.get('generic_performance_plan')
    expected = cell.get('generic_source_plan_sha256')
    if isinstance(binding, dict):
        expected = binding.get('sha256') or expected
        binding = binding.get('path')
    if not binding:
        raise ValueError('Selected cell lacks a Generic completion source plan')
    path = Path(binding).expanduser().resolve(strict=True)
    digest = completed.file_sha256(path)
    if expected and digest != expected:
        raise ValueError('Selected Generic source plan changed')
    plan = completed.read_json(path)
    rows = [row for row in plan.get('rows', [])
            if all(row.get(key) == cell.get(key) for key in ('model_id', 'setup_id', 'case_id'))
            and (not cell.get('generic_case_key') or row.get('case_key') == cell['generic_case_key'])
            and (not cell.get('precision') or row.get('precision') == cell['precision'])]
    if len(rows) != 1:
        raise ValueError('Exactly one existing Generic Completed row is required for the selected cell')
    return path, digest, copy.deepcopy(rows[0])


def prepare_generic(cell, context):
    """Resolve one selected source row and generate fresh normal configuration offline."""
    _profile_budget(context)
    source, source_sha, row = _source(cell)
    root = _root(context)
    configured_primary = Path(context.get('source_primary_run') or context['primary_run']).resolve(strict=True)
    primary = Path(row['original_run']).resolve(strict=True)
    if primary != configured_primary:
        source_plan = completed.read_json(source)
        bindings = [item for item in source_plan.get('source_run_bindings', [])
                    if Path(item['archive_root']).resolve() == primary
                    and item['historical_root'] == row.get('historical_original_run')]
        if (Path(source_plan.get('original_run') or '').resolve() != configured_primary
                or len(bindings) != 1 or completed.file_sha256(primary / 'run_manifest.json')
                != bindings[0]['run_manifest_sha256']):
            raise ValueError('Selected Generic row lacks the saved primary/supplement run binding')
    for key, value in (('warmup_count', 100), ('task_count', 1000), ('repetitions', 3)):
        if row.get(key) != value:
            raise ValueError('Selected Generic work budget differs: ' + key)
    if row.get('build_allowed') is not False or row.get('fallback_allowed') is not False:
        raise ValueError('Generic acceptance requires existing artifacts without build or fallback')
    completed.case_name(row)  # Validate path components before creating outputs.
    output = root / 'cells' / (row['model_id'] + '__' + row['setup_id']) / 'generic'
    completed.outside_primary(output, primary)
    if output.exists():
        raise FileExistsError('Generic acceptance cell output must be fresh: ' + str(output))
    source_target = completed.bound_targets(primary, [row])[row['setup_id']]
    registry_path = Path(context.get('hardware_registry_path') or context['hardware_registry']).resolve(strict=True)
    registry, target, resources, calibration, collector = source_and_target(
        registry_path, row['setup_id'], source_target)
    current_target = (context.get('targets') or {}).get(row['setup_id'])
    if current_target:
        from .execution_binding import physical_dut_key
        if physical_dut_key(current_target) != physical_dut_key(target):
            raise ValueError('Saved acceptance target differs from selected Generic DUT')
    # The selected historical result is evidence of support, never the fresh result.
    historical_result = row.get('result_path')
    for name in ('runtime_config', 'staging_files', 'remote_root', 'result_path',
                 'runtime_environment_comparison', 'dispatch_attempted', 'execution_status',
                 'execution_error', 'completion_observed_at', 'dispatch_wall_seconds'):
        row.pop(name, None)
    output.mkdir(parents=True, exist_ok=False)
    scope = output / 'inputs' / 'selected_scope.csv'
    scope.parent.mkdir(parents=True, exist_ok=False)
    with scope.open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row['identity']))
        writer.writeheader()
        writer.writerow(row['identity'])
    scope_plan = output / 'inputs' / 'scope_plan.json'
    plan = {'schema': 'onnx-splitpoint/generic-completion-plan', 'schema_version': 1,
        'scope_mode': 'audited_selective_remeasurement', 'authorized_row_count': 1, 'row_count': 1,
        'original_run': str(primary), 'scope_source': str(scope),
        'scope_source_sha256': completed.file_sha256(scope),
        'source_execution_plans': [{'path': str(source), 'sha256': source_sha}],
        'source_run_bindings': [], 'start_case_keys': [row['case_key']], 'rows': [row],
        'planned_repetitions': 3, 'planned_measured_completions': 3000,
        'build_allowed': False, 'fallback_allowed': False, 'automatic_retry': False,
        'selection_policy': 'one_preselected_supported_split_per_model_and_setup',
        'remeasurement_reason': 'finite_fresh_hardware_acceptance'}
    completed.write_json(scope_plan, plan)
    execution = completed.prepare_execution(scope_plan, output)
    bundle_root = output / 'runner-bundle'
    bundle = completed.prepare_runner_bundle(bundle_root)
    prepared_plan = completed.read_json(execution)
    completed.bind_runtime_source_closure(prepared_plan['rows'][0], bundle_root)
    completed.write_json(execution, prepared_plan)
    registry_snapshot = output / 'inputs' / 'hardware_snapshot.json'
    completed.write_json(registry_snapshot, registry)
    prepared = {'schema': 'onnx-splitpoint/hardware-acceptance-generic/v1',
        'output_root': str(output), 'acceptance_root': str(root),
        'execution_plan': str(execution), 'execution_plan_sha256': completed.file_sha256(execution),
        'source_plan': {'path': str(source), 'sha256': source_sha},
        'configured_primary_run': str(configured_primary), 'selected_original_run': str(primary),
        'historical_result_reference': historical_result, 'references_reused': True,
        'runner_bundle': str(bundle), 'runner_bundle_sha256': completed.file_sha256(bundle),
        'registry_snapshot': str(registry_snapshot), 'target': target, 'resources': resources,
        'calibration': calibration, 'collector_binary': collector,
        'hardware_executed': False, 'performance_processes': 3, 'energy_captures': 3,
        'warmup_count': 100, 'task_count': 1000, 'duration_s': 60.,
        'performance_recipe': {'warmup_count': 100, 'task_count': 1000, 'repetitions': 3,
                               'process_repetition_policy': 'separate_processes'},
        'energy_recipe': {'warmup_count': 0, 'task_count': None, 'duration_s': 60., 'repetitions': 3,
                          'count_policy': 'all_admitted_tasks_completed_after_lossless_drain',
                          'capture_scope': 'whole_workload_command_including_initialization_and_postflight',
                          'physical_scope': 'FS', 'window_label': 'command',
                          'recipe_source': 'onnx_splitpoint_tool.runners.generic_energy.build_contract'},
        'build_allowed': False, 'quality_dataset_capture_requested': False}
    completed.write_json(output / 'PREPARED.json', prepared)
    return prepared


def _budget(prepared, row, context):
    from ..energy.task_budget import EnergyTaskBudget
    root = _root(context)
    path = Path(context.get('budget_file') or context.get('budget_path') or
                root / 'energy' / 'energy_task_budget.json').resolve()
    if path != root / 'energy' / 'energy_task_budget.json' or path.is_symlink():
        raise ValueError('Only the new finite acceptance budget may receive Generic captures')
    path.parent.mkdir(parents=True, exist_ok=True)
    identity = row['identity']
    key = json.dumps(['generic_completed:' + identity['comparison_backend'], identity['model_id'],
                      identity['case_id'], identity['setup_id']], separators=(',', ':'))
    source_id = str(prepared['target']['energy']['urecs_address']).strip().lower()
    with os.fdopen(os.open(path, os.O_RDWR | os.O_CREAT, 0o600), 'r+', encoding='utf-8') as stream:
        EnergyTaskBudget(stream, {'max_chains': 9, 'max_retries': 2, 'max_transport_failures': 20},
                         campaign_row=key, source_id=source_id,
                         cancel_event=context.get('cancel_event'), environment=context.get('environment'))
    return path


def _lease_prefix(transport, context):
    from ..remote.process_lease import harden_remote_process_lease_ssh_argv
    return [context.get('python') or sys.executable, '-B', '-m',
            'onnx_splitpoint_tool.remote.process_lease_cli', 'exec',
            '--label', 'generic-completed-energy', '--timeout-s', '850', '--',
            *harden_remote_process_lease_ssh_argv(transport._ssh_base())]


def _sample_loss_retry(plan_path, context):
    """Permit only normally classified sample loss within unchanged row limits."""
    from ..energy.task_budget import (_pure_sample_loss_failure, source_completion,
                                     _released_sample_loss_resources)
    plan = completed.read_json(plan_path)
    budget = completed.read_json(Path(plan['campaign_budget_file']))
    task = budget['tasks'][plan['campaign_row_id']]
    if not task['chains']:
        return False
    chain = task['chains'][-1]
    if (chain.get('finished') is not True or chain.get('valid') is True
            or chain.get('source_completion_verified') is not True
            or chain.get('transport_reasons') != ['marker_dropped_samples_nonzero']
            or not source_completion(chain['run_directory']).get('verified')):
        return False
    if not _pure_sample_loss_failure(chain, plan['campaign_row_id'], ''):
        return False
    cancel = context.get('cancel_event')
    if cancel is not None and cancel.is_set():
        return False
    broker = context.get('resource_broker')
    if broker is not None and broker.stopped_sources:
        return False
    with _released_sample_loss_resources(chain, plan['campaign_source_id'], context.get('environment')):
        completed.check_energy_campaign(plan, logical_repeat=chain['logical_repeat'],
                                        parent_run_dir=_root(context))
    return True


def _evidence(output, row, energy_plans):
    from ..generic_completion_reporting import validate_completion_case
    raw_path = Path(row['result_path'])
    raw_error = ''
    try:
        raw = completed.read_json(raw_path) if raw_path.is_file() else None
    except (ValueError, OSError) as exc:
        raw = None
        raw_error = str(exc)
    projection = validate_completion_case(raw, row, result_root=raw_path.parent)
    if raw_error:
        projection.update(status='blocked', errors=[raw_error])
    attempts = raw_path.parent / 'dispatch_attempts.json'
    performance_attempts = completed.read_json(attempts).get('attempts', []) if attempts.exists() else []
    starts = [path for plan in energy_plans for path in plan.parent.glob('capture_*_started.json')]
    chains = []
    if energy_plans:
        plan = completed.read_json(energy_plans[-1])
        budget = completed.read_json(Path(plan['campaign_budget_file']))
        chains = [chain for chain in budget['tasks'][plan['campaign_row_id']]['chains']
                  if chain.get('run_directory') and Path(chain['run_directory']).resolve().is_relative_to(output)]
    return projection, performance_attempts, starts, chains


def run_generic(prepared, context):
    """Dispatch fresh performance and normal FS captures under the GUI controller."""
    from ..generic_completion_reporting import write_completion_reports
    from ..process_control import bind_process_registry, bind_workflow_resource_options
    from ..remote.process_lease import RemoteProcessLeaseJournal
    _profile_budget(context)
    if context.get('execute_allowed') is not True:
        raise ValueError('Hardware dispatch requires the normal acceptance controller')
    if Path(prepared['acceptance_root']).resolve() != _root(context):
        raise ValueError('Prepared Generic acceptance belongs to another run')
    plan_path = Path(prepared['execution_plan'])
    if completed.file_sha256(plan_path) != prepared['execution_plan_sha256']:
        raise ValueError('Prepared Generic execution plan changed')
    bundle = Path(prepared['runner_bundle'])
    if completed.file_sha256(bundle) != prepared['runner_bundle_sha256']:
        raise ValueError('Prepared Generic runner bundle changed')
    journal = RemoteProcessLeaseJournal.from_environment(required=True)
    if os.environ.get('ONNX_SPLITPOINT_CONTROLLER_RESOURCES') != 'required':
        raise RuntimeError('Generic acceptance requires the existing controller resource broker')
    leases = context.get('remote_process_registry')
    processes = context.get('process_registry')
    if leases is None or processes is None or leases.configured_scope != journal.scope:
        raise ValueError('Generic acceptance lacks the normal controller process ownership')
    plan = completed.read_json(plan_path)
    row = plan['rows'][0]
    output = Path(prepared['output_root'])
    if Path(row['result_path']).exists():
        raise FileExistsError('Fresh acceptance cannot reuse a historical or previous result')
    # The exclusive start is a product dispatch record, never a manual claim.
    with (output / 'production-start.json').open('x', encoding='utf-8') as stream:
        json.dump({'started_at': completed.now(), 'execution_id': plan['execution_id'],
                   'run_id': journal.scope.run_id, 'session_id': journal.scope.session_id,
                   'plan': str(plan_path), 'selected_identity': row['identity'],
                   'purpose': 'finite_fresh_hardware_acceptance'}, stream, indent=2)
    log = context.get('log') or print
    raw_transport = completed.transport_for(prepared['target'], output / 'diagnostics',
        scope=journal.scope, registry=leases, cancel=context.get('cancel_event'), log=log)
    transport = ResourceTransport(raw_transport, setup=row['setup_id'], root=output / 'transfers')
    energy_plans = []
    energy = None
    error = ''
    phase = 'preflight'
    try:
        with bind_process_registry(processes), bind_workflow_resource_options(
                controller_setup_id=row['setup_id'], controller_attempt_dir=str(output)):
            completed.observe_owners({row['setup_id']: prepared['target']}, output / 'owner_preflight')
            configuration_path = output / 'configuration.json'
            snapshot = completed.capture_configuration(transport, prepared['target'],
                primary=Path(row['original_run']), output=configuration_path, phase='pre')
            completed.verify_runtime_environment(transport, row, output / 'runtime-environment.json')
            row['runtime_environment_comparison'] = completed.configuration_comparison(row, snapshot, configuration_path)
            completed.upload_new_file(transport, bundle, row['remote_root'] + '/runner-bundle.tar.gz',
                                      expected_sha=prepared['runner_bundle_sha256'])
            completed.checked_remote('tar -xzf ' + shlex.quote(row['remote_root'] + '/runner-bundle.tar.gz')
                                     + ' -C ' + shlex.quote(row['remote_root']), transport)
            phase = 'performance'
            completed.run_case(row, transport, output, 1, total=1)
            completed.write_json(plan_path, plan)
            write_completion_reports(plan, output / 'results', output / 'reports')
            phase = 'energy'
            budget = _budget(prepared, row, context)
            normal = completed.prepare_energy_case(row, performance_result=Path(row['result_path']),
                output=output / 'measurement', collector_python=context.get('python') or sys.executable,
                collector_script=Path(context['controller_root']) / 'scripts' / 'energy_measurement_cli.py',
                hardware_setups_file=Path(prepared['registry_snapshot']),
                remote_command_prefix=_lease_prefix(raw_transport, context), campaign_budget_file=budget,
                duration_s=60., require_environment_observation=True, environment_run_id=journal.scope.run_id)
            energy_plans.append(output / 'measurement' / 'generic_energy_plan.json')
            # Normal resumes preserve already valid repeats and the original limits.
            for attempt in range(7):
                try:
                    energy = completed.run_energy_case(energy_plans[-1], row, transport,
                        process_registry=processes, cancel_event=context.get('cancel_event'))
                    if energy.get('normal_report_verified') is not True or energy.get('valid_repetitions') != 3:
                        raise RuntimeError('Normal Generic energy import did not verify all three captures')
                    break
                except RuntimeError:
                    if attempt == 6 or not _sample_loss_retry(energy_plans[-1], context):
                        raise
                    resumed_output = output / 'resume_attempts' / f'normal_sample_loss_{attempt + 1}' / 'measurement'
                    completed.prepare_energy_resume(energy_plans[0], row, output=resumed_output,
                        previous_plans=energy_plans, transport=transport, parent_run_dir=_root(context))
                    energy_plans.append(resumed_output / 'generic_energy_plan.json')
            phase = 'complete'
    except Exception as exc:
        error = type(exc).__name__ + ': ' + str(exc)
        row.update(execution_status='failed' if row.get('dispatch_attempted') else 'blocked', execution_error=error)
        log('[acceptance generic] ' + row['case_key'] + ': ' + error)
    finally:
        completed.write_json(plan_path, plan)
        write_completion_reports(plan, output / 'results', output / 'reports')
    projection, attempts, starts, chains = _evidence(output, row, energy_plans)
    captures = [chain for chain in chains if chain.get('collector_started') is True]
    process_reports = {rep.get('runtime_instance_id') or (rep.get('pid'), rep.get('process_start_ticks'))
                       for rep in projection.get('raw_attempts', [])
                       if isinstance(rep, dict) and type(rep.get('pid')) is int and rep['pid'] > 0}
    energy_repeats = energy.get('valid_repetitions', 0) if energy else 0
    if energy is None and energy_plans:
        descriptors = {}
        for energy_plan in energy_plans:
            manifest = energy_plan.parent / 'capture_sources.json'
            if manifest.exists():
                for descriptor in completed.read_json(manifest).get('sources', []):
                    try:
                        from ..generic_energy_reporting import import_generic_energy_capture
                        normal = completed.read_json(energy_plan)
                        import_generic_energy_capture(descriptor, normal['config'],
                            expected_setup_id=row['setup_id'], expected_campaign=normal)
                        descriptors[descriptor['repetition_index']] = descriptor
                    except (ValueError, RuntimeError):
                        pass
        energy_repeats = len(descriptors)
    paths = [str(output / 'PREPARED.json'), str(output / 'production-start.json'), str(plan_path),
             str(output / 'reports' / 'generic_completion_summary.json')]
    paths.extend(str(path) for path in energy_plans)
    if energy is not None:
        paths.append(str(energy_plans[-1].parent / 'reports' / 'generic_completed_task_energy.json'))
    result = {'status': 'completed' if phase == 'complete' else ('failed' if attempts or starts else 'blocked'),
        'fresh_hardware_executed': bool(process_reports or captures), 'new_capture_count': len(captures),
        'collector_cli_attempts': len(starts), 'new_performance_dispatch_attempt_count': len(attempts),
        'new_performance_process_count': len(process_reports),
        'failed_capture_count': sum(chain.get('finished') is True and chain.get('valid') is not True for chain in captures),
        'unknown_capture_count': sum(chain.get('finished') is not True for chain in captures),
        'physical_execution_uncertain': bool((attempts and not process_reports) or (starts and not captures)),
        'valid_repeats': {'performance': projection['valid_repetition_count'], 'energy': energy_repeats},
        'failed_attempts': {'performance': sum(a.get('returncode') != 0 for a in attempts),
                            'energy': max(0, len(starts) - energy_repeats)},
        'output_validation': {'status': projection['status'], 'procedure': 'normal_bound_functional_output_validation'},
        'performance': projection, 'energy': energy or {'status': 'incomplete', 'valid_repetitions': energy_repeats},
        'import_projection': {'status': 'completed' if energy is not None and energy.get('normal_report_verified') else 'incomplete',
                              'normal_report_verified': bool(energy and energy.get('normal_report_verified'))},
        'evidence_paths': paths, 'error': error, 'stopped_phase': '' if phase == 'complete' else phase,
        'references_reused': {'historical_performance_result': prepared['historical_result_reference'],
                              'native_functional_input_and_quality_reference': row['native_source']},
        'new_dataset_quality_capture': False}
    completed.write_json(output / 'generic_result.json', result)
    return result
