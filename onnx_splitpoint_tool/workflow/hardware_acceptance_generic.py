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
import re
from datetime import datetime
from pathlib import Path
import shlex
import sys
from typing import Mapping

from . import generic_completed_task as completed
from .generic_energy_parent import ResourceTransport, source_and_target


def _root(context):
    return Path(context.get('acceptance_root') or context['run_dir']).expanduser().resolve()


def _campaign_key(identity):
    return json.dumps(['generic_completed:' + identity['comparison_backend'], identity['model_id'],
                       identity['case_id'], identity['setup_id']], separators=(',', ':'))


def _profile_budget(context):
    mode, repetitions = _acceptance_recipe(context)
    expected = {'expected_duration_s': 60., 'expected_repetitions': repetitions,
                'expected_frames': 1000, 'expected_warmup': 100}
    for name, value in expected.items():
        if context.get(name, value) != value:
            raise ValueError('Generic Completed regular acceptance recipe differs: ' + name)
    profile = context.get('profile') or context.get('source_profile') or {}
    if isinstance(profile, dict):
        native = profile.get('native_producers') or profile.get('native_producer') or {}
        if isinstance(native, dict) and (native.get('energy') or {}).get('task_budget'):
            from ..energy.task_budget import campaign_budget_policy
            policy = campaign_budget_policy(native)
            if policy.get('max_retries') != 2 or policy.get('max_transport_failures') != 20:
                raise ValueError('Generic Completed regular energy retry policy differs')


def _acceptance_recipe(context):
    profile = context.get('profile') or context.get('source_profile') or {}
    acceptance = profile.get('hardware_acceptance') or {}
    profile_mode = acceptance.get('mode', 'scientific')
    mode = context.get('acceptance_mode', profile_mode)
    if mode != profile_mode:
        raise ValueError('Generic acceptance mode differs from its explicit profile')
    repetitions = completed._acceptance_repetitions({'acceptance_mode': mode,
        'requested_repetitions': context.get('requested_repetitions', 1 if mode == 'functional' else 3)})
    return mode, repetitions


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


def _bound_reference(value, root=None):
    if not isinstance(value, Mapping) or not re.fullmatch(r'[a-f0-9]{64}', str(value.get('sha256') or '')):
        raise ValueError('Generic fresh performance evidence requires an explicit SHA256')
    path = Path(str(value.get('path') or ''))
    if (not path.is_absolute() or str(path.resolve()) != str(path)
            or any(part.is_symlink() for part in (path, *path.parents))
            or not path.is_file() or completed.file_sha256(path) != value['sha256']):
        raise ValueError('Generic fresh performance evidence changed or is not canonical')
    if root is not None and not path.is_relative_to(root):
        raise ValueError('Generic fresh performance evidence escapes its closed origin')
    return path


def _performance_reference(cell):
    value = cell.get('fresh_performance_reference')
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError('Generic fresh performance reference must be hash-bound')
    if 'path' in value:
        value = completed.read_json(_bound_reference(value))
    return copy.deepcopy(dict(value))


def _closed_origin(reference, context):
    from ..benchmark.evaluation_profiles import load_evaluation_profile
    manifest_path = _bound_reference(reference.get('run_manifest'))
    origin = manifest_path.parent
    current = _root(context)
    profile_path = _bound_reference(reference.get('profile'), origin)
    closure_path = _bound_reference(reference.get('closure'), origin)
    if (manifest_path.name != 'run_manifest.json' or profile_path != origin / 'profile.yaml'
            or closure_path != origin / 'reports/artifact_index_closure.json'):
        raise ValueError('Generic fresh performance origin has nonstandard terminal paths')
    manifest = completed.read_json(manifest_path)
    closure = completed.read_json(closure_path)
    old_profile = load_evaluation_profile(profile_path, validate=False).raw_profile
    old_acceptance = old_profile.get('hardware_acceptance') or {}
    current_profile = context.get('profile') or context.get('source_profile') or {}
    current_acceptance = current_profile.get('hardware_acceptance') or {}
    if (old_acceptance.get('enabled') is not True or current_acceptance.get('enabled') is not True
            or old_acceptance.get('schema') != 'onnx-splitpoint/hardware-acceptance/v1'
            or (manifest.get('hardware_acceptance') or {}).get('schema') != old_acceptance['schema']
            or (manifest.get('hardware_acceptance') or {}).get('source_campaign_resumed') is not False
            or manifest.get('status') not in {'ok', 'partial', 'failed', 'cancelled'}
            or closure.get('schema') != 'onnx-splitpoint/artifact-index-terminal-closure'
            or closure.get('status') != 'pass' or closure.get('run_id') != manifest.get('run_id')
            or manifest.get('run_id') != origin.name or closure.get('workflow_status') != manifest.get('status')):
        raise ValueError('Generic fresh performance origin is not a closed normal acceptance run')
    matrix = Path(str(current_acceptance.get('matrix_evidence') or '')).resolve(strict=True)
    old_matrix = Path(str(old_acceptance.get('matrix_evidence') or '')).resolve(strict=True)
    if (matrix != old_matrix or origin == current or origin.parent != current.parent
            or not origin.is_relative_to(matrix.parent / 'hardware_runs')
            or not current.is_relative_to(matrix.parent / 'hardware_runs')):
        raise ValueError('Generic fresh performance origin is outside the current acceptance phase')
    fixed = completed.read_json(matrix)
    created = datetime.fromisoformat(str(manifest.get('created_at')))
    if (created < datetime.fromisoformat(str(fixed.get('recorded_before_hardware_start')))
            or created > datetime.now(created.tzinfo)):
        raise ValueError('Generic fresh performance predates the fixed acceptance phase')
    for profile in (old_profile, current_profile):
        native = profile.get('native_producers') or {}
        energy = native.get('energy') or {}
        budget = energy.get('task_budget') or {}
        if (tuple(native.get(k) for k in ('warmup', 'frames', 'repetitions')) != (100, 1000, 3)
                or tuple(energy.get(k) for k in ('duration_s', 'physical_scope', 'window_label')) != (60, 'FS', 'command')
                or budget.get('enabled') is not True or budget.get('max_retries') != 2
                or budget.get('max_transport_failures') != 20):
            raise ValueError('Generic fresh performance profile budget differs')
    return origin, matrix, fixed, old_profile


def collect_generic_performance_case(prepared, context, reference):
    """Revalidate a closed, fresh same-phase cohort through the normal parser."""
    from ..generic_completion_reporting import validate_completion_case
    from ..runners.generic_energy import execution_environment
    origin, matrix, fixed, old_profile = _closed_origin(reference, context)
    original_prepared = _bound_reference(reference.get('prepared'), origin)
    old_output = origin / 'cells' / (prepared['model_id'] + '__' + prepared['setup_id']) / 'generic'
    old = completed.read_json(original_prepared)
    if original_prepared != old_output / 'PREPARED.json' or Path(old.get('output_root') or '') != old_output:
        raise ValueError('Generic fresh performance PREPARED belongs to another cell')
    for field in ('schema', 'source_plan', 'configured_primary_run', 'selected_original_run', 'target',
                  'performance_recipe', 'energy_recipe', 'collector_binary'):
        if old.get(field) != prepared.get(field):
            raise ValueError('Generic fresh performance source/target/recipe differs: ' + field)
    plan_path = _bound_reference(reference.get('execution_plan'), old_output)
    if plan_path != old_output / 'plan/execution-plan.json' or old.get('execution_plan') != str(plan_path):
        raise ValueError('Generic fresh performance execution plan path differs')
    plan = completed.read_json(plan_path)
    rows = completed.validate_plan(plan, Path(plan['scope_source']), Path(plan['original_run']))
    if len(rows) != 1 or plan.get('authorized_row_count') != 1:
        raise ValueError('Generic fresh performance origin is not a single selected cell')
    row = copy.deepcopy(rows[0])
    expected = prepared['selected_identity']
    if row.get('identity') != expected or not any(all(item.get(k) == expected[k] for k in
            ('model_id', 'setup_id', 'case_id')) for item in fixed.get('cells', [])):
        raise ValueError('Generic fresh performance selection is outside the fixed matrix')
    for profile in (old_profile, context.get('profile') or context['source_profile']):
        cells = (profile.get('hardware_acceptance') or {}).get('cells') or []
        if not any(all(item.get(k) == expected[k] for k in ('model_id', 'setup_id', 'case_id'))
                   and 'generic_completed' in [str(v).lower().replace(' ', '_') for v in item.get('runtime_variants', [])]
                   for item in cells):
            raise ValueError('Generic fresh performance cell is absent from its normal profile')
    source = completed.read_json(Path(prepared['source_plan']['path']))
    matches = [item for item in source['rows'] if item.get('identity') == expected]
    if len(matches) != 1 or completed.file_sha256(Path(prepared['source_plan']['path'])) != prepared['source_plan']['sha256']:
        raise ValueError('Generic fresh performance selected source changed')
    for field in ('input_binding', 'completion_contract', 'functional_reference', 'expected_runtime_artifacts',
                  'original_run_cfg', 'native_source', 'native_command_contract_sha256', 'measurement_boundary'):
        if row.get(field) != matches[0].get(field):
            raise ValueError('Generic fresh performance input/artifact/source differs: ' + field)
    performance_path = _bound_reference(reference.get('performance_result'), old_output)
    if performance_path != Path(row['result_path']) or not performance_path.is_relative_to(old_output / 'results'):
        raise ValueError('Generic fresh performance raw result origin differs')
    raw = completed.read_json(performance_path)
    projection = validate_completion_case(raw, row, result_root=performance_path.parent)
    if (projection['status'] != 'completed' or projection['valid_repetition_count'] != 3
            or raw.get('process_repetition_policy') != 'separate_processes'
            or raw.get('execution_id') != plan['execution_id']):
        raise ValueError('Generic fresh performance lacks three real completed processes')
    summary_path = _bound_reference(reference.get('summary'), old_output)
    summary = completed.read_json(summary_path)
    if (summary_path != old_output / 'reports/generic_completion_summary.json'
            or summary.get('planned_cases') != 1 or summary.get('valid_repetitions') != 3
            or summary.get('completed_task_count') != 3000 or summary.get('all_cases_measured') is not True):
        raise ValueError('Generic fresh performance normal summary is incomplete')
    attempts_path = _bound_reference(reference.get('dispatch_attempts'), old_output)
    attempts = completed.read_json(attempts_path).get('attempts') or []
    if attempts_path != performance_path.parent / 'dispatch_attempts.json' or len(attempts) != 3:
        raise ValueError('Generic fresh performance requires three original dispatch receipts')
    refs = reference.get('repeats') or []
    if len(refs) != 3:
        raise ValueError('Generic fresh performance requires three explicit runtime identities')
    config = row['runtime_config']
    sources = {item['path']: item['sha256'] for item in config.get('runtime_source_files') or []}
    if not sources or config.get('execution_environment') != execution_environment(
            row['remote_python_executable'], row.get('remote_environment') or {}):
        raise ValueError('Generic fresh performance runtime closure/environment differs')
    for index, (rep, ref, attempt) in enumerate(zip(projection['repetitions'], refs, attempts), 1):
        fields = ('repetition_index', 'runtime_instance_id', 'pid', 'process_start_ticks', 'boot_id')
        remote_config = str(Path(config['original_run_cfg']['native_split_quality_binding']).parent / 'generic_completion_config.json')
        command = ('cd -- ' + shlex.quote(row['remote_root']) + ' && ' + shlex.join([
            row['remote_python_executable'], '-B', '-u', row['remote_root'] + '/runner/run_split_onnxruntime.py',
            '--generic-completion-config', remote_config, '--generic-completion-repetition', str(index),
            '--out-dir', row['remote_root'] + '/results/' + row['case_output_relative']]))
        if (any(rep.get(k) != ref.get(k) for k in fields) or rep['repetition_index'] != index
                or rep.get('execution_id') != plan['execution_id']
                or rep.get('execution_identity') != projection['repetitions'][0].get('execution_identity')
                or not rep.get('execution_identity')
                or rep.get('runtime_source_binding') != {'before': sources, 'after': sources, 'unchanged': True}
                or attempt.get('repetition_index') != index or attempt.get('returncode') != 0
                or (attempt.get('download') or {}).get('result_collected') is not True
                or shlex.split(attempt.get('command') or '') != shlex.split(command)):
            raise ValueError('Generic fresh performance original process/dispatch binding differs')
    bundle = _bound_reference(reference.get('runner_bundle'), old_output)
    if str(bundle) != old.get('runner_bundle') or completed.file_sha256(bundle) != old.get('runner_bundle_sha256'):
        raise ValueError('Generic fresh performance original runner bundle changed')
    old_bundle = old_output / 'runner-bundle'
    current_bundle = Path(prepared['output_root']) / 'runner-bundle'
    names = ['runner/run_split_onnxruntime.py', *[str(p.relative_to(current_bundle))
             for p in sorted((current_bundle / 'splitpoint_runners').rglob('*.py'))]]
    old_names = ['runner/run_split_onnxruntime.py', *[str(p.relative_to(old_bundle))
                 for p in sorted((old_bundle / 'splitpoint_runners').rglob('*.py'))]]
    if names != old_names or not (old_bundle / names[0]).is_file():
        raise ValueError('Generic fresh performance generated suite closure differs')
    for name in names:
        digest = completed.file_sha256(old_bundle / name)
        if digest != sources.get(row['remote_root'] + '/' + name) or completed.file_sha256(current_bundle / name) != digest:
            raise ValueError('Generic fresh performance current generated source is incompatible: ' + name)
    scope = reference.get('reuse_scope') or {}
    budget_path = _bound_reference(scope.get('prior_budget'), origin)
    if budget_path != origin / 'energy/energy_task_budget.json':
        raise ValueError('Generic fresh performance prior budget belongs to another origin')
    prior = completed.read_json(budget_path)
    key = json.dumps(['generic_completed:' + expected['comparison_backend'], expected['model_id'],
                      expected['case_id'], expected['setup_id']], separators=(',', ':'))
    task = prior.get('tasks', {}).get(key) or {}
    source_id = str(prepared['target']['energy']['urecs_address']).strip().lower()
    if (prior.get('campaign') is not True or prior.get('max_transport_failures') != 20
            or task.get('source_id') != source_id
            or task.get('limits') != {'max_chains': 9, 'max_retries': 2, 'max_transport_failures': 20}):
        raise ValueError('Generic fresh performance prior energy policy/source differs')
    chain_fields = ('logical_repeat', 'run_directory', 'owner_pid', 'owner_start_ticks')
    chain_ids = [{k: chain.get(k) for k in chain_fields} for chain in task.get('chains') or []]
    if chain_ids != scope.get('prior_capture_chains'):
        raise ValueError('Generic fresh performance prior capture-chain identities differ')
    for chain in task['chains']:
        if (chain.get('finished') is not True or chain.get('logical_repeat') not in {'repeat:0', 'repeat:1', 'repeat:2'}
                or not Path(chain['run_directory']).resolve().is_relative_to(old_output)):
            raise ValueError('Generic fresh performance prior capture is not terminal or escapes its cell')
        if chain.get('valid') is True:
            raise ValueError('Generic performance-only continuation cannot recapture a prior accepted energy slot')
        if chain.get('source_completion_verified') is not True:
            _verify_source_release(scope, chain, prepared, origin, matrix.parent / 'hardware_runs')
    return {'case': row, 'projection': projection, 'runner_bundle': str(bundle),
        'proof': {'schema': 'onnx-splitpoint/hardware-acceptance-generic-performance-reuse/v1',
            'identity': expected, 'phase_matrix': {'path': str(matrix), 'sha256': completed.file_sha256(matrix)},
            'origin': {'run_id': origin.name, 'run_dir': str(origin), **copy.deepcopy(reference)},
            'retained_fresh_performance_processes': 3, 'new_performance_processes': 0,
            'old_campaign_or_cache_reused': False, 'normal_output_revalidation': 'passed',
            'generated_source_files_verified': len(names), 'prior_energy_chain_count': len(task['chains'])}}


def _verify_source_release(scope, chain, prepared, origin, phase_runs):
    """Bind normal recovery separately; retain the failed source close unchanged."""
    from .execution_binding import physical_dut_key
    recoveries = scope.get('source_recovery') or []
    if isinstance(recoveries, Mapping):
        recoveries = [recoveries]
    matches = []
    for descriptor in recoveries:
        request_path = _bound_reference(descriptor.get('request'), phase_runs)
        reply_path = _bound_reference(descriptor.get('reply'), phase_runs)
        request = completed.read_json(request_path)
        if request.get('attempt_dir') == chain['run_directory']:
            matches.append((request_path, reply_path, request, completed.read_json(reply_path)))
    if len(matches) != 1:
        raise ValueError('Generic prior unresolved source requires its exact normal recovery receipt')
    request_path, reply_path, request, reply = matches[0]
    recovery = reply.get('recovery') or {}
    identity_fields = ('run_id', 'session_id', 'operation_id', 'token')
    source = str(prepared['target']['energy']['urecs_address']).strip().lower()
    expected_resources = sorted([physical_dut_key(prepared['target']), 'source:' + source,
                                 'controller:capture', 'controller:nic'])
    if (not request_path.is_relative_to(origin) or not reply_path.is_relative_to(origin)
            or reply_path.name != request_path.name.removesuffix('.resource-request.json') + '.resource-reply.json'
            or request.get('purpose') != 'capture' or request.get('phase') != 'FINALIZING'
            or request.get('process_started') is not True or request.get('run_id') != origin.name
            or request.get('setup_id') != prepared['setup_id']
            or request.get('owner_pid') != chain.get('owner_pid')
            or request.get('owner_start_ticks') != chain.get('owner_start_ticks')
            or any(not request.get(k) or reply.get(k) != request[k] for k in identity_fields)
            or any(recovery.get(k) != request[k] for k in identity_fields[:-1])
            or reply.get('state') != 'STOP' or reply.get('reason') != 'campaign_source_completion_unresolved'
            or recovery.get('status') != 'released' or recovery.get('old_attempt_validated') is not False
            or recovery.get('measurements_started') != 0
            or (recovery.get('ownership_observation') or {}).get('ok') is not True
            or sorted(recovery.get('resources') or []) != expected_resources):
        raise ValueError('Generic prior source recovery does not prove exact normal physical release')


def _phase_energy_admission(prepared, context, plan, call):
    """Read existing phase budgets before each Single; never reset their history."""
    reference = prepared.get('fresh_performance_reference')
    if reference is None:
        if prepared.get('acceptance_mode') == 'functional':
            return _functional_energy_admission(prepared, context, plan, call)
        return None
    scope = reference['reuse_scope']
    _bound_reference(scope['prior_budget'])
    phase_matrix = Path(prepared['performance_reuse_proof']['phase_matrix']['path'])
    if completed.file_sha256(phase_matrix) != prepared['performance_reuse_proof']['phase_matrix']['sha256']:
        raise ValueError('Generic cumulative acceptance phase matrix changed')
    root = _root(context)
    row_id = plan['campaign_row_id']
    logical = 'repeat:' + str(call['repetition_index'] - 1)
    counts = dict.fromkeys(('repeat:0', 'repeat:1', 'repeat:2'), 0)
    evidence = []
    for directory in sorted(root.parent.iterdir()):
        path = directory / 'energy/energy_task_budget.json'
        if not path.is_file():
            continue
        budget = completed.read_json(path)
        task = (budget.get('tasks') or {}).get(row_id)
        if task is None:
            continue
        if (path.is_symlink() or budget.get('campaign') is not True or budget.get('max_transport_failures') != 20
                or task.get('source_id') != plan['campaign_source_id'] or task.get('limits') != plan['campaign_limits']):
            raise ValueError('Generic cumulative phase budget/source policy differs')
        if directory != root:
            from ..benchmark.evaluation_profiles import load_evaluation_profile
            manifest = completed.read_json(directory / 'run_manifest.json')
            closure = completed.read_json(directory / 'reports/artifact_index_closure.json')
            profile = load_evaluation_profile(directory / 'profile.yaml', validate=False).raw_profile
            matrix = Path((profile.get('hardware_acceptance') or {}).get('matrix_evidence') or '').resolve()
            if (matrix != phase_matrix or manifest.get('run_id') != directory.name
                    or manifest.get('status') not in {'ok', 'partial', 'failed', 'cancelled'}
                    or closure.get('status') != 'pass' or closure.get('run_id') != directory.name
                    or closure.get('workflow_status') != manifest['status']):
                raise ValueError('Generic cumulative predecessor is not a closed same-phase workflow')
        chains = task.get('chains') or []
        for chain in chains:
            position = chain.get('logical_repeat')
            if (position not in counts or chain.get('finished') is not True
                    or not Path(chain.get('run_directory') or '').resolve().is_relative_to(directory)):
                raise ValueError('Generic cumulative prior capture is active, unknown or outside its workflow')
            if chain.get('valid') is True and (directory != root or position == logical):
                raise RuntimeError('Generic continuation cannot recapture an accepted energy slot')
            if directory != root and chain.get('source_completion_verified') is not True:
                _verify_source_release(scope, chain, prepared, directory, phase_matrix.parent / 'hardware_runs')
            if directory == root and chain.get('source_completion_verified') is not True:
                raise RuntimeError('Generic current source completion is unresolved; normal recovery required')
            counts[position] += 1
        evidence.append({'path': str(path), 'sha256': completed.file_sha256(path), 'chains': len(chains),
                         'physical_collector_starts': sum(chain.get('collector_started') is True for chain in chains)})
    if sum(counts.values()) >= 9 or counts[logical] >= 3:
        raise RuntimeError('Generic cumulative phase chain/retry limit reached for ' + logical)
    return {'campaign_row_id': row_id, 'logical_repeat': logical, 'prior_and_current_chains': counts,
            'remaining_single_starts_for_slot': 3 - counts[logical], 'normal_single_internal_retries': 0,
            'remaining_single_starts_for_case': 9 - sum(counts.values()),
            'profile_max_retries': 2, 'profile_max_chains': 9, 'budget_evidence': evidence}


def _functional_energy_admission(prepared, context, plan, call):
    """Never give a failed functional capture fresh credit through a new run."""
    root = _root(context)
    if completed._acceptance_repetitions(plan) != 1 or call.get('repetition_index') != 1:
        raise ValueError('Functional Generic capture admission differs from its one-slot recipe')
    if (plan.get('campaign_row_id') != _campaign_key(prepared['selected_identity'])
            or plan.get('campaign_limits') != {'max_chains': 3, 'max_retries': 2, 'max_transport_failures': 20}
            or Path(plan['campaign_budget_file']).resolve() != root / 'energy/energy_task_budget.json'):
        raise ValueError('Functional Generic capture budget/selected row differs')
    evidence = []
    counts = 0
    for directory in sorted(root.parent.iterdir()):
        path = directory / 'energy/energy_task_budget.json'
        if not path.is_file():
            continue
        budget = completed.read_json(path)
        task = (budget.get('tasks') or {}).get(plan['campaign_row_id'])
        if task is None:
            continue
        if directory != root:
            if task.get('chains'):
                raise RuntimeError('Prior Generic capture requires reviewed continuation or accepted-slot reuse; preserve its budget')
            continue
        if (budget.get('campaign') is not True or task.get('limits') != plan['campaign_limits']
                or task.get('source_id') != plan['campaign_source_id']):
            raise ValueError('Functional Generic current campaign policy/source differs')
        chains = task.get('chains') or []
        for chain in chains:
            if (chain.get('logical_repeat') != 'repeat:0' or chain.get('finished') is not True
                    or chain.get('source_completion_verified') is not True
                    or not Path(chain.get('run_directory') or '').resolve().is_relative_to(root)):
                raise RuntimeError('Functional Generic previous capture is active or source completion unresolved')
            if chain.get('valid') is True:
                raise RuntimeError('Functional Generic accepted energy slot must not be recaptured')
        counts = len(chains)
        evidence.append({'path': str(path), 'sha256': completed.file_sha256(path), 'chains': counts,
                         'physical_collector_starts': sum(c.get('collector_started') is True for c in chains)})
    if not evidence or counts >= 3:
        raise RuntimeError('Functional Generic cumulative three-chain retry limit reached or budget absent')
    return {'campaign_row_id': plan['campaign_row_id'], 'logical_repeat': 'repeat:0',
        'prior_and_current_chains': {'repeat:0': counts}, 'remaining_single_starts_for_slot': 3 - counts,
        'remaining_single_starts_for_case': 3 - counts, 'normal_single_internal_retries': 0,
        'profile_max_retries': 2, 'profile_max_chains': 3, 'budget_evidence': evidence}


def prepare_generic(cell, context):
    """Resolve one selected source row and generate fresh normal configuration offline."""
    _profile_budget(context)
    mode, repetitions = _acceptance_recipe(context)
    reference = _performance_reference(cell)
    if reference is not None and mode == 'functional':
        raise ValueError('Functional Generic acceptance requires its dedicated one-process cohort')
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
    row.update(acceptance_mode=mode, requested_repetitions=repetitions)
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
        'acceptance_mode': mode, 'requested_repetitions': repetitions,
        'planned_repetitions': repetitions, 'planned_measured_completions': repetitions * 1000,
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
        'acceptance_mode': mode, 'requested_repetitions': repetitions,
        'model_id': row['model_id'], 'setup_id': row['setup_id'], 'selected_identity': row['identity'],
        'output_root': str(output), 'acceptance_root': str(root),
        'execution_plan': str(execution), 'execution_plan_sha256': completed.file_sha256(execution),
        'source_plan': {'path': str(source), 'sha256': source_sha},
        'configured_primary_run': str(configured_primary), 'selected_original_run': str(primary),
        'historical_result_reference': historical_result, 'references_reused': True,
        'runner_bundle': str(bundle), 'runner_bundle_sha256': completed.file_sha256(bundle),
        'registry_snapshot': str(registry_snapshot), 'target': target, 'resources': resources,
        'calibration': calibration, 'collector_binary': collector,
        'hardware_executed': False, 'performance_processes': repetitions, 'energy_captures': repetitions,
        'warmup_count': 100, 'task_count': 1000, 'duration_s': 60.,
        'performance_recipe': {'warmup_count': 100, 'task_count': 1000, 'repetitions': repetitions,
                               'process_repetition_policy': 'separate_processes'},
        'energy_recipe': {'warmup_count': 0, 'task_count': None, 'duration_s': 60., 'repetitions': repetitions,
                          'count_policy': 'all_admitted_tasks_completed_after_lossless_drain',
                          'capture_scope': 'whole_workload_command_including_initialization_and_postflight',
                          'physical_scope': 'FS', 'window_label': 'command',
                          'recipe_source': 'onnx_splitpoint_tool.runners.generic_energy.build_contract'},
        'build_allowed': False, 'quality_dataset_capture_requested': False}
    if reference is not None:
        retained = collect_generic_performance_case(prepared, context, reference)
        prepared.update(fresh_performance_reference=reference, performance_reuse_proof=retained['proof'],
                        new_performance_processes=0, retained_fresh_performance_processes=3,
                        performance_processes=0)
        completed.write_json(output / 'PERFORMANCE_REUSE.json', retained['proof'])
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
        EnergyTaskBudget(stream, {'max_chains': 3 * prepared.get('requested_repetitions', 3),
                                 'max_retries': 2, 'max_transport_failures': 20},
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
    mode, repetitions = _acceptance_recipe(context)
    if (prepared.get('acceptance_mode', 'scientific') != mode
            or prepared.get('requested_repetitions', 3) != repetitions):
        raise ValueError('Prepared Generic acceptance mode/recipe changed')
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
    reference = prepared.get('fresh_performance_reference')
    retained = collect_generic_performance_case(prepared, context, reference) if reference is not None else None
    if retained is not None and retained['proof'] != prepared['performance_reuse_proof']:
        raise ValueError('Prepared Generic retained performance proof changed')
    row = retained['case'] if retained is not None else plan['rows'][0]
    report_plan = {'rows': [row], 'performance_reuse': retained['proof']} if retained is not None else plan
    active_bundle = Path(retained['runner_bundle']) if retained is not None else bundle
    active_bundle_sha = completed.file_sha256(active_bundle) if retained is not None else prepared['runner_bundle_sha256']
    output = Path(prepared['output_root'])
    if retained is None and Path(row['result_path']).exists():
        raise FileExistsError('Fresh acceptance cannot reuse a historical or previous result')
    # The exclusive start is a product dispatch record, never a manual claim.
    with (output / 'production-start.json').open('x', encoding='utf-8') as stream:
        json.dump({'started_at': completed.now(), 'execution_id': plan['execution_id'],
                   'run_id': journal.scope.run_id, 'session_id': journal.scope.session_id,
                   'plan': str(plan_path), 'selected_identity': row['identity'],
                   'purpose': 'finite_fresh_hardware_acceptance',
                   'acceptance_mode': mode, 'requested_repetitions': repetitions,
                   'new_performance_processes': 0 if retained is not None else repetitions,
                   'retained_fresh_performance_origin': retained['proof']['origin'] if retained is not None else None}, stream, indent=2)
    log = context.get('log') or print
    raw_transport = completed.transport_for(prepared['target'], output / 'diagnostics',
        scope=journal.scope, registry=leases, cancel=context.get('cancel_event'), log=log)
    transport = ResourceTransport(raw_transport, setup=row['setup_id'], root=output / 'transfers')
    energy_plans = []
    phase_admissions = []
    def before_capture(normal_plan, call):
        evidence = _phase_energy_admission(prepared, context, normal_plan, call)
        phase_admissions.append(evidence)
        return evidence
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
            completed.upload_new_file(transport, active_bundle, row['remote_root'] + '/runner-bundle.tar.gz',
                                      expected_sha=active_bundle_sha)
            completed.checked_remote('tar -xzf ' + shlex.quote(row['remote_root'] + '/runner-bundle.tar.gz')
                                     + ' -C ' + shlex.quote(row['remote_root']), transport)
            phase = 'performance'
            if retained is None:
                completed.run_case(row, transport, output, 1, total=1)
            else:
                completed.stage_case(row, row['runtime_config'], row['staging_files'], transport,
                                     output / 'staging' / row['case_output_relative'])
            if mode == 'functional':
                actual, _, _, _ = _evidence(output, row, [])
                if actual['valid_repetition_count'] != 1 or actual['pending_repetition_indices'] != [2, 3]:
                    raise RuntimeError('Functional Generic performance lacks one normally validated process')
            completed.write_json(plan_path, plan)
            write_completion_reports(report_plan, output / 'results', output / 'reports')
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
                        process_registry=processes, cancel_event=context.get('cancel_event'),
                        before_capture=before_capture if retained is not None or mode == 'functional' else None)
                    if energy.get('normal_report_verified') is not True or energy.get('valid_repetitions') != repetitions:
                        raise RuntimeError('Normal Generic energy import did not verify the requested captures')
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
        write_completion_reports(report_plan, output / 'results', output / 'reports')
    projection, attempts, starts, chains = _evidence(output, row, energy_plans)
    if retained is not None:
        attempts = []
    captures = [chain for chain in chains if chain.get('collector_started') is True]
    process_reports = {rep.get('runtime_instance_id') or (rep.get('pid'), rep.get('process_start_ticks'))
                       for rep in projection.get('raw_attempts', [])
                       if isinstance(rep, dict) and type(rep.get('pid')) is int and rep['pid'] > 0}
    if retained is not None:
        process_reports = set()
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
    if retained is not None:
        paths.append(str(output / 'PERFORMANCE_REUSE.json'))
    result = {'status': 'completed' if phase == 'complete' else ('failed' if attempts or starts else 'blocked'),
        'acceptance_mode': mode, 'requested_repetitions': repetitions,
        'fresh_hardware_executed': bool(process_reports or captures), 'new_capture_count': len(captures),
        'collector_cli_attempts': len(starts), 'new_performance_dispatch_attempt_count': len(attempts),
        'new_performance_process_count': len(process_reports),
        'failed_capture_count': sum(chain.get('finished') is True and chain.get('valid') is not True for chain in captures),
        'unknown_capture_count': sum(chain.get('finished') is not True for chain in captures),
        'physical_execution_uncertain': bool((attempts and not process_reports) or (starts and not captures)),
        'valid_repeats': {'performance': projection['valid_repetition_count'], 'energy': energy_repeats},
        'failed_attempts': {'performance': sum(a.get('returncode') != 0 for a in attempts),
                            'energy': max(0, len(starts) - energy_repeats)},
        'output_validation': {'status': 'passed' if mode == 'functional' and projection['valid_repetition_count'] == 1 else projection['status'],
                              'scientific_status': projection['status'],
                              'procedure': 'normal_bound_functional_output_validation'},
        'performance': projection, 'energy': energy or {'status': 'incomplete', 'valid_repetitions': energy_repeats},
        'import_projection': {'status': 'completed' if energy is not None and energy.get('normal_report_verified') else 'incomplete',
                              'normal_report_verified': bool(energy and energy.get('normal_report_verified'))},
        'evidence_paths': paths, 'error': error, 'stopped_phase': '' if phase == 'complete' else phase,
        'references_reused': {'historical_performance_result': prepared['historical_result_reference'],
                              'native_functional_input_and_quality_reference': row['native_source']},
        'new_dataset_quality_capture': False}
    if mode == 'functional':
        report = {'schema': 'onnx-splitpoint/generic-functional-acceptance/v1',
            'acceptance_mode': mode, 'requested_performance_processes': 1, 'requested_energy_captures': 1,
            'status': result['status'], 'identity': row['identity'],
            'actual_valid_performance_processes': projection['valid_repetition_count'],
            'actual_valid_energy_captures': energy_repeats,
            'performance_result': {'path': row['result_path'],
                'sha256': completed.file_sha256(Path(row['result_path'])) if Path(row['result_path']).is_file() else None},
            'normal_performance_projection': projection, 'normal_energy_projection': result['energy'],
            'output_oracle': projection['repetitions'][0]['functional_control'] if projection['valid_repetition_count'] == 1 else {},
            'output_samples': projection['repetitions'][0]['raw_output_samples'] if projection['valid_repetition_count'] == 1 else {},
            'completed_result_artifact': projection['repetitions'][0]['completed_result_artifact'] if projection['valid_repetition_count'] == 1 else {},
            'functional_reference': row['runtime_config']['functional_reference'],
            'scientific_cohort_complete': False, 'scientific_claim_eligible': False,
            'scientific_statistics_published': False}
        path = output / 'FUNCTIONAL_ACCEPTANCE.json'
        completed.write_json(path, report)
        result['evidence_paths'].append(str(path))
        result.update(functional_acceptance=True, scientific_cohort_complete=False,
                      scientific_claim_eligible=False, scientific_statistics_published=False,
                      evaluation_role='functional_acceptance', diagnostic_only=True,
                      phase_energy_admissions=phase_admissions)
    if retained is not None:
        prior = completed.read_json(_bound_reference(reference['reuse_scope']['prior_budget']))
        prior_chains = prior['tasks'][_campaign_key(row['identity'])]['chains']
        result.update(retained_fresh_performance_processes=3, performance_origin=retained['proof']['origin'],
                      new_performance_process_count=0, new_performance_dispatch_attempt_count=0,
                      previous_energy_attempts={'chains': len(prior_chains),
                          'physical_collector_starts': sum(c.get('collector_started') is True for c in prior_chains),
                          'valid': sum(c.get('valid') is True for c in prior_chains),
                          'failed': sum(c.get('finished') is True and c.get('valid') is False for c in prior_chains)},
                      performance_reused_by_reference=True, reused_current_phase_performance=True,
                      phase_energy_admissions=phase_admissions, old_performance_raw_copied=False)
    completed.write_json(output / 'generic_result.json', result)
    return result
