"""The two explicitly authorized YOLO remainder blocks, using normal consumers.

This does not reopen any historical coordinator or performance checkpoint.
"""
from __future__ import annotations

from contextlib import contextmanager, ExitStack
from pathlib import Path
from types import SimpleNamespace
import copy
import json
import os
import re
import shutil
import signal
import sys
import threading
import traceback

from . import generic_completed_task as io
from . import yolo_augmentation as augmentation
from ..resources.remote_scripts import run_evalrun_native_producer_variants as coordinator

SPECIAL = ('yolov7_paper', 'b066', 'orin_nx_hailo8_01')
H8_REMAINING_ENERGY = {('yolo26s', 'b038', 'orin_nx_hailo8_01'),
                       ('yolov7_paper', 'b044', 'orin_nx_hailo8_01')}
SOURCE = Path(__file__).resolve().parents[2]


def remaining_inventory(plan, *, energy_continuation=False):
    """Use actual observations, never historical terminal status as a success."""
    from .supplement import completed_quality
    from ..generic_completion_reporting import load_valid_repetitions, validate_completion_case, completion_pair
    from ..native_rate_endpoints import rate_endpoint_fields
    from ..native_command_contract import verify_native_command_contract
    from ..resources.remote_scripts.native_producer_final_report import _repeat_fields, _validate_repeat_claim_evidence, _aggregate_repetitions
    run = Path(plan['augmentation_run'])
    allowed = {augmentation.key(row) for row in plan['rows'] if row['selection_role'] == 'primary'}
    native = io.read_json(run / 'reports/native_producer_combined_summary.json')
    rows = native['rows']
    if len(rows) != 12 or {augmentation.key(row) for row in rows} != allowed:
        raise ValueError('rest Native identity scope differs from the frozen twelve')
    successful = {}
    for row in rows:
        if row.get('ok') is not True:
            continue
        source = Path(row['report']).resolve()
        if not source.is_relative_to(run.resolve() / 'native_producers'):
            raise ValueError('preserved Native observation lies outside the original run')
        raw = io.read_json(source)
        command, reason = verify_native_command_contract(raw.get('native_command_contract'), expected_identity={
            field: row.get(field) for field in ('model', 'case', 'setup_id', 'backend', 'precision')})
        repeats = _repeat_fields((raw.get('endpoint_results') or {}).get('completed_task') or raw, raw)
        valid, errors = _validate_repeat_claim_evidence(repeats)
        rates = rate_endpoint_fields(raw)
        if (command is None or command.get('eval_run_id') != run.name or raw.get('ok') is not True
                or not valid or raw.get('repetition_count_valid') != 3
                or rates.get('completed_task_work_unit_counts') != [1000] * 3
                or rates.get('fps_repetition_samples') != row.get('fps_repetition_samples')
                or _aggregate_repetitions([{**row, **repeats}])[0]['repetition_records'] != row.get('repetition_records')):
            raise ValueError('rest Native repetitions are not bound: ' + str(reason or errors))
        successful[augmentation.key(row)] = row
    if set(successful) != allowed - {SPECIAL}:
        raise ValueError('rest entry requires exactly the eleven preserved Native cases')
    quality, _ = completed_quality(run)
    observed = {(r['model_id'], r['case_id'], r.get('source_setup_id') or r.get('setup_id')) for r in quality}
    if len(quality) != 12 or observed != allowed:
        raise ValueError('rest entry requires twelve completed original Quality requests')
    execution = io.read_json(Path(plan['output_root']) / 'completion/execution-plan.json')
    completed = set()
    for row in execution['rows']:
        case = augmentation.key(row)
        if case not in successful or case in completed:
            raise ValueError('rest completion identity is foreign or duplicated')
        path = Path(row['result_path'])
        if len(load_valid_repetitions(path, row)) != 3:
            raise ValueError('rest entry cannot replace a partial Generic attempt')
        observation = validate_completion_case(io.read_json(path), row, result_root=path.parent)
        if not completion_pair(observation, row)['eligible_for_technical_transfer']:
            raise ValueError('preserved rest completion pair is technically invalid')
        completed.add(case)
    starts = {augmentation.key(r) for r in plan['rows'] if r['case_key'] in plan['start_case_keys']}
    required_pairs = set(successful) if energy_continuation else starts
    if completed != required_pairs or len(completed) != (11 if energy_continuation else 3):
        raise ValueError('rest entry requires its complete preserved Generic pairs')
    if (run / 'reports/native_energy_measurements/native_producer_energy_results.json').exists():
        raise ValueError('rest entry will not silently repeat existing canonical energy')
    return {'native_cases': 11, 'native_repetitions': 33, 'generic_cases': len(completed), 'generic_repetitions': 3 * len(completed),
            'quality_results': 12, 'block1_generic': sorted(set(successful) - completed),
            'block1_energy': sorted(successful), 'block2_identity': list(SPECIAL)}


def energy_command(run, cfg, summary, validation, block, *, previous_results=None):
    """Exactly the normal coordinator's duration, budget and campaign binding."""
    ecfg = cfg.get('energy') or {}
    contract = coordinator._energy_contract_context(cfg, ecfg, run_dir=run, reports=block / 'reports')
    runs = coordinator._energy_repeat_count(cfg, ecfg)
    if runs != 3 or contract['physical_scope'] != 'FS' or contract['window_label'] != 'command':
        raise ValueError('rest energy differs from the existing FS/command/three-repetition contract')
    cmd = [sys.executable, '-u', str(coordinator._script('run_native_producer_energy_from_summary.py')),
        '--summary', str(summary), '--validation-summary', str(validation),
        '--out-dir', str(block / 'reports/native_energy_measurements'),
        *coordinator._common_energy_args(cfg, run_dir=run, duration_s=60.0, timeout_s=900), '--runs', str(runs)]
    for field in ('physical_scope', 'window_label', 'calibration_manifest', 'calibration_sha256',
                  'pipeline_contract_manifest', 'pipeline_contract_sha256', 'model_hash_map', 'model_hash_map_sha256'):
        cmd += ['--' + field.replace('_', '-'), str(contract[field])]
    cmd += coordinator._energy_window_ab_cli_args(cfg, ecfg)
    cmd += ['--measure-all-runtime-successful']
    if contract['final_energy_contract_enforced']:
        cmd += ['--require-runtime-work-units', '--require-command-window-alignment']
    else:
        cmd += ['--screening-energy']
    if previous_results is not None:
        cmd += ['--previous-results', str(previous_results)]
    return cmd


def energy_continuation(plan, previous_attempt, *, h8_reentry=False):
    """Admit the stored eleven-pair state through the existing supplement reader."""
    from ..native_energy_reporting import native_energy_results_path
    from ..resources.remote_scripts.run_native_producer_energy_from_summary import (
        _supplement_previous_results, _canonical_resume_execution_context, _RESUME_EXECUTION_CONTEXT_SCHEMA)
    previous = augmentation.execution_output_path(plan, previous_attempt)
    if previous == Path(plan['output_root']).resolve():
        raise ValueError('energy continuation requires its explicit predecessor attempt')
    start = io.read_json(previous / 'production-start.json')
    terminal = io.read_json(previous / 'terminal-status.json')
    if (Path(start.get('plan', '')).resolve() != Path(plan['output_root']).resolve() / 'plan/augmentation-plan.json'
            or start.get('automatic_restart') is not False or terminal.get('exitcode') != 1
            or terminal.get('status') != 'stopped' or not terminal.get('finished_at')):
        raise ValueError('energy predecessor is not the terminal failed rest attempt')
    block = Path(plan['augmentation_run']) / 'reports/rest_attempts' / previous.name / 'block1'
    result_path = native_energy_results_path(block, include_partial=True)
    if result_path is None or not result_path.is_file():
        raise ValueError('energy predecessor has no actual final or partial evidence')
    payload = io.read_json(result_path)
    candidate = copy.deepcopy(payload.get('plan_payload') or {})
    candidate['execution_context'] = {'schema': _RESUME_EXECUTION_CONTEXT_SCHEMA, 'schema_version': 1,
        **_canonical_resume_execution_context(payload)}
    reused = _supplement_previous_results(payload, candidate)
    inventory = remaining_inventory(plan, energy_continuation=True)
    planned = (payload.get('plan_payload') or {}).get('rows') or []
    if (len(planned) != 11 or {augmentation.key(row) for row in planned} != set(map(tuple, inventory['block1_energy']))
            or len(reused) != (9 if h8_reentry else 6)):
        raise ValueError('energy continuation does not preserve its exact prior energy cohort')
    if h8_reentry and ({augmentation.key(row) for row in planned}
                       - {augmentation.key(item['row']) for item in reused.values()} != H8_REMAINING_ENERGY):
        raise ValueError('authorized H8 continuation has a different open energy scope')
    return {'previous_attempt': str(previous), 'previous_block': str(block), 'previous_results': str(result_path),
            'reused_energy_rows': len(reused), 'remaining_energy_rows': 11 - len(reused), 'inventory': inventory}


def singlecase_command(run, cfg, output):
    from .checkpoints import canonical_json_sha256, load_stage_checkpoint
    stage = io.read_json(run / 'reports/native_producer_stage.json')
    matches = [r for r in stage['variant_results'] if r.get('case_map') == {'yolov7_paper': ['b066']}
               and '--backends' in r.get('cmd', []) and r['cmd'][r['cmd'].index('--backends') + 1] == 'hailo8']
    if len(matches) != 1 or matches[0].get('ok') is True:
        raise ValueError('rest singlecase predecessor is not the one failed H8/b066 leaf')
    record = matches[0]
    path = run / 'reports/native_producer_variants' / record['artifact_namespace'] / 'performance_stage_result.json'
    leaf = io.read_json(path)
    ancestor_path = Path(stage['native_performance_checkpoint']['path']).parent.parent / 'native_coordinator/stage_result.json'
    if not ancestor_path.resolve().is_relative_to(run.resolve() / 'jobs/supplement_attempts'):
        raise ValueError('failed leaf coordinator lies outside the run')
    ancestor_raw = io.read_json(ancestor_path)
    ancestor, reason = load_stage_checkpoint(ancestor_path, stage='native_coordinator', input_hash=ancestor_raw['input_hash'], run_root=run)
    if not ancestor or ancestor.get('complete') is not True or ancestor['state'] != 'failed':
        raise ValueError('rest predecessor coordinator invalid: ' + reason)
    expected = canonical_json_sha256({'coordinator_input_hash': ancestor['input_hash'],
        'variant': leaf['details']['variant'], 'cmd': record['cmd']})
    checked, reason = load_stage_checkpoint(path, stage='native_case_performance', input_hash=expected, run_root=run)
    if not checked or checked['state'] != 'failed' or checked.get('complete') is not True:
        raise ValueError('rest singlecase predecessor checkpoint invalid: ' + reason)
    variant = leaf['details']['variant']
    command = coordinator._build_update_cmd(run, stage['config'], variant, refresh_suites=False, timeout_s=7200)
    if command != record['cmd'] or command != leaf['details']['execution']['cmd']:
        raise ValueError('rest singlecase original invocation changed')
    namespace = 'case_rest_' + output.name
    if len(namespace) > 160 or not re.fullmatch(r'[A-Za-z0-9_.-]+', namespace):
        raise ValueError('unsafe rest artifact namespace')
    if (run / 'native_producers/variants' / namespace).exists() or (run / 'reports/native_producer_variants' / namespace).exists():
        raise FileExistsError('rest Native namespace already exists; no automatic retry')
    command[command.index('--artifact-namespace') + 1] = namespace
    # Eleven historical command contracts retain their exact old runner paths.
    remote = command[command.index('--remote-tool-dir') + 1].rstrip('/')
    command[command.index('--remote-tool-dir') + 1] = remote + '/attempts/' + output.name
    reference = cfg['augmentation_startcase_barrier']['original_reference_validation_summary']
    command += ['--existing-outputs-only', '--existing-reference-validation-summary', str(reference)]
    return command, namespace


def _recover_rest_run_fence(plan, previous, output, lock, fence, *, source_test=False, h8_reentry=False):
    """Only clear the run fence after the normal exact physical recovery."""
    from ..process_control import _proc_start_time
    from ..remote.resource_recovery import _rest_capture_journal, _rest_evidence_unchanged
    run = Path(plan['augmentation_run']).resolve(); previous = Path(previous).resolve()
    owner = lock.previous_owner
    identity = {'run_id': run.name, 'session_id': previous.name}
    cleanup = io.read_json(previous / 'cleanup-status.json')
    directory, start, rest_evidence = _rest_capture_journal(run, previous.name, previous)
    config = io.read_json(directory / 'controller.resources.json')
    if (fence.get('schema') != 'onnx-splitpoint/evaluation-run-lock-quarantine'
            or fence.get('run_dir') != str(run) or fence.get('run_id') != run.name
            or (fence.get('owner') or {}).get('reason') != 'rest_cleanup_unproven'
            or (fence.get('owner') or {}).get('pending') != 0
            or any(owner.get(k) != v or config.get(k) != v for k, v in identity.items())
            or owner.get('purpose') != 'yolo_rest_completion' or owner.get('pid') != config.get('parent_pid')
            or start.get('pid') != config.get('parent_pid')
            or config.get('run_dir') != str(run)
            or cleanup.get('active_owned_remote_operations') != 0 or cleanup.get('owned_local_processes') != 0
            or cleanup.get('broker_errors') != (fence.get('owner') or {}).get('errors')):
        raise ValueError('rest run quarantine has no exact predecessor cleanup binding')
    if list(directory.glob('*.remote-lease.json')):
        raise ValueError('rest predecessor still has unresolved remote ownership')
    replies = [io.read_json(p) for p in directory.glob('*.resource-reply.json')]
    stops = [reply for reply in replies if reply.get('state') != 'RELEASED']
    if len(stops) != 1:
        raise ValueError('rest predecessor does not have exactly its one source STOP')
    reply = stops[0]; operation = reply.get('operation_id')
    if not isinstance(operation, str) or not re.fullmatch(r'capture-[a-f0-9]{32}', operation):
        raise ValueError('invalid rest stopped operation identity')
    identity['operation_id'] = operation
    request = io.read_json(directory / (operation + '.resource-request.json'))
    recovery = reply.get('recovery') or {}
    resources = sorted(r for r in config.get('setups', {}).get(request.get('setup_id'), [])
                       if r.startswith(('dut:', 'source:')) or r in {'controller:nic', 'controller:capture'})
    if (reply.get('state') != 'STOP' or reply.get('reason') != 'campaign_source_completion_unresolved'
            or any(any(value.get(k) != v for k, v in identity.items()) for value in (reply, request, recovery))
            or reply.get('token') != request.get('token') or recovery.get('status') != 'released'
            or len(resources) != 4 or recovery.get('resources') != resources
            or not recovery.get('released_at') or recovery.get('measurements_started') != 0
            or (recovery.get('ownership_observation') or {}).get('ok') is not True
            or not (bool(recovery.get('operator_evidence')) ^ bool(recovery.get('retry_authorization')))):
        raise ValueError('rest physical source recovery is not released and exactly bound')
    for pid, ticks in ((config.get('parent_pid'), config.get('parent_start_ticks')),
                       (request.get('owner_pid'), request.get('owner_start_ticks')),
                       (request.get('collector_pid'), request.get('collector_start_ticks'))):
        if type(pid) is not int or pid <= 0 or not str(ticks or '').isdigit() or str(_proc_start_time(pid) or '') == str(ticks):
            raise ValueError('rest predecessor local owner is active or unbound')
    collector_cleanup = io.read_json(Path(request['attempt_dir']) / 'collector_stdout.log.cleanup.json')
    if (collector_cleanup.get('owned_tree_quiescent') is not True
            or any(collector_cleanup.get(k) != request.get(k) for k in ('collector_pid', 'collector_start_ticks'))):
        raise ValueError('rest collector cleanup is not bound')
    with ExitStack() as held:
        for resource in resources:
            original = (recovery.get('original_resources') or {}).get(resource) or {}
            old_owner = original.get('lock_owner') or {}; old_fence = original.get('quarantine') or {}
            lease = io.EvaluationRunLock.for_resource(resource, owner={})
            if source_test or h8_reentry:
                lease.acquire_for_recovery()
                held.callback(lease.release, write_released_state=False)
            current = io.read_json(lease.lock_path)
            if (any(old_owner.get(k) != v for k, v in identity.items())
                    or old_owner.get('physical_resource') != resource
                    or (old_fence.get('owner') or {}).get('resource_request') != request
                    or (old_fence.get('owner') or {}).get('reason') != 'campaign_source_completion_unresolved'
                    or lease.quarantine_path.exists() or current.get('state') != 'released'
                    or any((current.get('resource_recovery') or {}).get(k) != v for k, v in identity.items())
                    or (current.get('resource_recovery') or {}).get('resources') != resources):
                raise ValueError('rest physical recovery ownership or fence changed')
        if not _rest_evidence_unchanged(rest_evidence):
            raise ValueError('rest predecessor evidence changed during run fence admission')
        if source_test or h8_reentry:
            from ..energy.task_budget import install_authorized_source_reentry
            if not recovery.get('retry_authorization'):
                raise ValueError('source reentry requires the explicit one-attempt retry authorization')
            options = {}
            if h8_reentry:
                # Resource identities above are bound to the original broker,
                # released capture and physical owners; addresses are not policy.
                if request.get('setup_id') != 'orin_nx_hailo8_01':
                    raise ValueError('authorized H8 retry limits require the exact H8 source recovery')
                options['authorized_limits'] = {
                    json.dumps(['hailo8_to_trt', model, case, setup], separators=(',', ':')):
                        {'max_retries': 5, 'max_chains': 18, 'max_transport_failures': 20}
                    for model, case, setup in sorted(H8_REMAINING_ENERGY | {SPECIAL})}
            handoff = install_authorized_source_reentry(run / 'energy_task_budget.json',
                capture_request=request, recovery_reply=reply,
                new_parent={'run_id': run.name, 'session_id': output.name,
                            'pid': os.getpid(), 'start_ticks': _proc_start_time(os.getpid())}, **options)
            io.write_json(output / 'reports/source-budget-handoff.json', handoff)
    io.write_json(output / 'reports/rest-runlock-recovery.json', {'original_run_fence': fence,
        'original_run_owner': owner, 'physical_recovery_reply': reply,
        'previous_attempt': str(previous), 'recovered_by_attempt': str(output), 'at': io.now()})
    return True


@contextmanager
def owned_execution(plan, output, cfg, *, previous_attempt=None, source_test=False, h8_reentry=False):
    """Existing run lock, physical broker, journal and process cancellation."""
    from ..native_progress import run_streaming
    from ..process_control import ProcessTreeRegistry
    from ..remote.process_lease import ControllerResourceBroker, RemoteProcessLeaseRegistry, RemoteProcessLeaseScope
    from ..quality_service import ResourcePauseGate
    from ..energy.config import load_hardware_registry, energy_setup_from_registry
    from .runner import EvaluationWorkflowRunner
    run = Path(plan['augmentation_run']); session = output.name
    cancel = threading.Event(); stop = threading.Event(); errors = []
    registry = RemoteProcessLeaseRegistry(); processes = ProcessTreeRegistry()
    targets = io.bound_targets(run, [r for r in plan['rows'] if r['selection_role'] == 'primary'])
    owner = {'run_id': run.name, 'session_id': session, 'purpose': 'yolo_rest_completion'}
    lock = io.EvaluationRunLock(out_root=run.parent, run_dir=run, owner=owner)
    recovery = ((lambda fence: _recover_rest_run_fence(plan, previous_attempt, output, lock, fence,
                                                     source_test=source_test, h8_reentry=h8_reentry))
                if previous_attempt is not None else None)
    lock.acquire(recover_quarantine=recovery)
    try:
        if (source_test or h8_reentry) and not (output / 'reports/source-budget-handoff.json').is_file():
            raise ValueError('authorized source test lacks its exact predecessor budget handoff')
        if (output / 'production-start.json').exists():
            raise FileExistsError('rest attempt already started; no automatic restart')
        current_control = io.read_json(run / 'jobs/workflow_control.json')
        if (current_control.get('run_id') == run.name and current_control.get('session_id') == session
                and current_control.get('cancel_requested') is True):
            raise RuntimeError('current rest attempt is cancelled')
        io.observe_owners(targets, output / 'production-preflight')
        journal = registry.configure_journal(scope=RemoteProcessLeaseScope(run.name, session), journal_dir=output / 'remote-process-leases')
        registry_path = Path(cfg['_workflow_context']['hardware_setups_file'])
        hardware = load_hardware_registry(registry_path)
        resources = {}
        for setup_id, target in targets.items():
            setup = energy_setup_from_registry(hardware, setup_id, registry_path=registry_path)
            resources[setup_id] = [io.physical_dut_key(target), 'source:' + str(setup.urecs_address).strip().lower(),
                                  'controller:cpu', 'controller:io', 'controller:nic', 'controller:capture']
        broker = ControllerResourceBroker(journal, ResourcePauseGate(), run_dir=run, setups=resources)
        env = {**os.environ, **registry.journal_environment(), 'ONNX_SPLITPOINT_CONTROLLER_RESOURCES': 'required'}
        control = SimpleNamespace(_control_lock=threading.RLock(), _run_control_write_enabled=True,
            _terminal_sealing=False, _run_lock=lock, run_id=run.name, run_dir=run, session_id=session,
            _control_state='running', _control_reason='explicit_two_block_rest', _cancel_event=cancel)
        old_control = run / 'jobs/workflow_control.json'
        if old_control.is_file(): shutil.copy2(old_control, output / 'previous-workflow-control.json')
        EvaluationWorkflowRunner._write_workflow_control(control)
        def cancelled():
            current = io.read_json(old_control)
            if current.get('session_id') == session and current.get('run_id') == run.name and current.get('cancel_requested') is True:
                cancel.set()
            return cancel.is_set() or registry.cancelled or bool(errors)
        def pump():
            while not stop.wait(0.025):
                try:
                    broker.tick()
                    if broker.stopped_sources: raise RuntimeError('normal controller broker left a stopped source')
                except BaseException as exc:
                    errors.append(str(exc)); cancel.set(); return
        thread = threading.Thread(target=pump, name='yolo-rest-controller', daemon=True); thread.start()
        handlers = {signum: signal.signal(signum, lambda *_: cancel.set()) for signum in (signal.SIGINT, signal.SIGTERM)}
        def execute(command, label, timeout=21600, setup=None, attempt=None, extra_env=None):
            if cancelled(): raise RuntimeError('rest cancelled before next dispatch')
            local_env = dict(env)
            local_env.update(extra_env or {})
            if setup:
                local_env.update(ONNX_SPLITPOINT_RESOURCE_SETUP=setup, ONNX_SPLITPOINT_RESOURCE_ATTEMPT=str(attempt))
            io.write_json(output / 'commands' / (label + '.json'), {'argv': command, 'started_at': io.now()})
            io.write_json(output / 'progress.json', {'phase': label, 'updated_at': io.now(), 'measurement_success_not_implied': True})
            with (output / 'commands' / (label + '.log')).open('w') as log:
                def line(text):
                    log.write(text + '\n'); log.flush(); print(text, flush=True)
                result = run_streaming(command, cwd=SOURCE, env=local_env, timeout=timeout,
                    label=label, line_callback=line, cancel_event=cancel, should_cancel=cancelled,
                    process_registry=processes, before_terminate=lambda: registry.cancel_all(grace_s=3))
            io.write_json(output / 'commands' / (label + '-result.json'), {'returncode': result.returncode,
                'stdout': result.stdout, 'stderr': result.stderr, 'finished_at': io.now()})
            if result.returncode or cancelled():
                detail = (result.stdout + '\n' + result.stderr).strip()[-6000:]
                message = f'{label} failed rc={result.returncode}: {detail}'
                io.write_json(output / 'primary-failure.json', {'phase': label, 'returncode': result.returncode,
                    'error': message, 'source': str(output / 'commands' / (label + '-result.json')), 'at': io.now()})
                raise RuntimeError(message)
            return result
        primary_error = None
        try:
            yield execute, env, registry, cancel
        except BaseException as exc:
            primary_error = exc
            raise
        finally:
            try:
                if cancel.is_set() or registry.active_count(): registry.cancel_all(grace_s=3)
                if processes.remaining_owned_count(): processes.terminate_all(grace_s=3)
            except BaseException as exc:
                errors.append('owned process cleanup: ' + str(exc))
            stop.set(); thread.join(timeout=5)
            try:
                if not thread.is_alive(): broker.close()
            except BaseException as exc:
                errors.append('broker cleanup: ' + str(exc))
            pending = registry.active_count() + processes.remaining_owned_count()
            if broker.stopped_sources: errors.append('normal broker retained stopped sources: ' + repr(sorted(broker.stopped_sources)))
            if pending or errors or thread.is_alive():
                lock.commit_quarantine_fence({'reason': 'rest_cleanup_unproven', 'pending': pending, 'errors': errors})
            control._control_state = 'cancelled' if cancel.is_set() else 'finished'
            EvaluationWorkflowRunner._write_workflow_control(control)
            for signum, handler in handlers.items(): signal.signal(signum, handler)
            io.write_json(output / 'cleanup-status.json', {'at': io.now(), 'active_owned_remote_operations': registry.active_count(),
                'owned_local_processes': processes.remaining_owned_count(), 'broker_errors': errors, 'foreign_processes_stopped': False,
                'primary_error': str(primary_error) if primary_error is not None else '',
                'cleanup_error': 'rest cleanup is not proven' if pending or errors or thread.is_alive() else ''})
            if (pending or errors or thread.is_alive()) and primary_error is None:
                raise RuntimeError('rest cleanup is not proven')
    finally:
        lock.release()


def _copy_validation(source, block):
    path = block / 'reports/native_validation/native_producer_validation_summary.json'
    path.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, path)
    return path


def source_test_command(run, cfg, output, diagnostic):
    """The existing GUI energy-test consumer, one budgeted diagnostic capture."""
    from ..energy.task_budget import campaign_budget_policy
    policy = campaign_budget_policy(cfg)
    if not policy:
        raise ValueError('authorized source test requires the original campaign budget')
    setup_id = 'orin_nx_hailo10_01'
    row_id = json.dumps(['energy_setup_test', output.name, setup_id], separators=(',', ':'))
    return [sys.executable, '-u', str(coordinator._script('energy_measurement_cli.py')), 'test-setup',
        '--setup-id', setup_id, '--hardware-setups-file', str(cfg['_workflow_context']['hardware_setups_file']),
        '--acquire', '--sleep-s', '2', '--out', str(diagnostic),
        '--campaign-budget-file', str(run / 'energy_task_budget.json'), '--campaign-row-id', row_id,
        '--campaign-repeats', '1', '--campaign-max-retries', '0',
        '--campaign-max-transport-failures', str(policy['max_transport_failures'])], row_id


def _verify_source_test(run, cfg, output, diagnostic, row_id, guard):
    from ..energy.collector import (_energy_repeat_contract_complete, _command_window_binding,
                                    _read_yaml, _read_workload_timing)
    from ..energy.task_budget import source_completion
    from ..energy.config import load_hardware_registry, energy_defaults_from_registry, energy_setup_from_registry
    registry_path = Path(cfg['_workflow_context']['hardware_setups_file'])
    hardware = load_hardware_registry(registry_path)
    defaults = energy_defaults_from_registry(hardware)
    setup_id = 'orin_nx_hailo10_01'
    setup = energy_setup_from_registry(hardware, setup_id, registry_path=registry_path)
    result = io.read_json(diagnostic / 'energy_summary.json'); rows = result.get('runs') or []
    if (result.get('ok') is not True or result.get('run_count') != 1
            or result.get('command') != 'sleep 2' or result.get('setup_id') != setup_id
            or result.get('urecs_address') != setup.urecs_address
            or result.get('diagnostic_only') is not True
            or result.get('invalid_repeat_max_retries') != 0 or result.get('repeat_retry_attempt_count') != 0
            or (result.get('energy_window_method_ab') or {}).get('exact_run_count_requested') is not True
            or not _energy_repeat_contract_complete(rows, requested_count=1, require_ab=False)):
        raise ValueError('source diagnostic did not complete its one normal valid capture')
    attempt = diagnostic / 'run_000'; row = rows[0]
    binding = row.get('command_window_trace_binding') or {}
    marker = Path((row.get('command_window_request') or {}).get('marker_path') or '')
    started = io.read_json(guard).get('entries') or []
    expected_options = [f'--address={setup.urecs_address}', f'--data-port={setup.data_port}',
                        f'--channel={defaults.channel}', f'--sample-rate={defaults.sample_rate}']
    if (row.get('collector_rc') != 0 or row.get('collector_started') is not True
            or Path(row.get('storage_dir') or '').resolve().parent != attempt.resolve()
            or row.get('energy_configured_workload_duration_s') != 2.0
            or not all(value in row.get('collector_cmd', []) for value in expected_options)
            or binding.get('verified') is not True or not marker.is_file()
            or not marker.resolve().is_relative_to(attempt.resolve())
            or binding.get('marker_sha256') != io.file_sha256(marker)
            or started != [r for r in started if r.get('kind') == 'collector' and r.get('identity') == str(attempt.resolve())]
            or len(started) != 1 or not source_completion(attempt).get('verified')):
        raise ValueError('source diagnostic sample, single-start or closure evidence is incomplete')
    markers = io.read_json(marker); stream = markers.get('stream') or {}
    if (markers.get('valid_for_final_energy') is not True or stream.get('sample_rate_hz') != defaults.sample_rate
            or type(stream.get('total_samples')) is not int or stream['total_samples'] <= 0
            or stream.get('dropped_samples_in_window') != 0):
        raise ValueError('source diagnostic has no valid sample window')
    results_path = Path(row.get('results_yaml') or '')
    verified = _command_window_binding(attempt, _read_yaml(results_path) or {},
        _read_workload_timing(attempt / 'workload_timing.txt'), result_path=results_path)
    if verified.get('verified') is not True:
        raise ValueError('source diagnostic stored sample/marker/result binding no longer verifies')
    budget = io.read_json(run / 'energy_task_budget.json'); task = budget.get('tasks', {}).get(row_id) or {}
    chains = task.get('chains') or []
    if (len(chains) != 1 or task.get('stop_reason') or task.get('source_id') != setup.urecs_address
            or any(chains[0].get(field) is not True for field in ('collector_started', 'finished', 'valid', 'source_completion_verified'))
            or Path(chains[0].get('run_directory') or '').resolve() != attempt.resolve()):
        raise ValueError('source diagnostic is not bound to its original campaign budget')
    journal = output / 'remote-process-leases'
    requests = [io.read_json(p) for p in journal.glob('*.resource-request.json')]
    captures = [r for r in requests if r.get('purpose') == 'capture'
                and Path(r.get('attempt_dir') or '').resolve().is_relative_to(diagnostic.resolve())]
    if len(captures) != 1:
        raise ValueError('source diagnostic capture ownership is not unique')
    request = captures[0]; reply = io.read_json(journal / (request['operation_id'] + '.resource-reply.json'))
    cleanup = io.read_json(attempt / 'collector_stdout.log.cleanup.json')
    if (request.get('run_id') != run.name or request.get('session_id') != output.name
            or request.get('setup_id') != setup_id or Path(request['attempt_dir']).resolve() != attempt.resolve()
            or reply.get('state') != 'RELEASED'
            or any(reply.get(k) != request.get(k) for k in ('operation_id', 'token', 'run_id', 'session_id'))
            or cleanup.get('owned_tree_quiescent') is not True
            or any(cleanup.get(k) != request.get(k) for k in ('collector_pid', 'collector_start_ticks'))):
        raise ValueError('source diagnostic resource release is not proven')
    proof = {'status': 'passed', 'diagnostic_only': True, 'physical_starts': 1, 'rest_started': False,
        'summary': str(diagnostic / 'energy_summary.json'), 'marker': str(marker),
        'source_completion': source_completion(attempt), 'resource_reply': reply,
        'sample_count': stream['total_samples'], 'campaign_diagnostic_row': row_id, 'at': io.now()}
    io.write_json(output / 'reports/source-test-acceptance.json', proof)
    return proof


def _run_source_test(run, cfg, output, work, execute, registry):
    diagnostic = work / 'source_test'
    command, row_id = source_test_command(run, cfg, output, diagnostic)
    guard = output / 'source-test-starts.json'
    with guard.open('x') as stream:
        json.dump({'schema': 'onnx-splitpoint/controlled-test-starts', 'limits': {'collector': 1}, 'entries': []}, stream)
    execute(command, 'source-test', 300, extra_env={'ONNX_SPLITPOINT_TEST_START_JOURNAL': str(guard)})
    if registry.active_count():
        raise RuntimeError('source diagnostic retained owned remote operations')
    proof = _verify_source_test(run, cfg, output, diagnostic, row_id, guard)
    print('[rest] source diagnostic PASS: one capture, samples, verified source end and resource release', flush=True)
    return proof


def _verify_energy(block, identities):
    from ..native_energy_reporting import collect_native_energy
    rows = collect_native_energy(block)
    expected = {tuple(value) for value in identities}
    if (len(rows) != len(expected) or {augmentation.key(row) for row in rows} != expected
            or any(row.get('ok') is not True or row.get('energy_repeat_valid_n') != 3 for row in rows)):
        raise ValueError('normal energy collector did not verify all three replicates of the selected block')
    return rows


def preserved_sources(plan):
    """Compact original evidence only; no new seal or artifact identity."""
    run = Path(plan['augmentation_run'])
    paths = [run / 'reports' / name for name in ('native_producer_combined_summary.json',
        'native_producer_stage.json', 'native_producer_stage_config.json', 'native_expected_matrix.json')]
    paths += [run / 'quality_management/central_quality_summary.json']
    for row in io.read_json(paths[0])['rows']:
        if row.get('report'): paths.append(Path(row['report']))
    # Include every stored Quality request via the actual central summary.
    for row in io.read_json(run / 'quality_management/central_quality_summary.json').get('results', []):
        paths.append(run / row['source_request'])
    execution = io.read_json(Path(plan['output_root']) / 'completion/execution-plan.json')
    for row in execution['rows']:
        path = Path(row['result_path']); paths.append(path)
        for repetition in io.read_json(path).get('repetitions', []):
            paths += [path.parent / value['path'] for value in (repetition.get('raw_output_samples') or {}).values()]
            if repetition.get('completed_result_artifact'): paths.append(path.parent / repetition['completed_result_artifact']['path'])
    return {str(path): {'sha256': io.file_sha256(path), 'size': path.stat().st_size} for path in sorted(set(paths))}


def _completion_command(plan_path, summary, validation, output, case_key='remaining', native_root=None):
    command = [sys.executable, '-u', '-m', 'onnx_splitpoint_tool.workflow.yolo_augmentation',
        '--completion-barrier', str(plan_path), '--native-summary', str(summary), '--validation-summary', str(validation),
        '--case-key', case_key, '--attempt-dir', str(output)]
    if native_root: command += ['--native-report-root', str(native_root)]
    return command


def _merge_native(run, cfg, namespace, block, previous_summary, execute):
    reports = block / 'reports'; reports.mkdir(parents=True, exist_ok=True)
    root = run / 'native_producers/variants' / namespace
    report = [sys.executable, '-u', str(coordinator._script('native_producer_final_report.py')),
        '--root', str(root), '--recursive', '--out-dir', str(reports), '--eval-root', str(run),
        '--previous-summary', str(previous_summary)]
    execute(report, 'block2-native-final-report', 600)
    summary = reports / 'native_producer_combined_summary.json'
    observed = io.read_json(summary)['rows']
    original = io.read_json(previous_summary)['rows']
    if len(observed) != 12 or {augmentation.key(r) for r in observed} != {augmentation.key(r) for r in original}:
        raise ValueError('normal rest merge changed the frozen Native cohort')
    for row in original:
        if augmentation.key(row) == SPECIAL: continue
        current = next(r for r in observed if augmentation.key(r) == augmentation.key(row))
        for field in ('report', 'native_command_contract_sha256', 'repetition_records', 'fps_repetition_samples'):
            if current.get(field) != row.get(field): raise ValueError('rest merge changed preserved Native measurements')
    previous_matrix = io.read_json(run / 'reports/native_expected_matrix.json')
    expected = previous_matrix.get('expected_native_rows') or previous_matrix.get('expected_rows')
    if not expected: raise ValueError('original frozen Native expected matrix missing')
    matrix = reports / 'native_expected_matrix.json'
    io.write_json(matrix, coordinator._native_performance_expected_matrix(expected, observed))
    validation = reports / 'native_validation/native_producer_validation_summary.json'
    validate = [sys.executable, '-u', str(coordinator._script('native_producer_validate_visualize.py')),
        '--summary', str(summary), '--out-dir', str(validation.parent), '--eval-root', str(run), '--root', str(root),
        '--central-quality-summary', str(run / 'quality_management/central_quality_summary.json'),
        '--quality-gate-json', json.dumps(cfg['quality_gate_policy'], sort_keys=True, separators=(',', ':')),
        *coordinator._existing_reference_validation_args(cfg)]
    execute(validate, 'block2-native-stored-validation', 900)
    payload = io.read_json(validation)
    if not payload.get('rows') or payload.get('technical_error_count') or payload.get('empty_output_error'):
        raise ValueError('rest Native stored-output validation failed')
    execute(report + ['--quality-summary', str(validation), '--expected-matrix', str(matrix)], 'block2-native-quality-report', 600)
    return summary, validation


def _delivery_evidence(output, work, plan):
    """Put bounded normal report copies into the existing diagnostic bundle."""
    destination = output / 'reports/rest-diagnostics'
    destination.mkdir(parents=True, exist_ok=True)
    sources = list((output / 'commands').glob('*.json'))
    sources += [Path(plan['output_root']) / 'completion/execution-plan.json']
    sources += [output / 'source-test-starts.json', work / 'source_test/energy_summary.json',
        work / 'source_test/run_000/energy_summary.json', work / 'source_test/run_000/collector_stdout.log',
        work / 'source_test/run_000/collector_stdout.log.cleanup.json',
        work / 'source_test/run_000/collector_storage/command_window_markers.json']
    names = ('native_producer_combined_summary.json', 'native_expected_matrix.json',
        'native_validation/native_producer_validation_summary.json',
        'native_energy_measurements/native_producer_energy_plan.json',
        'native_energy_measurements/native_producer_energy_results.json',
        'native_energy_measurements/native_producer_energy_results.partial.json')
    for block in ('block1', 'block2'):
        sources += [work / block / 'reports' / name for name in names]
    index = []
    for ordinal, source in enumerate(sources):
        if not source.is_file(): continue
        row = {'source': str(source), 'size': source.stat().st_size, 'sha256': io.file_sha256(source)}
        if row['size'] <= 16 * 1024 * 1024:
            target = destination / f'{ordinal:03d}_{source.name}'
            shutil.copy2(source, target); row['bundle_file'] = target.name
        else:
            row['bundle_file'] = None; row['reason'] = 'large original evidence remains at its recorded source'
        index.append(row)
    io.write_json(destination / 'source-index.json', {'files': index})


def _energy_delivery_roots(work, continuation):
    from ..native_energy_reporting import native_energy_results_path
    from ..resources.remote_scripts.run_native_producer_energy_from_summary import _energy_result_reusable, _resume_row_identity
    first = work / 'block1'; second = work / 'block2'
    current = native_energy_results_path(first, include_partial=True)
    roots = [first] if current.is_file() else []
    if continuation:
        previous = io.read_json(Path(continuation['previous_results']))
        retained = {_resume_row_identity(row['row']): row for row in previous['rows'] if _energy_result_reusable(row)}
        new_rows = io.read_json(current).get('rows', []) if current.is_file() else []
        observed = {_resume_row_identity(row['row']): row for row in new_rows}
        imported = all(identity in observed and _energy_result_reusable(observed[identity])
            and all(observed[identity]['run'].get(field) == old['run'].get(field)
                    for field in ('energy_aggregate_path', 'energy_aggregate_sha256'))
            for identity, old in retained.items())
        if imported and set(observed) == set(retained) and len(previous['rows']) > len(retained):
            # An interrupted child may not publish its pending row at all.
            # Keep the predecessor's incomplete row available to the normal
            # verified campaign-repeat reader; the nine imported aggregates
            # are the same bytes and must not become a second cohort.
            roots = [Path(continuation['previous_block'])]
        if not imported:
            if any(_energy_result_reusable(row) or (row.get('run', {}).get('energy_aggregate_valid_repeat_count') or 0) > 0
                   for row in new_rows):
                raise ValueError('new energy results did not preserve every verified previous aggregate')
            roots = [Path(continuation['previous_block'])]
    if native_energy_results_path(second, include_partial=True).is_file(): roots.append(second)
    return roots


def finish_rest(plan_path, attempt_dir, *, include_b066=False, previous_energy_attempt=None, source_test=False, h8_reentry=False):
    plan_path = Path(plan_path).resolve(); plan = augmentation.load_frozen_plan(plan_path)
    output = augmentation.execution_output_path(plan, attempt_dir)
    if attempt_dir is None or output == Path(plan['output_root']).resolve():
        raise ValueError('rest completion requires its own new attempt directory')
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', output.name): raise ValueError('unsafe rest attempt name')
    if (output / 'production-start.json').exists(): raise FileExistsError('rest attempt already started; no automatic restart')
    if source_test and h8_reentry:
        raise ValueError('authorized H8 energy reentry must not add a separate diagnostic capture')
    continuation_args = {'h8_reentry': True} if h8_reentry else {}
    continuation = energy_continuation(plan, previous_energy_attempt, **continuation_args) if previous_energy_attempt is not None else None
    if (source_test or h8_reentry) and continuation is None:
        raise ValueError('authorized source test requires the exact previous energy attempt')
    if continuation and Path(continuation['previous_attempt']) == output:
        raise ValueError('energy continuation requires a new attempt identity')
    inventory = continuation['inventory'] if continuation else remaining_inventory(plan)
    run = Path(plan['augmentation_run']); cfg = io.read_json(run / 'reports/native_producer_stage_config.json')
    summary = run / 'reports/native_producer_combined_summary.json'
    validation = run / 'reports/native_validation/native_producer_validation_summary.json'
    work = run / 'reports/rest_attempts' / output.name
    if work.exists(): raise FileExistsError('rest work namespace already exists')
    output.mkdir(parents=True, exist_ok=True)
    if not (output / 'runner-bundle.tar.gz').is_file(): raise ValueError('tested Generic runner bundle missing')
    before = {**augmentation.preservation_inventory(plan), **preserved_sources(plan)}
    io.write_json(output / 'preservation-before.json', before)
    energy_roots = []; final_summary = summary; rc = 1; status = 'blocked'; error = ''
    try:
        ownership_args = {'previous_attempt': continuation['previous_attempt']} if continuation else {}
        if source_test: ownership_args['source_test'] = True
        if h8_reentry: ownership_args['h8_reentry'] = True
        with owned_execution(plan, output, cfg, **ownership_args) as (execute, env, registry, cancel):
            with (output / 'production-start.json').open('x') as stream:
                json.dump({'started_at': io.now(), 'pid': os.getpid(), 'plan': str(plan_path),
                           'include_b066': include_b066, 'inventory': inventory, 'automatic_restart': False,
                           'energy_continuation': continuation, 'source_test_requested': source_test,
                           'authorized_h8_energy_reentry': h8_reentry}, stream, indent=2)
            work.mkdir(parents=True); io.write_json(output / 'rest-inventory.json', inventory)
            for name in ('native_producer_combined_summary.json', 'native_producer_stage.json', 'native_producer_stage_config.json', 'native_expected_matrix.json'):
                shutil.copy2(run / 'reports' / name, work / name)
            if source_test:
                _run_source_test(run, cfg, output, work, execute, registry)
            block1 = work / 'block1'; _copy_validation(validation, block1)
            if not continuation:
                execute(_completion_command(plan_path, summary, validation, output), 'block1-eight-generic')
            energy_args = {'previous_results': Path(continuation['previous_results'])} if continuation else {}
            execute(energy_command(run, cfg, summary, validation, block1, **energy_args),
                    'block1-remaining-energy' if continuation else 'block1-eleven-energy', 43200)
            _verify_energy(block1, inventory['block1_energy'])
            io.write_json(output / 'block1-complete.json', {'at': io.now(), 'generic_requested': 0 if continuation else 8,
                'native_preserved': 11, 'energy_root': str(block1), 'energy_continuation': continuation})
            status, rc = 'partial_b066_pending', 2
            if include_b066:
                from ..remote.process_lease import controller_activity
                command, namespace = singlecase_command(run, cfg, output)
                native_attempt = run / 'reports/native_producer_variants' / namespace
                native_attempt.mkdir(parents=True, exist_ok=False)
                # New tool path isolates changed scripts from eleven historical commands.
                with controller_activity(setup_id=SPECIAL[2], attempt_dir=native_attempt, purpose='dut_job', cancel_event=cancel, env=env):
                    execute(command, 'block2-b066-native', 22000, setup=SPECIAL[2], attempt=native_attempt)
                block2 = work / 'block2'
                final_summary, new_validation = _merge_native(run, cfg, namespace, block2, work / 'native_producer_combined_summary.json', execute)
                execute(_completion_command(plan_path, final_summary, new_validation, output, '|'.join(SPECIAL), block2), 'block2-b066-generic')
                selected = copy.deepcopy(io.read_json(final_summary))
                selected['rows'] = [r for r in selected['rows'] if augmentation.key(r) == SPECIAL]
                if len(selected['rows']) != 1 or selected['rows'][0].get('ok') is not True:
                    raise ValueError('b066 energy lacks its one successful measured Native observation')
                energy_summary = run / 'reports' / ('rest_' + output.name + '_block2_energy_summary.json')
                io.write_json(energy_summary, selected)
                next_cfg = copy.deepcopy(cfg); next_cfg['remote_tool_dir'] = command[command.index('--remote-tool-dir') + 1]
                execute(energy_command(run, next_cfg, energy_summary, new_validation, block2), 'block2-b066-energy', 7200)
                _verify_energy(block2, [SPECIAL])
                status, rc = 'complete', 0
    except BaseException as exc:
        error = f'{type(exc).__name__}: {exc}'; traceback.print_exc(); status, rc = 'stopped', 1
        cleanup_path = output / 'cleanup-status.json'
        cleanup_error = io.read_json(cleanup_path).get('cleanup_error', '') if cleanup_path.is_file() else ''
        io.write_json(output / 'technical-stop.json', {'at': io.now(), 'error': error,
            'primary_error': error, 'cleanup_error': cleanup_error, 'automatic_restart': False})
    finally:
        after = {path: {'sha256': io.file_sha256(Path(path)), 'size': Path(path).stat().st_size}
                 for path in before if Path(path).is_file()}
        try:
            energy_roots = _energy_delivery_roots(work, continuation)
        except Exception as exc:
            # Keep the verified predecessor visible while refusing an inconsistent
            # new projection; every new raw result remains in the diagnostics.
            energy_roots = [Path(continuation['previous_block'])] if continuation else []
            io.write_json(output / 'energy-source-conflict.json', {'error': str(exc), 'primary_error': error,
                'previous_results': continuation['previous_results'] if continuation else None})
            status, rc, error = 'energy_source_conflict', 1, error + '; energy source projection: ' + str(exc)
        io.write_json(output / 'preservation-after.json', {'unchanged': before == after, 'files': after})
        if before != after: status, rc, error = 'preservation_conflict', 1, error + '; closed baseline changed'
        io.write_json(output / 'rest-result-sources.json', {'native_summary': str(final_summary),
            'energy_roots': [str(p) for p in energy_roots], 'frozen_scope_count': 12,
            'partial_energy_evidence': [str(path) for path in work.glob('block*/reports/native_energy_measurements/*.partial.json')]})
        try:
            _delivery_evidence(output, work, plan)
        except Exception as exc:
            status, rc, error = 'diagnostic_failed', 1, error + '; ' + str(exc)
        rc = augmentation.write_final_delivery(plan, status, error, rc, execution_output=output,
            native_summary=final_summary, energy_results=energy_roots)
    return rc
