#!/usr/bin/env python3
"""One Generic completed-task point, three normal 60-second energy captures.

Offline prepare requires its three fresh, source-bound performance processes.
run requires an explicit coordinator slot handoff. No deployment, implicit retry,
recovery, budget initialization/reset, tuning, or archive write is provided.
Authorized pure sample loss permits at most three clean normal resumes per call.
"""
from __future__ import annotations
import argparse
import copy
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import sys
import threading
import uuid

W = Path(__file__).resolve().parents[2]
R = None
BUDGET = DEFAULT_PLAN = FINAL_PLAN = FINAL_FREEZE = None
EXPECTED_PLAN_SHA256 = '77f3a669cce6212d5ca78dcae4ab47959b7a33635a0ca4dc4f75db486f2f03be'
DEFAULT_REGISTRY = Path.home()/'.onnx_splitpoint_tool/hardware_setups.yaml'
SETUPS = ('orin_nx_deepx_m1_01', 'orin_nx_hailo10_01', 'orin_nx_hailo8_01')
ADAPTER_PATH = Path(__file__).resolve()


def configure(run=None, *, plan_path=None, adapter_path=None):
    global R, W, BUDGET, DEFAULT_PLAN, FINAL_PLAN, FINAL_FREEZE, ADAPTER_PATH
    from .energy_parent_flow import parent_context
    context = parent_context(run, plan_path=plan_path, adapter_path=adapter_path)
    R, W, BUDGET, ADAPTER_PATH = context['report_root'], context['controller_root'], context['budget'], context['adapter']
    DEFAULT_PLAN = R/'generic_energy_extension/confirmation_v2/plan/execution-plan.json'
    FINAL_PLAN = R/'generic_energy_extension/confirmation_final_20261008/plan/execution-plan.json'
    FINAL_FREEZE = R/'final_implementation_20261008/IMPLEMENTATION_FREEZE.json'
    return context


def now(): return datetime.now(timezone.utc).isoformat()
def read(path): return json.loads(Path(path).read_text())
def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''): digest.update(block)
    return digest.hexdigest()
def canonical(value):
    from onnx_splitpoint_tool.runners.generic_energy import canonical_hash
    return canonical_hash(value)
def write(path, value):
    from onnx_splitpoint_tool.benchmark.schema import write_json_atomic
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(path, value)
def file_binding(path):
    path = Path(path).resolve(strict=True)
    return {'path': str(path), 'sha256': sha(path)}
def bound_file(binding):
    path = Path(binding['path'])
    if not path.is_file() or sha(path) != binding['sha256']:
        raise ValueError('Frozen input changed: '+str(path))
    return path


def controller_sources():
    # Bind the real normal collector/report/ownership dependencies, without
    # copying them or claiming this controller is an executed DUT algorithm.
    names = {
        'scripts/energy_measurement_cli.py',
        'scripts/run_native_producer_energy_from_summary.py',
        'scripts/native_producer_energy_plan.py',
        'onnx_splitpoint_tool/workflow/generic_completed_task.py',
        'onnx_splitpoint_tool/workflow/run_control.py',
        'onnx_splitpoint_tool/workflow/execution_binding.py',
        'onnx_splitpoint_tool/runners/generic_energy.py',
        'onnx_splitpoint_tool/runners/task_completion.py',
        'onnx_splitpoint_tool/generic_energy_reporting.py',
        'onnx_splitpoint_tool/generic_completion_reporting.py',
        'onnx_splitpoint_tool/native_progress.py',
        'onnx_splitpoint_tool/native_command_contract.py',
        'onnx_splitpoint_tool/process_control.py',
        'onnx_splitpoint_tool/platform_power.py',
        'onnx_splitpoint_tool/quality_service.py',
        'onnx_splitpoint_tool/benchmark/schema.py',
    }
    names.update(('onnx_splitpoint_tool/workflow/generic_energy_parent.py',
                  'onnx_splitpoint_tool/workflow/native_energy_parent.py',
                  'onnx_splitpoint_tool/workflow/energy_resume.py',
                  'onnx_splitpoint_tool/workflow/energy_parent_flow.py'))
    for folder in ('energy', 'remote'):
        names.update(str(p.relative_to(W)) for p in (W/'onnx_splitpoint_tool'/folder).rglob('*.py'))
    return [file_binding(W/name) for name in sorted(names)]


def select_case(plan_path, setup, case_id):
    configure(plan_path=plan_path)
    plan_path = Path(plan_path).resolve(strict=True)
    followup = plan_path not in (DEFAULT_PLAN, FINAL_PLAN)
    if followup and (plan_path.parent.parent.parent != R/'generic_followup/batches'
                     or plan_path.parent.name != 'plan' or plan_path.name != 'execution-plan.json'):
        raise ValueError('Only frozen nine-case, exact final six-case or normal prepared final follow-up plans are in scope')
    if plan_path == DEFAULT_PLAN and sha(plan_path) != EXPECTED_PLAN_SHA256:
        raise ValueError('Only the frozen nine-case Generic confirmation plan is in scope')
    plan = read(plan_path)
    final = plan_path == FINAL_PLAN
    if followup:
        from onnx_splitpoint_tool.workflow.generic_completed_task import validate_plan
        preparation = read(plan_path.parent.parent/'PREPARATION_ACCEPTANCE.json')
        if (preparation.get('status') != 'prepared_with_normal_workflow'
                or preparation.get('execution_plan') != str(plan_path)
                or preparation.get('execution_plan_sha256') != sha(plan_path)
                or plan.get('followup_scope') not in ('detectors27','classifiers168','original_yolov7_final_three_hailo10')
                or plan.get('final_source_freeze',{}).get('path') != str(FINAL_FREEZE)
                or plan.get('build_allowed') is not False or plan.get('fallback_allowed') is not False
                or plan.get('automatic_retry') is not False):
            raise ValueError('Normal final follow-up preparation or bounded policy differs')
    if final:
        receipt = read(FINAL_PLAN.parent.parent/'BUNDLE_PREPARATION.json')
        if (receipt.get('plan') != str(plan_path) or receipt.get('plan_sha256') != sha(plan_path)
                or receipt.get('normal_scope_validation') != 'pass'
                or receipt.get('final_source_freeze') != plan.get('final_source_freeze')):
            raise ValueError('Final Generic bundle preparation binding differs')
        freeze = bound_file(plan['final_source_freeze'])
        bundle = bound_file({'path': plan['runner_bundle_source'], 'sha256': plan['runner_bundle_sha256']})
        if (freeze != FINAL_FREEZE or read(freeze).get('validated') is not True
                or bundle != FINAL_PLAN.parent.parent/'runner-bundle.tar.gz'
                or receipt.get('bundle') != str(bundle) or receipt.get('bundle_sha256') != sha(bundle)
                or plan.get('build_allowed') is not False or plan.get('fallback_allowed') is not False
                or plan.get('automatic_retry') is not False
                or plan.get('energy_requires_fresh_same_bundle_three_process_performance') is not True
                or plan.get('shared_energy_campaign_budget') != str(BUDGET)):
            raise ValueError('Final Generic freeze, bundle or bounded policy differs')
        wanted = {(s, c) for s in ('orin_nx_deepx_m1_01', 'orin_nx_hailo8_01') for c in ('b009','b044','b066')}
        actual = [(r.get('setup_id'), r.get('case_id')) for r in plan['rows']]
        if len(actual) != 6 or set(actual) != wanted or plan.get('row_count') != 6:
            raise ValueError('Final Generic six-case scope differs')
    matches = [x for x in plan['rows'] if x['setup_id'] == setup
               and (x['case_id'] == case_id or x.get('case_key') == case_id)]
    if len(matches) != 1 or setup not in SETUPS or (not followup and matches[0]['case_id'] not in ('b009','b044','b066')):
        raise ValueError('Exactly one audited Generic case is required')
    row = matches[0]
    if (not followup and row['model_id'] != 'yolov7_paper') or row['identity'] != row['runtime_config']['identity']:
        raise ValueError('Generic confirmation identity differs')
    result = Path(row['result_path']).resolve()
    if not result.is_relative_to(plan_path.parent.parent/'results'):
        raise ValueError('Only the new performance cohort is admitted')
    if final:
        from onnx_splitpoint_tool.workflow.generic_completed_task import bind_runtime_source_closure
        bind_runtime_source_closure(copy.deepcopy(row), FINAL_PLAN.parent.parent/'runner-bundle')
    if followup:
        bound_file({'path':plan['ready_source_scope'],'sha256':plan['ready_source_scope_sha256']})
        validate_plan(plan, Path(plan['scope_source']), Path(plan['original_run']))
        verify_final_followup(row,plan)
        verified_cohort(row)
    return plan, row


def performance_admission(row, *, plan=None):
    from onnx_splitpoint_tool.generic_completion_reporting import validate_completion_case, completion_pair
    result = Path(row['result_path'])
    if not result.is_file():
        raise ValueError('READY_TO_PREPARE_AFTER_PERFORMANCE: three fresh processes are missing')
    raw = read(result)
    projection = validate_completion_case(raw, row, result_root=result.parent)
    if projection.get('status') != 'completed' or projection.get('valid_repetition_count') != 3:
        raise ValueError('Three current, independently validated performance processes required')
    final = plan is not None and bool(plan.get('final_source_freeze'))
    followup = final and bool(plan.get('followup_scope'))
    if followup:
        selected = (Path(plan['output_root'])/'plan/execution-plan.json').resolve(strict=True)
        if (selected in (DEFAULT_PLAN,FINAL_PLAN)
                or selected.parent.parent.parent != FINAL_FREEZE.parent.parent/'generic_followup/batches'):
            raise ValueError('Follow-up admission requires the actual normal prepared batch path')
    if final:
        sources = {x['path']: x['sha256'] for x in row['runtime_config']['runtime_source_files']}
        if (raw.get('process_repetition_policy') != 'separate_processes' or not sources
                or any(rep.get('runtime_source_binding') != {'before':sources,'after':sources,'unchanged':True}
                       for rep in projection['repetitions'])):
            raise ValueError('Final Generic performance full source closure differs')
        success = read(result.parent/'coordinator-success.json')
        if success.get('status') != 'complete' or success.get('result_sha256') != sha(result):
            raise ValueError('Final Generic coordinator success differs')
        current_path, pairs_path = R/'current_results/generic_current.json', R/'current_results/generic_cross_runner.json'
        selected_plan = Path(plan['output_root'])/'plan/execution-plan.json' if followup else FINAL_PLAN
        if followup and read(selected_plan) != plan:
            raise ValueError('Normal follow-up execution plan changed')
        current = [r for r in read(current_path)['rows'] if r.get('case_key') == row['case_key']]
        if (len(current) != 1 or current[0].get('status') != 'completed'
                or current[0].get('execution_plan') != str(selected_plan)
                or current[0].get('execution_plan_sha256') != sha(selected_plan)
                or current[0].get('current_result_path') != str(result)
                or current[0].get('raw_result_sha256') != sha(result)
                or (not followup and current[0].get('normal_quality_eligible') is not True)):
            raise ValueError('Final Generic exact current performance/Quality admission incomplete')
        current = current[0]
        proof = bound_file({'path':current['native_cohort_proof'],'sha256':current['native_cohort_proof_sha256']})
        if followup:
            from onnx_splitpoint_tool.runners.generic_energy import build_contract,execution_environment
            repetitions = projection['repetitions']
            config = row['runtime_config']
            execution = repetitions[0].get('execution_identity')
            modes = {rep.get('completion_runtime_mode','classification_top1_top5') for rep in repetitions}
            runners = [s['path'] for s in config['runtime_source_files'] if s['path'].endswith('/runner/run_split_onnxruntime.py')]
            if (read(proof) != row['fresh_native_cohort']
                    or len({(rep['boot_id'],rep['pid'],rep['process_start_ticks']) for rep in repetitions}) != 3
                    or not execution or any(rep.get('execution_identity') != execution for rep in repetitions)
                    or config.get('execution_environment') != execution_environment(row['remote_python_executable'],row.get('remote_environment') or {})
                    or len(modes) != 1 or len(runners) != 1):
                raise ValueError('Normal follow-up input/process/mode/environment binding differs')
            build_contract(config,source_files=config['runtime_source_files'],runner_path=runners[0],
                           completion_runtime_mode=next(iter(modes)),duration_s=60.,execution_identity=execution)
        else:
            quality = current['native_quality_provenance']
            if quality.get('normal_report_verified') is not True:
                raise ValueError('Final Generic fresh Native Quality proof unverified')
            bound_file({'path':str(R/quality['normal_report']),'sha256':quality['normal_report_sha256']})
            bound_file({'path':quality['quality_summary'],'sha256':quality['quality_summary_sha256']})
        pairs = [r for r in read(pairs_path)['pairs'] if r.get('generic_source_path') == str(result)
                 and all(r.get(k) == row[k] for k in ('model_id','case_id','setup_id','precision'))]
        if (len(pairs) != 1 or pairs[0].get('eligible_for_technical_transfer') is not True
                or (not followup and pairs[0].get('eligible_for_quality_transfer') is not True)):
            raise ValueError('Final Generic normal current Quality pair missing or excluded')
        pair = pairs[0]
        current_bindings = {'current_report':file_binding(current_path),'normal_pair_report':file_binding(pairs_path),
                            'native_cohort_proof':file_binding(proof),'coordinator_success':file_binding(result.parent/'coordinator-success.json')}
        if followup:
            current_bindings['followup_plan'] = file_binding(selected_plan)
    else:
        pair = completion_pair(projection, row)
        current_bindings = {}
    # The exact final cohort retains its actual normal Quality/technical pair.
    # The Product importer reports Claim=False after valid captures when ranking
    # is excluded; that status is preserved rather than made a capture veto.
    if not final and pair.get('eligible_for_claim_transfer') is not True:
        raise ValueError('Normal current performance/Quality admission incomplete; no diagnostic capture')
    return {'performance_result': file_binding(result), 'normal_pair': pair,
            'valid_repetitions': 3, 'fresh_confirmation_only': True, **current_bindings}


def source_and_target(registry_path, setup_id, plan_target):
    from onnx_splitpoint_tool.energy.config import load_hardware_registry, energy_setup_from_registry
    from onnx_splitpoint_tool.energy.full_system_gain import verify_full_system_current_scale_calibration
    from onnx_splitpoint_tool.platform_power import host_config_from_setup
    from onnx_splitpoint_tool.workflow.execution_binding import physical_dut_key
    registry = load_hardware_registry(registry_path)
    rows = [x for x in registry['hardware_setups'] if x.get('id') == setup_id]
    if len(rows) != 1: raise ValueError('Unique physical registry setup required')
    row = rows[0]; host = host_config_from_setup(row)
    target = {**row, 'remote': {'host': host.host, 'user': host.user, 'port': host.port,
        'remote_base_dir': host.remote_base_dir, 'ssh_extra_args': host.ssh_extra_args}}
    if physical_dut_key(target) != physical_dut_key(plan_target):
        raise ValueError('Registry and measured performance DUT differ')
    for key in ('host', 'user', 'port'):
        if str(target['remote'][key]) != str(plan_target['remote'][key]):
            raise ValueError('Registry and performance SSH binding differ')
    view = energy_setup_from_registry(registry, setup_id, registry_path=registry_path)
    calibration = verify_full_system_current_scale_calibration(view, physical_scope='FS')
    if calibration.get('full_system_current_scale_verified') is not True:
        raise ValueError('Full-system calibration is not verified')
    defaults = registry['energy_defaults']
    if defaults.get('physical_scope', defaults.get('measurement_physical_scope')) != 'FS':
        raise ValueError('Normal collector must resolve physical scope FS')
    collector = file_binding(defaults['collector_binary'])
    if collector['sha256'] != defaults['collector_sha256']:
        raise ValueError('Normal collector binary differs from registry')
    source = str(view.urecs_address).strip().lower()
    if source != str(plan_target['energy']['urecs_address']).strip().lower():
        raise ValueError('Performance/registry physical source binding differs')
    resources = [physical_dut_key(target), 'source:'+source,
                 'controller:cpu', 'controller:io', 'controller:nic', 'controller:capture']
    return registry, target, resources, calibration, collector


def leased_prefix(target):
    # prepare_energy_case appends one shell payload. Raw SSH has exactly that
    # one remote-command argument; the normal lease CLI wraps it in its launcher.
    from onnx_splitpoint_tool.workflow.generic_completed_task import transport_for
    from onnx_splitpoint_tool.remote.process_lease import harden_remote_process_lease_ssh_argv
    transport = transport_for(target, R/'operations/generic_energy_parent/not_executed')
    raw_prefix = harden_remote_process_lease_ssh_argv(transport._ssh_base())
    return [sys.executable, '-B', '-m', 'onnx_splitpoint_tool.remote.process_lease_cli',
            'exec', '--label', 'generic-completed-energy', '--timeout-s', '850', '--', *raw_prefix]


def verify_calls(plan, prefix):
    calls = plan['calls']
    if len(calls) != 3 or [x['repetition_index'] for x in calls] != [1,2,3]:
        raise ValueError('Exactly three ordered Single captures required')
    identity = plan['identity']
    expected_key = json.dumps(['generic_completed:'+identity['comparison_backend'], identity['model_id'],
                              identity['case_id'], identity['setup_id']], separators=(',',':'))
    if plan['campaign_row_id'] != expected_key or Path(plan['campaign_budget_file']) != BUDGET:
        raise ValueError('Generic case must use the original shared campaign budget and regime key')
    for i, call in enumerate(calls):
        argv = call['argv']
        expected = {'--duration':'60.0','--runs':'1','--invalid-repeat-max-retries':'0',
                    '--campaign-budget-file': str(BUDGET), '--campaign-row-id': expected_key,
                    '--campaign-repeats':'3','--campaign-max-retries':'2',
                    '--campaign-max-transport-failures':'20','--task-logical-repeat':f'repeat:{i}'}
        if canonical(argv) != call['command_sha256']:
            raise ValueError('Normal collector argv hash changed')
        if any(argv.count(k) != 1 or argv[argv.index(k)+1] != v for k,v in expected.items()):
            raise ValueError('Bounded collector policy changed')
        if '--task-budget-file' in argv or any(argv.count(k) != 1 for k in (
            '--exact-run-count','--require-runtime-work-units','--require-command-window-alignment')):
            raise ValueError('Normal collector exact-count/work-unit/window policy missing')
        for kind in ('workload_command', 'preflight_command'):
            path = bound_file({'path':call[kind+'_file'], 'sha256':call[kind+'_sha256']})
            command = shlex.split(path.read_text())
            if command[:-1] != prefix or len(command) != len(prefix)+1:
                raise ValueError('Remote payload bypasses the exact journaled SSH prefix')


def prepare(args):
    configure(plan_path=args.plan)
    from onnx_splitpoint_tool.workflow.generic_completed_task import prepare_energy_case, check_energy_campaign
    plan, row = select_case(args.plan, args.setup, args.case)
    admission = performance_admission(row, plan=plan) # Before any output tree, staging, or budget write.
    run = Path(args.run_dir).resolve()
    if not run.is_relative_to(R/'energy/generic_completed') or run == R/'energy/generic_completed':
        raise ValueError('Use a fresh run under R/energy/generic_completed')
    if run.exists(): raise FileExistsError('Fresh parent output required; no implicit retry')
    registry_path = Path(args.registry).resolve(strict=True)
    registry, target, resources, calibration, collector = source_and_target(
        registry_path, args.setup, plan['hardware_targets'][args.setup])
    budget_before = sha(BUDGET)
    prefix = leased_prefix(target)
    # Read the actual normal STOP and limits before creating the prepared plan.
    budget_projection = {'identity':row['identity'], 'campaign_budget_file':str(BUDGET),
        'campaign_row_id':json.dumps(['generic_completed:'+row['identity']['comparison_backend'],
            row['model_id'],row['case_id'],row['setup_id']],separators=(',',':')),
        'campaign_source_id':str(target['energy']['urecs_address']).strip().lower(),
        'campaign_limits':{'max_chains':9,'max_retries':2,'max_transport_failures':20},
        'hardware_setups_file':str(registry_path)}
    for logical in ('repeat:0','repeat:1','repeat:2'):
        check_energy_campaign(budget_projection, logical_repeat=logical)
    run.mkdir(parents=True, exist_ok=False)
    registry_snapshot = run/'inputs/hardware_snapshot.json'; write(registry_snapshot, registry)
    case_path = run/'inputs/case.json'; write(case_path,row)
    normal = prepare_energy_case(row, performance_result=Path(row['result_path']), output=run/'measurement',
        collector_python=sys.executable, collector_script=W/'scripts/energy_measurement_cli.py',
        hardware_setups_file=registry_snapshot, remote_command_prefix=prefix,
        campaign_budget_file=BUDGET, duration_s=60.,
        require_environment_observation=True, environment_run_id=run.name)
    verify_calls(normal,prefix)
    parent = {'schema':'revision3/generic-energy-parent/v1','prepared_at':now(),'run_dir':str(run),
        'performance_plan':file_binding(args.plan),'case':file_binding(case_path),
        'normal_energy_plan':file_binding(run/'measurement/generic_energy_plan.json'),
        'registry_source':file_binding(registry_path),'registry_snapshot':file_binding(registry_snapshot),
        'adapter':file_binding(ADAPTER_PATH),'collector_binary':collector,'python':file_binding(sys.executable),
        'python_executable':sys.executable,'controller_sources':controller_sources(),
        'performance_admission':admission,'target':target,'resources':resources,
        'remote_command_prefix':prefix,'calibration':calibration,'selected_identity':row['identity'],
        'campaign_budget_file':str(BUDGET),'budget_read_before_sha256':budget_before,
        'budget_read_after_sha256':sha(BUDGET),'automatic_retry':False,
        'point_count':1,'capture_count':3,'duration_s':60.,'hardware_executed':False}
    parent['require_environment_observation'] = True
    write(run/'parent_plan.json',parent)
    write(run/'PREPARED.json',{'status':'PREPARED_FOR_EXPLICIT_SLOT_HANDOFF','ok':True,
        'parent_plan_sha256':sha(run/'parent_plan.json'),'hardware_starts':0,
        'budget_changed_by_adapter':False,'ready_points':1,'energy_observations':0})
    return parent


def performance_admission_matches(saved, current):
    """Preserve the exact case admission across other rows' Current updates."""
    current=copy.deepcopy(current)
    for field in ('current_report','normal_pair_report'):
        if field in saved or field in current:
            if (field not in saved or field not in current
                    or current[field]['path']!=saved[field]['path']):return False
            # These cumulative reports are mutable. Their selected row, actual
            # performance sources, Native proof and normal pair were rechecked
            # above; retaining the saved hash preserves preparation provenance.
            current[field]['sha256']=saved[field]['sha256']
    return current==saved


def verify_parent(run, *, require_unstarted=True, allow_completed_repeats=False,
                  refreshable_controller_paths=()):
    from onnx_splitpoint_tool.workflow.generic_completed_task import check_energy_campaign
    configure(run)
    run = Path(run).resolve(strict=True); p = read(run/'parent_plan.json')
    prepared = read(run/'PREPARED.json')
    if prepared.get('ok') is not True or prepared['parent_plan_sha256'] != sha(run/'parent_plan.json'):
        raise ValueError('Parent plan missing or changed')
    if run != Path(p['run_dir']) or p['point_count'] != 1 or p['capture_count'] != 3 or p['duration_s'] != 60.:
        raise ValueError('Parent capture scope changed')
    if Path(p['campaign_budget_file']) != BUDGET: raise ValueError('Shared budget path changed')
    for key in ('performance_plan','case','normal_energy_plan','registry_source','registry_snapshot',
                'adapter','collector_binary','python'):
        if p[key]['path'] not in refreshable_controller_paths:bound_file(p[key])
    for binding in p['controller_sources']:
        if binding['path'] not in refreshable_controller_paths:bound_file(binding)
    if not refreshable_controller_paths:
        from .energy_parent_flow import verify_product_sources
        verify_product_sources(W,p['controller_sources'])
    if p['adapter']['path'] != str(ADAPTER_PATH.resolve()): raise ValueError('Different parent adapter')
    case = read(p['case']['path'])
    performance_plan, original_case = select_case(p['performance_plan']['path'],p['selected_identity']['setup_id'],case.get('case_key',p['selected_identity']['case_id']))
    if case != original_case or case['identity'] != p['selected_identity']:
        raise ValueError('Selected performance case changed')
    admission = performance_admission(case, plan=performance_plan)
    if not performance_admission_matches(p['performance_admission'],admission):
        raise ValueError('Current normal admission changed')
    normal = read(p['normal_energy_plan']['path']); verify_calls(normal,p['remote_command_prefix'])
    if normal['identity'] != case['identity'] or normal['performance_result_sha256'] != admission['performance_result']['sha256']:
        raise ValueError('Normal energy/performance identity changed')
    if require_unstarted and ((run/'production-start.json').exists() or
        any((run/'measurement'/f'capture_{i}_started.json').exists() for i in (1,2,3))):
        raise FileExistsError('Prior attempted capture requires explicit reviewed recovery; no automatic rerun')
    for logical in ('repeat:0','repeat:1','repeat:2'):
        check_energy_campaign(normal,logical_repeat=logical,allow_completed_repeats=allow_completed_repeats,
                              parent_run_dir=run)
    return p,case,normal


def latest_clean_attempt(run):
    configure(run)
    from onnx_splitpoint_tool.process_control import _proc_start_time
    started=[p for p in [run,*run.glob('resume_attempts/*')] if(p/'production-start.json').is_file()]
    if not started:raise ValueError('Generic resume requires an actual previous attempt')
    attempt=max(started,key=lambda p:(p/'production-start.json').stat().st_mtime)
    start=read(attempt/'production-start.json');cleanup=read(attempt/'cleanup-status.json')
    if cleanup.get('cleanup_proven') is False:
        from .energy_resume import released_predecessor
        released_predecessor(run,attempt)
    elif (cleanup.get('cleanup_proven') is not True or cleanup.get('owned_remote_active') != 0
            or cleanup.get('owned_local_active') != 0 or cleanup.get('errors')):
        raise ValueError('Generic resume requires the latest proven clean terminal parent')
    ticks=_proc_start_time(int(start.get('parent_pid') or 0))
    if ticks and (not start.get('parent_start_ticks') or ticks==start['parent_start_ticks']):
        raise ValueError('Generic previous parent still alive; no resume or controller refresh')
    return attempt


def previous_energy_plans(run,parent):
    paths=[bound_file(parent['normal_energy_plan'])]
    attempts=sorted((p for p in run.glob('resume_attempts/*') if(p/'production-start.json').is_file()),
                    key=lambda p:(p/'production-start.json').stat().st_mtime)
    for attempt in attempts:
        binding_path=attempt/'worker-plan.json'
        if binding_path.is_file():
            binding=read(binding_path)
            if Path(binding['path'])!=attempt/'measurement/generic_energy_plan.json':
                raise ValueError('Generic previous plan escapes its actual attempt')
            paths.append(bound_file(binding))
    return paths


def refresh_controller(args):
    """Refresh these controller fixes, with frozen inputs and old bindings retained."""
    from onnx_splitpoint_tool.workflow.run_control import EvaluationRunLock
    configure(args.run_dir)
    run=Path(args.run_dir).resolve(strict=True)
    if not run.is_relative_to(R/'energy/generic_completed') or run==R/'energy/generic_completed':
        raise ValueError('Controller refresh requires this campaign Generic parent')
    previous=latest_clean_attempt(run) if(run/'production-start.json').is_file() else None
    owner={'run_id':run.name,'session_id':'generic-binding-'+uuid.uuid4().hex,'purpose':'generic_controller_binding_refresh'}
    lock=EvaluationRunLock(out_root=run.parent,run_dir=run,owner=owner)
    if previous is not None and read(previous/'cleanup-status.json').get('cleanup_proven') is False:
        from .energy_resume import recover_run_fence
        lock.acquire(recover_quarantine=lambda fence:recover_run_fence(run,run/'attempts'/owner['session_id'],
            owner['session_id'],lock,fence,previous))
    else:lock.acquire()
    try:
        if(run/'production-start.json').is_file():latest_clean_attempt(run)
        allowed={str(ADAPTER_PATH.resolve()),str(W/'onnx_splitpoint_tool/workflow/generic_completed_task.py'),
                 str(W/'onnx_splitpoint_tool/energy/collector.py'),
                 str(W/'onnx_splitpoint_tool/remote/resource_recovery.py'),
                 str(W/'onnx_splitpoint_tool/energy/recovery_accounting.py'),
                 str(W/'onnx_splitpoint_tool/energy/task_budget.py'),
                 str(W/'onnx_splitpoint_tool/energy/environment_observer.py'),
                 str(W/'onnx_splitpoint_tool/generic_energy_reporting.py'),
                 str(W/'onnx_splitpoint_tool/generic_completion_reporting.py'),
                 *[str(W/'onnx_splitpoint_tool/workflow'/name) for name in ('generic_energy_parent.py',
                     'native_energy_parent.py','energy_resume.py','energy_parent_flow.py')],
                 str(W/'scripts/native_producer_energy_plan.py'),
                 str(W/'scripts/run_native_producer_energy_from_summary.py')}
        parent,case,normal=verify_parent(run,require_unstarted=False,allow_completed_repeats=True,
                                         refreshable_controller_paths=allowed)
        changed=sorted({b['path'] for b in [parent['adapter'],*parent['controller_sources']]
                        if b['path'] in allowed and sha(b['path'])!=b['sha256']})
        before_budget=sha(BUDGET);history=run/'controller_binding_refresh'/uuid.uuid4().hex
        history.mkdir(parents=True,exist_ok=False)
        import shutil
        for name in ('parent_plan.json','PREPARED.json'):shutil.copy2(run/name,history/name)
        before_normal = parent['normal_energy_plan']
        context_restored = False
        if not(run/'production-start.json').is_file() and not normal.get('execution_environment_restored'):
            from onnx_splitpoint_tool.workflow.generic_completed_task import prepare_energy_resume
            restored = prepare_energy_resume(Path(before_normal['path']),case,
                output=history/'measurement',previous_plans=[Path(before_normal['path'])],parent_run_dir=run)
            if restored['resume_reused_sources'] or any((run/'measurement'/f'capture_{i}_started.json').exists() for i in (1,2,3)):
                raise ValueError('Prepared execution-context refresh found an attempted capture')
            verify_calls(restored,parent['remote_command_prefix'])
            restored['execution_environment_restored'] = True
            write(history/'measurement/generic_energy_plan.json',restored)
            parent['normal_energy_plan'] = file_binding(history/'measurement/generic_energy_plan.json')
            context_restored = True
        existing={b['path'] for b in parent['controller_sources']}
        parent['controller_sources'] += [b for b in controller_sources() if b['path'] not in existing
            and '/workflow/' in b['path'] and Path(b['path']).name in {
                'generic_energy_parent.py','native_energy_parent.py','energy_resume.py','energy_parent_flow.py'}]
        parent['adapter']=file_binding(ADAPTER_PATH)
        parent['controller_sources']=[file_binding(b['path']) if b['path'] in allowed else b for b in parent['controller_sources']]
        write(run/'parent_plan.json',parent)
        prepared=read(history/'PREPARED.json');prepared['parent_plan_sha256']=sha(run/'parent_plan.json')
        write(run/'PREPARED.json',prepared)
        write(history/'REFRESH_RECEIPT.json',{'at':now(),'original_parent_plan':file_binding(history/'parent_plan.json'),
            'current_parent_plan':file_binding(run/'parent_plan.json'),'refreshed_controller_paths':changed,
            'normal_energy_plan_unchanged':before_normal if not context_restored else False,
            'normal_energy_plan_before':before_normal,'normal_energy_plan_after':parent['normal_energy_plan'],
            'prepared_measured_execution_context_restored':context_restored,
            'budget_before_sha256':before_budget,'budget_after_sha256':sha(BUDGET),
            'scientific_inputs_and_runtime_unchanged':True,'hardware_started':False})
    finally:lock.release()


class ResourceTransport:
    """Only metadata/config/result transport, each under the normal transfer claim."""
    METHODS = {'run','run_read_only','run_streaming','scp_upload','scp_download'}
    def __init__(self, transport, *, setup, root):
        self.transport=transport; self.setup=setup; self.root=Path(root); self.index=0
    def __getattr__(self,name):
        method=getattr(self.transport,name)
        if name not in self.METHODS: return method
        def owned(*args,**kwargs):
            from onnx_splitpoint_tool.remote.process_lease import controller_activity
            from onnx_splitpoint_tool.process_control import bind_workflow_resource_options
            self.index+=1
            with controller_activity(setup_id=self.setup,attempt_dir=self.root/f'dut_metadata_{self.index:03d}',purpose='dut_job'):
                attempt=self.root/f'transfer_{self.index:03d}'
                with controller_activity(setup_id=self.setup,attempt_dir=attempt,purpose='transfer') as claim:
                    with bind_workflow_resource_options(controller_setup_id=self.setup,
                            controller_attempt_dir=str(attempt),
                            controller_activity_operation=claim.operation if claim is not None else None,
                            controller_activity_resources=['controller:cpu','controller:io','controller:nic','controller:transfer'] if claim is not None else []):
                        return method(*args,**kwargs)
        return owned


def worker(run,attempt_dir=None):
    from onnx_splitpoint_tool.remote.process_lease import RemoteProcessLeaseJournal, RemoteProcessLeaseRegistry
    from onnx_splitpoint_tool.process_control import ProcessTreeRegistry, bind_process_registry
    from onnx_splitpoint_tool.workflow.generic_completed_task import transport_for, run_energy_case, prepare_energy_resume
    journal=RemoteProcessLeaseJournal.from_environment(required=True)
    if os.environ.get('ONNX_SPLITPOINT_CONTROLLER_RESOURCES')!='required':
        raise RuntimeError('Normal resource-owning parent required')
    run=Path(run).resolve();attempt=Path(attempt_dir).resolve() if attempt_dir else run
    if attempt!=run and (attempt.parent!=run/'resume_attempts' or not attempt.is_dir()):
        raise ValueError('Generic worker attempt escapes its original parent')
    parent,case,normal=verify_parent(run,require_unstarted=False,allow_completed_repeats=attempt!=run)
    started=read(attempt/'production-start.json')
    if started['session_id']!=journal.scope.session_id or started['run_id']!=journal.scope.run_id:
        raise ValueError('Worker journal differs from exact started parent')
    leases=RemoteProcessLeaseRegistry(); leases.configure_journal(scope=journal.scope,journal_dir=journal.directory)
    processes=ProcessTreeRegistry()
    transport=ResourceTransport(transport_for(parent['target'],attempt/'transfers',scope=journal.scope,registry=leases),
                               setup=case['setup_id'],root=attempt/'transfers')
    try:
        with bind_process_registry(processes):
            plan_path=Path(parent['normal_energy_plan']['path'])
            if attempt!=run:
                previous=[bound_file(b) for b in started['previous_energy_plans']]
                if previous!=previous_energy_plans(run,parent):
                    raise ValueError('Generic resume prior plans differ from actual parent history')
                resumed=prepare_energy_resume(plan_path,case,output=attempt/'measurement',
                    previous_plans=previous,transport=transport,parent_run_dir=run)
                verify_calls(resumed,parent['remote_command_prefix'])
                plan_path=attempt/'measurement/generic_energy_plan.json'
                write(attempt/'worker-plan.json',file_binding(plan_path))
        admission=parent['performance_admission']
        result=run_energy_case(plan_path,case,transport,
            performance_admission=admission if admission.get('normal_pair_report') else None)
        write(attempt/'normal_result.json',result)
        if (result.get('normal_report_verified') is not True
                or (result.get('energy_efficiency_claim_eligible') is not True
                    and parent['performance_plan']['path'] != str(FINAL_PLAN)
                    and parent.get('performance_admission',{}).get('followup_plan') != parent['performance_plan'])):
            raise RuntimeError('Normal Generic energy final admission failed')
        return result
    finally:
        if leases.active_count(): leases.cancel_all(grace_s=3)
        if processes.remaining_owned_count(): processes.terminate_all(grace_s=3)
        write(attempt/'worker_cleanup.json',{'owned_remote_active':leases.active_count(),
            'owned_local_active':processes.remaining_owned_count(),'parent_retains_terminal_cleanup_authority':True})


def _sample_loss_retry_attempt(run, *, started_ns, returncode):
    """Read the worker's budget authorization after the normal Parent exited."""
    from onnx_splitpoint_tool.energy.task_budget import (campaign_budget_stop_reason,
        _pure_sample_loss_failure, source_completion)
    run = Path(run).resolve(); attempt = latest_clean_attempt(run)
    if (attempt/'production-start.json').stat().st_mtime_ns < started_ns:
        raise ValueError('Generic retry has no new normal terminal Parent')
    start = read(attempt/'production-start.json')
    if read(attempt/'production-result.json')['returncode'] != returncode:
        raise ValueError('Generic retry Parent result differs')
    parent = read(run/'parent_plan.json')
    binding = read(attempt/'worker-plan.json') if attempt != run else parent['normal_energy_plan']
    plan_path = bound_file(binding)
    if attempt != run and plan_path != attempt/'measurement/generic_energy_plan.json':
        raise ValueError('Generic retry plan escapes its normal attempt')
    plan = read(plan_path); cleanup = read(plan_path.parent/'controller_cleanup.json')
    retry = cleanup.get('sample_loss_retry') or {}
    expected_parent = {'run_id':run.name,'session_id':start['session_id'],
        'pid':start['parent_pid'],'start_ticks':start['parent_start_ticks']}
    if (cleanup.get('owned_local_remaining') != 0
            or retry.get('kind') != 'pure_marker_dropped_samples_nonzero'
            or retry.get('parent') != expected_parent or retry.get('plan') != file_binding(plan_path)
            or retry.get('campaign_budget_file') != str(BUDGET)
            or retry.get('campaign_row_id') != plan['campaign_row_id']
            or retry.get('source_id') != plan['campaign_source_id']
            or retry.get('logical_repeat') not in {'repeat:0','repeat:1','repeat:2'}):
        raise ValueError('Generic retry lacks exact normal pure-loss authorization')
    budget = read(BUDGET); row = retry['campaign_row_id']; source = retry['source_id']
    task = budget['tasks'][row]
    policy = (budget['sources'][source].get('continuous_admission') or {}).get('sample_loss_retries') or {}
    chains = [c for c in task['chains'] if c.get('logical_repeat') == retry['logical_repeat'] and not c.get('cohort_id')]
    if (not chains or chains[-1] != retry.get('chain')
            or any(c.get('valid') is True or c.get('finished') is not True for c in chains)
            or chains[-1].get('source_completion_verified') is not True
            or chains[-1].get('transport_reasons') != ['marker_dropped_samples_nonzero']
            or not source_completion(chains[-1]['run_directory']).get('verified')
            or not _pure_sample_loss_failure(chains[-1],row,'')
            or task['source_id'] != source or row not in policy.get('row_ids', [])
            or retry.get('authorization') != {k:policy.get(k) for k in ('authorization_file','authorization_sha256')}
            or campaign_budget_stop_reason(budget,row,source_id=source,parent_run_dir=run)):
        raise ValueError('Generic retry campaign chain or current authority changed')
    return attempt


def run_prepared(args):
    """A finite series of ordinary Parents; every resume needs prior clean exit."""
    if getattr(args,'single_parent_attempt',False):return _run_prepared_once(args)
    from onnx_splitpoint_tool.native_progress import run_streaming
    from onnx_splitpoint_tool.process_control import ProcessTreeRegistry
    from onnx_splitpoint_tool.energy.collector import _collector_reconnect_backoff
    import time
    run = Path(args.run_dir).resolve(); action = getattr(args,'action','run')
    previous = getattr(args,'previous_attempt',''); registry = ProcessTreeRegistry()
    cancel = threading.Event()
    handlers = {s:signal.signal(s,lambda *_:cancel.set()) for s in (signal.SIGINT,signal.SIGTERM)}
    try:
        for series in range(4):
            command = [sys.executable,'-B',str(ADAPTER_PATH.resolve()),action,
                '--run-dir',str(run),'--slot-handoff',args.slot_handoff,'--single-parent-attempt']
            if previous:command += ['--previous-attempt',str(previous)]
            started_ns = time.time_ns()
            result = run_streaming(command,cwd=W,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},
                timeout=3800,label='generic-normal-parent-series',line_callback=lambda text:print(text,flush=True),
                cancel_event=cancel,process_registry=registry)
            if result.returncode == 0 and not cancel.is_set():return
            if cancel.is_set() or series == 3:
                raise RuntimeError('Generic normal Parent series incomplete; preserve every attempt')
            previous = _sample_loss_retry_attempt(run,started_ns=started_ns,returncode=result.returncode)
            pause = _collector_reconnect_backoff(5.0,cancel_event=cancel)
            if pause.get('cancelled') or cancel.is_set():
                raise RuntimeError('Generic normal Parent series cancelled before resume')
            action = 'resume'
    finally:
        if registry.remaining_owned_count():registry.terminate_all(grace_s=3)
        for signum,handler in handlers.items():signal.signal(signum,handler)


def _run_prepared_once(args):
    from onnx_splitpoint_tool.native_progress import run_streaming
    from onnx_splitpoint_tool.process_control import ProcessTreeRegistry
    from onnx_splitpoint_tool.remote.process_lease import ControllerResourceBroker, RemoteProcessLeaseRegistry, RemoteProcessLeaseScope
    from onnx_splitpoint_tool.quality_service import ResourcePauseGate
    from onnx_splitpoint_tool.workflow.run_control import EvaluationRunLock
    from onnx_splitpoint_tool.workflow.generic_completed_task import observe_owners
    run=Path(args.run_dir).resolve();resuming=getattr(args,'action','run')=='resume'
    parent,case,normal=verify_parent(run,require_unstarted=not resuming,allow_completed_repeats=resuming)
    previous=latest_clean_attempt(run) if resuming else None
    if previous is not None and getattr(args,'previous_attempt','') and Path(args.previous_attempt).resolve()!=previous:
        raise ValueError('Generic resume requires its latest actual clean predecessor')
    if not args.slot_handoff: raise ValueError('Explicit coordinator slot handoff required')
    session='generic-energy-'+uuid.uuid4().hex
    owner={'run_id':run.name,'session_id':session,'purpose':'revision3_generic_completed_energy'}
    lock=EvaluationRunLock(out_root=run.parent,run_dir=run,owner=owner)
    if previous is not None and read(previous/'cleanup-status.json').get('cleanup_proven') is False:
        from .energy_resume import recover_run_fence
        lock.acquire(recover_quarantine=lambda fence:recover_run_fence(run,run/'attempts'/session,
            session,lock,fence,previous))
    else:lock.acquire()
    cancel,stop=threading.Event(),threading.Event()
    leases,processes=RemoteProcessLeaseRegistry(),ProcessTreeRegistry()
    broker=thread=None; handlers={}; errors=[]; primary=None
    attempt=run
    try:
        if resuming:
            attempt=run/'resume_attempts'/uuid.uuid4().hex;attempt.mkdir(parents=True,exist_ok=False)
            if latest_clean_attempt(run)!=previous:raise ValueError('Generic predecessor changed before acquired parent lock')
        observe_owners({case['setup_id']:parent['target']},attempt/'owner_preflight')
        journal=leases.configure_journal(scope=RemoteProcessLeaseScope(run.name,session),journal_dir=attempt/'remote-process-leases')
        broker=ControllerResourceBroker(journal,ResourcePauseGate(),run_dir=run,setups={case['setup_id']:parent['resources']})
        env={**os.environ,**leases.journal_environment(),'ONNX_SPLITPOINT_CONTROLLER_RESOURCES':'required','PYTHONDONTWRITEBYTECODE':'1'}
        def pump():
            while not stop.wait(.025):
                try: broker.tick()
                except BaseException as exc: errors.append(str(exc)); cancel.set(); return
        thread=threading.Thread(target=pump,name='generic-energy-resource-broker',daemon=True); thread.start()
        handlers={s:signal.signal(s,lambda *_:cancel.set()) for s in (signal.SIGINT,signal.SIGTERM)}
        command=[parent['python_executable'],'-B',str(ADAPTER_PATH.resolve()),'worker','--run-dir',str(run)]
        if resuming:command+=['--attempt-dir',str(attempt)]
        from onnx_splitpoint_tool.process_control import _proc_start_time
        started={'started_at':now(),'slot_handoff':args.slot_handoff,
            'run_id':run.name,'session_id':session,'parent_pid':os.getpid(),'argv':command,
            'parent_start_ticks':_proc_start_time(os.getpid()),'attempt_dir':str(attempt),
            'selected_identity':case['identity'],'normal_plan_sha256':parent['normal_energy_plan']['sha256'],
            'no_deployment_or_build':True}
        if resuming:
            started['previous_attempt']=str(previous)
            started['previous_energy_plans']=[file_binding(p) for p in previous_energy_plans(run,parent)]
        write(attempt/'production-start.json',started)
        with (attempt/'production.log').open('x') as log:
            def line(text): log.write(text+'\n'); log.flush(); print(text,flush=True)
            result=run_streaming(command,cwd=W,env=env,timeout=3600,label='generic-completed-energy-parent',
                line_callback=line,cancel_event=cancel,should_cancel=lambda:cancel.is_set() or bool(errors) or leases.cancelled,
                process_registry=processes,before_terminate=lambda:leases.cancel_all(grace_s=3))
        write(attempt/'production-result.json',{'finished_at':now(),'returncode':result.returncode})
        if result.returncode or cancel.is_set():
            raise RuntimeError('Generic capture stopped; preserve every attempt; no automatic retry')
        final=read(attempt/'normal_result.json')
        if (final.get('normal_report_verified') is not True
                or (final.get('energy_efficiency_claim_eligible') is not True
                    and parent['performance_plan']['path'] != str(FINAL_PLAN)
                    and parent.get('performance_admission',{}).get('followup_plan') != parent['performance_plan'])):
            raise RuntimeError('Normal Generic energy final admission absent')
    except BaseException as exc:
        primary=exc
        if not resuming or attempt!=run:
            write(attempt/'failure.json',{'at':now(),'error':str(exc),'automatic_restart':False})
        raise
    finally:
        try:
            if cancel.is_set() or leases.active_count(): leases.cancel_all(grace_s=3)
            if processes.remaining_owned_count(): processes.terminate_all(grace_s=3)
        except BaseException as exc: errors.append('owned cleanup: '+str(exc))
        stop.set()
        if thread is not None: thread.join(timeout=5)
        try:
            if broker is not None and (thread is None or not thread.is_alive()): broker.close()
        except BaseException as exc: errors.append('broker cleanup: '+str(exc))
        pending=leases.active_count()+processes.remaining_owned_count()
        if broker is not None and broker.stopped_sources: errors.append('stopped physical resources: '+repr(sorted(broker.stopped_sources)))
        unproven=bool(pending or errors or (thread is not None and thread.is_alive()))
        if unproven: lock.commit_quarantine_fence({'reason':'generic_energy_cleanup_unproven','pending':pending,'errors':errors})
        for signum,handler in handlers.items(): signal.signal(signum,handler)
        if not resuming or attempt!=run:
            write(attempt/'cleanup-status.json',{'at':now(),'cleanup_proven':not unproven,
                'owned_remote_active':leases.active_count(),'owned_local_active':processes.remaining_owned_count(),
                'errors':errors,'foreign_processes_stopped':False})
        if resuming and attempt!=run:
            write(run/'LATEST_ATTEMPT.json',{'at':now(),'attempt_dir':str(attempt),
                'cleanup':str(attempt/'cleanup-status.json'),'normal_result':str(attempt/'normal_result.json'),
                'normal_plan':str(attempt/'measurement/generic_energy_plan.json')})
        lock.release()
        if unproven and primary is None: raise RuntimeError('Terminal ownership cleanup unproven; no restart')


def readiness(args):
    # No controller/DUT/source action and no budget mutation.
    plan=read(DEFAULT_PLAN)
    records=[]
    for row in plan['rows']:
        path=Path(row['result_path'])
        records.append({'identity':row['identity'],'performance_result':str(path),'performance_present':path.is_file(),
                        'energy_ready':False,'status':'READY_TO_PREPARE_AFTER_PERFORMANCE' if not path.is_file() else 'REQUIRES_NORMAL_ADMISSION_AND_PREPARE'})
    result={'schema':'revision3/generic-energy-adapter-readiness/v1','at':now(),'adapter':file_binding(ADAPTER_PATH),
        'performance_plan':file_binding(DEFAULT_PLAN),'rows':records,'energy_ready_points':0,'hardware_starts':0,
        'budget_modified':False,'scope':'one explicit case per parent; exactly three Single captures; serial coordination'}
    write(args.output,result); return result


def main(argv=None, *, adapter_path=None):
    parser=argparse.ArgumentParser(description=__doc__); sub=parser.add_subparsers(dest='action',required=True)
    p=sub.add_parser('prepare'); p.add_argument('--plan',default=None); p.add_argument('--setup',choices=SETUPS,required=True)
    p.add_argument('--case',required=True,help='Unique short case or existing full case_key.'); p.add_argument('--run-dir',required=True)
    p.add_argument('--registry',default=str(DEFAULT_REGISTRY))
    p=sub.add_parser('check'); p.add_argument('--run-dir',required=True);p.add_argument('--allow-started',action='store_true')
    p=sub.add_parser('run'); p.add_argument('--run-dir',required=True); p.add_argument('--slot-handoff',required=True)
    p.add_argument('--single-parent-attempt',action='store_true',help=argparse.SUPPRESS)
    p=sub.add_parser('resume'); p.add_argument('--run-dir',required=True);p.add_argument('--slot-handoff',required=True)
    p.add_argument('--previous-attempt',default='')
    p.add_argument('--single-parent-attempt',action='store_true',help=argparse.SUPPRESS)
    p=sub.add_parser('refresh-controller');p.add_argument('--run-dir',required=True)
    p=sub.add_parser('worker'); p.add_argument('--run-dir',required=True);p.add_argument('--attempt-dir')
    p=sub.add_parser('readiness'); p.add_argument('--output',required=True)
    args=parser.parse_args(argv)
    configure(getattr(args,'run_dir',None), plan_path=getattr(args,'plan',None), adapter_path=adapter_path)
    if args.action=='prepare':
        args.plan=args.plan or str(DEFAULT_PLAN)
        prepare(args)
    elif args.action=='check':
        if args.allow_started:latest_clean_attempt(Path(args.run_dir).resolve())
        verify_parent(args.run_dir,require_unstarted=not args.allow_started,allow_completed_repeats=args.allow_started)
        print('CHECK_OK; zero hardware starts')
    elif args.action in ('run','resume'): run_prepared(args)
    elif args.action=='refresh-controller':refresh_controller(args);print('CONTROLLER_REFRESH_OK; zero hardware starts')
    elif args.action=='worker': worker(args.run_dir,args.attempt_dir)
    else: readiness(args)
    return 0


import math
from onnx_splitpoint_tool.native_command_contract import verify_native_command_contract
from onnx_splitpoint_tool.native_detection_postprocess import verify_detection_completion_execution_contract


def check_final_local_sources(freeze,local_runtime,*,frozen_bundle=None):
 """Keep original DUT snapshots exact; bind only explicit local report changes."""
 snapshot=Path(freeze['snapshot']).resolve();local_runtime=Path(local_runtime).resolve()
 reporting=None
 bundle_verified=False
 allowed={'scripts/native_producer_final_report.py',
          'onnx_splitpoint_tool/workflow/cross_runner_reporting.py',
          'scripts/native_full_baseline_eval_runner.py'}
 scoped_controller={'onnx_splitpoint_tool/energy/caller_managed_reporting.py',
     'onnx_splitpoint_tool/energy/collector.py','onnx_splitpoint_tool/energy/recovery_accounting.py',
     'onnx_splitpoint_tool/energy/task_budget.py','onnx_splitpoint_tool/generic_energy_reporting.py',
     'onnx_splitpoint_tool/remote/resource_recovery.py','onnx_splitpoint_tool/generic_completion_reporting.py',
     'onnx_splitpoint_tool/native_detection_postprocess.py','onnx_splitpoint_tool/native_energy_reporting.py',
     'onnx_splitpoint_tool/workflow/generic_completed_task.py'}
 quality_templates={'onnx_splitpoint_tool/resources/templates/benchmark_suite.py.txt',
                    'onnx_splitpoint_tool/resources/templates/run_split_onnxruntime.py.txt'}
 allowed|=scoped_controller|quality_templates
 for item in freeze['runtime_files']:
  rel=item['relative_path'];expected=item['sha256']
  if sha(snapshot/rel)!=expected:raise RuntimeError('frozen final snapshot changed: '+rel)
  actual=sha(local_runtime/rel)
  if actual==expected:continue
  if local_runtime!=W.resolve() or rel not in allowed:raise RuntimeError('frozen final source changed: '+rel)
  if reporting is None:
   reporting=read(R/'REPORTING_SOURCE_SUPPLEMENT.json').get('final_runtime_controller_changes',{})
   final=R/'final_implementation_20261008/IMPLEMENTATION_FREEZE.json'
   if (not(reporting.get('source_dependencies_only_report') is True
           or reporting.get('scoped_local_source_changes') is True)
       or reporting.get('DUT_files_changed') is not False
       or reporting.get('parent_freeze_sha256')!=sha(final)
       or read(final)!=freeze):raise RuntimeError('final controller reporting supplement differs')
  matches=[x for x in reporting.get('changes',[]) if x.get('path')==rel]
  if (len(matches)!=1 or matches[0].get('before_sha256')!=expected
      or matches[0].get('after_sha256')!=actual
      or sha(matches[0]['candidate_path'])!=actual):
   raise RuntimeError('local controller report bytes not bound: '+rel)
  if rel in scoped_controller and matches[0].get('scope')!='controller_or_postflight':
   raise RuntimeError('controller change scope missing: '+rel)
  if rel in quality_templates:
   if (matches[0].get('scope')!='isolated_quality_template'
       or reporting.get('original_measured_bundles_changed') is not False
       or not isinstance(frozen_bundle,dict)):
    raise RuntimeError('Quality template requires an unchanged measured bundle: '+rel)
   if not bundle_verified:
    bundle=Path(frozen_bundle.get('path',''))
    original=read(R/'generic_energy_extension/confirmation_final_20261008/BUNDLE_PREPARATION.json')
    if (original.get('final_source_freeze')!={'path':str(final),'sha256':sha(final)}
        or original.get('normal_scope_validation')!='pass'
        or bundle.resolve()!=Path(original.get('bundle','')).resolve()
        or frozen_bundle.get('sha256')!=original.get('bundle_sha256')
        or not bundle.is_file()or bundle.is_symlink()
        or sha(bundle)!=original['bundle_sha256']):
     raise RuntimeError('original measured Generic bundle changed')
    bundle_verified=True

def final_sources(freeze_path,bundle_path):
 freeze_path=Path(freeze_path).resolve(strict=True);bundle_path=Path(bundle_path).resolve(strict=True)
 expected=R/'final_implementation_20261008/IMPLEMENTATION_FREEZE.json'
 if freeze_path!=expected:raise ValueError('exact final implementation freeze required')
 freeze=read(freeze_path)
 if freeze.get('validated') is not True:raise ValueError('final source freeze not validated')
 preparation=read(R/'generic_energy_extension/confirmation_final_20261008/BUNDLE_PREPARATION.json')
 binding={'path':str(freeze_path),'sha256':sha(freeze_path)}
 if (preparation['final_source_freeze']!=binding or preparation['normal_scope_validation']!='pass'
     or Path(preparation['bundle']).resolve()!=bundle_path or sha(bundle_path)!=preparation['bundle_sha256']):
  raise ValueError('exact existing final generated bundle binding differs')
 check_final_local_sources(freeze,W,frozen_bundle={'path':str(bundle_path),'sha256':preparation['bundle_sha256']})
 return binding

def prepared(command,task):
 contract=command['prepared_input_contract'];entries=contract.get('entries') or [{'name':contract.get('name') or ('images' if task=='detection' else 'input'),'artifact_name':'prepared_input','shape':contract.get('shape'),'dtype':contract.get('dtype')}]
 if len(entries)!=1:raise ValueError('one exact declared prepared feed required')
 e=entries[0];a=command['artifacts'][e['artifact_name']]
 if not isinstance(e.get('shape'),list) or len(e['shape'])!=3 or e['shape'][-1]!=3 or e['dtype']!='uint8':raise ValueError('ordinary completed entry does not support this prepared feed')
 if contract.get('source_image_sha256')!=command['input_image_sha256']:raise ValueError('prepared source image conflict')
 return contract,{**e,**a,'remote_path':a['path']}

def planned_canonical_rgb(row,nplan,commands,prepared_specs,normalized,proofs):
 # Preserve existing planner declarations. The original three H10-Y7 final
 # plans predate this field but already bind the exact same canonical bytes.
 if 'canonical_rgb_sha256' in nplan['expected']:return nplan['expected']['canonical_rgb_sha256']
 if (row['model_id'],row['setup_id'],row['direction'],row['task'])!=('yolov7_paper','orin_nx_hailo10_01','hailo10h_to_trt','detection') or row['case_id'] not in {'b009','b044','b066'}:
  raise ValueError('Native planned canonical RGB declaration missing')
 from onnx_splitpoint_tool.preprocessing_contract import preprocessing_contract_sha256
 parity_rows=read(R/'archive_checks/historical_prepared_rgb_parity.json')['rows']
 matches=[p for p in parity_rows if (p['model'],p['setup'])==(row['model_id'],row['setup_id'])]
 if len(matches)!=1:raise ValueError('H10-Y7 existing prepared RGB parity missing or ambiguous')
 parity=matches[0];timing=normalized[0].get('timing_input_binding') or {}
 if (parity.get('exact_equal') is not True or parity.get('changed_channels')!=0
     or parity.get('historical_rgb_sha256')!=parity.get('canonical_rgb_sha256')
     or sha(parity['old_prepared_path'])!=parity['old_prepared_file_sha256']
     or parity['source_image_sha256']!=nplan['expected']['input_image_sha256']
     or timing.get('canonical_rgb_sha256')!=parity['canonical_rgb_sha256']
     or timing.get('input_image_sha256')!=parity['source_image_sha256']
     or timing.get('geometry')!=parity['geometry']
     or timing.get('preprocessing_contract_sha256')!=preprocessing_contract_sha256(timing.get('preprocessing_contract') or {})):
  raise ValueError('H10-Y7 exact prepared RGB parity or preprocessing differs')
 for command,(contract,entry),normal,proof in zip(commands,prepared_specs,normalized,proofs):
  if (normal.get('timing_input_binding')!=timing or contract.get('timing_input_binding')!=timing
      or read(proof['raw_report']).get('timing_input_binding')!=timing
      or command['input_image_sha256']!=parity['source_image_sha256']
      or entry['sha256']!=parity['old_prepared_file_sha256']
      or entry['dtype']!=parity['old_dtype'] or entry['shape']!=parity['old_shape']):
   raise ValueError('H10-Y7 signed prepared input or timing mirror differs')
 return parity['canonical_rgb_sha256']

def bind_fresh_reference(original_row,dep):
 row=copy.deepcopy(original_row);np=Path(dep['native_plan'])
 if sha(np)!=dep['native_plan_sha256']:raise ValueError('Native planned binding changed')
 nplan=read(np);summary_path=Path(dep['required_native_summary'])
 if not summary_path.is_file():return None
 summary=read(summary_path);records=summary.get('rows',[])
 if summary.get('status')!='complete' or summary.get('plan_sha256')!=sha(np) or len(records)!=3 or {r['repeat'] for r in records}!={1,2,3}:raise ValueError('Native completed three-process cohort not bound')
 if len({r['runner_pid'] for r in records})!=3 or len({r['runner_start_monotonic_ns'] for r in records})!=3:raise ValueError('Native process independence missing')
 proofs=[];commands=[];prepared_specs=[];normalized=[]
 for r in records:
  p=Path(r['raw_report']);n=Path(r['report']);success=n.parent/'success.json';receipt=read(success)
  if sha(p)!=receipt['raw_report_sha256'] or sha(n)!=receipt['report_sha256'] or not all(receipt['checks'].values()):raise ValueError('Native successful bytes no longer match')
  raw=read(p);command,status=verify_native_command_contract(raw.get('native_command_contract'))
  if command is None:raise ValueError('Native command invalid: '+status)
  if (command['model'],command['case'],command['setup_id'],command['precision'])!=(row['model_id'],row['case_id'],row['setup_id'],row['precision']):raise ValueError('Native exact identity mismatch')
  if command['input_image_sha256']!=nplan['expected']['input_image_sha256']:raise ValueError('Native source image mismatch')
  commands.append(command);prepared_specs.append(prepared(command,row['task']));normal=read(n);normalized.append(normal)
  if normal.get('completed_work_units')!=1000 or normal.get('completed_frames')!=1000 or normal.get('postprocess_completed_frames')!=1000 or normal.get('postprocess_completion_verified') is not True:raise ValueError('Native measured completed-task count missing')
  proofs.append({'repeat':r['repeat'],'raw_report':str(p),'raw_report_sha256':sha(p),'normalized_report':str(n),'normalized_report_sha256':sha(n),'success_receipt':str(success),'success_receipt_sha256':sha(success),'runner_pid':r['runner_pid'],'runner_start_monotonic_ns':r['runner_start_monotonic_ns'],'native_command_contract_sha256':command['contract_sha256']})
 feed_keys=[(p[1]['sha256'],p[1]['shape'],p[1]['dtype']) for p in prepared_specs]
 if any(k!=feed_keys[0] for k in feed_keys):raise ValueError('Native repetition prepared feeds differ')
 command=commands[0];prepared_contract,entry=prepared_specs[0]
 for role,native_role in [('engine','engine'),('part1_runtime','dxnn' if row['direction']=='deepx_to_trt' else 'hef')]:
  actual=command['artifacts'][native_role];old=row['expected_runtime_artifacts'][role]
  if actual['path']!=old['path'] or actual['sha256']!=old['sha256']:raise ValueError('existing Generic engine or accelerator artifact changed')
 row['historical_native_reference']={'source':row['native_source'],'command_contract':row['native_command_contract'],'completion_contract':row.get('completion_contract'),'input_binding':row['input_binding'],'functional_reference':row['functional_reference']}
 row.update(native_source=records[0]['raw_report'],native_command_contract=command,native_command_contract_sha256=command['contract_sha256'],native_quality_binding_sha256=command['native_split_quality_binding']['binding_sha256'])
 current_input={'prepared_input_contract':prepared_contract,'entries':[entry],'source_image':command['input_image'],'source_image_sha256':command['input_image_sha256'],'sequence':'single_exact_prepared_feed_repeated','batch_size':1,'storage_at_timer_start':'host_memory_prepared_runtime_tensor','preprocessing_timed':False,'runtime_adapter_must_preserve_exact_prepared_bytes':True,'source':row['native_source'],'local_prepared_file_status':'remote_exact_bytes_verified_by_bound_native_artifact_and_generic_preflight'}
 if row['task']=='classification' and prepared_contract.get('preprocess')=='shared_prepared_rgb_uint8':
  # The normal classifier reports bind the shared RGB bytes in their signed
  # prepared contract, rather than the detector-only timing_input_binding.
  canonical=nplan['expected'].get('canonical_rgb_sha256')
  for contract,feed in prepared_specs:
   if (not canonical or contract.get('preprocess')!='shared_prepared_rgb_uint8'
       or contract.get('shared_rgb_source_sha256')!=canonical or feed['sha256']!=canonical):
    raise ValueError('new Native classification shared prepared RGB proof missing or differs from plan')
  current_input['canonical_rgb_sha256']=canonical
 else:
  timing=normalized[0].get('timing_input_binding') or {}
  canonical=planned_canonical_rgb(row,nplan,commands,prepared_specs,normalized,proofs)
  for report in normalized:
   bound=report.get('timing_input_binding') or {}
   if not canonical or bound.get('canonical_rgb_sha256')!=canonical:raise ValueError('new Native canonical RGB proof missing or differs from plan')
   if bound.get('input_image_sha256')!=command['input_image_sha256'] or not bound.get('preprocessing_contract_sha256'):raise ValueError('new Native source image or preprocessing contract proof missing')
   if bound!=timing:raise ValueError('new Native repetition canonical input bindings differ')
  current_input.update({k:timing[k] for k in ['canonical_rgb_sha256','preprocessing_contract_sha256','input_image_sha256','image_sequence'] if k in timing})
 row['input_binding']=current_input
 manifest_artifact=command['artifacts']['semantic_output_manifest'];downloads=read(Path(records[0]['report']).parent/'download-proof.json');matches=[d for d in downloads if d['remote_path']==manifest_artifact['path']]
 if len(matches)!=1 or matches[0]['sha256']!=manifest_artifact['sha256'] or sha(matches[0]['local_path'])!=manifest_artifact['sha256']:raise ValueError('fresh Native functional manifest not downloaded and hash bound')
 row['functional_reference']={**row['functional_reference'],'source_report':row['native_source'],'output_manifest':matches[0]['local_path'],'output_manifest_sha256':manifest_artifact['sha256'],'role':'fresh_same_canonical_input_native_semantic_reference'}
 if row['task']=='detection':
  contracts=[verify_detection_completion_execution_contract(c['runtime_options']['completion_execution_contract']) for c in commands]
  if any(c!=contracts[0] for c in contracts):
   from onnx_splitpoint_tool.resources.remote_scripts.native_producer_final_report import _verified_repetition_operating_point
   points=[_verified_repetition_operating_point({**report,'operating_point_source_report':proof['normalized_report'],
     'report_sha256':proof['normalized_report_sha256']},require_observation=True)
     for report,proof in zip(normalized,proofs)]
   if (not all(points) or not all(p.get('completion_contract_provenance_normalized') is True for p in points)
       or len({p['operating_point_sha256'] for p in points})!=1):raise ValueError('Native repetition completion operating point differs or is not verified')
   if any(report.get('native_command_contract')!=command for report,command in zip(normalized,commands)):
    raise ValueError('Native normalized/raw original command differs')
  row['completion_contract']=contracts[0]
 times=[float(n['makespan_ms'])/1000 for n in normalized]
 if not all(math.isfinite(t) and t>0 for t in times):raise ValueError('Native makespan invalid')
 row['native_measurement_boundary']={**row['native_measurement_boundary'],'source':str(summary_path),'native_command_contract_sha256':command['contract_sha256'],'native_command_contract_sha256s':[c['contract_sha256'] for c in commands],'observed_completed_counts':[1000]*3,'observed_measurement_times_s':times,'process_repetition_policy':'separate_processes'}
 row['fresh_native_cohort']={'summary':str(summary_path),'summary_sha256':sha(summary_path),'plan':str(np),'plan_sha256':sha(np),'rows':proofs,'role':'fresh_performance_and_current_input_reference'}
 row['preparation_status']='current_native_cohort_bound';row['final_comparison_policy']={'legacy_pair_report_role':'historical_diagnostic_only','legacy_pair_ratio_is_final_result':False,'native_performance_source':'fresh_native_three_separate_process_repetitions'}
 return row

def verify_final_followup(row,plan):
 # The established follow-up entry binds a *new* Native input/reference via
 # bind_fresh_reference. The preserved-input nine-case entry is a different
 # admission; do not overwrite or reinterpret its historical input contract.
 import ast,typing
 from onnx_splitpoint_tool.workflow.generic_completed_task import runtime_configuration,bind_runtime_source_closure
 bundle=Path(plan['runner_bundle_source']);binding=final_sources(plan['final_source_freeze']['path'],bundle)
 if binding!=plan['final_source_freeze'] or sha(bundle)!=plan['runner_bundle_sha256']:raise ValueError('final follow-up source/bundle changed')
 native_plan=read(row['fresh_native_cohort']['plan'])
 if (Path(native_plan['source_seal']).resolve()!=Path(binding['path'])
     or native_plan['source_seal_sha256']!=binding['sha256']):raise ValueError('follow-up Native source freeze differs')
 configuration,files=runtime_configuration(row,row['remote_root'],plan['execution_id'])
 expected=copy.deepcopy(row);expected.update(runtime_config=configuration,staging_files=files)
 root=bundle.with_suffix('').with_suffix('')
 bind_runtime_source_closure(expected,root)
 if expected['runtime_config']!=row['runtime_config'] or files!=row['staging_files']:raise ValueError('normal follow-up runtime configuration changed')
 runner=root/'runner/run_split_onnxruntime.py';node=next(item for item in ast.parse(runner.read_text()).body
     if isinstance(item,ast.FunctionDef) and item.name=='_completion_validate_prepared_input_contract')
 namespace={'Mapping':typing.Mapping};exec(compile(ast.Module(body=[node],type_ignores=[]),str(runner),'exec'),namespace)
 backend={'orin_nx_deepx_m1_01':'deepx_m1','orin_nx_hailo8_01':'hailo8','orin_nx_hailo10_01':'hailo10h'}[row['setup_id']]
 namespace[node.name](row['runtime_config'],backend)
 return {'status':'normal_final_followup_configuration_and_metadata_verified','case_key':row['case_key'],'hardware_started':False}

def verified_cohort(row):
 dep={'native_plan':row['fresh_native_cohort']['plan'],'native_plan_sha256':row['fresh_native_cohort']['plan_sha256'],'required_native_summary':row['fresh_native_cohort']['summary']}
 bound=bind_fresh_reference(row,dep)
 if bound is None or bound['fresh_native_cohort']!=row['fresh_native_cohort']:raise ValueError('prepared fresh Native cohort changed')
 if bound['input_binding']!=row['input_binding'] or bound['functional_reference']!=row['functional_reference']:raise ValueError('prepared input or semantic reference changed')
 return row['fresh_native_cohort']

if __name__ == "__main__":
    main()

if __name__=="__main__": raise SystemExit(main())
