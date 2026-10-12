#!/usr/bin/env python3
"""Bounded external adapter over existing Native Energy and ownership APIs.

prepare uses the normal dry-run admission, which can include live DUT checks.
run starts real hardware and requires the
coordinator's explicit measurement-slot handoff. Explicit recovery/resume use the
regular product resource and budget APIs. No deployment or build.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import threading
import uuid
from datetime import datetime, timezone

TOOL = Path(__file__).resolve().parents[2]
ROOT = None
ADAPTER_PATH = Path(__file__).resolve()
from onnx_splitpoint_tool.energy.task_budget import final_cohort_authorization
from onnx_splitpoint_tool.resources.remote_scripts.run_native_producer_energy_from_summary import _p03_resume_admission


def configure(run, *, adapter_path=None):
    global ROOT, TOOL, ADAPTER_PATH
    from .energy_parent_flow import parent_context
    context = parent_context(run, adapter_path=adapter_path)
    ROOT, TOOL, ADAPTER_PATH = context['report_root'], context['controller_root'], context['adapter']
    return context

SETUPS = {
    'orin_nx_hailo8_01': ('hailo8', 'native_full_hailo8', 'hailo8_to_trt'),
    'orin_nx_hailo10_01': ('hailo10', 'native_full_hailo10h', 'hailo10h_to_trt'),
    'orin_nx_deepx_m1_01': ('deepx', 'native_full_deepx', 'deepx_to_trt'),
}


def _final_resume_previous_results(run, prior, started):
    """Keep the latest predecessor; follow only its proven zero-capture pointer."""
    run, prior = Path(run).resolve(), Path(prior).resolve()
    argv = started.get('argv')
    if not isinstance(argv,list) or any(not isinstance(x,str) for x in argv):
        raise ValueError('normal predecessor argv is invalid')
    def option(name):
        occurrences=[i for i,x in enumerate(argv) if x==name or x.startswith(name+'=')]
        if len(occurrences)!=1:raise ValueError('normal predecessor option must be unique: '+name)
        i=occurrences[0]
        if argv[i].startswith(name+'='):return argv[i].split('=',1)[1]
        if i+1>=len(argv) or argv[i+1].startswith('--'):raise ValueError('normal predecessor option is missing: '+name)
        return argv[i+1]
    output=Path(option('--out-dir'))
    if not output.is_absolute() or output.resolve()!=output or not output.is_relative_to(run/'measurement'):
        raise ValueError('normal predecessor output is outside this Parent measurement')
    for name in ('native_producer_energy_results.json','native_producer_energy_results.partial.json'):
        candidate=output/name
        if candidate.is_file() and not candidate.is_symlink():return candidate
    if (prior.parent!=run/'resume_attempts' or output!=run/'measurement/resume_attempts'/prior.name
            or started.get('run_id')!=run.name):
        raise ValueError('normal durable predecessor result is missing')
    cleanup=read(prior/'cleanup-status.json')
    if (cleanup.get('cleanup_proven')is not True or cleanup.get('owned_remote_active')!=0
            or cleanup.get('owned_local_active')!=0 or cleanup.get('errors')):
        raise ValueError('zero-capture predecessor cleanup is unproven')
    failure_path=output/'resume_failure.json'
    if not failure_path.is_file() or failure_path.is_symlink():raise ValueError('zero-capture predecessor failure is missing')
    failure=read(failure_path)
    if (failure.get('status')!='resume_preparation_or_cohort_preflight_failed'
            or failure.get('ok')is not False or failure.get('complete')is not False
            or failure.get('resume_merge_published')is not False
            or any(type(failure.get(k))is not int or failure[k]!=0 for k in
                ('measurement_wrapper_started_count','started_measurement_count',
                 'collector_started_repeat_count','workload_started_repeat_count'))):
        raise ValueError('predecessor is not a proven zero-capture preflight failure')
    preparation=failure.get('resume_preparation')or{}
    binding=preparation.get('cohort_preflight')or{}
    cohort_path=output/'resume_cohort_preflight/resume_cohort_preflight.json'
    if (preparation.get('ok')is not False or preparation.get('measurement_wrapper_allowed')is not False
            or binding.get('report_path')!=str(cohort_path)
            or not cohort_path.is_file() or cohort_path.is_symlink()
            or binding!=read(cohort_path)):
        raise ValueError('zero-capture predecessor preflight evidence binding is invalid')
    cohort=read(cohort_path)
    from onnx_splitpoint_tool.resume_cohort_preflight import _canonical_sha256
    unsigned=dict(cohort);unsigned.pop('report_sha256',None)
    if cohort.get('report_sha256')!=_canonical_sha256(unsigned):
        raise ValueError('zero-capture predecessor preflight seal is invalid')
    if (cohort.get('status')!='failed' or cohort.get('ok')is not False
            or cohort.get('measurement_wrapper_allowed')is not False
            or any(cohort.get(k)is not False for k in
                   ('collector_started','measurement_wrapper_started','workload_started'))
            or any((output/'measurements').glob('**/energy_summary.json'))
            or any((output/'measurements').glob('**/energy_aggregate.json'))
            or read(prior/'production-result.json').get('returncode')!=2):
        raise ValueError('zero-capture predecessor physical start cannot be ruled out')
    previous=Path(option('--previous-results'))
    if (not previous.is_absolute() or previous.resolve()!=previous or previous.is_symlink()
            or not previous.is_relative_to(run/'measurement') or previous.is_relative_to(output)
            or previous.name not in {'native_producer_energy_results.json','native_producer_energy_results.partial.json'}
            or not previous.is_file()):
        raise ValueError('zero-capture predecessor previous-results pointer is invalid')
    return previous


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    from onnx_splitpoint_tool.benchmark.schema import write_json_atomic
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(Path(path), value)


def identity(row):
    return tuple(str(row.get(k) or '') for k in ('backend', 'model', 'case', 'setup_id'))


def _finish_dispatcher_result(run, config, result_output, returncode, cancelled, final_resume):
    """Close only the intended prefix; retain the actual normal result bytes."""
    source = result_output/'native_producer_energy_results.json'
    limited_success = False
    if config.get('capture_limit') and returncode == 2 and not cancelled and source.is_file():
        report = read(source)
        observations = report.get('rows', [])
        captured = [tuple(k) for k in config['capture_identities']]
        expected = [tuple(k) for k in config['expected_identities']]
        active = [x for x in observations if identity(x['row']) in captured]
        context = [x for x in observations if identity(x['row']) not in captured]
        def context_not_started(item):
            if item.get('skipped') != 'limit':return False
            observed_run = item.get('run') or {}
            if not observed_run:return True
            if set(observed_run) != {'measurement_start_observation'}:return False
            observed = observed_run['measurement_start_observation']
            return (isinstance(observed, dict)
                and all(observed.get(k) is False for k in
                    ('measurement_started', 'aggregate_verified_fallback', 'energy_summary_present'))
                and all(type(observed.get(k)) is int and observed[k] == 0 for k in
                    ('aggregate_run_count', 'collector_started_repeat_count',
                     'repeat_summary_file_count', 'workload_started_repeat_count',
                     'workload_timing_start_count')))
        limited_success = (
            len(observations) == len(expected)
            and sorted(identity(x['row']) for x in observations) == sorted(expected)
            and len(set(captured)) == len(captured) == config['capture_limit']
            and len(active) == len(captured)
            and all(x.get('ok') is True for x in active)
            and all(context_not_started(x) for x in context))
        if config.get('campaign_cohort') or config.get('require_environment_observation'):
            limited_success = limited_success and all(
                (x.get('run') or {}).get('energy_aggregate_complete') is True
                and (x.get('run') or {}).get('energy_aggregate_verified') is True
                for x in active)
        write(run/'bounded-result.json', {'at':now(), 'normal_dispatcher_returncode':returncode,
            'normal_dispatcher_overall_status':report.get('status'),
            'intended_capture_rows_verified':limited_success,
            'capture_count':len(active), 'context_rows_not_started':len(context),
            'normal_results_path':str(source), 'normal_results_sha256':sha(source)})
    if (returncode and not limited_success) or cancelled:
        raise RuntimeError('energy dispatcher failed or cancelled; preserve all partial evidence')
    if final_resume:
        import shutil
        # rc=2 is the normal context-limit result, after the complete prefix has
        # passed above. Copy without changing its overall ok=False or row paths.
        shutil.copy2(source, run/'measurement/native_producer_energy_results.json')
    return limited_success


def verify_yolov7_energy_context_reuse(rows, capture_identities):
    """Verify only the three already Current YOLOv7 Fulls outside the prefix."""
    from onnx_splitpoint_tool.resources.remote_scripts.native_producer_energy_plan import _canonical_json_sha256
    permitted = {('native_full_deepx', 'yolov7_paper', 'full', 'orin_nx_deepx_m1_01'),
                 ('native_full_tensorrt', 'yolov7_paper', 'full', 'orin_nx_deepx_m1_01'),
                 ('native_full_hailo8', 'yolov7_paper', 'full', 'orin_nx_hailo8_01')}
    if capture_identities is None:
        raise ValueError('Native energy reuse requires an exact capture prefix')
    captures = [tuple(k) for k in capture_identities]
    if [identity(row) for row in rows[:len(captures)]] != captures:
        raise ValueError('Native energy reuse capture prefix differs')
    contexts = {identity(row) for row in rows[len(captures):]} & permitted
    if set(captures) & permitted:
        raise ValueError('Already valid Native Full energy cannot be recaptured')
    if not contexts:
        return set(), []
    pointer = read(ROOT/'CURRENT_NORMAL_ENERGY.json')
    receipt_path = Path(pointer['receipt']).resolve()
    generation = Path(pointer['generation']).resolve()
    if (not generation.is_relative_to(ROOT/'reporting_caller_managed_energy/exports')
            or receipt_path != generation/'RECEIPT.json'
            or sha(receipt_path) != pointer['receipt_sha256']):
        raise ValueError('Original normal energy Current receipt differs')
    receipt = read(receipt_path)
    scientific = receipt['files']['scientific_energy_rows.json']
    scientific_path = Path(scientific['path']).resolve()
    original_import = Path(receipt['prior_receipt']).resolve()
    expected_source = ROOT/'energy/native_full_remaining5_qualified/measurement/native_producer_energy_results.json'
    if (scientific_path != generation/'scientific_energy_rows.json'
            or sha(scientific_path) != scientific['sha256']
            or not original_import.is_relative_to(ROOT/'quality_transfer/qualified_energy_import/native_full_remaining5_qualified')
            or sha(original_import) != receipt['prior_receipt_sha256']):
        raise ValueError('Original Native Full energy import binding differs')
    original = read(original_import)
    original_summary = Path(original['summary']).resolve()
    if (Path(original['source']).resolve() != expected_source
            or sha(expected_source) != original['source_sha256']
            or sha(original_summary) != original['summary_sha256']
            or original.get('actual_source_bytes_and_all_row_plan_run_payloads_unchanged') is not True):
        raise ValueError('Original Native Full energy evidence changed')
    original_rows = {identity(row): row for row in read(original_summary)['rows']}
    energy_rows = read(scientific_path)
    checks = [{'path': str(p), 'sha256': sha(p), 'context_only': True}
              for p in (receipt_path, scientific_path, original_import, expected_source, original_summary)]
    for row in rows[len(captures):]:
        key = identity(row)
        if key not in contexts:
            continue
        if _canonical_json_sha256(row) != _canonical_json_sha256(original_rows[key]):
            raise ValueError('Native Full context differs from original imported row')
        matches = [x for x in energy_rows if (x.get('backend'), x.get('model_id'),
                   x.get('case_id'), x.get('setup_id')) == key]
        proof = [x for x in original['proofs'] if tuple(x['identity']) == key]
        if (len(matches) != 1 or len(proof) != 1
                or any(matches[0].get(k) is not True for k in
                       ('energy_eligible', 'energy_quality_qualified', 'completion_pairing_eligible'))
                or matches[0].get('diagnostic_only') is not False
                or matches[0].get('screening_only') is not False):
            raise ValueError('Native Full energy is not normal qualified Current')
        proof = proof[0]
        if sha(proof['aggregate']) != proof['aggregate_sha256']:
            raise ValueError('Original Native Full energy aggregate changed')
        repeats = proof['repeat_proofs']
        if (len(repeats) != 3 or {x['logical_repeat'] for x in repeats} !=
                {'repeat:0', 'repeat:1', 'repeat:2'}):
            raise ValueError('Original Native Full energy repeats incomplete')
        for rep in repeats:
            if (rep.get('source_completion_verified') is not True
                    or rep.get('completed_tasks', 0) <= 0 or sha(rep['path']) != rep['sha256']):
                raise ValueError('Original Native Full energy repeat binding differs')
            checks.append({'path':rep['path'], 'sha256':rep['sha256'], 'context_only':True})
        checks.append({'path':proof['aggregate'], 'sha256':proof['aggregate_sha256'], 'context_only':True})
    return contexts, checks


def validate_scope(rows, phase, setups, *, context_only_identities=(), capture_identities=None):
    if not rows or not setups or set(setups) - SETUPS.keys():
        raise ValueError('empty or unknown setup scope')
    if phase != 'final':
        raise ValueError('Revision3 admits only audited new final points')
    import csv
    with (ROOT/'NATIVE_REMEASUREMENT_PLAN.csv').open(newline='') as handle:
        audit = list(csv.DictReader(handle))
    allowed = {(r['backend'],r['model'],r['case'],r['setup']) for r in audit
               if r.get('energy_decision',r['decision'])=='remeasure_required' and r['setup'] in setups}
    contexts={tuple(x) for x in context_only_identities}
    h10_contexts={('native_full_hailo10h',model,'full','orin_nx_hailo10_01')
                  for model in ('mobilenet_v3_large','regnet_x_1_6gf','resnet50')}
    if not contexts.issubset(h10_contexts) or (contexts and setups!=['orin_nx_hailo10_01']):
        raise ValueError('Only original H10 Classifier baselines may be context-only')
    allowed.update(contexts)
    actual = [identity(row) for row in rows]
    reused_contexts = set(actual) - allowed
    if reused_contexts:
        reuse_scope = {(r['backend'], r['model'], r['case'], r['setup']) for r in audit
                       if r.get('energy_decision') == 'reuse_verified_current_energy'
                       and r['setup'] in setups}
        if not reused_contexts.issubset(reuse_scope):
            raise ValueError('Native energy reuse context outside exact audited reuse scope')
        verified_contexts, _ = verify_yolov7_energy_context_reuse(rows, capture_identities)
        if not reused_contexts.issubset(verified_contexts):
            raise ValueError('unverified Native energy reuse context')
        allowed.update(reused_contexts)
    expected=set(actual)
    if len(actual)!=len(expected) or not expected.issubset(allowed):
        raise ValueError('summary contains duplicate or unaudited operating points')
    if {r[3] for r in expected} != set(setups):
        raise ValueError('setup scope must exactly match selected rows')
    if any(row.get('ok') is not True for row in rows):
        raise ValueError('all input Native rows must be successful original observations')
    return sorted(expected)


def targets_from_registry(registry, wanted):
    from onnx_splitpoint_tool.platform_power import host_config_from_setup
    targets = {}
    for setup in wanted:
        matches = [r for r in registry['hardware_setups'] if r.get('id') == setup]
        if len(matches) != 1:
            raise ValueError('missing or duplicated registry setup ' + setup)
        row = matches[0]
        host = host_config_from_setup(row)
        targets[setup] = {**row, 'remote': {'host': host.host, 'user': host.user,
            'port': host.port, 'remote_base_dir': host.remote_base_dir,
            'ssh_extra_args': host.ssh_extra_args}}
    return targets


def verify_fresh_before_reports(payload):
    """Accept reused immutable contracts, never reused performance outcomes."""
    rows = payload.get('rows', [])
    proof = payload.get('fresh_performance_provenance') or (rows[0].get('fresh_performance_provenance') if len(rows)==1 else {}) or {}
    if (proof.get('phase') != 'h8_before' or proof.get('old_measurements_reused') is not False
            or Path(str(proof.get('cohort_root') or '')).resolve() != ROOT):
        raise ValueError('fresh H8 before performance provenance missing')
    reports = [Path(p).resolve() for p in proof.get('repeat_reports', [])]
    if len(reports) != 3 or len(set(reports)) != 3 or any(not p.is_relative_to(ROOT) for p in reports):
        raise ValueError('need three distinct fresh local repeat reports inside cohort')
    checks, rates = [], []
    for path in reports:
        report = read(path)
        if (report.get('ok') is not True or report.get('completed_work_units') != 1000
                or report.get('postprocess_completed_frames') != 1000
                or report.get('warmup_count') != 100
                or report.get('requested_duration_s') != 0
                or report.get('postprocess_completion_verified') is not True):
            raise ValueError('fresh repeat does not prove 100 warmups/1000 completed tasks')
        rate = float(report['completed_work_units']) / float(report['measured_duration_s'])
        if abs(rate-float(report['fps_makespan'])) > 1e-8:
            raise ValueError('fresh raw rate differs from N/T')
        rates.append(rate)
        checks.append({'path':str(path),'sha256':sha(path),'fps':rate})
    if abs(statistics.median(rates)-float(rows[0]['fps_makespan'])) > 1e-8:
        raise ValueError('summary rate differs from three fresh raw rates')
    return checks


def capture_policy_preflight(config, result):
    """Check the declared policy; no capture or scientific claim is granted."""
    from onnx_splitpoint_tool.native_energy_quality_admission import verify_sealed_energy_quality_admission
    allowed = config.get('allow_diagnostic_capture') is True
    expected = [tuple(x) for x in config['capture_identities']]
    native_backends = {'native_full_tensorrt'} | {b for v in SETUPS.values() for b in v[1:]}
    decisions, actual, errors = [], [], []
    if result.get('status') != 'dry_run':
        errors.append('normal dry-run admission missing')
    for item in result.get('rows', []):
        if item.get('skipped') == 'limit':
            continue
        row = item.get('row') or {}
        row_errors = []
        try:
            key = identity(row)
            actual.append(key)
            if item.get('ok') is False or item.get('dry_run') is not True:
                raise ValueError('normal row preflight failed')
            if row['backend'] not in native_backends or _p03_resume_admission(row) is None:
                raise ValueError('native technical admission missing or wrong regime')
            verify_sealed_energy_quality_admission(row, required=True)
        except (KeyError, ValueError) as exc:
            key = tuple(row.get(k) for k in ('backend','model','case','setup_id'))
            row_errors.append(str(exc))
        diagnostic = row.get('diagnostic_only') is True
        admission = row.get('energy_quality_admission') or {}
        reason = str(admission.get('runtime_observation_reason') or '')
        for exclusion in (result.get('plan_payload') or {}).get('quality_pairing_posthoc_excluded_rows', []):
            if tuple(exclusion.get(k) for k in ('backend','model','case','setup_id')) == key:
                reason += ':'+str(exclusion.get('reason') or '')
        if diagnostic and not allowed:
            row_errors.append('final capture requires non-diagnostic admission: '+reason)
        elif not diagnostic and (row.get('diagnostic_only') is not False
                or row.get('energy_quality_qualified') is not True
                or row.get('claim_eligible') is not True):
            row_errors.append('final quality/claim preflight admission missing')
        decisions.append({'identity':list(key),'effective_policy':'diagnostic_only' if diagnostic else 'final_preflight',
                          'reason':reason,'errors':row_errors})
        errors.extend(row_errors)
    if not expected or actual != expected or len(set(actual)) != len(actual):
        errors.append('prepared capture scope missing, duplicated or changed')
    return {'ok':not errors,'allow_diagnostic_capture':allowed,
            'diagnostic_capture_count':sum(x['effective_policy']=='diagnostic_only' for x in decisions),
            'capture_completed':False,'energy_claim_granted':False,
            'capture_evidence_still_required':['exact completed tasks and runtime/input/scheduling binding',
                'calibrated sensor/command window, losses, drain and source closure',
                'global resource quiet and available clock/environment/thermal observations'],
            'rows':decisions,'errors':errors}


def require_capture_policy(config):
    result = read(Path(config.get('plancheck_root') or Path(config['run_dir'])/'plancheck')/'native_producer_energy_results.json')
    policy = capture_policy_preflight(config, result)
    if not policy['ok']:
        raise ValueError('capture policy rejected before start: '+json.dumps({'errors':policy['errors'],'rows':policy['rows']}))
    return policy


def command_for(config, output, *, dry_run=False):
    if not dry_run:
        # Also covers the existing resume adapter, before any collector start.
        require_capture_policy(config)
    cmd = [config['python'], '-u', str(TOOL/'scripts/run_native_producer_energy_from_summary.py'),
           '--summary', config['summary'], '--out-dir', str(output),
           '--hardware-setups-file', config['registry'],
           '--remote-root', config['remote_root'], '--remote-tool-dir', config['remote_tool_dir'],
           '--duration-s', '60', '--runs', '3', '--frames', '0', '--warmup', '0',
           '--timeout', '900', '--physical-scope', 'FS', '--window-label', 'command',
           '--require-runtime-work-units', '--require-command-window-alignment',
           '--campaign-budget-file', config['campaign_budget_file'],
           '--campaign-max-retries', '2', '--campaign-max-transport-failures', '20']
    for setup, target in config['targets'].items():
        key = SETUPS[setup][0]
        remote = target['remote']
        if int(remote['port']) != 22:
            raise ValueError('this bounded adapter has only verified port-22 targets')
        cmd += ['--'+key+'-ssh', remote['user']+'@'+remote['host'],
                '--'+key+'-env', config['remote_environments'][setup]]
    if config.get('validation_summary'):
        cmd += ['--validation-summary', config['validation_summary']]
    if config.get('capture_limit'):
        cmd += ['--limit', str(config['capture_limit'])]
    if config.get('campaign_cohort'):
        cmd += ['--campaign-cohort',config['campaign_cohort'],
                '--completed-row-validator',config['completed_row_validator']]
    elif config.get('require_environment_observation'):
        cmd += ['--require-environment-observation','--environment-run-id',Path(config['run_dir']).name,'--completed-row-validator',config['completed_row_validator']]
    if config.get('window_method_ab'):
        cmd += ['--window-method-ab-json', json.dumps(config['window_method_ab'], separators=(',', ':'))]
    if dry_run:
        cmd.append('--dry-run')
    return cmd



def _dx_classification_callback_source_update(source):
    """Verify the already recorded DX56 template update outside the Full callback."""
    import ast
    role='onnx_splitpoint_tool/resources/templates/benchmark_suite.py.txt'
    prior='75b8807f50f5523bfdccd048831ed15a332e1692ab731d6d48f15305c08e80f5'
    current='2379a5354ff890a9694922d7221bf154266ab684644f486189c867a2ad809a04'
    recorded=ROOT/'operations/token_saving_batch_20261009/continuous_20261010/dx56'
    before=recorded/'before'/role;after=recorded/'after'/role
    live=TOOL/role
    try:
        if (Path(source['current_path'])!=live or source['current_sha256']!=prior
                or source.get('classification_executed_prefix_ast_identical')is not True
                or sha(source['historical_path'])!=source['historical_sha256']
                or sha(before)!=prior or sha(after)!=current or sha(live)!=current):
            return False
        records=[x for x in read(recorded/'DX56_IMPLEMENTATION_RECEIPT.json')['source_files']
                 if x.get('role')==role]
        if (len(records)!=1 or records[0].get('sha256')!=current
                or records[0].get('snapshot')!=str(after)):
            return False
        callbacks=[]
        for path in (Path(source['historical_path']),before,after,live):
            matches=[n for n in ast.walk(ast.parse(path.read_text()))
                     if isinstance(n,ast.FunctionDef)and n.name=='_complete_full_task']
            if len(matches)!=1:return False
            node=matches[0]
            prefix=ast.Module(body=node.body[:3],type_ignores=[])
            digest=hashlib.sha256(ast.dump(prefix,include_attributes=False).encode()).hexdigest()
            if digest!=source.get('ast_sha256'):return False
            callbacks.append(ast.dump(node,include_attributes=False))
        # The accepted historical proof is prefix-scoped; the known new update
        # must preserve the entire formerly approved current callback as well.
        return callbacks[1]==callbacks[2]==callbacks[3]
    except (OSError,ValueError,KeyError,TypeError,SyntaxError):
        return False


def verify_dx_classification_reuse(row):
    """Check the three already accepted single-observation DX classifier reuses."""
    from onnx_splitpoint_tool.resources.remote_scripts.native_producer_energy_plan import _canonical_json_sha256
    permitted={('native_full_deepx',model,'full','orin_nx_deepx_m1_01')
               for model in ('mobilenet_v3_large','regnet_x_1_6gf','resnet50')}
    if identity(row) not in permitted or row.get('task')!='classification':
        raise ValueError('not one of the three verified DX classifier reuses')
    proof=row.get('reuse_provenance') or {}
    path=(ROOT/str(proof.get('receipt') or '')).resolve()
    if path!=(ROOT/'native_audit/CLASSIFICATION_FULL_REUSE.json').resolve() or sha(path)!=proof.get('receipt_sha256'):
        raise ValueError('existing classification reuse receipt missing or changed')
    receipt=read(path)['rows'][proof['row_index']]
    if identity(receipt)!=identity(row) or _canonical_json_sha256(receipt)!=proof.get('row_sha256'):
        raise ValueError('classification reuse row binding differs')
    if (proof.get('source_binding')!=receipt['source_binding']
            or proof.get('energy_disposition')!=receipt['energy_disposition']
            or proof.get('historical_protocol')!='existing_1000_completed_task_observation; no_new_three_process_claim'):
        raise ValueError('existing classification reuse proof differs')
    if (receipt['performance_disposition']!='reuse_existing_observation_unchanged_classification_timed_path'
            or receipt['energy_disposition']!='current_helper_timed_dispatch_changed_remeasurement_required'):
        raise ValueError('existing performance/energy reuse disposition differs')
    for role,source in receipt['source_binding'].items():
        for epoch in ('historical','current'):
            if sha(source[epoch+'_path'])!=source[epoch+'_sha256']:
                if (role!='performance_callback' or epoch!='current'
                        or not _dx_classification_callback_source_update(source)):
                    raise ValueError('approved classification source bytes changed: '+role)
    if (receipt['source_binding']['classification']['byte_identical'] is not True
            or receipt['source_binding']['performance_callback']['classification_executed_prefix_ast_identical'] is not True):
        raise ValueError('classification performance execution equivalence not verified')
    original=receipt['historical_command_contract'];contract=read(original['path'])
    if (sha(original['path'])!=original['file_sha256']
            or contract['contract_sha256']!=original['contract_sha256']):
        raise ValueError('historical classification command binding differs')
    if contract['contract_sha256']!=proof.get('historical_command_contract_sha256'):
        raise ValueError('historical classification command digest differs')
    actual=receipt['actual_performance_report'];raw=read(actual['path'])
    if sha(actual['path'])!=actual['sha256'] or actual['sha256']!=proof.get('historical_raw_report_sha256'):
        raise ValueError('historical classification performance bytes changed')
    if not isinstance(raw,list) or len(raw)!=1:
        raise ValueError('exact single historical classification observation required')
    report=raw[0];prepared=report['deepx_prepared_feed_benchmark']
    if (report.get('task')!='classification' or prepared.get('task')!='classification'
            or prepared.get('warmup_count')!=100 or prepared.get('completed_work_units')!=1000
            or report.get('completed_work_units')!=1000 or report.get('postprocess_completed_frames')!=1000
            or report.get('postprocess_completion_verified') is not True
            or prepared.get('sequence_policy')!='single_prepared_feed_repeated'
            or prepared.get('unique_input_images')!=1 or actual.get('concurrency')!=1):
        raise ValueError('original classification task/count/scheduling contract differs')
    if abs(float(row['fps_makespan'])-float(actual['fps']))>1e-8 or float(report['fps_makespan'])!=float(actual['fps']):
        raise ValueError('historical classification rate changed')
    inp=receipt['actual_input']
    if (inp.get('hash_match') is not True or sha(inp['path'])!=inp['expected_sha256']
            or inp['expected_sha256']!=inp['observed_sha256']):
        raise ValueError('historical classification input bytes differ')
    if inp['observed_sha256']!=receipt['canonical_rgb_sha256']:
        raise ValueError('canonical classification RGB differs')
    current=row['full_command_contract'];work=current['energy_workload'];prior=contract['energy_workload']
    document=dict(current);declared=document.pop('contract_sha256')
    if _canonical_json_sha256(document)!=declared or declared!=row.get('full_command_contract_sha256'):
        raise ValueError('current classification command contract binding differs')
    if (identity(current)!=identity(row) or work.get('task')!='classification'
            or row.get('measurement_concurrency')!=1 or current['runtime_options'].get('inflight')!=1):
        raise ValueError('current classification task or scheduling differs')
    for role in ('dxnn','source_onnx','runtime_input_tensor','input_manifest'):
        if current['artifacts'][role]['sha256']!=contract['artifacts'][role]['sha256']:
            raise ValueError('current classification artifact differs: '+role)
    # The normal current energy builder emits runtime artifacts, while the
    # original performance report remains bound above by its reuse receipt.
    if ('performance_report' in current['artifacts']
            and current['artifacts']['performance_report']['sha256']!=contract['artifacts']['performance_report']['sha256']):
        raise ValueError('current classification artifact differs: performance_report')
    for field in ('input_image_sha256','source_model_sha256','runtime_preprocessing_binding'):
        if current[field]!=contract[field]:raise ValueError('current classification binding differs: '+field)
    for field in ('kind','task','completed_task_stage','completed_task_contract_family','e2e_scope',
                  'classification_postprocess_required','input_contract','runtime_preprocessing_binding'):
        if work[field]!=prior[field]:raise ValueError('current classification workload differs: '+field)
    if current['artifacts']['hotloop_runner']['sha256']!=receipt['source_binding']['energy_callback']['current_sha256']:
        raise ValueError('current classification energy helper not admitted')
    if row.get('fresh_performance_provenance') or row.get('repetition_records'):
        raise ValueError('single historical observation cannot claim new repetitions')
    return {'path':actual['path'],'sha256':actual['sha256'],'fps':actual['fps'],
            'reuse_provenance':proof,'performance_observation_count':1,
            'completed_work_units':1000,'measurement_concurrency':1}


def verify_h10_classification_context_reuse(row, payload, capture_identities):
    """Verify one existing Vendor baseline strictly outside a fresh capture prefix."""
    import hashlib
    vendor_root = ROOT/'quality_transfer/final_h10_vendor_classifier_reuse_20261008'
    origins = {'performance_context':vendor_root/'normal_consumer_20261008/preserved_performance_context.json',
               'validation':vendor_root/'full_native_producer_validation_summary.json',
               'receipt':vendor_root/'QUALITY_REUSE_CONTEXT_RECEIPT.json'}
    key = identity(row)
    prefix = payload.get('revision3_capture_prefix') or {}
    captures = [list(x) for x in capture_identities]
    declarations = payload.get('revision3_context_only_h10_reuse') or []
    matches = [item for item in declarations if item.get('identity') == list(key)]
    if (key[0]!='native_full_hailo10h' or key[3]!='orin_nx_hailo10_01' or key[2]!='full'
            or key[1] not in {'mobilenet_v3_large','regnet_x_1_6gf','resnet50'}
            or row.get('task')!='classification' or list(key) in captures
            or prefix.get('identities')!=captures or type(prefix.get('count')) is not int
            or prefix['count']!=len(captures) or list(key) not in prefix.get('context_only_identities',[])
            or len(matches)!=1
            or not any(x==('native_full_tensorrt',key[1],'full',key[3]) for x in capture_identities)
            or not any(x[0]=='hailo10h_to_trt' and x[1]==key[1] and x[3]==key[3] for x in capture_identities)):
        raise ValueError('H10 Vendor reuse requires matching fresh baselines and exact context-only scope')
    binding = matches[0]
    for field, path in origins.items():
        if Path(binding.get(field,'')).resolve()!=path.resolve() or sha(path)!=binding.get(field+'_sha256'):
            raise ValueError('Original H10 Vendor context binding changed: '+field)
    original_rows=[x for x in read(origins['performance_context'])['rows'] if identity(x)==key]
    quality=[x for x in read(origins['validation'])['rows'] if identity(x)==key]
    receipt_rows=[x for x in read(origins['receipt'])['rows']
                  if tuple(x['identity'])==(key[1],key[3],key[0],key[2])]
    if len(original_rows)!=1 or original_rows[0]!=row or len(quality)!=1 or len(receipt_rows)!=1:
        raise ValueError('H10 Vendor context differs from the original observation')
    quality=quality[0]; bound=receipt_rows[0]; reuse=row['reuse_provenance']
    receipt_path=ROOT/'native_audit/CLASSIFICATION_FULL_REUSE.json'
    if (Path(ROOT/reuse['receipt']).resolve()!=receipt_path.resolve()
            or sha(receipt_path)!=reuse['receipt_sha256'] or sha(receipt_path)!=bound['reuse_receipt_sha256']
            or bound['reuse_receipt_row_index']!=reuse['row_index']):
        raise ValueError('Original H10 reuse receipt changed')
    historical=read(receipt_path)['rows'][reuse['row_index']]
    historical_sha=hashlib.sha256(json.dumps(historical,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    actual=historical['actual_performance_report']; command=historical['historical_command_contract']
    if (identity(historical)!=key or historical_sha!=reuse['row_sha256']
            or historical['performance_disposition']!='reuse_existing_observation_unchanged_classification_timed_path'
            or bound['actual_observations']!=1 or bound['new_observations']!=0
            or actual['completed_frames']!=1000 or actual['postprocess_completion_verified'] is not True
            or sha(actual['path'])!=actual['sha256'] or actual['sha256']!=reuse['historical_raw_report_sha256']
            or float(row['fps_makespan'])!=float(actual['fps'])
            or sha(command['path'])!=command['file_sha256']
            or command['contract_sha256']!=row['full_command_contract_sha256']
            or command['contract_sha256']!=quality['full_command_contract_sha256']
            or historical['actual_input']['hash_match'] is not True
            or historical['actual_input']['observed_sha256']!=quality['full_command_contract']['artifacts']['runtime_input_tensor']['sha256']
            or not all(quality.get(k) is True for k in ('central_quality_evidence_verified',
                'precision_quality_binding_verified','task_quality_observation_valid','quality_claim_result_verified'))):
        raise ValueError('Original H10 single-observation/Quality evidence differs')
    for field in ('reference_projection','saved_logit_manifest','normal_saved_semantic_validation'):
        if sha(bound[field])!=bound[field+'_sha256']:
            raise ValueError('Original H10 Quality evidence changed: '+field)
    classification=historical['source_binding']['classification']
    callback=bound['current_callback_reverification']
    prior=historical['source_binding']['performance_and_energy_callback']
    if (classification['byte_identical'] is not True
            or sha(classification['current_path'])!=classification['historical_sha256']
            or callback['path']!=prior['current_path'] or sha(callback['path'])!=callback['sha256']
            or callback['identical_to_exact_original_and_existing_receipt'] is not True
            or callback['classification_early_return_ast_sha256']!=prior['classification_ast_sha256']
            or callback['whole_throughput_branch_ast_sha256']!=prior['throughput_ast_sha256']):
        raise ValueError('Original H10 Classification runtime equivalence changed')
    return {'path':actual['path'],'sha256':actual['sha256'],'fps':actual['fps'],
            'context_only':True,'performance_observation_count':1,'new_performance_observations':0,
            'identity':list(key),'original_reuse_receipt':str(receipt_path),'original_reuse_receipt_sha256':sha(receipt_path)}


def verify_revision3_performance(payload, *, capture_identities=None):
    """Require three verified actual new processes per selected operating point."""
    freeze=read(ROOT/'IMPLEMENTATION_FREEZE.json')
    if freeze.get('validated') is not True: raise ValueError('implementation freeze missing')
    proofs=[]
    for row in payload.get('rows',[]):
        if row.get('reuse_provenance'):
            if row.get('backend')=='native_full_hailo10h':
                if capture_identities is None:
                    raise ValueError('H10 historical baseline cannot be a capture target')
                proofs.append(verify_h10_classification_context_reuse(row,payload,capture_identities))
            else:proofs.append(verify_dx_classification_reuse(row))
            continue
        proof=row.get('fresh_performance_provenance') or {}
        path=Path(str(proof.get('summary') or '')).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():raise ValueError('fresh performance summary outside revision3')
        if sha(path)!=proof.get('summary_sha256'):raise ValueError('fresh performance summary bytes changed')
        summary=read(path)
        if summary.get('status')!='complete' or len(summary.get('rows',[]))!=3:raise ValueError('three valid new repeats required')
        reps=summary['rows'];pids=[r['runner_pid'] for r in reps]
        if len(set(pids))!=3:raise ValueError('independent process evidence required')
        for rep in reps:
            reportpath=Path(rep['report']).resolve()
            if not reportpath.is_relative_to(ROOT):raise ValueError('report outside new root')
            report=read(reportpath)
            counts=[report.get(k) for k in ('completed_work_units','postprocess_completed_frames') if k in report]
            if not counts or any(n!=1000 for n in counts):raise ValueError('1000 actual completed tasks required')
            warmup=report.get('warmup_count',report.get('warmup_frames',report.get('warmup')))
            if warmup!=100:raise ValueError('100 completed warmup contract required')
            if abs(float(report['fps_makespan'])-float(rep['fps']))>1e-8:raise ValueError('fresh rate mismatch')
            proofs.append(dict(path=str(reportpath),sha256=sha(reportpath),fps=rep['fps']))
        if abs(statistics.median(r['fps'] for r in reps)-float(row['fps_makespan']))>1e-8:raise ValueError('summary rate mismatch')
    return proofs


def prepare(args):
    import yaml
    from onnx_splitpoint_tool.energy.config import energy_setup_from_registry
    from onnx_splitpoint_tool.energy.full_system_gain import verify_full_system_current_scale_calibration
    configure(args.run_dir)
    run = Path(args.run_dir).resolve()
    if not run.is_relative_to(ROOT) or run == ROOT:
        raise ValueError('new run must be strictly inside this diagnosis cohort')
    source = Path(args.summary).resolve()
    if not source.is_relative_to(ROOT): raise ValueError('assembled summary must be in new Revision3 root')
    payload = read(source)
    rows = payload.get('rows', [])
    if args.capture_limit < 0 or args.capture_limit > len(rows):
        raise ValueError('capture limit must be within audited context rows')
    capture_identities = [identity(row) for row in rows[:args.capture_limit or len(rows)]]
    context_only=[item['identity'] for item in payload.get('revision3_context_only_h10_reuse',[])]
    if context_only and ({tuple(x)for x in context_only}!={identity(row)for row in rows[len(capture_identities):]}
            or {tuple(x)for x in context_only}&set(capture_identities)):
        raise ValueError('Historical H10 baseline must remain outside the exact capture prefix')
    expected = validate_scope(rows, args.phase, args.setup, context_only_identities=context_only,
                              capture_identities=capture_identities)
    fresh_checks = verify_revision3_performance(payload, capture_identities=capture_identities)
    _, reused_energy_checks = verify_yolov7_energy_context_reuse(rows, capture_identities)
    fresh_checks.extend(reused_energy_checks)
    if args.phase != 'final': raise ValueError('Revision3 only admits distinct final operating points')
    registry_source = Path(args.registry).expanduser().resolve()
    registry = yaml.safe_load(registry_source.read_text())
    targets = targets_from_registry(registry, args.setup)
    checks = {}
    for setup in args.setup:
        view = energy_setup_from_registry(registry, setup, registry_path=registry_source)
        checks[setup] = verify_full_system_current_scale_calibration(view, physical_scope='FS')
        if checks[setup].get('full_system_current_scale_verified') is not True:
            raise ValueError('existing FS calibration not verified: ' + setup)
    collector = registry['energy_defaults']['collector_binary']
    if sha(collector) != registry['energy_defaults']['collector_sha256']:
        raise ValueError('collector bytes differ from registry')
    if run.exists():
        old=read(run/'energy_parent_plan.json')
        budget=read(ROOT/'energy/energy_task_budget.json')
        cohort=getattr(args,'campaign_cohort',None)
        fresh_environment_retry=(getattr(args,'require_environment_observation',False)
            and old.get('require_environment_observation')is True and not old.get('campaign_cohort')
            and old.get('capture_identities')==[list(x)for x in capture_identities]
            and not any(budget['tasks'].get(json.dumps(k,separators=(',',':')),{}).get('chains')for k in old['capture_identities']))
        if not fresh_environment_retry and (not cohort or old.get('campaign_cohort')!=cohort or old['summary']!=str(source)
                or (run/'production-start.json').exists() or any(c.get('cohort_id')==cohort
                    for task in budget['tasks'].values()for c in task['chains'])):
            raise ValueError('Only the same never-started authorized cohort can repeat preparation')
        if fresh_environment_retry and (old['summary']!=str(source) or (run/'production-start.json').exists()):
            raise ValueError('Fresh preparation retry needs unchanged summary path and zero starts')
        import shutil
        previous=run/'prepare_attempts'/uuid.uuid4().hex;previous.mkdir(parents=True,exist_ok=False)
        for name in ('energy_parent_plan.json','PREPARED.json','plancheck_command.json','plancheck.log'):
            if(run/name).is_file():shutil.copy2(run/name,previous/name)
    else:run.mkdir(parents=True,exist_ok=False)
    summary_snapshot = run/'inputs/native_summary.json'
    # The normal planner resolves immutable Quality authority beside the summary.
    # Keep this actual assembled source location; save a separate byte-identical snapshot.
    summary = source
    registry_path = run/'inputs/hardware_snapshot.json'
    write(summary_snapshot, payload)
    authority_files=[]
    candidates=[source.parent.parent,source.parent,*list(source.parents[2:5])]
    authority_root=next((p for p in candidates if (p/'run_manifest.json').is_file() or (p/'reports/native_producer_stage.json').is_file()),None)
    if authority_root:
        if not authority_root.is_relative_to(ROOT):raise ValueError('quality authority must be staged in new root')
        authority_files=[{'path':str(p),'sha256':sha(p)} for p in (authority_root/'run_manifest.json',authority_root/'reports/native_producer_stage.json') if p.is_file()]
    write(registry_path, registry)
    import shutil
    audited_scope=run/'inputs/audited_scope.csv'
    shutil.copyfile(ROOT/'NATIVE_REMEASUREMENT_PLAN.csv',audited_scope)
    validation = ''
    if args.validation_summary:
        validation = str(run/'inputs/validation_summary.json')
        write(validation, read(args.validation_summary))
    window = read(ROOT.parent/'yolov7-full-fix-20261005/provenance/energy_inventory/historical_h8_energy_plan_row.json').get('window_method_ab')
    config = {'schema': 'yolov7-full-fix/bounded-energy-parent-v1', 'prepared_at': now(),
        'run_dir': str(run), 'phase': args.phase, 'expected_identities': expected,
        'allow_diagnostic_capture': args.allow_diagnostic_capture,
        'capture_limit': args.capture_limit, 'capture_identities': capture_identities,
        'campaign_budget_file': str(ROOT/'energy/energy_task_budget.json'),
        'audited_scope': str(audited_scope),
        'audited_scope_sha256': sha(audited_scope),
        'summary_source': str(source), 'summary_source_sha256': sha(source),
        'summary': str(summary), 'summary_sha256': sha(summary),
        'summary_snapshot': str(summary_snapshot), 'summary_snapshot_sha256': sha(summary_snapshot),
        'quality_authority_files': authority_files,
        'fresh_before_repeat_checks': fresh_checks,
        'registry_source': str(registry_source), 'registry_source_sha256': sha(registry_source),
        'registry': str(registry_path), 'registry_sha256': sha(registry_path),
        'validation_summary': validation,
        'validation_summary_sha256': sha(validation) if validation else '',
        'python': sys.executable, 'remote_root': args.remote_root,
        'remote_tool_dir': args.remote_tool_dir, 'targets': targets,
        'remote_environments': {s: next(r for r in registry['hardware_setups'] if r['id']==s)['runtime']['activate'] for s in args.setup},
        'window_method_ab': window, 'fs_calibration_checks': checks,
        'collector_sha256': sha(collector), 'force_rebuild': False,
        'adapter_sha256': sha(ADAPTER_PATH),
        'controller_source_sha256': {name:sha(TOOL/name) for name in (
            'scripts/run_native_producer_energy_from_summary.py',
            'scripts/native_producer_energy_plan.py', 'scripts/energy_measurement_cli.py',
            'onnx_splitpoint_tool/energy/collector.py', 'onnx_splitpoint_tool/energy/task_budget.py',
            'onnx_splitpoint_tool/remote/process_lease.py',
            *('onnx_splitpoint_tool/workflow/'+name for name in (
                'generic_energy_parent.py','native_energy_parent.py','energy_resume.py','energy_parent_flow.py')))},
        'measurement_started': False}
    if getattr(args,'campaign_cohort',None) or getattr(args,'require_environment_observation',False):
        import shlex
        if getattr(args,'campaign_cohort',None):
            config['campaign_cohort']=args.campaign_cohort
            if args.campaign_cohort != run.name:raise ValueError('final cohort must bind this parent generation')
        else: config['require_environment_observation']=True
        config['completed_row_validator']=shlex.join([sys.executable,'-B',str(ROOT/'quality_transfer/import_qualified_energy.py'),
            '--cohort',str(run),'--allow-partial','--require-complete-point'] + (['--final-cohort',args.campaign_cohort] if getattr(args,'campaign_cohort',None) else []))
        config['controller_source_sha256']['onnx_splitpoint_tool/energy/environment_observer.py']=sha(TOOL/'onnx_splitpoint_tool/energy/environment_observer.py')
        config['controller_source_sha256']['onnx_splitpoint_tool/native_energy_reporting.py']=sha(TOOL/'onnx_splitpoint_tool/native_energy_reporting.py')
        config['completed_row_validator_sha256']=sha(ROOT/'quality_transfer/import_qualified_energy.py')
    plancheck=run/'plancheck'
    if (plancheck/'native_producer_energy_results.json').exists():
        plancheck=run/'prepare_attempts'/uuid.uuid4().hex/'plancheck'
    config['plancheck_root']=str(plancheck)
    write(run/'energy_parent_plan.json', config)
    command = command_for(config, plancheck, dry_run=True)
    write(run/'plancheck_command.json', {'argv': command, 'hardware_execution': False})
    with (run/'plancheck.log').open('w') as log:
        cp = subprocess.run(command, cwd=TOOL, stdout=log, stderr=subprocess.STDOUT,
                            timeout=180, env={**os.environ, 'PYTHONDONTWRITEBYTECODE':'1'})
    result = read(plancheck/'native_producer_energy_results.json')
    planned = [identity(x['row']) for x in result.get('rows', [])]
    valid = cp.returncode == 0 and result.get('status') == 'dry_run' and sorted(planned) == expected
    admitted = [item for item in result.get('rows', []) if item.get('dry_run') is True]
    limited = [item for item in result.get('rows', []) if item.get('skipped') == 'limit']
    valid = valid and [identity(x['row']) for x in admitted] == capture_identities
    valid = valid and len(admitted) + len(limited) == len(expected)
    # Context-only rows retain all normal pairing/Quality work but cannot start.
    for item in admitted:
        row = item['row']
        if _p03_resume_admission(row) is None:
            valid = False
    policy = capture_policy_preflight(config, result)
    valid = valid and policy['ok']
    write(run/'PREPARED.json', {'prepared_at': now(), 'ok': valid, 'plancheck_returncode': cp.returncode,
        'capture_policy_preflight':policy,
        'plancheck_status': result.get('status'), 'expected_count': len(expected), 'planned_count': len(planned),
        'capture_admission_scope':'existing_native_planner_runtime_capture_predicate',
        'diagnostic_capture_count':sum(x['row'].get('diagnostic_only') is True for x in admitted),
        'qualified_claim_capture_count':sum(x['row'].get('diagnostic_only') is False
            and x['row'].get('energy_quality_qualified') is True and x['row'].get('claim_eligible') is True for x in admitted),
        'collector_starts': 0, 'workload_starts': 0, 'plan_sha256': sha(run/'energy_parent_plan.json')})
    if not valid:
        raise RuntimeError('normal energy plan/policy rejected before capture: '+json.dumps({'errors':policy['errors'],'rows':policy['rows']}))
    print(json.dumps({'status':'PREPARED_NOT_STARTED','run_dir':str(run),'rows':len(expected)}))


def refresh_controller(args):
    """Existing controller source binding, refreshed without changing inputs/results."""
    configure(args.run_dir)
    run=Path(args.run_dir).resolve();config=read(run/'energy_parent_plan.json')
    budget=read(config['campaign_budget_file']);grant=final_cohort_authorization(budget,config.get('campaign_cohort'))
    fresh_environment=config.get('require_environment_observation')is True and not config.get('campaign_cohort')
    if (not fresh_environment and (not config.get('campaign_cohort')or grant.get('cohort_id')!=config['campaign_cohort']
            or run!=Path(grant.get('parent_run_dir','')))):
        raise ValueError('controller refresh only for the exact cleanly closed authorized final cohort')
    if(run/'production-start.json').is_file():
        if not args.previous_attempt or not Path(args.previous_attempt).resolve().is_relative_to(run):
            raise ValueError('controller refresh requires the actual clean terminal predecessor')
        if read(Path(args.previous_attempt)/'cleanup-status.json').get('cleanup_proven')is not True:
            from .energy_resume import released_predecessor
            released_predecessor(run,Path(args.previous_attempt))
    else:
        if any(((config.get('campaign_cohort')and chain.get('cohort_id')==config['campaign_cohort'])
                or Path(chain.get('run_directory','/')).is_relative_to(run))
               for task in budget['tasks'].values()for chain in task['chains']):
            raise ValueError('unstarted controller refresh requires zero existing parent/cohort chains')
    for field in ('summary','registry','validation_summary','audited_scope','registry_source'):
        if config.get(field)and sha(config[field])!=config[field+'_sha256']:raise ValueError('original input changed: '+field)
    from onnx_splitpoint_tool.energy.task_budget import campaign_budget_stop_reason
    if fresh_environment:
        from onnx_splitpoint_tool.energy.task_budget import _derived_unstarted_capture, _source_reentry
        for identity in config['capture_identities']:
            source=config['targets'][identity[3]]['energy']['urecs_address']
            row_id=json.dumps(identity,separators=(',',':'))
            if budget['sources'][source].get('continuous_admission')is not None:
                continuous=_source_reentry(budget,source,{})
                if continuous.get('continuous_campaign')is not True or row_id not in continuous['row_ids']:
                    raise ValueError('fresh controller supplement row outside authorized continuous campaign')
            elif _derived_unstarted_capture(budget,source,row_id,parent_run_dir=run)is None:
                raise ValueError('fresh controller supplement requires the exact authorized recovered batch scope')
    for identity in config['capture_identities']:
        source=config['targets'][identity[3]]['energy']['urecs_address']
        if campaign_budget_stop_reason(budget,json.dumps(identity,separators=(',',':')),
                source_id=source,parent_run_dir=run):raise ValueError('controller refresh cannot lift an effective STOP')
    path=run/'controller-source-supplement.json'
    if path.exists():
        import shutil
        history=run/'resume_attempts'/('controller-binding-'+uuid.uuid4().hex);history.mkdir(parents=True,exist_ok=False)
        shutil.copy2(path,history/path.name)
    sources={k:sha(TOOL/k)for k in config['controller_source_sha256']}
    sources['onnx_splitpoint_tool/energy/recovery_accounting.py']=sha(TOOL/'onnx_splitpoint_tool/energy/recovery_accounting.py')
    for name in ('generic_energy_parent.py','native_energy_parent.py','energy_resume.py','energy_parent_flow.py'):
        relative='onnx_splitpoint_tool/workflow/'+name
        sources[relative]=sha(TOOL/relative)
    write(path,{'at':now(),'campaign_cohort':config.get('campaign_cohort'),
        'environment_observation_run_id':run.name if fresh_environment else None,'parent_config_sha256':sha(run/'energy_parent_plan.json'),
        'controller_source_sha256':sources,
        'adapter_sha256':sha(ADAPTER_PATH),'completed_row_validator_sha256':sha(ROOT/'quality_transfer/import_qualified_energy.py'),
        'original_inputs_parent_configuration_results_budgets_unchanged':True,'DUT_runtime_changed':False})


def run_prepared(args):
    from onnx_splitpoint_tool.native_progress import run_streaming
    from onnx_splitpoint_tool.process_control import ProcessTreeRegistry
    from onnx_splitpoint_tool.remote.process_lease import ControllerResourceBroker, RemoteProcessLeaseRegistry, RemoteProcessLeaseScope
    from onnx_splitpoint_tool.quality_service import ResourcePauseGate
    from onnx_splitpoint_tool.workflow.run_control import EvaluationRunLock
    from onnx_splitpoint_tool.workflow.execution_binding import physical_dut_key
    from onnx_splitpoint_tool.workflow.generic_completed_task import observe_owners
    configure(args.run_dir)
    run = Path(args.run_dir).resolve()
    config = read(run/'energy_parent_plan.json')
    binding=config
    supplement=run/'controller-source-supplement.json'
    if (config.get('campaign_cohort')or config.get('require_environment_observation'))and supplement.is_file():
        binding=read(supplement)
        if (binding.get('campaign_cohort')!=config.get('campaign_cohort')
                or(config.get('require_environment_observation')and binding.get('environment_observation_run_id')!=run.name)
                or binding.get('parent_config_sha256')!=sha(run/'energy_parent_plan.json')):
            raise ValueError('controller supplement differs from immutable prepared parent')
    from .energy_parent_flow import verify_product_sources
    verify_product_sources(TOOL,binding['controller_source_sha256'])
    prepared = read(run/'PREPARED.json')
    if prepared.get('ok') is not True or prepared['plan_sha256'] != sha(run/'energy_parent_plan.json'):
        raise ValueError('prepared plan missing or changed')
    require_capture_policy(config)
    if config.get('campaign_cohort') or config.get('require_environment_observation'):
        if sha(ROOT/'quality_transfer/import_qualified_energy.py') != binding['completed_row_validator_sha256']:
            raise ValueError('normal completed-point importer changed since prepare')
    if config.get('campaign_cohort'):
        grant=final_cohort_authorization(read(config['campaign_budget_file']),config['campaign_cohort'])
        if grant.get('cohort_id')!=config['campaign_cohort'] or sorted(grant.get('row_ids',[]))!=sorted(
                json.dumps(k,separators=(',',':'))for k in config['capture_identities']):
            raise ValueError('exact final cohort authorization missing')
    for field in ('summary','registry','validation_summary'):
        if config.get(field) and sha(config[field]) != config[field+'_sha256']:
            raise ValueError('frozen input changed: ' + field)
    if sha(config['audited_scope'])!=config['audited_scope_sha256']:raise ValueError('audited case scope changed')
    for bound in config.get('quality_authority_files',[]):
        if sha(bound['path'])!=bound['sha256']:raise ValueError('frozen quality authority changed')
    if sha(config['registry_source']) != config['registry_source_sha256']:
        raise ValueError('live registry changed since preparation')
    if sha(ADAPTER_PATH) != binding['adapter_sha256']:
        raise ValueError('external adapter changed since preparation')
    for name, expected in binding['controller_source_sha256'].items():
        if sha(TOOL/name) != expected:
            raise ValueError('controller source changed since preparation: ' + name)
    from onnx_splitpoint_tool.energy.task_budget import campaign_budget_stop_reason
    live_budget=read(config['campaign_budget_file'])
    if any(campaign_budget_stop_reason(live_budget,json.dumps(k,separators=(',',':')),
            source_id=config['targets'][k[3]]['energy']['urecs_address'],parent_run_dir=run)
            for k in config['capture_identities']):
        raise ValueError('existing effective task/source STOP blocks normal parent admission')
    for check in config.get('fresh_before_repeat_checks', []):
        if sha(check['path']) != check['sha256']:
            raise ValueError('fresh performance report changed since preparation')
    if run != Path(config['run_dir']) or not run.is_relative_to(ROOT):
        raise ValueError('run path changed')
    if not args.slot_handoff:
        raise ValueError('explicit coordinator slot handoff required')
    final_resume=getattr(args,'final_cohort_resume',False)
    if (run/'production-start.json').exists() and not final_resume:
        raise FileExistsError('attempt already started; no implicit rerun or budget reset')
    attempt=run; recovered_prior=False
    if final_resume:
        if not (config.get('campaign_cohort') or config.get('require_environment_observation')):
            raise ValueError('final cohort resume requires its original qualified scope')
        if bool(getattr(args,'failed_attempt','')) and bool(getattr(args,'previous_attempt','')):
            raise ValueError('select exactly one final cohort predecessor')
        if not(run/'production-start.json').is_file():
            raise ValueError('final cohort has never started; use its normal run action')
        selected=getattr(args,'failed_attempt','') or args.previous_attempt
        if not selected:raise ValueError('final cohort resume requires an explicit actual predecessor')
        prior=Path(selected).resolve()
        if not prior.is_relative_to(run):
            raise ValueError('final cohort resume needs its actual predecessor')
        if read(prior/'cleanup-status.json').get('cleanup_proven')is not True:
            from .energy_resume import released_predecessor
            released_predecessor(run,prior); recovered_prior=True
        budget=read(config['campaign_budget_file'])
        source_ids={row['energy']['urecs_address']for row in read(config['registry'])['hardware_setups']if row['id']in config['targets']}
        from onnx_splitpoint_tool.energy.task_budget import campaign_budget_stop_reason
        if any(campaign_budget_stop_reason(budget,json.dumps(k,separators=(',',':')),
                source_id=config['targets'][k[3]]['energy']['urecs_address'],parent_run_dir=run)for k in config['capture_identities']):
            raise ValueError('existing task/source STOP blocks final cohort continuation')
        started=read(prior/'production-start.json')
        prior_results=_final_resume_previous_results(run,prior,started)
        pending=True
        if prior_results.is_file()and any((x.get('run')or{}).get('energy_aggregate_complete')is True
                                         for x in read(prior_results).get('rows',[])):
            import shlex
            from .energy_parent_import import product_import_command
            checked=subprocess.run(product_import_command(config['completed_row_validator'])+['--source',str(prior_results)],
                cwd=TOOL,capture_output=True,text=True,timeout=180)
            if checked.returncode:raise ValueError('Previously completed final points still fail normal import; no resume capture: '+checked.stderr[-2000:])
            imported=json.loads(checked.stdout)
            if len(imported.get('results',[]))==len(config['capture_identities']):
                # The existing normal importer can close an all-complete
                # durable partial without reacquiring or relabeling any data.
                write(run/'FINAL_COHORT_RESULT.json',{'at':now(),'status':'ALL_SELECTED_FINAL_POINTS_NORMAL_IMPORTED',
                    'source':str(prior_results),'import':imported,'new_physical_starts':0})
                return
        attempt=run/'resume_attempts'/uuid.uuid4().hex
        attempt.mkdir(parents=True,exist_ok=False)
    session = 'energy-' + uuid.uuid4().hex
    owner = {'run_id':run.name, 'session_id':session, 'purpose':'yolov7_full_fix_energy'}
    lock = EvaluationRunLock(out_root=run.parent, run_dir=run, owner=owner)
    if recovered_prior:
        from .energy_resume import recover_run_fence
        lock.acquire(recover_quarantine=lambda fence:recover_run_fence(run,attempt,session,lock,fence,prior))
    else:lock.acquire()
    cancel, stop = threading.Event(), threading.Event()
    registry, processes = RemoteProcessLeaseRegistry(), ProcessTreeRegistry()
    broker = thread = None
    handlers = {}
    errors = []
    primary = None
    try:
        observe_owners(config['targets'], attempt/'owner_preflight')
        journal = registry.configure_journal(scope=RemoteProcessLeaseScope(run.name,session), journal_dir=attempt/'remote-process-leases')
        raw_registry = read(config['registry'])
        resources = {}
        for setup, target in config['targets'].items():
            row = next(r for r in raw_registry['hardware_setups'] if r['id']==setup)
            resources[setup] = [physical_dut_key(target), 'source:'+row['energy']['urecs_address'],
                'controller:cpu','controller:io','controller:nic','controller:capture']
        broker = ControllerResourceBroker(journal, ResourcePauseGate(), run_dir=run, setups=resources)
        env = {**os.environ, **registry.journal_environment(),
               'ONNX_SPLITPOINT_CONTROLLER_RESOURCES':'required', 'PYTHONDONTWRITEBYTECODE':'1'}
        def pump():
            while not stop.wait(.025):
                try:
                    broker.tick()
                    # A STOP blocks further starts inside the normal broker.
                    # Let the current collector finish durable failure accounting.
                    # Final cleanup still retains STOP and quarantines unproven ownership.
                except BaseException as exc:
                    errors.append(str(exc)); cancel.set(); return
        thread = threading.Thread(target=pump, name='bounded-energy-controller', daemon=True)
        thread.start()
        handlers = {s:signal.signal(s,lambda *_:cancel.set()) for s in (signal.SIGINT,signal.SIGTERM)}
        result_output=run/'measurement'if not final_resume else run/'measurement/resume_attempts'/attempt.name
        command = command_for(config, result_output)
        if final_resume:
            command+=['--previous-results',str(prior_results)]
        write(attempt/'production-start.json', {'started_at':now(),'slot_handoff':args.slot_handoff,
            'run_id':run.name,'session_id':session,'parent_pid':os.getpid(),'argv':command,
            'no_deployment_or_build':True})
        with (attempt/'production.log').open('w') as log:
            def line(text):
                log.write(text+'\n');log.flush();print(text,flush=True)
            result = run_streaming(command,cwd=TOOL,env=env,timeout=21600,
                label='yolov7-bounded-energy',line_callback=line,cancel_event=cancel,
                should_cancel=lambda:cancel.is_set() or bool(errors) or registry.cancelled,
                process_registry=processes,before_terminate=lambda:registry.cancel_all(grace_s=3))
        write(attempt/'production-result.json', {'finished_at':now(),'returncode':result.returncode,
            'stdout':result.stdout,'stderr':result.stderr})
        _finish_dispatcher_result(run, config, result_output, result.returncode,
            cancel.is_set(), final_resume)
    except BaseException as exc:
        primary = exc
        write(attempt/'failure.json', {'at':now(),'error':str(exc),'automatic_restart':False})
        raise
    finally:
        try:
            if cancel.is_set() or registry.active_count():
                registry.cancel_all(grace_s=3)
            if processes.remaining_owned_count():
                processes.terminate_all(grace_s=3)
        except BaseException as exc:
            errors.append('owned cleanup: '+str(exc))
        stop.set()
        if thread is not None:
            thread.join(timeout=5)
        try:
            if broker is not None and (thread is None or not thread.is_alive()):
                broker.close()
        except BaseException as exc:
            errors.append('broker cleanup: '+str(exc))
        pending = registry.active_count()+processes.remaining_owned_count()
        if broker is not None and broker.stopped_sources:
            errors.append('stopped physical resources: '+repr(sorted(broker.stopped_sources)))
        unproven = bool(pending or errors or (thread is not None and thread.is_alive()))
        if unproven:
            lock.commit_quarantine_fence({'reason':'yolov7_energy_cleanup_unproven','pending':pending,'errors':errors})
        for signum, handler in handlers.items():
            signal.signal(signum,handler)
        write(attempt/'cleanup-status.json', {'at':now(),'cleanup_proven':not unproven,
            'owned_remote_active':registry.active_count(),'owned_local_active':processes.remaining_owned_count(),
            'errors':errors,'foreign_processes_stopped':False})
        lock.release()
        if unproven and primary is None:
            raise RuntimeError('normal ownership cleanup unproven; no automatic restart')


def main(argv=None, *, adapter_path=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action',required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--summary',required=True)
    p.add_argument('--run-dir',required=True)
    p.add_argument('--phase',choices=['final'],required=True)
    p.add_argument('--allow-diagnostic-capture',action='store_true',
                   help='Explicitly permit technically admitted diagnostic captures; never grants scientific claims.')
    p.add_argument('--campaign-cohort',default=None)
    p.add_argument('--require-environment-observation',action='store_true')
    p.add_argument('--setup',action='append',required=True,choices=sorted(SETUPS))
    p.add_argument('--registry',default=str(Path.home()/'.onnx_splitpoint_tool/hardware_setups.yaml'))
    p.add_argument('--validation-summary',default='')
    p.add_argument('--capture-limit',type=int,default=0)
    p.add_argument('--remote-root',required=True)
    p.add_argument('--remote-tool-dir',default='/home/nx/ONNX-Splitpoint-Tool')
    p = sub.add_parser('run')
    p.add_argument('--run-dir',required=True)
    p.add_argument('--slot-handoff',required=True,help='Coordinator handoff identity, recorded verbatim.')
    p = sub.add_parser('recover')
    p.add_argument('--run-dir', required=True)
    p.add_argument('--authorization', required=True)
    p.add_argument('--failed-attempt', required=True)
    p = sub.add_parser('reconcile')
    p.add_argument('--run-dir', required=True)
    p.add_argument('--failed-attempt', required=True)
    p = sub.add_parser('resume')
    p.add_argument('--run-dir', required=True)
    p.add_argument('--slot-handoff', required=True)
    p.add_argument('--failed-attempt', default='', help='Exact stopped parent attempt whose recovery is handed off.')
    p.add_argument('--previous-attempt', default='', help='Clean terminal parent to continue without reacquiring valid repeats.')
    p = sub.add_parser('refresh-controller')
    p.add_argument('--run-dir',required=True);p.add_argument('--previous-attempt',default='')
    args = parser.parse_args(argv)
    if args.action in ('recover','reconcile') and (Path(args.run_dir)/'parent_plan.json').is_file():
        context=configure(args.run_dir)
        if adapter_path is not None and Path(adapter_path).resolve()!=context['report_root']/'operations/energy_parent_qualified.py':
            raise ValueError('Generic recovery requires the exact historical normal recovery entry')
    else:configure(args.run_dir,adapter_path=adapter_path)
    if args.action=='refresh-controller':refresh_controller(args);return
    if args.action in ('recover', 'resume', 'reconcile'):
        if args.action == 'resume':
            require_capture_policy(read(Path(args.run_dir)/'energy_parent_plan.json'))
        from . import energy_resume
        if args.action in ('recover', 'reconcile'):
            getattr(energy_resume,args.action)(args)
        else:
            if any(read(Path(args.run_dir)/'energy_parent_plan.json').get(k)for k in ('campaign_cohort','require_environment_observation')):
                args.final_cohort_resume=True
                run_prepared(args)
            else:energy_resume.resume(args, sys.modules[__name__])
    else:
        (prepare if args.action=='prepare' else run_prepared)(args)


if __name__=='__main__':
    main()
