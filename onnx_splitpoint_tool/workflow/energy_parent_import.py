"""Import saved Native/Generic energy without private Python entry points.

This is local result processing only. It cannot launch a collector or model.
Legacy saved commands keep their controls and resolve to this product module.
"""
from __future__ import annotations
import argparse
from contextlib import redirect_stdout
import hashlib
import io
import json
import math
from pathlib import Path
from . import generic_energy_parent as parent
from .energy_current_projection import read, sha, write
from ..native_energy_reporting import collect_native_energy
from ..energy.task_budget import read_verified_campaign_repeat


def product_import_command(command):
    """Resolve preserved legacy import entries without changing saved plans."""
    import shlex
    command = shlex.split(command) if isinstance(command, str) else list(command)
    scripts = [(index, Path(arg)) for index, arg in enumerate(command)
               if isinstance(arg, str) and arg.endswith('.py')]
    if len(scripts) == 1 and scripts[0][1].name in {
        'import_generic_energy.py', 'import_qualified_energy.py',
    }:
        index, script = scripts[0]
        if index < 1 or any(arg not in {'-B', '-u'} for arg in command[1:index]):
            raise ValueError('unsupported saved energy import Python prefix')
        # A saved working directory may contain an older checkout. Isolated
        # Python still loads the ordinary installed distribution and its .pth
        # metadata, while excluding cwd/PYTHONPATH/user-site shadowing.
        flags = command[1:index]
        if '-B' not in flags:
            flags = ['-B'] + flags
        return [command[0], '-I'] + flags + ['-m', 'onnx_splitpoint_tool.workflow.energy_parent_import'] + command[index + 1:]
    return command


def key(r):return tuple(r[k] for k in ['backend','model','case','setup_id'])


def import_cohort(cohort,allow_partial=False,source_override=None):
 from .energy_parent_flow import parent_context
 cohort=Path(cohort).resolve(strict=True);R=parent_context(cohort)['report_root']
 assert cohort.parent==R/'energy'
 configpath=cohort/'energy_parent_plan.json';config=read(configpath);source=cohort/'measurement/native_producer_energy_results.json';partial=False
 if not source.is_file() and allow_partial:source=cohort/'measurement/native_producer_energy_results.partial.json';partial=True
 if source_override:
  source=Path(source_override).resolve();assert source.is_relative_to(cohort/'measurement') and source.name in {'native_producer_energy_results.json','native_producer_energy_results.partial.json'}
  partial=source.name.endswith('.partial.json');assert not partial or allow_partial
 if not source.is_file():return {'status':'NO_PUBLISHED_NORMAL_RESULT_YET','cohort':str(cohort)}
 prepared=read(cohort/'PREPARED.json');assert prepared.get('ok')is True and prepared['plan_sha256']==sha(configpath)
 for field in ('registry','audited_scope'):
  assert sha(config[field])==config[field+'_sha256']
 content=source.read_bytes();digest=hashlib.sha256(content).hexdigest();actual=json.loads(content);validation=Path(config['validation_summary']);assert sha(validation)==config['validation_summary_sha256'];summary=Path(config['summary']);assert sha(summary)==config['summary_sha256']
 for bound in config.get('quality_authority_files',[]):assert sha(bound['path'])==bound['sha256']
 budgetpath=R/'energy/energy_task_budget.json';budgetbytes=budgetpath.read_bytes();budget=json.loads(budgetbytes);budgetsha=hashlib.sha256(budgetbytes).hexdigest();proofs=[]
 for item in actual.get('rows',[]):
  row=item['row'];run=item.get('run') or {}
  if item.get('ok') is not True or not run.get('energy_aggregate_complete'):continue
  assert run['energy_aggregate_verified'] is True and type(row.get('diagnostic_only'))is bool
  assert row.get('measurement_requested_repeats')==row.get('measurement_effective_repeats')==3
  assert list(key(row))in config['capture_identities']
  assert (row.get('campaign_cohort')or'')==(config.get('campaign_cohort')or'')
  if config.get('campaign_cohort'):assert row['diagnostic_only']is False and row['energy_quality_qualified']is True
  aggregatepath=Path(run['energy_aggregate_path']);assert sha(aggregatepath)==run['energy_aggregate_sha256'];aggregate=read(aggregatepath);assert aggregate==run['energy_aggregate'] and len(aggregate['runs'])==3 and aggregate.get('repeat_contract_complete')is True
  task=budget['tasks'][json.dumps(list(key(row)),separators=(',',':'))];assert task['limits']=={'max_chains':9,'max_retries':2,'max_transport_failures':20}
  observed={str(Path(x['storage_dir']).parent) for x in aggregate['runs'] if x.get('storage_dir')};chains=sorted([c for c in task['chains'] if c.get('valid') and c['run_directory'] in observed and (c.get('cohort_id')or'')==(config.get('campaign_cohort')or'')],key=lambda c:c['logical_repeat']);assert len(chains)==3 and {c['logical_repeat']for c in chains}=={'repeat:0','repeat:1','repeat:2'} and all(c.get('finished')is True and c.get('source_completion_verified')is True for c in chains)
  repeatproof=[]
  for chain in chains:
   result=read_verified_campaign_repeat(chain,command=Path(row['command_file']).read_text().strip(),command_contract_sha256=row['successful_command_contract_sha256'],duration_s=60,require_ab=True);n=result['energy_work_units_used'];energy=result['energy_total_j'];assert type(n)is int and n>0 and energy>0 and math.isfinite(energy) and result['runtime_work_unit_gate_status']=='pass' and result['full_system_current_scale_verified'] is True;assert math.isclose(energy/n,result['energy_per_work_unit_j'],rel_tol=1e-12)
   assert result['energy_configured_workload_duration_s']==60 and result['scientific_primary_active_duration_s']>=60 and result['runtime_work_unit_evidence']['exact']is True and result['runtime_work_unit_evidence']['count']==n
   if config.get('require_environment_observation'):assert result.get('environment_observation_required')is True and result.get('campaign_run_id')==cohort.name
   repeatpath=Path(chain['run_directory'])/'energy_summary.json';repeatproof.append({'path':str(repeatpath),'sha256':sha(repeatpath),'logical_repeat':chain['logical_repeat'],'completed_tasks':n,'energy_j':energy,'joules_per_task':energy/n,'source_completion_verified':chain['source_completion_verified']})
  proofs.append({'identity':key(row),'aggregate':str(aggregatepath),'aggregate_sha256':sha(aggregatepath),'native_command_contract_sha256':row['successful_command_contract_sha256'],'repeat_proofs':repeatproof})
 if not proofs:return {'status':'NO_COMPLETE_ACQUISITION_YET','source':str(source),'source_sha256':digest}
 out=R/'quality_transfer/qualified_energy_import'/cohort.name/digest;view=out/'normal_view';resultname='native_producer_energy_results.partial.json' if partial else 'native_producer_energy_results.json';destination=view/'reports/native_energy_measurements'/resultname
 if (out/'RECEIPT.json').is_file():
  saved=read(out/'RECEIPT.json');assert sha(destination)==saved['source_sha256']==digest
  for field,path in [('parent_config',configpath),('validation',validation),('summary',summary)]:assert saved[field]==str(path) and saved[field+'_sha256']==sha(path)
  return {'status':'NORMAL_SAVED_ENERGY_IMPORTED','snapshot':str(out),'results':saved['normal_results'],'existing_import_reused':True}
 destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(content);vp=view/'reports/native_validation/native_producer_validation_summary.json';vp.parent.mkdir(parents=True,exist_ok=True);vp.write_bytes(validation.read_bytes());(out/'campaign_budget_snapshot.json').write_bytes(budgetbytes)
 normalized=collect_native_energy(view,include_partial=partial);receipts={'source':str(source),'source_sha256':digest,'source_is_active_partial_publication':partial,'parent_config':str(configpath),'parent_config_sha256':sha(configpath),'validation':str(validation),'validation_sha256':sha(validation),'summary':str(summary),'summary_sha256':sha(summary),'budget_snapshot_sha256':budgetsha,'actual_source_bytes_and_all_row_plan_run_payloads_unchanged':True,'normal_consumer':'native_energy_reporting.collect_native_energy','proofs':proofs,'hardware_started':False}
 write(out/'normal_energy_rows.json',{'rows':normalized,'provenance':receipts});wanted={tuple(p['identity']) for p in proofs};accepted=[r for r in normalized if key(r) in wanted];assert len(accepted)==len(proofs)
 fields=['backend','model','case','setup_id','energy_quality_qualified','completion_pairing_eligible','completion_pairing_status','quality_claim_result_verified','claim_eligible','claim_exclusion_reasons','diagnostic_only','energy_per_work_j','energy_per_work_j_ci_low','energy_per_work_j_ci_high','energy_repeat_valid_n','energy_repeat_requested_n','campaign_cohort','environment_observation_status']
 assert all(r.get('ok')is True and r.get('energy_calibration_verified')is True and r.get('energy_repeat_valid_n')==r.get('energy_repeat_requested_n')==3 for r in accepted)
 if config.get('campaign_cohort'):assert all(r.get('campaign_cohort')==config['campaign_cohort'] and r.get('environment_observation_status')=='pass' for r in accepted)
 if config.get('require_environment_observation'):
  assert all(r.get('environment_observation_status')=='pass' and r.get('diagnostic_only')is False for r in accepted)
  receipts['environment_observation_run_id']=cohort.name
 receipts['normal_results']=[{k:r.get(k) for k in fields} for r in accepted];write(out/'RECEIPT.json',receipts);write(out.parent/'LATEST.json',{'snapshot':str(out),'receipt':str(out/'RECEIPT.json'),'receipt_sha256':sha(out/'RECEIPT.json'),'rows':len(accepted),'claimed':sum(r.get('claim_eligible') is True for r in accepted),'source_sha256':digest});return {'status':'NORMAL_SAVED_ENERGY_IMPORTED','snapshot':str(out),'results':receipts['normal_results']}


def resolve_result(run):
    run = Path(run).resolve(strict=True)
    R = parent.configure(run)['report_root']
    if not run.is_relative_to(R / 'energy/generic_completed') or run == R / 'energy/generic_completed':
        raise ValueError('Generic import requires this campaign existing Parent')
    attempt = parent.latest_clean_attempt(run).resolve(strict=True)
    if attempt != run and attempt.parent != run / 'resume_attempts':
        raise ValueError('Generic latest attempt escapes its original Parent')
    source = attempt / 'normal_result.json'
    pointer = run / 'LATEST_ATTEMPT.json'
    if attempt != run:
        saved = read(pointer)
        expected = {'attempt_dir': attempt, 'normal_result': source,
                    'normal_plan': attempt / 'measurement/generic_energy_plan.json',
                    'cleanup': attempt / 'cleanup-status.json'}
        if any(Path(saved.get(key, '')).resolve() != path for key, path in expected.items()):
            raise ValueError('Generic latest result pointer differs from actual clean terminal attempt')
    if not source.is_file():
        raise ValueError('Generic latest clean attempt has no completed normal result')
    return run, attempt, source


def selected_row(payload, case, exporter):
    identity = case['identity']
    rows = [row for row in payload['rows']
            if all(row.get(key) == identity[key] for key in ('model_id', 'case_id', 'setup_id', 'precision'))
            and exporter.canonical_direction(row.get('direction', '')) == exporter.canonical_direction(identity['direction'])
            and (not case.get('case_key') or row.get('case_key') == case['case_key'])]
    if len(rows) != 1:
        raise ValueError('Generic original case has no unique selected Current row')
    return rows[0]


def import_parent(run):
    run, attempt, source = resolve_result(run)
    R = parent.configure(run)['report_root']
    p = read(run / 'parent_plan.json')
    case = read(parent.bound_file(p['case']))
    from . import energy_current_projection as exporter
    current_path = R / 'current_results/generic_current.json'
    current_digest = sha(current_path)
    source_digest = sha(source)
    with redirect_stdout(io.StringIO()):
        projection = exporter.project_current_energy(case, source, current_path=current_path, expected_current_sha256=current_digest)
    admitted = selected_row(read(current_path), case, exporter)
    if sha(source) != source_digest or any(admitted.get(key) != value for key, value in projection.items()):
        raise ValueError('Generic Current did not retain the exact normal energy projection')
    identity = case['identity']
    result = {'backend': identity.get('comparison_backend', identity['direction']),
              'model': identity['model_id'], 'case': identity['case_id'], 'setup_id': identity['setup_id'],
              'precision': identity['precision'], 'direction': identity['direction'], 'regime': 'generic_completed',
              'normal_energy_eligible': admitted['normal_energy_eligible'],
              'claim_eligible': admitted['energy_efficiency_claim_eligible'],
              'energy_per_work_j': admitted['energy_per_task_j'], 'tasks_per_j': admitted['tasks_per_j'],
              'energy_per_work_j_ci': admitted['energy_per_task_j_ci'],
              'energy_repeat_valid_n': len(admitted['energy_repeats']), 'energy_repeat_requested_n': p['capture_count'],
              'energy_status': admitted['energy_status'], 'energy_provenance': admitted['energy_provenance']}
    receipt = {'status': 'NORMAL_SAVED_GENERIC_ENERGY_IMPORTED', 'source': str(source),
               'source_sha256': source_digest, 'parent': str(run), 'actual_attempt': str(attempt),
               'parent_config': str(run / 'parent_plan.json'), 'parent_config_sha256': sha(run / 'parent_plan.json'),
               'normal_consumer': 'generic_energy_reporting.collect_generic_energy_case',
               'current_consumer': 'workflow.energy_current_projection.generic_energy_projection',
               'current_projection': 'workflow.energy_current_projection.project_current_energy',
               'previous_current_sha256': current_digest,
               'normal_results': [result], 'hardware_started': False,
               'current': str(current_path), 'current_sha256': sha(current_path),
               'comparison_matrix_refresh': 'separate_report_operation'}
    digest = hashlib.sha256(json.dumps(receipt, sort_keys=True).encode()).hexdigest()
    output = R / 'quality_transfer/qualified_energy_import' / run.name / digest
    write(output / 'RECEIPT.json', receipt)
    write(output.parent / 'LATEST.json', {'snapshot': str(output), 'receipt': str(output / 'RECEIPT.json'),
          'receipt_sha256': sha(output / 'RECEIPT.json'), 'rows': 1,
          'claimed': int(result['claim_eligible'] is True), 'source_sha256': source_digest})
    return {'status': receipt['status'], 'snapshot': str(output), 'receipt': str(output / 'RECEIPT.json'),
            'results': [result]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--parent', type=Path)
    selection.add_argument('--cohort', type=Path)
    parser.add_argument('--allow-partial', action='store_true')
    parser.add_argument('--source', type=Path)
    parser.add_argument('--require-complete-point', action='store_true')
    parser.add_argument('--final-cohort', default='')
    args = parser.parse_args(argv)
    if args.parent:
        if args.allow_partial or args.source or args.final_cohort:
            parser.error('Native result controls require --cohort')
        result = import_parent(args.parent)
    else:
        if args.final_cohort:
            assert read(args.cohort/'energy_parent_plan.json').get('campaign_cohort') == args.final_cohort
        result = import_cohort(args.cohort, args.allow_partial, args.source)
    if args.require_complete_point:
        assert result['status'] in {'NORMAL_SAVED_ENERGY_IMPORTED', 'NORMAL_SAVED_GENERIC_ENERGY_IMPORTED'} and result['results']
    if args.final_cohort and result.get('results'):
        assert all(r['diagnostic_only'] is False and r['energy_quality_qualified'] is True
                   and r.get('environment_observation_status') == 'pass'
                   and r.get('campaign_cohort') == args.final_cohort for r in result['results'])
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
