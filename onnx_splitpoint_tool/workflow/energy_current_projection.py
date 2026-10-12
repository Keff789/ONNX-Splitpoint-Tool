"""Reprocess saved energy using packaged normal result consumers.

Source, budget, comparison and controller-refresh bindings are retained.
The report root comes from the saved parent; no private Python is loaded.
"""
from __future__ import annotations
import copy
import csv
import datetime
import hashlib
import io
import json
from pathlib import Path
from ..ranking_methods import canonical_direction
from .checkpoints import atomic_write_json, atomic_write_text


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, payload)


def csvwrite(path, rows, fields):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows({k:json.dumps(v,sort_keys=True) if isinstance(v,(dict,list)) else v
                     for k,v in row.items()} for row in rows)
    atomic_write_text(Path(path), stream.getvalue())


def _generic_energy_preparation(root,p,parent):
 prepared=parent.bound_file(p['normal_energy_plan']).resolve(strict=True)
 original=root/'measurement/generic_energy_plan.json'
 if prepared==original:return prepared,original,[]
 relative=prepared.relative_to(root)
 if (len(relative.parts)!=4 or relative.parts[0]!='controller_binding_refresh'
     or relative.parts[2:]!=('measurement','generic_energy_plan.json')):
  raise ValueError('Generic prepared energy plan is outside the normal controller refresh path')
 history=prepared.parent.parent;receipt_path=history/'REFRESH_RECEIPT.json';receipt=read(receipt_path)
 saved_binding=receipt['original_parent_plan']
 if Path(saved_binding['path'])!=history/'parent_plan.json':
  raise ValueError('Generic context refresh original Parent escapes its history')
 saved=read(parent.bound_file(saved_binding));saved_prepared=read(history/'PREPARED.json')
 def immutable_match(before,after):
  fields=lambda payload:{k:v for k,v in payload.items()if k not in ('adapter','controller_sources','normal_energy_plan','performance_admission')}
  return (fields(before)==fields(after) and
   parent.comparison_refresh_matches(before.get('performance_admission',{}),after.get('performance_admission',{})))
 if (saved_prepared.get('ok') is not True or saved_prepared.get('parent_plan_sha256')!=saved_binding['sha256']
     or not immutable_match(saved,p) or receipt.get('normal_energy_plan_before')!=saved['normal_energy_plan']
     or receipt.get('normal_energy_plan_after')!=p['normal_energy_plan']
     or Path(saved['normal_energy_plan']['path'])!=original
     or receipt.get('prepared_measured_execution_context_restored') is not True
     or receipt.get('normal_energy_plan_unchanged') is not False
     or receipt.get('scientific_inputs_and_runtime_unchanged') is not True
     or receipt.get('hardware_started') is not False
     or any(not isinstance(receipt.get(k),str) or len(str(receipt.get(k,'')))!=64 or any(c not in '0123456789abcdef' for c in str(receipt.get(k,'')))
            for k in ('budget_before_sha256','budget_after_sha256'))
     or Path(receipt['current_parent_plan']['path'])!=root/'parent_plan.json'):
  raise ValueError('Generic prepared context refresh receipt or scientific Parent binding differs')
 parent.bound_file(saved['normal_energy_plan'])
 # The shared budget can advance through other normal cases during metadata
 # refresh. These original digests remain recorded; the actual capture importer
 # independently verifies this exact task, source, limits and accepted chains.
 # Later normal controller-only refreshes can retain this exact prepared plan.
 reached={receipt['current_parent_plan']['sha256']};used=[receipt_path]
 actual_parent_sha=sha(root/'parent_plan.json')
 if actual_parent_sha not in reached:
  remaining=list((root/'controller_binding_refresh').glob('*/REFRESH_RECEIPT.json'))
  while actual_parent_sha not in reached:
   progress=False
   for path in remaining[:]:
    item=read(path);before=item.get('original_parent_plan',{});after=item.get('current_parent_plan',{})
    if before.get('sha256')not in reached:continue
    if (Path(before.get('path',''))!=path.parent/'parent_plan.json'
        or Path(after.get('path',''))!=root/'parent_plan.json'
        or item.get('normal_energy_plan_before')!=p['normal_energy_plan']
        or item.get('normal_energy_plan_after')!=p['normal_energy_plan']
        or item.get('normal_energy_plan_unchanged')!=p['normal_energy_plan']
        or item.get('prepared_measured_execution_context_restored') is not False
        or item.get('scientific_inputs_and_runtime_unchanged') is not True
        or item.get('hardware_started') is not False
        or any(not isinstance(item.get(k),str) or len(str(item.get(k,'')))!=64 or any(c not in '0123456789abcdef' for c in str(item.get(k,'')))
               for k in ('budget_before_sha256','budget_after_sha256'))
        or not immutable_match(read(parent.bound_file(before)),p)):
     raise ValueError('Generic later controller refresh changed the prepared scientific context')
    reached.add(after['sha256']);used.append(path);remaining.remove(path);progress=True
   if not progress:raise ValueError('Generic prepared context refresh does not bind the current Parent')
 return prepared,original,used


def _verify_generic_energy_command_plan(plan,original):
 from onnx_splitpoint_tool.workflow.generic_completed_task import _verified_energy_resume_command_context
 from onnx_splitpoint_tool.runners.generic_energy import canonical_hash
 frozen=('identity','config','config_path','config_sha256','remote_config_path','performance_result',
  'performance_result_sha256','collector_script','collector_script_sha256','hardware_setups_file',
  'hardware_setups_file_sha256','campaign_budget_file','campaign_row_id','campaign_source_id',
  'campaign_limits','require_environment_observation','environment_run_id')
 if any(plan.get(k)!=original.get(k)for k in frozen):
  raise ValueError('Generic resume scientific input or shared campaign differs from original parent')
 if (sorted(c['repetition_index']for c in plan['calls'])!=[1,2,3]
     or [c['repetition_index']for c in plan['calls']]!=[c['repetition_index']for c in original['calls']]):
  raise ValueError('Generic actual capture positions differ from original prepared commands')
 for call,base in zip(plan['calls'],original['calls']):
  restored=_verified_energy_resume_command_context(call,base,original['config'])
  argv=list(call['argv']);expected=list(base['argv'])
  for flag in (('--out','--run-id','--command-file','--preflight-command-file')
               if restored else ('--out','--run-id')):
   if argv.count(flag)!=1 or expected.count(flag)!=1:raise ValueError('Generic resume capture identity differs')
   argv[argv.index(flag)+1]=expected[expected.index(flag)+1]
  if argv!=expected or canonical_hash(call['argv'])!=call['command_sha256']:
   raise ValueError('Generic resume changed measured collector command or source binding')


def generic_energy_projection(case,current,source):
 # Reimport the original three physical captures through the normal Product
 # reader. The Parent's mutable Current snapshot was checked before capture;
 # the selected performance result and immutable prepared operation remain
 # the binding for later exports.
 from onnx_splitpoint_tool.generic_energy_reporting import collect_generic_energy_case
 from . import generic_energy_parent as parent
 source=Path(source).resolve(strict=True);attempt=source.parent
 root=attempt.parent.parent if attempt.parent.name=='resume_attempts' else attempt
 R=parent.configure(root)['report_root']
 if (source.name!='normal_result.json' or not root.is_relative_to(R/'energy/generic_completed')
     or root==R/'energy/generic_completed' or (attempt!=root and attempt.parent!=root/'resume_attempts')):
  raise ValueError('Generic energy source must be an existing normal completed Parent result')
 parent.configure(root)
 p=read(root/'parent_plan.json');prepared_parent=read(root/'PREPARED.json')
 if (p.get('schema')!='revision3/generic-energy-parent/v1'
     or p.get('run_dir')!=str(root) or p.get('point_count')!=1 or p.get('capture_count')!=3
     or p.get('duration_s')!=60. or p.get('automatic_retry') is not False
     or prepared_parent.get('ok') is not True
     or prepared_parent.get('parent_plan_sha256')!=sha(root/'parent_plan.json')):
  raise ValueError('Generic energy original prepared Parent binding differs')
 saved_case=read(parent.bound_file(p['case']))
 prepared_plan_path,original_plan_path,refresh_receipts=_generic_energy_preparation(root,p,parent)
 original_plan=read(original_plan_path);plan_path=prepared_plan_path
 if prepared_plan_path!=original_plan_path:
  _verify_generic_energy_command_plan(read(prepared_plan_path),original_plan)
 if attempt!=root:
  binding=read(attempt/'worker-plan.json');plan_path=parent.bound_file(binding)
  if plan_path!=attempt/'measurement/generic_energy_plan.json':
   raise ValueError('Generic resume actual worker plan escapes its attempt')
 plan=read(plan_path)
 performance_plan=parent.bound_file(p['performance_plan'])
 if (saved_case!=case or p.get('selected_identity')!=case['identity']
     or current.get('status')!='completed' or current.get('native_pair_status')!='technical_comparable'
     or performance_plan!=Path(current['execution_plan']).resolve()
     or p['performance_plan']['sha256']!=current['execution_plan_sha256']
     or original_plan_path!=root/'measurement/generic_energy_plan.json'
     or plan.get('identity')!=case['identity']
     or Path(plan['performance_result']).resolve()!=Path(current['current_result_path']).resolve()
     or plan['performance_result_sha256']!=current['raw_result_sha256']):
  raise ValueError('Generic energy current case/performance cohort differs')
 _verify_generic_energy_command_plan(plan,original_plan)
 parent.verify_calls(plan,p['remote_command_prefix'])
 if (plan.get('config_sha256')!=sha(plan['config_path']) or read(plan['config_path'])!=plan['config']):
  raise ValueError('Generic energy prepared configuration changed')
 started=read(attempt/'production-start.json');finished=read(attempt/'production-result.json')
 cleanup=read(attempt/'cleanup-status.json');worker=read(attempt/'worker_cleanup.json')
 if (started.get('selected_identity')!=case['identity']
     or started.get('normal_plan_sha256')!=p['normal_energy_plan']['sha256']
     or started.get('no_deployment_or_build') is not True or finished.get('returncode')!=0
     or cleanup.get('cleanup_proven') is not True or cleanup.get('owned_remote_active')!=0
     or cleanup.get('owned_local_active')!=0 or cleanup.get('errors')
     or worker.get('owned_remote_active')!=0 or worker.get('owned_local_active')!=0):
  raise ValueError('Generic energy actual Parent execution/cleanup not closed')
 captures_path=plan_path.parent/'capture_sources.json';captures=read(captures_path)
 if (captures.get('schema')!='onnx-splitpoint/generic-energy-capture-sources/v1'
     or captures.get('plan_sha256')!=sha(plan_path)):
  raise ValueError('Generic energy original capture manifest/plan differs')
 published=read(source)
 # Historical reports retain their original comparison. New normal reports
 # reuse only the exact bound comparison admitted by this same Parent.
 admission=p.get('performance_admission') or {}
 admitted=(admission if published.get('normal_generic_performance_pair')==admission.get('normal_pair')
           and admission.get('normal_pair_report') else None)
 observation=collect_generic_energy_case(plan,captures.get('sources') or [],saved_case,
     performance_admission=admitted)
 report_path=plan_path.parent/'reports/generic_completed_task_energy.json'
 if (observation!=read(report_path) or observation!=read(source)
     or observation.get('status')!='completed' or observation.get('normal_report_verified') is not True
     or observation.get('valid_repetitions')!=3 or observation.get('identity')!=case['identity']):
  raise ValueError('Generic energy normal reimport differs from published original report')
 stats=observation['statistics']['energy_per_completed_task_j']
 return {'normal_energy_eligible':True,'energy_status':'normal_report_verified_current_configuration',
  'energy_per_task_j':observation['energy_per_completed_task_j'],'tasks_per_j':observation['completed_tasks_per_j'],
  'energy_per_task_j_ci':[stats['ci_low'],stats['ci_high']],'energy_repeats':observation['observations'],
  'energy_repeat_statistics':observation['statistics'],
  'energy_efficiency_claim_eligible':observation['energy_efficiency_claim_eligible'] is True and current.get('normal_claim_eligible') is True,
  'energy_provenance':{'normal_report_verified':True,'normal_result':str(source),'normal_result_sha256':sha(source),
   'normal_report':str(report_path),'normal_report_sha256':sha(report_path),
   'normal_source_report':str(report_path),'normal_source_report_sha256':sha(report_path),
   'parent_plan':str(root/'parent_plan.json'),'parent_plan_sha256':sha(root/'parent_plan.json'),
   'parent_attempt':str(attempt),'parent_cleanup':str(attempt/'cleanup-status.json'),
   'parent_cleanup_sha256':sha(attempt/'cleanup-status.json'),
   'normal_energy_plan':str(plan_path),'normal_energy_plan_sha256':sha(plan_path),
   'original_energy_plan':str(original_plan_path),'original_energy_plan_sha256':sha(original_plan_path),
   'prepared_energy_plan':str(prepared_plan_path),'prepared_energy_plan_sha256':sha(prepared_plan_path),
   'controller_refresh_receipts':[{'path':str(path),'sha256':sha(path)}for path in refresh_receipts],
   'capture_sources':str(captures_path),'capture_sources_sha256':sha(captures_path),
   'performance_result':str(Path(plan['performance_result']).resolve()),'performance_result_sha256':plan['performance_result_sha256'],
   'parent_cleanup_proven':True,'normal_energy_report_claim_eligible':observation['energy_efficiency_claim_eligible'],
   'scientific_primary_method':observation['scientific_primary_method'],
   'raw_metric':'mean_per_repeat(E/N)','efficiency_metric':'mean_per_repeat(N/E)'}}


def project_current_energy(case,source,*,current_path,expected_current_sha256):
 # Energy adds evidence to an already admitted exact performance/Quality row.
 # Full exports remain the normal path for newly changed performance/Quality.
 current_path=Path(current_path);content=current_path.read_bytes()
 if hashlib.sha256(content).hexdigest()!=expected_current_sha256:
  raise ValueError('Generic Current changed before targeted energy import')
 previous=json.loads(content);identity=case['identity']
 matches=[i for i,row in enumerate(previous['rows'])
  if all(row.get(k)==identity[k]for k in ('model_id','case_id','setup_id','precision'))
  and canonical_direction(row.get('direction',''))==canonical_direction(identity['direction'])
  and(not case.get('case_key')or row.get('case_key')==case['case_key'])]
 if len(matches)!=1:raise ValueError('Generic exact energy case missing or ambiguous in Current')
 index=matches[0];current=previous['rows'][index]
 projection=generic_energy_projection(case,current,source)
 payload=copy.deepcopy(previous);row=payload['rows'][index];row.update(projection);row.pop('energy_error',None)
 if any(row!=previous['rows'][i]for i,row in enumerate(payload['rows'])if i!=index):
  raise ValueError('Generic targeted energy import changed another Current row')
 key=(*tuple(identity[k]for k in ('model_id','case_id','setup_id','precision')),canonical_direction(identity['direction']))
 payload['energy_errors']=[error for error in previous.get('energy_errors',[])
  if not(error.get('case_key')==case.get('case_key')and case.get('case_key'))
  and tuple(error.get('identity')or())!=key]
 payload['updated_at']=datetime.datetime.now(datetime.timezone.utc).isoformat()
 if sha(current_path)!=expected_current_sha256:
  raise ValueError('Generic Current changed while normal energy sources were verified')
 write(current_path,payload)
 csvwrite(current_path.with_suffix('.csv'),payload['rows'],_current_csv_fields())
 return row


def _current_csv_fields():
 return ['model_id','case_id','direction','precision','setup_id','comparison_backend','case_key','task','completed_stage','completed_comparison_endpoint_hash','physical_source_output_endpoint_id','status','reason','valid_repetitions','fps_median','fps_samples','completed_counts','measured_makespans_s','pids','process_start_ticks','boot_ids','native_pair_status','fresh_native_fps_median','native_over_generic_completed_task_ratio','normal_quality_eligible','normal_claim_eligible','request_latency_status','quality_status','energy_status','normal_energy_eligible','energy_per_task_j','tasks_per_j','energy_per_task_j_ci','energy_efficiency_claim_eligible','energy_provenance','energy_error','current_result_path','raw_result_sha256','error']
