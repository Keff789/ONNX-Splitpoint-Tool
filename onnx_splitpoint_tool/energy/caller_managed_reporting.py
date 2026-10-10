"""Read-only normal report import of separately acquired single captures.

Capture descriptors are provenance, never acceptance assertions. Each source is
re-imported through the ordinary producer aggregate verifier. Raw run indices
and collector artifacts remain unchanged; logical indices exist only in the
report projection passed to the existing final reporting gates.
"""
from __future__ import annotations
from collections.abc import Mapping
from pathlib import Path
import copy,fcntl,hashlib,json,math,shlex

SCHEMA = 'onnx-splitpoint/caller-managed-energy-captures/v1'

def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def _read(path):return json.loads(Path(path).read_text())
def _importer():
 try:
  from scripts import run_native_producer_energy_from_summary as importer
 except ImportError:
  from onnx_splitpoint_tool.resources.remote_scripts import run_native_producer_energy_from_summary as importer
 return importer
def _option(argv,name):
 return _importer()._single_command_option_parts(argv,name)[0]
def _number(value):
 if isinstance(value,bool)or not isinstance(value,(int,float))or not math.isfinite(value):raise ValueError('nonfinite or missing capture metric')
 return float(value)
def _quality(row):
 return _importer()._energy_quality_result_projection(row,measurement_started=True,raw_energy_collected=True)
def _full_expectation(binding,row):
 path=Path(binding['path'])
 if _sha(path)!=binding['sha256']:raise ValueError('Full performance mode sidecar changed')
 expected=_read(path);request=Path(expected['request'])
 if _sha(request)!=expected['request_sha256']:raise ValueError('Full performance request changed')
 request_row=_read(request)['row'];mode=expected['expected_actual_performance_mode']
 if mode!=expected['expected_auto_resolved_energy_mode']or any(expected[k]!=row[k]for k in['backend','model','setup_id']):raise ValueError('Full performance mode identity differs')
 contract_path=Path(row['command_contract_file'])
 if _sha(contract_path)!=row['command_contract_file_sha256']:raise ValueError('Full command contract file changed')
 contract=_read(contract_path)
 if contract['contract_sha256']!=row['successful_command_contract_sha256']:raise ValueError('Full source contract differs')
 workload=contract['energy_workload'];concurrency=request_row.get('measurement_concurrency')
 helper_role=workload.get('runner_artifact');input_role=workload.get('runtime_input_artifact')
 if contract.get('artifacts',{}).get(helper_role,{}).get('sha256')!=expected['helper_sha256']:raise ValueError('Full expected mode helper differs from sealed runtime')
 if input_role and contract['artifacts'][input_role]['sha256']!=request_row.get('runtime_input_sha256'):raise ValueError('Full exact input differs from actual performance')
 if concurrency is not None and workload.get('inflight',contract['runtime_options'].get('inflight'))!=concurrency:raise ValueError('Full concurrency differs from actual performance')
 frozen=request_row.get('frozen_host_postprocess_contract')
 if frozen and workload.get('frozen_postprocess_contract',{}).get('contract_sha256')!=frozen.get('contract_sha256'):raise ValueError('Full frozen task differs from actual performance')
 sources=expected['performance_sources']
 if len(sources)!=3 or len({(s['runner_pid'],s['runner_start_monotonic_ns'])for s in sources})!=3:raise ValueError('three distinct actual Full performance processes required')
 if any(len({s[k]for s in sources})!=3 for k in['repeat','raw_report','raw_report_sha256','normalized_report','normalized_report_sha256']):raise ValueError('duplicate actual Full performance report')
 observations={s['repeat']:s for s in request_row['fresh_performance_provenance']['observations']}
 repetitions=request_row['repetition_records']
 for source in sources:
  observation=observations[source['repeat']]
  if any(source[k]!=observation[k]for k in source):raise ValueError('Full process descriptor differs from bound performance observation')
  directory=Path(source['normalized_report']).parent;success_path=directory/'success.json'
  if _sha(success_path)!=observation['success_receipt_sha256']:raise ValueError('Full successful report receipt changed')
  success=_read(success_path);telemetry=_read(directory/'telemetry/telemetry_receipt.json')
  identity='actual_process:'+row['setup_id']+':'+str(source['runner_pid'])+':'+str(source['runner_start_monotonic_ns'])
  matching=[r for r in repetitions if r.get('runtime_instance_id')==identity]
  if len(matching)!=1 or any(telemetry.get(k)!=source[k]for k in['runner_pid','runner_start_monotonic_ns'])or telemetry.get('runner_rc')!=0:raise ValueError('Full descriptor differs from actual successful process')
  repetition=matching[0]
  if telemetry.get('argv')!=repetition['actual_execution_argv']or telemetry.get('env_overlay')!=repetition['actual_execution_env']or success['command_sha256']!=observation['actual_command_sha256']or success['raw_report_sha256']!=source['raw_report_sha256']or success['report_sha256']!=source['normalized_report_sha256']:raise ValueError('Full actual command/report process binding differs')
  for kind in['raw_report','normalized_report']:
   if _sha(source[kind])!=source[kind+'_sha256']:raise ValueError('Full performance report changed')
  report=_read(source['normalized_report'])
  if report.get('postprocess_runtime_mode')!=mode or report.get('runtime_input_binding_verified')is not True:raise ValueError('Full performance actual mode/input unverified')
 for name in['helper','selector']:
  if _sha(expected[name])!=expected[name+'_sha256']:raise ValueError('Full mode implementation changed')
 return expected

def _full_runtime_proof(importer,row,command,output,record,expected):
 directory,_=importer._selected_energy_repeat_run_dir(record,measurement_root=output,logical_index=0,campaign_command=command)
 stdout=directory/'workload_stdout.log';binding=record['collector_and_workload_log_diagnostics']['workload_stdout']
 if binding.get('available')is not True or stdout.stat().st_size>8*1024*1024 or stdout.stat().st_size!=binding['bytes']or _sha(stdout)!=binding['sha256']:raise ValueError('Full runtime stdout not bound')
 values=[]
 for line in stdout.read_text().splitlines():
  try:value=json.loads(line)
  except ValueError:continue
  if isinstance(value,dict)and value.get('fresh_report_verified')is True:values.append(value)
 if len(values)!=1:raise ValueError('one actual Full completion report required')
 value=values[0];count=record['energy_work_units_used']
 if any(value.get(k)!=row[k]for k in['backend','model'])or value.get('source_contract_sha256')!=row['successful_command_contract_sha256']or value.get('postprocess_runtime_mode')!=expected['expected_actual_performance_mode']or value.get('postprocess_completion_verified')is not True or value.get('runtime_input_binding_verified')is not True or value.get('minimum_duration_satisfied')is not True or value.get('completed_work_units')!=count or value.get('postprocess_completed_frames')!=count or value.get('requested_duration_s')!=row['duration_s']:raise ValueError('Full fresh mode/input/completed count/duration differs')
 return {'stdout_sha256':binding['sha256'],'report':value}

def _aggregate(records,aggregates):
 from onnx_splitpoint_tool.energy.collector import _repeat_statistics,_energy_repeat_contract_complete
 # Shared scalar/contract metadata may be carried only when all actual source
 # aggregates agree. Per-repeat means and counts are always recalculated below.
 common={k:copy.deepcopy(v)for k,v in aggregates[0].items()if all(a.get(k)==v for a in aggregates[1:])and k not in{'runs','command','energy_window_method_ab'}and not(k.startswith('avg_')or 'statistics'in k or k.endswith('_count')or k.endswith('_runs'))}
 invariants=['setup_id','run_id','scientific_primary_method','scientific_primary_method_frozen','energy_physical_scope','energy_physical_scope_canonical','energy_window_requested','energy_window_effective_values','full_system_current_scale_factor_applied','energy_calibration_sha256','host_normalization_role','preflight_expected_command_contract_sha256']
 for key in invariants:
  if any(a.get(key)!=aggregates[0].get(key)for a in aggregates[1:]):raise ValueError('mixed capture policy: '+key)
 if not _energy_repeat_contract_complete(records,requested_count=3,require_ab=True):raise ValueError('normal three-repeat primary gate rejected')
 level=.95
 def stats(values):return _repeat_statistics(values,level)
 energies=[_number(r['scientific_primary_energy_j'])for r in records];counts=[_number(r['energy_work_units_used'])for r in records]
 if any(n<=0 for n in counts)or any(e<=0 for e in energies):raise ValueError('nonpositive energy or completed count')
 primary={'method':'command_marker_window','scientific_role':'primary','energy_j':stats(energies),'energy_per_work_unit_j':stats([e/n for e,n in zip(energies,counts)]),'work_units_per_j':stats([n/e for e,n in zip(energies,counts)]),'active_duration_s':stats([_number(r['scientific_primary_active_duration_s'])for r in records]),'avg_power_w':stats([_number(r['scientific_primary_avg_power_w'])for r in records])}
 shadow=[r for r in records if r.get('chapter4_legacy_shadow_status')=='available'and isinstance(r.get('chapter4_legacy_shadow_energy_j'),(int,float))]
 shadowstats={'method':'chapter4_legacy','scientific_role':'shadow_only','affects_primary_result':False,'affects_final_gate':False,'energy_j':stats([_number(r['chapter4_legacy_shadow_energy_j'])for r in shadow]),'energy_per_work_unit_j':stats([_number(r['chapter4_legacy_shadow_energy_j'])/_number(r['energy_work_units_used'])for r in shadow])}
 comparisons=[r['window_method_comparison']for r in records if isinstance(r.get('window_method_comparison'),dict)and r['window_method_comparison'].get('status')=='ok'and r['window_method_comparison'].get('same_raw_trace_verified')is True]
 abstats={'diagnostic_only':True,'difference_direction':'legacy_minus_command_window','affects_primary_result':False,'affects_final_gate':False}
 for metric in['energy_j','duration_s','average_power_w']:
  abstats[metric]={f:stats([_number(c['legacy_minus_command_window'][metric][f])for c in comparisons if isinstance(c.get('legacy_minus_command_window',{}).get(metric,{}).get(f),(int,float))])for f in['absolute','relative_percent']}
 common.update(schema='onnx-splitpoint/derived-caller-managed-energy-aggregate/v1',ok=True,status='verified_source_capture_projection',run_count=3,requested_valid_repeat_count=3,requested_run_count=3,effective_run_count=3,materialized_logical_repeat_count=3,repeat_contract_complete=True,repeat_contract_status='complete',raw_postprocessed_run_count=3,valid_postprocessed_runs=3,preflight_verified_run_count=3,runtime_completed_work_unit_run_count=3,postprocess_status='ok',final_energy_gate_status='pass',runs=records,confidence_level=level,scientific_primary_method='command_marker_window',scientific_primary_method_frozen=True,scientific_primary_energy_status='available',scientific_primary_valid_run_count=3,scientific_primary_energy_statistics=primary,scientific_primary_energy_total_j=primary['energy_j']['mean'],scientific_primary_energy_per_work_unit_j=primary['energy_per_work_unit_j']['mean'],scientific_primary_active_duration_s=primary['active_duration_s']['mean'],scientific_primary_avg_power_w=primary['avg_power_w']['mean'],scientific_primary_claim_eligible=all(r.get('energy_efficiency_claim_eligible')is True for r in records),energy_efficiency_claim_eligible=all(r.get('energy_efficiency_claim_eligible')is True for r in records),avg_energy_work_units_used=stats(counts)['mean'],avg_energy_work_units_per_j=primary['work_units_per_j']['mean'],scientific_shadow_energy_status='available'if len(shadow)==3 else'partially_available'if shadow else'unavailable',scientific_shadow_valid_run_count=len(shadow),scientific_shadow_claim_eligible=False,scientific_shadow_affects_primary_result=False,scientific_shadow_affects_final_gate=False,chapter4_legacy_shadow_statistics=shadowstats,chapter4_legacy_shadow_energy_total_j=shadowstats['energy_j']['mean'],chapter4_legacy_shadow_energy_per_work_unit_j=shadowstats['energy_per_work_unit_j']['mean'],window_method_comparison_statistics=abstats,legacy_window_comparison_attempted_runs=3,legacy_window_comparison_successful_runs=len(comparisons),legacy_window_comparison_diagnostic_only=True)
 for key in['energy_total_j','avg_power_w','active_duration_s','energy_per_work_unit_j','energy_work_units_per_j','energy_dynamic_j','host_normalized_energy_est_j','host_normalized_energy_per_work_unit_est_j','host_normalized_average_power_est_w']:
  if all(isinstance(r.get(key),(int,float))for r in records):common['avg_'+key]=stats([_number(r[key])for r in records])['mean']
 return common

def import_caller_managed_energy_item(item):
 """Normalize an explicit source set before ordinary collect_native_energy gates."""
 run=item.get('run')or{}
 if run.get('caller_managed_energy_capture_schema')!=SCHEMA:return item
 try:
  importer=_importer()
  row=item['row'];sources=run['caller_managed_energy_capture_sources'];digest=row['successful_command_contract_sha256']
  if len(sources)!=3 or sorted(s['logical_repeat_index']for s in sources)!=[0,1,2]:raise ValueError('three logical source positions required')
  if row.get('measurement_requested_repeats')!=3 or row.get('measurement_effective_repeats')!=3:raise ValueError('original three-repeat planning policy required')
  if row.get('diagnostic_only')is not False or _quality(row)['energy_quality_qualified']is not True:raise ValueError('qualified non-diagnostic source operation required')
  full=row['backend'].startswith('native_full_');expected=_full_expectation(run['full_mode_expectation_sidecar'],row)if full else None
  records=[];aggregates=[];proofs=[];identities=[];bindings=[]
  normal_argv=shlex.split(row['measure_command'])
  for source in sorted(sources,key=lambda s:s['logical_repeat_index']):
   argv=source['command'];command=shlex.join(argv);receipt=source['process_receipt'];output=Path(source['output_dir']).resolve();raw=Path(source['raw_aggregate_path']).resolve()
   if raw!=output/'energy_aggregate.json'or _sha(raw)!=source['raw_aggregate_sha256']:raise ValueError('raw capture hash or path changed')
   if _sha(source['process_receipt_path'])!=source['process_receipt_sha256']or _read(source['process_receipt_path'])!=receipt:raise ValueError('actual process receipt changed')
   if hashlib.sha256(command.encode()).hexdigest()!=source['command_sha256']or receipt['argv']!=argv or receipt['command_sha256']!=source['command_sha256']or receipt.get('returncode')!=0 or type(receipt.get('started_ns'))is not int or type(receipt.get('finished_ns'))is not int or receipt['finished_ns']<=receipt['started_ns']or receipt.get('aggregate_absent_before_execution')is not True:raise ValueError('actual capture process receipt mismatch')
   if _option(argv,'--runs')!='1'or '--exact-run-count'not in argv or _option(argv,'--out')!=str(output)or _option(argv,'--task-logical-repeat')!=source['logical_repeat']:raise ValueError('actual single capture envelope mismatch')
   restored=list(argv);idx=restored.index('--task-logical-repeat');del restored[idx:idx+2];restored.remove('--exact-run-count');restored=importer._replace_command_option(restored,'--runs','3');restored=importer._replace_command_option(restored,'--out',_option(normal_argv,'--out'))
   if restored!=normal_argv:raise ValueError('capture command changed planned runtime/quality/source policy')
   attached=importer._attach_energy_aggregate({'rc':0},command,output,expected_runs=1,expected_effective_runs=1,expected_setup_id=row['setup_id'],expected_run_id=_option(argv,'--run-id'),expected_command_contract_sha256=digest,execution_started_ns=receipt['started_ns'],aggregate_absent_before_execution=True,expected_native_row=row)
   if attached.get('energy_aggregate_verified')is not True:raise ValueError('ordinary single aggregate verification rejected: '+repr(attached.get('energy_aggregate_validation_errors',[])+attached.get('energy_aggregate_completion_errors',[])))
   aggregate=attached['energy_aggregate'];values=aggregate['runs']
   if aggregate.get('diagnostic_only')is True or any(r.get('diagnostic_only')is True for r in values):raise ValueError('explicit diagnostic capture cannot be admitted')
   if len(values)!=1 or values[0]['run_index']!=0 or not aggregate.get('caller_managed_repeat_capture_complete'):raise ValueError('one actual caller-managed raw repeat required')
   record=values[0];count=record['energy_work_units_used'];nonce=record['preflight_evidence']['nonce'];directory,_=importer._selected_energy_repeat_run_dir(record,measurement_root=output,logical_index=0,campaign_command=command)
   if record.get('campaign_reused_repeat')is not None or not nonce or type(count)is not int or count<=0 or record.get('runtime_work_unit_evidence',{}).get('exact')is not True or record.get('runtime_work_unit_evidence',{}).get('count')!=count:raise ValueError('fresh physical exact completed-task count required')
   if _number(record.get('energy_configured_workload_duration_s'))!=60 or _number(record.get('scientific_primary_active_duration_s'))<60:raise ValueError('actual completed-task duration too short')
   with Path(_option(argv,'--campaign-budget-file')).open()as stream:
    fcntl.flock(stream,fcntl.LOCK_SH);budget=json.load(stream)
   task=budget['tasks'][_option(argv,'--campaign-row-id')]
   if task['limits']!={'max_chains':9,'max_transport_failures':20,'max_retries':2}:raise ValueError('original physical case budget changed')
   cohort=importer._command_cohort(command)
   if cohort!=(row.get('campaign_cohort')or''):raise ValueError('capture cohort differs from normal source operation')
   matching=[c for c in task['chains']if c.get('logical_repeat')==source['logical_repeat']and c.get('valid')is True and c.get('finished')is True and (c.get('cohort_id')or'')==cohort]
   if len(matching)!=1 or matching[0]['run_directory']!=str(directory)or matching[0].get('source_completion_verified')is not True:raise ValueError('actual capture differs from accepted original case-budget chain')
   if _sha(raw)!=source['raw_aggregate_sha256']:raise ValueError('raw capture changed during read-only import')
   identities.append((str(output),str(directory.resolve()),nonce,source['logical_repeat'],source['raw_aggregate_sha256']))
   index=source['logical_repeat_index'];records.append({**copy.deepcopy(record),'logical_repeat_index':index,'caller_managed_source_aggregate_sha256':source['raw_aggregate_sha256']});aggregates.append(aggregate)
   if full:bindings.append(_full_runtime_proof(importer,row,command,output,record,expected))
   else:
    evidence=attached.get('fresh_energy_completion_evidence')or[]
    if len(evidence)!=1:raise ValueError('fresh split completion evidence missing')
    proofs.append({**copy.deepcopy(evidence[0]),'logical_repeat_index':index})
   bindings.append({'logical_repeat_index':index,'raw_aggregate_path':str(raw),'raw_aggregate_sha256':source['raw_aggregate_sha256'],'capture_directory':str(directory),'preflight_nonce':nonce})
  if any(len({x[i]for x in identities})!=3 for i in range(5)):raise ValueError('duplicate actual capture directory, nonce, or logical identity')
  aggregate=_aggregate(records,aggregates);projected=copy.deepcopy(item);projected['run']={**copy.deepcopy(run),'rc':0,'energy_aggregate':aggregate,'energy_aggregate_embedded':True,'energy_aggregate_verified':True,'energy_aggregate_complete':True,'caller_managed_capture_import_status':'verified','caller_managed_capture_source_bindings':bindings,'fresh_energy_completion_evidence':proofs,**_quality(row)};projected['ok']=True;return projected
 except (OSError,ValueError,TypeError,KeyError,IndexError)as exc:
  return {'row':copy.deepcopy(item.get('row')or{}),'ok':False,'run':{'rc':2,'caller_managed_capture_import_status':'rejected','caller_managed_capture_import_error':str(exc),'energy_aggregate':{'run_count':3,'valid_postprocessed_runs':0,'repeat_contract_complete':False,'final_energy_gate_status':'fail','scientific_primary_method_frozen':True,'scientific_primary_method':'command_marker_window','scientific_primary_energy_status':'unavailable'}}}
