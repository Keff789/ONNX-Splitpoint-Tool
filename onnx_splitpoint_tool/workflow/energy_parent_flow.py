"""Continue one existing prepared energy parent through its normal product APIs.

Parent plans, campaign budgets, resource journals and bound import commands are
existing inputs. This module neither creates a campaign nor selects new cases.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import time
import signal
import threading
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace

from .artifacts import write_json

GENERIC_SCHEMA='revision3/generic-energy-parent/v1'
NATIVE_SCHEMA='yolov7-full-fix/bounded-energy-parent-v1'
PRODUCT_SOURCES=('generic_energy_parent.py','native_energy_parent.py','energy_resume.py','energy_parent_flow.py')


def read(path):
    path=Path(path)
    if not path.is_file() or path.is_symlink():raise ValueError('regular parent evidence required: '+str(path))
    return json.loads(path.read_text())


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def parent_plan_path(run):
    run=Path(run).expanduser().resolve(strict=True)
    paths=[p for p in (run/'parent_plan.json',run/'energy_parent_plan.json') if p.is_file()]
    if len(paths)!=1:raise ValueError('one exact saved energy parent plan required')
    p=paths[0];config=read(p)
    expected=GENERIC_SCHEMA if p.name=='parent_plan.json' else NATIVE_SCHEMA
    if config.get('schema')!=expected or config.get('run_dir')!=str(run):
        raise ValueError('saved energy parent schema or run identity differs')
    return p,config


def is_energy_parent(run):
    run=Path(run)
    return (run/'parent_plan.json').is_file() or (run/'energy_parent_plan.json').is_file()


def parent_context(run=None, *, plan_path=None, adapter_path=None, bound_context=None):
    """Resolve report data from the saved budget, code from the actual plan."""
    import onnx_splitpoint_tool
    controller=Path(onnx_splitpoint_tool.__file__).resolve().parent.parent
    config=None
    if run is not None:
        run=Path(run).expanduser().resolve()
        if is_energy_parent(run):
            _,config=parent_plan_path(run)
            budget=Path(config['campaign_budget_file']).resolve(strict=True)
            root=budget.parent.parent
            generic=config['schema']==GENERIC_SCHEMA
            expected=root/'energy'/('generic_completed' if generic else '')/run.name
            if run!=expected:raise ValueError('prepared parent is outside its exact shared-budget layout')
            if generic:
                binding=config['normal_energy_plan'];normal_path=Path(binding['path'])
                if sha(normal_path)!=binding['sha256']:raise ValueError('saved normal energy plan changed')
                normal=read(normal_path)
                if normal['campaign_budget_file']!=str(budget) or normal['identity']!=config['selected_identity']:
                    raise ValueError('normal energy plan differs from parent scope')
                script=Path(normal['collector_script']).resolve(strict=True)
                if script.name!='energy_measurement_cli.py' or script.parent.name!='scripts':
                    raise ValueError('normal collector source root unavailable')
                controller=script.parent.parent
                original_adapter=Path(config['adapter']['path']).resolve(strict=True)
            else:
                # Native preparation stores the actual normal planner argv.
                check=Path(config.get('plancheck_root') or run/'plancheck')/'native_producer_energy_results.json'
                if check.is_file():
                    argv=(read(check).get('plan') or {}).get('cmd') or []
                    scripts=[Path(a) for a in argv if isinstance(a,str) and Path(a).name=='native_producer_energy_plan.py']
                    if len(scripts)==1 and scripts[0].is_absolute() and scripts[0].is_file():
                        controller=scripts[0].resolve().parent.parent
                original_adapter=root/'operations/energy_parent_qualified.py'
        else:
            # Compatibility preparation derives its report root from the
            # requested normal parent location, never an absolute campaign.
            if run.parent.name=='generic_completed' and run.parent.parent.name=='energy':
                root=run.parents[2];generic=True
            elif run.parent.name=='energy':root=run.parent.parent;generic=False
            else:raise ValueError('normal parent preparation layout required')
            budget=root/'energy/energy_task_budget.json'
            original_adapter=root/('operations/generic_energy_parent/adapter.py' if generic else 'operations/energy_parent_qualified.py')
    elif plan_path is not None:
        path=Path(plan_path).resolve(strict=True);plan=read(path)
        if plan.get('shared_energy_campaign_budget'):
            budget=Path(plan['shared_energy_campaign_budget']).resolve(strict=True)
            root=budget.parent.parent
        else:
            # Normal prepared follow-up performance plans contain rows and an
            # output_root, not an energy budget. Derive ONLY their existing
            # exact report layout; the later Parent gates keep science/Scope.
            output=Path(str(plan.get('output_root') or ''))
            if (plan.get('schema')!='onnx-splitpoint/generic-completion-plan'
                    or not isinstance(plan.get('rows'),list) or not plan['rows']
                    or path.name!='execution-plan.json' or path.parent.name!='plan'
                    or path.parents[2].name!='batches' or path.parents[3].name!='generic_followup'
                    or not output.is_absolute() or output.resolve()!=path.parent.parent):
                raise ValueError('normal follow-up performance plan context unavailable')
            root=path.parents[4]
            budget=(root/'energy/energy_task_budget.json').resolve(strict=True)
        original_adapter=root/'operations/generic_energy_parent/adapter.py'
        if bound_context is not None:
            if (root!=Path(bound_context['report_root']).resolve()
                    or budget!=Path(bound_context['budget']).resolve()
                    or not path.is_relative_to(root)):
                raise ValueError('performance plan differs from already bound Parent report/budget context')
            controller=Path(bound_context['controller_root']).resolve(strict=True)
            if original_adapter.resolve()!=Path(bound_context['adapter']).resolve():
                raise ValueError('performance plan differs from already bound Parent adapter')
    elif adapter_path is not None:
        original_adapter=Path(adapter_path).resolve(strict=True)
        root=original_adapter.parents[2] if original_adapter.parent.name=='generic_energy_parent' else original_adapter.parents[1]
        budget=root/'energy/energy_task_budget.json'
    else:raise ValueError('saved parent or performance plan required')
    if budget.name!='energy_task_budget.json' or budget.parent.name!='energy':
        raise ValueError('normal shared campaign budget layout required')
    if adapter_path is not None and Path(adapter_path).resolve()!=original_adapter.resolve():
        raise ValueError('compatibility entry differs from its saved parent adapter')
    return {'report_root':root,'controller_root':controller,'budget':budget,
            'adapter':original_adapter.resolve(),'config':config}



def verify_product_sources(controller_root, bindings):
    """The installed implementation must match its ordinary controller bindings."""
    import onnx_splitpoint_tool
    installed=Path(onnx_splitpoint_tool.__file__).resolve().parent.parent
    source_hashes=({item['path']:item['sha256'] for item in bindings}
                   if isinstance(bindings,list) else {str(Path(controller_root)/key):value for key,value in bindings.items()})
    for name in PRODUCT_SOURCES:
        relative=Path('onnx_splitpoint_tool/workflow')/name
        path=Path(controller_root)/relative
        expected=source_hashes.get(str(path))
        if not expected or sha(path)!=expected or sha(installed/relative)!=expected:
            raise ValueError('actual installed product controller source is not normally bound: '+name)
    for path,expected in source_hashes.items():
        try:relative=Path(path).relative_to(controller_root)
        except ValueError:continue
        if relative.parts and relative.parts[0]=='onnx_splitpoint_tool' and sha(installed/relative)!=expected:
            raise ValueError('actual installed controller differs from its normal binding: '+str(relative))
    for name in ('native_producer_final_report.py','native_producer_energy_plan.py','run_native_producer_energy_from_summary.py'):
        script=Path(controller_root)/'scripts'/name
        expected=source_hashes.get(str(script))
        if expected and sha(installed/'onnx_splitpoint_tool/resources/remote_scripts'/name)!=expected:
            raise ValueError('installed normal controller reader differs from its original script binding: '+name)

def latest_attempt(run):
    attempts=[p for p in [run,*run.glob('resume_attempts/*')] if(p/'production-start.json').is_file()]
    return max(attempts,key=lambda p:(p/'production-start.json').stat().st_mtime) if attempts else None


def _command(argv, *, parent, handoff=None, previous=None):
    argv=shlex.split(argv) if isinstance(argv,str) else list(argv)
    if not argv or not all(isinstance(a,str) and a for a in argv):raise ValueError('saved normal command missing')
    for key,value in {'parent':parent,'handoff':handoff or '', 'previous_attempt':previous or ''}.items():
        argv=[a.replace('{'+key+'}',str(value)) for a in argv]
    return argv



def _parent_command(group, config, *, parent, handoff, previous=None):
    """Keep the existing serial wrapper's stored Native command fallback."""
    key='resume_command' if previous is not None else 'run_command'
    if group.get(key):
        return _command(group[key],parent=parent,handoff=handoff,previous=previous)
    if config['schema']!=NATIVE_SCHEMA:
        raise ValueError('Generic parent requires its stored normal run/resume command')
    prepared=_command(group.get('prepare_command'),parent=parent) if group.get('prepare_command') else []
    if not prepared or prepared.count('prepare')!=1:
        raise ValueError('legacy Native parent has no exact stored preparation entry')
    prefix=prepared[:prepared.index('prepare')]
    if not prefix:raise ValueError('stored Native preparation entry missing')
    return prefix+(['resume','--previous-attempt',str(previous)] if previous is not None else ['run'])+[
        '--run-dir',str(parent),'--slot-handoff',str(handoff)]


def _parent_importer(group, config, *, parent):
    stored=group.get('import_command')
    if stored is None and config['schema']==NATIVE_SCHEMA:
        stored=config.get('import_command') or config.get('completed_row_validator')
    if not stored:raise ValueError('stored normal parent importer missing')
    command=[arg for arg in _command(stored,parent=parent) if arg!='--allow-partial']
    # Historical Native plans store a shell-style Python name. Use the SAME
    # explicitly bound interpreter without rewriting that original command.
    if config['schema']==NATIVE_SCHEMA and command[0] in {'python','python3'}:
        command[0]=config['python']
    return command

def saved_group(run,config,root):
    queue=root/'operations/token_saving_batch_20261009/queue.json'
    payload=read(queue)
    generic=config['schema']==GENERIC_SCHEMA
    regime='generic_completed' if generic else 'native'
    matches=[g for g in payload['groups'] if Path(g['parent']).resolve()==run and g.get('regime','native')==regime]
    if len(matches)!=1:raise ValueError('saved parent has no unique normal queue identity')
    group=matches[0]
    if generic:
        identity=config['selected_identity'];expected=[[ 'generic_completed:'+identity['comparison_backend'],
            identity['model_id'],identity['case_id'],identity['setup_id']]]
    else:expected=config['capture_identities']
    if (group['identities']!=expected or group['points']!=len(expected)
            or {row[3] for row in expected}!={group['setup']}):
        raise ValueError('stored queue scope differs from the exact prepared parent')
    return queue,group


def describe_parent(run):
    run=Path(run).expanduser().resolve(strict=True);path,config=parent_plan_path(run)
    prepared=read(run/'PREPARED.json');generic=config['schema']==GENERIC_SCHEMA
    if prepared.get('ok') is not True or prepared.get('parent_plan_sha256' if generic else 'plan_sha256')!=sha(path):
        raise ValueError('prepared parent binding changed')
    context=parent_context(run);queue,group=saved_group(run,config,context['report_root'])
    previous=latest_attempt(run)
    if previous is None:state='PREPARED'
    elif (previous/'normal_result.json').is_file() if generic else (run/'measurement/native_producer_energy_results.json').is_file():
        state='IMPORT_PENDING'
    else:state='RESUME_REQUIRED'
    importer=_parent_importer(group,config,parent=run)
    parent_flag='--parent' if generic else '--cohort'
    if (importer.count(parent_flag)!=1 or importer[importer.index(parent_flag)+1]!=str(run)
            or importer[0]!=config.get('python_executable',config.get('python'))):
        raise ValueError('saved importer targets another parent or interpreter')
    scripts=[Path(a) for a in importer if a.endswith('.py')]
    if len(scripts)!=1 or not scripts[0].is_absolute() or not scripts[0].is_file() or scripts[0].is_symlink():
        raise ValueError('saved importer source unavailable')
    return {'run':run,'path':path,'config':config,'context':context,'queue':queue,'group':group,
            'previous':previous,'state':state,'importer':importer,'importer_script':scripts[0]}


def _import_saved(item):
    # No parent/collector callback is reachable in this function.
    before=sha(item['importer_script'])
    completed=subprocess.run(item['importer'],cwd=item['context']['controller_root'],capture_output=True,text=True,timeout=180)
    log=item['context']['report_root']/'operations/token_saving_batch_20261009'/(item['run'].name+'_import.log')
    log.parent.mkdir(parents=True,exist_ok=True)
    log.write_text(completed.stdout+completed.stderr)
    if sha(item['importer_script'])!=before:raise ValueError('bound normal importer changed during projection')
    if completed.returncode:raise RuntimeError('normal saved import failed; no capture retry: '+completed.stderr[-2000:])
    result=json.loads(completed.stdout)
    if len(result.get('results',[]))!=item['group']['points']:
        raise ValueError('normal imported parent remains incomplete; preserve accepted points')
    return result



def _run_parent_command(item, command):
    """Use ordinary owned streaming and let the normal Parent finish cleanup."""
    from onnx_splitpoint_tool.native_progress import run_streaming
    from onnx_splitpoint_tool.process_control import ProcessTreeRegistry
    registry=ProcessTreeRegistry();cancel=threading.Event()
    handlers={signum:signal.signal(signum,lambda *_:cancel.set())
              for signum in (signal.SIGINT,signal.SIGTERM)}
    def cooperative_cleanup():
        # The registry supplies identity-checked direct children. Compare the
        # saved argv again before signalling only this normal parent; never a
        # collector, DUT process, process group or unrecorded owner.
        owners=registry.active_pids()
        if len(owners)>1:raise ValueError('normal parent command ownership is ambiguous')
        if owners:
            pid=owners[0];path=Path('/proc')/str(pid)/'cmdline'
            actual=[part for part in path.read_bytes().split(b'\0') if part]
            if actual!=[part.encode() for part in command]:
                raise ValueError('normal parent command identity changed before cancellation')
            os.kill(pid,signal.SIGINT)
            deadline=time.monotonic()+30.
            while pid in registry.active_pids() and time.monotonic()<deadline:
                time.sleep(.05)
        # run_streaming retains its ordinary bounded owned-tree fallback if
        # cooperative cleanup cannot terminate the parent; uncertainty stays
        # in the normal parent journals/quarantines, never an implicit retry.
    try:
        result=run_streaming(command,cwd=item['context']['controller_root'],
            env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},label='normal-energy-parent',
            cancel_event=cancel,process_registry=registry,before_terminate=cooperative_cleanup)
        if cancel.is_set():raise RuntimeError('normal energy parent cancelled; preserve cleanup/source evidence')
        return result
    finally:
        for signum,handler in handlers.items():signal.signal(signum,handler)


def _timeout_recovery_authorization(item, *, started_ns, returncode):
    """Record one standing-authorized recovery of an exact terminal timeout."""
    generic=item['config']['schema']==GENERIC_SCHEMA
    if item['config']['schema']not in (GENERIC_SCHEMA,NATIVE_SCHEMA)or not returncode:return None
    from . import energy_resume
    from onnx_splitpoint_tool.process_control import _proc_start_time
    from onnx_splitpoint_tool.energy.task_budget import _source_reentry, _reentry_history, source_completion, final_cohort_authorization
    run=item['run'];attempt=latest_attempt(run)
    if attempt is None:return None
    terminal=read(attempt/'cleanup-status.json')
    if terminal.get('cleanup_proven') is True:return None
    start=read(attempt/'production-start.json');finished=read(attempt/'production-result.json')
    if ((attempt/'production-start.json').stat().st_mtime_ns<started_ns
            or finished.get('returncode')!=returncode
            or (generic and start.get('selected_identity')!=item['config']['selected_identity'])
            or start.get('no_deployment_or_build') is not True):
        raise ValueError('timeout recovery requires this exact new failed normal Parent')
    directory,request,reply,_=energy_resume.predecessor(run,attempt)
    if (request.get('phase')!='FINALIZING' or request.get('process_started') is not True
            or reply.get('reason')!='campaign_source_completion_unresolved' or reply.get('recovery')
            or not isinstance(request.get('token'),str)or len(request['token'])!=32
            or any(c not in '0123456789abcdef'for c in request['token'])):
        raise ValueError('automatic recovery is restricted to an unrecovered FINALIZING source timeout')
    if generic:
        binding=read(attempt/'worker-plan.json')if attempt!=run else item['config']['normal_energy_plan']
        plan_path=Path(binding['path']);plan=read(plan_path)
        if (sha(plan_path)!=binding['sha256'] or plan.get('identity')!=item['config']['selected_identity']
                or (attempt!=run and plan_path!=attempt/'measurement/generic_energy_plan.json')
                or plan.get('campaign_budget_file')!=str(item['context']['budget'])):
            raise ValueError('timeout recovery normal plan or shared budget differs')
        source=plan['campaign_source_id'];row=plan['campaign_row_id'];limits=plan['campaign_limits'];cohort=''
        identity=item['config']['selected_identity']
        expected_row=json.dumps(['generic_completed:'+identity['comparison_backend'],
            identity['model_id'],identity['case_id'],identity['setup_id']],separators=(',',':'))
        scope=[row];setup=identity['setup_id']
    else:
        config=item['config'];setup=request['setup_id'];cohort=config.get('campaign_cohort')
        ids=config.get('capture_identities')or[]
        scope=[json.dumps(value,separators=(',',':'))for value in ids]
        if (not cohort or config.get('campaign_budget_file')!=str(item['context']['budget'])
                or config.get('capture_limit')!=len(ids)or not ids or len(set(scope))!=len(ids)
                or any(len(value)!=4 or value[3]!=setup for value in ids)):
            raise ValueError('Native timeout requires the exact stored final capture prefix/cohort')
        for name in ('summary_source','summary','summary_snapshot','registry_source','registry','validation_summary','audited_scope'):
            path=Path(config[name])
            if not path.is_file()or path.is_symlink()or sha(path)!=config[name+'_sha256']:
                raise ValueError('Native timeout original preparation source changed: '+name)
        target=config['targets'][setup]
        source=str(target['energy']['urecs_address']).strip().lower()
        budget=read(item['context']['budget'])
        matched_rows=[key for key,task in budget['tasks'].items()for chain in task['chains']
                      if chain.get('run_directory')==request['attempt_dir']]
        if len(matched_rows)!=1 or matched_rows[0]not in scope:
            raise ValueError('Native timeout chain is outside the exact prepared capture prefix')
        row=expected_row=matched_rows[0]
        argv=start.get('argv')or[]
        def option(flag):
            if argv.count(flag)!=1 or argv.index(flag)+1>=len(argv):
                raise ValueError('Native timeout original campaign option missing or ambiguous: '+flag)
            return argv[argv.index(flag)+1]
        if (option('--runs')!='3'or option('--campaign-cohort')!=cohort
                or option('--campaign-budget-file')!=str(item['context']['budget'])):
            raise ValueError('Native timeout original repeat/cohort/budget binding differs')
        retries=int(option('--campaign-max-retries'));failures=int(option('--campaign-max-transport-failures'))
        limits={'max_chains':3*(retries+1),'max_retries':retries,'max_transport_failures':failures}
        final=final_cohort_authorization(budget,cohort)
        if (final.get('parent_run_dir')!=str(run)or final.get('source_id')!=source
                or final.get('row_ids')!=sorted(scope)or final.get('repeats')!=3
                or final.get('purpose')!='final'or final.get('environment_observation_required')is not True):
            raise ValueError('Native timeout final cohort does not bind the unchanged prepared scope')
    cfg=read(directory/'controller.resources.json')
    resources=sorted(r for r in cfg['setups'][setup]
                     if r.startswith(('source:','dut:'))or r in {'controller:capture','controller:nic'})
    if (len(set(resources))!=len(resources)or len(resources)!=4
            or resources.count('source:'+source)!=1 or sum(r.startswith('dut:')for r in resources)!=1
            or not {'controller:capture','controller:nic'}.issubset(resources)
            or setup!=request['setup_id']or row!=expected_row
            or cfg.get('parent_pid')!=start.get('parent_pid')
            or (generic and cfg.get('parent_start_ticks')!=start.get('parent_start_ticks'))
            or terminal.get('cleanup_proven') is not False
            or terminal.get('owned_remote_active')!=0 or terminal.get('owned_local_active')!=0
            or terminal.get('errors')!=['stopped physical resources: '+repr(resources)]):
        raise ValueError('timeout recovery requires only the exact four stopped physical resources')
    for pid,ticks in ((cfg['parent_pid'],cfg['parent_start_ticks']),
                      (request['owner_pid'],request['owner_start_ticks']),
                      (request['collector_pid'],request['collector_start_ticks'])):
        if (type(pid)is not int or pid<=0 or not str(ticks or '').isdigit()
                or _proc_start_time(pid)==str(ticks)):
            raise ValueError('timeout recovery predecessor identity missing or still active')
    capture=Path(request['attempt_dir']);cleanup=read(capture/'collector_stdout.log.cleanup.json')
    summary=read(capture/'energy_summary.json');completion=source_completion(capture)
    calibration=summary.get('energy_calibration_verification')or{}
    close=completion['source_close'];reason=close[0][3]if len(close)==1 else ''
    if (cleanup.get('owned_tree_quiescent') is not True or cleanup.get('process_started') is not True
            or any(cleanup.get(k)!=request[k]for k in ('collector_pid','collector_start_ticks'))
            or summary.get('cancelled') is not False or summary.get('collector_started') is not True
            or summary.get('status')!='collector_failed'
            or summary.get('preflight_status')!='verified'
            or summary.get('final_energy_gate_reasons')!=['capture_release_failed']
            or summary.get('environment_observation_error')not in (None,'required_capture_environment_missing')
            or (summary.get('workload_timing')or{}).get('rc')not in (None,0)
            or calibration.get('runtime_binding_errors')
            or (calibration.get('runtime_fail_closed_required')is True and calibration.get('verified')is not True)
            or completion['verified']or len(completion['go'])!=1 or completion['protocol_end']
            or len(close)!=1 or close[0][2]!='false' or close[0][1]!=completion['go'][0][1]
            or 'silence timeout'not in reason
            or not any(marker in reason for marker in ('receive socket error','ReceiveSocketTimeout'))):
        raise ValueError('automatic recovery requires the original quiescent receive-silence timeout')
    budget=read(item['context']['budget'])if generic else budget;grant=_source_reentry(budget,source,{})
    task=budget['tasks'].get(row)or{};chains=task.get('chains')or[]
    matched=[chain for chain in chains if chain.get('run_directory')==str(capture)]
    if (not grant or grant.get('continuous_campaign') is not True or row not in grant.get('row_ids',[])
            or grant.get('source_stop_reason')!=reply['reason']or grant.get('task_stop_reasons')!={row:reply['reason']}
            or not set(scope).issubset(grant.get('row_ids',[]))
            or task.get('source_id')!=source or task.get('limits')!=limits
            or len(matched)!=1):
        raise ValueError('timeout recovery lacks this current continuous task/source authority')
    chain=matched[0];logical=chain.get('logical_repeat')
    if (logical not in {'repeat:0','repeat:1','repeat:2'} or (chain.get('cohort_id')or'')!=cohort
            or chain.get('finished') is not True or chain.get('valid') is not False
            or chain.get('collector_started') is not True or chain.get('workload_started') is not True
            or chain.get('source_completion_verified') is not False
            or chain.get('transport_reasons')!=['transport_log_error']
            or any(chain.get(k)!=request[k]for k in ('owner_pid','owner_start_ticks'))):
        raise ValueError('timeout recovery differs from the original invalid physical chain')
    accepted=[c for c in chains if c.get('valid') is True and (c.get('cohort_id')or'')==cohort]
    slots={c.get('logical_repeat')for c in accepted}
    if (logical in slots or len(slots)!=len(accepted)or not slots<{'repeat:0','repeat:1','repeat:2'}
            or any(c.get('finished')is not True or c.get('source_completion_verified')is not True
                   or not Path(c['run_directory']).is_relative_to(run)for c in accepted)):
        raise ValueError('timeout recovery must preserve accepted slots and target only missing repeats')
    history=list(_reentry_history(grant))
    unresolved=[c for t in budget['tasks'].values()if t.get('source_id')==source for c in t['chains']
                if (not c.get('finished')or(c.get('collector_started')and c.get('source_completion_verified')is not True))
                and not any(c==old['chain']for old in history)]
    if unresolved!=[chain]:raise ValueError('another unresolved source chain blocks automatic recovery')
    remaining=[]
    for key in scope:
        scoped=budget['tasks'].get(key)or{}
        valid=[c for c in scoped.get('chains',[])if c.get('valid')is True and (c.get('cohort_id')or'')==cohort]
        valid_slots={c.get('logical_repeat')for c in valid}
        if (scoped.get('source_id')!=source or scoped.get('limits')!=limits
                or len(valid_slots)!=len(valid)or not valid_slots.issubset({'repeat:0','repeat:1','repeat:2'})
                or any(c.get('finished')is not True or c.get('source_completion_verified')is not True
                       or not Path(c['run_directory']).is_relative_to(run)for c in valid)):
            raise ValueError('timeout recovery prepared cohort has invalid accepted slot bindings')
        remaining.append({'identity':json.loads(key),'missing_valid_repeats':3-len(valid_slots)})
    import yaml
    output=run/'attempts'/('energy-recovery-'+uuid.uuid4().hex);output.mkdir(parents=True,exist_ok=False)
    profile={'schema':'revision3/bounded-source-recovery-profile','run_dir':str(run),'failed_attempt':str(attempt),
        'historical_operation':request['operation_id'],'source':source,'setup_id':setup,'regime':'generic_completed'if generic else'native',
        'selected_remaining_points':remaining,
        'duration_s':60,'same_original_budget':str(item['context']['budget']),'unchanged_limits':task['limits'],
        'source_completion_remains_unproven':True,'authorization_basis':grant['authorization_file'],
        'authorization_basis_sha256':grant['authorization_sha256'],'no_device_reset':True,'no_build_or_deployment':True}
    if not generic:profile['campaign_cohort']=cohort
    profile_path=output/'one_workflow.yaml';profile_path.write_text(yaml.safe_dump(profile,sort_keys=False))
    authorization=output/'AUTHORIZATION.json'
    write_json(authorization,{'source':'source:'+source,'operator':'Normal product applying the bound continuous user authorization',
        'authorized_at':datetime.now(timezone.utc).isoformat(),
        'reason':'Exact terminal receive-silence timeout '+request['operation_id']+'; standing authority '+
            grant['authorization_file']+' sha256='+grant['authorization_sha256']+'; preserve source-unclosed history and accepted slots',
        'new_workflow_profile':str(profile_path),'authorize_new_attempt':True,'scope':'one_new_workflow',
        'source_completion_unproven':True})
    return attempt,authorization


def _recover_parent(item, authorization, *, expected_previous=None):
    from .energy_resume import recover
    previous=item['previous']
    if expected_previous is not None and (previous!=expected_previous or latest_attempt(item['run'])!=expected_previous):
        raise ValueError('timeout predecessor changed before normal recovery')
    recover(SimpleNamespace(run_dir=str(item['run']),failed_attempt=str(previous),authorization=str(authorization)))
    from . import generic_energy_parent,native_energy_parent
    parent=generic_energy_parent if item['config']['schema']==GENERIC_SCHEMA else native_energy_parent
    parent.configure(item['run'])
    parent.refresh_controller(SimpleNamespace(run_dir=str(item['run']),previous_attempt=str(previous)))
    return describe_parent(item['run'])

def run_parent(run, *, plan_only=False, output_dir=None, import_only=False, resume=False,
               recovery_authorization=None):
    item=describe_parent(run);context=item['context'];previous=item['previous']
    if plan_only:
        if output_dir is None or Path(output_dir).resolve().is_relative_to(item['run']):
            raise ValueError('energy parent preview requires an external output directory')
        output=Path(output_dir).resolve();output.mkdir(parents=True,exist_ok=True)
        for p in (item['path'],item['run']/'PREPARED.json'):shutil.copy2(p,output/p.name)
        result={'status':item['state'],'parent':str(item['run']),'identities':item['group']['identities'],
                'regime':item['group'].get('regime','native'),'setup':item['group']['setup'],
                'queue':str(item['queue']),'queue_sha256':sha(item['queue']),
                'prepared_plan_sha256':sha(item['path']),'import_command':item['importer'],
                'hardware_started':False,'budget_changed':False}
        print(json.dumps(result,indent=2));return 0
    if import_only or (item['state']=='IMPORT_PENDING' and not resume):
        result=_import_saved(item);print(json.dumps(result,indent=2));return 0
    if previous is not None and not resume:
        raise ValueError('started parent requires --resume or --import-only; never an implicit rerun')
    if previous is None and resume:raise ValueError('resume requires its actual previous parent')
    if recovery_authorization is not None:
        # The exact recovered STOP/handoff makes a regular metadata refresh
        # admissible. Keep the original stopped plan until recovery completes.
        item=_recover_parent(item,recovery_authorization);context=item['context'];previous=item['previous']
    handoff=context['report_root']/'operations/token_saving_batch_20261009'/(item['run'].name+'_handoff.json')
    command=_parent_command(item['group'],item['config'],parent=item['run'],handoff=handoff,previous=previous)
    if command.count('--run-dir')!=1 or command[command.index('--run-dir')+1]!=str(item['run']):
        raise ValueError('stored normal executor differs from selected parent')
    write_json(handoff,{'parent':str(item['run']),'group':{'parent':str(item['run']),
        'regime':item['group'].get('regime','native'),'setup':item['group']['setup']},
        'authorization_file':str(context['report_root']/'operations/token_saving_batch_20261009/AUFTRAG_AKTUELL.txt'),
        'executor':command,'importer':item['importer'],'decision':'normal product entry; existing live admission and resource claims'})
    while True:
        started_ns=time.time_ns()
        completed=_run_parent_command(item,command)
        if completed.returncode==0:break
        # The existing coordinator bounds each series to three fresh resumes.
        # Only its exact, finished, source-closed pure-loss receipt permits the
        # next series under the unchanged standing sample-loss authorization.
        timeout=_timeout_recovery_authorization(item,started_ns=started_ns,returncode=completed.returncode)
        if timeout is None:
            if item['config']['schema']!=GENERIC_SCHEMA:
                raise RuntimeError('normal energy parent stopped; original attempts preserved')
            from . import generic_energy_parent
            generic_energy_parent.configure(item['run'])
            previous=generic_energy_parent._sample_loss_retry_attempt(item['run'],
                started_ns=started_ns,returncode=completed.returncode)
        else:
            failed,authorization=timeout
            item=_recover_parent(describe_parent(item['run']),authorization,expected_previous=failed)
            context=item['context'];previous=item['previous']
        from onnx_splitpoint_tool.energy.collector import _collector_reconnect_backoff
        pause=_collector_reconnect_backoff(5.0)
        if pause.get('cancelled'):raise RuntimeError('normal energy continuation cancelled before resume')
        item=describe_parent(item['run'])
        command=_parent_command(item['group'],item['config'],parent=item['run'],handoff=handoff,previous=previous)
        if command.count('--run-dir')!=1 or command[command.index('--run-dir')+1]!=str(item['run']):
            raise ValueError('stored normal resume targets another parent')
        write_json(handoff,{'parent':str(item['run']),'group':{'parent':str(item['run']),
            'regime':item['group'].get('regime','native'),'setup':item['group']['setup']},
            'authorization_file':str(context['report_root']/'operations/token_saving_batch_20261009/AUFTRAG_AKTUELL.txt'),
            'executor':command,'importer':item['importer'],
            'decision':('existing source-closed pure-loss authority; normal finite-series resume'if timeout is None else
                'exact standing-authorized receive-timeout recovery; old source completion remains unverified; normal resume')})
    result=_import_saved(describe_parent(item['run']))
    print(json.dumps(result,indent=2));return 0
