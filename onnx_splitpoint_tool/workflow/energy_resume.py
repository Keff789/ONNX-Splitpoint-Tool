"""Explicit bounded-parent recovery and serial repeat continuation.

All device work goes through the saved product CLI and ControllerResourceBroker.
Historical attempts and campaign limits remain immutable. Machine-verified accepted repeats are reused before serial continuation.
"""
from contextlib import ExitStack
from pathlib import Path
import hashlib
import json
import os
import shlex
import signal
import threading
import time
import uuid

WORKTREE = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    from onnx_splitpoint_tool.benchmark.schema import write_json_atomic
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(Path(path), data)


def now():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def budget_path(run):
    run=Path(run)
    config=read(run/('energy_parent_plan.json' if (run/'energy_parent_plan.json').exists() else 'parent_plan.json'))
    path=Path(config['campaign_budget_file']).resolve()
    root=path.parent.parent
    if path.name!='energy_task_budget.json' or path.parent.name!='energy' or not run.resolve().is_relative_to(root/'energy'):
        raise ValueError('shared campaign budget binding required')
    return path


def predecessor(run, attempt):
    from onnx_splitpoint_tool.remote.resource_recovery import _bounded_energy_journal
    start = read(attempt / 'production-start.json')
    directory, _, evidence = _bounded_energy_journal(run, start['session_id'], attempt)
    stops = [read(p) for p in directory.glob('*.resource-reply.json') if read(p).get('state') != 'RELEASED']
    if len(stops) != 1 or stops[0].get('state') != 'STOP':
        raise ValueError('exactly one historical source STOP required')
    reply = stops[0]
    request = read(directory / (reply['operation_id'] + '.resource-request.json'))
    generic = (run / 'parent_plan.json') in evidence
    if generic:
        config=evidence[run/'parent_plan.json']
        normal_path=(Path(read(attempt/'worker-plan.json')['path']) if attempt!=run else
                     Path(config['normal_energy_plan']['path']))
        capture_root=normal_path.parent
    else:capture_root=attempt
    if not generic and attempt.parent == run / 'resume_attempts':
        argv = start.get('argv')
        if not isinstance(argv,list) or any(not isinstance(arg,str) for arg in argv):
            raise ValueError('normal resumed parent argv missing')
        options = [i for i,arg in enumerate(argv) if arg == '--out-dir' or arg.startswith('--out-dir=')]
        if len(options) != 1:
            raise ValueError('normal resumed parent requires one exact output binding')
        index = options[0]
        output = (argv[index].split('=',1)[1] if argv[index].startswith('--out-dir=') else
                  argv[index+1] if index+1 < len(argv) and not argv[index+1].startswith('--') else '')
        expected = run / 'measurement/resume_attempts' / attempt.name
        if not output or Path(output) != expected or Path(output).resolve() != expected:
            raise ValueError('normal resumed parent output differs from its exact measurement attempt')
        capture_root = expected
    if (request.get('session_id') != start['session_id'] or request.get('run_id') != run.name
            or any(reply.get(k) != request.get(k) for k in ('run_id','session_id','operation_id','token'))
            or not Path(request['attempt_dir']).resolve().is_relative_to(capture_root)):
        raise ValueError('source STOP is not bound to the selected parent attempt')
    return directory, request, reply, evidence


def released_predecessor(run, attempt, *, require_budget_handoff=True):
    """Read the exact normal release without relabeling the old cleanup/result."""
    run, attempt = Path(run).resolve(), Path(attempt).resolve()
    directory, request, reply, evidence = predecessor(run, attempt)
    recovery = reply.get('recovery') or {}; resources = recovery.get('resources') or []
    identity = {key: request[key] for key in ('run_id', 'session_id', 'operation_id')}
    cleanup = read(attempt / 'cleanup-status.json')
    if (recovery.get('status') != 'released' or recovery.get('old_attempt_validated') is not False
            or recovery.get('measurements_started') != 0
            or (recovery.get('ownership_observation') or {}).get('ok') is not True
            or len(resources) != len(set(resources))
            or len(resources) != 4 or {'controller:capture','controller:nic'} - set(resources)
            or sum(r.startswith('source:') for r in resources) != 1
            or sum(r.startswith('dut:') for r in resources) != 1
            or any(recovery.get(k) != v for k,v in identity.items())
            or cleanup.get('cleanup_proven') is not False
            or cleanup.get('errors') != ['stopped physical resources: ' + repr(resources)]
            or list(directory.glob('*.remote-lease.json'))):
        raise ValueError('exact normal recovered source-only predecessor required')
    if require_budget_handoff:
        from onnx_splitpoint_tool.energy.task_budget import campaign_budget_stop_reason
        budget = read(budget_path(run)); source = next(r[7:] for r in resources if r.startswith('source:'))
        grant = budget['sources'][source].get('continuous_admission') or {}
        config = evidence.get(run/'parent_plan.json')
        if config is not None:
            normal_path = (Path(read(attempt/'worker-plan.json')['path']) if attempt != run else
                           Path(config['normal_energy_plan']['path']))
            normal = evidence[normal_path]
            if normal['campaign_source_id'] != source:
                raise ValueError('recovered Generic source differs from its normal plan')
            row_ids = [normal['campaign_row_id']]
        else:
            config = evidence[run/'energy_parent_plan.json']
            row_ids = [json.dumps(key,separators=(',',':')) for key in config['capture_identities']]
        if (grant.get('capture_request') != request or grant.get('recovery_reply') != reply
                or any(campaign_budget_stop_reason(budget,row,
                    source_id=source,parent_run_dir=run) for row in row_ids)):
            raise ValueError('exact recovered continuous-source handoff required')
    return directory, request, reply, evidence


def recover(args):
    from onnx_splitpoint_tool.remote.resource_recovery import recover_capture_resources
    from onnx_splitpoint_tool.workflow.run_control import EvaluationRunLock
    run = Path(args.run_dir).resolve()
    lock = EvaluationRunLock(out_root=run.parent, run_dir=run, owner={})
    lock.acquire_for_recovery()
    try:
        _, request, reply, _ = predecessor(run, Path(args.failed_attempt).resolve())
        if request.get("phase")=="ACQUIRING" and reply.get("reason")=="controller_resource_owner_lost":
            from onnx_splitpoint_tool.energy.recovery_accounting import reconcile_stopped_capture
            reconcile_stopped_capture(budget_path(run),parent_attempt=Path(args.failed_attempt).resolve(),
                operation_id=request["operation_id"],run_dir=run)
        result = recover_capture_resources(run_dir=run, session_id=request['session_id'],
            operation_id=request['operation_id'], setup_id=request['setup_id'],
            retry_authorization=read(args.authorization), bounded_energy_attempt_dir=Path(args.failed_attempt).resolve())
        source = next(r[7:] for r in result['resources'] if r.startswith('source:'))
        if read(budget_path(run))['sources'][source].get('continuous_admission'):
            session = 'energy-recovery-' + uuid.uuid4().hex
            output = run/'attempts'/session; output.mkdir(parents=True,exist_ok=False)
            recover_run_fence(run,output,session,lock,read(lock.quarantine_path),Path(args.failed_attempt).resolve())
            result['source_budget_handoff'] = str(output/'source-budget-handoff.json')
        print(json.dumps(result), flush=True)
        return result
    finally:
        lock.release(write_released_state=False)


def reconcile(args):
    from onnx_splitpoint_tool.energy.recovery_accounting import reconcile_stopped_capture
    from onnx_splitpoint_tool.workflow.run_control import EvaluationRunLock
    run = Path(args.run_dir).resolve()
    attempt = Path(args.failed_attempt).resolve()
    lock = EvaluationRunLock(out_root=run.parent, run_dir=run, owner={})
    lock.acquire_for_recovery()
    try:
        _, request, _, _ = predecessor(run, attempt)
        receipt = reconcile_stopped_capture(budget_path(run),
            parent_attempt=attempt, operation_id=request['operation_id'],run_dir=run)
        print(json.dumps(receipt), flush=True)
        return receipt
    finally:
        lock.release(write_released_state=False)


def released_recovery_owner(lease, resource, identity, resources, recovery):
    """Allow a subsequently released shared controller while its flock is held."""
    from datetime import datetime
    import socket
    from onnx_splitpoint_tool.process_control import _proc_start_time
    current = lease.previous_owner
    if lease.quarantine_path.exists() or current.get('state') != 'released':
        raise ValueError('released physical ownership changed')
    binding = current.get('resource_recovery') or {}
    if all(binding.get(k) == v for k,v in identity.items()) and binding.get('resources') == resources:
        return
    if resource not in {'controller:capture','controller:nic'}:
        raise ValueError('released source/DUT recovery identity changed')
    if (current.get('physical_resource') != resource
            or current.get('hostname') != socket.gethostname()
            or any(not current.get(k) for k in ('run_id','session_id','operation_id','released_at','pid'))
            or any(bool(_proc_start_time(current[k])) for k in ('pid','owner_pid') if current.get(k))):
        raise ValueError('later shared controller release is not terminal')
    try:
        if datetime.fromisoformat(current['released_at']) <= datetime.fromisoformat(recovery['released_at']):
            raise ValueError('shared controller release predates recovery')
    except (KeyError, TypeError) as exc:
        raise ValueError('shared controller release time missing') from exc


def recover_run_fence(run, output, session, lock, fence, attempt):
    from onnx_splitpoint_tool.remote.resource_recovery import _rest_evidence_unchanged
    from onnx_splitpoint_tool.workflow.run_control import EvaluationRunLock
    from onnx_splitpoint_tool.energy.task_budget import install_authorized_source_reentry
    from onnx_splitpoint_tool.process_control import _proc_start_time
    directory, request, reply, evidence = predecessor(run, attempt)
    cfg = read(directory / 'controller.resources.json')
    cleanup = read(attempt / 'cleanup-status.json')
    owner = lock.previous_owner
    identity = {key: request[key] for key in ('run_id', 'session_id', 'operation_id')}
    generic = (run/'parent_plan.json') in evidence
    reason = 'generic_energy_cleanup_unproven' if generic else 'yolov7_energy_cleanup_unproven'
    purpose = 'revision3_generic_completed_energy' if generic else 'yolov7_full_fix_energy'
    if (fence.get('schema') != 'onnx-splitpoint/evaluation-run-lock-quarantine'
            or fence.get('run_dir') != str(run) or fence.get('run_id') != run.name
            or fence.get('owner') != {'reason':reason, 'pending':0, 'errors':cleanup['errors']}
            or owner.get('purpose') != purpose
            or owner.get('pid') != cfg['parent_pid']
            or any(owner.get(k) != identity[k] for k in ('run_id','session_id'))):
        raise ValueError('run fence is not bound to the exact terminal energy parent')
    recovery = reply.get('recovery') or {}
    resources = recovery.get('resources') or []
    if (recovery.get('status') != 'released' or len(resources) != 4
            or any(recovery.get(k) != v for k,v in identity.items())
            or list(directory.glob('*.remote-lease.json'))):
        raise ValueError('physical recovery incomplete')
    for pid,ticks in ((cfg['parent_pid'],cfg['parent_start_ticks']),
                      (request['owner_pid'],request['owner_start_ticks']),
                      (request['collector_pid'],request['collector_start_ticks'])):
        if _proc_start_time(pid) == ticks:
            raise ValueError('historical owner still active')
    with ExitStack() as held:
        for resource in resources:
            lease = EvaluationRunLock.for_resource(resource, owner={})
            lease.acquire_for_recovery()
            held.callback(lease.release, write_released_state=False)
            released_recovery_owner(lease,resource,identity,resources,recovery)
        if not _rest_evidence_unchanged(evidence):
            raise ValueError('historical terminal evidence changed')
        grant = install_authorized_source_reentry(budget_path(run),
            capture_request=request, recovery_reply=reply,
            new_parent={'run_id':run.name,'session_id':session,'pid':os.getpid(),
                        'start_ticks':_proc_start_time(os.getpid())})
        if grant.get('continuous_campaign'):
            from onnx_splitpoint_tool.energy.task_budget import authorize_campaign_source_continuation
            source = next(r[7:] for r in resources if r.startswith('source:'))
            grant = authorize_campaign_source_continuation(budget_path(run),source_id=source,
                authorization_file=grant['authorization_file'],row_ids=grant['row_ids'])
        write(output/'source-budget-handoff.json',grant)
        write(output/'runlock-recovery.json',{'original_fence':fence,'original_owner':owner,
            'physical_recovery_reply':reply,'at':now()})
    return True


def repeat_command(run, output, index):
    plan = read(run/'measurement/plan/native_producer_energy_plan.json')
    if len(plan['rows']) != 1:
        raise ValueError('bounded continuation requires exactly one audited operating point')
    row = plan['rows'][0]
    config=read(run/'energy_parent_plan.json')
    identity=[row[k] for k in ('backend','model','case','setup_id')]
    if config['expected_identities'] != [identity]:
        raise ValueError('continuation differs from originally admitted operating point')
    command = shlex.split(row['measure_command'])
    expected = {'--runs':'3','--duration':'60','--campaign-repeats':'3',
        '--campaign-max-retries':'2','--campaign-max-transport-failures':'20',
        '--campaign-budget-file':str(budget_path(run))}
    for option,value in expected.items():
        if command.count(option)!=1 or command[command.index(option)+1] != value:
            raise ValueError('frozen measurement command drift: '+option)
    if index not in range(3):
        raise ValueError('repeat index out of range')
    command[command.index('--runs')+1] = '1'
    command[command.index('--out')+1] = str(output / f'repeat_{index}')
    # Scheduling one acquisition is independent of the unchanged campaign retry
    # budget. This first continuation must never launch an automatic second one.
    command[command.index('--invalid-repeat-max-retries')+1] = '0'
    command += ['--exact-run-count','--task-logical-repeat',f'repeat:{index}',
                '--reuse-campaign-repeats']
    return command, row


def continue_clean_parent(run, output, session, previous):
    from onnx_splitpoint_tool.energy.task_budget import continue_authorized_source_reentry
    from onnx_splitpoint_tool.workflow.run_control import EvaluationRunLock
    from onnx_splitpoint_tool.process_control import _proc_start_time
    config = read(Path(previous)/'remote-process-leases/controller.resources.json')
    resources = sorted({r for values in config['setups'].values() for r in values
                        if r.startswith(('dut:','source:')) or r in {'controller:capture','controller:nic'}})
    if len(resources)!=4:
        raise ValueError('clean continuation needs the exact four resources')
    with ExitStack() as held:
        for resource in resources:
            lease=EvaluationRunLock.for_resource(resource,owner={})
            lease.acquire_for_recovery();held.callback(lease.release,write_released_state=False)
            if lease.quarantine_path.exists() or lease.previous_owner.get('state')!='released':
                raise ValueError('physical resource not cleanly released')
        grant=continue_authorized_source_reentry(budget_path(run),
            previous_attempt_dir=previous,new_parent={'run_id':run.name,'session_id':session,
            'pid':os.getpid(),'start_ticks':_proc_start_time(os.getpid())})
        write(output/'source-budget-handoff.json',grant)


def verify_repeat(run, row, index):
    from onnx_splitpoint_tool.energy.task_budget import read_verified_campaign_repeat
    budget = read(budget_path(run))
    key = json.dumps([row[k] for k in ('backend','model','case','setup_id')],separators=(',',':'))
    cohort=read(run/'energy_parent_plan.json').get('campaign_cohort')or''
    chains = [c for c in budget['tasks'][key]['chains'] if c['logical_repeat']==f'repeat:{index}' and c.get('valid') and (c.get('cohort_id')or'')==cohort]
    if len(chains)!=1:
        raise ValueError('repeat has no unique accepted campaign evidence')
    return read_verified_campaign_repeat(chains[0],command=Path(row['command_file']).read_text().strip(),
        command_contract_sha256=row['successful_command_contract_sha256'],duration_s=60,require_ab=True)


def accounting_overlay(output, tool):
    """Use the ordinary frozen CLI, changing only controller error accounting.

    The original package still supplies calibration, calculations and runtime.
    Snapshot the two corrected modules per parent; do not deploy the Full fix
    or change the frozen controller installation for a before measurement.
    """
    import shutil
    directory = output/'controller-accounting'
    directory.mkdir()
    hashes = {}
    for name in ('collector.py','task_budget.py'):
        source = WORKTREE/'onnx_splitpoint_tool/energy'/name
        dest = directory/name
        shutil.copyfile(source,dest)
        hashes[str(dest)] = sha(dest)
    launcher = directory/'energy_cli.py'
    launcher.write_text('import sys, runpy, importlib.util\n'
        + f'sys.path.insert(0, {str(tool)!r})\n'
        + 'import onnx_splitpoint_tool as package\n'
        + f'spec = importlib.util.spec_from_file_location("onnx_splitpoint_tool.energy", {str(tool/"onnx_splitpoint_tool/energy/__init__.py")!r}, submodule_search_locations={[str(directory),str(tool/"onnx_splitpoint_tool/energy")]!r})\n'
        + 'energy = importlib.util.module_from_spec(spec)\n'
        + 'sys.modules[spec.name] = energy\n'
        + 'spec.loader.exec_module(energy)\n'
        + 'package.energy = energy\n'
        + f'sys.argv[0] = {str(tool/"scripts/energy_measurement_cli.py")!r}\n'
        + 'runpy.run_path(sys.argv[0], run_name="__main__")\n')
    hashes[str(launcher)] = sha(launcher)
    write(output/'controller-accounting.json',{'original_cli':str(tool/'scripts/energy_measurement_cli.py'),
        'original_cli_sha256':sha(tool/'scripts/energy_measurement_cli.py'),'overrides_sha256':hashes,
        'measurement_runtime_deployed':False})
    return launcher, hashes


def bind_accounting_cli(command, launcher, tool):
    original = str(tool/'scripts/energy_measurement_cli.py')
    if command.count(original) != 1:
        raise ValueError('unexpected frozen CLI path')
    index = command.index(original)
    if index < 1 or any(flag not in {'-B','-u'} for flag in command[1:index]):
        raise ValueError('unexpected frozen Python invocation')
    result = list(command)
    result[index] = str(launcher)
    return result


def resume(args, parent):
    from onnx_splitpoint_tool.native_progress import run_streaming
    from onnx_splitpoint_tool.process_control import ProcessTreeRegistry
    from onnx_splitpoint_tool.remote.process_lease import ControllerResourceBroker, RemoteProcessLeaseRegistry, RemoteProcessLeaseScope
    from onnx_splitpoint_tool.quality_service import ResourcePauseGate
    from onnx_splitpoint_tool.workflow.run_control import EvaluationRunLock
    from onnx_splitpoint_tool.workflow.generic_completed_task import observe_owners
    run = Path(args.run_dir).resolve()
    if bool(args.previous_attempt) == bool(args.failed_attempt):
        raise ValueError('select exactly one clean or failed predecessor')
    config = read(run/'energy_parent_plan.json')
    if read(run/'PREPARED.json')['plan_sha256'] != sha(run/'energy_parent_plan.json'):
        raise ValueError('original prepared plan drift')
    for field in ('summary','registry','validation_summary'):
        if config.get(field) and sha(config[field]) != config[field+'_sha256']:
            raise ValueError('frozen input changed: '+field)
    if sha(config['registry_source']) != config['registry_source_sha256']:
        raise ValueError('live registry changed')
    for name,expected in config['controller_source_sha256'].items():
        if sha(parent.TOOL/name)!=expected:
            raise ValueError('original controller source changed: '+name)
    for check in config['fresh_before_repeat_checks']:
        if sha(check['path'])!=check['sha256']:
            raise ValueError('valid performance evidence changed')
    session = 'energy-resume-'+uuid.uuid4().hex
    output = run/'attempts'/session
    output.mkdir(parents=True,exist_ok=False)
    write(output/'resume-binding.json',{'at':now(),'original_plan_sha256':sha(run/'energy_parent_plan.json'),
        'adapter_sha256':sha(__file__),'entrypoint_sha256':sha(parent.ADAPTER_PATH),
        'recovery_source_sha256':sha(WORKTREE/'onnx_splitpoint_tool/remote/resource_recovery.py'),
        'failed_attempt':args.failed_attempt,'slot_handoff':args.slot_handoff})
    lock = EvaluationRunLock(out_root=run.parent,run_dir=run,
        owner={'run_id':run.name,'session_id':session,'purpose':'yolov7_full_fix_energy'})
    if args.previous_attempt:
        lock.acquire()
    else:
        lock.acquire(recover_quarantine=lambda fence:recover_run_fence(run,output,session,lock,fence,Path(args.failed_attempt).resolve()))
    cancel,stop = threading.Event(),threading.Event()
    registry,processes = RemoteProcessLeaseRegistry(),ProcessTreeRegistry()
    broker=thread=None; handlers={}; errors=[]; primary=None; rc=1
    try:
        if args.previous_attempt:
            continue_clean_parent(run,output,session,Path(args.previous_attempt).resolve())
        if not (output/'source-budget-handoff.json').is_file():
            raise ValueError('normal explicit budget handoff missing')
        launcher, controller_hashes = accounting_overlay(output,parent.TOOL)
        observe_owners(config['targets'],output/'owner_preflight')
        journal=registry.configure_journal(scope=RemoteProcessLeaseScope(run.name,session),journal_dir=output/'remote-process-leases')
        original=read(run/'remote-process-leases/controller.resources.json')
        broker=ControllerResourceBroker(journal,ResourcePauseGate(),run_dir=run,setups=original['setups'])
        env={**os.environ,**registry.journal_environment(),'ONNX_SPLITPOINT_CONTROLLER_RESOURCES':'required','PYTHONDONTWRITEBYTECODE':'1'}
        def pump():
            while not stop.wait(.025):
                try:
                    broker.tick()
                    # Let the collector finish its normal error accounting.
                    # STOP prevents any subsequent capture in the broker itself.
                except BaseException as exc:
                    errors.append(str(exc));cancel.set();return
        thread=threading.Thread(target=pump,daemon=True);thread.start()
        handlers={s:signal.signal(s,lambda *_:cancel.set()) for s in (signal.SIGINT,signal.SIGTERM)}
        write(output/'production-start.json',{'started_at':now(),'run_id':run.name,'session_id':session,
            'parent_pid':os.getpid(),'slot_handoff':args.slot_handoff,'no_deployment_or_build':True})
        for index in range(3):
            command,row=repeat_command(run,output,index)
            command=bind_accounting_cli(command,launcher,parent.TOOL)
            write(output/f'repeat_{index}_command.json',{'argv':command})
            with (output/f'repeat_{index}.log').open('x') as log:
                def line(text):
                    log.write(text+'\n');log.flush()
                    if text.startswith('[native-'):
                        print(text,flush=True)
                result=run_streaming(command,cwd=parent.TOOL,env=env,timeout=1000,
                    label=f'revision3-repeat-{index}',line_callback=line,cancel_event=cancel,
                    should_cancel=lambda:cancel.is_set() or bool(errors) or registry.cancelled,
                    process_registry=processes,before_terminate=lambda:registry.cancel_all(grace_s=3))
            rc=result.returncode
            write(output/f'repeat_{index}_result.json',{'finished_at':now(),'returncode':rc,'stdout':result.stdout,'stderr':result.stderr})
            if rc or cancel.is_set() or broker.stopped_sources:
                raise RuntimeError(f'repeat {index} failed rc={rc}; stopped resources={sorted(broker.stopped_sources)}')
            if any(sha(path)!=expected for path,expected in controller_hashes.items()):
                raise ValueError('controller accounting changed during measurement')
            proof=verify_repeat(run,row,index)
            write(output/f'repeat_{index}_accepted.json',proof)
            print(json.dumps({'status':'REPEAT_VALID','repeat':index,'output':str(output)}),flush=True)
        rc=0
    except BaseException as exc:
        primary=exc;rc=rc or 1
        write(output/'failure.json',{'at':now(),'error':str(exc),'automatic_restart':False})
        raise
    finally:
        write(output/'production-result.json',{'finished_at':now(),'returncode':rc})
        try:
            if cancel.is_set() or registry.active_count():registry.cancel_all(grace_s=3)
            if processes.remaining_owned_count():processes.terminate_all(grace_s=3)
        except BaseException as exc:errors.append('owned cleanup: '+str(exc))
        stop.set()
        if thread:thread.join(timeout=5)
        try:
            if broker and (not thread or not thread.is_alive()):broker.close()
        except BaseException as exc:errors.append('broker cleanup: '+str(exc))
        if broker and broker.stopped_sources:errors.append('stopped physical resources: '+repr(sorted(broker.stopped_sources)))
        pending=registry.active_count()+processes.remaining_owned_count()
        unproven=bool(pending or errors or (thread and thread.is_alive()))
        if unproven:lock.commit_quarantine_fence({'reason':'yolov7_energy_cleanup_unproven','pending':pending,'errors':errors})
        for s,h in handlers.items():signal.signal(s,h)
        write(output/'cleanup-status.json',{'at':now(),'cleanup_proven':not unproven,
            'owned_remote_active':registry.active_count(),'owned_local_active':processes.remaining_owned_count(),
            'errors':errors,'foreign_processes_stopped':False})
        lock.release()
        if unproven and primary is None:raise RuntimeError('cleanup unproven')
