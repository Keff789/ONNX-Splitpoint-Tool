#!/usr/bin/env python3
"""Local sequential DFC/TensorFlow GPU smoke. No installation and no model builds."""
from __future__ import annotations
import argparse, contextlib, ctypes, errno, fcntl, json, os, re, signal, subprocess, sys, tempfile, time, zipfile
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from worker import REQUIRED

ROOT=Path(__file__).resolve().parent

def platform_interlock_path():
    # Same stable inode as workflow.run_control.platform_workflow_interlock_path.
    return Path.home()/'.onnx_splitpoint_tool/locks/workflow_platform_interlock.lock'

class BusyError(RuntimeError): pass

def write_json(p,data):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n');os.replace(tmp,p)

@contextlib.contextmanager
def interlock(path):
    path.parent.mkdir(parents=True,exist_ok=True)
    fd=os.open(path,os.O_CREAT|os.O_RDWR|os.O_CLOEXEC|os.O_NOFOLLOW,0o600)
    try:
        try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as e:raise BusyError('workflow_or_platform_operation_active') from e
        yield
    finally:os.close(fd)  # never delete or truncate the shared lock

def child_env(parent,venv,work,gpu):
    env=dict(parent)
    removed={k:env.pop(k) for k in ('PYTHONPATH','PYTHONHOME','VIRTUAL_ENV') if k in env}
    for sub in ('tmp','cuda_cache','xdg_cache'): (work/sub).mkdir(parents=True,exist_ok=True)
    changes={'CUDA_VISIBLE_DEVICES':gpu,'ONNX_SPLITPOINT_HAILO_ALLOW_GPU':'1',
        'VIRTUAL_ENV':str(venv),'PATH':str(venv/'bin')+os.pathsep+env.get('PATH',''),
        'PYTHONDONTWRITEBYTECODE':'1','PYTHONUNBUFFERED':'1','TMPDIR':str(work/'tmp'),
        'CUDA_CACHE_PATH':str(work/'cuda_cache'),'XDG_CACHE_HOME':str(work/'xdg_cache'),
        'TF_FORCE_GPU_ALLOW_GROWTH':'true','TF_CPP_MIN_LOG_LEVEL':'0'}
    env.update(changes)
    return env,{'child_only_overrides':changes,'removed_python_environment_keys':sorted(removed),
                'parent_cuda_visible_devices':parent.get('CUDA_VISIBLE_DEVICES'),
                'cuda_toolkit_not_changed':True,'parent_environment_modified':False}

def reap_group(pgid):
    while True:
        try:
            pid,_=os.waitpid(-pgid,os.WNOHANG)
            if pid<=0:break
        except ChildProcessError:break

def live_group(pgid):
    # /proc may be mounted in an ancestor PID namespace. Match our namespace
    # before comparing namespace-local IDs returned by Popen/os.killpg; raw
    # /proc/<pid>/stat PGIDs can otherwise falsely report an empty group.
    found=[]
    namespace=os.readlink('/proc/self/ns/pid')
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            if os.readlink(p/'ns/pid') != namespace:continue
            fields=dict(line.split(':',1) for line in (p/'status').read_text().splitlines() if ':' in line)
            group=int(fields['NSpgid'].split()[-1])
            pid=int(fields['NSpid'].split()[-1])
            state=fields['State'].strip().split()[0]
            if group==pgid and state not in ('Z','X'):found.append(pid)
        except (OSError,ValueError,IndexError):continue
    return found

def finish_group(proc,reason):
    proc.poll();ids=live_group(proc.pid)
    unexpected = reason == 'normal' and bool(ids)
    if ids or proc.returncode is None:
        try:os.killpg(proc.pid,signal.SIGTERM)
        except ProcessLookupError:pass
        until=time.monotonic()+5
        while time.monotonic()<until and live_group(proc.pid):
            time.sleep(.05);proc.poll()
            if proc.returncode is not None:reap_group(proc.pid)
        if live_group(proc.pid) or proc.poll() is None:
            try:os.killpg(proc.pid,signal.SIGKILL)
            except ProcessLookupError:pass
    try:proc.wait(timeout=5)
    except subprocess.TimeoutExpired:pass
    if proc.returncode is not None:reap_group(proc.pid)
    remain=live_group(proc.pid)
    return {'reason':reason,'cleanup_complete':proc.returncode is not None and not remain,'remaining_group_pids':remain,
            'unexpected_live_processes_after_worker_exit':unexpected}

def run_command(command,work,env,timeout,heartbeat=15):
    start=time.monotonic();proc=None;reason='normal';meta={}
    with (work/'console.log').open('wb') as log:
        try:
            proc=subprocess.Popen(command,cwd=work,env=env,stdout=log,stderr=subprocess.STDOUT,
                                  close_fds=True,start_new_session=True)
            nextbeat=start+heartbeat
            while proc.poll() is None:
                now=time.monotonic()
                if now-start>=timeout:reason='timeout';break
                if now>=nextbeat:
                    print(f'HEARTBEAT={work.name} elapsed_s={int(now-start)} limit_s={timeout}',flush=True);nextbeat=now+heartbeat
                time.sleep(.1)
        except BaseException:
            reason='interrupted'
            raise
        finally:
            if proc is not None:meta=finish_group(proc,reason)
    return {'returncode':proc.returncode,'elapsed_s':round(time.monotonic()-start,3),'timed_out':reason=='timeout',**meta}

def valid_pass(data,process, family):
    if not (isinstance(data,dict) and data.get('schema')=='hailo-gpu-compute-smoke-r1' and data.get('family')==family):return False
    if not (data.get('status')=='compute_pass' and process.get('returncode')==0 and not process.get('timed_out') and process.get('cleanup_complete') and not process.get('unexpected_live_processes_after_worker_exit')):return False
    passed={r.get('name') for r in data.get('checks',[]) if r.get('status')=='pass'}
    return set(REQUIRED)|{'dfc_sdk_import'} <= passed

def evidence_zip(output):
    target=output.with_suffix('.zip')
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        # Explicit files only; never archive CUDA caches, core dumps, models, or arbitrary directory trees.
        wanted=[output/'collection_summary.json']
        for family in ('hailo8','hailo10h'):
            wanted += [output/family/n for n in ('gpu_compute_result.json','process_result.json','environment_overrides.json','console.log','compiler_context.json')]
            wanted += sorted((output/family).glob('onnx_hailo_cuda_view_*_ptxas_invocations.jsonl'))
        wanted += [output/'source'/n for n in ('collect.py','worker.py','worker_context_v27934.py','run_hailo_gpu_smoke.sh','README.md')]
        for p in wanted:
            if p.is_file() and not p.is_symlink():z.write(p,p.relative_to(output))
    return target

def collect(args):
    if sys.platform!='linux':raise RuntimeError('Linux_required')
    try:
        # Reap vendor compiler helpers from the probe process group after a timeout.
        if ctypes.CDLL(None,use_errno=True).prctl(36,1,0,0,0)!=0:raise OSError(ctypes.get_errno(),'prctl')
    except Exception as e:raise RuntimeError('process_supervision_unavailable:'+str(e)) from e
    lock=platform_interlock_path()
    with interlock(lock.expanduser()):
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        args.output_parent.expanduser().mkdir(parents=True,exist_ok=True)
        output=Path(tempfile.mkdtemp(prefix='hailo_gpu_compute_r1_'+stamp+'_',dir=args.output_parent.expanduser())).resolve()
        summary={'schema':'hailo-gpu-smoke-collection-r1','status':'running','diagnostic_only':True,
          'model_compiler_invoked':False,'model_acceptance':'NOT_EVALUATED_BY_DIAGNOSTIC',
          'tool_configuration_modified':False,'package_installations':False,'sequential':True,'families':[],
          'expected_families':args.families,'scope':'SDK import and synthetic TensorFlow GPU/XLA computations; NOT a HEF build'}
        print('SMOKE_STARTED=YES\nEVIDENCE_DIRECTORY='+str(output),flush=True)
        print('MODE=synthetic_gpu_compute_only; no model build; no install; sequential',flush=True)
        (output/'source').mkdir()
        for n in ('collect.py','worker.py','worker_context_v27934.py','run_hailo_gpu_smoke.sh','README.md'):
            (output/'source'/n).write_bytes((ROOT/n).read_bytes())
        try:
            for family in args.families:
                venv=(args.venv_hailo8 if family=='hailo8' else args.venv_hailo10).expanduser().absolute()
                work=output/family;work.mkdir()
                row={'family':family,'venv':str(venv),'status':'running'};summary['families'].append(row)
                write_json(output/'collection_summary.json',summary)
                print('FAMILY_START='+family+' VENV='+str(venv),flush=True)
                python=venv/'bin/python'
                if not python.is_file() or not os.access(python,os.X_OK) or not (venv/'pyvenv.cfg').is_file():
                    row.update(status='setup_failed',error='expected_DFC_venv_missing');print('FAMILY_ERROR='+family+': expected_DFC_venv_missing',flush=True);continue
                env,changes=child_env(os.environ,venv,work,args.gpu)
                write_json(work/'environment_overrides.json',changes)
                command=[str(python),'-I','-B',str(ROOT/'worker.py'),'--output',str(work),'--expected-venv',str(venv),'--family',family]
                try:
                    with contextlib.ExitStack() as compiler_lifetime:
                        if getattr(args,'compiler_context',False):
                            # The ordinary product resolver and last pre-SDK hook
                            # also own this changed-entrypoint regression. The
                            # R2 worker's numerical checks remain byte unchanged.
                            source_root=ROOT.parents[1]
                            sys.path.insert(0,str(source_root))
                            from onnx_splitpoint_tool.hailo_compiler_context import resolve_hailo_compiler_context, compiler_child_environment
                            context=resolve_hailo_compiler_context(python,family,
                                job_override={'device':'gpu','gpu_selector':args.gpu},
                                parent_env=os.environ,work_dir=work)
                            # The common lifetime helper applies the resolved
                            # overlay to the actual child, including fresh
                            # validation, just as in the normal builder.
                            overlay_manifest=context.get('dependency_manifest')
                            if overlay_manifest:
                                if context.get('family')!='hailo8' or context.get('device')!='gpu':
                                    raise ValueError('hailo8_overlay_cannot_be_used_by_other_family')
                                changes['dependency_overlay']={'family':context['family'],
                                    'manifest_path':str(overlay_manifest)}
                            for setting in ('TF_NUM_INTEROP_THREADS','TF_NUM_INTRAOP_THREADS','TF_FORCE_GPU_ALLOW_GROWTH'):
                                if setting in os.environ:env[setting]=os.environ[setting]
                            env,effective=compiler_lifetime.enter_context(compiler_child_environment(context,parent_env=env,work_dir=work))
                            write_json(work/'compiler_context.json',effective)
                            worker_path=ROOT/'worker_context_v27934.py'
                            launch=("import sys,runpy;sys.path.insert(0,"+repr(str(source_root))+");"
                                "from onnx_splitpoint_tool.cuda_probe import auto_configure_cuda;"
                                "auto_configure_cuda();sys.argv="+repr([str(worker_path),'--output',str(work),'--expected-venv',str(venv),'--family',family])+";"
                                "runpy.run_path("+repr(str(worker_path))+",run_name='__main__')")
                            command=[str(python),'-I','-B','-c',launch]
                            changes['compiler_context']=effective
                            changes['context_policy']='v34 shared resolver and last pre-SDK auto_configure_cuda hook'
                            changes['child_only_overrides'].update({k:env[k] for k in ('CUDA_VISIBLE_DEVICES','CUDA_HOME','CUDA_PATH','XLA_FLAGS','PATH','LD_LIBRARY_PATH','TF_NUM_INTEROP_THREADS','TF_NUM_INTRAOP_THREADS','TF_FORCE_GPU_ALLOW_GROWTH') if k in env})
                            write_json(work/'environment_overrides.json',changes)
                        proc=run_command(command,work,env,args.timeout)
                    write_json(work/'process_result.json',proc);row['process']=proc
                    try:data=json.loads((work/'gpu_compute_result.json').read_text())
                    except (OSError,ValueError):data={}
                    row['status']='compute_pass' if valid_pass(data,proc,family) else 'failed'
                    row['worker_status']=data.get('status','no_final_worker_result');row['error']=data.get('error','')
                    row['failed_checks']=[{'name':r.get('name'),'error':r.get('error','')} for r in data.get('checks',[]) if r.get('status')!='pass']
                    if proc['timed_out']:row['error']='worker_timeout'
                    if not proc['cleanup_complete']:
                        row['error']='process_cleanup_incomplete';print('FAMILY_ERROR='+family+': '+row['error'],flush=True);break
                    if proc.get('unexpected_live_processes_after_worker_exit'):
                        row['error']='unexpected_live_processes_after_worker_exit'
                except Exception as exc:row.update(status='failed',error=f'{type(exc).__name__}: {exc}')
                print('FAMILY_STATUS='+family+':'+row['status'],flush=True)
                if row.get('error'):print('ERROR='+row['error'],flush=True)
                for r in row.get('failed_checks',[]):print('CHECK_ERROR='+str(r['name'])+': '+str(r['error']),flush=True)
            n=sum(r['status']=='compute_pass' for r in summary['families'])
            summary['status']='compute_pass' if n==len(args.families) else 'partial' if n else 'failed'
        except KeyboardInterrupt:summary['status']='interrupted'
        finally:
            write_json(output/'collection_summary.json',summary);zp=evidence_zip(output)
            print('GPU_SMOKE_STATUS='+summary['status'],flush=True)
            print('MODEL_BUILD=NOT_RUN\nMODEL_ACCEPTANCE=NOT_EVALUATED_BY_DIAGNOSTIC\nDIAGNOSTIC_ZIP='+str(zp),flush=True)
        return 0 if summary['status']=='compute_pass' else 2

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    h=Path.home()/'.onnx_splitpoint_tool/hailo'
    parser.add_argument('--venv-hailo8',type=Path,default=h/'venv_hailo8')
    parser.add_argument('--venv-hailo10',type=Path,default=h/'venv_hailo10')
    parser.add_argument('--families',nargs='+',choices=('hailo8','hailo10h'),default=['hailo8','hailo10h'])
    parser.add_argument('--gpu',default='0',help='GPU selection for child processes only, default 0')
    parser.add_argument('--timeout',type=int,default=180,help='Seconds per DFC venv, default 180')
    parser.add_argument('--compiler-context',action='store_true',help='Use v34 normal compiler resolver and unchanged R2 compute worker')
    parser.add_argument('--output-parent',type=Path,default=Path.home()/'Downloads')
    a=parser.parse_args()
    if not re.fullmatch(r'[0-9]+|GPU-[A-Za-z0-9-]+',a.gpu):parser.error('Select a single GPU index or UUID, not -1 or a device list')
    if not 10<=a.timeout<=180:parser.error('--timeout must be between 10 and 180')
    a.families=list(dict.fromkeys(a.families))
    try:return collect(a)
    except BusyError:
        print('SMOKE_STARTED=NO\nREASON=workflow_or_platform_operation_active\nClose active measurements/builds; never delete the lock file.',flush=True);return 3
    except Exception as exc:
        print('SMOKE_STARTED=NO\nERROR='+type(exc).__name__+': '+str(exc),flush=True);return 2
if __name__=='__main__':raise SystemExit(main())
