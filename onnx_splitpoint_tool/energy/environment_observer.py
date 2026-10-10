"""Low-rate existing host telemetry over the normal read-only SSH transport."""
from __future__ import annotations
import inspect
import json
import os
from pathlib import Path
import threading
import time


def _workers():
    """Self-contained DUT probe: real workload roots, descendants and threads."""
    import os
    import json
    from pathlib import Path
    roots = {"native_deepx_trt_e2e_from_benchmarkset.py", "native_full_baseline_eval_runner.py",
             "native_hailo_trt_fifo_from_benchmarkset.py", "native_hailo10_trt_e2e_from_benchmarkset.py"}
    processes = {}
    for p in Path('/proc').iterdir():
        if not p.name.isdecimal(): continue
        try:
            args = p.joinpath('cmdline').read_bytes().decode(errors='replace').strip('\0').split('\0')
            stat = p.joinpath('stat').read_text().rsplit(') ',1)[1].split()
            processes[int(p.name)] = (int(stat[1]), stat[19], args)
        except (OSError, ValueError, IndexError): continue
    selected = {}
    for pid, (ppid,ticks,args) in processes.items():
        generic = (any(Path(a).name == 'run_split_onnxruntime.py' for a in args)
                   and '--generic-completion-energy' in args)
        if not generic and (not any(Path(a).name in roots for a in args) or '--energy-workload-only' not in args): continue
        def option(name):
            try:return args[args.index(name)+1]
            except (ValueError,IndexError):return ''
        output = option('--out-dir')
        if not output and any(Path(a).name == 'native_hailo_trt_fifo_from_benchmarkset.py' for a in args):
            report = option('--result-json')
            if report and Path(report).is_absolute():
                output = str(Path(report).parent)
        nonce=option('--energy-preflight-nonce')or option('--preflight-nonce')
        if generic:
            if not Path(output).is_absolute() or '/energy/' not in output or not nonce or Path(output).name != nonce: continue
        elif '/replay.' not in output: continue
        contract=option('--source-contract-sha256')or (option('--energy-command-contract-sha256')if generic else '')
        if not contract and option('--energy-command-contract-file'):
            try:contract=json.loads(Path(option('--energy-command-contract-file')).read_text()).get('contract_sha256','')
            except (OSError,ValueError):pass
        selected[pid] = {'root_pid':pid,'output_directory':output,'argv':args,
                         'preflight_nonce':nonce,'command_contract_sha256':contract}
    result = []
    for root, binding in selected.items():
        descendants = {root}
        for _ in range(len(processes)):
            added = {pid for pid,(parent,_,_) in processes.items() if parent in descendants} - descendants
            if not added: break
            descendants.update(added)
        workers = []
        for pid in sorted(descendants):
            try:
                parent,ticks,args = processes[pid]
                threads = [{'tid':int(t.name),'allowed_cpus':sorted(os.sched_getaffinity(int(t.name)))}
                           for t in Path(f'/proc/{pid}/task').iterdir() if t.name.isdecimal()]
                workers.append({'pid':pid,'parent_pid':parent,'start_ticks':ticks,'argv':args,
                                'allowed_cpus':sorted(os.sched_getaffinity(pid)),'threads':threads})
            except (OSError, KeyError): continue
        if workers: result.append({**binding,'workers':workers})
    return result


def snapshot(transport, *, setup_id, run_id, group, dynamic=False):
    from .. import resources_utils
    code = resources_utils.read_text('resources','remote_scripts','native_host_telemetry.py')
    script = "import json,os\nns={'__name__':'__energy_environment__'}\nexec("+repr(code)+",ns)\n"
    if dynamic:
        script += "ns['_command']=lambda argv,**kw: {'status':'configuration_only_before_after','argv':argv}\n"
    args = dict(phase='during' if dynamic else 'configuration',backend='native',setup_id=setup_id,
                run_id=run_id,capture_group=group)
    script += inspect.getsource(_workers)+"\np=ns['capture'](**"+repr(args)+")\np['workload_roots']=_workers()\np['worker_observation_available']=bool(os.sched_getaffinity(0))\nprint('ENVIRONMENT='+json.dumps(p))\n"
    before = time.time_ns()
    rc, stdout = transport.run_read_only("python3 -B - <<'ENERGY_ENVIRONMENT'\n"+script+"ENERGY_ENVIRONMENT",timeout=30)
    after = time.time_ns()
    lines = [s[len('ENVIRONMENT='):]for s in stdout.splitlines()if s.startswith('ENVIRONMENT=')]
    if rc != 0 or len(lines)!=1: raise RuntimeError('DUT environment observation transport failed')
    payload = json.loads(lines[0])
    payload.update(controller_request_start_ns=before,controller_request_end_ns=after)
    if not any(r.get('scaling_cur_freq')for r in payload.get('cpu_frequency',[])) or not (payload.get('host')or{}).get('loadavg') or not payload.get('worker_observation_available'):
        raise RuntimeError('Required DUT CPU clocks/resource observation unavailable')
    return payload


class EnergyEnvironmentObserver:
    def __init__(self, setup, output, environment, cohort_id):
        from ..remote.ssh_transport import HostConfig,SSHTransport
        from ..remote.process_lease import RemoteProcessLeaseJournal
        journal = RemoteProcessLeaseJournal.from_environment({**os.environ,**(environment or {})},required=True)
        host = HostConfig(id=setup.setup_id,label=setup.setup_id,host=setup.jetson_address,
                          user=setup.jetson_user,port=setup.jetson_port,ssh_extra_args=setup.jetson_ssh_extra_args)
        self.transport = SSHTransport(host,log=lambda *_:None,remote_lease_scope=journal.scope)
        self.path = Path(output)/'environment_samples.jsonl'
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.args = dict(setup_id=setup.setup_id,run_id=cohort_id,group=str(Path(output).resolve()))
        self.stop_event = threading.Event()
        self.thread = None

    def _append(self, payload):
        with self.path.open('a')as stream:
            stream.write(json.dumps(payload)+'\n');stream.flush();os.fsync(stream.fileno())

    def __enter__(self):
        self._append(snapshot(self.transport,**self.args))
        def sample():
            while not self.stop_event.wait(5):
                try:self._append(snapshot(self.transport,**self.args,dynamic=True))
                except Exception as exc:self._append({'error':str(exc),'controller_request_start_ns':time.time_ns()})
        self.thread=threading.Thread(target=sample,name='energy-environment-observer',daemon=True)
        self.thread.start()
        return self

    def __exit__(self,*_):
        self.stop_event.set()
        if self.thread:self.thread.join(35)
        try:self._append(snapshot(self.transport,**self.args))
        except Exception as exc:self._append({'phase':'post_configuration','error':str(exc),
            'valid_finished_repeats_preserved':True,'controller_request_start_ns':time.time_ns()})


def validate_environment(run_dir, entry, *, freeze=False):
    root = Path(run_dir)
    samples = next((p/'environment_samples.jsonl'for p in [root,*root.parents]
                    if (p/'environment_samples.jsonl').is_file()),None)
    trace = entry.get('command_window_trace_binding')or{}
    start,end = trace.get('command_start_realtime_ns'),trace.get('command_end_realtime_ns')
    evidence = {'ok':False,'reason':'required_capture_environment_missing','samples_path':str(samples or ''),
                'command_start_realtime_ns':start,'command_end_realtime_ns':end,'observed_sample_count':0,
                'thermal_required':False}
    frozen = root/'environment_observation.json'
    if (samples is None and not frozen.is_file()) or not start or not end:return evidence
    if frozen.is_file():
        records=json.loads(frozen.read_text())['observations']
    else:
        records = [json.loads(s)for s in samples.read_text().splitlines()if s.strip()]
    during = [s for s in records if s.get('phase')=='during'
              and start<=s.get('controller_request_start_ns',0)<=s.get('controller_request_end_ns',0)<=end]
    preflight=entry.get('preflight_evidence')or{}
    nonce=preflight.get('nonce');contract=preflight.get('expected_command_contract_sha256')
    try:setup_id=json.loads(entry['campaign_row'])[3]
    except (KeyError,ValueError,IndexError):setup_id=None
    def bound(sample):
        group=Path(str(sample.get('capture_group')or''))
        return (nonce and contract and setup_id and sample.get('setup_id')==setup_id
                and sample.get('run_id')==(entry.get('campaign_cohort')or entry.get('campaign_run_id')) and group.is_absolute()
                and root.is_relative_to(group))
    valid = [s for s in during if bound(s)and any(r.get('scaling_cur_freq')for r in s.get('cpu_frequency',[]))and(s.get('host')or{}).get('loadavg')
             and any(w.get('pid')and w.get('start_ticks')and w.get('allowed_cpus')and w.get('threads')
                     for r in s.get('workload_roots',[])if r.get('preflight_nonce')==nonce
                     and r.get('command_contract_sha256')==contract for w in r.get('workers',[]))]
    evidence.update(ok=bool(valid),reason=''if valid else 'required_during_window_DUT_worker_clock_resource_evidence_missing',
                    observed_sample_count=len(valid),thermal_available=any(s.get('thermal_zones')for s in during),
                    worker_ids=sorted({(w['pid'],w['start_ticks'])for s in valid
                        for r in s['workload_roots']if r.get('preflight_nonce')==nonce
                        and r.get('command_contract_sha256')==contract for w in r['workers']}))
    if freeze and not frozen.exists():
        from .collector import _write_json
        before=[s for s in records if s.get('phase')=='configuration' and s.get('controller_request_end_ns',0)<start]
        _write_json(frozen,{'observations':before[-1:]+during})
    if frozen.is_file():
        from .collector import _sha256_path
        evidence.update(evidence_path=str(frozen),evidence_sha256=_sha256_path(frozen))
    return evidence
