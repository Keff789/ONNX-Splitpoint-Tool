"""Exact Native observer failure/reuse regressions; local processes only."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT)]
from onnx_splitpoint_tool.energy import environment_observer as observer, task_budget, collector

# Normal H10 E2E argv shape; no runner inference or DUT connection is started.
NATIVE_ARGV = ['/home/nx/ONNX-Splitpoint-Tool/scripts/native_hailo10_trt_e2e_from_benchmarkset.py', '--benchmark-set', '/fixture/benchmark_set', '--case', 'b001', '--precision', 'uint8_dequant_fp16', '--image', '/fixture/image.JPEG', '--frames', '1000', '--duration-s', '60.0', '--warmup', '0', '--queue-depth', '3', '--source-contract-sha256', 'efd0da19c8cd7ef043db9db821888165a4a6bf98d53f6b750d7facca3622c3bf', '--energy-workload-only', '--energy-preflight-attestation', '/fixture/preflight.json', '--energy-preflight-nonce', '8010c4c93bbc04fcaca056cfc05d9fa9989d5de9f7fe11f9e9d70ca4e6b7ee69', '--hw-arch', 'hailo10h', '--inflight', '8', '--producer-impl', 'auto', '--boundary-layout', 'memory_nhwc_to_nchw', '--out-dir', '$fresh_output_root', '--task', 'classification', '--preprocess-mode', 'resize', '--letterbox-pad-value', '0', '--dequant-scale', '0.12348616868257523', '--dequant-zero-point', '4.0', '--quantized-inputs', '--quantized-outputs', '--copy-outputs']



@pytest.mark.parametrize('script,energy,output,found', [
    ('native_hailo10_trt_e2e_from_benchmarkset.py', True, True, True),
    ('native_hailo_trt_fifo_from_benchmarkset.py', True, True, True),
    ('native_deepx_trt_e2e_from_benchmarkset.py', True, True, True),
    ('native_full_baseline_eval_runner.py', True, True, True),
    ('unknown_native_e2e_from_benchmarkset.py', True, True, False),
    ('native_hailo10_trt_e2e_from_benchmarkset.py', False, True, False),
    ('native_hailo10_trt_e2e_from_benchmarkset.py', True, False, False),
])
def test_actual_native_cmdline_root_and_guards(tmp_path, script, energy, output, found):
    argv = list(NATIVE_ARGV)
    argv[0] = '/actual/scripts/'+script
    argv[argv.index('--out-dir')+1] = str(tmp_path/('replay.original' if output else 'foreign'))
    if not energy:
        argv.remove('--energy-workload-only')
    p = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(30)', *argv],
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        time.sleep(.08)
        roots = [r for r in observer._workers() if r['root_pid']==p.pid]
        assert bool(roots) is found
        if found:
            root = roots[0]
            assert root['argv'][3:]==argv
            assert root['preflight_nonce']==argv[argv.index('--energy-preflight-nonce')+1]
            assert root['command_contract_sha256']==argv[argv.index('--source-contract-sha256')+1]
            worker = root['workers'][0]
            assert worker['pid']==p.pid and worker['start_ticks'] and worker['allowed_cpus']
            assert worker['threads'] and all(t['allowed_cpus'] for t in worker['threads'])
    finally:
        p.terminate()
        p.wait(timeout=5)


@pytest.mark.parametrize('bad', [None, 'no_workers', 'wrong_nonce', 'wrong_contract', 'wrong_setup', 'outside_window', 'no_affinity'])
def test_normal_environment_binding_requires_window_worker_evidence(tmp_path, bad):
    root = tmp_path/'capture/run_000'
    root.mkdir(parents=True)
    entry = {'campaign_cohort':'actual-parent', 'campaign_row':json.dumps(['backend','model','case','actual-setup']),
             'command_window_trace_binding':{'command_start_realtime_ns':100,'command_end_realtime_ns':200},
             'preflight_evidence':{'nonce':'a'*64,'expected_command_contract_sha256':'b'*64}}
    sample = {'phase':'during','controller_request_start_ns':110,'controller_request_end_ns':120,
              'setup_id':'actual-setup','run_id':'actual-parent','capture_group':str(root.parent),
              'cpu_frequency':[{'scaling_cur_freq':1234}],'host':{'loadavg':[.1,.1,.1]},
              'workload_roots':[{'preflight_nonce':'a'*64,'command_contract_sha256':'b'*64,
                 'workers':[{'pid':123,'start_ticks':'456','allowed_cpus':[0,1],
                             'threads':[{'tid':123,'allowed_cpus':[0,1]}]}]}]}
    if bad=='no_workers':sample['workload_roots']=[]
    elif bad=='wrong_nonce':sample['workload_roots'][0]['preflight_nonce']='c'*64
    elif bad=='wrong_contract':sample['workload_roots'][0]['command_contract_sha256']='c'*64
    elif bad=='wrong_setup':sample['setup_id']='foreign'
    elif bad=='outside_window':sample['controller_request_end_ns']=201
    elif bad=='no_affinity':sample['workload_roots'][0]['workers'][0]['allowed_cpus']=[]
    (root.parent/'environment_samples.jsonl').write_text(json.dumps(sample)+'\n')
    result = observer.validate_environment(root, entry)
    assert result['ok'] is (bad is None)
    assert result['observed_sample_count']==int(bad is None)
    assert not (root/'environment_observation.json').exists()


def actual_chains():
    cohort='actual-parent'
    chains=[{'logical_repeat':f'repeat:{i}','cohort_id':cohort,'finished':True,
             'valid':False,'collector_started':True,'source_completion_verified':True,
             'run_directory':f'/original/attempt/run_{i:03d}','transport_reasons':[]}
            for i in range(3)]
    return {}, {'row_id':'row','chains':chains,
                'limits':{'max_chains':9,'max_retries':2,'max_transport_failures':20}}


def test_normal_reuse_skips_old_invalid_chains_then_uses_first_valid_slots(tmp_path, monkeypatch):
    proof, data = actual_chains()
    budget = task_budget.EnergyTaskBudget.__new__(task_budget.EnergyTaskBudget)
    budget.campaign_row = data['row_id']
    budget.cohort_id = data['chains'][0]['cohort_id']
    budget.data = {'chains':copy.deepcopy(data['chains']),'limits':data['limits'],'source_id':'source'}
    budget.source = {}
    budget.source_id = 'source'
    budget._reentry = None
    budget.root = {'tasks':{'task':budget.data}}
    budget.checkpoint_path = tmp_path/'copy-budget'
    budget._check_cohort = lambda:None
    budget.is_set = lambda:False
    budget.counts = lambda:{'transport_failures':0,'begun_chains':len(budget.data['chains'])}
    kwargs = dict(command='original',command_contract_sha256='original',duration_s=60,require_ab=False)
    read = task_budget.EnergyTaskBudget.reusable_repeat.__wrapped__
    allowed = task_budget.EnergyTaskBudget.allowed.__wrapped__
    old = copy.deepcopy(budget.data['chains'])
    for i in range(3):
        assert read(budget, f'repeat:{i}', **kwargs) is None
        assert allowed(budget, f'repeat:{i}') is True
    verified = []
    def verify(chain, **kwargs):
        verified.append(chain['run_directory'])
        return {'original_verified':chain['run_directory']}
    monkeypatch.setattr(task_budget, 'read_verified_campaign_repeat', verify)
    for i in range(3):
        new = {**old[i],'valid':True,'run_directory':str(tmp_path/f'first-valid/run_{i:03d}')}
        budget.data['chains'].append(new)
        entry, reference = read(budget, f'repeat:{i}', **kwargs)
        assert entry['original_verified']==reference['run_directory']==new['run_directory']
        assert allowed(budget, f'repeat:{i}') is False
    assert len(verified)==3 and budget.data['chains'][:3]==old
    assert len(budget.data['chains'])==6 and data['limits']=={'max_chains':9,'max_retries':2,'max_transport_failures':20}


def test_campaign_cohort_starts_observer_before_capture_without_explicit_flag(tmp_path, monkeypatch):
    events = []
    class Budget:
        def __init__(self, *args, **kwargs):pass
        def is_set(self):return False
        def report(self, result, out):return result
    class Observer:
        def __init__(self, setup, out, env, cohort):assert cohort=='actual-parent'
        def __enter__(self):events.append('observer_started')
        def __exit__(self, *args):events.append('observer_finished')
    monkeypatch.setattr(task_budget,'EnergyTaskBudget',Budget)
    from onnx_splitpoint_tool.energy import environment_observer
    monkeypatch.setattr(environment_observer,'EnergyEnvironmentObserver',Observer)
    @task_budget.bounded_energy_task
    def capture(command, out, **kwargs):
        assert events==['observer_started']
        events.append('capture_called')
        return {'ok':False}
    capture('unchanged',tmp_path/'out',campaign_budget_file=tmp_path/'copy-budget',
        campaign_cohort='actual-parent',campaign_row_id='row',campaign_repeats=3,
        campaign_max_retries=2,campaign_max_transport_failures=20,
        require_runtime_work_units=True,require_command_window_alignment=True,
        require_environment_observation=False,setup=SimpleNamespace(urecs_address='source'))
    assert events==['observer_started','capture_called','observer_finished']
