"""Offline tests. Fake TensorFlow below is NEVER hardware evidence."""
from pathlib import Path
import contextlib, copy, importlib.util, json, os, signal, subprocess, sys, textwrap, time, zipfile
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]/'scripts/hailo_gpu_diagnostics';sys.path.insert(0,str(ROOT))
import collect as c
import worker as w


def controller_command(lock, *args):
    # Test-process injection only. Product CLI has no lock bypass option.
    bootstrap = ("import sys;from pathlib import Path;sys.path.insert(0,"+repr(str(ROOT))+");"
                 "import collect;c=collect;c.platform_interlock_path=lambda:Path("+repr(str(lock))+");"
                 "sys.argv=['collect.py']+sys.argv[1:];raise SystemExit(c.main())")
    return [sys.executable, '-I', '-B', '-c', bootstrap, *map(str, args)]

class Tensor:
    def __init__(self,a,device='/device:GPU:0'):self.a=np.asarray(a);self.device=device
    def numpy(self):return self.a

def test_real_numpy_convolution():
    x=np.ones((1,4,4,2),np.float32);k=np.ones((3,3,2,3),np.float32)
    out=w.conv_reference(np,x,k);assert out.shape==(1,2,2,3);assert (out==18).all()

@pytest.mark.parametrize('device',['/device:CPU:0','/CPU:0',''])
def test_reject_cpu_result(device):
    with pytest.raises(RuntimeError):w.assess_tensor(np,Tensor([1],device),np.array([1]))

@pytest.mark.parametrize('value',[float('nan'),float('inf'),5.0])
def test_reject_bad_gpu_result(value):
    with pytest.raises((ValueError,AssertionError)):w.assess_tensor(np,Tensor([value]),np.array([1.]))

def test_correct_gpu_metadata_numeric_check():
    assert w.assess_tensor(np,Tensor([1.]),np.array([1.]))['cpu_fallback_accepted'] is False

def test_env_changes_are_child_only(tmp_path):
    parent={'PATH':'/usr/bin','CUDA_VISIBLE_DEVICES':'-1','PYTHONPATH':'/deepx-overlay','CUDA_HOME':'/usr/local/cuda','XLA_FLAGS':'--xla_gpu_cuda_data_dir=/old'}
    before=copy.deepcopy(parent);env,detail=c.child_env(parent,Path('/venv'),tmp_path,'0')
    assert parent==before and 'PYTHONPATH' not in env and env['CUDA_VISIBLE_DEVICES']=='0'
    assert env['CUDA_HOME']==parent['CUDA_HOME'] and env['XLA_FLAGS']==parent['XLA_FLAGS']
    assert detail['parent_cuda_visible_devices']=='-1'

def test_busy_lock_blocks_real_controller_before_output(tmp_path):
    lock=tmp_path/'lock';out=tmp_path/'out'
    with c.interlock(lock):
        p=subprocess.run(controller_command(lock, '--output-parent', out),capture_output=True,text=True,timeout=10)
    assert p.returncode==3 and 'SMOKE_STARTED=NO' in p.stdout and not out.exists()
    assert lock.exists() and lock.stat().st_size==0

def test_lock_released_without_deleting(tmp_path):
    p=tmp_path/'lock'
    with c.interlock(p):pass
    inode=p.stat().st_ino
    with c.interlock(p):assert p.stat().st_ino==inode

def test_missing_venvs_full_cli_zip(tmp_path):
    p=subprocess.run(controller_command(tmp_path/'lock', '--output-parent', tmp_path, '--venv-hailo8', tmp_path/'missing8', '--venv-hailo10', tmp_path/'missing10'),capture_output=True,text=True,timeout=10)
    assert p.returncode==2
    z=list(tmp_path.glob('hailo_gpu_compute_r1_*.zip'));assert len(z)==1
    with zipfile.ZipFile(z[0]) as f:
        r=json.loads(f.read('collection_summary.json'));assert len(r['families'])==2
        assert all(x['error']=='expected_DFC_venv_missing' for x in r['families'])

def test_zip_is_explicit_results_only(tmp_path):
    out=tmp_path/'ev';out.mkdir();(out/'collection_summary.json').write_text('{}')
    for rel in ['hailo8/model.hef','hailo8/tmp/a.cubin','hailo8/fake.npy','hailo8/private.pem']:
        p=out/rel;p.parent.mkdir(exist_ok=True,parents=True);p.write_bytes(b'must not archive')
    with zipfile.ZipFile(c.evidence_zip(out)) as z:assert z.namelist()==['collection_summary.json']

def test_aggregate_does_not_accept_label_only():
    p={'returncode':0,'timed_out':False,'cleanup_complete':True}
    r={'schema':'hailo-gpu-compute-smoke-r1','family':'hailo8','status':'compute_pass','checks':[]}
    assert not c.valid_pass(r,p,'hailo8')
    r['checks']=[{'name':n,'status':'pass'} for n in w.REQUIRED+('dfc_sdk_import',)]
    assert c.valid_pass(r,p,'hailo8')
    assert not c.valid_pass(r,{**p,'timed_out':True},'hailo8')
    assert not c.valid_pass(r,{**p,'returncode':-6},'hailo8')
    assert not c.valid_pass(r,{**p,'unexpected_live_processes_after_worker_exit':True},'hailo8')
    assert not c.valid_pass(r,{**p,'cleanup_complete':False},'hailo8')

def test_real_subprocess_timeout_preserves_log_and_cleans(tmp_path):
    script=tmp_path/'hang.py';script.write_text('import signal,time\nsignal.signal(signal.SIGTERM,signal.SIG_IGN)\nprint("READY",flush=True)\ntime.sleep(100)\n')
    r=c.run_command([sys.executable,'-I','-S','-B',str(script)],tmp_path,os.environ.copy(),1,heartbeat=.3)
    assert r['timed_out'] and r['cleanup_complete'] and r['returncode'] is not None
    assert 'READY' in (tmp_path/'console.log').read_text()

def test_real_subprocess_crash_returncode(tmp_path):
    r=c.run_command([sys.executable,'-I','-B','-c','import os;os._exit(7)'],tmp_path,os.environ.copy(),5)
    assert r['returncode']==7 and r['cleanup_complete'] and not r['timed_out']

FAKE_TF=r'''
# TEST FIXTURE: all computations below are NumPy, NOT actual GPU evidence.
import numpy as np
from contextlib import nullcontext
__version__='TEST_SIMULATED_NO_GPU'
class T:
 def __init__(self,x):self.x=np.asarray(x);self.device='/device:GPU:0'
 def numpy(self):return self.x
 def __add__(self,b):return T(self.x+(b.x if isinstance(b,T) else b))
 def __mul__(self,b):return T(self.x*(b.x if isinstance(b,T) else b))
class GPU:name='/physical_device:GPU:0'
class Exp:
 def get_device_details(self,g):return {'device_name':'TEST FAKE GPU','compute_capability':(6,1)}
 def set_memory_growth(self,g,b):self.m=b
 def get_memory_growth(self,g):return self.m
class Th:
 def set_inter_op_parallelism_threads(self,n):pass
 def set_intra_op_parallelism_threads(self,n):pass
class C:
 experimental=Exp();threading=Th()
 def set_soft_device_placement(self,x):self.s=x
 def get_soft_device_placement(self):return self.s
 def list_physical_devices(self,t):return [GPU()]
 def set_visible_devices(self,g,t):pass
config=C()
class S:
 def get_build_info(self):return {'SIMULATED':True}
sysconfig=S()
class D:
 def set_log_device_placement(self,b):pass
debugging=D()
def device(x):return nullcontext()
def constant(x):return T(x)
def matmul(a,b):return T(a.x@b.x)
class N:
 def conv2d(self,a,b,strides,padding):
  x=a.x;k=b.x;out=np.empty((1,x.shape[1]-2,x.shape[2]-2,4),np.float32)
  for i in range(out.shape[1]):
   for j in range(out.shape[2]):out[:,i,j,:]=np.einsum('bhwc,hwco->bo',x[:,i:i+3,j:j+3,:],k)
  return T(out)
nn=N()
class M:
 def sin(self,x):return T(np.sin(x.x))
 def exp(self,x):return T(np.exp(x.x))
math=M()
def function(jit_compile,autograph):
 assert jit_compile is True and autograph is False
 return lambda f:f
'''

@pytest.fixture(scope='module')
def fake_venv(tmp_path_factory):
    p=tmp_path_factory.mktemp('fake_dfc')/'venv'
    subprocess.run([sys.executable,'-m','venv','--without-pip','--system-site-packages',str(p)],check=True,capture_output=True,timeout=20)
    site=next((p/'lib').glob('python*/site-packages'))
    (site/'hailo_sdk_client.py').write_text('__version__="TEST_FAKE"\nclass ClientRunner: pass\n')
    (site/'tensorflow.py').write_text(FAKE_TF)
    return p

def test_full_real_process_cli_with_simulated_vendor_api(tmp_path,fake_venv):
    before={p:p.read_bytes() for p in fake_venv.rglob('*.py')}
    cmd=controller_command(tmp_path/'lock', '--output-parent', tmp_path, '--venv-hailo8', fake_venv, '--venv-hailo10', fake_venv)
    p=subprocess.run(cmd,capture_output=True,text=True,timeout=30)
    assert p.returncode==0,p.stdout+'\n'+p.stderr
    z=next(tmp_path.glob('hailo_gpu_compute_r1_*.zip'))
    with zipfile.ZipFile(z) as f:
        s=json.loads(f.read('collection_summary.json'));assert s['status']=='compute_pass'
        for family in ['hailo8','hailo10h']:
            data=json.loads(f.read(family+'/gpu_compute_result.json'))
            assert data['tensorflow']['version']=='TEST_SIMULATED_NO_GPU'
            assert data['model_compiler_invoked'] is False and data['status']=='compute_pass'
    assert before=={p:p.read_bytes() for p in before}
    assert not list(fake_venv.rglob('tensorflow*.pyc'))

def test_real_worker_wrong_venv_rejected(tmp_path):
    p=subprocess.run([sys.executable,'-I','-B',str(ROOT/'worker.py'),'--output',str(tmp_path),
        '--expected-venv',str(tmp_path/'not_this_python'),'--family','hailo8'],capture_output=True,text=True,timeout=10)
    assert p.returncode==2
    assert 'wrong_interpreter_prefix' in json.loads((tmp_path/'gpu_compute_result.json').read_text())['error']


def test_worker_left_live_child_is_not_normal_completion(tmp_path):
    script=tmp_path/'leave_child.py'
    script.write_text('import subprocess,sys\nsubprocess.Popen([sys.executable,"-I","-S","-B","-c","import time;time.sleep(60)"])\n')
    result=c.run_command([sys.executable,'-I','-S','-B',str(script)],tmp_path,os.environ.copy(),5)
    assert result['returncode']==0 and result['cleanup_complete']
    assert result['unexpected_live_processes_after_worker_exit']


def test_product_default_lock_is_exact_workflow_lock():
    from onnx_splitpoint_tool.workflow.run_control import platform_workflow_interlock_path
    assert c.platform_interlock_path() == platform_workflow_interlock_path()


@pytest.mark.parametrize('args', [('--timeout', '181'), ('--timeout', '600'), ('--lock', '/tmp/unrelated')])
def test_product_cli_refuses_long_probe_or_alternative_lock(args, tmp_path):
    result = subprocess.run([sys.executable, '-I', '-B', str(ROOT/'collect.py'), *args,
        '--output-parent', str(tmp_path/'not_started')], text=True, capture_output=True, timeout=10)
    assert result.returncode == 2
    assert not (tmp_path/'not_started').exists()


def test_existing_shared_workflow_lock_prevents_gpu_child(tmp_path):
    import fcntl
    lock=tmp_path/'workflow.lock';out=tmp_path/'must_not_exist'
    with lock.open('a+') as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
        result=subprocess.run(controller_command(lock, '--output-parent', out),
                              capture_output=True, text=True, timeout=10)
    assert result.returncode == 3 and 'SMOKE_STARTED=NO' in result.stdout
    assert not out.exists()
