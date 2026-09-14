#!/usr/bin/env python3
"""Synthetic TensorFlow GPU computations in an existing DFC venv. No model build."""
from __future__ import annotations
import argparse, importlib, importlib.metadata, json, os, resource, sys, time, traceback
from pathlib import Path

REQUIRED = ('matmul_gpu', 'conv2d_gpu', 'xla_math_gpu', 'xla_conv2d_gpu')
ENV_KEYS = ('CUDA_VISIBLE_DEVICES','CUDA_HOME','CUDA_PATH','LD_LIBRARY_PATH','PATH',
            'XLA_FLAGS','TF_XLA_FLAGS','ONNX_SPLITPOINT_HAILO_ALLOW_GPU',
            'CUDA_CACHE_PATH','TMPDIR','TF_FORCE_GPU_ALLOW_GROWTH',
            'TF_NUM_INTEROP_THREADS','TF_NUM_INTRAOP_THREADS','TF_CPP_VMODULE')

def write_json(path: Path, data: dict) -> None:
    tmp=path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False,default=str)+'\n',encoding='utf-8')
    os.replace(tmp,path)

def conv_reference(np, x, kernel):
    """Small independent NHWC/VALID convolution, not a TensorFlow CPU fallback."""
    kh,kw,cin,cout=kernel.shape
    out=np.empty((x.shape[0],x.shape[1]-kh+1,x.shape[2]-kw+1,cout),dtype=np.float32)
    for i in range(out.shape[1]):
        for j in range(out.shape[2]):
            out[:,i,j,:]=np.einsum('bhwc,hwco->bo',x[:,i:i+kh,j:j+kw,:],kernel)
    return out

def assess_tensor(np, tensor, reference):
    device=str(tensor.device)
    if '/device:GPU:' not in device and not device.upper().startswith('/GPU:'):
        raise RuntimeError('cpu_fallback_or_non_gpu_output:'+device)
    actual=np.asarray(tensor.numpy())  # synchronize actual device execution
    if actual.shape != reference.shape: raise ValueError('unexpected_output_shape')
    if not np.isfinite(actual).all(): raise ValueError('nonfinite_gpu_output')
    np.testing.assert_allclose(actual,reference,rtol=3e-4,atol=3e-4)
    return {'device':device,'shape':list(actual.shape),'dtype':str(actual.dtype),
            'finite':True,'max_abs_error':float(np.max(np.abs(actual-reference))),
            'rtol':3e-4,'atol':3e-4,'cpu_fallback_accepted':False}

def execute(output: Path, expected_venv: Path, family: str) -> int:
    output.mkdir(parents=True,exist_ok=True)
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    report={'schema':'hailo-gpu-compute-smoke-r1','revision':'R1_FIX1','family':family,'status':'running',
            'diagnostic_only':True,'model_compiler_invoked':False,'model_acceptance':'NOT_EVALUATED',
            'pid':os.getpid(),'python':sys.executable,'sys_prefix':sys.prefix,
            'environment':{k:os.environ[k] for k in ENV_KEYS if k in os.environ},'checks':[]}
    def save(): write_json(output/'gpu_compute_result.json',report)
    def phase(name,fn):
        report['phase']=name;save();print('PHASE_START='+name,flush=True);t=time.monotonic()
        try:
            detail=fn();row={'name':name,'status':'pass','detail':detail}
        except Exception as exc:
            row={'name':name,'status':'failed','error':f'{type(exc).__name__}: {exc}','traceback':traceback.format_exc()}
        row['elapsed_s']=round(time.monotonic()-t,4);report['checks'].append(row);save()
        print('PHASE_END='+name+' STATUS='+row['status'],flush=True)
        if row.get('error'): print('ERROR='+row['error'],flush=True)
        return row
    try:
        if Path(sys.prefix).resolve()!=expected_venv.resolve():
            raise RuntimeError(f'wrong_interpreter_prefix:{sys.prefix};expected={expected_venv}')
        report['phase']='package_inventory';save()
        report['packages']={}
        for d in importlib.metadata.distributions():
            name=d.metadata.get('Name','')
            if any(k in name.lower() for k in ('hailo','tensorflow','keras','numpy','nvidia','cuda','cudnn')):
                report['packages'][name]=d.version
        def sdk():
            mod=importlib.import_module('hailo_sdk_client')
            if not hasattr(mod,'ClientRunner'):raise RuntimeError('hailo_sdk_ClientRunner_missing')
            return {'module_file':getattr(mod,'__file__',None),'version':getattr(mod,'__version__',None),
                    'runner_instantiated':False}
        sdk_check=phase('dfc_sdk_import',sdk)
        if sdk_check['status']!='pass':raise RuntimeError('dfc_sdk_import_failed')
        report['tensorflow_imported_by_sdk'] = 'tensorflow' in sys.modules
        # Read-only optional observation; never creates a TF context.
        context_module = sys.modules.get('tensorflow.python.eager.context')
        context_object = getattr(context_module, '_context', None)
        report['context_initialized_after_sdk'] = (
            getattr(context_object, '_context_handle', None) is not None
            if context_module is not None else None
        )
        report['environment_after_sdk'] = {k:os.environ[k] for k in ENV_KEYS if k in os.environ}
        if os.environ.get('CUDA_VISIBLE_DEVICES') != report['environment'].get('CUDA_VISIBLE_DEVICES'):
            raise RuntimeError('sdk_changed_requested_cuda_visibility')
        report['phase']='tensorflow_import';save();print('PHASE_START=tensorflow_import',flush=True)
        import numpy as np
        import tensorflow as tf
        report['tensorflow']={'version':tf.__version__,'module_file':tf.__file__,'build_info':tf.sysconfig.get_build_info()}
        # FIX1: the SDK import may ALREADY have initialized TF. Do not call
        # thread-count, visibility or memory-growth setters afterwards.
        # Those requests are passed in the environment before Python starts.
        report['phase']='tensorflow_gpu_setup';save()
        report['threading_policy'] = {
            'method':'environment_before_sdk_import',
            'requested_inter_op':os.environ.get('TF_NUM_INTEROP_THREADS'),
            'requested_intra_op':os.environ.get('TF_NUM_INTRAOP_THREADS'),
            'late_thread_setters_called':False,
        }
        for kind in ('inter', 'intra'):
            getter = getattr(tf.config.threading, f'get_{kind}_op_parallelism_threads', None)
            try:
                report['threading_policy'][f'api_{kind}_op'] = getter() if getter else None
            except Exception as exc:
                report['threading_policy'][f'api_{kind}_op_error'] = str(exc)
        physical=tf.config.list_physical_devices('GPU')
        report['gpus']=[{'name':g.name, **tf.config.experimental.get_device_details(g)} for g in physical]
        save()
        if not physical:raise RuntimeError('tensorflow_no_visible_gpu')
        visible=tf.config.get_visible_devices('GPU')
        report['visible_gpus']=[g.name for g in visible]
        if len(visible)!=1:raise RuntimeError('tensorflow_gpu_selection_not_unique:'+str(len(visible)))
        report['memory_growth_policy'] = {
            'method':'TF_FORCE_GPU_ALLOW_GROWTH_before_sdk_import',
            'environment_request':os.environ.get('TF_FORCE_GPU_ALLOW_GROWTH'),
            'late_memory_growth_setter_called':False,
            'note':'API value may not reflect the environment override; not an allocator-size proof',
        }
        try:report['memory_growth']=tf.config.experimental.get_memory_growth(visible[0])
        except Exception as exc:report['memory_growth_observation_error']=str(exc)
        # Soft placement is a runtime policy, unlike immutable device/thread config.
        tf.config.set_soft_device_placement(False)
        tf.debugging.set_log_device_placement(True)
        report['soft_device_placement']=tf.config.get_soft_device_placement()
        if report['soft_device_placement'] is not False:raise RuntimeError('soft_device_placement_not_disabled')
        save()
        print('PHASE_END=tensorflow_gpu_setup STATUS=pass',flush=True)
        rng=np.random.default_rng(20260909)
        a=rng.uniform(-.5,.5,(64,64)).astype(np.float32);b=rng.uniform(-.5,.5,(64,64)).astype(np.float32)
        x=rng.uniform(-.5,.5,(1,16,16,3)).astype(np.float32);k=rng.uniform(-.5,.5,(3,3,3,4)).astype(np.float32)
        mm=np.matmul(a,b);cc=conv_reference(np,x,k)
        # All operands are device tensors; the independent references stay in NumPy.
        report['phase']='gpu_operand_allocation';save()
        with tf.device('/GPU:0'):
            at=tf.constant(a);bt=tf.constant(b);xt=tf.constant(x);kt=tf.constant(k)
        def eager_mm():
            with tf.device('/GPU:0'):y=tf.matmul(at,bt)
            return assess_tensor(np,y,mm)
        def eager_conv():
            with tf.device('/GPU:0'):y=tf.nn.conv2d(xt,kt,strides=1,padding='VALID')
            return assess_tensor(np,y,cc)
        def xla_math():
            @tf.function(jit_compile=True,autograph=False)
            def kernel(a,b):
                z=tf.matmul(a,b)*0.1
                return tf.math.sin(z)+tf.math.exp(z)
            with tf.device('/GPU:0'):y=kernel(at,bt)
            d=assess_tensor(np,y,np.sin(mm*.1)+np.exp(mm*.1));d['jit_compile_required']=True;return d
        def xla_conv():
            @tf.function(jit_compile=True,autograph=False)
            def kernel(x,k):
                y=tf.nn.conv2d(x,k,strides=1,padding='VALID')
                return y+tf.math.sin(y)
            with tf.device('/GPU:0'):y=kernel(xt,kt)
            d=assess_tensor(np,y,cc+np.sin(cc));d['jit_compile_required']=True;return d
        for name,fn in zip(REQUIRED,(eager_mm,eager_conv,xla_math,xla_conv)):phase(name,fn)
        passed={r['name'] for r in report['checks'] if r['status']=='pass'}
        report['status']='compute_pass' if set(REQUIRED)|{'dfc_sdk_import'} <= passed else 'compute_failed'
    except Exception as exc:
        report['failure_phase']=report.get('phase','unknown')
        report['status']='setup_failed';report['error']=f'{type(exc).__name__}: {exc}';report['traceback']=traceback.format_exc()
        print('WORKER_ERROR='+report['error'],flush=True)
    finally:
        try:
            report['loaded_cuda_libraries']=sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                if line.split() and any(t in line for t in ('libcuda.','libcudnn','libcublas','libcudart','libnvrtc','libcufft','libcusolver','libcusparse','libcurand','libnvJitLink'))})
        except OSError:report['loaded_cuda_libraries_unavailable']=True
        report['phase']='finished';save();print('WORKER_STATUS='+report['status'],flush=True)
    return 0 if report['status']=='compute_pass' else 2

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--expected-venv',type=Path,required=True);parser.add_argument('--family',required=True)
    a=parser.parse_args();raise SystemExit(execute(a.output,a.expected_venv,a.family))
