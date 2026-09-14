from __future__ import annotations
import ctypes
import hashlib
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
import numpy as np
import pytest
from tests.test_v27931_classification_diagnostic_workflow import fixture_run,restore_isolated_imports,c,SOURCE
sys.path.insert(0,str(SOURCE/'scripts/classification_probe_v27931/lib'))
import trt_control


def _digest_value(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()


@pytest.mark.parametrize('model',['mobilenet_v3_large','resnet50','regnet_x_1_6gf'])
def test_t07_3_exact_original_setup_trt_binding_resolves(model,tmp_path):
    original=json.loads((SOURCE/f'tests/fixtures/v27931_classification/{model}_original_trt_full_request.json').read_text())
    target=tmp_path/'models'/model/'benchmark_results/quality_inputs'/original['setup_id']/'results/results_native_full_tensorrt/task_quality_inputs/full_request.json';target.parent.mkdir(parents=True);target.write_text(json.dumps(original))
    result=trt_control.resolve_trt_control(tmp_path,model,original['setup_id'],original['producer_identity']['source_onnx']['sha256'],SOURCE)
    assert result['status']=='bound_remote_artifacts_pending' and result['hardware_executed'] is False
    assert trt_control.resolve_trt_control(tmp_path,model,'other_setup',original['producer_identity']['source_onnx']['sha256'],SOURCE)['status']=='unavailable'
    original['producer_identity']['engine']['sha256']='a'*64;target.write_text(json.dumps(original))
    with pytest.raises(ValueError):trt_control.resolve_trt_control(tmp_path,model,original['setup_id'],original['producer_identity']['source_onnx']['sha256'],SOURCE)


def synthetic_control(tmp_path,f):
    cache=tmp_path/'explicit_synthetic_trt_cache';cache.mkdir();source=cache/'source.onnx';source.write_bytes(f.onnx.read_bytes());engine=cache/'full.engine';engine.write_bytes(b'SYNTHETIC_NOT_REAL_TRT_ENGINE');exe=cache/'trtexec';exe.write_bytes(b'NEVER_EXECUTE')
    receipt={'schema':'onnx-splitpoint/tensorrt-engine-build-receipt','schema_version':1,'build_returncode':0,'dry_run':False,'source_onnx':str(source),'source_onnx_sha256':c.digest(source),'engine':str(engine),'engine_sha256':c.digest(engine),'trtexec':str(exe),'trtexec_sha256':c.digest(exe),'command':[str(exe),'--onnx='+str(source),'--saveEngine='+str(engine),'--fp16']};receipt['receipt_sha256']=_digest_value(receipt)
    rp=cache/'receipt.json';rp.write_text(json.dumps(receipt,indent=2))
    def artifact(p):return {'path':str(p),'sha256':c.digest(p),'size_bytes':p.stat().st_size}
    producer={'source_onnx':artifact(source),'build_onnx':artifact(source),'engine':artifact(engine),'trtexec':artifact(exe),'engine_build_receipt':{'path':str(rp),'sha256':_digest_value(receipt),'size_bytes':len(json.dumps(receipt,sort_keys=True,separators=(',',':')).encode()),'receipt':receipt},'engine_build_receipt_file_sha256':c.digest(rp),'producer_identity_sha256':'a'*64}
    return {'status':'bound_remote_artifacts_pending','producer_identity':producer},engine


def fake_cuda_native_trt_class(source,created):
    Native=trt_control.native_trt_class_original(source)
    class CUDA:
        cudaMemcpyKind=SimpleNamespace(cudaMemcpyHostToDevice=1,cudaMemcpyDeviceToHost=2)
        def __init__(self):self.buffers={}
        def alloc(self,n):
            b=ctypes.create_string_buffer(int(n));p=ctypes.addressof(b);self.buffers[p]=b;return 0,p
        cudaMalloc=alloc;cudaMallocHost=alloc
        def cudaFree(self,p):self.buffers.pop(int(p),None);return 0
        cudaFreeHost=cudaFree
        def cudaMemcpyAsync(self,dst,src,n,kind,stream):ctypes.memmove(int(dst),int(src),int(n));return 0
        def cudaStreamSynchronize(self,s):return 0
        def cudaStreamDestroy(self,s):return 0
    def create(engine_path):
        obj=object.__new__(Native);obj.cudart=CUDA();obj.stream=1
        for key in ('inputs','outputs'):setattr(obj,key,[])
        for key in ('shapes','dtypes','dev','host_in','host_out','host_ptr','_host_backing'):setattr(obj,key,{})
        obj.ctx=SimpleNamespace(set_tensor_address=lambda n,p:True,execute_async_v3=lambda s:True)
        obj._reg('input',(1,3,224,224),np.dtype('float32'),True);obj._reg('logits',(1,1000),np.dtype('float32'),False)
        values=np.arange(1000,dtype=np.float32)[None];ctypes.memmove(obj.dev['logits'],values.ctypes.data,values.nbytes)
        created.append(obj);return obj
    return create


def test_t07_8_19_actual_native_trt_run_and_preparation_with_fake_cuda_only(tmp_path,monkeypatch):
    f=fixture_run(tmp_path,count=2);control,original_engine=synthetic_control(tmp_path,f);f.request['trt_control']=control
    before=original_engine.read_bytes();created=[]
    monkeypatch.setattr(trt_control,'native_trt_class_original',trt_control.native_trt_class,raising=False)
    monkeypatch.setattr(trt_control,'native_trt_class',lambda source:fake_cuda_native_trt_class(source,created))
    (f.out/'remote').mkdir()
    result=trt_control.run_trt_control(f.request,f.stage,f.out/'remote')
    assert result['status']=='observed',result
    assert result['sample_count']==2 and result['hardware_executed'] is True  # Actual adapter call, explicitly simulated CUDA in this test.
    assert all(r['prediction']['top1']==999 and not r['input_modified_by_engine'] for r in result['rows'])
    assert created[0].stream==0 and not created[0].dev
    assert original_engine.read_bytes()==before
    assert (f.stage/'trt_control_engine.engine').read_bytes()==before
    assert result['counts_as_benchmark'] is False


def test_t07_25_trt_control_refuses_changed_engine_without_build_or_device(tmp_path,monkeypatch):
    f=fixture_run(tmp_path,count=1);control,engine=synthetic_control(tmp_path,f);f.request['trt_control']=control;engine.write_bytes(b'tampered');(f.out/'remote').mkdir()
    def forbidden(*a):raise AssertionError('device opened despite bad artifact')
    monkeypatch.setattr(trt_control,'native_trt_class',forbidden)
    result=trt_control.run_trt_control(f.request,f.stage,f.out/'remote')
    assert result['status']=='unavailable' and not result['hardware_executed']
    assert 'artifact mismatch' in result['reason']
    assert not (f.stage/'trt_control_engine.engine').exists()


def test_t07_8_actual_trt_input_name_must_match_source_contract(tmp_path,monkeypatch):
    f=fixture_run(tmp_path,count=1);control,_=synthetic_control(tmp_path,f);f.request['trt_control']=control;created=[]
    monkeypatch.setattr(trt_control,'native_trt_class_original',trt_control.native_trt_class,raising=False)
    def wrong_factory(source):
        real=fake_cuda_native_trt_class(source,created)
        def make(path):
            obj=real(path);obj.inputs=['unbound_input'];return obj
        return make
    monkeypatch.setattr(trt_control,'native_trt_class',wrong_factory)
    (f.out/'remote').mkdir()
    result=trt_control.run_trt_control(f.request,f.stage,f.out/'remote')
    assert result['status']=='unavailable' and not result['hardware_executed']
    assert 'input_name_binding_mismatch' in result['reason']
    assert created[0].stream==0 and not created[0].dev
    assert not result['rows']
