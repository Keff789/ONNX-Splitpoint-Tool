"""Optional exact setup-local TensorRT control through the existing NativeTRT.

No engine discovery by basename, build dispatch, receipt rewriting or semantic
record stand-ins. The original remote cache paths are validated before copying
only the existing engine into this diagnostic's own staging directory.
"""
from __future__ import annotations
import ast
import ctypes
import ctypes.util
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
import typing
from smoke_common import read_json,write_json,digest,sha_token,extract_functions,save_tensors,stats,logits_record,compare


def resolve_trt_control(run,model,setup_id,source_sha256,tool_dir):
    path=Path(run)/'models'/model/'benchmark_results/quality_inputs'/setup_id/'results/results_native_full_tensorrt/task_quality_inputs/full_request.json'
    if not path.is_file():return {'status':'unavailable','reason':'setup_local_trt_full_quality_binding_missing','expected_request_path':str(path)}
    payload=read_json(path)
    if payload.get('model_id')!=model or payload.get('setup_id')!=setup_id or payload.get('task')!='classification' or payload.get('variant')!='full':
        raise ValueError('trt_full_control_model_setup_task_identity_mismatch')
    sys.path.insert(0,str(tool_dir))
    from onnx_splitpoint_tool.quality_service import _validate_tensorrt_candidate_execution_contract
    producer,_=_validate_tensorrt_candidate_execution_contract(payload.get('producer_identity'),role='diagnostic TRT control',task='classification')
    if producer.get('model_id')!=model or producer.get('setup_id')!=setup_id:
        raise ValueError('trt_full_control_producer_model_setup_mismatch')
    if sha_token(producer['source_onnx']['sha256'])!=sha_token(source_sha256):
        raise ValueError('trt_full_control_float_source_mismatch')
    return {'status':'bound_remote_artifacts_pending','binding_request_sha256':digest(path),
            'binding_request_path':str(path),'producer_identity':producer,
            'model_id':model,'setup_id':setup_id,'source_onnx_sha256':sha_token(source_sha256),
            'hardware_executed':False}


def identity_arguments(control):
    p=control['producer_identity'];build=p['build_onnx'];engine=p['engine'];exe=p['trtexec'];receipt=p['engine_build_receipt']
    if sha_token(build['sha256'])!=sha_token(p['source_onnx']['sha256']):
        raise ValueError('trt_full_control_source_build_conflict')
    # Same source bytes can be verified at the bound content cache path; a
    # deleted old run directory is never required as a second source copy.
    return SimpleNamespace(explicit_full_source_onnx=build['path'],explicit_full_build_onnx=build['path'],
        explicit_full_engine=engine['path'],explicit_full_trtexec=exe['path'],explicit_full_build_receipt=receipt['path'],
        expected_source_onnx_sha256=sha_token(build['sha256']),expected_build_onnx_sha256=sha_token(build['sha256']),
        expected_engine_sha256=sha_token(engine['sha256']),expected_trtexec_sha256=sha_token(exe['sha256']),
        expected_engine_build_receipt_file_sha256=sha_token(p['engine_build_receipt_file_sha256']),
        expected_engine_build_receipt_sha256=sha_token(receipt['sha256']),
        expected_trt_engine_build_receipt_sha256=sha_token(receipt['receipt']['receipt_sha256']),
        expected_source_onnx_size_bytes=int(build['size_bytes']),expected_build_onnx_size_bytes=int(build['size_bytes']),
        expected_engine_size_bytes=int(engine['size_bytes']),expected_trtexec_size_bytes=int(exe['size_bytes']),
        expected_engine_build_receipt_size_bytes=int(receipt['size_bytes']),
        quality_first_producer_identity_sha256=sha_token(p['producer_identity_sha256']))


def native_trt_class(source):
    """Load unchanged current class bodies, avoiding unrelated Hailo CLI imports."""
    import numpy as np
    tree=ast.parse(Path(source).read_text(),filename=str(source))
    wanted={'_CudaCompat','_pointer_value','NativeTRT'}
    nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in wanted]
    if {n.name for n in nodes}!=wanted:raise ValueError('current_native_trt_definitions_missing')
    env={'np':np,'ctypes':ctypes,'Path':Path,**vars(typing)}
    body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),*nodes]
    exec(compile(ast.fix_missing_locations(ast.Module(body=body,type_ignores=[])),str(source),'exec'),env)
    return env['NativeTRT']


def run_trt_control(request,stage,out):
    import numpy as np
    from PIL import Image
    control=request.get('trt_control') or {'status':'unavailable','reason':'setup_local_trt_full_quality_binding_missing'}
    if control.get('status')!='bound_remote_artifacts_pending':return dict(control,hardware_executed=False)
    suite=Path(stage)/'suite';target=Path(out)/'trt_control';target.mkdir(exist_ok=True)
    result={'status':'started','hardware_executed':False,'rows':[],'counts_as_benchmark':False,
            'runtime':'current_production_NativeTRT','producer_identity_sha256':control['producer_identity']['producer_identity_sha256']}
    engine=None
    try:
        funcs=extract_functions(suite/'native_full_semantic_dump.py',['_verify_explicit_trt_identity','_prepare_image_tensor'],{'np':np,'Image':Image})
        verified=funcs['_verify_explicit_trt_identity'](identity_arguments(control))
        owned_engine=Path(stage)/'trt_control_engine.engine';shutil.copyfile(verified['paths']['engine'],owned_engine)
        if digest(owned_engine)!=verified['hashes']['engine']:raise ValueError('staged_trt_engine_content_mismatch')
        engine=native_trt_class(suite/'native_hailo10_trt_e2e_from_benchmarkset.py')(owned_engine)
        if len(engine.inputs)!=1:raise ValueError('trt_control_single_input_required')
        inp=engine.inputs[0];names=list(engine.outputs)
        if not request.get('source_input_name') or inp!=request['source_input_name']:
            raise ValueError('trt_control_input_name_binding_mismatch')
        if names!=request['onnx_output_names']:raise ValueError('trt_control_output_name_binding_mismatch')
        if tuple(engine.shapes[inp])!=(1,3,*request['input_hw']) or np.dtype(engine.dtypes[inp])!=np.dtype('float32'):
            raise ValueError('trt_control_source_input_binding_mismatch')
        for i,sample in enumerate(request['samples'][:request['hardware_samples']]):
            image=suite/sample['image']
            if digest(image)!=sample['sha256']:raise ValueError('trt_control_sample_content_mismatch')
            feed,_,prep=funcs['_prepare_image_tensor'](image,engine.shapes[inp],engine.dtypes[inp],task='classification')
            before=feed.copy();raw=engine.run({inp:feed});result['hardware_executed']=True
            if set(raw)!=set(names):raise ValueError('trt_control_runtime_output_names_mismatch')
            copied=[np.asarray(raw[name]).copy() for name in names]
            record=logits_record(copied,names,sample['label_id'])
            save_tensors(target/f'trt_{i:03d}.npz',feed=before,**{f'output_{j:02d}':v for j,v in enumerate(copied)})
            result['rows'].append({'image_id':sample['image_id'],'label_id':sample['label_id'],'prediction':record,
                'input':stats(before),'input_modified_by_engine':not np.array_equal(before,feed),'preprocessing':prep})
            if result['rows'][-1]['input_modified_by_engine']:raise ValueError('trt_control_input_mutated')
            write_json(target/'result.json',result)
        result['status']='observed';result['sample_count']=len(result['rows']);result['identity_verification']=verified['status']
    except Exception as exc:
        result.update(status='unavailable' if not result['hardware_executed'] else 'failed',reason=f'{type(exc).__name__}: {exc}')
    finally:
        if engine is not None:engine.close()
        write_json(target/'result.json',result)
    return result
