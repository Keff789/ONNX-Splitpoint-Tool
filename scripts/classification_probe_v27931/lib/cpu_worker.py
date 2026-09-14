#!/usr/bin/env python3
"""Untimed CPU controls on the exact existing float ONNX; no model writes."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import sys
import traceback
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parent))
from smoke_common import *


def inspect_graph(path):
    import onnx
    from onnx import numpy_helper
    model = onnx.load(str(path), load_external_data=False)
    constants = {v.name: v for v in model.graph.initializer}
    def info(v):
        t = v.type.tensor_type
        return {'name': v.name, 'elem_type': int(t.elem_type),
                'shape': [int(d.dim_value) if d.HasField('dim_value') else str(d.dim_param) for d in t.shape.dim]}
    def node_info(n):
        fields = {'name': n.name, 'op_type': n.op_type, 'inputs': list(n.input), 'outputs': list(n.output)}
        vals = {}
        for name in n.input:
            if name not in constants:
                continue
            t = constants[name]; count = math.prod(t.dims)
            if count <= 24 and not t.external_data:
                try: vals[name] = numpy_helper.to_array(t).tolist()
                except Exception as exc: vals[name] = type(exc).__name__
        fields['small_initializer_values'] = vals
        return fields
    inits = set(constants); inputs = [v for v in model.graph.input if v.name not in inits]
    # Follow the input-side dataflow for six levels, bounded to 48 nodes.
    frontier = {v.name for v in inputs}; captured = []; seen = set()
    for depth in range(6):
        next_frontier = set()
        for index, n in enumerate(model.graph.node):
            if index not in seen and any(name in frontier for name in n.input):
                captured.append({'depth': depth, **node_info(n)}); seen.add(index); next_frontier.update(n.output)
                if len(captured) >= 48: break
        frontier = next_frontier
        if not frontier or len(captured) >= 48: break
    external = [v.name for v in model.graph.initializer if v.external_data]
    return {'status': 'observed', 'inputs': [info(v) for v in inputs], 'outputs': [info(v) for v in model.graph.output],
            'node_count': len(model.graph.node), 'input_dataflow': captured,
            'metadata': {p.key:p.value for p in model.metadata_props}, 'external_initializers': external,
            'claim': 'bounded_graph_observation_not_a_proof_of_absent_folded_normalization'}


def run(request, stage, output):
    import numpy as np
    from PIL import Image
    output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0,str(request['tool_dir']))
    from onnx_splitpoint_tool.deepx.preprocessing_probe import resolve_source_build_semantics
    suite = stage/'suite'; bench = isolated_product(suite)
    from splitpoint_runners import native_full_input as nfi
    from splitpoint_runners.harness import classification as cls
    harness = cls.ClassificationHarness(labels=[str(i) for i in range(1000)])
    semantic = extract_functions(suite/'native_full_semantic_dump.py', ['_prepare_image_tensor'], {'np': np, 'Image': Image})
    generic = extract_functions(suite/'run_split_onnxruntime.py', ['_load_image_as_nchw'], {'np':np, 'Image':Image}, method=('DeepXSession','_candidate_inputs'))
    c = read_json(suite/'deepx/deepx_m1/full/output_contract.json'); inp = c['input']
    h,w = request['input_hw']; model = Path(request['local_onnx'])
    if digest(model) != request['expected_onnx_sha256']:
        raise ValueError('original_onnx_changed_before_cpu_control')
    summary = {'schema':SCHEMA,'stage':'cpu','status':'started','counts_as_benchmark':False,
               'model_id':request['model_id'],'request_nonce':request['nonce'],'rows':[], 'versions':{'numpy':np.__version__, 'python':sys.version},
               'errors':[], 'graph':{}, 'hypothesis':'scale_only_vs_imagenet_mean_std_on_same_float_graph'}
    try:
        summary['graph'] = inspect_graph(model)
    except Exception as exc:
        summary['graph'] = {'status':'unavailable','error':f'{type(exc).__name__}: {exc}'}
    write_json(output/'graph.json', summary['graph'])
    summary['source_build_semantics']=resolve_source_build_semantics(model_id=request['model_id'],graph_observation=summary['graph'],output_contract=c,profile_mode=request.get('classification_mode','unresolved'),export_metadata={'preprocess':request.get('export_preprocess',{})})
    if request.get('stage') in {'contracts','calibration-audit'}:
        summary.update(status='complete',hardware_executed=False,compiler_invoked=False,model_acceptance='NOT_EVALUATED',diagnostic_only=True)
        write_json(output/'cpu_result.json',summary)
        return summary
    ort_session = None; names = []
    try:
        if summary['graph'].get('external_initializers'):
            raise ValueError('external_ONNX_payloads_require_separate_content_binding')
        import onnxruntime as ort
        options = ort.SessionOptions(); options.intra_op_num_threads=2; options.inter_op_num_threads=1
        ort_session=ort.InferenceSession(str(model), sess_options=options, providers=['CPUExecutionProvider'])
        inputs=ort_session.get_inputs(); names=[o.name for o in ort_session.get_outputs()]
        if len(inputs)!=1 or inputs[0].shape!=[1,3,h,w] or inputs[0].type!='tensor(float)':
            raise ValueError('CPU_model_requires_recorded_static_float32_NCHW_input')
        expected_input=request.get('source_input_name')
        if not expected_input or expected_input!=c.get('source_model_input',{}).get('name') or inputs[0].name!=expected_input:
            raise ValueError('source_model_input_name_disagrees_with_saved_contract')
        if len(names)!=1 or names!=request['onnx_output_names']:
            raise ValueError('source_model_output_names_disagree_with_saved_contract')
        input_name=inputs[0].name; summary['versions']['onnxruntime']=ort.__version__
        summary['cpu_provider']=ort_session.get_providers()
        if summary['cpu_provider']!=['CPUExecutionProvider']:
            raise ValueError('unexpected_execution_provider')
    except Exception as exc:
        summary['errors'].append({'stage':'CPUInferenceSession','error':f'{type(exc).__name__}: {exc}'})
        ort_session=None
    for ordinal, sample in enumerate(request['samples']):
        image=suite/sample['image']; row={'ordinal':ordinal,'image_id':sample['image_id'],'label_id':sample['label_id'], 'controls':{}}
        try:
            if digest(image)!=sample['sha256']:
                raise ValueError('selected_image_hash_mismatch')
            raw, rgb, _=nfi._prepare_tensor(image, shape=inp['shape'],dtype=np.dtype(inp['dtype']), layout=inp['layout'],task='classification',normalization=inp['normalization'],preprocess_mode=inp['preprocess_mode'],pad_value=inp['letterbox_pad_value'])
            scale,_,_=nfi._prepare_tensor(image,shape=[1,3,h,w],dtype=np.dtype('float32'),layout='NCHW',task='classification',normalization='scale_0_1',preprocess_mode='resize',pad_value=0)
            mean,_,_=nfi._prepare_tensor(image,shape=[1,3,h,w],dtype=np.dtype('float32'),layout='NCHW',task='classification',normalization='imagenet_mean_std',preprocess_mode='resize',pad_value=0)
            sem,_,_=semantic['_prepare_image_tensor'](image,inp['shape'],np.dtype(inp['dtype']),task='classification',normalization=inp['normalization'],preprocess_mode=inp['preprocess_mode'],letterbox_pad=inp['letterbox_pad_value'])
            canonical=generic['_canonical_image_preprocessing_contract']('classification',(h,w))
            ref=generic['_load_image_as_nchw'](image,target_hw=(h,w),dtype=np.dtype('float32'),scale='imagenet',preprocessing_contract=canonical)
            if ref is None: raise ValueError('canonical_product_reference_input_unavailable')
            crop=next(iter(harness.make_inputs(SimpleNamespace(image_path=image,input_name='input',input_shape=[1,3,h,w])).values()))
            candidates=generic['_candidate_inputs'](SimpleNamespace(runtime_input_shapes={'input':(1,h,w,3)}),ref,'input')
            row['input_comparisons']={'native_vs_semantic':compare(raw,sem), 'mean_std_vs_canonical_reference':compare(mean,ref),
                'canonical_direct_vs_harness_crop256':compare(ref,crop),
                'generic_adapter_candidates':[{'index':i,'tensor':stats(a), 'native_comparison':compare(raw,a)} for i,a in enumerate(candidates)],
                'generic_adapter_scope':'candidate_generation_only; no unsafe fallback candidate is sent to hardware'}
            arrays={'prepared_rgb':rgb,'scale_only':scale,'mean_std':mean,'canonical_reference':ref,'harness_crop256':crop}
            arms={'scale_only':scale, 'imagenet_mean_std':mean, 'canonical_reference':ref, 'harness_crop256':crop}
            meta=request.get('export_preprocess') or {}
            if meta and meta.get('resize_size') and meta.get('crop_size') and len(meta.get('mean',[]))==3 and len(meta.get('std',[]))==3:
                # Explicit diagnostic arm, not a change to the canonical reference.
                with Image.open(image) as source:
                    alternative=cls._center_crop(cls._resize_short_side(source.convert('RGB'),int(meta['resize_size'][0])),(h,w))
                    alt=np.asarray(alternative,dtype=np.float32)/np.float32(255)
                alt=(alt-np.asarray(meta['mean'],dtype=np.float32))/np.asarray(meta['std'],dtype=np.float32)
                alt=np.ascontiguousarray(alt.transpose(2,0,1)[None])
                arms['export_metadata_geometry']=alt
                row['metadata_geometry_scope']='explicit_diagnostic_arm_using_recorded_resize_and_mean_std; not_a_new_reference'
            for arm, tensor in arms.items():
                row['controls'][arm]={'input':stats(tensor),'status':'input_observed'}
                if ort_session and request.get('stage')!='inputs':
                    outputs=ort_session.run(None,{input_name:tensor})
                    pred=logits_record(outputs,names,sample['label_id'],harness)
                    row['controls'][arm].update(status='inferred',prediction=pred)
                    for i,a in enumerate(outputs): arrays[f'output_{arm}_{i:02d}']=np.asarray(a).copy()
            if ordinal>=4:
                arrays={k:v for k,v in arrays.items() if k.startswith('output_')}
            save_tensors(output/f'cpu_{ordinal:03d}.npz',**arrays)
            row['status']='observed'
        except Exception as exc:
            row.update(status='failed',error=f'{type(exc).__name__}: {exc}')
            summary['errors'].append({'image_id':sample['image_id'],'error':row['error']})
        summary['rows'].append(row)
        write_json(output/'cpu_result.json',summary)
        print(f"CPU {request['model_id']} {ordinal+1}/{len(request['samples'])} {row['status']}",flush=True)
    summary['synthetic_input_sentinels']=[]
    for sentinel_index,(height,width) in enumerate(((57,91),(91,57),(63,95))):
        rgb=np.zeros((height,width,3),dtype=np.uint8); yy,xx=np.indices((height,width))
        rgb[...,sentinel_index]=255;rgb[::2,:, (sentinel_index+1)%3]=(xx[::2]*7)%256
        path=output/f'sentinel_input_{sentinel_index}.png';Image.fromarray(rgb).save(path)
        direct,_,_=nfi._prepare_tensor(path,shape=inp['shape'],dtype=np.dtype(inp['dtype']),layout=inp['layout'],task='classification',normalization=inp['normalization'],preprocess_mode=inp['preprocess_mode'],pad_value=inp['letterbox_pad_value'])
        semantic_feed,_,_=semantic['_prepare_image_tensor'](path,inp['shape'],np.dtype(inp['dtype']),task='classification',normalization=inp['normalization'],preprocess_mode=inp['preprocess_mode'],letterbox_pad=inp['letterbox_pad_value'])
        summary['synthetic_input_sentinels'].append({'id':f'rgb_geometry_{sentinel_index}','synthetic':True,'accuracy_evaluated':False,'hardware_executed':False,'original_hw':[height,width],'native_input':stats(direct),'native_vs_semantic':compare(direct,semantic_feed),'rgb_vs_bgr':compare(direct,np.ascontiguousarray(direct[...,::-1]))})
        save_tensors(output/f'sentinel_input_{sentinel_index}.npz',native_feed=direct,semantic_feed=semantic_feed)
    if digest(model)!=request['expected_onnx_sha256']:
        summary['errors'].append({'error':'original_onnx_changed_during_control'})
    summary['status']='complete' if not summary['errors'] and summary['graph'].get('status')=='observed' else 'partial'
    summary['model_acceptance']='NOT_EVALUATED'; summary['compiler_invoked']=False
    summary['hardware_executed']=False;summary['diagnostic_only']=True
    write_json(output/'cpu_result.json',summary)
    return summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--request',required=True,type=Path);p.add_argument('--out',required=True,type=Path)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    try:
        result=run(read_json(a.request),a.request.parent,a.out)
        return 0 if result['status']=='complete' else 2
    except Exception as exc:
        write_json(a.out/'cpu_result.json',{'schema':SCHEMA,'status':'failed','error':f'{type(exc).__name__}: {exc}','traceback':traceback.format_exc()})
        traceback.print_exc();return 2

if __name__=='__main__':raise SystemExit(main())
