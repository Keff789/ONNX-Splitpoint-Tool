#!/usr/bin/env python3
"""Real DXRT full-model calls, with unchanged product preprocessing captured."""
from __future__ import annotations
import argparse
import gc
import os
from pathlib import Path
import sys
import traceback
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parent))
from smoke_common import *


def calibration_replay(suite, request, output, bench):
    """Explicit config replay, NOT an assertion about DXNN internals/vendor execution."""
    import numpy as np
    calibration=request.get('calibration_samples',[])
    ids=[s['image_id'] for s in calibration]
    if len(ids)!=len(set(ids)):raise ValueError('duplicate_calibration_sample_identity')
    if len(ids)>8:raise ValueError('calibration_sample_budget_exceeded')
    h,w=request['input_hw']
    expected=[{'resize':{'width':w,'height':h}}, {'div':{'x':255.0}},
              {'convertColor':{'form':'BGR2RGB'}}, {'transpose':{'axis':[2,0,1]}}, {'expandDim':{'axis':0}}]
    ops=((request.get('calibration_config') or {}).get('default_loader') or {}).get('preprocessings')
    if ops!=expected:return {'status':'not_evaluated','reason':'loader_recipe_not_exactly_supported','observed_preprocessings':ops,'historical_calibration_evidence_missing':True}
    try:import cv2
    except ImportError:return {'status':'not_evaluated','reason':'recorded_loader_geometry_dependency_unavailable','historical_calibration_evidence_missing':True,'vendor_loader_executed':False}
    rows=[]
    for item in request.get('calibration_samples',[]):
        image=suite/item['image']
        if digest(image)!=item['sha256']:raise ValueError('calibration_image_digest_mismatch')
        source=cv2.imread(str(image))
        if source is None:raise ValueError('calibration_cv2_imread_failed')
        # The operations are only replayed for the exact recorded supported recipe.
        cfg=request.get('calibration_config') or {}; ops=(cfg.get('default_loader') or {}).get('preprocessings')
        h,w=request['input_hw']
        expected=[{'resize':{'width':w,'height':h}}, {'div':{'x':255.0}},
                  {'convertColor':{'form':'BGR2RGB'}}, {'transpose':{'axis':[2,0,1]}}, {'expandDim':{'axis':0}}]
        if ops!=expected:
            return {'status':'not_evaluated','reason':'loader_recipe_not_exactly_supported', 'observed_preprocessings':ops}
        # OpenCV default resize is one explicit hypothesis; original vendor
        # loader source is collected separately to check its actual semantics.
        a=cv2.resize(source,(w,h),interpolation=cv2.INTER_LINEAR)
        a=a/255.0; a=cv2.cvtColor(a.astype(np.float32),cv2.COLOR_BGR2RGB)
        a=np.ascontiguousarray(a.transpose(2,0,1)[None],dtype=np.float32)
        contract=read_json(suite/'deepx/deepx_m1/full/output_contract.json')
        feed,_=bench._deepx_prepare_image_from_contract(source,h,contract,'classification')
        runtime_scale=np.ascontiguousarray(feed.transpose(2,0,1)[None].astype(np.float32)/np.float32(255.0))
        save_tensors(output/f"calibration_{len(rows):03d}.npz",config_replay=a,runtime_logical_scale=runtime_scale,runtime_uint8=feed)
        rows.append({'image_id':item['image_id'],'comparison':compare(a,runtime_scale), 'config_replay':stats(a)})
    return {'status':'reconstructed_with_current_loader' if rows else 'not_evaluated','rows':rows,
            'historical_calibration_evidence_missing':True,'logical_input_status':'reconstructed',
            'scope':'recorded_part1_loader_config_replay_with_explicit_cv2_INTER_LINEAR_hypothesis',
            'vendor_loader_executed':False,'compiled_internal_normalization_observed':False,
            'note':'div dtype/cvtColor casts are diagnostic; compare collected vendor source before attributing causality'}


def run(request, stage, out):
    if request.get('task') == 'detection':
        from detection_control import run_detection
        return run_detection(request, stage, out)
    import numpy as np
    from PIL import Image
    import dx_engine
    suite=stage/'suite'; out.mkdir(parents=True,exist_ok=True)
    bench=isolated_product(suite)
    from splitpoint_runners import native_full_input as nfi
    from splitpoint_runners.harness.classification import ClassificationHarness
    harness=ClassificationHarness(labels=[str(i) for i in range(1000)])
    model=suite/'deepx/deepx_m1/full/model.dxnn'
    if digest(model)!=request['expected_dxnn_sha256']:raise ValueError('staged_full_dxnn_digest_mismatch')
    contract=read_json(model.with_name('output_contract.json')); inp=contract['input']
    if inp['layout']!='HWC' or inp['dtype']!='uint8' or inp['shape']!=[*request['input_hw'],3]:
        raise ValueError('this_smoke_requires_recorded_HWC_uint8_contract')
    samples=request['samples'][:request['hardware_samples']]
    for s in samples:
        if digest(suite/s['image'])!=s['sha256']:raise ValueError('staged_input_image_digest_mismatch')
    names=[str(x['name']) for x in contract['outputs']]
    if len(names)!=1 or names!=request['onnx_output_names']:raise ValueError('full_output_name_binding_not_unique')
    summary={'schema':SCHEMA,'status':'started','model_id':request['model_id'],'diagnostic_only':True,
             'counts_as_benchmark':False,'model_acceptance':'NOT_EVALUATED','compiler_invoked':False,'request_nonce':request['nonce'],
             'versions':{'python':sys.version,'numpy':np.__version__,'dx_engine_module':str(dx_engine.__file__)},
             'quality_calls':[], 'native_calls':[], 'errors':[], 'hardware_executed':False,
             'collection_status':'pending','pipeline_consistency':'not_evaluated','root_cause_status':'hardware_pending'}
    try:
        from PIL import __version__ as pillow_version
        summary['versions']['pillow']=pillow_version
        import cv2
        summary['versions']['opencv']=cv2.__version__
    except ImportError as exc:summary['versions']['optional_import_note']=str(exc)
    if request.get('stage')=='calibration-audit':
        calibration=calibration_replay(suite,request,out,bench)
        summary.update(calibration=calibration,status='complete' if calibration.get('rows') else 'partial',hardware_executed=False,
                       collection_status='complete' if calibration.get('rows') else 'incomplete',pipeline_consistency='reconstructed_scope_only',root_cause_status='historical_calibration_evidence_missing')
        write_json(out/'remote_result.json',summary)
        return summary
    real_class=dx_engine.InferenceEngine
    class CaptureEngine:
        def __init__(self,path,*args,**kwargs):
            if Path(path).resolve()!=model.resolve():raise ValueError('unexpected_quality_engine_path')
            self.engine=real_class(path,*args,**kwargs)
        def run(self, feeds):
            index=len(summary['quality_calls'])
            if index>=len(samples):raise ValueError('quality_inference_budget_exceeded')
            # Count attempts before inference. Otherwise an exception on one
            # image would shift the identities of all subsequent captures.
            row={'ordinal':index,'image_id':samples[index]['image_id'],'status':'attempted'}
            summary['quality_calls'].append(row)
            try:
                if len(feeds)!=1 or list(feeds[0].shape)!=inp['shape'] or feeds[0].dtype!=np.uint8:
                    raise ValueError('quality_feed_violates_exact_input_contract')
                before=np.asarray(feeds[0]).copy()
                row['feed']=stats(before)
                raw=self.engine.run(feeds)
                summary['hardware_executed']=True
                values=list(raw) if isinstance(raw,(list,tuple)) else [raw]
                saved=[np.asarray(x).copy() for x in values]
                if sum(x.nbytes for x in saved)>1024*1024:raise ValueError('unexpected_full_output_size')
                save_tensors(out/f'quality_{index:03d}.npz',feed=before,**{f'output_{i:02d}':v for i,v in enumerate(saved)})
                row['input_modified_by_engine']=not np.array_equal(before,feeds[0])
                row['independent']=logits_record(saved,names,samples[index]['label_id'],harness)
                row['status']='observed'
                print(f"QUALITY {request['model_id']} {index+1}/{len(samples)}",flush=True)
                return raw
            except Exception as exc:
                row.update(status='failed',error=f'{type(exc).__name__}: {exc}')
                raise
            finally:
                write_json(out/'remote_result.json',summary)
        def __getattr__(self,name):return getattr(self.engine,name)
    quality_dir=suite/'results/smoke_quality';quality_dir.mkdir(parents=True,exist_ok=True)
    try:
        dx_engine.InferenceEngine=CaptureEngine
        runrow=dict(request['original_full_run'])
        runrow.update(validation_images='images',validation_max_images=len(samples),model_id=request['model_id'])
        semantic=bench._run_deepx_semantic_validation(suite,model,runrow,SimpleNamespace(validation_images='',benchmark_task='classification',validation_max_images=len(samples)),quality_dir)
        summary['product_quality_result']=semantic
        p=quality_dir/'classification_topk.json'
        if p.is_file():
            exported=read_json(p);write_json(out/'product_classification_topk.json',exported)
            rows=exported.get('images',[])
            if len(rows)!=len(samples):summary['errors'].append({'stage':'quality_records','error':'product_record_count_mismatch'})
            by_image={Path(pr['image']).name:pr for pr in rows}
            for row in summary['quality_calls']:
                i=row['ordinal'];expected=samples[i]
                pr=by_image.get(expected['image_id'])
                if pr is None:
                    row['export_record_missing']=True
                    continue
                row['product_quality_record']=pr
                row['export_image_matches']=Path(pr['image']).name==expected['image_id']
                row['export_label_matches']=pr.get('label_id')==expected['label_id']
                row['export_top1_matches_independent']=pr.get('top1')==row.get('independent',{}).get('top1')
                row['export_top5_set_matches_independent']=set(pr.get('top5',[]))==set(row.get('independent',{}).get('top5',[]))
        if semantic.get('status')!='ok' or semantic.get('error_count')!=0 or len(summary['quality_calls'])!=len(samples):
            summary['errors'].append({'stage':'product_quality','error':'product_quality_not_complete','detail':semantic})
    except Exception as exc:
        summary['errors'].append({'stage':'product_quality','error':f'{type(exc).__name__}: {exc}','traceback':traceback.format_exc()})
    finally:
        dx_engine.InferenceEngine=real_class
        gc.collect()
    # This is an untimed call through the normal shared input sealer/loader.
    # It is NOT a timing-runner or energy-runner acceptance test.
    engine=real_class(str(model))
    try:
        for i,sample in enumerate(samples):
            row={'ordinal':i,'image_id':sample['image_id']}
            try:
                image=suite/sample['image']
                sealed=nfi.prepare_and_seal_deepx_native_full_input(image_path=image,input_contract=contract,task='classification',
                    out_dir=out/f'sealed_{i:03d}',model=request['model_id'],setup_id=request['setup_id'],comparison_backend='deepx')
                loaded=nfi.load_sealed_deepx_native_full_input(sealed['manifest_path'],image_path=image,input_contract=contract,
                    task='classification',expected_model=request['model_id'],expected_setup_id=request['setup_id'],expected_comparison_backend='deepx')
                feed=np.ascontiguousarray(loaded['runtime_input']);before=feed.copy()
                raw=engine.run([feed]);summary['hardware_executed']=True;values=list(raw) if isinstance(raw,(list,tuple)) else [raw]
                saved=[np.asarray(x).copy() for x in values]
                if sum(x.nbytes for x in saved)>1024*1024:raise ValueError('unexpected_full_output_size')
                save_tensors(out/f'native_{i:03d}.npz',feed=before,**{f'output_{n:02d}':v for n,v in enumerate(saved)})
                row.update(status='observed',feed=stats(before),input_modified_by_engine=not np.array_equal(before,feed),
                           prediction=logits_record(saved,names,sample['label_id'],harness),sealer_roundtrip=compare(sealed['runtime_input'],loaded['runtime_input']))
                quality=out/f'quality_{i:03d}.npz'
                if quality.is_file():
                    with np.load(quality,allow_pickle=False) as q:
                        row['native_vs_quality_feed']=compare(before,q['feed'])
                        if len(saved)==1:row['native_vs_quality_output']=compare(saved[0],q['output_00'])
                print(f"NATIVE {request['model_id']} {i+1}/{len(samples)}",flush=True)
            except Exception as exc:
                row.update(status='failed',error=f'{type(exc).__name__}: {exc}')
                summary['errors'].append({'stage':'native','image_id':sample['image_id'],'error':row['error']})
            summary['native_calls'].append(row);write_json(out/'remote_result.json',summary)
        summary['sentinel_repeats']=[]
        if samples and not summary['errors']:
            # The first original sample is fixed before any result. Three extra
            # calls own their copies; a reused output buffer cannot rewrite them.
            sentinel=samples[0]; image=suite/sentinel['image']
            feed,_,_=nfi._prepare_tensor(image,shape=inp['shape'],dtype=np.dtype(inp['dtype']),layout=inp['layout'],task='classification',normalization=inp['normalization'],preprocess_mode=inp['preprocess_mode'],pad_value=inp['letterbox_pad_value'])
            first=None
            for repeat in range(3):
                raw=engine.run([feed]);summary['hardware_executed']=True
                saved=[np.asarray(v).copy() for v in (raw if isinstance(raw,(list,tuple)) else [raw])]
                prediction=logits_record(saved,names,sentinel['label_id'],harness)
                save_tensors(out/f'sentinel_{repeat:03d}.npz',feed=feed,**{f'output_{i:02d}':v for i,v in enumerate(saved)})
                if first is None:first=[v.copy() for v in saved]
                summary['sentinel_repeats'].append({'image_id':sentinel['image_id'],'repeat':repeat,'prediction':prediction,'against_first':[compare(x,y) for x,y in zip(first,saved)]})
    finally:
        del engine;gc.collect()
    try:
        summary['calibration']=calibration_replay(suite,request,out,bench)
    except Exception as exc:
        summary['calibration']={'status':'failed','error':f'{type(exc).__name__}: {exc}'}
    if request.get('split') and request['split'].get('status')=='ready':
        split=request['split'];dx=suite/'split/model.dxnn'
        engine=None
        try:
            if digest(dx)!=split['sha256']:raise ValueError('staged_part1_sha256_mismatch')
            engine=real_class(str(dx)); sc=read_json(suite/'split/output_contract.json');si=sc['input'];observations=[]
            boundary_names=split.get('output_names')
            if not isinstance(boundary_names,list) or len(boundary_names)!=1 or not isinstance(boundary_names[0],str) or not boundary_names[0].strip() or boundary_names!=[v.get('name') for v in sc.get('outputs',[])]:
                raise ValueError('part1_single_named_boundary_required')
            for i,sample in enumerate(samples):
                feed,_,_=nfi._prepare_tensor(suite/sample['image'],shape=si['shape'],dtype=np.dtype(si['dtype']),layout=si['layout'],
                    task='classification',normalization=si['normalization'],preprocess_mode=si['preprocess_mode'],pad_value=si['letterbox_pad_value'])
                feed_before=feed.copy()
                raw=engine.run([feed]);vs=list(raw) if isinstance(raw,(list,tuple)) else [raw]
                arrs=[np.asarray(x).copy() for x in vs]
                if len(arrs)!=len(boundary_names):raise ValueError('part1_output_count_binding_mismatch')
                if any(not np.issubdtype(x.dtype,np.floating) or not np.isfinite(x).all() for x in arrs):
                    raise ValueError('part1_finite_floating_boundary_required')
                if sum(x.nbytes for x in arrs)>16*1024*1024:raise ValueError('part1_output_exceeds_diagnostic_budget')
                save_tensors(out/f'part1_{i:03d}.npz',feed=feed_before,**{f'output_{n:02d}':v for n,v in enumerate(arrs)})
                observations.append({'image_id':sample['image_id'],'outputs':[stats(v) for v in arrs], 'output_names':split['output_names'], 'input_modified_by_engine':not np.array_equal(feed_before,feed)})
            summary['split']={'status':'observed','rows':observations,'scope':'DeepX_part1_only; host_tail_control_is_separate'}
        except Exception as exc:summary['split']={'status':'failed','error':f'{type(exc).__name__}: {exc}'}
        finally:
            if engine is not None:del engine
            gc.collect()
    else:summary['split']=request.get('split',{'status':'not_evaluated'})
    from trt_control import run_trt_control
    summary['trt_full_control']=run_trt_control(request,stage,out)
    summary['optional_trt_control_missing']=summary['trt_full_control'].get('status')!='observed'
    checks=[]
    for row in summary['quality_calls']:
        checks.extend(v for k,v in row.items() if k.startswith('export_') and isinstance(v,bool))
        checks.append(row.get('input_modified_by_engine') is False)
    for row in summary['native_calls']:
        checks.extend(row.get(k,{}).get('exact_equal') is True for k in ('native_vs_quality_feed','native_vs_quality_output','sealer_roundtrip'))
        checks.append(row.get('input_modified_by_engine') is False)
    if checks and not all(checks):summary['errors'].append({'stage':'pipeline_parity','error':'observed_production_feed_output_or_record_mismatch'})
    summary['collection_status']='complete' if len(summary['native_calls'])==len(samples) and len(summary['quality_calls'])==len(samples) else 'incomplete'
    summary['pipeline_consistency']='consistent_observed_scope' if checks and all(checks) and not summary['errors'] else 'failed_or_incomplete'
    summary['root_cause_status']='compiled_model_difference_subcause_unresolved' if summary['pipeline_consistency']=='consistent_observed_scope' else 'pipeline_mismatch_or_missing_evidence'
    summary['status']='complete' if not summary['errors'] else 'partial'
    summary['optional_calibration_or_split_missing']=any(summary.get(k,{}).get('status')!='observed' for k in ['calibration','split'])
    write_json(out/'remote_result.json',summary)
    return summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--request',required=True,type=Path);p.add_argument('--runtime',action='store_true')
    a=p.parse_args();a.request=a.request.resolve();stage=a.request.parent;out=stage/'results';out.mkdir(exist_ok=True)
    try:
        request=read_json(a.request)
        if request.get('schema')!=SCHEMA:raise ValueError('wrong_request_schema')
        if not a.runtime:
            py=Path(request['runtime_venv']).expanduser()/'bin/python'
            if not py.is_file():raise ValueError('recorded_runtime_venv_missing:'+str(py))
            os.execve(str(py),[str(py),'-I','-B',str(Path(__file__).resolve()),'--request',str(a.request),'--runtime'],clean_env(stage))
        result=run(request,stage,out)
        print('REMOTE_EVIDENCE_STATUS='+result['status'],flush=True)
        return 0 if result['status']=='complete' else 2
    except Exception as exc:
        # Preserve incremental observations after a late failure.
        f=out/'remote_result.json';r=read_json(f) if f.is_file() else {'schema':SCHEMA,'request_nonce':locals().get('request',{}).get('nonce')}
        r.update(status='failed',fatal_error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc());write_json(f,r)
        traceback.print_exc();return 2

if __name__=='__main__':raise SystemExit(main())
