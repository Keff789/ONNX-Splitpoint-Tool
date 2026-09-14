#!/usr/bin/env python3
"""Pair observations by image identity; never turn smoke accuracy into acceptance."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import traceback
sys.path.insert(0,str(Path(__file__).resolve().parent))
from smoke_common import *


def aggregate(rows):
    return {'sample_count':len(rows),'top1_accuracy':sum(bool(r['top1_correct']) for r in rows)/len(rows) if rows else None,
            'top5_accuracy':sum(bool(r['top5_correct']) for r in rows)/len(rows) if rows else None}


def feature_for_input(raw, target):
    import numpy as np
    a=np.asarray(raw);target=tuple(target)
    if any(not isinstance(x,int) or x<=0 for x in target):raise ValueError('part2_dynamic_shape_not_supported')
    if a.shape==target:return np.ascontiguousarray(a),'identity'
    # Explicit singleton removal/addition for global-pooled feature vectors.
    if np.squeeze(a).shape==np.squeeze(np.empty(target)).shape and a.size==int(np.prod(target)) and sum(x>1 for x in a.shape)<=1:
        return np.ascontiguousarray(a.reshape(target)),'singleton_feature_vector_reshape'
    if a.ndim==4 and tuple(a.transpose(0,3,1,2).shape)==target:
        return np.ascontiguousarray(a.transpose(0,3,1,2)),'NHWC_to_NCHW'
    if a.ndim==3 and tuple(a.transpose(2,0,1)[None].shape)==target:
        return np.ascontiguousarray(a.transpose(2,0,1)[None]),'HWC_to_NCHW'
    raise ValueError('part1_to_CPU_tail_layout_not_uniquely_supported:'+str(a.shape)+'->'+str(target))


def split_control(request, directory):
    import numpy as np
    split=request.get('split') or {}
    if not split.get('local_part1') or not split.get('local_part2'):
        return {'status':'not_evaluated','reason':'exact_local_split_source_paths_unavailable'}
    if not (directory/'remote/part1_000.npz').is_file():return {'status':'not_evaluated','reason':'no_hardware_part1_outputs'}
    try:
        import onnxruntime as ort
        p1,p2=Path(split['local_part1']),Path(split['local_part2'])
        if digest(p1)!=split['source_onnx_sha256'] or digest(p2)!=split['part2_sha256_observed']:raise ValueError('split_source_changed')
        opts=ort.SessionOptions();opts.intra_op_num_threads=2;opts.inter_op_num_threads=1
        s1=ort.InferenceSession(str(p1),sess_options=opts,providers=['CPUExecutionProvider'])
        s2=ort.InferenceSession(str(p2),sess_options=opts,providers=['CPUExecutionProvider'])
        if len(s1.get_inputs())!=1 or len(s1.get_outputs())!=1 or len(s2.get_inputs())!=1:raise ValueError('only_single_boundary_split_supported')
        bname=s2.get_inputs()[0].name
        if split['output_names']!=[s1.get_outputs()[0].name] or split['output_names']!=[bname]:raise ValueError('split_boundary_names_mismatch')
        outnames=[v.name for v in s2.get_outputs()];rows=[]
        for i,sample in enumerate(request['samples'][:request['hardware_samples']]):
            with np.load(directory/f'remote/part1_{i:03d}.npz',allow_pickle=False) as ar:
                if sorted(k for k in ar.files if k.startswith('output_'))!=['output_00']:
                    raise ValueError('part1_capture_output_count_binding_mismatch')
                observed=ar['output_00'].copy();feed=ar['feed'].copy()
            if not np.issubdtype(observed.dtype,np.floating) or not np.isfinite(observed).all():raise ValueError('part1_finite_floating_boundary_required')
            scale=np.ascontiguousarray(feed.transpose(2,0,1)[None].astype(np.float32)/np.float32(255))
            mean=(scale-np.array([.485,.456,.406],dtype=np.float32).reshape(1,3,1,1))/np.array([.229,.224,.225],dtype=np.float32).reshape(1,3,1,1)
            feature,mapping=feature_for_input(observed,s2.get_inputs()[0].shape)
            if s2.get_inputs()[0].type!='tensor(float)' or feature.dtype!=np.float32:raise ValueError('CPU_tail_requires_float32_no_implicit_cast')
            tail=s2.run(None,{bname:feature});rec=logits_record(tail,outnames,sample['label_id'])
            row={'image_id':sample['image_id'],'layout_mapping':mapping,'deepx_part1_CPU_tail':rec,'float_controls':{}}
            arrays={'deepx_part1_CPU_tail':tail[0]}
            for arm,data in [('scale_only',scale),('imagenet_mean_std',mean)]:
                base=s1.run(None,{s1.get_inputs()[0].name:data});floattail=s2.run(None,{bname:base[0]})
                row['float_controls'][arm]={'interface_comparison':compare(feature,base[0]),'tail_comparison':compare(tail[0],floattail[0]),
                    'prediction':logits_record(floattail,outnames,sample['label_id'])}
                with np.load(directory/f'cpu/cpu_{i:03d}.npz',allow_pickle=False) as full:
                    key='output_'+arm+'_00'
                    if key in full:row['float_controls'][arm]['float_split_vs_original_full']=compare(floattail[0],full[key])
                arrays[arm]=floattail[0]
            save_tensors(directory/f'split_control_{i:03d}.npz',**arrays);rows.append(row)
        return {'status':'observed','rows':rows,'scope':'DeepX_part1_plus_existing_FLOAT_ONNX_CPU_tail; NOT TensorRT_pipeline_acceptance',
                'part2_hash_scope':'observed_current_declared_run_file; no historical_hash_claim'}
    except Exception as exc:return {'status':'failed','error':f'{type(exc).__name__}: {exc}','traceback':traceback.format_exc()}


def run(directory):
    import numpy as np
    req=read_json(directory/'evidence_request.json')
    cp=read_json(directory/'cpu/cpu_result.json') if (directory/'cpu/cpu_result.json').is_file() else {}
    rp=read_json(directory/'remote/remote_result.json') if (directory/'remote/remote_result.json').is_file() else {}
    out={'schema':SCHEMA,'model_id':req['model_id'],'model_acceptance':'NOT_EVALUATED','counts_as_benchmark':False,
         'source_build_semantics':cp.get('source_build_semantics',{}), 'hardware_executed':rp.get('hardware_executed',False), 'purpose':'evidence_for_later_root_cause_review; no automatic_production_fix', 'cpu_controls':{}, 'paired_hardware':[],
         'full_runtime_vs_quality_findings':[], 'not_established':['DXNN_internal_preprocessing','quantization_as_cause','B500_quality_acceptance','FPS_or_energy','full_TensorRT_hardware_parity']}
    for arm in ('scale_only','imagenet_mean_std','canonical_reference','harness_crop256','export_metadata_geometry'):
        values=[r.get('controls',{}).get(arm,{}).get('prediction') for r in cp.get('rows',[])]
        out['cpu_controls'][arm]=aggregate([r for r in values if r is not None])
    cpu_rows={r['image_id']:r for r in cp.get('rows',[])}
    quality={r['image_id']:r for r in rp.get('quality_calls',[])}
    native={r['image_id']:r for r in rp.get('native_calls',[])}
    trt={r['image_id']:r for r in rp.get('trt_full_control',{}).get('rows',[])}
    for ordinal,sample in enumerate(req['samples'][:req['hardware_samples']]):
        image_id=sample['image_id'];c=cpu_rows.get(image_id,{});n=native.get(image_id,{});q=quality.get(image_id,{})
        row={'image_id':image_id,'label_id':sample['label_id'],'native_top1':n.get('prediction',{}).get('top1'),
             'quality_top1':q.get('independent',{}).get('top1'),'cpu_top1':{k:v.get('prediction',{}).get('top1') for k,v in c.get('controls',{}).items()}}
        row['trt_full_control']=trt.get(image_id,{'status':'missing_control'})
        row['native_vs_quality_feed']=n.get('native_vs_quality_feed');row['native_vs_quality_output']=n.get('native_vs_quality_output')
        row['quality_export_checks']={k:v for k,v in q.items() if k.startswith('export_')}
        p=directory/f'cpu/cpu_{ordinal:03d}.npz';d=directory/f'remote/native_{ordinal:03d}.npz'
        if p.is_file() and d.is_file():
            with np.load(p,allow_pickle=False) as cc,np.load(d,allow_pickle=False) as dd:
                row['output_comparisons']={k:compare(dd['output_00'],cc[k]) for k in cc.files if k.startswith('output_')}
        out['paired_hardware'].append(row)
    out['trt_full_control']=rp.get('trt_full_control',{'status':'not_run','hardware_executed':False})
    if out['trt_full_control'].get('status')=='observed':
        out['not_established'].remove('full_TensorRT_hardware_parity')
        out['trt_control_scope']='same_setup_exact_engine_and_bound_samples; predictions_may_differ'
    out['hardware_native']=aggregate([r['prediction'] for r in native.values() if 'prediction' in r])
    out['hardware_quality']=aggregate([r['independent'] for r in quality.values() if 'independent' in r])
    out['split_control']=split_control(req,directory)
    split_rows={r['image_id']:r for r in out['split_control'].get('rows',[])}
    full_errors={k for k,v in native.items() if v.get('prediction',{}).get('top1_correct') is False}
    split_errors={k for k,v in split_rows.items() if v.get('deepx_part1_CPU_tail',{}).get('top1_correct') is False}
    out['full_split_error_images']={'status':'compared' if split_rows else 'missing_split_control',
        'full_image_ids':sorted(native),'split_image_ids':sorted(split_rows),
        'full_error_ids':sorted(full_errors),'split_error_ids':sorted(split_errors),
        'shared_error_ids':sorted(full_errors&split_errors),'common_cause_proven':False}

    write_json(directory/'analysis.json',out)
    lines=['# DeepX Pre-/Postprocessing-Smoke: '+req['model_id'],'',
           '**Diagnose, keine Modell-/Performance-/Energieabnahme.**','',
           '## CPU-Kontrollarme auf demselben unveränderten ONNX','',
           '| Arm | Bilder | Top-1 | Top-5 |','|---|---:|---:|---:|']
    def pct(value):return 'nicht ermittelt' if value is None else f'{100*value:.2f} %'
    for k,v in out['cpu_controls'].items():lines.append(f"| {k} | {v['sample_count']} | {pct(v['top1_accuracy'])} | {pct(v['top5_accuracy'])} |")
    lines+=['','## DeepX Full','',f"Native/sealed: {out['hardware_native']['sample_count']} Bilder, Top-1 {pct(out['hardware_native']['top1_accuracy'])}.",
            f"Produktiver Quality-Pfad: {out['hardware_quality']['sample_count']} Bilder, Top-1 {pct(out['hardware_quality']['top1_accuracy'])}.",
            '', 'CPU- und Hardwarequoten haben ggf. unterschiedliche Nenner. Für die Ursachenprüfung die bildweise gepaarten Zeilen in `analysis.json` verwenden.',
            '', '## Grenzen','',
            'Die erste Auswahl folgt ausschließlich den ersten IDs des ursprünglichen B500-Auftrags, nicht den Erfolgen eines Kontrollarms. Kein neuer Referenzvertrag wird erzeugt.',
            'Der Crop-Arm und gegebenenfalls der Metadaten-Arm sind ausdrücklich alternative Eingabekontrollen. Die Kampagnenreferenz wird nicht überschrieben.',
            'Der Kalibrierungsreplay ist ein gekennzeichneter Nachbau der gespeicherten Konfigurationsfolge, kein direkt ausgeführter DX-COM-Loader.',
            'Ein optionaler Splitvergleich nutzt den vorhandenen Float-ONNX-Tail auf der CPU, nicht die TensorRT-Pipeline.',
            'Abweichungen sind Evidence, nicht automatisch ein Fehlerbeweis. Die zentrale B500-Statistik und Detectionmodelle werden hier nicht ausgeführt.',
            '', 'Rohwerte liegen in NPZ-Dateien. Paketstatus `evidence_collected` bestätigt nur das Einsammeln, nicht eine bestandene Qualität.','']
    (directory/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    return out


def main():
    p=argparse.ArgumentParser();p.add_argument('--model-dir',type=Path,required=True);a=p.parse_args()
    try:run(a.model_dir);return 0
    except Exception as exc:
        write_json(a.model_dir/'analysis_error.json',{'error':f'{type(exc).__name__}: {exc}','traceback':traceback.format_exc()});traceback.print_exc();return 2
if __name__=='__main__':raise SystemExit(main())
