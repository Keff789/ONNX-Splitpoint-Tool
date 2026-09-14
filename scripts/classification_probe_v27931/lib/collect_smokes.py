#!/usr/bin/env python3
"""Collect bounded, standalone DeepX pre/postprocessing evidence."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import traceback
import uuid
import zipfile
sys.path.insert(0,str(Path(__file__).resolve().parent))
from smoke_common import *
BUNDLE=Path(__file__).resolve().parents[1]


@contextmanager
def workflow_gate(path):
    path=Path(path)
    # Use the very same inode as the tool, never unlink/truncate lock metadata.
    path.parent.mkdir(parents=True,exist_ok=True)
    flags=os.O_CREAT|os.O_RDWR|getattr(os,'O_NOFOLLOW',0)
    fd=os.open(path,flags,0o600)
    try:
        try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('workflow_or_platform_operation_active; close running measurements first')
        yield
    finally:
        fcntl.flock(fd,fcntl.LOCK_UN);os.close(fd)


def select_target(run):
    matches=[x for x in read_json(run/'hardware_matrix.json')['hardware_targets'] if x.get('accelerator')=='deepx_m1' and x.get('enabled',True)]
    if len(matches)!=1:raise ValueError('DeepX_hardware_target_not_unique')
    target=matches[0];rt=target.get('runtime') or target['remote'];host,user=str(rt.get('host','')),str(rt.get('user',''))
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.:-]*',host) or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_-]*',user):
        raise ValueError('unsafe_ssh_target')
    port=int(rt.get('port',22))
    if not 1<=port<=65535:raise ValueError('invalid_ssh_port')
    if rt.get('ssh_extra_args'):raise ValueError('nonempty_ssh_extra_args_need_explicit_review')
    venv=str((target.get('build_environment') or {}).get('runtime_venv') or '')
    if not venv:
        args=shlex.split(rt.get('remote_venv',''))
        if len(args)!=2 or args[0] not in {'source','.'} or not args[1].endswith('/bin/activate'):raise ValueError('runtime_venv_unresolved')
        venv=args[1][:-13]
    if not venv.startswith(('/','~/')) or any(c in venv for c in '\r\n\0'):raise ValueError('invalid_venv_path')
    return target,{'host':host,'user':user,'port':port,'runtime_venv':venv}


def sample_rows(manifest):
    payload=read_json(manifest)
    records=payload.get('samples') if isinstance(payload.get('samples'),list) else payload.get('items')
    if not isinstance(records,list):raise ValueError('unsupported_sample_manifest:'+str(manifest))
    root=Path(str(payload.get('root') or manifest.parent)).expanduser()
    if not root.is_absolute():root=manifest.parent/root
    labels={}
    lm=payload.get('labels') or {}
    if isinstance(lm,dict) and lm.get('path'):
        lp=Path(lm['path']).expanduser();lp=lp if lp.is_absolute() else manifest.parent/lp
        if lp.is_file():
            for i,line in enumerate(lp.read_text().splitlines()):
                token=line.strip().split(' ',1)[0]
                if re.fullmatch('n[0-9]+',token):labels[token]=i
    rows={}
    for row in records:
        if not isinstance(row,dict):raise ValueError('sample_record_not_object')
        raw=row.get('image') or row.get('relative_path') or row.get('path') or ''
        if not raw:raise ValueError('sample_image_path_missing')
        p=Path(raw).expanduser();p=p if p.is_absolute() else root/p
        image_id=p.name
        if image_id in rows:raise ValueError('duplicate_image_basename_in_manifest:'+image_id)
        label=row.get('label_id')
        if label is None:label=labels.get(str(row.get('class_name') or p.parent.name))
        rows[image_id]={'path':p,'label_id':label,'declared_sha256':row.get('source_sha256') or row.get('sha256') or '', 'raw_record':row}
    return rows


def select_samples(paths, wanted, require_labels=True):
    diagnostics=[]
    for p in dict.fromkeys(Path(x).expanduser() for x in paths if x):
        if not p.is_file():diagnostics.append({'path':str(p),'status':'missing'});continue
        try:
            rows=sample_rows(p);chosen=[]
            if wanted is not None and len(wanted)!=len(set(wanted)):raise ValueError('duplicate_selected_image_id')
            ids=wanted if wanted is not None else list(rows)[:8]
            for image_id in ids:
                if image_id not in rows:raise ValueError('selected_sample_not_in_manifest:'+image_id)
                row=rows[image_id];source=row['path']
                if not source.is_file():raise FileNotFoundError('selected_image_missing:'+str(source))
                label=row['label_id']
                if label is not None and (isinstance(label,bool) or not isinstance(label,int) or not 0<=label<1000):raise ValueError('label_not_explicit_zero_based_1000_class_index')
                if require_labels and label is None:raise ValueError('selected_label_missing:'+image_id)
                actual=digest(source)
                if row['declared_sha256'] and actual!=sha_token(row['declared_sha256']):raise ValueError('selected_image_digest_mismatch:'+image_id)
                chosen.append({'path':str(source.resolve()),'image_id':image_id,'label_id':label,'sha256':actual,
                               'original_content_hash_verified':bool(row['declared_sha256'])})
            if not chosen:raise ValueError('empty_sample_selection')
            return chosen,{'path':str(p),'sha256':digest(p),'status':'resolved','selection_policy':'first_N_in_original_quality_request' if wanted is not None else 'first_8_manifest_records'}
        except Exception as exc:
            # A corrupt identity/label is not a missing-path fallback case.
            if not isinstance(exc,FileNotFoundError):
                raise
            diagnostics.append({'path':str(p),'error':f'{type(exc).__name__}: {exc}'})
    raise ValueError('sample_resolution_failed:'+json.dumps(diagnostics))


def copy_source_snapshot(suite_original, tool, stage_suite):
    """Always execute the installed release; historical suite bytes are evidence only."""
    refs=read_json(BUNDLE/'product_sources.json');rows=[]
    for row in refs:
        rel=row['relative_path']; found=Path(tool)/row['original_source']
        if not found.is_file():raise FileNotFoundError('installed_product_source_missing:'+str(found))
        dest=stage_suite/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(found,dest)
        rows.append({'relative_path':rel,'origin':'installed_tool','source_path':str(found),
                     'sha256':digest(dest),'historical_v30_sha256':row['sha256'],
                     'matches_v30_reference':digest(dest)==row['sha256']})
    return rows


def prepare_model(run, model, target, remote, args, output):
    suite=run/f'models/{model}/benchmark_set/legacy_suite';plan=read_json(suite/'benchmark_plan.json')
    matches=[r for r in plan['runs'] if r['id']=='deepx_m1_full']
    if len(matches)!=1:raise ValueError('original_full_run_not_unique')
    fullrow=matches[0]
    if fullrow.get('setup_id')!=target['id']:raise ValueError('full_plan_setup_mismatch')
    contract_path=suite/'deepx/deepx_m1/full/output_contract.json';contract=read_json(contract_path)
    if contract.get('model_id')!=model or contract.get('source_model_input',{}).get('layout')!='NCHW':raise ValueError('source_model_identity_or_layout_mismatch')
    source_input_name=contract['source_model_input'].get('name')
    if not isinstance(source_input_name,str) or not source_input_name.strip():raise ValueError('source_input_name_binding_missing')
    inp=contract['input'];hw=inp['shape'][:2]
    if inp['layout']!='HWC' or inp['shape']!=[224,224,3] or inp['dtype']!='uint8' or inp['preprocess_mode']!='resize':raise ValueError('unsupported_classification_runtime_contract')
    if not contract.get('build_onnx_sha256'):raise ValueError('build_onnx_identity_missing')
    meta=read_json(run/f'models/{model}/model_manifest.json')
    onnx,onnx_checked=exact_artifact([suite/f'models/{model}.onnx',meta.get('resolved_path'),meta.get('file',{}).get('path')],contract['source_onnx_sha256'])
    dxnn,dx_checked=exact_artifact([contract_path.parent/'model.dxnn',contract.get('suite_artifact_path'),contract.get('artifact_path')],contract.get('suite_artifact_sha256') or contract.get('artifact_sha256'))
    req_path=run/f'models/{model}/benchmark_results/quality_inputs/{target["id"]}/results/deepx_m1_full/task_quality_inputs/full_request.json'
    historical=read_json(req_path);ids=historical['expected_image_ids']
    if len(ids)!=len(set(ids)) or len(ids)<args.cpu_samples:raise ValueError('invalid_original_quality_id_set')
    selected,manifest=select_samples([suite/fullrow['validation_images']/'manifest.json',run/'campaign/inputs/dataset_classification_validation.json',fullrow.get('validation_manifest'),plan.get('campaign',{}).get('dataset_manifests',{}).get('classification',{}).get('validation')],ids[:args.cpu_samples])
    stage=output/'_stage';stage_suite=stage/'suite';stage_suite.mkdir(parents=True)
    provenance=copy_source_snapshot(suite,args.tool_dir,stage_suite)
    for src,rel in [(contract_path,'deepx/deepx_m1/full/output_contract.json'),(suite/'output_contracts.json','output_contracts.json')]:
        dst=stage_suite/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
    shutil.copyfile(dxnn,stage_suite/'deepx/deepx_m1/full/model.dxnn')
    calibration=[];cal_cfg={};split={'status':'not_evaluated','reason':'no_single_bound_split_found'}
    native=run/f'native_producers/deepx/{model}/benchmark_set'
    split_contracts=list(native.glob('b*/deepx/deepx_m1/part1/output_contract.json'))
    if len(split_contracts)==1:
        scp=split_contracts[0];sc=read_json(scp)
        cfgp=scp.with_name('config_deepx.json')
        if cfgp.is_file():cal_cfg=read_json(cfgp)
        statusp=scp.with_name('deepx_part1_artifact_status.json');status=read_json(statusp) if statusp.is_file() else {}
        try:
            dp,checked=exact_artifact([scp.with_name('model.dxnn'),status.get('cache_lookup',{}).get('artifact')],sc['artifact_sha256'])
            split={'status':'ready','case_id':sc['case_id'],'sha256':sc['artifact_sha256'],'local_dxnn':str(dp),
                   'output_names':[x['name'] for x in sc['outputs']],'checked':checked,'source_onnx_sha256':sc['source_onnx_sha256']}
            (stage_suite/'split').mkdir();shutil.copyfile(dp,stage_suite/'split/model.dxnn');shutil.copyfile(scp,stage_suite/'split/output_contract.json')
            # Tail execution will remain a separate CPU diagnostic, not TRT evidence.
            case=scp.parents[3];sm=case/'split_manifest.json'
            if sm.is_file():
                s=read_json(sm);shutil.copyfile(sm,stage_suite/'split/split_manifest.json')
                p1=case/s['part1'];p2=case/s['part2']
                if p1.is_file() and digest(p1)==sc['source_onnx_sha256'] and p2.is_file():
                    split.update(local_part1=str(p1.resolve()),local_part2=str(p2.resolve()),part2_sha256_observed=digest(p2))
        except Exception as exc:split={'status':'unavailable','reason':f'{type(exc).__name__}: {exc}'}
    try:
        cal_paths=[run/'campaign/inputs/dataset_classification_calibration.json',plan.get('campaign',{}).get('dataset_manifests',{}).get('classification',{}).get('calibration')]
        calibration,cal_manifest=select_samples(cal_paths,None,require_labels=False)
    except Exception as exc:cal_manifest={'status':'unavailable','reason':f'{type(exc).__name__}: {exc}'}
    if sum(Path(s['path']).stat().st_size for s in selected+calibration)>32*1024*1024:
        raise ValueError('selected_original_image_payload_budget_exceeded')
    for sample in selected:
        dst=stage_suite/'images'/sample['image_id'];dst.parent.mkdir(exist_ok=True);shutil.copyfile(sample['path'],dst);sample['image']='images/'+sample['image_id']
    for sample in calibration:
        dst=stage_suite/'calibration'/sample['image_id'];dst.parent.mkdir(exist_ok=True);shutil.copyfile(sample['path'],dst);sample['image']='calibration/'+sample['image_id']
    write_json(stage_suite/'images/manifest.json',{'samples':[{'image':s['image_id'],'label_id':s['label_id']} for s in selected[:args.hardware_samples]]})
    export=meta.get('profile_entry',{}).get('export_metadata') or {}
    profile_mode='unresolved';profile_path=run/'profile.yaml'
    if profile_path.is_file():
        import yaml
        profile=yaml.safe_load(profile_path.read_text())
        if not isinstance(profile,dict):raise ValueError('profile_not_object')
        profile_mode=str((profile.get('deepx_build') or {}).get('classification_preprocessing') or 'unresolved')
    names=[str(x['name']) for x in contract['outputs']]
    request={'schema':SCHEMA,'nonce':uuid.uuid4().hex,'original_run_id':run.name,'model_id':model,'setup_id':target['id'],
             'tool_dir':str(args.tool_dir),'runtime_venv':remote['runtime_venv'],'original_full_run':fullrow,'input_hw':hw,'local_onnx':str(onnx),
             'expected_onnx_sha256':sha_token(contract['source_onnx_sha256']),'expected_dxnn_sha256':sha_token(contract.get('suite_artifact_sha256') or contract['artifact_sha256']),
             'source_input_name':source_input_name,'onnx_output_names':names,'samples':selected,'hardware_samples':args.hardware_samples,
             'stage':getattr(args,'stage','hardware-paired'),'classification_mode':profile_mode,'contract_classification_mode':contract.get('classification_preprocessing','unresolved'),'build_onnx_sha256':sha_token(contract['build_onnx_sha256']),
             'historical_calibration_evidence':'historical_calibration_evidence_missing',
             'calibration_samples':calibration,'calibration_config':cal_cfg,'export_preprocess':export.get('preprocess',{}),
             'split':split,'diagnostic_only':True,'counts_as_benchmark':False}
    shutil.copytree(BUNDLE/'lib',stage/'lib',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ('deepx_full_workflow_smoke_worker_v27930.py','deepx_full_output_probe_worker_v27930.py'):
        shutil.copyfile(args.tool_dir/'scripts'/name,stage/'lib'/name)
    from trt_control import resolve_trt_control
    try:
        request['trt_control']=resolve_trt_control(run,model,target['id'],request['expected_onnx_sha256'],args.tool_dir)
    except Exception as exc:
        request['trt_control']={'status':'invalid_binding','reason':f'{type(exc).__name__}: {exc}','hardware_executed':False}
    write_json(stage/'request.json',request)
    write_json(output/'evidence_request.json',request)
    write_json(output/'resolution.json',{'onnx_candidates':onnx_checked,'dxnn_candidates':dx_checked,'validation_manifest':manifest,
               'profile_classification_mode':profile_mode,'profile_sha256':digest(profile_path) if profile_path.is_file() else None,
               'calibration_manifest':cal_manifest,'source_provenance':provenance,'source_contract':contract,'export_metadata':export})
    # Keep the actual product bytes in the evidence, not only asserted identities.
    for row in provenance:
        src=stage_suite/row['relative_path'];dest=output/'product_sources'/row['relative_path'];dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dest)
    # Keep selected original images for later replay; never embed the large model binaries.
    for sample in selected+calibration:
        dst=output/'selected_inputs'/sample['image'];dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(stage_suite/sample['image'],dst)
    shutil.copyfile(suite/'output_contracts.json',output/'original_output_contracts.json')
    shutil.copyfile(req_path,output/'original_full_quality_request.json')
    shutil.copyfile(contract_path,output/'original_full_output_contract.json')
    if cal_cfg:write_json(output/'original_part1_config.json',cal_cfg)
    return stage,request


def ssh_base(remote):
    return ['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=10','-o','ServerAliveCountMax=3','-p',str(remote['port']),remote['user']+'@'+remote['host']]


def scp_base(remote):
    return ['scp','-q','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=10','-o','ServerAliveCountMax=3','-P',str(remote['port'])]


def remote_path_ok(value):return re.fullmatch(r'/tmp/onnx-deepx-classification-v27931-[A-Za-z0-9]{10}',value) is not None


def run_remote(stage,remote,out):
    dest=remote['user']+'@'+remote['host'];remote_dir='';dispatched=False;cleanup_confirmed=False;result={}
    request=read_json(stage/'request.json')
    try:
        made=subprocess.run(ssh_base(remote)+['mktemp -d /tmp/onnx-deepx-classification-v27931-XXXXXXXXXX'],text=True,capture_output=True,timeout=30,check=True)
        remote_dir=made.stdout.strip()
        if not remote_path_ok(remote_dir):raise ValueError('remote_temp_directory_invalid')
        print('REMOTE_PROBE_DIR='+remote_dir,flush=True)
        cp=run_bounded(scp_base(remote)+['-r',str(stage/'suite'),str(stage/'lib'),str(stage/'request.json'),dest+':'+remote_dir+'/'],out/'transfer.log',180,heartbeat='Transfer')
        if cp['returncode']!=0:raise RuntimeError('remote_transfer_failed:'+str(cp))
        dispatched=True
        cmd=shlex.join(['python3','-I','-B',remote_dir+'/lib/remote_supervisor.py','--stage',remote_dir,'--timeout','180'])
        result['execution']=run_bounded(ssh_base(remote)+[cmd],out/'remote_console.log',230,heartbeat='DeepX-Probe')
        # Return code 2 is diagnostic partial, not a reason to discard evidence.
    except Exception as exc:
        result['transport_error']=f'{type(exc).__name__}: {exc}'
    finally:
        if remote_path_ok(remote_dir):
            if dispatched:
                try:
                    stopcmd=shlex.join(['python3','-I','-B',remote_dir+'/lib/remote_supervisor.py','--stage',remote_dir,'--stop'])
                    stop=subprocess.run(ssh_base(remote)+[stopcmd],capture_output=True,text=True,timeout=25)
                    result['stop_returncode']=stop.returncode;result['stop_output']=stop.stdout+stop.stderr
                    cleanup_confirmed=stop.returncode==0
                except Exception as exc:result['stop_error']=f'{type(exc).__name__}: {exc}'
                try:
                    cp=run_bounded(scp_base(remote)+['-r',dest+':'+remote_dir+'/results',str(out/'remote')],out/'collection.log',90)
                    result['collection']=cp
                    captured=out/'remote/remote_result.json'
                    if cp['returncode']==0 and captured.is_file():
                        actual=read_json(captured)
                        result['request_binding_verified']=actual.get('request_nonce')==request['nonce']
                        if not result['request_binding_verified']:
                            result['collection_error']='remote_request_nonce_mismatch'
                except Exception as exc:result['collection_error']=f'{type(exc).__name__}: {exc}'
            else:cleanup_confirmed=True
            result['cleanup_confirmed']=cleanup_confirmed;result['remote_directory']=remote_dir
            if cleanup_confirmed and (out/'remote/remote_result.json').is_file():
                try:
                    cleanup=subprocess.run(ssh_base(remote)+[shlex.join(['rm','-rf','--',remote_dir])],text=True,capture_output=True,timeout=25)
                    result['remote_temporary_removed']=cleanup.returncode==0
                except Exception as exc:result['cleanup_error']=f'{type(exc).__name__}: {exc}'
        write_json(out/'transport.json',result)
    return result


def archive_output(output):
    archive=output.with_suffix('.zip')
    files=[p for p in sorted(output.rglob('*')) if p.is_file() and not p.is_symlink() and '_stage' not in p.relative_to(output).parts]
    total=sum(p.stat().st_size for p in files)
    if total>256*1024*1024:
        # Preserve data on disk; still return a bounded error/status/log bundle.
        files=[p for p in files if p.suffix.lower() in {'.json','.log','.md','.txt','.csv','.py'} and p.stat().st_size<16*1024*1024]
        write_json(output/'archive_limit.json',{'status':'raw_data_left_on_disk','uncompressed_bytes':total})
        files.append(output/'archive_limit.json')
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for p in files:z.write(p,p.relative_to(output))
    with zipfile.ZipFile(archive) as z:
        if z.testzip():raise ValueError('diagnostic_zip_crc_error')
    return archive


def collect_vendor_sources(venv,out):
    """No vendor import/execution: copy bounded loader source text for interpretation."""
    roots=list((Path(venv).expanduser()/'lib').glob('python*/site-packages/dx_com/dataloader'))
    rows=[];total=0
    for root in roots:
        for p in sorted(root.rglob('*.py'))[:30]:
            size=p.stat().st_size
            if size>200000 or total+size>1000000:continue
            dst=out/'vendor_loader_source'/p.relative_to(root);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dst)
            rows.append({'path':str(p),'sha256':digest(p),'size_bytes':size});total+=size
    write_json(out/'vendor_loader_source_status.json',{'status':'collected' if rows else 'not_found','files':rows,'vendor_code_imported':False})


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,default=Path.home()/'Models/EvaluationRuns'/DEFAULT_RUN)
    p.add_argument('--tool-dir',type=Path,default=Path.home()/'ONNX-Splitpoint-Tool')
    p.add_argument('--models',nargs='+',choices=(*MODELS, 'yolo26s'),default=list(MODELS))
    p.add_argument('--detection-image-ids',nargs=4,help='S6 only: four original quality-request image IDs, fixed before inference')
    p.add_argument('--cpu-samples',type=int,default=32);p.add_argument('--hardware-samples',type=int,default=16)
    p.add_argument('--cpu-python',type=Path);p.add_argument('--output-dir',type=Path,default=Path.home()/'Downloads')
    p.add_argument('--stage',choices=('contracts','inputs','cpu-paired','hardware-paired','calibration-audit','detection-control'),default='hardware-paired')
    p.add_argument('--offline-only',action='store_true');p.add_argument('--plan-only',action='store_true')
    args=p.parse_args(argv)
    if args.stage == 'detection-control':
        from detection_control import collect_detection
        return collect_detection(args)
    if 'yolo26s' in args.models or args.detection_image_ids:
        p.error('YOLO26s/fixed detection IDs require --stage detection-control')
    if args.stage in {'contracts','inputs','cpu-paired'}:args.offline_only=True
    if not 1<=args.hardware_samples<=16 or not args.hardware_samples<=args.cpu_samples<=32:p.error('limits: 1 <= hardware <=16 and hardware <= CPU <=32')
    if len(args.models)!=len(set(args.models)):p.error('duplicate model requested')
    args.tool_dir=args.tool_dir.expanduser().resolve();args.run_dir=args.run_dir.expanduser().resolve()
    args.output_dir=args.output_dir.expanduser().resolve()
    if args.output_dir.is_relative_to(args.run_dir) or args.output_dir.is_relative_to(args.tool_dir):
        p.error('diagnostic output must be outside the original run and installed source')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    output=Path(tempfile.mkdtemp(prefix='deepx_classification_v27931_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_',dir=args.output_dir))
    summary={'schema':SCHEMA,'status':'started','output_dir':str(output),'run_dir':str(args.run_dir),'models':[],
             'hardware_executed':False,'collection_status':'pending','pipeline_consistency':'not_evaluated','root_cause_status':'hardware_pending',
             'mode':'plan_only' if args.plan_only else ('offline_only' if args.offline_only else 'local_CPU_and_remote_DeepX'),
             'counts_as_benchmark':False,'model_acceptance':'NOT_EVALUATED','compiler_invoked':False,
             'tool_installation_modified':False,'quality_thresholds_modified':False}
    py=args.cpu_python or args.tool_dir/'.venv/bin/python'
    print('EVIDENCE_DIRECTORY='+str(output),flush=True)
    print('MODE=diagnostic_only; no install; no compile; no energy; sequential',flush=True)
    try:
        with workflow_gate(Path.home()/'.onnx_splitpoint_tool/locks/workflow_platform_interlock.lock'):
            target,remote=select_target(args.run_dir);summary['setup_id']=target['id'];summary['remote']=remote
            for executable in (() if args.offline_only or args.plan_only else ('ssh','scp')):
                if not shutil.which(executable):raise ValueError('required_executable_missing:'+executable)
            collect_vendor_sources(target.get('build_environment',{}).get('compiler_venv','/nonexistent'),output)
            for model in args.models:
                mdir=output/model;mdir.mkdir();row={'model_id':model,'status':'started'};summary['models'].append(row)
                print('\nMODEL_START='+model,flush=True)
                try:
                    stage,request=prepare_model(args.run_dir,model,target,remote,args,mdir)
                    print('LOCAL_ASSETS=verified; CPU='+str(len(request['samples']))+'; hardware='+str(args.hardware_samples),flush=True)
                    if args.plan_only:row['status']='planned';continue
                    if not py.is_file():raise ValueError('existing_CPU_interpreter_missing:'+str(py))
                    row['cpu_process']=run_bounded([str(py),'-I','-B',str(stage/'lib/cpu_worker.py'),'--request',str(stage/'request.json'),'--out',str(mdir/'cpu')],mdir/'cpu_console.log',180,env=clean_env(mdir),heartbeat='CPU-Probe')
                    if not args.offline_only:
                        row['remote_transport']=run_remote(stage,remote,mdir)
                        if not row['remote_transport'].get('cleanup_confirmed',False):
                            row['status']='remote_cleanup_unconfirmed';raise RuntimeError('REMOTE_CLEANUP_UNCONFIRMED; no further model is started')
                    row['analysis_process']=run_bounded([str(py),'-I','-B',str(stage/'lib/analyze_evidence.py'),'--model-dir',str(mdir)],mdir/'analysis.log',120,env=clean_env(mdir))
                    cp=read_json(mdir/'cpu/cpu_result.json');rp=read_json(mdir/'remote/remote_result.json') if (mdir/'remote/remote_result.json').is_file() else {}
                    row['cpu_status']=cp['status'];row['remote_status']=rp.get('status','not_run');row['hardware_executed']=rp.get('hardware_executed') is True
                    row['source_build_semantics']=cp.get('source_build_semantics',{})
                    row['pipeline_consistency']='failed_source_build_semantics' if row['source_build_semantics'].get('status')=='mismatch' else rp.get('pipeline_consistency','not_evaluated')
                    row['status']='evidence_collected' if cp['status']=='complete' and cp.get('request_nonce')==request['nonce'] and (args.offline_only or (rp.get('status')=='complete' and row['remote_transport'].get('request_binding_verified') is True)) and row['analysis_process']['returncode']==0 else 'partial_evidence'
                except Exception as exc:
                    row.update(error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
                    if row['status']=='started':row['status']='failed'
                    write_json(mdir/'error.json',row)
                    print('MODEL_ERROR='+row['error'],flush=True)
                finally:
                    # Only self-created staging files; existing ONNX/DXNN paths
                    # were read directly or copied, never moved or unlinked.
                    staged=mdir/'_stage'
                    if staged.is_dir():shutil.rmtree(staged)
                    write_json(output/'collection_summary.json',summary)
                print('MODEL_STATUS='+model+':'+row['status'],flush=True)
                if row['status']=='remote_cleanup_unconfirmed':break
            statuses=[r['status'] for r in summary['models']]
            summary['status']='planned' if args.plan_only and all(s=='planned' for s in statuses) else ('evidence_collected' if len(statuses)==len(args.models) and all(s=='evidence_collected' for s in statuses) else 'partial_evidence')
            if summary['status']=='evidence_collected' and any(r.get('pipeline_consistency')=='failed_source_build_semantics' for r in summary['models']):
                summary['status']='pipeline_mismatch'
    except KeyboardInterrupt:summary.update(status='interrupted',error='user_interrupted; inspect transport cleanup status before another run')
    except Exception as exc:summary.update(status='failed',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
    finally:
        summary['collection_status']='complete' if summary['status'] in {'planned','evidence_collected','pipeline_mismatch'} else 'incomplete'
        scopes=[r.get('pipeline_consistency','not_evaluated') for r in summary['models']]
        summary['pipeline_consistency']=aggregate_pipeline_consistency(scopes,plan_only=args.plan_only)
        summary['hardware_executed']=any(r.get('hardware_executed') is True for r in summary['models'])
        write_json(output/'collection_summary.json',summary)
        archive=archive_output(output)
        print('\nSMOKE_EVIDENCE_STATUS='+summary['status'],flush=True)
        print('MODEL_ACCEPTANCE=NOT_EVALUATED_BY_DIAGNOSTIC',flush=True)
        print('DIAGNOSTIC_ZIP='+str(archive),flush=True)
    return 0 if summary['status'] in {'evidence_collected','planned'} else 2

def aggregate_pipeline_consistency(scopes,*,plan_only=False):
    """Collection success does not establish unobserved runtime parity."""
    if plan_only:return 'not_evaluated'
    if 'failed_source_build_semantics' in scopes:return 'failed_source_build_semantics'
    if any(s.startswith('failed') for s in scopes):return 'failed_or_incomplete'
    if scopes and all(s=='consistent_observed_scope' for s in scopes):return 'consistent_observed_scope'
    if scopes and all(s=='reconstructed_scope_only' for s in scopes):return 'reconstructed_scope_only'
    return 'not_evaluated'


if __name__=='__main__':raise SystemExit(main())
