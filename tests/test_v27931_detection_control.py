"""S6 uses real staging, sealed input, production decoder and record writer.

The four RGB images and vendor outputs are synthetic fixtures; no DeepX hardware
or model quality acceptance is claimed by these software integration tests.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / 'scripts/classification_probe_v27931/lib'
FIX = ROOT / 'tests/fixtures/v27931_detection_control'
sys.path.insert(0, str(LIB))
import detection_control as dc
from smoke_common import digest, read_json, write_json


def case(tmp_path):
    run = tmp_path / 'original_run'
    suite = run / 'models/yolo26s/benchmark_set/legacy_suite'
    full = suite / 'deepx/deepx_m1/full'
    full.mkdir(parents=True)
    dxnn = full / 'model.dxnn'; dxnn.write_bytes(b'SYNTHETIC_TEST_ONLY_DXNN')
    onnx = suite / 'models/yolo26s.onnx'; onnx.parent.mkdir(); onnx.write_bytes(b'SYNTHETIC_TEST_ONLY_ONNX')
    contract = read_json(FIX / 'original_full_output_contract.json')
    contract.update(source_onnx_sha256=digest(onnx), build_onnx_sha256=digest(onnx),
                    suite_artifact_sha256=digest(dxnn), artifact_sha256=digest(dxnn),
                    artifact_path=str(dxnn), suite_artifact_path=str(dxnn))
    write_json(full / 'output_contract.json', contract)
    (suite / 'output_contracts.json').write_bytes((FIX / 'original_output_contracts.json').read_bytes())
    images = suite / 'validation'; images.mkdir()
    ids = []
    for i, size in enumerate(((640,361), (320,640), (640,480), (640,640))):
        image_id = f'synthetic_{i}.png'; ids.append(image_id)
        Image.new('RGB', size, (17+i, 101+i, 223-i)).save(images / image_id)
    write_json(images / 'manifest.json', {'samples':[{'image': i, 'annotations': []} for i in ids]})
    target = {'id':'synthetic_deepx_setup', 'accelerator':'deepx_m1', 'runtime':{'host':'synthetic-host','user':'test'}, 'build_environment':{'runtime_venv':'/synthetic/venv'}}
    remote = {'host':'synthetic-host', 'user':'test', 'port':22, 'runtime_venv':'/synthetic/venv'}
    write_json(run / 'hardware_matrix.json', {'hardware_targets':[target]})
    write_json(suite / 'benchmark_plan.json', {'runs':[{'id':'deepx_m1_full', 'model_id':'yolo26s',
        'setup_id':target['id'], 'contract_path':'deepx/deepx_m1/full/output_contract.json',
        'backend':'deepx_m1', 'variant':'full', 'validation_images':'validation', 'benchmark_task':'detection'}]})
    write_json(run / 'models/yolo26s/model_manifest.json', {'resolved_path':str(onnx)})
    write_json(run / f'models/yolo26s/benchmark_results/quality_inputs/{target["id"]}/results/deepx_m1_full/task_quality_inputs/full_request.json', {'expected_image_ids':ids})
    output = tmp_path / 'diagnostic'; output.mkdir()
    args = SimpleNamespace(models=['yolo26s'], tool_dir=ROOT, detection_image_ids=None,
                           run_dir=run, output_dir=tmp_path/'collections', plan_only=False, offline_only=False)
    return SimpleNamespace(run=run,suite=suite,full=full,output=output,args=args,target=target,remote=remote,ids=ids)


def execute(c, *, bad=''):
    stage, request = dc.prepare_detection_model(c.run,c.target,c.remote,c.args,c.output)
    script = stage / 'test_synthetic_engine.py'
    script.write_text('''import json,sys,types
from pathlib import Path
import numpy as np
stage=Path(sys.argv[1]); bad=sys.argv[2]
sys.path.insert(0,str(stage/'lib'))
from smoke_common import read_json,write_json
import remote_worker
calls=[]
class Engine:
    def __init__(self,path): pass
    def run(self,feeds):
        calls.append(1)
        output=np.zeros((1,300,6),np.float32)
        output[0,0]=[10,159,40,199,.9,2]
        output[0,1]=[11,160,41,200,.8,2]
        if bad=='nan': output[0,0,4]=np.nan
        if bad=='class': output[0,0,5]=80
        if bad=='input': feeds[0][0,0,0]^=1
        if bad=='dtype': output=output.astype(np.int16)
        return [output]
sys.modules['dx_engine']=types.SimpleNamespace(InferenceEngine=Engine)
request=read_json(stage/'request.json')
if bad=='image': (stage/'suite'/request['samples'][0]['image']).write_bytes(b'changed')
try:
    result=remote_worker.run(request,stage,stage/'out')
except Exception as exc:
    result={'status':'exception','error':str(exc)}
write_json(stage/'test_result.json',{'result':result,'engine_calls':len(calls),'test_engine':'synthetic_only'})
''')
    proc = subprocess.run([sys.executable,str(script),str(stage),bad],env={**os.environ,'ORT_DISABLE_TELEMETRY':'1'}, text=True,capture_output=True,timeout=60)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return stage, request, read_json(stage/'test_result.json')


def test_T07_29_actual_staging_worker_quality_records_preserve_decoded_nms_once(tmp_path):
    c=case(tmp_path)
    original={str(p):digest(p) for p in c.run.rglob('*') if p.is_file()}
    stage,request,observed=execute(c)
    result=observed['result']
    assert result['status']=='complete', json.dumps(result,indent=2)
    assert result['pipeline_consistency']=='consistent_same_raw_output_scope'
    assert observed['engine_calls']==4  # two consumers reuse one actual raw result
    assert result['coco_ap_evaluated'] is False
    assert result['quality_acceptance']=='NOT_EVALUATED'
    assert request['selection_frozen_before_inference'] is True
    assert len({s['aspect_ratio'] for s in request['samples']})==4
    for row in result['rows']:
        assert row['same_raw_record_parity'] is True
        assert row['exact_once_completion'] is True
        assert row['host_nms_applied'] is False
        assert len(row['host_detections'])==2  # overlapping same-class boxes must survive
        assert row['raw_outputs_unchanged_by_host'] is True
        assert row['original_geometry_binding'] is True
    first=result['rows'][0]['host_detections'][0]
    assert [first[key] for key in ('x1','y1','x2','y2')]==[10.0,20.0,40.0,60.0]  # odd pad 139, not 139.5; inverse once
    assert (stage/'out/product_quality_detections.json').is_file()
    assert len(list((stage/'out').glob('raw_*.npz')))==4
    assert original=={str(p):digest(p) for p in c.run.rglob('*') if p.is_file()}


@pytest.mark.parametrize('bad',['nan','class','input','image','dtype'])
def test_T07_29_actual_worker_rejects_corruption_and_retains_raw_evidence(tmp_path,bad):
    stage,_,observed=execute(case(tmp_path),bad=bad)
    result=observed['result']
    assert result['status']!='complete'
    assert result.get('pipeline_consistency')!='consistent_same_raw_output_scope'
    if bad=='image':
        assert observed['engine_calls']==0
        assert 'image_digest_mismatch' in result['error']
    else:
        assert observed['engine_calls']==4
        assert len(list((stage/'out').glob('raw_*.npz')))==4
        if bad=='nan':
            with np.load(stage/'out/raw_000.npz') as raw:
                assert np.isnan(raw['output_00'][0,0,4])


@pytest.mark.parametrize('bad',['duplicate','missing','same_aspect','missing_dxnn','unknown_stage'])
def test_T07_29_fixed_selection_and_contracts_fail_before_hardware(tmp_path,bad):
    c=case(tmp_path)
    if bad=='duplicate': c.args.detection_image_ids=[c.ids[0]]*4
    if bad=='missing': c.args.detection_image_ids=c.ids[:3]+['not_in_request.png']
    if bad=='same_aspect':
        c.args.detection_image_ids=c.ids
        Image.new('RGB',(640,361)).save(c.suite/'validation'/c.ids[1])
    if bad=='missing_dxnn': (c.full/'model.dxnn').unlink()
    if bad=='unknown_stage':
        path=c.full/'output_contract.json'; contract=read_json(path);contract['contract_family']='auto';write_json(path,contract)
    with pytest.raises(ValueError): dc.prepare_detection_model(c.run,c.target,c.remote,c.args,c.output)
    assert not (c.output/'_stage').exists()


@pytest.mark.parametrize('destination',['run','tool'])
def test_T07_29_actual_cli_refuses_output_inside_originals(tmp_path,destination):
    c=case(tmp_path); path=c.run if destination=='run' else ROOT
    before={str(p) for p in path.iterdir()}
    proc=subprocess.run([sys.executable,str(LIB/'collect_smokes.py'),'--stage','detection-control','--models','yolo26s',
        '--run-dir',str(c.run),'--tool-dir',str(ROOT),'--output-dir',str(path),'--plan-only'],text=True,capture_output=True,timeout=20)
    assert proc.returncode==2
    assert 'outside the original run and installed source' in proc.stderr
    assert before=={str(p) for p in path.iterdir()}


def test_T07_29_original_endpoint_fixtures_have_exact_provenance():
    for row in read_json(FIX/'PROVENANCE.json')['fixtures']:
        assert digest(FIX/row['fixture'])==row['sha256']


@pytest.mark.parametrize('bad',['partial','nonce'])
def test_T07_29_collector_zip_does_not_turn_partial_or_unbound_result_into_pass(tmp_path,monkeypatch,bad):
    import collect_smokes
    from contextlib import nullcontext
    c=case(tmp_path)
    monkeypatch.setattr(collect_smokes,'workflow_gate',lambda path:nullcontext())
    def transport(stage,remote,out):
        request=read_json(stage/'request.json')
        write_json(out/'remote/remote_result.json',{'status':'partial' if bad=='partial' else 'complete',
            'request_nonce':'wrong' if bad=='nonce' else request['nonce'], 'hardware_executed':False,
            'collection_status':'incomplete' if bad=='partial' else 'complete','pipeline_consistency':'not_evaluated'})
        return {'request_binding_verified':True,'cleanup_confirmed':True}
    monkeypatch.setattr(collect_smokes,'run_remote',transport)
    monkeypatch.setattr(dc.shutil,'which',lambda name:'/synthetic/'+name)
    assert dc.collect_detection(c.args)==2
    outputs=list(c.args.output_dir.glob('deepx_detection_s6_v27931_*'))
    folder=next(p for p in outputs if p.is_dir())
    assert read_json(folder/'collection_summary.json')['status']=='partial_evidence'
    assert folder.with_suffix('.zip').is_file()
    assert not (folder/'_stage').exists()
