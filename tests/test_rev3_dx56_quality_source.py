"""Direct Native RGB Classification Quality, durable candidate and normal export."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from tests.test_v2912_generic_split_completion import _module


def input_contract(shape=(2, 3, 3)):
    return {'task':'classification','layout':'HWC','dtype':'uint8','color_space':'RGB',
            'preprocess_mode':'resize','shape':list(shape),
            'classification_preprocessing':'imagenet_mean_std',
            'embedded_numeric_path':'dxcom_div255_then_build_onnx_imagenet_mean_std'}


def test_direct_input_preserves_uint8_codes_and_native_geometry(tmp_path, monkeypatch):
    from onnx_splitpoint_tool.preprocessing_contract import prepare_rgb_uint8_for_declared_mode
    runner = _module(monkeypatch)
    values = np.broadcast_to(np.arange(256,dtype=np.uint8)[None,:,None],(3,256,3)).copy()
    image = tmp_path/'all_codes.png';Image.fromarray(values).save(image)
    direct = runner._direct_rgb_classification_quality_input(image,input_contract(values.shape))
    assert direct.dtype == np.uint8 and direct.flags.c_contiguous
    assert direct.tobytes() == values.tobytes()
    resized = runner._direct_rgb_classification_quality_input(image,input_contract())
    native,_ = prepare_rgb_uint8_for_declared_mode(values,target_hw=(2,3),preprocess_mode='resize',pad_value=0)
    assert resized.tobytes() == native.tobytes()
    normalized=(values.astype(np.float32)/255-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
    reconstructed=np.clip((normalized*np.array([.229,.224,.225],np.float32)+np.array([.485,.456,.406],np.float32))*255,0,255).astype(np.uint8)
    assert np.any(reconstructed != direct)


@pytest.mark.parametrize('change',[{'dtype':'float32'},{'layout':'NCHW'},{'preprocess_mode':'letterbox'},
                                   {'classification_preprocessing':'none'},{'shape':[2,3,4]}])
def test_direct_input_rejects_unbound_numeric_paths(tmp_path,monkeypatch,change):
    runner=_module(monkeypatch)
    with pytest.raises(ValueError,match='exact bound Native RGB'):
        runner._direct_rgb_classification_quality_input(tmp_path/'absent.png',{**input_contract(),**change})


def test_persistent_source_resumes_without_duplicate_predictions(tmp_path,monkeypatch):
    runner=_module(monkeypatch)
    samples=[{'image':tmp_path/f'{index}.png','label_id':index,'label_name':None} for index in range(3)]
    calls=[]
    fail=[True]
    def infer(path,prepared):
        if path.name=='1.png' and fail[0]:
            raise RuntimeError('device interruption')
        calls.append(path.name)
        return ['logits'],[np.arange(6,dtype=np.float32)[None]]
    args=dict(samples=samples,out_dir=tmp_path,identity={'bound_source':'current'},
              prepare=lambda path:np.zeros((2,3,3),np.uint8),infer=infer,
              metrics=lambda names,outputs,label:{'top1_hit':True,'top5_hit':True})
    with pytest.raises(RuntimeError,match='device interruption'):
        runner._persistent_classification_quality_source(**args)
    fail[0]=False
    rows=runner._persistent_classification_quality_source(**args)
    assert calls==['0.png','1.png','2.png']
    assert len(rows)==3
    assert runner._persistent_classification_quality_source(**args)==rows
    assert calls==['0.png','1.png','2.png']
    progress=json.loads((tmp_path/'task_quality_inputs/composed_classification_progress.json').read_text())
    assert progress['completed_images']==3 and progress['status']=='predictions_complete'
    with pytest.raises(ValueError,match='identity differs'):
        runner._persistent_classification_quality_source(**{**args,'identity':{'bound_source':'changed'}})
    journal=tmp_path/'task_quality_inputs/composed_classification_predictions.jsonl'
    lines=journal.read_text().splitlines();entry=json.loads(lines[1]);entry['row']['gt']['top1_hit']=False
    lines[1]=json.dumps(entry);journal.write_text('\n'.join(lines)+'\n')
    with pytest.raises(ValueError,match='saved prediction is invalid'):
        runner._persistent_classification_quality_source(**args)


def test_dataset_preflight_rejects_changed_prepared_image_bytes(tmp_path,monkeypatch):
    runner=_module(monkeypatch)
    image=tmp_path/'sample.png';Image.new('RGB',(2,3),(4,5,6)).save(image)
    sample={'image':image,'label_id':7}
    manifest={'samples':[{'image':image.name,'label_id':7,
                         'source_sha256':'sha256:'+hashlib.sha256(image.read_bytes()).hexdigest()}]}
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    proof=runner._classification_quality_dataset_preflight([sample],tmp_path)
    assert proof['source_hash_verified_images']==1
    Image.new('RGB',(2,3),(5,5,6)).save(image)
    with pytest.raises(ValueError,match='image bytes differ'):
        runner._classification_quality_dataset_preflight([sample],tmp_path)


def test_real_generated_classification_source_uses_direct_rgb_and_normal_consumer(tmp_path,monkeypatch):
    from tests.yolo_quality_generic_fixture import make_generic_fixture
    from onnx_splitpoint_tool.native_split_quality import known_native_split_policy
    row=dict(model_id='mobilenet_v3_large',case_id='b001',setup_id='orin_nx_deepx_m1_01',direction='deepx_to_trt')
    row['precision']=known_native_split_policy(model_id=row['model_id'],case_id=row['case_id'],
        setup_id=row['setup_id'],backend=row['direction'])['precision']
    outputs={'logits':np.arange(1000,0,-1,dtype=np.float32)[None]}
    result=make_generic_fixture(tmp_path,monkeypatch,row,{'eval_run_id':'dx56-source-only',
        'quality_source_only':True,'task':'classification','endpoint_mode':'classification_logits','outputs':outputs})
    request=result['request'];report=result['raw_report']
    assert request['task']=='classification' and request['record_count']==5000
    assert report['quality_source_only'] is True and report['performance_measurement_executed'] is False
    assert result['calls']['full']==0
    encoding=request['quality_contract']['preprocessing']['runtime_input_encoding']
    assert encoding['numeric_path']=='direct_HWC_RGB_uint8' and encoding['input_dtype']=='uint8'
    journal=Path(result['composed_request_path']).parent/'composed_classification_predictions.jsonl'
    entries=[json.loads(line) for line in journal.read_text().splitlines()]
    assert len(entries)==5001
    expected=np.asarray(Image.open(tmp_path/'dataset/00000.png').convert('RGB').resize((64,64),resample=Image.BILINEAR))
    assert entries[1]['row']['direct_rgb_input_sha256']==hashlib.sha256(expected.tobytes()).hexdigest()
    assert all(entry['row']['gt']['top1_hit'] for entry in entries[1:])
    assert result['reference_path'].read_bytes()==result['reference_bytes']
