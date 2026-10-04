"""Cross-run decoder equality keeps each sealed source provenance intact."""
import json

import numpy as np
import pytest

from onnx_splitpoint_tool.native_detection_postprocess import build_detection_completion_execution_contract
from onnx_splitpoint_tool.native_output_endpoint import load_authoritative_output_contract, runtime_output_contract
from scripts.native_producer_validate_visualize import _stored_split_processor_compatibility


def sample(root, *, confidence=.25, iou=.45, original_wh=(640,480), extra=None):
    root.mkdir(parents=True)
    declaration={'schema':'onnx-splitpoint/output-contract','schema_version':1,
        'model_id':'yolo26s','task':'detection','backend':'tensorrt','variant':'full','contract_status':'recorded',
        'endpoint_mode':'decoded_nms','host_tail_required':False,'postprocessing_required':False,
        'source_coordinate_space':'model_input_letterbox_xyxy_pixels',**(extra or {})}
    (root/'output_contracts.json').write_text(json.dumps({'model_id':'yolo26s','contracts':[declaration]}))
    outputs={'output0':np.asarray([[[80.,160.,320.,480.,.9,2.]]],np.float32)}
    declared=load_authoritative_output_contract(root,backend='tensorrt',model_id='yolo26s',task='detection')
    source=runtime_output_contract('detection',outputs,raw_fallback=False,declared_contract=declared)
    execution=build_detection_completion_execution_contract(model_id='yolo26s',outputs=outputs,
        input_hw=(640,640),original_wh=original_wh,source_endpoint_contract=source,
        preprocess={'mode':'letterbox','rgb':True,'pad_value':114},confidence_threshold=confidence,iou_threshold=iou)
    manifest=root/'dump.json'
    manifest.write_text(json.dumps(dict(source,authoritative_output_contract_resolution=declared)))
    return manifest,outputs,execution


def compare(current, original):
    return _stored_split_processor_compatibility(*current,*original)


def test_different_run_paths_keep_both_original_hashes_and_same_decoder(tmp_path):
    old=sample(tmp_path/'original');new=sample(tmp_path/'new')
    before={path:path.read_bytes() for path in (old[0],new[0])}
    assert old[2]['processor_contract'] != new[2]['processor_contract']
    result=compare(new,old)
    assert result['status']=='independently_verified_same_processor_primitives'
    for name,value in [('original',old),('current',new)]:
        assert result[name]['processor_contract_sha256']==value[2]['processor_contract']['contract_sha256']
        assert result[name]['execution_contract_sha256']==value[2]['contract_sha256']
        assert result[name]['declaration_source']==str(value[0].parent/'output_contracts.json')
    assert all(path.read_bytes()==data for path,data in before.items())


@pytest.mark.parametrize('change', ['confidence','nms','input_geometry','decoder_metadata',
                                    'execution_sha','processor_sha','source_sha','endpoint','stored_attestation'])
def test_origin_projection_never_accepts_decoder_input_endpoint_or_hash_drift(tmp_path,change):
    old=sample(tmp_path/'original')
    kwargs={'confidence':{'confidence':.3},'nms':{'iou':.4},'input_geometry':{'original_wh':(640,400)},
            'decoder_metadata':{'extra':{'unrecognized_decoder_option':'different'}}}.get(change,{})
    if change in {'confidence','nms'}:
        with pytest.raises((ValueError,RuntimeError),match='canonical_policy_required'):
            sample(tmp_path/'new',**kwargs)
        return
    new=sample(tmp_path/'new',**kwargs)
    execution=new[2]
    if change=='execution_sha':execution['contract_sha256']='f'*64
    elif change=='processor_sha':execution['processor_contract']['contract_sha256']='f'*64
    elif change=='source_sha':execution['processor_contract']['source_output_endpoint_attestation_sha256']='f'*64
    elif change=='endpoint':execution['source_endpoint']['endpoint_contract_hash']='f'*64
    elif change=='stored_attestation':
        payload=json.loads(new[0].read_text());payload['output_endpoint_attestation']['attested']=False
        new[0].write_text(json.dumps(payload))
    with pytest.raises((ValueError,RuntimeError)):
        compare(new,old)


def test_both_source_attestations_are_reverified(tmp_path):
    old=sample(tmp_path/'original');new=sample(tmp_path/'new')
    payload=json.loads(old[0].read_text())
    payload['authoritative_output_contract_resolution']['source_contracts_sha256']='f'*64
    old[0].write_text(json.dumps(payload))
    with pytest.raises((ValueError,RuntimeError)):
        compare(new,old)


@pytest.mark.parametrize('changed',[False,True])
def test_collected_remote_origin_requires_exact_existing_suite_bytes(tmp_path,changed):
    old=sample(tmp_path/'remote/benchmark_set');new=sample(tmp_path/'current/benchmark_set')
    collected=tmp_path/'collected/benchmark_set';collected.parent.mkdir()
    original_path=old[0].parent
    original_path.rename(collected)
    old=(collected/old[0].name,old[1],old[2])
    if changed:
        with (collected/'output_contracts.json').open('a') as stream:stream.write(' ')
        with pytest.raises(ValueError,match='source_declaration_unavailable_or_changed'):
            compare(new,old)
    else:
        proof=compare(new,old)
        assert proof['original']['declaration_source']==str(original_path/'output_contracts.json')
        assert proof['original']['verified_local_declaration_source']==str(collected/'output_contracts.json')
