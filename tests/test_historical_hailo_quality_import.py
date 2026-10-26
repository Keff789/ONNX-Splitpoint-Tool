"""Canonical historical Hailo operator/bridge comparisons; no devices/services."""
from copy import deepcopy
from unittest.mock import patch
import math

import numpy as np
import pytest

from onnx_splitpoint_tool import native_detection_postprocess as ndp
from onnx_splitpoint_tool.runners.harness.yolo import YOLOV7_PAPER_ONNX_SHA256
from onnx_splitpoint_tool.workflow import historical_quality_import as subject
from test_generic_quality_existing_bridge import bridge_fixture
from test_historical_quality_import import OLD_IMPLEMENTATION
from test_historical_paired_quality_import import PairedFixture


def completion(family, *, historical=False, confidence=.25, original_wh=(640,426),
               harness_implementation=None):
    if family=='raw_head':
        model='yolov7_paper';model_sha=YOLOV7_PAPER_ONNX_SHA256
        outputs={f'stride{s}':np.zeros((1,3,640//s,640//s,85),np.float32) for s in (8,16,32)}
    else:
        model='yolo11l';model_sha='e'*64
        outputs={'output0':np.zeros((1,84,8400),np.float32)}
    signature=ndp.tensor_signature(outputs)
    source={'task':'detection','stage':family,'contract_family':family,
        'endpoint_contract_complete':True,'endpoint_contract_hash':'d'*64,
        'model_sha256':model_sha,'tensor_signature':signature,
        'output_endpoint_attestation':{'attested':True,'status':'passed','endpoint':family,
            'stage':family,'endpoint_contract_hash':'d'*64,'tensor_signature':signature}}
    implementation=deepcopy(ndp._implementation_artifacts())
    if historical:implementation['native_detection_postprocess']['sha256']=OLD_IMPLEMENTATION
    if harness_implementation:implementation['yolo_harness']['sha256']=harness_implementation
    with patch.object(ndp,'_implementation_artifacts',return_value=implementation):
        result=ndp.build_detection_completion_execution_contract(model_id=model,outputs=outputs,
            input_hw=[640,640],original_wh=list(original_wh),source_endpoint_contract=source,
            confidence_threshold=confidence)
    return result,model_sha


@pytest.mark.parametrize('family',['raw_head','decoded_pre_nms'])
def test_canonical_raw_and_pre_nms_reporting_seals_compare_without_contract_rewrite(family):
    old,sha=completion(family,historical=True);current,_=completion(family)
    original=deepcopy((old,current))
    assert old['contract_sha256']!=current['contract_sha256']
    assert subject._operator(old,model_sha256=sha)==subject._operator(current,model_sha256=sha)
    assert (old,current)==original


def test_scientific_pre_nms_geometry_remains_exact_after_canonical_resealing():
    old,sha=completion('decoded_pre_nms',historical=True)
    changed,_=completion('decoded_pre_nms',original_wh=(640,427))
    assert subject._operator(old,model_sha256=sha)!=subject._operator(changed,model_sha256=sha)


def test_existing_canonical_verifier_still_rejects_a_changed_threshold():
    with pytest.raises(ndp.FrozenPostprocessError):
        completion('decoded_pre_nms',confidence=.31)


def test_reporting_compatibility_never_accepts_a_different_sigmoid_arithmetic_module():
    contract,sha=completion('raw_head',historical=True,harness_implementation='f'*64)
    with pytest.raises(ndp.FrozenPostprocessError,match='sigmoid_arithmetic_implementation_mismatch'):
        subject._operator(contract,model_sha256=sha)


@pytest.mark.parametrize('field',['invariant_contract_sha256','invariant_identity'])
def test_inner_derived_identity_must_be_verified_before_it_is_omitted(field):
    contract,sha=completion('decoded_pre_nms')
    processor=contract['processor_contract']
    processor[field]='0'*64 if field.endswith('sha256') else {}
    processor.pop('contract_sha256');processor['contract_sha256']=ndp.canonical_json_sha256(processor)
    contract.pop('contract_sha256');contract['contract_sha256']=ndp.canonical_json_sha256(contract)
    with pytest.raises(ndp.FrozenPostprocessError):subject._operator(contract,model_sha256=sha)


def hailo_feed(tmp_path):
    fixture,bridge,_=bridge_fixture(tmp_path)
    config=fixture['config'];boundary=bridge['boundary_contract'];shape=boundary['boundary_tensor_shape']
    prepared={'format':'npy','dtype':'uint8','shape':[2,2,3],'sha256':'a'*64}
    rgb={'shape':prepared['shape'],'dtype':'uint8','tensor_sha256':'7'*64}
    completed={'representative_input_comparison':{'prepared_tensor':rgb,'actual_runtime_input':{'images':deepcopy(rgb)}}}
    command={'runtime_options':{'quantized_inputs':True,'quantized_outputs':True,'copy_outputs':True,
        'canonical_input_slot_names':['images'],'canonical_output_slot_names':[boundary['boundary_tensor_name']],
        'runtime_input_shapes':{'images':prepared['shape']},
        'runtime_output_shapes':{boundary['boundary_tensor_name']:shape}},
        'artifacts':{'prepared_input_00':{'sha256':prepared['sha256']}}}
    native={'trt_input_dtype':'uint8','trt_inputs':[boundary['boundary_tensor_name']],
        'trt_input_bytes':math.prod(shape),'boundary_copy_count':1,
        'boundary_copy_policy':'single_owned_contiguous_copy_from_reusable_accelerator_binding',
        'trt_copy_outputs':True,'measurement_boundary':'workers_ready_to_last_completed_task_frame',
        'boundary_layout':boundary['boundary_layout'],
        'boundary_contract':{'bridge_schema':'onnx-splitpoint/uint8-dequant-bridge',
            'metadata_sha256':bridge['artifacts']['native_trt_meta']['sha256'],
            'boundary_layout_effective':boundary['boundary_layout'],
            'dequant_scale':boundary['dequant_scale'],'dequant_zero_point':boundary['dequant_zero_point']}}
    return native,command,config,prepared,completed


def test_actual_sealed_bridge_and_exact_numpy_feed_authorize_owned_uint8_boundary(tmp_path):
    args=hailo_feed(tmp_path);before=deepcopy(args)
    assert subject._paired_hailo_input(*args)=='7'*64
    assert args==before


@pytest.mark.parametrize('fault',['source_SHA','bridge_seal','bridge_policy','selected_artifact','boundary_dtype',
    'boundary_bytes','copy_count','copy_policy','layout','dequant_scale','native_meta','materialization',
    'input_dtype','input_name','input_SHA','actual_input','input_shape'])
def test_bridge_and_actual_input_or_owned_boundary_difference_reject(tmp_path,fault):
    args=list(hailo_feed(tmp_path));native,command,config,prepared,completed=args
    if fault=='source_SHA':config['existing_part2_bridge']['binding_source']['sha256']='0'*64
    elif fault=='bridge_seal':config['existing_part2_bridge']['binding_sha256']='0'*64
    elif fault=='bridge_policy':config['existing_part2_bridge']['policy_sha256']='0'*64
    elif fault=='selected_artifact':config['generic_artifact_binding']['artifacts']['engine']['sha256']='0'*64
    elif fault=='boundary_dtype':native['trt_input_dtype']='float32'
    elif fault=='boundary_bytes':native['trt_input_bytes']+=1
    elif fault=='copy_count':native['boundary_copy_count']=True
    elif fault=='copy_policy':native['boundary_copy_policy']='borrowed'
    elif fault=='layout':native['boundary_contract']['boundary_layout_effective']='nchw'
    elif fault=='dequant_scale':native['boundary_contract']['dequant_scale']+=.1
    elif fault=='native_meta':native['boundary_contract']['metadata_sha256']='0'*64
    elif fault=='materialization':native['trt_copy_outputs']=False
    elif fault=='input_dtype':prepared['dtype']='float32'
    elif fault=='input_name':command['runtime_options']['canonical_input_slot_names']=['foreign']
    elif fault=='input_SHA':command['artifacts']['prepared_input_00']['sha256']='0'*64
    elif fault=='actual_input':completed['representative_input_comparison']['actual_runtime_input']['images']['tensor_sha256']='0'*64
    else:command['runtime_options']['runtime_input_shapes']['images']=[1,3,2,2]
    with pytest.raises((ValueError,OSError)):subject._paired_hailo_input(*args)


@pytest.mark.parametrize('fault',[None,'HEF','architecture','source_graph'])
def test_original_paired_hef_receipt_and_graph_are_still_bound(tmp_path,monkeypatch,fault):
    f=PairedFixture(tmp_path,monkeypatch)
    f.identity.update(direction='hailo10h_to_trt',comparison_backend='hailo10h',setup_id='orin_nx_hailo10_01')
    f.cache['backend']='h10'
    f.receipt.update(schema='onnx-splitpoint/hailo-hef-build-receipt/v2',hw_arch='hailo10h',
        hef_sha256=f.artifacts['part1_runtime']['sha256'],hef_size_bytes=f.artifacts['part1_runtime']['size_bytes'])
    if fault=='HEF':f.receipt['hef_sha256']='0'*64
    elif fault=='architecture':f.receipt['hw_arch']='hailo8'
    elif fault=='source_graph':f.receipt['source_onnx_sha256']='0'*64
    f.bind();f.target['source_authorities']['cache_inventory']['selector']['backend']='h10'
    if fault is None:subject._paired_artifacts(f.target,f.identity,f.artifacts)
    else:
        with pytest.raises(ValueError):subject._paired_artifacts(f.target,f.identity,f.artifacts)
