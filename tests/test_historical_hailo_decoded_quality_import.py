"""Decoded Hailo operator dispatch and sealed compiled slots; no devices/services."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from onnx_splitpoint_tool import native_detection_postprocess as native
from onnx_splitpoint_tool.workflow import historical_quality_import as subject
from test_generic_quality_completion import descriptor
from test_historical_hailo_quality_import import completion, hailo_feed
from test_historical_paired_quality_import import PairedFixture
from test_historical_quality_import import contract


def compiled_feed(tmp_path):
    args = hailo_feed(tmp_path)
    report, command, config, prepared, completed = args
    bridge = subject._json(config['existing_part2_bridge']['binding_source'])['native_split_quality_binding']
    runtime_name = bridge['boundary_metadata_payload']['boundary_tensor']['runtime_name']
    command['runtime_options'].update(canonical_input_slot_names=['compiled/input'],
        canonical_output_slot_names=[runtime_name],
        runtime_input_shapes={'compiled/input': prepared['shape']},
        runtime_output_shapes={runtime_name: bridge['boundary_contract']['boundary_tensor_shape']})
    return args


def test_sealed_compiled_streams_preserve_the_owned_feed_and_original_declarations(tmp_path):
    args = compiled_feed(tmp_path)
    before = deepcopy(args)
    assert subject._paired_hailo_input(*args) == '7'*64
    assert args == before


@pytest.mark.parametrize('fault', ['output_name', 'output_shape', 'input_shape', 'multiple_inputs',
    'empty_input', 'dtype', 'missing_seal', 'missing_metadata', 'prepared_bytes', 'observed_input'])
def test_compiled_slots_cannot_bypass_sealed_metadata_or_original_uint8_input(tmp_path, fault):
    args = compiled_feed(tmp_path)
    report, command, config, prepared, completed = args
    options = command['runtime_options']
    if fault == 'output_name':
        options['canonical_output_slot_names'] = ['foreign']
        options['runtime_output_shapes'] = {'foreign': next(iter(options['runtime_output_shapes'].values()))}
    elif fault == 'output_shape':
        options['runtime_output_shapes'][options['canonical_output_slot_names'][0]] = [8]
    elif fault == 'input_shape': options['runtime_input_shapes']['compiled/input'] = [1,3,2,2]
    elif fault == 'multiple_inputs':
        options['canonical_input_slot_names'].append('second')
        options['runtime_input_shapes']['second'] = prepared['shape']
    elif fault == 'empty_input':
        options['canonical_input_slot_names'] = ['']
        options['runtime_input_shapes'] = {'': prepared['shape']}
    elif fault == 'dtype': report['trt_input_dtype'] = 'float32'
    elif fault == 'missing_seal': config['existing_part2_bridge'].pop('binding_sha256')
    elif fault == 'missing_metadata':
        source = config['existing_part2_bridge']['binding_source']
        path = Path(source['path'])
        body = json.loads(path.read_text())
        body['native_split_quality_binding'].pop('boundary_metadata_payload')
        path.write_text(json.dumps(body))
        config['existing_part2_bridge']['binding_source'] = {**descriptor(path), 'pointer': source['pointer']}
    elif fault == 'prepared_bytes': command['artifacts']['prepared_input_00']['sha256'] = '0'*64
    else: completed['representative_input_comparison']['actual_runtime_input']['images']['tensor_sha256'] = '0'*64
    with pytest.raises(ValueError): subject._paired_hailo_input(*args)


@pytest.mark.parametrize('family', ['decoded_nms', 'decoded_pre_nms', 'raw_head'])
def test_all_existing_canonical_families_remain_exact_after_reporting_seal_comparison(family):
    if family == 'decoded_nms':
        old, current, model_sha = contract(historical=True), contract(), 'e'*64
    else:
        old, model_sha = completion(family, historical=True)
        current, _ = completion(family)
    before = deepcopy((old,current))
    a = subject._operator(old, model_sha256=model_sha)
    b = subject._operator(current, model_sha256=model_sha)
    assert a == b
    assert a['completion_mode'] == old['completion_mode']
    assert a['source_endpoint']['stage'] == old['source_endpoint']['stage']
    for key in ['source_nms_attested', 'host_nms_applied', 'confidence_threshold', 'iou_threshold',
                'max_detections', 'input_hw', 'original_wh', 'source_output_format']:
        if key in old['processor_contract']:
            assert a['processor_contract'][key] == old['processor_contract'][key]
    assert (old,current) == before


@pytest.mark.parametrize('field,value', [('host_nms_applied', True), ('source_nms_attested', False),
    ('confidence_threshold', .31), ('iou_threshold', .46), ('max_detections', 301)])
def test_changed_nms_parameters_remain_rejected_after_outer_resealing(field, value):
    changed = contract()
    changed['processor_contract'][field] = value
    changed['processor_contract'].pop('contract_sha256')
    changed['processor_contract']['contract_sha256'] = native.canonical_json_sha256(changed['processor_contract'])
    changed.pop('contract_sha256')
    changed['contract_sha256'] = native.canonical_json_sha256(changed)
    with pytest.raises(native.FrozenPostprocessError):subject._operator(changed, model_sha256='e'*64)


def hailo_pair(tmp_path, monkeypatch):
    f = PairedFixture(tmp_path,monkeypatch)
    f.identity.update(direction='hailo10h_to_trt', comparison_backend='hailo10h', setup_id='orin_nx_hailo10_01')
    f.uid = 'generic|yolo26s|H10|hailo10h_to_trt|b321'
    f.target['generic_uid'] = f.uid
    f.config['identity'] = deepcopy(f.identity)
    f.qrow['identity'] = f.config['identity']
    return f


def test_matching_decoded_hailo_dispatch_reaches_the_unchanged_artifact_guard(tmp_path,monkeypatch):
    f = hailo_pair(tmp_path,monkeypatch)
    class ReachedOriginalArtifactGuard(Exception):pass
    def original_artifacts(target, identity, artifacts):
        assert target is f.target and identity['comparison_backend'] == 'hailo10h'
        raise ReachedOriginalArtifactGuard
    monkeypatch.setattr(subject,'_paired_artifacts',original_artifacts)
    with pytest.raises(ReachedOriginalArtifactGuard):subject._admit(f.target,f.bind())


@pytest.mark.parametrize('other_family',['decoded_pre_nms','raw_head'])
def test_mismatched_canonical_family_never_reaches_projection(tmp_path,monkeypatch,other_family):
    f = hailo_pair(tmp_path,monkeypatch)
    different,_ = completion(other_family)
    f.config['completion_contract'] = different
    f.qrow['completion_contract'] = different
    with pytest.raises(ValueError,match='scientific completed operator parameters differ'):
        subject._admit(f.target,f.bind())
    assert not f.normal_calls
