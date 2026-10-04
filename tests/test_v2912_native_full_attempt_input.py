"""Resolve a collected TRT Full input without running any model session."""
from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import pytest
from scripts import native_producer_validate_visualize as validator


def _fixture(tmp_path):
    path = Path(__file__).with_name('test_v270i_native_full_validation_p0.py')
    spec = importlib.util.spec_from_file_location('full_attempt_input_fixture', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    old = module._full_validation_fixture(tmp_path / 'legacy')
    relative = Path('full_attempts/tensorrt/native_full_outputs/model=resnet50/backend=native_full_tensorrt/setup=orin_nx_hailo8_01/comparison=hailo8')
    directory = tmp_path / 'collected' / relative
    shutil.copytree(old['dump_dir'], directory)
    input_path = directory / 'native_full_input_manifest.json'
    output_path = directory / 'native_full_outputs_manifest.json'
    ip = json.loads(input_path.read_text());ip['backend'] = 'native_full_tensorrt'
    input_path.write_text(json.dumps(ip))
    op = json.loads(output_path.read_text());op['backend'] = 'native_full_tensorrt'
    op['input_manifest'] = str(Path('/remote/attempt-id') / relative / input_path.name)
    op['input_manifest_sha256'] = validator._sha256_file(input_path)
    output_path.write_text(json.dumps(op))
    return input_path, output_path


def test_collected_full_attempt_validates_original_input_bytes(tmp_path):
    input_path, output_path = _fixture(tmp_path)
    before = {p:p.read_bytes() for p in output_path.parent.iterdir() if p.is_file()}
    assert validator._find_native_full_input_manifest_for_output(output_path) == input_path
    assert all(p.read_bytes() == contents for p,contents in before.items())


@pytest.mark.parametrize('field,value', [
    ('setup_id', 'different_setup'), ('model', 'different_model'),
    ('comparison_backend', 'deepx'), ('runtime_input_sha256', '1' * 64),
    ('runtime_input_shape', [1,3,3,3]), ('input_image_sha256', '2' * 64),
])
def test_attempt_layout_does_not_weaken_identity_or_input_checks(tmp_path, field, value):
    input_path, output_path = _fixture(tmp_path)
    ip = json.loads(input_path.read_text()); ip[field] = value
    input_path.write_text(json.dumps(ip))
    op = json.loads(output_path.read_text())
    op['input_manifest_sha256'] = validator._sha256_file(input_path)
    output_path.write_text(json.dumps(op))
    assert validator._find_native_full_input_manifest_for_output(output_path) is None


@pytest.mark.parametrize('remote', [
    '/remote/full_attempts/tensorrt/native_full_outputs/model=other/backend=native_full_tensorrt/setup=orin_nx_hailo8_01/comparison=hailo8/native_full_input_manifest.json',
    '/remote/full_attempts/tensorrt/native_full_outputs/model=resnet50/backend=native_full_tensorrt/setup=other/comparison=hailo8/native_full_input_manifest.json',
    '/remote/full_attempts/tensorrt/native_full_outputs/model=resnet50/backend=native_full_tensorrt/setup=orin_nx_hailo8_01/comparison=hailo8/../native_full_input_manifest.json',
    '/remote/full_attempts/other/native_full_outputs/model=resnet50/backend=native_full_tensorrt/setup=orin_nx_hailo8_01/comparison=hailo8/native_full_input_manifest.json',
])
def test_attempt_layout_requires_exact_portable_suffix(tmp_path, remote):
    _, output_path = _fixture(tmp_path)
    op = json.loads(output_path.read_text());op['input_manifest'] = remote
    output_path.write_text(json.dumps(op))
    assert validator._find_native_full_input_manifest_for_output(output_path) is None


def test_existing_outputs_mode_never_calls_live_self_reference(tmp_path, monkeypatch):
    input_path, manifest = _fixture(tmp_path)
    out = tmp_path / 'validation'
    out.mkdir()
    def forbidden(*args, **kwargs):
        raise AssertionError('inference/probe was called')
    monkeypatch.setattr(validator, '_full_onnx_self_reference_detection', forbidden)
    monkeypatch.setattr(validator, '_auto_run_yolo_full_self_reference_probe', forbidden)
    result = validator._visualize_detection(
        manifest, out, 0, None, tmp_path,
        endpoint_evidence={'backend': 'native_full_tensorrt', 'case': 'full'},
        existing_outputs_only=True, existing_reference_rows=[],
    )
    assert result['self_reference_report']['reason'] == 'existing_full_reference_identity_missing_or_ambiguous'
    assert result['semantic_available'] is False
    assert result['semantic_ok'] == 'unavailable'


def _merge_fixture():
    authority = {'valid': True, 'run_id': 'original', 'stage_run_id': 'original',
                 'profile_start_snapshot_sha256': 'a' * 64, 'profile_selection_fingerprint': 'b' * 64}
    row = {'backend': 'native_full_tensorrt', 'model': 'model', 'case': 'full',
           'setup_id': 'setup', 'comparison_backend': 'hailo8', 'precision': 'fp16',
           'full_command_contract_sha256': 'c' * 64, 'source_request_sha256': 'd' * 64,
           'row_index': 12, 'semantic_ok': 'unavailable'}
    previous = {'schema': 'onnx-splitpoint/native-producer-validation-summary',
                'accuracy_gate_policy_sha256': validator.AccuracyGatePolicy().sha256(),
                'native_split_quality_authority': dict(authority), 'diagnostic_only': False,
                'rows': [row, {**row, 'model': 'untouched', 'semantic_ok': False}]}
    current = [{**row, 'row_index': 0, 'semantic_ok': True}]
    return previous, current, authority


def test_validation_subset_merge_preserves_negative_originals_and_prior_status():
    previous, current, authority = _merge_fixture()
    merged = validator._merge_previous_validation_rows(previous, current, authority, validator.AccuracyGatePolicy(), False)
    assert merged[0]['row_index'] == 12
    assert merged[0]['previous_validation_projection']['semantic_ok'] == 'unavailable'
    assert merged[1] == previous['rows'][1]
    assert previous['rows'][0]['semantic_ok'] == 'unavailable'


@pytest.mark.parametrize('mutation', ['run', 'stage', 'snapshot', 'selection', 'policy', 'diagnostic', 'authority_invalid', 'request', 'command', 'missing', 'duplicate'])
def test_validation_subset_merge_rejects_other_run_policy_or_source(mutation):
    previous, current, authority = _merge_fixture()
    if mutation in ('run', 'stage', 'snapshot', 'selection'):
        field = {'run':'run_id', 'stage':'stage_run_id', 'snapshot':'profile_start_snapshot_sha256', 'selection':'profile_selection_fingerprint'}[mutation]
        previous['native_split_quality_authority'][field] = 'other'
    elif mutation == 'policy': previous['accuracy_gate_policy_sha256'] = 'f' * 64
    elif mutation == 'diagnostic': previous['diagnostic_only'] = True
    elif mutation == 'authority_invalid': authority['valid'] = False
    elif mutation == 'request': current[0]['source_request_sha256'] = 'f' * 64
    elif mutation == 'command': current[0]['full_command_contract_sha256'] = 'f' * 64
    elif mutation == 'missing': current[0]['model'] = 'missing'
    elif mutation == 'duplicate': previous['rows'].append(dict(previous['rows'][0]))
    with pytest.raises(ValueError):
        validator._merge_previous_validation_rows(previous, current, authority, validator.AccuracyGatePolicy(), False)


def test_existing_reference_reports_verified_input_kind_and_uses_saved_detections(tmp_path, monkeypatch):
    # The existing byte/endpoint verifiers are separately covered above and by
    # the integrity suite; isolate their successful hand-off to the local matcher.
    model = tmp_path / 'model.onnx'; model.write_bytes(b'original-model')
    model_sha = validator._sha256_file(model)
    native_manifest = tmp_path / 'native.json'; native_manifest.write_text('{}')
    source_manifest = tmp_path / 'source.json'; source_manifest.write_text('{}')
    input_path = tmp_path / 'input.json'
    input_payload = {'input_image_sha256':'a'*64, 'runtime_input_sha256':'b'*64,
                     'runtime_input_shape':[1,3,2,2], 'runtime_input_dtype':'float32'}
    input_path.write_text(json.dumps(input_payload))
    detections = [{'class_id':1, 'score':0.9, 'x1':0.0, 'y1':0.0, 'x2':2.0, 'y2':2.0}]
    comparison = {'endpoint_contract_hash':'c'*64}
    policy = validator.AccuracyGatePolicy()
    reference = {'completed_v2_verified':True, 'boundary_manifest':str(input_path),
                 'full_onnx':str(model), 'reference_detections':detections,
                 'full_reference_result_sha256':validator._canonical_detection_json_sha256(detections),
                 'reference_input_evidence':{'source_runtime_input_sha256':'b'*64,
                      'derived_reference_tensor_sha256':'b'*64,'target_shape':[1,3,2,2],'target_dtype':'float32'},
                 'completed_task_comparison_endpoint_contract':comparison,
                 'completed_task_comparison_endpoint_contract_hash':'c'*64,
                 'numerical_similarity_threshold':0.8, 'numerical_similarity_mean_iou_threshold':0.85,
                 'numerical_similarity_policy_id':policy.native_self_reference_policy_id,
                 'numerical_similarity_iou_threshold':0.5,'numerical_similarity_confidence_threshold':0.25}
    saved = tmp_path / 'reference.json'
    saved.write_text(json.dumps({'manifest':str(source_manifest),'self_reference_report':reference}))
    row = {'backend':'native_full_tensorrt','case':'full','model':'model','setup_id':'setup',
           'comparison_backend':'hailo8','model_sha256':model_sha,
           'full_command_contract':{'source_model_sha256':model_sha},
           'completed_task_endpoint_attestation':{'frozen_decoded_nms_normalization_result':{'detections_sha256':'d'*64}}}
    source = {**row,'backend':'native_full_hailo8','task_validation':str(saved),'dump_manifest':str(source_manifest)}
    monkeypatch.setattr(validator, '_find_native_full_input_manifest_for_output', lambda *a,**k: input_path)
    monkeypatch.setattr(validator, 'load_dump', lambda *a: ({},{}))
    monkeypatch.setattr(validator, '_verified_completed_v2_contract', lambda *a,**k: ('integrated_accelerator_plus_frozen_normalization',{},comparison))
    monkeypatch.setattr(validator, '_verified_frozen_completed_result_artifact', lambda *a,**k: detections)
    result = validator._existing_full_reference_detection(native_manifest,row,[source],policy)
    assert result['available'] is True
    assert result['self_reference_input_manifest_kind'] == 'native_full_input_manifest'
    assert result['numerical_similarity_pass'] is True
    assert result['match']['match_ratio'] == 1.0
    assert result['semantic_result_binding_status'] == 'exact_same_hotloop_completed_artifact'
