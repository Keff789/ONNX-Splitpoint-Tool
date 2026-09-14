"""AP2: original producer replay plus SYNTHETIC exporter/loader integration.

The archived request has no included candidate payload and is never repaired.
All resealing below applies exclusively to newly exported synthetic objects.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from onnx_splitpoint_tool import quality_service as service
from onnx_splitpoint_tool.quality_cache import json_fingerprint
from onnx_splitpoint_tool.native_detection_postprocess import (
    FrozenDetectionPostprocessor,
    build_completed_detection_endpoint_attestation,
    build_frozen_postprocess_contract,
)
from onnx_splitpoint_tool.native_output_endpoint import (
    load_authoritative_output_contract, runtime_output_contract,
)
from tests.test_v269a_deepx_full_central_quality import (
    _base_tree, _cpu_contract, _detection_semantic, _performance_input_evidence,
    _run, _suite_module,
)

FIXTURE = Path(__file__).parent / 'fixtures/v27930/quality_original_full_request.json'
COMPLETED_FIELDS = (
    'completed_task_endpoint_contract', 'completed_task_endpoint_contract_hash',
    'completed_task_output_endpoint_id', 'completed_task_endpoint_attestation',
    'completed_task_endpoint_attestation_sha256', 'quality_join_endpoint',
)


def _validate(producer):
    return service._validate_deepx_candidate_execution_contract(
        producer, role='request', task='detection',
    )


def test_q01_original_producer_passes_without_changing_recorded_identity():
    original_bytes = FIXTURE.read_bytes()
    request = json.loads(original_bytes)
    producer = request['producer_identity']
    before = copy.deepcopy(producer)
    validated, digest = _validate(producer)
    assert validated == before == producer
    assert digest == '38d890596eb856bf19be7cab9dede7cbe1b5b2a3165a7b6919bcde843e147770'
    assert FIXTURE.read_bytes() == original_bytes
    assert producer['endpoint']['identity']['stage'] == 'decoded_pre_nms'
    assert producer['dataset']['image_count'] == 500
    assert producer['completed_task_endpoint_attestation']['completed_frames'] == 5
    assert producer['completed_task_endpoint_contract_hash'] != producer['completed_task_endpoint_attestation']['endpoint_contract_hash']
    compact = producer['endpoint']['identity']['tensor_signature']['tensors'][0]
    full = producer['quality_record_endpoint']['identity']['decoder_contract']['frozen_postprocess_contract']['raw_output_tensor_signature']['tensors'][0]
    assert set(compact) == {'index', 'rank', 'shape'}
    assert set(full) == {'index', 'rank', 'shape', 'name', 'dtype'}


def _synthetic_export(tmp_path, *, predicted=True):
    module = _suite_module()
    samples = [
        {'image': name, 'annotations': [{'bbox': [10, 20, 30, 40], 'category_id': 1}]}
        for name in ('synthetic_wide.jpg', 'synthetic_tall.jpg')
    ]
    root, source, dxnn = _base_tree(tmp_path, task='detection', samples=samples)
    source.write_bytes(b'SYNTHETIC ONNX identity; no hardware execution')
    dxnn.write_bytes(b'SYNTHETIC DXNN identity; outputs generated in offline test')
    for name, size in zip(('synthetic_wide.jpg', 'synthetic_tall.jpg'), ((640, 361), (320, 640))):
        Image.new('RGB', size, (20, 30, 40)).save(root / 'validation' / name)
    authoritative = {
        'schema': 'onnx-splitpoint/output-contract', 'schema_version': 1,
        'model_id': 'yolo11l', 'backend': 'deepx_m1', 'variant': 'full',
        'task': 'detection', 'endpoint_mode': 'decoded_pre_nms',
        'contract_status': 'recorded', 'host_tail_required': True,
        'postprocessing_required': True,
    }
    (root / 'output_contracts.json').write_text(json.dumps({
        'schema': 'onnx-splitpoint/output-contracts', 'schema_version': 1,
        'model_id': 'yolo11l', 'task': 'detection', 'contracts': [authoritative],
    }))
    contract_path = root / 'deepx/deepx_m1/full/output_contract.json'
    contract = json.loads(contract_path.read_text())
    contract['model_id'] = 'yolo11l'
    contract['contract_family'] = 'decoded_pre_nms'
    contract['outputs'] = [{'name': 'model_outputs', 'dtype': 'float32', 'shape': [1, 84, 8400]}]
    contract['endpoint_semantic_attestation'].update({
        'model_id': 'yolo11l', 'endpoint_mode': 'decoded_pre_nms',
        'source_endpoint_semantics': 'decoded_pre_nms',
        'authoritative_contract': authoritative,
        'authoritative_contract_sha256': json_fingerprint(authoritative),
    })
    contract_path.write_text(json.dumps(contract))
    run = _run('detection')
    run.update({'id': 'synthetic_deepx_quality_v27930', 'model_id': 'yolo11l', 'model_suite': {'primary': 'yolo11l'}})
    # Existing fixture helper creates explicit synthetic input audit records;
    # actual frozen builders/processors below consume our unmodified outputs.
    semantic = _detection_semantic(root, samples)
    detection_path = root / semantic['detections_json']
    payload = json.loads(detection_path.read_text())
    decoders = []
    for ordinal, record in enumerate(payload['images']):
        audit = record['preprocessing_audit']
        height, width = audit['source_shape_hw']
        outputs = {'model_outputs': np.zeros((1, 84, 8400), dtype=np.float32)}
        outputs['model_outputs'][0, :4, 0] = [25 + audit['pad_x'], 40 + audit['pad_y'], 30, 40]
        has_prediction = predicted is True or (predicted == 'inconclusive' and ordinal == 0)
        outputs['model_outputs'][0, 4, 0] = .9 if has_prediction else 0
        frozen = build_frozen_postprocess_contract(
            model_id='yolo11l', source_contract_family='decoded_pre_nms',
            outputs=outputs, input_hw=[640, 640], original_wh=[width, height],
        )
        processor = FrozenDetectionPostprocessor(frozen)
        result = processor.process(outputs, original_wh=[width, height])
        decoder = {
            'status': 'ok', 'pass': True, 'family': 'yolo11',
            'decoder_id': frozen['decoder_id'], 'decoder_format': frozen['decoder_format'],
            'nms_included': True, 'source_endpoint_semantics': 'decoded_pre_nms',
            'source_endpoint_has_integrated_nms': False,
            'host_decoder_applied': True, 'host_nms_applied': True,
            'confidence_threshold': .25, 'nms_iou_threshold': .45, 'nms_max_detections': 300,
            'frozen_postprocess_contract': frozen,
            'frozen_postprocess_contract_sha256': frozen['contract_sha256'],
            'frozen_postprocess_invariant_identity': frozen['invariant_identity'],
            'frozen_postprocess_invariant_contract_sha256': frozen['invariant_contract_sha256'],
            'per_image_original_wh': [width, height], 'frozen_postprocess_result': result,
            'output_shapes': [[1, 84, 8400]],
        }
        decoders.append(decoder)
        record.update({'decoder_contract': decoder, 'detections': list(processor.last_detections), 'num_detections': len(processor.last_detections)})
    detection_path.write_text(json.dumps(payload))
    semantic['decoder_postprocess_contract']['contracts'] = decoders
    # Measured completion deliberately belongs to the OTHER image (tall), with
    # five frames, independently of the two candidate/GT records above.
    decoder = decoders[-1]
    declared = load_authoritative_output_contract(root, backend='deepx_m1', model_id='yolo11l', variant='full', task='detection')
    physical = runtime_output_contract('detection', outputs, raw_fallback=True, declared_contract=declared)
    attestation = build_completed_detection_endpoint_attestation(
        decoder['frozen_postprocess_contract'], decoder['frozen_postprocess_result'],
        completed_frames=5, postprocess_completed_frames=5,
        source_endpoint_contract_hash=physical['endpoint_contract_hash'],
    )
    measured_record = payload['images'][-1]
    measured = {
        **_performance_input_evidence(Path(measured_record['image']), measured_record['preprocessing_audit']),
        'completed_frames': 5, 'postprocess_included': True,
        'postprocess_completed_frames': 5, 'postprocess_completion_verified': True,
        'completed_task_stage': 'decoded_nms', 'completed_task_contract_family': 'decoded_nms',
        'completed_task_completion_mode': 'frozen_host_tail', 'completed_task_endpoint_attested': True,
        'completed_task_endpoint_attestation_status': 'passed',
        'frozen_host_postprocess_contract': decoder['frozen_postprocess_contract'],
        'completed_task_endpoint_attestation': attestation,
        **{key: attestation[key] for key in ('completed_task_comparison_endpoint_contract', 'completed_task_comparison_endpoint_contract_hash', 'completed_task_comparison_output_endpoint_id')},
    }
    exported = module._deepx_export_central_quality_request(
        root, dxnn, run, semantic, root / 'results/deepx_m1_full', completed_task_evidence=measured,
    )
    request_path = Path(exported['request']['path'])
    candidate_path = request_path.parent / exported['candidate']['path']
    candidate = json.loads(candidate_path.read_text())
    ground_truth = {record['image_id']: record['ground_truth'] for record in candidate['records']}
    cpu_contract = _cpu_contract(source=source, validation=root / 'validation', image_ids=sorted(ground_truth), ground_truth=ground_truth)
    reference = {
        'schema': 'onnx-splitpoint/task-quality-reference-input', 'schema_version': 1,
        'task': 'detection', 'pairing_key': 'image_id', 'reference_role': 'canonical_cpu_ort',
        'semantic_reference_only': True, 'provenance_required': True,
        'quality_contract': cpu_contract, 'quality_contract_sha256': cpu_contract['quality_contract_sha256'],
        'records': [
            {'image_id': record['image_id'], 'ground_truth': record['ground_truth'], 'reference': ([] if predicted == 'inconclusive' and record['image_id'] == 'synthetic_wide.jpg' else [{**box, 'score': .95} for box in record['ground_truth']])}
            for record in reversed(candidate['records'])
        ],
    }
    reference_path = root / 'SYNTHETIC_cpu_reference.json'
    reference_path.write_text(json.dumps(reference, sort_keys=True))
    return exported, candidate, request_path, reference_path


@pytest.fixture
def exported_case(tmp_path):
    return _synthetic_export(tmp_path)


def test_q06_q08_real_exporter_loader_image_pairing_and_variable_geometry(exported_case):
    exported, candidate, request_path, reference_path = exported_case
    loaded = service.quality_request_from_manifest(request_path, reference_artifact=reference_path)
    assert len(loaded.candidate_records) == len(loaded.reference_records) == 2
    producer = exported['producer_identity']
    assert producer['endpoint']['identity']['stage'] == 'decoded_pre_nms'
    assert producer['dataset']['image_count'] == 2
    assert producer['completed_task_endpoint_attestation']['completed_frames'] == 5
    first = producer['quality_record_endpoint']['identity']['decoder_contract']['frozen_postprocess_contract']
    measured = producer['completed_task_endpoint_attestation']['completed_endpoint_contract']
    assert first['original_wh'] != measured['original_wh']
    assert first['invariant_contract_sha256'] == measured['frozen_postprocess_invariant_contract_sha256']
    _, payload = service.prepare_evaluation(loaded)
    assert [item['image_id'] for item in payload['candidate_records']] == [item['image_id'] for item in payload['reference_records']]


def _reseal_synthetic(producer):
    """Rehash outer wrappers only; leaves mutated inner semantics intact."""
    quality = producer['quality_contract']
    for name in ('endpoint', 'quality_record_endpoint', 'preprocessing', 'precision'):
        producer[name]['sha256'] = json_fingerprint(producer[name]['identity'])
    producer['endpoint_contract_hash'] = producer['endpoint']['sha256']
    quality['quality_record_endpoint'] = copy.deepcopy(producer['quality_record_endpoint'])
    producer['quality_record_endpoint_contract_sha256'] = producer['quality_record_endpoint']['sha256']
    quality['quality_record_endpoint_contract_sha256'] = producer['quality_record_endpoint']['sha256']
    quality['quality_contract_sha256'] = service._quality_contract_digest(quality)
    producer['quality_contract_sha256'] = quality['quality_contract_sha256']
    if 'completed_task_endpoint_attestation' in producer:
        producer['completed_task_endpoint_attestation_sha256'] = json_fingerprint(producer['completed_task_endpoint_attestation'])
    producer.pop('producer_identity_sha256', None)
    producer['producer_identity_sha256'] = json_fingerprint(producer)
    return producer


@pytest.mark.parametrize('mutation,expected', [
    ('unknown_stage', 'runtime endpoint stage is invalid'),
    ('missing_nms', 'pre-NMS source or applied host decoder/NMS differs'),
    ('all_completion_missing', 'completed-task endpoint binding is incomplete'),
    ('one_completion_missing', 'completed-task endpoint binding is incomplete'),
    ('shape', 'tensor signature differs'),
    ('ordinal', 'tensor signature differs'),
    ('input_binding', 'prepared-input evidence binding is inconsistent'),
    ('frozen_invalid', 'frozen_postprocess_contract_sha256_mismatch'),
    ('source_semantics', 'pre-NMS source or applied host decoder/NMS differs'),
    ('integrated_nms', 'pre-NMS source or applied host decoder/NMS differs'),
    ('threshold', 'host decoder/NMS settings differ'),
    ('decoder_id', 'host decoder/NMS settings differ'),
    ('physical_hash', 'physical completion attestation differs'),
    ('completion_count', 'completed_endpoint_runtime_attestation_invalid'),
    ('completion_false', 'physical completion attestation differs'),
    ('model', 'quality model binding is inconsistent'),
    ('dataset', 'quality dataset binding is inconsistent'),
])
def test_q03_q07_semantic_negatives_reach_checks_with_resealed_synthetic_wrappers(exported_case, mutation, expected):
    producer = copy.deepcopy(exported_case[0]['producer_identity'])
    endpoint = producer['endpoint']['identity']
    quality = producer['quality_record_endpoint']['identity']
    if mutation == 'unknown_stage': endpoint['stage'] = 'unknown_synthetic'
    elif mutation == 'missing_nms': quality['host_decoder_nms']['nms_applied'] = False
    elif mutation == 'all_completion_missing':
        for field in COMPLETED_FIELDS: producer.pop(field)
    elif mutation == 'one_completion_missing': producer.pop(COMPLETED_FIELDS[0])
    elif mutation == 'shape': endpoint['tensor_signature']['tensors'][0]['shape'][1] = 85
    elif mutation == 'ordinal': endpoint['tensor_signature']['tensors'][0]['index'] = 1
    elif mutation == 'input_binding': producer['prepared_input_evidence']['record_count'] = 999
    elif mutation == 'frozen_invalid': quality['decoder_contract']['frozen_postprocess_contract']['decoder_id'] = 'unknown_synthetic'
    elif mutation == 'source_semantics': quality['source_endpoint']['semantics'] = 'raw_head'
    elif mutation == 'integrated_nms': quality['source_endpoint']['has_integrated_nms'] = True
    elif mutation == 'threshold': quality['host_decoder_nms']['iou_threshold'] = .5
    elif mutation == 'decoder_id': quality['host_decoder_nms']['decoder_id'] = 'SYNTHETIC_wrong_decoder'
    elif mutation == 'physical_hash': producer['completed_task_endpoint_attestation']['endpoint_contract_hash'] = 'f' * 64
    elif mutation == 'completion_count': producer['completed_task_endpoint_attestation']['postprocess_completed_frames'] = 4
    elif mutation == 'completion_false': producer['completed_task_endpoint_attestation']['postprocess_completion_verified'] = False
    elif mutation == 'model': producer['model']['source_onnx_sha256'] = 'e' * 64
    elif mutation == 'dataset': producer['dataset']['manifest_sha256'] = 'e' * 64
    _reseal_synthetic(producer)
    with pytest.raises(service.QualityArtifactIntegrityError, match=expected):
        _validate(producer)


def test_q07_corrupt_outer_integrity_still_rejected(exported_case):
    producer = copy.deepcopy(exported_case[0]['producer_identity'])
    producer['backend'] = 'modified-without-rehash'
    with pytest.raises(service.QualityArtifactIntegrityError, match='SHA-256 mismatch'):
        _validate(producer)


@pytest.mark.parametrize('mutation', ['candidate_bytes', 'duplicate_image_id', 'missing_image_id', 'dataset_reference'])
def test_q07_full_loader_rejects_payload_and_dataset_corruption(exported_case, mutation):
    _, candidate, request_path, reference_path = exported_case
    request = json.loads(request_path.read_text())
    candidate_path = request_path.parent / request['candidate']['path']
    if mutation == 'candidate_bytes':
        request['candidate']['sha256'] = '0' * 64
        request_path.write_text(json.dumps(request))
    elif mutation in ('duplicate_image_id', 'missing_image_id'):
        if mutation == 'duplicate_image_id':
            candidate['records'][1]['image_id'] = candidate['records'][0]['image_id']
        else:
            candidate['records'][1].pop('image_id')
        candidate_path.write_text(json.dumps(candidate))
        request['candidate']['sha256'] = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
        request['candidate']['size_bytes'] = candidate_path.stat().st_size
        request_path.write_text(json.dumps(request))
    elif mutation == 'dataset_reference':
        reference = json.loads(reference_path.read_text())
        reference['records'][0]['ground_truth'][0]['x1'] += 1
        reference_path.write_text(json.dumps(reference))
    expected = {
        'candidate_bytes': 'SHA-256 mismatch',
        'duplicate_image_id': 'duplicate image_id',
        'missing_image_id': "missing 'image_id'",
        'dataset_reference': 'ground truth differ',
    }[mutation]
    with pytest.raises((service.QualityArtifactIntegrityError, ValueError), match=expected):
        service.quality_request_from_manifest(request_path, reference_artifact=reference_path)


@pytest.mark.parametrize('predicted,decision', [(False, 'fail'), ('inconclusive', 'inconclusive')])
def test_f05_technically_loaded_candidate_retains_quality_decision(tmp_path, predicted, decision):
    _, _, request_path, reference_path = _synthetic_export(tmp_path, predicted=predicted)
    loaded = service.quality_request_from_manifest(request_path, reference_artifact=reference_path)
    _, payload = service.prepare_evaluation(loaded)
    result = service._evaluate_payload(payload)
    assert result['status'] == 'completed', result
    assert result['decision'] == decision, result


@pytest.mark.parametrize('field', COMPLETED_FIELDS)
def test_q03_each_completion_field_is_required(exported_case, field):
    producer = copy.deepcopy(exported_case[0]['producer_identity'])
    producer.pop(field)
    # Preserve the absence itself when rebuilding outer synthetic digests.
    producer.pop('producer_identity_sha256', None)
    producer['producer_identity_sha256'] = json_fingerprint(producer)
    with pytest.raises(service.QualityArtifactIntegrityError, match='completed-task endpoint binding is incomplete'):
        _validate(producer)


def test_q08_runtime_metadata_projection_does_not_require_unlike_source_hashes(exported_case):
    producer = copy.deepcopy(exported_case[0]['producer_identity'])
    frozen = producer['quality_record_endpoint']['identity']['decoder_contract']['frozen_postprocess_contract']
    producer['endpoint']['identity']['tensor_signature'] = copy.deepcopy(frozen['raw_output_tensor_signature'])
    _reseal_synthetic(producer)
    native_source_hash = producer['completed_task_endpoint_attestation']['completed_endpoint_contract']['source_endpoint_contract_hash']
    assert producer['endpoint_contract_hash'] != native_source_hash
    assert _validate(producer)[1] == producer['producer_identity_sha256']
