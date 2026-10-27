from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import importlib.util
import hashlib
import json
import ast
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from onnx_splitpoint_tool import native_dataset_quality_consumer as consumer
from onnx_splitpoint_tool.native_split_quality import seal_native_split_quality_binding
from onnx_splitpoint_tool.quality_cache import json_fingerprint
from onnx_splitpoint_tool.quality_service import QualityArtifactIntegrityError, QualityEvaluationRequest, _validate_candidate_execution_contract


def fixture_module(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def producer(tmp_path):
    receipt = fixture_module('test_v269f_native_split_receipt_validation')
    payload, _ = receipt._fixture_payload(tmp_path)
    prior = seal_native_split_quality_binding(payload)
    quality = fixture_module('test_v269d_trt_central_quality_producer')._detection_quality_contract()
    descriptor = {'path': 'actual-receipt.json', 'sha256': 'a' * 64, 'size_bytes': 10}
    root = Path(consumer.__file__).resolve().parents[1]
    source_descriptors = []
    for relative in sorted(consumer.required_native_dataset_source_paths('hailo8_to_trt')):
        source = root / relative
        source_descriptors.append({'path': str(source), 'relative_path': relative,
            'sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'size_bytes': source.stat().st_size})
    endpoint = {'endpoint_contract_hash': 'd' * 64}
    completion = {'completed_endpoint_contract': endpoint}
    completion['contract_sha256'] = json_fingerprint(completion)
    value = {
        'schema': consumer.NATIVE_DATASET_PRODUCER_SCHEMA, 'schema_version': 1,
        'producer_kind': 'native_dataset_completed_quality', 'task': 'detection', 'variant': 'composed',
        'execution_role': 'native_dataset_quality_candidate', 'performance_claims_emitted': False,
        'additional_nms_applied': False, 'eval_run_id': 'fresh_native_capture',
        'model_id': 'yolo26s', 'case_id': 'b038', 'setup_id': 'hailo8_setup',
        'source_run_id': 'hailo8_to_trt', 'backend': 'hailo8_to_trt', 'runtime_precision_identity': 'uint8_dequant_fp16',
        'prior_artifact_selection': prior, 'preselection': deepcopy(prior['preselection']), 'artifacts': deepcopy(prior['artifacts']),
        'quality_contract': quality, 'dataset': quality['dataset'], 'preprocessing': quality['preprocessing'],
        'model': {'source_onnx_sha256': quality['model']['sha256']}, 'source_model_sha256': quality['model']['sha256'],
        'record_count': 1, 'completed_frames': 1, 'candidate_records_sha256': 'b' * 64,
        'completion_evidence': descriptor, 'timed_fps_authority': descriptor, 'implementation_source_files': source_descriptors,
        'endpoint_contract': endpoint, 'endpoint_contract_hash': endpoint['endpoint_contract_hash'],
        'candidate_execution_completion_contract': completion,
        'candidate_execution_completion_contract_sha256': completion['contract_sha256'],
    }
    for name, source in [('quality_contract_sha256', quality['quality_contract_sha256']),
                         ('preprocessing_contract_sha256', quality['preprocessing']['sha256']),
                         ('decoder_contract_sha256', quality['decoder']['sha256']),
                         ('nms_contract_sha256', quality['nms']['sha256']),
                         ('quality_record_endpoint_contract_sha256', quality['quality_record_endpoint']['sha256'])]:
        value[name] = source
    return seal(value)


def seal(value):
    value = deepcopy(value)
    value.pop('producer_identity_sha256', None)
    value['producer_identity_sha256'] = json_fingerprint(value)
    return value


def test_native_dispatch_preserves_prior_authority_and_fresh_identity(producer):
    validated, digest = _validate_candidate_execution_contract(producer, role='request', task='detection')
    assert validated == producer and digest == producer['producer_identity_sha256']
    assert validated['eval_run_id'] != validated['prior_artifact_selection']['eval_run_id']


@pytest.mark.parametrize('mutation', ['artifact', 'old_eval', 'extra_nms', 'count', 'generic_kind', 'tampered_seal', 'source', 'missing_source', 'duplicate_source'])
def test_native_producer_rejects_unsafe_associations(producer, mutation):
    value = deepcopy(producer)
    if mutation == 'artifact':
        value['artifacts']['engine']['sha256'] = 'c' * 64
    elif mutation == 'old_eval':
        value['eval_run_id'] = value['prior_artifact_selection']['eval_run_id']
    elif mutation == 'extra_nms':
        value['additional_nms_applied'] = True
    elif mutation == 'count':
        value['completed_frames'] = 2
    elif mutation == 'generic_kind':
        value['producer_kind'] = 'generic_composed_quality_candidate'
    elif mutation == 'source':
        value['implementation_source_files'][0]['sha256'] = '0' * 64
    elif mutation == 'missing_source':
        value['implementation_source_files'].pop()
    elif mutation == 'duplicate_source':
        value['implementation_source_files'].append(deepcopy(value['implementation_source_files'][0]))
    else:
        value['producer_identity_sha256'] = '0' * 64
    if mutation != 'tampered_seal':
        value = seal(value)
    with pytest.raises(QualityArtifactIntegrityError):
        consumer.validate_native_dataset_quality_producer(value, role='request', task='detection')


def test_completion_records_use_normal_verifier_and_exact_actual_detections(tmp_path, monkeypatch):
    from onnx_splitpoint_tool import native_detection_postprocess as postprocess
    from onnx_splitpoint_tool.preprocessing_contract import canonical_image_preprocessing_contract
    from onnx_splitpoint_tool.quality_service import _load_quality_artifact
    records = [{'image_id': 'a', 'candidate': []}]
    preprocessing = canonical_image_preprocessing_contract('detection', [2, 2])
    contract = {'completion_mode': 'actual', 'model_id': 'yolo26s', 'model_family': 'yolo', 'execution_policy': 'normal',
                'source_endpoint': {'actual': True}, 'processor_contract': {'input_hw': [2, 2], 'iou_threshold': .45}}
    row = {'image_id': 'a', 'image_sha256': 'a' * 64, 'execution_contract': deepcopy(contract),
           'execution_attestation': {'observation_relation': 'same_hotloop_sentinel'},
           'prepared_input_snapshot': {'inputs': [{'name': 'input', 'dtype': 'uint8', 'shape': [1, 2, 2, 3], 'sha256': 'c' * 64}]},
           'timing_input_binding': {'input_image_sha256': 'a' * 64, 'preprocessing_contract': preprocessing,
                                    'preprocessing_contract_sha256': json_fingerprint(preprocessing)}, 'boundary_proof': {'actual': True}}
    payload = {'schema': 'onnx-splitpoint/native-dataset-quality-completion-evidence', 'schema_version': 1,
               'records': [row], 'record_count': 1, 'additional_nms_applied': False}
    evidence = tmp_path / 'completion.json'
    encoded = json.dumps(payload).encode()
    evidence.write_bytes(encoded)
    import hashlib
    descriptor = {'path': evidence.name, 'sha256': hashlib.sha256(encoded).hexdigest(), 'size_bytes': len(encoded)}
    producer = {'completion_evidence': descriptor, 'record_count': 1, 'candidate_records_sha256': json_fingerprint(records),
                'candidate_execution_completion_contract': deepcopy(contract),
                'quality_contract': {'preprocessing': {'identity': preprocessing}},
                'prior_artifact_selection': {'preselection': {'preprocess_mode': 'letterbox', 'letterbox_pad_value': 114}}}
    seen = []
    def verify(raw, **kwargs):
        seen.append(kwargs)
        return {'completed_work_units': 1, 'completion_count': 1, 'last_result': {'detections': []}}
    monkeypatch.setattr(postprocess, 'verify_detection_completion_execution_attestation', verify)
    monkeypatch.setattr(postprocess, 'verify_detection_completion_execution_contract', lambda value: deepcopy(value))
    consumer.validate_native_dataset_completion_evidence(producer, tmp_path / 'composed_request.json', records)
    assert seen == [{'execution_contract': contract, 'expected_observation_relation': 'same_hotloop_sentinel'}]
    for wrong_source in [False, True]:
        altered = deepcopy(payload)
        if wrong_source:
            altered['records'][0]['execution_contract']['source_endpoint'] = {'different': True}
        else:
            altered['records'][0]['execution_contract']['processor_contract']['iou_threshold'] = .9
        encoded_altered = json.dumps(altered).encode()
        evidence.write_bytes(encoded_altered)
        producer['completion_evidence'] = dict(descriptor, sha256=hashlib.sha256(encoded_altered).hexdigest(), size_bytes=len(encoded_altered))
        with pytest.raises(QualityArtifactIntegrityError, match='policy differs|source endpoint differs'):
            consumer.validate_native_dataset_completion_evidence(producer, tmp_path / 'composed_request.json', records)
    evidence.write_bytes(encoded)
    producer['completion_evidence'] = descriptor
    changed = deepcopy(records)
    changed[0]['candidate'] = [{'x1': 0}]
    producer['candidate_records_sha256'] = json_fingerprint(changed)
    with pytest.raises(QualityArtifactIntegrityError, match='actual completion'):
        consumer.validate_native_dataset_completion_evidence(producer, tmp_path / 'composed_request.json', changed)
    evidence.write_text('{}')
    producer['candidate_records_sha256'] = json_fingerprint(records)
    with pytest.raises(QualityArtifactIntegrityError, match='size mismatch|SHA-256 mismatch'):
        consumer.validate_native_dataset_completion_evidence(producer, tmp_path / 'composed_request.json', records)


def test_consumer_reuses_sealed_cpu_handoff_and_injected_service(tmp_path, monkeypatch):
    from onnx_splitpoint_tool import quality_service
    from onnx_splitpoint_tool.workflow.runner import EvaluationWorkflowRunner as Runner
    request_path = tmp_path / 'quality_inputs/hailo8_setup/results/b038/results_hailo8_to_trt/task_quality_inputs/composed_request.json'
    request_path.parent.mkdir(parents=True)
    request_path.write_text(json.dumps({'model_id': 'yolo26s', 'variant': 'composed', 'metric_gate_config': {}, 'policy_sha256': 'c' * 64,
                                      'producer_identity': {'schema': consumer.NATIVE_DATASET_PRODUCER_SCHEMA}}))
    reference = {'path': str(tmp_path / 'canonical_cpu_reference.json'), 'sha256': 'a' * 64, 'size_bytes': 6}
    status = {**reference, 'reference_path': reference['path'], 'reference_sha256': reference['sha256'], 'reference_size_bytes': 6}
    monkeypatch.setattr(Runner, '_read_management_reference_status', classmethod(lambda cls, **kwargs: ([], status, b'sealed')))
    loaded_request = QualityEvaluationRequest([], [{'image_id': i} for i in range(5000)], [], {}, 1000, 42, .95, .05)
    seen = []
    def load(path, **kwargs):
        seen.append(kwargs)
        return loaded_request
    monkeypatch.setattr(quality_service, 'quality_request_from_manifest', load)
    identity = {'identity_valid': True, 'producer_identity_validated': True, 'producer_identity': {'schema': consumer.NATIVE_DATASET_PRODUCER_SCHEMA},
                'artifact_provenance_claim_eligible': True, 'source_run_id': 'hailo8_to_trt', 'case_id': 'b038'}
    monkeypatch.setattr(Runner, '_quality_request_identity', staticmethod(lambda *args, **kwargs: identity))
    monkeypatch.setattr(Runner, '_central_reference_contract', lambda *args: (SimpleNamespace(fingerprint=lambda: 'original_cpu'), {'manifest_path': 'normal-store.json'}))
    monkeypatch.setattr(quality_service, 'ManagementQualityService', lambda *args, **kwargs: pytest.fail('consumer created a second pool'))
    loaded = consumer.load_native_quality_request(request_path, run_dir=tmp_path, model_id='yolo26s',
        reference_run_dir=tmp_path, reference_artifact=reference, reference_status=status,
        expected_reference_source_contract_sha256='b' * 64,
        expected_metric_gate_config={}, expected_policy_sha256='c' * 64, expected_seed=42)
    assert seen == [{'reference_artifact': reference, 'reference_artifact_bytes': b'sealed'}]
    assert loaded.request.reference_identity == 'original_cpu'
    injected = SimpleNamespace(evaluate=lambda request: {'status': 'completed', 'decision': 'accuracy_loss'})
    result = consumer.evaluate_native_quality_request(loaded, service=injected)
    assert result['scientific_status'] == 'accuracy_loss' and result['technical_status'] == 'completed'
    assert result['collection_eval_run_id'] == tmp_path.name
    loaded_request = replace(loaded_request, repetitions=999)
    with pytest.raises(QualityArtifactIntegrityError, match='N5000/B1000'):
        consumer.load_native_quality_request(request_path, run_dir=tmp_path, model_id='yolo26s', reference_run_dir=tmp_path,
            reference_artifact=reference, reference_status=status, expected_reference_source_contract_sha256='b' * 64,
            expected_metric_gate_config={}, expected_policy_sha256='c' * 64, expected_seed=42)
    loaded_request = replace(loaded_request, repetitions=1000, seed=43)
    with pytest.raises(QualityArtifactIntegrityError, match='seed/confidence'):
        consumer.load_native_quality_request(request_path, run_dir=tmp_path, model_id='yolo26s', reference_run_dir=tmp_path,
            reference_artifact=reference, reference_status=status, expected_reference_source_contract_sha256='b' * 64,
            expected_metric_gate_config={}, expected_policy_sha256='c' * 64, expected_seed=42)
    with pytest.raises(QualityArtifactIntegrityError, match='original quality policy'):
        consumer.load_native_quality_request(request_path, run_dir=tmp_path, model_id='yolo26s', reference_run_dir=tmp_path,
            reference_artifact=reference, reference_status=status, expected_reference_source_contract_sha256='b' * 64,
            expected_metric_gate_config={'wrong': True}, expected_policy_sha256='c' * 64, expected_seed=42)


@pytest.mark.parametrize('canonical_coco', [False, True])
def test_actual_native_class_id_only_has_identical_ap_to_named_cpu_class(canonical_coco):
    import numpy as np
    from onnx_splitpoint_tool.accuracy_reporting import DEFAULT_REPORTING_POLICY
    from onnx_splitpoint_tool.quality_metrics import detection_quality_evaluator
    # Both retained CPU references encode COCO "person" as contiguous class 0.
    # Native completion records contain exactly these six fields, without names.
    native = {'x1': 10., 'y1': 20., 'x2': 40., 'y2': 60., 'score': .9, 'class_id': 0}
    named = dict(native, class_name='person')
    ground_truth = [dict(named, score=1.)]
    reference = [{'image_id': 'person', 'ground_truth': ground_truth, 'reference': [named]}]
    annotations = [{'image_id': 'person', 'ground_truth': ground_truth}]
    config = {'non_inferiority_margin': .05, 'guardrails': {'ap50_margin': .05, 'ap75_margin': .05}}
    if canonical_coco:
        config['reporting_policy'] = deepcopy(DEFAULT_REPORTING_POLICY)
    def metrics(detection):
        candidate = [{'image_id': 'person', 'ground_truth': ground_truth, 'candidate': [detection]}]
        return detection_quality_evaluator(reference, candidate, annotations, config).evaluate(np.array([1]))
    actual = metrics(native)
    assert actual == metrics(named)
    assert actual['primary']['candidate'] == pytest.approx(1.)
    assert actual['primary']['reference'] == pytest.approx(1.)
    # COCO category id 1 must not be confused with the model's contiguous id 0.
    assert metrics(dict(native, class_id=1))['primary']['candidate'] == 0.


def test_explicit_published_producer_scope_keeps_old_hashes_and_same_science(tmp_path, monkeypatch, producer):
    from onnx_splitpoint_tool import source_integrity
    current = Path(consumer.__file__).resolve().parents[1]
    published = tmp_path / 'published_source'
    # Copy only the finite source closure and statistical consumers used by the
    # scope, leaving the installed-source verifier as the separately tested boundary.
    relatives = consumer.required_native_dataset_source_paths('hailo8_to_trt') | consumer.required_native_dataset_source_paths('hailo10_to_trt')
    relatives |= {'onnx_splitpoint_tool/' + name for name in (
        'quality_service.py', 'quality_metrics.py', 'quality_cache.py', 'quality_statistics_config.py',
        'accuracy_reporting.py', 'quality_result_enrichment.py', 'quality_lifecycle.py', 'preprocessing_contract.py',
        'native_command_contract.py', 'native_split_quality.py', 'native_output_endpoint.py')}
    relatives.add('onnx_splitpoint_tool/workflow/runner.py')
    for name in relatives:
        target = published / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(current / name, target)
    capture = published / 'onnx_splitpoint_tool/native_dataset_quality.py'
    tree = ast.parse(capture.read_text())
    next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'load_capture_config').body.insert(0, ast.Pass())
    capture.write_text(ast.unparse(tree) + '\n')
    manifest = published / 'SOURCE_MANIFEST.json'
    manifest.write_text('{}')
    def descriptor(path):
        encoded = path.read_bytes()
        return {'path': str(path), 'sha256': hashlib.sha256(encoded).hexdigest(), 'size_bytes': len(encoded)}
    release_path = tmp_path / 'SOURCEFREEZE_READY.json'
    release = {'pushed': True, 'commit': 'c' * 40, 'source_export_root': str(published), 'source_manifest': descriptor(manifest)}
    release_path.write_text(json.dumps(release))
    release_descriptor = descriptor(release_path)
    calls = []
    monkeypatch.setattr(source_integrity, 'verify_installed_source_integrity', lambda root: calls.append(root) or {'ok': True})
    old_producer = deepcopy(producer)
    for source in old_producer['implementation_source_files']:
        source.update(descriptor(published / source['relative_path']))
    old_producer = seal(old_producer)
    # A root supplied inside a candidate is never trusted.
    old_producer['source_export_root'] = str(published)
    old_producer = seal(old_producer)
    with pytest.raises(QualityArtifactIntegrityError, match='source differs'):
        consumer.validate_native_dataset_quality_producer(old_producer, role='request', task='detection')
    with consumer.published_native_producer_source_scope(release_descriptor) as proof:
        assert proof['commit'] == 'c' * 40
        assert consumer.validate_native_dataset_quality_producer(old_producer, role='request', task='detection')[0] == old_producer
    assert calls == [published]
    with pytest.raises(QualityArtifactIntegrityError, match='source differs'):
        consumer.validate_native_dataset_quality_producer(old_producer, role='request', task='detection')
    # A published location cannot excuse a different metric/runtime implementation.
    metrics = published / 'onnx_splitpoint_tool/quality_metrics.py'
    metrics.write_text(metrics.read_text() + '\n# different source\n')
    with pytest.raises(QualityArtifactIntegrityError, match='scientific or runtime source differs'):
        with consumer.published_native_producer_source_scope(release_descriptor):
            pytest.fail('different scientific source admitted')


def test_h10_source_paths_follow_normal_backend_alias_canonicalization():
    expected = consumer.required_native_dataset_source_paths('hailo10h_to_trt')
    assert len(expected) == 7
    assert consumer.required_native_dataset_source_paths('hailo10_to_trt') == expected
    assert consumer.required_native_dataset_source_paths('hailo10_to_tensorrt') == expected
