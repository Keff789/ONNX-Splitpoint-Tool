"""Stored CPU detector reuse never launches a model and keeps Split proofs."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from onnx_splitpoint_tool.native_command_contract import seal_native_command_contract
from onnx_splitpoint_tool.native_detection_postprocess import (
    DetectionCompletionRuntime, canonical_json_sha256,
    persist_detection_completion_execution_artifacts,
)
from scripts import native_producer_validate_visualize as validator
from test_completed_attestation_contract_repairs import _decoded_execution_contract
from test_v269f_native_split_final_energy_integrity import _quality_first_row


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


@pytest.fixture
def input_proof(tmp_path):
    row, output, _ = _quality_first_row(tmp_path)
    boundary = Path(row['native_fifo_boundary_manifest'])
    raw = boundary.parent / 'prepared_input.bin'
    raw.write_bytes(bytes(range(12)))
    artifacts = [{'path': raw.name, 'sha256': validator._sha256_file(raw), 'size_bytes': 12}]
    write(boundary, {
        'schema': 'onnx-splitpoint/native-boundary-dump',
        'file': str(boundary.parent / 'boundary.bin'), 'input_dump': raw.name,
        'input_shape_hwc': [2, 2, 3], 'input_dump_sha256': artifacts[0]['sha256'],
        'preprocess': {'mode': 'letterbox', 'pad_value': 114, 'rgb': True},
        'payload_artifacts': artifacts, 'payload_artifacts_sha256': canonical_json_sha256(artifacts),
    })
    command = copy.deepcopy(row['native_command_contract'])
    command['artifacts']['semantic_boundary_manifest'].update(
        sha256=validator._sha256_file(boundary), size_bytes=boundary.stat().st_size,
    )
    command = seal_native_command_contract(command)
    row.update(native_command_contract=command, native_command_contract_sha256=command['contract_sha256'],
               source_run_id=command['source_run_id'])
    attestation = row['native_split_quality_consumer_attestation']
    attestation.update(command_contract_sha256=command['contract_sha256'],
                       semantic_boundary_manifest_sha256=validator._sha256_file(boundary))
    attestation.pop('attestation_sha256')
    attestation['attestation_sha256'] = canonical_json_sha256(attestation)
    feed = {}
    assert validator._native_input_dump_feed(boundary, [1, 3, 2, 2], evidence=feed) is not None
    return row, output, boundary, raw, feed


def test_input_reuses_full_command_consumer_payload_verifier_read_only(input_proof):
    row, output, boundary, raw, feed = input_proof
    before = {p: p.read_bytes() for p in (output, boundary, raw)}
    result, evidence = validator._existing_split_reference_input(output, row, feed)
    assert result == boundary and evidence == feed
    assert all(p.read_bytes() == original for p, original in before.items())


@pytest.mark.parametrize('fault', ['run', 'setup', 'missing_case', 'request', 'command', 'consumer', 'payload', 'input', 'boundary', 'tensor', 'transform'])
def test_input_rejects_changed_origin_or_bytes(input_proof, fault):
    row, output, boundary, raw, feed = input_proof
    if fault == 'run': row['eval_run_id'] = 'other-run'
    elif fault == 'setup': row['setup_id'] = 'other-setup'
    elif fault == 'missing_case': row.pop('case')
    elif fault == 'request': row['source_request_sha256'] = 'f' * 64
    elif fault == 'command': row['native_command_contract_sha256'] = 'f' * 64
    elif fault == 'consumer': row['native_split_quality_consumer_attestation']['eval_run_id'] = 'other-run'
    elif fault == 'payload': (output.parent / 'output.bin').write_bytes(b'changed')
    elif fault == 'input': raw.write_bytes(bytes(reversed(range(12))))
    elif fault == 'boundary': boundary.write_text('{}')
    elif fault == 'tensor': feed['derived_reference_tensor_sha256'] = 'f' * 64
    elif fault == 'transform': feed['transformation_id'] = 'different-transform'
    with pytest.raises(ValueError):
        validator._existing_split_reference_input(output, row, feed)


@pytest.fixture
def detector_pair(tmp_path, monkeypatch):
    # The full Split proof is exercised above with the real Quality-FIRST
    # receipt fixture. Here isolate cross-case matching, retaining real input
    # conversion, completion contracts, attestations and persisted results.
    outputs = {'detections': np.asarray([[[8., 20., 32., 44., .9, 2.]]], dtype=np.float32)}
    execution = _decoded_execution_contract(outputs)
    model = tmp_path / 'model.onnx'; model.write_bytes(b'original-cpu-model')
    policy = validator.AccuracyGatePolicy()
    rows = []
    input_evidence = {}
    for run, case in [('original', 'b003'), ('augmentation', 'b066')]:
        directory = tmp_path / run; directory.mkdir()
        manifest = directory / 'outputs.json'; write(manifest, {})
        boundary = directory / 'boundary.json'
        raw = directory / 'input.bin'; raw.write_bytes(bytes(range(12)))
        write(boundary, {'input_dump': str(raw), 'input_shape_hwc': [2, 2, 3],
                         'input_dump_sha256': validator._sha256_file(raw),
                         'preprocess': {'mode': 'letterbox', 'pad_value': 114, 'rgb': True}})
        validator._native_input_dump_feed(boundary, [1, 3, 2, 2], evidence=input_evidence)
        runtime = DetectionCompletionRuntime(execution)
        runtime.process(outputs)
        attestation = runtime.attestation(completed_work_units=1)
        evidence = persist_detection_completion_execution_artifacts(
            {'completed_task_endpoint_attestation': attestation}, output_path=directory / 'completed_result.json',
        )
        rows.append({
            'backend': 'deepx_to_trt', 'model': 'yolo26s', 'case': case,
            'setup_id': 'deepx-setup', 'comparison_backend': 'deepx', 'precision': 'fp16',
            'eval_run_id': run, 'source_request_sha256': ('a' if run == 'original' else 'b') * 64,
            'model_sha256': validator._sha256_file(model), 'input_image_sha256': 'c' * 64,
            'dump_manifest': str(manifest), 'boundary_manifest': str(boundary),
            'native_command_contract': {'input_image_sha256': 'c' * 64,
                                       'prepared_input_contract': {'source_image_sha256': 'c' * 64}},
            'completed_task_completion_mode': 'detection_completion_execution_v1',
            'completion_execution_contract': execution, 'completed_task_endpoint_attestation': attestation,
            'completed_task_endpoint_contract': execution['completed_endpoint_contract'],
            'completed_task_comparison_endpoint_contract': execution['comparison_endpoint_contract'],
            'endpoint_contract_hash': execution['source_endpoint']['endpoint_contract_hash'], **evidence,
        })
    source, row = rows
    detections = source['completed_task_endpoint_attestation']['last_result']['detections']
    reference = {
        'completed_v2_verified': True, 'self_reference_input_manifest_kind': 'split_boundary_manifest',
        'full_onnx': str(model), 'boundary_manifest': source['boundary_manifest'],
        'reference_input_evidence': dict(input_evidence),
        'reference_detections': detections, 'full_reference_result_sha256': canonical_json_sha256(detections),
        'completed_task_comparison_endpoint_contract': execution['comparison_endpoint_contract'],
        'completed_task_comparison_endpoint_contract_hash': execution['comparison_endpoint_contract']['endpoint_contract_hash'],
        'numerical_similarity_threshold': policy.native_self_reference_min_match,
        'numerical_similarity_mean_iou_threshold': policy.native_self_reference_min_mean_iou,
        'numerical_similarity_policy_id': policy.native_self_reference_policy_id,
        'numerical_similarity_iou_threshold': policy.native_self_reference_iou_threshold,
        'numerical_similarity_confidence_threshold': policy.native_self_reference_confidence_threshold,
    }
    report = tmp_path / 'original' / 'validation.json'
    source['task_validation'] = str(report)
    saved = {'schema': 'onnx-splitpoint/native-detection-semantic-validation',
             'manifest': source['dump_manifest'], 'self_reference_report': reference}
    write(report, saved)
    def derive(manifest, item, expected):
        path = Path(item['boundary_manifest']); observed = {}
        validator._native_input_dump_feed(path, expected['target_shape'], evidence=observed)
        if observed != expected: raise ValueError('derived tensor differs')
        return path, observed
    def forbidden(*args, **kwargs):
        raise AssertionError('CPU inference or probe called')
    monkeypatch.setattr(validator, '_existing_split_reference_input', derive)
    monkeypatch.setattr(validator, 'load_dump', lambda *args: (outputs, {}))
    monkeypatch.setattr(validator, '_full_onnx_self_reference_detection', forbidden)
    monkeypatch.setattr(validator, '_auto_run_yolo_full_self_reference_probe', forbidden)
    return row, source, saved, policy


def compare(pair):
    row, source, _, policy = pair
    return validator._existing_split_reference_detection(Path(row['dump_manifest']), row, [source], policy)


def test_different_new_run_and_boundary_reuse_original_cpu_result(detector_pair):
    row, source, _, _ = detector_pair
    before = Path(source['task_validation']).read_bytes()
    result = compare(detector_pair)
    assert result['available'] is True, result
    assert result['numerical_similarity_pass'] is True
    assert result['reference_eval_run_id'] == 'original'
    assert result['reference_case'] == 'b003' and row['case'] == 'b066'
    assert result['self_reference_input_manifest_kind'] == 'split_boundary_manifest'
    assert result['semantic_result_binding_status'] == 'exact_same_hotloop_completed_artifact'
    assert Path(source['task_validation']).read_bytes() == before


@pytest.mark.parametrize('fault', ['model', 'setup', 'precision', 'image', 'input', 'processor', 'endpoint', 'attestation', 'artifact', 'reference_hash', 'reference_kind', 'policy', 'source_manifest'])
def test_stored_reference_requires_exact_model_input_completion_and_artifact(detector_pair, fault):
    row, source, saved, _ = detector_pair
    if fault == 'model': row['model_sha256'] = 'f' * 64
    elif fault == 'setup': row['setup_id'] = 'other'
    elif fault == 'precision': row['precision'] = 'fp32'
    elif fault == 'image': row['input_image_sha256'] = 'f' * 64
    elif fault == 'input': Path(row['boundary_manifest']).with_name('input.bin').write_bytes(bytes(reversed(range(12))))
    elif fault == 'processor': row['completion_execution_contract'] = copy.deepcopy(row['completion_execution_contract']); row['completion_execution_contract']['processor_contract']['iou_threshold'] = .8
    elif fault == 'endpoint': row['endpoint_contract_hash'] = 'f' * 64
    elif fault == 'attestation': row['completed_task_endpoint_attestation']['exact_result_claim_bound'] = False
    elif fault == 'artifact': Path(row['completed_task_result_artifact_path']).write_text('{}')
    elif fault == 'reference_hash': saved['self_reference_report']['full_reference_result_sha256'] = 'f' * 64
    elif fault == 'reference_kind': saved['self_reference_report']['self_reference_input_manifest_kind'] = 'unbound'
    elif fault == 'policy': saved['self_reference_report']['numerical_similarity_threshold'] = .1
    elif fault == 'source_manifest': saved['manifest'] = '/wrong/manifest.json'
    write(Path(source['task_validation']), saved)
    assert compare(detector_pair)['available'] is False


def test_negative_original_measurement_does_not_select_another_reference(detector_pair):
    row, source, saved, _ = detector_pair
    source.update(semantic_ok=False, numerical_similarity_pass=False, accuracy_gate_pass=False)
    saved['self_reference_report'].update(ok=False, numerical_similarity_pass=False)
    write(Path(source['task_validation']), saved)
    assert compare(detector_pair)['available'] is True


def test_visualizer_offline_split_path_does_not_call_cpu(detector_pair, tmp_path):
    row, source, _, policy = detector_pair
    out = tmp_path / 'visual'; out.mkdir()
    result = validator._visualize_detection(Path(row['dump_manifest']), out, 0, None, tmp_path,
                                           policy=policy, endpoint_evidence=row,
                                           existing_outputs_only=True, existing_reference_rows=[source])
    assert result['self_reference_report']['available'] is True
    assert result['semantic_ok'] is True
