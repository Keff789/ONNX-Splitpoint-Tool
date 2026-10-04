"""A new local validation cannot replace retained functional measurement proof."""
import copy

import pytest

from onnx_splitpoint_tool.workflow.yolo_augmentation import _same_completion_reference


def reference():
    return {
        'historical_semantic_ok': 'unavailable', 'quality_gate_status': 'reported',
        'source_report': '/original/native.json', 'output_manifest': '/original/outputs.json',
        'input_image_sha256': 'original-input', 'model_sha256': 'original-model',
        'output_manifest_sha256': 'original-manifest',
        'outputs': [{'path': '/original/output.bin', 'sha256': 'original-output'}],
        'completed_result_path': '/original/completed.json',
        'completed_result_file_sha256': 'original-completed',
        'validation_identity': {'model': 'yolo11l', 'case': 'b120', 'setup_id': 'h8'},
    }


def test_new_local_validation_keeps_original_reference_and_claim_annotation():
    old = reference()
    new = {**copy.deepcopy(old), 'historical_semantic_ok': True}
    before = copy.deepcopy(old), copy.deepcopy(new)
    assert _same_completion_reference(old, new)
    assert (old, new) == before


@pytest.mark.parametrize('field', [
    'source_report', 'output_manifest', 'input_image_sha256', 'model_sha256',
    'output_manifest_sha256', 'outputs', 'completed_result_path',
    'completed_result_file_sha256', 'validation_identity', 'quality_gate_status',
])
def test_new_validation_never_permits_changed_underlying_proof_or_claim(field):
    old = reference()
    new = {**copy.deepcopy(old), 'historical_semantic_ok': True, field: 'changed'}
    assert not _same_completion_reference(old, new)


@pytest.mark.parametrize('old_status,new_status', [
    (True, False), (True, 'unavailable'), ('unavailable', False),
    (False, True), (None, True), ('unavailable', 'true'),
])
def test_other_semantic_transitions_remain_rejected(old_status, new_status):
    old = {**reference(), 'historical_semantic_ok': old_status}
    new = {**copy.deepcopy(old), 'historical_semantic_ok': new_status}
    assert not _same_completion_reference(old, new)
