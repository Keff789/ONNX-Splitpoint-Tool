"""Separately valid producer proofs must describe the same candidate."""
import hashlib
from pathlib import Path

import pytest

from onnx_splitpoint_tool.native_split_quality import validate_native_split_quality_binding
from onnx_splitpoint_tool.quality_cache import json_fingerprint
from onnx_splitpoint_tool.quality_service import (
    QualityArtifactIntegrityError, _validate_candidate_execution_contract,
    _validate_generic_native_split_link,
)
from tests.test_v269d_trt_central_quality_producer import _detection_quality_contract
from tests.test_v27911_composed_provenance_binding import _functions, RUNNER
from tests.test_yolo_augmentation_quality_release import binding, STARTS


def proofs(tmp_path, changed=None):
    model, case, setup, backend = STARTS[0]
    native = binding(tmp_path / 'native', model, case, setup, backend, 'augmentation')
    native, status = validate_native_split_quality_binding(native)
    assert native is not None, status
    full = tmp_path / 'full.onnx'; full.write_bytes(b'closed-original-full')
    part1 = tmp_path / 'part1.onnx'; part1.write_bytes(b'original-part1')
    quality = _detection_quality_contract()
    quality['model']['sha256'] = hashlib.sha256(full.read_bytes()).hexdigest()
    quality.pop('quality_contract_sha256')
    quality['quality_contract_sha256'] = json_fingerprint(quality)
    context = dict(eval_run_id='augmentation', model_id=model, case_id=case, setup_id=setup,
        source_run_id=backend, stage1_backend='hailo8', stage2_backend='tensorrt',
        source_model_path=str(full), part1_model_path=str(part1),
        part2_model_path=native['artifacts']['source_part2_onnx']['path'],
        implementation_runner_path=str(RUNNER), runtime_artifacts=[
            dict(role='stage1_runtime', backend='hailo8', path=native['artifacts']['part1_runtime']['path']),
            dict(role='stage2_runtime', backend='native_tensorrt', path=native['artifacts']['engine']['path'])])
    precision = native['preselection']['precision']
    if changed in {'stage1_runtime', 'stage2_runtime', 'part2_model_path'}:
        other = tmp_path / 'other-artifact'; other.write_bytes(b'independently-valid-other-artifact')
        if changed == 'part2_model_path':
            context[changed] = str(other)
        else:
            next(row for row in context['runtime_artifacts'] if row['role'] == changed)['path'] = str(other)
    elif changed == 'precision':
        precision = 'fp16'
    elif changed:
        context[changed] = 'deepx_to_trt' if changed == 'source_run_id' else 'different-' + context[changed]
    namespace = _functions(['_quality_json_safe', '_quality_file_sha256', '_quality_contract_sha256',
        '_quality_artifact_binding', '_build_generic_composed_quality_producer_binding'])
    producer, _ = namespace['_build_generic_composed_quality_producer_binding'](
        context=context, task='detection', quality_contract=quality,
        endpoint_contract={'endpoint_contract_complete':True, 'endpoint_contract_hash':'2' * 64,
                           'stage':'post_nms_detections', 'contract_family':'detections'},
        runtime_precision_identity=precision)
    producer, _ = _validate_candidate_execution_contract(producer, role='candidate', task='detection')
    return producer, native


def test_matching_independently_verified_proofs_join(tmp_path):
    _validate_generic_native_split_link(*proofs(tmp_path))


@pytest.mark.parametrize('changed', ['eval_run_id', 'model_id', 'case_id', 'setup_id',
    'source_run_id', 'precision', 'stage1_runtime', 'stage2_runtime', 'part2_model_path'])
def test_valid_proof_from_another_identity_or_artifact_does_not_join(tmp_path, changed):
    producer, native = proofs(tmp_path, changed)
    with pytest.raises(QualityArtifactIntegrityError, match='Generic/Native split'):
        _validate_generic_native_split_link(producer, native)
