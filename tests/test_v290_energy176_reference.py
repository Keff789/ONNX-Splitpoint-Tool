"""Offline replay of the observed management CPU coordinate contract failure."""
import copy
import json
import os
from pathlib import Path

import numpy as np
import pytest

from onnx_splitpoint_tool import management_reference as ref
from onnx_splitpoint_tool import native_output_endpoint as endpoint
from onnx_splitpoint_tool.native_detection_postprocess import (
    FrozenDecodedNmsPostprocessor, build_frozen_decoded_nms_normalization_contract,
)

BRINGUP = Path('/home/kmika/Models/EvaluationRuns/native_resnet_yolo26s_hailo10h_bringup_v1_20260921_143220')
SUITE = BRINGUP / 'models/yolo26s/benchmark_set/legacy_suite'


def clone(tmp_path, model='renamed_detector', boundary='another_boundary'):
    suite = tmp_path / 'suite'; suite.mkdir()
    original = SUITE / 'output_contracts.json'
    payload = json.loads(original.read_text())
    payload['model_id'] = model
    for row in payload['contracts']: row['model_id'] = model
    source = tmp_path / 'original_sidecar.json'
    source.write_text(json.dumps(payload))
    os.link(source, suite / 'output_contracts.json')
    case = suite / boundary; case.mkdir()
    (case / 'split_manifest.json').write_text(json.dumps({'full_model': str(SUITE / 'models/yolo26s.onnx')}))
    return suite, source


def test_archived_input_cpu_output_transfer_same_graph_keep_source_immutable(tmp_path):
    suite, original = clone(tmp_path)
    before = original.read_bytes()
    unavailable = endpoint.load_authoritative_output_contract(suite, backend='cpu', model_id='renamed_detector', task='detection')
    assert not unavailable.get('source_coordinate_space')
    assert ref._bind_cpu_reference_output_contract(suite, 'renamed_detector')
    assert original.read_bytes() == before
    declaration = endpoint.load_authoritative_output_contract(suite, backend='cpu', model_id='renamed_detector', task='detection')
    cuda = endpoint.load_authoritative_output_contract(suite, backend='cuda', model_id='renamed_detector', task='detection')
    assert declaration['source_coordinate_space'] == cuda['source_coordinate_space']
    # The old management workspace was cleaned by its owner. Replay one saved
    # prepared input on CPU instead of claiming that missing outputs survived.
    import onnxruntime as ort
    saved = next((BRINGUP / 'native_producers/hailo10h/yolo26s').rglob('runtime_input.bin'))
    values = np.fromfile(saved, dtype=np.uint8).reshape(640,640,3)
    options = ort.SessionOptions(); options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(SUITE / 'models/yolo26s.onnx'), options, providers=['CPUExecutionProvider'])
    result = session.run(None, {'images': values.transpose(2,0,1)[None].astype(np.float32) / 255.0})
    outputs = dict(zip([o.name for o in session.get_outputs()], result))
    np.savez(tmp_path / 'replayed_cpu_outputs.npz', **outputs)
    observed = endpoint.runtime_output_contract('detection', outputs, raw_fallback=True, declared_contract=declaration)
    contract = build_frozen_decoded_nms_normalization_contract(model_id='renamed_detector',
        outputs=outputs, input_hw=[640,640], original_wh=[640,483],
        preprocess={'mode': 'letterbox', 'letterbox_pad_value': 114, 'color_space': 'RGB'},
        source_completed=True, source_coordinate_space=declaration['source_coordinate_space'],
        source_endpoint_contract_hash=observed['endpoint_contract_hash'],
        source_output_endpoint_attestation=observed['output_endpoint_attestation'])
    processor = FrozenDecodedNmsPostprocessor(contract)
    processor.process(outputs)
    assert processor.completed_count == 1
    before_cpu = (suite / 'output_contracts.json').read_bytes()
    assert not ref._bind_cpu_reference_output_contract(suite, 'renamed_detector')
    assert (suite / 'output_contracts.json').read_bytes() == before_cpu


@pytest.mark.parametrize('damage', ['missing_coordinate', 'foreign_model', 'foreign_graph', 'missing_proof'])
def test_missing_or_foreign_coordinate_evidence_is_not_promoted(tmp_path, damage):
    suite, _ = clone(tmp_path)
    p = suite / 'output_contracts.json'; payload = json.loads(p.read_text())
    row = payload['contracts'][0]
    if damage == 'missing_coordinate': row.pop('source_coordinate_space')
    elif damage == 'foreign_model': row['model_id'] = 'foreign'
    elif damage == 'foreign_graph': row['source_onnx_detection_endpoint']['candidate_selection']['source_onnx_sha256'] = 'a' * 64
    else: row['source_onnx_detection_endpoint'].pop('candidate_selection')
    p.write_text(json.dumps(payload))
    if damage == 'foreign_graph':
        with pytest.raises(ValueError, match='graph_contract_mismatch'):
            ref._bind_cpu_reference_output_contract(suite, 'renamed_detector')
    else:
        assert not ref._bind_cpu_reference_output_contract(suite, 'renamed_detector')
    assert not endpoint.load_authoritative_output_contract(suite, backend='cpu', model_id='renamed_detector', task='detection').get('stage')


def test_coordinate_sidecar_participates_in_reference_identity(tmp_path):
    suite, _ = clone(tmp_path)
    before = ref._source_contract(suite, {}, {})
    p = suite / 'output_contracts.json'; payload = json.loads(p.read_text())
    payload['contracts'][0].pop('source_coordinate_space')
    p.write_text(json.dumps(payload))
    assert ref._source_contract(suite, {}, {}) != before


def test_outer_rc_zero_without_required_output_is_failure(tmp_path):
    from test_v27519_management_cpu_production_chain import _write_minimal_reference_suite
    suite = tmp_path / 'suite'
    _write_minimal_reference_suite(suite, contract_marker='no-output', top1_hit=True)
    (suite / 'benchmark_suite.py').write_text("print('Full task completion unavailable: FrozenPostprocessError: missing attestation')\n")
    result = ref.generate_management_cpu_reference(suite_dir=suite, output_dir=tmp_path/'reference', model_id='renamed_detector', timeout_s=20)
    assert result['return_code'] == 0
    assert result['status'] == 'failed'
    assert result['error'] == 'quality_reference_not_emitted'
