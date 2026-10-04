"""Normal generic producer writer -> Normalizer -> strict pair identity."""
import copy
import json

import pytest

from onnx_splitpoint_tool.quality_cache import json_fingerprint
from onnx_splitpoint_tool.workflow.results import normalize_benchmark_row
from onnx_splitpoint_tool.workflow.cross_runner_reporting import _merge_central_request_projection
from tests.test_v27911_composed_provenance_binding import _binding, _portable_request


def request(tmp_path, backend='hailo8_to_trt'):
    producer, completion, contract = _binding(tmp_path/'artifacts', setup_id='setup', source_run_id=backend)
    path = _portable_request(tmp_path/'request', producer, completion, contract)
    return json.loads(path.read_text())


def normalized(tmp_path, payload, *, variant='composed'):
    return normalize_benchmark_row({'case_id':'b001', 'variant':variant, 'primary_variant':variant,
        'run_id':payload['backend'], 'backend':payload['backend'], 'setup_id':'setup',
        'runtime_ok':True, 'total_latency_ms':10.,
        'task_quality_input_requests_by_variant':{variant:payload}},
        model_id='resnet50', source_path=tmp_path/'raw.json')


@pytest.mark.parametrize('backend,expected', [('hailo8_to_trt','hailo8'),
    ('hailo10h_to_trt','hailo10h'), ('deepx_to_trt','deepx_m1'),
    ('tensorrt_to_tensorrt','tensorrt')])
def test_normal_writer_direction_projects_origin_and_joins_central_identity(tmp_path, backend, expected):
    row = normalized(tmp_path, request(tmp_path, backend))
    identity = row['quality_request_identities_by_variant']['composed']
    assert identity['identity_valid'] is True, identity['identity_errors']
    assert identity['producer_backend'] == expected
    _, errors = _merge_central_request_projection(row, {'producer_backend':expected})
    assert errors == []


@pytest.mark.parametrize('field,value', [('stage1_backend','deepx'),
    ('stage2_backend','cpu'), ('backend','deepx_to_trt'),
    ('producer_backend','tensorrt'), ('producer_identity_sha256','0'*64)])
def test_wrong_producer_cannot_replace_direction_binding(tmp_path, field, value):
    payload = request(tmp_path)
    producer = payload['producer_identity']
    producer[field] = value
    if field != 'producer_identity_sha256':
        # Independently resealed negative proof tests direction consistency,
        # not merely rejection of a stale checksum.
        completion = producer['candidate_execution_completion_contract']
        if field in completion:
            completion[field] = value
            completion.pop('contract_sha256')
            completion['contract_sha256'] = json_fingerprint(completion)
            producer['candidate_execution_completion_contract_sha256'] = completion['contract_sha256']
        producer.pop('producer_identity_sha256')
        producer['producer_identity_sha256'] = json_fingerprint(producer)
    row = normalized(tmp_path, payload)
    identity = row['quality_request_identities_by_variant']['composed']
    assert identity['identity_valid'] is False
    assert identity['identity_errors']
    _, errors = _merge_central_request_projection(row, {'producer_backend':'hailo8'})
    assert errors


def test_foreign_independently_valid_producer_direction_stays_rejected(tmp_path):
    payload = request(tmp_path/'original')
    other = request(tmp_path/'foreign', 'deepx_to_trt')
    payload['producer_identity'] = other['producer_identity']
    row = normalized(tmp_path, payload)
    assert row['quality_request_identities_by_variant']['composed']['identity_valid'] is False


def test_explicit_request_producer_conflict_is_not_silently_replaced(tmp_path):
    payload = request(tmp_path)
    payload['producer_backend'] = 'tensorrt'
    row = normalized(tmp_path, payload)
    assert row['quality_request_identities_by_variant']['composed']['identity_valid'] is False


def test_single_tensorrt_full_producer_keeps_existing_projection(tmp_path):
    payload = request(tmp_path)
    payload.update(variant='full', backend='tensorrt', source_run_id='tensorrt')
    payload['producer_identity'] = {'schema':'onnx-splitpoint/tensorrt-quality-producer-identity',
        'backend':'tensorrt', 'source_run_id':'tensorrt'}
    row = normalized(tmp_path, payload, variant='full')
    assert row['quality_request_identities_by_variant']['full']['producer_backend'] == 'tensorrt'
