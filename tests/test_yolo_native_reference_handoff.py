"""The actual CLI handoff binds stored references before semantic comparison.

The central source, Native command and consumer attestation use normal readers.
Only the visualization leaf is observed here; stored-output comparison and its
model/input/endpoint negatives are covered by test_yolo_augmentation_native_reference.
"""
from __future__ import annotations

import ast
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import native_producer_validate_visualize as validator
from onnx_splitpoint_tool.native_split_quality import (
    seal_native_split_quality_binding, select_central_native_split_quality_binding,
)
from onnx_splitpoint_tool.native_command_contract import canonical_json_sha256
from tests.test_v269f_native_split_receipt_validation import (
    _fixture_payload, _native_row_for_binding,
)


@pytest.fixture
def handoff(tmp_path):
    payload, _ = _fixture_payload(tmp_path)
    binding = seal_native_split_quality_binding(payload)
    policy = validator.AccuracyGatePolicy()
    provenance = {
        'source_request_sha256': '3' * 64, 'model_sha256': '4' * 64,
        'validation_dataset_sha256': '5' * 64,
        'validation_dataset_image_ids_sha256': '6' * 64,
        'validation_dataset_ground_truth_sha256': '7' * 64,
        'policy_sha256': policy.sha256(), 'quality_contract_sha256': 'a' * 64,
        'preprocessing_contract_sha256': 'b' * 64,
        'decoder_contract_sha256': 'c' * 64, 'nms_contract_sha256': 'd' * 64,
    }
    identity = {
        'identity_valid': True, 'eval_run_id': binding['eval_run_id'],
        'model_id': 'yolo26s', 'task': 'detection', 'case_id': 'b038',
        'source_run_id': 'hailo8_to_trt', 'setup_id': 'hailo8_setup',
        'variant': 'composed', 'endpoint_contract_hash': '9' * 64,
        'runtime_precision_identity': 'uint8_dequant_fp16',
        'native_split_quality_binding_required': True,
        'native_split_quality_binding': copy.deepcopy(binding),
        'native_split_quality_binding_sha256': binding['binding_sha256'],
        **provenance,
    }
    result = {
        **identity, 'status': 'completed', 'technical_status': 'completed',
        'decision': 'reference_close', 'source_setup_id': 'hailo8_setup',
        'request_identity': copy.deepcopy(identity),
    }
    selected = select_central_native_split_quality_binding(
        binding, source_request_sha256=provenance['source_request_sha256'],
        central_result_sha256=canonical_json_sha256(result),
        central_identity={key: identity[key] for key in (
            'eval_run_id', 'model_id', 'case_id', 'source_run_id', 'setup_id',
            'task', 'variant', 'runtime_precision_identity')},
    )
    row = _native_row_for_binding(selected)
    row.update(endpoint_contract_complete=True, endpoint_contract_hash='9' * 64,
               runtime_precision_identity='uint8_dequant_fp16', model_sha256='')
    return row, result, policy


def execute_cli_detection_handoff(tmp_path, row, result, policy, *, offline=True):
    """Execute the unchanged detection branch from the real main() function."""
    tree = ast.parse(Path(validator.__file__).read_text())
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name == 'main')
    branch = next(node for node in ast.walk(main)
                  if isinstance(node, ast.If) and ast.unparse(node.test) == "task == 'classification'")
    seen = []

    def observe(*args, endpoint_evidence, **kwargs):
        seen.append(copy.deepcopy(endpoint_evidence))
        # No fabricated positive semantic outcome: this unit test observes
        # only the source binding at the real comparator call boundary.
        return {'ok': False, 'semantic_ok': 'unavailable', 'semantic_available': False}

    namespace = dict(vars(validator), rec=row, backend=row['backend'],
        ns=SimpleNamespace(existing_outputs_only=offline, max_detection_outputs=0),
        central_results=[result], gate_policy=policy, split_quality_authority=None,
        central_result_sha256=None, roots=[tmp_path], dump=tmp_path/'output.json',
        row_dir=tmp_path, ref_report=None, eval_root=tmp_path, report=None,
        existing_reference_rows=[], _visualize_detection=observe)
    exec(compile(ast.Module(body=branch.orelse, type_ignores=[]), validator.__file__, 'exec'), namespace)
    assert len(seen) == 1
    return seen[0]


def test_first_ungated_native_report_gets_verified_model_before_comparator(tmp_path, handoff):
    row, result, policy = handoff
    before = copy.deepcopy(row['native_command_contract'])
    assert row['model_sha256'] == ''
    observed = execute_cli_detection_handoff(tmp_path, row, result, policy)
    assert observed['model_sha256'] == result['model_sha256']
    assert observed['central_quality_evidence_verified'] is True
    assert observed['quality_first_binding_status'] == 'central_native_exact_engine_command_boundary_match'
    assert row['native_command_contract'] == before
    assert row['semantic_ok'] == 'unavailable'


@pytest.mark.parametrize('fault', ['request', 'model', 'setup', 'precision', 'endpoint', 'consumer', 'raw_model'])
def test_reference_handoff_never_repairs_conflicting_evidence(tmp_path, handoff, fault):
    row, result, policy = handoff
    if fault == 'request': result['source_request_sha256'] = 'e' * 64
    elif fault == 'model': result['model_sha256'] = 'e' * 64
    elif fault == 'setup': result['source_setup_id'] = 'other'
    elif fault == 'precision': result['runtime_precision_identity'] = 'fp32'
    elif fault == 'endpoint': result['endpoint_contract_hash'] = 'e' * 64
    elif fault == 'consumer': row['native_split_quality_consumer_attestation']['binding_sha256'] = 'e' * 64
    elif fault == 'raw_model': row['model_sha256'] = 'e' * 64
    observed = execute_cli_detection_handoff(tmp_path, row, result, policy)
    assert observed['central_quality_evidence_verified'] is False
    assert observed['model_sha256'] == ('e' * 64 if fault == 'raw_model' else '')


def test_regular_inference_mode_keeps_existing_handoff(tmp_path, handoff):
    row, result, policy = handoff
    observed = execute_cli_detection_handoff(tmp_path, row, result, policy, offline=False)
    assert observed['model_sha256'] == ''


def test_packaged_validator_is_exact_script_mirror():
    root = Path(__file__).resolve().parents[1]
    assert (root/'scripts/native_producer_validate_visualize.py').read_bytes() == (
        root/'onnx_splitpoint_tool/resources/remote_scripts/native_producer_validate_visualize.py').read_bytes()
