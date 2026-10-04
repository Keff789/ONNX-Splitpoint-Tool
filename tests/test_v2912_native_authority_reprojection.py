"""Recompute an obsolete report decision using unchanged measured evidence."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

from scripts import native_producer_final_report as report


def _fixture(tmp_path):
    path = Path(__file__).with_name('test_v269f_native_split_final_energy_integrity.py')
    spec = importlib.util.spec_from_file_location('authority_reprojection_fixture', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    row, manifest, payload = module._quality_first_row(tmp_path)
    raw_path = manifest.parent / 'native_result.json'
    raw_path.write_text(json.dumps(row))
    row['report'] = str(raw_path)
    row['native_split_semantic_binding_required'] = True
    row['native_split_semantic_binding_valid'] = False
    row['native_split_semantic_binding_status'] = 'authority_eval_run_id_mismatch'
    row['native_split_final_portable_binding_valid'] = False
    row['native_split_final_portable_binding_status'] = 'native_split_quality_authority_eval_run_id_mismatch'
    row['native_split_quality_authority_previous_errors'] = ['native_stage_workflow_version_mismatch']
    row['performance_repetitions'] = [{'fps': 10}, {'fps': 11}, {'fps': 12}]
    authority = copy.deepcopy(row['native_split_quality_authority'])
    return row, authority, raw_path, payload


def test_obsolete_authority_decision_replays_existing_evidence_without_mutation(tmp_path):
    row, authority, raw_path, payload = _fixture(tmp_path)
    before = copy.deepcopy(row)
    raw_bytes, payload_bytes = raw_path.read_bytes(), payload.read_bytes()
    report._apply_native_split_authority_to_rows([row], authority)
    assert row['native_split_final_portable_binding_valid'] is True
    assert row['native_split_final_portable_binding_status'] == 'portable_binding_command_and_consumer_attestation_exact_match'
    assert row['native_split_semantic_binding_valid'] is True
    assert row['performance_repetitions'] == before['performance_repetitions']
    history = row['native_split_quality_previous_projection']
    assert history['native_split_final_portable_binding_valid'] is False
    assert history['native_split_quality_authority_previous_errors'] == ['native_stage_workflow_version_mismatch']
    assert raw_path.read_bytes() == raw_bytes and payload.read_bytes() == payload_bytes
    stable = copy.deepcopy(row)
    report._apply_native_split_authority_to_rows([row], authority)
    assert row == stable


@pytest.mark.parametrize('mutation', ['wrong_run', 'wrong_setup', 'wrong_request', 'wrong_consumer', 'payload', 'missing_result', 'duplicate_result', 'authority_invalid'])
def test_current_authority_does_not_release_invalid_original_evidence(tmp_path, mutation):
    row, authority, raw_path, payload = _fixture(tmp_path)
    if mutation == 'wrong_run':
        row['eval_run_id'] = 'different_run'
    elif mutation == 'wrong_setup':
        row['setup_id'] = 'other_setup'
    elif mutation == 'wrong_request':
        row['source_request_sha256'] = 'f' * 64
    elif mutation == 'wrong_consumer':
        row['native_split_quality_consumer_attestation']['eval_run_id'] = 'different_run'
    elif mutation == 'payload':
        payload.write_bytes(b'tampered')
    elif mutation == 'missing_result':
        raw_path.unlink()
    elif mutation == 'duplicate_result':
        raw_path.write_text('{"ok":true,"ok":false}')
    elif mutation == 'authority_invalid':
        authority.update(valid=False, errors=['native_stage_workflow_version_mismatch'])
    report._apply_native_split_authority_to_rows([row], authority)
    assert row['native_split_final_portable_binding_valid'] is False
    assert row['native_split_semantic_binding_valid'] is False
    if mutation == 'authority_invalid':
        assert row['native_split_final_portable_binding_status'] == 'native_split_quality_authority_invalid'
        assert row['native_split_quality_authority_errors'] == ['native_stage_workflow_version_mismatch']


def test_unrelated_negative_projection_is_preserved(tmp_path):
    row, authority, _, _ = _fixture(tmp_path)
    row['native_split_final_portable_binding_status'] = 'semantic_payload_invalid'
    report._apply_native_split_authority_to_rows([row], authority)
    assert row['native_split_final_portable_binding_valid'] is False
    assert row['native_split_final_portable_binding_status'] == 'semantic_payload_invalid'
