"""Actual two-starter admission, only in the explicitly protected external COW."""
import copy
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from tests.test_yolo_measured_completion_reentry import measured, edit
from onnx_splitpoint_tool.workflow import yolo_augmentation_reentry as admission
from onnx_splitpoint_tool.workflow.supplement import (
    build_supplement_plan, prepare_source_evidence, preserve_metadata,
)


def test_real_two_pairs_preserve_each_original_leaf_and_twelve_quality(measured):
    run, plan, proof = measured
    leaves = proof['preserved_native_leaves']
    assert [row['case_key'] for row in leaves] == [
        'yolo11l|b120|orin_nx_hailo8_01', 'yolo26s|b038|orin_nx_hailo10_01']
    assert leaves[0]['coordinator_input_hash'] == proof['two_completed_pairs_ancestor']['preserved_native_coordinator_input_hash']
    assert leaves[1]['coordinator_input_hash'] == proof['predecessor_input_hash']
    assert leaves[0]['coordinator_input_hash'] != leaves[1]['coordinator_input_hash']
    assert [len(row['repetitions']) for row in proof['preserved_generic_results']] == [3, 3]
    original, index = prepare_source_evidence(run)
    projected = build_supplement_plan(run, original_root=original,
        source_index={row['path']: row for row in index['artifacts']})
    assert projected['counts']['generic'] == {'preserve': 12}
    assert projected['counts']['native'] == {'preserve_and_validate_locally': 2, 'execute': 10}
    assert projected['completed_quality_count'] == 12 and not projected['pending_quality_rows']


def test_normal_snapshot_then_new_canonical_stage_keeps_both_hashes(measured, edit):
    run, plan, before = measured
    original = preserve_metadata(run, 'pytest-two-completed-pairs')
    try:
        proof = admission.validate_augmentation_reentry(run, plan, original)
        cfg = json.loads((original/'reports/native_producer_stage_config.json').read_text())
        cfg['augmentation_startcase_barrier']['reentry'] = proof
        edit(run/'reports/native_producer_stage.json', {'state': 'running', 'session_id': 'new'})
        assert admission.native_reentry_root(run, cfg) == original.parent/'native'
        for row in proof['preserved_native_leaves']:
            path = run/row['path']; raw = path.read_bytes(); leaf = json.loads(raw)
            assert admission.completed_native_leaf_input(run, cfg, path,
                leaf['details']['variant'], leaf['details']['execution']['cmd'], 'new-input') == leaf['input_hash']
            assert path.read_bytes() == raw
            changed = copy.deepcopy(leaf['details']['variant'])
            changed['quality_snapshot_sha256'] = 'f'*64
            with pytest.raises(ValueError, match='current_leaf_contract_changed'):
                admission.completed_native_leaf_input(run, cfg, path, changed,
                    leaf['details']['execution']['cmd'], 'new-input')
        assert (original/Path(before['predecessor_checkpoint']).relative_to(run)).read_bytes() == Path(before['predecessor_checkpoint']).read_bytes()
    finally:
        shutil.rmtree(original.parent)


@pytest.mark.parametrize('model', ['yolo11l', 'yolo26s'])
@pytest.mark.parametrize('fault', ['native_failed', 'native_count', 'native_runtime_ids', 'native_time',
                                  'generic_partial', 'generic_sample', 'generic_identity'])
def test_either_observation_drift_blocks_reentry(measured, edit, model, fault):
    run, plan, _ = measured
    if fault.startswith('native'):
        summary = json.loads((run/'reports/native_producer_summary.json').read_text())
        row = next(row for row in summary['rows'] if row['model'] == model)
        path = Path(row['report']); raw = json.loads(path.read_text())
        if fault == 'native_failed': raw['ok'] = False
        elif fault == 'native_count': raw['repetition_count_valid'] = 2
        else:
            completed = (raw.get('endpoint_results') or {}).get('completed_task') or raw
            records = completed['repetition_records']
            if fault == 'native_time': records[0]['makespan_ms'] += 100
            else:
                for record in records: record['runtime_instance_id'] = records[0]['runtime_instance_id']
        edit(path, raw)
        index_path = run/'artifact_index.json'; index = json.loads(index_path.read_text())
        for item in index['artifacts']:
            if item['path'] == path.relative_to(run).as_posix():
                item.update(size_bytes=path.stat().st_size,
                    sha256='sha256:'+hashlib.sha256(path.read_bytes()).hexdigest())
        edit(index_path, index)
    else:
        ep = plan.parent.parent/'completion/execution-plan.json'; execution = json.loads(ep.read_text())
        case = next(row for row in execution['rows'] if row['identity']['model_id'] == model)
        path = Path(case['result_path']); result = json.loads(path.read_text())
        if fault == 'generic_partial':
            result['repetitions'] = result['repetitions'][:2]; edit(path, result)
        elif fault == 'generic_sample':
            sample = next(iter(result['repetitions'][0]['raw_output_samples'].values()))
            edit(path.parent/sample['path'], b'changed preserved output')
        else:
            case['identity']['setup_id'] = 'foreign'; edit(ep, execution)
    with pytest.raises(ValueError): admission.validate_augmentation_reentry(run, plan)


@pytest.mark.parametrize('fault', ['third_leaf', 'third_generic', 'wrong_leaf_hash', 'repeated_repair', 'current_cancel'])
def test_exact_two_starter_scope_and_terminal_failure_required(measured, edit, monkeypatch, fault):
    run, plan, proof = measured
    if fault == 'third_leaf':
        edit(run/'reports/native_producer_variants/foreign/performance_stage_result.json', {})
    elif fault == 'third_generic':
        ep = plan.parent.parent/'completion/execution-plan.json'; execution = json.loads(ep.read_text())
        execution['rows'].append(copy.deepcopy(execution['rows'][0])); edit(ep, execution)
    elif fault == 'wrong_leaf_hash':
        path = run/proof['preserved_native_leaves'][1]['path']; leaf = json.loads(path.read_text())
        leaf['input_hash'] = proof['preserved_native_leaves'][0]['input_hash']; edit(path, leaf)
    else:
        original = admission._object
        def changed(path):
            value = original(path)
            if fault == 'repeated_repair' and Path(path) == run/'reports/native_producer_stage_config.json':
                value['augmentation_startcase_barrier']['reentry']['two_completed_pairs_ancestor'] = {'forbidden': 'repeat'}
            if fault == 'current_cancel' and Path(path) == run/'reports/native_producer_stage.json':
                value['error'] = admission.CANCEL_NONSTART
            return value
        monkeypatch.setattr(admission, '_object', changed)
    with pytest.raises(ValueError): admission.validate_augmentation_reentry(run, plan)
