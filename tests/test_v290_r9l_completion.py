"""Archived YOLOv7 reports: strict negatives, current producer, isolated errors."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from test_v283_r9k_raw_full_contract import FOLDER, SOURCE, TEMPLATE, runner, offline_report, read
from test_v290_r9l_scope import evidence
from onnx_splitpoint_tool.workflow.results import normalize_benchmark_files
from onnx_splitpoint_tool.workflow.result_context import load_benchmark_source_contexts
from onnx_splitpoint_tool.workflow.scientific_reporting import _scientific_row, _write_reports


def test_original_yolov7_conflict_preserves_other_variants_in_workflow_import(tmp_path):
    paths = sorted(FOLDER.glob('benchmark_results_*.json'))
    before = {p: p.read_bytes() for p in paths}
    with pytest.raises(ValueError, match='generic_completion_projection_conflict:host_tail_available'):
        normalize_benchmark_files(model_id='yolov7_paper', source_paths=paths)
    rows, sources = normalize_benchmark_files(model_id='yolov7_paper', source_paths=paths,
        source_contexts=load_benchmark_source_contexts(FOLDER, model_id='yolov7_paper'),
        record_completion_errors=True)
    assert len(rows) == 8
    failed, = [r for r in rows if r.get('normalization_error')]
    assert failed['variant'] == 'full' and failed['backend'] == 'tensorrt'
    assert failed['runtime_executable'] is False and failed['structural_contract_pass'] is False
    assert failed['measurement_valid'] is False
    assert sum(r['runtime_executable'] is True for r in rows) == 7
    assert sum(len(s.get('normalization_errors', [])) for s in sources) == 1
    scientific = [_scientific_row(r) for r in rows]
    _write_reports(tmp_path / 'scientific', {'run_id': 'offline_original_eval1', 'rows': scientific})
    observations = read(tmp_path / 'scientific/performance_observations.json')
    assert len(observations) == 7
    diagnostic = next(r for r in scientific if r.get('normalization_error'))
    assert diagnostic['normalization_error'] == failed['normalization_error']
    assert before == {p: p.read_bytes() for p in paths}
    evidence('yolov7_original.json', dict(sources=[str(p) for p in paths], first_error=failed['normalization_error'],
        raw_full_host_tail_available=read(SOURCE)[0]['deployment_contracts_by_variant']['full']['host_tail_available'],
        measured_frames=read(SOURCE)[0]['generic_full_completion_evidence']['expected_frames'],
        retained_rows=8, independent_observations=len(observations), invalid_variant='tensorrt/full',
        raw_files_unchanged=True, quality_synthesized=False))


def test_current_producer_metadata_closes_original_conflict_offline(runner, tmp_path):
    source, regenerated = offline_report(runner, tmp_path)
    contexts = load_benchmark_source_contexts(FOLDER, model_id='yolov7_paper')
    for context in contexts:
        if context['source_path'] == str(SOURCE): context['source_path'] = str(source)
    paths = [source if p == SOURCE else p for p in sorted(FOLDER.glob('benchmark_results_*.json'))]
    rows, _ = normalize_benchmark_files(model_id='yolov7_paper', source_paths=paths, source_contexts=contexts)
    assert len(rows) == 8 and all(r['structural_contract_pass'] for r in rows)
    assert all(r['task_quality_status'] == 'pending_central_evaluation' for r in rows)
    _write_reports(tmp_path / 'scientific', {'run_id': 'offline_current_producer', 'rows': [_scientific_row(r) for r in rows]})
    assert len(read(tmp_path / 'scientific/performance_observations.json')) == 8
    assert regenerated['generic_full_completion_evidence'] == read(SOURCE)[0]['generic_full_completion_evidence']
    evidence('yolov7_current.json', dict(source=str(SOURCE), template=str(TEMPLATE), output=str(source),
        regenerated_fields=['deployment_contracts_by_variant', 'deployment_contract_summary', 'deployment_contract.contracts_by_variant', 'deployment_contract.contract_summary'],
        structural_rows=8, observations=8, central_quality_status='pending_central_evaluation',
        measured_evidence_unchanged=True, hardware_inference=False))


@pytest.mark.parametrize('model,boundary', [('renamed_detector', 'b017'), ('unrelated_detector', 'b411')])
def test_renaming_only_the_row_cannot_forge_a_bound_decoder(runner, tmp_path, model, boundary):
    source, raw = offline_report(runner, tmp_path)
    raw.update(model_id=model, case_id=boundary)
    source.write_text(json.dumps([raw]))
    rows, _ = normalize_benchmark_files(model_id=model, source_paths=[source])
    full = next(r for r in rows if r['variant'] == 'full')
    assert full['model_id'] == model and full['source_case_id'] == boundary
    assert full['structural_contract_pass'] is False
    assert full['host_postprocessing_evidence_status'] == 'failed_invalid_generic_host_postprocess_evidence'
    assert full['postprocess_completed_frames'] == 5


@pytest.mark.parametrize('damage', ['host_tail', 'completion', 'frames', 'timer', 'producer', 'alias', 'hash'])
def test_workflow_records_real_conflicts_without_losing_good_split(runner, tmp_path, damage):
    source, raw = offline_report(runner, tmp_path)
    if damage == 'host_tail': raw['deployment_contracts_by_variant']['full']['host_tail_available'] = False
    elif damage == 'completion': raw['deployment_contracts_by_variant']['full']['postprocess_completion_verified'] = False
    elif damage == 'frames': raw['generic_full_completion_evidence']['expected_frames'] += 1
    elif damage == 'timer': raw['generic_full_completion_evidence']['frames'][0]['tail_end_ns'] = 1
    elif damage == 'producer': raw['generic_full_completion_evidence']['producer'] = 'native_full_deepx'
    elif damage == 'hash': raw['deployment_contract']['full_primary_host_tail_sha256'] = '0' * 64
    else:
        # An explicitly primary Full alias must describe the same timer.
        raw['primary_variant'] = 'full'
        raw['generic_completion_evidence'] = deepcopy(raw['generic_full_completion_evidence'])
        raw['generic_completion_evidence']['frames'][0]['timer_end_ns'] += 1
    source.write_text(json.dumps([raw]))
    before = source.read_bytes()
    rows, sources = normalize_benchmark_files(model_id='yolov7_paper', source_paths=[source], record_completion_errors=True)
    failed = [r for r in rows if r.get('normalization_error')]
    assert failed and all(r['runtime_executable'] is False for r in failed)
    if damage != 'alias':
        split, = [r for r in rows if r['variant'] == 'split']
        assert split['runtime_executable'] is True and not split.get('normalization_error')
    assert source.read_bytes() == before


@pytest.mark.parametrize('reverse', [False, True])
def test_dispatch_owner_survives_three_setup_order(tmp_path, monkeypatch, reverse):
    import test_v27521_setup_local_trt_dispatch as original
    targets = original._targets
    monkeypatch.setattr(original, '_targets', lambda **kw: list(reversed(targets(**kw))) if reverse else targets(**kw))
    original.test_execution_dispatches_three_setup_local_trt_producers_before_threads(tmp_path, monkeypatch)


@pytest.mark.parametrize('classes,boundary', [(3, 'b017'), (7, 'b411')])
def test_valid_renamed_detector_and_other_classes_reach_generic_consumer(tmp_path, classes, boundary):
    import numpy as np
    from test_v283_r9j_continuation import test_topk_contract_is_independent_of_candidate_count_classes_and_names, processor
    from onnx_splitpoint_tool.runners.task_completion import TimedTaskCompletion
    test_topk_contract_is_independent_of_candidate_count_classes_and_names(tmp_path, 37, 17, classes)
    array = np.tile(np.array([1, 150, 20, 170, .8, classes - 1], np.float32), (1, 17, 1))
    runtime, outputs, endpoint, _ = processor(tmp_path, array)
    timer = TimedTaskCompletion('generic_deepx_full', 'detection', lambda: runtime.completed_count, contract=runtime.contract)
    for _ in range(3): timer.run(lambda: outputs, lambda data: runtime.process(data, original_wh=[640, 480]))
    raw = dict(model_id='renamed_detector', case_id=boundary, primary_variant='full', backend='deepx_m1',
               task='detection', runtime_ok=True, full_mean_ms=1.,
               measurement_endpoint='completed_detection', generic_completion_evidence=timer.report(0, 3),
               endpoint_contract_complete=True, output_endpoint_attestation=endpoint['output_endpoint_attestation'])
    path = tmp_path / 'benchmark_results_deepx_m1_full_auto.json'; path.write_text(json.dumps([raw]))
    rows, _ = normalize_benchmark_files(model_id='renamed_detector', source_paths=[path])
    assert all(r['postprocess_completed_frames'] == 3 and r['structural_contract_pass'] for r in rows)
    assert runtime.contract['materialization_contract']['candidate_selection']['class_count'] == classes
    assert all(_scientific_row(r)['measurement_endpoint'] == 'completed_detection' for r in rows)
