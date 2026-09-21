"""EVAL1 raw Full completion: producer metadata and strict stored-result import.

Archived files are read only. Offline exports exercise the current generator;
they do not repair EVAL1 or supply its missing central quality results.
"""
import ast
from copy import deepcopy
import csv
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from onnx_splitpoint_tool.runners.task_completion import TimedTaskCompletion
from onnx_splitpoint_tool.workflow.results import normalize_benchmark_files
from onnx_splitpoint_tool.workflow.scientific_reporting import _scientific_row, _write_reports
from test_v275_preprocessing_contract import _runner_module


RUN = Path('/home/kmika/Models/EvaluationRuns/v283_R9K_Abschluss_20260920_151548_6412w66g/eval_01_lh_nyn4q/r9j_eval_20260920_165839')
FOLDER = RUN / 'models/yolov7_paper/benchmark_results'
SOURCE = FOLDER / 'benchmark_results_ort_tensorrt_auto.json'
TEMPLATE = Path(__file__).resolve().parents[1] / 'onnx_splitpoint_tool/resources/templates/run_split_onnxruntime.py.txt'


def read(path):
    return json.loads(path.read_text())


@pytest.fixture
def runner(monkeypatch):
    import onnx_splitpoint_tool.native_detection_postprocess as pp
    import onnx_splitpoint_tool.runners.task_completion as tc
    monkeypatch.setitem(sys.modules, 'splitpoint_runners.native_detection_postprocess', pp)
    monkeypatch.setitem(sys.modules, 'splitpoint_runners.task_completion', tc)
    return _runner_module()


def export_contracts(runner, evidence, *, provider='tensorrt', status='ok', scope='generic_full_completed_task', task=None):
    """Replay archived timer frames through the real report and export statements."""
    from types import SimpleNamespace
    timer = TimedTaskCompletion(evidence['producer'], evidence['task'], lambda: 0,
                                contract=evidence['postprocess_contract'])
    timer.frames = deepcopy(evidence['frames'])
    tree = ast.parse(TEMPLATE.read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    # Include the production assembly in its original order, including report()
    # and every variant's actual call into the deployment-contract generator.
    first = next(i for i, n in enumerate(main.body) if isinstance(n, ast.Assign)
                 and ast.unparse(n.targets[0]) == 'stage2_input_contract_kind')
    last = next(i for i, n in enumerate(main.body) if isinstance(n, ast.Assign)
                and ast.unparse(n.targets[0]) == 'deployment_contract_summary')
    ns = dict(runner.__dict__)
    ns.update(args=SimpleNamespace(warmup=0, runs=evidence['expected_frames']),
              requested_benchmark_task=task or evidence['task'], effective_benchmark_task=evidence['task'],
              full_tok=provider, stage2_tok=provider, stage2_accel_calibration_gate={},
              hailo_full_endpoint_mode='', hailo_full_raw_detection_head=False,
              hailo_part2_raw_detection_head=False, hailo_part2_host_tail_required=False,
              ort_p2_host_tail_sess=None, generic_full_frozen_postprocessor=None,
              variant_status={'full': status}, p2_host_tail_path=None,
              full_primary_timing_scope=scope, generic_other_full_timer=timer,
              generic_full_completion_timer=timer, primary_variant='composed')
    exec(compile(ast.Module(body=main.body[first:last + 1], type_ignores=[]), str(TEMPLATE), 'exec'), ns)
    return ns


def offline_report(runner, tmp_path):
    raw, = read(SOURCE)
    evidence = raw['generic_full_completion_evidence']
    ns = export_contracts(runner, evidence)
    # Regenerate only the producer's deployment declarations in a separate file;
    # all measured evidence, metrics, identities and negative fields are kept.
    contracts = ns['deployment_contracts_by_variant']
    summary = ns['deployment_contract_summary']
    raw['deployment_contracts_by_variant'] = contracts
    raw['deployment_contract_summary'] = summary
    raw['deployment_contract']['contracts_by_variant'] = deepcopy(contracts)
    raw['deployment_contract']['contract_summary'] = deepcopy(summary)
    source = tmp_path / SOURCE.name
    source.write_text(json.dumps([raw]))
    return source, raw


def test_original_eval_conflict_remains_negative():
    raw, = read(SOURCE)
    assert raw['generic_full_completion_evidence']['expected_frames'] == 5
    assert raw['deployment_contracts_by_variant']['full']['host_tail_available'] is False
    with pytest.raises(ValueError, match='generic_completion_projection_conflict:host_tail_available'):
        normalize_benchmark_files(model_id='yolov7_paper', source_paths=[SOURCE])
    assert not (FOLDER / 'normalized_results.json').exists()


def test_offline_export_reaches_sealed_matrix_without_inventing_central_quality(runner, tmp_path):
    from onnx_splitpoint_tool.workflow.result_context import load_benchmark_source_contexts
    from onnx_splitpoint_tool.workflow.required_run_scope import required_measurements_from_scope
    from onnx_splitpoint_tool.workflow.runner import (
        _bind_results_to_required_scope_v2796, required_profile_outcomes_v282,
    )
    source, _ = offline_report(runner, tmp_path)
    contexts = load_benchmark_source_contexts(FOLDER, model_id='yolov7_paper')
    for context in contexts:
        if context['source_path'] == str(SOURCE):
            context['source_path'] = str(source)
    sources = [source if p == SOURCE else p for p in sorted(FOLDER.glob('benchmark_results_*.json'))]
    rows, _ = normalize_benchmark_files(model_id='yolov7_paper', source_paths=sources,
                                       source_contexts=contexts)
    required = required_measurements_from_scope(read(FOLDER.parent / 'benchmark_set/required_run_scope.json'))
    bound, errors = _bind_results_to_required_scope_v2796(required, rows)
    assert errors == []
    outcomes = required_profile_outcomes_v282(required, bound, bound, {})
    assert len(bound) == outcomes['required_result_count'] == 8
    assert outcomes['missing_result_count'] == outcomes['duplicate_result_count'] == 0
    assert outcomes['measurement_values_synthesized'] is False
    assert all(row['structural_contract_pass'] is True for row in bound)
    assert all(row['task_quality_status'] == 'pending_central_evaluation' for row in bound)
    trt = [row for row in bound if row['backend'] == 'tensorrt']
    assert {row['variant'] for row in trt} == {'full', 'split'}
    assert {row['setup_id'] for row in trt} == {'orin_nx_deepx_m1_01'}


@pytest.mark.parametrize('provider', ['tensorrt', 'cuda_ort', 'cpu_ort'])
def test_export_uses_measured_raw_full_contract_without_leaking_to_split(runner, provider):
    raw, = read(SOURCE)
    ns = export_contracts(runner, raw['generic_full_completion_evidence'], provider=provider)
    full = ns['deployment_contracts_by_variant']['full']
    assert full['raw_head_contract_present'] is True
    assert full['host_tail_required'] is full['host_tail_available'] is True
    assert full['raw_head_contract_status'] == 'raw_head_plus_host_tail'
    for variant in ('part1', 'part2', 'composed'):
        assert ns['deployment_contracts_by_variant'][variant]['host_tail_available'] is False
        assert ns['deployment_contracts_by_variant'][variant]['raw_head_contract_present'] is False


@pytest.mark.parametrize('damage', ['frames', 'timer', 'producer', 'contract_hash', 'task'])
def test_export_rejects_invalid_completion(runner, damage):
    from onnx_splitpoint_tool.native_detection_postprocess import FrozenPostprocessError
    raw, = read(SOURCE)
    evidence = raw['generic_full_completion_evidence']
    if damage == 'frames': evidence['expected_frames'] += 1
    elif damage == 'timer': evidence['frames'][0]['tail_end_ns'] = evidence['frames'][0]['timer_end_ns'] + 1
    elif damage == 'producer': evidence['producer'] = 'generic_hailo_full'
    elif damage == 'task': evidence['task'] = 'classification'
    else: evidence['postprocess_contract']['confidence_threshold'] = .5
    with pytest.raises((ValueError, FrozenPostprocessError)):
        export_contracts(runner, evidence)


def test_auto_task_uses_verified_detection_evidence(runner):
    raw, = read(SOURCE)
    ns = export_contracts(runner, raw['generic_full_completion_evidence'], task='auto')
    assert ns['deployment_contracts_by_variant']['full']['host_tail_available'] is True


@pytest.mark.parametrize('provider', ['hailo8', 'hailo10'])
def test_existing_hailo_full_exports_keep_measured_tail(runner, provider):
    raw, = read(FOLDER / f'benchmark_results_{provider}_auto.json')
    ns = export_contracts(runner, raw['generic_completion_evidence'], provider=provider,
                          scope='hailo_raw_head_plus_frozen_decode_nms')
    assert ns['deployment_contracts_by_variant']['full']['host_tail_available'] is True


@pytest.mark.parametrize('status,scope', [('error', 'generic_full_completed_task'), ('ok', 'backend_endpoint')])
def test_unmeasured_full_never_acquires_host_tail(runner, status, scope):
    raw, = read(SOURCE)
    ns = export_contracts(runner, raw['generic_full_completion_evidence'], status=status, scope=scope)
    assert ns['generic_full_completion_evidence'] == {}
    assert ns['deployment_contracts_by_variant']['full']['host_tail_available'] is False


def test_real_cpu_decoder_timer_survives_generator_loader_and_scientific_csv(runner, tmp_path):
    from onnx_splitpoint_tool.native_detection_postprocess import FrozenDetectionPostprocessor
    raw, = read(SOURCE)
    contract = raw['generic_full_completion_evidence']['postprocess_contract']
    processor = FrozenDetectionPostprocessor(contract)
    outputs = {t['name']: np.full(t['shape'], -20, dtype=np.float32)
               for t in contract['raw_output_tensor_signature']['tensors']}
    timer = TimedTaskCompletion('generic_ort_full', 'detection', lambda: processor.completed_count,
                                contract=contract)
    for _ in range(3):
        timer.run(lambda: outputs, processor.process)
    evidence = timer.report(0, 3)
    assert processor.completed_count == 3
    ns = export_contracts(runner, evidence)
    source, raw = offline_report(runner, tmp_path)
    raw['generic_full_completion_evidence'] = ns['generic_full_completion_evidence']
    source.write_text(json.dumps([raw]))
    rows, _ = normalize_benchmark_files(model_id='yolov7_paper', source_paths=[source])
    full = next(r for r in rows if r['variant'] == 'full')
    split = next(r for r in rows if r['variant'] == 'split')
    assert full['host_postprocessing_evidence_status'] == 'passed'
    assert full['structural_contract_pass'] is True
    assert full['measurement_endpoint'] == 'completed_detection'
    assert full['postprocess_completed_frames'] == 3
    assert split['measurement_endpoint'] == 'p2_output'
    assert not split.get('generic_completion_evidence')
    report = tmp_path / 'scientific'
    _write_reports(report, {'run_id': 'offline_cpu_callback', 'rows': [_scientific_row(full)]})
    observed, = read(report / 'performance_observations.json')
    with (report / 'performance_observations.csv').open(newline='') as f:
        exported, = csv.DictReader(f)
    assert observed['measurement_endpoint'] == exported['measurement_endpoint'] == 'completed_detection'
    assert observed['postprocess_completed_frames'] == 3


@pytest.mark.parametrize('damage', ['negative_tail', 'negative_completion', 'frames', 'producer', 'alias', 'hash', 'timeout', 'exit'])
def test_generated_metadata_does_not_hide_incoming_conflicts(runner, tmp_path, damage):
    source, raw = offline_report(runner, tmp_path)
    if damage == 'negative_tail': raw['deployment_contracts_by_variant']['full']['host_tail_available'] = False
    elif damage == 'negative_completion': raw['deployment_contracts_by_variant']['full']['postprocess_completion_verified'] = False
    elif damage == 'frames': raw['generic_full_completion_evidence']['expected_frames'] += 1
    elif damage == 'producer': raw['generic_full_completion_evidence']['producer'] = 'generic_hailo_full'
    elif damage == 'hash': raw['deployment_contract']['full_primary_host_tail_sha256'] = '0' * 64
    elif damage == 'alias':
        raw['primary_variant'] = 'full'
        raw['generic_completion_evidence'] = deepcopy(raw['generic_full_completion_evidence'])
        raw['generic_completion_evidence']['frames'][0]['timer_end_ns'] += 1
    elif damage == 'timeout': raw.update(runtime_ok=False, error_class='timeout')
    else: raw['runner_returncode'] = -11
    source.write_text(json.dumps([raw]))
    if damage in {'timeout', 'exit'}:
        rows, _ = normalize_benchmark_files(model_id='yolov7_paper', source_paths=[source])
        full = next(r for r in rows if r['variant'] == 'full')
        assert full['runtime_executable'] is False
        report = tmp_path / 'scientific'
        _write_reports(report, {'run_id': 'offline_negative', 'rows': [_scientific_row(full)]})
        assert read(report / 'performance_observations.json') == []
    else:
        with pytest.raises(ValueError):
            normalize_benchmark_files(model_id='yolov7_paper', source_paths=[source])
