"""B3 manifest replay at the generated Full timing and real consumer boundaries."""
import ast
from copy import deepcopy
import csv
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from onnx_splitpoint_tool.runners.task_completion import TimedTaskCompletion, validate_completion
from onnx_splitpoint_tool.workflow.results import normalize_benchmark_files
from onnx_splitpoint_tool.workflow.scientific_reporting import _scientific_row, _write_reports
from test_v275_preprocessing_contract import _runner_module


ROOT = Path(__file__).resolve().parents[1]
B3 = Path('/home/kmika/Models/EvaluationRuns/v283_R9K_Abschluss_20260920_151548_6412w66g/b_03_vx2wkiu3/r9j_b_20260920_153607')
TEMPLATE = ROOT / 'onnx_splitpoint_tool/resources/templates/run_split_onnxruntime.py.txt'


def read(path):
    return json.loads(path.read_text())


def generated_full(tmp_path, monkeypatch, *, backend='tensorrt', task='detection', damage=None):
    import onnx_splitpoint_tool.native_detection_postprocess as pp
    import onnx_splitpoint_tool.native_output_endpoint as ep
    import onnx_splitpoint_tool.runners.harness.classification as classification
    for name, module in [('native_detection_postprocess', pp), ('native_output_endpoint', ep),
                         ('harness.classification', classification)]:
        monkeypatch.setitem(sys.modules, 'splitpoint_runners.' + name, module)
    suite = B3 / 'models/yolo26s/benchmark_set/legacy_suite'
    manifest = read(suite / 'b364/split_manifest.json')
    assert manifest['hailo']['full_endpoint_mode'] == 'raw_detection_head'
    outputs = {'output0': np.zeros((1, 300, 6), dtype=np.float32)}
    outputs['output0'][0, :2] = [[100, 100, 200, 200, .9, 0], [101, 101, 201, 201, .8, 0]]
    if task == 'classification':
        outputs = {'logits': np.array([[1, 4, 2]], dtype=np.float32)}
    if damage == 'nonfinite':
        outputs['output0'][0, 2, 0] = np.nan
    elif damage == 'layout':
        outputs['output0'] = np.zeros((1, 300, 7), dtype=np.float32)
    image = tmp_path / 'input.png'
    Image.new('RGB', (640, 480)).save(image)
    calls = []

    class SDK:
        def run(self, *_):
            calls.append('infer')
            return list(outputs.values())

    sdk = SDK()

    def infer(**_):
        return dict(zip(outputs, sdk.run())), {}

    tree = ast.parse(TEMPLATE.read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    initialize = next(n for n in main.body if isinstance(n, ast.If)
                      and 'not hailo_full_raw_detection_head' in ast.unparse(n.test))
    callback = next(n for n in main.body if isinstance(n, ast.FunctionDef)
                    and n.name == '_run_other_full_completed')
    timing = next(n for n in main.body if isinstance(n, ast.If)
                  and ast.unparse(n.test) == "'full' in variants and (not variant_errors.get('full'))")
    export = next(n for n in main.body if isinstance(n, ast.If)
                  and 'full_primary_timing_scope == \'generic_full_completed_task\'' in ast.unparse(n.test))
    ns = dict(_runner_module().__dict__)
    ns.update(variants=['full'], hailo_full_raw_detection_head=True, hailo_full=None,
              full_inputs_hailo=None, variant_errors={}, variant_status={},
              args=SimpleNamespace(benchmark_task=task, model_id='yolo26s', quality_evidence_model_id='',
                                   warmup=1, runs=3, verbose_runs=False, progress_every=0),
              manifest=manifest, plan={}, img_path=image, img_hw=(640, 640),
              base_dir=suite / 'b364', full_path=suite / 'models/yolo26s.onnx', full_tok=backend,
              generic_other_full_processor=None, generic_other_full_timer=None,
              _generic_task=task, _decoder_model_sha256='', TimedTaskCompletion=TimedTaskCompletion,
              _run_full_variant_outputs_map=infer, native_full_sess=sdk, feeds_full={},
              full_primary_timing_scope='backend_endpoint', generic_full_completion_evidence={})
    exec(compile(ast.Module(body=[initialize, callback, timing, export], type_ignores=[]), str(TEMPLATE), 'exec'), ns)
    ns['sdk_calls'] = calls
    return ns


@pytest.mark.parametrize('backend', ['tensorrt', 'cuda_ort', 'cpu_ort'])
@pytest.mark.parametrize('task', ['detection', 'classification'])
def test_non_hailo_full_completes_despite_shared_raw_head_manifest(tmp_path, monkeypatch, backend, task):
    ns = generated_full(tmp_path, monkeypatch, backend=backend, task=task)
    assert not ns['variant_errors'], ns['variant_errors']
    assert ns['full_primary_timing_scope'] == 'generic_full_completed_task'
    evidence = ns['generic_full_completion_evidence']
    assert validate_completion(evidence, task=task, producer='generic_ort_full')
    assert len(evidence['frames']) == len(ns['full_runs_ms']) == 3
    assert len(ns['sdk_calls']) == 5  # one untimed probe, warmup, three measured calls
    assert ns['generic_other_full_processor'].completed_count == 4
    if task == 'detection':
        assert len(ns['generic_other_full_processor'].last_detections) == 2


@pytest.mark.parametrize('damage', ['nonfinite', 'layout'])
def test_invalid_non_hailo_full_cannot_fall_back_to_raw_timing(tmp_path, monkeypatch, damage):
    ns = generated_full(tmp_path, monkeypatch, damage=damage)
    assert ns['variant_errors'].get('full')
    assert ns['variant_status'].get('full') != 'ok'
    assert ns['generic_full_completion_evidence'] == {}
    assert len(ns['sdk_calls']) == 1


def test_active_hailo_raw_head_keeps_its_existing_completion_path():
    tree = ast.parse(TEMPLATE.read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    initialize = next(n for n in main.body if isinstance(n, ast.If)
                      and 'not hailo_full_raw_detection_head' in ast.unparse(n.test))
    ns = dict(variants=['full'], hailo_full=object(), hailo_full_raw_detection_head=True,
              variant_errors={}, generic_other_full_processor=None, generic_other_full_timer=None)
    exec(compile(ast.Module(body=[initialize], type_ignores=[]), str(TEMPLATE), 'exec'), ns)
    assert ns['generic_other_full_processor'] is ns['generic_other_full_timer'] is None
    assert ns['variant_errors'] == {}


def test_b3_original_completion_remains_missing_on_file_import():
    path = B3 / 'models/yolo26s/benchmark_results/benchmark_results_ort_tensorrt_auto.json'
    raw, = read(path)
    assert raw['generic_full_completion_evidence'] == {}
    assert raw['deployment_contract']['full_primary_timing_scope'] == 'backend_endpoint'
    rows, _ = normalize_benchmark_files(model_id='yolo26s', source_paths=[path])
    full = next(row for row in rows if row['variant'] == 'full')
    assert full['measurement_endpoint'] == 'raw_model_outputs'
    with pytest.raises(ValueError):
        validate_completion(full.get('generic_completion_evidence'), task='detection', producer='generic_ort_full')


def test_measured_trt_full_reaches_scientific_csv_with_historical_split_preserved(tmp_path, monkeypatch):
    ns = generated_full(tmp_path, monkeypatch)
    assert ns['full_primary_timing_scope'] == 'generic_full_completed_task'
    raw, = read(B3 / 'models/yolo26s/benchmark_results/benchmark_results_ort_tensorrt_auto.json')
    # This separate offline report uses the executed CPU test callback above;
    # it is not a correction of B3 or a new TensorRT measurement/quality claim.
    raw = deepcopy(raw)
    raw['generic_full_completion_evidence'] = ns['generic_full_completion_evidence']
    raw['full_mean_ms'] = ns['full_mean']
    raw['full_measurement_endpoint'] = 'completed_detection'
    raw['measurement_endpoints_by_variant']['full'] = 'completed_detection'
    raw['deployment_contract']['full_primary_timing_scope'] = ns['full_primary_timing_scope']
    path = tmp_path / 'benchmark_results_ort_tensorrt_auto.json'
    path.write_text(json.dumps([raw]))
    rows, _ = normalize_benchmark_files(model_id='yolo26s', source_paths=[path])
    full = next(row for row in rows if row['variant'] == 'full')
    split = next(row for row in rows if row['variant'] == 'split')
    assert full['endpoint_contract_complete'] is True
    assert full['structural_contract_pass'] is True
    assert split['measurement_endpoint'] == 'p2_output'
    assert not split.get('generic_completion_evidence')
    report = tmp_path / 'scientific'
    _write_reports(report, {'run_id': 'offline_b3_callback', 'rows': [_scientific_row(full)]})
    observed, = read(report / 'performance_observations.json')
    with (report / 'performance_observations.csv').open(newline='') as handle:
        exported, = csv.DictReader(handle)
    assert observed['measurement_endpoint'] == exported['measurement_endpoint'] == 'completed_detection'
    assert observed['postprocess_completed_frames'] == 3
