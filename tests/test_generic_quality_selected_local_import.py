"""Selective local imports over fresh OWN full20 fixtures; no hardware."""
from __future__ import annotations

import hashlib
import json

import pytest
from onnx import TensorProto

from onnx_splitpoint_tool.benchmark.generic_quality_transport import (
    selected_source_import_excludes, selected_source_transport_excludes,
)
from onnx_splitpoint_tool.workflow.artifacts import file_record
from onnx_splitpoint_tool.workflow.generator_binding import import_existing_suite
from onnx_splitpoint_tool.workflow.generic_quality_sidecars import materialize_generic_quality_sidecars
from onnx_splitpoint_tool.workflow.runner import benchmark_set_postcondition_v60v
from onnx_splitpoint_tool.gui.controller import _copy_runner_lib
from onnx_splitpoint_tool.split_export_runners import write_runner_skeleton_onnxruntime
from test_generic_quality_selected_transport import (
    MODEL, RUN, BINARY_SUFFIXES, case_binary, fixture, json_file,
    republish, tiny_inline_model, write,
)
from test_v2796_artifact_index_closure import _runner


def prepare(tmp_path, *, images=False):
    state = fixture(tmp_path, images=images)
    _copy_runner_lib(state['suite'])
    write_runner_skeleton_onnxruntime(str(state['suite'] / 'b001'))
    runtime = [state['suite'] / 'b001/run_split_onnxruntime.py',
               *sorted((state['suite'] / 'splitpoint_runners').rglob('*.py'))]
    state['config']['runtime_source_files'] = [
        {'path': path.relative_to(state['suite']).as_posix(),
         'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
         'size_bytes': path.stat().st_size} for path in runtime]
    # Engines retain the ordinary absolute DUT cache binding; portable assets
    # support JSON/HEF/DXNN only, rather than a new engine staging convention.
    state['config']['expected_runtime_artifacts']['engine']['path'] = '/OWN_DUT_CACHE/selected.engine'
    republish(state)
    catalogue_path = state['suite'] / 'benchmark_set.json'
    catalogue = json.loads(catalogue_path.read_text())
    for index, case in enumerate(catalogue['cases'], 1):
        case.update(case_id=case['case_dir'], boundary=index)
    json_file(catalogue_path, catalogue)
    state['candidate_plan'] = {'model_id': MODEL, 'selected_candidates': [
        {'case_id': case, 'split_index': index}
        for index, case in enumerate(state['cases'], 1)]}
    state['profile'] = {'run_profiles': [state['run']],
                        'quality_gate': {'statistics': {'execution_location': 'central_management'}}}
    state['run_root'] = tmp_path / 'new_run'
    return state


def selection(state, **kwargs):
    return selected_source_import_excludes(
        state['suite'], runs=state['profile']['run_profiles'], model_id=MODEL, **kwargs)


def normal_import(state):
    return import_existing_suite(
        model_id=MODEL, row={'benchmark_set_dir': str(state['suite'])},
        model_dir=state['run_root'] / 'models' / MODEL,
        run_dir=state['run_root'], profile_id='OWN_CURRENT', run_id='new_run',
        prediction={}, candidate_plan=state['candidate_plan'], targets=['hailo10', 'tensorrt'],
        profile_payload=state['profile'])


def test_normal_import_preserves_full20_shared_inputs_sources_and_terminal_coverage(tmp_path):
    state = prepare(tmp_path, images=True)
    source = state['suite']
    expected = selected_source_transport_excludes(source, active_run_ids=[RUN], model_id=MODEL)
    # The imported suite's historical executable work is not current authority.
    json_file(source / 'benchmark_plan.json', {'runs': [{'id': 'OWN_OLD_NATIVE'}]})
    (source / 'quality_runtime/b001.json').unlink()
    before = {p.relative_to(source).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in source.rglob('*') if p.is_file()}
    assert selection(state) == expected
    result = normal_import(state)
    suite = result.suite_dir
    for case in state['cases']:
        assert (suite / case).is_dir()
        for name in ('split_manifest.json', 'gate.json', 'metadata.bin.json'):
            relative = case + '/' + name
            assert hashlib.sha256((suite / relative).read_bytes()).hexdigest() == before[relative]
        for suffix in BINARY_SUFFIXES:
            assert (suite / case_binary(case, suffix)).is_file() is (case == 'b001')
        assert (suite / case / 'part2.onnx').is_file() is (case == 'b001')
    assert (suite / 'models/model.onnx').read_bytes() == (source / 'models/model.onnx').read_bytes()
    dataset_manifest = state['dataset'].relative_to(source) / 'manifest.json'
    assert (suite / dataset_manifest).read_bytes() == (source / dataset_manifest).read_bytes()
    assert len(list((suite / state['dataset'].relative_to(source) / 'images').iterdir())) == 5000
    actual = json.loads((suite / 'benchmark_set.json').read_text())
    assert len(actual['cases']) == 20
    assert benchmark_set_postcondition_v60v(suite.parent)['valid'] is True
    plan = json.loads((suite / 'benchmark_plan.json').read_text())
    records = materialize_generic_quality_sidecars(suite_dir=suite, model_id=MODEL, benchmark_plan=plan)
    assert [r['case_id'] for r in records] == ['b001']
    assert selected_source_transport_excludes(suite, active_run_ids=[RUN], model_id=MODEL) == expected
    generator_input = json.loads(result.artifacts['generator_input_json'].read_text())
    assert tuple(generator_input['local_runtime_binary_excludes']) == expected
    runner = _runner(state['run_root'])
    runner.run_id = 'new_run'
    runner.artifact_index['run_id'] = 'new_run'
    runner.outputs = {key: str(path) for key, path in result.artifacts.items()}
    runner.artifact_index['artifacts'] = [file_record(
        path, root=state['run_root'], kind='stage_artifact', producer_stage='OWN_IMPORT')
        for path in result.artifacts.values()]
    runner._finalize_artifact_index(status='partial')
    assert runner._verify_terminal_artifact_index(required_paths=runner._terminal_evidence_candidates())['ok']
    closure = json.loads((state['run_root'] / 'reports/artifact_index_closure.json').read_text())
    assert closure['status'] == 'pass' and closure['workflow_status'] == 'partial'
    indexed = json.loads(runner.artifact_index_path.read_text())['artifacts']
    assert not any('/b002/part1.onnx' in r['path'] for r in indexed)
    assert {p.relative_to(source).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in source.rglob('*') if p.is_file()} == before
    for path in suite.rglob('*'):
        assert not path.is_symlink()
        if path.is_file() and (source / path.relative_to(suite)).is_file():
            assert path.stat().st_ino != (source / path.relative_to(suite)).stat().st_ino


def test_exact_new_asset_can_be_supplied_by_original_generic_config(tmp_path):
    state = prepare(tmp_path)
    item = state['config']['expected_runtime_artifacts']['part1_runtime']
    original = write(tmp_path / 'own_backend_cache/selected.hef', b'OWN selected HEF')
    item['local_path'] = original['path']
    (state['suite'] / item['path']).unlink()
    republish(state)
    assert selection(state)
    result = normal_import(state)
    plan = json.loads((result.suite_dir / 'benchmark_plan.json').read_text())
    materialize_generic_quality_sidecars(suite_dir=result.suite_dir, model_id=MODEL, benchmark_plan=plan)
    assert (result.suite_dir / item['path']).read_bytes() == b'OWN selected HEF'


@pytest.mark.parametrize('fault', ['unknown_schema', 'unknown_metadata', 'build_policy', 'missing_old_runtime', 'external_shared_graph'])
def test_unproved_import_eligibility_retains_full_copy(tmp_path, fault):
    state = prepare(tmp_path)
    if fault == 'unknown_schema':
        state['config']['schema'] = 'OWN_UNKNOWN_CONFIG_SCHEMA'
    elif fault == 'unknown_metadata':
        state['config'].pop('expected_runtime_artifacts')
    elif fault == 'build_policy':
        state['run'].pop('native_trt_build')
    elif fault == 'missing_old_runtime':
        (state['suite'] / 'splitpoint_runners/generic_quality_completion.py').unlink()
    else:
        model = tiny_inline_model()
        tensor = model.graph.initializer[0]
        tensor.ClearField('float_data')
        tensor.data_location = TensorProto.EXTERNAL
        tensor.external_data.add(key='location', value='../b002/part1.data')
        write(state['suite'] / 'models/model.onnx', model.SerializeToString())
    republish(state)
    logs = []
    assert selection(state, log=logs.append) == ()
    assert logs and 'retaining full suite:' in logs[-1]
    result = normal_import(state)
    assert (result.suite_dir / 'b020/part1.onnx').read_bytes() == (state['suite'] / 'b020/part1.onnx').read_bytes()
    assert json.loads(result.artifacts['generator_input_json'].read_text())['local_runtime_binary_excludes'] == []


@pytest.mark.parametrize('fault', ['changed_original_config', 'foreign_case_dependency', 'escape_dependency', 'wrong_identity'])
def test_explicit_import_binding_fault_rejects_before_copy(tmp_path, fault):
    state = prepare(tmp_path)
    if fault == 'changed_original_config':
        state['origin'].write_text(state['origin'].read_text() + ' ')
    else:
        if fault == 'foreign_case_dependency':
            state['config']['expected_runtime_artifacts']['engine']['path'] = 'b002/part1.engine'
        elif fault == 'escape_dependency':
            state['config']['expected_runtime_artifacts']['source_part1_onnx']['path'] = '../foreign.onnx'
        else:
            state['config']['identity']['setup_id'] = 'OWN_FOREIGN'
        republish(state)
    with pytest.raises(ValueError):
        normal_import(state)
    assert not state['run_root'].exists()


def test_non_generic_import_retains_all_case_binaries(tmp_path):
    state = prepare(tmp_path)
    state['profile'] = {}
    result = normal_import(state)
    assert (result.suite_dir / 'b020/part1.onnx').read_bytes() == (state['suite'] / 'b020/part1.onnx').read_bytes()
