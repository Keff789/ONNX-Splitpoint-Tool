"""Current source bindings survive normal import of an older OWN donor runtime."""
from copy import deepcopy
import json

import pytest

from onnx_splitpoint_tool.benchmark.generic_quality_transport import selected_source_transport_excludes
from onnx_splitpoint_tool.workflow.generic_quality_sidecars import _check_sources, materialize_generic_quality_sidecars
from test_generic_quality_selected_local_import import prepare, selection, normal_import
from test_generic_quality_selected_transport import MODEL, RUN, republish


def full_runtime(tmp_path, *, stale=True):
    state = prepare(tmp_path)
    assert len(state['config']['runtime_source_files']) == 37
    members = {path.relative_to(state['suite']).as_posix(): path.read_bytes()
               for path in (state['suite'] / 'splitpoint_runners').rglob('*') if path.is_file()}
    members['b001/run_split_onnxruntime.py'] = (state['suite'] / 'b001/run_split_onnxruntime.py').read_bytes()
    assert len(members) == 42 and sum(name.endswith('.py') for name in members) == 37
    state['current_runtime_members'] = members
    republish(state)
    if stale:
        (state['suite'] / 'b001/run_split_onnxruntime.py').write_bytes(b'# OWN historical runner of a different size\n')
        (state['suite'] / 'splitpoint_runners/generic_quality_completion.py').write_bytes(b'# OWN historical completion module\n')
    return state


def snapshot(root):
    return {path.relative_to(root).as_posix(): path.read_bytes()
            for path in root.rglob('*') if path.is_file()}


def test_old_donor_runtime_is_replaced_only_after_complete_current37_proof(tmp_path):
    state = full_runtime(tmp_path)
    before = snapshot(state['suite'])
    selected = selection(state)
    assert len(selected) == 133
    assert snapshot(state['suite']) == before
    result = normal_import(state)
    suite = result.suite_dir
    assert not (suite / 'b020/part1.onnx').exists()
    assert len(json.loads((suite / 'benchmark_set.json').read_text())['cases']) == 20
    plan = json.loads((suite / 'benchmark_plan.json').read_text())
    materialize_generic_quality_sidecars(suite_dir=suite, model_id=MODEL, benchmark_plan=plan)
    _check_sources(state['config'], suite)
    for relative, body in state['current_runtime_members'].items():
        assert (suite / relative).read_bytes() == body
    assert selected_source_transport_excludes(suite, active_run_ids=[RUN], model_id=MODEL) == selected
    assert snapshot(state['suite']) == before
    assert all('size_bytes' in value for value in state['config']['runtime_source_files'])


@pytest.mark.parametrize('fault', [
    'runner_sha', 'module_sha', 'runner_size', 'module_size', 'missing_size', 'bool_size',
    'duplicate_source', 'unknown_source', 'foreign_case', 'escape', 'absolute',
])
def test_wrong_current_runtime_binding_rejects_before_private_import(tmp_path, fault):
    state = full_runtime(tmp_path)
    sources = state['config']['runtime_source_files']
    runner = next(value for value in sources if value['path'] == 'b001/run_split_onnxruntime.py')
    module = next(value for value in sources if value['path'] == 'splitpoint_runners/generic_quality_completion.py')
    if fault in {'runner_sha', 'module_sha'}:
        (runner if fault == 'runner_sha' else module)['sha256'] = '0' * 64
    elif fault in {'runner_size', 'module_size'}:
        (runner if fault == 'runner_size' else module)['size_bytes'] += 1
    elif fault == 'missing_size':
        runner.pop('size_bytes')
    elif fault == 'bool_size':
        runner['size_bytes'] = True
    elif fault == 'duplicate_source':
        sources[-1] = deepcopy(sources[0])
    elif fault == 'unknown_source':
        runner['path'] = 'splitpoint_runners/OWN_unknown.py'
    elif fault == 'foreign_case':
        runner['path'] = 'b002/run_split_onnxruntime.py'
    elif fault == 'escape':
        runner['path'] = '../OWN_runtime.py'
    else:
        runner['path'] = str(tmp_path / 'OWN_runtime.py')
    republish(state)
    before = snapshot(state['suite'])
    with pytest.raises((ValueError, OSError)):
        normal_import(state)
    assert not state['run_root'].exists()
    assert snapshot(state['suite']) == before


@pytest.mark.parametrize('fault', ['artifact_size', 'artifact_missing', 'artifact_foreign_case', 'donor_symlink'])
def test_regenerated_runtime_never_substitutes_artifacts_or_unsafe_donor_paths(tmp_path, fault):
    state = full_runtime(tmp_path)
    if fault == 'artifact_size':
        state['config']['expected_runtime_artifacts']['source_part1_onnx']['size_bytes'] += 1
    elif fault == 'artifact_missing':
        (state['suite'] / 'b001/part1.onnx').unlink()
    elif fault == 'artifact_foreign_case':
        state['config']['expected_runtime_artifacts']['source_part1_onnx']['path'] = 'b002/part1.onnx'
    else:
        runner = state['suite'] / 'b001/run_split_onnxruntime.py'
        elsewhere = tmp_path / 'OWN_historical_runner.py'
        runner.rename(elsewhere)
        runner.symlink_to(elsewhere)
    republish(state)
    if fault == 'artifact_missing':
        # Missing original graph retains the existing full-copy fallback; the
        # ordinary graph/sidecar admission will still reject the missing graph.
        assert selection(state) == ()
        return
    with pytest.raises((ValueError, OSError)):
        normal_import(state)
    assert not state['run_root'].exists()


def test_postcopy_runtime_change_is_still_rejected_by_normal_source_guard(tmp_path):
    state = full_runtime(tmp_path)
    result = normal_import(state)
    target = result.suite_dir / 'b001/run_split_onnxruntime.py'
    target.write_bytes(target.read_bytes() + b'# OWN late drift\n')
    plan = json.loads((result.suite_dir / 'benchmark_plan.json').read_text())
    with pytest.raises(ValueError, match='generic_quality_sidecar_runtime_source_changed'):
        materialize_generic_quality_sidecars(suite_dir=result.suite_dir, model_id=MODEL, benchmark_plan=plan)


def test_remote_transport_still_checks_actual_runtime_sizes_without_regeneration(tmp_path):
    state = full_runtime(tmp_path)
    with pytest.raises(ValueError, match='generic_quality_transport_dependency_unproved'):
        selected_source_transport_excludes(state['suite'], active_run_ids=[RUN], model_id=MODEL)
