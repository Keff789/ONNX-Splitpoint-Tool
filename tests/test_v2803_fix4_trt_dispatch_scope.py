"""Actual generated dispatch requirements and read-only compiler cache probes."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import onnx
from onnx import TensorProto, helper
import pytest
import yaml

from onnx_splitpoint_tool.benchmark import remote_run
from onnx_splitpoint_tool.workflow.artifact_cache_preflight import (
    _trt_observations, build_artifact_cache_preflight,
    resolve_artifact_cache_preflight_policy,
)
from onnx_splitpoint_tool.workflow.benchmark_binding import _benchmark_runs_from_profile
from onnx_splitpoint_tool.workflow.profile_options import load_runtime_profile_snapshot
from tests.test_v27920_remote_trt_cache_preflight import (
    _LocalReadOnlyTransport, _builder_abi, _builder_abi_sha256, _owner,
    _receipt, _sha, _suite,
)
from tests.test_v2803_scope_and_diagnostic_claims import _gate, _prepare_sources


def _generic(**extra):
    return dict(id='ort_tensorrt', type='onnxruntime', provider='tensorrt',
        stage1={'provider': 'tensorrt'}, stage2={'provider': 'tensorrt'}, **extra)


def _matrix():
    return dict(id='hailo10_to_tensorrt', type='matrix',
        stage1={'hw_arch': 'hailo10h'}, stage2={'provider': 'tensorrt'},
        variants=['part1', 'part2', 'composed'])


@pytest.mark.parametrize('variants,p1,p2,full', [
    (None, True, True, True), (['full'], False, False, True),
    (['part1'], True, False, False), (['part2'], False, True, False),
    (['composed'], True, True, False), (['full', 'composed'], True, True, True),
])
def test_generic_dispatch_variants_have_exact_engine_requirements(tmp_path, variants, p1, p2, full):
    row = _generic(**({'variants': variants} if variants is not None else {}))
    suite = _suite(tmp_path, [row])
    required = remote_run._trt_preflight_run_requirements(suite)
    assert required['full_required'] is full
    assert required['p1_cases'] == (['b024'] if p1 else [])
    assert required['generic_p2_cases'] == (['b024'] if p2 else [])
    assert required['p2_cases'] == []  # never mislabel ordinary engines as vendor bridges


def test_disabled_and_inactive_generic_splits_are_not_required(tmp_path):
    suite = _suite(tmp_path, [_generic(), dict(_generic(), id='disabled', enabled=False), _matrix()])
    required = remote_run._trt_preflight_run_requirements(suite, active_run_ids=['hailo10_to_tensorrt'])
    assert required['p1_cases'] == required['generic_p2_cases'] == []
    assert required['p2_cases'] == ['b024']


@pytest.mark.parametrize("quality_only", ["", "--quality-only-run-ids ort_tensorrt",
                                        "--quality-only-run-ids=ort_tensorrt"])
def test_setup_quality_only_probe_keeps_full_and_native_p2(tmp_path, quality_only):
    suite = _real_suite(tmp_path, [_generic(), _matrix()])
    result = remote_run.probe_remote_trt_artifact_cache(
        transport=None, suite_dir=suite, setup_id="host", setup_accelerator="hailo10h",
        active_run_ids=["ort_tensorrt", "hailo10_to_tensorrt"],
        args=remote_run.RemoteBenchmarkArgs(add_args=quality_only),
    )
    required = result["requirement_plan"]
    assert required["full_run_ids"] == ["ort_tensorrt"]
    assert required["p2_cases"] == ["b024"]
    assert required["p2_run_ids_by_case"] == {"b024": ["hailo10_to_tensorrt"]}
    assert required["p1_cases"] == ([] if quality_only else ["b024"])
    assert required["generic_p2_cases"] == ([] if quality_only else ["b024"])
    assert {(row["role"], row["item_id"]) for row in result["requirements"]} == (
        {("trt_full", "host/full"), ("trt_p2", "host/b024")}
        | (set() if quality_only else {("trt_p1", "host/b024"), ("trt_p2", "host/b024:generic")})
    )


def test_quality_only_flag_does_not_remove_vendor_to_trt_requirement(tmp_path):
    suite = _suite(tmp_path, [_generic(), _matrix()])
    required = remote_run._trt_preflight_run_requirements(
        suite, quality_only_run_ids=["hailo10_to_tensorrt"],
    )
    assert required["p1_cases"] == required["generic_p2_cases"] == required["p2_cases"] == ["b024"]


def test_old_full_and_native_hits_do_not_conceal_unprobed_generic_engines(tmp_path):
    suite = _suite(tmp_path, [_generic(), _matrix()])
    policy = resolve_artifact_cache_preflight_policy({'artifact_cache_preflight': {
        'default_expectation': 'warm', 'block_on_unexpected_cold_builds': True}})
    remote = [dict(model_id='model', role=role, item_id='host/' + item,
        status='HIT', reason='verified_existing') for role, item in [('trt_full', 'full'), ('trt_p2', 'b024')]]
    rows, roles = _trt_observations(run_root=tmp_path, model_id='model', policy=policy,
        suite_dir=suite, remote_observations=remote)
    assert {(r.role, r.item_id) for r in rows if r.status == 'UNKNOWN'} == {
        ('trt_p1', 'b024'), ('trt_p2', 'b024:generic')}
    report = build_artifact_cache_preflight(model_ids=['model'], observations=rows,
        applicable_roles={'model': roles}, block_on_unexpected_cold_builds=True)
    assert report['runtime_dispatch_allowed'] is False


def _real_suite(tmp_path, runs):
    suite = _suite(tmp_path / 'input', runs)
    for role, source in [('full', suite / 'models/model.onnx'),
                         ('part1', suite / 'b024/model_part1_b24.onnx'),
                         ('part2', suite / 'b024/model_part2_b24.onnx')]:
        graph = helper.make_graph([helper.make_node('Identity', ['x'], ['y'])], role,
            [helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 3, 4, 4])],
            [helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 3, 4, 4])])
        onnx.save(helper.make_model(graph), source)
    return suite


def test_actual_readonly_probe_cold_then_published_generic_engines(tmp_path):
    suite = _real_suite(tmp_path, [_generic()])
    builder = tmp_path / 'trtexec'; builder.write_bytes(b'compiler fixture, never executed')
    abi = _builder_abi(builder)
    base = tmp_path / 'remote'; base.mkdir()
    key = remote_run._stable_trt_engine_cache_key(suite, builder_abi=abi)
    namespace = base / '_onnx_splitpoint_cache/tensorrt_managed_v27516' / key
    sources = {'full': suite / 'models/model.onnx', 'part1': suite / 'b024/model_part1_b24.onnx',
               'part2': suite / 'b024/model_part2_b24.onnx'}
    leaves = {}
    def publish(role):
        source = sources[role]
        leaf = namespace / remote_run._trt_persistent_engine_relative_dir(role=role,
            case_id='' if role == 'full' else 'b024', source_onnx_sha256=_sha(source), precision='fp16')
        _receipt(leaf=leaf, source_bytes=source.read_bytes(), builder=builder, role=role)
        _owner(namespace, builder_abi_sha256=_builder_abi_sha256(abi))
        leaves[role] = leaf
    def probe():
        before = {str(p): p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
        result = remote_run.probe_remote_trt_artifact_cache(transport=_LocalReadOnlyTransport(base),
            suite_dir=suite, setup_id='host', setup_accelerator='hailo10h',
            resolved_remote_base=str(base), builder_abi=abi)
        assert result['hardware_action_performed'] is False
        assert before == {str(p): p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
        return result
    publish('full')
    result = probe()
    assert [(r['role'], r['status']) for r in result['observations']] == [
        ('trt_full', 'HIT'), ('trt_p1', 'MISS'), ('trt_p2', 'MISS')]
    assert result['requirements'][-1]['engine_precision'] == 'fp16'
    assert result['requirements'][-1]['recipe_source'] == 'generic_same_backend_split'
    publish('part1'); publish('part2')
    assert all(r['status'] == 'HIT' for r in probe()['observations'])
    (leaves['part2'] / 'part2_fp16.engine').write_bytes(b'corrupt engine')
    assert probe()['observations'][-1]['status'] == 'MISS'


def test_generic_and_native_part2_recipes_stay_separate(tmp_path, monkeypatch):
    suite = _real_suite(tmp_path, [_generic(), _matrix()])
    monkeypatch.setattr(remote_run, '_trt_preflight_native_policy', lambda *a, **kw:
        {'precision': 'uint8_cast_fp16', 'policy_sha256': 'f' * 64})
    # No backend runtime/source available: still report the exact requested
    # recipes separately, UNKNOWN rather than treating either as a HIT.
    result = remote_run.probe_remote_trt_artifact_cache(transport=None,
        suite_dir=suite, setup_id='host', setup_accelerator='hailo10h')
    p2 = [r for r in result['requirements'] if r['role'] == 'trt_p2']
    assert {(r['item_id'], r['engine_precision']) for r in p2} == {
        ('host/b024', 'uint8_cast_fp16'), ('host/b024:generic', 'fp16')}
    assert all(r['status'] == 'UNKNOWN' for r in result['observations'])


def test_yolo_only_preparation_preserves_contracts_and_generated_full_companion(tmp_path):
    path, source = _prepare_sources(tmp_path)
    before = path.read_bytes()
    source_payload = yaml.safe_load(before)
    result = _gate().prepare_profile(path, source, only_model='yolo11l')
    assert path.read_bytes() == before
    output = tmp_path / 'prepared.yaml'; output.write_text(yaml.safe_dump(result))
    loaded, _ = load_runtime_profile_snapshot(str(output))
    assert [r['id'] for r in loaded['model_suite']['primary']] == ['yolo11l']
    assert loaded['selection_policy']['forced_cases'] == {'yolo11l': ['b062']}
    assert loaded['native_producers']['case_map'] == {'yolo11l': ['b062']}
    assert loaded['native_producers']['full_baselines']['enabled'] is True
    assert loaded['native_producers']['full_baselines']['backends'] == ['hailo10h', 'tensorrt']
    assert loaded['hailo_build']['calib_count'] == loaded['deepx_build']['calib_count'] == 500
    assert loaded['native_producers']['build_missing_engines'] is False
    assert loaded['hardware']['resolved_targets'] == [r for r in source_payload['hardware']['resolved_targets'] if r['id'] == 'orin_nx_hailo10_01']
    runs = _benchmark_runs_from_profile(loaded, ['hailo10h'])
    by_id = {r['id']: r for r in runs}
    assert by_id['ort_tensorrt']['variants'] == ['full']
    assert 'variants' not in by_id['hailo10']  # existing Hailo generator owns Full-only default
    original_split = next(r for r in source_payload['run_profiles'] if r['id'] == 'hailo10_to_tensorrt')
    assert by_id['hailo10_to_tensorrt']['stage1'] == original_split['stage1']
    assert by_id['hailo10_to_tensorrt']['stage2'] == original_split['stage2']
    assert by_id['hailo10_to_tensorrt'].get('variants') == original_split.get('variants')
    from tests.test_v2772_hailo_full_scope import _resnet_hailo_to_trt_plan
    from onnx_splitpoint_tool.workflow.legacy_benchmarkset_binding import _apply_declared_run_variants
    generated = _resnet_hailo_to_trt_plan(full_hef_policy='end')
    executable = _apply_declared_run_variants(generated.bench_plan_runs, loaded)
    suite = _suite(tmp_path / 'projected', executable)
    required = remote_run._trt_preflight_run_requirements(suite, active_run_ids=['ort_tensorrt', 'hailo10', 'hailo10_to_tensorrt'])
    assert required['full_required'] and required['p2_cases'] == ['b024']
    assert required['p1_cases'] == required['generic_p2_cases'] == []


def test_actual_legacy_generator_carries_validated_reference_variants(tmp_path, monkeypatch):
    # Exercise the real legacy entry point and writer, including its accelerator
    # boolean→run-plan conversion. Only use tiny synthetic models/images; no
    # compiler, remote transport or GPU is part of generation in this test.
    from tests import test_v2802_cpu_reference_binding as real_generator
    from onnx_splitpoint_tool.workflow.legacy_benchmarkset_binding import materialize_legacy_benchmark_set
    from onnx_splitpoint_tool.execution_plan import _expected_normalized_rows
    prep = tmp_path / 'prepare'; prep.mkdir()
    path, source = _prepare_sources(prep)
    profile = _gate().prepare_profile(path, source, only_model='yolo11l')
    serialized = prep / 'fixed.yaml'; serialized.write_text(yaml.safe_dump(profile))
    loaded, _ = load_runtime_profile_snapshot(str(serialized))
    full_recipe = next(copy.deepcopy(r) for r in loaded['run_profiles'] if r['id'] == 'ort_tensorrt')
    def invoke(**kwargs):
        synthetic = copy.deepcopy(kwargs['profile_payload'])
        synthetic['run_profiles'].append(full_recipe)
        kwargs['profile_payload'] = synthetic
        kwargs['targets'] = ['cpu_ort', 'tensorrt']
        kwargs['prediction']['candidates'] = [
            {'case_id': 'b001', 'boundary': 1},
        ]
        return materialize_legacy_benchmark_set(**kwargs)
    monkeypatch.setattr(real_generator, 'materialize_legacy_benchmark_set', invoke)
    suite = real_generator.make_generated_reference_suite(tmp_path / 'generated', 'classification')
    plan = json.loads((suite / 'benchmark_plan.json').read_text())
    trt = next(r for r in plan['runs'] if r['id'] == 'ort_tensorrt')
    assert trt['variants'] == ['full']
    requirements = remote_run._trt_preflight_run_requirements(suite, active_run_ids=['ort_tensorrt'])
    assert requirements['full_required']
    assert requirements['p1_cases'] == requirements['p2_cases'] == requirements['generic_p2_cases'] == []
    assert _expected_normalized_rows(['ort_tensorrt', 'hailo10', 'hailo10_to_tensorrt'], 1, loaded['run_profiles']) == 3


@pytest.mark.parametrize('variants', [[], ['bogus'], ['full', 'full'], 'full'])
def test_profile_schema_rejects_invalid_variant_scope(tmp_path, variants):
    path, source = _prepare_sources(tmp_path)
    profile = _gate().prepare_profile(path, source, only_model='yolo11l')
    profile['run_profiles'][0]['variants'] = variants
    output = tmp_path / 'bad.yaml'; output.write_text(yaml.safe_dump(profile))
    with pytest.raises(ValueError, match='Invalid evaluation profile'):
        load_runtime_profile_snapshot(str(output))


@pytest.mark.parametrize('run_type', ['matrix', 'split', 'onnxruntime', 'ort', 'same_backend_reference'])
def test_explicit_homogeneous_trt_matrix_uses_ordinary_part2(tmp_path, run_type):
    row = _generic(variants=['composed']); row['type'] = run_type
    row['id'] = 'trt_diagnostic'; row.pop('provider')
    suite = _real_suite(tmp_path, [row])
    result = remote_run.probe_remote_trt_artifact_cache(transport=None,
        suite_dir=suite, setup_id='host', setup_accelerator='hailo10h')
    requirements = result['requirement_plan']
    assert requirements['p1_cases'] == requirements['generic_p2_cases'] == ['b024']
    assert requirements['p2_cases'] == []
    p2 = next(r for r in result['requirements'] if r['role'] == 'trt_p2')
    assert p2['engine_precision'] == 'fp16' and p2['item_id'] == 'host/b024:generic'


@pytest.mark.parametrize('run_id', ['ort_cpu', 'ort_cuda', 'hailo10', 'hailo10_to_tensorrt'])
def test_new_profile_variant_override_rejects_unsupported_references(tmp_path, run_id):
    path, source = _prepare_sources(tmp_path)
    profile = _gate().prepare_profile(path, source, only_model='yolo11l')
    profile['run_profiles'][0]['id'] = run_id
    output = tmp_path / 'unsupported.yaml'; output.write_text(yaml.safe_dump(profile))
    with pytest.raises(ValueError, match='Invalid evaluation profile'):
        load_runtime_profile_snapshot(str(output))


def test_provider_only_matrix_never_guesses_trt_stage_direction(tmp_path):
    suite = _suite(tmp_path, [{'id': 'unresolved_matrix', 'type': 'matrix', 'provider': 'tensorrt'}])
    required = remote_run._trt_preflight_run_requirements(suite)
    assert required['p1_cases'] == required['p2_cases'] == required['generic_p2_cases'] == []
    assert required['full_required'] is False
