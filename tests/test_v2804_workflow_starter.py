"""Standard G5 preparation, process boundary and one-command finalization.

Fixtures never claim accelerator execution. The installed target starter is
what performs the real normal workflow; it is not a global cache-only probe.
"""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import sys
import zipfile
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def _gate():
    return _load('v2804_gate_under_test', ROOT / 'scripts/reference_workflow_gate_v2804.py')


def _sources(tmp_path):
    previous = _load('v2803_gate_fixtures', ROOT / 'tests/test_v2803_scope_and_diagnostic_claims.py')
    profile, run = previous._prepare_sources(tmp_path)
    (run / 'profile.yaml').write_bytes(profile.read_bytes())
    return run


def test_t0606_real_normal_profile_loader_standard_two_tasks_no_mutation(tmp_path):
    from onnx_splitpoint_tool.cache_verify_policy import cache_verify_guard
    from onnx_splitpoint_tool.workflow.profile_options import load_runtime_profile_snapshot
    source = _sources(tmp_path)
    before = {str(p): p.read_bytes() for p in source.rglob('*') if p.is_file()}
    gate = _gate()
    profile, report = gate.prepare(source, tmp_path / 'prepared')
    actual, _ = load_runtime_profile_snapshot(str(profile))
    assert report['status'] == 'prepared_not_executed'
    assert report['hardware_execution'] == 'not_run'
    assert actual['selection_policy']['forced_cases'] == {'mobilenet_v3_large': ['b056'], 'yolo11l': ['b062']}
    assert actual['validation_execution']['max_items'] == {'classification': 500, 'detection': 500}
    assert actual['quality_gate']['statistics']['bootstrap_repetitions'] == 500
    assert actual['hailo_build']['calib_count'] == actual['deepx_build']['calib_count'] == 500
    assert actual['hailo_build']['optimization_level'] == 1
    assert actual['hailo_build']['force_build'] is actual['deepx_build']['force_build'] is False
    assert actual['deepx_build']['classification_preprocessing'] == 'imagenet_mean_std'
    assert actual['artifact_cache_preflight'] == {'enabled': True, 'default_expectation': 'warm', 'block_on_unexpected_cold_builds': True}
    assert actual['native_producers']['build_missing_engines'] is False
    assert actual['native_producers']['enabled'] is True
    assert actual['native_producers']['energy']['enabled'] is False
    assert not cache_verify_guard(actual)
    assert {str(p): p.read_bytes() for p in source.rglob('*') if p.is_file()} == before


@pytest.mark.parametrize('failure', ['missing_original', 'missing_split', 'wrong_recipe'])
def test_t0606_preparation_blocks_bad_inputs_before_workflow(tmp_path, failure):
    source = _sources(tmp_path)
    profile = yaml.safe_load((source / 'profile.yaml').read_text())
    if failure == 'missing_original':
        selected = next(m for m in profile['model_suite']['primary'] if m['id'] == 'yolo11l')
        Path(selected['onnx']).unlink()
    elif failure == 'missing_split':
        contract = source / 'models/yolo11l/benchmark_set/legacy_suite/benchmark_set.json'
        data = json.loads(contract.read_text()); data['cases'] = []
        contract.write_text(json.dumps(data))
    else:
        profile['execution_preset']['snapshot']['build']['hailo']['optimization_level'] = 2
        (source / 'profile.yaml').write_text(yaml.safe_dump(profile))
    before = (source / 'profile.yaml').read_bytes()
    with pytest.raises(ValueError, match='original_model_missing|fixed_case_missing|recipe_differs'):
        _gate().prepare(source, tmp_path / 'prepared')
    assert (source / 'profile.yaml').read_bytes() == before
    assert not (tmp_path / 'prepared').exists()


def test_t0606_explicit_source_cannot_be_replaced_by_newer_empty_output(tmp_path):
    source = _sources(tmp_path)
    newer = tmp_path / 'v2804_normal_workflow_newer/runs/empty'; newer.mkdir(parents=True)
    gate = _gate()
    _, report = gate.prepare(source, tmp_path / 'prepared')
    assert report['source_run'] == str(source.resolve())
    with pytest.raises(FileNotFoundError):
        gate.prepare(tmp_path / 'missing_source', tmp_path / 'other')
    assert newer.is_dir()


def test_t0606_real_subprocess_logging_keeps_exit_code(tmp_path, capsys):
    log = tmp_path / 'child.log'
    result = _gate().run_logged([sys.executable, '-I', '-B', '-c', 'print("CHILD_EXECUTED"); raise SystemExit(7)'], tmp_path, log)
    assert result['returncode'] == 7 and not result['cancelled']
    assert log.read_text().strip() == 'CHILD_EXECUTED'
    assert 'CHILD_EXECUTED' in capsys.readouterr().out


@pytest.mark.parametrize('quality', ['pass', 'fail', 'inconclusive'])
def test_t0606_one_command_reaches_debug_export_and_keeps_quality_distinct(tmp_path, monkeypatch, quality):
    gate = _gate()
    tool = tmp_path / 'tool'; python = tool / '.venv/bin/python'
    python.parent.mkdir(parents=True); python.symlink_to(sys.executable)
    monkeypatch.setattr(gate, 'ROOT', tool)
    out = tmp_path / 'output'; out.mkdir()
    source = tmp_path / 'source'; source.mkdir()
    profile = out / 'prepared/profile.yaml'
    def prepare(*args, **kwargs):
        profile.parent.mkdir(); profile.write_text('fixture')
        return profile, {'hardware_execution': 'not_run'}
    calls = []
    def run_logged(command, cwd, log, **kwargs):
        calls.append(command)
        Path(log).write_text('subprocess fixture; no accelerator')
        if 'onnx_splitpoint_tool.workflow.run_evaluation' in command:
            manifest = out / 'runs/actual_run/run_manifest.json'
            manifest.parent.mkdir(parents=True); manifest.write_text('{}')
            assert '--profile-driven' in command and '--require-fresh-run' in command
            assert 'cache_verify_only' not in command
        elif '--output-dir' in command:
            export = Path(command[command.index('--output-dir') + 1]); export.mkdir()
            (export / 'debug_export_acceptance.json').write_text(json.dumps({'status': 'PASS'}))
            (export / 'debug_pack.zip').write_bytes(b'source debug fixture')
        else:
            Path(command[command.index('--out') + 1]).write_bytes(b'debug fixture')
        return {'returncode': 0, 'cancelled': False, 'forced_termination': None}
    monkeypatch.setattr(gate, 'prepare', prepare)
    monkeypatch.setattr(gate, 'run_logged', run_logged)
    from types import SimpleNamespace
    monkeypatch.setattr(gate, '_reuse_gate', lambda: SimpleNamespace(collect=lambda *a: {'status': 'two_fresh_process_hits_pass'}))
    monkeypatch.setattr(gate, 'inspect_run', lambda run: {'technical_acceptance': 'pass', 'quality_decision': quality})
    assert gate.execute(source, out, export_source=True) == 0
    assert len(calls) == 3
    with zipfile.ZipFile(out.with_suffix('.zip')) as archive:
        summary = json.loads(archive.read('output/acceptance.json'))
        assert summary['technical_acceptance'] == 'pass' and summary['quality_decision'] == quality
        assert summary['scientific_final_claim'] is False
        assert 'output/debug_pack.zip' in archive.namelist()
        assert 'output/source_export/debug_pack.zip' in archive.namelist()


def test_t0605_evidence_atomic_failure_no_final_partial_or_run_binaries(tmp_path, monkeypatch):
    gate = _gate()
    out = tmp_path / 'evidence'; out.mkdir()
    (out / 'acceptance.json').write_text('{}')
    model = out / 'runs/test/model.hef'; model.parent.mkdir(parents=True); model.write_bytes(b'private binary')
    original = zipfile.ZipFile.write
    def fail(*args, **kwargs):
        raise OSError('synthetic ENOSPC')
    monkeypatch.setattr(zipfile.ZipFile, 'write', fail)
    with pytest.raises(OSError, match='ENOSPC'):
        gate.archive_evidence(out)
    assert not out.with_suffix('.zip').exists()
    assert not out.with_suffix('.zip.part').exists()
    monkeypatch.setattr(zipfile.ZipFile, 'write', original)
    archive = gate.archive_evidence(out)
    with zipfile.ZipFile(archive) as zf:
        assert zf.namelist() == ['evidence/acceptance.json']
    before = archive.read_bytes()
    with pytest.raises(FileExistsError): gate.archive_evidence(out)
    assert archive.read_bytes() == before


def test_t0302_explicit_overlay_argument_only_changes_private_profile(tmp_path):
    source = _sources(tmp_path)
    before = (source / 'profile.yaml').read_bytes()
    value = str(tmp_path / 'not_required_for_HIT' / 'overlay_manifest.json')
    profile, _ = _gate().prepare(source, tmp_path / 'prepared', hailo8_dependency_manifest=value)
    actual = yaml.safe_load(profile.read_text())
    assert actual['hailo_build']['compute_by_family']['hailo8']['dependency_manifest'] == value
    assert actual['hailo_build']['compute_by_family']['hailo8']['device'] == 'gpu'
    assert actual['hailo_build']['compute_by_family']['hailo10h'] == yaml.safe_load(before)['hailo_build']['compute_by_family']['hailo10h']
    assert (source / 'profile.yaml').read_bytes() == before


def test_t0605_controller_output_error_cleans_its_live_child_and_preserves_primary_error(tmp_path, monkeypatch):
    import builtins
    import os
    gate = _gate()
    def broken_output(*args, **kwargs):
        if args and 'CHILD_READY' in str(args[0]):
            raise OSError('synthetic controller output closed')
        return builtins.print(*args, **kwargs)
    monkeypatch.setattr(gate, 'print', broken_output, raising=False)
    log = tmp_path / 'child.log'
    with pytest.raises(OSError, match='controller output closed') as caught:
        gate.run_logged([sys.executable, '-I', '-B', '-u', '-c',
                         'import os,time; print("CHILD_READY",os.getpid(),flush=True); time.sleep(60)'], tmp_path, log)
    pid = int(log.read_text().splitlines()[0].split()[1])
    with pytest.raises(ProcessLookupError): os.kill(pid, 0)
    assert caught.value.workflow_cleanup['status'] in {'terminated_after_controller_error', 'killed_residual_owned_group_after_controller_error'}


@pytest.mark.parametrize('missing,reason', [
    ('hailo10', 'hailo_full_quality_missing'),
    ('native_full_tensorrt', 'native_tensorrt_full_quality_missing'),
    ('hailo10h_to_trt', 'bound_split_quality_missing'),
])
def test_t0606_native_performance_or_generic_full_cannot_replace_required_quality_endpoint(missing, reason):
    # Actual .3 result addressing: generic Full rows carry the selected case;
    # Native Full TRT uses full. Its generic ort_tensorrt companion is distinct.
    scope = {('hailo10', 'b062', 'full'), ('hailo10h_to_trt', 'b062', 'composed'),
             ('native_full_tensorrt', 'full', 'full'), ('ort_tensorrt', 'b062', 'full')}
    gate = _gate()
    gate.require_central_endpoint_coverage(scope, 'yolo11l', 'b062')
    missing_scope = {entry for entry in scope if entry[0] != missing}
    with pytest.raises(ValueError, match=reason):
        gate.require_central_endpoint_coverage(missing_scope, 'yolo11l', 'b062')
