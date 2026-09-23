"""AP01 settings transport; synthetic CPU statistics only, no workflow start."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest
import yaml

from onnx_splitpoint_tool import quality_service as service_module
from onnx_splitpoint_tool import run_modes
from onnx_splitpoint_tool.benchmark.evaluation_profiles import (
    load_evaluation_profile,
    save_evaluation_profile_yaml,
    validate_evaluation_profile_payload,
)
from onnx_splitpoint_tool.quality_statistics_config import (
    DEFAULTS, reference_threads, statistics_options,
)


def _profile(*, preset=True):
    source = Path(__file__).resolve().parents[1] / 'onnx_splitpoint_tool/resources/evaluation_profiles/smoke_regression_v1.yaml'
    payload = yaml.safe_load(source.read_text())
    if preset:
        payload['execution_preset'] = {
            'id': 'final', 'follow_tool_config': False,
            'snapshot': run_modes.default_run_modes_config()['modes']['final'],
        }
        payload, _ = run_modes.apply_run_mode(payload)
    else:
        payload.pop('execution_preset', None)
    return payload


def _custom(profile):
    stats = profile.setdefault('quality_gate', {}).setdefault('statistics', {})
    stats.update(engine='optimized_coco_v1', workers=2, block_repetitions=128,
                 checkpoint_blocks=True, prepared_cache_limit_mib=64,
                 max_active_requests=1, execution_location='central_management')
    profile['quality_gate']['management_reference'] = {'intra_op_threads': 6}
    return profile


@pytest.mark.parametrize('workers', [1, 4, 8])
def test_legacy_profile_keeps_statistics_workers_and_reference_inheritance(tmp_path, workers):
    profile = _profile(preset=False)
    profile.setdefault('quality_gate', {}).setdefault('statistics', {})['workers'] = workers
    path = tmp_path / 'legacy.yaml'
    save_evaluation_profile_yaml(path, profile)
    before = path.read_bytes()
    loaded = load_evaluation_profile(path)
    assert loaded.raw_profile['quality_gate']['statistics']['workers'] == workers
    assert reference_threads(loaded.raw_profile) == workers
    assert statistics_options(loaded.raw_profile) == DEFAULTS
    assert path.read_bytes() == before


def test_normal_save_load_snapshot_preserves_independent_execution_controls(tmp_path):
    profile = _custom(_profile())
    before = copy.deepcopy(profile)
    path = tmp_path / 'optimized.yaml'
    save_evaluation_profile_yaml(path, profile)
    loaded = load_evaluation_profile(path)
    expected = profile['quality_gate']['statistics']
    actual = loaded.raw_profile['quality_gate']['statistics']
    assert actual == expected
    assert loaded.start_snapshot['resolved_profile']['quality_gate']['statistics'] == expected
    assert reference_threads(loaded.raw_profile) == 6
    assert loaded.raw_profile['quality_gate']['management_reference'] == {'intra_op_threads': 6}
    assert actual['bootstrap_repetitions'] == 5000
    assert actual['seed'] == 20260710
    assert actual['confidence_level'] == .95
    assert loaded.raw_profile['validation_execution']['max_items'] == {'classification': 5000, 'detection': 5000}
    assert profile == before


@pytest.mark.parametrize('block', [128, 256, 512])
def test_run_mode_registry_save_reload_materialization_and_snapshot(tmp_path, block):
    config = run_modes.default_run_modes_config()
    config['modes']['final']['quality'].update(
        workers=3, reference_intra_op_threads=7,
        statistics_engine='optimized_coco_v1',
        statistics_block_repetitions=block,
        statistics_checkpoint_blocks=True,
        statistics_prepared_cache_limit_mib=128,
    )
    path = tmp_path / 'run_modes.yaml'
    run_modes.save_run_modes_config(path, config)
    loaded = run_modes.load_run_modes_config(path)
    profile = _profile()
    profile.pop('quality_gate')
    profile['execution_preset'] = {'id': 'final', 'follow_tool_config': True}
    resolved, _ = run_modes.apply_run_mode(profile, config=loaded, config_path=path)
    stats = resolved['quality_gate']['statistics']
    assert stats['workers'] == 3
    assert stats['engine'] == 'optimized_coco_v1'
    assert stats['block_repetitions'] == block
    assert stats['checkpoint_blocks'] is True
    assert stats['prepared_cache_limit_mib'] == 128
    assert stats['bootstrap_repetitions'] == 5000
    assert reference_threads(resolved) == 7
    assert resolved['execution_preset']['snapshot']['quality'] == config['modes']['final']['quality']


@pytest.mark.parametrize('field,value', [
    ('engine', 'unknown'), ('engine', None),
    ('max_active_requests', 0), ('max_active_requests', 3),
    ('block_repetitions', 0), ('block_repetitions', 5001),
    ('block_repetitions', True), ('block_repetitions', '128'),
    ('block_repetitions', None), ('checkpoint_blocks', 'false'),
    ('checkpoint_blocks', 1), ('checkpoint_blocks', None),
    ('prepared_cache_limit_mib', 0), ('prepared_cache_limit_mib', 15),
    ('prepared_cache_limit_mib', 65537), ('prepared_cache_limit_mib', None),
    ('workers', 0), ('workers', 65), ('workers', True), ('workers', '4'), ('workers', None),
])
def test_invalid_optimized_controls_are_rejected_before_normal_profile_save(tmp_path, field, value):
    profile = _custom(_profile())
    profile['quality_gate']['statistics'][field] = value
    with pytest.raises(ValueError):
        validate_evaluation_profile_payload(profile)
    target = tmp_path / 'must_not_publish.yaml'
    with pytest.raises(ValueError):
        save_evaluation_profile_yaml(target, profile)
    assert not target.exists()


@pytest.mark.parametrize('value', [0, -1, 65, True, '6', 2.5])
def test_invalid_explicit_reference_threads_fail_without_changing_workers(value):
    profile = _custom(_profile())
    profile['quality_gate']['management_reference']['intra_op_threads'] = value
    with pytest.raises(ValueError, match='intra_op_threads'):
        reference_threads(profile)
    with pytest.raises(ValueError):
        validate_evaluation_profile_payload(profile)
    assert profile['quality_gate']['statistics']['workers'] == 2


def test_unknown_optimized_option_cannot_disappear_during_materialization(tmp_path):
    profile = _custom(_profile())
    profile['quality_gate']['statistics']['block_repetition_typo'] = 128
    with pytest.raises(ValueError, match='unknown optimized statistics'):
        run_modes.apply_run_mode(profile)
    target = tmp_path / 'must_not_publish.yaml'
    with pytest.raises(ValueError, match='unknown optimized statistics'):
        save_evaluation_profile_yaml(target, profile)
    assert not target.exists()


def _service_owner(profile, run_dir):
    from onnx_splitpoint_tool.workflow.runner import EvaluationWorkflowRunner
    owner = object.__new__(EvaluationWorkflowRunner)
    owner.profile_payload = profile
    owner.run_dir = run_dir
    owner._management_services_lock = threading.RLock()
    owner._management_admission_closed = False
    owner._central_quality_service = None
    owner._central_quality_reference_store = None
    owner._central_quality_coord_executor = None
    return owner


def test_normal_runner_entry_uses_actual_spawn_service_and_saved_settings(tmp_path, monkeypatch):
    # No EvaluationWorkflowRunner constructor/run call: only the normal service
    # entry with a saved/loaded immutable snapshot. Tiny paired-mean CPU work.
    profile = _custom(_profile())
    path = tmp_path / 'profile.yaml'
    save_evaluation_profile_yaml(path, profile)
    loaded = load_evaluation_profile(path)
    owner = _service_owner(loaded.start_snapshot['resolved_profile'], tmp_path / 'derived')
    real_executor = service_module.ProcessPoolExecutor
    submitted = []
    spawned = []
    class CapturedProcessPool(real_executor):
        def __init__(self, *args, **kwargs):
            spawned.append((kwargs['max_workers'], kwargs['mp_context'].get_start_method()))
            super().__init__(*args, **kwargs)
        def submit(self, fn, *args, **kwargs):
            if args and isinstance(args[0], dict):
                submitted.append(copy.deepcopy(args[0].get('_statistics', {})))
            return super().submit(fn, *args, **kwargs)
    monkeypatch.setattr(service_module, 'ProcessPoolExecutor', CapturedProcessPool)
    request = service_module.QualityEvaluationRequest(
        reference_records=[{'image_id': i, 'value': float(i % 2)} for i in range(6)],
        candidate_records=[{'image_id': i, 'value': float(i % 2) - .01} for i in range(6)],
        annotations=[], metric_gate_config={'primary_metric': 'mean'},
        repetitions=11, seed=20260710, confidence_level=.95,
        non_inferiority_margin=.1, evaluator_factory='paired_mean',
    )
    service = owner._ensure_central_quality_service()
    try:
        result = service.evaluate(request, timeout=30)
        assert spawned == [(2, 'spawn')]
        assert service.workers == 2
        assert service.statistics['engine'] == 'optimized_coco_v1'
        assert service.statistics['block_repetitions'] == 128
        assert result['primary']['bootstrap_repetitions_requested'] == 11
        assert result['statistics_observation']['engine'] == 'optimized_coco_v1'
        assert submitted
        for options in submitted:
            assert options['engine'] == 'optimized_coco_v1'
            assert options['checkpoint_blocks'] is True
        assert reference_threads(owner.profile_payload) == 6
    finally:
        service.shutdown(wait=True)
        owner._central_quality_coord_executor.shutdown(wait=True)


def test_reference_schedule_passes_independent_threads_without_running_inference(tmp_path, monkeypatch):
    from onnx_splitpoint_tool import management_reference
    from onnx_splitpoint_tool.workflow import runner
    profile = _custom(_profile())
    owner = _service_owner(profile, tmp_path / 'derived')
    calls = []
    class CaptureReferenceExecutor:
        def submit(self, function, **kwargs):
            calls.append((function, kwargs))
            return SimpleNamespace()
    owner._management_reference_executor = CaptureReferenceExecutor()
    owner._management_reference_futures = {}
    owner._management_reference_source_contracts = {}
    owner._management_cancel_event = threading.Event()
    owner._process_registry = object()
    owner.log = lambda *_: None
    owner._emit_log = lambda *_: None
    monkeypatch.setattr(management_reference, '_source_contract', lambda *_: 'a' * 64)
    suite = tmp_path / 'suite'
    suite.mkdir()
    (suite / 'benchmark_plan.json').write_text('{}')
    (suite / 'benchmark_set.json').write_text('{}')
    owner._schedule_management_cpu_reference('synthetic', suite)
    assert len(calls) == 1
    function, kwargs = calls[0]
    assert function == owner._run_budgeted_management_reference
    assert kwargs["reference_function"] is management_reference.generate_management_cpu_reference
    assert kwargs['workers'] == 6
    assert runner._quality_workers_v263(profile) == 2
    assert owner._management_reference_source_contracts == {'synthetic': 'a' * 64}


@pytest.fixture
def tk_root():
    import tkinter as tk
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f'real Tk display unavailable; no product GUI started: {exc}')
    root.withdraw()
    yield root
    root.destroy()


def test_real_profile_editor_saves_and_reloads_independent_controls(tk_root, tmp_path, monkeypatch):
    import tkinter as tk
    from onnx_splitpoint_tool.gui import profile_editor
    original = tmp_path / 'input.yaml'
    save_evaluation_profile_yaml(original, _profile())
    errors = []
    monkeypatch.setattr(profile_editor.messagebox, 'showerror', lambda *args, **kwargs: errors.append(args))
    editor = profile_editor.EvaluationProfileEditor(tk_root, profile_var=tk.StringVar(tk_root, str(original)))
    editor.withdraw()
    try:
        assert editor._load_profile(str(original)), errors
        editor.var_quality_workers.set(2)
        editor.var_reference_threads.set('6')
        editor.var_statistics_engine.set('optimized_coco_v1')
        editor.var_statistics_block.set(128)
        editor.var_statistics_checkpoint.set(True)
        editor.var_statistics_cache_mib.set(64)
        saved = tmp_path / 'saved.yaml'
        monkeypatch.setattr(profile_editor.filedialog, 'asksaveasfilename', lambda **kwargs: str(saved))
        editor._save(use_after=False)
        assert saved.is_file(), errors
        assert editor._load_profile(str(saved)), errors
        assert editor.var_quality_workers.get() == 2
        assert editor.var_reference_threads.get() == '6'
        assert editor.var_statistics_engine.get() == 'optimized_coco_v1'
        assert editor.var_statistics_block.get() == 128
        assert editor.var_statistics_checkpoint.get() is True
        loaded = load_evaluation_profile(saved)
        assert loaded.start_snapshot['resolved_profile']['quality_gate']['statistics']['workers'] == 2
    finally:
        editor.destroy()


def test_real_mode_editor_roundtrip_independent_controls(tk_root, tmp_path, monkeypatch):
    from onnx_splitpoint_tool.gui import run_mode_editor
    saved = []
    errors = []
    monkeypatch.setattr(run_mode_editor.messagebox, 'showerror', lambda *args, **kwargs: errors.append(args))
    dialog = run_mode_editor.RunModeEditDialog(tk_root, mode_id='final', config=run_modes.default_run_modes_config(), on_saved=lambda config: saved.append(config))
    dialog.withdraw()
    dialog.vars['quality.workers'].set('2')
    dialog.vars['quality.reference_intra_op_threads'].set('6')
    dialog.vars['quality.statistics_engine'].set('optimized_coco_v1')
    dialog.vars['quality.statistics_block_repetitions'].set('128')
    dialog.vars['quality.statistics_checkpoint_blocks'].set(True)
    dialog._save()
    assert saved, errors
    path = tmp_path / 'run_modes.yaml'
    run_modes.save_run_modes_config(path, saved[0])
    quality = run_modes.load_run_modes_config(path)['modes']['final']['quality']
    assert quality['workers'] == 2
    assert quality['reference_intra_op_threads'] == 6
    assert quality['statistics_engine'] == 'optimized_coco_v1'
    assert quality['statistics_block_repetitions'] == 128
    assert quality['statistics_checkpoint_blocks'] is True


def test_optimized_run_mode_rejects_unknown_statistics_prefix_before_save(tmp_path):
    config = run_modes.default_run_modes_config()
    config['modes']['final']['quality'].update(
        statistics_engine='optimized_coco_v1', statistics_block_repetition_typo=128,
    )
    with pytest.raises(ValueError, match='statistics_block_repetition_typo'):
        run_modes.validate_run_modes_config(config)
    path = tmp_path / 'must_not_publish_registry.yaml'
    with pytest.raises(ValueError, match='statistics_block_repetition_typo'):
        run_modes.save_run_modes_config(path, config)
    assert not path.exists()


def test_optimized_run_mode_preserves_unrelated_quality_extensions():
    config = run_modes.default_run_modes_config()
    quality = config['modes']['final']['quality']
    quality['statistics_engine'] = 'optimized_coco_v1'
    quality['external_reporting_extension'] = {'preserve': True}
    validated = run_modes.validate_run_modes_config(config)
    assert validated['modes']['final']['quality']['external_reporting_extension'] == {'preserve': True}


def test_direct_service_constructor_rejects_unsupported_engine(tmp_path):
    # Context manager guarantees cleanup if an invalid setting is accidentally
    # accepted. No request, worker job or inference is submitted.
    with pytest.raises(ValueError, match='engine'):
        with service_module.ManagementQualityService(
            tmp_path / 'cache', workers=1, statistics={'engine': 'unsupported_engine'},
        ):
            pass


@pytest.mark.parametrize('workers', [0, -1, 65])
def test_real_profile_editor_rejects_invalid_optimized_worker_count(
    tk_root, tmp_path, monkeypatch, workers,
):
    import tkinter as tk
    from onnx_splitpoint_tool.gui import profile_editor

    original = tmp_path / 'input.yaml'
    save_evaluation_profile_yaml(original, _profile())
    original_bytes = original.read_bytes()
    errors = []
    monkeypatch.setattr(
        profile_editor.messagebox, 'showerror',
        lambda *args, **kwargs: errors.append(args),
    )
    editor = profile_editor.EvaluationProfileEditor(
        tk_root, profile_var=tk.StringVar(tk_root, str(original)),
    )
    editor.withdraw()
    try:
        assert editor._load_profile(str(original)), errors
        editor.var_statistics_engine.set('optimized_coco_v1')
        editor.var_quality_workers.set(workers)
        with pytest.raises(ValueError, match=r'quality_gate\.statistics\.workers'):
            editor._build_payload()
        assert editor.var_quality_workers.get() == workers
        assert original.read_bytes() == original_bytes
    finally:
        editor.destroy()
