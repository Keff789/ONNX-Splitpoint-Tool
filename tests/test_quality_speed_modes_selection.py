"""Normal mode selection and actual spawn-boundary validation; no workflow start."""
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from onnx_splitpoint_tool import quality_service, run_modes
from onnx_splitpoint_tool.benchmark.evaluation_profiles import load_evaluation_profile, save_evaluation_profile_yaml


@pytest.fixture
def editor(tmp_path, monkeypatch, request):
    import tkinter as tk
    from onnx_splitpoint_tool.gui import profile_editor
    config = run_modes.default_run_modes_config()
    config["default_mode"] = getattr(request, "param", "standard")
    config["modes"]["final"]["quality"].update(
        statistics_engine="optimized_coco_v1", statistics_block_repetitions=128,
        statistics_checkpoint_blocks=True, statistics_prepared_cache_limit_mib=64,
        reference_intra_op_threads=7, workers=3)
    run_modes.save_run_modes_config(run_modes.default_run_modes_path(), config)
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"withdrawn Tk display unavailable: {exc}")
    root.withdraw()
    errors = []
    monkeypatch.setattr(profile_editor.messagebox, "showerror", lambda *args, **kwargs: errors.append(args))
    instance = profile_editor.EvaluationProfileEditor(root)
    instance.withdraw()
    try:
        # Supply ordinary fixture models to make the normal Save/Load snapshot
        # path valid; no model file is loaded and no runner is started.
        source = Path(__file__).resolve().parents[1] / "onnx_splitpoint_tool/resources/evaluation_profiles/smoke_regression_v1.yaml"
        fixture = yaml.safe_load(source.read_text())
        instance.models = deepcopy(fixture["model_suite"]["primary"])
        instance.var_auto_bind_dataset_registry.set(False)
        yield instance, config, errors
    finally:
        instance.destroy()
        root.destroy()


@pytest.mark.parametrize("editor", ["final"], indirect=True)
def test_new_profile_initial_default_mode_has_matching_statistics_controls(editor):
    instance, _config, errors = editor
    # Opening a new profile with Final already selected must not require a
    # redundant mode toggle before its configured statistics become effective.
    assert instance.var_run_mode_id.get() == "final"
    assert instance.var_statistics_engine.get() == "optimized_coco_v1", errors
    assert instance.var_statistics_block.get() == 128
    assert instance.var_statistics_checkpoint.get() is True
    assert instance.var_statistics_cache_mib.get() == 64
    assert instance.var_reference_threads.get() == "7"
    assert instance.var_quality_workers.get() == 3


def test_new_profile_selects_configured_statistics_and_freezes_same_effective_values(editor, tmp_path):
    instance, config, errors = editor
    instance.var_run_mode_id.set("final")  # normal registered Tk variable callback
    expected = config["modes"]["final"]["quality"]
    assert instance.var_statistics_engine.get() == expected["statistics_engine"], errors
    assert instance.var_statistics_block.get() == expected["statistics_block_repetitions"]
    assert instance.var_statistics_checkpoint.get() is True
    assert instance.var_statistics_cache_mib.get() == 64
    assert instance.var_reference_threads.get() == "7"
    assert instance.var_quality_workers.get() == 3
    payload = instance._build_payload()
    path = tmp_path / "selected_mode.yaml"
    save_evaluation_profile_yaml(path, payload)
    loaded = load_evaluation_profile(path)
    actual = loaded.start_snapshot["resolved_profile"]["quality_gate"]
    assert actual["statistics"]["engine"] == "optimized_coco_v1"
    assert actual["statistics"]["block_repetitions"] == 128
    assert actual["statistics"]["checkpoint_blocks"] is True
    assert actual["statistics"]["prepared_cache_limit_mib"] == 64
    assert actual["statistics"]["workers"] == 3
    assert actual["management_reference"]["intra_op_threads"] == 7
    assert actual["statistics"]["bootstrap_repetitions"] == 5000


def test_explicit_widget_edits_survive_mode_selection(editor):
    instance, _config, errors = editor
    instance.var_run_mode_id.set("standard")
    instance.var_statistics_engine.set("optimized_coco_v1")
    instance.var_statistics_block.set(512)
    instance.var_statistics_checkpoint.set(True)
    instance.var_statistics_cache_mib.set(96)
    instance.var_reference_threads.set("6")
    instance.var_run_mode_id.set("final")
    assert instance.var_statistics_engine.get() == "optimized_coco_v1", errors
    assert instance.var_statistics_block.get() == 512
    assert instance.var_statistics_checkpoint.get() is True
    assert instance.var_statistics_cache_mib.get() == 96
    assert instance.var_reference_threads.get() == "6"
    payload = instance._build_payload()["quality_gate"]
    assert payload["statistics"]["block_repetitions"] == 512
    assert payload["statistics"]["prepared_cache_limit_mib"] == 96
    assert payload["management_reference"]["intra_op_threads"] == 6


def test_untouched_widgets_follow_subsequent_mode_selection(editor):
    instance, config, errors = editor
    instance.var_run_mode_id.set("final")
    instance.var_run_mode_id.set("standard")
    expected = config["modes"]["standard"]["quality"]
    assert instance.var_statistics_engine.get() == expected["statistics_engine"], errors
    assert instance.var_statistics_block.get() == expected["statistics_block_repetitions"]
    assert instance.var_statistics_checkpoint.get() == expected["statistics_checkpoint_blocks"]
    assert instance.var_statistics_cache_mib.get() == expected["statistics_prepared_cache_limit_mib"]
    assert instance.var_reference_threads.get() == ""


@pytest.mark.parametrize("workers", [True, False, 1.9, "2", None, 0, -1, 65, 1_000_000])
def test_direct_optimized_service_rejects_invalid_workers_before_cache_or_spawn(tmp_path, monkeypatch, workers):
    calls = []
    def forbidden_pool(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("invalid workers reached the process pool")
    monkeypatch.setattr(quality_service, "ProcessPoolExecutor", forbidden_pool)
    cache = tmp_path / "must_not_exist"
    with pytest.raises(ValueError, match="optimized statistics workers must be an integer"):
        quality_service.ManagementQualityService(cache, workers=workers,
            statistics={"engine": "optimized_coco_v1"})
    assert calls == []
    assert not cache.exists()
