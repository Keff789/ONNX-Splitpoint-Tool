"""Constructed boundary/concurrency cases, distinct from original Force evidence."""
from __future__ import annotations

import copy
import multiprocessing
from pathlib import Path

import pytest
import yaml

from onnx_splitpoint_tool.config_values import parse_config_bool, validate_profile_config_booleans
from onnx_splitpoint_tool import run_modes as rm


@pytest.mark.parametrize("value", [False, True])
def test_real_bool_save_reload_and_materialization(tmp_path, value):
    config = rm.default_run_modes_config()
    for mode in config["modes"].values():
        for backend in ("hailo", "deepx"):
            mode["build"][backend]["force_build"] = value
    path = tmp_path / "registry.yaml"
    rm.save_run_modes_config(path, config, expected_revision="")
    loaded = rm.load_run_modes_config(path)
    assert loaded == config
    for mode in config["modes"]:
        resolved, _ = rm.apply_run_mode({"execution_preset": {"id": mode}}, config=loaded, config_path=path)
        assert resolved["hailo_build"]["force_build"] is value
        assert resolved["deepx_build"]["force_build"] is value


def test_missing_force_is_false_for_each_mode():
    config = rm.default_run_modes_config()
    for mid, mode in config["modes"].items():
        for backend in ("hailo", "deepx"):
            mode["build"][backend].pop("force_build")
        result, _ = rm.apply_run_mode({}, mode_id=mid, config=config)
        assert result["hailo_build"]["force_build"] is False
        assert result["deepx_build"]["force_build"] is False


@pytest.mark.parametrize("value", ["false", "true", "", "yes", "unknown", None, 0, 1, [], [False], {}])
@pytest.mark.parametrize("backend", ["hailo", "deepx"])
def test_invalid_force_rejected_before_merge_or_materialization(value, backend):
    field = f"modes.final.build.{backend}.force_build"
    config = rm.default_run_modes_config()
    config["modes"]["final"]["build"][backend]["force_build"] = value
    with pytest.raises(ValueError, match=f"config_boolean_invalid:{field}"):
        rm.validate_run_modes_config(config)
    with pytest.raises(ValueError, match=f"config_boolean_invalid:{field}"):
        rm.apply_run_mode({}, mode_id="final", config=config)
    with pytest.raises(ValueError, match=f"config_boolean_invalid:{field}"):
        rm._materialized_blocks("final", config["modes"]["final"], profile_name="synthetic")


def test_explicit_legacy_parser_opt_in_is_separate_from_normal_load():
    assert parse_config_bool(" false ", field="synthetic", allow_legacy_text=True) is False
    assert parse_config_bool(" TRUE ", field="synthetic", allow_legacy_text=True) is True
    with pytest.raises(ValueError, match="config_boolean_invalid:synthetic"):
        parse_config_bool("false", field="synthetic")


@pytest.mark.parametrize("block", ["hailo_build", "deepx_build"])
@pytest.mark.parametrize("value", ["false", "true", None, 0, [], "invalid"])
def test_profile_force_rejected_even_when_registry_would_override(block, value):
    profile = {block: {"force_build": value}}
    with pytest.raises(ValueError, match=f"config_boolean_invalid:{block}.force_build"):
        validate_profile_config_booleans(profile)
    with pytest.raises(ValueError, match=f"config_boolean_invalid:{block}.force_build"):
        rm.apply_run_mode(profile, config=rm.default_run_modes_config())


@pytest.mark.parametrize("fragment, field", [
    ({"modes": None}, "modes"),
    ({"modes": {"final": []}}, "modes.final"),
    ({"modes": {"final": {"build": None}}}, "modes.final.build"),
    ({"modes": {"final": {"build": {"hailo": []}}}}, "modes.final.build.hailo"),
])
def test_structures_rejected_before_defaults(fragment, field):
    with pytest.raises(ValueError, match=f"config_mapping_invalid:{field}"):
        rm.validate_run_modes_config(fragment)


def test_stale_true_snapshot_cannot_erase_new_false_and_extensions(tmp_path):
    path = tmp_path / "registry.yaml"
    config = rm.default_run_modes_config()
    config["modes"]["final"]["build"]["hailo"]["force_build"] = True
    rm.save_run_modes_config(path, config)
    stale = rm.load_run_modes_config(path)
    revision = rm.run_modes_revision(stale)
    current = copy.deepcopy(stale)
    current["modes"]["final"]["build"]["hailo"]["force_build"] = False
    current["modes"]["individual_night"] = {"label": "Custom", "profile_ref": "user/night.yaml", "extension": {"leave": [1, 2]}}
    current["custom_root"] = {"retained": True}
    rm.save_run_modes_config(path, current, expected_revision=revision)
    before = path.read_bytes()
    with pytest.raises(rm.RunModesConflictError) as error:
        rm.save_run_modes_config(path, stale, expected_revision=revision, baseline=stale)
    assert "modes.final.build.hailo.force_build" in error.value.changed_fields
    assert "Neu laden" in str(error.value)
    assert path.read_bytes() == before
    assert rm.load_run_modes_config(path) == current


def _concurrent_writer(path, barrier, output, backend):
    try:
        loaded = rm.load_run_modes_config(path)
        revision = rm.run_modes_revision(loaded)
        baseline = copy.deepcopy(loaded)
        loaded["modes"]["final"]["build"][backend]["force_build"] = True
        barrier.wait(timeout=20)
        rm.save_run_modes_config(path, loaded, expected_revision=revision, baseline=baseline)
        output.put((backend, "saved"))
    except rm.RunModesConflictError:
        output.put((backend, "conflict"))


def test_two_processes_compare_revision_inside_same_lock(tmp_path):
    path = tmp_path / "registry.yaml"
    rm.save_run_modes_config(path, rm.default_run_modes_config())
    context = multiprocessing.get_context("spawn")
    barrier, output = context.Barrier(2), context.Queue()
    children = [context.Process(target=_concurrent_writer, args=(path, barrier, output, backend)) for backend in ("hailo", "deepx")]
    try:
        for child in children:
            child.start()
        outcomes = [output.get(timeout=30) for _ in children]
        for child in children:
            child.join(timeout=10)
            assert child.exitcode == 0
        assert sorted(result for _, result in outcomes) == ["conflict", "saved"]
        winner = next(backend for backend, status in outcomes if status == "saved")
        final = rm.load_run_modes_config(path)["modes"]["final"]["build"]
        assert final[winner]["force_build"] is True
        assert sum(final[backend]["force_build"] for backend in ("hailo", "deepx")) == 1
    finally:
        for child in children:
            if child.is_alive():
                child.terminate()
                child.join(timeout=5)
        output.close()


def test_interrupted_atomic_replace_keeps_original_yaml(tmp_path, monkeypatch):
    path = tmp_path / "registry.yaml"
    config = rm.default_run_modes_config()
    rm.save_run_modes_config(path, config)
    original = path.read_bytes()
    revision = rm.run_modes_revision(config)
    config["modes"]["final"]["build"]["deepx"]["force_build"] = True
    def interrupted(*args):
        raise OSError("synthetic interruption before replace")
    monkeypatch.setattr(rm.os, "replace", interrupted)
    with pytest.raises(OSError, match="synthetic interruption"):
        rm.save_run_modes_config(path, config, expected_revision=revision)
    assert path.read_bytes() == original
    assert yaml.safe_load(original)["modes"]["final"]["build"]["deepx"]["force_build"] is False
    assert not list(tmp_path.glob("*.tmp"))


def test_summary_reports_effective_central_override_and_explicit_profile_source(tmp_path):
    config = rm.default_run_modes_config()
    config["default_mode"] = "smoke"
    config["modes"]["final"]["build"]["hailo"]["force_build"] = True
    profile = {"execution_preset": {"id": "final"}, "deepx_build": {"classification_preprocessing": "current_scale_only"}}
    result, audit = rm.apply_run_mode(profile, config=config, config_path=tmp_path / "actual.yaml")
    summary = rm.profile_build_summary(result)
    assert "final" in summary["values_source"] and "actual.yaml" in summary["values_source"]
    assert summary["hailo_force_build"] is True
    assert summary["deepx_force_build"] is False
    assert "bewusst übergangen" in summary["hailo_force_text"]
    assert summary["hailo_cache_integrity"] == "relaxed"
    assert summary["deepx_classification_preprocessing"] == "current_scale_only"
    assert summary["deepx_classification_source"] == "Evaluationsprofil"
    assert audit["build_summary"] == summary
    assert "Hailo Force: AN" in rm.run_mode_profile_brief(result)


def test_frozen_snapshot_does_not_read_current_registry_or_rewrite_source(tmp_path):
    current = tmp_path / "registry.yaml"
    current.write_text("invalid: [unterminated")
    snapshot = rm.default_run_modes_config()["modes"]["final"]
    snapshot["build"]["deepx"]["force_build"] = True
    profile = {"execution_preset": {"id": "final", "follow_tool_config": False, "snapshot": snapshot, "config_path": str(current), "config_sha256": "historical-value"}}
    before = copy.deepcopy(profile)
    result, audit = rm.apply_run_mode(profile)
    assert profile == before
    assert current.read_text() == "invalid: [unterminated"
    assert result["execution_preset"]["snapshot"] == snapshot
    assert audit["config_sha256"] == "historical-value"
    assert result["deepx_build"]["force_build"] is True
    assert "Profilsnapshot" in rm.profile_build_summary(result)["values_source"]


def test_frozen_profile_missing_snapshot_does_not_fall_back_to_registry():
    with pytest.raises(ValueError, match="run_mode_snapshot_missing"):
        rm.apply_run_mode({"execution_preset": {"follow_tool_config": False}})


@pytest.mark.parametrize("description", sorted(rm._LEGACY_FINAL_DESCRIPTIONS) + ["My own strict research description"])
def test_exact_old_description_only_migration_preserves_force_and_registry_bytes(tmp_path, description):
    config = rm.default_run_modes_config()
    final = config["modes"]["final"]
    final["description"] = description
    final["build"]["hailo"]["force_build"] = True
    final["build"]["deepx"]["force_build"] = True
    path = tmp_path / "registry.yaml"
    path.write_text(yaml.safe_dump(config))
    original = path.read_bytes()
    loaded = rm.load_run_modes_config(path)
    expected = rm.default_run_modes_config()["modes"]["final"]["description"] if description in rm._LEGACY_FINAL_DESCRIPTIONS else description
    assert loaded["modes"]["final"]["description"] == expected
    assert loaded["modes"]["final"]["build"]["hailo"]["force_build"] is True
    assert loaded["modes"]["final"]["build"]["deepx"]["force_build"] is True
    assert path.read_bytes() == original


def test_default_preprocessing_source_survives_rematerialization(tmp_path):
    config = rm.default_run_modes_config()
    first, _ = rm.apply_run_mode({"execution_preset": {"id": "standard"}}, config=config, config_path=tmp_path / "registry.yaml")
    second, _ = rm.apply_run_mode(first, config=config)
    for result in (first, second):
        summary = rm.profile_build_summary(result)
        assert summary["deepx_classification_preprocessing"] == "imagenet_mean_std"
        assert summary["deepx_classification_source"] == summary["values_source"]
    config["modes"]["standard"]["build"]["deepx"]["classification_preprocessing"] = "current_scale_only"
    changed, _ = rm.apply_run_mode(second, config=config)
    assert changed["deepx_build"]["classification_preprocessing"] == "current_scale_only"
    assert rm.profile_build_summary(changed)["deepx_classification_source"] == rm.profile_build_summary(changed)["values_source"]
