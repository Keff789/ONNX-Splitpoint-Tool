"""AP0: preserve the actual history separately from v33 robustness changes.

The frozen v30/v32 resolvers are diagnostic fixtures, never production imports.
The three-field historical hash delta is specifically v30 -> audited v32; a
subsequent explicit description migration in v33 does not rewrite that history.
No hardware, compiler, user registry or archived run is touched by these tests.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import types
import zipfile

import pytest
import yaml


FIXTURES = Path(__file__).parent / "fixtures/v27933_force_origin"
ORIGINAL_ZIP = FIXTURES / "force_origin_audit_20260909T095730Z_ni3utrwy.zip"
MODES = ("smoke", "standard", "final")
BACKENDS = ("hailo", "deepx")


def _original_files():
    with zipfile.ZipFile(ORIGINAL_ZIP) as archive:
        assert archive.testzip() is None
        return {name: archive.read(name) for name in archive.namelist()
                if not name.endswith("/")}


def _frozen_module(monkeypatch, name, source):
    module = types.ModuleType(name)
    module.__file__ = f"<original-evidence:{name}>"
    # Compile in memory so the immutable evidence never receives bytecode.
    monkeypatch.setitem(sys.modules, name, module)
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


@pytest.fixture
def evidence(monkeypatch):
    before = {p: p.read_bytes() for p in FIXTURES.rglob("*") if p.is_file()}
    original = _original_files()
    old = _frozen_module(
        monkeypatch, "onnx_splitpoint_tool._original_force_v30",
        (FIXTURES / "run_modes_v30_original.py").read_bytes())
    audited = _frozen_module(
        monkeypatch, "onnx_splitpoint_tool._original_force_v32",
        original["installed_source/onnx_splitpoint_tool/run_modes.py"])
    yield types.SimpleNamespace(
        files=original, old=old, audited=audited,
        raw=yaml.safe_load(original["registries/00_run_modes.yaml"]),
        audit=json.loads(original["FORCE_ORIGIN_AUDIT.json"]),
        source=yaml.safe_load(original["selected_run/profile_source.yaml"]),
        effective=yaml.safe_load(original["selected_run/profile.yaml"]),
    )
    after = {p: p.read_bytes() for p in FIXTURES.rglob("*") if p.is_file()}
    assert after == before, "Original evidence changed during diagnostic replay"


def _force(mode):
    return tuple(mode["build"][backend]["force_build"] for backend in BACKENDS)


def _difference(old, new, prefix=""):
    if isinstance(old, dict) and isinstance(new, dict):
        result = []
        for key in sorted(old.keys() | new.keys()):
            path = f"{prefix}.{key}" if prefix else key
            if key not in old:
                result.append((path, "<absent>", new[key]))
            elif key not in new:
                result.append((path, old[key], "<absent>"))
            else:
                result.extend(_difference(old[key], new[key], path))
        return result
    return [] if old == new else [(prefix, old, new)]


def test_t33_01_original_modules_bind_to_complete_fix2_manifest(evidence):
    manifest = json.loads((FIXTURES / "baseline_source_manifest.json").read_text())
    report = json.loads((FIXTURES / "baseline_audit.json").read_text())
    files = {row["path"]: row for row in manifest["files"]}
    assert len(files) == manifest["file_count"] == 1532
    assert manifest["package_version"] == "2.79.32"
    assert manifest["workflow_version"] == "v2.79.32-native-full-failure-closure"
    assert report["baseline_delivery"].endswith("_FIX2_COMPLETE_DELIVERY_BUNDLE")
    assert report["source_manifest_payloads_verified"] == 1532
    assert len(evidence.audit["installed_sources"]) == 6
    for path, declaration in evidence.audit["installed_sources"].items():
        data = evidence.files[f"installed_source/{path}"]
        assert hashlib.sha256(data).hexdigest() == declaration["sha256"] == files[path]["sha256"]
        assert len(data) == files[path]["size"]
    identity = evidence.files["installed_source/onnx_splitpoint_tool/release_identity.py"].decode()
    assert 'VERSION = "2.79.32"' in identity
    assert 'BUILD_ID = "v2.79.32-native-full-failure-closure"' in identity
    q3 = json.loads((FIXTURES / "Q3_critical_module_sha256.json").read_text())
    mismatches = [path for path, digest in q3.items()
                  if digest != "sha256:" + files[f"onnx_splitpoint_tool/{path}"]["sha256"]]
    assert len(q3) == 171
    assert mismatches == ["resources/templates/benchmark_suite.py.txt"]
    assert report["Q3_matching_FIX2_module_count"] == 170


def test_t33_02_actual_registry_has_six_booleans_and_existing_final_force(evidence):
    assert evidence.raw["schema_version"] == 13
    assert evidence.raw["default_mode"] == "standard"
    for mode in MODES:
        values = _force(evidence.raw["modes"][mode])
        assert all(type(value) is bool for value in values)
        assert values == ((True, True) if mode == "final" else (False, False))
    assert evidence.raw["modes"]["final"]["build"]["hailo"]["cache_integrity"] == "relaxed"
    packaged = yaml.safe_load(evidence.files[
        "installed_source/onnx_splitpoint_tool/resources/run_modes/default_run_modes.yaml"])
    for config in (packaged, evidence.audited.default_run_modes_config()):
        assert all(_force(config["modes"][mode]) == (False, False) for mode in MODES)


def test_t33_03_replay_actual_historical_resolvers_matches_16_profiles(evidence):
    old = evidence.old.validate_run_modes_config(evidence.raw)
    audited = evidence.audited.validate_run_modes_config(evidence.raw)
    old_hash = evidence.old._json_hash(old)
    audited_hash = evidence.audited._json_hash(audited)
    assert old_hash == "15919a1020541b0f2e268bc2fe7514df6bb1e32fb73937963b26ef347299ec60"
    assert audited_hash == "a99fefe81a2cd9734ed8ef2bcfaa3799a019e3f3c0d1bc1fb1720eb2e7db31bc"
    history = [row["summary"]["execution_preset"]
               for row in evidence.audit["historical_run_profiles"]]
    assert len(history) == 16
    assert all(row["config_sha256"] == old_hash for row in history)
    assert min(row["resolved_at"] for row in history) == "2026-09-03T21:42:02+02:00"
    assert evidence.audit["selected_run_profile"]["execution_preset"]["config_sha256"] == audited_hash
    assert evidence.effective["execution_preset"]["config_sha256"] == audited_hash
    assert sorted(_difference(old, audited)) == sorted([
        (f"modes.{mode}.build.deepx.classification_preprocessing", "<absent>", "imagenet_mean_std")
        for mode in MODES
    ])
    assert all(_force(old["modes"][mode]) == _force(audited["modes"][mode]) for mode in MODES)


def test_t33_04_original_copies_match_collected_bytes_and_have_no_models(evidence):
    checked = 0
    for row in evidence.audit["read_inventory"]:
        if row.get("copied_to"):
            data = evidence.files[row["copied_to"]]
            assert hashlib.sha256(data).hexdigest() == row["sha256"]
            assert len(data) == row["size_bytes"]
            assert row["stable_read"] is True
            checked += 1
    assert checked == 9
    assert not any(Path(name).suffix.lower() in {
        ".onnx", ".hef", ".dxnn", ".jpg", ".jpeg", ".png", ".npy", ".npz"
    } for name in evidence.files)
    assert evidence.audit["diagnostic_only"] is True
    assert evidence.audit["modifies_tool_or_config"] is False
    # This assertion preserves the evidence limit: none of the sixteen old
    # profile files is claimed to be present merely because its fields exist.
    assert len(evidence.audit["historical_run_profiles"]) == 16
    assert not any(name.startswith("historical_run_profiles/") for name in evidence.files)


def test_original_mode_selection_exposes_existing_values_without_new_writer(evidence):
    before = copy.deepcopy(evidence.raw)
    for mode in MODES:
        effective, _ = evidence.audited.apply_run_mode(
            evidence.source, mode_id=mode, config=evidence.raw,
            config_path="/diagnostic-read-only/central-run-modes.yaml", follow_tool_config=True)
        assert (effective["hailo_build"]["force_build"], effective["deepx_build"]["force_build"]) == (
            (True, True) if mode == "final" else (False, False))
        assert effective["deepx_build"]["classification_preprocessing"] == "current_scale_only"
        assert effective["hailo_build"]["cache_integrity"] == "relaxed"
    assert evidence.raw == before


def test_synthetic_legacy_string_reproducer_is_separate_from_real_history(evidence):
    synthetic = copy.deepcopy(evidence.raw)
    for backend in BACKENDS:
        synthetic["modes"]["final"]["build"][backend]["force_build"] = "false"
    effective, _ = evidence.audited.apply_run_mode(
        evidence.source, mode_id="final", config=synthetic,
        config_path="/synthetic-negative-control/run-modes.yaml", follow_tool_config=True)
    # This intentionally reproduces the OLD resolver defect, not v33 behavior.
    assert effective["hailo_build"]["force_build"] is True
    assert effective["deepx_build"]["force_build"] is True
    assert _force(evidence.raw["modes"]["final"]) == (True, True)
    assert all(type(value) is bool for mode in MODES for value in _force(evidence.raw["modes"][mode]))
    replay = json.loads((FIXTURES / "REPLAY_FORCE_ORIGIN.json").read_text())
    by_name = {check["name"]: check for check in replay["checks"]}
    assert by_name["synthetic_string_false_bug_reproduced_not_historical_cause"]["detail"]["synthetic_only"] is True
    assert by_name["installed_stale_panel_bug_reproduced_no_historical_event_proof"]["detail"]["synthetic_only"] is True
    assert "no proof of initial writer" in replay["conclusion"]
