"""Exact metadata inventory and explicit collector registry selection; no devices."""
from __future__ import annotations

import ast
from pathlib import Path
import re

import pytest
import yaml

from onnx_splitpoint_tool.energy.config import EnergyDefaults
from onnx_splitpoint_tool.release_identity import BUILD_ID, VERSION
from onnx_splitpoint_tool import source_integrity
from scripts import build_source_manifest
from scripts import energy_measurement_cli as cli


METADATA = (".gitignore", "AGENTS.md", "LICENSE")


def release_tree(tmp_path):
    root = tmp_path / "release"
    files = {
        "pyproject.toml": f'[project]\nversion = "{VERSION}"\n',
        "onnx_splitpoint_tool/release_identity.py":
            f'VERSION = "{VERSION}"\nBUILD_ID = "{BUILD_ID}"\n',
        "onnx_splitpoint_tool/runtime.py": "value = 1\n",
    }
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    return root


def verify_both(root):
    return [build_source_manifest.verify(root, scope="installed"),
            source_integrity.verify_installed_source_integrity(root)]


def test_exact_metadata_are_indexed_and_verified(tmp_path):
    root = release_tree(tmp_path)
    for name in METADATA:
        (root / name).write_text(f"metadata {name}\n")
    manifest = build_source_manifest.build(root)
    assert set(METADATA) <= {row["path"] for row in manifest["files"]}
    assert all(result["ok"] for result in verify_both(root))


@pytest.mark.parametrize("name", METADATA)
@pytest.mark.parametrize("mutation", ("changed", "missing", "symlink", "unindexed"))
def test_metadata_cannot_bypass_integrity(tmp_path, name, mutation):
    root = release_tree(tmp_path)
    path = root / name
    if mutation != "unindexed":
        path.write_text("original metadata\n")
    build_source_manifest.build(root)
    if mutation == "changed":
        path.write_text("changed metadata\n")
    elif mutation == "missing":
        path.unlink()
    elif mutation == "symlink":
        target = tmp_path / "same-content"
        target.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(target)
    else:
        path.write_text("unindexed metadata\n")
    assert all(not result["ok"] for result in verify_both(root))


@pytest.mark.parametrize("name", ("unknown.txt", "agents.md", ".Gitignore", "LICENSE.extra", "nested/AGENTS.md"))
def test_metadata_fix_does_not_allow_unknown_extras(tmp_path, name):
    root = release_tree(tmp_path)
    build_source_manifest.build(root)
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("extra\n")
    assert all(not result["ok"] for result in verify_both(root))


def test_real_source_change_still_fails(tmp_path):
    root = release_tree(tmp_path)
    build_source_manifest.build(root)
    (root / "onnx_splitpoint_tool/runtime.py").write_text("value = 2\n")
    assert all(not result["ok"] for result in verify_both(root))


def test_updater_and_verifiers_share_exact_metadata_policy():
    updater = (Path(__file__).parents[1] / "scripts/update_source_release.sh").read_text()
    match = re.search(r"^ALLOWED_ROOT_FILES = (\{.*?^\})", updater, re.M | re.S)
    assert match
    updater_allowed = ast.literal_eval(match.group(1))
    assert updater_allowed == build_source_manifest.ALLOWED_ROOT_FILES == source_integrity._ALLOWED_ROOT_FILES
    assert set(METADATA) <= updater_allowed


def test_explicit_legacy_registry_selects_its_collector(tmp_path, monkeypatch):
    registry = tmp_path / "hardware_setups.yaml"
    candidate = str(tmp_path / "collector_candidate/urecs-data-collector")
    registry.write_text(yaml.safe_dump({
        "schema": "onnx-splitpoint/hardware-setups", "schema_version": 2,
        "energy_defaults": {"collector_binary": candidate, "pre_duration_s": 5.,
                            "post_duration_s": 5., "sample_rate": 2000, "channel": 0},
        "hardware_setups": [{"id": "local-test", "accelerator": "test",
                             "energy": {"enabled": True, "urecs_address": "127.0.0.1"}}],
    }))
    monkeypatch.setattr(cli, "load_energy_defaults", lambda: EnergyDefaults(collector_binary="stale-collector"))
    defaults, setup = cli._measurement_context("local-test", registry)
    assert defaults.collector_binary == candidate
    assert (defaults.pre_duration_s, defaults.post_duration_s, defaults.sample_rate, defaults.channel) == (5., 5., 2000, 0)
    assert setup.urecs_address == "127.0.0.1"
    assert not setup.calibration_manifest


def test_implicit_legacy_registry_keeps_standalone_defaults(monkeypatch):
    from types import SimpleNamespace
    setup = SimpleNamespace(calibration_manifest="", calibration_sha256="", expected_channel_bindings=())
    standalone = EnergyDefaults(collector_binary="legacy-collector")
    monkeypatch.setattr(cli, "load_hardware_registry", lambda: {})
    monkeypatch.setattr(cli, "energy_setup_from_registry", lambda *args, **kwargs: setup)
    monkeypatch.setattr(cli, "get_setup_energy", lambda setup_id: setup)
    monkeypatch.setattr(cli, "load_energy_defaults", lambda: standalone)
    assert cli._measurement_context("local-test") == (standalone, setup)
