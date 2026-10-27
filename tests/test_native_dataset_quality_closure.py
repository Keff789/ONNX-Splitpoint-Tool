"""Cold-stage the published Native Quality inventory, without hardware."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from onnx_splitpoint_tool.remote_runtime_closure import (
    NATIVE_DATASET_QUALITY_PACKAGE_CLOSURE,
    NATIVE_DATASET_QUALITY_RESOURCE_CLOSURE,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def cold_stage(tmp_path: Path) -> tuple[Path, Path, set[str]]:
    stage = tmp_path / "stage"
    cwd = tmp_path / "independent_cwd"
    cwd.mkdir()
    inventory = {relative for relative, _, _ in NATIVE_DATASET_QUALITY_PACKAGE_CLOSURE}
    inventory.update(relative for relative, _ in NATIVE_DATASET_QUALITY_RESOURCE_CLOSURE)
    for relative in inventory:
        source, target = ROOT / relative, stage / relative
        assert source.is_file(), relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    assert {path.relative_to(stage).as_posix() for path in stage.rglob("*") if path.is_file()} == inventory
    return stage, cwd, inventory


def _probe(stage: Path, cwd: Path, *, first: str) -> subprocess.CompletedProcess[str]:
    code = r'''
import hashlib, importlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root))
assert sys.flags.isolated and sys.dont_write_bytecode
assert not pathlib.Path.cwd().is_relative_to(root)
if sys.argv[2] == "h10":
    h10 = importlib.import_module("onnx_splitpoint_tool.native_dataset_quality_h10")._load_native_runtime()
    h8 = importlib.import_module("onnx_splitpoint_tool.native_dataset_quality_h8")._load_native_runtime()
else:
    h8 = importlib.import_module("onnx_splitpoint_tool.native_dataset_quality_h8")._load_native_runtime()
    h10 = importlib.import_module("onnx_splitpoint_tool.native_dataset_quality_h10")._load_native_runtime()
capture = importlib.import_module("onnx_splitpoint_tool.native_dataset_quality")
assert capture.IMAGE_COUNT == 5000
assert h10.NativeTRT is not None and h8.NativeTRT is not None
proofs = {}
for name, module in sorted(sys.modules.items()):
    relevant = (name == "onnx_splitpoint_tool" or name.startswith("onnx_splitpoint_tool.")
                or name.startswith("scripts.native_") or name == "native_hailo10_trt_e2e_from_benchmarkset")
    if not relevant or not getattr(module, "__file__", None):
        continue
    path = pathlib.Path(module.__file__).resolve()
    if not path.is_relative_to(root):
        raise RuntimeError("unbound_native_quality_import:" + name + ":" + str(path))
    proofs[name] = {"relative_path": path.relative_to(root).as_posix(),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
assert not {"tensorrt", "hailo_platform", "cuda"}.intersection(sys.modules)
print(json.dumps(proofs, sort_keys=True))
'''
    environment = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE="1")
    return subprocess.run([sys.executable, "-I", "-B", "-c", code, str(stage), first],
        cwd=cwd, env=environment, capture_output=True, text=True, timeout=30)


@pytest.mark.parametrize("first", ["h10", "h8"])
def test_exact_cold_stage_imports_both_actual_native_paths(cold_stage: tuple[Path, Path, set[str]], first: str) -> None:
    stage, cwd, inventory = cold_stage
    result = _probe(stage, cwd, first=first)
    assert result.returncode == 0, result.stderr
    proofs = json.loads(result.stdout)
    assert {"onnx_splitpoint_tool.native_dataset_quality", "onnx_splitpoint_tool.native_dataset_quality_export",
            "onnx_splitpoint_tool.quality_service", "onnx_splitpoint_tool.quality_cache",
            "onnx_splitpoint_tool.native_dataset_quality_h10", "onnx_splitpoint_tool.native_dataset_quality_h8",
            "scripts.native_hailo10_trt_e2e_from_benchmarkset", "native_hailo10_trt_e2e_from_benchmarkset",
            "onnx_splitpoint_tool._native_dataset_quality_h8_existing"} <= set(proofs)
    for name, proof in proofs.items():
        assert proof["relative_path"] in inventory, name
        assert proof["sha256"] == hashlib.sha256((ROOT / proof["relative_path"]).read_bytes()).hexdigest(), name
    assert not list(stage.rglob("__pycache__"))


def test_cold_stage_cli_help_checks_dependencies_without_hardware(cold_stage: tuple[Path, Path, set[str]]) -> None:
    stage, cwd, _ = cold_stage
    result = subprocess.run([sys.executable, "-I", "-B", str(stage / "scripts/native_dataset_quality.py"), "--help"],
        cwd=cwd, env=dict(os.environ, PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE="1"),
        capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "--config" in result.stdout and "--out" in result.stdout
    assert not list(stage.rglob("__pycache__"))


def test_missing_cold_stage_dependency_cannot_fall_back_to_installed_source(cold_stage: tuple[Path, Path, set[str]]) -> None:
    stage, cwd, _ = cold_stage
    (stage / "onnx_splitpoint_tool/quality_cache.py").unlink()
    result = _probe(stage, cwd, first="h10")
    assert result.returncode != 0
    assert "onnx_splitpoint_tool.quality_cache" in result.stderr
    assert "ModuleNotFoundError" in result.stderr
