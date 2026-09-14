from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path):
    name = "v27921_script_" + path.stem + "_" + path.parent.name
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("directory", [
    "scripts", "onnx_splitpoint_tool/resources/remote_scripts",
])
@pytest.mark.parametrize("script", [
    "native_full_baseline_eval_runner.py",
    "smoke_hailo10_full_from_benchmarkset.py",
    "native_hailo_trt_fifo_from_benchmarkset.py",
])
def test_runtime_discovery_skips_backup_and_follows_active_alias(tmp_path, directory, script):
    module = _load(ROOT / directory / script)
    is_split = script == "native_hailo_trt_fifo_from_benchmarkset.py"
    leaf = tmp_path / "b398" if is_split else tmp_path
    leaf = leaf / "hailo" / "hailo10h" / ("part1" if is_split else "full")
    stale = leaf / ".hailo-generations" / "000-old" / "compiled.hef"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"old generation")

    def find():
        if is_split:
            return module._find_hef(tmp_path, "b398", "hailo10h")
        if script == "native_full_baseline_eval_runner.py":
            return module._find_hailo_full_hef(tmp_path, "hailo10h")
        return module.find_hef(tmp_path, "hailo10h")

    if script == "native_full_baseline_eval_runner.py":
        assert find() is None
    else:
        with pytest.raises(FileNotFoundError):
            find()

    active = leaf / ".hailo-generations" / "999-current" / "compiled.hef"
    active.parent.mkdir(parents=True)
    active.write_bytes(b"current generation")
    (leaf / "compiled.hef").symlink_to(active.relative_to(leaf))
    assert find() == active.resolve()


def test_modified_remote_helpers_are_exact_mirrors():
    for name in (
        "native_full_baseline_eval_runner.py",
        "smoke_hailo10_full_from_benchmarkset.py",
        "native_hailo_trt_fifo_from_benchmarkset.py",
    ):
        assert (ROOT / "scripts" / name).read_bytes() == (
            ROOT / "onnx_splitpoint_tool/resources/remote_scripts" / name
        ).read_bytes()
