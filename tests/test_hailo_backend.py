from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from onnx_splitpoint_tool.runners._types import GraphPlan, RunCfg, SampleCfg, StagePlan
from onnx_splitpoint_tool.runners.backends.hailo_backend import (
    HailoBackend,
    HailoBackendUnavailable,
)
from onnx_splitpoint_tool.runners.graph_runner import GraphRunner
from onnx_splitpoint_tool.runners.harness.base import Harness


class _DummyHarness(Harness):
    name = "dummy"

    def __init__(self, inputs: dict[str, np.ndarray]):
        self._inputs = {k: np.asarray(v) for k, v in inputs.items()}

    def make_inputs(self, sample_cfg: SampleCfg) -> dict[str, np.ndarray]:
        # Ignore sample_cfg for unit tests.
        return {k: v.copy() for k, v in self._inputs.items()}

    def postprocess(self, outputs: dict[str, np.ndarray], context: dict) -> dict:
        # Record last outputs for assertions.
        self.last_outputs = {k: np.asarray(v) for k, v in outputs.items()}
        self.last_context = dict(context)
        return {}

    def accuracy_proxy(self, ref: dict[str, np.ndarray], out: dict[str, np.ndarray]) -> dict:
        return {}


class _FakeRuntime:
    """Deterministic runtime stub used for unit tests."""

    def __init__(self, hef_path: Path, opts: dict):
        self.hef_path = Path(hef_path)
        self.opts = dict(opts)
        self.input_names = self.opts.get("input_names", ["x"])
        self.output_names = self.opts.get("output_names", ["y"])
        self.closed = False

    def infer(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        x = np.asarray(inputs[self.input_names[0]])
        # Simple, deterministic "model": y = x + 1
        return {self.output_names[0]: x + 1}

    def close(self) -> None:
        self.closed = True


def _fake_compile(onnx_path: Path, hef_path: Path, hw_arch: str, input_shapes=None) -> None:
    hef_path = Path(hef_path)
    hef_path.parent.mkdir(parents=True, exist_ok=True)
    hef_path.write_bytes(b"FAKE_HEF")


def _patch_hailo_deps_present(monkeypatch: pytest.MonkeyPatch):
    """Pretend hailo_platform + hailo_sdk_client are importable."""

    real_find_spec = importlib.util.find_spec

    class _Spec:  # minimal stand-in for ModuleSpec
        pass

    def fake_find_spec(name: str, *args, **kwargs):
        if name in {"hailo_platform", "hailo_sdk_client"}:
            return _Spec()
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)


def _compute_cache_key(*, model_path: str, opts: dict, tool_version: str) -> str:
    # Mirrors the backend's deterministic key logic.
    hw_arch = str(opts.get("hailo_hw_arch", "hailo"))
    input_shapes = opts.get("hailo_input_shapes")
    compile_opts = {
        "hw_arch": hw_arch,
        "input_shapes": input_shapes,
        "extra": {k: v for k, v in opts.items() if k.startswith("hailo_")},
    }

    # Test environment defaults: no Hailo packages installed.
    p = Path(model_path)
    model_sha = hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else ""
    payload = {
        "model_path": model_path,
        "model_sha256": model_sha,
        "compile_opts": compile_opts,
        "tool_version": tool_version,
        "use_wsl": bool(opts.get("hailo_use_wsl", False)),
        "wsl_distro": opts.get("hailo_wsl_distro") or opts.get("wsl_distro") or None,
        "dfc_version": "",
        "hailo_runtime_version": "",
    }
    s = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def test_hailo_is_available_false_when_missing_dependencies(monkeypatch: pytest.MonkeyPatch):
    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name: str, *args, **kwargs):
        if name in {"hailo_platform", "hailo_sdk_client"}:
            return None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)

    b = HailoBackend(require_device=False)
    assert b.is_available() is False


def test_prepare_uses_cache_if_hef_exists(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _patch_hailo_deps_present(monkeypatch)

    artifacts_dir = tmp_path / "art"
    artifacts_dir.mkdir()

    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"dummy")

    opts = {"hailo_hw_arch": "hailo8"}
    tool_version = "test"
    cache_key = _compute_cache_key(model_path=str(onnx_path), opts=opts, tool_version=tool_version)
    expected_hef = (artifacts_dir / f"{onnx_path.stem}_{cache_key}.hef")
    expected_hef.write_bytes(b"PREBUILT")

    calls = {"n": 0}

    def compile_should_not_run(*args, **kwargs):
        calls["n"] += 1
        raise AssertionError("compile_fn should not have been called")

    backend = HailoBackend(
        require_device=False,
        tool_version=tool_version,
        compile_fn=compile_should_not_run,
        runtime_factory=lambda hef, o: _FakeRuntime(hef, {**o, "input_names": ["x"], "output_names": ["y"]}),
    )

    run_cfg = RunCfg(model_path=str(onnx_path), options=opts)
    handle = backend.prepare(run_cfg, artifacts_dir)

    assert calls["n"] == 0
    assert "x" in handle.input_names
    assert "y" in handle.output_names


def test_prepare_compiles_and_returns_preparedhandle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _patch_hailo_deps_present(monkeypatch)

    artifacts_dir = tmp_path / "art"
    artifacts_dir.mkdir()

    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"dummy")

    calls = {"n": 0}

    def compile_counting(onnx_p: Path, hef_p: Path, hw: str, input_shapes=None):
        calls["n"] += 1
        _fake_compile(onnx_p, hef_p, hw, input_shapes)

    backend = HailoBackend(
        require_device=False,
        tool_version="test",
        compile_fn=compile_counting,
        runtime_factory=lambda hef, o: _FakeRuntime(hef, {**o, "input_names": ["x"], "output_names": ["y"]}),
    )
    run_cfg = RunCfg(model_path=str(onnx_path), options={"hailo_hw_arch": "hailo8"})

    handle = backend.prepare(run_cfg, artifacts_dir)
    assert calls["n"] == 1
    assert handle.input_names == ["x"]
    assert handle.output_names == ["y"]
    assert Path(handle.handle.meta["hef_path"]).exists()


def test_run_returns_outputs_and_meta(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _patch_hailo_deps_present(monkeypatch)

    artifacts_dir = tmp_path / "art"
    artifacts_dir.mkdir()

    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"dummy")

    backend = HailoBackend(
        require_device=False,
        tool_version="test",
        compile_fn=_fake_compile,
        runtime_factory=lambda hef, o: _FakeRuntime(hef, {**o, "input_names": ["x"], "output_names": ["y"]}),
    )
    run_cfg = RunCfg(model_path=str(onnx_path), options={"hailo_hw_arch": "hailo8"})
    handle = backend.prepare(run_cfg, artifacts_dir)

    x = np.array([1, 2, 3], dtype=np.float32)
    out = backend.run(handle, {"x": x})
    assert "y" in out.outputs
    np.testing.assert_array_equal(out.outputs["y"], x + 1)


def test_backend_contract_compatible_with_graph_runner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _patch_hailo_deps_present(monkeypatch)

    # GraphRunner uses a Harness to create inputs.
    harness = _DummyHarness(inputs={"x": np.array([0.0, 1.0], dtype=np.float32)})

    backend = HailoBackend(
        require_device=False,
        tool_version="test",
        compile_fn=_fake_compile,
        runtime_factory=lambda hef, o: _FakeRuntime(hef, {**o, "input_names": ["x"], "output_names": ["y"]}),
    )

    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"dummy")
    run_cfg = RunCfg(model_path=str(onnx_path), options={"hailo_hw_arch": "hailo8"})

    plan = GraphPlan(
        label="g",
        sample_cfg=SampleCfg(),
        warmup_runs=1,
        measured_runs=2,
        measure_interface=False,
        stages=[StagePlan(name="full", backend_name="hailo", run_cfg=run_cfg)],
    )

    runner = GraphRunner(backends={"hailo": backend})
    res = runner.run_graph(plan, harness=harness, artifacts_dir=tmp_path / "artifacts")

    assert res.status == "ok"
    assert "stage" in res.metrics
    assert "full" in res.metrics["stage"]
    assert "runtime_ms" in res.metrics["stage"]["full"]
    assert len(res.metrics["stage"]["full"]["runtime_ms"]["measured_runs"]) == 2
    # Ensure the harness saw the backend outputs
    np.testing.assert_array_equal(harness.last_outputs["y"], np.array([1.0, 2.0], dtype=np.float32))


def test_prepare_raises_clean_error_when_no_compiler_and_no_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    real_find_spec = importlib.util.find_spec

    class _Spec:
        pass

    def fake_find_spec(name: str, *args, **kwargs):
        if name == "hailo_sdk_client":
            return None
        if name == "hailo_platform":
            return _Spec()
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)

    artifacts_dir = tmp_path / "art"
    artifacts_dir.mkdir()
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"dummy")

    backend = HailoBackend(
        require_device=False,
        tool_version="test",
        runtime_factory=lambda hef, o: _FakeRuntime(hef, o),
    )
    run_cfg = RunCfg(model_path=str(onnx_path), options={"hailo_hw_arch": "hailo8"})

    with pytest.raises(HailoBackendUnavailable):
        backend.prepare(run_cfg, artifacts_dir)
