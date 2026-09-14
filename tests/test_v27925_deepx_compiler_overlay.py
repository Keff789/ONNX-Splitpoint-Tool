"""Real isolated child processes, with fake compiler modules and no hardware."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest

from onnx_splitpoint_tool.deepx import compiler, env_status


@pytest.fixture
def compiler_fixture(tmp_path, monkeypatch):
    vendor = tmp_path / "vendor packages"
    overlay = tmp_path / "cu126 overlay"
    venv = tmp_path / "compiler venv"
    for directory in (vendor, overlay, venv / "bin"):
        directory.mkdir(parents=True)
    torch_template = '''from types import SimpleNamespace
import os
__version__ = VERSION
version = SimpleNamespace(cuda=CUDA)
cuda = SimpleNamespace(is_available=lambda: os.environ.get("CUDA_VISIBLE_DEVICES") not in ("", "-1"),
 current_device=lambda: 0, get_device_name=lambda i: "GTX1080Ti fixture",
 get_device_capability=lambda i: (6,1), get_arch_list=lambda: ARCHES)
# Synthetic CUDA arithmetic for child-context tests. No hardware claim.
class Tensor:
    def __init__(self, shape, value): self.shape, self.value = tuple(shape), value
    def __matmul__(self, other): return Tensor((self.shape[0], other.shape[1]), self.shape[1]*self.value*other.value)
float32 = "float32"
def ones(shape, **kwargs): return Tensor(shape, 1.0)
def full(shape, value, **kwargs): return Tensor(shape, value)
def full_like(t, value): return Tensor(t.shape, value)
def allclose(a, b): return a.shape == b.shape and a.value == b.value
nn = SimpleNamespace(functional=SimpleNamespace(conv2d=lambda x,w: Tensor((x.shape[0],w.shape[0],x.shape[2]-w.shape[2]+1,x.shape[3]-w.shape[3]+1),x.shape[1]*w.shape[2]*w.shape[3]*x.value*w.value)))
cuda.synchronize = lambda: None
'''
    for directory, version, cuda, arches in ((vendor, "original+cu130", "13.0", ["sm_75"]), (overlay, "overlay+cu126", "12.6", ["sm_60"])):
        (directory / "torch").mkdir()
        (directory / "torch/__init__.py").write_text(torch_template.replace("VERSION", repr(version)).replace("CUDA)", repr(cuda) + ")").replace("ARCHES", repr(arches)))
    (vendor / "numpy.py").write_text("# Fake compiler dependency; no numerical execution in this test.\n")
    (vendor / "cv2.py").write_text("IMREAD_COLOR=1\ndef imread(*args): return None\n")
    (vendor / "dx_engine.py").write_text("import torch\n__version__ = torch.__version__\n")
    (vendor / "dx_com.py").write_text('''import json, os, sys
from pathlib import Path
import torch
__version__ = "unchanged-dxcom"
def compile(model, config, output_dir, **kwargs):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "model.dxnn").write_bytes(b"fake compiled artifact")
    hook = sys.modules.get("sitecustomize")
    (out / "observed_environment.json").write_text(json.dumps({
        "torch": torch.__version__, "torch_file": torch.__file__,
        "pythonpath": os.environ.get("PYTHONPATH"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "sitecustomize": str(getattr(hook, "__file__", ""))}))
''')
    python = venv / "bin/python"
    python.write_text("#!/bin/sh\nexec " + shlex.quote(sys.executable) + ' "$@"\n')
    python.chmod(0o755)
    (venv / "bin/activate").write_text("export PATH=" + shlex.quote(str(venv / "bin")) + ':"$PATH"\n' + "export PYTHONPATH=" + shlex.quote(str(vendor)) + "\n")
    cli = venv / "bin/dxcom"
    cli.write_text("#!" + str(sys.executable) + '''
import argparse, dx_com
p=argparse.ArgumentParser()
for key in ("m", "c", "o"): p.add_argument("-" + key)
p.add_argument("--opt_level")
a=p.parse_args()
dx_com.compile(model=a.m, config=a.c, output_dir=a.o)
''')
    cli.chmod(0o755)
    model = tmp_path / "model.onnx"
    model.write_bytes(b"fake ONNX")
    config = tmp_path / "config.json"
    config.write_text('{"inputs": {"x": [1, 3, 2, 2]}}')
    manifest = tmp_path / "activations.json"
    manifest.write_text('{"tensors": [{"name": "x"}]}')
    (tmp_path / "sample_000.npy").write_bytes(b"not loaded by fake compiler")
    monkeypatch.setenv("PYTHONPATH", str(vendor))
    monkeypatch.setenv("ONNX_SPLITPOINT_DEEPX_COMPILER_OVERLAY", str(overlay))
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("ONNX_SPLITPOINT_ARTIFACT_POLICY", raising=False)
    return dict(vendor=vendor, overlay=overlay, venv=venv, python=python, model=model, config=config, manifest=manifest)


def test_compiler_probes_use_overlay_runtime_and_parent_do_not(compiler_fixture):
    f = compiler_fixture
    before = dict(os.environ)
    result = env_status.inspect_deepx_environment(probe_import=True, config={
        "dx_all_suite_root": str(f["venv"].parent), "compiler_venv": str(f["venv"]), "runtime_venv": str(f["venv"]),
    })
    assert result["compiler_ready"]
    assert result["compiler_imports"][0]["ok"]
    assert result["compiler_imports"][0]["package_version"] == "unchanged-dxcom"
    assert result["runtime_imports"][0]["package_version"] == "original+cu130"
    assert result["compiler_cuda_preflight"]["torch_version"] == "overlay+cu126"
    assert result["compiler_cuda_preflight"]["torch_cuda_version"] == "12.6"
    assert result["compiler_cuda_preflight"]["compiler_overlay"] == str(f["overlay"])
    assert "Compiler overlay: " + str(f["overlay"]) in env_status.format_deepx_status_text(status=result)
    assert dict(os.environ) == before
    ordinary = subprocess.check_output([str(f["python"]), "-c", "import torch; print(torch.__version__)"], text=True)
    assert ordinary.strip() == "original+cu130"


@pytest.mark.parametrize("kind", ["cli", "npz_shim", "tensor_loader"])
def test_all_compiler_entrypoints_use_scoped_overlay_and_record_it(compiler_fixture, tmp_path, kind):
    f = compiler_fixture
    out = tmp_path / kind
    common = dict(onnx_path=f["model"], config_path=f["config"], output_dir=out, compiler_venv=f["venv"], timeout_s=20)
    if kind == "cli":
        result = compiler.compile_dxnn(**common)
    elif kind == "npz_shim":
        result = compiler.compile_dxnn_with_npz_cv2_shim(**common, input_name="x", input_shape=[1, 3, 2, 2])
    else:
        result = compiler.compile_dxnn_with_tensor_loader(**common, activation_manifest=f["manifest"])
    assert result.ok, Path(result.log_path).read_text()
    observed = json.loads((out / "observed_environment.json").read_text())
    assert observed["torch"] == "overlay+cu126"
    assert observed["cuda_visible_devices"] is None
    paths = observed["pythonpath"].split(os.pathsep)
    if kind == "tensor_loader":
        assert paths[0] == str(out / "tensor_loader_sitecustomize")
        assert observed["sitecustomize"] == str(out / "tensor_loader_sitecustomize/sitecustomize.py")
        paths = paths[1:]
    assert paths == [str(f["overlay"]), str(f["vendor"])]
    report = json.loads((out / ("build_manifest_tensor_loader.json" if kind == "tensor_loader" else "build_manifest.json")).read_text())
    assert report["compiler_cuda_preflight"]["torch_version"] == "overlay+cu126"
    assert report["compiler_cuda_preflight"]["compiler_overlay"] == str(f["overlay"])


@pytest.mark.parametrize("mask", ["", "-1", "0", "GPU-fixture-id"])
def test_compiler_environment_preserves_explicit_gpu_masks(compiler_fixture, monkeypatch, mask):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mask)
    env = env_status.compiler_subprocess_environment()
    assert env["CUDA_VISIBLE_DEVICES"] == mask
    result = env_status.probe_compiler_cuda_architecture(compiler_fixture["python"])
    assert result["cuda_visible_devices"] == mask
    assert result["status"] == ("not_applicable" if mask in ("", "-1") else "compatible")


@pytest.mark.parametrize("kind", ["cli", "npz_shim", "tensor_loader"])
def test_invalid_overlay_blocks_cold_compile_as_environment_problem(compiler_fixture, monkeypatch, tmp_path, kind):
    monkeypatch.setenv("ONNX_SPLITPOINT_DEEPX_COMPILER_OVERLAY", str(tmp_path / "missing-overlay"))
    monkeypatch.setattr(compiler, "_run_owned_compiler", lambda *a, **k: pytest.fail("invalid environment was dispatched"))
    f = compiler_fixture
    common = dict(onnx_path=f["model"], config_path=f["config"], output_dir=tmp_path / kind, compiler_venv=f["venv"])
    if kind == "cli":
        result = compiler.compile_dxnn(**common)
    elif kind == "npz_shim":
        result = compiler.compile_dxnn_with_npz_cv2_shim(**common, input_name="x", input_shape=[1, 3, 2, 2])
    else:
        result = compiler.compile_dxnn_with_tensor_loader(**common, activation_manifest=f["manifest"])
    assert result.status == "compiler_environment_incompatible"
    report = json.loads((Path(result.output_dir) / "build_manifest.json").read_text())
    assert report["compiler_dispatched"] is False
    assert report["compiler_cuda_preflight"]["reason"] == "deepx_compiler_overlay_invalid"
    assert "COMPILE_INFEASIBLE" not in json.dumps(report)


def test_without_overlay_existing_inheritance_is_unchanged(compiler_fixture, monkeypatch):
    monkeypatch.delenv("ONNX_SPLITPOINT_DEEPX_COMPILER_OVERLAY")
    assert env_status.compiler_subprocess_environment() is None
    status = env_status.probe_compiler_cuda_architecture(compiler_fixture["python"])
    assert status["torch_version"] == "original+cu130"
    assert status["status"] == "incompatible"
    assert "compiler_overlay" not in status


def test_relative_overlay_is_resolved_before_compiler_shell_changes_directory(compiler_fixture, monkeypatch):
    f = compiler_fixture
    monkeypatch.chdir(f["overlay"].parent)
    monkeypatch.setenv("ONNX_SPLITPOINT_DEEPX_COMPILER_OVERLAY", f["overlay"].name)
    child_env = env_status.compiler_subprocess_environment()
    assert child_env["PYTHONPATH"].split(os.pathsep)[0] == str(f["overlay"])
    result = env_status._run_owned_probe(
        [str(f["python"]), "-c", "import os; os.chdir('/'); import torch; print(torch.__version__)"],
        timeout_s=10, env=child_env,
    )
    assert result.returncode == 0
    assert result.stdout.strip() == "overlay+cu126"


def test_cache_only_policy_does_not_even_probe_invalid_overlay(compiler_fixture, monkeypatch):
    monkeypatch.setenv("ONNX_SPLITPOINT_DEEPX_COMPILER_OVERLAY", "/nonexistent/overlay")
    monkeypatch.setattr(env_status, "compiler_dispatch_forbidden", lambda: True)
    monkeypatch.setattr(env_status, "_run_owned_probe", lambda *a, **k: pytest.fail("compiler process started"))
    assert env_status.probe_compiler_cuda_architecture(compiler_fixture["python"])["status"] == "not_probed"


def test_overlay_is_not_a_new_compiler_cache_identity(compiler_fixture, monkeypatch):
    from onnx_splitpoint_tool.workflow.deepx_build_binding import _deepx_compiler_identity
    f = compiler_fixture
    cfg = {"dx_all_suite_root": str(f["venv"].parent), "compiler_venv": str(f["venv"]), "runtime_venv": str(f["venv"])}
    with_overlay = env_status.inspect_deepx_environment(probe_import=True, config=cfg)
    monkeypatch.delenv("ONNX_SPLITPOINT_DEEPX_COMPILER_OVERLAY")
    without_overlay = env_status.inspect_deepx_environment(probe_import=True, config=cfg)
    assert _deepx_compiler_identity(cfg={}, environment_status=with_overlay) == _deepx_compiler_identity(cfg={}, environment_status=without_overlay)
