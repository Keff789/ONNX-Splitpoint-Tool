"""T35.02–10: real overlay/collector/pre-SDK child boundary, no CUDA execution.

The sole replaced compute boundary runs an actual Python environment recorder
through the unchanged production process supervisor. A recorder is deliberately
never reported as compute_pass: numerical GPU/XLA validation remains separate.
V280_F1_BASELINE_COLLECTOR optionally points to the exact v34 collector for the
pre-patch red assertion; normal release acceptance always loads shipped source.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import sys
import zipfile

import pytest

from onnx_splitpoint_tool import hailo_compiler_context as cc
from onnx_splitpoint_tool import hailo_dependency_plan as hp

SOURCE = Path(__file__).resolve().parents[1]
OVERLAY_ENV = "ONNX_SPLITPOINT_HAILO8_DEPENDENCY_MANIFEST"
ENV_KEYS = ("LD_LIBRARY_PATH", "PATH", "CUDA_HOME", "CUDA_PATH", "CUDA_VISIBLE_DEVICES",
            "XLA_FLAGS", "TF_NUM_INTEROP_THREADS", "TF_NUM_INTRAOP_THREADS",
            "TF_FORCE_GPU_ALLOW_GROWTH", "PYTHONPATH", OVERLAY_ENV)


def _metadata(site, name, version, requires=(), extras=()):
    dist = site / (name.replace("-", "_") + "-" + version + ".dist-info")
    dist.mkdir(parents=True)
    path = dist / "METADATA"
    path.write_text("Metadata-Version: 2.1\nName: " + name + "\nVersion: " + version + "\n" +
                    "".join("Requires-Dist: " + value + "\n" for value in requires) +
                    "".join("Provides-Extra: " + value + "\n" for value in extras))
    return path


def _venv(root):
    (root / "bin").mkdir(parents=True)
    (root / "bin/python").symlink_to(sys.executable)
    (root / "pyvenv.cfg").write_text("include-system-site-packages = false\n")
    site = root / "lib" / ("python" + ".".join(map(str, sys.version_info[:2]))) / "site-packages"
    site.mkdir(parents=True)
    return site


def _assembler():
    return ("#!" + sys.executable + "\nimport sys\nfrom pathlib import Path\n"
            "if '--version' in sys.argv: print('synthetic fixture ptxas 12.5.82')\n"
            "else: Path(sys.argv[sys.argv.index('-o')+1]).write_bytes(b'fixture object; not CUDA')\n")


def _local_components(site):
    root = site / "triton/backends/nvidia"
    (root / "bin").mkdir(parents=True)
    (root / "lib").mkdir()
    (root / "bin/ptxas").write_text(_assembler())
    (root / "bin/ptxas").chmod(0o700)
    (root / "lib/libdevice.10.bc").write_text("synthetic bitcode, never loaded")


@pytest.fixture
def boundary(tmp_path, monkeypatch):
    h8 = tmp_path / "venv_hailo8"
    site = _venv(h8)
    _metadata(site, "hailo-dataflow-compiler", "3.33.1", ["tensorflow==2.18.0"])
    tf = _metadata(site, "tensorflow", "2.18.0",
                   ['nvidia-cuda-nvcc-cu12==12.5.82; extra == "and-cuda"'], ["and-cuda"])
    wheel = tmp_path / "nvidia_cuda_nvcc_cu12-12.5.82-py3-none-any.whl"
    files = {
        "nvidia_cuda_nvcc_cu12-12.5.82.dist-info/METADATA":
            "Metadata-Version: 2.1\nName: nvidia-cuda-nvcc-cu12\nVersion: 12.5.82\n",
        "nvidia/cuda_nvcc/lib/libfixture.so.12": "synthetic library bytes, never loaded",
        "nvidia/cuda_nvcc/bin/ptxas": _assembler(),
        "nvidia/cuda_nvcc/nvvm/libdevice/libdevice.10.bc": "synthetic bitcode, never loaded",
    }
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, data in files.items():
            info = zipfile.ZipInfo(name)
            info.external_attr = ((0o100755 if name.endswith("/ptxas") else 0o100644) << 16)
            archive.writestr(info, data)
    plan = hp.build_plan(hp.collect_inventory(h8 / "bin/python"), tmp_path / "hailo8_cuda_reviewed",
                         packages=["nvidia-cuda-nvcc-cu12"], wheels=[wheel])
    review = tmp_path / "review.json"
    review.write_text(json.dumps(plan, sort_keys=True))
    manifest = hp.stage_reviewed_plan(plan, expected_plan_sha256=hashlib.sha256(review.read_bytes()).hexdigest(),
                                      plan_file=review)
    assert json.loads(manifest.read_text())["hardware_execution"] == "NOT_RUN"
    h10 = tmp_path / "venv_hailo10"
    _local_components(_venv(h10))
    external = tmp_path / "external_bin"
    external.mkdir()
    smi = external / "nvidia-smi"
    smi.write_text("#!" + sys.executable + "\nprint('0, GPU-fixture-zero, 6.1')\n")
    smi.chmod(0o700)
    for key in list(os.environ):
        if key.startswith(("ONNX_SPLITPOINT_HAILO", "SPLITPOINT_HAILO")) or key in {"CUDA_VISIBLE_DEVICES", "PYTHONHOME", "VIRTUAL_ENV"}:
            monkeypatch.delenv(key)
    monkeypatch.setenv("PATH", str(external) + os.pathsep + os.defpath)
    monkeypatch.setenv(OVERLAY_ENV, str(manifest))
    monkeypatch.setenv("LD_LIBRARY_PATH", "/unchanged/base/libs")
    monkeypatch.setenv("XLA_FLAGS", "--xla_dump_to=/unchanged/xla")
    monkeypatch.setenv("TF_NUM_INTEROP_THREADS", "7")
    monkeypatch.setenv("TF_NUM_INTRAOP_THREADS", "9")
    monkeypatch.setenv("TF_FORCE_GPU_ALLOW_GROWTH", "false")
    monkeypatch.setenv("PYTHONPATH", "/parent/module/path")
    path = Path(os.environ.get("V280_F1_BASELINE_COLLECTOR", SOURCE / "scripts/hailo_gpu_diagnostics/collect.py"))
    spec = importlib.util.spec_from_file_location("v280_boundary_collector", path)
    collector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(collector)
    monkeypatch.setattr(collector, "platform_interlock_path", lambda: tmp_path / "workflow.lock")
    return {"root": tmp_path, "h8": h8, "h10": h10, "site": site, "tf": tf,
            "manifest": manifest, "collector": collector,
            "overlay_lib": str(manifest.parent / "packages/nvidia/cuda_nvcc/lib")}


# This child imports only the ordinary pre-SDK hook, never a vendor framework.
RECORDER = r'''
import json, os, signal, sys, time
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from onnx_splitpoint_tool.cuda_probe import auto_configure_cuda
work = Path(sys.argv[2]); keys = json.loads(sys.argv[3]); mode = sys.argv[4]
before = {key: os.environ.get(key) for key in keys}
context = auto_configure_cuda()
after = {key: os.environ.get(key) for key in keys}
view = Path(context['view_root'])
record = {'before': before, 'after': after, 'context': context, 'pid': os.getpid(),
          'parent_pid': os.getppid(), 'view_alive_after_hook': view.is_dir()}
(work/'boundary.json').write_text(json.dumps(record))
def terminated(signum, frame):
    (work/'termination.json').write_text(json.dumps({'view_alive': view.is_dir(), 'signal': signum}))
    raise SystemExit(128 + signum)
signal.signal(signal.SIGTERM, terminated)
(work/'ready').touch()
if mode == 'interrupted':
    os.kill(os.getppid(), signal.SIGINT)
if mode in ('timeout', 'interrupted'):
    time.sleep(20)
if mode == 'failed':
    raise SystemExit(7)
(work/'gpu_compute_result.json').write_text(json.dumps({
    'schema': 'hailo-gpu-compute-smoke-r1', 'family': context['family'],
    'status': 'environment_recorded_only', 'checks': [],
    'error': 'GPU and TensorFlow were not executed by this boundary fixture'}))
'''


def _collect(boundary, monkeypatch, *, families=("hailo8",), mode="normal", spawn=True):
    collector = boundary["collector"]
    captured = []
    real_run = collector.run_command

    def recorder(command, work, env, timeout):
        assert spawn, "invalid selection must never reach the compute spawn boundary"
        # Keep actual collector construction, resolver, context and supervisor;
        # only replace GPU-specific numerical work by an external recorder.
        assert command[1:4] == ["-I", "-B", "-c"]
        assert "auto_configure_cuda()" in command[4]
        captured.append({"env": {key: env.get(key) for key in ENV_KEYS}, "work": work})
        result = real_run([command[0], "-I", "-B", "-c", RECORDER, str(SOURCE), str(work),
                           json.dumps(ENV_KEYS), mode], work, env, timeout)
        captured[-1]["result"] = result
        return result

    monkeypatch.setattr(collector, "run_command", recorder)
    args = argparse.Namespace(output_parent=boundary["root"] / "output", families=list(families),
        venv_hailo8=boundary["h8"], venv_hailo10=boundary["h10"], gpu="0",
        timeout=0.5 if mode == "timeout" else 5, compiler_context=True)
    parent = dict(os.environ)
    rc = collector.collect(args)
    assert dict(os.environ) == parent
    outputs = sorted(args.output_parent.iterdir(), key=lambda p: p.stat().st_mtime_ns)
    output = [p for p in outputs if p.is_dir()][-1]
    summary = json.loads((output / "collection_summary.json").read_text())
    assert rc == 2, "a non-GPU recorder must never yield a GPU PASS"
    assert summary["status"] != "compute_pass"
    return captured, summary, output


def _assert_recorded(captured, expected_lib):
    assert len(captured) == 1
    row = captured[0]
    data = json.loads((row["work"] / "boundary.json").read_text())
    assert data["pid"] != os.getpid() and data["parent_pid"] == os.getpid()
    assert data["before"]["LD_LIBRARY_PATH"] == expected_lib
    assert data["after"] == data["before"]
    assert data["view_alive_after_hook"] is True
    assert data["context"]["mode"] == "gpu_resolved"
    assert data["context"]["target_arch"] == "sm_61"
    assert not Path(data["context"]["view_root"]).exists()
    return data


def test_t35_02_real_overlay_collector_child_and_pre_sdk_hook(boundary, monkeypatch):
    captured, _, _ = _collect(boundary, monkeypatch)
    data = _assert_recorded(captured, boundary["overlay_lib"] + ":/unchanged/base/libs")
    assert data["context"]["dependency_manifest"] == str(boundary["manifest"])
    assert data["context"]["component_source"] == "selected_hailo8_dependency_overlay"


def test_t35_03_no_inherited_library_path_has_no_empty_entry(boundary, monkeypatch):
    monkeypatch.delenv("LD_LIBRARY_PATH")
    captured, _, _ = _collect(boundary, monkeypatch)
    data = _assert_recorded(captured, boundary["overlay_lib"])
    assert "" not in data["before"]["LD_LIBRARY_PATH"].split(os.pathsep)


def test_t35_04_repeated_independent_collectors_do_not_accumulate(boundary, monkeypatch):
    original_run = boundary["collector"].run_command
    observed = []
    for _ in range(2):
        monkeypatch.setattr(boundary["collector"], "run_command", original_run)
        captured, _, _ = _collect(boundary, monkeypatch)
        observed.append(_assert_recorded(captured, boundary["overlay_lib"] + ":/unchanged/base/libs"))
    assert observed[0]["before"]["LD_LIBRARY_PATH"] == observed[1]["before"]["LD_LIBRARY_PATH"]


@pytest.mark.parametrize("damage", ["missing", "invalid_json", "family", "venv"])
def test_t35_05_invalid_manifest_or_binding_prevents_compute(boundary, monkeypatch, damage):
    manifest = boundary["manifest"]
    if damage == "missing":
        manifest.unlink()
    elif damage == "invalid_json":
        manifest.write_text("invalid json")
    else:
        data = json.loads(manifest.read_text())
        data["family" if damage == "family" else "selected_venv"] = "wrong"
        manifest.write_text(json.dumps(data))
    captured, summary, _ = _collect(boundary, monkeypatch, spawn=False)
    assert not captured
    assert summary["families"][0]["status"] == "failed"
    assert summary["package_installations"] is False


@pytest.mark.parametrize("damage", ["manifest_family", "framework_metadata", "overlay_metadata", "overlay_hook"])
def test_t35_06_binding_is_revalidated_after_resolver(boundary, monkeypatch, damage):
    real = cc.resolve_hailo_compiler_context
    def mutate_after_resolution(*args, **kwargs):
        context = real(*args, **kwargs)
        manifest = boundary["manifest"]
        if damage == "manifest_family":
            data = json.loads(manifest.read_text()); data["family"] = "hailo10h"
            manifest.write_text(json.dumps(data))
        elif damage == "framework_metadata":
            path = boundary["tf"]
            path.write_text(path.read_text().replace("2.18.0", "2.19.0"))
        elif damage == "overlay_metadata":
            path = next((manifest.parent / "packages").glob("*.dist-info/METADATA"))
            path.write_text(path.read_text().replace("12.5.82", "12.5.83"))
        else:
            (manifest.parent / "packages/danger.pth").write_text("import forbidden\n")
        return context
    monkeypatch.setattr(cc, "resolve_hailo_compiler_context", mutate_after_resolution)
    captured, summary, _ = _collect(boundary, monkeypatch, spawn=False)
    assert not captured
    assert "hailo8_overlay" in summary["families"][0]["error"]


@pytest.mark.parametrize("changed", [{"family": "hailo10h"}, {"device": "cpu"}])
def test_t35_05_invalid_resolved_context_never_applies_h8_overlay(boundary, monkeypatch, changed):
    real = cc.resolve_hailo_compiler_context
    def corrupt_selection(*args, **kwargs):
        return {**real(*args, **kwargs), **changed}
    monkeypatch.setattr(cc, "resolve_hailo_compiler_context", corrupt_selection)
    captured, summary, _ = _collect(boundary, monkeypatch, spawn=False)
    assert not captured
    assert "hailo8_overlay_cannot_be_used_by_other_family" in summary["families"][0]["error"]


def test_t35_07_h8_without_overlay_uses_its_own_components(boundary, monkeypatch):
    monkeypatch.delenv(OVERLAY_ENV)
    _local_components(boundary["site"])
    captured, _, _ = _collect(boundary, monkeypatch)
    data = _assert_recorded(captured, "/unchanged/base/libs")
    assert "dependency_manifest" not in data["context"]
    assert data["context"]["component_source"] != "selected_hailo8_dependency_overlay"


def test_t35_08_sequential_h8_h10_does_not_leak_overlay(boundary, monkeypatch):
    captured, _, _ = _collect(boundary, monkeypatch, families=("hailo8", "hailo10h"))
    assert len(captured) == 2
    h8 = _assert_recorded(captured[:1], boundary["overlay_lib"] + ":/unchanged/base/libs")
    h10 = _assert_recorded(captured[1:], "/unchanged/base/libs")
    assert h8["context"]["family"] == "hailo8"
    assert h10["context"]["family"] == "hailo10h"
    assert "dependency_manifest" not in h10["context"]
    assert os.environ[OVERLAY_ENV] == str(boundary["manifest"])


@pytest.mark.parametrize("mode", ["normal", "failed", "timeout", "interrupted"])
def test_t35_09_parent_threads_flags_and_private_view_lifetime(boundary, monkeypatch, mode):
    captured, summary, _ = _collect(boundary, monkeypatch, mode=mode)
    data = _assert_recorded(captured, boundary["overlay_lib"] + ":/unchanged/base/libs")
    assert data["before"]["TF_NUM_INTEROP_THREADS"] == "7"
    assert data["before"]["TF_NUM_INTRAOP_THREADS"] == "9"
    assert data["before"]["TF_FORCE_GPU_ALLOW_GROWTH"] == "false"
    assert "--xla_dump_to=/unchanged/xla" in shlex.split(data["before"]["XLA_FLAGS"])
    assert data["before"]["PYTHONPATH"] is None
    if mode in {"timeout", "interrupted"}:
        terminated = json.loads((captured[0]["work"] / "termination.json").read_text())
        assert terminated["view_alive"] is True
    if mode == "interrupted":
        assert summary["status"] == "interrupted"
        assert boundary["collector"].live_group(data["pid"]) == []
    else:
        assert captured[0]["result"]["cleanup_complete"] is True
        assert captured[0]["result"]["remaining_group_pids"] == []
        assert captured[0]["result"]["timed_out"] is (mode == "timeout")


def test_t35_10_environment_report_matches_actual_child_and_compact_zip(boundary, monkeypatch):
    captured, summary, output = _collect(boundary, monkeypatch)
    data = _assert_recorded(captured, boundary["overlay_lib"] + ":/unchanged/base/libs")
    report_path = captured[0]["work"] / "environment_overrides.json"
    report = json.loads(report_path.read_text())
    for key, value in report["child_only_overrides"].items():
        if key in ENV_KEYS:
            assert value == data["before"][key], key
    assert report["child_only_overrides"]["LD_LIBRARY_PATH"] == data["before"]["LD_LIBRARY_PATH"]
    assert report["dependency_overlay"] == {"family": "hailo8", "manifest_path": str(boundary["manifest"])}
    assert report["parent_environment_modified"] is False
    assert summary["model_compiler_invoked"] is False
    with zipfile.ZipFile(output.with_suffix(".zip")) as archive:
        assert archive.read("hailo8/environment_overrides.json") == report_path.read_bytes()
        assert archive.testzip() is None
