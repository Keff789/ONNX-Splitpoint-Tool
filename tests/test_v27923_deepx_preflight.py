from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from onnx_splitpoint_tool.deepx import compiler, env_status


def _metadata(capability=(6, 1), arches=None, available=True):
    return {
        "torch_version": "audit-torch",
        "cuda_available": available,
        "device_name": "GTX 1080 Ti",
        "device_capability": list(capability),
        "compiled_architectures": arches if arches is not None else ["sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120"],
    }


@pytest.mark.parametrize(("payload", "status"), [
    (_metadata(), "incompatible"),
    (_metadata((8, 6)), "compatible"),
    (_metadata((8, 6), ["sm_80"]), "compatible"),
    (_metadata((8, 6), ["compute_75"]), "compatible"),
    (_metadata((8, 0), ["sm_86"]), "incompatible"),
    (_metadata(available=False), "not_applicable"),
    (_metadata(arches=[]), "unknown"),
    (_metadata(arches=["sm_90a"]), "unknown"),
    ({}, "unknown"),
])
def test_architecture_metadata_classification(payload, status):
    assert env_status._cuda_architecture_status(payload)["status"] == status


def test_probe_uses_selected_compiler_python_and_metadata_only(tmp_path, monkeypatch):
    python = tmp_path / "actual-dxcom-venv/bin/python"
    python.parent.mkdir(parents=True)
    python.touch()
    calls = []

    def probe(args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args, 0, "torch warning\nDEEPX_COMPILER_CUDA_METADATA=" + json.dumps(_metadata()) + "\n", "")

    monkeypatch.setattr(env_status, "_run_owned_probe", probe)
    monkeypatch.setattr(env_status, "compiler_dispatch_forbidden", lambda: False)
    result = env_status.probe_compiler_cuda_architecture(python)
    assert result["status"] == "incompatible"
    assert result["reason"] == "deepx_compiler_cuda_architecture_unsupported"
    assert result["compiler_python"] == str(python)
    assert calls[0][0][:2] == [str(python), "-c"]
    assert calls[0][1]["timeout_s"] == 15
    assert "get_arch_list" in calls[0][0][2]
    assert "dx_com" not in calls[0][0][2]
    assert "torch.empty" not in calls[0][0][2]
    assert "sm_61" in env_status.compiler_cuda_preflight_message(result)


@pytest.mark.parametrize("failure", ["timeout", "import", "malformed"])
def test_probe_unavailable_is_unknown_not_known_incompatible(tmp_path, monkeypatch, failure):
    python = tmp_path / "python"
    python.touch()

    def probe(args, **kwargs):
        if failure == "timeout":
            raise subprocess.TimeoutExpired(args, 15)
        return subprocess.CompletedProcess(args, 1 if failure == "import" else 0, "unavailable", "")

    monkeypatch.setattr(env_status, "compiler_dispatch_forbidden", lambda: False)
    monkeypatch.setattr(env_status, "_run_owned_probe", probe)
    result = env_status.probe_compiler_cuda_architecture(python)
    assert result["status"] == "unknown"


def test_cache_only_policy_never_starts_compiler_python(tmp_path, monkeypatch):
    monkeypatch.setattr(env_status, "compiler_dispatch_forbidden", lambda: True)
    monkeypatch.setattr(env_status, "_run_owned_probe", lambda *a, **k: pytest.fail("compiler probe started"))
    result = env_status.probe_compiler_cuda_architecture(tmp_path / "python")
    assert result["status"] == "not_probed"


@pytest.mark.parametrize(("payload", "ready"), [
    (_metadata(), False), (_metadata((8, 6)), True),
    (_metadata(available=False), True), ({}, True),
])
def test_normal_non_import_environment_preflight_checks_cuda(tmp_path, monkeypatch, payload, ready):
    root = tmp_path / "suite"
    venv = root / "compiler"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin/python").touch()
    (venv / "bin/dxcom").touch()
    monkeypatch.setattr(env_status, "_which_many", lambda names: dict.fromkeys(names))
    monkeypatch.setattr(env_status, "_python_tag", lambda *a, **k: "cp311")
    monkeypatch.setattr(env_status, "compiler_dispatch_forbidden", lambda: False)
    calls = []

    def cuda_probe(python, **kwargs):
        calls.append(python)
        return {"compiler_python": str(python), **env_status._cuda_architecture_status(payload)}

    monkeypatch.setattr(env_status, "probe_compiler_cuda_architecture", cuda_probe)
    result = env_status.inspect_deepx_environment(probe=False, probe_import=False, config={
        "dx_all_suite_root": str(root), "compiler_venv": str(venv), "runtime_venv": str(venv),
    })
    assert result["compiler_ready"] is ready
    assert result["compiler_imports"] == []
    assert calls == [venv / "bin/python"]
    if not ready:
        assert any("deepx_compiler_cuda_architecture_unsupported" in hint for hint in result["hints"])


@pytest.mark.parametrize(("function", "kwargs"), [
    (compiler.compile_dxnn, {}),
    (compiler.compile_dxnn_with_npz_cv2_shim, {"input_name": "x", "input_shape": [1, 3, 2, 2]}),
    (compiler.compile_dxnn_with_tensor_loader, {"activation_manifest": "unused.json"}),
])
def test_actual_compiler_entry_points_block_known_incompatible_host(tmp_path, monkeypatch, function, kwargs):
    status = env_status._cuda_architecture_status(_metadata())
    monkeypatch.setattr(compiler, "probe_compiler_cuda_architecture", lambda *a, **k: status)
    monkeypatch.setattr(compiler, "_run_owned_compiler", lambda *a, **k: pytest.fail("DX-COM dispatched"))
    result = function(onnx_path="model.onnx", config_path="config.json", output_dir=tmp_path, compiler_venv=tmp_path / "compiler", **kwargs)
    assert result.status == "compiler_environment_incompatible"
    assert result.ok is False
    assert result.returncode is None
    manifest = json.loads((tmp_path / "build_manifest.json").read_text())
    assert manifest["compiler_dispatched"] is False
    assert "COMPILE_INFEASIBLE" not in json.dumps(manifest)


@pytest.mark.parametrize("status", ["compatible", "not_applicable", "not_probed"])
def test_non_incompatible_metadata_does_not_invent_a_block(tmp_path, monkeypatch, status):
    monkeypatch.setattr(compiler, "probe_compiler_cuda_architecture", lambda *a, **k: {"status": status, "operations_probe_status": "not_applicable" if status == "not_applicable" else "pass"})
    assert compiler._cuda_preflight_failure(venv=tmp_path, onnx_path=Path("m.onnx"), config_path=Path("c.json"), output_dir=tmp_path, log_path=tmp_path / "build.log") is None
    assert not (tmp_path / "build_manifest.json").exists()


@pytest.mark.parametrize("role", ["manual_full", "manual_part1", "workflow_full"])
def test_valid_dxnn_cache_reuse_survives_incompatible_compiler(tmp_path, monkeypatch, role):
    from onnx_splitpoint_tool.gui import benchmark_workflow
    from onnx_splitpoint_tool.workflow import deepx_build_binding

    calibration = tmp_path / "calibration"
    calibration.mkdir()
    (calibration / "sample.jpg").write_bytes(b"image")
    source = tmp_path / "model.onnx"
    cache_root = tmp_path / "cache"
    environment = {"compiler_ready": True, "runtime_ready": True, "cache_dir": str(cache_root), "compiler_version": "unchanged-dxcom"}
    monkeypatch.delenv("ONNX_SPLITPOINT_ARTIFACT_POLICY", raising=False)
    monkeypatch.setattr(env_status, "inspect_deepx_environment", lambda **k: dict(environment))
    monkeypatch.setattr(deepx_build_binding, "inspect_deepx_environment", lambda **k: dict(environment))
    monkeypatch.setattr(deepx_build_binding, "_onnx_first_input_info", lambda path: ("images", [1, 3, 640, 640]))
    calls = []

    def fake_compile(**kwargs):
        assert environment["compiler_ready"], "incompatible cold compiler was dispatched"
        output = Path(kwargs["output_dir"])
        output.mkdir(parents=True, exist_ok=True)
        artifact = output / "compiled.dxnn"
        artifact.write_bytes(b"valuable existing DXNN")
        calls.append(artifact)
        return compiler.DeepXBuildResult(ok=True, status="ok", onnx_path=str(kwargs["onnx_path"]), config_path=str(kwargs["config_path"]), output_dir=str(output), dxnn_path=str(artifact))

    monkeypatch.setattr(compiler, "compile_dxnn", fake_compile)
    monkeypatch.setattr(deepx_build_binding, "compile_dxnn", fake_compile)

    def run(name, content=b"same ONNX content"):
        source.write_bytes(content)
        suite = tmp_path / name
        if role == "manual_part1":
            case = suite / "b024"
            case.mkdir(parents=True)
            (case / "part1.onnx").write_bytes(content)
            (case / "split_manifest.json").write_text(json.dumps({"part1_model": "part1.onnx"}))
            result = benchmark_workflow._materialize_manual_deepx_part1_artifacts(
                out_dir=suite, bench_plan_runs=[{"id": "deepx_m1_to_tensorrt", "type": "matrix", "stage1": "deepx_m1", "stage2": "tensorrt"}],
                validation_images="", fallback_calib_dir=str(calibration), calibration_num=1, task_hint="detection",
            )
            return result, result.get("cases", [{}])[0]
        if role == "manual_full":
            result = benchmark_workflow._materialize_manual_deepx_full_artifact(
                out_dir=suite, model_path=str(source), model=SimpleNamespace(graph=None), bench_plan_runs=[{"type": "deepx"}],
                validation_images="", validation_max_images=1, fallback_calib_dir=str(calibration), calibration_num=1, task_hint="detection",
            )
            return result, result
        suite.mkdir()
        result = deepx_build_binding.materialize_deepx_build_binding(
            run_dir=suite, model_id="model", model_path=str(source), row={"task": "detection", "input_shape": [1, 3, 640, 640]},
            profile_payload={"deepx_build": {"mode": "reuse_and_build_missing", "cache_dir": str(cache_root), "calibration_dir": str(calibration)}},
            targets=["deepx_m1"], benchmark_set_contract={"suite_dir": str(suite)},
        )
        status = json.loads(Path(result["artifacts"]["deepx_artifact_status_json"]).read_text())
        return result, status

    first, _ = run("run1")
    assert first["status"] == "ok"
    mismatch = env_status._cuda_architecture_status(_metadata())
    environment.update(compiler_ready=False, compiler_cuda_preflight=mismatch, hints=[env_status.compiler_cuda_preflight_message(mismatch)])
    second, cached = run("run2")
    assert second["status"] == "ok"
    assert cached["cache_lookup"]["outcome"] == "HIT"
    third, cold = run("run3", b"different ONNX content")
    assert third["status"] != "ok"
    assert cold["cache_lookup"]["outcome"] == "MISS"
    assert "deepx_compiler_cuda_architecture_unsupported" in json.dumps(cold)
    assert len(calls) == 1
