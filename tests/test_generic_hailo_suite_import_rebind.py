"""Historical Hailo requests become scoped, reuse-only current import calls."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from onnx_splitpoint_tool.workflow.deferred_hailo_builds import finalize_deferred_hailo_builds
from test_generic_quality_suite_import import _fixture, _import, _write


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prepared(tmp_path, backend="hailo8"):
    data = _fixture(tmp_path)
    run = data["profile"]["run_profiles"][0]
    run.update(id=backend + "_to_trt", stage1=backend,
               hardware_setup_id=backend + "_setup")
    data["profile"]["hailo_build"] = {
        "mode": "reuse_only", "force_build": False,
        "calib_count": 500, "calib_batch_size": 8, "optimization_level": 1,
    }
    case = data["source"] / "b001"
    case.mkdir(exist_ok=True)
    manifest = {"boundary": 1, "part1": "part1.onnx", "part2": "part2.onnx",
                "cut_tensors": ["cut_tensor"], "part1_cut_names": ["cut_tensor"]}
    _write(case / "split_manifest.json", manifest)
    (case / "part1.onnx").write_bytes(b"unchanged selected ONNX")
    out = case / "hailo" / backend / "part1"
    out.mkdir(parents=True)
    (out / "compiled.hef").write_bytes(b"historical verified HEF")
    receipt = out / "hailo_hef_build_receipt.json"
    receipt.write_bytes(b'{"historical_build_receipt":"never rewrite"}\n')
    config_source = run["generic_quality_completion_sources"]["b001"]
    config_path = Path(config_source["path"])
    config = json.loads(config_path.read_text())
    config["identity"]["setup_id"] = run["hardware_setup_id"]
    config["expected_runtime_artifacts"] = {"part1_runtime": {
        "sha256": _digest(out / "compiled.hef"),
        "size_bytes": (out / "compiled.hef").stat().st_size}}
    _write(config_path, config)
    config_source.update(sha256=_digest(config_path), size_bytes=config_path.stat().st_size)
    run["generic_quality_completion_config_sha256"]["b001"] = _digest(config_path)
    old = tmp_path / "retired_original_location"
    request = out / "deferred_hailo_build.json"
    _write(request, {"schema": "onnx-splitpoint/deferred-hailo-build/v1",
        "status": "completed", "source_onnx": str(old / "b001/part1.onnx"),
        "source_onnx_sha256": "sha256:" + _digest(case / "part1.onnx"),
        "build_observation": {"cache_hit": False, "compiler_dispatch_count": 1},
        "kwargs": {"outdir": str(old / "b001/hailo" / backend / "part1"),
            "hw_arch": backend, "net_name": "model_part1_b001", "force": False,
            "cache_only": False, "calib_count": 500, "calib_batch_size": 8,
            "opt_level": 1, "build_evidence_context": {
                "model_id": "model", "boundary": 1, "stage": "part1",
                "split_manifest": copy.deepcopy(manifest)}}, "environment": {}})
    data.update(request=request, receipt=receipt, run=run, config=config_path)
    return data


@pytest.mark.parametrize("backend", ["hailo8", "hailo10"])
def test_normal_import_preserves_history_and_stages_reuse_only_local_request(tmp_path, backend):
    data = _prepared(tmp_path, backend)
    saved_request, saved_receipt = data["request"].read_bytes(), data["receipt"].read_bytes()
    result = _import(data)
    request_path = result.suite_dir / data["request"].relative_to(data["source"])
    request = json.loads(request_path.read_text())
    assert request["status"] == "pending"
    assert "build_observation" not in request
    assert request["kwargs"]["cache_only"] is True and request["kwargs"]["force"] is False
    assert Path(request["kwargs"]["outdir"]) == request_path.parent
    assert Path(request["source_onnx"]) == result.suite_dir / "b001/part1.onnx"
    assert request["source_onnx_sha256"] == "sha256:" + _digest(Path(request["source_onnx"]))
    origin = request["import_source"]
    assert Path(origin["request"]["path"]) == data["request"]
    assert Path(origin["history"]["path"]).read_bytes() == saved_request
    assert origin["profile_payload"] == data["profile"]
    assert origin["bench_plan_runs"] == json.loads((result.suite_dir / "benchmark_plan.json").read_text())["runs"]
    assert data["request"].read_bytes() == saved_request
    assert data["receipt"].read_bytes() == saved_receipt
    assert (request_path.parent / data["receipt"].name).read_bytes() == saved_receipt


def test_rebound_request_reaches_ordinary_cache_builder_with_no_cold_authority(tmp_path):
    data = _prepared(tmp_path)
    result = _import(data)
    calls = []

    def cache_builder(source, **kwargs):
        assert Path(source).is_relative_to(result.suite_dir)
        assert Path(kwargs["outdir"]).is_relative_to(result.suite_dir)
        assert kwargs["cache_only"] is True and kwargs["force"] is False
        calls.append(source)
        return SimpleNamespace(ok=True, hef_path=str(Path(kwargs["outdir"]) / "compiled.hef"),
            error="", failure_kind="", details={"cache_hit": True, "compiler_dispatch_count": 0}, calib_info={})

    outcome = finalize_deferred_hailo_builds(model_dir=data["run_root"] / "models/model",
        run_dir=data["run_root"], profile_payload=data["profile"], build_fn=cache_builder)
    assert len(calls) == 1
    assert outcome["jobs"][0]["cache_hit"] is True
    assert outcome["jobs"][0]["compiler_dispatch_count"] == 0


@pytest.mark.parametrize("fault,match", [
    ("source", "source_onnx_changed"), ("hef", "runtime_artifact_changed"),
    ("cut", "cut_contract_changed"), ("outdir", "original_output_scope_invalid"),
    ("source_path", "original_source_scope_invalid"), ("backend", "backend_changed"),
    ("config_identity", "config_identity_changed"),
    ("compiler", "compiler_contract_changed"), ("policy", "requires_restrictive_build_policy"),
])
def test_import_rejects_changed_source_artifact_cut_identity_or_policy(tmp_path, fault, match):
    data = _prepared(tmp_path)
    request = json.loads(data["request"].read_text())
    if fault == "source": (data["source"] / "b001/part1.onnx").write_bytes(b"changed ONNX")
    elif fault == "hef": (data["request"].parent / "compiled.hef").write_bytes(b"different HEF")
    elif fault == "cut": request["kwargs"]["build_evidence_context"]["split_manifest"]["cut_tensors"] = ["other_cut"]
    elif fault == "outdir": request["kwargs"]["outdir"] = str(tmp_path / "unrelated")
    elif fault == "source_path": request["source_onnx"] = str(tmp_path / "unrelated.onnx")
    elif fault == "backend": request["kwargs"]["hw_arch"] = "hailo10"
    elif fault == "compiler": data["profile"]["hailo_build"]["optimization_level"] = 2
    elif fault == "policy": data["profile"]["hailo_build"]["mode"] = "build_missing"
    else:
        config = json.loads(data["config"].read_text())
        config["identity"]["case_id"] = "b020"
        _write(data["config"], config)
        data["run"]["generic_quality_completion_sources"]["b001"].update(
            sha256=_digest(data["config"]), size_bytes=data["config"].stat().st_size)
        data["run"]["generic_quality_completion_config_sha256"]["b001"] = _digest(data["config"])
    _write(data["request"], request)
    original = data["request"].read_bytes()
    with pytest.raises(ValueError, match=match):
        _import(data)
    assert data["request"].read_bytes() == original


def test_unselected_hailo_requests_and_receipts_are_only_copied(tmp_path):
    data = _prepared(tmp_path)
    unrelated = data["source"] / "b020/hailo/hailo8/part1/deferred_hailo_build.json"
    unrelated.parent.mkdir(parents=True)
    unrelated.write_bytes(b"old unrequested invalid JSON remains history\n")
    result = _import(data)
    assert (result.suite_dir / unrelated.relative_to(data["source"])).read_bytes() == unrelated.read_bytes()
