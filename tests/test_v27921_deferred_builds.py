from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from onnx_splitpoint_tool.benchmark.services import (
    BenchmarkGenerationExecutionCallbacks, BenchmarkGenerationExecutionConfig,
    BenchmarkGenerationExecutionService, BenchmarkGenerationRuntime,
)
from onnx_splitpoint_tool.workflow.deferred_hailo_builds import (
    REQUEST_NAME, cache_preflight_builder, finalize_deferred_hailo_builds,
)


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _result(ok=False, hef_path=None):
    return SimpleNamespace(ok=ok, hef_path=hef_path, error="" if ok else "cache_miss_blocked", details={}, calib_info={}, failure_kind="cache_miss_blocked")


def test_actual_case_loop_probes_then_continues_only_frozen_selection(tmp_path, monkeypatch):
    from onnx_splitpoint_tool import api as asc

    phases = []
    monkeypatch.setattr(asc, "cut_tensors_for_boundary", lambda *a, **k: ["cut"])
    def split(*a, **k):
        phases.append("split")
        return object(), object(), {"cut_tensors": ["cut"]}
    monkeypatch.setattr(asc, "split_model_on_cut_tensors", split)
    monkeypatch.setattr(asc, "save_model", lambda m, p: Path(p).write_bytes(b"onnx"))
    monkeypatch.setattr(asc, "write_runner_skeleton_onnxruntime", lambda d, **k: str(Path(d) / "runner.py"))
    model = tmp_path / "models" / "yolo26m"
    suite = model / "benchmark_set" / "legacy_suite"
    suite.mkdir(parents=True)
    runtime = BenchmarkGenerationRuntime(
        out_dir=suite, bench_log_path=suite / "benchmark.log",
        state_path=suite / "generation_state.json", requested_cases=1,
        ranked_candidates=[398, 400], candidate_search_pool=[398, 400],
        model_name="unit", model_source="unit.onnx", hef_full_policy="end",
    )
    def builder(source, **kwargs):
        if kwargs["cache_only"]:
            phases.append("cache_probe")
            return _result()
        assert phases[-1] == "matrix_saved"
        phases.append("compile")
        hef = Path(kwargs["outdir"]) / "compiled.hef"
        hef.write_bytes(b"hef")
        return _result(True, str(hef))
    runs = [{"id": "hailo10_to_tensorrt", "type": "matrix", "stage1": {"type": "hailo", "hw_arch": "hailo10"}, "stage2": {"type": "onnxruntime", "provider": "tensorrt"}, "variants": ["composed"]}]
    cfg = BenchmarkGenerationExecutionConfig(
        runtime=runtime, target_cases=1, gap=0, ranked_candidates=[398, 400],
        candidate_search_pool=[398, 400], out_dir=suite, base="unit", pad=3,
        strict_boundary=False, model=object(), nodes=[], order=[], analysis_payload={},
        bench_plan_runs=runs, hef_targets=["hailo10"], hef_part1=True,
        hailo_build_hef_fn=cache_preflight_builder(builder),
        require_complete_hailo_matrix_per_case=True, defer_hailo_builds=True,
    )
    callbacks = BenchmarkGenerationExecutionCallbacks(
        log=lambda *a, **k: None, queue_put=lambda *a: None,
        persist_state=lambda **k: None, publish_hailo_diagnostics=lambda *a, **k: None,
        predicted_metrics_for_boundary=lambda *a: {},
        hailo_parse_entry_for_boundary=lambda *a: None, hailo_parse_scalar_fields=lambda *a: {},
    )
    assert BenchmarkGenerationExecutionService().execute_case_build_loop(cfg, callbacks) == [398]
    assert phases == ["split", "cache_probe"]
    assert not runtime.discarded_cases
    assert not (suite / "b400").exists()
    _write(suite / "benchmark_set.json", {"cases": runtime.cases, "plan": {"runs": runs}})
    _write(model / "benchmark_set" / "benchmark_set.json", {"cases": runtime.cases})
    phases.append("matrix_saved")
    result = finalize_deferred_hailo_builds(model_dir=model, run_dir=tmp_path, profile_payload={}, build_fn=builder)
    assert result["status"] == "completed"
    assert phases == ["split", "cache_probe", "matrix_saved", "compile"]
    case = json.loads((suite / "b398" / "split_manifest.json").read_text())
    assert case["hailo"]["hefs"]["hailo10"]["part1"].endswith("compiled.hef")
    formal = json.loads((model / "benchmark_set" / "benchmark_set.json").read_text())
    assert formal["cases"][0]["hailo_case_variant_availability"]["hailo10h"]["part1"] is True


def _prepared_request(tmp_path, *, force=False, cache_only=False):
    model = tmp_path / "models" / "demo"
    suite = model / "benchmark_set" / "legacy_suite"
    output = suite / "b001" / "hailo" / "hailo8" / "part1"
    output.mkdir(parents=True)
    source = suite / "b001" / "part1.onnx"
    source.write_bytes(b"selected-onnx")
    cache_preflight_builder(lambda source, **args: _result())(
        source, outdir=str(output), hw_arch="hailo8", net_name="demo_part1_b1",
        force=force, cache_only=cache_only,
    )
    _write(suite / "benchmark_set.json", {"cases": [{"case_dir": "b001", "boundary": 1}], "plan": {"runs": []}})
    _write(suite / "b001" / "split_manifest.json", {})
    return model, suite, source, output


def test_changed_source_blocks_compiler(tmp_path):
    model, suite, source, output = _prepared_request(tmp_path)
    source.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="changed after cache preflight"):
        finalize_deferred_hailo_builds(model_dir=model, run_dir=tmp_path, profile_payload={}, build_fn=lambda *a, **k: pytest.fail("compiler called"))


def test_unselected_requests_are_never_built(tmp_path):
    model, suite, source, output = _prepared_request(tmp_path)
    _write(suite / "benchmark_set.json", {"cases": [], "plan": {"runs": []}})
    result = finalize_deferred_hailo_builds(model_dir=model, run_dir=tmp_path, profile_payload={}, build_fn=lambda *a, **k: pytest.fail("compiler called"))
    assert result["jobs"] == []


@pytest.mark.parametrize("force,cache_only", [(True, False), (False, True)])
def test_original_force_and_smoke_cache_policy_survive_probe(tmp_path, force, cache_only):
    model, suite, source, output = _prepared_request(tmp_path, force=force, cache_only=cache_only)
    observed = []
    def builder(source, **kwargs):
        observed.append((kwargs["force"], kwargs["cache_only"]))
        return _result()
    if force:
        from onnx_splitpoint_tool.build_dispatch_policy import ProductiveForceBuildDisabled
        # v34 preserves the old request for evidence but forbids replaying
        # its Force flag into a productive continuation.
        with pytest.raises(ProductiveForceBuildDisabled):
            finalize_deferred_hailo_builds(model_dir=model, run_dir=tmp_path, profile_payload={}, build_fn=builder)
        assert observed == []
    else:
        finalize_deferred_hailo_builds(model_dir=model, run_dir=tmp_path, profile_payload={}, build_fn=builder)
        assert observed == [(force, cache_only)]


def test_build_environment_is_restored(tmp_path, monkeypatch):
    monkeypatch.setenv("ONNX_SPLITPOINT_HAILO_CACHE_ROOT", "captured-cache")
    model, suite, source, output = _prepared_request(tmp_path)
    monkeypatch.setenv("ONNX_SPLITPOINT_HAILO_CACHE_ROOT", "outer-cache")
    import os
    def builder(source, **kwargs):
        assert os.environ["ONNX_SPLITPOINT_HAILO_CACHE_ROOT"] == "captured-cache"
        raise RuntimeError("build failure")
    with pytest.raises(RuntimeError, match="build failure"):
        finalize_deferred_hailo_builds(model_dir=model, run_dir=tmp_path, profile_payload={}, build_fn=builder)
    assert os.environ["ONNX_SPLITPOINT_HAILO_CACHE_ROOT"] == "outer-cache"


def test_full_cold_raw_fallback_keeps_endpoint_contract(tmp_path, monkeypatch):
    from onnx_splitpoint_tool.benchmark import model_preparation
    model = tmp_path / "models" / "yolo11l"
    suite = model / "benchmark_set" / "legacy_suite"
    output = suite / "hailo" / "hailo8" / "full"
    output.mkdir(parents=True)
    source = suite / "full.onnx"
    source.write_bytes(b"full-onnx")
    phases = []
    def builder(source, **kwargs):
        phases.append((bool(kwargs.get("cache_only")), tuple(kwargs.get("end_node_names") or [])))
        if kwargs.get("cache_only") or not kwargs.get("end_node_names"):
            return _result()
        hef = Path(kwargs["outdir"]) / "compiled.hef"
        hef.write_bytes(b"raw-head")
        return _result(True, str(hef))
    cache_preflight_builder(builder)(source, outdir=str(output), hw_arch="hailo8", net_name="yolo11l_full", cache_only=False, force=False)
    assert phases == [(True, ())]
    monkeypatch.setattr(model_preparation, "infer_yolo_raw_detection_head_end_nodes", lambda *a: ["cv2/Conv", "cv3/Conv"])
    _write(suite / "benchmark_set.json", {"cases": [], "plan": {"runs": []}})
    result = finalize_deferred_hailo_builds(model_dir=model, run_dir=tmp_path, profile_payload={}, build_fn=builder)
    assert result["status"] == "completed"
    assert phases == [(True, ()), (False, ()), (True, ("cv2/Conv", "cv3/Conv")), (False, ("cv2/Conv", "cv3/Conv"))]
    full = json.loads((suite / "benchmark_set.json").read_text())["hailo"]["hefs"]["hailo8"]
    assert full["full_output_contract"]["requires_external_postprocess"] is True
    assert full["full_end_node_names"] == ["cv2/Conv", "cv3/Conv"]
    assert full["full_true_full_model"] is False


def test_completed_hailo_revalidates_cache_on_resume(tmp_path):
    model, suite, source, output = _prepared_request(tmp_path)
    calls = []
    def builder(source, **kwargs):
        calls.append(kwargs.get("force"))
        hef = output / "compiled.hef"
        hef.write_bytes(b"rebuilt")
        return _result(True, str(hef))
    finalize_deferred_hailo_builds(model_dir=model, run_dir=tmp_path, profile_payload={}, build_fn=builder)
    (output / "compiled.hef").unlink()
    finalize_deferred_hailo_builds(model_dir=model, run_dir=tmp_path, profile_payload={}, build_fn=builder)
    assert len(calls) == 2
    assert (output / "compiled.hef").is_file()


def test_deepx_continuation_preserves_selected_sources_and_reports_failure(tmp_path, monkeypatch):
    from onnx_splitpoint_tool.workflow.deferred_hailo_builds import defer_deepx_part1_build
    from onnx_splitpoint_tool.gui import benchmark_workflow
    model = tmp_path / "models" / "demo"
    suite = model / "benchmark_set" / "legacy_suite"
    source = suite / "b001" / "part1.onnx"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"selected-source")
    _write(suite / "b001" / "split_manifest.json", {"part1_model": "part1.onnx"})
    _write(suite / "benchmark_set.json", {"cases": [{"case_dir": "b001"}], "plan": {"runs": []}})
    defer_deepx_part1_build(out_dir=suite, bench_plan_runs=[], validation_images="", fallback_calib_dir="")
    calls = []
    def materialize(**args):
        assert isinstance(args["out_dir"], Path)
        assert args["selected_case_dirs"] == ["b001"]
        calls.append(args)
        return {"status": "failed", "failed_count": 1}
    monkeypatch.setattr(benchmark_workflow, "_materialize_manual_deepx_part1_artifacts", materialize)
    result = finalize_deferred_hailo_builds(model_dir=model, run_dir=tmp_path, profile_payload={})
    assert result["status"] == "partial"
    assert len(calls) == 1
    source.write_bytes(b"changed-source")
    with pytest.raises(RuntimeError, match="Selected DeepX ONNX changed"):
        finalize_deferred_hailo_builds(model_dir=model, run_dir=tmp_path, profile_payload={})
    assert len(calls) == 1


def test_gate_selection_probe_report_precedes_compiler(tmp_path):
    from onnx_splitpoint_tool.workflow.deferred_hailo_builds import selection_preflight_builder
    output = tmp_path / "b398" / "hailo" / "hailo10" / "part1"
    phases = []
    def builder(source, **kwargs):
        if kwargs["cache_only"]:
            phases.append("probe")
            return _result()
        report = json.loads((output / "selection_probe_cache_preflight.json").read_text())
        assert report["scope"] == "selection_probe"
        assert report["boundary"] == "b398"
        assert report["status"] == "MISS"
        phases.append("compile")
        return _result(True)
    wrapped = selection_preflight_builder(builder, model_id="yolo26m", profile_payload={})
    assert wrapped("source.onnx", outdir=str(output), hw_arch="hailo10h", net_name="yolo26m_part1_b398", cache_only=False, force=False).ok
    assert phases == ["probe", "compile"]


@pytest.mark.parametrize("failure", ["cache_miss_blocked", "probe_unavailable"])
def test_gate_strict_warm_blocks_cold_or_unknown_without_compiler(tmp_path, failure):
    from onnx_splitpoint_tool.workflow.deferred_hailo_builds import selection_preflight_builder
    from onnx_splitpoint_tool.cache_verify_policy import CacheVerifyPolicyError
    def builder(source, **kwargs):
        assert kwargs["cache_only"] is True
        result = _result()
        result.failure_kind = failure
        return result
    wrapped = selection_preflight_builder(builder, model_id="yolo26m", profile_payload={"artifact_cache_preflight": {"require_warm_cache": True}})
    with pytest.raises(CacheVerifyPolicyError, match="cache_preflight_selection_blocked"):
        wrapped("source.onnx", outdir=str(tmp_path), hw_arch="hailo10h", net_name="yolo26m_part1_b398", cache_only=False)
    assert json.loads((tmp_path / "selection_probe_cache_preflight.json").read_text())["compiler_dispatch_allowed"] is False


def test_gate_compiler_identity_unavailable_is_unknown(tmp_path):
    from onnx_splitpoint_tool.workflow.deferred_hailo_builds import selection_preflight_builder
    from onnx_splitpoint_tool.cache_verify_policy import CacheVerifyPolicyError
    def builder(source, **kwargs):
        assert kwargs["cache_only"]
        result = _result()
        result.unsupported_reason = "compiler_identity_unavailable"
        result.details = {"compiler_identity_available": False}
        return result
    wrapped = selection_preflight_builder(builder, model_id="demo", profile_payload={"artifact_cache_preflight": {"strict": True}})
    with pytest.raises(CacheVerifyPolicyError):
        wrapped("source.onnx", outdir=str(tmp_path), hw_arch="hailo8", net_name="demo_part1_b1", cache_only=False)
    report = json.loads((tmp_path / "selection_probe_cache_preflight.json").read_text())
    assert report["status"] == "UNKNOWN"
    assert report["expected_cold_build"] is False


@pytest.mark.parametrize("warm", [False, True])
def test_global_cache_verify_continuation_never_compiles(tmp_path, warm):
    model, suite, source, output = _prepared_request(tmp_path)
    calls = []
    def builder(source, **kwargs):
        assert kwargs["cache_only"] is True
        assert kwargs["force"] is False
        calls.append("probe")
        if warm:
            hef = output / "compiled.hef"
            hef.write_bytes(b"restored-validated-cache")
            return _result(True, str(hef))
        return _result()
    if warm:
        result = finalize_deferred_hailo_builds(model_dir=model, run_dir=tmp_path, profile_payload={"execution_guard": {"mode": "cache_verify_only"}}, build_fn=builder)
        assert result["status"] == "completed"
    else:
        from onnx_splitpoint_tool.cache_verify_policy import CacheVerifyPolicyError
        with pytest.raises(CacheVerifyPolicyError, match="cache_miss_blocked"):
            finalize_deferred_hailo_builds(model_dir=model, run_dir=tmp_path, profile_payload={"execution_guard": {"mode": "cache_verify_only"}}, build_fn=builder)
    assert calls == ["probe"]
