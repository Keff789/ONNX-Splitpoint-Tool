from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from onnx_splitpoint_tool.hailo_build_context import make_build_evidence_context
from onnx_splitpoint_tool.benchmark.services import (
    BenchmarkGenerationExecutionCallbacks, BenchmarkGenerationExecutionConfig,
    BenchmarkGenerationExecutionService, BenchmarkGenerationRuntime,
)
from onnx_splitpoint_tool.workflow.deferred_hailo_builds import cache_preflight_builder


def _negative():
    return SimpleNamespace(
        ok=False, skipped=True, timed_out=False, hef_path=None,
        failure_kind="known_negative_build_evidence", unsupported_reason="COMPILE_INFEASIBLE",
        error="exact negative build evidence: COMPILE_INFEASIBLE", elapsed_s=0,
        details={"build_evidence": {"status": "HIT", "state": "COMPILE_INFEASIBLE",
                 "reusable": True, "reason": "exact_deterministic_outcome",
                 "evidence_origin": {"source": "earlier_run"}}}, calib_info={},
    )


def test_context_is_real_source_identity_and_frozen_contract(tmp_path):
    source = tmp_path / "full.onnx"
    source.write_bytes(b"full model version one")
    manifest = {"boundary": 364, "cut_tensors": ["cut"], "hailo": {"part2_output_strategy": "raw"}}
    first = make_build_evidence_context(source, stage="part2", split_manifest=manifest)
    manifest["cut_tensors"].append("later")
    assert first["split_manifest"]["cut_tensors"] == ["cut"]
    assert first["full_source_onnx_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    source.write_bytes(b"full model version two")
    second = make_build_evidence_context(source, stage="part2", split_manifest=manifest)
    assert second["full_source_onnx_sha256"] != first["full_source_onnx_sha256"]
    full = make_build_evidence_context(source, stage="full")
    assert "boundary" not in full
    assert full["split_manifest"] == {}


def test_windows_context_defers_admission_without_losing_actual_source(tmp_path, monkeypatch):
    from onnx_splitpoint_tool import build_evidence
    source = tmp_path / "full.onnx"
    source.write_bytes(b"actual full source")
    def unsupported(*args, **kwargs):
        raise build_evidence.BuildEvidenceError("nofollow_platform_unsupported")
    monkeypatch.setattr(build_evidence, "_read_regular_nofollow", unsupported)
    context = make_build_evidence_context(source, stage="part1", split_manifest={"boundary": 364, "cut_tensors": ["cut"]})
    assert context["full_source_onnx_path"] == str(source)
    assert context["source_identity_deferred_to_compiler"] is True
    assert context["split_manifest"]["boundary"] == 364
    assert "identity_error" not in context
    assert "full_source_onnx_sha256" not in context


@pytest.mark.parametrize("stage,manifest", [("part1", {}), ("part2", {"boundary": True}), ("bad", {})])
def test_incomplete_context_is_unavailable_not_fabricated(tmp_path, stage, manifest):
    source = tmp_path / "full.onnx"
    source.write_bytes(b"model")
    context = make_build_evidence_context(source, stage=stage, split_manifest=manifest)
    assert context["identity_error"]
    assert "full_source_onnx_sha256" not in context


def test_normal_case_loop_carries_context_and_keeps_negative_selected_case(tmp_path, monkeypatch):
    from onnx_splitpoint_tool import api as asc

    monkeypatch.setattr(asc, "cut_tensors_for_boundary", lambda *a, **k: ["cut"])
    monkeypatch.setattr(asc, "split_model_on_cut_tensors", lambda *a, **k: (object(), object(), {"cut_tensors": ["cut"]}))
    monkeypatch.setattr(asc, "save_model", lambda model, path: Path(path).write_bytes(b"selected split"))
    monkeypatch.setattr(asc, "write_runner_skeleton_onnxruntime", lambda directory, **k: str(Path(directory) / "runner.py"))
    source = tmp_path / "yolo26s.onnx"
    source.write_bytes(b"full yolo26s")
    suite = tmp_path / "benchmark_set" / "legacy_suite"
    suite.mkdir(parents=True)
    runtime = BenchmarkGenerationRuntime(
        out_dir=suite, bench_log_path=suite / "benchmark.log", state_path=suite / "state.json",
        requested_cases=1, ranked_candidates=[364, 365], candidate_search_pool=[364, 365],
        model_name="unit", model_source=str(source), hef_full_policy="end",
    )
    captured = []
    def builder(path, **kwargs):
        assert kwargs["cache_only"] is True, "negative probe must never dispatch compiler"
        captured.append(kwargs["build_evidence_context"])
        return _negative()
    cfg = BenchmarkGenerationExecutionConfig(
        runtime=runtime, target_cases=1, gap=0, ranked_candidates=[364, 365],
        candidate_search_pool=[364, 365], out_dir=suite, base="unit", pad=3,
        full_model_src=str(source), strict_boundary=False, model=object(), nodes=[], order=[],
        analysis_payload={}, hef_targets=["hailo8"], hef_part1=True, hef_part2=True,
        hailo_build_hef_fn=cache_preflight_builder(builder), defer_hailo_builds=True,
    )
    logs = []
    callbacks = BenchmarkGenerationExecutionCallbacks(
        log=lambda message, **kwargs: logs.append(message), queue_put=lambda *a: None,
        persist_state=lambda **k: None, publish_hailo_diagnostics=lambda *a, **k: None,
        predicted_metrics_for_boundary=lambda *a: {}, hailo_parse_entry_for_boundary=lambda *a: None,
        hailo_parse_scalar_fields=lambda *a: {},
    )
    assert BenchmarkGenerationExecutionService().execute_case_build_loop(cfg, callbacks) == [364]
    assert {context["stage"] for context in captured} == {"part1", "part2"}
    assert all(context["boundary"] == 364 and context["full_source_onnx_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest() for context in captured)
    assert not (suite / "b365").exists()
    assert not runtime.discarded_cases
    manifest = json.loads((suite / "b364/split_manifest.json").read_text())
    for stage in ("part1", "part2"):
        summary = manifest["hailo"]["hefs"]["hailo8"][f"{stage}_build"]
        assert summary["negative_evidence_hit"] is True
        assert summary["context_mode"] == "known_infeasible"
        assert summary["cache_hit"] is False
    assert any("KNOWN_INFEASIBLE" in line for line in logs)


def test_gate_shared_negative_skips_parser_without_old_index_binding():
    from test_v2783_hailo8_first_feasibility import _Builder, _run, _target_outcome
    calls = []
    def behavior(target, cache_only):
        assert cache_only, "known negative must not spend a cold-build reservation"
        result = _target_outcome(target, ok=False)
        if target == "hailo8":
            result["target_output"]["part1_build"]["build_evidence"] = _negative().details["build_evidence"]
        return result
    builder = _Builder(behavior)
    result = _run(builder, parser=lambda *a, **k: pytest.fail("known negative reparsed"))
    assert result["candidate_receipt"]["target_outcomes"]["hailo8"] == "COMPILE_INFEASIBLE"
    assert result["state"]["cold_builds"] == 0


def test_gate_parser_failure_is_persisted_and_reused_on_fresh_run(tmp_path, monkeypatch):
    from test_v2783_hailo8_first_feasibility import _Builder, _run, _target_outcome
    from test_v2783_build_evidence import _base_key_kwargs
    from onnx_splitpoint_tool.build_evidence import canonical_build_key
    from onnx_splitpoint_tool.build_evidence_store import BuildEvidenceStore

    monkeypatch.setenv("ONNX_SPLITPOINT_BUILD_EVIDENCE_ROOT", str(tmp_path / "evidence"))
    key = canonical_build_key(**_base_key_kwargs())
    def behavior(target, cache_only):
        assert cache_only
        result = _target_outcome(target, ok=False)
        if target == "hailo8":
            info = BuildEvidenceStore().lookup(key).as_dict()
            info["key"] = key
            result["target_output"]["part1_build"]["build_evidence"] = info
        return result
    first = _run(_Builder(behavior), parser=lambda *a, **k: SimpleNamespace(ok=False, error="UnsupportedShuffleLayerError", elapsed_s=0.1))
    assert first["candidate_receipt"]["target_outcomes"]["hailo8"] == "PARSER_UNSUPPORTED"
    assert BuildEvidenceStore().lookup(key).state == "PARSER_UNSUPPORTED"
    second = _run(_Builder(behavior), parser=lambda *a, **k: pytest.fail("persisted parser failure retried"))
    assert second["candidate_receipt"]["target_outcomes"]["hailo8"] == "PARSER_UNSUPPORTED"
    assert second["state"]["cold_builds"] == 0


def test_runner_refreshes_saved_negatives_before_materializing_matrix_inputs(tmp_path, monkeypatch):
    from onnx_splitpoint_tool.workflow import runner as runner_module
    from onnx_splitpoint_tool.workflow import deferred_hailo_builds as deferred

    runner = object.__new__(runner_module.EvaluationWorkflowRunner)
    runner.run_dir = tmp_path
    runner.options = SimpleNamespace()
    runner.log = lambda message: None
    bdir = tmp_path / "models/yolo26s/benchmark_set"
    bdir.mkdir(parents=True)
    contract = bdir / "benchmark_set.json"
    contract.write_text(json.dumps({"probe": "historical_negative"}))
    calls = []

    def refresh(**kwargs):
        assert kwargs["model_dir"] == bdir.parent
        calls.append("cache_only_refresh")
        contract.write_text(json.dumps({"probe": "current_identity"}))
        return {"status": "refreshed", "jobs": []}

    def materialize(**kwargs):
        assert calls[0] == "cache_only_refresh"
        assert kwargs["benchmark_set_contract"]["probe"] == "current_identity"
        calls.append("materialize")

    monkeypatch.setattr(deferred, "refresh_deferred_hailo_negative_probes", refresh)
    monkeypatch.setattr(runner_module, "materialize_backend_artifact_decisions", materialize)
    monkeypatch.setattr(runner_module, "materialize_hailo_artifact_service_binding", materialize)
    assert runner._materialize_local_cache_preflight_inputs([{"id": "yolo26s"}], ["hailo8"]) == []
    assert calls == ["cache_only_refresh", "materialize", "materialize"]
