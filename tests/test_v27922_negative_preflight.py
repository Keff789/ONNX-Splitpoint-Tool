from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from onnx_splitpoint_tool.cache_verify_policy import CacheVerifyPolicyError
from onnx_splitpoint_tool.workflow.artifact_cache_preflight import (
    build_artifact_cache_preflight,
    collect_model_artifact_cache_probes,
    known_negative_build_evidence,
    render_artifact_cache_preflight_log_lines,
    resolve_artifact_cache_preflight_policy,
    write_artifact_cache_preflight,
)
from onnx_splitpoint_tool.workflow.deferred_hailo_builds import (
    REQUEST_NAME,
    cache_preflight_builder,
    finalize_deferred_hailo_builds,
    selection_preflight_builder,
    refresh_deferred_hailo_negative_probes,
)


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _negative(state="COMPILE_INFEASIBLE"):
    return {
        "status": "HIT", "reusable": True, "negative_evidence_hit": True,
        "state": state, "reason": "exact negative evidence: concat22 / Agent infeasible",
        "cache_key_v3": "exact-existing-backend-key",
        "evidence_origin": "/preserved/build_evidence/old_run.json",
    }


def _result(*, negative=False, ok=False, hef_path=None):
    return SimpleNamespace(
        ok=ok, hef_path=hef_path, skipped=True,
        error="known negative" if negative else "cache_miss_blocked",
        details={"build_evidence": _negative()} if negative else {},
        calib_info={}, failure_kind="known_negative_build_evidence" if negative else "cache_miss_blocked",
    )


def _suite(root):
    model = root / "models" / "yolo26s"
    formal = model / "benchmark_set"
    suite = formal / "legacy_suite"
    output = suite / "b364" / "hailo" / "hailo8" / "part1"
    output.mkdir(parents=True)
    source = suite / "b364" / "part1.onnx"
    source.write_bytes(b"frozen-selected-part")
    _write(suite / "b364" / "split_manifest.json", {"part1_model": "part1.onnx"})
    _write(suite / "benchmark_set.json", {"cases": [{"case_id": "b364", "case_dir": "b364", "boundary": 364}], "plan": {"runs": []}})
    _write(formal / "hailo_artifact_service_plan.json", {
        "case_hef_requests": [{"case_id": "b364", "stage": "part1", "backend": "hailo8", "status": "known_infeasible"}],
    })
    return model, suite, output, source


@pytest.mark.parametrize("force", [False, True])
def test_exact_negative_is_terminal_and_never_deferred_or_force_retried(tmp_path, force):
    model, suite, output, source = _suite(tmp_path)
    calls = []
    def builder(source, **kwargs):
        calls.append(kwargs)
        assert kwargs["cache_only"] is True
        return _result(negative=True)
    cache_preflight_builder(builder)(source, outdir=str(output), hw_arch="hailo8", net_name="yolo26s_part1_b364", force=force, cache_only=False)
    request = json.loads((output / REQUEST_NAME).read_text())
    assert request["status"] == "known_infeasible"
    assert request["build_evidence"]["state"] == "COMPILE_INFEASIBLE"
    outcome = finalize_deferred_hailo_builds(model_dir=model, run_dir=tmp_path, profile_payload={}, build_fn=builder)
    assert len(calls) == 2
    assert all(call["cache_only"] is True for call in calls)
    assert outcome["jobs"][0]["status"] == "known_infeasible"
    assert outcome["jobs"][0]["compiler_dispatched"] is False
    assert json.loads((suite / "benchmark_set.json").read_text())["cases"][0]["case_id"] == "b364"


def test_negative_matrix_is_distinct_and_does_not_globally_block_other_work(tmp_path):
    model, suite, output, source = _suite(tmp_path)
    cache_preflight_builder(lambda *a, **k: _result(negative=True))(
        source, outdir=str(output), hw_arch="hailo8", net_name="yolo26s_part1_b364", force=True,
    )
    observations, roles = collect_model_artifact_cache_probes(
        run_dir=tmp_path, model_id="yolo26s", targets=["hailo8"],
        policy=resolve_artifact_cache_preflight_policy({}),
    )
    report = build_artifact_cache_preflight(
        model_ids=["yolo26s"], observations=observations,
        applicable_roles={"yolo26s": list(roles)}, block_on_unexpected_cold_builds=True,
    )
    row = report["artifact_matrix"][0]
    assert row["status"] == "KNOWN_INFEASIBLE"
    assert row["boundary"] == "b364" and row["artifact_stage"] == "part1"
    assert row["identity"] == "exact-existing-backend-key"
    assert row["evidence_origin"] == "/preserved/build_evidence/old_run.json"
    assert row["compiler_dispatch_allowed"] is False
    assert row["runtime_artifact_available"] is False
    assert report["runtime_dispatch_allowed"] is True
    assert report["cold_builds_required"] == report["hit_count"] == report["unknown_count"] == 0
    assert report["known_infeasible_count"] == 1
    assert report["status"] == "known_infeasible_artifacts"
    assert report["matrix"][0]["cells"]["hailo8_hef"]["status"] == "KNOWN_INFEASIBLE"
    text = "\n".join(render_artifact_cache_preflight_log_lines(report))
    assert "known_infeasible: model=yolo26s boundary=b364 backend=hailo8" in text
    assert "evidence_origin=/preserved/build_evidence/old_run.json" in text
    assert "expected_cold_build:" not in text
    paths = write_artifact_cache_preflight(report, output_dir=tmp_path / "reports")
    with paths["artifact_cache_preflight_items_csv"].open() as handle:
        assert list(csv.DictReader(handle))[0]["evidence_origin"] == row["evidence_origin"]
    assert "KNOWN_INFEASIBLE" in paths["artifact_cache_preflight_md"].read_text()


@pytest.mark.parametrize("policy", [{}, {"artifact_cache_preflight": {"require_warm_cache": True}}, {"execution_guard": {"mode": "cache_verify_only"}}])
def test_gate_probe_returns_negative_without_compiler_or_campaign_exception(tmp_path, policy):
    output = tmp_path / "b364" / "hailo" / "hailo8" / "part1"
    calls = []
    def builder(source, **kwargs):
        calls.append(kwargs)
        assert kwargs["cache_only"] is True
        return _result(negative=True)
    result = selection_preflight_builder(builder, model_id="yolo26s", profile_payload=policy)(
        "model.onnx", outdir=str(output), hw_arch="hailo8", net_name="yolo26s_part1_b364", force=True,
    )
    assert not result.ok
    assert len(calls) == 1
    report = json.loads((output / "selection_probe_cache_preflight.json").read_text())
    assert report["status"] == "KNOWN_INFEASIBLE"
    assert report["expected_cold_build"] is False
    assert report["compiler_dispatch_allowed"] is False


def test_changed_endpoint_fresh_probe_can_become_cold_and_continue(tmp_path):
    model, suite, output, source = _suite(tmp_path)
    phases = []
    def builder(source, **kwargs):
        nodes = kwargs.get("end_node_names", [])
        phases.append((list(nodes), kwargs.get("cache_only")))
        if not nodes:
            return _result(negative=True)
        if kwargs.get("cache_only"):
            return _result()
        hef = output / "compiled.hef"
        hef.write_bytes(b"new-endpoint-artifact")
        return _result(ok=True, hef_path=str(hef))
    wrapped = cache_preflight_builder(builder)
    common = dict(outdir=str(output), hw_arch="hailo8", net_name="yolo26s_part1_b364", cache_only=False)
    wrapped(source, **common)
    assert json.loads((output / REQUEST_NAME).read_text())["status"] == "known_infeasible"
    wrapped(source, **common, end_node_names=["different_endpoint"])
    assert json.loads((output / REQUEST_NAME).read_text())["status"] == "pending"
    outcome = finalize_deferred_hailo_builds(model_dir=model, run_dir=tmp_path, profile_payload={}, build_fn=builder)
    assert outcome["status"] == "completed"
    assert phases == [([], True), (["different_endpoint"], True), (["different_endpoint"], False)]


def test_fresh_probe_supersedes_stale_negative_side_file(tmp_path):
    model, suite, output, source = _suite(tmp_path)
    _write(output / "hailo_negative_evidence.json", {"hw_arch": "hailo8", "net_name": "yolo26s_part1_b364", "build_evidence": _negative()})
    cache_preflight_builder(lambda *a, **k: _result())(source, outdir=str(output), hw_arch="hailo8", net_name="yolo26s_part1_b364")
    observations, _ = collect_model_artifact_cache_probes(
        run_dir=tmp_path, model_id="yolo26s", targets=["hailo8"],
        policy=resolve_artifact_cache_preflight_policy({}),
    )
    assert observations[0].status != "KNOWN_INFEASIBLE"


def test_failure_text_without_exact_reusable_decision_never_becomes_negative():
    for patch in ({"status": "MISS"}, {"reusable": False}, {"negative_evidence_hit": False}, {"state": "TRANSIENT_INFRASTRUCTURE"}):
        decision = _negative() | patch
        assert known_negative_build_evidence({"build_evidence": decision}) == {}
    assert known_negative_build_evidence({"error": "Agent infeasible", "status": "COMPILE_INFEASIBLE"}) == {}


def test_mixed_matrix_keeps_expected_cold_work_eligible_with_known_negative():
    report = build_artifact_cache_preflight(
        model_ids=["yolo26s"], applicable_roles={"yolo26s": ["hailo8", "hailo10h"]},
        block_on_unexpected_cold_builds=True, observations=[
            {"model_id": "yolo26s", "role": "hailo8", "item_id": "b364:part1",
             "status": "KNOWN_INFEASIBLE", "reason": "COMPILE_INFEASIBLE", "expectation": "warm"},
            {"model_id": "yolo26s", "role": "hailo10h", "item_id": "b364:part1",
             "status": "MISS", "reason": "exact_cache_artifact_missing", "expectation": "cold"},
        ],
    )
    assert report["runtime_dispatch_allowed"] is True
    assert report["known_infeasible_count"] == report["cold_builds_required"] == 1
    assert [row["backend"] for row in report["cold_build_rows"]] == ["hailo10h"]


def test_continuation_negative_learned_since_matrix_does_not_abort_cache_verify(tmp_path):
    model, suite, output, source = _suite(tmp_path)
    cache_preflight_builder(lambda *a, **k: _result())(
        source, outdir=str(output), hw_arch="hailo8", net_name="yolo26s_part1_b364", cache_only=True,
    )
    result = finalize_deferred_hailo_builds(
        model_dir=model, run_dir=tmp_path, profile_payload={"execution_guard": {"mode": "cache_verify_only"}},
        build_fn=lambda *a, **k: _result(negative=True),
    )
    assert result["jobs"][0]["status"] == "known_infeasible"
    assert json.loads((output / REQUEST_NAME).read_text())["status"] == "known_infeasible"


def test_full_fallback_negative_probe_never_retries_raw_identity(tmp_path, monkeypatch):
    from onnx_splitpoint_tool.benchmark import model_preparation
    model, suite, output, source = _suite(tmp_path)
    output = suite / "hailo" / "hailo8" / "full"
    calls = []
    def builder(source, **kwargs):
        calls.append((kwargs.get("cache_only"), tuple(kwargs.get("end_node_names") or [])))
        return _result(negative=bool(kwargs.get("end_node_names")))
    cache_preflight_builder(builder)(source, outdir=str(output), hw_arch="hailo8", net_name="yolo26s_full", cache_only=False)
    monkeypatch.setattr(model_preparation, "infer_yolo_raw_detection_head_end_nodes", lambda *a: ["raw_head"])
    result = finalize_deferred_hailo_builds(model_dir=model, run_dir=tmp_path, profile_payload={}, build_fn=builder)
    assert result["jobs"][0]["status"] == "known_infeasible"
    assert calls == [(True, ()), (False, ()), (True, ("raw_head",))]


def test_terminal_negative_rescued_hef_reprobes_and_restores_on_resume(tmp_path):
    model, suite, output, source = _suite(tmp_path)
    cache_preflight_builder(lambda *a, **k: _result(negative=True))(
        source, outdir=str(output), hw_arch="hailo8", net_name="yolo26s_part1_b364",
    )
    calls = []
    def rescued(source, **kwargs):
        calls.append(kwargs)
        assert kwargs["cache_only"] is True and kwargs["force"] is False
        hef = output / "compiled.hef"
        hef.write_bytes(b"rescued-and-backend-verified")
        return _result(ok=True, hef_path=str(hef))
    result = finalize_deferred_hailo_builds(model_dir=model, run_dir=tmp_path, profile_payload={}, build_fn=rescued)
    assert len(calls) == 1
    assert result["jobs"][0]["status"] == "completed"
    saved = json.loads((output / REQUEST_NAME).read_text())
    assert saved["cache_probe_status"] == "HIT"
    manifest = json.loads((suite / "b364" / "split_manifest.json").read_text())
    assert manifest["hailo"]["hefs"]["hailo8"]["part1"].endswith("compiled.hef")


@pytest.mark.parametrize("change", ["sdk", "calibration", "full_source"])
def test_changed_negative_identity_is_refreshed_before_matrix_and_cold_build(tmp_path, change):
    model, suite, output, source = _suite(tmp_path)
    active_identity = {"sdk": "old", "calibration": "old", "full_source": "old"}
    phases = []
    def builder(source, **kwargs):
        phases.append((kwargs.get("cache_only"), dict(active_identity)))
        if active_identity[change] == "old":
            return _result(negative=True)
        if kwargs.get("cache_only"):
            return _result()
        hef = output / "compiled.hef"
        hef.write_bytes(b"different-identity-build")
        return _result(ok=True, hef_path=str(hef))
    cache_preflight_builder(builder)(source, outdir=str(output), hw_arch="hailo8", net_name="yolo26s_part1_b364", cache_only=False)
    active_identity[change] = "new"
    refreshed = refresh_deferred_hailo_negative_probes(model_dir=model, build_fn=builder)
    assert refreshed["jobs"][0]["status"] == "MISS"
    assert all(probe for probe, identity in phases)
    observations, roles = collect_model_artifact_cache_probes(
        run_dir=tmp_path, model_id="yolo26s", targets=["hailo8"], policy=resolve_artifact_cache_preflight_policy({}),
    )
    report = build_artifact_cache_preflight(model_ids=["yolo26s"], observations=observations, applicable_roles={"yolo26s": list(roles)})
    assert report["known_infeasible_count"] == 0 and report["cold_builds_required"] == 1
    write_artifact_cache_preflight(report, output_dir=tmp_path / "refreshed-matrix")
    result = finalize_deferred_hailo_builds(model_dir=model, run_dir=tmp_path, profile_payload={}, build_fn=builder)
    assert result["jobs"][0]["status"] == "completed"
    assert [probe for probe, identity in phases] == [True, True, False]


@pytest.mark.parametrize("profile", [{}, {"execution_guard": {"mode": "cache_verify_only"}}])
def test_changed_negative_after_matrix_demands_refresh_before_any_compiler(tmp_path, profile):
    model, suite, output, source = _suite(tmp_path)
    cache_preflight_builder(lambda *a, **k: _result(negative=True))(
        source, outdir=str(output), hw_arch="hailo8", net_name="yolo26s_part1_b364", cache_only=False,
    )
    calls = []
    def miss(source, **kwargs):
        calls.append(kwargs)
        assert kwargs["cache_only"] is True and kwargs["force"] is False
        return _result()
    with pytest.raises((RuntimeError, CacheVerifyPolicyError), match="cache_preflight_refresh_required"):
        finalize_deferred_hailo_builds(model_dir=model, run_dir=tmp_path, profile_payload=profile, build_fn=miss)
    assert len(calls) == 1
    saved = json.loads((output / REQUEST_NAME).read_text())
    assert saved["status"] == "pending"
    assert saved["cache_preflight_refresh_required"] is True
    with pytest.raises((RuntimeError, CacheVerifyPolicyError), match="cache_preflight_refresh_required"):
        finalize_deferred_hailo_builds(model_dir=model, run_dir=tmp_path, profile_payload=profile, build_fn=miss)
    assert len(calls) == 1


def test_unavailable_identity_refresh_does_not_inherit_old_negative_or_force_cold(tmp_path):
    model, suite, output, source = _suite(tmp_path)
    cache_preflight_builder(lambda *a, **k: _result(negative=True))(
        source, outdir=str(output), hw_arch="hailo8", net_name="yolo26s_part1_b364", force=True,
    )
    def unavailable(*args, **kwargs):
        assert kwargs["cache_only"] is True
        return SimpleNamespace(ok=False, error="compiler_identity_unavailable", failure_kind="probe_unavailable",
                               details={"compiler_identity_available": False})
    refresh_deferred_hailo_negative_probes(model_dir=model, build_fn=unavailable)
    observations, _ = collect_model_artifact_cache_probes(
        run_dir=tmp_path, model_id="yolo26s", targets=["hailo8"], policy=resolve_artifact_cache_preflight_policy({}),
    )
    assert observations[0].status == "UNKNOWN"
    assert observations[0].reason == "probe_unavailable"
