"""The finalizer preserves a strict warm contract after an earlier real HIT.

Only the outer compiler/cache leaf is controlled. Request persistence, context
matching, policy resolution, continuation and readiness projection are real.
The simulated cold branch writes only this test's temporary artifact.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from onnx_splitpoint_tool.workflow.deferred_hailo_builds import (
    REQUEST_NAME,
    cache_preflight_builder,
    finalize_deferred_hailo_builds,
)


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _strict_warm(*, cold=None):
    return {"artifact_cache_preflight": {
        "enabled": True, "default_expectation": "warm",
        "block_on_unexpected_cold_builds": True,
        "expected_cold": [] if cold is None else [cold],
    }}


def _prepared_hit(tmp_path, *, stage="part1", failure="cache_miss_blocked", cache_only=False):
    model = tmp_path / "models/demo"
    suite = model / "benchmark_set/legacy_suite"
    case = suite / "b001"
    output = (suite if stage == "full" else case) / "hailo/hailo8" / stage
    output.mkdir(parents=True)
    source = (suite if stage == "full" else case) / (stage + ".onnx")
    source.write_bytes(b"stable-selected-source")
    artifact = output / "existing.hef"
    artifact.write_bytes(b"validated-existing-cache")
    events = []

    def builder(selected_source, **kwargs):
        assert Path(selected_source) == source
        assert kwargs["force"] is False
        if artifact.exists():
            events.append("cache_hit")
            return SimpleNamespace(ok=True, hef_path=str(artifact), error="", failure_kind="",
                details={"cache_hit": True, "compiler_dispatch_count": 0}, calib_info={})
        if kwargs["cache_only"]:
            events.append("probe_" + failure)
            return SimpleNamespace(ok=False, hef_path=None, error=failure, failure_kind=failure,
                details={"cache_hit": False, "compiler_dispatch_count": 0}, calib_info={})
        # A concrete observable cold-dispatch boundary, confined to tmp_path.
        events.append("cold_dispatch")
        artifact.write_bytes(b"controlled-cold-result")
        return SimpleNamespace(ok=True, hef_path=str(artifact), error="", failure_kind="",
            details={"cache_hit": False, "compiler_dispatch_count": 1}, calib_info={})

    binding = {"model_id": "demo", "stage": stage}
    if stage != "full":
        binding["boundary"] = 1
        _write(case / "split_manifest.json", {"part1_model": "part1.onnx"})
    _write(suite / "benchmark_set.json", {
        "model_id": "demo", "cases": [] if stage == "full" else [{"case_dir": "b001", "boundary": 1}],
        "plan": {"runs": []},
    })
    result = cache_preflight_builder(builder)(source, outdir=str(output), hw_arch="hailo8",
        net_name="demo_" + stage, force=False, cache_only=cache_only,
        build_evidence_context=binding)
    assert result.ok and events == ["cache_hit"]
    request_path = output / REQUEST_NAME
    request = json.loads(request_path.read_text())
    assert request["cache_probe_status"] == "HIT"
    assert request["status"] == "pending"
    assert request["kwargs"]["cache_only"] is cache_only
    return model, artifact, request_path, events, builder


@pytest.mark.parametrize("stage", ["part1", "full"])
@pytest.mark.parametrize("failure", ["cache_miss_blocked", "probe_unavailable"])
def test_hit_then_miss_or_unknown_does_not_cold_dispatch_under_strict_warm(tmp_path, stage, failure):
    model, artifact, request_path, events, builder = _prepared_hit(tmp_path, stage=stage, failure=failure)
    artifact.unlink()  # The earlier HIT disappears before normal continuation.
    result = finalize_deferred_hailo_builds(model_dir=model, run_dir=tmp_path,
        profile_payload=_strict_warm(), build_fn=builder)
    assert events == ["cache_hit", "probe_" + failure]
    assert not artifact.exists()
    assert result["status"] == "partial"
    assert len(result["jobs"]) == 1
    assert result["jobs"][0]["status"] == "failed"
    assert result["jobs"][0]["compiler_dispatch_count"] == 0
    assert result["jobs"][0]["preflight_decision"] == "HIT"
    assert json.loads(request_path.read_text())["status"] == "failed"


def test_strict_warm_reuses_existing_hit_and_revalidates_completed_resume(tmp_path):
    model, artifact, request_path, events, builder = _prepared_hit(tmp_path)
    first = finalize_deferred_hailo_builds(model_dir=model, run_dir=tmp_path,
        profile_payload=_strict_warm(), build_fn=builder)
    assert first["status"] == "completed"
    assert events == ["cache_hit", "cache_hit"]
    assert json.loads(request_path.read_text())["status"] == "completed"
    artifact.unlink()
    second = finalize_deferred_hailo_builds(model_dir=model, run_dir=tmp_path,
        profile_payload=_strict_warm(), build_fn=builder)
    assert second["status"] == "partial"
    assert events == ["cache_hit", "cache_hit", "probe_cache_miss_blocked"]
    assert not artifact.exists()


@pytest.mark.parametrize("declaration,allowed", [
    ({"model_id": "demo", "role": "hailo8_hef", "item_id": "b001:part1"}, True),
    ({"model_id": "other", "role": "hailo8_hef", "item_id": "b001:part1"}, False),
    ({"model_id": "demo", "role": "hailo10_hef", "item_id": "b001:part1"}, False),
    ({"model_id": "demo", "role": "hailo8_hef", "item_id": "b002:part1"}, False),
    ({"model_id": "demo", "role": "hailo8_hef", "item_id": "b001:part2"}, False),
])
def test_only_exact_expected_cold_declaration_authorizes_continuation(tmp_path, declaration, allowed):
    model, artifact, _, events, builder = _prepared_hit(tmp_path)
    artifact.unlink()
    result = finalize_deferred_hailo_builds(model_dir=model, run_dir=tmp_path,
        profile_payload=_strict_warm(cold=declaration), build_fn=builder)
    assert events == ["cache_hit", "cold_dispatch" if allowed else "probe_cache_miss_blocked"]
    assert artifact.exists() is allowed
    assert result["status"] == ("completed" if allowed else "partial")
    assert result["jobs"][0]["compiler_dispatch_count"] == int(allowed)


@pytest.mark.parametrize("cache_only", [False, True])
def test_legacy_non_strict_continuation_keeps_original_build_policy(tmp_path, cache_only):
    model, artifact, _, events, builder = _prepared_hit(tmp_path, cache_only=cache_only)
    artifact.unlink()
    result = finalize_deferred_hailo_builds(model_dir=model, run_dir=tmp_path,
        profile_payload={}, build_fn=builder)
    assert events == ["cache_hit", "probe_cache_miss_blocked" if cache_only else "cold_dispatch"]
    assert artifact.exists() is not cache_only
    assert result["status"] == ("partial" if cache_only else "completed")


def test_explicitly_disabled_warm_policy_keeps_legacy_build_allowed(tmp_path):
    model, artifact, _, events, builder = _prepared_hit(tmp_path)
    artifact.unlink()
    profile = _strict_warm()
    profile["artifact_cache_preflight"]["enabled"] = False
    result = finalize_deferred_hailo_builds(model_dir=model, run_dir=tmp_path,
        profile_payload=profile, build_fn=builder)
    assert events == ["cache_hit", "cold_dispatch"]
    assert artifact.read_bytes() == b"controlled-cold-result"
    assert result["status"] == "completed"
    assert result["jobs"][0]["compiler_dispatch_count"] == 1
