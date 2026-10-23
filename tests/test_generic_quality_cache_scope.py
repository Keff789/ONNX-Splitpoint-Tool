"""Copied catalogue artifacts do not enlarge authorized Generic capture work."""
from __future__ import annotations

from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from onnx_splitpoint_tool.workflow import artifact_cache_preflight as cache
from test_generic_deepx_deferred_import import _prepared, _write


def _fixture(tmp_path, monkeypatch, backend="hailo8", selected_status="HIT"):
    data = _prepared(tmp_path)
    data.runs[0].update(id=f"{backend}_to_trt", stage1=backend, type="matrix",
                        hardware_setup_id=f"{backend}_setup")
    _write(data.suite / "benchmark_plan.json", {"runs": data.runs})
    data.profile["artifact_cache_preflight"] = {"require_warm_cache": True}
    selected = set(data.runs[0]["case_ids"])
    service = {"case_hef_requests": [], "full_baseline_requests": []}
    for arch in ("hailo8", "hailo10"):
        service["full_baseline_requests"].append({"backend": arch, "requested": True,
                                                  "status": "pending"})
        for case in data.cases:
            for stage in ("part1", "part2"):
                outcome = selected_status if arch == backend and case in selected and stage == "part1" else "UNKNOWN"
                path = data.suite / case / "hailo" / arch / stage / "compiled.hef"
                if outcome == "HIT":
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b"synthetic cached HEF")
                service["case_hef_requests"].append({
                    "backend": arch, "case_id": case, "stage": stage,
                    "status": "cache_miss" if outcome == "MISS" else "pending",
                    "hef_path": str(path)})
    data.service_path = data.model_dir / "benchmark_set/hailo_artifact_service_plan.json"
    _write(data.service_path, service)
    for case in data.cases:
        for stage in ("part1", "part2"):
            outcome = selected_status if backend == "deepx_m1" and case in selected and stage == "part1" else "UNKNOWN"
            _write(data.suite / case / "deepx" / f"deepx_{stage}_artifact_status.json", {
                "status": "pending", "cache_lookup": {"outcome": outcome,
                                                             "reason": "synthetic_cache_probe"}})
    _write(data.model_dir / "benchmark_set/deepx/deepx_artifact_status.json", {"status": "pending"})
    import onnx_splitpoint_tool.hailo_backend as hailo
    monkeypatch.setattr(hailo, "_load_valid_hailo_receipt", lambda path: {
        "hw_arch": path.parent.parent.name, "cache_key": "synthetic-verified-key"})
    data.remote = [{"model_id": "model", "role": cache.ROLE_TRT_P2,
                    "item_id": f"{backend}_setup/{case}", "status": "HIT",
                    "reason": "synthetic_current_remote_receipt"} for case in sorted(selected)]
    return data


def _collect(data, *, profile=None):
    return cache.collect_model_artifact_cache_probes(
        run_dir=data.run_dir, model_id="model",
        targets=["hailo8", "hailo10", "deepx_m1", "tensorrt"],
        policy=cache.resolve_artifact_cache_preflight_policy(data.profile),
        profile_payload=data.profile if profile is None else profile,
        remote_trt_observations=data.remote)


def _report(data, **kwargs):
    rows, roles = _collect(data, **kwargs)
    return cache.build_artifact_cache_preflight(
        model_ids=["model"], observations=rows, applicable_roles={"model": sorted(roles)},
        default_expectation="warm", block_on_unexpected_cold_builds=True)


@pytest.mark.parametrize("backend,role", [("hailo8", cache.ROLE_HAILO8),
                                         ("hailo10", cache.ROLE_HAILO10),
                                         ("deepx_m1", cache.ROLE_DEEPX)])
def test_only_current_backend_case_stage_is_required(tmp_path, monkeypatch, backend, role):
    data = _fixture(tmp_path, monkeypatch, backend)
    originals = {path: path.read_bytes() for path in data.model_dir.rglob("*.json")}
    report = _report(data)
    assert report["runtime_dispatch_allowed"] is True
    assert report["unknown_count"] == report["confirmed_miss_count"] == 0
    assert {(row["role"], row["item_id"]) for row in report["artifact_matrix"]} == (
        {(role, f"{case}:part1") for case in data.runs[0]["case_ids"]}
        | {(cache.ROLE_TRT_P2, row["item_id"]) for row in data.remote})
    for path, content in originals.items():
        assert path.read_bytes() == content


@pytest.mark.parametrize("backend", ["hailo8", "hailo10", "deepx_m1"])
@pytest.mark.parametrize("outcome", ["UNKNOWN", "MISS"])
def test_required_unknown_and_miss_remain_blocking(tmp_path, monkeypatch, backend, outcome):
    data = _fixture(tmp_path, monkeypatch, backend, outcome)
    report = _report(data)
    assert report["runtime_dispatch_allowed"] is False
    selected = [row for row in report["artifact_matrix"] if row["role"] != cache.ROLE_TRT_P2]
    assert {row["status"] for row in selected} == {outcome}
    assert {row["item_id"] for row in selected} == {f"{case}:part1" for case in data.runs[0]["case_ids"]}


def test_missing_selected_probe_is_not_hidden(tmp_path, monkeypatch):
    data = _fixture(tmp_path, monkeypatch)
    service = json.loads(data.service_path.read_text())
    service["case_hef_requests"] = [row for row in service["case_hef_requests"]
                                    if not (row["backend"] == "hailo8" and row["case_id"] == "b001")]
    _write(data.service_path, service)
    report = _report(data)
    assert report["runtime_dispatch_allowed"] is False
    missing = next(row for row in report["artifact_matrix"] if row["item_id"] == "b001:part1")
    assert missing["status"] == "UNKNOWN"
    assert missing["reason"] == "required_artifact_cache_probe_missing"


@pytest.mark.parametrize("fault", ["missing_plan", "plan_drift", "outside_catalogue", "unbound_runtime"])
def test_unresolved_scope_rejects_instead_of_granting_hits(tmp_path, monkeypatch, fault):
    data = _fixture(tmp_path, monkeypatch)
    plan_path = data.suite / "benchmark_plan.json"
    plan = json.loads(plan_path.read_text())
    if fault == "missing_plan":
        plan_path.unlink()
    elif fault == "plan_drift":
        plan["runs"][0]["hardware_setup_id"] = "unbound_setup"
        _write(plan_path, plan)
    elif fault == "outside_catalogue":
        contract = json.loads((data.suite / "benchmark_set.json").read_text())
        contract["cases"] = contract["cases"][1:]
        _write(data.suite / "benchmark_set.json", contract)
    else:
        plan["runs"].append({"id": "unbound", "type": "matrix", "stage1": "tensorrt", "stage2": "deepx_m1"})
        _write(plan_path, plan)
    with pytest.raises((ValueError, FileNotFoundError)):
        _collect(data)


def test_actual_unrequested_cold_build_still_blocks(tmp_path, monkeypatch):
    data = _fixture(tmp_path, monkeypatch)
    service = json.loads(data.service_path.read_text())
    row = next(row for row in service["case_hef_requests"]
               if row["backend"] == "hailo10" and row["case_id"] == "b020" and row["stage"] == "part2")
    row.update(status="ready_built", build_attempt={"compiler_dispatch_count": 1})
    path = data.suite / "b020/hailo/hailo10/part2/compiled.hef"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"synthetic actual preparation build")
    _write(data.service_path, service)
    report = _report(data)
    assert report["runtime_dispatch_allowed"] is False
    assert report["unexpected_cold_builds"] == 1
    built = next(row for row in report["artifact_matrix"] if row["role"] == cache.ROLE_HAILO10)
    assert built["status"] == "MISS" and built["evidence"]["current_build"] is True


def test_explicit_current_full_reference_remains_required(tmp_path, monkeypatch):
    data = _fixture(tmp_path, monkeypatch)
    reference = {"id": "hailo10_full", "type": "full", "provider": "hailo10",
                 "variants": ["full"], "semantic_reference_only": True}
    data.profile["run_profiles"].append(reference)
    _write(data.suite / "benchmark_plan.json", {"runs": data.profile["run_profiles"]})
    report = _report(data)
    assert report["runtime_dispatch_allowed"] is False
    full = next(row for row in report["artifact_matrix"] if row["role"] == cache.ROLE_HAILO10)
    assert full["item_id"] == "full" and full["status"] == "UNKNOWN"


@pytest.mark.parametrize("kind", ["legacy", "native", "energy"])
def test_other_workflows_keep_conservative_catalogue_checks(tmp_path, monkeypatch, kind):
    data = _fixture(tmp_path, monkeypatch)
    profile = {} if kind == "legacy" else deepcopy(data.profile)
    if kind in {"native", "energy"}:
        profile["run_profiles"].append({"id": kind, "type": "onnxruntime", "provider": "cpu"})
    report = _report(data, profile=profile)
    assert report["runtime_dispatch_allowed"] is False
    assert any(row["role"] == cache.ROLE_DEEPX and row["status"] == "UNKNOWN"
               for row in report["artifact_matrix"])


@pytest.mark.parametrize("plan_drift", [False, True])
def test_normal_preflight_supplies_current_profile_authority(tmp_path, monkeypatch, plan_drift):
    from test_v27920_artifact_cache_preflight_runner import _bare_runner

    data = _fixture(tmp_path, monkeypatch)
    if plan_drift:
        plan = json.loads((data.suite / "benchmark_plan.json").read_text())
        plan["runs"][0]["hardware_setup_id"] = "unbound_setup"
        _write(data.suite / "benchmark_plan.json", plan)
    runner = _bare_runner()
    runner.run_dir = data.run_dir
    runner.profile_payload = data.profile
    runner.options = SimpleNamespace(benchmark_execution_backend="local")
    runner.outputs, runner.report_paths = {}, []
    runner.log = runner._emit_log = lambda _message: None
    runner._targets = lambda: ["hailo8"]
    runner._materialize_local_cache_preflight_inputs = lambda *_args: []
    original = cache._trt_observations
    monkeypatch.setattr(cache, "_trt_observations", lambda **kwargs: original(**{
        **kwargs, "remote_observations": data.remote}))
    artifacts, metrics, _message, status = runner._stage_artifact_cache_preflight([{"id": "model"}])
    report = json.loads(artifacts["artifact_cache_preflight_json"].read_text())
    if plan_drift:
        assert status == "failed" and metrics["runtime_dispatch_allowed"] is False
        assert report["unknown_count"] > 0
        assert any("profile_plan_drift" in row["error"] for row in report["collection_errors"])
    else:
        assert status == "ok" and metrics["runtime_dispatch_allowed"] is True
        assert report["unknown_count"] == 0 and report["collection_errors"] == []
