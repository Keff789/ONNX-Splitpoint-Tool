"""Imported DeepX requests retain source identity and never acquire build rights."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from onnx_splitpoint_tool.workflow import deferred_hailo_builds as deferred
from onnx_splitpoint_tool.workflow.artifact_cache_preflight import (
    collect_model_artifact_cache_probes,
    resolve_artifact_cache_preflight_policy,
)


IMPORTED_REQUEST = "deferred_deepx_part1_import_request.json"


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _sha(path):
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _run(cases):
    return {"id": "deepx_m1_to_tensorrt", "type": "mixed_backend",
            "stage1": "deepx_m1", "stage2": "tensorrt",
            "variants": ["composed"], "case_ids": list(cases)}


def _save_request(source, cases):
    for case in cases:
        _write(source / case / "split_manifest.json", {
            "boundary": int(case[1:]), "part1_model": "part1.onnx"})
        (source / case / "part1.onnx").write_bytes(f"synthetic-selected-{case}".encode())
    deferred.defer_deepx_part1_build(
        out_dir=source, bench_plan_runs=[_run(cases)],
        validation_images="/saved/calibration-images", fallback_calib_dir="/saved/fallback",
        calibration_num=37, calibration_manifest="/saved/calibration-manifest.json",
        strict_calibration_source=True, task_hint="detection",
        force_build=False, cache_only=False,
        build_config={"mode": "build_missing", "opt_level": 2,
                      "compiler_venv": "/saved/compiler-venv", "force_build": False},
        profile_payload={"profile_id": "historical-build-policy",
                         "deepx_build": {"mode": "build_missing", "force_build": False}},
    )


def _prepared(tmp_path):
    source = tmp_path / "historical_suite"
    cases = [f"b{i:03d}" for i in range(1, 21)]
    catalogue = [{"case_id": case, "case_dir": case, "boundary": i}
                 for i, case in enumerate(cases, 1)]
    _write(source / "benchmark_set.json", {"model_id": "model", "cases": catalogue})
    _save_request(source, cases)
    run_dir = tmp_path / "current_run"
    model_dir = run_dir / "models/model"
    suite = model_dir / "benchmark_set/legacy_suite"
    shutil.copytree(source, suite)
    runs = [_run(cases[:4])]
    runs[0].update({
        "quality_source_only": True, "hardware_setup_id": "deepx_m1_setup",
        "generic_quality_completion_configs": {
            case: f"quality_runtime/{case}.json" for case in cases[:4]},
        "generic_quality_completion_config_sha256": {case: "a" * 64 for case in cases[:4]},
        "generic_quality_completion_sources": {
            case: {"path": str(tmp_path / "configs" / f"{case}.json"),
                   "sha256": "a" * 64, "size_bytes": 2} for case in cases[:4]},
    })
    _write(suite / "benchmark_plan.json", {"model_id": "model", "runs": runs})
    profile = {"profile_id": "current-cache-policy", "run_profiles": runs,
               "deepx_build": {"mode": "reuse_only", "force_build": False},
               "quality_gate": {"statistics": {"execution_location": "central_management"}}}
    return SimpleNamespace(source=source, suite=suite, cases=cases, runs=runs,
                           profile=profile, run_dir=run_dir, model_dir=model_dir)


def _rebind(data):
    return deferred.rebind_imported_deepx_part1_request(
        data.source, data.suite, profile_payload=data.profile, bench_plan_runs=data.runs)


def _materializer(monkeypatch, calls, *, outcome="HIT"):
    from onnx_splitpoint_tool.gui import benchmark_workflow

    def materialize(**args):
        # The only compiler boundary in these tests. Any build-enabled replay
        # fails before synthetic status publication.
        assert args["cache_only"] is True
        assert args["force_build"] is False
        assert isinstance(args["out_dir"], Path)
        calls.append(args)
        for case in args["selected_case_dirs"]:
            _write(args["out_dir"] / case / "deepx/deepx_m1/part1/deepx_part1_artifact_status.json", {
                "status": "ready_reused" if outcome == "HIT" else "failed",
                "cache_lookup": {"outcome": outcome,
                                 "reason": "verified_synthetic_receipt" if outcome == "HIT"
                                 else "compiler_identity_unresolved"},
            })
        return {"status": "ok" if outcome == "HIT" else "failed",
                "failed_count": 0 if outcome == "HIT" else len(args["selected_case_dirs"])}

    monkeypatch.setattr(benchmark_workflow, "_materialize_manual_deepx_part1_artifacts", materialize)


@pytest.mark.parametrize("mode", ["reuse_only", "cache_verify_only"])
def test_relocated_request_probes_all_twenty_with_current_policy(tmp_path, monkeypatch, mode):
    data = _prepared(tmp_path)
    data.profile["deepx_build"]["mode"] = mode
    original = (data.source / deferred.DEEPX_PART1_REQUEST).read_bytes()
    saved = json.loads(original)
    source_bytes = {case: (data.source / case / "part1.onnx").read_bytes()
                    for case in data.cases}
    path = _rebind(data)
    assert path == data.suite / IMPORTED_REQUEST
    rebound = json.loads(path.read_text())
    assert rebound["kwargs"]["out_dir"] == str(data.suite)
    assert rebound["kwargs"]["selected_case_dirs"] == data.cases
    assert rebound["kwargs"]["cache_only"] is True
    assert rebound["kwargs"]["force_build"] is False
    assert rebound["kwargs"]["profile_payload"] == data.profile
    assert rebound["kwargs"]["bench_plan_runs"] == data.runs
    assert rebound["kwargs"]["build_config"]["mode"] == mode
    assert rebound["import_source"]["request"] == {
        "path": str(data.source / deferred.DEEPX_PART1_REQUEST),
        "sha256": _sha(data.source / deferred.DEEPX_PART1_REQUEST),
        "size_bytes": len(original)}
    assert rebound["source_onnx_sha256"] == {
        str(data.suite / case / "part1.onnx"): _sha(data.source / case / "part1.onnx")
        for case in data.cases}
    for key in ("validation_images", "fallback_calib_dir", "calibration_num",
                "calibration_manifest", "strict_calibration_source", "task_hint"):
        assert rebound["kwargs"][key] == saved["kwargs"][key]
    for key in ("opt_level", "compiler_venv"):
        assert rebound["kwargs"]["build_config"][key] == saved["kwargs"]["build_config"][key]
    calls = []
    _materializer(monkeypatch, calls)
    assert deferred.probe_deferred_deepx_part1_cache(data.suite)["status"] == "ok"
    assert len(calls) == 1
    assert calls[0]["out_dir"] == data.suite
    assert calls[0]["selected_case_dirs"] == data.cases
    assert calls[0]["profile_payload"] == data.profile
    assert json.loads((data.suite / "benchmark_plan.json").read_text())["runs"] == data.runs
    assert len(data.runs[0]["case_ids"]) == 4
    for suite in (data.source, data.suite):
        assert (suite / deferred.DEEPX_PART1_REQUEST).read_bytes() == original
        assert {case: (suite / case / "part1.onnx").read_bytes()
                for case in data.cases} == source_bytes


@pytest.mark.parametrize("fault", ["changed_onnx", "wrong_case", "path_escape", "missing_binding"])
def test_rebinding_rejects_changed_or_unbound_source_before_materializing(tmp_path, monkeypatch, fault):
    data = _prepared(tmp_path)
    original_path = data.source / deferred.DEEPX_PART1_REQUEST
    saved = json.loads(original_path.read_text())
    if fault == "changed_onnx":
        (data.suite / "b020/part1.onnx").write_bytes(b"changed copied ONNX")
    elif fault == "wrong_case":
        saved["kwargs"]["selected_case_dirs"][-1] = "b999"
    elif fault == "path_escape":
        outside = data.source.parent / "outside.onnx"
        outside.write_bytes(b"outside selected suite")
        for suite in (data.source, data.suite):
            _write(suite / "b020/split_manifest.json", {"part1_model": str(outside)})
        saved["source_onnx_sha256"].pop(str(data.source / "b020/part1.onnx"))
        saved["source_onnx_sha256"][str(outside)] = _sha(outside)
    else:
        saved["source_onnx_sha256"].pop(str(data.source / "b020/part1.onnx"))
    if fault != "changed_onnx":
        _write(original_path, saved)
        (data.suite / deferred.DEEPX_PART1_REQUEST).write_bytes(original_path.read_bytes())
    calls = []
    _materializer(monkeypatch, calls)
    with pytest.raises((ValueError, RuntimeError)):
        _rebind(data)
    assert not (data.suite / IMPORTED_REQUEST).exists()
    assert calls == []


@pytest.mark.parametrize("policy", [
    {"mode": "build_missing", "force_build": False},
    {"mode": "auto", "force_build": False},
    {"mode": "reuse_only", "force_build": True},
])
def test_rebinding_rejects_current_build_permission_or_force(tmp_path, policy):
    data = _prepared(tmp_path)
    data.profile["deepx_build"] = policy
    with pytest.raises((ValueError, RuntimeError)):
        _rebind(data)
    assert not (data.suite / IMPORTED_REQUEST).exists()


def test_explicit_compiler_recipe_disagreement_is_rejected(tmp_path):
    data = _prepared(tmp_path)
    data.profile["deepx_build"]["opt_level"] = 3
    with pytest.raises((ValueError, RuntimeError)):
        _rebind(data)
    assert not (data.suite / IMPORTED_REQUEST).exists()


@pytest.mark.parametrize("scope", ["hailo_only", "non_generic_deepx"])
def test_non_deepx_or_non_generic_import_does_not_derive_a_request(tmp_path, scope):
    data = _prepared(tmp_path)
    data.profile.pop("deepx_build")
    if scope == "hailo_only":
        data.runs[0].update(id="hailo8_to_trt", stage1="hailo8",
                            hardware_setup_id="hailo8_setup")
    else:
        data.runs[:] = [_run(data.cases[:4])]
    _write(data.suite / "benchmark_plan.json", {"model_id": "model", "runs": data.runs})
    original = (data.source / deferred.DEEPX_PART1_REQUEST).read_bytes()
    assert _rebind(data) is None
    assert not (data.suite / IMPORTED_REQUEST).exists()
    assert (data.source / deferred.DEEPX_PART1_REQUEST).read_bytes() == original
    assert (data.suite / deferred.DEEPX_PART1_REQUEST).read_bytes() == original


@pytest.mark.parametrize("consumer", ["probe", "finalizer"])
def test_historical_request_digest_is_checked_before_each_consumer(tmp_path, monkeypatch, consumer):
    data = _prepared(tmp_path)
    _rebind(data)
    origin = data.source / deferred.DEEPX_PART1_REQUEST
    saved = json.loads(origin.read_text())
    saved["kwargs"]["calibration_num"] += 1
    _write(origin, saved)
    calls = []
    _materializer(monkeypatch, calls)
    with pytest.raises(RuntimeError, match="source request changed"):
        if consumer == "probe":
            deferred.probe_deferred_deepx_part1_cache(data.suite)
        else:
            deferred.finalize_deferred_hailo_builds(
                model_dir=data.model_dir, run_dir=data.run_dir, profile_payload=data.profile)
    assert calls == []


@pytest.mark.parametrize("consumer", ["probe", "finalizer"])
@pytest.mark.parametrize("fault", ["onnx_and_digest", "selected_cases", "calibration", "compiler_recipe"])
def test_consumers_compare_derived_scientific_contract_with_original(tmp_path, monkeypatch, consumer, fault):
    data = _prepared(tmp_path)
    path = _rebind(data)
    original = (data.source / deferred.DEEPX_PART1_REQUEST).read_bytes()
    rebound = json.loads(path.read_text())
    if fault == "onnx_and_digest":
        onnx = data.suite / "b020/part1.onnx"
        onnx.write_bytes(b"coordinated changed ONNX and derived digest")
        rebound["source_onnx_sha256"][str(onnx)] = _sha(onnx)
    elif fault == "selected_cases":
        rebound["kwargs"]["selected_case_dirs"] = data.cases[:4]
        rebound["source_onnx_sha256"] = {
            str(data.suite / case / "part1.onnx"): _sha(data.suite / case / "part1.onnx")
            for case in data.cases[:4]}
    elif fault == "calibration":
        rebound["kwargs"]["calibration_num"] += 1
    else:
        rebound["kwargs"]["build_config"]["opt_level"] += 1
    _write(path, rebound)
    # The original provenance pin remains valid. Matching mutated local bytes
    # cannot replace the original request's scientific identity.
    assert rebound["import_source"]["request"]["sha256"] == _sha(
        data.source / deferred.DEEPX_PART1_REQUEST)
    calls = []
    _materializer(monkeypatch, calls)
    with pytest.raises((ValueError, RuntimeError)):
        if consumer == "probe":
            deferred.probe_deferred_deepx_part1_cache(data.suite)
        else:
            deferred.finalize_deferred_hailo_builds(
                model_dir=data.model_dir, run_dir=data.run_dir, profile_payload=data.profile)
    assert calls == []
    assert (data.source / deferred.DEEPX_PART1_REQUEST).read_bytes() == original
    assert (data.suite / deferred.DEEPX_PART1_REQUEST).read_bytes() == original


def test_unknown_probe_stays_unknown_for_all_twenty_in_normal_collector(tmp_path, monkeypatch):
    data = _prepared(tmp_path)
    _rebind(data)
    calls = []
    _materializer(monkeypatch, calls, outcome="UNKNOWN")
    assert deferred.probe_deferred_deepx_part1_cache(data.suite)["status"] == "failed"
    observations, _roles = collect_model_artifact_cache_probes(
        run_dir=data.run_dir, model_id="model", targets=["deepx_m1"],
        policy=resolve_artifact_cache_preflight_policy(data.profile))
    deepx = [row for row in observations if row.role == "deepx"]
    assert {row.item_id for row in deepx} == {f"{case}:part1" for case in data.cases}
    assert {row.status for row in deepx} == {"UNKNOWN"}
    assert {row.reason for row in deepx} == {"compiler_identity_unresolved"}
    assert calls[0]["selected_case_dirs"] == data.cases


def test_legacy_same_path_probe_keeps_original_request_and_policy(tmp_path, monkeypatch):
    data = _prepared(tmp_path)
    original = (data.source / deferred.DEEPX_PART1_REQUEST).read_bytes()
    calls = []
    _materializer(monkeypatch, calls)
    assert deferred.probe_deferred_deepx_part1_cache(data.source)["status"] == "ok"
    assert calls[0]["out_dir"] == data.source
    assert calls[0]["profile_payload"]["profile_id"] == "historical-build-policy"
    assert (data.source / deferred.DEEPX_PART1_REQUEST).read_bytes() == original
    assert not (data.source / IMPORTED_REQUEST).exists()


def test_imported_finalizer_remains_cache_only_and_preserves_legacy_bytes(tmp_path, monkeypatch):
    data = _prepared(tmp_path)
    path = _rebind(data)
    original = (data.suite / deferred.DEEPX_PART1_REQUEST).read_bytes()
    calls = []
    _materializer(monkeypatch, calls)
    result = deferred.finalize_deferred_hailo_builds(
        model_dir=data.model_dir, run_dir=data.run_dir, profile_payload=data.profile)
    assert result["status"] == "completed"
    assert len(calls) == 1
    assert calls[0]["selected_case_dirs"] == data.runs[0]["case_ids"]
    assert calls[0]["profile_payload"] == data.profile
    assert result["jobs"][0]["request"] == str(path)
    assert (data.suite / deferred.DEEPX_PART1_REQUEST).read_bytes() == original
    assert (data.source / deferred.DEEPX_PART1_REQUEST).read_bytes() == original
    assert json.loads(path.read_text())["kwargs"]["cache_only"] is True


@pytest.mark.parametrize("fault,reason", [
    ("cache_only", "build policy invalid"),
    ("out_dir", "suite identity mismatch"),
])
def test_finalizer_rejects_attempt_to_enable_build_in_imported_request(tmp_path, monkeypatch, fault, reason):
    data = _prepared(tmp_path)
    path = _rebind(data)
    original = (data.source / deferred.DEEPX_PART1_REQUEST).read_bytes()
    source_bytes = {case: (data.source / case / "part1.onnx").read_bytes()
                    for case in data.cases}
    rebound = json.loads(path.read_text())
    rebound["kwargs"][fault] = False if fault == "cache_only" else str(data.source)
    _write(path, rebound)
    calls = []
    _materializer(monkeypatch, calls)
    with pytest.raises(RuntimeError, match=reason):
        deferred.finalize_deferred_hailo_builds(
            model_dir=data.model_dir, run_dir=data.run_dir, profile_payload=data.profile)
    assert calls == []
    assert (data.source / deferred.DEEPX_PART1_REQUEST).read_bytes() == original
    assert (data.suite / deferred.DEEPX_PART1_REQUEST).read_bytes() == original
    assert {case: (data.source / case / "part1.onnx").read_bytes()
            for case in data.cases} == source_bytes


def test_actual_generic_import_rebinds_without_replacing_saved_request(tmp_path, monkeypatch):
    from test_generic_quality_suite_import import _fixture, _import

    data = _fixture(tmp_path)
    data["profile"]["deepx_build"] = {"mode": "reuse_only", "force_build": False}
    cases = [row["case_id"] for row in data["catalogue"]]
    _save_request(data["source"], cases)
    original = (data["source"] / deferred.DEEPX_PART1_REQUEST).read_bytes()
    imported = _import(data)
    assert (imported.suite_dir / IMPORTED_REQUEST).is_file()
    assert (imported.suite_dir / deferred.DEEPX_PART1_REQUEST).read_bytes() == original
    calls = []
    _materializer(monkeypatch, calls)
    assert deferred.probe_deferred_deepx_part1_cache(imported.suite_dir)["status"] == "ok"
    assert calls[0]["out_dir"] == imported.suite_dir
    assert calls[0]["selected_case_dirs"] == cases
    assert calls[0]["profile_payload"] == data["profile"]
    assert (data["source"] / deferred.DEEPX_PART1_REQUEST).read_bytes() == original


@pytest.mark.parametrize("source_has_derived_request", [False, True])
def test_actual_hailo_only_generic_import_needs_no_deepx_mode(tmp_path, source_has_derived_request):
    from test_generic_quality_suite_import import _fixture, _import
    from onnx_splitpoint_tool.runners.generic_quality_accounting import authorized_runs

    data = _fixture(tmp_path)
    data["profile"]["run_profiles"] = [data["profile"]["run_profiles"][0]]
    cases = [row["case_id"] for row in data["catalogue"]]
    _save_request(data["source"], cases)
    original = (data["source"] / deferred.DEEPX_PART1_REQUEST).read_bytes()
    if source_has_derived_request:
        prior = data["source"] / IMPORTED_REQUEST
        _write(prior, {"status": "completed", "synthetic_prior_import": True})
        prior_bytes = prior.read_bytes()
        history_name = prior.stem + "." + _sha(prior).removeprefix("sha256:") + ".json"
    imported = _import(data, targets=["hailo8", "tensorrt"])
    assert not (imported.suite_dir / IMPORTED_REQUEST).exists()
    assert (imported.suite_dir / deferred.DEEPX_PART1_REQUEST).read_bytes() == original
    assert (data["source"] / deferred.DEEPX_PART1_REQUEST).read_bytes() == original
    if source_has_derived_request:
        assert prior.read_bytes() == prior_bytes
        assert (imported.suite_dir / history_name).read_bytes() == prior_bytes
    runtime = json.loads(imported.artifacts["legacy_suite_benchmark_plan_json"].read_text())
    assert [row["id"] for row in authorized_runs(runtime, data["profile"])] == [
        data["profile"]["run_profiles"][0]["id"]]
