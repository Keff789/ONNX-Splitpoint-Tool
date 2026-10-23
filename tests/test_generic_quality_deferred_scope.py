"""Generic continuation consumes current rows before opening historical requests."""
from __future__ import annotations

import builtins
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from onnx_splitpoint_tool.workflow import deferred_hailo_builds as deferred
from test_generic_deepx_deferred_import import _prepared, _rebind, _write


def _finalize(data, **kwargs):
    return deferred.finalize_deferred_hailo_builds(
        model_dir=data.model_dir, run_dir=data.run_dir,
        profile_payload=data.profile, **kwargs)


def _hailo_scope(data):
    data.runs[0].update(id="hailo8_to_trt", stage1="hailo8", hardware_setup_id="hailo8_setup")
    _write(data.suite / "benchmark_plan.json", {"model_id": "model", "runs": data.runs})


def _result(output=None):
    artifact = output / "compiled.hef" if output is not None else None
    if artifact is not None:
        artifact.write_bytes(b"synthetic verified cached HEF")
    return SimpleNamespace(
        ok=artifact is not None, hef_path=str(artifact) if artifact is not None else None,
        error="" if artifact is not None else "cache_miss_blocked",
        failure_kind="" if artifact is not None else "cache_miss_blocked",
        details={"cache_hit": artifact is not None, "compiler_dispatch_count": 0}, calib_info={})


def _hailo_request(data, case="b001", backend="hailo8"):
    output = data.suite / case / "hailo" / backend / "part1"
    deferred.cache_preflight_builder(lambda *a, **k: _result())(
        data.suite / case / "part1.onnx", outdir=str(output), hw_arch=backend,
        net_name=f"model_{case}_part1", cache_only=True, force=False,
        build_evidence_context={"model_id": "model", "boundary": int(case[1:]), "stage": "part1"})
    return output / deferred.REQUEST_NAME


def _unrequested(data, *, deepx_only):
    paths = [
        data.suite / "b001/hailo/hailo10/part1" / deferred.REQUEST_NAME,
        data.suite / "b001/hailo/hailo8/part2" / deferred.REQUEST_NAME,
        data.suite / "hailo/hailo8/full" / deferred.REQUEST_NAME,
        data.suite / "hailo/hailo10/full" / deferred.REQUEST_NAME,
        data.suite / "b001" / deferred.DEEPX_PART2_REQUEST,
        data.suite / "b020" / deferred.DEEPX_PART2_REQUEST,
    ]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"unrequested historical invalid JSON\n")
    if deepx_only:
        # This is the original failure: an accepted catalogue case has an
        # unrequested H8 request still naming its historical output directory.
        old = data.suite / "b003/hailo/hailo8/part1" / deferred.REQUEST_NAME
        _write(old, {"status": "pending", "kwargs": {
            "outdir": str(data.source / "b003/hailo/hailo8/part1")}})
        paths.append(old)
    else:
        for name in (deferred.DEEPX_PART1_REQUEST, deferred.DEEPX_PART1_IMPORT_REQUEST):
            path = data.suite / name
            path.write_bytes(b"unrequested historical DeepX invalid JSON\n")
            paths.append(path)
    return {path: path.read_bytes() for path in paths}


def _forbid_reads(monkeypatch, paths):
    original = deferred._read

    def read(path):
        assert Path(path) not in paths, f"unrequested request was read: {path}"
        return original(path)

    monkeypatch.setattr(deferred, "_read", read)


def _forbid_sdk_imports(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        assert not any(name == token or name.endswith("." + token)
                       for token in ("deepx.compiler", "deepx.activation_proxy", "hailo_sdk_client")), name
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)


def _assert_unchanged(saved):
    for path, original in saved.items():
        assert path.read_bytes() == original


def test_deepx_preflight_keeps_twenty_but_normal_finalizer_consumes_only_current_four(tmp_path, monkeypatch):
    from onnx_splitpoint_tool.gui import benchmark_workflow

    data = _prepared(tmp_path)
    imported = _rebind(data)
    untouched = _unrequested(data, deepx_only=True)
    _forbid_reads(monkeypatch, untouched)
    _forbid_sdk_imports(monkeypatch)
    monkeypatch.setattr(benchmark_workflow, "resolve_hailo_benchmark_helpers",
                        lambda **k: pytest.fail("unrequested Hailo builder resolved"))
    calls = []

    def materialize(**args):
        assert args["cache_only"] is True and args["force_build"] is False
        assert args["out_dir"] == data.suite
        calls.append(list(args["selected_case_dirs"]))
        return {"status": "ok", "failed_count": 0, "cases": []}

    monkeypatch.setattr(benchmark_workflow, "_materialize_manual_deepx_part1_artifacts", materialize)
    assert deferred.probe_deferred_deepx_part1_cache(data.suite)["status"] == "ok"
    result = _finalize(data)
    assert calls == [data.cases, data.runs[0]["case_ids"]]
    assert result["status"] == "completed"
    assert [row["request"] for row in result["jobs"]] == [str(imported)]
    assert len(json.loads(imported.read_text())["source_onnx_sha256"]) == 20
    assert len(json.loads(imported.read_text())["kwargs"]["selected_case_dirs"]) == 20
    _assert_unchanged(untouched)


def test_hailo_only_generic_finalizer_skips_other_backends_full_and_deepx_loader(tmp_path, monkeypatch):
    from onnx_splitpoint_tool.gui import benchmark_workflow

    data = _prepared(tmp_path)
    _hailo_scope(data)
    requested = [_hailo_request(data, case) for case in data.runs[0]["case_ids"]]
    untouched = _unrequested(data, deepx_only=False)
    _forbid_reads(monkeypatch, untouched)
    _forbid_sdk_imports(monkeypatch)
    monkeypatch.setattr(deferred, "_deepx_part1_request_path",
                        lambda *a: pytest.fail("unrequested DeepX Part1 loader entered"))
    calls = []
    resolutions = []

    def cached(source, **args):
        assert args["cache_only"] is True and args["force"] is False
        calls.append((args["hw_arch"], args["build_evidence_context"]["boundary"]))
        return _result(Path(args["outdir"]))

    def helpers(**kwargs):
        resolutions.append(kwargs)
        return SimpleNamespace(hailo_build_hef_fn=cached)

    monkeypatch.setattr(benchmark_workflow, "resolve_hailo_benchmark_helpers", helpers)
    result = _finalize(data)
    assert result["status"] == "completed"
    assert calls == [("hailo8", i) for i in range(1, 5)]
    assert len(resolutions) == 1
    assert {row["request"] for row in result["jobs"]} == {str(path) for path in requested}
    _assert_unchanged(untouched)


@pytest.mark.parametrize("fault", ["outdir", "arch"])
def test_requested_hailo_identity_mismatch_still_rejects(tmp_path, monkeypatch, fault):
    data = _prepared(tmp_path)
    _hailo_scope(data)
    path = _hailo_request(data)
    request = json.loads(path.read_text())
    if fault == "outdir":
        request["kwargs"]["outdir"] = str(data.source / "b001/hailo/hailo8/part1")
    else:
        request["kwargs"]["hw_arch"] = "hailo10"
    _write(path, request)
    original = path.read_bytes()
    with pytest.raises((ValueError, RuntimeError)):
        _finalize(data, build_fn=lambda *a, **k: pytest.fail("mismatched requested Hailo entered builder"))
    assert path.read_bytes() == original


def test_unknown_request_placement_rejects_before_read_or_builder_resolution(tmp_path, monkeypatch):
    from onnx_splitpoint_tool.gui import benchmark_workflow

    data = _prepared(tmp_path)
    _hailo_scope(data)
    path = data.suite / "unknown_placement" / deferred.REQUEST_NAME
    path.parent.mkdir()
    path.write_bytes(b"not JSON: path must reject before reading\n")
    _forbid_reads(monkeypatch, {path})
    monkeypatch.setattr(benchmark_workflow, "resolve_hailo_benchmark_helpers",
                        lambda **k: pytest.fail("unknown request resolved Hailo builder"))
    monkeypatch.setattr(benchmark_workflow, "_materialize_manual_deepx_part1_artifacts",
                        lambda **k: pytest.fail("unknown request reached DeepX materializer"))
    with pytest.raises((ValueError, RuntimeError)):
        _finalize(data)


@pytest.mark.parametrize("fault", ["plan_profile_drift", "missing_plan", "missing_authority",
                                   "missing_central_statistics", "malformed_plan", "mixed_runtime"])
def test_generic_authority_is_required_before_any_continuation(tmp_path, monkeypatch, fault):
    data = _prepared(tmp_path)
    _hailo_scope(data)
    path = _hailo_request(data)
    plan_path = data.suite / "benchmark_plan.json"
    plan = json.loads(plan_path.read_text())
    if fault == "plan_profile_drift":
        plan["runs"][0]["hardware_setup_id"] = "other_hailo_setup"
        _write(plan_path, plan)
    elif fault == "missing_plan":
        plan_path.unlink()
    elif fault == "missing_authority":
        _write(plan_path, {"model_id": "model", "runs": []})
    elif fault == "missing_central_statistics":
        data.profile["quality_gate"]["statistics"]["execution_location"] = "remote"
    elif fault == "malformed_plan":
        _write(plan_path, [])
    else:
        plan["runs"].append({"id": "unbound_runtime", "type": "mixed_backend",
                             "stage1": "tensorrt", "stage2": "deepx_m1",
                             "variants": ["composed"], "case_ids": ["b001"]})
        _write(plan_path, plan)
    original = path.read_bytes()
    _forbid_reads(monkeypatch, {path})
    expected_error = FileNotFoundError if fault == "missing_plan" else (ValueError, RuntimeError)
    with pytest.raises(expected_error):
        _finalize(data, build_fn=lambda *a, **k: pytest.fail("unbound Generic request reached builder"))
    assert path.read_bytes() == original


def test_legacy_without_generic_source_intent_keeps_normal_saved_request_path(tmp_path):
    from test_v27921_deferred_builds import _prepared_request

    model, suite, source, output = _prepared_request(tmp_path, cache_only=True)
    calls = []

    def cached(actual_source, **args):
        assert args["cache_only"] is True and args["force"] is False
        calls.append(Path(actual_source))
        return _result(Path(args["outdir"]))

    result = deferred.finalize_deferred_hailo_builds(
        model_dir=model, run_dir=tmp_path, profile_payload={}, build_fn=cached)
    assert result["status"] == "completed"
    assert calls == [source]
    assert result["jobs"][0]["request"] == str(output / deferred.REQUEST_NAME)
