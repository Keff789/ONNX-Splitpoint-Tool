"""Normal selective DeepX imports over fresh full20 fixtures; no SDK or hardware."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from onnx import TensorProto

from onnx_splitpoint_tool.workflow import deferred_hailo_builds as deferred
from onnx_splitpoint_tool.workflow.generator_binding import import_existing_suite
from test_generic_deepx_deferred_import import IMPORTED_REQUEST, _materializer, _run, _sha
from test_generic_quality_selected_local_import import prepare, selection
from test_generic_quality_selected_transport import (
    MODEL, BINARY_SUFFIXES, case_binary, json_file, republish, tiny_inline_model, write,
)


def _prepared(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    state = prepare(tmp_path)
    source = state["suite"]
    state["run"].update(
        id="deepx_m1_to_tensorrt", stage1="deepx_m1",
        hardware_setup_id="OWN_DEEPX_SETUP",
    )
    state["config"]["identity"].update(
        setup_id="OWN_DEEPX_SETUP", comparison_backend="deepx_m1",
    )
    artifact = write(
        source / "quality_runtime/assets/b001/part1_runtime/selected.dxnn",
        b"OWN exact selected DeepX artifact",
    )
    artifact["path"] = Path(artifact["path"]).relative_to(source).as_posix()
    state["config"]["expected_runtime_artifacts"]["part1_runtime"] = artifact
    state["profile"]["deepx_build"] = {"mode": "reuse_only", "force_build": False}

    # The ordinary deferred writer binds all historical catalogue P1 graphs.
    # Keep those real inline ModelProto bytes, so selection can be proved.
    inline = tiny_inline_model().SerializeToString()
    for case in state["cases"]:
        json_file(source / case / "split_manifest.json", {
            "boundary": int(case[1:]), "part1_model": "part1.onnx",
        })
        write(source / case / "part1.onnx", inline)
    selected = source / "b001/part1.onnx"
    state["config"]["expected_runtime_artifacts"]["source_part1_onnx"].update(
        sha256=hashlib.sha256(selected.read_bytes()).hexdigest(),
        size_bytes=selected.stat().st_size,
    )
    republish(state)
    deferred.defer_deepx_part1_build(
        out_dir=source, bench_plan_runs=[_run(state["cases"])],
        validation_images="/OWN_SAVED/calibration-images", fallback_calib_dir="/OWN_SAVED/fallback",
        calibration_num=37, calibration_manifest="/OWN_SAVED/calibration-manifest.json",
        strict_calibration_source=True, task_hint="detection",
        force_build=False, cache_only=False,
        build_config={"mode": "build_missing", "opt_level": 2,
                      "compiler_venv": "/OWN_SAVED/compiler-venv", "force_build": False},
        profile_payload={"profile_id": "OWN_HISTORICAL_BUILD_POLICY",
                         "deepx_build": {"mode": "build_missing", "force_build": False}},
    )
    return state


def _import(state):
    return import_existing_suite(
        model_id=MODEL, row={"benchmark_set_dir": str(state["suite"])},
        model_dir=state["run_root"] / "models" / MODEL,
        run_dir=state["run_root"], profile_id="OWN_CURRENT", run_id="new_run",
        prediction={}, candidate_plan=state["candidate_plan"], targets=["deepx_m1", "tensorrt"],
        profile_payload=state["profile"],
    )


def _consume(state, suite, consumer):
    if consumer == "probe":
        return deferred.probe_deferred_deepx_part1_cache(suite)
    return deferred.finalize_deferred_hailo_builds(
        model_dir=state["run_root"] / "models" / MODEL,
        run_dir=state["run_root"], profile_payload=state["profile"],
    )


def _fallback(state, fault):
    if fault == "unknown_config":
        state["config"]["schema"] = "OWN_UNKNOWN_SCHEMA"
        republish(state)
    else:
        model = tiny_inline_model()
        tensor = model.graph.initializer[0]
        tensor.ClearField("float_data")
        tensor.data_location = TensorProto.EXTERNAL
        tensor.external_data.add(key="location", value="../b002/part1.data")
        write(state["suite"] / "models/model.onnx", model.SerializeToString())
    assert selection(state) == ()


def test_actual_selected_deepx_import_keeps_full20_and_consumes_only_current_case(tmp_path, monkeypatch):
    state = _prepared(tmp_path)
    source = state["suite"]
    original = (source / deferred.DEEPX_PART1_REQUEST).read_bytes()
    saved = json.loads(original)
    expected_excludes = selection(state)
    assert expected_excludes and saved["kwargs"]["selected_case_dirs"] == state["cases"]
    imported = _import(state)
    suite = imported.suite_dir
    assert len(json.loads((suite / "benchmark_set.json").read_text())["cases"]) == 20
    for case in state["cases"]:
        assert (suite / case).is_dir()
        assert (suite / case / "split_manifest.json").read_bytes() == (source / case / "split_manifest.json").read_bytes()
        for suffix in BINARY_SUFFIXES:
            assert (suite / case_binary(case, suffix)).is_file() is (case == "b001")
    request_path = suite / IMPORTED_REQUEST
    rebound = json.loads(request_path.read_text())
    assert rebound["kwargs"]["selected_case_dirs"] == ["b001"]
    assert rebound["source_onnx_sha256"] == {str(suite / "b001/part1.onnx"): saved["source_onnx_sha256"][str(source / "b001/part1.onnx")]}
    assert tuple(rebound["import_source"]["local_runtime_binary_excludes"]) == expected_excludes
    assert rebound["import_source"]["request"]["sha256"] == _sha(source / deferred.DEEPX_PART1_REQUEST)
    calls = []
    _materializer(monkeypatch, calls)
    assert _consume(state, suite, "probe")["status"] == "ok"
    assert _consume(state, suite, "finalizer")["status"] == "completed"
    assert len(calls) == 2
    assert all(call["selected_case_dirs"] == ["b001"] for call in calls)
    assert all(call["out_dir"] == suite and call["profile_payload"] == state["profile"] for call in calls)
    assert json.loads(request_path.read_text())["kwargs"]["cache_only"] is True
    assert (source / deferred.DEEPX_PART1_REQUEST).read_bytes() == original
    assert (suite / deferred.DEEPX_PART1_REQUEST).read_bytes() == original


def test_selected_p1_missing_or_changed_still_blocks_normal_import(tmp_path, monkeypatch):
    calls = []
    _materializer(monkeypatch, calls)
    for fault in ("missing", "changed"):
        state = _prepared(tmp_path / fault)
        selected = state["suite"] / "b001/part1.onnx"
        if fault == "missing":
            selected.unlink()
        else:
            model = tiny_inline_model()
            model.graph.initializer[0].float_data[0] = 2.0
            write(selected, model.SerializeToString())
        with pytest.raises((ValueError, RuntimeError, OSError)):
            _import(state)
        assert not (state["run_root"] / "models" / MODEL / "benchmark_set/legacy_suite" / IMPORTED_REQUEST).exists()
    assert calls == []


def test_each_consumer_revalidates_original_request_selected_scope_and_source_digest(tmp_path, monkeypatch):
    calls = []
    _materializer(monkeypatch, calls)
    for consumer in ("probe", "finalizer"):
        for fault in ("original_request", "selected_cases", "source_digest"):
            state = _prepared(tmp_path / (consumer + "_" + fault))
            suite = _import(state).suite_dir
            request_path = suite / IMPORTED_REQUEST
            rebound = json.loads(request_path.read_text())
            if fault == "original_request":
                original = state["suite"] / deferred.DEEPX_PART1_REQUEST
                saved = json.loads(original.read_text())
                saved["kwargs"]["calibration_num"] += 1
                json_file(original, saved)
            elif fault == "selected_cases":
                rebound["kwargs"]["selected_case_dirs"] = ["b002"]
                json_file(request_path, rebound)
            else:
                selected = suite / "b001/part1.onnx"
                model = tiny_inline_model()
                model.graph.initializer[0].float_data[0] = 2.0
                write(selected, model.SerializeToString())
                rebound["source_onnx_sha256"][str(selected)] = _sha(selected)
                json_file(request_path, rebound)
            with pytest.raises((ValueError, RuntimeError)):
                _consume(state, suite, consumer)
    assert calls == []


def test_unknown_and_external_fallback_keep_full20_request_and_probe_scope(tmp_path, monkeypatch):
    calls = []
    _materializer(monkeypatch, calls)
    for fault in ("unknown_config", "external_graph"):
        state = _prepared(tmp_path / fault)
        _fallback(state, fault)
        original = (state["suite"] / deferred.DEEPX_PART1_REQUEST).read_bytes()
        suite = _import(state).suite_dir
        rebound = json.loads((suite / IMPORTED_REQUEST).read_text())
        assert rebound["kwargs"]["selected_case_dirs"] == state["cases"]
        assert len(rebound["source_onnx_sha256"]) == 20
        assert "local_runtime_binary_excludes" not in rebound["import_source"]
        assert json.loads((suite.parent / "generator_input.json").read_text())["local_runtime_binary_excludes"] == []
        assert all((suite / case / "part1.onnx").is_file() for case in state["cases"])
        assert _consume(state, suite, "probe")["status"] == "ok"
        assert calls[-1]["selected_case_dirs"] == state["cases"]
        assert (suite / deferred.DEEPX_PART1_REQUEST).read_bytes() == original
    assert len(calls) == 2


def test_full_copy_fallback_keeps_strict_unselected_p1_validation(tmp_path, monkeypatch):
    calls = []
    _materializer(monkeypatch, calls)
    for fault in ("unknown_config", "external_graph"):
        state = _prepared(tmp_path / fault)
        _fallback(state, fault)
        unselected = state["suite"] / "b003/part1.onnx"
        model = tiny_inline_model()
        model.graph.initializer[0].float_data[0] = 2.0
        write(unselected, model.SerializeToString())
        with pytest.raises(ValueError, match="Imported DeepX Part1 ONNX changed: b003"):
            _import(state)
    assert calls == []
