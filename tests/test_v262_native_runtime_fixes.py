from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import numpy as np


def _sealed_input_fixture(
    monkeypatch, module, tmp_path: Path, image_path: Path,
):
    artifact_root = (
        tmp_path / "native_full_outputs"
        / "model=resnet50" / "backend=native_full_deepx"
        / "setup=unspecified" / "comparison=unspecified"
    )
    artifact_root.mkdir(parents=True)
    tensor = artifact_root / "runtime_input.bin"
    tensor.write_bytes(np.zeros((224, 224, 3), dtype=np.uint8).tobytes())
    digest = hashlib.sha256(tensor.read_bytes()).hexdigest()
    semantic_identity = {"schema": "test-preprocessing", "version": 3}
    numeric_identity = {"schema": "test-numeric-input", "version": 3}
    manifest = artifact_root / "native_full_input_manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    validated = {
        "runtime_input_file": str(tensor),
        "runtime_input_name": "input",
        "runtime_input_shape": [224, 224, 3],
        "runtime_input_dtype": "uint8",
        "runtime_input_bytes": tensor.stat().st_size,
        "runtime_input_sha256": digest,
        "runtime_input_layout": "HWC",
        "runtime_preprocessing_identity": semantic_identity,
        "runtime_preprocessing_sha256": "a" * 64,
        "runtime_numeric_input_identity": numeric_identity,
        "runtime_numeric_input_sha256": "b" * 64,
        "input_image": str(image_path),
        "input_image_sha256": hashlib.sha256(
            image_path.read_bytes()
        ).hexdigest(),
    }
    monkeypatch.setattr(
        module,
        "_validated_runtime_input_manifest",
        lambda _path, **_kwargs: (dict(validated), "verified"),
    )
    return manifest, {
        "prepared_input_file": str(tensor),
        "prepared_input_sha256": digest,
        "prepared_input_file_sha256": digest,
        "prepared_input_bytes": tensor.stat().st_size,
        "prepared_input_name": "input",
        "prepared_input_shape": [224, 224, 3],
        "prepared_input_dtype": "uint8",
        "prepared_input_layout": "HWC",
        "runtime_preprocessing_identity": semantic_identity,
        "runtime_preprocessing_sha256": "a" * 64,
        "runtime_numeric_input_identity": numeric_identity,
        "runtime_numeric_input_sha256": "b" * 64,
        "prepared_input_source_image_id": image_path.name,
        "prepared_input_source_image_sha256": hashlib.sha256(
            image_path.read_bytes()
        ).hexdigest(),
        "prepared_input_binding_verified": True,
        "prepared_input_source": "sealed_semantic_dump_runtime_tensor",
        "prepared_feed_contract_version": "deepx-sealed-runtime-input-v3",
    }


def _load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _template_function(name: str):
    text = Path(
        "onnx_splitpoint_tool/resources/templates/benchmark_suite.py.txt"
    ).read_text(encoding="utf-8")
    tree = ast.parse(text)
    node = next(
        item for item in tree.body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == name
    )
    namespace: dict[str, Any] = {
        "Any": Any,
        "Dict": Dict,
        "List": List,
        "Path": Path,
        "json": json,
        "os": os,
        "subprocess": subprocess,
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<benchmark-suite-template>", "exec"), namespace)
    return namespace[name], namespace


def test_native_full_report_rebase_uses_producer_model_context_for_six_copies(
    tmp_path: Path,
) -> None:
    mod = _load_script(
        "v262_native_context_validator",
        Path("scripts/native_producer_validate_visualize.py"),
    )
    producers = {
        "hailo8": "orin_nx_hailo8_01",
        "hailo10h": "orin_nx_hailo10_01",
        "deepx": "orin_nx_deepx_m1_01",
    }
    models = ("resnet50", "yolo26s")
    expected: dict[tuple[str, str], Path] = {}
    for producer in producers:
        for model in models:
            report = (
                tmp_path / "native_producers" / producer / model / "benchmark_set"
                / "native_trt" / "full" / "fp16" / "native_trt_meta.json"
            )
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_text(
                json.dumps({"producer": producer, "model": model}), encoding="utf-8"
            )
            expected[(producer, model)] = report.resolve()

    for producer, setup_id in producers.items():
        for model in models:
            row = {
                "model": model,
                "backend": "native_full_tensorrt",
                "case": "full",
                "execution_mode": "native_full_baseline",
                "setup_id": setup_id,
                "comparison_backend": producer,
                # Exercise copied-summary rebasing for one producer and the
                # authoritative local source_root path for the others.
                "source_root": (
                    f"/old-host/eval/native_producers/{producer}"
                    if producer == "deepx"
                    else str(tmp_path / "native_producers" / producer)
                ),
                "report": (
                    f"/remote/{model}/benchmark_set/native_trt/full/fp16/"
                    "native_trt_meta.json"
                ),
            }
            assert mod._find_report(row, [tmp_path]) == expected[(producer, model)]


def test_hailo_vstreams_gets_independent_owned_writable_input() -> None:
    from onnx_splitpoint_tool.runners.backends.hailo_backend import _HailoSession

    base = np.arange(12, dtype=np.uint8)
    external_view = base.reshape(2, 2, 3)
    assert external_view.flags.c_contiguous
    assert external_view.flags.writeable
    assert not external_view.flags.owndata

    class Pipe:
        def infer(self, inputs):
            arr = inputs["hef_input"]
            assert arr.flags.c_contiguous
            assert arr.flags.writeable
            assert arr.flags.owndata
            assert not np.shares_memory(arr, external_view)
            return {"hef_output": np.asarray([[1.0]], dtype=np.float32)}

    session = object.__new__(_HailoSession)
    session._pipe = Pipe()
    session._network_group = object()
    session._network_group_params = object()
    session.persistent_activation = True
    session._active_handle = object()
    session.input_names = ["input"]
    session.runtime_input_shapes = {"input": (2, 2, 3)}
    session._input_name_canonical_to_hef = {"input": "hef_input"}
    session._input_contig_cache = {}
    session._hef_output_names = ["hef_output"]
    session._output_name_hef_to_canonical = {"hef_output": "output"}
    session.output_shapes = {"output": (1,)}

    result = session.infer({"input": external_view})
    assert result["output"].tolist() == [1.0]


def test_deepx_prepared_feed_success_is_not_invalidated_by_diagnostic_cli(
    tmp_path: Path,
) -> None:
    function, namespace = _template_function("_run_deepx_full_run")
    dxnn = tmp_path / "deepx" / "deepx_m1" / "full" / "model.dxnn"
    dxnn.parent.mkdir(parents=True)
    dxnn.write_bytes(b"dxnn")

    namespace["subprocess"] = SimpleNamespace(
        run=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("run_model must not execute inside measurement-only energy windows")
        ),
        TimeoutExpired=subprocess.TimeoutExpired,
        PIPE=subprocess.PIPE,
    )
    namespace["_parse_deepx_run_model_output"] = lambda _text: {
        "latency_ms": 99.0,
        "fps": 999.0,
    }
    namespace["_run_deepx_prepared_feed_benchmark"] = lambda *_args, **_kwargs: {
        "enabled": True,
        "status": "ok",
        "mean_ms": 4.0,
        "fps_from_mean_latency": 250.0,
        "requested_frames": 100,
        "completed_frames": 100,
        "completed_work_units": 100,
        "completed_work_units_source": "dx_engine_prepared_feed_timed_loop",
        "completed_work_units_status": "exact_runtime_counter",
        "unique_input_images": 1,
        "sequence_policy": "single_prepared_feed_repeated",
        "image": "prepared.png",
        "image_source": "suite_test_image_imagenet",
    }
    namespace["_run_deepx_semantic_validation"] = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("semantic validation must be skipped in measurement-only mode")
    )

    args = SimpleNamespace(
        runs=5,
        timeout=30,
        energy_measurement_only=True,
        throughput_frames=100,
        validation_images="",
    )
    rows = function(
        tmp_path,
        {"id": "deepx_m1_full", "dxnn_path": str(dxnn)},
        args,
        expected_endpoint_identity={
            "model_id": "resnet50",
            "backend": "deepx_m1",
            "variant": "full",
        },
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["runtime_ok"] is True
    assert row["runtime_executable"] is True
    assert row["contract_consistent"] is True
    assert row["performance_benchmark_source"] == "dx_engine_prepared_feed"
    assert row["completed_work_units"] == 100
    assert row["diagnostic_run_model_status"] == "intentionally_skipped_measurement_only"
    assert row["diagnostic_run_model_ok"] is None
    assert row["diagnostic_run_model_returncode"] is None
    assert row["semantic_validation_intentionally_skipped"] is True
    assert row["task_valid"] is False
    assert row["accuracy_gate_pass"] is False
    assert row["eligible_for_ranking"] is False
    assert "diagnostic_warning_class" not in row
    assert "error_class" not in row


def test_deepx_wrapper_prefers_root_result_and_rejects_explicit_failed_row(
    monkeypatch, tmp_path: Path,
) -> None:
    mod = _load_script(
        "v262_deepx_result_discovery",
        Path("scripts/native_full_baseline_eval_runner.py"),
    )
    monkeypatch.setattr(mod, "_suite_python_env", lambda _ns, _backend: (sys.executable, {}, []))
    monkeypatch.setattr(
        mod,
        "_run",
        lambda *_args, **_kwargs: {"rc": 0, "returncode": 0, "timed_out": False},
    )
    exact_image = tmp_path / "exact_pair.jpg"
    exact_image.write_bytes(b"exact-pair-image")
    prepared_manifest, prepared_binding = _sealed_input_fixture(
        monkeypatch, mod, tmp_path, exact_image,
    )
    monkeypatch.setattr(mod, "_first_case", lambda _root: ("b001", tmp_path, tmp_path / "runner.py"))
    monkeypatch.setattr(mod, "_resolve_image", lambda *_args, **_kwargs: (exact_image, "exact_map"))
    current = tmp_path / "benchmark_results_deepx_m1_full_auto.json"
    stale = tmp_path / "copied" / "benchmark_results_deepx_m1_full_auto.json"
    stale.parent.mkdir()
    stale.write_text(json.dumps([{
        "run_id": "deepx_m1_full",
        "variant": "full",
        "runtime_ok": True,
        "pipeline_fps_selected": 999.0,
        "performance_benchmark_source": "dx_engine_prepared_feed",
        "prepared_feed_contract_version": "deepx-sealed-runtime-input-v3",
        "completed_frames": 10,
    }]), encoding="utf-8")
    row = {
        "run_id": "deepx_m1_full",
        "variant": "full",
        "runtime_ok": True,
        "pipeline_fps_selected": 25.0,
        "pipeline_cycle_selected_ms": 40.0,
        "performance_benchmark_source": "dx_engine_prepared_feed",
        "prepared_feed_contract_version": "deepx-sealed-runtime-input-v3",
        "completed_frames": 10,
        "deepx_prepared_feed_benchmark": {
            "image": str(exact_image),
            "completed_frames": 10,
            "completed_work_units": 10,
            "completed_work_units_source":
                "dx_engine_prepared_feed_timed_loop",
            "completed_work_units_status": "exact_runtime_counter",
            "makespan_s": 0.4,
            "fps_makespan": 25.0,
            "mean_ms": 40.0,
            **prepared_binding,
            "input_contract": {"input": {"dtype": "uint8", "shape": [224, 224, 3]}},
        },
    }
    current.write_text(json.dumps([row]), encoding="utf-8")
    ns = SimpleNamespace(
        frames=10,
        warmup=0,
        timeout=30,
        duration_s=1.0,
        engine_build_python="auto",
        diagnostic_deepx_input_probes=False,
    )

    result = mod._generic_full_via_suite(
        tmp_path, "resnet50", "native_full_deepx", "deepx_m1_full", ns,
        prepared_input_manifest=prepared_manifest,
    )
    assert result["ok"] is True
    assert result["fps_makespan"] == 25.0
    assert Path(result["result_source"]) == current
    assert result["completed_work_units_status"] == "exact_runtime_counter"

    row["runtime_ok"] = False
    current.write_text(json.dumps([row]), encoding="utf-8")
    rejected = mod._generic_full_via_suite(
        tmp_path, "resnet50", "native_full_deepx", "deepx_m1_full", ns,
        prepared_input_manifest=prepared_manifest,
    )
    assert rejected["ok"] is False
    assert rejected["fps_makespan"] is None
    assert rejected["completed_work_units"] is None

    row["runtime_ok"] = True
    row.pop("prepared_feed_contract_version")
    row["deepx_prepared_feed_benchmark"].pop(
        "prepared_feed_contract_version"
    )
    current.write_text(json.dumps([row]), encoding="utf-8")
    stale_contract = mod._generic_full_via_suite(
        tmp_path, "resnet50", "native_full_deepx", "deepx_m1_full", ns,
        prepared_input_manifest=prepared_manifest,
    )
    assert stale_contract["ok"] is False
    assert stale_contract["fps_makespan"] is None
    assert stale_contract["prepared_feed_contract_binding_ok"] is False
    assert stale_contract["failure_reason"] == "deepx_prepared_feed_contract_version_mismatch"


def test_deepx_diagnostic_failure_is_warning_when_prepared_feed_succeeds(
    tmp_path: Path,
) -> None:
    function, namespace = _template_function("_run_deepx_full_run")
    dxnn = tmp_path / "model.dxnn"
    dxnn.write_bytes(b"dxnn")
    namespace["subprocess"] = SimpleNamespace(
        run=lambda *_args, **_kwargs: SimpleNamespace(
            returncode=7, stdout="", stderr="diagnostic failed",
        ),
        TimeoutExpired=subprocess.TimeoutExpired,
        PIPE=subprocess.PIPE,
    )
    namespace["_parse_deepx_run_model_output"] = lambda _text: {}
    namespace["_run_deepx_prepared_feed_benchmark"] = lambda *_args, **_kwargs: {
        "status": "ok",
        "mean_ms": 5.0,
        "requested_frames": 5,
        "completed_frames": 5,
        "completed_work_units": 5,
        "completed_work_units_source": "dx_engine_prepared_feed_timed_loop",
        "completed_work_units_status": "exact_runtime_counter",
    }
    namespace["_run_deepx_semantic_validation"] = lambda *_args, **_kwargs: {
        "enabled": False,
        "status": "no_validation_images",
    }
    row = function(
        tmp_path,
        {"id": "deepx_m1_full", "dxnn_path": str(dxnn)},
        SimpleNamespace(
            runs=5,
            timeout=30,
            energy_measurement_only=False,
            validation_images="",
        ),
        expected_endpoint_identity={
            "model_id": "resnet50",
            "backend": "deepx_m1",
            "variant": "full",
        },
    )[0]
    assert row["runtime_ok"] is True
    assert row["contract_consistent"] is True
    assert row["diagnostic_run_model_status"] == "failed"
    assert row["diagnostic_warning_class"] == "deepx_run_model_diagnostic_failed"
    assert row["returncode"] == 0
    assert row["eligible_for_ranking"] is False
