from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "onnx_splitpoint_tool/resources/templates/benchmark_suite.py.txt"
FULL_RUNNER = ROOT / "scripts/native_full_baseline_eval_runner.py"


def _sealed_input_fixture(
    monkeypatch, module, tmp_path: Path, image_path: Path,
):
    artifact_root = (
        tmp_path / "native_full_outputs"
        / "model=yolo26s" / "backend=native_full_deepx"
        / "setup=unspecified" / "comparison=unspecified"
    )
    artifact_root.mkdir(parents=True)
    tensor = artifact_root / "runtime_input.bin"
    tensor.write_bytes(np.zeros((640, 640, 3), dtype=np.uint8).tobytes())
    digest = hashlib.sha256(tensor.read_bytes()).hexdigest()
    semantic_identity = {"schema": "test-preprocessing", "version": 3}
    numeric_identity = {"schema": "test-numeric-input", "version": 3}
    manifest = artifact_root / "native_full_input_manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    validated = {
        "runtime_input_file": str(tensor),
        "runtime_input_name": "input",
        "runtime_input_shape": [640, 640, 3],
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
        "prepared_input_shape": [640, 640, 3],
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


def _load_script(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _suite_module() -> ModuleType:
    module = ModuleType("_osp_v269b_deepx_suite")
    module.__file__ = str(SUITE)
    module.__package__ = ""
    exec(compile(SUITE.read_text(encoding="utf-8"), str(SUITE), "exec"), module.__dict__)
    return module


def _run32_row(image: Path) -> dict:
    makespan = 3.6423022099770606
    fps = 27.45516276109055
    return {
        "run_id": "deepx_m1_full",
        "variant": "full",
        "runtime_ok": True,
        "fps_makespan": fps,
        "latency_mean_ms": 36.415131862740964,
        "measured_makespan_s": makespan,
        "completed_frames": 100,
        "performance_benchmark_source": "dx_engine_prepared_feed",
        "prepared_feed_contract_version": "deepx-sealed-runtime-input-v3",
        "deepx_prepared_feed_benchmark": {
            "image": str(image),
            "completed_frames": 100,
            "completed_work_units": 100,
            "completed_work_units_source":
                "dx_engine_prepared_feed_timed_loop",
            "completed_work_units_status": "exact_runtime_counter",
            "makespan_s": makespan,
            "fps_makespan": fps,
            "mean_ms": 36.415131862740964,
            "p50_ms": 36.36913513764739,
            "p95_ms": 36.80382993770763,
            "input_contract": {
                "input": {"dtype": "uint8", "shape": [640, 640, 3], "layout": "HWC"},
            },
        },
    }


def _write_run32_pair(root: Path, row: dict, *, csv_fps: float | None = None) -> tuple[Path, Path]:
    json_path = root / "benchmark_results_deepx_m1_full_auto.json"
    csv_path = root / "benchmark_results_deepx_m1_full_auto.csv"
    json_path.write_text(json.dumps([row]), encoding="utf-8")
    flat = {
        "run_id": row["run_id"], "variant": "full",
        "fps_makespan": row["fps_makespan"] if csv_fps is None else csv_fps,
        "latency_mean_ms": row["latency_mean_ms"],
        "measured_makespan_s": row["measured_makespan_s"],
        "completed_frames": row["completed_frames"],
        # This is exactly why CSV cannot be canonical: nested evidence is a
        # Python-looking string, not a mapping.
        "deepx_prepared_feed_benchmark": str(row["deepx_prepared_feed_benchmark"]),
    }
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat))
        writer.writeheader(); writer.writerow(flat)
    # Reproduce Run32 discovery where the convenience CSV sorts as newer.
    stamp = json_path.stat().st_mtime + 2.0
    os.utime(csv_path, (stamp, stamp))
    return json_path, csv_path


def test_run32_deepx_outer_makespan_uses_canonical_json_not_newer_csv(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    module = _load_script("_osp_v269b_full_runner", FULL_RUNNER)
    image = tmp_path / "000000005600.jpg"
    image.write_bytes(b"run32-image")
    prepared_manifest, prepared_binding = _sealed_input_fixture(
        monkeypatch, module, tmp_path, image,
    )
    run32_row = _run32_row(image)
    run32_row["deepx_prepared_feed_benchmark"].update(
        prepared_binding
    )
    json_path, csv_path = _write_run32_pair(tmp_path, run32_row)

    metrics = module._metrics_from_rows([csv_path, json_path])
    assert Path(metrics["result_source"]) == json_path
    assert metrics["result_source_kind"] == "json"
    assert metrics["source_consistent"] is True
    assert isinstance(metrics["result_row"]["deepx_prepared_feed_benchmark"], dict)

    monkeypatch.setattr(module, "_suite_python_env", lambda *_args: (sys.executable, {}, []))
    monkeypatch.setattr(module, "_run", lambda *_args, **_kwargs: {"rc": 0, "returncode": 0, "timed_out": False})
    monkeypatch.setattr(module, "_first_case", lambda _root: ("b038", tmp_path, tmp_path / "runner.py"))
    monkeypatch.setattr(module, "_resolve_image", lambda *_args, **_kwargs: (image, "exact_map"))
    ns = SimpleNamespace(
        frames=100, warmup=10, timeout=30, duration_s=0.0,
        engine_build_python="auto", diagnostic_deepx_input_probes=False,
    )
    result = module._generic_full_via_suite(
        tmp_path, "yolo26s", "native_full_deepx", "deepx_m1_full", ns,
        prepared_input_manifest=prepared_manifest,
    )
    assert result["ok"] is True
    assert result["completed_work_units"] == 100
    assert result["measured_makespan_s"] == pytest.approx(3.6423022099770606)
    assert result["fps_makespan"] == pytest.approx(27.45516276109055)
    assert result["result_source_kind"] == "json"
    assert Path(result["result_source"]) == json_path
    assert result["measured_duration_s"] == result["measured_makespan_s"]
    from onnx_splitpoint_tool.native_rate_endpoints import rate_endpoint_fields
    projected = rate_endpoint_fields(module._aggregate_full_repetitions([result], requested=1))
    assert projected["completed_task_fps"] is None  # Run32 lacks completed detection attestation
    assert projected["historical_fps"] == result["fps_makespan"]


def test_deepx_json_csv_true_scalar_disagreement_fails_closed(tmp_path: Path) -> None:
    module = _load_script("_osp_v269b_full_runner_conflict", FULL_RUNNER)
    image = tmp_path / "image.jpg"; image.write_bytes(b"x")
    json_path, csv_path = _write_run32_pair(tmp_path, _run32_row(image), csv_fps=99.0)
    metrics = module._metrics_from_rows([csv_path, json_path])
    assert metrics["result_source_kind"] == "json"
    assert metrics["source_consistent"] is False
    assert metrics["source_consistency_reason"] == "canonical_json_csv_scalar_conflict"
    assert metrics["source_conflicts"][0]["field"] == "fps_makespan"


def test_deepx_json_csv_normal_six_decimal_rounding_is_not_a_conflict(tmp_path: Path) -> None:
    module = _load_script("_osp_v269b_full_runner_rounding", FULL_RUNNER)
    image = tmp_path / "image.jpg"; image.write_bytes(b"x")
    row = _run32_row(image)
    json_path, csv_path = _write_run32_pair(
        tmp_path, row, csv_fps=round(float(row["fps_makespan"]), 6),
    )
    metrics = module._metrics_from_rows([csv_path, json_path])
    assert metrics["result_source_kind"] == "json"
    assert metrics["source_consistent"] is True
    assert metrics["source_conflicts"] == []


def _recorded_endpoint_contract() -> dict:
    return {
        "schema": "onnx-splitpoint/output-contract", "schema_version": 1,
        "model_id": "yolo26s", "backend": "deepx_m1", "variant": "full",
        "contract_status": "recorded", "endpoint_mode": "decoded",
        "host_tail_required": False, "postprocessing_required": False,
        "task": "detection", "coordinate_format": "xyxy_score_class",
        "coordinate_space": "model_input_letterbox_xyxy_pixels",
        "source_coordinate_space": "model_input_letterbox_xyxy_pixels",
        "full_end_node_names": [], "artifact_path": "", "warning": "",
    }


def _write_endpoint_tree(root: Path, *, conflicting_artifact: bool = False) -> dict:
    (root / "deepx/deepx_m1/full").mkdir(parents=True)
    artifact = {
        "backend": "deepx_m1", "variant": "full",
        "input": {
            "shape": [640, 640, 3], "layout": "HWC", "dtype": "uint8",
            "normalization": "embedded_dxcom_preprocessing", "color_space": "RGB",
            "preprocess_mode": "letterbox", "task": "detection", "letterbox_pad_value": 114,
        },
        "outputs": [{"name": "model_outputs", "dtype": "float32", "shape": None}],
        "postprocessing": {
            "type": "yolo_host_decode_or_model_postprocess" if conflicting_artifact else "integrated_decoded_nms",
            "host_required": conflicting_artifact,
            "nms_on_host": conflicting_artifact,
        },
    }
    if not conflicting_artifact:
        artifact.update({
            "endpoint_mode": "decoded", "contract_family": "decoded_nms",
            "host_tail_required": False, "postprocessing_required": False,
        })
    (root / "deepx/deepx_m1/full/output_contract.json").write_text(
        json.dumps(artifact), encoding="utf-8",
    )
    (root / "output_contracts.json").write_text(json.dumps({
        "schema": "onnx-splitpoint/output-contracts", "schema_version": 1,
        "model_id": "yolo26s", "contracts": [_recorded_endpoint_contract()],
    }), encoding="utf-8")
    return artifact


def test_run32_bn6_requires_and_accepts_recorded_decoded_endpoint(tmp_path: Path, monkeypatch) -> None:
    # Direct template loading has no exported suite tree. Bind the same current
    # product modules under their exported names, without replacing functions.
    from onnx_splitpoint_tool import native_detection_postprocess, native_output_endpoint
    package = ModuleType("splitpoint_runners"); package.__path__ = []
    monkeypatch.setitem(sys.modules, "splitpoint_runners", package)
    monkeypatch.setitem(sys.modules, "splitpoint_runners.native_detection_postprocess", native_detection_postprocess)
    monkeypatch.setitem(sys.modules, "splitpoint_runners.native_output_endpoint", native_output_endpoint)
    module = _suite_module()
    root = tmp_path / "yolo26s"; _write_endpoint_tree(root)
    run = {
        "id": "deepx_m1_full", "model_id": "yolo26s", "benchmark_task": "detection",
        "contract_path": "deepx/deepx_m1/full/output_contract.json",
    }
    _, contract = module._deepx_input_size_from_contract(root, run, 640)
    assert contract["endpoint_contract_binding_status"] == "attested"
    assert contract["endpoint_mode"] == "decoded"
    assert contract["postprocessing"] == {
        "type": "integrated_decoded_nms", "host_required": False, "nms_on_host": False,
    }
    output = np.asarray([[[10.0, 20.0, 30.0, 40.0, 0.9, 1.0]]], dtype=np.float32)
    detections, decoder = module._deepx_detection_decode(
        root=root, run=run, contract=contract, outputs=[output],
        orig_shape=(640, 640, 3), scale=1.0, pad_x=0, pad_y=0,
    )
    assert decoder["pass"] is True, decoder
    assert decoder["source_endpoint_has_integrated_nms"] is True
    # v2.79.31 uses the attested completion materializer for an already-NMS
    # endpoint. It must not run another host NMS; declaration is still required.
    assert decoder["host_nms_applied"] is False
    assert decoder["canonical_completion_policy_id"] == (
        "decoded_nms_xyxy_original_classaware_postfilter_v2"
    )
    assert decoder["runtime_value_semantic_attestation"]["values_decoded_xyxy_score_class"] is True
    assert detections[0]["class_id"] == 1

    # The identical tensor without the recorded contract remains unavailable;
    # no [1,N,6] shape heuristic may attest endpoint semantics.
    _, shape_only = module._deepx_detection_decode(
        root=root, run=run, contract={}, outputs=[output],
        orig_shape=(640, 640, 3), scale=1.0, pad_x=0, pad_y=0,
    )
    assert shape_only["pass"] is False
    assert shape_only["status"] == "bn6_endpoint_not_explicitly_attested"


def test_run32_per_artifact_host_tail_contradiction_fails_closed(tmp_path: Path) -> None:
    module = _suite_module()
    root = tmp_path / "yolo26s"; _write_endpoint_tree(root, conflicting_artifact=True)
    run = {
        "id": "deepx_m1_full", "model_id": "yolo26s", "benchmark_task": "detection",
        "contract_path": "deepx/deepx_m1/full/output_contract.json",
    }
    _, contract = module._deepx_input_size_from_contract(root, run, 640)
    assert contract["endpoint_contract_binding_status"] == "conflict"
    assert "postprocessing.host_required" in contract["endpoint_contract_binding_conflicts"]
    _, decoder = module._deepx_detection_decode(
        root=root, run=run, contract=contract,
        outputs=[np.zeros((1, 300, 6), dtype=np.float32)],
        orig_shape=(640, 640, 3), scale=1.0, pad_x=0, pad_y=0,
    )
    assert decoder["pass"] is False
    assert decoder["status"] == "bn6_endpoint_not_explicitly_attested"
    assert "disagrees" in decoder["attestation_reason"]
