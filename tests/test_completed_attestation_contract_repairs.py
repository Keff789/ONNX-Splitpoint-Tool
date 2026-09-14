from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from onnx_splitpoint_tool.hailo_backend import (
    _hailo_cache_key,
    _write_hailo_receipt,
)
from onnx_splitpoint_tool.native_output_endpoint import (
    load_authoritative_output_contract,
)
from onnx_splitpoint_tool.native_detection_postprocess import (
    DetectionCompletionRuntime,
    FrozenDecodedNmsPostprocessor,
    FrozenPostprocessError,
    build_detection_completion_execution_contract,
    build_frozen_decoded_nms_normalization_contract,
    build_letterbox_geometry_contract,
    build_normalized_detection_endpoint_attestation,
    persist_detection_completion_execution_artifact,
    tensor_signature,
)
from onnx_splitpoint_tool.preprocessing_contract import (
    canonical_image_preprocessing_contract,
    prepare_rgb_uint8_image,
    preprocessing_contract_sha256,
)
from onnx_splitpoint_tool.workflow.benchmark_binding import (
    _promote_verified_hailo_full_contracts,
)


ROOT = Path(__file__).resolve().parents[1]


def _seal_test_hailo_hef(
    hef: Path, *, task: str, hw_arch: str,
    end_nodes: list[str] | None = None,
) -> tuple[Path, Path]:
    source = hef.parent / "receipt_source.onnx"
    source.write_bytes(f"{task}-source-onnx".encode("utf-8"))
    compiler = hef.parent / "receipt_compiler.onnx"
    compiler.write_bytes(f"{task}-compiler-fixed-onnx".encode("utf-8"))
    contract = canonical_image_preprocessing_contract(
        task, (224, 224) if task == "classification" else (640, 640)
    )
    cache_key, cache_payload = _hailo_cache_key(
        model_path=compiler,
        activation_part1=None,
        hw_arch=hw_arch,
        opt_level=1,
        calib_dir=None,
        calib_count=64,
        calib_batch_size=8,
        extra_model_script="",
        start_nodes=None,
        end_nodes=end_nodes,
        preprocessing_contract=contract,
    )
    _write_hailo_receipt(
        hef_path=hef,
        source_onnx=source,
        compiler_onnx=compiler,
        hw_arch=hw_arch,
        net_name="receipt_test",
        preprocessing_contract=contract,
        preprocessing_sha256=preprocessing_contract_sha256(contract),
        cache_key=cache_key,
        cache_payload=cache_payload,
        calibration_identity=str(cache_payload["calibration_identity"]),
        calibration_count=int(cache_payload["calibration_count"]),
    )
    return source, compiler


def _load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(
        f"test_completed_attestation_{name}", path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


trt_hotloop = _load_script("native_trt_full_completed_hotloop")
selfref = _load_script("native_yolo_full_self_reference_probe")
hailo_runner = _load_script("smoke_hailo10_hef_runner")
visualizer = _load_script("native_producer_validate_visualize")
e2e_coordinator = _load_script("native_producer_e2e_eval_runner")


def _direct_bn6_outputs() -> dict[str, np.ndarray]:
    return {
        "detections": np.asarray(
            [[
                [8.0, 20.0, 32.0, 44.0, 0.9, 2.0],
                [1.0, 1.0, 3.0, 3.0, 0.1, 1.0],
            ]],
            dtype=np.float32,
        ),
    }


def _direct_source_attestation(
    outputs: dict[str, np.ndarray],
    *,
    endpoint_hash: str = "a" * 64,
) -> dict[str, object]:
    return {
        "schema": "onnx-splitpoint/runtime-output-endpoint-attestation",
        "schema_version": 3,
        "endpoint": "decoded_nms",
        "stage": "decoded_nms",
        "attested": True,
        "status": "passed",
        "values_decoded_xyxy_score_class": True,
        "declaration_attested": True,
        "endpoint_contract_hash": endpoint_hash,
        "tensor_signature": tensor_signature(outputs),
        "declared_contract": {
            "model_id": "yolo26s",
            "source_coordinate_space": (
                "model_input_letterbox_xyxy_pixels"
            ),
        },
    }


def _direct_normalization_contract(
    outputs: dict[str, np.ndarray] | None = None,
) -> dict[str, object]:
    values = outputs or _direct_bn6_outputs()
    preprocess = canonical_image_preprocessing_contract(
        "detection", (64, 64),
    )
    return build_frozen_decoded_nms_normalization_contract(
        model_id="yolo26s",
        outputs=values,
        input_hw=[64, 64],
        original_wh=[80, 40],
        preprocess=preprocess,
        source_coordinate_space=(
            "model_input_letterbox_xyxy_pixels"
        ),
        source_endpoint_contract_hash="a" * 64,
        source_output_endpoint_attestation=(
            _direct_source_attestation(values)
        ),
    )


def _decoded_execution_contract(
    outputs: dict[str, np.ndarray],
) -> dict[str, object]:
    endpoint_hash = "a" * 64
    attestation = _direct_source_attestation(
        outputs, endpoint_hash=endpoint_hash,
    )
    source_endpoint = {
        "task": "detection",
        "stage": "decoded_nms",
        "contract_family": "decoded_nms",
        "endpoint_contract_complete": True,
        "endpoint_contract_hash": endpoint_hash,
        "output_endpoint_id": f"detection:decoded_nms:{endpoint_hash}",
        "tensor_signature": tensor_signature(outputs),
        "output_endpoint_attestation": attestation,
    }
    return build_detection_completion_execution_contract(
        model_id="yolo26s",
        outputs=outputs,
        input_hw=[64, 64],
        original_wh=[80, 40],
        source_endpoint_contract=source_endpoint,
        preprocess=canonical_image_preprocessing_contract(
            "detection", (64, 64),
        ),
    )


def _selfref_paths(tmp_path: Path) -> dict[str, Path]:
    benchmark_set = tmp_path / "yolo26s" / "benchmark_set"
    models = benchmark_set / "models"
    dump = tmp_path / "run" / "dump"
    models.mkdir(parents=True)
    dump.mkdir(parents=True)
    full = models / "yolo26s.onnx"
    full.write_bytes(b"test-onnx")
    boundary = dump / "native_full_input_manifest.json"
    boundary.write_text(json.dumps({
        "schema": "onnx-splitpoint/native-full-input-dump",
        "schema_version": 1,
        "input_shape_hwc": [2, 2, 3],
    }), encoding="utf-8")
    native_manifest = dump / "native_full_outputs_manifest.json"
    native_manifest.write_text(json.dumps({
        "schema": "onnx-splitpoint/runner-output-dump",
        "schema_version": 4,
        "outputs": [],
    }), encoding="utf-8")
    native_report = tmp_path / "run" / "runtime.json"
    native_report.write_text(json.dumps({
        "completed_task_completion_mode": "frozen_host_tail",
    }), encoding="utf-8")
    return {
        "benchmark_set": benchmark_set,
        "full": full,
        "boundary": boundary,
        "native_manifest": native_manifest,
        "native_report": native_report,
        "out": tmp_path / "self_reference.json",
    }


def _run_selfref(
    monkeypatch: pytest.MonkeyPatch,
    paths: dict[str, Path],
) -> int:
    monkeypatch.setattr(sys, "argv", [
        "native_yolo_full_self_reference_probe.py",
        "--benchmark-set", str(paths["benchmark_set"]),
        "--case", "full",
        "--boundary-manifest", str(paths["boundary"]),
        "--native-output-manifest", str(paths["native_manifest"]),
        "--native-report", str(paths["native_report"]),
        "--full-onnx", str(paths["full"]),
        "--out", str(paths["out"]),
    ])
    monkeypatch.setattr(
        selfref,
        "_run_full_onnx",
        lambda *_args, **_kwargs: (
            ["output"],
            [np.zeros((1, 1, 6), dtype=np.float32)],
            np.zeros((1, 2, 2, 3), dtype=np.uint8),
            {"input_name": "images"},
        ),
    )
    monkeypatch.setattr(
        selfref,
        "load_dump",
        lambda *_args, **_kwargs: (
            {
                f"raw_{index}": np.zeros((2, 2, 4), dtype=np.float32)
                for index in range(6)
            },
            {"producer": "native_full_hailo10h"},
        ),
    )
    monkeypatch.setattr(
        selfref,
        "_candidates",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("Native Full must not use generic raw decoding")
        ),
    )
    return selfref.main()


def test_native_full_self_reference_uses_completed_v2_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _selfref_paths(tmp_path)
    detection = {
        "class_id": 1,
        "score": 0.9,
        "x1": 1.0,
        "y1": 2.0,
        "x2": 10.0,
        "y2": 12.0,
    }
    monkeypatch.setattr(
        selfref,
        "_completed_v2_self_reference_detection",
        lambda *_args, **_kwargs: {
            "available": True,
            "completed_v2_verified": True,
            "expected_contract_source": (
                "verified_completed_task_comparison_endpoint_v2"
            ),
            "full_mode": "completed_v2:full_bn6_normalized",
            "native_mode": "completed_v2:frozen_host_tail",
            "reference_detections": [detection],
            "native_detections": [detection],
        },
    )

    assert _run_selfref(monkeypatch, paths) == 0
    payload = json.loads(paths["out"].read_text(encoding="utf-8"))
    assert payload["schema_version"] == 6
    assert payload["ok"] is True
    assert payload["semantic_available"] is True
    assert payload["expected_contract_family"] == "decoded_nms"
    assert payload["decode_mode"] == "completed_v2:frozen_host_tail"
    assert payload["native_detections"] == [detection]


def test_native_full_self_reference_pass_markdown_scopes_claim_to_single_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _selfref_paths(tmp_path)
    detection = {
        "class_id": 1,
        "score": 0.9,
        "x1": 1.0,
        "y1": 2.0,
        "x2": 10.0,
        "y2": 12.0,
    }
    monkeypatch.setattr(
        selfref,
        "_completed_v2_self_reference_detection",
        lambda *_args, **_kwargs: {
            "available": True,
            "completed_v2_verified": True,
            "expected_contract_source": (
                "verified_completed_task_comparison_endpoint_v2"
            ),
            "full_mode": "completed_v2:full_bn6_normalized",
            "native_mode": "completed_v2:frozen_host_tail",
            "reference_detections": [detection],
            "native_detections": [detection],
        },
    )

    assert _run_selfref(monkeypatch, paths) == 0
    payload = json.loads(paths["out"].read_text(encoding="utf-8"))
    markdown = paths["out"].with_suffix(".md").read_text(encoding="utf-8")
    assert payload["diagnosis"] == (
        "native_semantic_matches_full_self_reference"
    )
    assert "for the tested input" in markdown
    assert "configured self-reference similarity policy" in markdown
    assert (
        "does not evaluate or override dataset-wide AP/accuracy" in markdown
    )
    assert "external AP50/reference is the problem" not in markdown


def test_native_full_self_reference_unavailable_is_nonzero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _selfref_paths(tmp_path)
    monkeypatch.setattr(
        selfref,
        "_completed_v2_self_reference_detection",
        lambda *_args, **_kwargs: {
            "available": False,
            "completed_v2_verified": False,
            "reason": "completed_v2_native_result_attestation_mismatch",
        },
    )

    assert _run_selfref(monkeypatch, paths) != 0
    payload = json.loads(paths["out"].read_text(encoding="utf-8"))
    assert payload["ok"] is False
    assert payload["semantic_ok"] is False
    assert payload["semantic_available"] is False
    assert payload["diagnosis"] == "completed_v2_evidence_unavailable"


def test_native_full_self_reference_semantic_failure_is_nonzero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _selfref_paths(tmp_path)
    reference = {
        "class_id": 1,
        "score": 0.9,
        "x1": 1.0,
        "y1": 2.0,
        "x2": 10.0,
        "y2": 12.0,
    }
    native = {**reference, "class_id": 2, "x1": 100.0, "x2": 110.0}
    monkeypatch.setattr(
        selfref,
        "_completed_v2_self_reference_detection",
        lambda *_args, **_kwargs: {
            "available": True,
            "completed_v2_verified": True,
            "full_mode": "completed_v2:full_bn6_normalized",
            "native_mode": "completed_v2:frozen_host_tail",
            "reference_detections": [reference],
            "native_detections": [native],
        },
    )

    assert _run_selfref(monkeypatch, paths) != 0
    payload = json.loads(paths["out"].read_text(encoding="utf-8"))
    assert payload["semantic_available"] is True
    assert payload["semantic_ok"] is False
    assert payload["ok"] is False
    assert payload["diagnosis"] == (
        "native_semantic_differs_from_full_self_reference"
    )


def test_native_full_self_reference_missing_exact_input_is_nonzero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _selfref_paths(tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "native_yolo_full_self_reference_probe.py",
        "--benchmark-set", str(paths["benchmark_set"]),
        "--case", "full",
        "--boundary-manifest", str(paths["boundary"]),
        "--native-output-manifest", str(paths["native_manifest"]),
        "--native-report", str(paths["native_report"]),
        "--full-onnx", str(paths["full"]),
        "--out", str(paths["out"]),
    ])
    monkeypatch.setattr(
        selfref,
        "_run_full_onnx",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("exact runtime input is missing")
        ),
    )

    assert selfref.main() != 0
    payload = json.loads(paths["out"].read_text(encoding="utf-8"))
    assert payload["ok"] is False
    assert payload["semantic_available"] is False
    assert payload["diagnosis"] == "full_onnx_input_evidence_unavailable"


def test_runtime_input_evidence_attests_exact_tensor_and_file(
    tmp_path: Path,
) -> None:
    tensor = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    input_file = tmp_path / "runtime_input.bin"
    input_file.write_bytes(tensor.tobytes())
    expected_sha = hashlib.sha256(tensor.tobytes()).hexdigest()

    evidence = hailo_runner._runtime_input_evidence(
        inputs={"images": tensor},
        prepared_input_names=["images"],
        prepared_runtime_shapes={"images": tensor.shape},
        quantized_inputs=True,
        runtime_input_file=input_file,
    )
    assert evidence == {
        "runtime_input_name": "images",
        "runtime_input_shape": [2, 2, 3],
        "runtime_input_dtype": "uint8",
        "runtime_input_bytes": 12,
        "runtime_input_sha256": expected_sha,
        "runtime_input_file": str(input_file.resolve()),
        "runtime_input_binding_verified": True,
        "runtime_input_preflight_bound": False,
        "runtime_input_hash_source": "post_hotloop_exact_tensor_bytes",
    }

    contract = {
        "runtime_input_name": "images",
        "runtime_input_shape": [2, 2, 3],
        "runtime_input_dtype": "uint8",
        "runtime_input_bytes": 12,
        "runtime_input_sha256": expected_sha,
    }
    bound = hailo_runner._runtime_input_evidence(
        inputs={"images": tensor},
        prepared_input_names=["images"],
        prepared_runtime_shapes={"images": tensor.shape},
        quantized_inputs=True,
        bound_runtime_contract=contract,
        preverified_sha256=expected_sha,
        runtime_input_file=input_file,
    )
    assert bound["runtime_input_binding_verified"] is True
    assert bound["runtime_input_preflight_bound"] is True
    with pytest.raises(RuntimeError, match="content binding mismatch"):
        hailo_runner._runtime_input_evidence(
            inputs={"images": tensor},
            prepared_input_names=["images"],
            prepared_runtime_shapes={"images": tensor.shape},
            quantized_inputs=True,
            bound_runtime_contract={**contract, "runtime_input_sha256": "0" * 64},
            preverified_sha256="0" * 64,
            runtime_input_file=input_file,
        )


def test_direct_bn6_processor_emits_canonical_completed_artifact() -> None:
    outputs = _direct_bn6_outputs()
    contract = _direct_normalization_contract(outputs)
    processor = FrozenDecodedNmsPostprocessor(contract)
    result = processor.process(outputs, original_wh=[80, 40])

    assert result["record_schema"] == "xyxy_score_class_id_v1"
    assert result["completed_result_artifact"]["detections"] == result[
        "detections"
    ]
    assert result["completed_result_artifact_sha256"] == (
        trt_hotloop._canonical_json_sha256(
            result["completed_result_artifact"]
        )
    )
    attestation = build_normalized_detection_endpoint_attestation(
        contract,
        result,
        completed_frames=1,
        postprocess_completed_frames=1,
    )
    assert attestation["attested"] is True
    assert attestation["completed_task_completion_mode"] == (
        "integrated_accelerator_plus_frozen_normalization"
    )


def test_direct_v1_hash_only_attestation_requires_explicit_legacy_mode() -> None:
    outputs = _direct_bn6_outputs()
    contract = _direct_normalization_contract(outputs)
    result = FrozenDecodedNmsPostprocessor(contract).process(
        outputs, original_wh=[80, 40],
    )
    legacy_result = {
        key: result[key]
        for key in (
            "task",
            "contract_family",
            "decoder_format",
            "coordinate_space",
            "detection_count",
            "detections_sha256",
            "normalization_contract_sha256",
        )
    }
    with pytest.raises(
        FrozenPostprocessError,
        match="direct_normalization_result_artifact_invalid",
    ):
        build_normalized_detection_endpoint_attestation(
            contract,
            legacy_result,
            completed_frames=1,
            postprocess_completed_frames=1,
        )
    attestation = build_normalized_detection_endpoint_attestation(
        contract,
        legacy_result,
        completed_frames=1,
        postprocess_completed_frames=1,
        allow_legacy_hash_only_v1=True,
    )
    assert attestation[
        "frozen_decoded_nms_normalization_result"
    ] == legacy_result


def test_trt_direct_bn6_hotloop_persists_completed_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outputs = _direct_bn6_outputs()
    contract = _direct_normalization_contract(outputs)
    tensor = np.zeros((1, 64, 64, 3), dtype=np.float32)
    runtime_input = tmp_path / "runtime_input.bin"
    runtime_input.write_bytes(tensor.tobytes())
    tensor_sha = hashlib.sha256(runtime_input.read_bytes()).hexdigest()
    input_manifest = tmp_path / "native_full_input_manifest.json"
    input_manifest.write_text(json.dumps({
        "schema": "onnx-splitpoint/native-full-input-dump",
        "schema_version": 2,
        "case": "full",
        "runtime_input_file": str(runtime_input),
        "runtime_input_name": "images",
        "runtime_input_shape": list(tensor.shape),
        "runtime_input_dtype": str(tensor.dtype),
        "runtime_input_bytes": int(tensor.nbytes),
        "runtime_input_sha256": tensor_sha,
        "original_image_wh": [80, 40],
        "letterbox_geometry_contract": contract[
            "letterbox_geometry_contract"
        ],
        "letterbox_geometry_contract_sha256": contract[
            "letterbox_geometry_contract_sha256"
        ],
    }), encoding="utf-8")
    engine = tmp_path / "model.engine"
    engine.write_bytes(b"engine")
    report = tmp_path / "trt_runtime.json"

    class _FakeNativeTRT:
        inputs = ["images"]
        shapes = {"images": tensor.shape}
        dtypes = {"images": tensor.dtype}

        def __init__(self, _engine: Path) -> None:
            self.closed = False

        def prepare_inputs(self, values: dict[str, np.ndarray]) -> None:
            assert np.array_equal(values["images"], tensor)

        def run_prepared(self) -> dict[str, np.ndarray]:
            return {
                name: np.array(value, copy=True)
                for name, value in outputs.items()
            }

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(trt_hotloop, "NativeTRT", _FakeNativeTRT)
    monkeypatch.setattr(sys, "argv", [
        "native_trt_full_completed_hotloop.py",
        "--engine", str(engine),
        "--input-manifest", str(input_manifest),
        "--frozen-decoded-nms-normalization-contract-json",
        json.dumps(contract),
        "--frames", "2",
        "--warmup", "1",
        "--json-out", str(report),
        "--expected-runner-sha256",
        trt_hotloop._sha256_file(Path(trt_hotloop.__file__)),
        "--expected-engine-sha256",
        hashlib.sha256(engine.read_bytes()).hexdigest(),
        "--expected-input-manifest-sha256",
        hashlib.sha256(input_manifest.read_bytes()).hexdigest(),
        "--expected-runtime-input-sha256", tensor_sha,
    ])

    assert trt_hotloop.main() == 0
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["ok"] is True
    assert payload["normalization_frozen"] is True
    assert payload["postprocess_completed_frames"] == 2
    assert payload["completed_task_completion_mode"] == (
        "integrated_accelerator_plus_frozen_normalization"
    )
    assert payload["completed_task_result_artifact_saved"] is True
    artifact_path = Path(payload["completed_task_result_artifact_path"])
    assert artifact_path.is_file()
    assert (
        hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        == payload["completed_task_result_artifact_file_sha256"]
        == payload["completed_task_result_artifact_sha256"]
    )


def test_completed_result_is_materialized_and_hash_verified(
    tmp_path: Path,
) -> None:
    artifact = {
        "schema": "test/completed-result",
        "schema_version": 1,
        "detections": [],
    }
    hailo_sha = hashlib.sha256(
        hailo_runner.canonical_json_bytes(artifact)
    ).hexdigest()
    hailo_path = tmp_path / "hailo.completed.json"
    evidence = hailo_runner._persist_completed_result_artifact(
        artifact, hailo_sha, hailo_path,
    )
    assert evidence["saved"] is True
    assert evidence["file_sha256"] == hailo_sha
    assert hashlib.sha256(hailo_path.read_bytes()).hexdigest() == hailo_sha

    trt_path = tmp_path / "trt.completed.json"
    trt_sha = trt_hotloop._canonical_json_sha256(artifact)
    trt_evidence = trt_hotloop._persist_completed_result_artifact(
        artifact, trt_sha, trt_path,
    )
    assert trt_evidence["saved"] is True
    assert trt_evidence["file_sha256"] == trt_sha
    with pytest.raises(RuntimeError, match="canonical SHA-256 mismatch"):
        trt_hotloop._persist_completed_result_artifact(
            artifact, "0" * 64, tmp_path / "must-not-exist.json",
        )

    direct_artifact = trt_hotloop._completed_detection_artifact({
        "detections": [
            {
                "class_id": 3, "score": 0.4,
                "x1": 1.0, "y1": 2.0, "x2": 3.0, "y2": 4.0,
            },
            {
                "class_id": 2, "score": 0.9,
                "x1": 5.0, "y1": 6.0, "x2": 7.0, "y2": 8.0,
            },
        ],
    })
    assert direct_artifact["schema"] == (
        "onnx-splitpoint/frozen-completed-detection-result-artifact"
    )
    assert [item["score"] for item in direct_artifact["detections"]] == [
        0.9, 0.4,
    ]


def test_completed_result_consumer_requires_persisted_file_and_hash(
    tmp_path: Path,
) -> None:
    detections = [{
        "class_id": 2,
        "score": 0.9,
        "x1": 1.0,
        "y1": 2.0,
        "x2": 7.0,
        "y2": 8.0,
    }]
    artifact = {
        "schema": "onnx-splitpoint/frozen-completed-detection-result-artifact",
        "schema_version": 1,
        "record_schema": "xyxy_score_class_id_v1",
        "coordinate_space": "original_image_xyxy_pixels",
        "sort_policy": "score_desc_class_id_asc_xyxy_lexicographic_v1",
        "detections": detections,
    }
    artifact_bytes = json.dumps(
        artifact,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    artifact_sha256 = hashlib.sha256(artifact_bytes).hexdigest()
    native_report = tmp_path / "runtime.json"
    native_report.write_text("{}", encoding="utf-8")
    artifact_path = tmp_path / "completed_task_result_artifact.json"
    artifact_path.write_bytes(artifact_bytes)
    sealed_result = {
        "coordinate_space": "original_image_xyxy_pixels",
        "record_schema": "xyxy_score_class_id_v1",
        "canonical_sort_policy": (
            "score_desc_class_id_asc_xyxy_lexicographic_v1"
        ),
        "detection_count": 1,
        "detections": detections,
        "detections_sha256": visualizer._canonical_detection_json_sha256(
            detections
        ),
        "completed_result_artifact": artifact,
        "completed_result_artifact_sha256": artifact_sha256,
    }
    endpoint_evidence = {
        "completed_task_result_artifact_saved": True,
        "completed_task_result_artifact": artifact,
        "completed_task_result_artifact_sha256": artifact_sha256,
        "completed_task_result_artifact_path": (
            "/home/nx/producer/completed_task_result_artifact.json"
        ),
        "completed_task_result_artifact_file_sha256": artifact_sha256,
    }

    assert visualizer._verified_frozen_completed_result_artifact(
        sealed_result,
        endpoint_evidence,
        native_report=native_report,
    ) == detections

    artifact_path.write_text("{}", encoding="utf-8")
    with pytest.raises(
        visualizer._FrozenPostprocessError,
        match="completed_v2_hotloop_result_artifact_persistence_invalid",
    ):
        visualizer._verified_frozen_completed_result_artifact(
            sealed_result,
            endpoint_evidence,
            native_report=native_report,
        )

    artifact_path.write_bytes(artifact_bytes)
    endpoint_evidence["completed_task_result_artifact_saved"] = False
    with pytest.raises(
        visualizer._FrozenPostprocessError,
        match="completed_v2_hotloop_result_artifact_persistence_invalid",
    ):
        visualizer._verified_frozen_completed_result_artifact(
            sealed_result,
            endpoint_evidence,
            native_report=native_report,
        )


def test_collected_deepx_completed_artifact_rebases_exact_smoke_suffix(
    tmp_path: Path,
) -> None:
    benchmark_set = (
        tmp_path / "smoke_20260807" / "native_producers" / "deepx"
        / "yolo26s" / "benchmark_set"
    )
    native_report = benchmark_set / "benchmark_results_deepx_m1_full_auto.json"
    native_report.parent.mkdir(parents=True)
    native_report.write_text("{}", encoding="utf-8")
    detections = [{
        "class_id": 0,
        "score": 0.9343951940536499,
        "x1": 233.7249755859375,
        "y1": 81.5643310546875,
        "x2": 394.23828125,
        "y2": 269.93408203125,
    }]
    artifact = {
        "schema": "onnx-splitpoint/frozen-completed-detection-result-artifact",
        "schema_version": 1,
        "record_schema": "xyxy_score_class_id_v1",
        "coordinate_space": "original_image_xyxy_pixels",
        "sort_policy": "score_desc_class_id_asc_xyxy_lexicographic_v1",
        "detections": detections,
    }
    artifact_bytes = json.dumps(
        artifact,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    artifact_sha256 = hashlib.sha256(artifact_bytes).hexdigest()
    collected_artifact = (
        benchmark_set / "results" / "deepx_m1_full"
        / "deepx_prepared_feed.completed_task_result_artifact.json"
    )
    collected_artifact.parent.mkdir(parents=True)
    collected_artifact.write_bytes(artifact_bytes)
    remote_artifact = (
        "/home/nx/native_fifo_evalsets/"
        "resnet_yolo26s_yolo7_20260807_103543/yolo26s/benchmark_set/"
        "results/deepx_m1_full/"
        "deepx_prepared_feed.completed_task_result_artifact.json"
    )
    sealed_result = {
        "coordinate_space": "original_image_xyxy_pixels",
        "record_schema": "xyxy_score_class_id_v1",
        "canonical_sort_policy": (
            "score_desc_class_id_asc_xyxy_lexicographic_v1"
        ),
        "detection_count": 1,
        "detections": detections,
        "detections_sha256": visualizer._canonical_detection_json_sha256(
            detections
        ),
        "completed_result_artifact": artifact,
        "completed_result_artifact_sha256": artifact_sha256,
    }
    endpoint_evidence = {
        "completed_task_result_artifact_saved": True,
        "completed_task_result_artifact": artifact,
        "completed_task_result_artifact_sha256": artifact_sha256,
        "completed_task_result_artifact_path": remote_artifact,
        "completed_task_result_artifact_file_sha256": artifact_sha256,
    }

    assert visualizer._verified_frozen_completed_result_artifact(
        sealed_result,
        endpoint_evidence,
        native_report=native_report,
        artifact_root=benchmark_set,
    ) == detections

    endpoint_evidence["completed_task_result_artifact_path"] = (
        remote_artifact.replace(
            "/results/deepx_m1_full/", "/../outside/",
        )
    )
    with pytest.raises(
        visualizer._FrozenPostprocessError,
        match="completed_v2_hotloop_result_artifact_persistence_invalid",
    ):
        visualizer._verified_frozen_completed_result_artifact(
            sealed_result,
            endpoint_evidence,
            native_report=native_report,
            artifact_root=benchmark_set,
        )


def test_direct_bn6_consumer_uses_persisted_artifact_without_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outputs = _direct_bn6_outputs()
    contract = _direct_normalization_contract(outputs)
    processor = FrozenDecodedNmsPostprocessor(contract)
    sealed = processor.process(outputs, original_wh=[80, 40])
    attestation = build_normalized_detection_endpoint_attestation(
        contract,
        sealed,
        completed_frames=1,
        postprocess_completed_frames=1,
    )
    artifact = sealed["completed_result_artifact"]
    artifact_sha = sealed["completed_result_artifact_sha256"]
    native_report = tmp_path / "runtime.json"
    native_report.write_text("{}", encoding="utf-8")
    artifact_file = tmp_path / "completed.json"
    artifact_file.write_bytes(trt_hotloop._canonical_json_bytes(artifact))
    evidence = {
        "completed_task_endpoint_attestation": attestation,
        "frozen_decoded_nms_normalization_result": sealed,
        "completed_task_result_artifact_saved": True,
        "completed_task_result_artifact": artifact,
        "completed_task_result_artifact_sha256": artifact_sha,
        "completed_task_result_artifact_path": (
            "/remote/run/completed.json"
        ),
        "completed_task_result_artifact_file_sha256": artifact_sha,
    }
    comparison = attestation[
        "completed_task_comparison_endpoint_contract"
    ]
    monkeypatch.setattr(
        visualizer,
        "_verified_completed_v2_contract",
        lambda *_args, **_kwargs: (
            "integrated_accelerator_plus_frozen_normalization",
            contract,
            comparison,
        ),
    )

    class _ReplayForbidden:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            raise AssertionError("persisted Direct-BN6 evidence must not replay")

    monkeypatch.setattr(
        visualizer, "_FrozenDecodedNmsPostprocessor", _ReplayForbidden,
    )
    result = visualizer._completed_v2_self_reference_detection(
        outputs,
        outputs,
        evidence,
        native_report=native_report,
    )
    assert result["available"] is True
    assert result["exact_completed_result_identity_bound"] is True
    assert result["semantic_result_binding_status"] == (
        "exact_same_hotloop_completed_artifact"
    )


def test_direct_bn6_consumer_rejects_embedded_only_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outputs = _direct_bn6_outputs()
    contract = _direct_normalization_contract(outputs)
    processor = FrozenDecodedNmsPostprocessor(contract)
    sealed = processor.process(outputs, original_wh=[80, 40])
    attestation = build_normalized_detection_endpoint_attestation(
        contract,
        sealed,
        completed_frames=1,
        postprocess_completed_frames=1,
    )
    comparison = attestation[
        "completed_task_comparison_endpoint_contract"
    ]
    monkeypatch.setattr(
        visualizer,
        "_verified_completed_v2_contract",
        lambda *_args, **_kwargs: (
            "integrated_accelerator_plus_frozen_normalization",
            contract,
            comparison,
        ),
    )
    result = visualizer._completed_v2_self_reference_detection(
        outputs,
        outputs,
        {
            "completed_task_endpoint_attestation": attestation,
            "frozen_decoded_nms_normalization_result": sealed,
            "completed_task_result_artifact_saved": False,
            "completed_task_result_artifact": sealed[
                "completed_result_artifact"
            ],
            "completed_task_result_artifact_sha256": sealed[
                "completed_result_artifact_sha256"
            ],
        },
        native_report=tmp_path / "runtime.json",
    )
    assert result["available"] is False
    assert result["reason"].endswith(
        "completed_v2_hotloop_result_artifact_persistence_invalid"
    )


def test_completion_execution_consumer_uses_persisted_artifact_and_no_second_nms(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outputs = {
        "detections": np.asarray([[  # deliberately overlapping, already NMS'd
            [8.0, 20.0, 32.0, 44.0, 0.9, 2.0],
            [9.0, 21.0, 31.0, 43.0, 0.8, 2.0],
        ]], dtype=np.float32),
    }
    execution = _decoded_execution_contract(outputs)
    runtime = DetectionCompletionRuntime(execution)
    sealed = runtime.process(outputs)
    attestation = runtime.attestation(completed_work_units=1)
    artifact = attestation["artifact"]
    artifact_sha = attestation["artifact_sha256"]
    native_report = tmp_path / "runtime.json"
    native_report.write_text("{}", encoding="utf-8")
    artifact_file = tmp_path / "execution_completed.json"
    artifact_file.write_bytes(trt_hotloop._canonical_json_bytes(artifact))
    evidence = {
        "completed_task_endpoint_attestation": attestation,
        "completed_task_result_artifact_saved": True,
        "completed_task_result_artifact": artifact,
        "completed_task_result_artifact_sha256": artifact_sha,
        "completed_task_result_artifact_path": (
            "/remote/run/execution_completed.json"
        ),
        "completed_task_result_artifact_file_sha256": artifact_sha,
    }
    comparison = execution["comparison_endpoint_contract"]
    monkeypatch.setattr(
        visualizer,
        "_verified_completed_v2_contract",
        lambda *_args, **_kwargs: (
            "detection_completion_execution_v1",
            execution,
            comparison,
        ),
    )

    class _SecondNmsForbidden:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            raise AssertionError(
                "decoded-NMS execution comparison must not run YoloHarness NMS"
            )

    monkeypatch.setattr(
        visualizer, "_CanonicalYoloHarness", _SecondNmsForbidden,
    )
    result = visualizer._completed_v2_self_reference_detection(
        outputs,
        outputs,
        evidence,
        native_report=native_report,
    )
    assert result["available"] is True
    assert len(result["native_detections"]) == 2
    assert len(result["reference_detections"]) == 2
    assert result["native_detections"] == result["reference_detections"]
    assert result["exact_completed_result_identity_bound"] is True


def test_completion_execution_consumer_rejects_embedded_only_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outputs = _direct_bn6_outputs()
    execution = _decoded_execution_contract(outputs)
    runtime = DetectionCompletionRuntime(execution)
    runtime.process(outputs)
    attestation = runtime.attestation(completed_work_units=1)
    monkeypatch.setattr(
        visualizer,
        "_verified_completed_v2_contract",
        lambda *_args, **_kwargs: (
            "detection_completion_execution_v1",
            execution,
            execution["comparison_endpoint_contract"],
        ),
    )
    result = visualizer._completed_v2_self_reference_detection(
        outputs,
        outputs,
        {
            "completed_task_endpoint_attestation": attestation,
            "completed_task_result_artifact_saved": False,
            "completed_task_result_artifact": attestation["artifact"],
            "completed_task_result_artifact_sha256": attestation[
                "artifact_sha256"
            ],
        },
        native_report=tmp_path / "runtime.json",
    )
    assert result["available"] is False
    assert result["reason"].endswith(
        "completed_v2_execution_result_artifact_persistence_invalid"
    )


def test_completion_execution_producer_persists_exact_artifact(
    tmp_path: Path,
) -> None:
    outputs = _direct_bn6_outputs()
    execution = _decoded_execution_contract(outputs)
    runtime = DetectionCompletionRuntime(execution)
    runtime.process(outputs)
    attestation = runtime.attestation(completed_work_units=1)
    payload = {
        "completion_execution_attestation": attestation,
        "completed_task_endpoint_attestation": attestation,
    }
    persisted = persist_detection_completion_execution_artifact(
        payload,
        output_path=tmp_path / "completed_execution.json",
    )
    artifact_path = Path(
        persisted["completed_task_result_artifact_path"]
    )
    assert persisted["completed_task_result_artifact_saved"] is True
    assert artifact_path.is_file()
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == (
        attestation["artifact"]
    )
    assert (
        hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        == attestation["artifact_sha256"]
        == persisted["completed_task_result_artifact_sha256"]
        == persisted["completed_task_result_artifact_file_sha256"]
    )


def test_e2e_coordinator_rejects_unpersisted_execution_artifact(
    tmp_path: Path,
) -> None:
    outputs = _direct_bn6_outputs()
    execution = _decoded_execution_contract(outputs)
    runtime = DetectionCompletionRuntime(execution)
    runtime.process(outputs)
    attestation = runtime.attestation(completed_work_units=1)
    completed_endpoint = attestation["completed_endpoint_contract"]
    comparison_endpoint = attestation["comparison_endpoint_contract"]
    result = {
        "completed_frames": 1,
        "completed_work_units": 1,
        "postprocess_included": True,
        "postprocess_completed_frames": 1,
        "postprocess_completion_verified": True,
        "completion_observation_relation": "same_hotloop_sentinel",
        "completion_exact_result_claim_bound": True,
        "measurement_boundary": (
            "workers_ready_to_last_completed_task_frame"
        ),
        "last_completion_source": (
            "same_hotloop_completed_task_sentinel"
        ),
        "completion_execution_attestation": attestation,
        "completed_task_endpoint_attestation": attestation,
        "completed_task_endpoint_contract": completed_endpoint,
        "comparison_endpoint_contract": comparison_endpoint,
        "completion_execution_contract": execution,
        "completion_execution_contract_sha256": execution[
            "contract_sha256"
        ],
        **{
            f"completion_{field}": attestation[field]
            for field in (
                "artifact_sha256",
                "schema_sha256",
                "content_sha256",
                "invocation_sha256",
                "relation_sha256",
            )
        },
    }
    result.update(
        persist_detection_completion_execution_artifact(
            result,
            output_path=tmp_path / "coordinator_completed.json",
        )
    )
    assert e2e_coordinator._same_hotloop_completion_error(
        result, require_contract=True,
    ) == ""
    result["completed_task_result_artifact_saved"] = False
    assert e2e_coordinator._same_hotloop_completion_error(
        result, require_contract=True,
    ) == "detection_same_hotloop_completion_artifact_not_persisted"


def _raw_suite_contract(model_id: str) -> dict[str, object]:
    return {
        "model_id": model_id,
        "backend": "hailo10h",
        "variant": "full",
        "task": "detection",
        "endpoint_mode": "decoded",
        "host_tail_required": False,
        "postprocessing_required": False,
        "contract_status": "recorded",
        "source_onnx_multiscale_raw_head": False,
        "output_format": "bn6_detections",
        "output_record_format": "xyxy_score_class",
        "coordinate_format": "xyxy_score_class",
        "coordinate_space": "model_input_letterbox_xyxy_pixels",
        "source_coordinate_space": "model_input_letterbox_xyxy_pixels",
        "score_semantics": "probability_0_1",
        "class_id_semantics": "integer_model_label_index",
    }


def test_hailo_direct_bn6_hotloop_normalizes_and_persists_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outputs = _direct_bn6_outputs()
    preprocessing = canonical_image_preprocessing_contract(
        "detection", (64, 64),
    )
    preprocessing_sha = preprocessing_contract_sha256(preprocessing)
    tensor = np.zeros((1, 64, 64, 3), dtype=np.uint8)
    runtime_input = tmp_path / "runtime_input.bin"
    runtime_input.write_bytes(tensor.tobytes())
    tensor_sha = hashlib.sha256(runtime_input.read_bytes()).hexdigest()
    preprocess = {
        "mode": "letterbox_rgb_uint8",
        "pad_value": 114,
        "rgb": True,
        "color_space": "RGB",
        "layout": "NHWC",
        "quantized_inputs": True,
        "ort_model_scale": "norm",
        "preprocessing_contract": preprocessing,
        "preprocessing_contract_sha256": preprocessing_sha,
    }
    runtime_contract = {
        "schema": "onnx-splitpoint/preverified-runtime-input-tensor",
        "schema_version": 1,
        "runtime_input_name": "images",
        "runtime_input_shape": list(tensor.shape),
        "runtime_input_dtype": "uint8",
        "runtime_input_bytes": int(tensor.nbytes),
        "runtime_input_sha256": tensor_sha,
        "preprocess": preprocess,
    }
    output_contracts = tmp_path / "output_contracts.json"
    output_contracts.write_text(json.dumps({
        "model_id": "yolo26s",
        "task": "detection",
        "contracts": [_raw_suite_contract("yolo26s")],
    }), encoding="utf-8")
    hef = tmp_path / "compiled.hef"
    hef.write_bytes(b"hef")
    report = tmp_path / "hailo_runtime.json"

    class _FakeHailoBackend:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def prepare(self, _cfg: object, _artifacts: Path) -> object:
            handle = SimpleNamespace(
                runtime_input_shapes={"images": tensor.shape},
                input_shapes={"images": tensor.shape},
                runtime_output_shapes={
                    "detections": outputs["detections"].shape,
                },
                session=SimpleNamespace(hotloop=False),
            )
            return SimpleNamespace(
                handle=handle,
                input_names=["images"],
                output_names=["detections"],
            )

        def run(self, _prepared: object, values: object) -> object:
            assert np.array_equal(values["images"], tensor)
            return SimpleNamespace(
                outputs={
                    name: np.array(value, copy=True)
                    for name, value in outputs.items()
                },
                metrics={"infer_ms": 1.0},
            )

        def cleanup(self, _prepared: object) -> None:
            pass

    monkeypatch.setattr(hailo_runner, "HailoBackend", _FakeHailoBackend)
    monkeypatch.setattr(sys, "argv", [
        "smoke_hailo10_hef_runner.py",
        "--hef", str(hef),
        "--hw-arch", "hailo10h",
        "--artifacts-dir", str(tmp_path / "artifacts"),
        "--runtime-input-bin", str(runtime_input),
        "--runtime-input-contract-json", json.dumps(runtime_contract),
        "--preverified-runtime-input-sha256", tensor_sha,
        "--canonical-input-slot-names-json", '["images"]',
        "--canonical-output-slot-names-json", '["detections"]',
        "--task", "detection",
        "--backend-label", "native_full_hailo10h",
        "--model", "yolo26s",
        "--declared-output-contract-json", str(output_contracts),
        "--original-image-wh-json", "[80,40]",
        "--throughput-mode",
        "--counted-hotloop-only",
        "--warmup", "0",
        "--frames", "2",
        "--inflight", "1",
        "--json-out", str(report),
    ])

    assert hailo_runner.main() == 0
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["ok"] is True
    assert payload["normalization_frozen"] is True
    assert payload["postprocess_included"] is True
    assert payload["postprocess_completed_frames"] == 2
    assert payload["completed_task_endpoint_attested"] is True
    assert payload["completed_task_completion_mode"] == (
        "integrated_accelerator_plus_frozen_normalization"
    )
    assert payload["completed_task_result_artifact_saved"] is True
    assert payload["preprocessing_contract"] == preprocessing
    assert payload["preprocessing_contract_sha256"] == preprocessing_sha
    assert payload["frozen_decoded_nms_normalization_result"][
        "detections"
    ][0]["y1"] == pytest.approx(5.0)
    artifact_path = Path(payload["completed_task_result_artifact_path"])
    assert artifact_path.is_file()
    assert hashlib.sha256(artifact_path.read_bytes()).hexdigest() == (
        payload["completed_task_result_artifact_file_sha256"]
    )

    replay_report = tmp_path / "hailo_direct_energy_replay.json"
    monkeypatch.setattr(sys, "argv", [
        "smoke_hailo10_hef_runner.py",
        "--hef", str(hef),
        "--hw-arch", "hailo10h",
        "--artifacts-dir", str(tmp_path / "replay_artifacts"),
        "--runtime-input-bin", str(runtime_input),
        "--runtime-input-contract-json", json.dumps(runtime_contract),
        "--preverified-runtime-input-sha256", tensor_sha,
        "--canonical-input-slot-names-json", '["images"]',
        "--canonical-output-slot-names-json", '["detections"]',
        "--task", "detection",
        "--backend-label", "native_full_hailo10h",
        "--model", "yolo26s",
        "--frozen-decoded-nms-normalization-contract-json",
        json.dumps(
            payload[
                "frozen_decoded_nms_normalization_contract"
            ]
        ),
        "--original-image-wh-json", "[80,40]",
        "--throughput-mode",
        "--counted-hotloop-only",
        "--warmup", "0",
        "--frames", "2",
        "--inflight", "1",
        "--json-out", str(replay_report),
    ])
    assert hailo_runner.main() == 0
    replay = json.loads(replay_report.read_text(encoding="utf-8"))
    assert replay["normalization_frozen"] is True
    assert replay["host_postprocess_frozen"] is False
    assert replay["postprocess_completed_frames"] == 2
    assert replay["completed_task_result_artifact_saved"] is True
    assert replay[
        "frozen_decoded_nms_normalization_contract_sha256"
    ] == payload[
        "frozen_decoded_nms_normalization_contract_sha256"
    ]


def test_hailo_runner_asymmetric_letterbox_matches_shared_pixels(
    tmp_path: Path,
) -> None:
    yy, xx = np.indices((37, 53))
    source = np.stack([
        (xx * 7 + yy * 3) % 256,
        (xx * 11 + yy * 5 + 17) % 256,
        (xx * 13 + yy * 19 + 29) % 256,
    ], axis=-1).astype(np.uint8)
    image = tmp_path / "asymmetric.png"
    Image.fromarray(source, mode="RGB").save(image)
    contract = canonical_image_preprocessing_contract(
        "detection", (63, 79),
    )
    expected, expected_geometry = prepare_rgb_uint8_image(
        source, contract,
    )

    tensor, prepared, evidence = hailo_runner._image_tensor(
        image,
        (1, 63, 79, 3),
        quantized=True,
        task="detection",
        preprocess_mode="letterbox",
        letterbox_pad=114,
    )

    assert np.array_equal(prepared, expected)
    assert np.array_equal(tensor, expected[None, ...])
    assert hashlib.sha256(prepared.tobytes()).hexdigest() == (
        hashlib.sha256(expected.tobytes()).hexdigest()
    )
    assert evidence["preprocessing_contract"] == contract
    assert evidence["preprocessing_contract_sha256"] == (
        preprocessing_contract_sha256(contract)
    )
    assert evidence["preprocessing_geometry"] == expected_geometry


def test_hailo_native_full_bn6_without_authoritative_contract_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outputs = _direct_bn6_outputs()
    preprocessing = canonical_image_preprocessing_contract(
        "detection", (64, 64),
    )
    tensor = np.zeros((1, 64, 64, 3), dtype=np.uint8)
    runtime_input = tmp_path / "runtime_input.bin"
    runtime_input.write_bytes(tensor.tobytes())
    tensor_sha = hashlib.sha256(runtime_input.read_bytes()).hexdigest()
    runtime_contract = {
        "schema": "onnx-splitpoint/preverified-runtime-input-tensor",
        "schema_version": 1,
        "runtime_input_name": "images",
        "runtime_input_shape": list(tensor.shape),
        "runtime_input_dtype": "uint8",
        "runtime_input_bytes": int(tensor.nbytes),
        "runtime_input_sha256": tensor_sha,
        "preprocess": {
            "mode": "letterbox_rgb_uint8",
            "pad_value": 114,
            "rgb": True,
            "color_space": "RGB",
            "layout": "NHWC",
            "preprocessing_contract": preprocessing,
            "preprocessing_contract_sha256": (
                preprocessing_contract_sha256(preprocessing)
            ),
        },
    }
    hef = tmp_path / "compiled.hef"
    hef.write_bytes(b"hef")

    class _FakeHailoBackend:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def prepare(self, _cfg: object, _artifacts: Path) -> object:
            return SimpleNamespace(
                handle=SimpleNamespace(
                    runtime_input_shapes={"images": tensor.shape},
                    input_shapes={"images": tensor.shape},
                    runtime_output_shapes={
                        "detections": outputs["detections"].shape,
                    },
                    session=SimpleNamespace(hotloop=False),
                ),
                input_names=["images"],
                output_names=["detections"],
            )

        def run(self, _prepared: object, _values: object) -> object:
            return SimpleNamespace(outputs=outputs, metrics={"infer_ms": 1.0})

        def cleanup(self, _prepared: object) -> None:
            pass

    monkeypatch.setattr(hailo_runner, "HailoBackend", _FakeHailoBackend)
    monkeypatch.setattr(sys, "argv", [
        "smoke_hailo10_hef_runner.py",
        "--hef", str(hef),
        "--runtime-input-bin", str(runtime_input),
        "--runtime-input-contract-json", json.dumps(runtime_contract),
        "--preverified-runtime-input-sha256", tensor_sha,
        "--canonical-input-slot-names-json", '["images"]',
        "--canonical-output-slot-names-json", '["detections"]',
        "--task", "detection",
        "--backend-label", "native_full_hailo10h",
        "--model", "yolo26s",
        "--original-image-wh-json", "[80,40]",
        "--throughput-mode",
        "--counted-hotloop-only",
        "--warmup", "0",
        "--frames", "1",
        "--json-out", str(tmp_path / "must_not_be_ok.json"),
    ])

    with pytest.raises(RuntimeError, match="no authoritative"):
        hailo_runner.main()


def test_hailo_raw_reconciliation_atomically_removes_bn6_metadata(
    tmp_path: Path,
) -> None:
    suite = tmp_path / "benchmark_set"
    hef = suite / "hailo" / "hailo10" / "full" / "compiled.hef"
    hef.parent.mkdir(parents=True)
    hef.write_bytes(b"verified-hef")
    nodes = ["conv61", "conv64", "conv77", "conv80", "conv91", "conv94"]
    source_onnx, compiler_onnx = _seal_test_hailo_hef(
        hef,
        task="detection",
        hw_arch="hailo10h",
        end_nodes=nodes,
    )
    hef_sha = hashlib.sha256(hef.read_bytes()).hexdigest()
    suite_bench = {
        "hailo": {
            "hefs": {
                "hailo10": {
                    "full": str(hef),
                    "full_build": {
                        "ok": True,
                        "artifact_hash": hef_sha,
                        "source_onnx_path": str(source_onnx),
                        "compiler_onnx_path": str(compiler_onnx),
                    },
                    "full_endpoint_mode": "raw_detection_head",
                    "full_end_node_names": nodes,
                    "full_output_contract": {
                        "mode": "yolo_raw_head",
                        "requires_external_postprocess": True,
                        "end_node_names": nodes,
                    },
                },
            },
        },
    }
    contracts = [_raw_suite_contract("yolo26s")]
    promotions = _promote_verified_hailo_full_contracts(
        suite_dir=suite,
        model_id="yolo26s",
        task="detection",
        suite_bench=suite_bench,
        contracts=contracts,
        copied_verified={},
    )
    assert len(promotions) == 1
    contract = contracts[0]
    assert contract["stage"] == "raw_head"
    assert contract["contract_family"] == "raw_head"
    assert contract["endpoint_mode"] == "raw_detection_head"
    assert contract["output_format"] == "raw_detection_tensors"
    assert contract["compiled_artifact_raw_head"] is True
    assert contract["source_onnx_multiscale_raw_head"] is False
    for stale in (
        "output_record_format",
        "coordinate_format",
        "coordinate_space",
        "source_coordinate_space",
        "score_semantics",
        "class_id_semantics",
    ):
        assert stale not in contract

    (suite / "output_contracts.json").write_text(json.dumps({
        "model_id": "yolo26s",
        "task": "detection",
        "contracts": contracts,
    }), encoding="utf-8")
    resolved = load_authoritative_output_contract(
        suite,
        backend="hailo10h",
        model_id="yolo26s",
        task="detection",
    )
    assert resolved["contract_resolution_status"] == "attested"
    assert resolved["stage"] == "raw_head"
    assert resolved["output_format"] == "raw_detection_tensors"


def test_hailo_promotion_accepts_real_generator_model_hints_idempotently(
    tmp_path: Path,
) -> None:
    suite = tmp_path / "benchmark_set"
    hef = suite / "hailo" / "hailo10" / "full" / "compiled.hef"
    hef.parent.mkdir(parents=True)
    hef.write_bytes(b"generator-shaped-v275-hef")
    nodes = ["head_a", "head_b"]
    source_onnx, compiler_onnx = _seal_test_hailo_hef(
        hef, task="detection", hw_arch="hailo10h", end_nodes=nodes,
    )
    source_relative = source_onnx.relative_to(suite)
    compiler_relative = compiler_onnx.relative_to(suite)
    suite_bench = {
        # These are the actual generator fields.  Deliberately omit the old
        # synthetic source_onnx_path test-only alias.
        "model": str(source_relative),
        "model_source": str(source_onnx),
        "hailo": {"hefs": {"hailo10": {
            "full": str(hef.relative_to(suite)),
            "full_build": {
                "ok": True,
                "artifact_hash": hashlib.sha256(hef.read_bytes()).hexdigest(),
                "fixed_onnx_path": str(compiler_relative),
            },
            "full_endpoint_mode": "raw_detection_head",
            "full_end_node_names": nodes,
            "full_output_contract": {
                "mode": "yolo_raw_head",
                "requires_external_postprocess": True,
                "end_node_names": nodes,
            },
        }}},
    }
    contracts = [_raw_suite_contract("yolo26s")]

    first = _promote_verified_hailo_full_contracts(
        suite_dir=suite, model_id="yolo26s", task="detection",
        suite_bench=suite_bench, contracts=contracts, copied_verified={},
    )
    first_contract = json.loads(json.dumps(contracts[0]))
    second = _promote_verified_hailo_full_contracts(
        suite_dir=suite, model_id="yolo26s", task="detection",
        suite_bench=suite_bench, contracts=contracts, copied_verified={},
    )

    assert len(first) == 1
    assert len(second) == 1
    assert contracts[0] == first_contract
    assert contracts[0]["artifact_binding_status"] == "verified"
    assert len(contracts[0]["hailo_build_receipt_file_sha256"]) == 64
    assert len(contracts[0]["hailo_build_receipt_identity_sha256"]) == 64


def test_hailo_cache_hit_promotes_receipt_signed_compiler_sibling_without_build_result(
    tmp_path: Path,
) -> None:
    suite = tmp_path / "benchmark_set"
    hef = suite / "hailo/hailo10/full/compiled.hef"
    hef.parent.mkdir(parents=True)
    hef.write_bytes(b"cache-hit-hef")
    nodes = ["head_a", "head_b", "head_c"]
    source_onnx, compiler_onnx = _seal_test_hailo_hef(
        hef, task="detection", hw_arch="hailo10h", end_nodes=nodes,
    )
    suite_bench = {
        "model_source": str(source_onnx),
        "hailo": {"hefs": {"hailo10": {
            "full": str(hef),
            # Deliberately omit fixed_onnx_path, compiler_onnx_path and a
            # hailo_hef_build_result.json, as on the observed cache-hit path.
            "full_build": {
                "ok": True,
                "artifact_hash": hashlib.sha256(
                    hef.read_bytes()
                ).hexdigest(),
            },
            "full_endpoint_mode": "raw_detection_head",
            "full_end_node_names": nodes,
            "full_output_contract": {
                "mode": "yolo_raw_head",
                "requires_external_postprocess": True,
                "end_node_names": nodes,
            },
        }}},
    }
    contracts = [_raw_suite_contract("yolo26s")]

    promotions = _promote_verified_hailo_full_contracts(
        suite_dir=suite,
        model_id="yolo26s",
        task="detection",
        suite_bench=suite_bench,
        contracts=contracts,
        copied_verified={},
    )

    assert len(promotions) == 1
    assert contracts[0]["artifact_binding_status"] == "verified"
    assert contracts[0]["compiler_onnx_filename"] == compiler_onnx.name
    assert contracts[0]["compiler_onnx_path"] == (
        compiler_onnx.relative_to(suite).as_posix()
    )


def test_loader_rejects_mixed_raw_and_bn6_contract(tmp_path: Path) -> None:
    suite = tmp_path / "benchmark_set"
    suite.mkdir()
    contract = {
        "model_id": "yolo26s",
        "backend": "cpu_ort",
        "variant": "full",
        "task": "detection",
        "stage": "raw_head",
        "contract_family": "raw_head",
        "endpoint_mode": "raw_detection_head",
        "host_tail_required": True,
        "postprocessing_required": True,
        "requires_external_postprocess": True,
        "contract_status": "recorded",
        "output_format": "bn6_detections",
        "output_record_format": "xyxy_score_class",
    }
    (suite / "output_contracts.json").write_text(json.dumps({
        "model_id": "yolo26s",
        "task": "detection",
        "contracts": [contract],
    }), encoding="utf-8")
    resolved = load_authoritative_output_contract(
        suite,
        backend="cpu_ort",
        model_id="yolo26s",
        task="detection",
    )
    assert resolved["contract_resolution_status"] == "conflict"
    assert "raw_endpoint_output_format_conflict" in resolved[
        "contract_resolution_errors"
    ]
    assert "raw_endpoint_contains_decoded_detection_metadata" in resolved[
        "contract_resolution_errors"
    ]


@pytest.mark.parametrize("name", [
    "native_yolo_full_self_reference_probe.py",
    "smoke_hailo10_hef_runner.py",
    "native_trt_full_completed_hotloop.py",
    "native_full_baseline_eval_runner.py",
    "native_producer_validate_visualize.py",
    "native_producer_final_report.py",
    "native_deepx_trt_e2e_from_benchmarkset.py",
    "native_hailo10_trt_e2e_from_benchmarkset.py",
    "native_hailo_trt_fifo_from_benchmarkset.py",
    "native_producer_e2e_eval_runner.py",
])
def test_remote_script_mirrors_are_byte_identical(name: str) -> None:
    root = Path(__file__).resolve().parents[1]
    assert (root / "scripts" / name).read_bytes() == (
        root / "onnx_splitpoint_tool" / "resources" / "remote_scripts" / name
    ).read_bytes()
