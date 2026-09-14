from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from onnx_splitpoint_tool.runners.native_split_quality_runtime import (
    prepare_quality_first_boundary_input,
)
from scripts import native_producer_validate_visualize as validator
from tests import test_v269f_hailo_trt_interface_contract as interface_fixture


ROOT = Path(__file__).resolve().parents[1]


def _layout() -> dict[str, object]:
    return {
        "applied": True,
        "effective": "memory_nhwc_to_nchw",
        "memory_shape": [1, 2, 3, 4],
        "perm": [0, 3, 1, 2],
    }


@pytest.mark.parametrize("source_kind", ["hwc", "nhwc", "nchw"])
def test_quality_first_boundary_restores_physical_memory_order(
    source_kind: str,
) -> None:
    physical = np.arange(24, dtype=np.float32).reshape(1, 2, 3, 4)
    canonical = physical.transpose(0, 3, 1, 2)
    source = {
        "hwc": physical[0],
        "nhwc": physical,
        "nchw": canonical,
    }[source_kind]

    feed = prepare_quality_first_boundary_input(
        source,
        target_shape=(1, 4, 2, 3),
        target_dtype=np.float32,
        boundary_layout=_layout(),
        boundary_transform="layout_only",
    )

    simulated_engine_output = feed.reshape(1, 2, 3, 4).transpose(0, 3, 1, 2)
    assert feed.shape == (1, 4, 2, 3)
    assert feed.flags.c_contiguous
    np.testing.assert_array_equal(simulated_engine_output, canonical)


def test_quality_first_boundary_quantizes_before_internal_dequant_bridge() -> None:
    scale = 0.25
    zero_point = 17.0
    physical_uint8 = np.arange(24, dtype=np.uint8).reshape(1, 2, 3, 4)
    canonical_float = (
        physical_uint8.transpose(0, 3, 1, 2).astype(np.float32) - zero_point
    ) * scale

    feed = prepare_quality_first_boundary_input(
        canonical_float,
        target_shape=(1, 4, 2, 3),
        target_dtype=np.uint8,
        boundary_layout=_layout(),
        boundary_transform="uint8_dequant_then_layout",
        dequant_scale=scale,
        dequant_zero_point=zero_point,
    )

    simulated_engine_output = (
        (feed.reshape(1, 2, 3, 4).astype(np.float32) - zero_point) * scale
    ).transpose(0, 3, 1, 2)
    np.testing.assert_allclose(simulated_engine_output, canonical_float)


@pytest.mark.parametrize(
    ("source", "layout", "error"),
    [
        (
            np.zeros((1, 2, 4, 3), np.float32),
            _layout(),
            "native_split_quality_boundary_source_shape_not_declared",
        ),
        (
            np.zeros((1, 4, 2, 3), np.float32),
            {**_layout(), "perm": [0, 3, 3, 1]},
            "native_split_quality_boundary_layout_contract_invalid",
        ),
        (
            np.zeros((1, 4, 2, 3), np.float32),
            {**_layout(), "memory_shape": [1, 3, 2, 4]},
            "native_split_quality_boundary_layout_contract_invalid",
        ),
    ],
)
def test_quality_first_boundary_rejects_undeclared_layouts(
    source: np.ndarray,
    layout: dict[str, object],
    error: str,
) -> None:
    with pytest.raises(ValueError, match=error):
        prepare_quality_first_boundary_input(
            source,
            target_shape=(1, 4, 2, 3),
            target_dtype=np.float32,
            boundary_layout=layout,
            boundary_transform="layout_only",
        )


def test_generated_runner_binds_bridge_input_before_legacy_adapter() -> None:
    template = (
        ROOT
        / "onnx_splitpoint_tool/resources/templates/run_split_onnxruntime.py.txt"
    ).read_text(encoding="utf-8")
    helper_call = template.index("arr = prepare_quality_first_boundary_input(")
    early_return = template.index("return arr", helper_call)
    legacy_adapter = template.index("_try_adapt_tensor(", helper_call)

    assert "native split Quality-FIRST runtime input name mismatch" in template
    assert helper_call < early_return < legacy_adapter


def _yolov7_heads() -> dict[str, np.ndarray]:
    heads = {
        "yolo_head_80x80": np.full(
            (1, 3, 80, 80, 85), -20.0, dtype=np.float32,
        ),
        "yolo_head_40x40": np.full(
            (1, 3, 40, 40, 85), -20.0, dtype=np.float32,
        ),
        "yolo_head_20x20": np.full(
            (1, 3, 20, 20, 85), -20.0, dtype=np.float32,
        ),
    }
    strong = heads["yolo_head_80x80"][0, 0, 20, 30]
    strong[:4] = 0.0
    strong[4] = 20.0
    strong[5] = 20.0
    return heads


def test_validator_rejects_unbound_yolov7_multiscale_decoder() -> None:
    candidates = validator._decode_layout_candidates(
        _yolov7_heads(), img_w=640, img_h=640, conf=0.25,
    )
    canonical = [
        candidate
        for candidate in candidates
        if candidate.get("kind") == "raw_yolo_multiscale"
    ]

    assert canonical == []
    rejected = [
        candidate
        for candidate in candidates
        if candidate.get("kind")
        == "rejected_unbound_yolov7_model_contract"
    ]
    assert len(rejected) == 1
    assert rejected[0]["detections"] == []
    assert rejected[0]["debug"]["claim_capable"] is False


def _probe_payload(recorded_manifest: Path, *, version: int = 3) -> dict:
    payload = {
        "schema": "onnx-splitpoint/native-yolo-full-self-reference-probe",
        "schema_version": version,
        "native_output_manifest": str(recorded_manifest),
        "expected_contract_family": "raw_head",
        "contract_family_match": True,
        "diagnosis": "native_semantic_matches_full_self_reference",
        "semantic_available": True,
        "semantic_ok": True,
        "ok": True,
        "best": {
            "native_mode": "canonical_multiscale:raw",
            "full_mode": "canonical_multiscale:raw",
            "match": {
                "matched": 1, "ref_count": 1, "match_ratio": 1.0,
            },
        },
    }
    if version >= 4:
        payload["native_output_manifest_sha256"] = hashlib.sha256(
            recorded_manifest.read_bytes()
        ).hexdigest()
    return payload


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def test_cached_split_probe_cannot_validate_native_full_or_other_model(
    tmp_path: Path,
) -> None:
    target = (
        tmp_path / "native_producers/hailo10h/yolov7_paper/benchmark_set"
        / "native_full_outputs/model=yolov7_paper"
        / "backend=native_full_hailo10h/setup=setup/comparison=hailo10h"
        / "native_full_outputs_manifest.json"
    )
    _write_json(target, {"output_contract": "raw_head"})
    foreign_manifest = (
        tmp_path / "native_producers/hailo8/yolo26s/benchmark_set"
        / "native_pipeline/b038/hailo_to_trt/uint8_dequant_fp16"
        / "native_outputs/native_outputs_manifest.json"
    )
    _write_json(foreign_manifest, {"output_contract": "raw_head"})
    foreign_probe = (
        foreign_manifest.parent.parent
        / "native_yolo_full_self_reference_probe.json"
    )
    _write_json(foreign_probe, _probe_payload(foreign_manifest))

    result = validator._precomputed_full_self_reference_detection(
        target, roots=[tmp_path],
    )

    assert result is None
    assert validator._find_boundary_manifest_for_output(
        target, roots=[tmp_path],
    ) is None


def test_generic_cached_probe_requires_exact_legacy_workdir_or_v4_manifest_hash(
    tmp_path: Path,
) -> None:
    target = (
        tmp_path / "copy/yolo26s/benchmark_set"
        / "native_pipeline/b044/hailo10h_to_trt/float32_layout_fp16"
        / "native_outputs/native_outputs_manifest.json"
    )
    _write_json(target, {"output_contract": "raw_head"})
    workdir = target.parent.parent
    legacy_probe = workdir / "native_yolo_full_self_reference_probe.json"
    _write_json(legacy_probe, _probe_payload(target))
    assert validator._precomputed_full_self_reference_detection(target) is not None

    legacy_probe.unlink()
    rebased_recorded = Path(
        "/remote/yolo26s/benchmark_set/native_pipeline/b044/"
        "hailo10h_to_trt/float32_layout_fp16/native_outputs/"
        "native_outputs_manifest.json"
    )
    v4_probe = (
        tmp_path / "remote_cache/yolo26s/benchmark_set"
        / "native_pipeline/b044/hailo10h_to_trt/float32_layout_fp16"
        / "native_yolo_full_self_reference_probe.json"
    )
    v4_payload = _probe_payload(target, version=6)
    v4_payload["native_output_manifest"] = str(rebased_recorded)
    _write_json(v4_probe, v4_payload)
    assert validator._precomputed_full_self_reference_detection(
        target, roots=[tmp_path / "remote_cache"],
    ) is not None

    v4_payload["native_output_manifest_sha256"] = "0" * 64
    _write_json(v4_probe, v4_payload)
    assert validator._precomputed_full_self_reference_detection(
        target, roots=[tmp_path / "remote_cache"],
    ) is None


def test_schema6_cached_yolov7_raw_head_probe_is_never_claim_admissible(
    tmp_path: Path,
) -> None:
    target = (
        tmp_path / "copy/yolov7_paper/benchmark_set"
        / "native_pipeline/b044/hailo10h_to_trt/float32_layout_fp16"
        / "native_outputs/native_outputs_manifest.json"
    )
    _write_json(target, {
        "model_id": "yolov7_paper",
        "output_contract": "raw_head",
    })
    workdir = target.parent.parent
    probe = workdir / "native_yolo_full_self_reference_probe.json"
    payload = _probe_payload(target, version=6)
    _write_json(probe, payload)

    assert validator._cached_probe_matches_native_output(
        payload,
        probe_path=probe,
        target_manifest=target,
        target_workdir=workdir,
        target_contract_family="raw_head",
    ) is False
    assert validator._precomputed_full_self_reference_detection(target) is None

    result = validator._full_onnx_self_reference_detection(
        target, tmp_path,
    )
    assert result["available"] is False
    assert result["ok"] is False
    assert result["semantic_ok"] is False
    assert result["semantic_available"] is False
    assert result["contract_family_match"] is False
    assert result["diagnosis"] == (
        "yolov7_paper_raw_head_requires_model_bound_decoder_contract"
    )


def test_path_hints_recognize_yolov7_and_native_full() -> None:
    hints = validator._case_backend_precision_model_hints(
        "/run/yolov7_paper/benchmark_set/native_full_outputs/"
        "model=yolov7_paper/backend=native_full_hailo10h/"
        "setup=s/comparison=hailo10h/native_full_outputs_manifest.json"
    )
    assert hints[1] == "native_full_hailo10h"
    assert hints[3] == "yolov7_paper"


def test_model_scoped_image_resolution_rejects_cross_model_and_ambiguity(
    tmp_path: Path,
) -> None:
    filename = "000000005600.jpg"
    yolo26 = (
        tmp_path / "native_producers/hailo8/yolo26s/benchmark_set"
        / "resources/validation/detection/set" / filename
    )
    yolov7 = (
        tmp_path / "native_producers/hailo8/yolov7_paper/benchmark_set"
        / "resources/validation/detection/set" / filename
    )
    yolo26.parent.mkdir(parents=True)
    yolov7.parent.mkdir(parents=True)
    yolo26.write_bytes(b"yolo26")
    yolov7.write_bytes(b"yolov7")
    manifest = (
        yolov7.parents[3] / "native_pipeline/b044/backend/precision"
        / "native_outputs/native_outputs_manifest.json"
    )
    _write_json(manifest, {})

    resolved = validator._find_image(
        tmp_path,
        filename,
        model="yolov7_paper",
        manifest=manifest,
        roots=[tmp_path],
    )
    assert resolved == yolov7.resolve()

    duplicate = yolov7.parents[2] / "other" / filename
    duplicate.parent.mkdir(parents=True)
    duplicate.write_bytes(b"different-yolov7-image")
    assert validator._find_image(
        tmp_path,
        filename,
        model="yolov7_paper",
        manifest=manifest,
        roots=[tmp_path],
    ) is None


def test_hailo_interface_uses_portable_bound_metadata_when_file_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = interface_fixture._fixture(tmp_path)
    metadata_size = fixture["metadata_path"].stat().st_size
    metadata_sha = interface_fixture._sha256_file(fixture["metadata_path"])
    portable_binding = {
        "artifacts": {
            "native_trt_meta": {
                "path": str(fixture["remote_metadata"]),
                "sha256": metadata_sha,
                "size_bytes": metadata_size,
            },
        },
        "native_trt_meta_file_sha256": metadata_sha,
        "native_trt_meta_file_size_bytes": metadata_size,
        "native_trt_meta_payload": copy.deepcopy(fixture["metadata"]),
    }
    marker = {"binding_sha256": "portable-test-marker"}
    fixture["row"]["native_split_quality_binding"] = marker
    fixture["metadata_path"].unlink()
    observed: dict[str, object] = {}

    def _portable_bind(*, native_row, quality_binding, verification_mode):
        observed.update(native_row)
        assert quality_binding == marker
        assert verification_mode == "portable"
        return copy.deepcopy(portable_binding), "portable_verified"

    monkeypatch.setattr(
        interface_fixture.VALIDATOR,
        "bind_quality_to_native_split",
        _portable_bind,
    )

    result = interface_fixture._validate(fixture)

    assert result["interface_contract_pass"] is True
    verification = result["interface_contract_verification"]
    assert verification["metadata_resolution"] == (
        "portable_native_split_quality_binding"
    )
    assert verification["metadata_sha256"] == metadata_sha
    assert observed["backend"] == "hailo8_to_trt"
    assert observed["model"] == "resnet50"
    assert observed["case"] == "b052"


def test_hailo_interface_accepts_declared_batchless_physical_memory_shape(
    tmp_path: Path,
) -> None:
    fixture = interface_fixture._fixture(tmp_path)
    fixture["metadata"]["uint8_cast_bridge"].update({
        "replaced_uses": 1,
        "boundary_layout": {
            "requested": "memory_nhwc_to_nchw",
            "effective": "memory_nhwc_to_nchw",
            "applied": True,
            "memory_shape": [1, 2, 2, 2],
            "perm": [0, 3, 1, 2],
        },
    })
    fixture["manifest"]["shape"] = [2, 2, 2]
    interface_fixture._write_json(
        fixture["boundary_manifest"], fixture["manifest"],
    )
    interface_fixture._rebind_contract(fixture)

    result = interface_fixture._validate(fixture)

    assert result["interface_contract_pass"] is True
    assert result["interface_contract_status"] == (
        "verified_native_command_metadata_boundary_and_bridge"
    )


def test_hailo_interface_rejects_invalid_portable_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = interface_fixture._fixture(tmp_path)
    fixture["row"]["native_split_quality_binding"] = {
        "binding_sha256": "invalid",
    }
    fixture["metadata_path"].unlink()
    monkeypatch.setattr(
        interface_fixture.VALIDATOR,
        "bind_quality_to_native_split",
        lambda **_kwargs: (None, "tampered_embedded_bytes"),
    )

    result = interface_fixture._validate(fixture)

    assert result["interface_contract_pass"] is False
    assert result["interface_contract_status"] == (
        "native_trt_metadata_portable_binding_invalid:"
        "tampered_embedded_bytes"
    )
