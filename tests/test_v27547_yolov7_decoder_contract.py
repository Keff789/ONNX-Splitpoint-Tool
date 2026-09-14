from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest

from onnx_splitpoint_tool.native_detection_postprocess import (
    FrozenDetectionPostprocessor,
    FrozenPostprocessError,
    build_completed_detection_comparison_endpoint_contract,
    build_completed_detection_endpoint_contract,
    build_detection_completion_execution_contract,
    build_frozen_postprocess_contract,
    build_yolov7_head_mapping,
    canonical_json_sha256,
    tensor_signature,
    verify_completed_detection_comparison_endpoint_contract,
    verify_detection_completion_execution_contract,
    YOLOV7_LEGACY_TINY_ANCHORS_BY_STRIDE,
)
from onnx_splitpoint_tool.runners.harness.base import (
    postprocess_result_to_dict,
)
from onnx_splitpoint_tool.runners.harness.yolo import (
    YOLOV7_LEGACY_TINY_ANCHOR_TABLE_ID,
    YOLOV7_PAPER_MODEL_ID,
    YOLOV7_PAPER_ONNX_SHA256,
    YOLOV7_STANDARD_ANCHOR_TABLE_ID,
    YOLOV7_STANDARD_ANCHORS_640,
    YoloHarness,
    registered_yolov7_decoder_contract,
    verify_yolov7_decoder_contract,
)
from onnx_splitpoint_tool.validation import official_coco
from scripts import probe_yolov7_decoder_ab as probe
from scripts import native_producer_validate_visualize as validator


@pytest.fixture(scope="module")
def yolov7_heads() -> dict[str, np.ndarray]:
    heads = {
        "vendor_p5": np.full(
            (1, 3, 20, 20, 85), -20.0, dtype=np.float32,
        ),
        "vendor_p3": np.full(
            (1, 3, 80, 80, 85), -20.0, dtype=np.float32,
        ),
        "vendor_p4": np.full(
            (1, 3, 40, 40, 85), -20.0, dtype=np.float32,
        ),
    }
    record = heads["vendor_p3"][0, 0, 10, 10]
    record[:4] = 0.0
    record[4] = 20.0
    record[5 + 11] = 20.0
    return heads


def _contract(*, variant: str = "standard", policy: str = "production") -> dict:
    return registered_yolov7_decoder_contract(
        model_id=YOLOV7_PAPER_MODEL_ID,
        model_sha256=YOLOV7_PAPER_ONNX_SHA256,
        activation_mode="logits",
        variant=variant,
        policy=policy,
    )


def _rows(result: object) -> list[dict[str, object]]:
    payload = postprocess_result_to_dict(result)["json"]
    return [
        {key: row[key] for key in ("class_id", "score", "x1", "y1", "x2", "y2")}
        for row in payload["detections"]
    ]


def _reseal_endpoint(endpoint: dict, **changes: object) -> dict:
    identity = {
        key: value
        for key, value in endpoint.items()
        if key not in {
            "endpoint_contract_complete", "endpoint_contract_hash",
            "output_endpoint_id",
        }
    }
    identity.update(changes)
    digest = canonical_json_sha256(identity)
    return {
        **identity,
        "endpoint_contract_complete": True,
        "endpoint_contract_hash": digest,
        "output_endpoint_id": f"detection:decoded_nms:comparison:{digest}",
    }


def test_registered_contract_is_exact_model_bound_and_hash_stable() -> None:
    contract = _contract()
    assert contract["model_sha256"] == YOLOV7_PAPER_ONNX_SHA256
    assert contract["input_hw"] == [640, 640]
    assert contract["anchor_table_id"] == YOLOV7_STANDARD_ANCHOR_TABLE_ID
    assert [row["stride"] for row in contract["anchors_by_stride"]] == [8, 16, 32]
    assert [row["anchors_wh"] for row in contract["anchors_by_stride"]] == [
        [[12, 16], [19, 36], [40, 28]],
        [[36, 75], [76, 55], [72, 146]],
        [[142, 110], [192, 243], [459, 401]],
    ]
    assert verify_yolov7_decoder_contract(contract) == contract
    assert len(contract["decoder_contract_sha256"]) == 64
    assert _contract() == contract


@pytest.mark.parametrize("bad_hash", ["", "0" * 64, "not-a-sha256"])
def test_registered_contract_rejects_missing_wrong_or_invalid_model_hash(
    bad_hash: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match="model_sha256"):
        registered_yolov7_decoder_contract(
            model_id=YOLOV7_PAPER_MODEL_ID,
            model_sha256=bad_hash,
            activation_mode="logits",
        )


def test_registered_contract_requires_exact_640_and_valid_contract_hash() -> None:
    with pytest.raises(ValueError, match="input_hw_invalid"):
        registered_yolov7_decoder_contract(
            model_id=YOLOV7_PAPER_MODEL_ID,
            model_sha256=YOLOV7_PAPER_ONNX_SHA256,
            activation_mode="logits",
            input_hw=[608, 608],
        )
    tampered = copy.deepcopy(_contract())
    tampered["anchors_by_stride"][0]["anchors_wh"][0][0] = 13
    with pytest.raises(ValueError, match="sha256_mismatch"):
        verify_yolov7_decoder_contract(tampered)


def test_legacy_tiny_and_standard_vary_only_pre_registered_anchor_fields() -> None:
    legacy = _contract(variant="legacy_tiny")
    standard = _contract()
    assert legacy["anchor_table_id"] == YOLOV7_LEGACY_TINY_ANCHOR_TABLE_ID
    assert probe._contract_diff_keys(legacy, standard) == [
        "anchor_table_id", "anchors_by_stride",
        "decoder_contract_sha256", "use_scope",
    ]
    assert probe._contract_pairing_identity(legacy) == (
        probe._contract_pairing_identity(standard)
    )


def test_exact_yolov7_raw_head_has_no_silent_legacy_default(
    yolov7_heads: dict[str, np.ndarray],
) -> None:
    with pytest.raises(ValueError, match="requires_model_bound_decoder"):
        YoloHarness(model_id=YOLOV7_PAPER_MODEL_ID).postprocess(
            yolov7_heads, {"input_hw": [640, 640]}
        )


def test_exact_yolov7_geometry_has_no_unidentified_tiny_fallback(
    yolov7_heads: dict[str, np.ndarray],
) -> None:
    with pytest.raises(ValueError, match="requires_model_bound_decoder"):
        YoloHarness().postprocess(
            yolov7_heads, {"input_hw": [640, 640]},
        )

    candidates = validator._decode_layout_candidates(
        yolov7_heads, img_w=640, img_h=640, conf=0.25,
    )
    assert not any(
        candidate.get("kind") == "raw_yolo_multiscale"
        for candidate in candidates
    )
    rejected = [
        candidate for candidate in candidates
        if candidate.get("kind")
        == "rejected_unbound_yolov7_model_contract"
    ]
    assert len(rejected) == 1
    assert rejected[0]["detections"] == []
    assert rejected[0]["debug"]["claim_capable"] is False


@pytest.mark.parametrize(
    "outputs",
    [
        {"detections": np.zeros((1, 1, 6), dtype=np.float32)},
        {"unknown": np.zeros((7,), dtype=np.float32)},
    ],
    ids=["bn6", "unknown"],
)
def test_bound_yolov7_contract_cannot_bypass_into_another_decoder(
    outputs: dict[str, np.ndarray],
) -> None:
    harness = YoloHarness(
        model_id=YOLOV7_PAPER_MODEL_ID,
        multiscale_decoder_contract=_contract(),
    )
    with pytest.raises(ValueError, match="output_format_mismatch"):
        harness.postprocess(outputs, {"input_hw": [640, 640]})


def test_standard_and_legacy_anchor_decodes_are_observably_different(
    yolov7_heads: dict[str, np.ndarray],
) -> None:
    standard = YoloHarness(
        model_id=YOLOV7_PAPER_MODEL_ID,
        multiscale_decoder_contract=_contract(),
    )
    legacy = YoloHarness(
        model_id=YOLOV7_PAPER_MODEL_ID,
        multiscale_decoder_contract=_contract(variant="legacy_tiny"),
        allow_diagnostic_yolov7_contract=True,
    )
    standard_rows = _rows(standard.postprocess(
        yolov7_heads, {"input_hw": [640, 640], "original_wh": [640, 640]},
    ))
    legacy_rows = _rows(legacy.postprocess(
        yolov7_heads, {"input_hw": [640, 640], "original_wh": [640, 640]},
    ))
    assert len(standard_rows) == len(legacy_rows) == 1
    assert standard_rows[0]["class_id"] == legacy_rows[0]["class_id"] == 11
    assert standard_rows[0] != legacy_rows[0]


def test_generic_and_native_use_the_same_standard_contract(
    yolov7_heads: dict[str, np.ndarray],
) -> None:
    frozen = build_frozen_postprocess_contract(
        model_id=YOLOV7_PAPER_MODEL_ID,
        model_sha256=YOLOV7_PAPER_ONNX_SHA256,
        outputs=yolov7_heads,
        input_hw=[640, 640],
        original_wh=[640, 640],
    )
    generic = YoloHarness(
        model_id=YOLOV7_PAPER_MODEL_ID,
        multiscale_decoder_contract=frozen["model_bound_decoder_contract"],
    )
    generic_rows = _rows(generic.postprocess(
        yolov7_heads, {"input_hw": [640, 640], "original_wh": [640, 640]},
    ))
    native = FrozenDetectionPostprocessor(frozen).process(
        yolov7_heads, original_wh=[640, 640],
    )
    assert generic_rows == native["detections"]
    assert frozen["anchor_table_id"] == YOLOV7_STANDARD_ANCHOR_TABLE_ID
    assert frozen["model_sha256"] == YOLOV7_PAPER_ONNX_SHA256


def test_head_mapping_is_stride_ordered_and_binds_standard_values(
    yolov7_heads: dict[str, np.ndarray],
) -> None:
    mapping = build_yolov7_head_mapping(yolov7_heads, input_hw=[640, 640])
    assert mapping["schema_version"] == 2
    assert mapping["anchor_table_id"] == YOLOV7_STANDARD_ANCHOR_TABLE_ID
    assert [row["stride"] for row in mapping["heads"]] == [8, 16, 32]
    assert [row["anchor_wh"] for row in mapping["heads"]] == [
        np.asarray(YOLOV7_STANDARD_ANCHORS_640[stride]).astype(int).tolist()
        for stride in (8, 16, 32)
    ]
    invalid = dict(yolov7_heads)
    invalid.pop("vendor_p5")
    with pytest.raises(FrozenPostprocessError, match="exactly_three_outputs"):
        build_yolov7_head_mapping(invalid, input_hw=[640, 640])


def test_head_mapping_binds_each_supported_activation_semantics(
    yolov7_heads: dict[str, np.ndarray],
) -> None:
    logits = build_yolov7_head_mapping(yolov7_heads, input_hw=[640, 640])
    activated_heads = {
        name: np.zeros_like(value) for name, value in yolov7_heads.items()
    }
    activated = build_yolov7_head_mapping(
        activated_heads, input_hw=[640, 640],
    )
    objcls_heads = {
        name: np.zeros_like(value) for name, value in yolov7_heads.items()
    }
    for value in objcls_heads.values():
        value[..., :4] = -2.0
    objcls = build_yolov7_head_mapping(
        objcls_heads, input_hw=[640, 640],
    )
    assert logits["activation_mode"] == "logits"
    assert logits["objectness_class_combination"] == (
        "sigmoid_objectness_times_sigmoid_class"
    )
    assert activated["activation_mode"] == "activated"
    assert activated["objectness_class_combination"] == (
        "objectness_probability_times_class_probability"
    )
    assert objcls["activation_mode"] == "objcls_activated"
    assert "xywh_logits" in objcls["head_record_semantics"]


@pytest.mark.parametrize(
    "mutation",
    [
        "rectangular", "duplicate", "missing", "extra",
        "ch84", "ch86", "na2", "na4", "nonfinite",
    ],
)
def test_model_bound_decoder_rejects_non_exact_head_sets(
    yolov7_heads: dict[str, np.ndarray], mutation: str,
) -> None:
    outputs = dict(yolov7_heads)
    if mutation == "rectangular":
        outputs["vendor_p3"] = np.zeros(
            (1, 3, 80, 40, 85), dtype=np.float32,
        )
    elif mutation == "duplicate":
        outputs["vendor_p5"] = outputs["vendor_p4"]
    elif mutation == "missing":
        outputs.pop("vendor_p5")
    elif mutation == "extra":
        outputs["unexpected_extra"] = outputs["vendor_p5"]
    elif mutation in {"ch84", "ch86"}:
        channels = int(mutation[2:])
        outputs["vendor_p3"] = np.zeros(
            (1, 3, 80, 80, channels), dtype=np.float32,
        )
    elif mutation in {"na2", "na4"}:
        anchors = int(mutation[2:])
        outputs["vendor_p3"] = np.zeros(
            (1, anchors, 80, 80, 85), dtype=np.float32,
        )
    else:
        nonfinite = outputs["vendor_p3"].copy()
        nonfinite[0, 0, 0, 0, 0] = np.nan
        outputs["vendor_p3"] = nonfinite
    harness = YoloHarness(
        model_id=YOLOV7_PAPER_MODEL_ID,
        multiscale_decoder_contract=_contract(),
    )
    with pytest.raises(ValueError, match="yolov7_decoder_contract"):
        harness.postprocess(outputs, {"input_hw": [640, 640]})


@pytest.mark.parametrize(
    ("kwargs", "changed"),
    [
        ({"conf_thresh": 0.20}, "confidence"),
        ({"iou_thresh": 0.50}, "iou"),
        ({"max_det": 100}, "max_det"),
    ],
)
def test_model_bound_decoder_rejects_runtime_nms_policy_mismatch(
    kwargs: dict[str, object], changed: str,
) -> None:
    with pytest.raises(
        ValueError, match="yolov7_decoder_contract_nms_policy_mismatch",
    ):
        YoloHarness(
            model_id=YOLOV7_PAPER_MODEL_ID,
            multiscale_decoder_contract=_contract(),
            **kwargs,
        )
    assert changed


def test_probe_generic_rows_use_native_canonical_tie_order() -> None:
    payload = {
        "detections": [
            {"class_id": 2, "score": 0.5, "x1": 5, "y1": 1,
             "x2": 8, "y2": 4},
            {"class_id": 1, "score": 0.5, "x1": 7, "y1": 1,
             "x2": 9, "y2": 4},
            {"class_id": 1, "score": 0.5, "x1": 3, "y1": 1,
             "x2": 6, "y2": 4},
        ],
    }
    rows = probe._canonical_detection_rows(payload)
    assert [(row["class_id"], row["x1"]) for row in rows] == [
        (1, 3.0), (1, 7.0), (2, 5.0),
    ]


def test_completed_endpoint_binds_model_contract_anchor_id_values_and_hash(
    yolov7_heads: dict[str, np.ndarray],
) -> None:
    frozen = build_frozen_postprocess_contract(
        model_id=YOLOV7_PAPER_MODEL_ID,
        model_sha256=YOLOV7_PAPER_ONNX_SHA256,
        outputs=yolov7_heads,
        input_hw=[640, 640],
        original_wh=[640, 640],
    )
    endpoint = build_completed_detection_endpoint_contract(frozen)
    assert frozen["schema_version"] == 2
    assert endpoint["schema_version"] == 2
    assert endpoint["model_sha256"] == YOLOV7_PAPER_ONNX_SHA256
    assert endpoint["anchor_table_id"] == YOLOV7_STANDARD_ANCHOR_TABLE_ID
    assert endpoint["anchors_by_stride"] == frozen["anchors_by_stride"]
    assert endpoint["model_bound_decoder_contract_sha256"] == (
        frozen["model_bound_decoder_contract_sha256"]
    )


def _raw_source(outputs: dict[str, np.ndarray]) -> dict:
    signature = tensor_signature(outputs)
    endpoint_hash = "e" * 64
    return {
        "task": "detection",
        "stage": "raw_head",
        "contract_family": "raw_head",
        "endpoint_contract_complete": True,
        "endpoint_contract_hash": endpoint_hash,
        "model_sha256": YOLOV7_PAPER_ONNX_SHA256,
        "tensor_signature": signature,
        "output_endpoint_attestation": {
            "attested": True,
            "status": "passed",
            "stage": "raw_head",
            "endpoint": "raw_head",
            "endpoint_contract_hash": endpoint_hash,
            "tensor_signature": signature,
        },
    }


def _reseal_execution(contract: dict, mapping: dict) -> dict:
    identity = {
        key: value for key, value in contract.items()
        if key != "contract_sha256"
    }
    identity["yolov7_head_mapping"] = mapping
    return {**identity, "contract_sha256": canonical_json_sha256(identity)}


def test_yolov7_completion_accepts_hashless_raw_source_and_binds_registry(
    yolov7_heads: dict[str, np.ndarray],
) -> None:
    source = _raw_source(yolov7_heads)
    source.pop("model_sha256")
    execution = build_detection_completion_execution_contract(
        model_id=YOLOV7_PAPER_MODEL_ID,
        outputs=yolov7_heads,
        input_hw=[640, 640],
        original_wh=[640, 640],
        source_endpoint_contract=source,
    )

    assert "model_sha256" not in execution["source_endpoint"]
    assert execution["processor_contract"]["model_sha256"] == (
        YOLOV7_PAPER_ONNX_SHA256
    )
    assert verify_detection_completion_execution_contract(execution) == execution


def test_yolov7_completion_rejects_wrong_explicit_source_hash(
    yolov7_heads: dict[str, np.ndarray],
) -> None:
    source = _raw_source(yolov7_heads)
    source["model_sha256"] = "0" * 64

    with pytest.raises(
        FrozenPostprocessError,
        match="completion_source_yolov7_model_sha256_mismatch",
    ):
        build_detection_completion_execution_contract(
            model_id=YOLOV7_PAPER_MODEL_ID,
            outputs=yolov7_heads,
            input_hw=[640, 640],
            original_wh=[640, 640],
            source_endpoint_contract=source,
        )


def test_yolov7_completion_accepts_correct_explicit_source_hash(
    yolov7_heads: dict[str, np.ndarray],
) -> None:
    execution = build_detection_completion_execution_contract(
        model_id=YOLOV7_PAPER_MODEL_ID,
        outputs=yolov7_heads,
        input_hw=[640, 640],
        original_wh=[640, 640],
        source_endpoint_contract=_raw_source(yolov7_heads),
    )

    assert execution["source_endpoint"]["model_sha256"] == (
        YOLOV7_PAPER_ONNX_SHA256
    )
    assert verify_detection_completion_execution_contract(execution) == execution


def test_current_execution_rejects_legacy_or_activation_mismatched_mapping(
    yolov7_heads: dict[str, np.ndarray],
) -> None:
    execution = build_detection_completion_execution_contract(
        model_id=YOLOV7_PAPER_MODEL_ID,
        outputs=yolov7_heads,
        input_hw=[640, 640],
        original_wh=[640, 640],
        source_endpoint_contract=_raw_source(yolov7_heads),
    )
    verify_detection_completion_execution_contract(execution)

    mapping = copy.deepcopy(execution["yolov7_head_mapping"])
    mapping.pop("head_mapping_sha256")
    mapping.pop("activation_mode")
    mapping["schema_version"] = 1
    mapping["anchor_table_id"] = "yolov5_yolov7_640_anchor_table_v1"
    mapping["head_record_semantics"] = (
        "batch_anchor_grid_y_grid_x_xywh_objectness_class_logits"
    )
    mapping["objectness_class_combination"] = (
        "sigmoid_objectness_times_sigmoid_class"
    )
    for head in mapping["heads"]:
        head["anchor_wh"] = [
            list(pair)
            for pair in YOLOV7_LEGACY_TINY_ANCHORS_BY_STRIDE[
                int(head["stride"])
            ]
        ]
    mapping["head_mapping_sha256"] = canonical_json_sha256(mapping)
    with pytest.raises(FrozenPostprocessError, match="mapping_decoder_mismatch"):
        verify_detection_completion_execution_contract(
            _reseal_execution(execution, mapping)
        )

    activation_mismatch = copy.deepcopy(execution["yolov7_head_mapping"])
    activation_mismatch.pop("head_mapping_sha256")
    activation_mismatch["activation_mode"] = "activated"
    activation_mismatch["head_record_semantics"] = (
        "batch_anchor_grid_y_grid_x_xywh_objectness_class_probabilities"
    )
    activation_mismatch["objectness_class_combination"] = (
        "objectness_probability_times_class_probability"
    )
    activation_mismatch["head_mapping_sha256"] = canonical_json_sha256(
        activation_mismatch
    )
    with pytest.raises(FrozenPostprocessError, match="mapping_decoder_mismatch"):
        verify_detection_completion_execution_contract(
            _reseal_execution(execution, activation_mismatch)
        )


def test_v1_decoder_and_format_mutations_are_rejected(
    yolov7_heads: dict[str, np.ndarray],
) -> None:
    frozen = build_frozen_postprocess_contract(
        model_id=YOLOV7_PAPER_MODEL_ID,
        model_sha256=YOLOV7_PAPER_ONNX_SHA256,
        outputs=yolov7_heads,
        input_hw=[640, 640],
        original_wh=[640, 640],
    )
    endpoint = build_completed_detection_comparison_endpoint_contract(
        frozen, schema_version=1,
    )
    verify_completed_detection_comparison_endpoint_contract(endpoint)
    for changes in (
        {"decoder_semantics_id": "wrong_decoder"},
        {"decoder_format": "wrong_format"},
    ):
        with pytest.raises(FrozenPostprocessError):
            verify_completed_detection_comparison_endpoint_contract(
                _reseal_endpoint(endpoint, **changes)
            )


def test_yolo26_keeps_v27546_schema_versions() -> None:
    outputs: dict[str, np.ndarray] = {}
    conv = 61
    for side in (80, 40, 20):
        outputs[f"yolo26s_full/conv{conv}"] = np.zeros(
            (side, side, 4), dtype=np.float32,
        )
        outputs[f"yolo26s_full/conv{conv + 3}"] = np.full(
            (side, side, 80), -20.0, dtype=np.float32,
        )
        conv += 16
    frozen = build_frozen_postprocess_contract(
        model_id="yolo26s", outputs=outputs,
        input_hw=[640, 640], original_wh=[640, 480],
    )
    endpoint = build_completed_detection_endpoint_contract(frozen)
    assert frozen["schema_version"] == 1
    assert endpoint["schema_version"] == 1
    assert "model_bound_decoder_contract" not in frozen
    assert "anchor_table_id" not in endpoint


def test_probe_preflight_is_pinned_and_coco_mapping_is_official(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert probe.COCO80_CATEGORY_IDS[11] == 13
    assert hashlib.sha256(
        probe._canonical_bytes(probe.COCO80_CATEGORY_IDS)
    ).hexdigest() == probe.COCO80_CATEGORY_IDS_SHA256
    coco = probe._to_coco([{
        "class_id": 11, "score": 0.9,
        "x1": 1.0, "y1": 2.0, "x2": 6.0, "y2": 10.0,
    }], image_id=42)
    assert coco == [{
        "image_id": 42, "category_id": 13,
        "bbox": [1.0, 2.0, 5.0, 8.0], "score": 0.9,
    }]
    with pytest.raises(probe.ProbeError, match="out_of_range"):
        probe._to_coco([{
            "class_id": 80, "score": 0.9,
            "x1": 1, "y1": 1, "x2": 2, "y2": 2,
        }], image_id=42)

    monkeypatch.setattr(probe, "_dependency_preflight", lambda **_kw: {})
    monkeypatch.setattr(
        probe, "_implementation_provenance", lambda: {
            "source_manifest": {
                "available": True,
                "package_version": probe.tool_identity.__version__,
                "workflow_version": probe.tool_identity.__build_id__,
                "all_implementation_files_listed_and_matching": True,
            },
        },
    )
    files = {}
    for name in ("model", "manifest", "request", "annotations"):
        path = tmp_path / name
        path.write_text("{}", encoding="utf-8")
        files[name] = path
    args = argparse.Namespace(
        model=str(files["model"]),
        validation_manifest=str(files["manifest"]),
        selection_request=str(files["request"]),
        annotations=str(files["annotations"]),
        images_root="", output_dir=str(tmp_path / "out"),
        archive_path="", overlay_count=0, preflight_only=True,
        install_missing=False,
    )
    with pytest.raises(probe.ProbeError, match="model_sha256_mismatch"):
        probe._preflight(args)
    parsed = probe._parser().parse_args([
        "--model", "m", "--validation-manifest", "v",
        "--selection-request", "s", "--annotations", "a",
        "--output-dir", "o", "--preflight-only",
    ])
    assert parsed.preflight_only is True


def test_probe_archive_is_deterministic_and_cpu_only(tmp_path: Path) -> None:
    root = tmp_path / "evidence"
    root.mkdir()
    (root / "probe_summary.json").write_text("{}\n", encoding="utf-8")
    (root / "SHA256SUMS.txt").write_text("x\n", encoding="utf-8")
    first = probe._deterministic_archive(root, tmp_path / "first.tar.gz")
    second = probe._deterministic_archive(root, tmp_path / "second.tar.gz")
    assert first.read_bytes() == second.read_bytes()
    manifest = probe._write_archive_manifest(
        root, first, status="completed", acceptance="accepted",
    )
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert manifest_payload["archive_sha256"] == probe._sha256_file(first)
    assert manifest_payload["archive_member_count"] == 2
    assert manifest_payload["probe_status"] == "completed"
    source = Path(probe.__file__).read_text(encoding="utf-8")
    assert 'providers=["CPUExecutionProvider"]' in source
    assert "paramiko" not in source.lower()
    assert "ssh " not in source.lower()


def test_probe_upstream_health_gate_is_absolute_and_pre_registered() -> None:
    assert _contract()["nms_identity"]["max_detections"] == 300
    assert probe.OFFICIAL_COCO_MAX_DETS == (1, 10, 100)
    assert all(probe._upstream_health_gates({
        "AP_50_95": 0.25, "AP_50": 0.40, "AP_75": 0.25,
    }).values())
    cases = [
        (
            {"AP_50_95": 0.249, "AP_50": 0.40, "AP_75": 0.25},
            "upstream_sanity_ap_50_95_at_least_025",
        ),
        (
            {"AP_50_95": 0.25, "AP_50": 0.399, "AP_75": 0.25},
            "upstream_sanity_ap50_at_least_040",
        ),
        (
            {"AP_50_95": 0.25, "AP_50": 0.40, "AP_75": 0.249},
            "upstream_sanity_ap75_at_least_025",
        ),
        (
            {"AP_50_95": 0.30, "AP_50": 0.60, "AP_75": 0.299},
            "upstream_sanity_ap75_ap50_ratio_at_least_050",
        ),
    ]
    for metrics, key in cases:
        assert probe._upstream_health_gates(metrics)[key] is False


def test_probe_production_health_gate_is_absolute_and_pre_registered() -> None:
    floors = {"AP_50_95": 0.20, "AP_50": 0.35, "AP_75": 0.20}
    assert all(probe._production_health_gates(floors).values())
    cases = [
        (
            {"AP_50_95": 0.199, "AP_50": 0.35, "AP_75": 0.20},
            "standard_production_ap_50_95_at_least_020",
        ),
        (
            {"AP_50_95": 0.20, "AP_50": 0.349, "AP_75": 0.20},
            "standard_production_ap50_at_least_035",
        ),
        (
            {"AP_50_95": 0.20, "AP_50": 0.35, "AP_75": 0.199},
            "standard_production_ap75_at_least_020",
        ),
        (
            {"AP_50_95": 0.25, "AP_50": 0.50, "AP_75": 0.224},
            "standard_production_ap75_ap50_ratio_at_least_045",
        ),
    ]
    for metrics, key in cases:
        assert probe._production_health_gates(metrics)[key] is False


def test_probe_requires_authoritative_pycocotools_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        probe.importlib_metadata, "version", lambda _name: "2.0.6",
    )
    with pytest.raises(probe.ProbeError, match="too_old"):
        probe._verified_pycocotools_version()
    monkeypatch.setattr(
        probe.importlib_metadata, "version", lambda _name: "2.0.7",
    )
    assert probe._verified_pycocotools_version() == "2.0.7"


def test_probe_records_pillow_preprocess_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        probe.importlib_metadata, "version", lambda name: "11.3.0"
        if name == "Pillow" else "2.0.7",
    )
    evidence = probe._pillow_runtime_evidence()
    assert evidence["pillow_version"] == "11.3.0"
    assert evidence["pillow_module"].endswith("PIL/__init__.py")


def test_probe_exact_use_snapshots_are_immutable_after_source_mutation(
    tmp_path: Path,
) -> None:
    from PIL import Image

    model = tmp_path / "model.onnx"
    model.write_bytes(b"exact-model-bytes")
    expected = hashlib.sha256(model.read_bytes()).hexdigest()
    model_snapshot = probe._read_exact_bytes(
        model, expected, label="model",
    )
    model.write_bytes(b"mutated-after-preflight")
    assert hashlib.sha256(model_snapshot).hexdigest() == expected
    assert model_snapshot != model.read_bytes()

    image_buffer = probe.io.BytesIO()
    Image.new("RGB", (11, 7), (10, 20, 30)).save(
        image_buffer, format="PNG",
    )
    image = tmp_path / "image.png"
    image.write_bytes(image_buffer.getvalue())
    image_sha = hashlib.sha256(image.read_bytes()).hexdigest()
    image_snapshot = probe._read_exact_bytes(
        image, image_sha, label="selected_image:image.png",
    )
    image.write_bytes(b"not-an-image-anymore")
    tensor, original_wh = probe._preprocess(image_snapshot)
    assert original_wh == (11, 7)
    assert tensor.shape == (1, 3, 640, 640)

    source = Path(probe.__file__).read_text(encoding="utf-8")
    assert 'ort.InferenceSession(\n        preflight["model_bytes"]' in source
    assert "annotations_exact_use_snapshot_mismatch" in source


def test_probe_records_exact_implementation_files_and_manifest_evidence() -> None:
    provenance = probe._implementation_provenance()
    assert provenance["tool_version"]
    assert provenance["tool_build_id"]
    assert set(provenance["files"]) == {
        "probe", "generic_decoder", "native_decoder", "official_coco",
    }
    for record in provenance["files"].values():
        path = probe.ROOT / record["relative_path"]
        assert record["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert record["size_bytes"] == path.stat().st_size
    assert "sha256" in provenance["source_manifest"]


def test_probe_full_run_requires_verified_source_manifest() -> None:
    valid = {
        "source_manifest": {
            "available": True,
            "package_version": probe.tool_identity.__version__,
            "workflow_version": probe.tool_identity.__build_id__,
            "all_implementation_files_listed_and_matching": True,
        },
    }
    probe._require_verified_implementation_provenance(valid)
    for key, value in (
        ("available", False),
        ("package_version", "2.75.46"),
        ("workflow_version", "wrong-build"),
        ("all_implementation_files_listed_and_matching", False),
    ):
        tampered = copy.deepcopy(valid)
        tampered["source_manifest"][key] = value
        with pytest.raises(
            probe.ProbeError,
            match="implementation_source_manifest_not_verified",
        ):
            probe._require_verified_implementation_provenance(tampered)


def test_official_coco_annotation_snapshot_provenance_is_opt_in_compatible() -> None:
    parameter = inspect.signature(
        official_coco.evaluate_coco_bbox
    ).parameters["annotations_provenance"]
    assert parameter.default is None
    source = inspect.getsource(official_coco.evaluate_coco_bbox)
    assert '"annotations_exact_use": provenance or None' in source


def test_official_coco_rejects_annotation_provenance_hash_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    annotations = tmp_path / "annotations.json"
    annotations.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        official_coco, "pycocotools_status",
        lambda: {"available": False, "error": "not installed in unit test"},
    )
    with pytest.raises(
        ValueError, match="annotations_provenance_sha256_mismatch",
    ):
        official_coco.evaluate_coco_bbox(
            annotations=annotations,
            predictions=[],
            output_dir=tmp_path / "bad",
            annotations_provenance={
                "source_sha256": "0" * 64,
                "exact_use_snapshot_sha256": "0" * 64,
            },
        )
    payload = official_coco.evaluate_coco_bbox(
        annotations=annotations,
        predictions=[],
        output_dir=tmp_path / "default",
    )
    assert payload["status"] == "unavailable"


def test_probe_parity_jsonl_is_canonical_and_archived_without_owner_marker(
    tmp_path: Path,
) -> None:
    output = tmp_path / "out"
    output.mkdir()
    marker = output / probe._OWNERSHIP_MARKER
    marker.write_text("owned\n", encoding="utf-8")
    rows = [{
        "sequence_index": 1,
        "image": "0001.jpg",
        "image_sha256": "a" * 64,
        "coco_image_id": 1,
        "generic_sha256": "b" * 64,
        "native_sha256": "b" * 64,
        "equal": True,
    }]
    probe._write_jsonl(output / "generic_native_parity.jsonl", rows)
    assert (output / "generic_native_parity.jsonl").read_bytes() == (
        probe._canonical_bytes(rows[0]) + b"\n"
    )
    probe._sha256sums(output)
    assert probe._OWNERSHIP_MARKER not in (
        output / "SHA256SUMS.txt"
    ).read_text(encoding="utf-8")
    archive = probe._deterministic_archive(output, tmp_path / "out.tar.gz")
    with probe.tarfile.open(archive, "r:gz") as handle:
        names = handle.getnames()
    assert not any(name.endswith(probe._OWNERSHIP_MARKER) for name in names)
    assert any(name.endswith("generic_native_parity.jsonl") for name in names)


def test_probe_full_path_uses_snapshots_and_emits_auditable_parity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from PIL import Image

    image_buffer = probe.io.BytesIO()
    Image.new("RGB", (80, 60), (32, 64, 96)).save(
        image_buffer, format="PNG",
    )
    image_bytes = image_buffer.getvalue()
    image_sha256 = hashlib.sha256(image_bytes).hexdigest()
    annotations_bytes = b'{"images":[],"annotations":[],"categories":[]}\n'
    annotations_sha256 = hashlib.sha256(annotations_bytes).hexdigest()
    model_bytes = b"preflight-verified-model-byte-snapshot"

    model_path = tmp_path / "model.onnx"
    image_path = tmp_path / "0001.png"
    annotations_path = tmp_path / "annotations.json"
    model_path.write_bytes(b"mutated-model-path")
    image_path.write_bytes(b"mutated-image-path")
    annotations_path.write_bytes(b"mutated-annotations-path")

    heads = [
        np.full((1, 3, 80, 80, 85), -20.0, dtype=np.float32),
        np.full((1, 3, 40, 40, 85), -20.0, dtype=np.float32),
        np.full((1, 3, 20, 20, 85), -20.0, dtype=np.float32),
    ]
    heads[0][0, 0, 30, 30, :4] = 0.0
    heads[0][0, 0, 30, 30, 4] = 10.0
    heads[0][0, 0, 30, 30, 5] = 10.0

    class _Meta:
        def __init__(self, name: str, shape: list[int] | None = None) -> None:
            self.name = name
            self.shape = shape

    class _Session:
        def __init__(self, source: bytes, *, providers: list[str]) -> None:
            assert source == model_bytes
            assert providers == ["CPUExecutionProvider"]

        def get_inputs(self) -> list[_Meta]:
            return [_Meta("images", [1, 3, 640, 640])]

        def get_outputs(self) -> list[_Meta]:
            return [_Meta("p3"), _Meta("p4"), _Meta("p5")]

        def run(self, _names: object, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
            assert feeds["images"].shape == (1, 3, 640, 640)
            return [np.array(value, copy=True) for value in heads]

    monkeypatch.setitem(
        sys.modules, "onnxruntime",
        types.SimpleNamespace(InferenceSession=_Session),
    )
    monkeypatch.setattr(probe, "EXPECTED_SELECTED_IMAGE_COUNT", 1)
    monkeypatch.setattr(
        probe, "EXPECTED_ANNOTATIONS_SHA256", annotations_sha256,
    )
    implementation = {
        "tool_version": probe.tool_identity.__version__,
        "tool_build_id": probe.tool_identity.__build_id__,
        "source_manifest": {
            "available": True,
            "package_version": probe.tool_identity.__version__,
            "workflow_version": probe.tool_identity.__build_id__,
            "all_implementation_files_listed_and_matching": True,
        },
    }
    preflight = {
        "dependencies": {},
        "implementation_provenance": implementation,
        "model": model_path,
        "model_bytes": model_bytes,
        "manifest_path": tmp_path / "manifest.json",
        "manifest_bytes": b"{}\n",
        "request_path": tmp_path / "request.json",
        "request_bytes": b"{}\n",
        "annotations_path": annotations_path,
        "annotations_bytes": annotations_bytes,
        "selected": [{
            "absolute_path": str(image_path),
            "image_bytes": image_bytes,
            "verified_image_sha256": image_sha256,
        }],
        "selected_ids": [image_path.name],
        "images_root": tmp_path,
        "category_mapping": {
            "selected_file_to_image_id": {image_path.name: 1},
            "class_index_to_category_id": probe.COCO80_CATEGORY_IDS,
            "mapping_sha256": probe.COCO80_CATEGORY_IDS_SHA256,
        },
    }
    monkeypatch.setattr(probe, "_preflight", lambda _args: preflight)

    official_calls = []
    metrics = {
        "legacy_tiny_production": {
            "AP_50_95": 0.24, "AP_50": 0.45, "AP_75": 0.20,
        },
        "standard_production": {
            "AP_50_95": 0.30, "AP_50": 0.50, "AP_75": 0.30,
        },
        "standard_upstream_sanity": {
            "AP_50_95": 0.30, "AP_50": 0.50, "AP_75": 0.30,
        },
    }

    def _official(**kwargs: object) -> dict[str, object]:
        snapshot = Path(str(kwargs["annotations"]))
        assert snapshot.read_bytes() == annotations_bytes
        provenance = kwargs["annotations_provenance"]
        assert isinstance(provenance, dict)
        assert provenance["exact_use_snapshot_sha256"] == annotations_sha256
        name = str(kwargs["variant"])
        official_calls.append(name)
        return {
            "status": "ok",
            "metrics": metrics[name],
            "annotations_sha256": annotations_sha256,
            "annotations_exact_use": provenance,
            "evaluation_payload_sha256": hashlib.sha256(
                name.encode("utf-8")
            ).hexdigest(),
        }

    monkeypatch.setattr(probe, "evaluate_coco_bbox", _official)
    output = tmp_path / "probe-output"
    args = argparse.Namespace(
        output_dir=str(output), archive_path="", overlay_count=0,
        preflight_only=False, install_missing=False,
        _ownership_token="unit-test-owner",
    )
    result = probe.run(args)
    assert result["status"] == "completed"
    assert result["acceptance"]["status"] == "accepted"
    assert official_calls == list(metrics)
    parity = result["generic_native_parity"]
    assert parity["row_count"] == 1
    assert parity["all_equal"] is True
    assert parity["chains_equal"] is True
    lines = (output / "generic_native_parity.jsonl").read_text(
        encoding="utf-8",
    ).splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["equal"] is True
    assert not (output / probe._OWNERSHIP_MARKER).exists()
    assert Path(str(output) + ".tar.gz").is_file()
    assert Path(str(output) + ".tar.gz.manifest.json").is_file()


def test_probe_does_not_mutate_preexisting_nonempty_or_unsafe_archive_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_run = probe.run
    monkeypatch.setattr(
        probe, "_dependency_preflight", lambda **_kwargs: {},
    )
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "keep.bin"
    sentinel.write_bytes(b"do-not-touch")
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    rc = probe.main([
        "--model", str(tmp_path / "missing.onnx"),
        "--validation-manifest", str(tmp_path / "missing-manifest.json"),
        "--selection-request", str(tmp_path / "missing-request.json"),
        "--annotations", str(tmp_path / "missing-annotations.json"),
        "--output-dir", str(output), "--preflight-only",
    ])
    assert rc == 2
    assert {path.name: path.read_bytes() for path in output.iterdir()} == before
    assert not Path(str(output) + ".tar.gz").exists()

    unsafe_output = tmp_path / "unsafe"
    unsafe_archive = unsafe_output / "inside.tar.gz"
    rc = probe.main([
        "--model", str(tmp_path / "missing.onnx"),
        "--validation-manifest", str(tmp_path / "missing-manifest.json"),
        "--selection-request", str(tmp_path / "missing-request.json"),
        "--annotations", str(tmp_path / "missing-annotations.json"),
        "--output-dir", str(unsafe_output),
        "--archive-path", str(unsafe_archive), "--preflight-only",
    ])
    assert rc == 2
    assert not unsafe_output.exists()

    preflight_output = tmp_path / "preflight-failure"
    rc = probe.main([
        "--model", str(tmp_path / "missing.onnx"),
        "--validation-manifest", str(tmp_path / "missing-manifest.json"),
        "--selection-request", str(tmp_path / "missing-request.json"),
        "--annotations", str(tmp_path / "missing-annotations.json"),
        "--output-dir", str(preflight_output), "--preflight-only",
    ])
    assert rc == 2
    assert not preflight_output.exists()
    assert not Path(str(preflight_output) + ".tar.gz").exists()

    owned_output = tmp_path / "owned-failure"
    def _owned_failure(args: argparse.Namespace) -> dict:
        output_dir = Path(args.output_dir)
        probe._claim_output_dir(
            output_dir, str(args._ownership_token),
        )
        probe._write_json(output_dir / "probe_summary.json", {
            "schema": probe.PROBE_SCHEMA,
            "schema_version": probe.PROBE_SCHEMA_VERSION,
            "status": "failed",
            "acceptance": {"status": "rejected", "passed": False},
        })
        raise probe.ProbeError("pre_registered_yolov7_decoder_ab_gate_failed")

    monkeypatch.setattr(probe, "run", _owned_failure)
    rc = probe.main([
        "--model", str(tmp_path / "missing.onnx"),
        "--validation-manifest", str(tmp_path / "missing-manifest.json"),
        "--selection-request", str(tmp_path / "missing-request.json"),
        "--annotations", str(tmp_path / "missing-annotations.json"),
        "--output-dir", str(owned_output), "--preflight-only",
    ])
    assert rc == 2
    failure_archive = Path(str(owned_output) + ".tar.gz")
    failure_manifest = Path(str(failure_archive) + ".manifest.json")
    assert failure_archive.is_file()
    assert failure_manifest.is_file()
    failure_payload = json.loads(failure_manifest.read_text(encoding="utf-8"))
    assert failure_payload["probe_status"] == "failed"
    assert failure_payload["acceptance"] == "rejected"
    assert failure_payload["archive_sha256"] == probe._sha256_file(
        failure_archive
    )

    monkeypatch.setattr(probe, "run", original_run)
    stale_output = tmp_path / "stale-sidecar-output"
    stale_archive = Path(str(stale_output) + ".tar.gz")
    stale_sidecar = Path(str(stale_archive) + ".manifest.json")
    stale_sidecar.write_bytes(b"foreign-sidecar")
    rc = probe.main([
        "--model", str(tmp_path / "missing.onnx"),
        "--validation-manifest", str(tmp_path / "missing-manifest.json"),
        "--selection-request", str(tmp_path / "missing-request.json"),
        "--annotations", str(tmp_path / "missing-annotations.json"),
        "--output-dir", str(stale_output), "--preflight-only",
    ])
    assert rc == 2
    assert not stale_archive.exists()
    assert stale_sidecar.read_bytes() == b"foreign-sidecar"

    raced_output = tmp_path / "raced-output"
    raced_file = raced_output / "foreign.bin"
    def _foreign_write_before_claim(_args: argparse.Namespace) -> dict:
        raced_output.mkdir()
        raced_file.write_bytes(b"foreign")
        raise probe.ProbeError("output_dir_not_empty")

    monkeypatch.setattr(probe, "run", _foreign_write_before_claim)
    rc = probe.main([
        "--model", "m", "--validation-manifest", "v",
        "--selection-request", "s", "--annotations", "a",
        "--output-dir", str(raced_output),
    ])
    assert rc == 2
    assert raced_file.read_bytes() == b"foreign"
    assert list(raced_output.iterdir()) == [raced_file]
    assert not Path(str(raced_output) + ".tar.gz").exists()
