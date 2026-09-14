from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType
from typing import Any

import pytest
import yaml

from onnx_splitpoint_tool.benchmark.evaluation_profiles import (
    profile_model_entries,
    validate_evaluation_profile_payload,
)
from onnx_splitpoint_tool.campaign import (
    create_campaign_skeleton,
    create_pipeline_contract_manifest,
    verify_pipeline_contract_manifest,
)
from onnx_splitpoint_tool.preprocessing_contract import (
    canonical_image_preprocessing_contract,
    preprocessing_contract_sha256,
)


ROOT = Path(__file__).resolve().parents[1]
PROFILE_ROOT = ROOT / "onnx_splitpoint_tool/resources/evaluation_profiles"
TEMPLATE_ROOT = ROOT / "onnx_splitpoint_tool/resources/campaign_templates"
MATRIX_PROFILE = PROFILE_ROOT / "thesis_final_evaluated_matrix_v1.yaml"
GENERALIZATION_PROFILE = PROFILE_ROOT / "thesis_final_campaign_v1.yaml"
GENERATOR_PATH = ROOT / "scripts/create_v27528_yolov7_final_canary_profile.py"
RUNNER_PATH = ROOT / "scripts/run_v27528_yolov7_final_canary.sh"


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    assert isinstance(payload, dict)
    return payload


def _load_generator() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "v27528_yolov7_paper_final_canary_generator", GENERATOR_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("path", [MATRIX_PROFILE, GENERALIZATION_PROFILE])
def test_final_profiles_use_only_exact_yolov7_paper_identity(path: Path) -> None:
    profile = validate_evaluation_profile_payload(_load_yaml(path), source=str(path))
    active = profile_model_entries(profile, include_reserve=True)
    yolo7_rows = [
        row
        for row in active
        if str(row.get("family_id") or "").lower() == "yolo7"
        or "yolov7" in str(row.get("id") or "").lower()
    ]
    assert len(yolo7_rows) == 1
    assert yolo7_rows[0]["id"] == "yolov7_paper"
    assert yolo7_rows[0]["path"] == "TODO_REPLACE_WITH_EXACT_YOLOV7_PAPER_ONNX"


def test_evaluated_matrix_has_the_exact_three_model_ids() -> None:
    profile = _load_yaml(MATRIX_PROFILE)
    ids = [row["id"] for row in profile["model_suite"]["primary"]]
    assert ids == ["resnet50", "yolo26s", "yolov7_paper"]


@pytest.mark.parametrize(
    "filename,task,target_hw,expected_sha",
    [
        (
            "classification_preprocessing.yaml",
            "classification",
            [224, 224],
            "ea28cf5ac35bd4c9a3321ac97fd559f32fd7a661dc4a93c324fe3ffc54188fa9",
        ),
        (
            "detection_preprocessing.yaml",
            "detection",
            [640, 640],
            "411898b92628e2a769a2b8ea7aa35faff35373cfacb863c3ab34b243c105b70e",
        ),
    ],
)
def test_stock_preprocessing_is_canonical_and_runtime_numeric_is_attested(
    filename: str, task: str, target_hw: list[int], expected_sha: str
) -> None:
    payload = _load_yaml(TEMPLATE_ROOT / "configs" / filename)
    identity = canonical_image_preprocessing_contract(task, target_hw)
    assert payload["locked"] is True
    assert payload["identity"] == identity
    assert payload["identity_sha256"] == expected_sha
    assert preprocessing_contract_sha256(identity) == expected_sha
    assert payload["runtime_numeric_input"] == {
        "policy": "backend_attested_per_execution",
        "schema": "onnx-splitpoint/runtime-numeric-input-identity",
        "schema_version": 1,
        "semantic_preprocessing_sha256": expected_sha,
    }
    for forbidden in (
        "input_layout",
        "input_dtype",
        "resize_short_side",
        "center_crop",
        "stride_alignment",
    ):
        assert forbidden not in payload


def test_stock_detection_decoder_covers_exact_matrix_endpoints() -> None:
    payload = _load_yaml(TEMPLATE_ROOT / "configs/detection_decoder.yaml")
    assert payload["locked"] is True
    assert set(payload["models"]) == {"yolo26s", "yolov7_paper"}

    yolo26 = payload["models"]["yolo26s"]
    assert yolo26["source_endpoint_stage"] == "decoded_nms"
    assert yolo26["source_output_format"] == "bn6_detections"
    assert yolo26["source_output_names"] == ["output0"]
    assert yolo26["source_endpoint_is_raw"] is False
    assert yolo26["source_endpoint_has_integrated_nms"] is True
    assert yolo26["source_tensor_contract"] == [
        {
            "name": "output0",
            "shape": [1, 300, 6],
            "record_format": "xyxy_score_class",
        }
    ]

    yolov7 = payload["models"]["yolov7_paper"]
    assert yolov7["source_endpoint_stage"] == "raw_head"
    assert yolov7["source_output_format"] == "multiscale_head"
    assert yolov7["source_output_names"] == ["output", "clone_1", "clone_2"]
    assert yolov7["source_endpoint_is_raw"] is True
    assert yolov7["source_endpoint_has_integrated_nms"] is False
    assert yolov7["source_tensor_contract"] == [
        {"name": "output", "shape": [1, 3, 80, 80, 85]},
        {"name": "clone_1", "shape": [1, 3, 40, 40, 85]},
        {"name": "clone_2", "shape": [1, 3, 20, 20, 85]},
    ]


def test_stock_detection_nms_is_exact_and_class_aware() -> None:
    payload = _load_yaml(TEMPLATE_ROOT / "configs/detection_nms.yaml")
    assert payload["locked"] is True
    assert payload["score_threshold"] == 0.25
    assert payload["confidence_threshold"] == 0.25
    assert payload["iou_threshold"] == 0.45
    assert payload["max_detections"] == 300
    assert payload["class_aware"] is True
    assert payload["class_agnostic"] is False
    assert payload["multi_label"] is False


def test_stock_init_builds_locked_placeholder_free_matrix_manifest(
    tmp_path: Path,
) -> None:
    skeleton = create_campaign_skeleton(tmp_path / "campaign_inputs")
    manifest_path = create_pipeline_contract_manifest(
        spec=skeleton["contract_spec"],
        output=tmp_path / "pipeline_contract_manifest.json",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    verification = verify_pipeline_contract_manifest(manifest, require_locked=True)

    assert [row["id"] for row in manifest["contracts"]] == [
        "classification_preprocessing",
        "detection_preprocessing",
        "detection_decoder",
        "detection_nms",
    ]
    assert verification["ok"] is True
    assert verification["locked_ok"] is True
    assert verification["placeholder_free_ok"] is True
    assert verification["adapter_contract_count"] == 0


def test_v27528_canary_accepts_only_one_exact_yolov7_paper_id() -> None:
    generator = _load_generator()
    source = _load_yaml(MATRIX_PROFILE)
    canary = generator.build_canary(
        source, profile_id="exact_yolov7_paper_canary_test"
    )
    active = profile_model_entries(canary, include_reserve=True)
    assert [row["id"] for row in active] == ["yolov7_paper"]

    for non_exact in ("yolov7", "YOLOV7_PAPER", "yolov7_paper "):
        invalid = copy.deepcopy(source)
        invalid["model_suite"]["primary"][-1]["id"] = non_exact
        with pytest.raises(
            ValueError,
            match="expected exactly one active model with id 'yolov7_paper'",
        ):
            generator.build_canary(invalid, profile_id="must_fail_closed")


def test_v27528_canary_cli_writes_exact_profile_and_blocking_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    generator = _load_generator()
    source = tmp_path / "source.yaml"
    output = tmp_path / "canary.yaml"
    preflight = tmp_path / "preflight"
    source.write_text(
        yaml.safe_dump(_load_yaml(MATRIX_PROFILE), sort_keys=False),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(GENERATOR_PATH),
            "--source-profile",
            str(source),
            "--out",
            str(output),
            "--preflight-dir",
            str(preflight),
        ],
    )

    assert generator.main() == 2
    generated = _load_yaml(output)
    assert generated["name"] == generator.DEFAULT_PROFILE_ID
    assert [
        row["id"]
        for row in profile_model_entries(generated, include_reserve=True)
    ] == ["yolov7_paper"]
    readiness = json.loads(
        (preflight / "campaign_readiness.json").read_text(encoding="utf-8")
    )
    assert readiness["final_ready"] is False
    assert readiness["required_failure_count"] > 0


def test_v27528_runner_has_stable_exact_identity_and_audited_delegate() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    assert (
        "yolov7_paper_final_contract_canary_v27528" in source
    )
    assert 'exec bash "$TOOL/scripts/run_v27527_yolov7_final_canary.sh" "$@"' in source

    generator = _load_generator()
    assert generator.DEFAULT_PROFILE_ID == "yolov7_paper_final_contract_canary_v27528"
