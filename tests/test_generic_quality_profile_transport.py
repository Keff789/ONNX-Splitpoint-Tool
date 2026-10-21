"""The normal profile schema and both generated-plan handoffs retain opt-in."""
from pathlib import Path

import pytest
import yaml

from onnx_splitpoint_tool.benchmark.evaluation_profiles import validate_evaluation_profile_payload
from onnx_splitpoint_tool.workflow.benchmark_binding import _benchmark_runs_from_profile
from onnx_splitpoint_tool.workflow.legacy_benchmarkset_binding import _apply_declared_run_variants

ROOT = Path(__file__).resolve().parents[1]


def profile():
    payload = yaml.safe_load((ROOT / "onnx_splitpoint_tool/resources/evaluation_profiles/native_resnet_yolo26s_hailo8_smoke_v1.yaml").read_text())
    payload["run_profiles"] = [{
        "id": "hailo10_to_tensorrt", "type": "mixed_backend", "stage1": "hailo10", "stage2": "tensorrt",
        "hardware_setup_id": "orin_nx_hailo10_01", "variants": ["composed"], "case_ids": ["b039"],
        "quality_source_only": True, "image_scale": "norm", "native_trt_build": False,
        "validation_images": "resources/validation/detection/val2017_n5000_s20260710",
        "validation_max_images": 5000, "validation_reference_mode": "auto", "mini_coco_ap50": True,
        "trt_runtime": "native", "native_trt_precision": "fp16",
        "generic_quality_completion_configs": {"b039": "quality_runtime/b039.json"},
        "generic_quality_completion_config_sha256": {"b039": "a" * 64},
        "generic_quality_completion_sources": {"b039": {"path": "/controlled/source.json", "sha256": "a" * 64, "size_bytes": 12}},
    }]
    payload.setdefault("quality_gate", {}).setdefault("management_reference", {})["existing_reference"] = {
        "reference_role": "canonical_cpu_ort", "semantic_reference_only": True,
        "source_evaluation_run_id": "original", "reference_path": "/controlled/old_reference.json",
        "reference_sha256": "b" * 64, "reference_size_bytes": 123,
        "quality_contract": {"frozen": True}, "quality_contract_sha256": "c" * 64,
        "source_status": {"path": "/controlled/status.json", "sha256": "d" * 64, "size_bytes": 124},
        "policy_source": {"path": "/controlled/policy.json", "sha256": "e" * 64, "size_bytes": 125},
    }
    return payload


def test_real_schema_then_formal_and_legacy_plan_handoffs_preserve_quality_scope():
    payload = validate_evaluation_profile_payload(profile(), source="controlled test")
    source = payload["run_profiles"][0]
    formal = _benchmark_runs_from_profile(payload, ["hailo10", "tensorrt"])
    generated = [{"id": source["id"], "type": "matrix", "stage1": "hailo10", "stage2": "tensorrt",
                  "variants": ["full", "part1", "part2", "composed"]}]
    legacy = _apply_declared_run_variants(generated, payload)
    for rows in (formal, legacy):
        row = next(r for r in rows if r["id"] == source["id"])
        assert row["type"] == "matrix"
        for field in ("case_ids", "quality_source_only", "generic_quality_completion_configs",
                      "generic_quality_completion_config_sha256", "generic_quality_completion_sources",
                      "validation_images", "validation_max_images", "native_trt_build"):
            assert row[field] == source[field]


@pytest.mark.parametrize("damage", ["null_reference", "bad_hash", "source_escape", "not_source_only",
                                   "empty_cases", "wrong_stage", "extra_property"])
def test_normal_schema_rejects_malformed_explicit_quality_profile(damage):
    payload = profile()
    row = payload["run_profiles"][0]
    if damage == "null_reference":
        payload["quality_gate"]["management_reference"]["existing_reference"] = None
    elif damage == "bad_hash":
        row["generic_quality_completion_config_sha256"]["b039"] = "not a hash"
    elif damage == "source_escape":
        row["generic_quality_completion_configs"]["b039"] = "../elsewhere.json"
    elif damage == "not_source_only":
        row["quality_source_only"] = False
    elif damage == "empty_cases":
        row["case_ids"] = []
    elif damage == "wrong_stage":
        row["stage1"] = "onnxruntime_cpu"
    else:
        payload["quality_gate"]["management_reference"]["existing_reference"]["made_up"] = True
    with pytest.raises(ValueError, match="Invalid evaluation profile"):
        validate_evaluation_profile_payload(payload, source="controlled negative test")


def test_default_profile_rows_are_unchanged_without_generic_opt_in():
    payload = profile()
    plain = {k: v for k, v in payload["run_profiles"][0].items()
             if k in {"id", "type", "stage1", "stage2", "variants"}}
    payload["run_profiles"] = [plain]
    source = [{"id": plain["id"], "type": "matrix", "variants": ["full", "part1", "part2", "composed"]}]
    assert _apply_declared_run_variants(source, payload) == [{**source[0], "variants": ["composed"]}]
    assert next(r for r in _benchmark_runs_from_profile(payload, []) if r["id"] == plain["id"])["type"] == "mixed_backend"
