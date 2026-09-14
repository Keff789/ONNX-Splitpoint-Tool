from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest

from onnx_splitpoint_tool.native_output_endpoint import (
    load_authoritative_output_contract,
    runtime_output_contract,
)
from onnx_splitpoint_tool.workflow.results import normalize_benchmark_files


ROOT = Path(__file__).resolve().parents[1]
SUITE_TEMPLATE = ROOT / "onnx_splitpoint_tool/resources/templates/benchmark_suite.py.txt"


def _full_companion_rows(*, conflicting_endpoint: bool = False) -> list[dict]:
    cases = ("b044", "b116", "b216")
    rows: list[dict] = []
    for index, case_id in enumerate(cases):
        endpoint = ("2" if conflicting_endpoint and index == 2 else "1") * 64
        rows.append({
            "case_id": case_id,
            "run_id": "ort_tensorrt",
            "primary_variant": "composed",
            "stage1_provider": "tensorrt",
            "stage2_provider": "tensorrt",
            "full_provider": "tensorrt",
            "composed_mean_ms": 7.0 + index,
            "full_mean_ms": 4.0 + index / 10.0,
            "measured_variants": ["composed", "full"],
            "variant_status": {"composed": "ok", "full": "ok"},
            "runtime_ok": True,
            "validation_ok": True,
            "final_pass": True,
            "full_preprocessing_contract_sha256": "a" * 64,
            "full_endpoint_contract_hash": endpoint,
            "full_model_sha256": "b" * 64,
            "full_runtime_artifact_sha256": "c" * 64,
            "full_runtime_precision_identity": "fp16",
            "full_endpoint_contract_complete": True,
        })
    return rows


def _normalize(tmp_path: Path, rows: list[dict]) -> list[dict]:
    source = tmp_path / "benchmark_results_matrix.json"
    source.write_text(json.dumps(rows), encoding="utf-8")
    normalized, _sources = normalize_benchmark_files(
        model_id="yolov7_paper", source_paths=[source],
    )
    return normalized


def test_three_case_full_companions_become_one_global_full_row(
    tmp_path: Path,
) -> None:
    rows = _normalize(tmp_path, _full_companion_rows())
    full = [row for row in rows if row["variant"] == "full"]
    split = [row for row in rows if row["variant"] == "split"]

    assert len(full) == 1
    assert len(split) == 3
    result = full[0]
    assert result["case_id"] == "full"
    assert result["full_source_case_ids"] == ["b044", "b116", "b216"]
    assert result["dedupe_group_size"] == 3
    assert result["stage1_provider"] == result["stage2_provider"] == ""
    assert result["full_baseline_frozen_identity_status"] == "consistent"
    assert result["full_baseline_frozen_identity_complete"] is True
    assert result["preprocessing_contract_sha256"] == "a" * 64
    assert result["endpoint_contract_hash"] == "1" * 64
    assert result["runtime_artifact_sha256"] == "c" * 64
    assert result["runtime_precision_identity"] == "fp16"


def test_genuine_full_rows_ignore_stale_split_stage_labels(
    tmp_path: Path,
) -> None:
    source_rows = _full_companion_rows()
    for row in source_rows:
        row.update({
            "primary_variant": "full",
            "variant": "full",
            "backend": "tensorrt",
            "run_id": "tensorrt",
        })

    rows = _normalize(tmp_path, source_rows)

    assert len(rows) == 1
    result = rows[0]
    assert result["case_id"] == "full"
    assert result["stage1_provider"] == result["stage2_provider"] == ""
    assert result["full_source_case_ids"] == ["b044", "b116", "b216"]
    assert result["dedupe_group_size"] == 3


def test_real_overnight_trt_full_uses_dedicated_owner_and_aggregates_cases(
    tmp_path: Path,
) -> None:
    # Exact Full means observed for yolo26s in the 2.69c overnight artefacts.
    sources = (
        ("hailo10_to_trt", "hailo10", (7.942485809326172, 7.931375503540039, 8.236551284790039)),
        ("hailo8_to_trt", "hailo8", (7.865095138549805, 8.15587043762207, 8.153009414672852)),
        ("ort_tensorrt", "tensorrt", (8.247566223144531, 7.994794845581055, 8.00776481628418)),
    )
    cases = ("b038", "b142", "b036")
    rows: list[dict] = []
    for run_id, stage1, latencies in sources:
        for case_id, latency in zip(cases, latencies):
            row = _full_companion_rows()[0]
            row.update({
                "case_id": case_id,
                "run_id": run_id,
                "stage1_provider": stage1,
                "stage2_provider": "tensorrt",
                "full_provider": "tensorrt",
                "full_mean_ms": latency,
            })
            rows.append(row)

    result = next(row for row in _normalize(tmp_path, rows) if row["variant"] == "full")

    owner_values = list(sources[-1][2])
    assert result["dedupe_group_size"] == 9
    assert result["dedupe_selected_group_size"] == 3
    assert result["run_id"] == "ort_tensorrt"
    assert result["full_baseline_source_status"] == "selected"
    assert result["full_baseline_source_role"] == "same_backend_reference"
    assert result["full_baseline_source_run_ids"] == ["ort_tensorrt"]
    assert result["full_baseline_ignored_source_run_ids"] == [
        "hailo10_to_trt", "hailo8_to_trt",
    ]
    assert result["full_baseline_latency_sample_count"] == 3
    assert [sample["latency_ms"] for sample in result["full_baseline_latency_samples"]] == [
        owner_values[2], owner_values[0], owner_values[1],
    ]
    assert result["total_latency_ms"] == pytest.approx(sum(owner_values) / 3.0)


def test_two_distinct_full_owner_run_ids_fail_closed(tmp_path: Path) -> None:
    rows = _full_companion_rows()[:2]
    rows[0]["run_id"] = "ort_tensorrt"
    rows[1]["run_id"] = "alternate_tensorrt_reference"

    result = next(row for row in _normalize(tmp_path, rows) if row["variant"] == "full")

    assert result["full_baseline_source_status"] == "source_conflict"
    assert result["full_baseline_source_complete"] is False
    assert result["full_baseline_source_run_ids"] == [
        "alternate_tensorrt_reference", "ort_tensorrt",
    ]
    assert result["row_conflict"] == "full_baseline_source_identity_conflict"
    assert result["final_pass"] is False
    assert result["eligible_for_ranking"] is False


def test_divergent_duplicate_full_sample_representations_fail_closed(
    tmp_path: Path,
) -> None:
    rows = _full_companion_rows()[:2]
    for row in rows:
        row.update({
            "case_id": "b001",
            "run_id": "ort_tensorrt",
            "full_mean_ms": 8.0,
            "full_endpoint_contract_hash": "1" * 64,
        })
    rows[1]["full_mean_ms"] = 99.0
    rows[1]["full_endpoint_contract_hash"] = "2" * 64

    result = next(row for row in _normalize(tmp_path, rows) if row["variant"] == "full")

    assert result["dedupe_group_size"] == 2
    assert result["dedupe_selected_group_size"] == 2
    assert result["full_baseline_source_status"] == "source_conflict"
    assert result["full_baseline_source_complete"] is False
    assert result["full_baseline_duplicate_source_conflict_count"] == 1
    conflict = result["full_baseline_duplicate_source_conflicts"][0]
    assert conflict["source_run_id"] == "ort_tensorrt"
    assert conflict["source_case_id"] == "b001"
    assert conflict["row_count"] == 2
    assert "multiple_equal_priority_representations" in conflict["reasons"]
    assert "evidence_mismatch:endpoint_contract_hash" in conflict["reasons"]
    assert "evidence_mismatch:total_latency_ms" in conflict["reasons"]
    assert result["full_baseline_frozen_identity_status"] == "conflict"
    assert result["full_baseline_frozen_identity_conflicts"] == [
        "endpoint_contract_hash",
    ]
    assert result["row_conflict"] == "full_baseline_source_identity_conflict"
    assert result["final_pass"] is False
    assert result["eligible_for_ranking"] is False


def test_latency_only_divergent_duplicate_full_sample_fails_closed(
    tmp_path: Path,
) -> None:
    rows = _full_companion_rows()[:2]
    for row in rows:
        row.update({
            "case_id": "b001",
            "run_id": "ort_tensorrt",
            "full_mean_ms": 8.0,
            "full_endpoint_contract_hash": "1" * 64,
        })
    rows[1]["full_mean_ms"] = 99.0

    result = next(row for row in _normalize(tmp_path, rows) if row["variant"] == "full")

    assert result["full_baseline_frozen_identity_status"] == "consistent"
    assert result["full_baseline_source_status"] == "source_conflict"
    assert result["full_baseline_source_complete"] is False
    conflict = result["full_baseline_duplicate_source_conflicts"][0]
    assert "evidence_mismatch:total_latency_ms" in conflict["reasons"]
    assert "multiple_equal_priority_representations" in conflict["reasons"]
    assert not any(
        "endpoint_contract_hash" in reason for reason in conflict["reasons"]
    )
    assert result["total_latency_ms"] == pytest.approx(53.5)
    assert result["row_conflict"] == "full_baseline_source_identity_conflict"
    assert result["final_pass"] is False
    assert result["eligible_for_ranking"] is False


def _suite_owner_functions() -> dict[str, Any]:
    names = {
        "_resolve_stage_token", "_normalize_variants", "_canonical_full_backend",
        "_full_backend_for_run", "_assign_full_baseline_owners",
        "_apply_full_owner_case_variant",
    }
    tree = ast.parse(SUITE_TEMPLATE.read_text(encoding="utf-8"))
    nodes = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    namespace: dict[str, Any] = {
        "Any": Any, "Dict": Dict, "List": List, "Optional": Optional,
        "Tuple": Tuple,
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SUITE_TEMPLATE), "exec"), namespace)
    return namespace


def test_suite_measures_full_only_on_canonical_backend_owner() -> None:
    funcs = _suite_owner_functions()
    runs = [
        {
            "id": "ort_tensorrt", "type": "onnxruntime", "provider": "tensorrt",
            "stage1": {"type": "onnxruntime", "provider": "tensorrt"},
            "stage2": {"type": "onnxruntime", "provider": "tensorrt"},
        },
        {
            "id": "hailo8_to_trt", "type": "matrix", "provider": "tensorrt",
            "stage1": {"type": "hailo", "hw_arch": "hailo8"},
            "stage2": {"type": "onnxruntime", "provider": "tensorrt"},
            "variants": ["full", "part1", "part2", "composed"],
        },
        {
            "id": "hailo10_to_trt", "type": "matrix", "provider": "tensorrt",
            "stage1": {"type": "hailo", "hw_arch": "hailo10"},
            "stage2": {"type": "onnxruntime", "provider": "tensorrt"},
            "variants": ["full", "part1", "part2", "composed"],
        },
        {
            "id": "hailo8", "type": "hailo", "hw_arch": "hailo8",
            "variants": ["full"],
        },
    ]

    assigned = funcs["_assign_full_baseline_owners"](runs)
    by_id = {row["id"]: row for row in assigned}

    assert by_id["ort_tensorrt"]["_full_baseline_owner"] is True
    assert "full" in by_id["ort_tensorrt"]["variants"]
    assert by_id["hailo8_to_trt"]["_full_baseline_owner"] is False
    assert by_id["hailo10_to_trt"]["_full_baseline_owner"] is False
    assert "full" not in by_id["hailo8_to_trt"]["variants"]
    assert "full" not in by_id["hailo10_to_trt"]["variants"]
    assert by_id["hailo8"]["_full_baseline_owner"] is True

    first = funcs["_apply_full_owner_case_variant"](
        by_id["ort_tensorrt"], by_id["ort_tensorrt"]["variants"], 0,
    )
    later = funcs["_apply_full_owner_case_variant"](
        by_id["ort_tensorrt"], by_id["ort_tensorrt"]["variants"], 1,
    )
    assert "full" in first
    assert "full" not in later
    assert "composed" in later

    isolated_companion = funcs["_assign_full_baseline_owners"]([
        {
            "id": "hailo10_to_trt", "type": "matrix", "provider": "tensorrt",
            "stage1": {"type": "hailo", "hw_arch": "hailo10"},
            "stage2": {"type": "onnxruntime", "provider": "tensorrt"},
            "variants": ["full", "part1", "part2", "composed"],
        },
    ])[0]
    assert isolated_companion["_full_baseline_owner"] is False
    assert isolated_companion["_full_baseline_owner_run_id"] == ""
    assert isolated_companion["variants"] == ["part1", "part2", "composed"]
    assert isolated_companion["_native_full_trt_quality_companion"] is False


def test_three_case_full_identity_conflict_fails_closed(
    tmp_path: Path,
) -> None:
    rows = _normalize(
        tmp_path, _full_companion_rows(conflicting_endpoint=True),
    )
    full = [row for row in rows if row["variant"] == "full"]

    assert len(full) == 1
    result = full[0]
    assert result["case_id"] == "full"
    assert result["full_source_case_ids"] == ["b044", "b116", "b216"]
    assert result["full_baseline_frozen_identity_status"] == "conflict"
    assert result["full_baseline_frozen_identity_conflicts"] == [
        "endpoint_contract_hash",
    ]
    assert result["full_baseline_frozen_identity_values"][
        "endpoint_contract_hash"
    ] == ["1" * 64, "2" * 64]
    assert result["row_conflict"] == "full_baseline_frozen_identity_conflict"
    assert result["contract_consistent"] is False
    assert result["structural_contract_pass"] is False
    assert result["evidence_axes"]["structural_contract"]["pass"] is False
    assert result["structural_contract_reason"].startswith(
        "full_baseline_frozen_identity_conflict:"
    )
    assert result["final_pass"] is False
    assert result["eligible_for_ranking"] is False
    # The physical timing remains available for diagnosis; only scientific
    # admission is closed.
    assert result["runtime_ok"] is True


def test_legacy_raw_vs_norm_full_preprocessing_conflict_fails_closed(
    tmp_path: Path,
) -> None:
    source_rows = _full_companion_rows()
    for index, row in enumerate(source_rows):
        row.pop("full_preprocessing_contract_sha256")
        row["benchmark_input_policy"] = {
            "image_scale": "raw" if index == 2 else "norm",
            "preprocess_mode": "letterbox",
            "color_space": "RGB",
            "letterbox_pad_value": 114,
            # Timing-loop fields are intentionally not part of the semantic hash.
            "loop_count": 5 + index,
        }
    rows = _normalize(tmp_path, source_rows)
    result = next(row for row in rows if row["variant"] == "full")

    assert result["full_baseline_frozen_identity_status"] == "conflict"
    assert result["full_baseline_frozen_identity_conflicts"] == [
        "preprocessing_contract_sha256",
    ]
    assert len(result["full_baseline_frozen_identity_values"][
        "preprocessing_contract_sha256"
    ]) == 2
    assert result["contract_consistent"] is False
    assert result["structural_contract_pass"] is False
    assert result["evidence_axes"]["structural_contract"]["pass"] is False


def test_classification_endpoint_hash_normalizes_optional_batch_axis(
    tmp_path: Path,
) -> None:
    suite = tmp_path / "classification_suite"
    suite.mkdir()
    (suite / "output_contracts.json").write_text(json.dumps({
        "model_id": "resnet50", "task": "classification",
        "contracts": [{
            "model_id": "resnet50", "backend": "cuda_ort",
            "variant": "full", "task": "classification",
            "endpoint_mode": "classification_logits",
            "contract_status": "recorded", "host_tail_required": False,
            "postprocessing_required": False,
        }],
    }), encoding="utf-8")
    declaration = load_authoritative_output_contract(
        suite, backend="tensorrt", model_id="resnet50",
        variant="full", task="classification",
    )
    rank1 = runtime_output_contract(
        "classification", {"logits": np.zeros((1000,), dtype=np.float32)},
        raw_fallback=False, declared_contract=declaration,
    )
    rank2 = runtime_output_contract(
        "classification", {"logits": np.zeros((1, 1000), dtype=np.float32)},
        raw_fallback=False, declared_contract=declaration,
    )
    batch2 = runtime_output_contract(
        "classification", {"logits": np.zeros((2, 1000), dtype=np.float32)},
        raw_fallback=False, declared_contract=declaration,
    )

    assert rank1["endpoint_contract_hash"] == rank2["endpoint_contract_hash"]
    assert rank1["endpoint_contract_hash"] != batch2["endpoint_contract_hash"]
    assert rank1["tensor_signature"]["tensors"][0]["shape"] == [1000]
    assert rank2["tensor_signature"]["tensors"][0]["shape"] == [1, 1000]


def test_raw_detection_head_declaration_uses_canonical_raw_head_family(
    tmp_path: Path,
) -> None:
    suite = tmp_path / "raw_suite"
    suite.mkdir()
    (suite / "output_contracts.json").write_text(json.dumps({
        "model_id": "yolo26s", "task": "detection",
        "contracts": [{
            "model_id": "yolo26s", "backend": "cuda_ort",
            "variant": "full", "task": "detection",
            "endpoint_mode": "raw_detection_head",
            "contract_status": "recorded", "host_tail_required": True,
            "postprocessing_required": True,
        }],
    }), encoding="utf-8")
    declaration = load_authoritative_output_contract(
        suite, backend="tensorrt", model_id="yolo26s",
        variant="full", task="detection",
    )
    contract = runtime_output_contract(
        "detection",
        {
            "head_80": np.zeros((1, 3, 80, 80, 85), dtype=np.float32),
            "head_40": np.zeros((1, 3, 40, 40, 85), dtype=np.float32),
            "head_20": np.zeros((1, 3, 20, 20, 85), dtype=np.float32),
        },
        raw_fallback=False,
        declared_contract=declaration,
    )

    assert contract["contract_family"] == "raw_head"
    assert contract["stage"] == "raw_head"
    assert contract["endpoint_contract_complete"] is True
    assert len(contract["endpoint_contract_hash"]) == 64


def _suite_contracts(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "output_contracts.json").write_text(json.dumps({
        "schema": "onnx-splitpoint/output-contracts",
        "schema_version": 1,
        "model_id": "yolo26s",
        "contracts": [
            {
                "model_id": "yolo26s", "backend": "cuda_ort",
                "task": "detection",
                "variant": "full", "endpoint_mode": "decoded",
                "contract_status": "recorded", "host_tail_required": False,
                "postprocessing_required": False,
            },
            {
                "model_id": "yolo26s", "backend": "hailo10",
                "task": "detection",
                "variant": "full", "endpoint_mode": "raw_detection_head",
                "contract_status": "recorded", "host_tail_required": True,
                "postprocessing_required": True,
            },
        ],
    }), encoding="utf-8")


def test_authoritative_suite_contract_binds_alias_and_still_checks_values(
    tmp_path: Path,
) -> None:
    suite = tmp_path / "suite"
    _suite_contracts(suite)
    declaration = load_authoritative_output_contract(
        suite, backend="native_full_tensorrt", model_id="yolo26s",
        variant="full", task="detection",
    )
    assert declaration["contract_resolution_status"] == "attested"
    assert declaration["backend"] == "cuda_ort"
    assert declaration["stage"] == "decoded_nms"

    decoded = np.zeros((1, 4, 6), dtype=np.float32)
    decoded[..., 2:4] = 1.0
    decoded[..., 4] = 0.5
    valid = runtime_output_contract(
        "detection", {"output0": decoded}, raw_fallback=False,
        declared_contract=declaration,
    )
    assert valid["contract_family"] == "decoded_nms"
    assert valid["endpoint_contract_complete"] is True

    invalid_values = decoded.copy()
    invalid_values[..., 4] = 3.0
    rejected = runtime_output_contract(
        "detection", {"output0": invalid_values}, raw_fallback=False,
        declared_contract=declaration,
    )
    assert rejected["contract_family"] == "unknown"
    assert rejected["endpoint_contract_complete"] is False

    raw = load_authoritative_output_contract(
        suite, backend="native_full_hailo10h", model_id="yolo26s",
        variant="full", task="detection",
    )
    assert raw["backend"] == "hailo10"
    assert raw["stage"] == "raw_head"


def test_authoritative_suite_contract_rejects_wrong_model_and_unsafe_decoded(
    tmp_path: Path,
) -> None:
    suite = tmp_path / "suite"
    _suite_contracts(suite)

    wrong_model = load_authoritative_output_contract(
        suite, backend="tensorrt", model_id="yolov7_paper",
        variant="full", task="detection",
    )
    assert "stage" not in wrong_model
    assert wrong_model["contract_resolution_reason"] == (
        "suite_output_contract_model_mismatch"
    )

    payload = json.loads((suite / "output_contracts.json").read_text(encoding="utf-8"))
    payload["contracts"][0]["host_tail_required"] = True
    (suite / "output_contracts.json").write_text(
        json.dumps(payload), encoding="utf-8",
    )
    unsafe = load_authoritative_output_contract(
        suite, backend="tensorrt", model_id="yolo26s",
        variant="full", task="detection",
    )
    assert "stage" not in unsafe
    assert unsafe["contract_resolution_status"] == "conflict"
    assert "endpoint_postprocessing_flags_conflict" in unsafe[
        "contract_resolution_errors"
    ]


def test_authoritative_suite_contract_rejects_unrecorded_raw_head(
    tmp_path: Path,
) -> None:
    suite = tmp_path / "suite"
    _suite_contracts(suite)
    payload = json.loads((suite / "output_contracts.json").read_text(encoding="utf-8"))
    payload["contracts"][1]["contract_status"] = "inferred"
    (suite / "output_contracts.json").write_text(
        json.dumps(payload), encoding="utf-8",
    )

    unsafe = load_authoritative_output_contract(
        suite, backend="hailo10h", model_id="yolo26s",
        variant="full", task="detection",
    )

    assert "stage" not in unsafe
    assert "endpoint_contract_not_recorded" in unsafe[
        "contract_resolution_errors"
    ]
