from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from onnx_splitpoint_tool.workflow.runner import EvaluationWorkflowRunner


ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str, relative: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _pending_gate(variant: str, request_sha256: str) -> dict:
    return {
        "schema": "onnx-splitpoint/task-quality-gate",
        "schema_version": 2,
        "variant": variant,
        "status": "pending_central_evaluation",
        "decision": "pending_central_evaluation",
        "quality_input_request": {
            "variant": variant,
            "status": "pending_central_evaluation",
            "request": {"path": f"{variant}_request.json", "sha256": request_sha256},
        },
    }


def _quality_result(variant: str, request_sha256: str, *, setup: str = "setup-a") -> dict:
    return {
        "model_id": "resnet50",
        "task": "classification",
        "case_id": "b052",
        "run_id": "ort_tensorrt",
        "source_run_id": "ort_tensorrt",
        "source_setup_id": setup,
        "variant": variant,
        "status": "completed",
        "technical_status": "completed",
        "scientific_status": "pass",
        "decision": "pass",
        "n": 16,
        "source_request": f"quality_inputs/{setup}/results/b052/results_ort_tensorrt/{variant}_request.json",
        "source_request_sha256": f"sha256:{request_sha256}",
        "request_identity": {
            "model_id": "resnet50", "case_id": "b052", "source_run_id": "ort_tensorrt",
            "setup_id": setup, "variant": variant, "source_request_sha256": request_sha256,
        },
        "primary": {
            "metric": "top1_accuracy", "candidate": 0.8, "reference": 0.8,
            "delta": 0.0, "ci_low": 0.0, "ci_high": 0.0, "margin": 0.01,
        },
    }


def _split_quality_row(full_sha: str, composed_sha: str) -> dict:
    composed = _pending_gate("composed", composed_sha)
    return {
        "model_id": "resnet50",
        "task": "classification",
        "case_id": "b052",
        "run_id": "ort_tensorrt",
        "quality_source_run_id": "ort_tensorrt",
        "variant": "split",
        "primary_variant": "composed",
        "quality_source_variant": "composed",
        "source_paths": [
            "/run/remote_diagnostics/setup-a/case_reports/results/b052/"
            "results_ort_tensorrt/validation_report.json"
        ],
        "task_quality_gates_by_variant": {
            "full": _pending_gate("full", full_sha),
            "composed": composed,
        },
        "task_quality_gate": composed,
        "quality_evaluation_pending": True,
        "task_quality_policy": {
            "dataset_tier": "screening",
            "classification_max_top1_drop": 0.01,
        },
        "buildable": True,
        "runtime_executable": True,
        "contract_consistent": True,
    }


def test_supplemental_full_quality_result_updates_exact_variant_slot(tmp_path: Path) -> None:
    full_sha = "a" * 64
    composed_sha = "b" * 64
    normalized = tmp_path / "models/resnet50/benchmark_results/normalized_results.json"
    normalized.parent.mkdir(parents=True)
    normalized.write_text(
        json.dumps({"results": [_split_quality_row(full_sha, composed_sha)]}),
        encoding="utf-8",
    )
    runner = object.__new__(EvaluationWorkflowRunner)
    runner.run_dir = tmp_path
    runner.profile_payload = {
        "quality_gate": {"statistics": {"execution_location": "central_management", "workers": 4}}
    }

    # Deliberately process Full first: result completion order is nondeterministic.
    merge = runner._merge_central_quality_results([
        _quality_result("full", full_sha),
        _quality_result("composed", composed_sha),
    ])
    row = json.loads(normalized.read_text(encoding="utf-8"))["results"][0]

    assert merge["unmatched_result_count"] == 0
    assert merge["matched_completed_count"] == 2
    assert merge["matched_primary_result_count"] == 1
    assert merge["matched_supplemental_result_count"] == 1
    assert row["task_quality_gates_by_variant"]["full"]["decision"] == "pass"
    assert row["task_quality_gates_by_variant"]["composed"]["decision"] == "pass"
    assert row["task_quality_gate"]["variant"] == "composed"
    assert row["central_quality_supplemental_results"]["full"]["status"] == "completed"
    payload = json.loads(normalized.read_text(encoding="utf-8"))
    assert payload["central_quality_evaluation"]["status"] == "completed"
    assert payload["central_quality_evaluation"]["matched_result_count"] == 2


@pytest.mark.parametrize(
    ("setup", "request_sha"),
    [("wrong-setup", "a" * 64), ("setup-a", "c" * 64)],
)
def test_supplemental_variant_join_fails_closed_on_identity_mismatch(
    tmp_path: Path, setup: str, request_sha: str,
) -> None:
    normalized = tmp_path / "models/resnet50/benchmark_results/normalized_results.json"
    normalized.parent.mkdir(parents=True)
    normalized.write_text(
        json.dumps({"results": [_split_quality_row("a" * 64, "b" * 64)]}),
        encoding="utf-8",
    )
    runner = object.__new__(EvaluationWorkflowRunner)
    runner.run_dir = tmp_path
    runner.profile_payload = {
        "quality_gate": {"statistics": {"execution_location": "central_management", "workers": 4}}
    }

    merge = runner._merge_central_quality_results([
        _quality_result("full", request_sha, setup=setup),
    ])

    assert merge["matched_supplemental_result_count"] == 0
    assert merge["unmatched_result_count"] == 1
    assert merge["unmatched_results"][0]["join_status"] == "no_exact_row"
    payload = json.loads(normalized.read_text(encoding="utf-8"))
    assert payload["central_quality_evaluation"]["status"] == "partial"


def test_deepx_boundary_dump_binds_exact_selected_letterbox_tensor(tmp_path: Path) -> None:
    from PIL import Image

    deepx = _load_script("v267_deepx_exact_input", "scripts/native_deepx_trt_e2e_from_benchmarkset.py")
    source = np.zeros((4, 8, 3), dtype=np.uint8)
    source[..., 0] = 17
    source_image = tmp_path / "source.png"
    Image.fromarray(source, mode="RGB").save(source_image)
    selected = np.full((8, 8, 3), 114, dtype=np.uint8)
    selected[2:6, :, 0] = 17
    selected[2:6, :, 1:] = 0
    boundary = np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)

    manifest_path = Path(deepx._dump_boundary(
        boundary,
        selected,
        tmp_path / "boundary",
        producer="test",
        backend="deepx_to_trt",
        case="b038",
        precision="fp16",
        trt_input_name="part2_input",
        trt_input_dtype="float32",
        input_image=str(source_image),
        preprocess_mode="letterbox",
        letterbox_pad_value=114,
    ))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    exact_path = Path(manifest["selected_input_dump"])
    compatibility_path = Path(manifest["input_dump"])

    assert manifest["schema_version"] == 3
    assert manifest["preprocess"]["mode_effective"] == "letterbox"
    assert manifest["preprocess"]["pad_value"] == 114
    assert manifest["preprocess"]["input_source"] == "deepx_selected_input_exact_hwc_uint8"
    assert np.array_equal(np.load(exact_path, allow_pickle=False), selected)
    assert compatibility_path.read_bytes() == selected.tobytes()
    assert hashlib.sha256(exact_path.read_bytes()).hexdigest() == manifest["selected_input_dump_sha256"]


def test_yolo_probe_prefers_hash_verified_exact_selected_input_and_rejects_tampering(
    tmp_path: Path,
) -> None:
    probe = _load_script("v267_yolo_exact_input", "scripts/native_yolo_full_self_reference_probe.py")
    selected = np.full((8, 8, 3), 114, dtype=np.uint8)
    selected[2:6, :, :] = np.arange(3, dtype=np.uint8)
    selected_path = tmp_path / "deepx_selected_input.npy"
    np.save(selected_path, selected, allow_pickle=False)
    manifest_path = tmp_path / "native_fifo_boundary_manifest.json"
    manifest = {
        "schema": "onnx-splitpoint/native-boundary-dump",
        "schema_version": 3,
        "backend": "deepx_to_trt",
        "selected_input_dump": str(selected_path),
        "selected_input_dump_sha256": hashlib.sha256(selected_path.read_bytes()).hexdigest(),
        "selected_input_shape": [8, 8, 3],
        "selected_input_dtype": "uint8",
        "preprocess": {"mode_effective": "letterbox", "pad_value": 114, "ort_model_scale": "norm"},
        "_manifest_path": str(manifest_path),
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    feed = probe._input_dump_feed_from_manifest(manifest, [1, 3, 8, 8], "native")
    expected = np.transpose(selected.astype(np.float32) / 255.0, (2, 0, 1))[None]
    assert feed.flags.c_contiguous
    assert np.array_equal(feed, expected)

    selected_path.write_bytes(selected_path.read_bytes() + b"tampered")
    with pytest.raises(RuntimeError, match="sha256_mismatch"):
        probe._input_dump_feed_from_manifest(manifest, [1, 3, 8, 8], "native")


def test_missing_exact_deepx_input_is_unavailable_not_semantic_failure() -> None:
    probe = _load_script("v267_yolo_missing_input", "scripts/native_yolo_full_self_reference_probe.py")
    validator = _load_script("v267_validator_missing_input", "scripts/native_producer_validate_visualize.py")
    manifest = {
        "schema_version": 3,
        "backend": "deepx_to_trt",
        "preprocess": {"mode_effective": "letterbox", "pad_value": 114},
    }
    with pytest.raises(RuntimeError, match="exact_selected_input_evidence_missing"):
        probe._input_dump_feed_from_manifest(manifest, [1, 3, 8, 8], "native")

    converted = validator._probe_payload_to_self_reference({
        "schema": "onnx-splitpoint/native-yolo-full-self-reference-probe",
        "schema_version": 3,
        "ok": False,
        "semantic_ok": False,
        "semantic_available": False,
        "diagnosis": "full_onnx_input_evidence_unavailable",
        "evidence_error": "RuntimeError: exact_selected_input_evidence_missing",
    })
    assert converted is not None
    assert converted["semantic_available"] is False
    assert converted["ok"] is False
    assert "exact_selected_input_evidence_missing" in converted["reason"]


def test_v267_remote_script_mirrors_are_identical() -> None:
    for name in (
        "native_deepx_trt_e2e_from_benchmarkset.py",
        "native_yolo_full_self_reference_probe.py",
        "native_producer_validate_visualize.py",
    ):
        assert (ROOT / "scripts" / name).read_bytes() == (
            ROOT / "onnx_splitpoint_tool/resources/remote_scripts" / name
        ).read_bytes()
