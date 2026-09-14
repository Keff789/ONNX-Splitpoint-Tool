from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
from typing import Any

import pytest

from onnx_splitpoint_tool.quality_service import (
    CPUQualityReferenceStore,
    QualityArtifactIntegrityError,
    make_cpu_reference_identity,
    quality_request_from_manifest,
)
from onnx_splitpoint_tool.workflow.results import _apply_quality_source_identity_v265


def _reference_records() -> list[dict[str, Any]]:
    return [
        {
            "image_id": f"image-{index}",
            "label_id": index,
            "reference": {"top1_hit": True, "top5_hit": True},
        }
        for index in range(8)
    ]


def _identity(records: list[dict[str, Any]]):
    return make_cpu_reference_identity(
        model="model-contract",
        dataset="dataset-contract",
        preprocessing="preprocessing-contract",
        decoder="decoder-contract",
        prediction_records=records,
    )


def _process_materialize(
    root: str,
    records: list[dict[str, Any]],
    start: Any,
    output: Any,
) -> None:
    start.wait(timeout=10.0)
    try:
        result = CPUQualityReferenceStore(root).materialize(
            _identity(records), lambda: records
        )
        output.put(("ok", bool(result["cache_hit"])))
    except Exception as exc:  # pragma: no cover - returned to parent assertion
        output.put(("error", f"{type(exc).__name__}: {exc}"))


def test_reference_store_serializes_independent_instances_and_publishes_atomically(
    tmp_path: Path,
) -> None:
    records = _reference_records()
    identity = _identity(records)
    calls = 0

    def materialize() -> dict[str, Any]:
        nonlocal calls

        def generate() -> list[dict[str, Any]]:
            nonlocal calls
            calls += 1
            return records

        # Deliberately use one store instance per coordinator task.  This is
        # the Run27 failure mode that five per-instance RLocks did not protect.
        return CPUQualityReferenceStore(tmp_path / "store").materialize(identity, generate)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _index: materialize(), range(16)))

    assert calls == 1
    assert sum(not result["cache_hit"] for result in results) == 1
    assert sum(result["cache_hit"] for result in results) == 15
    artifact_dir = next(path for path in (tmp_path / "store").glob("*/*") if path.is_dir())
    assert sorted(path.name for path in artifact_dir.iterdir()) == ["manifest.json", "predictions.json"]
    assert not list((tmp_path / "store").rglob("*.staging"))


@pytest.mark.skipif(os.name != "posix", reason="campaign process locking uses POSIX flock")
def test_reference_store_serializes_independent_processes(tmp_path: Path) -> None:
    records = _reference_records()
    context = mp.get_context("fork")
    start = context.Event()
    output = context.Queue()
    processes = [
        context.Process(
            target=_process_materialize,
            args=(str(tmp_path / "store"), records, start, output),
        )
        for _index in range(4)
    ]
    for process in processes:
        process.start()
    start.set()
    outcomes = [output.get(timeout=15.0) for _process in processes]
    for process in processes:
        process.join(timeout=15.0)
        assert process.exitcode == 0

    assert all(status == "ok" for status, _value in outcomes), outcomes
    assert sum(value is False for _status, value in outcomes) == 1
    assert sum(value is True for _status, value in outcomes) == 3


def _quality_row(*, run: str, setup: str, variant: str = "full") -> dict[str, Any]:
    return {
        "model_id": "resnet50",
        "task": "classification",
        "case_id": "b052",
        "variant": variant,
        "primary_variant": "composed" if variant == "split" else "full",
        "quality_source_variant": "composed" if variant == "split" else "full",
        "quality_source_run_id": run,
        "quality_source_setup_ids": [setup],
        "technical_status": "completed",
        "quality_evaluation_pending": True,
        "task_quality_policy": {"dataset_tier": "screening"},
    }


def _quality_result(
    *,
    run: str,
    setup: str,
    variant: str = "full",
    status: str = "completed",
    request: str = "request.json",
) -> dict[str, Any]:
    return {
        "status": status,
        "technical_status": "completed" if status == "completed" else "failed",
        "scientific_status": "pass" if status == "completed" else "unavailable",
        "decision": "pass" if status == "completed" else "unavailable",
        "model_id": "resnet50",
        "task": "classification",
        "case_id": "b052",
        "variant": variant,
        "source_run_id": run,
        "source_setup_id": setup,
        "source_request": request,
        "source_request_sha256": "a" * 64,
        "request_identity": {
            "model_id": "resnet50",
            "case_id": "b052",
            "variant": variant,
            "source_run_id": run,
            "setup_id": setup,
        },
        "primary": {
            "metric": "top1_accuracy",
            "candidate": 0.8,
            "reference": 0.8,
            "delta": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "margin": 0.01,
            "decision": "pass",
        },
        "guardrails": {},
        "n": 16,
    }


def test_central_quality_join_is_exact_one_to_one_and_failure_is_fail_closed(
    tmp_path: Path,
) -> None:
    # Imported here so the store tests remain independent of optional report
    # dependencies during collection.
    from onnx_splitpoint_tool.workflow.runner import EvaluationWorkflowRunner

    normalized = tmp_path / "models/resnet50/benchmark_results/normalized_results.json"
    normalized.parent.mkdir(parents=True)
    rows = [
        _quality_row(run="hailo10", setup="orin_nx_hailo10_01"),
        _quality_row(run="hailo10_to_trt", setup="orin_nx_hailo10_01"),
        _quality_row(run="deepx_m1_to_trt", setup="orin_nx_deepx_m1_01", variant="split"),
        _quality_row(run="ort_tensorrt", setup="orin_nx_deepx_m1_01", variant="split"),
    ]
    normalized.write_text(json.dumps({"results": rows}), encoding="utf-8")

    runner = object.__new__(EvaluationWorkflowRunner)
    runner.run_dir = tmp_path
    runner.profile_payload = {
        "quality_gate": {"statistics": {"execution_location": "central_management", "workers": 4}}
    }
    results = [
        _quality_result(
            run="hailo10", setup="orin_nx_hailo10_01", request="hailo10/full_request.json"
        ),
        _quality_result(
            run="deepx_m1_to_trt",
            setup="orin_nx_deepx_m1_01",
            variant="composed",
            status="failed",
            request="deepx/composed_request.json",
        ),
        # Same source run but wrong setup: it must remain transparent and
        # unmatched instead of borrowing the DeepX/ORT row.
        _quality_result(
            run="ort_tensorrt",
            setup="wrong_setup",
            variant="composed",
            request="wrong/composed_request.json",
        ),
    ]

    merge = runner._merge_central_quality_results(results)
    merged = json.loads(normalized.read_text(encoding="utf-8"))["results"]

    assert merge["matched_completed_count"] == 1
    assert merge["matched_failed_count"] == 1
    assert merge["unmatched_result_count"] == 1
    assert merge["unmatched_results"][0]["join_status"] == "no_exact_row"
    assert merged[0]["central_quality_result_path"] == "hailo10/full_request.json"
    assert "central_quality_result_path" not in merged[1]
    assert merged[2]["task_quality_gate"]["decision"] == "unavailable"
    assert merged[2]["central_quality_technical_status"] == "failed"
    assert merged[2]["technical_status"] == "completed"
    assert merged[3]["quality_evaluation_pending"] is True


def test_normalized_quality_source_identity_uses_exact_run_path() -> None:
    row = {
        "source_tag": "hailo10_to_trt_auto",
        "run_id": "hailo10_to_tensorrt",
        "variant": "split",
        "primary_variant": "composed",
        "source_paths": [
            "/run/remote_diagnostics/orin_nx_hailo10_01/lean_bundle/b052/"
            "results_hailo10_to_trt/validation_report.json",
            "/run/remote_diagnostics/orin_nx_hailo8_01/lean_bundle/b052/"
            "results_hailo8_to_trt/validation_report.json",
        ],
    }

    _apply_quality_source_identity_v265(row)

    assert row["quality_source_run_id"] == "hailo10_to_trt"
    assert row["quality_source_setup_ids"] == ["orin_nx_hailo10_01"]
    assert row["quality_source_variant"] == "composed"


def _write_artifact(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    path.write_bytes(encoded)
    return {
        "path": path.name,
        "size_bytes": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def test_classification_request_fails_closed_with_precise_missing_metric(tmp_path: Path) -> None:
    reference_payload = {
        "schema": "onnx-splitpoint/task-quality-reference-input",
        "schema_version": 1,
        "task": "classification",
        "pairing_key": "image_id",
        "reference_role": "canonical_cpu_ort",
        "semantic_reference_only": True,
        "records": [{
            "image_id": "i0",
            "label_id": 3,
            "reference": {"top1_hit": True, "top5_hit": True},
        }],
    }
    candidate_payload = {
        "schema": "onnx-splitpoint/task-quality-candidate-input",
        "schema_version": 1,
        "task": "classification",
        "variant": "composed",
        "pairing_key": "image_id",
        "records": [{"image_id": "i0", "label_id": 3, "candidate": {}}],
    }
    reference = _write_artifact(tmp_path / "reference.json", reference_payload)
    candidate = _write_artifact(tmp_path / "candidate.json", candidate_payload)
    request = {
        "schema": "onnx-splitpoint/central-quality-evaluation-request",
        "schema_version": 1,
        "task": "classification",
        "variant": "composed",
        "pairing_key": "image_id",
        "execution_location": "management_node",
        "reference": reference,
        "candidate": candidate,
        "record_count": 1,
        "reference_record_count": 1,
        "metric_gate_config": {"non_inferiority_margin": 0.01},
        "statistics": {"bootstrap_repetitions": 25, "confidence_level": 0.95},
    }
    request_path = tmp_path / "composed_request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")

    with pytest.raises(QualityArtifactIntegrityError, match=r"candidate\.top1_hit"):
        quality_request_from_manifest(request_path)


def test_runner_preserves_labeled_candidate_when_optional_reference_is_unavailable() -> None:
    source = Path(
        "onnx_splitpoint_tool/resources/templates/run_split_onnxruntime.py.txt"
    ).read_text(encoding="utf-8")

    assert "def _run_dataset_candidate_cls(" in source
    assert 'row["candidate_metric_status"] = "available_reference_comparison_unavailable"' in source
    assert 'row["gt"] = _classification_gt_metrics(' in source
    assert 'row["task_quality_reference_source"] = "management_cpu_reference_pending"' in source
    assert 'metrics_variant["reference_comparison_status"] = "delegated_to_central_management"' in source
    assert 'comparisons[v]["semantic_validation_status"] = "delegated_to_central_management"' in source
