from __future__ import annotations

from concurrent.futures import TimeoutError as FutureTimeoutError
import hashlib
import json
from pathlib import Path

import pytest

from onnx_splitpoint_tool.quality_cache import (
    CPUQualityReferenceIdentity,
    PersistentQualityCache,
    image_ids_fingerprint,
    prediction_fingerprint,
)
from onnx_splitpoint_tool.quality_service import (
    CPUQualityReferenceStore,
    EVALUATE_PAIRED_QUALITY_UNCERTAINTY,
    GENERATE_CPU_QUALITY_REFERENCE,
    HAILO_BUILD_RESOURCE_REASON,
    ImagePairingError,
    ManagementQualityService,
    QualityArtifactIntegrityError,
    QualityEvaluationRequest,
    URECS_RESOURCE_REASON,
    make_cpu_reference_identity,
    pair_prediction_records,
    prepare_evaluation,
    quality_request_from_manifest,
)


def _rows(values: list[float], *, reverse: bool = False):
    rows = [{"image_id": index + 1, "value": value} for index, value in enumerate(values)]
    return list(reversed(rows)) if reverse else rows


def _request(candidate: list[float], *, reverse: bool = False, repetitions: int = 80):
    return QualityEvaluationRequest(
        reference_records=_rows([1.0, 0.0, 1.0, 1.0, 0.0, 1.0], reverse=reverse),
        candidate_records=_rows(candidate, reverse=not reverse),
        annotations=[{"image_id": index + 1, "label": int(index % 2)} for index in range(6)],
        metric_gate_config={"primary_metric": "top1_accuracy", "policy_id": "test-v263"},
        repetitions=repetitions,
        seed=20260710,
        confidence_level=0.95,
        non_inferiority_margin=0.25,
        evaluator_factory="paired_mean",
        value_field="value",
        request_id="test-request",
    )


def test_same_image_ids_but_different_predictions_never_collide_in_cache(tmp_path: Path) -> None:
    first = _request([1.0, 0.0, 1.0, 0.0, 0.0, 1.0])
    second = _request([1.0, 1.0, 1.0, 0.0, 0.0, 1.0])
    first_key, _ = prepare_evaluation(first)
    second_key, _ = prepare_evaluation(second)

    assert first_key != second_key
    assert prediction_fingerprint(first.candidate_records) != prediction_fingerprint(second.candidate_records)

    with ManagementQualityService(tmp_path / "cache", workers=2) as service:
        first_result = service.evaluate(first)
        second_result = service.evaluate(second)
        first_cached = service.evaluate(first)

    assert first_result["evaluation_fingerprint"] == first_key
    assert second_result["evaluation_fingerprint"] == second_key
    assert first_result["primary"]["delta"] != second_result["primary"]["delta"]
    assert first_result["cache_hit"] is False
    assert first_cached["cache_hit"] is True
    assert len(list((tmp_path / "cache").rglob("*.json"))) == 2


def test_pairing_is_order_independent_and_canonical() -> None:
    reference = [
        {"image_id": "z", "value": 3},
        {"image_id": "a", "value": 1},
        {"image_id": "m", "value": 2},
    ]
    candidate = [
        {"image_id": "m", "value": 20},
        {"image_id": "z", "value": 30},
        {"image_id": "a", "value": 10},
    ]
    paired = pair_prediction_records(reference, candidate)

    assert paired.image_ids == ("a", "m", "z")
    assert [row["value"] for row in paired.reference] == [1, 2, 3]
    assert [row["value"] for row in paired.candidate] == [10, 20, 30]


@pytest.mark.parametrize(
    "reference,candidate,expected",
    [
        (
            [{"image_id": 1, "value": 1}, {"image_id": 1, "value": 2}],
            [{"image_id": 1, "value": 1}],
            "duplicate",
        ),
        (
            [{"image_id": 1, "value": 1}, {"image_id": 2, "value": 2}],
            [{"image_id": 1, "value": 1}],
            "missing candidate IDs",
        ),
        (
            [{"image_id": 1, "value": 1}],
            [{"image_id": 1, "value": 1}, {"image_id": 2, "value": 2}],
            "unexpected candidate IDs",
        ),
    ],
)
def test_pairing_rejects_duplicates_and_missing_or_extra_ids(reference, candidate, expected) -> None:
    with pytest.raises(ImagePairingError, match=expected):
        pair_prediction_records(reference, candidate)


def test_one_and_four_processes_produce_identical_statistical_result(tmp_path: Path) -> None:
    request = _request([1.0, 0.0, 1.0, 0.0, 1.0, 1.0], repetitions=120)
    with ManagementQualityService(tmp_path / "one", workers=1) as one:
        result_one = one.evaluate(request)
    with ManagementQualityService(tmp_path / "four", workers=4) as four:
        result_four = four.evaluate(request)

    assert result_one["evaluation_fingerprint"] == result_four["evaluation_fingerprint"]
    assert result_one["decision"] == result_four["decision"]
    assert result_one["bootstrap_workers_effective"] == 1
    assert result_four["bootstrap_workers_requested"] == 4
    assert result_four["bootstrap_workers_effective"] == 4
    assert result_four["bootstrap_sharding"] == "contiguous_deterministic_resample_plan"
    primary_one = dict(result_one["primary"])
    primary_four = dict(result_four["primary"])
    primary_one.pop("bootstrap_elapsed_s")
    primary_four.pop("bootstrap_elapsed_s")
    assert primary_one == primary_four


def test_resource_gate_pauses_new_work_until_resumed(tmp_path: Path) -> None:
    request = _request([1.0, 0.0, 1.0, 0.0, 1.0, 1.0], repetitions=40)
    with ManagementQualityService(tmp_path / "paused", workers=1) as service:
        service.pause(URECS_RESOURCE_REASON)
        service.pause(HAILO_BUILD_RESOURCE_REASON)
        future = service.submit(request)
        with pytest.raises(FutureTimeoutError):
            future.result(timeout=0.15)
        service.resume(URECS_RESOURCE_REASON)
        with pytest.raises(FutureTimeoutError):
            future.result(timeout=0.15)
        service.resume(HAILO_BUILD_RESOURCE_REASON)
        result = future.result(timeout=10.0)

    assert result["status_name"] == EVALUATE_PAIRED_QUALITY_UNCERTAINTY
    assert result["status"] == "completed"
    assert result["execution_location"] == "management_node"
    assert result["gpu_used"] is False


def test_cpu_reference_is_reused_and_excluded_from_performance_claims(tmp_path: Path) -> None:
    predictions = _rows([1.0, 0.0, 1.0])
    identity = make_cpu_reference_identity(
        model="model-sha256",
        dataset="dataset-sha256",
        preprocessing="preprocess-contract-v1",
        decoder="decoder-contract-v1",
        prediction_records=predictions,
    )
    identity_before_inference = make_cpu_reference_identity(
        model="model-sha256",
        dataset="dataset-sha256",
        preprocessing="preprocess-contract-v1",
        decoder="decoder-contract-v1",
        image_ids=[row["image_id"] for row in predictions],
    )
    assert identity_before_inference == identity
    calls = 0

    def generate():
        nonlocal calls
        calls += 1
        return predictions

    store = CPUQualityReferenceStore(tmp_path / "references")
    generated = store.materialize(identity, generate)
    reused = store.materialize(identity, generate)

    assert calls == 1
    assert generated["status_name"] == GENERATE_CPU_QUALITY_REFERENCE
    assert generated["cache_hit"] is False
    assert reused["cache_hit"] is True
    assert reused["manifest"]["provider"] == "onnxruntime_cpu"
    assert reused["manifest"]["semantic_reference_only"] is True
    assert reused["manifest"]["include_in_latency_fps_energy"] is False
    assert reused["manifest"]["include_in_ranking"] is False
    assert reused["manifest"]["include_in_pareto"] is False


def test_gpu_reference_identity_is_rejected() -> None:
    with pytest.raises(ValueError, match="ONNX Runtime CPU"):
        CPUQualityReferenceIdentity(
            model="m",
            dataset="d",
            preprocessing="p",
            decoder="x",
            image_ids="i",
            provider="onnxruntime_cuda",
            execution_device="management_gpu",
        )


def _write_runner_artifact(path: Path, payload: dict) -> dict:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    path.write_bytes(encoded)
    return {
        "path": path.name,
        "size_bytes": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _write_runner_request(tmp_path: Path, *, task: str, reference_records: list, candidate_records: list) -> Path:
    reference_payload = {
        "schema": "onnx-splitpoint/task-quality-reference-input",
        "schema_version": 1,
        "task": task,
        "pairing_key": "image_id",
        "reference_role": "canonical_cpu_ort",
        "semantic_reference_only": True,
        "records": reference_records,
    }
    candidate_payload = {
        "schema": "onnx-splitpoint/task-quality-candidate-input",
        "schema_version": 1,
        "task": task,
        "variant": "composed",
        "pairing_key": "image_id",
        "records": candidate_records,
    }
    reference = _write_runner_artifact(tmp_path / f"canonical_{task}_reference.json", reference_payload)
    candidate = _write_runner_artifact(tmp_path / "composed_candidate.json", candidate_payload)
    request = {
        "schema": "onnx-splitpoint/central-quality-evaluation-request",
        "schema_version": 1,
        "status": "pending_central_evaluation",
        "task": task,
        "variant": "composed",
        "pairing_key": "image_id",
        "execution_location": "management_node",
        "requested_by": "central_management",
        "reference": reference,
        "candidate": candidate,
        "record_count": len(candidate_records),
        "reference_record_count": len(reference_records),
        "policy_sha256": "a" * 64,
        "metric_gate_config": {
            "primary_metric": "top1_accuracy" if task == "classification" else "coco_ap_50_95",
            "non_inferiority_margin": 0.01,
            "guardrails": {"top5_accuracy_margin": 0.01, "ap50_margin": 0.01},
        },
        "statistics": {
            "method": "paired_bootstrap",
            "bootstrap_repetitions": 30,
            "seed": 20260710,
            "confidence_level": 0.95,
            "decision": "lower_one_sided_bound",
        },
    }
    request_path = tmp_path / "composed_request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    return request_path


def test_runner_manifest_loads_once_and_evaluates_classification_guardrail(tmp_path: Path) -> None:
    reference = [
        {"image_id": "a", "label_id": 1, "label_name": "one", "reference": {"top1_hit": True, "top5_hit": True}},
        {"image_id": "b", "label_id": 2, "label_name": "two", "reference": {"top1_hit": False, "top5_hit": True}},
        {"image_id": "c", "label_id": 3, "label_name": "three", "reference": {"top1_hit": True, "top5_hit": True}},
    ]
    candidate = [
        {"image_id": "c", "candidate": {"top1_hit": True, "top5_hit": True}},
        {"image_id": "a", "candidate": {"top1_hit": True, "top5_hit": True}},
        {"image_id": "b", "candidate": {"top1_hit": False, "top5_hit": True}},
    ]
    request_path = _write_runner_request(
        tmp_path, task="classification", reference_records=reference, candidate_records=candidate
    )
    request = quality_request_from_manifest(request_path)
    with ManagementQualityService(tmp_path / "cache", workers=2) as service:
        result = service.evaluate(request)

    assert result["decision"] == "pass"
    assert result["primary"]["metric"] == "top1_accuracy"
    assert result["primary"]["delta"] == 0.0
    assert result["guardrails"]["top5_accuracy"]["metric"] == "top5_accuracy"
    assert result["primary"]["bootstrap_skipped_reason"] == "candidate_reference_identical"


def test_runner_manifest_rejects_tampered_artifact(tmp_path: Path) -> None:
    reference = [{"image_id": "a", "reference": {"top1_hit": True, "top5_hit": True}}]
    candidate = [{"image_id": "a", "candidate": {"top1_hit": True, "top5_hit": True}}]
    request_path = _write_runner_request(
        tmp_path, task="classification", reference_records=reference, candidate_records=candidate
    )
    (tmp_path / "composed_candidate.json").write_text("{}", encoding="utf-8")
    with pytest.raises(QualityArtifactIntegrityError, match="size mismatch|SHA-256 mismatch"):
        quality_request_from_manifest(request_path)


def test_management_reference_override_replaces_remote_reference(tmp_path: Path) -> None:
    reference = [{"image_id": "a", "reference": {"top1_hit": True, "top5_hit": True}}]
    candidate = [{"image_id": "a", "candidate": {"top1_hit": True, "top5_hit": True}}]
    request_path = _write_runner_request(
        tmp_path, task="classification", reference_records=reference, candidate_records=candidate
    )
    management_reference = tmp_path / "management_cpu_reference.json"
    management_reference.write_bytes((tmp_path / "canonical_classification_reference.json").read_bytes())
    # The remote descriptor is now invalid.  The explicit management artifact
    # must replace it rather than silently falling back to the remote CPU file.
    (tmp_path / "canonical_classification_reference.json").write_text("{}", encoding="utf-8")
    request = quality_request_from_manifest(
        request_path,
        reference_artifact=management_reference,
    )
    with ManagementQualityService(tmp_path / "cache", workers=1) as service:
        result = service.evaluate(request)
    assert result["decision"] == "pass"
    assert result["reference_identity"].startswith("runner_reference_artifact:")


def test_remote_placeholder_requires_management_reference_and_checks_exact_image_ids(tmp_path: Path) -> None:
    reference = [{"image_id": "a", "label_id": 1, "reference": {"top1_hit": True, "top5_hit": True}}]
    candidate = [{
        "image_id": "a", "label_id": 1, "label_name": "one",
        "candidate": {"top1_hit": True, "top5_hit": True},
    }]
    request_path = _write_runner_request(
        tmp_path, task="classification", reference_records=reference, candidate_records=candidate
    )
    management_reference = tmp_path / "management_cpu_reference.json"
    management_reference.write_bytes((tmp_path / "canonical_classification_reference.json").read_bytes())
    manifest = json.loads(request_path.read_text(encoding="utf-8"))
    manifest["reference"] = {
        "source": "management_cpu_reference",
        "reference_role": "canonical_cpu_ort",
        "required": True,
        "expected_image_ids": ["a"],
        "expected_image_ids_sha256": image_ids_fingerprint(["a"]),
        "record_count": 1,
    }
    manifest["expected_image_ids"] = ["a"]
    manifest["expected_image_ids_sha256"] = image_ids_fingerprint(["a"])
    request_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(QualityArtifactIntegrityError, match="no path"):
        quality_request_from_manifest(request_path)
    loaded = quality_request_from_manifest(request_path, reference_artifact=management_reference)
    assert loaded.annotations == [{"image_id": "a", "label_id": 1, "label_name": "one"}]

    wrong_reference = tmp_path / "wrong_management_reference.json"
    wrong_payload = json.loads(management_reference.read_text(encoding="utf-8"))
    wrong_payload["records"][0]["image_id"] = "b"
    wrong_reference.write_text(json.dumps(wrong_payload), encoding="utf-8")
    with pytest.raises(ImagePairingError, match="image_id sets differ"):
        quality_request_from_manifest(request_path, reference_artifact=wrong_reference)


def test_detection_runner_payload_uses_prepared_cached_metric(tmp_path: Path) -> None:
    detection = {"class_id": 0, "score": 0.9, "x1": 0.0, "y1": 0.0, "x2": 10.0, "y2": 10.0}
    ground_truth = {"class_id": 0, "x1": 0.0, "y1": 0.0, "x2": 10.0, "y2": 10.0}
    reference = [
        {"image_id": "a", "ground_truth": [ground_truth], "reference": [detection]},
        {"image_id": "b", "ground_truth": [ground_truth], "reference": [detection]},
    ]
    candidate = [
        {"image_id": "b", "candidate": [detection]},
        {"image_id": "a", "candidate": [detection]},
    ]
    request_path = _write_runner_request(
        tmp_path, task="detection", reference_records=reference, candidate_records=candidate
    )
    request = quality_request_from_manifest(request_path)
    with ManagementQualityService(tmp_path / "cache", workers=1) as service:
        result = service.evaluate(request)

    assert result["decision"] == "pass"
    assert result["primary"]["metric"] == "coco_ap_50_95"
    assert result["primary"]["candidate"] == pytest.approx(1.0)
    assert result["guardrails"]["ap50"]["candidate"] == pytest.approx(1.0)
    assert result["primary"]["bootstrap_engine"].endswith(":detection_quality_evaluator")
