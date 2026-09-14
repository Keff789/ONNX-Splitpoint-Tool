from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import onnx_splitpoint_tool.quality_replay as quality_replay_module
from onnx_splitpoint_tool.quality_cache import (
    CACHE_RESULT_CONTRACT_VERSION,
    CACHE_SCHEMA,
    CACHE_SCHEMA_VERSION,
    EVALUATION_FINGERPRINT_SCHEMA,
    PersistentQualityCache,
    QualityFingerprintError,
    evaluation_fingerprint,
    json_fingerprint,
)
from onnx_splitpoint_tool.quality_metrics import detection_quality_evaluator
from onnx_splitpoint_tool.quality_replay import (
    CANONICAL_FULL_ONLY_IDENTITIES,
    OfflineQualityReplayError,
    REPLAY_CSV_NAME,
    REPLAY_OUTPUT_NAME,
    replay_evaluation_run,
)
from onnx_splitpoint_tool.quality_service import (
    DETECTION_QUALITY_ALGORITHM_VERSION,
    ManagementQualityService,
    QualityEvaluationRequest,
    _cached_guardrail_contract_matches_request,
    prepare_evaluation,
    quality_request_from_manifest,
    require_configured_guardrails,
)


def _box(*, x2: float = 10.0, score: float = 0.9) -> dict:
    return {
        "class_id": 0,
        "score": score,
        "x1": 0.0,
        "y1": 0.0,
        "x2": x2,
        "y2": 10.0,
    }


def _ground_truth() -> dict:
    row = _box()
    row.pop("score")
    return row


def test_detection_evaluator_materialises_configured_ap75() -> None:
    reference = [{"image_id": "i1", "reference": [_box()], "ground_truth": [_ground_truth()]}]
    candidate = [{"image_id": "i1", "candidate": [_box(x2=7.0)]}]
    evaluator = detection_quality_evaluator(
        reference,
        candidate,
        [{"image_id": "i1", "ground_truth": [_ground_truth()]}],
        {
            "metric_gate_config": {
                "task": "detection",
                "non_inferiority_margin": 1.0,
                "guardrails": {"ap50_margin": 0.01, "ap75_margin": 0.01},
            }
        },
    )

    result = evaluator.evaluate(np.ones((1,), dtype=np.float64))

    assert result["guardrails"]["ap50"]["candidate"] == pytest.approx(1.0)
    assert result["guardrails"]["ap50"]["reference"] == pytest.approx(1.0)
    assert result["guardrails"]["ap75"]["candidate"] == pytest.approx(0.0)
    assert result["guardrails"]["ap75"]["reference"] == pytest.approx(1.0)
    assert result["guardrails"]["ap75"]["delta"] == pytest.approx(-1.0)


def test_configured_but_missing_guardrail_fails_closed() -> None:
    with pytest.raises(ValueError, match=r"omitted configured guardrail\(s\): ap75.*fails closed"):
        require_configured_guardrails(
            {
                "task": "detection",
                "guardrails": {"ap50_margin": 0.01, "ap75_margin": 0.01},
            },
            {"primary": {}, "guardrails": {"ap50": {}}},
        )


def test_cache_result_guardrail_list_must_match_current_request_exactly() -> None:
    gate = {
        "task": "detection",
        "guardrails": {"ap50_margin": 0.01, "ap75_margin": 0.01},
    }
    stale = {
        "configured_guardrails": ["ap50"],
        "guardrails": {"ap50": {"decision": "pass"}},
        "guardrail_contract_complete": True,
    }
    current = {
        **stale,
        "configured_guardrails": ["ap50", "ap75"],
        "guardrails": {
            **stale["guardrails"],
            "ap75": {"decision": "pass"},
        },
    }

    assert _cached_guardrail_contract_matches_request(gate, stale) is False
    assert _cached_guardrail_contract_matches_request(gate, current) is True


def test_v27521_cache_envelope_and_algorithm_key_are_not_reused(tmp_path: Path) -> None:
    common = {
        "reference_predictions_sha256": "1" * 64,
        "candidate_predictions_sha256": "2" * 64,
        "annotations_sha256": "3" * 64,
        "metric_gate_config": {
            "primary_metric": "coco_ap_50_95",
            "guardrails": {"ap50_margin": 0.01, "ap75_margin": 0.01},
        },
        "seed_schema": {"seed": 7, "repetitions": 10, "image_count": 2},
    }
    old_key = evaluation_fingerprint(
        **common,
        algorithm_version="management_paired_quality_v2:detection-cached-matching-v1",
    )
    new_key = evaluation_fingerprint(
        **common,
        algorithm_version=DETECTION_QUALITY_ALGORITHM_VERSION,
    )
    assert old_key != new_key
    assert EVALUATION_FINGERPRINT_SCHEMA.endswith("v3")  # nullable uncertainty contract
    assert CACHE_SCHEMA_VERSION == 3

    cache = PersistentQualityCache(tmp_path / "cache")
    stale_path = cache.path_for(new_key)
    stale_path.parent.mkdir(parents=True)
    stale_result = {"decision": "pass", "guardrails": {"ap50": {"decision": "pass"}}}
    stale_path.write_text(
        json.dumps(
            {
                "schema": CACHE_SCHEMA,
                "schema_version": 1,
                "evaluation_fingerprint": new_key,
                "result_sha256": json_fingerprint(stale_result),
                "result": stale_result,
            }
        ),
        encoding="utf-8",
    )
    assert cache.get(new_key) is None


def _v2_cache_result() -> dict:
    return {
        "quality_result_contract_version": CACHE_RESULT_CONTRACT_VERSION,
        "guardrail_contract_complete": True,
        "configured_guardrails": ["ap50", "ap75"],
        "guardrails": {
            "ap50": {"decision": "pass"},
            "ap75": {"decision": "pass"},
        },
        "decision": "pass",
    }


def test_v2_cache_get_rejects_forged_complete_guardrail_contract(tmp_path: Path) -> None:
    cache = PersistentQualityCache(tmp_path / "cache")
    key = "4" * 64
    forged = _v2_cache_result()
    forged["guardrails"].pop("ap75")
    path = cache.path_for(key)
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "schema": CACHE_SCHEMA,
                "schema_version": CACHE_SCHEMA_VERSION,
                "result_contract_version": CACHE_RESULT_CONTRACT_VERSION,
                "evaluation_fingerprint": key,
                "result_sha256": json_fingerprint(forged),
                "result": forged,
            }
        ),
        encoding="utf-8",
    )

    assert cache.get(key) is None


def test_v2_cache_put_rejects_forged_complete_guardrail_contract(tmp_path: Path) -> None:
    cache = PersistentQualityCache(tmp_path / "cache")
    forged = _v2_cache_result()
    forged["guardrails"].pop("ap75")

    with pytest.raises(
        QualityFingerprintError,
        match=r"omitted configured guardrail\(s\): ap75",
    ):
        cache.put("5" * 64, forged)


@pytest.mark.parametrize(
    "configured_guardrails",
    ["ap50", ["ap50", "ap50"], ["ap50", ""], ["ap50", 75]],
)
def test_v2_cache_put_rejects_invalid_or_duplicate_configured_guardrails(
    tmp_path: Path,
    configured_guardrails: object,
) -> None:
    cache = PersistentQualityCache(tmp_path / "cache")
    result = _v2_cache_result()
    result["configured_guardrails"] = configured_guardrails

    with pytest.raises(QualityFingerprintError, match="configured_guardrails"):
        cache.put("6" * 64, result)


def _request_with_scoped_metadata(
    *, request_id: str, reference_identity: str, completion_sha256: str
) -> QualityEvaluationRequest:
    records = [{"image_id": "i1", "value": 1.0}]
    return QualityEvaluationRequest(
        reference_records=records,
        candidate_records=records,
        annotations=[],
        metric_gate_config={"primary_metric": "paired_mean", "guardrails": {}},
        repetitions=4,
        seed=20260710,
        confidence_level=0.95,
        non_inferiority_margin=0.01,
        request_id=request_id,
        reference_identity=reference_identity,
        candidate_execution_completion_contract_sha256=completion_sha256,
    )


def _assert_request_scoped_metadata(
    result: dict,
    *,
    request_id: str,
    reference_identity: str,
    completion_sha256: str,
) -> None:
    assert result["request_id"] == request_id
    assert result["reference_identity"] == reference_identity
    assert (
        result["candidate_execution_completion_contract_sha256"]
        == completion_sha256
    )


def test_cache_hit_rebinds_request_scoped_metadata(tmp_path: Path) -> None:
    first_request = _request_with_scoped_metadata(
        request_id="first-request",
        reference_identity="first-reference",
        completion_sha256="1" * 64,
    )
    second_request = _request_with_scoped_metadata(
        request_id="second-request",
        reference_identity="second-reference",
        completion_sha256="2" * 64,
    )
    assert prepare_evaluation(first_request)[0] == prepare_evaluation(second_request)[0]

    with ManagementQualityService(tmp_path / "cache", workers=1) as service:
        first = service.evaluate(first_request, timeout=10.0)
        second = service.evaluate(second_request, timeout=10.0)

    assert first["cache_hit"] is False
    assert second["cache_hit"] is True
    _assert_request_scoped_metadata(
        first,
        request_id="first-request",
        reference_identity="first-reference",
        completion_sha256="1" * 64,
    )
    _assert_request_scoped_metadata(
        second,
        request_id="second-request",
        reference_identity="second-reference",
        completion_sha256="2" * 64,
    )


def test_inflight_waiters_each_rebind_request_scoped_metadata(tmp_path: Path) -> None:
    first_request = _request_with_scoped_metadata(
        request_id="inflight-first",
        reference_identity="inflight-reference-first",
        completion_sha256="3" * 64,
    )
    second_request = _request_with_scoped_metadata(
        request_id="inflight-second",
        reference_identity="inflight-reference-second",
        completion_sha256="4" * 64,
    )
    assert prepare_evaluation(first_request)[0] == prepare_evaluation(second_request)[0]

    with ManagementQualityService(tmp_path / "cache", workers=1) as service:
        service.pause("test-inflight-coalescing")
        try:
            first_future = service.submit(first_request)
            second_future = service.submit(second_request)
        finally:
            service.resume("test-inflight-coalescing")
        first = first_future.result(timeout=10.0)
        second = second_future.result(timeout=10.0)

    assert first["cache_hit"] is False
    assert second["cache_hit"] is False
    assert first["evaluation_fingerprint"] == second["evaluation_fingerprint"]
    _assert_request_scoped_metadata(
        first,
        request_id="inflight-first",
        reference_identity="inflight-reference-first",
        completion_sha256="3" * 64,
    )
    _assert_request_scoped_metadata(
        second,
        request_id="inflight-second",
        reference_identity="inflight-reference-second",
        completion_sha256="4" * 64,
    )


def _write_json(path: Path, payload: dict) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    path.write_bytes(encoded)
    return {
        "path": path.name,
        "size_bytes": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _reference_payload() -> dict:
    return {
        "schema": "onnx-splitpoint/task-quality-reference-input",
        "schema_version": 1,
        "task": "detection",
        "pairing_key": "image_id",
        "reference_role": "canonical_cpu_ort",
        "semantic_reference_only": True,
        "records": [
            {
                "image_id": "i1",
                "ground_truth": [_ground_truth()],
                "reference": [_box()],
            }
        ],
    }


def _candidate_payload(variant: str = "full") -> dict:
    return {
        "schema": "onnx-splitpoint/task-quality-candidate-input",
        "schema_version": 1,
        "task": "detection",
        "variant": variant,
        "pairing_key": "image_id",
        "records": [{"image_id": "i1", "candidate": [_box()]}],
    }


def _make_replay_run(tmp_path: Path, *, write_requests: bool = True) -> tuple[Path, bytes]:
    run_dir = tmp_path / "historical_run"
    reference_path = (
        run_dir / "quality_management" / "references" / "yolov7_paper"
        / "canonical_cpu_reference.json"
    )
    _write_json(reference_path, _reference_payload())
    reference_file_sha = hashlib.sha256(reference_path.read_bytes()).hexdigest()

    rows = []

    def add_request_row(
        *, index: int, run_id: str, setup_id: str, variant: str, case_id: str
    ) -> None:
        request_path = (
            run_dir / "models" / "yolov7_paper" / "benchmark_results"
            / "quality_inputs" / setup_id / f"request_{index}" / f"{variant}_request.json"
        )
        if write_requests:
            candidate_descriptor = _write_json(
                request_path.parent / f"{variant}_candidate.json", _candidate_payload(variant)
            )
            request = {
                "schema": "onnx-splitpoint/central-quality-evaluation-request",
                "schema_version": 1,
                "status": "pending_central_evaluation",
                "task": "detection",
                "variant": variant,
                "pairing_key": "image_id",
                "execution_location": "management_node",
                "requested_by": "central_management",
                "source_run_id": run_id,
                "setup_id": setup_id,
                "reference": {"source": "management_cpu_reference", "required": True},
                "candidate": candidate_descriptor,
                "policy_sha256": "a" * 64,
                "metric_gate_config": {
                    "primary_metric": "coco_ap_50_95",
                    "non_inferiority_margin": 0.01,
                    "guardrails": {"ap50_margin": 0.01, "ap75_margin": 0.01},
                },
                "statistics": {
                    "method": "paired_bootstrap",
                    "bootstrap_repetitions": 5,
                    "seed": 20260710,
                    "confidence_level": 0.95,
                    "decision": "lower_one_sided_bound",
                },
            }
            _write_json(request_path, request)
            request_sha = hashlib.sha256(request_path.read_bytes()).hexdigest()
        else:
            request_sha = "a" * 64
        row = {
                "model_id": "yolov7_paper",
                "task": "detection",
                "case_id": case_id,
                "variant": variant,
                "source_run_id": run_id,
                "source_setup_id": setup_id,
                "source_request": request_path.relative_to(run_dir).as_posix(),
                "source_request_sha256": f"sha256:{request_sha}",
                "algorithm_version": "management_paired_quality_v2:detection-cached-matching-v1",
                "evaluation_fingerprint": str(index) * 64,
                "reference_identity": "b" * 64,
                "management_cpu_reference": {
                    "reference_path": str(reference_path),
                    "reference_sha256": f"sha256:{reference_file_sha}",
                },
                "decision": "pass",
                "reference_predictions_sha256": "1" * 64,
                "candidate_predictions_sha256": "2" * 64,
                "annotations_sha256": "3" * 64,
            }
        if write_requests:
            request = quality_request_from_manifest(
                request_path,
                verify_artifacts=True,
                reference_artifact=reference_path,
            )
            _, prepared = prepare_evaluation(request)
            for field in (
                "reference_predictions_sha256",
                "candidate_predictions_sha256",
                "annotations_sha256",
            ):
                row[field] = prepared[field]
        rows.append(row)

    for index, (run_id, setup_id, variant) in enumerate(CANONICAL_FULL_ONLY_IDENTITIES):
        add_request_row(
            index=index,
            run_id=run_id,
            setup_id=setup_id,
            variant=variant,
            case_id="full" if run_id == "native_full_tensorrt" else "b044",
        )
    # The two generic TRT rows are additional diagnostics.  In particular,
    # ort_tensorrt/full must not leak into --full-only just because its variant
    # happens to be named "full".
    add_request_row(
        index=4,
        run_id="ort_tensorrt",
        setup_id="orin_nx_hailo8_01",
        variant="composed",
        case_id="b044",
    )
    add_request_row(
        index=5,
        run_id="ort_tensorrt",
        setup_id="orin_nx_hailo8_01",
        variant="full",
        case_id="b044",
    )
    summary_path = run_dir / "quality_management" / "central_quality_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps({
        "schema": "onnx-splitpoint/central-quality-summary",
        "schema_version": 1,
        "status": "ok",
        "request_count": len(rows),
        "results": rows,
    }), encoding="utf-8")
    return run_dir, summary_path.read_bytes()


def test_full_only_replay_uses_four_exact_identities_and_keeps_run_read_only(
    tmp_path: Path,
) -> None:
    run_dir, original_summary = _make_replay_run(tmp_path)
    out_dir = tmp_path / "separate_replay"

    output = replay_evaluation_run(run_dir, out_dir=out_dir, workers=2, full_only=True)

    assert output["request_count"] == 4
    assert output["technical_status"] == "ok"
    assert output["scientific_status"] == "pass"
    assert output["scientific_pass"] is True
    assert output["hardware_executed"] is False
    assert output["historical_summary_mutated"] is False
    assert output["canonical_full_only_status"] == "complete"
    assert len(output["canonical_full_only_results"]) == 4
    assert output["canonical_full_only_decision_counts"] == {"pass": 4}
    assert output["canonical_full_only_quality_decision"] == "pass"
    assert [_identity_from_result(row) for row in output["results"]] == list(
        CANONICAL_FULL_ONLY_IDENTITIES
    )
    assert all(row["algorithm_version"] == DETECTION_QUALITY_ALGORITHM_VERSION for row in output["results"])
    assert all(row["guardrail_contract_complete"] is True for row in output["results"])
    assert all("ap75" in row["guardrails"] for row in output["results"])
    assert len({(row["source_run_id"], row["source_setup_id"]) for row in output["results"]}) == 4
    assert (out_dir / REPLAY_OUTPUT_NAME).is_file()
    assert (out_dir / REPLAY_CSV_NAME).is_file()
    with (out_dir / REPLAY_CSV_NAME).open(
        "r", encoding="utf-8", newline=""
    ) as stream:
        header = next(csv.reader(stream))
    assert len(header) == len(set(header))
    assert header.count("ap50_delta") == 1
    assert header.count("ap75_delta") == 1
    assert (run_dir / "quality_management" / "central_quality_summary.json").read_bytes() == original_summary
    assert not (run_dir / "quality_management" / "offline_replay_v27522").exists()


def test_default_replay_reports_all_six_without_candidate_sha_deduplication(
    tmp_path: Path,
) -> None:
    run_dir, _ = _make_replay_run(tmp_path)
    out_dir = tmp_path / "all_rows_replay"

    output = replay_evaluation_run(run_dir, out_dir=out_dir, workers=2)

    assert output["request_count"] == 6
    assert len(output["results"]) == 6
    assert len(output["canonical_full_only_results"]) == 4
    assert output["canonical_full_only_status"] == "complete"
    assert output["decision_counts"] == {"pass": 6}
    assert output["canonical_full_only_decision_counts"] == {"pass": 4}
    assert output["canonical_full_only_quality_decision"] == "pass"
    assert sum(row["source_run_id"] == "ort_tensorrt" for row in output["results"]) == 2
    assert len({row["evaluation_fingerprint"] for row in output["results"]}) < 6
    assert len(
        {
            (row["source_run_id"], row["source_setup_id"], row["variant"], row["source_request"])
            for row in output["results"]
        }
    ) == 6


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("wrong_schema", "schema is not the sealed replay source"),
        ("bool_version", "schema_version is not the supported"),
        ("future_version", "schema_version is not the supported"),
        ("non_mapping_result", "results must be an array of objects"),
        ("count_mismatch", "request_count does not exactly match"),
    ],
)
def test_replay_requires_a_lossless_sealed_central_summary(
    tmp_path: Path, case: str, message: str,
) -> None:
    run_dir, _ = _make_replay_run(tmp_path)
    summary_path = (
        run_dir / "quality_management" / "central_quality_summary.json"
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if case == "wrong_schema":
        summary["schema"] = "onnx-splitpoint/not-central-quality"
    elif case == "bool_version":
        summary["schema_version"] = True
    elif case == "future_version":
        summary["schema_version"] = 2
    elif case == "non_mapping_result":
        summary["results"].append("malformed-result")
        summary["request_count"] += 1
    elif case == "count_mismatch":
        summary["request_count"] -= 1
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    out_dir = tmp_path / "replay_must_not_exist"

    with pytest.raises(OfflineQualityReplayError, match=message):
        replay_evaluation_run(run_dir, out_dir=out_dir, workers=1)

    assert not out_dir.exists()


def test_replay_refuses_candidate_drift_before_final_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir, _ = _make_replay_run(tmp_path)
    out_dir = tmp_path / "candidate_drift_replay"
    original_result_row = quality_replay_module._result_row
    mutated = False

    def result_row_with_candidate_mutation(
        source: dict, request_path: Path, result: dict, source_run: Path,
    ) -> dict:
        nonlocal mutated
        row = original_result_row(
            source, request_path, result, source_run
        )
        if not mutated:
            manifest = json.loads(request_path.read_text(encoding="utf-8"))
            candidate_path = (
                request_path.parent / manifest["candidate"]["path"]
            ).resolve()
            candidate_path.write_text(
                json.dumps({"tampered_after_load": True}), encoding="utf-8"
            )
            mutated = True
        return row

    monkeypatch.setattr(
        quality_replay_module, "_result_row",
        result_row_with_candidate_mutation,
    )

    with pytest.raises(
        OfflineQualityReplayError,
        match="historical replay inputs changed during offline evaluation",
    ):
        replay_evaluation_run(run_dir, out_dir=out_dir, workers=1)

    assert mutated is True
    assert not (out_dir / REPLAY_OUTPUT_NAME).exists()
    assert not (out_dir / REPLAY_CSV_NAME).exists()


def _identity_from_result(row: dict) -> tuple[str, str, str]:
    return row["source_run_id"], row["source_setup_id"], row["variant"]


def test_replay_lists_exact_missing_request_paths_and_rejects_in_run_output(tmp_path: Path) -> None:
    run_dir, _ = _make_replay_run(tmp_path, write_requests=False)
    expected = (
        run_dir / "models" / "yolov7_paper" / "benchmark_results"
        / "quality_inputs" / "orin_nx_hailo8_01" / "request_0" / "full_request.json"
    )
    with pytest.raises(OfflineQualityReplayError) as missing_error:
        replay_evaluation_run(run_dir, out_dir=tmp_path / "outside", full_only=True)
    message = str(missing_error.value)
    assert "original EvaluationRun" in message
    assert str(expected) in message
    assert "compact debug/report packs are insufficient" in message

    complete_run, _ = _make_replay_run(tmp_path / "complete")
    with pytest.raises(OfflineQualityReplayError, match="outside the historical EvaluationRun"):
        replay_evaluation_run(
            complete_run,
            out_dir=complete_run / "quality_management" / "replay",
            workers=1,
            full_only=True,
        )


def test_replay_rejects_unsafe_model_reference_member(tmp_path: Path) -> None:
    run_dir, _ = _make_replay_run(tmp_path)
    summary_path = run_dir / "quality_management" / "central_quality_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["results"][0]["model_id"] = "../../outside"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(
        OfflineQualityReplayError,
        match="model_id is not a safe EvaluationRun member",
    ):
        replay_evaluation_run(
            run_dir,
            out_dir=tmp_path / "outside_replay",
            workers=1,
        )


def test_replay_rejects_tampered_management_reference(tmp_path: Path) -> None:
    run_dir, _ = _make_replay_run(tmp_path)
    reference_path = (
        run_dir / "quality_management" / "references" / "yolov7_paper"
        / "canonical_cpu_reference.json"
    )
    reference_path.write_text('{"records": []}', encoding="utf-8")

    with pytest.raises(
        OfflineQualityReplayError,
        match="management reference SHA-256 mismatch",
    ):
        replay_evaluation_run(
            run_dir,
            out_dir=tmp_path / "tampered_replay",
            workers=1,
        )


def test_replay_requires_historical_request_and_prediction_hash_bindings(
    tmp_path: Path,
) -> None:
    run_dir, _ = _make_replay_run(tmp_path)
    summary_path = run_dir / "quality_management/central_quality_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["results"][0].pop("source_request_sha256")
    summary["results"][1].pop("candidate_predictions_sha256")
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(
        OfflineQualityReplayError,
        match="source request with a valid SHA-256",
    ):
        replay_evaluation_run(
            run_dir, out_dir=tmp_path / "unbound_replay", workers=1,
        )


def test_replay_rejects_internally_rehashed_but_historically_changed_candidate(
    tmp_path: Path,
) -> None:
    run_dir, _ = _make_replay_run(tmp_path)
    summary_path = run_dir / "quality_management/central_quality_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    source = summary["results"][0]
    request_path = run_dir / source["source_request"]
    request = json.loads(request_path.read_text(encoding="utf-8"))
    candidate_path = request_path.parent / request["candidate"]["path"]
    changed = _candidate_payload("full")
    changed["records"][0]["candidate"][0]["x2"] = 4.0
    request["candidate"] = _write_json(candidate_path, changed)
    _write_json(request_path, request)
    source["source_request_sha256"] = (
        "sha256:" + hashlib.sha256(request_path.read_bytes()).hexdigest()
    )
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(
        OfflineQualityReplayError,
        match="prediction/annotation identity mismatch.*candidate_predictions_sha256",
    ):
        replay_evaluation_run(
            run_dir, out_dir=tmp_path / "changed_candidate_replay", workers=1,
        )
