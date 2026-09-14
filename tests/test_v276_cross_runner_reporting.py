from __future__ import annotations

import json
from pathlib import Path

import pytest

from onnx_splitpoint_tool.workflow.cross_runner_reporting import (
    compute_cross_runner_report,
)
from onnx_splitpoint_tool.ranking_methods import direction_parts
from onnx_splitpoint_tool.workflow.results import normalize_benchmark_row


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _endpoint(model: str) -> tuple[str, str, str]:
    if model == "resnet50":
        return "classification", "classification_logits", "a" * 64
    if model == "yolo26s":
        return "detection", "decoded_nms", "b" * 64
    return "detection", "decoded_nms:comparison", "c" * 64


def _build_bounded_24_fixture(
    generic_root: Path,
    native_root: Path,
) -> list[dict[str, object]]:
    models = {
        "resnet50": ("b001", "b052", "b095"),
        "yolo26s": ("b002", "b005", "b038"),
        "yolov7_paper": ("b002", "b044"),
    }
    backends = {
        "hailo8": (
            "hailo8_to_tensorrt", "hailo8_to_trt",
            "orin_nx_hailo8_01",
        ),
        "hailo10h": (
            "hailo10h_to_tensorrt", "hailo10h_to_trt",
            "orin_nx_hailo10_01",
        ),
        "deepx": (
            "deepx_m1_to_tensorrt", "deepx_to_trt",
            "orin_nx_deepx_m1_01",
        ),
    }
    quality_pass = {
        ("deepx", "yolo26s"),
        ("deepx", "yolov7_paper"),
        ("hailo10h", "yolov7_paper"),
        ("hailo8", "yolov7_paper"),
    }

    generic_rows: list[dict[str, object]] = []
    native_rows: list[dict[str, object]] = []
    validation_rows: list[dict[str, object]] = []
    expected_rows: list[dict[str, object]] = []
    central_results: list[dict[str, object]] = []

    for backend, (generic_direction, native_direction, setup_id) in backends.items():
        for model, cases in models.items():
            task, stage, endpoint_hash = _endpoint(model)
            precision = (
                "uint8_dequant_fp16"
                if backend == "hailo8" and model != "resnet50"
                else "float32_layout_fp16"
            )
            generic_cycles = [float(index + 1) for index in range(len(cases))]
            native_cycles = list(generic_cycles)
            if backend == "hailo8" and model == "resnet50":
                native_cycles = [1.0, 3.0, 2.0]

            for index, case in enumerate(cases):
                passed = (backend, model) in quality_pass
                decision = "pass" if passed else "fail"
                endpoint_contract = {
                    "task": task,
                    "stage": stage,
                    "endpoint_contract_complete": True,
                    "endpoint_contract_hash": endpoint_hash,
                    "output_endpoint_attestation": {
                        "attested": True,
                        "status": "passed",
                        "endpoint_contract_hash": endpoint_hash,
                    },
                }
                relative_request = Path(
                    "models", model, "benchmark_results", "quality_inputs",
                    setup_id, "results", case,
                    f"results_{native_direction}", "task_quality_inputs",
                    "composed_request.json",
                )
                _write_json(generic_root / relative_request, {
                    "variant": "composed",
                    "task": task,
                    "runtime_precision_identity": precision,
                    "endpoint_contract_hash": endpoint_hash,
                    "endpoint_contract": endpoint_contract,
                })
                central_identity = {
                    "identity_valid": True,
                    "identity_errors": [],
                    "model_id": model,
                    "case_id": case,
                    "setup_id": setup_id,
                    "source_run_id": native_direction,
                    "backend": native_direction,
                    "task": task,
                    "stage": stage,
                    "runtime_precision_identity": precision,
                    "endpoint_contract_complete": True,
                    "endpoint_contract_hash": endpoint_hash,
                }
                central_results.append({
                    "model_id": model,
                    "case_id": case,
                    "source_setup_id": setup_id,
                    "source_run_id": native_direction,
                    "variant": "composed",
                    "task": task,
                    "runtime_precision_identity": precision,
                    "endpoint_contract_hash": endpoint_hash,
                    "source_request": relative_request.as_posix(),
                    "request_identity": central_identity,
                })
                generic_row: dict[str, object] = {
                    "model_id": model,
                    "case_id": case,
                    "direction": generic_direction,
                    "backend": generic_direction,
                    "setup_id": setup_id,
                    "runner_regime": "generic",
                    "variant": "split",
                    "cycle_ms": generic_cycles[index],
                    "task_quality_status": decision,
                    "contract_consistent": False,
                    "eligible_for_ranking": False,
                }
                # Mirrors the archived-row enrichment performed before the
                # Cross-runner call: exact central scalars are already present,
                # while this module adds the composed-request attestation.
                if backend == "hailo8" and model == "resnet50" and index == 0:
                    generic_row["quality_request_identities_by_variant"] = {
                        "composed": central_identity,
                    }
                generic_rows.append(generic_row)

                common_native = {
                    "model": model,
                    "case": case,
                    "backend": native_direction,
                    "precision": precision,
                    "setup_id": setup_id,
                    "comparison_backend": backend,
                    **endpoint_contract,
                }
                native_rows.append({
                    **common_native,
                    "ok": True,
                    "status": "ok",
                    "fps_makespan": 1000.0 / native_cycles[index],
                    "output_endpoint_match": True,
                    "comparison_stratum_explicit": True,
                    # The frozen performance projection predates the replayed
                    # exact Quality binding. Validation below is authoritative.
                    "quality_evidence_verified": False,
                    "precision_quality_verified": False,
                    "performance_claim_eligible": False,
                    "repeat_claim_gate_pass": True,
                })
                validation_rows.append({
                    **common_native,
                    "contract_consistent": True,
                    "semantic_ok": True,
                    "claim_ok": True,
                    "status": "claim_ok",
                    "task_quality_status": decision,
                    "central_quality_evidence_verified": True,
                    "precision_quality_binding_verified": True,
                    "task_valid": passed,
                    "accuracy_gate_pass": passed,
                    "eligible_for_ranking": False,
                    "gate_status": "screening_only",
                })
                expected_rows.append({
                    "backend": backend,
                    "backend_key": backend,
                    "model": model,
                    "case": case,
                    "setup_id": setup_id,
                    "execution_mode": "native_split",
                })

    # A measured Generic split outside the frozen Native supplement is not a
    # failed Native counterpart and must not pollute identity exclusions.
    generic_rows.append({
        "model_id": "resnet50",
        "case_id": "b777",
        "direction": "hailo8_to_tensorrt",
        "backend": "hailo8_to_tensorrt",
        "setup_id": "orin_nx_hailo8_01",
        "runner_regime": "generic",
        "variant": "split",
        "cycle_ms": 7.0,
    })

    _write_json(
        generic_root / "quality_management" / "central_quality_summary.json",
        {"results": central_results},
    )
    _write_json(
        native_root / "reports" / "native_expected_matrix.json",
        {"expected_row_count": 24, "expected_rows": expected_rows},
    )
    _write_json(
        native_root / "reports" / "native_producer_combined_summary.json",
        {"rows": native_rows},
    )
    _write_json(
        native_root / "reports" / "native_validation"
        / "native_producer_validation_summary.json",
        {"rows": validation_rows},
    )
    return generic_rows


def _normalize_generic_rows_like_full_evalrun(
    rows: list[dict[str, object]],
    raw_root: Path,
) -> list[dict[str, object]]:
    """Exercise the production normalization path used by a full EvalRun."""

    normalized: list[dict[str, object]] = []
    for index, source in enumerate(rows):
        raw = dict(source)
        direction = str(raw.pop("direction"))
        stage1, stage2 = direction_parts(direction)
        raw.update({
            "primary_variant": "composed",
            "run_id": direction,
            "stage1_provider": stage1,
            "stage2_provider": stage2,
            "total_latency_ms": raw.pop("cycle_ms"),
        })
        raw.pop("backend", None)
        row = normalize_benchmark_row(
            raw,
            model_id=str(raw["model_id"]),
            source_path=(
                raw_root / f"results_{direction}" / f"result_{index}.json"
            ),
            tag=direction,
        )
        # The workflow retains the logical dispatch direction alongside the
        # normalized backend alias.  Keep that exact production join surface.
        row["direction"] = direction
        row["runner_regime"] = "generic"
        row["task_quality_status"] = source.get("task_quality_status")
        normalized.append(row)
    return normalized


def test_bounded_native_intersection_recovers_24_exact_pairs_and_micro_metrics(
    tmp_path: Path,
) -> None:
    generic_root = tmp_path / "generic"
    native_root = tmp_path / "native"
    generic_rows = _build_bounded_24_fixture(generic_root, native_root)

    report = compute_cross_runner_report(
        generic_root,
        generic_rows,
        native_run_dir=native_root,
        minimum_candidates=3,
    )

    assert report["planned_native_intersection_source"] == (
        "native_expected_matrix.expected_rows"
    )
    assert report["planned_native_intersection_count"] == 24
    assert report["planned_native_intersection_paired_count"] == 24
    assert report["planned_native_intersection_complete"] is True
    assert report["generic_inside_planned_intersection_count"] == 24
    assert report["generic_outside_planned_intersection_count"] == 1
    assert report["generic_identity_projection_count"] == 24
    assert report["identity_exclusion_count"] == 0

    assert report["pair_count"] == 24
    assert report["technical_pair_count"] == 24
    assert report["quality_pair_count"] == 9
    assert report["claim_pair_count"] == 0
    assert all(
        pair["generic_identity_projection_source"]
        == "central_quality_request_identity+composed_request"
        for pair in report["pairs"]
    )

    assert len(report["groups"]) == 9
    macro = report["macro"]
    assert macro["technical_group_count"] == 6
    assert macro["technical_subminimum_group_count"] == 3
    assert macro["quality_group_count"] == 1
    assert macro["validated_group_count"] == 0
    assert macro["technical_micro_pairwise_concordant_count"] == 17
    assert macro["technical_micro_pairwise_comparable_count"] == 18
    assert macro["technical_micro_pairwise_concordance"] == pytest.approx(
        17 / 18,
    )
    assert macro["technical_hit_at_1_count"] == 6
    assert macro["technical_hit_at_1_comparable_group_count"] == 6
    assert macro["technical_hit_at_1_fraction"] == 1.0
    assert macro["technical_regret_at_1_mean"] == 0.0
    assert macro["technical_regret_at_1_median"] == 0.0

    by_backend = macro["technical_metrics_by_backend"]
    assert by_backend["hailo8"]["concordant_pair_count"] == 5
    assert by_backend["hailo8"]["comparable_pair_count"] == 6
    assert by_backend["hailo8"]["concordance_fraction"] == pytest.approx(5 / 6)
    for backend in ("hailo10h", "deepx_m1"):
        assert by_backend[backend]["concordant_pair_count"] == 6
        assert by_backend[backend]["comparable_pair_count"] == 6
        assert by_backend[backend]["concordance_fraction"] == 1.0

    # The three YOLOv7 n=2 groups stay visible as diagnostics, but do not
    # inflate the six predeclared n>=3 correlation groups.
    yolo7 = [
        row for row in report["groups"]
        if row["model_id"] == "yolov7_paper"
    ]
    assert len(yolo7) == 3
    assert all(row["technical_candidate_count"] == 2 for row in yolo7)
    assert all(row["technical_pairwise_concordance"] is None for row in yolo7)
    assert all(
        row["technical_diagnostic_pairwise_concordance"] == 1.0
        for row in yolo7
    )


def test_full_run_normalized_metric_aliases_recover_same_24_pairs(
    tmp_path: Path,
) -> None:
    """The full EvalRun path must behave like the archived debug-pack path.

    Normalized benchmark rows retain their explicit metric field names until
    the later scientific-row projection.  The v2.76.1 fixture covered only the
    already-projected ``cycle_ms`` representation and therefore missed the
    production full-run path.
    """

    generic_root = tmp_path / "generic"
    native_root = tmp_path / "native"
    generic_rows = _build_bounded_24_fixture(generic_root, native_root)

    normalized_aliases = (
        "pipeline_cycle_selected_ms",
        "total_latency_ms",
        "split_latency_e2e_ms",
        "throughput_primary_fps",
        "pipeline_fps_selected",
        "heterogeneous_pipeline_fps",
    )
    for index, row in enumerate(generic_rows):
        cycle = row.pop("cycle_ms")
        alias = normalized_aliases[index % len(normalized_aliases)]
        row[alias] = (
            1000.0 / float(cycle)
            if "fps" in alias else cycle
        )
        # This is the historical normalized-row shape which triggered the
        # v2.76.2 full-run failure: a missing source declaration was exposed as
        # False.  Its provenance marker makes clear that this is a compatibility
        # default, not an explicit contradiction of the central request.
        row["endpoint_contract_complete"] = False
        row["endpoint_contract_complete_explicit"] = False

    report = compute_cross_runner_report(
        generic_root,
        generic_rows,
        native_run_dir=native_root,
        minimum_candidates=3,
    )

    assert report["planned_native_intersection_paired_count"] == 24
    assert report["planned_native_intersection_complete"] is True
    assert report["pair_count"] == 24
    assert report["identity_exclusion_count"] == 0
    assert report["macro"]["technical_micro_pairwise_concordance"] == pytest.approx(
        17 / 18,
    )


def test_full_evalrun_normalization_recovers_24_pairs_without_weakening_gate(
    tmp_path: Path,
) -> None:
    generic_root = tmp_path / "generic"
    native_root = tmp_path / "native"
    source_rows = _build_bounded_24_fixture(generic_root, native_root)
    normalized_rows = _normalize_generic_rows_like_full_evalrun(
        source_rows, tmp_path / "raw_results",
    )

    # Every raw fixture row omits the field.  Normalization keeps the legacy
    # False value but records that it was not explicitly supplied.
    assert all(
        row["endpoint_contract_complete"] is False
        and row["endpoint_contract_complete_explicit"] is False
        for row in normalized_rows
    )

    report = compute_cross_runner_report(
        generic_root,
        normalized_rows,
        native_run_dir=native_root,
        minimum_candidates=3,
    )

    assert report["planned_native_intersection_paired_count"] == 24
    assert report["planned_native_intersection_complete"] is True
    assert report["pair_count"] == 24
    assert report["identity_exclusion_count"] == 0
    assert report["technical_pair_count"] == 24
    assert report["quality_pair_count"] == 9
    assert report["legacy_endpoint_contract_default_migration_count"] == 0
    assert report["macro"]["technical_micro_pairwise_concordance"] == pytest.approx(
        17 / 18,
    )


def test_archived_normalized_rows_recover_legacy_false_default_after_exact_projection(
    tmp_path: Path,
) -> None:
    """Replay the actual pre-v2.77 normalized-results representation.

    Historical ``normalized_results.json`` files already materialised a missing
    producer declaration as ``endpoint_contract_complete=False`` and did not yet
    carry ``endpoint_contract_complete_explicit``.  A read-only report replay
    loads those archived rows directly instead of normalising the raw producer
    payload again.  The exact central request may migrate that unprovenanced
    compatibility value, while an explicitly marked contradiction must continue
    to fail closed (covered by the following test).
    """

    generic_root = tmp_path / "generic"
    native_root = tmp_path / "native"
    archived_rows = _build_bounded_24_fixture(generic_root, native_root)
    for row in archived_rows:
        if row.get("case_id") == "b777":
            continue
        row["endpoint_contract_complete"] = False
        row.pop("endpoint_contract_complete_explicit", None)
        row.update({
            "source_path": (
                f"models/{row['model_id']}/benchmark_results/result.json"
            ),
            "source_tag": str(row["direction"]),
            "row_identity": (
                f"{row['model_id']}|{row['case_id']}|{row['direction']}"
            ),
            "dedupe_group_size": 1,
            "dedupe_selected_group_size": 1,
            "ingestion_source_priority": 0,
            "output_endpoint_attestation": {},
        })

    report = compute_cross_runner_report(
        generic_root,
        archived_rows,
        native_run_dir=native_root,
        minimum_candidates=3,
        generic_input_source="normalized_benchmark_results",
    )

    assert report["planned_native_intersection_paired_count"] == 24
    assert report["planned_native_intersection_complete"] is True
    assert report["pair_count"] == 24
    assert report["identity_exclusion_count"] == 0
    assert report["technical_pair_count"] == 24
    assert report["quality_pair_count"] == 9
    assert report["legacy_endpoint_contract_default_migration_count"] == 24
    assert len(report["legacy_endpoint_contract_default_migrations"]) == 24
    assert all(
        item["archived_row_value"] is False
        and item["archived_row_explicitness_marker"] == "absent"
        and item["projected_value"] is True
        and item["projection_source"].startswith("central_quality_request_identity")
        and item["endpoint_contract_hash"]
        for item in report["legacy_endpoint_contract_default_migrations"]
    )
    assert report["macro"]["technical_micro_pairwise_concordance"] == pytest.approx(
        17 / 18,
    )


def test_unprovenanced_rows_outside_normalized_archive_stay_fail_closed(
    tmp_path: Path,
) -> None:
    generic_root = tmp_path / "generic"
    native_root = tmp_path / "native"
    rows = _build_bounded_24_fixture(generic_root, native_root)
    for row in rows:
        if row.get("case_id") == "b777":
            continue
        row["endpoint_contract_complete"] = False
        row.pop("endpoint_contract_complete_explicit", None)

    report = compute_cross_runner_report(
        generic_root,
        rows,
        native_run_dir=native_root,
        minimum_candidates=3,
    )

    assert report["pair_count"] == 0
    assert report["legacy_endpoint_contract_default_migration_count"] == 0
    assert report["identity_exclusion_reason_counts"][
        "variant_identity_conflict"
    ] == 24


def test_archived_default_without_exact_attestation_stays_fail_closed(
    tmp_path: Path,
) -> None:
    generic_root = tmp_path / "generic"
    native_root = tmp_path / "native"
    rows = _build_bounded_24_fixture(generic_root, native_root)
    for request_path in generic_root.rglob("composed_request.json"):
        request = json.loads(request_path.read_text(encoding="utf-8"))
        request["endpoint_contract"].pop(
            "output_endpoint_attestation", None,
        )
        _write_json(request_path, request)
    for row in rows:
        if row.get("case_id") == "b777":
            continue
        row["endpoint_contract_complete"] = False
        row.pop("endpoint_contract_complete_explicit", None)
        row["output_endpoint_attestation"] = {}

    report = compute_cross_runner_report(
        generic_root,
        rows,
        native_run_dir=native_root,
        minimum_candidates=3,
        generic_input_source="normalized_benchmark_results",
    )

    assert report["pair_count"] == 0
    assert report["legacy_endpoint_contract_default_migration_count"] == 0
    assert report["identity_exclusion_reason_counts"][
        "variant_identity_conflict"
    ] == 24


def test_explicit_endpoint_completeness_conflict_still_fails_closed(
    tmp_path: Path,
) -> None:
    generic_root = tmp_path / "generic"
    native_root = tmp_path / "native"
    source_rows = _build_bounded_24_fixture(generic_root, native_root)
    source_rows[0]["endpoint_contract_complete"] = False
    normalized_rows = _normalize_generic_rows_like_full_evalrun(
        source_rows, tmp_path / "raw_results",
    )

    assert normalized_rows[0]["endpoint_contract_complete"] is False
    assert normalized_rows[0]["endpoint_contract_complete_explicit"] is True

    report = compute_cross_runner_report(
        generic_root,
        normalized_rows,
        native_run_dir=native_root,
        minimum_candidates=3,
    )

    assert report["pair_count"] == 23
    assert report["planned_native_intersection_complete"] is False
    assert report["identity_exclusion_reason_counts"][
        "archived_central_request_projection_conflict"
    ] == 1
    assert any(
        "archived_central_projection_endpoint_contract_complete_conflict"
        in error
        for item in report["identity_exclusions"]
        for error in item.get("identity_errors", [])
    )


def test_planned_identity_request_drift_fails_closed(tmp_path: Path) -> None:
    generic_root = tmp_path / "generic"
    native_root = tmp_path / "native"
    generic_rows = _build_bounded_24_fixture(generic_root, native_root)
    target = generic_rows[0]
    target["precision"] = "conflicting_precision"

    report = compute_cross_runner_report(
        generic_root,
        generic_rows,
        native_run_dir=native_root,
        minimum_candidates=3,
    )

    assert report["pair_count"] == 23
    assert report["planned_native_intersection_complete"] is False
    assert report["identity_exclusion_reason_counts"][
        "variant_identity_conflict"
    ] == 1
    assert any(
        "explicit_embedded_runtime_precision_identity_conflict" in error
        for item in report["identity_exclusions"]
        for error in item.get("identity_errors", [])
    )
