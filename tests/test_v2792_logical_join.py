from __future__ import annotations

from onnx_splitpoint_tool.workflow.central_quality_join import (
    join_quality_results_by_request_sha,
)
from onnx_splitpoint_tool.workflow.logical_measurement import (
    annotate_logical_measurements,
    canonical_runtime_precision_identity,
    select_logical_primary_rows,
    summarize_logical_measurements,
)

REQ = "a" * 64
ART = "b" * 64


def _row(**extra):
    row = {
        "model_id": "yolov7_paper", "case_id": "full",
        "run_id": "deepx_m1_full", "source_run_id": "deepx_m1_full",
        "backend": "deepx_m1", "variant": "full", "task": "detection",
        "source_request_sha256": REQ, "dxnn_sha256": ART,
        "precision": "fp16", "source_path": "generic.json",
    }
    row.update(extra)
    return row


def test_direct_setup_and_setup_less_mirror_form_one_logical_measurement() -> None:
    rows = annotate_logical_measurements([
        _row(setup_id="orin_nx_deepx_m1_01", source_path="direct.json", pipeline_fps_selected=4.0),
        _row(setup_id="", source_path="mirror.json"),
    ])
    ids = {row["logical_measurement_id"] for row in rows}
    assert len(ids) == 1
    assert sum(row["logical_measurement_primary"] is True for row in rows) == 1
    mirror = next(row for row in rows if row["representation_role"] == "setup_less_mirror")
    assert mirror["mirror_provenance_verified"] is True
    summary = summarize_logical_measurements(rows)
    assert summary["logical_measurement_count"] == 1
    assert summary["mirror_representation_count"] == 1


def test_distinct_real_setups_are_never_deduplicated() -> None:
    rows = annotate_logical_measurements([
        _row(setup_id="setup_a", source_path="a.json"),
        _row(setup_id="setup_b", source_path="b.json"),
    ])
    assert len({row["logical_measurement_id"] for row in rows}) == 2


def test_exact_request_sha_join_prefers_direct_setup() -> None:
    rows = annotate_logical_measurements([
        _row(setup_id="orin_nx_deepx_m1_01", source_path="direct.json"),
        _row(setup_id="", source_path="mirror.json"),
    ])
    joined = join_quality_results_by_request_sha(
        rows=rows,
        results=[{
            "model_id": "yolov7_paper", "case_id": "full",
            "source_run_id": "deepx_m1_full", "variant": "full",
            "setup_id": "orin_nx_deepx_m1_01",
            "source_request_sha256": REQ,
        }],
    )
    assert joined["matched_primary_count"] == 1
    matched = rows[joined["joins"][0]["row_index"]]
    assert matched["setup_id"] == "orin_nx_deepx_m1_01"
    assert joined["unmatched_count"] == 0
    assert joined["ambiguous_count"] == 0


def test_request_sha_join_without_result_setup_still_excludes_mirror_metadata() -> None:
    rows = annotate_logical_measurements([
        _row(setup_id="orin_nx_deepx_m1_01", source_path="direct.json"),
        _row(setup_id="", source_path="mirror.json"),
    ])
    joined = join_quality_results_by_request_sha(
        rows=rows,
        results=[{
            "model_id": "yolov7_paper", "case_id": "full",
            "source_run_id": "deepx_m1_full", "variant": "full",
            "source_request_sha256": REQ,
        }],
    )
    assert joined["matched_primary_count"] == 1
    assert joined["unmatched_count"] == 0
    assert joined["ambiguous_count"] == 0
    assert rows[joined["joins"][0]["row_index"]]["setup_id"]


def test_deepx_precision_alias_only_canonicalizes_for_same_artifact() -> None:
    same = annotate_logical_measurements([
        _row(precision="fp16", dxnn_sha256=ART),
        _row(precision="float32_layout_fp16", dxnn_sha256=ART),
    ])
    assert all(
        row["runtime_precision_identity_canonical"]
        == f"deepx_dxnn_sha256:{ART}"
        for row in same
    )
    assert all(
        row["execution_precision_declared_candidates"]
        == ["float32_layout_fp16"]
        for row in same
    )
    assert not any(row.get("deepx_precision_identity_conflict") for row in same)

    different = annotate_logical_measurements([
        _row(precision="fp16", dxnn_sha256="c" * 64),
        _row(precision="int8", dxnn_sha256="d" * 64),
    ])
    assert {
        row["runtime_precision_identity"] for row in different
    } == {
        "deepx_dxnn_sha256:" + "c" * 64,
        "deepx_dxnn_sha256:" + "d" * 64,
    }
    assert not any(
        row.get("deepx_precision_identity_conflict") for row in different
    )


def test_deepx_precision_never_uses_shared_trt_engine_as_dxnn_proof() -> None:
    rows = annotate_logical_measurements([
        _row(
            precision="fp16", dxnn_sha256="c" * 64,
            part2_engine_sha256="e" * 64,
        ),
        _row(
            precision="int8", dxnn_sha256="d" * 64,
            part2_engine_sha256="e" * 64,
        ),
    ])
    assert {
        row["runtime_precision_identity"] for row in rows
    } == {
        "deepx_dxnn_sha256:" + "c" * 64,
        "deepx_dxnn_sha256:" + "d" * 64,
    }
    assert all(
        row.get("runtime_artifact_sha256") != "e" * 64 for row in rows
    )


def test_duplicate_looking_rows_without_exact_proof_remain_separate() -> None:
    rows = annotate_logical_measurements([
        _row(
            setup_id="orin_nx_deepx_m1_01", source_request_sha256="",
            dxnn_sha256="", source_path="one.json",
        ),
        _row(
            setup_id="orin_nx_deepx_m1_01", source_request_sha256="",
            dxnn_sha256="", source_path="two.json",
        ),
    ])
    assert len({row["logical_measurement_id"] for row in rows}) == 2
    assert not any(row.get("mirror_provenance_verified") for row in rows)


def test_exact_request_sha_join_rejects_conflicting_backend() -> None:
    rows = annotate_logical_measurements([
        _row(setup_id="orin_nx_deepx_m1_01", backend="deepx_m1"),
    ])
    joined = join_quality_results_by_request_sha(
        rows=rows,
        results=[{
            "model_id": "yolov7_paper", "case_id": "full",
            "source_run_id": "deepx_m1_full", "variant": "full",
            "setup_id": "orin_nx_deepx_m1_01",
            "backend": "hailo8",
            "source_request_sha256": REQ,
        }],
    )
    assert joined["matched_primary_count"] == 0
    assert joined["unmatched_count"] == 1


def test_companion_results_never_fill_primary_matrix() -> None:
    rows = annotate_logical_measurements([
        _row(setup_id="orin_nx_deepx_m1_01"),
    ])
    joined = join_quality_results_by_request_sha(
        rows=rows,
        results=[{
            "model_id": "yolov7_paper", "case_id": "full",
            "source_run_id": "native_full_tensorrt", "variant": "full",
            "result_class": "summary_only",
            "source_request_sha256": REQ,
        }],
    )
    assert joined["matched_primary_count"] == 0
    assert joined["companion_count"] == 1
    assert joined["unmatched_count"] == 0


def test_deepx_string_and_structured_contract_are_one_identity() -> None:
    structured = {
        "schema": "onnx-splitpoint/deepx-runtime-precision-contract",
        "artifact_kind": "dxnn",
        "artifact_sha256": ART,
        "precision_semantics": "opaque_vendor_compiled_artifact_identity",
    }
    rows = annotate_logical_measurements([
        _row(
            runtime_precision_identity=f"deepx_dxnn_sha256:{ART}",
            producer_identity={"runtime_precision_identity": structured},
            frozen_identity_evidence={
                "values": {"runtime_precision_identity": [structured]},
                "invalid": {},
            },
        ),
    ])
    assert rows[0]["runtime_precision_identity"] == (
        f"deepx_dxnn_sha256:{ART}"
    )
    assert rows[0]["runtime_artifact_sha256"] == ART
    assert rows[0]["deepx_precision_identity_conflict"] is False
    assert rows[0]["frozen_identity_evidence"]["values"][
        "runtime_precision_identity"
    ] == [f"deepx_dxnn_sha256:{ART}"]


def test_deepx_precision_contract_invalid_sha_fails_closed() -> None:
    canonical, error = canonical_runtime_precision_identity({
        "schema": "onnx-splitpoint/deepx-runtime-precision-contract",
        "artifact_kind": "dxnn",
        "artifact_sha256": "invalid",
    })
    assert canonical == ""
    assert error == "deepx_precision_artifact_sha256_invalid"


def test_same_artifact_with_conflicting_artifact_identity_fails_closed() -> None:
    rows = annotate_logical_measurements([
        _row(
            dxnn_sha256=ART,
            runtime_precision_identity=(
                "deepx_dxnn_sha256:" + "c" * 64
            ),
        ),
    ])
    assert rows[0]["deepx_precision_identity_conflict"] is True
    assert rows[0]["runtime_precision_identity"] == ""


def test_verified_mirror_with_different_dxnn_artifact_fails_closed() -> None:
    rows = annotate_logical_measurements([
        _row(
            setup_id="orin_nx_deepx_m1_01", dxnn_sha256="c" * 64,
            source_path="direct.json",
        ),
        _row(
            setup_id="", dxnn_sha256="d" * 64,
            source_path="mirror.json",
        ),
    ])
    assert len({row["logical_measurement_id"] for row in rows}) == 1
    assert all(row["deepx_precision_identity_conflict"] for row in rows)
    assert all(row["runtime_precision_identity"] == "" for row in rows)
    assert all(
        "deepx_logical_measurement_artifact_conflict"
        in row["deepx_precision_identity_resolution_errors"]
        for row in rows
    )


def test_same_dxnn_with_conflicting_numeric_declarations_fails_closed() -> None:
    rows = annotate_logical_measurements([
        _row(
            setup_id="orin_nx_deepx_m1_01", precision="fp16",
            dxnn_sha256=ART, source_path="direct.json",
        ),
        _row(
            setup_id="", precision="int8", dxnn_sha256=ART,
            source_path="mirror.json",
        ),
    ])
    assert all(row["deepx_precision_identity_conflict"] for row in rows)
    assert all(row["runtime_precision_identity"] == "" for row in rows)
    assert all(
        "deepx_numeric_precision_declaration_conflict"
        in row["deepx_precision_identity_resolution_errors"]
        for row in rows
    )


def test_two_direct_setup_rows_remain_two_logical_primaries() -> None:
    rows = annotate_logical_measurements([
        _row(setup_id="orin_nx_deepx_m1_01", source_path="one.json"),
        _row(setup_id="orin_nx_deepx_m1_01", source_path="two.json"),
    ])
    primaries, errors = select_logical_primary_rows(rows)
    assert errors == []
    assert len(primaries) == 2
    assert len({row["logical_measurement_id"] for row in rows}) == 2
