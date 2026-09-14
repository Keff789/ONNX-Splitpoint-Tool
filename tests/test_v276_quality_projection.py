from __future__ import annotations

import csv
from pathlib import Path

import pytest

from onnx_splitpoint_tool.reporting_quality_decomposition import (
    build_quality_decomposition,
    write_quality_decomposition,
)
from onnx_splitpoint_tool.workflow.scientific_reporting import (
    _task_quality_bound_display,
    _write_tex_table,
    project_central_quality_status,
)


def _central_result(
    *, variant: str, backend: str, case_id: str, top1: float, top5: float,
) -> dict:
    return {
        "model_id": "resnet50",
        "case_id": case_id,
        "variant": variant,
        "backend": backend,
        "source_run_id": backend,
        "setup_id": "orin_nx_hailo8_01",
        "task": "classification",
        "technical_status": "completed",
        "decision": "pass",
        "reference_identity": "reference-resnet50-b500",
        "validation_dataset_sha256": "a" * 64,
        "preprocessing_contract_sha256": "b" * 64,
        "task_quality_policy_sha256": "c" * 64,
        "primary": {
            "metric": "top1_accuracy",
            "candidate": top1,
            "reference": 0.80,
            "delta": top1 - 0.80,
            "margin": 0.01,
            "decision": "pass",
            "bootstrap_skipped_reason": "candidate_reference_identical"
            if top1 == 0.80 else "",
        },
        "guardrails": {
            "top5_accuracy": {
                "metric": "top5_accuracy",
                "candidate": top5,
                "reference": 0.95,
                "delta": top5 - 0.95,
                "margin": 0.01,
                "decision": "pass",
            },
        },
        "metric_gate_config": {
            "primary_metric": "top1_accuracy",
            "guardrails": {"top5_accuracy_margin": 0.01},
        },
    }


def _projected_rows() -> list[dict]:
    source = {
        "status": "ok",
        "request_count": 2,
        "results": [
            _central_result(
                variant="full", backend="hailo8", case_id="full",
                top1=0.78, top5=0.94,
            ),
            _central_result(
                variant="composed", backend="hailo8_to_trt", case_id="b001",
                top1=0.77, top5=0.93,
            ),
        ],
    }
    return list(project_central_quality_status(source)["results"])


def test_paired_central_results_populate_reference_and_loss_surfaces() -> None:
    references, decomposition = build_quality_decomposition(_projected_rows())
    assert len(references) == 2
    assert len(decomposition) == 1
    split_reference = next(
        row for row in references if row["case_id"] == "b001"
    )
    assert split_reference["reference_comparison_source"] == (
        "paired_central_quality_result"
    )
    assert split_reference["reference_Top1"] == 0.80
    assert split_reference["row_Top1"] == 0.77
    assert split_reference["delta_vs_full_onnx_Top1"] == pytest.approx(-0.03)
    loss = decomposition[0]
    assert loss["status"] == "ok"
    assert loss["vendor_loss_Top1"] == pytest.approx(-0.02)
    assert loss["split_extra_loss_Top1"] == pytest.approx(-0.01)
    assert loss["total_split_loss_Top1"] == pytest.approx(-0.03)
    assert loss["vendor_loss_Top5"] == pytest.approx(-0.01)
    assert loss["split_extra_loss_Top5"] == pytest.approx(-0.01)


def test_central_gate_reason_uses_only_source_decision_and_skip_fields() -> None:
    source = _central_result(
        variant="full", backend="tensorrt", case_id="full",
        top1=0.80, top5=0.95,
    )
    row = project_central_quality_status({
        "status": "ok", "request_count": 1, "results": [source],
    })["results"][0]
    reasons = row["task_quality_gate_reasons"]
    assert "primary:top1_accuracy:decision:pass" in reasons
    assert (
        "primary:top1_accuracy:bootstrap_skipped_reason:"
        "candidate_reference_identical"
    ) in reasons
    assert "guardrail:top5_accuracy:decision:pass" in reasons
    assert row["task_quality_gate_reason"] == ";".join(reasons)


def test_quality_decomposition_writer_is_nonempty_for_projected_rows(
    tmp_path: Path,
) -> None:
    result = write_quality_decomposition(tmp_path, _projected_rows())
    assert result == {
        "reference_comparison_count": 2,
        "loss_decomposition_count": 1,
    }
    with (tmp_path / "task_quality_reference_comparison.csv").open(
        newline="", encoding="utf-8",
    ) as handle:
        reference_rows = list(csv.DictReader(handle))
    with (tmp_path / "task_quality_loss_decomposition.csv").open(
        newline="", encoding="utf-8",
    ) as handle:
        loss_rows = list(csv.DictReader(handle))
    assert len(reference_rows) == 2
    assert len(loss_rows) == 1


def test_reference_projection_exposes_primary_decision_and_skip_reason(
    tmp_path: Path,
) -> None:
    source = _central_result(
        variant="full",
        backend="hailo8",
        case_id="full",
        top1=0.60,
        top5=0.95,
    )
    source["decision"] = "fail"
    source["primary"].update({
        "decision": "fail",
        "status": "fail",
        "bootstrap_repetitions_requested": 500,
        "bootstrap_repetitions": 0,
        "bootstrap_skipped_reason": (
            "point_estimate_below_non_inferiority_margin"
        ),
    })
    projected = project_central_quality_status({
        "status": "ok",
        "request_count": 1,
        "results": [source],
    })["results"]

    references, _ = build_quality_decomposition(projected)
    assert len(references) == 1
    reference = references[0]
    assert reference["comparison_status"] == "ok"
    assert reference["quality_decision"] == "fail"
    assert reference["status"] == "fail"
    assert reference["decision_Top1"] == "fail"
    assert reference["bootstrap_skipped_reason_Top1"] == (
        "point_estimate_below_non_inferiority_margin"
    )
    assert reference["decision_Top5"] == "pass"
    assert reference["bootstrap_skipped_reason_Top5"] in {None, ""}

    write_quality_decomposition(tmp_path, projected)
    markdown = (
        tmp_path / "task_quality_reference_comparison.md"
    ).read_text(encoding="utf-8")
    assert "quality_decision" in markdown
    assert "comparison_status" in markdown
    assert "decision_Top1" in markdown
    assert "bootstrap_skipped_reason_Top1" in markdown
    assert (
        "point_estimate_below_non_inferiority_margin" in markdown
    )


def test_fast_fail_bound_is_not_presented_as_computed_lcb(
    tmp_path: Path,
) -> None:
    fast_fail = _task_quality_bound_display({
        "model_id": "resnet50",
        "task_quality_ci_low": -0.12,
        "task_quality_bootstrap_repetitions": 0,
        "task_quality_bootstrap_skipped_reason": (
            "point_estimate_below_non_inferiority_margin"
        ),
    })
    assert fast_fail["task_quality_ci_low"] is None  # v31 read-only legacy projection
    assert fast_fail["task_quality_ci_computed"] is False
    assert fast_fail["task_quality_bound_value"] is None
    assert fast_fail["task_quality_bound_evidence"] == (
        "not_computed:point_estimate_below_non_inferiority_margin"
    )

    computed = _task_quality_bound_display({
        "model_id": "resnet50",
        "task_quality_ci_low": -0.006,
        "task_quality_bootstrap_repetitions": 500,
        "task_quality_bootstrap_skipped_reason": "",
    })
    assert computed["task_quality_ci_computed"] is True
    assert computed["task_quality_bound_value"] == -0.006
    assert computed["task_quality_bound_evidence"] == (
        "computed_95_percent_lcb"
    )

    tex = _write_tex_table(
        tmp_path / "task_quality_gates.tex",
        [fast_fail, computed],
        [
            ("model_id", "Model", "text"),
            ("task_quality_bound_value", "Lower bound", "number"),
            ("task_quality_bound_evidence", "Bound evidence", "text"),
        ],
        "Task-quality gates.",
        "tab:task-quality-gates",
    ).read_text(encoding="utf-8")
    assert "Lower bound" in tex
    assert "not\\_computed:point\\_estimate\\_below" in tex
    assert r"95\textbackslash{}\% LCB" not in tex
