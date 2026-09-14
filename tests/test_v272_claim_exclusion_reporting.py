from __future__ import annotations

import csv
import json
import zipfile
from pathlib import Path
from typing import Any

import onnx_splitpoint_tool.workflow.scientific_reporting as reporting
from onnx_splitpoint_tool.workflow.analysis_pack import (
    CANONICAL_RESULT_FILES,
    create_analysis_pack,
)


def _performance(
    *,
    model: str,
    backend: str,
    setup: str,
    case: str,
    reasons: list[str] | None = None,
    reason: str = "",
) -> dict[str, Any]:
    row = {
        "row_role": "performance_observation",
        "row_status": "available",
        "model_id": model,
        "backend": backend,
        "setup_id": setup,
        "case_id": case,
        "variant": "split",
        "runtime_executable": True,
        "throughput_fps": 100.0,
        "performance_eligible": True,
        "performance_claim_eligible": False,
        "eligibility_status": "screening_only",
    }
    if reasons is not None:
        row["performance_claim_exclusion_reasons"] = reasons
    if reason:
        row["exclusion_reason"] = reason
    return row


def _energy(
    *,
    model: str,
    backend: str,
    setup: str,
    case: str,
    reasons: list[str],
) -> dict[str, Any]:
    return {
        "row_role": "native_energy_measurement",
        "row_status": "available",
        "model_id": model,
        "backend": backend,
        "setup_id": setup,
        "case_id": case,
        "measurement_ok": True,
        "energy_per_work_j": 0.1,
        "energy_eligible": True,
        "energy_claim_eligible": False,
        "scientific_claim_exclusion_reasons": reasons,
        "eligibility_status": "screening_only",
    }


def _fieldnames(path: Path) -> list[str]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle).fieldnames or [])


def _payload(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "created_at": "2026-07-29T00:00:00+02:00",
        "profile_id": "v272_exclusion_regression",
        "rows": rows,
        "summary": {},
        "native_energy_observations": [],
    }


def test_v272_claim_exclusion_breakdown_is_four_dimensional_and_stable(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        reporting, "_make_figures", lambda *_args, **_kwargs: [],
    )
    rows = [
        _performance(
            model="z_model",
            backend="backend_2",
            setup="setup_2",
            case="case_z",
            reasons=["reason_b", "reason_a"],
        ),
        _energy(
            model="energy_model",
            backend="backend_3",
            setup="setup_3",
            case="case_e1",
            reasons=["energy_b", "energy_a"],
        ),
        _performance(
            model="a_model",
            backend="backend_1",
            setup="setup_1",
            case="case_a2",
            reason="reason_c",
        ),
        _energy(
            model="energy_model",
            backend="backend_3",
            setup="setup_3",
            case="case_e2",
            reasons=["energy_a"],
        ),
        _performance(
            model="a_model",
            backend="backend_1",
            setup="setup_1",
            case="case_a1",
            reason="reason_c",
        ),
    ]
    first_run = tmp_path / "first_run"
    first = first_run / "reports" / "scientific"
    second = tmp_path / "second"

    reporting._write_reports(first, _payload(rows))
    reporting._write_reports(second, _payload(list(reversed(rows))))
    (first_run / "run_manifest.json").write_text(
        json.dumps(
            {
                "schema": "onnx-splitpoint/evaluation-run-manifest",
                "schema_version": 1,
                "run_id": first_run.name,
                "status": "ok",
                "tool_version": "2.72.0",
                "workflow_version": "v2.72.0",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    stable_artifacts = (
        "claim_exclusion_summary.json",
        "claim_exclusion_details.csv",
        "claim_exclusion_grouped_counts.csv",
        "claim_exclusion_dimension_counts.csv",
        "claim_exclusion_summary.md",
    )
    for name in stable_artifacts:
        assert (first / name).read_bytes() == (second / name).read_bytes()

    summary = json.loads(
        (first / "claim_exclusion_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["performance_excluded_count"] == 3
    assert summary["energy_excluded_count"] == 2
    assert summary["excluded_row_count"] == 5
    assert summary["exclusion_reason_assignment_count"] == 7
    assert summary["group_dimensions"] == [
        "model_id", "backend", "setup_id", "reason",
    ]
    assert summary["dimension_order"] == [
        "model", "backend", "setup", "reason",
    ]

    details = summary["exclusion_details"]
    detail_sort_key = lambda row: (
        0 if row["claim_kind"] == "performance" else 1,
        row["model_id"],
        row["backend"],
        row["setup_id"],
        row["reason"],
        row["task"],
        row["case_id"],
        row["variant"],
        row["row_status"],
        row["eligibility_status"],
    )
    assert details == sorted(details, key=detail_sort_key)
    assert all(
        {"model_id", "backend", "setup_id", "reason"}.issubset(row)
        for row in details
    )

    grouped = summary["grouped_exclusion_counts"]
    assert {
        (
            row["claim_kind"],
            row["model_id"],
            row["backend"],
            row["setup_id"],
            row["reason"],
        ): row["count"]
        for row in grouped
    } == {
        (
            "performance", "a_model", "backend_1", "setup_1",
            "reason_c",
        ): 2,
        (
            "performance", "z_model", "backend_2", "setup_2",
            "reason_a",
        ): 1,
        (
            "performance", "z_model", "backend_2", "setup_2",
            "reason_b",
        ): 1,
        (
            "energy", "energy_model", "backend_3", "setup_3",
            "energy_a",
        ): 2,
        (
            "energy", "energy_model", "backend_3", "setup_3",
            "energy_b",
        ): 1,
    }

    dimensions = summary["dimension_exclusion_counts"]
    assert [
        (row["claim_kind"], row["dimension"], row["value"], row["count"])
        for row in dimensions
    ] == [
        ("performance", "model", "a_model", 2),
        ("performance", "model", "z_model", 2),
        ("performance", "backend", "backend_1", 2),
        ("performance", "backend", "backend_2", 2),
        ("performance", "setup", "setup_1", 2),
        ("performance", "setup", "setup_2", 2),
        ("performance", "reason", "reason_a", 1),
        ("performance", "reason", "reason_b", 1),
        ("performance", "reason", "reason_c", 2),
        ("energy", "model", "energy_model", 3),
        ("energy", "backend", "backend_3", 3),
        ("energy", "setup", "setup_3", 3),
        ("energy", "reason", "energy_a", 2),
        ("energy", "reason", "energy_b", 1),
    ]

    assert _fieldnames(first / "claim_eligible_performance.csv") == list(
        reporting.CLAIM_PERFORMANCE_CSV_FIELDS
    )
    assert _fieldnames(first / "claim_eligible_energy.csv") == list(
        reporting.CLAIM_ENERGY_CSV_FIELDS
    )
    assert _fieldnames(first / "claim_exclusion_details.csv") == list(
        reporting.CLAIM_EXCLUSION_DETAIL_CSV_FIELDS
    )
    assert _fieldnames(
        first / "claim_exclusion_grouped_counts.csv"
    ) == list(reporting.CLAIM_EXCLUSION_GROUP_CSV_FIELDS)
    assert _fieldnames(
        first / "claim_exclusion_dimension_counts.csv"
    ) == list(reporting.CLAIM_EXCLUSION_DIMENSION_CSV_FIELDS)

    canonical = json.loads(
        (first / "scientific_report.json").read_text(encoding="utf-8")
    )
    assert canonical["claim_exclusion_summary"] == summary
    markdown = (first / "scientific_report.md").read_text(
        encoding="utf-8"
    )
    assert "## Claim exclusion breakdown" in markdown
    assert "Grouped by model / backend / setup / reason" in markdown
    assert "energy_model" in markdown
    assert "energy_a" in markdown

    assert all(name in CANONICAL_RESULT_FILES for name in stable_artifacts)
    pack_path = tmp_path / "analysis_pack.zip"
    pack_result = create_analysis_pack(
        first_run,
        pack_path,
        tool_version="2.72.0",
        materialize_missing_report=True,
    )
    assert Path(pack_result["zip_path"]) == pack_path
    with zipfile.ZipFile(pack_path) as archive:
        for name in stable_artifacts:
            assert archive.read(f"01_results/{name}") == (
                first / name
            ).read_bytes()

        packed_summary = json.loads(
            archive.read(
                "01_results/claim_exclusion_summary.json"
            ).decode("utf-8")
        )
        assert packed_summary == summary
        assert {
            key: packed_summary[key]
            for key in (
                "performance_excluded_count",
                "energy_excluded_count",
                "excluded_row_count",
                "exclusion_reason_assignment_count",
            )
        } == {
            "performance_excluded_count": 3,
            "energy_excluded_count": 2,
            "excluded_row_count": 5,
            "exclusion_reason_assignment_count": 7,
        }

        assert archive.read(
            "00_overview/scientific_report.json"
        ) == (first / "scientific_report.json").read_bytes()
        packed_canonical = json.loads(
            archive.read(
                "00_overview/scientific_report.json"
            ).decode("utf-8")
        )
        assert packed_canonical["claim_exclusion_summary"] == summary
