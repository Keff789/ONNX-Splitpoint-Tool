from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from onnx_splitpoint_tool.native_split_quality_authority import (
    CURRENT_NATIVE_SPLIT_QUALITY_WORKFLOW,
    resolve_native_split_quality_authority,
)


ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(
        f"v27521_reporting_{path.stem}", path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


COORDINATOR = _load_script("run_evalrun_native_producer_variants.py")
FINAL_REPORT = _load_script("native_producer_final_report.py")


MODELS = ("resnet50", "yolo26s", "yolov7_paper")
SETUPS = {
    "hailo8": "orin_nx_hailo8_01",
    "hailo10h": "orin_nx_hailo10_01",
    "deepx": "orin_nx_deepx_m1_01",
}


def _full_row(producer: str, model: str, backend: str) -> dict[str, object]:
    return {
        "execution_mode": "native_full_baseline",
        "backend_key": producer,
        "backend": f"native_full_{backend}",
        "setup_id": SETUPS[producer],
        "comparison_backend": producer,
        "model": model,
        "case": "full",
    }


def _full_matrix_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for producer in SETUPS:
        for model in MODELS:
            rows.append(_full_row(producer, model, producer))
            rows.append(_full_row(producer, model, "tensorrt"))
    return rows


def test_expected_matrix_keeps_18_denominator_and_classifies_six_unstarted_trt_rows() -> None:
    expected = _full_matrix_rows()
    actual = [
        {**row, "ok": True, "status": "ok"}
        for row in expected
        if not (
            row["backend"] == "native_full_tensorrt"
            and row["setup_id"] in {
                SETUPS["hailo8"], SETUPS["hailo10h"],
            }
        )
    ]

    matrix = COORDINATOR._native_performance_expected_matrix(
        expected,
        actual,
        setup_local_tensorrt_errors={
            SETUPS["hailo8"]: "central producer absent",
            SETUPS["hailo10h"]: "central producer absent",
        },
    )

    assert matrix["expected_row_count"] == 18
    assert matrix["present_expected_row_count"] == 12
    assert matrix["successful_expected_row_count"] == 12
    assert matrix["failed_expected_row_count"] == 0
    assert matrix["missing_expected_row_count"] == 6
    assert matrix["execution_success_complete"] is False
    assert all(
        row["failure_reason"]
        == "setup_local_tensorrt_quality_producer_missing"
        and row["execution_attempted"] is False
        and row["transfer_attempted"] is False
        for row in matrix["missing_expected_rows"]
    )


def test_runtime_transfer_failure_is_not_relabelled_as_missing_trt_quality() -> None:
    expected = [_full_row("hailo8", "resnet50", "hailo8")]
    actual = [{
        **expected[0],
        "ok": False,
        "status": "failed",
        "failure_reason": "native_transfer_failed",
        "status_detail": "rsync failed",
    }]

    matrix = COORDINATOR._native_performance_expected_matrix(
        expected,
        actual,
        setup_local_tensorrt_errors={
            SETUPS["hailo8"]: "unrelated TensorRT producer error",
        },
    )

    assert matrix["missing_expected_row_count"] == 0
    assert matrix["failed_expected_row_count"] == 1
    assert matrix["failed_expected_rows"][0]["failure_reason"] == (
        "native_transfer_failed"
    )


def test_evidence_summary_is_partial_for_12_of_18_even_with_stale_complete_alias() -> None:
    rows = [{"ok": True, "execution_mode": "native_full_baseline"}] * 12
    matrix = {
        "expected_row_count": 18,
        "present_expected_row_count": 12,
        "successful_expected_row_count": 12,
        "failed_expected_row_count": 0,
        "missing_expected_row_count": 6,
        "matrix_complete": True,
    }

    summary = FINAL_REPORT._evidence_summary(rows, matrix)

    assert summary["evidence_status"] == "partial"
    assert summary["matrix_complete"] is False
    assert summary["matrix_counts_consistent"] is True
    assert summary["expected_row_count"] == 18
    assert summary["present_expected_row_count"] == 12
    assert summary["failed_or_unsupported_count"] == 6


def test_model_json_and_markdown_share_matrix_bound_partial_status(
    tmp_path: Path,
) -> None:
    rows = [{
        **_full_row("deepx", "resnet50", "deepx"),
        "ok": True,
        "status": "ok",
    }]
    matrix = {
        "expected_rows": [
            _full_row("deepx", "resnet50", "deepx"),
            _full_row("deepx", "resnet50", "tensorrt"),
        ],
        "expected_row_count": 2,
        "present_expected_row_count": 1,
        "successful_expected_row_count": 1,
        "failed_expected_row_count": 0,
        "missing_expected_row_count": 1,
        "present_expected_rows": [rows[0]],
        "successful_expected_rows": [rows[0]],
        "failed_expected_rows": [],
        "missing_expected_rows": [
            _full_row("deepx", "resnet50", "tensorrt")
        ],
    }

    outputs = FINAL_REPORT._write_model_scoped_reports(
        tmp_path, rows, ["backend", "model", "status", "ok"], matrix,
    )
    payload = json.loads(Path(outputs["resnet50"]).read_text(encoding="utf-8"))
    markdown = Path(outputs["resnet50"]).with_suffix(".md").read_text(
        encoding="utf-8",
    )

    assert payload["evidence_status"] == "partial"
    assert payload["present_expected_row_count"] == 1
    assert payload["expected_row_count"] == 2
    assert "Evidence status: **partial**" in markdown
    assert "Matrix coverage: **1 / 2** present" in markdown
    assert "Evidence status: **complete**" not in markdown


def test_full_only_split_materialization_is_valid_without_central_summary(
    tmp_path: Path,
) -> None:
    cfg = {
        "backends": ["hailo8", "hailo10h", "deepx"],
        "split_backends": [],
        "remotes": {
            backend: {"setup_id": setup}
            for backend, setup in SETUPS.items()
        },
    }
    paths, plan = COORDINATOR._materialize_native_split_quality_binding_sets(
        tmp_path / "full-only-run", cfg, [{"id": "default"}],
    )

    assert paths == {"v000_default": {}}
    assert plan["required"] is True
    assert plan["applicable"] is False
    assert plan["status"] == "not_applicable_full_only"
    assert plan["errors_by_variant_setup"] == {}


def test_full_only_theoretical_matrix_contains_full_rows_and_zero_split_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = {
        "backends": ["hailo8"],
        "split_backends": [],
        "remotes": {"hailo8": {"setup_id": SETUPS["hailo8"]}},
        "full_baselines": {
            "enabled": True,
            "backends_by_producer": {
                "hailo8": ["hailo8", "tensorrt"],
            },
        },
    }
    monkeypatch.setattr(
        COORDINATOR,
        "_effective_variant_case_map",
        lambda *_args, **_kwargs: {"resnet50": ["b001"]},
    )

    rows, _setups, _validation, _full = (
        COORDINATOR._variant_expected_energy_rows(
            tmp_path, cfg, [{"id": "default"}],
        )
    )

    assert len(rows) == 2
    assert {row["backend"] for row in rows} == {
        "native_full_hailo8", "native_full_tensorrt",
    }
    assert not any(row["execution_mode"] == "native_split" for row in rows)


def test_full_only_authority_is_required_not_applicable_and_valid(
    tmp_path: Path,
) -> None:
    run = tmp_path / "full-only-authority"
    reports = run / "reports"
    reports.mkdir(parents=True)
    snapshot_sha = "a" * 64
    selection_sha = "b" * 64
    manifest = {
        "schema": "onnx-splitpoint/evaluation-run-manifest",
        "run_id": run.name,
        "workflow_version": CURRENT_NATIVE_SPLIT_QUALITY_WORKFLOW,
        "current_workflow_version": CURRENT_NATIVE_SPLIT_QUALITY_WORKFLOW,
        "execution_sessions": [{
            "workflow_version": CURRENT_NATIVE_SPLIT_QUALITY_WORKFLOW,
        }],
        "profile_start_snapshot": {
            "snapshot_sha256": snapshot_sha,
            "requested_selection": {"snapshot_sha256": selection_sha},
            "resolved_selection": {"snapshot_sha256": selection_sha},
        },
    }
    stage = {
        "schema": "onnx-splitpoint/native-producer-stage",
        "run_id": run.name,
        "workflow_version": CURRENT_NATIVE_SPLIT_QUALITY_WORKFLOW,
        "profile_start_snapshot_sha256": snapshot_sha,
        "profile_selection_snapshot_sha256": selection_sha,
        "native_split_quality_first": {
            "required": True,
            "applicable": False,
            "status": "not_applicable_full_only",
        },
    }
    (run / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (reports / "native_producer_stage.json").write_text(
        json.dumps(stage), encoding="utf-8",
    )

    authority = resolve_native_split_quality_authority(
        run_manifest_path=run / "run_manifest.json",
        stage_path=reports / "native_producer_stage.json",
    )

    assert authority["valid"] is True
    assert authority["native_split_quality_required"] is True
    assert authority["native_split_quality_applicable"] is False
    assert authority["stage_required"] is True
    assert authority["stage_applicable"] is False
    assert authority["errors"] == []
