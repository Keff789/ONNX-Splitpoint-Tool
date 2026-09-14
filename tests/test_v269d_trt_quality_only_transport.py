from __future__ import annotations

import json
import subprocess
from pathlib import Path

from onnx_splitpoint_tool.benchmark.remote_run import (
    RemoteBenchmarkArgs,
    _remote_result_collect_script,
)
from onnx_splitpoint_tool.workflow.execution_binding import (
    _copy_remote_result_files,
)
from onnx_splitpoint_tool.workflow.results import (
    _rows_from_json,
    expand_normalized_benchmark_rows,
)
from onnx_splitpoint_tool.workflow.runner import EvaluationWorkflowRunner


def test_quality_only_report_never_becomes_a_performance_row(
    tmp_path: Path,
) -> None:
    report = {
        "schema": "onnx-splitpoint/tensorrt-full-quality-evidence-report",
        "schema_version": 1,
        "quality_evidence_only": True,
        "performance_claims_emitted": False,
        "execution_role": "full_quality_only",
        # Deliberately adversarial timing-shaped content: the explicit quality
        # boundary must win even if a stale producer accidentally writes this.
        "run_cfg": {"quality_evidence_only": True, "provider": "tensorrt"},
        "timings": {"full": {"mean_ms": 1.25}},
        "variant_status": {"full": "ok"},
        "case_id": "full",
    }
    path = tmp_path / "validation_report.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    assert _rows_from_json(path) == []
    assert expand_normalized_benchmark_rows(
        report, model_id="resnet50", source_path=path,
        tag="native_full_tensorrt",
    ) == []


def test_normal_full_owner_still_produces_one_row(tmp_path: Path) -> None:
    row = {
        "case_id": "b001",
        "variant": "full",
        "primary_variant": "full",
        "provider": "tensorrt",
        "full_provider": "tensorrt",
        "full_mean_ms": 2.0,
        "full_baseline_owner": True,
        "full_measurement_requested_in_case": True,
        "runtime_ok": True,
    }
    rows = expand_normalized_benchmark_rows(
        row, model_id="resnet50",
        source_path=tmp_path / "benchmark_results_ort_tensorrt.json",
        tag="ort_tensorrt",
    )
    assert len(rows) == 1
    assert rows[0]["variant"] == "full"


def test_quality_companion_collection_uses_one_physical_setup_prefix(
    tmp_path: Path,
) -> None:
    setup_id = "orin_nx_hailo8_01"
    suite = tmp_path / "suite"
    remote_results = tmp_path / "remote_results"
    request_dir = (
        suite / "native_full_quality" / "quality_inputs" / setup_id
        / "results_native_full_tensorrt" / "task_quality_inputs"
    )
    request_dir.mkdir(parents=True)
    (request_dir / "full_request.json").write_text("{}", encoding="utf-8")
    (request_dir / "full_candidate.json").write_text("{}", encoding="utf-8")

    script = _remote_result_collect_script(
        remote_results_dir=str(remote_results), remote_suite_dir=str(suite),
    )
    subprocess.run(["bash", "-c", script], check=True)
    flattened = (
        remote_results / "results_native_full_tensorrt"
        / "task_quality_inputs" / "full_request.json"
    )
    assert flattened.is_file()

    destination = tmp_path / "management"
    _copy_remote_result_files(
        remote_results, destination, flat_prefix=setup_id,
    )
    copied = (
        destination / "quality_inputs" / setup_id
        / "results_native_full_tensorrt" / "task_quality_inputs"
        / "full_request.json"
    )
    assert copied.is_file()
    case_id, source_run_id = EvaluationWorkflowRunner._quality_request_scope(copied)
    assert case_id == ""
    assert source_run_id == "native_full_tensorrt"


def test_remote_args_expose_complete_quality_identity() -> None:
    args = RemoteBenchmarkArgs(
        quality_evidence_eval_id="eval-34",
        quality_evidence_model_id="resnet50",
        quality_evidence_setup_id="orin_nx_hailo8_01",
        quality_evidence_endpoint_id="tensorrt_at_hailo8_full",
    )
    assert args.quality_evidence_eval_id == "eval-34"
    assert args.quality_evidence_model_id == "resnet50"
    assert args.quality_evidence_setup_id == "orin_nx_hailo8_01"
    assert args.quality_evidence_endpoint_id == "tensorrt_at_hailo8_full"

    source = Path(
        "onnx_splitpoint_tool/benchmark/remote_run.py"
    ).read_text(encoding="utf-8")
    for token in (
        "--quality-evidence-eval-id",
        "--quality-evidence-model-id",
        "--quality-evidence-setup-id",
        "--quality-evidence-endpoint-id",
    ):
        assert token in source
