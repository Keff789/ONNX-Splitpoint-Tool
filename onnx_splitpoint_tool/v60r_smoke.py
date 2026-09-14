from __future__ import annotations

"""Hardware-free v60r checks layered on top of the v60q smoke suite."""

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any, Sequence

from .deepx.artifacts import cache_dxnn_artifact, deepx_cache_key, deepx_cached_artifact_compatible
from .workflow.contracts import WorkflowOptions
from .workflow.execution_binding import _remote_args_from_options
from .workflow.runner import (
    _benchmark_stage_status_v60p,
    _benchmark_stage_status_v60r,
    _selected_run_completeness_v60r,
    expected_profile_measurements_v60r,
    missing_profile_measurements_v60r,
    validation_cardinality_mismatches_v60r,
)
from .v60q_smoke import _run_checks as _run_v60q_checks


def _extra_checks() -> list[dict[str, Any]]:
    checks: list[tuple[str, Any]] = []

    def check(name: str):
        def deco(fn):
            checks.append((name, fn))
            return fn
        return deco

    @check("run_mode_validation_budget_is_authoritative")
    def _() -> None:
        options = WorkflowOptions(profile="profile.yaml", out="out")
        options.remote_validation_max_images = 50
        profile = {
            "execution_preset": {"id": "smoke", "snapshot": {}},
            "validation_execution": {"max_items": {"classification": 16, "detection": 12}},
        }
        assert _remote_args_from_options(options, profile, model_task="classification").validation_max_images == 16
        assert _remote_args_from_options(options, profile, model_task="detection").validation_max_images == 12

    @check("deepx_part1_cache_is_task_and_manifest_bound")
    def _() -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            onnx = root / "part1.onnx"; onnx.write_bytes(b"onnx")
            cfg = root / "config.json"; cfg.write_text("{}", encoding="utf-8")
            dxnn = root / "model.dxnn"; dxnn.write_bytes(b"dxnn")
            det = {"task": "detection", "calibration_manifest_identity": "sha256:det", "calibration_count": 500}
            cls = {"task": "classification", "calibration_manifest_identity": "sha256:cls", "calibration_count": 500}
            det_key = deepx_cache_key(onnx_path=onnx, config_path=cfg, target="deepx_m1", variant="part1_b038", cache_contract=det)
            cls_key = deepx_cache_key(onnx_path=onnx, config_path=cfg, target="deepx_m1", variant="part1_b038", cache_contract=cls)
            assert det_key != cls_key
            cache_root = root / "cache"
            cache_dxnn_artifact(dxnn_path=dxnn, cache_root=cache_root, cache_key=det_key, manifest={"cache_contract": det})
            assert deepx_cached_artifact_compatible(cache_dir=cache_root / det_key, expected_contract=det)[0] is True
            assert deepx_cached_artifact_compatible(cache_dir=cache_root / det_key, expected_contract=cls)[0] is False

    @check("runtime_validation_cardinality_contract_is_enforced")
    def _() -> None:
        plan = {"runs": [{"id": "ort_cpu", "validation_items_requested": 12, "validation_max_images": 12, "validation_budget_authoritative": True}]}
        rows = [{"backend": "cpu_ort", "variant": "split", "case_id": "b038", "task_quality_gate": {"n": 50}}]
        mismatch = validation_cardinality_mismatches_v60r(benchmark_plan=plan, normalized_rows=rows)
        assert len(mismatch) == 1 and mismatch[0]["requested_count"] == 12 and mismatch[0]["evaluated_count"] == 50

    @check("required_profile_matrix_detects_missing_full_backend")
    def _() -> None:
        plan = {"runs": [{"id": "hailo8"}, {"id": "hailo8_to_trt"}]}
        bset = {"cases": [{"case_dir": "b038", "boundary": 38}]}
        expected = expected_profile_measurements_v60r(model_id="yolo26s", benchmark_plan=plan, benchmark_set_contract=bset)
        measured = [{"backend": "hailo8_to_tensorrt", "variant": "split", "case_id": "b038"}]
        missing = missing_profile_measurements_v60r(expected, measured)
        assert [(r["backend"], r["variant"]) for r in missing] == [("hailo8", "full")]
        assert _benchmark_stage_status_v60r(normalized_row_count=1, executor_status="ok", executor_metrics={}, required_missing_count=1) == "partial"

    @check("missing_selected_full_baseline_marks_matrix_partial")
    def _() -> None:
        plan = {"runs": [
            {"id": "hailo8", "type": "hailo"},
            {"id": "hailo8_to_trt", "type": "matrix"},
        ]}
        sources = [
            {"path": "/tmp/benchmark_results_hailo8_auto.json", "tag": "hailo8_auto", "row_count": 0},
            {"path": "/tmp/benchmark_results_hailo8_to_trt_auto.json", "tag": "hailo8_to_trt_auto", "row_count": 1},
        ]
        result = _selected_run_completeness_v60r(benchmark_plan=plan, source_records=sources)
        assert result["missing_full_baseline_run_ids"] == ["hailo8"]
        assert _benchmark_stage_status_v60p(
            normalized_row_count=1,
            executor_status="ok",
            executor_metrics={"selected_run_missing_count": 1},
        ) == "partial"

    results: list[dict[str, Any]] = []
    for name, fn in checks:
        try:
            fn()
            results.append({"name": name, "status": "pass"})
        except Exception as exc:
            results.append({"name": name, "status": "fail", "error": f"{type(exc).__name__}: {exc}"})
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="onnx-splitpoint-smoke-v60r", description="Run fast v60r run-budget, DeepX-cache and matrix-completeness checks.")
    parser.add_argument("--json", default="", help="Optional JSON output path.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    results = _run_v60q_checks() + _extra_checks()
    passed = sum(row["status"] == "pass" for row in results)
    failed = len(results) - passed
    payload = {
        "schema": "onnx-splitpoint/v60r-smoke",
        "status": "ok" if failed == 0 else "failed",
        "passed": passed,
        "failed": failed,
        "checks": results,
    }
    if str(args.json or "").strip():
        path = Path(args.json).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for row in results:
        suffix = "" if row["status"] == "pass" else f" — {row.get('error', '')}"
        print(f"[{row['status'].upper()}] {row['name']}{suffix}")
    print(f"v60r smoke: {'ok' if failed == 0 else 'FAILED'} ({passed} passed, {failed} failed)")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
