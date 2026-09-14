from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from onnx_splitpoint_tool import __version__
from onnx_splitpoint_tool.workflow.native_transfer import (
    build_native_transfer_inventory,
    build_native_validation_image_map,
    classify_native_transfer_failure,
    parse_df_available_bytes,
    required_remote_bytes,
    safe_native_remote_root,
    write_rsync_files_from,
)
from onnx_splitpoint_tool.workflow.runner import (
    WORKFLOW_VERSION,
    _native_concise_summary_v60w,
    _native_expected_matrix_status_v60y,
)


def test_v60y_version_and_workflow() -> None:
    assert __version__ in {"0.14.25+v60z.nativefullquality", "0.14.26+v61a.nativefullenergyprogress", "0.14.27+v61b.nativeintegrationfix", "0.14.28+v61c.nativefullpairedenergyfix", "0.14.29+v61d.nativefullsemanticfix", "0.14.30+v61e.standardguifix", "2.61.0+v61e", "2.62.0", "2.63.0", "2.64.0", "2.65.0", "2.66.0", "2.67.0", "2.68.0", "2.69.6", "2.70.6", "2.70.7", "2.70.8", "2.70.9", "2.70.10", "2.71.1", "2.71.2", "2.71.3", "2.71.4", "2.72.0", "2.72.1", "2.72.2", "2.72.3", "2.72.4", "2.72.5", "2.72.6", "2.72.7", "2.73.0", "2.73.6", "2.73.7", "2.75.2", "2.75.3", "2.75.7", "2.75.8", "2.75.9", "2.75.10", "2.75.11", "2.75.12", "2.75.13", "2.75.14", "2.75.15", "2.75.16", "2.75.17", "2.75.20", "2.75.21", "2.75.22", "2.75.24", "2.75.26", "2.75.27", "2.75.28", "2.75.30", "2.75.31", "2.75.32", "2.75.38", "2.75.39", "2.75.40", "2.75.41", "2.75.42", "2.75.46", "2.75.47"}
    assert WORKFLOW_VERSION in {"v60z-native-full-energy-quality-evidence", "v61a-native-full-energy-progress-fixes", "v61b-native-integration-live-energy-fixes", "v61c-native-full-paired-energy-fixes", "v61d-native-full-semantic-hailo8-fixes", "v61e-standard-run-gui-diagnostics-fixes", "v2.61e-campaign-contract-hardening", "v2.62-window-validation-native-binding", "v2.63-campaign-ready", "v2.64-campaign-ready", "v2.65-campaign-ready", "v2.66-campaign-ready", "v2.67-campaign-ready", "v2.68-level-playing-field", "v2.69f-hardware-smoke-native-energy-repair", "v2.70f-legacy-log-callback-repair", "v2.70g-native-validation-bridge-repair", "v2.70h-remote-suite-bootstrap-repair", "v2.70i-native-full-evidence-repair", "v2.70j-native-evidence-reporting-repair", "v2.71.1-live-evidence-binding-repair", "v2.71.2-evidence-status-energy-resume-repair", "v2.71.3-resume-artifact-restage-repair", "v2.71.4-resume-cleanup-root-rehydration", "v2.72.0-completed-detection-evidence-contracts", "v2.72.1-mixed-runtime-energy-contract-repair", "v2.72.2-offline-energy-completed-consumer-repair", "v2.72.3-campaign-gate-quality-binding-repair", "v2.72.4-campaign-gate-quality-contract-mirror-repair", "v2.72.5-campaign-gate-hailo8-resume-contract-repair", "v2.72.6-campaign-gate-hailo8-preflight-artifact-rehydration-repair", "v2.72.7-native-energy-runtime-success-admission-repair", "v2.73.0-completed-quality-evidence-repair", "v2.73.6-single-writer-process-tree-cancellation", "v2.73.7-atomic-stage-energy-checkpoints", "v2.75.2-native-runtime-closure-partial-energy-repair", "v2.75.3-cache-verify-only-gate", "v2.75.7-cache-canary-remote-runtime-closure-repair", "v2.75.8-native-runtime-holdout-contract-repair", "v2.75.9-smoke-cache-reuse-repair", "v2.75.10-historical-cache-multihit-replay-repair", "v2.75.11-canonical-native-result-verification-repair", "v2.75.12-canonical-native-result-status-alias-repair", "v2.75.13-native-full-contract-line-repair", "v2.75.14-vendor-full-nine-row-repair", "v2.75.15-vendor-full-field-contract-repair", "v2.75.16-read-only-run-storage-admission", "v2.75.17-vendor-quality-diagnostic-hailo-alias-repair", "v2.75.20-remote-boundary-plan-finalization-repair", "v2.75.21-setup-local-tensorrt-quality-dispatch-repair", "v2.75.22-quality-replay-reporting-repair", "v2.75.24-standard-hef-preupload-gate", "v2.75.26-trt-quality-join-mean-iou-repair", "v2.75.27-final-claim-scope-cache-key-repair", "v2.75.28-final-campaign-bootstrap-energy-provenance-repair", "v2.75.30-score-independent-native-ranking-audit", "v2.75.31-compact-debug-ranking-gui-repair", "v2.75.32-audit-ranking-e2e-repair", "v2.75.38-deepx-full-quality-dxnn-dispatch-repair", "v2.75.39-pre-mutation-remote-failure-quarantine-repair", "v2.75.40-deepx-preprocessing-ab-and-full-only-quality-export-repair", "v2.75.41-deepx-calibration-500-vs-1000-canary", "v2.75.42-real-evidence-and-quality-companion-repair", "v2.75.46-native-full-onnx-attestation-standard-quality-projection", "v2.75.47-yolov7-decoder-contract-debug-pack-audit-intent"}


def test_native_transfer_inventory_excludes_generic_bulk(tmp_path: Path) -> None:
    root = tmp_path / "suite"
    keep = [
        root / "benchmark_set.json",
        root / "benchmark_plan.json",
        root / "models" / "model.onnx",
        root / "b001" / "part1.onnx",
        root / "b001" / "hailo" / "hailo10" / "part1" / "compiled.hef",
        root / "b001" / "deepx" / "deepx_m1" / "part1" / "model.dxnn",
        root / "native_trt" / "b001" / "part2" / "fp16" / "part2_fp16.engine",
        root / "resources" / "validation" / "classification" / "subset" / "images" / "n1" / "x.JPEG",
    ]
    drop = [
        root / "dist" / "suite_bundle.tar.gz",
        root / "scientific_report" / "report.json",
        root / "b001" / "results_hailo8" / "validation_report.json",
        root / "b001" / "native_pipeline" / "stale.json",
        root / "activation_calibration" / "proxy.npy",
        root / "b001" / "hailo" / "hailo10" / "part1" / "quantized.har",
        root / "benchmark_results_hailo8.json",
    ]
    for p in keep + drop:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x" * (17 if p in keep else 31))
    inv = build_native_transfer_inventory(root)
    paths = set(inv["relative_paths"])
    for p in keep:
        assert p.relative_to(root).as_posix() in paths
    for p in drop:
        assert p.relative_to(root).as_posix() not in paths
    assert inv["file_count"] == len(keep)
    assert inv["excluded_file_count"] == len(drop)
    files_from = write_rsync_files_from(inv, tmp_path / "files.txt")
    assert files_from.read_text().splitlines() == inv["relative_paths"]


def test_remote_storage_helpers() -> None:
    out = "Filesystem 1024-blocks Used Available Capacity Mounted on\n/dev/x 1000 400 600 40% /home\n"
    assert parse_df_available_bytes(out) == 600 * 1024
    assert classify_native_transfer_failure("No space left on device (28)") == "remote_disk_insufficient"
    assert classify_native_transfer_failure("rsync error") == "remote_rsync_failed"
    assert required_remote_bytes(1000) > 1000
    assert safe_native_remote_root("/home/nx/native_fifo_evalsets/run_20260716_123456", "run_20260716_123456")
    assert not safe_native_remote_root("/home/nx/other/run_20260716_123456", "run_20260716_123456")


def _write_validation_report(path: Path, image: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"run_cfg": {"image": image}}), encoding="utf-8")


def test_validation_image_map_prefers_generic_reference_and_falls_back(tmp_path: Path) -> None:
    run = tmp_path / "run"
    bs = run / "models" / "resnet50" / "benchmark_set" / "legacy_suite"
    image = bs / "resources" / "validation" / "classification" / "subset" / "images" / "n1" / "exact.JPEG"
    image.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(b"jpg")
    _write_validation_report(
        run / "models" / "resnet50" / "benchmark_results" / "remote_diagnostics" / "case_reports" / "results" / "b052" / "results_ort_cpu" / "validation_report.json",
        "/remote/suite/resources/validation/classification/subset/images/n1/exact.JPEG",
    )
    mapping, sources = build_native_validation_image_map(
        run, ["resnet50"], {"resnet50": ["b052"]}, {"resnet50": bs}
    )
    assert mapping == {"resnet50": {"b052": "exact.JPEG"}}
    assert sources["resnet50"]["b052"].startswith("generic_ort_validation:")

    ybs = run / "models" / "yolo26s" / "benchmark_set" / "legacy_suite"
    yimg = ybs / "resources" / "validation" / "detection" / "subset" / "0001.jpg"
    yimg.parent.mkdir(parents=True, exist_ok=True)
    yimg.write_bytes(b"jpg")
    mapping, sources = build_native_validation_image_map(
        run, ["yolo26s"], {"yolo26s": ["b038"]}, {"yolo26s": ybs}
    )
    assert mapping["yolo26s"]["b038"].endswith("0001.jpg")
    assert sources["yolo26s"]["b038"] == "materialised_validation_subset:first_sorted"


def test_expected_matrix_marks_whole_missing_backend() -> None:
    expected = [
        {"backend_key": "deepx", "backend": "deepx_to_trt", "model": "resnet50", "case": "b052", "precision": "fp16"},
        {"backend_key": "hailo10h", "backend": "hailo10h_to_trt", "model": "resnet50", "case": "b052", "precision": "float32_layout_fp16"},
        {"backend_key": "hailo10h", "backend": "hailo10h_to_trt", "model": "yolo26s", "case": "b038", "precision": "float32_layout_fp16"},
    ]
    actual = [{"backend": "deepx_to_trt", "model": "resnet50", "case": "b052", "precision": "fp16", "ok": True}]
    backends = [
        {"backend": "deepx", "ok": True},
        {"backend": "hailo10h", "ok": False, "failure_reason": "remote_disk_insufficient", "error": "No space left on device"},
    ]
    result = _native_expected_matrix_status_v60y(expected, actual, backends)
    assert result["expected_row_count"] == 3
    assert result["present_expected_row_count"] == 1
    assert result["missing_expected_row_count"] == 2
    assert not result["matrix_complete"]
    assert {r["failure_reason"] for r in result["missing_expected_rows"]} == {"remote_disk_insufficient"}


def test_concise_summary_includes_missing_expected_rows(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "native_producer_combined_summary.json").write_text(json.dumps({
        "rows": [{"backend": "deepx_to_trt", "model": "resnet50", "case": "b052", "precision": "fp16", "ok": True, "fps_makespan": 100.0}]
    }), encoding="utf-8")
    paths, rows = _native_concise_summary_v60w(reports, missing_expected_rows=[{
        "backend": "hailo10h_to_trt", "backend_key": "hailo10h", "model": "resnet50", "case": "b052",
        "precision": "float32_layout_fp16", "failure_reason": "remote_disk_insufficient",
    }])
    assert len(rows) == 2
    missing = next(r for r in rows if r["backend"] == "hailo10h_to_trt")
    assert missing["runtime_status"] == "missing"
    assert missing["failure_reason"] == "remote_disk_insufficient"
    assert paths["native_stage_concise_summary_json"].is_file()


def _load_script_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_e2e_wrapper_uses_exact_image_and_clears_success_error(tmp_path: Path, monkeypatch) -> None:
    source = Path(__file__).resolve().parents[1] / "scripts" / "native_producer_e2e_eval_runner.py"
    mod = _load_script_module("native_producer_e2e_eval_runner_v60y_test", source)
    root = tmp_path / "root"
    bs = root / "resnet50" / "benchmark_set" / "legacy_suite"
    (bs / "benchmark_set.json").parent.mkdir(parents=True, exist_ok=True)
    (bs / "benchmark_set.json").write_text(json.dumps({"task": "classification"}), encoding="utf-8")
    (bs / "b052" / "deepx" / "deepx_m1" / "part1").mkdir(parents=True)
    (bs / "b052" / "deepx" / "deepx_m1" / "part1" / "model.dxnn").write_bytes(b"dxnn")
    engine = bs / "native_trt/b052/part2/fp16/part2_fp16.engine"
    engine.parent.mkdir(parents=True)
    engine.write_bytes(b"engine")
    img = bs / "resources" / "validation" / "classification" / "subset" / "images" / "n1" / "sample.JPEG"
    img.parent.mkdir(parents=True, exist_ok=True)
    img.write_bytes(b"image")
    result_path = bs / "native_pipeline" / "b052" / "deepx_to_trt" / "fp16" / "deepx_native_fifo_e2e_results.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)

    def fake_run(cmd, **kwargs):
        assert "--image" in cmd
        assert str(img.resolve()) in cmd
        result_path.write_text(json.dumps({"ok": True, "fps_makespan": 123.0, "paper_equivalent_fps": 124.0, "handoff_ms": 0.2}), encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout='{"ok": true}', stderr='')

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv", [
        str(source), "--root", str(root), "--backend", "deepx", "--models", "resnet50",
        "--case-map", json.dumps({"resnet50": ["b052"]}), "--precision", "fp16",
    ])
    assert mod.main() == 0
    summary = json.loads((root / "analysis_tables" / "native_deepx_producer_e2e_eval.json").read_text())
    row = summary["rows"][0]
    assert row["ok"] is True
    assert row["failure_reason"] == ""
    assert row["error"] == ""
    assert row["status_detail"] == "ok"
    assert row["input_image"] == str(img.resolve())
    assert row["input_image_source"] == "materialised_validation_subset:first_sorted"
    assert len(row["input_image_sha256"]) == 64
