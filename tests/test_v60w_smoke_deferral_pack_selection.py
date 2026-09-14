from __future__ import annotations

import json
from pathlib import Path

from onnx_splitpoint_tool import __version__
from onnx_splitpoint_tool.workflow.execution_binding import _run_ids_for_hardware_target
from onnx_splitpoint_tool.workflow.legacy_benchmarkset_binding import (
    _apply_deferred_hailo_full_builds,
    _discover_deferred_hailo_full_builds,
)
from onnx_splitpoint_tool.workflow.run_discovery import discover_evaluation_run
from onnx_splitpoint_tool.workflow.runner import WORKFLOW_VERSION, _native_concise_summary_v60w

ROOT = Path(__file__).resolve().parents[1]


def test_v60w_versions() -> None:
    assert __version__ in {"0.14.22+v60w.smokedeferralpackfix", "0.14.23+v60x.nativeevidencefix", "0.14.25+v60z.nativefullquality", "0.14.26+v61a.nativefullenergyprogress", "0.14.27+v61b.nativeintegrationfix", "0.14.28+v61c.nativefullpairedenergyfix", "0.14.29+v61d.nativefullsemanticfix", "0.14.30+v61e.standardguifix", "2.61.0+v61e", "2.62.0", "2.63.0", "2.64.0", "2.65.0", "2.66.0", "2.67.0", "2.68.0", "2.69.6", "2.70.6", "2.70.7", "2.70.8", "2.70.9", "2.70.10", "2.71.1", "2.71.2", "2.71.3", "2.71.4", "2.72.0", "2.72.1", "2.72.2", "2.72.3", "2.72.4", "2.72.5", "2.72.6", "2.72.7", "2.73.0", "2.73.6", "2.73.7", "2.75.2", "2.75.3", "2.75.7", "2.75.8", "2.75.9", "2.75.10", "2.75.11", "2.75.12", "2.75.13", "2.75.14", "2.75.15", "2.75.16", "2.75.17", "2.75.20", "2.75.21", "2.75.22", "2.75.24", "2.75.26", "2.75.27", "2.75.28", "2.75.30", "2.75.31", "2.75.32", "2.75.38", "2.75.39", "2.75.40", "2.75.41", "2.75.42", "2.75.46", "2.75.47"}
    assert WORKFLOW_VERSION in {"v60w-smoke-deferral-pack-selection-fixes", "v60x-native-evidence-contract-fixes", "v60z-native-full-energy-quality-evidence", "v61a-native-full-energy-progress-fixes", "v61b-native-integration-live-energy-fixes", "v61c-native-full-paired-energy-fixes", "v61d-native-full-semantic-hailo8-fixes", "v61e-standard-run-gui-diagnostics-fixes", "v2.61e-campaign-contract-hardening", "v2.62-window-validation-native-binding", "v2.63-campaign-ready", "v2.64-campaign-ready", "v2.65-campaign-ready", "v2.66-campaign-ready", "v2.67-campaign-ready", "v2.68-level-playing-field", "v2.69f-hardware-smoke-native-energy-repair", "v2.70f-legacy-log-callback-repair", "v2.70g-native-validation-bridge-repair", "v2.70h-remote-suite-bootstrap-repair", "v2.70i-native-full-evidence-repair", "v2.70j-native-evidence-reporting-repair", "v2.71.1-live-evidence-binding-repair", "v2.71.2-evidence-status-energy-resume-repair", "v2.71.3-resume-artifact-restage-repair", "v2.71.4-resume-cleanup-root-rehydration", "v2.72.0-completed-detection-evidence-contracts", "v2.72.1-mixed-runtime-energy-contract-repair", "v2.72.2-offline-energy-completed-consumer-repair", "v2.72.3-campaign-gate-quality-binding-repair", "v2.72.4-campaign-gate-quality-contract-mirror-repair", "v2.72.5-campaign-gate-hailo8-resume-contract-repair", "v2.72.6-campaign-gate-hailo8-preflight-artifact-rehydration-repair", "v2.72.7-native-energy-runtime-success-admission-repair", "v2.73.0-completed-quality-evidence-repair", "v2.73.6-single-writer-process-tree-cancellation", "v2.73.7-atomic-stage-energy-checkpoints", "v2.75.2-native-runtime-closure-partial-energy-repair", "v2.75.3-cache-verify-only-gate", "v2.75.7-cache-canary-remote-runtime-closure-repair", "v2.75.8-native-runtime-holdout-contract-repair", "v2.75.9-smoke-cache-reuse-repair", "v2.75.10-historical-cache-multihit-replay-repair", "v2.75.11-canonical-native-result-verification-repair", "v2.75.12-canonical-native-result-status-alias-repair", "v2.75.13-native-full-contract-line-repair", "v2.75.14-vendor-full-nine-row-repair", "v2.75.15-vendor-full-field-contract-repair", "v2.75.16-read-only-run-storage-admission", "v2.75.17-vendor-quality-diagnostic-hailo-alias-repair", "v2.75.20-remote-boundary-plan-finalization-repair", "v2.75.21-setup-local-tensorrt-quality-dispatch-repair", "v2.75.22-quality-replay-reporting-repair", "v2.75.24-standard-hef-preupload-gate", "v2.75.26-trt-quality-join-mean-iou-repair", "v2.75.27-final-claim-scope-cache-key-repair", "v2.75.28-final-campaign-bootstrap-energy-provenance-repair", "v2.75.30-score-independent-native-ranking-audit", "v2.75.31-compact-debug-ranking-gui-repair", "v2.75.32-audit-ranking-e2e-repair", "v2.75.38-deepx-full-quality-dxnn-dispatch-repair", "v2.75.39-pre-mutation-remote-failure-quarantine-repair", "v2.75.40-deepx-preprocessing-ab-and-full-only-quality-export-repair", "v2.75.41-deepx-calibration-500-vs-1000-canary", "v2.75.42-real-evidence-and-quality-companion-repair", "v2.75.46-native-full-onnx-attestation-standard-quality-projection", "v2.75.47-yolov7-decoder-contract-debug-pack-audit-intent"}


def test_smoke_deferred_hailo_full_is_annotated_but_split_remains_active(tmp_path: Path) -> None:
    full = tmp_path / "suite" / "hailo" / "hailo8" / "full"
    full.mkdir(parents=True)
    (full / "cold_build_request.json").write_text("{}", encoding="utf-8")
    (full / "hailo_hef_build_result.json").write_text(
        json.dumps({
            "ok": False,
            "failure_kind": "deferred_cold_full_cache_miss",
            "unsupported_reason": "cache_only_policy",
            "error": "Smoke cold-build policy deferred a Hailo Full cache miss.",
            "calib_info": {"cache_key": "abc"},
        }),
        encoding="utf-8",
    )
    builds = _discover_deferred_hailo_full_builds(tmp_path / "suite")
    assert len(builds) == 1
    plan = _apply_deferred_hailo_full_builds({
        "runs": [
            {"id": "hailo8", "type": "hailo", "variants": ["full"]},
            {"id": "hailo8_to_trt", "type": "matrix", "stage1": {"type": "hailo", "target": "hailo8"}, "stage2": {"provider": "tensorrt"}},
        ]
    }, builds)
    rows = {row["id"]: row for row in plan["runs"]}
    assert rows["hailo8"]["deferred"] is True
    assert rows["hailo8"]["required"] is False
    assert not rows["hailo8_to_trt"].get("deferred")
    selected = _run_ids_for_hardware_target({"id": "h8", "accelerator": "hailo8"}, plan)
    assert "hailo8" not in selected
    assert "hailo8_to_trt" in selected


def test_pack_discovery_prefers_newest_when_explicit_old_run_is_still_valid(tmp_path: Path) -> None:
    root = tmp_path / "EvaluationRuns"
    old = root / "old_run"
    new = root / "new_run"
    for path, created in ((old, "2026-07-15T18:00:00+02:00"), (new, "2026-07-16T08:00:00+02:00")):
        path.mkdir(parents=True)
        (path / "run_manifest.json").write_text(json.dumps({
            "schema": "onnx-splitpoint/evaluation-run-manifest",
            "schema_version": 1,
            "run_id": path.name,
            "profile_id": "test",
            "status": "partial",
            "created_at": created,
        }), encoding="utf-8")
        (path / "profile.yaml").write_text("profile_id: test\n", encoding="utf-8")
        # Automatic Debug discovery deliberately ignores manifest-only stubs.
        # These fixtures represent useful interrupted runs, so bind them to
        # substantive forensic evidence just like the real workflow does.
        (path / "evaluation_workflow.log").write_text(
            f"{path.name}: interrupted after remote preflight\n",
            encoding="utf-8",
        )
    latest = root / "_latest_evaluation_workflow.log"
    latest.write_text(f"Run directory: {new}\n", encoding="utf-8")

    explicit = discover_evaluation_run(preferred=[old], output_roots=[root], latest_logs=[latest], prefer_valid_explicit=True)
    latest_result = discover_evaluation_run(preferred=[old], output_roots=[root], latest_logs=[latest], prefer_valid_explicit=False)
    assert explicit.selected == old
    assert latest_result.selected == new


def test_gui_pack_actions_request_latest_run_and_pack_identity() -> None:
    text = (ROOT / "onnx_splitpoint_tool" / "gui" / "app.py").read_text(encoding="utf-8")
    assert 'purpose="debug"' in text
    assert 'purpose="analysis"' in text
    assert "newest_identified_evaluation_run_for_debug" in text
    assert "ONNX_SPLITPOINT_EXPORT_DIR" in text
    analysis = (ROOT / "onnx_splitpoint_tool" / "workflow" / "analysis_pack.py").read_text(encoding="utf-8")
    assert "pack_source_identity.json" in analysis


def test_generated_suite_filters_deferred_plan_rows() -> None:
    text = (ROOT / "onnx_splitpoint_tool" / "resources" / "templates" / "benchmark_suite.py.txt").read_text(encoding="utf-8")
    assert "[plan] skipping intentionally deferred runs:" in text
    compile(text, "benchmark_suite.py.txt", "exec")


def test_native_concise_summary_preserves_runtime_and_semantic_status(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    (reports / "native_validation").mkdir(parents=True)
    (reports / "native_producer_combined_summary.json").write_text(json.dumps({
        "rows": [
            {"model": "resnet50", "backend": "deepx_to_trt", "case": "b052", "precision": "fp16", "ok": True, "fps_makespan": 123.4},
            {"model": "yolo26s", "backend": "hailo8_to_trt", "case": "b038", "precision": "fp16", "ok": False, "failure_reason": "bridge_failed"},
        ]
    }), encoding="utf-8")
    (reports / "native_validation" / "native_producer_validation_summary.json").write_text(json.dumps({
        "rows": [
            {"model": "resnet50", "backend": "deepx_to_trt", "case": "b052", "precision": "fp16", "status": "pass", "semantic_ok": True, "claim_ok": True},
            {"model": "yolo26s", "backend": "hailo8_to_trt", "case": "b038", "precision": "fp16", "status": "fail", "semantic_ok": False},
        ]
    }), encoding="utf-8")
    paths, rows = _native_concise_summary_v60w(reports)
    assert len(rows) == 2
    assert rows[0]["runtime_status"] == "ok"
    assert rows[0]["semantic_status"] == "pass"
    assert rows[1]["failure_reason"] == "bridge_failed"
    assert paths["native_stage_concise_summary_json"].is_file()
    assert paths["native_stage_concise_summary_csv"].is_file()
