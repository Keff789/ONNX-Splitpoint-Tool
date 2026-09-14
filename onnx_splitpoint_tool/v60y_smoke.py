from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any, Callable

from . import __version__
from .workflow.native_transfer import (
    build_native_transfer_inventory,
    build_native_validation_image_map,
    classify_native_transfer_failure,
    safe_native_remote_root,
)
from .workflow.runner import WORKFLOW_VERSION, _native_expected_matrix_status_v60y


def main() -> int:
    ap = argparse.ArgumentParser(description="Hardware-free v60y regression smoke")
    ap.add_argument("--json", default="")
    ns = ap.parse_args()
    rows: list[dict[str, Any]] = []

    def check(name: str, fn: Callable[[], None]) -> None:
        try:
            fn()
            rows.append({"name": name, "ok": True, "error": ""})
        except Exception as exc:
            rows.append({"name": name, "ok": False, "error": f"{type(exc).__name__}: {exc}"})

    check("version", lambda: (_ for _ in ()).throw(AssertionError(__version__)) if __version__ not in {"0.14.25+v60z.nativefullquality", "0.14.26+v61a.nativefullenergyprogress", "0.14.27+v61b.nativeintegrationfix", "0.14.28+v61c.nativefullpairedenergyfix", "0.14.29+v61d.nativefullsemanticfix", "0.14.30+v61e.standardguifix", "2.61.0+v61e", "2.62.0", "2.63.0", "2.64.0", "2.65.0", "2.66.0", "2.67.0", "2.68.0", "2.69.6", "2.70.6", "2.70.7", "2.70.8", "2.70.9", "2.70.10", "2.71.1", "2.71.2", "2.71.3", "2.71.4", "2.72.0", "2.72.1", "2.72.2", "2.72.3", "2.72.4", "2.72.5", "2.72.6", "2.72.7", "2.73.0", "2.73.6", "2.73.7", "2.75.2", "2.75.3", "2.75.7", "2.75.8", "2.75.9", "2.75.10", "2.75.11", "2.75.12", "2.75.13", "2.75.14", "2.75.15", "2.75.16", "2.75.17", "2.75.20", "2.75.21", "2.75.22", "2.75.24", "2.75.25", "2.75.26", "2.75.27", "2.75.28", "2.75.30", "2.75.31", "2.75.32", "2.75.38", "2.75.39", "2.75.40", "2.75.41", "2.75.42", "2.75.46", "2.75.47"} else None)
    check("workflow", lambda: (_ for _ in ()).throw(AssertionError(WORKFLOW_VERSION)) if WORKFLOW_VERSION not in {"v60z-native-full-energy-quality-evidence", "v61a-native-full-energy-progress-fixes", "v61b-native-integration-live-energy-fixes", "v61c-native-full-paired-energy-fixes", "v61d-native-full-semantic-hailo8-fixes", "v61e-standard-run-gui-diagnostics-fixes", "v2.61e-campaign-contract-hardening", "v2.62-window-validation-native-binding", "v2.63-campaign-ready", "v2.64-campaign-ready", "v2.65-campaign-ready", "v2.66-campaign-ready", "v2.67-campaign-ready", "v2.68-level-playing-field", "v2.69f-hardware-smoke-native-energy-repair", "v2.70f-legacy-log-callback-repair", "v2.70g-native-validation-bridge-repair", "v2.70h-remote-suite-bootstrap-repair", "v2.70i-native-full-evidence-repair", "v2.70j-native-evidence-reporting-repair", "v2.71.1-live-evidence-binding-repair", "v2.71.2-evidence-status-energy-resume-repair", "v2.71.3-resume-artifact-restage-repair", "v2.71.4-resume-cleanup-root-rehydration", "v2.72.0-completed-detection-evidence-contracts", "v2.72.1-mixed-runtime-energy-contract-repair", "v2.72.2-offline-energy-completed-consumer-repair", "v2.72.3-campaign-gate-quality-binding-repair", "v2.72.4-campaign-gate-quality-contract-mirror-repair", "v2.72.5-campaign-gate-hailo8-resume-contract-repair", "v2.72.6-campaign-gate-hailo8-preflight-artifact-rehydration-repair", "v2.72.7-native-energy-runtime-success-admission-repair", "v2.73.0-completed-quality-evidence-repair", "v2.73.6-single-writer-process-tree-cancellation", "v2.73.7-atomic-stage-energy-checkpoints", "v2.75.2-native-runtime-closure-partial-energy-repair", "v2.75.3-cache-verify-only-gate", "v2.75.7-cache-canary-remote-runtime-closure-repair", "v2.75.8-native-runtime-holdout-contract-repair", "v2.75.9-smoke-cache-reuse-repair", "v2.75.10-historical-cache-multihit-replay-repair", "v2.75.11-canonical-native-result-verification-repair", "v2.75.12-canonical-native-result-status-alias-repair", "v2.75.13-native-full-contract-line-repair", "v2.75.14-vendor-full-nine-row-repair", "v2.75.15-vendor-full-field-contract-repair", "v2.75.16-read-only-run-storage-admission", "v2.75.17-vendor-quality-diagnostic-hailo-alias-repair", "v2.75.20-remote-boundary-plan-finalization-repair", "v2.75.21-setup-local-tensorrt-quality-dispatch-repair", "v2.75.22-quality-replay-reporting-repair", "v2.75.24-standard-hef-preupload-gate", "v2.75.26-trt-quality-join-mean-iou-repair", "v2.75.27-final-claim-scope-cache-key-repair", "v2.75.28-final-campaign-bootstrap-energy-provenance-repair", "v2.75.30-score-independent-native-ranking-audit", "v2.75.31-compact-debug-ranking-gui-repair", "v2.75.32-audit-ranking-e2e-repair", "v2.75.38-deepx-full-quality-dxnn-dispatch-repair", "v2.75.39-pre-mutation-remote-failure-quarantine-repair", "v2.75.40-deepx-preprocessing-ab-and-full-only-quality-export-repair", "v2.75.41-deepx-calibration-500-vs-1000-canary", "v2.75.42-real-evidence-and-quality-companion-repair", "v2.75.46-native-full-onnx-attestation-standard-quality-projection", "v2.75.47-yolov7-decoder-contract-debug-pack-audit-intent"} else None)
    check("disk_failure_classification", lambda: (_ for _ in ()).throw(AssertionError()) if classify_native_transfer_failure("No space left on device (28)") != "remote_disk_insufficient" else None)
    check("safe_cleanup_guard", lambda: (_ for _ in ()).throw(AssertionError()) if not safe_native_remote_root("/home/nx/native_fifo_evalsets/run_20260716_120000", "run_20260716_120000") else None)

    def inventory_check() -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            keep = root / "b001" / "hailo" / "hailo8" / "part1" / "compiled.hef"
            drop = root / "dist" / "suite_bundle.tar.gz"
            img = root / "resources" / "validation" / "classification" / "subset" / "images" / "n1" / "a.JPEG"
            for p in (keep, drop, img):
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b"x")
            inv = build_native_transfer_inventory(root)
            paths = set(inv["relative_paths"])
            assert keep.relative_to(root).as_posix() in paths
            assert img.relative_to(root).as_posix() in paths
            assert drop.relative_to(root).as_posix() not in paths
    check("lean_native_transfer", inventory_check)

    def image_map_check() -> None:
        with tempfile.TemporaryDirectory() as td:
            run = Path(td)
            bs = run / "models" / "m" / "benchmark_set" / "legacy_suite"
            img = bs / "resources" / "validation" / "classification" / "subset" / "images" / "n" / "a.JPEG"
            img.parent.mkdir(parents=True, exist_ok=True)
            img.write_bytes(b"x")
            report = run / "models" / "m" / "benchmark_results" / "remote_diagnostics" / "case_reports" / "results" / "b001" / "results_ort_cpu" / "validation_report.json"
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_text(json.dumps({"run_cfg": {"image": "/remote/a.JPEG"}}), encoding="utf-8")
            mapping, _ = build_native_validation_image_map(run, ["m"], {"m": ["b001"]}, {"m": bs})
            assert mapping == {"m": {"b001": "a.JPEG"}}
    check("generic_to_native_exact_image", image_map_check)

    def matrix_check() -> None:
        expected = [
            {"backend_key": "deepx", "backend": "deepx_to_trt", "model": "m", "case": "b001", "precision": "fp16"},
            {"backend_key": "hailo10h", "backend": "hailo10h_to_trt", "model": "m", "case": "b001", "precision": "fp16"},
        ]
        actual = [{"backend": "deepx_to_trt", "model": "m", "case": "b001", "precision": "fp16", "ok": True}]
        result = _native_expected_matrix_status_v60y(expected, actual, [{"backend": "hailo10h", "ok": False, "failure_reason": "remote_disk_insufficient"}])
        assert result["missing_expected_row_count"] == 1
        assert result["missing_expected_rows"][0]["failure_reason"] == "remote_disk_insufficient"
    check("native_expected_matrix", matrix_check)

    failed = [row for row in rows if not row["ok"]]
    payload = {"schema": "onnx-splitpoint/v60y-smoke", "version": __version__, "passed": len(rows) - len(failed), "failed": len(failed), "rows": rows}
    if ns.json:
        Path(ns.json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"v60y smoke: {'ok' if not failed else 'failed'} ({payload['passed']} passed, {payload['failed']} failed)")
    for row in failed:
        print(f"  FAIL {row['name']}: {row['error']}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
