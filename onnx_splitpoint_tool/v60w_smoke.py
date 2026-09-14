from __future__ import annotations

"""Hardware-free regression smoke for v60w Smoke deferral and pack selection."""

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any, Sequence

from . import __version__
from .v60v_smoke import _run_checks as _base_checks
from .workflow.execution_binding import _run_ids_for_hardware_target
from .workflow.legacy_benchmarkset_binding import _apply_deferred_hailo_full_builds
from .workflow.run_discovery import discover_evaluation_run
from .workflow.runner import WORKFLOW_VERSION, _native_concise_summary_v60w


def _run_checks() -> list[dict[str, Any]]:
    checks = list(_base_checks())

    def run(name: str, fn) -> None:
        try:
            fn()
            checks.append({"name": name, "status": "pass"})
        except Exception as exc:
            checks.append({"name": name, "status": "fail", "error": f"{type(exc).__name__}: {exc}"})

    run("v60w_version", lambda: (_ for _ in ()).throw(AssertionError(__version__)) if __version__ not in {"0.14.22+v60w.smokedeferralpackfix", "0.14.23+v60x.nativeevidencefix", "0.14.25+v60z.nativefullquality", "0.14.26+v61a.nativefullenergyprogress", "0.14.27+v61b.nativeintegrationfix", "0.14.28+v61c.nativefullpairedenergyfix", "0.14.29+v61d.nativefullsemanticfix", "0.14.30+v61e.standardguifix", "2.61.0+v61e", "2.62.0", "2.63.0", "2.64.0", "2.65.0", "2.66.0", "2.67.0", "2.68.0", "2.69.6", "2.70.6", "2.70.7", "2.70.8", "2.70.9", "2.70.10", "2.71.1", "2.71.2", "2.71.3", "2.71.4", "2.72.0", "2.72.1", "2.72.2", "2.72.3", "2.72.4", "2.72.5", "2.72.6", "2.72.7", "2.73.0", "2.73.6", "2.73.7", "2.75.2", "2.75.3", "2.75.7", "2.75.8", "2.75.9", "2.75.10", "2.75.11", "2.75.12", "2.75.13", "2.75.14", "2.75.15", "2.75.16", "2.75.17", "2.75.20", "2.75.21", "2.75.22", "2.75.24", "2.75.25", "2.75.26", "2.75.27", "2.75.28", "2.75.30", "2.75.31", "2.75.32", "2.75.38", "2.75.39", "2.75.40", "2.75.41", "2.75.42", "2.75.46", "2.75.47"} else None)
    run("v60w_workflow", lambda: (_ for _ in ()).throw(AssertionError(WORKFLOW_VERSION)) if WORKFLOW_VERSION not in {"v60w-smoke-deferral-pack-selection-fixes", "v60x-native-evidence-contract-fixes", "v60z-native-full-energy-quality-evidence", "v61a-native-full-energy-progress-fixes", "v61b-native-integration-live-energy-fixes", "v61c-native-full-paired-energy-fixes", "v61d-native-full-semantic-hailo8-fixes", "v61e-standard-run-gui-diagnostics-fixes", "v2.61e-campaign-contract-hardening", "v2.62-window-validation-native-binding", "v2.63-campaign-ready", "v2.64-campaign-ready", "v2.65-campaign-ready", "v2.66-campaign-ready", "v2.67-campaign-ready", "v2.68-level-playing-field", "v2.69f-hardware-smoke-native-energy-repair", "v2.70f-legacy-log-callback-repair", "v2.70g-native-validation-bridge-repair", "v2.70h-remote-suite-bootstrap-repair", "v2.70i-native-full-evidence-repair", "v2.70j-native-evidence-reporting-repair", "v2.71.1-live-evidence-binding-repair", "v2.71.2-evidence-status-energy-resume-repair", "v2.71.3-resume-artifact-restage-repair", "v2.71.4-resume-cleanup-root-rehydration", "v2.72.0-completed-detection-evidence-contracts", "v2.72.1-mixed-runtime-energy-contract-repair", "v2.72.2-offline-energy-completed-consumer-repair", "v2.72.3-campaign-gate-quality-binding-repair", "v2.72.4-campaign-gate-quality-contract-mirror-repair", "v2.72.5-campaign-gate-hailo8-resume-contract-repair", "v2.72.6-campaign-gate-hailo8-preflight-artifact-rehydration-repair", "v2.72.7-native-energy-runtime-success-admission-repair", "v2.73.0-completed-quality-evidence-repair", "v2.73.6-single-writer-process-tree-cancellation", "v2.73.7-atomic-stage-energy-checkpoints", "v2.75.2-native-runtime-closure-partial-energy-repair", "v2.75.3-cache-verify-only-gate", "v2.75.7-cache-canary-remote-runtime-closure-repair", "v2.75.8-native-runtime-holdout-contract-repair", "v2.75.9-smoke-cache-reuse-repair", "v2.75.10-historical-cache-multihit-replay-repair", "v2.75.11-canonical-native-result-verification-repair", "v2.75.12-canonical-native-result-status-alias-repair", "v2.75.13-native-full-contract-line-repair", "v2.75.14-vendor-full-nine-row-repair", "v2.75.15-vendor-full-field-contract-repair", "v2.75.16-read-only-run-storage-admission", "v2.75.17-vendor-quality-diagnostic-hailo-alias-repair", "v2.75.20-remote-boundary-plan-finalization-repair", "v2.75.21-setup-local-tensorrt-quality-dispatch-repair", "v2.75.22-quality-replay-reporting-repair", "v2.75.24-standard-hef-preupload-gate", "v2.75.26-trt-quality-join-mean-iou-repair", "v2.75.27-final-claim-scope-cache-key-repair", "v2.75.28-final-campaign-bootstrap-energy-provenance-repair", "v2.75.30-score-independent-native-ranking-audit", "v2.75.31-compact-debug-ranking-gui-repair", "v2.75.32-audit-ranking-e2e-repair", "v2.75.38-deepx-full-quality-dxnn-dispatch-repair", "v2.75.39-pre-mutation-remote-failure-quarantine-repair", "v2.75.40-deepx-preprocessing-ab-and-full-only-quality-export-repair", "v2.75.41-deepx-calibration-500-vs-1000-canary", "v2.75.42-real-evidence-and-quality-companion-repair", "v2.75.46-native-full-onnx-attestation-standard-quality-projection", "v2.75.47-yolov7-decoder-contract-debug-pack-audit-intent"} else None)

    def deferred_dispatch() -> None:
        plan = _apply_deferred_hailo_full_builds({
            "runs": [
                {"id": "hailo8", "type": "hailo", "variants": ["full"]},
                {"id": "hailo8_to_trt", "type": "matrix", "stage1": {"type": "hailo", "target": "hailo8"}},
            ]
        }, [{"target": "hailo8", "reason": "deferred_cold_full_cache_miss"}])
        ids = _run_ids_for_hardware_target({"id": "h8", "accelerator": "hailo8"}, plan)
        assert ids == ["hailo8_to_trt"]
    run("deferred_full_not_dispatched", deferred_dispatch)

    def latest_pack_run() -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            old = root / "old"; new = root / "new"
            for path, ts in ((old, "2026-07-15T00:00:00+02:00"), (new, "2026-07-16T00:00:00+02:00")):
                path.mkdir()
                (path / "run_manifest.json").write_text(json.dumps({
                    "schema": "onnx-splitpoint/evaluation-run-manifest",
                    "schema_version": 1,
                    "run_id": path.name,
                    "profile_id": "test",
                    "status": "partial",
                    "created_at": ts,
                }), encoding="utf-8")
                (path / "profile.yaml").write_text("profile_id: test\n", encoding="utf-8")
            log = root / "_latest_evaluation_workflow.log"
            log.write_text(f"Run directory: {new}\n", encoding="utf-8")
            result = discover_evaluation_run(preferred=[old], output_roots=[root], latest_logs=[log], prefer_valid_explicit=False)
            assert result.selected == new
    run("pack_prefers_latest_run", latest_pack_run)

    def concise_native() -> None:
        with tempfile.TemporaryDirectory() as td:
            reports = Path(td)
            (reports / "native_validation").mkdir()
            (reports / "native_producer_summary.json").write_text(json.dumps({"rows": [{"model": "m", "backend": "b", "case": "c", "ok": False, "failure_reason": "x"}]}), encoding="utf-8")
            (reports / "native_validation" / "native_producer_validation_summary.json").write_text(json.dumps({"rows": []}), encoding="utf-8")
            _, rows = _native_concise_summary_v60w(reports)
            assert rows[0]["failure_reason"] == "x"
    run("native_concise_summary", concise_native)

    return checks


def run_smoke() -> dict[str, Any]:
    checks = _run_checks()
    failed = sum(row["status"] != "pass" for row in checks)
    return {
        "schema": "onnx-splitpoint/v60w-smoke",
        "tool_version": __version__,
        "workflow_version": WORKFLOW_VERSION,
        "status": "ok" if failed == 0 else "failed",
        "passed": len(checks) - failed,
        "failed": failed,
        "checks": checks,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="onnx-splitpoint-smoke-v60w")
    parser.add_argument("--json", default="")
    args = parser.parse_args(list(argv) if argv is not None else None)
    payload = run_smoke()
    if args.json:
        path = Path(args.json).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"v60w smoke: {'ok' if payload['status'] == 'ok' else 'FAILED'} ({payload['passed']} passed, {payload['failed']} failed)")
    return 0 if payload["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
