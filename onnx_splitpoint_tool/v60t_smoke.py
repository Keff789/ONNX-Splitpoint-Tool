from __future__ import annotations

"""Fast hardware-free smoke checks for v60t result transport and reporting."""

import argparse
import io
import json
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Sequence

from . import __version__
from .run_modes import apply_run_mode, default_run_modes_config
from .workflow.cross_runner_reporting import compute_cross_runner_report
from .workflow.execution_binding import _copy_remote_result_files
from .workflow.runner import WORKFLOW_VERSION


def _run_checks() -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []

    def run(name: str, fn) -> None:
        try:
            fn(); checks.append({"name": name, "status": "pass"})
        except Exception as exc:
            checks.append({"name": name, "status": "fail", "error": f"{type(exc).__name__}: {exc}"})

    run("version", lambda: (_ for _ in ()).throw(AssertionError(__version__)) if __version__ not in {"0.14.19+v60t.resultnativefix", "0.14.20+v60u.nativecontractfix", "0.14.21+v60v.generationnativefix", "0.14.22+v60w.smokedeferralpackfix", "0.14.23+v60x.nativeevidencefix", "0.14.25+v60z.nativefullquality", "0.14.26+v61a.nativefullenergyprogress", "0.14.27+v61b.nativeintegrationfix", "0.14.28+v61c.nativefullpairedenergyfix", "0.14.29+v61d.nativefullsemanticfix", "0.14.30+v61e.standardguifix", "2.61.0+v61e", "2.62.0", "2.63.0", "2.64.0", "2.65.0", "2.66.0", "2.67.0", "2.68.0", "2.69.6", "2.70.6", "2.70.7", "2.70.8", "2.70.9", "2.70.10", "2.71.1", "2.71.2", "2.71.3", "2.71.4", "2.72.0", "2.72.1", "2.72.2", "2.72.3", "2.72.4", "2.72.5", "2.72.6", "2.72.7", "2.73.0", "2.73.6", "2.73.7", "2.75.2", "2.75.3", "2.75.7", "2.75.8", "2.75.9", "2.75.10", "2.75.11", "2.75.12", "2.75.13", "2.75.14", "2.75.15", "2.75.16", "2.75.17", "2.75.20", "2.75.21", "2.75.22", "2.75.24", "2.75.25", "2.75.26", "2.75.27", "2.75.28", "2.75.30", "2.75.31", "2.75.32", "2.75.38", "2.75.39", "2.75.40", "2.75.41", "2.75.42", "2.75.46", "2.75.47"} else None)
    run("workflow_version", lambda: (_ for _ in ()).throw(AssertionError(WORKFLOW_VERSION)) if WORKFLOW_VERSION not in {"v60t-result-native-ranking-fixes", "v60u-coldbuild-native-contract-diagnostics", "v60v-generation-native-runtime-fixes", "v60w-smoke-deferral-pack-selection-fixes", "v60x-native-evidence-contract-fixes", "v60z-native-full-energy-quality-evidence", "v61a-native-full-energy-progress-fixes", "v61b-native-integration-live-energy-fixes", "v61c-native-full-paired-energy-fixes", "v61d-native-full-semantic-hailo8-fixes", "v61e-standard-run-gui-diagnostics-fixes", "v2.61e-campaign-contract-hardening", "v2.62-window-validation-native-binding", "v2.63-campaign-ready", "v2.64-campaign-ready", "v2.65-campaign-ready", "v2.66-campaign-ready", "v2.67-campaign-ready", "v2.68-level-playing-field", "v2.69f-hardware-smoke-native-energy-repair", "v2.70f-legacy-log-callback-repair", "v2.70g-native-validation-bridge-repair", "v2.70h-remote-suite-bootstrap-repair", "v2.70i-native-full-evidence-repair", "v2.70j-native-evidence-reporting-repair", "v2.71.1-live-evidence-binding-repair", "v2.71.2-evidence-status-energy-resume-repair", "v2.71.3-resume-artifact-restage-repair", "v2.71.4-resume-cleanup-root-rehydration", "v2.72.0-completed-detection-evidence-contracts", "v2.72.1-mixed-runtime-energy-contract-repair", "v2.72.2-offline-energy-completed-consumer-repair", "v2.72.3-campaign-gate-quality-binding-repair", "v2.72.4-campaign-gate-quality-contract-mirror-repair", "v2.72.5-campaign-gate-hailo8-resume-contract-repair", "v2.72.6-campaign-gate-hailo8-preflight-artifact-rehydration-repair", "v2.72.7-native-energy-runtime-success-admission-repair", "v2.73.0-completed-quality-evidence-repair", "v2.73.6-single-writer-process-tree-cancellation", "v2.73.7-atomic-stage-energy-checkpoints", "v2.75.2-native-runtime-closure-partial-energy-repair", "v2.75.3-cache-verify-only-gate", "v2.75.7-cache-canary-remote-runtime-closure-repair", "v2.75.8-native-runtime-holdout-contract-repair", "v2.75.9-smoke-cache-reuse-repair", "v2.75.10-historical-cache-multihit-replay-repair", "v2.75.11-canonical-native-result-verification-repair", "v2.75.12-canonical-native-result-status-alias-repair", "v2.75.13-native-full-contract-line-repair", "v2.75.14-vendor-full-nine-row-repair", "v2.75.15-vendor-full-field-contract-repair", "v2.75.16-read-only-run-storage-admission", "v2.75.17-vendor-quality-diagnostic-hailo-alias-repair", "v2.75.20-remote-boundary-plan-finalization-repair", "v2.75.21-setup-local-tensorrt-quality-dispatch-repair", "v2.75.22-quality-replay-reporting-repair", "v2.75.24-standard-hef-preupload-gate", "v2.75.26-trt-quality-join-mean-iou-repair", "v2.75.27-final-claim-scope-cache-key-repair", "v2.75.28-final-campaign-bootstrap-energy-provenance-repair", "v2.75.30-score-independent-native-ranking-audit", "v2.75.31-compact-debug-ranking-gui-repair", "v2.75.32-audit-ranking-e2e-repair", "v2.75.38-deepx-full-quality-dxnn-dispatch-repair", "v2.75.39-pre-mutation-remote-failure-quarantine-repair", "v2.75.40-deepx-preprocessing-ab-and-full-only-quality-export-repair", "v2.75.41-deepx-calibration-500-vs-1000-canary", "v2.75.42-real-evidence-and-quality-companion-repair", "v2.75.46-native-full-onnx-attestation-standard-quality-projection", "v2.75.47-yolov7-decoder-contract-debug-pack-audit-intent"} else None)

    def energy() -> None:
        cfg = default_run_modes_config()
        profile = {
            "name": "smoke", "model_suite": {"primary": [{"id": "m", "task": "detection"}]},
            "run_profiles": [{"id": "deepx_m1_to_tensorrt", "stage1": "deepx_m1", "stage2": "tensorrt"}],
            "execution_preset": {"id": "standard", "follow_tool_config": False, "snapshot": cfg["modes"]["standard"], "overrides": {"native_enabled": True, "energy_enabled": True}},
        }
        resolved, _ = apply_run_mode(profile)
        assert resolved["energy"]["enabled"] is False
        assert resolved["native_producers"]["energy"]["enabled"] is True
        assert resolved["native_producers"]["energy"]["mode"] == "measure"
    run("native_energy_measure_semantics", energy)

    def result_transport() -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); remote = root / "remote"; (remote / "results").mkdir(parents=True)
            (remote / "results" / "benchmark_results_large.json").write_text(json.dumps([{"blob": "x" * (3 * 1024 * 1024)}]), encoding="utf-8")
            dst = root / "dst"; _copy_remote_result_files(remote, dst, flat_prefix="smoke")
            assert (dst / "benchmark_results_large.json").stat().st_size > 2 * 1024 * 1024
            manifest = json.loads((dst / "remote_diagnostics" / "smoke" / "result_copy_manifest.json").read_text())
            assert manifest["canonical_result_count"] >= 1
    run("large_canonical_result_transport", result_transport)

    def fallback() -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); remote = root / "remote"; remote.mkdir()
            payload = b'[{"ok": true}]'; info = tarfile.TarInfo("x/benchmark_results_fallback.json"); info.size = len(payload)
            with tarfile.open(remote / "results_bundle_lean.tar.gz", "w:gz") as tf: tf.addfile(info, io.BytesIO(payload))
            dst = root / "dst"; _copy_remote_result_files(remote, dst, flat_prefix="smoke")
            assert (dst / "benchmark_results_fallback.json").is_file()
    run("lean_bundle_canonical_fallback", fallback)

    def cross_runner() -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); reports = root / "reports"; (reports / "native_validation").mkdir(parents=True)
            generic=[]; native=[]; valid=[]
            for i,(g,n) in enumerate(((10,5),(20,8),(30,12)),1):
                c=f"b{i:03d}"; generic.append({"model_id":"m","case_id":c,"direction":"deepx_to_trt","backend":"deepx_to_trt","variant":"split","runner_regime":"generic","cycle_ms":g,"task_quality_status":"pass","contract_consistent":True}); native.append({"model":"m","case":c,"backend":"deepx_to_trt","ok":True,"cycle_ms":n,"precision":"fp16"}); valid.append({"model":"m","case":c,"backend":"deepx_to_trt","contract_consistent":True,"semantic_ok":True})
            (reports / "native_producer_combined_summary.json").write_text(json.dumps({"rows":native}))
            (reports / "native_validation" / "native_producer_validation_summary.json").write_text(json.dumps({"rows":valid}))
            result=compute_cross_runner_report(root,generic); assert result["status"]=="ok" and result["groups"][0]["spearman_rho"]==1.0
    run("generic_to_native_ranking_transfer", cross_runner)

    package_root = Path(__file__).resolve().parent
    run("artifact_registration_hook", lambda: (_ for _ in ()).throw(AssertionError()) if "register_benchmark_set_artifacts(" not in (package_root / "workflow/legacy_benchmarkset_binding.py").read_text(encoding="utf-8") else None)
    run("native_failure_diagnostics", lambda: (_ for _ in ()).throw(AssertionError()) if "failure_reason" not in (package_root / "resources/remote_scripts/native_full_baseline_eval_runner.py").read_text(encoding="utf-8") else None)
    run("cross_runner_report_wired", lambda: (_ for _ in ()).throw(AssertionError()) if "cross_runner_ranking_validation.csv" not in (package_root / "workflow/scientific_reporting.py").read_text(encoding="utf-8") else None)
    return checks


def run_smoke() -> dict[str, Any]:
    checks = _run_checks(); failed = sum(row["status"] != "pass" for row in checks)
    return {"schema": "onnx-splitpoint/v60t-smoke", "tool_version": __version__, "workflow_version": WORKFLOW_VERSION, "status": "ok" if not failed else "failed", "passed": len(checks)-failed, "failed": failed, "checks": checks}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="onnx-splitpoint-smoke-v60t")
    parser.add_argument("--json", default="")
    args = parser.parse_args(list(argv) if argv is not None else None)
    payload = run_smoke()
    if args.json:
        p=Path(args.json).expanduser().resolve(); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(payload,indent=2)+"\n")
    print(f"v60t smoke: {'ok' if payload['status']=='ok' else 'FAILED'} ({payload['passed']} passed, {payload['failed']} failed)")
    return 0 if payload["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
