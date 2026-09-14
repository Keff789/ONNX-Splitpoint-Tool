from __future__ import annotations

"""Hardware-free regression smoke for v60v generation/native-runtime recovery."""

import argparse
import importlib.util
import json
import os
import tempfile
import threading
import time
import zipfile
from pathlib import Path
from typing import Any, Sequence

from . import __version__
from .artifact_store import ArtifactStore
from .build_scheduler import BuildScheduler, BuildTaskSpec
from .run_modes import RUN_MODE_SCHEMA_VERSION, apply_run_mode, default_run_modes_config
from .workflow.cross_runner_reporting import compute_cross_runner_report
from .workflow.execution_binding import _copy_remote_result_files
from .workflow.runner import WORKFLOW_VERSION
from .workflow.zip_utils import write_path_portable


def _script_asset(name: str) -> Path:
    """Resolve a Native helper in both source-tree and installed-wheel layouts."""
    package_root = Path(__file__).resolve().parent
    candidates = (
        package_root.parent / "scripts" / name,
        package_root / "resources" / "remote_scripts" / name,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Native helper asset not found: {name}; checked: {candidates}")


def _load_script_asset(module_name: str, name: str):
    script = _script_asset(name)
    added = False
    if str(script.parent) not in __import__("sys").path:
        __import__("sys").path.insert(0, str(script.parent))
        added = True
    try:
        spec = importlib.util.spec_from_file_location(module_name, script)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if added:
            try:
                __import__("sys").path.remove(str(script.parent))
            except ValueError:
                pass


def _run_checks() -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []

    def run(name: str, fn) -> None:
        try:
            fn()
            checks.append({"name": name, "status": "pass"})
        except Exception as exc:
            checks.append({"name": name, "status": "fail", "error": f"{type(exc).__name__}: {exc}"})

    run("version", lambda: (_ for _ in ()).throw(AssertionError(__version__)) if __version__ not in {"0.14.21+v60v.generationnativefix", "0.14.22+v60w.smokedeferralpackfix", "0.14.23+v60x.nativeevidencefix", "0.14.25+v60z.nativefullquality", "0.14.26+v61a.nativefullenergyprogress", "0.14.27+v61b.nativeintegrationfix", "0.14.28+v61c.nativefullpairedenergyfix", "0.14.29+v61d.nativefullsemanticfix", "0.14.30+v61e.standardguifix", "2.61.0+v61e", "2.62.0", "2.63.0", "2.64.0", "2.65.0", "2.66.0", "2.67.0", "2.68.0", "2.69.6", "2.70.6", "2.70.7", "2.70.8", "2.70.9", "2.70.10", "2.71.1", "2.71.2", "2.71.3", "2.71.4", "2.72.0", "2.72.1", "2.72.2", "2.72.3", "2.72.4", "2.72.5", "2.72.6", "2.72.7", "2.73.0", "2.73.6", "2.73.7", "2.75.2", "2.75.3", "2.75.7", "2.75.8", "2.75.9", "2.75.10", "2.75.11", "2.75.12", "2.75.13", "2.75.14", "2.75.15", "2.75.16", "2.75.17", "2.75.20", "2.75.21", "2.75.22", "2.75.24", "2.75.25", "2.75.26", "2.75.27", "2.75.28", "2.75.30", "2.75.31", "2.75.32", "2.75.38", "2.75.39", "2.75.40", "2.75.41", "2.75.42", "2.75.46", "2.75.47"} else None)
    run("workflow_version", lambda: (_ for _ in ()).throw(AssertionError(WORKFLOW_VERSION)) if WORKFLOW_VERSION not in {"v60v-generation-native-runtime-fixes", "v60w-smoke-deferral-pack-selection-fixes", "v60x-native-evidence-contract-fixes", "v60z-native-full-energy-quality-evidence", "v61a-native-full-energy-progress-fixes", "v61b-native-integration-live-energy-fixes", "v61c-native-full-paired-energy-fixes", "v61d-native-full-semantic-hailo8-fixes", "v61e-standard-run-gui-diagnostics-fixes", "v2.61e-campaign-contract-hardening", "v2.62-window-validation-native-binding", "v2.63-campaign-ready", "v2.64-campaign-ready", "v2.65-campaign-ready", "v2.66-campaign-ready", "v2.67-campaign-ready", "v2.68-level-playing-field", "v2.69f-hardware-smoke-native-energy-repair", "v2.70f-legacy-log-callback-repair", "v2.70g-native-validation-bridge-repair", "v2.70h-remote-suite-bootstrap-repair", "v2.70i-native-full-evidence-repair", "v2.70j-native-evidence-reporting-repair", "v2.71.1-live-evidence-binding-repair", "v2.71.2-evidence-status-energy-resume-repair", "v2.71.3-resume-artifact-restage-repair", "v2.71.4-resume-cleanup-root-rehydration", "v2.72.0-completed-detection-evidence-contracts", "v2.72.1-mixed-runtime-energy-contract-repair", "v2.72.2-offline-energy-completed-consumer-repair", "v2.72.3-campaign-gate-quality-binding-repair", "v2.72.4-campaign-gate-quality-contract-mirror-repair", "v2.72.5-campaign-gate-hailo8-resume-contract-repair", "v2.72.6-campaign-gate-hailo8-preflight-artifact-rehydration-repair", "v2.72.7-native-energy-runtime-success-admission-repair", "v2.73.0-completed-quality-evidence-repair", "v2.73.6-single-writer-process-tree-cancellation", "v2.73.7-atomic-stage-energy-checkpoints", "v2.75.2-native-runtime-closure-partial-energy-repair", "v2.75.3-cache-verify-only-gate", "v2.75.7-cache-canary-remote-runtime-closure-repair", "v2.75.8-native-runtime-holdout-contract-repair", "v2.75.9-smoke-cache-reuse-repair", "v2.75.10-historical-cache-multihit-replay-repair", "v2.75.11-canonical-native-result-verification-repair", "v2.75.12-canonical-native-result-status-alias-repair", "v2.75.13-native-full-contract-line-repair", "v2.75.14-vendor-full-nine-row-repair", "v2.75.15-vendor-full-field-contract-repair", "v2.75.16-read-only-run-storage-admission", "v2.75.17-vendor-quality-diagnostic-hailo-alias-repair", "v2.75.20-remote-boundary-plan-finalization-repair", "v2.75.21-setup-local-tensorrt-quality-dispatch-repair", "v2.75.22-quality-replay-reporting-repair", "v2.75.24-standard-hef-preupload-gate", "v2.75.26-trt-quality-join-mean-iou-repair", "v2.75.27-final-claim-scope-cache-key-repair", "v2.75.28-final-campaign-bootstrap-energy-provenance-repair", "v2.75.30-score-independent-native-ranking-audit", "v2.75.31-compact-debug-ranking-gui-repair", "v2.75.32-audit-ranking-e2e-repair", "v2.75.38-deepx-full-quality-dxnn-dispatch-repair", "v2.75.39-pre-mutation-remote-failure-quarantine-repair", "v2.75.40-deepx-preprocessing-ab-and-full-only-quality-export-repair", "v2.75.41-deepx-calibration-500-vs-1000-canary", "v2.75.42-real-evidence-and-quality-companion-repair", "v2.75.46-native-full-onnx-attestation-standard-quality-projection", "v2.75.47-yolov7-decoder-contract-debug-pack-audit-intent"} else None)
    run("run_mode_schema", lambda: (_ for _ in ()).throw(AssertionError(RUN_MODE_SCHEMA_VERSION)) if RUN_MODE_SCHEMA_VERSION not in {5, 7, 8, 9, 10, 11, 12} else None)

    def cold_policy() -> None:
        cfg = default_run_modes_config()
        assert cfg["modes"]["smoke"]["build"]["hailo"]["full_baseline_cold_build_policy"] == "cache_or_defer"
        assert cfg["modes"]["standard"]["build"]["hailo"]["full_baseline_cold_build_policy"] == "build_missing"
    run("cold_full_policy", cold_policy)

    def semantic_result_count() -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            remote = root / "remote" / "results"
            remote.mkdir(parents=True)
            (remote / "benchmark_results_empty.json").write_text("[]", encoding="utf-8")
            (remote / "benchmark_results_empty.error.txt").write_text("No results collected", encoding="utf-8")
            (remote / "benchmark_results_ok.json").write_text(json.dumps([{"run_id": "ok", "fps": 1.0}]), encoding="utf-8")
            dst = root / "dst"
            _copy_remote_result_files(remote.parent, dst, flat_prefix="target")
            manifest = json.loads((dst / "remote_diagnostics" / "target" / "result_copy_manifest.json").read_text())
            assert manifest["canonical_file_count"] == 2
            assert manifest["canonical_nonempty_row_count"] == 1
            assert "ok" in manifest["canonical_run_ids_with_rows"]
            assert "empty" in manifest["canonical_run_ids_without_rows"]
            assert manifest["status"] == "partial"
    run("semantic_canonical_result_count", semantic_result_count)

    def portable_zip() -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src = root / "epoch.txt"
            src.write_text("diagnostic", encoding="utf-8")
            os.utime(src, (0, 0))
            out = root / "pack.zip"
            with zipfile.ZipFile(out, "w") as zf:
                write_path_portable(zf, src, "epoch.txt")
            with zipfile.ZipFile(out) as zf:
                info = zf.getinfo("epoch.txt")
                assert info.date_time[0] >= 1980
                assert zf.read("epoch.txt") == b"diagnostic"
    run("portable_debug_zip_timestamp", portable_zip)

    def artifact_restore() -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            store = ArtifactStore(root / "store")
            src = root / "compiled.hef"
            src.write_bytes(b"hef-v60u")
            rec = store.register(source_path=src, kind="hailo_hef", contract={"model": "m", "arch": "hailo8"})
            src.unlink()
            dst = root / "restored" / "compiled.hef"
            store.materialize(rec, dst, reference="v60u-smoke")
            assert dst.read_bytes() == b"hef-v60u"
    run("artifact_store_object_restore", artifact_restore)

    def scheduler_overlap() -> None:
        barrier = threading.Barrier(2)
        intervals: dict[str, tuple[float, float]] = {}
        def work(name: str) -> str:
            start = time.monotonic()
            barrier.wait(timeout=2)
            time.sleep(0.05)
            intervals[name] = (start, time.monotonic())
            return name
        with BuildScheduler(max_workers=2, cpu_tokens=8, family_limits={"hailo8": 1, "deepx": 1}) as scheduler:
            a = scheduler.submit(BuildTaskSpec("hailo", "hailo8", 2), work, "hailo")
            b = scheduler.submit(BuildTaskSpec("deepx", "deepx", 2), work, "deepx")
            assert {a.result(timeout=3), b.result(timeout=3)} == {"hailo", "deepx"}
        assert max(intervals["hailo"][0], intervals["deepx"][0]) < min(intervals["hailo"][1], intervals["deepx"][1])
    run("parallel_cold_build_scheduler", scheduler_overlap)


    def native_energy_semantics() -> None:
        cfg = default_run_modes_config()
        profile = {
            "name": "v60u-energy-smoke",
            "model_suite": {"primary": [{"id": "m", "task": "detection"}]},
            "selection_policy": {"max_accepted_cases_per_model": 1},
            "run_profiles": [{"id": "deepx_m1_to_tensorrt"}],
            "execution_preset": {
                "id": "smoke", "follow_tool_config": False,
                "snapshot": cfg["modes"]["smoke"],
                "overrides": {"native_enabled": True, "energy_enabled": True},
            },
        }
        resolved, _ = apply_run_mode(profile)
        assert resolved["energy"]["enabled"] is False
        assert resolved["energy"]["generic_enabled"] is False
        assert resolved["native_producers"]["energy"]["enabled"] is True
        assert resolved["native_producers"]["energy"]["mode"] == "measure"
    run("native_energy_measure_semantics", native_energy_semantics)

    def cross_runner_three_candidates() -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            reports = root / "reports"
            (reports / "native_validation").mkdir(parents=True)
            generic_rows = []
            native_rows = []
            validation_rows = []
            for idx, (g, n) in enumerate(((10.0, 5.0), (20.0, 8.0), (30.0, 12.0)), start=1):
                case = f"b{idx:03d}"
                generic_rows.append({
                    "model_id": "m", "case_id": case, "direction": "deepx_m1_to_tensorrt",
                    "backend": "deepx_m1_to_tensorrt", "variant": "split",
                    "runner_regime": "generic", "cycle_ms": g,
                    "task_quality_status": "pass", "contract_consistent": True,
                })
                native_rows.append({
                    "model": "m", "case": case, "backend": "deepx_to_trt",
                    "ok": True, "cycle_ms": n, "precision": "fp16", "status": "ok",
                })
                validation_rows.append({
                    "model": "m", "case": case, "backend": "deepx_to_trt",
                    "contract_consistent": True, "semantic_ok": True, "claim_ok": True,
                })
            (reports / "native_producer_combined_summary.json").write_text(json.dumps({"rows": native_rows}), encoding="utf-8")
            (reports / "native_validation" / "native_producer_validation_summary.json").write_text(json.dumps({"rows": validation_rows}), encoding="utf-8")
            result = compute_cross_runner_report(root, generic_rows, minimum_candidates=3)
            assert result["status"] == "ok"
            assert result["eligible_pair_count"] == 3
            assert result["groups"][0]["spearman_rho"] == 1.0
    run("cross_runner_three_candidate_transfer", cross_runner_three_candidates)

    def targeted_profile_generator() -> None:
        from .v60u_targeted_smokes import SCENARIOS, generate_profiles
        import yaml
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "profile.yaml"
            source.write_text(yaml.safe_dump({
                "name": "demo",
                "model_suite": {"primary": [{"id": "m", "task": "detection"}]},
                "selection_policy": {"max_accepted_cases_per_model": 1},
                "run_profiles": [{"id": "deepx_m1_to_tensorrt"}],
            }, sort_keys=False), encoding="utf-8")
            manifest = generate_profiles(source, root / "out", SCENARIOS)
            assert len(manifest["generated"]) == len(SCENARIOS)
    run("targeted_hardware_smoke_profiles", targeted_profile_generator)

    def contract_metadata_gate() -> None:
        module = _load_script_asset("v60u_native_validator_smoke", "native_producer_validate_visualize.py")
        with tempfile.TemporaryDirectory() as td:
            manifest = Path(td) / "native_outputs_manifest.json"
            manifest.write_text(json.dumps({"outputs": []}), encoding="utf-8")
            result = module._enforce_detection_contract({
                "ok": True, "semantic_ok": True, "semantic_available": True,
                "decode_mode": "output0:raw",
                "best": {"native_mode": "output0:raw", "full_mode": "output0:raw"},
            }, manifest)
            assert result["ok"] is False
            assert result["diagnosis"] == "detection_contract_metadata_unavailable"
    run("detection_contract_metadata_gate", contract_metadata_gate)

    package_root = Path(__file__).resolve().parent
    run("native_diagnostic_fields", lambda: (_ for _ in ()).throw(AssertionError()) if "failure_reason" not in _script_asset("native_fifo_eval_runner.py").read_text(encoding="utf-8") else None)
    run("native_contract_family_gate", lambda: (_ for _ in ()).throw(AssertionError()) if "detection_contract_family_mismatch" not in _script_asset("native_producer_validate_visualize.py").read_text(encoding="utf-8") else None)
    run("debug_pack_native_analysis_tables", lambda: (_ for _ in ()).throw(AssertionError()) if 'for root_rel in ("native_producers", "reports/native_validation")' not in (package_root / "gui/app.py").read_text(encoding="utf-8") else None)
    run("hailo_attempt_provenance", lambda: (_ for _ in ()).throw(AssertionError()) if "attempted_timeout" not in (package_root / "workflow/hailo_remote_binding.py").read_text(encoding="utf-8") else None)
    run("cache_only_hef", lambda: (_ for _ in ()).throw(AssertionError()) if "deferred_cold_full_cache_miss" not in (package_root / "hailo_backend.py").read_text(encoding="utf-8") else None)

    def generation_postcondition() -> None:
        from .workflow.runner import benchmark_set_postcondition_v60v
        with tempfile.TemporaryDirectory() as td:
            base = Path(td) / "benchmark_set"
            suite = base / "legacy_suite"
            (suite / "b001").mkdir(parents=True)
            (suite / "benchmark_set.json").write_text(json.dumps({"cases": [{"case_id": "b001"}]}), encoding="utf-8")
            (suite / "benchmark_plan.json").write_text(json.dumps({"runs": [{"id": "cpu"}]}), encoding="utf-8")
            (suite / "benchmark_suite.py").write_text("print('ok')\n", encoding="utf-8")
            assert benchmark_set_postcondition_v60v(base)["valid"] is True
            (suite / "benchmark_set.json").write_text('{"cases": []}', encoding="utf-8")
            assert benchmark_set_postcondition_v60v(base)["valid"] is False
    run("atomic_benchmark_set_postcondition", generation_postcondition)

    run("mutable_mapping_generation_regression", lambda: __import__("onnx_splitpoint_tool.benchmark.services", fromlist=["MutableMapping"]).MutableMapping)
    run("hailo8_engine_python_resolver", lambda: (_ for _ in ()).throw(AssertionError()) if "_candidate_engine_pythons" not in _script_asset("native_fifo_smoke_matrix.py").read_text(encoding="utf-8") else None)
    run("native_e2e_steps_csv", lambda: (_ for _ in ()).throw(AssertionError()) if "steps_json" not in _script_asset("native_producer_e2e_eval_runner.py").read_text(encoding="utf-8") else None)

    return checks


def run_smoke() -> dict[str, Any]:
    checks = _run_checks()
    failed = sum(row["status"] != "pass" for row in checks)
    return {
        "schema": "onnx-splitpoint/v60v-smoke",
        "tool_version": __version__,
        "workflow_version": WORKFLOW_VERSION,
        "status": "ok" if failed == 0 else "failed",
        "passed": len(checks) - failed,
        "failed": failed,
        "checks": checks,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="onnx-splitpoint-smoke-v60v")
    parser.add_argument("--json", default="")
    args = parser.parse_args(list(argv) if argv is not None else None)
    payload = run_smoke()
    if args.json:
        path = Path(args.json).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"v60v smoke: {'ok' if payload['status'] == 'ok' else 'FAILED'} ({payload['passed']} passed, {payload['failed']} failed)")
    return 0 if payload["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
