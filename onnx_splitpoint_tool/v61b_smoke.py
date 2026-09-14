from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from . import __version__
from .native_energy_reporting import build_native_energy_pairs, scientific_energy_rows
from .native_full_quality import normalise_evaluation_profile
from .native_progress import NativeProgressJournal, run_streaming
from .workflow.runner import (
    WORKFLOW_VERSION,
    _native_expected_full_rows_v61b,
    _native_full_backends_by_producer_v61b,
)

EXPECTED_VERSION = ("0.14.27+v61b.nativeintegrationfix", "2.61.0+v61e", "2.62.0", "2.63.0", "2.64.0", "2.65.0", "2.66.0", "2.67.0", "2.68.0", "2.69.6", "2.70.6", "2.70.7", "2.70.8", "2.70.9", "2.70.10", "2.71.1", "2.71.2", "2.71.3", "2.71.4", "2.72.0", "2.72.1", "2.72.2", "2.72.3", "2.72.4", "2.72.5", "2.72.6", "2.73.0", "2.73.6", "2.73.7", "2.75.2", "2.75.7", "2.75.8", "2.75.10", "2.75.11", "2.75.12", "2.75.14", "2.75.15", "2.75.16", "2.75.17", "2.75.20", "2.75.21", "2.75.22", "2.75.24", "2.75.26")
EXPECTED_WORKFLOW = (
    "v61b-native-integration-live-energy-fixes",
    "v2.61e-campaign-contract-hardening",
    "v2.62-window-validation-native-binding",
    "v2.63-campaign-ready",
    "v2.64-campaign-ready", "v2.65-campaign-ready", "v2.66-campaign-ready", "v2.67-campaign-ready", "v2.68-level-playing-field", "v2.69f-hardware-smoke-native-energy-repair", "v2.70f-legacy-log-callback-repair", "v2.70g-native-validation-bridge-repair", "v2.70h-remote-suite-bootstrap-repair", "v2.70i-native-full-evidence-repair", "v2.70j-native-evidence-reporting-repair", "v2.71.1-live-evidence-binding-repair", "v2.71.2-evidence-status-energy-resume-repair", "v2.71.3-resume-artifact-restage-repair", "v2.71.4-resume-cleanup-root-rehydration", "v2.72.0-completed-detection-evidence-contracts", "v2.72.1-mixed-runtime-energy-contract-repair", "v2.72.2-offline-energy-completed-consumer-repair", "v2.72.3-campaign-gate-quality-binding-repair", "v2.72.4-campaign-gate-quality-contract-mirror-repair", "v2.72.5-campaign-gate-hailo8-resume-contract-repair", "v2.72.6-campaign-gate-hailo8-preflight-artifact-rehydration-repair", "v2.73.0-completed-quality-evidence-repair", "v2.73.6-single-writer-process-tree-cancellation", "v2.73.7-atomic-stage-energy-checkpoints", "v2.75.2-native-runtime-closure-partial-energy-repair", "v2.75.3-cache-verify-only-gate", "v2.75.7-cache-canary-remote-runtime-closure-repair", "v2.75.8-native-runtime-holdout-contract-repair", "v2.75.10-historical-cache-multihit-replay-repair", "v2.75.11-canonical-native-result-verification-repair", "v2.75.12-canonical-native-result-status-alias-repair", "v2.75.14-vendor-full-nine-row-repair", "v2.75.15-vendor-full-field-contract-repair", "v2.75.16-read-only-run-storage-admission", "v2.75.17-vendor-quality-diagnostic-hailo-alias-repair", "v2.75.20-remote-boundary-plan-finalization-repair", "v2.75.21-setup-local-tensorrt-quality-dispatch-repair", "v2.75.22-quality-replay-reporting-repair", "v2.75.24-standard-hef-preupload-gate", "v2.75.26-trt-quality-join-mean-iou-repair",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Hardware-free v61b integration smoke test")
    parser.add_argument("--json", default="")
    args = parser.parse_args()
    checks: list[dict[str, object]] = []

    def check(name, fn):
        try:
            fn()
            checks.append({"name": name, "ok": True})
        except Exception as exc:  # pragma: no cover - user-visible diagnostics
            checks.append({"name": name, "ok": False, "error": f"{type(exc).__name__}: {exc}"})

    check("version", lambda: (_ for _ in ()).throw(AssertionError(__version__)) if __version__ not in EXPECTED_VERSION else None)
    check("workflow", lambda: (_ for _ in ()).throw(AssertionError(WORKFLOW_VERSION)) if WORKFLOW_VERSION not in EXPECTED_WORKFLOW else None)

    def profile_resolution() -> None:
        profile = {
            "run_profiles": [
                {"id": "ort_tensorrt", "full": "tensorrt"},
                {"id": "hailo8", "full": "hailo8"},
                {"id": "hailo8_to_trt"},
                {"id": "hailo10", "full": "hailo10"},
                {"id": "hailo10_to_tensorrt"},
                {"id": "deepx_m1_full", "full": "deepx_m1"},
                {"id": "deepx_m1_to_tensorrt"},
            ],
            "native_producers": {"enabled": True, "energy": {"enabled": True}},
            "execution_preset": {"overrides": {"native_enabled": True, "energy_enabled": True}},
            "energy": {},
        }
        normalise_evaluation_profile(profile)
        expected = {
            "hailo8": ["hailo8", "tensorrt"],
            "hailo10h": ["hailo10h", "tensorrt"],
            "deepx": ["deepx", "tensorrt"],
        }
        actual = profile["native_producers"]["full_baselines"]["backends_by_producer"]
        assert actual == expected, actual
        energy = profile["native_producers"]["energy"]
        assert energy["enabled"] and energy["mode"] == "measure"
        assert energy["include_split_rows"] and energy["include_full_baselines"]

    check("profile_resolution", profile_resolution)

    def matrix() -> None:
        full_cfg = {
            "enabled": True,
            "backends_by_producer": {
                "hailo8": ["hailo8", "tensorrt"],
                "hailo10h": ["hailo10h", "tensorrt"],
                "deepx": ["deepx", "tensorrt"],
            },
        }
        mapping = _native_full_backends_by_producer_v61b(full_cfg, ["hailo8", "hailo10h", "deepx"])
        rows = _native_expected_full_rows_v61b(
            ["resnet50", "yolo26s"],
            mapping,
            {"hailo8": "h8", "hailo10h": "h10", "deepx": "dx"},
        )
        assert len(rows) == 12, len(rows)
        assert len({(r["setup_id"], r["model"], r["backend"]) for r in rows}) == 12

    check("producer_local_full_matrix", matrix)

    def silent_heartbeat() -> None:
        seen: list[str] = []
        with tempfile.TemporaryDirectory() as td:
            proc = run_streaming(
                [sys.executable, "-u", "-c", "import time; time.sleep(0.12)"],
                label="v61b-silent",
                heartbeat_s=0.03,
                journal=NativeProgressJournal(Path(td) / "native_progress.jsonl", Path(td) / "native_progress.json"),
                line_callback=seen.append,
            )
            assert proc.returncode == 0
            assert any("HEARTBEAT" in line for line in seen), seen
            assert (Path(td) / "native_progress.jsonl").is_file()

    check("parent_heartbeat", silent_heartbeat)

    def energy_ingestion() -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            src = run_dir / "reports" / "native_energy_measurements"
            src.mkdir(parents=True)
            payload_rows = []
            for backend, mode, energy, power in [
                ("hailo8_to_trt", "native_split", 0.8, 8.0),
                ("native_full_hailo8", "native_full_baseline", 1.0, 10.0),
                ("native_full_tensorrt", "native_full_baseline", 0.9, 9.0),
            ]:
                payload_rows.append({
                    "ok": True,
                    "row": {
                        "setup_id": "h8", "model": "m", "backend": backend,
                        "execution_mode": mode, "energy_scope": "full_system",
                        "energy_window": "command_window",
                    },
                    "run": {
                        "rc": 0,
                        "stdout": json.dumps({
                            "energy_per_inference_j": energy, "avg_power_w": power,
                            "energy_work_units_used": 100,
                        }),
                    },
                })
            (src / "native_producer_energy_results.json").write_text(
                json.dumps({"rows": payload_rows}), encoding="utf-8"
            )
            scientific = scientific_energy_rows(run_dir)
            assert len(scientific) == 3, scientific
            from .native_energy_reporting import collect_native_energy
            pairs = build_native_energy_pairs(collect_native_energy(run_dir))
            assert len(pairs) == 2, pairs
            assert {p["baseline_kind"] for p in pairs} == {"vendor_full", "tensorrt_full"}

    check("native_energy_ingestion_and_pairing", energy_ingestion)

    result = {
        "ok": all(bool(row["ok"]) for row in checks),
        "passed": sum(bool(row["ok"]) for row in checks),
        "failed": sum(not bool(row["ok"]) for row in checks),
        "checks": checks,
    }
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"v61b smoke: {'ok' if result['ok'] else 'failed'} ({result['passed']} passed, {result['failed']} failed)")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
