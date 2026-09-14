from __future__ import annotations

"""Fast, hardware-free smoke checks for v60o performance/native-energy rules."""

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any, Callable, Sequence

from .benchmark.remote_run import _extract_run_ids_from_add_args, _filter_benchmark_plan_for_run_ids
from .campaign import create_dataset_manifest, verify_dataset_manifest
from .execution_plan import build_effective_execution_plan
from .run_modes import apply_run_mode, default_run_modes_config
from .v60m_policy import normalize_profile
from .workflow.legacy_benchmarkset_binding import _infer_run_switches
from .workflow.execution_binding import _energy_enabled_for_profile
from .workflow.results import _best_full_latency


def _base_profile(mode: str = "standard", *, native: bool = True, energy: bool = False) -> dict[str, Any]:
    return {
        "name": "v60o_smoke",
        "selection_policy": {
            "max_accepted_cases_per_model": 1,
            "preferred_shortlist": 5,
            "min_gap": 1,
            "candidate_search_pool": "auto",
            "selection_strategy": "stratified_windows",
        },
        "model_suite": {
            "primary": [
                {"id": "resnet50", "task": "classification", "evaluation_role": "development", "enabled": True},
                {"id": "yolo26s", "task": "detection", "evaluation_role": "development", "enabled": True},
            ]
        },
        "run_profiles": [
            {"id": "hailo8", "full": "hailo8", "stage1": "hailo8", "stage2": "hailo8", "enabled": True},
            {"id": "hailo8_to_trt", "stage1": "hailo8", "stage2": "tensorrt", "enabled": True},
            {"id": "deepx_m1", "full": "deepx_m1", "stage1": "deepx_m1", "stage2": "deepx_m1", "enabled": True},
        ],
        "execution_preset": {
            "id": mode,
            "follow_tool_config": False,
            "snapshot": default_run_modes_config()["modes"][mode],
            "overrides": {"native_enabled": native, "energy_enabled": energy},
        },
    }


def _run_checks() -> list[dict[str, Any]]:
    checks: list[tuple[str, Callable[[], None]]] = []

    def check(name: str):
        def deco(fn: Callable[[], None]) -> Callable[[], None]:
            checks.append((name, fn))
            return fn
        return deco

    @check("generic_energy_is_hard_disabled")
    def _() -> None:
        profile, _audit = apply_run_mode(_base_profile("final", native=True, energy=True))
        assert profile["energy"]["enabled"] is False
        assert profile["energy"]["generic_enabled"] is False
        assert profile["energy"]["measurement_path"] == "native_only"
        assert _energy_enabled_for_profile(object(), profile) is False

    @check("native_energy_follows_native_and_energy_switches")
    def _() -> None:
        enabled, _ = apply_run_mode(_base_profile("final", native=True, energy=True))
        native_off, _ = apply_run_mode(_base_profile("final", native=False, energy=True))
        energy_off, _ = apply_run_mode(_base_profile("final", native=True, energy=False))
        assert enabled["native_producers"]["energy"]["enabled"] is True
        assert native_off["native_producers"]["energy"]["enabled"] is False
        assert energy_off["native_producers"]["energy"]["enabled"] is False

    @check("native_energy_survives_profile_normalization")
    def _() -> None:
        profile, _ = apply_run_mode(_base_profile("final", native=True, energy=True))
        normalized, audit = normalize_profile(profile)
        assert normalized["native_producers"]["energy"]["enabled"] is True
        assert audit["energy"]["master_enabled"] is True

    @check("unused_hailo_part2_build_is_not_requested")
    def _() -> None:
        profile = _base_profile("standard", native=True, energy=False)
        switches = _infer_run_switches(profile, [])
        assert switches["hailo_part1_requested"] is True
        assert switches["hailo_part2_requested"] is False

    @check("standard_uses_sampled_integrity_and_500_items")
    def _() -> None:
        profile, _ = apply_run_mode(_base_profile("standard", native=True, energy=False))
        assert profile["integrity_policy"]["mode"] == "relaxed"
        assert profile["integrity_policy"]["dataset_sample_size"] > 0
        assert profile["validation_execution"]["max_items"] == {"classification": 500, "detection": 500}
        assert profile["hailo_build"]["calib_count"] == 500
        assert profile["hailo_build"]["calibration_storage"] == "memmap"

    @check("final_quality_uses_standard_integrity")
    def _() -> None:
        profile, _ = apply_run_mode(_base_profile("final", native=True, energy=True))
        assert profile["integrity_policy"]["mode"] == "relaxed"
        assert profile["integrity_policy"]["dataset_sample_size"] == 24
        assert profile["campaign"]["mode"] == "development"

    @check("sampled_manifest_verification")
    def _() -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "images"
            root.mkdir()
            for idx in range(20):
                (root / f"n{idx:04d}.jpg").write_bytes((f"sample-{idx}" * 4).encode("utf-8"))
            manifest_path = Path(temp) / "manifest.json"
            create_dataset_manifest(
                task="classification", role="validation", dataset_id="v60o", split="val",
                root=root, output=manifest_path,
            )
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            result = verify_dataset_manifest(payload, verification_mode="sampled", sample_size=5)
            assert result["ok"] is True
            assert result["checked_item_count"] == 5

    @check("setup_local_run_id_batching")
    def _() -> None:
        ids = _extract_run_ids_from_add_args("--run-ids hailo8,hailo8_to_trt")
        assert ids == ["hailo8", "hailo8_to_trt"]
        plan = {"runs": [{"id": "hailo8"}, {"id": "hailo8_to_trt"}, {"id": "deepx_m1"}]}
        filtered = _filter_benchmark_plan_for_run_ids(plan, ids)
        assert [row["id"] for row in filtered["runs"]] == ids

    @check("effective_plan_exposes_batching_and_native_energy")
    def _() -> None:
        profile, _ = apply_run_mode(_base_profile("standard", native=True, energy=True))
        plan = build_effective_execution_plan(profile)
        assert plan["generic_energy_enabled"] is False
        assert plan["native_energy_enabled"] is True
        assert plan["batched_remote_dispatches_total"] < plan["generic_rows_total"]
        assert plan["uploads_per_model_setup"] == 1

    @check("tensorrt_full_uses_its_own_timing")
    def _() -> None:
        row = {"latency_ms": 100.0, "timings": {"composed": {"mean_ms": 100.0}, "full": {"mean_ms": 9.0}}}
        total, raw, _e2e, status = _best_full_latency(row)
        assert total == 9.0 and raw == 9.0 and status == "full"

    @check("runtime_template_contains_safe_bootstrap_shortcuts")
    def _() -> None:
        source = (Path(__file__).parent / "resources" / "templates" / "run_split_onnxruntime.py.txt").read_text(encoding="utf-8")
        assert "point_estimate_below_non_inferiority_margin" in source
        assert "candidate_reference_identical" in source
        assert "bootstrap_repetitions_requested" in source
        assert "paired_detection_cached_matching_v1" in source

    @check("native_outputs_are_tagged_and_merged")
    def _() -> None:
        resources = Path(__file__).parent / "resources" / "remote_scripts"
        runner = (resources / "native_producer_e2e_eval_runner.py").read_text(encoding="utf-8")
        report = (resources / "native_producer_final_report.py").read_text(encoding="utf-8")
        assert "--analysis-tag" in runner
        assert "native_hailo10h_producer_e2e_eval*.json" in report

    results: list[dict[str, Any]] = []
    for name, fn in checks:
        try:
            fn()
            results.append({"name": name, "status": "pass"})
        except Exception as exc:
            results.append({"name": name, "status": "fail", "error": f"{type(exc).__name__}: {exc}"})
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="onnx-splitpoint-smoke-v60o", description="Run fast v60o performance/native-energy checks without hardware.")
    parser.add_argument("--json", default="", help="Optional JSON output path.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    results = _run_checks()
    passed = sum(row["status"] == "pass" for row in results)
    failed = len(results) - passed
    payload = {
        "schema": "onnx-splitpoint/v60o-smoke",
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
    print(f"v60o smoke: {'ok' if failed == 0 else 'FAILED'} ({passed} passed, {failed} failed)")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
