from __future__ import annotations

"""Fast console smoke tests for the v60n run-mode abstraction."""

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Sequence

import yaml

from .benchmark.evaluation_profiles import validate_evaluation_profile_payload
from .run_modes import apply_run_mode, default_run_modes_config, mode_summary, validate_run_modes_config
from .workflow.hardware_matrix import normalize_hardware_targets


def _base_profile(mode: str, *, native: bool | None = None, energy: bool | None = None) -> dict[str, Any]:
    overrides: dict[str, bool] = {}
    if native is not None:
        overrides["native_enabled"] = native
    if energy is not None:
        overrides["energy_enabled"] = energy
    return {
        "name": f"v60n_{mode}_smoke",
        "purpose": "v60n smoke",
        "selection_policy": {
            "max_accepted_cases_per_model": 1,
            "preferred_shortlist": 3,
            "min_gap": 1,
            "candidate_search_pool": "auto",
            "selection_strategy": "stratified_windows",
        },
        "model_suite": {"primary": [{"id": "resnet50", "task": "classification", "evaluation_role": "development"}]},
        "run_profiles": [
            {"id": "ort_tensorrt", "type": "same_backend_reference", "full": "tensorrt", "stage1": "tensorrt", "stage2": "tensorrt", "required": True},
            {"id": "hailo8_to_trt", "type": "mixed_backend", "stage1": "hailo8", "stage2": "tensorrt", "required": False},
        ],
        "execution_preset": {"id": mode, "follow_tool_config": False, "snapshot": default_run_modes_config()["modes"][mode], "overrides": overrides},
    }


def _run_checks() -> list[dict[str, Any]]:
    checks: list[tuple[str, Callable[[], None]]] = []

    def check(name: str):
        def deco(fn: Callable[[], None]) -> Callable[[], None]:
            checks.append((name, fn)); return fn
        return deco

    @check("default_registry_valid")
    def _() -> None:
        cfg = validate_run_modes_config(default_run_modes_config())
        assert list(cfg["modes"]) == ["smoke", "standard", "final"]
        assert cfg["default_mode"] == "standard"

    @check("smoke_is_ultra_light")
    def _() -> None:
        profile, _ = apply_run_mode(_base_profile("smoke", native=False, energy=False))
        assert profile["execution_preset"]["effective"]["calibration_items"] == {"classification": 8, "detection": 8}
        assert profile["validation_execution"]["max_items"]["classification"] <= 32
        assert profile["quality_gate"]["statistics"]["bootstrap_repetitions"] <= 50
        assert profile["benchmark_execution"]["runs"] == 1
        assert profile["integrity_policy"]["mode"] == "fast"

    @check("standard_is_balanced")
    def _() -> None:
        profile, _ = apply_run_mode(_base_profile("standard", native=True, energy=False))
        assert profile["validation_execution"]["max_items"] == {"classification": 500, "detection": 500}
        assert profile["quality_gate"]["statistics"]["bootstrap_repetitions"] == 500
        assert profile["hailo_build"]["optimization_level"] == 1
        assert profile["native_producers"]["enabled"] is True

    @check("final_quality_is_standard_path_5000")
    def _() -> None:
        profile, _ = apply_run_mode(_base_profile("final", native=True, energy=True))
        assert profile["validation_execution"]["max_items"] == {"classification": 5000, "detection": 5000}
        assert profile["quality_gate"]["statistics"]["bootstrap_repetitions"] == 5000
        assert profile["integrity_policy"]["mode"] == "relaxed"
        assert profile["integrity_policy"]["dataset_sample_size"] == 24
        assert profile["campaign"]["mode"] == "development"
        assert profile["benchmark_execution"] == {"provider": "auto", "warmup": 3, "runs": 5, "timeout_s": 0}
        assert profile["native_producers"]["repetitions"] == 3
        assert profile["official_coco_evaluation"]["required_for_final"] is False

    @check("energy_master_disables_native_energy")
    def _() -> None:
        profile, _ = apply_run_mode(_base_profile("final", native=True, energy=False))
        assert profile["energy"]["enabled"] is False
        assert profile["native_producers"]["enabled"] is True
        assert profile["native_producers"]["energy"]["enabled"] is False
        assert profile["native_producers"]["energy"]["mode"] == "plan"

    @check("profile_has_no_remote_host_complexity")
    def _() -> None:
        profile, _ = apply_run_mode(_base_profile("standard"))
        for key in ("remote_execution", "hardware_setups", "hardware_groups", "hardware_targets", "build_environments"):
            assert key not in profile
        assert profile["hardware"] == {"selected_setups": [], "selected_groups": []}

    @check("central_hardware_registry_is_derived")
    def _() -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            registry = root / "hardware_setups.yaml"
            registry.write_text(yaml.safe_dump({
                "hardware_setups": [{
                    "id": "test_hailo8",
                    "label": "test Hailo-8",
                    "accelerator": "hailo8",
                    "enabled": True,
                    "host": {"address": "192.0.2.10", "user": "nx", "port": 22, "base_dir": "~/runs"},
                    "runtime": {"provider": "hailo8", "activate": "source ~/venv/bin/activate"},
                    "build": {"environment_id": "h8"},
                }],
                "build_environments": [{"id": "h8", "kind": "hailo8_dfc"}],
            }, sort_keys=False), encoding="utf-8")
            profile, _ = apply_run_mode(_base_profile("smoke"))
            profile["hardware"]["setups_file"] = str(registry)
            targets = normalize_hardware_targets(profile)
            assert len(targets) == 1
            assert targets[0]["id"] == "test_hailo8"
            assert targets[0]["runtime"]["host"] == "192.0.2.10"

    @check("schema_accepts_materialized_profile")
    def _() -> None:
        profile, _ = apply_run_mode(_base_profile("standard"))
        validate_evaluation_profile_payload(profile, source="v60n smoke")

    @check("mode_summary_is_readable")
    def _() -> None:
        text = mode_summary("final", default_run_modes_config())
        assert "5000" in text
        assert "Reproducibility: strict" in text

    results: list[dict[str, Any]] = []
    for name, fn in checks:
        try:
            fn()
            results.append({"name": name, "status": "pass"})
        except Exception as exc:
            results.append({"name": name, "status": "fail", "error": f"{type(exc).__name__}: {exc}"})
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="onnx-splitpoint-smoke-v60n", description="Run fast v60n run-mode checks without hardware.")
    parser.add_argument("--json", default="", help="Optional JSON output path.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    results = _run_checks()
    passed = sum(row["status"] == "pass" for row in results)
    failed = len(results) - passed
    payload = {"schema": "onnx-splitpoint/v60n-smoke", "status": "ok" if failed == 0 else "failed", "passed": passed, "failed": failed, "checks": results}
    if str(args.json or "").strip():
        path = Path(args.json).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for row in results:
        suffix = "" if row["status"] == "pass" else f" — {row.get('error', '')}"
        print(f"[{row['status'].upper()}] {row['name']}{suffix}")
    print(f"v60n smoke: {'ok' if failed == 0 else 'FAILED'} ({passed} passed, {failed} failed)")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
