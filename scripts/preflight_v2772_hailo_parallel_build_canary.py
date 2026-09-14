#!/usr/bin/env python3
"""Fail-closed preflight for the v2.77.2 Hailo pair-build canary."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping


EXPECTED_VERSION = "2.77.2"
EXPECTED_PROFILE = "resnet50_v2772_hailo_parallel_build_canary"


def _plain(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _write(payload: Mapping[str, Any], path: Path | None) -> None:
    text = json.dumps(_plain(payload), ensure_ascii=False, indent=2, sort_keys=True)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text, flush=True)


def _probe_payload(result: Any) -> dict[str, Any]:
    value = _plain(result)
    if isinstance(value, Mapping):
        return dict(value)
    return {"ok": False, "reason": f"unexpected probe result: {type(result).__name__}"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--models-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    out_path = Path(args.json_out).expanduser().resolve() if args.json_out else None
    evidence: dict[str, Any] = {
        "schema": "onnx-splitpoint/v2772-hailo-parallel-canary-preflight/v1",
        "status": "FAIL",
        "profile": str(Path(args.profile).expanduser().resolve()),
    }

    try:
        from onnx_splitpoint_tool import __version__
        from onnx_splitpoint_tool.benchmark.services import (
            BenchmarkGenerationService,
            _hailo_pair_parallel_decision_v27550,
            _physical_hailo_targets_for_build,
            _v60s_build_scheduler_config,
        )
        from onnx_splitpoint_tool.hailo_backend import hailo_probe_via_venv
        from onnx_splitpoint_tool.workflow.legacy_benchmarkset_binding import (
            _hailo_full_hef_policy_v2772,
            _profile_build_scheduler_config,
            _profile_targets,
        )
        from onnx_splitpoint_tool.workflow.profile_options import (
            load_runtime_profile_snapshot,
            workflow_options_from_profile_snapshot,
        )

        _require(__version__ == EXPECTED_VERSION, f"tool_version:{__version__}!={EXPECTED_VERSION}")
        _require(
            str(
                os.environ.get("ONNX_SPLITPOINT_ENABLE_HAILO_SAME_BACKEND_SPLIT")
                or ""
            ).strip().lower()
            not in {"1", "true", "yes", "on"},
            "same_backend_hailo_split_enabled",
        )

        resolved, snapshot = load_runtime_profile_snapshot(args.profile)
        opts = workflow_options_from_profile_snapshot(
            profile_request=args.profile,
            out_root=args.out_root,
            start_snapshot=snapshot,
            models_root=args.models_root,
            required_run_mode="smoke",
            require_fresh_run=False,
        )

        profile_id = str(resolved.get("name") or "")
        _require(profile_id == EXPECTED_PROFILE, f"profile_id:{profile_id}")
        _require(
            not isinstance(resolved.get("execution_preset"), Mapping),
            "execution_preset_not_allowed",
        )
        _require(str(opts.execution_mode) == "generate_benchmarksets", "execution_mode_not_generate_benchmarksets")
        _require(bool(opts.skip_benchmarks), "runtime_benchmarks_not_disabled")
        _require(bool(opts.no_remote), "remote_execution_not_disabled")
        _require(not bool(opts.resume), "resume_not_allowed")
        _require(not bool(opts.dry_run), "dry_run_not_allowed")
        _require(
            str(opts.stop_after or "") == "build_backend_artifacts",
            f"stop_after:{opts.stop_after}",
        )
        _require(int(opts.benchmark_runs) == 1, "benchmark_runs_not_one")
        _require(int(opts.benchmark_timeout_s) == 0, "benchmark_timeout_not_zero")

        workflow_cfg = dict(resolved.get("workflow") or {})
        benchmark_cfg = dict(resolved.get("benchmark_execution") or {})
        remote_cfg = dict(resolved.get("remote_execution") or {})
        preparation_cfg = dict(resolved.get("model_preparation") or {})
        smoke_cfg = dict(resolved.get("hardware_smoke") or {})
        _require(
            workflow_cfg.get("parallel_remote_setups") is False,
            "parallel_remote_setups_enabled",
        )
        _require(workflow_cfg.get("max_parallel_setups") == 1, "parallel_setup_limit_mismatch")
        _require(workflow_cfg.get("max_parallel_uploads") == 0, "parallel_upload_limit_mismatch")
        _require(workflow_cfg.get("powercalc_workers") == 0, "powercalc_workers_enabled")
        _require(benchmark_cfg.get("runs") == 1, "benchmark_profile_runs_mismatch")
        _require(benchmark_cfg.get("warmup") == 0, "benchmark_profile_warmup_mismatch")
        _require(remote_cfg.get("enabled") is False, "remote_profile_enabled")
        _require(preparation_cfg.get("mode") == "current", "model_preparation_not_current")
        _require(smoke_cfg.get("mode") == "disabled", "hardware_smoke_not_disabled")

        _require(str(opts.hailo_build_mode) == "reuse_and_build_missing", "hailo_build_mode_mismatch")
        _require(str(opts.hailo_build_backend) == "auto", "hailo_backend_must_be_auto")
        _require(bool(opts.hailo_force_build), "hailo_force_build_not_enabled")
        _require(not bool(opts.hailo_build_full), "hailo_full_build_enabled")
        _require(bool(opts.hailo_build_part1), "hailo_part1_build_disabled")
        _require(not bool(opts.hailo_build_part2), "hailo_part2_build_enabled")
        _require(int(opts.hailo_build_timeout_s) == 21600, "hailo_timeout_mismatch")
        _require(int(opts.hailo_calib_count) == 8, "hailo_calibration_count_mismatch")
        _require(
            _hailo_full_hef_policy_v2772(bool(opts.hailo_build_full)) == "skip",
            "full_hef_policy_not_skip",
        )

        run_profiles = [
            row for row in list(resolved.get("run_profiles") or [])
            if isinstance(row, Mapping) and bool(row.get("enabled", True))
        ]
        run_profile_ids = [str(row.get("id") or "") for row in run_profiles]
        _require(
            run_profile_ids == ["hailo8_to_trt", "hailo10_to_tensorrt"],
            f"run_profiles:{run_profile_ids}",
        )
        _require(all(bool(row.get("required")) for row in run_profiles), "run_profile_not_required")

        forced = dict((resolved.get("selection_policy") or {}).get("forced_cases") or {})
        _require(forced == {"resnet50": ["b052"]}, f"forced_cases:{forced}")
        models = list((resolved.get("model_suite") or {}).get("primary") or [])
        _require(len(models) == 1 and str(models[0].get("id") or "") == "resnet50", "model_suite_not_exact")
        model_path = Path(str(models[0].get("onnx") or "")).expanduser().resolve()
        _require(model_path.is_file(), f"model_missing:{model_path}")

        manifests = dict((resolved.get("campaign") or {}).get("dataset_manifests") or {})
        classification = dict(manifests.get("classification") or {})
        calibration_manifest = Path(str(classification.get("calibration") or "")).expanduser().resolve()
        _require(calibration_manifest.is_file(), f"calibration_manifest_missing:{calibration_manifest}")
        json.loads(calibration_manifest.read_text(encoding="utf-8"))

        hailo_cfg = dict(resolved.get("hailo_build") or {})
        artifact_store = dict(resolved.get("artifact_store") or {})
        native_cfg = dict(resolved.get("native_producers") or {})
        energy_cfg = dict(resolved.get("energy") or {})
        ranking_cfg = dict(resolved.get("ranking_validation") or {})
        _require(hailo_cfg.get("cache_enabled") is False, "hailo_cache_not_disabled")
        _require(artifact_store.get("enabled") is False, "artifact_store_not_disabled")
        _require(artifact_store.get("register_hailo") is False, "artifact_store_hailo_registration_enabled")
        _require(native_cfg.get("enabled") is False, "native_execution_enabled")
        _require(energy_cfg.get("enabled") is False, "generic_energy_enabled")
        _require(energy_cfg.get("requested_native_energy") is False, "native_energy_requested")
        _require(ranking_cfg.get("enabled") is False, "ranking_enabled")
        validation_cfg = dict(resolved.get("validation") or {})
        _require(
            validation_cfg.get("split_fidelity_reference_mode") == "cpu_full",
            "validation_reference_not_cpu_full",
        )

        # Exercise the exact v2.77.2 plan projection before any compiler
        # process is launched. This catches a regression that silently
        # re-introduces a suite Full HEF through matrix variants.
        projected_plan = BenchmarkGenerationService().build_run_plan(
            acc_cpu=False,
            acc_cuda=False,
            acc_trt=True,
            acc_h8=True,
            acc_h10=True,
            acc_deepx=False,
            hailo8_hw="hailo8",
            hailo10_hw="hailo10",
            validation_reference_mode="cpu_full",
            benchmark_task="classification",
            hailo_preset="Custom",
            hailo_custom_full=False,
            hailo_custom_composed=True,
            hailo_custom_part1=True,
            hailo_custom_part2=False,
            matrix_trt_to_hailo=False,
            matrix_hailo_to_trt=True,
            matrix_deepx_to_trt=False,
            matrix_trt_to_deepx=False,
            full_hef_policy="skip",
        )
        _require(projected_plan.hef_full is False, "projected_plan_full_enabled")
        _require(projected_plan.hef_part1 is True, "projected_plan_part1_disabled")
        _require(projected_plan.hef_part2 is False, "projected_plan_part2_enabled")
        projected_hailo_runs = [
            row
            for row in list(projected_plan.bench_plan_runs or [])
            if isinstance(row, Mapping)
            and (
                str(row.get("type") or "") == "hailo"
                or any(
                    isinstance(row.get(stage), Mapping)
                    and str(row[stage].get("type") or "") == "hailo"
                    for stage in ("stage1", "stage2")
                )
            )
        ]
        _require(
            all(
                "full" not in {
                    str(value).strip().lower()
                    for value in list(row.get("variants") or [])
                }
                for row in projected_hailo_runs
            ),
            "projected_plan_contains_full_variant",
        )

        logical_targets = _profile_targets(resolved, list(opts.hailo_build_targets or []))
        physical_targets = _physical_hailo_targets_for_build(logical_targets)
        _require(physical_targets == ["hailo8", "hailo10"], f"physical_targets:{physical_targets}")

        scheduler_raw = _profile_build_scheduler_config(resolved)
        scheduler = _v60s_build_scheduler_config(scheduler_raw, mode="smoke")
        expected_scheduler = {
            "enabled": True,
            "max_workers": 2,
            "cpu_tokens": 8,
            "ram_mb": 12288,
            "ram_reserve_mb": 2048,
        }
        for key, expected in expected_scheduler.items():
            _require(scheduler.get(key) == expected, f"scheduler_{key}:{scheduler.get(key)}!={expected}")
        _require((os.cpu_count() or 0) >= 8, f"host_cpu_count:{os.cpu_count() or 0}<8")

        probes: dict[str, Any] = {}
        for target in physical_targets:
            probe = _probe_payload(hailo_probe_via_venv(hw_arch=target, timeout_s=120))
            probes[target] = probe
            _require(probe.get("ok") is True, f"managed_venv_probe_failed:{target}:{probe.get('reason')}")

        decision = _hailo_pair_parallel_decision_v27550(
            physical_targets,
            scheduler,
            backend=str(opts.hailo_build_backend),
        )
        _require(decision.get("requested") is True, f"pair_not_requested:{decision}")
        _require(decision.get("effective") is True, f"pair_not_effective:{decision.get('reason')}")
        _require(decision.get("reason") == "resources_available", f"pair_reason:{decision.get('reason')}")
        _require(decision.get("effective_backend") == "venv", f"effective_backend:{decision.get('effective_backend')}")
        _require(int(decision.get("cpu_required") or 0) == 8, "pair_cpu_requirement_mismatch")
        _require(int(decision.get("ram_required_mb") or 0) == 12288, "pair_ram_requirement_mismatch")

        evidence.update(
            {
                "status": "PASS",
                "tool_version": __version__,
                "profile_id": profile_id,
                "model": str(model_path),
                "calibration_manifest": str(calibration_manifest),
                "run_profiles": run_profile_ids,
                "forced_cases": forced,
                "logical_targets": logical_targets,
                "physical_targets": physical_targets,
                "scheduler": scheduler,
                "pair_decision": decision,
                "projected_plan": {
                    "hef_full": bool(projected_plan.hef_full),
                    "hef_part1": bool(projected_plan.hef_part1),
                    "hef_part2": bool(projected_plan.hef_part2),
                    "hailo_run_ids": [
                        str(row.get("id") or "") for row in projected_hailo_runs
                    ],
                    "hailo_run_variants": {
                        str(row.get("id") or ""): list(row.get("variants") or [])
                        for row in projected_hailo_runs
                    },
                },
                "managed_venv_probes": probes,
                "contract": {
                    "part1_only": True,
                    "stop_after": "build_backend_artifacts",
                    "execution_preset_present": False,
                    "force_build": True,
                    "hailo_cache_enabled": False,
                    "artifact_store_enabled": False,
                    "runtime_enabled": False,
                    "native_enabled": False,
                    "quality_evidence": False,
                    "ranking_enabled": False,
                    "energy_enabled": False,
                },
            }
        )
        _write(evidence, out_path)
        return 0
    except Exception as exc:
        evidence["error"] = f"{type(exc).__name__}: {exc}"
        _write(evidence, out_path)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
