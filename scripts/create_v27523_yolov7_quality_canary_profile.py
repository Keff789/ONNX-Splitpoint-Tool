#!/usr/bin/env python3
from __future__ import annotations

"""Create the sealed YOLOv7 Hailo-8/Hailo-10H Full-only Quality profile.

The profile deliberately remains a development/acceptance campaign while it
uses the final-depth Detection sample, Hailo optimiser and bootstrap axes.  It
therefore exercises the intended hardware/Quality contract without silently
activating unrelated final-campaign freeze, ranking or Energy requirements.
"""

import argparse
import copy
from datetime import datetime
from pathlib import Path
import sys
from typing import Any, Mapping

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from onnx_splitpoint_tool.benchmark.evaluation_profiles import (
    load_evaluation_profile,
    validate_evaluation_profile_payload,
)
from onnx_splitpoint_tool.execution_plan import build_effective_execution_plan
from onnx_splitpoint_tool.run_modes import (
    _legacy_v11_final_mode,
    apply_run_mode,
    default_run_modes_config,
    default_run_modes_path,
)
from onnx_splitpoint_tool.workflow.full_only_quality_canary import (
    resolve_full_only_quality_canary,
)


H8_SETUP = "orin_nx_hailo8_01"
H10_SETUP = "orin_nx_hailo10_01"
VALIDATION_ITEMS = 5000
CALIBRATION_ITEMS = 500
BOOTSTRAP_REPETITIONS = 5000


def _expected_full_quality_identities() -> list[dict[str, Any]]:
    """Return the exact, ordered scientific identity contract for this run."""

    common = {
        "variant": "full",
        "execution_role": "full_quality_only",
        "performance_claims_emitted": False,
    }
    return [
        {
            "id": "hailo8_full",
            "source_run_id": "hailo8",
            "run_id": "hailo8",
            "dispatch_run_id": "hailo8",
            "setup_id": H8_SETUP,
            "backend": "hailo8",
            **common,
        },
        {
            "id": "tensorrt_at_hailo8_full",
            "source_run_id": "native_full_tensorrt",
            "run_id": "native_full_tensorrt",
            "dispatch_run_id": "ort_tensorrt",
            "setup_id": H8_SETUP,
            "backend": "tensorrt",
            **common,
        },
        {
            "id": "hailo10h_full",
            "source_run_id": "hailo10",
            "run_id": "hailo10",
            "dispatch_run_id": "hailo10",
            "setup_id": H10_SETUP,
            "backend": "hailo10h",
            **common,
        },
        {
            "id": "tensorrt_at_hailo10h_full",
            "source_run_id": "native_full_tensorrt",
            "run_id": "native_full_tensorrt",
            "dispatch_run_id": "ort_tensorrt",
            "setup_id": H10_SETUP,
            "backend": "tensorrt",
            **common,
        },
    ]


def _profile_id() -> str:
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    return f"yolov7_full_only_quality_canary_v27523_{stamp}"


def _frozen_acceptance_mode() -> dict[str, Any]:
    config = default_run_modes_config()
    mode = copy.deepcopy(config["modes"]["standard"])
    # Retain the v2.75.23 canary's explicitly frozen Hailo depth.  The
    # user-facing schema-v12 Final Quality mode intentionally uses Standard
    # compiler settings and must not be overloaded to satisfy this historical
    # specialist profile.
    final_mode = _legacy_v11_final_mode()

    mode["label"] = "YOLOv7 Full-only Quality acceptance"
    mode["description"] = (
        "Development acceptance with the final-depth Detection, Hailo and "
        "paired-bootstrap axes; no Native, Energy, ranking or performance claims."
    )
    mode["recommended_for"] = (
        "Hailo-8/Hailo-10H Full Quality versus setup-local TensorRT."
    )
    mode["defaults"] = {
        "native_enabled": False,
        "energy_enabled": False,
    }
    mode["data"]["calibration_items"] = {
        "classification": CALIBRATION_ITEMS,
        "detection": CALIBRATION_ITEMS,
    }
    mode["data"]["validation_items"]["detection"] = VALIDATION_ITEMS
    mode["build"]["hailo"] = copy.deepcopy(final_mode["build"]["hailo"])
    mode["quality"].update(
        {
            "profile_id": "task_quality_development_5000_v27523",
            "dataset_tier": "screening",
            "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
            "official_coco_enabled": False,
            "official_coco_required": False,
            "archive_coco_eval_tensors": False,
        }
    )
    mode["ranking"]["enabled"] = False
    return mode


def build_profile(*, profile_id: str, models_root: Path) -> dict[str, Any]:
    model_path = (models_root / "yolov7_paper.onnx").resolve(strict=True)
    profile: dict[str, Any] = {
        "name": profile_id,
        "purpose": (
            "YOLOv7 Full-only Quality acceptance: Hailo-8/Hailo-10H versus "
            "setup-local TensorRT; 5000 COCO validation images, AP50:95/AP50/AP75, "
            "Native and Energy disabled."
        ),
        "models_root_hint": str(models_root),
        "selection_policy": {
            "max_accepted_cases_per_model": 1,
            "preferred_shortlist": 1,
            "min_gap": 1,
            "candidate_search_pool": "auto",
            "require_single_part2_input": False,
            "selection_strategy": "stratified_windows",
            "report_blocked_configs": True,
            "report_plan_adjustments": True,
            "keep_partial_hailo_cases": False,
            "full_model_hailo_preflight_policy": "skip",
        },
        "model_suite": {
            "primary": [
                {
                    "id": "yolov7_paper",
                    "task": "detection",
                    "family": "yolov7",
                    "source": "ultralytics",
                    "semantic_dataset": "coco2017_val",
                    "development_subset": "coco_50",
                    "input_shape": [1, 3, 640, 640],
                    "onnx": str(model_path),
                    "evaluation_role": "development",
                    "validation_tier": "screening",
                    "candidate_universe_complete": False,
                    "candidate_universe": {
                        "mode": "declared_shortlist",
                        "seed": 20260710,
                    },
                }
            ],
            "reserve": [],
        },
        "run_profiles": [
            {
                "id": "ort_tensorrt",
                "type": "same_backend_reference",
                "full": "tensorrt",
                "stage1": "tensorrt",
                "stage2": "tensorrt",
                "required": True,
                "enabled": True,
            },
            {
                "id": "hailo8",
                "type": "same_backend_reference",
                "full": "hailo8",
                "stage1": "hailo8",
                "stage2": "hailo8",
                "required": True,
                "treat_preflight_block_as_result": True,
                "enabled": True,
            },
            {
                "id": "hailo10",
                "type": "same_backend_reference",
                "full": "hailo10h",
                "stage1": "hailo10h",
                "stage2": "hailo10h",
                "required": True,
                "treat_preflight_block_as_result": True,
                "enabled": True,
            },
        ],
        "execution_preset": {
            "id": "standard",
            "follow_tool_config": False,
            "config_path": str(default_run_modes_path()),
            "overrides": {
                "native_enabled": False,
                "energy_enabled": False,
            },
            "snapshot": _frozen_acceptance_mode(),
        },
        "quality_canary": {
            "enabled": True,
            "execution_scope": "full_only",
            "full_run_ids": [
                {
                    "id": "hailo8_full",
                    "run_id": "hailo8",
                    "setup_id": H8_SETUP,
                    "backend": "hailo8",
                    "variant": "full",
                    "execution_role": "full_quality_only",
                    "performance_claims_emitted": False,
                },
                {
                    "id": "hailo10h_full",
                    "run_id": "hailo10",
                    "setup_id": H10_SETUP,
                    "backend": "hailo10h",
                    "variant": "full",
                    "execution_role": "full_quality_only",
                    "performance_claims_emitted": False,
                },
            ],
            "setup_local_tensorrt_companions": [
                {
                    "id": "tensorrt_at_hailo8_full",
                    "run_id": "ort_tensorrt",
                    "setup_id": H8_SETUP,
                    "backend": "tensorrt",
                    "variant": "full",
                    "execution_role": "full_quality_only",
                    "performance_claims_emitted": False,
                },
                {
                    "id": "tensorrt_at_hailo10h_full",
                    "run_id": "ort_tensorrt",
                    "setup_id": H10_SETUP,
                    "backend": "tensorrt",
                    "variant": "full",
                    "execution_role": "full_quality_only",
                    "performance_claims_emitted": False,
                },
            ],
        },
    }
    resolved, _audit = apply_run_mode(
        profile, config=default_run_modes_config()
    )
    return validate_evaluation_profile_payload(resolved)


def verify_profile(profile: Mapping[str, Any]) -> dict[str, Any]:
    canary = resolve_full_only_quality_canary(
        profile, plan_rows=list(profile.get("run_profiles") or [])
    )
    if canary.get("ok") is not True:
        raise ValueError(
            "Full-only Quality contract is invalid: "
            + ", ".join(str(item) for item in canary.get("errors") or [])
        )
    plan = build_effective_execution_plan(profile)
    expected = {
        "quality_canary_enabled": True,
        "generic_rows_total": 0,
        "expected_full_quality_results_total": 4,
        "remote_run_invocations_total": 2,
        "native_enabled": False,
        "native_energy_enabled": False,
        "performance_claims_emitted": False,
    }
    mismatches = {
        key: {"expected": value, "observed": plan.get(key)}
        for key, value in expected.items()
        if plan.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Full-only effective plan mismatch: {mismatches}")
    if list(plan.get("models") or []) != ["yolov7_paper"]:
        raise ValueError(f"Unexpected model plan: {plan.get('models')!r}")
    expected_identities = _expected_full_quality_identities()
    if list(plan.get("expected_full_quality_identities") or []) != (
        expected_identities
    ):
        raise ValueError(
            "Full-only identity contract mismatch: "
            f"{plan.get('expected_full_quality_identities')!r}"
        )
    expected_setup_groups = {
        H8_SETUP: ["hailo8", "ort_tensorrt"],
        H10_SETUP: ["hailo10", "ort_tensorrt"],
    }
    if dict(plan.get("setup_groups") or {}) != expected_setup_groups:
        raise ValueError(
            f"Full-only setup dispatch mismatch: {plan.get('setup_groups')!r}"
        )
    if list(plan.get("effective_generic_run_ids") or []):
        raise ValueError(
            "Full-only plan contains generic run IDs: "
            f"{plan.get('effective_generic_run_ids')!r}"
        )
    if list(plan.get("management_reference_profiles") or []) != ["ort_cpu"]:
        raise ValueError(
            "Unexpected management reference plan: "
            f"{plan.get('management_reference_profiles')!r}"
        )
    hailo = profile.get("hailo_build") or {}
    validation = profile.get("validation_execution") or {}
    statistics = (profile.get("quality_gate") or {}).get("statistics") or {}
    calibration_items = plan.get("calibration_items") or {}
    validation_items = plan.get("validation_items") or {}
    if (
        hailo.get("mode") != "reuse_and_build_missing"
        or hailo.get("preset") != "final"
        or int(hailo.get("optimization_level") or -1) != 2
        or int(hailo.get("calib_count") or 0) != CALIBRATION_ITEMS
        or int(hailo.get("calib_batch_size") or 0) != 8
        or hailo.get("calibration_storage") != "memmap"
        or hailo.get("cache_integrity") != "strict"
        or int(hailo.get("timeout_s") or 0) != 10800
        or hailo.get("build_full") is not True
        or hailo.get("force_build") is not False
        or int(calibration_items.get("detection") or 0)
        != CALIBRATION_ITEMS
        or int((validation.get("max_items") or {}).get("detection") or 0)
        != VALIDATION_ITEMS
        or int(validation_items.get("detection") or 0) != VALIDATION_ITEMS
        or int(statistics.get("bootstrap_repetitions") or 0)
        != BOOTSTRAP_REPETITIONS
        or int(plan.get("bootstrap_repetitions") or 0)
        != BOOTSTRAP_REPETITIONS
    ):
        raise ValueError("Final-depth Quality/Hailo axes were not materialized")
    expected_statistics = {
        "method": "paired_bootstrap",
        "confidence_level": 0.95,
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "seed": 20260710,
        "decision": "lower_one_sided_bound",
        "execution_location": "central_management",
        "workers": 4,
    }
    if {
        key: statistics.get(key) for key in expected_statistics
    } != expected_statistics:
        raise ValueError(f"Quality statistics contract mismatch: {statistics!r}")
    detection = (profile.get("quality_gate") or {}).get("detection") or {}
    guardrails = detection.get("guardrails") or {}
    if (
        detection.get("primary_metric") != "coco_ap_50_95"
        or float(detection.get("non_inferiority_margin") or -1.0) != 0.01
        or float(guardrails.get("ap50_margin") or -1.0) != 0.01
        or float(guardrails.get("ap75_margin") or -1.0) != 0.01
    ):
        raise ValueError("AP50:95/AP50/AP75 margins changed unexpectedly")
    native = profile.get("native_producers") or {}
    native_energy = native.get("energy") or {}
    energy = profile.get("energy") or {}
    workflow = profile.get("workflow") or {}
    if (
        native.get("enabled") is not False
        or native_energy.get("enabled") is not False
        or energy.get("enabled") is not False
        or energy.get("generic_enabled") is not False
        or energy.get("requested_native_energy") is not False
        or workflow.get("skip_runtime_benchmarks") is not False
    ):
        raise ValueError("Native/Energy/Quality-runtime switch contract mismatch")
    return plan


def write_profile(*, destination: Path, models_root: Path, profile_id: str) -> Path:
    profile = build_profile(profile_id=profile_id, models_root=models_root)
    verify_profile(profile)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        yaml.safe_dump(profile, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    loaded = load_evaluation_profile(str(destination), validate=True)
    if loaded is None or isinstance(loaded, tuple):
        raise RuntimeError(f"Generated profile cannot be reloaded: {destination}")
    verify_profile(dict(loaded.raw_profile or {}))
    return destination.resolve(strict=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--models-root", required=True)
    parser.add_argument("--profile-id", default="")
    args = parser.parse_args()

    models_root = Path(args.models_root).expanduser().resolve(strict=True)
    profile_id = str(args.profile_id or "").strip() or _profile_id()
    destination = Path(args.out).expanduser()
    path = write_profile(
        destination=destination,
        models_root=models_root,
        profile_id=profile_id,
    )
    print(f"PROFILE={path}")
    print(f"PROFILE_ID={profile_id}")
    print("FULL_ONLY_PLAN=PASS generic_rows=0 quality_results=4 remote_invocations=2")
    print(
        "QUALITY_AXES="
        f"validation={VALIDATION_ITEMS} bootstrap={BOOTSTRAP_REPETITIONS} "
        "metrics=AP50:95,AP50,AP75"
    )
    print(
        "HAILO_AXES="
        f"preset=final opt=2 calibration={CALIBRATION_ITEMS}"
    )
    print("NATIVE=off")
    print("ENERGY=off")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
