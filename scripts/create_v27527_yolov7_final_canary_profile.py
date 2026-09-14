#!/usr/bin/env python3
from __future__ import annotations

"""Project one sealed profile into a one-model YOLOv7 Final-contract canary."""

import argparse
import copy
from pathlib import Path
import sys
from typing import Any, Mapping

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from onnx_splitpoint_tool.benchmark.evaluation_profiles import (
    validate_evaluation_profile_payload,
)
from onnx_splitpoint_tool.campaign import (
    EVALUATED_MATRIX_CLAIM_SCOPE,
    build_campaign_readiness,
    readiness_markdown,
)
from onnx_splitpoint_tool.run_modes import (
    _legacy_v11_final_mode,
    apply_run_mode,
    default_run_modes_config,
)
from onnx_splitpoint_tool.workflow.artifacts import write_json, write_text


SEED = 20260710


def _load(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, Mapping):
        raise ValueError(f"profile must be a mapping: {path}")
    return copy.deepcopy(dict(value))


def _yolov7_rows(profile: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return only the audited paper-model identity, never a fuzzy alias."""

    suite = profile.get("model_suite") if isinstance(profile.get("model_suite"), Mapping) else {}
    rows: list[dict[str, Any]] = []
    for tier in ("primary", "reserve"):
        for item in list(suite.get(tier) or []):
            if not isinstance(item, Mapping) or item.get("enabled") is False:
                continue
            if item.get("id") == "yolov7_paper":
                rows.append(copy.deepcopy(dict(item)))
    return rows


def build_canary(source: Mapping[str, Any], *, profile_id: str) -> dict[str, Any]:
    matches = _yolov7_rows(source)
    if len(matches) != 1:
        raise ValueError(
            "expected exactly one active model with id 'yolov7_paper', "
            f"found {len(matches)}: "
            f"{[row.get('id') for row in matches]}"
        )
    model = matches[0]
    model.update({
        "enabled": True,
        "evaluation_role": "development",
        "generalization_scope": "development",
        "validation_tier": "final",
        "candidate_universe_complete": False,
        "candidate_universe": {"mode": "declared_shortlist", "seed": SEED},
    })

    draft = copy.deepcopy(dict(source))
    draft["name"] = profile_id
    draft["purpose"] = (
        "YOLOv7 one-model Final-contract canary for Hailo-8, Hailo-10H and "
        "DeepX. It exercises the evaluated-matrix gates without authorizing "
        "the subsequent full three-model campaign."
    )
    draft["model_suite"] = {"primary": [model], "reserve": []}
    draft.pop("quality_canary", None)
    campaign = dict(draft.get("campaign") or {})
    campaign.update({
        "id": profile_id,
        "claim_scope": EVALUATED_MATRIX_CLAIM_SCOPE,
        "mode": "final",
        "enforcement": "strict",
        "frozen_before_final_campaign": True,
        "require_protocol_freeze": False,
        "require_fitted_stage_time": False,
        "require_native_handover_model": False,
        "require_campaign_freeze": False,
        "require_prediction_freeze_approval": False,
        "prediction_freeze_enabled": False,
    })
    campaign["holdout_registry"] = ""
    campaign["ranking_model_bundle"] = ""
    draft["campaign"] = campaign
    draft["ranking_validation"] = {"enabled": False}

    # This retained v2.75.27 canary deliberately exercises the historical,
    # explicit strict campaign contract.  The user-facing ``final`` mode is
    # Final Quality (Standard+) from schema v12 onward and must never inherit
    # these canary-only execution settings.
    canary_modes = default_run_modes_config()
    canary_modes["modes"]["final"] = _legacy_v11_final_mode()
    resolved, _audit = apply_run_mode(
        draft,
        mode_id="final",
        config=canary_modes,
        follow_tool_config=True,
    )
    resolved["execution_preset"]["follow_tool_config"] = False
    # Keep the generated profile byte-stable across an identical warm replay;
    # the EvaluationRun itself already records creation time.
    resolved["execution_preset"].pop("resolved_at", None)
    resolved["selection_policy"].update({
        "max_accepted_cases_per_model": 1,
        "preferred_shortlist": 5,
        "candidate_search_pool": "auto",
        "selection_strategy": "candidate_universe",
        "report_blocked_configs": True,
        "report_plan_adjustments": True,
    })
    resolved["ranking_validation"]["enabled"] = False

    native = resolved.get("native_producers") if isinstance(resolved.get("native_producers"), Mapping) else {}
    native = dict(native)
    native.update({
        "enabled": True,
        "consumer": "tensorrt",
        "case_policy": "preferred_then_backfill",
        "strict_supported_only": True,
        "repetitions": 5,
    })
    full = dict(native.get("full_baselines") or {})
    full.update({
        "enabled": True,
        "required": True,
        "same_runtime_contract": True,
        "backends": ["tensorrt", "hailo8", "hailo10h", "deepx"],
        "backends_by_producer": {
            "hailo8": ["tensorrt", "hailo8"],
            "hailo10h": ["tensorrt", "hailo10h"],
            "deepx": ["tensorrt", "deepx"],
        },
    })
    native["full_baselines"] = full
    native_energy = dict(native.get("energy") or {})
    native_energy.update({"enabled": True, "mode": "measure", "phases": ["streaming"]})
    native["energy"] = native_energy
    resolved["native_producers"] = native

    energy = dict(resolved.get("energy") or {})
    energy.update({
        "enabled": False,
        "generic_enabled": False,
        "requested_native_energy": True,
        "measurement_path": "native_only",
        "repeats": 5,
        "strict": True,
        "phases": ["streaming"],
    })
    resolved["energy"] = energy
    measurement = dict(resolved.get("measurement_campaign") or {})
    system_power = dict(measurement.get("system_power") or {})
    system_power.update({
        "scope": "FS",
        "window": "command",
        "raw_primary": True,
        "repeats": 5,
        "confidence_level": 0.95,
        "randomize_run_order": True,
        "randomization_seed": SEED,
    })
    measurement["system_power"] = system_power
    resolved["measurement_campaign"] = measurement
    resolved["workflow"]["powercalc_workers"] = 1
    resolved["implementation_note"] = (
        "Generated by create_v27527_yolov7_final_canary_profile.py. Preflight "
        "must be final_ready before hardware dispatch. Energy runs serially. "
        "A passing canary is necessary but does not itself authorize Final."
    )
    return validate_evaluation_profile_payload(resolved)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-profile", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--preflight-dir", default="")
    parser.add_argument("--profile-id", default="")
    args = parser.parse_args()

    source_path = Path(args.source_profile).expanduser().resolve(strict=True)
    destination = Path(args.out).expanduser().resolve()
    profile_id = str(args.profile_id or "").strip() or "yolov7_final_contract_canary_v27527"
    profile = build_canary(_load(source_path), profile_id=profile_id)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        yaml.safe_dump(profile, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    report = build_campaign_readiness(profile, profile_path=destination)
    preflight_dir = (
        Path(args.preflight_dir).expanduser().resolve()
        if args.preflight_dir
        else destination.with_suffix("").with_name(destination.stem + "_preflight")
    )
    preflight_dir.mkdir(parents=True, exist_ok=True)
    write_json(preflight_dir / "campaign_readiness.json", report)
    write_text(preflight_dir / "campaign_readiness.md", readiness_markdown(report))
    print(f"PROFILE={destination}")
    print(f"PREFLIGHT={preflight_dir}")
    print(f"CLAIM_SCOPE={report.get('claim_scope')}")
    print(f"STATUS={report.get('status')}")
    print(f"REQUIRED_FAILURES={report.get('required_failure_count')}")
    return 0 if report.get("final_ready") else 2


if __name__ == "__main__":
    raise SystemExit(main())
