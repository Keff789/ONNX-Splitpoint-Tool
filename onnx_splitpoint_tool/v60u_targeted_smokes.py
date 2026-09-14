from __future__ import annotations

"""Generate focused v60u hardware-smoke profiles from an existing profile.

The normal Smoke mode is deliberately cache-first and may defer a heavy missing
Hailo Full baseline.  The focused profiles generated here exercise one concern at
a time without turning the regular Smoke run into a long, ambiguous campaign.
"""

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

import yaml

from . import __version__
from .run_modes import default_run_modes_config
from .workflow.runner import WORKFLOW_VERSION

SCENARIOS = (
    "cross_runner",
    "native_full",
    "native_energy",
    "artifact_restore",
    "parallel_cold_build",
)


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _base_profile(payload: Mapping[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(dict(payload))
    out.setdefault("name", "evaluation")
    out.setdefault("selection_policy", {})
    out.setdefault("execution_preset", {})
    return out


def _smoke_snapshot() -> dict[str, Any]:
    return copy.deepcopy(default_run_modes_config()["modes"]["smoke"])


def derive_profile(payload: Mapping[str, Any], scenario: str) -> dict[str, Any]:
    scenario = str(scenario or "").strip().lower()
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown v60u smoke scenario: {scenario!r}")
    out = _base_profile(payload)
    base_name = str(out.get("name") or "evaluation")
    out["name"] = f"{base_name}_v60u_{scenario}"
    snapshot = _smoke_snapshot()
    selection = _mapping(out.get("selection_policy"))
    overrides = {"native_enabled": True, "energy_enabled": False}

    # Every focused smoke records native row failures and explicit contract
    # families.  Full baselines are enabled only where the scenario requires
    # them, so the inexpensive default Smoke remains inexpensive.
    snapshot["runtime"]["native"]["validation"] = True
    snapshot["runtime"]["native"]["dump_outputs"] = True
    snapshot["runtime"]["native"]["preserve_row_failures"] = True
    snapshot["runtime"]["native"]["contract_selection"] = "metadata_first"

    if scenario == "cross_runner":
        selection["max_accepted_cases_per_model"] = max(3, int(selection.get("max_accepted_cases_per_model") or 0))
        selection["preferred_shortlist_size"] = max(3, int(selection.get("preferred_shortlist_size") or 0))
        snapshot["ranking"]["enabled"] = True
        snapshot["runtime"]["native"]["full_baselines"] = True
        snapshot["build"]["hailo"]["full_baseline_cold_build_policy"] = "build_missing"
        snapshot["build"]["hailo"]["deferred_full_baseline_required"] = True
        snapshot["build"]["hailo"]["timeout_s"] = max(3600, int(snapshot["build"]["hailo"].get("timeout_s") or 0))
        snapshot["build"]["hailo"]["cold_build_timeout_s"] = max(3600, int(snapshot["build"]["hailo"].get("cold_build_timeout_s") or 0))
    elif scenario == "native_full":
        selection["max_accepted_cases_per_model"] = 1
        snapshot["runtime"]["native"]["full_baselines"] = True
        snapshot["build"]["hailo"]["full_baseline_cold_build_policy"] = "build_missing"
        snapshot["build"]["hailo"]["deferred_full_baseline_required"] = True
        snapshot["build"]["hailo"]["timeout_s"] = max(3600, int(snapshot["build"]["hailo"].get("timeout_s") or 0))
        snapshot["build"]["hailo"]["cold_build_timeout_s"] = max(3600, int(snapshot["build"]["hailo"].get("cold_build_timeout_s") or 0))
    elif scenario == "native_energy":
        selection["max_accepted_cases_per_model"] = 1
        snapshot["runtime"]["native"]["full_baselines"] = True
        snapshot["build"]["hailo"]["full_baseline_cold_build_policy"] = "build_missing"
        snapshot["build"]["hailo"]["deferred_full_baseline_required"] = True
        snapshot["energy"]["native_mode"] = "measure"
        overrides["energy_enabled"] = True
    elif scenario == "artifact_restore":
        # This profile exercises normal execution after the legacy cache entry
        # has been moved aside.  The Artifact Library should materialize the
        # exact registered object without invoking the compiler.
        selection["max_accepted_cases_per_model"] = 1
        snapshot["runtime"]["native"]["full_baselines"] = False
        snapshot["build"]["artifact_store"]["enabled"] = True
        snapshot["build"]["artifact_store"]["verify_on_reuse"] = "strict"
        snapshot["build"]["hailo"]["full_baseline_cold_build_policy"] = "cache_or_defer"
    elif scenario == "parallel_cold_build":
        selection["max_accepted_cases_per_model"] = 1
        overrides["native_enabled"] = False
        snapshot["runtime"]["native"]["full_baselines"] = False
        snapshot["build"]["scheduler"]["enabled"] = True
        snapshot["build"]["scheduler"]["max_workers"] = max(3, int(snapshot["build"]["scheduler"].get("max_workers") or 0))
        snapshot["build"]["hailo"]["full_baseline_cold_build_policy"] = "build_missing"
        snapshot["build"]["hailo"]["deferred_full_baseline_required"] = True
        snapshot["build"]["hailo"]["timeout_s"] = max(3600, int(snapshot["build"]["hailo"].get("timeout_s") or 0))

    out["selection_policy"] = selection
    out["execution_preset"] = {
        "id": "smoke",
        "follow_tool_config": False,
        "snapshot": snapshot,
        "overrides": overrides,
        "generated_by": "onnx-splitpoint-targeted-smokes-v60u",
        "tool_version": __version__,
        "workflow_version": WORKFLOW_VERSION,
        "scenario": scenario,
    }
    return out


def _scenario_note(name: str) -> str:
    return {
        "cross_runner": "At least three shared Generic/Native candidates; validates rank transfer.",
        "native_full": "Builds/runs Native Full baselines and preserves explicit failure diagnostics.",
        "native_energy": "Runs exactly the Native system-energy measurement path (Generic energy stays off).",
        "artifact_restore": "Verifies materialisation from the central Artifact Library after moving a legacy cache entry aside.",
        "parallel_cold_build": "Intentionally permits cold Hailo-8/Hailo-10/DeepX misses to validate scheduler overlap.",
    }[name]


def generate_profiles(profile_path: Path, out_dir: Path, scenarios: Sequence[str]) -> dict[str, Any]:
    payload = yaml.safe_load(profile_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"Profile root must be a mapping: {profile_path}")
    out_dir.mkdir(parents=True, exist_ok=True)
    generated: list[dict[str, Any]] = []
    for scenario in scenarios:
        derived = derive_profile(payload, scenario)
        path = out_dir / f"{profile_path.stem}_v60u_{scenario}.yaml"
        path.write_text(yaml.safe_dump(derived, sort_keys=False, allow_unicode=True), encoding="utf-8")
        generated.append({"scenario": scenario, "profile": str(path), "note": _scenario_note(scenario)})
    manifest = {
        "schema": "onnx-splitpoint/v60u-targeted-smoke-profiles",
        "schema_version": 1,
        "tool_version": __version__,
        "workflow_version": WORKFLOW_VERSION,
        "source_profile": str(profile_path),
        "generated": generated,
    }
    (out_dir / "v60u_targeted_smoke_profiles.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    md = ["# v60u targeted hardware smokes", "", f"Source profile: `{profile_path}`", "", "| Scenario | Profile | Purpose |", "|---|---|---|"]
    for row in generated:
        md.append(f"| `{row['scenario']}` | `{row['profile']}` | {row['note']} |")
    md += [
        "",
        "The regular Smoke profile remains cache-first. The `native_full`, `native_energy`, and `parallel_cold_build` profiles intentionally permit expensive cold builds.",
        "Move or quarantine only the exact cache object under test; do not delete the entire Artifact Library.",
    ]
    (out_dir / "v60u_targeted_smoke_profiles.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="onnx-splitpoint-targeted-smokes-v60u")
    parser.add_argument("--profile", required=True, help="Existing evaluation profile YAML")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--scenario", action="append", choices=SCENARIOS, default=[])
    args = parser.parse_args(list(argv) if argv is not None else None)
    scenarios = tuple(args.scenario) if args.scenario else SCENARIOS
    manifest = generate_profiles(Path(args.profile).expanduser().resolve(), Path(args.out_dir).expanduser().resolve(), scenarios)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
