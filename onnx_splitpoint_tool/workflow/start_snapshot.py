from __future__ import annotations

from ..config_values import validate_profile_config_booleans

"""Immutable, fail-closed profile snapshot used when an EvalRun starts.

The GUI preview, the queue operation and the workflow worker used to resolve an
Evaluation Profile independently.  A profile file or the central run-mode
registry could therefore change between those operations and the run would
archive a different model/native/energy selection than the one shown before
start.  This module provides one small, central contract which is usable by the
GUI and by non-GUI callers alike.
"""

import copy
import hashlib
import json
from typing import Any, Dict, Mapping, Sequence

from ..cache_verify_policy import (
    CacheVerifyPolicyError,
    apply_cache_verify_only_policy,
    cache_verify_guard,
    validate_cache_verify_only_profile,
)
from ..execution_plan import build_effective_execution_plan


class StartSnapshotConsistencyError(ValueError):
    """Raised before execution when request, resolution and plan disagree."""


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def snapshot_payload_sha256(value: Any) -> str:
    """Public canonical hash used by frozen subcontracts such as hardware."""

    return _canonical_hash(value)


def _execution_stable_value(value: Any) -> Any:
    """Drop timestamps while retaining every execution-relevant binding."""

    if isinstance(value, Mapping):
        return {
            str(key): _execution_stable_value(item)
            for key, item in value.items()
            if str(key) not in {"created_at", "resolved_at", "updated_at"}
        }
    if isinstance(value, (list, tuple)):
        return [_execution_stable_value(item) for item in value]
    return copy.deepcopy(value)


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        values = value.split(",")
    elif isinstance(value, Sequence):
        values = list(value)
    else:
        values = [value]
    return [str(item).strip() for item in values if str(item).strip()]


def _truth(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, Mapping):
        if "enabled" in value:
            return _truth(value.get("enabled"))
        return bool(value)
    return str(value or "").strip().lower() in {"1", "true", "yes", "on", "enabled"}


def _enabled_rows(value: Any) -> list[Mapping[str, Any]]:
    return [
        row for row in list(value or [])
        if isinstance(row, Mapping) and _truth(row.get("enabled", True))
    ]


def _row_ids(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    return [
        str(row.get("id") or row.get("full") or row.get("path") or "").strip()
        for row in rows
        if str(row.get("id") or row.get("full") or row.get("path") or "").strip()
    ]


def _explicit_execution_request(profile: Mapping[str, Any]) -> Dict[str, Any]:
    """Return only execution switches explicitly present in the request.

    Missing values are deliberately represented as ``None``.  The selected
    run mode is then allowed to supply its default; values the user explicitly
    selected must survive resolution exactly.
    """

    preset = profile.get("execution_preset") if isinstance(profile.get("execution_preset"), Mapping) else {}
    overrides = preset.get("overrides") if isinstance(preset.get("overrides"), Mapping) else {}
    native_cfg = profile.get("native_producers") if isinstance(profile.get("native_producers"), Mapping) else {}
    native_energy = native_cfg.get("energy") if isinstance(native_cfg.get("energy"), Mapping) else {}
    energy_cfg = profile.get("energy") if isinstance(profile.get("energy"), Mapping) else {}

    native: bool | None = None
    if "native_enabled" in overrides:
        native = _truth(overrides.get("native_enabled"))
    elif "enabled" in native_cfg:
        native = _truth(native_cfg.get("enabled"))

    energy: bool | None = None
    if "energy_enabled" in overrides:
        energy = _truth(overrides.get("energy_enabled"))
    elif "requested_native_energy" in energy_cfg:
        energy = _truth(energy_cfg.get("requested_native_energy"))
    elif "enabled" in native_energy:
        energy = _truth(native_energy.get("enabled"))

    return {"native_enabled": native, "energy_enabled": energy}


def profile_selection_view(profile: Mapping[str, Any], *, explicit_request: bool = False) -> Dict[str, Any]:
    """Return the selection displayed to the user, without deep mode details."""

    preset = profile.get("execution_preset") if isinstance(profile.get("execution_preset"), Mapping) else {}
    overrides = preset.get("overrides") if isinstance(preset.get("overrides"), Mapping) else {}
    suite = profile.get("model_suite") if isinstance(profile.get("model_suite"), Mapping) else {}
    selection = profile.get("selection_policy") if isinstance(profile.get("selection_policy"), Mapping) else {}
    execution = _explicit_execution_request(profile)
    if not explicit_request:
        if isinstance(profile.get("execution_preset"), Mapping):
            execution = {
                "native_enabled": _truth(overrides.get("native_enabled")),
                "energy_enabled": _truth(overrides.get("energy_enabled")),
            }
        else:
            execution = _explicit_execution_request(profile)
            execution = {
                "native_enabled": bool(execution.get("native_enabled")),
                "energy_enabled": bool(execution.get("energy_enabled")),
            }
    selection_keys = (
        "max_accepted_cases_per_model",
        "preferred_shortlist",
        "selection_strategy",
        "min_gap",
        "candidate_search_pool",
        "require_single_part2_input",
        "score_independent_audit_enabled",
        "audit_candidate_universe",
        "audit_size",
        "minimum_valid_audit_candidates",
        "audit_seed",
    )
    guard = cache_verify_guard(profile)
    return {
        "profile_name": str(profile.get("name") or "").strip(),
        "campaign_id": str(
            (profile.get("campaign") or {}).get("id")
            if isinstance(profile.get("campaign"), Mapping)
            else ""
        ).strip(),
        "run_mode": str(preset.get("id") or "").strip(),
        "follow_tool_config": bool(preset.get("follow_tool_config", True)),
        "config_sha256": str(preset.get("config_sha256") or ""),
        "snapshot_sha256": str(preset.get("snapshot_sha256") or ""),
        "native_enabled": execution.get("native_enabled"),
        "energy_enabled": execution.get("energy_enabled"),
        "models": _row_ids(_enabled_rows(suite.get("primary"))),
        "run_profiles": _row_ids(_enabled_rows(profile.get("run_profiles"))),
        "selection_policy": {
            key: copy.deepcopy(selection.get(key))
            for key in selection_keys
            if key in selection
        },
        "execution_guard": (
            {
                "mode": str(guard.get("mode") or ""),
                "expected_plan": copy.deepcopy(dict(guard.get("expected_plan") or {})),
            }
            if guard
            else None
        ),
    }


def _append_mismatch(
    mismatches: list[Dict[str, Any]],
    field: str,
    requested: Any,
    resolved: Any,
    *,
    only_if_requested: bool = False,
) -> None:
    if only_if_requested and requested is None:
        return
    if requested != resolved:
        mismatches.append({"field": field, "requested": requested, "resolved": resolved})


def build_profile_start_snapshot(
    *,
    profile_request: str,
    source_profile: Mapping[str, Any],
    resolved_profile: Mapping[str, Any],
    profile_id: str,
    profile_path: str,
    profile_source: str,
    runtime_bindings: Mapping[str, Any] | None = None,
    schema_version: int = 2,
) -> Dict[str, Any]:
    """Validate and freeze one EvalRun start profile.

    No expected Smoke/Standard/Final values are hard-coded.  The function only
    asserts that explicit request values survive resolution and that the
    resolved materialised blocks agree with their own execution plan.
    """

    if int(schema_version) not in {1, 2}:
        raise StartSnapshotConsistencyError(
            "Evaluation start snapshot schema is unsupported."
        )

    requested = profile_selection_view(source_profile, explicit_request=True)
    resolved = profile_selection_view(resolved_profile)
    # ``execution_preset.snapshot_sha256`` identifies the selected central
    # run-mode record.  A hand-written source profile legitimately has no such
    # value before ``apply_run_mode`` materialises it, so it cannot also serve
    # as the immutable hash of the visible selection itself.  Freeze both
    # selection views independently and retain the legacy run-mode hash as a
    # separate field inside each view.
    if int(schema_version) >= 2:
        requested["selection_snapshot_sha256"] = _canonical_hash(
            _execution_stable_value(requested)
        )
        resolved["selection_snapshot_sha256"] = _canonical_hash(
            _execution_stable_value(resolved)
        )
    plan = build_effective_execution_plan(resolved_profile)
    try:
        cache_verify_attestation = validate_cache_verify_only_profile(
            resolved_profile
        )
    except CacheVerifyPolicyError as exc:
        raise StartSnapshotConsistencyError(str(exc)) from exc
    mismatches: list[Dict[str, Any]] = []

    for field in (
        "profile_name",
        "run_mode",
        "models",
        "run_profiles",
        "selection_policy",
        "execution_guard",
    ):
        requested_value = requested.get(field)
        if field in {"profile_name", "run_mode"} and requested_value in {None, ""}:
            continue
        _append_mismatch(mismatches, field, requested_value, resolved.get(field))
    for field in ("native_enabled", "energy_enabled"):
        _append_mismatch(
            mismatches,
            field,
            requested.get(field),
            resolved.get(field),
            only_if_requested=True,
        )

    native_cfg = resolved_profile.get("native_producers") if isinstance(resolved_profile.get("native_producers"), Mapping) else {}
    native_energy = native_cfg.get("energy") if isinstance(native_cfg.get("energy"), Mapping) else {}
    top_energy = resolved_profile.get("energy") if isinstance(resolved_profile.get("energy"), Mapping) else {}
    resolved_native = bool(resolved.get("native_enabled"))
    resolved_energy_request = bool(resolved.get("energy_enabled"))
    materialized_native = _truth(native_cfg.get("enabled"))
    if "requested_native_energy" in top_energy:
        materialized_energy_request = _truth(top_energy.get("requested_native_energy"))
    elif not isinstance(resolved_profile.get("execution_preset"), Mapping):
        materialized_energy_request = resolved_energy_request
    else:
        materialized_energy_request = False
    materialized_native_energy = _truth(native_energy.get("enabled"))

    _append_mismatch(mismatches, "materialized.native_producers.enabled", resolved_native, materialized_native)
    if "requested_native_energy" in top_energy or isinstance(resolved_profile.get("execution_preset"), Mapping):
        _append_mismatch(
            mismatches,
            "materialized.energy.requested_native_energy",
            resolved_energy_request,
            materialized_energy_request,
        )
    _append_mismatch(
        mismatches,
        "materialized.native_producers.energy.enabled",
        bool(resolved_native and resolved_energy_request),
        materialized_native_energy,
    )
    # Legacy profiles predate execution_preset.  Their execution plan labels
    # the implicit fallback as ``standard``; that is not a conflicting user
    # request and therefore must not make profile loading fail.
    if str(resolved.get("run_mode") or ""):
        _append_mismatch(mismatches, "execution_plan.run_mode", resolved.get("run_mode"), str(plan.get("run_mode") or ""))
    _append_mismatch(mismatches, "execution_plan.models", resolved.get("models"), list(plan.get("models") or []))
    _append_mismatch(
        mismatches,
        "execution_plan.logical_run_profiles",
        resolved.get("run_profiles"),
        list(plan.get("logical_run_profiles") or []),
    )
    if isinstance(resolved_profile.get("execution_preset"), Mapping):
        _append_mismatch(mismatches, "execution_plan.native_enabled", resolved_native, bool(plan.get("native_enabled")))
        _append_mismatch(
            mismatches,
            "execution_plan.native_energy_requested",
            resolved_energy_request,
            bool(plan.get("native_energy_requested")),
        )
        _append_mismatch(
            mismatches,
            "execution_plan.native_energy_enabled",
            bool(resolved_native and resolved_energy_request),
            bool(plan.get("native_energy_enabled")),
        )

    if mismatches:
        detail = "; ".join(
            f"{item['field']}: requested={item['requested']!r}, resolved={item['resolved']!r}"
            for item in mismatches[:8]
        )
        raise StartSnapshotConsistencyError(
            "Evaluation start blocked because the requested/visible profile selection does not match "
            f"the resolved execution snapshot ({detail}). Reload the profile summary and start again."
        )

    fingerprint_payload = {
        "profile_request": str(profile_request or ""),
        "requested_selection": requested,
        "resolved_selection": resolved,
        "effective_execution_plan": plan,
        "cache_verify_attestation": cache_verify_attestation,
    }
    snapshot = {
        "schema": "onnx-splitpoint/evaluation-start-snapshot",
        "schema_version": int(schema_version),
        "profile_request": str(profile_request or ""),
        "profile_id": str(profile_id or ""),
        "profile_path": str(profile_path or ""),
        "profile_source": str(profile_source or ""),
        "requested_selection": requested,
        "resolved_selection": resolved,
        "effective_execution_plan": copy.deepcopy(plan),
        "cache_verify_attestation": copy.deepcopy(cache_verify_attestation),
        "source_profile_sha256": _canonical_hash(source_profile),
        "resolved_profile_sha256": _canonical_hash(resolved_profile),
        "resolved_execution_sha256": _canonical_hash({
            "profile": _execution_stable_value(resolved_profile),
            "runtime_bindings": _execution_stable_value(runtime_bindings or {}),
        }),
        "selection_fingerprint": _canonical_hash(fingerprint_payload),
        "runtime_bindings": copy.deepcopy(dict(runtime_bindings or {})),
        "source_profile": copy.deepcopy(dict(source_profile or {})),
        "resolved_profile": copy.deepcopy(dict(resolved_profile or {})),
        "consistency": {"status": "ok", "mismatches": []},
    }
    snapshot["snapshot_sha256"] = _canonical_hash({
        key: value for key, value in snapshot.items() if key != "snapshot_sha256"
    })
    return snapshot


def materialize_runtime_profile(
    profile: Mapping[str, Any],
    *,
    profile_path: str = "",
    options: Any = None,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Resolve all profile mutations before the immutable start snapshot.

    Ranking-bundle and dataset-registry binding may read external files.  The
    EnergyDefaults-backed marker/window probe used to be resolved even later in
    ``runner.run``.  Performing all three operations here lets the GUI preview
    and worker consume exactly one final runtime profile.
    """

    from ..campaign import apply_ranking_model_bundle
    from .dataset_binding import bind_profile_dataset_registry
    from .hardware_matrix import normalize_hardware_targets

    resolved, ranking = apply_ranking_model_bundle(
        copy.deepcopy(dict(profile or {})),
        profile_path=profile_path or None,
    )
    resolved, dataset = bind_profile_dataset_registry(
        resolved,
        profile_path=profile_path or None,
        verify_manifests=False,
    )

    hardware = dict(resolved.get("hardware") or {}) if isinstance(resolved.get("hardware"), Mapping) else {}
    if options is not None:
        setups_file = str(getattr(options, "hardware_setups_file", "") or "").strip()
        if setups_file:
            hardware["setups_file"] = setups_file
        setup_ids = _string_list(getattr(options, "hardware_setup_ids", []))
        group_ids = _string_list(getattr(options, "hardware_group_ids", []))
        if setup_ids:
            base = _string_list(hardware.get("selected_setups"))
            for setup_id in setup_ids:
                if setup_id not in base:
                    base.append(setup_id)
            hardware["selected_setups"] = base
        if group_ids:
            base_groups = _string_list(hardware.get("selected_groups"))
            for group_id in group_ids:
                if group_id not in base_groups:
                    base_groups.append(group_id)
            hardware["selected_groups"] = base_groups
    resolved["hardware"] = hardware
    frozen_hardware_targets = normalize_hardware_targets(resolved)
    hardware = dict(resolved.get("hardware") or {})
    hardware["resolved_targets"] = copy.deepcopy(frozen_hardware_targets)
    hardware["resolved_targets_sha256"] = _canonical_hash(frozen_hardware_targets)
    hardware["resolution_frozen_at_start"] = True
    resolved["hardware"] = hardware

    native = dict(resolved.get("native_producers") or {})
    energy = dict(native.get("energy") or {})
    raw_probe = energy.get("window_method_validation_probe")
    probe = dict(raw_probe) if isinstance(raw_probe, Mapping) else {}
    try:
        from ..energy.config import load_energy_defaults

        defaults = load_energy_defaults()
    except Exception:
        defaults = None

    def _positive_float(value: Any) -> float | None:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        return parsed if parsed > 0 else None

    # A zero duration delegates to Tool Config / EnergyDefaults.  Resolve that
    # delegation before hashing the GUI-visible start snapshot; execution and
    # resume must not read a newer duration later.
    raw_energy_duration = _positive_float(energy.get("duration_s"))
    raw_measurement_duration = _positive_float(
        energy.get("measurement_duration_s")
    )
    raw_native_duration = _positive_float(native.get("native_energy_duration_s"))
    default_native_duration = _positive_float(
        getattr(defaults, "native_energy_duration_s", 60.0)
    ) or 60.0
    if raw_energy_duration is not None:
        native_energy_duration = raw_energy_duration
        native_energy_duration_source = "profile.native_producers.energy.duration_s"
    elif raw_measurement_duration is not None:
        native_energy_duration = raw_measurement_duration
        native_energy_duration_source = (
            "profile.native_producers.energy.measurement_duration_s"
        )
    elif raw_native_duration is not None:
        native_energy_duration = raw_native_duration
        native_energy_duration_source = (
            "profile.native_producers.native_energy_duration_s"
        )
    else:
        native_energy_duration = default_native_duration
        native_energy_duration_source = "energy_defaults.native_energy_duration_s"
    energy["duration_s"] = float(native_energy_duration)
    energy["duration_source"] = native_energy_duration_source

    def _opt(name: str, default: Any = None) -> Any:
        return getattr(options, name, default) if options is not None else default

    def _pick_bool(option_name: str, key: str, default: bool) -> tuple[bool, str]:
        option = _opt(option_name, None)
        if option is not None:
            return bool(option), "cli"
        if key in probe:
            return bool(probe.get(key)), "profile"
        return bool(default), "energy_defaults"

    configured_enabled, enabled_source = _pick_bool(
        "window_method_validation_probe_enabled",
        "enabled",
        bool(getattr(defaults, "window_method_validation_probe_enabled", True)),
    )
    include_raw, raw_source = _pick_bool(
        "window_method_validation_probe_include_raw_parquet",
        "include_raw_parquet",
        bool(getattr(defaults, "window_method_validation_probe_include_raw_parquet", True)),
    )
    strict, strict_source = _pick_bool(
        "window_method_validation_probe_strict",
        "strict",
        bool(getattr(defaults, "window_method_validation_probe_strict", True)),
    )
    raw_probe_duration = _positive_float(probe.get("duration_s"))
    probe_duration = (
        raw_probe_duration
        if raw_probe_duration is not None
        else float(native_energy_duration)
    )
    probe_duration_source = (
        "profile.native_producers.energy.window_method_validation_probe.duration_s"
        if raw_probe_duration is not None
        else native_energy_duration_source
    )
    cli_repeats = int(_opt("window_method_validation_probe_repeats", 0) or 0)
    if cli_repeats > 0:
        repeats, repeats_source = cli_repeats, "cli"
    elif probe.get("repeats") not in (None, ""):
        try:
            repeats = max(1, int(probe.get("repeats")))
        except Exception:
            repeats = 3
        repeats_source = "profile"
    else:
        repeats = max(1, int(getattr(defaults, "window_method_validation_probe_repeats", 3) or 3))
        repeats_source = "energy_defaults"

    raw_probe_retries = probe.get(
        "max_acquisition_retries", probe.get("max_reconnect_retries")
    )
    if raw_probe_retries not in (None, ""):
        try:
            max_acquisition_retries = max(0, min(1, int(raw_probe_retries)))
        except Exception:
            max_acquisition_retries = 1
        max_acquisition_retries_source = "profile"
    else:
        max_acquisition_retries = max(
            0, min(1, int(getattr(defaults, "invalid_repeat_max_retries", 1) or 0))
        )
        max_acquisition_retries_source = "energy_defaults.invalid_repeat_max_retries"

    raw_retry_backoff = probe.get(
        "acquisition_retry_backoff_s", probe.get("reconnect_backoff_s")
    )
    if raw_retry_backoff not in (None, ""):
        try:
            acquisition_retry_backoff_s = max(
                0.0, min(60.0, float(raw_retry_backoff))
            )
        except Exception:
            acquisition_retry_backoff_s = 5.0
        acquisition_retry_backoff_source = "profile"
    else:
        acquisition_retry_backoff_s = max(
            0.0,
            min(
                60.0,
                float(
                    getattr(defaults, "invalid_repeat_reconnect_backoff_s", 5.0)
                    or 0.0
                ),
            ),
        )
        acquisition_retry_backoff_source = (
            "energy_defaults.invalid_repeat_reconnect_backoff_s"
        )

    energy_mode = str(energy.get("mode") or "plan").strip().lower()
    native_energy_requested = bool(
        energy.get("enabled") and energy_mode == "measure"
        or _opt("native_producer_energy_enabled", False)
    )
    explicitly_requested = bool(
        _opt("window_method_validation_probe_enabled", None) is True
        or ("enabled" in probe and bool(probe.get("enabled")))
    )
    effective_enabled = bool(configured_enabled and (native_energy_requested or explicitly_requested))
    probe_resolution = {
        "enabled": effective_enabled,
        "configured_enabled": configured_enabled,
        "enabled_source": enabled_source,
        "activation_reason": (
            "native_energy_measure_requested" if native_energy_requested
            else "probe_explicitly_requested" if explicitly_requested
            else "not_activated_without_native_energy_or_explicit_probe_request"
        ),
        "repeats": repeats,
        "repeats_source": repeats_source,
        "max_acquisition_retries": max_acquisition_retries,
        "max_acquisition_retries_source": max_acquisition_retries_source,
        "acquisition_retry_backoff_s": acquisition_retry_backoff_s,
        "acquisition_retry_backoff_source": acquisition_retry_backoff_source,
        "collector_invalid_repeat_retries": 0,
        "retry_scope": "fresh_outer_collector_attempt_acquisition_integrity_only",
        "minimum_sensitivity_repeats": 3,
        "minimum_decision_repeats": 3,
        "sensitivity_summary_capable_repeat_count": repeats >= 3,
        "decision_capable_repeat_count": repeats >= 3,
        "include_raw_parquet": include_raw,
        "include_raw_parquet_source": raw_source,
        "raw_parquet_required_for_probe": True,
        "strict": strict,
        "strict_source": strict_source,
        "duration_s": float(probe_duration),
        "duration_source": probe_duration_source,
        "screening_only": True,
        "diagnostic_only": True,
        "eligible_for_energy_results_import": False,
        "eligible_for_scientific_claim": False,
        "affects_final_energy_gate": False,
        "workflow_blocking_scope": "none",
        "workflow_blocking_requested": False,
    }
    energy["window_method_validation_probe"] = probe_resolution
    native["energy"] = energy
    resolved["native_producers"] = native
    # Re-project after dataset/hardware/energy materialisation and independently
    # attest the final runtime profile.  This is the profile the worker will
    # consume; no later registry lookup may relax the cache-only contract.
    resolved = apply_cache_verify_only_policy(resolved)
    try:
        cache_verify_attestation = validate_cache_verify_only_profile(resolved)
    except CacheVerifyPolicyError as exc:
        raise StartSnapshotConsistencyError(str(exc)) from exc
    bindings = {
        "runtime_materialized": True,
        "ranking_model_bundle_resolution": copy.deepcopy(dict(ranking or {})),
        "dataset_registry_binding": copy.deepcopy(dict(dataset or {})),
        "window_method_validation_probe": copy.deepcopy(probe_resolution),
        "native_energy_duration_s": float(native_energy_duration),
        "native_energy_duration_source": native_energy_duration_source,
        "hardware_targets": copy.deepcopy(frozen_hardware_targets),
        "hardware_targets_sha256": str(hardware.get("resolved_targets_sha256") or ""),
        "cache_verify_attestation": copy.deepcopy(cache_verify_attestation),
    }
    return resolved, bindings


def resolve_runtime_profile_start_snapshot(
    *,
    profile_request: str,
    source_profile: Mapping[str, Any],
    resolved_profile: Mapping[str, Any],
    profile_id: str,
    profile_path: str,
    profile_source: str,
    options: Any = None,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    runtime_profile, bindings = materialize_runtime_profile(
        resolved_profile,
        profile_path=profile_path,
        options=options,
    )
    snapshot = build_profile_start_snapshot(
        profile_request=profile_request,
        source_profile=source_profile,
        resolved_profile=runtime_profile,
        profile_id=profile_id,
        profile_path=profile_path,
        profile_source=profile_source,
        runtime_bindings=bindings,
    )
    return runtime_profile, snapshot


def validate_profile_start_snapshot(snapshot: Mapping[str, Any]) -> Dict[str, Any]:
    """Revalidate an in-memory GUI snapshot immediately before worker use."""

    data = copy.deepcopy(dict(snapshot or {}))
    if (
        data.get("schema") != "onnx-splitpoint/evaluation-start-snapshot"
        or data.get("schema_version") not in {1, 2}
    ):
        raise StartSnapshotConsistencyError("Evaluation start snapshot is missing or has an unsupported schema.")
    expected_snapshot_hash = str(data.get("snapshot_sha256") or "")
    actual_snapshot_hash = _canonical_hash({
        key: value for key, value in data.items() if key != "snapshot_sha256"
    })
    if not expected_snapshot_hash or expected_snapshot_hash != actual_snapshot_hash:
        raise StartSnapshotConsistencyError("Evaluation start snapshot hash mismatch; refusing to start with a modified snapshot.")
    resolved_profile = data.get("resolved_profile")
    source_profile = data.get("source_profile")
    if not isinstance(resolved_profile, Mapping) or not isinstance(source_profile, Mapping):
        raise StartSnapshotConsistencyError("Evaluation start snapshot does not contain both source and resolved profiles.")
    validate_profile_config_booleans(source_profile)
    validate_profile_config_booleans(resolved_profile)
    if str(data.get("resolved_profile_sha256") or "") != _canonical_hash(resolved_profile):
        raise StartSnapshotConsistencyError("Resolved Evaluation Profile hash mismatch in the start snapshot.")
    if str(data.get("source_profile_sha256") or "") != _canonical_hash(source_profile):
        raise StartSnapshotConsistencyError("Source Evaluation Profile hash mismatch in the start snapshot.")

    rebuilt = build_profile_start_snapshot(
        profile_request=str(data.get("profile_request") or ""),
        source_profile=source_profile,
        resolved_profile=resolved_profile,
        profile_id=str(data.get("profile_id") or ""),
        profile_path=str(data.get("profile_path") or ""),
        profile_source=str(data.get("profile_source") or ""),
        runtime_bindings=(data.get("runtime_bindings") if isinstance(data.get("runtime_bindings"), Mapping) else {}),
        schema_version=int(data.get("schema_version") or 1),
    )
    for key, label in (
        ("resolved_execution_sha256", "resolved execution hash"),
        ("selection_fingerprint", "selection fingerprint"),
        ("snapshot_sha256", "canonical snapshot hash"),
    ):
        if str(rebuilt.get(key) or "") != str(data.get(key) or ""):
            raise StartSnapshotConsistencyError(
                f"Evaluation start {label} changed during validation."
            )
    return data


def public_start_snapshot_metadata(snapshot: Mapping[str, Any]) -> Dict[str, Any]:
    """Return manifest-safe metadata without duplicating full profile payloads."""

    data = dict(snapshot or {})
    return {
        key: copy.deepcopy(data.get(key))
        for key in (
            "schema",
            "schema_version",
            "profile_request",
            "profile_id",
            "profile_path",
            "profile_source",
            "requested_selection",
            "resolved_selection",
            "source_profile_sha256",
            "resolved_profile_sha256",
            "resolved_execution_sha256",
            "selection_fingerprint",
            "cache_verify_attestation",
            "runtime_bindings",
            "snapshot_sha256",
            "consistency",
        )
        if key in data
    }


def start_snapshot_matches_preview(
    preview: Mapping[str, Any] | None,
    current: Mapping[str, Any] | None,
    *,
    profile_request: str,
) -> bool:
    """Return true only for the exact execution-stable profile shown in GUI."""

    shown = dict(preview or {})
    actual = dict(current or {})
    if not shown or not actual:
        return False
    if str(shown.get("profile_request") or "") != str(profile_request or ""):
        return False
    for key in (
        "source_profile_sha256",
        "resolved_execution_sha256",
        "selection_fingerprint",
    ):
        if not str(shown.get(key) or "") or str(shown.get(key) or "") != str(actual.get(key) or ""):
            return False
    return True
