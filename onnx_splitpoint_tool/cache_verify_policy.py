from __future__ import annotations

from .config_values import parse_config_bool

"""Fail-closed cache-verification policy for small hardware canaries.

``cache_verify_only`` is deliberately orthogonal to Smoke/Standard/Final.  A
run may therefore retain an explicitly attested compiler identity
(calibration, optimisation and preprocessing) while every compiler/build
dispatch is forbidden.  Exact cache restoration and Hailo v2-to-v3 receipt
migration remain allowed; a miss is a terminal pre-dispatch result.
"""

import copy
import contextlib
import contextvars
import hashlib
import json
import os
import re
import threading
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence


CACHE_VERIFY_ONLY = "cache_verify_only"
ARTIFACT_POLICY_ENV = "ONNX_SPLITPOINT_ARTIFACT_POLICY"
_ACTIVE_ARTIFACT_POLICY: contextvars.ContextVar[str] = contextvars.ContextVar(
    "onnx_splitpoint_active_artifact_policy", default="normal"
)
_POLICY_DEPTH_LOCK = threading.RLock()
_CACHE_VERIFY_PROCESS_DEPTH = 0


class CacheVerifyPolicyError(ValueError):
    """Raised before run mutation when the cache canary contract drifts."""


def _token(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _truth(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, Mapping):
        return _truth(value.get("enabled")) if "enabled" in value else bool(value)
    return _token(value) in {"1", "true", "yes", "on", "enabled"}


def _enabled_rows(value: Any) -> list[Mapping[str, Any]]:
    return [
        row
        for row in list(value or [])
        if isinstance(row, Mapping) and _truth(row.get("enabled", True))
    ]


def _unique_strings(value: Any) -> list[str]:
    if value is None:
        rows: Sequence[Any] = []
    elif isinstance(value, str):
        rows = value.replace(";", ",").split(",")
    elif isinstance(value, Sequence):
        rows = value
    else:
        rows = [value]
    out: list[str] = []
    for item in rows:
        text = str(item or "").strip()
        if text and text not in out:
            out.append(text)
    return out


def _case_map(value: Any) -> Dict[str, list[str]]:
    if not isinstance(value, Mapping):
        return {}
    out: Dict[str, list[str]] = {}
    for raw_model, raw_cases in value.items():
        model = str(raw_model or "").strip()
        if not model:
            continue
        cases = sorted(_unique_strings(raw_cases))
        if cases:
            out[model] = cases
    return dict(sorted(out.items()))


def _task_item_counts(value: Any) -> Dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    out: Dict[str, int] = {}
    for task in ("classification", "detection"):
        raw = value.get(task)
        if type(raw) is not int or int(raw) < 1:
            continue
        out[task] = int(raw)
    return out


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _compatible_hailo_receipt_arch(backend: str, hw_arch: Any) -> bool:
    expected = _token(backend)
    observed = _token(hw_arch)
    if expected == "hailo8":
        return observed == "hailo8"
    if expected in {"hailo10", "hailo10h"}:
        # Older receipts used the generic family name, which DFC resolves to
        # Hailo-10H. Hailo-10P is deliberately not interchangeable.
        return observed in {"hailo10", "hailo10h"}
    return False


def _receipt_bound_hailo_hef(
    hef: Path,
    *,
    backend: str,
    model_id: str,
    case_id: str,
) -> tuple[Dict[str, Any] | None, str]:
    """Verify the minimum cryptographic receipt binding for one HEF."""

    receipt_path = hef.parent / "hailo_hef_build_receipt.json"
    if not receipt_path.is_file():
        return None, "receipt_missing"
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, f"receipt_json_invalid:{type(exc).__name__}"
    if not isinstance(receipt, Mapping):
        return None, "receipt_json_not_object"

    try:
        actual_size = int(hef.stat().st_size)
        actual_sha256 = _file_sha256(hef)
    except Exception as exc:
        return None, f"hef_read_failed:{type(exc).__name__}"
    receipt_size = receipt.get("hef_size_bytes")
    receipt_sha256 = str(receipt.get("hef_sha256") or "").strip().lower()
    receipt_hw_arch = str(receipt.get("hw_arch") or "").strip()
    if type(receipt_size) is not int or receipt_size != actual_size:
        return None, "receipt_hef_size_mismatch"
    if receipt_sha256 != actual_sha256:
        return None, "receipt_hef_sha256_mismatch"
    if not _compatible_hailo_receipt_arch(backend, receipt_hw_arch):
        return None, "receipt_hw_arch_mismatch"

    cache_payload = receipt.get("cache_payload")
    if not isinstance(cache_payload, Mapping):
        return None, "receipt_cache_payload_missing"
    if str(cache_payload.get("schema") or "") != (
        "onnx-splitpoint/hailo-hef-cache-key-v3"
    ):
        return None, "receipt_cache_payload_not_v3"
    try:
        # Local import avoids a module cycle: hailo_backend imports this policy
        # for its last compiler fences.  The validator executes no compiler
        # code and verifies the full receipt/cache/preprocessing identity.
        from .hailo_backend import _load_valid_hailo_receipt

        validated_receipt = _load_valid_hailo_receipt(hef)
    except Exception as exc:
        return None, f"receipt_semantic_validation_failed:{type(exc).__name__}"
    if validated_receipt is None:
        return None, "receipt_semantic_contract_invalid"
    case_match = re.fullmatch(r"b0*(\d+)", _token(case_id))
    if case_match is None:
        return None, "receipt_expected_case_invalid"
    expected_net_name = (
        f"{Path(str(model_id or '').strip()).stem}_part1_b"
        f"{int(case_match.group(1))}"
    )
    receipt_net_name = str(validated_receipt.get("net_name") or "").strip()
    if receipt_net_name != expected_net_name:
        return None, "receipt_net_name_mismatch"

    return {
        "status": "receipt_bound",
        "hef_path": str(hef),
        "hef_sha256": actual_sha256,
        "hef_size_bytes": actual_size,
        "receipt_path": str(receipt_path),
        "receipt_schema": str(receipt.get("schema") or ""),
        "receipt_hw_arch": receipt_hw_arch,
        "receipt_cache_key": str(validated_receipt.get("cache_key") or ""),
        "receipt_net_name": receipt_net_name,
        "expected_net_name": expected_net_name,
        "receipt_sdk_version": str(
            validated_receipt.get("hailo_sdk_version") or ""
        ),
        "expected_backend": str(backend),
    }, ""


def cache_verify_guard(profile: Mapping[str, Any]) -> Dict[str, Any]:
    raw = profile.get("execution_guard") if isinstance(profile, Mapping) else None
    guard = dict(raw or {}) if isinstance(raw, Mapping) else {}
    return guard if _token(guard.get("mode")) == CACHE_VERIFY_ONLY else {}


def cache_verify_only_enabled(profile: Mapping[str, Any] | None = None) -> bool:
    with _POLICY_DEPTH_LOCK:
        process_bound = _CACHE_VERIFY_PROCESS_DEPTH > 0
    if isinstance(profile, Mapping) and cache_verify_guard(profile):
        return True
    return (
        process_bound
        or
        _token(_ACTIVE_ARTIFACT_POLICY.get()) == CACHE_VERIFY_ONLY
        or _token(os.environ.get(ARTIFACT_POLICY_ENV)) == CACHE_VERIFY_ONLY
    )


@contextlib.contextmanager
def bind_artifact_policy(profile: Mapping[str, Any] | None = None):
    """Bind the resolved policy to the current execution context.

    Context-local propagation closes in-process compiler paths without mutating
    the process environment.  Child processes still receive the explicit
    environment/CLI contract at their launch boundary.
    """

    global _CACHE_VERIFY_PROCESS_DEPTH
    policy = CACHE_VERIFY_ONLY if cache_verify_only_enabled(profile) else "normal"
    token = _ACTIVE_ARTIFACT_POLICY.set(policy)
    process_bound = policy == CACHE_VERIFY_ONLY
    if process_bound:
        # Internal generator/compiler workers may use ordinary threads, which
        # intentionally do not inherit ContextVars.  A process-wide fail-closed
        # depth keeps the last compiler fences active in those workers.  A
        # concurrent normal run is therefore blocked rather than permitted to
        # compile while a cache-verification canary owns this process.
        with _POLICY_DEPTH_LOCK:
            _CACHE_VERIFY_PROCESS_DEPTH += 1
    try:
        yield policy
    finally:
        if process_bound:
            with _POLICY_DEPTH_LOCK:
                _CACHE_VERIFY_PROCESS_DEPTH = max(
                    0, _CACHE_VERIFY_PROCESS_DEPTH - 1
                )
        _ACTIVE_ARTIFACT_POLICY.reset(token)


def compiler_dispatch_forbidden(profile: Mapping[str, Any] | None = None) -> bool:
    """Public low-level fence used immediately before compiler processes."""

    return cache_verify_only_enabled(profile)


def cache_miss_blocked_message(family: str, detail: str = "") -> str:
    suffix = f": {str(detail).strip()}" if str(detail or "").strip() else ""
    return (
        f"cache_miss_blocked[{str(family or 'artifact').strip()}]{suffix}; "
        "artifact policy cache_verify_only forbids compiler/build dispatch"
    )


def _native_case_map(native: Mapping[str, Any]) -> Dict[str, list[str]]:
    merged: Dict[str, list[str]] = {}
    direct = _case_map(native.get("case_map"))
    for model, cases in direct.items():
        merged.setdefault(model, []).extend(cases)
    for variant in _enabled_rows(native.get("variants")):
        for model, cases in _case_map(variant.get("case_map")).items():
            merged.setdefault(model, []).extend(cases)
    return {
        model: sorted(set(cases))
        for model, cases in sorted(merged.items())
        if cases
    }


def _native_rows(profile: Mapping[str, Any]) -> list[Dict[str, str]]:
    suite = profile.get("model_suite") if isinstance(profile.get("model_suite"), Mapping) else {}
    models = [
        str(row.get("id") or "").strip()
        for row in _enabled_rows(suite.get("primary"))
        if str(row.get("id") or "").strip()
    ]
    native = profile.get("native_producers") if isinstance(profile.get("native_producers"), Mapping) else {}
    backends = _unique_strings(native.get("backends"))
    setups = _unique_strings(
        (profile.get("hardware") or {}).get("selected_setups")
        if isinstance(profile.get("hardware"), Mapping)
        else []
    )
    default_setup = setups[0] if len(setups) == 1 else ""
    rows: list[Dict[str, str]] = []
    variants = _enabled_rows(native.get("variants"))
    if not variants:
        variants = [{"id": "default", "case_map": _native_case_map(native)}]
    for variant in variants:
        variant_id = str(variant.get("id") or "default").strip() or "default"
        cases = _case_map(variant.get("case_map")) or _case_map(native.get("case_map"))
        variant_backends = _unique_strings(variant.get("backends")) or backends
        setup_id = str(variant.get("setup_id") or default_setup).strip()
        for model in models:
            for case_id in cases.get(model, []):
                for backend in variant_backends:
                    rows.append({
                        "model_id": model,
                        "case_id": case_id,
                        "backend": backend,
                        "setup_id": setup_id,
                        "variant_id": variant_id,
                    })
    return sorted(
        rows,
        key=lambda row: (
            row["model_id"], row["case_id"], row["backend"],
            row["setup_id"], row["variant_id"],
        ),
    )


def cache_verify_contract_view(profile: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the exact execution-facing matrix asserted by the guard."""

    preset = profile.get("execution_preset") if isinstance(profile.get("execution_preset"), Mapping) else {}
    suite = profile.get("model_suite") if isinstance(profile.get("model_suite"), Mapping) else {}
    selection = profile.get("selection_policy") if isinstance(profile.get("selection_policy"), Mapping) else {}
    workflow = profile.get("workflow") if isinstance(profile.get("workflow"), Mapping) else {}
    hailo = profile.get("hailo_build") if isinstance(profile.get("hailo_build"), Mapping) else {}
    deepx = profile.get("deepx_build") if isinstance(profile.get("deepx_build"), Mapping) else {}
    scheduler = profile.get("build_scheduler") if isinstance(profile.get("build_scheduler"), Mapping) else {}
    native = profile.get("native_producers") if isinstance(profile.get("native_producers"), Mapping) else {}
    native_energy = native.get("energy") if isinstance(native.get("energy"), Mapping) else {}
    native_full = native.get("full_baselines") if isinstance(native.get("full_baselines"), Mapping) else {}
    hardware = profile.get("hardware") if isinstance(profile.get("hardware"), Mapping) else {}

    full_backends = _unique_strings(native_full.get("backends")) if _truth(native_full.get("enabled")) else []
    force_variants = [
        str(row.get("id") or f"variant_{idx}")
        for idx, row in enumerate(_enabled_rows(native.get("variants")), start=1)
        if _truth(row.get("force_rebuild_engines"))
        or _truth(row.get("native_force_rebuild_engines"))
    ]
    return {
        "run_mode": str(preset.get("id") or "").strip(),
        "calibration_items": _task_item_counts(
            (preset.get("effective") or {}).get("calibration_items")
            if isinstance(preset.get("effective"), Mapping) else {}
        ),
        "models": [
            str(row.get("id") or "").strip()
            for row in _enabled_rows(suite.get("primary"))
            if str(row.get("id") or "").strip()
        ],
        "logical_run_profiles": [
            str(row.get("id") or row.get("full") or "").strip()
            for row in _enabled_rows(profile.get("run_profiles"))
            if str(row.get("id") or row.get("full") or "").strip()
        ],
        "forced_case_map": _case_map(selection.get("forced_cases")),
        "native_backends": _unique_strings(native.get("backends")),
        "native_case_map": _native_case_map(native),
        "native_rows": _native_rows(profile),
        "native_full_backends": sorted(full_backends),
        "hardware_setup_ids": _unique_strings(hardware.get("selected_setups")),
        "generic_runtime_enabled": not _truth(workflow.get("skip_runtime_benchmarks")),
        "native_enabled": _truth(native.get("enabled")),
        "native_energy_enabled": bool(
            _truth(native_energy.get("enabled"))
            and _token(native_energy.get("mode") or "measure") == "measure"
        ),
        "native_frames": int(native.get("frames") or 0),
        "native_warmup": int(native.get("warmup") or 0),
        "native_repetitions": int(native.get("repetitions") or 1),
        "hailo_targets": _unique_strings(hailo.get("targets"))
        or _unique_strings(hailo.get("hw_arch")),
        "hailo_build_full": _truth(hailo.get("build_full")),
        "hailo_build_part1": _truth(hailo.get("build_part1")),
        "hailo_build_part2": _truth(hailo.get("build_part2")),
        "hailo_preset": _token(hailo.get("preset")),
        "hailo_optimization_level": int(hailo.get("optimization_level") or 0),
        "hailo_calib_count": int(
            hailo.get("calib_count") or hailo.get("calibration_items") or 0
        ),
        "hailo_calib_batch_size": int(
            hailo.get("calib_batch_size")
            or hailo.get("calibration_batch_size") or 0
        ),
        "hailo_calibration_storage": _token(
            hailo.get("calibration_storage")
        ),
        "hailo_calibration_memory_cap_mb": int(
            hailo.get("calibration_memory_cap_mb") or 0
        ),
        "hailo_cache_integrity": _token(hailo.get("cache_integrity")),
        "hailo_build_mode": _token(hailo.get("mode")),
        "hailo_force_build": parse_config_bool(hailo.get("force_build", False), field="hailo_build.force_build"),
        "deepx_build_mode": _token(deepx.get("mode")),
        "deepx_force_build": parse_config_bool(deepx.get("force_build", False), field="deepx_build.force_build"),
        "native_build_missing_engines": _truth(native.get("build_missing_engines")),
        "native_force_rebuild_variants": force_variants,
        "deepx_prefetch_enabled": _truth(scheduler.get("prefetch_deepx_full")),
    }


def apply_cache_verify_only_policy(profile: Mapping[str, Any]) -> Dict[str, Any]:
    """Project immutable no-build controls after normal run-mode resolution."""

    resolved = copy.deepcopy(dict(profile or {}))
    guard = cache_verify_guard(resolved)
    if not guard:
        return resolved
    expected = guard.get("expected_plan") if isinstance(guard.get("expected_plan"), Mapping) else {}
    if not expected:
        raise CacheVerifyPolicyError(
            "cache_verify_only requires a non-empty execution_guard.expected_plan"
        )
    missing_expected = sorted(_MANDATORY_EXPECTED_FIELDS - set(expected))
    if missing_expected:
        raise CacheVerifyPolicyError(
            "cache_verify_only expected_plan is incomplete: "
            + ", ".join(missing_expected)
        )

    def required_int(field: str, *, minimum: int) -> int:
        value = expected.get(field)
        if type(value) is not int or int(value) < minimum:
            raise CacheVerifyPolicyError(
                f"cache_verify_only expected_plan.{field} must be an integer "
                f">= {minimum}"
            )
        return int(value)

    def required_token(field: str) -> str:
        value = str(expected.get(field) or "").strip()
        if not value:
            raise CacheVerifyPolicyError(
                f"cache_verify_only expected_plan.{field} must not be empty"
            )
        return value

    workflow = dict(resolved.get("workflow") or {})
    workflow["execution_mode"] = "generate_and_run"
    workflow["skip_runtime_benchmarks"] = True
    workflow["cache_verify_only"] = True
    resolved["workflow"] = workflow

    calibration_items = _task_item_counts(expected.get("calibration_items"))
    if set(calibration_items) != {"classification", "detection"}:
        raise CacheVerifyPolicyError(
            "cache_verify_only requires positive classification and detection "
            "calibration_items in execution_guard.expected_plan"
        )

    preset = dict(resolved.get("execution_preset") or {})
    effective = dict(preset.get("effective") or {})
    effective["calibration_items"] = copy.deepcopy(calibration_items)
    effective["cache_identity_source"] = "execution_guard.expected_plan"
    preset["effective"] = effective

    # The central Tool Config remains recorded by config_sha256, but the
    # canary's artifact-producing axes are frozen by the guard.  This prevents
    # an edited Smoke mode from silently addressing another HEF while keeping
    # the resolved snapshot internally self-consistent.
    snapshot = copy.deepcopy(dict(preset.get("snapshot") or {}))
    snapshot_data = dict(snapshot.get("data") or {})
    snapshot_data["calibration_items"] = copy.deepcopy(calibration_items)
    snapshot["data"] = snapshot_data
    snapshot_build = dict(snapshot.get("build") or {})
    snapshot_hailo = dict(snapshot_build.get("hailo") or {})

    hailo_identity = {
        "preset": required_token("hailo_preset"),
        "optimization_level": required_int(
            "hailo_optimization_level", minimum=0,
        ),
        "calibration_items": required_int("hailo_calib_count", minimum=1),
        "calibration_batch_size": required_int(
            "hailo_calib_batch_size", minimum=1,
        ),
        "calibration_storage": required_token("hailo_calibration_storage"),
        "calibration_memory_cap_mb": required_int(
            "hailo_calibration_memory_cap_mb", minimum=32,
        ),
        "cache_integrity": required_token("hailo_cache_integrity"),
    }
    snapshot_hailo.update(hailo_identity)
    snapshot_build["hailo"] = snapshot_hailo
    snapshot["build"] = snapshot_build
    preset["snapshot"] = snapshot
    preset["snapshot_sha256"] = hashlib.sha256(json.dumps(
        snapshot, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), default=str,
    ).encode("utf-8")).hexdigest()
    resolved["execution_preset"] = preset

    hailo = dict(resolved.get("hailo_build") or {})
    hailo.update({
        "mode": CACHE_VERIFY_ONLY,
        "force_build": False,
        "build_full": _truth(expected.get("hailo_build_full")),
        "build_part1": _truth(expected.get("hailo_build_part1")),
        "build_part2": _truth(expected.get("hailo_build_part2")),
        "preset": hailo_identity["preset"],
        "optimization_level": hailo_identity["optimization_level"],
        "calib_count": hailo_identity["calibration_items"],
        "calib_batch_size": hailo_identity["calibration_batch_size"],
        "calibration_storage": hailo_identity["calibration_storage"],
        "calibration_memory_cap_mb": hailo_identity[
            "calibration_memory_cap_mb"
        ],
        "cache_integrity": hailo_identity["cache_integrity"],
    })
    resolved["hailo_build"] = hailo

    deepx = dict(resolved.get("deepx_build") or {})
    deepx.update({"mode": CACHE_VERIFY_ONLY, "force_build": False})
    resolved["deepx_build"] = deepx

    scheduler = dict(resolved.get("build_scheduler") or {})
    scheduler.update({"prefetch_deepx_full": False, "pipeline_next_model": False})
    resolved["build_scheduler"] = scheduler

    native = dict(resolved.get("native_producers") or {})
    native["build_missing_engines"] = False
    native["cache_verify_only"] = True
    native["smoke_diagnostic_quality_continue"] = True
    variants = []
    for row in list(native.get("variants") or []):
        if not isinstance(row, Mapping):
            continue
        item = copy.deepcopy(dict(row))
        item["force_rebuild_engines"] = False
        item["native_force_rebuild_engines"] = False
        variants.append(item)
    if "variants" in native:
        native["variants"] = variants
    resolved["native_producers"] = native

    quality = dict(resolved.get("quality_gate") or {})
    quality["diagnostic_only"] = True
    quality["claim_eligible"] = False
    enforcement = dict(quality.get("enforcement") or {})
    enforcement.update({
        "technical_quality_error": "partial_continue_diagnostic",
        "metric_threshold_miss": "warning_only",
        "native_energy_after_technical_error": "continue_available_diagnostics",
        "claim_eligibility": "always_false",
    })
    quality["enforcement"] = enforcement
    resolved["quality_gate"] = quality

    remote_cache = dict(resolved.get("remote_cache") or {})
    remote_cache["reuse_tensorrt_engines"] = True
    resolved["remote_cache"] = remote_cache

    setup_ids = _unique_strings(expected.get("hardware_setup_ids"))
    if setup_ids:
        hardware = dict(resolved.get("hardware") or {})
        hardware["selected_setups"] = setup_ids
        hardware["selected_groups"] = []
        resolved["hardware"] = hardware

    guard = copy.deepcopy(guard)
    guard["mode"] = CACHE_VERIFY_ONLY
    guard["policy_sha256"] = _canonical_sha256({
        "mode": CACHE_VERIFY_ONLY,
        "expected_plan": expected,
    })
    guard["compiler_dispatch"] = {
        "hailo_dfc": False,
        "deepx_dx_com": False,
        "tensorrt_build": False,
    }
    resolved["execution_guard"] = guard
    preset = dict(resolved.get("execution_preset") or {})
    effective = dict(preset.get("effective") or {})
    effective.update({
        "artifact_policy": CACHE_VERIFY_ONLY,
        "compiler_dispatch_allowed": False,
    })
    preset["effective"] = effective
    resolved["execution_preset"] = preset
    return resolved


_MANDATORY_EXPECTED_FIELDS = {
    "run_mode",
    "calibration_items",
    "models",
    "logical_run_profiles",
    "forced_case_map",
    "native_backends",
    "native_case_map",
    "native_rows",
    "native_full_backends",
    "hardware_setup_ids",
    "generic_runtime_enabled",
    "native_enabled",
    "native_energy_enabled",
    "native_frames",
    "native_warmup",
    "native_repetitions",
    "hailo_targets",
    "hailo_build_full",
    "hailo_build_part1",
    "hailo_build_part2",
    "hailo_preset",
    "hailo_optimization_level",
    "hailo_calib_count",
    "hailo_calib_batch_size",
    "hailo_calibration_storage",
    "hailo_calibration_memory_cap_mb",
    "hailo_cache_integrity",
}


def validate_cache_verify_only_profile(profile: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate the resolved matrix and all no-build invariants.

    The returned attestation is safe to persist in the start snapshot.  This
    function performs no writes and starts no processes.
    """

    guard = cache_verify_guard(profile)
    if not guard:
        return {}
    expected = guard.get("expected_plan") if isinstance(guard.get("expected_plan"), Mapping) else {}
    missing_expected = sorted(_MANDATORY_EXPECTED_FIELDS - set(expected))
    if missing_expected:
        raise CacheVerifyPolicyError(
            "cache_verify_only expected_plan is incomplete: "
            + ", ".join(missing_expected)
        )

    actual = cache_verify_contract_view(profile)
    mismatches: list[Dict[str, Any]] = []
    for field, expected_value in expected.items():
        if field not in actual:
            mismatches.append({
                "field": field,
                "expected": copy.deepcopy(expected_value),
                "actual": "<unsupported expected field>",
            })
        elif actual[field] != expected_value:
            mismatches.append({
                "field": field,
                "expected": copy.deepcopy(expected_value),
                "actual": copy.deepcopy(actual[field]),
            })

    safety_expected = {
        "hailo_build_mode": CACHE_VERIFY_ONLY,
        "hailo_force_build": False,
        "deepx_build_mode": CACHE_VERIFY_ONLY,
        "deepx_force_build": False,
        "native_build_missing_engines": False,
        "native_force_rebuild_variants": [],
        "deepx_prefetch_enabled": False,
        "generic_runtime_enabled": False,
    }
    for field, expected_value in safety_expected.items():
        if actual.get(field) != expected_value:
            mismatches.append({
                "field": f"safety.{field}",
                "expected": expected_value,
                "actual": copy.deepcopy(actual.get(field)),
            })
    if actual.get("forced_case_map") != actual.get("native_case_map"):
        mismatches.append({
            "field": "safety.forced_case_map_equals_native_case_map",
            "expected": copy.deepcopy(actual.get("native_case_map")),
            "actual": copy.deepcopy(actual.get("forced_case_map")),
        })

    if mismatches:
        detail = "; ".join(
            f"{row['field']}: expected={row['expected']!r}, actual={row['actual']!r}"
            for row in mismatches[:8]
        )
        raise CacheVerifyPolicyError(
            "cache_verify_only start blocked before run creation: " + detail
        )
    attestation = {
        "schema": "onnx-splitpoint/cache-verify-attestation",
        "schema_version": 1,
        "mode": CACHE_VERIFY_ONLY,
        "status": "verified",
        "compiler_dispatch_allowed": False,
        "expected_plan": copy.deepcopy(dict(expected)),
        "actual_plan": actual,
        "policy_sha256": str(guard.get("policy_sha256") or _canonical_sha256({
            "mode": CACHE_VERIFY_ONLY,
            "expected_plan": expected,
        })),
    }
    attestation["attestation_sha256"] = _canonical_sha256(attestation)
    return attestation


def _benchmark_set_case_ids(benchmark_set: Path) -> list[str]:
    contract_path = Path(benchmark_set) / "benchmark_set.json"
    try:
        payload = json.loads(contract_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise CacheVerifyPolicyError(
            "cache_verify_only BenchmarkSet attestation failed: "
            f"cannot read {contract_path}: {type(exc).__name__}: {exc}"
        ) from exc
    case_ids: list[str] = []
    for row in list(payload.get("cases") or []):
        if isinstance(row, Mapping):
            case_id = str(
                row.get("id")
                or row.get("case_id")
                or row.get("case")
                or row.get("case_dir")
                or row.get("folder")
                or ""
            ).strip()
            if not case_id:
                try:
                    case_id = f"b{int(row.get('boundary', row.get('split_index'))):03d}"
                except Exception:
                    case_id = ""
        else:
            case_id = str(row or "").strip()
        if case_id.isdigit():
            case_id = f"b{int(case_id):03d}"
        if case_id:
            case_ids.append(case_id)
    return sorted(set(case_ids))


def attest_cache_verify_benchmark_sets(
    profile: Mapping[str, Any],
    benchmark_sets: Mapping[str, Path],
) -> Dict[str, Any]:
    """Attest the generated physical suites before any Native/SSH dispatch.

    This second gate deliberately observes the generated directories instead
    of trusting the profile a second time.  It also requires the producer-side
    artifact and its receipt to exist; remote TensorRT cache identity is
    independently checked by the Quality-FIRST binding on the target host.
    """

    attestation = validate_cache_verify_only_profile(profile)
    if not attestation:
        return {}
    expected = attestation["expected_plan"]
    expected_models = sorted(str(value) for value in expected["models"])
    actual_models = sorted(str(value) for value in benchmark_sets)
    failures: list[str] = []
    if actual_models != expected_models:
        failures.append(
            f"models expected={expected_models!r} actual={actual_models!r}"
        )

    observed: Dict[str, Any] = {}
    expected_case_map = _case_map(expected.get("native_case_map"))
    expected_rows = [dict(row) for row in list(expected.get("native_rows") or [])]
    for model_id in expected_models:
        root = Path(benchmark_sets.get(model_id) or "")
        expected_cases = list(expected_case_map.get(model_id) or [])
        contract_cases = _benchmark_set_case_ids(root) if root else []
        directory_cases = sorted(
            path.name for path in root.glob("b*") if path.is_dir()
        ) if root else []
        if contract_cases != expected_cases:
            failures.append(
                f"{model_id}.contract_cases expected={expected_cases!r} "
                f"actual={contract_cases!r}"
            )
        if directory_cases != expected_cases:
            failures.append(
                f"{model_id}.directory_cases expected={expected_cases!r} "
                f"actual={directory_cases!r}"
            )
        model_artifacts: Dict[str, Any] = {}
        model_receipt_evidence: Dict[str, Any] = {}
        for case_id in expected_cases:
            case_dir = root / case_id
            if not (case_dir / "split_manifest.json").is_file():
                failures.append(f"{model_id}/{case_id}: split_manifest.json missing")
            row_backends = sorted({
                str(row.get("backend") or "").strip().lower()
                for row in expected_rows
                if str(row.get("model_id") or "") == model_id
                and str(row.get("case_id") or "") == case_id
            })
            for backend in row_backends:
                if backend.startswith("hailo"):
                    family = "hailo10" if "10" in backend else "hailo8"
                    roots = list(case_dir.glob(f"hailo/{family}*/part1"))
                    hefs = [
                        path for artifact_root in roots
                        for path in artifact_root.rglob("*.hef")
                        if path.is_file() and path.stat().st_size > 0
                    ]
                    verified: list[tuple[Path, Dict[str, Any]]] = []
                    rejected: list[Dict[str, str]] = []
                    for hef in hefs:
                        evidence, reason = _receipt_bound_hailo_hef(
                            hef,
                            backend=backend,
                            model_id=model_id,
                            case_id=case_id,
                        )
                        relative_hef = str(hef.relative_to(root))
                        if evidence is None:
                            rejected.append({
                                "hef_path": relative_hef,
                                "reason": reason,
                            })
                            continue
                        evidence = dict(evidence)
                        evidence["hef_path"] = relative_hef
                        evidence["receipt_path"] = str(
                            (hef.parent / "hailo_hef_build_receipt.json")
                            .relative_to(root)
                        )
                        verified.append((hef, evidence))
                    if not verified:
                        rejection_detail = ", ".join(
                            f"{row['hef_path']}={row['reason']}"
                            for row in rejected[:4]
                        ) or "no_nonempty_hef_candidate"
                        failures.append(cache_miss_blocked_message(
                            "hailo_dfc",
                            f"no receipt-bound Part1 HEF for "
                            f"{model_id}/{case_id}/{backend} ({rejection_detail})",
                        ))
                    model_artifacts[f"{case_id}:{backend}"] = [
                        str(path.relative_to(root)) for path, _evidence in verified
                    ]
                    model_receipt_evidence[f"{case_id}:{backend}"] = {
                        "verified": [evidence for _path, evidence in verified],
                        "rejected": rejected,
                    }
                elif "deepx" in backend:
                    dxnn = [
                        path for path in (case_dir / "deepx").rglob("*.dxnn")
                        if path.is_file() and path.stat().st_size > 0
                    ] if (case_dir / "deepx").is_dir() else []
                    if not dxnn:
                        failures.append(cache_miss_blocked_message(
                            "deepx_dx_com",
                            f"no prepared DXNN for {model_id}/{case_id}/{backend}",
                        ))
                    model_artifacts[f"{case_id}:{backend}"] = [
                        str(path.relative_to(root)) for path in dxnn
                    ]
        observed[model_id] = {
            "benchmark_set": str(root),
            "contract_cases": contract_cases,
            "directory_cases": directory_cases,
            "artifacts": model_artifacts,
            "artifact_receipt_evidence": model_receipt_evidence,
        }
    if failures:
        raise CacheVerifyPolicyError(
            "cache_verify_only Native pre-dispatch attestation failed: "
            + "; ".join(failures[:12])
        )
    result = {
        "schema": "onnx-splitpoint/cache-verify-benchmarkset-attestation",
        "schema_version": 1,
        "status": "verified",
        "compiler_dispatch_allowed": False,
        "expected_plan_sha256": _canonical_sha256(expected),
        "observed": observed,
    }
    result["attestation_sha256"] = _canonical_sha256(result)
    return result
