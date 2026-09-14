from __future__ import annotations

"""Canonical Native performance effort contract.

The resolved top-level ``native_producers`` block is the execution authority.
The immutable run-mode snapshot supplies defaults only; it is never a second
source of runtime values.  Tool/release versions are deliberately absent from
the contract so an unchanged measurement request remains reusable.
"""

import hashlib
import json
from typing import Any, Mapping


NATIVE_EXECUTION_CONTRACT_SCHEMA = (
    "onnx-splitpoint/native-execution-contract"
)
NATIVE_EXECUTION_CONTRACT_VERSION = 1
NATIVE_EXECUTION_FIELDS = (
    "frames",
    "warmup",
    "repetitions",
    "queue_depth",
    "inflight",
)

_MODE_DEFAULTS = {
    "smoke": {
        "frames": 100,
        "warmup": 10,
        "repetitions": 1,
        "queue_depth": 2,
        "inflight": 4,
    },
    "standard": {
        "frames": 1000,
        "warmup": 100,
        "repetitions": 3,
        "queue_depth": 3,
        "inflight": 8,
    },
    "final": {
        # Final Quality is Standard with a larger validation/bootstrap budget;
        # its compact fallback must therefore retain Standard effort as well.
        "frames": 1000,
        "warmup": 100,
        "repetitions": 3,
        "queue_depth": 3,
        "inflight": 8,
    },
}


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _strict_nonnegative_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"native_execution_contract_{field}_invalid")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"native_execution_contract_{field}_invalid"
        ) from exc
    minimum = 0 if field == "warmup" else 1
    if parsed < minimum:
        raise ValueError(f"native_execution_contract_{field}_invalid")
    return parsed


def _run_mode_native_defaults(
    profile: Mapping[str, Any],
) -> Mapping[str, Any]:
    preset = (
        profile.get("execution_preset")
        if isinstance(profile.get("execution_preset"), Mapping)
        else {}
    )
    snapshot = (
        preset.get("snapshot")
        if isinstance(preset.get("snapshot"), Mapping)
        else {}
    )
    runtime = (
        snapshot.get("runtime")
        if isinstance(snapshot.get("runtime"), Mapping)
        else {}
    )
    return (
        runtime.get("native")
        if isinstance(runtime.get("native"), Mapping)
        else {}
    )


def build_native_execution_contract(
    config: Mapping[str, Any],
    *,
    run_mode: str = "",
    default_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one exact effort contract from an already resolved config."""

    defaults = dict(default_config or {})
    builtin_defaults = _MODE_DEFAULTS.get(
        str(run_mode or "standard").strip().lower(),
        _MODE_DEFAULTS["standard"],
    )
    values: dict[str, int] = {}
    sources: dict[str, str] = {}
    for field in NATIVE_EXECUTION_FIELDS:
        if field in config and config.get(field) is not None:
            raw = config.get(field)
            source = "profile"
        elif field in defaults and defaults.get(field) is not None:
            raw = defaults.get(field)
            source = "mode_default"
        else:
            raw = builtin_defaults[field]
            source = "builtin_default"
        values[field] = _strict_nonnegative_int(raw, field=field)
        sources[field] = source
    body: dict[str, Any] = {
        "schema": NATIVE_EXECUTION_CONTRACT_SCHEMA,
        "schema_version": NATIVE_EXECUTION_CONTRACT_VERSION,
        "run_mode": str(run_mode or "").strip().lower(),
        **values,
        "field_sources": sources,
    }
    body["contract_sha256"] = _canonical_sha256(body)
    return body


def resolve_native_execution_contract(
    profile: Mapping[str, Any],
    *,
    explicit_cli_overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve the canonical contract used by plan and execution."""

    top_level = (
        dict(profile.get("native_producers") or {})
        if isinstance(profile.get("native_producers"), Mapping)
        else {}
    )
    explicit = dict(explicit_cli_overrides or {})
    for field in NATIVE_EXECUTION_FIELDS:
        if field in explicit and explicit.get(field) is not None:
            top_level[field] = explicit[field]
    preset = (
        profile.get("execution_preset")
        if isinstance(profile.get("execution_preset"), Mapping)
        else {}
    )
    contract = build_native_execution_contract(
        top_level,
        run_mode=str(preset.get("id") or "standard"),
        default_config=_run_mode_native_defaults(profile),
    )
    for field in NATIVE_EXECUTION_FIELDS:
        if field in explicit and explicit.get(field) is not None:
            contract["field_sources"][field] = "explicit_cli"
    unsigned = dict(contract)
    unsigned.pop("contract_sha256", None)
    contract["contract_sha256"] = _canonical_sha256(unsigned)
    return contract


def verify_native_execution_contract(
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    supplied = dict(contract)
    expected_sha = str(supplied.pop("contract_sha256", "")).strip().lower()
    if (
        supplied.get("schema") != NATIVE_EXECUTION_CONTRACT_SCHEMA
        or supplied.get("schema_version") != NATIVE_EXECUTION_CONTRACT_VERSION
        or not expected_sha
        or _canonical_sha256(supplied) != expected_sha
    ):
        raise ValueError("native_execution_contract_identity_invalid")
    sources = supplied.get("field_sources")
    if not isinstance(sources, Mapping):
        raise ValueError("native_execution_contract_field_sources_invalid")
    for field in NATIVE_EXECUTION_FIELDS:
        supplied[field] = _strict_nonnegative_int(
            supplied.get(field), field=field,
        )
        if not str(sources.get(field) or "").strip():
            raise ValueError(
                f"native_execution_contract_{field}_source_invalid"
            )
    supplied["contract_sha256"] = expected_sha
    return supplied


def enforce_native_variant_execution_contract(
    variant: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Reject silent per-variant effort drift; identical repeats are harmless."""

    verified = verify_native_execution_contract(contract)
    for field in NATIVE_EXECUTION_FIELDS:
        if field not in variant or variant.get(field) is None:
            continue
        observed = _strict_nonnegative_int(variant.get(field), field=field)
        if observed != verified[field]:
            raise ValueError(
                "native_execution_contract_variant_override_forbidden:"
                f"{field}:{observed}!={verified[field]}"
            )
    return verified
