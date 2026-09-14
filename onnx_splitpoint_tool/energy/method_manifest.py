"""Prepare and bind the validated full-system u.RECS energy method.

This module deliberately contains no platform-control operations.  Preparing
the method records an explicit human attestation, hashes the exact local
measurement implementation, verifies every configured runtime binding and
only then publishes the manifest reference to the hardware registry.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

from ..campaign import (
    INHERITED_VALIDATED_ENERGY_METHOD_MODE,
    WACHSMUTH_ENERGY_METHOD_REFERENCE,
    create_energy_calibration_manifest,
    verify_energy_calibration_manifest,
)
from ..source_integrity import (
    create_source_integrity_binding,
    verify_installed_source_integrity,
    verify_source_integrity_binding,
)
from ..workflow.artifacts import now_iso
from . import config as energy_config
from .collector import _expand_binary, _verify_calibration_manifest
from .config import EnergyDefaults, energy_defaults_from_registry, load_hardware_registry


STANDARD_PLATFORM_SETUP_IDS: tuple[str, ...] = (
    "orin_nx_hailo8_01",
    "orin_nx_hailo10_01",
    "orin_nx_deepx_m1_01",
)

EXACT_IMPLEMENTATION_REUSE_POLICY = "exact_validated_implementation_reuse"
CANONICAL_IMPLEMENTATION_SOURCE_IDS: tuple[str, ...] = (
    "tool_source_collector",
    "tool_source_config",
    "tool_source_metrics",
    "tool_source_energy_measurement_cli",
    "tool_source_native_producer_energy_plan",
    "tool_source_run_native_producer_energy_from_summary",
)
CANONICAL_IMPLEMENTATION_BINARY_IDS: tuple[str, ...] = (
    "urecs_data_collector_binary",
    "power_calculations_binary",
)
CANONICAL_IMPLEMENTATION_ARTIFACT_IDS: tuple[str, ...] = (
    *CANONICAL_IMPLEMENTATION_SOURCE_IDS,
    *CANONICAL_IMPLEMENTATION_BINARY_IDS,
)


class EnergyMethodManifestError(RuntimeError):
    """The FS energy-method manifest could not be prepared safely."""


class DuplicateHardwareSetupIdError(EnergyMethodManifestError):
    """The hardware registry contains an ambiguous setup identity."""

    def __init__(self, setup_ids: Sequence[str]) -> None:
        self.setup_ids = tuple(sorted(set(str(value) for value in setup_ids)))
        super().__init__(
            "duplicate hardware setup id(s): " + ", ".join(self.setup_ids)
        )


class InvalidEnergyDataPortError(EnergyMethodManifestError):
    """The raw registry data port would otherwise be silently defaulted."""

    def __init__(self, value: Any) -> None:
        self.value = value
        super().__init__(
            "energy_defaults.data_port must be an integer UDP port in range "
            f"1..65535 (got {value!r})"
        )


class InvalidEnergyMethodRegistryContractError(EnergyMethodManifestError):
    """Raw registry values violate the claim-bearing method contract."""

    def __init__(self, errors: Sequence[str]) -> None:
        self.errors = tuple(str(value) for value in errors)
        super().__init__("; ".join(self.errors))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_setup_id(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and value
        and value == value.strip()
        and not any(
            character.isspace()
            or ord(character) < 32
            or ord(character) == 127
            for character in value
        )
    )


def _setup_rows(registry: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows: dict[str, Mapping[str, Any]] = {}
    duplicates: set[str] = set()
    for row in list(registry.get("hardware_setups") or []):
        if not isinstance(row, Mapping):
            continue
        raw_setup_id = row.get("id")
        if not _canonical_setup_id(raw_setup_id):
            raise InvalidEnergyMethodRegistryContractError(
                [
                    "hardware setup id must be a nonempty canonical string "
                    "without whitespace/control characters "
                    f"(got {raw_setup_id!r})"
                ]
            )
        setup_id = raw_setup_id
        if setup_id in rows:
            duplicates.add(setup_id)
            continue
        rows[setup_id] = row
    if duplicates:
        raise DuplicateHardwareSetupIdError(sorted(duplicates))
    return rows


def _target_setup_ids(
    registry: Mapping[str, Any], requested: Sequence[str] | None
) -> tuple[str, ...]:
    if requested:
        raw_values = list(requested)
    else:
        groups = (
            dict(registry.get("hardware_groups") or {})
            if isinstance(registry.get("hardware_groups"), Mapping)
            else {}
        )
        group = groups.get("all_accelerators")
        raw_values = (
            list(group or [])
            if isinstance(group, list)
            else list(STANDARD_PLATFORM_SETUP_IDS)
        )
    invalid = [value for value in raw_values if not _canonical_setup_id(value)]
    if invalid:
        raise InvalidEnergyMethodRegistryContractError(
            [
                "target setup id must be a nonempty canonical string without "
                f"whitespace/control characters (got {value!r})"
                for value in invalid
            ]
        )
    values = list(raw_values)
    unique = tuple(dict.fromkeys(value for value in values if value))
    if not unique:
        raise EnergyMethodManifestError("no target hardware setup was selected")
    return unique


def _strict_raw_energy_defaults(
    registry: Mapping[str, Any],
) -> tuple[int, int, int]:
    """Read numeric method identity without permissive config coercion.

    Missing keys retain documented defaults. Explicit values must already be
    canonical integers; strings, booleans and invalid ranges cannot be
    converted or silently replaced by defaults.
    """
    if "energy_defaults" in registry and not isinstance(
        registry.get("energy_defaults"), Mapping
    ):
        raise InvalidEnergyMethodRegistryContractError(
            ["energy_defaults must be a mapping when explicitly configured"]
        )
    raw_defaults = (
        dict(registry.get("energy_defaults") or {})
        if isinstance(registry.get("energy_defaults"), Mapping)
        else {}
    )
    raw_value = raw_defaults.get("data_port", 3000)
    if (
        isinstance(raw_value, bool)
        or not isinstance(raw_value, int)
        or not 1 <= raw_value <= 65535
    ):
        raise InvalidEnergyDataPortError(raw_value)
    raw_channel = raw_defaults.get("channel", 0)
    raw_sample_rate = raw_defaults.get("sample_rate", 2000)
    errors: list[str] = []
    if "enabled" in raw_defaults and not isinstance(
        raw_defaults.get("enabled"), bool
    ):
        errors.append(
            "energy_defaults.enabled must be a literal boolean when "
            f"configured (got {raw_defaults.get('enabled')!r})"
        )
    if (
        isinstance(raw_channel, bool)
        or not isinstance(raw_channel, int)
        or raw_channel < 0
    ):
        errors.append(
            "energy_defaults.channel must be a non-negative integer "
            f"(got {raw_channel!r})"
        )
    if (
        isinstance(raw_sample_rate, bool)
        or not isinstance(raw_sample_rate, int)
        or raw_sample_rate <= 0
    ):
        errors.append(
            "energy_defaults.sample_rate must be a positive integer "
            f"(got {raw_sample_rate!r})"
        )
    scope_keys = [
        key
        for key in ("physical_scope", "measurement_physical_scope")
        if key in raw_defaults
    ]
    if not scope_keys:
        errors.append(
            "energy_defaults.physical_scope must be explicitly configured "
            "as 'FS' or 'FULL_SYSTEM'"
        )
    for key in scope_keys:
        value = raw_defaults.get(key)
        if not (
            isinstance(value, str) and value in {"FS", "FULL_SYSTEM"}
        ):
            errors.append(
                f"energy_defaults.{key} must be the exact full-system "
                f"value 'FS' or 'FULL_SYSTEM' (got {value!r})"
            )
    if len(scope_keys) == 2 and (
        raw_defaults.get("physical_scope")
        != raw_defaults.get("measurement_physical_scope")
    ):
        errors.append(
            "energy_defaults physical_scope aliases must be exactly equal"
        )

    window_keys = [
        key
        for key in ("window_label", "measurement_window")
        if key in raw_defaults
    ]
    if not window_keys:
        errors.append(
            "energy_defaults.window_label must be explicitly configured "
            "as 'command'"
        )
    for key in window_keys:
        value = raw_defaults.get(key)
        if not (isinstance(value, str) and value == "command"):
            errors.append(
                f"energy_defaults.{key} must be the exact string "
                f"'command' (got {value!r})"
            )
    if len(window_keys) == 2 and (
        raw_defaults.get("window_label")
        != raw_defaults.get("measurement_window")
    ):
        errors.append(
            "energy_defaults window_label aliases must be exactly equal"
        )
    for key in ("collector_binary", "power_calculations_binary"):
        if key not in raw_defaults:
            continue
        value = raw_defaults.get(key)
        if (
            not isinstance(value, str)
            or not value
            or value != value.strip()
            or any(ord(character) < 32 or ord(character) == 127 for character in value)
        ):
            errors.append(
                f"energy_defaults.{key} must be a nonempty canonical string "
                f"(got {value!r})"
            )
    if "mode" in raw_defaults:
        raw_mode = raw_defaults.get("mode")
        if raw_mode != "fast_firmware":
            errors.append(
                "energy_defaults.mode must be the canonical string "
                f"'fast_firmware' (got {raw_mode!r})"
            )
    if errors:
        raise InvalidEnergyMethodRegistryContractError(errors)
    return raw_value, raw_channel, raw_sample_rate


def _prepare_energy_defaults_candidate(
    registry: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the only legacy defaults migration admitted by preparation.

    ``prepare-energy-method`` is an explicit, human-attested and hardware-free
    transition into the claim-bearing full-system method.  Registries written
    before v2.79.12 either omitted the reporting fields or carried the historic
    ``physical_scope: MB`` default.  Those two exact legacy states may be
    upgraded in memory.  Every malformed, padded, conflicting or otherwise
    explicit value is left untouched so :func:`_strict_raw_energy_defaults`
    rejects it before an artefact or registry commit can be published.

    The caller owns persistence.  This helper deliberately performs no I/O.
    """

    candidate = copy.deepcopy(dict(registry or {}))
    raw_defaults = candidate.get("energy_defaults")
    if raw_defaults is None:
        defaults: dict[str, Any] = {}
    elif isinstance(raw_defaults, Mapping):
        defaults = copy.deepcopy(dict(raw_defaults))
    else:
        # Preserve the invalid value for the strict parser and report no
        # migration. Replacing it with a mapping would hide corruption.
        return candidate, {
            "schema": "onnx-splitpoint/energy-defaults-method-migration",
            "schema_version": 1,
            "applied": False,
            "changes": [],
            "effective_physical_scope": None,
            "effective_window_label": None,
        }

    changes: list[dict[str, Any]] = []
    scope_keys = [
        key
        for key in ("physical_scope", "measurement_physical_scope")
        if key in defaults
    ]
    if not scope_keys:
        defaults["physical_scope"] = "FS"
        changes.append(
            {
                "field": "energy_defaults.physical_scope",
                "previous_state": "missing",
                "previous_value": None,
                "new_value": "FS",
                "reason": "explicit_full_system_method_preparation",
            }
        )
    elif all(defaults.get(key) == "MB" for key in scope_keys):
        for key in scope_keys:
            defaults[key] = "FS"
            changes.append(
                {
                    "field": f"energy_defaults.{key}",
                    "previous_state": "historical_default",
                    "previous_value": "MB",
                    "new_value": "FS",
                    "reason": "explicit_full_system_method_preparation",
                }
            )

    window_keys = [
        key
        for key in ("window_label", "measurement_window")
        if key in defaults
    ]
    if not window_keys:
        defaults["window_label"] = "command"
        changes.append(
            {
                "field": "energy_defaults.window_label",
                "previous_state": "missing",
                "previous_value": None,
                "new_value": "command",
                "reason": "explicit_command_window_method_preparation",
            }
        )

    candidate["energy_defaults"] = defaults
    effective_scope = (
        defaults.get("physical_scope")
        if "physical_scope" in defaults
        else defaults.get("measurement_physical_scope")
    )
    effective_window = (
        defaults.get("window_label")
        if "window_label" in defaults
        else defaults.get("measurement_window")
    )
    return candidate, {
        "schema": "onnx-splitpoint/energy-defaults-method-migration",
        "schema_version": 1,
        "applied": bool(changes),
        "changes": changes,
        "effective_physical_scope": effective_scope,
        "effective_window_label": effective_window,
    }


def _normalise_registry_in_memory(registry: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the normal read-time registry merge without writing its source."""

    normalise = getattr(energy_config, "_merge_default_hardware_setups", None)
    if not callable(normalise):  # pragma: no cover - same-package invariant
        raise EnergyMethodManifestError(
            "hardware registry in-memory normalizer is unavailable"
        )
    return normalise(copy.deepcopy(dict(registry or {})))


def _read_raw_registry(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    yaml_module = getattr(energy_config, "yaml", None)
    if yaml_module is None:
        raise InvalidEnergyMethodRegistryContractError(
            ["PyYAML is required to inspect the raw hardware registry"]
        )
    loaded = yaml_module.safe_load(path.read_text(encoding="utf-8"))
    if loaded is None:
        return {}
    if not isinstance(loaded, Mapping):
        raise InvalidEnergyMethodRegistryContractError(
            ["hardware registry root must be a mapping"]
        )
    return dict(loaded)


def _strict_setup_urecs_address(
    setup: Mapping[str, Any],
    *,
    setup_id: str,
    expected_data_port: int,
    expected_channel: int,
    expected_sample_rate: int,
) -> str:
    if "energy" in setup and not isinstance(setup.get("energy"), Mapping):
        raise InvalidEnergyMethodRegistryContractError(
            [f"hardware setup {setup_id} energy must be a mapping"]
        )
    energy = (
        dict(setup.get("energy") or {})
        if isinstance(setup.get("energy"), Mapping)
        else {}
    )
    enabled = energy.get("enabled")
    if enabled is not True:
        raise InvalidEnergyMethodRegistryContractError(
            [
                f"hardware setup {setup_id} energy.enabled must be the "
                f"literal boolean true (got {enabled!r})"
            ]
        )
    numeric_overrides = (
        ("data_port", expected_data_port, 1, 65535),
        ("channel", expected_channel, 0, None),
        ("sample_rate", expected_sample_rate, 1, None),
        ("sample_rate_hz", expected_sample_rate, 1, None),
    )
    numeric_errors: list[str] = []
    for key, expected, minimum, maximum in numeric_overrides:
        if key not in energy:
            continue
        value = energy.get(key)
        valid_range = bool(
            isinstance(value, int)
            and not isinstance(value, bool)
            and value >= minimum
            and (maximum is None or value <= maximum)
        )
        if not valid_range or value != expected:
            numeric_errors.append(
                f"hardware setup {setup_id} energy.{key} must be integer "
                f"{expected} inherited from energy_defaults (got {value!r})"
            )
    if numeric_errors:
        raise InvalidEnergyMethodRegistryContractError(numeric_errors)
    unsupported_overrides = [
        key
        for key in (
            "collector_binary",
            "power_calculations_binary",
            "mode",
            "physical_scope",
            "window_label",
            "measurement_physical_scope",
            "measurement_window",
        )
        if key in energy
    ]
    if unsupported_overrides:
        raise InvalidEnergyMethodRegistryContractError(
            [
                f"hardware setup {setup_id} energy.{key} is a global-only "
                "method field and must not be overridden"
                for key in unsupported_overrides
            ]
        )
    def _canonical_address(value: Any) -> bool:
        return bool(
            isinstance(value, str)
            and value
            and value == value.strip()
            and not any(
                character.isspace()
                or ord(character) < 32
                or ord(character) == 127
                for character in value
            )
        )

    address_keys = [
        key for key in ("urecs_address", "address") if key in energy
    ]
    address_errors: list[str] = []
    for key in address_keys:
        value = energy.get(key)
        if not _canonical_address(value):
            address_errors.append(
                f"u.RECS address alias energy.{key} for setup {setup_id} "
                "must be a nonempty canonical string without "
                f"whitespace/control characters (got {value!r})"
            )
    if len(address_keys) == 2 and (
        energy.get("urecs_address") != energy.get("address")
    ):
        address_errors.append(
            f"u.RECS address aliases for setup {setup_id} must be equal"
        )
    if not address_keys:
        address_errors.append(
            f"u.RECS address for setup {setup_id} is missing"
        )
    if address_errors:
        raise InvalidEnergyMethodRegistryContractError(address_errors)
    raw_address = energy.get(
        "urecs_address", energy.get("address")
    )

    return raw_address


def _strict_raw_target_setups(
    registry: Mapping[str, Any],
    setup_ids: Sequence[str],
    *,
    expected_data_port: int,
    expected_channel: int,
    expected_sample_rate: int,
) -> None:
    if "hardware_setups" in registry and not isinstance(
        registry.get("hardware_setups"), list
    ):
        raise InvalidEnergyMethodRegistryContractError(
            ["hardware_setups must be a list when explicitly configured"]
        )
    raw_rows = list(registry.get("hardware_setups") or [])
    if any(not isinstance(row, Mapping) for row in raw_rows):
        raise InvalidEnergyMethodRegistryContractError(
            ["every hardware_setups row must be a mapping"]
        )
    rows = _setup_rows(registry)
    for setup_id in setup_ids:
        setup = rows.get(str(setup_id))
        if setup is not None:
            _strict_setup_urecs_address(
                setup,
                setup_id=str(setup_id),
                expected_data_port=expected_data_port,
                expected_channel=expected_channel,
                expected_sample_rate=expected_sample_rate,
            )


def _runtime_contract(
    registry: Mapping[str, Any], setup_ids: Sequence[str]
) -> tuple[EnergyDefaults, list[dict[str, Any]]]:
    raw_data_port, raw_channel, raw_sample_rate = (
        _strict_raw_energy_defaults(registry)
    )
    defaults = energy_defaults_from_registry(registry)
    errors: list[str] = []
    if str(defaults.mode or "").strip().lower().replace("-", "_") != "fast_firmware":
        errors.append("energy_defaults.mode must be fast_firmware")
    if isinstance(defaults.channel, bool) or int(defaults.channel) != 0:
        errors.append("energy_defaults.channel must be 0")
    elif int(defaults.channel) != raw_channel:
        errors.append("energy_defaults.channel was not preserved exactly")
    if isinstance(defaults.sample_rate, bool) or int(defaults.sample_rate) != 2000:
        errors.append("energy_defaults.sample_rate must be 2000")
    elif int(defaults.sample_rate) != raw_sample_rate:
        errors.append("energy_defaults.sample_rate was not preserved exactly")
    if (
        isinstance(defaults.data_port, bool)
        or not 1 <= int(defaults.data_port) <= 65535
        or int(defaults.data_port) != raw_data_port
    ):
        errors.append("energy_defaults.data_port must be a valid UDP port")
    if str(defaults.physical_scope) not in {"FS", "FULL_SYSTEM"}:
        errors.append("energy_defaults.physical_scope must be full-system")
    if str(defaults.window_label) != "command":
        errors.append("energy_defaults.window_label must be command")
    if Path(str(defaults.collector_binary or "")).name != "urecs-data-collector":
        errors.append("collector binary must resolve as urecs-data-collector")
    if Path(str(defaults.power_calculations_binary or "")).name != "power_calculations":
        errors.append("postprocessor binary must resolve as power_calculations")

    rows = _setup_rows(registry)
    bindings: list[dict[str, Any]] = []
    for setup_id in setup_ids:
        setup = rows.get(str(setup_id))
        if setup is None:
            errors.append(f"hardware setup missing: {setup_id}")
            continue
        urecs_address = _strict_setup_urecs_address(
            setup,
            setup_id=str(setup_id),
            expected_data_port=raw_data_port,
            expected_channel=raw_channel,
            expected_sample_rate=raw_sample_rate,
        )
        bindings.append(
            {
                "setup_id": str(setup_id),
                "urecs_address": urecs_address,
                "data_port": int(defaults.data_port),
                "channel": 0,
                "sample_rate_hz": 2000,
                "scope": "FS",
                "measurement_point": "complete_system_input",
            }
        )
    if errors:
        raise EnergyMethodManifestError("; ".join(errors))
    return defaults, bindings


def _runtime_fingerprint(
    defaults: EnergyDefaults, bindings: Sequence[Mapping[str, Any]]
) -> str:
    payload = {
        "collector_binary": str(defaults.collector_binary),
        "postprocessor_binary": str(defaults.power_calculations_binary),
        "collector_mode": str(defaults.mode),
        "data_port": int(defaults.data_port),
        "channel": int(defaults.channel),
        "sample_rate_hz": int(defaults.sample_rate),
        "physical_scope": str(defaults.physical_scope),
        "window_label": str(defaults.window_label),
        "bindings": [dict(row) for row in bindings],
    }
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()


def _binding_projection(
    source: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return the exact claim-bearing channel-binding identity."""
    return sorted(
        [
            {
                "setup_id": row.get("setup_id"),
                "urecs_address": row.get("urecs_address"),
                "data_port": row.get("data_port"),
                "channel": row.get("channel"),
                "sample_rate_hz": row.get("sample_rate_hz"),
                "scope": row.get("scope"),
                "measurement_point": row.get("measurement_point"),
            }
            for row in source
        ],
        key=lambda row: str(row.get("setup_id") or ""),
    )


def _normalise_sha256(value: Any) -> str:
    digest = str(value or "").strip().lower()
    return digest.split(":", 1)[1] if digest.startswith("sha256:") else digest


def _configured_binding_contract(
    registry: Mapping[str, Any],
    *,
    manifest: str | Path,
    declared_sha256: str,
) -> tuple[EnergyDefaults, list[dict[str, Any]], tuple[str, ...]]:
    """Resolve every setup sharing one canonical configured manifest."""
    try:
        canonical_manifest = Path(manifest).expanduser().resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise EnergyMethodManifestError(
            f"configured energy-method manifest cannot be resolved: {manifest}"
        ) from exc
    declared = _normalise_sha256(declared_sha256)
    target_ids: list[str] = []
    for setup_id, setup in _setup_rows(registry).items():
        energy = (
            dict(setup.get("energy") or {})
            if isinstance(setup.get("energy"), Mapping)
            else {}
        )
        candidate_raw = str(energy.get("calibration_manifest") or "").strip()
        candidate_sha = _normalise_sha256(energy.get("calibration_sha256"))
        if not candidate_raw or candidate_sha != declared:
            continue
        try:
            candidate = Path(candidate_raw).expanduser().resolve(strict=True)
        except (FileNotFoundError, OSError):
            continue
        if candidate == canonical_manifest:
            target_ids.append(setup_id)
    targets = tuple(sorted(target_ids))
    if not targets:
        raise EnergyMethodManifestError(
            "no registry setup shares the configured canonical method path and SHA-256"
        )
    defaults, bindings = _runtime_contract(registry, targets)
    return defaults, bindings, targets


def _active_binary_path(value: Any, *, role: str) -> Path:
    expanded = _expand_binary(str(value or ""))
    try:
        path = Path(expanded).expanduser().resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise EnergyMethodManifestError(
            f"active {role} binary could not be resolved: {value!r}"
        ) from exc
    if not path.is_file() or not os.access(path, os.X_OK):
        raise EnergyMethodManifestError(
            f"active {role} binary is not an executable regular file: {path}"
        )
    return path


def _canonical_implementation_artifacts(
    defaults: EnergyDefaults,
) -> tuple[tuple[str, Path], ...]:
    """Resolve the exact local implementation admitted for method reuse."""
    package_root = Path(__file__).resolve().parents[1]
    sources = (
        package_root / "energy" / "collector.py",
        package_root / "energy" / "config.py",
        package_root / "energy" / "metrics.py",
        package_root / "resources" / "remote_scripts" / "energy_measurement_cli.py",
        package_root / "resources" / "remote_scripts" / "native_producer_energy_plan.py",
        package_root
        / "resources"
        / "remote_scripts"
        / "run_native_producer_energy_from_summary.py",
    )
    return (
        *tuple(zip(CANONICAL_IMPLEMENTATION_SOURCE_IDS, sources)),
        (
            "urecs_data_collector_binary",
            _active_binary_path(defaults.collector_binary, role="collector"),
        ),
        (
            "power_calculations_binary",
            _active_binary_path(
                defaults.power_calculations_binary, role="postprocessor"
            ),
        ),
    )


def _strict_configured_method_admission(
    manifest: str | Path,
    *,
    defaults: EnergyDefaults,
    expected_channel_bindings: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply the non-optional exact-implementation admission contract.

    The general campaign verifier keeps historical call sites compatible when
    no expected bindings are supplied.  A registry-configured method is an
    execution preflight, however, and must never inherit that relaxed mode.
    It is admitted only when every canonical source/binary identity occurs
    exactly once and still resolves to the exact locally executed content.
    """
    report: dict[str, Any] = {
        "schema": "onnx-splitpoint/configured-energy-method-admission",
        "schema_version": 1,
        "ok": False,
        "policy": "",
        "policy_ok": False,
        "required_artifact_ids": list(
            CANONICAL_IMPLEMENTATION_ARTIFACT_IDS
        ),
        "artifact_ids_unique": False,
        "artifact_id_set_exact": False,
        "channel_binding_set_exact": False,
        "source_integrity_binding_verification": {},
        "artifacts": {},
        "errors": [],
    }
    errors: list[str] = []
    try:
        path = Path(manifest).expanduser().resolve(strict=True)
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(
            "configured_method_manifest_unreadable:"
            f"{type(exc).__name__}"
        )
        report["errors"] = errors
        return report
    if not isinstance(loaded, Mapping):
        errors.append("configured_method_manifest_not_mapping")
        report["errors"] = errors
        return report

    required_top_level_fields = {
        "schema", "schema_version", "source_spec", "source_spec_sha256",
        "created_at", "locked", "evidence_mode", "channel_id", "scope",
        "measurement_point", "sample_rate_hz", "method", "sensor_chain",
        "calibration", "uncertainty", "validation_reference",
        "reuse_attestation", "source_release_integrity", "channel_bindings",
        "artifacts", "artifact_set_sha256", "channel_binding_set_sha256",
        "verification", "manifest_payload_sha256",
    }
    report["top_level_fields_exact"] = bool(
        set(loaded) == required_top_level_fields
    )
    if not report["top_level_fields_exact"]:
        errors.append("configured_method_top_level_fields_not_exact")

    evidence_mode_value = loaded.get("evidence_mode")
    evidence_mode = (
        evidence_mode_value if isinstance(evidence_mode_value, str) else ""
    )
    schema_version = loaded.get("schema_version")
    if evidence_mode != INHERITED_VALIDATED_ENERGY_METHOD_MODE:
        errors.append("configured_method_evidence_mode_not_inherited_validated_method")
    if type(schema_version) is not int or schema_version != 2:
        errors.append("configured_method_schema_version_not_2")

    literal_method_identity = {
        "channel_id": "urecs_fs_input_channel_0",
        "measurement_point": "complete_system_input",
        "scope": "FS",
    }
    for key, expected in literal_method_identity.items():
        value = loaded.get(key)
        if not isinstance(value, str) or value != expected:
            errors.append(f"configured_method_{key}_not_literal_{expected}")
    top_sample_rate = loaded.get("sample_rate_hz")
    if type(top_sample_rate) is not int or top_sample_rate != 2000:
        errors.append("configured_method_top_sample_rate_not_exact_2000")

    current_source_integrity = verify_installed_source_integrity()
    source_integrity_verification = verify_source_integrity_binding(
        loaded.get("source_release_integrity")
        if isinstance(loaded.get("source_release_integrity"), Mapping)
        else None,
        current_report=current_source_integrity,
    )
    report["source_integrity_binding_verification"] = (
        source_integrity_verification
    )
    if (
        source_integrity_verification.get("ok") is not True
        or source_integrity_verification.get("status") != "verified"
    ):
        errors.append("configured_method_source_integrity_binding_invalid")
        errors.extend(
            "configured_method_" + str(value)
            for value in list(source_integrity_verification.get("errors") or [])
        )

    method = (
        dict(loaded.get("method") or {})
        if isinstance(loaded.get("method"), Mapping)
        else {}
    )
    policy_value = method.get("implementation_policy")
    policy = policy_value if isinstance(policy_value, str) else ""
    report["policy"] = policy
    report["policy_ok"] = policy == EXACT_IMPLEMENTATION_REUSE_POLICY
    if not report["policy_ok"]:
        errors.append("configured_method_exact_implementation_policy_missing")
    method_sample_rate = method.get("sample_rate_hz")
    if type(method_sample_rate) is not int or method_sample_rate != 2000:
        errors.append("configured_method_method_sample_rate_not_exact_2000")
    method_data_port = method.get("data_port")
    report["method_data_port"] = method_data_port
    report["expected_data_port"] = int(defaults.data_port)
    report["method_data_port_ok"] = bool(
        not isinstance(method_data_port, bool)
        and isinstance(method_data_port, int)
        and method_data_port == int(defaults.data_port)
    )
    if not report["method_data_port_ok"]:
        errors.append("configured_method_data_port_mismatch")

    raw_bindings_value = loaded.get("channel_bindings")
    binding_list_type_ok = isinstance(raw_bindings_value, list)
    raw_bindings = (
        list(raw_bindings_value) if binding_list_type_ok else []
    )
    bindings = [
        dict(row)
        for row in raw_bindings
        if isinstance(row, Mapping)
    ]
    required_binding_fields = {
        "setup_id",
        "urecs_address",
        "data_port",
        "channel",
        "sample_rate_hz",
        "scope",
        "measurement_point",
    }
    binding_rows_exact = bool(
        binding_list_type_ok
        and len(bindings) == len(raw_bindings)
        and all(set(row) == required_binding_fields for row in bindings)
    )
    report["channel_bindings_list_type_ok"] = binding_list_type_ok
    report["channel_binding_rows_exact"] = binding_rows_exact
    if not binding_rows_exact:
        errors.append("configured_method_channel_binding_fields_not_exact")
    binding_literal_errors: list[str] = []
    for index, row in enumerate(bindings):
        setup_id_value = row.get("setup_id")
        address_value = row.get("urecs_address")
        if not (
            isinstance(setup_id_value, str)
            and setup_id_value
            and setup_id_value == setup_id_value.strip()
            and not any(
                character.isspace()
                or ord(character) < 32
                or ord(character) == 127
                for character in setup_id_value
            )
        ):
            binding_literal_errors.append(f"{index}:setup_id")
        if not (
            isinstance(address_value, str)
            and address_value
            and address_value == address_value.strip()
            and not any(
                character.isspace()
                or ord(character) < 32
                or ord(character) == 127
                for character in address_value
            )
        ):
            binding_literal_errors.append(f"{index}:urecs_address")
        for key, expected in (
            ("data_port", int(defaults.data_port)),
            ("channel", int(defaults.channel)),
            ("sample_rate_hz", int(defaults.sample_rate)),
        ):
            value = row.get(key)
            if type(value) is not int or value != expected:
                binding_literal_errors.append(f"{index}:{key}")
        if not (
            isinstance(row.get("scope"), str)
            and row.get("scope") == "FS"
        ):
            binding_literal_errors.append(f"{index}:scope")
        if not (
            isinstance(row.get("measurement_point"), str)
            and row.get("measurement_point") == "complete_system_input"
        ):
            binding_literal_errors.append(f"{index}:measurement_point")
    report["channel_binding_literal_errors"] = binding_literal_errors
    report["channel_binding_literals_exact"] = not binding_literal_errors
    if binding_literal_errors:
        errors.append("configured_method_channel_binding_literals_not_exact")
    binding_port_errors: list[str] = []
    for row in bindings:
        binding_id = str(row.get("setup_id") or "").strip() or "missing"
        binding_port = row.get("data_port")
        if (
            isinstance(binding_port, bool)
            or not isinstance(binding_port, int)
            or binding_port != int(defaults.data_port)
        ):
            binding_port_errors.append(binding_id)
    report["binding_data_port_errors"] = binding_port_errors
    if not bindings or binding_port_errors:
        errors.append("configured_method_binding_data_port_mismatch")
    actual_binding_projection = _binding_projection(bindings)
    expected_binding_projection = _binding_projection(
        expected_channel_bindings
    )
    report["channel_bindings"] = actual_binding_projection
    report["expected_channel_bindings"] = expected_binding_projection
    report["channel_binding_set_exact"] = bool(
        actual_binding_projection == expected_binding_projection
    )
    if not report["channel_binding_set_exact"]:
        errors.append("configured_method_channel_binding_set_mismatch")

    raw_artifacts_value = loaded.get("artifacts")
    artifact_list_type_ok = isinstance(raw_artifacts_value, list)
    raw_artifacts = (
        list(raw_artifacts_value) if artifact_list_type_ok else []
    )
    artifacts = [
        dict(row) for row in raw_artifacts if isinstance(row, Mapping)
    ]
    report["artifacts_list_type_ok"] = artifact_list_type_ok
    if not artifact_list_type_ok or len(artifacts) != len(raw_artifacts):
        errors.append("configured_method_artifact_row_not_mapping")
    artifact_fields_exact = bool(
        artifact_list_type_ok
        and len(artifacts) == len(raw_artifacts)
        and all(
            set(row) == {"id", "kind", "path", "size_bytes", "sha256"}
            for row in artifacts
        )
    )
    report["artifact_fields_exact"] = artifact_fields_exact
    if not artifact_fields_exact:
        errors.append("configured_method_artifact_fields_not_exact")
    ids = [
        row.get("id") if isinstance(row.get("id"), str) else ""
        for row in artifacts
    ]
    report["artifact_ids_unique"] = bool(
        ids and all(ids) and len(ids) == len(set(ids))
    )
    if not report["artifact_ids_unique"]:
        errors.append("configured_method_artifact_ids_not_unique")
    report["artifact_id_set_exact"] = bool(
        len(ids) == len(CANONICAL_IMPLEMENTATION_ARTIFACT_IDS)
        and set(ids) == set(CANONICAL_IMPLEMENTATION_ARTIFACT_IDS)
    )
    if not report["artifact_id_set_exact"]:
        errors.append("configured_method_artifact_set_not_exact")

    try:
        expected_artifacts = _canonical_implementation_artifacts(defaults)
    except Exception as exc:
        errors.append(
            "configured_method_expected_implementation_unavailable:"
            f"{type(exc).__name__}"
        )
        report["errors"] = errors
        return report

    artifact_reports: dict[str, dict[str, Any]] = {}
    for artifact_id, expected_raw_path in expected_artifacts:
        expected_path = expected_raw_path.expanduser().resolve(strict=True)
        expected_sha256 = _sha256_file(expected_path)
        expected_size = int(expected_path.stat().st_size)
        matches = [
            row
            for row in artifacts
            if isinstance(row.get("id"), str)
            and row.get("id") == artifact_id
        ]
        item: dict[str, Any] = {
            "count": len(matches),
            "expected_path": str(expected_path),
            "expected_sha256": expected_sha256,
            "expected_size_bytes": expected_size,
            "path": "",
            "sha256": "",
            "size_bytes": None,
            "kind_ok": False,
            "path_ok": False,
            "sha256_ok": False,
            "size_ok": False,
            "verified": False,
        }
        artifact_reports[artifact_id] = item
        if len(matches) != 1:
            errors.append(
                f"configured_method_artifact_{artifact_id}_"
                + ("missing" if not matches else "ambiguous")
            )
            continue
        row = matches[0]
        item["kind_ok"] = bool(
            isinstance(row.get("kind"), str)
            and row.get("kind") == "measurement_implementation"
        )
        raw_path_value = row.get("path")
        raw_path = raw_path_value if isinstance(raw_path_value, str) else ""
        item["path"] = raw_path
        try:
            actual_path = Path(raw_path).expanduser().resolve(strict=True)
        except (FileNotFoundError, OSError):
            actual_path = None
        if actual_path is not None and not actual_path.is_file():
            actual_path = None
        item["path_ok"] = bool(
            raw_path == str(expected_path) and actual_path == expected_path
        )
        digest_value = row.get("sha256")
        declared_digest_raw = (
            digest_value if isinstance(digest_value, str) else ""
        )
        declared_digest = declared_digest_raw.lower()
        if declared_digest.startswith("sha256:"):
            declared_digest = declared_digest.split(":", 1)[1]
        item["sha256"] = declared_digest_raw
        try:
            actual_sha256 = (
                _sha256_file(actual_path) if actual_path is not None else ""
            )
            actual_size = (
                int(actual_path.stat().st_size)
                if actual_path is not None
                else None
            )
        except OSError:
            actual_sha256 = ""
            actual_size = None
        item["sha256_ok"] = bool(
            len(declared_digest) == 64
            and all(character in "0123456789abcdef" for character in declared_digest)
            and declared_digest == expected_sha256
            and actual_path is not None
            and actual_sha256 == expected_sha256
        )
        declared_size = row.get("size_bytes")
        item["size_bytes"] = declared_size
        item["size_ok"] = bool(
            not isinstance(declared_size, bool)
            and isinstance(declared_size, int)
            and declared_size == expected_size
            and actual_path is not None
            and actual_size == expected_size
        )
        item["verified"] = bool(
            item["kind_ok"]
            and item["path_ok"]
            and item["sha256_ok"]
            and item["size_ok"]
        )
        if not item["verified"]:
            errors.append(
                f"configured_method_artifact_{artifact_id}_identity_mismatch"
            )
    report["artifacts"] = artifact_reports
    report["errors"] = errors
    report["ok"] = not errors
    return report


def _verify_reference(
    *,
    setup_id: str,
    setup: Mapping[str, Any],
    defaults: EnergyDefaults,
    manifest: str | Path,
    declared_sha256: str,
    expected_channel_bindings: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    energy = (
        dict(setup.get("energy") or {})
        if isinstance(setup.get("energy"), Mapping)
        else {}
    )
    verification = _verify_calibration_manifest(
        manifest,
        declared_sha256,
        setup_id=setup_id,
        urecs_address=str(
            energy.get("urecs_address") or energy.get("address") or ""
        ).strip(),
        data_port=int(defaults.data_port),
        channel=int(defaults.channel),
        sample_rate_hz=int(defaults.sample_rate),
        physical_scope="FS",
        collector_mode=str(defaults.mode),
        collector_binary=str(defaults.collector_binary),
        postprocessor_binary=str(defaults.power_calculations_binary),
        expected_channel_bindings=expected_channel_bindings,
    )
    admission = _strict_configured_method_admission(
        manifest,
        defaults=defaults,
        expected_channel_bindings=expected_channel_bindings,
    )
    verification["configured_method_admission"] = admission
    verification["configured_method_admission_errors"] = list(
        admission.get("errors") or []
    )
    if verification.get("verified") is True and admission.get("ok") is not True:
        verification["verified"] = False
        verification["status"] = "configured_energy_method_admission_failed"
    return verification


def verify_configured_energy_method(
    setup_id: str,
    *,
    registry_path: str | Path | None = None,
    registry: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify one registry-bound FS energy method without touching hardware."""
    sid = setup_id if isinstance(setup_id, str) else ""
    raw_registry_error: InvalidEnergyMethodRegistryContractError | None = None
    if registry is not None:
        raw_current = dict(registry)
        current = dict(registry)
    else:
        path = energy_config._expand(  # type: ignore[attr-defined]
            registry_path or energy_config.default_registry_path()
        )
        try:
            raw_current = _read_raw_registry(path)
        except InvalidEnergyMethodRegistryContractError as exc:
            raw_registry_error = exc
            raw_current = {}
        current = load_hardware_registry(path)
    base: dict[str, Any] = {
        "schema": "onnx-splitpoint/configured-energy-method-verification",
        "schema_version": 1,
        "setup_id": sid,
        "configured": False,
        "verified": False,
        "status": "setup_not_found",
        "path": "",
        "declared_sha256": "",
        "sha256": "",
        "runtime_binding_errors": [],
        "configured_method_admission_errors": [],
        "selection_errors": [],
        "duplicate_setup_ids": [],
        "configuration_errors": [],
        "verification": {},
    }
    if not _canonical_setup_id(setup_id):
        base["status"] = "invalid_setup_id"
        base["selection_errors"] = [
            "setup id must be a nonempty canonical string without "
            "whitespace/control characters"
        ]
        return base
    try:
        setup = _setup_rows(current).get(sid)
    except DuplicateHardwareSetupIdError as exc:
        base["status"] = "duplicate_hardware_setup_id"
        base["selection_errors"] = [str(exc)]
        base["duplicate_setup_ids"] = list(exc.setup_ids)
        return base
    except InvalidEnergyMethodRegistryContractError as exc:
        base["status"] = "invalid_energy_method_registry_contract"
        base["configuration_errors"] = list(exc.errors)
        return base
    if setup is None:
        return base
    base["status"] = "missing"
    energy = (
        dict(setup.get("energy") or {})
        if isinstance(setup.get("energy"), Mapping)
        else {}
    )
    raw_path = str(energy.get("calibration_manifest") or "").strip()
    declared = str(energy.get("calibration_sha256") or "").strip().lower()
    base["path"] = raw_path
    base["declared_sha256"] = declared
    base["configured"] = bool(raw_path and declared)
    try:
        if raw_registry_error is not None:
            raise raw_registry_error
        raw_port, raw_channel, raw_sample_rate = _strict_raw_energy_defaults(
            raw_current
        )
        _strict_raw_target_setups(
            raw_current,
            [sid],
            expected_data_port=raw_port,
            expected_channel=raw_channel,
            expected_sample_rate=raw_sample_rate,
        )
        current_port, current_channel, current_sample_rate = (
            _strict_raw_energy_defaults(current)
        )
        _strict_setup_urecs_address(
            setup,
            setup_id=sid,
            expected_data_port=current_port,
            expected_channel=current_channel,
            expected_sample_rate=current_sample_rate,
        )
    except InvalidEnergyDataPortError as exc:
        base["status"] = "invalid_energy_defaults_data_port"
        base["configuration_errors"] = [str(exc)]
        return base
    except InvalidEnergyMethodRegistryContractError as exc:
        base["status"] = "invalid_energy_method_registry_contract"
        base["configuration_errors"] = list(exc.errors)
        return base
    if not raw_path and not declared:
        return base
    if not raw_path or not declared:
        base["status"] = "incomplete_configuration"
        return base
    base["configured"] = True
    try:
        defaults, expected_bindings, binding_target_ids = (
            _configured_binding_contract(
                current,
                manifest=raw_path,
                declared_sha256=declared,
            )
        )
        _strict_raw_target_setups(
            raw_current,
            binding_target_ids,
            expected_data_port=raw_port,
            expected_channel=raw_channel,
            expected_sample_rate=raw_sample_rate,
        )
    except EnergyMethodManifestError as exc:
        base["status"] = "configured_energy_method_binding_contract_invalid"
        base["configuration_errors"] = [str(exc)]
        return base
    verification = _verify_reference(
        setup_id=sid,
        setup=setup,
        defaults=defaults,
        manifest=raw_path,
        declared_sha256=declared,
        expected_channel_bindings=expected_bindings,
    )
    base["verification"] = verification
    base["verified"] = verification.get("verified") is True
    base["status"] = str(verification.get("status") or "verification_failed")
    base["path"] = str(verification.get("path") or raw_path)
    base["sha256"] = str(verification.get("actual_sha256") or "").lower()
    base["runtime_binding_errors"] = list(
        verification.get("runtime_binding_errors") or []
    )
    base["configured_method_admission_errors"] = list(
        verification.get("configured_method_admission_errors") or []
    )
    admission = (
        dict(verification.get("configured_method_admission") or {})
        if isinstance(verification.get("configured_method_admission"), Mapping)
        else {}
    )
    base["configured_method_admission"] = admission
    base["source_integrity_verification"] = copy.deepcopy(
        admission.get("source_integrity_binding_verification") or {}
    )
    return base


def _default_output_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    return energy_config.default_energy_home() / "full_system_methods" / stamp


def _write_json_new(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=False, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def prepare_configured_energy_method(
    *,
    attested_by: str,
    accepted_validated_method_reuse: bool,
    registry_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    setup_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Create, verify and atomically configure one shared FS method manifest."""
    if accepted_validated_method_reuse is not True:
        raise EnergyMethodManifestError(
            "explicit --accept-validated-method-reuse attestation is required"
        )
    if (
        not isinstance(attested_by, str)
        or not attested_by
        or attested_by != attested_by.strip()
        or any(
            ord(character) < 32 or ord(character) == 127
            for character in attested_by
        )
    ):
        raise EnergyMethodManifestError(
            "--attested-by must be a nonempty canonical string without "
            "surrounding/control whitespace"
        )
    signer = attested_by

    # Release-source identity is a prerequisite for issuing a claim-bearing
    # method document.  This must run before creating the output directory or
    # touching the hardware registry.
    source_integrity_report = verify_installed_source_integrity()
    if (
        source_integrity_report.get("ok") is not True
        or source_integrity_report.get("status") != "verified"
    ):
        raise EnergyMethodManifestError(
            "installed release source integrity preflight failed: "
            + json.dumps(source_integrity_report, sort_keys=True)
        )
    try:
        source_integrity_binding = create_source_integrity_binding(
            source_integrity_report
        )
    except Exception as exc:
        raise EnergyMethodManifestError(
            "installed release source integrity binding failed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    path = energy_config._expand(  # type: ignore[attr-defined]
        registry_path or energy_config.default_registry_path()
    )
    raw_registry = _read_raw_registry(path)
    prepared_registry, _defaults_migration = _prepare_energy_defaults_candidate(
        raw_registry
    )
    raw_port, raw_channel, raw_sample_rate = _strict_raw_energy_defaults(
        prepared_registry
    )
    # Do not call load_hardware_registry here: its narrow legacy-token upgrade
    # may persist on read. Method preparation promises a single final registry
    # commit only after every source, binary and manifest check has passed, so
    # all normalization stays in memory until that point.
    registry = _normalise_registry_in_memory(prepared_registry)
    targets = _target_setup_ids(registry, setup_ids)
    _strict_raw_target_setups(
        raw_registry,
        targets,
        expected_data_port=raw_port,
        expected_channel=raw_channel,
        expected_sample_rate=raw_sample_rate,
    )
    defaults, bindings = _runtime_contract(registry, targets)
    fingerprint = _runtime_fingerprint(defaults, bindings)

    destination = Path(output_dir).expanduser().resolve() if output_dir else _default_output_dir().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    spec_path = destination / "urecs_fs_input_validated_method_spec.json"
    manifest_path = destination / "urecs_fs_input_validated_method_manifest.json"
    if spec_path.exists() or manifest_path.exists():
        raise EnergyMethodManifestError(
            f"refusing to overwrite an existing energy-method artefact in {destination}"
        )

    implementation_artifacts = _canonical_implementation_artifacts(defaults)
    spec = {
        "schema": "onnx-splitpoint/energy-method-source-spec",
        "schema_version": 2,
        "portable_paths": True,
        "evidence_mode": INHERITED_VALIDATED_ENERGY_METHOD_MODE,
        "channel_id": "urecs_fs_input_channel_0",
        "scope": "FS",
        "measurement_point": "complete_system_input",
        "sample_rate_hz": 2000,
        "locked": True,
        "method": {
            "collector": "urecs-data-collector",
            "collector_mode": "fast_firmware",
            "data_port": int(defaults.data_port),
            "postprocessor": "power_calculations",
            "sample_rate_hz": 2000,
            "output_semantics": "calibrated_input_energy_unsubtracted",
            "implementation_policy": EXACT_IMPLEMENTATION_REUSE_POLICY,
        },
        "validation_reference": dict(WACHSMUTH_ENERGY_METHOD_REFERENCE),
        "reuse_attestation": {
            "validated_method_accepted": True,
            "exact_implementation_reused": True,
            "new_calibration_required": False,
            "attested_by": signer,
            "attested_at": now_iso(),
            "statement": (
                "The configured u.RECS channel 0 measures complete-system input "
                "power for every bound setup. The independently validated method "
                "is accepted and the exact hashed implementation is reused; this "
                "operation does not claim or manufacture a new sensor calibration."
            ),
        },
        "source_release_integrity": source_integrity_binding,
        "channel_bindings": bindings,
        "artifacts": [
            *[
                {
                    "id": artifact_id,
                    "kind": "measurement_implementation",
                    "path": str(source),
                }
                for artifact_id, source in implementation_artifacts
            ],
        ],
    }
    _write_json_new(spec_path, spec)
    try:
        create_energy_calibration_manifest(spec=spec_path, output=manifest_path)
    except Exception as exc:
        raise EnergyMethodManifestError(
            f"energy-method manifest creation failed: {type(exc).__name__}: {exc}"
        ) from exc
    manifest_path.chmod(0o600)
    digest = _sha256_file(manifest_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    semantic = verify_energy_calibration_manifest(
        payload,
        require_final=True,
        expected_channel_bindings=bindings,
        require_source_integrity_binding=True,
    )
    if not semantic.get("ok"):
        raise EnergyMethodManifestError(
            "generated energy-method manifest failed semantic verification: "
            + json.dumps(semantic, sort_keys=True)
        )

    rows = _setup_rows(registry)
    binding_verifications: dict[str, dict[str, Any]] = {}
    for setup_id in targets:
        verification = _verify_reference(
            setup_id=setup_id,
            setup=rows[setup_id],
            defaults=defaults,
            manifest=manifest_path,
            declared_sha256=digest,
            expected_channel_bindings=bindings,
        )
        binding_verifications[setup_id] = verification
        if verification.get("verified") is not True:
            raise EnergyMethodManifestError(
                f"runtime binding verification failed for {setup_id}: "
                + json.dumps(verification, sort_keys=True)
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    with energy_config._hardware_registry_write_lock(path):  # type: ignore[attr-defined]
        fresh_raw = _read_raw_registry(path)
        fresh_candidate, committed_defaults_migration = (
            _prepare_energy_defaults_candidate(fresh_raw)
        )
        _strict_raw_energy_defaults(fresh_candidate)
        fresh = _normalise_registry_in_memory(fresh_candidate)
        fresh_defaults, fresh_bindings = _runtime_contract(fresh, targets)
        if _runtime_fingerprint(fresh_defaults, fresh_bindings) != fingerprint:
            raise EnergyMethodManifestError(
                "hardware energy configuration changed while the manifest was prepared; "
                "registry was not modified"
            )
        found: set[str] = set()
        for raw in list(fresh.get("hardware_setups") or []):
            if not isinstance(raw, MutableMapping):
                continue
            raw_setup_id = raw.get("id")
            if not _canonical_setup_id(raw_setup_id):
                raise EnergyMethodManifestError(
                    "hardware setup id became noncanonical during commit"
                )
            setup_id = raw_setup_id
            if setup_id not in targets:
                continue
            energy = (
                dict(raw.get("energy") or {})
                if isinstance(raw.get("energy"), Mapping)
                else {}
            )
            energy["calibration_manifest"] = str(manifest_path.resolve())
            energy["calibration_sha256"] = digest
            raw["energy"] = energy
            found.add(setup_id)
        if found != set(targets):
            raise EnergyMethodManifestError(
                "target setup set changed while the manifest was prepared; registry was not modified"
            )
        energy_config._write_hardware_registry_atomic(fresh, path)  # type: ignore[attr-defined]

    configured = {
        setup_id: verify_configured_energy_method(
            setup_id, registry_path=path
        )
        for setup_id in targets
    }
    if not all(item.get("verified") for item in configured.values()):
        raise EnergyMethodManifestError(
            "post-commit configured method verification failed: "
            + json.dumps(configured, sort_keys=True)
        )
    return {
        "schema": "onnx-splitpoint/energy-method-preparation",
        "schema_version": 1,
        "ok": True,
        "prepared_at": now_iso(),
        "attested_by": signer,
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": digest,
        "source_spec": str(spec_path.resolve()),
        "source_release_integrity": copy.deepcopy(source_integrity_binding),
        "energy_defaults_migration": copy.deepcopy(
            committed_defaults_migration
        ),
        "registry": str(path.resolve()),
        "setup_ids": list(targets),
        "semantic_verification": semantic,
        "configured_verification": configured,
        "hardware_action_performed": False,
    }


__all__ = [
    "CANONICAL_IMPLEMENTATION_ARTIFACT_IDS",
    "EnergyMethodManifestError",
    "DuplicateHardwareSetupIdError",
    "InvalidEnergyDataPortError",
    "InvalidEnergyMethodRegistryContractError",
    "STANDARD_PLATFORM_SETUP_IDS",
    "prepare_configured_energy_method",
    "verify_configured_energy_method",
]
