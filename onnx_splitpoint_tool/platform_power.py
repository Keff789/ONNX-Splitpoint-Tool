"""u.RECS/Jetson platform power control and M.2 idle-power calibration.

The u.RECS firmware exposes two *toggle* commands via UDP: ``jetson`` and
``m.2``.  A toggle-only interface cannot report the physical switch state, so
this module deliberately separates:

* u.RECS reachability (one bounded ICMP probe),
* Jetson readiness (authenticated SSH), and
* accelerator presence (read-only probe on the Jetson).

Power-changing operations are fail-closed and serialized.  They never run in
the background on a timer and they never infer an M.2 state merely from a sent
UDP datagram.  The calibration workflow saves ``accelerator_idle_w`` only after
both measured states were verified successfully.
"""

from __future__ import annotations

import copy
import json
import math
import os
import platform
import re
import shlex
import socket
import subprocess
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, MutableMapping, Optional

from .energy import config as energy_config
from .energy.config import (
    EnergyDefaults,
    energy_defaults_from_registry,
    energy_setup_from_registry,
    energy_measurements_root,
    load_hardware_registry,
)
from .energy.full_system_gain import (
    FULL_SYSTEM_CURRENT_SCALE_LOAD_CONNECTION,
    FULL_SYSTEM_CURRENT_SCALE_MODEL,
    FULL_SYSTEM_CURRENT_SCALE_SCHEMA,
    FULL_SYSTEM_CURRENT_SCALE_SCHEMA_VERSION,
    FULL_SYSTEM_CURRENT_SCALE_TARGET_CURRENTS_A,
    sha256_file,
)
from .remote.ssh_transport import HostConfig, SSHTransport
from .workflow.run_control import platform_workflow_interlock_path


class PlatformPowerError(RuntimeError):
    """Base class for fail-closed platform-power operations."""


class PlatformConfigurationError(PlatformPowerError):
    """The selected setup does not contain the required host/u.RECS data."""


class PlatformStateError(PlatformPowerError):
    """The observed platform state is unsafe or ambiguous for a toggle."""


class PlatformBusyError(PlatformPowerError):
    """A benchmark/workflow or another platform-power operation is active."""


class PlatformCalibrationCancelled(PlatformPowerError):
    """The operator cancelled a guided calibration before commit."""


FULL_SYSTEM_CALIBRATION_TARGET_CURRENTS_A = (
    FULL_SYSTEM_CURRENT_SCALE_TARGET_CURRENTS_A
)
FULL_SYSTEM_CALIBRATION_DEFAULT_VOLTAGE_V = 19.0
FULL_SYSTEM_CALIBRATION_HARD_FACTOR_MIN = 0.5
FULL_SYSTEM_CALIBRATION_HARD_FACTOR_MAX = 1.5


DEFAULT_POWER_CONTROL: dict[str, Any] = {
    "enabled": True,
    "udp_port": 3000,
    # Interactive ``nc -u HOST 3000`` sends the command when Enter is pressed.
    # Keep this explicit and configurable because some firmware builds consume
    # the token without a line ending.
    "udp_terminator": "lf",  # lf | crlf | none
    "jetson_command": "jetson",
    "m2_command": "m.2",
    "status_ping_timeout_s": 1.5,
    "ssh_probe_timeout_s": 5.0,
    "m2_probe_timeout_s": 12.0,
    "shutdown_preflight_command": "sudo -n true",
    "shutdown_command": "sudo -n systemctl poweroff",
    "shutdown_timeout_s": 120.0,
    "boot_timeout_s": 240.0,
    "ssh_poll_interval_s": 3.0,
    "shutdown_settle_s": 5.0,
    "power_toggle_settle_s": 3.0,
    "m2_toggle_settle_s": 3.0,
    # Accelerator drivers can enumerate after SSH becomes ready.  Never treat
    # the first post-boot absence as proof that the M.2 rail is off.
    "m2_post_boot_settle_s": 5.0,
    "m2_verify_observations": 2,
    "m2_verify_interval_s": 1.0,
    "calibration_stabilize_s": 30.0,
    "calibration_measure_s": 30.0,
    "full_system_calibration_load_settle_s": 5.0,
    "full_system_calibration_minimum_delta_w": 2.0,
    "full_system_calibration_max_point_spread_pct": 2.0,
    "full_system_calibration_max_idle_drift_w": 0.5,
    "full_system_calibration_min_factor": 0.90,
    "full_system_calibration_max_factor": 1.10,
    "full_system_calibration_reference_current_tolerance_pct": 10.0,
    "require_ping_before_toggle": True,
    "require_positive_calibration_delta": True,
    "minimum_calibration_delta_w": 0.02,
}


@dataclass(frozen=True)
class PlatformStatus:
    setup_id: str
    checked_at: str
    urecs_address: str
    urecs_host: str
    urecs_port: int
    urecs_configured: bool
    urecs_reachable: Optional[bool]
    urecs_detail: str
    jetson_host: str
    jetson_configured: bool
    jetson_ssh_ready: Optional[bool]
    jetson_detail: str
    accelerator: str
    m2_present: Optional[bool]
    m2_detail: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CalibrationResult:
    setup_id: str
    accelerator: str
    idle_power_without_m2_w: float
    idle_power_with_m2_w: float
    accelerator_idle_power_w: float
    started_at: str
    finished_at: str
    output_dir: str
    saved: bool
    restored_m2_on: bool
    evidence: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FullSystemInputCalibrationResult:
    setup_id: str
    accelerator: str
    scale_factor: float
    idle_before_w: float
    idle_after_w: float
    point_results: tuple[dict[str, Any], ...]
    point_spread_pct: float
    max_fit_residual_pct: float
    warnings: tuple[str, ...]
    started_at: str
    finished_at: str
    output_dir: str
    evidence_path: str
    evidence_sha256: str
    saved: bool
    quality_passed: bool
    restored_jetson_ready: bool
    restored_m2_on: bool
    invalidated_idle_baselines: bool
    evidence: dict[str, Any]
    initial_state_restored: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


ProgressCallback = Callable[[str], None]
OperatorPromptCallback = Callable[[Mapping[str, Any]], Mapping[str, Any] | None]


def _now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _emit(callback: ProgressCallback | None, text: str) -> None:
    if callback is None:
        return
    try:
        callback(str(text))
    except Exception:
        pass


def _number(raw: Any, default: float, *, minimum: float | None = None) -> float:
    try:
        value = float(raw)
    except Exception:
        value = float(default)
    if not math.isfinite(value):
        value = float(default)
    if minimum is not None:
        value = max(float(minimum), value)
    return value


def _integer(raw: Any, default: int, *, minimum: int | None = None) -> int:
    try:
        value = int(raw)
    except Exception:
        value = int(default)
    if minimum is not None:
        value = max(int(minimum), value)
    return value


def _bool(raw: Any, default: bool = False) -> bool:
    if isinstance(raw, bool):
        return raw
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on", "enabled"}


_POWER_CONTROL_NUMERIC_FIELDS: dict[
    str, tuple[int | float, int | float, int | float | None, bool]
] = {
    # key: (release default, inclusive minimum, inclusive maximum, integer)
    "udp_port": (3000, 1, 65535, True),
    "status_ping_timeout_s": (1.5, 0.2, None, False),
    "ssh_probe_timeout_s": (5.0, 1.0, None, False),
    "m2_probe_timeout_s": (12.0, 1.0, None, False),
    "shutdown_timeout_s": (120.0, 10.0, None, False),
    "boot_timeout_s": (240.0, 10.0, None, False),
    "ssh_poll_interval_s": (3.0, 0.5, None, False),
    "shutdown_settle_s": (5.0, 0.0, None, False),
    "power_toggle_settle_s": (3.0, 0.0, None, False),
    "m2_toggle_settle_s": (3.0, 0.0, None, False),
    "m2_post_boot_settle_s": (5.0, 0.0, None, False),
    "m2_verify_observations": (2, 2, None, True),
    "m2_verify_interval_s": (1.0, 0.0, None, False),
    "calibration_stabilize_s": (30.0, 0.0, None, False),
    "calibration_measure_s": (30.0, 5.0, None, False),
    "full_system_calibration_load_settle_s": (5.0, 0.0, 60.0, False),
    "full_system_calibration_minimum_delta_w": (2.0, 0.0, None, False),
    "full_system_calibration_max_point_spread_pct": (2.0, 0.0, 50.0, False),
    "full_system_calibration_max_idle_drift_w": (0.5, 0.0, None, False),
    "full_system_calibration_min_factor": (0.90, 0.5, 1.5, False),
    "full_system_calibration_max_factor": (1.10, 0.5, 1.5, False),
    "full_system_calibration_reference_current_tolerance_pct": (10.0, 0.0, 50.0, False),
    "minimum_calibration_delta_w": (0.02, 0.0, None, False),
}


def _strict_power_control_number(
    value: Any,
    *,
    field: str,
    minimum: int | float,
    maximum: int | float | None,
    integer: bool,
) -> int | float:
    """Validate one explicit timing/count/port value without coercion.

    Power-control numbers ultimately drive timeouts, sleeps, observation
    counts, and physical rail operations.  Accepting strings, booleans, NaN,
    infinity, or clamping an out-of-range value would silently change the
    requested safety sequence.
    """

    if integer:
        if type(value) is not int:
            expected = "an integer"
            raise PlatformConfigurationError(
                f"{field} must be {expected} in the configured range"
            )
        parsed: int | float = value
    else:
        if type(value) not in {int, float}:
            raise PlatformConfigurationError(
                f"{field} must be a finite number in the configured range"
            )
        parsed = float(value)
        if not math.isfinite(parsed):
            raise PlatformConfigurationError(
                f"{field} must be a finite number in the configured range"
            )

    if parsed < minimum or (maximum is not None and parsed > maximum):
        upper = "" if maximum is None else f"..{maximum:g}"
        lower = f">={minimum:g}" if maximum is None else f"{minimum:g}"
        expected_range = f"{lower}{upper}"
        raise PlatformConfigurationError(
            f"{field} must be in range {expected_range}"
        )
    return int(parsed) if integer else float(parsed)


def _require_exact_bool_argument(value: Any, *, field: str) -> bool:
    """Reject truthy lookalikes for public safety/override arguments."""

    if type(value) is not bool:
        raise PlatformConfigurationError(f"{field} must be true or false")
    return value


def _strict_registry_path_argument(
    value: str | Path | None,
) -> str | Path | None:
    """Validate an optional public registry selector without default aliases.

    Only ``None`` requests the default registry.  In particular, an empty
    string must not flow through ``path or default_registry_path()`` and select
    a real deployment registry unexpectedly.
    """

    if value is None:
        return None
    if not isinstance(value, (str, Path)):
        raise PlatformConfigurationError(
            "registry_path must be a non-empty string or Path, or omitted"
        )
    if isinstance(value, str) and not value:
        raise PlatformConfigurationError(
            "registry_path must be a non-empty string or Path, or omitted"
        )
    return value


def merged_power_control(raw: Mapping[str, Any] | None) -> dict[str, Any]:
    if raw is not None and not isinstance(raw, Mapping):
        raise PlatformConfigurationError("power_control must be a mapping")
    configured = dict(raw or {})
    out = dict(DEFAULT_POWER_CONTROL)
    for key, value in configured.items():
        out[str(key)] = value
    for key, (default, minimum, maximum, integer) in (
        _POWER_CONTROL_NUMERIC_FIELDS.items()
    ):
        candidate = configured[key] if key in configured else default
        out[key] = _strict_power_control_number(
            candidate,
            field=f"power_control.{key}",
            minimum=minimum,
            maximum=maximum,
            integer=integer,
        )
    out["require_ping_before_toggle"] = _bool(out.get("require_ping_before_toggle"), True)
    out["require_positive_calibration_delta"] = _bool(out.get("require_positive_calibration_delta"), True)
    out["enabled"] = _bool(out.get("enabled"), True)
    if float(out["full_system_calibration_min_factor"]) > float(
        out["full_system_calibration_max_factor"]
    ):
        raise PlatformConfigurationError(
            "power_control.full_system_calibration_min_factor must be <= "
            "power_control.full_system_calibration_max_factor"
        )
    return out


def _setup_by_id(registry: Mapping[str, Any], setup_id: str) -> dict[str, Any]:
    def canonical_id(value: Any, *, field: str) -> str:
        if (
            not isinstance(value, str)
            or not value
            or value != value.strip()
            or any(
                character.isspace()
                or ord(character) < 32
                or ord(character) == 127
                for character in value
            )
        ):
            raise PlatformConfigurationError(
                f"{field} must be a canonical non-empty string without "
                "whitespace or control characters"
            )
        return value

    sid = canonical_id(setup_id, field="requested hardware setup id")
    raw_rows = registry.get("hardware_setups")
    if not isinstance(raw_rows, list):
        raise PlatformConfigurationError(
            "hardware_setups must be a list for platform power"
        )
    matches: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_rows):
        if not isinstance(raw, Mapping):
            raise PlatformConfigurationError(
                f"hardware_setups[{index}] must be a mapping"
            )
        row_id = canonical_id(
            raw.get("id"), field=f"hardware_setups[{index}].id"
        )
        if row_id == sid:
            matches.append(dict(raw))
    if not matches:
        raise PlatformConfigurationError(f"Unknown hardware setup: {sid or '<empty>'}")
    if len(matches) != 1:
        raise PlatformConfigurationError(
            f"Duplicate hardware setup id is unsafe for platform power: {sid or '<empty>'}"
        )
    return matches[0]


def resolve_setup(
    setup_id: str,
    *,
    registry_path: str | Path | None = None,
    registry: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    reg = (
        dict(registry)
        if registry is not None
        else dict(load_hardware_registry(registry_path))
    )
    setup = _setup_by_id(reg, setup_id)
    energy = dict(setup.get("energy") or {}) if isinstance(setup.get("energy"), Mapping) else {}
    power = merged_power_control(setup.get("power_control") if isinstance(setup.get("power_control"), Mapping) else {})
    return reg, setup, {"energy": energy, "power_control": power}


def _platform_control_configuration_projection(
    registry: Mapping[str, Any],
    setup_id: str,
) -> dict[str, Any]:
    """Freeze every registry value that can select or drive a rail toggle."""

    setup = _setup_by_id(registry, setup_id)
    energy = (
        dict(setup.get("energy") or {})
        if isinstance(setup.get("energy"), Mapping)
        else {}
    )
    raw_power = (
        dict(setup.get("power_control") or {})
        if isinstance(setup.get("power_control"), Mapping)
        else {}
    )
    return {
        "schema": "onnx-splitpoint/platform-control-config-projection",
        "schema_version": 1,
        "setup_id": str(setup_id),
        "accelerator": str(setup.get("accelerator") or "").strip(),
        # Preserve raw values as well as effective values.  Otherwise two
        # different invalid ports could both be collapsed by tolerant status
        # merging and evade the operation-lock race check.
        "host": copy.deepcopy(setup.get("host")),
        "remote": copy.deepcopy(setup.get("remote")),
        "urecs_address": copy.deepcopy(energy.get("urecs_address")),
        "power_control_raw": copy.deepcopy(raw_power),
        "power_control_effective": merged_power_control(raw_power),
    }


def _reload_locked_platform_configuration(
    setup_id: str,
    *,
    registry_path: str | Path | None,
    expected_projection: Mapping[str, Any],
    expected_udp_preflight: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Reload and bind a platform target after acquiring its operation lock."""

    registry = load_hardware_registry(registry_path)
    current_projection = _platform_control_configuration_projection(
        registry, setup_id
    )
    if current_projection != dict(expected_projection):
        raise PlatformStateError(
            "ABORT_NO_MUTATION: platform configuration changed while waiting "
            "for the operation lock"
        )
    reg, setup, cfg = resolve_setup(setup_id, registry=registry)
    udp_preflight = _admit_pinned_udp_preflight(
        setup,
        cfg,
        expected_udp_preflight,
    )
    _require_power_control_enabled(cfg["power_control"])
    return reg, setup, cfg, udp_preflight


def _calibration_configuration_projection(
    registry: Mapping[str, Any],
    setup_id: str,
) -> dict[str, Any]:
    """Project every configuration field that authorizes one calibration.

    The projection is captured before the first hardware mutation and compared
    under the registry write lock at commit time.  Unrelated registry edits can
    still be merged, but a host, u.RECS endpoint, measurement-default or
    power-control change invalidates the in-flight result instead of silently
    rebinding it.
    """

    setup = _setup_by_id(registry, setup_id)
    energy = (
        dict(setup.get("energy") or {})
        if isinstance(setup.get("energy"), Mapping)
        else {}
    )
    host_identity = _jetson_host_identity(setup)
    defaults = energy_defaults_from_registry(registry)
    power = merged_power_control(
        setup.get("power_control")
        if isinstance(setup.get("power_control"), Mapping)
        else {}
    )
    return {
        "schema": "onnx-splitpoint/m2-idle-calibration-config-projection",
        "schema_version": 1,
        "setup_id": str(setup_id),
        "accelerator": str(setup.get("accelerator") or "").strip(),
        "host": copy.deepcopy(host_identity),
        "energy": {
            "enabled": _bool(energy.get("enabled"), False),
            "urecs_address": str(energy.get("urecs_address") or "").strip(),
        },
        # Use the complete effective power-control mapping.  Every value here
        # can affect a shutdown, toggle, recovery, state-verification or delta
        # admission decision during the physical sequence.
        "power_control": copy.deepcopy(power),
        # Ports, pre/post windows, environment, retention and window flags all
        # affect the physical capture or its postprocessing.
        "energy_defaults": copy.deepcopy(defaults.to_dict()),
    }


def _validate_platform_ssh_configuration(
    setup: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the exact SSH target before interpreting reachability as power."""

    raw_host = setup.get("host")
    if raw_host is None:
        host: dict[str, Any] = {}
    elif isinstance(raw_host, str):
        # Historical registries supported a scalar host address.
        if not raw_host.strip():
            raise PlatformConfigurationError("Jetson host address is blank")
        host = {"address": raw_host}
    elif isinstance(raw_host, Mapping):
        host = dict(raw_host)
    else:
        raise PlatformConfigurationError(
            "setup.host must be a mapping or legacy scalar address string"
        )

    raw_remote = setup.get("remote")
    if raw_remote is None:
        remote: dict[str, Any] = {}
    elif isinstance(raw_remote, Mapping):
        remote = dict(raw_remote)
    else:
        raise PlatformConfigurationError("setup.remote must be a mapping")

    identity = energy_config.resolve_jetson_identity(setup)
    if identity.get("valid") is not True:
        errors = ",".join(str(value) for value in identity.get("errors") or [])
        raise PlatformConfigurationError(
            "Invalid Jetson SSH configuration"
            + (f": {errors}" if errors else "")
        )
    return {
        "host": host,
        "remote": remote,
        "address": identity["address"],
        "user": identity["user"],
        "port": identity["port"],
        "ssh_extra_args": identity["ssh_extra_args"],
    }


def host_config_from_setup(setup: Mapping[str, Any]) -> HostConfig:
    validated = _validate_platform_ssh_configuration(setup)
    host = dict(validated["host"])
    remote = dict(validated["remote"])
    return HostConfig(
        id=str(setup.get("id") or "platform"),
        label=str(setup.get("label") or setup.get("id") or "Jetson"),
        host=str(validated["address"]),
        user=str(validated["user"]),
        port=int(validated["port"]),
        remote_base_dir=str(host.get("base_dir") or remote.get("remote_base_dir") or "~/splitpoint_runs"),
        ssh_extra_args=str(validated["ssh_extra_args"]),
    )


def _jetson_host_identity(setup: Mapping[str, Any]) -> dict[str, Any]:
    """Return the exact resolved non-secret SSH endpoint identity."""

    host = host_config_from_setup(setup)
    port = int(host.port)
    if not str(host.host) or not str(host.user):
        raise PlatformConfigurationError(
            "Jetson host identity requires address and user"
        )
    if port < 1 or port > 65535:
        raise PlatformConfigurationError(
            f"Jetson host port is outside 1..65535: {port}"
        )
    return {
        "address": str(host.host),
        "user": str(host.user),
        "port": port,
        "ssh_extra_args": energy_config.normalise_ssh_extra_args(
            host.ssh_extra_args
        ),
    }


def parse_urecs_address(address: str, default_port: int = 3000) -> tuple[str, int]:
    value = str(address or "").strip()
    if not value:
        return "", int(default_port)
    # Bracketed IPv6: [fd00::1]:3000
    match = re.fullmatch(r"\[([^]]+)](?::([0-9]+))?", value)
    if match:
        return match.group(1), _integer(match.group(2), default_port, minimum=1)
    # Unbracketed IPv6 has more than one colon and therefore no unambiguous port.
    if value.count(":") > 1:
        return value, int(default_port)
    if ":" in value:
        host, maybe_port = value.rsplit(":", 1)
        if maybe_port.isdigit():
            return host.strip(), _integer(maybe_port, default_port, minimum=1)
    return value, int(default_port)


def _strict_udp_port(value: Any, *, field: str) -> int:
    """Return one explicitly valid UDP port without coercive fallback."""

    if isinstance(value, bool):
        raise PlatformConfigurationError(f"{field} must be an integer from 1 to 65535")
    if isinstance(value, int):
        port = value
    elif isinstance(value, str) and re.fullmatch(r"[0-9]+", value):
        port = int(value)
    else:
        raise PlatformConfigurationError(f"{field} must be an integer from 1 to 65535")
    if not 1 <= port <= 65535:
        raise PlatformConfigurationError(f"{field} must be an integer from 1 to 65535")
    return port


def _parse_urecs_endpoint_strict(address: Any, default_port: Any) -> tuple[str, int]:
    """Parse a configured u.RECS endpoint without DNS or network activity."""

    port = _strict_udp_port(default_port, field="power_control.udp_port")
    if not isinstance(address, str) or not address or address != address.strip():
        raise PlatformConfigurationError(
            "energy.urecs_address must be a non-empty address without surrounding whitespace"
        )
    if any(ord(ch) < 0x21 or ord(ch) == 0x7F for ch in address):
        raise PlatformConfigurationError(
            "energy.urecs_address must not contain whitespace or control characters"
        )

    bracketed = re.fullmatch(r"\[([^\[\]]+)](?::([^:]+))?", address)
    if bracketed:
        host = bracketed.group(1)
        embedded_port = bracketed.group(2)
        if embedded_port is not None:
            port = _strict_udp_port(
                embedded_port, field="energy.urecs_address embedded UDP port"
            )
    elif address.startswith("[") or address.endswith("]"):
        raise PlatformConfigurationError("energy.urecs_address has invalid IPv6 brackets")
    elif address.count(":") > 1:
        # An unbracketed IPv6 literal has no unambiguous embedded port.
        host = address
    elif ":" in address:
        host, embedded_port = address.rsplit(":", 1)
        if not host:
            raise PlatformConfigurationError("energy.urecs_address host is empty")
        port = _strict_udp_port(
            embedded_port, field="energy.urecs_address embedded UDP port"
        )
    else:
        host = address

    if not host or any(ch in host for ch in "[]/\\"):
        raise PlatformConfigurationError("energy.urecs_address host is invalid")
    return host, port


def _validate_udp_command_token(value: Any, *, field: str) -> str:
    """Validate one exact firmware token without trimming or substitution."""

    if not isinstance(value, str) or not value:
        raise PlatformConfigurationError(
            f"{field} must be one exact non-empty ASCII token"
        )
    try:
        encoded = value.encode("ascii", errors="strict")
    except UnicodeEncodeError as exc:
        raise PlatformConfigurationError(
            f"{field} must be one exact non-empty ASCII token"
        ) from exc
    if not encoded or any(byte < 0x21 or byte > 0x7E for byte in encoded):
        raise PlatformConfigurationError(
            f"{field} must be one exact non-empty ASCII token"
        )
    return value


def _validate_remote_command(value: Any, *, field: str) -> str:
    """Validate one configured one-line SSH command without adding defaults."""

    if not isinstance(value, str) or not value.strip():
        raise PlatformConfigurationError(
            f"{field} must be a non-empty one-line string"
        )
    if any(ch in value for ch in "\r\n\x00"):
        raise PlatformConfigurationError(
            f"{field} must be a non-empty one-line string"
        )
    return value.strip()


def _strict_config_bool(value: Any, *, field: str, default: bool) -> bool:
    """Parse one safety boolean without treating a typo as false."""

    if value is None:
        # ``default`` applies only when the key is absent.  An explicitly
        # configured null is ambiguous and must never authorize a physical
        # operation by silently becoming the release default.
        raise PlatformConfigurationError(
            f"{field} must be an explicit true/false value"
        )
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "on", "enabled"}:
            return True
        if normalized in {"false", "no", "off", "disabled"}:
            return False
    raise PlatformConfigurationError(
        f"{field} must be an explicit true/false value"
    )


def _resolve_udp_endpoints(host: str, port: int) -> list[tuple[Any, ...]]:
    """Resolve the UDP target before a physical operation can begin."""

    try:
        infos = socket.getaddrinfo(host, int(port), type=socket.SOCK_DGRAM)
    except OSError as exc:
        raise PlatformConfigurationError(
            f"Cannot resolve u.RECS address {host}:{port}: {exc}"
        ) from exc
    if not infos:
        raise PlatformConfigurationError(
            f"Cannot resolve u.RECS address {host}:{port}"
        )
    return list(infos)


def _terminator_bytes(mode: Any) -> bytes:
    if not isinstance(mode, str):
        raise PlatformConfigurationError(
            "power_control.udp_terminator must be one of: lf, crlf, none"
        )
    normalized = mode.strip().lower()
    if normalized == "lf":
        return b"\n"
    if normalized == "crlf":
        return b"\r\n"
    if normalized == "none":
        return b""
    raise PlatformConfigurationError(
        "power_control.udp_terminator must be one of: lf, crlf, none"
    )


def _validate_platform_udp_configuration(
    setup: Mapping[str, Any],
    cfg: Mapping[str, Any],
    *,
    resolve_endpoints: bool = True,
) -> dict[str, Any]:
    """Pure fail-closed preflight for every power-changing UDP dependency."""

    _validate_platform_ssh_configuration(setup)
    configured_power = setup.get("power_control")
    if "power_control" in setup and not isinstance(configured_power, Mapping):
        raise PlatformConfigurationError(
            "setup.power_control must be a mapping for platform power operations"
        )
    energy = dict(cfg.get("energy") or {})
    configured_effective_power = cfg.get("power_control")
    if configured_effective_power is not None and not isinstance(
        configured_effective_power, Mapping
    ):
        raise PlatformConfigurationError(
            "power_control must be a mapping for platform power operations"
        )
    # Revalidate even when the caller supplied a pre-merged mapping.  This
    # makes this pure preflight independently fail-closed for direct callers.
    power = merged_power_control(configured_effective_power)
    raw_power = (
        dict(setup.get("power_control") or {})
        if isinstance(setup.get("power_control"), Mapping)
        else {}
    )
    if raw_power:
        # The raw setup is authoritative for explicitly configured fields.
        # Validate it separately so an invalid explicit null/string/value
        # cannot hide behind a valid effective default in ``cfg``.
        merged_power_control(raw_power)
    safety_bools: dict[str, bool] = {}
    for key, default in (
        ("enabled", True),
        ("require_ping_before_toggle", True),
        ("require_positive_calibration_delta", True),
    ):
        raw_value = (
            raw_power[key]
            if key in raw_power
            else power.get(key, default)
        )
        safety_bools[key] = _strict_config_bool(
            raw_value,
            field=f"power_control.{key}",
            default=default,
        )
    if not safety_bools["enabled"]:
        raise PlatformConfigurationError("Platform power control is disabled")

    # Use the explicit raw value when present.  This prevents tolerant status
    # merging from turning an invalid operation port into an implicit default.
    port_value = (
        raw_power["udp_port"]
        if "udp_port" in raw_power
        else power.get("udp_port", DEFAULT_POWER_CONTROL["udp_port"])
    )
    host, port = _parse_urecs_endpoint_strict(
        energy.get("urecs_address"), port_value
    )

    values: dict[str, Any] = {}
    for key in ("jetson_command", "m2_command", "udp_terminator"):
        values[key] = (
            raw_power[key]
            if key in raw_power
            else power.get(key, DEFAULT_POWER_CONTROL[key])
        )
    jetson_command = _validate_udp_command_token(
        values["jetson_command"], field="power_control.jetson_command"
    )
    m2_command = _validate_udp_command_token(
        values["m2_command"], field="power_control.m2_command"
    )
    terminator = values["udp_terminator"]
    _terminator_bytes(terminator)
    remote_commands: dict[str, str] = {}
    for key in ("shutdown_preflight_command", "shutdown_command"):
        # Missing fields inherit the release default.  An explicitly present
        # null/empty/non-string value is an error and may never disable the
        # sudo gate or resurrect a destructive command fallback.
        raw_value = (
            raw_power[key]
            if key in raw_power
            else power.get(key, DEFAULT_POWER_CONTROL[key])
        )
        remote_commands[key] = _validate_remote_command(
            raw_value, field=f"power_control.{key}"
        )
    resolved = _resolve_udp_endpoints(host, port) if resolve_endpoints else []
    return {
        "address": str(energy.get("urecs_address")),
        "host": host,
        "port": port,
        "udp_terminator": str(terminator).strip().lower(),
        "jetson_command": jetson_command,
        "m2_command": m2_command,
        "resolved_endpoints": tuple(resolved),
        "resolved_endpoint_count": len(resolved),
        "safety_bools": safety_bools,
        **remote_commands,
    }


_UDP_PREFLIGHT_BINDING_KEYS = (
    "address",
    "host",
    "port",
    "udp_terminator",
    "jetson_command",
    "m2_command",
    "safety_bools",
    "shutdown_preflight_command",
    "shutdown_command",
)


def _admit_pinned_udp_preflight(
    setup: Mapping[str, Any],
    cfg: Mapping[str, Any],
    preflight: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind one prior DNS result to the unchanged operation configuration."""

    current = _validate_platform_udp_configuration(
        setup,
        cfg,
        resolve_endpoints=False,
    )
    if any(
        current.get(key) != preflight.get(key)
        for key in _UDP_PREFLIGHT_BINDING_KEYS
    ):
        raise PlatformStateError(
            "ABORT_NO_MUTATION: UDP preflight binding does not match the "
            "current platform configuration"
        )
    endpoints = preflight.get("resolved_endpoints")
    if not isinstance(endpoints, (tuple, list)) or not endpoints:
        raise PlatformConfigurationError(
            "UDP preflight contains no pinned resolved endpoint"
        )
    pinned = tuple(endpoints)
    for entry in pinned:
        if not isinstance(entry, tuple) or len(entry) != 5:
            raise PlatformConfigurationError(
                "UDP preflight contains an invalid pinned endpoint"
            )
        family, socktype, _proto, _canonname, sockaddr = entry
        if (
            family not in {socket.AF_INET, socket.AF_INET6}
            or socktype != socket.SOCK_DGRAM
            or not isinstance(sockaddr, tuple)
            or len(sockaddr) < 2
            or type(sockaddr[1]) is not int
            or sockaddr[1] != int(current["port"])
        ):
            raise PlatformConfigurationError(
                "UDP preflight contains an invalid pinned endpoint"
            )
    current["resolved_endpoints"] = pinned
    current["resolved_endpoint_count"] = len(pinned)
    return current


def send_udp_command(
    address: str,
    command: str,
    *,
    port: int = 3000,
    terminator: str = "lf",
    timeout_s: float = 2.0,
    socket_factory: Callable[..., socket.socket] = socket.socket,
    _resolved_endpoints: tuple[tuple[Any, ...], ...] | None = None,
) -> dict[str, Any]:
    host, parsed_port = _parse_urecs_endpoint_strict(address, port)
    token = _validate_udp_command_token(command, field="u.RECS command")
    payload = token.encode("ascii", errors="strict") + _terminator_bytes(terminator)
    infos = (
        list(_resolved_endpoints)
        if _resolved_endpoints is not None
        else _resolve_udp_endpoints(host, parsed_port)
    )
    if not infos:
        raise PlatformConfigurationError(
            "UDP send requires at least one resolved endpoint"
        )
    last_error: Exception | None = None
    for family, socktype, proto, _canonname, sockaddr in infos:
        sock: socket.socket | None = None
        try:
            sock = socket_factory(family, socktype, proto)
            sock.settimeout(max(0.2, float(timeout_s)))
            sent = sock.sendto(payload, sockaddr)
            if sent != len(payload):
                raise OSError(f"short UDP send: {sent}/{len(payload)} bytes")
            return {
                "ok": True,
                "local_send_ok": True,
                "host": host,
                "port": int(parsed_port),
                "command": token,
                "payload_bytes": len(payload),
                "terminator": str(terminator),
                "sent_at": _now_iso(),
                "delivery_evidence": "local_udp_send_only_unacknowledged",
                "controller_acknowledged": False,
                "physical_state_confirmed": False,
                "evidence_note": (
                    "UDP sendto completed locally; u.RECS provides no command "
                    "acknowledgement and no physical rail-state response"
                ),
            }
        except Exception as exc:  # pragma: no branch - address fallback
            last_error = exc
        finally:
            if sock is not None:
                try:
                    sock.close()
                except Exception:
                    pass
    raise PlatformPowerError(
        f"UDP command {token!r} to {host}:{parsed_port} failed: {last_error}"
    )


def ping_host(
    host: str,
    *,
    timeout_s: float = 1.5,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> tuple[bool, str]:
    target = str(host or "").strip()
    if not target:
        return False, "not configured"
    timeout = max(0.2, float(timeout_s))
    system = platform.system().lower()
    if system == "windows":
        cmd = ["ping", "-n", "1", "-w", str(max(200, int(timeout * 1000))), target]
    elif system == "darwin":
        cmd = ["ping", "-c", "1", "-W", str(max(1000, int(timeout * 1000))), target]
    else:
        cmd = ["ping", "-c", "1", "-W", str(max(1, int(math.ceil(timeout)))), target]
    try:
        proc = runner(cmd, text=True, capture_output=True, timeout=timeout + 1.5)
    except FileNotFoundError:
        return False, "ping binary not found"
    except subprocess.TimeoutExpired:
        return False, f"ping timeout after {timeout:.1f}s"
    except Exception as exc:
        return False, f"ping failed: {type(exc).__name__}: {exc}"
    text = ((proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")).strip()
    tail = " | ".join(line.strip() for line in text.splitlines()[-2:] if line.strip())
    return proc.returncode == 0, tail or f"ping rc={proc.returncode}"


def accelerator_probe_script(accelerator: str) -> str:
    acc = str(accelerator or "").strip().lower().replace("-", "_")
    if acc.startswith("hailo"):
        return r"""
set +e
present=0
known=0
reason="not detected"
if compgen -G '/dev/hailo*' >/dev/null 2>&1; then
  present=1; known=1; reason="/dev/hailo*"
else
  scan=""
  if command -v hailortcli >/dev/null 2>&1; then
    known=1
    scan="$(timeout 8s hailortcli scan 2>&1)"
    printf '%s\n' "$scan"
    if printf '%s\n' "$scan" | grep -Eiq '(hailo[- ]?[0-9]|device[^:]*:|0000:[0-9a-f]{2}:)' \
       && ! printf '%s\n' "$scan" | grep -Eiq '(no devices|not found|zero devices)'; then
      present=1; reason="hailortcli scan"
    fi
  fi
  if [ "$present" -eq 0 ] && command -v lspci >/dev/null 2>&1; then
    known=1
    if lspci -nn | grep -Eiq '(hailo|1e60:)'; then
      present=1; reason="lspci"
    fi
  fi
fi
if [ "$known" -eq 0 ]; then reason="no supported Hailo presence probe available"; fi
printf '__SPLITPOINT_M2_KNOWN__=%s\n' "$known"
printf '__SPLITPOINT_M2_PRESENT__=%s\n' "$present"
printf '__SPLITPOINT_M2_REASON__=%s\n' "$reason"
exit 0
""".strip()
    if acc in {"deepx", "deepx_m1", "dx_m1", "dxm1"}:
        return r"""
set +e
present=0
known=0
reason="not detected"
if compgen -G '/dev/dxrt*' >/dev/null 2>&1; then
  present=1; known=1; reason="/dev/dxrt*"
elif command -v lspci >/dev/null 2>&1; then
  known=1
  if lspci -nn | grep -Eiq '(deepx|1ff4:)'; then
    present=1; reason="lspci"
  fi
fi
if [ "$present" -eq 0 ] && command -v dxrt-cli >/dev/null 2>&1; then
  known=1
  scan="$(timeout 8s dxrt-cli -s 2>&1)"
  scan_rc=$?
  printf '%s\n' "$scan"
  if [ "$scan_rc" -eq 0 ] && printf '%s\n' "$scan" | grep -Eiq '(device|dx[-_ ]?m1|deepx|ready)' \
     && ! printf '%s\n' "$scan" | grep -Eiq '(no devices|not found|error|failed)'; then
    present=1; reason="dxrt-cli"
  fi
fi
if [ "$known" -eq 0 ]; then reason="no supported DeepX presence probe available"; fi
printf '__SPLITPOINT_M2_KNOWN__=%s\n' "$known"
printf '__SPLITPOINT_M2_PRESENT__=%s\n' "$present"
printf '__SPLITPOINT_M2_REASON__=%s\n' "$reason"
exit 0
""".strip()
    return r"""
set +e
present=0
known=0
reason="accelerator-specific M.2 probe unsupported"
printf '__SPLITPOINT_M2_KNOWN__=%s\n' "$known"
printf '__SPLITPOINT_M2_PRESENT__=%s\n' "$present"
printf '__SPLITPOINT_M2_REASON__=%s\n' "$reason"
exit 0
""".strip()


def probe_accelerator_presence(
    setup: Mapping[str, Any],
    *,
    transport_factory: Callable[[HostConfig], SSHTransport] = SSHTransport,
    timeout_s: float = 12.0,
) -> tuple[Optional[bool], str]:
    host = host_config_from_setup(setup)
    transport = transport_factory(host)
    rc, out = transport.run_read_only(
        accelerator_probe_script(str(setup.get("accelerator") or "")),
        timeout_s=max(1, int(math.ceil(timeout_s))),
    )
    text = str(out or "")
    known_markers = re.findall(r"__SPLITPOINT_M2_KNOWN__=([01])", text)
    match = re.findall(r"__SPLITPOINT_M2_PRESENT__=([01])", text)
    reasons = re.findall(r"__SPLITPOINT_M2_REASON__=([^\r\n]+)", text)
    reason = reasons[-1].strip() if reasons else ""
    if rc != 0 or not known_markers or not match:
        tail = " | ".join(line.strip() for line in text.splitlines()[-4:] if line.strip())
        return None, f"probe rc={rc}; {tail or 'no state marker'}"
    if known_markers[-1] != "1":
        return None, reason or "accelerator-specific M.2 probe unsupported"
    present = match[-1] == "1"
    return present, reason or ("detected" if present else "not detected")


def probe_platform_status(
    setup_id: str,
    *,
    registry_path: str | Path | None = None,
    registry: Mapping[str, Any] | None = None,
    transport_factory: Callable[[HostConfig], SSHTransport] = SSHTransport,
    ping_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> PlatformStatus:
    _reg, setup, cfg = resolve_setup(setup_id, registry_path=registry_path, registry=registry)
    energy = cfg["energy"]
    power = cfg["power_control"]
    urecs_address = str(energy.get("urecs_address") or "").strip()
    urecs_host, urecs_port = parse_urecs_address(urecs_address, int(power["udp_port"]))
    if urecs_host:
        urecs_ok, urecs_detail = ping_host(
            urecs_host,
            timeout_s=float(power["status_ping_timeout_s"]),
            runner=ping_runner,
        )
    else:
        urecs_ok, urecs_detail = None, "not configured"

    jetson_host = ""
    jetson_ready: Optional[bool] = None
    jetson_detail = "not configured"
    try:
        host = host_config_from_setup(setup)
        jetson_host = host.user_host_pretty
        transport = transport_factory(host)
        jetson_ready, jetson_detail = transport.test_connection(
            timeout_s=max(1, int(math.ceil(float(power["ssh_probe_timeout_s"]))))
        )
    except PlatformConfigurationError as exc:
        jetson_detail = str(exc)

    m2_present: Optional[bool] = None
    m2_detail = "Jetson SSH not ready"
    if jetson_ready is True:
        m2_present, m2_detail = probe_accelerator_presence(
            setup,
            transport_factory=transport_factory,
            timeout_s=float(power["m2_probe_timeout_s"]),
        )

    return PlatformStatus(
        setup_id=str(setup.get("id") or setup_id),
        checked_at=_now_iso(),
        urecs_address=urecs_address,
        urecs_host=urecs_host,
        urecs_port=int(urecs_port),
        urecs_configured=bool(urecs_host),
        urecs_reachable=urecs_ok,
        urecs_detail=urecs_detail,
        jetson_host=jetson_host,
        jetson_configured=bool(jetson_host),
        jetson_ssh_ready=jetson_ready,
        jetson_detail=jetson_detail,
        accelerator=str(setup.get("accelerator") or ""),
        m2_present=m2_present,
        m2_detail=m2_detail,
    )


def _lock_root() -> Path:
    return Path(os.path.expanduser("~/.onnx_splitpoint_tool/locks"))


def held_workflow_locks(
    *,
    exclude_names: tuple[str, ...] = (
        "platform_power",
        "workflow_platform_interlock",
    ),
) -> list[str]:
    """Return lock files currently held by another process.

    The long-run launchers place their global ``flock`` files in this directory.
    Probing the lock itself is more reliable than looking for a PID name and also
    detects an older Tool version.  Unheld stale lock files are ignored.
    """

    if os.name != "posix":
        return []
    try:
        import fcntl
    except Exception:
        return []
    root = _lock_root()
    if not root.is_dir():
        return []
    held: list[str] = []
    for path in sorted(root.glob("*.lock")):
        name = path.name.lower()
        if any(token.lower() in name for token in exclude_names):
            continue
        try:
            fh = path.open("a+b")
        except Exception:
            continue
        try:
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                held.append(str(path))
            else:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        finally:
            fh.close()
    return held


def assert_no_active_workflow(*, force: bool = False) -> None:
    _require_exact_bool_argument(force, field="force")
    if force:
        return
    held = held_workflow_locks()
    if held:
        raise PlatformBusyError(
            "A benchmark/workflow lock is active; power switching is blocked: "
            + ", ".join(held)
        )


@contextmanager
def platform_operation_lock(
    setup_id: str,
    *,
    force_active_workflow: bool = False,
) -> Iterator[None]:
    _require_exact_bool_argument(
        force_active_workflow, field="force_active_workflow"
    )
    root = _lock_root()
    root.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(setup_id or "setup"))
    path = root / f"platform_power_{safe}.lock"
    interlock_fh: Any = None
    fh = path.open("a+b")
    try:
        if os.name == "posix":
            import fcntl

            interlock_path = platform_workflow_interlock_path()
            interlock_path.parent.mkdir(parents=True, exist_ok=True)
            interlock_fh = interlock_path.open("a+b")
            try:
                fcntl.flock(
                    interlock_fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB
                )
            except (BlockingIOError, OSError) as exc:
                raise PlatformBusyError(
                    "An EvaluationRun is active; platform-power operation is blocked"
                ) from exc
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise PlatformBusyError(
                    f"Another platform-power operation is active for {setup_id}"
                ) from exc
        # This second check runs only after the cross-process power gate is
        # owned.  It catches legacy shell launchers while the shared interlock
        # prevents a normal EvaluationRun from starting after this point.
        assert_no_active_workflow(force=force_active_workflow)
        yield
    finally:
        if os.name == "posix":
            try:
                import fcntl

                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
        fh.close()
        if interlock_fh is not None:
            try:
                if os.name == "posix":
                    import fcntl

                    fcntl.flock(interlock_fh.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
            finally:
                interlock_fh.close()


def _require_power_control_enabled(power: Mapping[str, Any]) -> None:
    if not _bool(power.get("enabled"), True):
        raise PlatformConfigurationError(
            "Platform power control is disabled for the selected hardware setup"
        )


def _require_urecs_ready(status: PlatformStatus, power: Mapping[str, Any]) -> None:
    _require_power_control_enabled(power)
    if not status.urecs_configured:
        raise PlatformConfigurationError("u.RECS address is not configured")
    if _bool(power.get("require_ping_before_toggle"), True) and status.urecs_reachable is not True:
        raise PlatformStateError(
            "u.RECS did not answer the bounded reachability probe; toggle blocked. "
            "Set power_control.require_ping_before_toggle=false only after verifying the route."
        )


def _require_calibration_urecs_ready(
    status: PlatformStatus, power: Mapping[str, Any]
) -> None:
    """Require the measurement endpoint without authorizing a rail toggle."""

    if not status.urecs_configured:
        raise PlatformConfigurationError("u.RECS address is not configured")
    if (
        _bool(power.get("require_ping_before_toggle"), True)
        and status.urecs_reachable is not True
    ):
        raise PlatformStateError(
            "u.RECS did not answer the bounded reachability probe; "
            "full-system calibration measurement blocked."
        )


def _wait_for_ssh(
    host: HostConfig,
    *,
    desired_up: bool,
    timeout_s: float,
    poll_s: float,
    transport_factory: Callable[[HostConfig], SSHTransport],
    callback: ProgressCallback | None = None,
) -> tuple[bool, str]:
    deadline = time.monotonic() + max(1.0, float(timeout_s))
    down_streak = 0
    last = "not checked"
    while time.monotonic() < deadline:
        remaining = max(1.0, deadline - time.monotonic())
        ok, detail = transport_factory(host).test_connection(
            timeout_s=max(1, min(5, int(math.ceil(remaining))))
        )
        last = detail
        if desired_up and ok:
            return True, detail
        if not desired_up:
            down_streak = down_streak + 1 if not ok else 0
            if down_streak >= 2:
                return True, detail
        _emit(callback, f"Waiting for Jetson SSH {'up' if desired_up else 'down'} …")
        time.sleep(max(0.5, float(poll_s)))
    return False, last


def _schedule_shutdown(
    setup: Mapping[str, Any],
    power: Mapping[str, Any],
    *,
    transport_factory: Callable[[HostConfig], SSHTransport],
) -> dict[str, Any]:
    host = host_config_from_setup(setup)
    transport = transport_factory(host)
    preflight = _validate_remote_command(
        power.get("shutdown_preflight_command"),
        field="power_control.shutdown_preflight_command",
    )
    rc, out = transport.run_read_only(preflight, timeout_s=15)
    if rc != 0:
        raise PlatformStateError(
            f"Jetson shutdown preflight failed (rc={rc}): {out.strip()}"
        )
    shutdown_command = _validate_remote_command(
        power.get("shutdown_command"),
        field="power_control.shutdown_command",
    )
    delayed = f"sleep 1; {shutdown_command}"
    command = (
        "nohup bash -lc "
        + shlex.quote(delayed)
        + " >/tmp/onnx_splitpoint_platform_poweroff.log 2>&1 </dev/null & "
        + "echo __SPLITPOINT_SHUTDOWN_SCHEDULED__"
    )
    rc, out = transport.run(command, timeout_s=15)
    if rc != 0 or "__SPLITPOINT_SHUTDOWN_SCHEDULED__" not in str(out or ""):
        raise PlatformStateError(
            f"Could not schedule Jetson shutdown (rc={rc}): {str(out or '').strip()}"
        )
    return {"ok": True, "command": shutdown_command, "output": str(out or "").strip()}


def _send_configured_toggle(
    setup: Mapping[str, Any],
    cfg: Mapping[str, Any],
    kind: str,
    *,
    udp_preflight: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    validated = (
        _validate_platform_udp_configuration(setup, cfg)
        if udp_preflight is None
        else _admit_pinned_udp_preflight(setup, cfg, udp_preflight)
    )
    command_key = "jetson_command" if kind == "jetson" else "m2_command"
    return send_udp_command(
        validated["address"],
        validated[command_key],
        port=int(validated["port"]),
        terminator=validated["udp_terminator"],
        _resolved_endpoints=tuple(validated["resolved_endpoints"]),
    )


def _stop_jetson_and_cut_rail(
    setup: Mapping[str, Any],
    cfg: Mapping[str, Any],
    *,
    transport_factory: Callable[[HostConfig], SSHTransport],
    callback: ProgressCallback | None,
    udp_preflight: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    validated_udp = (
        _validate_platform_udp_configuration(setup, cfg)
        if udp_preflight is None
        else _admit_pinned_udp_preflight(setup, cfg, udp_preflight)
    )
    power = dict(cfg.get("power_control") or {})
    host = host_config_from_setup(setup)
    _emit(callback, "Scheduling a graceful Jetson shutdown …")
    events: list[dict[str, Any]] = [{"step": "shutdown_schedule", **_schedule_shutdown(setup, power, transport_factory=transport_factory)}]
    ok, detail = _wait_for_ssh(
        host,
        desired_up=False,
        timeout_s=float(power["shutdown_timeout_s"]),
        poll_s=float(power["ssh_poll_interval_s"]),
        transport_factory=transport_factory,
        callback=callback,
    )
    events.append({"step": "ssh_down", "ok": ok, "detail": detail})
    if not ok:
        raise PlatformStateError("Jetson did not become unreachable after graceful shutdown")
    settle = float(power["shutdown_settle_s"])
    if settle > 0:
        _emit(
            callback,
            f"Jetson SSH is down; waiting {settle:.0f}s before the controller toggle …",
        )
        time.sleep(settle)
    _emit(
        callback,
        "Sending unacknowledged Jetson toggle; intended target is SSH not ready …",
    )
    events.append(
        {
            "step": "jetson_toggle_local_send",
            "intended_target": "ssh_not_ready",
            **_send_configured_toggle(
                setup,
                cfg,
                "jetson",
                udp_preflight=validated_udp,
            ),
        }
    )
    time.sleep(float(power["power_toggle_settle_s"]))
    return events


def _start_jetson(
    setup: Mapping[str, Any],
    cfg: Mapping[str, Any],
    *,
    transport_factory: Callable[[HostConfig], SSHTransport],
    callback: ProgressCallback | None,
    udp_preflight: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    validated_udp = (
        _validate_platform_udp_configuration(setup, cfg)
        if udp_preflight is None
        else _admit_pinned_udp_preflight(setup, cfg, udp_preflight)
    )
    power = dict(cfg.get("power_control") or {})
    host = host_config_from_setup(setup)
    _emit(
        callback,
        "Sending unacknowledged Jetson toggle; intended target is SSH ready …",
    )
    events = [
        {
            "step": "jetson_toggle_local_send",
            "intended_target": "ssh_ready",
            **_send_configured_toggle(
                setup,
                cfg,
                "jetson",
                udp_preflight=validated_udp,
            ),
        }
    ]
    time.sleep(float(power["power_toggle_settle_s"]))
    ok, detail = _wait_for_ssh(
        host,
        desired_up=True,
        timeout_s=float(power["boot_timeout_s"]),
        poll_s=float(power["ssh_poll_interval_s"]),
        transport_factory=transport_factory,
        callback=callback,
    )
    events.append({"step": "ssh_up", "ok": ok, "detail": detail})
    if not ok:
        raise PlatformStateError(
            "Jetson did not become SSH-ready after the unacknowledged controller toggle send"
        )
    return events


def _set_jetson_state_locked(
    setup_id: str,
    desired_up: bool,
    *,
    expected_ssh_ready: Optional[bool],
    registry: Mapping[str, Any],
    setup: Mapping[str, Any],
    cfg: Mapping[str, Any],
    callback: ProgressCallback | None,
    transport_factory: Callable[[HostConfig], SSHTransport],
    initial_status: PlatformStatus | None = None,
    udp_preflight: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Implement an explicit Jetson target while the operation lock is held."""

    validated_udp = (
        _validate_platform_udp_configuration(setup, cfg)
        if udp_preflight is None
        else _admit_pinned_udp_preflight(setup, cfg, udp_preflight)
    )
    status = initial_status or probe_platform_status(
        setup_id, registry=registry, transport_factory=transport_factory
    )
    _require_urecs_ready(status, cfg["power_control"])
    observed = status.jetson_ssh_ready
    if expected_ssh_ready is not None and observed is not expected_ssh_ready:
        raise PlatformStateError(
            "ABORT_NO_MUTATION: Jetson SSH observation changed after operator "
            f"confirmation (expected={expected_ssh_ready}, observed={observed})"
        )
    if observed is None:
        raise PlatformStateError(
            "ABORT_NO_MUTATION: Jetson SSH state is unknown; explicit target operation blocked"
        )

    events: list[dict[str, Any]] = []
    if bool(observed) == bool(desired_up):
        return {
            "ok": True,
            "action": (
                "jetson_target_ssh_ready_already_observed"
                if desired_up
                else "jetson_target_ssh_not_ready_already_observed"
            ),
            "changed": False,
            "desired_ssh_ready": bool(desired_up),
            "expected_ssh_ready": expected_ssh_ready,
            "physical_rail_state": (
                "powered_observed_via_authenticated_ssh"
                if desired_up
                else "unknown_not_inferred_from_ssh_unreachability"
            ),
            "before": status.to_dict(),
            "after": status.to_dict(),
            "events": events,
        }

    if desired_up is False:
        events.extend(
            _stop_jetson_and_cut_rail(
                setup,
                cfg,
                transport_factory=transport_factory,
                callback=callback,
                udp_preflight=validated_udp,
            )
        )
        final = probe_platform_status(
            setup_id, registry=registry, transport_factory=transport_factory
        )
        if final.jetson_ssh_ready is not False:
            raise PlatformStateError(
                "Jetson SSH-not-ready target was not verified after the controller toggle send"
            )
        return {
            "ok": True,
            "action": "jetson_target_ssh_not_ready",
            "changed": True,
            "desired_ssh_ready": False,
            "expected_ssh_ready": expected_ssh_ready,
            "physical_rail_state": "unknown_controller_does_not_report_rail_state",
            "before": status.to_dict(),
            "after": final.to_dict(),
            "events": events,
        }

    events.extend(
        _start_jetson(
            setup,
            cfg,
            transport_factory=transport_factory,
            callback=callback,
            udp_preflight=validated_udp,
        )
    )
    final = probe_platform_status(
        setup_id, registry=registry, transport_factory=transport_factory
    )
    if final.jetson_ssh_ready is not True:
        raise PlatformStateError(
            "Jetson SSH-ready target was not verified after the controller toggle send"
        )
    return {
        "ok": True,
        "action": "jetson_target_ssh_ready",
        "changed": True,
        "desired_ssh_ready": True,
        "expected_ssh_ready": expected_ssh_ready,
        "physical_rail_state": "powered_observed_via_authenticated_ssh",
        "before": status.to_dict(),
        "after": final.to_dict(),
        "events": events,
    }


def set_jetson_state(
    setup_id: str,
    desired_up: bool,
    *,
    expected_ssh_ready: Optional[bool] = None,
    registry_path: str | Path | None = None,
    force_active_workflow: bool = False,
    callback: ProgressCallback | None = None,
    transport_factory: Callable[[HostConfig], SSHTransport] = SSHTransport,
) -> dict[str, Any]:
    """Move the Jetson toward an explicit SSH-observable target.

    ``expected_ssh_ready`` binds a GUI confirmation to the exact observation
    shown to the operator.  The observation is refreshed under the global
    power lock; drift aborts before shutdown scheduling or UDP transmission.
    A successful local UDP send is never represented as a controller
    acknowledgement or as proof of a physical rail state.
    """

    _require_exact_bool_argument(desired_up, field="desired_up")
    _require_exact_bool_argument(
        force_active_workflow, field="force_active_workflow"
    )
    registry_path = _strict_registry_path_argument(registry_path)
    if expected_ssh_ready is not None and type(expected_ssh_ready) is not bool:
        raise PlatformConfigurationError(
            "expected_ssh_ready must be true, false, or omitted"
        )
    reg, setup, cfg = resolve_setup(setup_id, registry_path=registry_path)
    udp_preflight = _validate_platform_udp_configuration(setup, cfg)
    _require_power_control_enabled(cfg["power_control"])
    projection = _platform_control_configuration_projection(reg, setup_id)
    with platform_operation_lock(
        setup_id, force_active_workflow=force_active_workflow
    ):
        reg, setup, cfg, udp_preflight = _reload_locked_platform_configuration(
            setup_id,
            registry_path=registry_path,
            expected_projection=projection,
            expected_udp_preflight=udp_preflight,
        )
        return _set_jetson_state_locked(
            setup_id,
            desired_up,
            expected_ssh_ready=expected_ssh_ready,
            registry=reg,
            setup=setup,
            cfg=cfg,
            callback=callback,
            transport_factory=transport_factory,
            udp_preflight=udp_preflight,
        )


def toggle_jetson(
    setup_id: str,
    *,
    registry_path: str | Path | None = None,
    force_unknown_off_state: bool = False,
    force_active_workflow: bool = False,
    callback: ProgressCallback | None = None,
    transport_factory: Callable[[HostConfig], SSHTransport] = SSHTransport,
) -> dict[str, Any]:
    """Compatibility toggle whose inversion is chosen under the power lock."""

    _require_exact_bool_argument(
        force_unknown_off_state, field="force_unknown_off_state"
    )
    _require_exact_bool_argument(
        force_active_workflow, field="force_active_workflow"
    )
    registry_path = _strict_registry_path_argument(registry_path)
    reg, setup, cfg = resolve_setup(setup_id, registry_path=registry_path)
    udp_preflight = _validate_platform_udp_configuration(setup, cfg)
    _require_power_control_enabled(cfg["power_control"])
    projection = _platform_control_configuration_projection(reg, setup_id)
    with platform_operation_lock(
        setup_id, force_active_workflow=force_active_workflow
    ):
        reg, setup, cfg, udp_preflight = _reload_locked_platform_configuration(
            setup_id,
            registry_path=registry_path,
            expected_projection=projection,
            expected_udp_preflight=udp_preflight,
        )
        status = probe_platform_status(
            setup_id, registry=reg, transport_factory=transport_factory
        )
        if status.jetson_ssh_ready is True:
            desired_up = False
        elif status.jetson_ssh_ready is False and force_unknown_off_state:
            desired_up = True
        else:
            raise PlatformStateError(
                "Jetson SSH is not ready. The rail may be off, booting, or the network may be broken; "
                "the legacy toggle is blocked unless the unknown/off observation is explicitly confirmed."
            )
        result = _set_jetson_state_locked(
            setup_id,
            desired_up,
            expected_ssh_ready=status.jetson_ssh_ready,
            registry=reg,
            setup=setup,
            cfg=cfg,
            callback=callback,
            transport_factory=transport_factory,
            initial_status=status,
            udp_preflight=udp_preflight,
        )
        result["compatibility_api"] = "toggle_jetson"
        return result


def set_m2_state(
    setup_id: str,
    desired_present: bool,
    *,
    registry_path: str | Path | None = None,
    force_active_workflow: bool = False,
    callback: ProgressCallback | None = None,
    transport_factory: Callable[[HostConfig], SSHTransport] = SSHTransport,
) -> dict[str, Any]:
    _require_exact_bool_argument(desired_present, field="desired_present")
    _require_exact_bool_argument(
        force_active_workflow, field="force_active_workflow"
    )
    registry_path = _strict_registry_path_argument(registry_path)
    reg, setup, cfg = resolve_setup(setup_id, registry_path=registry_path)
    udp_preflight = _validate_platform_udp_configuration(setup, cfg)
    _require_power_control_enabled(cfg["power_control"])
    projection = _platform_control_configuration_projection(reg, setup_id)
    with platform_operation_lock(
        setup_id, force_active_workflow=force_active_workflow
    ):
        reg, setup, cfg, udp_preflight = _reload_locked_platform_configuration(
            setup_id,
            registry_path=registry_path,
            expected_projection=projection,
            expected_udp_preflight=udp_preflight,
        )
        return _set_m2_state_locked(
            setup_id,
            desired_present,
            registry=reg,
            setup=setup,
            cfg=cfg,
            callback=callback,
            transport_factory=transport_factory,
            udp_preflight=udp_preflight,
        )


def _observe_consistent_m2_state(
    setup_id: str,
    expected_present: bool,
    *,
    registry: Mapping[str, Any],
    power: Mapping[str, Any],
    transport_factory: Callable[[HostConfig], SSHTransport],
    callback: ProgressCallback | None,
) -> tuple[PlatformStatus, list[dict[str, Any]]]:
    """Require repeated post-boot observations of one accelerator state.

    SSH may be ready before the PCIe device and its driver have enumerated.
    Therefore an immediate ``m2_present=False`` is not sufficient evidence for
    an off target.  A configurable settle period is followed by at least two
    exact, consistent observations.  These prove only accelerator presence or
    absence through SSH, never the physical controller rail state.
    """

    settle_s = float(power["m2_post_boot_settle_s"])
    if settle_s > 0:
        _emit(
            callback,
            f"Waiting {settle_s:.0f}s for post-boot M.2 enumeration before verification …",
        )
        time.sleep(settle_s)
    count = max(2, int(power["m2_verify_observations"]))
    interval_s = max(0.0, float(power["m2_verify_interval_s"]))
    observations: list[dict[str, Any]] = []
    final: PlatformStatus | None = None
    for index in range(count):
        final = probe_platform_status(
            setup_id, registry=registry, transport_factory=transport_factory
        )
        observations.append(final.to_dict())
        if final.jetson_ssh_ready is not True:
            raise PlatformStateError(
                f"Jetson was not SSH-ready during M.2 verification observation {index + 1}/{count}"
            )
        if final.m2_present is not expected_present:
            observed = (
                "unknown" if final.m2_present is None else "present" if final.m2_present else "absent"
            )
            raise PlatformStateError(
                "M.2 verification mismatch at observation "
                f"{index + 1}/{count}: requested accelerator "
                f"{'present' if expected_present else 'absent'}, observed {observed}"
            )
        if index + 1 < count and interval_s > 0:
            time.sleep(interval_s)
    assert final is not None
    return final, observations


def _set_m2_state_locked(
    setup_id: str,
    desired_present: bool,
    *,
    registry: Mapping[str, Any],
    setup: Mapping[str, Any],
    cfg: Mapping[str, Any],
    callback: ProgressCallback | None,
    transport_factory: Callable[[HostConfig], SSHTransport],
    initial_status: PlatformStatus | None = None,
    udp_preflight: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    validated_udp = (
        _validate_platform_udp_configuration(setup, cfg)
        if udp_preflight is None
        else _admit_pinned_udp_preflight(setup, cfg, udp_preflight)
    )
    status = initial_status or probe_platform_status(
        setup_id, registry=registry, transport_factory=transport_factory
    )
    _require_urecs_ready(status, cfg["power_control"])
    if status.jetson_ssh_ready is not True:
        raise PlatformStateError("M.2 switching requires an SSH-ready Jetson")
    if status.m2_present is None:
        raise PlatformStateError("M.2 state is unknown; toggle blocked")
    if bool(status.m2_present) == bool(desired_present):
        return {
            "ok": True,
            "action": (
                "m2_target_accelerator_present_already_observed"
                if desired_present
                else "m2_target_accelerator_absent_already_observed"
            ),
            "changed": False,
            "desired_accelerator_present": bool(desired_present),
            "physical_rail_state": "unknown_controller_does_not_report_rail_state",
            "before": status.to_dict(),
            "after": status.to_dict(),
            "events": [],
        }

    events = _stop_jetson_and_cut_rail(
        setup,
        cfg,
        transport_factory=transport_factory,
        callback=callback,
        udp_preflight=validated_udp,
    )
    _emit(
        callback,
        "Sending an unacknowledged M.2 toggle; intended observed target is "
        f"accelerator {'present' if desired_present else 'absent'} …",
    )
    events.append(
        {
            "step": "m2_toggle_local_send",
            "intended_accelerator_present": bool(desired_present),
            **_send_configured_toggle(
                setup,
                cfg,
                "m2",
                udp_preflight=validated_udp,
            ),
        }
    )
    time.sleep(float(cfg["power_control"]["m2_toggle_settle_s"]))
    events.extend(
        _start_jetson(
            setup,
            cfg,
            transport_factory=transport_factory,
            callback=callback,
            udp_preflight=validated_udp,
        )
    )
    final, observations = _observe_consistent_m2_state(
        setup_id,
        desired_present,
        registry=registry,
        power=cfg["power_control"],
        transport_factory=transport_factory,
        callback=callback,
    )
    events.append(
        {
            "step": "m2_post_boot_observations",
            "expected_accelerator_present": bool(desired_present),
            "physical_rail_state": "unknown_controller_does_not_report_rail_state",
            "observations": observations,
        }
    )
    return {
        "ok": True,
        "action": (
            "m2_target_accelerator_present"
            if desired_present
            else "m2_target_accelerator_absent"
        ),
        "changed": True,
        "desired_accelerator_present": bool(desired_present),
        "physical_rail_state": "unknown_controller_does_not_report_rail_state",
        "before": status.to_dict(),
        "after": final.to_dict(),
        "events": events,
    }


def toggle_m2(
    setup_id: str,
    *,
    registry_path: str | Path | None = None,
    force_active_workflow: bool = False,
    callback: ProgressCallback | None = None,
    transport_factory: Callable[[HostConfig], SSHTransport] = SSHTransport,
) -> dict[str, Any]:
    _require_exact_bool_argument(
        force_active_workflow, field="force_active_workflow"
    )
    registry_path = _strict_registry_path_argument(registry_path)
    reg, setup, cfg = resolve_setup(setup_id, registry_path=registry_path)
    udp_preflight = _validate_platform_udp_configuration(setup, cfg)
    _require_power_control_enabled(cfg["power_control"])
    projection = _platform_control_configuration_projection(reg, setup_id)
    with platform_operation_lock(
        setup_id, force_active_workflow=force_active_workflow
    ):
        reg, setup, cfg, udp_preflight = _reload_locked_platform_configuration(
            setup_id,
            registry_path=registry_path,
            expected_projection=projection,
            expected_udp_preflight=udp_preflight,
        )
        status = probe_platform_status(
            setup_id, registry=reg, transport_factory=transport_factory
        )
        if status.m2_present is None:
            raise PlatformStateError("M.2 state is unknown; toggle blocked")
        result = _set_m2_state_locked(
            setup_id,
            not bool(status.m2_present),
            registry=reg,
            setup=setup,
            cfg=cfg,
            callback=callback,
            transport_factory=transport_factory,
            initial_status=status,
            udp_preflight=udp_preflight,
        )
        result["compatibility_api"] = "toggle_m2"
        return result


def _measurement_gate_reasons(result: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []

    def add(raw: Any) -> None:
        values = raw if isinstance(raw, (list, tuple, set)) else [raw]
        for value in values:
            text = str(value or "").strip()
            if text and text not in reasons:
                reasons.append(text)

    add(result.get("final_energy_gate_reasons"))
    for failure in list(result.get("final_energy_gate_failures") or []):
        if isinstance(failure, Mapping):
            add(failure.get("reasons"))
    for run in list(result.get("runs") or []):
        if isinstance(run, Mapping):
            add(run.get("final_energy_gate_reasons"))
    return reasons


def _measurement_failure_detail(
    result: Mapping[str, Any],
    *,
    output_dir: str | Path | None = None,
) -> str:
    fields: list[str] = [f"status={result.get('status') or 'invalid'}"]
    reasons = _measurement_gate_reasons(result)
    if reasons:
        fields.append("final_energy_gate_reasons=" + ",".join(reasons))
    acquisition_reasons = [
        str(value).strip()
        for value in list(result.get("acquisition_integrity_failure_reasons") or [])
        if str(value).strip()
    ]
    if acquisition_reasons:
        fields.append(
            "acquisition_integrity_failure_reasons="
            + ",".join(acquisition_reasons)
        )
    if result.get("invalid_repeat_retry_suppressed_by_exact_run_count") is True:
        fields.append("invalid_repeat_retry_suppressed_by_exact_run_count=true")
    calibration_status = str(
        result.get("full_system_scope_calibration_status") or ""
    ).strip()
    if calibration_status:
        fields.append(f"calibration_status={calibration_status}")
    verification = result.get("energy_calibration_verification")
    if isinstance(verification, Mapping):
        method_status = str(verification.get("status") or "").strip()
        if method_status and method_status != calibration_status:
            fields.append(f"calibration_verification={method_status}")
        binding_errors = list(verification.get("runtime_binding_errors") or [])
        if binding_errors:
            fields.append(
                "runtime_binding_errors="
                + ",".join(str(value) for value in binding_errors)
            )
    error = str(result.get("error") or result.get("reason") or "").strip()
    if error:
        fields.append(f"error={error}")
    if output_dir is not None:
        fields.append(f"evidence_dir={Path(output_dir).expanduser().resolve()}")
    return "; ".join(fields)


def _measurement_power(
    result: Mapping[str, Any],
    *,
    output_dir: str | Path | None = None,
) -> float:
    value = result.get("avg_power_w")
    try:
        if isinstance(value, bool):
            raise TypeError("boolean is not a power measurement")
        power = float(value)
    except Exception as exc:
        raise PlatformPowerError(
            "u.RECS measurement returned no average power: "
            + _measurement_failure_detail(result, output_dir=output_dir)
        ) from exc
    if result.get("ok") is not True or not math.isfinite(power) or power <= 0.0:
        raise PlatformPowerError(
            "u.RECS measurement failed: "
            + _measurement_failure_detail(result, output_dir=output_dir)
        )
    return power


def measure_idle_power(
    setup_id: str,
    *,
    registry: Mapping[str, Any],
    setup: Mapping[str, Any],
    cfg: Mapping[str, Any],
    state_label: str,
    duration_s: float,
    output_dir: str | Path,
    callback: ProgressCallback | None = None,
    measurement_runner: Callable[..., dict[str, Any]] | None = None,
    energy_method: Mapping[str, Any] | None = None,
    bypass_full_system_current_scale: bool = False,
    claim_exclusion_reason: str = "m2_accelerator_idle_power_calibration",
    run_id_prefix: str = "m2_idle_calibration",
    exact_run_count: bool = True,
    invalid_repeat_max_retries: int | None = None,
    require_command_window_alignment: bool = False,
) -> tuple[float, dict[str, Any]]:
    # Accepted only for source compatibility with callers from v2.79.12/13.
    # The direct v2.79.14 calibration deliberately ignores this legacy input.
    del energy_method
    _require_exact_bool_argument(
        bypass_full_system_current_scale,
        field="bypass_full_system_current_scale",
    )
    _require_exact_bool_argument(exact_run_count, field="exact_run_count")
    _require_exact_bool_argument(
        require_command_window_alignment,
        field="require_command_window_alignment",
    )
    if invalid_repeat_max_retries is not None and (
        isinstance(invalid_repeat_max_retries, bool)
        or not isinstance(invalid_repeat_max_retries, int)
        or invalid_repeat_max_retries < 0
        or invalid_repeat_max_retries > 3
    ):
        raise PlatformConfigurationError(
            "invalid_repeat_max_retries must be an integer in range 0..3 or omitted"
        )
    if measurement_runner is None:
        from .energy.collector import run_fast_firmware_measurement

        measurement_runner = run_fast_firmware_measurement
    energy = dict(cfg.get("energy") or {})
    address = str(energy.get("urecs_address") or "").strip()
    if not address:
        raise PlatformConfigurationError("u.RECS address is not configured")
    defaults: EnergyDefaults = copy.deepcopy(energy_defaults_from_registry(registry))
    defaults.run_count = 1
    defaults.physical_scope = "FS"
    # Keep the hardware-registry contract spelling.  The collector canonicalizes
    # ``command`` to the command-marker window for alignment and retry logic.
    selected_window_label = "command"
    defaults.window_label = selected_window_label
    defaults.compare_legacy_window = False
    defaults.window_ab_enabled = False
    defaults.keep_raw_parquet = True
    defaults.postprocess_with_power_calculations = True
    setup_energy = copy.deepcopy(
        energy_setup_from_registry(registry, setup_id)
    )
    setup_energy.enabled = True
    setup_energy.urecs_address = address
    # The calibration needs the two absolute full-system means.  It is a
    # diagnostic capture used only to calculate the local M.2 idle delta, not
    # an energy-efficiency claim.  Keep method provenance out of this isolated
    # runtime copy even when an older registry still contains such fields.
    setup_energy.idle_baseline_w = None
    setup_energy.accelerator_idle_w = None
    # The FS input-gain routine must use the untrimmed chain, while the M.2
    # idle calibration must use the verified factor so its saved watt value is
    # already in the corrected absolute scale.
    if bypass_full_system_current_scale:
        setup_energy.full_system_current_scale_factor = None
        setup_energy.full_system_current_scale_calibrated_at = ""
        setup_energy.full_system_current_scale_calibration_evidence = ""
        setup_energy.full_system_current_scale_calibration_sha256 = ""
    setup_energy.calibration_manifest = ""
    setup_energy.calibration_sha256 = ""
    setup_energy.expected_channel_bindings = ()
    setup_energy.expected_channel_bindings_valid = False
    setup_energy.expected_channel_binding_errors = ()
    duration = max(5.0, float(duration_s))
    command = "bash -lc " + shlex.quote(
        f"echo __SPLITPOINT_IDLE_CALIBRATION_STATE__={shlex.quote(state_label)}; sleep {duration:.3f}"
    )
    _emit(callback, f"Measuring {state_label} idle power for {duration:.0f}s …")
    result = measurement_runner(
        command,
        Path(output_dir),
        setup=setup_energy,
        defaults=defaults,
        duration_s=duration,
        run_count=1,
        exact_run_count=exact_run_count,
        invalid_repeat_max_retries=invalid_repeat_max_retries,
        compare_legacy_window=False,
        setup_id=setup_id,
        run_id=f"{run_id_prefix}_{state_label}",
        physical_scope="FS",
        window_label=selected_window_label,
        require_command_window_alignment=require_command_window_alignment,
        diagnostic_only=True,
        claim_exclusion_reason=claim_exclusion_reason,
    )
    result = dict(result)
    try:
        power_value = _measurement_power(result, output_dir=output_dir)
    except PlatformPowerError as exc:
        # A failed aggregate is still important calibration evidence.  Carry
        # the complete returned summary to the guided workflow instead of
        # reducing it to an exception string and losing the acquisition gate.
        try:
            setattr(exc, "measurement_result", result)
            setattr(
                exc,
                "measurement_output_dir",
                str(Path(output_dir).resolve()),
            )
        except Exception:
            pass
        raise
    return power_value, result


def _save_accelerator_idle_power(
    registry: MutableMapping[str, Any],
    setup_id: str,
    value_w: float,
    *,
    registry_path: str | Path | None,
    calibration_record: Mapping[str, Any],
    expected_configuration_projection: Mapping[str, Any],
) -> None:
    # A calibration runs for minutes, so the registry snapshot loaded at its
    # start is intentionally *not* authoritative at commit time.  Serialize
    # against all normal registry saves, reload the current bytes while holding
    # that stable lock, and merge only calibration-owned fields.
    path = energy_config._expand(  # type: ignore[attr-defined]
        registry_path or energy_config.default_registry_path()
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with energy_config._hardware_registry_write_lock(path):  # type: ignore[attr-defined]
        fresh_registry = load_hardware_registry(path)
        fresh_projection = _calibration_configuration_projection(
            fresh_registry,
            setup_id,
        )
        if dict(fresh_projection) != dict(expected_configuration_projection):
            raise PlatformStateError(
                "configuration_changed_during_calibration"
            )
        _merge_accelerator_idle_power_fields(
            fresh_registry,
            setup_id,
            value_w,
            calibration_record=calibration_record,
        )
        energy_config._write_hardware_registry_atomic(  # type: ignore[attr-defined]
            fresh_registry, path
        )


def _merge_accelerator_idle_power_fields(
    registry: MutableMapping[str, Any],
    setup_id: str,
    value_w: float,
    *,
    calibration_record: Mapping[str, Any],
) -> None:
    found = False
    setups = list(registry.get("hardware_setups") or [])
    for raw in setups:
        if not isinstance(raw, MutableMapping):
            continue
        if str(raw.get("id") or "") != str(setup_id):
            continue
        energy = dict(raw.get("energy") or {}) if isinstance(raw.get("energy"), Mapping) else {}
        energy["accelerator_idle_w"] = float(value_w)
        energy["accelerator_idle_calibrated_at"] = str(calibration_record.get("finished_at") or _now_iso())
        energy["accelerator_idle_calibration_evidence"] = str(
            calibration_record.get("evidence_path")
            or calibration_record.get("output_dir")
            or ""
        )
        # A successful simple calibration supersedes any older sealed binding.
        # Leaving those fields behind would make downstream readers associate
        # the new scalar with obsolete evidence.
        energy.pop("accelerator_idle_calibration_binding_path", None)
        energy.pop("accelerator_idle_calibration_binding_sha256", None)
        raw["energy"] = energy
        found = True
        break
    if not found:
        raise PlatformConfigurationError(f"Cannot save calibration; setup disappeared: {setup_id}")
    registry["hardware_setups"] = setups


def _verify_calibration_capture_state(
    setup_id: str,
    expected_m2_present: bool,
    *,
    registry: Mapping[str, Any],
    transport_factory: Callable[[HostConfig], SSHTransport],
    phase: str,
) -> PlatformStatus:
    """Bind a calibration capture boundary to a fresh observed state."""

    status = probe_platform_status(
        setup_id, registry=registry, transport_factory=transport_factory
    )
    if status.jetson_ssh_ready is not True:
        raise PlatformStateError(
            f"Calibration {phase}: Jetson is not authenticated-SSH ready"
        )
    if status.m2_present is not expected_m2_present:
        observed = (
            "unknown"
            if status.m2_present is None
            else "present"
            if status.m2_present
            else "absent"
        )
        raise PlatformStateError(
            f"Calibration {phase}: expected accelerator "
            f"{'present' if expected_m2_present else 'absent'}, observed {observed}"
        )
    return status


def _prepare_calibration_output_root(root: Path) -> Path:
    """Create or admit one empty output root before hardware is touched."""

    candidate = root.expanduser().resolve()
    if candidate.exists():
        if candidate.is_symlink() or not candidate.is_dir():
            raise PlatformConfigurationError(
                f"Calibration output root is not a canonical directory: {candidate}"
            )
        if any(candidate.iterdir()):
            raise PlatformConfigurationError(
                f"Calibration output root is not empty: {candidate}"
            )
    else:
        candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def _verify_locked_calibration_output_root(
    root: Path,
    *,
    operational_evidence_path: Path,
) -> None:
    """Reject artefact injection while the caller waited for the power lock."""

    try:
        resolved = root.resolve(strict=True)
    except Exception as exc:
        raise PlatformConfigurationError(
            "Calibration output root disappeared while waiting for the platform lock"
        ) from exc
    if resolved != root or root.is_symlink() or not root.is_dir():
        raise PlatformConfigurationError(
            "Calibration output root changed while waiting for the platform lock"
        )
    expected = {operational_evidence_path.name}
    actual = {path.name for path in root.iterdir()}
    if actual != expected or not operational_evidence_path.is_file():
        raise PlatformConfigurationError(
            "Calibration output collision detected before hardware mutation"
        )


def calibrate_m2_accelerator_idle_power(
    setup_id: str,
    *,
    registry_path: str | Path | None = None,
    stabilize_s: float | None = None,
    measure_s: float | None = None,
    output_dir: str | Path | None = None,
    force_active_workflow: bool = False,
    callback: ProgressCallback | None = None,
    transport_factory: Callable[[HostConfig], SSHTransport] = SSHTransport,
    measurement_runner: Callable[..., dict[str, Any]] | None = None,
) -> CalibrationResult:
    """Measure Jetson idle power with M.2 off and on, then save the difference.

    Preconditions and every transition are verified.  The old registry value is
    left untouched until both captures succeeded and the M.2 accelerator was
    restored to the on/present state.
    """

    _require_exact_bool_argument(
        force_active_workflow, field="force_active_workflow"
    )
    registry_path = _strict_registry_path_argument(registry_path)
    # CLI/API overrides are just as claim-bearing as registry values.  Validate
    # them before resolving the setup or creating an evidence directory, and
    # never let float("nan"), infinity, strings, booleans, or clamping alter the
    # requested physical measurement window.
    stabilize_override = (
        None
        if stabilize_s is None
        else _strict_power_control_number(
            stabilize_s,
            field="stabilize_s",
            minimum=0.0,
            maximum=None,
            integer=False,
        )
    )
    measure_override = (
        None
        if measure_s is None
        else _strict_power_control_number(
            measure_s,
            field="measure_s",
            minimum=5.0,
            maximum=None,
            integer=False,
        )
    )
    started_at = _now_iso()
    registry, setup, cfg = resolve_setup(setup_id, registry_path=registry_path)
    configuration_projection = _calibration_configuration_projection(
        registry,
        setup_id,
    )
    power = cfg["power_control"]
    udp_preflight = _validate_platform_udp_configuration(setup, cfg)
    _require_power_control_enabled(power)
    stabilize = (
        float(power["calibration_stabilize_s"])
        if stabilize_override is None
        else float(stabilize_override)
    )
    measure = (
        float(power["calibration_measure_s"])
        if measure_override is None
        else float(measure_override)
    )
    stamp = time.strftime("%Y%m%d_%H%M%S")
    root = Path(output_dir) if output_dir else energy_measurements_root() / "Calibrations" / setup_id / stamp
    root = _prepare_calibration_output_root(root)
    evidence: dict[str, Any] = {
        "schema": "onnx-splitpoint/m2-idle-power-calibration",
        "schema_version": 2,
        "setup_id": setup_id,
        "accelerator": str(setup.get("accelerator") or ""),
        "urecs_address": str(
            dict(cfg.get("energy") or {}).get("urecs_address") or ""
        ).strip(),
        "data_port": int(energy_defaults_from_registry(registry).data_port),
        "started_at": started_at,
        "stabilize_s": stabilize,
        "measure_s": measure,
        "configuration_projection": configuration_projection,
        "physical_scope": "FS",
        "measurement_scope": "full_system",
        "events": [],
        "status": "running",
        "saved": False,
    }
    evidence_path = root / "m2_idle_power_calibration.json"

    def checkpoint() -> None:
        tmp = evidence_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp, evidence_path)

    checkpoint()
    restored = False
    preflight_completed = False
    off_power: float | None = None
    on_power: float | None = None
    off_measurement: dict[str, Any] | None = None
    on_measurement: dict[str, Any] | None = None
    with platform_operation_lock(
        setup_id, force_active_workflow=force_active_workflow
    ):
        try:
            # The caller can wait for this global lock.  Re-read the registry
            # only after admission and reject any relevant change from the
            # pre-lock snapshot before using configuration or touching power.
            locked_registry = load_hardware_registry(registry_path)
            locked_projection = _calibration_configuration_projection(
                locked_registry,
                setup_id,
            )
            if locked_projection != configuration_projection:
                raise PlatformStateError(
                    "configuration_changed_while_waiting_for_platform_lock"
                )
            registry, setup, cfg = resolve_setup(
                setup_id,
                registry=locked_registry,
            )
            power = cfg["power_control"]
            udp_preflight = _admit_pinned_udp_preflight(
                setup,
                cfg,
                udp_preflight,
            )
            _require_power_control_enabled(power)
            _verify_locked_calibration_output_root(
                root,
                operational_evidence_path=evidence_path,
            )
            initial = probe_platform_status(
                setup_id, registry=registry, transport_factory=transport_factory
            )
            evidence["initial_status"] = initial.to_dict()
            checkpoint()
            _require_urecs_ready(initial, power)
            if initial.jetson_ssh_ready is not True:
                raise PlatformStateError("Calibration requires an SSH-ready Jetson")
            if initial.m2_present is not True:
                raise PlatformStateError(
                    "Calibration starts only from a verified M.2-present state"
                )
            preflight_completed = True
            restored = True
            evidence["preflight"] = "passed"
            checkpoint()

            _emit(callback, "Preparing verified M.2-off state …")
            off_transition = _set_m2_state_locked(
                setup_id,
                False,
                registry=registry,
                setup=setup,
                cfg=cfg,
                callback=callback,
                transport_factory=transport_factory,
                udp_preflight=udp_preflight,
            )
            evidence["events"].append({"transition": "m2_off", "result": off_transition})
            checkpoint()
            if stabilize > 0:
                _emit(callback, f"Stabilizing M.2-off idle state for {stabilize:.0f}s …")
                time.sleep(stabilize)
            off_pre = _verify_calibration_capture_state(
                setup_id,
                False,
                registry=registry,
                transport_factory=transport_factory,
                phase="m2_off/pre_measurement",
            )
            evidence["m2_off"] = {
                "pre_measurement_status": off_pre.to_dict(),
            }
            checkpoint()
            off_power, off_measurement = measure_idle_power(
                setup_id,
                registry=registry,
                setup=setup,
                cfg=cfg,
                state_label="m2_off",
                duration_s=measure,
                output_dir=root / "m2_off",
                callback=callback,
                measurement_runner=measurement_runner,
            )
            evidence["m2_off"].update(
                {
                    "avg_power_w": off_power,
                    "measurement": off_measurement,
                }
            )
            checkpoint()
            off_post = _verify_calibration_capture_state(
                setup_id,
                False,
                registry=registry,
                transport_factory=transport_factory,
                phase="m2_off/post_measurement",
            )
            evidence["m2_off"]["post_measurement_status"] = off_post.to_dict()
            checkpoint()

            _emit(callback, "Preparing verified M.2-on state …")
            on_transition = _set_m2_state_locked(
                setup_id,
                True,
                registry=registry,
                setup=setup,
                cfg=cfg,
                callback=callback,
                transport_factory=transport_factory,
                udp_preflight=udp_preflight,
            )
            restored = True
            evidence["events"].append({"transition": "m2_on", "result": on_transition})
            checkpoint()
            if stabilize > 0:
                _emit(callback, f"Stabilizing M.2-on idle state for {stabilize:.0f}s …")
                time.sleep(stabilize)
            on_pre = _verify_calibration_capture_state(
                setup_id,
                True,
                registry=registry,
                transport_factory=transport_factory,
                phase="m2_on/pre_measurement",
            )
            evidence["m2_on"] = {
                "pre_measurement_status": on_pre.to_dict(),
            }
            checkpoint()
            on_power, on_measurement = measure_idle_power(
                setup_id,
                registry=registry,
                setup=setup,
                cfg=cfg,
                state_label="m2_on",
                duration_s=measure,
                output_dir=root / "m2_on",
                callback=callback,
                measurement_runner=measurement_runner,
            )
            evidence["m2_on"].update(
                {
                    "avg_power_w": on_power,
                    "measurement": on_measurement,
                }
            )
            checkpoint()
            on_post = _verify_calibration_capture_state(
                setup_id,
                True,
                registry=registry,
                transport_factory=transport_factory,
                phase="m2_on/post_measurement",
            )
            evidence["m2_on"]["post_measurement_status"] = on_post.to_dict()
            delta = float(on_power - off_power)
            evidence["idle_power_without_m2_w"] = float(off_power)
            evidence["idle_power_with_m2_w"] = float(on_power)
            evidence["accelerator_idle_power_w"] = delta
            minimum = float(power["minimum_calibration_delta_w"])
            if _bool(power.get("require_positive_calibration_delta"), True) and delta < minimum:
                raise PlatformStateError(
                    f"Measured accelerator idle delta {delta:.4f} W is below the configured minimum {minimum:.4f} W"
                )

            finished_at = _now_iso()
            evidence["finished_at"] = finished_at
            evidence["status"] = "ok"
            evidence["restored_m2_on"] = True
            evidence["saved"] = False
            checkpoint()
            _save_accelerator_idle_power(
                registry,
                setup_id,
                delta,
                registry_path=registry_path,
                calibration_record={
                    "finished_at": finished_at,
                    "evidence_path": str(evidence_path),
                },
                expected_configuration_projection=configuration_projection,
            )
            evidence["saved"] = True
            checkpoint()
            _emit(callback, f"Calibration saved: accelerator idle power = {delta:.4f} W")
            return CalibrationResult(
                setup_id=setup_id,
                accelerator=str(setup.get("accelerator") or ""),
                idle_power_without_m2_w=float(off_power),
                idle_power_with_m2_w=float(on_power),
                accelerator_idle_power_w=delta,
                started_at=started_at,
                finished_at=finished_at,
                output_dir=str(root),
                saved=True,
                restored_m2_on=True,
                evidence=evidence,
            )
        except BaseException as exc:
            evidence["status"] = "failed"
            evidence["error_type"] = type(exc).__name__
            evidence["error"] = f"{type(exc).__name__}: {exc}"
            evidence["interrupted"] = isinstance(exc, (KeyboardInterrupt, SystemExit))
            # A failed preflight has not authorized any hardware mutation.  In
            # particular, starting from M.2-absent must never trigger a
            # surprise recovery power-cycle.  Once preflight passed, recovery
            # is still bounded by fresh SSH and accelerator-state evidence.
            if not preflight_completed:
                evidence["recovery_skipped_reason"] = "preflight_not_completed"
            else:
                try:
                    current = probe_platform_status(
                        setup_id,
                        registry=registry,
                        transport_factory=transport_factory,
                    )
                    evidence["failure_status"] = current.to_dict()
                    restored = current.m2_present is True
                    if (
                        current.jetson_ssh_ready is True
                        and current.m2_present is False
                    ):
                        _emit(
                            callback,
                            "Calibration failed; restoring the verified M.2-on state …",
                        )
                        recovery = _set_m2_state_locked(
                            setup_id,
                            True,
                            registry=registry,
                            setup=setup,
                            cfg=cfg,
                            callback=callback,
                            transport_factory=transport_factory,
                            udp_preflight=udp_preflight,
                        )
                        evidence["recovery"] = recovery
                        restored = True
                    elif not restored:
                        evidence["recovery_skipped_reason"] = (
                            "fresh_state_not_safe_for_recovery"
                        )
                except BaseException as recovery_exc:
                    restored = False
                    evidence["recovery_error"] = (
                        f"{type(recovery_exc).__name__}: {recovery_exc}"
                    )
            evidence["restored_m2_on"] = restored
            evidence["finished_at"] = _now_iso()
            checkpoint()
            try:
                setattr(exc, "evidence_path", str(evidence_path))
                setattr(exc, "evidence_dir", str(root))
            except Exception:
                pass
            if isinstance(exc, PlatformPowerError):
                marker = f"calibration_evidence={evidence_path}"
                if marker not in str(exc):
                    remaining = tuple(getattr(exc, "args", ()))[1:]
                    exc.args = (f"{exc}; {marker}", *remaining)
            raise



def _full_system_calibration_configuration_projection(
    registry: Mapping[str, Any],
    setup_id: str,
) -> dict[str, Any]:
    """Bind a guided FS calibration to its exact setup and previous trim."""

    projection = _calibration_configuration_projection(registry, setup_id)
    setup = _setup_by_id(registry, setup_id)
    energy = (
        dict(setup.get("energy") or {})
        if isinstance(setup.get("energy"), Mapping)
        else {}
    )
    projection["schema"] = (
        "onnx-splitpoint/full-system-input-scale-calibration-config-projection"
    )
    projection["existing_full_system_current_scale"] = {
        "factor": energy.get("full_system_current_scale_factor"),
        "calibrated_at": str(
            energy.get("full_system_current_scale_calibrated_at") or ""
        ),
        "evidence": str(
            energy.get("full_system_current_scale_calibration_evidence") or ""
        ),
        "sha256": str(
            energy.get("full_system_current_scale_calibration_sha256") or ""
        ).strip().lower(),
    }
    return projection


def _prompt_operator(
    callback: OperatorPromptCallback | None,
    request: Mapping[str, Any],
    *,
    cancellation_is_error: bool = True,
) -> dict[str, Any]:
    if callback is None:
        raise PlatformConfigurationError(
            "Guided full-system calibration requires an operator prompt callback"
        )
    response = callback(dict(request))
    if response is None:
        if cancellation_is_error:
            raise PlatformCalibrationCancelled(
                f"Calibration cancelled at {request.get('step_id') or 'operator prompt'}"
            )
        return {}
    if not isinstance(response, Mapping):
        raise PlatformConfigurationError(
            "Operator prompt callback must return a mapping or None"
        )
    parsed = dict(response)
    if cancellation_is_error and (
        parsed.get("cancelled") is True or parsed.get("confirmed") is not True
    ):
        raise PlatformCalibrationCancelled(
            f"Calibration cancelled at {request.get('step_id') or 'operator prompt'}"
        )
    return parsed


def _operator_positive_number(
    response: Mapping[str, Any],
    key: str,
    *,
    minimum: float,
    maximum: float,
) -> float:
    value = response.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PlatformConfigurationError(
            f"Operator value {key} must be a finite number"
        )
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < minimum or parsed > maximum:
        raise PlatformConfigurationError(
            f"Operator value {key} must be in range {minimum:g}..{maximum:g}"
        )
    return parsed


def _capture_full_system_calibration_power(
    setup_id: str,
    *,
    registry: Mapping[str, Any],
    setup: Mapping[str, Any],
    cfg: Mapping[str, Any],
    state_label: str,
    duration_s: float,
    output_dir: Path,
    callback: ProgressCallback | None,
    measurement_runner: Callable[..., dict[str, Any]] | None,
) -> dict[str, Any]:
    started_at = _now_iso()
    started_monotonic = time.monotonic()
    try:
        power_w, measurement = measure_idle_power(
            setup_id,
            registry=registry,
            setup=setup,
            cfg=cfg,
            state_label=state_label,
            duration_s=duration_s,
            output_dir=output_dir,
            callback=callback,
            measurement_runner=measurement_runner,
            bypass_full_system_current_scale=True,
            claim_exclusion_reason="full_system_input_scale_calibration",
            run_id_prefix="full_system_input_scale_calibration",
            # The collector may repeat one acquisition-integrity failure.  It
            # still applies the same physical-data and final-energy gates; the
            # retry only replaces the old exact-run suppression.
            exact_run_count=False,
            invalid_repeat_max_retries=1,
            require_command_window_alignment=True,
        )
    except BaseException as exc:
        finished_monotonic = time.monotonic()
        failed_capture = {
            "state_label": state_label,
            "status": "failed",
            "started_at": started_at,
            "finished_at": _now_iso(),
            "capture_center_monotonic_s": (
                started_monotonic + finished_monotonic
            )
            / 2.0,
            "output_dir": str(output_dir.resolve()),
            "error_type": type(exc).__name__,
            "error": f"{type(exc).__name__}: {exc}",
        }
        raw_measurement = getattr(exc, "measurement_result", None)
        if isinstance(raw_measurement, Mapping):
            failed_capture["measurement"] = dict(raw_measurement)
        try:
            setattr(exc, "calibration_capture", failed_capture)
        except Exception:
            pass
        raise
    finished_monotonic = time.monotonic()
    return {
        "state_label": state_label,
        "status": "ok",
        "started_at": started_at,
        "finished_at": _now_iso(),
        "capture_center_monotonic_s": (
            started_monotonic + finished_monotonic
        ) / 2.0,
        "output_dir": str(output_dir.resolve()),
        "avg_power_w": float(power_w),
        "measurement": measurement,
    }


def _restore_full_system_calibration_start_state(
    setup_id: str,
    *,
    initial_status: PlatformStatus,
    registry: Mapping[str, Any],
    setup: Mapping[str, Any],
    cfg: Mapping[str, Any],
    callback: ProgressCallback | None,
    transport_factory: Callable[[HostConfig], SSHTransport],
    udp_preflight: Mapping[str, Any] | None,
) -> tuple[PlatformStatus, list[dict[str, Any]]]:
    """Restore only the Jetson state observed and authorized at entry.

    Full-system input-gain calibration never controls the M.2 rail.  When the
    operator explicitly admitted an already SSH-down Jetson, even Jetson power
    control remains untouched: an unexpected later SSH-up observation is
    reported as external state drift rather than answered with a blind toggle.
    """

    events: list[dict[str, Any]] = []
    intended_jetson_ready = initial_status.jetson_ssh_ready
    if intended_jetson_ready not in {True, False}:
        raise PlatformStateError(
            "Cannot restore calibration state because the initial Jetson SSH "
            "observation was unknown"
        )
    current = probe_platform_status(
        setup_id, registry=registry, transport_factory=transport_factory
    )
    if current.jetson_ssh_ready is None:
        raise PlatformStateError(
            "Cannot restore calibration state because Jetson SSH is unknown"
        )

    if intended_jetson_ready is False:
        if current.jetson_ssh_ready is not False:
            raise PlatformStateError(
                "Jetson became SSH-ready during an already-off calibration; "
                "refusing an unrequested hardware toggle"
            )
        return current, events

    if current.jetson_ssh_ready is False:
        _emit(callback, "Restoring the initially SSH-ready Jetson …")
        restore_jetson = _set_jetson_state_locked(
            setup_id,
            True,
            expected_ssh_ready=False,
            registry=registry,
            setup=setup,
            cfg=cfg,
            callback=callback,
            transport_factory=transport_factory,
            initial_status=current,
            udp_preflight=udp_preflight,
        )
        events.append({"transition": "restore_jetson_on", "result": restore_jetson})
        current = probe_platform_status(
            setup_id, registry=registry, transport_factory=transport_factory
        )
    if current.jetson_ssh_ready is not True:
        raise PlatformStateError(
            "Jetson was not restored to its initial SSH-ready state"
        )

    # No M.2 action is authorized in this workflow.  If both observations are
    # available, flag external/driver state drift rather than trying to repair
    # it through the toggle-only controller.
    if (
        initial_status.m2_present is not None
        and current.m2_present is not None
        and current.m2_present is not initial_status.m2_present
    ):
        raise PlatformStateError(
            "M.2 presence changed during full-system calibration; no M.2 "
            "hardware action was performed"
        )
    return current, events


def _merge_full_system_current_scale_fields(
    registry: MutableMapping[str, Any],
    setup_id: str,
    factor: float,
    *,
    calibration_record: Mapping[str, Any],
) -> bool:
    """Merge the verified FS factor and invalidate old-scale idle baselines."""

    setups = list(registry.get("hardware_setups") or [])
    for raw in setups:
        if not isinstance(raw, MutableMapping):
            continue
        if str(raw.get("id") or "") != str(setup_id):
            continue
        energy = (
            dict(raw.get("energy") or {})
            if isinstance(raw.get("energy"), Mapping)
            else {}
        )
        invalidated = any(
            energy.get(key) not in {None, ""}
            for key in (
                "idle_baseline_w",
                "accelerator_idle_w",
                "accelerator_idle_calibrated_at",
                "accelerator_idle_calibration_evidence",
                "accelerator_idle_calibration_binding_path",
                "accelerator_idle_calibration_binding_sha256",
            )
        )
        energy["full_system_current_scale_factor"] = float(factor)
        energy["full_system_current_scale_calibrated_at"] = str(
            calibration_record.get("finished_at") or _now_iso()
        )
        energy["full_system_current_scale_calibration_evidence"] = str(
            calibration_record.get("evidence_path") or ""
        )
        energy["full_system_current_scale_calibration_sha256"] = str(
            calibration_record.get("evidence_sha256") or ""
        ).strip().lower()
        # Values measured before this absolute-scale correction cannot safely
        # be mixed with corrected captures.  Force a fresh idle calibration.
        energy["idle_baseline_w"] = None
        energy["accelerator_idle_w"] = None
        energy["accelerator_idle_calibrated_at"] = ""
        energy["accelerator_idle_calibration_evidence"] = ""
        energy["accelerator_idle_calibration_binding_path"] = ""
        energy["accelerator_idle_calibration_binding_sha256"] = ""
        raw["energy"] = energy
        registry["hardware_setups"] = setups
        return invalidated
    raise PlatformConfigurationError(
        f"Cannot save full-system calibration; setup disappeared: {setup_id}"
    )


def _save_full_system_current_scale(
    setup_id: str,
    factor: float,
    *,
    registry_path: str | Path | None,
    calibration_record: Mapping[str, Any],
    expected_configuration_projection: Mapping[str, Any],
) -> bool:
    path = energy_config._expand(  # type: ignore[attr-defined]
        registry_path or energy_config.default_registry_path()
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with energy_config._hardware_registry_write_lock(path):  # type: ignore[attr-defined]
        fresh_registry = load_hardware_registry(path)
        fresh_projection = _full_system_calibration_configuration_projection(
            fresh_registry, setup_id
        )
        if dict(fresh_projection) != dict(expected_configuration_projection):
            raise PlatformStateError("configuration_changed_during_calibration")
        invalidated = _merge_full_system_current_scale_fields(
            fresh_registry,
            setup_id,
            factor,
            calibration_record=calibration_record,
        )
        energy_config._write_hardware_registry_atomic(  # type: ignore[attr-defined]
            fresh_registry, path
        )
        return invalidated


def _full_system_calibration_analysis(
    idle_before: Mapping[str, Any],
    loaded_captures: list[dict[str, Any]],
    idle_after: Mapping[str, Any],
    *,
    idle_between: Mapping[str, Any] | None = None,
    power_control: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fit the two load increments and evaluate the calibration quality gate."""

    if len(loaded_captures) != len(FULL_SYSTEM_CALIBRATION_TARGET_CURRENTS_A):
        raise PlatformStateError(
            "Full-system calibration requires exactly the 0.5 A and 1.0 A captures"
        )
    power = dict(DEFAULT_POWER_CONTROL)
    power.update(dict(power_control or {}))
    before_power = float(idle_before["avg_power_w"])
    after_power = float(idle_after["avg_power_w"])
    between_power = (
        float(idle_between["avg_power_w"])
        if idle_between is not None
        else None
    )
    idle_powers = [before_power, after_power]
    if between_power is not None:
        idle_powers.insert(1, between_power)
    if any(not math.isfinite(value) or value <= 0.0 for value in idle_powers):
        raise PlatformStateError("Idle captures must contain positive finite power")

    point_results: list[dict[str, Any]] = []
    before_time = float(idle_before.get("capture_center_monotonic_s", 0.0))
    after_time = float(idle_after.get("capture_center_monotonic_s", before_time))
    span = after_time - before_time
    for index, capture in enumerate(loaded_captures):
        target_current = float(FULL_SYSTEM_CALIBRATION_TARGET_CURRENTS_A[index])
        actual_target = float(capture["target_current_a"])
        if not math.isclose(actual_target, target_current, rel_tol=0.0, abs_tol=1e-9):
            raise PlatformConfigurationError(
                "Full-system calibration target-current order is invalid"
            )
        if between_power is not None:
            adjacent = (
                (before_power, between_power)
                if index == 0
                else (between_power, after_power)
            )
            idle_baseline = sum(adjacent) / 2.0
            baseline_method = "mean_of_adjacent_idle_windows"
            position: float | None = None
        else:
            capture_time = float(
                capture.get("capture_center_monotonic_s", before_time)
            )
            if span > 0.0:
                position = min(
                    1.0, max(0.0, (capture_time - before_time) / span)
                )
                idle_baseline = before_power + (
                    after_power - before_power
                ) * position
            else:
                position = 0.5
                idle_baseline = (before_power + after_power) / 2.0
            baseline_method = "linear_idle_interpolation_compatibility"
        measured_delta = float(capture["avg_power_w"]) - idle_baseline
        if not math.isfinite(measured_delta) or measured_delta <= 0.0:
            raise PlatformStateError(
                "Loaded full-system capture did not produce a positive power increment"
            )
        current_a = float(capture["reference_current_a"])
        voltage_v = float(capture["reference_voltage_v"])
        reference_power = current_a * voltage_v
        if (
            not math.isfinite(current_a)
            or not math.isfinite(voltage_v)
            or not math.isfinite(reference_power)
            or current_a <= 0.0
            or voltage_v <= 0.0
        ):
            raise PlatformConfigurationError(
                "Reference load current and voltage must produce positive finite power"
            )
        current_error_pct = abs(current_a - target_current) / target_current * 100.0
        point_results.append(
            {
                "target_current_a": target_current,
                "reference_current_a": current_a,
                "reference_voltage_v": voltage_v,
                "reference_power_w": reference_power,
                "reference_current_error_pct": current_error_pct,
                "measured_total_power_w": float(capture["avg_power_w"]),
                "idle_baseline_power_w": idle_baseline,
                "idle_baseline_method": baseline_method,
                "idle_interpolation_position": position,
                "measured_increment_w": measured_delta,
                "point_scale_factor": reference_power / measured_delta,
            }
        )

    denominator = sum(
        float(point["measured_increment_w"]) ** 2 for point in point_results
    )
    if denominator <= 0.0:
        raise PlatformStateError("Full-system calibration fit is singular")
    factor = sum(
        float(point["measured_increment_w"])
        * float(point["reference_power_w"])
        for point in point_results
    ) / denominator
    if (
        not math.isfinite(factor)
        or factor < FULL_SYSTEM_CALIBRATION_HARD_FACTOR_MIN
        or factor > FULL_SYSTEM_CALIBRATION_HARD_FACTOR_MAX
    ):
        raise PlatformStateError(
            "Calculated full-system scale factor is outside the hard plausibility "
            f"range {FULL_SYSTEM_CALIBRATION_HARD_FACTOR_MIN:g}.."
            f"{FULL_SYSTEM_CALIBRATION_HARD_FACTOR_MAX:g}: {factor:.6f}"
        )
    point_factors = [float(point["point_scale_factor"]) for point in point_results]
    mean_factor = sum(point_factors) / len(point_factors)
    point_spread_pct = (
        (max(point_factors) - min(point_factors)) / abs(mean_factor) * 100.0
        if mean_factor != 0.0
        else math.inf
    )
    residuals: list[float] = []
    for point in point_results:
        reference = float(point["reference_power_w"])
        corrected = factor * float(point["measured_increment_w"])
        residual_pct = abs(corrected - reference) / reference * 100.0
        point["fitted_corrected_increment_w"] = corrected
        point["fit_residual_pct"] = residual_pct
        residuals.append(residual_pct)
    max_fit_residual_pct = max(residuals) if residuals else math.inf
    idle_drift_w = max(idle_powers) - min(idle_powers)

    limits = {
        "minimum_delta_w": float(
            power["full_system_calibration_minimum_delta_w"]
        ),
        "max_point_spread_pct": float(
            power["full_system_calibration_max_point_spread_pct"]
        ),
        "max_idle_drift_w": float(
            power["full_system_calibration_max_idle_drift_w"]
        ),
        "min_factor": float(power["full_system_calibration_min_factor"]),
        "max_factor": float(power["full_system_calibration_max_factor"]),
        "reference_current_tolerance_pct": float(
            power["full_system_calibration_reference_current_tolerance_pct"]
        ),
    }
    # Separate conditions which make the fitted factor technically unusable
    # from diagnostics which merely describe the measured calibration curve.
    # In particular, two finite positive load points do not become invalid just
    # because their independently derived point factors narrowly exceed a
    # configured plausibility threshold. Those observations remain visible in
    # the evidence, but do not prevent saving the origin-constrained fit.
    reasons: list[str] = []
    warning_reasons: list[str] = []
    if any(
        float(point["measured_increment_w"]) < limits["minimum_delta_w"]
        for point in point_results
    ):
        reasons.append("measured_increment_below_minimum")
    if point_spread_pct > limits["max_point_spread_pct"]:
        warning_reasons.append("point_factor_spread_too_high")
    if idle_drift_w > limits["max_idle_drift_w"]:
        reasons.append("idle_drift_too_high")
    if not limits["min_factor"] <= factor <= limits["max_factor"]:
        warning_reasons.append("scale_factor_outside_configured_range")
    if any(
        float(point["reference_current_error_pct"])
        > limits["reference_current_tolerance_pct"]
        for point in point_results
    ):
        warning_reasons.append("reference_current_outside_target_tolerance")

    messages = {
        "measured_increment_below_minimum": (
            "At least one measured load increment is below the configured minimum."
        ),
        "point_factor_spread_too_high": (
            f"The two point factors differ by {point_spread_pct:.3f}% "
            f"(limit {limits['max_point_spread_pct']:.3f}%)."
        ),
        "idle_drift_too_high": (
            f"Idle drift is {idle_drift_w:.3f} W "
            f"(limit {limits['max_idle_drift_w']:.3f} W)."
        ),
        "scale_factor_outside_configured_range": (
            f"The fitted factor {factor:.6f} is outside "
            f"{limits['min_factor']:.6f}..{limits['max_factor']:.6f}."
        ),
        "reference_current_outside_target_tolerance": (
            "The electronic-load current readback differs too much from its target."
        ),
    }
    warnings = tuple(messages[reason] for reason in warning_reasons)
    return {
        "model": FULL_SYSTEM_CURRENT_SCALE_MODEL,
        "scale_factor": float(factor),
        "point_results": tuple(point_results),
        "point_factors": tuple(point_factors),
        "point_spread_pct": float(point_spread_pct),
        "max_fit_residual_pct": float(max_fit_residual_pct),
        "idle_powers_w": tuple(idle_powers),
        "idle_drift_w": float(idle_drift_w),
        "warnings": warnings,
        "quality_gate": {
            "pass": not reasons,
            "reasons": reasons,
            "warning_reasons": warning_reasons,
            "limits": limits,
        },
    }


def _full_system_calibration_fit(
    idle_before: Mapping[str, Any],
    loaded_captures: list[dict[str, Any]],
    idle_after: Mapping[str, Any],
    *,
    idle_between: Mapping[str, Any] | None = None,
    power_control: Mapping[str, Any] | None = None,
) -> tuple[float, tuple[dict[str, Any], ...], float, float, tuple[str, ...]]:
    """Compatibility projection for tests/readers of the original helper."""

    analysis = _full_system_calibration_analysis(
        idle_before,
        loaded_captures,
        idle_after,
        idle_between=idle_between,
        power_control=power_control,
    )
    return (
        float(analysis["scale_factor"]),
        tuple(analysis["point_results"]),
        float(analysis["point_spread_pct"]),
        float(analysis["max_fit_residual_pct"]),
        tuple(analysis["warnings"]),
    )


def calibrate_full_system_input_scale(
    setup_id: str,
    *,
    registry_path: str | Path | None = None,
    stabilize_s: float | None = None,
    measure_s: float | None = None,
    load_settle_s: float | None = None,
    output_dir: str | Path | None = None,
    confirm_jetson_already_off: bool = False,
    force_active_workflow: bool = False,
    callback: ProgressCallback | None = None,
    operator_prompt: OperatorPromptCallback | None = None,
    transport_factory: Callable[[HostConfig], SSHTransport] = SSHTransport,
    measurement_runner: Callable[..., dict[str, Any]] | None = None,
) -> FullSystemInputCalibrationResult:
    """Guide and persist a verified two-point DC gain trim for u.RECS FS power.

    The electronic load is connected as a sink from ``9V_20V_IN`` on the load
    side of R16 to GND.  An SSH-ready Jetson is shut down once and restored once;
    an already SSH-down Jetson is accepted only with the caller's explicit
    ``confirm_jetson_already_off`` authorization and remains untouched.  The M.2
    rail is never toggled.  For each 0.5 A/1.0 A load point the routine records
    adjacent idle windows, computes ``reference increment / measured increment``,
    restores the initial Jetson state, and only then offers the factor for saving.
    """

    _require_exact_bool_argument(
        force_active_workflow, field="force_active_workflow"
    )
    _require_exact_bool_argument(
        confirm_jetson_already_off,
        field="confirm_jetson_already_off",
    )
    registry_path = _strict_registry_path_argument(registry_path)
    stabilize_override = (
        None
        if stabilize_s is None
        else _strict_power_control_number(
            stabilize_s,
            field="stabilize_s",
            minimum=0.0,
            maximum=None,
            integer=False,
        )
    )
    measure_override = (
        None
        if measure_s is None
        else _strict_power_control_number(
            measure_s,
            field="measure_s",
            minimum=5.0,
            maximum=None,
            integer=False,
        )
    )
    load_settle_override = (
        None
        if load_settle_s is None
        else _strict_power_control_number(
            load_settle_s,
            field="load_settle_s",
            minimum=0.0,
            maximum=60.0,
            integer=False,
        )
    )

    started_at = _now_iso()
    registry, setup, cfg = resolve_setup(setup_id, registry_path=registry_path)
    projection = _full_system_calibration_configuration_projection(
        registry, setup_id
    )
    power = cfg["power_control"]
    # An already-off calibration performs no rail action and therefore must
    # not depend on M.2/Jetson toggle-command validation. Jetson-control
    # preflight is deferred until the initial probe proves that shutdown is
    # actually required.
    udp_preflight: Mapping[str, Any] | None = None
    stabilize = (
        float(power["calibration_stabilize_s"])
        if stabilize_override is None
        else float(stabilize_override)
    )
    measure = (
        float(power["calibration_measure_s"])
        if measure_override is None
        else float(measure_override)
    )
    load_settle = (
        float(power["full_system_calibration_load_settle_s"])
        if load_settle_override is None
        else float(load_settle_override)
    )
    stamp = time.strftime("%Y%m%d_%H%M%S")
    root = (
        Path(output_dir)
        if output_dir is not None
        else energy_measurements_root()
        / "Calibrations"
        / setup_id
        / f"{stamp}_full_system_input_scale"
    )
    root = _prepare_calibration_output_root(root)
    operational_path = root / "full_system_input_scale_calibration_operational.json"
    final_evidence_path = root / "full_system_input_scale_calibration.json"
    operational: dict[str, Any] = {
        "schema": "onnx-splitpoint/full-system-input-scale-calibration-operational",
        "schema_version": 1,
        "setup_id": setup_id,
        "accelerator": str(setup.get("accelerator") or ""),
        "started_at": started_at,
        "status": "running",
        "saved": False,
        "configuration_projection": projection,
        "events": [],
        "captures": {},
        "recovery": {},
        "m2_untouched": True,
        "initial_state_restored": False,
    }

    def checkpoint() -> None:
        tmp = operational_path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(operational, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(tmp, operational_path)

    def capture_state(label: str) -> PlatformStatus:
        status = probe_platform_status(
            setup_id, registry=registry, transport_factory=transport_factory
        )
        _require_calibration_urecs_ready(status, power)
        if status.jetson_ssh_ready is not False:
            raise PlatformStateError(
                f"Full-system calibration {label}: Jetson must remain SSH-not-ready"
            )
        return status

    def capture_power(
        label: str,
        *,
        reference: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        capture_state(f"before {label}")
        reference_fields = dict(reference or {})
        operational["captures"][label] = {
            "state_label": label,
            "status": "running",
            "output_dir": str((root / label).resolve()),
            **reference_fields,
        }
        checkpoint()
        try:
            capture = _capture_full_system_calibration_power(
                setup_id,
                registry=registry,
                setup=setup,
                cfg=cfg,
                state_label=label,
                duration_s=measure,
                output_dir=root / label,
                callback=callback,
                measurement_runner=measurement_runner,
            )
        except BaseException as exc:
            failed = getattr(exc, "calibration_capture", None)
            failed_capture = (
                dict(failed)
                if isinstance(failed, Mapping)
                else {
                    "state_label": label,
                    "status": "failed",
                    "output_dir": str((root / label).resolve()),
                    "error_type": type(exc).__name__,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            failed_capture.update(reference_fields)
            operational["captures"][label] = failed_capture
            checkpoint()
            raise
        capture.update(reference_fields)
        capture_state(f"after {label}")
        operational["captures"][label] = capture
        checkpoint()
        return capture

    checkpoint()
    preflight_completed = False
    platform_mutated = False
    zero_load_confirmed = False
    restored_jetson = False
    restored_m2 = False
    initial_state_restored = False
    restoration_attempted = False
    initial: PlatformStatus | None = None
    final_evidence_sha256 = ""
    final_evidence: dict[str, Any] = {}
    invalidated_idle_baselines = False

    with platform_operation_lock(
        setup_id, force_active_workflow=force_active_workflow
    ):
        try:
            locked_registry = load_hardware_registry(registry_path)
            locked_projection = _full_system_calibration_configuration_projection(
                locked_registry, setup_id
            )
            if locked_projection != projection:
                raise PlatformStateError(
                    "configuration_changed_while_waiting_for_platform_lock"
                )
            registry, setup, cfg = resolve_setup(
                setup_id, registry=locked_registry
            )
            power = cfg["power_control"]
            _verify_locked_calibration_output_root(
                root, operational_evidence_path=operational_path
            )
            initial = probe_platform_status(
                setup_id, registry=registry, transport_factory=transport_factory
            )
            operational["initial_status"] = initial.to_dict()
            operational["start_state_policy"] = {
                "jetson_ssh_ready": initial.jetson_ssh_ready,
                "jetson_already_off_explicitly_confirmed": bool(
                    confirm_jetson_already_off
                ),
                "m2_present_observation": initial.m2_present,
                "m2_control": "untouched",
            }
            checkpoint()
            _require_calibration_urecs_ready(initial, power)
            if initial.jetson_ssh_ready is None:
                raise PlatformStateError(
                    "ABORT_NO_MUTATION: Full-system calibration cannot distinguish "
                    "the Jetson state because the SSH observation is unknown"
                )
            if (
                initial.jetson_ssh_ready is False
                and confirm_jetson_already_off is not True
            ):
                raise PlatformStateError(
                    "ABORT_NO_MUTATION: Jetson SSH is down, but SSH unreachability "
                    "does not prove that the power rail is off; explicitly confirm "
                    "that the Jetson was intentionally powered off"
                )
            if initial.jetson_ssh_ready is True:
                udp_preflight = _validate_platform_udp_configuration(setup, cfg)
                _require_power_control_enabled(power)

            _prompt_operator(
                operator_prompt,
                {
                    "kind": "confirm",
                    "step_id": "preflight",
                    "step_index": 1,
                    "step_count": 7,
                    "title": "Prepare full-system calibration",
                    "message": (
                        "Connect the electronic load as a sink from 9V_20V_IN "
                        "on the load side of R16 to GND. Keep its output OFF / "
                        "at 0 A. M.2 remains untouched. "
                        + (
                            "The already-off Jetson remains off without a power "
                            "command."
                            if initial.jetson_ssh_ready is False
                            else "The Jetson will be shut down once for the captures."
                        )
                    ),
                    "confirm_label": "Load connected and OFF — continue",
                },
            )
            zero_load_confirmed = True
            preflight_completed = True
            operational["preflight"] = "passed"
            checkpoint()

            if initial.jetson_ssh_ready is True:
                _emit(callback, "Turning the Jetson off for load-step captures …")
                # A post-send verification failure can occur after the
                # toggle-only controller consumed the command.  Mark mutation
                # possible before entering the helper so recovery is attempted.
                platform_mutated = True
                jetson_off = _set_jetson_state_locked(
                    setup_id,
                    False,
                    expected_ssh_ready=True,
                    registry=registry,
                    setup=setup,
                    cfg=cfg,
                    callback=callback,
                    transport_factory=transport_factory,
                    initial_status=initial,
                    udp_preflight=udp_preflight,
                )
                operational["events"].append(
                    {"transition": "jetson_off", "result": jetson_off}
                )
                checkpoint()
                capture_state("after automatic shutdown")
            else:
                operational["events"].append(
                    {
                        "transition": "jetson_already_off_confirmed",
                        "hardware_action_performed": False,
                    }
                )
                checkpoint()
                capture_state("after explicit already-off confirmation")
            if stabilize > 0.0:
                _emit(
                    callback,
                    f"Stabilizing Jetson-off state for {stabilize:.0f}s …",
                )
                time.sleep(stabilize)

            _prompt_operator(
                operator_prompt,
                {
                    "kind": "zero_load",
                    "step_id": "idle_before",
                    "step_index": 2,
                    "step_count": 7,
                    "title": "Idle before 0.5 A",
                    "message": (
                        "Set the electronic load output OFF / to 0 A and wait "
                        "until its display is stable."
                    ),
                    "confirm_label": "0 A confirmed — measure idle",
                },
            )
            zero_load_confirmed = True
            idle_before = capture_power("idle_before")

            # From this point the operator may already have enabled the load.
            # If the values dialog is cancelled or validation fails, recovery
            # must explicitly confirm 0 A before the initial state is restored.
            zero_load_confirmed = False
            response_05 = _prompt_operator(
                operator_prompt,
                {
                    "kind": "load_step",
                    "step_id": "load_0.5A",
                    "step_index": 3,
                    "step_count": 7,
                    "title": "Apply 0.5 A load",
                    "message": (
                        "Enable constant-current mode at 0.5 A. Enter the actual "
                        "current and voltage shown by the electronic load."
                    ),
                    "target_current_a": 0.5,
                    "default_current_a": 0.5,
                    "default_voltage_v": FULL_SYSTEM_CALIBRATION_DEFAULT_VOLTAGE_V,
                    "confirm_label": "Values stable — measure 0.5 A",
                },
            )
            current_05 = _operator_positive_number(
                response_05, "current_a", minimum=0.05, maximum=5.0
            )
            voltage_05 = _operator_positive_number(
                response_05, "voltage_v", minimum=8.0, maximum=21.0
            )
            if load_settle > 0.0:
                _emit(callback, f"Waiting {load_settle:.0f}s for 0.5 A to settle …")
                time.sleep(load_settle)
            load_05 = capture_power(
                "load_0.5A",
                reference={
                    "target_current_a": 0.5,
                    "reference_current_a": current_05,
                    "reference_voltage_v": voltage_05,
                    "reference_power_w": current_05 * voltage_05,
                },
            )

            _prompt_operator(
                operator_prompt,
                {
                    "kind": "zero_load",
                    "step_id": "idle_between",
                    "step_index": 4,
                    "step_count": 7,
                    "title": "Idle after 0.5 A / before 1.0 A",
                    "message": (
                        "Switch the electronic load output OFF / back to 0 A. "
                        "This idle window closes the 0.5 A point and opens the "
                        "1.0 A point."
                    ),
                    "confirm_label": "0 A confirmed — measure middle idle",
                },
            )
            zero_load_confirmed = True
            if load_settle > 0.0:
                time.sleep(load_settle)
            idle_between = capture_power("idle_between")

            # As above, entering the load-step dialog means that a non-zero
            # load is possible even if no valid response is returned.
            zero_load_confirmed = False
            response_10 = _prompt_operator(
                operator_prompt,
                {
                    "kind": "load_step",
                    "step_id": "load_1A",
                    "step_index": 5,
                    "step_count": 7,
                    "title": "Apply 1.0 A load",
                    "message": (
                        "Enable constant-current mode at 1.0 A. Enter the actual "
                        "current and voltage shown by the electronic load."
                    ),
                    "target_current_a": 1.0,
                    "default_current_a": 1.0,
                    "default_voltage_v": voltage_05,
                    "confirm_label": "Values stable — measure 1.0 A",
                },
            )
            current_10 = _operator_positive_number(
                response_10, "current_a", minimum=0.05, maximum=5.0
            )
            voltage_10 = _operator_positive_number(
                response_10, "voltage_v", minimum=8.0, maximum=21.0
            )
            if load_settle > 0.0:
                _emit(callback, f"Waiting {load_settle:.0f}s for 1.0 A to settle …")
                time.sleep(load_settle)
            load_10 = capture_power(
                "load_1A",
                reference={
                    "target_current_a": 1.0,
                    "reference_current_a": current_10,
                    "reference_voltage_v": voltage_10,
                    "reference_power_w": current_10 * voltage_10,
                },
            )

            _prompt_operator(
                operator_prompt,
                {
                    "kind": "zero_load",
                    "step_id": "idle_after",
                    "step_index": 6,
                    "step_count": 7,
                    "title": "Final idle after 1.0 A",
                    "message": (
                        "Switch the electronic load output OFF / back to 0 A. "
                        "The final idle window completes the calibration."
                    ),
                    "confirm_label": "0 A confirmed — measure final idle",
                },
            )
            zero_load_confirmed = True
            if load_settle > 0.0:
                time.sleep(load_settle)
            idle_after = capture_power("idle_after")

            loaded_captures = [load_05, load_10]
            analysis = _full_system_calibration_analysis(
                idle_before,
                loaded_captures,
                idle_after,
                idle_between=idle_between,
                power_control=power,
            )
            factor = float(analysis["scale_factor"])
            point_results = tuple(analysis["point_results"])
            quality_gate = dict(analysis["quality_gate"])
            operational["fit"] = {
                key: value
                for key, value in analysis.items()
                if key not in {"quality_gate"}
            }
            operational["quality_gate"] = quality_gate
            checkpoint()

            _emit(
                callback,
                "Restoring the initial Jetson state; M.2 stays untouched …",
            )
            if initial is None:
                raise PlatformStateError("Initial calibration state was not recorded")
            # The controller is toggle-only. Once restoration begins, an SSH
            # timeout cannot distinguish "command not consumed" from "Jetson
            # still booting after command". Mark the attempt first so failure
            # handling never sends a second, potentially reversing toggle.
            restoration_attempted = True
            operational["restoration_attempted"] = True
            checkpoint()
            final_status, restore_events = (
                _restore_full_system_calibration_start_state(
                    setup_id,
                    initial_status=initial,
                    registry=registry,
                    setup=setup,
                    cfg=cfg,
                    callback=callback,
                    transport_factory=transport_factory,
                    udp_preflight=udp_preflight,
                )
            )
            operational["events"].extend(restore_events)
            operational["final_status"] = final_status.to_dict()
            restored_jetson = final_status.jetson_ssh_ready is True
            restored_m2 = final_status.m2_present is True
            initial_state_restored = (
                final_status.jetson_ssh_ready is initial.jetson_ssh_ready
            )
            operational["initial_state_restored"] = initial_state_restored
            checkpoint()
            if not initial_state_restored:
                raise PlatformStateError(
                    "Full-system calibration did not restore the initial Jetson state"
                )

            review = _prompt_operator(
                operator_prompt,
                {
                    "kind": "review",
                    "step_id": "review",
                    "step_index": 7,
                    "step_count": 7,
                    "title": "Review full-system calibration",
                    "message": (
                        "The initial Jetson state is restored and M.2 was not "
                        "toggled. Saving writes the verified "
                        "factor for this setup and clears old idle baselines, "
                        "which must then be calibrated again."
                    ),
                    "scale_factor": factor,
                    "point_results": list(point_results),
                    "point_spread_pct": float(analysis["point_spread_pct"]),
                    "max_fit_residual_pct": float(
                        analysis["max_fit_residual_pct"]
                    ),
                    "idle_drift_w": float(analysis["idle_drift_w"]),
                    "quality_passed": quality_gate.get("pass") is True,
                    "quality_reasons": list(quality_gate.get("reasons") or []),
                    "warnings": list(analysis["warnings"]),
                    "confirm_label": "Save calibration",
                    "decline_label": "Finish without saving",
                },
                cancellation_is_error=False,
            )
            save_requested = bool(review.get("save") is True)
            if save_requested and quality_gate.get("pass") is not True:
                raise PlatformStateError(
                    "Full-system calibration quality gate failed: "
                    + ",".join(str(value) for value in quality_gate.get("reasons") or [])
                )

            finished_at = _now_iso()
            setup_energy = energy_setup_from_registry(registry, setup_id)
            final_evidence = {
                "schema": FULL_SYSTEM_CURRENT_SCALE_SCHEMA,
                "schema_version": FULL_SYSTEM_CURRENT_SCALE_SCHEMA_VERSION,
                "status": "ok",
                "save_requested": save_requested,
                "setup_id": setup_id,
                "accelerator": str(setup.get("accelerator") or ""),
                "physical_scope": "FS",
                "measurement_scope": "full_system_input",
                "load_connection": FULL_SYSTEM_CURRENT_SCALE_LOAD_CONNECTION,
                "setup_binding": {
                    "urecs_address": str(setup_energy.urecs_address),
                    "data_port": int(setup_energy.data_port),
                },
                "started_at": started_at,
                "finished_at": finished_at,
                "stabilize_s": stabilize,
                "measure_s": measure,
                "load_settle_s": load_settle,
                "target_currents_a": list(
                    FULL_SYSTEM_CALIBRATION_TARGET_CURRENTS_A
                ),
                "configuration_projection": projection,
                "captures": {
                    "idle_before": idle_before,
                    "load_0.5A": load_05,
                    "idle_between": idle_between,
                    "load_1A": load_10,
                    "idle_after": idle_after,
                },
                "fit": {
                    "model": FULL_SYSTEM_CURRENT_SCALE_MODEL,
                    "scale_factor": factor,
                    "point_results": list(point_results),
                    "point_factors": list(analysis["point_factors"]),
                    "point_spread_pct": float(analysis["point_spread_pct"]),
                    "max_fit_residual_pct": float(
                        analysis["max_fit_residual_pct"]
                    ),
                    "idle_powers_w": list(analysis["idle_powers_w"]),
                    "idle_drift_w": float(analysis["idle_drift_w"]),
                },
                "quality_gate": quality_gate,
                "warnings": list(analysis["warnings"]),
                "restoration": {
                    "jetson_ssh_ready": restored_jetson,
                    "m2_present": restored_m2,
                    "initial_jetson_ssh_ready": initial.jetson_ssh_ready,
                    "final_jetson_ssh_ready": final_status.jetson_ssh_ready,
                    "initial_m2_present": initial.m2_present,
                    "final_m2_present": final_status.m2_present,
                    "initial_state_restored": initial_state_restored,
                    "jetson_initial_state_restored": initial_state_restored,
                    "m2_untouched": True,
                    "final_status": final_status.to_dict(),
                },
            }
            # Immutable evidence: never overwrite a pre-existing final record.
            with final_evidence_path.open("x", encoding="utf-8") as handle:
                json.dump(final_evidence, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            final_evidence_sha256 = sha256_file(final_evidence_path)

            if save_requested:
                invalidated_idle_baselines = _save_full_system_current_scale(
                    setup_id,
                    factor,
                    registry_path=registry_path,
                    calibration_record={
                        "finished_at": finished_at,
                        "evidence_path": str(final_evidence_path),
                        "evidence_sha256": final_evidence_sha256,
                    },
                    expected_configuration_projection=projection,
                )
            operational.update(
                {
                    "finished_at": finished_at,
                    "status": "ok",
                    "saved": save_requested,
                    "quality_gate": quality_gate,
                    "restored_jetson_ready": restored_jetson,
                    "restored_m2_on": restored_m2,
                    "initial_state_restored": initial_state_restored,
                    "m2_untouched": True,
                    "final_evidence_path": str(final_evidence_path),
                    "final_evidence_sha256": final_evidence_sha256,
                    "invalidated_idle_baselines": invalidated_idle_baselines,
                }
            )
            checkpoint()
            _emit(
                callback,
                (
                    f"Full-system calibration saved: factor={factor:.8f}"
                    if save_requested
                    else f"Full-system calibration finished without saving: factor={factor:.8f}"
                ),
            )
            return FullSystemInputCalibrationResult(
                setup_id=setup_id,
                accelerator=str(setup.get("accelerator") or ""),
                scale_factor=factor,
                idle_before_w=float(idle_before["avg_power_w"]),
                idle_after_w=float(idle_after["avg_power_w"]),
                point_results=point_results,
                point_spread_pct=float(analysis["point_spread_pct"]),
                max_fit_residual_pct=float(analysis["max_fit_residual_pct"]),
                warnings=tuple(analysis["warnings"]),
                started_at=started_at,
                finished_at=finished_at,
                output_dir=str(root),
                evidence_path=str(final_evidence_path),
                evidence_sha256=final_evidence_sha256,
                saved=save_requested,
                quality_passed=quality_gate.get("pass") is True,
                restored_jetson_ready=restored_jetson,
                restored_m2_on=restored_m2,
                invalidated_idle_baselines=invalidated_idle_baselines,
                evidence=final_evidence,
                initial_state_restored=initial_state_restored,
            )
        except BaseException as exc:
            operational["status"] = (
                "cancelled"
                if isinstance(exc, PlatformCalibrationCancelled)
                else "failed"
            )
            operational["error_type"] = type(exc).__name__
            operational["error"] = f"{type(exc).__name__}: {exc}"
            operational["interrupted"] = isinstance(
                exc, (KeyboardInterrupt, SystemExit)
            )
            if not preflight_completed or initial is None:
                operational["recovery_skipped_reason"] = "preflight_not_completed"
            else:
                try:
                    # Load safety is independent of platform mutation.  An
                    # already-off start performs no hardware action, but the
                    # operator may still have enabled 0.5 A or 1.0 A before a
                    # failed/cancelled capture.
                    if not zero_load_confirmed:
                        _prompt_operator(
                            operator_prompt,
                            {
                                "kind": "recovery_zero_load",
                                "step_id": "recovery_zero_load",
                                "step_index": 6,
                                "step_count": 7,
                                "title": "Remove load before recovery",
                                "message": (
                                    "Set the electronic load output OFF / to 0 A. "
                                    "The tool checks/restores the initial Jetson "
                                    "state only after this confirmation. M.2 is "
                                    "never toggled by this calibration."
                                ),
                                "confirm_label": "0 A confirmed — restore initial state",
                            },
                        )
                        zero_load_confirmed = True
                    if restoration_attempted:
                        # A restoration helper may have sent the toggle before
                        # timing out on SSH. Never retry that toggle
                        # automatically: doing so could turn the Jetson off
                        # again. Later failures after a completed restoration
                        # likewise need no second hardware action.
                        recovery_status = (
                            "initial_state_already_restored_no_action"
                            if initial_state_restored
                            else "restore_attempt_ambiguous_manual_verification_required"
                        )
                        operational["recovery"] = {
                            "status": recovery_status,
                            "events": [],
                            "platform_mutated": platform_mutated,
                            "m2_untouched": True,
                            "automated_restore_retry_suppressed": True,
                        }
                        if not initial_state_restored:
                            operational["recovery_skipped_reason"] = (
                                "restore_command_may_have_been_sent_"
                                "manual_verification_required"
                            )
                    else:
                        final_status, restore_events = (
                            _restore_full_system_calibration_start_state(
                                setup_id,
                                initial_status=initial,
                                registry=registry,
                                setup=setup,
                                cfg=cfg,
                                callback=callback,
                                transport_factory=transport_factory,
                                udp_preflight=udp_preflight,
                            )
                        )
                        operational["recovery"] = {
                            "status": "initial_state_restored",
                            "events": restore_events,
                            "final_status": final_status.to_dict(),
                            "platform_mutated": platform_mutated,
                            "m2_untouched": True,
                        }
                        restored_jetson = final_status.jetson_ssh_ready is True
                        restored_m2 = final_status.m2_present is True
                        initial_state_restored = (
                            final_status.jetson_ssh_ready
                            is initial.jetson_ssh_ready
                        )
                except BaseException as recovery_exc:
                    operational["recovery_error"] = (
                        f"{type(recovery_exc).__name__}: {recovery_exc}"
                    )
            operational["restored_jetson_ready"] = restored_jetson
            operational["restored_m2_on"] = restored_m2
            operational["initial_state_restored"] = initial_state_restored
            operational["m2_untouched"] = True
            operational["finished_at"] = _now_iso()
            checkpoint()
            try:
                setattr(exc, "evidence_path", str(operational_path))
                setattr(exc, "evidence_dir", str(root))
            except Exception:
                pass
            if isinstance(exc, PlatformPowerError):
                marker = f"calibration_evidence={operational_path}"
                if marker not in str(exc):
                    remaining = tuple(getattr(exc, "args", ()))[1:]
                    exc.args = (f"{exc}; {marker}", *remaining)
            raise


def registry_setup_choices(
    *, registry_path: str | Path | None = None
) -> list[tuple[str, str]]:
    registry = load_hardware_registry(registry_path)
    choices: list[tuple[str, str]] = []
    for raw in list(registry.get("hardware_setups") or []):
        if not isinstance(raw, Mapping):
            continue
        sid = str(raw.get("id") or "").strip()
        if not sid:
            continue
        label = str(raw.get("label") or sid).strip()
        acc = str(raw.get("accelerator") or "").strip()
        choices.append((sid, f"{label} ({acc})" if acc else label))
    return choices
