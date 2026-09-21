from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping
from contextlib import contextmanager
import copy
import hashlib
import json
import math
import os
import re
import shlex
import stat
import shutil

from onnx_splitpoint_tool.workflow.artifacts import now_iso, sha256_json, write_json

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore


ENERGY_LIFECYCLE_STATES = (
    "requested",
    "configured",
    "scheduled",
    "running",
    "completed",
    "skipped",
    "failed",
)

ENERGY_PRIMARY_METHOD = "command_marker_window"
ENERGY_SHADOW_METHOD = "chapter4_legacy_window"

# The u.RECS firmware command is punctuation-sensitive.  ``m2`` was the
# shipped 2.79.9--2.79.11 default but is not the controller token used by the
# deployed platforms.  Keep both values explicit so registry migration can be
# deliberately narrow and never rewrite an operator-defined command.
PLATFORM_POWER_M2_COMMAND = "m.2"
PLATFORM_POWER_LEGACY_M2_COMMAND = "m2"

# Private, in-memory metadata used by the GUI.  It is deliberately not part of
# the YAML schema and is stripped by the central writer even if a caller passes
# a revisioned payload through unchanged.
HARDWARE_REGISTRY_REVISION_KEY = "__onnx_splitpoint_registry_file_sha256"


class HardwareRegistryConflictError(RuntimeError):
    """A guarded registry save was based on stale exact file bytes."""

    def __init__(self, path: Path, expected_sha256: str, actual_sha256: str) -> None:
        self.path = Path(path)
        self.expected_sha256 = str(expected_sha256)
        self.actual_sha256 = str(actual_sha256)
        expected = self.expected_sha256 or "<missing>"
        actual = self.actual_sha256 or "<missing>"
        super().__init__(
            "Hardware registry changed after it was loaded; stale save rejected. "
            f"Reload and retry: {self.path} "
            f"(expected SHA-256 {expected}, current SHA-256 {actual})"
        )

# Read compatibility for profiles and archived artefacts created by 2.63--2.66.
# These identifiers describe the names used before the paired u.RECS evidence
# was reviewed.  New runs must serialize the role-explicit identifiers above.
ENERGY_AB_BASELINE_METHOD = "chapter4_baseline"
ENERGY_AB_CANDIDATE_METHOD = "candidate_v263"
ENERGY_METHOD_ALIASES = {
    ENERGY_AB_CANDIDATE_METHOD: ENERGY_PRIMARY_METHOD,
    "collector_sample_marker_crop": ENERGY_PRIMARY_METHOD,
    ENERGY_AB_BASELINE_METHOD: ENERGY_SHADOW_METHOD,
    "historical_power_edge_detection_with_estimated_duration": ENERGY_SHADOW_METHOD,
}


@dataclass(frozen=True)
class EnergyABConfig:
    """Paired postprocessing comparison on one immutable raw capture.

    The reviewed command-marker method is the scientific primary.  The method
    used for the Chapter-4 calibration measurements is retained as a same-trace
    sensitivity result and can neither replace nor invalidate the primary.
    """

    enabled: bool = True
    primary_method: str = ENERGY_PRIMARY_METHOD
    shadow_method: str = ENERGY_SHADOW_METHOD
    same_raw_capture: bool = True
    mode: str = "shadow"
    auto_switch: bool = False
    smoke_repeats: int = 3
    include_raw_parquet: bool = True
    strict: bool = True
    requires_picoscope: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EffectiveEnergyState:
    """Serializable lifecycle state with cumulative, unambiguous flags."""

    status: str
    requested: bool
    configured: bool
    scheduled: bool = False
    running: bool = False
    completed: bool = False
    skipped: bool = False
    failed: bool = False
    reason: str = ""

    def __post_init__(self) -> None:
        if self.status not in ENERGY_LIFECYCLE_STATES:
            raise ValueError(f"unsupported energy lifecycle state: {self.status!r}")
        if self.completed and (self.skipped or self.failed):
            raise ValueError("completed energy state cannot also be skipped or failed")
        if self.skipped and self.failed:
            raise ValueError("energy state cannot be skipped and failed")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EnergyDefaults:
    enabled: bool = False
    collector_binary: str = "urecs-data-collector"
    collector_sha256: str = ""
    power_calculations_binary: str = "power_calculations"
    mode: str = "fast_firmware"
    data_port: int = 3000
    channel: int = 0
    sample_rate: int = 2000
    environment: str = "Jetson"
    pre_duration_s: float = 5.0
    post_duration_s: float = 5.0
    duration_margin_s: float = 1.0
    # Capture reserve for interpreter/SSH/runtime initialization outside the
    # inner inference loop. It does not change the command-marker crop.
    # The Sept-13 full TRT command took up to 10.04 s for a 1 s inner loop.
    command_startup_budget_s: float = 15.0
    # Minimum active measurement window for u.RECS energy phases. Fast runs such
    # as DeepX Full can finish in a few seconds; such short windows are dominated
    # by idle/pre/post overhead and produce misleading energy-per-inference values.
    min_active_duration_s: float = 30.0
    power_estimated_duration_margin_s: float = 2.0
    # v59ce: central duration-based energy configuration.  Workloads used for
    # energy measurements should run for a fixed wall-clock duration instead
    # of a fixed frame count; this keeps u.RECS windows stable across models.
    measurement_duration_s: float = 60.0
    native_measurement_duration_s: float = 60.0
    # v56r: mimic measurement_suite.py by default: power_calculations cuts the
    # measurement window using its trigger/default detector.  Complete-window mode
    # is available for diagnostics, but is no longer silently used unless requested.
    # Values: trimmed | complete | auto_fallback
    power_window_mode: str = "trimmed"
    run_count: int = 3
    # Statistical contract for repeated command windows.  The collector writes
    # sample standard deviation and a two-sided Student-t confidence interval
    # for all central energy/power metrics when at least two valid repeats are
    # available.
    confidence_level: float = 0.95
    # Physical/reporting labels are kept separate from the runner's dispatch
    # scope (row_variant vs. dispatch).  They travel with every aggregate so a
    # command-window result cannot later be mistaken for an AO/MB/FS value.
    physical_scope: str = "FS"
    window_label: str = "command"
    keep_raw_parquet: bool = True
    postprocess_with_power_calculations: bool = True
    # Validation bridge for the method used during the Chapter-4 calibration
    # work.  When a v2 command-window marker is available, run the historical
    # power-edge/estimated-duration postprocessor a second time on the *same*
    # raw trace.  Its result is diagnostic only and can never replace or relax
    # the marker-bound Final result.
    compare_legacy_window: bool = True
    # Separate, screening-only method-validation probe.  It deliberately runs
    # outside Native Energy pairing/claim ingestion and evaluates the marker-v2
    # and historical flank windows on the same raw trace.  Three independent
    # traces are the minimum for a decision-capable screening result.
    window_method_validation_probe_enabled: bool = True
    window_method_validation_probe_repeats: int = 3
    window_method_validation_probe_include_raw_parquet: bool = True
    window_method_validation_probe_strict: bool = True
    # v2.63 canonical name for the paired method check.  The older probe fields
    # above remain readable so existing 2.62 hardware registries are valid.
    window_ab_enabled: bool = True
    window_ab_primary_method: str = ENERGY_PRIMARY_METHOD
    window_ab_shadow_method: str = ENERGY_SHADOW_METHOD
    # Deprecated input aliases retained for hardware registries written by
    # 2.63--2.66.  They are never used to select the v2.67 scientific role.
    window_ab_baseline_method: str = ENERGY_AB_BASELINE_METHOD
    window_ab_candidate_method: str = ENERGY_AB_CANDIDATE_METHOD
    window_ab_same_raw_capture: bool = True
    window_ab_mode: str = "shadow"
    window_ab_auto_switch: bool = False
    window_ab_smoke_repeats: int = 3
    window_ab_include_raw_parquet: bool = True
    window_ab_strict: bool = True
    window_ab_requires_picoscope: bool = False
    # A collector/marker transport hiccup may invalidate one otherwise
    # independent command window.  Retry only that logical repeat, at most this
    # many times.  Failed attempts remain archived and never enter aggregates.
    invalid_repeat_max_retries: int = 1
    # Give the u.RECS stream endpoint time to release the previous connection
    # before a fresh collector process reconnects after a first-sample failure.
    invalid_repeat_reconnect_backoff_s: float = 5.0
    include_raw_parquet_in_debug_pack: bool = False
    complete_measurement_window: bool = True
    # v58b: write progress/heartbeat status while long u.RECS windows run.
    heartbeat_s: int = 60
    # v59ce: Native Producer Energy is time based.  Frame counts for
    # remote native commands are derived from the native FPS and this duration.
    # Configure here once in Tool Config / Hardware, not per EvalRun.
    native_energy_duration_s: float = 60.0
    # Optional generic fixed workload duration for future generic energy modes.
    generic_energy_duration_s: float = 60.0
    # Runtime-only source-validation metadata.  Registry parsers set these
    # fields before applying backwards-compatible value coercion so an
    # inherited claim cannot mistake a coerced malformed YAML value for a
    # validated acquisition binding.  They are never serialized.
    registry_contract_valid: bool = True
    registry_contract_errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("registry_contract_valid", None)
        data.pop("registry_contract_errors", None)
        return data


@dataclass
class EnergySetup:
    setup_id: str = ""
    accelerator: str = ""
    jetson_address: str = ""
    jetson_user: str = ""
    jetson_port: int = 22
    jetson_ssh_extra_args: str = ""
    jetson_identity_valid: bool = True
    jetson_identity_errors: tuple[str, ...] = ()
    enabled: bool = False
    urecs_address: str = ""
    data_port: int = 3000
    data_port_valid: bool = True
    idle_baseline_w: float | None = None
    accelerator_idle_w: float | None = None
    accelerator_idle_calibrated_at: str = ""
    accelerator_idle_calibration_evidence: str = ""
    accelerator_idle_calibration_binding_path: str = ""
    accelerator_idle_calibration_binding_sha256: str = ""
    # Per-u.RECS-board DC gain trim for the full-system input current chain.
    # The factor is reference_increment / measured_increment and is applied
    # only to FS/FULL_SYSTEM power and energy values.
    full_system_current_scale_factor: float | None = None
    full_system_current_scale_calibrated_at: str = ""
    full_system_current_scale_calibration_evidence: str = ""
    full_system_current_scale_calibration_sha256: str = ""
    calibration_manifest: str = ""
    calibration_sha256: str = ""
    # See EnergyDefaults.registry_contract_valid.  Direct programmatic
    # construction remains valid by default; raw registry parsers preserve
    # malformed-source evidence here for the collector's inherited-method gate.
    registry_contract_valid: bool = True
    registry_contract_errors: tuple[str, ...] = ()
    expected_channel_bindings: tuple[dict[str, Any], ...] = ()
    expected_channel_bindings_valid: bool = False
    expected_channel_binding_errors: tuple[str, ...] = ()
    # Runtime-only provenance for the exact hardware-registry snapshot from
    # which this setup was projected. Claim-bearing collection must never
    # fall back to the process-global default registry after a custom
    # Evaluation Workflow registry selected and calibrated the setup.
    hardware_registry_path: str = ""
    hardware_registry_snapshot_sha256: str = ""
    hardware_registry_provenance_valid: bool = False
    hardware_registry_provenance_errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("setup_id", None)
        d.pop("registry_contract_valid", None)
        d.pop("registry_contract_errors", None)
        d.pop("expected_channel_bindings", None)
        d.pop("expected_channel_bindings_valid", None)
        d.pop("expected_channel_binding_errors", None)
        d.pop("hardware_registry_path", None)
        d.pop("hardware_registry_snapshot_sha256", None)
        d.pop("hardware_registry_provenance_valid", None)
        d.pop("hardware_registry_provenance_errors", None)
        return d


def normalise_ssh_extra_args(value: Any) -> str:
    """Return the canonical argv-equivalent form used by SSHTransport.

    ``SSHTransport`` applies ``shlex.split`` and falls back to whitespace
    splitting for malformed legacy values.  Persisting the corresponding
    ``shlex.join`` form makes whitespace/quoting presentation irrelevant while
    still binding every effective SSH argument (including HostName and
    ProxyCommand overrides).
    """

    text = str(value or "").strip()
    if not text:
        return ""
    try:
        parts = shlex.split(text)
    except (TypeError, ValueError):
        parts = text.split()
    return shlex.join(parts)


def resolve_jetson_identity(raw_setup: Mapping[str, Any] | None) -> dict[str, Any]:
    """Resolve and validate the non-secret SSH endpoint without masking errors.

    Older registries keep the endpoint in ``remote`` while current registries
    use ``host``.  The value precedence matches platform control, but explicit
    malformed values (notably port 0/non-integer and non-string SSH arguments)
    remain visible as errors instead of silently becoming defaults.
    """

    source = raw_setup if isinstance(raw_setup, Mapping) else {}
    errors: list[str] = []

    host_raw = source.get("host")
    if host_raw is None:
        host: dict[str, Any] = {}
    elif isinstance(host_raw, str):
        host = {"address": host_raw}
    elif isinstance(host_raw, Mapping):
        host = dict(host_raw)
    else:
        host = {}
        errors.append("jetson_host_config_invalid")
    remote_raw = source.get("remote")
    if remote_raw is None:
        remote: dict[str, Any] = {}
    elif isinstance(remote_raw, Mapping):
        remote = dict(remote_raw)
    else:
        remote = {}
        errors.append("jetson_remote_config_invalid")

    def exact_identity_text(value: Any) -> tuple[str, bool]:
        if not isinstance(value, str) or not value or value != value.strip():
            return "", False
        if any(
            ch.isspace() or ord(ch) < 32 or ord(ch) == 127 for ch in value
        ):
            return "", False
        return value, True

    address_entries: list[tuple[str, Any]] = []
    for owner, mapping, key in (
        ("host_address", host, "address"),
        ("host_host", host, "host"),
        ("remote_host", remote, "host"),
        ("remote_address", remote, "address"),
    ):
        if key in mapping:
            address_entries.append((owner, mapping.get(key)))
    canonical_addresses: dict[str, str] = {}
    for owner, raw_value in address_entries:
        canonical, valid = exact_identity_text(raw_value)
        if not valid:
            errors.append(f"jetson_{owner}_not_canonical")
            errors.append("jetson_address_not_string_or_blank")
        else:
            canonical_addresses[owner] = canonical
    if (
        "host_address" in canonical_addresses
        and "host_host" in canonical_addresses
        and canonical_addresses["host_address"]
        != canonical_addresses["host_host"]
    ):
        errors.append("jetson_host_address_alias_mismatch")
    host_address = canonical_addresses.get(
        "host_address", canonical_addresses.get("host_host", "")
    )
    remote_address = canonical_addresses.get(
        "remote_host", canonical_addresses.get("remote_address", "")
    )
    if host_address and remote_address and host_address != remote_address:
        errors.append("jetson_host_remote_address_mismatch")
    if address_entries:
        first_address_owner, _first_address_raw = address_entries[0]
        address = canonical_addresses.get(first_address_owner, "")
    else:
        address = ""
        errors.append("jetson_address_missing")

    user_entries: list[tuple[str, Any]] = []
    if "user" in host:
        user_entries.append(("host_user", host.get("user")))
    if "user" in remote:
        user_entries.append(("remote_user", remote.get("user")))
    canonical_users: dict[str, str] = {}
    for owner, raw_value in user_entries:
        canonical, valid = exact_identity_text(raw_value)
        if not valid:
            errors.append(f"jetson_{owner}_not_canonical")
            errors.append("jetson_user_not_string_or_blank")
        else:
            canonical_users[owner] = canonical
    if (
        "host_user" in canonical_users
        and "remote_user" in canonical_users
        and canonical_users["host_user"] != canonical_users["remote_user"]
    ):
        errors.append("jetson_host_remote_user_mismatch")
    if user_entries:
        user = canonical_users.get(user_entries[0][0], "")
    else:
        user = "nx"

    def exact_port(owner: str, value: Any) -> tuple[int, bool]:
        if type(value) is not int:
            errors.append(f"jetson_{owner}_port_not_integer")
            errors.append("jetson_port_not_integer")
            return 0, False
        if not 1 <= value <= 65535:
            errors.append(f"jetson_{owner}_port_out_of_range")
            errors.append("jetson_port_out_of_range")
            return value, False
        return value, True

    host_port_present = "port" in host
    remote_port_present = "port" in remote
    host_port, host_port_valid = (
        exact_port("host", host.get("port"))
        if host_port_present
        else (22, True)
    )
    remote_port, remote_port_valid = (
        exact_port("remote", remote.get("port"))
        if remote_port_present
        else (22, True)
    )
    if (
        host_port_present
        and remote_port_present
        and host_port_valid
        and remote_port_valid
        and host_port != remote_port
    ):
        errors.append("jetson_host_remote_port_mismatch")
    port = host_port if host_port_present else remote_port

    host_ssh_present = "ssh_extra_args" in host
    remote_ssh_present = "ssh_extra_args" in remote
    ssh_values: dict[str, str] = {}
    for owner, present, value in (
        ("host", host_ssh_present, host.get("ssh_extra_args")),
        ("remote", remote_ssh_present, remote.get("ssh_extra_args")),
    ):
        if not present:
            continue
        if not isinstance(value, str):
            errors.append(f"jetson_{owner}_ssh_extra_args_not_string")
            errors.append("jetson_ssh_extra_args_not_string")
            ssh_values[owner] = ""
        else:
            ssh_values[owner] = normalise_ssh_extra_args(value)
    if (
        host_ssh_present
        and remote_ssh_present
        and all(owner in ssh_values for owner in ("host", "remote"))
        and ssh_values["host"] != ssh_values["remote"]
    ):
        errors.append("jetson_host_remote_ssh_extra_args_mismatch")
    ssh_extra_args = (
        ssh_values.get("host", "")
        if host_ssh_present
        else ssh_values.get("remote", "")
    )

    return {
        "address": address,
        "user": user,
        "port": port,
        "ssh_extra_args": ssh_extra_args,
        "valid": not errors,
        "errors": tuple(dict.fromkeys(errors)),
    }


class DuplicateEnergySetupIdError(ValueError):
    """The selected energy setup identity is ambiguous in the registry."""

    def __init__(self, setup_id: str) -> None:
        self.setup_id = str(setup_id or "").strip()
        super().__init__(
            "duplicate hardware setup id for energy claim: "
            + (self.setup_id or "<empty>")
        )


# Backwards-compatible alias used by early v56a/b helper scripts.
EnergySetupConfig = EnergySetup


def default_energy_home() -> Path:
    return Path(os.path.expanduser("~/.onnx_splitpoint_tool/energy"))




def configured_workdir_root(explicit: str | Path | None = None) -> Path:
    """Return the human-facing working directory root used for results/artifacts.

    Priority:
      1. explicit argument
      2. ONNX_SPLITPOINT_WORKDIR
      3. persisted GUI setting output_dir / working_dir
      4. current project directory

    This intentionally differs from ``~/.onnx_splitpoint_tool`` which is used
    for persistent state/configuration only.  Energy measurements, BenchmarkSets
    and EvaluationRuns should be stored in the working directory.
    """
    if explicit:
        return Path(os.path.expandvars(os.path.expanduser(str(explicit)))).resolve()
    env = str(os.environ.get("ONNX_SPLITPOINT_WORKDIR", "") or "").strip()
    if env:
        return Path(os.path.expandvars(os.path.expanduser(env))).resolve()
    try:
        from onnx_splitpoint_tool.settings.store import SettingsStore
        data = SettingsStore().load()
        val = str(data.get("output_dir") or data.get("working_dir") or "").strip()
        if val:
            return Path(os.path.expandvars(os.path.expanduser(val))).resolve()
    except Exception:
        pass
    try:
        from onnx_splitpoint_tool.paths import splitpoint_project_root
        return Path(splitpoint_project_root()).expanduser().resolve()
    except Exception:
        return Path.cwd().resolve()


def energy_measurements_root(workdir: str | Path | None = None) -> Path:
    """Return the canonical working-dir location for u.RECS measurements."""
    try:
        from onnx_splitpoint_tool.workdir import ensure_workdir
        return ensure_workdir(configured_workdir_root(workdir)).energy_measurements
    except Exception:
        root = configured_workdir_root(workdir) / "EnergyMeasurements"
        root.mkdir(parents=True, exist_ok=True)
        return root


def default_energy_config_file() -> Path:
    return Path(os.path.expanduser("~/.onnx_splitpoint_tool/energy_config.yaml"))


def default_registry_path() -> Path:
    return Path(os.path.expanduser("~/.onnx_splitpoint_tool/hardware_setups.yaml"))


def _bool(v: Any, default: bool = False) -> bool:
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in {"1", "true", "yes", "on", "y", "enabled"}


def _float_or_none(v: Any) -> float | None:
    if v is None or str(v).strip() == "":
        return None
    try:
        return float(v)
    except Exception:
        return None


def _int(v: Any, default: int) -> int:
    try:
        return int(float(v))
    except Exception:
        return default


def _float(v: Any, default: float) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _strict_registry_int(
    value: Any,
    *,
    default: int,
    minimum: int,
    maximum: int | None = None,
) -> tuple[int, bool]:
    """Parse a registry integer without hiding string/bool/float coercion."""

    if type(value) is not int:
        return default, False
    parsed = int(value)
    valid = parsed >= minimum and (maximum is None or parsed <= maximum)
    return parsed, valid


def _strict_optional_registry_number(
    raw: Mapping[str, Any], key: str,
) -> tuple[float | None, bool]:
    """Parse one optional claim-bearing numeric registry field exactly.

    A missing or explicit null value means "not calibrated" and is valid.
    Present values must already be finite YAML/JSON integers or floats;
    strings and booleans are never coerced into scientific evidence.
    """

    if key not in raw or raw.get(key) is None:
        return None, True
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None, False
    parsed = float(value)
    if not math.isfinite(parsed):
        return None, False
    return parsed, True


def _strict_urecs_address(value: Any) -> tuple[str, bool, str]:
    """Return one safe u.RECS endpoint and a stable validation reason."""

    if not isinstance(value, str):
        return "", False, "setup_urecs_address_not_string"
    address = value.strip()
    if not address:
        return address, False, "setup_urecs_address_blank"
    if any(ch.isspace() or ord(ch) < 32 or ord(ch) == 127 for ch in value):
        return address, False, "setup_urecs_address_contains_whitespace_or_control"
    return address, True, ""


def _strict_setup_id(value: Any) -> tuple[str, bool, str]:
    """Return a claim-safe setup id without string/whitespace coercion."""

    if not isinstance(value, str):
        return "", False, "hardware_setup_id_not_string"
    if not value:
        return "", False, "hardware_setup_id_blank"
    if value != value.strip() or any(
        ch.isspace() or ord(ch) < 32 or ord(ch) == 127 for ch in value
    ):
        return "", False, "hardware_setup_id_contains_whitespace_or_control"
    return value, True, ""


def _strict_urecs_address_fields(
    raw: Mapping[str, Any],
) -> tuple[str, bool, tuple[str, ...]]:
    """Resolve the supported address alias without hiding dual-key conflicts."""

    has_primary = "urecs_address" in raw
    has_alias = "address" in raw
    errors: list[str] = []
    primary_value = raw.get("urecs_address") if has_primary else None
    alias_value = raw.get("address") if has_alias else None
    primary_address = ""
    alias_address = ""
    primary_valid = False
    alias_valid = False
    if has_primary:
        primary_address, primary_valid, primary_error = _strict_urecs_address(
            primary_value
        )
        if not primary_valid:
            errors.append(primary_error)
    if has_alias:
        alias_address, alias_valid, alias_error = _strict_urecs_address(
            alias_value
        )
        if not alias_valid:
            errors.append(
                alias_error.replace(
                    "setup_urecs_address", "setup_energy_address", 1
                )
            )
    if not has_primary and not has_alias:
        errors.append("setup_urecs_address_blank")
    if has_primary and has_alias:
        if primary_valid and alias_valid and primary_address != alias_address:
            errors.append("setup_urecs_address_alias_mismatch")
        valid = bool(primary_valid and alias_valid and not errors)
        return primary_address, valid, tuple(dict.fromkeys(errors))
    if has_primary:
        return primary_address, primary_valid, tuple(dict.fromkeys(errors))
    return alias_address, alias_valid, tuple(dict.fromkeys(errors))


def _setup_claim_override_errors(
    energy: Mapping[str, Any],
    *,
    default_collector_binary: Any,
    default_power_binary: Any,
    default_mode: Any,
) -> tuple[str, ...]:
    """Validate claim-bearing per-setup keys that execution reads globally."""

    errors: list[str] = []
    if "enabled" in energy and type(energy.get("enabled")) is not bool:
        errors.append("setup_energy_enabled_not_bool")
    if energy.get("enabled") is not True:
        errors.append("setup_energy_not_enabled")
    del default_collector_binary, default_power_binary, default_mode
    for key in (
        "collector_binary",
        "power_calculations_binary",
        "mode",
        "physical_scope",
        "measurement_physical_scope",
        "window_label",
        "measurement_window",
    ):
        if key not in energy:
            continue
        errors.append(f"setup_{key}_override_not_allowed")
    return tuple(dict.fromkeys(errors))


def _normalise_registry_sha256(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    digest = value.strip().lower()
    if digest.startswith("sha256:"):
        digest = digest.split(":", 1)[1]
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        return ""
    return digest


def _parse_full_system_current_scale_fields(
    raw: Mapping[str, Any],
    registry_errors: list[str],
) -> tuple[float | None, str, str, str]:
    """Strictly parse the optional, evidence-bound FS DC gain trim."""

    factor, factor_valid = _strict_optional_registry_number(
        raw, "full_system_current_scale_factor"
    )
    if factor is not None and not 0.5 <= factor <= 1.5:
        factor = None
        factor_valid = False
    calibrated_raw = raw.get("full_system_current_scale_calibrated_at", "")
    evidence_raw = raw.get("full_system_current_scale_calibration_evidence", "")
    sha_raw = raw.get("full_system_current_scale_calibration_sha256", "")
    calibrated_at = (
        calibrated_raw.strip() if isinstance(calibrated_raw, str) else ""
    )
    evidence_path = evidence_raw.strip() if isinstance(evidence_raw, str) else ""
    evidence_sha256 = _normalise_registry_sha256(sha_raw)
    configured = bool(
        raw.get("full_system_current_scale_factor") is not None
        or str(calibrated_raw or "").strip()
        or str(evidence_raw or "").strip()
        or str(sha_raw or "").strip()
    )
    if not factor_valid:
        registry_errors.append(
            "setup_full_system_current_scale_factor_missing_or_invalid"
        )
    if configured and factor is None:
        registry_errors.append("setup_full_system_current_scale_incomplete")
    if configured and not calibrated_at:
        registry_errors.append(
            "setup_full_system_current_scale_calibrated_at_missing_or_invalid"
        )
    if configured and not evidence_path:
        registry_errors.append(
            "setup_full_system_current_scale_evidence_missing_or_invalid"
        )
    if configured and not evidence_sha256:
        registry_errors.append(
            "setup_full_system_current_scale_sha256_missing_or_invalid"
        )
    return factor, calibrated_at, evidence_path, evidence_sha256


def _canonical_registry_manifest_path(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        return ""
    try:
        return str(_expand(value.strip()).resolve())
    except (OSError, RuntimeError, ValueError):
        return ""


def _expected_channel_bindings_from_registry(
    registry: Mapping[str, Any],
    *,
    selected_setup_id: str,
) -> tuple[tuple[dict[str, Any], ...], bool, tuple[str, ...]]:
    """Project the complete method-bound setup set from fresh registry data.

    Membership is independent of the manifest: every unique registry setup
    carrying the same canonical manifest path and normalized digest as the
    selected setup belongs to the expected set.  Consequently a maliciously
    re-signed manifest cannot add, omit or rebind a setup while still passing a
    per-setup lookup.
    """

    errors: list[str] = []
    raw_rows = registry.get("hardware_setups")
    if not isinstance(raw_rows, list):
        return (), False, ("expected_channel_bindings_hardware_setups_not_list",)
    rows = [row for row in raw_rows if isinstance(row, Mapping)]
    if len(rows) != len(raw_rows):
        errors.append("expected_channel_bindings_setup_row_not_mapping")
    for index, row in enumerate(rows):
        _setup_id, setup_id_valid, setup_id_error = _strict_setup_id(
            row.get("id")
        )
        if not setup_id_valid:
            errors.append(
                f"expected_channel_bindings_{setup_id_error}:{index}"
            )
    ids = [
        setup_id
        for row in rows
        for setup_id, valid, _error in (_strict_setup_id(row.get("id")),)
        if valid
    ]
    if len(ids) != len(set(ids)):
        errors.append("expected_channel_bindings_setup_id_duplicate")

    selected_id, selected_id_valid, selected_id_error = _strict_setup_id(
        selected_setup_id
    )
    if not selected_id_valid:
        errors.append(
            f"expected_channel_bindings_selected_{selected_id_error}"
        )
        return (), False, tuple(dict.fromkeys(errors))

    selected_matches = [
        row
        for row in rows
        if _strict_setup_id(row.get("id"))[:2] == (selected_id, True)
    ]
    if len(selected_matches) != 1:
        errors.append("expected_channel_bindings_selected_setup_not_unique")
        return (), False, tuple(dict.fromkeys(errors))
    selected_energy_raw = selected_matches[0].get("energy")
    if not isinstance(selected_energy_raw, Mapping):
        errors.append("expected_channel_bindings_selected_energy_not_mapping")
        return (), False, tuple(dict.fromkeys(errors))
    selected_energy = dict(selected_energy_raw)
    selected_manifest = _canonical_registry_manifest_path(
        selected_energy.get("calibration_manifest")
    )
    selected_digest = _normalise_registry_sha256(
        selected_energy.get("calibration_sha256")
    )
    if not selected_manifest:
        errors.append("expected_channel_bindings_manifest_missing_or_invalid")
    if not selected_digest:
        errors.append("expected_channel_bindings_sha256_missing_or_invalid")

    defaults_raw_value = registry.get("energy_defaults")
    if "energy_defaults" in registry and not isinstance(
        defaults_raw_value, Mapping
    ):
        errors.append("expected_channel_bindings_energy_defaults_not_mapping")
        defaults_raw: dict[str, Any] = {}
    else:
        defaults_raw = dict(defaults_raw_value or {})
    data_port, data_port_valid = _strict_registry_int(
        defaults_raw.get("data_port", 3000),
        default=0,
        minimum=1,
        maximum=65535,
    )
    channel, channel_valid = _strict_registry_int(
        defaults_raw.get("channel", 0), default=0, minimum=0
    )
    sample_rate, sample_rate_valid = _strict_registry_int(
        defaults_raw.get("sample_rate", 2000), default=0, minimum=1
    )
    default_collector_binary = defaults_raw.get(
        "collector_binary", "urecs-data-collector"
    )
    default_power_binary = defaults_raw.get(
        "power_calculations_binary", "power_calculations"
    )
    default_mode = defaults_raw.get("mode", "fast_firmware")
    if not data_port_valid:
        errors.append("expected_channel_bindings_data_port_invalid")
    if not channel_valid:
        errors.append("expected_channel_bindings_channel_invalid")
    if not sample_rate_valid:
        errors.append("expected_channel_bindings_sample_rate_invalid")

    bindings: list[dict[str, Any]] = []
    if selected_manifest and selected_digest:
        for row_index, row in enumerate(rows):
            setup_id, setup_id_valid, _setup_id_error = _strict_setup_id(
                row.get("id")
            )
            setup_label = setup_id if setup_id_valid else f"row_{row_index}"
            energy_raw = row.get("energy")
            if energy_raw is None:
                continue
            if not isinstance(energy_raw, Mapping):
                errors.append(
                    f"expected_channel_bindings_energy_not_mapping:{setup_label}"
                )
                continue
            energy = dict(energy_raw)
            candidate_manifest = _canonical_registry_manifest_path(
                energy.get("calibration_manifest")
            )
            candidate_digest = _normalise_registry_sha256(
                energy.get("calibration_sha256")
            )
            if not candidate_manifest and not candidate_digest:
                continue
            if candidate_manifest != selected_manifest or candidate_digest != selected_digest:
                continue
            errors.extend(
                f"expected_channel_bindings_{reason}:{setup_label}"
                for reason in _setup_claim_override_errors(
                    energy,
                    default_collector_binary=default_collector_binary,
                    default_power_binary=default_power_binary,
                    default_mode=default_mode,
                )
            )
            for setup_key, global_value, global_valid, minimum in (
                ("data_port", data_port, data_port_valid, 1),
                ("channel", channel, channel_valid, 0),
                ("sample_rate", sample_rate, sample_rate_valid, 1),
                ("sample_rate_hz", sample_rate, sample_rate_valid, 1),
            ):
                if setup_key not in energy:
                    continue
                upper = 65535 if setup_key == "data_port" else None
                setup_value, setup_value_valid = _strict_registry_int(
                    energy.get(setup_key),
                    default=0,
                    minimum=minimum,
                    maximum=upper,
                )
                if not setup_value_valid:
                    errors.append(
                        f"expected_channel_bindings_{setup_key}_invalid:{setup_label}"
                    )
                elif not global_valid or setup_value != global_value:
                    errors.append(
                        f"expected_channel_bindings_{setup_key}_defaults_mismatch:{setup_label}"
                    )
            if (
                "sample_rate" in energy
                and "sample_rate_hz" in energy
                and energy.get("sample_rate") != energy.get("sample_rate_hz")
            ):
                errors.append(
                    f"expected_channel_bindings_sample_rate_alias_mismatch:{setup_label}"
                )
            (
                urecs_address,
                urecs_valid,
                urecs_errors,
            ) = _strict_urecs_address_fields(energy)
            if not urecs_valid:
                errors.extend(
                    f"expected_channel_bindings_{reason}:{setup_label}"
                    for reason in urecs_errors
                )
                continue
            if not setup_id_valid:
                continue
            bindings.append(
                {
                    "setup_id": setup_id,
                    "urecs_address": urecs_address,
                    "data_port": data_port,
                    "channel": channel,
                    "sample_rate_hz": sample_rate,
                    "scope": "FS",
                    "measurement_point": "complete_system_input",
                }
            )
    bindings.sort(key=lambda row: str(row.get("setup_id") or ""))
    if not bindings:
        errors.append("expected_channel_bindings_empty")
    elif selected_id not in {
        str(row.get("setup_id") or "") for row in bindings
    }:
        errors.append("expected_channel_bindings_selected_setup_missing")
    unique_errors = tuple(dict.fromkeys(errors))
    return tuple(bindings), not unique_errors, unique_errors


def resolve_energy_ab_config(
    raw: Mapping[str, Any] | None = None,
    *,
    legacy_probe: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve the role-explicit v2.67 A/B contract.

    The exact 2.63--2.66 ``baseline_method``/``candidate_method`` pair is
    accepted as an input alias so persisted Tool Config files keep working.
    The resolved snapshot always records the reviewed v2.67 roles explicitly;
    unsupported method names still fail closed.
    """
    data = dict(raw or {})
    legacy = dict(legacy_probe or {})
    legacy_used = not bool(data) and bool(legacy)
    source = data if data else legacy
    legacy_baseline = str(source.get("baseline_method") or "").strip()
    legacy_candidate = str(source.get("candidate_method") or "").strip()
    legacy_method_aliases_applied = bool(
        source
        and
        not source.get("primary_method")
        and not source.get("shadow_method")
        and (
            legacy_baseline in {"", ENERGY_AB_BASELINE_METHOD}
            and legacy_candidate in {"", ENERGY_AB_CANDIDATE_METHOD}
        )
    )
    cfg = EnergyABConfig(
        enabled=_bool(source.get("enabled"), True),
        primary_method=str(source.get("primary_method") or ENERGY_PRIMARY_METHOD),
        shadow_method=str(source.get("shadow_method") or ENERGY_SHADOW_METHOD),
        same_raw_capture=_bool(source.get("same_raw_capture"), True),
        mode=str(source.get("mode") or "shadow").strip().lower(),
        auto_switch=_bool(source.get("auto_switch"), False),
        smoke_repeats=max(1, _int(source.get("smoke_repeats", source.get("repeats")), 3)),
        include_raw_parquet=_bool(source.get("include_raw_parquet"), True),
        strict=_bool(source.get("strict"), True),
        requires_picoscope=_bool(source.get("requires_picoscope", source.get("require_picoscope")), False),
    )
    errors: list[str] = []
    if source.get("baseline_method") not in (None, "", ENERGY_AB_BASELINE_METHOD):
        errors.append("unsupported_legacy_baseline_method")
    if source.get("candidate_method") not in (None, "", ENERGY_AB_CANDIDATE_METHOD):
        errors.append("unsupported_legacy_candidate_method")
    if cfg.primary_method != ENERGY_PRIMARY_METHOD:
        errors.append("primary_method_must_be_command_marker_window")
    if cfg.shadow_method != ENERGY_SHADOW_METHOD:
        errors.append("shadow_method_must_be_chapter4_legacy_window")
    if not cfg.same_raw_capture:
        errors.append("same_raw_capture_required")
    if cfg.mode != "shadow":
        errors.append("shadow_mode_required")
    if cfg.auto_switch:
        errors.append("automatic_method_switch_forbidden")
    if cfg.requires_picoscope:
        errors.append("picoscope_not_required_for_postprocessing_ab")
    result = cfg.to_dict()
    result.update({
        "schema": "onnx-splitpoint/energy-window-ab-config",
        "schema_version": 2,
        "valid": not errors,
        "validation_errors": errors,
        "configuration_source": "legacy_window_method_validation_probe" if legacy_used else "energy.window_method_ab",
        "scientific_method_decision": "frozen_command_marker_primary_chapter4_shadow",
        "scientific_primary_method": ENERGY_PRIMARY_METHOD,
        "scientific_shadow_method": ENERGY_SHADOW_METHOD,
        "shadow_role": "same_trace_sensitivity_only",
        "legacy_method_aliases_applied": legacy_method_aliases_applied,
        "deprecated_input_aliases": {
            "baseline_method": ENERGY_AB_BASELINE_METHOD,
            "candidate_method": ENERGY_AB_CANDIDATE_METHOD,
        },
        # Output aliases keep 2.63--2.66 profile plumbing readable.  Runtime
        # role selection never consumes these fields in v2.67.
        "baseline_method": ENERGY_AB_BASELINE_METHOD,
        "candidate_method": ENERGY_AB_CANDIDATE_METHOD,
        "candidate_eligible_for_auto_import": False,
        "candidate_eligible_for_auto_switch": False,
    })
    return result


def apply_energy_ab_config(
    defaults: EnergyDefaults,
    raw: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Bind one explicit A/B request to the collector defaults.

    Native Energy is launched through a separate CLI process, so profile-only
    values are otherwise lost when that process reloads the hardware registry.
    Keeping this binding next to :func:`resolve_energy_ab_config` gives the
    generic and Native paths the same fail-closed runtime contract.
    """
    resolved = resolve_energy_ab_config(raw)
    defaults.window_ab_enabled = _bool(resolved.get("enabled"), True)
    defaults.window_ab_primary_method = str(
        resolved.get("primary_method") or ENERGY_PRIMARY_METHOD
    )
    defaults.window_ab_shadow_method = str(
        resolved.get("shadow_method") or ENERGY_SHADOW_METHOD
    )
    # Retain the exact legacy aliases for old generic-runner plumbing.  They do
    # not determine the v2.67 scientific roles.
    defaults.window_ab_baseline_method = str(
        ENERGY_AB_BASELINE_METHOD
    )
    defaults.window_ab_candidate_method = str(
        ENERGY_AB_CANDIDATE_METHOD
    )
    defaults.window_ab_same_raw_capture = _bool(
        resolved.get("same_raw_capture"), True
    )
    defaults.window_ab_mode = str(resolved.get("mode") or "shadow")
    defaults.window_ab_auto_switch = _bool(resolved.get("auto_switch"), False)
    defaults.window_ab_smoke_repeats = max(
        1, _int(resolved.get("smoke_repeats"), 3)
    )
    defaults.window_ab_include_raw_parquet = _bool(
        resolved.get("include_raw_parquet"), True
    )
    defaults.window_ab_strict = _bool(resolved.get("strict"), True)
    defaults.window_ab_requires_picoscope = _bool(
        resolved.get("requires_picoscope"), False
    )
    # A requested same-trace shadow is attempted for every retained trace.  A
    # shadow failure remains non-blocking, but the raw Parquet cannot be
    # discarded when the frozen request asks to archive it.
    defaults.compare_legacy_window = bool(defaults.window_ab_enabled)
    defaults.keep_raw_parquet = bool(
        defaults.keep_raw_parquet or defaults.window_ab_include_raw_parquet
    )
    return resolved


def energy_ab_cli_args(raw: Mapping[str, Any] | None) -> list[str]:
    """Serialize an explicit A/B request for a Native measurement subprocess."""
    if not isinstance(raw, Mapping) or not raw:
        return []
    resolved = resolve_energy_ab_config(raw)
    if resolved.get("enabled") and not resolved.get("valid"):
        raise ValueError(
            "invalid frozen Native Energy A/B contract: "
            + ", ".join(str(item) for item in resolved.get("validation_errors") or [])
        )
    return [
        "--window-method-ab-json",
        json.dumps(resolved, sort_keys=True, separators=(",", ":")),
    ]


def _state_for_status(
    status: str,
    *,
    requested: bool,
    configured: bool,
    reason: str = "",
) -> EffectiveEnergyState:
    status = str(status or "").strip().lower()
    if status not in ENERGY_LIFECYCLE_STATES:
        raise ValueError(f"unsupported energy lifecycle state: {status!r}")
    scheduled = status in {"scheduled", "running", "completed"}
    return EffectiveEnergyState(
        status=status,
        requested=bool(requested),
        configured=bool(configured),
        scheduled=scheduled,
        running=status == "running",
        completed=status == "completed",
        skipped=status == "skipped",
        failed=status == "failed",
        reason=str(reason or ""),
    )


def transition_energy_state(
    state: EffectiveEnergyState | Mapping[str, Any],
    target: str,
    *,
    reason: str = "",
) -> EffectiveEnergyState:
    """Advance a lifecycle without allowing completed/failed work to revive."""
    current = state if isinstance(state, EffectiveEnergyState) else EffectiveEnergyState(**{
        key: value
        for key, value in dict(state or {}).items()
        if key in EffectiveEnergyState.__dataclass_fields__
    })
    target_l = str(target or "").strip().lower()
    allowed = {
        "requested": {"configured", "skipped", "failed"},
        "configured": {"scheduled", "skipped", "failed"},
        "scheduled": {"running", "skipped", "failed"},
        "running": {"completed", "failed"},
        "completed": set(),
        "skipped": set(),
        "failed": set(),
    }
    if target_l not in allowed.get(current.status, set()):
        raise ValueError(f"invalid energy state transition: {current.status} -> {target_l}")
    return _state_for_status(
        target_l,
        requested=current.requested,
        configured=current.configured or target_l in {"configured", "scheduled", "running", "completed"},
        reason=reason,
    )


def resolve_effective_energy_config(
    profile: Mapping[str, Any] | None,
    *,
    defaults: EnergyDefaults | Mapping[str, Any] | None = None,
    lifecycle_status: str | None = None,
    reason: str = "",
) -> dict[str, Any]:
    """Resolve one authoritative energy request from new and legacy profiles.

    For new profiles, ``energy.requested`` is authoritative.  For 2.62
    profiles the function recognises ``native_producers.energy.enabled`` and
    the old ``requested_native_energy``/``measurement_path=native_only`` shape.
    This removes the misleading situation where generic energy was shown as
    disabled even though native energy had been requested.
    """
    payload = dict(profile or {})
    top = dict(payload.get("energy") or {}) if isinstance(payload.get("energy"), Mapping) else {}
    native_root = dict(payload.get("native_producers") or {}) if isinstance(payload.get("native_producers"), Mapping) else {}
    native = dict(native_root.get("energy") or {}) if isinstance(native_root.get("energy"), Mapping) else {}
    default_map = defaults.to_dict() if isinstance(defaults, EnergyDefaults) else dict(defaults or {})

    native_enabled = _bool(native.get("enabled"), False)
    # ``energy.enabled`` is an old umbrella request and is ambiguous in
    # Native-capable profiles.  Generic execution requires its dedicated flag;
    # an old enabled-only request therefore becomes a visible configuration
    # blocker below instead of being planned but silently omitted at dispatch.
    generic_enabled = _bool(top.get("generic_enabled"), False)
    compatibility_notes: list[str] = []
    if "requested" in top:
        requested = _bool(top.get("requested"), False)
        request_source = "energy.requested"
    elif _bool(top.get("requested_native_energy"), False):
        requested = True
        request_source = "energy.requested_native_energy"
        compatibility_notes.append("legacy_requested_native_energy")
    elif str(top.get("measurement_path") or "").strip().lower() == "native_only" and native_enabled:
        requested = True
        request_source = "legacy_native_only_path"
        compatibility_notes.append("top_level_enabled_described_generic_path_only")
    elif "enabled" in top:
        requested = _bool(top.get("enabled"), False)
        request_source = "energy.enabled"
    elif native_enabled:
        requested = True
        request_source = "native_producers.energy.enabled"
        compatibility_notes.append("legacy_native_energy_enablement")
    else:
        requested = _bool(default_map.get("enabled"), False)
        request_source = "energy_defaults.enabled"

    path = str(top.get("measurement_path") or "").strip().lower()
    if not path:
        if native_enabled and generic_enabled:
            path = "native_and_generic"
        elif native_enabled:
            path = "native_only"
        elif generic_enabled or requested:
            path = "generic"
        else:
            path = "disabled"

    errors: list[str] = []
    if requested and path in {"native_only", "native_and_generic"} and not native_enabled:
        errors.append("native_energy_path_requested_but_native_energy_disabled")
    if requested and path in {"generic", "native_and_generic"} and not generic_enabled:
        errors.append(
            "generic_energy_path_requested_but_generic_energy_disabled"
        )
    if requested and path == "disabled":
        errors.append("energy_requested_with_disabled_measurement_path")
    if not requested and "requested" in top and (native_enabled or generic_enabled):
        errors.append("explicit_energy_request_false_conflicts_with_enabled_execution_path")

    ab_raw = top.get("window_method_ab") if isinstance(top.get("window_method_ab"), Mapping) else {}
    legacy_probe = native.get("window_method_validation_probe") if isinstance(native.get("window_method_validation_probe"), Mapping) else {}
    ab = resolve_energy_ab_config(ab_raw, legacy_probe=legacy_probe)
    if requested and ab.get("enabled") and not ab.get("valid"):
        errors.extend(f"window_method_ab:{item}" for item in list(ab.get("validation_errors") or []))

    configured = bool(requested and not errors)
    if lifecycle_status:
        status = str(lifecycle_status).strip().lower()
    elif errors:
        status = "failed"
    elif requested:
        status = "configured"
    else:
        status = "skipped"
    state_reason = reason or (";".join(errors) if errors else ("not_requested" if not requested else ""))
    state = _state_for_status(status, requested=requested, configured=configured, reason=state_reason)

    repeats = max(1, _int(top.get("repeats", top.get("repeat_override")), _int(default_map.get("run_count"), 3)))
    effective = {
        "schema": "onnx-splitpoint/effective-energy-config",
        "schema_version": 1,
        "request_source": request_source,
        "measurement_path": path,
        "native_energy_enabled": native_enabled,
        "generic_energy_enabled": generic_enabled,
        "repeats": repeats,
        "lifecycle": state.to_dict(),
        "window_method_ab": ab,
        "configuration_errors": errors,
        "compatibility_notes": compatibility_notes,
    }
    effective["config_identity_sha256"] = sha256_json({
        key: value for key, value in effective.items() if key not in {"config_identity_sha256", "lifecycle"}
    })
    return effective


def write_effective_energy_manifest(
    output: str | Path,
    profile: Mapping[str, Any],
    *,
    defaults: EnergyDefaults | Mapping[str, Any] | None = None,
    lifecycle_status: str | None = None,
    reason: str = "",
) -> Path:
    """Persist the effective request and lifecycle as protocol evidence."""
    effective = resolve_effective_energy_config(
        profile,
        defaults=defaults,
        lifecycle_status=lifecycle_status,
        reason=reason,
    )
    manifest = {
        "schema": "onnx-splitpoint/energy-protocol-manifest",
        "schema_version": 1,
        "created_at": now_iso(),
        "effective_energy": effective,
    }
    manifest["manifest_payload_sha256"] = sha256_json({
        key: value for key, value in manifest.items() if key != "manifest_payload_sha256"
    })
    return write_json(output, manifest)


def _expand(path: str | Path) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(path))))


def _default_registry() -> dict[str, Any]:
    return {
        "schema": "onnx-splitpoint/hardware-setups",
        "schema_version": 2,
        "hardware_setups": [],
        "energy_defaults": EnergyDefaults().to_dict(),
    }


def _migrate_legacy_m2_command_defaults(data: dict[str, Any]) -> bool:
    """Replace only the known legacy M.2 token (or an absent default).

    Controller commands are user-configurable and may differ between firmware
    deployments.  Consequently, this migration intentionally does *not* trim,
    case-fold, or otherwise reinterpret an existing value.  Only a missing or
    YAML-null value and the exact historical default ``m2`` become ``m.2``.
    Every other custom value is preserved exactly.
    """

    changed = False
    setups = data.get("hardware_setups")
    if not isinstance(setups, list):
        return False
    for raw in setups:
        if not isinstance(raw, dict):
            continue
        original = raw.get("power_control")
        if "power_control" in raw and original is None:
            # A null mapping is not the same as an absent legacy block.  Keep
            # it visible so physical-operation admission can reject the
            # ambiguous configuration instead of materialising safe-looking
            # defaults around it.  A null *m2_command inside a mapping* remains
            # the documented legacy token migration below.
            continue
        if original is not None and not isinstance(original, Mapping):
            # An invalid/custom non-mapping is outside the additive migration's
            # authority.  Validation elsewhere may reject it, but do not erase
            # it while merely loading the registry.
            continue
        power = dict(original or {})
        missing = "m2_command" not in power or power.get("m2_command") is None
        legacy = power.get("m2_command") == PLATFORM_POWER_LEGACY_M2_COMMAND
        if not (missing or legacy):
            continue
        power["m2_command"] = PLATFORM_POWER_M2_COMMAND
        raw["power_control"] = power
        changed = True
    return changed


def _migrate_legacy_energy_scope_default(data: dict[str, Any]) -> bool:
    """Upgrade only a missing or exact historical MB default to FS.

    Full-system input power is the physical measurement used by the current
    campaign.  Preserve malformed values, aliases that disagree, and explicit
    AO/custom scopes so validation can still report them honestly.  This is a
    narrow configuration migration, not a relabelling of historical results.
    """

    if "energy_defaults" not in data:
        data["energy_defaults"] = {
            "physical_scope": "FS",
            "window_label": "command",
        }
        return True
    raw = data.get("energy_defaults")
    if not isinstance(raw, Mapping):
        return False
    defaults = dict(raw)
    scope_keys = [
        key
        for key in ("physical_scope", "measurement_physical_scope")
        if key in defaults
    ]
    changed = False
    if not scope_keys:
        defaults["physical_scope"] = "FS"
        changed = True
    elif all(defaults.get(key) == "MB" for key in scope_keys):
        for key in scope_keys:
            defaults[key] = "FS"
        changed = True
    if changed:
        data["energy_defaults"] = defaults
    return changed


def _merge_default_hardware_setups(data: dict[str, Any]) -> dict[str, Any]:
    """Ensure the central hardware registry has the three standard setup rows.

    v59b: energy/default saves must never collapse hardware_setups to an empty
    list.  Some earlier code paths wrote only energy_defaults into
    hardware_setups.yaml before the GUI had materialized the default setup rows.
    Merge missing defaults by id while preserving user-entered host/u.RECS data.
    """
    try:
        from onnx_splitpoint_tool.workflow.hardware_matrix import _default_hardware_registry  # type: ignore
        defaults = _default_hardware_registry()
    except Exception:
        defaults = _default_registry()
    existing = list(data.get("hardware_setups") or []) if isinstance(data.get("hardware_setups"), list) else []
    by_id = {str(x.get("id") or ""): x for x in existing if isinstance(x, dict)}
    for raw in list(defaults.get("hardware_setups") or []):
        if not isinstance(raw, dict):
            continue
        sid = str(raw.get("id") or "")
        if sid and sid not in by_id:
            copied = dict(raw)
            existing.append(copied)
            by_id[sid] = copied
            continue
        current = by_id.get(sid)
        if isinstance(current, dict):
            # v2.79.9 migration: add the new power-control defaults without
            # replacing user-entered command, timeout or safety overrides.
            default_power = dict(raw.get("power_control") or {}) if isinstance(raw.get("power_control"), Mapping) else {}
            configured_power = current.get("power_control")
            current_power = dict(configured_power or {}) if isinstance(configured_power, Mapping) else {}
            if default_power and (
                "power_control" not in current
                or isinstance(configured_power, Mapping)
            ):
                merged_power = dict(default_power)
                merged_power.update(current_power)
                current["power_control"] = merged_power
            default_energy = dict(raw.get("energy") or {}) if isinstance(raw.get("energy"), Mapping) else {}
            current_energy = dict(current.get("energy") or {}) if isinstance(current.get("energy"), Mapping) else {}
            if default_energy:
                merged_energy = dict(default_energy)
                merged_energy.update(current_energy)
                current["energy"] = merged_energy
    _migrate_legacy_m2_command_defaults(data)
    _migrate_legacy_energy_scope_default(data)
    data["hardware_setups"] = existing
    data["schema_version"] = max(2, int(data.get("schema_version") or 1))
    data.setdefault("hardware_groups", defaults.get("hardware_groups", {}))
    if not data.get("build_environments") and defaults.get("build_environments"):
        data["build_environments"] = defaults.get("build_environments")
    return data


def _hardware_registry_file_sha256_unlocked(path: Path) -> str:
    """Return the SHA-256 of the exact registry bytes, or ``""`` if absent.

    Callers that use the value as a compare-and-swap token must hold
    ``_hardware_registry_write_lock(path)`` while reading it.
    """

    try:
        payload = path.read_bytes()
    except FileNotFoundError:
        return ""
    return hashlib.sha256(payload).hexdigest()


def hardware_registry_file_sha256(path: str | Path | None = None) -> str:
    """Return an exact-byte revision token for the central registry."""

    p = _expand(path or default_registry_path())
    with _hardware_registry_write_lock(p):
        return _hardware_registry_file_sha256_unlocked(p)


def _read_hardware_registry_unlocked(path: Path) -> tuple[dict[str, Any], bool]:
    """Read and normalize a registry while its caller controls locking.

    The boolean reports whether a narrow legacy M.2 token or physical-scope
    migration must be persisted.  Missing setup/default rows are normalized in
    memory exactly as they were by ``load_hardware_registry`` historically.
    """

    if not path.exists():
        return _merge_default_hardware_setups(_default_registry()), False
    if yaml is None:
        raise RuntimeError("PyYAML is required to read hardware_setups.yaml")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        data = {}
    migration_probe = copy.deepcopy(data)
    m2_migration_needed = _migrate_legacy_m2_command_defaults(migration_probe)
    scope_migration_needed = _migrate_legacy_energy_scope_default(
        migration_probe
    )
    migration_needed = bool(m2_migration_needed or scope_migration_needed)
    data.setdefault("hardware_setups", [])
    data.setdefault("energy_defaults", EnergyDefaults().to_dict())
    return _merge_default_hardware_setups(data), migration_needed


def load_hardware_registry(path: str | Path | None = None) -> dict[str, Any]:
    p = _expand(path or default_registry_path())
    normalized, migration_needed = _read_hardware_registry_unlocked(p)
    if migration_needed:
        # Loading is also the upgrade path for an existing central registry.
        # Use a non-blocking lock so this remains safe when a caller (notably
        # the calibration commit) already owns the registry write lock.  In
        # that case the normalized value is returned and the caller's eventual
        # atomic write persists it.
        persisted = _try_persist_hardware_registry_migration(p)
        if persisted is not None:
            return persisted
    return normalized


@contextmanager
def _hardware_registry_write_lock(path: Path):
    """Serialize registry replacements across Tool processes.

    The hardware registry contains calibrated values and host credentials.  A
    partially written YAML file is therefore worse than a failed save.  Keep a
    separate, stable lock inode and replace the data file atomically while the
    lock is held.
    """

    lock_path = path.with_name(f".{path.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+b")
    try:
        if os.name == "posix":
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        elif os.name == "nt":  # pragma: no cover - Smartmirror runs Linux
            import msvcrt

            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        yield
    finally:
        try:
            if os.name == "posix":
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            elif os.name == "nt":  # pragma: no cover
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        finally:
            handle.close()


def _write_hardware_registry_atomic(payload: Mapping[str, Any], path: Path) -> None:
    rendered = yaml.safe_dump(  # type: ignore[union-attr]
        dict(payload), sort_keys=False, allow_unicode=True
    )
    previous_mode: int | None = None
    try:
        previous_mode = stat.S_IMODE(path.stat().st_mode)
    except FileNotFoundError:
        pass
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, previous_mode if previous_mode is not None else 0o600)
        if path.exists():
            shutil.copystat(path, temporary)
        os.replace(temporary, path)
        if os.name == "posix":
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _try_persist_hardware_registry_migration(path: Path) -> dict[str, Any] | None:
    """Atomically persist a load-time migration if no writer owns the lock.

    ``None`` means persistence was intentionally deferred because another
    writer (possibly this process's outer calibration transaction) owns the
    lock.  The caller still receives the normalized in-memory registry.
    """

    if yaml is None or os.name != "posix":  # pragma: no cover - target is Linux
        return None
    import fcntl

    lock_path = path.with_name(f".{path.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+b")
    locked = False
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked = True
        except BlockingIOError:
            return None
        fresh = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(fresh, dict):
            fresh = {}
        normalized = _merge_default_hardware_setups(fresh)
        normalized.setdefault("schema", "onnx-splitpoint/hardware-setups")
        normalized.setdefault("schema_version", 2)
        normalized.setdefault("hardware_setups", [])
        normalized.setdefault("energy_defaults", EnergyDefaults().to_dict())
        _write_hardware_registry_atomic(normalized, path)
        return normalized
    finally:
        if locked:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def load_hardware_registry_with_revision(
    path: str | Path | None = None,
) -> tuple[dict[str, Any], str]:
    """Load one lock-consistent GUI snapshot and its exact file revision.

    Unlike a separate ``load`` followed by ``stat``/hashing, this operation
    cannot associate an old in-memory payload with a newer file revision.  A
    pending additive M.2-token migration is committed under the same lock, and
    the returned revision always describes the bytes left on disk.
    """

    p = _expand(path or default_registry_path())
    with _hardware_registry_write_lock(p):
        normalized, migration_needed = _read_hardware_registry_unlocked(p)
        if migration_needed:
            normalized.setdefault("schema", "onnx-splitpoint/hardware-setups")
            normalized.setdefault("schema_version", 2)
            normalized.setdefault("hardware_setups", [])
            normalized.setdefault("energy_defaults", EnergyDefaults().to_dict())
            _write_hardware_registry_atomic(normalized, p)
        revision = _hardware_registry_file_sha256_unlocked(p)
    return normalized, revision


def save_hardware_registry(
    registry: Mapping[str, Any],
    path: str | Path | None = None,
    *,
    expected_file_sha256: str | None = None,
) -> Path:
    """Atomically replace the registry, optionally guarded by exact-byte CAS.

    ``expected_file_sha256=None`` preserves the historical unconditional API.
    An empty expected value explicitly means that the caller loaded a missing
    file and will only create it if it is still absent.
    """

    p = _expand(path or default_registry_path())
    p.parent.mkdir(parents=True, exist_ok=True)
    if yaml is None:
        raise RuntimeError("PyYAML is required to write hardware_setups.yaml")
    payload = copy.deepcopy(dict(registry or {}))
    payload.pop(HARDWARE_REGISTRY_REVISION_KEY, None)
    payload.setdefault("schema", "onnx-splitpoint/hardware-setups")
    payload.setdefault("schema_version", 2)
    payload.setdefault("hardware_setups", [])
    payload.setdefault("energy_defaults", EnergyDefaults().to_dict())
    payload = _merge_default_hardware_setups(payload)
    with _hardware_registry_write_lock(p):
        if expected_file_sha256 is not None:
            expected = str(expected_file_sha256)
            current = _hardware_registry_file_sha256_unlocked(p)
            if current != expected:
                raise HardwareRegistryConflictError(p, expected, current)
        _write_hardware_registry_atomic(payload, p)
    return p


def _defaults_from_mapping(raw: Any) -> EnergyDefaults:
    registry_errors: list[str] = []
    if raw is None:
        raw = {}
    elif isinstance(raw, Mapping):
        raw = dict(raw)
    else:
        raw = {}
        registry_errors.append("energy_defaults_not_mapping")
    data_port, data_port_valid = _strict_registry_int(
        raw.get("data_port", 3000), default=3000, minimum=1, maximum=65535
    )
    channel, channel_valid = _strict_registry_int(
        raw.get("channel", 0), default=0, minimum=0
    )
    sample_rate, sample_rate_valid = _strict_registry_int(
        raw.get("sample_rate", 2000), default=2000, minimum=1
    )
    if not data_port_valid:
        registry_errors.append("energy_defaults_data_port_missing_or_invalid")
    if not channel_valid:
        registry_errors.append("energy_defaults_channel_missing_or_invalid")
    if not sample_rate_valid:
        registry_errors.append("energy_defaults_sample_rate_missing_or_invalid")
    if "enabled" in raw and type(raw.get("enabled")) is not bool:
        registry_errors.append("energy_defaults_enabled_not_bool")
    for binary_key in ("collector_binary", "power_calculations_binary"):
        if binary_key not in raw:
            continue
        binary_value = raw.get(binary_key)
        if (
            not isinstance(binary_value, str)
            or not binary_value.strip()
            or binary_value != binary_value.strip()
            or any(ord(ch) < 32 or ord(ch) == 127 for ch in binary_value)
        ):
            registry_errors.append(
                f"energy_defaults_{binary_key}_missing_or_invalid"
            )
    if "mode" in raw:
        mode_value = raw.get("mode")
        if not isinstance(mode_value, str) or not mode_value.strip():
            registry_errors.append("energy_defaults_mode_missing_or_invalid")
        elif mode_value.strip().lower().replace("-", "_") != "fast_firmware":
            registry_errors.append("energy_defaults_mode_not_fast_firmware")
    physical_scope_value = (
        raw.get("physical_scope")
        if "physical_scope" in raw
        else raw.get("measurement_physical_scope", "FS")
    )
    for scope_key in ("physical_scope", "measurement_physical_scope"):
        if scope_key not in raw:
            continue
        scope_value = raw.get(scope_key)
        if (
            not isinstance(scope_value, str)
            or not scope_value
            or scope_value != scope_value.strip()
            or scope_value not in {"AO", "MB", "FS", "FULL_SYSTEM"}
        ):
            registry_errors.append(
                f"energy_defaults_{scope_key}_missing_or_invalid"
            )
    if (
        "physical_scope" in raw
        and "measurement_physical_scope" in raw
        and raw.get("physical_scope") != raw.get("measurement_physical_scope")
    ):
        registry_errors.append("energy_defaults_physical_scope_alias_mismatch")
    window_label_value = (
        raw.get("window_label")
        if "window_label" in raw
        else raw.get("measurement_window", "command")
    )
    for window_key in ("window_label", "measurement_window"):
        if window_key not in raw:
            continue
        window_value = raw.get(window_key)
        if (
            not isinstance(window_value, str)
            or not window_value
            or window_value != window_value.strip()
            or window_value
            not in {"command", "trimmed_activity_window", "complete_measurement_window"}
        ):
            registry_errors.append(
                f"energy_defaults_{window_key}_missing_or_invalid"
            )
    if (
        "window_label" in raw
        and "measurement_window" in raw
        and raw.get("window_label") != raw.get("measurement_window")
    ):
        registry_errors.append("energy_defaults_window_label_alias_mismatch")
    ab_raw = raw.get("window_method_ab") if isinstance(raw.get("window_method_ab"), Mapping) else {}
    legacy_ab = {
        "enabled": raw.get("window_method_validation_probe_enabled"),
        "repeats": raw.get("window_method_validation_probe_repeats"),
        "include_raw_parquet": raw.get("window_method_validation_probe_include_raw_parquet"),
        "strict": raw.get("window_method_validation_probe_strict"),
    }
    ab = resolve_energy_ab_config(ab_raw, legacy_probe=legacy_ab)
    return EnergyDefaults(
        enabled=_bool(raw.get("enabled"), False),
        collector_binary=str(raw.get("collector_binary") or "urecs-data-collector"),
        collector_sha256=str(raw.get("collector_sha256") or ""),
        power_calculations_binary=str(raw.get("power_calculations_binary") or "power_calculations"),
        mode=str(raw.get("mode") or "fast_firmware"),
        data_port=data_port,
        channel=channel,
        sample_rate=sample_rate,
        environment=str(raw.get("environment") or "Jetson"),
        pre_duration_s=_float(raw.get("pre_duration_s"), 5.0),
        post_duration_s=_float(raw.get("post_duration_s"), 5.0),
        duration_margin_s=_float(raw.get("duration_margin_s"), 1.0),
        command_startup_budget_s=max(0.0, min(120.0, _float(raw.get("command_startup_budget_s"), 15.0))),
        min_active_duration_s=_float(raw.get("min_active_duration_s"), 30.0),
        power_estimated_duration_margin_s=_float(raw.get("power_estimated_duration_margin_s"), 2.0),
        measurement_duration_s=_float(raw.get("measurement_duration_s"), _float(raw.get("duration_s"), 60.0)),
        native_measurement_duration_s=_float(raw.get("native_measurement_duration_s"), _float(raw.get("measurement_duration_s"), _float(raw.get("duration_s"), 60.0))),
        power_window_mode=str(raw.get("power_window_mode") or raw.get("measurement_window_mode") or "trimmed"),
        run_count=max(1, _int(raw.get("run_count"), 3)),
        confidence_level=min(0.999, max(0.50, _float(raw.get("confidence_level"), 0.95))),
        physical_scope=str(physical_scope_value),
        window_label=str(window_label_value),
        keep_raw_parquet=_bool(raw.get("keep_raw_parquet"), True),
        postprocess_with_power_calculations=_bool(raw.get("postprocess_with_power_calculations"), True),
        compare_legacy_window=_bool(raw.get("compare_legacy_window"), True),
        window_method_validation_probe_enabled=_bool(
            raw.get("window_method_validation_probe_enabled"), True
        ),
        window_method_validation_probe_repeats=max(
            1, _int(raw.get("window_method_validation_probe_repeats"), 3)
        ),
        window_method_validation_probe_include_raw_parquet=_bool(
            raw.get("window_method_validation_probe_include_raw_parquet"), True
        ),
        window_method_validation_probe_strict=_bool(
            raw.get("window_method_validation_probe_strict"), True
        ),
        window_ab_enabled=_bool(ab.get("enabled"), True),
        window_ab_primary_method=str(ab.get("primary_method") or ENERGY_PRIMARY_METHOD),
        window_ab_shadow_method=str(ab.get("shadow_method") or ENERGY_SHADOW_METHOD),
        window_ab_baseline_method=str(ab.get("baseline_method") or ENERGY_AB_BASELINE_METHOD),
        window_ab_candidate_method=str(ab.get("candidate_method") or ENERGY_AB_CANDIDATE_METHOD),
        window_ab_same_raw_capture=_bool(ab.get("same_raw_capture"), True),
        window_ab_mode=str(ab.get("mode") or "shadow"),
        window_ab_auto_switch=_bool(ab.get("auto_switch"), False),
        window_ab_smoke_repeats=max(1, _int(ab.get("smoke_repeats"), 3)),
        window_ab_include_raw_parquet=_bool(ab.get("include_raw_parquet"), True),
        window_ab_strict=_bool(ab.get("strict"), True),
        window_ab_requires_picoscope=_bool(ab.get("requires_picoscope"), False),
        invalid_repeat_max_retries=max(
            0, min(3, _int(raw.get("invalid_repeat_max_retries"), 1))
        ),
        invalid_repeat_reconnect_backoff_s=max(
            0.0, min(60.0, _float(raw.get("invalid_repeat_reconnect_backoff_s"), 5.0))
        ),
        include_raw_parquet_in_debug_pack=_bool(raw.get("include_raw_parquet_in_debug_pack"), False),
        complete_measurement_window=_bool(raw.get("complete_measurement_window"), True),
        native_energy_duration_s=_float(raw.get("native_energy_duration_s"), 60.0),
        generic_energy_duration_s=_float(raw.get("generic_energy_duration_s"), 60.0),
        registry_contract_valid=not registry_errors,
        registry_contract_errors=tuple(registry_errors),
    )


def energy_defaults_from_registry(registry: Mapping[str, Any] | None) -> EnergyDefaults:
    if not isinstance(registry, Mapping):
        return _defaults_from_mapping(None)
    if "energy_defaults" not in registry:
        return _defaults_from_mapping(None)
    return _defaults_from_mapping(registry.get("energy_defaults"))


def energy_setup_from_raw(raw_setup: Mapping[str, Any] | None) -> EnergySetup:
    registry_errors: list[str] = []
    if raw_setup is None:
        source: dict[str, Any] = {}
    elif isinstance(raw_setup, Mapping):
        source = dict(raw_setup)
    else:
        source = {}
        registry_errors.append("hardware_setup_not_mapping")
    if "energy" not in source:
        raw: dict[str, Any] = {}
    elif isinstance(source.get("energy"), Mapping):
        raw = dict(source.get("energy") or {})
    else:
        raw = {}
        registry_errors.append("setup_energy_not_mapping")
    setup_id, setup_id_valid, setup_id_error = _strict_setup_id(source.get("id"))
    if not setup_id_valid:
        registry_errors.append(setup_id_error)
    identity = resolve_jetson_identity(source)
    data_port, data_port_valid = _strict_registry_int(
        raw.get("data_port", 3000), default=0, minimum=1, maximum=65535
    )
    if not data_port_valid:
        registry_errors.append("setup_data_port_missing_or_invalid")
    if "enabled" in raw and type(raw.get("enabled")) is not bool:
        registry_errors.append("setup_energy_enabled_not_bool")
    urecs_address, urecs_valid, urecs_errors = _strict_urecs_address_fields(raw)
    if not urecs_valid:
        registry_errors.extend(urecs_errors)
    accelerator_idle_w, accelerator_idle_w_valid = (
        _strict_optional_registry_number(raw, "accelerator_idle_w")
    )
    if not accelerator_idle_w_valid:
        registry_errors.append("setup_accelerator_idle_w_missing_or_invalid")
    (
        full_system_current_scale_factor,
        full_system_current_scale_calibrated_at,
        full_system_current_scale_calibration_evidence,
        full_system_current_scale_calibration_sha256,
    ) = _parse_full_system_current_scale_fields(raw, registry_errors)
    return EnergySetup(
        setup_id=setup_id,
        accelerator=str(source.get("accelerator") or ""),
        jetson_address=str(identity["address"]),
        jetson_user=str(identity["user"]),
        jetson_port=int(identity["port"]),
        jetson_ssh_extra_args=str(identity["ssh_extra_args"]),
        jetson_identity_valid=bool(identity["valid"]),
        jetson_identity_errors=tuple(identity["errors"]),
        enabled=_bool(raw.get("enabled"), False),
        urecs_address=urecs_address,
        data_port=data_port,
        data_port_valid=data_port_valid,
        idle_baseline_w=_float_or_none(raw.get("idle_baseline_w")),
        accelerator_idle_w=accelerator_idle_w,
        accelerator_idle_calibrated_at=str(raw.get("accelerator_idle_calibrated_at") or ""),
        accelerator_idle_calibration_evidence=str(raw.get("accelerator_idle_calibration_evidence") or ""),
        accelerator_idle_calibration_binding_path=str(raw.get("accelerator_idle_calibration_binding_path") or ""),
        accelerator_idle_calibration_binding_sha256=str(raw.get("accelerator_idle_calibration_binding_sha256") or "").strip().lower(),
        full_system_current_scale_factor=full_system_current_scale_factor,
        full_system_current_scale_calibrated_at=(
            full_system_current_scale_calibrated_at
        ),
        full_system_current_scale_calibration_evidence=(
            full_system_current_scale_calibration_evidence
        ),
        full_system_current_scale_calibration_sha256=(
            full_system_current_scale_calibration_sha256
        ),
        calibration_manifest=str(raw.get("calibration_manifest") or ""),
        calibration_sha256=str(raw.get("calibration_sha256") or "").strip().lower(),
        registry_contract_valid=not registry_errors,
        registry_contract_errors=tuple(registry_errors),
    )


def setup_from_mapping(raw: Mapping[str, Any] | None) -> EnergySetup:
    registry_errors: list[str] = []
    if raw is None:
        source: dict[str, Any] = {}
    elif isinstance(raw, Mapping):
        source = dict(raw)
    else:
        source = {}
        registry_errors.append("energy_setup_not_mapping")
    setup_id_source = (
        source.get("setup_id")
        if "setup_id" in source
        else source.get("id")
    )
    setup_id, setup_id_valid, setup_id_error = _strict_setup_id(
        setup_id_source
    )
    if not setup_id_valid:
        registry_errors.append(setup_id_error)
    data_port, strict_data_port_valid = _strict_registry_int(
        source.get("data_port", 3000), default=0, minimum=1, maximum=65535
    )
    declared_data_port_valid = source.get("data_port_valid", True) is True
    data_port_valid = bool(strict_data_port_valid and declared_data_port_valid)
    if not data_port_valid:
        registry_errors.append("setup_data_port_missing_or_invalid")
    if "enabled" in source and type(source.get("enabled")) is not bool:
        registry_errors.append("setup_energy_enabled_not_bool")
    urecs_address, urecs_valid, urecs_errors = _strict_urecs_address_fields(
        source
    )
    if not urecs_valid:
        registry_errors.extend(urecs_errors)
    (
        full_system_current_scale_factor,
        full_system_current_scale_calibrated_at,
        full_system_current_scale_calibration_evidence,
        full_system_current_scale_calibration_sha256,
    ) = _parse_full_system_current_scale_fields(source, registry_errors)
    return EnergySetup(
        setup_id=setup_id,
        accelerator=str(source.get("accelerator") or ""),
        jetson_address=str(source.get("jetson_address") or ""),
        jetson_user=str(source.get("jetson_user") or ""),
        jetson_port=_int(source.get("jetson_port"), 22),
        jetson_ssh_extra_args=normalise_ssh_extra_args(
            source.get("jetson_ssh_extra_args") or ""
        ),
        jetson_identity_valid=_bool(
            source.get("jetson_identity_valid"), True
        ),
        jetson_identity_errors=tuple(
            str(value)
            for value in list(source.get("jetson_identity_errors") or [])
        ),
        enabled=_bool(source.get("enabled"), False),
        urecs_address=urecs_address,
        data_port=data_port,
        data_port_valid=data_port_valid,
        idle_baseline_w=_float_or_none(source.get("idle_baseline_w")),
        accelerator_idle_w=_float_or_none(source.get("accelerator_idle_w")),
        accelerator_idle_calibrated_at=str(source.get("accelerator_idle_calibrated_at") or ""),
        accelerator_idle_calibration_evidence=str(source.get("accelerator_idle_calibration_evidence") or ""),
        accelerator_idle_calibration_binding_path=str(source.get("accelerator_idle_calibration_binding_path") or ""),
        accelerator_idle_calibration_binding_sha256=str(source.get("accelerator_idle_calibration_binding_sha256") or "").strip().lower(),
        full_system_current_scale_factor=full_system_current_scale_factor,
        full_system_current_scale_calibrated_at=(
            full_system_current_scale_calibrated_at
        ),
        full_system_current_scale_calibration_evidence=(
            full_system_current_scale_calibration_evidence
        ),
        full_system_current_scale_calibration_sha256=(
            full_system_current_scale_calibration_sha256
        ),
        calibration_manifest=str(source.get("calibration_manifest") or ""),
        calibration_sha256=str(source.get("calibration_sha256") or "").strip().lower(),
        registry_contract_valid=not registry_errors,
        registry_contract_errors=tuple(registry_errors),
    )


def hardware_registry_snapshot_sha256(
    registry: Mapping[str, Any] | None,
) -> str:
    """Hash the exact normalized registry snapshot used by a claim path.

    The private GUI CAS revision is file metadata rather than registry
    content and is therefore excluded. All other fields remain bound: a
    change to any setup, method reference, endpoint or policy invalidates the
    projected EnergySetup before acquisition starts.
    """

    payload = dict(registry or {}) if isinstance(registry, Mapping) else {}
    payload.pop(HARDWARE_REGISTRY_REVISION_KEY, None)
    digest = sha256_json(payload)
    return (
        digest.split(":", 1)[1]
        if isinstance(digest, str) and digest.startswith("sha256:")
        else str(digest or "")
    )


def energy_setup_from_registry(
    registry: Mapping[str, Any],
    setup_id: str,
    *,
    registry_path: str | Path | None = None,
) -> EnergySetup:
    provenance_path = ""
    provenance_errors: list[str] = []
    if registry_path not in (None, ""):
        raw_path = _expand(registry_path)  # type: ignore[arg-type]
        try:
            canonical_path = raw_path.resolve(strict=True)
        except OSError:
            canonical_path = raw_path.expanduser().resolve(strict=False)
            provenance_errors.append("hardware_registry_path_unreadable")
        if not canonical_path.is_file():
            provenance_errors.append("hardware_registry_path_not_regular_file")
        if raw_path.is_symlink():
            provenance_errors.append("hardware_registry_path_is_symlink")
        provenance_path = str(canonical_path)
    else:
        provenance_errors.append("hardware_registry_path_missing")
    provenance_sha256 = hardware_registry_snapshot_sha256(registry)
    if re.fullmatch(r"[0-9a-f]{64}", provenance_sha256) is None:
        provenance_errors.append("hardware_registry_snapshot_sha256_invalid")

    def _bind_registry_provenance(setup: EnergySetup) -> EnergySetup:
        setup.hardware_registry_path = provenance_path
        setup.hardware_registry_snapshot_sha256 = provenance_sha256
        setup.hardware_registry_provenance_errors = tuple(
            dict.fromkeys(provenance_errors)
        )
        setup.hardware_registry_provenance_valid = not (
            setup.hardware_registry_provenance_errors
        )
        return setup

    requested, requested_valid, requested_error = _strict_setup_id(setup_id)
    if not requested_valid:
        return _bind_registry_provenance(EnergySetup(
            setup_id="",
            registry_contract_valid=False,
            registry_contract_errors=(f"requested_{requested_error}",),
        ))
    matches = [
        raw
        for raw in list((registry or {}).get("hardware_setups") or [])
        if isinstance(raw, Mapping)
        and _strict_setup_id(raw.get("id"))[:2] == (requested, True)
    ]
    if len(matches) > 1:
        raise DuplicateEnergySetupIdError(requested)
    if matches:
        setup = energy_setup_from_raw(matches[0])
        registry_errors = list(setup.registry_contract_errors)
        defaults_value = (registry or {}).get("energy_defaults")
        if "energy_defaults" in (registry or {}) and not isinstance(
            defaults_value, Mapping
        ):
            defaults_raw: dict[str, Any] = {}
            registry_errors.append("energy_defaults_not_mapping")
        else:
            defaults_raw = dict(defaults_value or {})
        default_port, default_port_valid = _strict_registry_int(
            defaults_raw.get("data_port", 3000),
            default=0,
            minimum=1,
            maximum=65535,
        )
        if not default_port_valid:
            registry_errors.append("energy_defaults_data_port_missing_or_invalid")
        default_channel, default_channel_valid = _strict_registry_int(
            defaults_raw.get("channel", 0), default=0, minimum=0
        )
        default_rate, default_rate_valid = _strict_registry_int(
            defaults_raw.get("sample_rate", 2000), default=0, minimum=1
        )
        selected_energy = (
            dict(matches[0].get("energy") or {})
            if isinstance(matches[0].get("energy"), Mapping)
            else {}
        )
        registry_errors.extend(
            _setup_claim_override_errors(
                selected_energy,
                default_collector_binary=defaults_raw.get(
                    "collector_binary", "urecs-data-collector"
                ),
                default_power_binary=defaults_raw.get(
                    "power_calculations_binary", "power_calculations"
                ),
                default_mode=defaults_raw.get("mode", "fast_firmware"),
            )
        )
        if "data_port" in selected_energy:
            setup.data_port_valid = bool(
                setup.data_port_valid
                and default_port_valid
                and setup.data_port == default_port
            )
            if (
                type(setup.data_port) is int
                and default_port_valid
                and setup.data_port != default_port
            ):
                registry_errors.append("setup_data_port_defaults_mismatch")
        else:
            setup.data_port = default_port
            setup.data_port_valid = default_port_valid
        for setup_key, default_value, default_valid, invalid_reason, mismatch_reason in (
            (
                "channel",
                default_channel,
                default_channel_valid,
                "setup_channel_missing_or_invalid",
                "setup_channel_defaults_mismatch",
            ),
            (
                "sample_rate",
                default_rate,
                default_rate_valid,
                "setup_sample_rate_missing_or_invalid",
                "setup_sample_rate_defaults_mismatch",
            ),
            (
                "sample_rate_hz",
                default_rate,
                default_rate_valid,
                "setup_sample_rate_hz_missing_or_invalid",
                "setup_sample_rate_hz_defaults_mismatch",
            ),
        ):
            if setup_key not in selected_energy:
                continue
            minimum = 0 if setup_key == "channel" else 1
            setup_value, setup_value_valid = _strict_registry_int(
                selected_energy.get(setup_key),
                default=0,
                minimum=minimum,
            )
            if not setup_value_valid or not default_valid:
                registry_errors.append(invalid_reason)
            elif setup_value != default_value:
                registry_errors.append(mismatch_reason)
        if (
            "sample_rate" in selected_energy
            and "sample_rate_hz" in selected_energy
            and selected_energy.get("sample_rate")
            != selected_energy.get("sample_rate_hz")
        ):
            registry_errors.append("setup_sample_rate_alias_mismatch")
        setup.registry_contract_errors = tuple(dict.fromkeys(registry_errors))
        setup.registry_contract_valid = not setup.registry_contract_errors
        (
            setup.expected_channel_bindings,
            setup.expected_channel_bindings_valid,
            setup.expected_channel_binding_errors,
        ) = _expected_channel_bindings_from_registry(
            registry,
            selected_setup_id=requested,
        )
        return _bind_registry_provenance(setup)
    return _bind_registry_provenance(EnergySetup(setup_id=requested))


def load_energy_defaults(path: str | Path | None = None) -> EnergyDefaults:
    # Prefer dedicated energy_config.yaml when it exists; otherwise use the central hardware registry.
    p = _expand(path or default_energy_config_file())
    if p.exists() and yaml is not None:
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        if isinstance(data, Mapping):
            return _defaults_from_mapping(
                data.get("energy_defaults")
                if "energy_defaults" in data
                else data
            )
    return energy_defaults_from_registry(load_hardware_registry())


def save_energy_defaults(defaults: EnergyDefaults | Mapping[str, Any], path: str | Path | None = None) -> Path:
    d = defaults.to_dict() if isinstance(defaults, EnergyDefaults) else _defaults_from_mapping(defaults).to_dict()
    # Keep a small standalone file and mirror into the hardware registry.
    p = _expand(path or default_energy_config_file())
    p.parent.mkdir(parents=True, exist_ok=True)
    if yaml is None:
        raise RuntimeError("PyYAML is required to write energy_config.yaml")
    registry_path = _expand(default_registry_path())
    if registry_path == p:
        raise ValueError("energy defaults file and hardware registry must be distinct")
    standalone = {
        "schema": "onnx-splitpoint/energy-config",
        "schema_version": 1,
        "energy_defaults": d,
    }
    # Do not perform a load/unlock/save sequence here: a completed platform
    # calibration could otherwise be overwritten by that stale full-registry
    # snapshot.  Re-read and modify only energy_defaults while holding the
    # central registry's stable write lock.  Keep that lock across the
    # standalone mirror write too, so two simultaneous defaults saves cannot
    # leave the mirror and central registry at opposite generations.
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with _hardware_registry_write_lock(registry_path):
        # The standalone mirror gets the same fsync + atomic-replace treatment
        # as the registry, so readers never observe partially rendered YAML.
        with _hardware_registry_write_lock(p):
            _write_hardware_registry_atomic(standalone, p)
        reg, _migration_needed = _read_hardware_registry_unlocked(registry_path)
        reg["energy_defaults"] = copy.deepcopy(d)
        reg.setdefault("schema", "onnx-splitpoint/hardware-setups")
        reg.setdefault("schema_version", 2)
        reg.setdefault("hardware_setups", [])
        _write_hardware_registry_atomic(reg, registry_path)
    return p


def get_setup_energy(setup_id: str, registry_path: str | Path | None = None) -> EnergySetup:
    selected_input = _expand(registry_path or default_registry_path())
    return energy_setup_from_registry(
        load_hardware_registry(selected_input),
        setup_id,
        registry_path=selected_input,
    )


REVIEWED_COLLECTOR_SHA256 = "913c3f745a71809d85c1e98f56d0ca3265bb5e5492ad2b72407f7ce9bd8c3a46"


def resolve_collector_binding(registry: Mapping[str, Any]) -> dict[str, str]:
    """Resolve once, including bytes; execution must use this absolute path."""
    from .collector import _expand_binary
    configured = energy_defaults_from_registry(registry).collector_binary
    path = Path(_expand_binary(configured)).expanduser().absolute()
    digest = ""
    if path.is_file():
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
    expected = str((registry.get("energy_defaults") or {}).get("collector_sha256") or "")
    if expected and digest and expected != digest:
        raise ValueError("Native collector bytes differ from configured binding: " + str(path))
    return {"collector_binary": str(path), "collector_sha256": digest,
            "collector_source": configured}


def install_reviewed_collector(source: str | Path, backup_dir: str | Path) -> dict[str, Any]:
    """Explicit, idempotent normal config migration. Never build or fall back."""
    source = Path(source).expanduser().resolve()
    with source.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    if digest != REVIEWED_COLLECTOR_SHA256:
        raise ValueError("Reviewed collector SHA256 mismatch")
    registry_path = Path(default_registry_path()).expanduser().resolve()
    mirror_path = default_energy_config_file().expanduser().resolve()
    destination = registry_path.parent / "collectors" / "r6-reviewed" / "urecs-data-collector"
    backup = Path(backup_dir).expanduser().resolve()
    backup.mkdir(parents=True, exist_ok=True)
    changes = []
    with _hardware_registry_write_lock(registry_path):
        with _hardware_registry_write_lock(mirror_path):
            payloads = []
            for path in (registry_path, mirror_path):
                if not path.is_file():
                    continue
                raw = yaml.safe_load(path.read_text())
                defaults = raw.get("energy_defaults") or {}
                old = str(defaults.get("collector_binary") or "urecs-data-collector")
                if old not in {"urecs-data-collector", str(Path.home()/".cargo/bin/urecs-data-collector"), str(destination)}:
                    raise ValueError("Explicit collector binding conflict: " + old)
                for setup in raw.get("hardware_setups", []):
                    if setup.get("enabled", True) and "collector_binary" in (setup.get("energy") or {}):
                        raise ValueError("Setup collector override conflict: " + str(setup.get("id")))
                after = copy.deepcopy(raw)
                after.setdefault("energy_defaults", {}).update(collector_binary=str(destination), collector_sha256=digest)
                if after != raw:
                    saved = backup / path.name
                    if saved.exists():
                        raise ValueError("Backup already exists for a different migration: " + str(saved))
                    shutil.copy2(path, saved)
                    payloads.append((path, after))
                    changes.append({"file": str(path), "before": defaults,
                                    "after": after["energy_defaults"], "backup": str(saved)})
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                with destination.open("rb") as handle:
                    if hashlib.file_digest(handle, "sha256").hexdigest() != digest:
                        raise ValueError("Managed collector already exists with different bytes")
            else:
                temporary = destination.with_suffix(".tmp")
                shutil.copy2(source, temporary)
                os.replace(temporary, destination)
            for path, after in payloads:
                _write_hardware_registry_atomic(after, path)
    return {"collector_binary": str(destination), "collector_sha256": digest,
            "source": str(source), "changes": changes}

# v60m: the top-level energy switch is authoritative for all native energy paths.
from onnx_splitpoint_tool.v60m_policy import install_energy_object_guards as _v60m_install_energy_guards
_v60m_install_energy_guards(globals())
