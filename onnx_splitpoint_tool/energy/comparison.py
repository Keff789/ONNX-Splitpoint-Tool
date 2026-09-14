"""Fail-closed energy comparison semantics.

Raw u.RECS values are acquisition evidence and are never overwritten here.
Only an exact TensorRT Full row may use the separately recorded host-normalized
estimate, and only when its identity and accelerator-idle calibration were
verified by the collector.  Current calibrations use one ordinary JSON record;
the historical immutable binding remains readable for existing installations.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shlex
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


HOST_NORMALIZATION_ROLE_NONE = "none"
HOST_NORMALIZATION_ROLE_TENSORRT_FULL = "tensorrt_full"
TENSORRT_FULL_SOURCE_RUN_IDS = frozenset(
    {"native_full_tensorrt", "ort_tensorrt", "tensorrt", "tensorrt_full", "trt_full"}
)
CALIBRATION_BINDING_SCHEMA = "onnx-splitpoint/accelerator-idle-calibration-binding"
CALIBRATION_BINDING_SCHEMA_VERSION = 2
SIMPLE_CALIBRATION_EVIDENCE_SCHEMA = "onnx-splitpoint/m2-idle-power-calibration"
ENERGY_CALIBRATION_MANIFEST_SCHEMA = "onnx-splitpoint/energy-calibration-manifest"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")

_CALIBRATION_BINDING_V2_FIELDS = frozenset(
    {
        "schema", "schema_version", "setup_id", "accelerator",
        "urecs_address", "data_port", "jetson_host", "started_at",
        "finished_at", "idle_power_without_m2_w",
        "idle_power_with_m2_w", "accelerator_idle_power_w",
        "restored_m2_on", "status", "energy_calibration_manifest",
        "calibration_evidence", "captures", "state_observations",
        "transitions", "binding_payload_sha256",
    }
)
_CALIBRATION_EVIDENCE_V1_FIELDS = frozenset(
    {
        "schema", "schema_version", "setup_id", "accelerator",
        "urecs_address", "data_port", "jetson_host", "started_at",
        "finished_at", "idle_power_without_m2_w",
        "idle_power_with_m2_w", "accelerator_idle_power_w",
        "restored_m2_on", "status", "energy_calibration_manifest",
        "captures", "state_observations", "transitions",
    }
)
_COMMAND_WINDOW_BINDING_V2_FIELDS = frozenset(
    {
        "schema", "schema_version", "binding_method", "run_id",
        "window_id", "source", "request_path", "request_sha256",
        "marker_path", "marker_sha256", "trace_path", "trace_sha256",
        "command_path", "command_sha256", "workload_command_path",
        "workload_command_sha256", "timing_path", "timing_sha256",
        "postprocessor_result_path", "postprocessor_result_sha256",
        "result_path", "result_sha256", "first_sample_index",
        "last_sample_index", "sample_count", "interval_count",
        "sample_rate_hz", "duration_s", "process_rc",
        "command_process_id", "command_start_realtime_ns",
        "command_end_realtime_ns", "command_start_monotonic_ns",
        "command_end_monotonic_ns",
        "command_start_clock_read_uncertainty_ns",
        "command_end_clock_read_uncertainty_ns", "dropped_samples",
        "boundary_uncertainty_samples",
        "maximum_boundary_uncertainty_samples", "trace_covers_window",
        "energy_semantics", "energy_field", "energy_j",
        "created_at_unix_ns",
    }
)

_COMMAND_WINDOW_REQUEST_V2_FIELDS = frozenset(
    {
        "schema", "schema_version", "binding_method", "run_id",
        "window_id", "source", "marker_path", "marker_sha256",
        "trace_path", "trace_sha256", "command_path", "command_sha256",
        "workload_command_path", "workload_command_sha256", "timing_path",
        "timing_sha256", "first_sample_index", "last_sample_index",
        "sample_count", "interval_count", "sample_rate_hz", "duration_s",
        "process_rc", "process_id", "process_error",
        "start_clock_read_uncertainty_ns", "end_clock_read_uncertainty_ns",
        "command_start_realtime_ns", "command_end_realtime_ns",
        "command_start_monotonic_ns", "command_end_monotonic_ns",
        "dropped_samples", "boundary_uncertainty_samples",
        "maximum_boundary_uncertainty_samples", "trace_covers_window",
        "valid_for_final_energy", "total_samples", "index_semantics",
        "energy_semantics", "energy_field", "created_at_unix_ns",
    }
)
_COMMAND_WINDOW_MARKER_V2_FIELDS = frozenset(
    {
        "schema", "schema_version", "run_id", "window_id", "command",
        "stream", "valid_for_final_energy", "validation_errors",
    }
)
_COMMAND_WINDOW_MARKER_COMMAND_V2_FIELDS = frozenset(
    {
        "text", "argv", "process_id", "process_rc", "process_error",
        "start_realtime_ns", "end_realtime_ns", "start_monotonic_ns",
        "end_monotonic_ns", "start_clock_read_uncertainty_ns",
        "end_clock_read_uncertainty_ns",
    }
)
_COMMAND_WINDOW_MARKER_STREAM_V2_FIELDS = frozenset(
    {
        "source", "trace_path", "sample_rate_hz", "first_sample_index",
        "last_sample_index", "total_samples", "boundary_uncertainty_samples",
        "dropped_samples", "trace_covers_window",
    }
)
_POSTPROCESSOR_WINDOW_RESULT_V2_FIELDS = frozenset(
    {
        "schema", "schema_version", "status", "run_id", "window_id",
        "source", "request_path", "marker_path", "trace_path",
        "command_path", "timing_path", "results_path", "request_sha256",
        "marker_sha256", "trace_sha256", "command_sha256", "timing_sha256",
        "first_sample_index", "last_sample_index", "sample_count",
        "interval_count", "sample_rate_hz", "duration_s", "process_rc",
        "command_process_id", "command_start_realtime_ns",
        "command_end_realtime_ns", "command_start_monotonic_ns",
        "command_end_monotonic_ns",
        "command_start_clock_read_uncertainty_ns",
        "command_end_clock_read_uncertainty_ns", "drop_count",
        "boundary_uncertainty_samples",
        "maximum_boundary_uncertainty_samples", "trace_covers_window",
        "energy_semantics", "energy_field", "energy_j",
    }
)


def _get(source: Any, name: str, default: Any = None) -> Any:
    if isinstance(source, Mapping):
        return source.get(name, default)
    return getattr(source, name, default)


def _finite(value: Any, *, positive: bool = False) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(result) or (positive and result <= 0.0):
        return None
    return result


def _strict_finite_number(
    value: Any,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> float | None:
    if type(value) not in (int, float):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if (
        not math.isfinite(result)
        or (positive and result <= 0.0)
        or (nonnegative and result < 0.0)
    ):
        return None
    return result


def _normalise_accelerator(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip(
        "_"
    )


def _normalise_sha256(value: Any) -> str:
    text = str(value or "").strip().lower()
    return text.split(":", 1)[1] if text.startswith("sha256:") else text


def _normalise_ssh_extra_args(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        parts = shlex.split(text)
    except (TypeError, ValueError):
        parts = text.split()
    return shlex.join(parts)


def _canonical_identity_token(value: Any) -> bool:
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


def _first_finite(row: Mapping[str, Any], names: tuple[str, ...], *, positive: bool = False) -> float | None:
    for name in names:
        value = _finite(row.get(name), positive=positive)
        if value is not None:
            return value
    return None


def _strict_coherent_finite(
    row: Mapping[str, Any],
    names: tuple[str, ...],
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> tuple[float | None, bool]:
    """Resolve one true alias family without coercion or rebinding.

    ``_first_finite`` remains intentionally permissive for reading historical
    diagnostic rows. Claim admission must not inherit that coercion, nor may a
    conflicting duplicate alias silently lose to whichever alias happens to be
    listed first.  The boolean return value describes exact/coherent admission;
    an entirely absent family is exact but has no resolved value.
    """

    populated = [
        row.get(name)
        for name in names
        if name in row and row.get(name) is not None
    ]
    if not populated:
        return None, True
    values = [
        _strict_finite_number(
            value,
            positive=positive,
            nonnegative=nonnegative,
        )
        for value in populated
    ]
    if any(value is None for value in values):
        return None, False
    resolved = float(values[0])
    coherent = all(
        math.isclose(
            resolved,
            float(value),
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        for value in values[1:]
    )
    return (resolved if coherent else None), coherent


def _all_populated_literal_true(
    row: Mapping[str, Any], names: tuple[str, ...]
) -> bool:
    values = [
        row.get(name)
        for name in names
        if name in row and row.get(name) is not None
    ]
    return bool(values) and all(value is True for value in values)


def _arithmetic_close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-12)


def _uncertainty_is_valid(
    point: float | None,
    sample_stddev: float | None,
    ci_low: float | None,
    ci_high: float | None,
    *,
    aliases_coherent: bool,
    stddev_populated: bool,
    ci_low_populated: bool,
    ci_high_populated: bool,
) -> bool:
    """Validate an optional uncertainty block for claim admission."""

    if not aliases_coherent:
        return False
    uncertainty_populated = bool(
        stddev_populated or ci_low_populated or ci_high_populated
    )
    if not uncertainty_populated:
        return True
    if not (stddev_populated and ci_low_populated and ci_high_populated):
        return False
    return bool(
        point is not None
        and sample_stddev is not None
        and ci_low is not None
        and ci_high is not None
        and ci_low <= point <= ci_high
    )


def _has_populated_metric(
    row: Mapping[str, Any], names: tuple[str, ...]
) -> bool:
    return any(name in row and row.get(name) is not None for name in names)


def _canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    """Hash one JSON mapping independently of its presentation on disk."""

    unsigned = {
        key: value
        for key, value in payload.items()
        if key != "binding_payload_sha256"
    }
    try:
        rendered = json.dumps(
            unsigned,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError):
        return ""
    return hashlib.sha256(rendered).hexdigest()


def _canonical_local_file(
    raw_path: Any,
    raw_sha256: Any,
    *,
    contained_by: Path | None = None,
) -> tuple[Path | None, bytes | None, str | None]:
    """Read one immutable local artefact identity, rejecting path aliases."""

    if (
        not isinstance(raw_path, str)
        or not raw_path
        or raw_path != raw_path.strip()
    ):
        return None, None, "path_missing"
    path_text = raw_path
    if (
        not isinstance(raw_sha256, str)
        or raw_sha256 != raw_sha256.strip().lower()
        or _SHA256_RE.fullmatch(raw_sha256) is None
    ):
        return None, None, "sha256_invalid"
    expected_sha = raw_sha256
    if "://" in path_text:
        return None, None, "path_not_local"
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        return None, None, "path_not_absolute"
    try:
        resolved = path.resolve(strict=True)
    except OSError:
        return None, None, "file_missing"
    if resolved != path or path.is_symlink() or not resolved.is_file():
        return None, None, "path_not_canonical_regular_file"
    if contained_by is not None:
        try:
            resolved.relative_to(contained_by)
        except ValueError:
            return None, None, "path_outside_calibration_root"
    try:
        payload_bytes = resolved.read_bytes()
    except OSError:
        return None, None, "file_unreadable"
    if hashlib.sha256(payload_bytes).hexdigest() != expected_sha:
        return None, None, "sha256_mismatch"
    return resolved, payload_bytes, None


def _artifact_identity(
    source: Any,
    *,
    contained_by: Path | None = None,
    expected_fields: frozenset[str] = frozenset({"path", "sha256"}),
) -> tuple[Path | None, bytes | None, str | None]:
    if not isinstance(source, Mapping):
        return None, None, "identity_missing"
    if set(source) != expected_fields:
        return None, None, "identity_fields_mismatch"
    return _canonical_local_file(
        source.get("path"),
        source.get("sha256"),
        contained_by=contained_by,
    )


def _json_mapping(payload_bytes: bytes | None) -> Mapping[str, Any] | None:
    if payload_bytes is None:
        return None
    try:
        payload = json.loads(payload_bytes.decode("utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, Mapping) else None


def _structured_mapping(path: Path) -> Mapping[str, Any] | None:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        value = json.loads(text)
    except Exception:
        try:
            import yaml  # type: ignore

            value = yaml.safe_load(text)
        except Exception:
            return None
    return value if isinstance(value, Mapping) else None


def _timing_mapping(path: Path) -> Mapping[str, Any] | None:
    try:
        rows = [
            line.split("=", 1)
            for line in path.read_text(encoding="utf-8").splitlines()
        ]
        if (
            any(len(row) != 2 for row in rows)
            or len(rows) != 3
            or len({row[0] for row in rows}) != 3
        ):
            return None
        values = dict(rows)
        if set(values) != {"start_ns", "end_ns", "rc"}:
            return None
        if any(
            not isinstance(value, str)
            or re.fullmatch(r"[0-9]+", value) is None
            for value in values.values()
        ):
            return None
        return {
            "start_ns": int(values["start_ns"]),
            "end_ns": int(values["end_ns"]),
            "rc": int(values["rc"]),
        }
    except Exception:
        return None


def _canonical_utc_timestamp(value: Any) -> datetime | None:
    """Accept only the exact UTC form emitted by ``datetime.isoformat``."""

    if not isinstance(value, str) or not value or value != value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(None):
        return None
    if not value.endswith("+00:00") or parsed.isoformat() != value:
        return None
    return parsed


def _is_parquet_file(payload: bytes | None) -> bool:
    if (
        not isinstance(payload, bytes)
        or len(payload) < 13
        or payload[:4] != b"PAR1"
        or payload[-4:] != b"PAR1"
    ):
        return False
    # Parquet places a little-endian metadata length immediately before the
    # trailing magic.  Checking that footer boundary rejects arbitrary bytes
    # merely wrapped in PAR1 while remaining independent of optional pyarrow.
    metadata_size = int.from_bytes(payload[-8:-4], "little", signed=False)
    return 0 < metadata_size <= len(payload) - 12


def _configured_calibration_delta_policy(
    registry: Mapping[str, Any] | None,
    setup_id: str,
) -> tuple[bool, float, list[str]]:
    """Resolve the persisted positive-delta policy without lossy coercion."""

    required = True
    minimum = 0.02
    failures: list[str] = []
    if registry is None:
        return required, minimum, failures
    rows = registry.get("hardware_setups")
    if not isinstance(rows, list):
        return required, minimum, failures
    matches = [
        row for row in rows
        if isinstance(row, Mapping) and row.get("id") == setup_id
    ]
    if len(matches) != 1:
        return required, minimum, failures
    raw_power = matches[0].get("power_control")
    if raw_power is None:
        return required, minimum, failures
    if not isinstance(raw_power, Mapping):
        return required, minimum, ["power_control_not_mapping"]
    if "require_positive_calibration_delta" in raw_power:
        value = raw_power.get("require_positive_calibration_delta")
        if type(value) is not bool:
            failures.append("require_positive_calibration_delta_invalid")
        else:
            required = value
    if "minimum_calibration_delta_w" in raw_power:
        parsed = _strict_finite_number(
            raw_power.get("minimum_calibration_delta_w")
        )
        if parsed is None or parsed < 0.0:
            failures.append("minimum_calibration_delta_w_invalid")
        else:
            minimum = parsed
    return required, minimum, failures


def _same_number(left: Any, right: Any, *, tolerance: float = 1e-9) -> bool:
    a = _finite(left)
    b = _finite(right)
    return bool(
        a is not None
        and b is not None
        and math.isclose(a, b, rel_tol=0.0, abs_tol=tolerance)
    )


def _has_exact_int_fields(
    payload: Mapping[str, Any], fields: tuple[str, ...]
) -> bool:
    return all(type(payload.get(field)) is int for field in fields)


def _verify_simple_accelerator_idle_calibration(
    setup: Any,
    *,
    legacy_binding_configured: bool,
) -> dict[str, Any] | None:
    """Verify the current, deliberately small calibration JSON contract.

    The evidence file is not sealed and has no digest.  It records the two
    observed full-system means and the value atomically saved in the hardware
    registry.  This helper therefore checks only identity, state, timestamps
    and arithmetic.  Returning ``None`` selects the historical binding verifier
    for an existing pre-v2.79.14 registry.
    """

    evidence_raw = _get(setup, "accelerator_idle_calibration_evidence", "")
    if evidence_raw in (None, ""):
        return None
    evidence_text = evidence_raw if isinstance(evidence_raw, str) else ""
    calibrated_at_raw = _get(setup, "accelerator_idle_calibrated_at", "")
    calibrated_at_text = (
        calibrated_at_raw if isinstance(calibrated_at_raw, str) else ""
    )
    base: dict[str, Any] = {
        "accelerator_idle_calibration_verified": False,
        "accelerator_idle_calibration_status": (
            "unavailable_simple_evidence_path_invalid"
        ),
        "accelerator_idle_calibration_evidence": evidence_text,
        "accelerator_idle_calibrated_at": calibrated_at_text,
        "accelerator_idle_calibration_mode": "simple_json",
    }
    if (
        not evidence_text
        or evidence_text != evidence_text.strip()
        or "\x00" in evidence_text
    ):
        return base

    path = Path(evidence_text).expanduser()
    # Historical calibrations stored the calibration *directory* in this field
    # and the immutable JSON binding in separate path/SHA fields.  Do not make
    # that old layout look like a malformed current record.
    if legacy_binding_configured and path.suffix.lower() != ".json":
        return None
    if not path.is_absolute():
        return base
    try:
        resolved_path = path.resolve(strict=True)
    except OSError:
        base["accelerator_idle_calibration_status"] = (
            "unavailable_simple_evidence_unreadable"
        )
        return base
    if resolved_path != path or path.is_symlink() or not resolved_path.is_file():
        return base
    try:
        payload = json.loads(resolved_path.read_text(encoding="utf-8"))
    except OSError:
        base["accelerator_idle_calibration_status"] = (
            "unavailable_simple_evidence_unreadable"
        )
        return base
    except Exception:
        base["accelerator_idle_calibration_status"] = (
            "unavailable_simple_evidence_invalid_json"
        )
        return base
    if not isinstance(payload, Mapping):
        base["accelerator_idle_calibration_status"] = (
            "unavailable_simple_evidence_invalid_payload"
        )
        return base
    if payload.get("schema") != SIMPLE_CALIBRATION_EVIDENCE_SCHEMA:
        if legacy_binding_configured:
            return None
        base["accelerator_idle_calibration_status"] = (
            "unavailable_simple_evidence_schema_unsupported"
        )
        return base

    failures: list[str] = []
    if payload.get("schema_version") != 2:
        failures.append("schema_version_unsupported")
    setup_id_raw = _get(setup, "setup_id", "")
    setup_id = setup_id_raw if isinstance(setup_id_raw, str) else ""
    if not setup_id or payload.get("setup_id") != setup_id:
        failures.append("setup_id_mismatch")
    accelerator_raw = _get(setup, "accelerator", "")
    accelerator = (
        accelerator_raw if isinstance(accelerator_raw, str) else ""
    )
    if not accelerator or payload.get("accelerator") != accelerator:
        failures.append("accelerator_mismatch")
    urecs_raw = _get(setup, "urecs_address", "")
    urecs_address = urecs_raw if isinstance(urecs_raw, str) else ""
    if not urecs_address or payload.get("urecs_address") != urecs_address:
        failures.append("urecs_address_mismatch")
    data_port = _get(setup, "data_port", None)
    if (
        isinstance(data_port, bool)
        or not isinstance(data_port, int)
        or payload.get("data_port") != data_port
    ):
        failures.append("data_port_mismatch")
    if payload.get("physical_scope") != "FS":
        failures.append("physical_scope_not_full_system")
    if payload.get("status") != "ok":
        failures.append("status_not_ok")
    if payload.get("saved") is not True:
        failures.append("registry_save_not_confirmed")
    if payload.get("restored_m2_on") is not True:
        failures.append("m2_restore_not_verified")

    started_at = _canonical_utc_timestamp(payload.get("started_at"))
    finished_at = _canonical_utc_timestamp(payload.get("finished_at"))
    calibrated_at = _canonical_utc_timestamp(calibrated_at_raw)
    if started_at is None:
        failures.append("started_at_not_canonical_utc_timestamp")
    if finished_at is None:
        failures.append("finished_at_not_canonical_utc_timestamp")
    if calibrated_at is None:
        failures.append("registry_calibrated_at_not_canonical_utc_timestamp")
    if (
        started_at is not None
        and finished_at is not None
        and finished_at <= started_at
    ):
        failures.append("calibration_timestamp_chronology_invalid")
    if (
        finished_at is not None
        and calibrated_at is not None
        and calibrated_at != finished_at
    ):
        failures.append("registry_calibrated_at_mismatch")

    def phase_power(
        *, phase: str, top_level_alias: str
    ) -> tuple[float | None, list[str]]:
        phase_failures: list[str] = []
        phase_payload = payload.get(phase)
        nested_present = isinstance(phase_payload, Mapping) and (
            "avg_power_w" in phase_payload
        )
        alias_present = top_level_alias in payload
        nested_value = (
            _strict_finite_number(phase_payload.get("avg_power_w"), positive=True)
            if nested_present and isinstance(phase_payload, Mapping)
            else None
        )
        alias_value = (
            _strict_finite_number(payload.get(top_level_alias), positive=True)
            if alias_present
            else None
        )
        if nested_present and nested_value is None:
            phase_failures.append(f"{phase}_avg_power_invalid")
        if alias_present and alias_value is None:
            phase_failures.append(f"{top_level_alias}_invalid")
        if not nested_present and not alias_present:
            phase_failures.append(f"{phase}_avg_power_missing")
            return None, phase_failures
        if (
            nested_value is not None
            and alias_value is not None
            and not math.isclose(
                nested_value, alias_value, rel_tol=0.0, abs_tol=1e-9
            )
        ):
            phase_failures.append(f"{phase}_top_level_alias_mismatch")
        return (
            alias_value if alias_present else nested_value,
            phase_failures,
        )

    off_w, off_failures = phase_power(
        phase="m2_off", top_level_alias="idle_power_without_m2_w"
    )
    on_w, on_failures = phase_power(
        phase="m2_on", top_level_alias="idle_power_with_m2_w"
    )
    failures.extend(off_failures)
    failures.extend(on_failures)
    delta_w = _strict_finite_number(
        payload.get("accelerator_idle_power_w"), positive=True
    )
    registry_w = _strict_finite_number(
        _get(setup, "accelerator_idle_w"), positive=True
    )
    if delta_w is None:
        failures.append("accelerator_idle_power_invalid")
    if registry_w is None:
        failures.append("registry_accelerator_idle_power_invalid")
    if (
        off_w is not None
        and on_w is not None
        and delta_w is not None
        and not math.isclose(
            on_w - off_w, delta_w, rel_tol=0.0, abs_tol=1e-9
        )
    ):
        failures.append("accelerator_idle_power_delta_mismatch")
    if (
        delta_w is not None
        and registry_w is not None
        and not math.isclose(
            delta_w, registry_w, rel_tol=0.0, abs_tol=1e-9
        )
    ):
        failures.append("registry_accelerator_idle_power_mismatch")

    if failures:
        base["accelerator_idle_calibration_status"] = (
            "unavailable_simple_evidence_validation_failed"
        )
        base["accelerator_idle_calibration_failure_reasons"] = failures
        return base
    base.update(
        {
            "accelerator_idle_calibration_verified": True,
            "accelerator_idle_calibration_status": "verified",
            "accelerator_idle_calibration_evidence": str(resolved_path),
            "accelerator_idle_calibrated_at": str(payload.get("finished_at")),
            "accelerator_idle_calibration_physical_scope": "FS",
        }
    )
    return base


def verify_accelerator_idle_calibration_binding(
    setup: Any,
    *,
    registry_path: str | Path | None = None,
    registry: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify the current simple record or one historical immutable binding.

    A manually entered ``accelerator_idle_w`` remains insufficient for a
    claim-carrying TensorRT Full comparison.  Current calibrations pair the
    registry scalar with one small, unhashed full-system JSON record.  Existing
    schema-v2 immutable bindings remain accepted without migration.
    """

    setup_id_raw = _get(setup, "setup_id", "")
    setup_id = setup_id_raw if isinstance(setup_id_raw, str) else ""
    value_w = _strict_finite_number(_get(setup, "accelerator_idle_w"))
    path_text = str(_get(setup, "accelerator_idle_calibration_binding_path", "") or "").strip()
    expected_sha = str(_get(setup, "accelerator_idle_calibration_binding_sha256", "") or "").strip().lower()
    legacy_binding_configured = bool(
        path_text
        and _SHA256_RE.fullmatch(expected_sha) is not None
    )
    simple_verification = _verify_simple_accelerator_idle_calibration(
        setup,
        legacy_binding_configured=legacy_binding_configured,
    )
    if simple_verification is not None:
        return simple_verification
    base = {
        "accelerator_idle_calibration_verified": False,
        "accelerator_idle_calibration_status": "unavailable_missing_binding",
        "accelerator_idle_calibration_binding_path": path_text,
        "accelerator_idle_calibration_binding_sha256": expected_sha,
    }
    if value_w is None or value_w < 0.0:
        base["accelerator_idle_calibration_status"] = "unavailable_invalid_accelerator_idle_w"
        return base
    if not path_text or len(expected_sha) != 64 or any(c not in "0123456789abcdef" for c in expected_sha):
        return base
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        base["accelerator_idle_calibration_status"] = (
            "unavailable_binding_path_not_absolute"
        )
        return base
    try:
        resolved_binding_path = path.resolve(strict=True)
    except OSError:
        resolved_binding_path = path
    if resolved_binding_path != path or path.is_symlink():
        base["accelerator_idle_calibration_status"] = (
            "unavailable_binding_path_not_canonical"
        )
        return base
    try:
        payload_bytes = path.read_bytes()
    except OSError:
        base["accelerator_idle_calibration_status"] = "unavailable_binding_unreadable"
        return base
    actual_sha = hashlib.sha256(payload_bytes).hexdigest()
    base["accelerator_idle_calibration_binding_actual_sha256"] = actual_sha
    if actual_sha != expected_sha:
        base["accelerator_idle_calibration_status"] = "unavailable_binding_sha256_mismatch"
        return base
    try:
        binding = json.loads(payload_bytes.decode("utf-8"))
    except Exception:
        base["accelerator_idle_calibration_status"] = "unavailable_binding_invalid_json"
        return base
    if not isinstance(binding, Mapping):
        base["accelerator_idle_calibration_status"] = "unavailable_binding_invalid_payload"
        return base
    schema_version = binding.get("schema_version")
    if (
        binding.get("schema") != CALIBRATION_BINDING_SCHEMA
        or isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
    ):
        base["accelerator_idle_calibration_status"] = (
            "unavailable_binding_schema_unsupported"
        )
        return base
    if schema_version == 1:
        # V1 binds only the derived scalar.  Keep its status explicit so an old
        # local calibration remains diagnosable, but never promote it to a
        # claim-carrying TensorRT Full correction.
        base["accelerator_idle_calibration_status"] = (
            "unavailable_binding_legacy_v1_untrusted"
        )
        return base
    if schema_version != CALIBRATION_BINDING_SCHEMA_VERSION:
        base["accelerator_idle_calibration_status"] = (
            "unavailable_binding_schema_unsupported"
        )
        return base

    failures: list[str] = []

    if set(binding) != _CALIBRATION_BINDING_V2_FIELDS:
        failures.append("binding_v2_fields_mismatch")

    declared_payload_sha = str(
        binding.get("binding_payload_sha256") or ""
    ).strip()
    if _SHA256_RE.fullmatch(declared_payload_sha) is None:
        failures.append("binding_payload_sha256_invalid")
    elif _canonical_json_sha256(binding) != declared_payload_sha:
        failures.append("binding_payload_sha256_mismatch")

    bound_setup_id_raw = binding.get("setup_id")
    bound_address_raw = binding.get("urecs_address")
    bound_setup_id = (
        bound_setup_id_raw if isinstance(bound_setup_id_raw, str) else ""
    )
    bound_address = (
        bound_address_raw if isinstance(bound_address_raw, str) else ""
    )
    bound_accelerator_raw = binding.get("accelerator")
    current_accelerator_raw = _get(setup, "accelerator", "")
    bound_accelerator = (
        bound_accelerator_raw
        if isinstance(bound_accelerator_raw, str)
        else ""
    )
    current_accelerator = (
        current_accelerator_raw
        if isinstance(current_accelerator_raw, str)
        else ""
    )
    expected_address_raw = _get(setup, "urecs_address", "")
    expected_address = (
        expected_address_raw if isinstance(expected_address_raw, str) else ""
    )
    bound_data_port_raw = binding.get("data_port")
    bound_data_port = (
        int(bound_data_port_raw)
        if isinstance(bound_data_port_raw, int)
        and not isinstance(bound_data_port_raw, bool)
        and 1 <= bound_data_port_raw <= 65535
        else None
    )
    current_data_port_raw = _get(setup, "data_port", None)
    current_data_port = (
        int(current_data_port_raw)
        if isinstance(current_data_port_raw, int)
        and not isinstance(current_data_port_raw, bool)
        and 1 <= current_data_port_raw <= 65535
        else None
    )
    if not _canonical_identity_token(bound_setup_id):
        failures.append("setup_id_not_canonical")
    if not _canonical_identity_token(setup_id):
        failures.append("current_setup_id_not_canonical")
    if bound_setup_id != setup_id:
        failures.append("setup_id_mismatch")
    if not _canonical_identity_token(bound_address):
        failures.append("urecs_address_not_canonical")
    if not _canonical_identity_token(expected_address):
        failures.append("current_urecs_address_not_canonical")
    elif bound_address != expected_address:
        failures.append("urecs_address_mismatch")
    if bound_data_port is None:
        failures.append("data_port_invalid")
    if _get(setup, "data_port_valid", True) is not True:
        failures.append("current_data_port_invalid")
    elif current_data_port is None:
        failures.append("current_data_port_invalid")
    elif current_data_port != bound_data_port:
        failures.append("current_data_port_mismatch")
    if (
        not _canonical_identity_token(bound_accelerator)
        or _normalise_accelerator(bound_accelerator) != bound_accelerator
    ):
        failures.append("accelerator_identity_not_canonical")
    if (
        not _canonical_identity_token(current_accelerator)
        or _normalise_accelerator(current_accelerator) != current_accelerator
    ):
        failures.append("current_accelerator_identity_not_canonical")
    elif current_accelerator != bound_accelerator:
        failures.append("current_accelerator_mismatch")

    bound_jetson = binding.get("jetson_host")
    if not isinstance(bound_jetson, Mapping):
        failures.append("jetson_host_identity_missing")
        bound_jetson = {}
    elif set(bound_jetson) != {
        "address", "user", "port", "ssh_extra_args"
    }:
        failures.append("jetson_host_identity_fields_mismatch")
    bound_jetson_address_raw = bound_jetson.get("address")
    bound_jetson_user_raw = bound_jetson.get("user")
    bound_jetson_address = (
        bound_jetson_address_raw
        if isinstance(bound_jetson_address_raw, str)
        else ""
    )
    bound_jetson_user = (
        bound_jetson_user_raw
        if isinstance(bound_jetson_user_raw, str)
        else ""
    )
    bound_jetson_port_raw = bound_jetson.get("port")
    bound_jetson_port = (
        int(bound_jetson_port_raw)
        if isinstance(bound_jetson_port_raw, int)
        and not isinstance(bound_jetson_port_raw, bool)
        and 1 <= bound_jetson_port_raw <= 65535
        else None
    )
    bound_jetson_ssh_extra_args_raw = bound_jetson.get("ssh_extra_args")
    bound_jetson_ssh_extra_args = _normalise_ssh_extra_args(
        bound_jetson_ssh_extra_args_raw
    )
    current_jetson_address_raw = _get(setup, "jetson_address", "")
    current_jetson_user_raw = _get(setup, "jetson_user", "")
    current_jetson_address = (
        current_jetson_address_raw
        if isinstance(current_jetson_address_raw, str)
        else ""
    )
    current_jetson_user = (
        current_jetson_user_raw
        if isinstance(current_jetson_user_raw, str)
        else ""
    )
    current_jetson_port_raw = _get(setup, "jetson_port", None)
    current_jetson_port = (
        int(current_jetson_port_raw)
        if isinstance(current_jetson_port_raw, int)
        and not isinstance(current_jetson_port_raw, bool)
        and 1 <= current_jetson_port_raw <= 65535
        else None
    )
    current_jetson_ssh_extra_args = _normalise_ssh_extra_args(
        _get(setup, "jetson_ssh_extra_args", "")
    )
    current_jetson_identity_errors = tuple(
        str(value)
        for value in list(_get(setup, "jetson_identity_errors", ()) or ())
    )
    if (
        _get(setup, "jetson_identity_valid", True) is not True
        or current_jetson_identity_errors
    ):
        failures.append("current_jetson_identity_invalid")
        failures.extend(
            f"current_{reason}" for reason in current_jetson_identity_errors
        )
    if not _canonical_identity_token(bound_jetson_address):
        failures.append("jetson_host_address_not_canonical")
    if not _canonical_identity_token(bound_jetson_user):
        failures.append("jetson_host_user_not_canonical")
    if bound_jetson_port is None:
        failures.append("jetson_host_port_invalid")
    if not isinstance(bound_jetson_ssh_extra_args_raw, str):
        failures.append("jetson_host_ssh_extra_args_invalid")
    elif bound_jetson_ssh_extra_args_raw != bound_jetson_ssh_extra_args:
        failures.append("jetson_host_ssh_extra_args_not_normalised")
    if not _canonical_identity_token(current_jetson_address):
        failures.append("current_jetson_address_not_canonical")
    elif current_jetson_address != bound_jetson_address:
        failures.append("current_jetson_address_mismatch")
    if not _canonical_identity_token(current_jetson_user):
        failures.append("current_jetson_user_not_canonical")
    elif current_jetson_user != bound_jetson_user:
        failures.append("current_jetson_user_mismatch")
    if current_jetson_port is None:
        failures.append("current_jetson_port_invalid")
    elif current_jetson_port != bound_jetson_port:
        failures.append("current_jetson_port_mismatch")
    if current_jetson_ssh_extra_args != bound_jetson_ssh_extra_args:
        failures.append("current_jetson_ssh_extra_args_mismatch")
    if str(binding.get("status") or "") != "ok":
        failures.append("binding_status_not_ok")
    if binding.get("restored_m2_on") is not True:
        failures.append("m2_restore_not_verified")
    started_at = _canonical_utc_timestamp(binding.get("started_at"))
    finished_at = _canonical_utc_timestamp(binding.get("finished_at"))
    if started_at is None:
        failures.append("started_at_not_canonical_utc_timestamp")
    if finished_at is None:
        failures.append("finished_at_not_canonical_utc_timestamp")
    if (
        started_at is not None
        and finished_at is not None
        and finished_at <= started_at
    ):
        failures.append("calibration_timestamp_chronology_invalid")

    off_w = _strict_finite_number(binding.get("idle_power_without_m2_w"))
    on_w = _strict_finite_number(binding.get("idle_power_with_m2_w"))
    bound_w = _strict_finite_number(binding.get("accelerator_idle_power_w"))
    if off_w is None or off_w <= 0.0:
        failures.append("idle_power_without_m2_invalid")
    if on_w is None or on_w <= 0.0:
        failures.append("idle_power_with_m2_invalid")
    if bound_w is None or bound_w < 0.0:
        failures.append("accelerator_idle_power_invalid")
    if (
        off_w is not None
        and on_w is not None
        and bound_w is not None
        and not math.isclose(on_w - off_w, bound_w, rel_tol=0.0, abs_tol=1e-9)
    ):
        failures.append("accelerator_idle_power_delta_mismatch")
    if bound_w is not None and not math.isclose(
        bound_w, value_w, rel_tol=0.0, abs_tol=1e-9
    ):
        failures.append("registry_accelerator_idle_power_mismatch")
    require_positive_delta, minimum_delta_w, delta_policy_failures = (
        _configured_calibration_delta_policy(registry, setup_id)
    )
    failures.extend(delta_policy_failures)
    if (
        require_positive_delta
        and bound_w is not None
        and bound_w < minimum_delta_w
    ):
        failures.append("accelerator_idle_power_below_configured_minimum")

    observation_keys = {
        "setup_id",
        "urecs_address",
        "jetson_host",
        "jetson_ssh_ready",
        "m2_present",
    }

    def validate_observation(
        observation: Any,
        *,
        label: str,
        expected_m2_present: bool,
    ) -> Mapping[str, Any] | None:
        if not isinstance(observation, Mapping):
            failures.append(f"{label}_missing")
            return None
        if set(observation) != observation_keys:
            failures.append(f"{label}_fields_mismatch")
        if str(observation.get("setup_id") or "") != setup_id:
            failures.append(f"{label}_setup_id_mismatch")
        if str(observation.get("urecs_address") or "") != bound_address:
            failures.append(f"{label}_urecs_address_mismatch")
        if observation.get("jetson_host") != bound_jetson:
            failures.append(f"{label}_jetson_host_mismatch")
        if observation.get("jetson_ssh_ready") is not True:
            failures.append(f"{label}_jetson_ssh_not_ready")
        if observation.get("m2_present") is not expected_m2_present:
            failures.append(f"{label}_m2_state_mismatch")
        return observation

    state_observations = binding.get("state_observations")
    validated_observations: dict[str, Mapping[str, Any]] = {}
    if not isinstance(state_observations, Mapping):
        failures.append("state_observations_missing")
        state_observations = {}
    elif set(state_observations) != {"initial", "m2_off", "m2_on"}:
        failures.append("state_observations_fields_mismatch")
    initial_observation = validate_observation(
        state_observations.get("initial"),
        label="state_initial",
        expected_m2_present=True,
    )
    if initial_observation is not None:
        validated_observations["initial"] = initial_observation
    for phase, expected_present in (("m2_off", False), ("m2_on", True)):
        phase_observations = state_observations.get(phase)
        if not isinstance(phase_observations, Mapping):
            failures.append(f"state_{phase}_observations_missing")
            continue
        if set(phase_observations) != {
            "pre_measurement", "post_measurement"
        }:
            failures.append(f"state_{phase}_observation_fields_mismatch")
        for boundary in ("pre_measurement", "post_measurement"):
            observation = validate_observation(
                phase_observations.get(boundary),
                label=f"state_{phase}_{boundary}",
                expected_m2_present=expected_present,
            )
            if observation is not None:
                validated_observations[f"{phase}_{boundary}"] = observation

    transitions = binding.get("transitions")
    if not isinstance(transitions, Mapping):
        failures.append("transitions_missing")
        transitions = {}
    elif set(transitions) != {"m2_off", "m2_on"}:
        failures.append("transitions_fields_mismatch")
    transition_keys = {
        "ok", "changed", "desired_accelerator_present", "after"
    }
    for phase, expected_present in (("m2_off", False), ("m2_on", True)):
        transition = transitions.get(phase)
        if not isinstance(transition, Mapping):
            failures.append(f"transition_{phase}_missing")
            continue
        if set(transition) != transition_keys:
            failures.append(f"transition_{phase}_fields_mismatch")
        if transition.get("ok") is not True:
            failures.append(f"transition_{phase}_not_ok")
        if transition.get("changed") is not True:
            failures.append(f"transition_{phase}_not_changed")
        if transition.get("desired_accelerator_present") is not expected_present:
            failures.append(f"transition_{phase}_target_mismatch")
        after = validate_observation(
            transition.get("after"),
            label=f"transition_{phase}_after",
            expected_m2_present=expected_present,
        )
        pre_measurement = validated_observations.get(
            f"{phase}_pre_measurement"
        )
        if (
            after is not None
            and pre_measurement is not None
            and dict(after) != dict(pre_measurement)
        ):
            failures.append(f"transition_{phase}_after_pre_state_mismatch")

    calibration_root = path.parent.resolve()
    referenced_paths: list[Path] = []

    method = binding.get("energy_calibration_manifest")
    method_path: Path | None = None
    method_payload: Mapping[str, Any] | None = None
    method_sha = ""
    runtime_binding_id = ""
    if not isinstance(method, Mapping):
        failures.append("energy_calibration_manifest_identity_missing")
    else:
        method_sha = str(method.get("sha256") or "").strip()
        method_path, method_bytes, method_error = _artifact_identity(
            method,
            expected_fields=frozenset(
                {"path", "sha256", "verification_status", "runtime_binding_id"}
            ),
        )
        if method_error:
            failures.append(f"energy_calibration_manifest_{method_error}")
        else:
            method_payload = _json_mapping(method_bytes)
            if method_payload is None:
                failures.append("energy_calibration_manifest_invalid_json")
            elif (
                method_payload.get("schema")
                != ENERGY_CALIBRATION_MANIFEST_SCHEMA
                or method_payload.get("schema_version") != 2
                or str(method_payload.get("evidence_mode") or "")
                != "inherited_validated_method"
            ):
                failures.append("energy_calibration_manifest_contract_mismatch")
            else:
                try:
                    from ..campaign import verify_energy_calibration_manifest

                    method_verification = verify_energy_calibration_manifest(
                        method_payload,
                        require_final=True,
                    )
                except Exception:
                    method_verification = {"ok": False}
                if method_verification.get("ok") is not True:
                    failures.append("energy_calibration_manifest_content_unverified")
        if (
            str(method.get("verification_status") or "")
            != "inherited_validated_method_verified"
        ):
            failures.append("energy_calibration_manifest_status_unverified")
        runtime_binding_id = str(
            method.get("runtime_binding_id") or ""
        ).strip()
        if not runtime_binding_id:
            failures.append("energy_calibration_runtime_binding_id_missing")

    if method_payload is not None:
        bindings = [
            row
            for row in list(method_payload.get("channel_bindings") or [])
            if isinstance(row, Mapping)
            and str(row.get("setup_id") or "").strip() == setup_id
        ]
        if len(bindings) != 1:
            failures.append("energy_calibration_setup_binding_not_unique")
        else:
            manifest_binding = bindings[0]
            expected_runtime_id = str(
                manifest_binding.get("binding_id")
                or manifest_binding.get("setup_id")
                or ""
            ).strip()
            if runtime_binding_id != expected_runtime_id:
                failures.append("energy_calibration_runtime_binding_id_mismatch")
            if str(manifest_binding.get("urecs_address") or "").strip() != bound_address:
                failures.append("energy_calibration_urecs_address_mismatch")
            if manifest_binding.get("channel") != 0:
                failures.append("energy_calibration_channel_mismatch")
            if manifest_binding.get("sample_rate_hz") != 2000:
                failures.append("energy_calibration_sample_rate_mismatch")
            if manifest_binding.get("data_port") != bound_data_port:
                failures.append("energy_calibration_data_port_mismatch")
            if str(manifest_binding.get("scope") or "").strip().upper() != "FS":
                failures.append("energy_calibration_scope_mismatch")

    # The idle scalar and the subsequent claim measurement must use the exact
    # same full-system method identity.  Verifying the historical binding alone
    # is insufficient: a registry may legitimately be moved to a newer method
    # manifest after calibration, in which case the idle scalar must be
    # recalibrated rather than silently paired across methods.
    current_method_path_text = str(
        _get(setup, "calibration_manifest", "") or ""
    ).strip()
    current_method_sha = _normalise_sha256(
        _get(setup, "calibration_sha256", "")
    )
    current_method_path, _current_method_bytes, current_method_error = (
        _canonical_local_file(
            current_method_path_text,
            current_method_sha,
        )
    )
    if current_method_error:
        failures.append(f"current_energy_method_{current_method_error}")
    else:
        if current_method_path != method_path:
            failures.append("current_energy_method_path_mismatch")
        if current_method_sha != method_sha:
            failures.append("current_energy_method_sha256_mismatch")
        # Claim correction is a current-runtime decision, not merely a check
        # that an old calibration record still hashes.  Re-run the strict
        # registry-configured method admission so exact implementation and
        # installed-source integrity are both current at comparison time.
        try:
            from .method_manifest import verify_configured_energy_method

            configured_method = verify_configured_energy_method(
                setup_id,
                registry_path=registry_path,
                registry=registry,
            )
        except Exception as exc:
            configured_method = {
                "verified": False,
                "status": "verification_exception",
                "error": f"{type(exc).__name__}:{exc}",
            }
        configured_admission = (
            configured_method.get("configured_method_admission")
            if isinstance(
                configured_method.get("configured_method_admission"), Mapping
            )
            else {}
        )
        source_integrity = (
            configured_method.get("source_integrity_verification")
            if isinstance(
                configured_method.get("source_integrity_verification"), Mapping
            )
            else {}
        )
        configured_path_text = configured_method.get("path")
        try:
            configured_path = Path(
                configured_path_text
                if isinstance(configured_path_text, str)
                else ""
            ).expanduser().resolve(strict=True)
        except OSError:
            configured_path = None
        configured_sha = _normalise_sha256(
            configured_method.get("sha256")
        )
        if not (
            configured_method.get("verified") is True
            and configured_method.get("status")
            == "inherited_validated_method_verified"
            and configured_admission.get("ok") is True
            and source_integrity.get("ok") is True
            and source_integrity.get("status") == "verified"
            and configured_path == current_method_path
            and configured_sha == current_method_sha
        ):
            failures.append("current_configured_energy_method_not_strictly_verified")

    evidence = binding.get("calibration_evidence")
    evidence_path, evidence_bytes, evidence_error = _artifact_identity(
        evidence,
        contained_by=calibration_root,
    )
    if evidence_error:
        failures.append(f"calibration_evidence_{evidence_error}")
    else:
        assert evidence_path is not None
        referenced_paths.append(evidence_path)
        evidence_payload = _json_mapping(evidence_bytes)
        if evidence_payload is None:
            failures.append("calibration_evidence_invalid_json")
        else:
            if set(evidence_payload) != _CALIBRATION_EVIDENCE_V1_FIELDS:
                failures.append("calibration_evidence_fields_mismatch")
            if (
                evidence_payload.get("schema")
                != "onnx-splitpoint/accelerator-idle-calibration-evidence"
                or type(evidence_payload.get("schema_version")) is not int
                or evidence_payload.get("schema_version") != 1
            ):
                failures.append("calibration_evidence_schema_mismatch")
            if str(evidence_payload.get("setup_id") or "") != setup_id:
                failures.append("calibration_evidence_setup_id_mismatch")
            if str(evidence_payload.get("accelerator") or "") != str(
                binding.get("accelerator") or ""
            ):
                failures.append("calibration_evidence_accelerator_mismatch")
            if str(evidence_payload.get("urecs_address") or "") != bound_address:
                failures.append("calibration_evidence_urecs_address_mismatch")
            evidence_port = evidence_payload.get("data_port")
            if (
                type(evidence_port) is not int
                or evidence_port != bound_data_port
            ):
                failures.append("calibration_evidence_data_port_mismatch")
            if evidence_payload.get("jetson_host") != bound_jetson:
                failures.append("calibration_evidence_jetson_host_mismatch")
            if (
                _canonical_utc_timestamp(evidence_payload.get("started_at"))
                is None
                or evidence_payload.get("started_at") != binding.get("started_at")
            ):
                failures.append("calibration_evidence_started_at_mismatch")
            if (
                _canonical_utc_timestamp(evidence_payload.get("finished_at"))
                is None
                or evidence_payload.get("finished_at") != binding.get("finished_at")
            ):
                failures.append("calibration_evidence_finished_at_mismatch")
            if str(evidence_payload.get("status") or "") != "ok":
                failures.append("calibration_evidence_status_not_ok")
            if evidence_payload.get("restored_m2_on") is not True:
                failures.append("calibration_evidence_m2_restore_not_verified")
            for field, value, positive in (
                ("idle_power_without_m2_w", off_w, True),
                ("idle_power_with_m2_w", on_w, True),
                ("accelerator_idle_power_w", bound_w, False),
            ):
                if _strict_finite_number(
                    evidence_payload.get(field), positive=positive
                ) is None:
                    failures.append(
                        f"calibration_evidence_{field}_not_exact_number"
                    )
            if not _same_number(
                evidence_payload.get("idle_power_without_m2_w"), off_w
            ):
                failures.append("calibration_evidence_m2_off_power_mismatch")
            if not _same_number(
                evidence_payload.get("idle_power_with_m2_w"), on_w
            ):
                failures.append("calibration_evidence_m2_on_power_mismatch")
            if not _same_number(
                evidence_payload.get("accelerator_idle_power_w"), bound_w
            ):
                failures.append("calibration_evidence_delta_mismatch")
            if evidence_payload.get("energy_calibration_manifest") != method:
                failures.append("calibration_evidence_method_identity_mismatch")
            if evidence_payload.get("captures") != binding.get("captures"):
                failures.append("calibration_evidence_capture_identity_mismatch")
            if evidence_payload.get("state_observations") != state_observations:
                failures.append("calibration_evidence_state_observations_mismatch")
            if evidence_payload.get("transitions") != transitions:
                failures.append("calibration_evidence_transitions_mismatch")

    captures = binding.get("captures")
    if not isinstance(captures, Mapping):
        failures.append("captures_missing")
        captures = {}
    elif set(captures) != {"m2_off", "m2_on"}:
        failures.append("captures_fields_mismatch")
    for phase, expected_power in (("m2_off", off_w), ("m2_on", on_w)):
        capture = captures.get(phase)
        if not isinstance(capture, Mapping):
            failures.append(f"{phase}_capture_missing")
            continue
        unknown_roles = set(capture) - {
            "avg_power_w",
            "aggregate",
            "report",
            "raw",
            "run",
            "command_window_binding",
            "postprocessor_result",
        }
        if unknown_roles:
            failures.append(
                f"{phase}_capture_unknown_roles:"
                + ",".join(sorted(str(value) for value in unknown_roles))
            )
        capture_power = _strict_finite_number(capture.get("avg_power_w"))
        if expected_power is None or not _same_number(capture_power, expected_power):
            failures.append(f"{phase}_capture_power_mismatch")
        if capture_power is None:
            failures.append(f"{phase}_capture_power_not_exact_number")
        loaded_by_role: dict[str, Mapping[str, Any] | None] = {}
        bytes_by_role: dict[str, bytes] = {}
        paths_by_role: dict[str, Path] = {}
        hashes_by_role: dict[str, str] = {}
        required_roles = ("aggregate", "report", "raw", "run")
        optional_roles = ("command_window_binding", "postprocessor_result")
        for role in (*required_roles, *optional_roles):
            if role in optional_roles and capture.get(role) is None:
                continue
            artifact_path, artifact_bytes, artifact_error = _artifact_identity(
                capture.get(role),
                contained_by=calibration_root,
            )
            if artifact_error:
                failures.append(f"{phase}_{role}_{artifact_error}")
                continue
            assert artifact_path is not None
            referenced_paths.append(artifact_path)
            paths_by_role[role] = artifact_path
            bytes_by_role[role] = artifact_bytes or b""
            hashes_by_role[role] = str(
                (capture.get(role) or {}).get("sha256")
                if isinstance(capture.get(role), Mapping)
                else ""
            )
            if role == "raw":
                if (
                    artifact_path.suffix.lower() != ".parquet"
                    or not _is_parquet_file(artifact_bytes)
                ):
                    failures.append(f"{phase}_raw_not_nonempty_parquet")
                continue
            if role == "postprocessor_result":
                if not artifact_bytes:
                    failures.append(f"{phase}_postprocessor_result_empty")
                continue
            loaded = _json_mapping(artifact_bytes)
            loaded_by_role[role] = loaded
            if loaded is None:
                failures.append(f"{phase}_{role}_invalid_json")

        phase_root = (calibration_root / phase).resolve()
        exact_role_paths = {
            "aggregate": phase_root / "energy_aggregate.json",
            "report": phase_root / "energy_summary.json",
            "run": phase_root / "run_000" / "energy_summary.json",
            "command_window_binding": (
                phase_root / "run_000" / "command_window_binding.json"
            ),
        }
        for role, expected_path in exact_role_paths.items():
            if paths_by_role.get(role) != expected_path:
                failures.append(f"{phase}_{role}_role_path_mismatch")
        for role in ("raw", "postprocessor_result"):
            role_path = paths_by_role.get(role)
            if role_path is not None:
                try:
                    relative = role_path.relative_to(phase_root / "run_000")
                except ValueError:
                    failures.append(f"{phase}_{role}_phase_path_mismatch")
                else:
                    if role == "raw" and relative.parts[:1] != (
                        "collector_storage",
                    ):
                        failures.append(f"{phase}_raw_role_path_mismatch")
                    if role == "postprocessor_result" and role_path.name not in {
                        "results.yaml", "results.yml", "results.json"
                    }:
                        failures.append(
                            f"{phase}_postprocessor_result_role_path_mismatch"
                        )
        if (
            "aggregate" in bytes_by_role
            and "report" in bytes_by_role
            and bytes_by_role["aggregate"] != bytes_by_role["report"]
        ):
            failures.append(f"{phase}_aggregate_report_content_mismatch")

        for role in ("aggregate", "report"):
            payload = loaded_by_role.get(role)
            if payload is None:
                continue
            if payload.get("ok") is not True:
                failures.append(f"{phase}_{role}_not_ok")
            status_value = payload.get("status")
            if type(status_value) is not str or status_value not in {
                "ok", "ok_caller_managed_repeat_capture"
            }:
                failures.append(f"{phase}_{role}_status_mismatch")
            # Current production summaries are deliberately schema-less.  Do
            # not accept attacker-added pseudo-schema fields as if they were a
            # recognized versioned contract.
            if "schema" in payload or "schema_version" in payload:
                failures.append(f"{phase}_{role}_unexpected_schema_fields")
            if str(payload.get("setup_id") or "") != setup_id:
                failures.append(f"{phase}_{role}_setup_id_mismatch")
            if payload.get("run_id") != f"m2_idle_calibration_{phase}":
                failures.append(f"{phase}_{role}_run_id_mismatch")
            if _strict_finite_number(payload.get("avg_power_w")) is None:
                failures.append(f"{phase}_{role}_power_not_exact_number")
            if not _same_number(payload.get("avg_power_w"), capture_power):
                failures.append(f"{phase}_{role}_power_mismatch")
            if str(payload.get("energy_calibration_sha256") or "") != method_sha:
                failures.append(f"{phase}_{role}_method_sha256_mismatch")
            verification = payload.get("energy_calibration_verification")
            if not isinstance(verification, Mapping):
                failures.append(f"{phase}_{role}_method_verification_missing")
            elif not (
                verification.get("verified") is True
                and str(verification.get("status") or "")
                == "inherited_validated_method_verified"
                and str(verification.get("runtime_binding_id") or "")
                == runtime_binding_id
            ):
                failures.append(f"{phase}_{role}_method_verification_mismatch")

        run_payload = loaded_by_role.get("run")
        if run_payload is not None:
            if str(run_payload.get("status") or "") != "collector_finished":
                failures.append(f"{phase}_run_status_not_finished")
            if (
                type(run_payload.get("collector_rc")) is not int
                or run_payload.get("collector_rc") != 0
            ):
                failures.append(f"{phase}_run_collector_failed")
            if (
                type(run_payload.get("workload_command_rc")) is not int
                or run_payload.get("workload_command_rc") != 0
            ):
                failures.append(f"{phase}_run_workload_failed")
            if str(run_payload.get("postprocess_status") or "") != "ok":
                failures.append(f"{phase}_run_postprocess_failed")
            if str(run_payload.get("final_energy_gate_status") or "") != "pass":
                failures.append(f"{phase}_run_final_gate_failed")
            if _strict_finite_number(run_payload.get("avg_power_w")) is None:
                failures.append(f"{phase}_run_power_not_exact_number")
            if not _same_number(run_payload.get("avg_power_w"), capture_power):
                failures.append(f"{phase}_run_power_mismatch")
            if str(run_payload.get("energy_calibration_sha256") or "") != method_sha:
                failures.append(f"{phase}_run_method_sha256_mismatch")
            run_verification = run_payload.get("energy_calibration_verification")
            if not isinstance(run_verification, Mapping):
                failures.append(f"{phase}_run_method_verification_missing")
            elif not (
                run_verification.get("verified") is True
                and str(run_verification.get("status") or "")
                == "inherited_validated_method_verified"
                and str(run_verification.get("runtime_binding_id") or "")
                == runtime_binding_id
            ):
                failures.append(f"{phase}_run_method_verification_mismatch")

            provenance = run_payload.get("raw_input_energy_provenance")
            if (
                str(run_payload.get("raw_input_energy_provenance_status") or "")
                != "verified_calibrated_input_energy_unsubtracted"
                or not isinstance(provenance, Mapping)
                or str(provenance.get("binding_status") or "") != "verified"
            ):
                failures.append(f"{phase}_run_raw_provenance_unverified")
            else:
                trace_path_text = str(provenance.get("trace_path") or "").strip()
                trace_sha = str(provenance.get("trace_sha256") or "").strip()
                try:
                    trace_path = Path(trace_path_text).expanduser().resolve(
                        strict=True
                    )
                except OSError:
                    trace_path = None
                if (
                    trace_path != paths_by_role.get("raw")
                    or trace_sha != hashes_by_role.get("raw")
                ):
                    failures.append(f"{phase}_run_raw_identity_mismatch")

                result_path_text = str(
                    provenance.get("result_path") or ""
                ).strip()
                result_sha = str(provenance.get("result_sha256") or "").strip()
                try:
                    result_path = Path(result_path_text).expanduser().resolve(
                        strict=True
                    )
                except OSError:
                    result_path = None
                if "postprocessor_result" not in paths_by_role:
                    failures.append(
                        f"{phase}_postprocessor_result_identity_missing"
                    )
                elif (
                    result_path != paths_by_role.get("postprocessor_result")
                    or result_sha != hashes_by_role.get("postprocessor_result")
                ):
                    failures.append(
                        f"{phase}_run_postprocessor_result_identity_mismatch"
                    )

            trace_binding = run_payload.get("command_window_trace_binding")
            trace_binding = (
                trace_binding if isinstance(trace_binding, Mapping) else {}
            )
            command_binding_text = str(
                run_payload.get("command_window_binding_manifest")
                or trace_binding.get("manifest_path")
                or ""
            ).strip()
            try:
                command_binding_path = Path(
                    command_binding_text
                ).expanduser().resolve(strict=True)
            except OSError:
                command_binding_path = None
            if (
                str(trace_binding.get("status") or "") != "verified"
                or "command_window_binding" not in paths_by_role
            ):
                failures.append(
                    f"{phase}_command_window_binding_identity_missing_or_unverified"
                )
            elif command_binding_path != paths_by_role.get(
                "command_window_binding"
            ):
                failures.append(
                    f"{phase}_run_command_window_binding_identity_mismatch"
                )

            command_payload = loaded_by_role.get("command_window_binding")
            if command_payload is not None:
                if set(command_payload) != _COMMAND_WINDOW_BINDING_V2_FIELDS:
                    failures.append(
                        f"{phase}_command_window_binding_fields_mismatch"
                    )
                command_artifacts: dict[str, Path] = {}
                for artifact_name in (
                    "request", "marker", "trace", "command",
                    "workload_command", "timing", "postprocessor_result",
                    "result",
                ):
                    artifact_path, _artifact_bytes, artifact_error = (
                        _canonical_local_file(
                            command_payload.get(f"{artifact_name}_path"),
                            command_payload.get(f"{artifact_name}_sha256"),
                            contained_by=calibration_root,
                        )
                    )
                    if artifact_error:
                        failures.append(
                            f"{phase}_command_window_{artifact_name}_{artifact_error}"
                        )
                    elif artifact_path is not None:
                        command_artifacts[artifact_name] = artifact_path
                if command_artifacts.get("trace") != paths_by_role.get("raw"):
                    failures.append(
                        f"{phase}_command_window_trace_identity_mismatch"
                    )
                if command_artifacts.get("result") != paths_by_role.get(
                    "postprocessor_result"
                ):
                    failures.append(
                        f"{phase}_command_window_selected_result_identity_mismatch"
                    )
                if len(command_artifacts) == 8:
                    request_payload = _structured_mapping(
                        command_artifacts["request"]
                    )
                    marker_payload = _structured_mapping(
                        command_artifacts["marker"]
                    )
                    postprocessor_payload = _structured_mapping(
                        command_artifacts["postprocessor_result"]
                    )
                    selected_result = _structured_mapping(
                        command_artifacts["result"]
                    )
                    timing_payload = _timing_mapping(
                        command_artifacts["timing"]
                    )
                    if selected_result is None:
                        failures.append(
                            f"{phase}_command_window_selected_result_invalid"
                        )
                    if timing_payload is None:
                        failures.append(
                            f"{phase}_command_window_timing_invalid"
                        )
                    if (
                        request_payload is None
                        or set(request_payload)
                        != _COMMAND_WINDOW_REQUEST_V2_FIELDS
                    ):
                        failures.append(
                            f"{phase}_command_window_request_fields_mismatch"
                        )
                    if (
                        marker_payload is None
                        or set(marker_payload)
                        != _COMMAND_WINDOW_MARKER_V2_FIELDS
                    ):
                        failures.append(
                            f"{phase}_command_window_marker_fields_mismatch"
                        )
                    elif not (
                        isinstance(marker_payload.get("command"), Mapping)
                        and set(marker_payload["command"])
                        == _COMMAND_WINDOW_MARKER_COMMAND_V2_FIELDS
                        and isinstance(marker_payload.get("stream"), Mapping)
                        and set(marker_payload["stream"])
                        == _COMMAND_WINDOW_MARKER_STREAM_V2_FIELDS
                    ):
                        failures.append(
                            f"{phase}_command_window_marker_nested_fields_mismatch"
                        )
                    if (
                        postprocessor_payload is None
                        or set(postprocessor_payload)
                        != _POSTPROCESSOR_WINDOW_RESULT_V2_FIELDS
                    ):
                        failures.append(
                            f"{phase}_command_window_postprocessor_fields_mismatch"
                        )
                    if request_payload is not None and not (
                        _has_exact_int_fields(
                            request_payload,
                            (
                                "schema_version", "first_sample_index",
                                "last_sample_index", "sample_count",
                                "interval_count", "sample_rate_hz",
                                "process_rc", "process_id",
                                "start_clock_read_uncertainty_ns",
                                "end_clock_read_uncertainty_ns",
                                "command_start_realtime_ns",
                                "command_end_realtime_ns",
                                "command_start_monotonic_ns",
                                "command_end_monotonic_ns", "dropped_samples",
                                "boundary_uncertainty_samples",
                                "maximum_boundary_uncertainty_samples",
                                "total_samples", "created_at_unix_ns",
                            ),
                        )
                        and request_payload.get("trace_covers_window") is True
                        and request_payload.get("valid_for_final_energy") is True
                        and _strict_finite_number(
                            request_payload.get("duration_s"), positive=True
                        )
                        is not None
                    ):
                        failures.append(
                            f"{phase}_command_window_request_types_mismatch"
                        )
                    if marker_payload is not None:
                        marker_command = marker_payload.get("command")
                        marker_stream = marker_payload.get("stream")
                        if not (
                            type(marker_payload.get("schema_version")) is int
                            and marker_payload.get("valid_for_final_energy")
                            is True
                            and isinstance(marker_payload.get("validation_errors"), list)
                            and isinstance(marker_command, Mapping)
                            and _has_exact_int_fields(
                                marker_command,
                                (
                                    "process_id", "process_rc",
                                    "start_realtime_ns", "end_realtime_ns",
                                    "start_monotonic_ns", "end_monotonic_ns",
                                    "start_clock_read_uncertainty_ns",
                                    "end_clock_read_uncertainty_ns",
                                ),
                            )
                            and isinstance(marker_stream, Mapping)
                            and _has_exact_int_fields(
                                marker_stream,
                                (
                                    "sample_rate_hz", "first_sample_index",
                                    "last_sample_index", "total_samples",
                                    "boundary_uncertainty_samples",
                                    "dropped_samples",
                                ),
                            )
                            and marker_stream.get("trace_covers_window") is True
                        ):
                            failures.append(
                                f"{phase}_command_window_marker_types_mismatch"
                            )
                    if postprocessor_payload is not None and not (
                        _has_exact_int_fields(
                            postprocessor_payload,
                            (
                                "schema_version", "first_sample_index",
                                "last_sample_index", "sample_count",
                                "interval_count", "sample_rate_hz",
                                "process_rc", "command_process_id",
                                "command_start_realtime_ns",
                                "command_end_realtime_ns",
                                "command_start_monotonic_ns",
                                "command_end_monotonic_ns",
                                "command_start_clock_read_uncertainty_ns",
                                "command_end_clock_read_uncertainty_ns",
                                "drop_count", "boundary_uncertainty_samples",
                                "maximum_boundary_uncertainty_samples",
                            ),
                        )
                        and postprocessor_payload.get("trace_covers_window") is True
                        and _strict_finite_number(
                            postprocessor_payload.get("duration_s"), positive=True
                        )
                        is not None
                        and _strict_finite_number(
                            postprocessor_payload.get("energy_j"), positive=True
                        )
                        is not None
                    ):
                        failures.append(
                            f"{phase}_command_window_postprocessor_types_mismatch"
                        )
                    if selected_result is not None and timing_payload is not None:
                        try:
                            from .collector import _command_window_binding_v2

                            replay = _command_window_binding_v2(
                                command_payload,
                                manifest_path=paths_by_role.get(
                                    "command_window_binding"
                                ),
                                result_data=selected_result,
                                timing=timing_payload,
                                result_path=command_artifacts["result"],
                            )
                        except Exception as exc:
                            replay = {
                                "verified": False,
                                "status": f"exception:{type(exc).__name__}",
                            }
                        if (
                            replay.get("verified") is not True
                            or replay.get("status") != "verified"
                            or replay.get(
                                "calibrated_input_energy_unsubtracted_verified"
                            )
                            is not True
                        ):
                            failures.append(
                                f"{phase}_command_window_replay_failed:"
                                + str(replay.get("status") or "invalid")
                            )
                    command_run_id = command_payload.get("run_id")
                    if command_run_id != f"m2_idle_calibration_{phase}":
                        failures.append(
                            f"{phase}_command_window_phase_identity_mismatch"
                        )
                    command_energy = _strict_finite_number(
                        command_payload.get("energy_j"), positive=True
                    )
                    command_duration = _strict_finite_number(
                        command_payload.get("duration_s"), positive=True
                    )
                    if (
                        command_energy is not None
                        and command_duration is not None
                        and capture_power is not None
                        and not math.isclose(
                            command_energy / command_duration,
                            capture_power,
                            rel_tol=1e-9,
                            abs_tol=1e-9,
                        )
                    ):
                        failures.append(
                            f"{phase}_command_window_energy_power_mismatch"
                        )
                if not (
                    command_payload.get("schema")
                    == "onnx-splitpoint/command-window-binding"
                    and type(command_payload.get("schema_version")) is int
                    and command_payload.get("schema_version") == 2
                    and command_payload.get("binding_method")
                    == "collector_sample_marker_crop"
                ):
                    failures.append(
                        f"{phase}_command_window_binding_contract_mismatch"
                    )
                if (
                    type(command_payload.get("sample_rate_hz")) is not int
                    or command_payload.get("sample_rate_hz") != 2000
                ):
                    failures.append(
                        f"{phase}_command_window_sample_rate_mismatch"
                    )
                if (
                    type(command_payload.get("process_rc")) is not int
                    or command_payload.get("process_rc") != 0
                ):
                    failures.append(
                        f"{phase}_command_window_process_rc_mismatch"
                    )
                if command_payload.get("trace_covers_window") is not True:
                    failures.append(
                        f"{phase}_command_window_trace_coverage_unverified"
                    )
                if (
                    type(command_payload.get("dropped_samples")) is not int
                    or command_payload.get("dropped_samples") != 0
                ):
                    failures.append(
                        f"{phase}_command_window_dropped_samples_not_zero"
                    )
                if (
                    command_payload.get("energy_semantics")
                    != "calibrated_input_energy_unsubtracted"
                ):
                    failures.append(
                        f"{phase}_command_window_energy_semantics_mismatch"
                    )
                source_value = command_payload.get("source")
                if source_value != "fast_firmware":
                    failures.append(
                        f"{phase}_command_window_source_mismatch"
                    )
                if command_payload.get("energy_field") != "firmware_results.energy":
                    failures.append(
                        f"{phase}_command_window_energy_field_mismatch"
                    )
                if _strict_finite_number(
                    command_payload.get("energy_j"), positive=True
                ) is None:
                    failures.append(
                        f"{phase}_command_window_energy_not_exact_positive_number"
                    )
                first_index = command_payload.get("first_sample_index")
                last_index = command_payload.get("last_sample_index")
                sample_count = command_payload.get("sample_count")
                interval_count = command_payload.get("interval_count")
                if not (
                    type(first_index) is int
                    and type(last_index) is int
                    and type(sample_count) is int
                    and type(interval_count) is int
                    and first_index >= 0
                    and last_index >= first_index
                    and sample_count == last_index - first_index + 1
                    and interval_count == sample_count - 1
                    and interval_count > 0
                ):
                    failures.append(
                        f"{phase}_command_window_sample_interval_mismatch"
                    )
                if _strict_finite_number(
                    command_payload.get("duration_s"), positive=True
                ) is None:
                    failures.append(
                        f"{phase}_command_window_duration_not_exact_positive_number"
                    )
                created_at = command_payload.get("created_at_unix_ns")
                if type(created_at) is not int or created_at <= 0:
                    failures.append(
                        f"{phase}_command_window_created_at_not_exact_positive_int"
                    )
                try:
                    bound_trace_path = Path(
                        str(command_payload.get("trace_path") or "")
                    ).expanduser().resolve(strict=True)
                except OSError:
                    bound_trace_path = None
                if (
                    bound_trace_path != paths_by_role.get("raw")
                    or str(command_payload.get("trace_sha256") or "")
                    != hashes_by_role.get("raw")
                ):
                    failures.append(
                        f"{phase}_command_window_raw_identity_mismatch"
                    )
                try:
                    bound_result_path = Path(
                        str(command_payload.get("result_path") or "")
                    ).expanduser().resolve(strict=True)
                except OSError:
                    bound_result_path = None
                if (
                    bound_result_path
                    != paths_by_role.get("postprocessor_result")
                    or str(command_payload.get("result_sha256") or "")
                    != hashes_by_role.get("postprocessor_result")
                ):
                    failures.append(
                        f"{phase}_command_window_result_identity_mismatch"
                    )

    if len(referenced_paths) != len(set(referenced_paths)):
        failures.append("calibration_evidence_paths_not_unique")
    if failures:
        base["accelerator_idle_calibration_status"] = (
            "unavailable_binding_v2_validation_failed"
        )
        base["accelerator_idle_calibration_failure_reasons"] = failures
        return base

    base.update(
        {
            "accelerator_idle_calibration_verified": True,
            "accelerator_idle_calibration_status": "verified",
            "accelerator_idle_calibrated_at": str(binding.get("finished_at") or ""),
            "accelerator_idle_calibration_evidence": str(evidence_path or ""),
            "accelerator_idle_calibration_method_manifest": str(
                method_path or ""
            ),
            "accelerator_idle_calibration_method_sha256": method_sha,
            "accelerator_idle_calibration_runtime_binding_id": (
                runtime_binding_id
            ),
        }
    )
    return base


def resolve_energy_comparison(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return raw, normalized and effective comparison metrics for one row."""

    role_input = row.get("host_normalization_role")
    role = str(role_input or HOST_NORMALIZATION_ROLE_NONE).strip().lower()
    role_is_canonical = bool(
        role_input in (None, "")
        or (
            type(role_input) is str
            and role_input in {
                HOST_NORMALIZATION_ROLE_NONE,
                HOST_NORMALIZATION_ROLE_TENSORRT_FULL,
            }
        )
    )
    claim_flag_names = (
        "energy_efficiency_claim_eligible",
        "energy_claim_eligible",
        "scientific_primary_claim_eligible",
        "claim_eligible",
    )
    claim_flags_exact = _all_populated_literal_true(row, claim_flag_names)
    raw_total_names = (
        "avg_energy_total_j",
        "energy_total_j",
        "scientific_primary_energy_total_j",
    )
    normalized_total_names = (
        "avg_host_normalized_energy_est_j",
        "host_normalized_energy_est_j",
    )
    raw_per_work_names = (
        "row_energy_streaming_j_per_frame", "energy_streaming_j_per_frame",
        "avg_energy_per_pipeline_frame_j", "avg_energy_per_work_unit_j",
        "row_energy_latency_j_per_inference", "energy_per_inference_j",
        "energy_per_work_j", "energy_per_work_unit_j",
        "scientific_primary_energy_per_work_unit_j",
    )
    normalized_per_work_names = (
        "row_host_normalized_energy_streaming_j_per_frame_est",
        "host_normalized_energy_streaming_j_per_frame_est",
        "avg_host_normalized_energy_per_work_unit_est_j",
        "host_normalized_energy_per_work_est_j",
        "host_normalized_energy_per_work_unit_est_j",
        "row_host_normalized_energy_latency_j_per_inference_est",
    )
    raw_power_names = (
        "energy_streaming_avg_power_w", "avg_power_w", "average_power_w",
        "energy_latency_avg_power_w", "scientific_primary_avg_power_w",
    )
    normalized_power_names = (
        "host_normalized_streaming_avg_power_est_w",
        "avg_host_normalized_average_power_est_w",
        "host_normalized_average_power_est_w",
    )
    raw_per_work_std_names = (
        "row_energy_streaming_j_per_frame_sample_stddev",
        "row_energy_latency_j_per_inference_sample_stddev",
        "energy_per_work_j_sample_stddev",
        "energy_per_work_unit_j_sample_stddev",
    )
    raw_per_work_ci_low_names = (
        "row_energy_streaming_j_per_frame_ci_low",
        "row_energy_latency_j_per_inference_ci_low",
        "energy_per_work_j_ci_low",
        "energy_per_work_unit_j_ci_low",
    )
    raw_per_work_ci_high_names = (
        "row_energy_streaming_j_per_frame_ci_high",
        "row_energy_latency_j_per_inference_ci_high",
        "energy_per_work_j_ci_high",
        "energy_per_work_unit_j_ci_high",
    )
    normalized_per_work_std_names = (
        "host_normalized_energy_per_work_est_j_sample_stddev",
        "host_normalized_energy_per_work_unit_est_j_sample_stddev",
    )
    normalized_per_work_ci_low_names = (
        "host_normalized_energy_per_work_est_j_ci_low",
        "host_normalized_energy_per_work_unit_est_j_ci_low",
    )
    normalized_per_work_ci_high_names = (
        "host_normalized_energy_per_work_est_j_ci_high",
        "host_normalized_energy_per_work_unit_est_j_ci_high",
    )
    stream_raw_per_work_names = (
        "row_energy_streaming_j_per_frame",
        "energy_streaming_j_per_frame",
        "avg_energy_per_pipeline_frame_j",
    )
    latency_raw_per_work_names = (
        "row_energy_latency_j_per_inference",
        "energy_per_inference_j",
    )
    generic_raw_per_work_names = (
        "avg_energy_per_work_unit_j",
        "energy_per_work_j",
        "energy_per_work_unit_j",
        "scientific_primary_energy_per_work_unit_j",
    )
    stream_normalized_per_work_names = (
        "row_host_normalized_energy_streaming_j_per_frame_est",
        "host_normalized_energy_streaming_j_per_frame_est",
    )
    latency_normalized_per_work_names = (
        "row_host_normalized_energy_latency_j_per_inference_est",
    )
    generic_normalized_per_work_names = (
        "avg_host_normalized_energy_per_work_unit_est_j",
        "host_normalized_energy_per_work_est_j",
        "host_normalized_energy_per_work_unit_est_j",
    )
    stream_raw_power_names = ("energy_streaming_avg_power_w",)
    latency_raw_power_names = ("energy_latency_avg_power_w",)
    generic_raw_power_names = (
        "avg_power_w",
        "average_power_w",
        "scientific_primary_avg_power_w",
    )
    stream_normalized_power_names = (
        "host_normalized_streaming_avg_power_est_w",
    )
    generic_normalized_power_names = (
        "avg_host_normalized_average_power_est_w",
        "host_normalized_average_power_est_w",
    )
    streaming_present = _has_populated_metric(
        row, stream_raw_per_work_names + stream_raw_power_names
    )
    latency_present = _has_populated_metric(
        row, latency_raw_per_work_names + latency_raw_power_names
    )
    dual_phase = streaming_present and latency_present
    active_phase = (
        "streaming"
        if streaming_present
        else "latency"
        if latency_present
        else "generic"
    )
    if active_phase == "streaming":
        selected_raw_per_work_names = stream_raw_per_work_names
        selected_normalized_per_work_names = (
            stream_normalized_per_work_names
        )
        selected_raw_power_names = stream_raw_power_names
        selected_normalized_power_names = stream_normalized_power_names
    elif active_phase == "latency":
        selected_raw_per_work_names = (
            latency_raw_per_work_names + generic_raw_per_work_names
        )
        selected_normalized_per_work_names = (
            latency_normalized_per_work_names
            + generic_normalized_per_work_names
        )
        selected_raw_power_names = (
            latency_raw_power_names + generic_raw_power_names
        )
        selected_normalized_power_names = generic_normalized_power_names
    else:
        selected_raw_per_work_names = generic_raw_per_work_names
        selected_normalized_per_work_names = (
            generic_normalized_per_work_names
        )
        selected_raw_power_names = generic_raw_power_names
        selected_normalized_power_names = generic_normalized_power_names
    if active_phase == "streaming" and not dual_phase:
        selected_raw_per_work_names += generic_raw_per_work_names
        selected_normalized_per_work_names += (
            generic_normalized_per_work_names
        )
        selected_raw_power_names += generic_raw_power_names
        selected_normalized_power_names += generic_normalized_power_names
    raw_total = _first_finite(row, raw_total_names, positive=True)
    normalized_total = _first_finite(
        row, normalized_total_names, positive=True
    )
    raw_per_work = _first_finite(
        row,
        raw_per_work_names,
        positive=True,
    )
    normalized_per_work = _first_finite(
        row,
        normalized_per_work_names,
        positive=True,
    )
    raw_power = _first_finite(row, raw_power_names, positive=True)
    normalized_power = _first_finite(
        row, normalized_power_names, positive=True
    )
    strict_raw_total, raw_total_coherent = _strict_coherent_finite(
        row, raw_total_names, positive=True
    )
    strict_normalized_total, normalized_total_coherent = (
        _strict_coherent_finite(
            row, normalized_total_names, positive=True
        )
    )
    strict_raw_per_work, raw_per_work_coherent = (
        _strict_coherent_finite(
            row, selected_raw_per_work_names, positive=True
        )
    )
    strict_normalized_per_work, normalized_per_work_coherent = (
        _strict_coherent_finite(
            row, selected_normalized_per_work_names, positive=True
        )
    )
    strict_raw_power, raw_power_coherent = _strict_coherent_finite(
        row, selected_raw_power_names, positive=True
    )
    strict_normalized_power, normalized_power_coherent = (
        _strict_coherent_finite(
            row, selected_normalized_power_names, positive=True
        )
    )
    raw_per_work_std = _first_finite(row, raw_per_work_std_names)
    raw_per_work_ci_low = _first_finite(row, raw_per_work_ci_low_names)
    raw_per_work_ci_high = _first_finite(row, raw_per_work_ci_high_names)
    normalized_per_work_std = _first_finite(
        row, normalized_per_work_std_names
    )
    normalized_per_work_ci_low = _first_finite(
        row, normalized_per_work_ci_low_names
    )
    normalized_per_work_ci_high = _first_finite(
        row, normalized_per_work_ci_high_names
    )
    if active_phase == "streaming":
        raw_uncertainty_std_names = (
            "row_energy_streaming_j_per_frame_sample_stddev",
        )
        raw_uncertainty_ci_low_names = (
            "row_energy_streaming_j_per_frame_ci_low",
        )
        raw_uncertainty_ci_high_names = (
            "row_energy_streaming_j_per_frame_ci_high",
        )
    else:
        raw_uncertainty_std_names = (
            "row_energy_latency_j_per_inference_sample_stddev",
            "energy_per_work_j_sample_stddev",
            "energy_per_work_unit_j_sample_stddev",
        )
        raw_uncertainty_ci_low_names = (
            "row_energy_latency_j_per_inference_ci_low",
            "energy_per_work_j_ci_low",
            "energy_per_work_unit_j_ci_low",
        )
        raw_uncertainty_ci_high_names = (
            "row_energy_latency_j_per_inference_ci_high",
            "energy_per_work_j_ci_high",
            "energy_per_work_unit_j_ci_high",
        )
    strict_raw_per_work_std, raw_std_coherent = _strict_coherent_finite(
        row, raw_uncertainty_std_names, nonnegative=True
    )
    strict_raw_per_work_ci_low, raw_ci_low_coherent = (
        _strict_coherent_finite(row, raw_uncertainty_ci_low_names)
    )
    strict_raw_per_work_ci_high, raw_ci_high_coherent = (
        _strict_coherent_finite(row, raw_uncertainty_ci_high_names)
    )
    strict_normalized_per_work_std, normalized_std_coherent = (
        _strict_coherent_finite(
            row,
            normalized_per_work_std_names,
            nonnegative=True,
        )
    )
    strict_normalized_per_work_ci_low, normalized_ci_low_coherent = (
        _strict_coherent_finite(
            row, normalized_per_work_ci_low_names
        )
    )
    strict_normalized_per_work_ci_high, normalized_ci_high_coherent = (
        _strict_coherent_finite(
            row, normalized_per_work_ci_high_names
        )
    )
    raw_uncertainty_valid = _uncertainty_is_valid(
        strict_raw_per_work,
        strict_raw_per_work_std,
        strict_raw_per_work_ci_low,
        strict_raw_per_work_ci_high,
        aliases_coherent=(
            raw_std_coherent
            and raw_ci_low_coherent
            and raw_ci_high_coherent
        ),
        stddev_populated=_has_populated_metric(
            row, raw_uncertainty_std_names
        ),
        ci_low_populated=_has_populated_metric(
            row, raw_uncertainty_ci_low_names
        ),
        ci_high_populated=_has_populated_metric(
            row, raw_uncertainty_ci_high_names
        ),
    )
    normalized_uncertainty_valid = _uncertainty_is_valid(
        strict_normalized_per_work,
        strict_normalized_per_work_std,
        strict_normalized_per_work_ci_low,
        strict_normalized_per_work_ci_high,
        aliases_coherent=(
            normalized_std_coherent
            and normalized_ci_low_coherent
            and normalized_ci_high_coherent
        ),
        stddev_populated=_has_populated_metric(
            row, normalized_per_work_std_names
        ),
        ci_low_populated=_has_populated_metric(
            row, normalized_per_work_ci_low_names
        ),
        ci_high_populated=_has_populated_metric(
            row, normalized_per_work_ci_high_names
        ),
    )
    result: dict[str, Any] = {
        "raw_energy_total_j": raw_total,
        "host_normalized_energy_total_est_j": normalized_total,
        "raw_energy_per_work_j": raw_per_work,
        "host_normalized_energy_per_work_est_j": normalized_per_work,
        "raw_average_power_w": raw_power,
        "host_normalized_average_power_est_w": normalized_power,
        "comparison_energy_total_j": None,
        "comparison_energy_per_work_j": None,
        "comparison_average_power_w": None,
        "energy_comparison_basis": "unavailable",
        "energy_comparison_status": "unavailable",
        "energy_comparison_claim_ready": False,
        "comparison_energy_per_work_sample_stddev_j": None,
        "comparison_energy_per_work_ci_low_j": None,
        "comparison_energy_per_work_ci_high_j": None,
    }
    if role == HOST_NORMALIZATION_ROLE_NONE:
        raw_claim_ready = bool(
            type(role_input) is str
            and role_input == HOST_NORMALIZATION_ROLE_NONE
            and claim_flags_exact
            and strict_raw_per_work is not None
            and raw_total_coherent
            and raw_per_work_coherent
            and raw_power_coherent
            and raw_uncertainty_valid
        )
        result.update(
            {
                "comparison_energy_total_j": raw_total,
                "comparison_energy_per_work_j": raw_per_work,
                "comparison_average_power_w": raw_power,
                "energy_comparison_basis": "raw_measured",
                "energy_comparison_status": "raw_measured" if any(v is not None for v in (raw_total, raw_per_work, raw_power)) else "raw_energy_unavailable",
                "energy_comparison_claim_ready": raw_claim_ready,
                "comparison_energy_per_work_sample_stddev_j": (
                    strict_raw_per_work_std
                    if raw_claim_ready
                    else raw_per_work_std
                ),
                "comparison_energy_per_work_ci_low_j": (
                    strict_raw_per_work_ci_low
                    if raw_claim_ready
                    else raw_per_work_ci_low
                ),
                "comparison_energy_per_work_ci_high_j": (
                    strict_raw_per_work_ci_high
                    if raw_claim_ready
                    else raw_per_work_ci_high
                ),
            }
        )
        return result
    if role != HOST_NORMALIZATION_ROLE_TENSORRT_FULL:
        result["energy_comparison_status"] = "unsupported_host_normalization_role"
        return result

    source_input = row.get("host_normalization_source_run_id")
    variant_input = row.get("host_normalization_target_variant")
    source_id = str(source_input or "").strip().lower()
    variant = str(variant_input or "").strip().lower()
    statuses = row.get("accelerator_idle_correction_statuses")
    scalar_status = row.get("accelerator_idle_correction_status")
    statuses_present = (
        "accelerator_idle_correction_statuses" in row
        and statuses is not None
    )
    scalar_status_present = (
        "accelerator_idle_correction_status" in row
        and scalar_status is not None
    )
    correction_status_ok = bool(
        (statuses_present or scalar_status_present)
        and (
            not statuses_present
            or (type(statuses) is list and statuses == ["applied"])
        )
        and (
            not scalar_status_present
            or (type(scalar_status) is str and scalar_status == "applied")
        )
    )
    identity_ok = bool(
        role_is_canonical
        and row.get("host_normalization_identity_verified") is True
        and type(source_input) is str
        and source_input == source_id
        and source_id in TENSORRT_FULL_SOURCE_RUN_IDS
        and type(variant_input) is str
        and variant_input == variant
        and variant == "full"
    )
    binding_sha = row.get("accelerator_idle_calibration_binding_sha256")
    calibrated_at = row.get("accelerator_idle_calibrated_at")
    evidence_path = row.get("accelerator_idle_calibration_evidence")
    legacy_calibration_reference = bool(
        type(binding_sha) is str
        and _SHA256_RE.fullmatch(binding_sha) is not None
    )
    simple_calibration_reference = bool(
        type(evidence_path) is str
        and bool(evidence_path.strip())
        and evidence_path == evidence_path.strip()
        and _canonical_utc_timestamp(calibrated_at) is not None
    )
    calibration_ok = bool(
        row.get("accelerator_idle_calibration_verified") is True
        and type(row.get("accelerator_idle_calibration_status")) is str
        and row.get("accelerator_idle_calibration_status") == "verified"
        and (
            simple_calibration_reference
            or legacy_calibration_reference
        )
    )
    correction_ok = bool(
        row.get("accelerator_idle_correction_requested") is True
        and row.get("accelerator_idle_correction_applied") is True
        and correction_status_ok
    )
    normalized_total_populated = _has_populated_metric(
        row, normalized_total_names
    )
    raw_total_populated = _has_populated_metric(row, raw_total_names)
    strict_accelerator_idle_w = _strict_finite_number(
        row.get("accelerator_idle_w_applied"), positive=True
    )
    arithmetic_ok = bool(
        strict_raw_per_work is not None
        and strict_normalized_per_work is not None
        and strict_raw_power is not None
        and strict_normalized_power is not None
        and strict_accelerator_idle_w is not None
        and _arithmetic_close(
            strict_raw_power - strict_normalized_power,
            strict_accelerator_idle_w,
        )
        and _arithmetic_close(
            strict_normalized_per_work / strict_raw_per_work,
            strict_normalized_power / strict_raw_power,
        )
    )
    if arithmetic_ok and not dual_phase and normalized_total_populated:
        arithmetic_ok = bool(
            strict_raw_total is not None
            and strict_normalized_total is not None
            and _arithmetic_close(
                strict_normalized_total / strict_raw_total,
                strict_normalized_power / strict_raw_power,
            )
        )
        duration_names = (
            "avg_active_duration_s",
            "active_duration_s",
            "scientific_primary_active_duration_s",
        )
        strict_duration, duration_coherent = _strict_coherent_finite(
            row, duration_names, positive=True
        )
        if _has_populated_metric(row, duration_names):
            arithmetic_ok = bool(
                arithmetic_ok
                and duration_coherent
                and strict_duration is not None
                and _arithmetic_close(
                    strict_raw_total - strict_normalized_total,
                    strict_accelerator_idle_w * strict_duration,
                )
            )
        work_unit_names = (
            "avg_energy_work_units_used",
            "energy_work_units_used",
            "energy_work_units",
            "work_units",
        )
        strict_work_units, work_units_coherent = _strict_coherent_finite(
            row, work_unit_names, positive=True
        )
        if _has_populated_metric(row, work_unit_names):
            arithmetic_ok = bool(
                arithmetic_ok
                and work_units_coherent
                and strict_work_units is not None
                and _arithmetic_close(
                    strict_raw_total,
                    strict_raw_per_work * strict_work_units,
                )
                and _arithmetic_close(
                    strict_normalized_total,
                    strict_normalized_per_work * strict_work_units,
                )
            )
    range_ok = bool(
        raw_total_coherent
        and normalized_total_coherent
        and raw_per_work_coherent
        and normalized_per_work_coherent
        and raw_power_coherent
        and normalized_power_coherent
        and raw_total_populated == normalized_total_populated
        and (
            not normalized_total_populated
            or (
                strict_normalized_total is not None
                and strict_raw_total is not None
                and strict_normalized_total <= strict_raw_total
            )
        )
        and strict_normalized_per_work is not None
        and strict_raw_per_work is not None
        and strict_normalized_per_work <= strict_raw_per_work
        and strict_normalized_power is not None
        and strict_raw_power is not None
        and strict_normalized_power <= strict_raw_power
        and raw_uncertainty_valid
        and normalized_uncertainty_valid
        and arithmetic_ok
    )
    if not identity_ok:
        status = "required_tensorrt_full_identity_unverified"
    elif not calibration_ok:
        status = "required_tensorrt_full_calibration_unverified"
    elif not correction_ok:
        status = "required_tensorrt_full_correction_unavailable"
    elif not range_ok:
        status = "required_tensorrt_full_normalized_metrics_invalid"
    else:
        status = "host_normalized_verified"
        result.update(
            {
                "comparison_energy_total_j": normalized_total,
                "comparison_energy_per_work_j": normalized_per_work,
                "comparison_average_power_w": normalized_power,
                "energy_comparison_basis": "host_normalized_accelerator_idle_subtracted",
                "energy_comparison_claim_ready": claim_flags_exact,
                "comparison_energy_per_work_sample_stddev_j": (
                    strict_normalized_per_work_std
                ),
                "comparison_energy_per_work_ci_low_j": (
                    strict_normalized_per_work_ci_low
                ),
                "comparison_energy_per_work_ci_high_j": (
                    strict_normalized_per_work_ci_high
                ),
            }
        )
    result["energy_comparison_status"] = status
    return result


def project_energy_comparison(row: Mapping[str, Any]) -> dict[str, Any]:
    """Copy a row and attach its centrally resolved comparison projection."""

    out = dict(row)
    out.update(resolve_energy_comparison(out))
    return out


__all__ = [
    "CALIBRATION_BINDING_SCHEMA",
    "CALIBRATION_BINDING_SCHEMA_VERSION",
    "HOST_NORMALIZATION_ROLE_NONE",
    "HOST_NORMALIZATION_ROLE_TENSORRT_FULL",
    "SIMPLE_CALIBRATION_EVIDENCE_SCHEMA",
    "TENSORRT_FULL_SOURCE_RUN_IDS",
    "project_energy_comparison",
    "resolve_energy_comparison",
    "verify_accelerator_idle_calibration_binding",
]
