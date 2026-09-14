#!/usr/bin/env python3
"""Capture auditable host power, clock and thermal state for Native runs.

The collector deliberately has no third-party dependency.  NVIDIA Jetson tools
are queried when present and the corresponding sysfs state is always retained
as a fallback.  A failed optional probe is evidence, not a benchmark failure.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA = "onnx-splitpoint/native-host-telemetry"
SCHEMA_VERSION = 1


def _read_text(path: Path, *, limit: int = 256_000) -> str | None:
    try:
        if not path.is_file():
            return None
        return path.read_text(encoding="utf-8", errors="replace")[:limit].strip()
    except Exception:
        return None


def _command(argv: list[str], *, timeout: float = 5.0) -> dict[str, Any]:
    executable = shutil.which(argv[0])
    record: dict[str, Any] = {
        "argv": argv,
        "available": bool(executable),
        "resolved_executable": executable or "",
    }
    if not executable:
        record.update({"status": "unavailable", "returncode": None, "stdout": "", "stderr": ""})
        return record
    try:
        proc = subprocess.run(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
            timeout=timeout,
            check=False,
        )
        record.update({
            "status": "ok" if proc.returncode == 0 else "failed",
            "returncode": int(proc.returncode),
            "stdout": (proc.stdout or "")[-256_000:],
            "stderr": (proc.stderr or "")[-64_000:],
        })
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else str(exc.stdout or "")
        stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else str(exc.stderr or "")
        record.update({
            "status": "timeout",
            "returncode": None,
            "stdout": stdout[-256_000:],
            "stderr": stderr[-64_000:],
        })
    except Exception as exc:
        record.update({
            "status": "error",
            "returncode": None,
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}",
        })
    return record


def _glob_records(pattern: str, fields: Iterable[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for directory in sorted(Path("/").glob(pattern.lstrip("/"))):
        if not directory.is_dir():
            continue
        row: dict[str, Any] = {"path": str(directory)}
        for field in fields:
            value = _read_text(directory / field)
            if value is not None:
                row[field] = value
        if len(row) > 1:
            records.append(row)
    return records


def _thermal_zones() -> list[dict[str, Any]]:
    rows = _glob_records("/sys/class/thermal/thermal_zone*", ("type", "temp", "mode", "policy"))
    for row in rows:
        try:
            raw = float(row.get("temp"))
            row["temperature_c"] = raw / 1000.0 if abs(raw) >= 1000 else raw
        except Exception:
            pass
    return rows


def _jtop_metadata() -> dict[str, Any]:
    available = importlib.util.find_spec("jtop") is not None
    version = ""
    if available:
        for distribution in ("jetson-stats", "jtop"):
            try:
                version = importlib.metadata.version(distribution)
                break
            except Exception:
                continue
    return {
        "python_module_available": available,
        "distribution_version": version,
        "note": "State is collected through non-interactive NVIDIA commands and sysfs; jtop is not opened interactively.",
    }


def capture(
    *, phase: str, backend: str, setup_id: str, run_id: str,
    capture_group: str = "default",
) -> dict[str, Any]:
    now = time.time()
    commands = {
        "nvpmodel_mode": _command(["nvpmodel", "-q"]),
        "nvpmodel_query": _command(["nvpmodel", "-q", "--verbose"]),
        "jetson_clocks_show": _command(["jetson_clocks", "--show"]),
        # tegrastats is a streaming command.  A short, deliberately timed-out
        # probe retains one or more live samples without leaving a daemon
        # behind; sysfs below remains the authoritative non-interactive
        # fallback when the utility is absent.
        "tegrastats_sample": _command(["tegrastats", "--interval", "500"], timeout=1.6),
        "nvidia_smi_query": _command([
            "nvidia-smi",
            "--query-gpu=name,uuid,pstate,temperature.gpu,power.draw,power.limit,clocks.current.graphics,clocks.current.memory,clocks.max.graphics,clocks.max.memory",
            "--format=csv,noheader,nounits",
        ]),
    }
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "captured_at_unix_s": now,
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(now)),
        "phase": phase,
        "capture_group": str(capture_group or "default"),
        "run_id": run_id,
        "backend": backend,
        "setup_id": setup_id,
        "host": {
            "hostname": socket.gethostname(),
            "fqdn": socket.getfqdn(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "kernel": platform.release(),
            "python": sys.version,
            "pid": os.getpid(),
            "loadavg": list(os.getloadavg()) if hasattr(os, "getloadavg") else None,
            "uptime_s": _read_text(Path("/proc/uptime")),
            "jetson_release": _read_text(Path("/etc/nv_tegra_release")),
        },
        "nvidia_commands": commands,
        "jtop": _jtop_metadata(),
        "cpu_frequency": _glob_records(
            "/sys/devices/system/cpu/cpu*/cpufreq",
            (
                "scaling_cur_freq",
                "scaling_min_freq",
                "scaling_max_freq",
                "cpuinfo_min_freq",
                "cpuinfo_max_freq",
                "scaling_governor",
                "energy_performance_preference",
            ),
        ),
        "device_frequency": _glob_records(
            "/sys/class/devfreq/*",
            ("name", "cur_freq", "min_freq", "max_freq", "governor", "available_frequencies", "available_governors"),
        ),
        "thermal_zones": _thermal_zones(),
        "cooling_devices": _glob_records(
            "/sys/class/thermal/cooling_device*",
            ("type", "cur_state", "max_state"),
        ),
    }


def _identity(payload: dict[str, Any]) -> dict[str, Any]:
    commands = payload.get("nvidia_commands") if isinstance(payload.get("nvidia_commands"), dict) else {}
    nvp_mode = commands.get("nvpmodel_mode") if isinstance(commands.get("nvpmodel_mode"), dict) else {}
    nvp = commands.get("nvpmodel_query") if isinstance(commands.get("nvpmodel_query"), dict) else {}
    clocks = commands.get("jetson_clocks_show") if isinstance(commands.get("jetson_clocks_show"), dict) else {}
    cpus = payload.get("cpu_frequency") if isinstance(payload.get("cpu_frequency"), list) else []
    devfreq = payload.get("device_frequency") if isinstance(payload.get("device_frequency"), list) else []
    return {
        "hostname": str((payload.get("host") or {}).get("hostname") or ""),
        "backend": str(payload.get("backend") or ""),
        "setup_id": str(payload.get("setup_id") or ""),
        "nvpmodel": "\n".join(
            value for value in (
                str(nvp_mode.get("stdout") or "").strip(),
                str(nvp.get("stdout") or "").strip(),
            ) if value
        ),
        "jetson_clocks": str(clocks.get("stdout") or "").strip(),
        "cpu_governors": sorted({str(row.get("scaling_governor")) for row in cpus if row.get("scaling_governor")}),
        "cpu_min_max": sorted({(str(row.get("scaling_min_freq") or ""), str(row.get("scaling_max_freq") or "")) for row in cpus}),
        "device_governors": sorted({(str(row.get("path") or ""), str(row.get("governor") or "")) for row in devfreq}),
        "device_min_max": sorted({(str(row.get("path") or ""), str(row.get("min_freq") or ""), str(row.get("max_freq") or "")) for row in devfreq}),
    }


def _capture_group(payload: Mapping[str, Any], path: Path) -> str:
    """Return an explicit or backwards-compatible pre/post pair identity."""
    explicit = str(payload.get("capture_group") or "").strip()
    if explicit:
        return explicit
    phase = str(payload.get("phase") or "").strip().lower()
    stem = path.stem
    suffix = f"_{phase}" if phase else ""
    if suffix and stem.endswith(suffix):
        return stem[:-len(suffix)] or "default"
    if stem == phase:
        return "default"
    return "default"


def _configured_value_present(value: Any) -> bool:
    return value not in (None, "", [], {}, ())


def compare_root(root: Path) -> dict[str, Any]:
    files = sorted(root.glob("**/host_telemetry/*.json"))
    captures: list[dict[str, Any]] = []
    for path in files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("schema") != SCHEMA:
                continue
            captures.append({
                "path": str(path),
                "phase": payload.get("phase"),
                "capture_group": _capture_group(payload, path),
                "captured_at": payload.get("captured_at"),
                "identity": _identity(payload),
                "maximum_temperature_c": max(
                    (float(row["temperature_c"]) for row in payload.get("thermal_zones", []) if row.get("temperature_c") is not None),
                    default=None,
                ),
            })
        except Exception as exc:
            captures.append({"path": str(path), "status": "invalid", "error": f"{type(exc).__name__}: {exc}"})

    by_pair: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in captures:
        identity = row.get("identity") if isinstance(row.get("identity"), dict) else {}
        setup_key = str(identity.get("setup_id") or identity.get("hostname") or "unknown")
        capture_group = str(row.get("capture_group") or "default")
        by_pair.setdefault((setup_key, capture_group), []).append(row)
    checks: list[dict[str, Any]] = []
    configuration_fields = (
        "nvpmodel",
        "cpu_governors",
        "cpu_min_max",
        "device_governors",
        "device_min_max",
    )
    for (setup_key, capture_group), rows in sorted(by_pair.items()):
        valid = [row for row in rows if isinstance(row.get("identity"), dict)]
        pre_rows = [row for row in valid if row.get("phase") == "pre"]
        post_rows = [row for row in valid if row.get("phase") == "post"]
        pre = pre_rows[0] if len(pre_rows) == 1 else None
        post = post_rows[0] if len(post_rows) == 1 else None
        missing_fields = [
            field for field in configuration_fields
            if not (
                pre and post
                and _configured_value_present(pre["identity"].get(field))
                and _configured_value_present(post["identity"].get(field))
            )
        ]
        changed = [
            field for field in configuration_fields
            if pre and post and pre["identity"].get(field) != post["identity"].get(field)
        ]
        pair_complete = bool(len(pre_rows) == 1 and len(post_rows) == 1)
        checks.append({
            "setup_id": setup_key,
            "capture_group": capture_group,
            "pre_count": len(pre_rows),
            "post_count": len(post_rows),
            "pre_present": len(pre_rows) > 0,
            "post_present": len(post_rows) > 0,
            "pair_complete_and_unambiguous": pair_complete,
            "configuration_fields_complete": bool(pair_complete and not missing_fields),
            "stable_configuration": bool(pair_complete and not missing_fields and not changed),
            "missing_fields": missing_fields,
            "changed_fields": changed,
            "pre_maximum_temperature_c": pre.get("maximum_temperature_c") if pre else None,
            "post_maximum_temperature_c": post.get("maximum_temperature_c") if post else None,
        })

    setup_ids = sorted({str(row.get("setup_id") or "unknown") for row in checks})
    expected_pre_rows: list[tuple[str, str, dict[str, Any] | None]] = []
    for check in checks:
        pair_rows = by_pair.get((str(check["setup_id"]), str(check["capture_group"])), [])
        pre_rows = [row for row in pair_rows if row.get("phase") == "pre" and isinstance(row.get("identity"), dict)]
        expected_pre_rows.append((
            str(check["setup_id"]),
            str(check["capture_group"]),
            pre_rows[0]["identity"] if len(pre_rows) == 1 else None,
        ))

    cross_setup = []
    for field in configuration_fields:
        observed = [
            (setup_id, capture_group, identity.get(field))
            for setup_id, capture_group, identity in expected_pre_rows
            if identity is not None and _configured_value_present(identity.get(field))
        ]
        encoded = {
            json.dumps(value, sort_keys=True, ensure_ascii=False)
            for _setup_id, _capture_group, value in observed
        }
        observed_setup_ids = sorted({setup_id for setup_id, _capture_group, _value in observed})
        expected_capture_count = len(expected_pre_rows)
        complete = bool(
            expected_capture_count
            and len(observed) == expected_capture_count
            and observed_setup_ids == setup_ids
        )
        cross_setup.append({
            "field": field,
            "expected_setup_count": len(setup_ids),
            "observed_setup_count": len(observed_setup_ids),
            "expected_pre_capture_count": expected_capture_count,
            "observed_pre_capture_count": len(observed),
            "complete_across_setups_and_capture_groups": complete,
            "distinct_value_count": len(encoded),
            "equal_across_observed_setups": bool(complete and len(encoded) == 1),
        })
    complete_pairs = [row for row in checks if row["pair_complete_and_unambiguous"]]
    return {
        "schema": "onnx-splitpoint/native-host-telemetry-summary",
        "schema_version": 2,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_root": str(root),
        "capture_count": len(captures),
        "setup_count": len(setup_ids),
        "capture_group_count": len(checks),
        "complete_pair_count": len(complete_pairs),
        "stable_pair_count": sum(bool(row["stable_configuration"]) for row in checks),
        "all_pairs_complete_and_stable": bool(checks) and all(row["stable_configuration"] for row in checks),
        # Backwards-compatible field name, kept fail-closed when any requested
        # capture group is missing, duplicated, incomplete or changed.
        "all_complete_pairs_stable": bool(checks) and all(row["stable_configuration"] for row in checks),
        "cross_setup_configuration": cross_setup,
        "all_observed_setup_configurations_equal": bool(cross_setup) and all(
            row["equal_across_observed_setups"] for row in cross_setup
        ),
        "checks": checks,
        "captures": captures,
        "claim_note": "Current clocks and temperatures are observations, not equality constraints. Power mode, governors and configured min/max frequencies are checked for pre/post drift.",
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--phase", choices=("pre", "post"), default="pre")
    parser.add_argument("--backend", default="")
    parser.add_argument("--setup-id", default="")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--capture-group", default="default")
    parser.add_argument("--compare-root", default="")
    args = parser.parse_args()
    output = Path(args.output).expanduser()
    if args.compare_root:
        payload = compare_root(Path(args.compare_root).expanduser())
    else:
        payload = capture(
            phase=args.phase,
            backend=args.backend,
            setup_id=args.setup_id,
            run_id=args.run_id,
            capture_group=args.capture_group,
        )
    _write_json(output, payload)
    print(json.dumps({"ok": True, "output": str(output), "schema": payload.get("schema")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
