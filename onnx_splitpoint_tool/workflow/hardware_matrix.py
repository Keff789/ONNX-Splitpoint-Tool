from __future__ import annotations

"""Hardware target matrix helpers for Evaluation Profiles.

The Evaluation Workflow treats an accelerator setup as two separate things:

* a build environment (local compiler/cache, e.g. Hailo DFC or DeepX DX-COM);
* a runtime setup (remote NX host, venv/setup command, provider and benchmark knobs).

This module normalizes the lightweight YAML blocks used by the GUI and runner.
It does not perform builds or SSH itself; it only gives the workflow a stable,
serializable hardware matrix artifact that downstream stages can consume.
"""

import json
import hashlib
import os
import re
import shlex
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import yaml

from ..native_full_quality import enabled_run_profiles


def _slug(value: Any, fallback: str = "target") -> str:
    import re
    s = str(value or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s or fallback


def canon_accelerator(value: Any) -> str:
    s = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if "deepx" in s or "dx_m1" in s or "dxm1" in s:
        return "deepx_m1"
    if "hailo10" in s or "hailo_10" in s:
        # Do not treat the leading 'h' of 'hailo10' as hailo10h.  Only explicit
        # variants such as hailo10h/hailo-10h and hailo10n/hailo-10n map to
        # variant tokens; a plain Hailo-10 setup stays canonical 'hailo10'.
        if "10n" in s or "10_n" in s or "hailo_10n" in s or s.endswith("_n"):
            return "hailo10n"
        if "10h" in s or "10_h" in s or "hailo_10h" in s or s.endswith("_h"):
            return "hailo10h"
        return "hailo10"
    if "hailo8" in s or s == "hailo":
        return "hailo8"
    if s in {"trt", "tensor_rt"}:
        return "tensorrt"
    if s in {"cpu", "ort_cpu"}:
        return "cpu_ort"
    if s in {"cuda", "ort_cuda"}:
        return "cuda_ort"
    return s


def accelerator_provider(accelerator: str) -> str:
    acc = canon_accelerator(accelerator)
    if acc.startswith("hailo"):
        return acc
    if acc == "deepx_m1":
        return "deepx_m1"
    if acc == "tensorrt":
        return "tensorrt"
    if acc == "cuda_ort":
        return "cuda"
    if acc == "cpu_ort":
        return "cpu"
    return acc or "auto"


def _as_list(value: Any) -> List[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    return []


def _as_mapping(value: Any) -> Dict[str, Any]:
    return dict(value or {}) if isinstance(value, Mapping) else {}


def _normalise_ssh_args(value: str) -> str:
    try:
        return shlex.join(shlex.split(value))
    except ValueError as exc:
        raise ValueError("remote identity ssh_extra_args are malformed") from exc


def _require_unambiguous_remote_identity(setup: Mapping[str, Any]) -> None:
    """Reject conflicting endpoint aliases before profile materialisation.

    Registry, legacy and inline profile shapes may all describe the same
    Jetson.  Precedence is unsafe for an energy claim: a stale secondary alias
    could bind energy evidence to one host while SSH targets another.
    """
    containers: list[tuple[str, Mapping[str, Any]]] = []
    host_raw = setup.get("host")
    if isinstance(host_raw, Mapping):
        containers.append(("host", host_raw))
    elif host_raw not in (None, "") and not isinstance(host_raw, str):
        raise ValueError("remote identity host alias must be a string or mapping")
    for name in ("remote", "remote_execution", "runtime"):
        raw = setup.get(name)
        if raw is None:
            continue
        if not isinstance(raw, Mapping):
            raise ValueError(f"remote identity {name} alias must be a mapping")
        containers.append((name, raw))

    candidates: dict[str, list[tuple[str, Any]]] = {
        "address": [],
        "user": [],
        "port": [],
        "ssh_extra_args": [],
    }
    if isinstance(host_raw, str) and host_raw:
        candidates["address"].append(("host", host_raw))
    for key in ("address", "user", "port", "ssh_extra_args"):
        if key in setup:
            candidates[key].append((f"setup.{key}", setup.get(key)))
    for name, mapping in containers:
        for key in ("address", "host"):
            if key in mapping:
                candidates["address"].append(
                    (f"{name}.{key}", mapping.get(key))
                )
        for key in ("user", "port", "ssh_extra_args"):
            if key in mapping:
                candidates[key].append(
                    (f"{name}.{key}", mapping.get(key))
                )

    normalised: dict[str, list[tuple[str, Any]]] = {
        key: [] for key in candidates
    }
    for name, value in candidates["address"]:
        if value in (None, ""):
            continue
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
            raise ValueError(f"remote identity {name} must be a canonical string")
        normalised["address"].append((name, value))
    for name, value in candidates["user"]:
        if value in (None, ""):
            continue
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
            raise ValueError(f"remote identity {name} must be a canonical string")
        normalised["user"].append((name, value))
    for name, value in candidates["port"]:
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 1 <= value <= 65535
        ):
            raise ValueError(f"remote identity {name} must be an integer port")
        normalised["port"].append((name, value))
    for name, value in candidates["ssh_extra_args"]:
        if value in (None, ""):
            continue
        if not isinstance(value, str):
            raise ValueError(f"remote identity {name} must be a string")
        normalised["ssh_extra_args"].append(
            (name, _normalise_ssh_args(value))
        )

    for field, values in normalised.items():
        distinct = {value for _name, value in values}
        if len(distinct) > 1:
            sources = ", ".join(name for name, _value in values)
            raise ValueError(
                f"conflicting remote identity aliases for {field}: {sources}"
            )


def _expand_path(value: Any) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(value or ""))))


def default_hardware_setups_file() -> Path:
    override = str(
        os.environ.get("ONNX_SPLITPOINT_HARDWARE_SETUPS_FILE") or ""
    ).strip()
    return _expand_path(
        override or "~/.onnx_splitpoint_tool/hardware_setups.yaml"
    )


def _default_hardware_registry() -> Dict[str, Any]:
    """Default fleet registry used when the user has not created one yet."""
    default_energy = {
        "enabled": False,
        "urecs_address": "",
        "idle_baseline_w": None,
        "accelerator_idle_w": None,
        "full_system_current_scale_factor": None,
        "full_system_current_scale_calibrated_at": "",
        "full_system_current_scale_calibration_evidence": "",
        "full_system_current_scale_calibration_sha256": "",
        # Full-system input calibration is a separately verified scientific
        # method manifest.  Platform idle calibration resolves these exact
        # setup-bound fields before any rail state is changed.
        "calibration_manifest": "",
        "calibration_sha256": "",
    }
    default_power_control = {
        "enabled": True,
        "udp_port": 3000,
        "udp_terminator": "lf",
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
    return {
        "schema": "onnx-splitpoint/hardware-setups",
        "schema_version": 2,
        "energy_defaults": {
            "enabled": False,
            "collector_binary": "urecs-data-collector",
            "power_calculations_binary": "power_calculations",
            "mode": "fast_firmware",
            "data_port": 3000,
            "channel": 0,
            "sample_rate": 2000,
            "physical_scope": "FS",
            "window_label": "command",
            "environment": "Jetson",
            "pre_duration_s": 5,
            "post_duration_s": 5,
            "duration_margin_s": 1.0,
            "power_estimated_duration_margin_s": 2.0,
            "run_count": 1,
            "keep_raw_parquet": True,
            "postprocess_with_power_calculations": True,
            "include_raw_parquet_in_debug_pack": False,
        },
        "hardware_setups": [
            {
                "id": "orin_nx_hailo8_01",
                "label": "Orin NX + Hailo-8",
                "accelerator": "hailo8",
                "host": {"address": "", "user": "nx", "port": 22, "base_dir": "~/splitpoint_runs"},
                "runtime": {
                    "kind": "hailort",
                    "provider": "hailo8",
                    "venv": "~/venvs/splitpoint-hailo8/bin/activate",
                    "activate": "source ~/venvs/splitpoint-hailo8/bin/activate",
                    "preflight": ["hailortcli scan", "python -c 'import hailo_platform; print(\"hailo OK\")'"],
                },
                "build": {"kind": "hailo_dfc", "arch": "hailo8", "mode": "reuse_and_build_missing", "environment_id": "hailo8_dfc_managed"},
                "energy": dict(default_energy),
                "power_control": dict(default_power_control),
                "tags": ["remote", "npu", "hailo", "hailo8"],
            },
            {
                "id": "orin_nx_hailo10_01",
                "label": "Orin NX + Hailo-10",
                "accelerator": "hailo10",
                "host": {"address": "", "user": "nx", "port": 22, "base_dir": "~/splitpoint_runs"},
                "runtime": {
                    "kind": "hailort",
                    "provider": "hailo10",
                    "venv": "~/venvs/splitpoint-hailo10/bin/activate",
                    "activate": "source ~/venvs/splitpoint-hailo10/bin/activate",
                    "preflight": ["hailortcli scan", "python -c 'import hailo_platform; print(\"hailo OK\")'"],
                },
                "build": {"kind": "hailo_dfc", "arch": "hailo10", "mode": "reuse_and_build_missing", "environment_id": "hailo10_dfc_managed"},
                "energy": dict(default_energy),
                "power_control": dict(default_power_control),
                "tags": ["remote", "npu", "hailo", "hailo10"],
            },
            {
                "id": "orin_nx_deepx_m1_01",
                "label": "Orin NX + DeepX DX-M1",
                "accelerator": "deepx_m1",
                "host": {"address": "", "user": "nx", "port": 22, "base_dir": "~/splitpoint_runs"},
                "runtime": {
                    "kind": "dxrt",
                    "provider": "deepx_m1",
                    "venv": "~/venvs/deepx-runtime/bin/activate",
                    "activate": "source ~/venvs/deepx-runtime/bin/activate",
                    "preflight": ["ls -l /dev/dxrt*", "dxrt-cli -s", "python -c 'from dx_engine import InferenceEngine; print(\"dx_engine OK\")'"],
                },
                "build": {"kind": "deepx_dxcom", "mode": "reuse_and_build_missing", "environment_id": "deepx_dxcom_x86"},
                "energy": dict(default_energy),
                "power_control": dict(default_power_control),
                "tags": ["remote", "npu", "deepx", "dx_m1"],
            },
        ],
        "hardware_groups": {
            "all_accelerators": ["orin_nx_hailo8_01", "orin_nx_hailo10_01", "orin_nx_deepx_m1_01"],
            "hailo": ["orin_nx_hailo8_01", "orin_nx_hailo10_01"],
            "deepx": ["orin_nx_deepx_m1_01"],
        },
        "build_environments": [
            {"id": "hailo8_dfc_managed", "kind": "hailo8_dfc", "host": "local", "shell": "bash", "venv_activate": "source ~/.onnx_splitpoint_tool/hailo/venv_hailo8/bin/activate", "cache_dir": "~/Models/BackendArtifacts/hailo"},
            {"id": "hailo10_dfc_managed", "kind": "hailo10_dfc", "host": "local", "shell": "bash", "venv_activate": "source ~/.onnx_splitpoint_tool/hailo/venv_hailo10/bin/activate", "cache_dir": "~/Models/BackendArtifacts/hailo"},
            {"id": "deepx_dxcom_x86", "kind": "deepx_dxcom", "host": "local", "shell": "bash", "dx_all_suite_root": "~/dx-all-suite", "venv_activate": "source ~/dx-all-suite/dx-compiler/venv-dx-compiler-local/bin/activate", "compiler_venv": "~/dx-all-suite/dx-compiler/venv-dx-compiler-local", "runtime_venv": "~/venvs/deepx-runtime", "cache_dir": "~/Models/BackendArtifacts/deepx"},
        ],
    }


def ensure_hardware_setups_file(path: str | Path | None = None) -> Path:
    p = _expand_path(path) if path else default_hardware_setups_file()
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        p.write_text(yaml.safe_dump(_default_hardware_registry(), sort_keys=False, allow_unicode=True), encoding="utf-8")
    else:
        # Existing central registries are upgraded through the same atomic,
        # narrowly scoped migration used by Tool Config.  The local import
        # avoids a module-import cycle with energy.config's default merger.
        from onnx_splitpoint_tool.energy.config import load_hardware_registry

        load_hardware_registry(p)
    return p


def _load_hardware_registry(profile: Mapping[str, Any]) -> tuple[Dict[str, Any], str]:
    hw = _as_mapping(profile.get("hardware"))
    file_value = hw.get("setups_file") or hw.get("config_file") or profile.get("hardware_setups_file") or ""
    registry: Dict[str, Any] = {}
    source = ""
    if file_value:
        p = _expand_path(file_value)
        source = str(p)
        if p.exists():
            data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            registry = dict(data) if isinstance(data, Mapping) else {}
        else:
            registry = dict(_default_hardware_registry())
    else:
        p = ensure_hardware_setups_file()
        source = str(p)
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        registry = dict(data) if isinstance(data, Mapping) else {}
    # A copied/frozen registry must not be rewritten merely because it is read,
    # but runtime normalization still needs the corrected firmware token.  The
    # central default path was already persisted atomically by ``ensure``.
    from onnx_splitpoint_tool.energy.config import _migrate_legacy_m2_command_defaults

    _migrate_legacy_m2_command_defaults(registry)
    # Inline hardware sections are legacy/sample-profile conveniences.  The
    # central registry ~/.onnx_splitpoint_tool/hardware_setups.yaml is the source
    # of truth for real machines.  Older built-in profiles contain placeholder
    # hardware_setups and used to override the central file, making configured
    # hosts/u.RECS addresses appear to reset after profile/tool changes.  Only
    # allow inline overrides when explicitly requested.
    hw_allow_inline = False
    try:
        hw_allow_inline = str(hw.get("allow_inline_setups") or profile.get("allow_inline_hardware_setups") or profile.get("use_inline_hardware_setups") or os.environ.get("ONNX_SPLITPOINT_ALLOW_INLINE_HARDWARE_SETUPS", "")).strip().lower() in {"1", "true", "yes", "on"}
    except Exception:
        hw_allow_inline = False
    if hw_allow_inline and isinstance(profile.get("hardware_setups"), list):
        registry["hardware_setups"] = list(profile.get("hardware_setups") or [])
    if hw_allow_inline and isinstance(profile.get("hardware_groups"), Mapping):
        registry["hardware_groups"] = dict(profile.get("hardware_groups") or {})
    if hw_allow_inline and isinstance(profile.get("build_environments"), list):
        registry["build_environments"] = list(profile.get("build_environments") or [])
    return registry, source


def _split_csv(v: Any) -> List[str]:
    if isinstance(v, Sequence) and not isinstance(v, (str, bytes, bytearray)):
        return [str(x).strip() for x in v if str(x).strip()]
    return [x.strip() for x in str(v or "").split(",") if x.strip()]


def _normalize_setup_id_v59n(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    m = re.search(r"orin_nx_[A-Za-z0-9_]+", text)
    if m:
        return m.group(0)
    return text.strip("[](){} \t\r\n\"'").strip()

def _normalize_setup_ids_v59n(values: Any) -> List[str]:
    out: List[str] = []
    for item in _split_csv(values):
        sid = _normalize_setup_id_v59n(item)
        if sid and sid not in out:
            out.append(sid)
    return out


def _registry_selected_setup_ids(profile: Mapping[str, Any], registry: Mapping[str, Any]) -> List[str]:
    hw = _as_mapping(profile.get("hardware"))
    selected = _normalize_setup_ids_v59n(hw.get("selected_setups") or profile.get("selected_hardware_setups"))
    groups = _split_csv(hw.get("selected_groups") or profile.get("selected_hardware_groups"))
    group_map = _as_mapping(registry.get("hardware_groups") or registry.get("groups"))
    for gid in groups:
        for sid in _normalize_setup_ids_v59n(group_map.get(gid)):
            if sid not in selected:
                selected.append(sid)
    return selected


def _setup_to_target(setup: Mapping[str, Any], *, build_envs: Sequence[Mapping[str, Any]], registry_source: str) -> Dict[str, Any]:
    _require_unambiguous_remote_identity(setup)
    acc = canon_accelerator(setup.get("accelerator") or setup.get("backend") or setup.get("target") or setup.get("kind") or setup.get("id"))
    host_raw = setup.get("host")
    host = _as_mapping(host_raw)
    runtime = _as_mapping(setup.get("runtime"))
    remote = _as_mapping(setup.get("remote") or setup.get("remote_execution"))
    # Profiles produced by the GUI may use flat fields (host/user/remote_venv).
    # Registry-style configs may use nested host/runtime/remote blocks.  Support
    # both so one Evaluation Profile can describe three independent Orin NX
    # deployments without manual re-entry.
    flat_host = str(host_raw or setup.get("address") or "") if not isinstance(host_raw, Mapping) else ""
    flat_user = str(setup.get("user") or "")
    flat_port = setup.get("port")
    flat_base = str(setup.get("remote_base_dir") or setup.get("base_dir") or "")
    flat_venv = str(setup.get("remote_venv") or setup.get("venv") or setup.get("venv_activate") or "")
    flat_provider = str(setup.get("provider") or "")
    merged_remote = {
        "enabled": bool(setup.get("enabled", remote.get("enabled", runtime.get("enabled", True)))),
        "host_id": str(setup.get("host_id") or setup.get("id") or ""),
        "physical_host_id": str(setup.get("physical_host_id") or host.get("physical_host_id") or remote.get("physical_host_id") or runtime.get("physical_host_id") or ""),
        "host": str(host.get("address") or host.get("host") or remote.get("host") or runtime.get("host") or flat_host or ""),
        "user": str(host.get("user") or remote.get("user") or runtime.get("user") or flat_user or ""),
        "port": int(host.get("port") or remote.get("port") or runtime.get("port") or flat_port or 22),
        "remote_base_dir": str(host.get("base_dir") or remote.get("remote_base_dir") or runtime.get("remote_base_dir") or flat_base or "~/splitpoint_runs"),
        "remote_venv": str(runtime.get("activate") or runtime.get("venv_activate") or runtime.get("venv") or remote.get("remote_venv") or remote.get("venv") or flat_venv or ""),
        "provider": str(runtime.get("provider") or remote.get("provider") or flat_provider or accelerator_provider(acc)),
        "transfer_mode": str(remote.get("transfer_mode") or setup.get("transfer_mode") or "bundle"),
        "reuse_bundle": bool(remote.get("reuse_bundle", setup.get("reuse_bundle", True))),
        "resume": bool(remote.get("resume", setup.get("resume", True))),
        "warmup": int(remote.get("warmup") or runtime.get("warmup") or setup.get("warmup") or 10),
        "iters": int(remote.get("iters") or runtime.get("iters") or setup.get("iters") or 50),
        "timeout_s": int(remote.get("timeout_s") or runtime.get("timeout_s") or setup.get("timeout_s") or 0),
        "ssh_extra_args": str(remote.get("ssh_extra_args") or host.get("ssh_extra_args") or setup.get("ssh_extra_args") or ""),
    }
    merged_remote["host_copy"] = _host_copy_from_remote(merged_remote)
    build = _as_mapping(setup.get("build"))
    env_id = str(build.get("environment_id") or setup.get("build_environment_id") or setup.get("build_env") or "")
    env = next((dict(x) for x in build_envs if str(x.get("id") or "") == env_id), {}) if env_id else {}
    return {
        "id": str(setup.get("id") or f"{acc}_target"),
        "label": str(setup.get("label") or setup.get("name") or acc),
        "accelerator": acc,
        "provider": str(setup.get("provider") or merged_remote.get("provider") or accelerator_provider(acc)),
        "enabled": bool(setup.get("enabled", True)),
        "runtime": merged_remote,
        "remote": merged_remote,
        "build_environment_id": env_id,
        "build_environment": env,
        "energy": dict(setup.get("energy") or {}) if isinstance(setup.get("energy"), Mapping) else {},
        "setup_source": registry_source,
        "setup_lock_id": str(setup.get("lock_id") or setup.get("id") or f"{acc}_target"),
        "tags": list(setup.get("tags") or []),
        "note": str(setup.get("note") or "Resolved from hardware setup registry."),
    }


def run_profile_accelerators(run_profiles: Sequence[Mapping[str, Any]]) -> List[str]:
    out: List[str] = []
    for raw in enabled_run_profiles(run_profiles):
        for key in ("id", "full", "stage1", "stage2", "full_reference"):
            acc = canon_accelerator(raw.get(key))
            if acc in {"hailo8", "hailo10", "hailo10n", "hailo10h", "deepx_m1"} and acc not in out:
                out.append(acc)
    return out


def _host_copy_from_remote(remote: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key in ("id", "label", "host", "user", "port", "remote_base_dir", "ssh_extra_args"):
        if key in remote and remote.get(key) not in (None, ""):
            out[key] = remote.get(key)
    return out


def _default_runtime_setup(profile: Mapping[str, Any]) -> Dict[str, Any]:
    remote = _as_mapping(profile.get("remote_execution") or profile.get("remote"))
    return {
        "enabled": bool(remote.get("enabled", False)),
        "host_id": str(remote.get("host_id") or remote.get("id") or ""),
        "host": str(remote.get("host") or ""),
        "user": str(remote.get("user") or ""),
        "port": int(remote.get("port") or 22),
        "remote_base_dir": str(remote.get("remote_base_dir") or "~/splitpoint_runs"),
        "remote_venv": str(remote.get("remote_venv") or remote.get("venv") or ""),
        "transfer_mode": str(remote.get("transfer_mode") or "bundle"),
        "reuse_bundle": bool(remote.get("reuse_bundle", True)),
        "resume": bool(remote.get("resume", True)),
        "warmup": int(remote.get("warmup") or 10),
        "iters": int(remote.get("iters") or 50),
        "timeout_s": int(remote.get("timeout_s") or 0),
        "ssh_extra_args": str(remote.get("ssh_extra_args") or ""),
        "host_copy": _host_copy_from_remote(remote),
    }


def normalize_hardware_targets(profile_payload: Mapping[str, Any]) -> List[Dict[str, Any]]:
    profile = dict(profile_payload or {}) if isinstance(profile_payload, Mapping) else {}
    hardware = _as_mapping(profile.get("hardware"))
    frozen_targets = hardware.get("resolved_targets")
    if bool(hardware.get("resolution_frozen_at_start")) and isinstance(frozen_targets, list):
        # EvalRun start snapshots embed the concrete physical mapping.  Never
        # re-read the mutable central registry after the user reviewed/started
        # that snapshot.
        rows = [dict(row) for row in frozen_targets if isinstance(row, Mapping)]
        actual_hash = "sha256:" + hashlib.sha256(json.dumps(
            rows,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")).hexdigest()
        expected_hash = str(hardware.get("resolved_targets_sha256") or "")
        if not expected_hash or expected_hash != actual_hash:
            raise ValueError("Frozen hardware target mapping hash mismatch; refusing to reload the mutable registry.")
        return rows
    registry, registry_source = _load_hardware_registry(profile)
    registry_build_envs = [dict(x) for x in _as_list(registry.get("build_environments")) if isinstance(x, Mapping)]
    registry_setups = [dict(x) for x in _as_list(registry.get("hardware_setups")) if isinstance(x, Mapping)]
    selected_ids = _registry_selected_setup_ids(profile, registry)

    # v60n: Evaluation Profiles select logical run profiles only.  Resolve the
    # concrete machines automatically from the central hardware registry.  One
    # enabled setup per requested accelerator is selected in registry order.
    run_profiles_early = [dict(x) for x in _as_list(profile.get("run_profiles")) if isinstance(x, Mapping)]
    requested_accs_early = run_profile_accelerators(run_profiles_early)
    if not selected_ids and registry_setups:
        def _matches(requested: str, actual: str) -> bool:
            r = canon_accelerator(requested)
            a = canon_accelerator(actual)
            if r == a:
                return True
            if r.startswith("hailo10") and a.startswith("hailo10"):
                return True
            return False

        for requested in requested_accs_early:
            for setup in registry_setups:
                if not bool(setup.get("enabled", True)):
                    continue
                if _matches(requested, setup.get("accelerator") or setup.get("backend") or setup.get("id")):
                    sid = str(setup.get("id") or "").strip()
                    if sid and sid not in selected_ids:
                        selected_ids.append(sid)
                    break

        # A TensorRT/CUDA-only remote run still needs one physical Jetson setup.
        # Pick the first enabled setup; pure CPU-only profiles may remain local.
        has_host_runtime = any(
            canon_accelerator(rp.get(k)) in {"tensorrt", "cuda_ort"}
            for rp in run_profiles_early
            for k in ("id", "full", "stage1", "stage2")
        )
        if has_host_runtime and not selected_ids:
            first = next((x for x in registry_setups if bool(x.get("enabled", True)) and str(x.get("id") or "").strip()), None)
            if first is not None:
                selected_ids.append(str(first.get("id")))

    # Preferred path. A profile may still select concrete setups/groups for
    # groups from a reusable registry.  This is what lets one eval run dispatch
    # Hailo-8, Hailo-10 and DeepX jobs to separate Orin NX hosts.
    if selected_ids:
        by_id = {str(x.get("id") or ""): x for x in registry_setups}
        out: List[Dict[str, Any]] = []
        for sid in selected_ids:
            raw = by_id.get(str(sid))
            if not raw:
                out.append({
                    "id": str(sid),
                    "label": str(sid),
                    "accelerator": "",
                    "provider": "",
                    "enabled": False,
                    "runtime": {},
                    "remote": {},
                    "build_environment_id": "",
                    "build_environment": {},
                    "setup_source": registry_source,
                    "setup_lock_id": str(sid),
                    "note": "Selected hardware setup id was not found in the registry.",
                    "error_class": "hardware_setup_not_found",
                })
                continue
            out.append(_setup_to_target(raw, build_envs=registry_build_envs, registry_source=registry_source))
        return out

    # Compatibility path: older profiles may embed hardware_targets, hardware_setups
    # or deployment_targets directly.  This keeps the v49-v51 profiles working.
    raw_targets = _as_list(profile.get("hardware_targets") or profile.get("deployment_targets"))
    # If hardware_setups is an inline list and no selected ids are present, treat
    # it as direct runtime targets for backwards compatibility.
    if not raw_targets and isinstance(profile.get("hardware_setups"), list):
        raw_targets = _as_list(profile.get("hardware_setups"))
    run_profiles = [dict(x) for x in _as_list(profile.get("run_profiles")) if isinstance(x, Mapping)]
    requested_accs = run_profile_accelerators(run_profiles)
    default_runtime = _default_runtime_setup(profile)
    build_envs = [dict(x) for x in _as_list(profile.get("build_environments")) if isinstance(x, Mapping)] or registry_build_envs

    by_kind: Dict[str, Dict[str, Any]] = {}
    for env in build_envs:
        kind = canon_accelerator(env.get("kind") or env.get("target") or env.get("accelerator"))
        if "hailo8" in str(env.get("kind") or "").lower():
            kind = "hailo8"
        elif "hailo10" in str(env.get("kind") or "").lower():
            kind = "hailo10"
        elif "deepx" in str(env.get("kind") or "").lower():
            kind = "deepx_m1"
        if kind and kind not in by_kind:
            by_kind[kind] = env

    normalized: List[Dict[str, Any]] = []
    for item in raw_targets:
        if not isinstance(item, Mapping):
            continue
        acc = canon_accelerator(item.get("accelerator") or item.get("target") or item.get("backend") or item.get("kind") or item.get("id"))
        if not acc:
            continue
        resolved = _setup_to_target(item, build_envs=build_envs, registry_source=registry_source)
        # If the inline entry only names an accelerator, inherit legacy profile
        # remote defaults. Flat inline fields still win via _setup_to_target().
        merged_runtime = dict(default_runtime)
        merged_runtime.update({k: v for k, v in dict(resolved.get("remote") or {}).items() if v not in (None, "")})
        if not merged_runtime.get("host") and default_runtime.get("host"):
            merged_runtime["host"] = default_runtime.get("host")
        if "host_copy" not in merged_runtime or not merged_runtime.get("host_copy"):
            merged_runtime["host_copy"] = _host_copy_from_remote(merged_runtime)
        resolved["runtime"] = merged_runtime
        resolved["remote"] = merged_runtime
        if not resolved.get("build_environment"):
            env = by_kind.get(acc) or (by_kind.get("hailo10") if acc.startswith("hailo10") else {})
            resolved["build_environment"] = env
            resolved["build_environment_id"] = str(env.get("id") or resolved.get("build_environment_id") or "")
        normalized.append(resolved)

    # If no explicit hardware targets are present, derive a simple matrix from
    # selected run profiles so older profiles still get a useful artifact.
    if not normalized:
        for acc in requested_accs:
            env = by_kind.get(acc) or (by_kind.get("hailo10") if acc.startswith("hailo10") else {})
            normalized.append({
                "id": f"{acc}_default",
                "label": acc,
                "accelerator": acc,
                "provider": accelerator_provider(acc),
                "enabled": True,
                "runtime": dict(default_runtime),
                "remote": dict(default_runtime),
                "build_environment_id": str(env.get("id") or ""),
                "build_environment": env,
                "setup_source": registry_source,
                "setup_lock_id": f"{acc}_default",
                "note": "Derived from run_profiles because no hardware.selected_setups/group was present.",
            })
    return normalized

def matrix_for_runtime(profile_payload: Mapping[str, Any]) -> List[Dict[str, Any]]:
    return [x for x in normalize_hardware_targets(profile_payload) if x.get("enabled")]


def write_hardware_matrix_artifact(path: str | Path, profile_payload: Mapping[str, Any]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    matrix = normalize_hardware_targets(profile_payload)
    payload = {
        "schema": "onnx-splitpoint/evaluation-hardware-matrix",
        "schema_version": 1,
        "hardware_target_count": len(matrix),
        "hardware_targets": matrix,
        "notes": [
            "Each hardware target is a deploy/run setup: accelerator token, build environment, and remote runtime host.",
            "The Evaluation Workflow can use this artifact to dispatch the same benchmarkset to multiple Orin NX setups.",
        ],
    }
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return p
