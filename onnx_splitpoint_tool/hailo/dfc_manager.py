"""DFC (Dataflow Compiler) profile manager for Hailo.

Problem
-------
Hailo-8 and Hailo-10 require different DFC compiler versions. Users often run
this tool on Windows while keeping the DFC installed in WSL2.

This module provides a small indirection layer:

    hw_arch ("hailo8", "hailo10", ...) -> profile -> (WSL distro, venv activate)

The manager is *pure Python* and does not import `hailo_sdk_client`.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


_AUTO_SENTINELS = {"", "auto", "managed", "default"}


@dataclass(frozen=True)
class DfcProfile:
    """A DFC installation profile.

    The profile describes how to reach a specific DFC version (usually via a
    dedicated virtualenv). Profiles can be used on native Linux or via WSL.

    Attributes
    ----------
    profile_id:
        Stable identifier (e.g. "hailo8", "hailo10").
    hw_arch_prefixes:
        A list of prefixes that map `hw_arch` strings to this profile.
        Example: ["hailo8"] matches "hailo8", "hailo8l", "hailo8r".
    wsl_venv_activate:
        Path to the venv activation script *inside Linux/WSL*.
        This should be a path like "~/.onnx_splitpoint_tool/.../bin/activate".
    wsl_distro:
        Optional WSL distro name to use for this profile (Windows only).
        If empty/None, the default WSL distro is used.
    wheel_dir:
        Optional relative path (from this package's resources/hailo/) where the
        DFC wheel(s) for this profile are stored.
    notes:
        Optional free-form note displayed in debug output.

    glibc_min:
        Optional minimum required glibc version ("major.minor") for the DFC
        wheel/runtime on this profile. This is used for fast pre-flight checks
        so we can warn users early when running on an older distro (e.g.
        Ubuntu 20.04 glibc 2.31 vs. DFC wheels requiring >= 2.34).
    """

    profile_id: str
    hw_arch_prefixes: Tuple[str, ...]
    wsl_venv_activate: str
    wsl_distro: Optional[str] = None
    wheel_dir: Optional[str] = None
    notes: str = ""
    glibc_min: Optional[str] = None


@dataclass(frozen=True)
class ResolvedDfcRuntime:
    """Resolved runtime settings for a given hw_arch/back-end request."""

    profile_id: Optional[str]
    hw_arch: str
    # Effective WSL settings (only meaningful when using WSL)
    wsl_distro: Optional[str]
    wsl_venv_activate: Optional[str]
    # True when the user provided an explicit override for venv activation
    using_user_venv_override: bool


class DfcManager:
    """Loads and resolves DFC profiles."""

    def __init__(self, profiles: Iterable[DfcProfile]):
        self._profiles: Tuple[DfcProfile, ...] = tuple(profiles)
        self._profiles_by_id: Dict[str, DfcProfile] = {p.profile_id: p for p in self._profiles}

    # ------------------------------ loading ------------------------------

    @staticmethod
    def _resources_root() -> Path:
        # .../onnx_splitpoint_tool/hailo/dfc_manager.py -> parents[1] == onnx_splitpoint_tool
        return Path(__file__).resolve().parents[1] / "resources" / "hailo"

    @classmethod
    def from_resources(cls) -> "DfcManager":
        """Load profiles from resources/hailo/profiles.json (fallback to defaults)."""

        cfg_path = cls._resources_root() / "profiles.json"
        if cfg_path.exists():
            try:
                data = json.loads(cfg_path.read_text(encoding="utf-8"))
                profs = _parse_profiles_json(data)
                if profs:
                    return cls(profs)
            except Exception:
                # Fall back to defaults.
                pass
        return cls(default_profiles())

    # ------------------------------ access ------------------------------

    @property
    def profiles(self) -> Tuple[DfcProfile, ...]:
        return self._profiles

    def get_profile(self, profile_id: str) -> Optional[DfcProfile]:
        return self._profiles_by_id.get(str(profile_id or "").strip().lower())

    # ------------------------------ resolve ------------------------------

    def resolve_profile_for_hw_arch(self, hw_arch: str) -> Optional[DfcProfile]:
        """Return the best matching profile for the given `hw_arch`.

        Matching is done by prefix (case-insensitive). The longest prefix wins.
        """

        hw = (hw_arch or "").strip().lower()
        if not hw:
            return None

        best: Optional[DfcProfile] = None
        best_len = -1
        for p in self._profiles:
            for pref in p.hw_arch_prefixes:
                pr = str(pref or "").strip().lower()
                if not pr:
                    continue
                if hw.startswith(pr) and len(pr) > best_len:
                    best = p
                    best_len = len(pr)
        return best

    def resolve_wsl_runtime(
        self,
        *,
        hw_arch: str,
        wsl_distro: Optional[str] = None,
        wsl_venv_activate: Optional[str] = None,
    ) -> ResolvedDfcRuntime:
        """Resolve the effective WSL distro + venv activation path.

        Rules
        -----
        - If `wsl_venv_activate` is a non-empty string that is NOT an auto
          sentinel ("auto"/"managed"), it is treated as a user override.
        - Otherwise, we pick a profile via `hw_arch` and return its defaults.
        - `wsl_distro` overrides the profile's distro when provided.

        The function never probes the environment; it only resolves strings.
        """

        hw = (hw_arch or "").strip()
        distro_override = (wsl_distro or "").strip() or None

        venv_raw = (wsl_venv_activate or "").strip()
        if venv_raw and venv_raw.lower() not in _AUTO_SENTINELS:
            # Explicit override.
            return ResolvedDfcRuntime(
                profile_id=None,
                hw_arch=hw,
                wsl_distro=distro_override,
                wsl_venv_activate=venv_raw,
                using_user_venv_override=True,
            )

        # Managed/profile-based.
        prof = self.resolve_profile_for_hw_arch(hw)
        if prof is None:
            return ResolvedDfcRuntime(
                profile_id=None,
                hw_arch=hw,
                wsl_distro=distro_override,
                wsl_venv_activate=None,
                using_user_venv_override=False,
            )

        # Allow per-profile environment variable override (useful for CI / custom installs).
        env_key = f"ONNX_SPLITPOINT_HAILO_VENV_{prof.profile_id.upper()}"
        env_venv = (os.environ.get(env_key) or "").strip()
        venv_eff = env_venv if env_venv else str(prof.wsl_venv_activate)

        distro_eff = distro_override
        if distro_eff is None:
            distro_eff = (prof.wsl_distro or "").strip() or None

        return ResolvedDfcRuntime(
            profile_id=prof.profile_id,
            hw_arch=hw,
            wsl_distro=distro_eff,
            wsl_venv_activate=venv_eff,
            using_user_venv_override=False,
        )


# ------------------------------ defaults ------------------------------


def default_profiles() -> List[DfcProfile]:
    """Reasonable built-in defaults.

    These defaults assume that provisioning creates separate venvs under:

      ~/.onnx_splitpoint_tool/hailo/venv_<profile>/

    You can override paths via resources/hailo/profiles.json.
    """

    base = "~/.onnx_splitpoint_tool/hailo"
    return [
        DfcProfile(
            profile_id="hailo8",
            hw_arch_prefixes=("hailo8",),
            wsl_venv_activate=f"{base}/venv_hailo8/bin/activate",
            wsl_distro=None,
            wheel_dir="hailo8",
            notes="DFC 3.x (Hailo-8 family)",
            glibc_min="2.34",
        ),
        DfcProfile(
            profile_id="hailo10",
            hw_arch_prefixes=("hailo10",),
            wsl_venv_activate=f"{base}/venv_hailo10/bin/activate",
            wsl_distro=None,
            wheel_dir="hailo10",
            notes="DFC 5.x (Hailo-10 family)",
            glibc_min="2.34",
        ),
    ]


def _parse_profiles_json(data: Dict[str, Any]) -> List[DfcProfile]:
    """Parse the profiles.json structure."""

    out: List[DfcProfile] = []
    if not isinstance(data, dict):
        return out

    profs = data.get("profiles")
    if isinstance(profs, dict):
        # {"hailo8": {...}, "hailo10": {...}}
        items = [(k, v) for k, v in profs.items()]
    elif isinstance(profs, list):
        # [{"profile_id": "hailo8", ...}, ...]
        items = [(None, v) for v in profs]
    else:
        return out

    for key, v in items:
        if not isinstance(v, dict):
            continue
        pid = str(v.get("profile_id") or key or "").strip().lower()
        if not pid:
            continue
        prefixes_raw = v.get("hw_arch_prefixes") or v.get("hw_arch_prefix")
        prefixes: List[str] = []
        if isinstance(prefixes_raw, str):
            prefixes = [prefixes_raw]
        elif isinstance(prefixes_raw, list):
            prefixes = [str(x) for x in prefixes_raw if str(x).strip()]
        if not prefixes:
            # Sensible default: profile_id itself.
            prefixes = [pid]

        wsl_venv_activate = str(v.get("wsl_venv_activate") or v.get("venv_activate") or "").strip()
        if not wsl_venv_activate:
            base = "~/.onnx_splitpoint_tool/hailo"
            wsl_venv_activate = f"{base}/venv_{pid}/bin/activate"

        wsl_distro = str(v.get("wsl_distro") or "").strip() or None
        wheel_dir = str(v.get("wheel_dir") or "").strip() or None
        notes = str(v.get("notes") or "")
        glibc_min = str(v.get("glibc_min") or "").strip() or None

        out.append(
            DfcProfile(
                profile_id=pid,
                hw_arch_prefixes=tuple(prefixes),
                wsl_venv_activate=wsl_venv_activate,
                wsl_distro=wsl_distro,
                wheel_dir=wheel_dir,
                notes=notes,
                glibc_min=glibc_min,
            )
        )

    return out


# ------------------------------ singleton ------------------------------


_DEFAULT_MANAGER: Optional[DfcManager] = None


def get_dfc_manager() -> DfcManager:
    global _DEFAULT_MANAGER
    if _DEFAULT_MANAGER is None:
        _DEFAULT_MANAGER = DfcManager.from_resources()
    return _DEFAULT_MANAGER
