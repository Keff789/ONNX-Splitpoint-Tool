from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


_AUTO_SENTINELS = {"", "auto", "managed", "default"}


@dataclass(frozen=True)
class DfcProfile:
    profile_id: str
    hw_arch_prefixes: Tuple[str, ...]
    wsl_venv_activate: str
    wsl_distro: Optional[str] = None
    wheel_dir: Optional[str] = None
    notes: str = ""
    glibc_min: Optional[str] = None


@dataclass(frozen=True)
class ResolvedDfcRuntime:
    profile_id: Optional[str]
    hw_arch: str
    wsl_distro: Optional[str]
    wsl_venv_activate: Optional[str]
    using_user_venv_override: bool


class DfcManager:
    def __init__(self, profiles: Iterable[DfcProfile]):
        self._profiles: Tuple[DfcProfile, ...] = tuple(profiles)
        self._profiles_by_id: Dict[str, DfcProfile] = {p.profile_id: p for p in self._profiles}

    @staticmethod
    def _resources_root() -> Path:
        return Path(__file__).resolve().parents[2] / "resources" / "hailo"

    @classmethod
    def from_resources(cls) -> "DfcManager":
        cfg_path = cls._resources_root() / "profiles.json"
        if cfg_path.exists():
            try:
                data = json.loads(cfg_path.read_text(encoding="utf-8"))
                profiles = _parse_profiles_json(data)
                if profiles:
                    return cls(profiles)
            except Exception:
                pass
        return cls(default_profiles())

    @property
    def profiles(self) -> Tuple[DfcProfile, ...]:
        return self._profiles

    def get_profile(self, profile_id: str) -> Optional[DfcProfile]:
        return self._profiles_by_id.get(str(profile_id or "").strip().lower())

    def resolve_profile_for_hw_arch(self, hw_arch: str) -> Optional[DfcProfile]:
        hw = (hw_arch or "").strip().lower()
        if not hw:
            return None
        best: Optional[DfcProfile] = None
        best_len = -1
        for profile in self._profiles:
            for prefix in profile.hw_arch_prefixes:
                pref = str(prefix or "").strip().lower()
                if pref and hw.startswith(pref) and len(pref) > best_len:
                    best = profile
                    best_len = len(pref)
        return best

    def resolve_wsl_runtime(
        self,
        *,
        hw_arch: str,
        wsl_distro: Optional[str] = None,
        wsl_venv_activate: Optional[str] = None,
    ) -> ResolvedDfcRuntime:
        hw = (hw_arch or "").strip()
        distro_override = (wsl_distro or "").strip() or None
        venv_raw = (wsl_venv_activate or "").strip()
        if venv_raw and venv_raw.lower() not in _AUTO_SENTINELS:
            return ResolvedDfcRuntime(
                profile_id=None,
                hw_arch=hw,
                wsl_distro=distro_override,
                wsl_venv_activate=venv_raw,
                using_user_venv_override=True,
            )

        profile = self.resolve_profile_for_hw_arch(hw)
        if profile is None:
            return ResolvedDfcRuntime(
                profile_id=None,
                hw_arch=hw,
                wsl_distro=distro_override,
                wsl_venv_activate=None,
                using_user_venv_override=False,
            )

        env_key = f"ONNX_SPLITPOINT_HAILO_VENV_{profile.profile_id.upper()}"
        env_venv = (os.environ.get(env_key) or "").strip()
        venv_eff = env_venv if env_venv else str(profile.wsl_venv_activate)
        distro_eff = distro_override if distro_override is not None else ((profile.wsl_distro or "").strip() or None)

        return ResolvedDfcRuntime(
            profile_id=profile.profile_id,
            hw_arch=hw,
            wsl_distro=distro_eff,
            wsl_venv_activate=venv_eff,
            using_user_venv_override=False,
        )


def default_profiles() -> List[DfcProfile]:
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
    out: List[DfcProfile] = []
    if not isinstance(data, dict):
        return out

    profiles = data.get("profiles")
    if isinstance(profiles, dict):
        items = [(k, v) for k, v in profiles.items()]
    elif isinstance(profiles, list):
        items = [(None, v) for v in profiles]
    else:
        return out

    for key, value in items:
        if not isinstance(value, dict):
            continue
        pid = str(value.get("profile_id") or key or "").strip().lower()
        if not pid:
            continue
        prefixes_raw = value.get("hw_arch_prefixes") or value.get("hw_arch_prefix")
        prefixes: List[str] = []
        if isinstance(prefixes_raw, str):
            prefixes = [prefixes_raw]
        elif isinstance(prefixes_raw, list):
            prefixes = [str(x) for x in prefixes_raw if str(x).strip()]
        if not prefixes:
            prefixes = [pid]

        wsl_venv_activate = str(value.get("wsl_venv_activate") or value.get("venv_activate") or "").strip()
        if not wsl_venv_activate:
            base = "~/.onnx_splitpoint_tool/hailo"
            wsl_venv_activate = f"{base}/venv_{pid}/bin/activate"

        out.append(
            DfcProfile(
                profile_id=pid,
                hw_arch_prefixes=tuple(prefixes),
                wsl_venv_activate=wsl_venv_activate,
                wsl_distro=str(value.get("wsl_distro") or "").strip() or None,
                wheel_dir=str(value.get("wheel_dir") or "").strip() or None,
                notes=str(value.get("notes") or ""),
                glibc_min=str(value.get("glibc_min") or "").strip() or None,
            )
        )
    return out


_DEFAULT_MANAGER: Optional[DfcManager] = None


def get_dfc_manager() -> DfcManager:
    global _DEFAULT_MANAGER
    if _DEFAULT_MANAGER is None:
        _DEFAULT_MANAGER = DfcManager.from_resources()
    return _DEFAULT_MANAGER

