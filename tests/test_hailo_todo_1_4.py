from __future__ import annotations

from pathlib import Path

import onnx_splitpoint_tool.runners.backends.hailo_backend as hb_mod
from onnx_splitpoint_tool.runners.backends.hailo_backend import HailoBackend


def test_hailo_availability_checks_are_called_and_sanitized(monkeypatch):
    calls: list[str] = []

    def fake_check_wsl_reachable():
        calls.append("wsl")
        return False, "WSL unreachable\x1b[31m\r"

    def fake_check_dfc_installed():
        calls.append("dfc")
        return False, "DFC missing\r\x1b[0m"

    def fake_check_device_present(dev_path: str = "/dev/hailo0"):
        calls.append("dev")
        return False, f"Device missing: {dev_path}\r\x1b[0m"

    # Patch the imported symbols in hailo_backend.py (it used `from .hailo_utils import ...`).
    monkeypatch.setattr(hb_mod, "check_wsl_reachable", fake_check_wsl_reachable)
    monkeypatch.setattr(hb_mod, "check_dfc_installed", fake_check_dfc_installed)
    monkeypatch.setattr(hb_mod, "check_device_present", fake_check_device_present)

    # Pretend runtime is present so we only test the new checks here.
    def fake_find_spec(name: str):
        return object()

    monkeypatch.setattr(hb_mod.importlib.util, "find_spec", fake_find_spec)

    backend = HailoBackend(require_device=True)
    ok = backend.is_available(needs_compiler=True)
    assert ok is False

    reason = backend.unavailable_reason
    assert "WSL unreachable" in reason
    assert "DFC missing" in reason
    assert "Device missing" in reason

    # Clean, user-facing formatting
    assert "\x1b" not in reason
    assert "\r" not in reason

    # Sanity: all checks executed
    assert calls == ["wsl", "dfc", "dev"]


def test_no_legacy_hailo_scripts_or_dfc_manager_referenced():
    repo_root = Path(__file__).resolve().parents[1]
    scan_root = repo_root / "onnx_splitpoint_tool"

    # These scripts still exist as deprecated wrappers, but must not be referenced.
    legacy_needles = [
        "wsl_hailo_build_hef",
        "wsl_hailo_check",
        "dfc_manager",
    ]

    allowlist = {
        scan_root / "wsl_hailo_build_hef.py",
        scan_root / "wsl_hailo_check.py",
    }

    offenders: list[str] = []
    for py in scan_root.rglob("*.py"):
        if py in allowlist:
            continue
        txt = py.read_text(encoding="utf-8", errors="ignore")
        for needle in legacy_needles:
            if needle in txt:
                offenders.append(f"{py}: contains '{needle}'")

    assert offenders == []
