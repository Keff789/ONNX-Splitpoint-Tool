from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml

from .deepx.env_status import inspect_deepx_environment, default_dx_all_suite_root, default_compiler_venv, default_runtime_venv
from .hailo.dfc_env_status import inspect_profiles as inspect_hailo_profiles

CONFIG_PATH = Path(os.path.expanduser("~/.onnx_splitpoint_tool/build_environments.yaml"))


def default_config() -> dict[str, Any]:
    dx_root = default_dx_all_suite_root()
    dx_venv = default_compiler_venv(dx_root)
    dx_run_venv = default_runtime_venv(dx_root)
    return {
        "build_environments": [
            {
                "id": "hailo8_dfc_managed",
                "kind": "hailo8_dfc",
                "host": "local",
                "shell": "bash",
                "workdir": "~/.onnx_splitpoint_tool/hailo/builds/hailo8",
                "venv_activate": "source ~/.onnx_splitpoint_tool/hailo/venv_hailo8/bin/activate",
                "cache_dir": "~/Models/BackendArtifacts/hailo",
            },
            {
                "id": "hailo10_dfc_managed",
                "kind": "hailo10_dfc",
                "host": "local",
                "shell": "bash",
                "workdir": "~/.onnx_splitpoint_tool/hailo/builds/hailo10",
                "venv_activate": "source ~/.onnx_splitpoint_tool/hailo/venv_hailo10/bin/activate",
                "cache_dir": "~/Models/BackendArtifacts/hailo",
            },
            {
                "id": "deepx_dxcom_x86",
                "kind": "deepx_dxcom",
                "host": "local",
                "shell": "bash",
                "dx_all_suite_root": str(dx_root),
                "venv_activate": f"source {dx_venv}/bin/activate",
                "compiler_venv": str(dx_venv),
                "runtime_venv": str(dx_run_venv),
                "runtime_venv_activate": f"source {dx_run_venv}/bin/activate",
                "cache_dir": "~/Models/BackendArtifacts/deepx",
            },
        ]
    }


def ensure_config(path: Path = CONFIG_PATH) -> Path:
    path = Path(path).expanduser()
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(default_config(), sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    p = ensure_config(path)
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception:
        data = {}
    if not isinstance(data, dict):
        data = {}
    if not isinstance(data.get("build_environments"), list):
        data["build_environments"] = default_config()["build_environments"]
    return data


def _env_by_kind(config: Mapping[str, Any], needle: str) -> Optional[dict[str, Any]]:
    for e in config.get("build_environments") or []:
        if not isinstance(e, dict):
            continue
        if needle in str(e.get("kind") or "").lower():
            return dict(e)
    return None


def status_all(*, probe_import: bool = False) -> dict[str, Any]:
    cfg = load_config()
    envs = list(cfg.get("build_environments") or [])
    status_records: list[dict[str, Any]] = []

    hailo_statuses = []
    try:
        hailo_statuses = inspect_hailo_profiles(probe_import=probe_import)
    except Exception as exc:
        hailo_statuses = [{"profile_id": "hailo", "ready": False, "status": "error", "reason": f"{type(exc).__name__}: {exc}"}]
    for h in hailo_statuses:
        pid = str(h.get("profile_id") or h.get("id") or "hailo")
        status_records.append({
            "id": f"{pid}_dfc_managed",
            "kind": f"{pid}_dfc",
            "ok": bool(h.get("ready")),
            "ready": bool(h.get("ready")),
            "status": h.get("status") or ("ok" if h.get("ready") else "not_ready"),
            "details": h,
        })

    dx_cfg = _env_by_kind(cfg, "deepx") or {}
    try:
        st_dx = inspect_deepx_environment(probe_import=probe_import, config=dx_cfg)
        status_records.append({
            "id": dx_cfg.get("id") or "deepx_dxcom_x86",
            "kind": dx_cfg.get("kind") or "deepx_dxcom",
            # ok means fully usable for ONNX -> DXNN builds and DX-RT runtime.
            # Runtime-only readiness is useful, but must not be displayed as
            # "provisioned" because compilation will still fail.
            "ok": bool(st_dx.get("ready")),
            "ready": bool(st_dx.get("ready")),
            "runtime_ready": bool(st_dx.get("runtime_ready")),
            "compiler_ready": bool(st_dx.get("compiler_ready")),
            "status": st_dx.get("status"),
            "details": st_dx,
        })
    except Exception as exc:
        status_records.append({"id": "deepx_dxcom_x86", "kind": "deepx_dxcom", "ok": False, "ready": False, "status": "error", "details": {"reason": f"{type(exc).__name__}: {exc}"}})

    return {"schema": "onnx-splitpoint/build-environments-status", "config_path": str(ensure_config()), "environments": status_records, "ok": all(bool(x.get("ok")) for x in status_records if x.get("kind"))}


def format_status(payload: Mapping[str, Any]) -> str:
    lines = ["Accelerator build/runtime environments", "", f"Config: {payload.get('config_path')}", ""]
    for e in payload.get("environments") or []:
        lines.append(f"[{e.get('id')}] {e.get('kind')}")
        lines.append(f"  status: {e.get('status')}  ok={bool(e.get('ok'))} ready={bool(e.get('ready'))}")
        d = e.get("details") or {}
        if isinstance(d, Mapping):
            for key in ("venv_path", "wheel_dir", "dx_all_suite_root", "compiler_venv", "compiler_cli", "runtime_venv", "cache_dir", "reason"):
                if d.get(key):
                    lines.append(f"  {key}: {d.get(key)}")
            hints = d.get("hints") or []
            for h in hints:
                lines.append(f"  hint: {h}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    import argparse, json
    ap = argparse.ArgumentParser(description="Inspect accelerator build environments")
    ap.add_argument("--status", action="store_true", help="Print status (default)")
    ap.add_argument("--probe-import", action="store_true", help="Run import probes where supported")
    ap.add_argument("--json", action="store_true", help="Print JSON")
    ap.add_argument("--ensure-config", action="store_true", help="Create user config if missing and print its path")
    ns = ap.parse_args(argv)
    if ns.ensure_config:
        print(ensure_config())
        return 0
    payload = status_all(probe_import=bool(ns.probe_import))
    if ns.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(format_status(payload))
    return 0 if bool(payload.get("ok")) else 3


if __name__ == "__main__":
    raise SystemExit(main())
