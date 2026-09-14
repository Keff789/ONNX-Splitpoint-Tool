"""Check effective Hailo compiler contexts only after final artifact lookup.

This is execution provenance, never a recipe/cache identity. No SDK import,
model compilation, cache mutation or discovery of unselected overlays occurs.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping


def preflight_cold_hailo_contexts(report: Mapping[str, Any], profile: Mapping[str, Any],
                                *, work_dir: Path) -> dict[str, Any]:
    rows = [dict(row) for row in report.get("cold_build_rows", [])
            if isinstance(row, Mapping) and row.get("compiler_dispatch_allowed") is True
            and row.get("status") == "MISS"
            and not row.get("runtime_artifact_available")
            and not (row.get("evidence") or {}).get("compiler_cache_only")
            and row.get("role") in {"hailo8_hef", "hailo10_hef"}]
    result = {"schema": "onnx-splitpoint/hailo-cold-compiler-preflight", "schema_version": 1,
              "status": "not_required", "compiler_dispatch_allowed": True,
              "selection_phase": "final_selected_cases_after_cache_and_negative_lookup",
              "cold_rows": rows, "contexts": [], "model_build": "NOT_RUN",
              "sdk_import": "NOT_RUN", "artifact_identity_changed": False}
    if not rows:
        return result
    from ..cache_verify_policy import cache_verify_only_enabled
    if report.get("runtime_dispatch_allowed") is False or cache_verify_only_enabled(profile):
        result.update(status="not_required", reason="compiler_dispatch_already_prohibited")
        return result
    # The Windows compiler is resolved inside its existing WSL worker; host
    # pathlib must not reject the selected Linux venv before dispatch.
    if os.name == "nt":
        result.update(status="deferred_to_wsl_worker", reason="linux_context_requires_wsl")
        return result
    from ..hailo_backend import _resolve_managed_venv_python, _managed_venv_child_env
    from ..hailo_compiler_context import resolve_hailo_compiler_context, compiler_child_environment
    hailo = dict(profile.get("hailo_build") or {})
    probe_cache: dict = {}
    families = sorted({"hailo8" if row["role"] == "hailo8_hef" else "hailo10h" for row in rows})
    for family in families:
        item = {"family": family, "status": "failed",
                "cold_rows": [row for row in rows if row["role"] == (
                    "hailo8_hef" if family == "hailo8" else "hailo10_hef")]}
        try:
            _, python, _ = _resolve_managed_venv_python(
                hw_arch=family, venv_activate="auto")
            env = _managed_venv_child_env(python)
            context = resolve_hailo_compiler_context(
                python, family, compute_by_family=hailo.get("compute_by_family"),
                parent_env=env, probe_cache=probe_cache, work_dir=work_dir)
            # Validate the actual child library projection and private CUDA
            # view, including context-manager cleanup, without loading DFC.
            with compiler_child_environment(context, parent_env=env, work_dir=work_dir) as (_, effective):
                item["context"] = dict(effective)
            item["status"] = "pass"
        except Exception as exc:
            item.update(reason=getattr(exc, "reason", "hailo_compiler_context_unavailable"),
                        error=f"{type(exc).__name__}: {exc}",
                        details=getattr(exc, "details", {}))
        result["contexts"].append(item)
    passed = all(item["status"] == "pass" for item in result["contexts"])
    result.update(status="pass" if passed else "failed", compiler_dispatch_allowed=passed)
    return result
