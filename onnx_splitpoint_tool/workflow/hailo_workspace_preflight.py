"""Early, job-local DFC resource snapshot from the prepared cold-build contract.

The builder repeats the same check immediately before SDK dispatch. This is
neither a space reservation nor artifact/recipe identity or negative evidence.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

from .artifacts import now_iso


def preflight_cold_hailo_workspaces(report: Mapping[str, Any],
                                    profile: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema": "onnx-splitpoint/hailo-workspace-preflight", "schema_version": 1,
        "created_at": now_iso(), "status": "not_required", "jobs": [],
        "selection_phase": "final_selected_cases_after_cache_and_negative_lookup",
        "scope": "job_local", "reservation": False, "repeat_at_dispatch": True,
        "compiler_dispatch_count": 0, "model_build": "NOT_RUN",
        "concurrency": "Existing scheduler; each actual start repeats admission. Job estimates are not summed.",
    }
    from ..cache_verify_policy import cache_verify_only_enabled
    if report.get("runtime_dispatch_allowed") is False or cache_verify_only_enabled(profile):
        result["reason"] = "compiler_dispatch_already_prohibited"
        return result
    rows = [row for row in report.get("cold_build_rows", [])
            if isinstance(row, Mapping) and row.get("status") == "MISS"
            and row.get("compiler_dispatch_allowed") is True
            and not row.get("runtime_artifact_available")
            and row.get("role") in {"hailo8_hef", "hailo10_hef"}
            and not (row.get("evidence") or {}).get("compiler_cache_only")]
    if not rows:
        return result
    from ..hailo_backend import hailo_dfc_workspace_preflight
    for row in rows:
        evidence = row.get("evidence") or {}
        contract = evidence.get("workspace_contract") or {}
        contract = contract if isinstance(contract, Mapping) else {}
        job = {key: row.get(key) for key in (
            "model_id", "boundary", "backend", "artifact_stage", "role", "item_id")}
        job.update(created_at=now_iso(), status="unknown", scope="job_local",
                   reason="local_dfc_workspace_unresolved",
                   message="Nicht gestartet: DFC-Arbeitsbereich noch nicht prüfbar",
                   deferred_request=evidence.get("deferred_request", ""),
                   workspace_contract=dict(contract), compiler_dispatch_count=0)
        target = str(contract.get("out_dir") or "")
        artifact = str(row.get("artifact_path") or "")
        if os.name == "nt":
            job.update(reason="linux_workspace_requires_wsl", status="deferred_to_wsl_worker")
        elif not target or not Path(target).is_absolute():
            job["detail"] = "Exact prepared output directory unavailable"
        elif not artifact or Path(artifact).resolve().parent != Path(target).resolve():
            job["detail"] = "Prepared workspace and selected artifact output directory differ"
        elif contract.get("source") != "prepared_hailo_build_contract":
            job["detail"] = "Effective prepared calibration contract unavailable"
        else:
            probe = hailo_dfc_workspace_preflight(
                target, calibration_count=contract.get("effective_calibration_count"),
                input_shapes=contract.get("calibration_identity_shapes") or [])
            job.update(workspace=probe.get("workspace"), calculation=probe.get("calculation"),
                       problems=probe.get("problems"), status=probe["status"])
            if probe["status"] == "passed":
                job.update(reason="", message="Arbeitsraum ausreichend; erneute Prüfung vor Compilerstart")
            elif probe["status"] == "failed":
                job.update(reason="local_dfc_workspace_insufficient",
                           message="Nicht gestartet: Arbeitsbereich zu klein oder nicht beschreibbar")
            elif probe["status"] == "disabled":
                job.update(reason="workspace_preflight_explicitly_disabled",
                           message="Arbeitsraumprüfung ausdrücklich deaktiviert; kein Platznachweis")
        result["jobs"].append(job)
    statuses = {job["status"] for job in result["jobs"]}
    result["status"] = ("partial" if "failed" in statuses else "unknown" if "unknown" in statuses
                        else "deferred_to_wsl_worker" if "deferred_to_wsl_worker" in statuses
                        else "disabled" if "disabled" in statuses else "passed")
    result["blocked_job_count"] = sum(job["status"] == "failed" for job in result["jobs"])
    result["unknown_job_count"] = sum(job["status"] == "unknown" for job in result["jobs"])
    return result
