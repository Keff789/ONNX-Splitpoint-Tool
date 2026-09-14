from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

from .artifacts import now_iso, read_json, relpath, write_json


def initial_manifest(
    *,
    run_id: str,
    profile_id: str,
    profile_hash: str,
    tool_version: str,
    options: Mapping[str, Any],
    profile_path: str,
    targets: list[str],
    run_profiles: list[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "workflow_schema_version": 1,
        "run_id": run_id,
        "profile_id": profile_id,
        "profile_hash": profile_hash,
        "profile_path": profile_path,
        "tool_version": tool_version,
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "status": "running",
        "options": dict(options or {}),
        "targets": list(targets or []),
        "run_profiles": [dict(x or {}) for x in list(run_profiles or [])],
        "models": [],
        "stages": [],
        "hardware": {
            "hailo_arch": "hailo8" if "hailo8" in set(targets or []) else None,
            "cuda_available": None,
            "hailo_dfc_version": None,
        },
        "output_contracts": [],
        "reports": [],
    }


def initial_artifact_index(*, run_id: str) -> dict[str, Any]:
    return {"schema_version": 1, "run_id": run_id, "created_at": now_iso(), "updated_at": now_iso(), "artifacts": []}


class ManifestStore:
    def __init__(self, run_root: Path, run_id: str) -> None:
        self.run_root = Path(run_root)
        self.run_id = str(run_id)
        self.manifest_path = self.run_root / "run_manifest.json"
        self.artifact_index_path = self.run_root / "artifact_index.json"
        self._manifest: dict[str, Any] = {}
        self._artifact_index: dict[str, Any] = {}

    @property
    def manifest(self) -> dict[str, Any]:
        return self._manifest

    @property
    def artifact_index(self) -> dict[str, Any]:
        return self._artifact_index

    def load_or_create(self, manifest: Mapping[str, Any], artifact_index: Optional[Mapping[str, Any]] = None) -> None:
        self._manifest = dict(read_json(self.manifest_path) or {}) if self.manifest_path.is_file() else dict(manifest or {})
        self._artifact_index = (
            dict(read_json(self.artifact_index_path) or {})
            if self.artifact_index_path.is_file()
            else dict(artifact_index or initial_artifact_index(run_id=self.run_id))
        )
        self.save()

    def save(self) -> None:
        self._manifest["updated_at"] = now_iso()
        self._artifact_index["updated_at"] = now_iso()
        write_json(self.manifest_path, self._manifest)
        write_json(self.artifact_index_path, self._artifact_index)

    def add_stage(self, stage_result: Mapping[str, Any], *, stage_result_path: Path) -> None:
        e = dict(stage_result or {})
        e["stage_result_path"] = relpath(stage_result_path, self.run_root)
        stages = [dict(x or {}) for x in list(self._manifest.get("stages") or [])]
        stages = [x for x in stages if not (x.get("stage") == e.get("stage") and x.get("model_id") == e.get("model_id"))]
        stages.append(e)
        self._manifest["stages"] = stages
        self.save()

    def add_model(self, model_manifest: Mapping[str, Any]) -> None:
        mid = str((model_manifest or {}).get("model_id") or "").strip()
        models = [dict(x or {}) for x in list(self._manifest.get("models") or [])]
        if mid:
            models = [x for x in models if str(x.get("model_id") or "") != mid]
        models.append(dict(model_manifest or {}))
        self._manifest["models"] = models
        self.save()

    def add_output_contracts(self, contracts: list[Mapping[str, Any]]) -> None:
        existing = [dict(x or {}) for x in list(self._manifest.get("output_contracts") or [])]
        keys = {(x.get("model_id"), x.get("backend"), x.get("variant")) for x in contracts}
        existing = [x for x in existing if (x.get("model_id"), x.get("backend"), x.get("variant")) not in keys]
        existing.extend([dict(x or {}) for x in contracts])
        self._manifest["output_contracts"] = existing
        self.save()

    def add_artifact(self, record: Mapping[str, Any]) -> None:
        rec = dict(record or {})
        if not rec.get("path"):
            return
        arts = [dict(x or {}) for x in list(self._artifact_index.get("artifacts") or [])]
        arts = [x for x in arts if str(x.get("path") or "") != str(rec.get("path") or "")]
        arts.append(rec)
        self._artifact_index["artifacts"] = arts
        self.save()

    def mark_status(self, status: str, *, reports: Optional[list[str]] = None) -> None:
        self._manifest["status"] = str(status)
        if reports is not None:
            self._manifest["reports"] = list(reports)
        self.save()
