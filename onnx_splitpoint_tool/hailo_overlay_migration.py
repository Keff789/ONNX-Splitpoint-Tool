"""Scoped v2.81 repair of absent H8 GPU overlay selections.

Preserve explicit none/other selections and every non-H8 setting. The reviewed
component binding comes from the completed combined smoke; it is not discovery
or permission to install packages or alter a cache key.
"""
from __future__ import annotations

from contextlib import ExitStack
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import yaml


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def reviewed_selection(tool: Path, evidence: Path | None = None) -> dict[str, Any]:
    binding = json.loads((tool / "onnx_splitpoint_tool/resources/hailo/reviewed_overlay_v281.json").read_text())
    if evidence is not None and digest(evidence) != binding["source_evidence_sha256"]:
        raise ValueError("hailo8_reviewed_evidence_changed")
    selected = binding["selection"]
    manifest = Path(selected["path"])
    if manifest.is_symlink() or digest(manifest) != selected["manifest_sha256"]:
        raise ValueError("hailo8_reviewed_manifest_changed")
    from .hailo_dependency_plan import validated_overlay_components
    manifest_data = json.loads(manifest.read_text())
    python = Path(manifest_data["python_executable"])
    if str(python.parent.parent) != selected["selected_venv"]:
        raise ValueError("hailo8_reviewed_venv_changed")
    components = validated_overlay_components(family="hailo8", selected_python=python, manifest_path=manifest)
    if not components:
        raise ValueError("hailo8_reviewed_components_missing")
    for name, expected in selected["component_sha256"].items():
        if digest(Path(components[name])) != expected:
            raise ValueError("hailo8_reviewed_component_changed:" + name)
    return {"dependency_manifest": str(manifest), "manifest_sha256": selected["manifest_sha256"],
            "selected_python": str(python), "source_evidence_sha256": binding["source_evidence_sha256"],
            "validation": "reviewed_binding_and_current_venv_metadata", "hardware_execution": "NOT_RUN"}


def migrate_payload(payload: dict, manifest: str, *, registry: bool = False) -> tuple[dict, list[str]]:
    from .run_modes import _json_hash
    result = copy.deepcopy(payload)
    changes: list[str] = []

    def change(hailo, prefix):
        if not isinstance(hailo, dict):
            return False
        compute = hailo.get("compute_by_family")
        entry = compute.get("hailo8") if isinstance(compute, dict) else None
        # Only an explicit existing GPU choice with an absent manifest is in
        # scope. CPU, legacy environment choices and explicit empty remain so.
        if isinstance(entry, str) and entry.lower().strip() == "gpu":
            entry = {"device": "gpu"}
            compute["hailo8"] = entry
        if not isinstance(entry, dict) or entry.get("device") != "gpu" or "dependency_manifest" in entry:
            return False
        entry["dependency_manifest"] = manifest
        changes.append(prefix + ".compute_by_family.hailo8.dependency_manifest")
        return True

    if registry:
        for name, mode in (result.get("modes") or {}).items():
            change((mode.get("build") or {}).get("hailo"), "modes." + name + ".build.hailo")
    else:
        old_compute = copy.deepcopy((result.get("hailo_build") or {}).get("compute_by_family"))
        hb_changed = change(result.get("hailo_build"), "hailo_build")
        preset = result.get("execution_preset") or {}
        provenance = preset.get("build_provenance") or {}
        if hb_changed and provenance.get("hailo_compute_by_family") == old_compute:
            provenance["hailo_compute_by_family"] = copy.deepcopy(result["hailo_build"]["compute_by_family"])
            changes.append("execution_preset.build_provenance.hailo_compute_by_family")
        snapshot = preset.get("snapshot") or {}
        if change((snapshot.get("build") or {}).get("hailo"), "execution_preset.snapshot.build.hailo"):
            preset["snapshot_sha256"] = _json_hash(snapshot)
            changes.append("execution_preset.snapshot_sha256")
    return result, changes


def _atomic_bytes(path: Path, content: bytes, mode: int = 0o600) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def migrate_configs(tool: Path, output_dir: Path, *, evidence: Path | None = None,
                    registry_path: Path | None = None) -> dict[str, Any]:
    import fcntl
    from .run_modes import default_run_modes_path, validate_run_modes_config
    from .workflow.run_control import platform_workflow_interlock_path
    output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"status": "failed", "changed_files": [], "skipped_files": [],
                              "hardware_execution": "NOT_RUN", "model_build": "NOT_RUN"}
    with ExitStack() as stack:
        lock_path = platform_workflow_interlock_path()
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = stack.enter_context(lock_path.open("a+b"))
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Read only the packaged selection hint while establishing scope.
        # An all-CPU/explicit-none/other-venv configuration needs no overlay,
        # so a deleted historical overlay must not turn a no-op into failure.
        binding = json.loads((tool / "onnx_splitpoint_tool/resources/hailo/reviewed_overlay_v281.json").read_text())
        selected = {"dependency_manifest": binding["selection"]["path"],
                    "selected_python": str(Path(binding["selection"]["selected_venv"]) / "bin/python")}
        registry_path = registry_path or default_run_modes_path()
        paths = sorted(set((tool / "profiles").glob("*.yaml")) | set((tool / "profiles").glob("*.yml")))
        if registry_path.is_file():
            paths.insert(0, registry_path)
            registry_lock = stack.enter_context(registry_path.with_name("." + registry_path.name + ".lock").open("a+b"))
            fcntl.flock(registry_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pending = []
        for path in paths:
            if path.is_symlink():
                raise ValueError("hailo8_config_symlink_rejected:" + str(path))
            before = path.read_bytes()
            original = yaml.safe_load(before)
            if not isinstance(original, dict):
                raise ValueError("hailo8_config_not_mapping:" + str(path))
            after, fields = migrate_payload(original, selected["dependency_manifest"], registry=path == registry_path)
            if not fields:
                report["skipped_files"].append({"path": str(path), "reason": "no_absent_hailo8_gpu_selection"})
                continue
            if path == registry_path:
                validate_run_modes_config(after)  # validate without adding/replacing unrelated defaults
            serialized = yaml.safe_dump(after, sort_keys=False, allow_unicode=True).encode()
            pending.append((path, before, serialized, fields))
        if not pending:
            report.update(status="not_required", dependency_manifest=None,
                          reason="no_absent_hailo8_gpu_selection")
            _atomic_bytes(output_dir / "hailo8_overlay_migration.json", (json.dumps(report, indent=2, sort_keys=True) + "\n").encode())
            return report
        # The productive compiler selects its venv through DfcManager's
        # persistent profiles.json and family environment override. The
        # evaluation profile has no venv_activate setting in its schema.
        from .hailo_backend import _resolve_managed_venv_python
        profile_id, actual_python, actual_activate = _resolve_managed_venv_python(
            hw_arch="hailo8", venv_activate="auto")
        actual_python = Path(actual_python).expanduser().absolute()
        actual_root = actual_python.parent.parent.resolve()
        reviewed_root = Path(binding["selection"]["selected_venv"]).expanduser().resolve()
        report["dfc_runtime_selection"] = {"profile_id": profile_id,
            "selected_python": str(actual_python), "selected_activate": str(actual_activate),
            "selected_venv": str(actual_root), "reviewed_venv": str(reviewed_root)}
        if (actual_root != reviewed_root
                or actual_python != Path(selected["selected_python"]).expanduser().absolute()
                or Path(actual_activate).expanduser().absolute().parent.parent.resolve() != reviewed_root):
            report.update(status="not_required", dependency_manifest=None,
                          reason="selected_hailo8_venv_differs_from_reviewed")
            _atomic_bytes(output_dir / "hailo8_overlay_migration.json", (json.dumps(report, indent=2, sort_keys=True) + "\n").encode())
            return report
        # Only an actual, authorized change validates the reviewed local
        # components. No global defaults or alternate overlays are selected.
        report.update(reviewed_selection(tool, evidence))
        backup_root = output_dir / "private_backups"
        backup_root.mkdir(mode=0o700, exist_ok=True)
        os.chmod(backup_root, 0o700)
        for index, (path, before, after, fields) in enumerate(pending):
            backup = backup_root / f"{index:03d}_{path.name}"
            if backup.exists():
                raise ValueError("hailo8_backup_already_exists:" + str(backup))
            _atomic_bytes(backup, before)
        # Check all originals again before the first write. The workflow and
        # registry locks remain held through verification and any rollback.
        if any(path.read_bytes() != before for path, before, _, _ in pending):
            raise ValueError("hailo8_configuration_changed_before_write")
        written = []
        try:
            for path, before, after, fields in pending:
                mode = path.stat().st_mode & 0o777
                _atomic_bytes(path, after, mode)
                written.append((path, before, after, mode))
                if path.read_bytes() != after:
                    raise ValueError("hailo8_configuration_verification_failed:" + str(path))
                report["changed_files"].append({"path": str(path), "fields": fields,
                    "before_sha256": hashlib.sha256(before).hexdigest(), "after_sha256": hashlib.sha256(after).hexdigest()})
            report["status"] = "pass"
            _atomic_bytes(output_dir / "hailo8_overlay_migration.json", (json.dumps(report, indent=2, sort_keys=True) + "\n").encode())
        except BaseException:
            for path, before, after, mode in reversed(written):
                if path.read_bytes() == after:
                    _atomic_bytes(path, before, mode)
            raise
    return report
