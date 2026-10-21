"""Exact existing Native points in the ordinary Evaluation Workflow.

This selector reuses the existing finite execution, numerical, collector and
import adapters. It has no worker, queue, transport or retry implementation.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import re
from typing import Mapping

import yaml

from . import hardware_acceptance as acceptance
from . import hardware_acceptance_native as native

SCHEMA = "onnx-splitpoint/native-refresh/v1"
MATRIX_SCHEMA = "onnx-splitpoint/native-refresh-selection/v1"


def enabled(profile: Mapping) -> bool:
    value = profile.get("native_refresh") or {}
    if not isinstance(value, Mapping) or type(value.get("enabled", False)) is not bool:
        raise ValueError("native_refresh.enabled must be boolean")
    return value.get("enabled") is True


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _bound(value):
    if not isinstance(value, Mapping) or not re.fullmatch(r"[a-f0-9]{64}", str(value.get("sha256") or "")):
        raise ValueError("Native refresh evidence requires an existing path and SHA256")
    binding = acceptance.file_binding(value)
    path = Path(binding["path"])
    if any(item.is_symlink() for item in (path, *path.parents)):
        raise ValueError("Native refresh evidence must use canonical regular paths")
    return path


def _uid(cell):
    return "|".join(("native", str(cell["model_id"]), "DeepX", str(cell.get("backend") or cell.get("native_backend")), str(cell["case_id"])))


def validate_request(profile: Mapping, options=None) -> dict:
    """Validate a model's exact saved UID group without widening acceptance."""
    if not enabled(profile) or acceptance.enabled(profile):
        raise ValueError("Native refresh requires its own sole enabled selection")
    config = copy.deepcopy(dict(profile["native_refresh"]))
    if config.get("schema") != SCHEMA or config.get("phase", "complete") != "complete":
        raise ValueError("unsupported Native refresh schema or phase")
    snapshot_path = _bound(config.get("source_snapshot"))
    snapshot = _read(snapshot_path)
    if not isinstance(snapshot, list):
        raise ValueError("Native refresh requires the existing normal row selection")
    selected = [row for row in snapshot if row.get("runner") == "native" and row.get("setup_id") == "DeepX"
                and row.get("backend") in {"deepx_to_trt", "native_full_deepx"}]
    expected = {row["uid"] for row in selected}
    if len(selected) != 75 or len(expected) != 75 or sum(row.get("kind") == "split" for row in selected) != 68:
        raise ValueError("Native refresh source selection must contain the exact 68 splits and seven Fulls")
    matrix_path = _bound(config.get("matrix_evidence"))
    matrix = _read(matrix_path)
    if (matrix.get("schema") != MATRIX_SCHEMA or not matrix.get("recorded_before_hardware_start")
            or matrix.get("source_snapshot") != acceptance.file_binding(config["source_snapshot"])
            or len(matrix.get("cells") or []) != 75
            or {_uid(cell) for cell in matrix["cells"]} != expected):
        raise ValueError("Native refresh matrix differs from the exact existing Native selection")
    cells = config.get("cells")
    if not isinstance(cells, list) or not cells or len(cells) > 21:
        raise ValueError("Native refresh requires one finite model group of at most 21 points")
    models = {str(cell.get("model_id") or "") for cell in cells}
    if len(models) != 1:
        raise ValueError("Native refresh groups all cuts in one normal model job")
    if {_uid(cell) for cell in cells} != {row["uid"] for row in selected if row["model_id"] in models} or len({_uid(c) for c in cells}) != len(cells):
        raise ValueError("Native refresh must retain every original UID in the selected model group")
    native_config = profile.get("native_producers") or {}
    full_config = native_config.get("full_baselines") or {}
    if (native_config.get("deepx_runtime_mode") != "A4" or native_config.get("build_missing_engines") is not False
            or set(native_config.get("backends") or []) != {"deepx"}
            or set(full_config.get("active_producers") or []) != {"deepx"}
            or set(native_config.get("split_backends") or []) not in ({"deepx"}, {"deepx_to_trt"})
            or full_config.get("backends_by_producer") != {"deepx": ["deepx"]}
            or native_config.get("force_rebuild") is True):
        raise ValueError("Native refresh requires explicit DeepX-only A4 on existing artifacts")
    normalized, regular = [], None
    for original in cells:
        if (original.get("setup_id") != "orin_nx_deepx_m1_01"
                or original.get("backend", original.get("native_backend")) not in {"deepx_to_trt", "native_full_deepx"}
                or original.get("runtime_variants") != ["native"]):
            raise ValueError("Native refresh cannot select Generic, Hailo or optional GPU controls")
        # Reuse the existing recipe/source validation one cell at a time. Its
        # public one-split/model acceptance restriction remains unchanged.
        single = copy.deepcopy(config)
        single.update(schema=acceptance.SCHEMA, cells=[original], mode="scientific", phase="complete")
        compatibility = {**profile, "hardware_acceptance": single}
        adapted_options = copy.copy(options)
        if adapted_options is not None:
            adapted_options.resume = False  # Refresh has its own preserved-slot Resume.
        checked = acceptance.validate_request(compatibility, adapted_options)
        cell = checked["cells"][0]
        images = _read(_bound(cell.get("numerical_images")))
        if len(images.get("images") or []) != 1 or len(images.get("source_images") or []) != 1:
            raise ValueError("Native refresh uses the existing bounded one-image point probe")
        cell["uid"] = _uid(cell)
        if cell.get("cohort_reference"):
            _bound(cell["cohort_reference"])
        normalized.append(cell)
        regular = checked["regular_budget"]
    config.update(cells=normalized, mode="scientific", phase="complete", regular_budget=regular,
                  source_report_root=checked["source_report_root"], source_primary_run=checked["source_primary_run"])
    return config


def summary_lines(profile: Mapping) -> list[str]:
    config = validate_request(profile)
    cells = config["cells"]
    requested_reuse = sum(bool(cell.get("cohort_reference")) for cell in cells)
    return [f"Native-Neuaufnahme: {len(cells)} feste Originalpunkte für {cells[0]['model_id']}; alle vorhandenen Cuts im selben Modelljob.",
            "DeepX A4, Batch 1, eine Engine, höchstens vier offene Jobs, Split FIFO 3; keine Modellbauten.",
            "Je offenem Punkt: 3 frische Prozesse mit 100 Warmups / 1000 Completions und 3 gültige 60-s-Energiecaptures; Energie-Warmup 0.",
            "Numerik: vorhandene begrenzte Einbild-Punktprüfung und höchstens 64 Smoke-Tasks; keine erneute 32-Bild-/Dataset-Quality-Schleife.",
            f"Zur Prüfung gebundene 3+3-Kohorten: {requested_reuse}; Reuse zählt erst nach tatsächlicher UID-/Quellen-/Importprüfung.",
            "Resume erhält gültige Slots und ursprüngliche Fehlstartbudgets. Generic, Hailo und optionale GPUkontrollen bleiben unverändert.",
            *[f"  {cell['uid']}" for cell in cells]]


def _cohort_proof(cell, prepared, reference):
    """Revalidate a complete original normal A4 cohort, never a row-count claim."""
    result_path = _bound(reference)
    output = result_path.parent
    identity = prepared["identity"]
    component = "__".join(identity[key] for key in ("model", "setup_id", "case"))
    if result_path.name != "RESULT.json" or output.name != component or output.parent.name != "native":
        raise ValueError("Native cohort reference is not an original normal point result")
    run = output.parent.parent
    manifest = _read(run / "run_manifest.json")
    workflow_origin = _normal_workflow_origin(run, manifest, identity)
    closure = _read(run / "reports/artifact_index_closure.json")
    result, previous = _read(result_path), _read(output / "PREPARED.json")
    if (manifest.get("run_id") != run.name or manifest.get("status") not in {"ok", "partial"}
            or closure.get("schema") != "onnx-splitpoint/artifact-index-terminal-closure"
            or closure.get("status") != "pass" or closure.get("run_id") != run.name
            or closure.get("workflow_status") != manifest.get("status")
            or result.get("status") != "completed" or result.get("normal_projection") is not True
            or result.get("valid_repeats") != {"performance": 3, "energy": 3}
            or previous.get("identity") != identity or previous.get("deepx_runtime_mode") != "A4"):
        raise ValueError("Native cohort lacks a complete closed normal A4 3+3 result")
    old_request = manifest.get("hardware_acceptance") or manifest.get("native_refresh") or {}
    if old_request.get("source_campaign_resumed") is not False:
        raise ValueError("Native cohort is not a new normal bounded generation")
    for field in ("source_plan", "source_plan_sha256", "source_primary_run", "source_raw_report_sha256", "source_validation", "source_validation_sha256",
                  "source_contract", "target_binding", "performance_recipe", "energy_recipe", "full_baseline_references", "deepx_runtime_mode"):
        if previous.get(field) != prepared.get(field):
            raise ValueError("Native cohort source/artifact/input/quality binding differs: " + field)
    summary_path = output / "reports/native_producer_combined_summary.json"
    summary = _read(summary_path).get("rows") or []
    if len(summary) != 1 or any(summary[0].get(k) != v for k, v in identity.items()) or summary[0].get("repetition_count_valid") != 3:
        raise ValueError("Native cohort normal performance import is incomplete")
    slots = native._same_run_performance_slots(previous)["slots"]
    if set(slots) != {1, 2, 3}:
        raise ValueError("Native cohort normal slot selector lacks three accepted performance leaves")
    runtime_ids = [slots[repeat]["runtime_instance_id"] for repeat in (1, 2, 3)]
    leaves = [acceptance.file_binding(slots[repeat]["report"]) for repeat in (1, 2, 3)]
    imported = native.verify_performance_import(previous, [Path(slots[repeat]["report"]) for repeat in (1, 2, 3)])
    from .native_energy_reentry import selected_energy_view
    energy_view = selected_energy_view(previous)
    energy = [row for row in native.collect_native_energy(energy_view, include_partial=False)
              if all(row.get(k) == v for k, v in identity.items())]
    if (len(energy) != 1 or energy[0].get("ok") is not True or energy[0].get("energy_repeat_valid_n") != 3
            or energy[0].get("deepx_runtime_mode") != "A4" or energy[0].get("energy_calibration_verified") is not True
            or energy[0].get("environment_observation_status") != "pass"):
        raise ValueError("Native cohort normal energy/calibration/environment import is incomplete")
    energy_result = energy_view / "reports/native_energy_measurements/native_producer_energy_results.json"
    contract_name = "full_command_contract" if identity["case"] == "full" else "native_command_contract"
    command_hash = (summary[0].get(contract_name) or {}).get("contract_sha256")
    if imported["summary"] != summary_path or imported["command_contract_sha256"] != command_hash:
        raise ValueError("Native cohort normal import differs from its accepted performance slots")
    _energy_plan_for_cohort(energy_result, identity, command_hash)
    native._verify_a4_energy_row(previous, energy[0], command_hash)
    return {"identity": identity, "uid": _uid(cell), "reference": acceptance.file_binding(result_path),
            "run_manifest": acceptance.file_binding(run / "run_manifest.json"),
            "closure": acceptance.file_binding(run / "reports/artifact_index_closure.json"),
            "prepared": acceptance.file_binding(output / "PREPARED.json"),
            "performance_summary": acceptance.file_binding(summary_path), "performance_reports": leaves,
            "performance_import": acceptance.file_binding(output / "performance_import.json"),
            "energy_result": acceptance.file_binding(energy_result), "runtime_instance_ids": runtime_ids,
            "valid_repeats": {"performance": 3, "energy": 3},
            # Preserve existing GUI reuse proofs byte-for-byte in structure.
            # Other ordinary starts retain their observed context without a
            # fabricated GUI identity (historical CLI starts may have none).
            **{"GUI_origin" if workflow_origin["context"].get("kind") == "gui" else "workflow_origin": workflow_origin},
            "original_result": result}


def _energy_plan_for_cohort(path, identity, command_hash):
    """Use the ordinary collector's rows[].row plan, not a fabricated view."""
    plans = [record.get("row") or {} for record in _read(path).get("rows") or []]
    if (len(plans) != 1 or any(plans[0].get(key) != value for key, value in identity.items())
            or not command_hash or plans[0].get("successful_command_contract_sha256") != command_hash
            or float(plans[0].get("duration_s") or 0) != 60
            or plans[0].get("measurement_effective_repeats") != 3
            or int(plans[0].get("warmup") or 0) != 0):
        raise ValueError("Native cohort energy does not match its identity, performance command and 3x60-second recipe")
    return plans[0]


def _normal_workflow_origin(run, manifest, identity):
    """Reconstruct the ordinary frozen start; origin is observational only."""
    from .start_snapshot import build_profile_start_snapshot, public_start_snapshot_metadata
    origin = manifest.get("start_origin") or {}
    sessions = manifest.get("execution_sessions") or []
    if (not isinstance(sessions, list)
            or not any(isinstance(session, Mapping) and session.get("status") in {"ok", "partial"}
                       for session in sessions)):
        raise ValueError("Native cohort has no completed ordinary workflow execution session")
    metadata = manifest.get("profile_start_snapshot") or {}
    if (metadata.get("schema") != "onnx-splitpoint/evaluation-start-snapshot"
            or _read(run / "profile_start_snapshot.json") != metadata):
        raise ValueError("Native workflow cohort lacks its ordinary saved profile snapshot")
    source = yaml.safe_load((run / "profile_source.yaml").read_text())
    resolved = yaml.safe_load((run / "profile.yaml").read_text())
    native_config = resolved.get("native_producers") or {}
    request = native.request_config(resolved)
    if (request.get("enabled") is not True or native_config.get("deepx_runtime_mode") != "A4"
            or tuple(native_config.get(key) for key in ("warmup", "frames", "repetitions")) != (100, 1000, 3)
            or not any(row.get("model_id") == identity["model"] and row.get("setup_id") == identity["setup_id"]
                       and row.get("case_id") == identity["case"] and row.get("runtime_variants") == ["native"]
                       and (row.get("backend") or row.get("native_backend")) == identity["backend"]
                       for row in request.get("cells") or [])):
        raise ValueError("Native workflow cohort is outside its original saved A4 3-process selection")
    rebuilt = public_start_snapshot_metadata(build_profile_start_snapshot(
        profile_request=str(metadata.get("profile_request") or ""), source_profile=source, resolved_profile=resolved,
        profile_id=str(metadata.get("profile_id") or ""), profile_path=str(metadata.get("profile_path") or ""),
        profile_source=str(metadata.get("profile_source") or ""), runtime_bindings=metadata.get("runtime_bindings") or {},
        schema_version=int(metadata.get("schema_version") or 1)))
    for key in ("source_profile_sha256", "resolved_profile_sha256", "resolved_execution_sha256", "snapshot_sha256", "selection_fingerprint"):
        if not metadata.get(key) or rebuilt.get(key) != metadata[key]:
            raise ValueError("Native workflow cohort profile snapshot binding differs: " + key)
    return {"context": origin, "snapshot": acceptance.file_binding(run / "profile_start_snapshot.json")}


def execute_point(cell, prepared, context, execute):
    """Existing product numeric and complete paths, or verified whole-cohort reuse."""
    output = Path(prepared["output"])
    if cell.get("cohort_reference"):
        proof = _cohort_proof(cell, prepared, cell["cohort_reference"])
        proof_path = output / "COHORT_REUSE.json"
        native._write(proof_path, proof)
        result = {**proof["original_result"], "status": "completed", "reused_cohort": True, "fresh_hardware_executed": False,
                "new_performance_processes": 0, "new_capture_count": 0,
                "retained_fresh_performance_processes": 3, "retained_valid_energy_captures": 3,
                "retained_failed_attempts": proof["original_result"].get("failed_attempts"),
                "retained_energy_start_accounting": proof["original_result"].get("energy_start_accounting"),
                "failed_attempts": 0, "attempts": 0, "energy_start_accounting": {"collector_starts": 0, "workload_starts": 0},
                "valid_repeats": {"performance": 3, "energy": 3}, "normal_projection": True,
                "cohort_reuse_proof": str(proof_path), "evidence_paths": [str(proof_path)],
                "original_evidence_paths": proof["original_result"].get("evidence_paths", []),
                "quality_dataset_reexecutions": 0, "full_recaptures": 0}
        native._write(output / "RESULT.json", result)
        return result
    numerical = output / "NUMERICAL_RESULT.json"
    if not numerical.is_file():
        from .hardware_acceptance_numerical import run_numerical_native
        result = acceptance._accepted_numerical_result(run_numerical_native(
            prepared, context, acceptance.file_binding(cell["numerical_images"])), context)
        if result.get("status") != "numerical_passed":
            return {**result, "status": "failed", "numerical_status": result.get("status")}
    bound_cell = {**cell, "numerical_reference": acceptance.file_binding(numerical)}
    proof = acceptance._verify_numerical_reference(bound_cell, prepared, context)
    native._write(output / "NUMERICAL_ADMISSION.json", proof)
    return execute(prepared, context)


def accepted_result(result, context):
    """Retained cohorts never masquerade as newly acquired capture slots."""
    if result.get("reused_cohort") is not True:
        return acceptance._accepted_result(result, context)
    path = Path(str(result.get("cohort_reuse_proof") or "")).resolve(strict=True)
    if not path.is_relative_to(Path(context["run_dir"])) or path.name != "COHORT_REUSE.json":
        raise ValueError("Native reuse proof is outside this normal refresh job")
    proof = _read(path)
    prepared = _read(path.parent / "PREPARED.json")
    identity = prepared["identity"]
    cell = {"model_id": identity["model"], "setup_id": identity["setup_id"], "case_id": identity["case"], "backend": identity["backend"]}
    if (_cohort_proof(cell, prepared, proof["reference"]) != proof
            or result.get("valid_repeats") != {"performance": 3, "energy": 3}
            or result.get("new_capture_count") != 0 or result.get("new_performance_processes") != 0
            or result.get("fresh_hardware_executed") is not False):
        raise ValueError("Native reused cohort or physical start accounting changed")
    return {**result, "status": "ok"}


def run_refresh(runner):
    """Enter the existing ordinary workflow executor after exact selection."""
    config = validate_request(runner.profile_payload, runner.options)
    return acceptance.run_acceptance(runner, config=config, workflow_kind="native_refresh")
