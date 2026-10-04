"""Actual measured starter admission in an explicitly isolated report COW.

Set YOLO_MEASURED_REENTRY_RUN/PLAN only in the externally protected copy;
ordinary portable tests continue covering the original nonstart paths.
"""
from __future__ import annotations
import copy
import hashlib
import json
import os
import shutil
from pathlib import Path

import pytest

from onnx_splitpoint_tool.workflow import yolo_augmentation_reentry as admission
from onnx_splitpoint_tool.workflow.supplement import preserve_metadata, prepare_source_evidence, build_supplement_plan


@pytest.fixture(scope="module")
def measured():
    if not os.environ.get("YOLO_MEASURED_REENTRY_RUN"):
        pytest.skip("requires the explicit external measured-run COW")
    run = Path(os.environ["YOLO_MEASURED_REENTRY_RUN"])
    plan = Path(os.environ["YOLO_MEASURED_REENTRY_PLAN"])
    assert (run / ".measured-reentry-cow").read_text() == "isolated test copy\n"
    proof = admission.validate_augmentation_reentry(run, plan)
    return run, plan, proof


@pytest.fixture
def edit():
    saved = {}
    def update(path, value):
        path = Path(path)
        saved.setdefault(path, path.read_bytes() if path.is_file() else None)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value if isinstance(value, bytes) else json.dumps(value).encode())
    yield update
    for path, raw in saved.items():
        if raw is None:
            path.unlink(missing_ok=True)
        else:
            path.write_bytes(raw)


def test_actual_plan_preserves_twelve_quality_and_completed_native(measured):
    run, plan, proof = measured
    original, index = prepare_source_evidence(run)
    result = build_supplement_plan(run, original_root=original,
        source_index={row["path"]: row for row in index["artifacts"]})
    assert result["counts"]["generic"] == {"preserve": 12}
    assert result["completed_quality_count"] == 12 and result["pending_quality_rows"] == []
    assert result["counts"]["native"] == {"preserve_and_validate_locally": 1, "execute": 11}
    assert proof["kind"] == "augmentation_completed_native_first_generic"


def test_snapshot_survives_new_canonical_stage_and_exact_leaf_reuse(measured, edit):
    run, plan, before = measured
    original = preserve_metadata(run, "pytest-measured-native-continuation")
    try:
        proof = admission.validate_augmentation_reentry(run, plan, original)
        cfg = json.loads((original / "reports/native_producer_stage_config.json").read_text())
        cfg["augmentation_startcase_barrier"]["reentry"] = proof
        leaf_path = run / proof["preserved_native_leaves"][0]["path"]
        leaf_bytes = leaf_path.read_bytes()
        leaf = json.loads(leaf_bytes)
        edit(run / "reports/native_producer_stage.json", {"state": "running", "session_id": "new"})
        assert admission.native_reentry_root(run, cfg) == original.parent / "native"
        assert admission.completed_native_leaf_input(run, cfg, leaf_path,
            leaf["details"]["variant"], leaf["details"]["execution"]["cmd"], "new-coordinator-input") == leaf["input_hash"]
        assert leaf_path.read_bytes() == leaf_bytes
        assert (original / Path(before["predecessor_checkpoint"]).relative_to(run)).read_bytes() == Path(before["predecessor_checkpoint"]).read_bytes()
        for drift in ("variant", "command"):
            variant = copy.deepcopy(leaf["details"]["variant"])
            command = list(leaf["details"]["execution"]["cmd"])
            if drift == "variant": variant["quality_snapshot_sha256"] = "f" * 64
            else: command[command.index("--frames") + 1] = "999"
            with pytest.raises(ValueError, match="current_leaf_contract_changed"):
                admission.completed_native_leaf_input(run, cfg, leaf_path, variant, command, "new-input")
    finally:
        shutil.rmtree(original.parent)


@pytest.mark.parametrize("fault", ["leaf", "additional_leaf", "raw", "raw_count", "raw_time", "raw_runtime_ids",
    "summary", "invocation", "performance", "checkpoint", "quality_request", "quality_candidate", "plan"])
def test_actual_measured_reentry_rejects_drift(measured, edit, fault):
    run, plan, proof = measured
    native = run / proof["predecessor_native_relative"]
    leaf = run / proof["preserved_native_leaves"][0]["path"]
    summary = json.loads((run / "reports/native_producer_summary.json").read_text())
    if fault == "leaf": edit(leaf, {})
    elif fault == "additional_leaf": edit(leaf.parent.parent / "foreign/performance_stage_result.json", {})
    elif fault.startswith("raw"):
        path = Path(summary["rows"][0]["report"])
        raw = json.loads(path.read_text())
        if fault == "raw": raw["ok"] = False
        elif fault == "raw_count": raw["repetition_count_valid"] = 2
        elif fault == "raw_time": raw["endpoint_results"]["completed_task"]["repetition_records"][0]["makespan_ms"] += 100
        else:
            records = raw["endpoint_results"]["completed_task"]["repetition_records"]
            for record in records: record["runtime_instance_id"] = records[0]["runtime_instance_id"]
        edit(path, raw)
        # Even a freshly indexed changed file cannot change the saved rates
        # or turn a failed/partial observation into the required Native3.
        index_path = run / "artifact_index.json"
        index = json.loads(index_path.read_text())
        for row in index["artifacts"]:
            if row["path"] == path.relative_to(run).as_posix():
                row.update(size_bytes=path.stat().st_size, sha256="sha256:" + hashlib.sha256(path.read_bytes()).hexdigest())
        edit(index_path, index)
    elif fault == "summary": edit(run / "reports/native_producer_summary.json", {"rows": []})
    elif fault == "invocation": edit(native / "native_invocation_config.json", {})
    elif fault == "performance": edit(native / "native_performance/stage_result.json", {})
    elif fault == "checkpoint": edit(native / "native_coordinator/stage_result.json", {})
    elif fault in {"quality_request", "quality_candidate"}:
        quality = json.loads((run / "quality_management/central_quality_summary.json").read_text())
        request = run / quality["results"][0]["source_request"]
        if fault == "quality_candidate": request = request.parent / json.loads(request.read_text())["candidate"]["path"]
        edit(request, b"{}")
    else:
        # Plan is mounted read-only in the external harness; exercise drift at
        # its reader without touching the user's frozen plan.
        original_object = admission._object
        with pytest.MonkeyPatch.context() as patch:
            def altered(path):
                result = original_object(path)
                if Path(path) == plan: result["rows"][0]["case_id"] = "foreign"
                return result
            patch.setattr(admission, "_object", altered)
            with pytest.raises(ValueError): admission.validate_augmentation_reentry(run, plan)
        return
    with pytest.raises(ValueError): admission.validate_augmentation_reentry(run, plan)
