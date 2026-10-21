from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

from onnx_splitpoint_tool.native_energy_quality_admission import (
    verify_sealed_energy_quality_admission,
)
from tests import test_v2727_native_energy_runtime_admission as runtime_admission

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PRECISION = "deepx_dxnn_sha256:" + "b" * 64


def _load_planner(relative_path):
    path = ROOT / relative_path
    name = "v2956_precision_planner_" + str(abs(hash(str(path))))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("relative_path", [
    "scripts/native_producer_energy_plan.py",
    "onnx_splitpoint_tool/resources/remote_scripts/native_producer_energy_plan.py",
])
@pytest.mark.parametrize("comparison_precision", ["", "uint8_cast_fp16"])
def test_full_plan_precision_matches_sealed_admission_and_keeps_quality_facts(
    tmp_path, monkeypatch, relative_path, comparison_precision,
):
    planner = _load_planner(relative_path)
    rows = runtime_admission._runtime_matrix_rows()
    for row in rows:
        if row["backend"] == "native_full_deepx":
            row.update(
                precision=comparison_precision,
                comparison_precision=comparison_precision,
                legacy_comparison_precision=comparison_precision,
                runtime_precision_identity=RUNTIME_PRECISION,
            )
    monkeypatch.setattr(runtime_admission, "_load_planner", lambda: planner)
    monkeypatch.setattr(runtime_admission, "_runtime_matrix_rows", lambda: rows)
    # Exercise the real final plan writer and the strict executor's validation
    # through the existing 45-row fixture, including unrelated backends.
    runtime_admission.test_runtime_success_mode_plans_the_complete_45_row_native_matrix(
        tmp_path, monkeypatch,
    )
    payload = json.loads((tmp_path / "energy-plan/native_producer_energy_plan.json").read_text())
    for row in payload["rows"]:
        if row["backend"] != "native_full_deepx":
            continue
        expected = comparison_precision or "posthoc_annotation_unavailable"
        assert row["precision"] == expected
        assert row["energy_quality_admission"]["precision"] == expected
        assert row["comparison_precision"] == comparison_precision
        assert row["legacy_comparison_precision"] == comparison_precision
        assert row["full_runtime_precision"] == RUNTIME_PRECISION
        assert row["execution_precision"] == RUNTIME_PRECISION
        assert row["precision_quality_binding_verified"] is False
        assert row["central_quality_evidence_verified"] is False
        assert row["accuracy_gate_pass"] is False
        assert row["diagnostic_only"] is True
        assert row["energy_claim_eligible"] is False
        verify_sealed_energy_quality_admission(row, required=True)
        changed = copy.deepcopy(row)
        changed["precision"] = "actual_precision_drift"
        with pytest.raises(ValueError, match="energy_quality_admission_precision_drift"):
            verify_sealed_energy_quality_admission(changed, required=True)


def test_precision_fallback_does_not_supply_quality_or_nonruntime_identity():
    planner = _load_planner("scripts/native_producer_energy_plan.py")
    row = {
        "backend": "native_full_deepx",
        "precision": "",
        "runtime_precision_identity": RUNTIME_PRECISION,
    }
    original = copy.deepcopy(row)
    assert planner._energy_quality_admission_precision(
        row, None, runtime_observation_allowed=False,
    ) == ""
    assert planner._energy_quality_admission_precision(
        row, None, runtime_observation_allowed=True,
    ) == "posthoc_annotation_unavailable"
    assert row == original
