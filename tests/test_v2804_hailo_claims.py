"""AP5 uses a derived original Q2 terminal report, not unavailable Q1 arrays.

The existing terminal projection is exercised offline. This is neither another
GPU/runtime test nor an independent recount against the absent original labels.
"""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import zipfile

import pytest


ROOT = Path(__file__).parents[1]
FIXTURE = ROOT / "tests/fixtures/v2804_hailo8/Q2_terminal_report.json"
spec = importlib.util.spec_from_file_location("runtime_claims_v2804", ROOT / "scripts/hailo_model_runtime_probe_v27934.py")
runtime = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(runtime)


@pytest.fixture
def terminal():
    record = json.loads(FIXTURE.read_text())
    assert record["derived_from"] == "complete printed JSON in terminal transcript"
    assert record["source"]["filename"] == "Eingefügter Text(20260912-121049).txt"
    return record["report"]


def test_t05_01_g3_does_not_require_equal_top1_or_grant_quality(tmp_path, terminal):
    before = copy.deepcopy(terminal)
    output = tmp_path / "runtime"
    output.mkdir()
    assert runtime._finish_evidence(output, terminal, False) == 0
    stored = json.loads((output / "comparison.json").read_text())
    assert stored["g3_status"] == "pass"
    assert stored["quality_status"] == "not_evaluated_against_campaign_gate"
    assert stored["claim_eligible"] is False
    assert [stored["stages"][s]["top1_correct"] for s in ("cpu_hef", "gpu_hef")] == [8, 10]
    assert stored["stages"] == before["stages"]


def test_t05_02_03_missing_emulation_and_internal_uint8_remain_missing(tmp_path, terminal):
    output = tmp_path / "runtime"
    output.mkdir()
    runtime._finish_evidence(output, terminal, False)
    with zipfile.ZipFile(tmp_path / "runtime_evidence.zip") as archive:
        stored = json.loads(archive.read("runtime/comparison.json"))
    assert stored["stages"]["full_precision"] == {"status": "not_available"}
    assert stored["stages"]["quantized_emulation"] == {"status": "not_available"}
    assert stored["input_format"] == "FLOAT32"
    assert stored["actual_vstream_feed_equal"] is True
    assert stored["internal_uint8_feed_observed"] is False
    assert all(stored["stages"][s]["item_count"] == 16
               for s in ("source_float", "build_float", "cpu_hef", "gpu_hef"))


def test_t05_04_05_terminal_evidence_preserves_plan_and_original_quality(tmp_path, terminal):
    output = tmp_path / "runtime"
    output.mkdir()
    plan = {"runtime_status": "not_run", "quality_status": "not_evaluated",
            "claim_eligible": False, "source_role": "initial_plan"}
    (output / "plan.json").write_text(json.dumps(plan))
    original_plan = (output / "plan.json").read_bytes()
    runtime._finish_evidence(output, terminal, False)
    assert (output / "plan.json").read_bytes() == original_plan
    assert terminal["runtime_status"] == "pass"
    assert terminal["original_quality_decision_preserved"] is True
    assert terminal["compiler_invoked"] is False and terminal["energy_invoked"] is False
    assert terminal["setup_id"] == "orin_nx_hailo8_01"
    assert terminal["stages"]["cpu_hef"]["origin"]["artifact_sha256"] != terminal["stages"]["gpu_hef"]["origin"]["artifact_sha256"]
    assert "speedup" not in terminal
    assert terminal["thresholds_changed"] is False

