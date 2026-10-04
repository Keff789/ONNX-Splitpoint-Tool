"""Frozen H8 oracle admission; no native runtime, compiler or inference."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import pytest

from onnx_splitpoint_tool import native_detection_postprocess as ndp

ROOT = Path(__file__).resolve().parents[1]
RESOURCE = ROOT / "onnx_splitpoint_tool/resources/native_concurrent_three_stage_yolov7"


@pytest.fixture
def canary(monkeypatch):
    monkeypatch.syspath_prepend(str(RESOURCE))
    spec = importlib.util.spec_from_file_location("h8_oracle_canary_test", RESOURCE / "three_stage_canary.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def frozen_contract():
    return json.loads((RESOURCE / "reference/current_completed_task_result.json").read_text())[
        "completion_execution_contract"]["processor_contract"]


def test_original_contract_uses_existing_versioned_verifier_without_resealing(canary):
    contract = frozen_contract()
    original = copy.deepcopy(contract)
    processor, yolo, checks = canary._verified_oracle_sources(ROOT, contract)
    assert processor is ndp
    assert contract == original
    assert all(row["passed"] for row in checks)
    assert {row["role"] for row in checks if not row["byte_identical"]} == {
        "native_detection_postprocess", "yolo_harness"}
    verified = yolo.verify_yolov7_decoder_contract(contract["model_bound_decoder_contract"])
    assert verified["schema_version"] == 1
    assert "sigmoid_arithmetic" not in verified
    assert {row["expected_sha256"] for row in checks} == {
        item["sha256"] for item in original["implementation_artifacts"].values()}


@pytest.mark.parametrize("mutation", ["unknown_source", "source_alias", "threshold", "decoder_version", "contract_hash"])
def test_existing_contract_guards_reject_changed_proof(canary, mutation):
    contract = frozen_contract()
    if mutation == "unknown_source":
        contract["implementation_artifacts"]["yolo_harness"]["sha256"] = "0" * 64
    elif mutation == "source_alias":
        contract["implementation_artifacts"]["yolo_harness"]["relative_path"] = "elsewhere/yolo.py"
    elif mutation == "threshold":
        contract["confidence_threshold"] = 0.5
    elif mutation == "decoder_version":
        contract["model_bound_decoder_contract"]["schema_version"] = 2
    else:
        contract["contract_sha256"] = "0" * 64
    if mutation != "contract_hash":
        unsigned = {key: value for key, value in contract.items() if key != "contract_sha256"}
        contract["contract_sha256"] = ndp.canonical_json_sha256(unsigned)
    with pytest.raises((ndp.FrozenPostprocessError, RuntimeError)):
        canary._verified_oracle_sources(ROOT, contract)


@pytest.mark.parametrize("damage", ["missing", "changed"])
def test_loaded_sources_must_match_selected_tool_root(canary, tmp_path, damage):
    contract = frozen_contract()
    for item in contract["implementation_artifacts"].values():
        relative = item["relative_path"]
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    target = tmp_path / contract["implementation_artifacts"]["yolo_harness"]["relative_path"]
    if damage == "missing":
        target.unlink()
    else:
        target.write_bytes(target.read_bytes() + b"\n# altered\n")
    with pytest.raises(RuntimeError, match="frozen_oracle_source_mismatch"):
        canary._verified_oracle_sources(tmp_path, contract)


def test_normal_canary_main_reaches_artifact_boundary_after_real_oracle_admission(canary, monkeypatch, tmp_path):
    reference = RESOURCE / "reference/multi_image_fast_decode_canary_report.json"
    corpus = tmp_path / "corpus.json"
    corpus.write_text(json.dumps(json.loads(reference.read_text())["corpus"]))
    monkeypatch.setattr(sys, "argv", ["three_stage_canary.py", "--tool-root", str(ROOT),
        "--corpus", str(corpus), "--reference-report", str(reference), "--out-root", str(tmp_path / "out")])
    checked = []
    original = canary._verified_oracle_sources

    def observe(*args):
        value = original(*args)
        checked.append(value[2])
        return value

    class StopBeforeRuntime(BaseException):
        pass

    def stop(_spec):
        assert len(checked) == 1 and all(item["passed"] for item in checked[0])
        raise StopBeforeRuntime()

    monkeypatch.setattr(canary, "_verified_oracle_sources", observe)
    monkeypatch.setattr(canary, "choose_exact_artifact", stop)
    with pytest.raises(StopBeforeRuntime):
        canary.main()


def test_frozen_reference_metadata_tamper_remains_negative(canary, monkeypatch, tmp_path):
    reference = RESOURCE / "reference/multi_image_fast_decode_canary_report.json"
    corpus = tmp_path / "corpus.json"
    corpus.write_text(json.dumps(json.loads(reference.read_text())["corpus"]))
    expected = json.loads((RESOURCE / "expected_artifacts.json").read_text())
    expected["reference_metadata"]["current_result"]["sha256"] = "0" * 64
    expected_path = tmp_path / "expected.json"
    expected_path.write_text(json.dumps(expected))
    monkeypatch.setattr(sys, "argv", ["three_stage_canary.py", "--tool-root", str(ROOT),
        "--expected", str(expected_path), "--corpus", str(corpus),
        "--reference-report", str(reference), "--out-root", str(tmp_path / "out")])
    monkeypatch.setattr(canary, "choose_exact_artifact", lambda _: pytest.fail("artifact/runtime reached"))
    assert canary.main() == 2
    report = json.loads(next((tmp_path / "out").glob("*/three_stage_canary_report.json")).read_text())
    assert "frozen_reference_metadata_mismatch" in json.dumps(report)
