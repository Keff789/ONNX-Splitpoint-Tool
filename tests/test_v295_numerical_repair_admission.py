"""Technical repair retains original numerical attempts without a lifetime cap."""
import copy
from pathlib import Path

import pytest

from onnx_splitpoint_tool.workflow import hardware_acceptance_native as native
from onnx_splitpoint_tool.workflow import hardware_acceptance_numerical as numerical
from test_v2951_deepx_lifecycle import phase_prepared, prior_attempt


def test_repaired_third_attempt_retains_every_original_receipt_and_current_preparation(tmp_path):
    prepared, context = phase_prepared(tmp_path)
    native._write(Path(prepared["output"]) / "PREPARED.json", prepared)
    cells = [prior_attempt(prepared, name, dispatched=number > 0, state="terminal")
             for number, name in enumerate(("old_preworker", "old_reader_failure", "further_documented_repair"))]
    native._write(cells[1] / "RUNTIME_DEPLOYMENT.json", {"status": "passed", "inference_submitted": False})
    paths = [path for cell in cells for path in cell.rglob("*") if path.is_file()]
    original = {str(path): path.read_bytes() for path in paths}
    matrix_bytes = Path(prepared["phase_matrix"]["path"]).read_bytes()
    admission = numerical._admit_numerical_attempt(prepared, context, 32)
    assert admission["attempt_number"] == 4 and admission["attempt_limit"] is None
    assert admission["remaining_attempts_after_admission"] is None
    assert admission["historical_matrix_attempt_limit"] == 2
    assert admission["technical_repair_after_prior_failure"] is True
    assert "documented repair" in admission["technical_repair_policy"]
    assert {row["path"] for row in admission["prior_attempts"]} == {str(cell / "NUMERICAL_RESULT.json") for cell in cells}
    for row in admission["prior_attempts"]:
        assert native._sha(Path(row["path"])) == row["sha256"]
        assert native._sha(Path(row["prepared"]["path"])) == row["prepared"]["sha256"]
    assert any("runtime_deployment" in row for row in admission["prior_attempts"])
    assert admission["current_preparation"]["prepared"]["sha256"] == native._sha(Path(prepared["output"]) / "PREPARED.json")
    assert {str(path): path.read_bytes() for path in paths} == original
    assert Path(prepared["phase_matrix"]["path"]).read_bytes() == matrix_bytes
    assert not (Path(prepared["output"]) / "NUMERICAL_RESULT.json").exists()


def test_changing_run_or_profile_name_does_not_reset_attempt_history(tmp_path):
    prepared, context = phase_prepared(tmp_path)
    prior_attempt(prepared, "first", state="terminal")
    prior_attempt(prepared, "second", state="terminal")
    other = copy.deepcopy(prepared)
    run = Path(context["run_dir"]).parent / "different_gui_profile"
    other["output"] = str(run / "native" / Path(prepared["output"]).name)
    proof = numerical._admit_numerical_attempt(other, dict(context, run_dir=str(run)), 32)
    assert proof["attempt_number"] == 3 and len(proof["prior_attempts"]) == 2


def test_lifted_technical_cap_never_allows_recapture_of_accepted_numeric_pass(tmp_path):
    prepared, context = phase_prepared(tmp_path)
    for name in ("first_failure", "second_failure", "third_failure"):
        prior_attempt(prepared, name, state="terminal")
    prior_attempt(prepared, "accepted", status="numerical_passed", state="terminal")
    with pytest.raises(RuntimeError, match="PASS must be reused"):
        numerical._admit_numerical_attempt(prepared, context, 32)


def test_current_prepared_checkpoint_drift_is_not_a_technical_repair_authorization(tmp_path):
    prepared, context = phase_prepared(tmp_path)
    prior_attempt(prepared, "failure", state="terminal")
    native._write(Path(prepared["output"]) / "PREPARED.json", dict(prepared, deepx_runtime_mode="S1"))
    with pytest.raises(ValueError, match="preparation differs"):
        numerical._admit_numerical_attempt(prepared, context, 32)
