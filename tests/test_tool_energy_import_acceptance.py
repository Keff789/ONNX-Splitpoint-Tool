"""Portable saved-result import and projection, with no collector available."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import shutil

import pytest

from onnx_splitpoint_tool.workflow import energy_current_projection as projection
from onnx_splitpoint_tool.workflow import energy_parent_import as importer
from onnx_splitpoint_tool.workflow.energy_parent_flow import NATIVE_SCHEMA


@pytest.mark.parametrize("previous_csv", [None, "", "model_id,case_id\n"])
def test_targeted_projection_writes_complete_csv_and_retains_other_rows(tmp_path, monkeypatch, previous_csv):
    identity = {"model_id": "fixture", "case_id": "b001", "setup_id": "setup",
                "precision": "fp16", "direction": "deepx_m1_to_tensorrt"}
    selected = {**identity, "case_key": "chosen", "normal_claim_eligible": False}
    other = {**identity, "case_id": "b002", "case_key": "retained",
             "normal_claim_eligible": True, "energy_per_task_j": 19.0}
    path = tmp_path / "generic_current.json"
    original = {"rows": [selected, other], "energy_errors": []}
    path.write_text(json.dumps(original))
    if previous_csv is not None:
        path.with_suffix(".csv").write_text(previous_csv)
    values = {"energy_status": "verified", "energy_per_task_j": 2.5,
              "energy_per_task_j_ci": [2.0, 3.0], "energy_efficiency_claim_eligible": False,
              "energy_provenance": {"normal_report_verified": True}, "energy_repeats": [{"index": 0}]}
    monkeypatch.setattr(projection, "generic_energy_projection", lambda *args: values)
    projection.project_current_energy({"identity": identity, "case_key": "chosen"}, tmp_path / "result",
        current_path=path, expected_current_sha256=projection.sha(path))
    actual = json.loads(path.read_text())
    assert actual["rows"][1] == other
    assert actual["rows"][0]["normal_claim_eligible"] is False
    with path.with_suffix(".csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["energy_status"] == "verified"
    assert float(rows[0]["energy_per_task_j"]) == 2.5
    assert json.loads(rows[0]["energy_per_task_j_ci"]) == [2.0, 3.0]
    assert json.loads(rows[0]["energy_provenance"]) == values["energy_provenance"]
    assert rows[0]["energy_efficiency_claim_eligible"] == "False"


def test_changed_current_refuses_before_any_write(tmp_path, monkeypatch):
    path = tmp_path / "generic_current.json"
    path.write_text('{"rows":[]}')
    before = path.read_bytes()
    monkeypatch.setattr(projection, "generic_energy_projection", lambda *args: pytest.fail("source read after changed Current"))
    with pytest.raises(ValueError, match="Current changed"):
        projection.project_current_energy({}, None, current_path=path, expected_current_sha256="0" * 64)
    assert path.read_bytes() == before
    assert not path.with_suffix(".csv").exists()


@pytest.mark.parametrize("legacy", ["import_generic_energy.py", "import_qualified_energy.py"])
@pytest.mark.parametrize("flags", [[], ["-B"]])
def test_legacy_command_resolves_installed_module_even_from_older_checkout(tmp_path, legacy, flags):
    shadow = tmp_path / "old-checkout" / "onnx_splitpoint_tool"
    shadow.mkdir(parents=True)
    (shadow / "__init__.py").write_text("raise RuntimeError('old checkout must not shadow installation')")
    old = [sys.executable, *flags, str(tmp_path / legacy), "--help"]
    command = importer.product_import_command(old)
    assert old == [sys.executable, *flags, str(tmp_path / legacy), "--help"]
    assert command == [sys.executable, "-I", "-B", "-m",
                       "onnx_splitpoint_tool.workflow.energy_parent_import", "--help"]
    result = subprocess.run(command, cwd=shadow.parent, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "Import saved Native/Generic energy" in result.stdout
    assert not (shadow / "__pycache__").exists()


def native_parent(tmp_path):
    root = tmp_path / "report"
    cohort = root / "energy" / "saved"
    cohort.mkdir(parents=True)
    budget = root / "energy" / "energy_task_budget.json"
    budget.write_text('{}')
    (cohort / "energy_parent_plan.json").write_text(json.dumps({"schema": NATIVE_SCHEMA,
        "run_dir": str(cohort), "campaign_budget_file": str(budget)}))
    return cohort, budget


def test_native_missing_saved_result_does_not_create_capture_or_reset_budget(tmp_path):
    cohort, budget = native_parent(tmp_path)
    before = budget.read_bytes()
    result = importer.import_cohort(cohort)
    assert result["status"] == "NO_PUBLISHED_NORMAL_RESULT_YET"
    assert budget.read_bytes() == before
    assert not (cohort / "production-start.json").exists()


def test_native_source_override_cannot_escape_saved_measurement(tmp_path):
    cohort, budget = native_parent(tmp_path)
    outside = tmp_path / "native_producer_energy_results.json"
    outside.write_text('{}')
    with pytest.raises(AssertionError):
        importer.import_cohort(cohort, source_override=outside)
    assert budget.read_text() == '{}'


def test_normal_resume_keeps_collector_command_without_private_overlay(tmp_path):
    import onnx_splitpoint_tool
    from onnx_splitpoint_tool.workflow import energy_resume
    installed = Path(onnx_splitpoint_tool.__file__).resolve().parent.parent
    tool = tmp_path / "saved-tool"
    for relative in ("onnx_splitpoint_tool/energy/collector.py", "onnx_splitpoint_tool/energy/task_budget.py",
                     "scripts/energy_measurement_cli.py"):
        destination = tool / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(installed / relative, destination)
    output = tmp_path / "attempt"
    output.mkdir()
    original = [sys.executable, "-B", str(tool / "scripts/energy_measurement_cli.py"), "--help"]
    launcher, hashes = energy_resume.accounting_overlay(output, tool)
    assert energy_resume.bind_accounting_cli(original, launcher, tool) == original
    assert not (output / "controller-accounting").exists()
    assert json.loads((output / "controller-accounting.json").read_text())["private_module_overlay"] is False
    (tool / "onnx_splitpoint_tool/energy/task_budget.py").write_text("changed")
    with pytest.raises(ValueError, match="installed energy source differs"):
        energy_resume.accounting_overlay(output, tool)
