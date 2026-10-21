"""Normal terminal reference status; full export checks remain mandatory."""
import copy

import pytest

from onnx_splitpoint_tool import generic_only_plan as admission
from test_generic_only_plan import document, quality_fixture


@pytest.mark.parametrize("cpu_status", ["completed", "cache_hit"])
@pytest.mark.parametrize("quality_status", ["reference_close", "accuracy_loss", "inconclusive"])
def test_successful_cpu_reference_status_preserves_quality_decision(tmp_path, monkeypatch, cpu_status, quality_status):
    row, result = quality_fixture(tmp_path, monkeypatch, quality_status)
    result["management_cpu_reference"]["status"] = cpu_status
    row["quality_reference"]["result"] = document(tmp_path / "result.json", result)
    proof = admission.validate_quality_reference(row)
    assert proof["status"] == quality_status
    assert proof["quality_eligible"] == (quality_status == "reference_close")
    assert result["management_cpu_reference"]["status"] == cpu_status


@pytest.mark.parametrize("cpu_status", [None, "", "queued", "running", "failed", "incomplete"])
def test_nonterminal_or_unsuccessful_reference_cannot_admit(tmp_path, monkeypatch, cpu_status):
    row, result = quality_fixture(tmp_path, monkeypatch)
    result["management_cpu_reference"]["status"] = cpu_status
    row["quality_reference"]["result"] = document(tmp_path / "result.json", result)
    with pytest.raises(ValueError, match="CPU reference"):
        admission.validate_quality_reference(row)


@pytest.mark.parametrize("field,value", [
    ("reference_sha256", "f" * 64), ("provider", "cuda"), ("semantic_reference_only", False)])
def test_cache_hit_does_not_override_cpu_artifact_or_semantics(tmp_path, monkeypatch, field, value):
    row, result = quality_fixture(tmp_path, monkeypatch)
    result["management_cpu_reference"].update(status="cache_hit", **{field: value})
    row["quality_reference"]["result"] = document(tmp_path / "result.json", result)
    with pytest.raises(ValueError, match="CPU reference"):
        admission.validate_quality_reference(row)


@pytest.mark.parametrize("source", ["request", "cpu_reference", "policy_source"])
def test_cache_hit_still_requires_actual_bound_source_bytes(tmp_path, monkeypatch, source):
    row, result = quality_fixture(tmp_path, monkeypatch)
    result["management_cpu_reference"]["status"] = "cache_hit"
    row["quality_reference"]["result"] = document(tmp_path / "result.json", result)
    row["quality_reference"][source] = copy.deepcopy(row["quality_reference"][source])
    row["quality_reference"][source]["path"] = str(tmp_path / ("missing-" + source + ".json"))
    with pytest.raises((OSError, ValueError)):
        admission.validate_quality_reference(row)


@pytest.mark.parametrize("validator", ["completed_export", "portable_cpu_records"])
def test_cache_hit_cannot_bypass_full_normal_validation(tmp_path, monkeypatch, validator):
    from onnx_splitpoint_tool import quality_service
    row, result = quality_fixture(tmp_path, monkeypatch)
    result["management_cpu_reference"]["status"] = "cache_hit"
    row["quality_reference"]["result"] = document(tmp_path / "result.json", result)
    def reject(*args, **kwargs):
        raise ValueError("actual full normal validation failed")
    if validator == "completed_export":
        monkeypatch.setattr(admission, "_completed_quality_binding", reject)
    else:
        monkeypatch.setattr(quality_service, "quality_request_from_manifest", reject)
    with pytest.raises(ValueError, match="actual full normal validation failed"):
        admission.validate_quality_reference(row)
