"""Controller serializer to unchanged normal authority; metadata only, no files/HW."""
from copy import deepcopy

import pytest

from onnx_splitpoint_tool.runners import generic_quality_accounting as accounting
from onnx_splitpoint_tool.workflow.generator_binding import (
    _logical_profile_or_target_runs, _stage_descriptor_for_target,
)


CASES = ["b124", "b177", "b239", "b248"]


def scope(backend="deepx_m1"):
    descriptors = {case: {"path": "/controlled/configs/" + case + ".json",
        "sha256": str(index + 1) * 64, "size_bytes": 1234 + index}
        for index, case in enumerate(CASES)}
    row = {"id": backend + "_to_tensorrt", "type": "mixed_backend",
        "stage1": backend, "stage2": "tensorrt", "hardware_setup_id": backend + "_controlled_setup",
        "quality_source_only": True, "variants": ["composed"], "case_ids": list(CASES),
        "generic_quality_completion_configs": {case: "quality_runtime/" + case + ".json" for case in CASES},
        "generic_quality_completion_config_sha256": {case: source["sha256"] for case, source in descriptors.items()},
        "generic_quality_completion_sources": descriptors,
        "validation_images": "resources/validation/detection/val2017_n5000_s20260710",
        "validation_max_images": 5000}
    profile = {"run_profiles": [row], "quality_gate": {
        "statistics": {"execution_location": "central_management"}}}
    catalogue_ids = CASES + ["b" + str(index).zfill(3) for index in range(1, 17)]
    catalogue = [{"case_id": case, "boundary": int(case[1:])} for case in catalogue_ids]
    catalogue_before = deepcopy(catalogue)
    plan = {"runs": _logical_profile_or_target_runs(profile, [backend, "tensorrt"], catalogue, {}),
            "cases": catalogue}
    assert catalogue == catalogue_before
    return profile, plan


@pytest.mark.parametrize("backend", ["deepx_m1", "hailo8", "hailo10"])
def test_normal_serialized_scope_preserves_exact_profile_authority(backend):
    profile, plan = scope(backend)
    authorized = accounting.authorized_runs(plan, profile)
    assert len(authorized) == 1
    assert accounting._binding(authorized[0]) == accounting._binding(profile["run_profiles"][0])
    assert authorized[0]["case_ids"] == CASES
    for field in ("generic_quality_completion_configs", "generic_quality_completion_config_sha256",
                  "generic_quality_completion_sources"):
        assert authorized[0][field] == profile["run_profiles"][0][field]
    assert len(plan["cases"]) == 20
    if backend == "deepx_m1":
        assert authorized[0]["stage1"] == {"type": "deepx", "target": "deepx_m1",
                                           "backend": "deepx_m1", "artifact_kind": "dxnn"}
    else:
        assert authorized[0]["stage1"] == {"type": "hailo", "hw_arch": backend}


@pytest.mark.parametrize("alias", ["deepx_m1", "deepx", "dx_m1"])
def test_existing_deepx_aliases_keep_the_same_canonical_target(alias):
    stage = _stage_descriptor_for_target(alias)
    assert stage["backend"] == stage["target"] == "deepx_m1"
    assert stage["type"] == "deepx" and stage["artifact_kind"] == "dxnn"


@pytest.mark.parametrize("fault", ["missing_backend", "wrong_backend", "unknown_stage", "changed_case_scope"])
def test_unchanged_authority_rejects_missing_or_mismatched_scope(fault):
    profile, plan = scope()
    if fault == "missing_backend":
        plan["runs"][0]["stage1"].pop("backend")  # The exact pre-fix generated representation.
    elif fault == "wrong_backend":
        plan["runs"][0]["stage1"]["backend"] = "hailo8"
    elif fault == "unknown_stage":
        profile["run_profiles"][0]["stage1"] = "unknown_device"
    else:
        plan["runs"][0]["case_ids"] = CASES[:-1]
    with pytest.raises(ValueError, match="generic_quality_accounting_"):
        accounting.authorized_runs(plan, profile)
