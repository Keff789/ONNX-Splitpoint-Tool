"""Historical observer admission requires the real original selected sources."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from onnx_splitpoint_tool.runners import generic_quality_completion as quality
from onnx_splitpoint_tool.workflow import generic_quality_sidecars as sidecars
from onnx_splitpoint_tool.workflow.generic_completed_task import prepare_runner_bundle
from test_generic_quality_completion import descriptor
from test_generic_quality_hailo8 import h8_config, h8_capture


@pytest.fixture
def admitted(tmp_path):
    fixture = h8_config(tmp_path)
    suite = tmp_path / "suite"
    prepare_runner_bundle(suite)
    (suite / "runner").rename(suite / "b039")
    config = fixture["config"]
    config["path_root"] = "suite_root"
    config["runtime_source_files"] = [
        {**descriptor(path), "path": str(path.relative_to(suite))}
        for path in sorted(suite.rglob("*.py"))]
    assert len(config["runtime_source_files"]) == 37
    original = tmp_path / "original_config.json"
    copied = suite / "quality_runtime/b039.json"
    copied.parent.mkdir()
    encoded = json.dumps(config)
    original.write_text(encoded)
    copied.write_text(encoded)
    row = {"id": "hailo8_to_trt", "type": "matrix", "stage1": "hailo8", "stage2": "tensorrt",
           "variants": ["composed"], "quality_source_only": True, "case_ids": ["b039"],
           "hardware_setup_id": config["identity"]["setup_id"], "validation_max_images": 2,
           "generic_quality_completion_configs": {"b039": "quality_runtime/b039.json"},
           "generic_quality_completion_config_sha256": {"b039": descriptor(original)["sha256"]},
           "generic_quality_completion_sources": {"b039": descriptor(original)}}
    profile = {"quality_gate": {"statistics": {"execution_location": "central_management"}}, "run_profiles": [row]}
    plan = {"model_id": "yolo26s", "runs": [deepcopy(row)]}
    arguments = {"profile": profile, "benchmark_plan": plan, "suite_dir": suite,
                 "config_path": original, "model_id": "yolo26s"}
    evidence = {"config_file": descriptor(copied), "identity": config["identity"],
                "base_completion_contract": config["completion_contract"],
                "image_ids": config["expected_image_ids"], "image_count": 2,
                "runtime_artifacts": config["expected_runtime_artifacts"],
                "runtime_source_files": [{**entry, "path": str(suite / entry["path"])}
                                         for entry in config["runtime_source_files"]]}
    return arguments, evidence, config, original, copied


def test_original_config_and_actual_37_sources_authorize_their_exact_projection(admitted):
    arguments, evidence, *_ = admitted
    authority = sidecars.admit_generic_quality_producer_sources(**arguments)
    assert authority.validate_evidence_sources(evidence) is None
    with pytest.raises(AttributeError):
        authority.any_source_hash = "a" * 64


@pytest.mark.parametrize("fault", ["profile_binding", "source_bytes", "missing_source", "missing_declared_source"])
def test_unbound_or_incomplete_original_sources_reject(admitted, fault):
    arguments, _evidence, config, original, copied = admitted
    if fault == "profile_binding":
        arguments["profile"]["run_profiles"][0]["generic_quality_completion_sources"]["b039"]["path"] += ".unbound"
    elif fault == "source_bytes":
        path = arguments["suite_dir"] / config["runtime_source_files"][0]["path"]
        path.write_bytes(path.read_bytes() + b"# changed source\n")
    elif fault == "missing_source":
        (arguments["suite_dir"] / config["runtime_source_files"][0]["path"]).unlink()
    else:
        config["runtime_source_files"].pop()
        original.write_text(json.dumps(config)); copied.write_bytes(original.read_bytes())
        for row in (arguments["profile"]["run_profiles"][0], arguments["benchmark_plan"]["runs"][0]):
            row["generic_quality_completion_sources"]["b039"] = descriptor(original)
            row["generic_quality_completion_config_sha256"]["b039"] = descriptor(original)["sha256"]
    with pytest.raises((ValueError, OSError)):
        sidecars.admit_generic_quality_producer_sources(**arguments)


@pytest.mark.parametrize("fault", ["source_after_admission", "evidence_source", "missing_image", "artifact"])
def test_authority_rechecks_real_bytes_and_frozen_evidence_on_every_use(admitted, fault):
    arguments, evidence, config, *_ = admitted
    authority = sidecars.admit_generic_quality_producer_sources(**arguments)
    evidence = deepcopy(evidence)
    if fault == "source_after_admission":
        path = arguments["suite_dir"] / config["runtime_source_files"][0]["path"]
        path.write_bytes(path.read_bytes() + b"# changed after admission\n")
    elif fault == "evidence_source": evidence["runtime_source_files"][0]["sha256"] = "a" * 64
    elif fault == "missing_image": evidence["image_ids"].pop()
    else: next(iter(evidence["runtime_artifacts"].values()))["sha256"] = "a" * 64
    with pytest.raises(ValueError):
        authority.validate_evidence_sources(evidence)


def test_arbitrary_caller_hash_map_is_not_a_producer_capability(tmp_path):
    fixture = h8_config(tmp_path)
    evidence, records = h8_capture(fixture)
    with pytest.raises(ValueError, match="producer-source authority invalid"):
        quality._validate_evidence(evidence, records, fixture["config"]["completion_contract"],
            fixture["config"]["expected_runtime_artifacts"], producer_source_authority={"sha256": "a" * 64})


def test_normal_quality_reference_forwards_the_admitted_capability(admitted, monkeypatch):
    from onnx_splitpoint_tool import generic_only_plan
    arguments, *_ = admitted
    authority = sidecars.admit_generic_quality_producer_sources(**arguments)
    calls = []
    monkeypatch.setattr(generic_only_plan, "_validate_quality_reference",
        lambda row, **kwargs: calls.append((row, kwargs["producer_source_authority"])) or {"verified": True})
    row = {"task": "detection"}
    assert generic_only_plan.validate_quality_reference(row, producer_source_authority=authority) == {"verified": True}
    assert calls == [(row, authority)]
