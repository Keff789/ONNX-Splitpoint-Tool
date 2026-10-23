"""Explicit existing compiler bridge uses the genuine normal eight-file guard."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from onnx_splitpoint_tool import native_split_quality
from onnx_splitpoint_tool.runners import generic_quality_completion as quality
from tests.test_generic_quality_completion import descriptor
from tests.test_generic_quality_hailo8 import h8_config, h8_capture
from tests.test_v269f_native_split_receipt_validation import _fixture_payload


def bridge_fixture(tmp_path):
    fixture = h8_config(tmp_path)
    payload, paths = _fixture_payload(tmp_path/"actual_normal_binding")
    bound = native_split_quality.seal_native_split_quality_binding(payload)
    source = tmp_path/"original_composed_request.json"
    source.write_text(json.dumps({"native_split_quality_binding":bound}))
    cfg=fixture["config"]
    cfg["runtime_source_files"].append(descriptor(native_split_quality.__file__))
    selected=bound["preselection"]
    cfg["identity"].update(case_id=selected["case_id"], setup_id=selected["setup_id"],precision=selected["precision"])
    cfg["representative_input"]["runtime_input_adapter"]["runtime_input_format"] = "uint8"
    fixture["context"].update(case_id=selected["case_id"],setup_id=selected["setup_id"])
    for role in ("part1_runtime","boundary_metadata","source_part2_onnx","build_part2_onnx","engine","engine_build_receipt"):
        cfg["expected_runtime_artifacts"][role]=deepcopy(bound["artifacts"][role])
    fixture["context"]["part2_model_path"]=cfg["expected_runtime_artifacts"]["source_part2_onnx"]["path"]
    fixture["context"]["runtime_artifacts"]=[{"role":role,**cfg["expected_runtime_artifacts"][artifact]}
        for role,artifact in [("stage1_runtime","part1_runtime"),("stage2_runtime","engine")]]
    cfg["generic_artifact_binding"]["artifacts"]=deepcopy(cfg["expected_runtime_artifacts"])
    cfg["original_run_cfg"]={"native_trt_precision":selected["precision"],"native_trt_workspace_mb":4096,
        "boundary_mode":"raw_uint8_hailo"}
    cfg["existing_part2_bridge"]={"schema":"onnx-splitpoint/existing-part2-bridge-reference","schema_version":1,
        "binding_sha256":bound["binding_sha256"],"policy_sha256":selected["policy_sha256"],"precision":selected["precision"],
        "binding_source":{**descriptor(source),"pointer":"/native_split_quality_binding"}}
    fixture["path"].write_text(json.dumps(cfg))
    return fixture,bound,paths


def test_actual_normal_eight_file_guard_runs_before_and_after_session_boundary(tmp_path,monkeypatch):
    fixture,bound,paths=bridge_fixture(tmp_path)
    original=native_split_quality.validate_native_split_quality_binding;calls=[]
    def observed(*args,**kwargs):
        calls.append(kwargs["verification_mode"])
        return original(*args,**kwargs)
    monkeypatch.setattr(native_split_quality,"validate_native_split_quality_binding",observed)
    quality.load_quality_completion_config(fixture["path"],check_files=True)
    h8_capture(fixture)
    assert calls==["local","local"]
    # trtexec is an eighth original artifact outside the six shared selected
    # descriptors. Reinitializing the observer must still rehash/reject it.
    paths["trtexec"].write_bytes(paths["trtexec"].read_bytes()+b"changed")
    with pytest.raises(ValueError,match="normal local validator"):
        h8_capture(fixture)


@pytest.mark.parametrize("fault",["source_bytes","wrong_seal","wrong_policy","wrong_pointer","wrong_identity","selected_engine_path","missing_source","validator_source_missing"])
def test_existing_bridge_cannot_authorize_a_different_or_unverified_execution(tmp_path,fault):
    fixture,bound,paths=bridge_fixture(tmp_path);cfg=fixture["config"];bridge=cfg["existing_part2_bridge"]
    if fault=="source_bytes":Path(bridge["binding_source"]["path"]).write_text('{}')
    elif fault=="wrong_seal":bridge["binding_sha256"]="f"*64
    elif fault=="wrong_policy":bridge["policy_sha256"]="f"*64
    elif fault=="wrong_pointer":bridge["binding_source"]["pointer"]="/foreign"
    elif fault=="wrong_identity":cfg["identity"]["case_id"]="b999"
    elif fault=="missing_source":bridge.pop("binding_source")
    elif fault=="validator_source_missing":cfg["runtime_source_files"]=[r for r in cfg["runtime_source_files"] if r["path"]!=str(Path(native_split_quality.__file__).resolve())]
    else:
        p=tmp_path/"different_engine_location.engine";p.write_bytes(paths["engine"].read_bytes())
        cfg["expected_runtime_artifacts"]["engine"]=descriptor(p)
        cfg["generic_artifact_binding"]["artifacts"]=deepcopy(cfg["expected_runtime_artifacts"])
    fixture["path"].write_text(json.dumps(cfg))
    with pytest.raises(ValueError):quality.load_quality_completion_config(fixture["path"],check_files=True)


def test_metadata_only_requires_exact_source_descriptor_without_claiming_file_rehash(tmp_path):
    fixture,_,_=bridge_fixture(tmp_path);cfg=fixture["config"]
    cfg["existing_part2_bridge"]["binding_source"]["path"]="/not_staged_yet/original_request.json"
    fixture["path"].write_text(json.dumps(cfg))
    assert quality.load_quality_completion_config(fixture["path"],check_files=False)["existing_part2_bridge"]
    with pytest.raises(FileNotFoundError):quality.load_quality_completion_config(fixture["path"],check_files=True)
