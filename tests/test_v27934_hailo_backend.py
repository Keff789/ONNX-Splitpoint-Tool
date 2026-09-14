"""Real managed subprocess, calibration and publication paths; SDK is the boundary.

No hardware success is inferred from the small synthetic SDK used here.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from onnx_splitpoint_tool import hailo_backend as backend
from onnx_splitpoint_tool import hailo_compiler_context as context
from onnx_splitpoint_tool.preprocessing_contract import canonical_image_preprocessing_contract


@pytest.fixture
def managed(tmp_path, monkeypatch):
    source = tmp_path / "model.onnx"
    inp = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 8, 8])
    out = helper.make_tensor_value_info("outputs", TensorProto.FLOAT, [1, 3, 8, 8])
    onnx.save(helper.make_model(helper.make_graph([helper.make_node("Identity", ["images"], ["outputs"])], "fixture", [inp], [out])), source)
    calib = tmp_path / "calibration"
    calib.mkdir()
    np.save(calib / "sample.npy", np.zeros((8, 8, 3), np.uint8))
    sdk = tmp_path / "sdk_boundary"
    sdk.mkdir()
    (sdk / "hailo_sdk_client.py").write_text('''
import json, os
from pathlib import Path
__version__ = "BOUNDARY_TEST"
class ClientRunner:
    def __init__(self, **kwargs):
        p = Path.cwd()/"sdk_observed.json"
        p.write_text(json.dumps({"environment":dict(os.environ),"pid":os.getpid(),"cwd":str(Path.cwd())}))
    def translate_onnx_model(self, **kwargs): pass
    def get_hn_dict(self):
        return {"layers":{"images":{"type":"input_layer","output_shape":[1,8,8,3]}}}
    def load_model_script(self, script):
        (Path.cwd()/"actual_model_script.txt").write_text(script)
    def optimize(self, values):
        assert values["images"].shape == (1,8,8,3), values["images"].shape
        if os.environ.get("TEST_SDK_FAIL") == "optimize":
            raise RuntimeError("synthetic SDK optimization infrastructure failure")
    def save_har(self,path): Path(path).write_bytes(b"synthetic SDK HAR")
    def compile(self):
        if os.environ.get("TEST_SDK_FAIL") == "empty": return b""
        return b"synthetic SDK HEF payload"
''')
    monkeypatch.setenv("PYTHONPATH", str(sdk)+os.pathsep+str(Path(__file__).resolve().parents[1]))
    monkeypatch.setattr(backend,"_resolve_managed_venv_python",lambda **kw: ("synthetic-selected",Path(sys.executable),str(Path(sys.executable).parent/"activate")))
    token = "hailo_sdk_client:BOUNDARY_TEST"
    monkeypatch.setattr(backend,"_hailo_sdk_version_token_from_managed_venv",lambda **kw: token)
    monkeypatch.setattr(backend,"_hailo_sdk_version_token",lambda: token)
    monkeypatch.setenv("ONNX_SPLITPOINT_HAILO_CALIBRATION_STORAGE", "memmap")
    monkeypatch.setenv("ONNX_SPLITPOINT_HAILO_CALIB_CAP_MB", "64")
    monkeypatch.setenv("ONNX_SPLITPOINT_HAILO_HEARTBEAT_S", "0")
    monkeypatch.setenv("ONNX_SPLITPOINT_HAILO_DFC_MIN_FREE_BYTES", "1")
    monkeypatch.setenv("ONNX_SPLITPOINT_HAILO_DFC_WORKSPACE_RESERVE_BYTES", "1")
    monkeypatch.setenv("ONNX_SPLITPOINT_HAILO_DFC_MIN_FREE_INODES", "1")
    monkeypatch.setenv("ONNX_SPLITPOINT_HAILO_CACHE_ROOT", str(tmp_path/"production_cache"))
    monkeypatch.setenv("ONNX_SPLITPOINT_ARTIFACT_STORE_ROOT", str(tmp_path/"production_store"))
    monkeypatch.setenv("ONNX_SPLITPOINT_BUILD_EVIDENCE_ROOT", str(tmp_path/"production_evidence"))
    monkeypatch.setenv("ONNX_SPLITPOINT_HAILO_CACHE_ENABLED", "1")
    monkeypatch.setenv("ONNX_SPLITPOINT_ARTIFACT_STORE_ENABLED", "1")
    monkeypatch.setenv("ONNX_SPLITPOINT_TOOL_ROOT", str(tmp_path/"production_logs"))
    for key in (
        "ONNX_SPLITPOINT_HAILO_COMPUTE_BY_FAMILY", "ONNX_SPLITPOINT_HAILO_COMPUTE_OVERRIDE",
        "ONNX_SPLITPOINT_HAILO_COMPUTE", "ONNX_SPLITPOINT_HAILO_ALLOW_GPU",
        "SPLITPOINT_HAILO_ALLOW_GPU", "CUDA_VISIBLE_DEVICES", "XLA_FLAGS",
        "ONNX_SPLITPOINT_CUDA_MASK_SOURCE", "ONNX_SPLITPOINT_HAILO_RESOLVED_COMPILER_CONTEXT",
        "ONNX_SPLITPOINT_HAILO8_DEPENDENCY_MANIFEST", "ONNX_SPLITPOINT_HAILO_CALIB_MANIFEST",
    ):
        monkeypatch.delenv(key,raising=False)
    return dict(onnx_path=source,hw_arch="hailo10h",net_name="fixture_full",
                net_input_shapes={"images":[1,3,8,8]},task="classification",
                preprocessing_contract=canonical_image_preprocessing_contract("classification",(8,8)),
                fixup=False,calib_dir=calib,calib_count=1,calib_batch_size=1,keep_artifacts=True)


def _tree(root):
    return {str(p.relative_to(root)):p.read_bytes() for p in root.rglob("*") if p.is_file()} if root.exists() else {}


def test_isolated_real_child_does_not_publish_or_modify_parent(managed,tmp_path):
    original = dict(os.environ)
    out = tmp_path/"diagnostic"
    result = backend.hailo_build_hef_auto(**managed,outdir=out,backend="venv",force=True,publish_artifacts=False,compute_device="cpu")
    assert result.ok, result.error
    assert dict(os.environ) == original
    for name in ("production_cache","production_store","production_evidence","production_logs"):
        assert not (tmp_path/name).exists(), name
    observed=json.loads((out/"sdk_logs"/"sdk_observed.json").read_text())
    assert observed["pid"] != os.getpid()
    assert observed["environment"]["CUDA_VISIBLE_DEVICES"] == "-1"
    assert Path(observed["environment"]["HOME"]).is_relative_to(out)
    assert result.calib_info["publish_artifacts"] is False
    assert result.calib_info["used_count"] == 1
    assert Path(result.parsed_har_path).is_file() and Path(result.quant_har_path).is_file()
    assert not list(out.rglob("*.mmap"))
    assert backend._load_valid_hailo_receipt(Path(result.hef_path)) is None
    receipt = backend._load_valid_hailo_receipt(Path(result.hef_path),allow_diagnostic=True)
    assert receipt and receipt["diagnostic_only"] is True
    events=json.loads((out/"hailo_build_phases.json").read_text())["events"]
    assert [v["phase"] for v in events if v["state"]=="completed"] == ["sdk_initialization","translate","calibration_materialization","optimize","compile","publication"]
    assert all(v.get("elapsed_s",0)>=0 for v in events)


@pytest.mark.parametrize("failure,stage",[("optimize","optimize"),("empty","compile")])
def test_isolated_failure_keeps_primary_phase_without_publication(managed,tmp_path,monkeypatch,failure,stage):
    monkeypatch.setenv("TEST_SDK_FAIL",failure)
    out=tmp_path/"failed_diagnostic"
    result=backend.hailo_build_hef_auto(**managed,outdir=out,backend="venv",force=True,publish_artifacts=False,compute_device="cpu")
    assert not result.ok
    assert result.last_stage == stage
    assert result.details["phase_events"][-1]["state"]=="failed"
    assert not (out/"compiled.hef").exists()
    for name in ("production_cache","production_store","production_evidence","production_logs"):
        assert not (tmp_path/name).exists()


def test_normal_build_then_gpu_request_reuses_without_probe(managed,tmp_path,monkeypatch):
    first=backend.hailo_build_hef_auto(**managed,outdir=tmp_path/"normal",backend="venv",compute_device="cpu")
    assert first.ok, first.error
    assert first.calib_info["cache_hit"] is False
    receipt=backend._load_valid_hailo_receipt(Path(first.hef_path))
    assert receipt and receipt["publish_artifacts"] is True
    assert (tmp_path/"production_cache"/receipt["cache_key"]/"compiled.hef").is_file()
    assert list((tmp_path/"production_store").rglob("*.hef"))
    def forbidden(*a,**kw):
        pytest.fail("GPU probe or managed child reached despite exact artifact HIT")
    monkeypatch.setattr(context,"resolve_hailo_compiler_context",forbidden)
    monkeypatch.setattr(backend,"_run_streamed_subprocess",forbidden)
    second=backend.hailo_build_hef_auto(**managed,outdir=tmp_path/"second",backend="venv",compute_device="gpu")
    assert second.ok and second.calib_info["cache_hit"]
    assert Path(second.hef_path).read_bytes()==Path(first.hef_path).read_bytes()
    direct=backend.hailo_build_hef_via_venv(**managed,outdir=tmp_path/"direct",compute_device="gpu")
    assert direct.ok and direct.calib_info["cache_hit"]


def test_gpu_marker_reaches_real_sdk_boundary_without_cpu_remasking(managed,tmp_path,monkeypatch):
    binary=tmp_path/"ptxas";binary.write_text("#!/bin/sh\nexit 0\n");binary.chmod(0o700)
    lib=tmp_path/"libdevice.10.bc";lib.write_bytes(b"synthetic-component")
    monkeypatch.setattr(context,"resolve_hailo_compiler_context",lambda *a,**kw: dict(
        schema_version=1,device="gpu",family="hailo10h",source="test_boundary",gpu_uuid="GPU-SYNTHETIC",
        venv_python=sys.executable,ptxas_path=str(binary),libdevice_path=str(lib),target_probe={"output_size":1}))
    out=tmp_path/"gpu_diagnostic"
    result=backend.hailo_build_hef_auto(**managed,outdir=out,backend="venv",force=True,publish_artifacts=False,compute_device="gpu")
    assert result.ok,result.error
    observed=json.loads((out/"sdk_logs"/"sdk_observed.json").read_text())["environment"]
    assert observed["CUDA_VISIBLE_DEVICES"]=="GPU-SYNTHETIC"
    effective=json.loads(observed[context.CONTEXT_ENV])
    assert observed["CUDA_HOME"]==effective["view_root"]
    assert not Path(effective["view_root"]).exists()
    assert result.calib_info["gpu_execution_status"]=="gpu_execution_unproven"


def test_public_force_guard_precedes_any_attempt_directory(tmp_path):
    with pytest.raises(ValueError,match="force_build_disabled_for_productive_jobs"):
        backend.hailo_build_hef_auto(tmp_path/"missing.onnx",outdir=tmp_path/"out",force=True)
    assert not (tmp_path/"out").exists()


def test_nonpublication_requires_child_and_explicit_isolated_paths(tmp_path):
    with pytest.raises(ValueError,match="isolated_managed_child"):
        backend.hailo_build_hef(tmp_path/"missing.onnx",outdir=tmp_path/"out",publish_artifacts=False)
    with pytest.raises(ValueError,match="outdir_required"):
        backend.hailo_build_hef_via_venv(tmp_path/"missing.onnx",publish_artifacts=False)
    bad=tmp_path/"space name"
    with pytest.raises(ValueError,match="whitespace_free"):
        backend._hailo_diagnostic_child_environment({},bad)
    assert not bad.exists()


def test_diagnostic_symlink_cannot_redirect_private_state(tmp_path):
    root=tmp_path/"job";root.mkdir()
    outside=tmp_path/"outside";outside.mkdir()
    (root/"diagnostic_state").symlink_to(outside,target_is_directory=True)
    with pytest.raises(ValueError,match="symlink"):
        backend._hailo_diagnostic_child_environment({},root)
    assert list(outside.iterdir())==[]


def test_compute_and_diagnostic_policy_do_not_change_artifact_identity(managed):
    base=backend._v60s_hailo_contract(managed)
    assert base==backend._v60s_hailo_contract({**managed,"compute_device":"gpu","gpu_selector":"0","compute_by_family":{"hailo10h":{"device":"gpu"}},"compiler_context":{"view_root":"/temporary/job"},"publish_artifacts":False})


def test_explicit_compute_auto_never_falls_back_to_local_sdk(managed,tmp_path,monkeypatch):
    failed=backend.HailoHefBuildResult(ok=False,elapsed_s=0,hw_arch="hailo10h",net_name="fixture_full",backend="venv",error="Failed to resolve managed DFC venv")
    monkeypatch.setattr(backend,"hailo_build_hef_via_venv",lambda *a,**kw: failed)
    monkeypatch.setattr(backend,"auto_prefers_subprocess",lambda: False)
    monkeypatch.setattr(backend,"hailo_sdk_available",lambda: True)
    monkeypatch.setattr(backend,"hailo_build_hef",lambda *a,**kw: pytest.fail("explicit GPU fell back into controller SDK"))
    result=backend.hailo_build_hef_auto(**managed,outdir=tmp_path/"selected",compute_device="gpu")
    assert result is failed


def test_public_diagnostic_rejects_symlink_before_attempt_or_writes(tmp_path):
    external=tmp_path/"external";external.mkdir()
    link=tmp_path/"link";link.symlink_to(external,target_is_directory=True)
    with pytest.raises(ValueError,match="without_symlinks"):
        backend.hailo_build_hef_auto(tmp_path/"source.onnx",outdir=link,publish_artifacts=False,force=True)
    assert not list(external.iterdir())


@pytest.mark.parametrize("value",["false","0",None,0])
def test_public_nonpublication_flag_rejects_invalid_boolean_before_attempt(tmp_path,value):
    with pytest.raises(ValueError,match="config_boolean_invalid:hailo_build.publish_artifacts"):
        backend.hailo_build_hef_auto(tmp_path/"source.onnx",outdir=tmp_path/"job",publish_artifacts=value,force=True)
    assert not (tmp_path/"job").exists()


def test_attempt_uses_requested_model_budget_not_3600_default(managed):
    bound=backend._v60s_hailo_auto_bound((),{**managed,"wsl_timeout_s":10800,"compute_device":"gpu"})
    assert bound["wsl_timeout_s"]==10800
    assert bound["compute_device"]=="gpu"
    assert "wsl_timeout_s" not in backend._v60s_hailo_contract(bound)


def test_streamed_child_without_workflow_registry_has_owned_cleanup(tmp_path,monkeypatch):
    from onnx_splitpoint_tool import process_control
    monkeypatch.setattr(backend,"current_process_registry",lambda: None)
    created=[]
    Original=process_control.ProcessTreeRegistry
    class Registry(Original):
        def __init__(self):
            super().__init__();created.append(self)
    monkeypatch.setattr(process_control,"ProcessTreeRegistry",Registry)
    result=backend._run_streamed_subprocess([sys.executable,"-B","-c","import time; time.sleep(10)"],hard_timeout_s=0.2)
    assert result.timed_out and result.returncode != 0
    assert len(created)==1 and created[0].is_quiescent()
    assert result.cleanup["remaining_process_count"]==0
