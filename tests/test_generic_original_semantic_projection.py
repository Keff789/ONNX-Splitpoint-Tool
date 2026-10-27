"""Original completion semantics with independently verified producer roles."""
import ast
from copy import deepcopy
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import pytest

from onnx_splitpoint_tool.runners import generic_quality_completion as quality
from onnx_splitpoint_tool.split_export_runners import write_runner_skeleton_onnxruntime
from tests import test_generic_quality_completion as helpers


def _fixture(tmp_path):
    f = helpers.make_config(tmp_path)
    original = deepcopy(f["config"]["source_endpoint_contract"])
    base = deepcopy(f["config"]["completion_contract"])
    full = tmp_path / "actual-full-source.onnx"
    full.write_bytes(b"independently bound full ONNX source")
    artifacts = f["config"]["expected_runtime_artifacts"]
    authority = {**original, "full_model_sha256": quality._sha(full),
                 "terminal_model_sha256": artifacts["source_part2_onnx"]["sha256"]}
    fields = quality._endpoint.project_generic_quality_endpoint(authority,
        full_model=helpers.descriptor(full), part2_source=artifacts["source_part2_onnx"],
        original_semantic_endpoint=original, original_completion_contract=base)
    f["config"].update(fields)
    f["config"]["identity"]["comparison_backend"] = "hailo10h"
    f["context"]["source_model_path"] = str(full)
    f["path"].write_text(json.dumps(f["config"]))
    f.update(full=full, original=original, original_base=base)
    return f


def _actual_rendered_main_guard(tmp_path, fixture, monkeypatch, *, full=None, part2=None):
    generated = tmp_path / "rendered"
    write_runner_skeleton_onnxruntime(str(generated))
    tree = ast.parse((generated / "run_split_onnxruntime.py").read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    block = next(n for n in main.body if isinstance(n, ast.If)
                 and "generic_quality_completion_config" in ast.unparse(n.test))
    # Execute the actual complete selected-runtime identity/model guard section,
    # ending immediately before engine setup (which is outside this CPU test).
    stop = next(i for i, n in enumerate(block.body) if isinstance(n, ast.Expr)
                and isinstance(n.value, ast.Call)
                and getattr(n.value.func, "id", "") == "verify_quality_generated_model_sources")
    section = ast.Module(body=[ast.If(test=block.test, body=block.body[:stop+1], orelse=[])], type_ignores=[])
    ast.fix_missing_locations(section)
    package = ModuleType("splitpoint_runners"); package.__path__ = []
    monkeypatch.setitem(sys.modules, "splitpoint_runners", package)
    monkeypatch.setitem(sys.modules, "splitpoint_runners.generic_quality_completion", quality)
    artifacts = fixture["config"]["expected_runtime_artifacts"]
    namespace = {"Path": Path, "args": SimpleNamespace(
        generic_quality_completion_config=str(fixture["path"]), quality_source_only=True,
        native_split_quality_binding="", stage1_provider="hailo10h", stage2_provider="tensorrt",
        provider="", quality_evidence_setup_id="test_only_h10"),
        "_canonical_validation_backend_token": lambda token: token,
        "_decoder_model_id": "yolo26s", "base_dir": tmp_path / "b039",
        "full_path": full or fixture["full"],
        "p1_path": Path(artifacts["source_part1_onnx"]["path"]),
        "p2_path": part2 or Path(artifacts["source_part2_onnx"]["path"])}
    exec(compile(section, "actual_rendered_main_quality_guard", "exec"), namespace)
    return namespace["quality_completion"]


def test_actual_main_then_observer_finish_export_and_consumer_preserve_original(tmp_path, monkeypatch):
    f = _fixture(tmp_path)
    loaded = _actual_rendered_main_guard(tmp_path, f, monkeypatch)
    assert loaded["source_endpoint_contract"] == f["original"]
    assert loaded["completion_contract"] == f["original_base"]
    original_capture = helpers.capture

    def timed_capture(fixture):
        normal_class = quality.GenericQualityCompletion
        class Timed(normal_class):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.begin_capture()
        with monkeypatch.context() as patch:
            patch.setattr(quality, "GenericQualityCompletion", Timed)
            runtime, records = original_capture(fixture)
        runtime.finish_capture()
        return runtime, records

    monkeypatch.setattr(helpers, "capture", timed_capture)
    request, result, request_path, _ = helpers.envelope(f, tmp_path)
    evidence = request[quality.FIELD]
    assert "completion_contract" not in evidence
    assert evidence["base_completion_contract"] == f["original_base"]
    assert evidence["source_endpoint_contract"] == f["original"]
    assert evidence["capture_timing"]["image_count"] == 2
    verified = quality.validate_completed_quality_binding(request, result,
        f["original_base"], f["config"]["expected_runtime_artifacts"],
        request_path=request_path, reference_descriptor={"expected_image_ids": ["a.png", "b.png"]})
    assert verified["all_actual_completed_records_verified"] and verified["image_count"] == 2


@pytest.mark.parametrize("fault", ["full", "part2", "semantic", "proof_hash", "missing_proof", "extra_field_removal"])
def test_actual_main_rejects_wrong_producer_or_semantic_proof(tmp_path, monkeypatch, fault):
    f = _fixture(tmp_path)
    kwargs = {}
    if fault == "full":
        kwargs["full"] = Path(f["config"]["expected_runtime_artifacts"]["source_part2_onnx"]["path"])
    elif fault == "part2":
        kwargs["part2"] = Path(f["config"]["expected_runtime_artifacts"]["source_part1_onnx"]["path"])
    elif fault == "semantic":
        f["config"]["source_endpoint_contract"]["stage"] = "raw_head"
    elif fault == "proof_hash":
        f["config"]["source_endpoint_projection"]["projection_sha256"] = "0" * 64
    elif fault == "missing_proof":
        f["config"].pop("source_endpoint_projection"); f["config"].pop("source_endpoint_authority")
        f["config"]["identity"]["model_sha256"] = quality._sha(f["full"])
    else:
        f["config"]["source_endpoint_projection"]["projected_fields_removed"].append("tensor_signature")
    f["path"].write_text(json.dumps(f["config"]))
    with pytest.raises(ValueError):
        _actual_rendered_main_guard(tmp_path, f, monkeypatch, **kwargs)


@pytest.mark.parametrize("fault", ["semantic", "base", "full", "part2"])
def test_creation_rejects_other_semantics_or_wrong_role_bytes(tmp_path, fault):
    f = _fixture(tmp_path)
    semantic = deepcopy(f["original"]); base = deepcopy(f["original_base"])
    full = helpers.descriptor(f["full"])
    part2 = deepcopy(f["config"]["expected_runtime_artifacts"]["source_part2_onnx"])
    if fault == "semantic": semantic["endpoint_contract_complete"] = False
    elif fault == "base": base["source_endpoint"]["model_sha256"] = full["sha256"]
    elif fault == "full": full["sha256"] = "0" * 64
    else: part2["sha256"] = "0" * 64
    with pytest.raises((ValueError, RuntimeError)):
        quality._endpoint.project_generic_quality_endpoint(f["config"]["source_endpoint_authority"],
            full_model=full, part2_source=part2, original_semantic_endpoint=semantic,
            original_completion_contract=base)
