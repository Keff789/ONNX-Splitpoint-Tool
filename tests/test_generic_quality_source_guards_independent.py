"""Independent source/profile/receipt transport checks; no accelerator runs."""
import ast
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from onnx_splitpoint_tool.runners import generic_quality_completion
from tests.test_generic_quality_completion import projected_fixture
from tests.test_generic_quality_source_guards import (
    TEMPLATES, child_namespace, functions, selection, transport,
)


def test_explicit_generic_source_requires_profile_hashes(transport):
    suite, case, _, _, calls = transport
    select, run, options = selection(transport)
    run.pop("generic_quality_completion_config_sha256")
    with pytest.raises(ValueError, match="bound profile source bytes"):
        select(suite, run, case, **options)
    assert calls == []


def test_legacy_native_source_only_flag_keeps_ordinary_selection(transport):
    suite, case, _, _, calls = transport
    select, _, options = selection(transport)
    assert select(suite, {"quality_source_only": True}, case, **options) is None
    assert calls == []


def test_explicit_null_generic_mapping_is_rejected(transport):
    suite, case, _, _, calls = transport
    select, run, options = selection(transport)
    run["generic_quality_completion_configs"] = None
    with pytest.raises(ValueError, match="exact composed case mapping"):
        select(suite, run, case, **options)
    assert calls == []


def test_suite_rejects_config_symlink_escape_before_loading(transport, tmp_path):
    suite, case, path, _, calls = transport
    select, run, options = selection(transport)
    outside = tmp_path / "outside_config.json"
    outside.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(outside)
    with pytest.raises(ValueError, match="escapes its suite"):
        select(suite, run, case, **options)
    assert calls == []


def test_generic_child_forwards_exact_cpu_quality_identity(transport):
    _, case, _, _, _ = transport
    child, kwargs, commands, _ = child_namespace(transport)
    child(case, **kwargs)
    command = commands[0]
    for flag, source in (
        ("--quality-evidence-eval-id", "quality_evidence_eval_id"),
        ("--quality-evidence-model-id", "quality_evidence_model_id"),
        ("--quality-evidence-setup-id", "quality_evidence_setup_id"),
        ("--quality-evidence-source-run-id", "quality_evidence_source_run_id"),
    ):
        assert command.count(flag) == 1
        assert command[command.index(flag) + 1] == kwargs[source]


def _artifact_block():
    source = TEMPLATES / "run_split_onnxruntime.py.txt"
    module = ast.parse(source.read_text())
    main = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    matches = [node for node in main.body if isinstance(node, ast.If)
               and "generic_quality_completion_config" in ast.unparse(node.test)
               and "_verified_generic_only_trt_build_receipt" in ast.unparse(node)]
    assert len(matches) == 1
    return compile(ast.Module(body=matches, type_ignores=[]), str(source), "exec")


@pytest.fixture
def artifact_projection(tmp_path, monkeypatch):
    suite = tmp_path / "suite"
    suite.mkdir()
    fixture = projected_fixture(suite)
    config = fixture["config"]
    config["identity"]["comparison_backend"] = "hailo10h"
    config["original_run_cfg"] = {"native_trt_precision": "fp16", "native_trt_workspace_mb": 2048,
                                  "boundary_mode": "canonical_float32"}
    full = Path(fixture["context"]["source_model_path"])
    fixture["path"].write_text(json.dumps(config))
    monkeypatch.setitem(sys.modules, "splitpoint_runners.generic_quality_completion", generic_quality_completion)
    artifacts = config["expected_runtime_artifacts"]
    base = suite / "b039"
    args = SimpleNamespace(generic_quality_completion_config=str(fixture["path"]),
        stage1_provider="hailo10", stage2_provider="tensorrt", provider="hailo10",
        quality_source_only=True, native_split_quality_binding="", quality_evidence_setup_id="test_only_h10",
        native_trt_precision="fp32", native_trt_workspace_mb=4096, boundary_mode="different_default",
        native_trt_build=True, native_trt_no_fallback=False, trt_runtime="native_preferred")
    namespace = functions("run_split_onnxruntime.py.txt", ["_canonical_validation_backend_token"])
    calls = []
    def receipt(**options):
        calls.append(options)
        return {"status": "normal receipt verifier boundary reached"}
    namespace.update(args=args, base_dir=base, _decoder_model_id="yolo26s", full_path=full,
        p1_path=Path(artifacts["source_part1_onnx"]["path"]),
        p2_path=Path(artifacts["source_part2_onnx"]["path"]),
        _sp_file_sha256=lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        _verify_explicit_native_trt_engine_receipt=receipt)
    return namespace, config, calls, _artifact_block()


def test_quality_artifact_init_loads_real_files_and_exact_receipt_projection(artifact_projection):
    namespace, config, calls, code = artifact_projection
    exec(code, namespace)
    args = namespace["args"]
    artifacts = config["expected_runtime_artifacts"]
    assert calls == [{"source_onnx": Path(artifacts["build_part2_onnx"]["path"]),
        "engine_path": Path(artifacts["engine"]["path"]),
        "receipt_path": Path(artifacts["engine_build_receipt"]["path"]),
        "expected_precision": "fp16", "expected_workspace_mb": 2048}]
    assert args.native_trt_build is False and args.native_trt_no_fallback is True
    assert args.trt_runtime == args.trt_runtime_mode == "native"
    assert args.boundary_mode == "canonical_float32"
    assert args.native_split_quality_binding == ""
    assert set(args._generic_only_artifact_paths) == generic_quality_completion.ARTIFACT_ROLES


@pytest.mark.parametrize("role", ["p1_path", "p2_path", "full_path"])
def test_quality_artifact_init_rejects_other_actual_onnx_before_receipt(artifact_projection, tmp_path, role):
    namespace, _, calls, code = artifact_projection
    wrong = tmp_path / "other_actual.onnx"
    wrong.write_bytes(b"wrong actual source graph")
    namespace[role] = wrong
    with pytest.raises(ValueError, match="model metadata differs"):
        exec(code, namespace)
    assert calls == []


@pytest.mark.parametrize("role", ["engine", "part1_runtime"])
def test_quality_artifact_init_rejects_modified_compiled_bytes_before_receipt(artifact_projection, role):
    namespace, config, calls, code = artifact_projection
    Path(config["expected_runtime_artifacts"][role]["path"]).write_bytes(b"changed compiled fixture")
    with pytest.raises(ValueError, match="bytes changed"):
        exec(code, namespace)
    assert calls == []


def test_quality_artifact_init_cannot_swallow_normal_receipt_failure(artifact_projection):
    namespace, _, _, code = artifact_projection
    def rejected(**_):
        raise ValueError("normal TensorRT receipt rejected")
    namespace["_verify_explicit_native_trt_engine_receipt"] = rejected
    with pytest.raises(ValueError, match="receipt rejected"):
        exec(code, namespace)
