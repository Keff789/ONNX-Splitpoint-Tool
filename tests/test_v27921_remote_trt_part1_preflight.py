from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import onnx
from onnx import TensorProto, helper

from onnx_splitpoint_tool.benchmark import remote_run
from tests.test_v27920_remote_trt_cache_preflight import (
    _LocalReadOnlyTransport, _builder_abi, _builder_abi_sha256, _owner,
    _receipt, _sha, _suite,
)


def _fixture(tmp_path: Path, *, dynamic: bool = False, precision: str = "fp16"):
    suite = _suite(tmp_path / "input", [{
        "id": "trt_to_hailo8", "type": "matrix", "case_id": 24,
        "stage1": {"provider": "tensorrt"},
        "stage2": {"hw_arch": "hailo8"}, "variants": ["composed"],
    }])
    source = suite / "b024/model_part1_b24.onnx"
    shape = ["batch" if dynamic else 1, 3, 224, 224]
    graph = helper.make_graph(
        [helper.make_node("Identity", ["images"], ["activation"])], "part1",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info("activation", TensorProto.FLOAT, shape)],
    )
    onnx.save(helper.make_model(graph), source)
    builder = tmp_path / "trtexec"
    builder.write_bytes(b"test-builder-bytes-never-executed")
    abi = _builder_abi(builder)
    args = SimpleNamespace(add_args=f"--native-trt-precision {precision}")
    base = tmp_path / "remote"
    base.mkdir()
    key = remote_run._stable_trt_engine_cache_key(suite, args=args, builder_abi=abi,
                                                active_run_ids=["trt_to_hailo8"])
    namespace = base / "_onnx_splitpoint_cache/tensorrt_managed_v27516" / key
    leaf = namespace / remote_run._trt_persistent_engine_relative_dir(
        role="part1", case_id="b024", source_onnx_sha256=_sha(source), precision=precision,
    )
    return suite, source, builder, abi, args, base, namespace, leaf


def _probe(fixture):
    suite, _source, _builder, abi, args, base, _namespace, _leaf = fixture
    return remote_run.probe_remote_trt_artifact_cache(
        transport=_LocalReadOnlyTransport(base), suite_dir=suite,
        setup_id="orin_nx_hailo8_01", setup_accelerator="hailo8", args=args,
        active_run_ids=["trt_to_hailo8"], resolved_remote_base=str(base), builder_abi=abi,
    )


def test_reverse_direction_requires_part1_and_never_part2(tmp_path: Path):
    fixture = _fixture(tmp_path)
    suite = fixture[0]
    required = remote_run._trt_preflight_run_requirements(suite, active_run_ids=["trt_to_hailo8"])
    assert required["p1_cases"] == ["b024"]
    assert required["p1_run_ids_by_case"] == {"b024": ["trt_to_hailo8"]}
    assert required["p2_cases"] == [] and not required["full_required"]
    result = _probe(fixture)
    assert [(row["role"], row["status"], row["item_id"]) for row in result["observations"]] == [
        ("trt_p1", "MISS", "orin_nx_hailo8_01/b024"),
    ]
    assert result["hardware_action_performed"] is False


def test_exact_part1_receipt_uses_image_shape_and_keeps_special_precision_tag(tmp_path: Path):
    fixture = _fixture(tmp_path, dynamic=True, precision="uint8_cast_fp16")
    suite, source, builder, abi, _args, _base, namespace, leaf = fixture
    _receipt(leaf=leaf, source_bytes=source.read_bytes(), builder=builder,
             role="part1", shapes="images:1x3x224x224", engine_precision="uint8_cast_fp16")
    _owner(namespace, builder_abi_sha256=_builder_abi_sha256(abi))
    before = {str(path): path.read_bytes() for path in namespace.rglob("*") if path.is_file()}
    result = _probe(fixture)
    row = result["observations"][0]
    assert row["status"] == "HIT" and row["role"] == "trt_p1"
    assert row["artifact_path"].endswith("part1_uint8_cast_fp16.engine")
    assert row["evidence"]["source_binding"] == "direct_part1_source"
    assert row["evidence"]["shapes"] == "images:1x3x224x224"
    assert result["requirements"][0]["shape_contract"]["complete"] is True
    assert result["requirements"][0]["engine_precision"] == "uint8_cast_fp16"
    assert before == {str(path): path.read_bytes() for path in namespace.rglob("*") if path.is_file()}
    assert _sha(source) in remote_run._trt_preflight_source_shape_contracts(suite)


def test_part2_receipt_and_activation_shape_cannot_attest_part1(tmp_path: Path):
    fixture = _fixture(tmp_path, dynamic=True)
    _suite_dir, source, builder, abi, _args, _base, namespace, leaf = fixture
    wrong_role = namespace / remote_run._trt_persistent_engine_relative_dir(
        role="part2", case_id="b024", source_onnx_sha256=_sha(source), precision="fp16",
    )
    _receipt(leaf=wrong_role, source_bytes=source.read_bytes(), builder=builder, role="part2")
    _owner(namespace, builder_abi_sha256=_builder_abi_sha256(abi))
    result = _probe(fixture)
    assert result["observations"][0]["status"] == "MISS"
    _receipt(leaf=leaf, source_bytes=source.read_bytes(), builder=builder,
             role="part1", shapes="activation:1x64x56x56")
    result = _probe(fixture)
    assert result["observations"][0]["status"] == "MISS"
    assert result["observations"][0]["reason"] == "receipt_shape_mismatch"


def test_missing_part1_source_stays_unknown_and_never_uses_part2(tmp_path: Path):
    fixture = _fixture(tmp_path)
    fixture[1].unlink()
    result = _probe(fixture)
    assert result["observations"][0]["role"] == "trt_p1"
    assert result["observations"][0]["status"] == "UNKNOWN"
    assert result["observations"][0]["reason"] == "local_source_identity_unavailable"
    assert result["requirements"][0]["source_sha256"] == ""


def test_disabled_and_part2_only_reverse_rows_do_not_require_part1(tmp_path: Path):
    fixture = _fixture(tmp_path)
    suite = fixture[0]
    plan_path = suite / "benchmark_plan.json"
    plan = json.loads(plan_path.read_text())
    plan["runs"][0]["variants"] = ["part2"]
    plan_path.write_text(json.dumps(plan))
    assert remote_run._trt_preflight_run_requirements(suite)["p1_cases"] == []
    plan["runs"][0].update(variants=["composed"], enabled=False)
    plan_path.write_text(json.dumps(plan))
    assert remote_run._trt_preflight_run_requirements(suite)["p1_cases"] == []
