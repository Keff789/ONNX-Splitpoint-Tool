"""Actual MobileNet metadata replay; all model/engine payloads are synthetic.

Exercises the actual remote probe script locally without compiler, hardware,
CUDA or network. Historic H8/H10 geometry and receipt fields are preserved.
"""
from __future__ import annotations
import copy
import json
from pathlib import Path
import pytest
from onnx_splitpoint_tool.benchmark import remote_run
from onnx_splitpoint_tool.native_command_contract import canonical_json_sha256
from onnx_splitpoint_tool.native_split_quality import (
    known_native_split_policy, materialize_native_split_preselection,
    seal_native_split_quality_binding, validate_native_split_quality_binding,
)
from tests.test_v27920_remote_trt_cache_preflight import (
    _LocalReadOnlyTransport, _builder_abi, _builder_abi_sha256, _owner,
    _sha, _vendor_native_quality_validator,
)
PAIRS = json.loads((Path(__file__).parent / "fixtures" /
    "v2803_fix2_trt_mobilenet_pairs.json").read_text())["pairs"]


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))


def _artifact(path):
    return {"path": str(path.resolve()), "sha256": _sha(path),
            "size_bytes": path.stat().st_size}


def _replace(value, replacements):
    if isinstance(value, dict):
        return {k: _replace(v, replacements) for k, v in value.items()}
    if isinstance(value, list):
        return [_replace(v, replacements) for v in value]
    if isinstance(value, str):
        for old, new in replacements.items():
            value = value.replace(old, new)
    return value


def _pair_fixture(tmp_path, index):
    original = copy.deepcopy(PAIRS[index]["binding"])
    selection = original["preselection"]
    backend, setup = selection["backend"], selection["setup_id"]
    precision = selection["precision"]
    suite = tmp_path / "suite"
    (suite / "models").mkdir(parents=True)
    (suite / "models/model.onnx").write_bytes(b"synthetic-full-mobilenet")
    (suite / "b135").mkdir()
    raw_source = suite / "b135/mobilenet_v3_large_part2_b135.onnx"
    raw_source.write_bytes(b"synthetic-canonical-part2-identical-across-backends")
    _write(suite / "b135/split_manifest.json", {})
    _write(suite / "benchmark_set.json", {
        "model_id": "mobilenet_v3_large", "model": "models/model.onnx",
        "benchmark_task": "classification", "cases": [{"id": "b135"}],
    })
    run = {"id": backend, "type": "matrix", "case_id": "b135",
           "stage1": {"hw_arch": backend.removesuffix("_to_trt")},
           "stage2": {"provider": "tensorrt"}, "variants": ["composed"]}
    _write(suite / "benchmark_plan.json", {"runs": [run],
        "native_split_quality_selection": {
            "schema": "onnx-splitpoint/native-split-quality-selection",
            "schema_version": 1, "applicable": True,
            "split_backends": [backend.removesuffix("_to_trt")],
            "split_selection_source": "native_backends",
        },
    })
    _vendor_native_quality_validator(suite)
    builder = tmp_path / "bin/trtexec"
    builder.parent.mkdir()
    builder.write_bytes(b"#!/bin/sh\nexit 99\n")  # must never execute
    builder.chmod(0o755)
    abi = _builder_abi(builder)
    base = tmp_path / "remote"
    stable_key = remote_run._stable_trt_engine_cache_key(
        suite, builder_abi=abi, active_run_ids=[backend])
    namespace = base / "_onnx_splitpoint_cache/tensorrt_managed_v27516" / stable_key
    root = namespace / "native_split_quality" / setup / "mobilenet_v3_large/b135" / backend / "recorded-pair"
    leaf = root / "engine_cache/b135/part2" / precision
    paths, replacements = {}, {}
    for role, old in original["artifacts"].items():
        if role == "trtexec":
            paths[role] = builder
        elif role in {"part1_runtime", "boundary_metadata", "native_trt_meta"}:
            paths[role] = root / Path(old["path"]).name
        elif role == "source_part2_onnx":
            paths[role] = root / "benchmark_set/b135/source_part2.onnx"
        else:
            paths[role] = leaf / Path(old["path"]).name
        replacements[old["path"]] = str(paths[role].resolve())
        if role not in {"boundary_metadata", "native_trt_meta", "engine_build_receipt", "trtexec"}:
            paths[role].parent.mkdir(parents=True, exist_ok=True)
            paths[role].write_bytes(raw_source.read_bytes() if role == "source_part2_onnx"
                                   else ("synthetic-" + role + "-" + backend).encode())
        if paths[role].is_file():
            replacements[old["sha256"]] = _sha(paths[role])
    metadata = _replace(original["boundary_metadata_payload"], replacements)
    metadata.pop("metadata_sha256")
    metadata["part1_artifact_size_bytes"] = paths["part1_runtime"].stat().st_size
    metadata["metadata_sha256"] = canonical_json_sha256(metadata)
    _write(paths["boundary_metadata"], metadata)
    policy = known_native_split_policy(model_id="mobilenet_v3_large", case_id="b135",
                                      setup_id=setup, backend=backend)
    selected = materialize_native_split_preselection(
        policy=policy, part1_artifact=_artifact(paths["part1_runtime"]),
        boundary_metadata=metadata, boundary_metadata_artifact=_artifact(paths["boundary_metadata"]))
    receipt = _replace(original["engine_build_receipt"], replacements)
    receipt.pop("receipt_sha256")
    receipt["receipt_sha256"] = canonical_json_sha256(receipt)
    _write(paths["engine_build_receipt"], receipt)
    meta = _replace(original["native_trt_meta_payload"], replacements)
    meta["engine_build_receipt"] = receipt
    meta["build"]["cmd"] = receipt["command"]
    _write(paths["native_trt_meta"], meta)
    boundary = {key: selected[key] for key in original["boundary_contract"]}
    payload = {
        "eval_run_id": "synthetic-metadata-replay", "source_run_id": backend,
        "quality_completed": True, "performance_claims_emitted": False,
        "preselection": selected, "preselection_sha256": selected["selection_sha256"],
        "artifacts": {role: _artifact(path) for role, path in paths.items()},
        "boundary_contract": boundary, "boundary_contract_sha256": canonical_json_sha256(boundary),
        "engine_build_receipt": receipt, "engine_build_receipt_sha256": receipt["receipt_sha256"],
        "native_trt_meta": meta, "native_trt_meta_sha256": canonical_json_sha256(meta),
        "producer_command": ["python", "native_trt_from_benchmarkset.py"],
    }
    selected_part1 = suite / "b135/hailo" / backend.removesuffix("_to_trt") / "part1/compiled.hef"
    selected_part1.parent.mkdir(parents=True)
    selected_part1.write_bytes(paths["part1_runtime"].read_bytes())
    binding = seal_native_split_quality_binding(payload)
    verified, status = validate_native_split_quality_binding(binding, verification_mode="local")
    assert verified, status
    binding_path = root / "native_split_quality_binding.json"
    _write(binding_path, binding)
    _owner(namespace, builder_abi_sha256=_builder_abi_sha256(abi))
    return dict(suite=suite, base=base, abi=abi, setup=setup, backend=backend,
                precision=precision, paths=paths, binding_path=binding_path,
                binding=binding, source=raw_source, selected_part1=selected_part1)


def _probe(fixture):
    return remote_run.probe_remote_trt_artifact_cache(
        transport=_LocalReadOnlyTransport(fixture["base"]), suite_dir=fixture["suite"],
        setup_id=fixture["setup"], setup_accelerator=fixture["backend"].removesuffix("_to_trt"),
        active_run_ids=[fixture["backend"]], resolved_remote_base=str(fixture["base"]),
        builder_abi=fixture["abi"],
    )


@pytest.mark.parametrize("index", [0, 1], ids=["hailo8", "hailo10"])
def test_recorded_mobilenet_pair_projects_actual_bound_bridge_and_hits(tmp_path, index):
    fixture = _pair_fixture(tmp_path, index)
    before = {p: p.read_bytes() for p in fixture["base"].rglob("*") if p.is_file()}
    result = _probe(fixture)
    requirement = result["requirements"][0]
    assert result["runtime_contract"]["native_trt_precision"] == "fp16"
    assert requirement["engine_precision"] == fixture["precision"]
    assert requirement["recipe_source"] == "native_split_quality_policy"
    row = result["observations"][0]
    assert row["status"] == "HIT", row
    assert row["receipt_path"] == str(fixture["paths"]["engine_build_receipt"])
    assert row["evidence"]["source_binding"] == "strict_native_split_quality_binding"
    assert before == {p: p.read_bytes() for p in fixture["base"].rglob("*") if p.is_file()}


@pytest.mark.parametrize("index", [0, 1], ids=["hailo8", "hailo10"])
@pytest.mark.parametrize("damage", ["source", "bridge", "unbound", "identity", "part1", "part1_missing"])
def test_recorded_pair_rejects_changed_source_bridge_or_missing_binding(tmp_path, index, damage):
    fixture = _pair_fixture(tmp_path, index)
    if damage == "source":
        fixture["source"].write_bytes(b"different-canonical-source")
    elif damage == "bridge":
        fixture["paths"]["build_part2_onnx"].write_bytes(b"changed-bridge")
    elif damage == "unbound":
        fixture["binding_path"].unlink()
    elif damage == "part1":
        fixture["selected_part1"].write_bytes(b"different-current-hef")
    elif damage == "part1_missing":
        fixture["selected_part1"].unlink()
    else:
        fixture["binding"]["preselection"]["setup_id"] = "other-setup"
        _write(fixture["binding_path"], fixture["binding"])
    result = _probe(fixture)
    assert result["observations"][0]["status"] != "HIT", result


@pytest.mark.parametrize("selection", ["native_disabled", "other_backend", "part2_only"])
def test_generic_rows_keep_generic_recipe(tmp_path, selection):
    fixture = _pair_fixture(tmp_path, 0)
    plan_path = fixture["suite"] / "benchmark_plan.json"
    plan = json.loads(plan_path.read_text())
    if selection == "part2_only":
        plan["runs"][0]["variants"] = ["part2"]
    else:
        plan["native_split_quality_selection"]["split_backends"] = [] if selection == "native_disabled" else ["hailo10h"]
        plan["native_split_quality_selection"]["applicable"] = selection != "native_disabled"
        plan["native_split_quality_selection"]["split_selection_source"] = selection
    _write(plan_path, plan)
    result = _probe(fixture)
    assert result["requirements"][0]["engine_precision"] == "fp16"
    assert result["observations"][0]["status"] != "HIT"


def test_legacy_composed_plan_without_selection_uses_runtime_policy(tmp_path):
    fixture = _pair_fixture(tmp_path, 1)
    plan_path = fixture["suite"] / "benchmark_plan.json"
    plan = json.loads(plan_path.read_text())
    plan.pop("native_split_quality_selection")
    _write(plan_path, plan)
    result = _probe(fixture)
    assert result["requirements"][0]["engine_precision"] == "uint8_dequant_fp16"
    assert result["observations"][0]["status"] == "HIT"


@pytest.mark.parametrize("invalid", [None, {"schema_version": 1}])
def test_invalid_native_selection_blocks_probe_without_transport(tmp_path, invalid):
    fixture = _pair_fixture(tmp_path, 0)
    plan_path = fixture["suite"] / "benchmark_plan.json"
    plan = json.loads(plan_path.read_text())
    plan["native_split_quality_selection"] = invalid
    _write(plan_path, plan)
    transport = _LocalReadOnlyTransport(fixture["base"])
    result = remote_run.probe_remote_trt_artifact_cache(
        transport=transport, suite_dir=fixture["suite"], setup_id=fixture["setup"],
        active_run_ids=[fixture["backend"]], builder_abi=fixture["abi"],
        resolved_remote_base=str(fixture["base"]),
    )
    assert result["status"] == "unknown"
    assert result["observations"][0]["status"] == "UNKNOWN"
    assert not transport.commands
