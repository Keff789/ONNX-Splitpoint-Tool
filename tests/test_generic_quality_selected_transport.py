"""Selected Source transport over fresh full-catalogue fixtures only."""
import builtins
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile

import onnx
from onnx import TensorProto, helper as onnx_helper
import pytest

from onnx_splitpoint_tool.benchmark import remote_run
from onnx_splitpoint_tool.benchmark.generic_quality_transport import selected_source_transport_excludes
from onnx_splitpoint_tool.remote.bundle import build_suite_bundle, remote_minimal_bundle_patterns


MODEL = "own_model"
SETUP = "own_h10_setup"
RUN = "own_generic_quality"
BINARY_SUFFIXES = ("onnx", "hef", "dxnn", "engine", "plan", "data", "bin")


def write(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    return {"path": str(path), "sha256": hashlib.sha256(body).hexdigest(), "size_bytes": len(body)}


def json_file(path, value):
    return write(path, (json.dumps(value, sort_keys=True) + "\n").encode())


def case_binary(case, suffix):
    return case + ("/hailo/part1.hef" if suffix == "hef" else "/part1." + suffix)


def tiny_inline_model():
    tensor = onnx_helper.make_tensor("own_weight", TensorProto.FLOAT, [1], [1.0])
    graph = onnx_helper.make_graph(
        [onnx_helper.make_node("Add", ["x", "own_weight"], ["y"])],
        "OWN_FIXTURE_ONLY",
        [onnx_helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
        [onnx_helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
        initializer=[tensor],
    )
    return onnx_helper.make_model(graph, producer_name="OWN_FIXTURE_ONLY")


def fixture(tmp_path, *, images=False):
    suite = tmp_path / "suite"
    suite.mkdir()
    cases = [f"b{number:03}" for number in range(1, 21)]
    inline = tiny_inline_model().SerializeToString()
    for case in cases:
        for suffix in BINARY_SUFFIXES:
            body = inline if case == "b001" and suffix == "onnx" else ("OWN " + case + " " + suffix).encode()
            write(suite / case_binary(case, suffix), body)
        write(suite / case / "part2.onnx", inline if case == "b001" else ("OWN part2 " + case).encode())
        json_file(suite / case / "split_manifest.json", {"OWN_FIXTURE_ONLY": True, "case_id": case})
        json_file(suite / case / "gate.json", {"OWN_FIXTURE_ONLY": True})
        write(suite / case / "metadata.bin.json", b'{"OWN_metadata":true}')
    runtime = write(suite / "splitpoint_runners/own_completion.py", b"# OWN fixture runtime metadata\n")
    full = write(suite / "models/model.onnx", inline)
    dataset = suite / "resources/validation/detection/val2017_n5000_s20260710"
    ids = [f"{number:012}.jpg" for number in range(1, 5001)]
    json_file(dataset / "manifest.json", {"OWN_FIXTURE_ONLY": True, "image_ids": ids, "image_count": 5000})
    if images:
        for image in ids:
            write(dataset / "images" / image, ("OWN image " + image).encode())
    assets = {
        "part1_runtime": write(suite / "quality_runtime/assets/b001/part1_runtime/selected.hef", b"OWN selected HEF"),
        "engine": write(suite / "quality_runtime/assets/b001/engine/selected.engine", b"OWN selected engine"),
    }
    for role, name in (("source_part1_onnx", "part1.onnx"), ("source_part2_onnx", "part2.onnx")):
        path = suite / "b001" / name
        body = path.read_bytes()
        assets[role] = {"path": str(path), "sha256": hashlib.sha256(body).hexdigest(), "size_bytes": len(body)}
    for value in assets.values():
        value["path"] = Path(value["path"]).relative_to(suite).as_posix()
    config = {
        "schema": "onnx-splitpoint/generic-composed-quality-completion-config", "schema_version": 1,
        "path_root": "suite_root",
        "identity": {"model_id": MODEL, "case_id": "b001", "setup_id": SETUP, "comparison_backend": "hailo10h", "precision": "float32_layout_fp16"},
        "expected_runtime_artifacts": assets,
        "runtime_source_files": [{**runtime, "path": "splitpoint_runners/own_completion.py"}],
    }
    run = {
        "id": RUN, "type": "matrix", "variants": ["composed"], "quality_source_only": True,
        "native_trt_build": False, "hardware_setup_id": SETUP, "case_ids": ["b001"],
        "stage1": {"type": "hailo", "hw_arch": "hailo10"},
        "stage2": {"type": "onnxruntime", "provider": "tensorrt"},
        "validation_images": dataset.relative_to(suite).as_posix(), "validation_max_images": 5000,
        "generic_quality_completion_configs": {"b001": "quality_runtime/b001.json"},
    }
    passive = {"id": "own_cpu_reference", "execution_location": "central_management", "semantic_reference_only": True, "variants": ["full"]}
    plan = {"model_id": MODEL, "runs": [run, passive]}
    json_file(suite / "benchmark_set.json", {"model_id": MODEL, "model_name": MODEL, "model": "models/model.onnx", "benchmark_task": "detection", "cases": [{"id": case, "case_dir": case, "folder": case} for case in cases]})
    state = {"suite": suite, "cases": cases, "config": config, "run": run, "plan": plan, "dataset": dataset, "ids": ids, "full": full, "origin": tmp_path / "own_source_config.json"}
    republish(state)
    return state


def republish(state):
    descriptor = json_file(state["suite"] / "quality_runtime/b001.json", state["config"])
    state["run"]["generic_quality_completion_config_sha256"] = {"b001": descriptor["sha256"]}
    origin = json_file(state["origin"], state["config"])
    assert origin["sha256"] == descriptor["sha256"]
    state["run"]["generic_quality_completion_sources"] = {"b001": origin}
    json_file(state["suite"] / "benchmark_plan.json", state["plan"])


def excludes(state, active=(RUN,)):
    return selected_source_transport_excludes(state["suite"], active_run_ids=active, model_id=MODEL)


def archive(state, path, transport_excludes=()):
    includes, defaults = remote_minimal_bundle_patterns()
    includes.append(state["dataset"].relative_to(state["suite"]).as_posix() + "/**")
    build_suite_bundle(state["suite"], path, includes=includes, excludes=tuple(defaults) + tuple(transport_excludes))
    with tarfile.open(path) as handle:
        return {member.name.removeprefix("./"): handle.extractfile(member).read() for member in handle.getmembers() if member.isfile()}


def test_selected_transport_keeps_full_catalogue_metadata_and_all_shared_5000(tmp_path):
    state = fixture(tmp_path, images=True)
    suite = state["suite"]
    before = {path.relative_to(suite).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in suite.rglob("*") if path.is_file()}
    selected_excludes = excludes(state)
    assert type(selected_excludes) is tuple and selected_excludes
    baseline = archive(state, tmp_path / "baseline.tar.gz")
    members = archive(state, tmp_path / "selected.tar.gz", selected_excludes)
    for case in state["cases"]:
        for name in ("split_manifest.json",):
            relative = case + "/" + name
            assert hashlib.sha256(members[relative]).hexdigest() == before[relative]
        for suffix in BINARY_SUFFIXES:
            relative = case_binary(case, suffix)
            assert (relative in members) is (case == "b001")
        assert (case + "/part2.onnx" in members) is (case == "b001")
    removed = {case_binary(case, suffix) for case in state["cases"][1:] for suffix in BINARY_SUFFIXES}
    removed.update(case + "/part2.onnx" for case in state["cases"][1:])
    assert set(baseline) - set(members) == removed
    assert not (set(members) - set(baseline))
    for name in members:
        assert hashlib.sha256(members[name]).hexdigest() == before[name]
    assert len([name for name in members if "/images/" in name]) == 5000
    assert len(json.loads(members["benchmark_set.json"])["cases"]) == 20
    assert json.loads(members["benchmark_plan.json"])["runs"][0]["case_ids"] == ["b001"]
    assert {path.relative_to(suite).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in suite.rglob("*") if path.is_file()} == before


@pytest.mark.parametrize("scope", ["empty", "unknown", "mixed", "other_executable_worker"])
def test_default_unknown_and_mixed_active_scope_keep_transport_unchanged(tmp_path, scope):
    state = fixture(tmp_path)
    active = () if scope == "empty" else ("not_an_existing_run",)
    if scope == "mixed":
        state["plan"]["runs"].append({"id": "own_native", "variants": ["full"], "case_ids": ["b002"]})
        republish(state)
        active = (RUN, "own_native")
    elif scope == "other_executable_worker":
        state["plan"]["runs"].append({"id": "own_other_setup", "quality_source_only": True, "hardware_setup_id": "own_other_setup"})
        republish(state)
        active = (RUN,)
    assert excludes(state, active) == ()


@pytest.mark.parametrize("fault", ["missing_config", "hash", "foreign_identity", "unknown_case", "escape", "missing_dependency"])
def test_explicit_source_transport_rejects_unproven_selection(tmp_path, fault):
    state = fixture(tmp_path)
    suite = state["suite"]
    if fault == "missing_config":
        (suite / "quality_runtime/b001.json").unlink()
    elif fault == "hash":
        state["run"]["generic_quality_completion_config_sha256"]["b001"] = "0" * 64
        json_file(suite / "benchmark_plan.json", state["plan"])
    elif fault == "foreign_identity":
        state["config"]["identity"]["setup_id"] = "own_foreign_setup"
        republish(state)
    elif fault == "unknown_case":
        state["run"]["case_ids"] = ["b999"]
        state["run"]["generic_quality_completion_configs"] = {"b999": "quality_runtime/b999.json"}
        state["run"]["generic_quality_completion_config_sha256"] = {"b999": state["run"]["generic_quality_completion_config_sha256"]["b001"]}
        state["run"]["generic_quality_completion_sources"] = {"b999": state["run"]["generic_quality_completion_sources"]["b001"]}
        json_file(suite / "benchmark_plan.json", state["plan"])
    elif fault == "escape":
        write(tmp_path / "own_escape.onnx", b"OWN outside suite")
        state["config"]["expected_runtime_artifacts"]["source_part1_onnx"]["path"] = "../own_escape.onnx"
        republish(state)
    else:
        (suite / "b001/part2.onnx").unlink()
    with pytest.raises((ValueError, OSError)):
        excludes(state)


def test_identical_excludes_drive_archive_and_capacity_without_reserve_changes(tmp_path, monkeypatch):
    state = fixture(tmp_path)
    suite = state["suite"]
    reserve = 7 * 1024**3
    monkeypatch.setattr(remote_run, "_remote_storage_reserve_bytes", lambda: reserve)
    args = remote_run.RemoteBenchmarkArgs(add_args="--run-id " + RUN, transfer_mode="bundle")
    selected_excludes = excludes(state)
    assert remote_run._source_transport_excludes_for_args(suite, args) == selected_excludes
    full_members = archive(state, tmp_path / "full.tar.gz")
    selected_members = archive(state, tmp_path / "selected.tar.gz", selected_excludes)
    removed = set(full_members) - set(selected_members)
    removed_bytes = sum(len(full_members[name]) for name in removed)
    complete = remote_run._suite_storage_footprint(suite)
    selected = remote_run._suite_storage_footprint(suite, transport_excludes=selected_excludes)
    assert complete["total_bytes"] - selected["total_bytes"] == removed_bytes
    assert complete["file_count"] - selected["file_count"] == len(removed)
    assert complete["directory_count"] == selected["directory_count"]
    automatic = remote_run._remote_run_capacity_requirement_for_args(suite, args=args)
    explicit = remote_run._remote_run_capacity_requirement_for_args(suite, args=args, transport_excludes=selected_excludes)
    unchanged = remote_run._remote_run_capacity_requirement_for_args(suite, args=args, transport_excludes=())
    assert automatic == explicit
    assert automatic["total_bytes"] == selected["total_bytes"]
    assert automatic["transport_peak_bytes"] == 2 * selected["total_bytes"]
    assert automatic["safety_reserve_bytes"] == unchanged["safety_reserve_bytes"] == reserve
    for field in ("case_count", "plan_run_count", "work_item_count", "runtime_output_log_bytes", "trt_engine_cache_bytes", "trt_build_scratch_bytes"):
        assert automatic[field] == unchanged[field]
    for field in ("warm_required_free_bytes", "cold_required_free_bytes", "required_free_bytes"):
        assert unchanged[field] - automatic[field] == 2 * removed_bytes
    for field in ("warm_required_free_inodes", "cold_required_free_inodes", "required_free_inodes"):
        assert unchanged[field] - automatic[field] == 2 * len(removed)


def test_direct_and_default_args_retain_full_transport_and_capacity(tmp_path):
    state = fixture(tmp_path)
    suite = state["suite"]
    direct = remote_run.RemoteBenchmarkArgs(add_args="--run-id " + RUN, transfer_mode="direct")
    assert remote_run._source_transport_excludes_for_args(suite, direct) == ()
    assert remote_run._source_transport_excludes_for_args(suite, None) == ()
    assert remote_run._source_transport_excludes_for_args(suite, remote_run.RemoteBenchmarkArgs()) == ()
    footprint = remote_run._suite_storage_footprint(suite)
    for args in (None, direct):
        automatic = remote_run._remote_run_capacity_requirement_for_args(suite, args=args)
        explicit = remote_run._remote_run_capacity_requirement_for_args(suite, args=args, transport_excludes=())
        assert automatic == explicit
        assert automatic["total_bytes"] == footprint["total_bytes"]


def test_typed_normal_and_legacy_stage_forms_select_identical_transport(tmp_path):
    state = fixture(tmp_path)
    typed = excludes(state)
    state["run"]["stage1"] = "hailo10"
    state["run"]["stage2"] = "tensorrt"
    republish(state)
    assert excludes(state) == typed
    assert typed


def test_shared_external_tensor_retains_full_transport_without_reading_blob(tmp_path, monkeypatch):
    state = fixture(tmp_path)
    suite = state["suite"]
    model = tiny_inline_model()
    tensor = model.graph.initializer[0]
    tensor.ClearField("float_data")
    tensor.data_location = TensorProto.EXTERNAL
    tensor.external_data.add(key="location", value="../b002/part1.data")
    write(suite / "models/model.onnx", model.SerializeToString())
    blob = suite / "b002/part1.data"
    attempted_blob_reads = []

    def guarded(real):
        def call(path, *args, **kwargs):
            if isinstance(path, (str, bytes, os.PathLike)) and os.path.abspath(os.fsdecode(path)) == str(blob):
                attempted_blob_reads.append(os.fsdecode(path))
                raise AssertionError("external blob must never be opened")
            return real(path, *args, **kwargs)
        return call

    for module in (builtins, io, os):
        monkeypatch.setattr(module, "open", guarded(module.open))
    logs = []
    assert selected_source_transport_excludes(suite, active_run_ids=(RUN,), model_id=MODEL, log=logs.append) == ()
    assert any("retaining full suite: external_or_unknown_tensor_storage:models/model.onnx" in row for row in logs)
    args = remote_run.RemoteBenchmarkArgs(add_args="--run-id " + RUN, transfer_mode="bundle")
    automatic = remote_run._remote_run_capacity_requirement_for_args(suite, args=args)
    unchanged = remote_run._remote_run_capacity_requirement_for_args(suite, args=args, transport_excludes=())
    assert automatic == unchanged
    assert automatic["total_bytes"] == remote_run._suite_storage_footprint(suite)["total_bytes"]
    assert attempted_blob_reads == []
