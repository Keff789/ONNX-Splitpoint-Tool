"""Actual package writer -> extracted suite -> real dataset consumer, no devices."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import threading

import pytest
import onnx
from onnx import TensorProto, helper

from onnx_splitpoint_tool.benchmark import remote_run
from onnx_splitpoint_tool.gui.controller import write_benchmark_suite_script
from onnx_splitpoint_tool.remote.bundle import build_suite_bundle, remote_minimal_bundle_patterns
from onnx_splitpoint_tool.remote.process_lease import RemoteProcessLeaseRegistry
from onnx_splitpoint_tool.remote.ssh_transport import HostConfig
from onnx_splitpoint_tool.split_export_runners import write_runner_skeleton_onnxruntime
from onnx_splitpoint_tool.workflow.contracts import WorkflowOptions
from onnx_splitpoint_tool.workflow.execution_binding import _remote_args_from_options


def load_module(path, name, monkeypatch):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def prepared_dataset(suite, count=5000):
    root = suite / "resources/validation/detection/val2017_n5000_s20260710"
    root.mkdir(parents=True)
    # Admission reads identity and file sizes, never decodes an image.
    seed = root / "000000000000.jpg"
    seed.write_bytes(b"no inference: controlled image bytes")
    ann = seed.with_suffix(".json")
    ann.write_text("[]")
    samples = []
    for index in range(count):
        name = f"{index:012d}.jpg"
        if index:
            os.link(seed, root / name)
            os.link(ann, (root / name).with_suffix(".json"))
        samples.append({"image": name, "image_id": index, "sample_id": str(index),
                        "source_size_bytes": seed.stat().st_size})
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({"schema": "onnx-splitpoint/detection-validation-manifest",
        "schema_version": 2, "samples": samples, "filtered_coco_annotations": "instances_subset.json"}))
    (root / "instances_subset.json").write_text(json.dumps({"images": [
        {"id": row["image_id"], "file_name": row["image"]} for row in samples], "annotations": [], "categories": []}))
    reference = {"quality_contract": {"dataset": {
        "image_count": count, "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()}}}
    return root, reference


def profile_for(run_id, backend):
    return {"run_profiles": [{"id": run_id, "type": "mixed_backend", "stage1": backend,
        "stage2": "tensorrt", "variants": ["composed"]}],
        "validation_execution": {"max_items": {"detection": 5000}}, "remote_benchmark": {}}


class AtRemoteDispatch(BaseException):
    pass


def controlled_transport(monkeypatch):
    class Transport:
        uploads = []
        def __init__(self, host, **kwargs):
            self.host = host
        def resolve_path_read_only(self, *args, **kwargs):
            return "/controlled-new-attempt"
        def run_read_only(self, *args, **kwargs):
            return 0, ""
        def run(self, *args, **kwargs):
            return 0, ""
        def scp_upload(self, *args, **kwargs):
            self.uploads.append(args)
            return 0, ""
        def run_streaming(self, command, **kwargs):
            raise AtRemoteDispatch(command)
    monkeypatch.setattr(remote_run, "SSHTransport", Transport)
    monkeypatch.setattr(remote_run, "_remote_trt_builder_abi", lambda *a, **k:
                        {"schema": "software-fixture-no-device", "tensorrt_version": "fixture"})
    monkeypatch.setattr(remote_run, "_remote_storage_preflight", lambda *a, **k:
                        {"ok": True, "free_bytes": 10**12, "free_inodes": 10**6})
    def no_refresh(*args, **kwargs):
        pytest.fail("The dispatch must consume the finalized attempt suite without rewriting it")
    monkeypatch.setattr(remote_run, "refresh_suite_harness", no_refresh)
    return Transport


@pytest.mark.parametrize("model,case_id,run_id,backend,setup", [
    ("yolo11l", "b120", "hailo8_to_trt", "hailo8", "orin_nx_hailo8_01"),
    ("yolo26s", "b038", "hailo10_to_tensorrt", "hailo10", "orin_nx_hailo10_01"),
    ("yolov7_paper", "b044", "deepx_m1_to_tensorrt", "deepx_m1", "orin_nx_deepx_m1_01"),
])
def test_failed_bundle_reentry_keeps_original_and_transports_bound_n5000(
        tmp_path, monkeypatch, model, case_id, run_id, backend, setup):
    original = tmp_path / "failed_attempt/suite"
    case = original / case_id
    case.mkdir(parents=True)
    full = original / "models" / (model + ".onnx")
    full.parent.mkdir()
    graph = helper.make_graph([helper.make_node("Identity", ["images"], ["output"])], model,
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 4, 4])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 4, 4])])
    onnx.save(helper.make_model(graph), full)
    (case / "split_manifest.json").write_text("{}")
    write_runner_skeleton_onnxruntime(str(case), target="cpu")
    write_benchmark_suite_script(original)
    dataset, reference = prepared_dataset(original)
    relative_source = dataset.relative_to(original).as_posix()
    plan = {"runs": [{"id": run_id, "type": "mixed_backend", "stage1": backend,
        "stage2": "tensorrt", "variants": ["composed"], "benchmark_task": "detection",
        "validation_images": relative_source, "validation_max_images": 5000,
        "validation_items_requested": 5000, "validation_budget_authoritative": True}]}
    (original / "benchmark_plan.json").write_text(json.dumps(plan))
    (original / "benchmark_set.json").write_text(json.dumps({"model_name": model, "cases": [], "plan": plan}))
    includes, excludes = remote_minimal_bundle_patterns()
    includes = [value for value in includes if value != "resources/validation/**"]
    bundle = original / "dist/suite_bundle.tar.gz"
    build_suite_bundle(suite_dir=original, out_path=bundle, includes=includes, excludes=excludes)
    with tarfile.open(bundle) as archive:
        assert not any("resources/validation/" in item.name for item in archive.getmembers())
    original_bytes = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in [
        bundle, bundle.with_name(bundle.name + ".manifest.json"), original / "benchmark_plan.json",
        original / "benchmark_suite.py", case / "run_split_onnxruntime.py"]}
    # A separate normal attempt receives the existing model/data bytes and the
    # old cached bundle. The real writer must notice its missing exact root.
    attempt = tmp_path / "new_attempt/suite"
    shutil.copytree(original, attempt)
    args = _remote_args_from_options(WorkflowOptions(profile="fixture", out=str(tmp_path)),
                                    profile_for(run_id, backend), model_task="detection", model_id=model)
    assert args.validation_budget_authoritative is True and args.validation_max_images == 5000
    args.runtime_suite_finalized = True
    args.resume = False
    args.add_args = "--run-ids " + run_id
    controlled_transport(monkeypatch)
    with pytest.raises(AtRemoteDispatch):
        remote_run.run_remote_benchmark(
            host=HostConfig(id=setup, label=setup, host="127.0.0.1", user="fixture"),
            benchmark_set_json=attempt / "benchmark_set.json", local_working_dir=tmp_path / "work",
            run_id="repair-" + model, args=args, log=lambda *a: None, progress=lambda *a: None,
            cancel_event=threading.Event(), remote_process_registry=RemoteProcessLeaseRegistry())
    remote_root = tmp_path / "controlled_remote/suite"
    with tarfile.open(attempt / "dist/suite_bundle.tar.gz") as archive:
        archive.extractall(remote_root, filter="data")
    remote_case = remote_root / case_id
    suite_module = load_module(remote_root / "benchmark_suite.py", "dataset_suite_" + model, monkeypatch)
    runner_module = load_module(remote_case / "run_split_onnxruntime.py", "dataset_runner_" + model, monkeypatch)
    monkeypatch.setattr(suite_module, "_provider_unavailable_reason", lambda *a, **k: None)
    observed = []
    class AtInference(BaseException):
        pass
    def inference_leaf(command, **kwargs):
        source = command[command.index("--validation-images") + 1]
        count = int(command[command.index("--validation-max-images") + 1])
        images = runner_module._collect_validation_image_paths(source, remote_case, count)
        assert len(images) == len({p.name for p in images}) == 5000
        assert {p.name for p in images} == {f"{index:012d}.jpg" for index in range(5000)}
        assert all(p.with_suffix(".json").read_text() == "[]" for p in images)
        observed.append(command)
        raise AtInference()
    monkeypatch.setattr(suite_module.subprocess, "run", inference_leaf)
    with pytest.raises(AtInference):
        suite_module._run_case(remote_case, run_id, "tensorrt", "default", "balanced", "auto",
            3, 5, 30, stage1=backend, stage2="tensorrt", variants=["composed"],
            validation_images=relative_source, validation_max_images=5000, benchmark_task="detection",
            management_cpu_reference=reference, quality_evidence_eval_id="repair",
            quality_evidence_model_id=model, quality_evidence_setup_id=setup,
            quality_evidence_source_run_id=run_id, native_trt_build=False)
    assert len(observed) == 1
    assert {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in original_bytes} == original_bytes


@pytest.mark.parametrize("damage,reason", [
    ("missing", "source_missing"), ("only50", "sample_count_mismatch"),
    ("foreign_reference", "manifest_mismatch"), ("wrong_budget", "count_invalid"),
    ("missing_image", "sample_missing_or_invalid"), ("missing_annotation", "sample_missing_or_invalid"),
])
def test_bound_dataset_errors_stop_before_any_child(tmp_path, monkeypatch, damage, reason):
    case = tmp_path / "suite/b120"
    case.mkdir(parents=True)
    count = 5000 if damage == "only50" else 60
    dataset, reference = prepared_dataset(case.parent, count=count)
    suite_path = Path(write_benchmark_suite_script(case.parent))
    suite = load_module(suite_path, "dataset_negative_" + damage, monkeypatch)
    source = str(dataset)
    if damage == "missing":
        source += "_absent"
    elif damage == "only50":
        payload = json.loads((dataset / "manifest.json").read_text())
        payload["samples"] = payload["samples"][:50]
        (dataset / "manifest.json").write_text(json.dumps(payload))
        # Even a self-consistent smaller manifest cannot satisfy N5000.
        reference["quality_contract"]["dataset"]["manifest_sha256"] = hashlib.sha256(
            (dataset / "manifest.json").read_bytes()).hexdigest()
    elif damage == "foreign_reference":
        reference["quality_contract"]["dataset"]["manifest_sha256"] = "0" * 64
    elif damage == "wrong_budget":
        count = 50
    elif damage == "missing_image":
        (dataset / "000000000059.jpg").unlink()
    else:
        (dataset / "000000000059.json").unlink()
    monkeypatch.setattr(suite.subprocess, "run", lambda *a, **k: pytest.fail("no child may start"))
    with pytest.raises(RuntimeError, match=reason):
        suite._run_case(case, "hailo8_to_trt", "tensorrt", "default", "balanced", "auto", 3, 5, 30,
            stage1="hailo8", stage2="tensorrt", variants=["composed"],
            validation_images=source, validation_max_images=count, benchmark_task="detection",
            management_cpu_reference=reference)


def test_other_profiles_do_not_gain_an_authoritative_budget():
    profile = profile_for("hailo8_to_trt", "hailo8")
    profile["run_profiles"][0]["variants"] = ["full", "composed"]
    args = _remote_args_from_options(WorkflowOptions(profile="fixture", out="unused"), profile,
                                    model_task="detection")
    assert args.validation_budget_authoritative is False


def test_missing_prepared_source_stops_normal_package_before_upload(tmp_path, monkeypatch):
    suite = tmp_path / "suite"
    suite.mkdir()
    plan = {"runs": [{"id": "ort_cpu", "type": "onnxruntime", "provider": "cpu",
        "benchmark_task": "detection", "validation_images": "resources/validation/detection/missing",
        "validation_max_images": 5000, "validation_budget_authoritative": True}]}
    (suite / "benchmark_plan.json").write_text(json.dumps(plan))
    (suite / "benchmark_set.json").write_text(json.dumps({"model_name": "fixture", "cases": [], "plan": plan}))
    transport = controlled_transport(monkeypatch)
    args = _remote_args_from_options(WorkflowOptions(profile="fixture", out=str(tmp_path)),
                                    profile_for("hailo8_to_trt", "hailo8"), model_task="detection")
    args.runtime_suite_finalized = True
    args.resume = False
    remote_run.run_remote_benchmark(
        host=HostConfig(id="fixture", label="fixture", host="127.0.0.1", user="fixture"),
        benchmark_set_json=suite / "benchmark_set.json", local_working_dir=tmp_path / "work",
        run_id="missing", args=args, log=lambda *a: None, progress=lambda *a: None,
        cancel_event=threading.Event(), remote_process_registry=RemoteProcessLeaseRegistry())
    status = json.loads((tmp_path / "work/Results/suite/1/missing/run_status.json").read_text())
    assert status["status"] == "failed"
    assert "Authoritative suite-local validation source is missing before upload" in status["fail_reason"]["message"]
    assert transport.uploads == []
