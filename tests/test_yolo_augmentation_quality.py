"""Cross-run reference reuse through real readers/loaders; never inference."""
import copy
import hashlib
import json
from pathlib import Path
import threading

import pytest

from onnx_splitpoint_tool.management_reference import _source_contract
from onnx_splitpoint_tool.quality_cache import image_ids_fingerprint, json_fingerprint
from onnx_splitpoint_tool.quality_service import quality_request_from_manifest
from onnx_splitpoint_tool.workflow.runner import EvaluationWorkflowRunner
from onnx_splitpoint_tool.yolo_augmentation_quality import OriginalCPUReferenceMixin
from tests.test_v269d_trt_central_quality_producer import _detection_quality_contract


class AugmentationRunner(OriginalCPUReferenceMixin, EvaluationWorkflowRunner):
    pass


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, sort_keys=True).encode()
    path.write_bytes(encoded)
    return {"path": str(path), "size_bytes": len(encoded), "sha256": hashlib.sha256(encoded).hexdigest()}


def seal(contract):
    contract.pop("quality_contract_sha256", None)
    contract["quality_contract_sha256"] = json_fingerprint(contract)
    return contract


@pytest.fixture
def sample(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("reference reuse must never generate or dispatch inference")
    monkeypatch.setattr("onnx_splitpoint_tool.management_reference.generate_management_cpu_reference", forbidden)
    original, new = tmp_path / "closed_original", tmp_path / "new_augmentation"
    write(original / "run_manifest.json", {"run_id": original.name})
    suite = original / "models/yolo11l/benchmark_set/legacy_suite"
    plan, contract = {"model_id": "yolo11l", "runs": []}, {"model_id": "yolo11l"}
    write(suite / "benchmark_plan.json", plan)
    write(suite / "benchmark_set.json", contract)
    (suite / "benchmark_suite.py").write_text("original writer\n")
    source_sha = _source_contract(suite, plan, contract).removeprefix("sha256:")
    rows = [{"image_id": f"{i:05}.jpg", "reference": [], "ground_truth": []} for i in range(5000)]
    quality = _detection_quality_contract()
    quality["dataset"].update(image_count=5000,
        image_ids_sha256=image_ids_fingerprint([r["image_id"] for r in rows]),
        ground_truth_sha256=json_fingerprint([{k:r[k] for k in ("image_id", "ground_truth")} for r in rows]))
    seal(quality)
    payload = {"schema": "onnx-splitpoint/task-quality-reference-input", "schema_version": 1,
        "task": "detection", "pairing_key": "image_id", "reference_role": "canonical_cpu_ort",
        "semantic_reference_only": True, "provenance_required": True,
        "quality_contract": quality, "quality_contract_sha256": quality["quality_contract_sha256"], "records": rows}
    path = original / f"quality_management/references/yolo11l/by_source_contract/{source_sha}/canonical_cpu_reference.json"
    descriptor = write(path, payload)
    status = {"schema": "onnx-splitpoint/management-cpu-reference-job", "schema_version": 1,
        "status": "completed", "model_id": "yolo11l", "reference_path": str(path),
        "reference_sha256": descriptor["sha256"], "reference_size_bytes": descriptor["size_bytes"],
        "execution_location": "central_management", "provider": "onnxruntime_cpu",
        "semantic_reference_only": True, "include_in_latency_fps_energy": False,
        "include_in_ranking": False, "include_in_pareto": False, "reference_storage": "immutable_source_contract",
        "reference_immutable": True, "source_contract_sha256": source_sha,
        "task": "detection", "record_count": 5000, "image_ids_sha256": quality["dataset"]["image_ids_sha256"],
        "quality_contract": quality, "quality_contract_sha256": quality["quality_contract_sha256"]}
    status_path = original / "quality_management/references/yolo11l/management_cpu_reference_status.json"
    write(status_path, status)
    write(new / "models/yolo11l/model_manifest.json", {"model_id": "yolo11l", "model_sha256": quality["model"]["sha256"]})
    runner = object.__new__(AugmentationRunner)
    runner.run_dir, runner.run_id, runner.augmentation_reference_run = new, new.name, original
    runner._management_services_lock = threading.RLock()
    runner._management_admission_closed = False
    runner._management_reference_futures, runner._management_reference_source_contracts = {}, {}
    return runner, original, path, status_path, quality, rows


def request_bundle(root, quality, rows):
    candidate_rows = [{"image_id": r["image_id"], "ground_truth": r["ground_truth"], "candidate": []} for r in rows]
    candidate = {"schema": "onnx-splitpoint/task-quality-candidate-input", "schema_version": 1,
        "task": "detection", "variant": "composed", "pairing_key": "image_id", "provenance_required": True,
        "quality_contract": quality, "quality_contract_sha256": quality["quality_contract_sha256"], "records": candidate_rows}
    desc = write(root / "candidate.json", candidate)
    request = {"schema": "onnx-splitpoint/central-quality-evaluation-request", "schema_version": 1,
        "task": "detection", "variant": "composed", "pairing_key": "image_id", "execution_location": "management_node",
        "eval_run_id": root.name, "provenance_required": True, "quality_contract": quality,
        "quality_contract_sha256": quality["quality_contract_sha256"],
        "reference": {"source": "management_cpu_reference", "quality_contract_sha256": quality["quality_contract_sha256"]},
        "candidate": desc, "record_count": len(rows), "reference_record_count": len(rows),
        "policy_sha256": "b" * 64, "metric_gate_config": {"primary_metric": "coco_ap_50_95", "non_inferiority_margin": .01},
        "statistics": {"method": "paired_bootstrap", "bootstrap_repetitions": 1000, "seed": 20260710,
                       "confidence_level": .95, "decision": "lower_one_sided_bound"}}
    path = root / "composed_request.json"
    write(path, request)
    return path


def load(runner, request):
    _, status, encoded = runner._management_reference_records("yolo11l")
    return quality_request_from_manifest(request,
        reference_artifact={"path": status["reference_path"], "size_bytes": status["reference_size_bytes"],
                            "sha256": status["reference_sha256"]}, reference_artifact_bytes=encoded)


def test_new_request_uses_original_reference_and_preserves_original_bytes(sample):
    runner, original, path, _, quality, rows = sample
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in original.rglob("*") if p.is_file()}
    runner._schedule_management_cpu_reference("yolo11l", runner.run_dir / "new_suite")
    first = runner._management_reference_futures["yolo11l"]
    runner._restore_management_cpu_reference("yolo11l")
    assert runner._management_reference_futures["yolo11l"] is first
    records, status, encoded = runner._management_reference_records("yolo11l")
    assert len(records) == 5000 and encoded == path.read_bytes()
    assert status["reference_path"] == str(path)
    assert status["source_evaluation_run_id"] == original.name
    request = request_bundle(runner.run_dir, quality, rows)
    loaded = load(runner, request)
    assert len(loaded.reference_records) == 5000
    assert loaded.repetitions == 1000 and loaded.seed == 20260710
    assert json.loads(request.read_text())["eval_run_id"] == runner.run_id != original.name
    assert not (runner.run_dir / "quality_management/references").exists()
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in original.rglob("*") if p.is_file()}


@pytest.mark.parametrize("damage", ["model", "source", "reference", "status", "run_id", "same_run", "symlink"])
def test_reference_admission_rejects_wrong_or_changed_sources(sample, damage):
    runner, original, path, status_path, quality, _ = sample
    if damage == "model":
        write(runner.run_dir / "models/yolo11l/model_manifest.json", {"model_sha256": "a" * 64})
    elif damage == "source":
        (original / "models/yolo11l/benchmark_set/legacy_suite/benchmark_suite.py").write_text("changed source")
    elif damage == "reference":
        path.write_bytes(path.read_bytes() + b" ")
    elif damage == "status":
        status = json.loads(status_path.read_text());status["record_count"] = 4999;write(status_path, status)
    elif damage == "run_id":
        write(original / "run_manifest.json", {"run_id": "some_other_run"})
    elif damage == "same_run":
        runner.run_dir = original
    else:
        target = path.with_name("moved.json");path.rename(target);path.symlink_to(target)
    with pytest.raises((ValueError, RuntimeError)):
        runner._schedule_management_cpu_reference("yolo11l", None)
    assert not runner._management_reference_futures


def test_reference_is_rechecked_after_admission(sample):
    runner, _, path, _, _, _ = sample
    runner._schedule_management_cpu_reference("yolo11l", None)
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(RuntimeError):
        runner._management_reference_records("yolo11l")


@pytest.mark.parametrize("field", ["model", "dataset", "preprocessing", "decoder", "nms"])
def test_actual_quality_loader_rejects_new_contract_drift(sample, field):
    runner, _, _, _, quality, rows = sample
    runner._schedule_management_cpu_reference("yolo11l", None)
    changed = copy.deepcopy(quality)
    if field == "model":
        changed[field]["sha256"] = "f" * 64
    elif field == "dataset":
        changed[field]["manifest_sha256"] = "f" * 64
    else:
        changed[field]["sha256"] = "f" * 64
    seal(changed)
    request = request_bundle(runner.run_dir, changed, rows)
    with pytest.raises(ValueError):
        load(runner, request)


def test_original_reader_keeps_global_cross_run_guard(sample):
    runner, _, _, _, _, _ = sample
    runner._schedule_management_cpu_reference("yolo11l", None)
    with pytest.raises(RuntimeError, match="escapes the active run"):
        EvaluationWorkflowRunner._management_reference_records(runner, "yolo11l")


def test_other_models_and_unadmitted_reads_are_rejected(sample):
    runner, _, _, _, _, _ = sample
    with pytest.raises(ValueError, match="outside the authorized cohort"):
        runner._schedule_management_cpu_reference("yolo26m", None)
    with pytest.raises(RuntimeError, match="not admitted"):
        runner._management_reference_records("yolo11l")


def test_current_generic_writer_binds_old_reference_contract_and_current_producer(sample, tmp_path):
    from tests.test_v27911_composed_provenance_binding import _functions, RUNNER
    runner, _, _, _, old, rows = sample
    runner._schedule_management_cpu_reference("yolo11l", None)
    _, status, _ = runner._management_reference_records("yolo11l")
    namespace = _functions([
        "_task_quality_execution_location", "_quality_json_safe", "_write_stable_quality_json",
        "_quality_file_sha256", "_quality_contract_sha256", "_quality_artifact_binding",
        "_build_generic_composed_quality_producer_binding", "_export_central_quality_inputs",
        "_existing_management_reference",
    ])
    current = copy.deepcopy(old)
    live_sha = hashlib.sha256(RUNNER.read_bytes()).hexdigest()
    for component in ("decoder", "nms", "quality_record_endpoint"):
        identity = current[component]["identity"]
        identity["implementation_runner_sha256"] = live_sha
        if component == "quality_record_endpoint":
            identity.update(decoder_contract_sha256=current["decoder"]["sha256"],
                            nms_contract_sha256=current["nms"]["sha256"],
                            vendored_endpoint_attestor_sha256=live_sha)
        current[component]["sha256"] = json_fingerprint(identity)
    current["quality_record_endpoint_contract_sha256"] = current["quality_record_endpoint"]["sha256"]
    seal(current)
    namespace["_verified_suite_vendored_endpoint_attestor_sha256"] = lambda: live_sha
    descriptor = dict(status, reference_role="canonical_cpu_ort")
    admitted = namespace["_existing_management_reference"](
        descriptor, model_sha256=current["model"]["sha256"],
        variants=["composed"], central=True, image_count=5000,
    )
    paths = {}
    for name, content in (("full", b"full-source-onnx"), ("part1", b"part1"),
                          ("part2", b"part2"), ("p1-runtime", b"dxnn"), ("p2-runtime", b"engine")):
        paths[name] = tmp_path / name; paths[name].write_bytes(content)
    exported = namespace["_export_central_quality_inputs"](
        out_dir=runner.run_dir / "new_candidate", task="detection", variant="composed",
        policy={"_existing_management_reference": admitted, "policy_sha256": "b" * 64,
                "statistics": {"execution_location": "central_management", "bootstrap_repetitions": 1000}},
        gt_by_image={row["image_id"]: [] for row in rows},
        candidate_by_image={row["image_id"]: [] for row in rows},
        quality_contract=current,
        endpoint_contract={"endpoint_contract_complete": True, "endpoint_contract_hash": "2" * 64,
                           "stage": "raw_head", "contract_family": "raw_head"},
        runtime_precision_identity="fp16",
        composed_producer_context={
            "eval_run_id": runner.run_id, "model_id": "yolo11l", "setup_id": "deepx-setup",
            "source_run_id": "deepx_to_trt", "case_id": "b066",
            "stage1_backend": "deepx", "stage2_backend": "native_tensorrt",
            "source_model_path": str(paths["full"]), "part1_model_path": str(paths["part1"]),
            "part2_model_path": str(paths["part2"]), "implementation_runner_path": str(RUNNER),
            "runtime_artifacts": [{"role": role, "backend": backend, "path": str(paths[path])}
                                  for role, backend, path in (("stage1_runtime", "deepx", "p1-runtime"),
                                                               ("stage2_runtime", "native_tensorrt", "p2-runtime"))],
        },
    )
    request_path = Path(exported["request"]["path"])
    assert exported["quality_contract_sha256"] == current["quality_contract_sha256"] != old["quality_contract_sha256"]
    assert exported["reference"]["quality_contract_sha256"] == old["quality_contract_sha256"]
    assert exported["producer_identity"]["eval_run_id"] == runner.run_id
    assert len(load(runner, request_path).reference_records) == 5000
    broken = json.loads(request_path.read_text())
    broken["reference"]["quality_contract_sha256"] = current["quality_contract_sha256"]
    write(request_path, broken)
    with pytest.raises(ValueError, match="reference descriptor quality contract"):
        load(runner, request_path)
