"""Normal Full staging tolerates selective remote trees without case metadata."""
import base64
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zlib
from types import SimpleNamespace

import pytest

from onnx_splitpoint_tool.workflow import hardware_acceptance_native as native
from onnx_splitpoint_tool.resources.remote_scripts.native_full_baseline_eval_runner import _first_case


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value if isinstance(value, bytes) else json.dumps(value, sort_keys=True).encode())
    return path


def inventory(root):
    return {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def bind_output_authority(local, model, task, contract):
    raw = {"model_id": model, "backend": "deepx_m1", "variant": "full", "task": task,
           "contract_status": "recorded", "endpoint_mode": "decoded",
           "host_tail_required": False, "postprocessing_required": False}
    outputs = put(local / "output_contracts.json", {"model_id": model, "task": task, "contracts": [raw]})
    contract["energy_workload"]["input_contract"]["endpoint_semantic_attestation"] = {
        "status": "attested", "pass": True, "model_id": model, "backend": "deepx_m1", "variant": "full",
        "source_file_sha256": native._sha(outputs), "authoritative_contract": raw,
        "authoritative_contract_sha256": hashlib.sha256(json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}


@pytest.fixture
def selective_full(tmp_path):
    model, case = "mobilenet_v3_large", "b001"
    primary = tmp_path / "original_local_run"
    local = primary / "native_producers/deepx" / model / "benchmark_set"
    remote, fresh = tmp_path / "selective_remote", tmp_path / "private_fresh"
    benchmark = {"schema": "onnx-splitpoint/benchmark-set", "schema_version": 2,
                 "model_name": model, "model": "models/" + model + ".onnx", "task": "classification",
                 "plan": {"runs": [{"id": "deepx_m1_full", "type": "deepx", "backend": "deepx_m1",
                    "provider": "deepx_m1", "variants": ["full"],
                    "dxnn_path": "deepx/deepx_m1/full/model.dxnn", "contract_path": "deepx/deepx_m1/full/output_contract.json",
                    "benchmark_task": "classification", "validation_budget_authoritative": True,
                    "validation_items_requested": 5000, "validation_max_images": 5000,
                    "validation_images": "resources/validation/classification/original5000/images",
                    "validation_manifest": "resources/validation/classification/original5000/manifest.json",
                    "mini_classification_eval": True}]},
                 "cases": [{"case_id": case, "folder": case, "case_dir": case,
                            "manifest": "split_manifest.json", "boundary": 1}]}
    split = {"schema": "onnx-splitpoint/split-manifest", "schema_version": 1, "boundary": 1,
             "full_model": "../models/" + model + ".onnx",
             "models": {"full": {"path": "../models/" + model + ".onnx"}}}
    put(local / "benchmark_set.json", benchmark)
    put(local / case / "split_manifest.json", split)
    contents = {"benchmark_set.json": (local / "benchmark_set.json").read_bytes(),
                "models/" + model + ".onnx": b"existing full onnx bytes",
                "deepx/deepx_m1/full/model.dxnn": b"existing compiled full dxnn bytes",
                "timing.jpg": b"existing timing image bytes"}
    for relative, value in contents.items():
        put(remote / relative, value)
    digest = lambda name: hashlib.sha256(contents[name]).hexdigest()
    contract = {"model": model, "input_case": case,
                "source_model_sha256": digest("models/" + model + ".onnx"),
                "input_image": str(remote / "timing.jpg"), "input_image_sha256": digest("timing.jpg"),
                "artifacts": {"dxnn": {"sha256": digest("deepx/deepx_m1/full/model.dxnn")},
                              "benchmark_set_manifest": {"sha256": digest("benchmark_set.json")}},
                "energy_workload": {"input_contract": {"model_id": model,
                    "artifact_sha256": digest("deepx/deepx_m1/full/model.dxnn"),
                    "input": {"shape": [224, 224, 3], "dtype": "uint8", "layout": "HWC"}}}}
    bind_output_authority(local, model, "classification", contract)
    metadata = native._full_case_metadata(primary, contract, deepx_runtime_mode="A4")
    item = {"argv": [sys.executable, "--deepx-runtime-mode", "A4"], "source_benchmark_set": str(remote),
            "full_benchmark_set": str(fresh), "full_source_contract": contract,
            "full_source_case_metadata": metadata}
    return SimpleNamespace(primary=primary, local=local, remote=remote, fresh=fresh, item=item,
                           contract=contract, metadata=metadata, tmp=tmp_path, case=case)


class LocalTransport:
    """Execute the actual transmitted normal staging commands without SSH."""
    def __init__(self, cwd):
        self.cwd, self.calls = cwd, []

    def run(self, command, **kwargs):
        self.calls.append(("run", command))
        result = subprocess.run(command, shell=True, executable="/bin/bash", cwd=self.cwd,
                                capture_output=True, text=True, timeout=kwargs["timeout"])
        return result.returncode, result.stdout + result.stderr

    def scp_download(self, source, destination, **kwargs):
        self.calls.append(("download", source))
        shutil.copyfile(source, destination)
        return 0, ""

    def scp_upload(self, source, destination, **kwargs):
        self.calls.append(("upload", destination))
        shutil.copyfile(source, destination)
        return 0, ""


@pytest.mark.parametrize("task", ["classification", "detection"])
def test_actual_normal_staging_hands_off_exact_declared_metadata_and_generates_runner(selective_full, monkeypatch, task):
    from onnx_splitpoint_tool.benchmark import suite_refresh
    point = selective_full
    if task == "detection":
        benchmark = json.loads((point.local / "benchmark_set.json").read_text())
        benchmark["task"] = task
        benchmark["plan"]["runs"][0].update(benchmark_task=task, validation_images="resources/validation/detection/original5000/images",
            mini_classification_eval=False, mini_coco_ap50=True)
        put(point.local / "benchmark_set.json", benchmark)
        shutil.copyfile(point.local / "benchmark_set.json", point.remote / "benchmark_set.json")
        point.contract["artifacts"]["benchmark_set_manifest"]["sha256"] = native._sha(point.local / "benchmark_set.json")
        bind_output_authority(point.local, "mobilenet_v3_large", task, point.contract)
        point.item["full_source_case_metadata"] = native._full_case_metadata(point.primary, point.contract, deepx_runtime_mode="A4")
    def forbidden_dataset_provision(*args, **kwargs):
        pytest.fail("software-only Full staging must not provision a validation dataset")
    monkeypatch.setattr(suite_refresh, "provision_classification_validation_source_to_suite", forbidden_dataset_provision)
    monkeypatch.setattr(suite_refresh, "provision_detection_validation_source_to_suite", forbidden_dataset_provision)
    before_remote, before_local = inventory(point.remote), inventory(point.local)
    original_contract = copy.deepcopy(point.contract)
    assert not list(point.remote.glob("b*/split_manifest.json"))
    transport = LocalTransport(point.tmp)
    receipts = point.tmp / "receipts"
    native._stage_full_benchmark_set(transport, point.item, receipts)
    assert inventory(point.remote) == before_remote and inventory(point.local) == before_local
    assert point.contract == original_contract
    assert (point.fresh / "benchmark_set.json").read_bytes() == before_local["benchmark_set.json"]
    assert (receipts / "software/benchmark_set.json").read_bytes() == before_local["benchmark_set.json"]
    assert (point.fresh / point.case / "split_manifest.json").read_bytes() == before_local[point.case + "/split_manifest.json"]
    case, directory, runner = _first_case(point.fresh)
    assert case == point.case and directory == point.fresh / point.case and runner.is_file()
    assert (point.fresh / "splitpoint_runners/native_deepx_async.py").is_file()
    assert "--deepx-runtime-mode" in (point.fresh / "benchmark_suite.py").read_text()
    assert not list(point.fresh.glob("b*/*.onnx"))
    proof = json.loads((receipts / "software_staging.json").read_text())
    assert proof["returncode"] == 0 and proof["suite_refresh"]["validation_resource_provisioned"] is False
    assert proof["suite_refresh"]["software_only"] is True and not (receipts / "software/resources").exists()
    assert [operation for operation, _ in transport.calls].count("upload") == 1


def test_default_suite_refresh_retains_existing_validation_provisioning(selective_full, monkeypatch):
    from onnx_splitpoint_tool.benchmark import suite_refresh
    calls = []
    monkeypatch.setattr(suite_refresh, "provision_classification_validation_source_to_suite",
                        lambda *args, **kwargs: calls.append((args, kwargs)))
    suite_refresh.refresh_suite_harness(selective_full.local, mini_coco_ap50=False, mini_classification_eval=False)
    assert len(calls) == 1 and calls[0][1]["max_images"] == 5000


@pytest.mark.parametrize("change", ["benchmark_hash", "case_missing", "different_declared_case", "different_full_model"])
def test_original_local_metadata_must_match_sealed_declared_case(selective_full, change):
    point = selective_full
    if change == "benchmark_hash":
        put(point.local / "benchmark_set.json", {"cases": []})
    elif change == "case_missing":
        (point.local / point.case / "split_manifest.json").unlink()
    elif change == "different_declared_case":
        point.contract["input_case"] = "b099"
    else:
        split = json.loads((point.local / point.case / "split_manifest.json").read_text())
        split["models"]["full"]["path"] = "../models/other.onnx"
        put(point.local / point.case / "split_manifest.json", split)
    with pytest.raises((ValueError, FileNotFoundError)):
        native._full_case_metadata(point.primary, point.contract, deepx_runtime_mode="A4")


def test_local_case_drift_is_rejected_before_any_transport(selective_full):
    point = selective_full
    put(point.local / point.case / "split_manifest.json", {"changed": True})
    transport = LocalTransport(point.tmp)
    with pytest.raises(ValueError, match="changed"):
        native._stage_full_benchmark_set(transport, point.item, point.tmp / "receipts")
    assert not transport.calls and not point.fresh.exists()


def test_downloaded_metadata_is_hash_checked_before_software_refresh(selective_full):
    point = selective_full
    class CorruptDownload(LocalTransport):
        def scp_download(self, source, destination, **kwargs):
            result = super().scp_download(source, destination, **kwargs)
            Path(destination).write_bytes(b"different metadata")
            return result
    transport = CorruptDownload(point.tmp)
    with pytest.raises(ValueError, match="download differs"):
        native._stage_full_benchmark_set(transport, point.item, point.tmp / "receipts")
    assert not (point.fresh / "benchmark_suite.py").exists()
    assert not any(operation == "upload" for operation, _ in transport.calls)


@pytest.mark.parametrize("change", ["remote_benchmark", "payload_bytes", "case_identity", "different_full_model"])
def test_transmitted_helper_rejects_metadata_mismatch_before_destination(selective_full, change):
    point = selective_full
    metadata = copy.deepcopy(point.metadata)
    raw = (point.local / point.case / "split_manifest.json").read_bytes()
    if change == "remote_benchmark":
        put(point.remote / "benchmark_set.json", {"cases": []})
    elif change == "payload_bytes":
        raw += b"\n"
    elif change == "case_identity":
        metadata["case_id"] = "b099"
    else:
        split = json.loads(raw)
        split["full_model"] = "../models/other.onnx"
        raw = json.dumps(split).encode()
        metadata["case_manifest"].update(sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))
    metadata["case_manifest_zlib_base64"] = base64.b64encode(zlib.compress(raw)).decode()
    metadata["output_contracts_zlib_base64"] = base64.b64encode(zlib.compress((point.local / "output_contracts.json").read_bytes())).decode()
    with pytest.raises(ValueError, match="metadata|manifest"):
        native._copy_full_benchmark(point.remote, point.fresh, point.contract, None, metadata, "A4")
    assert not point.fresh.exists()


def test_no_arbitrary_other_case_fallback_when_declared_case_is_missing(selective_full):
    point = selective_full
    put(point.remote / "b099/split_manifest.json", {})
    with pytest.raises(ValueError, match="declared case"):
        native._copy_full_benchmark(point.remote, point.fresh, point.contract)
    assert not point.fresh.exists()
