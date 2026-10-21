"""Actual Full contract writer/readers and ordinary scoped result collection."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from scripts import native_full_baseline_eval_runner as full
from onnx_splitpoint_tool.resources.remote_scripts import native_producer_energy_plan as energy_plan
from onnx_splitpoint_tool.workflow import hardware_acceptance_native as native
from onnx_splitpoint_tool.workflow import hardware_acceptance_numerical as numerical
from test_v27930_native_full_semantic_merge import prepare_runner_case, write_json
from test_v27930_deepx_semantic_pre_nms import load_semantic_module
from test_v295_native_deepx_full_a4 import _sdk, _suite


def actual_full_case(tmp_path, monkeypatch, *, frames=64, warmup=0, numerical_only=True):
    """Only physical SDK/process boundaries are simulated; contract writing is real."""
    values = [np.arange(1000, dtype=np.float32)[None]]
    case = prepare_runner_case(tmp_path, monkeypatch, model="mobilenet_v3_large", task="classification", output=values)
    contract = {"model_id": case.model, "task": case.task, "backend": "deepx_m1",
        "classification_preprocessing": "imagenet_mean_std", "source_onnx_sha256": full._sha256_file(case.root / "models" / (case.model + ".onnx")),
        "input": {"name": "input", "shape": [32, 32, 3], "dtype": "uint8", "layout": "HWC",
                  "normalization": "embedded_dxcom_preprocessing", "color_space": "RGB", "preprocess_mode": "resize"},
        "outputs": [{"name": "logits", "shape": [1, 1000], "dtype": "float32"}]}
    case.contract = contract
    write_json(case.full / "output_contract.json", contract)
    authority = {"model_id": case.model, "backend": "deepx_m1", "variant": "full", "task": case.task,
        "contract_status": "recorded", "endpoint_mode": "decoded", "host_tail_required": False, "postprocessing_required": False}
    write_json(case.root / "output_contracts.json", {"model_id": case.model, "task": case.task, "contracts": [authority]})
    case.ns.deepx_runtime_mode = "A4"
    case.ns.frames, case.ns.warmup, case.ns.repetitions = frames, warmup, 1
    case.ns.out_dir = str(tmp_path / "separate_report_output")
    case.ns.workspace_mb, case.ns.inflight = 4096, 4
    case.ns.engine_build_python = sys.executable
    case.ns.no_shapes = case.ns.diagnostic_deepx_input_probes = False
    case.ns.image_map_data = {case.model: {"full": str(case.image)}}
    case.ns.deepx_numerical_only = numerical_only
    images = [str(case.image)]
    for index in range(31):
        image = tmp_path / "frozen_images" / (str(index) + ".png")
        image.parent.mkdir(exist_ok=True)
        Image.new("RGB", (32, 32), (index, index + 1, index + 2)).save(image)
        images.append(str(image))
    image_list = tmp_path / "numerical_images.json"
    write_json(image_list, {"images": images})
    case.ns.deepx_numerical_images = str(image_list) if numerical_only else ""
    observed = _sdk(monkeypatch, contract, values)
    semantic_module, suite = load_semantic_module(), _suite()

    def process(command, **kwargs):
        def option(flag): return command[command.index(flag) + 1]
        if "native_full_semantic_dump.py" in command[1]:
            monkeypatch.setattr(sys, "argv", command[1:])
            rc = semantic_module.main()
        else:
            args = SimpleNamespace(runs=3, warmup=warmup, throughput_frames=frames,
                throughput_warmup_frames=warmup, deepx_runtime_mode="A4", energy_measurement_only=True,
                prepared_feed_image=str(case.image), prepared_input_manifest=option("--prepared-input-manifest"),
                native_full_postprocess_runtime_mode="auto", quality_evidence_model_id=case.model,
                quality_evidence_setup_id=case.setup_id, deepx_numerical_images=case.ns.deepx_numerical_images,
                deepx_numerical_only=numerical_only)
            run = {"id": "deepx_m1_full", "benchmark_task": case.task, "backend": "deepx_m1",
                   "provider": "deepx_m1", "variants": ["full"], "model_id": case.model}
            rows = suite._run_deepx_full_run(case.root, run, args,
                expected_endpoint_identity={"model_id": case.model, "backend": "deepx_m1", "variant": "full"})
            write_json(case.root / "benchmark_results_deepx_m1_full_auto.json", rows)
            rc = 0 if rows[0].get("runtime_ok") else 1
        return {"rc": rc, "timed_out": False, "stdout": "", "stderr": "", "cmd": command}

    monkeypatch.setattr(full, "_run", process)
    semantic, blocked = full._deepx_full_series_preflight(case.root, case.model, case.ns)
    assert blocked is None, blocked
    case.ns.deepx_full_precomputed_semantic = semantic
    row = full._row_for_backend(case.root, case.model, "deepx", case.ns)
    assert row["ok"] is True, row
    # Execute the actual main envelope/identity/aggregation/contract writer.
    # Reuse the already completed real worker values at its process boundary;
    # no second semantic or inference series is hidden in this roundtrip.
    with monkeypatch.context() as main_patch:
        main_patch.setattr(full, "_deepx_full_series_preflight", lambda *args: (semantic, None))
        main_patch.setattr(full, "_row_for_backend", lambda *args: copy.deepcopy(row))
        argv = [full.__file__, "--root", str(case.root.parent.parent), "--models", case.model,
            "--backends", "deepx", "--frames", str(frames), "--warmup", str(warmup), "--repetitions", "1",
            "--setup-id", case.setup_id, "--comparison-backend", "deepx", "--engine-build-python", sys.executable,
            "--deepx-runtime-mode", "A4", "--dump-outputs", "--out-dir", case.ns.out_dir]
        if numerical_only:
            argv.extend(["--deepx-numerical-images", str(image_list), "--deepx-numerical-only"])
        main_patch.setattr(sys, "argv", argv)
        assert full.main() == 0
    raw = json.loads((Path(case.ns.out_dir) / "analysis_tables/native_full_baseline_eval.json").read_text())
    row = raw["rows"][0]
    command_contract = row["full_command_contract"]
    assert command_contract["complete"] is True, command_contract["energy_workload"]
    row.update(full_command_contract=command_contract, full_command_contract_sha256=command_contract["contract_sha256"])
    case.row, case.command_contract, case.observed = row, command_contract, observed
    case.raw = raw
    return case


@pytest.mark.parametrize("frames,warmup,numerical_only", [(64, 0, True), (1000, 100, False)])
def test_real_full_writer_separate_report_dir_roundtrips_strict_readers(tmp_path, monkeypatch, frames, warmup, numerical_only):
    case = actual_full_case(tmp_path, monkeypatch, frames=frames, warmup=warmup, numerical_only=numerical_only)
    contract = case.command_contract
    identity = {key: contract[key] for key in ("backend", "model", "case", "setup_id")}
    expected = {**identity, "remote_root": contract["root"], "remote_tool_dir": str(full.ROOT)}
    verified, reason = energy_plan._verify_full_command_contract(contract, expected_identity=expected)
    assert verified is not None, reason
    replay, reason = full._verified_full_energy_contract(contract, expected_root=Path(contract["root"]))
    assert replay is not None, reason
    semantic_root = full._native_full_dump_dir(case.root, case.model, "native_full_deepx", case.ns)
    assert semantic_root.is_relative_to(case.root) and not semantic_root.is_relative_to(Path(case.ns.out_dir))
    assert case.row["preparation_evidence"]["report"] == str(semantic_root / "native_full_semantic_dump.json")
    if numerical_only:
        prepared = {"identity": identity, "source_contract": contract, "remote_tool_dir": str(full.ROOT)}
        result = numerical.validate_numerical_raw(case.raw, prepared, expected_images=32)
        assert (result["serial_reference_count"], result["mixed_completed_count"], result["smoke_completed_count"]) == (32, 32, 64)
        assert result["device_completion_proven"] is True
    else:
        proof = native._validate_full_raw(case.raw, identity, contract, remote_root=contract["root"], remote_tool_dir=str(full.ROOT))
        assert proof["completed_tasks"] == 1000


@pytest.mark.parametrize("role", ["input_manifest", "runtime_input_tensor"])
def test_foreign_hash_equal_artifact_role_remains_rejected(tmp_path, monkeypatch, role):
    case = actual_full_case(tmp_path, monkeypatch)
    contract = copy.deepcopy(case.command_contract)
    source = Path(contract["artifacts"][role]["path"])
    foreign = tmp_path / "unrelated" / source.name
    foreign.parent.mkdir(exist_ok=True)
    shutil.copyfile(source, foreign)
    contract["artifacts"][role]["path"] = str(foreign)
    contract["contract_sha256"] = full._canonical_json_sha256({key: value for key, value in contract.items() if key != "contract_sha256"})
    expected = {key: contract[key] for key in ("backend", "model", "case", "setup_id")}
    expected.update(remote_root=contract["root"], remote_tool_dir=str(full.ROOT))
    verified, reason = energy_plan._verify_full_command_contract(contract, expected_identity=expected)
    assert verified is None and reason == "full_command_contract_deepx_artifact_role_path_mismatch"


class LocalDownload:
    def __init__(self): self.sources = []
    def scp_download(self, source, destination, **kwargs):
        self.sources.append(source)
        shutil.copytree(source, Path(destination) / Path(source).name)
        return 0, ""


def stage_output_authority(item, directory, source):
    """Mirror the normal pre-worker staging download, with its original binding."""
    target = directory / "software/output_contracts.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    item["full_source_case_metadata"] = {"output_contracts_manifest": {
        "path": str(source), "sha256": full._sha256_file(source)}}
    return target


@pytest.mark.parametrize("publication", ["canonical", "early_guard", "foreign"])
def test_collector_canonical_subtree_only_after_exact_publication(tmp_path, publication):
    benchmark = tmp_path / "remote/inputs/mobilenet_v3_large/benchmark_set"
    output = tmp_path / "remote/outputs"
    semantic = benchmark / "native_full_outputs/model=mobilenet_v3_large/backend=native_full_deepx/setup=setup/comparison=deepx"
    semantic.mkdir(parents=True)
    for name in ("native_full_semantic_dump.json", "native_full_input_manifest.json", "native_full_outputs_manifest.json"):
        write_json(semantic / name, {"marker": name})
    (semantic / "runtime_input.bin").write_bytes(b"exact prepared tensor")
    preparation = {"report": str(semantic / "native_full_semantic_dump.json")}
    if publication == "early_guard": preparation = {"status": "full_suite_route_invalid"}
    if publication == "foreign": preparation["report"] = str(tmp_path / "foreign/native_full_semantic_dump.json")
    write_json(output / "analysis_tables/native_full_baseline_eval.json", {"rows": [{"preparation_evidence": preparation}]})
    item = {"native_full": True, "full_benchmark_set": str(benchmark), "remote_output_root": str(output),
        "result_json": str(output / "analysis_tables/native_full_baseline_eval.json"),
        "argv": [sys.executable, "worker.py", "--models", "mobilenet_v3_large", "--setup-id", "setup",
                 "--comparison-backend", "deepx", "--deepx-runtime-mode", "A4"]}
    write_json(benchmark / "output_contracts.json", {"original_authority": True})
    staged_authority = stage_output_authority(item, tmp_path / "collector", benchmark / "output_contracts.json")
    transport = LocalDownload()
    report = native._collect_repeat(transport, item, tmp_path / "collector")
    assert report.is_file() and report.parent.name == "analysis_tables"
    local = report.parent.parent / "mobilenet_v3_large/benchmark_set" / semantic.relative_to(benchmark)
    if publication == "canonical":
        assert len(transport.sources) == 2
        assert transport.sources[1] == str(benchmark / "native_full_outputs")
        assert (local / "runtime_input.bin").read_bytes() == b"exact prepared tensor"
        assert (report.parent.parent / "mobilenet_v3_large/benchmark_set/output_contracts.json").read_bytes() == staged_authority.read_bytes()
        assert all((local / name).is_file() for name in ("native_full_input_manifest.json", "native_full_outputs_manifest.json", "native_full_semantic_dump.json"))
    else:
        assert len(transport.sources) == 1 and not local.exists()


@pytest.mark.parametrize("authority_failure", ["missing_file", "changed_file", "missing_binding"])
def test_collector_rejects_missing_or_changed_staged_authority_without_source_fallback(tmp_path, authority_failure):
    benchmark = tmp_path / "remote/inputs/mobilenet_v3_large/benchmark_set"
    output = tmp_path / "remote/outputs"
    semantic = benchmark / "native_full_outputs/model=mobilenet_v3_large/backend=native_full_deepx/setup=setup/comparison=deepx"
    write_json(semantic / "native_full_semantic_dump.json", {"marker": "original"})
    write_json(benchmark / "output_contracts.json", {"original_authority": True})
    write_json(output / "analysis_tables/native_full_baseline_eval.json", {"rows": [{
        "preparation_evidence": {"report": str(semantic / "native_full_semantic_dump.json")}}]})
    item = {"native_full": True, "full_benchmark_set": str(benchmark), "remote_output_root": str(output),
        "result_json": str(output / "analysis_tables/native_full_baseline_eval.json"),
        "argv": [sys.executable, "worker.py", "--models", "mobilenet_v3_large", "--setup-id", "setup",
                 "--comparison-backend", "deepx", "--deepx-runtime-mode", "A4"]}
    directory = tmp_path / "collector"
    authority = stage_output_authority(item, directory, benchmark / "output_contracts.json")
    if authority_failure == "missing_file":
        authority.unlink()
    elif authority_failure == "changed_file":
        authority.write_bytes(b"changed after normal staging")
    else:
        item.pop("full_source_case_metadata")
    transport = LocalDownload()
    with pytest.raises((ValueError, FileNotFoundError)):
        native._collect_repeat(transport, item, directory)
    assert transport.sources == [str(output)]
    assert not (directory / "download/outputs/mobilenet_v3_large/benchmark_set").exists()
    assert (benchmark / "output_contracts.json").is_file()


def test_s1_and_other_full_backends_keep_explicit_output_location(tmp_path):
    for backend, mode in (("native_full_deepx", "S1"), ("native_full_tensorrt", "A4")):
        ns = SimpleNamespace(out_dir=str(tmp_path / "reports"), deepx_runtime_mode=mode)
        assert full._native_full_dump_dir(tmp_path / "benchmark_set", "model", backend, ns).is_relative_to(tmp_path / "reports")


def test_real_writer_collector_and_existing_manifest_import_readback(tmp_path, monkeypatch):
    from scripts import native_producer_final_report as final_report
    from scripts import native_producer_validate_visualize as visualize
    from onnx_splitpoint_tool.resources.remote_scripts.validate_output_dumps import load_dump
    case = actual_full_case(tmp_path, monkeypatch)
    output = Path(case.ns.out_dir)
    item = {"native_full": True, "full_benchmark_set": str(case.root), "remote_output_root": str(output),
        "result_json": str(output / "analysis_tables/native_full_baseline_eval.json"),
        "argv": [sys.executable, "worker.py", "--models", case.model, "--setup-id", case.setup_id,
                 "--comparison-backend", "deepx", "--deepx-runtime-mode", "A4"]}
    staged_authority = stage_output_authority(item, tmp_path / "collected_point", case.root / "output_contracts.json")
    report = native._collect_repeat(LocalDownload(), item, tmp_path / "collected_point")
    collected = report.parent.parent
    original_contract = copy.deepcopy(case.command_contract)
    case.root.rename(tmp_path / "software_test_remote_unavailable")
    raw = native._read(report)
    assert raw["rows"][0]["full_command_contract"] == original_contract
    projected = final_report._rows_from_native_full(collected)
    assert len(projected) == 1 and projected[0]["full_command_contract"] == original_contract
    manifest, reason = visualize._extract_manifest_from_json_obj(raw["rows"][0], [collected], strict_unique=True, row=raw["rows"][0])
    assert manifest is not None, reason
    assert manifest.is_relative_to(collected / case.model / "benchmark_set")
    assert (collected / case.model / "benchmark_set/output_contracts.json").read_bytes() == staged_authority.read_bytes()
    inputs = visualize._find_native_full_input_manifest_for_output(manifest, endpoint_evidence=raw["rows"][0])
    assert inputs == manifest.parent / "native_full_input_manifest.json"
    assert full._sha256_file(inputs) == original_contract["artifacts"]["input_manifest"]["sha256"]
    assert full._sha256_file(inputs.parent / "runtime_input.bin") == original_contract["artifacts"]["runtime_input_tensor"]["sha256"]
    outputs, metadata = load_dump(str(manifest))
    assert list(outputs) == ["logits"] and outputs["logits"].shape == (1, 1000)
    assert metadata["model"] == case.model


def test_collector_rejects_downloaded_symlink_before_secondary_write(tmp_path):
    output, outside = tmp_path / "remote/outputs", tmp_path / "outside"
    outside.mkdir()
    benchmark = tmp_path / "remote/inputs/mobilenet_v3_large/benchmark_set"
    publication = str(benchmark / "native_full_outputs/model=mobilenet_v3_large/backend=native_full_deepx/setup=setup/comparison=deepx/native_full_semantic_dump.json")
    write_json(output / "analysis_tables/native_full_baseline_eval.json", {"rows": [{"preparation_evidence": {"report": publication}}]})
    (output / "mobilenet_v3_large").symlink_to(outside, target_is_directory=True)
    item = {"native_full": True, "full_benchmark_set": str(benchmark), "remote_output_root": str(output),
        "result_json": str(output / "analysis_tables/native_full_baseline_eval.json"),
        "argv": [sys.executable, "--models", "mobilenet_v3_large", "--setup-id", "setup",
                 "--comparison-backend", "deepx", "--deepx-runtime-mode", "A4"]}
    class SymlinkDownload(LocalDownload):
        def scp_download(self, source, destination, **kwargs):
            self.sources.append(source)
            shutil.copytree(source, Path(destination) / Path(source).name, symlinks=True)
            return 0, ""
    transport = SymlinkDownload()
    with pytest.raises(ValueError, match="unsafe symlink"):
        native._collect_repeat(transport, item, tmp_path / "collector")
    assert len(transport.sources) == 1 and not list(outside.iterdir())
