from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import deepx_full_output_probe as driver
from scripts import deepx_full_output_probe_worker as worker


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


@pytest.fixture
def original(tmp_path):
    root = tmp_path / "original suite"
    dxnn = root / "deepx/deepx_m1/full/model.dxnn"
    dxnn.parent.mkdir(parents=True)
    dxnn.write_bytes(b"fake model; tests never load a vendor engine")
    image = root / "resources/validation/000000212226.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"fake image; mock probe does not decode it")
    (root / "splitpoint_runners").mkdir()
    (root / "splitpoint_runners/__init__.py").write_text("")
    (root / "splitpoint_runners/native_detection_postprocess.py").write_text("ORIGINAL = True\n")
    run = {"id": "deepx_m1_full", "setup_id": "deepx_setup", "benchmark_task": "detection"}
    write_json(root / "benchmark_plan.json", {"runs": [run]})
    request = {
        "schema": driver.REQUEST_SCHEMA, "original_run_id": "original",
        "model_id": "yolo11l", "setup_id": "deepx_setup",
        "remote_suite": str(root), "dxnn_path": str(dxnn), "image_path": str(image),
        "expected_dxnn_sha256": hashlib.sha256(dxnn.read_bytes()).hexdigest(),
        "runtime_venv": str(tmp_path / "venv"),
    }
    return root, request


def inventory(path):
    return {str(p.relative_to(path)): p.read_bytes() for p in path.rglob("*") if p.is_file()}


def test_short_worker_process_keeps_original_suite_and_error(original, tmp_path):
    root, request = original
    before = inventory(root)
    work = tmp_path / "probe"
    work.mkdir()
    write_json(work / "probe_request.json", request)
    (work / "native_detection_postprocess.py").write_text("DIAGNOSTIC = True\n")
    (work / "benchmark_suite.py.txt").write_text('''
def run_deepx_output_value_probe(root, dxnn, run, args, results):
    from splitpoint_runners import native_detection_postprocess as pp
    assert pp.DIAGNOSTIC
    assert args.runs == 1 and args.warmup == 0
    assert args.prepared_feed_image.endswith("000000212226.jpg")
    assert args.quality_evidence_model_id == "yolo11l"
    assert args.prepared_input_manifest == ""
    return {"status": "runtime_failed", "error": "FrozenPostprocessError: decoded_pre_nms_values_invalid",
            "diagnostic_only": True, "counts_as_benchmark": False,
            "output_value_diagnostics": {"score_above_one_count": 1}}
''')
    process = subprocess.run([sys.executable, "-I", "-B", str(Path(worker.__file__).resolve()),
                              "--request", str(work / "probe_request.json"), "--runtime"],
                             capture_output=True, text=True, timeout=20)
    assert process.returncode == 0, process.stdout + process.stderr
    result = json.loads((work / "results/deepx_output_value_probe.json").read_text())
    assert result["error"].endswith("decoded_pre_nms_values_invalid")
    assert result["probe_source"]["dxnn_sha256"] == request["expected_dxnn_sha256"]
    assert result["probe_source"]["compiler_invoked"] is False
    assert inventory(root) == before
    assert (work / "results/probe_input.jpg").read_bytes() == Path(request["image_path"]).read_bytes()


def test_worker_rejects_changed_original_model_before_loading_runtime(original, tmp_path):
    root, request = original
    Path(request["dxnn_path"]).write_bytes(b"different model")
    with pytest.raises(ValueError, match="original_dxnn_sha256_mismatch"):
        worker.run_probe(request, tmp_path)
    assert not (tmp_path / "runtime").exists()


def management_run(tmp_path, remote_suite="/home/nx/splitpoint_runs/old/1/suite"):
    root = tmp_path / "run"
    base = root / "models/yolo11l"
    write_json(base / "benchmark_results/benchmark_results_deepx_m1_full_auto.json", [{
        "run_id": "deepx_m1_full", "backend": "deepx_m1", "variant": "full",
        "dxnn_path": remote_suite + "/deepx/deepx_m1/full/model.dxnn",
        "deepx_prepared_feed_benchmark": {"image": remote_suite + "/resources/000000212226.jpg"},
        "error_detail": "FrozenPostprocessError: decoded_pre_nms_values_invalid",
    }])
    suite = base / "benchmark_set/legacy_suite"
    write_json(suite / "deepx/deepx_m1/full/output_contract.json", {
        "model_id": "yolo11l", "endpoint_mode": "decoded_pre_nms", "artifact_sha256": "a" * 64})
    write_json(suite / "benchmark_plan.json", {"runs": [{"id": "deepx_m1_full", "setup_id": "chosen"}]})
    write_json(root / "hardware_matrix.json", {"hardware_targets": [
        {"id": "other", "accelerator": "deepx_m1", "runtime": {"host": "bad"}},
        {"id": "chosen", "accelerator": "deepx_m1", "runtime": {
            "host": "192.168.0.102", "user": "nx", "port": 2222,
            "remote_venv": "source ~/venvs/deepx-runtime/bin/activate"}}]})
    return root


def test_plan_reuses_exact_failure_image_and_setup_not_first_target(tmp_path):
    root = management_run(tmp_path)
    request, remote = driver.resolve_request(root)
    assert request["image_path"].endswith("000000212226.jpg")
    assert request["setup_id"] == "chosen"
    assert request["runtime_venv"] == "~/venvs/deepx-runtime"
    assert remote == {"host": "192.168.0.102", "user": "nx", "port": 2222}
    assert request["remote_suite"] == "/home/nx/splitpoint_runs/old/1/suite"


def test_plan_only_never_contacts_hardware(tmp_path, monkeypatch):
    root = management_run(tmp_path)
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: pytest.fail("SSH must not run"))
    assert driver.main(["--run-dir", str(root), "--plan-only"]) == 0


def test_invalid_collected_json_is_failure_and_remote_copy_is_retained(tmp_path, monkeypatch):
    root = management_run(tmp_path)
    output = tmp_path / "downloads"
    calls = []
    monkeypatch.setattr(driver.shutil, "which", lambda name: "/usr/bin/" + name)
    def fake_run(command, **kwargs):
        calls.append(command)
        if command[0] == "ssh" and command[-1].startswith("mktemp"):
            return subprocess.CompletedProcess(command, 0, "/tmp/onnx-v27927-full-probe-1234567890\n", "")
        if command[0] == "scp" and "-r" in command:
            target = Path(command[-1])
            target.mkdir(parents=True)
            (target / "deepx_output_value_probe.json").write_text("not valid JSON")
        return subprocess.CompletedProcess(command, 0, "", "")
    monkeypatch.setattr(subprocess, "run", fake_run)
    assert driver.main(["--run-dir", str(root), "--output-dir", str(output)]) == 2
    assert not any(command[-1].startswith("rm ") for command in calls)
    archives = list(output.glob("*.zip"))
    assert len(archives) == 1
    import zipfile
    with zipfile.ZipFile(archives[0]) as archive:
        result = json.loads(archive.read("collection_summary.json"))
        assert result["status"] == "collection_failed"
        assert result["remote_probe_dir"].endswith("1234567890")


def test_unknown_failing_image_is_not_replaced_by_split_image(tmp_path):
    root = management_run(tmp_path)
    result = root / "models/yolo11l/benchmark_results/benchmark_results_deepx_m1_full_auto.json"
    rows = json.loads(result.read_text())
    rows[0]["deepx_prepared_feed_benchmark"]["image"] = ""
    write_json(result, rows)
    with pytest.raises(ValueError, match="original_prepared_feed_image_missing"):
        driver.resolve_request(root)


@pytest.mark.parametrize("path", ["/tmp", "/home/nx", "/tmp/onnx-v27927-full-probe-abc/../x",
                                 "/tmp/onnx-v27927-full-probe-1234567890;echo bad"])
def test_cleanup_only_accepts_own_allocated_directory(path):
    assert not driver.remote_temp_valid(path)


def test_collection_archive_includes_readable_failure_and_tensor(tmp_path):
    output = tmp_path / "probe"
    output.mkdir()
    (output / "raw_output.npz").write_bytes(b"sample")
    write_json(output / "result.json", {"status": "runtime_failed"})
    archive = driver.make_archive(output)
    import zipfile
    with zipfile.ZipFile(archive) as bundle:
        assert set(bundle.namelist()) == {"raw_output.npz", "result.json"}
        assert bundle.testzip() is None


def test_normal_debug_pack_retains_full_origin_and_prepared_input_contract(tmp_path):
    from onnx_splitpoint_tool.workflow.debug_pack import create_evaluation_debug_pack
    import zipfile
    run = tmp_path / "run"
    write_json(run / "run_manifest.json", {"status": "failed"})
    (run / "evaluation_workflow.log").write_text("failed\n")
    prefix = "models/yolo11l/benchmark_set/legacy_suite/results/deepx_m1_full/"
    origin = prefix + "original_full_failure.json"
    prepared = prefix + "prepared_input/native_full_input_manifest.json"
    write_json(run / origin, {"original_full_error": "decoded_pre_nms_values_invalid"})
    write_json(run / prepared, {"shape": [640, 640, 3]})
    (run / prefix / "prepared_input/input.npy").write_bytes(b"tensor not part of the ordinary diagnostic pack")
    archive = tmp_path / "debug.zip"
    create_evaluation_debug_pack(run, archive)
    with zipfile.ZipFile(archive) as bundle:
        assert origin in bundle.namelist()
        assert prepared in bundle.namelist()
        assert prefix + "prepared_input/input.npy" not in bundle.namelist()
