from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from onnx_splitpoint_tool.benchmark.remote_run import _remote_result_collect_script
from onnx_splitpoint_tool.runners.native_full_input import prepare_and_seal_deepx_native_full_input
from onnx_splitpoint_tool.workflow.execution_binding import _materialize_deepx_prepared_input
from onnx_splitpoint_tool.workflow.native_transfer import build_native_transfer_inventory
from scripts import native_full_baseline_eval_runner as runner


MODEL = "yolo11l"
SETUP = "orin_nx_deepx_m1_01"
ERROR = "FrozenPostprocessError: decoded_pre_nms_values_invalid"


def _failure_row():
    return {
        "run_id": "deepx_m1_full", "backend": "deepx_m1", "variant": "full",
        "model_id": MODEL, "setup_id": SETUP, "runtime_ok": False,
        "deepx_prepared_feed_benchmark": {"status": "runtime_failed", "error": ERROR},
    }


def _download(tmp_path, row):
    remote_suite = tmp_path / "remote_suite"
    remote_suite.mkdir()
    (remote_suite / "benchmark_results_deepx_m1_full_auto.json").write_text(json.dumps([row]))
    local_run = tmp_path / "downloaded_run"
    subprocess.run(["bash", "-lc", _remote_result_collect_script(
        remote_results_dir=str(local_run / "results"), remote_suite_dir=str(remote_suite),
    )], check=True, capture_output=True)
    return remote_suite, local_run


def _stage_inventory(suite, staged):
    inventory = build_native_transfer_inventory(suite)
    for relative in inventory["relative_paths"]:
        destination = staged / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(suite / relative, destination)
    return inventory


def _ns():
    return SimpleNamespace(setup_id=SETUP, comparison_backend="deepx", repetitions=3,
                           comparison_precision="uint8_cast_fp16", image_map_data={})


def test_original_full_failure_survives_actual_lean_transfer_without_prepared_input(tmp_path, monkeypatch):
    _remote, local_run = _download(tmp_path, _failure_row())
    suite = tmp_path / "models" / MODEL / "benchmark_set" / "legacy_suite"
    result = _materialize_deepx_prepared_input(
        remote_local_run_dir=local_run, suite_dir=suite,
        result_dir=tmp_path / "results", model_id=MODEL, target_id=SETUP,
    )
    assert result["status"] == "blocked_by_original_full_failure"
    assert result["prepared_input_admitted"] is False
    staged = tmp_path / "native_fifo_evalsets" / MODEL / "benchmark_set"
    inventory = _stage_inventory(suite, staged)
    assert "results/deepx_m1_full/original_full_failure.json" in inventory["relative_paths"]
    assert not any(Path(path).name.startswith("benchmark_results") for path in inventory["relative_paths"])
    assert not list(staged.rglob("native_full_input_manifest.json"))
    monkeypatch.setattr(runner, "_first_case", lambda *_: ("b003", staged, staged / "unused.py"))
    monkeypatch.setattr(runner, "_resolve_image", lambda *_: (staged / "image.jpg", "fixture"))
    monkeypatch.setattr(runner, "_run", lambda *_args, **_kwargs: pytest.fail("No performance child may start"))
    _semantic, blocked = runner._deepx_full_series_preflight(staged, MODEL, _ns())
    assert blocked["status"] == "blocked_before_repetitions"
    assert blocked["failure_reason"] == "deepx_shared_prepared_input_manifest_missing"
    assert blocked["original_full_error"] == ERROR
    assert blocked["preparation_count_attempted"] == 1
    assert blocked["repetition_count_attempted"] == blocked["repetition_count_valid"] == 0
    assert blocked["repetition_count_requested"] == 3
    assert blocked["repetition_records"] == []
    assert blocked["repetition_runtime_instance_ids"] == []
    assert "returncode" not in blocked and "timed_out" not in blocked
    assert blocked["runtime_success"] is False and blocked["fps_makespan"] is None


@pytest.mark.parametrize("mutation", ["model", "setup", "run", "success"])
def test_failure_transport_never_borrows_another_identity_or_success(tmp_path, mutation):
    row = _failure_row()
    if mutation == "model":
        row["model_id"] = "yolo26m"
    elif mutation == "setup":
        row["setup_id"] = "other_deepx_host"
    elif mutation == "run":
        row["run_id"] = "deepx_m1_to_tensorrt"
    else:
        row["runtime_ok"] = True
    _remote, local_run = _download(tmp_path, row)
    suite = tmp_path / "suite"
    _materialize_deepx_prepared_input(remote_local_run_dir=local_run, suite_dir=suite,
                                     result_dir=tmp_path / "results", model_id=MODEL, target_id=SETUP)
    assert not list(suite.rglob("original_full_failure.json"))


def test_full_input_does_not_borrow_split_float32_layout_tensor(tmp_path, monkeypatch):
    suite = tmp_path / "benchmark_set"
    split_input = (suite / "native_pipeline/b003/deepx_to_trt/float32_layout_fp16"
                   / "native_fifo_boundary/native_fifo_boundary_manifest.json")
    split_input.parent.mkdir(parents=True)
    split_input.write_text(json.dumps({"input_image": "000000000632.jpg", "input_dtype": "uint8"}))
    monkeypatch.setattr(runner, "_first_case", lambda *_: ("b003", suite, suite / "unused.py"))
    monkeypatch.setattr(runner, "_resolve_image", lambda *_: (suite / "000000000632.jpg", "fixture"))
    _semantic, blocked = runner._deepx_full_series_preflight(suite, MODEL, _ns())
    assert blocked["failure_reason"] == "deepx_shared_prepared_input_manifest_missing"
    assert blocked["repetition_count_attempted"] == 0
    assert not (suite / "results/deepx_m1_full/prepared_input").exists()


@pytest.mark.parametrize("runtime_failed", [False, True])
def test_full_input_admission_after_collect_materialize_and_stage(tmp_path, runtime_failed):
    remote = tmp_path / "remote_suite"
    image = remote / "resources/validation/000000000632.jpg"
    image.parent.mkdir(parents=True)
    Image.new("RGB", (7, 3), color=(11, 22, 33)).save(image)
    prepared = remote / "results/deepx_m1_full/prepared_input"
    sealed = prepare_and_seal_deepx_native_full_input(
        image_path=image, input_contract={"input": {"name": "images", "shape": [1, 4, 4, 3],
            "dtype": "uint8", "layout": "NHWC", "color_space": "RGB",
            "normalization": "embedded_dxcom_preprocessing", "preprocess_mode": "letterbox",
            "letterbox_pad_value": 114}}, task="detection", out_dir=prepared,
        model=MODEL, setup_id=SETUP, comparison_backend="deepx",
    )
    manifest = sealed["payload"]
    row = _failure_row()
    row.update(runtime_ok=True, task="detection")
    row["deepx_prepared_feed_benchmark"] = {
        "status": "ok", "task": "detection", "prepared_input_binding_verified": True,
        "prepared_input_manifest_sha256": sealed["manifest_sha256"],
        "prepared_input_sha256": manifest["runtime_input_sha256"],
        "prepared_input_file_sha256": manifest["runtime_input_sha256"],
        "prepared_input_bytes": manifest["runtime_input_bytes"],
        "prepared_input_source_image_sha256": manifest["input_image_sha256"],
        "input_contract": {"model_id": MODEL},
    }
    if runtime_failed:
        row["runtime_ok"] = False
        row["deepx_prepared_feed_benchmark"].update(status="runtime_failed", error=ERROR)
    (remote / "benchmark_results_deepx_m1_full_auto.json").write_text(json.dumps([row]))
    local_run = tmp_path / "downloaded_run"
    subprocess.run(["bash", "-lc", _remote_result_collect_script(
        remote_results_dir=str(local_run / "results"), remote_suite_dir=str(remote),
    )], check=True, capture_output=True)
    suite = tmp_path / "local_suite"
    admission = _materialize_deepx_prepared_input(
        remote_local_run_dir=local_run, suite_dir=suite,
        result_dir=tmp_path / "results", model_id=MODEL, target_id=SETUP,
    )
    staged = tmp_path / "staged_suite"
    inventory = _stage_inventory(suite, staged)
    if runtime_failed:
        assert admission["status"] == "blocked_by_original_full_failure"
        assert admission["original_full_error"] == ERROR
        assert admission["prepared_input_admitted"] is False
        assert not (suite / "results/deepx_m1_full/prepared_input").exists()
        assert inventory["relative_paths"] == ["results/deepx_m1_full/original_full_failure.json"]
        context = runner._deepx_original_full_failure(staged, MODEL, "deepx_m1_full", setup_id=SETUP)
        assert context["original_full_error"] == ERROR
        return
    assert admission["status"] == "verified_exact"
    for role in ["runtime_input.bin", "input_rgb_uint8.bin", "native_full_input_manifest.json"]:
        relative = "results/deepx_m1_full/prepared_input/" + role
        assert relative in inventory["relative_paths"]
        assert (staged / relative).read_bytes() == (prepared / role).read_bytes()
    assert not list(staged.rglob("original_full_failure.json"))
    checked, status = runner._validated_runtime_input_manifest(
        staged / "results/deepx_m1_full/prepared_input/native_full_input_manifest.json",
        allowed_root=staged / "results/deepx_m1_full/prepared_input",
    )
    assert checked is not None, status


def test_semantic_preparation_is_reused_across_three_runtime_repetitions(tmp_path, monkeypatch):
    # Detection preparation now requires a genuine completed NMS proof. Use the
    # real semantic/suite adapters with only their physical process substituted.
    from test_v27930_native_full_semantic_merge import prepare_runner_case
    case = prepare_runner_case(tmp_path, monkeypatch)
    prepared, blocked = runner._deepx_full_series_preflight(case.root, case.model, case.ns)
    assert blocked is None
    case.ns.deepx_full_precomputed_semantic = prepared
    for _ in range(3):
        assert runner._row_for_backend(case.root, case.model, "deepx", case.ns)["runtime_success"] is True
    assert case.processes == ["semantic", "performance", "performance", "performance"]


def test_main_exports_blocked_preparation_as_zero_performance_attempts(tmp_path, monkeypatch):
    suite = tmp_path / MODEL / "benchmark_set"
    suite.mkdir(parents=True)
    (suite / "benchmark_set.json").write_text("{}")
    monkeypatch.setattr(runner, "_select_engine_python", lambda *_: ("", {}))
    monkeypatch.setattr(runner, "_site_packages_for_python", lambda *_: [])
    monkeypatch.setattr(runner, "_first_case", lambda *_: ("b003", suite, suite / "unused.py"))
    monkeypatch.setattr(runner, "_resolve_image", lambda *_: (suite / "image.jpg", "fixture"))
    monkeypatch.setattr(runner, "_row_for_backend", lambda *_: pytest.fail("No repetition may be invoked"))
    monkeypatch.setattr(runner.sys, "argv", ["native_full_baseline_eval_runner.py", "--root", str(tmp_path),
                        "--models", MODEL, "--backends", "deepx", "--setup-id", SETUP,
                        "--comparison-backend", "deepx", "--comparison-precision", "uint8_cast_fp16",
                        "--repetitions", "3"])
    assert runner.main() == 3
    payload = json.loads((tmp_path / "analysis_tables/native_full_baseline_eval.json").read_text())
    row = payload["rows"][0]
    assert row["status"] == "blocked_before_repetitions"
    assert row["failure_reason"] == "deepx_shared_prepared_input_manifest_missing"
    assert row["repetition_count_attempted"] == 0
    assert row["preparation_count_attempted"] == 1
    assert row["repetition_count_requested"] == 3
    assert row["comparison_precision"] == "uint8_cast_fp16"
    assert row["full_runtime_precision"] == ""
