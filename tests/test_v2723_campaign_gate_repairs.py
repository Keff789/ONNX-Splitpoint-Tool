from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys
from typing import Any

import numpy as np
import pytest

from onnx_splitpoint_tool.workflow.runner import EvaluationWorkflowRunner, WorkflowOptions
from onnx_splitpoint_tool.workflow.execution_binding import (
    _copy_remote_result_files,
)
from onnx_splitpoint_tool.workflow.start_snapshot import (
    build_profile_start_snapshot,
    validate_profile_start_snapshot,
)
from tests.test_v269e_start_snapshot_backfill import (
    _source_profile,
    _start_snapshot,
)


ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str) -> Any:
    path = ROOT / "scripts" / name
    module_name = f"_v2723_{path.stem}_{id(path)}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_requested_selection_gets_its_own_canonical_hash() -> None:
    source = _source_profile(native=True, energy=True)
    assert source["execution_preset"].get("snapshot_sha256", "") == ""

    snapshot = _start_snapshot(source)
    requested = snapshot["requested_selection"]
    resolved = snapshot["resolved_selection"]

    assert requested["snapshot_sha256"] == ""
    assert requested["selection_snapshot_sha256"].startswith("sha256:")
    assert len(requested["selection_snapshot_sha256"]) == 71
    assert resolved["selection_snapshot_sha256"].startswith("sha256:")
    assert len(resolved["selection_snapshot_sha256"]) == 71


def test_legacy_v1_start_snapshot_rebuild_keeps_its_original_hash_domain(
) -> None:
    current = _start_snapshot(_source_profile(native=True, energy=True))
    assert current["schema_version"] == 2

    legacy = build_profile_start_snapshot(
        profile_request=str(current["profile_request"]),
        source_profile=current["source_profile"],
        resolved_profile=current["resolved_profile"],
        profile_id=str(current["profile_id"]),
        profile_path=str(current["profile_path"]),
        profile_source=str(current["profile_source"]),
        runtime_bindings=current["runtime_bindings"],
        schema_version=1,
    )

    assert legacy["schema_version"] == 1
    assert "selection_snapshot_sha256" not in legacy["requested_selection"]
    assert "selection_snapshot_sha256" not in legacy["resolved_selection"]
    assert validate_profile_start_snapshot(legacy) == legacy


def test_quality_request_setup_is_recovered_from_both_archive_layouts(
    tmp_path: Path,
) -> None:
    canonical = (
        tmp_path / "quality_inputs/orin_nx_hailo8_01/results_native_full/"
        "b038/task_quality_inputs/full_request.json"
    )
    remote = (
        tmp_path / "remote_diagnostics/orin_nx_hailo8_01/results_native_full/"
        "b038/task_quality_inputs/full_request.json"
    )
    assert EvaluationWorkflowRunner._quality_request_setup_id(canonical) == (
        "orin_nx_hailo8_01"
    )
    assert EvaluationWorkflowRunner._quality_request_setup_id(remote) == (
        "orin_nx_hailo8_01"
    )


def test_central_queue_prefers_canonical_quality_tree_over_remote_mirror(
    tmp_path: Path,
) -> None:
    root = (
        tmp_path / "models/yolo26s/benchmark_results"
    )
    canonical = (
        root / "quality_inputs/orin_nx_hailo8_01/results_native_full/"
        "b038/task_quality_inputs/full_request.json"
    )
    mirror = (
        root / "remote_diagnostics/orin_nx_hailo8_01/results_native_full/"
        "b038/task_quality_inputs/full_request.json"
    )
    remote_only = (
        root / "remote_diagnostics/orin_nx_hailo8_01/results_split/"
        "b039/task_quality_inputs/split_request.json"
    )
    for path in (canonical, mirror, remote_only):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")

    submitted: list[Path] = []

    class _Executor:
        def submit(self, _callable: Any, _model_id: str, path: Path) -> object:
            submitted.append(path)
            return object()

    runner = EvaluationWorkflowRunner(WorkflowOptions(profile="", out=str(tmp_path)))
    runner.profile_payload = {
        "quality_gate": {
            "statistics": {"execution_location": "central_management"}
        }
    }
    runner.run_dir = tmp_path
    runner._central_quality_futures = {}
    runner._central_quality_coord_executor = _Executor()
    runner._ensure_central_quality_service = lambda: None
    runner._emit_log = lambda _message: None

    assert runner._queue_central_quality_requests("yolo26s") == 2
    assert submitted == [canonical, remote_only]


def test_central_queue_keeps_conflicting_mirror_for_fail_closed_merge(
    tmp_path: Path,
) -> None:
    root = tmp_path / "models/yolo26s/benchmark_results"
    canonical = (
        root / "quality_inputs/orin_nx_hailo8_01/results_native_full/"
        "b038/task_quality_inputs/full_request.json"
    )
    mirror = (
        root / "remote_diagnostics/orin_nx_hailo8_01/results_native_full/"
        "b038/task_quality_inputs/full_request.json"
    )
    canonical.parent.mkdir(parents=True, exist_ok=True)
    mirror.parent.mkdir(parents=True, exist_ok=True)
    canonical.write_text('{"request": "canonical"}', encoding="utf-8")
    mirror.write_text('{"request": "drifted"}', encoding="utf-8")

    submitted: list[Path] = []

    class _Executor:
        def submit(self, _callable: Any, _model_id: str, path: Path) -> object:
            submitted.append(path)
            return object()

    runner = EvaluationWorkflowRunner(WorkflowOptions(profile="", out=str(tmp_path)))
    runner.profile_payload = {
        "quality_gate": {
            "statistics": {"execution_location": "central_management"}
        }
    }
    runner.run_dir = tmp_path
    runner._central_quality_futures = {}
    runner._central_quality_coord_executor = _Executor()
    runner._ensure_central_quality_service = lambda: None
    runner._emit_log = lambda _message: None

    assert runner._queue_central_quality_requests("yolo26s") == 2
    assert set(submitted) == {canonical, mirror}


def test_central_queue_ignores_exact_unscoped_remote_run_staging_copy(
    tmp_path: Path,
) -> None:
    root = tmp_path / "models/yolo26s/benchmark_results"
    remote = root / "remote_runs/eval_yolo26s_orin_nx_hailo8_01"
    staging = (
        remote / "results/b038/"
        "results_hailo8_to_trt/task_quality_inputs/composed_request.json"
    )
    staging.parent.mkdir(parents=True, exist_ok=True)
    staging.write_text('{"same": "signed-request"}', encoding="utf-8")
    _copy_remote_result_files(
        remote, root, flat_prefix="orin_nx_hailo8_01",
    )
    canonical = (
        root / "quality_inputs/orin_nx_hailo8_01/results/b038/"
        "results_hailo8_to_trt/task_quality_inputs/composed_request.json"
    )
    assert canonical.is_file()

    submitted: list[Path] = []

    class _Executor:
        def submit(self, _callable: Any, _model_id: str, path: Path) -> object:
            submitted.append(path)
            return object()

    runner = EvaluationWorkflowRunner(WorkflowOptions(profile="", out=str(tmp_path)))
    runner.profile_payload = {
        "quality_gate": {
            "statistics": {"execution_location": "central_management"}
        }
    }
    runner.run_dir = tmp_path
    runner._central_quality_futures = {}
    runner._central_quality_coord_executor = _Executor()
    runner._ensure_central_quality_service = lambda: None
    runner._emit_log = lambda _message: None

    assert runner._queue_central_quality_requests("yolo26s") == 1
    assert submitted == [canonical]


def test_central_queue_keeps_drifted_remote_run_staging_copy_fail_closed(
    tmp_path: Path,
) -> None:
    root = tmp_path / "models/yolo26s/benchmark_results"
    remote = root / "remote_runs/eval_yolo26s_orin_nx_hailo8_01"
    staging = (
        remote / "results/b038/"
        "results_hailo8/task_quality_inputs/full_request.json"
    )
    staging.parent.mkdir(parents=True, exist_ok=True)
    staging.write_text('{"request": "canonical"}', encoding="utf-8")
    _copy_remote_result_files(
        remote, root, flat_prefix="orin_nx_hailo8_01",
    )
    canonical = (
        root / "quality_inputs/orin_nx_hailo8_01/results/b038/"
        "results_hailo8/task_quality_inputs/full_request.json"
    )
    assert canonical.is_file()
    staging.write_text('{"request": "drifted"}', encoding="utf-8")

    submitted: list[Path] = []

    class _Executor:
        def submit(self, _callable: Any, _model_id: str, path: Path) -> object:
            submitted.append(path)
            return object()

    runner = EvaluationWorkflowRunner(WorkflowOptions(profile="", out=str(tmp_path)))
    runner.profile_payload = {
        "quality_gate": {
            "statistics": {"execution_location": "central_management"}
        }
    }
    runner.run_dir = tmp_path
    runner._central_quality_futures = {}
    runner._central_quality_coord_executor = _Executor()
    runner._ensure_central_quality_service = lambda: None
    runner._emit_log = lambda _message: None

    assert runner._queue_central_quality_requests("yolo26s") == 2
    assert set(submitted) == {canonical, staging}


def test_central_queue_keeps_remote_only_unscoped_staging_request(
    tmp_path: Path,
) -> None:
    root = tmp_path / "models/yolo26s/benchmark_results"
    staging = (
        root / "remote_runs/eval_yolo26s_orin_nx_hailo8_01/results/b038/"
        "results_hailo8/task_quality_inputs/full_request.json"
    )
    staging.parent.mkdir(parents=True, exist_ok=True)
    staging.write_text('{"remote": "only"}', encoding="utf-8")

    submitted: list[Path] = []

    class _Executor:
        def submit(self, _callable: Any, _model_id: str, path: Path) -> object:
            submitted.append(path)
            return object()

    runner = EvaluationWorkflowRunner(WorkflowOptions(profile="", out=str(tmp_path)))
    runner.profile_payload = {
        "quality_gate": {
            "statistics": {"execution_location": "central_management"}
        }
    }
    runner.run_dir = tmp_path
    runner._central_quality_futures = {}
    runner._central_quality_coord_executor = _Executor()
    runner._ensure_central_quality_service = lambda: None
    runner._emit_log = lambda _message: None

    assert runner._queue_central_quality_requests("yolo26s") == 1
    assert submitted == [staging]


def test_central_queue_keeps_unproven_cross_setup_staging_match(
    tmp_path: Path,
) -> None:
    root = tmp_path / "models/yolo26s/benchmark_results"
    canonical = (
        root / "quality_inputs/orin_nx_hailo8_01/results/b038/"
        "results_hailo8/task_quality_inputs/full_request.json"
    )
    unproven_other_setup = (
        root / "remote_runs/eval_yolo26s_orin_nx_hailo8_02/results/b038/"
        "results_hailo8/task_quality_inputs/full_request.json"
    )
    for path in (canonical, unproven_other_setup):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"same": "signed-request"}', encoding="utf-8")

    submitted: list[Path] = []

    class _Executor:
        def submit(self, _callable: Any, _model_id: str, path: Path) -> object:
            submitted.append(path)
            return object()

    runner = EvaluationWorkflowRunner(WorkflowOptions(profile="", out=str(tmp_path)))
    runner.profile_payload = {
        "quality_gate": {
            "statistics": {"execution_location": "central_management"}
        }
    }
    runner.run_dir = tmp_path
    runner._central_quality_futures = {}
    runner._central_quality_coord_executor = _Executor()
    runner._ensure_central_quality_service = lambda: None
    runner._emit_log = lambda _message: None

    assert runner._queue_central_quality_requests("yolo26s") == 2
    assert set(submitted) == {canonical, unproven_other_setup}


def test_central_queue_preserves_two_manifest_bound_setup_scopes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "models/yolo26s/benchmark_results"
    setup_ids = ("orin_nx_hailo8_01", "orin_nx_hailo8_02")
    canonical_paths: list[Path] = []
    raw_paths: list[Path] = []
    for setup_id in setup_ids:
        remote = root / "remote_runs" / f"eval_yolo26s_{setup_id}"
        raw = (
            remote / "results/b038/results_hailo8/"
            "task_quality_inputs/full_request.json"
        )
        raw.parent.mkdir(parents=True, exist_ok=True)
        raw.write_text('{"same": "signed-request"}', encoding="utf-8")
        _copy_remote_result_files(remote, root, flat_prefix=setup_id)
        canonical = (
            root / "quality_inputs" / setup_id / "results/b038/"
            "results_hailo8/task_quality_inputs/full_request.json"
        )
        assert canonical.is_file()
        raw_paths.append(raw)
        canonical_paths.append(canonical)

    submitted: list[Path] = []

    class _Executor:
        def submit(self, _callable: Any, _model_id: str, path: Path) -> object:
            submitted.append(path)
            return object()

    runner = EvaluationWorkflowRunner(WorkflowOptions(profile="", out=str(tmp_path)))
    runner.profile_payload = {
        "quality_gate": {
            "statistics": {"execution_location": "central_management"}
        }
    }
    runner.run_dir = tmp_path
    runner._central_quality_futures = {}
    runner._central_quality_coord_executor = _Executor()
    runner._ensure_central_quality_service = lambda: None
    runner._emit_log = lambda _message: None

    assert runner._queue_central_quality_requests("yolo26s") == 2
    assert set(submitted) == set(canonical_paths)
    assert not set(submitted).intersection(raw_paths)


def test_trt_central_result_mirrors_deduplicate_by_semantic_result() -> None:
    validator = _load_script("native_producer_validate_visualize.py")
    base = {
        "status": "completed",
        "technical_status": "completed",
        "scientific_status": "pass",
        "decision": "pass",
        "evaluation_fingerprint": "e" * 64,
        "reference_identity": "r" * 64,
        "n": 12,
        "primary": {"metric": "coco_ap_50_95", "decision": "pass"},
        "guardrails": {"ap50": {"decision": "pass"}},
        "producer_binding_eligible": True,
        "source_request_sha256": "1" * 64,
        "source_request": "quality_inputs/setup/request.json",
        "generated_at": "first-mirror",
    }
    mirror = {
        **base,
        "source_request": "remote_diagnostics/setup/request.json",
        "generated_at": "second-mirror",
    }

    first = validator._central_result_mirror_signature(
        base, trt_quality_first=True,
    )
    second = validator._central_result_mirror_signature(
        mirror, trt_quality_first=True,
    )
    assert first == second

    changed = json.loads(json.dumps(mirror))
    changed["primary"]["decision"] = "fail"
    assert validator._central_result_mirror_signature(
        changed, trt_quality_first=True,
    ) != first
    request_drift = {**mirror, "source_request_sha256": "2" * 64}
    assert validator._central_result_mirror_signature(
        request_drift, trt_quality_first=True,
    ) != first
    assert validator._central_result_mirror_signature(
        base, trt_quality_first=False,
    ) != validator._central_result_mirror_signature(
        mirror, trt_quality_first=False,
    )


def test_copied_output_manifest_rebases_only_exact_sibling(
    tmp_path: Path,
) -> None:
    loader = _load_script("validate_output_dumps.py")
    output = tmp_path / "output_000.bin"
    array = np.arange(6, dtype=np.float32).reshape(1, 1, 6)
    output.write_bytes(array.tobytes())
    manifest = tmp_path / "native_fifo_output_manifest.json"
    manifest.write_text(json.dumps({
        "outputs": [{
            "name": "output0",
            "dtype": "float32",
            "shape": [1, 1, 6],
            "file": "/home/nx/old-run/native_fifo_outputs/output_000.bin",
        }],
    }), encoding="utf-8")

    loaded, _payload = loader._load_manifest(manifest)
    np.testing.assert_array_equal(loaded["output0"], array)

    output.rename(tmp_path / "other.bin")
    with pytest.raises(FileNotFoundError, match="output tensor is missing"):
        loader._load_manifest(manifest)


def test_hailo_python_boundary_manifest_is_complete_and_portable(
    tmp_path: Path,
) -> None:
    runner = _load_script("native_hailo_trt_fifo_from_benchmarkset.py")
    image = tmp_path / "image.jpg"
    image.write_bytes(b"fixture-image")
    boundary_dir = tmp_path / "boundary"
    boundary = np.zeros((1, 4, 2, 2), dtype=np.float16)
    args = SimpleNamespace(
        dump_outputs=False,
        dump_boundary=True,
        boundary_dir=str(boundary_dir),
        preprocess_mode_effective="letterbox",
        letterbox_pad_value=114,
        precision="uint8_dequant_fp16",
    )

    emitted = runner._dump_hailo8_python_semantic_evidence(
        work=tmp_path,
        inputs={"images": np.zeros((1, 3, 2, 2), dtype=np.uint8)},
        boundary=boundary,
        boundary_meta={"trt_input_name": "stage2_input"},
        trt_outputs={},
        image=image,
        args=args,
    )
    manifest = Path(emitted["native_fifo_boundary_manifest"])
    payload = json.loads(manifest.read_text(encoding="utf-8"))

    assert payload["trt_input_name"] == "stage2_input"
    assert payload["trt_input_dtype"] == "float16"
    assert payload["trt_input_bytes"] == boundary.nbytes
    assert payload["file"] == "boundary.bin"
    assert payload["input_dump"] == "prepared_input.bin"
    assert {
        row["path"] for row in payload["payload_artifacts"]
    } == {"boundary.bin", "prepared_input.bin"}
    assert all(
        not Path(row["path"]).is_absolute()
        for row in payload["payload_artifacts"]
    )
