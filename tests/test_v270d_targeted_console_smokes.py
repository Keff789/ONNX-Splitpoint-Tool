from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from onnx_splitpoint_tool.hailo_backend import (
    _hailo_cache_key,
    _write_hailo_receipt,
)
from onnx_splitpoint_tool.native_split_quality_authority import (
    SELECTION_FINGERPRINT_WORKFLOW,
    resolve_native_split_quality_authority,
)
from onnx_splitpoint_tool.native_output_endpoint import (
    load_authoritative_output_contract,
)
from onnx_splitpoint_tool.native_detection_postprocess import (
    FrozenDetectionPostprocessor,
    build_frozen_postprocess_contract,
)
from onnx_splitpoint_tool.preprocessing_contract import (
    canonical_image_preprocessing_contract,
    preprocessing_contract_sha256,
)
from onnx_splitpoint_tool.workflow.benchmark_binding import (
    _promote_verified_hailo_full_contracts,
    materialize_backend_artifact_decisions,
)
from onnx_splitpoint_tool.workflow.runner import (
    _native_selection_contract_runs_v270e,
)


ROOT = Path(__file__).resolve().parents[1]


def _seal_test_hailo_hef(
    hef: Path, *, hw_arch: str, end_nodes: list[str] | None = None
) -> tuple[Path, Path]:
    source = hef.parent / "receipt_source.onnx"
    source.write_bytes(b"detection-source-onnx")
    compiler = hef.parent / "receipt_compiler.onnx"
    compiler.write_bytes(b"detection-compiler-fixed-onnx")
    contract = canonical_image_preprocessing_contract("detection", (640, 640))
    cache_key, cache_payload = _hailo_cache_key(
        model_path=compiler,
        activation_part1=None,
        hw_arch=hw_arch,
        opt_level=1,
        calib_dir=None,
        calib_count=64,
        calib_batch_size=8,
        extra_model_script="",
        start_nodes=None,
        end_nodes=end_nodes,
        preprocessing_contract=contract,
    )
    _write_hailo_receipt(
        hef_path=hef,
        source_onnx=source,
        compiler_onnx=compiler,
        hw_arch=hw_arch,
        net_name="receipt_test",
        preprocessing_contract=contract,
        preprocessing_sha256=preprocessing_contract_sha256(contract),
        cache_key=cache_key,
        cache_payload=cache_payload,
        calibration_identity=str(cache_payload["calibration_identity"]),
        calibration_count=int(cache_payload["calibration_count"]),
    )
    return source, compiler


def _load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(
        f"v270d_{path.stem}", path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_source_manifest_excludes_console_smoke_results(
    tmp_path: Path,
) -> None:
    module = _load_script("build_source_manifest.py")
    report = tmp_path / "console_smokes" / "run" / "summary.json"
    report.parent.mkdir(parents=True)
    report.write_text("{}\n", encoding="utf-8")
    assert module._included(report, tmp_path) is False


class _Buffer:
    def __init__(self, array: np.ndarray) -> None:
        self.array = array

    def get_buffer(self) -> np.ndarray:
        return self.array


class _RawSession:
    _hef_output_names = ["hef_output"]
    _output_name_hef_to_canonical = {"hef_output": "canonical_cut"}
    output_shapes = {"canonical_cut": (1, 4, 2, 3)}

    def __init__(self) -> None:
        self.array = np.arange(24, dtype=np.float32).reshape(1, 2, 3, 4)
        self.calls: list[str] = []

    def infer(self, *_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("public canonical infer path must not be used")

    def _prepare_infer_inputs(self, inputs: dict[str, np.ndarray]):
        self.calls.append("prepare")
        return inputs

    def _create_reusable_binding_slot(self):
        self.calls.append("create")
        return {"binding": object()}

    def _fill_reusable_slot_inputs(self, _slot, _inputs) -> None:
        self.calls.append("fill")

    def _submit_reusable_slot(self, _slot, _inputs, *, copy_inputs: bool) -> None:
        assert copy_inputs is False
        self.calls.append("submit")

    def _wait_reusable_slot(self, _slot) -> None:
        self.calls.append("wait")

    def _binding_output(self, _binding, name: str) -> _Buffer:
        assert name == "hef_output"
        return _Buffer(self.array)


class _FakeTRT:
    inputs = ["part2_input"]
    shapes = {"part2_input": (1, 4, 2, 3)}
    dtypes = {"part2_input": np.dtype(np.float32)}


def test_hailo10_semantic_sample_uses_raw_slot_and_strict_shape() -> None:
    module = _load_script("native_hailo10_trt_e2e_from_benchmarkset.py")
    session = _RawSession()
    previous = module._STRICT_SPLIT_BOUNDARY
    previous_evidence = module._STRICT_SPLIT_BOUNDARY_EVIDENCE
    try:
        module._STRICT_SPLIT_BOUNDARY = {
            "name": "canonical_cut",
            "runtime_name": "canonical_cut",
            "shape": [2, 3, 4],
            "dtype": "float32",
            "metadata_sha256": "a" * 64,
        }
        module._STRICT_SPLIT_BOUNDARY_EVIDENCE = {}
        outputs = module._capture_raw_hailo10_sample(
            session, {"input": np.zeros((1,), dtype=np.uint8)},
        )
        name, array = module._pick_hailo_output(outputs, _FakeTRT())
        assert session.calls == ["prepare", "create", "fill", "submit", "wait"]
        assert name == "part2_input"
        assert array.shape == (2, 3, 4)
        assert module._STRICT_SPLIT_BOUNDARY_EVIDENCE["status"] == (
            "exact_runtime_boundary_verified"
        )

        with pytest.raises(
            RuntimeError,
            match="native_split_quality_runtime_boundary_shape_mismatch",
        ):
            module._pick_hailo_output(
                {
                    "canonical_cut": np.zeros(
                        (1, 4, 2, 3), dtype=np.float32,
                    ),
                },
                _FakeTRT(),
            )
    finally:
        module._STRICT_SPLIT_BOUNDARY = previous
        module._STRICT_SPLIT_BOUNDARY_EVIDENCE = previous_evidence


@pytest.mark.parametrize(
    ("raw_shape", "trt_shape"),
    [
        ((28, 28, 512), (1, 512, 28, 28)),
        ((80, 80, 256), (1, 256, 80, 80)),
    ],
)
def test_hailo10_boundary_manifest_separates_raw_and_trt_shapes(
    tmp_path: Path, raw_shape: tuple[int, ...], trt_shape: tuple[int, ...],
) -> None:
    module = _load_script("native_hailo10_trt_e2e_from_benchmarkset.py")

    class TRT:
        inputs = ["cut"]
        shapes = {"cut": trt_shape}
        dtypes = {"cut": np.dtype(np.float32)}

    manifest_path = module._dump_hailo10_boundary(
        tmp_path,
        boundary_name="runtime_cut",
        boundary=np.zeros(raw_shape, dtype=np.float32),
        inputs={"input": np.zeros((16, 16, 3), dtype=np.uint8)},
        trt=TRT(),
        case="b001",
        precision="float32_layout_fp16",
        boundary_layout="memory_nhwc_to_nchw",
    )
    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    assert payload["shape"] == list(raw_shape)
    assert payload["runtime_boundary_shape"] == list(raw_shape)
    assert payload["trt_input_shape"] == list(trt_shape)
    assert payload["boundary_layout"] == "memory_nhwc_to_nchw"
    assert payload["layout_transform_owner"] == "tensorrt_part2_input_bridge"


def test_hailo10_contract_plan_is_selection_derived_and_layout_exact() -> None:
    runs = _native_selection_contract_runs_v270e(
        "hailo10h",
        {
            "resnet50": ["b039", "b052", "b082"],
            "yolo26s": ["b036", "b038"],
            "yolov7_paper": ["b044"],
        },
    )
    assert {run["models"][0] for run in runs} == {
        "resnet50", "yolo26s", "yolov7_paper",
    }
    assert all(
        run["boundary_layout"] == "memory_nhwc_to_nchw"
        and run["precision"] == "uint8_dequant_fp16"
        and run["hailo_format"] == "uint8"
        and run["selection_source"] == "effective_deployment_selection"
        for run in runs
    )
    assert {
        case
        for run in runs
        for cases in run["case_map"].values()
        for case in cases
    } == {"b039", "b052", "b082", "b036", "b038", "b044"}


def _write_authority_context(
    run: Path, *, stage_fingerprint: str | None = "c" * 64,
    manifest_fingerprint: str | None = "c" * 64,
    workflow: str = SELECTION_FINGERPRINT_WORKFLOW,
) -> None:
    (run / "reports").mkdir(parents=True)
    snapshot: dict[str, Any] = {
        "snapshot_sha256": "a" * 64,
        "requested_selection": {"snapshot_sha256": "1" * 64},
        "resolved_selection": {"snapshot_sha256": "2" * 64},
    }
    if manifest_fingerprint is not None:
        snapshot["selection_fingerprint"] = manifest_fingerprint
    manifest = {
        "schema": "onnx-splitpoint/evaluation-run-manifest",
        "run_id": run.name,
        "workflow_version": workflow,
        "current_workflow_version": workflow,
        "current_tool_version": "2.70.6",
        "execution_sessions": [{"workflow_version": workflow}],
        "profile_start_snapshot": snapshot,
    }
    stage: dict[str, Any] = {
        "schema": "onnx-splitpoint/native-producer-stage",
        "run_id": run.name,
        "workflow_version": workflow,
        "profile_start_snapshot_sha256": "a" * 64,
        "profile_selection_snapshot_sha256": "1" * 64,
        "native_split_quality_first": {"required": True},
    }
    if stage_fingerprint is not None:
        stage["profile_selection_fingerprint"] = stage_fingerprint
    (run / "run_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8",
    )
    (run / "reports/native_producer_stage.json").write_text(
        json.dumps(stage), encoding="utf-8",
    )


def _authority(run: Path) -> dict[str, Any]:
    return resolve_native_split_quality_authority(
        run_manifest_path=run / "run_manifest.json",
        stage_path=run / "reports/native_producer_stage.json",
    )


def test_requested_and_resolved_selection_shas_may_differ_with_fingerprint(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    _write_authority_context(run)
    authority = _authority(run)
    assert authority["valid"] is True
    assert authority["profile_requested_selection_snapshot_sha256"] == "1" * 64
    assert authority["profile_resolved_selection_snapshot_sha256"] == "2" * 64
    assert authority["profile_selection_fingerprint"] == "c" * 64
    assert authority["selection_identity_mode"] == (
        "canonical_selection_fingerprint"
    )


@pytest.mark.parametrize(
    ("stage_fingerprint", "manifest_fingerprint", "error"),
    [
        ("d" * 64, "c" * 64, "native_stage_selection_fingerprint_mismatch"),
        (None, "c" * 64, "native_stage_selection_fingerprint_missing"),
        ("c" * 64, None, "run_manifest_selection_fingerprint_missing_or_invalid"),
    ],
)
def test_selection_fingerprint_is_fail_closed(
    tmp_path: Path,
    stage_fingerprint: str | None,
    manifest_fingerprint: str | None,
    error: str,
) -> None:
    run = tmp_path / "run"
    _write_authority_context(
        run,
        stage_fingerprint=stage_fingerprint,
        manifest_fingerprint=manifest_fingerprint,
    )
    authority = _authority(run)
    assert authority["valid"] is False
    assert error in authority["errors"]


def test_v270c_stage_without_explicit_fingerprint_remains_readable(
    tmp_path: Path,
) -> None:
    workflow = "v2.70c-standard-native-postprocess-repair"
    run = tmp_path / "run"
    _write_authority_context(
        run, stage_fingerprint=None, workflow=workflow,
    )
    authority = _authority(run)
    assert authority["valid"] is True
    assert authority["profile_selection_fingerprint"] == "c" * 64


def _raw_suite(tmp_path: Path) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    suite = tmp_path / "suite"
    artifact = suite / "hailo/hailo10/full/compiled.hef"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"verified-yolo26-hailo10-raw-head")
    nodes = [f"/head/{index}" for index in range(6)]
    source_onnx, compiler_onnx = _seal_test_hailo_hef(
        artifact, hw_arch="hailo10h", end_nodes=nodes
    )
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    benchmark = {
        "hailo": {"hefs": {"hailo10": {
            "full": "hailo/hailo10/full/compiled.hef",
            "full_build": {
                "ok": True,
                "artifact_hash": digest,
                "source_onnx_path": str(source_onnx),
                "compiler_onnx_path": str(compiler_onnx),
            },
            "full_endpoint_mode": "raw_detection_head",
            "full_end_node_names": nodes,
            "full_output_contract": {
                "mode": "yolo26_one2one_raw_head",
                "requires_external_postprocess": True,
                "end_node_names": nodes,
            },
        }}},
    }
    contracts = [{
        "schema": "onnx-splitpoint/output-contract",
        "model_id": "yolo26s",
        "task": "detection",
        "backend": "hailo10",
        "variant": "full",
        "endpoint_mode": "decoded",
        "host_tail_required": False,
        "postprocessing_required": False,
        "full_end_node_names": [],
    }]
    return suite, benchmark, contracts


def test_verified_hailo10_suite_raw_head_reconciles_planned_decoded_contract(
    tmp_path: Path,
) -> None:
    suite, benchmark, contracts = _raw_suite(tmp_path)
    promotions = _promote_verified_hailo_full_contracts(
        suite_dir=suite,
        model_id="yolo26s",
        task="detection",
        suite_bench=benchmark,
        contracts=contracts,
        copied_verified={},
    )
    assert len(promotions) == 1
    contract = contracts[0]
    assert contract["endpoint_mode"] == "raw_detection_head"
    assert contract["host_tail_required"] is True
    assert contract["postprocessing_required"] is True
    assert len(contract["full_end_node_names"]) == 6
    assert contract["artifact_binding_status"] == "verified"
    assert contract["contract_reconciliation_status"] == (
        "verified_suite_artifact_raw_head"
    )
    (suite / "output_contracts.json").write_text(json.dumps({
        "model_id": "yolo26s",
        "task": "detection",
        "contracts": contracts,
    }), encoding="utf-8")
    declaration = load_authoritative_output_contract(
        suite, backend="native_full_hailo10h",
        model_id="yolo26s", task="detection",
    )
    assert declaration["contract_resolution_status"] == "attested"
    assert declaration["stage"] == "raw_head"
    runner = _load_script("smoke_hailo10_hef_runner.py")
    outputs = {
        f"head_{index}": np.zeros((1, channels, side, side), dtype=np.float32)
        for index, (channels, side) in enumerate([
            (4, 80), (80, 80), (4, 40),
            (80, 40), (4, 20), (80, 20),
        ])
    }
    raw_endpoint = runner._output_contract(
        "detection", outputs, declaration,
    )
    assert raw_endpoint["endpoint_contract_complete"] is True
    assert raw_endpoint["contract_family"] == "raw_head"
    frozen = build_frozen_postprocess_contract(
        model_id="yolo26s", outputs=outputs,
        input_hw=[640, 640], original_wh=[640, 480],
    )
    processor = FrozenDetectionPostprocessor(frozen)
    frozen_result = processor.process(outputs, original_wh=[640, 480])
    image = tmp_path / "image.jpg"
    image.write_bytes(b"diagnostic-image-identity")
    manifest_path, _ = runner._write_output_dump(
        outputs, tmp_path / "online_dump",
        backend="native_full_hailo10h", model="yolo26s",
        setup_id="hailo10_setup",
        comparison_backend="hailo10h", task="detection",
        image=image,
        input_hwc=np.zeros((640, 640, 3), dtype=np.uint8),
        input_tensor=np.zeros((640, 640, 3), dtype=np.uint8),
        input_name="input",
        preprocess={"mode": "letterbox_rgb_uint8"},
        declared_output_contract=declaration,
        frozen_postprocess_contract=frozen,
        frozen_postprocess_result=frozen_result,
    )
    online_manifest = json.loads(
        manifest_path.read_text(encoding="utf-8"),
    )
    assert online_manifest["contract_family"] == "raw_head"
    assert online_manifest["host_postprocess_frozen"] is True
    assert online_manifest["postprocess_included"] is True
    assert len(
        online_manifest["frozen_host_postprocess_contract_sha256"]
    ) == 64

    (suite / contract["artifact_path"]).write_bytes(b"tampered")
    declaration = load_authoritative_output_contract(
        suite, backend="native_full_hailo10h",
        model_id="yolo26s", task="detection",
    )
    assert declaration["contract_resolution_status"] == "conflict"
    assert "hailo10_recorded_artifact_sha256_mismatch" in declaration[
        "contract_resolution_errors"
    ]


@pytest.mark.parametrize("receipt_state", ["missing", "tampered"])
def test_hailo10_raw_head_receipt_failure_blocks_promotion(
    tmp_path: Path, receipt_state: str,
) -> None:
    suite, benchmark, contracts = _raw_suite(tmp_path)
    receipt_path = (
        suite / "hailo/hailo10/full/hailo_hef_build_receipt.json"
    )
    if receipt_state == "missing":
        receipt_path.unlink()
    else:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["preprocessing_contract"]["pad_value"] = 0
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    promotions = _promote_verified_hailo_full_contracts(
        suite_dir=suite,
        model_id="yolo26s",
        task="detection",
        suite_bench=benchmark,
        contracts=contracts,
        copied_verified={},
    )
    assert promotions == []
    assert contracts[0].get("contract_status") != "recorded"
    assert contracts[0].get("artifact_binding_status") != "verified"


def test_materializer_preserves_receipt_bound_raw_endpoint_over_decoded_plan(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    suite = run_root / "models/yolo26s/benchmark_set"
    suite.mkdir(parents=True)
    prepared = tmp_path / "prepared.hef"
    prepared.write_bytes(b"prepared-raw-head-hef")
    nodes = [f"/head/{index}" for index in range(6)]
    source_onnx, compiler_onnx = _seal_test_hailo_hef(
        prepared, hw_arch="hailo10h", end_nodes=nodes
    )
    raw_contract = {
        "schema": "onnx-splitpoint/output-contract",
        "model_id": "yolo26s",
        "task": "detection",
        "backend": "hailo10",
        "variant": "full",
        "endpoint_mode": "raw_detection_head",
        "mode": "yolo26_one2one_raw_head",
        "requires_external_postprocess": True,
        "host_tail_required": True,
        "postprocessing_required": True,
        "full_end_node_names": nodes,
        "end_node_names": nodes,
    }
    (suite / "benchmark_set.json").write_text(json.dumps({
        "model_id": "yolo26s",
        "benchmark_task": "detection",
        "hailo": {"hefs": {"hailo10": {
            "full": "hailo/hailo10/full/compiled.hef",
            "full_endpoint_mode": "raw_detection_head",
            "full_end_node_names": nodes,
            "full_build": {
                "ok": True,
                "source_onnx_path": str(source_onnx),
                "compiler_onnx_path": str(compiler_onnx),
            },
            "full_output_contract": raw_contract,
        }}},
    }), encoding="utf-8")
    provisional = {
        "model_id": "yolo26s",
        "task": "detection",
        "backend": "hailo10",
        "variant": "full",
        "endpoint_mode": "decoded_nms",
        "host_tail_required": False,
        "postprocessing_required": False,
        "contract_status": "pending_build_or_prepare",
    }

    result = materialize_backend_artifact_decisions(
        run_dir=run_root,
        model_id="yolo26s",
        targets=["hailo10"],
        full_baseline_plan={
            "task": "detection",
            "baselines": [{
                **provisional,
                "artifact_path": str(prepared),
                "source_onnx_path": str(source_onnx),
                "compiler_onnx_path": str(compiler_onnx),
            }],
        },
        output_contracts={
            "model_id": "yolo26s",
            "task": "detection",
            "contracts": [provisional],
        },
        benchmark_set_contract={
            "materialized": False,
            "materialization_scope": "contract_only",
            "source_of_truth_for_real_runs": "legacy_benchmarkset_generator",
        },
    )

    assert result["metrics"]["recorded_hailo_full_contracts"] == 1
    contract = json.loads(
        (suite / "output_contracts.json").read_text(encoding="utf-8")
    )["contracts"][0]
    assert contract["endpoint_mode"] == "raw_detection_head"
    assert contract["host_tail_required"] is True
    assert contract["full_end_node_names"] == nodes
    assert contract["hailo_build_receipt_end_nodes"] == nodes
    benchmark = json.loads(
        (suite / "benchmark_set.json").read_text(encoding="utf-8")
    )
    meta = benchmark["hailo"]["hefs"]["hailo10"]
    assert meta["full_endpoint_mode"] == "raw_detection_head"
    assert meta["full_receipt_attested"] is True
    compiler_copy = suite / meta["full_build_receipt"]["compiler_onnx_path"]
    assert compiler_copy.is_file()
    assert hashlib.sha256(compiler_copy.read_bytes()).hexdigest() == (
        hashlib.sha256(compiler_onnx.read_bytes()).hexdigest()
    )


@pytest.mark.parametrize(
    "corruption", ["hash", "nodes", "receipt_nodes", "external", "alias"]
)
def test_hailo10_raw_head_reconciliation_conflicts_fail_closed(
    tmp_path: Path, corruption: str,
) -> None:
    suite, benchmark, contracts = _raw_suite(tmp_path)
    meta = benchmark["hailo"]["hefs"]["hailo10"]
    if corruption == "hash":
        meta["full_build"]["artifact_hash"] = "0" * 64
    elif corruption == "nodes":
        meta["full_output_contract"]["end_node_names"] = ["/different"]
    elif corruption == "receipt_nodes":
        tampered = [f"/tampered/{index}" for index in range(6)]
        meta["full_end_node_names"] = tampered
        meta["full_output_contract"]["end_node_names"] = tampered
    elif corruption == "external":
        meta["full_output_contract"]["requires_external_postprocess"] = False
    elif corruption == "alias":
        conflicting = json.loads(json.dumps(meta))
        conflicting_nodes = [f"/alias-conflict/{index}" for index in range(6)]
        conflicting["full_end_node_names"] = conflicting_nodes
        conflicting["full_output_contract"][
            "end_node_names"
        ] = conflicting_nodes
        benchmark["hailo"]["hefs"]["hailo10h"] = conflicting
    promotions = _promote_verified_hailo_full_contracts(
        suite_dir=suite,
        model_id="yolo26s",
        task="detection",
        suite_bench=benchmark,
        contracts=contracts,
        copied_verified={},
    )
    assert promotions == []
    assert contracts[0].get("artifact_binding_status") != "verified"


def test_identical_hailo10_alias_raw_metadata_deduplicates(
    tmp_path: Path,
) -> None:
    suite, benchmark, contracts = _raw_suite(tmp_path)
    benchmark["hailo"]["hefs"]["hailo10h"] = json.loads(json.dumps(
        benchmark["hailo"]["hefs"]["hailo10"]
    ))

    promotions = _promote_verified_hailo_full_contracts(
        suite_dir=suite,
        model_id="yolo26s",
        task="detection",
        suite_bench=benchmark,
        contracts=contracts,
        copied_verified={},
    )

    assert len(promotions) == 1
    assert contracts[0]["artifact_binding_status"] == "verified"


def _write_boundary(
    root: Path,
    *,
    branch: str,
    backend: str,
    boundary: np.ndarray,
    target_shape: list[int],
    input_dump: bytes,
) -> None:
    folder = root / branch / "native_fifo_boundary"
    folder.mkdir(parents=True)
    boundary_path = folder / "boundary.bin"
    input_path = folder / "input_rgb_uint8.bin"
    boundary_path.write_bytes(boundary.tobytes())
    input_path.write_bytes(input_dump)
    manifest = {
        "schema": "onnx-splitpoint/native-boundary-dump",
        "schema_version": 3,
        "backend": backend,
        "dtype": str(boundary.dtype),
        "shape": list(boundary.shape),
        "nbytes": boundary.nbytes,
        "file": str(boundary_path),
        "file_sha256": hashlib.sha256(boundary.tobytes()).hexdigest(),
        "trt_input_name": "cut",
        "trt_input_shape": target_shape,
        "input_dump": str(input_path),
        "input_dump_sha256": hashlib.sha256(input_dump).hexdigest(),
        "input_shape_hwc": [2, 2, 3],
    }
    (folder / "native_fifo_boundary_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8",
    )


def test_offline_boundary_audit_gates_peer_comparison_on_exact_input(
    tmp_path: Path,
) -> None:
    module = _load_script("offline_native_boundary_contract.py")
    raw_nhwc = np.arange(24, dtype=np.uint8).reshape(1, 2, 3, 4)
    canonical = np.transpose(raw_nhwc.astype(np.float32), (0, 3, 1, 2))
    input_dump = bytes(range(12))
    _write_boundary(
        tmp_path,
        branch="hailo8",
        backend="hailo8_to_trt",
        boundary=raw_nhwc,
        target_shape=[1, 4, 2, 3],
        input_dump=input_dump,
    )
    _write_boundary(
        tmp_path,
        branch="deepx",
        backend="deepx_to_trt",
        boundary=canonical,
        target_shape=[1, 4, 2, 3],
        input_dump=input_dump,
    )
    report = module.audit_pack(tmp_path)
    assert report["ok"] is True
    assert report["technical_pass_count"] == 2
    assert report["eligible_peer_comparison_count"] == 1
    best = report["peer_comparisons"][0]["best_hypothesis"]
    assert {
        best["left_layout"], best["right_layout"],
    } == {"as_input", "memory_nhwc_to_nchw"}
    assert abs(best["correlation"]) == pytest.approx(1.0)

    # Exact input identity is mandatory; a different dump must suppress peers.
    input_path = (
        tmp_path
        / "deepx/native_fifo_boundary/input_rgb_uint8.bin"
    )
    input_path.write_bytes(b"x" * 12)
    manifest_path = (
        tmp_path
        / "deepx/native_fifo_boundary/native_fifo_boundary_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["input_dump_sha256"] = hashlib.sha256(b"x" * 12).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    report = module.audit_pack(tmp_path)
    assert report["ok"] is True
    assert report["eligible_peer_comparison_count"] == 0


def test_remote_hailo10_runner_mirror_is_byte_identical() -> None:
    primary = ROOT / "scripts/native_hailo10_trt_e2e_from_benchmarkset.py"
    mirror = (
        ROOT
        / "onnx_splitpoint_tool/resources/remote_scripts"
        / primary.name
    )
    assert primary.read_bytes() == mirror.read_bytes()
