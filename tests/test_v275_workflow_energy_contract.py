from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import numpy as np
from PIL import Image

from onnx_splitpoint_tool.native_command_contract import canonical_json_sha256
from onnx_splitpoint_tool.hailo_backend import (
    _hailo_cache_key,
    _write_hailo_receipt,
)
from onnx_splitpoint_tool.native_full_quality import resolve_native_full_plan
from onnx_splitpoint_tool.preprocessing_contract import (
    canonical_image_preprocessing_contract,
    preprocessing_contract_sha256,
    runtime_numeric_input_identity,
    runtime_numeric_input_identity_errors,
)
from onnx_splitpoint_tool.remote.process_lease import RemoteProcessLeaseScope
from onnx_splitpoint_tool.run_modes import (
    apply_run_mode,
    default_run_modes_config,
    native_full_backends_from_run_profiles,
)
from onnx_splitpoint_tool.runners.native_full_input import (
    load_sealed_deepx_native_full_input,
)
from onnx_splitpoint_tool.workflow.evidence_status import (
    blocking_status,
    derive_native_evidence_status,
)
from onnx_splitpoint_tool.workflow.runner import (
    EvaluationWorkflowRunner,
    WorkflowOptions,
    _native_full_quality_binding_set_v275,
)
from scripts import native_full_baseline_eval_runner as full_runner
from scripts import native_full_semantic_dump as semantic_dump
from scripts import run_evalrun_native_producer_variants as variants


ROOT = Path(__file__).resolve().parents[1]


def _full_profiles() -> list[dict[str, object]]:
    return [
        {
            "id": "ort_tensorrt",
            "type": "same_backend_reference",
            "full": "tensorrt",
            "stage1": "tensorrt",
            "stage2": "tensorrt",
        },
        {"id": "hailo8_full", "full": "hailo8"},
        {"id": "hailo10h_full", "full": "hailo10h"},
        {"id": "deepx_m1_full", "full": "deepx"},
        {
            "id": "hailo8_to_trt",
            "stage1": "hailo8",
            "stage2": "tensorrt",
        },
        {
            "id": "hailo10h_to_trt",
            "stage1": "hailo10h",
            "stage2": "tensorrt",
        },
        {
            "id": "deepx_to_trt",
            "stage1": "deepx",
            "stage2": "tensorrt",
        },
    ]


def _complete_matrix(count: int) -> dict[str, int]:
    return {
        "expected_row_count": count,
        "present_expected_row_count": count,
        "successful_expected_row_count": count,
        "failed_expected_row_count": 0,
        "missing_expected_row_count": 0,
    }


def _terminal_json(text: str) -> dict:
    lines = text.splitlines()
    for index in range(len(lines) - 1, -1, -1):
        if lines[index].startswith("{"):
            try:
                value = json.loads("\n".join(lines[index:]))
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
    raise AssertionError(f"no terminal JSON object in output:\n{text}")


def _run_script(name: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / name), *args],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=45,
        check=False,
    )


def test_run_mode_projection_preserves_native_runtime_and_energy_contract() -> None:
    config = default_run_modes_config()
    profile = {
        "name": "v275_projection",
        "purpose": "contract test",
        "run_profiles": _full_profiles(),
        "execution_preset": {
            "id": "standard",
            "follow_tool_config": True,
            # Deliberately stale mirrors: explicit profile blocks must win.
            "overrides": {
                "native_enabled": False,
                "energy_enabled": False,
            },
        },
        "native_producers": {
            "enabled": True,
            "frames": 777,
            "warmup": 77,
            "repetitions": 4,
            "queue_depth": 9,
            "inflight": 11,
            "runtime_api": "infer_model",
            "energy": {
                "enabled": True,
                "mode": "measure",
                "duration_s": 123,
                "runs": 7,
                "custom_sensor_contract": "sealed",
            },
        },
        "energy": {
            "enabled": False,
            "requested_native_energy": True,
            "repeat_override": 7,
            "custom_policy": "preserve-me",
        },
    }

    resolved, audit = apply_run_mode(
        profile,
        mode_id="smoke",
        config=config,
    )

    assert audit["mode_id"] == "smoke"
    assert resolved["execution_preset"]["id"] == "smoke"
    assert resolved["execution_preset"]["overrides"] == {
        "native_enabled": True,
        "energy_enabled": True,
    }
    for field in (
        "frames", "warmup", "repetitions", "queue_depth", "inflight",
        "runtime_api",
    ):
        assert resolved["native_producers"][field] == profile[
            "native_producers"
        ][field]
    assert resolved["native_producers"]["energy"] == profile[
        "native_producers"
    ]["energy"]
    assert resolved["energy"]["enabled"] is False
    assert resolved["energy"]["requested_native_energy"] is True
    assert resolved["energy"]["repeat_override"] == 7
    assert resolved["energy"]["custom_policy"] == "preserve-me"
    assert resolved["execution_preset"]["effective"][
        "native_performance_repetitions"
    ] == 4


def test_four_full_checkboxes_map_exactly_and_split_rows_do_not_imply_full() -> None:
    profiles = _full_profiles()
    assert native_full_backends_from_run_profiles(profiles) == [
        "tensorrt", "hailo8", "hailo10h", "deepx",
    ]

    plan = resolve_native_full_plan({
        "run_profiles": profiles,
        "native_producers": {"enabled": True},
    })
    assert plan.backends_by_producer == {
        "hailo8": ("hailo8", "tensorrt"),
        "hailo10h": ("hailo10h", "tensorrt"),
        "deepx": ("deepx", "tensorrt"),
    }

    split_only = resolve_native_full_plan({
        "run_profiles": [
            {
                "id": "hailo8_to_trt",
                "stage1": "hailo8",
                "stage2": "tensorrt",
            },
        ],
        "native_producers": {"enabled": True},
    })
    assert split_only.enabled is False
    assert split_only.backends_by_producer == {}


def test_disabled_energy_is_complete_not_applicable_and_nonblocking() -> None:
    evidence = derive_native_evidence_status(
        run_mode="standard",
        expected_matrix=_complete_matrix(1),
        validation_payload=None,
        validation_requested=False,
        energy_requested=False,
    )

    assert evidence["energy"]["status"] == "not_applicable"
    assert evidence["energy"]["requested"] is False
    assert evidence["energy"]["planned_measurements_complete"] is True
    assert evidence["evidence_complete"] is True
    assert blocking_status(evidence, run_mode="standard") == ""


def test_explicit_missing_or_empty_selections_fail_closed_before_work(
    tmp_path: Path,
) -> None:
    root = tmp_path / "staged"
    root.mkdir()

    full = _run_script(
        "native_full_baseline_eval_runner.py",
        "--root", str(root),
        "--models", "missing_model",
        "--backends", "hailo8",
        "--engine-build-python", sys.executable,
    )
    assert full.returncode == 3, full.stderr
    full_payload = _terminal_json(full.stdout)
    assert full_payload["ok"] is False
    assert full_payload["status"] == "selection_invalid"
    assert full_payload["rows"] == 0
    assert any(
        value.startswith("benchmark_set_index_missing:")
        for value in full_payload["selection_errors"]
    )

    fifo = _run_script(
        "native_fifo_eval_runner.py",
        "--root", str(root),
        "--models", "missing_model",
        "--case-map", "{}",
    )
    assert fifo.returncode == 3, fifo.stderr
    fifo_payload = _terminal_json(fifo.stdout)
    assert fifo_payload["ok"] is False
    assert fifo_payload["status"] == "selection_invalid"
    assert fifo_payload["rows"] == 0
    assert "case_map_missing_requested_models:missing_model" in fifo_payload[
        "selection_errors"
    ]

    e2e = _run_script(
        "native_producer_e2e_eval_runner.py",
        "--root", str(root),
        "--backend", "hailo10h",
        "--models", "missing_model",
        "--case-map", "{}",
    )
    assert e2e.returncode == 3, e2e.stderr
    e2e_payload = _terminal_json(e2e.stdout)
    assert e2e_payload["ok"] is False
    assert e2e_payload["status"] == "selection_invalid"
    assert e2e_payload["rows"] == 0


@pytest.mark.parametrize(
    ("run_mode", "expected_status", "diagnostic_only"),
    (
        ("smoke", "partial", True),
        ("standard", "failed", False),
        ("final", "failed", False),
    ),
)
def test_workflow_selection_preflight_keeps_smoke_diagnostic_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    run_mode: str,
    expected_status: str,
    diagnostic_only: bool,
) -> None:
    runner = EvaluationWorkflowRunner(
        WorkflowOptions(profile="", out=str(tmp_path)),
    )
    runner.run_id = f"selection_{run_mode}"
    runner.run_dir = tmp_path / runner.run_id
    runner._remote_process_registry.configure_journal(
        scope=RemoteProcessLeaseScope(runner.run_id, runner.session_id),
        journal_dir=runner.run_dir / "reports" / "remote_lease_journal",
    )
    runner.profile_payload = {"execution_preset": {"id": run_mode}}
    runner.manifest = {"models": {"missing_model": {}}}
    cfg = {
        "enabled": True,
        "models": ["missing_model"],
        "backends": ["hailo8"],
        "case_policy": "case_map_only",
        "case_map": {"missing_model": ["b001"]},
        "energy": {"enabled": False},
    }
    monkeypatch.setattr(runner, "_native_producer_config", lambda: cfg)

    paths, details, _message, status = (
        runner._stage_run_native_producers()
    )

    assert status == expected_status
    assert details["row_count"] == 0
    assert details["technical_quality_failure"] is True
    payload = json.loads(
        paths["native_producer_stage_json"].read_text(encoding="utf-8")
    )
    assert payload["status"] == expected_status
    assert payload["diagnostic_only"] is diagnostic_only
    assert payload["transfer_attempted"] is False
    assert payload["started_remote_count"] == 0
    assert payload["claim_eligible"] is False


def test_models_only_variant_expands_to_frozen_cases(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    benchmark_set = run_dir / "models" / "resnet50" / "benchmark_set"
    (benchmark_set / "b052").mkdir(parents=True)
    (benchmark_set / "b104").mkdir()

    effective = variants._effective_variant_case_map(
        run_dir,
        {"models": ["resnet50"]},
    )

    assert effective == {"resnet50": ["b052", "b104"]}


def test_models_only_variant_resolves_fresh_legacy_suite_cases(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    benchmark_set = run_dir / "models" / "resnet50" / "benchmark_set"
    legacy_suite = benchmark_set / "legacy_suite"
    (legacy_suite / "b052").mkdir(parents=True)

    effective = variants._effective_variant_case_map(
        run_dir,
        {"models": ["resnet50"]},
    )

    assert effective == {"resnet50": ["b052"]}


def test_explicit_variant_case_is_checked_against_legacy_suite(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    legacy_suite = (
        run_dir / "models" / "resnet50" / "benchmark_set" / "legacy_suite"
    )
    (legacy_suite / "b052").mkdir(parents=True)

    effective = variants._effective_variant_case_map(
        run_dir,
        {
            "models": ["resnet50"],
            "case_map": {"resnet50": ["b052"]},
        },
    )

    assert effective == {"resnet50": ["b052"]}


@pytest.mark.parametrize(
    (
        "producer", "backend", "source_run_id", "setup_id", "model_id",
        "runtime_precision_prefix",
    ),
    (
        (
            "deepx", "deepx", "deepx_m1_full",
            "orin_nx_deepx_m1_01", "yolo26s", "deepx_dxnn_sha256",
        ),
        (
            "deepx", "deepx", "deepx_full",
            "orin_nx_deepx_m1_01", "yolov7_paper", "deepx_dxnn_sha256",
        ),
        (
            "hailo8", "hailo8", "hailo8",
            "orin_nx_hailo8_01", "yolov7_paper", "hailo_hef_sha256",
        ),
        (
            "hailo10h", "hailo10h", "hailo10",
            "orin_nx_hailo10_01", "yolov7_paper", "hailo_hef_sha256",
        ),
    ),
    ids=(
        "deepx-yolo26s", "deepx-yolov7", "hailo8-yolov7",
        "hailo10h-yolov7",
    ),
)
def test_vendor_full_quality_binding_is_hash_sealed_and_exact(
    tmp_path: Path,
    producer: str,
    backend: str,
    source_run_id: str,
    setup_id: str,
    model_id: str,
    runtime_precision_prefix: str,
) -> None:
    run_dir = tmp_path / "run_275"
    summary_path = (
        run_dir / "quality_management" / "central_quality_summary.json"
    )
    request_path = run_dir / "models" / model_id / "request.json"
    request_path.parent.mkdir(parents=True)
    summary_path.parent.mkdir(parents=True)

    preprocessing_identity = canonical_image_preprocessing_contract(
        "detection", (640, 640),
    )
    preprocessing_sha = canonical_json_sha256(preprocessing_identity)
    runtime_numeric_identity, runtime_numeric_sha = (
        runtime_numeric_input_identity(
            backend=f"native_full_{backend}",
            task="detection",
            preprocessing_contract_sha256_value=preprocessing_sha,
            runtime_input_name="images",
            runtime_input_shape=[1, 640, 640, 3],
            runtime_input_dtype="uint8",
            runtime_input_layout="NHWC",
            runtime_color_space="RGB",
            runtime_normalization=(
                "embedded_dxcom_preprocessing"
                if backend == "deepx"
                else "embedded_hailo_quantization"
            ),
        )
    )
    model_sha = "1" * 64
    compiled_sha = "2" * 64
    dataset_sha = "3" * 64
    image_ids_sha = "4" * 64
    ground_truth_sha = "5" * 64
    policy_sha = "6" * 64
    decoder_identity = {"schema": "test/decoder", "version": 1}
    decoder_sha = canonical_json_sha256(decoder_identity)
    nms_identity = {"schema": "test/nms", "version": 1}
    nms_sha = canonical_json_sha256(nms_identity)
    endpoint_identity = {
        "schema": "test/runtime-endpoint", "version": 1,
    }
    endpoint_sha = canonical_json_sha256(endpoint_identity)
    endpoint_record_identity = {
        "schema": "test/quality-record-endpoint", "version": 1,
    }
    endpoint_record_sha = canonical_json_sha256(
        endpoint_record_identity
    )
    prepared_join = {
        "schema": (
            "onnx-splitpoint/deepx-performance-quality-input-binding"
        ),
        "schema_version": 1,
        "binding_verified": True,
        "source_image_id": "000000000632.jpg",
        "source_image_sha256": "c" * 64,
        "prepared_input_sha256": "d" * 64,
        "prepared_input_bytes": 640 * 640 * 3,
        "prepared_input_name": "images",
        "prepared_input_shape": [1, 640, 640, 3],
        "prepared_input_dtype": "uint8",
        "prepared_input_layout": "NHWC",
        "runtime_preprocessing_sha256": preprocessing_sha,
        "runtime_numeric_input_sha256": runtime_numeric_sha,
    }
    prepared_join_sha = canonical_json_sha256(prepared_join)
    prepared_evidence = {
        "schema": "onnx-splitpoint/deepx-quality-prepared-input-set",
        "schema_version": 1,
        "record_count": 1,
        "records_sha256": "e" * 64,
        "runtime_numeric_input_identity": runtime_numeric_identity,
        "runtime_numeric_input_sha256": runtime_numeric_sha,
        "performance_quality_input_binding": prepared_join,
        "performance_quality_input_binding_sha256": prepared_join_sha,
    }
    quality_contract = {
        "schema": "test/vendor-central-quality-record-contract",
        "schema_version": 1,
        "task": "detection",
        "variant": "full",
        "model": {"source_onnx_sha256": model_sha},
        "dataset": {
            "manifest_sha256": dataset_sha,
            "image_ids_sha256": image_ids_sha,
            "ground_truth_sha256": ground_truth_sha,
            "image_count": 1,
        },
        "preprocessing": {
            "identity": preprocessing_identity,
            "sha256": preprocessing_sha,
        },
        "decoder": {
            "identity": decoder_identity, "sha256": decoder_sha,
        },
        "nms": {"identity": nms_identity, "sha256": nms_sha},
        "quality_record_endpoint": {
            "identity": endpoint_record_identity,
            "sha256": endpoint_record_sha,
        },
        "quality_record_endpoint_contract_sha256": endpoint_record_sha,
    }
    if backend == "deepx":
        quality_contract["prepared_input_evidence"] = prepared_evidence
    quality_sha = canonical_json_sha256(quality_contract)
    quality_contract["quality_contract_sha256"] = quality_sha
    runtime_precision = f"{runtime_precision_prefix}:{compiled_sha}"
    request = {
        "schema": "onnx-splitpoint/central-quality-evaluation-request",
        "schema_version": 1,
        "task": "detection",
        "variant": "full",
        "quality_contract": quality_contract,
        "quality_contract_sha256": quality_sha,
        "preprocessing_contract_sha256": preprocessing_sha,
        "decoder_contract_sha256": decoder_sha,
        "nms_contract_sha256": nms_sha,
        "quality_record_endpoint_contract_sha256": endpoint_record_sha,
        "endpoint_contract_hash": endpoint_sha,
        "runtime_precision_identity": runtime_precision,
        "policy_sha256": policy_sha,
    }
    if backend == "deepx":
        precision_identity = {
            "schema": "test/deepx-runtime-precision",
            "artifact_sha256": compiled_sha,
        }
        producer_identity = {
            "schema": (
                "onnx-splitpoint/central-quality-producer-identity"
            ),
            "schema_version": 1,
            "backend": "deepx_m1",
            "source_run_id": source_run_id,
            "case_id": "full",
            "variant": "full",
            "task": "detection",
            "model": {
                "source_onnx_sha256": model_sha,
                "runtime_artifact_sha256": compiled_sha,
            },
            "dataset": dict(quality_contract["dataset"]),
            "preprocessing": dict(quality_contract["preprocessing"]),
            "prepared_input_evidence": prepared_evidence,
            "endpoint": {
                "identity": endpoint_identity,
                "sha256": endpoint_sha,
            },
            "quality_record_endpoint": dict(
                quality_contract["quality_record_endpoint"]
            ),
            "precision": {
                "identity": precision_identity,
                "sha256": canonical_json_sha256(precision_identity),
            },
            "quality_contract": quality_contract,
            "quality_contract_sha256": quality_sha,
            "preprocessing_contract_sha256": preprocessing_sha,
            "prepared_input_evidence_sha256": "e" * 64,
            "prepared_input_join_binding": prepared_join,
            "prepared_input_join_binding_sha256": prepared_join_sha,
            "decoder_contract_sha256": decoder_sha,
            "nms_contract_sha256": nms_sha,
            "endpoint_contract_hash": endpoint_sha,
            "quality_record_endpoint_contract_sha256": (
                endpoint_record_sha
            ),
            "runtime_precision_identity": runtime_precision,
        }
        producer_sha = canonical_json_sha256(producer_identity)
        producer_identity["producer_identity_sha256"] = producer_sha
        request.update({
            "prepared_input_evidence_sha256": "e" * 64,
            "prepared_input_join_binding": prepared_join,
            "prepared_input_join_binding_sha256": prepared_join_sha,
            "producer_identity": producer_identity,
            "producer_identity_sha256": producer_sha,
        })
    request_path.write_text(
        json.dumps(request, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    request_sha = hashlib.sha256(request_path.read_bytes()).hexdigest()

    result = {
        "model_id": model_id,
        "source_setup_id": setup_id,
        "source_run_id": source_run_id,
        "eval_run_id": "run_275",
        "variant": "full",
        "status": "completed",
        "technical_status": "completed",
        "task": "detection",
        "case_id": "full",
        "source_request": f"models/{model_id}/request.json",
        "source_request_sha256": request_sha,
        "model_sha256": model_sha,
        "validation_dataset_sha256": dataset_sha,
        "validation_dataset_image_ids_sha256": image_ids_sha,
        "validation_dataset_ground_truth_sha256": ground_truth_sha,
        "task_quality_policy_sha256": policy_sha,
        "quality_contract_sha256": quality_sha,
        "preprocessing_contract_sha256": preprocessing_sha,
        "decoder_contract_sha256": decoder_sha,
        "nms_contract_sha256": nms_sha,
        "quality_record_endpoint_contract_sha256": endpoint_record_sha,
        "runtime_precision_identity": runtime_precision,
        "endpoint_contract_hash": endpoint_sha,
        "request_identity": {"identity_valid": True},
    }
    summary_path.write_text(
        json.dumps({"results": [result]}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    binding_set, errors = _native_full_quality_binding_set_v275(
        summary_path,
        eval_run_id="run_275",
        setup_id=setup_id,
        producer=producer,
        full_backends=(backend, "tensorrt"),
        model_ids=(model_id,),
    )

    assert errors == []
    binding_key = f"native_full_{backend}|{model_id}"
    assert set(binding_set["bindings_by_backend_model"]) == {binding_key}
    binding = binding_set["bindings_by_backend_model"][
        binding_key
    ]
    assert binding["source_request_file_sha256"] == request_sha
    assert binding["task_quality_policy_sha256"] == policy_sha
    assert "runtime_quality_gate_policy_sha256" not in binding
    assert binding["preprocessing_contract_sha256"] == preprocessing_sha
    assert binding["binding_sha256"] == canonical_json_sha256({
        key: value for key, value in binding.items()
        if key != "binding_sha256"
    })
    assert binding_set["binding_set_sha256"] == canonical_json_sha256({
        key: value for key, value in binding_set.items()
        if key != "binding_set_sha256"
    })

    binding_path = (
        run_dir / "quality_first"
        / "vendor_full_quality_request_binding_set.json"
    )
    binding_path.parent.mkdir(parents=True, exist_ok=True)
    binding_path.write_text(
        json.dumps(binding_set, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    ns = argparse.Namespace(
        quality_request_binding_set=str(binding_path),
        setup_id=setup_id,
        comparison_backend=producer,
        root=str(run_dir),
        letterbox_pad_value=114,
    )
    loaded, load_status = full_runner._load_full_quality_binding_set(ns)
    assert load_status == "quality_request_binding_set_verified"
    ns.quality_request_binding_set_data = loaded
    row = {
        "backend": f"native_full_{backend}",
        "ok": True,
        "runtime_preprocess_mode": "letterbox_rgb_uint8",
        "runtime_input_shape": [1, 640, 640, 3],
        "runtime_input_dtype": "uint8",
        "runtime_input_layout": "NHWC",
        "runtime_color_space": "RGB",
        "runtime_normalization": runtime_numeric_identity[
            "runtime_normalization"
        ],
        "runtime_preprocessing_identity": preprocessing_identity,
        "runtime_preprocessing_sha256": preprocessing_sha,
        "runtime_numeric_input_identity": runtime_numeric_identity,
        "runtime_numeric_input_sha256": runtime_numeric_sha,
        "endpoint_contract_hash": endpoint_sha,
    }
    if backend == "deepx":
        row.update({
            "prepared_input_sha256": prepared_join[
                "prepared_input_sha256"
            ],
            "prepared_input_bytes": prepared_join[
                "prepared_input_bytes"
            ],
            "prepared_input_name": prepared_join[
                "prepared_input_name"
            ],
            "prepared_input_shape": prepared_join[
                "prepared_input_shape"
            ],
            "prepared_input_dtype": prepared_join[
                "prepared_input_dtype"
            ],
            "prepared_input_layout": prepared_join[
                "prepared_input_layout"
            ],
            "prepared_input_source_image_id": prepared_join[
                "source_image_id"
            ],
            "prepared_input_source_image_sha256": prepared_join[
                "source_image_sha256"
            ],
        })
    full_contract = {
        "model_binding": {
            "source_onnx_sha256": model_sha,
            "compiled_artifact_sha256": compiled_sha,
        },
        "runtime_preprocessing_binding": {
            "status": (
                "runtime_preprocessing_and_numeric_identity_verified_exact"
            ),
            "identity": preprocessing_identity,
            "sha256": preprocessing_sha,
            "runtime_numeric_input_identity": runtime_numeric_identity,
            "runtime_numeric_input_sha256": runtime_numeric_sha,
        },
    }
    if backend.startswith("hailo"):
        full_contract.update({
            "hailo_hef_build_receipt_status": (
                "hailo_hef_build_receipt_verified_exact"
            ),
            "hailo_hef_build_receipt_file_sha256": "c" * 64,
            "hailo_hef_build_receipt_sha256": "d" * 64,
            "hailo_hef_preprocessing_contract": preprocessing_identity,
            "hailo_hef_preprocessing_contract_sha256": preprocessing_sha,
        })
    attached, attach_status = full_runner._attach_full_quality_request_binding(
        row,
        full_contract,
        model=model_id,
        ns=ns,
    )
    assert attach_status == "quality_request_binding_verified_exact"
    assert attached["ok"] is True
    assert attached["quality_request_binding_status"] == "verified_exact"
    assert attached["source_request_sha256"] == request_sha
    assert attached["task_quality_policy_sha256"] == policy_sha
    assert "runtime_quality_gate_policy_sha256" not in attached

    mismatch, mismatch_status = (
        full_runner._attach_full_quality_request_binding(
            {
                **row,
                "runtime_preprocess_mode": "resize_rgb_uint8",
            },
            full_contract,
            model=model_id,
            ns=ns,
        )
    )
    assert mismatch_status == "quality_request_binding_failed"
    # Quality provenance is fail-closed for claims, but it must not rewrite a
    # completed hardware observation or block the remaining Native rows.
    assert mismatch["ok"] is True
    assert mismatch["performance_claim_eligible"] is False
    assert mismatch["energy_claim_eligible"] is False
    assert mismatch["scientific_claim_eligible"] is False
    assert mismatch["eligible_for_ranking"] is False
    assert "observed_runtime_preprocess_mode_mismatch" in mismatch[
        "quality_request_binding_errors"
    ]

    for field, invalid_value, expected_error in (
        (
            "runtime_color_space", "BGR",
            "observed_runtime_color_space_mismatch",
        ),
        (
            "runtime_input_shape", [1, 224, 224, 3],
            "observed_runtime_input_shape_mismatch",
        ),
        (
            "runtime_input_dtype", "int8",
            "observed_runtime_input_dtype_mismatch",
        ),
        (
            "runtime_normalization", "subtract_magic",
            "observed_runtime_normalization_mismatch",
        ),
    ):
        rejected, rejected_status = (
            full_runner._attach_full_quality_request_binding(
                {**row, field: invalid_value, "ok": True},
                full_contract,
                model=model_id,
                ns=ns,
            )
        )
        assert rejected_status == "quality_request_binding_failed"
        assert rejected["ok"] is True
        assert rejected["scientific_claim_eligible"] is False
        assert expected_error in rejected[
            "quality_request_binding_errors"
        ]

    if backend == "deepx":
        for field, invalid_value, expected_error in (
            (
                "prepared_input_sha256", "f" * 64,
                "quality_performance_prepared_input_sha256_mismatch",
            ),
            (
                "prepared_input_bytes", 1,
                "quality_performance_prepared_input_bytes_mismatch",
            ),
            (
                "prepared_input_name", "other",
                "quality_performance_prepared_input_name_mismatch",
            ),
            (
                "prepared_input_shape", [1, 1, 1, 1],
                "quality_performance_prepared_input_shape_mismatch",
            ),
            (
                "prepared_input_dtype", "int8",
                "quality_performance_prepared_input_dtype_mismatch",
            ),
            (
                "prepared_input_layout", "NCHW",
                "quality_performance_prepared_input_layout_mismatch",
            ),
            (
                "prepared_input_source_image_id", "other.jpg",
                "quality_performance_source_image_id_mismatch",
            ),
            (
                "prepared_input_source_image_sha256", "f" * 64,
                "quality_performance_source_image_sha256_mismatch",
            ),
        ):
            rejected, rejected_status = (
                full_runner._attach_full_quality_request_binding(
                    {**row, field: invalid_value, "ok": True},
                    full_contract,
                    model=model_id,
                    ns=ns,
                )
            )
            assert rejected_status == "quality_request_binding_failed"
            assert expected_error in rejected[
                "quality_request_binding_errors"
            ]

    if backend == "deepx" and model_id == "yolo26s":
        original_summary_bytes = summary_path.read_bytes()
        original_binding_bytes = binding_path.read_bytes()

        tampered_result = dict(result)
        tampered_result["quality_contract_sha256"] = "f" * 64
        summary_path.write_text(
            json.dumps(
                {"results": [tampered_result]},
                indent=2, sort_keys=True,
            ) + "\n",
            encoding="utf-8",
        )
        tampered_set, tampered_errors = (
            _native_full_quality_binding_set_v275(
                summary_path,
                eval_run_id="run_275",
                setup_id=setup_id,
                producer=producer,
                full_backends=(backend,),
                model_ids=(model_id,),
            )
        )
        assert tampered_set["complete"] is False
        assert any(
            "central_summary_claim_mismatch:quality_contract_sha256"
            in error for error in tampered_errors
        )
        summary_path.write_bytes(original_summary_bytes)

        source_request_cases: list[tuple[str, str]] = [
            (str(request_path), "source_request_path_not_relative"),
            (
                f"../models/{model_id}/request.json",
                "source_request_path_not_relative",
            ),
        ]
        leaf_link = request_path.parent / "request-link.json"
        leaf_link.symlink_to(request_path)
        source_request_cases.append((
            f"models/{model_id}/{leaf_link.name}",
            "source_request_path_contains_symlink",
        ))
        real_parent = run_dir / "request-store"
        real_parent.mkdir()
        (real_parent / "request.json").write_bytes(
            request_path.read_bytes()
        )
        linked_parent = request_path.parent / "linked-parent"
        linked_parent.symlink_to(real_parent, target_is_directory=True)
        source_request_cases.append((
            f"models/{model_id}/linked-parent/request.json",
            "source_request_path_contains_symlink",
        ))
        for source_request, expected_error in source_request_cases:
            bad_result = {**result, "source_request": source_request}
            summary_path.write_text(
                json.dumps(
                    {"results": [bad_result]},
                    indent=2, sort_keys=True,
                ) + "\n",
                encoding="utf-8",
            )
            rejected_set, rejected_errors = (
                _native_full_quality_binding_set_v275(
                    summary_path,
                    eval_run_id="run_275",
                    setup_id=setup_id,
                    producer=producer,
                    full_backends=(backend,),
                    model_ids=(model_id,),
                )
            )
            assert rejected_set["complete"] is False
            assert any(
                expected_error in error for error in rejected_errors
            )
        summary_path.write_bytes(original_summary_bytes)

        summary_real = summary_path.with_name("summary-real.json")
        summary_path.rename(summary_real)
        summary_path.symlink_to(summary_real)
        rejected_set, rejected_errors = (
            _native_full_quality_binding_set_v275(
                summary_path,
                eval_run_id="run_275",
                setup_id=setup_id,
                producer=producer,
                full_backends=(backend,),
                model_ids=(model_id,),
            )
        )
        assert rejected_set["complete"] is False
        assert "central_quality_summary_role_path_mismatch" in rejected_errors
        summary_path.unlink()
        summary_real.rename(summary_path)

        quality_dir = summary_path.parent
        quality_dir_real = run_dir / "quality-management-real"
        quality_dir.rename(quality_dir_real)
        quality_dir.symlink_to(
            quality_dir_real, target_is_directory=True,
        )
        rejected_set, rejected_errors = (
            _native_full_quality_binding_set_v275(
                summary_path,
                eval_run_id="run_275",
                setup_id=setup_id,
                producer=producer,
                full_backends=(backend,),
                model_ids=(model_id,),
            )
        )
        assert rejected_set["complete"] is False
        assert "central_quality_summary_role_path_mismatch" in rejected_errors
        quality_dir.unlink()
        quality_dir_real.rename(quality_dir)

        binding_real = binding_path.with_name("binding-real.json")
        binding_path.rename(binding_real)
        binding_path.symlink_to(binding_real)
        assert full_runner._load_full_quality_binding_set(ns) == (
            {}, "quality_request_binding_set_role_path_mismatch",
        )
        binding_path.unlink()
        binding_real.rename(binding_path)

        binding_dir = binding_path.parent
        binding_dir_real = run_dir / "quality-first-real"
        binding_dir.rename(binding_dir_real)
        binding_dir.symlink_to(binding_dir_real, target_is_directory=True)
        assert full_runner._load_full_quality_binding_set(ns) == (
            {}, "quality_request_binding_set_role_path_mismatch",
        )
        binding_dir.unlink()
        binding_dir_real.rename(binding_dir)

        replayed = json.loads(original_binding_bytes)
        replayed["eval_run_id"] = "different_run"
        replayed.pop("binding_set_sha256")
        replayed["binding_set_sha256"] = canonical_json_sha256(replayed)
        binding_path.write_text(json.dumps(replayed), encoding="utf-8")
        assert full_runner._load_full_quality_binding_set(ns) == (
            {}, "quality_request_binding_set_identity_invalid",
        )
        binding_path.write_bytes(original_binding_bytes)


def test_vendor_full_binding_missing_set_key_and_partial_set_fail_closed(
    tmp_path: Path,
) -> None:
    row = {
        "backend": "native_full_deepx",
        "model": "yolo26s",
        "ok": True,
    }
    ns = argparse.Namespace(
        quality_request_binding_set_data={},
        setup_id="orin_nx_deepx_m1_01",
        comparison_backend="deepx",
    )
    rejected, status = full_runner._attach_full_quality_request_binding(
        dict(row), {}, model="yolo26s", ns=ns,
    )
    assert status == "quality_request_binding_failed"
    assert rejected["quality_request_binding_errors"] == [
        "quality_request_binding_set_missing"
    ]

    ns.quality_request_binding_set_data = {
        "bindings_by_backend_model": {},
    }
    rejected, status = full_runner._attach_full_quality_request_binding(
        dict(row), {}, model="yolo26s", ns=ns,
    )
    assert status == "quality_request_binding_failed"
    assert rejected["quality_request_binding_errors"] == [
        "quality_request_binding_key_missing:native_full_deepx|yolo26s"
    ]

    partial = {
        "schema": "onnx-splitpoint/native-full-quality-request-binding-set",
        "schema_version": 1,
        "eval_run_id": "run_275",
        "setup_id": ns.setup_id,
        "comparison_backend": ns.comparison_backend,
        "required_binding_keys": [
            "native_full_deepx|yolo26s",
            "native_full_deepx|yolov7_paper",
        ],
        "complete": False,
        "bindings_by_backend_model": {},
    }
    partial["binding_set_sha256"] = canonical_json_sha256(partial)
    run_root = tmp_path / "run_275"
    path = (
        run_root / "quality_first"
        / "vendor_full_quality_request_binding_set.json"
    )
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(partial), encoding="utf-8")
    load_ns = argparse.Namespace(
        quality_request_binding_set=str(path),
        setup_id=ns.setup_id,
        comparison_backend=ns.comparison_backend,
        root=str(run_root),
    )
    loaded, load_status = full_runner._load_full_quality_binding_set(
        load_ns
    )
    assert loaded == {}
    assert load_status == "quality_request_binding_set_identity_invalid"


def test_vendor_full_binding_generator_reports_every_missing_pair(
    tmp_path: Path,
) -> None:
    summary = (
        tmp_path / "run_275" / "quality_management"
        / "central_quality_summary.json"
    )
    summary.parent.mkdir(parents=True)
    summary.write_text('{"results": []}\n', encoding="utf-8")

    payload, errors = _native_full_quality_binding_set_v275(
        summary,
        eval_run_id="run_275",
        setup_id="orin_nx_deepx_m1_01",
        producer="deepx",
        full_backends=("deepx",),
        model_ids=("yolo26s", "yolov7_paper"),
    )

    assert payload["complete"] is False
    assert payload["bindings_by_backend_model"] == {}
    assert payload["required_binding_keys"] == [
        "native_full_deepx|yolo26s",
        "native_full_deepx|yolov7_paper",
    ]
    assert set(errors) == {
        "missing_vendor_full_quality:native_full_deepx|yolo26s",
        "missing_vendor_full_quality:native_full_deepx|yolov7_paper",
    }


def test_hailo_full_requires_exact_sibling_build_receipt(
    tmp_path: Path,
) -> None:
    benchmark_set = tmp_path / "benchmark_set"
    source = benchmark_set / "models" / "yolov7_paper.onnx"
    compiler = benchmark_set / "models" / "yolov7_paper_hailo_fixed.onnx"
    hef = benchmark_set / "hailo" / "hailo10" / "full" / "compiled.hef"
    source.parent.mkdir(parents=True)
    hef.parent.mkdir(parents=True)
    source.write_bytes(b"source-onnx")
    compiler.write_bytes(b"fixed-compiler-onnx")
    hef.write_bytes(b"compiled-hef")
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    hef_sha = hashlib.sha256(hef.read_bytes()).hexdigest()
    preprocessing = canonical_image_preprocessing_contract(
        "detection", (640, 640),
    )
    preprocessing_sha = preprocessing_contract_sha256(preprocessing)
    cache_key, cache_payload = _hailo_cache_key(
        model_path=compiler,
        activation_part1=None,
        hw_arch="hailo10h",
        opt_level=1,
        calib_dir=None,
        calib_count=64,
        effective_calib_count=54,
        calibration_storage="memory",
        calibration_memory_cap_bytes=256 * 1024 * 1024,
        calib_batch_size=8,
        extra_model_script="",
        start_nodes=None,
        end_nodes=None,
        preprocessing_contract=preprocessing,
    )
    receipt = _write_hailo_receipt(
        hef_path=hef,
        source_onnx=source,
        compiler_onnx=compiler,
        hw_arch="hailo10h",
        net_name="yolov7_paper",
        preprocessing_contract=preprocessing,
        preprocessing_sha256=preprocessing_sha,
        cache_key=cache_key,
        cache_payload=cache_payload,
        calibration_identity=str(cache_payload["calibration_identity"]),
        calibration_count=int(cache_payload["calibration_count"]),
    )
    receipt_path = hef.parent / "hailo_hef_build_receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    verified, status = full_runner._verified_hailo_full_build_receipt(
        hef=hef,
        benchmark_set=benchmark_set,
        model="yolov7_paper",
        hw_arch="hailo10h",
    )
    assert status == "hailo_hef_build_receipt_verified_exact"
    assert verified is not None
    assert verified["source_onnx_sha256"] == source_sha
    assert verified["preprocessing_contract_sha256"] == preprocessing_sha

    receipt["preprocessing_contract"]["color_space"] = "BGR"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    rejected, reason = full_runner._verified_hailo_full_build_receipt(
        hef=hef,
        benchmark_set=benchmark_set,
        model="yolov7_paper",
        hw_arch="hailo10h",
    )
    assert rejected is None
    assert reason == "hailo_hef_build_receipt_identity_invalid"


def test_semantic_dump_manifest_seals_semantic_and_numeric_input_identities(
    tmp_path: Path,
) -> None:
    image = tmp_path / "image.png"
    Image.fromarray(
        np.full((3, 5, 3), 127, dtype=np.uint8), mode="RGB",
    ).save(image)
    input_hwc = np.full((2, 2, 3), 127, dtype=np.uint8)
    input_tensor = np.transpose(
        input_hwc.astype(np.float32) / 255.0, (2, 0, 1),
    )[None]
    preprocessing = {
        "mode": "resize_rgb_uint8",
        "pad_value": 0,
        "rgb": True,
        "ort_model_scale": "imagenet",
        "layout": "NCHW",
        "normalization": "imagenet_mean_std",
        "color_space": "RGB",
        "input_domain": "uint8_0_255",
        "resize_interpolation": "bilinear",
        "resize_rounding": "python_round_ties_to_even",
        "placement": "not_applicable",
    }
    _output_manifest, input_manifest = semantic_dump._write_manifests(
        outputs={"logits": np.zeros((1, 1000), dtype=np.float32)},
        out_dir=tmp_path / "dump",
        backend="deepx",
        model="resnet50",
        setup_id="orin_nx_deepx_m1_01",
        comparison_backend="deepx",
        task="classification",
        image=image,
        input_hwc=input_hwc,
        preprocess=preprocessing,
        input_tensor=input_tensor,
        input_name="images",
        runtime_meta={},
    )
    payload = json.loads(input_manifest.read_text(encoding="utf-8"))
    expected = canonical_image_preprocessing_contract(
        "classification", (2, 2),
    )
    assert payload["runtime_preprocessing_identity"] == expected
    assert payload["runtime_preprocessing_sha256"] == (
        canonical_json_sha256(expected)
    )
    assert payload["runtime_numeric_input_identity"][
        "preprocessing_contract_sha256"
    ] == payload["runtime_preprocessing_sha256"]
    assert payload["runtime_numeric_input_identity"][
        "runtime_input_shape"
    ] == [1, 3, 2, 2]
    assert payload["runtime_numeric_input_identity"][
        "runtime_input_layout"
    ] == "NCHW"
    assert payload["input_dump"] == "input_rgb_uint8.bin"
    assert payload["runtime_input_file"] == "runtime_input.bin"
    loaded = load_sealed_deepx_native_full_input(
        input_manifest,
        image_path=image,
        input_contract={
            "input": {
                "name": "images",
                "shape": [1, 3, 2, 2],
                "dtype": "float32",
                "layout": "NCHW",
                "color_space": "RGB",
                "normalization": "imagenet_mean_std",
                "preprocess_mode": "resize",
                "letterbox_pad_value": 0,
            },
        },
        task="classification",
        expected_model="resnet50",
        expected_setup_id="orin_nx_deepx_m1_01",
        expected_comparison_backend="deepx",
    )
    assert loaded["runtime_input"].tobytes() == input_tensor.tobytes()


def test_runtime_numeric_identity_rejects_malformed_version_and_domain() -> None:
    preprocessing = canonical_image_preprocessing_contract(
        "detection", (640, 640),
    )
    preprocessing_sha = canonical_json_sha256(preprocessing)
    identity, _identity_sha = runtime_numeric_input_identity(
        backend="native_full_deepx",
        task="detection",
        preprocessing_contract_sha256_value=preprocessing_sha,
        runtime_input_name="images",
        runtime_input_shape=[1, 640, 640, 3],
        runtime_input_dtype="uint8",
        runtime_input_layout="NHWC",
        runtime_color_space="RGB",
        runtime_normalization="embedded_dxcom_preprocessing",
    )
    identity["schema_version"] = {"malformed": True}
    identity["runtime_numeric_domain"] = "float_0_1"
    errors = runtime_numeric_input_identity_errors(
        identity, preprocessing,
    )
    assert "runtime_numeric_input_schema_version_mismatch" in errors
    assert "runtime_numeric_input_domain_mismatch" in errors


def test_deepx_repetition_prepared_input_identity_drift_fails_closed() -> None:
    base = {
        "ok": True,
        "status": "ok",
        "backend": "native_full_deepx",
        "model": "yolo26s",
        "case": "full",
        "setup_id": "orin_nx_deepx_m1_01",
        "comparison_backend": "deepx",
        "execution_precision": "opaque_vendor",
        "run_id": "deepx_m1_full",
        "prepared_feed_contract_version": (
            "deepx-sealed-runtime-input-v3"
        ),
        "prepared_input_sha256": "a" * 64,
        "prepared_input_bytes": 1 * 640 * 640 * 3,
        "prepared_input_name": "images",
        "prepared_input_shape": [1, 640, 640, 3],
        "prepared_input_dtype": "uint8",
        "prepared_input_layout": "NHWC",
        "prepared_input_source_image_id": "000000000632.jpg",
        "prepared_input_source_image_sha256": "b" * 64,
        "runtime_preprocessing_sha256": "c" * 64,
        "runtime_numeric_input_sha256": "d" * 64,
        "fps_makespan": 15.0,
        "latency_mean_ms": 66.0,
        "completed_work_units": 20,
    }
    first = {**base, "runtime_instance_id": "run-1"}
    second = {
        **base,
        "runtime_instance_id": "run-2",
        "prepared_input_sha256": "e" * 64,
    }

    aggregate = full_runner._aggregate_full_repetitions(
        [first, second], requested=2,
    )

    assert aggregate["ok"] is False
    assert aggregate["status"] == "identity_drift"
    assert aggregate["failure_reason"] == (
        "native_full_repetition_identity_drift"
    )
    assert "prepared_input_sha256" in aggregate[
        "repetition_identity_drift_fields"
    ]


def test_completed_v2_artifact_must_be_persisted_exactly(
    tmp_path: Path,
) -> None:
    artifact = {
        "schema": (
            "onnx-splitpoint/frozen-completed-detection-result-artifact"
        ),
        "schema_version": 1,
        "detections": [],
    }
    artifact_bytes = json.dumps(
        artifact,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    artifact_sha = hashlib.sha256(artifact_bytes).hexdigest()
    artifact_path = tmp_path / "completed.json"
    artifact_path.write_bytes(artifact_bytes)
    report = {
        "completed_task_result_artifact_saved": True,
        "completed_task_result_artifact": artifact,
        "completed_task_result_artifact_sha256": artifact_sha,
        "completed_task_result_artifact_path": str(artifact_path),
        "completed_task_result_artifact_file_sha256": artifact_sha,
    }
    sealed_result = {
        "completed_result_artifact": artifact,
        "completed_result_artifact_sha256": artifact_sha,
    }
    assert full_runner._completed_result_artifact_persistence_status(
        report, sealed_result=sealed_result,
        allowed_root=tmp_path, expected_path=artifact_path,
    ) == (True, "verified_exact")

    artifact_path.write_text("{}", encoding="utf-8")
    verified, status = (
        full_runner._completed_result_artifact_persistence_status(
            report, sealed_result=sealed_result,
            allowed_root=tmp_path, expected_path=artifact_path,
        )
    )
    assert verified is False
    assert status == "completed_task_result_artifact_persistence_mismatch"


def test_all_changed_remote_script_mirrors_are_byte_identical() -> None:
    names = (
        "native_fifo_eval_runner.py",
        "native_full_baseline_eval_runner.py",
        "native_full_semantic_dump.py",
        "native_producer_e2e_eval_runner.py",
        "native_producer_final_report.py",
        "native_producer_validate_visualize.py",
        "native_trt_full_completed_hotloop.py",
        "native_yolo_full_self_reference_probe.py",
        "run_evalrun_native_producer_variants.py",
        "smoke_hailo10_hef_runner.py",
        "update_evalset_native_producers.py",
    )
    for name in names:
        assert (ROOT / "scripts" / name).read_bytes() == (
            ROOT
            / "onnx_splitpoint_tool"
            / "resources"
            / "remote_scripts"
            / name
        ).read_bytes(), name


def test_remote_self_reference_sidecar_block_fails_closed() -> None:
    source = (
        ROOT / "scripts" / "update_evalset_native_producers.py"
    ).read_text(encoding="utf-8")
    assert (
        "ok = bool(rows) and all(r.get('ok') is True for r in rows)"
        in source
    )
    assert "if not rows:\n    sys.exit(3)" in source
    assert "sys.exit(0 if ok else 4)" in source
    assert "if rows else True" not in source
