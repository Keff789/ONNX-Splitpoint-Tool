from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    module_name = f"v27511_{name}_{id(path)}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _semantic_binding(*, eval_run_id: str = "eval-current") -> dict[str, Any]:
    return {
        "eval_run_id": eval_run_id,
        "source_run_id": "hailo8_to_trt",
        "binding_sha256": "a" * 64,
        "cache_verify_replay": {
            "artifact_policy": "cache_verify_only",
            "compiler_dispatched": False,
            "local_validation_status": "local_files_rehashed_test",
        },
    }


def test_cache_binding_set_hashes_are_diagnostic_but_compiler_fence_is_hard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matrix = _load_script("native_fifo_smoke_matrix")
    monkeypatch.setenv("ONNX_SPLITPOINT_ARTIFACT_POLICY", "cache_verify_only")
    monkeypatch.setattr(
        matrix,
        "validate_native_split_quality_binding",
        lambda binding, **_kwargs: (binding, "portable_test"),
    )
    payload = {
        "schema": "onnx-splitpoint/native-split-quality-binding-set",
        "schema_version": 2,
        "mode": "cache_verify_only",
        "diagnostic_only": True,
        "claim_eligible": False,
        "eval_run_id": "eval-current",
        "setup_id": "orin_nx_hailo8_01",
        "binding_set_sha256": "deliberately-drifted",
        "cache_verify_attestation_sha256": "also-drifted",
        "cache_verify_attestation": {
            "status": "verified",
            "artifact_policy": "cache_verify_only",
            "compiler_dispatch_allowed": False,
            "compiler_dispatched": False,
        },
        "bindings_by_model_case_backend": {
            "resnet50|b052|hailo8_to_trt": _semantic_binding(),
        },
    }

    assert matrix._valid_quality_first_binding_set(
        payload, setup_id="orin_nx_hailo8_01",
    )

    payload["cache_verify_attestation"]["compiler_dispatched"] = True
    assert not matrix._valid_quality_first_binding_set(
        payload, setup_id="orin_nx_hailo8_01",
    )


def _matrix_row(result: Path) -> dict[str, Any]:
    return {
        "cases": [{
            "case_id": "b052",
            "status": "ok",
            "result_ok": True,
            "returncode": 0,
            "timed_out": False,
            "child_result_fresh": True,
            "native_split_quality_required": True,
            "native_fifo_result": str(result),
            "native_fifo_result_sha256": "0" * 64,
            "native_fifo_result_size_bytes": 1,
            "eval_run_id": "eval-current",
            "source_run_id": "hailo8_to_trt",
            "native_split_quality_eval_run_id": "eval-current",
            "native_split_quality_source_run_id": "hailo8_to_trt",
            "native_split_quality_binding_sha256": "wrapper-hash",
            "native_split_quality_cache_verify_source_binding_sha256": (
                "wrapper-source"
            ),
            "native_split_quality_cache_verify_replay_sha256": (
                "wrapper-replay"
            ),
            "native_split_quality_binding": {"wrapper": True},
            "native_split_quality_consumer_attestation": {"wrapper": True},
        }],
    }


def test_cache_runner_keeps_current_success_when_only_hash_mirrors_drift(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    runner = _load_script("native_fifo_eval_runner")
    monkeypatch.setenv("ONNX_SPLITPOINT_ARTIFACT_POLICY", "cache_verify_only")
    result = tmp_path / "native_fifo_results.json"
    result.write_text(json.dumps({
        "ok": True,
        "eval_run_id": "eval-current",
        "source_run_id": "hailo8_to_trt",
        "native_split_quality_eval_run_id": "eval-current",
        "native_split_quality_source_run_id": "hailo8_to_trt",
        "native_split_quality_binding_sha256": "raw-hash",
        "native_split_quality_cache_verify_source_binding_sha256": "raw-source",
        "native_split_quality_cache_verify_replay_sha256": "raw-replay",
        "native_split_quality_binding": {"raw": True},
        "native_split_quality_consumer_attestation": {"raw": True},
    }), encoding="utf-8")

    rows = runner._extract_rows(
        "resnet50", tmp_path, _matrix_row(result),
        setup_id="orin_nx_hailo8_01",
    )

    assert len(rows) == 1
    assert rows[0]["result_ok"] is True
    assert rows[0]["status"] == "ok"
    assert any(
        "native_fifo_result_sha256" in item
        for item in rows[0]["cache_verify_diagnostics"]
    )


def test_cache_runner_still_rejects_current_execution_id_drift(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    runner = _load_script("native_fifo_eval_runner")
    monkeypatch.setenv("ONNX_SPLITPOINT_ARTIFACT_POLICY", "cache_verify_only")
    result = tmp_path / "native_fifo_results.json"
    result.write_text(json.dumps({
        "ok": True,
        "eval_run_id": "stale-eval",
        "source_run_id": "hailo8_to_trt",
        "native_split_quality_eval_run_id": "eval-current",
        "native_split_quality_source_run_id": "hailo8_to_trt",
    }), encoding="utf-8")

    rows = runner._extract_rows(
        "resnet50", tmp_path, _matrix_row(result),
        setup_id="orin_nx_hailo8_01",
    )

    assert rows[0]["result_ok"] is False
    assert "eval_run_id" in rows[0]["failure_reason"]


def test_cache_consumer_join_falls_back_only_after_semantic_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native = _load_script("native_hailo_trt_fifo_from_benchmarkset")
    binding = _semantic_binding()
    row = {
        "backend": "hailo8_to_trt",
        "model_id": "resnet50",
        "case_id": "b052",
        "setup_id": "orin_nx_hailo8_01",
        "task": "classification",
        "precision": "float32_layout_fp16",
        "eval_run_id": "eval-current",
        "source_run_id": "hailo8_to_trt",
    }
    monkeypatch.setattr(
        native, "bind_quality_to_native_split",
        lambda **_kwargs: (None, "command_hash_mismatch"),
    )
    monkeypatch.setattr(
        native, "validate_native_split_quality_binding",
        lambda candidate, **_kwargs: (candidate, "portable_test"),
    )
    monkeypatch.setenv("ONNX_SPLITPOINT_ARTIFACT_POLICY", "cache_verify_only")

    joined, status, diagnostic = native._bind_quality_for_execution(
        native_row=row, quality_binding=binding,
    )

    assert joined == binding
    assert status == "cache_verify_semantic_binding_consumed"
    assert diagnostic.endswith("command_hash_mismatch")

    stale = dict(row, eval_run_id="stale-eval")
    with pytest.raises(RuntimeError, match="eval_run_id_mismatch"):
        native._bind_quality_for_execution(
            native_row=stale, quality_binding=binding,
        )

    monkeypatch.setenv("ONNX_SPLITPOINT_ARTIFACT_POLICY", "standard")
    with pytest.raises(RuntimeError, match="consumer_join_failed"):
        native._bind_quality_for_execution(
            native_row=row, quality_binding=binding,
        )
