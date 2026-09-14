from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load_gate():
    path = ROOT / "scripts" / "verify_v2798_yolo11_r8b_gate.py"
    spec = importlib.util.spec_from_file_location("v2798_recovery_gate", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gate = _load_gate()


def _manifest(tmp_path: Path) -> dict:
    return {
        "schema": "onnx-splitpoint/yolo11-r8b-recovery-manifest",
        "schema_version": 1,
        "created_at": "2026-09-01T00:00:00+00:00",
        "target_version": "2.79.8",
        "target_build_id": "v2.79.8-yolo11-gate-profile-schema-closure",
        "source_version": "2.79.6",
        "source_build_id": "v2.79.6-remaining-changes-yolo11-admission-closure",
        "source_access_mode": "read_only_hash_bound_no_copy",
        "source_run_dir": str(tmp_path / "preserved_v2796_run"),
        "source_run_id": "preserved_v2796_run",
        "model_id": "yolo11l",
        "model_sha256": gate.MODEL_SHA256,
        "imported_terminal_count": 1,
        "imports": {
            "hailo8_b067_composed": {
                "terminal_eligible": True,
                "evidence_mode": "read_only_import",
                "source_case_id": "b067",
                "source_row_sha256": "1" * 64,
                "result": {"relative_path": "result.json", "sha256": "2" * 64, "size_bytes": 9},
                "runtime_target": {
                    "setup_id": "orin_nx_hailo8_01",
                    "provider": "hailo8",
                    "run_id": "hailo8_to_trt",
                },
                "runtime_contract": {
                    "run_id": "hailo8_to_trt",
                    "backend": "hailo8_to_tensorrt",
                    "stage1": "hailo8",
                    "stage2": "native_tensorrt",
                    "variant": "composed",
                    "throughput_mode": "measured_streaming",
                    "streaming_impl": "native_fifo",
                    "fps_makespan": 37.09,
                },
            },
        },
        "diagnostic_only": {
            "hailo10h_full_generic": {
                "terminal_eligible": False,
                "evidence_mode": "diagnostic_only",
                "reason": "v2796_generic_full_runner_not_native_full",
            },
        },
        "fresh_required": list(gate.FRESH_REQUIRED),
        "policy": {
            "only_terminal_import": "hailo8_b067_composed",
            "generic_full_rows_terminal_eligible": False,
            "source_artifact_index_trusted": False,
            "source_files_verified_individually": True,
        },
    }


def _write(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_recovery_manifest_is_recomputed_and_hash_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _manifest(tmp_path)
    path = tmp_path / "v2798_recovery_manifest.json"
    _write(path, payload)
    recomputed = json.loads(json.dumps(payload))
    recomputed["created_at"] = "2026-09-01T00:00:01+00:00"
    monkeypatch.setattr(
        gate.recovery_tool, "build_manifest", lambda _source: recomputed,
    )

    verified, identity = gate._validate_recovery_manifest(path)
    assert verified["imports"].keys() == {"hailo8_b067_composed"}
    assert identity["path"] == str(path)
    assert identity["sha256"] == gate.base._file_sha(path)
    assert identity["size_bytes"] == path.stat().st_size


def test_recovery_rejects_source_hash_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _manifest(tmp_path)
    path = tmp_path / "v2798_recovery_manifest.json"
    _write(path, payload)
    recomputed = json.loads(json.dumps(payload))
    recomputed["imports"]["hailo8_b067_composed"]["source_row_sha256"] = "3" * 64
    monkeypatch.setattr(
        gate.recovery_tool, "build_manifest", lambda _source: recomputed,
    )
    with pytest.raises(gate.GateError, match="current_source_hash_mismatch"):
        gate._validate_recovery_manifest(path)


def test_recovery_never_promotes_generic_hailo10_full(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _manifest(tmp_path)
    payload["diagnostic_only"]["hailo10h_full_generic"]["terminal_eligible"] = True
    path = tmp_path / "v2798_recovery_manifest.json"
    _write(path, payload)
    monkeypatch.setattr(
        gate.recovery_tool, "build_manifest", lambda _source: payload,
    )
    with pytest.raises(gate.GateError, match="generic_terminal_eligible"):
        gate._validate_recovery_manifest(path)


def test_imported_verdict_path_has_exact_frozen_identity(tmp_path: Path) -> None:
    payload = _manifest(tmp_path)
    receipt = {"sha256": "4" * 64}
    row = gate._imported_path(payload, receipt)
    assert (
        row["backend"], row["setup_id"], row["run_id"],
        row["variant"], row["evidence_mode"], row["terminal_status"],
    ) == (
        "hailo8_to_tensorrt", "orin_nx_hailo8_01", "hailo8_to_trt",
        "composed", "read_only_import", "success",
    )
    assert row["runtime_attestation"] == {
        "status": "passed",
        "provider_bound": True,
        "hardware_bound": True,
        "variant_bound": True,
        "current_artifact_bytes_bound": True,
        "makespan_bound": True,
    }
