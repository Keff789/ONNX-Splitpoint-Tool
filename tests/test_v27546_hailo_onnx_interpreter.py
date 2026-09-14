from __future__ import annotations

import builtins
import hashlib
import json
import subprocess
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any

import onnx
import pytest

from scripts import native_full_baseline_eval_runner as full_runner


def _write_raw_head_onnx(path: Path) -> None:
    outputs = [
        onnx.helper.make_tensor_value_info(
            name, onnx.TensorProto.FLOAT, shape,
        )
        for name, shape in (
            ("head_80", [1, 3, 80, 80, 85]),
            ("head_40", [1, 3, 40, 40, 85]),
            ("head_20", [1, 3, 20, 20, 85]),
        )
    ]
    path.write_bytes(
        onnx.helper.make_model(
            onnx.helper.make_graph([], "raw_heads", outputs, outputs),
        ).SerializeToString()
    )


def _marked_probe(payload: dict[str, Any]) -> str:
    return full_runner._HAILO_RAW_HEAD_PROBE_MARKER + json.dumps(payload)


def _valid_probe_outputs() -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "element_type": int(onnx.TensorProto.FLOAT),
            "has_shape": True,
            "dimensions": [
                {"kind": "value", "value": value} for value in shape
            ],
        }
        for name, shape in (
            ("head_80", [1, 3, 80, 80, 85]),
            ("head_40", [1, 3, 40, 40, 85]),
            ("head_20", [1, 3, 20, 20, 85]),
        )
    ]


def test_parent_without_onnx_uses_selected_onnx_capable_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "raw-head.onnx"
    _write_raw_head_onnx(source)
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    compiler_sha = "a" * 64
    real_import = builtins.__import__

    def parent_without_onnx(
        name: str, globals: Any = None, locals: Any = None,
        fromlist: Any = (), level: int = 0,
    ) -> Any:
        if name == "onnx" or name.startswith("onnx."):
            raise ModuleNotFoundError("parent intentionally has no onnx")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", parent_without_onnx)
    diagnostics: dict[str, Any] = {}
    attestation, status = full_runner._onnx_multiscale_raw_head_attestation(
        source,
        source_onnx_sha256=source_sha,
        compiler_onnx_sha256=compiler_sha,
        onnx_python=sys.executable,
        diagnostics_out=diagnostics,
    )

    assert status == "hailo_source_raw_head_attestation_verified"
    assert attestation is not None
    assert [row["shape"] for row in attestation["outputs"]] == [
        [1, 3, 80, 80, 85],
        [1, 3, 40, 40, 85],
        [1, 3, 20, 20, 85],
    ]
    assert diagnostics["verification_status"] == "verified"
    assert diagnostics["probe_returncode"] == 0
    assert diagnostics["onnx_python"] == sys.executable


def test_parent_rejects_noncanonical_success_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "opaque.onnx"
    source.write_bytes(b"parent-hash-bound-source")
    payload = {
        "schema": full_runner._HAILO_RAW_HEAD_PROBE_SCHEMA,
        "schema_version": full_runner._HAILO_RAW_HEAD_PROBE_VERSION,
        "status": "ok",
        "source_onnx_sha256": hashlib.sha256(
            source.read_bytes()
        ).hexdigest(),
        "outputs": _valid_probe_outputs(),
    }
    payload["outputs"][0]["rank"] = 5

    def forged_probe(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=[], returncode=0, stdout=_marked_probe(payload), stderr="",
        )

    monkeypatch.setattr(subprocess, "run", forged_probe)
    diagnostics: dict[str, Any] = {}
    attestation, status = full_runner._onnx_multiscale_raw_head_attestation(
        source,
        source_onnx_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        compiler_onnx_sha256="b" * 64,
        onnx_python=sys.executable,
        diagnostics_out=diagnostics,
    )

    assert attestation is None
    assert status == "hailo_source_raw_head_onnx_unreadable"
    assert diagnostics["failure_phase"] == "validate_parent_payload"
    assert diagnostics["exception_type"] == "RawHeadProbeProtocolError"


def test_parent_rejects_source_onnx_hash_drift_during_child_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "mutable.onnx"
    source.write_bytes(b"receipt-bound-source-before-probe")
    expected_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    payload = {
        "schema": full_runner._HAILO_RAW_HEAD_PROBE_SCHEMA,
        "schema_version": full_runner._HAILO_RAW_HEAD_PROBE_VERSION,
        "status": "ok",
        "source_onnx_sha256": expected_sha,
        "outputs": _valid_probe_outputs(),
    }

    def mutating_probe(
        *_args: Any, **_kwargs: Any,
    ) -> subprocess.CompletedProcess[str]:
        source.write_bytes(b"different-source-after-child-probe")
        return subprocess.CompletedProcess(
            args=[], returncode=0, stdout=_marked_probe(payload), stderr="",
        )

    monkeypatch.setattr(subprocess, "run", mutating_probe)
    diagnostics: dict[str, Any] = {}
    attestation, status = full_runner._onnx_multiscale_raw_head_attestation(
        source,
        source_onnx_sha256=expected_sha,
        compiler_onnx_sha256="d" * 64,
        onnx_python=sys.executable,
        diagnostics_out=diagnostics,
    )

    assert attestation is None
    assert status == "hailo_source_raw_head_onnx_identity_drift"
    assert diagnostics["failure_phase"] == (
        "validate_parent_identity_post_probe"
    )
    assert diagnostics["exception_type"] == "OnnxIdentityDriftError"
    assert diagnostics["expected_source_onnx_sha256"] == expected_sha
    assert diagnostics["observed_source_onnx_sha256"] == hashlib.sha256(
        source.read_bytes()
    ).hexdigest()


def test_parent_rejects_swap_restore_around_child_parse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "swap-restore.onnx"
    original_bytes = b"receipt-bound-source-before-probe"
    parsed_bytes = b"different-source-seen-only-by-child"
    source.write_bytes(original_bytes)
    expected_sha = hashlib.sha256(original_bytes).hexdigest()
    payload = {
        "schema": full_runner._HAILO_RAW_HEAD_PROBE_SCHEMA,
        "schema_version": full_runner._HAILO_RAW_HEAD_PROBE_VERSION,
        "status": "ok",
        "source_onnx_sha256": hashlib.sha256(parsed_bytes).hexdigest(),
        "outputs": _valid_probe_outputs(),
    }

    def swap_restore_probe(
        *_args: Any, **_kwargs: Any,
    ) -> subprocess.CompletedProcess[str]:
        source.write_bytes(parsed_bytes)
        source.write_bytes(original_bytes)
        return subprocess.CompletedProcess(
            args=[], returncode=0, stdout=_marked_probe(payload), stderr="",
        )

    monkeypatch.setattr(subprocess, "run", swap_restore_probe)
    diagnostics: dict[str, Any] = {}
    attestation, status = full_runner._onnx_multiscale_raw_head_attestation(
        source,
        source_onnx_sha256=expected_sha,
        compiler_onnx_sha256="e" * 64,
        onnx_python=sys.executable,
        diagnostics_out=diagnostics,
    )

    assert attestation is None
    assert status == "hailo_source_raw_head_onnx_identity_drift"
    assert diagnostics["failure_phase"] == "validate_child_parsed_identity"
    assert diagnostics["exception_type"] == "OnnxIdentityDriftError"
    assert diagnostics["expected_source_onnx_sha256"] == expected_sha
    assert diagnostics["observed_source_onnx_sha256"] == hashlib.sha256(
        parsed_bytes
    ).hexdigest()


def test_initial_source_hash_io_error_is_structured_and_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "unreadable-during-hash.onnx"
    source.write_bytes(b"receipt-bound-source")
    expected_sha = hashlib.sha256(source.read_bytes()).hexdigest()

    def failed_hash(_path: Path) -> str:
        raise PermissionError("source became unreadable")

    monkeypatch.setattr(full_runner, "_sha256_file", failed_hash)
    diagnostics: dict[str, Any] = {}
    attestation, status = full_runner._onnx_multiscale_raw_head_attestation(
        source,
        source_onnx_sha256=expected_sha,
        compiler_onnx_sha256="f" * 64,
        onnx_python=sys.executable,
        diagnostics_out=diagnostics,
    )

    assert attestation is None
    assert status == "hailo_source_raw_head_onnx_identity_invalid"
    assert diagnostics["failure_phase"] == "validate_parent_identity"
    assert diagnostics["exception_type"] == "PermissionError"
    assert diagnostics["exception_detail"] == "source became unreadable"
    assert diagnostics["expected_source_onnx_sha256"] == expected_sha


def test_child_import_exception_type_and_detail_are_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "opaque.onnx"
    source.write_bytes(b"parent-hash-bound-source")
    payload = {
        "schema": full_runner._HAILO_RAW_HEAD_PROBE_SCHEMA,
        "schema_version": full_runner._HAILO_RAW_HEAD_PROBE_VERSION,
        "status": "error",
        "failure_phase": "import_onnx",
        "exception_type": "ModuleNotFoundError",
        "exception_detail": "No module named 'onnx' in selected child",
    }

    def failed_probe(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=[], returncode=21, stdout=_marked_probe(payload),
            stderr="child stderr evidence",
        )

    monkeypatch.setattr(subprocess, "run", failed_probe)
    diagnostics: dict[str, Any] = {}
    attestation, status = full_runner._onnx_multiscale_raw_head_attestation(
        source,
        source_onnx_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        compiler_onnx_sha256="c" * 64,
        onnx_python=sys.executable,
        diagnostics_out=diagnostics,
    )

    assert attestation is None
    assert status == "hailo_source_raw_head_onnx_unreadable"
    assert diagnostics["failure_phase"] == "import_onnx"
    assert diagnostics["exception_type"] == "ModuleNotFoundError"
    assert diagnostics["exception_detail"] == (
        "No module named 'onnx' in selected child"
    )
    assert diagnostics["probe_returncode"] == 21
    assert diagnostics["probe_stderr_tail"] == "child stderr evidence"


def test_native_failure_row_keeps_receipt_probe_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark_set = tmp_path / "benchmark_set"
    hef = benchmark_set / "hailo" / "hailo8" / "full" / "compiled.hef"
    hef.parent.mkdir(parents=True)
    hef.write_bytes(b"hef")

    monkeypatch.setattr(
        full_runner, "_find_hailo_full_hef", lambda *_args: hef,
    )

    def failed_receipt(**kwargs: Any) -> tuple[None, str]:
        diagnostics = kwargs["diagnostics_out"]
        diagnostics.update({
            "failure_phase": "import_onnx",
            "exception_type": "ModuleNotFoundError",
            "exception_detail": "No module named 'onnx'",
        })
        assert kwargs["onnx_python"] == "/engine/onnx-python"
        return None, "hailo_source_raw_head_onnx_unreadable"

    monkeypatch.setattr(
        full_runner, "_verified_hailo_full_build_receipt", failed_receipt,
    )
    row = full_runner._native_hailo_full(
        benchmark_set,
        "yolov7_paper",
        "hailo8",
        Namespace(
            frames=3,
            duration_s=0.0,
            expected_hef_sha256="",
            engine_python_selected="/engine/onnx-python",
        ),
    )

    assert row["failure_reason"] == "hailo_source_raw_head_onnx_unreadable"
    assert row["returncode"] == 4
    assert row["status_detail"] == (
        "ModuleNotFoundError: No module named 'onnx'"
    )
    assert row["error"] == row["status_detail"]
    assert row["hailo_hef_build_receipt_diagnostics"] == {
        "failure_phase": "import_onnx",
        "exception_type": "ModuleNotFoundError",
        "exception_detail": "No module named 'onnx'",
    }


def test_packaged_native_full_runner_is_byte_identical() -> None:
    root = Path(__file__).resolve().parents[1]
    assert (
        root / "scripts" / "native_full_baseline_eval_runner.py"
    ).read_bytes() == (
        root
        / "onnx_splitpoint_tool"
        / "resources"
        / "remote_scripts"
        / "native_full_baseline_eval_runner.py"
    ).read_bytes()
