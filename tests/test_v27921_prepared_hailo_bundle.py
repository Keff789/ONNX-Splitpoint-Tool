from pathlib import Path
import json

import pytest

from onnx_splitpoint_tool.hailo_backend import _hailo_cache_key, _write_hailo_receipt
from onnx_splitpoint_tool.preprocessing_contract import (
    canonical_image_preprocessing_contract,
    preprocessing_contract_sha256,
)
from onnx_splitpoint_tool.workflow.benchmark_binding import materialize_backend_artifact_decisions


def _prepare(root: Path, hef_bytes: bytes) -> tuple[Path, dict]:
    source_dir = root / "prepared"
    source_dir.mkdir(exist_ok=True)
    hef = source_dir / "compiled.hef"
    hef.write_bytes(hef_bytes)
    source = source_dir / "source.onnx"
    source.write_bytes(b"source-model")
    compiler = source_dir / "compiler.onnx"
    compiler.write_bytes(b"compiler-model")
    preprocessing = canonical_image_preprocessing_contract("detection", (640, 640))
    key, payload = _hailo_cache_key(
        model_path=compiler, activation_part1=None, hw_arch="hailo8",
        opt_level=1, calib_dir=None, calib_count=64, calib_batch_size=8,
        extra_model_script="", start_nodes=None, end_nodes=None,
        preprocessing_contract=preprocessing,
    )
    _write_hailo_receipt(
        hef_path=hef, source_onnx=source, compiler_onnx=compiler,
        hw_arch="hailo8", net_name="prepared_test",
        preprocessing_contract=preprocessing,
        preprocessing_sha256=preprocessing_contract_sha256(preprocessing),
        cache_key=key, cache_payload=payload,
        calibration_identity=payload["calibration_identity"],
        calibration_count=payload["calibration_count"],
    )
    suite = root / "run/models/yolo26s/benchmark_set/legacy_suite"
    suite.mkdir(parents=True, exist_ok=True)
    if not (suite / "benchmark_set.json").is_file():
        (suite / "benchmark_set.json").write_text(json.dumps({"model_id": "yolo26s", "cases": []}))
    args = {
        "run_dir": root / "run", "model_id": "yolo26s", "targets": ["hailo8"],
        "full_baseline_plan": {"task": "detection", "baselines": [{
            "backend": "hailo8", "variant": "full", "artifact_path": str(hef),
            "source_onnx_path": str(source), "compiler_onnx_path": str(compiler),
        }]},
        "output_contracts": {"task": "detection", "contracts": []},
        "benchmark_set_contract": {"materialized": True, "legacy_suite_dir": str(suite)},
    }
    return suite / "hailo/hailo8/full/compiled.hef", args


def _decision(result: dict) -> dict:
    return json.loads(result["artifacts"]["backend_artifact_decisions_json"].read_text())["baseline_decisions"][0]


def test_prepared_baseline_publishes_all_three_and_repeated_binding_reuses_generation(tmp_path: Path) -> None:
    dest, args = _prepare(tmp_path, b"first-hef")
    assert _decision(materialize_backend_artifact_decisions(**args))["copied_to_suite"] is True
    assert dest.is_symlink()
    committed = dest.resolve()
    receipt = committed.parent / "hailo_hef_build_receipt.json"
    meta = committed.parent / "cache_meta.json"
    assert receipt.is_file() and meta.is_file()
    assert json.loads(meta.read_text())["hef_sha256"] == json.loads(receipt.read_text())["hef_sha256"]
    assert _decision(materialize_backend_artifact_decisions(**args))["copied_to_suite"] is True
    assert dest.resolve() == committed


def test_prepared_baseline_commit_failure_preserves_previous_complete_generation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from onnx_splitpoint_tool import hailo_cache_bundle

    dest, args = _prepare(tmp_path, b"first-hef")
    assert _decision(materialize_backend_artifact_decisions(**args))["copied_to_suite"] is True
    previous = dest.resolve()
    old_bytes = {name: (previous.parent / name).read_bytes() for name in (
        "compiled.hef", "hailo_hef_build_receipt.json", "cache_meta.json",
    )}
    _, args = _prepare(tmp_path, b"second-hef")
    replace_link = hailo_cache_bundle._replace_link

    def fail_commit(path: Path, target: str) -> None:
        if path.name == hailo_cache_bundle.POINTER_NAME:
            raise OSError("injected commit failure")
        replace_link(path, target)

    monkeypatch.setattr(hailo_cache_bundle, "_replace_link", fail_commit)
    decision = _decision(materialize_backend_artifact_decisions(**args))
    assert decision["decision"] == "reuse_copy_failed"
    assert "injected commit failure" in decision["error_detail"]
    assert dest.resolve() == previous
    assert {name: (previous.parent / name).read_bytes() for name in old_bytes} == old_bytes
