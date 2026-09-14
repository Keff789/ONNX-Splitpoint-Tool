from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import onnx
import pytest
from onnx import TensorProto, helper

from onnx_splitpoint_tool.runners import native_split_quality_runtime as runtime


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_source_part2(path: Path) -> None:
    graph = helper.make_graph(
        [helper.make_node("Identity", ["cut"], ["output0"])],
        "quality-first-real-builder-reuse",
        [helper.make_tensor_value_info(
            "cut", TensorProto.FLOAT, [1, 2, 2, 2],
        )],
        [helper.make_tensor_value_info(
            "output0", TensorProto.FLOAT, [1, 2, 2, 2],
        )],
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(helper.make_model(graph), str(path))


def _write_fake_trtexec(path: Path) -> None:
    """Install only the external TensorRT boundary used by the real builder.

    The regression deliberately executes the packaged
    ``native_trt_from_benchmarkset.py`` in a real child process.  TensorRT is
    the only unavailable hardware dependency in the local test environment,
    so this executable models its narrow CLI contract and records actual
    engine builds.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""#!{sys.executable}
import hashlib
import os
import sys
from pathlib import Path

log = Path(os.environ["QUALITY_FIRST_FAKE_TRTEXEC_LOG"])
with log.open("a", encoding="utf-8") as handle:
    handle.write(" ".join(sys.argv[1:]) + "\\n")

if "--help" in sys.argv[1:]:
    print("--memPoolSize")
    raise SystemExit(0)

onnx_arg = next((value for value in sys.argv[1:] if value.startswith("--onnx=")), "")
engine_arg = next((value for value in sys.argv[1:] if value.startswith("--saveEngine=")), "")
if not onnx_arg or not engine_arg:
    raise SystemExit(64)
source = Path(onnx_arg.split("=", 1)[1])
engine = Path(engine_arg.split("=", 1)[1])
engine.parent.mkdir(parents=True, exist_ok=True)
engine.write_bytes(
    b"quality-first-fake-tensorrt-engine-v1\\0"
    + hashlib.sha256(source.read_bytes()).digest()
)
raise SystemExit(0)
""",
        encoding="utf-8",
    )
    path.chmod(0o755)


def _build_invocations(path: Path) -> list[str]:
    if not path.is_file():
        return []
    return [
        line for line in path.read_text(encoding="utf-8").splitlines()
        if "--saveEngine=" in line
    ]


def test_quality_first_real_builder_verify_first_preserves_sealed_leaf_and_rebuilds_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark_set = tmp_path / "benchmark_set"
    case_dir = benchmark_set / "b038"
    source_part1 = case_dir / "hailo/hailo8/part1/compiled.hef"
    source_part1.parent.mkdir(parents=True, exist_ok=True)
    source_part1.write_bytes(b"temporary-hailo8-part1-v1")
    _write_source_part2(case_dir / "yolo26s_part2_native.onnx")

    def fake_hailo_metadata(
        *, part1: Path, part2_input: dict[str, Any], policy: dict[str, Any],
    ) -> dict[str, Any]:
        assert part1 == source_part1.resolve()
        assert part2_input == {
            "name": "cut", "shape": [1, 2, 2, 2], "dtype": "float32",
        }
        assert policy["precision"] == "uint8_dequant_fp16"
        return {
            "name": "cut",
            "runtime_name": "cut/hailort",
            "shape": [2, 2, 2],
            "canonical_part2_shape": [1, 2, 2, 2],
            "dtype": "uint8",
            "quantization": {
                "source": "hailort_hef_output_vstream_info",
                "scale": 0.03125,
                "zero_point": 17.0,
            },
        }

    monkeypatch.setattr(runtime, "_hailo_metadata", fake_hailo_metadata)
    fake_bin = tmp_path / "fake-bin"
    fake_trtexec = fake_bin / "trtexec"
    fake_log = tmp_path / "fake-trtexec-invocations.log"
    _write_fake_trtexec(fake_trtexec)
    monkeypatch.setenv(
        "PATH", str(fake_bin) + os.pathsep + os.environ.get("PATH", ""),
    )
    monkeypatch.setenv("QUALITY_FIRST_FAKE_TRTEXEC_LOG", str(fake_log))
    monkeypatch.delenv("ONNX_SPLITPOINT_ARTIFACT_POLICY", raising=False)

    kwargs = {
        "benchmark_set": benchmark_set,
        "case_id": "b038",
        "model_id": "yolo26s",
        "setup_id": "hailo8_setup",
        "backend": "hailo8_to_trt",
        "eval_run_id": "quality-first-real-builder-test",
        "source_run_id": "hailo8_to_trt",
        "cache_root": tmp_path / "persistent-cache",
        "output_path": tmp_path / "collected/native_split_quality_binding.json",
        "workspace_mb": 64,
        "timeout_s": 90,
    }

    cold = runtime.prepare_native_split_quality_binding(**kwargs)
    engine = Path(cold["engine_path"])
    receipt = Path(cold["receipt_path"])
    assert _build_invocations(fake_log) and len(_build_invocations(fake_log)) == 1
    engine_identity = (
        _sha256(engine), engine.stat().st_mtime_ns, engine.read_bytes(),
    )
    receipt_identity = (
        _sha256(receipt), receipt.stat().st_mtime_ns, receipt.read_bytes(),
    )

    # This is the real packaged builder's --no-build receipt-verification
    # path, not a mocked subprocess.  It may reproduce bridge metadata, but it
    # must not touch either sealed leaf artifact.
    warm = runtime.prepare_native_split_quality_binding(**kwargs)
    assert "--no-build" in warm["binding"]["producer_command"]
    assert len(_build_invocations(fake_log)) == 1
    assert (
        _sha256(engine), engine.stat().st_mtime_ns, engine.read_bytes(),
    ) == engine_identity
    assert (
        _sha256(receipt), receipt.stat().st_mtime_ns, receipt.read_bytes(),
    ) == receipt_identity
    assert (
        warm["binding"]["native_trt_meta"]["engine_build_receipt_status"]
        == "engine_build_receipt_verified"
    )

    # A modified engine fails the real receipt check.  The verify-first call
    # is rejected and the producer then enters the ordinary builder path,
    # which is the only path permitted to replace engine and receipt.
    engine.write_bytes(b"deliberately-tampered-engine")
    assert _sha256(engine) != engine_identity[0]
    rebuilt = runtime.prepare_native_split_quality_binding(**kwargs)
    assert "--no-build" not in rebuilt["binding"]["producer_command"]
    assert len(_build_invocations(fake_log)) == 2
    assert _sha256(engine) == engine_identity[0]
    assert _sha256(receipt) == receipt_identity[0]
    assert (
        rebuilt["binding"]["native_trt_meta"]["engine_build_receipt_status"]
        == "engine_build_receipt_verified"
    )
