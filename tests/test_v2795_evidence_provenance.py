from __future__ import annotations

import copy
import importlib.util
import json
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import onnx_splitpoint_tool as pkg


ROOT = Path(__file__).resolve().parents[1]
CONCURRENT_HELPER = (
    ROOT / "scripts/native_hailo_trt_concurrent_three_stage_from_benchmarkset.py"
)
CONCURRENT_MIRROR = (
    ROOT
    / "onnx_splitpoint_tool/resources/remote_scripts"
    / "native_hailo_trt_concurrent_three_stage_from_benchmarkset.py"
)
FIFO_HELPER = ROOT / "scripts/native_hailo_trt_fifo_from_benchmarkset.py"
FIFO_MIRROR = (
    ROOT
    / "onnx_splitpoint_tool/resources/remote_scripts"
    / "native_hailo_trt_fifo_from_benchmarkset.py"
)
FIXTURE = ROOT / "tests/fixtures/v2793/yolov7_b066_three_stage_report.json"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    # Python 3.12's dataclass processing resolves postponed annotations via
    # ``sys.modules`` while the module is executing.  Mirror normal import
    # semantics instead of invoking the loader with an unregistered module.
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def test_concurrent_evidence_runner_mirrors_are_byte_identical():
    assert CONCURRENT_HELPER.read_bytes() == CONCURRENT_MIRROR.read_bytes()
    assert FIFO_HELPER.read_bytes() == FIFO_MIRROR.read_bytes()


def test_source_snapshot_uses_current_package_version_prefix(tmp_path: Path):
    module = _load(CONCURRENT_HELPER, "v2795_snapshot_current")
    archive = module._create_source_snapshot(ROOT, tmp_path / "source_snapshot.zip")
    expected_prefix = f"ONNX-Splitpoint-Tool_v{pkg.__version__}/"
    with zipfile.ZipFile(archive) as zf:
        names = zf.namelist()
    assert len(names) == 5
    assert all(name.startswith(expected_prefix) for name in names)
    assert not any(name.startswith("ONNX-Splitpoint-Tool_v2.79.2/") for name in names)


def test_source_snapshot_prefix_is_version_dynamic(tmp_path: Path):
    module = _load(CONCURRENT_HELPER, "v2795_snapshot_dynamic")
    tool_root = tmp_path / "synthetic_tool"
    members = (
        "onnx_splitpoint_tool/__init__.py",
        "onnx_splitpoint_tool/release_identity.py",
        "onnx_splitpoint_tool/native_three_stage.py",
        "scripts/native_hailo_trt_fifo_from_benchmarkset.py",
        "scripts/native_hailo_trt_concurrent_three_stage_from_benchmarkset.py",
    )
    for rel in members:
        path = tool_root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if rel.endswith("release_identity.py"):
            payload = 'VERSION = "9.8.7"\n'
        elif rel.endswith("__init__.py"):
            payload = "from .release_identity import VERSION as __version__\n"
        else:
            payload = "# fixture\n"
        path.write_text(payload, encoding="utf-8")

    archive = module._create_source_snapshot(tool_root, tmp_path / "dynamic.zip")
    with zipfile.ZipFile(archive) as zf:
        assert set(zf.namelist()) == {
            f"ONNX-Splitpoint-Tool_v9.8.7/{rel}" for rel in members
        }


def test_product_projection_fails_closed_when_oracle_parity_failed(tmp_path: Path):
    module = _load(CONCURRENT_HELPER, "v2795_oracle_projection")
    report = copy.deepcopy(json.loads(FIXTURE.read_text(encoding="utf-8")))
    report["postflight_quality_oracle"]["all_exact"] = False
    binding_path = tmp_path / "binding.json"
    binding_path.write_text("{}\n", encoding="utf-8")
    args = SimpleNamespace(
        model_id="yolov7_paper",
        case="b066",
        setup_id="orin_nx_hailo8_01",
        eval_run_id="fixture",
        source_run_id="hailo8_to_trt",
        precision="uint8_dequant_fp16",
        native_split_quality_binding=str(binding_path),
    )

    result = module._project_result(
        report=report,
        args=args,
        binding={},
        report_path=tmp_path / "report.json",
        resource=tmp_path,
    )

    assert result["oracle_parity"]["status"] == "failed"
    assert result["quality_oracle_status"] == "failed"
    assert result["phases"]["quality_oracle"]["status"] == "failed"
    assert result["ok"] is False
    assert result["endpoint_relation_verified"] is False
    assert result["failure_reason"] == "oracle_parity_status:failed"


def test_fifo_evidence_labels_and_oracle_gate_track_current_release():
    module = _load(FIFO_HELPER, "v2795_fifo_evidence")
    assert module._CONCURRENT_LOG_PREFIX == (
        f"[native-fifo][v{pkg.__version__}][concurrent]"
    )
    assert "[native-fifo][v2.79.2][concurrent]" not in FIFO_HELPER.read_text(
        encoding="utf-8"
    )
    assert module._v2791_oracle_parity_passed(
        {
            "quality_oracle_status": "passed",
            "oracle_parity": {"status": "passed"},
        }
    )
    assert not module._v2791_oracle_parity_passed(
        {
            "quality_oracle_status": "passed",
            "oracle_parity": {"status": "failed"},
        }
    )
    assert not module._v2791_oracle_parity_passed(
        {"quality_oracle_status": "passed"}
    )
