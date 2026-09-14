from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/native_hailo_trt_fifo_from_benchmarkset.py"
HELPER = ROOT / "scripts/native_hailo_trt_concurrent_three_stage_from_benchmarkset.py"


def _module():
    spec = importlib.util.spec_from_file_location("v2791_runner", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _args(**updates):
    values = dict(
        task="detection", detection_endpoints="dual",
        completion_runtime_mode="fast_oracle_outside_timing",
        hw_arch="hailo8", case="b066", precision="uint8_dequant_fp16",
        model_id="yolov7_paper", native_split_quality_binding="/tmp/binding.json",
        run=True, benchmark_set="/tmp/benchmark_set", image="/tmp/image.jpg",
        frames=1000, warmup=100, repetitions=3, queue_depth=3,
        post_queue_depth=4, duration_s=0.0, setup_id="orin_nx_hailo8_01",
        eval_run_id="run", source_run_id="hailo8_to_trt",
        expected_image_sha256="image", expected_hef_sha256="hef",
        expected_engine_sha256="engine",
        expected_boundary_layout="memory_nhwc_to_nchw",
        concurrent_timeout_s=3600.0,
    )
    values.update(updates)
    return SimpleNamespace(**values)


def test_v2791_source_identity() -> None:
    import onnx_splitpoint_tool as pkg
    from onnx_splitpoint_tool.release_identity import BUILD_ID, RELEASE, VERSION
    from onnx_splitpoint_tool.workflow.runner import WORKFLOW_VERSION
    assert pkg.__version__ == VERSION
    assert pkg.__release__ == RELEASE
    assert pkg.__build_id__ == BUILD_ID
    assert WORKFLOW_VERSION == pkg.__build_id__
    assert HELPER.is_file()


def test_v2791_exact_contract_is_admitted_fail_closed() -> None:
    m = _module()
    assert m._v2791_use_concurrent_three_stage(_args()) is True
    assert m._v2791_use_concurrent_three_stage(_args(case="b065")) is False
    assert m._v2791_use_concurrent_three_stage(_args(hw_arch="hailo10")) is False
    assert m._v2791_use_concurrent_three_stage(_args(precision="uint8_cast_fp16")) is False
    assert m._v2791_use_concurrent_three_stage(_args(model_id="yolo26s")) is False
    assert m._v2791_use_concurrent_three_stage(_args(native_split_quality_binding="")) is False


def test_v2791_command_uses_binding_and_concurrent_helper() -> None:
    m = _module(); args = _args()
    cmd = m._v2791_concurrent_three_stage_command(
        args=args, work=Path("/tmp/work"), result_json=Path("/tmp/result.json")
    )
    assert str(HELPER) in cmd
    assert cmd.count("--native-split-quality-binding") == 1
    i = cmd.index("--native-split-quality-binding")
    assert cmd[i + 1] == args.native_split_quality_binding
    assert "--post-queue-depth" in cmd
    assert "--expected-boundary-layout" in cmd


def test_v2791_legacy_dual_children_both_keep_quality_binding() -> None:
    m = _module(); args = _args()
    args.build=True; args.device_id=""; args.expected_runner_sha256=""
    args.source_contract_sha256=""; args.expected_executable_sha256=""
    args.raw_preprocess_scope="paper_image"; args.preprocess_mode="letterbox"
    args.letterbox_pad_value=114; args.hailo_format="uint8"; args.copy_outputs=True
    for endpoint in ("raw_model_outputs", "completed_task"):
        cmd=m._dual_child_command(args=args,endpoint=endpoint,work=Path("/tmp/work"),
            result_json=Path(f"/tmp/{endpoint}.json"),
            config_json=Path(f"/tmp/{endpoint}-config.json"))
        assert "--native-split-quality-binding" in cmd
        i=cmd.index("--native-split-quality-binding")
        assert cmd[i+1] == args.native_split_quality_binding
