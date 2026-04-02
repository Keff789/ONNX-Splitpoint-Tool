from pathlib import Path

CONTROLLER = Path("onnx_splitpoint_tool/gui/benchmark_workflow.py")


def test_execution_config_constructor_does_not_receive_orchestration_only_hailo_parse_check_fn() -> None:
    src = CONTROLLER.read_text(encoding="utf-8")
    marker = "execution_cfg = BenchmarkGenerationExecutionConfig("
    start = src.find(marker)
    assert start != -1, "execution config constructor not found"
    end = src.find("execution_callbacks = BenchmarkGenerationExecutionCallbacks(", start)
    assert end != -1, "execution config block terminator not found"
    block = src[start:end]
    assert "hailo_parse_check_fn=" not in block


def test_orchestration_config_still_receives_hailo_parse_check_fn() -> None:
    src = CONTROLLER.read_text(encoding="utf-8")
    marker = "orchestration_cfg = BenchmarkGenerationOrchestrationConfig("
    start = src.find(marker)
    assert start != -1, "orchestration config constructor not found"
    end = src.find("orchestration_result = orchestration_service.run(orchestration_cfg)", start)
    assert end != -1, "orchestration config block terminator not found"
    block = src[start:end]
    assert "hailo_parse_check_fn=hailo_parse_check_fn" in block
