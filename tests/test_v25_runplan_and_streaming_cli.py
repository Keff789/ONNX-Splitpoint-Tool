from __future__ import annotations

from pathlib import Path

from onnx_splitpoint_tool.benchmark.services import BenchmarkGenerationService
from onnx_splitpoint_tool.cli import main


def test_streaming_preset_roundtrip():
    service = BenchmarkGenerationService()
    default = service.resolve_streaming_preset("default")
    assert default == {"frames": 24, "warmup": 6, "queue_depth": 2}
    assert service.detect_streaming_preset(24, 6, 2) == "default"
    assert service.detect_streaming_preset(999, 1, 1) == "custom"


def test_build_run_plan_matrix_and_hailo_flags():
    service = BenchmarkGenerationService()
    plan = service.build_run_plan(
        acc_cpu=False,
        acc_cuda=True,
        acc_trt=True,
        acc_h8=True,
        acc_h10=False,
        hailo8_hw="hailo8",
        hailo10_hw="",
        image_scale="auto",
        hailo_preset="Everything",
        hailo_custom_full=True,
        hailo_custom_composed=True,
        hailo_custom_part1=True,
        hailo_custom_part2=True,
        matrix_trt_to_hailo=True,
        matrix_hailo_to_trt=False,
        full_hef_policy="end",
    )
    run_ids = [str(r.get("id")) for r in plan.bench_plan_runs]
    assert "ort_cuda" in run_ids
    assert "ort_tensorrt" in run_ids
    assert "hailo8" in run_ids
    assert "trt_to_hailo8" in run_ids
    assert plan.hef_targets == ["hailo8"]
    assert plan.hailo_selected is True
    assert plan.hef_full is True
    assert plan.hef_part1 is True
    assert plan.hef_part2 is True


def test_cli_benchmark_remote(monkeypatch, tmp_path: Path, capsys):
    calls = {}

    class DummyRemoteService:
        def run(self, **kwargs):
            calls.update(kwargs)
            return {"status": "ok", "local_run_dir": str(tmp_path / "run")}

    monkeypatch.setattr("onnx_splitpoint_tool.cli.RemoteBenchmarkService", DummyRemoteService)
    rc = main([
        "benchmark-remote",
        str(tmp_path / "benchmark_set.json"),
        "--host",
        "user@example.org",
        "--working-dir",
        str(tmp_path / "work"),
        "--provider",
        "auto",
        "--throughput-frames",
        "12",
        "--throughput-warmup-frames",
        "3",
        "--throughput-queue-depth",
        "2",
    ])
    out = capsys.readouterr().out
    assert rc == 0
    assert "\"status\": \"ok\"" in out
    assert calls["host"].host == "example.org"
    assert calls["host"].user == "user"
    assert calls["args"].throughput_frames == 12
    assert calls["args"].throughput_warmup_frames == 3
    assert calls["args"].throughput_queue_depth == 2
