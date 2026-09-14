from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace
import threading
import time

from onnx_splitpoint_tool.workflow.execution_binding import (
    _performance_run_ids_v263,
    execute_benchmark_suite_if_requested,
)
from onnx_splitpoint_tool.workflow.runner import _performance_plan_v263


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_central_performance_plan_removes_only_explicit_cpu_reference() -> None:
    plan = {
        "runs": [
            {"id": "ort_cpu", "provider": "cpu"},
            {"id": "semantic_reference", "provider": "onnxruntime_cpu"},
            {"id": "cuda_ort", "provider": "cuda"},
            {"id": "trt", "provider": "tensorrt"},
            {"id": "deepx", "provider": "deepx_m1"},
        ]
    }
    assert _performance_run_ids_v263(plan) == ["cuda_ort", "trt", "deepx"]
    filtered = _performance_plan_v263(
        plan,
        {"quality_gate": {"statistics": {"execution_location": "central_management"}}},
    )
    assert [row["id"] for row in filtered["runs"]] == ["cuda_ort", "trt", "deepx"]
    assert filtered["performance_excluded_run_ids"] == ["ort_cpu", "semantic_reference"]


def test_management_runner_template_honours_explicit_ort_thread_contract() -> None:
    template = (
        Path(__file__).parents[1]
        / "onnx_splitpoint_tool/resources/templates/run_split_onnxruntime.py.txt"
    ).read_text(encoding="utf-8")
    ast.parse(template)
    assert 'os.environ.get("ONNX_SPLITPOINT_CPU_THREADS")' in template
    assert "sess_options.intra_op_num_threads = _cpu_threads" in template
    assert "sess_options.inter_op_num_threads = 1" in template


def test_local_suite_cancel_terminates_promptly(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    suite = tmp_path / "suite"
    suite.mkdir(parents=True)
    _write_json(
        suite / "benchmark_set.json",
        {"schema": "test-suite", "cases": [{"id": "b001"}]},
    )
    plan = {"runs": [{"id": "deepx", "provider": "deepx_m1"}]}
    _write_json(suite / "benchmark_plan.json", plan)
    (suite / "benchmark_suite.py").write_text(
        "import time\ntime.sleep(30)\n",
        encoding="utf-8",
    )
    options = SimpleNamespace(
        execution_mode="generate_and_run",
        skip_benchmarks=False,
        dry_run=False,
        no_remote=True,
        benchmark_execution_backend="local",
        benchmark_provider="",
        benchmark_preset="",
        benchmark_image="",
        benchmark_warmup=0,
        benchmark_runs=1,
        benchmark_timeout_s=0,
        benchmark_extra_args=[],
    )
    cancel = threading.Event()
    timer = threading.Timer(0.2, cancel.set)
    timer.start()
    started = time.monotonic()
    try:
        result = execute_benchmark_suite_if_requested(
            run_dir=run_root,
            model_id="model",
            options=options,
            benchmark_set_contract={"materialized": True, "suite_dir": str(suite)},
            benchmark_plan=plan,
            profile_payload={
                "quality_gate": {
                    "statistics": {"execution_location": "central_management"}
                }
            },
            model_entry={"task": "classification"},
            cancel_event=cancel,
        )
    finally:
        timer.cancel()
    assert result.status == "cancelled"
    assert result.metrics["cancelled"] is True
    assert time.monotonic() - started < 6.0
