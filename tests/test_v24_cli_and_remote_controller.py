from __future__ import annotations

import json
from pathlib import Path

from onnx_splitpoint_tool.benchmark.remote_run import RemoteBenchmarkArgs
from onnx_splitpoint_tool.benchmark.services import RemoteBenchmarkCallbacks, RemoteBenchmarkController
from onnx_splitpoint_tool.cli import main


class _DummyRemoteService:
    def __init__(self, out_dir: Path):
        self.out_dir = Path(out_dir)

    def run(self, **_kwargs):
        results_dir = self.out_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        (results_dir / "benchmark_suite_status_matrix.md").write_text("status matrix line\n", encoding="utf-8")
        return {
            "status": "ok",
            "local_run_dir": str(self.out_dir),
        }


def test_remote_benchmark_controller_emits_finish_and_matrix(tmp_path: Path):
    logs: list[str] = []
    finishes: list[str] = []
    results: list[tuple[str, dict]] = []
    controller = RemoteBenchmarkController(service=_DummyRemoteService(tmp_path / "run"))
    callbacks = RemoteBenchmarkCallbacks(
        log=logs.append,
        progress=lambda _p, _lbl: None,
        finish=finishes.append,
        result=lambda kind, out: results.append((kind, out)),
    )
    out = controller.run(
        host=type("Host", (), {"id": "dummy"})(),
        benchmark_set_json=tmp_path / "benchmark_set.json",
        local_working_dir=tmp_path,
        run_id="run1",
        args=RemoteBenchmarkArgs(),
        cancel_event=None,
        callbacks=callbacks,
    )
    assert out["status"] == "ok"
    assert finishes == ["Done"]
    assert results and results[0][0] == "ok"
    assert any("status matrix line" in line for line in logs)


def test_cli_benchmark_schema_roundtrip(tmp_path: Path, capsys):
    payload = {
        "schema": "onnx-splitpoint/benchmark-set",
        "schema_version": 1,
        "tool": {"gui": "test-gui"},
        "cases": [],
    }
    path = tmp_path / "benchmark_set.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    rc = main(["benchmark-schema", str(path)])
    captured = capsys.readouterr().out
    assert rc == 0
    assert "schema_version=2" in captured
    assert "changed=" in captured
