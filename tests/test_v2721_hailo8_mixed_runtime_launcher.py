from __future__ import annotations

import importlib.util
import hashlib
import json
import os
import re
import sys
from pathlib import Path
import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(
        f"v2721_{path.stem}_{id(path)}", path
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _step(
    *,
    rc: int = 0,
    stdout: str = "",
    stderr: str = "",
    timed_out: bool = False,
) -> dict:
    return {
        "rc": rc,
        "elapsed_s": 0.01,
        "stdout_tail": stdout,
        "stderr_tail": stderr,
        "timed_out": timed_out,
    }


def _quality_mirrors(prefix: str = "a") -> dict:
    return {
        "eval_run_id": f"{prefix}-eval",
        "source_run_id": f"{prefix}-source",
        "native_split_quality_binding_sha256": prefix * 64,
        "native_split_quality_eval_run_id": f"{prefix}-eval",
        "native_split_quality_source_run_id": f"{prefix}-source",
        "source_request_sha256": prefix * 64,
        "native_split_quality_source_request_sha256": prefix * 64,
        "native_split_quality_central_result_sha256": prefix * 64,
        "native_split_quality_selection_sha256": prefix * 64,
        "native_split_quality_binding": {"binding": prefix},
        "native_split_quality_consumer_attestation": {"attested": True},
    }


def test_mixed_runtime_preflight_uses_existing_python_and_full_source_closure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    matrix = _load_script("native_fifo_smoke_matrix.py")
    python_executable = tmp_path / "system-python3"
    python_executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    python_executable.chmod(0o755)
    hailo_site = tmp_path / "hailo-site"
    (hailo_site / "hailo_platform").mkdir(parents=True)

    assert matrix._mixed_runtime_python(str(python_executable)) == str(
        python_executable.resolve()
    )

    captured: dict = {}
    contract = {
        "status": "ready",
        "runtime_mode": "system_tensorrt_with_process_local_hailo_sites",
        "python_executable": str(python_executable),
        "resolved_python_executable": str(python_executable.resolve()),
        "extra_sites": [str(hailo_site.resolve())],
        "modules": {
            "tensorrt": "/usr/lib/tensorrt/__init__.py",
            "hailo_platform": str(hailo_site / "hailo_platform/__init__.py"),
            "numpy": "/usr/lib/numpy/__init__.py",
            "PIL": "/usr/lib/PIL/__init__.py",
        },
        "module_origins_ok": True,
        "cudart": "libcudart.so",
        "source_closure_ok": True,
    }

    def fake_run(cmd, timeout=None, env=None):
        captured.update({"cmd": cmd, "timeout": timeout, "env": env})
        return _step(stdout=json.dumps(contract))

    monkeypatch.setattr(matrix, "_run", fake_run)
    result = matrix._mixed_runtime_preflight(
        str(python_executable.resolve()),
        [str(hailo_site.resolve())],
        timeout=17.0,
    )

    assert result["ok"] is True
    assert result["contract"] == contract
    assert captured["cmd"][:3] == [
        str(python_executable.resolve()),
        "-B",
        "-c",
    ]
    assert captured["timeout"] == 17.0
    assert captured["env"]["PYTHONDONTWRITEBYTECODE"] == "1"
    assert captured["env"]["SPLITPOINT_EXTRA_SITES"] == str(
        hailo_site.resolve()
    )
    probe_source = captured["cmd"][3]
    for token in (
        "site.addsitedir(path)",
        'import_module(name)',
        '"tensorrt"',
        '"hailo_platform"',
        'find_library("cudart")',
        "from native_hailo10_trt_e2e_from_benchmarkset import NativeTRT",
        "from onnx_splitpoint_tool.runners.backends.hailo_backend import HailoBackend",
        "hailo_platform_origin_not_in_injected_sites",
        "tensorrt_origin_shadowed_by_hailo_sites",
    ):
        assert token in probe_source


def test_fresh_system_launcher_discovers_hailo_sites_from_separate_interpreter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    matrix = _load_script("native_fifo_smoke_matrix.py")
    hailo_site = tmp_path / "hailo-venv/lib/python3.10/site-packages"
    (hailo_site / "hailo_platform").mkdir(parents=True)

    monkeypatch.setattr(
        matrix, "_hailo_site_dirs_from_current_interpreter", lambda: [],
    )
    monkeypatch.setattr(matrix, "_python_major_minor", lambda _py: (3, 10))
    monkeypatch.setattr(
        matrix, "_candidate_hailo_pythons", lambda: ["/home/nx/hailo_py/bin/python"],
    )
    calls: list[tuple[str, tuple[int, int] | None]] = []

    def from_interpreter(python_executable: str, *, expected_abi):
        calls.append((python_executable, expected_abi))
        return [str(hailo_site.resolve())]

    monkeypatch.setattr(
        matrix, "_hailo_site_dirs_from_interpreter", from_interpreter,
    )
    assert matrix._hailo_site_dirs_for_mixed_runtime("/usr/bin/python3") == [
        str(hailo_site.resolve())
    ]
    assert calls == [("/home/nx/hailo_py/bin/python", (3, 10))]

    child_source = (
        ROOT / "scripts/native_hailo_trt_fifo_from_benchmarkset.py"
    ).read_text(encoding="utf-8")
    activation = child_source.index(
        "_PROCESS_LOCAL_EXTRA_SITES = _activate_process_local_extra_sites()"
    )
    assert activation < child_source.index("import numpy as np")
    assert activation < child_source.index(
        "import native_hailo10_trt_e2e_from_benchmarkset"
    )


def test_inner_wrapper_preserves_child_error_and_ignores_stale_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    matrix = _load_script("native_fifo_smoke_matrix.py")
    benchmark_set = tmp_path / "benchmark_set"
    (benchmark_set / "b038").mkdir(parents=True)
    (benchmark_set / "benchmark_set.json").write_text(
        json.dumps(
            {
                "task": "detection",
                "cases": [{"case_id": "b038"}],
            }
        ),
        encoding="utf-8",
    )
    image = benchmark_set / "resources/test_image_coco.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"image")
    stale_result = (
        benchmark_set
        / "native_pipeline/b038/hailo_to_trt/"
        "uint8_cast_fp16/native_fifo_results.json"
    )
    stale_result.parent.mkdir(parents=True)
    stale_result.write_text(
        json.dumps(
            {
                "ok": True,
                "fps_makespan": 999.0,
                **_quality_mirrors("f"),
            }
        ),
        encoding="utf-8",
    )

    child_calls: list[tuple[list[str], dict | None]] = []

    def fake_run(command, timeout=None, env=None):
        joined = " ".join(str(item) for item in command)
        if "native_fifo_capability_report.py" in joined:
            report = (
                benchmark_set
                / "analysis_tables/native_fifo_capability_report.json"
            )
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_text(
                json.dumps(
                    {
                        "cases": [
                            {
                                "case_id": "b038",
                                "native_fifo_supported": True,
                                "native_fifo_ok": False,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            return _step()
        if "native_hailo_trt_fifo_from_benchmarkset.py" in joined:
            child_calls.append((list(command), env))
            return _step(
                rc=1,
                stderr=(
                    "Traceback (most recent call last):\n"
                    "ModuleNotFoundError: No module named 'tensorrt'\n"
                ),
            )
        raise AssertionError(f"unexpected command: {command}")

    preflight = {
        "name": "hailo8_detection_mixed_runtime_preflight",
        **_step(),
        "ok": True,
        "contract": {
            "status": "ready",
            "source_closure_ok": True,
            "resolved_python_executable": "/usr/bin/python3",
        },
        "failure_reason": "",
    }
    monkeypatch.setattr(matrix, "_run", fake_run)
    monkeypatch.setattr(
        matrix, "_mixed_runtime_python", lambda requested: "/usr/bin/python3"
    )
    monkeypatch.setattr(
        matrix,
        "_hailo_site_dirs_from_current_interpreter",
        lambda: ["/home/nx/hailo_py/lib/python3.10/site-packages"],
    )
    monkeypatch.setattr(
        matrix, "_mixed_runtime_preflight", lambda *a, **k: preflight
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "native_fifo_smoke_matrix.py",
            "--benchmark-set",
            str(benchmark_set),
            "--model-id",
            "yolo26s",
            "--case",
            "b038",
            "--task",
            "detection",
            "--image",
            str(image),
            "--no-validate-dumps",
        ],
    )

    assert matrix.main() == 0
    payload = json.loads(
        (
            benchmark_set / "analysis_tables/native_fifo_smoke_matrix.json"
        ).read_text(encoding="utf-8")
    )
    row = payload["cases"][0]
    assert row["status"] == "failed"
    assert row["result_ok"] is False
    assert row["child_result_fresh"] is False
    assert row["returncode"] == 1
    assert row["failure_reason"] == "native_runner_nonzero_exit"
    assert "ModuleNotFoundError" in row["error"]
    assert "tensorrt" in row["stderr_tail"]
    assert "mirror_mismatch" not in row["failure_reason"]
    assert row["fps_makespan"] is None

    assert len(child_calls) == 1
    child_command, child_env = child_calls[0]
    assert child_command[0] == "/usr/bin/python3"
    assert child_env is not None
    assert (
        child_env["SPLITPOINT_EXTRA_SITES"]
        == "/home/nx/hailo_py/lib/python3.10/site-packages"
    )


def test_outer_wrapper_preserves_inner_child_failure_over_stale_mirrors(
    tmp_path: Path,
) -> None:
    outer = _load_script("native_fifo_eval_runner.py")
    assert not hasattr(outer, "args")
    result_path = tmp_path / "stale-native-result.json"
    result_path.write_text(
        json.dumps({"ok": True, **_quality_mirrors("f")}),
        encoding="utf-8",
    )
    matrix = {
        "cases": [
            {
                "case_id": "b038",
                "status": "failed",
                "result_ok": False,
                "returncode": 1,
                "timed_out": False,
                "child_result_fresh": False,
                "native_fifo_result": str(result_path),
                "native_fifo_result_sha256": hashlib.sha256(
                    result_path.read_bytes()
                ).hexdigest(),
                "native_fifo_result_size_bytes": result_path.stat().st_size,
                "native_split_quality_required": True,
                "performance_claims_emitted": False,
                "failure_reason": "native_runner_nonzero_exit",
                "status_detail": "native_runner_nonzero_exit",
                "error": "ModuleNotFoundError: No module named 'tensorrt'",
                "stderr_tail": (
                    "ModuleNotFoundError: No module named 'tensorrt'"
                ),
                **_quality_mirrors("a"),
            }
        ]
    }

    rows = outer._extract_rows(
        "yolo26s",
        tmp_path,
        matrix,
        setup_id="orin_nx_hailo8_01",
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["result_ok"] is False
    assert row["status"] == "failed"
    assert row["failure_reason"] == "native_runner_nonzero_exit"
    assert "ModuleNotFoundError" in row["error"]
    assert "mirror_mismatch" not in row["failure_reason"]
    assert row["setup_id"] == "orin_nx_hailo8_01"
    assert row["comparison_backend"] == "hailo8"


def test_outer_wrapper_successful_child_mirror_drift_remains_fail_closed(
    tmp_path: Path,
) -> None:
    outer = _load_script("native_fifo_eval_runner.py")
    assert not hasattr(outer, "args")
    child = _quality_mirrors("a")
    wrapper = dict(child)
    wrapper["eval_run_id"] = "tampered-eval"
    result_path = tmp_path / "fresh-native-result.json"
    result_path.write_text(
        json.dumps({"ok": True, "fps_makespan": 12.5, **child}),
        encoding="utf-8",
    )
    matrix = {
        "cases": [
            {
                "case_id": "b044",
                "status": "ok",
                "result_ok": True,
                "returncode": 0,
                "timed_out": False,
                "child_result_fresh": True,
                "native_fifo_result": str(result_path),
                "native_split_quality_required": True,
                "performance_claims_emitted": True,
                "native_fifo_result_sha256": hashlib.sha256(
                    result_path.read_bytes()
                ).hexdigest(),
                "native_fifo_result_size_bytes": result_path.stat().st_size,
                **wrapper,
            }
        ]
    }

    rows = outer._extract_rows(
        "yolov7_paper",
        tmp_path,
        matrix,
        setup_id="orin_nx_hailo8_01",
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["result_ok"] is False
    assert row["status"] == "failed"
    assert row["performance_claims_emitted"] is False
    assert row["failure_reason"] == (
        "native_split_quality_wrapper_mirror_mismatch:eval_run_id"
    )


def test_outer_wrapper_setup_id_precedence_result_matrix_cli(
    tmp_path: Path,
) -> None:
    outer = _load_script("native_fifo_eval_runner.py")
    assert not hasattr(outer, "args")

    result_path = tmp_path / "native-result.json"
    result_path.write_text(
        json.dumps({"ok": True, "setup_id": "result-setup"}),
        encoding="utf-8",
    )
    matrix = {
        "cases": [
            {
                "case_id": "b001",
                "status": "ok",
                "result_ok": True,
                "native_fifo_result": str(result_path),
                "setup_id": "matrix-setup",
            },
            {
                "case_id": "b002",
                "status": "failed",
                "result_ok": False,
                "setup_id": "matrix-setup",
            },
            {
                "case_id": "b003",
                "status": "failed",
                "result_ok": False,
            },
        ]
    }

    rows = outer._extract_rows(
        "resnet50",
        tmp_path,
        matrix,
        setup_id="cli-setup",
    )

    assert [row["setup_id"] for row in rows] == [
        "result-setup",
        "matrix-setup",
        "cli-setup",
    ]
    assert {row["comparison_backend"] for row in rows} == {"hailo8"}


def test_outer_main_passes_cli_setup_id_to_early_failure_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outer = _load_script("native_fifo_eval_runner.py")
    assert not hasattr(outer, "args")

    root = tmp_path / "native-root"
    benchmark_set = root / "yolo26s" / "benchmark_set"
    analysis = benchmark_set / "analysis_tables"
    analysis.mkdir(parents=True)
    (benchmark_set / "benchmark_set.json").write_text(
        json.dumps({"task": "detection"}),
        encoding="utf-8",
    )
    (benchmark_set / "benchmark_suite.py").write_text("", encoding="utf-8")
    (analysis / "native_fifo_smoke_matrix.json").write_text(
        json.dumps(
            {
                "ok": True,
                "cases": [
                    {
                        "case_id": "b026",
                        "status": "failed",
                        "result_ok": False,
                        "returncode": 1,
                        "timed_out": False,
                        "child_result_fresh": False,
                        "failure_reason": (
                            "hailo8_detection_mixed_runtime_preflight_failed"
                        ),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(outer, "_run", lambda *args, **kwargs: _step())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "native_fifo_eval_runner.py",
            "--root",
            str(root),
            "--models",
            "yolo26s",
            "--setup-id",
            "orin_nx_hailo8_01",
        ],
    )

    # The row is preserved for diagnosis, including the CLI setup identity,
    # but a requested run containing only failed rows is fail-closed in v2.75.
    assert outer.main() == 3
    summary = json.loads(
        (root / "analysis_tables/native_fifo_eval_runner.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["ok"] is False
    assert summary["orchestration_status"] == "failed"
    assert summary["row_count"] == 1
    assert summary["rows"][0]["setup_id"] == "orin_nx_hailo8_01"
    assert summary["rows"][0]["comparison_backend"] == "hailo8"
    assert summary["rows"][0]["failure_reason"] == (
        "hailo8_detection_mixed_runtime_preflight_failed"
    )


def test_hailo8_remote_sync_stages_native_trt_source_sibling() -> None:
    def hailo8_dependency_blocks(source: str) -> list[str]:
        return [
            match.group("body")
            for match in re.finditer(
                r"if b == [\"']hailo8[\"']:(?P<body>.*?)"
                r"(?:elif b == [\"']hailo10h[\"']|else:)",
                source,
                re.DOTALL,
            )
            if "native_fifo_smoke_matrix.py" in match.group("body")
        ]

    for relative in (
        "onnx_splitpoint_tool/workflow/runner.py",
        "scripts/update_evalset_native_producers.py",
    ):
        source = (ROOT / relative).read_text(encoding="utf-8")
        blocks = hailo8_dependency_blocks(source)
        assert blocks, f"no Hailo-8 dependency block found in {relative}"
        assert all(
            "native_hailo_trt_fifo_from_benchmarkset.py" in block
            and "native_hailo10_trt_e2e_from_benchmarkset.py" in block
            and "class NativeTRT" in block
            for block in blocks
        ), f"incomplete Hailo-8 mixed runtime source closure in {relative}"

    child = (
        ROOT / "scripts/native_hailo_trt_fifo_from_benchmarkset.py"
    ).read_text(encoding="utf-8")
    assert "native_hailo10_trt_e2e_from_benchmarkset" in child
    sibling = ROOT / "scripts/native_hailo10_trt_e2e_from_benchmarkset.py"
    remote_sibling = (
        ROOT
        / "onnx_splitpoint_tool/resources/remote_scripts/"
        "native_hailo10_trt_e2e_from_benchmarkset.py"
    )
    assert sibling.is_file()
    assert remote_sibling.is_file()
    assert sibling.read_bytes() == remote_sibling.read_bytes()
