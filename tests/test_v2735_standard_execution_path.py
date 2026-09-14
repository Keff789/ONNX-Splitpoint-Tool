from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from onnx_splitpoint_tool.workflow.contracts import WorkflowOptions
from onnx_splitpoint_tool.workflow.execution_binding import (
    _resolve_suite_dir,
    execute_benchmark_suite_if_requested,
)
from onnx_splitpoint_tool.workflow.profile_options import (
    workflow_options_from_profile_snapshot,
)
from onnx_splitpoint_tool.workflow.runner import (
    EvaluationWorkflowRunner,
    benchmark_set_postcondition_v60v,
)
from onnx_splitpoint_tool.workflow.start_snapshot import (
    build_profile_start_snapshot,
)


ROOT = Path(__file__).resolve().parents[1]
STANDARD_LAUNCHER = ROOT / "scripts" / "run_fresh_standard_workflow.sh"


def _standard_profile() -> dict[str, object]:
    return {
        "name": "standard_fixture",
        "workflow": {"execution_mode": "generate_and_run"},
        "benchmark_execution": {
            "provider": "auto",
            "warmup": 3,
            "runs": 5,
            "backend": "remote",
        },
        "hailo_build": {
            "mode": "reuse_and_build_missing",
            "preset": "balanced",
            "optimization_level": 1,
            "calib_count": 500,
        },
        "hardware": {
            "selected_setups": [],
            "resolved_targets": [
                {
                    "id": "orin_deepx",
                    "accelerator": "deepx_m1",
                    "remote": {"enabled": True, "host": "10.0.0.3"},
                },
                {
                    "id": "orin_hailo8",
                    "accelerator": "hailo8",
                    "remote": {"enabled": True, "host": "10.0.0.8"},
                },
                {
                    "id": "orin_hailo10h",
                    "accelerator": "hailo10h",
                    "remote": {"enabled": True, "host": "10.0.0.10"},
                },
            ],
        },
        "remote_execution": {
            "warmup": 3,
            "iters": 5,
            "repeats": 1,
        },
        "native_producers": {
            "enabled": True,
            "frames": 1000,
            "warmup": 100,
            "energy": {
                "enabled": True,
                "mode": "measure",
                "duration_s": 60,
            },
        },
        "energy": {"requested_native_energy": True},
        "model_suite": {"primary": []},
        "run_profiles": [],
    }


def _snapshot(profile: dict[str, object]) -> dict[str, object]:
    return build_profile_start_snapshot(
        profile_request="standard profile.yaml",
        source_profile=profile,
        resolved_profile=profile,
        profile_id="standard_fixture",
        profile_path="standard profile.yaml",
        profile_source="file",
        runtime_bindings={"runtime_materialized": True},
    )


def test_standard_snapshot_maps_the_gui_runtime_values() -> None:
    profile = _standard_profile()
    snapshot = _snapshot(profile)
    options = workflow_options_from_profile_snapshot(
        profile_request="standard profile.yaml",
        out_root="/tmp/Evaluation Runs",
        start_snapshot=snapshot,
        required_run_mode="standard",
        require_fresh_run=True,
    )

    assert options.execution_mode == "generate_and_run"
    assert options.benchmark_warmup == 3
    assert options.benchmark_runs == 5
    assert options.hailo_build_mode == "reuse_and_build_missing"
    assert options.hailo_preset == "balanced"
    assert options.hailo_optimization_level == 1
    assert options.hailo_calib_count == 500
    assert options.remote_warmup == 3
    assert options.remote_iters == 5
    assert options.remote_working_dir == "/tmp/Evaluation Runs/RemoteBenchmarkRuns"
    # Frozen hardware targets are authoritative even when selected_setups was
    # cleared after resolution. The old GUI-only mapper archived stale True.
    assert options.no_remote is False
    assert options.resume is False
    assert options.required_run_mode == "standard"
    assert options.require_fresh_run is True
    frozen_native = options.profile_start_snapshot["resolved_profile"][
        "native_producers"
    ]
    assert frozen_native["enabled"] is True
    assert frozen_native["energy"]["mode"] == "measure"
    assert frozen_native["energy"]["duration_s"] == 60


def test_profile_driven_cli_uses_the_shared_snapshot_mapping(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from onnx_splitpoint_tool.workflow import profile_options, run_evaluation

    profile = _standard_profile()
    snapshot = _snapshot(profile)
    captured: dict[str, WorkflowOptions] = {}

    monkeypatch.setattr(
        profile_options,
        "load_runtime_profile_snapshot",
        lambda _request: (profile, snapshot),
    )

    class _Runner:
        def __init__(self, options: WorkflowOptions, **_kwargs: object) -> None:
            captured["options"] = options

        def run(self) -> SimpleNamespace:
            return SimpleNamespace(
                ok=True,
                status="ok",
                run_dir="/tmp/run",
                manifest_path="/tmp/run/run_manifest.json",
                to_dict=lambda: {"ok": True, "status": "ok"},
            )

    monkeypatch.setattr(run_evaluation, "EvaluationWorkflowRunner", _Runner)
    rc = run_evaluation.main(
        [
            "--profile-driven",
            "--require-run-mode",
            "standard",
            "--require-fresh-run",
            "--profile",
            "standard profile.yaml",
            "--out",
            "/tmp/Evaluation Runs",
            "--execution-mode",
            "generate_and_run",
            "--only-model",
            "yolo26s",
            "--json",
        ]
    )

    assert rc == 0, capsys.readouterr().out
    options = captured["options"]
    assert options.only_model == "yolo26s"
    assert options.benchmark_warmup == 3
    assert options.benchmark_runs == 5
    assert options.hailo_preset == "balanced"
    assert options.hailo_calib_count == 500
    assert options.remote_warmup == 3
    assert options.remote_iters == 5
    assert options.profile_start_snapshot == snapshot


def test_profile_driven_cli_rejects_nonstandard_override_surface() -> None:
    from onnx_splitpoint_tool.workflow.run_evaluation import (
        _validate_profile_driven_args,
    )

    with pytest.raises(ValueError, match="unsupported options: --resume"):
        _validate_profile_driven_args(
            ["--profile-driven", "--profile", "p.yaml", "--resume"]
        )


def test_required_mode_and_fresh_guards_fail_before_run_creation() -> None:
    options = WorkflowOptions(
        profile="unused.yaml",
        out="/tmp/unused",
        execution_mode="generate_and_run",
        required_run_mode="standard",
        require_fresh_run=True,
    )
    runner = EvaluationWorkflowRunner(options)
    opened: list[bool] = []

    def _load_smoke() -> None:
        runner.profile_payload = {"execution_preset": {"id": "smoke"}}

    runner._load_profile = _load_smoke  # type: ignore[method-assign]
    runner._open_run_dir = lambda: opened.append(True)  # type: ignore[method-assign]
    with pytest.raises(ValueError, match="required run mode 'standard'"):
        runner.run()
    assert opened == []

    runner.profile_payload = _standard_profile()
    runner.options.resume = True
    with pytest.raises(ValueError, match="resume"):
        runner._validate_requested_start_contract()
    assert opened == []


def _write_realistic_suite(
    run_root: Path,
    model_id: str,
    case_id: str,
    *,
    include_case_runner: bool = True,
) -> tuple[Path, Path]:
    wrapper = run_root / "models" / model_id / "benchmark_set"
    suite = wrapper / "legacy_suite"
    case_dir = suite / case_id
    case_dir.mkdir(parents=True)
    (case_dir / "part1.onnx").write_bytes(b"fixture-part1")
    (case_dir / "part2.onnx").write_bytes(b"fixture-part2")
    if include_case_runner:
        (case_dir / "run_split_onnxruntime.py").write_text(
            "from pathlib import Path\n"
            "Path(__file__).with_name('case_runner_executed.txt').write_text('ok', encoding='utf-8')\n",
            encoding="utf-8",
        )
    (suite / "benchmark_set.json").write_text(
        json.dumps(
            {
                "schema": "fixture/benchmark-set",
                "cases": [{"case_id": case_id, "case_dir": case_id}],
            }
        ),
        encoding="utf-8",
    )
    plan = {"runs": [{"id": "ort_cpu", "case_id": case_id}]}
    (suite / "benchmark_plan.json").write_text(
        json.dumps(plan), encoding="utf-8"
    )
    (suite / "benchmark_suite.py").write_text(
        "import json\n"
        "import subprocess\n"
        "import sys\n"
        f"subprocess.run([sys.executable, '{case_id}/run_split_onnxruntime.py'], check=True)\n"
        "with open('benchmark_results.json', 'w', encoding='utf-8') as handle:\n"
        "    json.dump({'results': [{'run_id': 'ort_cpu', 'latency_ms': 1.0}]}, handle)\n",
        encoding="utf-8",
    )
    # Reproduce the real topology: the wrapper exists but is explicitly not a
    # runnable suite; only its nested legacy_suite is complete.
    (wrapper / "suite_generation_postcondition.json").write_text(
        json.dumps(
            {
                "valid": False,
                "selected_suite_dir": str(suite),
                "candidates": [
                    {"suite_dir": str(suite), "valid": True},
                    {"suite_dir": str(wrapper), "valid": False},
                ],
            }
        ),
        encoding="utf-8",
    )
    return wrapper, suite


@pytest.mark.parametrize(
    ("model_id", "case_id"),
    [("yolo26s", "b036"), ("yolov7_paper", "b044")],
)
def test_standard_product_binding_executes_the_nested_legacy_harness(
    tmp_path: Path,
    model_id: str,
    case_id: str,
) -> None:
    run_root = tmp_path / "EvaluationRuns" / "standard-run"
    wrapper, suite = _write_realistic_suite(run_root, model_id, case_id)
    postcondition = benchmark_set_postcondition_v60v(wrapper)
    assert postcondition["valid"] is True
    assert Path(postcondition["selected_suite_dir"]) == suite

    contract = {
        "materialized": True,
        "legacy_suite_dir": postcondition["selected_suite_dir"],
    }
    assert _resolve_suite_dir(
        run_root, run_root / "models" / model_id, contract
    ) == suite.resolve()
    options = SimpleNamespace(
        execution_mode="generate_and_run",
        skip_benchmarks=False,
        dry_run=False,
        no_remote=False,
        benchmark_execution_backend="local",
        benchmark_provider="",
        benchmark_preset="",
        benchmark_image="",
        benchmark_warmup=3,
        benchmark_runs=5,
        benchmark_timeout_s=10,
        benchmark_extra_args=[],
    )
    result = execute_benchmark_suite_if_requested(
        run_dir=run_root,
        model_id=model_id,
        options=options,
        benchmark_set_contract=contract,
        benchmark_plan={"runs": [{"id": "ort_cpu", "case_id": case_id}]},
        profile_payload={},
        model_entry={"task": "detection"},
    )

    assert result.status == "ok"
    assert result.metrics["returncode"] == 0
    assert result.command[1] == str(suite / "benchmark_suite.py")
    assert (suite / case_id / "case_runner_executed.txt").read_text(
        encoding="utf-8"
    ) == "ok"
    dispatch = json.loads(
        (
            run_root
            / "models"
            / model_id
            / "benchmark_results"
            / "benchmark_execution_dispatch.json"
        ).read_text(encoding="utf-8")
    )
    assert Path(dispatch["cwd"]) == suite
    assert Path(dispatch["cwd"]) != wrapper


def test_missing_real_case_runner_cannot_report_execution_pass(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "EvaluationRuns" / "standard-run"
    wrapper, suite = _write_realistic_suite(
        run_root, "yolo26s", "b036", include_case_runner=False
    )
    postcondition = benchmark_set_postcondition_v60v(wrapper)
    options = SimpleNamespace(
        execution_mode="generate_and_run",
        skip_benchmarks=False,
        dry_run=False,
        no_remote=False,
        benchmark_execution_backend="local",
        benchmark_provider="",
        benchmark_preset="",
        benchmark_image="",
        benchmark_warmup=3,
        benchmark_runs=5,
        benchmark_timeout_s=10,
        benchmark_extra_args=[],
    )
    result = execute_benchmark_suite_if_requested(
        run_dir=run_root,
        model_id="yolo26s",
        options=options,
        benchmark_set_contract={
            "materialized": True,
            "legacy_suite_dir": postcondition["selected_suite_dir"],
        },
        benchmark_plan={"runs": [{"id": "ort_cpu", "case_id": "b036"}]},
        profile_payload={},
        model_entry={"task": "detection"},
    )

    assert result.status == "partial"
    assert result.metrics["returncode"] != 0
    assert result.metrics["result_file_count"] == 0
    assert not (suite / "b036" / "case_runner_executed.txt").exists()


def test_standard_launcher_is_source_safe_pure_and_shell_quotes_paths(
    tmp_path: Path,
) -> None:
    profile = tmp_path / "ResNet & YOLO standard.yaml"
    profile.write_text("{}\n", encoding="utf-8")
    out_root = tmp_path / "Evaluation Runs"
    env = dict(os.environ)
    env["PY"] = sys.executable

    sourced = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; printf "CALLER_STILL_RUNNING\\n"',
            "source-test",
            str(STANDARD_LAUNCHER),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )
    assert sourced.returncode == 0, sourced.stderr
    assert sourced.stdout == "CALLER_STILL_RUNNING\n"

    planned = subprocess.run(
        [
            "bash",
            str(STANDARD_LAUNCHER),
            "--profile",
            str(profile),
            "--out",
            str(out_root),
            "--only-model",
            "yolo26s",
            "--plan",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )
    assert planned.returncode == 0, planned.stderr
    prefix = "Standard workflow command:"
    assert planned.stdout.startswith(prefix)
    tokens = shlex.split(planned.stdout[len(prefix) :].strip())
    assert tokens == [
        sys.executable,
        "-B",
        "-m",
        "onnx_splitpoint_tool.workflow.run_evaluation",
        "--profile-driven",
        "--require-run-mode",
        "standard",
        "--require-fresh-run",
        "--profile",
        str(profile),
        "--out",
        str(out_root),
        "--execution-mode",
        "generate_and_run",
        "--only-model",
        "yolo26s",
    ]

    source = STANDARD_LAUNCHER.read_text(encoding="utf-8")
    assert source.count("onnx_splitpoint_tool.workflow.run_evaluation") == 1
    assert 'exec "${command[@]}"' in source
    for forbidden in (
        "run_v2732",
        "v2732_semantic_quality_gate",
        "native_fifo_eval_runner",
        "run_native_producer_energy",
        "suite_bundle",
    ):
        assert forbidden not in source


def test_failed_parallel_acceptance_system_is_not_shipped() -> None:
    obsolete = [
        "scripts/run_v2732_full_hardware_gates.py",
        "scripts/run_v2732_hardware_acceptance.sh",
        "scripts/v2732_hardware_acceptance_verify.py",
        "scripts/v2732_semantic_quality_gate.py",
        "tests/test_v2732_fresh_selected_energy.py",
        "tests/test_v2732_full_hardware_coordinator.py",
        "tests/test_v2732_hardware_acceptance_harness.py",
        "tests/test_v2732_hardware_acceptance_verify.py",
        "tests/test_v2732_semantic_quality_gate.py",
    ]
    assert [name for name in obsolete if (ROOT / name).exists()] == []

    for relative in (
        "onnx_splitpoint_tool/native_command_contract.py",
        "onnx_splitpoint_tool/resume_artifact_rehydration.py",
        "onnx_splitpoint_tool/resume_cohort_preflight.py",
        "onnx_splitpoint_tool/resume_preparation.py",
        "scripts/native_producer_energy_plan.py",
        "scripts/native_split_energy_preflight.py",
        "scripts/run_native_producer_energy_from_summary.py",
    ):
        assert "fresh_selected" not in (ROOT / relative).read_text(
            encoding="utf-8"
        )
