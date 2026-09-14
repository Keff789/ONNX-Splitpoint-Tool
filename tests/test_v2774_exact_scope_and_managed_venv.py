from __future__ import annotations

import os
import shutil
from pathlib import Path
import sys
from types import SimpleNamespace

import onnx
from onnx import TensorProto, helper
import pytest

from onnx_splitpoint_tool.benchmark.services import (
    BenchmarkGenerationExecutionCallbacks,
    BenchmarkGenerationExecutionConfig,
    BenchmarkGenerationExecutionService,
    BenchmarkGenerationOrchestrationService,
    BenchmarkGenerationRuntime,
)
from onnx_splitpoint_tool import hailo_backend
from onnx_splitpoint_tool.hailo_backend import _managed_venv_child_env
from onnx_splitpoint_tool.workflow.legacy_benchmarkset_binding import (
    _resolve_generation_candidate_scope,
)


FORCED_CASES = ("b066", "b088", "b104", "b199")


def _candidate(case_id: str) -> dict[str, object]:
    boundary = int(case_id[1:])
    return {
        "case_id": case_id,
        "boundary": boundary,
        "split_index": boundary,
    }


def test_forced_cases_are_the_exact_generator_scope() -> None:
    selected = [_candidate(case_id) for case_id in FORCED_CASES]
    prediction = {
        "candidates": [
            *selected,
            *[_candidate(f"b{boundary:03d}") for boundary in range(1, 383)
              if boundary not in {66, 88, 104, 199}],
        ]
    }
    plan = {
        "model_id": "yolo26s",
        "requested_cases": 4,
        "selection_strategy": "stratified_windows",
        "selected_candidates": selected,
    }

    ranked, pool, requested, exact = _resolve_generation_candidate_scope(
        plan,
        prediction,
        expected_selection_policy={
            "forced_cases": {"yolo26s": list(FORCED_CASES)},
        },
    )

    assert exact is True
    assert requested == 4
    assert ranked == [66, 88, 104, 199]
    assert pool == [66, 88, 104, 199]


def test_forced_scope_accepts_only_attested_capability_replacement() -> None:
    projected = [
        _candidate(case_id)
        for case_id in ("b066", "b088", "b105", "b199")
    ]
    prediction = {
        "candidates": [
            *projected,
            _candidate("b104"),
            _candidate("b199"),
            _candidate("b362"),
        ]
    }
    plan = {
        "model_id": "yolo26s",
        "requested_cases": 4,
        "selected_candidates": projected,
        "native_capability_excluded_candidates": [
            {
                **_candidate("b104"),
                "exclude_source": "native_split_capability",
                "exclude_reason": "part2_input_count_not_one",
                "native_capability_contract": (
                    "exactly_one_part2_external_input"
                ),
                "observed_part2_input_count": 2,
            }
        ],
        "native_capability_backfills": [
            {
                **_candidate("b105"),
                "native_backfill_replaces_case": "b104",
                "origin": "native_capability_backfill",
                "native_backfill_scope": (
                    "same_stratified_window_or_global_rank_fallback"
                ),
            }
        ],
    }

    ranked, pool, requested, exact = _resolve_generation_candidate_scope(
        plan,
        prediction,
        expected_selection_policy={
            "forced_cases": {"yolo26s": list(FORCED_CASES)},
        },
    )

    assert exact is True
    assert requested == 4
    assert ranked == pool == [66, 88, 105, 199]


def test_forced_scope_rejects_unattested_selected_case_drift() -> None:
    with pytest.raises(ValueError, match="do not match the declared forced"):
        _resolve_generation_candidate_scope(
            {
                "model_id": "yolo26s",
                "requested_cases": 4,
                "selected_candidates": [
                    _candidate(case_id)
                    for case_id in ("b066", "b088", "b105", "b199")
                ],
            },
            {"candidates": [_candidate("b105")]},
            expected_selection_policy={
                "forced_cases": {"yolo26s": list(FORCED_CASES)},
            },
        )


def test_forced_scope_preserves_attested_capability_shortfall() -> None:
    ranked, pool, requested, exact = _resolve_generation_candidate_scope(
        {
            "model_id": "yolo26s",
            "requested_cases": 3,
            "selected_candidates": [
                _candidate("b066"),
                _candidate("b088"),
            ],
            "native_capability_excluded_candidates": [
                {
                    **_candidate("b104"),
                    "exclude_source": "native_split_capability",
                    "exclude_reason": "part2_input_count_not_one",
                    "native_capability_contract": (
                        "exactly_one_part2_external_input"
                    ),
                    "observed_part2_input_count": 2,
                }
            ],
        },
        {"candidates": [_candidate("b199")]},
        expected_selection_policy={
            "forced_cases": {
                "yolo26s": ["b066", "b088", "b104"],
            },
        },
    )

    assert exact is True
    assert requested == 3
    assert ranked == pool == [66, 88]


@pytest.mark.parametrize(
    "excluded_patch",
    [
        {"exclude_source": "manual"},
        {"exclude_reason": "some_other_reason"},
        {"native_capability_contract": "unattested"},
        {"observed_part2_input_count": 1},
    ],
)
def test_forced_scope_rejects_unattested_capability_exclusion(
    excluded_patch: dict[str, object],
) -> None:
    excluded = {
        **_candidate("b104"),
        "exclude_source": "native_split_capability",
        "exclude_reason": "part2_input_count_not_one",
        "native_capability_contract": "exactly_one_part2_external_input",
        "observed_part2_input_count": 2,
        **excluded_patch,
    }
    with pytest.raises(ValueError, match="unattested native capability"):
        _resolve_generation_candidate_scope(
            {
                "model_id": "yolo26s",
                "requested_cases": 3,
                "selected_candidates": [
                    _candidate("b066"),
                    _candidate("b088"),
                ],
                "native_capability_excluded_candidates": [excluded],
            },
            {"candidates": []},
            expected_selection_policy={
                "forced_cases": {
                    "yolo26s": ["b066", "b088", "b104"],
                },
            },
        )


def test_forced_scope_rejects_declared_case_as_capability_replacement() -> None:
    with pytest.raises(ValueError, match="not bound to one excluded forced"):
        _resolve_generation_candidate_scope(
            {
                "model_id": "yolo26s",
                "requested_cases": 3,
                "selected_candidates": [
                    _candidate("b066"),
                    _candidate("b088"),
                ],
                "native_capability_excluded_candidates": [
                    {
                        **_candidate("b104"),
                        "exclude_source": "native_split_capability",
                        "exclude_reason": "part2_input_count_not_one",
                        "native_capability_contract": (
                            "exactly_one_part2_external_input"
                        ),
                        "observed_part2_input_count": 2,
                    }
                ],
                "native_capability_backfills": [
                    {
                        **_candidate("b088"),
                        "native_backfill_replaces_case": "b104",
                        "origin": "native_capability_backfill",
                        "native_backfill_scope": (
                            "same_stratified_window_or_global_rank_fallback"
                        ),
                    }
                ],
            },
            {"candidates": []},
            expected_selection_policy={
                "forced_cases": {
                    "yolo26s": ["b066", "b088", "b104"],
                },
            },
        )


@pytest.mark.parametrize(
    "selected",
    [
        [],
        [_candidate("b066"), _candidate("b066")],
        [{"case_id": "candidate_x", "boundary": 66}],
    ],
)
def test_forced_scope_is_fail_closed_for_invalid_resolved_plan(
    selected: list[dict[str, object]],
) -> None:
    with pytest.raises(ValueError, match="Forced candidate scope"):
        _resolve_generation_candidate_scope(
            {
                "model_id": "yolo26s",
                "requested_cases": 4,
                "selected_candidates": selected,
            },
            {"candidates": [_candidate("b066")]},
            expected_selection_policy={
                "forced_cases": {"yolo26s": list(FORCED_CASES)},
            },
        )


def _yolo26_split_only_cfg() -> SimpleNamespace:
    execution_cfg = SimpleNamespace(
        base="yolo26s",
        full_model_src="/models/yolo26s.onnx",
        full_model_dst="/models/yolo26s.onnx",
        hef_targets=["hailo10"],
    )
    return SimpleNamespace(
        base="yolo26s",
        full_model_src="/models/yolo26s.onnx",
        full_model_dst="/models/yolo26s.onnx",
        out_dir=Path("/runs/models/yolo26s/benchmark_set"),
        bench_log_path="/runs/models/yolo26s/benchmark_set/generation.log",
        analysis_payload={},
        analysis_params_payload={},
        execution_cfg=execution_cfg,
        bench_plan_runs=[{
            "id": "hailo10_to_trt",
            "type": "mixed_backend",
            "variants": ["part1", "composed"],
            "stage1": {"type": "hailo", "hw_arch": "hailo10"},
            "stage2": {"type": "tensorrt"},
        }],
        hef_targets=["hailo10"],
        hailo_selected=True,
        hef_full=False,
        hef_part1=True,
        hef_part2=False,
        hailo_cache_only=False,
        full_hef_policy="skip",
        hailo_build_hef_fn=lambda *_args, **_kwargs: None,
    )


def test_yolo26_full_skip_is_authoritative_in_all_override_paths() -> None:
    service = BenchmarkGenerationOrchestrationService()
    cfg = _yolo26_split_only_cfg()
    logs: list[str] = []

    assert service._force_yolo26_suite_full_baseline_if_needed(
        cfg,
        log=logs.append,
    ) is cfg
    assert service._ensure_yolo26_full_hailo_baseline_plan(
        cfg,
        log=logs.append,
    ) is cfg
    assert service._yolo26_should_build_full_first(cfg) is False
    assert cfg.hef_full is False
    assert logs == []


def test_prepared_full_baseline_is_not_materialized_under_skip(
    tmp_path: Path,
) -> None:
    prepared_hef = tmp_path / "prepared" / "compiled.hef"
    prepared_hef.parent.mkdir()
    prepared_hef.write_bytes(b"prepared-full-hef")
    suite_out = tmp_path / "suite"
    cfg = SimpleNamespace(
        full_hef_policy="skip",
        hailo_cache_only=False,
        hef_full=True,
        prepared_full_hailo_baseline={
            "ok": True,
            "hef_path": str(prepared_hef),
            "hw_arch": "hailo10h",
        },
        out_dir=suite_out,
        hef_targets=["hailo10"],
    )
    suite_hefs: dict[str, dict[str, object]] = {}

    used = BenchmarkGenerationOrchestrationService()._materialize_prepared_full_hailo_baseline(
        cfg,
        log=lambda *_args, **_kwargs: None,
        suite_hailo_hefs=suite_hefs,
    )

    assert used is False
    assert suite_hefs == {}
    assert not suite_out.exists()


def _passthrough_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2])
    skip = helper.make_tensor_value_info("skip", TensorProto.FLOAT, [1, 2])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])
    graph = helper.make_graph(
        [
            helper.make_node("Relu", ["x"], ["cut"], name="left"),
            helper.make_node("Add", ["cut", "skip"], ["joined"], name="join"),
            helper.make_node("Relu", ["joined"], ["y"], name="right"),
        ],
        "forced-no-backfill",
        [x, skip],
        [y],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])


def test_rejected_forced_runtime_scope_cannot_backfill(
    tmp_path: Path,
) -> None:
    model = _passthrough_model()
    out_dir = tmp_path / "suite"
    out_dir.mkdir()
    runtime = BenchmarkGenerationRuntime(
        out_dir=out_dir,
        bench_log_path=out_dir / "benchmark.log",
        state_path=out_dir / "generation_state.json",
        requested_cases=1,
        ranked_candidates=[0],
        candidate_search_pool=[0],
        model_name="forced",
        model_source="forced.onnx",
        hef_full_policy="skip",
    )
    cfg = BenchmarkGenerationExecutionConfig(
        runtime=runtime,
        target_cases=1,
        gap=0,
        ranked_candidates=[0],
        candidate_search_pool=[0],
        out_dir=out_dir,
        base="forced",
        pad=3,
        strict_boundary=False,
        model=model,
        nodes=list(model.graph.node),
        order=[0, 1, 2],
        analysis_payload={},
        require_single_part2_input=True,
    )
    callbacks = BenchmarkGenerationExecutionCallbacks(
        log=lambda *_args, **_kwargs: None,
        queue_put=lambda *_args, **_kwargs: None,
        persist_state=lambda **_kwargs: None,
        publish_hailo_diagnostics=lambda *_args, **_kwargs: None,
        predicted_metrics_for_boundary=lambda *_args, **_kwargs: {},
        hailo_parse_entry_for_boundary=lambda *_args, **_kwargs: None,
        hailo_parse_scalar_fields=lambda *_args, **_kwargs: {},
    )

    chosen = BenchmarkGenerationExecutionService().execute_case_build_loop(
        cfg, callbacks
    )

    assert chosen == []
    assert [row["boundary"] for row in runtime.discarded_cases] == [0]
    assert not (out_dir / "b001").exists()


def test_managed_venv_child_env_projects_activation_semantics(
    tmp_path: Path,
) -> None:
    venv = tmp_path / "managed_hailo10"
    python = venv / "bin" / "python"
    python.parent.mkdir(parents=True)
    onnxsim = python.parent / "onnxsim"
    onnxsim.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    onnxsim.chmod(0o755)
    inherited = os.pathsep.join(["/usr/local/bin", "/usr/bin"])

    env = _managed_venv_child_env(
        python,
        {
            "PATH": inherited,
            "PYTHONHOME": "/wrong/python/home",
            "KEEP_ME": "yes",
        },
    )

    assert env["VIRTUAL_ENV"] == str(venv.resolve())
    assert env["PATH"].split(os.pathsep) == [
        str((venv / "bin").resolve()),
        "/usr/local/bin",
        "/usr/bin",
    ]
    assert "PYTHONHOME" not in env
    assert env["KEEP_ME"] == "yes"
    assert shutil.which("onnxsim", path=env["PATH"]) == str(onnxsim.resolve())


@pytest.mark.skipif(os.name != "posix", reason="POSIX venv symlink regression")
def test_managed_venv_child_env_preserves_posix_python_symlink_location(
    tmp_path: Path,
) -> None:
    venv = tmp_path / "managed_hailo10"
    python = venv / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.symlink_to(Path(sys.executable).resolve())
    assert python.is_symlink()
    assert python.resolve().parent != python.parent
    onnxsim = python.parent / "onnxsim"
    onnxsim.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    onnxsim.chmod(0o755)

    env = _managed_venv_child_env(
        python,
        {"PATH": os.pathsep.join(["/usr/local/bin", "/usr/bin"])},
    )

    assert env["VIRTUAL_ENV"] == str(venv)
    assert env["PATH"].split(os.pathsep)[0] == str(venv / "bin")
    assert shutil.which("onnxsim", path=env["PATH"]) == str(onnxsim)


def test_managed_venv_build_dispatch_receives_projected_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    managed = tmp_path / "managed_hailo10"
    python = managed / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("", encoding="utf-8")
    onnxsim = python.parent / "onnxsim"
    onnxsim.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    onnxsim.chmod(0o755)
    model = tmp_path / "part1.onnx"
    model.write_bytes(b"test-onnx")
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        hailo_backend,
        "_resolve_managed_venv_python",
        lambda **_kwargs: ("managed-hailo10", python, managed / "bin/activate"),
    )

    def _fake_run(_cmd, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            returncode=1,
            stdout=(
                '__SPLITPOINT_HAILO_RESULT__{"ok":false,'
                '"error":"synthetic stop"}\n'
            ),
            stderr="",
            timed_out=False,
            timeout_kind=None,
            last_stage="translate",
            stage_history=[],
            elapsed_s=0.01,
        )

    monkeypatch.setattr(hailo_backend, "_run_streamed_subprocess", _fake_run)

    result = hailo_backend.hailo_build_hef_via_venv(
        model,
        hw_arch="hailo10h",
        outdir=tmp_path / "out",
    )

    assert result.ok is False
    env = dict(captured["env"])
    assert env["VIRTUAL_ENV"] == str(managed.resolve())
    assert env["PATH"].split(os.pathsep)[0] == str(python.parent.resolve())
    assert shutil.which("onnxsim", path=env["PATH"]) == str(onnxsim.resolve())
