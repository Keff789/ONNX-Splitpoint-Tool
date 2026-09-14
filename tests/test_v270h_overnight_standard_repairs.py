from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from onnx_splitpoint_tool.benchmark.remote_run import (
    _assert_generated_runner_is_self_consistent,
)
from onnx_splitpoint_tool.benchmark.suite_refresh import (
    assert_generated_runner_is_self_consistent,
)
from onnx_splitpoint_tool.gui.panels.panel_evaluation_workflow import (
    _commit_profile_summary,
)
from onnx_splitpoint_tool.native_performance_reporting import (
    collect_native_performance_matrix,
)
from onnx_splitpoint_tool.run_modes import (
    apply_run_mode,
    default_run_modes_config,
)


ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_suite_bootstrap_precedes_first_vendored_quality_import() -> None:
    source = (
        ROOT / "onnx_splitpoint_tool/resources/templates"
        / "run_split_onnxruntime.py.txt"
    ).read_text(encoding="utf-8")
    bootstrap = source.index("\n_maybe_add_suite_runtime_to_syspath()\n")
    quality_import = source.index(
        "from splitpoint_runners.native_split_quality_runtime import"
    )
    assert bootstrap < quality_import
    assert source.count("\n_maybe_add_suite_runtime_to_syspath()\n") == 1


@pytest.mark.parametrize(
    "self_check",
    [
        _assert_generated_runner_is_self_consistent,
        assert_generated_runner_is_self_consistent,
    ],
)
def test_static_packaging_guards_reject_late_suite_bootstrap(
    tmp_path: Path,
    self_check,
) -> None:
    runner = tmp_path / "run_split_onnxruntime.py"
    runner.write_text(
        "\n".join(
            [
                "from splitpoint_runners.native_split_quality_runtime import helper",
                "def _maybe_add_suite_runtime_to_syspath():",
                "    return None",
                "_maybe_add_suite_runtime_to_syspath()",
                "",
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="before the suite root"):
        self_check(runner)


def test_failed_summary_render_cannot_advance_start_guard() -> None:
    class FailingWidget:
        text = "Part-2 inputs=1=off"

        def configure(self, **_kwargs) -> None:
            return None

        def delete(self, *_args) -> None:
            raise RuntimeError("simulated Tcl failure")

        def insert(self, *_args) -> None:
            self.text = str(_args[-1])

    app = SimpleNamespace(
        _evaluation_workflow_visible_start_snapshot={"snapshot_sha256": "old"}
    )
    widget = FailingWidget()
    assert _commit_profile_summary(
        app,
        widget,
        "Part-2 inputs=1=on",
        {"profile_request": "profile", "snapshot_sha256": "new"},
    ) is False
    assert widget.text == "Part-2 inputs=1=off"
    assert app._evaluation_workflow_visible_start_snapshot == {}


def test_smoke_and_standard_keep_automatic_model_hashing() -> None:
    config = default_run_modes_config()
    for mode in ("smoke", "standard", "final"):
        profile = {
            "name": f"hash_{mode}",
            "selection_policy": {
                "max_accepted_cases_per_model": 1,
                "preferred_shortlist": 1,
                "selection_strategy": "stratified_windows",
                "min_gap": 1,
                "candidate_search_pool": "auto",
            },
            "model_suite": {
                "primary": [
                    {
                        "id": "resnet50",
                        "task": "classification",
                        "enabled": True,
                    }
                ]
            },
            "run_profiles": [{"id": "ort_tensorrt", "enabled": True}],
            "execution_preset": {
                "id": mode,
                "follow_tool_config": True,
                "overrides": {
                    "native_enabled": False,
                    "energy_enabled": False,
                },
            },
        }
        resolved, _audit = apply_run_mode(profile, config=config)
        assert resolved["workflow"]["no_model_hash"] is False


def test_preflight_matrix_writer_precedes_central_binding_resolution() -> None:
    source = (
        ROOT / "onnx_splitpoint_tool/workflow/runner.py"
    ).read_text(encoding="utf-8")
    provisional = source.index(
        "initial_expected_matrix = _native_expected_matrix_status_v60y("
    )
    persisted = source.index(
        "_persist_native_expected_matrix(initial_expected_matrix)"
    )
    preflight = source.index(
        "central_binding_preflight: Dict[str, Dict[str, Any]] = {}"
    )
    assert provisional < persisted < preflight


def test_explicit_zero_native_matrix_counts_remain_authoritative(
    tmp_path: Path,
) -> None:
    reports = tmp_path / "reports"
    _write_json(
        reports / "native_expected_matrix.json",
        {
            "expected_row_count": 45,
            "present_expected_row_count": 0,
            "successful_expected_row_count": 0,
            "failed_expected_row_count": 0,
            "missing_expected_row_count": 45,
            "present_expected_rows": [],
        },
    )
    _write_json(
        reports / "native_producer_summary.json",
        {
            "rows": [
                {
                    "model": "resnet50",
                    "backend": "hailo8_to_trt",
                    "case": "b001",
                    "status": "ok",
                    "ok": True,
                    "fps_makespan": 10.0,
                }
            ]
        },
    )
    matrix = collect_native_performance_matrix(tmp_path)
    assert matrix["expected_row_count"] == 45
    assert matrix["present_expected_row_count"] == 0
    assert matrix["missing_expected_row_count"] == 45
    assert matrix["matrix_complete"] is False

