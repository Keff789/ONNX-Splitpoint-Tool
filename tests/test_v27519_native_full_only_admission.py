from __future__ import annotations

import json
from pathlib import Path

import pytest

from onnx_splitpoint_tool.remote.process_lease import RemoteProcessLeaseScope
from onnx_splitpoint_tool.workflow.runner import (
    EvaluationWorkflowRunner,
    WorkflowOptions,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_full_only_native_admits_suite_without_split_case_directories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Full baseline consumes suite resources, not a synthetic ``b*`` row."""

    runner = EvaluationWorkflowRunner(
        WorkflowOptions(profile="", out=str(tmp_path)),
    )
    runner.run_id = "full_only_no_cases"
    runner.run_dir = tmp_path / runner.run_id
    runner.run_dir.mkdir()
    runner._remote_process_registry.configure_journal(
        scope=RemoteProcessLeaseScope(runner.run_id, runner.session_id),
        journal_dir=runner.run_dir / "reports" / "remote_lease_journal",
    )
    runner.manifest = {"models": {"resnet50": {}}}
    runner.profile_payload = {
        "execution_preset": {"id": "smoke"},
        "run_profiles": [
            {
                "id": "ort_tensorrt", "type": "same_backend_reference",
                "full": "tensorrt", "stage1": "tensorrt",
                "stage2": "tensorrt",
            },
            {
                "id": "hailo8", "type": "same_backend_reference",
                "full": "hailo8", "stage1": "hailo8",
                "stage2": "hailo8",
            },
        ],
        "native_producers": {"enabled": True},
    }
    suite = (
        runner.run_dir / "models" / "resnet50" / "benchmark_set"
        / "legacy_suite"
    )
    _write_json(suite / "benchmark_set.json", {"cases": []})
    _write_json(suite / "benchmark_plan.json", {
        "runs": [{"id": "ort_tensorrt"}, {"id": "hailo8"}],
    })
    (suite / "benchmark_suite.py").write_text("pass\n", encoding="utf-8")

    cfg = {
        "enabled": True,
        "models": ["resnet50"],
        "backends": ["hailo8"],
        "split_backends": [],
        "full_baselines": {
            "enabled": True,
            "backends_by_producer": {
                "hailo8": ["hailo8", "tensorrt"],
            },
        },
        "energy": {"enabled": False},
        "validation": {"enabled": False},
        "copy_benchmarksets": False,
        "remotes": {},
    }
    monkeypatch.setattr(runner, "_native_producer_config", lambda: cfg)
    monkeypatch.setattr(
        "onnx_splitpoint_tool.workflow.runner.normalize_hardware_targets",
        lambda _profile: [],
    )

    paths, _details, _message, status = (
        runner._stage_run_native_producers()
    )

    assert status == "failed"  # planning finishes; absent SSH remains infrastructure failure
    stage = json.loads(
        paths["native_producer_stage_json"].read_text(encoding="utf-8")
    )
    assert "selection_errors" not in stage
    assert stage["failure_class"] == "global_remote_infrastructure"
    assert "remote_ssh_missing" in stage["failure_reason"]
    assert stage["native_split_requires_single_part2_input"] is False
    assert stage["native_split_selected_case_count"] == 0
    assert stage["native_split_supported_case_count"] == 0
    assert stage["benchmark_set_validation"]["resnet50"][
        "native_full_only_valid"
    ] is True
    rows = stage["expected_native_rows"]
    assert len(rows) == 2
    assert not any(row["execution_mode"] == "native_split" for row in rows)
    assert {
        row["backend"] for row in rows
    } == {"native_full_hailo8", "native_full_tensorrt"}
