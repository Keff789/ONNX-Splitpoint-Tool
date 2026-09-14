from __future__ import annotations

import csv
import json
from pathlib import Path
from unittest import mock

from onnx_splitpoint_tool.workflow.contracts import WorkflowOptions
from onnx_splitpoint_tool.workflow.runner import (
    EvaluationWorkflowRunner,
    _missing_source_tokens_v264,
    _native_concise_summary_v60w,
    _native_expected_matrix_status_v60y,
    _native_expected_full_rows_v61b,
    _native_preflight_asset_contract_v264,
    _native_split_energy_preflight_dependency_spec_v264,
    _native_window_probe_upstream_block_v264,
)


ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_preflight_dependency_spec_matches_source_and_packaged_mirror() -> None:
    name, tokens = _native_split_energy_preflight_dependency_spec_v264()
    assert name == "native_split_energy_preflight.py"
    assert "SPLIT_ENERGY_PREFLIGHT_STDOUT_MARKER" in tokens
    assert "__SPLITPOINT_PREFLIGHT_ATTESTATION__" not in tokens

    source = ROOT / "scripts" / name
    packaged = ROOT / "onnx_splitpoint_tool" / "resources" / "remote_scripts" / name
    assert source.read_bytes() == packaged.read_bytes()
    assert _missing_source_tokens_v264(source, tokens) == []
    assert _missing_source_tokens_v264(packaged, tokens) == []
    contract = _native_preflight_asset_contract_v264()
    assert contract["ok"] is True
    assert contract["authoritative_kind"] == "source_and_packaged_mirror"
    assert contract["source_packaged_bytes_equal"] is True


def test_preflight_asset_contract_accepts_packaged_resource_without_source(
    tmp_path: Path,
) -> None:
    name, _tokens = _native_split_energy_preflight_dependency_spec_v264()
    packaged = tmp_path / "site-packages" / "remote_scripts" / name
    packaged.parent.mkdir(parents=True)
    packaged.write_bytes((ROOT / "scripts" / name).read_bytes())
    contract = _native_preflight_asset_contract_v264(
        source_script=tmp_path / "wheel_has_no_top_level_scripts" / name,
        packaged_script=packaged,
    )
    assert contract["ok"] is True
    assert contract["source"]["present"] is False
    assert contract["packaged"]["capability_ok"] is True
    assert contract["authoritative_kind"] == "packaged_resource"
    assert contract["authoritative_path"] == str(packaged)


def test_campaign_preflight_fails_early_on_injected_source_mirror_mismatch(
    tmp_path: Path, monkeypatch,
) -> None:
    name, _tokens = _native_split_energy_preflight_dependency_spec_v264()
    original = (ROOT / "scripts" / name).read_bytes()
    source = tmp_path / "source" / name
    packaged = tmp_path / "package" / name
    source.parent.mkdir(parents=True)
    packaged.parent.mkdir(parents=True)
    source.write_bytes(original + b"\n# injected source-only mismatch\n")
    packaged.write_bytes(original)
    mismatch = _native_preflight_asset_contract_v264(
        source_script=source,
        packaged_script=packaged,
    )
    assert mismatch["ok"] is False
    assert mismatch["source_packaged_bytes_equal"] is False
    assert "source_packaged_preflight_mirror_mismatch" in mismatch["errors"]

    runner = EvaluationWorkflowRunner(WorkflowOptions(profile="", out=str(tmp_path)))
    runner.run_id = "v264_early_native_contract_failure"
    runner.run_dir = tmp_path / runner.run_id
    runner.profile_payload = {"campaign": {"mode": "development"}}
    monkeypatch.setattr(
        runner,
        "_native_producer_config",
        lambda: {"enabled": True},
    )
    monkeypatch.setattr(
        "onnx_splitpoint_tool.workflow.runner._native_preflight_asset_contract_v264",
        lambda: mismatch,
    )
    monkeypatch.setattr(
        "onnx_splitpoint_tool.workflow.runner.build_campaign_readiness",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("campaign readiness must not run after local Native contract failure")
        ),
    )

    artifacts, details, message, status = runner._stage_campaign_preflight()
    assert status == "failed"
    assert runner._stop_requested is True
    assert details["native_preflight_asset_contract_ok"] is False
    assert details["native_preflight_asset_contract_errors"] == mismatch["errors"]
    assert "no model build" in message
    archived = artifacts["native_preflight_asset_contract_json"]
    assert json.loads(archived.read_text(encoding="utf-8"))["errors"] == mismatch["errors"]


def test_campaign_preflight_archives_successful_native_asset_contract(
    tmp_path: Path, monkeypatch,
) -> None:
    runner = EvaluationWorkflowRunner(WorkflowOptions(profile="", out=str(tmp_path)))
    runner.run_id = "v264_early_native_contract_success"
    runner.run_dir = tmp_path / runner.run_id
    runner.profile_payload = {"campaign": {"mode": "development"}}
    monkeypatch.setattr(runner, "_native_producer_config", lambda: {"enabled": True})
    monkeypatch.setattr(
        "onnx_splitpoint_tool.workflow.runner.build_campaign_readiness",
        lambda *_args, **_kwargs: {
            "status": "development_ready",
            "ready": True,
            "required_failure_count": 0,
        },
    )
    monkeypatch.setattr(
        "onnx_splitpoint_tool.workflow.runner.readiness_markdown",
        lambda _report: "# ready\n",
    )
    monkeypatch.setattr(
        "onnx_splitpoint_tool.workflow.runner.materialize_campaign_inputs",
        lambda *_args, **_kwargs: {"artifacts": []},
    )

    artifacts, details, _message, status = runner._stage_campaign_preflight()
    assert status == "ok"
    assert runner._stop_requested is False
    assert details["native_enabled"] is True
    assert details["native_preflight_asset_contract_ok"] is True
    contract_path = artifacts["native_preflight_asset_contract_json"]
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    assert contract["ok"] is True
    assert contract["source_packaged_bytes_equal"] is True


def test_window_probe_block_preserves_native_transfer_root_cause(tmp_path: Path) -> None:
    block = _native_window_probe_upstream_block_v264(
        tmp_path / "missing_native_producer_summary.json",
        [
            {
                "backend": "hailo8",
                "ok": False,
                "status": "partial",
                "failure_reason": "native_transfer_failed",
                "error": "capability verification failed",
            },
            {
                "backend": "deepx",
                "ok": False,
                "status": "partial",
                "failure_reason": "native_transfer_failed",
                "error": "capability verification failed",
            },
        ],
    )
    assert block["status"] == "blocked_by_upstream"
    assert block["blocked_reason"] == "native_transfer_failed"
    assert block["upstream_condition"] == "native_producer_summary_missing"
    assert block["successful_native_row_count"] == 0
    assert block["upstream_failure_reasons"] == ["native_transfer_failed"]
    assert {row["backend"] for row in block["upstream_backend_failures"]} == {
        "hailo8", "deepx",
    }


def test_window_probe_upstream_check_allows_a_replay_capable_native_row(
    tmp_path: Path, monkeypatch,
) -> None:
    summary = tmp_path / "native_producer_summary.json"
    row = {"backend": "hailo8_to_trt", "model": "resnet50", "case": "b001", "ok": True}
    _write_json(summary, {"rows": [row]})
    monkeypatch.setattr(
        "onnx_splitpoint_tool.window_method_validation_probe._verified_native_contract",
        lambda source, target: ({"backend": source["backend"]}, "verified"),
    )
    assert _native_window_probe_upstream_block_v264(summary, []) == {}


def test_window_probe_is_blocked_by_full_only_success(tmp_path: Path) -> None:
    summary = tmp_path / "native_producer_summary.json"
    _write_json(summary, {"rows": [{
        "backend": "native_full_tensorrt",
        "model": "resnet50",
        "case": "full",
        "ok": True,
    }]})

    block = _native_window_probe_upstream_block_v264(summary, [])

    assert block["status"] == "blocked_by_upstream"
    assert block["upstream_condition"] == "no_replay_capable_native_split_rows"
    assert block["successful_native_row_count"] == 1
    assert block["replay_capable_native_row_count"] == 0
    assert block["replay_contract_rejections"][0]["reason"] == (
        "native_command_contract_probe_backend_unsupported"
    )


def test_window_probe_is_blocked_when_successful_split_contract_is_invalid(
    tmp_path: Path,
) -> None:
    summary = tmp_path / "native_producer_summary.json"
    _write_json(summary, {"rows": [{
        "backend": "hailo8_to_trt",
        "model": "resnet50",
        "case": "b001",
        "precision": "fp16",
        "ok": True,
        "native_command_contract": {"complete": True},
    }]})

    block = _native_window_probe_upstream_block_v264(summary, [])

    assert block["upstream_condition"] == "no_replay_capable_native_split_rows"
    assert block["replay_contract_rejections"][0]["reason"].startswith(
        "native_command_contract_"
    )


def test_strict_window_probe_is_non_strictly_blocked_when_native_summary_is_absent(
    tmp_path: Path, monkeypatch,
) -> None:
    runner = EvaluationWorkflowRunner(WorkflowOptions(profile="", out=str(tmp_path)))
    runner.run_id = "v264_upstream_probe_block"
    runner.run_dir = tmp_path / runner.run_id
    runner.profile_payload = {
        "campaign": {"mode": "development"},
        "measurement_campaign": {"system_power": {"scope": "system", "window": "command"}},
    }
    runner.manifest = {"models": {"resnet50": {}}}
    suite = runner.run_dir / "models" / "resnet50" / "benchmark_set"
    (suite / "b001").mkdir(parents=True)
    _write_json(suite / "benchmark_set.json", {"cases": [{"id": "b001"}]})
    _write_json(suite / "benchmark_plan.json", {"runs": [{"id": "split"}]})
    (suite / "benchmark_suite.py").write_text("# test harness\n", encoding="utf-8")
    cfg = {
        "enabled": True,
        "models": ["resnet50"],
        "backends": ["hailo8"],
        "remotes": {"hailo8": {}},
        "validation": {"enabled": False},
        "energy": {
            "enabled": False,
            "window_method_validation_probe": {
                "enabled": True,
                "strict": True,
                "repeats": 3,
            },
        },
        "cleanup_remote_native_root": False,
    }

    # No downstream probe subprocess is allowed to run in this scenario.
    monkeypatch.setattr(
        "onnx_splitpoint_tool.workflow.runner.run_streaming",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("window-method probe must not execute without Native evidence")
        ),
    )
    monkeypatch.setattr(
        "onnx_splitpoint_tool.workflow.runner.normalize_hardware_targets",
        lambda *_args, **_kwargs: [],
    )
    with mock.patch.object(runner, "_native_producer_config", return_value=cfg):
        paths, details, _message, status = runner._stage_run_native_producers()

    assert status == "partial"
    assert details["window_method_validation_probe_strict_failure"] is False
    probe = details["window_method_validation_probe"]
    assert probe["status"] == "blocked_by_upstream"
    assert probe["upstream_blocked"] is True
    assert probe["cmd_executed"] is False
    assert probe["rc"] is None
    assert probe["strict_requested"] is True
    assert probe["strict_failure"] is False
    report = json.loads(Path(probe["report"]).read_text(encoding="utf-8"))
    assert report["strict_requested"] is True
    assert report["strict_failure"] is False
    assert report["upstream_condition"] == "native_producer_summary_missing"
    assert paths["native_producer_stage_json"].is_file()


def test_concise_native_summary_keeps_all_eighteen_setup_local_expectations(
    tmp_path: Path,
) -> None:
    reports = tmp_path / "reports"
    reports.mkdir()
    mapping = {
        "hailo8": ["hailo8", "tensorrt"],
        "hailo10h": ["hailo10h", "tensorrt"],
        "deepx": ["deepx", "tensorrt"],
    }
    setup_ids = {"hailo8": "h8", "hailo10h": "h10", "deepx": "dx"}
    expected = _native_expected_full_rows_v61b(
        ["resnet50", "yolo26s"], mapping, setup_ids
    )
    for producer, backend in (
        ("hailo8", "hailo8_to_trt"),
        ("hailo10h", "hailo10h_to_trt"),
        ("deepx", "deepx_to_trt"),
    ):
        for model in ("resnet50", "yolo26s"):
            expected.append({
                "backend_key": producer,
                "backend": backend,
                "model": model,
                "case": "b001",
                "precision": "fp16",
                "execution_mode": "native_split",
                "failure_reason": "native_transfer_failed",
            })

    paths, rows = _native_concise_summary_v60w(
        reports, missing_expected_rows=expected
    )
    assert len(expected) == len(rows) == 18
    trt = [row for row in rows if row["backend"] == "native_full_tensorrt"]
    assert len(trt) == 6
    assert {(row["setup_id"], row["comparison_backend"]) for row in trt} == {
        ("h8", "hailo8"),
        ("h10", "hailo10h"),
        ("dx", "deepx"),
    }
    payload = json.loads(paths["native_stage_concise_summary_json"].read_text(encoding="utf-8"))
    assert payload["schema_version"] == 4
    with paths["native_stage_concise_summary_csv"].open(newline="", encoding="utf-8") as handle:
        assert {"setup_id", "comparison_backend"}.issubset(csv.DictReader(handle).fieldnames or [])


def test_native_full_matrix_identity_includes_comparison_backend() -> None:
    expected = [
        {
            "backend_key": "hailo8",
            "backend": "native_full_tensorrt",
            "model": "resnet50",
            "case": "full",
            "precision": "",
            "setup_id": "shared_setup",
            "comparison_backend": "hailo8",
        },
        {
            "backend_key": "deepx",
            "backend": "native_full_tensorrt",
            "model": "resnet50",
            "case": "full",
            "precision": "",
            "setup_id": "shared_setup",
            "comparison_backend": "deepx",
        },
    ]
    actual = [{**expected[0], "ok": True}]

    status = _native_expected_matrix_status_v60y(expected, actual, [])

    assert status["present_expected_row_count"] == 1
    assert status["missing_expected_row_count"] == 1
    assert status["matrix_complete"] is False
    assert status["missing_expected_rows"][0]["comparison_backend"] == "deepx"


def test_local_quality_policy_skip_is_non_blocking(tmp_path: Path) -> None:
    runner = object.__new__(EvaluationWorkflowRunner)
    runner.profile_payload = {
        "campaign": {"mode": "development"},
        "quality_gate": {"statistics": {"execution_location": "local"}},
    }
    runner.run_dir = tmp_path
    runner.stage_results = [{
        "model_id": None,
        "stage": "evaluate_quality",
        "status": "skipped",
        "notes": ["Central management quality evaluation is not selected by the profile."],
    }]

    status, decision = runner._derive_final_status()
    assert status == "ok"
    assert decision["blocking_reasons"] == []
    assert decision["non_blocking_reason_count"] == 1
    assert decision["non_blocking_reasons"][0]["stage"] == "evaluate_quality"


def test_central_quality_summary_records_resolved_custom_worker_count(tmp_path: Path) -> None:
    runner = EvaluationWorkflowRunner(WorkflowOptions(profile="", out=str(tmp_path)))
    runner.run_dir = tmp_path
    runner.profile_payload = {
        "quality_gate": {
            "statistics": {
                "execution_location": "central_management",
                "workers": 6,
            }
        }
    }

    artifacts, details, message, status = runner._stage_evaluate_quality()

    assert status == "skipped"
    assert details["request_count"] == 0
    payload = json.loads(artifacts["central_quality_summary_json"].read_text(encoding="utf-8"))
    assert payload["workers"] == 6
    assert "four processes" not in message
