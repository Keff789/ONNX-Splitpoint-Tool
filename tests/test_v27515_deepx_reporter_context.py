from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from onnx_splitpoint_tool.workflow import runner as workflow_runner
from scripts import native_producer_final_report as final_report
from scripts import run_evalrun_native_producer_variants as variants_runner
from scripts import update_evalset_native_producers as updater


def _context(
    *, setup_id: str = "orin_nx_deepx_m1_01",
    remote_root: str = "/srv/evals/run-a",
    remote_tool_dir: str = "/srv/ONNX-Splitpoint-Tool",
) -> dict[str, Any]:
    return {
        "schema": (
            "onnx-splitpoint/"
            "native-final-report-remote-execution-context"
        ),
        "schema_version": 1,
        "setup_id": setup_id,
        "remote_root": remote_root,
        "remote_tool_dir": remote_tool_dir,
    }


def test_reporter_remote_context_allowlist_selects_exact_setup_and_root() -> None:
    records = final_report._parse_remote_execution_context_allowlist([
        json.dumps(_context()),
        json.dumps(_context(remote_root="/srv/evals/run-b")),
        json.dumps(_context(setup_id="deepx-spare")),
    ])

    selected, status = (
        final_report._remote_execution_context_for_performance_row(
            {"setup_id": "orin_nx_deepx_m1_01"},
            {"root": "/srv/evals/run-b"},
            records,
        )
    )

    assert status == "remote_execution_context_verified_exact"
    assert selected == {
        "setup_id": "orin_nx_deepx_m1_01",
        "remote_root": "/srv/evals/run-b",
        "remote_tool_dir": "/srv/ONNX-Splitpoint-Tool",
    }


@pytest.mark.parametrize(
    "mutator",
    (
        lambda value: {**value, "remote_root": "srv/evals/run-a"},
        lambda value: {**value, "remote_root": "/srv/evals/../run-a"},
        lambda value: {**value, "setup_id": ""},
        lambda value: {**value, "schema_version": 2},
    ),
)
def test_reporter_remote_context_allowlist_rejects_untrusted_records(
    mutator: Any,
) -> None:
    with pytest.raises(ValueError):
        final_report._parse_remote_execution_context_allowlist([
            mutator(_context()),
        ])


def test_reporter_remote_context_allowlist_rejects_tool_dir_drift() -> None:
    with pytest.raises(
        ValueError, match="tool-dir conflicts for setup/root",
    ):
        final_report._parse_remote_execution_context_allowlist([
            _context(),
            _context(remote_tool_dir="/srv/other-tool"),
        ])


def test_deepx_final_report_passes_allowlisted_remote_context_to_strict_validator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    digest = "c" * 64
    captured: dict[str, Any] = {}

    def verify(
        contract: dict[str, Any], *, expected_identity: dict[str, Any],
    ) -> tuple[dict[str, Any], str]:
        captured.update(expected_identity)
        return contract, "full_command_contract_verified"

    monkeypatch.setattr(
        final_report, "_strict_verify_full_command_contract", verify,
    )
    monkeypatch.setattr(
        final_report, "_quality_binding_evidence",
        lambda _row: ({
            "model_sha256": "a" * 64,
            "command_contract_sha256": digest,
        }, []),
    )
    row = {
        "backend": "native_full_deepx",
        "model": "yolo26s",
        "case": "full",
        "setup_id": "orin_nx_deepx_m1_01",
        "comparison_backend": "deepx",
        "full_command_contract": {
            "root": "/srv/evals/run-a",
            "contract_sha256": digest,
        },
    }
    records = final_report._parse_remote_execution_context_allowlist([
        _context(),
    ])

    verified_digest, errors = (
        final_report._verified_performance_command_contract(
            row, remote_execution_contexts=records,
        )
    )

    assert verified_digest == digest
    assert errors == []
    assert captured["remote_root"] == "/srv/evals/run-a"
    assert captured["remote_tool_dir"] == "/srv/ONNX-Splitpoint-Tool"


def test_deepx_final_report_fails_before_validator_without_exact_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def verify(*_args: Any, **_kwargs: Any) -> tuple[dict[str, Any], str]:
        nonlocal called
        called = True
        return {}, "unexpected"

    monkeypatch.setattr(
        final_report, "_strict_verify_full_command_contract", verify,
    )
    monkeypatch.setattr(
        final_report, "_quality_binding_evidence",
        lambda _row: ({"model_sha256": "a" * 64}, []),
    )
    row = {
        "backend": "native_full_deepx",
        "model": "yolo26s",
        "case": "full",
        "setup_id": "orin_nx_deepx_m1_01",
        "full_command_contract": {
            "root": "/srv/evals/run-a",
            "contract_sha256": "c" * 64,
        },
    }

    verified_digest, errors = (
        final_report._verified_performance_command_contract(
            row, remote_execution_contexts=[
                _context(remote_root="/srv/evals/other"),
            ],
        )
    )

    assert verified_digest == ""
    assert called is False
    assert any(
        "full_command_contract_deepx_remote_execution_context_missing"
        in error for error in errors
    )


@pytest.mark.parametrize(
    "serializer",
    (
        updater._final_report_remote_context_args,
        workflow_runner._native_final_report_remote_context_args,
    ),
)
def test_productive_report_callers_serialize_external_deepx_allowlist(
    serializer: Any,
) -> None:
    args = serializer([{
        "backend": "deepx",
        "setup_id": "orin_nx_deepx_m1_01",
        "remote_root": "/srv/evals/run-a",
        "remote_tool_dir": "/srv/ONNX-Splitpoint-Tool",
    }])

    assert args[0] == "--remote-execution-context-json"
    assert json.loads(args[1]) == _context()


def test_variant_reporter_contexts_keep_same_setup_separate_by_contract_root(
    tmp_path: Path,
) -> None:
    base: dict[str, Any] = {
        "frames": 100,
        "warmup": 10,
        "repetitions": 1,
        "queue_depth": 2,
        "inflight": 4,
        "backends": ["deepx"],
        "remote_root": "/srv/evals",
        "remote_tool_dir": "/srv/ONNX-Splitpoint-Tool",
        "remotes": {
            "deepx": {"setup_id": "orin_nx_deepx_m1_01"},
        },
    }
    base["_native_execution_contract"] = (
        variants_runner.build_native_execution_contract(
            base, run_mode="smoke",
        )
    )
    run_dir = tmp_path / "run-a"
    variant_rows = [
        {"artifact_namespace": "variant_one"},
        {"artifact_namespace": "variant_two"},
    ]

    args = variants_runner._final_report_remote_context_args(
        run_dir, base, variant_rows,
    )
    records = [
        json.loads(args[index]) for index in range(1, len(args), 2)
    ]

    assert {record["remote_root"] for record in records} == {
        "/srv/evals/run-a/variants/variant_one",
        "/srv/evals/run-a/variants/variant_two",
    }
    assert {record["setup_id"] for record in records} == {
        "orin_nx_deepx_m1_01",
    }

    with pytest.raises(
        variants_runner.TensorRTQualityChainError,
        match="remote context drift",
    ):
        variants_runner._final_report_remote_context_args(
            run_dir,
            base,
            [
                {"artifact_namespace": "same"},
                {
                    "artifact_namespace": "same",
                    "remote_tool_dir": "/srv/other-tool",
                },
            ],
        )
