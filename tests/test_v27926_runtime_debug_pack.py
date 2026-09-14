"""Archive-level regression coverage for the missing DeepX Full diagnostics."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from onnx_splitpoint_tool.workflow import debug_pack as packs


MODEL = "models/yolo11l/"
RESULTS = MODEL + "benchmark_results/"
FULL = RESULTS + "benchmark_results_deepx_m1_full_auto.json"
ERROR = "native_full_raw_detection_model_identity_missing"
EXACT = [
    FULL,
    *[RESULTS + name for name in sorted(packs.RUNTIME_DIAGNOSTIC_FILES)],
    MODEL + "validation/validation_case_matrix.json",
    MODEL + "validation/validation_case_matrix.csv",
    MODEL + "model_manifest.json",
    MODEL + "full_baselines/output_contracts.json",
    MODEL + "benchmark_set/legacy_suite/benchmark_plan.json",
    MODEL + "benchmark_set/legacy_suite/output_contracts.json",
    MODEL + "benchmark_set/legacy_suite/deepx/deepx_m1/full/output_contract.json",
    RESULTS + "remote_diagnostics/logs/runner.log",
    RESULTS + "remote_diagnostics/orin_nx_deepx_m1_01/logs/runner.log",
    RESULTS + "remote_diagnostics/lean_bundle/benchmark_suite_status_matrix.json",
    RESULTS + "remote_diagnostics/orin_nx_deepx_m1_01/lean_bundle/benchmark_suite_status_matrix.json",
]


def _write(run: Path, member: str, payload: bytes = b"{}") -> Path:
    path = run / member
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _run(tmp_path: Path) -> Path:
    run = tmp_path / "run"
    _write(run, "evaluation_workflow.log", b"status=failed\n")
    _write(run, "run_manifest.json", b'{"status":"failed"}')
    return run


def _pack(tmp_path: Path, run: Path, **kwargs) -> tuple[dict, dict[str, bytes]]:
    output = tmp_path / "debug.zip"
    result = packs.create_evaluation_debug_pack(run, output, **kwargs)
    assert result["archive_verification"] == "verified"
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
        assert len(archive.namelist()) == len(set(archive.namelist()))
        members = {name: archive.read(name) for name in archive.namelist()}
    return json.loads(members["debug_pack_manifest.json"]), members


def test_full_failure_logs_matrix_and_contracts_are_retained_exactly(tmp_path):
    run = _run(tmp_path)
    payloads = {
        member: json.dumps({"source": member, "error": ERROR}).encode()
        for member in EXACT
    }
    for member, payload in payloads.items():
        _write(run, member, payload)
    manifest, members = _pack(tmp_path, run)
    for member, payload in payloads.items():
        assert members[member] == payload
        assert (run / member).read_bytes() == payload
    assert ERROR.encode() in members[FULL]
    diagnostics = manifest["runtime_execution_diagnostics"]
    assert diagnostics["complete"] and manifest["complete"]
    assert set(diagnostics["archived_members"]) == set(EXACT)


def test_selection_does_not_admit_tensor_model_or_unrelated_json_neighbors(tmp_path):
    run = _run(tmp_path)
    excluded = [
        RESULTS + "benchmark_results_deepx_m1_full_auto.csv",
        RESULTS + "unrelated.json",
        RESULTS + "predictions.json",
        MODEL + "benchmark_set/legacy_suite/deepx/deepx_m1/full/full.dxnn",
        MODEL + "benchmark_set/legacy_suite/deepx/deepx_m1/full/output.bin",
        MODEL + "benchmark_set/legacy_suite/deepx/deepx_m1/full/output_contract_extra.json",
        MODEL + "benchmark_set/legacy_suite/deepx/deepx_m1/b003/output_contract.json",
        RESULTS + "remote_diagnostics/lean_bundle/output_contracts.json",
        RESULTS + "remote_diagnostics/lean_bundle/benchmark_results_x.json",
        RESULTS + "remote_diagnostics/lean_bundle/logs/runner.log",
        RESULTS + "remote_diagnostics/lean_bundle/unrelated/benchmark_suite_status_matrix.json",
        "reports/arbitrary/lean_bundle/benchmark_suite_status_matrix.json",
        "native_producers/arbitrary/lean_bundle/benchmark_suite_status_matrix.json",
    ]
    for member in excluded:
        _write(run, member)
    admitted = RESULTS + "remote_diagnostics/lean_bundle/benchmark_suite_status_matrix.json"
    _write(run, admitted)
    _, members = _pack(tmp_path, run)
    assert admitted in members
    assert not set(excluded) & members.keys()


@pytest.mark.parametrize("declared", [
    FULL,
    "/home/kmika/Models/EvaluationRuns/old_run/" + FULL,
    Path(FULL).name,
])
def test_index_declared_missing_backend_result_makes_pack_partial(tmp_path, declared):
    run = _run(tmp_path)
    _write(run, RESULTS + "selected_run_completeness.json", json.dumps({
        "runs": [{"primary_result_path": declared}],
    }).encode())
    manifest, members = _pack(tmp_path, run)
    diagnostics = manifest["runtime_execution_diagnostics"]
    assert diagnostics["missing_source_members"] == [FULL]
    assert diagnostics["status"] == "partial"
    assert not manifest["complete"]
    assert "evaluation_workflow.log" in members


def test_existing_backend_result_resolves_all_duplicate_index_references(tmp_path):
    run = _run(tmp_path)
    for name in ("normalized_results.json", "selected_run_completeness.json"):
        _write(run, RESULTS + name, json.dumps({
            "rows": [{"source_path": FULL}, {"primary_result_path": FULL}],
        }).encode())
    _write(run, FULL, json.dumps({"status": "runtime_failed", "error": ERROR}).encode())
    manifest, members = _pack(tmp_path, run)
    assert manifest["complete"]
    assert manifest["runtime_execution_diagnostics"]["expected_members"].count(FULL) == 1
    assert ERROR.encode() in members[FULL]


def test_oversized_required_json_is_explicitly_omitted_and_not_complete(tmp_path, monkeypatch):
    monkeypatch.setattr(packs, "STRUCTURED_RESULT_MAX_FILE_BYTES", packs.DEFAULT_MAX_SMALL_FILE_BYTES)
    run = _run(tmp_path)
    _write(run, FULL, b"x" * (packs.DEFAULT_MAX_SMALL_FILE_BYTES + 1))
    _write(run, RESULTS + "selected_run_completeness.json", json.dumps({
        "runs": [{"primary_result_path": FULL}],
    }).encode())
    manifest, members = _pack(tmp_path, run)
    assert FULL not in members
    diagnostics = manifest["runtime_execution_diagnostics"]
    assert diagnostics["omitted_source_members"] == [{"path": FULL, "reason": "too large",
        "size_bytes": packs.DEFAULT_MAX_SMALL_FILE_BYTES + 1,
        "limit_bytes": packs.DEFAULT_MAX_SMALL_FILE_BYTES}]
    assert diagnostics["status"] == "partial"
    assert not diagnostics["complete"] and not manifest["complete"]


def test_large_runner_logs_use_existing_bounded_tails_and_preserve_host_names(tmp_path):
    run = _run(tmp_path)
    logs = [RESULTS + "remote_diagnostics/" + prefix + "logs/runner.log"
            for prefix in ("", "orin_nx_deepx_m1_01/")]
    for index, member in enumerate(logs):
        _write(run, member, b"old lines\n" * 128 + f"HOST-{index}:{ERROR}\n".encode())
    manifest, members = _pack(tmp_path, run, max_small_file_bytes=256, tail_bytes=128)
    for index, member in enumerate(logs):
        assert member not in members
        assert members[f"debug_tails/{member}.tail"].endswith(f"HOST-{index}:{ERROR}\n".encode())
    assert len(manifest["runtime_execution_diagnostics"]["bounded_log_tails"]) == 2
    assert not manifest["complete"]


@pytest.mark.parametrize("declared", [
    "/outside/benchmark_results_deepx_m1_full_auto.json",
    "../benchmark_results_deepx_m1_full_auto.json",
    "models/another/benchmark_results/benchmark_results_deepx_m1_full_auto.json",
])
def test_unrelated_or_traversing_reference_is_not_followed(tmp_path, declared):
    run = _run(tmp_path)
    external = _write(tmp_path, "benchmark_results_deepx_m1_full_auto.json", b"external secret")
    _write(run, RESULTS + "normalized_results.json", json.dumps({
        "rows": [{"source_path": declared}],
    }).encode())
    manifest, members = _pack(tmp_path, run)
    assert manifest["runtime_execution_diagnostics"]["reference_failures"]
    assert not manifest["complete"]
    assert not any(b"external secret" in payload for payload in members.values())
    assert external.read_bytes() == b"external secret"


def test_invalid_runtime_index_is_preserved_with_explicit_incomplete_status(tmp_path):
    run = _run(tmp_path)
    member = RESULTS + "selected_run_completeness.json"
    _write(run, member, b"{interrupted-write")
    manifest, members = _pack(tmp_path, run)
    assert members[member] == b"{interrupted-write"
    assert manifest["runtime_execution_diagnostics"]["reference_failures"] == [{
        "path": member, "reason": "runtime_index_invalid:JSONDecodeError",
    }]
    assert not manifest["complete"]
