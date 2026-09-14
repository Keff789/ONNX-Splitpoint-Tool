"""AP3B: real final writers, explicit provenance and bounded derived payloads."""
from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path
from unittest import mock

import pytest

from onnx_splitpoint_tool.workflow import compact_runtime_diagnostics as compact
from onnx_splitpoint_tool.workflow.execution_binding import (
    _inspect_canonical_result, _copy_remote_result_files,
)
from onnx_splitpoint_tool.workflow.results import (
    discover_result_files, normalize_benchmark_files,
)


def _source(tmp_path: Path, name: str = "benchmark_results_run.json") -> Path:
    path = tmp_path / "models" / "m1" / "benchmark_results" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _payload(index: int = 0, *, large: bool = False) -> list[dict]:
    return [{
        "model_id": f"model-{index}", "case_id": f"b{index:03d}", "run_id": f"run-{index}",
        "setup_id": f"device-{index}", "backend": "hailo8", "variant": "full",
        "precision": "uint8", "measurement_endpoint": "raw_accelerator",
        "quality_endpoint": "classification_logits", "status": "ok" if index % 2 == 0 else "blocked_upstream_quality",
        "runtime_ok": index % 2 == 0, "primary_error": "" if index % 2 == 0 else "quality_reference_invalid",
        "error_stage": "quality_gate", "attempted_repetitions": 3 if index % 2 == 0 else 0,
        "completed_repetitions": 3 if index % 2 == 0 else 0,
        "total_latency_ms": 10.25 + index,
        "quality": {"role": "central_quality_decision", "status": "PASS" if index % 2 == 0 else "ERROR", "n": 500,
                    "top1_accuracy": 0.7 + index / 1000, "metadata": {"secret": "NEVER_COPY"}},
        "mini_classification_eval": {"role": "generic_mini_diagnostic", "n": 16, "top1_accuracy": 0.75},
        "energy": {"scope": "FS", "window_label": "command", "duration_s": 1.0,
                   "repetitions": 3, "energy_j": 13.5 + index, "qualification": "screening",
                   "raw": {"samples": ["NEVER_COPY"]}},
        "details": {"metadata": {"raw_outputs": ["NEVER_COPY"]}},
        "predictions": "NEVER_COPY" + "x" * (2 * 1024 * 1024 if large else 100),
        "raw_outputs": [[1, 2, 3]], "base64": "NEVER_COPY",
    }]


def _read_summary(path: Path) -> dict:
    return json.loads(compact.companion_path(path).read_text())


def test_43_large_originals_use_real_canonical_inspector_once_without_model_access(tmp_path, monkeypatch):
    # Preserve the companion path at an explicitly reduced original-file budget.
    from onnx_splitpoint_tool.workflow import debug_pack as packs
    monkeypatch.setattr(packs, "STRUCTURED_RESULT_MAX_FILE_BYTES", 2 * 1024 * 1024)
    calls = []
    read_text = Path.read_text
    def counted(path, *args, **kwargs):
        if path.name.startswith("benchmark_results_") and path.parent.name == "benchmark_results":
            calls.append(path)
        if path.suffix in {".onnx", ".hef", ".dxnn", ".npz"}:
            raise AssertionError("projection accessed a model/raw binary")
        return read_text(path, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", counted)
    for index in range(43):
        source = _source(tmp_path, f"benchmark_results_run{index}.json")
        original = json.dumps(_payload(index, large=True)).encode()
        source.write_bytes(original)
        detail = _inspect_canonical_result(source)
        assert detail["parseable"] and detail["nonempty"] and detail["row_count"] == 1
        assert "diagnostic_summary" in detail, "production canonical parser must publish the projection"
        summary = _read_summary(source)
        assert summary["status"] == "projected"
        assert summary["source"]["path"] == source.relative_to(tmp_path).as_posix()
        assert summary["source"]["observed_size_bytes"] == len(original)
        assert summary["source_row_count"] == summary["projected_row_count"] == 1
        assert summary["rows"][0]["setup_id"] == f"device-{index}"
        assert summary["rows"][0]["total_latency_ms"] == 10.25 + index
        assert summary["rows"][0]["energy"]["energy_j"] == 13.5 + index
        assert len(compact.summary_bytes(summary)) < 10000
        assert b"NEVER_COPY" not in compact.summary_bytes(summary)
        assert source.read_bytes() == original
    assert len(calls) == 43
    # Complete production chain: companions selected by the actual exporter;
    # large originals cannot be reopened just to construct a summary.
    from onnx_splitpoint_tool.workflow.debug_pack import create_evaluation_debug_pack
    real_open = Path.open
    def no_reparse(path, *args, **kwargs):
        if path.name.startswith("benchmark_results_") and path.parent.name == "benchmark_results":
            raise AssertionError("export reread a large source with valid production companion")
        return real_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", no_reparse)
    result = create_evaluation_debug_pack(tmp_path, tmp_path.parent / (tmp_path.name + ".zip"))
    with zipfile.ZipFile(result["out_zip"]) as archive:
        assert archive.testzip() is None
        inventory = json.loads(archive.read("debug_pack_manifest.json"))["runtime_execution_diagnostics"]
        assert inventory["derived_summary_count"] == 43
        assert len(inventory["source_coverage"]) == 43
        assert inventory["compact_view_complete_for_discovered_sources"]
        assert not inventory["complete"]
        for row in inventory["source_coverage"]:
            assert row["summary_origin"] == "existing_companion"
            summary = json.loads(archive.read(row["summary_path"]))
            assert summary["projected_row_count"] == 1
            assert summary["projection_role"] == "derived_summary"
    assert len(calls) == 43


def test_remote_copy_actual_writer_publishes_companion_and_diagnostic_outcome(tmp_path):
    remote = tmp_path / "remote"
    remote.mkdir()
    remote_source = remote / "benchmark_results_case.json"
    remote_source.write_text(json.dumps(_payload(1, large=True)))
    target = _source(tmp_path).parent
    copied = _copy_remote_result_files(remote, target)
    manifest = json.loads((target / "remote_diagnostics" / "result_copy_manifest.json").read_text())
    canonical = [p for p in copied if p.get("canonical")]
    assert canonical
    source = Path(canonical[0]["destination"])
    summary = _read_summary(source)
    assert summary["rows"][0]["primary_error"] == "quality_reference_invalid"
    assert any(row.get("diagnostic_summary", {}).get("status") == "projected" for row in manifest["canonical_files"])


def test_actual_normalizer_projects_loaded_payload_and_summary_is_not_ingested(tmp_path, monkeypatch):
    source = _source(tmp_path)
    source.write_text(json.dumps(_payload()))
    reads = []
    original = Path.read_text
    def counted(path, *args, **kwargs):
        if path == source:
            reads.append(path)
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", counted)
    rows, sources = normalize_benchmark_files(model_id="m1", source_paths=[source.parent], write_diagnostic_summaries=True)
    assert rows
    assert sources[0]["diagnostic_summary"]["status"] == "projected"
    assert reads == [source]
    assert compact.companion_path(source) not in discover_result_files([source.parent])


def test_legacy_readonly_normalization_does_not_add_companion(tmp_path):
    source = _source(tmp_path)
    source.write_text(json.dumps(_payload()))
    normalize_benchmark_files(model_id="m1", source_paths=[source])
    assert not compact.companion_path(source).exists()


def test_real_run_benchmarks_stage_writes_and_registers_normalized_projection(tmp_path):
    # Actual stage; only remote hardware/finalization are fixtures. The payload
    # builder, normalized publication and our diagnostic writer remain real.
    from test_v27538_full_only_performance_matrix import _run_stage, _quality_plan, _quality_profile
    artifacts, metrics, message, status, workflow = _run_stage(
        tmp_path, benchmark_plan=_quality_plan(), profile=_quality_profile(),
        executor_metrics={"remote_dispatched": True, "quality_evidence_count": 2, "expected_full_quality_count": 2},
    )
    assert status == "ok", message
    source = artifacts["normalized_results_json"]
    original = json.loads(source.read_text())
    summary = _read_summary(source)
    assert compact.companion_path(source) in artifacts.values()
    assert metrics["normalized_diagnostic_summary"]["status"] == "projected"
    assert summary["source_schema"] == original["schema"]
    assert summary["source_context"]["model_id"] == original["model_id"]
    assert summary["source_context"]["evaluation_run_id"] == original["evaluation_run_id"]
    assert summary["source_context"]["status"] == "quality_evidence_only_complete"
    assert summary["source_row_count"] == original["result_count"] == 0
    # Normal stage registration uses precisely the returned artifact set.
    workflow._register_artifacts([compact.companion_path(source)], kind="stage_artifact", producer_stage="run_benchmarks", model_id="resnet50")
    assert any("diagnostic_summaries" in entry["path"] for entry in workflow.artifact_index["artifacts"])


def test_normalized_stage_diagnostic_write_failure_keeps_physical_success(tmp_path):
    from test_v27538_full_only_performance_matrix import _run_stage, _quality_plan, _quality_profile
    with mock.patch.object(compact, "write_runtime_companion", return_value={"status": "summary_unavailable", "reason": "companion_write_failed"}):
        _, metrics, message, status, _ = _run_stage(
            tmp_path, benchmark_plan=_quality_plan(), profile=_quality_profile(),
            executor_metrics={"remote_dispatched": True, "quality_evidence_count": 2, "expected_full_quality_count": 2},
        )
    assert status == "ok", message
    assert metrics["normalized_diagnostic_summary"]["reason"] == "companion_write_failed"


def test_exact_aggregate_roles_negative_states_and_source_pointer_are_preserved(tmp_path):
    payload = _payload(1)
    summary = compact.project_runtime_payload(payload, source_path="models/m/benchmark_results/benchmark_results_x.json")
    row = summary["rows"][0]
    assert row["source_pointer"] == "/0"
    for key in ("setup_id", "model_id", "case_id", "precision", "measurement_endpoint", "primary_error", "status", "runtime_ok", "attempted_repetitions"):
        assert row[key] == payload[0][key]
    assert row["quality"] == {k: v for k, v in payload[0]["quality"].items() if k != "metadata"}
    assert row["mini_classification_eval"] == payload[0]["mini_classification_eval"]
    assert row["energy"] == {k: v for k, v in payload[0]["energy"].items() if k != "raw"}
    assert summary["projection_role"] == "derived_summary" and summary["original_bytes"] is False


@pytest.mark.parametrize("payload,reason", [
    ({"schema": "unknown/future", "results": _payload()}, "unknown_schema"),
    ({"plausible": {"rows": _payload()}}, "unknown_rows_schema"),
    ([{"details": {"status": "PASS"}}], "unknown_row_schema"),
    ([17], "unknown_row_schema"),
    ("bytes", "unknown_payload_type"),
])
def test_unknown_schema_never_guesses_nested_metrics(payload, reason):
    summary = compact.project_runtime_payload(payload, source_path="models/m/benchmark_results/benchmark_results_x.json")
    assert summary["status"] == "summary_unavailable"
    assert summary["reason"] == reason
    assert summary["rows"] == []


def test_missing_identity_not_inferred_from_path_and_nonfinite_value_is_unavailable():
    summary = compact.project_runtime_payload([{"status": "ERROR", "latency_ms": float("nan")}], source_path="models/fake/benchmark_results/benchmark_results_setup_fake.json")
    row = summary["rows"][0]
    assert "model_id" not in row and "setup_id" not in row
    assert "setup_id" in row["unavailable_fields"]
    assert "latency_ms" not in row
    assert "/0/latency_ms:nonfinite" in row["omitted_fields"]
    json.loads(compact.summary_bytes(summary))


def test_budget_is_full_parseable_json_and_keeps_bounded_identity_errors():
    payload = _payload(1) * 1000
    summary = compact.project_runtime_payload(payload, source_path="models/m/benchmark_results/benchmark_results_x.json", max_bytes=3000)
    assert summary["status"] == "summary_unavailable" and summary["reason"] == "summary_size_limit"
    assert summary["source_row_count"] == 1000
    assert summary["projected_row_count"] == 0
    assert summary["identity_failure_overview"][0]["primary_error"] == "quality_reference_invalid"
    assert summary["identity_failure_overview_complete"] is False
    assert len(compact.summary_bytes(summary)) <= 3000
    json.loads(compact.summary_bytes(summary))


def test_declared_hash_does_not_claim_new_verification_and_companion_detects_stat_drift(tmp_path):
    source = _source(tmp_path)
    source.write_text(json.dumps(_payload()))
    observation = compact.source_observation(source)
    summary = compact.project_runtime_payload(_payload(), source_path=source.relative_to(tmp_path).as_posix(), source_stat=observation, declared_source_sha256="sha256:" + "a" * 64)
    assert summary["source"]["hash_verification"] == "declared_not_reverified"
    assert compact.companion_matches(summary, source_path=summary["source"]["path"], source_stat=observation)[0]
    assert not compact.companion_matches(summary, source_path="another.json", source_stat=observation)[0]
    assert not compact.companion_matches(summary, source_path=summary["source"]["path"], source_stat={**observation, "observed_size_bytes": observation["observed_size_bytes"] + 1})[0]


def test_same_final_state_reuses_companion_without_reproject_or_rewrite(tmp_path, monkeypatch):
    source = _source(tmp_path)
    payload = _payload()
    source.write_text(json.dumps(payload))
    first = compact.write_runtime_companion(source, payload)
    companion = compact.companion_path(source)
    before = companion.stat().st_mtime_ns
    monkeypatch.setattr(compact, "project_runtime_payload", lambda *a, **kw: pytest.fail("unchanged state reprojected"))
    second = compact.write_runtime_companion(source, payload)
    assert first["status"] == "projected" and second["status"] == "unchanged"
    assert companion.stat().st_mtime_ns == before


def test_source_mutation_during_parse_or_publish_does_not_bind_false_summary(tmp_path, monkeypatch):
    source = _source(tmp_path)
    source.write_text(json.dumps(_payload()))
    before = compact.source_observation(source)
    source.write_text(json.dumps(_payload(2)))
    assert compact.write_runtime_companion(source, _payload(), observed_before=before)["reason"] == "source_changed_during_parse"
    assert not compact.companion_path(source).exists()
    observe = compact.source_observation
    counter = [0]
    def changing(path):
        counter[0] += 1
        result = observe(path)
        if counter[0] > 1:
            result["observed_size_bytes"] += 1
        return result
    monkeypatch.setattr(compact, "source_observation", changing)
    assert compact.write_runtime_companion(source, _payload(2))["reason"] == "source_changed_before_publish"
    assert not compact.companion_path(source).exists()
    assert not list(compact.companion_path(source).parent.glob("*.tmp"))


@pytest.mark.parametrize("mode", ["external", "source_symlink", "parent_symlink", "companion_symlink"])
def test_writer_does_not_follow_external_or_symlink_paths(tmp_path, mode):
    source = _source(tmp_path)
    external = tmp_path / "external.json"
    external.write_text(json.dumps(_payload()))
    if mode == "external":
        source = external
    elif mode == "source_symlink":
        source.symlink_to(external)
    elif mode == "parent_symlink":
        source.parent.rmdir()
        outdir = tmp_path / "other"
        outdir.mkdir()
        source.parent.symlink_to(outdir, target_is_directory=True)
        source.write_text(json.dumps(_payload()))
    else:
        source.write_text(json.dumps(_payload()))
        companion = compact.companion_path(source)
        companion.parent.mkdir()
        companion.symlink_to(external)
    before = external.read_bytes()
    outcome = compact.write_runtime_companion(source, _payload())
    assert outcome["status"] == "summary_unavailable"
    assert external.read_bytes() == before


def test_real_writer_io_failure_is_diagnostic_only_and_atomic(tmp_path, monkeypatch):
    source = _source(tmp_path)
    source.write_text(json.dumps(_payload()))
    monkeypatch.setattr(compact.os, "replace", mock.Mock(side_effect=OSError("disk full")))
    detail = _inspect_canonical_result(source)
    assert detail["parseable"] and detail["nonempty"]
    assert detail["diagnostic_summary"]["reason"] == "companion_write_failed"
    assert not compact.companion_path(source).exists()
    assert not list(compact.companion_path(source).parent.glob("*.tmp"))


@pytest.mark.parametrize("where", ["root", "row", "known_nested", "source", "role"])
def test_companion_cannot_smuggle_raw_trees_with_matching_source_stat(tmp_path, where):
    source = _source(tmp_path)
    source.write_text(json.dumps(_payload()))
    observation = compact.source_observation(source)
    summary = compact.project_runtime_payload(_payload(), source_path=source.relative_to(tmp_path).as_posix(), source_stat=observation)
    if where == "root":
        summary["predictions"] = [1, 2, 3]
    elif where == "row":
        summary["rows"][0]["raw_outputs"] = [1, 2, 3]
    elif where == "known_nested":
        summary["rows"][0]["quality"]["metadata"] = {"raw": "secret"}
    elif where == "source":
        summary["source"]["metadata"] = {"secret": True}
    else:
        summary["projection_role"] = "original_measurement"
    assert compact.companion_matches(summary, source_path=source.relative_to(tmp_path).as_posix(), source_stat=observation) == (False, "companion_projection_invalid")


def test_unknown_normalized_schema_version_is_explicitly_unavailable():
    summary = compact.project_runtime_payload({"schema": "onnx-splitpoint/normalized-benchmark-results", "schema_version": 9000, "results": _payload()}, source_path="models/m/benchmark_results/normalized_results.json")
    assert summary["status"] == "summary_unavailable"
    assert summary["reason"] == "unknown_schema_version"


def test_observed_new_final_state_republishes_once_with_correct_row_identity(tmp_path):
    source = _source(tmp_path)
    source.write_text(json.dumps(_payload()))
    assert compact.write_runtime_companion(source, _payload())["status"] == "projected"
    source.write_text(json.dumps(_payload(1)))
    assert compact.write_runtime_companion(source, _payload(1))["status"] == "projected"
    assert _read_summary(source)["rows"][0]["setup_id"] == "device-1"
    assert compact.write_runtime_companion(source, _payload(1))["status"] == "unchanged"


def test_known_production_quality_shape_keeps_primary_population_and_pending_role():
    # Field structure from available night DeepX Full; these values are synthetic.
    payload = [{"case_id": "full", "classification_top1_accuracy": 0.7,
        "completed_frames": 1000, "completed_work_units": 1000,
        "semantic_validation_metric_gate": {"status": "ok", "top1": 0.7, "labeled_samples": 5000},
        "task_quality_gate": {"status": "pending_central_evaluation", "decision": "pending_central_evaluation",
            "primary": {"metric": "top1_accuracy", "candidate": 0.7, "reference": None, "delta": None,
                        "bootstrap_repetitions_requested": 5000, "bootstrap_skipped_reason": "delegated_to_central_management"},
            "guardrails": {"top5_accuracy": {"candidate": 0.9, "reference": None}},
            "quality_input_request": {"status": "pending_central_evaluation", "reference": {
                "reference_role": "canonical_cpu_ort", "semantic_reference_only": True, "expected_image_ids": ["NEVER_COPY"]}}},
        "task_quality_gates_by_variant": {"full": {"status": "pending_central_evaluation", "primary": {"candidate": 0.7}}}}]
    summary = compact.project_runtime_payload(payload, source_path="models/m/benchmark_results/benchmark_results_x.json", source_stat={"observed_size_bytes": 42, "observed_mtime_ns": 123})
    row = summary["rows"][0]
    assert row["classification_top1_accuracy"] == 0.7
    assert row["completed_frames"] == row["completed_work_units"] == 1000
    assert row["task_quality_gate"]["primary"] == payload[0]["task_quality_gate"]["primary"]
    assert row["task_quality_gate"]["guardrails"] == payload[0]["task_quality_gate"]["guardrails"]
    assert row["task_quality_gates_by_variant"] == payload[0]["task_quality_gates_by_variant"]
    assert row["task_quality_gate"]["quality_input_request"]["reference"]["reference_role"] == "canonical_cpu_ort"
    assert b"NEVER_COPY" not in compact.summary_bytes(summary)
    assert compact.companion_matches(summary, source_path=summary["source"]["path"], source_stat={"observed_size_bytes": 42, "observed_mtime_ns": 123})[0]
    row["task_quality_gate"]["primary"]["predictions"] = [1]
    assert not compact.companion_matches(summary, source_path=summary["source"]["path"], source_stat={"observed_size_bytes": 42, "observed_mtime_ns": 123})[0]


def test_real_normalized_stage_projects_large_measured_row_without_reclassifying_it(tmp_path):
    from test_v27538_full_only_performance_matrix import _run_stage
    original_row = _payload(0, large=True)[0]
    original_row.update(model_id="resnet50", case_id="b053", run_id="hailo8", variant="full")
    artifacts, _, _, _, _ = _run_stage(
        tmp_path,
        benchmark_plan={"runs": [{"id": "hailo8", "backend": "hailo8", "variant": "full", "variants": ["full"]}]},
        profile={"quality_gate": {"statistics": {"execution_location": "central_management"}}},
        executor_metrics={"remote_dispatched": True}, normalized_rows=[original_row],
    )
    source = artifacts["normalized_results_json"]
    normalized = json.loads(source.read_text())
    assert source.stat().st_size > compact.SUMMARY_MAX_BYTES
    summary = _read_summary(source)
    assert summary["source_row_count"] == len(normalized["results"]) == 1
    assert summary["rows"][0]["source_pointer"] == "/results/0"
    for key in ("model_id", "setup_id", "case_id", "variant", "runtime_ok", "total_latency_ms"):
        assert summary["rows"][0][key] == normalized["results"][0][key]
    assert summary["rows"][0]["quality"] == {k:v for k,v in original_row["quality"].items() if k != "metadata"}
    assert b"NEVER_COPY" not in compact.summary_bytes(summary)


@pytest.mark.parametrize("name", ["benchmark_results.json", "results.json"])
def test_actual_legacy_canonical_alias_writer_is_exported_without_broadening_paths(tmp_path, name, monkeypatch):
    from onnx_splitpoint_tool.workflow import debug_pack as packs
    monkeypatch.setattr(packs, "STRUCTURED_RESULT_MAX_FILE_BYTES", 2 * 1024 * 1024)
    from onnx_splitpoint_tool.workflow.debug_pack import (
        create_evaluation_debug_pack, _is_runtime_diagnostic, _is_runtime_summary,
    )
    source = _source(tmp_path, name)
    payload = _payload(1, large=True)
    source.write_text(json.dumps(payload))
    _, sources = normalize_benchmark_files(model_id="m1", source_paths=[source], write_diagnostic_summaries=True)
    assert sources[0]["diagnostic_summary"]["status"] == "projected"
    assert _is_runtime_diagnostic(source.relative_to(tmp_path).as_posix())
    assert _is_runtime_summary(compact.companion_path(source).relative_to(tmp_path).as_posix())
    assert not _is_runtime_diagnostic("models/m1/other/" + name)
    assert not _is_runtime_summary("models/m1/benchmark_results/arbitrary/" + Path(name).stem + ".summary.json")
    result = create_evaluation_debug_pack(tmp_path, tmp_path.parent / (tmp_path.name + ".zip"))
    with zipfile.ZipFile(result["out_zip"]) as archive:
        assert archive.testzip() is None
        inventory = json.loads(archive.read("debug_pack_manifest.json"))["runtime_execution_diagnostics"]
        assert inventory["derived_summary_count"] == 1
        row = inventory["source_coverage"][0]
        assert row["summary_origin"] == "existing_companion"
        summary = json.loads(archive.read(row["summary_path"]))
        assert summary["rows"][0]["primary_error"] == "quality_reference_invalid"
