from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from onnx_splitpoint_tool.workflow import debug_pack as debug_pack_module
from onnx_splitpoint_tool.workflow.debug_pack import (
    create_evaluation_debug_pack,
)
from onnx_splitpoint_tool.workflow.debug_pack_policy import (
    RANKING_AUDIT_ANALYSIS_FILES,
    RANKING_AUDIT_BENCHMARK_FILES,
    RANKING_AUDIT_MAX_FILE_BYTES,
    RANKING_AUDIT_REPORT_FILES,
)


def _identified_run(root: Path, *, audit_models: tuple[str, ...] = ()) -> Path:
    root.mkdir(parents=True)
    (root / "run_manifest.json").write_text(
        json.dumps({
            "schema": "onnx-splitpoint/evaluation-run-manifest",
            "schema_version": 1,
            "run_id": root.name,
            "status": "failed",
            "tool_version": "recorded-tool",
            "workflow_version": "recorded-workflow",
        }),
        encoding="utf-8",
    )
    model_rows = "\n".join(f"  - id: {model}\n" for model in audit_models)
    selection = (
        "selection_policy:\n"
        "  selection_strategy: score_independent_audit\n"
        "  score_independent_audit_enabled: true\n"
        if audit_models else ""
    )
    (root / "profile.yaml").write_text(
        selection + "model_suite:\n  primary:\n" + model_rows,
        encoding="utf-8",
    )
    return root


def _manifest(archive: zipfile.ZipFile) -> dict:
    return json.loads(archive.read("debug_pack_manifest.json"))


def _sha(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def test_full_root_log_and_source_identity_are_required_hashed_once(
    tmp_path: Path,
) -> None:
    run = _identified_run(tmp_path / "failed_run")
    log_bytes = (b"complete workflow diagnostic\n" * 100_000) + b"THE-END\n"
    (run / "evaluation_workflow.log").write_bytes(log_bytes)
    (run / "evaluation_workflow_tail.log").write_bytes(b"stale fallback\n")
    output = tmp_path / "compact.zip"

    result = create_evaluation_debug_pack(
        run, output, max_small_file_bytes=64
    )

    assert result["archive_verification"] == "verified"
    with zipfile.ZipFile(output) as archive:
        assert archive.namelist().count("evaluation_workflow.log") == 1
        assert "evaluation_workflow_tail.log" not in archive.namelist()
        assert archive.read("evaluation_workflow.log") == log_bytes
        assert archive.namelist().count("pack_source_identity.json") == 1
        source = json.loads(archive.read("pack_source_identity.json"))
        assert source["run_id"] == run.name
        assert source["tool_version_recorded"] == "recorded-tool"
        assert source["workflow_version_recorded"] == "recorded-workflow"
        assert source["run_status_recorded"] == "failed"
        assert source["selection_policy"] == "explicit_evaluation_run_argument"
        manifest = _manifest(archive)
        section = manifest["main_workflow_log"]
        assert section["complete"] is True
        assert section["size_bytes"] == len(log_bytes)
        assert section["sha256"] == _sha(log_bytes)

    sidecar = json.loads(
        output.with_name(output.name + ".manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert "evaluation_workflow.log" in sidecar["required_members"]
    assert "pack_source_identity.json" in sidecar["required_members"]


def test_bounded_tail_is_used_only_when_full_root_log_is_missing(
    tmp_path: Path,
) -> None:
    run = _identified_run(tmp_path / "old_failed_run")
    tail = (b"old workflow line\n" * 1000) + b"TAIL-END\n"
    (run / "evaluation_workflow_tail.log").write_bytes(tail)
    output = tmp_path / "tail-fallback.zip"

    create_evaluation_debug_pack(run, output, tail_bytes=1024)

    with zipfile.ZipFile(output) as archive:
        names = archive.namelist()
        assert "evaluation_workflow.log" not in names
        assert names.count("evaluation_workflow_tail.log") == 1
        archived_tail = archive.read("evaluation_workflow_tail.log")
        assert archived_tail.endswith(b"TAIL-END\n")
        assert len(archived_tail) < len(tail)
        section = _manifest(archive)["main_workflow_log"]
        assert section["completeness"] == "tail_only"
        assert section["complete"] is False
    sidecar = json.loads(
        output.with_name(output.name + ".manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert "evaluation_workflow_tail.log" in sidecar["required_members"]


def _write_complete_audit(run: Path, model_id: str) -> set[str]:
    members: set[str] = set()
    for name in RANKING_AUDIT_ANALYSIS_FILES:
        path = run / "models" / model_id / "analysis" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"enabled": True} if name == "audit_plan.json" else {"name": name}
        path.write_text(json.dumps(payload), encoding="utf-8")
        members.add(path.relative_to(run).as_posix())
    for name in RANKING_AUDIT_BENCHMARK_FILES:
        path = run / "models" / model_id / "benchmark_set" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"name": name}), encoding="utf-8")
        members.add(path.relative_to(run).as_posix())
    for relative in RANKING_AUDIT_REPORT_FILES:
        path = run / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"path": relative}), encoding="utf-8")
        members.add(relative)
    return members


def test_not_requested_scientific_reports_do_not_create_audit_intent(
    tmp_path: Path,
) -> None:
    run = _identified_run(tmp_path / "non_audit_run")
    (run / "evaluation_workflow.log").write_text(
        "complete\n", encoding="utf-8"
    )
    plan = run / "models/model_a/analysis/audit_plan.json"
    plan.parent.mkdir(parents=True)
    plan.write_text(
        json.dumps({"enabled": False, "status": "not_requested"}),
        encoding="utf-8",
    )
    report_payloads: dict[str, bytes] = {}
    for relative in RANKING_AUDIT_REPORT_FILES:
        path = run / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = (
            json.dumps({"status": "not_requested", "path": relative}) + "\n"
        ).encode("utf-8")
        path.write_bytes(payload)
        report_payloads[relative] = payload
    output = tmp_path / "non-audit.zip"

    create_evaluation_debug_pack(run, output)

    with zipfile.ZipFile(output) as archive:
        names = set(archive.namelist())
        assert set(report_payloads) <= names
        manifest = _manifest(archive)
        audit = manifest["ranking_audit_evidence"]
        assert audit["enabled"] is False
        assert audit["explicit_report_present"] is True
        assert audit["expected_members"] == []
        assert audit["missing_source_members"] == []
        assert audit["required_archive_members"] == []
        assert audit["complete"] is True
        assert manifest["complete"] is True
        records = {row["path"]: row for row in manifest["files"]}
        for relative, payload in report_payloads.items():
            assert archive.read(relative) == payload
            assert records[relative]["sha256"] == _sha(payload)


def test_profile_requested_partial_audit_is_incomplete_at_both_levels(
    tmp_path: Path,
) -> None:
    run = _identified_run(
        tmp_path / "profile_partial_audit", audit_models=("model_a",)
    )
    (run / "evaluation_workflow.log").write_text(
        "incomplete audit\n", encoding="utf-8"
    )
    plan = run / "models/model_a/analysis/audit_plan.json"
    plan.parent.mkdir(parents=True)
    plan.write_text('{"enabled": true}', encoding="utf-8")
    output = tmp_path / "profile-partial-audit.zip"

    create_evaluation_debug_pack(run, output)

    with zipfile.ZipFile(output) as archive:
        manifest = _manifest(archive)
        audit = manifest["ranking_audit_evidence"]
        assert audit["enabled"] is True
        assert audit["profile_requested"] is True
        assert audit["missing_source_members"]
        assert audit["complete"] is False
        assert manifest["complete"] is False


def test_plan_only_audit_intent_remains_fail_closed(tmp_path: Path) -> None:
    run = _identified_run(tmp_path / "plan_only_partial_audit")
    (run / "evaluation_workflow.log").write_text(
        "incomplete plan-only audit\n", encoding="utf-8"
    )
    plan = run / "models/model_a/analysis/audit_plan.json"
    plan.parent.mkdir(parents=True)
    plan.write_text('{"enabled": true}', encoding="utf-8")
    sibling = run / "models/model_b/analysis"
    sibling.mkdir(parents=True)
    output = tmp_path / "plan-only-partial-audit.zip"

    create_evaluation_debug_pack(run, output)

    with zipfile.ZipFile(output) as archive:
        manifest = _manifest(archive)
        audit = manifest["ranking_audit_evidence"]
        assert audit["enabled"] is True
        assert audit["profile_requested"] is False
        assert audit["enabled_plan_model_ids"] == ["model_a"]
        assert audit["model_ids"] == ["model_a", "model_b"]
        assert audit["missing_source_members"]
        assert any(
            path.startswith("models/model_b/")
            for path in audit["missing_source_members"]
        )
        assert audit["complete"] is False
        assert manifest["complete"] is False


def test_exact_audit_evidence_ignores_generic_cap_and_missing_source_is_visible(
    tmp_path: Path,
) -> None:
    run = _identified_run(tmp_path / "audit_run", audit_models=("model_a", "model_b"))
    (run / "evaluation_workflow.log").write_text("failed after model_a\n", encoding="utf-8")
    expected_present = _write_complete_audit(run, "model_a")
    large_plan = run / "models/model_a/analysis/final_candidate_plan.json"
    large_plan.write_text(
        json.dumps({"candidates": ["x" * 1024] * 4096}),
        encoding="utf-8",
    )
    output = tmp_path / "audit.zip"

    create_evaluation_debug_pack(run, output, max_small_file_bytes=128)

    with zipfile.ZipFile(output) as archive:
        names = set(archive.namelist())
        assert expected_present <= names
        assert archive.read(large_plan.relative_to(run).as_posix()) == large_plan.read_bytes()
        manifest = _manifest(archive)
        audit = manifest["ranking_audit_evidence"]
        assert audit["enabled"] is True
        assert audit["complete"] is False
        assert any(path.startswith("models/model_b/") for path in audit["missing_source_members"])
        record = next(
            row for row in manifest["files"]
            if row["path"] == large_plan.relative_to(run).as_posix()
        )
        assert record["size_bytes"] == large_plan.stat().st_size
        assert record["sha256"] == _sha(large_plan.read_bytes())
    sidecar = json.loads(
        output.with_name(output.name + ".manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert expected_present <= set(sidecar["required_members"])


def test_existing_oversized_audit_source_prevents_publication(tmp_path: Path) -> None:
    run = _identified_run(tmp_path / "audit_run", audit_models=("model_a",))
    (run / "evaluation_workflow.log").write_text("log\n", encoding="utf-8")
    plan = run / "models/model_a/analysis/audit_plan.json"
    plan.parent.mkdir(parents=True)
    plan.write_text('{"enabled": true}', encoding="utf-8")
    oversized = run / "models/model_a/analysis/final_candidate_plan.json"
    with oversized.open("wb") as handle:
        handle.truncate(RANKING_AUDIT_MAX_FILE_BYTES + 1)
    output = tmp_path / "must-not-exist.zip"

    with pytest.raises(RuntimeError, match="ranking_audit_evidence_not_publishable"):
        create_evaluation_debug_pack(run, output)

    assert not output.exists()
    assert not output.with_name(output.name + ".manifest.json").exists()


def test_existing_unsafe_audit_source_prevents_publication(tmp_path: Path) -> None:
    run = _identified_run(tmp_path / "audit_run", audit_models=("model_a",))
    (run / "evaluation_workflow.log").write_text("log\n", encoding="utf-8")
    outside = tmp_path / "outside-plan.json"
    outside.write_text('{"enabled": true}', encoding="utf-8")
    plan = run / "models/model_a/analysis/audit_plan.json"
    plan.parent.mkdir(parents=True)
    plan.symlink_to(outside)
    output = tmp_path / "must-not-exist.zip"

    with pytest.raises(RuntimeError, match="existing_source_error"):
        create_evaluation_debug_pack(run, output)

    assert not output.exists()


def test_compact_default_excludes_binary_figures_mirrors_and_replay_bodies(
    tmp_path: Path,
) -> None:
    run = _identified_run(tmp_path / "compact_run")
    (run / "evaluation_workflow.log").write_text("complete\n", encoding="utf-8")
    request = run / "models/m/benchmark_results/quality_inputs/task_quality_inputs/full_request.json"
    request.parent.mkdir(parents=True)
    candidate = request.with_name("full_candidate.json")
    candidate.write_text(json.dumps({"payload": "x" * (3 * 1024 * 1024)}), encoding="utf-8")
    reference = run / "quality_management/references/m/canonical_cpu_reference.json"
    annotations = run / "quality_management/references/m/annotations.json"
    reference.parent.mkdir(parents=True)
    reference.write_text('{"records": []}', encoding="utf-8")
    annotations.write_text('{"images": []}', encoding="utf-8")
    request.write_text(json.dumps({
        "candidate": {"path": candidate.name, "sha256": _sha(candidate.read_bytes())},
        "reference": {"path": reference.relative_to(run).as_posix()},
        "annotations": {"path": annotations.relative_to(run).as_posix()},
    }), encoding="utf-8")
    summary = run / "quality_management/central_quality_summary.json"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps({
        "results": [{
            "source_request": request.relative_to(run).as_posix(),
            "source_request_sha256": _sha(request.read_bytes()),
        }],
    }), encoding="utf-8")

    excluded = {
        "reports/tensor.bin": b"tensor",
        "reports/figure.png": b"png",
        "reports/paper.pdf": b"pdf",
        "reports/lean_bundle/summary.json": b"{}",
        "native_producers/h8/resources/descriptor.json": b"{}",
        "native_producers/h8/benchmark_set/final_candidate_plan.json": b"{}",
    }
    for relative, payload in excluded.items():
        path = run / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

    output = tmp_path / "compact.zip"
    create_evaluation_debug_pack(run, output)

    with zipfile.ZipFile(output) as archive:
        names = set(archive.namelist())
        assert request.relative_to(run).as_posix() in names
        assert summary.relative_to(run).as_posix() in names
        assert candidate.relative_to(run).as_posix() not in names
        assert reference.relative_to(run).as_posix() not in names
        assert annotations.relative_to(run).as_posix() not in names
        assert not (set(excluded) & names)
        central = _manifest(archive)["central_quality_replay_inputs"]
        assert central["mode"] == "request_descriptors_and_declared_decoded_predictions"
        assert central["replay_payloads_included"] is False
        assert central["archived_members"] == [request.relative_to(run).as_posix()]
        assert all(row["archived"] is False for row in central["referenced_replay_payloads"])


def test_compact_pack_keeps_hashed_per_model_validation_summaries(
    tmp_path: Path,
) -> None:
    run = _identified_run(tmp_path / "validation_summary_run")
    (run / "evaluation_workflow.log").write_text(
        "complete\n", encoding="utf-8"
    )
    expected: dict[str, bytes] = {}
    for model_id, invalid_count in (("resnet50", 13), ("yolo26s", 2)):
        path = (
            run
            / "models"
            / model_id
            / "validation"
            / "validation_summary.json"
        )
        path.parent.mkdir(parents=True)
        payload = (
            json.dumps({
                "model_id": model_id,
                "validated_result_count": 20,
                "invalid_result_count": invalid_count,
                "invalid_rows": [{"case_id": "b001", "reason": "test"}],
            }, sort_keys=True)
            + "\n"
        ).encode("utf-8")
        path.write_bytes(payload)
        expected[path.relative_to(run).as_posix()] = payload

    unrelated = run / "models/resnet50/validation/raw_predictions.json"
    unrelated.write_text('{"payload": "not compact evidence"}\n', encoding="utf-8")
    output = tmp_path / "validation-summaries.zip"

    create_evaluation_debug_pack(run, output)

    with zipfile.ZipFile(output) as archive:
        names = set(archive.namelist())
        assert set(expected) <= names
        assert unrelated.relative_to(run).as_posix() not in names
        manifest = _manifest(archive)
        section = manifest["model_validation_summaries"]
        assert section["present_source_members"] == sorted(expected)
        assert section["admitted_source_members"] == sorted(expected)
        assert section["archived_members"] == sorted(expected)
        assert section["omitted_source_members"] == []
        assert section["all_admitted_members_archived"] is True
        assert section["all_archived_members_sha256_recorded"] is True
        records = {row["path"]: row for row in manifest["files"]}
        for relative, payload in expected.items():
            assert archive.read(relative) == payload
            assert records[relative]["diagnostic_kind"] == (
                "model_validation_summary"
            )
            assert records[relative]["sha256"] == _sha(payload)


def test_model_validation_summaries_obey_file_and_total_limits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _identified_run(tmp_path / "bounded_validation_summary_run")
    (run / "evaluation_workflow.log").write_text(
        "complete\n", encoding="utf-8"
    )
    for model_id, size in (("model_a", 80), ("model_b", 80), ("model_c", 120)):
        path = (
            run
            / "models"
            / model_id
            / "validation"
            / "validation_summary.json"
        )
        path.parent.mkdir(parents=True)
        path.write_bytes(b"x" * size)

    monkeypatch.setattr(
        debug_pack_module,
        "MODEL_VALIDATION_SUMMARY_MAX_TOTAL_BYTES",
        100,
    )
    monkeypatch.setattr(debug_pack_module, "MODEL_VALIDATION_SUMMARY_MAX_FILE_BYTES", 100)
    output = tmp_path / "bounded-validation-summaries.zip"

    create_evaluation_debug_pack(
        run,
        output,
        max_small_file_bytes=100,
    )

    with zipfile.ZipFile(output) as archive:
        names = set(archive.namelist())
        model_a = "models/model_a/validation/validation_summary.json"
        model_b = "models/model_b/validation/validation_summary.json"
        model_c = "models/model_c/validation/validation_summary.json"
        assert model_a in names
        assert model_b not in names
        assert model_c not in names
        manifest = _manifest(archive)
        section = manifest["model_validation_summaries"]
        assert section["archived_members"] == [model_a]
        omitted = {row["path"]: row for row in section["omitted_source_members"]}
        assert omitted[model_b]["reason"] == (
            "model validation summary total limit exceeded"
        )
        assert omitted[model_c]["reason"] == (
            "model validation summary exceeds compact file limit"
        )
        assert section["archived_total_bytes"] == 80
        assert section["max_file_bytes"] == 100
        assert section["max_total_bytes"] == 100
        skipped = [
            row for row in manifest["skipped_examples"]
            if row.get("path") in {model_b, model_c}
        ]
        assert [row["path"] for row in skipped].count(model_b) == 1
        assert [row["path"] for row in skipped].count(model_c) == 1


def test_compact_pack_keeps_only_exact_backend_artifact_diagnostics(
    tmp_path: Path,
) -> None:
    run = _identified_run(tmp_path / "backend_artifact_run")
    (run / "evaluation_workflow.log").write_text(
        "backend build failed\n", encoding="utf-8"
    )
    expected: dict[str, bytes] = {
        "models/resnet50/benchmark_set/backend_artifact_decisions.json": (
            b'{"deepx":{"decision":"reuse"}}\n'
        ),
        "models/resnet50/benchmark_set/artifact_reuse_manifest.json": (
            b'{"deepx":{"cache_hit":true}}\n'
        ),
        "models/resnet50/benchmark_set/legacy_suite/"
        "deepx_prefetch_v60s.json": b'{"status":"ok"}\n',
        "models/resnet50/stages/build_backend_artifacts/stage_result.json": (
            b'{"status":"failed","reason":"compiler gate"}\n'
        ),
        # Direct/formal generators may materialise the same narrow receipt at
        # the benchmark-set root rather than below legacy_suite.
        "models/direct_model/benchmark_set/deepx_prefetch_v60s.json": (
            b'{"status":"cache_hit"}\n'
        ),
    }
    for relative, payload in expected.items():
        path = run / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    # The failing remote benchmark stage is compact control-plane evidence and
    # remains available even though the model tree is not broadly scanned.
    remote_stage = "models/resnet50/stages/run_benchmarks/stage_result.json"
    (run / remote_stage).parent.mkdir(parents=True, exist_ok=True)
    (run / remote_stage).write_bytes(b'{"status":"failed_to_dispatch"}\n')

    excluded = {
        "models/resnet50/benchmark_set/legacy_suite/model.dxnn": b"binary",
        "models/resnet50/benchmark_set/legacy_suite/benchmark_set.json": b"{}",
        "models/resnet50/other/backend_artifact_decisions.json": b"{}",
    }
    for relative, payload in excluded.items():
        path = run / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

    output = tmp_path / "backend-artifacts.zip"
    create_evaluation_debug_pack(run, output)

    with zipfile.ZipFile(output) as archive:
        names = set(archive.namelist())
        assert set(expected) <= names
        assert remote_stage in names
        assert not (set(excluded) & names)
        manifest = _manifest(archive)
        section = manifest["backend_artifact_diagnostics"]
        assert section["present_source_members"] == sorted(expected)
        assert section["admitted_source_members"] == sorted(expected)
        assert section["archived_members"] == sorted(expected)
        assert section["omitted_source_members"] == []
        assert section["all_admitted_members_archived"] is True
        assert section["all_archived_members_sha256_recorded"] is True
        records = {row["path"]: row for row in manifest["files"]}
        for relative, payload in expected.items():
            assert archive.read(relative) == payload
            assert records[relative]["diagnostic_kind"] == (
                "backend_artifact_diagnostic"
            )
            assert records[relative]["sha256"] == _sha(payload)
        assert records[remote_stage]["diagnostic_kind"] == (
            "remote_execution_failure_diagnostic"
        )

    sidecar = json.loads(
        output.with_name(output.name + ".manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert set(expected) <= set(sidecar["required_members"])


def test_backend_artifact_diagnostics_obey_file_and_total_limits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _identified_run(tmp_path / "bounded_backend_artifact_run")
    (run / "evaluation_workflow.log").write_text(
        "backend build failed\n", encoding="utf-8"
    )
    relative_a = (
        "models/model_a/benchmark_set/backend_artifact_decisions.json"
    )
    relative_b = (
        "models/model_b/benchmark_set/artifact_reuse_manifest.json"
    )
    relative_c = (
        "models/model_c/stages/build_backend_artifacts/stage_result.json"
    )
    for relative, size in (
        (relative_a, 80),
        (relative_b, 80),
        (relative_c, 120),
    ):
        path = run / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x" * size)

    monkeypatch.setattr(
        debug_pack_module,
        "BACKEND_ARTIFACT_DIAGNOSTIC_MAX_TOTAL_BYTES",
        100,
    )
    monkeypatch.setattr(debug_pack_module, "BACKEND_ARTIFACT_DIAGNOSTIC_MAX_FILE_BYTES", 100)
    output = tmp_path / "bounded-backend-artifacts.zip"
    create_evaluation_debug_pack(
        run,
        output,
        max_small_file_bytes=100,
    )

    with zipfile.ZipFile(output) as archive:
        names = set(archive.namelist())
        assert relative_a in names
        assert relative_b not in names
        assert relative_c not in names
        manifest = _manifest(archive)
        section = manifest["backend_artifact_diagnostics"]
        assert section["archived_members"] == [relative_a]
        omitted = {
            row["path"]: row for row in section["omitted_source_members"]
        }
        assert omitted[relative_b]["reason"] == (
            "backend artifact diagnostic total limit exceeded"
        )
        assert omitted[relative_c]["reason"] == (
            "backend artifact diagnostic exceeds compact file limit"
        )
        assert section["archived_total_bytes"] == 80
        assert section["max_file_bytes"] == 100
        assert section["max_total_bytes"] == 100
        skipped = [
            row for row in manifest["skipped_examples"]
            if row.get("path") in {relative_b, relative_c}
        ]
        assert [row["path"] for row in skipped].count(relative_b) == 1
        assert [row["path"] for row in skipped].count(relative_c) == 1


def test_explicit_window_probe_raw_payloads_have_a_total_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _identified_run(tmp_path / "probe_run")
    (run / "evaluation_workflow.log").write_text(
        "complete\n", encoding="utf-8"
    )
    with (run / "profile.yaml").open("a", encoding="utf-8") as handle:
        handle.write(
            "native_producers:\n"
            "  energy:\n"
            "    window_method_validation_probe:\n"
            "      include_raw_parquet: true\n"
        )
    probe = run / "reports" / "window_method_validation_probe"
    probe.mkdir(parents=True)
    (probe / "a.parquet").write_bytes(b"a" * 80)
    (probe / "b.parquet").write_bytes(b"b" * 80)
    monkeypatch.setattr(debug_pack_module, "PROBE_MAX_TOTAL_BYTES", 100)
    output = tmp_path / "probe.zip"

    create_evaluation_debug_pack(run, output)

    with zipfile.ZipFile(output) as archive:
        raw_names = {
            name for name in archive.namelist()
            if name.startswith("reports/window_method_validation_probe/")
            and name.endswith(".parquet")
        }
        assert len(raw_names) == 1
        manifest = _manifest(archive)
        probe_section = manifest["window_method_validation_probe"]
        assert probe_section["raw_parquet_count"] == 1
        assert probe_section["raw_parquet_total_bytes"] == 80
        assert probe_section["raw_parquet_max_total_bytes"] == 100
        assert any(
            row.get("reason") == "window probe raw total limit exceeded"
            for row in manifest["skipped_examples"]
        )


def test_known_byte_identical_aliases_keep_hashed_canonical_evidence(
    tmp_path: Path,
) -> None:
    run = _identified_run(tmp_path / "alias_run")
    log_payload = b"complete workflow log\nTHE-END\n"
    (run / "evaluation_workflow.log").write_bytes(log_payload)

    ranking_path = run / "reports/scientific/native_ranking_audit.csv"
    ranking_path.parent.mkdir(parents=True)
    ranking_payload = b"setup_id,spearman\nsetup-a,1.0\n"
    ranking_path.write_bytes(ranking_payload)

    reports = run / "reports"
    summary_payload = b'{"schema":"native-summary","rows":[1,2,3]}\n'
    (reports / "native_producer_summary.json").write_bytes(summary_payload)
    (reports / "native_producer_combined_summary.json").write_bytes(
        summary_payload
    )

    binding_payload = b'{"setup_id":"setup-a","bindings":{"b024":{}}}\n'
    binding_canonical = (
        reports
        / "native_quality_first/setup-a/native_split_quality_binding_set.json"
    )
    binding_canonical.parent.mkdir(parents=True)
    binding_canonical.write_bytes(binding_payload)
    binding_alias = (
        run
        / "native_producers/hailo8/quality_first"
        / "native_split_quality_binding_set.json"
    )
    binding_alias.parent.mkdir(parents=True)
    binding_alias.write_bytes(binding_payload)

    output = tmp_path / "aliases.zip"
    create_evaluation_debug_pack(run, output)

    summary_canonical_name = "reports/native_producer_summary.json"
    summary_alias_name = "reports/native_producer_combined_summary.json"
    binding_canonical_name = binding_canonical.relative_to(run).as_posix()
    binding_alias_name = binding_alias.relative_to(run).as_posix()
    with zipfile.ZipFile(output) as archive:
        names = set(archive.namelist())
        assert summary_canonical_name in names
        assert binding_canonical_name in names
        assert summary_alias_name not in names
        assert binding_alias_name not in names
        assert archive.read(summary_canonical_name) == summary_payload
        assert archive.read(binding_canonical_name) == binding_payload
        assert archive.read("evaluation_workflow.log") == log_payload
        assert archive.read(ranking_path.relative_to(run).as_posix()) == ranking_payload

        manifest = _manifest(archive)
        aliases = manifest["exact_duplicate_aliases"]
        assert aliases["omitted_alias_count"] == 2
        assert aliases["all_canonical_members_archived"] is True
        assert aliases["all_canonical_hashes_verified"] is True
        records = {
            row["omitted_path"]: row for row in aliases["omitted_aliases"]
        }
        for alias_name, canonical_name, payload in (
            (summary_alias_name, summary_canonical_name, summary_payload),
            (binding_alias_name, binding_canonical_name, binding_payload),
        ):
            record = records[alias_name]
            assert record["canonical_path"] == canonical_name
            assert record["retained_path"] == canonical_name
            assert record["sha256"] == _sha(payload)
            assert record["canonical_sha256"] == _sha(payload)
            assert record["retained_sha256"] == _sha(payload)
            assert record["canonical_archive_sha256"] == _sha(payload)
            assert record["canonical_hash_verified"] is True
            file_record = next(
                row for row in manifest["files"]
                if row["path"] == canonical_name
            )
            assert file_record["sha256"] == _sha(payload)

    sidecar = json.loads(
        output.with_name(output.name + ".manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary_canonical_name in sidecar["required_members"]
    assert binding_canonical_name in sidecar["required_members"]


def test_known_alias_paths_with_different_bytes_are_both_retained(
    tmp_path: Path,
) -> None:
    run = _identified_run(tmp_path / "different_alias_run")
    (run / "evaluation_workflow.log").write_text("complete\n", encoding="utf-8")
    reports = run / "reports"
    reports.mkdir(parents=True)
    (reports / "native_producer_summary.json").write_bytes(b'{"value":"a"}\n')
    (reports / "native_producer_combined_summary.json").write_bytes(
        b'{"value":"b"}\n'
    )

    binding_canonical = (
        reports
        / "native_quality_first/setup-a/native_split_quality_binding_set.json"
    )
    binding_canonical.parent.mkdir(parents=True)
    binding_canonical.write_bytes(b'{"value":"a"}\n')
    binding_alias = (
        run
        / "native_producers/hailo8/quality_first"
        / "native_split_quality_binding_set.json"
    )
    binding_alias.parent.mkdir(parents=True)
    binding_alias.write_bytes(b'{"value":"b"}\n')

    output = tmp_path / "different-aliases.zip"
    create_evaluation_debug_pack(run, output)

    with zipfile.ZipFile(output) as archive:
        names = set(archive.namelist())
        assert "reports/native_producer_summary.json" in names
        assert "reports/native_producer_combined_summary.json" in names
        assert binding_canonical.relative_to(run).as_posix() in names
        assert binding_alias.relative_to(run).as_posix() in names
        aliases = _manifest(archive)["exact_duplicate_aliases"]
        assert aliases["omitted_alias_count"] == 0
        assert aliases["omitted_aliases"] == []
        assert aliases["all_canonical_members_archived"] is True
        assert aliases["all_canonical_hashes_verified"] is True
