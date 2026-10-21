"""Selected-run CLI Resume restores the ordinary GUI request and runner checks."""
from __future__ import annotations

import json
from pathlib import Path
import shlex
from types import SimpleNamespace

import pytest

from onnx_splitpoint_tool.workflow import profile_options, run_evaluation
from onnx_splitpoint_tool.workflow.contracts import WorkflowOptions
from onnx_splitpoint_tool.workflow.profile_options import workflow_options_from_archived_run
from onnx_splitpoint_tool.workflow.run_control import stable_resume_options
from onnx_splitpoint_tool.workflow.runner import EvaluationWorkflowRunner
from onnx_splitpoint_tool.workflow.start_snapshot import build_profile_start_snapshot, public_start_snapshot_metadata
from onnx_splitpoint_tool.workflow.artifacts import sha256_json


@pytest.fixture
def archived_run(tmp_path):
    run = tmp_path / "selected_original_run"
    run.mkdir()
    profile = {"name": "original", "model_suite": {"primary": []}, "run_profiles": []}
    snapshot = build_profile_start_snapshot(profile_request="original.yaml", source_profile=profile,
        resolved_profile=profile, profile_id="original", profile_path="original.yaml", profile_source="file",
        runtime_bindings={"runtime_materialized": True})
    options = WorkflowOptions("missing_original.yaml", str(tmp_path), models_root="/original/models",
        execution_mode="generate_and_run", benchmark_warmup=17, benchmark_runs=23,
        remote_working_dir="/original/remote-cache", hailo_build_targets=[], hailo_keep_artifacts=True,
        hardware_setup_ids=["original_setup"], native_producer_deepx_runtime_mode="A4",
        profile_start_snapshot=snapshot, require_fresh_run=True, force_stage=["run_benchmarks"],
        stop_after="run_benchmarks", rerun_generated_only=True, resume_missing_full_quality_only=True,
        force_build_confirmed_backends=("deepx",), force_build_confirmation_source="old_confirmation")
    metadata = public_start_snapshot_metadata(snapshot)
    saved = {"schema": "onnx-splitpoint/evaluation-run-manifest", "run_id": run.name,
        "profile_hash": sha256_json(profile), "profile_start_snapshot": metadata, "options": options.to_dict()}
    for name, value in (("run_manifest.json", saved), ("profile.yaml", profile), ("profile_source.yaml", profile),
                        ("profile_start_snapshot.json", metadata)):
        (run / name).write_text(json.dumps(value))
    return run, options


def test_shared_loader_keeps_all_frozen_options_and_clears_only_invocation_actions(archived_run):
    run, original = archived_run
    before = (run / "run_manifest.json").read_bytes()
    options = workflow_options_from_archived_run(run)
    assert stable_resume_options(options) == stable_resume_options(original)
    assert options.profile_start_snapshot == original.profile_start_snapshot
    assert options.profile == original.profile
    assert options.resume is True and options.run_id == run.name and options.out == str(run.parent)
    assert options.force_build_confirmed_backends == () and options.force_build_confirmation_source == ""
    assert options.require_fresh_run is False and options.force_stage == [] and options.stop_after is None
    assert options.rerun_generated_only is False and options.resume_missing_full_quality_only is False
    assert (run / "run_manifest.json").read_bytes() == before


def test_gui_selected_run_uses_same_loader_without_any_window(archived_run):
    from onnx_splitpoint_tool.gui.app import SplitPointAnalyserGUI
    run, _ = archived_run
    app = SimpleNamespace(var_eval_workflow_last_run_dir=SimpleNamespace(get=lambda: str(run)))
    options = SplitPointAnalyserGUI._eval_workflow_snapshot_options(app, resume_override=True)
    assert options.to_dict() == workflow_options_from_archived_run(run).to_dict()
    preview = shlex.split(SplitPointAnalyserGUI._eval_workflow_command_preview(app, options))
    assert "--profile-driven" in preview and "--resume" in preview
    assert preview[preview.index("--run-id") + 1] == run.name
    assert "--models-root" not in preview and "--require-fresh-run" not in preview


@pytest.mark.parametrize("change", ["missing", "invalid_json", "schema", "run_id", "options", "run_link", "manifest_link"])
def test_shared_loader_rejects_invalid_selected_archive(archived_run, change, tmp_path):
    run, _ = archived_run
    manifest = run / "run_manifest.json"
    saved = json.loads(manifest.read_text())
    if change == "missing":
        manifest.unlink()
    elif change == "invalid_json":
        manifest.write_text("{")
    elif change in {"schema", "run_id", "options"}:
        saved[change] = [] if change == "options" else "wrong"
        manifest.write_text(json.dumps(saved))
    elif change == "run_link":
        link = tmp_path / "linked_run"
        link.symlink_to(run, target_is_directory=True)
        run = link
    else:
        original = run / "original_manifest.json"
        manifest.rename(original)
        manifest.symlink_to(original)
    with pytest.raises(ValueError, match="Resume"):
        workflow_options_from_archived_run(run)


def test_cli_resume_restores_exact_selected_options_without_current_profile_resolution(archived_run, monkeypatch, capsys):
    run, original = archived_run
    newer = run.parent / "newer_run"
    newer.mkdir()
    (newer / "run_manifest.json").write_text("{}")
    observed = {}
    monkeypatch.setattr(profile_options, "load_runtime_profile_snapshot", lambda *args: pytest.fail("Resume re-resolved today's profile"))

    class Runner:
        def __init__(self, options, **kwargs):
            observed["options"] = options

        def run(self):
            return SimpleNamespace(ok=True, status="ok", to_dict=lambda: {"ok": True, "status": "ok"})

    monkeypatch.setattr(run_evaluation, "EvaluationWorkflowRunner", Runner)
    rc = run_evaluation.main(["--profile-driven", "--profile", "missing_original.yaml", "--out", str(run.parent),
        "--resume", "--run-id", run.name, "--json"])
    assert rc == 0, capsys.readouterr().out
    options = observed["options"]
    assert stable_resume_options(options) == stable_resume_options(original)
    assert options.profile_start_snapshot == original.profile_start_snapshot
    assert options.resume is True and options.run_id == run.name and options.out == str(run.parent)
    assert options.force_stage == [] and options.rerun_generated_only is False
    assert options.gui_start_context["kind"] == "cli" and options.gui_start_context["profile_driven"] is True
    assert options.force_build_confirmed_backends == ()


@pytest.mark.parametrize("extra", [[], ["--run-id", ".."], ["--run-id", "../outside"],
    ["--run-id", "selected_original_run", "--models-root", "/override"],
    ["--run-id", "selected_original_run", "--only-model", "other"],
    ["--run-id", "selected_original_run", "--execution-mode", "contracts_only"],
    ["--run-id", "selected_original_run", "--require-fresh-run"],
    ["--run-id", "selected_original_run", "--force-stage", "run_benchmarks"]])
def test_cli_resume_rejects_implicit_target_or_runtime_overrides(archived_run, monkeypatch, capsys, extra):
    run, _ = archived_run
    monkeypatch.setattr(run_evaluation, "EvaluationWorkflowRunner", lambda *args, **kwargs: pytest.fail("Invalid Resume reached runner"))
    before = (run / "run_manifest.json").read_bytes()
    rc = run_evaluation.main(["--profile-driven", "--profile", "missing_original.yaml", "--out", str(run.parent),
        "--resume", "--json", *extra])
    assert rc == 1 and json.loads(capsys.readouterr().out)["technical_status"] == "failed"
    assert (run / "run_manifest.json").read_bytes() == before


def test_restored_options_still_use_runner_archived_snapshot_validation(archived_run):
    run, _ = archived_run
    options = workflow_options_from_archived_run(run)
    runner = EvaluationWorkflowRunner(options)
    runner._load_profile()
    runner.run_dir, runner.manifest_path = run, run / "run_manifest.json"
    runner._validate_resume_profile_snapshot()
    (run / "profile.yaml").write_text(json.dumps({"name": "changed"}))
    with pytest.raises(ValueError, match="archived profile.yaml"):
        runner._validate_resume_profile_snapshot()
