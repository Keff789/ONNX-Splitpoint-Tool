"""Synthetic consent failures; real runner lock/start and CLI boundaries."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from onnx_splitpoint_tool.force_build_admission import (
    ForceBuildConsentRequired, force_build_backends, require_force_build_consent,
)
from onnx_splitpoint_tool.build_dispatch_policy import ProductiveForceBuildDisabled
from onnx_splitpoint_tool.workflow.contracts import WorkflowOptions
from onnx_splitpoint_tool.workflow.run_control import stable_resume_options
from onnx_splitpoint_tool.workflow.runner import EvaluationWorkflowRunner
from onnx_splitpoint_tool.workflow.start_snapshot import (
    build_profile_start_snapshot, validate_profile_start_snapshot,
)
from _v27930_terminal_lifecycle_fixture import TerminalLifecycleRunner, options_for


def _profile(hailo=False, deepx=False):
    return {"name": "synthetic-force", "hailo_build": {"force_build": hailo},
            "deepx_build": {"force_build": deepx}}


@pytest.mark.parametrize("value", ["false", "true", "yes", 0, 1, None, [], [False]])
@pytest.mark.parametrize("backend", ["hailo", "deepx"])
def test_t33_10_profile_loader_and_runtime_reject_invalid_force(tmp_path, value, backend):
    from onnx_splitpoint_tool.benchmark.evaluation_profiles import load_evaluation_profile
    from onnx_splitpoint_tool.deepx.env_status import profile_compiler_configuration
    profile = _profile()
    profile[f"{backend}_build"]["force_build"] = value
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(profile))
    expected = rf"config_boolean_invalid:{backend}_build.force_build"
    with pytest.raises(ValueError, match=expected):
        load_evaluation_profile(path, validate=False)
    with pytest.raises(ValueError, match=expected):
        force_build_backends(profile)
    with pytest.raises(ValueError, match=expected):
        profile_compiler_configuration(profile)
    snapshot = build_profile_start_snapshot(
        profile_request=str(path), source_profile=profile, resolved_profile=profile,
        profile_id="invalid", profile_path=str(path), profile_source="file",
    )
    with pytest.raises(ValueError, match=expected):
        validate_profile_start_snapshot(snapshot)


def test_t33_18_consent_is_backend_specific_and_does_not_enable_force():
    profile = _profile(True, True)
    original = copy.deepcopy(profile)
    with pytest.raises(ForceBuildConsentRequired) as error:
        require_force_build_consent(profile, confirmed_backends=("hailo",))
    assert error.value.backends == ("deepx",)
    assert "DeepX" in str(error.value)
    assert "--confirm-force-build deepx" in str(error.value)
    record = require_force_build_consent(profile, confirmed_backends=("hailo", "deepx"))
    assert record["confirmed_backends"] == ["hailo", "deepx"]
    assert profile == original
    inactive = require_force_build_consent(_profile(), confirmed_backends=("hailo", "deepx"))
    assert inactive["required_backends"] == []
    assert inactive["compatible_cache_hits_intentionally_bypassed"] is False


@pytest.mark.parametrize("backend", ["hailo", "deepx"])
def test_t33_18_real_start_rejects_before_any_dispatch_and_accepts_once(tmp_path, backend):
    # v34 productive policy supersedes v33's confirmation opt-in. The original
    # v33 test was run unchanged on the source baseline and kept in its JUnit.
    from onnx_splitpoint_tool.build_dispatch_policy import ProductiveForceBuildDisabled
    dispatches = []

    class Runner(TerminalLifecycleRunner):
        def _load_profile(self):
            super()._load_profile()
            self.profile_payload.update(_profile(hailo=backend == "hailo", deepx=backend == "deepx"))

        def _run_model_pipeline_with_cache_preflight(self, rows):
            dispatches.extend(force_build_backends(self.profile_payload))

    options = options_for(tmp_path)
    denied = Runner(options)
    with pytest.raises(ProductiveForceBuildDisabled):
        denied.run()
    assert dispatches == []
    assert not denied.run_dir.exists()
    assert denied._run_lock is None

    options.force_build_confirmed_backends = (backend,)
    options.force_build_confirmation_source = "test_explicit_confirmation"
    accepted = Runner(options)
    with pytest.raises(ProductiveForceBuildDisabled):
        accepted.run()
    assert dispatches == []
    assert not accepted.run_dir.exists()
    assert accepted._run_lock is None
    assert options.force_build_confirmed_backends == ()
    options.run_id = "second-start"
    with pytest.raises(ProductiveForceBuildDisabled):
        Runner(options).run()
    assert dispatches == []


def test_t33_20_real_resume_needs_new_consent_and_preserves_all_creation_bytes(tmp_path):
    from test_v269e_start_snapshot_backfill import _source_profile, _start_snapshot, _write_resume_fixture
    source = _source_profile(native=False, energy=False)
    source["execution_preset"]["snapshot"]["build"]["hailo"]["force_build"] = True
    creation = _start_snapshot(source)
    options = WorkflowOptions(profile="requested_smoke_profile", out=str(tmp_path),
                              run_id="resume-run", resume=True, profile_start_snapshot=creation)
    run_dir, _ = _write_resume_fixture(tmp_path, creation_snapshot=creation, options=options)
    before = {path.relative_to(run_dir).as_posix(): path.read_bytes()
              for path in run_dir.rglob("*") if path.is_file()}
    runner = EvaluationWorkflowRunner(options)
    runner._run_locked = lambda: pytest.fail("unconfirmed resume dispatched")
    preview = runner.preview_force_build_start()
    assert preview["resume"] is True
    assert preview["backends"] == ("hailo",)
    with pytest.raises(ProductiveForceBuildDisabled, match="Resume"):
        runner.run()
    after = {path.relative_to(run_dir).as_posix(): path.read_bytes()
             for path in run_dir.rglob("*") if path.is_file()}
    assert after == before
    options.force_build_confirmed_backends = ("hailo",)
    # The old admission primitive remains independently testable. Real Resume
    # is rejected before that primitive under the v34 productive policy.
    runner._pending_force_build_confirmation = None
    runner._admit_force_build_start()
    assert runner._force_build_start_provenance["resume_requested"] is True
    assert runner._force_build_start_provenance["profile_source"] == "archived_run_snapshot"
    assert {path.relative_to(run_dir).as_posix(): path.read_bytes()
            for path in run_dir.rglob("*") if path.is_file()} == before


def test_t33_18_confirmation_is_outside_existing_resume_and_stage_identity(tmp_path):
    options = WorkflowOptions(profile="p", out=str(tmp_path))
    runner = EvaluationWorkflowRunner(options)
    resume_before = stable_resume_options(options)
    stage_before = runner._stable_stage_options_payload()
    options.force_build_confirmed_backends = ("hailo",)
    options.force_build_confirmation_source = "cli_explicit_option"
    assert stable_resume_options(options) == resume_before
    assert runner._stable_stage_options_payload() == stage_before


@pytest.mark.parametrize("args, expected, rc", [
    ([], (), 1),
    (["--confirm-force-build", "hailo"], ("hailo",), 0),
    (["--hailo-force-build"], ("hailo",), 0),
    (["--confirm-force-build", "deepx"], ("deepx",), 1),
])
def test_t33_10_18_cli_requires_explicit_backend_consent(monkeypatch, capsys, tmp_path, args, expected, rc):
    from onnx_splitpoint_tool.workflow import run_evaluation
    captured = []

    class Runner:
        def __init__(self, options, **kwargs):
            self.options = options

        def run(self):
            captured.append(self.options.force_build_confirmed_backends)
            require_force_build_consent(
                _profile(True), confirmed_backends=self.options.force_build_confirmed_backends,
                hailo_force_build=self.options.hailo_force_build,
            )
            return SimpleNamespace(ok=True, status="ok", to_dict=lambda: {"ok": True, "status": "ok"})

    monkeypatch.setattr(run_evaluation, "EvaluationWorkflowRunner", Runner)
    assert run_evaluation.main(["--profile", "synthetic.yaml", "--out", str(tmp_path), "--json", *args]) == rc
    result = json.loads(capsys.readouterr().out)
    assert captured == [expected]
    if rc:
        assert result["error_code"] == "force_build_confirmation_required"


def test_t33_18_profile_driven_cli_allows_confirmation_only():
    from onnx_splitpoint_tool.workflow.run_evaluation import _validate_profile_driven_args
    _validate_profile_driven_args(["--profile-driven", "--confirm-force-build", "deepx"])
    with pytest.raises(ValueError, match="unsupported options: --hailo-force-build"):
        _validate_profile_driven_args(["--profile-driven", "--hailo-force-build"])


def test_t33_20_archived_consent_text_cannot_authorize_another_start():
    profile = _profile(True)
    profile["force_build_confirmed_backends"] = ["hailo"]
    profile["force_build_start"] = {"confirmed_backends": ["hailo"]}
    with pytest.raises(ForceBuildConsentRequired):
        require_force_build_consent(profile, resume=True)


@pytest.mark.parametrize("name", ["hailo_build_hef", "hailo_build_hef_auto", "hailo_build_hef_via_venv", "hailo_build_hef_via_wsl"])
def test_t33_10_hailo_direct_build_rejects_text_before_compiler_or_artifact_write(tmp_path, name):
    from onnx_splitpoint_tool import hailo_backend
    out = tmp_path / "compiler-must-not-start"
    with pytest.raises(ValueError, match="config_boolean_invalid:hailo_build.force_build"):
        getattr(hailo_backend, name)(tmp_path / "missing.onnx", outdir=out, force="false")
    assert not out.exists()


def test_t33_20_resume_preview_shows_archived_force_without_rebinding_or_writing(tmp_path):
    from onnx_splitpoint_tool.workflow.artifacts import sha256_json
    archived = _profile(True, True)
    selected = tmp_path / "old-run"
    selected.mkdir()
    profile_path = selected / "profile.yaml"
    profile_path.write_text(yaml.safe_dump(archived))
    (selected / "run_manifest.json").write_text(json.dumps({"profile_hash": sha256_json(archived)}))
    before = profile_path.read_bytes()
    runner = EvaluationWorkflowRunner(WorkflowOptions(profile="today", out=str(tmp_path), run_id="old-run", resume=True))
    runner._load_profile = lambda: setattr(runner, "profile_payload", _profile(False, False))
    preview = runner.preview_force_build_start()
    assert preview["backends"] == ("hailo", "deepx")
    assert preview["profile_source"] == "archived_run_snapshot"
    assert preview["profile_path"] == str(profile_path)
    assert preview["profile_mismatch"] is True
    assert force_build_backends(runner.profile_payload) == ()
    assert profile_path.read_bytes() == before


def _load_resume_wrapper():
    import importlib.util
    path = Path(__file__).resolve().parents[1] / "scripts/resume_missing_full_quality.py"
    spec = importlib.util.spec_from_file_location("v33_resume_force_wrapper", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_t33_20_archived_wrapper_options_never_restore_consent(tmp_path):
    wrapper = _load_resume_wrapper()
    profile = tmp_path / "profile.yaml"
    profile.write_text(yaml.safe_dump(_profile(True)))
    options = WorkflowOptions(profile=str(profile), out=str(tmp_path),
                              force_build_confirmed_backends=("hailo", "deepx"),
                              force_build_confirmation_source="old_session")
    (tmp_path / "run_manifest.json").write_text(json.dumps({
        "schema": "onnx-splitpoint/evaluation-run-manifest", "options": options.to_dict(),
    }))
    restored = wrapper._options_from_manifest(tmp_path)
    assert restored.force_build_confirmed_backends == ()
    assert restored.force_build_confirmation_source == ""


def test_t33_20_wrapper_accepts_explicit_repeated_new_confirmation(tmp_path, monkeypatch, capsys):
    wrapper = _load_resume_wrapper()
    options = WorkflowOptions(profile="fixture", out=str(tmp_path))
    captured = []
    monkeypatch.setattr(wrapper, "_quality_result_row_count", lambda _path: 6)
    monkeypatch.setattr(wrapper, "_initial_resume_scope", lambda _path: {
        "preserved_results": 6, "allowed_model_rebuilds": [],
    })
    monkeypatch.setattr(wrapper, "_options_from_manifest", lambda _path: options)

    class Runner:
        def __init__(self, actual, **kwargs):
            captured.append(actual.force_build_confirmed_backends)

        def run(self):
            return SimpleNamespace()

    monkeypatch.setattr(wrapper, "EvaluationWorkflowRunner", Runner)
    monkeypatch.setattr(wrapper, "_verify_resume_outcome", lambda *args, **kwargs: {"status": "PASS"})
    monkeypatch.setattr("sys.argv", ["resume_missing_full_quality.py", "--run-dir", str(tmp_path),
                                   "--confirm-force-build", "hailo", "--confirm-force-build", "deepx"])
    assert wrapper.main() == 0
    assert captured == [("hailo", "deepx")]
    assert "MISSING_FULL_QUALITY_RESUME=PASS" in capsys.readouterr().out


@pytest.mark.parametrize("force, expected", [(False, False), (True, True), (None, False)])
def test_t33_10_compiler_environment_cannot_override_profile_force(monkeypatch, force, expected):
    from onnx_splitpoint_tool.deepx import env_status
    monkeypatch.setattr(env_status, "_select_compiler_fallback", lambda *args, **kwargs: {
        "force_build": True, "compiler_venv": "/synthetic-existing-venv",
    })
    profile = {"deepx_build": {}} if force is None else {"deepx_build": {"force_build": force}}
    assert env_status.profile_compiler_configuration(profile)["force_build"] is expected
    if expected:
        with pytest.raises(ForceBuildConsentRequired):
            require_force_build_consent(profile)
    else:
        assert require_force_build_consent(profile)["required_backends"] == []


def test_t33_18_denied_partial_consent_is_consumed_before_reusing_options(tmp_path):
    options = WorkflowOptions(profile="fixture", out=str(tmp_path),
                              force_build_confirmed_backends=("hailo",))
    runner = EvaluationWorkflowRunner(options)
    runner.profile_payload = _profile(True, True)
    with pytest.raises(ForceBuildConsentRequired):
        runner._admit_force_build_start()
    assert options.force_build_confirmed_backends == ()
    next_runner = EvaluationWorkflowRunner(options)
    next_runner.profile_payload = _profile(True, False)
    with pytest.raises(ForceBuildConsentRequired):
        next_runner._admit_force_build_start()


def test_t33_18_earlier_start_failure_also_consumes_confirmation(tmp_path):
    options = WorkflowOptions(profile="fixture", out=str(tmp_path),
                              force_build_confirmed_backends=("hailo",))
    runner = EvaluationWorkflowRunner(options)
    runner._load_profile = lambda: (_ for _ in ()).throw(ValueError("synthetic-load-error"))
    with pytest.raises(ValueError, match="synthetic-load-error"):
        runner.run()
    assert options.force_build_confirmed_backends == ()
    next_runner = EvaluationWorkflowRunner(options)
    next_runner.profile_payload = _profile(True)
    with pytest.raises(ForceBuildConsentRequired):
        next_runner._admit_force_build_start()


def test_t33_32_new_critical_modules_are_in_existing_build_inventory():
    import hashlib
    from onnx_splitpoint_tool.workflow.artifacts import package_build_snapshot

    snapshot = package_build_snapshot()
    package = Path(__file__).resolve().parents[1] / "onnx_splitpoint_tool"
    critical = snapshot["critical_module_sha256"]
    for name in ("config_values.py", "force_build_admission.py", "workflow/result_context.py"):
        assert name in critical
        assert critical[name] == "sha256:" + hashlib.sha256((package / name).read_bytes()).hexdigest()
    assert snapshot["critical_module_set_complete"] is True
