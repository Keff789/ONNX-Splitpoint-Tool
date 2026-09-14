"""v34 effective family dispatch, productive Force OFF and bounded preparation.

External compiler callbacks are synthetic. Configuration, wrappers, persisted
requests, GUI admission and the existing stage boundary are exercised directly.
"""
from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from onnx_splitpoint_tool import run_modes as rm
from onnx_splitpoint_tool.build_dispatch_policy import (
    ProductiveForceBuildDisabled, active_native_force_fields,
    bind_profile_hailo_builder, require_productive_force_off,
)
from onnx_splitpoint_tool.hailo_compiler_context import resolve_compute_selection
from onnx_splitpoint_tool.workflow.deferred_hailo_builds import (
    REQUEST_NAME, _completed_job_observation, cache_preflight_builder,
    finalize_deferred_hailo_builds,
)


@pytest.fixture(autouse=True)
def clean_compute_environment(monkeypatch):
    for key in tuple(os.environ):
        if "HAILO_COMPUTE" in key or key in {
            "ONNX_SPLITPOINT_HAILO_ALLOW_GPU", "SPLITPOINT_HAILO_ALLOW_GPU", "CUDA_VISIBLE_DEVICES",
        }:
            monkeypatch.delenv(key, raising=False)


def _profile():
    return {
        "hailo_build": {"force_build": False, "compute_by_family": {
            "hailo8": {"device": "cpu"}, "hailo10h": {"device": "gpu", "gpu_selector": "0"},
        }},
        "deepx_build": {"force_build": False, "classification_preprocessing": "imagenet_mean_std"},
    }


@pytest.mark.parametrize("mode", ["smoke", "standard", "final"])
def test_t34_04_08_modes_default_cpu_and_force_off(mode):
    cfg = rm.default_run_modes_config()
    resolved, audit = rm.apply_run_mode({}, config=cfg, mode_id=mode)
    summary = audit["build_summary"]
    assert summary == rm.profile_build_summary(resolved)
    assert {row["device"] for row in summary["hailo_compute_by_family"].values()} == {"cpu"}
    assert summary["hailo_force_build"] is summary["deepx_force_build"] is False
    assert summary["hailo_cache_integrity"] == "relaxed"


def test_t34_08_alias_normalized_before_packaged_cpu_defaults_merge():
    cfg = rm.validate_run_modes_config({"modes": {"standard": {"build": {"hailo": {
        "compute_by_family": {"hailo10": {"device": "gpu"}},
    }}}}})
    mapping = cfg["modes"]["standard"]["build"]["hailo"]["compute_by_family"]
    assert mapping == {"hailo10h": {"device": "gpu"}}
    assert resolve_compute_selection("hailo8", compute_by_family=mapping, env={})["device"] == "cpu"
    with pytest.raises(ValueError, match="alias_conflict"):
        rm.validate_run_modes_config({"modes": {"standard": {"build": {"hailo": {
            "compute_by_family": {"hailo10": {"device": "gpu"}, "hailo10h": {"device": "cpu"}},
        }}}}})


def test_t34_04_explicit_profile_compute_survives_reload_and_mode_change():
    profile = _profile()
    original = copy.deepcopy(profile)
    config = rm.default_run_modes_config()
    effective, _ = rm.apply_run_mode(profile, config=config, mode_id="standard")
    assert profile == original
    assert effective["execution_preset"]["build_provenance"]["hailo_compute_source"] == "evaluation_profile"
    again, _ = rm.apply_run_mode(effective, config=config, mode_id="final")
    assert again["hailo_build"]["compute_by_family"] == original["hailo_build"]["compute_by_family"]
    assert again["hailo_build"]["force_build"] is False


def test_t34_04_default_compute_is_not_promoted_to_stale_profile_override():
    config = rm.default_run_modes_config()
    effective, _ = rm.apply_run_mode({}, config=config, mode_id="standard")
    config["modes"]["standard"]["build"]["hailo"]["compute_by_family"]["hailo10h"]["device"] = "gpu"
    updated, _ = rm.apply_run_mode(effective, config=config)
    assert updated["hailo_build"]["compute_by_family"]["hailo10h"]["device"] == "gpu"
    assert updated["execution_preset"]["build_provenance"]["hailo_compute_source"] == "tool_config"


def test_t34_08_legacy_registry_absence_preserves_environment_source(tmp_path, monkeypatch):
    import yaml
    path = tmp_path / "legacy_registry.yaml"
    old = rm.default_run_modes_config()
    for mode in old["modes"].values():
        mode["build"]["hailo"].pop("compute_by_family")
    path.write_text(yaml.safe_dump(old))
    before = path.read_bytes()
    monkeypatch.setenv("ONNX_SPLITPOINT_HAILO_ALLOW_GPU", "1")
    loaded = rm.load_run_modes_config(path)
    resolved, _ = rm.apply_run_mode({}, config=loaded, config_path=path)
    summary = rm.profile_build_summary(resolved)["hailo_compute_by_family"]
    assert resolved["hailo_build"]["compute_by_family"] == {}
    assert summary["hailo10h"]["device"] == "gpu"
    assert summary["hailo10h"]["source"] == "environment:ONNX_SPLITPOINT_HAILO_ALLOW_GPU"
    assert summary["hailo8"]["device"] == "cpu"
    assert summary["hailo8"]["ignored_legacy"]
    assert path.read_bytes() == before


def test_t34_04_08_summary_and_threaded_dispatch_share_detached_family_mapping():
    profile = _profile()
    effective, _ = rm.apply_run_mode(profile, config=rm.default_run_modes_config())
    summary = rm.profile_build_summary(effective)
    seen = []
    def builder(source, **kwargs):
        seen.append(kwargs)
        return resolve_compute_selection(kwargs["hw_arch"], compute_by_family=kwargs["compute_by_family"])
    callback = bind_profile_hailo_builder(builder, effective)
    before = dict(os.environ)
    effective["hailo_build"]["compute_by_family"]["hailo10h"]["device"] = "cpu"
    with ThreadPoolExecutor(max_workers=1) as pool:
        selected = pool.submit(callback, "fixture.onnx", hw_arch="hailo10").result()
    assert selected["device"] == summary["hailo_compute_by_family"]["hailo10h"]["device"] == "gpu"
    assert selected["source"] == summary["hailo_compute_by_family"]["hailo10h"]["source"]
    assert seen[0]["compute_by_family"]["hailo8"]["device"] == "cpu"
    assert dict(os.environ) == before


def test_t34_08_compute_conflict_is_visible_without_blocking_cache_summary(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    summary = rm.profile_build_summary(_profile())
    assert summary["hailo_compute_by_family"]["hailo10h"]["status"] == "conflict"
    assert summary["hailo_compute_by_family"]["hailo10h"]["compatible_cache_reuse_allowed"] is True
    assert "Konflikt vor Kaltbuild" in summary["hailo_compute_text"]
    assert summary["hailo_compute_by_family"]["hailo8"]["device"] == "cpu"


@pytest.mark.parametrize("family", ["hailo", "deepx"])
def test_t34_06_productive_force_is_rejected_without_modifying_profile(family):
    profile = _profile()
    profile[f"{family}_build"]["force_build"] = True
    before = copy.deepcopy(profile)
    with pytest.raises(ProductiveForceBuildDisabled, match=f"{family}_build.force_build"):
        require_productive_force_off(profile)
    assert profile == before


@pytest.mark.parametrize("key", ["force_rebuild_engines", "native_force_rebuild_engines", "force_rebuild_native_engines"])
def test_t34_06_native_flags_are_checked_and_visible(key):
    profile = _profile()
    profile["native_producers"] = {"variants": [{"id": "v", key: True}]}
    field = f"native_producers.variants[0].{key}"
    assert active_native_force_fields(profile) == [field]
    assert rm.profile_build_summary(profile)["native_force_fields"] == [field]
    with pytest.raises(ProductiveForceBuildDisabled):
        require_productive_force_off(profile)
    profile["native_producers"]["variants"][0][key] = "false"
    with pytest.raises(ValueError, match="config_boolean_invalid"):
        require_productive_force_off(profile)


def test_t34_06_archived_snapshot_does_not_override_effective_force_off():
    profile = _profile()
    profile["execution_preset"] = {"snapshot": {"runtime": {"native": {"force_rebuild_engines": True}}}}
    require_productive_force_off(profile)
    with pytest.raises(ProductiveForceBuildDisabled, match="options.hailo_force_build"):
        require_productive_force_off(profile, hailo_force_build=True)


def test_t34_07_explicit_scale_only_is_visible_and_normal_admission_blocks_it():
    from onnx_splitpoint_tool.deepx.config import classification_profile_admission
    profile = _profile()
    profile["deepx_build"]["classification_preprocessing"] = "current_scale_only"
    resolved, audit = rm.apply_run_mode(profile, config=rm.default_run_modes_config())
    assert audit["build_summary"]["deepx_classification_source"] == "Evaluationsprofil"
    assert "current_scale_only" in rm.run_mode_profile_brief(resolved)
    assert classification_profile_admission(resolved, {"task": "classification"})["allowed"] is False
    assert classification_profile_admission(resolved, {"task": "detection"})["allowed"] is True


def test_t34_58_preparation_stage_boundary_survives_normal_reload():
    # Loader option API is inspected separately below; the persisted explicit
    # boundary must survive central mode resolution exactly.
    profile = _profile()
    profile["workflow"] = {"stop_after": "build_backend_artifacts"}
    resolved, _ = rm.apply_run_mode(profile, config=rm.default_run_modes_config())
    again, _ = rm.apply_run_mode(resolved, config=rm.default_run_modes_config())
    assert again["workflow"]["stop_after"] == "build_backend_artifacts"


def test_t34_58_existing_stop_boundary_waits_for_publication_and_stage_commit(tmp_path):
    from _v27930_terminal_lifecycle_fixture import TerminalLifecycleRunner, options_for
    from onnx_splitpoint_tool.workflow.artifacts import write_json
    observed = []
    class Runner(TerminalLifecycleRunner):
        def _run_model_pipeline_with_cache_preflight(self, rows):
            def publish():
                assert self._stop_requested is False
                output = self.run_dir / "reports" / "synthetic_published_artifact.json"
                receipt = self.run_dir / "reports" / "synthetic_published_receipt.json"
                write_json(output, {"synthetic": True})
                assert self._stop_requested is False
                write_json(receipt, {"artifact": output.name, "complete": True})
                return {"artifact": output, "receipt": receipt}, {}, "published", "ok"
            self._run_stage(None, "build_backend_artifacts", publish)
            stage = json.loads((self.run_dir / "stages/build_backend_artifacts/stage_result.json").read_text())
            observed.append((self._stop_requested, stage["status"]))
            if not self._stop_requested:
                pytest.fail("next dispatch was allowed after preparation boundary")
    options = options_for(tmp_path)
    options.stop_after = "build_backend_artifacts"
    runner = Runner(options)
    runner.run()
    assert observed == [(True, "ok")]
    assert runner._run_lock is None
    assert (runner.run_dir / "reports/artifact_index_closure.json").is_file()


def test_t34_04_51_60_persisted_preflight_and_continuation_keep_same_compute(tmp_path):
    from test_v27922_negative_preflight import _suite, _result
    model, suite, output, source = _suite(tmp_path)
    profile = _profile()
    calls = []
    def builder(source, **kwargs):
        calls.append(copy.deepcopy(kwargs))
        result = _result(ok=True, hef_path=str(output / "compiled.hef"))
        (output / "compiled.hef").write_bytes(b"synthetic-build-artifact")
        result.details = {"cache_hit": True, "compiler_dispatch_count": 0}
        return result
    wrapped = bind_profile_hailo_builder(cache_preflight_builder(builder), profile)
    wrapped(source, outdir=str(output), hw_arch="hailo10h", net_name="yolo26s_part1_b364", force=False,
            build_evidence_context={"model_id": "yolo26s", "boundary": 364, "stage": "part1"})
    stored = json.loads((output / REQUEST_NAME).read_text())
    assert stored["kwargs"]["compute_by_family"] == profile["hailo_build"]["compute_by_family"]
    finished = finalize_deferred_hailo_builds(model_dir=model, run_dir=tmp_path, profile_payload=profile, build_fn=builder)
    assert calls[0]["compute_by_family"] == calls[1]["compute_by_family"]
    job = finished["jobs"][0]
    assert job["model_id"] == "yolo26s" and job["boundary"] == "b364"
    assert job["backend"] == "hailo10h" and job["stage"] == "part1"
    assert job["preflight_decision"] == "HIT"
    assert job["compiler_dispatch_count"] == 0 and job["model_build_status"] == "reused"


def test_t34_60_missing_dispatch_evidence_stays_unknown():
    result = SimpleNamespace(ok=True, details={}, calib_info={}, hef_path="private.hef")
    observed = _completed_job_observation({"kwargs": {"net_name": "resnet50_full", "outdir": "/job/full"}}, result, model_id="resnet50")
    assert observed["compiler_dispatch_count"] is None
    assert observed["cache_hit"] is None
    assert observed["preflight_decision"] == "UNKNOWN"
    assert observed["next_required_action"] == "runtime_binding_and_quality"


@pytest.mark.parametrize("script,required,force_flag", [
    ("native_fifo_eval_runner.py", ["--root", "/not-read"], "--force-rebuild-engines"),
    ("native_fifo_smoke_matrix.py", ["--benchmark-set", "/not-read"], "--force-rebuild-engines"),
    ("native_trt_from_benchmarkset.py", ["--benchmark-set", "/not-read"], "--force-rebuild"),
    ("native_producer_e2e_eval_runner.py", ["--root", "/not-read", "--backend", "hailo10h"], "--force-rebuild-engine"),
    ("native_hailo10_trt_e2e_from_benchmarkset.py", ["--benchmark-set", "/not-read", "--case", "b001"], "--force-rebuild-engine"),
    ("update_evalset_native_producers.py", ["--eval-run-dir", "/not-read"], "--native-force-rebuild-engines"),
])
def test_t34_06_direct_native_force_cli_stops_before_work(script, required, force_flag):
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run([sys.executable, "-B", str(root / "scripts" / script), *required, force_flag], text=True, capture_output=True)
    assert completed.returncode == 2
    assert "productive_force_build_disabled" in completed.stderr
    assert (root / "scripts" / script).read_bytes() == (root / "onnx_splitpoint_tool/resources/remote_scripts" / script).read_bytes()


def test_t34_06_native_variant_force_config_stops_before_report_writes(tmp_path):
    root = Path(__file__).resolve().parents[1]
    config = tmp_path / "native.json"
    config.write_text(json.dumps({"enabled": True, "variants": [{"id": "legacy", "force_rebuild_engines": True}]}))
    run_dir = tmp_path / "untouched-run"
    completed = subprocess.run([sys.executable, "-B", str(root / "scripts/run_evalrun_native_producer_variants.py"),
                                "--eval-run-dir", str(run_dir), "--config", str(config)], text=True, capture_output=True)
    assert completed.returncode == 2
    assert "productive_force_build_disabled" in completed.stderr
    assert not run_dir.exists()


@pytest.mark.parametrize("stage", ["full", "part1"])
@pytest.mark.parametrize("source", ["argument", "build_config"])
def test_t34_06_manual_deepx_force_stops_before_local_writes(tmp_path, stage, source):
    from onnx_splitpoint_tool.gui import benchmark_workflow
    output = tmp_path / "not-created"
    common = dict(out_dir=output, bench_plan_runs=[{"type": "deepx"}], validation_images="",
                  fallback_calib_dir=None, force_build=source == "argument",
                  build_config={"force_build": source == "build_config"})
    callback = benchmark_workflow._materialize_manual_deepx_part1_artifacts
    if stage == "full":
        callback = benchmark_workflow._materialize_manual_deepx_full_artifact
        common.update(model_path="/not-read.onnx", model=None, validation_max_images=1)
    with pytest.raises(ProductiveForceBuildDisabled):
        callback(**common)
    assert not output.exists()


def test_t34_04_profile_editor_retains_explicit_compute_and_stop_boundary(monkeypatch):
    from test_profile_editor_campaign_roundtrip_v261e import _headless_editor
    editor = _headless_editor(monkeypatch)
    source = _profile()
    source["name"] = "synthetic-compute-profile"
    source["model_suite"] = {"primary": [{"id": "resnet50", "task": "classification", "enabled": True}]}
    source["run_profiles"] = [{"id": "hailo8", "enabled": True}]
    source["workflow"] = {"stop_after": "build_backend_artifacts"}
    source["execution_preset"] = {"id": "standard", "follow_tool_config": False,
                                  "snapshot": rm.default_run_modes_config()["modes"]["standard"]}
    resolved, _ = rm.apply_run_mode(source, config=rm.default_run_modes_config(), follow_tool_config=False)
    editor._apply_payload(resolved)
    rebuilt = editor._build_payload()
    assert rebuilt["hailo_build"]["compute_by_family"] == source["hailo_build"]["compute_by_family"]
    assert rebuilt["workflow"]["stop_after"] == "build_backend_artifacts"


@pytest.mark.parametrize("assignment,recipe_key", [
    ("r1_base_conv_retry", "end_node_names"), ("r1_retry", "extra_model_script"),
])
def test_t34_54_changed_recipe_retry_uses_normal_nonforce_dispatch(tmp_path, assignment, recipe_key):
    import ast
    from onnx_splitpoint_tool.hailo_build_context import make_build_evidence_context
    root = Path(__file__).resolve().parents[1]
    tree = ast.parse((root / "onnx_splitpoint_tool/benchmark/services.py").read_text())
    calls = [node.value for node in ast.walk(tree) if isinstance(node, ast.Assign)
             and any(isinstance(target, ast.Name) and target.id == assignment for target in node.targets)
             and isinstance(node.value, ast.Call)]
    assert len(calls) == 1
    source = tmp_path / "model.onnx"
    source.write_bytes(b"synthetic-full-source")
    observed = []
    def external_builder(source, **kwargs):
        observed.append(kwargs)
        return "existing-compatible-recipe-reusable"
    cfg = SimpleNamespace(
        hailo_build_hef_fn=bind_profile_hailo_builder(external_builder, _profile()),
        base="model", full_model_src=str(source), hef_fixup=False, hef_opt_level=1,
        hef_calib_dir="fixture-b500", hef_calib_count=500, hef_calib_bs=8,
        hef_wsl_distro=None, hef_wsl_venv="auto", benchmark_task="detection",
    )
    environment = dict(cfg=cfg, p1_build_path=str(source), build_backend="venv", hw_arch="hailo10h",
                       b=364, out_p1=str(tmp_path / "part1"), make_build_evidence_context=make_build_evidence_context,
                       manifest_out={"boundary":364}, cache_only_effective=False, timeout_effective=3600,
                       _on_hef_log=lambda *_args:None, explicit_end_nodes=["actual-dfc-output"],
                       salvage_script="model_optimization_flavor(compression_level=0)\n")
    result = eval(compile(ast.fix_missing_locations(ast.Expression(body=calls[0])), "<actual-recipe-retry-call>", "eval"), environment)
    assert result == "existing-compatible-recipe-reusable"
    assert observed[0]["force"] is False and observed[0]["cache_only"] is False
    assert observed[0][recipe_key]
    assert observed[0]["build_evidence_context"]["boundary"] == 364
    assert observed[0]["compute_by_family"] == _profile()["hailo_build"]["compute_by_family"]
