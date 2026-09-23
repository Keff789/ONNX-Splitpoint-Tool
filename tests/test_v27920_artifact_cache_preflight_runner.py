from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from onnx_splitpoint_tool.workflow.runner import (
    ALL_STAGES,
    BARRIER_STAGES,
    EvaluationWorkflowRunner,
    FINAL_STAGES,
    MODEL_EXECUTION_STAGES,
    MODEL_POST_BARRIER_STAGES,
    MODEL_PREPARATION_STAGES,
    MODEL_STAGES,
)


def _bare_runner() -> EvaluationWorkflowRunner:
    runner = object.__new__(EvaluationWorkflowRunner)
    runner._stop_requested = False
    runner.jobs = None
    runner.profile_payload = {}
    return runner


def _classification_cache_inputs(tmp_path):
    """Valid v31 MeanStd cache inputs; no compiler or device is involved."""
    import onnx
    from onnx import TensorProto, helper
    from PIL import Image
    from onnx_splitpoint_tool.campaign import create_dataset_manifest

    calibration_dir = tmp_path / "calibration"
    calibration_dir.mkdir()
    Image.new("RGB", (4, 4), (80, 120, 160)).save(calibration_dir / "sample.png")
    manifest = tmp_path / "calibration_manifest.json"
    create_dataset_manifest(
        task="classification", role="calibration", dataset_id="synthetic-cache-test",
        split="train", root=calibration_dir, output=manifest, hash_mode="content",
    )
    source = tmp_path / "model.onnx"
    graph = helper.make_graph(
        [helper.make_node("Identity", ["images"], ["features"])],
        "synthetic-cache-test",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 224, 224])],
        [helper.make_tensor_value_info("features", TensorProto.FLOAT, [1, 3, 224, 224])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    onnx.save_model(model, source)
    return source, calibration_dir, manifest


def _static_compiler_environment(tmp_path, **kwargs):
    # A valid key may use bounded installed metadata, never an active probe.
    assert kwargs["path_only"] is True
    assert kwargs["probe_import"] is False
    return {
        "compiler_ready": True,
        "runtime_ready": True,
        "cache_dir": str(tmp_path / "deepx-cache"),
        "compiler_imports": [{
            "module": "dx_com", "ok": True,
            "distribution": "dx-com", "package_version": "synthetic-cache-test",
        }],
    }


def test_stage_partition_places_preflight_between_preparation_and_execution() -> None:
    assert MODEL_STAGES == (
        MODEL_PREPARATION_STAGES
        + MODEL_POST_BARRIER_STAGES
        + MODEL_EXECUTION_STAGES
    )
    assert MODEL_PREPARATION_STAGES[-1] == "generate_benchmark_set"
    assert MODEL_POST_BARRIER_STAGES == ["build_backend_artifacts"]
    assert MODEL_EXECUTION_STAGES == [
        "run_benchmarks", "validate_outputs", "hardware_smoke",
    ]
    assert BARRIER_STAGES == ["artifact_cache_preflight"]
    assert "artifact_cache_preflight" not in FINAL_STAGES
    assert ALL_STAGES.index("generate_benchmark_set") < ALL_STAGES.index(
        "artifact_cache_preflight"
    ) < ALL_STAGES.index("build_backend_artifacts") < ALL_STAGES.index(
        "run_benchmarks"
    )


def test_generation_profile_defers_deepx_prefetch_without_mutating_profile() -> None:
    runner = _bare_runner()
    runner.profile_payload = {
        "build_scheduler": {
            "enabled": True,
            "prefetch_deepx_full": True,
            "max_workers": 3,
        },
    }
    runner._artifact_cache_preflight_pending = True

    fenced = runner._profile_for_benchmark_generation()

    assert fenced is not runner.profile_payload
    assert fenced["build_scheduler"] == {
        "enabled": True,
        "prefetch_deepx_full": False,
        "max_workers": 3,
        "deferred_by_artifact_cache_preflight": True,
    }
    assert runner.profile_payload["build_scheduler"][
        "prefetch_deepx_full"
    ] is True

    runner._artifact_cache_preflight_pending = False
    assert runner._profile_for_benchmark_generation() is runner.profile_payload


def test_deepx_preflight_probe_cannot_dispatch_compiler(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from onnx_splitpoint_tool.workflow import deepx_build_binding

    runner = _bare_runner()
    runner.run_dir = tmp_path
    runner.options = SimpleNamespace()
    source, calibration_dir, manifest = _classification_cache_inputs(tmp_path)
    runner.profile_payload = {
        "campaign": {"dataset_manifests": {"classification": {"calibration": str(manifest)}}},
        "deepx_build": {
            "mode": "reuse_and_build_missing",
            "force_build": True,
            "calibration_dir": str(calibration_dir),
            "calib_count": 1,
            "cache_dir": str(tmp_path / "deepx-cache"),
        },
    }
    runner.log = lambda _message: None
    (tmp_path / "models/model_a/benchmark_set").mkdir(parents=True)

    monkeypatch.setattr(
        deepx_build_binding,
        "inspect_deepx_environment",
        lambda **kwargs: _static_compiler_environment(tmp_path, **kwargs),
    )
    monkeypatch.setattr(
        deepx_build_binding,
        "compile_dxnn",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("DX-COM dispatched before cache preflight")
        ),
    )

    errors = runner._materialize_local_cache_preflight_inputs(
        [{
            "id": "model_a",
            "task": "classification",
            "resolved_path": str(source),
        }],
        ["deepx_m1"],
    )

    assert errors == []
    status = json.loads((
        tmp_path
        / "models/model_a/benchmark_set/deepx/deepx_artifact_status.json"
    ).read_text())
    assert status["mode"] == "cache_verify_only"
    assert status["cache_lookup"]["outcome"] == "MISS"
    assert status["build_status"] == "cache_miss_blocked"
    assert status["classification_preprocessing"] == "imagenet_mean_std"
    assert status["calibration_manifest_contract"]["status"] == "resolved"
    assert status["compiler_identity"]["status"] == "resolved"
    assert runner.profile_payload["deepx_build"] == {
        "mode": "reuse_and_build_missing",
        "force_build": True,
        "calibration_dir": str(calibration_dir),
        "calib_count": 1,
        "cache_dir": str(tmp_path / "deepx-cache"),
    }


def test_strict_deepx_miss_blocks_before_backend_build(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from onnx_splitpoint_tool.workflow import deepx_build_binding

    runner = _bare_runner()
    runner.run_dir = tmp_path
    runner.options = SimpleNamespace(benchmark_execution_backend="local")
    runner.outputs = {}
    runner.report_paths = []
    runner._targets = lambda: ["deepx_m1"]
    runner._emit_log = lambda _message: None
    runner.log = lambda _message: None
    source, calibration_dir, manifest = _classification_cache_inputs(tmp_path)
    runner.profile_payload = {
        "targets": ["deepx_m1"],
        "campaign": {"dataset_manifests": {"classification": {"calibration": str(manifest)}}},
        "artifact_cache_preflight": {"require_warm_cache": True},
        "deepx_build": {
            "mode": "reuse_and_build_missing",
            "calibration_dir": str(calibration_dir),
            "calib_count": 1,
            "cache_dir": str(tmp_path / "deepx-cache"),
        },
    }
    bdir = tmp_path / "models/model_a/benchmark_set"
    bdir.mkdir(parents=True)
    (bdir / "benchmark_set.json").write_text(json.dumps({
        "model_id": "model_a", "model": str(source), "cases": [],
    }))
    (bdir / "benchmark_plan.json").write_text(json.dumps({"runs": []}))

    monkeypatch.setattr(
        deepx_build_binding,
        "inspect_deepx_environment",
        lambda **kwargs: _static_compiler_environment(tmp_path, **kwargs),
    )
    monkeypatch.setattr(
        deepx_build_binding,
        "compile_dxnn",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("DX-COM dispatched by strict cache preflight")
        ),
    )

    artifacts, metrics, _message, status = (
        runner._stage_artifact_cache_preflight([{
            "id": "model_a",
            "task": "classification",
            "resolved_path": str(source),
        }])
    )

    assert status == "failed"
    assert metrics["runtime_dispatch_allowed"] is False
    assert metrics["unexpected_cold_builds"] == 1
    report = json.loads(artifacts["artifact_cache_preflight_json"].read_text())
    assert report["matrix"][0]["cells"]["deepx"]["status"] == "MISS"


def test_two_phase_pipeline_prepares_every_model_before_runtime_dispatch() -> None:
    runner = _bare_runner()
    calls: list[tuple[str, str, tuple[str, ...]]] = []

    def run_model(row, *, stage_names, start_job, finish_job):
        calls.append((
            "model", str(row["id"]), tuple(stage_names),
        ))

    def run_stage(model_id, stage, fn):
        del model_id, fn
        calls.append(("root", stage, ()))
        return SimpleNamespace(status="ok")

    runner._run_model = run_model
    runner._run_stage = run_stage
    runner._stage_artifact_cache_preflight = lambda rows: ({}, {}, "", "ok")

    runner._run_model_pipeline_with_cache_preflight([
        {"id": "model_a"}, {"id": "model_b"},
    ])

    assert calls == [
        ("model", "model_a", tuple(MODEL_PREPARATION_STAGES)),
        ("model", "model_b", tuple(MODEL_PREPARATION_STAGES)),
        ("root", "artifact_cache_preflight", ()),
        ("model", "model_a", tuple(
            MODEL_POST_BARRIER_STAGES + MODEL_EXECUTION_STAGES
        )),
        ("model", "model_b", tuple(
            MODEL_POST_BARRIER_STAGES + MODEL_EXECUTION_STAGES
        )),
    ]


def test_runtime_stop_after_keeps_debug_scope_to_first_model() -> None:
    runner = _bare_runner()
    runner.options = SimpleNamespace(stop_after="run_benchmarks")
    runner.profile_payload = {}
    calls: list[tuple[str, str, tuple[str, ...]]] = []

    def run_model(row, *, stage_names, start_job, finish_job):
        del start_job, finish_job
        calls.append(("model", str(row["id"]), tuple(stage_names)))
        if list(stage_names) == (
            MODEL_POST_BARRIER_STAGES + MODEL_EXECUTION_STAGES
        ):
            runner._stop_requested = True

    runner._run_model = run_model
    runner._run_stage = lambda _model, stage, _fn: (
        calls.append(("root", stage, ()))
        or SimpleNamespace(status="ok", details={})
    )

    runner._run_model_pipeline_with_cache_preflight([
        {"id": "model_a"}, {"id": "model_b"},
    ])

    assert calls == [
        ("model", "model_a", tuple(MODEL_PREPARATION_STAGES)),
        ("root", "artifact_cache_preflight", ()),
        ("model", "model_a", tuple(
            MODEL_POST_BARRIER_STAGES + MODEL_EXECUTION_STAGES
        )),
    ]


def test_preflight_publication_failure_blocks_compilers_even_in_diagnostic_mode() -> None:
    runner = _bare_runner()
    runner.profile_payload = {}
    calls: list[tuple[str, str, tuple[str, ...]]] = []
    runner._run_model = lambda row, **kwargs: calls.append(
        ("model", str(row["id"]), tuple(kwargs["stage_names"])),
    )
    runner._run_stage = lambda _model, stage, _fn: (
        calls.append(("root", stage, ()))
        or SimpleNamespace(status="failed", details={})
    )

    runner._run_model_pipeline_with_cache_preflight([{"id": "model_a"}])

    assert calls == [
        ("model", "model_a", tuple(MODEL_PREPARATION_STAGES)),
        ("root", "artifact_cache_preflight", ()),
    ]
    assert runner._stop_requested is True


def test_resume_never_reuses_external_cache_preflight(tmp_path) -> None:
    runner = _bare_runner()
    runner.run_dir = tmp_path
    runner.run_id = "run"
    runner.profile_id = "profile"
    runner.options = SimpleNamespace(resume=True)

    decision = runner._resume_reuse_decision(
        model_id=None,
        stage="artifact_cache_preflight",
        previous={
            "status": "ok",
            "state": "completed",
            "complete": True,
            "details": {"resume_key_hash": "same"},
            "artifacts": ["reports/artifact_cache_preflight.json"],
        },
        expected_hash="same",
        forced=False,
        stage_job_id="stage:workflow:artifact_cache_preflight",
        result_path=(
            tmp_path / "stages/artifact_cache_preflight/stage_result.json"
        ),
    )

    assert decision["reusable"] is False
    assert decision["reason"] == (
        "artifact_cache_preflight_requires_fresh_probe"
    )


def test_explicit_strict_preflight_failure_stops_all_runtime_dispatch() -> None:
    runner = _bare_runner()
    runner.profile_payload = {
        "artifact_cache_preflight": {"require_warm_cache": True},
    }
    calls: list[tuple[str, str, tuple[str, ...]]] = []

    class Jobs:
        def __init__(self):
            self.finished: list[str] = []

        def finish_model(self, model_id: str) -> None:
            self.finished.append(model_id)

        @staticmethod
        def artifacts():
            return []

    runner.jobs = Jobs()
    runner._register_artifacts = lambda *args, **kwargs: None

    def run_model(row, *, stage_names, start_job, finish_job):
        del start_job, finish_job
        calls.append(("model", str(row["id"]), tuple(stage_names)))

    def run_stage(model_id, stage, fn):
        del model_id, fn
        calls.append(("root", stage, ()))
        return SimpleNamespace(
            status="failed", details={"runtime_dispatch_allowed": False},
        )

    runner._run_model = run_model
    runner._run_stage = run_stage
    runner._stage_artifact_cache_preflight = lambda rows: ({}, {}, "", "failed")

    runner._run_model_pipeline_with_cache_preflight([
        {"id": "model_a"}, {"id": "model_b"},
    ])

    assert calls == [
        ("model", "model_a", tuple(MODEL_PREPARATION_STAGES)),
        ("model", "model_b", tuple(MODEL_PREPARATION_STAGES)),
        ("root", "artifact_cache_preflight", ()),
    ]
    assert runner.jobs.finished == ["model_a", "model_b"]
    assert runner._stop_requested is True


def test_execution_phase_preserves_prior_model_local_generation_failure(
    tmp_path,
) -> None:
    runner = _bare_runner()
    runner.run_dir = tmp_path
    runner.manifest = {"models": {}}
    runner.stage_results = [{
        "model_id": "broken_model",
        "stage": "generate_benchmark_set",
        "status": "failed",
    }]
    called: list[tuple[str, str]] = []

    def run_stage(model_id, stage, fn):
        called.append((model_id, stage))
        _artifacts, details, _message, status = fn()
        assert status == "skipped"
        assert details["blocked_by_stage"] == "generate_benchmark_set"
        return SimpleNamespace(status=status)

    runner._run_stage = run_stage
    runner._run_model(
        {"id": "broken_model"},
        stage_names=MODEL_EXECUTION_STAGES,
        start_job=False,
        finish_job=False,
    )

    assert called == [
        ("broken_model", "run_benchmarks"),
        ("broken_model", "validate_outputs"),
        ("broken_model", "hardware_smoke"),
    ]


def test_root_preflight_stage_is_diagnostic_and_writes_all_views(tmp_path) -> None:
    runner = _bare_runner()
    runner.run_dir = tmp_path
    runner.profile_payload = {"targets": ["tensorrt"]}
    runner.outputs = {}
    runner.report_paths = []
    runner._targets = lambda: ["tensorrt"]
    runner._emit_log = lambda _message: None
    bdir = tmp_path / "models/resnet50/benchmark_set"
    bdir.mkdir(parents=True)
    (bdir / "benchmark_set.json").write_text(json.dumps({
        "cases": [{"id": "b052"}],
    }))
    (bdir / "benchmark_plan.json").write_text(json.dumps({
        "runs": [{"id": "ort_tensorrt"}],
    }))

    artifacts, metrics, _message, status = (
        runner._stage_artifact_cache_preflight([{"id": "resnet50"}])
    )

    assert status == "ok"
    assert metrics["runtime_dispatch_allowed"] is True
    # No explicit variants means the ordinary TRT Full + P1/P2 recipe.
    # Diagnostic mode remains non-blocking, but must expose every unknown.
    assert metrics["unknown_cache_probes"] == 3
    assert set(artifacts) == {
        "artifact_cache_preflight_json",
        "artifact_cache_preflight_csv",
        "artifact_cache_preflight_md",
        "artifact_cache_preflight_items_csv",
        "remote_trt_cache_preflight_json",
        "hailo_compiler_preflight_json",
        "hailo_workspace_preflight_json",
    }
    report = json.loads(
        artifacts["artifact_cache_preflight_json"].read_text()
    )
    assert report["matrix"][0]["cells"]["trt_full"]["status"] == "UNKNOWN"
    assert report["matrix"][0]["cells"]["trt_p1"]["status"] == "UNKNOWN"
    assert report["matrix"][0]["cells"]["trt_p2"]["status"] == "UNKNOWN"


def test_disabled_preflight_performs_no_local_or_remote_probe(tmp_path) -> None:
    runner = _bare_runner()
    runner.run_dir = tmp_path
    runner.profile_payload = {
        "artifact_cache_preflight": {"enabled": False},
    }
    runner.outputs = {}
    runner.report_paths = []
    runner._targets = lambda: (_ for _ in ()).throw(
        AssertionError("disabled preflight inspected targets"),
    )
    runner._emit_log = lambda _message: None

    artifacts, metrics, _message, status = (
        runner._stage_artifact_cache_preflight([{"id": "resnet50"}])
    )

    assert status == "ok"
    assert metrics["artifact_cache_preflight_status"] == "disabled"
    assert metrics["unknown_cache_probes"] == 0
    remote = json.loads(
        artifacts["remote_trt_cache_preflight_json"].read_text()
    )
    assert remote["status"] == "disabled"
    assert remote["probe_count"] == 0


def test_local_execution_backend_does_not_probe_remote_hosts(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from onnx_splitpoint_tool.benchmark import remote_run
    from onnx_splitpoint_tool.workflow import execution_binding

    runner = _bare_runner()
    runner.run_dir = tmp_path
    runner.profile_payload = {"targets": ["tensorrt"]}
    runner.options = SimpleNamespace(benchmark_execution_backend="local")
    runner.outputs = {}
    runner.report_paths = []
    runner._targets = lambda: ["tensorrt"]
    runner._emit_log = lambda _message: None
    runner._profile_with_cli_hardware_overrides = lambda: {}
    runner._task_for = lambda _row: "classification"
    bdir = tmp_path / "models/resnet50/benchmark_set"
    bdir.mkdir(parents=True)
    (bdir / "benchmark_set.json").write_text(
        json.dumps({"cases": [{"id": "b052"}]}),
    )
    (bdir / "benchmark_plan.json").write_text(
        json.dumps({"runs": [{"id": "ort_tensorrt"}]}),
    )
    monkeypatch.setattr(
        execution_binding,
        "_hardware_targets_for_plan",
        lambda *_args, **_kwargs: [{
            "id": "remote",
            "accelerator": "hailo8",
            "remote": {"host": "192.0.2.1", "user": "nx"},
        }],
    )
    monkeypatch.setattr(
        remote_run,
        "probe_remote_trt_artifact_cache",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("local run opened a remote cache probe"),
        ),
    )

    artifacts, metrics, _message, status = (
        runner._stage_artifact_cache_preflight([{"id": "resnet50"}])
    )

    assert status == "ok"
    assert metrics["runtime_dispatch_allowed"] is True
    remote = json.loads(
        artifacts["remote_trt_cache_preflight_json"].read_text()
    )
    assert remote["probe_count"] == 0


def test_one_model_probe_setup_failure_does_not_skip_later_model(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from onnx_splitpoint_tool.workflow import runner as runner_module

    runner = _bare_runner()
    runner.run_dir = tmp_path
    runner.profile_payload = {"targets": ["tensorrt"]}
    runner.options = SimpleNamespace(benchmark_execution_backend="local")
    runner.outputs = {}
    runner.report_paths = []
    runner._targets = lambda: ["tensorrt"]
    runner._emit_log = lambda _message: None
    runner._profile_with_cli_hardware_overrides = lambda: {}
    runner._task_for = lambda _row: "classification"
    good = tmp_path / "models/good/benchmark_set"
    good.mkdir(parents=True)
    (good / "benchmark_set.json").write_text(json.dumps({
        "model_id": "good", "model": "model.onnx", "cases": [],
    }))
    (good / "benchmark_plan.json").write_text(json.dumps({
        "runs": [{"id": "ort_tensorrt"}],
    }))
    (good / "model.onnx").write_bytes(b"model")
    resolutions: list[str] = []

    def resolve(*, run_dir, model_id):
        del run_dir
        resolutions.append(model_id)
        if model_id == "bad":
            raise RuntimeError("bad suite")
        return good

    monkeypatch.setattr(
        runner_module, "resolve_generated_benchmark_suite_dir", resolve,
    )

    _artifacts, metrics, _message, status = (
        runner._stage_artifact_cache_preflight([
            {"id": "bad"}, {"id": "good"},
        ])
    )

    assert status == "ok"
    assert "good" in resolutions
    assert metrics["collection_error_count"] >= 1


@pytest.mark.parametrize(
    ("profile", "plan_rows", "expected_run_ids"),
    [
        (
            {
                "quality_gate": {
                    "statistics": {
                        "execution_location": "central_management",
                    },
                },
            },
            [
                {"id": "hailo8_to_trt", "stage1": "hailo8", "stage2": "tensorrt"},
                {"id": "ort_tensorrt", "provider": "tensorrt"},
            ],
            ["hailo8_to_trt", "ort_tensorrt"],
        ),
        (
            {},
            [{
                "id": "ort_tensorrt",
                "execution_scope": "full_only",
                "execution_role": "full_quality_only",
                "performance_claims_emitted": False,
            }],
            [],  # Incomplete Full-only contract also blocks in real dispatch.
        ),
    ],
)
def test_remote_preflight_run_ids_mirror_quality_dispatch(
    tmp_path, monkeypatch: pytest.MonkeyPatch, profile, plan_rows,
    expected_run_ids,
) -> None:
    """Normal central Quality retains vendor P2; Full-only does not invent it."""
    from onnx_splitpoint_tool.benchmark import remote_run
    from onnx_splitpoint_tool.workflow import execution_binding

    runner = _bare_runner()
    runner.run_dir = tmp_path
    runner.profile_payload = profile
    runner.options = SimpleNamespace(benchmark_execution_backend="remote")
    runner.outputs = {}
    runner.report_paths = []
    runner._targets = lambda: ["hailo8", "tensorrt"]
    runner._emit_log = lambda _message: None
    runner._profile_with_cli_hardware_overrides = lambda: profile
    runner._task_for = lambda _row: "detection"
    runner._materialize_local_cache_preflight_inputs = lambda *_args: []
    suite = tmp_path / "models/yolo/benchmark_set"
    suite.mkdir(parents=True)
    (suite / "benchmark_set.json").write_text(json.dumps({
        "model_id": "yolo", "model": "model.onnx",
        "cases": [{"id": "b024"}],
    }))
    (suite / "benchmark_plan.json").write_text(json.dumps({
        "runs": plan_rows,
    }))
    (suite / "model.onnx").write_bytes(b"model")
    monkeypatch.setattr(
        execution_binding,
        "_hardware_targets_for_plan",
        lambda *_args, **_kwargs: [{
            "id": "orin_nx_hailo8_01",
            "accelerator": "hailo8",
            "runtime": {},
        }],
    )
    captured: list[list[str]] = []

    def probe(**kwargs):
        captured.append(list(kwargs["active_run_ids"]))
        return {"status": "unknown", "observations": []}

    monkeypatch.setattr(remote_run, "probe_remote_trt_artifact_cache", probe)

    _artifacts, metrics, _message, status = (
        runner._stage_artifact_cache_preflight([{"id": "yolo"}])
    )

    assert status == "ok"
    assert captured == ([expected_run_ids] if expected_run_ids else [])
    assert metrics["collection_error_count"] == (0 if expected_run_ids else 1)


def test_parent_preflight_uses_same_h8_owner_h10_quality_companion_scope(tmp_path, monkeypatch):
    from onnx_splitpoint_tool.benchmark import remote_run
    from onnx_splitpoint_tool.workflow import execution_binding
    from tests.test_v2803_fix4_trt_dispatch_scope import _generic, _matrix, _real_suite

    targets = [
        {"id": "h8", "accelerator": "hailo8", "runtime": {}},
        {"id": "h10", "accelerator": "hailo10h", "runtime": {
            "add_args": "--quality-only-run-ids stale --run-ids wrong",
        }},
    ]
    vendor8 = dict(_matrix(), id="hailo8_to_tensorrt", stage1={"hw_arch": "hailo8"})
    plan_rows = [_generic(expected_setup_id="h8"), vendor8, _matrix()]
    generated = _real_suite(tmp_path, plan_rows)
    suite = tmp_path / "models/model/benchmark_set"
    suite.parent.mkdir(parents=True)
    generated.rename(suite)
    profile = {"quality_gate": {"statistics": {"execution_location": "central_management"}}}
    runner = _bare_runner()
    runner.run_dir = tmp_path
    runner.profile_payload = profile
    runner.options = SimpleNamespace(benchmark_execution_backend="remote")
    runner.outputs, runner.report_paths = {}, []
    runner._targets = lambda: ["tensorrt"]
    runner._emit_log = lambda _message: None
    runner._profile_with_cli_hardware_overrides = lambda: profile
    runner._task_for = lambda _row: "classification"
    runner._materialize_local_cache_preflight_inputs = lambda *_args: []
    monkeypatch.setattr(execution_binding, "_hardware_targets_for_plan", lambda *_args: targets)
    original_probe = remote_run.probe_remote_trt_artifact_cache
    captured = {}

    def probe(**kwargs):
        assert kwargs["transport"] is None  # No SSH or hardware in this fixture.
        report = original_probe(**kwargs)
        captured[kwargs["setup_id"]] = report
        return report

    monkeypatch.setattr(remote_run, "probe_remote_trt_artifact_cache", probe)
    _artifacts, metrics, _message, status = runner._stage_artifact_cache_preflight([{"id": "model"}])
    assert status == "ok" and metrics["collection_error_count"] == 0
    assert set(captured) == {"h8", "h10"}
    owner, companion = (captured[key]["requirement_plan"] for key in ("h8", "h10"))
    assert captured["h8"]["active_run_ids"] == ["hailo8_to_tensorrt", "ort_tensorrt"]
    assert captured["h10"]["active_run_ids"] == ["hailo10_to_tensorrt", "ort_tensorrt"]
    assert owner["full_required"] and companion["full_required"]
    assert owner["p1_cases"] == owner["generic_p2_cases"] == ["b024"]
    assert companion["p1_cases"] == companion["generic_p2_cases"] == []
    assert owner["p2_run_ids_by_case"] == {"b024": ["hailo8_to_tensorrt"]}
    assert companion["p2_run_ids_by_case"] == {"b024": ["hailo10_to_tensorrt"]}
