from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from onnx_splitpoint_tool.benchmark.hailo_policy import (
    build_case_hailo_variant_availability,
    case_hailo_backend_terminal_states,
    case_has_usable_hailo_variant,
    missing_case_hailo_requirements,
)
from onnx_splitpoint_tool.benchmark.services import (
    BenchmarkGenerationExecutionCallbacks,
    BenchmarkGenerationExecutionConfig,
    BenchmarkGenerationExecutionService,
    BenchmarkGenerationRuntime,
)
from onnx_splitpoint_tool.gui.controller import write_benchmark_suite_script
from onnx_splitpoint_tool.workflow.legacy_benchmarkset_binding import (
    _post_build_audit_minimum_projection,
    _require_complete_hailo_matrix_for_candidate_plan,
)


def _load_generated_suite(tmp_path: Path):
    script = Path(write_benchmark_suite_script(tmp_path))
    spec = importlib.util.spec_from_file_location(
        "benchmark_suite_v276_pipeline_repairs",
        script,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("runner_rc", "error_class", "failure_kind"),
    [
        (-11, "case_runner_signal_sigsegv", "signal"),
        (3, "case_runner_nonzero_exit", "nonzero_exit"),
    ],
)
def test_nonzero_runner_rc_overrides_success_report_without_mutating_raw_file(
    tmp_path: Path,
    runner_rc: int,
    error_class: str,
    failure_kind: str,
) -> None:
    suite = _load_generated_suite(tmp_path)
    case_dir = tmp_path / "b013"
    result_dir = case_dir / "results_hailo8_to_trt"
    result_dir.mkdir(parents=True)
    report_path = result_dir / "validation_report.json"
    raw_report = {
        "ok": True,
        "status": "ok",
        "buildable": True,
        "runtime_ok": True,
        "runtime_executable": True,
        "validation_ok": True,
        "final_pass": True,
        "final_pass_all": True,
        "interface_contract_pass": True,
        "accuracy_gate_policy_match": True,
        "task_quality_gate": {"decision": "pass"},
        "variant_status": {"part1": "ok", "composed": "ok"},
    }
    original_text = json.dumps(raw_report, indent=2, sort_keys=True) + "\n"
    report_path.write_text(original_text, encoding="utf-8")

    row = suite._collect_case_result(
        case_dir,
        result_dir.name,
        "hailo8_to_trt",
        "hailo8_to_trt",
        "auto",
        stage1="hailo8",
        stage2="tensorrt",
        variants=["part1", "composed"],
        runner_rc=runner_rc,
    )

    assert row is not None
    assert report_path.read_text(encoding="utf-8") == original_text
    assert row["runner_terminal_failure"] is True
    assert row["runner_returncode"] == runner_rc
    assert row["runner_failure_kind"] == failure_kind
    assert row["error_class"] == error_class
    assert row["runtime_ok"] is False
    assert row["execution_ok"] is False
    assert row["validation_ok"] is False
    assert row["final_pass_all"] is False
    assert row["ranking_eligible"] is False
    assert row["performance_eligible"] is False
    assert row["thesis_valid"] is False
    assert row["raw_report_variant_status"] == {
        "part1": "ok",
        "composed": "ok",
    }
    assert row["variant_status"] == {
        "part1": "runner_terminal_failure",
        "composed": "runner_terminal_failure",
    }


def test_zero_runner_rc_keeps_success_projection(tmp_path: Path) -> None:
    suite = _load_generated_suite(tmp_path)
    case_dir = tmp_path / "b044"
    result_dir = case_dir / "results_ort_tensorrt"
    result_dir.mkdir(parents=True)
    (result_dir / "validation_report.json").write_text(
        json.dumps({
            "buildable": True,
            "runtime_ok": True,
            "interface_contract_pass": True,
            "accuracy_gate_policy_match": True,
            "task_quality_gate": {"decision": "pass"},
        }),
        encoding="utf-8",
    )

    row = suite._collect_case_result(
        case_dir,
        result_dir.name,
        "ort_tensorrt",
        "ort_tensorrt",
        "auto",
        runner_rc=0,
    )

    assert row is not None
    assert row["execution_ok"] is True
    assert row["ranking_eligible"] is True
    assert "runner_terminal_failure" not in row


def _audit_plan_marker() -> dict:
    return {
        "candidate_universe_scope": "predeclared_audit_universe",
        "candidate_universe_mode": "deterministic_audit",
        "score_independent": True,
        "selection_uses_predictions": False,
        "selection_uses_measurements": False,
    }


def test_formal_audit_retains_backend_partial_cases_but_normal_generation_is_strict() -> None:
    assert _require_complete_hailo_matrix_for_candidate_plan(
        _audit_plan_marker()
    ) is False
    assert _require_complete_hailo_matrix_for_candidate_plan({
        "candidate_universe_scope": "deployment_shortlist",
    }) is True


def test_backend_terminal_states_keep_success_and_failure_separate() -> None:
    runs = [
        {
            "id": "hailo8_to_trt",
            "type": "matrix",
            "stage1": {"type": "hailo", "hw_arch": "hailo8"},
            "stage2": {"type": "onnxruntime", "provider": "tensorrt"},
            "variants": ["part1", "composed"],
        },
        {
            "id": "hailo10_to_trt",
            "type": "matrix",
            "stage1": {"type": "hailo", "hw_arch": "hailo10h"},
            "stage2": {"type": "onnxruntime", "provider": "tensorrt"},
            "variants": ["part1", "composed"],
        },
    ]
    states = case_hailo_backend_terminal_states(
        runs,
        {
            "hailo8": {
                "part1": True,
                "part1_failed": False,
                "part1_error": "",
            },
            "hailo10h": {
                "part1": False,
                "part1_failed": True,
                "part1_error": "'base_conv24' is not in list",
            },
        },
    )

    assert states == [
        {
            "hw_arch": "hailo10h",
            "variant": "part1",
            "status": "terminal_failed",
            "terminal": True,
            "available": False,
            "reason": "'base_conv24' is not in list",
        },
        {
            "hw_arch": "hailo8",
            "variant": "part1",
            "status": "ready",
            "terminal": True,
            "available": True,
            "reason": "artifact_available",
        },
    ]


@pytest.mark.parametrize("plan_hw", ["hailo10", "hailo10h"])
@pytest.mark.parametrize("evidence_hw", ["hailo10", "hailo10h"])
def test_hailo10_policy_aliases_share_one_physical_hailo10h_state(
    plan_hw: str,
    evidence_hw: str,
) -> None:
    runs = [{
        "id": "hailo10_to_trt",
        "type": "matrix",
        "stage1": {"type": "hailo", "hw_arch": plan_hw},
        "stage2": {"type": "onnxruntime", "provider": "tensorrt"},
        "variants": ["part1", "composed"],
    }]
    availability = {
        evidence_hw: {
            "part1": True,
            "part1_failed": False,
            "part1_error": "",
        },
    }

    assert case_has_usable_hailo_variant(runs, availability) is True
    assert missing_case_hailo_requirements(runs, availability) == []
    assert case_hailo_backend_terminal_states(runs, availability) == [{
        "hw_arch": "hailo10h",
        "variant": "part1",
        "status": "ready",
        "terminal": True,
        "available": True,
        "reason": "artifact_available",
    }]


def test_hailo10_policy_availability_physically_merges_legacy_alias_keys() -> None:
    availability = build_case_hailo_variant_availability(
        {
            "hailo10": {
                "full": "hailo/hailo10h/full/compiled.hef",
            },
        },
        {
            "hailo10h": {
                "part1": "hailo/hailo10h/part1/compiled.hef",
            },
        },
    )

    assert list(availability) == ["hailo10h"]
    assert availability["hailo10h"]["full"] is True
    assert availability["hailo10h"]["part1"] is True


def test_post_build_minimum_counts_audit_only_and_never_backfills() -> None:
    plan = {
        "minimum_valid_audit_candidates": 3,
        "audit_candidates": [
            {"case_id": f"b{boundary:03d}", "boundary": boundary}
            for boundary in range(1, 6)
        ],
        "deployment_shortlist": [
            {"case_id": "b090", "boundary": 90},
        ],
    }
    projection = _post_build_audit_minimum_projection(
        plan,
        [("b001", 1), ("b002", 2), ("b090", 90)],
    )

    assert projection["post_build_audit_materialized_count"] == 2
    assert projection["post_build_minimum_required"] == 3
    assert projection["post_build_minimum_status"] == "shortfall"
    assert projection["execution_status"] == (
        "insufficient_materialized_audit_candidates"
    )
    assert projection["repair_required"] is True
    assert projection["repair_status"] == "required_no_automatic_backfill"
    assert projection["automatic_post_build_backfill_performed"] is False
    assert projection["automatic_post_build_backfill_allowed"] is False


def test_base_conv_retry_resolves_all_declared_graph_outputs_from_graph_fixture() -> None:
    node_a = SimpleNamespace(name="splitpoint_identity_0", output=["cut_a"])
    node_b = SimpleNamespace(
        name="splitpoint_identity_1",
        output=["cut_b", "cut_c"],
    )
    model = SimpleNamespace(graph=SimpleNamespace(
        node=[node_a, node_b],
        output=[
            SimpleNamespace(name="cut_a"),
            SimpleNamespace(name="cut_b"),
            SimpleNamespace(name="cut_c"),
        ],
    ))

    assert BenchmarkGenerationExecutionService._graph_output_producer_node_names(
        model
    ) == ["splitpoint_identity_0", "splitpoint_identity_1"]
    service = BenchmarkGenerationExecutionService()
    yolo26_cfg = SimpleNamespace(base="yolo26s")
    for base_conv in (14, 24, 50):
        assert service._is_hailo_base_conv_resolution_failure(
            SimpleNamespace(error=f"'base_conv{base_conv}' is not in list"),
            cfg=yolo26_cfg,
        ) is True
    assert service._is_hailo_base_conv_resolution_failure(
        SimpleNamespace(error=(
            "'base_conv24' is not in list\n\n"
            "[debug_log] /tmp/hailo_venv_hef_fail_hailo10_b104.log\n"
            "Details were written to gui.log (Logs tab)."
        )),
        cfg=yolo26_cfg,
    ) is True


def test_base_conv_retry_refuses_partial_or_unnamed_endpoint_resolution() -> None:
    model_missing = SimpleNamespace(graph=SimpleNamespace(
        node=[SimpleNamespace(name="producer_a", output=["cut_a"])],
        output=[SimpleNamespace(name="cut_a"), SimpleNamespace(name="cut_b")],
    ))
    model_unnamed = SimpleNamespace(graph=SimpleNamespace(
        node=[SimpleNamespace(name="", output=["cut_a"])],
        output=[SimpleNamespace(name="cut_a")],
    ))

    resolver = (
        BenchmarkGenerationExecutionService
        ._graph_output_producer_node_names
    )
    assert resolver(model_missing) == []
    assert resolver(model_unnamed) == []
    service = BenchmarkGenerationExecutionService()
    yolo26_cfg = SimpleNamespace(base="yolo26s")
    assert service._is_hailo_base_conv_resolution_failure(
        SimpleNamespace(error="feature splitter output shape is ambiguous"),
        cfg=yolo26_cfg,
    ) is False
    assert service._is_hailo_base_conv_resolution_failure(
        SimpleNamespace(error=(
            "'base_conv24' is not in list\n"
            "unexpected second failure"
        )),
        cfg=yolo26_cfg,
    ) is False
    assert service._is_hailo_base_conv_resolution_failure(
        SimpleNamespace(error="'base_conv99' is not in list"),
        cfg=yolo26_cfg,
    ) is False
    assert service._is_hailo_base_conv_resolution_failure(
        SimpleNamespace(error="'base_conv24' is not in list"),
        cfg=SimpleNamespace(base="resnet50"),
    ) is False
    assert service._is_hailo_base_conv_resolution_failure(
        SimpleNamespace(error="'base_conv24' is not in list"),
        cfg=yolo26_cfg,
        boundary=66,
    ) is False
    assert service._is_hailo_base_conv_resolution_failure(
        SimpleNamespace(error="'base_conv14' is not in list"),
        cfg=yolo26_cfg,
        boundary=66,
    ) is True


def test_yolo26_partial_split_output_graph_refuses_explicit_retry() -> None:
    onnx = pytest.importorskip("onnx")
    helper = onnx.helper
    tensor = onnx.TensorProto
    graph = helper.make_graph(
        [
            helper.make_node(
                "Split",
                ["input"],
                ["split_a", "split_b"],
                name="/model.6/Split",
                axis=1,
            ),
            helper.make_node(
                "Relu",
                ["split_b"],
                ["relu_out"],
                name="/model.6/Relu",
            ),
        ],
        "yolo26_part1_split_fixture",
        [helper.make_tensor_value_info("input", tensor.FLOAT, [1, 4, 2, 2])],
        [
            helper.make_tensor_value_info("split_a", tensor.FLOAT, [1, 2, 2, 2]),
            helper.make_tensor_value_info("relu_out", tensor.FLOAT, [1, 2, 2, 2]),
        ],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 13)],
    )
    service = BenchmarkGenerationExecutionService()

    repaired, meta = service._materialize_hailo_part1_split_outputs(
        model,
        cfg=SimpleNamespace(base="yolo26s", benchmark_task="detection"),
        boundary=66,
    )
    endpoints = service._graph_output_producer_node_names(
        repaired,
        collapse_tool_identities=True,
        boundary=66,
        identity_fix=meta,
    )
    archived_endpoints = service._graph_output_producer_node_names(
        repaired,
        collapse_tool_identities=False,
    )

    assert meta["applied"] is True
    assert meta["materialized_original_outputs"] == ["split_a"]
    assert endpoints == []
    assert archived_endpoints == [
        "splitpoint_hailo_part1_output_identity_b66_0",
        "/model.6/Relu",
    ]
    onnx.checker.check_model(repaired)


def test_yolo26_complete_split_outputs_have_slot_safe_retry_projection() -> None:
    onnx = pytest.importorskip("onnx")
    helper = onnx.helper
    tensor = onnx.TensorProto
    graph = helper.make_graph(
        [
            helper.make_node(
                "Split",
                ["input"],
                ["split_a", "split_b"],
                name="/model.6/Split",
                axis=1,
            ),
        ],
        "yolo26_complete_split_fixture",
        [helper.make_tensor_value_info("input", tensor.FLOAT, [1, 4, 2, 2])],
        [
            helper.make_tensor_value_info("split_a", tensor.FLOAT, [1, 2, 2, 2]),
            helper.make_tensor_value_info("split_b", tensor.FLOAT, [1, 2, 2, 2]),
        ],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 13)],
    )
    service = BenchmarkGenerationExecutionService()
    repaired, meta = service._materialize_hailo_part1_split_outputs(
        model,
        cfg=SimpleNamespace(base="yolo26s", benchmark_task="detection"),
        boundary=66,
    )

    projection = service._graph_output_endpoint_projection(
        repaired,
        collapse_tool_identities=True,
        boundary=66,
        identity_fix=meta,
    )

    assert projection["end_node_names"] == ["/model.6/Split"]
    assert projection["expanded_output_tensors"] == ["split_a", "split_b"]
    assert [
        item["effective_producer_output_slot"]
        for item in projection["outputs"]
    ] == [0, 1]
    assert [item["graph_output_slot"] for item in projection["outputs"]] == [0, 1]


def test_yolo26_reordered_split_outputs_refuse_explicit_retry() -> None:
    onnx = pytest.importorskip("onnx")
    helper = onnx.helper
    tensor = onnx.TensorProto
    graph = helper.make_graph(
        [helper.make_node(
            "Split",
            ["input"],
            ["split_a", "split_b"],
            name="/model.6/Split",
            axis=1,
        )],
        "yolo26_reordered_split_fixture",
        [helper.make_tensor_value_info("input", tensor.FLOAT, [1, 4, 2, 2])],
        [
            helper.make_tensor_value_info("split_b", tensor.FLOAT, [1, 2, 2, 2]),
            helper.make_tensor_value_info("split_a", tensor.FLOAT, [1, 2, 2, 2]),
        ],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 13)],
    )
    service = BenchmarkGenerationExecutionService()
    repaired, meta = service._materialize_hailo_part1_split_outputs(
        model,
        cfg=SimpleNamespace(base="yolo26s", benchmark_task="detection"),
        boundary=66,
    )

    assert service._graph_output_endpoint_projection(
        repaired,
        collapse_tool_identities=True,
        boundary=66,
        identity_fix=meta,
    ) == {}


def test_endpoint_projection_rejects_duplicate_tensor_producers() -> None:
    model = SimpleNamespace(
        graph=SimpleNamespace(
            node=[
                SimpleNamespace(name="producer_a", output=["cut"]),
                SimpleNamespace(name="producer_b", output=["cut"]),
            ],
            output=[SimpleNamespace(name="cut")],
        )
    )

    assert (
        BenchmarkGenerationExecutionService
        ._graph_output_producer_node_names(model)
    ) == []


@pytest.mark.parametrize(
    ("first_error", "second_error", "expected_calls"),
    [
        (None, None, 1),
        ("'base_conv14' is not in list", None, 2),
        ("'base_conv99' is not in list", None, 1),
        ("'base_conv14' is not in list", "'base_conv14' is not in list", 2),
    ],
)
def test_production_yolo26_builder_dispatches_at_most_one_exact_endpoint_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    first_error: str | None,
    second_error: str | None,
    expected_calls: int,
) -> None:
    onnx = pytest.importorskip("onnx")
    helper = onnx.helper
    tensor = onnx.TensorProto
    from onnx_splitpoint_tool import api as asc

    p1 = helper.make_model(
        helper.make_graph(
            [helper.make_node(
                "Split",
                ["input"],
                ["split_a", "split_b"],
                name="/model.6/Split",
                axis=1,
            )],
            "production_retry_part1",
            [helper.make_tensor_value_info("input", tensor.FLOAT, [1, 4])],
            [
                helper.make_tensor_value_info("split_a", tensor.FLOAT, [1, 2]),
                helper.make_tensor_value_info("split_b", tensor.FLOAT, [1, 2]),
            ],
        ),
        opset_imports=[helper.make_opsetid("", 13)],
    )
    p2 = helper.make_model(
        helper.make_graph(
            [helper.make_node("Relu", ["cut"], ["output"], name="tail")],
            "production_retry_part2",
            [helper.make_tensor_value_info("cut", tensor.FLOAT, [1, 2])],
            [helper.make_tensor_value_info("output", tensor.FLOAT, [1, 2])],
        ),
        opset_imports=[helper.make_opsetid("", 13)],
    )
    monkeypatch.setattr(
        asc,
        "cut_tensors_for_boundary",
        lambda *_args, **_kwargs: ["split_a", "split_b"],
    )
    monkeypatch.setattr(
        asc,
        "split_model_on_cut_tensors",
        lambda *_args, **_kwargs: (
            p1,
            p2,
            {"cut_tensors": ["split_a", "split_b"]},
        ),
    )

    def _write_runner(case_dir, **_kwargs):
        runner = Path(case_dir) / "run_benchmark.py"
        runner.write_text("# generated test runner\n", encoding="utf-8")
        return str(runner)

    monkeypatch.setattr(asc, "write_runner_skeleton_onnxruntime", _write_runner)
    build_calls: list[dict] = []

    def _result(error: str | None, outdir: Path) -> SimpleNamespace:
        ok = error is None
        hef_path = None
        if ok:
            outdir.mkdir(parents=True, exist_ok=True)
            hef = outdir / "compiled.hef"
            hef.write_bytes(b"test-hef")
            hef_path = str(hef)
        return SimpleNamespace(
            ok=ok,
            skipped=False,
            timed_out=False,
            failure_kind=None,
            unsupported_reason=None,
            error=error,
            elapsed_s=0.01,
            context_count=1 if ok else None,
            detected={},
            process={},
            details={},
            calib_info={},
            hef_path=hef_path,
        )

    def _build(_model_path, **kwargs):
        build_calls.append(dict(kwargs))
        error = first_error if len(build_calls) == 1 else second_error
        return _result(error, Path(kwargs["outdir"]))

    out_dir = tmp_path / "suite"
    out_dir.mkdir()
    runtime = BenchmarkGenerationRuntime(
        out_dir=out_dir,
        bench_log_path=out_dir / "benchmark.log",
        state_path=out_dir / "generation_state.json",
        requested_cases=1,
        ranked_candidates=[66],
        candidate_search_pool=[66],
        model_name="yolo26s",
        model_source="yolo26s.onnx",
        hef_full_policy="skip",
    )
    cfg = BenchmarkGenerationExecutionConfig(
        runtime=runtime,
        target_cases=1,
        gap=0,
        ranked_candidates=[66],
        candidate_search_pool=[66],
        out_dir=out_dir,
        base="yolo26s",
        pad=3,
        strict_boundary=False,
        model=p1,
        nodes=list(p1.graph.node),
        order=[0],
        analysis_payload={},
        bench_plan_runs=[{
            "id": "hailo10_to_trt",
            "type": "matrix",
            "stage1": {"type": "hailo", "hw_arch": "hailo10"},
            "stage2": {"type": "tensorrt"},
            "variants": ["part1", "composed"],
        }],
        benchmark_task="detection",
        full_model_src="yolo26s.onnx",
        hef_targets=["hailo10"],
        hef_part1=True,
        hef_part2=False,
        hef_backend="venv",
        hef_wsl_venv="auto",
        hef_timeout_s=60,
        hailo_build_hef_fn=_build,
        hailo_salvage_enable=False,
        require_complete_hailo_matrix_per_case=False,
    )
    callbacks = BenchmarkGenerationExecutionCallbacks(
        log=lambda *_args, **_kwargs: None,
        queue_put=lambda *_args, **_kwargs: None,
        persist_state=lambda **_kwargs: None,
        publish_hailo_diagnostics=lambda *_args, **_kwargs: None,
        predicted_metrics_for_boundary=lambda *_args, **_kwargs: {},
        hailo_parse_entry_for_boundary=lambda *_args, **_kwargs: None,
        hailo_parse_scalar_fields=lambda *_args, **_kwargs: {},
    )

    BenchmarkGenerationExecutionService().execute_case_build_loop(cfg, callbacks)

    assert len(build_calls) == expected_calls
    assert "end_node_names" not in build_calls[0]
    if expected_calls == 2:
        assert build_calls[1]["end_node_names"] == ["/model.6/Split"]
    assert all("extra_model_script" not in call for call in build_calls)
