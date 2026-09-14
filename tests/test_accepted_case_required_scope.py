from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from onnx_splitpoint_tool.workflow.required_run_scope import (
    RequiredRunScopeError,
    audit_materialized_scope,
    build_global_required_run_scope,
    required_measurements_from_scope,
    seal_required_run_scope,
)
from onnx_splitpoint_tool.workflow.runner import (
    EvaluationWorkflowRunner,
    _accepted_case_contract_v27919,
    _benchmark_case_ids_v60r,
    _candidate_case_contract_v2792,
    _seal_contract_only_candidate_scope_v27919,
    _seal_model_scope_from_accepted_cases_v27919,
)
from onnx_splitpoint_tool.workflow import runner as runner_module


def _write_global_scope(root: Path, model_id: str) -> None:
    scope = build_global_required_run_scope(
        profile_id="case-scope-test",
        model_entries=[{"id": model_id, "task": "detection"}],
        effective_plan={
            "logical_run_profiles": ["hailo8", "hailo8_to_trt"],
            "effective_generic_run_ids": ["hailo8", "hailo8_to_trt"],
            "setup_groups": {
                "hailo8_setup": ["hailo8", "hailo8_to_trt"],
            },
        },
        hardware_targets=[{
            "id": "orin_nx_hailo8_01",
            "accelerator": "hailo8",
        }],
        created_at="before-build",
    )
    seal_required_run_scope(root / "required_run_scope.json", scope)


@pytest.mark.parametrize(
    ("model_id", "candidate_ids", "accepted_ids"),
    [
        ("yolo11l", ["b003", "b062", "b065"], ["b002", "b067"]),
        ("yolo26m", ["b038", "b041", "b398"], ["b002", "b072"]),
        (
            "yolo26s",
            ["b021", "b024", "b364"],
            ["b005", "b024", "b035"],
        ),
    ],
)
def test_required_scope_uses_final_accepted_backfill_cases(
    tmp_path: Path,
    model_id: str,
    candidate_ids: list[str],
    accepted_ids: list[str],
) -> None:
    _write_global_scope(tmp_path, model_id)
    candidate_contract = _candidate_case_contract_v2792({
        "selected_candidates": [
            {"case_id": case_id, "split_index": int(case_id[1:])}
            for case_id in candidate_ids
        ],
    })
    assert [
        row["case_id"] for row in candidate_contract["cases"]
    ] == candidate_ids

    benchmark_set = {
        "cases": [
            {"case_id": case_id, "boundary": int(case_id[1:])}
            for case_id in accepted_ids
        ],
        "rejected_cases": [
            {"case_id": case_id, "reason": "technical_reject"}
            for case_id in candidate_ids
            if case_id not in accepted_ids
        ],
    }
    scope, plan, _projection, accepted_contract = (
        _seal_model_scope_from_accepted_cases_v27919(
            run_dir=tmp_path,
            model_id=model_id,
            benchmark_plan_source={"runs": []},
            benchmark_set_contract=benchmark_set,
        )
    )

    assert accepted_contract["source"] == "benchmark_set.json:cases"
    assert [
        row["case_id"] for row in accepted_contract["cases"]
    ] == accepted_ids
    assert scope["accepted_case_ids"] == accepted_ids
    assert scope["case_scope_source"] == (
        "final_benchmark_set_accepted_cases"
    )
    assert scope["sealed_before_backend_compiler_dispatch"] is False
    assert (
        scope["sealed_after_case_acceptance_before_runtime_dispatch"] is True
    )
    assert scope["policy"][
        "rejected_candidate_is_not_a_missing_runtime_measurement"
    ] is True

    split_cases = {
        row["case_id"]
        for row in scope["identities"]
        if row["variant"] == "split"
    }
    assert split_cases == set(accepted_ids)
    assert split_cases.isdisjoint(set(candidate_ids) - set(accepted_ids))

    required = required_measurements_from_scope(scope)
    materialized = [dict(row) for row in required]
    audit = audit_materialized_scope(
        scope=scope,
        materialized_measurements=materialized,
    )
    assert audit["status"] == "match"
    assert audit["missing_from_materialized_count"] == 0
    assert audit["extra_in_materialized_count"] == 0
    assert {row["id"] for row in plan["runs"]} == {
        "hailo8", "hailo8_to_trt",
    }


def test_accepted_case_contract_deduplicates_without_reordering() -> None:
    contract = _accepted_case_contract_v27919({
        "cases": [
            {"id": "b067", "boundary": 67},
            {"case": "b002", "boundary": 2},
            {"case_id": "b067", "boundary": 67},
            "003",
        ],
    })
    assert [row["case_id"] for row in contract["cases"]] == [
        "b067", "b002", "b003",
    ]
    assert [row["boundary"] for row in contract["cases"]] == [67, 2, 3]
    assert _benchmark_case_ids_v60r({
        "cases": [{"id": "b001"}, {"case": "b002"}, "003"],
    }) == ["b001", "b002", "b003"]
    assert _benchmark_case_ids_v60r(contract) == ["b067", "b002", "b003"]


def test_contract_only_scope_is_explicitly_non_runtime(
    tmp_path: Path,
) -> None:
    model_id = "yolo26s"
    _write_global_scope(tmp_path, model_id)
    plan, projection = runner_module._authoritative_benchmark_plan_v2792(
        tmp_path, {"runs": []}, model_id=model_id,
    )
    candidate_contract = _candidate_case_contract_v2792({
        "selected_candidates": [{"case_id": "b024", "split_index": 24}],
    })
    scope = _seal_contract_only_candidate_scope_v27919(
        run_dir=tmp_path,
        model_id=model_id,
        benchmark_plan=plan,
        projection=projection,
        candidate_case_contract=candidate_contract,
    )

    assert scope["contract_only"] is True
    assert scope["runtime_dispatch_allowed"] is False
    assert scope["case_scope_source"] == (
        "candidate_plan_contract_only_no_runtime"
    )
    assert (
        scope["sealed_after_case_acceptance_before_runtime_dispatch"] is False
    )


def test_empty_accepted_case_list_blocks_only_model_scope(
    tmp_path: Path,
) -> None:
    model_id = "yolov7"
    _write_global_scope(tmp_path, model_id)

    with pytest.raises(RequiredRunScopeError, match="accepted_case_scope_empty"):
        _seal_model_scope_from_accepted_cases_v27919(
            run_dir=tmp_path,
            model_id=model_id,
            benchmark_plan_source={"runs": []},
            benchmark_set_contract={"cases": []},
        )

    assert not (
        tmp_path / "models" / model_id / "benchmark_set"
        / "required_run_scope.json"
    ).exists()
    # The campaign-wide model/run/setup contract remains intact; only this
    # model-local executable-case scope was rejected.
    assert (tmp_path / "required_run_scope.json").is_file()


def test_accepted_scope_is_immutable_before_runtime_dispatch(
    tmp_path: Path,
) -> None:
    model_id = "yolo26m"
    _write_global_scope(tmp_path, model_id)
    common = {
        "run_dir": tmp_path,
        "model_id": model_id,
        "benchmark_plan_source": {"runs": []},
    }
    _seal_model_scope_from_accepted_cases_v27919(
        **common,
        benchmark_set_contract={
            "cases": [{"case_id": "b002", "boundary": 2}],
        },
    )

    with pytest.raises(RequiredRunScopeError, match="immutable_mismatch"):
        _seal_model_scope_from_accepted_cases_v27919(
            **common,
            benchmark_set_contract={
                "cases": [{"case_id": "b072", "boundary": 72}],
            },
        )


def test_generation_stage_seals_runtime_scope_from_selected_suite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_id = "yolo26s"
    _write_global_scope(tmp_path, model_id)
    model_dir = tmp_path / "models" / model_id
    analysis_dir = model_dir / "analysis"
    analysis_dir.mkdir(parents=True)
    (analysis_dir / "prediction.json").write_text("{}", encoding="utf-8")
    (analysis_dir / "final_candidate_plan.json").write_text(
        json.dumps({
            "selected_candidates": [
                {"case_id": "b021", "split_index": 21},
                {"case_id": "b024", "split_index": 24},
                {"case_id": "b364", "split_index": 364},
            ],
        }),
        encoding="utf-8",
    )

    def fake_materialize(**_kwargs: object) -> SimpleNamespace:
        bdir = model_dir / "benchmark_set"
        suite = bdir / "legacy_suite"
        accepted = [
            {"case_id": "b005", "boundary": 5},
            {"case_id": "b024", "boundary": 24},
            {"case_id": "b035", "boundary": 35},
        ]
        plan = {"runs": [
            {"id": "hailo8"}, {"id": "hailo8_to_trt"},
        ]}
        for root in (bdir, suite):
            root.mkdir(parents=True, exist_ok=True)
            (root / "benchmark_set.json").write_text(
                json.dumps({"cases": accepted}), encoding="utf-8",
            )
            (root / "benchmark_plan.json").write_text(
                json.dumps(plan), encoding="utf-8",
            )
            (root / "benchmark_suite.py").write_text(
                "# executable harness\n", encoding="utf-8",
            )
            for row in accepted:
                (root / row["case_id"]).mkdir(exist_ok=True)
        return SimpleNamespace(
            artifacts={
                "benchmark_set_json": bdir / "benchmark_set.json",
                "legacy_suite_benchmark_set_json": (
                    suite / "benchmark_set.json"
                ),
            },
            metrics={"accepted_cases": 3, "rejected_cases": 2},
            message="fake generator complete",
            status="ok",
        )

    monkeypatch.setattr(
        runner_module, "materialize_legacy_benchmark_set", fake_materialize,
    )
    workflow = object.__new__(EvaluationWorkflowRunner)
    workflow.run_dir = tmp_path
    workflow.profile_id = "case-scope-test"
    workflow.run_id = "case-scope-test-run"
    workflow.profile_payload = {
        "benchmark_generator": {"mode": "legacy"},
    }
    workflow.options = SimpleNamespace(
        dry_run=False,
        no_remote=False,
        require_fresh_run=True,
    )
    workflow.manifest = {}
    workflow.warnings = []
    workflow.log = lambda _message: None
    workflow._targets = lambda: [{
        "id": "orin_nx_hailo8_01", "accelerator": "hailo8",
    }]
    workflow._execution_mode = lambda: "generate_and_run"
    workflow._cleanup_formal_direct_suite_artifacts = (
        lambda _model_id: None
    )

    artifacts, metrics, _message, status = (
        workflow._stage_generate_benchmark_set(
            model_id, {"id": model_id, "resolved_path": "/unused.onnx"},
        )
    )

    assert status == "ok"
    assert metrics["required_run_scope_sealed_after_case_acceptance"] is True
    assert metrics["required_run_scope_sealed_before_runtime_dispatch"] is True
    assert metrics["accepted_case_count"] == 3
    assert Path(artifacts["required_run_scope_json"]).is_file()
    scope = json.loads(Path(artifacts["required_run_scope_json"]).read_text())
    assert scope["accepted_case_ids"] == ["b005", "b024", "b035"]
    assert {
        row["case_id"]
        for row in scope["identities"]
        if row["variant"] == "split"
    } == {"b005", "b024", "b035"}
