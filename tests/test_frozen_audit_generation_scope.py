from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from onnx_splitpoint_tool.campaign import (
    create_candidate_universe_manifest,
    stable_candidate_identity,
)
from onnx_splitpoint_tool.workflow.artifacts import (
    sha256_file,
    sha256_json,
    sha256_payload,
    write_json,
)
from onnx_splitpoint_tool.workflow.legacy_benchmarkset_binding import (
    _native_capability_config,
    _native_split_requires_single_part2_input,
    _resolve_generation_candidate_scope,
    reconcile_candidate_plan_after_generation,
)


def _candidate(boundary: int, *, role: str = "") -> dict:
    row = {
        "case_id": f"b{boundary:03d}",
        "boundary": boundary,
        "split_index": boundary,
        "rank": boundary,
        "strict_ok": True,
        "part2_input_count": 1,
    }
    if role:
        row["candidate_execution_roles"] = [role]
    return row


def test_native_capability_uses_logical_split_plan_not_adapter_inventory() -> None:
    options = SimpleNamespace(
        native_producer_enabled=False,
        native_producer_backends=None,
    )
    full_only_profile = {
        "run_profiles": [{
            "id": "trt_full",
            "type": "same_backend_reference",
            "full": "tensorrt",
            "stage1": "tensorrt",
            "stage2": "tensorrt",
            "enabled": True,
        }],
        "native_producers": {
            "enabled": True,
            "backends": ["hailo8", "deepx_m1"],
        },
    }
    full_only_cfg = _native_capability_config(full_only_profile, options)
    assert full_only_cfg["split_backends"] == []
    assert _native_split_requires_single_part2_input(full_only_cfg) is False

    split_profile = copy.deepcopy(full_only_profile)
    split_profile["run_profiles"] = [{
        "id": "hailo8_to_tensorrt",
        "stage1": "hailo8",
        "stage2": "tensorrt",
        "enabled": True,
    }]
    split_cfg = _native_capability_config(split_profile, options)
    assert split_cfg["split_backends"] == ["hailo8"]
    assert _native_split_requires_single_part2_input(split_cfg) is True


def _candidate_plan_artifact_id(selected: list[dict]) -> str:
    return f"candidate_plan_resnet50_{sha256_payload(selected)[:12]}"


def _prediction_artifact_id(candidates: list[dict]) -> str:
    return f"prediction_resnet50_{sha256_payload(candidates)[:12]}"


def _frozen_audit_plan(tmp_path: Path) -> dict:
    model_path = tmp_path / "resnet50.onnx"
    model_path.write_bytes(b"real-model-identity-for-audit-binding")
    model_sha256 = sha256_file(model_path) or ""
    all_candidates = [_candidate(boundary) for boundary in range(1, 31)]

    # First determine the score-independent audit identities, then arrange the
    # predictor order so its authoritative top four contain two audit overlaps
    # and two deployment-only cases, matching the real runner shape.
    _unused_json, _unused_csv, provisional_universe = (
        create_candidate_universe_manifest(
            model_id="resnet50",
            candidates=all_candidates,
            mode="deterministic_audit",
            output_dir=tmp_path,
            audit_size=20,
            minimum_valid_candidates=10,
            seed=20260710,
            source_prediction_sha256="sha256:provisional",
            identity_context={"model_sha256": model_sha256},
            write_artifacts=False,
        )
    )
    audit_case_ids = [str(value) for value in provisional_universe["selected_case_ids"]]
    non_audit_case_ids = [
        str(row["case_id"])
        for row in all_candidates
        if str(row["case_id"]) not in set(audit_case_ids)
    ]
    predictor_case_order = [
        audit_case_ids[0],
        audit_case_ids[1],
        non_audit_case_ids[0],
        non_audit_case_ids[1],
    ]
    predictor_case_order.extend(
        str(row["case_id"])
        for row in all_candidates
        if str(row["case_id"]) not in set(predictor_case_order)
    )
    raw_by_case = {str(row["case_id"]): dict(row) for row in all_candidates}
    prediction_candidates = []
    for rank, case_id in enumerate(predictor_case_order, start=1):
        prediction_candidates.append({**raw_by_case[case_id], "rank": rank})
    prediction = {
        "schema": "onnx-splitpoint/split-prediction",
        "schema_version": 1,
        "model_id": "resnet50",
        "requested_cases": 4,
        "min_gap": 0,
        "selection_strategy": "score_independent_audit",
        "require_single_part2_input": False,
        "requested_require_single_part2_input": False,
        "native_split_requires_single_part2_input": True,
        "effective_require_single_part2_input": True,
        "artifact_id": _prediction_artifact_id(prediction_candidates),
        "candidates": prediction_candidates,
    }
    prediction_path = write_json(tmp_path / "prediction.json", prediction)
    prediction_sha256 = sha256_file(prediction_path) or ""
    universe_path, _universe_csv, universe = create_candidate_universe_manifest(
        model_id="resnet50",
        candidates=prediction_candidates,
        mode="deterministic_audit",
        output_dir=tmp_path,
        audit_size=20,
        minimum_valid_candidates=10,
        seed=20260710,
        source_prediction_sha256=prediction_sha256,
        identity_context={"model_sha256": model_sha256},
        write_artifacts=True,
    )
    universe_by_case = {
        str(row["case_id"]): dict(row)
        for row in list(universe["candidates"])
    }
    prediction_by_case = {
        str(row["case_id"]): dict(row) for row in prediction_candidates
    }
    audit = []
    for rank, case_id in enumerate(universe["selected_case_ids"], start=1):
        row = {
            **prediction_by_case[str(case_id)],
            "candidate_id": universe_by_case[str(case_id)]["candidate_id"],
            "candidate_identity_sha256": universe_by_case[str(case_id)]["candidate_identity_sha256"],
            "candidate_execution_roles": ["audit"],
            "origin": "score_independent_audit",
            "source_rank": prediction_by_case[str(case_id)]["rank"],
            "audit_rank": rank,
        }
        audit.append(row)

    # Twenty prospectively selected audit cases plus four deployment cases.
    # Two deployment cases overlap the audit, so the execution union has 22.
    deployment_case_ids = predictor_case_order[:4]
    deployment = []
    for rank, case_id in enumerate(deployment_case_ids, start=1):
        deployment.append({
            **prediction_by_case[case_id],
            "candidate_id": universe_by_case[case_id]["candidate_id"],
            "candidate_identity_sha256": universe_by_case[case_id]["candidate_identity_sha256"],
            "candidate_execution_roles": ["deployment_shortlist"],
            "origin": "deployment_shortlist",
            "source_rank": prediction_by_case[case_id]["rank"],
            "deployment_rank": rank,
        })

    selected_by_identity = {}
    for row in audit + deployment:
        identity = (str(row["case_id"]), int(row["boundary"]))
        if identity not in selected_by_identity:
            selected_by_identity[identity] = copy.deepcopy(row)
            continue
        merged = selected_by_identity[identity]
        merged["candidate_execution_roles"] = [
            "audit",
            "deployment_shortlist",
        ]
        merged["deployment_rank"] = row["deployment_rank"]
    selected = list(selected_by_identity.values())
    expected_boundaries = [int(row["boundary"]) for row in selected]
    plan = {
        "schema": "onnx-splitpoint/final-candidate-plan",
        "schema_version": 4,
        "model_id": "resnet50",
        "source_prediction_artifact_id": prediction["artifact_id"],
        "requested_cases": 4,
        "min_gap": 0,
        "selection_strategy": "score_independent_audit",
        "require_single_part2_input": False,
        "requested_require_single_part2_input": False,
        "native_split_requires_single_part2_input": True,
        "effective_require_single_part2_input": True,
        "candidate_universe_mode": universe["mode"],
        "candidate_universe_scope": universe["claim_scope"],
        "candidate_universe_sha256": universe["universe_sha256"],
        "minimum_valid_audit_candidates": 10,
        "score_independent": True,
        "selection_uses_predictions": False,
        "selection_uses_measurements": False,
        "audit_candidates": audit,
        "deployment_shortlist": deployment,
        "selected_candidates": selected,
    }
    plan["artifact_id"] = _candidate_plan_artifact_id(selected)
    selection_policy = {
        "selection_strategy": "score_independent_audit",
        "max_accepted_cases_per_model": 4,
        "min_gap": 0,
    }
    selection_input = {
        "schema": "onnx-splitpoint/candidate-selection-input",
        "schema_version": 3,
        "model_id": "resnet50",
        "source_prediction_artifact_id": prediction["artifact_id"],
        "candidate_universe_sha256": universe["universe_sha256"],
        "requested_cases": 4,
        "min_gap": 0,
        "selection_strategy": "score_independent_audit",
        "deployment_shortlist_count": 4,
        "eligible_candidate_count": len(prediction_candidates),
        "require_single_part2_input": False,
        "requested_require_single_part2_input": False,
        "native_split_requires_single_part2_input": True,
        "effective_require_single_part2_input": True,
        "profile_selection_policy": selection_policy,
    }
    selection_input_path = write_json(
        tmp_path / "selection_input.json",
        selection_input,
    )
    return {
        "plan": plan,
        "prediction": prediction,
        "expected_boundaries": expected_boundaries,
        "universe_path": universe_path,
        "prediction_path": prediction_path,
        "selection_input_path": selection_input_path,
        "model_path": model_path,
        "selection_policy": selection_policy,
        "selection_strategy": "score_independent_audit",
        "native_single_part2_input": True,
    }


def _binding_kwargs(fixture: dict) -> dict:
    return {
        "candidate_universe_path": fixture["universe_path"],
        "prediction_path": fixture["prediction_path"],
        "selection_input_path": fixture["selection_input_path"],
        "model_path": fixture["model_path"],
        "expected_selection_policy": fixture["selection_policy"],
        "expected_selection_strategy": fixture["selection_strategy"],
        "expected_native_single_part2_input": fixture[
            "native_single_part2_input"
        ],
    }


def _rebuild_selected_union(plan: dict) -> None:
    selected_by_identity = {}
    order_policy = str(
        plan.get("execution_union_order_policy")
        or "audit_then_deployment"
    )
    if order_policy == "forced_deployment_then_audit":
        ordered = list(plan["deployment_shortlist"]) + list(
            plan["audit_candidates"]
        )
    else:
        ordered = list(plan["audit_candidates"]) + list(
            plan["deployment_shortlist"]
        )
    for row in ordered:
        identity = (str(row["case_id"]), int(row["boundary"]))
        if identity not in selected_by_identity:
            selected_by_identity[identity] = copy.deepcopy(row)
            continue
        merged = selected_by_identity[identity]
        roles = list(merged.get("candidate_execution_roles") or [])
        for role in list(row.get("candidate_execution_roles") or []):
            if role not in roles:
                roles.append(role)
        merged["candidate_execution_roles"] = roles
        if row.get("audit_rank") is not None:
            merged["audit_rank"] = row["audit_rank"]
        if row.get("deployment_rank") is not None:
            merged["deployment_rank"] = row["deployment_rank"]
    plan["selected_candidates"] = list(selected_by_identity.values())
    plan["artifact_id"] = _candidate_plan_artifact_id(
        plan["selected_candidates"]
    )


def _force_deployment_anchor(fixture: dict) -> tuple[str, int]:
    plan = fixture["plan"]
    audit_case_ids = {
        str(row["case_id"])
        for row in plan["audit_candidates"]
    }
    anchor = next(
        copy.deepcopy(row)
        for row in plan["deployment_shortlist"]
        if str(row["case_id"]) not in audit_case_ids
    )
    anchor["candidate_execution_roles"] = ["deployment_shortlist"]
    anchor["deployment_rank"] = 1
    case_id = str(anchor["case_id"])
    boundary = int(anchor["boundary"])

    plan["deployment_shortlist"] = [anchor]
    plan["execution_union_order_policy"] = (
        "forced_deployment_then_audit"
    )
    _rebuild_selected_union(plan)
    fixture["selection_policy"]["forced_cases"] = {
        "resnet50": [case_id],
    }
    selection_input = json.loads(
        fixture["selection_input_path"].read_text(encoding="utf-8")
    )
    selection_input["profile_selection_policy"] = copy.deepcopy(
        fixture["selection_policy"]
    )
    selection_input["deployment_shortlist_count"] = 1
    selection_input["execution_union_order_policy"] = (
        "forced_deployment_then_audit"
    )
    write_json(fixture["selection_input_path"], selection_input)
    return case_id, boundary


def _rebind_changed_prediction(
    fixture: dict,
    *,
    eligible_candidate_count: int,
) -> dict:
    """Rebuild mutable outer checksums so tests reach the policy authority."""

    prediction = fixture["prediction"]
    prediction["artifact_id"] = _prediction_artifact_id(
        prediction["candidates"]
    )
    write_json(fixture["prediction_path"], prediction)

    universe = json.loads(
        fixture["universe_path"].read_text(encoding="utf-8")
    )
    universe["source_prediction_sha256"] = sha256_file(
        fixture["prediction_path"]
    )
    universe["universe_sha256"] = sha256_json({
        key: value
        for key, value in universe.items()
        if key not in {"universe_sha256", "created_at"}
    })
    write_json(fixture["universe_path"], universe)

    fixture["plan"]["source_prediction_artifact_id"] = prediction[
        "artifact_id"
    ]
    fixture["plan"]["candidate_universe_sha256"] = universe[
        "universe_sha256"
    ]
    selection_input = json.loads(
        fixture["selection_input_path"].read_text(encoding="utf-8")
    )
    selection_input["source_prediction_artifact_id"] = prediction[
        "artifact_id"
    ]
    selection_input["candidate_universe_sha256"] = universe[
        "universe_sha256"
    ]
    selection_input["eligible_candidate_count"] = eligible_candidate_count
    write_json(fixture["selection_input_path"], selection_input)
    return universe


def test_frozen_20_plus_4_audit_uses_22_case_exact_generation_union(
    tmp_path: Path,
) -> None:
    fixture = _frozen_audit_plan(tmp_path)
    plan = fixture["plan"]
    prediction = fixture["prediction"]
    expected_boundaries = fixture["expected_boundaries"]

    ranked, pool, requested, frozen = _resolve_generation_candidate_scope(
        plan,
        prediction,
        **_binding_kwargs(fixture),
    )

    assert frozen is True
    assert requested == 22
    assert ranked == expected_boundaries
    assert pool == expected_boundaries
    assert plan["requested_cases"] == 4
    assert len(set(pool)) == 22

    accepted = [
        {"folder": f"b{boundary:03d}", "boundary": boundary}
        for boundary in expected_boundaries
    ]
    final, trace = reconcile_candidate_plan_after_generation(
        plan,
        prediction=prediction,
        accepted_cases=accepted,
        rejected_cases=[],
        **_binding_kwargs(fixture),
    )
    assert final == plan
    assert trace["status"] == "matches_frozen_predeclared_execution_union"
    assert trace["execution_union_count"] == 22
    assert trace["claim_eligible"] is True


def test_forced_audit_deployment_anchor_is_first_in_exact_frozen_union(
    tmp_path: Path,
) -> None:
    fixture = _frozen_audit_plan(tmp_path)
    original_audit = [
        str(row["case_id"])
        for row in fixture["plan"]["audit_candidates"]
    ]
    anchor_case_id, anchor_boundary = _force_deployment_anchor(fixture)

    ranked, pool, requested, frozen = _resolve_generation_candidate_scope(
        fixture["plan"],
        fixture["prediction"],
        **_binding_kwargs(fixture),
    )

    assert frozen is True
    assert ranked == pool
    assert requested == len(ranked) == len(original_audit) + 1
    assert ranked[0] == anchor_boundary
    assert fixture["plan"]["selected_candidates"][0]["case_id"] == (
        anchor_case_id
    )
    assert [
        str(row["case_id"])
        for row in fixture["plan"]["audit_candidates"]
    ] == original_audit


def test_forced_audit_execution_order_tamper_is_rejected(
    tmp_path: Path,
) -> None:
    fixture = _frozen_audit_plan(tmp_path)
    _anchor_case_id, _anchor_boundary = _force_deployment_anchor(fixture)
    plan = fixture["plan"]

    # Rebind the selected payload coherently but in the old audit-first order.
    plan["execution_union_order_policy"] = "audit_then_deployment"
    _rebuild_selected_union(plan)
    plan["execution_union_order_policy"] = (
        "forced_deployment_then_audit"
    )

    with pytest.raises(
        ValueError,
        match="deduplicated audit/deployment union",
    ):
        _resolve_generation_candidate_scope(
            plan,
            fixture["prediction"],
            **_binding_kwargs(fixture),
        )


def test_forced_audit_missing_anchor_is_rejected_by_frozen_binding(
    tmp_path: Path,
) -> None:
    fixture = _frozen_audit_plan(tmp_path)
    plan = fixture["plan"]
    fixture["selection_policy"]["forced_cases"] = {
        "resnet50": ["b999"],
    }
    plan["execution_union_order_policy"] = (
        "forced_deployment_then_audit"
    )
    selection_input = json.loads(
        fixture["selection_input_path"].read_text(encoding="utf-8")
    )
    selection_input["profile_selection_policy"] = copy.deepcopy(
        fixture["selection_policy"]
    )
    selection_input["execution_union_order_policy"] = (
        "forced_deployment_then_audit"
    )
    write_json(fixture["selection_input_path"], selection_input)

    with pytest.raises(
        ValueError,
        match="absent from the resolved capability-eligible prediction",
    ):
        _resolve_generation_candidate_scope(
            plan,
            fixture["prediction"],
            **_binding_kwargs(fixture),
        )


def test_frozen_audit_records_unsupported_candidate_without_backfill(
    tmp_path: Path,
) -> None:
    fixture = _frozen_audit_plan(tmp_path)
    plan = fixture["plan"]
    prediction = fixture["prediction"]
    expected = fixture["expected_boundaries"]
    rejected_boundaries = [expected[6], expected[-3]]
    rejected_set = set(rejected_boundaries)
    accepted_boundaries = [
        boundary for boundary in expected if boundary not in rejected_set
    ]

    final, trace = reconcile_candidate_plan_after_generation(
        plan,
        prediction=prediction,
        accepted_cases=[
            {"folder": f"b{boundary:03d}", "boundary": boundary}
            for boundary in accepted_boundaries
        ],
        rejected_cases=[
            {
                "folder": f"b{boundary:03d}",
                "boundary": boundary,
                "reason": "hailo_hef_build_failed",
            }
            for boundary in rejected_boundaries
        ],
        **_binding_kwargs(fixture),
    )

    # The prospective plan stays immutable.  Runtime execution consumes the
    # accepted BenchmarkSet rows, while the trace retains the full attempted
    # audit ledger and the unsupported observations without inventing backfill.
    assert final == plan
    assert trace["changed"] is False
    assert trace["status"] == (
        "frozen_predeclared_execution_union_with_recorded_rejections"
    )
    assert trace["attempted_case_ids"] == [
        f"b{boundary:03d}" for boundary in expected
    ]
    assert trace["accepted_case_ids"] == [
        f"b{boundary:03d}" for boundary in accepted_boundaries
    ]
    assert trace["rejected_case_ids"] == [
        f"b{boundary:03d}" for boundary in rejected_boundaries
    ]
    assert [row["generation_status"] for row in trace["attempt_ledger"]] == [
        "rejected" if boundary in rejected_set else "accepted"
        for boundary in expected
    ]
    assert trace["materialized_count"] == 20
    assert trace["rejected_count"] == 2
    assert trace["backfilled_case_ids"] == []
    assert trace["claim_eligible"] is True


def test_frozen_audit_post_build_minimum_is_fail_closed_without_backfill(
    tmp_path: Path,
) -> None:
    fixture = _frozen_audit_plan(tmp_path)
    plan = fixture["plan"]
    prediction = fixture["prediction"]
    ordered = [
        (str(row["case_id"]), int(row["boundary"]))
        for row in plan["selected_candidates"]
    ]
    audit_ids = {
        str(row["case_id"])
        for row in plan["audit_candidates"]
    }
    retained_audit_ids = {
        str(row["case_id"])
        for row in plan["audit_candidates"][:8]
    }
    accepted = [
        identity
        for identity in ordered
        if identity[0] in retained_audit_ids or identity[0] not in audit_ids
    ]
    rejected = [identity for identity in ordered if identity not in accepted]

    final, trace = reconcile_candidate_plan_after_generation(
        plan,
        prediction=prediction,
        accepted_cases=[
            {"folder": case_id, "boundary": boundary}
            for case_id, boundary in accepted
        ],
        rejected_cases=[
            {
                "folder": case_id,
                "boundary": boundary,
                "reason": "backend_artifact_terminal_failure",
            }
            for case_id, boundary in rejected
        ],
        **_binding_kwargs(fixture),
    )

    assert final == plan
    assert trace["post_build_audit_materialized_count"] == 8
    assert trace["post_build_minimum_required"] == 10
    assert trace["post_build_minimum_status"] == "shortfall"
    assert trace["execution_status"] == (
        "insufficient_materialized_audit_candidates"
    )
    assert trace["repair_status"] == "required_no_automatic_backfill"
    assert trace["automatic_post_build_backfill_performed"] is False
    assert trace["backfilled_case_ids"] == []
    assert trace["claim_eligible"] is False


def test_frozen_zero_accepted_preserves_exact_union_for_direct_fallback(
    tmp_path: Path,
) -> None:
    fixture = _frozen_audit_plan(tmp_path)
    plan = fixture["plan"]
    prediction = fixture["prediction"]
    expected = fixture["expected_boundaries"]
    assert len(plan["selected_candidates"]) == 22
    assert len({row["case_id"] for row in plan["selected_candidates"]}) == 22

    final, trace = reconcile_candidate_plan_after_generation(
        plan,
        prediction=prediction,
        accepted_cases=[],
        rejected_cases=[
            {"folder": f"b{boundary:03d}", "boundary": boundary}
            for boundary in expected
        ],
        **_binding_kwargs(fixture),
    )

    assert final == plan
    assert trace["status"] == (
        "frozen_union_no_accepted_cases_pending_direct_fallback"
    )
    assert trace["execution_union_count"] == 22
    assert trace["attempted_case_ids"] == [
        f"b{boundary:03d}" for boundary in expected
    ]
    assert trace["rejected_case_ids"] == [
        f"b{boundary:03d}" for boundary in expected
    ]
    assert trace["claim_eligible"] is False


@pytest.mark.parametrize("replacement_kind", ["truncated", "external"])
def test_frozen_audit_fails_closed_on_truncation_or_external_backfill(
    tmp_path: Path,
    replacement_kind: str,
) -> None:
    fixture = _frozen_audit_plan(tmp_path)
    plan = fixture["plan"]
    prediction = fixture["prediction"]
    expected = fixture["expected_boundaries"]
    accepted_boundaries = (
        expected[:4]
        if replacement_kind == "truncated"
        else [*expected[:-1], 999]
    )
    accepted = [
        {"folder": f"b{boundary:03d}", "boundary": boundary}
        for boundary in accepted_boundaries
    ]

    with pytest.raises(ValueError, match="was not fully materialized"):
        reconcile_candidate_plan_after_generation(
            plan,
            prediction=prediction,
            accepted_cases=accepted,
            rejected_cases=[],
            **_binding_kwargs(fixture),
        )


def test_frozen_audit_rejects_candidate_classified_twice(
    tmp_path: Path,
) -> None:
    fixture = _frozen_audit_plan(tmp_path)
    expected = fixture["expected_boundaries"]

    with pytest.raises(ValueError, match="both accepted and rejected"):
        reconcile_candidate_plan_after_generation(
            fixture["plan"],
            prediction=fixture["prediction"],
            accepted_cases=[
                {"folder": f"b{boundary:03d}", "boundary": boundary}
                for boundary in expected
            ],
            rejected_cases=[
                {"folder": f"b{expected[0]:03d}", "boundary": expected[0]}
            ],
            **_binding_kwargs(fixture),
        )


def test_frozen_audit_rejects_external_rejected_candidate(
    tmp_path: Path,
) -> None:
    fixture = _frozen_audit_plan(tmp_path)
    expected = fixture["expected_boundaries"]

    with pytest.raises(ValueError, match="unexpected=.*b999"):
        reconcile_candidate_plan_after_generation(
            fixture["plan"],
            prediction=fixture["prediction"],
            accepted_cases=[
                {"folder": f"b{boundary:03d}", "boundary": boundary}
                for boundary in expected[:-1]
            ],
            rejected_cases=[{"folder": "b999", "boundary": 999}],
            **_binding_kwargs(fixture),
        )


@pytest.mark.parametrize("classification", ["accepted", "rejected"])
def test_frozen_audit_rejects_nondeterministic_classification_order(
    tmp_path: Path,
    classification: str,
) -> None:
    fixture = _frozen_audit_plan(tmp_path)
    expected = fixture["expected_boundaries"]
    if classification == "accepted":
        accepted_boundaries = list(expected)
        accepted_boundaries[0], accepted_boundaries[1] = (
            accepted_boundaries[1],
            accepted_boundaries[0],
        )
        rejected_boundaries: list[int] = []
    else:
        rejected_boundaries = [expected[5], expected[2]]
        rejected_set = set(rejected_boundaries)
        accepted_boundaries = [
            boundary for boundary in expected if boundary not in rejected_set
        ]

    with pytest.raises(
        ValueError,
        match=rf"Frozen {classification} cases do not follow.*deterministic",
    ):
        reconcile_candidate_plan_after_generation(
            fixture["plan"],
            prediction=fixture["prediction"],
            accepted_cases=[
                {"folder": f"b{boundary:03d}", "boundary": boundary}
                for boundary in accepted_boundaries
            ],
            rejected_cases=[
                {"folder": f"b{boundary:03d}", "boundary": boundary}
                for boundary in rejected_boundaries
            ],
            **_binding_kwargs(fixture),
        )


def test_frozen_audit_rejects_same_boundary_different_prediction_case_id(
    tmp_path: Path,
) -> None:
    fixture = _frozen_audit_plan(tmp_path)
    plan = fixture["plan"]
    prediction = fixture["prediction"]
    universe_path = fixture["universe_path"]
    prediction_path = fixture["prediction_path"]
    tampered = copy.deepcopy(prediction)
    target = str(plan["audit_candidates"][0]["case_id"])
    row = next(row for row in tampered["candidates"] if row["case_id"] == target)
    row["case_id"] = "tampered-same-boundary"
    tampered["artifact_id"] = _prediction_artifact_id(tampered["candidates"])
    write_json(prediction_path, tampered)

    # Rebind file/universe hashes so the canonical case_id+boundary join is the
    # remaining line of defence rather than a merely stale outer checksum.
    universe = json.loads(universe_path.read_text(encoding="utf-8"))
    universe["source_prediction_sha256"] = sha256_file(prediction_path)
    universe["universe_sha256"] = sha256_json({
        key: value
        for key, value in universe.items()
        if key not in {"universe_sha256", "created_at"}
    })
    write_json(universe_path, universe)
    plan["candidate_universe_sha256"] = universe["universe_sha256"]
    plan["source_prediction_artifact_id"] = tampered["artifact_id"]

    with pytest.raises(ValueError, match=r"not an exact case_id\+boundary bijection"):
        _resolve_generation_candidate_scope(
            plan,
            tampered,
            **_binding_kwargs(fixture),
        )


def test_frozen_audit_rejects_stale_candidate_identity_hash(
    tmp_path: Path,
) -> None:
    fixture = _frozen_audit_plan(tmp_path)
    plan = fixture["plan"]
    prediction = fixture["prediction"]
    plan["selected_candidates"][0]["candidate_identity_sha256"] = "sha256:stale"
    plan["artifact_id"] = _candidate_plan_artifact_id(
        plan["selected_candidates"]
    )

    with pytest.raises(ValueError, match="candidate identity hash mismatch"):
        _resolve_generation_candidate_scope(
            plan,
            prediction,
            **_binding_kwargs(fixture),
        )


def test_frozen_audit_rejects_stale_candidate_universe_binding(
    tmp_path: Path,
) -> None:
    fixture = _frozen_audit_plan(tmp_path)
    plan = fixture["plan"]
    prediction = fixture["prediction"]
    plan["candidate_universe_sha256"] = "sha256:stale"

    with pytest.raises(ValueError, match="stale or tampered"):
        _resolve_generation_candidate_scope(
            plan,
            prediction,
            **_binding_kwargs(fixture),
        )


def test_frozen_audit_rejects_candidate_universe_self_hash_tamper(
    tmp_path: Path,
) -> None:
    fixture = _frozen_audit_plan(tmp_path)
    plan = fixture["plan"]
    prediction = fixture["prediction"]
    universe_path = fixture["universe_path"]
    universe = json.loads(universe_path.read_text(encoding="utf-8"))
    universe["audit"]["seed"] += 1
    write_json(universe_path, universe)

    with pytest.raises(ValueError, match="stale or tampered"):
        _resolve_generation_candidate_scope(
            plan,
            prediction,
            **_binding_kwargs(fixture),
        )


def test_frozen_audit_rejects_prediction_file_hash_tamper(
    tmp_path: Path,
) -> None:
    fixture = _frozen_audit_plan(tmp_path)
    plan = fixture["plan"]
    prediction = fixture["prediction"]
    prediction_path = fixture["prediction_path"]
    tampered = {**prediction, "node_count": 999}
    write_json(prediction_path, tampered)

    with pytest.raises(ValueError, match="prediction content binding mismatch"):
        _resolve_generation_candidate_scope(
            plan,
            tampered,
            **_binding_kwargs(fixture),
        )


def test_frozen_audit_rejects_candidate_plan_artifact_tamper(
    tmp_path: Path,
) -> None:
    fixture = _frozen_audit_plan(tmp_path)
    plan = fixture["plan"]
    prediction = fixture["prediction"]
    plan["artifact_id"] = "candidate_plan_resnet50_stale"

    with pytest.raises(ValueError, match="candidate-plan artifact binding mismatch"):
        _resolve_generation_candidate_scope(
            plan,
            prediction,
            **_binding_kwargs(fixture),
        )


@pytest.mark.parametrize(
    "list_name",
    ["audit_candidates", "deployment_shortlist", "selected_candidates"],
)
def test_frozen_audit_rejects_duplicates_inside_each_authoritative_list(
    tmp_path: Path,
    list_name: str,
) -> None:
    fixture = _frozen_audit_plan(tmp_path)
    plan = fixture["plan"]
    plan[list_name].append(copy.deepcopy(plan[list_name][0]))
    if list_name == "selected_candidates":
        plan["artifact_id"] = _candidate_plan_artifact_id(
            plan["selected_candidates"]
        )

    with pytest.raises(ValueError, match="duplicate canonical identity"):
        _resolve_generation_candidate_scope(
            plan,
            fixture["prediction"],
            **_binding_kwargs(fixture),
        )


def test_frozen_audit_rejects_fifth_deployment_candidate(
    tmp_path: Path,
) -> None:
    fixture = _frozen_audit_plan(tmp_path)
    plan = fixture["plan"]
    prediction_row = fixture["prediction"]["candidates"][4]
    universe = json.loads(
        fixture["universe_path"].read_text(encoding="utf-8")
    )
    universe_row = next(
        row
        for row in universe["candidates"]
        if row["case_id"] == prediction_row["case_id"]
    )
    plan["deployment_shortlist"].append({
        **copy.deepcopy(prediction_row),
        "candidate_id": universe_row["candidate_id"],
        "candidate_identity_sha256": universe_row["candidate_identity_sha256"],
        "candidate_execution_roles": ["deployment_shortlist"],
        "origin": "deployment_shortlist",
        "source_rank": prediction_row["rank"],
        "deployment_rank": 5,
    })
    _rebuild_selected_union(plan)

    with pytest.raises(ValueError, match="deployment_shortlist exceeds"):
        _resolve_generation_candidate_scope(
            plan,
            fixture["prediction"],
            **_binding_kwargs(fixture),
        )


def test_frozen_audit_rejects_reordered_deployment_only_candidates(
    tmp_path: Path,
) -> None:
    fixture = _frozen_audit_plan(tmp_path)
    plan = fixture["plan"]
    deployment = plan["deployment_shortlist"]
    deployment[2], deployment[3] = deployment[3], deployment[2]
    for rank, row in enumerate(deployment, start=1):
        row["deployment_rank"] = rank
    _rebuild_selected_union(plan)

    with pytest.raises(ValueError, match="deterministic prediction/forced-case order"):
        _resolve_generation_candidate_scope(
            plan,
            fixture["prediction"],
            **_binding_kwargs(fixture),
        )


def test_frozen_audit_rejects_extra_universe_candidate_without_prediction(
    tmp_path: Path,
) -> None:
    fixture = _frozen_audit_plan(tmp_path)
    plan = fixture["plan"]
    universe_path = fixture["universe_path"]
    universe = json.loads(universe_path.read_text(encoding="utf-8"))
    extra = _candidate(999)
    extra.update(stable_candidate_identity(
        "resnet50",
        {**extra, "model_sha256": universe["model_sha256"]},
        len(universe["candidates"]) + 1,
    ))
    extra["selected_for_measurement"] = False
    extra["candidate_role"] = "not_in_audit"
    universe["candidates"].append(extra)
    universe["feasible_candidate_count"] = len(universe["candidates"])
    universe["feasible_candidate_identity_sha256"] = sha256_json([
        row["candidate_id"] for row in universe["candidates"]
    ])
    universe["universe_sha256"] = sha256_json({
        key: value
        for key, value in universe.items()
        if key not in {"universe_sha256", "created_at"}
    })
    write_json(universe_path, universe)
    plan["candidate_universe_sha256"] = universe["universe_sha256"]
    selection_input = json.loads(
        fixture["selection_input_path"].read_text(encoding="utf-8")
    )
    selection_input["candidate_universe_sha256"] = universe["universe_sha256"]
    write_json(fixture["selection_input_path"], selection_input)

    with pytest.raises(ValueError, match=r"not an exact case_id\+boundary bijection"):
        _resolve_generation_candidate_scope(
            plan,
            fixture["prediction"],
            **_binding_kwargs(fixture),
        )


def test_frozen_audit_rejects_duplicate_manifest_identity(
    tmp_path: Path,
) -> None:
    fixture = _frozen_audit_plan(tmp_path)
    plan = fixture["plan"]
    universe_path = fixture["universe_path"]
    universe = json.loads(universe_path.read_text(encoding="utf-8"))
    universe["candidates"].append(copy.deepcopy(universe["candidates"][0]))
    universe["feasible_candidate_count"] = len(universe["candidates"])
    universe["feasible_candidate_identity_sha256"] = sha256_json([
        row["candidate_id"] for row in universe["candidates"]
    ])
    universe["universe_sha256"] = sha256_json({
        key: value
        for key, value in universe.items()
        if key not in {"universe_sha256", "created_at"}
    })
    write_json(universe_path, universe)
    plan["candidate_universe_sha256"] = universe["universe_sha256"]
    selection_input = json.loads(
        fixture["selection_input_path"].read_text(encoding="utf-8")
    )
    selection_input["candidate_universe_sha256"] = universe["universe_sha256"]
    write_json(fixture["selection_input_path"], selection_input)

    with pytest.raises(ValueError, match="duplicate canonical identity"):
        _resolve_generation_candidate_scope(
            plan,
            fixture["prediction"],
            **_binding_kwargs(fixture),
        )


def test_frozen_audit_rejects_coherently_rehashed_wrong_model_identity(
    tmp_path: Path,
) -> None:
    fixture = _frozen_audit_plan(tmp_path)
    plan = fixture["plan"]
    universe_path = fixture["universe_path"]
    universe = json.loads(universe_path.read_text(encoding="utf-8"))
    universe["model_sha256"] = "sha256:coherently-wrong-model"
    canonical_by_case = {}
    for index, row in enumerate(universe["candidates"], start=1):
        canonical = stable_candidate_identity(
            "resnet50",
            {**row, "model_sha256": universe["model_sha256"]},
            index,
        )
        row.update(canonical)
        canonical_by_case[row["case_id"]] = canonical
    universe["selected_candidate_ids"] = [
        canonical_by_case[case_id]["candidate_id"]
        for case_id in universe["selected_case_ids"]
    ]
    universe["selected_candidate_identity_sha256"] = sha256_json(
        universe["selected_candidate_ids"]
    )
    universe["feasible_candidate_identity_sha256"] = sha256_json([
        row["candidate_id"] for row in universe["candidates"]
    ])
    universe["universe_sha256"] = sha256_json({
        key: value
        for key, value in universe.items()
        if key not in {"universe_sha256", "created_at"}
    })
    write_json(universe_path, universe)

    for key in ("audit_candidates", "deployment_shortlist", "selected_candidates"):
        for row in plan[key]:
            row.update(canonical_by_case[row["case_id"]])
    plan["candidate_universe_sha256"] = universe["universe_sha256"]
    plan["artifact_id"] = _candidate_plan_artifact_id(
        plan["selected_candidates"]
    )
    selection_input = json.loads(
        fixture["selection_input_path"].read_text(encoding="utf-8")
    )
    selection_input["candidate_universe_sha256"] = universe["universe_sha256"]
    write_json(fixture["selection_input_path"], selection_input)

    with pytest.raises(ValueError, match="does not match the actual ONNX model"):
        _resolve_generation_candidate_scope(
            plan,
            fixture["prediction"],
            **_binding_kwargs(fixture),
        )


def test_frozen_audit_rejects_selection_policy_rebinding(
    tmp_path: Path,
) -> None:
    fixture = _frozen_audit_plan(tmp_path)
    selection_input = json.loads(
        fixture["selection_input_path"].read_text(encoding="utf-8")
    )
    selection_input["profile_selection_policy"]["forced_cases"] = {
        "resnet50": ["b030", "b029", "b028", "b027"]
    }
    write_json(fixture["selection_input_path"], selection_input)

    with pytest.raises(ValueError, match="does not match the resolved evaluation profile"):
        _resolve_generation_candidate_scope(
            fixture["plan"],
            fixture["prediction"],
            **_binding_kwargs(fixture),
        )


def test_frozen_audit_rejects_coherently_trimmed_requested_case_budget(
    tmp_path: Path,
) -> None:
    fixture = _frozen_audit_plan(tmp_path)
    plan = fixture["plan"]
    plan["requested_cases"] = 3
    plan["deployment_shortlist"] = plan["deployment_shortlist"][:3]
    _rebuild_selected_union(plan)
    selection_input = json.loads(
        fixture["selection_input_path"].read_text(encoding="utf-8")
    )
    selection_input["requested_cases"] = 3
    selection_input["deployment_shortlist_count"] = 3
    write_json(fixture["selection_input_path"], selection_input)

    with pytest.raises(
        ValueError,
        match="requested_cases does not match the resolved evaluation profile",
    ):
        _resolve_generation_candidate_scope(
            plan,
            fixture["prediction"],
            **_binding_kwargs(fixture),
        )


def test_frozen_audit_rejects_coherently_rebound_min_gap(
    tmp_path: Path,
) -> None:
    fixture = _frozen_audit_plan(tmp_path)
    plan = fixture["plan"]
    plan["min_gap"] = 1
    selection_input = json.loads(
        fixture["selection_input_path"].read_text(encoding="utf-8")
    )
    selection_input["min_gap"] = 1
    write_json(fixture["selection_input_path"], selection_input)

    with pytest.raises(
        ValueError,
        match="min_gap does not match the resolved evaluation profile",
    ):
        _resolve_generation_candidate_scope(
            plan,
            fixture["prediction"],
            **_binding_kwargs(fixture),
        )


def test_frozen_audit_rejects_selection_strategy_rebound_from_profile(
    tmp_path: Path,
) -> None:
    fixture = _frozen_audit_plan(tmp_path)
    selection_input = json.loads(
        fixture["selection_input_path"].read_text(encoding="utf-8")
    )
    selection_input["selection_strategy"] = "stratified-windows"
    write_json(fixture["selection_input_path"], selection_input)

    with pytest.raises(
        ValueError,
        match="selection_strategy does not match the resolved evaluation profile",
    ):
        _resolve_generation_candidate_scope(
            fixture["plan"],
            fixture["prediction"],
            **_binding_kwargs(fixture),
        )


def test_frozen_audit_accepts_normalized_authoritative_analysis_strategy(
    tmp_path: Path,
    ) -> None:
    fixture = _frozen_audit_plan(tmp_path)
    fixture["selection_strategy"] = "ranking-audit"
    fixture["prediction"]["selection_strategy"] = "ranking_audit"
    selection_input = json.loads(
        fixture["selection_input_path"].read_text(encoding="utf-8")
    )
    selection_input["selection_strategy"] = "ranking_audit"
    write_json(fixture["selection_input_path"], selection_input)
    _rebind_changed_prediction(fixture, eligible_candidate_count=30)

    ranked, pool, requested, frozen = _resolve_generation_candidate_scope(
        fixture["plan"],
        fixture["prediction"],
        **_binding_kwargs(fixture),
    )

    assert frozen is True
    assert requested == 22
    assert ranked == pool == fixture["expected_boundaries"]


@pytest.mark.parametrize(
    ("field", "tampered_value"),
    [
        ("requested_cases", 999),
        ("requested_cases", "4.9"),
        ("min_gap", 99),
        ("min_gap", "0.9"),
        ("selection_strategy", "attacker_rebound"),
        ("require_single_part2_input", True),
        ("native_split_requires_single_part2_input", False),
        ("effective_require_single_part2_input", False),
    ],
)
def test_frozen_audit_rejects_rehashed_prediction_policy_metadata(
    tmp_path: Path,
    field: str,
    tampered_value: object,
) -> None:
    fixture = _frozen_audit_plan(tmp_path)
    fixture["prediction"][field] = tampered_value
    _rebind_changed_prediction(fixture, eligible_candidate_count=30)

    with pytest.raises(ValueError, match="Frozen prediction"):
        _resolve_generation_candidate_scope(
            fixture["plan"],
            fixture["prediction"],
            **_binding_kwargs(fixture),
        )


def test_frozen_audit_rejects_mutable_single_part2_capability_flags(
    tmp_path: Path,
) -> None:
    fixture = _frozen_audit_plan(tmp_path)
    plan = fixture["plan"]
    selection_input = json.loads(
        fixture["selection_input_path"].read_text(encoding="utf-8")
    )
    for evidence in (plan, selection_input):
        evidence["native_split_requires_single_part2_input"] = False
        evidence["effective_require_single_part2_input"] = False
    write_json(fixture["selection_input_path"], selection_input)

    with pytest.raises(ValueError, match="Part-2 capability flags do not match"):
        _resolve_generation_candidate_scope(
            plan,
            fixture["prediction"],
            **_binding_kwargs(fixture),
        )


@pytest.mark.parametrize("part2_input_count", [2, "1.9"])
def test_frozen_audit_rejects_multi_input_candidate_in_frozen_prediction(
    tmp_path: Path,
    part2_input_count: object,
) -> None:
    fixture = _frozen_audit_plan(tmp_path)
    plan = fixture["plan"]
    prediction = fixture["prediction"]
    prediction["candidates"][4]["part2_input_count"] = part2_input_count
    prediction["artifact_id"] = _prediction_artifact_id(
        prediction["candidates"]
    )
    write_json(fixture["prediction_path"], prediction)

    universe = json.loads(
        fixture["universe_path"].read_text(encoding="utf-8")
    )
    universe["source_prediction_sha256"] = sha256_file(
        fixture["prediction_path"]
    )
    universe["universe_sha256"] = sha256_json({
        key: value
        for key, value in universe.items()
        if key not in {"universe_sha256", "created_at"}
    })
    write_json(fixture["universe_path"], universe)
    plan["source_prediction_artifact_id"] = prediction["artifact_id"]
    plan["candidate_universe_sha256"] = universe["universe_sha256"]

    selection_input = json.loads(
        fixture["selection_input_path"].read_text(encoding="utf-8")
    )
    selection_input["source_prediction_artifact_id"] = prediction["artifact_id"]
    selection_input["candidate_universe_sha256"] = universe["universe_sha256"]
    selection_input["eligible_candidate_count"] = 29
    write_json(fixture["selection_input_path"], selection_input)

    with pytest.raises(
        ValueError,
        match=r"prediction\.candidates contains candidates outside.*single-Part2",
    ):
        _resolve_generation_candidate_scope(
            plan,
            prediction,
            **_binding_kwargs(fixture),
        )


def test_frozen_audit_rejects_coherently_rebuilt_21_case_union_after_multi_input(
    tmp_path: Path,
) -> None:
    fixture = _frozen_audit_plan(tmp_path)
    plan = fixture["plan"]
    prediction = fixture["prediction"]
    audit_case_ids = {
        str(row["case_id"]) for row in plan["audit_candidates"]
    }
    deployment_only = next(
        row
        for row in plan["deployment_shortlist"]
        if str(row["case_id"]) not in audit_case_ids
    )
    target_case_id = str(deployment_only["case_id"])
    prediction_row = next(
        row
        for row in prediction["candidates"]
        if str(row["case_id"]) == target_case_id
    )
    prediction_row["part2_input_count"] = 2

    universe = json.loads(
        fixture["universe_path"].read_text(encoding="utf-8")
    )
    universe_by_case = {
        str(row["case_id"]): row for row in universe["candidates"]
    }
    replacement_rows = [
        row
        for row in prediction["candidates"]
        if int(row.get("part2_input_count") or -1) == 1
    ][:4]
    plan["deployment_shortlist"] = [
        {
            **copy.deepcopy(row),
            "candidate_id": universe_by_case[str(row["case_id"])][
                "candidate_id"
            ],
            "candidate_identity_sha256": universe_by_case[
                str(row["case_id"])
            ]["candidate_identity_sha256"],
            "candidate_execution_roles": ["deployment_shortlist"],
            "origin": "deployment_shortlist",
            "source_rank": row["rank"],
            "deployment_rank": rank,
        }
        for rank, row in enumerate(replacement_rows, start=1)
    ]
    _rebuild_selected_union(plan)
    _rebind_changed_prediction(fixture, eligible_candidate_count=29)

    assert [
        row["case_id"] for row in plan["deployment_shortlist"]
    ] == ["b001", "b002", "b005", "b004"]
    assert len(plan["selected_candidates"]) == 21

    with pytest.raises(
        ValueError,
        match=r"prediction\.candidates contains candidates outside.*single-Part2",
    ):
        _resolve_generation_candidate_scope(
            plan,
            prediction,
            **_binding_kwargs(fixture),
        )


def test_ordinary_non_audit_generation_keeps_requested_limit_and_backfill_pool() -> None:
    selected = [_candidate(boundary) for boundary in range(1, 5)]
    prediction = {"candidates": [*selected, _candidate(5)]}
    plan = {
        "requested_cases": 4,
        "selection_strategy": "stratified_windows",
        "selected_candidates": selected,
        "policy_backfills": [],
    }

    ranked, pool, requested, frozen = _resolve_generation_candidate_scope(
        plan,
        prediction,
    )
    assert frozen is False
    assert requested == 4
    assert ranked == [1, 2, 3, 4]
    assert pool == [1, 2, 3, 4, 5]

    final, trace = reconcile_candidate_plan_after_generation(
        plan,
        prediction=prediction,
        accepted_cases=[
            {"folder": "b001", "boundary": 1},
            {"folder": "b002", "boundary": 2},
            {"folder": "b003", "boundary": 3},
            {"folder": "b005", "boundary": 5},
        ],
        rejected_cases=[{"folder": "b004", "boundary": 4}],
    )
    assert trace["changed"] is True
    assert trace["backfilled_case_ids"] == ["b005"]
    assert [row["case_id"] for row in final["selected_candidates"]] == [
        "b001",
        "b002",
        "b003",
        "b005",
    ]
