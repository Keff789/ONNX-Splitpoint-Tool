from __future__ import annotations

from onnx_splitpoint_tool.workflow.evidence_state_model import (
    project_evidence_state,
    summarize_evidence_states,
)


def test_b500_state_denominators_are_separate() -> None:
    states = []
    for _ in range(163):
        states.append(project_evidence_state({
            "presence": "present", "status": "completed",
            "measurement_endpoint": "p2_output",
        }))
    for _ in range(386):
        states.append(project_evidence_state({
            "presence": "present", "status": "completed",
            "measurement_endpoint": "completed_detection",
            "task_quality_gate": {
                "technical_status": "completed", "decision": "pass",
            },
        }))
    states.append(project_evidence_state({
        "presence": "present", "status": "runtime_failed",
        "measurement_endpoint": "completed_detection",
    }))
    states.append(project_evidence_state({}, scope={
        "variant": "full", "quality_applicability": "applicable",
    }))
    summary = summarize_evidence_states(states)
    assert summary["matrix_required"] == 551
    assert summary["matrix_present"] == 550
    assert summary["quality_not_applicable"] == 163
    assert summary["quality_applicable"] == 388
    assert summary["quality_completed"] == 386
    assert summary["quality_blocked"] == 1
    assert summary["quality_missing"] == 1


def test_sealed_applicable_scope_cannot_inherit_legacy_quality_na() -> None:
    state = project_evidence_state(
        {
            "presence": "present", "status": "completed", "variant": "part2",
            "task_quality_gates_by_variant": {
                "part2": {
                    "status": "technical_validation_only",
                    "decision": "not_applicable",
                },
            },
        },
        scope={
            "schema_version": 2,
            "quality_applicability": "applicable",
            "quality_endpoint": "completed_detection",
        },
    )
    assert state["quality_applicability"] == "applicable"
    assert state["quality_completion"] == "missing"
