from onnx_splitpoint_tool.workflow.cross_runner_reporting import _group_evidence_limits


def _pair(completed: bool) -> dict:
    return {"technical_transfer_eligibility_checks": {
        "generic_completed_task_measured": completed,
        "native_completed_task_measured": True,
        "exact_identity_complete": True,
    }}


def test_twenty_selected_boundaries_do_not_hide_unmeasured_completion():
    result = _group_evidence_limits([_pair(False) for _ in range(20)], 3)
    assert result["paired_candidate_count_sufficient"] is True
    assert result["technical_transfer_limitations"] == ["generic_completed_task_measured"]
    assert result["technical_transfer_exclusion_counts"] == {"generic_completed_task_measured": 20}


def test_single_yolov7_boundary_keeps_both_distinct_limits():
    result = _group_evidence_limits([_pair(False)], 3)
    assert result["paired_candidate_count_sufficient"] is False
    assert result["technical_transfer_limitations"] == [
        "insufficient_paired_candidates", "generic_completed_task_measured",
    ]


def test_measured_completion_does_not_invent_missing_evidence():
    result = _group_evidence_limits([_pair(True) for _ in range(3)], 3)
    assert result["technical_transfer_exclusion_counts"] == {}
    assert result["technical_transfer_limitations"] == []
