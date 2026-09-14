from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from onnx_splitpoint_tool.native_detection_postprocess import (
    FrozenDetectionPostprocessor,
    build_completed_detection_endpoint_attestation,
    build_frozen_postprocess_contract,
    canonical_json_sha256,
    persist_completed_result_artifact,
)
from onnx_splitpoint_tool.runners.harness.yolo import (
    YOLOV7_PAPER_ONNX_SHA256,
)
from onnx_splitpoint_tool.validation.accuracy_gates import (
    AccuracyGatePolicy,
)
from scripts import native_producer_validate_visualize as validator


def _raw_outputs() -> dict[str, np.ndarray]:
    return {
        "output": np.full(
            (1, 3, 80, 80, 85), -20.0, dtype=np.float32,
        ),
        "clone_1": np.full(
            (1, 3, 40, 40, 85), -20.0, dtype=np.float32,
        ),
        "clone_2": np.full(
            (1, 3, 20, 20, 85), -20.0, dtype=np.float32,
        ),
    }


def _completed_evidence(
    artifact_dir: Path | None = None,
) -> tuple[dict, dict[str, np.ndarray]]:
    outputs = _raw_outputs()
    frozen = build_frozen_postprocess_contract(
        model_id="yolov7_paper",
        outputs=outputs,
        input_hw=[640, 640],
        original_wh=[80, 60],
        model_sha256=YOLOV7_PAPER_ONNX_SHA256,
    )
    result = FrozenDetectionPostprocessor(frozen).process(
        outputs,
        original_wh=[80, 60],
    )
    completion = build_completed_detection_endpoint_attestation(
        frozen,
        result,
        completed_frames=3,
        postprocess_completed_frames=3,
        source_endpoint_contract_hash="b" * 64,
    )
    comparison = completion[
        "completed_task_comparison_endpoint_contract"
    ]
    evidence = {
        "frozen_host_postprocess_contract": frozen,
        "frozen_host_postprocess_contract_sha256": frozen[
            "contract_sha256"
        ],
        "frozen_host_postprocess_result": copy.deepcopy(result),
        "completed_task_endpoint_attestation": completion,
        "completed_task_endpoint_attested": True,
        "completed_task_endpoint_attestation_status": "passed",
        "completed_task_completion_mode": "frozen_host_tail",
        "completed_task_comparison_endpoint_contract": comparison,
        "completed_task_comparison_endpoint_contract_hash": comparison[
            "endpoint_contract_hash"
        ],
        "completed_task_comparison_output_endpoint_id": comparison[
            "output_endpoint_id"
        ],
    }
    if artifact_dir is not None:
        evidence.update(persist_completed_result_artifact(
            result["completed_result_artifact"],
            expected_sha256=result[
                "completed_result_artifact_sha256"
            ],
            output_path=(
                artifact_dir / "raw_completed_result.json"
            ),
        ))
    return evidence, outputs


def _full_outputs() -> dict[str, np.ndarray]:
    return {
        "detections": np.asarray(
            [[
                [8.0, 16.0, 32.0, 48.0, 0.9, 2.0],
                [0.0, 0.0, 4.0, 4.0, 0.1, 1.0],
            ]],
            dtype=np.float32,
        ),
    }


def _separate_hotloop_hash(
    evidence: dict,
    digest: str = "c" * 64,
) -> None:
    sealed = evidence["completed_task_endpoint_attestation"][
        "frozen_postprocess_result"
    ]
    # This helper models an archived V1 performance/semantic split where the
    # hotloop stored only a hash.  New Full hotloops carry the exact sentinel
    # detections and therefore cannot be converted into a portable mismatch
    # merely by editing that hash.
    for field in (
        "coordinate_space",
        "record_schema",
        "canonical_sort_policy",
        "detections",
        "completed_result_artifact",
        "completed_result_artifact_sha256",
    ):
        sealed.pop(field, None)
    sealed["detections_sha256"] = digest
    evidence["frozen_host_postprocess_result"] = copy.deepcopy(sealed)


def _screening_policy(**updates: object) -> AccuracyGatePolicy:
    values = {
        "dataset_tier": "screening",
        "frozen_before_final_campaign": False,
        "screening_eligible_for_ranking": False,
        "contract_only_eligible_for_ranking": False,
    }
    values.update(updates)
    return AccuracyGatePolicy(**values)


def test_hash_only_hotloop_drift_is_portable_screening_evidence() -> None:
    evidence, outputs = _completed_evidence()
    _separate_hotloop_hash(evidence)

    result = validator._completed_v2_self_reference_detection(
        _full_outputs(),
        outputs,
        evidence,
        policy=_screening_policy(),
    )

    assert result["available"] is True
    assert result["completed_v2_verified"] is True
    assert result["portable_result_hash_mismatch"] is True
    assert result["semantic_result_binding_status"] == (
        "portable_result_hash_mismatch"
    )
    assert result["exact_completed_result_identity_bound"] is False
    assert result[
        "completed_v2_exact_result_claim_binding"
    ] is False
    assert result["completed_v2_semantic_evidence_tier"] == (
        "development_screening_portable_replay"
    )
    assert result["performance_hotloop_result_sha256"] == "c" * 64
    assert (
        result["native_completed_result_sha256"]
        != result["performance_hotloop_result_sha256"]
    )


def test_exact_hash_is_named_completed_result_identity_only(
    tmp_path: Path,
) -> None:
    evidence, outputs = _completed_evidence(tmp_path)

    result = validator._completed_v2_self_reference_detection(
        _full_outputs(),
        outputs,
        evidence,
        policy=AccuracyGatePolicy(
            dataset_tier="final",
            frozen_before_final_campaign=True,
        ),
    )

    assert result["available"] is True
    assert result["semantic_result_binding_status"] == (
        "exact_same_hotloop_completed_artifact"
    )
    assert result["exact_completed_result_identity_bound"] is True
    assert result["completed_v2_exact_result_claim_binding"] is True
    assert result["completed_v2_semantic_evidence_tier"] == (
        "exact_same_hotloop_completed_artifact"
    )
    assert not any(
        "same_invocation" in key for key in result
    )


@pytest.mark.parametrize(
    "policy",
    [
        AccuracyGatePolicy(
            dataset_tier="final",
            frozen_before_final_campaign=True,
        ),
        _screening_policy(screening_eligible_for_ranking=True),
        _screening_policy(contract_only_eligible_for_ranking=True),
    ],
)
def test_claim_capable_policy_requires_exact_completed_result_hash(
    policy: AccuracyGatePolicy,
) -> None:
    evidence, outputs = _completed_evidence()
    _separate_hotloop_hash(evidence)

    result = validator._completed_v2_self_reference_detection(
        _full_outputs(),
        outputs,
        evidence,
        policy=policy,
    )

    assert result["available"] is False
    assert result["completed_v2_verified"] is False
    assert "portable_result_hash_mismatch" in result["reason"]


def test_default_direct_helper_remains_strict() -> None:
    evidence, outputs = _completed_evidence()
    _separate_hotloop_hash(evidence)

    result = validator._completed_v2_self_reference_detection(
        _full_outputs(),
        outputs,
        evidence,
    )

    assert result["available"] is False
    assert "portable_result_hash_mismatch" in result["reason"]


def test_non_hash_result_drift_is_not_portable() -> None:
    evidence, outputs = _completed_evidence()
    _separate_hotloop_hash(evidence)
    sealed = evidence["completed_task_endpoint_attestation"][
        "frozen_postprocess_result"
    ]
    sealed["detection_count"] = int(
        sealed["detection_count"]
    ) + 1
    evidence["frozen_host_postprocess_result"] = copy.deepcopy(sealed)

    result = validator._completed_v2_self_reference_detection(
        _full_outputs(),
        outputs,
        evidence,
        policy=_screening_policy(),
    )

    assert result["available"] is False
    assert "completed_v2_native_result_attestation_mismatch" in (
        result["reason"]
    )


def test_local_replay_self_consistency_always_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence, outputs = _completed_evidence()
    _separate_hotloop_hash(evidence)

    class BrokenReplay(FrozenDetectionPostprocessor):
        def process(
            self,
            native_outputs: dict[str, np.ndarray],
            *,
            original_wh: list[int] | tuple[int, int],
        ) -> dict:
            result = dict(super().process(
                native_outputs,
                original_wh=original_wh,
            ))
            result["detections_sha256"] = "d" * 64
            return result

    monkeypatch.setattr(
        validator,
        "_FrozenDetectionPostprocessor",
        BrokenReplay,
    )
    result = validator._completed_v2_self_reference_detection(
        _full_outputs(),
        outputs,
        evidence,
        policy=_screening_policy(),
    )

    assert result["available"] is False
    assert "completed_v2_local_replay_self_consistency_mismatch" in (
        result["reason"]
    )


@pytest.mark.parametrize(
    "tamper",
    ["contract", "endpoint", "frames", "implementation"],
)
def test_portable_screening_keeps_evidence_bindings_fail_closed(
    tamper: str,
) -> None:
    evidence, outputs = _completed_evidence()
    _separate_hotloop_hash(evidence)
    if tamper == "contract":
        evidence["frozen_host_postprocess_contract"][
            "confidence_threshold"
        ] = 0.5
    elif tamper == "endpoint":
        evidence[
            "completed_task_comparison_output_endpoint_id"
        ] = "detection:decoded_nms:comparison:" + "d" * 64
    elif tamper == "frames":
        evidence["completed_task_endpoint_attestation"][
            "postprocess_completed_frames"
        ] = 2
    else:
        frozen = evidence["frozen_host_postprocess_contract"]
        frozen["implementation_artifacts"][
            "native_detection_postprocess"
        ]["sha256"] = "d" * 64
        frozen["invariant_identity"]["implementation_artifacts"][
            "native_detection_postprocess"
        ]["sha256"] = "d" * 64
        frozen["invariant_contract_sha256"] = canonical_json_sha256(
            frozen["invariant_identity"]
        )
        unsigned = dict(frozen)
        unsigned.pop("contract_sha256")
        frozen["contract_sha256"] = canonical_json_sha256(unsigned)

    result = validator._completed_v2_self_reference_detection(
        _full_outputs(),
        outputs,
        evidence,
        policy=_screening_policy(),
    )

    assert result["available"] is False
    assert result["completed_v2_verified"] is False


def test_portable_binding_clamps_every_claim_axis() -> None:
    row = {
        "claim_eligible": True,
        "e2e_claim_eligible": True,
        "performance_claim_eligible": True,
        "energy_claim_eligible": True,
        "scientific_claim_eligible": True,
        "thesis_claim_eligible": True,
        "eligible_for_ranking": True,
        "ranking_eligible": True,
        "performance_eligible": True,
        "energy_eligible": True,
        "pareto_eligible": True,
        "thesis_comparison_eligible": True,
        "thesis_valid": True,
        "claim_ok": True,
        "structural_contract_pass": True,
    }
    validator._apply_completed_v2_semantic_binding(
        row,
        {
            "portable_result_hash_mismatch": True,
            "semantic_result_binding_status": (
                "portable_result_hash_mismatch"
            ),
            "exact_completed_result_identity_bound": False,
            "completed_v2_exact_result_claim_binding": False,
        },
    )

    assert row["structural_contract_pass"] is True
    assert row["claim_ok"] is True
    assert row["e2e_contract_reason"] == (
        "portable_result_hash_mismatch"
    )
    for field in (
        "claim_eligible",
        "e2e_claim_eligible",
        "performance_claim_eligible",
        "energy_claim_eligible",
        "scientific_claim_eligible",
        "thesis_claim_eligible",
        "eligible_for_ranking",
        "ranking_eligible",
        "performance_eligible",
        "energy_eligible",
        "pareto_eligible",
        "thesis_comparison_eligible",
        "thesis_valid",
    ):
        assert row[field] is False
