from __future__ import annotations

"""Structured acceptance plan for the focused v60u hardware smokes.

The local ``onnx-splitpoint-smoke-v60u`` command is intentionally hardware-free.
This module records the complementary accelerator checks in machine-readable
form so that a run can be planned, archived and reviewed without hiding the
fact that proprietary compilers, targets and u.RECS are required.
"""

from typing import Any

from . import __version__
from .workflow.runner import WORKFLOW_VERSION


def targeted_smoke_plan() -> dict[str, Any]:
    phases = [
        {
            "id": "native_contract_diagnostics_smoke",
            "purpose": "Preserve row-level failures and enforce archived detection-contract families.",
            "requirements": {
                "run_mode": "smoke",
                "cases_per_model": 1,
                "native_runner": True,
                "native_full": False,
                "native_energy": False,
            },
            "acceptance": [
                "every requested native row produces an analysis-table record",
                "rc=0 with a failed row yields evidence_status=partial",
                "raw_head and decoded_nms candidates are never compared across families",
                "failure_reason, returncode, timeout and stdout/stderr tails reach the Debug Pack",
            ],
        },
        {
            "id": "native_full_cold_build_smoke",
            "purpose": "Exercise one explicitly permitted cold Full-baseline build with preserved timeout provenance.",
            "requirements": {
                "scenario": "native_full",
                "full_baseline_cold_build_policy": "build_missing",
                "native_runner": True,
                "native_full": True,
                "native_energy": False,
            },
            "acceptance": [
                "cache misses are attempted only in this explicit scenario",
                "timeout is reported as attempted_timeout with last compiler stage and elapsed time",
                "a deferred regular Smoke baseline is not mislabeled not_dispatched",
            ],
        },
        {
            "id": "cross_runner_three_candidate_smoke",
            "purpose": "Validate Generic-to-Native rank-transfer reporting on at least three shared candidates.",
            "requirements": {
                "scenario": "cross_runner",
                "cases_per_model_min": 3,
                "native_runner": True,
                "native_full": True,
                "native_energy": False,
            },
            "acceptance": [
                "at least three semantically valid Generic/Native pairs in one backend direction",
                "Spearman, Kendall, pairwise concordance, Hit@k and Native regret are emitted",
                "fewer than three pairs is reported as insufficient_candidates",
            ],
        },
        {
            "id": "native_energy_measure_smoke",
            "purpose": "Verify that the simplified energy switch performs a physical Native-only measurement.",
            "requirements": {
                "scenario": "native_energy",
                "native_runner": True,
                "native_full": True,
                "native_energy": True,
                "generic_energy": False,
                "native_energy_mode": "measure",
            },
            "acceptance": [
                "effective plan states Generic energy off and Native energy mode measure",
                "one semantically valid split and its paired Native Full baseline are measured",
                "observed work-unit marker is preferred over FPS-times-duration fallback",
            ],
        },
        {
            "id": "artifact_restore_and_parallel_cold_miss_smoke",
            "purpose": "Verify central Artifact Library restore and controlled parallel compiler misses.",
            "requirements": {
                "scenarios": ["artifact_restore", "parallel_cold_build"],
                "artifact_store": True,
                "scheduler_workers_min": 3,
                "generic_energy": False,
                "native_energy": False,
            },
            "acceptance": [
                "an exact registered HEF/DXNN is restored after its legacy cache object is moved aside",
                "two different compiler families overlap in build_scheduler_v60s.jsonl",
                "same-family builds remain serialized",
                "no duplicate artifact publication occurs under concurrent runs",
            ],
        },
    ]
    return {
        "schema": "onnx-splitpoint/v60u-hardware-smoke-plan",
        "schema_version": 1,
        "tool_version": __version__,
        "workflow_version": WORKFLOW_VERSION,
        "hardware_required": True,
        "phases": phases,
    }
