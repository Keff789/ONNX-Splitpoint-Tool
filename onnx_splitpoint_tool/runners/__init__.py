"""Unified runner architecture (Phase 1).

This package defines:
- Backend interface (prepare/run/cleanup)
- Harness interface (make_inputs/postprocess/accuracy_proxy)
- GraphRunner orchestrator (1-stage or 2-stage)

The goal is to keep accelerator-specific logic inside backends,
model-family logic inside harnesses, and composition/metrics/artifacts
inside GraphRunner.
"""

from ._types import (
    BackendCaps,
    BackendRunOut,
    GraphPlan,
    GraphRunResult,
    RunCfg,
    SampleCfg,
    StagePlan,
)
from .graph_runner import GraphRunner

__all__ = [
    "BackendCaps",
    "BackendRunOut",
    "GraphPlan",
    "GraphRunResult",
    "RunCfg",
    "SampleCfg",
    "StagePlan",
    "GraphRunner",
]
