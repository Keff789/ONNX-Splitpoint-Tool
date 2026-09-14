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


def __getattr__(name: str):
    """Load the orchestrator only when it is actually requested.

    Generated benchmark suites vendor this package as top-level
    ``splitpoint_runners``.  Eagerly importing :mod:`graph_runner` also imported
    every optional accelerator backend, including modules whose relative
    imports are valid only inside the installed ``onnx_splitpoint_tool``
    package.  The lightweight quality/contract modules do not need any of
    those backends, so keep their self-contained import path genuinely lazy.
    """
    if name == "GraphRunner":
        from .graph_runner import GraphRunner

        return GraphRunner
    raise AttributeError(name)
