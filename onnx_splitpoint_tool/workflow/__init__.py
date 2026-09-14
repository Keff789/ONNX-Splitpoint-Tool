"""Formal Evaluation Workflow layer for ONNX Splitpoint Tool.

The runner is imported lazily.  This keeps lightweight support modules such as
``workflow.artifacts`` usable by campaign tooling without creating a circular
import through ``scientific_reporting -> campaign``.
"""

from .contracts import StageResult, WorkflowOptions, WorkflowRunResult

# Compatibility aliases for the v49 CLI/docs.
EvaluationWorkflowOptions = WorkflowOptions
EvaluationWorkflowResult = WorkflowRunResult


def __getattr__(name: str):
    if name == "EvaluationWorkflowRunner":
        from .runner import EvaluationWorkflowRunner

        return EvaluationWorkflowRunner
    raise AttributeError(name)


__all__ = [
    "StageResult",
    "WorkflowOptions",
    "WorkflowRunResult",
    "EvaluationWorkflowOptions",
    "EvaluationWorkflowResult",
    "EvaluationWorkflowRunner",
]
