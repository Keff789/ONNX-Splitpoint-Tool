"""Harness protocol.

Harnesses encapsulate *all* model-specific behavior:

- Input preprocessing (make_inputs)
- Optional proxy validation (accuracy_proxy)
- Output postprocessing (postprocess)

The rest of the codebase (runners / GUI) should treat harness outputs as opaque
data structures and render them generically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol, TypedDict, Union

import numpy as np

from .._types import SampleCfg


class PostprocessResultDict(TypedDict, total=False):
    """Stable, GUI-friendly postprocess result.

    This is intentionally lightweight and JSON-serializable so it can be shipped
    inside results bundles and rendered generically.
    """

    task: str
    schema_version: int
    json: Dict[str, Any]
    overlays: Dict[str, str]
    summary_text: str


POSTPROCESS_RESULT_SCHEMA_VERSION = 1


@dataclass
class PostprocessResult:
    """Internal convenience wrapper for postprocess results."""

    task: str
    json: Dict[str, Any]
    overlays: Dict[str, str] = field(default_factory=dict)
    summary_text: str = ""

    def to_dict(self) -> PostprocessResultDict:
        return {
            "schema_version": POSTPROCESS_RESULT_SCHEMA_VERSION,
            "task": self.task,
            "json": self.json,
            "overlays": dict(self.overlays) if self.overlays else {},
            "summary_text": self.summary_text or "",
        }


PostprocessResultLike = Union[PostprocessResult, PostprocessResultDict]


def postprocess_result_to_dict(result: PostprocessResultLike) -> PostprocessResultDict:
    """Normalize PostprocessResultLike into a plain dict."""

    if isinstance(result, PostprocessResult):
        return result.to_dict()
    # Assume already a dict-like object.
    return {
        "schema_version": int(result.get("schema_version", POSTPROCESS_RESULT_SCHEMA_VERSION)),
        "task": result.get("task", ""),
        "json": result.get("json", {}),
        "overlays": result.get("overlays", {}) or {},
        "summary_text": result.get("summary_text", "") or "",
    }


def validate_postprocess_result_dict(d: PostprocessResultDict) -> None:
    """Validate the postprocess contract.

    This is intentionally a *lightweight* validation (not a full JSON schema)
    to catch accidental contract breaks early (especially for new harnesses).
    """

    if not isinstance(d, dict):
        raise TypeError(f"PostprocessResult must be a dict, got {type(d)}")

    schema_version = d.get("schema_version", POSTPROCESS_RESULT_SCHEMA_VERSION)
    if not isinstance(schema_version, int) or schema_version < 1:
        raise ValueError(f"Invalid schema_version={schema_version!r}")

    task = d.get("task", "")
    if not isinstance(task, str) or not task.strip():
        raise ValueError(f"Invalid task={task!r}")

    overlays = d.get("overlays", {})
    if overlays is None:
        overlays = {}
    if not isinstance(overlays, dict):
        raise TypeError(f"overlays must be dict[str,str], got {type(overlays)}")
    for k, v in overlays.items():
        if not isinstance(k, str) or not isinstance(v, str):
            raise TypeError("overlays must be dict[str,str]")

    summary_text = d.get("summary_text", "")
    if summary_text is None:
        summary_text = ""
    if not isinstance(summary_text, str):
        raise TypeError("summary_text must be a string")

    payload = d.get("json", {})
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise TypeError(f"json payload must be a dict, got {type(payload)}")

    # Must be JSON-serializable (for results bundles / GUI).
    import json as _json

    try:
        _json.dumps(d)
    except TypeError as e:
        raise TypeError(f"PostprocessResult is not JSON-serializable: {e}")


class Harness(Protocol):
    def make_inputs(self, sample_cfg: SampleCfg) -> Dict[str, np.ndarray]:
        ...

    def postprocess(
        self, outputs: Dict[str, np.ndarray], context: Optional[Dict[str, Any]] = None
    ) -> PostprocessResultLike:
        ...

    def accuracy_proxy(
        self, outputs_full: Dict[str, np.ndarray], outputs_composed: Dict[str, np.ndarray]
    ) -> Dict[str, Any]:
        """Optional proxy validation.

        The default validation is strict elementwise comparison. For some models
        (notably YOLO on TensorRT), small numerical differences can invalidate
        elementwise checks. Harnesses may implement a more semantically
        meaningful proxy here.
        """

        ...

    def close(self) -> None:
        ...
