from __future__ import annotations

import pytest

from onnx_splitpoint_tool.runners.harness.base import (
    PostprocessResult,
    postprocess_result_to_dict,
    validate_postprocess_result_dict,
)


def test_postprocess_result_schema_validation() -> None:
    d = postprocess_result_to_dict(
        PostprocessResult(
            task="classification",
            json={"task": "classification", "topk": []},
            overlays={"main": "classification_full.png"},
            summary_text="ok",
        )
    )

    # Should not raise.
    validate_postprocess_result_dict(d)

    # Missing/invalid keys should raise.
    with pytest.raises(ValueError):
        validate_postprocess_result_dict({"task": "", "json": {}})  # type: ignore[arg-type]

    with pytest.raises(TypeError):
        validate_postprocess_result_dict({"task": "x", "json": "nope"})  # type: ignore[arg-type]
