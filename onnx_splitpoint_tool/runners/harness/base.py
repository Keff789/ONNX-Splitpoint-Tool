from __future__ import annotations

from typing import Any, Protocol

from .._types import SampleCfg


class Harness(Protocol):
    """Model-family harness contract."""

    name: str

    def make_inputs(self, sample_cfg: SampleCfg) -> dict: ...

    def postprocess(self, outputs: Any, context: dict) -> dict: ...

    def accuracy_proxy(self, ref: Any, out: Any) -> dict: ...
