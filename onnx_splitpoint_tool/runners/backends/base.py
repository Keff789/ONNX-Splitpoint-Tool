from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .._types import BackendCaps, BackendRunOut, RunCfg


@dataclass
class PreparedHandle:
    """Opaque prepared handle returned by Backend.prepare().

    Backends may subclass/extend.
    """

    # Minimal common fields for GraphRunner convenience.
    input_names: list[str]
    output_names: list[str]
    handle: Any


class Backend(Protocol):
    """Backend contract.

    Backends encapsulate accelerator-specific work:
    - preparing a runnable handle (session/engine/hef)
    - executing single inference
    - cleaning up resources
    """

    name: str
    capabilities: BackendCaps

    def prepare(self, run_cfg: RunCfg, artifacts_dir: Path) -> PreparedHandle: ...

    def run(self, prepared: PreparedHandle, inputs: dict) -> BackendRunOut: ...

    def cleanup(self, prepared: PreparedHandle) -> None: ...
