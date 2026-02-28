from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, Optional, Sequence


Status = Literal["ok", "failed", "partial", "cancelled"]


@dataclass(frozen=True)
class BackendCaps:
    """Capability flags for a backend.

    Keep this minimal and additive only.
    """

    supports_fp16: bool = False
    supports_cache_dir: bool = False
    needs_compiler: bool = False
    supports_two_stage: bool = True


@dataclass
class RunCfg:
    """Backend-specific configuration.

    This is intentionally generic. Backends may interpret `options`.
    """

    model_path: Path
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class SampleCfg:
    """Harness-specific sample configuration.

    For now we model input scaling policy explicitly.

    - input_scale: "raw" | "norm" | "auto"
    - allow_auto_scale: if False, "auto" must be resolved to a safe default.
    """

    input_scale: Literal["raw", "norm", "auto"] = "norm"
    allow_auto_scale: bool = False

    # Optional generic fields. Harnesses may use these.
    image_path: Optional[Path] = None
    input_name: Optional[str] = None
    input_shape: Optional[Sequence[int]] = None

    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class BackendRunOut:
    """Output of a single backend inference call."""

    outputs: Any
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class StagePlan:
    """One stage in a graph."""

    name: str
    backend_name: str
    run_cfg: RunCfg


@dataclass
class GraphPlan:
    """Plan for running a 1-stage or 2-stage graph."""

    stages: list[StagePlan]
    sample_cfg: SampleCfg = field(default_factory=SampleCfg)
    warmup_runs: int = 0
    measured_runs: int = 1

    # If True and stages==2, measure serialize/deserialize.
    measure_interface: bool = True

    # Optional label (useful for artifacts)
    label: str = "graph"


@dataclass
class GraphRunResult:
    """Structured output from GraphRunner."""

    schema_version: int = 1
    status: Status = "failed"

    plan: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)

    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=False, default=_json_default)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
