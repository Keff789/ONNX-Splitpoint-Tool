from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .base import PreparedHandle
from .._types import BackendCaps, BackendRunOut, RunCfg


@dataclass
class _OrtPrepared:
    session: Any
    input_names: list[str]
    output_names: list[str]


class OrtBackend:
    """ONNXRuntime backend.

    This backend is configured by provider list + optional provider options.

    Notes:
    - We import onnxruntime lazily so the module can be imported even in
      environments without ORT (tests may skip).
    """

    def __init__(
        self,
        name: str,
        providers: list[str],
        provider_options: Optional[list[dict[str, Any]]] = None,
        sess_options: Optional[dict[str, Any]] = None,
    ) -> None:
        self.name = name
        self.providers = list(providers)
        self.provider_options = provider_options
        self.sess_options = sess_options or {}

        self.capabilities = BackendCaps(
            supports_fp16=("TensorrtExecutionProvider" in self.providers
                           or "CUDAExecutionProvider" in self.providers),
            supports_cache_dir=("TensorrtExecutionProvider" in self.providers),
            needs_compiler=("TensorrtExecutionProvider" in self.providers),
            supports_two_stage=True,
        )

    def prepare(self, run_cfg: RunCfg, artifacts_dir: Path) -> PreparedHandle:
        import onnxruntime as ort  # type: ignore

        artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Optional: ensure TRT cache dir exists if configured via options.
        # We keep this generic: user passes provider options.
        if self.provider_options is not None:
            for p, opts in zip(self.providers, self.provider_options):
                if p == "TensorrtExecutionProvider":
                    cache_path = opts.get("trt_engine_cache_path") or opts.get("trt_engine_cache_dir")
                    if cache_path:
                        Path(cache_path).mkdir(parents=True, exist_ok=True)

        so = ort.SessionOptions()

        # Apply a small subset of options via attrs if present.
        for k, v in self.sess_options.items():
            if hasattr(so, k):
                setattr(so, k, v)

        # Some environments benefit from deterministic thread settings.
        # Only set if explicitly requested.

        model_path = str(run_cfg.model_path)

        if self.provider_options is None:
            sess = ort.InferenceSession(model_path, sess_options=so, providers=self.providers)
        else:
            sess = ort.InferenceSession(
                model_path,
                sess_options=so,
                providers=self.providers,
                provider_options=self.provider_options,
            )

        input_names = [i.name for i in sess.get_inputs()]
        output_names = [o.name for o in sess.get_outputs()]

        return PreparedHandle(
            input_names=input_names,
            output_names=output_names,
            handle=_OrtPrepared(session=sess, input_names=input_names, output_names=output_names),
        )

    def run(self, prepared: PreparedHandle, inputs: dict) -> BackendRunOut:
        prep: _OrtPrepared = prepared.handle

        # ORT accepts numpy arrays in dict.
        outs_list = prep.session.run(None, inputs)

        # Convert to stable dict mapping output_name -> ndarray
        outs = {name: arr for name, arr in zip(prep.output_names, outs_list)}
        return BackendRunOut(outputs=outs, metrics={})

    def cleanup(self, prepared: PreparedHandle) -> None:
        # ORT sessions are freed by GC; explicit cleanup is optional.
        # We still drop references to encourage release.
        try:
            prepared.handle.session = None  # type: ignore[attr-defined]
        except Exception:
            pass
