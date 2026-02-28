from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

import numpy as np


@dataclass
class TensorSpec:
    key: str
    shape: tuple[int, ...]
    dtype: str
    nbytes: int


@dataclass
class SerializedTensors:
    """A simple concatenated serialization of multiple tensors."""

    specs: list[TensorSpec]
    blob: bytes

    @property
    def total_bytes(self) -> int:
        return len(self.blob)


def serialize_tensors(tensors: dict[str, np.ndarray]) -> SerializedTensors:
    specs: list[TensorSpec] = []
    chunks: list[bytes] = []

    for key, arr in tensors.items():
        a = np.ascontiguousarray(arr)
        b = a.tobytes()
        specs.append(TensorSpec(key=str(key), shape=tuple(a.shape), dtype=str(a.dtype), nbytes=len(b)))
        chunks.append(b)

    return SerializedTensors(specs=specs, blob=b"".join(chunks))


def deserialize_tensors(serialized: SerializedTensors) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    offset = 0
    blob = serialized.blob

    for spec in serialized.specs:
        part = blob[offset : offset + spec.nbytes]
        offset += spec.nbytes

        arr = np.frombuffer(part, dtype=np.dtype(spec.dtype)).reshape(spec.shape)
        out[spec.key] = arr

    return out
