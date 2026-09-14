from __future__ import annotations

"""Canonical Native Performance/Energy matrix identities.

The configured Native matrix and the imported performance summary use two
different row schemas.  Planned rows name a physical producer (for example
``hailo10h``), while imported Split rows name the complete pipeline (for
example ``hailo10h_to_trt``).  Native Energy must bind both representations to
one denominator without making post-hoc Quality or a display precision part of
row membership.
"""

from typing import Any, Mapping, Sequence


def native_performance_token(value: Any) -> str:
    return str(value or "").strip().lower()


def canonical_native_backend(value: Any) -> str:
    token = native_performance_token(value).replace("-", "_")
    return {
        "hailo10": "hailo10h",
        "hailo10_to_trt": "hailo10h",
        "hailo10h_to_trt": "hailo10h",
        "hailo10_to_tensorrt": "hailo10h",
        "hailo10h_to_tensorrt": "hailo10h",
        "hailo8_to_trt": "hailo8",
        "hailo8_to_tensorrt": "hailo8",
        "deepx_m1": "deepx",
        "deepx_to_trt": "deepx",
        "deepx_to_tensorrt": "deepx",
        "deepx_m1_to_trt": "deepx",
        "deepx_m1_to_tensorrt": "deepx",
    }.get(token, token)


def bind_native_split_expected_setups(
    rows: Sequence[Mapping[str, Any]],
    setup_ids: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Bind every planned Split row to its resolved physical setup.

    Split rows are planned before the hardware registry is resolved.  Energy
    membership, however, includes setup and comparison identity.  Completing
    those dimensions once at plan time prevents one logical row from becoming
    two ledger identities later in the workflow.
    """

    normalized_setups: dict[str, str] = {}
    for raw_producer, raw_setup in setup_ids.items():
        producer = canonical_native_backend(raw_producer)
        setup = str(raw_setup or "").strip()
        if not producer or not setup:
            raise ValueError("native Split setup mapping is incomplete")
        previous = normalized_setups.get(producer)
        if previous is not None and previous != setup:
            raise ValueError(
                f"conflicting setup identities for Native producer {producer}"
            )
        normalized_setups[producer] = setup

    bound: list[dict[str, Any]] = []
    for raw_row in rows:
        row = dict(raw_row)
        if native_performance_mode(row) != "native_split":
            bound.append(row)
            continue
        producer = canonical_native_backend(
            row.get("backend_key") or row.get("backend")
        )
        setup = normalized_setups.get(producer, "")
        if not producer or not setup:
            raise ValueError(
                f"resolved setup identity missing for Native producer {producer!r}"
            )
        declared_setup = native_performance_alias(row, "setup_id", "setup")
        if declared_setup is None or (
            declared_setup and declared_setup != setup.lower()
        ):
            raise ValueError(
                f"planned Native Split setup identity conflicts for {producer}"
            )
        declared_comparison = canonical_native_backend(
            row.get("comparison_backend")
        )
        if declared_comparison and declared_comparison != producer:
            raise ValueError(
                f"planned Native Split comparison identity conflicts for {producer}"
            )
        row["setup_id"] = setup
        row["comparison_backend"] = producer
        if native_performance_identity(row) is None:
            raise ValueError(
                f"planned Native Split identity remains incomplete for {producer}"
            )
        bound.append(row)
    return bound


def native_performance_mode(row: Mapping[str, Any]) -> str:
    mode = native_performance_token(row.get("execution_mode")).replace(
        "-", "_",
    )
    backend_values = {
        native_performance_token(row.get(field)).replace("-", "_")
        for field in ("backend", "producer_backend")
        if row.get(field) not in (None, "")
    }
    if not backend_values and row.get("backend_key") not in (None, ""):
        backend_values.add(
            native_performance_token(row.get("backend_key")).replace(
                "-", "_",
            )
        )
    full_markers = {
        backend.startswith("native_full_") for backend in backend_values
    }
    if len(full_markers) > 1:
        return ""
    backend_is_full = next(iter(full_markers), False)
    if mode in {"native_full", "native_full_baseline", "full"}:
        return "native_full_baseline" if backend_is_full else ""
    if mode not in {"", "native", "native_split", "split"}:
        return ""
    return "" if backend_is_full and mode in {"native_split", "split"} else (
        "native_full_baseline" if backend_is_full else "native_split"
    )


def native_performance_backend_value(value: Any, mode: str) -> str:
    raw = native_performance_token(value).replace("-", "_")
    if mode == "native_full_baseline":
        suffix = raw[len("native_full_"):] if raw.startswith(
            "native_full_"
        ) else raw
        suffix = canonical_native_backend(suffix)
        return f"native_full_{suffix}" if suffix else ""
    producer = canonical_native_backend(raw)
    return {
        "hailo8": "hailo8_to_trt",
        "hailo10h": "hailo10h_to_trt",
        "deepx": "deepx_to_trt",
    }.get(producer, raw)


def native_performance_alias(
    row: Mapping[str, Any], *fields: str,
) -> str | None:
    values = {
        native_performance_token(row.get(field))
        for field in fields
        if row.get(field) not in (None, "")
    }
    if len(values) > 1:
        return None
    return next(iter(values), "")


def native_performance_identity(
    row: Mapping[str, Any],
) -> tuple[str, ...] | None:
    """Return one identity shared by plan, matrix and final-report schemas."""

    mode = native_performance_mode(row)
    if not mode:
        return None
    raw_backends = [
        row.get(field) for field in ("backend", "producer_backend")
        if row.get(field) not in (None, "")
    ]
    if not raw_backends and row.get("backend_key") not in (None, ""):
        raw_backends = [row.get("backend_key")]
    backends = {
        native_performance_backend_value(value, mode)
        for value in raw_backends
    }
    if len(backends) != 1:
        return None
    backend = next(iter(backends))
    model = native_performance_alias(row, "model", "model_id")
    case = native_performance_alias(row, "case", "case_id", "boundary")
    setup_id = native_performance_alias(row, "setup_id", "setup")
    if model is None or case is None or setup_id is None:
        return None
    backend_key = canonical_native_backend(row.get("backend_key"))
    comparison = canonical_native_backend(row.get("comparison_backend"))
    if mode == "native_split":
        producer = canonical_native_backend(backend)
        if backend_key and backend_key != producer:
            return None
        if comparison and comparison != producer:
            return None
        comparison = producer
    else:
        if not comparison or (backend_key and backend_key != comparison):
            return None
        full_producer = canonical_native_backend(
            backend[len("native_full_"):] if backend.startswith(
                "native_full_"
            ) else backend
        )
        if full_producer != "tensorrt" and full_producer != comparison:
            return None
    if mode == "native_full_baseline" and case != "full":
        return None
    identity = (mode, backend, model, case, setup_id, comparison)
    if not all(identity):
        return None
    return identity
