#!/usr/bin/env python3
"""Run one sealed DeepX Full prepared-input energy hotloop.

The numeric input is loaded exclusively from a preflight-attested tensor file.
This command never decodes an image and never performs preprocessing.  Every
detection work unit includes its frozen completion path, and the last measured
Completed-v2 result is persisted as an independently verifiable JSON artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import socket
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


CONTRACT_VERSION = "deepx-sealed-runtime-input-v3"
PREFLIGHT_SCHEMA = "onnx-splitpoint/energy-preflight-attestation"


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _strict_sha256(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text.startswith("sha256:"):
        text = text[7:]
    return (
        text
        if len(text) == 64
        and all(character in "0123456789abcdef" for character in text)
        else ""
    )


def _absolute_without_resolving(path: Path) -> Path:
    candidate = path.expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    return candidate


def _path_contains_symlink(path: Path) -> bool:
    candidate = _absolute_without_resolving(path)
    current = Path(candidate.anchor)
    for part in candidate.parts[1:]:
        current = current / part
        if current.is_symlink():
            return True
    return False


def _confined_regular_file(
    path: Path, *, allowed_root: Path, expected_path: Path,
) -> bool:
    candidate = _absolute_without_resolving(path)
    root = _absolute_without_resolving(allowed_root)
    exact = _absolute_without_resolving(expected_path)
    if (
        _path_contains_symlink(candidate)
        or _path_contains_symlink(root)
        or _path_contains_symlink(exact)
        or not candidate.is_file()
        or not root.is_dir()
    ):
        return False
    try:
        resolved = candidate.resolve(strict=True)
        resolved_root = root.resolve(strict=True)
        resolved.relative_to(resolved_root)
        return resolved == exact.resolve(strict=True)
    except (OSError, RuntimeError, ValueError):
        return False


def _write_report(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _fail(report: Path, status: str, **extra: Any) -> int:
    payload = {"ok": False, "status": status, **extra}
    try:
        _write_report(report, payload)
    except Exception:
        pass
    print(json.dumps(payload), file=sys.stderr, flush=True)
    return 5


def _verify_preflight(
    path: Path,
    *,
    nonce: str,
    contract_sha256: str,
    runner_sha256: str,
    dxnn_sha256: str,
    prepared_input_sha256: str,
) -> bool:
    """Verify the fresh attestation, including the exact runtime tensor."""
    try:
        if not path.is_file() or path.stat().st_size > 1024 * 1024:
            return False
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping):
            return False
        sealed = dict(raw)
        declared = _strict_sha256(sealed.pop("attestation_sha256", ""))
        artifacts = raw.get("verified_artifact_sha256")
        if not isinstance(artifacts, Mapping):
            return False
        now_ns = time.time_ns()
        return bool(
            raw.get("schema") == PREFLIGHT_SCHEMA
            and int(raw.get("schema_version") or 0) == 1
            and raw.get("ok") is True
            and str(raw.get("nonce") or "") == nonce
            and _strict_sha256(raw.get("command_contract_sha256"))
            == contract_sha256
            and str(raw.get("artifact_verification_status") or "")
            == "pass"
            and str(raw.get("host") or "") == socket.gethostname()
            and int(raw.get("created_at_unix_ns") or 0)
            <= now_ns
            <= int(raw.get("expires_at_unix_ns") or 0)
            and bool(declared)
            and _canonical_json_sha256(sealed) == declared
            and _strict_sha256(artifacts.get("hotloop_runner"))
            == runner_sha256
            and _strict_sha256(artifacts.get("dxnn")) == dxnn_sha256
            and _strict_sha256(
                artifacts.get("runtime_input_tensor")
            )
            == prepared_input_sha256
        )
    except Exception:
        return False


def _parse_json_mapping(text: str, reason: str) -> dict[str, Any]:
    try:
        value = json.loads(str(text))
    except Exception as exc:
        raise ValueError(reason) from exc
    if not isinstance(value, Mapping) or not value:
        raise ValueError(reason)
    return dict(value)


def _parse_shape(text: str) -> list[int]:
    try:
        value = json.loads(str(text))
    except Exception as exc:
        raise ValueError("prepared_input_shape_invalid") from exc
    if (
        not isinstance(value, list)
        or not value
        or any(
            isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or dimension <= 0
            for dimension in value
        )
    ):
        raise ValueError("prepared_input_shape_invalid")
    return [int(dimension) for dimension in value]


def _shape_geometry(shape: list[int], layout: str) -> dict[str, Any]:
    if layout == "HWC" and len(shape) == 3:
        height, width, channels = shape
        batch = None
    elif layout == "CHW" and len(shape) == 3:
        channels, height, width = shape
        batch = None
    elif layout == "NHWC" and len(shape) == 4:
        batch, height, width, channels = shape
    elif layout == "NCHW" and len(shape) == 4:
        batch, channels, height, width = shape
    else:
        raise ValueError("prepared_input_layout_shape_mismatch")
    if batch not in (None, 1) or channels not in {1, 3, 4}:
        raise ValueError("prepared_input_layout_shape_mismatch")
    return {
        "target_hw": [int(height), int(width)],
        "channels": int(channels),
    }


def _parse_original_wh(text: str) -> list[int]:
    try:
        value = json.loads(str(text))
    except Exception as exc:
        raise ValueError("original_image_wh_invalid") from exc
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(
            isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or dimension <= 0
            for dimension in value
        )
    ):
        raise ValueError("original_image_wh_invalid")
    return [int(value[0]), int(value[1])]


def _load_prepared_input(
    path: Path,
    *,
    expected_sha256: str,
    expected_bytes: int,
    expected_name: str,
    expected_shape: list[int],
    expected_dtype: str,
    expected_layout: str,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    """Read, hash, and materialize exactly the sealed tensor bytes."""
    if _path_contains_symlink(path) or not path.is_file():
        raise ValueError("prepared_input_file_invalid")
    if not expected_name.strip():
        raise ValueError("prepared_input_name_invalid")
    layout = str(expected_layout or "").strip().upper()
    geometry = _shape_geometry(expected_shape, layout)
    dtype_token = str(expected_dtype or "").strip().lower()
    try:
        dtype = np.dtype(dtype_token)
    except Exception as exc:
        raise ValueError("prepared_input_dtype_invalid") from exc
    if dtype_token != str(dtype) or str(dtype) not in {
        "uint8", "float16", "float32",
    }:
        raise ValueError("prepared_input_dtype_invalid")
    calculated_bytes = int(math.prod(expected_shape) * dtype.itemsize)
    if (
        isinstance(expected_bytes, bool)
        or expected_bytes <= 0
        or expected_bytes != calculated_bytes
        or int(path.stat().st_size) != calculated_bytes
    ):
        raise ValueError("prepared_input_byte_count_mismatch")
    tensor_bytes = path.read_bytes()
    if len(tensor_bytes) != calculated_bytes:
        raise ValueError("prepared_input_byte_count_mismatch")
    actual_sha256 = hashlib.sha256(tensor_bytes).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError("prepared_input_sha256_mismatch")
    tensor = np.frombuffer(tensor_bytes, dtype=dtype)
    if int(tensor.size) != int(math.prod(expected_shape)):
        raise ValueError("prepared_input_element_count_mismatch")
    feed = tensor.reshape(expected_shape).copy(order="C")
    if feed.tobytes(order="C") != tensor_bytes:
        raise ValueError("prepared_input_roundtrip_mismatch")
    return feed, geometry, {
        "runtime_input_binding_verified": True,
        "runtime_input_source": "preflight_bound_runtime_input_tensor",
        "runtime_input_file": str(_absolute_without_resolving(path)),
        "runtime_input_sha256": actual_sha256,
        "runtime_input_file_sha256": actual_sha256,
        "runtime_input_bytes": calculated_bytes,
        "runtime_input_name": expected_name.strip(),
        "runtime_input_shape": list(expected_shape),
        "runtime_input_dtype": str(dtype),
        "runtime_input_layout": layout,
    }


def _verify_runtime_identities(
    *,
    task: str,
    geometry: Mapping[str, Any],
    input_binding: Mapping[str, Any],
    preprocessing_json: str,
    preprocessing_sha256: str,
    numeric_json: str,
    numeric_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from onnx_splitpoint_tool.preprocessing_contract import (
        resolve_image_preprocessing_contract,
        runtime_numeric_input_identity,
        runtime_numeric_input_identity_errors,
    )

    preprocessing = _parse_json_mapping(
        preprocessing_json, "runtime_preprocessing_identity_invalid",
    )
    numeric = _parse_json_mapping(
        numeric_json, "runtime_numeric_input_identity_invalid",
    )
    if _canonical_json_sha256(preprocessing) != preprocessing_sha256:
        raise ValueError("runtime_preprocessing_identity_sha256_mismatch")
    verified_preprocessing, calculated_preprocessing_sha = (
        resolve_image_preprocessing_contract(
            task=task,
            target_hw=list(geometry.get("target_hw") or []),
            declared=preprocessing,
        )
    )
    if calculated_preprocessing_sha != preprocessing_sha256:
        raise ValueError("runtime_preprocessing_identity_sha256_mismatch")
    if _canonical_json_sha256(numeric) != numeric_sha256:
        raise ValueError("runtime_numeric_input_identity_sha256_mismatch")
    errors = runtime_numeric_input_identity_errors(
        numeric, verified_preprocessing,
    )
    if errors:
        raise ValueError(
            "runtime_numeric_input_identity_invalid:" + ",".join(errors)
        )
    expected_numeric, calculated_numeric_sha = runtime_numeric_input_identity(
        backend="native_full_deepx",
        task=task,
        preprocessing_contract_sha256_value=preprocessing_sha256,
        runtime_input_name=input_binding["runtime_input_name"],
        runtime_input_shape=input_binding["runtime_input_shape"],
        runtime_input_dtype=input_binding["runtime_input_dtype"],
        runtime_input_layout=input_binding["runtime_input_layout"],
        runtime_color_space=numeric.get("runtime_color_space"),
        runtime_normalization=numeric.get("runtime_normalization"),
    )
    if (
        dict(numeric) != expected_numeric
        or calculated_numeric_sha != numeric_sha256
    ):
        raise ValueError("runtime_numeric_input_identity_metadata_mismatch")
    return verified_preprocessing, expected_numeric


def _named_outputs(raw: Any, names: list[str]) -> dict[str, Any]:
    values = list(raw) if isinstance(raw, (list, tuple)) else [raw]
    if len(values) != len(names):
        raise ValueError(
            f"frozen_postprocess_output_count_mismatch:{len(values)}!={len(names)}"
        )
    return {name: value for name, value in zip(names, values)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dxnn", required=True)
    parser.add_argument("--prepared-input-file", required=True)
    parser.add_argument("--expected-prepared-input-path", default="")
    parser.add_argument("--expected-prepared-input-root", default="")
    parser.add_argument("--expected-prepared-input-sha256", required=True)
    parser.add_argument("--expected-prepared-input-bytes", type=int, required=True)
    parser.add_argument("--expected-prepared-input-name", required=True)
    parser.add_argument("--expected-prepared-input-shape-json", required=True)
    parser.add_argument("--expected-prepared-input-dtype", required=True)
    parser.add_argument("--expected-prepared-input-layout", required=True)
    parser.add_argument("--runtime-preprocessing-identity-json", required=True)
    parser.add_argument("--expected-runtime-preprocessing-sha256", required=True)
    parser.add_argument("--runtime-numeric-input-identity-json", required=True)
    parser.add_argument("--expected-runtime-numeric-input-sha256", required=True)
    parser.add_argument("--original-image-wh-json", required=True)
    parser.add_argument("--prepared-feed-contract-version", required=True)
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument(
        "--duration-s", type=float, default=0.0,
        help=(
            "Minimum measured duration; --frames remains a minimum work "
            "budget."
        ),
    )
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument(
        "--task", required=True, choices=["classification", "detection"],
    )
    parser.add_argument("--frozen-postprocess-contract-json", default="")
    parser.add_argument(
        "--frozen-decoded-nms-normalization-contract-json", default="",
    )
    parser.add_argument("--source-endpoint-contract-hash", default="")
    parser.add_argument("--json-out", required=True)
    parser.add_argument("--expected-runner-sha256", required=True)
    parser.add_argument("--expected-runner-path", default="")
    parser.add_argument("--expected-runner-root", default="")
    parser.add_argument("--expected-dxnn-sha256", required=True)
    parser.add_argument("--expected-dxnn-path", default="")
    parser.add_argument("--expected-dxnn-root", default="")
    parser.add_argument("--source-contract-sha256", required=True)
    parser.add_argument("--preflight-attestation", required=True)
    parser.add_argument("--preflight-nonce", required=True)
    ns = parser.parse_args()

    report = Path(ns.json_out).expanduser().resolve()
    runner_lexical = _absolute_without_resolving(Path(__file__))
    runner = runner_lexical.resolve()
    dxnn = _absolute_without_resolving(Path(ns.dxnn))
    prepared_input = _absolute_without_resolving(
        Path(ns.prepared_input_file)
    )
    expected = {
        "runner": _strict_sha256(ns.expected_runner_sha256),
        "dxnn": _strict_sha256(ns.expected_dxnn_sha256),
        "runtime_input": _strict_sha256(
            ns.expected_prepared_input_sha256
        ),
        "preprocessing": _strict_sha256(
            ns.expected_runtime_preprocessing_sha256
        ),
        "numeric": _strict_sha256(
            ns.expected_runtime_numeric_input_sha256
        ),
        "source_contract": _strict_sha256(ns.source_contract_sha256),
    }
    if not all(expected.values()):
        return _fail(report, "expected_contract_sha256_invalid")
    if str(ns.prepared_feed_contract_version) != CONTRACT_VERSION:
        return _fail(report, "prepared_feed_contract_version_mismatch")
    expected_prepared_path = Path(
        str(ns.expected_prepared_input_path or "")
    )
    expected_prepared_root = Path(
        str(ns.expected_prepared_input_root or "")
    )
    expected_dxnn_path = Path(str(ns.expected_dxnn_path or ""))
    expected_dxnn_root = Path(str(ns.expected_dxnn_root or ""))
    expected_runner_path = Path(str(ns.expected_runner_path or ""))
    expected_runner_root = Path(str(ns.expected_runner_root or ""))
    if (
        not str(ns.expected_prepared_input_path or "").strip()
        or not str(ns.expected_prepared_input_root or "").strip()
        or not str(ns.expected_dxnn_path or "").strip()
        or not str(ns.expected_dxnn_root or "").strip()
        or not str(ns.expected_runner_path or "").strip()
        or not str(ns.expected_runner_root or "").strip()
        or not _confined_regular_file(
            runner_lexical, allowed_root=expected_runner_root,
            expected_path=expected_runner_path,
        )
        or not _confined_regular_file(
            prepared_input, allowed_root=expected_prepared_root,
            expected_path=expected_prepared_path,
        )
        or not _confined_regular_file(
            dxnn, allowed_root=expected_dxnn_root,
            expected_path=expected_dxnn_path,
        )
    ):
        return _fail(report, "sealed_artifact_role_path_mismatch")
    if not _verify_preflight(
        Path(ns.preflight_attestation).expanduser(),
        nonce=str(ns.preflight_nonce),
        contract_sha256=expected["source_contract"],
        runner_sha256=expected["runner"],
        dxnn_sha256=expected["dxnn"],
        prepared_input_sha256=expected["runtime_input"],
    ):
        return _fail(report, "energy_preflight_attestation_invalid")
    if (
        not runner.is_file()
        or not dxnn.is_file()
        or not prepared_input.is_file()
    ):
        return _fail(
            report, "preflight_verified_artifact_missing_at_execution",
        )

    try:
        shape = _parse_shape(ns.expected_prepared_input_shape_json)
        feed, geometry, input_binding = _load_prepared_input(
            prepared_input,
            expected_sha256=expected["runtime_input"],
            expected_bytes=int(ns.expected_prepared_input_bytes),
            expected_name=str(ns.expected_prepared_input_name),
            expected_shape=shape,
            expected_dtype=str(ns.expected_prepared_input_dtype),
            expected_layout=str(ns.expected_prepared_input_layout),
        )
        original_wh = _parse_original_wh(ns.original_image_wh_json)
        preprocessing_identity, numeric_identity = (
            _verify_runtime_identities(
                task=str(ns.task),
                geometry=geometry,
                input_binding=input_binding,
                preprocessing_json=str(
                    ns.runtime_preprocessing_identity_json
                ),
                preprocessing_sha256=expected["preprocessing"],
                numeric_json=str(ns.runtime_numeric_input_identity_json),
                numeric_sha256=expected["numeric"],
            )
        )

        from onnx_splitpoint_tool.native_detection_postprocess import (
            FrozenDecodedNmsPostprocessor,
            FrozenDetectionPostprocessor,
            build_completed_detection_endpoint_attestation,
            build_letterbox_geometry_contract,
            build_normalized_detection_endpoint_attestation,
            persist_completed_result_artifact,
            verify_frozen_decoded_nms_normalization_contract,
            verify_frozen_postprocess_contract,
        )

        frozen_contract_text = str(
            ns.frozen_postprocess_contract_json or ""
        ).strip()
        direct_contract_text = str(
            ns.frozen_decoded_nms_normalization_contract_json or ""
        ).strip()
        if str(ns.task) == "detection" and (
            bool(frozen_contract_text) == bool(direct_contract_text)
        ):
            raise ValueError(
                "deepx_detection_exactly_one_completion_contract_required"
            )
        if str(ns.task) == "classification" and (
            frozen_contract_text or direct_contract_text
        ):
            raise ValueError(
                "deepx_classification_completion_contract_forbidden"
            )

        completion_processor: Any = None
        completion_kind = "not_applicable"
        completion_contract: dict[str, Any] = {}
        output_names: list[str] = []
        if frozen_contract_text:
            completion_contract = verify_frozen_postprocess_contract(
                json.loads(frozen_contract_text)
            )
            source_endpoint_hash = _strict_sha256(
                ns.source_endpoint_contract_hash
            )
            if not source_endpoint_hash:
                raise ValueError("source_endpoint_contract_hash_invalid")
            if (
                list(completion_contract.get("original_wh") or [])
                != original_wh
                or list(completion_contract.get("input_hw") or [])
                != list(geometry["target_hw"])
            ):
                raise ValueError(
                    "frozen_postprocess_input_geometry_mismatch"
                )
            signature = completion_contract.get(
                "raw_output_tensor_signature"
            )
            tensors = (
                signature.get("tensors")
                if isinstance(signature, Mapping) else None
            )
            if not isinstance(tensors, list) or not tensors:
                raise ValueError(
                    "frozen_postprocess_tensor_signature_missing"
                )
            output_names = [str(item.get("name") or "") for item in tensors]
            if (
                any(not name for name in output_names)
                or len(set(output_names)) != len(output_names)
            ):
                raise ValueError("frozen_postprocess_output_names_invalid")
            completion_processor = FrozenDetectionPostprocessor(
                completion_contract
            )
            completion_kind = "raw_host_tail"
        elif direct_contract_text:
            completion_contract = (
                verify_frozen_decoded_nms_normalization_contract(
                    json.loads(direct_contract_text)
                )
            )
            expected_geometry = build_letterbox_geometry_contract(
                input_hw=list(geometry["target_hw"]),
                original_wh=original_wh,
                preprocess={
                    "mode": preprocessing_identity.get(
                        "preprocess_mode"
                    ),
                    "letterbox_pad_value": preprocessing_identity.get(
                        "letterbox_pad_value"
                    ),
                    "color_space": preprocessing_identity.get(
                        "color_space"
                    ),
                },
            )
            if dict(
                completion_contract["letterbox_geometry_contract"]
            ) != dict(expected_geometry):
                raise ValueError(
                    "direct_normalization_input_geometry_mismatch"
                )
            signature = completion_contract.get(
                "source_output_tensor_signature"
            )
            tensors = (
                signature.get("tensors")
                if isinstance(signature, Mapping) else None
            )
            if not isinstance(tensors, list) or not tensors:
                raise ValueError(
                    "direct_normalization_tensor_signature_missing"
                )
            output_names = [str(item.get("name") or "") for item in tensors]
            if (
                any(not name for name in output_names)
                or len(set(output_names)) != len(output_names)
            ):
                raise ValueError(
                    "direct_normalization_output_names_invalid"
                )
            completion_processor = FrozenDecodedNmsPostprocessor(
                completion_contract
            )
            completion_kind = "direct_bn6_normalization"

        if str(ns.task) == "classification":
            from onnx_splitpoint_tool.runners.harness.classification import ClassificationCompletion
            completion_processor = ClassificationCompletion()
            completion_kind = "classification_top1_top5"

        def complete_outputs(raw):
            if str(ns.task) == "classification":
                values = list(raw) if isinstance(raw, (list, tuple)) else [raw]
                outputs = dict(raw) if isinstance(raw, Mapping) else {
                    f"output_{index}": value for index, value in enumerate(values)
                }
            else:
                outputs = _named_outputs(raw, output_names)
            return completion_processor.process(outputs, original_wh=original_wh)

        # Engine import/initialization is deliberately after every sealed
        # input and metadata check above.  No failing binding can infer once.
        if (
            not _confined_regular_file(
                runner_lexical, allowed_root=expected_runner_root,
                expected_path=expected_runner_path,
            )
            or not _confined_regular_file(
                prepared_input, allowed_root=expected_prepared_root,
                expected_path=expected_prepared_path,
            )
            or not _confined_regular_file(
                dxnn, allowed_root=expected_dxnn_root,
                expected_path=expected_dxnn_path,
            )
        ):
            raise ValueError("sealed_artifact_role_path_changed")
        from dx_engine import InferenceEngine  # type: ignore

        engine = InferenceEngine(str(dxnn))
        frames = max(1, int(ns.frames))
        requested_duration_s = max(0.0, float(ns.duration_s or 0.0))
        warmup = max(0, int(ns.warmup))
        for _ in range(warmup):
            raw = engine.run([feed])
            if completion_processor is not None:
                complete_outputs(raw)

        completion_result: dict[str, Any] = {}
        timings_ms: list[float] = []
        measured_started = time.perf_counter()
        while (
            len(timings_ms) < frames
            or time.perf_counter() - measured_started
            < requested_duration_s
        ):
            started = time.perf_counter()
            raw = engine.run([feed])
            if completion_processor is not None:
                completion_result = complete_outputs(raw)
            timings_ms.append((time.perf_counter() - started) * 1000.0)

        completed = len(timings_ms)
        measured_duration_s = max(
            0.0, time.perf_counter() - measured_started,
        )
        minimum_duration_satisfied = bool(
            requested_duration_s <= 0.0
            or measured_duration_s >= requested_duration_s
        )
        postprocess_required = str(ns.task) == "detection"
        postprocess_completed = (
            int(completion_processor.completed_count) - warmup
            if completion_processor is not None else 0
        )
        postprocess_count_ok = bool(
            completion_processor is not None
            and postprocess_completed == completed
        )

        completion_attestation: dict[str, Any] = {}
        persistence: dict[str, Any] = {
            "completed_task_result_artifact_saved": False,
            "completed_task_result_artifact": {},
            "completed_task_result_artifact_sha256": "",
            "completed_task_result_artifact_path": "",
            "completed_task_result_artifact_file_sha256": "",
        }
        completed_v2_ok = not postprocess_required
        if postprocess_required and postprocess_count_ok and completed > 0:
            if completion_kind == "raw_host_tail":
                completion_attestation = (
                    build_completed_detection_endpoint_attestation(
                        completion_contract,
                        completion_result,
                        completed_frames=completed,
                        postprocess_completed_frames=postprocess_completed,
                        source_endpoint_contract_hash=_strict_sha256(
                            ns.source_endpoint_contract_hash
                        ),
                    )
                )
            else:
                completion_attestation = (
                    build_normalized_detection_endpoint_attestation(
                        completion_contract,
                        completion_result,
                        completed_frames=completed,
                        postprocess_completed_frames=postprocess_completed,
                    )
                )
            result_artifact = completion_result.get(
                "completed_result_artifact"
            )
            result_artifact_sha256 = _strict_sha256(
                completion_result.get(
                    "completed_result_artifact_sha256"
                )
            )
            persistence = persist_completed_result_artifact(
                result_artifact,
                expected_sha256=result_artifact_sha256,
                output_path=report.with_name(
                    f"{report.stem}.completed_task_result_artifact.json"
                ),
            )
            completed_v2_ok = bool(
                completion_attestation.get("attested") is True
                and persistence[
                    "completed_task_result_artifact_saved"
                ] is True
                and persistence[
                    "completed_task_result_artifact_sha256"
                ]
                == persistence[
                    "completed_task_result_artifact_file_sha256"
                ]
            )

        ok = bool(
            completed >= frames
            and minimum_duration_satisfied
            and postprocess_count_ok
            and completed_v2_ok
        )
        endpoint_contract = completion_attestation.get(
            "completed_endpoint_contract"
        )
        payload = {
            "ok": ok,
            "status": "ok" if ok else "completed_work_units_mismatch",
            "benchmark_kind": "prepared_feed_dx_engine_energy_hotloop",
            "prepared_feed_contract_version": CONTRACT_VERSION,
            "requested_work_units": frames,
            "minimum_requested_work_units": frames,
            "completed_frames": completed,
            "completed_work_units": completed,
            "completed_work_units_source": (
                "dx_engine_output_plus_frozen_completion_success_counter"
                if postprocess_required
                else "dx_engine_prepared_feed_timed_loop"
            ),
            "completed_work_units_status": (
                "exact_runtime_counter" if ok else "count_mismatch"
            ),
            "warmup_count": warmup,
            "requested_duration_s": requested_duration_s,
            "measured_duration_s": measured_duration_s,
            "minimum_duration_satisfied": minimum_duration_satisfied,
            "measurement_control": (
                "minimum_frames_and_duration"
                if requested_duration_s > 0.0 else "exact_frames"
            ),
            "dxnn": str(dxnn),
            "dxnn_sha256": expected["dxnn"],
            **input_binding,
            "runtime_preprocessing_identity": preprocessing_identity,
            "runtime_preprocessing_sha256": expected["preprocessing"],
            "runtime_numeric_input_identity": numeric_identity,
            "runtime_numeric_input_sha256": expected["numeric"],
            "original_image_wh": original_wh,
            "image_decode_performed": False,
            "preprocessing_performed": False,
            "preprocessing_timed": False,
            "disk_io_timed": False,
            "e2e_scope": (
                "full_task_pipeline"
                if postprocess_required else "accelerator_only"
            ),
            "postprocess_included": postprocess_required,
            "host_postprocess_frozen": completion_kind == "raw_host_tail",
            "normalization_frozen": (
                completion_kind == "direct_bn6_normalization"
            ),
            "frozen_postprocess_contract": (
                completion_contract
                if completion_kind == "raw_host_tail" else {}
            ),
            "frozen_postprocess_contract_sha256": (
                str(completion_contract.get("contract_sha256") or "")
                if completion_kind == "raw_host_tail" else ""
            ),
            "frozen_decoded_nms_normalization_contract": (
                completion_contract
                if completion_kind == "direct_bn6_normalization" else {}
            ),
            "frozen_decoded_nms_normalization_contract_sha256": (
                str(completion_contract.get("contract_sha256") or "")
                if completion_kind == "direct_bn6_normalization" else ""
            ),
            "postprocess_completed_frames": postprocess_completed,
            "postprocess_completion_verified": postprocess_count_ok,
            "postprocess_completed_frames_status": (
                "exact_match" if postprocess_count_ok else "count_mismatch"
            ),
            "frozen_postprocess_result": (
                dict(completion_result)
                if completion_kind == "raw_host_tail" else {}
            ),
            "frozen_decoded_nms_normalization_result": (
                dict(completion_result)
                if completion_kind == "direct_bn6_normalization" else {}
            ),
            "completed_task_stage": str(
                completion_attestation.get("stage") or ""
            ),
            "completed_task_contract_family": (
                "decoded_nms"
                if completion_attestation.get("attested") is True else ""
            ),
            "completed_task_endpoint_contract": (
                dict(endpoint_contract)
                if isinstance(endpoint_contract, Mapping) else {}
            ),
            "completed_task_endpoint_contract_hash": str(
                completion_attestation.get("endpoint_contract_hash") or ""
            ),
            "completed_task_output_endpoint_id": str(
                completion_attestation.get("output_endpoint_id") or ""
            ),
            "completed_task_comparison_endpoint_contract": (
                dict(
                    completion_attestation.get(
                        "completed_task_comparison_endpoint_contract"
                    )
                )
                if isinstance(
                    completion_attestation.get(
                        "completed_task_comparison_endpoint_contract"
                    ),
                    Mapping,
                )
                else {}
            ),
            "completed_task_comparison_endpoint_contract_hash": str(
                completion_attestation.get(
                    "completed_task_comparison_endpoint_contract_hash"
                ) or ""
            ),
            "completed_task_comparison_output_endpoint_id": str(
                completion_attestation.get(
                    "completed_task_comparison_output_endpoint_id"
                ) or ""
            ),
            "completed_task_completion_mode": str(
                completion_attestation.get(
                    "completed_task_completion_mode"
                ) or ""
            ),
            "completed_task_endpoint_attested": (
                completion_attestation.get("attested") is True
            ),
            "completed_task_endpoint_attestation_status": (
                str(completion_attestation.get("status") or "")
                if postprocess_required else "not_applicable"
            ),
            "completed_task_endpoint_attestation": completion_attestation,
            "completed_task_result_artifact_verification_status": (
                "passed"
                if completed_v2_ok and postprocess_required
                else "not_applicable"
                if not postprocess_required
                else "failed"
            ),
            **persistence,
            "source_endpoint_contract_hash": _strict_sha256(
                ns.source_endpoint_contract_hash
            ),
            "mean_ms": (
                sum(timings_ms) / completed if completed else None
            ),
            "source_contract_sha256": expected["source_contract"],
            "preflight_verified": True,
        }
        if str(ns.task) == "classification":
            payload.update(completion_processor.report(postprocess_completed))
            payload["completed_work_units_source"] = "dx_engine_prepared_feed_top1_top5_timed_loop"
        _write_report(report, payload)
        print(json.dumps(payload), flush=True)
        return 0 if ok else 5
    except Exception as exc:
        return _fail(
            report,
            "deepx_full_energy_hotloop_failed",
            error=f"{type(exc).__name__}: {exc}",
        )


if __name__ == "__main__":
    raise SystemExit(main())
