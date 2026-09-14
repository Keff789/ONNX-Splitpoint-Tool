#!/usr/bin/env python3
"""Audit Native Split boundary dumps without accelerator hardware.

The audit verifies dump bytes, hashes, dtype/shape contracts and input identity.
When two backends used the exact same input dump and TensorRT boundary identity,
it also compares layout hypotheses diagnostically.  Results never become claim
evidence and are not written back into the EvaluationRun.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import numpy as np


def _safe_member(value: str) -> str:
    path = PurePosixPath(str(value).replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe ZIP member: {value}")
    return str(path)


def _strict_json(raw: bytes, *, label: str) -> dict[str, Any]:
    duplicate = False

    def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        nonlocal duplicate
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                duplicate = True
            result[key] = value
        return result

    value = json.loads(raw.decode("utf-8"), object_pairs_hook=_object)
    if duplicate:
        raise ValueError(f"duplicate JSON key: {label}")
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {label}")
    return value


class _Pack:
    def __init__(self, source: Path) -> None:
        self.source = source.expanduser().resolve()
        self.archive: zipfile.ZipFile | None = None
        self._members: set[str] = set()
        if self.source.is_file():
            self.archive = zipfile.ZipFile(self.source)
            for info in self.archive.infolist():
                name = _safe_member(info.filename)
                if name in self._members:
                    raise ValueError(f"duplicate ZIP member: {name}")
                self._members.add(name)
        elif not self.source.is_dir():
            raise FileNotFoundError(source)

    def close(self) -> None:
        if self.archive is not None:
            self.archive.close()

    def manifests(self) -> list[str]:
        suffix = "native_fifo_boundary_manifest.json"
        if self.archive is not None:
            return sorted(
                name for name in self._members if name.endswith(suffix)
            )
        return sorted(
            path.relative_to(self.source).as_posix()
            for path in self.source.rglob(suffix)
        )

    def read(self, member: str) -> bytes:
        safe = _safe_member(member)
        if self.archive is not None:
            return self.archive.read(safe)
        path = (self.source / safe).resolve()
        if path != self.source and self.source not in path.parents:
            raise ValueError(f"path escapes debug pack: {member}")
        return path.read_bytes()

    @staticmethod
    def sibling(manifest: str, declared_path: Any) -> str:
        filename = PurePosixPath(str(declared_path).replace("\\", "/")).name
        if not filename or filename in {".", ".."}:
            raise ValueError("payload filename missing")
        return str(PurePosixPath(manifest).parent / _safe_member(filename))


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _positive_shape(value: Any, *, label: str) -> list[int]:
    if (
        not isinstance(value, list)
        or not value
        or any(
            isinstance(item, bool)
            or not isinstance(item, int)
            or item <= 0
            for item in value
        )
    ):
        raise ValueError(f"{label} invalid")
    return [int(item) for item in value]


def _declared_payload_sha(manifest: Mapping[str, Any], role: str) -> str:
    direct = str(manifest.get(f"{role}_sha256") or "").strip().lower()
    if direct:
        return direct.removeprefix("sha256:")
    for row in list(manifest.get("payload_artifacts") or []):
        if isinstance(row, Mapping) and str(row.get("role") or "") == role:
            return str(row.get("sha256") or "").strip().lower().removeprefix(
                "sha256:"
            )
    return ""


def _views(array: np.ndarray, target_shape: list[int]) -> dict[str, np.ndarray]:
    flat = np.asarray(array).reshape(-1)
    views = {"as_input": flat}
    if len(target_shape) == 4:
        n, c, h, w = target_shape
        if int(flat.size) == n * c * h * w:
            nhwc = flat.reshape(n, h, w, c)
            views["memory_nhwc_to_nchw"] = np.transpose(
                nhwc, (0, 3, 1, 2),
            ).reshape(-1)
    return views


def _backend_from_manifest_path(name: str) -> str:
    parts = {part.lower() for part in PurePosixPath(name).parts}
    if "hailo10h" in parts or "hailo10" in parts:
        return "hailo10h_to_trt"
    if "hailo8" in parts:
        return "hailo8_to_trt"
    if "deepx" in parts:
        return "deepx_to_trt"
    return ""


def _audit_manifest(pack: _Pack, name: str) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    result: dict[str, Any] = {
        "manifest": name,
        "technical_ok": False,
        "status": "invalid",
        "claim_eligible": False,
        "diagnostic_only": True,
    }
    view_map: dict[str, np.ndarray] = {}
    try:
        manifest = _strict_json(pack.read(name), label=name)
        if manifest.get("schema") != "onnx-splitpoint/native-boundary-dump":
            raise ValueError("boundary manifest schema invalid")
        dtype = np.dtype(str(manifest.get("dtype") or ""))
        if dtype.hasobject or dtype.kind not in "fiu":
            raise ValueError("boundary dtype is not numeric")
        shape = _positive_shape(manifest.get("shape"), label="boundary shape")
        payload_name = pack.sibling(name, manifest.get("file"))
        payload = pack.read(payload_name)
        expected_bytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        if len(payload) != expected_bytes:
            raise ValueError("boundary payload byte length mismatch")
        for field in ("nbytes", "file_size_bytes"):
            if manifest.get(field) is not None and int(manifest[field]) != len(payload):
                raise ValueError(f"{field} mismatch")
        declared_sha = _declared_payload_sha(manifest, "file")
        actual_sha = _sha256(payload)
        if declared_sha and declared_sha != actual_sha:
            raise ValueError("boundary payload sha256 mismatch")
        array = np.frombuffer(payload, dtype=dtype).reshape(shape)
        if dtype.kind == "f" and not np.isfinite(array).all():
            raise ValueError("boundary payload contains non-finite values")

        target_shape = manifest.get("trt_input_shape")
        if target_shape is None:
            target_shape = shape
        target_shape = _positive_shape(
            target_shape, label="TensorRT input shape",
        )
        if int(np.prod(target_shape, dtype=np.int64)) != int(array.size):
            raise ValueError("runtime/TRT boundary element count mismatch")
        input_sha = ""
        input_dump = str(manifest.get("input_dump") or "").strip()
        if input_dump:
            input_payload = pack.read(pack.sibling(name, input_dump))
            input_sha = _sha256(input_payload)
            declared_input_sha = _declared_payload_sha(manifest, "input_dump")
            if declared_input_sha and input_sha != declared_input_sha:
                raise ValueError("input dump sha256 mismatch")
            input_shape = _positive_shape(
                manifest.get("input_shape_hwc"), label="input shape",
            )
            if len(input_payload) != int(np.prod(input_shape, dtype=np.int64)):
                raise ValueError("input dump byte length mismatch")

        sample = array.reshape(-1)
        stride = max(1, int(math.ceil(sample.size / 200_000)))
        sampled = sample[::stride].astype(np.float64, copy=False)
        result.update({
            "status": "technical_pass",
            "technical_ok": True,
            "schema_version": int(manifest.get("schema_version") or 0),
            "backend": str(
                manifest.get("backend") or _backend_from_manifest_path(name)
            ),
            "dtype": str(dtype),
            "shape": shape,
            "trt_input_name": str(manifest.get("trt_input_name") or ""),
            "trt_input_shape": target_shape,
            "boundary_layout": str(manifest.get("boundary_layout") or ""),
            "payload_member": payload_name,
            "payload_sha256": actual_sha,
            "payload_size_bytes": len(payload),
            "input_dump_sha256": input_sha,
            "element_count": int(array.size),
            "summary": {
                "minimum": float(np.min(sampled)),
                "maximum": float(np.max(sampled)),
                "mean": float(np.mean(sampled)),
                "standard_deviation": float(np.std(sampled)),
                "zero_fraction": float(np.mean(sampled == 0)),
                "sample_count": int(sampled.size),
                "sample_stride": stride,
            },
        })
        view_map = _views(array, target_shape)
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result, view_map


def _linear_compare(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    count = min(int(left.size), int(right.size))
    stride = max(1, int(math.ceil(count / 200_000)))
    x = left[:count:stride].astype(np.float64, copy=False)
    y = right[:count:stride].astype(np.float64, copy=False)
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    x_centered = x - x_mean
    y_centered = y - y_mean
    x_var = float(np.mean(x_centered * x_centered))
    y_var = float(np.mean(y_centered * y_centered))
    if x_var <= 0.0 or y_var <= 0.0:
        raise ValueError("constant peer boundary cannot be correlated")
    covariance = float(np.mean(x_centered * y_centered))
    scale = covariance / x_var
    offset = y_mean - scale * x_mean
    prediction = scale * x + offset
    residual = prediction - y
    correlation = covariance / math.sqrt(x_var * y_var)
    rmse = float(math.sqrt(float(np.mean(residual * residual))))
    return {
        "correlation": correlation,
        "r_squared": correlation * correlation,
        "linear_scale": scale,
        "linear_offset": offset,
        "mae_after_linear_fit": float(np.mean(np.abs(residual))),
        "rmse_after_linear_fit": rmse,
        "rmse_over_reference_std": rmse / math.sqrt(y_var),
        "sample_count": int(x.size),
        "sample_stride": stride,
    }


def audit_pack(source: Path) -> dict[str, Any]:
    pack = _Pack(source)
    rows: list[dict[str, Any]] = []
    views: dict[str, dict[str, np.ndarray]] = {}
    try:
        for name in pack.manifests():
            row, row_views = _audit_manifest(pack, name)
            rows.append(row)
            if row.get("technical_ok") is True:
                views[name] = row_views
    finally:
        pack.close()

    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            row.get("input_dump_sha256"),
            row.get("trt_input_name"),
            tuple(row.get("trt_input_shape") or []),
            row.get("element_count"),
        )
        if row.get("technical_ok") is True and key[0] and key[1]:
            groups.setdefault(key, []).append(row)

    comparisons: list[dict[str, Any]] = []
    for key, peers in groups.items():
        for left_index in range(len(peers)):
            for right_index in range(left_index + 1, len(peers)):
                left = peers[left_index]
                right = peers[right_index]
                if left.get("backend") == right.get("backend"):
                    continue
                hypotheses: list[dict[str, Any]] = []
                for left_layout, left_array in views[left["manifest"]].items():
                    for right_layout, right_array in views[right["manifest"]].items():
                        try:
                            metrics = _linear_compare(left_array, right_array)
                        except Exception:
                            continue
                        hypotheses.append({
                            "left_layout": left_layout,
                            "right_layout": right_layout,
                            **metrics,
                        })
                hypotheses.sort(
                    key=lambda item: abs(float(item["correlation"])),
                    reverse=True,
                )
                comparisons.append({
                    "left_manifest": left["manifest"],
                    "left_backend": left.get("backend"),
                    "right_manifest": right["manifest"],
                    "right_backend": right.get("backend"),
                    "input_dump_sha256": key[0],
                    "trt_input_name": key[1],
                    "trt_input_shape": list(key[2]),
                    "identity_gate": "exact_input_sha_tensor_name_shape",
                    "best_hypothesis": hypotheses[0] if hypotheses else {},
                    "hypotheses": hypotheses,
                    "diagnostic_only": True,
                    "claim_eligible": False,
                })

    failures = [row for row in rows if row.get("technical_ok") is not True]
    return {
        "schema": "onnx-splitpoint/offline-native-boundary-contract-audit",
        "schema_version": 1,
        "source": str(source),
        "manifest_count": len(rows),
        "technical_pass_count": len(rows) - len(failures),
        "technical_failure_count": len(failures),
        "eligible_peer_comparison_count": len(comparisons),
        "ok": bool(rows) and not failures,
        "diagnostic_only": True,
        "claim_eligible": False,
        "rows": rows,
        "peer_comparisons": comparisons,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate Native Split boundary dumps and compare exact-input peer "
            "layout hypotheses without accelerator hardware."
        )
    )
    parser.add_argument("debug_pack", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    try:
        report = audit_pack(args.debug_pack)
    except Exception as exc:
        report = {
            "schema": "onnx-splitpoint/offline-native-boundary-contract-audit",
            "schema_version": 1,
            "source": str(args.debug_pack),
            "ok": False,
            "diagnostic_only": True,
            "claim_eligible": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    encoded = json.dumps(
        report, indent=2, sort_keys=True, ensure_ascii=False,
    ) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if report.get("ok") is True or not args.strict else 3


if __name__ == "__main__":
    raise SystemExit(main())
