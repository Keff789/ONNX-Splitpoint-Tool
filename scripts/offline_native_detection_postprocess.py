#!/usr/bin/env python3
"""Audit Native-Full detection postprocessing without accelerator hardware.

The input may be an extracted debug-pack directory or the debug-pack ZIP
itself.  Raw tensor bytes are checked before the exact frozen decoder/NMS
contract is replayed.  A contract derived offline from an otherwise unbound
raw dump is deliberately labelled diagnostic and can never become claim
evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from onnx_splitpoint_tool.native_detection_postprocess import (  # noqa: E402
    FrozenDetectionPostprocessor,
    build_frozen_postprocess_contract,
    verify_frozen_postprocess_contract,
)
from onnx_splitpoint_tool.runners.harness.yolo import (  # noqa: E402
    YOLOV7_PAPER_ONNX_SHA256,
)


def _model_sha256_for_replay(
    model_id: str,
    manifest: Mapping[str, Any],
    archived_contract: Mapping[str, Any] | None,
) -> str:
    if "yolov7" not in str(model_id or "").strip().lower():
        return ""
    values: set[str] = set()
    sources = [manifest]
    if isinstance(archived_contract, Mapping):
        sources.append(archived_contract)
    for source in sources:
        for key in (
            "model_sha256", "full_model_sha256",
            "source_onnx_sha256", "onnx_sha256",
        ):
            value = str(source.get(key) or "").strip().lower()
            if value.startswith("sha256:"):
                value = value.split(":", 1)[1]
            if value:
                values.add(value)
    if values != {YOLOV7_PAPER_ONNX_SHA256}:
        raise ValueError(
            "offline_yolov7_replay_exact_model_sha256_required"
        )
    return YOLOV7_PAPER_ONNX_SHA256


def _strict_json_bytes(raw: bytes, *, label: str) -> dict[str, Any]:
    def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON key in {label}: {key}")
            value[key] = item
        return value

    parsed = json.loads(raw.decode("utf-8"), object_pairs_hook=_object)
    if not isinstance(parsed, dict):
        raise ValueError(f"JSON object required: {label}")
    return parsed


def _safe_zip_name(value: str) -> str:
    path = PurePosixPath(str(value).replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe ZIP member: {value}")
    return str(path)


class _Pack:
    def __init__(self, source: Path) -> None:
        self.source = source.resolve()
        self.archive: zipfile.ZipFile | None = None
        if self.source.is_file():
            self.archive = zipfile.ZipFile(self.source)
            for info in self.archive.infolist():
                _safe_zip_name(info.filename)
        elif not self.source.is_dir():
            raise FileNotFoundError(source)

    def close(self) -> None:
        if self.archive is not None:
            self.archive.close()

    def manifests(self) -> list[str]:
        if self.archive is not None:
            return sorted(
                _safe_zip_name(info.filename)
                for info in self.archive.infolist()
                if info.filename.endswith("native_full_outputs_manifest.json")
                and not info.is_dir()
            )
        return sorted(
            str(path.relative_to(self.source).as_posix())
            for path in self.source.rglob("native_full_outputs_manifest.json")
        )

    def read(self, member: str) -> bytes:
        safe = _safe_zip_name(member)
        if self.archive is not None:
            return self.archive.read(safe)
        path = (self.source / safe).resolve()
        if path != self.source and self.source not in path.parents:
            raise ValueError(f"path escapes debug pack: {member}")
        return path.read_bytes()

    def sibling(self, manifest: str, filename: str) -> str:
        return str(PurePosixPath(manifest).parent / _safe_zip_name(filename))


def _load_outputs(
    pack: _Pack, manifest_name: str, manifest: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], list[str]]:
    outputs: dict[str, np.ndarray] = {}
    warnings: list[str] = []
    rows = manifest.get("outputs")
    if not isinstance(rows, list) or not rows:
        raise ValueError("output entries missing")
    for index, raw_entry in enumerate(rows):
        if not isinstance(raw_entry, Mapping):
            raise ValueError(f"output entry {index} is not an object")
        filename = _safe_zip_name(str(raw_entry.get("file") or ""))
        if not filename or "/" in filename:
            raise ValueError(f"output entry {index} has unsafe relative file")
        data = pack.read(pack.sibling(manifest_name, filename))
        dtype = np.dtype(str(raw_entry.get("dtype") or ""))
        if dtype.hasobject:
            raise ValueError(f"output entry {index} uses object dtype")
        shape = raw_entry.get("shape")
        if (
            not isinstance(shape, list)
            or not shape
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
                for value in shape
            )
        ):
            raise ValueError(f"output entry {index} has invalid shape")
        expected_bytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        if len(data) != expected_bytes or int(
            raw_entry.get("bytes") or expected_bytes
        ) != expected_bytes:
            raise ValueError(f"output entry {index} byte length mismatch")
        declared_sha = str(raw_entry.get("sha256") or "").strip().lower()
        actual_sha = hashlib.sha256(data).hexdigest()
        if declared_sha and declared_sha != actual_sha:
            raise ValueError(f"output entry {index} sha256 mismatch")
        if not declared_sha:
            warnings.append(f"output_{index}_sha256_not_recorded")
        array = np.frombuffer(data, dtype=dtype).reshape(shape).copy()
        if array.dtype.kind not in "fiu" or not np.isfinite(array).all():
            raise ValueError(f"output entry {index} contains non-finite values")
        outputs[str(raw_entry.get("name") or f"output_{index}")] = array
    return outputs, warnings


def _input_geometry(
    pack: _Pack, manifest_name: str, manifest: Mapping[str, Any],
) -> tuple[list[int], list[int], str]:
    contract = manifest.get("frozen_host_postprocess_contract")
    if isinstance(contract, Mapping):
        return (
            [int(value) for value in contract.get("input_hw") or []],
            [int(value) for value in contract.get("original_wh") or []],
            "frozen_contract",
        )
    try:
        input_manifest = _strict_json_bytes(
            pack.read(pack.sibling(
                manifest_name, "native_full_input_manifest.json",
            )),
            label=f"{manifest_name}:input",
        )
    except Exception:
        input_manifest = {}
    shape = (
        input_manifest.get("runtime_input_shape")
        or input_manifest.get("input_shape_hwc")
        or []
    )
    input_hw: list[int] = []
    if isinstance(shape, list):
        dims = [int(value) for value in shape]
        if len(dims) == 4 and dims[1] in (1, 3, 4):
            input_hw = [dims[2], dims[3]]
        elif len(dims) == 4:
            input_hw = [dims[1], dims[2]]
        elif len(dims) == 3 and dims[0] in (1, 3, 4):
            input_hw = [dims[1], dims[2]]
        elif len(dims) == 3:
            input_hw = [dims[0], dims[1]]
    original_wh = manifest.get("original_image_wh")
    if (
        isinstance(original_wh, list)
        and len(original_wh) == 2
        and all(int(value) > 0 for value in original_wh)
    ):
        return input_hw, [int(value) for value in original_wh], "manifest"
    if len(input_hw) == 2 and all(value > 0 for value in input_hw):
        return input_hw, [input_hw[1], input_hw[0]], "model_input_fallback"
    return [], [], "unavailable"


def _validate_detections(
    detections: list[dict[str, Any]], original_wh: list[int],
) -> list[str]:
    warnings: list[str] = []
    if not detections:
        return ["empty_detection_result"]
    width, height = original_wh
    for index, detection in enumerate(detections):
        values = [
            detection.get("x1"), detection.get("y1"),
            detection.get("x2"), detection.get("y2"),
            detection.get("score"), detection.get("class_id"),
        ]
        if not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and np.isfinite(float(value))
            for value in values
        ):
            raise ValueError(f"detection {index} contains invalid values")
        x1, y1, x2, y2, score, class_id = (
            float(value) for value in values
        )
        if x2 < x1 or y2 < y1:
            raise ValueError(f"detection {index} has inverted coordinates")
        if (
            x1 < -1.0e-3 or y1 < -1.0e-3
            or x2 > width + 1.0e-3 or y2 > height + 1.0e-3
        ):
            raise ValueError(f"detection {index} is outside original image")
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"detection {index} score outside [0,1]")
        if class_id != int(class_id) or not 0 <= int(class_id) < 80:
            raise ValueError(f"detection {index} class id invalid")
    return warnings


def audit_manifest(
    pack: _Pack, manifest_name: str, *, derive_unbound_raw: bool,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "manifest": manifest_name,
        "status": "failed",
        "technical_ok": False,
        "claim_evidence": False,
        "warnings": [],
    }
    try:
        manifest = _strict_json_bytes(
            pack.read(manifest_name), label=manifest_name,
        )
        result.update({
            "model": str(manifest.get("model") or ""),
            "backend": str(manifest.get("backend") or ""),
            "task": str(manifest.get("task") or ""),
            "runtime_contract_family": str(
                manifest.get("contract_family") or ""
            ),
        })
        if result["task"].lower() != "detection":
            result["status"] = "skipped_non_detection"
            return result
        outputs, warnings = _load_outputs(pack, manifest_name, manifest)
        result["warnings"].extend(warnings)
        input_hw, original_wh, geometry_source = _input_geometry(
            pack, manifest_name, manifest,
        )
        result["input_hw"] = input_hw
        result["original_wh"] = original_wh
        result["geometry_source"] = geometry_source
        contract = manifest.get("frozen_host_postprocess_contract")
        contract_source = "archived_frozen_contract"
        declared = manifest.get("authoritative_output_contract_resolution")
        declared_family = (
            str(declared.get("contract_family") or "")
            if isinstance(declared, Mapping) else ""
        )
        if (
            declared_family == "decoded_nms"
            and len(outputs) != 1
        ):
            result["warnings"].append("declared_endpoint_conflict")
        if not isinstance(contract, Mapping):
            if (
                str(manifest.get("contract_family") or "").strip().lower()
                == "decoded_nms"
                and len(outputs) == 1
            ):
                decoded = next(iter(outputs.values()))
                if decoded.ndim < 2 or decoded.shape[-1] < 6:
                    raise ValueError(
                        "decoded_nms endpoint tensor must end in at least six columns"
                    )
                records = decoded.reshape(-1, decoded.shape[-1])[:, :6]
                records = records[records[:, 4] > 0.0]
                if records.size:
                    if np.any(records[:, 2] < records[:, 0]) or np.any(
                        records[:, 3] < records[:, 1]
                    ):
                        raise ValueError("decoded_nms endpoint has inverted boxes")
                    if np.any(records[:, 4] < 0.0) or np.any(
                        records[:, 4] > 1.0
                    ):
                        raise ValueError("decoded_nms endpoint score outside [0,1]")
                result.update({
                    "status": "decoded_endpoint_tensor_valid",
                    "technical_ok": True,
                    "contract_source": "accelerator_decoded_nms_no_host_replay",
                    "detection_count": int(records.shape[0]),
                    "decoder_format": "decoded_nms_tensor",
                })
                if not len(records):
                    result["warnings"].append("empty_detection_result")
                return result
            if not derive_unbound_raw:
                result["status"] = "raw_tensors_without_frozen_contract"
                return result
            if not input_hw or not original_wh:
                result["status"] = "raw_tensors_geometry_unavailable"
                return result
            contract = build_frozen_postprocess_contract(
                model_id=result["model"], outputs=outputs,
                input_hw=input_hw, original_wh=original_wh,
                model_sha256=_model_sha256_for_replay(
                    result["model"], manifest, None,
                ),
            )
            contract_source = "offline_derived_not_claim_evidence"
            result["warnings"].append(contract_source)
        try:
            verified = verify_frozen_postprocess_contract(
                contract, outputs=outputs,
            )
        except Exception as archived_exc:
            if (
                contract_source != "archived_frozen_contract"
                or not derive_unbound_raw
                or not input_hw
                or not original_wh
            ):
                raise
            # A debug pack from an older release correctly fails the current
            # implementation hash binding.  Preserve that fact, then build a
            # current diagnostic-only contract over the same archived tensors
            # so the proposed postprocess can still be tested offline.
            result["warnings"].append(
                "archived_frozen_contract_not_current:"
                f"{type(archived_exc).__name__}:{archived_exc}"
            )
            contract = build_frozen_postprocess_contract(
                model_id=result["model"], outputs=outputs,
                input_hw=input_hw, original_wh=original_wh,
                model_sha256=_model_sha256_for_replay(
                    result["model"], manifest,
                    contract if isinstance(contract, Mapping) else None,
                ),
            )
            contract_source = (
                "offline_current_implementation_replay_not_claim_evidence"
            )
            result["warnings"].append(contract_source)
            verified = verify_frozen_postprocess_contract(
                contract, outputs=outputs,
            )
        if verified.get("legacy_activation_strategy_unbound") is True:
            if (
                not derive_unbound_raw
                or not input_hw
                or not original_wh
            ):
                raise ValueError(
                    "archived YOLOv7 activation strategy is unbound"
                )
            # Verification above preserves the exact archived identity.  It
            # deliberately cannot be executed by the claim/runtime
            # postprocessor because pre-2.72 YOLOv7 contracts did not bind one
            # activation strategy.  For this offline diagnostic only, probe
            # the already archived tensors once and seal a fresh current
            # contract.  FrozenDetectionPostprocessor then still runs exactly
            # one contract-bound decoder; the derived result remains
            # permanently outside claim evidence.
            result["warnings"].append(
                "archived_frozen_contract_activation_strategy_unbound"
            )
            contract = build_frozen_postprocess_contract(
                model_id=result["model"], outputs=outputs,
                input_hw=input_hw, original_wh=original_wh,
                model_sha256=_model_sha256_for_replay(
                    result["model"], manifest,
                    contract if isinstance(contract, Mapping) else None,
                ),
            )
            contract_source = (
                "offline_current_implementation_replay_not_claim_evidence"
            )
            result["warnings"].append(contract_source)
            verified = verify_frozen_postprocess_contract(
                contract, outputs=outputs,
            )
        first = FrozenDetectionPostprocessor(verified)
        first_result = first.process(outputs, original_wh=original_wh)
        second = FrozenDetectionPostprocessor(verified)
        second_result = second.process(outputs, original_wh=original_wh)
        if first_result != second_result:
            raise ValueError("local replay is not deterministic")
        result["warnings"].extend(
            _validate_detections(first.last_detections, original_wh)
        )
        recorded = manifest.get("frozen_host_postprocess_result")
        if isinstance(recorded, Mapping):
            result["recorded_detection_count"] = recorded.get(
                "detection_count"
            )
            if (
                recorded.get("detection_count")
                != first_result.get("detection_count")
            ):
                result["warnings"].append(
                    "recorded_detection_count_mismatch"
                )
            if (
                recorded.get("detections_sha256")
                and recorded.get("detections_sha256")
                != first_result.get("detections_sha256")
            ):
                result["warnings"].append(
                    "portable_result_hash_mismatch"
                )
        result.update({
            "status": (
                "passed"
                if contract_source == "archived_frozen_contract"
                else contract_source
            ),
            "technical_ok": True,
            "contract_source": contract_source,
            "postprocess_contract_sha256": verified["contract_sha256"],
            "detection_count": first_result["detection_count"],
            "detections_sha256": first_result["detections_sha256"],
            "decoder_format": first_result["decoder_format"],
        })
    except Exception as exc:
        result["status"] = "technical_decode_failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def audit_pack(
    source: Path, *, derive_unbound_raw: bool = True,
) -> dict[str, Any]:
    pack = _Pack(source)
    try:
        rows = [
            audit_manifest(
                pack, name, derive_unbound_raw=derive_unbound_raw,
            )
            for name in pack.manifests()
        ]
    finally:
        pack.close()
    detection_rows = [
        row for row in rows if row.get("status") != "skipped_non_detection"
    ]
    failures = [
        row for row in detection_rows
        if row.get("technical_ok") is not True
    ]
    return {
        "schema": "onnx-splitpoint/offline-native-detection-postprocess-audit",
        "schema_version": 1,
        "source": str(source),
        "manifest_count": len(rows),
        "detection_manifest_count": len(detection_rows),
        "technical_pass_count": len(detection_rows) - len(failures),
        "technical_failure_count": len(failures),
        "ok": bool(detection_rows) and not failures,
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Replay frozen Native-Full detection decode/NMS from an extracted "
            "debug pack or directly from its ZIP."
        )
    )
    parser.add_argument("debug_pack", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--derive-unbound-raw", action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Derive a diagnostic-only frozen contract for raw dumps that lack "
            "one. This never creates claim evidence."
        ),
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="Return non-zero when any detection manifest is not replayable.",
    )
    args = parser.parse_args()
    try:
        report = audit_pack(
            args.debug_pack,
            derive_unbound_raw=bool(args.derive_unbound_raw),
        )
    except Exception as exc:
        print(
            json.dumps({
                "ok": False,
                "status": "debug_pack_invalid",
                "error": f"{type(exc).__name__}: {exc}",
            }, indent=2),
            file=sys.stderr,
        )
        return 2
    encoded = json.dumps(
        report, indent=2, sort_keys=True, ensure_ascii=False,
    ) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    if args.strict and not report["ok"]:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
