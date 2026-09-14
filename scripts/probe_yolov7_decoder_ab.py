#!/usr/bin/env python3
"""CPU-only, fail-closed YOLOv7 legacy-vs-standard decoder probe.

The probe executes ONNX Runtime exactly once per selected image and feeds the
same raw heads to three pre-registered postprocess contracts:

* ``legacy_tiny_production``: v2.75.46 conf/NMS policy, legacy Tiny anchors;
* ``standard_production``: identical policy, Standard-YOLOv7 anchors;
* ``standard_upstream_sanity``: Standard anchors and upstream-style validation
  thresholds (conf=0.001, NMS IoU=0.65, COCO maxDets=100).

Every metric is official ``pycocotools.COCOeval``.  No local AP proxy is used
as a fallback.  The v2.75.46 internal cached-matching AP numbers are explicitly
not compared to these official COCO metrics.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import re
import secrets
import sys
import tarfile
import tempfile
import traceback
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import onnx_splitpoint_tool as tool_identity  # noqa: E402

from onnx_splitpoint_tool.dependency_bootstrap import (  # noqa: E402
    ensure_dependency_groups_for_python,
    missing_specs_for_python,
    specs_for_groups,
)
from onnx_splitpoint_tool.native_detection_postprocess import (  # noqa: E402
    FrozenDetectionPostprocessor,
    build_frozen_postprocess_contract,
    canonical_json_sha256,
)
from onnx_splitpoint_tool.quality_cache import (  # noqa: E402
    image_ids_fingerprint,
)
from onnx_splitpoint_tool.runners.harness.base import (  # noqa: E402
    postprocess_result_to_dict,
)
from onnx_splitpoint_tool.runners.harness.yolo import (  # noqa: E402
    YOLOV7_LEGACY_TINY_ANCHOR_TABLE_ID,
    YOLOV7_PAPER_MODEL_ID,
    YOLOV7_PAPER_ONNX_SHA256,
    YOLOV7_STANDARD_ANCHOR_TABLE_ID,
    YoloHarness,
    infer_yolov7_activation_mode,
    registered_yolov7_decoder_contract,
)
from onnx_splitpoint_tool.validation.official_coco import (  # noqa: E402
    evaluate_coco_bbox,
)

PROBE_SCHEMA = "onnx-splitpoint/yolov7-decoder-ab-probe"
PROBE_SCHEMA_VERSION = 1
EXPECTED_VALIDATION_MANIFEST_SHA256 = (
    "2de36f0f8949e4f1fcbd0eefbda1f4d18b411208fa91e2dd2bf985a56d0f22e2"
)
EXPECTED_SELECTED_IMAGE_IDS_SHA256 = (
    "b8ac329f5d3e7e201a3938d36c63bfe96a13c8a2be6e4fe6ddfab89cbc135e45"
)
EXPECTED_SELECTION_REQUEST_SHA256 = (
    "3d51d22511681cc28b99c2d927fe08003b92a234a4ec101683a0875d0ae72c32"
)
EXPECTED_ANNOTATIONS_SHA256 = (
    "e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f"
)
EXPECTED_MODEL_SHA256 = YOLOV7_PAPER_ONNX_SHA256
EXPECTED_SELECTED_IMAGE_COUNT = 500
EXPECTED_MANIFEST_ITEM_COUNT = 5000
MIN_AP75_IMPROVEMENT = 0.05
MAX_AP_REGRESSION = 0.01
MIN_UPSTREAM_AP_50_95 = 0.25
MIN_UPSTREAM_AP_50 = 0.40
MIN_UPSTREAM_AP_75 = 0.25
MIN_UPSTREAM_AP75_AP50_RATIO = 0.50
MIN_PRODUCTION_AP_50_95 = 0.20
MIN_PRODUCTION_AP_50 = 0.35
MIN_PRODUCTION_AP_75 = 0.20
MIN_PRODUCTION_AP75_AP50_RATIO = 0.45
OFFICIAL_COCO_MAX_DETS = (1, 10, 100)
COCO80_CATEGORY_IDS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18,
    19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37,
    38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54,
    55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74,
    75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90,
]
COCO80_CATEGORY_IDS_SHA256 = (
    "d4834f6228bfbd023e9fa80dec51cac27141fd4dee771bd799f10bdcfecb0a74"
)


class ProbeError(RuntimeError):
    pass


_OWNERSHIP_MARKER = ".yolov7_decoder_ab_probe.in_progress"


def _verified_pycocotools_version() -> str:
    try:
        version = str(importlib_metadata.version("pycocotools")).strip()
    except Exception as exc:
        raise ProbeError("pycocotools_distribution_version_unavailable") from exc
    match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)(?:[.+-].*)?", version)
    if match is None:
        raise ProbeError(f"pycocotools_distribution_version_invalid:{version}")
    if tuple(int(value) for value in match.groups()) < (2, 0, 7):
        raise ProbeError(
            f"pycocotools_distribution_version_too_old:{version}<2.0.7"
        )
    return version


def _pillow_runtime_evidence() -> dict[str, str]:
    """Record the exact implementation used by the letterbox preprocessor."""
    import PIL  # type: ignore

    try:
        version = str(importlib_metadata.version("Pillow"))
    except importlib_metadata.PackageNotFoundError as exc:
        raise ProbeError("pillow_distribution_version_unavailable") from exc
    return {
        "pillow_module": str(getattr(PIL, "__file__", "")),
        "pillow_version": version,
    }


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_exact_bytes(
    path: Path, expected_sha256: str, *, label: str,
) -> bytes:
    """Read once and bind every later use to the bytes that were hashed."""
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ProbeError(f"{label}_read_failed:{path}") from exc
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected_sha256:
        raise ProbeError(
            f"{label}_sha256_mismatch:{actual}!={expected_sha256}"
        )
    return payload


def _load_json_bytes(payload: bytes, *, label: str) -> Any:
    try:
        return json.loads(payload.decode("utf-8"))
    except Exception as exc:
        raise ProbeError(f"{label}_json_invalid:{type(exc).__name__}") from exc


def _implementation_provenance() -> dict[str, Any]:
    paths = {
        "probe": Path(__file__).resolve(),
        "generic_decoder": ROOT / "onnx_splitpoint_tool/runners/harness/yolo.py",
        "native_decoder": ROOT / "onnx_splitpoint_tool/native_detection_postprocess.py",
        "official_coco": ROOT / "onnx_splitpoint_tool/validation/official_coco.py",
    }
    files = {
        name: {
            "relative_path": str(path.relative_to(ROOT)),
            "sha256": _sha256_file(path),
            "size_bytes": int(path.stat().st_size),
        }
        for name, path in paths.items()
    }
    source_manifest = ROOT / "SOURCE_MANIFEST.json"
    source_evidence: dict[str, Any] = {
        "available": source_manifest.is_file(),
    }
    if source_manifest.is_file():
        source_evidence["sha256"] = _sha256_file(source_manifest)
        try:
            manifest = _load_json(source_manifest, label="source_manifest")
            indexed = {
                str(row.get("path") or ""): row
                for row in list(manifest.get("files") or [])
                if isinstance(row, Mapping)
            }
            membership = {}
            for name, record in files.items():
                entry = indexed.get(str(record["relative_path"]))
                membership[name] = {
                    "listed": isinstance(entry, Mapping),
                    "sha256_matches": (
                        isinstance(entry, Mapping)
                        and str(entry.get("sha256") or "")
                        == record["sha256"]
                    ),
                }
            source_evidence.update({
                "package_version": str(manifest.get("package_version") or ""),
                "workflow_version": str(manifest.get("workflow_version") or ""),
                "membership": membership,
                "all_implementation_files_listed_and_matching": all(
                    row["listed"] and row["sha256_matches"]
                    for row in membership.values()
                ),
            })
        except Exception as exc:
            source_evidence["parse_error"] = type(exc).__name__
    return {
        "tool_version": str(tool_identity.__version__),
        "tool_build_id": str(tool_identity.__build_id__),
        "files": files,
        "source_manifest": source_evidence,
    }


def _require_verified_implementation_provenance(
    provenance: Mapping[str, Any],
) -> None:
    source = provenance.get("source_manifest")
    if (
        not isinstance(source, Mapping)
        or source.get("available") is not True
        or source.get("package_version") != str(tool_identity.__version__)
        or source.get("workflow_version")
        != str(tool_identity.__build_id__)
        or source.get("all_implementation_files_listed_and_matching")
        is not True
    ):
        raise ProbeError("implementation_source_manifest_not_verified")


def _write_json(path: Path, value: Any, *, canonical: bool = False) -> None:
    payload = (
        _canonical_bytes(value)
        if canonical
        else (json.dumps(
            value, indent=2, sort_keys=True, ensure_ascii=False,
            allow_nan=False,
        ) + "\n").encode("utf-8")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = b"".join(_canonical_bytes(dict(row)) + b"\n" for row in rows)
    path.write_bytes(payload)


def _require_file(path: str | Path, *, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise ProbeError(f"{label}_missing:{resolved}")
    return resolved


def _require_sha(path: Path, expected: str, *, label: str) -> str:
    actual = _sha256_file(path)
    if actual != expected:
        raise ProbeError(f"{label}_sha256_mismatch:{actual}!={expected}")
    return actual


def _load_json(path: Path, *, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ProbeError(f"{label}_json_invalid:{type(exc).__name__}") from exc


def _dependency_preflight(*, install_missing: bool) -> dict[str, Any]:
    specs = specs_for_groups(["yolov7_probe"])
    if install_missing:
        ok, remaining = ensure_dependency_groups_for_python(
            sys.executable, ["yolov7_probe"], log=print,
        )
    else:
        remaining = missing_specs_for_python(sys.executable, specs)
        ok = not remaining
    if not ok:
        packages = sorted({spec.package for spec in remaining})
        raise ProbeError(
            "required_probe_dependencies_missing:"
            + ",".join(packages)
            + "; install with: "
            + f"{sys.executable} -m onnx_splitpoint_tool.dependency_bootstrap "
            + "--groups yolov7_probe"
        )
    import onnxruntime as ort  # type: ignore
    import pycocotools  # type: ignore
    from PIL import Image  # noqa: F401
    pycocotools_version = _verified_pycocotools_version()
    pillow_evidence = _pillow_runtime_evidence()

    return {
        "python": sys.executable,
        "onnxruntime_version": str(ort.__version__),
        "pycocotools_module": str(getattr(pycocotools, "__file__", "")),
        "pycocotools_version": pycocotools_version,
        "numpy_version": str(np.__version__),
        **pillow_evidence,
    }


def _selected_items(
    manifest: Mapping[str, Any],
    request: Mapping[str, Any],
    *,
    images_root: Path | None,
) -> tuple[list[dict[str, Any]], Path, list[str]]:
    items = manifest.get("items")
    if (
        not isinstance(items, list)
        or int(manifest.get("item_count") or 0) != EXPECTED_MANIFEST_ITEM_COUNT
        or len(items) != EXPECTED_MANIFEST_ITEM_COUNT
    ):
        raise ProbeError("validation_manifest_expected_5000_items")
    expected_ids = request.get("expected_image_ids")
    declared_ids_sha = str(
        request.get("expected_image_ids_sha256") or ""
    ).strip().lower()
    if (
        not isinstance(expected_ids, list)
        or len(expected_ids) != EXPECTED_SELECTED_IMAGE_COUNT
        or declared_ids_sha != EXPECTED_SELECTED_IMAGE_IDS_SHA256
        or image_ids_fingerprint(expected_ids)
        != EXPECTED_SELECTED_IMAGE_IDS_SHA256
    ):
        raise ProbeError("selection_request_exact_v46_500_ids_mismatch")
    ids = [str(value) for value in expected_ids]
    if len(ids) != len(set(ids)):
        raise ProbeError("selection_request_duplicate_image_ids")

    root = (
        images_root.resolve()
        if images_root is not None
        else Path(str(manifest.get("root") or "")).expanduser().resolve()
    )
    if not root.is_dir():
        raise ProbeError(f"validation_images_root_missing:{root}")
    by_token: dict[str, dict[str, Any]] = {}
    for raw in items:
        if not isinstance(raw, Mapping):
            raise ProbeError("validation_manifest_item_invalid")
        row = dict(raw)
        rel = str(row.get("relative_path") or "")
        tokens = {
            rel,
            Path(rel).name,
            str(row.get("sample_id") or ""),
            str(row.get("image_id") or ""),
        }
        for token in tokens:
            if token:
                if token in by_token and by_token[token] != row:
                    raise ProbeError(
                        f"validation_manifest_item_identity_ambiguous:{token}"
                    )
                by_token[token] = row
    selected: list[dict[str, Any]] = []
    for token in ids:
        row = by_token.get(token) or by_token.get(Path(token).name)
        if row is None:
            raise ProbeError(f"selected_image_not_in_manifest:{token}")
        path = (root / str(row.get("relative_path") or "")).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ProbeError(f"selected_image_path_escape:{path}") from exc
        if not path.is_file():
            raise ProbeError(f"selected_image_missing:{path}")
        declared = str(row.get("sha256") or "").lower().removeprefix(
            "sha256:"
        )
        if len(declared) != 64:
            raise ProbeError(f"selected_image_sha256_mismatch:{path.name}")
        image_bytes = _read_exact_bytes(
            path, declared, label=f"selected_image:{path.name}",
        )
        selected.append({
            **row,
            "absolute_path": str(path),
            "verified_image_sha256": declared,
            "image_bytes": image_bytes,
        })
    return selected, root, ids


def _validate_coco_mapping(
    annotations: Mapping[str, Any], selected_ids: Sequence[str],
) -> dict[str, Any]:
    category_ids = sorted(
        int(row["id"])
        for row in list(annotations.get("categories") or [])
        if isinstance(row, Mapping) and row.get("id") is not None
    )
    if category_ids != COCO80_CATEGORY_IDS:
        raise ProbeError("annotations_coco80_category_mapping_mismatch")
    if hashlib.sha256(_canonical_bytes(category_ids)).hexdigest() != (
        COCO80_CATEGORY_IDS_SHA256
    ):
        raise ProbeError("coco80_category_mapping_hash_mismatch")
    images = {
        str(row.get("file_name") or ""): int(row.get("id"))
        for row in list(annotations.get("images") or [])
        if isinstance(row, Mapping)
    }
    selected_map: dict[str, int] = {}
    for token in selected_ids:
        name = Path(token).name
        if name not in images:
            raise ProbeError(f"selected_image_absent_from_annotations:{name}")
        selected_map[name] = images[name]
    if len(set(selected_map.values())) != EXPECTED_SELECTED_IMAGE_COUNT:
        raise ProbeError("selected_annotation_image_ids_not_unique")
    return {
        "class_index_to_category_id": category_ids,
        "mapping_sha256": COCO80_CATEGORY_IDS_SHA256,
        "selected_file_to_image_id": selected_map,
    }


def _preprocess(image_bytes: bytes) -> tuple[np.ndarray, tuple[int, int]]:
    from PIL import Image

    with Image.open(io.BytesIO(image_bytes)) as source:
        image = source.convert("RGB")
        original_wh = (int(image.size[0]), int(image.size[1]))
        gain = min(640.0 / original_wh[0], 640.0 / original_wh[1])
        resized_wh = (
            int(round(original_wh[0] * gain)),
            int(round(original_wh[1] * gain)),
        )
        resized = image.resize(resized_wh, resample=Image.BILINEAR)
        canvas = Image.new("RGB", (640, 640), color=(114, 114, 114))
        canvas.paste(
            resized,
            ((640 - resized_wh[0]) // 2, (640 - resized_wh[1]) // 2),
        )
        array = np.asarray(canvas, dtype=np.float32) / 255.0
    return np.ascontiguousarray(array.transpose(2, 0, 1)[None, ...]), original_wh


def _payload(
    harness: YoloHarness, outputs: Mapping[str, np.ndarray],
    *, original_wh: Sequence[int],
) -> dict[str, Any]:
    result = harness.postprocess(
        dict(outputs),
        context={
            "input_hw": [640, 640],
            "original_wh": list(original_wh),
            "model_id": YOLOV7_PAPER_MODEL_ID,
            "variant": "decoder_ab_probe",
        },
    )
    normalized = postprocess_result_to_dict(result)
    payload = normalized.get("json")
    if not isinstance(payload, Mapping):
        raise ProbeError("postprocess_payload_missing")
    return dict(payload)


def _canonical_detection_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for raw in list(payload.get("detections") or []):
        if not isinstance(raw, Mapping):
            raise ProbeError("postprocess_detection_record_invalid")
        class_id = int(raw.get("class_id"))
        if class_id < 0 or class_id >= len(COCO80_CATEGORY_IDS):
            raise ProbeError(f"postprocess_class_id_out_of_range:{class_id}")
        rows.append({
            "class_id": class_id,
            "score": float(raw.get("score")),
            "x1": float(raw.get("x1")),
            "y1": float(raw.get("y1")),
            "x2": float(raw.get("x2")),
            "y2": float(raw.get("y2")),
        })
    return sorted(
        rows,
        key=lambda row: (
            -row["score"], row["class_id"], row["x1"], row["y1"],
            row["x2"], row["y2"],
        ),
    )


def _to_coco(
    detections: Sequence[Mapping[str, Any]], *, image_id: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in detections:
        class_id = int(raw["class_id"])
        if class_id < 0 or class_id >= len(COCO80_CATEGORY_IDS):
            raise ProbeError(f"postprocess_class_id_out_of_range:{class_id}")
        x1, y1 = float(raw["x1"]), float(raw["y1"])
        x2, y2 = float(raw["x2"]), float(raw["y2"])
        width, height = x2 - x1, y2 - y1
        if width <= 0.0 or height <= 0.0:
            continue
        rows.append({
            "image_id": int(image_id),
            "category_id": int(COCO80_CATEGORY_IDS[class_id]),
            "bbox": [x1, y1, width, height],
            "score": float(raw["score"]),
        })
    return rows


def _sort_coco(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        (dict(row) for row in rows),
        key=lambda row: (
            int(row["image_id"]), -float(row["score"]),
            int(row["category_id"]), *(float(x) for x in row["bbox"]),
        ),
    )


def _contract_pairing_identity(contract: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: contract.get(key)
        for key in (
            "model_id", "model_sha256", "decoder_id",
            "postprocess_policy_id", "output_format", "input_hw",
            "head_to_stride_policy", "raw_record_semantics",
            "activation_mode", "decode_equations_id",
            "objectness_class_combination", "preprocess_identity",
            "nms_identity",
        )
    }


def _contract_diff_keys(
    left: Mapping[str, Any], right: Mapping[str, Any],
) -> list[str]:
    return sorted(
        key
        for key in set(left) | set(right)
        if left.get(key) != right.get(key)
    )


def _upstream_health_gates(metrics: Mapping[str, Any]) -> dict[str, bool]:
    ap = float(metrics["AP_50_95"])
    ap50 = float(metrics["AP_50"])
    ap75 = float(metrics["AP_75"])
    ratio = ap75 / ap50 if ap50 > 0.0 else 0.0
    return {
        "upstream_sanity_ap_50_95_at_least_025": (
            ap >= MIN_UPSTREAM_AP_50_95
        ),
        "upstream_sanity_ap50_at_least_040": ap50 >= MIN_UPSTREAM_AP_50,
        "upstream_sanity_ap75_at_least_025": ap75 >= MIN_UPSTREAM_AP_75,
        "upstream_sanity_ap75_ap50_ratio_at_least_050": (
            ratio >= MIN_UPSTREAM_AP75_AP50_RATIO
        ),
    }


def _production_health_gates(metrics: Mapping[str, Any]) -> dict[str, bool]:
    """Pre-registered catastrophic-corruption floors for the fixed B arm.

    These are deliberately weaker than the independent upstream-sanity floors.
    They establish completed-task decoder health under the retained production
    thresholds; they are not a canonical-accuracy claim.
    """
    ap = float(metrics["AP_50_95"])
    ap50 = float(metrics["AP_50"])
    ap75 = float(metrics["AP_75"])
    ratio = ap75 / ap50 if ap50 > 0.0 else 0.0
    return {
        "standard_production_ap_50_95_at_least_020": (
            ap >= MIN_PRODUCTION_AP_50_95
        ),
        "standard_production_ap50_at_least_035": (
            ap50 >= MIN_PRODUCTION_AP_50
        ),
        "standard_production_ap75_at_least_020": (
            ap75 >= MIN_PRODUCTION_AP_75
        ),
        "standard_production_ap75_ap50_ratio_at_least_045": (
            ratio >= MIN_PRODUCTION_AP75_AP50_RATIO
        ),
    }


def _overlay(
    image_bytes: bytes, output_path: Path,
    legacy: Sequence[Mapping[str, Any]], standard: Sequence[Mapping[str, Any]],
) -> None:
    from PIL import Image, ImageDraw

    with Image.open(io.BytesIO(image_bytes)) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    for row in list(legacy)[:100]:
        draw.rectangle(
            [row["x1"], row["y1"], row["x2"], row["y2"]],
            outline=(255, 64, 64), width=2,
        )
    for row in list(standard)[:100]:
        draw.rectangle(
            [row["x1"], row["y1"], row["x2"], row["y2"]],
            outline=(64, 255, 64), width=2,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def _sha256sums(root: Path) -> None:
    rows = []
    for path in sorted(root.rglob("*")):
        if (
            path.is_file()
            and path.name not in {"SHA256SUMS.txt", _OWNERSHIP_MARKER}
        ):
            rows.append(f"{_sha256_file(path)}  {path.relative_to(root).as_posix()}")
    (root / "SHA256SUMS.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")


def _deterministic_archive(root: Path, destination: Path) -> Path:
    if destination == root or root in destination.parents:
        raise ProbeError("archive_path_must_be_outside_output_dir")
    if destination.exists():
        raise ProbeError(f"archive_path_already_exists:{destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.tmp-{os.getpid()}-{secrets.token_hex(8)}"
    )
    try:
        with temporary.open("xb") as raw:
            with gzip.GzipFile(
                filename="", mode="wb", fileobj=raw, mtime=0,
            ) as compressed:
                with tarfile.open(
                    fileobj=compressed, mode="w", format=tarfile.USTAR_FORMAT,
                ) as archive:
                    for path in sorted(root.rglob("*")):
                        if not path.is_file() or path.name == _OWNERSHIP_MARKER:
                            continue
                        data = path.read_bytes()
                        info = tarfile.TarInfo(
                            "yolov7_decoder_ab_probe/"
                            + path.relative_to(root).as_posix()
                        )
                        info.size = len(data)
                        info.mtime = 0
                        info.uid = 0
                        info.gid = 0
                        info.uname = "root"
                        info.gname = "root"
                        info.mode = 0o644
                        archive.addfile(info, io.BytesIO(data))
        try:
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise ProbeError(
                f"archive_path_already_exists:{destination}"
            ) from exc
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _archive_path(args: argparse.Namespace, output_dir: Path) -> Path:
    explicit = str(getattr(args, "archive_path", "") or "").strip()
    return (
        Path(explicit).expanduser().resolve()
        if explicit else Path(str(output_dir) + ".tar.gz")
    )


def _archive_path_is_unsafe(output_dir: Path, archive_path: Path) -> bool:
    return archive_path == output_dir or output_dir in archive_path.parents


def _claim_output_dir(output_dir: Path, token: str) -> Path:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        output_dir.mkdir(exist_ok=False)
    except FileExistsError:
        if not output_dir.is_dir() or any(output_dir.iterdir()):
            raise ProbeError(f"output_dir_not_empty:{output_dir}")
    marker = output_dir / _OWNERSHIP_MARKER
    try:
        with marker.open("x", encoding="utf-8") as stream:
            stream.write(str(token) + "\n")
    except FileExistsError as exc:
        raise ProbeError(f"output_dir_ownership_conflict:{output_dir}") from exc
    foreign = [path for path in output_dir.iterdir() if path != marker]
    if foreign:
        marker.unlink(missing_ok=True)
        raise ProbeError(f"output_dir_not_empty:{output_dir}")
    return marker


def _owned_output_marker(output_dir: Path, token: str) -> Path | None:
    marker = output_dir / _OWNERSHIP_MARKER
    try:
        if marker.is_file() and marker.read_text(encoding="utf-8") == (
            str(token) + "\n"
        ):
            return marker
    except Exception:
        pass
    return None


def _archive_manifest_path(archive_path: Path) -> Path:
    return Path(str(archive_path) + ".manifest.json")


def _write_archive_manifest(
    output_dir: Path,
    archive_path: Path,
    *,
    status: str,
    acceptance: str | None,
) -> Path:
    summary_path = output_dir / "probe_summary.json"
    with tarfile.open(archive_path, mode="r:gz") as archive:
        archive_member_count = sum(
            1 for member in archive.getmembers() if member.isfile()
        )
    payload = {
        "schema": "onnx-splitpoint/yolov7-decoder-ab-probe-archive",
        "schema_version": 1,
        "probe_status": str(status),
        "acceptance": acceptance,
        "archive_basename": archive_path.name,
        "archive_sha256": _sha256_file(archive_path),
        "archive_size_bytes": int(archive_path.stat().st_size),
        "archive_member_count": archive_member_count,
        "probe_summary_sha256": (
            _sha256_file(summary_path) if summary_path.is_file() else None
        ),
    }
    destination = _archive_manifest_path(archive_path)
    if destination.exists():
        raise ProbeError(f"archive_manifest_already_exists:{destination}")
    temporary = destination.with_name(
        destination.name
        + f".tmp-{os.getpid()}-{secrets.token_hex(8)}"
    )
    try:
        _write_json(temporary, payload, canonical=True)
        try:
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise ProbeError(
                f"archive_manifest_already_exists:{destination}"
            ) from exc
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def _preflight(args: argparse.Namespace) -> dict[str, Any]:
    dependencies = _dependency_preflight(
        install_missing=bool(args.install_missing),
    )
    implementation = _implementation_provenance()
    _require_verified_implementation_provenance(implementation)
    model = _require_file(args.model, label="model")
    manifest_path = _require_file(
        args.validation_manifest, label="validation_manifest",
    )
    request_path = _require_file(
        args.selection_request, label="selection_request",
    )
    annotations_path = _require_file(args.annotations, label="annotations")
    model_bytes = _read_exact_bytes(
        model, EXPECTED_MODEL_SHA256, label="model",
    )
    manifest_bytes = _read_exact_bytes(
        manifest_path, EXPECTED_VALIDATION_MANIFEST_SHA256,
        label="validation_manifest",
    )
    request_bytes = _read_exact_bytes(
        request_path, EXPECTED_SELECTION_REQUEST_SHA256,
        label="selection_request",
    )
    annotations_bytes = _read_exact_bytes(
        annotations_path, EXPECTED_ANNOTATIONS_SHA256,
        label="annotations",
    )
    manifest = _load_json_bytes(
        manifest_bytes, label="validation_manifest",
    )
    request = _load_json_bytes(request_bytes, label="selection_request")
    annotations = _load_json_bytes(annotations_bytes, label="annotations")
    if not all(isinstance(value, Mapping) for value in (manifest, request, annotations)):
        raise ProbeError("probe_input_json_object_required")
    selected, images_root, selected_ids = _selected_items(
        manifest, request,
        images_root=(
            Path(args.images_root).expanduser()
            if str(args.images_root or "").strip() else None
        ),
    )
    category_mapping = _validate_coco_mapping(annotations, selected_ids)
    return {
        "dependencies": dependencies,
        "implementation_provenance": implementation,
        "model": model,
        "model_bytes": model_bytes,
        "manifest_path": manifest_path,
        "request_path": request_path,
        "annotations_path": annotations_path,
        "manifest": manifest,
        "manifest_bytes": manifest_bytes,
        "request": request,
        "request_bytes": request_bytes,
        "annotations": annotations,
        "annotations_bytes": annotations_bytes,
        "selected": selected,
        "selected_ids": selected_ids,
        "images_root": images_root,
        "category_mapping": category_mapping,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    preflight = _preflight(args)
    if args.preflight_only:
        return {
            "schema": PROBE_SCHEMA,
            "schema_version": PROBE_SCHEMA_VERSION,
            "status": "preflight_ok",
            "model_sha256": EXPECTED_MODEL_SHA256,
            "validation_manifest_sha256": EXPECTED_VALIDATION_MANIFEST_SHA256,
            "selection_request_sha256": EXPECTED_SELECTION_REQUEST_SHA256,
            "selected_image_ids_sha256": EXPECTED_SELECTED_IMAGE_IDS_SHA256,
            "selected_image_count": EXPECTED_SELECTED_IMAGE_COUNT,
            "annotations_sha256": EXPECTED_ANNOTATIONS_SHA256,
            "coco80_category_ids_sha256": COCO80_CATEGORY_IDS_SHA256,
            "dependencies": preflight["dependencies"],
            "implementation_provenance": preflight[
                "implementation_provenance"
            ],
        }

    output_dir = Path(args.output_dir).expanduser().resolve()
    archive_path = _archive_path(args, output_dir)
    if _archive_path_is_unsafe(output_dir, archive_path):
        raise ProbeError("archive_path_must_be_outside_output_dir")
    if archive_path.exists():
        raise ProbeError(f"archive_path_already_exists:{archive_path}")
    if _archive_manifest_path(archive_path).exists():
        raise ProbeError(
            "archive_manifest_already_exists:"
            f"{_archive_manifest_path(archive_path)}"
        )
    ownership_token = str(
        getattr(args, "_ownership_token", "") or ""
    )
    if not ownership_token:
        raise ProbeError("probe_output_ownership_token_missing")
    ownership_marker = _claim_output_dir(output_dir, ownership_token)

    import onnxruntime as ort  # type: ignore

    session = ort.InferenceSession(
        preflight["model_bytes"], providers=["CPUExecutionProvider"],
    )
    inputs = session.get_inputs()
    if len(inputs) != 1:
        raise ProbeError("model_exactly_one_input_required")
    input_meta = inputs[0]
    if list(input_meta.shape) != [1, 3, 640, 640]:
        raise ProbeError(f"model_input_shape_mismatch:{input_meta.shape}")
    output_meta = session.get_outputs()
    if len(output_meta) != 3:
        raise ProbeError("model_exactly_three_raw_heads_required")

    contracts: dict[str, dict[str, Any]] = {}
    harnesses: dict[str, YoloHarness] = {}
    predictions = {
        "legacy_tiny_production": [],
        "standard_production": [],
        "standard_upstream_sanity": [],
    }
    activation_mode = ""
    parity_failures: list[dict[str, Any]] = []
    parity_rows: list[dict[str, Any]] = []
    generic_parity_chain = hashlib.sha256()
    native_parity_chain = hashlib.sha256()
    raw_inference_count = 0
    raw_content_chain = hashlib.sha256()
    overlays_remaining = max(0, min(int(args.overlay_count), 20))

    for index, item in enumerate(preflight["selected"], start=1):
        path = Path(str(item["absolute_path"]))
        image_bytes = item["image_bytes"]
        if not isinstance(image_bytes, bytes):
            raise ProbeError(f"selected_image_snapshot_invalid:{path.name}")
        tensor, original_wh = _preprocess(image_bytes)
        raw_values = session.run(None, {str(input_meta.name): tensor})
        raw_inference_count += 1
        if len(raw_values) != 3:
            raise ProbeError("runtime_exactly_three_raw_heads_required")
        outputs = {
            str(meta.name): np.asarray(value)
            for meta, value in zip(output_meta, raw_values)
        }
        raw_content_chain.update(_canonical_bytes({
            "image": path.name,
            "outputs": [
                {
                    "name": name,
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "sha256": hashlib.sha256(
                        np.ascontiguousarray(value).tobytes()
                    ).hexdigest(),
                }
                for name, value in outputs.items()
            ],
        }))
        observed_activation = infer_yolov7_activation_mode(outputs)
        if not contracts:
            activation_mode = observed_activation
            contracts = {
                "legacy_tiny_production": registered_yolov7_decoder_contract(
                    model_id=YOLOV7_PAPER_MODEL_ID,
                    model_sha256=EXPECTED_MODEL_SHA256,
                    activation_mode=activation_mode,
                    variant="legacy_tiny",
                    policy="production",
                ),
                "standard_production": registered_yolov7_decoder_contract(
                    model_id=YOLOV7_PAPER_MODEL_ID,
                    model_sha256=EXPECTED_MODEL_SHA256,
                    activation_mode=activation_mode,
                    variant="standard",
                    policy="production",
                ),
                "standard_upstream_sanity": registered_yolov7_decoder_contract(
                    model_id=YOLOV7_PAPER_MODEL_ID,
                    model_sha256=EXPECTED_MODEL_SHA256,
                    activation_mode=activation_mode,
                    variant="standard",
                    policy="upstream_sanity",
                ),
            }
            for name, contract in contracts.items():
                nms = contract["nms_identity"]
                harnesses[name] = YoloHarness(
                    conf_thresh=float(nms["confidence_threshold"]),
                    iou_thresh=float(nms["iou_threshold"]),
                    max_det=int(nms["max_detections"]),
                    model_id=YOLOV7_PAPER_MODEL_ID,
                    multiscale_decoder_contract=contract,
                    allow_diagnostic_yolov7_contract=(
                        contract["use_scope"] != "production"
                    ),
                )
        elif observed_activation != activation_mode:
            raise ProbeError(
                "activation_mode_drift_across_selected_images:"
                f"{observed_activation}!={activation_mode}"
            )

        per_policy: dict[str, list[dict[str, Any]]] = {}
        image_id = int(
            preflight["category_mapping"]["selected_file_to_image_id"][
                path.name
            ]
        )
        for name, harness in harnesses.items():
            payload = _payload(harness, outputs, original_wh=original_wh)
            detections = _canonical_detection_rows(payload)
            per_policy[name] = detections
            predictions[name].extend(_to_coco(detections, image_id=image_id))

        frozen_contract = build_frozen_postprocess_contract(
            model_id=YOLOV7_PAPER_MODEL_ID,
            model_sha256=EXPECTED_MODEL_SHA256,
            outputs=outputs,
            input_hw=[640, 640],
            original_wh=original_wh,
        )
        native_rows = _canonical_detection_rows({
            "detections": FrozenDetectionPostprocessor(
                frozen_contract
            ).process(
                outputs, original_wh=original_wh,
            )["detections"],
        })
        generic_sha256 = canonical_json_sha256(
            per_policy["standard_production"]
        )
        native_sha256 = canonical_json_sha256(native_rows)
        parity_equal = (
            list(native_rows) == per_policy["standard_production"]
            and generic_sha256 == native_sha256
        )
        parity_row = {
            "sequence_index": index,
            "image": path.name,
            "image_sha256": item["verified_image_sha256"],
            "coco_image_id": image_id,
            "generic_sha256": generic_sha256,
            "native_sha256": native_sha256,
            "equal": parity_equal,
        }
        parity_rows.append(parity_row)
        generic_parity_chain.update(_canonical_bytes({
            "sequence_index": index,
            "image": path.name,
            "image_sha256": item["verified_image_sha256"],
            "detections_sha256": generic_sha256,
        }))
        native_parity_chain.update(_canonical_bytes({
            "sequence_index": index,
            "image": path.name,
            "image_sha256": item["verified_image_sha256"],
            "detections_sha256": native_sha256,
        }))
        if not parity_equal:
            parity_failures.append(parity_row)
        if overlays_remaining > 0:
            _overlay(
                image_bytes,
                output_dir / "overlays" / f"{path.stem}_legacy_red_standard_green.png",
                per_policy["legacy_tiny_production"],
                per_policy["standard_production"],
            )
            overlays_remaining -= 1
        if index % 25 == 0 or index == EXPECTED_SELECTED_IMAGE_COUNT:
            print(f"[yolov7-decoder-ab] {index}/{EXPECTED_SELECTED_IMAGE_COUNT}")

    if raw_inference_count != EXPECTED_SELECTED_IMAGE_COUNT:
        raise ProbeError("raw_onnx_inference_count_mismatch")
    _write_jsonl(output_dir / "generic_native_parity.jsonl", parity_rows)
    if parity_failures:
        raise ProbeError(
            f"generic_native_standard_parity_failed:{len(parity_failures)}"
        )

    for name in predictions:
        predictions[name] = _sort_coco(predictions[name])
        _write_json(
            output_dir / f"predictions_{name}.json",
            predictions[name], canonical=True,
        )

    official: dict[str, dict[str, Any]] = {}
    selected_coco_ids = sorted(
        preflight["category_mapping"]["selected_file_to_image_id"].values()
    )
    with tempfile.TemporaryDirectory(
        prefix="onnx_splitpoint_yolov7_annotations_",
    ) as snapshot_root:
        annotations_snapshot = Path(snapshot_root) / "instances_val2017.json"
        annotations_snapshot.write_bytes(preflight["annotations_bytes"])
        if _sha256_file(annotations_snapshot) != EXPECTED_ANNOTATIONS_SHA256:
            raise ProbeError("annotations_exact_use_snapshot_mismatch")
        for name, rows in predictions.items():
            official[name] = evaluate_coco_bbox(
                annotations=annotations_snapshot,
                predictions=rows,
                output_dir=output_dir / "official_coco" / name,
                image_ids=selected_coco_ids,
                category_ids=COCO80_CATEGORY_IDS,
                # COCOeval summarize() is defined for maxDets=100.  The paired
                # production decoder still retains max_det=300 before this
                # independent official-evaluation cap is applied.
                max_detections=OFFICIAL_COCO_MAX_DETS,
                archive_tensors=True,
                required=True,
                variant=name,
                annotations_provenance={
                    "kind": "preflight_hash_verified_immutable_byte_snapshot",
                    "source_path": str(preflight["annotations_path"]),
                    "source_sha256": EXPECTED_ANNOTATIONS_SHA256,
                    "exact_use_snapshot_sha256": (
                        EXPECTED_ANNOTATIONS_SHA256
                    ),
                    "snapshot_embedded_in_probe_archive": False,
                    "non_embedding_reason": (
                        "large_ground_truth_is_hash_bound_not_duplicated"
                    ),
                },
            )
            if official[name].get("status") != "ok":
                raise ProbeError(f"official_coco_not_completed:{name}")
            if official[name].get("annotations_sha256") != (
                EXPECTED_ANNOTATIONS_SHA256
            ):
                raise ProbeError(
                    f"official_coco_annotations_snapshot_mismatch:{name}"
                )
            exact_use = official[name].get("annotations_exact_use")
            if (
                not isinstance(exact_use, Mapping)
                or exact_use.get("exact_use_snapshot_sha256")
                != EXPECTED_ANNOTATIONS_SHA256
            ):
                raise ProbeError(
                    f"official_coco_annotations_provenance_mismatch:{name}"
                )

    a_metrics = official["legacy_tiny_production"]["metrics"]
    b_metrics = official["standard_production"]["metrics"]
    upstream_metrics = official["standard_upstream_sanity"]["metrics"]
    contract_diff_keys = _contract_diff_keys(
        contracts["legacy_tiny_production"],
        contracts["standard_production"],
    )
    allowed_contract_diff_keys = [
        "anchor_table_id", "anchors_by_stride",
        "decoder_contract_sha256", "use_scope",
    ]
    gates = {
        "exact_500_id_cohort": True,
        "one_raw_ort_pass_per_image": raw_inference_count
        == EXPECTED_SELECTED_IMAGE_COUNT,
        "generic_native_standard_exact_parity": (
            len(parity_rows) == EXPECTED_SELECTED_IMAGE_COUNT
            and not parity_failures
            and generic_parity_chain.hexdigest()
            == native_parity_chain.hexdigest()
        ),
        "official_coco_all_policies_completed": all(
            row.get("status") == "ok" for row in official.values()
        ),
        "paired_production_policy_equal_except_anchor_table": (
            _contract_pairing_identity(
                contracts["legacy_tiny_production"]
            ) == _contract_pairing_identity(
                contracts["standard_production"]
            )
        ),
        "paired_contract_diff_exactly_pre_registered": (
            contract_diff_keys == allowed_contract_diff_keys
        ),
        "standard_ap75_material_improvement": (
            float(b_metrics["AP_75"]) - float(a_metrics["AP_75"])
            >= MIN_AP75_IMPROVEMENT
        ),
        "standard_ap_no_regression_beyond_001": (
            float(b_metrics["AP_50_95"])
            >= float(a_metrics["AP_50_95"]) - MAX_AP_REGRESSION
        ),
        "standard_ap50_no_regression_beyond_001": (
            float(b_metrics["AP_50"])
            >= float(a_metrics["AP_50"]) - MAX_AP_REGRESSION
        ),
        **_production_health_gates(b_metrics),
        **_upstream_health_gates(upstream_metrics),
    }
    passed = all(bool(value) for value in gates.values())
    summary = {
        "schema": PROBE_SCHEMA,
        "schema_version": PROBE_SCHEMA_VERSION,
        "status": "completed" if passed else "failed",
        "acceptance": {
            "status": "accepted" if passed else "rejected",
            "passed": passed,
        },
        "model_id": YOLOV7_PAPER_MODEL_ID,
        "model_sha256": EXPECTED_MODEL_SHA256,
        "validation_manifest_sha256": EXPECTED_VALIDATION_MANIFEST_SHA256,
        "selection_request_sha256": EXPECTED_SELECTION_REQUEST_SHA256,
        "selected_image_ids_sha256": EXPECTED_SELECTED_IMAGE_IDS_SHA256,
        "selected_image_count": EXPECTED_SELECTED_IMAGE_COUNT,
        "annotations_sha256": EXPECTED_ANNOTATIONS_SHA256,
        "raw_onnx_inference_count": raw_inference_count,
        "raw_output_content_chain_sha256": raw_content_chain.hexdigest(),
        "generic_native_parity": {
            "artifact": "generic_native_parity.jsonl",
            "row_count": len(parity_rows),
            "equal_count": sum(
                1 for row in parity_rows if row["equal"] is True
            ),
            "all_equal": not parity_failures,
            "generic_canonical_chain_sha256": (
                generic_parity_chain.hexdigest()
            ),
            "native_canonical_chain_sha256": (
                native_parity_chain.hexdigest()
            ),
            "chains_equal": generic_parity_chain.hexdigest()
            == native_parity_chain.hexdigest(),
        },
        "activation_mode": activation_mode,
        "coco80_category_ids": COCO80_CATEGORY_IDS,
        "coco80_category_ids_sha256": COCO80_CATEGORY_IDS_SHA256,
        "contracts": contracts,
        "contract_sha256s": {
            name: contract["decoder_contract_sha256"]
            for name, contract in contracts.items()
        },
        "policies": {
            name: {
                "anchor_table_id": contract["anchor_table_id"],
                "postprocess_policy_id": contract["postprocess_policy_id"],
                "nms_identity": contract["nms_identity"],
                "use_scope": contract["use_scope"],
            }
            for name, contract in contracts.items()
        },
        "official_coco": {
            name: {
                "status": row["status"],
                "metrics": row["metrics"],
                "evaluation_max_dets": list(OFFICIAL_COCO_MAX_DETS),
                "evaluation_payload_sha256": row[
                    "evaluation_payload_sha256"
                ],
            }
            for name, row in official.items()
        },
        "paired_official_deltas_standard_minus_legacy": {
            key: float(b_metrics[key]) - float(a_metrics[key])
            for key in ("AP_50_95", "AP_50", "AP_75")
        },
        "contract_diff_keys": contract_diff_keys,
        "allowed_contract_diff_keys": allowed_contract_diff_keys,
        "pre_registered_gate": {
            "min_ap75_improvement": MIN_AP75_IMPROVEMENT,
            "max_ap_or_ap50_regression": MAX_AP_REGRESSION,
            "standard_production_completed_task_health_floors": {
                "AP_50_95": MIN_PRODUCTION_AP_50_95,
                "AP_50": MIN_PRODUCTION_AP_50,
                "AP_75": MIN_PRODUCTION_AP_75,
                "AP75_over_AP50": MIN_PRODUCTION_AP75_AP50_RATIO,
                "interpretation": (
                    "completed-task decoder health/no-catastrophic-corruption "
                    "only; not a canonical-accuracy claim"
                ),
            },
            "upstream_sanity_catastrophic_corruption_floors": {
                "AP_50_95": MIN_UPSTREAM_AP_50_95,
                "AP_50": MIN_UPSTREAM_AP_50,
                "AP_75": MIN_UPSTREAM_AP_75,
                "AP75_over_AP50": MIN_UPSTREAM_AP75_AP50_RATIO,
                "interpretation": (
                    "decoder health/no-catastrophic-corruption only; "
                    "not a canonical-accuracy claim"
                ),
            },
            "gates": gates,
            "passed": passed,
        },
        "v27546_internal_metric_reproduction": {
            "status": "not_attempted",
            "reason": (
                "v2.75.46 values used internal cached-matching-v2-ap75; "
                "official pycocotools A/B values are not numerically compared"
            ),
            "legacy_path_reproduction_scope": (
                "same pinned model/cohort/preprocess/conf/NMS/legacy anchors"
            ),
        },
        "canonical_accuracy_claim": False,
        "canonical_accuracy_claim_reason": (
            "paired decoder isolation only; no independent upstream prediction "
            "reference was supplied"
        ),
        "generic_native_parity_failures": parity_failures,
        "dependencies": preflight["dependencies"],
        "implementation_provenance": preflight[
            "implementation_provenance"
        ],
    }
    _write_json(output_dir / "probe_summary.json", summary)
    _write_json(output_dir / "decoder_contracts.json", contracts)
    _write_json(
        output_dir / "input_provenance.json",
        {
            "model": str(preflight["model"]),
            "validation_manifest": str(preflight["manifest_path"]),
            "selection_request": str(preflight["request_path"]),
            "selection_request_sha256": EXPECTED_SELECTION_REQUEST_SHA256,
            "annotations": str(preflight["annotations_path"]),
            "images_root": str(preflight["images_root"]),
            "selected_image_ids": preflight["selected_ids"],
            "selected_image_ids_sha256": EXPECTED_SELECTED_IMAGE_IDS_SHA256,
            "category_mapping": preflight["category_mapping"],
            "exact_use_snapshots": {
                "model_sha256": hashlib.sha256(
                    preflight["model_bytes"]
                ).hexdigest(),
                "validation_manifest_sha256": hashlib.sha256(
                    preflight["manifest_bytes"]
                ).hexdigest(),
                "selection_request_sha256": hashlib.sha256(
                    preflight["request_bytes"]
                ).hexdigest(),
                "annotations_sha256": hashlib.sha256(
                    preflight["annotations_bytes"]
                ).hexdigest(),
                "selected_image_count": len(preflight["selected"]),
                "selected_image_bytes_sha256_verified": all(
                    hashlib.sha256(row["image_bytes"]).hexdigest()
                    == row["verified_image_sha256"]
                    for row in preflight["selected"]
                ),
            },
            "implementation_provenance": preflight[
                "implementation_provenance"
            ],
        },
    )
    if not passed:
        raise ProbeError("pre_registered_yolov7_decoder_ab_gate_failed")
    _sha256sums(output_dir)
    _deterministic_archive(output_dir, archive_path)
    try:
        archive_manifest = _write_archive_manifest(
            output_dir, archive_path,
            status="completed", acceptance="accepted",
        )
    except Exception:
        archive_path.unlink(missing_ok=True)
        raise
    ownership_marker.unlink()
    summary["evidence_archive"] = str(archive_path)
    summary["evidence_archive_sha256"] = _sha256_file(archive_path)
    summary["evidence_archive_manifest"] = str(archive_manifest)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "CPU-only pinned YOLOv7 decoder A/B with mandatory official COCO"
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--validation-manifest", required=True)
    parser.add_argument("--selection-request", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--images-root", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--archive-path", default="",
        help="Deterministic evidence tar.gz (default: OUTPUT_DIR.tar.gz).",
    )
    parser.add_argument("--overlay-count", type=int, default=0)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--install-missing", action="store_true",
        help="Install the pinned yolov7_probe dependency group into this interpreter.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    ownership_token = secrets.token_hex(32)
    setattr(args, "_ownership_token", ownership_token)
    output_text = str(getattr(args, "output_dir", "") or "").strip()
    output = (
        Path(output_text).expanduser().resolve() if output_text else None
    )
    archive_unsafe = False
    if output is not None:
        try:
            archive_unsafe = _archive_path_is_unsafe(
                output, _archive_path(args, output),
            )
        except Exception:
            archive_unsafe = True
    try:
        result = run(args)
        print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
        return 0
    except Exception as exc:
        failure = {
            "schema": PROBE_SCHEMA,
            "schema_version": PROBE_SCHEMA_VERSION,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
        }
        if (
            output is not None
            and not archive_unsafe
            and _owned_output_marker(output, ownership_token) is not None
        ):
            try:
                _write_json(output / "probe_failure.json", failure)
                (output / "probe_traceback.txt").write_text(
                    traceback.format_exc(), encoding="utf-8",
                )
                owned_marker = _owned_output_marker(
                    output, ownership_token,
                )
                if owned_marker is None:
                    raise ProbeError("probe_output_ownership_lost")
                _sha256sums(output)
                archive = _archive_path(args, output)
                archive_manifest = _archive_manifest_path(archive)
                if not archive.exists() and not archive_manifest.exists():
                    _deterministic_archive(output, archive)
                    sidecar_acceptance = None
                    summary_path = output / "probe_summary.json"
                    if summary_path.is_file():
                        try:
                            summary_payload = _load_json(
                                summary_path, label="probe_summary",
                            )
                            acceptance_payload = (
                                summary_payload.get("acceptance")
                                if isinstance(summary_payload, Mapping)
                                else None
                            )
                            if (
                                isinstance(acceptance_payload, Mapping)
                                and acceptance_payload.get("status")
                                == "rejected"
                            ):
                                sidecar_acceptance = "rejected"
                        except Exception:
                            sidecar_acceptance = None
                    try:
                        _write_archive_manifest(
                            output, archive,
                            status="failed", acceptance=sidecar_acceptance,
                        )
                    except Exception:
                        archive.unlink(missing_ok=True)
                        raise
                    owned_marker.unlink()
            except Exception:
                pass
        print(json.dumps(failure, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
