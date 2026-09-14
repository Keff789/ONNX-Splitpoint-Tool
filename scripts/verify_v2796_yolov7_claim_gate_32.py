#!/usr/bin/env python3
"""Fail-closed verifier for the v2.79.6 YOLOv7 ``claim_gate_32``.

The verifier never executes inference and never changes captured evidence.  It
rehashes the complete release source manifest and the retained hardware
evidence, validates the immutable Three-Stage invocation, and writes only the
explicit output receipt requested by ``--output``.  Large runtime artifacts
(HEF and TensorRT engine) are represented by identities, not embedded bytes;
their independently captured identities must agree with both the frozen
quality binding and the native canary report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import sys
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


VERDICT_SCHEMA = "onnx-splitpoint/yolov7-claim-gate-32/v1"
CAPTURE_SCHEMA = "onnx-splitpoint/yolov7-claim-gate-32-capture/v1"
RESULT_SCHEMA = "onnx-splitpoint/native-dual-endpoint-result"
RESULT_SCHEMA_VERSION = 2
INVOCATION_SCHEMA = "onnx-splitpoint/three-stage-invocation"
ARTIFACT_INDEX_SCHEMA = "onnx-splitpoint/three-stage-artifact-index"
CANARY_SCHEMA = "onnx-splitpoint/yolov7-native-three-stage-canary"
CORPUS_SCHEMA = "onnx-splitpoint/yolov7-fast-decode-parity-corpus"
REFERENCE_SCHEMA = "onnx-splitpoint/yolov7-multi-image-fast-decode-parity-canary"
DATASET_SCHEMA = "onnx-splitpoint/dataset-manifest"
QUALITY_BINDING_SCHEMA = "onnx-splitpoint/native-split-quality-binding"
RUNTIME_SCHEMA = "onnx-splitpoint/yolov7-native-three-stage-runtime"
BUILD_ID = "v2.79.6-remaining-changes-yolo11-admission-closure"
PACKAGE_VERSION = "2.79.6"
MODEL_ID = "yolov7_paper"
CASE_ID = "b066"
SETUP_ID = "orin_nx_hailo8_01"
PRECISION = "uint8_dequant_fp16"
CLAIM_SCOPE = "claim_gate_32"
CLAIM_ITEM_COUNT = 32
DATASET_MANIFEST_SHA256 = "2de36f0f8949e4f1fcbd0eefbda1f4d18b411208fa91e2dd2bf985a56d0f22e2"
REFERENCE_REPORT_SHA256 = "4f4aff24452f5833bdcec8de31fa1f3173ccabc2a78372744f8bf59cd5ee0cc9"
MAX_JSON_BYTES = 64 * 1024 * 1024
MAX_ZIP_BYTES = 64 * 1024 * 1024
REQUIRED_SOURCE_FILES = (
    "onnx_splitpoint_tool/__init__.py",
    "onnx_splitpoint_tool/release_identity.py",
    "onnx_splitpoint_tool/native_three_stage.py",
    "scripts/native_hailo_trt_fifo_from_benchmarkset.py",
    "scripts/native_hailo_trt_concurrent_three_stage_from_benchmarkset.py",
    "onnx_splitpoint_tool/resources/remote_scripts/native_hailo_trt_concurrent_three_stage_from_benchmarkset.py",
    "onnx_splitpoint_tool/resources/native_concurrent_three_stage_yolov7/stage_corpus.py",
    "onnx_splitpoint_tool/resources/native_concurrent_three_stage_yolov7/three_stage_canary.py",
    "onnx_splitpoint_tool/resources/native_concurrent_three_stage_yolov7/reference/dataset_detection_validation.json",
    "onnx_splitpoint_tool/resources/native_concurrent_three_stage_yolov7/reference/multi_image_fast_decode_canary_report.json",
)
SNAPSHOT_MEMBERS = (
    "onnx_splitpoint_tool/__init__.py",
    "onnx_splitpoint_tool/release_identity.py",
    "onnx_splitpoint_tool/native_three_stage.py",
    "scripts/native_hailo_trt_fifo_from_benchmarkset.py",
    "scripts/native_hailo_trt_concurrent_three_stage_from_benchmarkset.py",
)


class ClaimGateError(ValueError):
    """A stable, user-visible fail-closed verification error."""


def _need(condition: Any, reason: str) -> None:
    if not condition:
        raise ClaimGateError(reason)


def _strip_sha(value: Any, label: str) -> str:
    token = str(value or "").strip().lower()
    if token.startswith("sha256:"):
        token = token[7:]
    _need(
        len(token) == 64
        and all(character in "0123456789abcdef" for character in token),
        "%s_invalid_sha256" % label,
    )
    return token


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _mapping(value: Any, label: str) -> Dict[str, Any]:
    _need(isinstance(value, Mapping), "%s_not_mapping" % label)
    return dict(value)


def _positive_number(value: Any, label: str) -> float:
    _need(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0.0,
        "%s_not_positive_finite" % label,
    )
    return float(value)


def _open_dir_nofollow(path: Path, label: str) -> None:
    token = os.fspath(path)
    _need(os.path.isabs(token), "%s_not_absolute" % label)
    _need(os.path.normpath(token) == token, "%s_not_canonical" % label)
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open("/", flags)
    try:
        for part in Path(token).parts[1:]:
            _need(part not in ("", ".", ".."), "%s_unsafe" % label)
            child = os.open(part, flags, dir_fd=fd)
            os.close(fd)
            fd = child
        _need(stat.S_ISDIR(os.fstat(fd).st_mode), "%s_not_directory" % label)
    except ClaimGateError:
        raise
    except OSError as exc:
        raise ClaimGateError(
            "%s_unsafe_or_missing:%s" % (label, type(exc).__name__)
        ) from exc
    finally:
        try:
            os.close(fd)
        except OSError:
            pass


def _safe_relative(value: Any, label: str) -> Path:
    token = str(value or "").strip().replace("\\", "/")
    pure = PurePosixPath(token)
    _need(token and not pure.is_absolute(), "%s_absolute_or_empty" % label)
    _need(
        all(part not in ("", ".", "..") for part in pure.parts),
        "%s_unsafe" % label,
    )
    return Path(*pure.parts)


def _safe_file(root: Path, relative: Any, label: str) -> Path:
    rel = _safe_relative(relative, label)
    _open_dir_nofollow(root, "%s_root" % label)
    current = root
    try:
        for index, part in enumerate(rel.parts):
            current = current / part
            info = current.lstat()
            _need(
                not stat.S_ISLNK(info.st_mode),
                "%s_symlink_component" % label,
            )
            if index + 1 < len(rel.parts):
                _need(
                    stat.S_ISDIR(info.st_mode),
                    "%s_ancestor_not_directory" % label,
                )
            else:
                _need(stat.S_ISREG(info.st_mode), "%s_not_regular" % label)
    except ClaimGateError:
        raise
    except OSError as exc:
        raise ClaimGateError(
            "%s_missing:%s" % (label, type(exc).__name__)
        ) from exc
    return current


def _safe_directory(root: Path, relative: Any, label: str) -> Path:
    rel = _safe_relative(relative, label)
    _open_dir_nofollow(root, "%s_root" % label)
    current = root
    try:
        for part in rel.parts:
            current = current / part
            info = current.lstat()
            _need(
                not stat.S_ISLNK(info.st_mode),
                "%s_symlink_component" % label,
            )
            _need(stat.S_ISDIR(info.st_mode), "%s_not_directory" % label)
    except ClaimGateError:
        raise
    except OSError as exc:
        raise ClaimGateError(
            "%s_missing:%s" % (label, type(exc).__name__)
        ) from exc
    return current


def _load_json_file(path: Path, label: str) -> Any:
    _need(path.stat().st_size <= MAX_JSON_BYTES, "%s_too_large" % label)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ClaimGateError("%s_invalid_json" % label) from exc


def _json(root: Path, relative: Any, label: str) -> Any:
    return _load_json_file(_safe_file(root, relative, label), label)


def _path_relative_to(path: Path, base: Path) -> Optional[Path]:
    try:
        return path.relative_to(base)
    except ValueError:
        return None


def _map_captured_path(
    value: Any,
    *,
    evidence_root: Path,
    original_root: Path,
    label: str,
    directory: bool = False,
) -> Path:
    token = str(value or "").strip()
    _need(token and os.path.isabs(token), "%s_not_absolute" % label)
    _need(os.path.normpath(token) == token, "%s_not_canonical" % label)
    captured = Path(token)
    relative = _path_relative_to(captured, original_root)
    if relative is None:
        relative = _path_relative_to(captured, evidence_root)
    _need(relative is not None, "%s_outside_captured_evidence" % label)
    _need(relative != Path("."), "%s_is_evidence_root" % label)
    if directory:
        return _safe_directory(evidence_root, relative, label)
    return _safe_file(evidence_root, relative, label)


def _validate_source(
    source_root: Path, expected_manifest_sha256: str
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    _open_dir_nofollow(source_root, "source_root")
    manifest_path = _safe_file(source_root, "SOURCE_MANIFEST.json", "source_manifest")
    actual_manifest_sha = _file_sha256(manifest_path)
    _need(
        actual_manifest_sha == expected_manifest_sha256,
        "source_manifest_sha256_mismatch",
    )
    manifest = _mapping(_load_json_file(manifest_path, "source_manifest"), "source_manifest")
    _need(
        manifest.get("schema") == "onnx-splitpoint/source-manifest-v1",
        "source_manifest_schema",
    )
    _need(manifest.get("package_version") == PACKAGE_VERSION, "source_package_version")
    _need(manifest.get("workflow_version") == BUILD_ID, "source_build_id")
    rows = manifest.get("files")
    _need(isinstance(rows, list), "source_manifest_files_not_list")
    _need(manifest.get("file_count") == len(rows), "source_manifest_file_count")
    by_path: Dict[str, Dict[str, Any]] = {}
    for index, raw in enumerate(rows):
        row = _mapping(raw, "source_manifest_row_%d" % index)
        relative = _safe_relative(row.get("path"), "source_manifest_path_%d" % index)
        key = relative.as_posix()
        _need(key not in by_path, "source_manifest_duplicate_path:%s" % key)
        path = _safe_file(source_root, relative, "source_file_%d" % index)
        expected_size = row.get("size")
        if expected_size is None:
            expected_size = row.get("size_bytes")
        _need(
            isinstance(expected_size, int) and expected_size >= 0,
            "source_manifest_size_invalid:%s" % key,
        )
        _need(path.stat().st_size == expected_size, "source_file_size_mismatch:%s" % key)
        expected_sha = _strip_sha(row.get("sha256"), "source_file_%s" % key)
        _need(_file_sha256(path) == expected_sha, "source_file_sha256_mismatch:%s" % key)
        by_path[key] = row
    for relative in REQUIRED_SOURCE_FILES:
        _need(relative in by_path, "source_required_file_unregistered:%s" % relative)
    helper = _safe_file(
        source_root,
        "scripts/native_hailo_trt_concurrent_three_stage_from_benchmarkset.py",
        "source_helper",
    )
    vendored = _safe_file(
        source_root,
        "onnx_splitpoint_tool/resources/remote_scripts/native_hailo_trt_concurrent_three_stage_from_benchmarkset.py",
        "source_vendored_helper",
    )
    _need(_file_sha256(helper) == _file_sha256(vendored), "source_helper_vendor_drift")
    return manifest, by_path


def _validate_snapshot(
    path: Path, source_root: Path, expected_sha256: str
) -> Dict[str, Any]:
    _need(path.stat().st_size <= MAX_ZIP_BYTES, "source_snapshot_too_large")
    _need(_file_sha256(path) == expected_sha256, "source_snapshot_sha256_mismatch")
    prefix = "ONNX-Splitpoint-Tool_v%s" % PACKAGE_VERSION
    expected_names = ["%s/%s" % (prefix, relative) for relative in SNAPSHOT_MEMBERS]
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            names = [item.filename for item in infos]
            _need(len(names) == len(set(names)), "source_snapshot_duplicate_members")
            _need(names == expected_names, "source_snapshot_member_set_or_order")
            _need(archive.testzip() is None, "source_snapshot_crc_failure")
            for info, relative in zip(infos, SNAPSHOT_MEMBERS):
                pure = PurePosixPath(info.filename)
                _need(
                    pure.as_posix() == info.filename
                    and not pure.is_absolute()
                    and all(part not in ("", ".", "..") for part in pure.parts),
                    "source_snapshot_unsafe_member",
                )
                mode = (info.external_attr >> 16) & 0o170000
                _need(mode != stat.S_IFLNK, "source_snapshot_symlink_member")
                current = _safe_file(source_root, relative, "source_snapshot_current")
                _need(
                    hashlib.sha256(archive.read(info)).hexdigest() == _file_sha256(current),
                    "source_snapshot_current_source_mismatch:%s" % relative,
                )
    except (OSError, zipfile.BadZipFile) as exc:
        raise ClaimGateError("source_snapshot_invalid_zip") from exc
    return {"path": path.name, "sha256": expected_sha256, "member_count": len(expected_names)}


def _artifact_identity(binding: Mapping[str, Any], role: str) -> Dict[str, Any]:
    artifacts = _mapping(binding.get("artifacts"), "quality_binding_artifacts")
    row = _mapping(artifacts.get(role), "quality_binding_artifact_%s" % role)
    size = row.get("size_bytes")
    _need(isinstance(size, int) and size > 0, "quality_binding_%s_size" % role)
    return {"sha256": _strip_sha(row.get("sha256"), role), "size_bytes": size}


def _validate_external_artifacts(
    capture: Mapping[str, Any], binding: Mapping[str, Any], report: Mapping[str, Any]
) -> Dict[str, Dict[str, Any]]:
    captured = _mapping(
        capture.get("external_runtime_artifacts"), "external_runtime_artifacts"
    )
    runtime = _mapping(report.get("runtime_artifacts"), "report_runtime_artifacts")
    output: Dict[str, Dict[str, Any]] = {}
    for role, report_sha_key in (("part1_runtime", "hef_sha256"), ("engine", "engine_sha256")):
        expected = _artifact_identity(binding, role)
        observed = _mapping(captured.get(role), "captured_artifact_%s" % role)
        _need(observed.get("verified_at_capture") is True, "captured_%s_not_verified" % role)
        _need(
            _strip_sha(observed.get("sha256"), "captured_%s" % role)
            == expected["sha256"],
            "captured_%s_sha256_mismatch" % role,
        )
        _need(observed.get("size_bytes") == expected["size_bytes"], "captured_%s_size_mismatch" % role)
        _need(
            _strip_sha(runtime.get(report_sha_key), "report_%s" % role)
            == expected["sha256"],
            "report_%s_sha256_mismatch" % role,
        )
        output[role] = expected
    _need(runtime.get("source_mutated") is False, "canary_source_mutated")
    return output


def _validate_corpus(
    corpus: Mapping[str, Any], dataset: Mapping[str, Any], reference: Mapping[str, Any],
    corpus_path: Path,
) -> Tuple[List[int], List[Dict[str, Any]]]:
    _need(corpus.get("schema") == CORPUS_SCHEMA, "corpus_schema")
    _need(corpus.get("schema_version") == 1, "corpus_schema_version")
    _need(corpus.get("selection_uses_model_predictions") is False, "corpus_prediction_selected")
    _need(corpus.get("dataset_id") == "coco2017-val", "corpus_dataset_id")
    _need(corpus.get("requested_count") == CLAIM_ITEM_COUNT, "corpus_requested_count")
    _need(corpus.get("selected_count") == CLAIM_ITEM_COUNT, "corpus_selected_count")
    _need(
        _strip_sha(corpus.get("dataset_manifest_sha256"), "corpus_dataset_manifest")
        == DATASET_MANIFEST_SHA256,
        "corpus_dataset_manifest_sha256_mismatch",
    )
    items = corpus.get("items")
    _need(isinstance(items, list) and len(items) == CLAIM_ITEM_COUNT, "corpus_items_count")
    dataset_rows = dataset.get("items")
    _need(isinstance(dataset_rows, list), "dataset_items_not_list")
    by_id = {
        int(_mapping(row, "dataset_item").get("image_id")): _mapping(row, "dataset_item")
        for row in dataset_rows
    }
    ref_rows = reference.get("rows")
    _need(isinstance(ref_rows, list) and len(ref_rows) == CLAIM_ITEM_COUNT, "reference_rows_count")
    reference_ids = [int(_mapping(row, "reference_row").get("image_id")) for row in ref_rows]
    image_ids: List[int] = []
    identities: List[Dict[str, Any]] = []
    manifest_root = corpus_path.parent
    for index, raw in enumerate(items):
        row = _mapping(raw, "corpus_item_%d" % index)
        _need(row.get("index") == index, "corpus_item_order_%d" % index)
        image_id = int(row.get("image_id") or -1)
        _need(image_id >= 0 and image_id not in image_ids, "corpus_image_id_%d" % index)
        image_ids.append(image_id)
        source = by_id.get(image_id)
        _need(source is not None, "corpus_unknown_image_id:%d" % image_id)
        _need(row.get("relative_path") == source.get("relative_path"), "corpus_relative_path:%d" % image_id)
        expected_sha = _strip_sha(source.get("sha256"), "dataset_image_%d" % image_id)
        staged_sha = _strip_sha(
            row.get("staged_sha256") or row.get("sha256"),
            "corpus_image_%d" % image_id,
        )
        _need(staged_sha == expected_sha, "corpus_source_sha256:%d" % image_id)
        staged = _safe_file(manifest_root, row.get("staged_file"), "corpus_staged_%d" % index)
        _need(staged.stat().st_size == source.get("size_bytes"), "corpus_staged_size:%d" % image_id)
        _need(_file_sha256(staged) == expected_sha, "corpus_staged_sha256:%d" % image_id)
        identities.append(
            {"index": index, "image_id": image_id, "sha256": expected_sha, "size_bytes": staged.stat().st_size}
        )
    _need(image_ids == reference_ids, "corpus_reference_order_mismatch")
    return image_ids, identities


def _validate_runtime_reports(
    report: Mapping[str, Any], *, evidence_root: Path, original_root: Path,
    expected_ids: Sequence[int],
) -> Tuple[int, int, List[Dict[str, Any]]]:
    repetitions = report.get("repetitions")
    _need(isinstance(repetitions, list) and repetitions, "canary_repetitions_missing")
    requested_total = 0
    completed_total = 0
    summaries: List[Dict[str, Any]] = []
    for index, raw in enumerate(repetitions):
        repetition = _mapping(raw, "canary_repetition_%d" % index)
        _need(repetition.get("repetition") == index + 1, "repetition_sequence")
        _need(repetition.get("return_code") == 0 and repetition.get("runtime_ok") is True, "repetition_not_successful")
        _need(list(repetition.get("callback_errors") or []) == [], "repetition_callback_errors")
        parity = _mapping(repetition.get("measurement_result_parity"), "repetition_parity")
        _need(parity.get("requested_images") == CLAIM_ITEM_COUNT, "repetition_parity_requested")
        _need(parity.get("exact_images") == CLAIM_ITEM_COUNT, "repetition_parity_exact")
        _need(parity.get("all_exact") is True, "repetition_parity_failed")
        parity_rows = parity.get("rows")
        _need(isinstance(parity_rows, list) and len(parity_rows) == CLAIM_ITEM_COUNT, "repetition_parity_rows")
        _need(
            [int(_mapping(row, "repetition_parity_row").get("image_id")) for row in parity_rows]
            == list(expected_ids),
            "repetition_parity_order",
        )
        _need(all(_mapping(row, "repetition_parity_row").get("exact") is True for row in parity_rows), "repetition_inexact_row")
        runtime_path = _map_captured_path(
            repetition.get("runtime_report_path"), evidence_root=evidence_root,
            original_root=original_root, label="runtime_report_%d" % index,
        )
        runtime = _mapping(_load_json_file(runtime_path, "runtime_report"), "runtime_report")
        _need(runtime.get("schema") == RUNTIME_SCHEMA and runtime.get("schema_version") == 1, "runtime_schema")
        _need(runtime.get("ok") is True, "runtime_not_ok")
        frames = runtime.get("frames")
        _need(isinstance(frames, int) and frames > 0, "runtime_frames")
        _need(runtime.get("image_count") == CLAIM_ITEM_COUNT, "runtime_image_count")
        _need(runtime.get("warmup_fully_drained_before_measurement") is True, "runtime_warmup_not_drained")
        warmup = _mapping(runtime.get("warmup_result"), "runtime_warmup")
        _need(
            isinstance(warmup.get("requested"), int)
            and warmup.get("requested") == warmup.get("raw_completed")
            and warmup.get("requested") == warmup.get("completed")
            and warmup.get("callback_failures") == 0,
            "runtime_warmup_cardinality",
        )
        raw_endpoint = _mapping(runtime.get("raw_model_outputs"), "runtime_raw_endpoint")
        completed_endpoint = _mapping(runtime.get("completed_detection"), "runtime_completed_endpoint")
        _need(raw_endpoint.get("completed_frames") == frames, "runtime_raw_work_units")
        _need(completed_endpoint.get("completed_frames") == frames, "runtime_completed_work_units")
        _positive_number(raw_endpoint.get("throughput_fps"), "runtime_raw_fps")
        _positive_number(completed_endpoint.get("throughput_fps"), "runtime_completed_fps")
        _need(runtime.get("callback_failures") == 0, "runtime_callback_failures")
        requested_total += frames
        completed_total += int(completed_endpoint.get("completed_frames"))
        summaries.append(
            {"repetition": index + 1, "requested_work_units": frames,
             "completed_work_units": int(completed_endpoint.get("completed_frames")),
             "runtime_report_sha256": _file_sha256(runtime_path)}
        )
    _need(requested_total == completed_total, "aggregate_completed_work_units_mismatch")
    return requested_total, completed_total, summaries


def _validate_artifact_index(
    result_path: Path, result: Mapping[str, Any], invocation: Mapping[str, Any],
    *, evidence_root: Path, original_root: Path,
) -> Dict[str, Any]:
    out_root = _map_captured_path(
        result.get("out_root"), evidence_root=evidence_root, original_root=original_root,
        label="three_stage_out_root", directory=True,
    )
    index_path = _safe_file(out_root, "three_stage_artifact_index.json", "three_stage_artifact_index")
    index = _mapping(_load_json_file(index_path, "three_stage_artifact_index"), "three_stage_artifact_index")
    _need(index.get("schema") == ARTIFACT_INDEX_SCHEMA and index.get("schema_version") == 1, "three_stage_artifact_index_schema")
    _need(index.get("out_root") == result.get("out_root"), "artifact_index_out_root")
    _need(index.get("out_root_binding_sha256") == result.get("out_root_binding_sha256"), "artifact_index_out_root_binding")
    _need(index.get("invocation_sha256") == invocation.get("invocation_sha256"), "artifact_index_invocation")
    artifacts = _mapping(index.get("artifacts"), "three_stage_artifacts")
    required = ("invocation_receipt", "remote_command_receipt", "canonical_result", "child_report", "child_console")
    verified: Dict[str, Dict[str, Any]] = {}
    for role in required:
        item = _mapping(artifacts.get(role), "artifact_%s" % role)
        path = _map_captured_path(
            item.get("path"), evidence_root=evidence_root, original_root=original_root,
            label="artifact_%s" % role,
        )
        expected_sha = _strip_sha(item.get("sha256"), "artifact_%s" % role)
        _need(path.stat().st_size == item.get("size_bytes"), "artifact_%s_size" % role)
        _need(_file_sha256(path) == expected_sha, "artifact_%s_sha256" % role)
        verified[role] = {"sha256": expected_sha, "size_bytes": path.stat().st_size}
    _need(artifacts.get("failure_receipt") is None, "unexpected_failure_receipt")
    _need(verified["canonical_result"]["sha256"] == _file_sha256(result_path), "artifact_index_result_identity")
    result_report_path = _map_captured_path(
        result.get("canary_report"), evidence_root=evidence_root,
        original_root=original_root, label="artifact_index_result_report",
    )
    _need(
        verified["child_report"]["sha256"] == _file_sha256(result_report_path),
        "artifact_index_child_report_identity",
    )
    invocation_path = _map_captured_path(
        _mapping(artifacts.get("invocation_receipt"), "invocation_artifact").get("path"),
        evidence_root=evidence_root, original_root=original_root, label="invocation_receipt",
    )
    _need(_load_json_file(invocation_path, "invocation_receipt") == invocation, "invocation_receipt_result_mismatch")
    command_path = _map_captured_path(
        _mapping(artifacts.get("remote_command_receipt"), "command_artifact").get("path"),
        evidence_root=evidence_root, original_root=original_root, label="remote_command_receipt",
    )
    command = _mapping(_load_json_file(command_path, "remote_command_receipt"), "remote_command_receipt")
    _need(command.get("schema") == "onnx-splitpoint/three-stage-remote-command-receipt", "remote_command_schema")
    _need(command.get("invocation_sha256") == invocation.get("invocation_sha256"), "remote_command_invocation")
    argv = command.get("argv")
    _need(isinstance(argv, list) and all(isinstance(value, str) for value in argv), "remote_command_argv")
    for flag, value in (
        ("--corpus", invocation.get("corpus_manifest")),
        ("--reference-report", invocation.get("reference_report")),
        ("--out-root", invocation.get("out_root")),
        ("--expected-corpus-count", str(CLAIM_ITEM_COUNT)),
    ):
        _need(flag in argv, "remote_command_missing_%s" % flag[2:])
        position = argv.index(flag)
        _need(position + 1 < len(argv) and argv[position + 1] == value, "remote_command_value_%s" % flag[2:])
    return {
        "sha256": _file_sha256(index_path),
        "artifacts": verified,
        "out_root_binding_sha256": result.get("out_root_binding_sha256"),
    }


def verify_claim_gate(
    *, result_json: Path, evidence_root: Path, source_root: Path,
    expected_source_manifest_sha256: str,
) -> Dict[str, Any]:
    evidence_root = Path(os.path.abspath(os.fspath(evidence_root)))
    source_root = Path(os.path.abspath(os.fspath(source_root)))
    _open_dir_nofollow(evidence_root, "evidence_root")
    expected_manifest_sha = _strip_sha(
        expected_source_manifest_sha256, "expected_source_manifest"
    )
    source_manifest, _ = _validate_source(source_root, expected_manifest_sha)

    result_path_input = Path(os.path.abspath(os.fspath(result_json)))
    relative_result = _path_relative_to(result_path_input, evidence_root)
    _need(relative_result is not None and relative_result != Path("."), "result_outside_evidence_root")
    result_path = _safe_file(evidence_root, relative_result, "result_json")
    result = _mapping(_load_json_file(result_path, "result_json"), "result_json")
    capture = _mapping(_json(evidence_root, "capture_manifest.json", "capture_manifest"), "capture_manifest")
    _need(capture.get("schema") == CAPTURE_SCHEMA and capture.get("schema_version") == 1, "capture_manifest_schema")
    original_token = str(capture.get("original_evidence_root") or "")
    _need(original_token and os.path.isabs(original_token) and os.path.normpath(original_token) == original_token, "capture_original_root")
    original_root = Path(original_token)
    _need(
        _strip_sha(capture.get("source_manifest_sha256"), "capture_source_manifest")
        == expected_manifest_sha,
        "capture_source_manifest_sha256_mismatch",
    )
    run_id = str(capture.get("run_id") or "").strip()
    _need(run_id and "/" not in run_id and "\\" not in run_id, "capture_run_id")

    required_result = {
        "ok": True,
        "schema": RESULT_SCHEMA,
        "schema_version": RESULT_SCHEMA_VERSION,
        "build_id": BUILD_ID,
        "performance_endpoint": "p2_output",
        "application_performance_endpoint": "completed_detection",
        "endpoint_execution_policy": "concurrent_three_stage_single_invocation",
        "three_stage_concurrency_directly_measured": True,
        "three_stage_hardware_integration_status": "passed",
        "quality_oracle_location": "outside_performance_timing",
        "quality_oracle_status": "passed",
        "endpoint_relation_verified": True,
        "directly_measured": True,
        "execution_scope": CLAIM_SCOPE,
        "claim_eligible": True,
        "expected_item_count": CLAIM_ITEM_COUNT,
        "child_returncode": 0,
        "vendored_runtime_rc": 0,
    }
    for key, expected in required_result.items():
        _need(result.get(key) == expected, "result_contract_%s" % key)
    p2_fps = _positive_number(result.get("p2_output_fps"), "p2_output_fps")
    completed_fps = _positive_number(result.get("completed_detection_fps"), "completed_detection_fps")
    ratio = _positive_number(result.get("completed_to_p2_ratio"), "completed_to_p2_ratio")
    _need(result.get("failure_class") in (None, "") and result.get("failure_reason") in (None, ""), "result_contains_failure")
    _need(str(result.get("canary_status") or "").upper().startswith("PASS"), "result_canary_status")
    stage_timings = _mapping(result.get("stage_timings"), "result_stage_timings")
    _need(
        stage_timings.get("schema")
        == "onnx-splitpoint/native-three-stage-timing-projection",
        "result_stage_timings_schema",
    )
    for stage in ("P1", "P2", "Post"):
        _need(bool(_mapping(stage_timings.get(stage), "result_stage_%s" % stage)), "result_stage_%s_empty" % stage)

    identity = _mapping(result.get("identity"), "result_identity")
    for key, expected in (
        ("model_id", MODEL_ID), ("case_id", CASE_ID), ("setup_id", SETUP_ID),
        ("precision", PRECISION), ("eval_run_id", run_id),
    ):
        _need(identity.get(key) == expected, "result_identity_%s" % key)
    _need(str(identity.get("source_run_id") or "").strip(), "result_source_run_id")

    invocation = _mapping(result.get("invocation"), "invocation")
    _need(invocation.get("schema") == INVOCATION_SCHEMA and invocation.get("schema_version") == 1, "invocation_schema")
    claimed_invocation_sha = _strip_sha(invocation.get("invocation_sha256"), "invocation")
    unsigned_invocation = dict(invocation)
    unsigned_invocation.pop("invocation_sha256", None)
    _need(_canonical_sha256(unsigned_invocation) == claimed_invocation_sha, "invocation_sha256_mismatch")
    for key, expected in (
        ("model_id", MODEL_ID), ("case_id", CASE_ID), ("setup_id", SETUP_ID),
        ("precision", PRECISION), ("execution_scope", CLAIM_SCOPE),
        ("expected_item_count", CLAIM_ITEM_COUNT), ("claim_eligible", True),
    ):
        _need(invocation.get(key) == expected, "invocation_%s" % key)
    expected_out_binding = _canonical_sha256(
        {
            "out_root": str(invocation.get("out_root") or ""),
            "execution_scope": CLAIM_SCOPE,
            "model_id": MODEL_ID,
            "case_id": CASE_ID,
            "setup_id": SETUP_ID,
        }
    )
    _need(
        invocation.get("out_root_binding_sha256") == expected_out_binding,
        "invocation_out_root_binding_sha256",
    )
    _need(result.get("out_root") == invocation.get("out_root"), "result_invocation_out_root")
    _need(
        result.get("out_root_binding_sha256") == expected_out_binding,
        "result_out_root_binding_sha256",
    )

    product = _mapping(result.get("product_execution_context"), "product_execution_context")
    _need(product.get("execution_scope") == CLAIM_SCOPE and product.get("claim_eligible") is True, "product_claim_scope")
    _need(product.get("performance_corpus_count") == CLAIM_ITEM_COUNT, "product_corpus_count")
    _need(product.get("quality_oracle_inside_performance_timing") is False, "product_oracle_inside_timing")
    path_fields = (
        ("performance_corpus_manifest", "corpus_manifest"),
        ("dataset_manifest", "dataset_manifest"),
        ("postflight_reference_report", "reference_report"),
        ("quality_binding", "quality_binding"),
        ("canary_out_root", "out_root"),
    )
    for product_key, invocation_key in path_fields:
        _need(product.get(product_key) == invocation.get(invocation_key), "product_invocation_path_%s" % product_key)
    hash_fields = (
        ("performance_corpus_manifest_sha256", "corpus_manifest_sha256"),
        ("dataset_manifest_sha256", "dataset_manifest_sha256"),
        ("postflight_reference_report_sha256", "reference_report_sha256"),
        ("quality_binding_sha256", "quality_binding_sha256"),
        ("canary_out_root_binding_sha256", "out_root_binding_sha256"),
    )
    for product_key, invocation_key in hash_fields:
        _need(product.get(product_key) == invocation.get(invocation_key), "product_invocation_hash_%s" % product_key)

    dataset_path = _map_captured_path(invocation.get("dataset_manifest"), evidence_root=evidence_root, original_root=original_root, label="dataset_manifest")
    reference_path = _map_captured_path(invocation.get("reference_report"), evidence_root=evidence_root, original_root=original_root, label="reference_report")
    corpus_path = _map_captured_path(invocation.get("corpus_manifest"), evidence_root=evidence_root, original_root=original_root, label="corpus_manifest")
    binding_path = _map_captured_path(invocation.get("quality_binding"), evidence_root=evidence_root, original_root=original_root, label="quality_binding")
    for path, claimed, expected, label in (
        (dataset_path, invocation.get("dataset_manifest_sha256"), DATASET_MANIFEST_SHA256, "dataset_manifest"),
        (reference_path, invocation.get("reference_report_sha256"), REFERENCE_REPORT_SHA256, "reference_report"),
        (corpus_path, invocation.get("corpus_manifest_sha256"), None, "corpus_manifest"),
        (binding_path, invocation.get("quality_binding_sha256"), None, "quality_binding"),
    ):
        actual = _file_sha256(path)
        _need(actual == _strip_sha(claimed, label), "%s_invocation_sha256" % label)
        if expected is not None:
            _need(actual == expected, "%s_frozen_sha256" % label)

    dataset = _mapping(_load_json_file(dataset_path, "dataset_manifest"), "dataset_manifest")
    _need(dataset.get("schema") == DATASET_SCHEMA and dataset.get("schema_version") == 1, "dataset_manifest_schema")
    _need(dataset.get("dataset_id") == "coco2017-val" and dataset.get("item_count") == 5000, "dataset_manifest_identity")
    dataset_items = dataset.get("items")
    _need(isinstance(dataset_items, list) and len(dataset_items) == 5000, "dataset_manifest_items")
    reference = _mapping(_load_json_file(reference_path, "reference_report"), "reference_report")
    _need(reference.get("schema") == REFERENCE_SCHEMA and reference.get("schema_version") == 1, "reference_report_schema")
    _need(str(reference.get("status") or "").upper().startswith("PASS"), "reference_report_status")
    reference_corpus = _mapping(reference.get("corpus"), "reference_corpus")
    _need(_strip_sha(reference_corpus.get("dataset_manifest_sha256"), "reference_dataset") == DATASET_MANIFEST_SHA256, "reference_dataset_sha256")
    corpus = _mapping(_load_json_file(corpus_path, "corpus_manifest"), "corpus_manifest")
    image_ids, image_identities = _validate_corpus(corpus, dataset, reference, corpus_path)

    binding = _mapping(_load_json_file(binding_path, "quality_binding"), "quality_binding")
    _need(binding.get("schema") == QUALITY_BINDING_SCHEMA, "quality_binding_schema")
    _need(binding.get("quality_completed") is True, "quality_binding_not_completed")
    _need(_file_sha256(binding_path) == _strip_sha(identity.get("binding_sha256"), "result_binding"), "result_binding_sha256")

    report_path = _map_captured_path(result.get("canary_report"), evidence_root=evidence_root, original_root=original_root, label="canary_report")
    report = _mapping(_load_json_file(report_path, "canary_report"), "canary_report")
    _need(report.get("schema") == CANARY_SCHEMA and report.get("schema_version") == 1, "canary_report_schema")
    _need(str(report.get("status") or "").upper().startswith("PASS"), "canary_report_status")
    _need(report.get("quality_oracle_outside_performance_timing") is True, "canary_oracle_inside_timing")
    _need(report.get("performance_hotloop_contains_crypto_hashing") is False, "canary_hotloop_hashing")
    _need(report.get("source_ok") is True, "canary_source_not_ok")
    _need(_mapping(report.get("corpus"), "canary_corpus") == corpus, "canary_corpus_manifest_mismatch")
    for label, rows in (("reference_metadata", report.get("reference_metadata_checks")), ("source", report.get("source_checks"))):
        _need(isinstance(rows, list) and rows, "canary_%s_checks_missing" % label)
        for row in rows:
            check = _mapping(row, "canary_%s_check" % label)
            _need(check.get("passed") is True, "canary_%s_check_failed" % label)
            _need(_strip_sha(check.get("expected_sha256"), "canary_%s_expected" % label) == _strip_sha(check.get("actual_sha256"), "canary_%s_actual" % label), "canary_%s_check_hash" % label)
    external_artifacts = _validate_external_artifacts(capture, binding, report)

    measurement = _mapping(report.get("measurement_contract"), "measurement_contract")
    _need(measurement.get("images_cycled") == CLAIM_ITEM_COUNT, "measurement_images_cycled")
    _need(measurement.get("completed_endpoint") == "completed_detection", "measurement_completed_endpoint")
    _need("outside" in str(measurement.get("quality_oracle") or ""), "measurement_oracle_timing")
    requested_work, completed_work, runtime_summaries = _validate_runtime_reports(
        report, evidence_root=evidence_root, original_root=original_root, expected_ids=image_ids
    )
    aggregate = _mapping(report.get("aggregate"), "canary_aggregate")
    _need(aggregate.get("repetitions_valid") == len(runtime_summaries), "aggregate_valid_repetitions")
    _need(aggregate.get("repetitions_requested") == len(runtime_summaries), "aggregate_requested_repetitions")

    postflight = _mapping(report.get("postflight_quality_oracle"), "postflight_quality_oracle")
    for key, expected in (
        ("performance_interpretation", "forbidden_verification_only"),
        ("return_code", 0), ("requested_images", CLAIM_ITEM_COUNT),
        ("verified_images", CLAIM_ITEM_COUNT), ("exact_images", CLAIM_ITEM_COUNT),
        ("all_exact", True),
    ):
        _need(postflight.get(key) == expected, "postflight_%s" % key)
    _need(list(postflight.get("callback_errors") or []) == [], "postflight_callback_errors")
    postflight_rows = postflight.get("rows")
    _need(isinstance(postflight_rows, list) and len(postflight_rows) == CLAIM_ITEM_COUNT, "postflight_rows")
    _need([int(_mapping(row, "postflight_row").get("image_id")) for row in postflight_rows] == image_ids, "postflight_order")
    for index, row in enumerate(postflight_rows):
        item = _mapping(row, "postflight_row")
        _need(item.get("sequence") == index, "postflight_row_sequence")
        _need(item.get("image_index") == index, "postflight_row_image_index")
        for key in ("raw_head_sha256_exact", "oracle_fast_exact", "current_oracle_equals_prior_oracle", "exact"):
            _need(item.get(key) is True, "postflight_row_%s" % key)

    projected = _mapping(result.get("oracle_parity"), "result_oracle_parity")
    _need(projected.get("schema") == "onnx-splitpoint/native-three-stage-oracle-parity" and projected.get("status") == "passed", "result_oracle_parity_status")
    _need(projected.get("inside_performance_timing") is False, "result_oracle_inside_timing")
    projected_postflight = _mapping(projected.get("postflight"), "result_projected_postflight")
    for key in ("requested_images", "verified_images", "exact_images", "all_exact"):
        _need(projected_postflight.get(key) == postflight.get(key), "result_projected_postflight_%s" % key)
    projected_repetitions = projected.get("measurement_repetitions")
    _need(
        isinstance(projected_repetitions, list)
        and len(projected_repetitions) == len(runtime_summaries),
        "result_projected_repetition_count",
    )
    for index, row in enumerate(projected_repetitions):
        item = _mapping(row, "result_projected_repetition")
        _need(item.get("repetition") == index + 1, "result_projected_repetition_sequence")
        _need(item.get("requested_images") == CLAIM_ITEM_COUNT, "result_projected_repetition_requested")
        _need(item.get("exact_images") == CLAIM_ITEM_COUNT, "result_projected_repetition_exact")
        _need(item.get("all_exact") is True, "result_projected_repetition_failed")

    artifact_index = _validate_artifact_index(
        result_path, result, invocation, evidence_root=evidence_root, original_root=original_root
    )
    snapshot_path = _map_captured_path(product.get("source_snapshot"), evidence_root=evidence_root, original_root=original_root, label="source_snapshot")
    snapshot_sha = _strip_sha(product.get("source_snapshot_sha256"), "source_snapshot")
    snapshot = _validate_snapshot(snapshot_path, source_root, snapshot_sha)
    _need(product.get("source_snapshot_prefix") == "ONNX-Splitpoint-Tool_v2.79.6", "source_snapshot_prefix")

    capture_sha = _file_sha256(_safe_file(evidence_root, "capture_manifest.json", "capture_manifest_identity"))
    result_sha = _file_sha256(result_path)
    report_sha = _file_sha256(report_path)
    receipt: Dict[str, Any] = {
        "schema": VERDICT_SCHEMA,
        "schema_version": 1,
        "status": "PASS",
        "ok": True,
        "valid_claim": True,
        "model_id": MODEL_ID,
        "case_id": CASE_ID,
        "claim_gate": CLAIM_SCOPE,
        "required_item_count": CLAIM_ITEM_COUNT,
        "passed_item_count": int(postflight.get("exact_images")),
        "claim_eligible": True,
        "run_id": run_id,
        "setup_id": SETUP_ID,
        "source_identity": {
            "package_version": source_manifest.get("package_version"),
            "build_id": source_manifest.get("workflow_version"),
            "source_manifest_sha256": expected_manifest_sha,
            "source_release_sha256": _strip_sha(capture.get("source_release_sha256"), "source_release"),
        },
        "invocation_identity": {
            "invocation_sha256": claimed_invocation_sha,
            "corpus_manifest_sha256": _file_sha256(corpus_path),
            "dataset_manifest_sha256": _file_sha256(dataset_path),
            "reference_report_sha256": _file_sha256(reference_path),
            "quality_binding_sha256": _file_sha256(binding_path),
            "out_root_binding_sha256": result.get("out_root_binding_sha256"),
            "ordered_image_ids": image_ids,
            "ordered_image_artifacts": image_identities,
        },
        "output_identity": {
            "result_sha256": result_sha,
            "canary_report_sha256": report_sha,
            "three_stage_artifact_index_sha256": artifact_index["sha256"],
            "source_snapshot_sha256": snapshot_sha,
            "p2_output_fps": p2_fps,
            "completed_detection_fps": completed_fps,
            "completed_to_p2_ratio": ratio,
            "requested_work_units": requested_work,
            "completed_work_units": completed_work,
            "runtime_repetitions": runtime_summaries,
        },
        "artifact_identities": external_artifacts,
        "evidence_identity": {
            "capture_manifest_sha256": capture_sha,
            "artifact_index": artifact_index,
            "source_snapshot": snapshot,
        },
        "checks": {
            "release_source_manifest": True,
            "source_snapshot": True,
            "immutable_invocation": True,
            "exact_ordered_corpus_32": True,
            "direct_concurrent_endpoints": True,
            "requested_equals_completed_work_units": requested_work == completed_work,
            "measurement_parity_32_of_32": True,
            "postflight_parity_32_of_32": True,
            "quality_oracle_outside_timing": True,
            "runtime_artifact_identities": True,
            "three_stage_artifact_index": True,
        },
    }
    return receipt


def _write_receipt(path: Path, payload: Mapping[str, Any]) -> None:
    token = os.fspath(path)
    _need(os.path.isabs(token), "output_not_absolute")
    _need(os.path.normpath(token) == token, "output_not_canonical")
    parent = path.parent
    _open_dir_nofollow(parent, "output_parent")
    if path.exists() or path.is_symlink():
        info = path.lstat()
        _need(stat.S_ISREG(info.st_mode) and not stat.S_ISLNK(info.st_mode), "output_unsafe_existing")
    temporary = parent / (".%s.%d.tmp" % (path.name, os.getpid()))
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(os.fspath(temporary), flags, 0o600)
    try:
        data = (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
        with os.fdopen(fd, "wb") as stream:
            fd = -1
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(os.fspath(temporary), token)
    finally:
        if fd >= 0:
            os.close(fd)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-json", required=True)
    parser.add_argument("--evidence-root", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--expected-source-manifest-sha256", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    output = Path(os.path.abspath(args.output))
    try:
        receipt = verify_claim_gate(
            result_json=Path(args.result_json),
            evidence_root=Path(args.evidence_root),
            source_root=Path(args.source_root),
            expected_source_manifest_sha256=args.expected_source_manifest_sha256,
        )
        _write_receipt(output, receipt)
        print(json.dumps(receipt, indent=2, ensure_ascii=False))
        print("V2796_YOLOV7_CLAIM_GATE=PASS")
        return 0
    except Exception as exc:
        failure = {
            "schema": VERDICT_SCHEMA,
            "schema_version": 1,
            "status": "FAIL",
            "ok": False,
            "valid_claim": False,
            "model_id": MODEL_ID,
            "case_id": CASE_ID,
            "claim_gate": CLAIM_SCOPE,
            "required_item_count": CLAIM_ITEM_COUNT,
            "passed_item_count": 0,
            "claim_eligible": False,
            "failure_class": type(exc).__name__,
            "failure_reason": str(exc),
        }
        try:
            _write_receipt(output, failure)
        except Exception as output_exc:
            failure["output_failure"] = "%s:%s" % (type(output_exc).__name__, output_exc)
        print(json.dumps(failure, indent=2, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
