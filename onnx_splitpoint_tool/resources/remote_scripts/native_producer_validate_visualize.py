#!/usr/bin/env python3
"""Native producer validation and visualization stage.

v59cq extends the native dump visualizer with conservative task-semantic checks.
It still separates artifact generation from claim validation:

- artifact_generated: dump/PNG/MD was created.
- tensor_ok: dump is readable and finite.
- semantic_ok: task-level comparison passed.
- legacy claim_ok / ok: contract-semantic compatibility; thesis ranking uses explicit buildable/runtime_executable/contract_consistent/task_valid/accuracy_gate_pass/eligible_for_ranking fields.

Classification semantic validation compares native TopK to an ORT-CPU/full
reference validation report when available.  Detection semantic validation decodes
common YOLO/NMS tensor formats, draws box overlays, and compares native boxes to
ORT-CPU/full reference boxes via IoU@0.5.  If no suitable reference or decoder is
available, semantic_ok is ``unavailable`` and the row is not claimable.
"""
from __future__ import annotations
from dataclasses import replace

import argparse
import copy
import csv
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

try:
    from onnx_splitpoint_tool.native_detection_postprocess import (
        FrozenDetectionPostprocessor as _FrozenDetectionPostprocessor,
        FrozenDecodedNmsPostprocessor as _FrozenDecodedNmsPostprocessor,
        DetectionCompletionRuntime as _DetectionCompletionRuntime,
        FrozenPostprocessError as _FrozenPostprocessError,
        build_normalized_detection_endpoint_attestation as _build_normalized_detection_endpoint_attestation,
        canonical_json_sha256 as _canonical_detection_json_sha256,
        verify_completed_detection_comparison_endpoint_contract,
        verify_detection_completion_execution_attestation,
        verify_detection_completion_execution_contract,
        verify_frozen_decoded_nms_normalization_contract,
        verify_frozen_postprocess_contract,
    )
    _verify_completed_detection_comparison_endpoint_contract = (
        verify_completed_detection_comparison_endpoint_contract
    )
    _verify_frozen_postprocess_contract = (
        verify_frozen_postprocess_contract
    )
    _verify_frozen_decoded_nms_normalization_contract = (
        verify_frozen_decoded_nms_normalization_contract
    )
    _verify_detection_completion_execution_contract = (
        verify_detection_completion_execution_contract
    )
    _verify_detection_completion_execution_attestation = (
        verify_detection_completion_execution_attestation
    )
except Exception:  # pragma: no cover - copied validators fail closed
    _FrozenDetectionPostprocessor = None
    _FrozenDecodedNmsPostprocessor = None
    _DetectionCompletionRuntime = None
    _FrozenPostprocessError = RuntimeError
    _build_normalized_detection_endpoint_attestation = None
    _canonical_detection_json_sha256 = None
    _verify_completed_detection_comparison_endpoint_contract = None
    _verify_frozen_decoded_nms_normalization_contract = None
    _verify_frozen_postprocess_contract = None
    _verify_detection_completion_execution_contract = None
    _verify_detection_completion_execution_attestation = None

from onnx_splitpoint_tool.native_output_endpoint import (
    DECODED_NMS_ATTESTATION_SOURCE,
)
from onnx_splitpoint_tool.native_command_contract import (
    verify_native_command_contract,
)
from onnx_splitpoint_tool.validation.host_postprocess import (
    resolve_host_postprocess_evidence,
)

try:
    from onnx_splitpoint_tool.runners.harness.base import (
        postprocess_result_to_dict as _canonical_postprocess_result_to_dict,
    )
    from onnx_splitpoint_tool.runners.harness.yolo import (
        YoloHarness as _CanonicalYoloHarness,
    )
except Exception:  # pragma: no cover - generated BenchmarkSet layout
    try:
        from splitpoint_runners.harness.base import (  # type: ignore
            postprocess_result_to_dict as _canonical_postprocess_result_to_dict,
        )
        from splitpoint_runners.harness.yolo import (  # type: ignore
            YoloHarness as _CanonicalYoloHarness,
        )
    except Exception:  # pragma: no cover - decoder absence remains fail-closed
        _canonical_postprocess_result_to_dict = None
        _CanonicalYoloHarness = None

try:
    from onnx_splitpoint_tool.native_split_quality import (
        bind_quality_to_native_split,
        validate_central_native_split_quality_selection,
        validate_native_split_quality_binding,
    )
except Exception:  # pragma: no cover - copied validators must fail closed
    bind_quality_to_native_split = None
    validate_central_native_split_quality_selection = None
    validate_native_split_quality_binding = None

try:
    from onnx_splitpoint_tool.native_split_quality_authority import (
        apply_native_split_quality_authority,
        canonical_native_split_backend,
        is_native_split_backend,
        native_split_quality_required_for_row,
        resolve_native_split_quality_authority,
    )
except Exception:  # pragma: no cover - copied validators must fail closed
    apply_native_split_quality_authority = None
    canonical_native_split_backend = None
    is_native_split_backend = None
    native_split_quality_required_for_row = None
    resolve_native_split_quality_authority = None

try:
    from onnx_splitpoint_tool.quality_service import (
        _validate_candidate_execution_contract as _validate_quality_producer_contract,
    )
except Exception:  # pragma: no cover - copied validators must fail closed
    _validate_quality_producer_contract = None

try:
    from scripts.native_producer_energy_plan import (
        _verify_full_command_contract as _strict_verify_full_command_contract,
    )
except Exception:  # pragma: no cover - standalone remote-script layout
    try:
        from native_producer_energy_plan import (  # type: ignore
            _verify_full_command_contract as _strict_verify_full_command_contract,
        )
    except Exception:  # pragma: no cover - missing verifier is not claimable
        _strict_verify_full_command_contract = None


_SHA256_HEX_RE = re.compile(r'^[0-9a-f]{64}$')
_TRT_QUALITY_PRODUCER_SCHEMA = (
    'onnx-splitpoint/tensorrt-central-quality-producer-identity'
)


def _normalize_sha256(value: Any) -> str:
    """Return one canonical SHA-256 hex digest or an empty string.

    Workflow file digests use the public ``sha256:<hex>`` serialization while
    several native producers emit bare hex.  Accept exactly those two forms;
    malformed values and repeated/foreign prefixes remain fail-closed.
    """
    text = str(value or '').strip().lower()
    if text.startswith('sha256:'):
        text = text[len('sha256:'):]
    return text if _SHA256_HEX_RE.fullmatch(text) else ''


def _consistent_sha256(
    *values: Any, required: bool = True,
) -> tuple[str, bool]:
    """Normalize duplicate SHA evidence and reject malformed/conflicting data."""
    observed: set[str] = set()
    present = False
    for value in values:
        if value in (None, '') or (isinstance(value, str) and not value.strip()):
            continue
        present = True
        normalized = _normalize_sha256(value)
        if not normalized:
            return '', False
        observed.add(normalized)
    if len(observed) > 1:
        return '', False
    if not present:
        return '', not required
    return next(iter(observed)), True


def _project_performance_input_contract_mode(
    *sources: Mapping[str, Any],
) -> tuple[str, str]:
    """Project one producer-sealed input mode without losing conflicts.

    Native Full DeepX writes this field into the performance result, while
    the semantic validator consumes a newly projected row.  Treat every
    available producer/report/dump copy as evidence: one normalized value is
    preserved, conflicting values fail closed, and absence stays explicit.
    """
    observed = {
        str(source.get('performance_input_contract_mode') or '').strip().lower()
        for source in sources
        if isinstance(source, Mapping)
        and str(
            source.get('performance_input_contract_mode') or ''
        ).strip()
    }
    if not observed:
        return 'missing', 'missing'
    if len(observed) != 1:
        return 'conflict', 'conflict'
    return next(iter(observed)), 'projected_consistent'


def _decoded_nms_attestation_passed(payload: dict[str, Any]) -> bool:
    attestation = payload.get('output_endpoint_attestation')
    endpoint_hash = _normalize_sha256(payload.get('endpoint_contract_hash'))
    return bool(
        str(payload.get('contract_source') or '').strip().lower()
        == DECODED_NMS_ATTESTATION_SOURCE
        and payload.get('endpoint_contract_complete') is True
        and bool(endpoint_hash)
        and isinstance(attestation, dict)
        and attestation.get('attested') is True
        and attestation.get('declaration_attested') is True
        and str(attestation.get('status') or '').strip().lower() == 'passed'
        and str(attestation.get('contract_source') or '').strip().lower()
        == DECODED_NMS_ATTESTATION_SOURCE
        and _normalize_sha256(attestation.get('endpoint_contract_hash')) == endpoint_hash
    )


def _completed_execution_attestation_passed(
    payload: Mapping[str, Any],
) -> bool:
    resolved = resolve_host_postprocess_evidence(payload)
    return bool(
        resolved.get("available") is True
        and str(resolved.get("status") or "") == "passed"
        and str(resolved.get("source") or "")
        in {"detection_completion_execution_v1", "native_three_stage_fast_oracle_outside_timing"}
    )


_COMPLETED_V2_PROJECTION_FIELDS = (
    "endpoint_contract_hash",
    "physical_endpoint_contract_hash",
    "output_endpoint_id",
    "physical_output_endpoint_id",
    "comparison_output_endpoint_id",
    "accelerator_endpoint_contract_hash",
    "host_postprocessing_available",
    "host_tail_available",
    "host_postprocess_required",
    "host_tail_required",
    "host_postprocessing_legacy_alias_conflict",
    "postprocess_included",
    "postprocess_completed_frames",
    "postprocess_completion_verified",
    "normalization_frozen",
    "frozen_decoded_nms_normalization_contract",
    "frozen_decoded_nms_normalization_contract_sha256",
    "frozen_decoded_nms_normalization_result",
    "direct_bn6_completion_projection_status",
    "completed_task_stage",
    "completed_task_contract_family",
    "completed_task_endpoint_contract",
    "completed_task_endpoint_contract_hash",
    "completed_task_output_endpoint_id",
    "completed_task_comparison_endpoint_contract",
    "completed_task_comparison_endpoint_contract_hash",
    "completed_task_comparison_output_endpoint_id",
    "completed_task_completion_mode",
    "completed_task_endpoint_attested",
    "completed_task_endpoint_attestation",
    "completed_task_endpoint_attestation_status",
    "completed_task_result_artifact_saved",
    "completed_task_result_artifact",
    "completed_task_result_artifact_sha256",
    "completed_task_result_artifact_path",
    "completed_task_result_artifact_file_sha256",
    "completed_task_result_artifact_verification_status",
    "completion_execution_contract",
    "completion_execution_contract_sha256",
    "completion_execution_attestation",
    "completed_work_units",
    "completed_frames",
    "completion_observation_relation",
    "completion_exact_result_claim_bound",
    "completion_artifact_sha256",
    "completion_schema_sha256",
    "completion_content_sha256",
    "completion_invocation_sha256",
    "completion_relation_sha256",
    "completion_sentinel_identity",
    "semantic_evidence_selection",
    "semantic_evidence_repetition_index",
    "semantic_evidence_repetition_id",
    "semantic_evidence_runtime_instance_id",
    "completed_v2_projection_conflicts",
)


def _copy_completed_v2_projection(
    target: dict[str, Any],
    source: Mapping[str, Any],
) -> None:
    """Preserve the exact sealed Completed-v2 projection for consumers.

    Values are intentionally copied without coercion.  A malformed duplicate
    must remain visible to the common fail-closed verifier instead of being
    normalized to ``None`` and silently discarded.
    """
    for field in _COMPLETED_V2_PROJECTION_FIELDS:
        if field in source:
            target[field] = copy.deepcopy(source.get(field))
    if target.get("completed_task_completion_mode") == "native_three_stage_fast_oracle_outside_timing":
        # v2.79.24 stored this identity inside the sealed execution contract but
        # omitted its scalar projection. Restore only absent values after the
        # oracle seal is verified; contradictory explicit values stay visible.
        try:
            from onnx_splitpoint_tool.native_three_stage import verify_fast_completion_attestation
            execution = target.get("completion_execution_contract") or {}
            verify_fast_completion_attestation(
                target.get("completed_task_endpoint_attestation"), execution_contract=execution,
            )
            physical = execution["source_endpoint"]
            for field, value in {
                "endpoint_contract_hash": physical["endpoint_contract_hash"],
                "accelerator_endpoint_contract_hash": physical["endpoint_contract_hash"],
                "output_endpoint_id": physical["output_endpoint_id"],
                "physical_output_endpoint_id": physical["output_endpoint_id"],
            }.items():
                if target.get(field) in (None, ""):
                    target[field] = value
        except Exception:
            pass


def _completed_frozen_nms_attestation_passed(payload: Mapping[str, Any]) -> bool:
    """Verify the completed task endpoint without relabelling its raw dump."""
    if (
        str(payload.get("completed_task_completion_mode") or "")
        in {"detection_completion_execution_v1", "native_three_stage_fast_oracle_outside_timing"}
    ):
        return _completed_execution_attestation_passed(payload)
    attestation = payload.get("completed_task_endpoint_attestation")
    if not isinstance(attestation, Mapping):
        return False
    contract = attestation.get("completed_endpoint_contract")
    if not isinstance(contract, Mapping):
        return False
    endpoint_hash = _normalize_sha256(
        attestation.get("endpoint_contract_hash")
    )
    contract_hash = _normalize_sha256(contract.get("endpoint_contract_hash"))
    frozen_hash = _normalize_sha256(
        attestation.get("frozen_postprocess_contract_sha256")
    )
    result = attestation.get("frozen_postprocess_result")
    frozen_contract = payload.get("frozen_host_postprocess_contract")
    comparison_contract = payload.get(
        "completed_task_comparison_endpoint_contract"
    )
    source_stage = str(attestation.get("source_stage") or "").strip().lower()
    if source_stage not in {"raw_head", "decoded_pre_nms"}:
        return False
    if source_stage == "decoded_pre_nms":
        try:
            verified_frozen = _verify_frozen_postprocess_contract(frozen_contract)
        except Exception:
            return False
        if verified_frozen["source_contract_family"] != source_stage:
            return False
    attested_alias = payload.get("completed_task_endpoint_attested")
    status_alias = str(
        payload.get("completed_task_endpoint_attestation_status") or ""
    ).strip().lower()
    # Older producer rows carried the complete, verified nested attestation
    # but omitted its two scalar compatibility aliases.  Missing aliases may
    # be derived from the nested object; explicit conflicts remain fail-closed.
    attested_alias_consistent = bool(
        attested_alias in (None, "") or attested_alias is True
    )
    status_alias_consistent = bool(
        not status_alias or status_alias == "passed"
    )
    try:
        completed_frames = int(attestation.get("completed_frames") or 0)
        postprocess_frames = int(
            attestation.get("postprocess_completed_frames") or 0
        )
    except (TypeError, ValueError):
        return False
    contract_identity = dict(contract)
    contract_identity.pop("endpoint_contract_complete", None)
    contract_identity.pop("endpoint_contract_hash", None)
    contract_identity.pop("output_endpoint_id", None)
    calculated_hash = hashlib.sha256(json.dumps(
        contract_identity, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()
    return bool(
        attestation.get("attested") is True
        and str(attestation.get("status") or "").strip().lower() == "passed"
        and str(attestation.get("task") or "").strip().lower() == "detection"
        and str(attestation.get("source_stage") or "").strip().lower()
        == source_stage
        and str(attestation.get("stage") or "").strip().lower()
        == "decoded_nms"
        and str(attestation.get("endpoint") or "").strip().lower()
        == "decoded_nms"
        and str(attestation.get("contract_source") or "").strip().lower()
        == "frozen_host_decode_nms_inside_measured_interval:v1"
        and endpoint_hash
        and endpoint_hash == contract_hash == calculated_hash
        and contract.get("endpoint_contract_complete") is True
        and str(contract.get("source_stage") or "").strip().lower()
        == source_stage
        and str(contract.get("completed_stage") or "").strip().lower()
        == "decoded_nms"
        and completed_frames > 0
        and postprocess_frames == completed_frames
        and attestation.get("postprocess_completion_verified") is True
        and frozen_hash
        and isinstance(result, Mapping)
        and str(result.get("contract_family") or "").strip().lower()
        == "decoded_nms"
        and _normalize_sha256(result.get("postprocess_contract_sha256"))
        == frozen_hash
        and isinstance(frozen_contract, Mapping)
        and _normalize_sha256(frozen_contract.get("contract_sha256"))
        == frozen_hash
        and isinstance(comparison_contract, Mapping)
        and dict(
            attestation.get(
                "completed_task_comparison_endpoint_contract"
            ) or {}
        ) == dict(comparison_contract)
        and _normalize_sha256(
            payload.get(
                "completed_task_comparison_endpoint_contract_hash"
            )
        ) == _normalize_sha256(
            comparison_contract.get("endpoint_contract_hash")
        )
        and str(
            payload.get(
                "completed_task_comparison_output_endpoint_id"
            ) or ""
        ) == str(comparison_contract.get("output_endpoint_id") or "")
        and str(
            attestation.get(
                "completed_task_comparison_output_endpoint_id"
            ) or ""
        ) == str(comparison_contract.get("output_endpoint_id") or "")
        and str(
            payload.get("completed_task_completion_mode") or ""
        ).strip().lower() == "frozen_host_tail"
        and attested_alias_consistent
        and status_alias_consistent
    )

from onnx_splitpoint_tool.validation.accuracy_gates import (
    AccuracyGatePolicy,
    apply_accuracy_gate_to_row,
    evaluate_detection_similarity,
)

try:
    from scripts.validate_output_dumps import load_dump, summarize  # type: ignore
except Exception:
    from validate_output_dumps import load_dump, summarize  # type: ignore
try:
    from scripts.validate_classification_output_dump import _topk, _select_logits  # type: ignore
except Exception:
    from validate_classification_output_dump import _topk, _select_logits  # type: ignore

try:
    from onnx_splitpoint_tool.validation.accuracy_gates import (
        DEFAULT_POLICY as ACCURACY_GATE_POLICY,
        apply_accuracy_gates,
        gate_counts,
    )
except Exception:  # pragma: no cover - validation must remain usable from copied scripts
    ACCURACY_GATE_POLICY = None
    def apply_accuracy_gates(row, policy=None, mutate=True):
        return row
    def gate_counts(rows):
        return {}

COCO80 = [
    'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light',
    'fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow',
    'elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee',
    'skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard',
    'tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple',
    'sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','couch',
    'potted plant','bed','dining table','toilet','tv','laptop','mouse','remote','keyboard','cell phone',
    'microwave','oven','toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear',
    'hair drier','toothbrush'
]


def _load_json(p: Path | str | None) -> Any:
    try:
        if p and Path(p).is_file():
            return json.loads(Path(p).read_text(encoding='utf-8'))
    except Exception:
        pass
    return None


def _load_strict_json_object(path: Path) -> dict[str, Any] | None:
    """Load one claim-critical JSON object and reject every duplicate key."""
    duplicate = False

    def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        nonlocal duplicate
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                duplicate = True
            value[key] = item
        return value

    try:
        parsed = json.loads(
            path.read_text(encoding='utf-8'), object_pairs_hook=_object,
        )
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) and not duplicate else None


def _sha256_file(path: Path | str | None) -> str:
    try:
        source = Path(path).expanduser() if path else None
        if not source or not source.is_file():
            return ''
        digest = hashlib.sha256()
        with source.open('rb') as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(chunk)
        return digest.hexdigest()
    except Exception:
        return ''


def _safe(s: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', str(s or '')).strip('_') or 'x'


class NativeFullBindingError(RuntimeError):
    """Raised when Native Full evidence cannot be bound unambiguously."""


def _unique_path_or_error(paths: Iterable[Path], *, context: str, strict: bool) -> Path | None:
    unique: dict[str, Path] = {}
    for path in paths:
        try:
            resolved = Path(path).expanduser().resolve()
        except Exception:
            resolved = Path(path).expanduser()
        if resolved.is_file():
            unique.setdefault(str(resolved), resolved)
    if len(unique) > 1 and strict:
        raise NativeFullBindingError(
            f'ambiguous_native_full_binding:{context}:matches={len(unique)}:'
            + '|'.join(sorted(unique))
        )
    return next(iter(unique.values()), None)


def _row_ok(v: Any) -> bool:
    return str(v).strip().lower() in {'1','true','yes','ok'} or v is True


def _explicit_bool(
    payload: Mapping[str, Any], keys: Iterable[str],
) -> bool | None:
    for key in keys:
        if key not in payload or payload.get(key) is None:
            continue
        value = payload.get(key)
        if isinstance(value, bool):
            return value
        token = str(value).strip().lower()
        if token in {'1', 'true', 'yes', 'ok', 'pass', 'passed', 'success'}:
            return True
        if token in {
            '0', 'false', 'no', 'fail', 'failed', 'error', 'unsupported',
        }:
            return False
    return None


def _task_for_model(model: str) -> str:
    m = str(model or '').lower()
    return 'detection' if m.startswith('yolo') or 'yolo' in m else 'classification'


def _producer_context_token(row: dict[str, Any]) -> str:
    """Return the producer tree token for one copied Native Full row.

    Native TensorRT Full reports have the same suffix on every accelerator
    host.  ``comparison_backend`` and ``source_root`` are therefore part of the
    evidence identity, not optional search hints.
    """
    raw = str(
        row.get('producer') or row.get('producer_backend') or
        row.get('comparison_backend') or ''
    ).strip().lower()
    aliases = {
        'hailo10': 'hailo10h', 'hailo10h': 'hailo10h', 'hailo15': 'hailo10h',
        'hailo8l': 'hailo8', 'hailo8': 'hailo8',
        'deepx_m1': 'deepx', 'dx_m1': 'deepx', 'deepx': 'deepx',
    }
    if raw in aliases:
        return aliases[raw]
    setup = str(row.get('setup_id') or '').strip().lower()
    for token, canonical in (
        ('hailo10', 'hailo10h'), ('hailo15', 'hailo10h'),
        ('hailo8', 'hailo8'), ('deepx', 'deepx'), ('dx_m1', 'deepx'),
    ):
        if token in setup:
            return canonical
    return raw


def _native_full_context_roots(row: dict[str, Any], roots: list[Path]) -> list[Path]:
    """Build identity-scoped roots for Native Full path rebasing.

    The combined native report records ``source_root`` for every row.  Prefer
    that authoritative producer root and its exact model child.  When a report
    has itself been copied, reconstruct the same scope from
    ``native_producers/<producer>/<model>``.  No candidate from another model
    or producer is admitted merely because it shares a relative suffix.
    """
    model = str(row.get('model') or '').strip()
    producer = _producer_context_token(row)
    candidates: list[Path] = []

    source_raw = str(row.get('source_root') or '').strip()
    if source_raw:
        source = Path(source_raw).expanduser()
        if source.exists():
            candidates.extend([source / model, source] if model else [source])
        else:
            # A copied summary can retain the old absolute source root.  Its
            # producer basename remains usable, but never as a global suffix
            # search without the model identity below.
            producer = producer or source.name.lower()

    for root in roots:
        root = Path(root).expanduser()
        options: list[Path] = []
        if producer and model:
            options.extend([
                root / 'native_producers' / producer / model,
                root / producer / model,
            ])
        if model:
            if root.name.lower() == producer or (
                len(root.parts) >= 2 and root.parts[-2].lower() == producer
                and root.name == model
            ):
                options.extend([root / model, root])
        if producer and root.name.lower() == producer:
            options.append(root)
        candidates.extend(options)

    unique: dict[str, Path] = {}
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except Exception:
            resolved = candidate
        if resolved.is_dir():
            unique.setdefault(str(resolved), resolved)
    return list(unique.values())


def _path_matches_native_full_context(path: Path, row: dict[str, Any]) -> bool:
    """Fail closed when a broad fallback crosses producer/model/setup scope."""
    parts = list(Path(path).parts)
    model = str(row.get('model') or '').strip()
    producer = _producer_context_token(row)
    setup = str(row.get('setup_id') or '').strip()

    if model:
        model_tokens = {model, f'model={_safe(model)}'}
        if not any(part in model_tokens for part in parts):
            return False
    if producer and 'native_producers' in parts:
        idx = parts.index('native_producers')
        if idx + 1 >= len(parts) or str(parts[idx + 1]).lower() != producer:
            return False
    encoded_setups = [part.split('=', 1)[1] for part in parts if part.startswith('setup=')]
    if encoded_setups and setup and setup not in encoded_setups:
        return False
    return True


def _find_report(row: dict[str, Any], roots: list[Path]) -> Path | None:
    strict_full = _is_native_full_row(row)

    def _matches_row(candidate: Path) -> bool:
        if strict_full:
            return _path_matches_native_full_context(candidate, row)
        # Split report suffixes become non-unique as soon as two models use the
        # same case id (for example yolo26s/b002 and yolov7_paper/b002).  Apply
        # the same model/case/backend/precision identity gate used for rebased
        # dump manifests before admitting a report candidate.
        return _manifest_matches_row(candidate, row)[0]

    rp = str(row.get('report') or row.get('result_json') or '').strip()
    if rp:
        p = Path(rp).expanduser()
        if p.is_file() and (strict_full or _matches_row(p)):
            return p
        parts = list(p.parts)
        for marker in ('native_pipeline', 'benchmark_set'):
            if marker in parts:
                suf = Path(*parts[parts.index(marker):])
                matches: list[Path] = []
                if strict_full:
                    # First perform an exact, identity-scoped rebase.  In the
                    # observed 3-producer x 2-model layout this resolves the
                    # otherwise identical native_trt_meta.json suffix without
                    # guessing or choosing the first match.
                    for context_root in _native_full_context_roots(row, roots):
                        direct = context_root / suf
                        if direct.is_file():
                            matches.append(direct)
                        matches.extend(
                            cand for cand in context_root.rglob(str(suf))
                            if cand.is_file() and _path_matches_native_full_context(cand, row)
                        )
                    hit = _unique_path_or_error(
                        matches, context=f'report_context_rebase:{suf}', strict=True,
                    )
                    if hit:
                        return hit
                    matches = []
                for r in roots:
                    matches.extend(
                        cand for cand in r.rglob(str(suf))
                        if cand.is_file() and _matches_row(cand)
                    )
                hit = _unique_path_or_error(
                    matches, context=f'report_rebase:{suf}', strict=strict_full,
                )
                if hit:
                    return hit
    backend = str(row.get('backend') or '')
    case = str(row.get('case') or row.get('case_id') or '')
    model = str(row.get('model') or '')
    patterns: list[str] = []
    precision = str(row.get('precision') or row.get('native_precision') or row.get('trt_precision') or '').strip()
    if case:
        if 'native_full' in backend:
            setup = _safe(str(row.get('setup_id') or '') or 'unspecified')
            comparison = _safe(str(row.get('comparison_backend') or '') or 'unspecified')
            patterns += [
                f'**/{model}/benchmark_set/native_full_outputs/model={_safe(model)}/backend={_safe(backend)}/setup={setup}/comparison={comparison}/native_full_*report.json',
                f'**/{model}/benchmark_set/native_full_outputs/model={_safe(model)}/backend={_safe(backend)}/setup={setup}/comparison={comparison}/native_full_semantic_dump.json',
            ]
        elif 'hailo10' in backend:
            if precision:
                patterns += [f'**/{model}/benchmark_set/native_pipeline/{case}/hailo10h_to_trt/{precision}/hailo10_native_fifo_e2e_results.json',
                             f'**/native_pipeline/{case}/hailo10h_to_trt/{precision}/hailo10_native_fifo_e2e_results.json']
            patterns += [f'**/{model}/benchmark_set/native_pipeline/{case}/hailo10h_to_trt/*/hailo10_native_fifo_e2e_results.json',
                         f'**/native_pipeline/{case}/hailo10h_to_trt/*/hailo10_native_fifo_e2e_results.json']
        elif 'deepx' in backend:
            if precision:
                patterns += [f'**/{model}/benchmark_set/native_pipeline/{case}/deepx_to_trt/{precision}/deepx_native_fifo_e2e_results.json',
                             f'**/native_pipeline/{case}/deepx_to_trt/{precision}/deepx_native_fifo_e2e_results.json']
            patterns += [f'**/{model}/benchmark_set/native_pipeline/{case}/deepx_to_trt/*/deepx_native_fifo_e2e_results.json',
                         f'**/native_pipeline/{case}/deepx_to_trt/*/deepx_native_fifo_e2e_results.json']
        else:
            if precision:
                patterns += [f'**/{model}/benchmark_set/native_pipeline/{case}/hailo_to_trt/{precision}/native_fifo_results.json',
                             f'**/native_pipeline/{case}/hailo_to_trt/{precision}/native_fifo_results.json']
            patterns += [f'**/{model}/benchmark_set/native_pipeline/{case}/hailo_to_trt/*/native_fifo_results.json',
                         f'**/native_pipeline/{case}/hailo_to_trt/*/native_fifo_results.json']
    matches = []
    for root in roots:
        for pat in patterns:
            matches.extend(
                candidate for candidate in root.glob(pat)
                if candidate.is_file() and (strict_full or _matches_row(candidate))
            )
    return _unique_path_or_error(
        matches, context=f'report_search:{model}:{backend}:{case}', strict=strict_full,
    )


def _extract_manifest_from_json_obj(
    obj: Any, roots: list[Path], *, strict_unique: bool = False,
    row: dict[str, Any] | None = None,
) -> tuple[Path | None, str]:
    """Return an explicitly recorded output-dump manifest from a JSON object.

    v59ei precision-safe rebasing: remote native reports often contain absolute
    paths from accelerator hosts.  Older rebasing kept only the suffix starting
    at native_outputs/native_fifo_outputs.  That is unsafe when multiple
    precision variants exist under the same backend/case, because both
    float32_layout_fp16 and uint8_cast_fp16 contain the same short suffix
    native_outputs/native_outputs_manifest.json.  Prefer suffixes that include
    native_pipeline/<case>/<backend>/<precision>/... and only fall back to the
    short suffix if no precise match exists.
    """
    if not isinstance(obj, dict):
        return None, 'no_json_obj'
    keys = (
        'native_fifo_output_manifest', 'output_manifest', 'outputs_manifest',
        'runner_outputs_manifest', 'native_output_manifest', 'native_outputs_manifest',
        'output_dump_manifest', 'dump_manifest', 'manifest'
    )
    for k in keys:
        v = str(obj.get(k) or '').strip()
        if not v:
            continue
        p = Path(v).expanduser()
        if p.is_file():
            return p, f'{k}_explicit'
        parts = list(p.parts)
        # Try precise suffixes first.  This keeps model/case/backend/precision.
        for marker in ('native_pipeline', 'benchmark_set'):
            if marker in parts:
                suf = Path(*parts[parts.index(marker):])
                matches: list[Path] = []
                for r in roots:
                    matches.extend(cand for cand in r.rglob(str(suf)) if cand.is_file())
                if row:
                    matches = [
                        cand for cand in matches
                        if _manifest_matches_row(cand, row)[0]
                    ]
                hit = _unique_path_or_error(
                    matches, context=f'{k}_rebased_{marker}:{suf}', strict=strict_unique,
                )
                if hit:
                    return hit, f'{k}_rebased_{marker}'
        # Last-resort compatibility fallback for old reports.
        for marker in ('native_fifo_outputs','native_outputs','runner_outputs','native_full_outputs'):
            if marker in parts:
                suf = Path(*parts[parts.index(marker):])
                matches = []
                for r in roots:
                    matches.extend(cand for cand in r.rglob(str(suf)) if cand.is_file())
                if row:
                    matches = [
                        cand for cand in matches
                        if _manifest_matches_row(cand, row)[0]
                    ]
                hit = _unique_path_or_error(
                    matches, context=f'{k}_rebased_short_{marker}:{suf}', strict=strict_unique,
                )
                if hit:
                    return hit, f'{k}_rebased_short_{marker}'
    return None, 'no_explicit_manifest'


def _is_native_full_row(row: dict[str, Any] | None) -> bool:
    if not row:
        return False
    b = str(row.get('backend') or '')
    c = str(row.get('case') or row.get('case_id') or '')
    return b.startswith('native_full_') or c == 'full' or str(row.get('execution_mode') or '') == 'native_full_baseline'


def _runtime_precision_identity(row: dict[str, Any]) -> str:
    if _is_native_full_row(row):
        value = row.get('full_runtime_precision') or row.get('execution_precision')
    else:
        value = row.get('execution_precision') or row.get('precision') or row.get('native_precision') or row.get('trt_precision')
    normalized = str(value or '').strip().lower().replace(' ', '')
    if normalized:
        return normalized
    backend = str(row.get('backend') or '').strip().lower()
    if backend == 'native_full_deepx':
        command = row.get('full_command_contract') if isinstance(row.get('full_command_contract'), dict) else {}
        artifacts = command.get('artifacts') if isinstance(command.get('artifacts'), dict) else {}
        dxnn = artifacts.get('dxnn') if isinstance(artifacts.get('dxnn'), dict) else {}
        sha = str(dxnn.get('sha256') or '').strip().lower()
        if len(sha) == 64 and all(ch in '0123456789abcdef' for ch in sha):
            return f'deepx_dxnn_sha256:{sha}'
    if backend in {'native_full_hailo8', 'native_full_hailo10h'}:
        command = row.get('full_command_contract') if isinstance(row.get('full_command_contract'), dict) else {}
        artifacts = command.get('artifacts') if isinstance(command.get('artifacts'), dict) else {}
        hef = artifacts.get('hef') if isinstance(artifacts.get('hef'), dict) else {}
        sha = str(hef.get('sha256') or '').strip().lower()
        if len(sha) == 64 and all(ch in '0123456789abcdef' for ch in sha):
            return f'hailo_hef_sha256:{sha}'
    return ''


_HAILO_TRT_SPLIT_BACKENDS = {'hailo8_to_trt', 'hailo10h_to_trt'}
_HAILO_TRT_BRIDGE_SCHEMA_BY_PRECISION = {
    'uint8_cast_fp16': 'onnx-splitpoint/uint8-cast-bridge',
    'uint8_dequant_fp16': 'onnx-splitpoint/uint8-dequant-bridge',
    'float32_layout_fp16': 'onnx-splitpoint/float32-layout-bridge',
}
_HAILO_TRT_LAYOUTS = {
    'as_input',
    'memory_nhwc_to_nchw',
    'memory_hwcn_to_nchw',
    'memory_nwhc_to_nchw',
    'memory_ncwh_to_nchw',
    'memory_chwn_to_nchw',
}


def _native_tensor_dtype(value: Any) -> str:
    token = str(value or '').strip().lower().replace(' ', '').replace('_', '')
    aliases = {
        'float': 'float32', 'fp32': 'float32', 'float32': 'float32',
        'tensor(float)': 'float32',
        'half': 'float16', 'fp16': 'float16', 'float16': 'float16',
        'tensor(float16)': 'float16',
        'uint8': 'uint8', 'tensor(uint8)': 'uint8',
        'int8': 'int8', 'tensor(int8)': 'int8',
    }
    return aliases.get(token, '')


def _static_tensor_shape(value: Any) -> list[int]:
    if not isinstance(value, (list, tuple)) or not value:
        return []
    try:
        shape = [int(dim) for dim in value]
    except (TypeError, ValueError, OverflowError):
        return []
    return shape if all(dim > 0 for dim in shape) else []


def _singleton_normalized_shape(value: Any) -> tuple[int, ...] | None:
    """Return a shape with only singleton axes removed.

    This is deliberately not an element-count normalisation.  Axis order and
    every non-singleton dimension remain part of the tensor identity.
    """
    shape = _static_tensor_shape(value)
    if not shape:
        return None
    return tuple(dim for dim in shape if dim != 1)


def _singleton_only_shape_equivalent(left: Any, right: Any) -> bool:
    """Accept only a rank difference made exclusively of singleton axes."""
    left_shape = _static_tensor_shape(left)
    right_shape = _static_tensor_shape(right)
    if not left_shape or not right_shape or left_shape == right_shape:
        return False
    return (
        _singleton_normalized_shape(left_shape)
        == _singleton_normalized_shape(right_shape)
    )


def _as_input_identity_singleton_contract_verified(
    *,
    backend: str,
    input_name: str,
    input_shape: list[int],
    input_dtype: str,
    manifest_shape: list[int],
    boundary: Mapping[str, Any],
    bridge: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> bool:
    """Verify the narrow no-layout singleton-squeeze exception.

    The producer runtime shape may omit axes of size one while TensorRT keeps
    them in its static input declaration.  That is byte-layout invariant for
    both FLOAT32 identity and elementwise UINT8-dequant bridges, but only when
    the sealed Quality-FIRST runtime, manifest, bridge and canonical Part-2
    evidence all agree and no layout transform is applied.
    """
    if not _singleton_only_shape_equivalent(manifest_shape, input_shape):
        return False

    layout = bridge.get('boundary_layout')
    layout = layout if isinstance(layout, Mapping) else {}
    replaced_uses = bridge.get('replaced_uses')
    quality_boundary = contract.get('quality_boundary_contract')
    quality_boundary = (
        quality_boundary if isinstance(quality_boundary, Mapping) else {}
    )
    boundary_transform = str(
        quality_boundary.get('boundary_transform') or ''
    ).strip().lower()
    precision = str(contract.get('precision') or '').strip().lower()
    bridge_schema = str(bridge.get('schema') or '').strip()
    identity_mode = bool(
        boundary_transform == 'identity'
        and precision == 'float32_layout_fp16'
        and input_dtype == 'float32'
        and bridge_schema in {
            '', 'onnx-splitpoint/float32-layout-bridge',
        }
        and type(replaced_uses) is int
        and replaced_uses == 0
    )
    dequant_mode = bool(
        boundary_transform == 'uint8_dequant'
        and precision == 'uint8_dequant_fp16'
        and input_dtype == 'uint8'
        and bridge_schema == 'onnx-splitpoint/uint8-dequant-bridge'
        and type(replaced_uses) is int
        and replaced_uses > 0
    )
    if not (
        (identity_mode or dequant_mode)
        and str(layout.get('requested') or '').strip().lower() == 'as_input'
        and str(layout.get('effective') or '').strip().lower() == 'as_input'
        and layout.get('applied') is False
        and str(bridge.get('input_name') or '').strip() == input_name
        and _static_tensor_shape(bridge.get('input_shape')) == input_shape
        and _native_tensor_dtype(bridge.get('input_dtype')) == input_dtype
    ):
        return False

    command_boundary = contract.get('boundary_contract')
    command_boundary = (
        command_boundary if isinstance(command_boundary, Mapping) else {}
    )
    preselection = contract.get('quality_preselection')
    preselection = preselection if isinstance(preselection, Mapping) else {}
    runtime = contract.get('runtime_boundary_evidence')
    runtime = runtime if isinstance(runtime, Mapping) else {}

    boundary_metadata_sha = _normalize_sha256(
        quality_boundary.get('boundary_metadata_sha256')
    )
    try:
        runtime_element_count = int(runtime.get('element_count'))
    except (TypeError, ValueError, OverflowError):
        runtime_element_count = -1

    return bool(
        str(command_boundary.get('boundary_layout_requested') or '')
        .strip().lower() == 'as_input'
        and str(command_boundary.get('boundary_layout_effective') or '')
        .strip().lower() == 'as_input'
        and str(quality_boundary.get('precision') or '').strip().lower()
        == str(contract.get('precision') or '').strip().lower()
        and str(quality_boundary.get('boundary_layout') or '')
        .strip().lower() == 'as_input'
        and str(quality_boundary.get('boundary_transform') or '')
        .strip().lower() == boundary_transform
        and str(quality_boundary.get('boundary_tensor_name') or '').strip()
        == input_name
        and _static_tensor_shape(
            quality_boundary.get('boundary_tensor_shape')
        ) == manifest_shape
        and _native_tensor_dtype(
            quality_boundary.get('boundary_tensor_dtype')
        ) == input_dtype
        and bool(boundary_metadata_sha)
        and str(preselection.get('boundary_layout') or '').strip().lower()
        == 'as_input'
        and str(preselection.get('boundary_transform') or '').strip().lower()
        == boundary_transform
        and str(preselection.get('boundary_tensor_name') or '').strip()
        == input_name
        and _static_tensor_shape(preselection.get('boundary_tensor_shape'))
        == manifest_shape
        and _native_tensor_dtype(preselection.get('boundary_tensor_dtype'))
        == input_dtype
        and _static_tensor_shape(preselection.get('canonical_part2_shape'))
        == input_shape
        and str(boundary.get('backend') or '').strip().lower() == backend
        and str(boundary.get('hailo_output_name') or '').strip() == input_name
        and str(boundary.get('trt_input_name') or '').strip() == input_name
        and str(boundary.get('boundary_layout') or '').strip().lower()
        == 'as_input'
        and str(boundary.get('boundary_shape_source') or '').strip().lower()
        == 'hailo_runtime_output_binding'
        and str(boundary.get('layout_transform_owner') or '').strip().lower()
        == 'tensorrt_part2_input_bridge'
        and _static_tensor_shape(boundary.get('runtime_boundary_shape'))
        == manifest_shape
        and _static_tensor_shape(boundary.get('trt_input_shape')) == input_shape
        and str(runtime.get('status') or '').strip().lower()
        == 'exact_runtime_boundary_verified'
        and str(runtime.get('runtime_name') or '').strip() == input_name
        and _static_tensor_shape(runtime.get('shape')) == manifest_shape
        and _native_tensor_dtype(runtime.get('dtype')) == input_dtype
        and runtime_element_count
        == int(np.prod(input_shape, dtype=np.int64))
        and str(runtime.get('trt_input_name') or '').strip() == input_name
        and _static_tensor_shape(runtime.get('trt_input_shape')) == input_shape
        and _normalize_sha256(
            runtime.get('binding_boundary_metadata_sha256')
        ) == boundary_metadata_sha
    )


def _integer_vector(value: Any) -> list[int] | None:
    if not isinstance(value, (list, tuple)):
        return None
    try:
        return [int(item) for item in value]
    except (TypeError, ValueError, OverflowError):
        return None


def _benchmark_set_for_report(report: Path | None) -> Path | None:
    if report is None:
        return None
    path = Path(report).expanduser()
    for candidate in (path.parent, *path.parents):
        if candidate.name == 'benchmark_set' and candidate.is_dir():
            return candidate.resolve()
    return None


def _path_has_model_benchmark_set(path: Path, model: str) -> bool:
    parts = list(path.parts)
    for index, token in enumerate(parts):
        if token == 'benchmark_set' and index > 0:
            return not model or parts[index - 1] == model
    return False


def _resolve_native_contract_artifact(
    value: Any,
    *,
    report: Path | None,
    roots: list[Path],
    model: str,
    fallback_relative: Path | None = None,
) -> tuple[Path | None, str]:
    """Resolve one copied Native artifact without crossing model boundaries."""
    raw = str(value or '').strip()
    direct = Path(raw).expanduser() if raw else None
    if direct is not None and direct.is_file():
        return direct.resolve(), 'explicit_local_path'

    benchmark_set = _benchmark_set_for_report(report)
    relative: Path | None = None
    if direct is not None and 'benchmark_set' in direct.parts:
        marker = list(direct.parts).index('benchmark_set')
        relative = Path(*list(direct.parts)[marker + 1:])
    elif direct is not None and not direct.is_absolute():
        relative = direct
    if benchmark_set is not None:
        for suffix, reason in (
            (relative, 'report_benchmark_set_rebase'),
            (fallback_relative, 'report_benchmark_set_expected_path'),
        ):
            if suffix is None:
                continue
            candidate = benchmark_set / suffix
            if candidate.is_file():
                return candidate.resolve(), reason

    suffixes = [suffix for suffix in (relative, fallback_relative) if suffix is not None]
    matches: dict[str, Path] = {}
    for root in roots:
        root = Path(root).expanduser()
        if not root.exists():
            continue
        for suffix in suffixes:
            try:
                candidates = root.rglob(str(Path('benchmark_set') / suffix))
                for candidate in candidates:
                    if candidate.is_file() and _path_has_model_benchmark_set(candidate, model):
                        resolved = candidate.resolve()
                        matches.setdefault(str(resolved), resolved)
            except Exception:
                continue
    if len(matches) == 1:
        return next(iter(matches.values())), 'unique_model_scoped_root_rebase'
    if len(matches) > 1:
        return None, 'ambiguous_model_scoped_artifact'
    return None, 'artifact_unavailable'


def _hailo_trt_interface_result(
    passed: bool | None,
    status: str,
    *,
    checks: list[str] | None = None,
    evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    verification = {
        'schema': 'onnx-splitpoint/native-hailo-trt-interface-verification',
        'schema_version': 1,
        'status': status,
        'pass': passed,
        'checks': list(checks or []),
    }
    if isinstance(evidence, Mapping):
        verification.update(dict(evidence))
    return {
        'interface_contract_pass': passed,
        'interface_check_pass': passed,
        'interface_contract_status': status,
        'interface_check_status': status,
        # This structural check does not claim strict numerical equivalence.
        'strict_boundary_numeric_pass': None,
        'strict_boundary_numeric_status': (
            'not_established_by_structural_interface_contract'
        ),
        'interface_contract_verification': verification,
    }


def _expected_hailo_layout_contract(
    layout: str, target_shape: list[int],
) -> tuple[list[int], list[int]] | None:
    if layout == 'as_input':
        return [], []
    if len(target_shape) != 4:
        return None
    n, channels, height, width = target_shape
    layouts = {
        'memory_nhwc_to_nchw': ([n, height, width, channels], [0, 3, 1, 2]),
        'memory_hwcn_to_nchw': ([height, width, channels, n], [3, 2, 0, 1]),
        'memory_nwhc_to_nchw': ([n, width, height, channels], [0, 3, 2, 1]),
        'memory_ncwh_to_nchw': ([n, channels, width, height], [0, 1, 3, 2]),
        'memory_chwn_to_nchw': ([channels, height, width, n], [3, 0, 1, 2]),
    }
    return layouts.get(layout)


def _validate_hailo_trt_interface_contract(
    row: dict[str, Any],
    native_report_payload: dict[str, Any],
    *,
    report: Path | None,
    roots: list[Path],
) -> dict[str, Any]:
    """Cryptographically bind a measured Hailo boundary to its TRT input.

    No producer-supplied pass Boolean is trusted.  The validator recomputes the
    command-contract hash, resolves the exact copied metadata/boundary files,
    and checks their complete tensor and bridge identities before admitting the
    heterogeneous interface.
    """
    backend = str(row.get('backend') or native_report_payload.get('backend') or '').strip().lower()
    if backend not in _HAILO_TRT_SPLIT_BACKENDS:
        return {}
    model = str(row.get('model') or native_report_payload.get('model') or '').strip()
    case = str(row.get('case') or row.get('case_id') or native_report_payload.get('case') or '').strip()
    precision = str(
        row.get('precision') or native_report_payload.get('precision') or ''
    ).strip().lower()
    checks: list[str] = []

    row_contract = row.get('native_command_contract')
    report_contract = native_report_payload.get('native_command_contract')
    contracts = [value for value in (row_contract, report_contract) if isinstance(value, Mapping)]
    if not contracts:
        return _hailo_trt_interface_result(
            None, 'native_command_contract_unavailable', checks=checks,
        )
    if len(contracts) > 1 and dict(contracts[0]) != dict(contracts[1]):
        return _hailo_trt_interface_result(
            False, 'native_command_contract_copies_conflict', checks=checks,
        )
    expected_identity = {
        'backend': backend,
        'model': model,
        'case': case,
        'precision': precision,
        'setup_id': str(row.get('setup_id') or native_report_payload.get('setup_id') or ''),
        'comparison_backend': str(
            row.get('comparison_backend')
            or native_report_payload.get('comparison_backend') or ''
        ),
    }
    contract, contract_status = verify_native_command_contract(
        contracts[0], expected_identity=expected_identity,
    )
    if contract is None:
        return _hailo_trt_interface_result(
            False, contract_status, checks=checks,
        )
    contract_sha, duplicate_sha_valid = _consistent_sha256(
        contract.get('contract_sha256'),
        row.get('native_command_contract_sha256'),
        native_report_payload.get('native_command_contract_sha256'),
        native_report_payload.get('workload_contract_sha256'),
        required=True,
    )
    if not duplicate_sha_valid or contract_sha != _normalize_sha256(contract.get('contract_sha256')):
        return _hailo_trt_interface_result(
            False, 'native_command_contract_duplicate_sha256_mismatch', checks=checks,
        )
    checks.append('native_command_contract_hash_schema_identity_verified')

    runtime_options = contract.get('runtime_options')
    if not isinstance(runtime_options, Mapping) or runtime_options.get('dump_boundary') is not True:
        return _hailo_trt_interface_result(
            None, 'native_boundary_dump_not_bound_by_command_contract', checks=checks,
        )
    boundary_contract = contract.get('boundary_contract')
    if not isinstance(boundary_contract, Mapping):
        return _hailo_trt_interface_result(
            None, 'native_trt_boundary_contract_unavailable', checks=checks,
        )
    metadata_raw = str(boundary_contract.get('metadata_path') or '').strip()
    metadata_sha = _normalize_sha256(boundary_contract.get('metadata_sha256'))
    if not metadata_raw or not metadata_sha:
        return _hailo_trt_interface_result(
            None, 'native_trt_metadata_binding_unavailable', checks=checks,
        )
    artifacts = contract.get('artifacts')
    artifacts = artifacts if isinstance(artifacts, Mapping) else {}
    engine_artifact = artifacts.get('engine')
    engine_artifact = engine_artifact if isinstance(engine_artifact, Mapping) else {}
    engine_raw = str(engine_artifact.get('path') or '').strip()
    if not engine_raw or Path(metadata_raw).parent != Path(engine_raw).parent:
        return _hailo_trt_interface_result(
            False, 'native_trt_metadata_engine_path_binding_mismatch', checks=checks,
        )
    metadata_path, metadata_resolution = _resolve_native_contract_artifact(
        metadata_raw, report=report, roots=roots, model=model,
        fallback_relative=(
            Path('native_trt') / case / 'part2' / precision / 'native_trt_meta.json'
        ),
    )
    metadata_evidence_path = str(metadata_path or metadata_raw)
    if metadata_path is not None:
        actual_metadata_sha = _sha256_file(metadata_path)
        if actual_metadata_sha != metadata_sha:
            return _hailo_trt_interface_result(
                False, 'native_trt_metadata_sha256_mismatch', checks=checks,
                evidence={'metadata_path': str(metadata_path)},
            )
        metadata = _load_json(metadata_path)
        if not isinstance(metadata, Mapping):
            return _hailo_trt_interface_result(
                False, 'native_trt_metadata_unreadable', checks=checks,
            )
        checks.append(
            'native_trt_metadata_file_hash_and_engine_binding_verified'
        )
    else:
        raw_bindings = [
            value for value in (
                row.get('native_split_quality_binding'),
                native_report_payload.get('native_split_quality_binding'),
            )
            if isinstance(value, Mapping)
        ]
        if not raw_bindings:
            return _hailo_trt_interface_result(
                None, f'native_trt_metadata_{metadata_resolution}', checks=checks,
            )
        if bind_quality_to_native_split is None:
            return _hailo_trt_interface_result(
                False, 'native_trt_metadata_portable_binding_verifier_unavailable',
                checks=checks,
            )
        joined_native_row = dict(native_report_payload)
        for field in (
            'backend', 'model', 'case', 'case_id', 'precision',
            'execution_precision', 'runtime_precision_identity', 'setup_id',
            'comparison_backend', 'task', 'native_command_contract',
            'native_command_contract_sha256',
            'native_split_quality_consumer_attestation',
            'native_split_quality_binding_sha256',
            'native_split_quality_eval_run_id',
            'native_split_quality_source_run_id',
            'eval_run_id', 'source_request_sha256',
            'native_split_quality_source_request_sha256',
            'native_split_quality_central_result_sha256',
            'native_split_quality_selection_sha256',
        ):
            value = row.get(field)
            if value not in (None, '', [], {}):
                existing = joined_native_row.get(field)
                if (
                    existing not in (None, '', [], {})
                    and existing != value
                ):
                    return _hailo_trt_interface_result(
                        False,
                        f'native_trt_metadata_portable_{field}_duplicate_mismatch',
                        checks=checks,
                    )
                joined_native_row[field] = value
        verified_bindings: list[dict[str, Any]] = []
        for raw_binding in raw_bindings:
            verified_binding, binding_status = bind_quality_to_native_split(
                native_row=joined_native_row,
                quality_binding=raw_binding,
                verification_mode='portable',
            )
            if verified_binding is None:
                return _hailo_trt_interface_result(
                    False,
                    f'native_trt_metadata_portable_binding_invalid:{binding_status}',
                    checks=checks,
                )
            verified_bindings.append(dict(verified_binding))
        if any(
            candidate != verified_bindings[0]
            for candidate in verified_bindings[1:]
        ):
            return _hailo_trt_interface_result(
                False, 'native_trt_metadata_portable_binding_copies_conflict',
                checks=checks,
            )
        portable_binding = verified_bindings[0]
        portable_artifacts = portable_binding.get('artifacts')
        portable_artifacts = (
            portable_artifacts if isinstance(portable_artifacts, Mapping) else {}
        )
        portable_metadata_artifact = portable_artifacts.get('native_trt_meta')
        portable_metadata_artifact = (
            portable_metadata_artifact
            if isinstance(portable_metadata_artifact, Mapping) else {}
        )
        try:
            portable_metadata_size = int(
                portable_metadata_artifact.get('size_bytes') or 0
            )
        except (TypeError, ValueError, OverflowError):
            portable_metadata_size = 0
        if (
            str(portable_metadata_artifact.get('path') or '') != metadata_raw
            or _normalize_sha256(portable_metadata_artifact.get('sha256'))
            != metadata_sha
            or _normalize_sha256(
                portable_binding.get('native_trt_meta_file_sha256')
            ) != metadata_sha
            or portable_metadata_size <= 0
            or portable_metadata_size
            != int(portable_binding.get('native_trt_meta_file_size_bytes') or 0)
        ):
            return _hailo_trt_interface_result(
                False, 'native_trt_metadata_portable_artifact_binding_mismatch',
                checks=checks,
            )
        metadata = portable_binding.get('native_trt_meta_payload')
        if not isinstance(metadata, Mapping):
            return _hailo_trt_interface_result(
                False, 'native_trt_metadata_portable_payload_missing',
                checks=checks,
            )
        metadata_resolution = 'portable_native_split_quality_binding'
        checks.append(
            'native_trt_metadata_portable_embedded_bytes_and_cross_links_verified'
        )
    if str(metadata.get('engine') or '').strip() != engine_raw:
        return _hailo_trt_interface_result(
            False, 'native_trt_metadata_engine_identity_mismatch', checks=checks,
        )
    if str(metadata.get('precision') or '').strip().lower() != precision:
        return _hailo_trt_interface_result(
            False, 'native_trt_metadata_precision_mismatch', checks=checks,
        )
    checks.append('native_trt_metadata_engine_and_precision_binding_verified')

    inputs = metadata.get('inputs')
    if not isinstance(inputs, list) or len(inputs) != 1 or not isinstance(inputs[0], Mapping):
        return _hailo_trt_interface_result(
            False, 'native_trt_exactly_one_input_required', checks=checks,
        )
    trt_input = inputs[0]
    input_name = str(trt_input.get('name') or '').strip()
    input_shape = _static_tensor_shape(trt_input.get('shape'))
    input_dtype = _native_tensor_dtype(trt_input.get('elem_type') or trt_input.get('dtype'))
    if (
        not input_name or not input_shape or not input_dtype
        or trt_input.get('has_dynamic') is True
    ):
        return _hailo_trt_interface_result(
            False, 'native_trt_input_contract_incomplete_or_dynamic', checks=checks,
        )
    dtype_bytes = {'uint8': 1, 'int8': 1, 'float16': 2, 'float32': 4}[input_dtype]
    expected_bytes = int(np.prod(input_shape, dtype=np.int64)) * dtype_bytes

    boundary_manifest_raw = (
        native_report_payload.get('native_fifo_boundary_manifest')
        or native_report_payload.get('boundary_manifest')
        or row.get('native_fifo_boundary_manifest')
        or row.get('boundary_manifest')
    )
    boundary_manifest, boundary_resolution = _resolve_native_contract_artifact(
        boundary_manifest_raw, report=report, roots=roots, model=model,
        fallback_relative=None,
    )
    if boundary_manifest is None and report is not None:
        expected_manifest = report.parent / 'native_fifo_boundary' / 'native_fifo_boundary_manifest.json'
        if expected_manifest.is_file():
            boundary_manifest = expected_manifest.resolve()
            boundary_resolution = 'report_sibling_boundary_manifest'
    if boundary_manifest is None:
        return _hailo_trt_interface_result(
            None, f'native_boundary_manifest_{boundary_resolution}', checks=checks,
        )
    boundary = _load_json(boundary_manifest)
    if not isinstance(boundary, Mapping):
        return _hailo_trt_interface_result(
            False, 'native_boundary_manifest_unreadable', checks=checks,
        )
    if (
        str(boundary.get('schema') or '') != 'onnx-splitpoint/native-boundary-dump'
        or int(boundary.get('schema_version') or 0) < 2
    ):
        return _hailo_trt_interface_result(
            False, 'native_boundary_manifest_schema_invalid', checks=checks,
        )
    manifest_shape = _static_tensor_shape(boundary.get('shape') or boundary.get('boundary_shape'))
    manifest_dtype = _native_tensor_dtype(boundary.get('dtype'))
    manifest_trt_dtype = _native_tensor_dtype(boundary.get('trt_input_dtype'))
    manifest_input_name = str(boundary.get('trt_input_name') or '').strip()
    shape_bridge = metadata.get('uint8_cast_bridge')
    shape_bridge = shape_bridge if isinstance(shape_bridge, Mapping) else {}
    shape_layout = shape_bridge.get('boundary_layout')
    shape_layout = shape_layout if isinstance(shape_layout, Mapping) else {}
    memory_shape = _static_tensor_shape(shape_layout.get('memory_shape'))
    declared_physical_shapes = [input_shape]
    if shape_layout.get('applied') is True and memory_shape:
        declared_physical_shapes.append(memory_shape)
        if len(memory_shape) > 1 and memory_shape[0] == 1:
            declared_physical_shapes.append(memory_shape[1:])
    exact_declared_shape = manifest_shape in declared_physical_shapes
    singleton_identity_shape = bool(
        not exact_declared_shape
        and _as_input_identity_singleton_contract_verified(
            backend=backend,
            input_name=input_name,
            input_shape=input_shape,
            input_dtype=input_dtype,
            manifest_shape=manifest_shape,
            boundary=boundary,
            bridge=shape_bridge,
            contract=contract,
        )
    )
    if (
        manifest_input_name != input_name
        or not (exact_declared_shape or singleton_identity_shape)
        or manifest_dtype != input_dtype
        or manifest_trt_dtype != input_dtype
    ):
        return _hailo_trt_interface_result(
            False, 'native_boundary_tensor_identity_mismatch', checks=checks,
            evidence={
                'expected_trt_input_name': input_name,
                'expected_trt_input_shape': input_shape,
                'expected_trt_input_dtype': input_dtype,
                'accepted_boundary_shapes': declared_physical_shapes,
                'observed_boundary_input_name': manifest_input_name,
                'observed_boundary_shape': manifest_shape,
                'observed_boundary_dtype': manifest_dtype,
                'observed_boundary_trt_input_dtype': manifest_trt_dtype,
                'boundary_manifest': str(boundary_manifest),
                'native_trt_metadata': metadata_evidence_path,
            },
        )
    if singleton_identity_shape:
        checks.append(
            'as_input_identity_singleton_shape_equivalence_verified'
        )
    try:
        declared_nbytes = int(boundary.get('nbytes'))
        declared_trt_bytes = int(boundary.get('trt_input_bytes'))
    except (TypeError, ValueError, OverflowError):
        return _hailo_trt_interface_result(
            False, 'native_boundary_byte_count_missing', checks=checks,
        )
    boundary_file, boundary_file_resolution = _resolve_native_contract_artifact(
        boundary.get('file'), report=report, roots=roots, model=model,
        fallback_relative=None,
    )
    if boundary_file is None:
        sibling = boundary_manifest.parent / Path(str(boundary.get('file') or '')).name
        if sibling.is_file():
            boundary_file = sibling.resolve()
            boundary_file_resolution = 'boundary_manifest_sibling_dump'
    if boundary_file is None:
        return _hailo_trt_interface_result(
            None, f'native_boundary_dump_{boundary_file_resolution}', checks=checks,
        )
    actual_bytes = int(boundary_file.stat().st_size)
    if not (
        expected_bytes > 0
        and declared_nbytes == expected_bytes
        and declared_trt_bytes == expected_bytes
        and actual_bytes == expected_bytes
    ):
        return _hailo_trt_interface_result(
            False, 'native_boundary_byte_count_mismatch', checks=checks,
        )
    try:
        report_trt_bytes = int(native_report_payload.get('trt_input_bytes'))
    except (TypeError, ValueError, OverflowError):
        report_trt_bytes = -1
    if (
        _native_tensor_dtype(native_report_payload.get('trt_input_dtype')) != input_dtype
        or report_trt_bytes != expected_bytes
    ):
        return _hailo_trt_interface_result(
            False, 'native_report_trt_input_dtype_or_bytes_mismatch', checks=checks,
        )
    report_inputs = native_report_payload.get('trt_inputs')
    if report_inputs not in (None, []) and (
        not isinstance(report_inputs, list)
        or len(report_inputs) != 1
        or str(report_inputs[0]) != input_name
    ):
        return _hailo_trt_interface_result(
            False, 'native_report_trt_input_name_mismatch', checks=checks,
        )
    checks.append('trt_input_name_shape_dtype_and_exact_bytes_verified')

    bridge = metadata.get('uint8_cast_bridge')
    if not isinstance(bridge, Mapping):
        return _hailo_trt_interface_result(
            False, 'native_trt_bridge_contract_missing', checks=checks,
        )
    expected_bridge_schema = _HAILO_TRT_BRIDGE_SCHEMA_BY_PRECISION.get(precision, '')
    bridge_schema = str(bridge.get('schema') or '').strip()
    if (
        not expected_bridge_schema
        or bridge_schema != expected_bridge_schema
        or str(boundary_contract.get('bridge_schema') or '').strip() != bridge_schema
    ):
        return _hailo_trt_interface_result(
            False, 'native_trt_bridge_schema_mismatch', checks=checks,
        )
    if (
        str(bridge.get('input_name') or '').strip() != input_name
        or _static_tensor_shape(bridge.get('input_shape')) != input_shape
        or _native_tensor_dtype(bridge.get('input_dtype')) != input_dtype
    ):
        return _hailo_trt_interface_result(
            False, 'native_trt_bridge_input_contract_mismatch', checks=checks,
        )
    layout = bridge.get('boundary_layout')
    if not isinstance(layout, Mapping):
        return _hailo_trt_interface_result(
            False, 'native_trt_bridge_layout_missing', checks=checks,
        )
    layout_requested = str(layout.get('requested') or '').strip().lower()
    layout_effective = str(layout.get('effective') or '').strip().lower()
    contract_requested = str(boundary_contract.get('boundary_layout_requested') or '').strip().lower()
    contract_effective = str(boundary_contract.get('boundary_layout_effective') or '').strip().lower()
    if (
        layout_requested not in _HAILO_TRT_LAYOUTS
        or layout_effective not in _HAILO_TRT_LAYOUTS
        or layout_requested != contract_requested
        or layout_effective != contract_effective
    ):
        return _hailo_trt_interface_result(
            False, 'native_trt_bridge_layout_binding_mismatch', checks=checks,
        )
    expected_layout = _expected_hailo_layout_contract(layout_effective, input_shape)
    if expected_layout is None:
        return _hailo_trt_interface_result(
            False, 'native_trt_bridge_layout_unsupported_for_shape', checks=checks,
        )
    expected_memory_shape, expected_perm = expected_layout
    try:
        replaced_uses = int(bridge.get('replaced_uses') or 0)
    except (TypeError, ValueError, OverflowError):
        replaced_uses = -1
    if layout_effective == 'as_input':
        layout_valid = layout.get('applied') is False
    else:
        layout_valid = bool(
            layout.get('applied') is True
            and _static_tensor_shape(layout.get('memory_shape')) == expected_memory_shape
            and _integer_vector(layout.get('perm')) == expected_perm
            and replaced_uses > 0
        )
    if not layout_valid:
        return _hailo_trt_interface_result(
            False, 'native_trt_bridge_layout_transform_mismatch', checks=checks,
        )
    report_layout = str(native_report_payload.get('boundary_layout') or '').strip().lower()
    if report_layout and report_layout != layout_effective:
        return _hailo_trt_interface_result(
            False, 'native_report_boundary_layout_mismatch', checks=checks,
        )
    if precision == 'uint8_dequant_fp16':
        try:
            bridge_scale = float(bridge.get('scale'))
            bridge_zero_point = float(bridge.get('zero_point'))
            contract_scale = float(boundary_contract.get('dequant_scale'))
            contract_zero_point = float(boundary_contract.get('dequant_zero_point'))
        except (TypeError, ValueError, OverflowError):
            return _hailo_trt_interface_result(
                False, 'native_trt_dequant_parameters_missing', checks=checks,
            )
        if not (
            math.isfinite(bridge_scale) and math.isfinite(bridge_zero_point)
            and math.isclose(bridge_scale, contract_scale, rel_tol=0.0, abs_tol=1e-12)
            and math.isclose(bridge_zero_point, contract_zero_point, rel_tol=0.0, abs_tol=1e-12)
        ):
            return _hailo_trt_interface_result(
                False, 'native_trt_dequant_parameter_binding_mismatch', checks=checks,
            )
    checks.append('bridge_schema_layout_and_parameters_verified')

    return _hailo_trt_interface_result(
        True, 'verified_native_command_metadata_boundary_and_bridge',
        checks=checks,
        evidence={
            'native_command_contract_sha256': contract_sha,
            'metadata_path': metadata_evidence_path,
            'metadata_sha256': metadata_sha,
            'metadata_resolution': metadata_resolution,
            'boundary_manifest': str(boundary_manifest),
            'boundary_manifest_resolution': boundary_resolution,
            'boundary_dump': str(boundary_file),
            'boundary_dump_sha256': _sha256_file(boundary_file),
            'boundary_dump_size_bytes': actual_bytes,
            'trt_input_name': input_name,
            'trt_input_shape': input_shape,
            'trt_input_dtype': input_dtype,
            'trt_input_bytes': expected_bytes,
            'bridge_schema': bridge_schema,
            'boundary_layout_requested': layout_requested,
            'boundary_layout_effective': layout_effective,
        },
    )


def _quality_run_token(value: Any) -> str:
    token = str(value or '').strip().lower().replace('-', '_')
    for prefix in ('benchmark_results_', 'results_'):
        if token.startswith(prefix):
            token = token[len(prefix):]
    if token.endswith('_auto'):
        token = token[:-5]
    token = token.replace('_to_tensorrt', '_to_trt')
    return {
        'hailo10_to_trt': 'hailo10h_to_trt',
        'deepx_m1_to_trt': 'deepx_to_trt',
    }.get(token, token)


def _expected_central_quality_runs(row: dict[str, Any]) -> set[str]:
    backend = _quality_run_token(row.get('backend'))
    aliases = {
        'hailo8_to_trt': {'hailo8_to_trt'},
        'hailo10h_to_trt': {'hailo10_to_trt', 'hailo10h_to_trt'},
        'deepx_to_trt': {'deepx_m1_to_trt', 'deepx_to_trt'},
        # Quality-FIRST Full TRT is a setup-local companion.  It is not the
        # generic ORT/TRT profile and may never borrow that profile's quality.
        'native_full_tensorrt': {'native_full_tensorrt'},
        'native_full_deepx': {'deepx_m1_full', 'deepx_full'},
        'native_full_hailo8': {'hailo8'},
        'native_full_hailo10h': {'hailo10', 'hailo10h'},
    }
    return {_quality_run_token(value) for value in aliases.get(backend, {backend})}


def _native_quality_case_ids(row: dict[str, Any]) -> set[str]:
    values = {str(row.get('case') or row.get('case_id') or '').strip().lower()}
    command = row.get('full_command_contract') if isinstance(row.get('full_command_contract'), dict) else {}
    values.add(str(command.get('input_case') or '').strip().lower())
    return {value for value in values if value}


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
    ).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def _central_result_mirror_signature(
    result: Mapping[str, Any], *, trt_quality_first: bool,
) -> str:
    """Hash one logical result while ignoring mirror-local diagnostics."""
    if not trt_quality_first:
        return _canonical_json_sha256(result)
    request_identity = (
        result.get('request_identity')
        if isinstance(result.get('request_identity'), Mapping)
        else {}
    )
    return _canonical_json_sha256({
        'model_id': result.get('model_id'),
        'case_id': result.get('case_id'),
        'variant': result.get('variant'),
        'task': result.get('task'),
        'source_setup_id': (
            result.get('source_setup_id')
            or result.get('setup_id')
            or request_identity.get('setup_id')
        ),
        'source_run_id': (
            result.get('source_run_id')
            or request_identity.get('source_run_id')
        ),
        'source_request_sha256': (
            result.get('source_request_sha256')
            or request_identity.get('source_request_sha256')
        ),
        'status': result.get('status'),
        'technical_status': result.get('technical_status'),
        'scientific_status': result.get('scientific_status'),
        'decision': result.get('decision'),
        'evaluation_fingerprint': result.get('evaluation_fingerprint'),
        'reference_identity': result.get('reference_identity'),
        'n': result.get('n'),
        'primary': result.get('primary'),
        'guardrails': result.get('guardrails'),
        'producer_binding_eligible': (
            result.get('producer_binding_eligible')
            if isinstance(result.get('producer_binding_eligible'), bool)
            else request_identity.get('producer_binding_eligible')
        ),
    })


def _validated_trt_quality_producer(
    value: Any, *, task: str, role: str,
) -> tuple[dict[str, Any], str, list[str]]:
    """Validate one signed Quality-FIRST producer with the central authority.

    The quality service validates the complete model/dataset/preprocessing,
    endpoint-authority, precision, policy and build-receipt graph.  The native
    validator adds the two execution-role constraints which distinguish the
    central quality companion from a performance result, plus the byte digest
    of the persisted receipt (which is deliberately distinct from the
    receipt's canonical object digest).
    """
    if _validate_quality_producer_contract is None:
        return {}, '', [f'{role}:quality_producer_validator_unavailable']
    if not isinstance(value, Mapping):
        return {}, '', [f'{role}:quality_producer_missing']
    producer = dict(value)
    try:
        validated, validated_sha = _validate_quality_producer_contract(
            producer, role=role, task=str(task or '').strip().lower(),
        )
    except Exception as exc:
        token = re.sub(
            r'[^a-z0-9_]+', '_', f'{type(exc).__name__}_{exc}'.lower(),
        ).strip('_')
        return {}, '', [f'{role}:quality_producer_invalid:{token[:240]}']
    validated = dict(validated)
    producer_sha = _normalize_sha256(validated_sha)
    file_sha = _normalize_sha256(
        validated.get('engine_build_receipt_file_sha256')
    )
    errors: list[str] = []
    if not producer_sha:
        errors.append(f'{role}:producer_identity_sha256_invalid')
    if (
        validated.get('schema') != _TRT_QUALITY_PRODUCER_SCHEMA
        or validated.get('execution_role') != 'full_quality_only'
        or validated.get('backend') != 'native_tensorrt'
        or validated.get('variant') != 'full'
        or validated.get('case_id') != 'full'
        or validated.get('source_run_id') != 'native_full_tensorrt'
        or validated.get('performance_claims_emitted') is not False
    ):
        errors.append(f'{role}:quality_only_scope_invalid')
    if not file_sha:
        errors.append(f'{role}:engine_build_receipt_file_sha256_invalid')
    return (validated if not errors else {}), (producer_sha if not errors else ''), errors


def _artifact_binding_matches(
    observed: Any, expected: Any, *, expected_sha: str | None = None,
) -> bool:
    if not isinstance(observed, Mapping) or not isinstance(expected, Mapping):
        return False
    try:
        observed_size = int(observed.get('size_bytes'))
        expected_size = int(expected.get('size_bytes'))
    except (TypeError, ValueError):
        return False
    observed_sha = _normalize_sha256(observed.get('sha256'))
    canonical_expected_sha = _normalize_sha256(
        expected_sha if expected_sha is not None else expected.get('sha256')
    )
    return bool(
        str(observed.get('path') or '').strip()
        == str(expected.get('path') or '').strip()
        and observed_sha and observed_sha == canonical_expected_sha
        and observed_size > 0 and observed_size == expected_size
    )


def _native_trt_quality_first_binding(
    row: Mapping[str, Any], *, task: str,
) -> tuple[dict[str, Any], str, list[str]]:
    """Verify the sealed Native-Full contract against its quality producer."""
    errors: list[str] = []
    raw_contract = row.get('full_command_contract')
    if not isinstance(raw_contract, Mapping):
        return {}, '', ['native:full_command_contract_missing']
    contract = dict(raw_contract)
    declared_contract_sha = _normalize_sha256(contract.get('contract_sha256'))
    unhashed_contract = dict(contract)
    unhashed_contract.pop('contract_sha256', None)
    if not declared_contract_sha or _canonical_json_sha256(unhashed_contract) != declared_contract_sha:
        errors.append('native:full_command_contract_sha256_mismatch')
    duplicate_contract_sha, duplicate_contract_valid = _consistent_sha256(
        row.get('full_command_contract_sha256'), declared_contract_sha,
    )
    if (
        not _normalize_sha256(row.get('full_command_contract_sha256'))
        or not duplicate_contract_valid
        or duplicate_contract_sha != declared_contract_sha
    ):
        errors.append('native:full_command_contract_duplicate_mismatch')

    producer, producer_sha, producer_errors = _validated_trt_quality_producer(
        contract.get('quality_first_producer_identity'),
        task=task, role='native full command contract',
    )
    errors.extend(producer_errors)
    declared_producer_sha, producer_duplicate_valid = _consistent_sha256(
        contract.get('quality_first_producer_identity_sha256'),
        row.get('quality_first_producer_identity_sha256'),
        producer_sha,
    )
    if not producer_duplicate_valid or declared_producer_sha != producer_sha:
        errors.append('native:quality_first_producer_sha256_mismatch')
    if row.get('quality_first_semantic_dump_binding_valid') is not True:
        errors.append('native:semantic_dump_producer_binding_missing_or_conflicting')
    direct_producer = row.get('quality_first_producer_identity')
    if direct_producer not in (None, {}) and direct_producer != producer:
        errors.append('native:quality_first_producer_duplicate_mismatch')

    if producer:
        expected_identity = {
            'backend': row.get('backend'),
            'model': row.get('model') or row.get('model_id'),
            'case': row.get('case') or row.get('case_id') or 'full',
            'setup_id': row.get('setup_id'),
            'comparison_backend': row.get('comparison_backend'),
            'model_sha256': (producer.get('source_onnx') or {}).get('sha256'),
            'source_model_sha256': (producer.get('source_onnx') or {}).get('sha256'),
        }
        if _strict_verify_full_command_contract is None:
            errors.append('native:full_command_contract_strict_verifier_unavailable')
        else:
            verified, status = _strict_verify_full_command_contract(
                contract, expected_identity=expected_identity,
            )
            if verified is None:
                errors.append(f'native:full_command_contract_invalid:{status}')

        row_model = str(row.get('model') or row.get('model_id') or '').strip().lower()
        row_setup = str(row.get('setup_id') or '').strip().lower()
        row_case = str(row.get('case') or row.get('case_id') or '').strip().lower()
        runtime_precision = str(
            row.get('runtime_precision_identity')
            or row.get('full_runtime_precision')
            or row.get('execution_precision') or ''
        ).strip().lower().replace(' ', '')
        if (
            row_model != str(producer.get('model_id') or '').strip().lower()
            or row_setup != str(producer.get('setup_id') or '').strip().lower()
            or row_case != 'full'
            or str(task or '').strip().lower() != str(producer.get('task') or '')
            or runtime_precision != str(producer.get('runtime_precision_identity') or '')
        ):
            errors.append('native:producer_scope_or_precision_mismatch')
        endpoint_sha, endpoint_valid = _consistent_sha256(
            row.get('endpoint_contract_hash'), producer.get('endpoint_contract_hash'),
        )
        if not endpoint_valid or not endpoint_sha:
            errors.append('native:producer_endpoint_mismatch')

        artifacts = contract.get('artifacts') if isinstance(contract.get('artifacts'), Mapping) else {}
        workload = contract.get('energy_workload') if isinstance(contract.get('energy_workload'), Mapping) else {}
        source_artifact = artifacts.get(str(workload.get('source_model_artifact') or ''))
        engine_artifact = artifacts.get(str(workload.get('engine_artifact') or ''))
        trtexec_artifact = artifacts.get(str(workload.get('trtexec_artifact') or ''))
        receipt_artifact = artifacts.get(str(workload.get('engine_build_receipt_artifact') or ''))
        receipt_binding = producer.get('engine_build_receipt') if isinstance(producer.get('engine_build_receipt'), Mapping) else {}
        if not _artifact_binding_matches(source_artifact, producer.get('build_onnx')):
            errors.append('native:build_onnx_artifact_mismatch')
        if not _artifact_binding_matches(engine_artifact, producer.get('engine')):
            errors.append('native:engine_artifact_mismatch')
        if not _artifact_binding_matches(trtexec_artifact, producer.get('trtexec')):
            errors.append('native:trtexec_artifact_mismatch')
        receipt_expected = dict(receipt_binding)
        receipt_expected['sha256'] = producer.get('engine_build_receipt_file_sha256')
        if not _artifact_binding_matches(
            receipt_artifact, receipt_expected,
            expected_sha=str(producer.get('engine_build_receipt_file_sha256') or ''),
        ):
            errors.append('native:engine_build_receipt_file_artifact_mismatch')
        if dict(contract.get('trt_engine_build_receipt') or {}) != dict(
            receipt_binding.get('receipt') or {}
        ):
            errors.append('native:engine_build_receipt_content_mismatch')
        receipt = receipt_binding.get('receipt') if isinstance(receipt_binding.get('receipt'), Mapping) else {}
        receipt_duplicates = (
            ('engine_build_receipt_path', str(contract.get('engine_build_receipt_path') or '').strip(), str(receipt_binding.get('path') or '').strip()),
            ('engine_build_receipt_sha256', _normalize_sha256(contract.get('engine_build_receipt_sha256')), _normalize_sha256(receipt_binding.get('sha256'))),
            ('engine_build_receipt_file_sha256', _normalize_sha256(contract.get('engine_build_receipt_file_sha256')), _normalize_sha256(producer.get('engine_build_receipt_file_sha256'))),
            ('trt_engine_build_receipt_sha256', _normalize_sha256(contract.get('trt_engine_build_receipt_sha256')), _normalize_sha256(receipt.get('receipt_sha256'))),
        )
        for field_name, observed, expected in receipt_duplicates:
            if not observed or observed != expected:
                errors.append(f'native:{field_name}_mismatch')
        try:
            receipt_size_match = (
                int(contract.get('engine_build_receipt_size_bytes'))
                == int(receipt_binding.get('size_bytes')) > 0
            )
        except (TypeError, ValueError):
            receipt_size_match = False
        if not receipt_size_match:
            errors.append('native:engine_build_receipt_size_bytes_mismatch')
        try:
            receipt_file_size_match = (
                int(contract.get('engine_build_receipt_file_size_bytes'))
                == int((receipt_artifact or {}).get('file_size_bytes')) > 0
            )
        except (TypeError, ValueError):
            receipt_file_size_match = False
        if not receipt_file_size_match:
            errors.append('native:engine_build_receipt_file_size_bytes_mismatch')
        if _normalize_sha256(contract.get('source_model_sha256')) != _normalize_sha256(
            (producer.get('source_onnx') or {}).get('sha256')
        ):
            errors.append('native:source_model_sha256_mismatch')

    return (producer if not errors else {}), (producer_sha if not errors else ''), list(dict.fromkeys(errors))


def _central_trt_quality_producer(
    result: Mapping[str, Any], *, task: str,
) -> tuple[dict[str, Any], str, list[str]]:
    """Validate both central copies and their completed summary-only scope."""
    errors: list[str] = []
    identity = result.get('request_identity') if isinstance(result.get('request_identity'), Mapping) else {}
    result_producer = result.get('producer_identity')
    nested_producer = identity.get('producer_identity') if isinstance(identity, Mapping) else None
    if result_producer != nested_producer:
        errors.append('central:producer_identity_copies_conflict')
    producer, producer_sha, producer_errors = _validated_trt_quality_producer(
        result_producer, task=task, role='central quality result',
    )
    errors.extend(producer_errors)
    if (
        int(identity.get('schema_version') or 0) != 4
        or identity.get('identity_valid') is not True
        or identity.get('producer_identity_validated') is not True
        or identity.get('producer_binding_eligible') is not True
        or result.get('producer_binding_eligible') is not True
        or str(result.get('status') or '') != 'completed'
        or str(result.get('technical_status') or 'completed') != 'completed'
    ):
        errors.append('central:summary_only_result_not_eligible')
    source = producer.get('source_onnx') if isinstance(producer.get('source_onnx'), Mapping) else {}
    build = producer.get('build_onnx') if isinstance(producer.get('build_onnx'), Mapping) else {}
    engine = producer.get('engine') if isinstance(producer.get('engine'), Mapping) else {}
    trtexec = producer.get('trtexec') if isinstance(producer.get('trtexec'), Mapping) else {}
    receipt_binding = producer.get('engine_build_receipt') if isinstance(producer.get('engine_build_receipt'), Mapping) else {}
    receipt = receipt_binding.get('receipt') if isinstance(receipt_binding.get('receipt'), Mapping) else {}
    inner_receipt = dict(receipt); inner_receipt.pop('receipt_sha256', None)
    expected_flat: dict[str, Any] = {
        'producer_identity_sha256': producer_sha,
        'eval_run_id': producer.get('eval_run_id'),
        'model_id': producer.get('model_id'),
        'setup_id': producer.get('setup_id'),
        'source_run_id': producer.get('source_run_id'),
        'originating_plan_run_id': producer.get('originating_plan_run_id'),
        'case_id': producer.get('case_id'),
        'execution_role': producer.get('execution_role'),
        'backend': producer.get('backend'),
        'variant': producer.get('variant'),
        'task': producer.get('task'),
        'performance_claims_emitted': producer.get('performance_claims_emitted'),
        'source_onnx_path': source.get('path'),
        'source_onnx_sha256': source.get('sha256'),
        'source_onnx_size_bytes': source.get('size_bytes'),
        'source_model_sha256': source.get('sha256'),
        'source_model_size_bytes': source.get('size_bytes'),
        'build_onnx_path': build.get('path'),
        'build_onnx_sha256': build.get('sha256'),
        'build_onnx_size_bytes': build.get('size_bytes'),
        'engine_path': engine.get('path'),
        'engine_sha256': engine.get('sha256'),
        'engine_size_bytes': engine.get('size_bytes'),
        'runtime_artifact_sha256': engine.get('sha256'),
        'runtime_artifact_size_bytes': engine.get('size_bytes'),
        'trtexec_path': trtexec.get('path'),
        'trtexec_sha256': trtexec.get('sha256'),
        'trtexec_size_bytes': trtexec.get('size_bytes'),
        'engine_build_receipt_path': receipt_binding.get('path'),
        'engine_build_receipt_sha256': receipt_binding.get('sha256'),
        'engine_build_receipt_file_sha256': producer.get(
            'engine_build_receipt_file_sha256'
        ),
        'engine_build_receipt_size_bytes': receipt_binding.get('size_bytes'),
        'trt_engine_build_receipt_sha256': receipt.get('receipt_sha256'),
        'trt_engine_build_receipt_size_bytes': len(json.dumps(
            inner_receipt, sort_keys=True, separators=(',', ':'),
            ensure_ascii=False,
        ).encode('utf-8')),
    }
    hash_fields = {
        name for name in expected_flat
        if name.endswith('_sha256')
    }
    size_fields = {
        name for name in expected_flat
        if name.endswith('_size_bytes')
    }
    for container_name, container in (('result', result), ('request_identity', identity)):
        for field_name, expected in expected_flat.items():
            if field_name not in container:
                errors.append(f'central:{container_name}_{field_name}_missing')
                continue
            observed = container.get(field_name)
            if field_name in hash_fields:
                equal = bool(
                    _normalize_sha256(observed)
                    and _normalize_sha256(observed) == _normalize_sha256(expected)
                )
            elif field_name in size_fields:
                try:
                    equal = int(observed) == int(expected) and int(expected) > 0
                except (TypeError, ValueError):
                    equal = False
            elif field_name == 'performance_claims_emitted':
                equal = isinstance(observed, bool) and observed is expected
            elif field_name == 'backend':
                observed_backend = str(observed or '').strip().lower()
                expected_backend = str(expected or '').strip().lower()
                # Central Quality exposes the logical backend as ``tensorrt``
                # while the signed execution producer remains
                # ``native_tensorrt``.  Accept only that narrow mirror alias;
                # source_run_id and every signed producer field stay exact.
                equal = bool(
                    observed_backend == expected_backend
                    or (
                        observed_backend == 'tensorrt'
                        and expected_backend == 'native_tensorrt'
                    )
                )
                if 'producer_backend' in container:
                    declared_producer_backend = container.get(
                        'producer_backend'
                    )
                    equal = bool(
                        equal
                        and str(declared_producer_backend).strip().lower()
                        == expected_backend
                    )
            else:
                equal = str(observed or '').strip() == str(expected or '').strip()
            if not equal:
                errors.append(f'central:{container_name}_{field_name}_mismatch')
    return (producer if not errors else {}), (producer_sha if not errors else ''), list(dict.fromkeys(errors))


def _native_trt_completed_quality_projection(
    producer: Mapping[str, Any], comparison: Mapping[str, Any],
    *, physical_endpoint_hash: str,
) -> tuple[bool, str]:
    """Prove that physical TRT Quality records implement completed V2.

    Central Quality is transported under the physical producer endpoint.  A
    detection decision can be projected to the completed decoded-NMS endpoint
    only when its sealed preprocessing/decoder/NMS contract is semantically
    identical to the already verified completed comparison contract.
    """
    quality = (
        producer.get('quality_contract')
        if isinstance(producer.get('quality_contract'), Mapping) else {}
    )
    model = quality.get('model') if isinstance(quality.get('model'), Mapping) else {}
    source = (
        producer.get('source_onnx')
        if isinstance(producer.get('source_onnx'), Mapping) else {}
    )
    preprocessing = (
        quality.get('preprocessing')
        if isinstance(quality.get('preprocessing'), Mapping) else {}
    )
    preprocess_identity = (
        preprocessing.get('identity')
        if isinstance(preprocessing.get('identity'), Mapping) else {}
    )
    decoder = quality.get('decoder') if isinstance(quality.get('decoder'), Mapping) else {}
    decoder_identity = (
        decoder.get('identity')
        if isinstance(decoder.get('identity'), Mapping) else {}
    )
    nms = quality.get('nms') if isinstance(quality.get('nms'), Mapping) else {}
    nms_identity = (
        nms.get('identity')
        if isinstance(nms.get('identity'), Mapping) else {}
    )
    quality_record = (
        producer.get('quality_record_endpoint')
        if isinstance(producer.get('quality_record_endpoint'), Mapping) else {}
    )
    quality_record_identity = (
        quality_record.get('identity')
        if isinstance(quality_record.get('identity'), Mapping) else {}
    )
    producer_endpoint = (
        producer.get('endpoint')
        if isinstance(producer.get('endpoint'), Mapping) else {}
    )
    producer_endpoint_identity = (
        producer_endpoint.get('identity')
        if isinstance(producer_endpoint.get('identity'), Mapping) else {}
    )
    source_format = str(
        decoder_identity.get('source_output_format') or ''
    ).strip().lower()
    raw_multiscale = source_format == 'multiscale_head'
    decoded_pre_nms = source_format == 'ultralytics_decoded'
    semantic_score = (
        decoder_identity.get('confidence_threshold')
        if raw_multiscale else
        nms_identity.get('detr_or_bn6_confidence_threshold')
    )
    semantic_iou = nms_identity.get(
        'iou_threshold' if raw_multiscale
        else 'detr_or_bn6_iou_threshold'
    )
    semantic_max = nms_identity.get(
        'max_detections' if raw_multiscale
        else 'detr_or_bn6_max_detections'
    )
    try:
        thresholds_match = bool(
            math.isclose(
                float(semantic_score),
                float(comparison.get('score_threshold')),
                rel_tol=0.0, abs_tol=1e-12,
            )
            and math.isclose(
                float(semantic_iou),
                float(comparison.get('iou_threshold')),
                rel_tol=0.0, abs_tol=1e-12,
            )
            and not isinstance(semantic_max, bool)
            and int(semantic_max) == int(comparison.get('max_detections'))
        )
    except (TypeError, ValueError, OverflowError):
        thresholds_match = False
    if (
        _normalize_sha256(producer.get('endpoint_contract_hash'))
        != _normalize_sha256(physical_endpoint_hash)
        or str(producer.get('task') or '').strip().lower() != 'detection'
        or str(producer.get('model_id') or '').strip().lower()
        != str(comparison.get('model_id') or '').strip().lower()
        or _normalize_sha256(model.get('sha256'))
        != _normalize_sha256(source.get('sha256'))
        or quality.get('task') != 'detection'
        or quality.get('canonical_record_endpoint')
        != 'decoded_xyxy_score_class_detections'
        or decoder_identity.get('canonical_record_endpoint')
        != 'decoded_xyxy_score_class_detections'
        or quality_record_identity.get('canonical_record_endpoint')
        != 'decoded_xyxy_score_class_detections'
        or _normalize_sha256(
            quality_record_identity.get('decoder_contract_sha256')
        ) != _normalize_sha256(producer.get('decoder_contract_sha256'))
        or _normalize_sha256(producer.get('decoder_contract_sha256'))
        != _normalize_sha256(decoder.get('sha256'))
        or _normalize_sha256(
            quality_record_identity.get('nms_contract_sha256')
        ) != _normalize_sha256(producer.get('nms_contract_sha256'))
        or _normalize_sha256(producer.get('nms_contract_sha256'))
        != _normalize_sha256(nms.get('sha256'))
        or list(preprocess_identity.get('target_hw') or [])
        != list(comparison.get('input_hw') or [])
        or source_format not in {'multiscale_head', 'ultralytics_decoded', 'bn6_detections'}
        or quality.get('source_endpoint_is_raw') is not raw_multiscale
        or (
            raw_multiscale
            and (
                producer_endpoint_identity.get('stage') != 'raw_head'
                or decoder_identity.get('source_endpoint_semantics')
                != 'raw_multiscale_head'
                or decoder_identity.get('source_endpoint_has_integrated_nms')
                is not False
            )
        )
        or (
            decoded_pre_nms
            and (
                producer_endpoint_identity.get('stage') != 'decoded_pre_nms'
                or decoder_identity.get('source_endpoint_semantics') != 'ultralytics_decoded'
                or decoder_identity.get('source_endpoint_has_integrated_nms') is not False
            )
        )
        or (
            not raw_multiscale and not decoded_pre_nms
            and (
                producer_endpoint_identity.get('stage') != 'decoded_nms'
                or decoder_identity.get('source_endpoint_semantics')
                != 'bn6_detections'
            )
        )
        or comparison.get('task') != 'detection'
        or comparison.get('stage') != 'decoded_nms'
        or comparison.get('contract_family') != 'decoded_nms'
        or comparison.get('class_aware') is not True
        or comparison.get('canonical_completion_policy_id')
        != 'decoded_nms_xyxy_original_classaware_postfilter_v2'
        or comparison.get('nms_semantics_id')
        != 'class_aware_nms_xyxy_v1'
        or not thresholds_match
    ):
        return False, 'native_trt_completed_quality_semantics_mismatch'
    return True, 'native_trt_physical_quality_semantically_projects_to_completed_v2'


def _optional_central_completed_projection_matches(
    result: Mapping[str, Any], identity: Mapping[str, Any],
    comparison: Mapping[str, Any],
) -> bool:
    """Accept absent legacy duplicates, but reject every partial/conflicting copy."""
    fields = (
        'completed_task_endpoint_contract',
        'completed_task_endpoint_contract_hash',
        'completed_task_output_endpoint_id',
        'quality_join_endpoint',
    )
    declared = any(
        container.get(field) not in (None, '', {}, [])
        for container in (result, identity)
        for field in fields
    )
    if not declared:
        return True
    expected = {
        'completed_task_endpoint_contract': dict(comparison),
        'completed_task_endpoint_contract_hash': str(
            comparison.get('endpoint_contract_hash') or ''
        ),
        'completed_task_output_endpoint_id': str(
            comparison.get('output_endpoint_id') or ''
        ),
        'quality_join_endpoint': 'completed_task_decoded_nms',
    }
    return all(
        all(container.get(field) == value for field, value in expected.items())
        for container in (result, identity)
    )


def _split_quality_first_is_explicit(row: Mapping[str, Any]) -> bool:
    """Distinguish new Quality-FIRST rows from compatible legacy split rows."""
    containers: list[Mapping[str, Any]] = [row]
    for field in ('native_command_contract', 'native_report'):
        nested = row.get(field)
        if isinstance(nested, Mapping):
            containers.append(nested)
    for container in containers:
        if container.get('native_split_quality_binding_required') is True:
            return True
        binding = container.get('native_split_quality_binding')
        if isinstance(binding, Mapping) and bool(binding):
            return True
        attestation = container.get('native_split_quality_consumer_attestation')
        if isinstance(attestation, Mapping) and bool(attestation):
            return True
        if any(
            str(container.get(field) or '').strip()
            for field in (
                'native_split_quality_binding_sha256',
                'native_split_quality_eval_run_id',
                'native_split_quality_source_run_id',
            )
        ):
            return True
    return False


def _explicit_historical_central_schema(
    central_results: Iterable[Mapping[str, Any]],
) -> bool:
    """Recognize only the pre-QF Central request schemas for API compatibility.

    Production invocations always supply run/stage authority.  This fallback
    exists for explicitly versioned historical unit/API consumers, and never
    treats a missing schema or a missing marker as legacy evidence.
    """
    versions: list[int] = []
    for result in central_results:
        identity = result.get('request_identity')
        if not isinstance(identity, Mapping):
            return False
        version = identity.get('schema_version')
        if isinstance(version, bool) or not isinstance(version, int):
            return False
        if version not in {1, 2, 3, 4}:
            return False
        if _split_quality_first_is_explicit(result) or _split_quality_first_is_explicit(identity):
            return False
        versions.append(version)
    return bool(versions)


def _strict_split_central_result_identity(
    result: Mapping[str, Any], identity: Mapping[str, Any],
) -> tuple[dict[str, str] | None, str]:
    """Require current split result/identity mirrors without fallbacks."""

    fields = (
        ('eval_run_id', 'eval_run_id', lambda value: str(value or '').strip()),
        ('model_id', 'model_id', lambda value: str(value or '').strip().lower()),
        ('case_id', 'case_id', lambda value: str(value or '').strip().lower()),
        ('source_run_id', 'source_run_id', _quality_run_token),
        ('source_setup_id', 'setup_id', lambda value: str(value or '').strip()),
        ('task', 'task', lambda value: str(value or '').strip().lower()),
        ('variant', 'variant', lambda value: str(value or '').strip().lower()),
        (
            'runtime_precision_identity', 'runtime_precision_identity',
            lambda value: str(value or '').strip().lower().replace(' ', ''),
        ),
    )
    exact: dict[str, str] = {}
    for result_field, identity_field, normalizer in fields:
        left = normalizer(result.get(result_field))
        right = normalizer(identity.get(identity_field))
        canonical = 'setup_id' if result_field == 'source_setup_id' else result_field
        if not left or not right:
            return None, f'central_native_split_{canonical}_mirror_missing'
        if left != right:
            return None, f'central_native_split_{canonical}_mirror_mismatch'
        exact[canonical] = left
    optional_aliases = (
        (result, 'setup_id', exact['setup_id'], lambda value: str(value or '').strip()),
        (result, 'run_id', exact['source_run_id'], _quality_run_token),
        (
            result, 'backend', exact['source_run_id'],
            canonical_native_split_backend or _quality_run_token,
        ),
        (
            identity, 'backend', exact['source_run_id'],
            canonical_native_split_backend or _quality_run_token,
        ),
    )
    for container, field, expected, normalizer in optional_aliases:
        if field not in container:
            continue
        if normalizer(container.get(field)) != normalizer(expected):
            return None, f'central_native_split_{field}_alias_mismatch'
    return exact, 'exact_current_split_identity_mirrors'


def _bind_central_quality_evidence(
    row: dict[str, Any], central_results: list[dict[str, Any]],
    policy: AccuracyGatePolicy,
    *, split_quality_authority: Mapping[str, Any] | None = None,
) -> None:
    """Bind one central decision to one Native precision row, or fail closed."""
    row['central_quality_evidence_verified'] = False
    row['precision_quality_verified'] = False
    row['precision_quality_binding_verified'] = False
    strict_trt_quality_first = (
        str(row.get('backend') or '').strip().lower() == 'native_full_tensorrt'
    )
    backend_token = str(row.get('backend') or '').strip().lower()
    strict_vendor_full_quality = backend_token in {
        'native_full_deepx', 'native_full_hailo8',
        'native_full_hailo10h',
    }
    vendor_quality_binding: dict[str, Any] = {}
    vendor_central_result_sha = ''
    if strict_vendor_full_quality:
        vendor_errors: list[str] = []
        raw_binding = row.get('quality_request_binding')
        vendor_quality_binding = (
            dict(raw_binding) if isinstance(raw_binding, Mapping) else {}
        )
        binding_sha = _normalize_sha256(
            row.get('quality_request_binding_sha256')
        )
        binding_set_sha = _normalize_sha256(
            row.get('quality_request_binding_set_sha256')
        )
        declared_binding_sha = _normalize_sha256(
            vendor_quality_binding.get('binding_sha256')
        )
        unhashed_binding = dict(vendor_quality_binding)
        unhashed_binding.pop('binding_sha256', None)
        for field, expected in (
            ('backend', backend_token),
            ('model_id', str(row.get('model') or row.get('model_id') or '')),
            ('setup_id', str(row.get('setup_id') or '')),
            ('task', str(row.get('task') or '')),
            ('variant', 'full'),
        ):
            if str(vendor_quality_binding.get(field) or '') != expected:
                vendor_errors.append(f'vendor_full_quality_identity_mismatch:{field}')
        if not str(vendor_quality_binding.get('source_case_id') or '').strip():
            vendor_errors.append('vendor_full_quality_source_case_missing')
        if row.get('quality_request_binding_status') != 'verified_exact':
            vendor_errors.append('vendor_full_quality_binding_status_invalid')
        if (
            not vendor_quality_binding
            or not binding_sha
            or binding_sha != declared_binding_sha
            or _canonical_json_sha256(unhashed_binding) != binding_sha
        ):
            vendor_errors.append('vendor_full_quality_binding_sha256_invalid')
        if not binding_set_sha:
            vendor_errors.append('vendor_full_quality_binding_set_sha256_missing')
        full_contract = row.get('full_command_contract')
        full_contract = (
            dict(full_contract)
            if isinstance(full_contract, Mapping) else {}
        )
        contract_binding = full_contract.get('quality_request_binding')
        if (
            not isinstance(contract_binding, Mapping)
            or dict(contract_binding) != vendor_quality_binding
            or _normalize_sha256(
                full_contract.get('quality_request_binding_sha256')
            ) != binding_sha
            or _normalize_sha256(
                full_contract.get('quality_request_binding_set_sha256')
            ) != binding_set_sha
        ):
            vendor_errors.append('vendor_full_command_binding_mismatch')
        claim_fields = [
            'source_request_sha256', 'model_sha256',
            'validation_dataset_sha256',
            'validation_dataset_image_ids_sha256',
            'validation_dataset_ground_truth_sha256',
            'task_quality_policy_sha256',
            'quality_contract_sha256',
            'preprocessing_contract_sha256',
            'quality_record_endpoint_contract_sha256',
        ]
        if str(row.get('task') or '').strip().lower() == 'detection':
            claim_fields += [
                'decoder_contract_sha256', 'nms_contract_sha256',
            ]
        for field in claim_fields:
            binding_value = _normalize_sha256(
                vendor_quality_binding.get(field)
            )
            row_value = _normalize_sha256(row.get(field))
            if not binding_value or row_value != binding_value:
                vendor_errors.append(
                    f'vendor_full_quality_claim_mismatch:{field}'
                )
        vendor_central_result_sha = _normalize_sha256(
            vendor_quality_binding.get('central_quality_result_sha256')
        )
        if not vendor_central_result_sha:
            vendor_errors.append(
                'vendor_full_central_quality_result_sha256_missing'
            )
        binding_preprocessing = vendor_quality_binding.get(
            'preprocessing_contract'
        )
        binding_preprocessing = (
            dict(binding_preprocessing)
            if isinstance(binding_preprocessing, Mapping) else {}
        )
        binding_preprocessing_identity = binding_preprocessing.get('identity')
        runtime_preprocessing_identity = row.get(
            'runtime_preprocessing_identity'
        )
        runtime_binding = full_contract.get('runtime_preprocessing_binding')
        runtime_binding = (
            dict(runtime_binding)
            if isinstance(runtime_binding, Mapping) else {}
        )
        runtime_binding_identity = runtime_binding.get('identity')
        runtime_binding_identity = (
            dict(runtime_binding_identity)
            if isinstance(runtime_binding_identity, Mapping) else {}
        )
        if (
            not isinstance(binding_preprocessing_identity, Mapping)
            or not isinstance(runtime_preprocessing_identity, Mapping)
            or dict(binding_preprocessing_identity)
            != dict(runtime_preprocessing_identity)
            or runtime_binding_identity != dict(binding_preprocessing_identity)
            or _normalize_sha256(
                runtime_binding.get('sha256')
            ) != _normalize_sha256(
                vendor_quality_binding.get(
                    'preprocessing_contract_sha256'
                )
            )
        ):
            vendor_errors.append(
                'vendor_full_runtime_preprocessing_binding_mismatch'
            )
        if vendor_errors:
            row['quality_first_binding_errors'] = list(dict.fromkeys(
                vendor_errors
            ))
            row['quality_first_binding_status'] = (
                'vendor_full_quality_binding_invalid_or_missing'
            )
            row['central_quality_binding_status'] = (
                'vendor_full_quality_binding_invalid_or_missing'
            )
            return
    split_backend = (
        bool(is_native_split_backend(row.get('backend')))
        if is_native_split_backend is not None else
        _quality_run_token(row.get('backend'))
        in {'hailo8_to_trt', 'hailo10h_to_trt', 'deepx_to_trt'}
    )
    if split_backend and split_quality_authority is not None:
        if apply_native_split_quality_authority is None:
            row['central_quality_binding_status'] = (
                'native_split_quality_authority_verifier_unavailable'
            )
            return
        apply_native_split_quality_authority(row, split_quality_authority)
    historical_legacy_api = bool(
        split_backend
        and split_quality_authority is None
        and _explicit_historical_central_schema(central_results)
    )
    authority_requires_qf = bool(
        split_backend
        and (
            native_split_quality_required_for_row(row, split_quality_authority)
            if native_split_quality_required_for_row is not None
            else True
        )
    )
    strict_split_quality_first = bool(
        split_backend
        and (
            _split_quality_first_is_explicit(row)
            or (authority_requires_qf and not historical_legacy_api)
        )
    )
    split_binding_rejections: list[str] = []
    if (
        strict_split_quality_first
        and split_quality_authority is not None
        and split_quality_authority.get('valid') is not True
    ):
        row['quality_first_binding_errors'] = list(
            split_quality_authority.get('errors')
            or ['native_split_quality_authority_invalid']
        )
        row['quality_first_binding_status'] = (
            'native_split_quality_authority_invalid'
        )
        row['central_quality_binding_status'] = (
            'native_split_quality_authority_invalid'
        )
        return
    if strict_split_quality_first and bind_quality_to_native_split is None:
        row['quality_first_binding_errors'] = [
            'native_split_quality_verifier_unavailable',
        ]
        row['quality_first_binding_status'] = (
            'native_split_quality_binding_invalid_or_missing'
        )
        row['central_quality_binding_status'] = (
            'native_split_quality_binding_invalid_or_missing'
        )
        return
    native_selected_binding: dict[str, Any] = {}
    native_central_selection: dict[str, Any] = {}
    if strict_split_quality_first:
        if (
            validate_native_split_quality_binding is None
            or validate_central_native_split_quality_selection is None
        ):
            row['quality_first_binding_errors'] = [
                'native_split_quality_selection_verifier_unavailable',
            ]
            row['central_quality_binding_status'] = (
                'native_split_quality_selection_verifier_unavailable'
            )
            return
        raw_native_binding = row.get('native_split_quality_binding')
        if not isinstance(raw_native_binding, Mapping):
            row['quality_first_binding_errors'] = [
                'native_split_quality_selected_binding_missing',
            ]
            row['central_quality_binding_status'] = (
                'native_split_quality_binding_invalid_or_missing'
            )
            return
        native_selected_binding, native_binding_status = (
            bind_quality_to_native_split(
                native_row=row,
                quality_binding=raw_native_binding,
                verification_mode='portable',
            )
        )
        if native_selected_binding is None:
            row['quality_first_binding_errors'] = [native_binding_status]
            row['central_quality_binding_status'] = (
                'native_split_quality_binding_invalid_or_missing'
            )
            return
        native_central_selection, selection_status = (
            validate_central_native_split_quality_selection(
                native_selected_binding, required=True,
            )
        )
        if native_central_selection is None:
            row['quality_first_binding_errors'] = [selection_status]
            row['central_quality_binding_status'] = (
                'native_split_quality_binding_invalid_or_missing'
            )
            return
        binding_eval_run_id = str(
            native_selected_binding.get('eval_run_id') or ''
        ).strip()
        authority_eval_run_id = str(
            (split_quality_authority or {}).get('run_id') or binding_eval_run_id
        ).strip()
        exact_eval_values = (
            binding_eval_run_id,
            str(native_central_selection.get('eval_run_id') or '').strip(),
            str(row.get('eval_run_id') or '').strip(),
            str(row.get('native_split_quality_eval_run_id') or '').strip(),
            authority_eval_run_id,
        )
        if (
            not authority_eval_run_id
            or any(value != authority_eval_run_id for value in exact_eval_values)
        ):
            row['quality_first_binding_errors'] = [
                'native_split_quality_authority_eval_run_id_mismatch',
            ]
            row['central_quality_binding_status'] = (
                'native_split_quality_authority_eval_run_id_mismatch'
            )
            return
    native_trt_producer: dict[str, Any] = {}
    native_trt_producer_sha = ''
    if strict_trt_quality_first:
        native_trt_producer, native_trt_producer_sha, native_errors = (
            _native_trt_quality_first_binding(row, task=str(row.get('task') or ''))
        )
        row['quality_first_binding_errors'] = native_errors
        row['quality_first_producer_identity_sha256'] = native_trt_producer_sha
        if native_errors:
            row['central_quality_binding_status'] = (
                'native_quality_first_binding_invalid'
            )
            return
    task = str(row.get('task') or '').strip().lower()
    physical_endpoint_hash = _normalize_sha256(
        row.get('endpoint_contract_hash')
    )
    endpoint_hash = physical_endpoint_hash
    completed_quality_join = False
    strict_trt_completed_bridge = False
    completed_projection_verified = False
    if _is_native_full_row(row) and task == 'detection':
        comparison = row.get(
            'completed_task_comparison_endpoint_contract'
        )
        try:
            verified_comparison = (
                _verify_completed_detection_comparison_endpoint_contract(
                    comparison
                )
                if (
                    _verify_completed_detection_comparison_endpoint_contract
                    is not None
                )
                else None
            )
            _completed_mode, _completed_source, verified_bound_comparison = (
                _verified_completed_v2_contract(row)
            )
        except Exception:
            verified_comparison = None
            verified_bound_comparison = None
        completed_hash = _normalize_sha256(
            row.get(
                'completed_task_comparison_endpoint_contract_hash'
            )
        )
        completed_id = str(
            row.get(
                'completed_task_comparison_output_endpoint_id'
            ) or ''
        )
        completed_projection_verified = bool(
            isinstance(verified_comparison, Mapping)
            and isinstance(verified_bound_comparison, Mapping)
            and dict(verified_bound_comparison)
            == dict(verified_comparison)
            and completed_hash
            == _normalize_sha256(
                verified_comparison.get('endpoint_contract_hash')
            )
            and completed_id
            == str(verified_comparison.get('output_endpoint_id') or '')
        )
        if completed_projection_verified and not strict_trt_quality_first:
            endpoint_hash = completed_hash
            completed_quality_join = True
        elif completed_projection_verified:
            projection_ok, projection_status = (
                _native_trt_completed_quality_projection(
                    native_trt_producer,
                    verified_comparison,
                    physical_endpoint_hash=physical_endpoint_hash,
                )
            )
            if not projection_ok:
                row['central_quality_binding_status'] = projection_status
                return
            # Central transports the physical producer identity.  The Quality
            # gate may use the completed comparison stratum only after the
            # sealed decoder/NMS semantics above prove them equivalent.
            endpoint_hash = completed_hash
            strict_trt_completed_bridge = True
            row['completed_task_quality_projection_verified'] = True
            row['completed_task_quality_projection_status'] = projection_status
            row['completed_task_quality_projection_contract_hash'] = (
                completed_hash
            )
            row['completed_task_quality_projection_output_endpoint_id'] = (
                completed_id
            )
        if strict_trt_quality_first and not completed_projection_verified:
            row['central_quality_binding_status'] = (
                'native_completed_task_projection_invalid'
            )
            return
    precision = str(row.get('runtime_precision_identity') or '').strip().lower().replace(' ', '')
    if (
        row.get('endpoint_contract_complete') is not True
        or not endpoint_hash
        or not precision
    ):
        row['central_quality_binding_status'] = 'native_endpoint_or_precision_identity_incomplete'
        return
    if completed_quality_join or strict_trt_completed_bridge:
        row['quality_join_endpoint'] = 'completed_task_decoded_nms'
        row['quality_join_endpoint_contract_hash'] = endpoint_hash
        row['physical_endpoint_contract_hash'] = physical_endpoint_hash
    else:
        row['endpoint_contract_hash'] = endpoint_hash
    model = str(row.get('model') or '').strip().lower()
    setup = str(row.get('setup_id') or '').strip().lower()
    variant = 'full' if _is_native_full_row(row) else 'composed'
    expected_runs = _expected_central_quality_runs(row)
    expected_cases = _native_quality_case_ids(row)
    if strict_vendor_full_quality:
        # A centrally evaluated Full model can originate from b038 while the
        # representative performance input came from a different boundary.
        # The verified binding names the source; variant=full is checked above
        # and again on the result. Never relabel a composed/split result.
        expected_cases = {str(vendor_quality_binding['source_case_id']).strip().lower()}
    if strict_trt_quality_first:
        expected_cases = {'full'}
    expected_policy_sha, expected_policy_valid = _consistent_sha256(policy.sha256())
    if not expected_policy_valid:
        row['central_quality_binding_status'] = 'quality_policy_identity_invalid'
        return

    def _mapping(source: Any, key: str) -> dict[str, Any]:
        if not isinstance(source, dict):
            return {}
        value = source.get(key)
        return value if isinstance(value, dict) else {}

    def _sha_from_sources(
        sources: Iterable[dict[str, Any]], aliases: tuple[str, ...], *, required: bool,
    ) -> tuple[str, bool]:
        return _consistent_sha256(
            *(
                source.get(alias)
                for source in sources
                for alias in aliases
                if isinstance(source, dict)
            ),
            required=required,
        )

    def _row_sha_matches(canonical: str, aliases: tuple[str, ...]) -> bool:
        row_value, valid = _sha_from_sources((row,), aliases, required=False)
        return bool(valid and (not row_value or (canonical and row_value == canonical)))

    matches: list[tuple[dict[str, Any], dict[str, Any]]] = []
    exact_split_result_seen = False
    expected_split_result_sha = (
        _normalize_sha256(native_selected_binding.get('central_result_sha256'))
        if strict_split_quality_first else ''
    )
    for result in central_results:
        identity = result.get('request_identity') if isinstance(result.get('request_identity'), dict) else {}
        if identity.get('identity_valid') is not True:
            continue
        if str(result.get('status') or '') != 'completed' or str(result.get('technical_status') or 'completed') != 'completed':
            continue
        if (
            strict_vendor_full_quality
            and _canonical_json_sha256(result)
            != vendor_central_result_sha
        ):
            continue
        if strict_split_quality_first:
            if _canonical_json_sha256(result) != expected_split_result_sha:
                continue
            exact_split_result_seen = True
            exact_identity, exact_status = _strict_split_central_result_identity(
                result, identity,
            )
            if exact_identity is None:
                split_binding_rejections.append(exact_status)
                continue
            result_model = exact_identity['model_id']
            result_task = exact_identity['task']
            result_variant = exact_identity['variant']
            result_run = _quality_run_token(exact_identity['source_run_id'])
            result_setup = exact_identity['setup_id'].lower()
            result_case = exact_identity['case_id']
            if exact_identity['eval_run_id'] != str(
                native_central_selection.get('eval_run_id') or ''
            ).strip():
                split_binding_rejections.append(
                    'central_native_split_eval_run_id_authority_mismatch'
                )
                continue
            receipt_identity = {
                'model_id': result_model,
                'case_id': result_case,
                'source_run_id': exact_identity['source_run_id'],
                'setup_id': exact_identity['setup_id'],
                'task': result_task,
                'variant': result_variant,
                'runtime_precision_identity': exact_identity[
                    'runtime_precision_identity'
                ],
            }
            if any(
                str(native_central_selection.get(field) or '').strip().lower()
                != str(expected or '').strip().lower()
                for field, expected in receipt_identity.items()
            ):
                split_binding_rejections.append(
                    'central_native_split_selection_receipt_identity_mismatch'
                )
                continue
        else:
            result_model = str(result.get('model_id') or identity.get('model_id') or '').strip().lower()
            result_task = str(result.get('task') or identity.get('task') or '').strip().lower()
            result_variant = str(result.get('variant') or identity.get('variant') or '').strip().lower()
            result_run = _quality_run_token(result.get('source_run_id') or result.get('run_id') or identity.get('source_run_id'))
            result_setup = str(result.get('source_setup_id') or identity.get('setup_id') or '').strip().lower()
            result_case = str(result.get('case_id') or identity.get('case_id') or '').strip().lower()
        if result_model != model:
            continue
        if result_task != task:
            continue
        if result_variant != variant:
            continue
        if result_run not in expected_runs:
            continue
        if result_setup != setup:
            continue
        if expected_cases and result_case not in expected_cases:
            continue
        central_split_binding: dict[str, Any] = {}
        central_split_binding_sha = ''
        if strict_split_quality_first:
            result_binding = result.get('native_split_quality_binding')
            identity_binding = identity.get('native_split_quality_binding')
            if (
                not isinstance(result_binding, dict)
                or not isinstance(identity_binding, dict)
                or result_binding != identity_binding
            ):
                split_binding_rejections.append(
                    'central_native_split_quality_binding_duplicate_mismatch'
                )
                continue
            producer_binding, split_status = (
                validate_native_split_quality_binding(
                    result_binding,
                    expected_identity={
                        'model': model, 'case': result_case,
                        'setup_id': setup,
                        'backend': exact_identity['source_run_id'],
                        'task': task,
                        'precision': precision,
                    },
                    verification_mode='portable',
                )
            )
            if producer_binding is None:
                split_binding_rejections.append(split_status)
                continue
            producer_binding_sha = _normalize_sha256(
                producer_binding.get('binding_sha256')
            )
            if any(
                _normalize_sha256(container.get(
                    'native_split_quality_binding_sha256'
                )) != producer_binding_sha
                for container in (result, identity)
            ):
                split_binding_rejections.append(
                    'central_native_split_quality_binding_sha256_duplicate_mismatch'
                )
                continue
            reconstructed_producer = copy.deepcopy(native_selected_binding)
            reconstructed_producer.pop('binding_sha256', None)
            for field in (
                'producer_binding_sha256', 'source_request_sha256',
                'central_result_sha256', 'central_quality_selection',
                'central_quality_selection_sha256',
            ):
                reconstructed_producer.pop(field, None)
            reconstructed_producer['binding_sha256'] = _normalize_sha256(
                native_selected_binding.get('producer_binding_sha256')
            )
            if (
                producer_binding_sha
                != _normalize_sha256(
                    native_selected_binding.get('producer_binding_sha256')
                )
                or producer_binding != reconstructed_producer
            ):
                split_binding_rejections.append(
                    'central_native_split_quality_producer_binding_mismatch'
                )
                continue
            central_result_sha = _canonical_json_sha256(result)
            if central_result_sha != _normalize_sha256(
                native_selected_binding.get('central_result_sha256')
            ):
                split_binding_rejections.append(
                    'central_native_split_quality_result_sha256_mismatch'
                )
                continue
            central_split_binding = native_selected_binding
            central_split_binding_sha = _normalize_sha256(
                native_selected_binding.get('binding_sha256')
            )
        central_trt_producer: dict[str, Any] = {}
        central_trt_producer_sha = ''
        if strict_trt_quality_first:
            central_trt_producer, central_trt_producer_sha, central_errors = (
                _central_trt_quality_producer(result, task=task)
            )
            if (
                central_errors
                or central_trt_producer_sha != native_trt_producer_sha
                or central_trt_producer != native_trt_producer
            ):
                continue
        flat_sources = (result, identity)
        central_source_endpoint_hash = ''
        if strict_trt_completed_bridge:
            result_endpoint_hash, valid = _sha_from_sources(
                flat_sources, ('endpoint_contract_hash',), required=True,
            )
            valid = bool(
                valid
                and result_endpoint_hash == physical_endpoint_hash
                and _optional_central_completed_projection_matches(
                    result, identity, verified_comparison,
                )
            )
            central_source_endpoint_hash = result_endpoint_hash
        elif completed_quality_join:
            result_endpoint_hash, valid = _sha_from_sources(
                flat_sources,
                ('completed_task_endpoint_contract_hash',),
                required=True,
            )
            result_completed_contract = (
                result.get('completed_task_endpoint_contract')
                if isinstance(
                    result.get('completed_task_endpoint_contract'),
                    Mapping,
                )
                else identity.get('completed_task_endpoint_contract')
                if isinstance(
                    identity.get('completed_task_endpoint_contract'),
                    Mapping,
                )
                else None
            )
            try:
                result_completed_contract = (
                    _verify_completed_detection_comparison_endpoint_contract(
                        result_completed_contract
                    )
                )
            except Exception:
                valid = False
            valid = bool(
                valid
                and isinstance(result_completed_contract, Mapping)
                and _normalize_sha256(
                    result_completed_contract.get(
                        'endpoint_contract_hash'
                    )
                ) == result_endpoint_hash
                and str(
                    result.get('quality_join_endpoint')
                    or identity.get('quality_join_endpoint') or ''
                ).strip().lower() == 'completed_task_decoded_nms'
            )
        else:
            result_endpoint_hash, valid = _sha_from_sources(
                flat_sources, ('endpoint_contract_hash',), required=True,
            )
        expected_result_endpoint_hash = (
            physical_endpoint_hash
            if strict_trt_completed_bridge else endpoint_hash
        )
        if not valid or result_endpoint_hash != expected_result_endpoint_hash:
            continue
        result_precision = str(result.get('runtime_precision_identity') or identity.get('runtime_precision_identity') or '').strip().lower().replace(' ', '')
        if result_precision != precision:
            continue
        request_sha, valid = _sha_from_sources(
            flat_sources, ('source_request_sha256', 'quality_source_sha256'),
            required=True,
        )
        if not valid:
            continue
        if strict_split_quality_first:
            exact_request_values = (
                _normalize_sha256(result.get('source_request_sha256')),
                _normalize_sha256(identity.get('source_request_sha256')),
                _normalize_sha256(
                    native_selected_binding.get('source_request_sha256')
                ),
                _normalize_sha256(row.get('source_request_sha256')),
                _normalize_sha256(
                    row.get('native_split_quality_source_request_sha256')
                ),
            )
            if (
                not request_sha
                or any(value != request_sha for value in exact_request_values)
            ):
                split_binding_rejections.append(
                    'central_native_split_quality_source_request_sha256_mismatch'
                )
                continue
        result_quality = _mapping(result, 'quality_contract')
        identity_quality = _mapping(identity, 'quality_contract')
        if strict_trt_quality_first:
            producer_quality = _mapping(central_trt_producer, 'quality_contract')
            if result_quality and result_quality != producer_quality:
                continue
            if identity_quality and identity_quality != producer_quality:
                continue
            result_quality = producer_quality
            identity_quality = producer_quality
        result_model = _mapping(result, 'model') or _mapping(result_quality, 'model')
        identity_model = _mapping(identity, 'model') or _mapping(identity_quality, 'model')
        result_dataset = _mapping(result, 'dataset') or _mapping(result_quality, 'dataset')
        identity_dataset = _mapping(identity, 'dataset') or _mapping(identity_quality, 'dataset')
        if strict_trt_quality_first:
            producer_model = _mapping(central_trt_producer, 'model')
            producer_dataset = _mapping(central_trt_producer, 'dataset')
            result_model = producer_model
            identity_model = producer_model
            result_dataset = producer_dataset
            identity_dataset = producer_dataset
        model_sha, model_valid = _consistent_sha256(
            *(
                source.get(alias)
                for source in flat_sources
                for alias in ('model_sha256', 'source_model_sha256', 'source_onnx_sha256')
            ),
            result_model.get('sha256'), identity_model.get('sha256'),
            required=True,
        )
        dataset_sha, dataset_valid = _consistent_sha256(
            *(
                source.get(alias)
                for source in flat_sources
                for alias in (
                    'validation_dataset_sha256', 'validation_dataset_manifest_sha256',
                    'dataset_manifest_sha256', 'manifest_sha256', 'dataset_sha256',
                )
            ),
            result_dataset.get('manifest_sha256'), result_dataset.get('sha256'),
            identity_dataset.get('manifest_sha256'), identity_dataset.get('sha256'),
            required=True,
        )
        image_ids_sha, image_ids_valid = _sha_from_sources(
            (*flat_sources, result_dataset, identity_dataset),
            (
                'validation_dataset_image_ids_sha256', 'validation_image_ids_sha256',
                'dataset_image_ids_sha256', 'image_ids_sha256',
            ),
            required=False,
        )
        ground_truth_sha, ground_truth_valid = _sha_from_sources(
            (*flat_sources, result_dataset, identity_dataset),
            (
                'validation_dataset_ground_truth_sha256', 'validation_ground_truth_sha256',
                'dataset_ground_truth_sha256', 'ground_truth_sha256',
            ),
            required=False,
        )
        policy_sha, policy_valid = _sha_from_sources(
            flat_sources,
            (
                'policy_sha256', 'task_quality_policy_sha256',
            ),
            required=True,
        )
        if (
            not model_valid or not dataset_valid
            or not image_ids_valid or not ground_truth_valid
            or not policy_valid
        ):
            continue
        if policy_sha != expected_policy_sha:
            if strict_split_quality_first:
                split_binding_rejections.append(
                    'central_native_split_quality_policy_mismatch'
                )
            continue
        if not all((
            _row_sha_matches(model_sha, ('model_sha256', 'source_model_sha256', 'source_onnx_sha256')),
            _row_sha_matches(
                dataset_sha,
                ('validation_dataset_sha256', 'validation_dataset_manifest_sha256', 'dataset_manifest_sha256', 'dataset_sha256'),
            ),
            _row_sha_matches(
                image_ids_sha,
                ('validation_dataset_image_ids_sha256', 'validation_image_ids_sha256', 'dataset_image_ids_sha256', 'image_ids_sha256'),
            ),
            _row_sha_matches(
                ground_truth_sha,
                ('validation_dataset_ground_truth_sha256', 'validation_ground_truth_sha256', 'dataset_ground_truth_sha256', 'ground_truth_sha256'),
            ),
            _row_sha_matches(
                policy_sha,
                ('task_quality_policy_sha256',),
            ),
        )):
            continue
        component_hashes: dict[str, str] = {}
        components_valid = True
        for field in (
            'quality_contract_sha256', 'preprocessing_contract_sha256',
            'decoder_contract_sha256', 'nms_contract_sha256',
            'quality_record_endpoint_contract_sha256',
        ):
            component_hashes[field], field_valid = _sha_from_sources(
                (*flat_sources, result_quality, identity_quality),
                (field,), required=False,
            )
            components_valid = components_valid and field_valid
        mandatory = ['quality_contract_sha256', 'preprocessing_contract_sha256']
        if task == 'detection':
            mandatory += ['decoder_contract_sha256', 'nms_contract_sha256']
        if not components_valid or any(not component_hashes[field] for field in mandatory):
            continue
        if any(
            not _row_sha_matches(component_hashes[field], (field,))
            for field in mandatory
        ):
            continue
        matches.append((result, {
            'source_request_sha256': request_sha,
            'model_sha256': model_sha,
            'validation_dataset_sha256': dataset_sha,
            'validation_dataset_image_ids_sha256': image_ids_sha,
            'validation_dataset_ground_truth_sha256': ground_truth_sha,
            'task_quality_policy_sha256': policy_sha,
            'runtime_quality_gate_policy_sha256': policy_sha,
            'quality_first_producer_identity_sha256': central_trt_producer_sha,
            'central_source_endpoint_contract_hash': (
                central_source_endpoint_hash
            ),
            'quality_join_endpoint_contract_hash': (
                endpoint_hash if strict_trt_completed_bridge else ''
            ),
            'trt_completed_quality_bridge_status': (
                row.get('completed_task_quality_projection_status')
                if strict_trt_completed_bridge else ''
            ),
            'native_split_quality_binding': central_split_binding,
            'native_split_quality_binding_sha256': central_split_binding_sha,
            'native_split_quality_source_request_sha256': _normalize_sha256(
                central_split_binding.get('source_request_sha256')
            ),
            'native_split_quality_central_result_sha256': _normalize_sha256(
                central_split_binding.get('central_result_sha256')
            ),
            'native_split_quality_selection_sha256': _normalize_sha256(
                central_split_binding.get('central_quality_selection_sha256')
            ),
            **component_hashes,
        }))
    if (
        strict_split_quality_first
        and not exact_split_result_seen
        and expected_split_result_sha
    ):
        split_binding_rejections.append(
            'central_native_split_quality_result_sha256_not_found'
        )
    row['central_quality_binding_raw_candidate_count'] = len(matches)
    if (
        strict_trt_quality_first or strict_split_quality_first
        or strict_vendor_full_quality
    ) and matches:
        # Collection may expose the same centrally evaluated JSON through two
        # mirror paths.  Collapse only an identical request+producer+result
        # signature; a second request, producer, or any result drift remains
        # a second candidate and therefore blocks the join.
        unique_matches: list[tuple[dict[str, Any], dict[str, Any]]] = []
        seen_signatures: set[tuple[str, str, str, str]] = set()
        for result, evidence in matches:
            try:
                # The same setup-local Full-TRT quality result can be
                # collected through two mirror paths.  The central merge
                # already treats those copies as one logical observation
                # when request, producer and scientific outcome agree.
                # Do the same here instead of hashing path/timestamp
                # diagnostics which legitimately differ between mirrors.
                result_signature = _central_result_mirror_signature(
                    result,
                    trt_quality_first=strict_trt_quality_first,
                )
            except Exception:
                result_signature = f'unserializable:{len(unique_matches)}'
            signature = (
                str(evidence.get('source_request_sha256') or ''),
                str(evidence.get('quality_first_producer_identity_sha256') or ''),
                str(evidence.get('native_split_quality_binding_sha256') or ''),
                result_signature,
            )
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            unique_matches.append((result, evidence))
        matches = unique_matches
    row['central_quality_binding_candidate_count'] = len(matches)
    if len(matches) != 1:
        if strict_split_quality_first and not matches and split_binding_rejections:
            row['quality_first_binding_errors'] = list(dict.fromkeys(
                split_binding_rejections
            ))
            row['quality_first_binding_status'] = (
                'native_split_quality_binding_invalid_or_missing'
            )
            row['central_quality_binding_status'] = (
                'native_split_quality_binding_invalid_or_missing'
            )
        else:
            row['central_quality_binding_status'] = (
                'ambiguous' if len(matches) > 1 else 'no_exact_identity_match'
            )
        return
    result, evidence = matches[0]
    identity = result.get('request_identity') if isinstance(result.get('request_identity'), dict) else {}
    decision = str(result.get('decision') or 'unavailable').strip().lower()
    gate = {
        'schema': 'onnx-splitpoint/task-quality-gate',
        'schema_version': 3,
        'task': task,
        'variant': variant,
        'tier': str(policy.dataset_tier or 'screening'),
        'canonical_reference': 'management_cpu_ort_full_onnx',
        'decision': decision,
        'status': decision,
        'primary': dict(result.get('primary') or {}),
        'guardrails': dict(result.get('guardrails') or {}),
        'policy': policy.as_dict(),
        'policy_sha256': expected_policy_sha,
        'execution_location': 'central_management',
        'quality_input_request': {
            'source_request_sha256': evidence['source_request_sha256'],
            'endpoint_contract_hash': endpoint_hash,
            'source_endpoint_contract_hash': evidence[
                'central_source_endpoint_contract_hash'
            ],
            'quality_join_endpoint': (
                'completed_task_decoded_nms'
                if strict_trt_completed_bridge else ''
            ),
            'quality_join_endpoint_contract_hash': evidence[
                'quality_join_endpoint_contract_hash'
            ],
            'runtime_precision_identity': precision,
            'model_sha256': evidence['model_sha256'],
            'validation_dataset_sha256': evidence['validation_dataset_sha256'],
            'validation_dataset_image_ids_sha256': evidence['validation_dataset_image_ids_sha256'],
            'validation_dataset_ground_truth_sha256': evidence['validation_dataset_ground_truth_sha256'],
        },
    }
    gate.update({k: result[k] for k in ('accuracy_assessment', 'reporting_policy', 'secondary_accuracy_assessments', 'accuracy_warnings', 'legacy_decision', 'n', 'reference_identity') if k in result})
    row.update({
        'task_quality_gate': gate,
        'task_quality_policy': policy.as_dict(),
        'central_quality_evidence_verified': True,
        # v2.72: exact evidence binding is independent from the observed
        # accuracy result.  ``accuracy_gate_pass`` remains the positive-result
        # axis used by claim/ranking gates.
        'precision_quality_verified': True,
        'precision_quality_binding_verified': True,
        'task_quality_observation_valid': decision in {
            'pass', 'fail', 'inconclusive', 'reference_close', 'accuracy_loss', 'not_estimable',
        },
        'central_quality_binding_status': 'exact_identity_match',
        'source_request_sha256': evidence['source_request_sha256'],
        'model_sha256': evidence['model_sha256'],
        'validation_dataset_sha256': evidence['validation_dataset_sha256'],
        'validation_dataset_image_ids_sha256': evidence['validation_dataset_image_ids_sha256'],
        'validation_dataset_ground_truth_sha256': evidence['validation_dataset_ground_truth_sha256'],
        'task_quality_policy_sha256': evidence['task_quality_policy_sha256'],
        'runtime_quality_gate_policy_sha256': evidence['runtime_quality_gate_policy_sha256'],
        'quality_contract_sha256': evidence['quality_contract_sha256'],
        'preprocessing_contract_sha256': evidence['preprocessing_contract_sha256'],
        'decoder_contract_sha256': evidence['decoder_contract_sha256'],
        'nms_contract_sha256': evidence['nms_contract_sha256'],
        'quality_record_endpoint_contract_sha256': evidence['quality_record_endpoint_contract_sha256'],
        'central_source_endpoint_contract_hash': evidence[
            'central_source_endpoint_contract_hash'
        ],
        'trt_completed_quality_bridge_status': evidence[
            'trt_completed_quality_bridge_status'
        ],
        'quality_source_run_id': str(result.get('source_run_id') or identity.get('source_run_id') or ''),
        'quality_source_setup_id': str(result.get('source_setup_id') or identity.get('setup_id') or ''),
        'quality_source_variant': variant,
    })
    if strict_trt_quality_first:
        row.update({
            'quality_first_producer_identity': native_trt_producer,
            'quality_first_producer_identity_sha256': native_trt_producer_sha,
            'quality_first_binding_errors': [],
            'quality_first_binding_status': 'central_native_exact_identity_match',
        })
    if strict_split_quality_first:
        row.update({
            'native_split_quality_binding': evidence[
                'native_split_quality_binding'
            ],
            'native_split_quality_binding_sha256': evidence[
                'native_split_quality_binding_sha256'
            ],
            'native_split_quality_source_request_sha256': evidence[
                'native_split_quality_source_request_sha256'
            ],
            'native_split_quality_central_result_sha256': evidence[
                'native_split_quality_central_result_sha256'
            ],
            'native_split_quality_selection_sha256': evidence[
                'native_split_quality_selection_sha256'
            ],
            'quality_first_binding_errors': [],
            'quality_first_binding_status': (
                'central_native_exact_engine_command_boundary_match'
            ),
        })
    if strict_vendor_full_quality:
        row.update({
            'quality_request_binding_status': 'verified_exact',
            'quality_first_binding_errors': [],
            'quality_first_binding_status': (
                'central_vendor_full_exact_request_binding_match'
            ),
            'vendor_full_central_quality_result_sha256': (
                vendor_central_result_sha
            ),
        })


def _native_full_e2e_contract_gate(
    manifest: Path, row: dict[str, Any], task: str,
) -> dict[str, Any]:
    """Fail closed for accelerator-only Full outputs.

    A vendor runtime returning YOLO raw heads has measured accelerator latency,
    not a complete detection pipeline.  The validator may still decode those
    tensors heuristically for diagnostics, but that decode is not a frozen,
    timed host tail and can therefore never make the Native Full row claimable.
    """
    if not _is_native_full_row(row):
        return {
            'e2e_claim_eligible': True,
            'e2e_scope': 'full_task_pipeline',
            'e2e_contract_reason': 'not_native_full',
        }
    payload = _load_json(manifest)
    payload = payload if isinstance(payload, dict) else {}
    if str(row.get('backend') or '').strip() == 'native_full_deepx':
        performance_mode = str(row.get('performance_input_contract_mode') or '').strip().lower()
        if performance_mode != 'explicit':
            return {
                'e2e_claim_eligible': False,
                'e2e_scope': 'diagnostic_input_autodetect',
                'e2e_contract_reason': 'deepx_performance_input_contract_not_explicit',
                'performance_input_contract_mode': performance_mode or 'missing',
                'contract_consistent': False,
            }
    input_contract_mode = str(payload.get('input_contract_mode') or 'explicit').strip().lower()
    if input_contract_mode != 'explicit':
        return {
            'e2e_claim_eligible': False,
            'e2e_scope': 'diagnostic_input_autodetect',
            'e2e_contract_reason': 'deepx_input_contract_not_explicit',
            'contract_consistent': False,
        }
    if str(task or '').lower() != 'detection':
        return {
            'e2e_claim_eligible': True,
            'e2e_scope': str(payload.get('e2e_scope') or 'full_task_pipeline'),
            'e2e_contract_reason': 'native_full_non_detection_explicit_input_contract',
        }
    if _completed_frozen_nms_attestation_passed(row):
        return {
            'e2e_claim_eligible': True,
            'e2e_scope': 'full_task_pipeline',
            'e2e_contract_reason': (
                'raw_accelerator_endpoint_plus_attested_frozen_timed_host_decode_nms'
            ),
            'requires_host_decode_nms': True,
            'host_postprocess_frozen': True,
            'postprocess_included': True,
            'postprocess_completion_verified': True,
            'completed_task_stage': 'decoded_nms',
            'completed_task_endpoint_attested': True,
        }
    family = str(payload.get('contract_family') or '').strip().lower()
    if family == 'raw_head':
        return {
            'e2e_claim_eligible': False,
            'e2e_scope': 'accelerator_only',
            'e2e_contract_reason': 'raw_head_without_frozen_timed_host_decode_nms',
            'requires_host_decode_nms': True,
            'host_postprocess_frozen': bool(payload.get('host_postprocess_frozen')),
            'contract_consistent': False,
        }
    if family != 'decoded_nms':
        return {
            'e2e_claim_eligible': False,
            'e2e_scope': str(payload.get('e2e_scope') or 'accelerator_only'),
            'e2e_contract_reason': 'native_full_detection_contract_unknown',
            'contract_consistent': False,
        }
    if not _decoded_nms_attestation_passed(payload):
        return {
            'e2e_claim_eligible': False,
            'e2e_scope': str(payload.get('e2e_scope') or 'accelerator_only'),
            'e2e_contract_reason': 'decoded_nms_runtime_value_attestation_missing_or_failed',
            'contract_consistent': False,
        }
    return {
        'e2e_claim_eligible': True,
        'e2e_scope': str(payload.get('e2e_scope') or 'full_task_pipeline'),
        'e2e_contract_reason': 'decoded_nms_included_in_runtime_output',
    }




def _manifest_matches_row(manifest: Path | None, row: dict[str, Any] | None) -> tuple[bool, str]:
    """Check that a candidate dump manifest belongs to exactly this native row.

    v59cs guard: older result JSONs could contain a stale output_manifest from a
    different model/case (for example a ResNet row pointing at a YOLO b038 dump).
    A missing dump is safer than validating the wrong tensor and producing a
    misleading PASS/FAIL image.
    """
    if not manifest or not row:
        return False, 'missing_manifest_or_row'
    text = str(manifest).replace('\\', '/')
    model = str(row.get('model') or '').strip()
    case = str(row.get('case') or row.get('case_id') or '').strip()
    backend = str(row.get('backend') or '').strip()
    precision = str(row.get('precision') or row.get('native_precision') or row.get('trt_precision') or '').strip()
    if _is_native_full_row(row):
        # Full-baseline dumps must be explicitly full/native-full outputs, not a
        # nearby split b*/native_pipeline dump.  If no exact full dump exists,
        # _find_dump should report missing_exact_full_dump.
        if '/native_pipeline/' in text or '/b0' in text or '/b1' in text or '/b2' in text or '/b3' in text or '/b4' in text:
            return False, 'full_row_points_to_split_dump'
        payload = _load_json(manifest)
        if not isinstance(payload, dict):
            return False, 'native_full_manifest_unreadable'
        expected = {
            'model': model,
            'backend': backend,
            'setup_id': str(row.get('setup_id') or '').strip(),
            'comparison_backend': str(row.get('comparison_backend') or '').strip(),
        }
        for key, value in expected.items():
            if key not in payload:
                return False, f'native_full_manifest_{key}_missing'
            actual = str(payload.get(key) or '').strip()
            if actual != value:
                return False, f'native_full_manifest_{key}_mismatch'
        if str(payload.get('case') or '').strip() != 'full':
            return False, 'native_full_manifest_case_mismatch'
        if str(payload.get('execution_mode') or '').strip() != 'native_full_baseline':
            return False, 'native_full_manifest_execution_mode_mismatch'
        path_tokens = (
            f'/model={_safe(model)}/',
            f'/backend={_safe(backend)}/',
            f"/setup={_safe(expected['setup_id'] or 'unspecified')}/",
            f"/comparison={_safe(expected['comparison_backend'] or 'unspecified')}/",
        )
        if not all(token in text for token in path_tokens):
            return False, 'native_full_manifest_path_identity_mismatch'
        return True, 'ok'
    if model and f'/{model}/' not in text:
        return False, 'model_mismatch'
    if case and f'/{case}/' not in text:
        return False, 'case_mismatch'
    # Backend-specific sanity checks; keep them permissive enough for copied roots.
    if 'hailo10' in backend and 'hailo10h_to_trt' not in text:
        return False, 'backend_mismatch'
    if 'deepx' in backend and 'deepx_to_trt' not in text:
        return False, 'backend_mismatch'
    if backend == 'hailo8_to_trt' and 'hailo_to_trt' not in text:
        return False, 'backend_mismatch'
    if precision and f'/{precision}/' not in text:
        return False, 'precision_mismatch'
    return True, 'ok'


def _find_dump(report: Path | None, roots: list[Path], row: dict[str, Any] | None = None) -> tuple[Path | None, str]:
    """Find the output dump manifest for exactly this row.

    v59cr intentionally removes the old global nearby_search fallback because it
    could map e.g. a ResNet row to a YOLO dump from the same EvalRun.  Missing is
    safer and more honest than validating the wrong tensor dump.
    """
    # Native Full rows carry the exact semantic dump directly.  Prefer the row
    # before consulting the performance report, because TensorRT/Hailo timing
    # reports are not required to know about output-dump manifests.
    if _is_native_full_row(row) and isinstance(row, dict):
        p_row, why_row = _extract_manifest_from_json_obj(
            row, roots, strict_unique=True, row=row,
        )
        if p_row:
            match, reason = _manifest_matches_row(p_row, row)
            if match:
                return p_row, f'full_row_{why_row}'
            raise NativeFullBindingError(
                f'native_full_row_manifest_identity_error:{reason}:{p_row}'
            )
    if not report:
        return None, 'missing_report'
    j = _load_json(report) or {}

    if _is_native_full_row(row):
        expected = {
            'model': str((row or {}).get('model') or '').strip(),
            'backend': str((row or {}).get('backend') or '').strip(),
            'setup_id': str((row or {}).get('setup_id') or '').strip(),
            'comparison_backend': str((row or {}).get('comparison_backend') or '').strip(),
        }
        candidates: list[dict[str, Any]] = []
        if isinstance(j, dict):
            for key in ('rows','results','baselines','items'):
                val = j.get(key)
                if isinstance(val, list):
                    candidates.extend([x for x in val if isinstance(x, dict)])
            candidates.append(j)
        manifests: list[Path] = []
        reasons: list[str] = []
        for obj in candidates:
            if any(key not in obj for key in expected):
                continue
            if any(str(obj.get(key) or '').strip() != value for key, value in expected.items()):
                continue
            p, why = _extract_manifest_from_json_obj(
                obj, roots, strict_unique=True, row=row,
            )
            if p:
                match, reason = _manifest_matches_row(p, row)
                if not match:
                    raise NativeFullBindingError(
                        f'native_full_report_manifest_identity_error:{reason}:{p}'
                    )
                manifests.append(p)
                reasons.append(why)
        hit = _unique_path_or_error(
            manifests,
            context='report_candidates:' + ':'.join(expected.values()),
            strict=True,
        )
        if hit:
            why = reasons[0] if len(reasons) == 1 else 'unique_report_candidate'
            return hit, f'full_exact_{why}'
        return None, 'missing_exact_full_dump'

    p, why = _extract_manifest_from_json_obj(j, roots, row=row)
    if p:
        match, reason = _manifest_matches_row(p, row)
        if match:
            return p, why
        # Explicit stale/wrong-row manifests are a hard warning.  Continue with
        # case-local search, but never accept the mismatching candidate.
        mismatch_reason = f'{why}_row_mismatch_{reason}'
    else:
        mismatch_reason = ''

    names = (
        'native_fifo_outputs_manifest.json', 'runner_outputs_manifest.json',
        'native_outputs_manifest.json', 'output_dump_manifest.json'
    )
    search_roots = []
    if report.parent.exists():
        search_roots.append(report.parent)
    if report.parent.parent.exists():
        search_roots.append(report.parent.parent)
    for sr in search_roots:
        for name in names:
            xs = sorted(sr.rglob(name))
            for cand in xs:
                match, reason = _manifest_matches_row(cand, row)
                if match:
                    return cand, 'case_local_search'
    return None, mismatch_reason or 'missing_exact_split_dump'

def _eval_root_from_summary(summary_path: Path) -> Path:
    # Usually <run>/reports/native_producer_summary.json.  Manual attachment
    # may place it below <run>/reports/native_producers/, so locate the frozen
    # run manifest instead of relying on a single parent depth.
    for candidate in (summary_path.parent, *summary_path.parents):
        if (candidate / 'run_manifest.json').is_file():
            return candidate
    if summary_path.parent.name == 'reports':
        return summary_path.parent.parent
    return summary_path.parent


def _native_split_authority(eval_root: Path) -> dict[str, Any]:
    if resolve_native_split_quality_authority is None:
        return {
            'schema': 'onnx-splitpoint/native-split-quality-authority',
            'schema_version': 1,
            'mode': 'invalid',
            'valid': False,
            'native_split_quality_required': True,
            'workflow_version': '',
            'run_id': '',
            'errors': ['native_split_quality_authority_verifier_unavailable'],
        }
    return resolve_native_split_quality_authority(
        run_manifest_path=eval_root / 'run_manifest.json',
        stage_path=eval_root / 'reports' / 'native_producer_stage.json',
    )


def _merge_native_split_quality_evidence(
    target: dict[str, Any], *sources: Mapping[str, Any],
) -> None:
    """Copy QF evidence while rejecting contradictory report/summary copies."""
    expanded: list[Mapping[str, Any]] = []
    for source in sources:
        if not isinstance(source, Mapping):
            continue
        expanded.append(source)
        command = source.get('native_command_contract')
        if isinstance(command, Mapping):
            expanded.append(command)
    fields = (
        'native_split_quality_binding_required',
        'native_split_quality_required',
        'native_split_quality_binding',
        'native_split_quality_binding_sha256',
        'native_split_quality_eval_run_id',
        'native_split_quality_source_run_id',
        'source_request_sha256',
        'native_split_quality_source_request_sha256',
        'native_split_quality_central_result_sha256',
        'native_split_quality_selection_sha256',
        'native_split_quality_consumer_attestation',
        'native_split_quality_consumer_status',
        'eval_run_id',
        'source_run_id',
    )
    conflicts: list[str] = []
    for field in fields:
        values = [
            source.get(field) for source in expanded
            if source.get(field) not in (None, '', {})
        ]
        if not values:
            continue
        canonical = [
            json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
            if isinstance(value, (dict, list)) else str(value)
            for value in values
        ]
        if len(set(canonical)) != 1:
            conflicts.append(field)
            continue
        target[field] = values[0]
    if conflicts:
        target['native_split_quality_required'] = True
        target['native_split_quality_binding_required'] = True
        target['native_split_quality_provenance_conflict'] = True
        target['native_split_quality_provenance_conflict_fields'] = conflicts
        target['native_split_quality_consumer_status'] = (
            'conflicting_duplicate_evidence'
        )


_VENDOR_FULL_QUALITY_BACKENDS = {
    'native_full_deepx',
    'native_full_hailo8',
    'native_full_hailo10h',
}


def _merge_vendor_full_quality_evidence(
    target: dict[str, Any], *sources: Mapping[str, Any],
) -> None:
    """Project one already verified Vendor-Full Quality binding unchanged.

    The Native-Full runner seals the request binding into both its summary row
    and the Full command contract.  Validation rebuilds a fresh row, so these
    mirrors must be transported explicitly.  Every populated mirror
    participates in the comparison; a contradictory copy is reported rather
    than selected or normalised away.
    """
    backend = str(target.get('backend') or '').strip().lower()
    if backend not in _VENDOR_FULL_QUALITY_BACKENDS:
        return

    # The freshly built validation row already contains some derived mirrors
    # and the copied Full command contract.  Treat that existing state as
    # evidence too, including its nested duplicates.
    top_sources = [target] + [
        source for source in sources if isinstance(source, Mapping)
    ]
    full_contracts = [
        source.get('full_command_contract')
        for source in top_sources
        if isinstance(source.get('full_command_contract'), Mapping)
    ]
    bindings = [
        source.get('quality_request_binding')
        for source in (*top_sources, *full_contracts)
        if isinstance(source.get('quality_request_binding'), Mapping)
    ]
    runtime_bindings = [
        contract.get('runtime_preprocessing_binding')
        for contract in full_contracts
        if isinstance(contract.get('runtime_preprocessing_binding'), Mapping)
    ]

    conflicts: list[str] = []

    def _present(value: Any) -> bool:
        return value not in (None, '', {}, [])

    def _canonical(value: Any) -> str:
        if isinstance(value, (Mapping, list, tuple)):
            return json.dumps(
                value, sort_keys=True, separators=(',', ':'),
                ensure_ascii=False,
            )
        return str(value)

    def _project(field: str, values: Iterable[Any]) -> None:
        # ``rec`` already contains identities derived by the validator (most
        # importantly runtime_precision_identity).  It is another mirror, not
        # a disposable default, and therefore participates in every conflict
        # check before the source value is copied.
        observed = [
            value for value in (target.get(field), *values)
            if _present(value)
        ]
        if not observed:
            return
        if len({_canonical(value) for value in observed}) != 1:
            conflicts.append(field)
            return
        target[field] = copy.deepcopy(observed[0])

    _project('full_command_contract', full_contracts)
    _project(
        'quality_request_binding',
        [source.get('quality_request_binding') for source in top_sources]
        + [contract.get('quality_request_binding') for contract in full_contracts],
    )
    _project(
        'quality_request_binding_status',
        [source.get('quality_request_binding_status') for source in top_sources],
    )
    _project(
        'quality_request_binding_sha256',
        [source.get('quality_request_binding_sha256') for source in top_sources]
        + [contract.get('quality_request_binding_sha256') for contract in full_contracts]
        + [binding.get('binding_sha256') for binding in bindings],
    )
    _project(
        'quality_request_binding_set_sha256',
        [source.get('quality_request_binding_set_sha256') for source in top_sources]
        + [contract.get('quality_request_binding_set_sha256') for contract in full_contracts],
    )

    claim_fields = (
        'source_request_sha256',
        'model_sha256',
        'validation_dataset_sha256',
        'validation_dataset_image_ids_sha256',
        'validation_dataset_ground_truth_sha256',
        'task_quality_policy_sha256',
        'quality_contract_sha256',
        'preprocessing_contract_sha256',
        'decoder_contract_sha256',
        'nms_contract_sha256',
        'quality_record_endpoint_contract_sha256',
        'central_quality_result_sha256',
        'prepared_input_evidence_sha256',
    )
    for field in claim_fields:
        _project(
            field,
            [source.get(field) for source in top_sources]
            + [binding.get(field) for binding in bindings],
        )

    _project(
        'quality_source_run_id',
        [source.get('quality_source_run_id') for source in top_sources]
        + [binding.get('source_run_id') for binding in bindings],
    )
    _project(
        'quality_source_case_id',
        [source.get('quality_source_case_id') for source in top_sources]
        + [binding.get('source_case_id') for binding in bindings],
    )
    _project(
        'preprocessing_contract',
        [source.get('preprocessing_contract') for source in top_sources]
        + [
            preprocessing.get('identity')
            for binding in bindings
            for preprocessing in [binding.get('preprocessing_contract')]
            if isinstance(preprocessing, Mapping)
        ],
    )
    _project(
        'runtime_precision_identity',
        [source.get('runtime_precision_identity') for source in top_sources]
        + [binding.get('runtime_precision_identity') for binding in bindings],
    )
    _project(
        'runtime_preprocessing_identity',
        [source.get('runtime_preprocessing_identity') for source in top_sources]
        + [binding.get('identity') for binding in runtime_bindings]
        + [
            preprocessing.get('identity')
            for binding in bindings
            for preprocessing in [binding.get('preprocessing_contract')]
            if isinstance(preprocessing, Mapping)
        ],
    )
    _project(
        'runtime_preprocessing_sha256',
        [source.get('runtime_preprocessing_sha256') for source in top_sources]
        + [binding.get('sha256') for binding in runtime_bindings]
        + [binding.get('preprocessing_contract_sha256') for binding in bindings]
        + [
            preprocessing.get('sha256')
            for binding in bindings
            for preprocessing in [binding.get('preprocessing_contract')]
            if isinstance(preprocessing, Mapping)
        ],
    )
    _project(
        'runtime_numeric_input_identity',
        [source.get('runtime_numeric_input_identity') for source in top_sources]
        + [binding.get('runtime_numeric_input_identity') for binding in runtime_bindings],
    )
    _project(
        'runtime_numeric_input_sha256',
        [source.get('runtime_numeric_input_sha256') for source in top_sources]
        + [binding.get('runtime_numeric_input_sha256') for binding in runtime_bindings],
    )

    if conflicts:
        conflict_fields = sorted(set(conflicts))
        target.update({
            'vendor_full_quality_provenance_conflict': True,
            'vendor_full_quality_provenance_conflict_fields': conflict_fields,
            'quality_request_binding_status': 'conflicting_duplicate_evidence',
            'quality_request_binding_errors': [
                f'vendor_full_quality_provenance_conflict:{field}'
                for field in conflict_fields
            ],
        })


def _enforce_historical_split_diagnostic_only(row: dict[str, Any]) -> None:
    if row.get('native_split_quality_authority_mode') != 'legacy':
        return
    reason = 'native_split_legacy_historical_diagnostic_only'
    row.update({
        'native_split_quality_legacy_status': 'historical_diagnostic_only',
        'execution_role': 'legacy_manual_diagnostic',
        'performance_claims_emitted': False,
        'claim_ok': False,
        'ok': False,
        'eligible_for_ranking': False,
        'performance_claim_eligible': False,
        'energy_claim_eligible': False,
        'pareto_eligible': False,
        'thesis_claim_eligible': False,
        'ranking_eligible': False,
        'performance_eligible': False,
        'energy_eligible': False,
        'thesis_valid': False,
        'status': 'historical_diagnostic_only',
        'gate_status': 'excluded_historical_diagnostic_only',
        'ranking_exclusion_reason': reason,
    })
    reasons = list(row.get('performance_claim_exclusion_reasons') or [])
    if reason not in reasons:
        reasons.append(reason)
    row['performance_claim_exclusion_reasons'] = reasons


def _find_reference_report(eval_root: Path, model: str, case: str, task: str) -> Path | None:
    model_root = eval_root / 'models' / model
    cases: list[str] = []
    if case and case != 'full':
        cases.append(case)
    # Full-native baselines do not have a native case; use first available case.
    if not cases:
        cr = model_root / 'benchmark_results' / 'remote_diagnostics' / 'case_reports' / 'results'
        if cr.exists():
            cases = [p.name for p in sorted(cr.iterdir()) if p.is_dir() and p.name.startswith('b')]
    for c in cases:
        for variant in ('results_ort_cpu', 'results_cpu', 'results_ort_tensorrt'):
            p = model_root / 'benchmark_results' / 'remote_diagnostics' / 'case_reports' / 'results' / c / variant / 'validation_report.json'
            if p.is_file():
                return p
        xs = sorted(model_root.glob(f'**/{c}/results_ort_cpu/validation_report.json'))
        if xs:
            return xs[0]
    xs = sorted(model_root.glob('**/results_ort_cpu/validation_report.json'))
    return xs[0] if xs else None



def _find_boundary_manifest_for_output(output_manifest: Path, roots: list[Path] | None = None, native_report: Path | None = None) -> Path | None:
    """Return the native boundary manifest matching a native FIFO output manifest.

    v59dt hardening: the local validation step may validate a copied/rebased
    output manifest.  In that case the sibling boundary directory can be missing
    or the JSON report may still contain the original remote absolute path.  We
    therefore resolve both direct paths and suffix-rebased paths under the
    validator roots before giving up.
    """
    roots = [Path(r).expanduser().resolve() for r in (roots or []) if str(r or '').strip()]
    output_text = str(output_manifest or '').replace('\\', '/').lower()
    if (
        '/native_full_outputs/' in output_text
        or Path(output_text).name.startswith('native_full_outputs_manifest')
    ):
        # Native-Full outputs have no split boundary.  A nearby/rglob split
        # boundary belongs to a different execution row and is never evidence
        # for this manifest.
        return None

    def _try_path(x: Any) -> Path | None:
        try:
            if not isinstance(x, (str, Path)) or not str(x).strip():
                return None
            q = Path(str(x)).expanduser()
            if q.is_file():
                return q.resolve()
            parts = list(q.parts)
            # Rebase absolute remote/local paths by stable suffixes that exist
            # inside copied native_producers/<backend> roots.
            for marker in ('native_fifo_boundary', 'native_full_outputs', 'native_pipeline', 'benchmark_set'):
                if marker in parts:
                    suf = Path(*parts[parts.index(marker):])
                    for r in roots:
                        cand = r / suf
                        if cand.is_file():
                            return cand.resolve()
                        # If the root is higher/lower than expected, allow rglob
                        # on the suffix string as a robust but scoped fallback.
                        for hit in r.rglob(str(suf)):
                            if hit.is_file():
                                return hit.resolve()
        except Exception:
            return None
        return None

    try:
        p = Path(output_manifest).expanduser().resolve()
        # .../<precision>/native_fifo_outputs/native_fifo_outputs_manifest.json
        cand = p.parent.parent / 'native_fifo_boundary' / 'native_fifo_boundary_manifest.json'
        if cand.is_file():
            return cand.resolve()

        # Some manifests/results record the boundary manifest explicitly.
        for src in (output_manifest, native_report):
            try:
                j = _load_json(Path(src)) if src else None
            except Exception:
                j = None
            if isinstance(j, dict):
                for k in ('native_fifo_boundary_manifest', 'boundary_manifest', 'native_boundary_manifest'):
                    hit = _try_path(j.get(k))
                    if hit:
                        return hit

        # Nearby fallbacks for copied/rebased roots.
        for name in ('native_fifo_boundary_manifest.json', 'native_boundary_manifest.json'):
            xs = sorted(p.parent.parent.rglob(name)) if p.parent.parent.exists() else []
            if xs:
                return xs[0].resolve()

        # Last scoped fallback: infer case/backend/precision from the output
        # manifest suffix and search matching copied roots.
        parts = list(p.parts)
        if 'native_pipeline' in parts:
            suf = Path(*parts[parts.index('native_pipeline'):])
            # Replace native_fifo_outputs/... by native_fifo_boundary/manifest.
            sparts = list(suf.parts)
            if 'native_fifo_outputs' in sparts or 'native_outputs' in sparts:
                i = sparts.index('native_fifo_outputs') if 'native_fifo_outputs' in sparts else sparts.index('native_outputs')
                bparts = sparts[:i] + ['native_fifo_boundary', 'native_fifo_boundary_manifest.json']
                bsuf = Path(*bparts)
                for r in roots:
                    cand = r / bsuf
                    if cand.is_file():
                        return cand.resolve()
                    for hit in r.rglob(str(bsuf)):
                        if hit.is_file():
                            return hit.resolve()
    except Exception:
        pass
    return None


def _benchmark_set_relative_path(value: Any) -> Path | None:
    """Return one traversal-free path below a recorded BenchmarkSet root."""
    raw = str(value or '').strip()
    if not raw:
        return None
    try:
        parts = list(Path(raw).expanduser().parts)
    except (OSError, RuntimeError, ValueError):
        return None
    indexes = [index for index, part in enumerate(parts) if part == 'benchmark_set']
    if len(indexes) != 1:
        return None
    relative_parts = parts[indexes[0] + 1:]
    if (
        not relative_parts
        or any(part in {'', '.', '..'} for part in relative_parts)
    ):
        return None
    relative = Path(*relative_parts)
    return None if relative.is_absolute() else relative


def _verified_full_command_input_manifest_binding(
    endpoint_evidence: Mapping[str, Any] | None,
    output_payload: Mapping[str, Any],
) -> tuple[Path, str] | None:
    """Verify the producer-sealed input-manifest path/hash mirror.

    Hailo Full v2 dumps written before collection did not repeat
    ``input_manifest_sha256`` in the output manifest.  The same path and digest
    are nevertheless sealed inside the hashed Full command contract.  Admit
    that mirror only after independently checking the complete contract hash,
    its duplicate top-level hash, row identity and portable BenchmarkSet path.
    """
    evidence = endpoint_evidence if isinstance(endpoint_evidence, Mapping) else {}
    raw_contract = evidence.get('full_command_contract')
    if not isinstance(raw_contract, Mapping):
        return None
    contract = dict(raw_contract)
    declared_contract_sha = _normalize_sha256(
        contract.pop('contract_sha256', '')
    )
    if (
        not declared_contract_sha
        or _canonical_json_sha256(contract) != declared_contract_sha
        or _normalize_sha256(
            evidence.get('full_command_contract_sha256')
        ) != declared_contract_sha
        or raw_contract.get('schema')
        != 'onnx-splitpoint/native-full-command-contract'
        or int(raw_contract.get('schema_version') or 0) < 1
        or raw_contract.get('complete') is not True
    ):
        return None
    for key in ('backend', 'model', 'setup_id', 'comparison_backend'):
        if (
            str(raw_contract.get(key) or '').strip()
            != str(output_payload.get(key) or '').strip()
        ):
            return None
    if (
        str(raw_contract.get('case') or '').strip() != 'full'
        or str(output_payload.get('case') or '').strip() != 'full'
    ):
        return None
    artifacts = raw_contract.get('artifacts')
    artifacts = artifacts if isinstance(artifacts, Mapping) else {}
    input_artifact = artifacts.get('input_manifest')
    input_artifact = (
        input_artifact if isinstance(input_artifact, Mapping) else {}
    )
    relative = _benchmark_set_relative_path(input_artifact.get('path'))
    digest = _normalize_sha256(input_artifact.get('sha256'))
    if relative is None or not digest:
        return None
    return relative, digest


def _find_native_full_input_manifest_for_output(
    output_manifest: Path,
    *,
    endpoint_evidence: Mapping[str, Any] | None = None,
) -> Path | None:
    """Return the exact sibling Native-Full input contract for an output dump."""
    try:
        output_path = Path(output_manifest).expanduser().resolve()
        output_payload = _load_json(output_path)
        if not isinstance(output_payload, dict):
            return None
        if (
            str(output_payload.get('case') or '').strip() != 'full'
            or str(output_payload.get('execution_mode') or '').strip()
            != 'native_full_baseline'
        ):
            return None
        command_binding = _verified_full_command_input_manifest_binding(
            endpoint_evidence, output_payload,
        )
        command_contract_present = bool(
            isinstance(endpoint_evidence, Mapping)
            and endpoint_evidence.get('full_command_contract') is not None
        )
        if command_contract_present and command_binding is None:
            return None
        candidates = [output_path.parent / 'native_full_input_manifest.json']
        for key in (
            'input_manifest', 'native_full_input_manifest',
            'boundary_manifest', 'native_boundary_manifest',
        ):
            raw = str(output_payload.get(key) or '').strip()
            if raw:
                candidates.append(output_path.parent / Path(raw).name)
        seen: set[str] = set()
        for candidate in candidates:
            candidate = candidate.expanduser().resolve()
            if str(candidate) in seen or not candidate.is_file():
                continue
            seen.add(str(candidate))
            if candidate.parent != output_path.parent:
                continue
            payload = _load_json(candidate)
            if not isinstance(payload, dict):
                continue
            if (
                payload.get('schema')
                != 'onnx-splitpoint/native-full-input-dump'
                or int(payload.get('schema_version') or 0) not in {1, 2}
                or str(payload.get('case') or '').strip() != 'full'
            ):
                continue
            if any(
                str(payload.get(key) or '').strip()
                != str(output_payload.get(key) or '').strip()
                for key in (
                    'backend', 'model', 'setup_id', 'comparison_backend',
                )
            ):
                continue
            declared_manifest_sha = _normalize_sha256(
                output_payload.get('input_manifest_sha256')
            )
            candidate_relative = _benchmark_set_relative_path(candidate)
            output_relative = _benchmark_set_relative_path(
                output_payload.get('input_manifest')
                or output_payload.get('native_full_input_manifest')
                or output_payload.get('boundary_manifest')
            )
            command_relative = command_binding[0] if command_binding else None
            command_sha = command_binding[1] if command_binding else ''
            if (
                candidate_relative is None
                or output_relative is None
                or candidate_relative != output_relative
                or (
                    command_binding is not None
                    and command_relative != candidate_relative
                )
                or (
                    declared_manifest_sha and command_sha
                    and declared_manifest_sha != command_sha
                )
            ):
                continue
            expected_manifest_sha = declared_manifest_sha or command_sha
            if (
                int(payload.get('schema_version') or 0) >= 2
                and (
                    not expected_manifest_sha
                    or _sha256_file(candidate) != expected_manifest_sha
                )
            ):
                continue
            input_image_sha = _normalize_sha256(
                payload.get('input_image_sha256')
                or payload.get('image_sha256')
            )
            output_image_sha = _normalize_sha256(
                output_payload.get('input_image_sha256')
            )
            if not input_image_sha or input_image_sha != output_image_sha:
                continue
            runtime_path = candidate.parent / Path(
                str(payload.get('runtime_input_file') or '')
            ).name
            if not str(payload.get('runtime_input_name') or '').strip():
                continue
            runtime_sha = _normalize_sha256(
                payload.get('runtime_input_sha256')
            )
            if (
                not runtime_path.is_file()
                or not runtime_sha
                or _sha256_file(runtime_path) != runtime_sha
                or int(payload.get('runtime_input_bytes') or 0)
                != int(runtime_path.stat().st_size)
            ):
                continue
            try:
                runtime_shape = [
                    int(value)
                    for value in list(payload.get('runtime_input_shape') or [])
                ]
                runtime_dtype = np.dtype(
                    str(payload.get('runtime_input_dtype') or '')
                )
                expected_runtime_bytes = int(
                    np.prod(runtime_shape, dtype=np.int64)
                ) * int(runtime_dtype.itemsize)
            except Exception:
                continue
            if (
                not runtime_shape
                or any(value <= 0 for value in runtime_shape)
                or expected_runtime_bytes != int(runtime_path.stat().st_size)
            ):
                continue
            input_dump = candidate.parent / Path(
                str(payload.get('input_dump') or '')
            ).name
            if not input_dump.is_file():
                continue
            if int(payload.get('schema_version') or 0) >= 2:
                dump_sha = _normalize_sha256(
                    payload.get('input_dump_sha256')
                )
                if (
                    not dump_sha
                    or _sha256_file(input_dump) != dump_sha
                    or int(payload.get('input_dump_bytes') or 0)
                    != int(input_dump.stat().st_size)
                ):
                    continue
            return candidate
    except Exception:
        return None
    return None


def _find_self_reference_input_manifest(
    output_manifest: Path,
    *,
    roots: list[Path] | None = None,
    native_report: Path | None = None,
    endpoint_evidence: Mapping[str, Any] | None = None,
) -> tuple[Path | None, str, str]:
    output_payload = _load_json(Path(output_manifest))
    output_payload = (
        output_payload if isinstance(output_payload, dict) else {}
    )
    case = str(output_payload.get('case') or '').strip().lower()
    execution_mode = str(
        output_payload.get('execution_mode') or ''
    ).strip().lower()
    backend = str(output_payload.get('backend') or '').strip().lower()
    full_markers = (
        case == 'full',
        execution_mode == 'native_full_baseline',
        backend.startswith('native_full_'),
    )
    if any(full_markers) and not all(full_markers):
        return None, 'invalid', 'native_execution_identity_conflict'
    if all(full_markers):
        manifest = _find_native_full_input_manifest_for_output(
            output_manifest,
            endpoint_evidence=endpoint_evidence,
        )
        return (
            manifest,
            'native_full_input_manifest',
            '' if manifest else 'native_full_input_manifest_missing_or_invalid',
        )
    manifest = _find_boundary_manifest_for_output(
        output_manifest, roots=roots, native_report=native_report,
    )
    return (
        manifest, 'split_boundary_manifest',
        '' if manifest else 'boundary_manifest_missing',
    )


def _find_full_onnx_for_native_manifest(output_manifest: Path, eval_root: Path, roots: list[Path] | None = None) -> Path | None:
    """Find the full model ONNX near the benchmark_set of a native dump."""
    try:
        p = Path(output_manifest).expanduser().resolve()
        for anc in [p.parent, *p.parents]:
            models = anc / 'models'
            if models.is_dir():
                xs = sorted([x for x in models.glob('*.onnx') if 'part1' not in x.name.lower() and 'part2' not in x.name.lower()])
                ys = [x for x in xs if 'yolo' in x.name.lower()]
                return (ys or xs)[0] if (ys or xs) else None
            if anc.name == 'benchmark_set':
                models = anc / 'models'
                if models.is_dir():
                    xs = sorted([x for x in models.glob('*.onnx') if 'part1' not in x.name.lower() and 'part2' not in x.name.lower()])
                    ys = [x for x in xs if 'yolo' in x.name.lower()]
                    return (ys or xs)[0] if (ys or xs) else None
    except Exception:
        pass
    try:
        xs = sorted([x for x in eval_root.glob('**/benchmark_set/models/*.onnx') if 'part1' not in x.name.lower() and 'part2' not in x.name.lower()])
        ys = [x for x in xs if 'yolo' in x.name.lower()]
        if ys or xs:
            return (ys or xs)[0]
    except Exception:
        pass
    try:
        for r in [Path(x).expanduser().resolve() for x in (roots or []) if str(x or '').strip()]:
            xs = sorted([x for x in r.glob('**/benchmark_set/models/*.onnx') if 'part1' not in x.name.lower() and 'part2' not in x.name.lower()])
            ys = [x for x in xs if 'yolo' in x.name.lower()]
            if ys or xs:
                return (ys or xs)[0]
    except Exception:
        pass
    return None


def _ort_input_shape(shape_obj: Any) -> list[int]:
    out=[]
    for x in shape_obj:
        try:
            out.append(int(x))
        except Exception:
            out.append(-1)
    return out


def _ort_input_dtype(inp: Any):
    t = str(getattr(inp, 'type', '') or '').lower()
    if 'float16' in t:
        return np.float16
    if 'float' in t:
        return np.float32
    if 'uint8' in t:
        return np.uint8
    return None


def _native_input_dump_feed(
    boundary_manifest: Path,
    input_shape: list[int],
    image_scale: str = 'native',
    *,
    target_dtype: Any | None = None,
    evidence: dict[str, Any] | None = None,
) -> np.ndarray | None:
    """Derive the ONNX feed from the exact sealed Native input bytes.

    Native Full vendor runtimes legitimately consume rank-3 HWC ``uint8``
    tensors.  ONNX self-reference models normally consume rank-4 NCHW floats.
    The two representations must therefore be kept separate: this helper
    verifies the sealed runtime tensor first and then permits only an explicit,
    manifest-declared HWC/CHW-to-batched-ONNX conversion.  It never reloads the
    source image and never applies a heuristic reshape.
    """
    if evidence is not None:
        evidence.clear()

    def _reject(reason: str) -> None:
        if evidence is not None:
            evidence.update({
                'status': 'unavailable',
                'reason': str(reason),
                'transformation_id': 'sealed_runtime_to_onnx_reference_v1',
            })

    def _shape_matches(expected: list[int], observed: Sequence[int]) -> bool:
        if len(expected) != len(observed):
            return False
        return all(
            int(declared) <= 0 or int(declared) == int(actual)
            for declared, actual in zip(expected, observed)
        )

    man = _load_json(boundary_manifest) or {}
    if man.get('schema') == 'onnx-splitpoint/native-full-input-dump':
        runtime_file = str(man.get('runtime_input_file') or '').strip()
        runtime_sha = _normalize_sha256(man.get('runtime_input_sha256'))
        runtime_shape_raw = man.get('runtime_input_shape')
        runtime_dtype_raw = str(man.get('runtime_input_dtype') or '').strip()
        if (
            not runtime_file or not runtime_sha
            or not isinstance(runtime_shape_raw, list)
            or not runtime_shape_raw or not runtime_dtype_raw
        ):
            _reject('sealed_runtime_manifest_fields_missing')
            return None
        runtime_path = Path(runtime_file).expanduser()
        if not runtime_path.is_absolute():
            runtime_path = (Path(boundary_manifest).parent / runtime_path).resolve()
        if not runtime_path.is_file():
            sibling = Path(boundary_manifest).parent / runtime_path.name
            runtime_path = sibling if sibling.is_file() else runtime_path
        try:
            runtime_shape = [int(value) for value in runtime_shape_raw]
            runtime_dtype = np.dtype(runtime_dtype_raw)
            expected_bytes = int(
                np.prod(runtime_shape, dtype=np.int64)
            ) * int(runtime_dtype.itemsize)
            if (
                any(value <= 0 for value in runtime_shape)
                or not runtime_path.is_file()
                or _sha256_file(runtime_path) != runtime_sha
                or int(man.get('runtime_input_bytes') or 0) != expected_bytes
                or int(runtime_path.stat().st_size) != expected_bytes
            ):
                _reject('sealed_runtime_tensor_binding_invalid')
                return None
            runtime = np.fromfile(
                str(runtime_path), dtype=runtime_dtype,
            ).reshape(runtime_shape)
            target = np.dtype(target_dtype) if target_dtype is not None else runtime_dtype
            preprocess = man.get('preprocess') if isinstance(
                man.get('preprocess'), Mapping,
            ) else {}
            declared_layouts = {
                str(value or '').strip().upper()
                for value in (
                    man.get('runtime_input_layout'),
                    preprocess.get('layout'),
                )
                if str(value or '').strip()
            }
            if len(declared_layouts) != 1:
                _reject(
                    'sealed_runtime_layout_missing'
                    if not declared_layouts
                    else 'sealed_runtime_layout_conflict'
                )
                return None
            source_layout = next(iter(declared_layouts))
            requested_scale = str(image_scale or 'native').strip().lower()
            manifest_scale = str(
                preprocess.get('ort_model_scale') or '',
            ).strip().lower()
            if requested_scale == 'native':
                requested_scale = manifest_scale

            if len(runtime_shape) == len(input_shape):
                if not _shape_matches(input_shape, runtime_shape):
                    _reject('sealed_runtime_onnx_shape_mismatch')
                    return None
                if runtime_dtype != target:
                    _reject('sealed_runtime_identity_dtype_mismatch')
                    return None
                if len(runtime_shape) == 4:
                    target_layout = (
                        'NCHW' if int(input_shape[1]) in (1, 3)
                        and int(input_shape[-1]) not in (1, 3)
                        else 'NHWC' if int(input_shape[-1]) in (1, 3)
                        and int(input_shape[1]) not in (1, 3)
                        else ''
                    )
                    if not target_layout or source_layout != target_layout:
                        _reject('sealed_runtime_rank4_layout_mismatch')
                        return None
                feed = np.asarray(runtime, dtype=target)
                transformation = 'sealed_runtime_identity_v1'
                target_layout = source_layout
            elif len(runtime_shape) == 3 and len(input_shape) == 4:
                if target not in {
                    np.dtype(np.float16), np.dtype(np.float32),
                }:
                    _reject('sealed_runtime_reference_dtype_unsupported')
                    return None
                if int(input_shape[0]) > 0 and int(input_shape[0]) != 1:
                    _reject('sealed_runtime_reference_batch_not_one')
                    return None
                if (
                    not requested_scale
                    or not manifest_scale
                    or requested_scale != manifest_scale
                ):
                    _reject('sealed_runtime_onnx_scale_missing_or_conflicting')
                    return None
                target_is_nchw = bool(
                    int(input_shape[1]) in (1, 3)
                    and int(input_shape[-1]) not in (1, 3)
                )
                target_is_nhwc = bool(
                    int(input_shape[-1]) in (1, 3)
                    and int(input_shape[1]) not in (1, 3)
                )
                if target_is_nchw == target_is_nhwc:
                    _reject('sealed_runtime_reference_layout_ambiguous')
                    return None
                target_layout = 'NCHW' if target_is_nchw else 'NHWC'
                if source_layout == 'HWC':
                    if int(runtime_shape[-1]) not in (1, 3):
                        _reject('sealed_runtime_hwc_channel_invalid')
                        return None
                    work = runtime.astype(np.float32)
                elif source_layout == 'CHW':
                    if int(runtime_shape[0]) not in (1, 3):
                        _reject('sealed_runtime_chw_channel_invalid')
                        return None
                    work = np.transpose(runtime, (1, 2, 0)).astype(np.float32)
                else:
                    _reject('sealed_runtime_rank3_layout_unsupported')
                    return None
                if runtime_dtype != np.dtype(np.uint8):
                    _reject('sealed_runtime_rank3_dtype_unsupported')
                    return None
                if requested_scale in {'norm', 'normalized', 'scale_0_1'}:
                    work /= 255.0
                elif requested_scale in {
                    'imagenet', 'imagenet_mean_std', 'torchvision',
                    'imagenet_standard',
                }:
                    work /= 255.0
                    if int(work.shape[-1]) != 3:
                        _reject('sealed_runtime_imagenet_channel_invalid')
                        return None
                    mean = np.asarray(
                        [0.485, 0.456, 0.406], dtype=np.float32,
                    ).reshape(1, 1, 3)
                    std = np.asarray(
                        [0.229, 0.224, 0.225], dtype=np.float32,
                    ).reshape(1, 1, 3)
                    work = (work - mean) / std
                else:
                    _reject('sealed_runtime_onnx_scale_unsupported')
                    return None
                feed = (
                    np.transpose(work, (2, 0, 1))[None]
                    if target_layout == 'NCHW'
                    else work[None]
                ).astype(target)
                transformation = 'sealed_runtime_to_onnx_reference_v1'
            else:
                _reject('sealed_runtime_rank_conversion_unsupported')
                return None

            feed = np.ascontiguousarray(feed)
            if (
                not _shape_matches(input_shape, feed.shape)
                or feed.dtype != target
                or (
                    feed.dtype.kind in 'fc'
                    and not bool(np.isfinite(feed).all())
                )
            ):
                _reject('sealed_runtime_derived_reference_invalid')
                return None
            if evidence is not None:
                evidence.update({
                    'status': 'passed',
                    'reason': '',
                    'transformation_id': transformation,
                    'source_runtime_input_sha256': runtime_sha,
                    'source_shape': [int(value) for value in runtime_shape],
                    'source_layout': source_layout,
                    'source_dtype': str(runtime_dtype),
                    'target_shape': [int(value) for value in feed.shape],
                    'target_layout': target_layout,
                    'target_dtype': str(feed.dtype),
                    'ort_model_scale': requested_scale,
                    'derived_reference_tensor_sha256': hashlib.sha256(
                        feed.tobytes(order='C'),
                    ).hexdigest(),
                    'c_contiguous': bool(feed.flags.c_contiguous),
                })
            return feed
        except Exception as exc:
            _reject(f'sealed_runtime_reference_exception:{type(exc).__name__}')
            return None
    p = man.get('input_dump') or man.get('preprocessed_input_file')
    if not isinstance(p, str) or not p:
        _reject('legacy_input_dump_missing')
        return None
    pp = Path(p).expanduser()
    if not pp.is_absolute():
        base = Path(str(man.get('file') or boundary_manifest)).expanduser().parent
        pp = (base / pp).resolve()
    if not pp.is_file():
        # Rebased/copy fallback: use sibling input dump.
        cand = boundary_manifest.parent / pp.name
        if cand.is_file():
            pp = cand
        else:
            _reject('legacy_input_dump_file_missing')
            return None
    shp = man.get('input_shape_hwc') or (man.get('preprocess') or {}).get('input_shape_hwc') or []
    if not shp and len(input_shape) == 4:
        if input_shape[1] in (1,3):
            shp = [int(input_shape[2]), int(input_shape[3]), int(input_shape[1])]
        elif input_shape[-1] in (1,3):
            shp = [int(input_shape[1]), int(input_shape[2]), int(input_shape[3])]
    if not isinstance(shp, list) or len(shp) != 3:
        _reject('legacy_input_shape_invalid')
        return None
    h,w,c = [int(x) for x in shp]
    raw = np.fromfile(str(pp), dtype=np.uint8)
    if raw.size != h*w*c:
        _reject('legacy_input_dump_size_mismatch')
        return None
    arr = raw.reshape((h,w,c)).astype(np.float32)
    mode = str(image_scale or 'native').strip().lower()
    if mode == 'native':
        mode = str((man.get('preprocess') or {}).get('ort_model_scale') or 'norm').lower()
    normalize = mode not in {'raw','native_raw','resize_raw','letterbox0_raw','letterbox114_raw','yolo_raw','yolo_letterbox_raw'}
    if normalize:
        arr /= 255.0
    if mode in {'imagenet', 'imagenet_mean_std', 'torchvision', 'imagenet_standard'} and arr.shape[-1] == 3:
        mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
        arr = (arr - mean) / std
    if len(input_shape)==4 and input_shape[1] in (1,3):
        feed = np.transpose(arr,(2,0,1))[None].astype(
            np.dtype(target_dtype or np.float32),
        )
    elif len(input_shape)==4 and input_shape[-1] in (1,3):
        feed = arr[None].astype(np.dtype(target_dtype or np.float32))
    else:
        _reject('legacy_reference_layout_unsupported')
        return None
    feed = np.ascontiguousarray(feed)
    if not _shape_matches(input_shape, feed.shape):
        _reject('legacy_reference_shape_mismatch')
        return None
    if evidence is not None:
        evidence.update({
            'status': 'passed',
            'reason': '',
            'transformation_id': 'legacy_image_dump_to_onnx_reference_v1',
            'source_shape': [h, w, c],
            'source_layout': 'HWC',
            'source_dtype': 'uint8',
            'target_shape': [int(value) for value in feed.shape],
            'target_layout': (
                'NCHW' if int(input_shape[1]) in (1, 3) else 'NHWC'
            ),
            'target_dtype': str(feed.dtype),
            'ort_model_scale': mode,
            'derived_reference_tensor_sha256': hashlib.sha256(
                feed.tobytes(order='C'),
            ).hexdigest(),
            'c_contiguous': bool(feed.flags.c_contiguous),
        })
    return feed


def _detection_contract_family(mode: str) -> str:
    text = str(mode or "").strip().lower()
    if (
        ":nms:" in text
        or "decoded" in text
        or "detections" in text
        or "frozen_host_tail" in text
        or "frozen_normalization" in text
        or "detection_completion_execution" in text
    ):
        return "decoded_nms"
    if text.endswith(":raw") or "raw_head" in text or "output0:raw" in text:
        return "raw_head"
    return "unknown"


def _contract_text_values(payload: Any, *, depth: int = 0) -> list[str]:
    if depth > 4:
        return []
    if isinstance(payload, dict):
        values: list[str] = []
        for key, value in payload.items():
            key_low = str(key).lower()
            if any(token in key_low for token in ("contract", "decoder", "output_format", "postprocess", "nms", "raw")):
                values.append(str(value))
            values.extend(_contract_text_values(value, depth=depth + 1))
        return values
    if isinstance(payload, list):
        values: list[str] = []
        for value in payload[:50]:
            values.extend(_contract_text_values(value, depth=depth + 1))
        return values
    return []


def _completed_v2_declared(payload: Mapping[str, Any] | None) -> bool:
    return bool(
        isinstance(payload, Mapping)
        and any(
            payload.get(field) not in (None, "", {}, [])
            for field in (
                "completed_task_endpoint_attestation",
                "completed_task_comparison_endpoint_contract",
                "completed_task_comparison_endpoint_contract_hash",
                "completed_task_comparison_output_endpoint_id",
                "completed_task_completion_mode",
            )
        )
    )


def _verified_completed_v2_frozen_contract(
    endpoint_evidence: Mapping[str, Any],
    *,
    native_tensors: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the raw-tail and comparison contracts only after strict V2 checks."""
    if (
        _FrozenDetectionPostprocessor is None
        or _verify_frozen_postprocess_contract is None
        or _verify_completed_detection_comparison_endpoint_contract is None
    ):
        raise _FrozenPostprocessError(
            "completed_v2_postprocess_verifier_unavailable"
        )
    if not _completed_frozen_nms_attestation_passed(endpoint_evidence):
        raise _FrozenPostprocessError(
            "completed_v2_endpoint_attestation_invalid"
        )
    frozen_raw = endpoint_evidence.get("frozen_host_postprocess_contract")
    comparison_raw = endpoint_evidence.get(
        "completed_task_comparison_endpoint_contract"
    )
    frozen = _verify_frozen_postprocess_contract(
        frozen_raw,
        outputs=native_tensors,
    )
    comparison = (
        _verify_completed_detection_comparison_endpoint_contract(
            comparison_raw,
            frozen_contract=frozen,
        )
    )
    expected_hash = _normalize_sha256(
        endpoint_evidence.get(
            "completed_task_comparison_endpoint_contract_hash"
        )
    )
    expected_id = str(
        endpoint_evidence.get(
            "completed_task_comparison_output_endpoint_id"
        ) or ""
    )
    if (
        expected_hash
        != _normalize_sha256(comparison.get("endpoint_contract_hash"))
        or expected_id != str(comparison.get("output_endpoint_id") or "")
    ):
        raise _FrozenPostprocessError(
            "completed_v2_comparison_identity_mismatch"
        )
    return dict(frozen), dict(comparison)


def _verified_completed_v2_direct_contract(
    endpoint_evidence: Mapping[str, Any],
    *,
    native_tensors: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a measured Direct-BN6 normalizer and its V2 endpoint.

    The physical BN6 output remains bound to its runtime endpoint attestation.
    Only the separately measured, sealed normalization may project it onto the
    backend-independent completed-task comparison endpoint.
    """
    if (
        _FrozenDecodedNmsPostprocessor is None
        or _build_normalized_detection_endpoint_attestation is None
        or _verify_frozen_decoded_nms_normalization_contract is None
        or _verify_completed_detection_comparison_endpoint_contract is None
        or _canonical_detection_json_sha256 is None
    ):
        raise _FrozenPostprocessError(
            "completed_v2_direct_normalization_verifier_unavailable"
        )
    if str(
        endpoint_evidence.get("completed_task_completion_mode") or ""
    ).strip() != "integrated_accelerator_plus_frozen_normalization":
        raise _FrozenPostprocessError(
            "completed_v2_direct_completion_mode_invalid"
        )

    full_contract_raw = endpoint_evidence.get("full_command_contract")
    if not isinstance(full_contract_raw, Mapping):
        raise _FrozenPostprocessError(
            "completed_v2_direct_full_command_contract_missing"
        )
    full_contract_identity = dict(full_contract_raw)
    full_contract_sha = _normalize_sha256(
        full_contract_identity.pop("contract_sha256", "")
    )
    if (
        not full_contract_sha
        or _canonical_json_sha256(full_contract_identity)
        != full_contract_sha
        or (
            _normalize_sha256(
                endpoint_evidence.get("full_command_contract_sha256")
            )
            not in {"", full_contract_sha}
        )
    ):
        raise _FrozenPostprocessError(
            "completed_v2_direct_full_command_contract_invalid"
        )
    workload = full_contract_identity.get("energy_workload")
    if not isinstance(workload, Mapping):
        raise _FrozenPostprocessError(
            "completed_v2_direct_energy_workload_missing"
        )

    workload_contract = workload.get(
        "frozen_decoded_nms_normalization_contract"
    )
    top_level_contract = endpoint_evidence.get(
        "frozen_decoded_nms_normalization_contract"
    )
    if (
        top_level_contract not in (None, {})
        and (
            not isinstance(top_level_contract, Mapping)
            or dict(top_level_contract) != dict(workload_contract or {})
        )
    ):
        raise _FrozenPostprocessError(
            "completed_v2_direct_normalization_contract_conflict"
        )
    source_attestation = endpoint_evidence.get(
        "output_endpoint_attestation"
    )
    if not isinstance(source_attestation, Mapping):
        raise _FrozenPostprocessError(
            "completed_v2_direct_source_attestation_missing"
        )
    verify_kwargs: dict[str, Any] = {}
    if native_tensors is not None:
        verify_kwargs = {
            "outputs": native_tensors,
            "source_output_endpoint_attestation": source_attestation,
        }
    direct = _verify_frozen_decoded_nms_normalization_contract(
        workload_contract,
        **verify_kwargs,
    )
    source_hash = _normalize_sha256(
        direct.get("source_endpoint_contract_hash")
    )
    source_id = str(direct.get("source_output_endpoint_id") or "")
    physical_hashes = {
        value
        for value in (
            _normalize_sha256(
                endpoint_evidence.get("endpoint_contract_hash")
            ),
            _normalize_sha256(
                endpoint_evidence.get("physical_endpoint_contract_hash")
            ),
        )
        if value
    }
    physical_ids = {
        str(value).strip()
        for value in (
            endpoint_evidence.get("output_endpoint_id"),
            endpoint_evidence.get("physical_output_endpoint_id"),
        )
        if str(value or "").strip()
        and ":comparison:" not in str(value)
    }
    if (
        not source_hash
        or source_id != f"detection:decoded_nms:{source_hash}"
        or physical_hashes not in ({source_hash}, set())
        or physical_ids not in ({source_id}, set())
        or _canonical_detection_json_sha256(source_attestation)
        != _normalize_sha256(
            direct.get("source_output_endpoint_attestation_sha256")
        )
        or _normalize_sha256(
            workload.get(
                "frozen_decoded_nms_normalization_contract_sha256"
            )
        ) != _normalize_sha256(direct.get("contract_sha256"))
        or _normalize_sha256(
            workload.get("source_endpoint_contract_hash")
        ) != source_hash
        or str(workload.get("source_output_endpoint_id") or "")
        != source_id
        or dict(workload.get("source_output_tensor_signature") or {})
        != dict(direct.get("source_output_tensor_signature") or {})
    ):
        raise _FrozenPostprocessError(
            "completed_v2_direct_source_identity_mismatch"
        )

    completion = endpoint_evidence.get(
        "completed_task_endpoint_attestation"
    )
    if not isinstance(completion, Mapping):
        raise _FrozenPostprocessError(
            "completed_v2_direct_endpoint_attestation_missing"
        )
    expected_completion = (
        _build_normalized_detection_endpoint_attestation(
            direct,
            completion.get(
                "frozen_decoded_nms_normalization_result"
            ) or {},
            completed_frames=completion.get("completed_frames"),
            postprocess_completed_frames=completion.get(
                "postprocess_completed_frames"
            ),
            allow_legacy_hash_only_v1=True,
        )
    )
    comparison_raw = endpoint_evidence.get(
        "completed_task_comparison_endpoint_contract"
    )
    comparison = (
        _verify_completed_detection_comparison_endpoint_contract(
            comparison_raw,
            direct_normalization_contract=direct,
        )
    )
    expected_hash = _normalize_sha256(
        endpoint_evidence.get(
            "completed_task_comparison_endpoint_contract_hash"
        )
    )
    expected_id = str(
        endpoint_evidence.get(
            "completed_task_comparison_output_endpoint_id"
        ) or ""
    )
    if (
        dict(completion) != dict(expected_completion)
        or dict(
            workload.get("completed_task_endpoint_attestation") or {}
        ) != dict(expected_completion)
        or completion.get("attested") is not True
        or str(completion.get("status") or "").strip().lower()
        != "passed"
        or endpoint_evidence.get("completed_task_endpoint_attested")
        is not True
        or dict(
            completion.get(
                "completed_task_comparison_endpoint_contract"
            ) or {}
        ) != dict(comparison)
        or expected_hash
        != _normalize_sha256(comparison.get("endpoint_contract_hash"))
        or expected_id != str(comparison.get("output_endpoint_id") or "")
    ):
        raise _FrozenPostprocessError(
            "completed_v2_direct_completion_identity_mismatch"
        )
    return dict(direct), dict(comparison)


def _verified_completed_v2_execution_contract(
    endpoint_evidence: Mapping[str, Any],
    *, native_tensors: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if (
        _verify_detection_completion_execution_contract is None
        or _verify_detection_completion_execution_attestation is None
        or _verify_completed_detection_comparison_endpoint_contract is None
    ):
        raise _FrozenPostprocessError(
            "completed_v2_execution_verifier_unavailable"
        )
    native_command = endpoint_evidence.get("native_command_contract")
    runtime_options = (
        native_command.get("runtime_options")
        if isinstance(native_command, Mapping) else {}
    )
    raw_execution = (
        endpoint_evidence.get("completion_execution_contract")
        if isinstance(
            endpoint_evidence.get("completion_execution_contract"),
            Mapping,
        )
        else runtime_options.get("completion_execution_contract")
        if isinstance(runtime_options, Mapping)
        else None
    )
    execution = _verify_detection_completion_execution_contract(
        raw_execution
    )
    if endpoint_evidence.get("completed_task_completion_mode") == "native_three_stage_fast_oracle_outside_timing":
        from onnx_splitpoint_tool.native_three_stage import verify_fast_completion_attestation
        attestation = verify_fast_completion_attestation(
            endpoint_evidence.get("completed_task_endpoint_attestation"),
            execution_contract=execution, outputs=native_tensors,
        )
        if resolve_host_postprocess_evidence(endpoint_evidence).get("available") is not True:
            raise _FrozenPostprocessError("completed_v2_fast_oracle_projection_mismatch")
    else:
        attestation = (
            _verify_detection_completion_execution_attestation(
                endpoint_evidence.get(
                    "completed_task_endpoint_attestation"
                ),
                execution_contract=execution,
                expected_observation_relation="same_hotloop_sentinel",
            )
        )
    completed_endpoint = dict(
        execution.get("completed_endpoint_contract") or {}
    )
    comparison = (
        _verify_completed_detection_comparison_endpoint_contract(
            execution.get("comparison_endpoint_contract")
        )
    )
    source = dict(execution.get("source_endpoint") or {})
    if (
        attestation.get("exact_result_claim_bound") is not True
        or dict(
            endpoint_evidence.get(
                "completed_task_endpoint_contract"
            ) or {}
        ) != completed_endpoint
        or dict(
            endpoint_evidence.get(
                "completed_task_comparison_endpoint_contract"
            ) or {}
        ) != comparison
        or _normalize_sha256(
            endpoint_evidence.get("endpoint_contract_hash")
        ) != _normalize_sha256(
            source.get("endpoint_contract_hash")
        )
    ):
        raise _FrozenPostprocessError(
            "completed_v2_execution_projection_mismatch"
        )
    return dict(execution), dict(comparison)


def _verified_completed_v2_contract(
    endpoint_evidence: Mapping[str, Any],
    *,
    native_tensors: Mapping[str, Any] | None = None,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    mode = str(
        endpoint_evidence.get("completed_task_completion_mode") or ""
    ).strip()
    if mode == "frozen_host_tail":
        source, comparison = _verified_completed_v2_frozen_contract(
            endpoint_evidence,
            native_tensors=native_tensors,
        )
    elif mode == "integrated_accelerator_plus_frozen_normalization":
        source, comparison = _verified_completed_v2_direct_contract(
            endpoint_evidence,
            native_tensors=native_tensors,
        )
    elif mode in {"detection_completion_execution_v1", "native_three_stage_fast_oracle_outside_timing"}:
        source, comparison = (
            _verified_completed_v2_execution_contract(
                endpoint_evidence, native_tensors=native_tensors,
            )
        )
    else:
        raise _FrozenPostprocessError(
            "completed_v2_completion_mode_unsupported"
        )
    return mode, source, comparison


def _portable_semantic_screening_allowed(
    policy: AccuracyGatePolicy | None,
) -> bool:
    """Return whether a separate semantic replay may remain screening-only.

    A completed performance hotloop and its later semantic dump can share the
    exact frozen contract without sharing one completed-result hash.  That is
    useful development evidence, but it is not a final exact-result binding.
    Keep the exception deliberately narrow and explicit.
    """
    if policy is None:
        return False
    return bool(
        str(policy.dataset_tier or "").strip().lower() == "screening"
        and policy.frozen_before_final_campaign is not True
        and policy.screening_eligible_for_ranking is not True
        and policy.contract_only_eligible_for_ranking is not True
    )


def _verified_completed_v2_result_binding(
    native_result: Mapping[str, Any],
    sealed_result: Mapping[str, Any] | None,
    native_detections: Iterable[Mapping[str, Any]],
    *,
    policy: AccuracyGatePolicy | None,
) -> dict[str, Any]:
    """Verify local replay and classify its binding to the hotloop result.

    The local result must always be self-consistent.  A hash-only difference
    from a separately measured hotloop is admitted solely as portable
    screening evidence; every other identity drift and every final/claimable
    use remains fail-closed.  Exact hash equality proves completed-result
    identity; it deliberately does not claim that two artefacts came from one
    invocation.
    """
    if not isinstance(sealed_result, Mapping):
        raise _FrozenPostprocessError(
            "completed_v2_native_result_attestation_mismatch"
        )
    local_result = dict(native_result)
    hotloop_result = dict(sealed_result)
    local_hash = _normalize_sha256(
        local_result.get("detections_sha256")
    )
    replay_hash = _normalize_sha256(
        _canonical_detection_json_sha256(
            [dict(value) for value in native_detections]
        )
    )
    hotloop_hash = _normalize_sha256(
        hotloop_result.get("detections_sha256")
    )
    if not local_hash or not replay_hash or local_hash != replay_hash:
        raise _FrozenPostprocessError(
            "completed_v2_local_replay_self_consistency_mismatch"
        )
    if local_result == hotloop_result:
        return {
            "semantic_result_binding_status": (
                "exact_completed_result_identity_match"
            ),
            "exact_completed_result_identity_bound": True,
            "portable_result_hash_mismatch": False,
            "completed_v2_exact_result_claim_binding": True,
            "completed_v2_semantic_evidence_tier": (
                "exact_completed_result_identity"
            ),
            "performance_hotloop_result_sha256": hotloop_hash,
        }

    local_identity = dict(local_result)
    hotloop_identity = dict(hotloop_result)
    local_identity.pop("detections_sha256", None)
    hotloop_identity.pop("detections_sha256", None)
    hash_only_mismatch = bool(
        hotloop_hash
        and hotloop_hash != local_hash
        and local_identity == hotloop_identity
    )
    if not hash_only_mismatch:
        raise _FrozenPostprocessError(
            "completed_v2_native_result_attestation_mismatch"
        )
    if not _portable_semantic_screening_allowed(policy):
        raise _FrozenPostprocessError(
            "portable_result_hash_mismatch_exact_identity_required"
        )
    return {
        "semantic_result_binding_status": (
            "portable_result_hash_mismatch"
        ),
        "exact_completed_result_identity_bound": False,
        "portable_result_hash_mismatch": True,
        "completed_v2_exact_result_claim_binding": False,
        "completed_v2_semantic_evidence_tier": (
            "development_screening_portable_replay"
        ),
        "performance_hotloop_result_sha256": hotloop_hash,
    }


def _safe_collected_artifact_candidate(
    artifact_root: Path,
    relative: Path,
) -> Path | None:
    """Resolve an exact collected artifact below a non-symlink root."""
    try:
        root = Path(artifact_root).expanduser()
        if (
            root.name != 'benchmark_set'
            or root.is_symlink()
            or not root.is_dir()
            or relative.is_absolute()
            or any(part in {'', '.', '..'} for part in relative.parts)
        ):
            return None
        current = root
        for part in relative.parts:
            current = current / part
            if current.is_symlink():
                return None
        if not current.is_file():
            return None
        resolved_root = root.resolve(strict=True)
        resolved = current.resolve(strict=True)
        resolved.relative_to(resolved_root)
        return resolved
    except (OSError, RuntimeError, ValueError):
        return None


def _persisted_artifact_candidates(
    recorded_path: Path,
    *,
    native_report: Path | None,
    artifact_root: Path | None,
) -> list[Path]:
    """Map a producer-host absolute path onto its collected Artifact Root.

    Rsync preserves the tree below ``benchmark_set`` but cannot rewrite JSON
    strings recorded on the producer.  Only the exact traversal-free suffix is
    rebased; basename-wide or recursive searches are deliberately excluded.
    """
    candidates: list[Path] = []
    if recorded_path.is_absolute():
        candidates.append(recorded_path)

    report_path = Path(native_report).expanduser() if native_report else None
    derived_root = (
        _benchmark_set_for_report(report_path)
        if report_path is not None else None
    )
    selected_root: Path | None = None
    if artifact_root is not None:
        try:
            explicit_root = Path(artifact_root).expanduser().resolve(strict=True)
            if derived_root is not None and explicit_root != derived_root:
                return candidates
            selected_root = explicit_root
        except (OSError, RuntimeError):
            return candidates
    else:
        selected_root = derived_root

    recorded_relative = _benchmark_set_relative_path(recorded_path)
    if selected_root is not None and recorded_relative is not None:
        rebased = _safe_collected_artifact_candidate(
            selected_root, recorded_relative,
        )
        if rebased is not None:
            candidates.append(rebased)

    # Preserve the narrow legacy sibling case for archived reports whose old
    # absolute path predates the BenchmarkSet marker.  With modern paths the
    # portable parent suffix must match exactly.
    if report_path is not None and report_path.is_file():
        sibling_allowed = recorded_relative is None
        if recorded_relative is not None and derived_root is not None:
            report_relative = _benchmark_set_relative_path(report_path)
            sibling_allowed = bool(
                report_relative is not None
                and recorded_relative.parent == report_relative.parent
            )
        if sibling_allowed:
            candidates.append(
                report_path.resolve().parent / recorded_path.name
            )

    unique: dict[str, Path] = {}
    for candidate in candidates:
        try:
            key = str(candidate.resolve()) if candidate.exists() else str(candidate)
        except (OSError, RuntimeError):
            key = str(candidate)
        unique.setdefault(key, candidate)
    return list(unique.values())


def _verified_frozen_completed_result_artifact(
    sealed_result: Mapping[str, Any],
    endpoint_evidence: Mapping[str, Any],
    *,
    native_report: Path | None = None,
    artifact_root: Path | None = None,
) -> list[dict[str, Any]] | None:
    """Return an exact legacy-hotloop result, or ``None`` for hash-only V1.

    New Native-Full raw-head runners persist the canonical, inverse-letterbox
    detections that actually completed the measured sentinel.  Archived V1
    rows only contain a result hash and deliberately continue through the
    independent raw-head replay fallback.
    """
    raw_artifact = sealed_result.get("completed_result_artifact")
    if raw_artifact is None:
        return None
    if not isinstance(raw_artifact, Mapping):
        raise _FrozenPostprocessError(
            "completed_v2_hotloop_result_artifact_invalid"
        )
    artifact = dict(raw_artifact)
    raw_detections = artifact.get("detections")
    if not isinstance(raw_detections, list):
        raise _FrozenPostprocessError(
            "completed_v2_hotloop_result_artifact_invalid"
        )
    detections: list[dict[str, Any]] = []
    for raw_detection in raw_detections:
        if (
            not isinstance(raw_detection, Mapping)
            or set(raw_detection) != {
                "class_id", "score", "x1", "y1", "x2", "y2",
            }
        ):
            raise _FrozenPostprocessError(
                "completed_v2_hotloop_result_artifact_invalid"
            )
        try:
            class_float = float(raw_detection["class_id"])
            detection = {
                "class_id": int(class_float),
                "score": float(raw_detection["score"]),
                "x1": float(raw_detection["x1"]),
                "y1": float(raw_detection["y1"]),
                "x2": float(raw_detection["x2"]),
                "y2": float(raw_detection["y2"]),
            }
        except (TypeError, ValueError, OverflowError) as exc:
            raise _FrozenPostprocessError(
                "completed_v2_hotloop_result_artifact_invalid"
            ) from exc
        if (
            isinstance(raw_detection.get("class_id"), bool)
            or class_float != detection["class_id"]
            or any(
                not math.isfinite(float(detection[field]))
                for field in ("score", "x1", "y1", "x2", "y2")
            )
            or not 0.0 <= detection["score"] <= 1.0
            or detection["x2"] < detection["x1"]
            or detection["y2"] < detection["y1"]
        ):
            raise _FrozenPostprocessError(
                "completed_v2_hotloop_result_artifact_invalid"
            )
        detections.append(detection)
    canonical = sorted(
        detections,
        key=lambda value: (
            -float(value["score"]),
            int(value["class_id"]),
            float(value["x1"]),
            float(value["y1"]),
            float(value["x2"]),
            float(value["y2"]),
        ),
    )
    expected_artifact = {
        "schema": (
            "onnx-splitpoint/frozen-completed-detection-result-artifact"
        ),
        "schema_version": 1,
        "record_schema": "xyxy_score_class_id_v1",
        "coordinate_space": "original_image_xyxy_pixels",
        "sort_policy": (
            "score_desc_class_id_asc_xyxy_lexicographic_v1"
        ),
        "detections": canonical,
    }
    artifact_sha256 = _canonical_detection_json_sha256(
        expected_artifact
    )
    persisted_artifact = endpoint_evidence.get(
        "completed_task_result_artifact"
    )
    persisted_path_raw = str(
        endpoint_evidence.get("completed_task_result_artifact_path") or ""
    ).strip()
    persisted_path = Path(persisted_path_raw).expanduser()
    persisted_file_sha256 = _normalize_sha256(
        endpoint_evidence.get(
            "completed_task_result_artifact_file_sha256"
        )
    )
    persisted_payload: Any = None
    verified_persisted_path: Path | None = None
    persistence_candidates: list[Path] = []
    if persisted_path_raw and persisted_path.is_absolute():
        persistence_candidates = _persisted_artifact_candidates(
            persisted_path,
            native_report=native_report,
            artifact_root=artifact_root,
        )
    for candidate in persistence_candidates:
        try:
            if (
                not candidate.is_file()
                or candidate.is_symlink()
                or _sha256_file(candidate) != artifact_sha256
            ):
                continue
            candidate_payload = json.loads(
                candidate.read_text(encoding="utf-8")
            )
            if candidate_payload == expected_artifact:
                persisted_payload = candidate_payload
                verified_persisted_path = candidate
                break
        except Exception:
            continue
    persistence_valid = bool(
        endpoint_evidence.get(
            "completed_task_result_artifact_saved"
        ) is True
        and isinstance(persisted_artifact, Mapping)
        and dict(persisted_artifact) == expected_artifact
        and persisted_path_raw
        and persisted_path.is_absolute()
        and verified_persisted_path is not None
        and persisted_file_sha256 == artifact_sha256
        and persisted_payload == expected_artifact
        and _normalize_sha256(
            endpoint_evidence.get(
                "completed_task_result_artifact_sha256"
            )
        ) == artifact_sha256
    )
    if (
        artifact != expected_artifact
        or detections != canonical
        or sealed_result.get("coordinate_space")
        != "original_image_xyxy_pixels"
        or sealed_result.get("record_schema")
        != "xyxy_score_class_id_v1"
        or sealed_result.get("canonical_sort_policy")
        != "score_desc_class_id_asc_xyxy_lexicographic_v1"
        or sealed_result.get("detections") != canonical
        or sealed_result.get("detection_count") != len(canonical)
        or _normalize_sha256(sealed_result.get("detections_sha256"))
        != _canonical_detection_json_sha256(canonical)
        or _normalize_sha256(
            sealed_result.get("completed_result_artifact_sha256")
        ) != artifact_sha256
    ):
        raise _FrozenPostprocessError(
            "completed_v2_hotloop_result_artifact_invalid"
        )
    if not persistence_valid:
        raise _FrozenPostprocessError(
            "completed_v2_hotloop_result_artifact_persistence_invalid"
        )
    return canonical


def _verified_execution_completed_result_artifact(
    sealed_result: Mapping[str, Any],
    attestation: Mapping[str, Any],
    endpoint_evidence: Mapping[str, Any],
    *,
    native_report: Path | None = None,
    artifact_root: Path | None = None,
) -> list[dict[str, Any]]:
    """Verify one persisted completion-execution invocation artifact."""
    raw_artifact = attestation.get("artifact")
    if not isinstance(raw_artifact, Mapping):
        raise _FrozenPostprocessError(
            "completed_v2_execution_result_artifact_invalid"
        )
    artifact = dict(raw_artifact)
    artifact_sha256 = _normalize_sha256(
        attestation.get("artifact_sha256")
    )
    if (
        not artifact_sha256
        or _canonical_detection_json_sha256(artifact)
        != artifact_sha256
        or dict(sealed_result.get("artifact") or {}) != artifact
        or _normalize_sha256(sealed_result.get("artifact_sha256"))
        != artifact_sha256
    ):
        raise _FrozenPostprocessError(
            "completed_v2_execution_result_artifact_invalid"
        )

    persisted_artifact = endpoint_evidence.get(
        "completed_task_result_artifact"
    )
    persisted_path_raw = str(
        endpoint_evidence.get("completed_task_result_artifact_path") or ""
    ).strip()
    persisted_file_sha256 = _normalize_sha256(
        endpoint_evidence.get(
            "completed_task_result_artifact_file_sha256"
        )
    )
    candidates: list[Path] = []
    persisted_path = Path(persisted_path_raw).expanduser()
    if persisted_path_raw and persisted_path.is_absolute():
        candidates = _persisted_artifact_candidates(
            persisted_path,
            native_report=native_report,
            artifact_root=artifact_root,
        )
    verified_path: Path | None = None
    for candidate in candidates:
        try:
            if (
                candidate.is_file()
                and not candidate.is_symlink()
                and _sha256_file(candidate) == artifact_sha256
                and json.loads(candidate.read_text(encoding="utf-8"))
                == artifact
            ):
                verified_path = candidate
                break
        except Exception:
            continue
    if not (
        endpoint_evidence.get("completed_task_result_artifact_saved")
        is True
        and isinstance(persisted_artifact, Mapping)
        and dict(persisted_artifact) == artifact
        and _normalize_sha256(
            endpoint_evidence.get(
                "completed_task_result_artifact_sha256"
            )
        ) == artifact_sha256
        and persisted_path_raw
        and persisted_path.is_absolute()
        and verified_path is not None
        and persisted_file_sha256 == artifact_sha256
    ):
        raise _FrozenPostprocessError(
            "completed_v2_execution_result_artifact_persistence_invalid"
        )
    detections = sealed_result.get("detections")
    if (
        not isinstance(detections, list)
        or artifact.get("detections") != detections
        or any(not isinstance(value, Mapping) for value in detections)
    ):
        raise _FrozenPostprocessError(
            "completed_v2_execution_result_artifact_invalid"
        )
    return [dict(value) for value in detections]


def _completed_v2_self_reference_detection(
    full_tensors: Mapping[str, Any],
    native_tensors: Mapping[str, Any],
    endpoint_evidence: Mapping[str, Any],
    *,
    policy: AccuracyGatePolicy | None = None,
    native_report: Path | None = None,
    artifact_root: Path | None = None,
    dump_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Project different physical outputs onto one verified task endpoint.

    A verified same-hotloop completion artifact is the primary Native semantic
    evidence.  Physical raw tensors remain boundary evidence and are decoded
    only as a compatibility fallback for archived hash-only attestations.
    Full-ONNX remains the semantic oracle and is decoded with the same sealed
    thresholds and geometry.
    """
    try:
        mode, source_contract, comparison = _verified_completed_v2_contract(
            endpoint_evidence,
            native_tensors=native_tensors,
        )
        if mode == "native_three_stage_fast_oracle_outside_timing" and dump_metadata is not None:
            from onnx_splitpoint_tool.native_three_stage import verify_fast_completion_dump_binding
            verify_fast_completion_dump_binding(
                endpoint_evidence["completed_task_endpoint_attestation"],
                dump_metadata, endpoint_evidence,
            )
        if (
            _CanonicalYoloHarness is None
            or _canonical_postprocess_result_to_dict is None
            or _canonical_detection_json_sha256 is None
        ):
            raise _FrozenPostprocessError(
                "completed_v2_canonical_decoder_unavailable"
            )
        attestation = endpoint_evidence.get(
            "completed_task_endpoint_attestation"
        )
        attestation = (
            dict(attestation) if isinstance(attestation, Mapping) else {}
        )
        if mode == "frozen_host_tail":
            semantic_contract = source_contract
            sealed_result = attestation.get("frozen_postprocess_result")
            if (
                not isinstance(sealed_result, Mapping)
                or dict(endpoint_evidence.get(
                    "frozen_host_postprocess_result"
                ) or {}) != dict(sealed_result)
            ):
                raise _FrozenPostprocessError(
                    "completed_v2_native_result_attestation_mismatch"
                )
            exact_detections = (
                _verified_frozen_completed_result_artifact(
                    sealed_result,
                    endpoint_evidence,
                    native_report=native_report,
                    artifact_root=artifact_root,
                )
            )
            if exact_detections is not None:
                native_result = dict(sealed_result)
                native_detections = exact_detections
                result_binding = {
                    "semantic_result_binding_status": (
                        "exact_same_hotloop_completed_artifact"
                    ),
                    "exact_completed_result_identity_bound": True,
                    "portable_result_hash_mismatch": False,
                    "completed_v2_exact_result_claim_binding": True,
                    "completed_v2_semantic_evidence_tier": (
                        "exact_same_hotloop_completed_artifact"
                    ),
                    "performance_hotloop_result_sha256": str(
                        sealed_result.get("detections_sha256") or ""
                    ),
                }
                native_mode = (
                    "completed_v2:frozen_host_tail"
                )
            else:
                native_processor = _FrozenDetectionPostprocessor(
                    semantic_contract
                )
                replay_result = native_processor.process(
                    native_tensors,
                    original_wh=semantic_contract["original_wh"],
                )
                native_result = {
                    key: replay_result.get(key)
                    for key in sealed_result
                }
                native_detections = [
                    dict(value)
                    for value in native_processor.last_detections
                ]
                result_binding = _verified_completed_v2_result_binding(
                    native_result,
                    sealed_result,
                    native_detections,
                    policy=policy,
                )
                native_mode = (
                    "completed_v2:frozen_host_tail"
                )
        elif mode == (
            "integrated_accelerator_plus_frozen_normalization"
        ):
            semantic_contract = source_contract
            sealed_result = attestation.get(
                "frozen_decoded_nms_normalization_result"
            )
            if (
                not isinstance(sealed_result, Mapping)
                or dict(endpoint_evidence.get(
                    "frozen_decoded_nms_normalization_result"
                ) or {}) != dict(sealed_result)
            ):
                raise _FrozenPostprocessError(
                    "completed_v2_native_result_attestation_mismatch"
                )
            exact_detections = (
                _verified_frozen_completed_result_artifact(
                    sealed_result,
                    endpoint_evidence,
                    native_report=native_report,
                    artifact_root=artifact_root,
                )
            )
            if exact_detections is not None:
                native_result = dict(sealed_result)
                native_detections = exact_detections
                result_binding = {
                    "semantic_result_binding_status": (
                        "exact_same_hotloop_completed_artifact"
                    ),
                    "exact_completed_result_identity_bound": True,
                    "portable_result_hash_mismatch": False,
                    "completed_v2_exact_result_claim_binding": True,
                    "completed_v2_semantic_evidence_tier": (
                        "exact_same_hotloop_completed_artifact"
                    ),
                    "performance_hotloop_result_sha256": str(
                        sealed_result.get("detections_sha256") or ""
                    ),
                }
            else:
                # Archived Direct-BN6 V1 rows carried only a detections hash.
                # Replay remains screening-compatible, but every new embedded
                # artifact must pass the persisted-file binding above.
                native_processor = _FrozenDecodedNmsPostprocessor(
                    semantic_contract
                )
                replay_result = native_processor.process(
                    native_tensors,
                    original_wh=semantic_contract["original_wh"],
                )
                native_result = {
                    key: replay_result.get(key)
                    for key in sealed_result
                }
                native_detections = [
                    dict(value)
                    for value in native_processor.last_detections
                ]
                result_binding = _verified_completed_v2_result_binding(
                    native_result,
                    sealed_result,
                    native_detections,
                    policy=policy,
                )
            native_mode = (
                "completed_v2:integrated_accelerator_plus_"
                "frozen_normalization"
            )
        else:
            semantic_contract = dict(
                source_contract.get("processor_contract") or {}
            )
            sealed_result = attestation.get("last_result")
            if not isinstance(sealed_result, Mapping):
                raise _FrozenPostprocessError(
                    "completed_v2_same_hotloop_result_missing"
                )
            native_result = dict(sealed_result)
            native_detections = (
                _verified_execution_completed_result_artifact(
                    sealed_result,
                    attestation,
                    endpoint_evidence,
                    native_report=native_report,
                    artifact_root=artifact_root,
                )
            )
            result_binding = {
                "semantic_result_binding_status": (
                    "exact_same_hotloop_completed_artifact"
                ),
                "exact_completed_result_identity_bound": True,
                "portable_result_hash_mismatch": False,
                "completed_v2_exact_result_claim_binding": True,
                "completed_v2_semantic_evidence_tier": (
                    "exact_same_hotloop_completed_artifact"
                ),
                "performance_hotloop_result_sha256": str(
                    attestation.get("content_sha256") or ""
                ),
            }
            native_mode = (
                "completed_v2:decoded_nms:detection_completion_execution_"
                "same_hotloop_completed_artifact"
            )
            if mode == "native_three_stage_fast_oracle_outside_timing":
                # Exact fast/oracle equality is verified from the dumped final
                # output. Preserve that the sealed artifact was made postflight.
                result_binding["semantic_result_binding_status"] = "verified_postflight_oracle_matches_fast_sentinel"
                result_binding["completed_v2_semantic_evidence_tier"] = "verified_postflight_oracle_matches_fast_sentinel"
                native_mode = "completed_v2:decoded_nms:verified_postflight_oracle"


        native_detections = sorted(
            [
                {
                    "class_id": int(value["class_id"]),
                    "score": float(value["score"]),
                    "x1": float(value["x1"]),
                    "y1": float(value["y1"]),
                    "x2": float(value["x2"]),
                    "y2": float(value["y2"]),
                }
                for value in native_detections
            ],
            key=lambda value: (
                -float(value["score"]),
                int(value["class_id"]),
                float(value["x1"]),
                float(value["y1"]),
                float(value["x2"]),
                float(value["y2"]),
            ),
        )
        if mode in {"detection_completion_execution_v1", "native_three_stage_fast_oracle_outside_timing"}:
            if _DetectionCompletionRuntime is None:
                raise _FrozenPostprocessError(
                    "completed_v2_execution_runtime_unavailable"
                )
            reference_runtime = _DetectionCompletionRuntime(
                source_contract,
                observation_relation="independent_replay",
            )
            reference_runtime.process(dict(full_tensors))
            full_payload = {
                "format": (
                    "detection_completion_execution:"
                    f"{source_contract.get('completion_mode') or 'unknown'}"
                ),
                "detections": [
                    dict(value)
                    for value in reference_runtime.last_detections
                ],
            }
        elif (
            not isinstance(
                semantic_contract.get("model_bound_decoder_contract"),
                Mapping,
            )
            and
            len(full_tensors) == 3
            and all(
                np.asarray(value).ndim == 5
                for value in full_tensors.values()
            )
        ):
            # A direct decoded-NMS contract can reject an obvious three-head
            # raw export structurally.  Do not instantiate any unbound legacy
            # decoder merely to discover that the endpoint formats differ.
            full_payload = {
                "format": "multiscale_head",
                "detections": [],
            }
        else:
            harness = _CanonicalYoloHarness(
                conf_thresh=float(
                    semantic_contract.get(
                        "confidence_threshold",
                        semantic_contract.get("score_threshold"),
                    )
                ),
                iou_thresh=float(semantic_contract["iou_threshold"]),
                max_det=int(semantic_contract["max_detections"]),
                model_id=str(semantic_contract.get("model_id") or ""),
                multiscale_decoder_contract=(
                    semantic_contract.get("model_bound_decoder_contract")
                    if isinstance(
                        semantic_contract.get(
                            "model_bound_decoder_contract"
                        ), Mapping,
                    )
                    and len(full_tensors) == 3
                    and all(
                        np.asarray(value).ndim == 5
                        for value in full_tensors.values()
                    )
                    else None
                ),
            )
            full_payload = _canonical_postprocess_result_to_dict(
                harness.postprocess(
                    dict(full_tensors),
                    {
                        "input_hw": list(semantic_contract["input_hw"]),
                        "original_wh": list(
                            semantic_contract["original_wh"]
                        ),
                        "variant": "full_onnx_completed_v2_reference",
                    },
                )
            )
            if isinstance(full_payload.get("json"), Mapping):
                full_payload = dict(full_payload["json"])
        full_reference_format = str(
            full_payload.get("format") or ""
        ).strip()
        allowed_full_reference_formats = {"bn6_detections"}
        if mode in {"detection_completion_execution_v1", "native_three_stage_fast_oracle_outside_timing"}:
            allowed_full_reference_formats = {
                "detection_completion_execution:"
                f"{source_contract.get('completion_mode') or 'unknown'}"
            }
        elif mode == "frozen_host_tail":
            sealed_decoder_format = str(
                semantic_contract.get("decoder_format") or ""
            ).strip()
            if sealed_decoder_format:
                allowed_full_reference_formats.add(
                    sealed_decoder_format
                )
            # YOLO11's source Full ONNX exposes the canonical decoded
            # pre-NMS tensor [1, 84, 8400], while the Hailo compiler endpoint
            # is intentionally cut back to six DFL16/C80 raw reg/cls heads.
            # These are different physical formats, but both are completed by
            # the canonical harness under the same sealed score/NMS/geometry
            # contract before they reach the verified decoded-NMS comparison
            # endpoint.  Requiring the Full reference to retain the Hailo raw
            # format rejects this valid producer pairing before any numerical
            # comparison can occur.
            if (
                str(
                    semantic_contract.get("model_family") or ""
                ).strip().lower()
                == "yolo11"
                and sealed_decoder_format == "ultralytics_regcls"
            ):
                allowed_full_reference_formats.add(
                    "ultralytics_decoded"
                )
        if full_reference_format not in allowed_full_reference_formats:
            raise _FrozenPostprocessError(
                "completed_v2_full_reference_format_not_contract_"
                f"compatible:{full_reference_format or 'missing'}"
            )
        full_reference_detections = full_payload.get("detections")
        if (
            not isinstance(full_reference_detections, list)
            or any(
                not isinstance(value, Mapping)
                for value in full_reference_detections
            )
        ):
            raise _FrozenPostprocessError(
                "completed_v2_full_reference_detections_invalid"
            )
        reference_detections = sorted(
            [
                {
                    "class_id": int(value["class_id"]),
                    "score": float(value["score"]),
                    "x1": float(value["x1"]),
                    "y1": float(value["y1"]),
                    "x2": float(value["x2"]),
                    "y2": float(value["y2"]),
                }
                for value in full_reference_detections
            ],
            key=lambda value: (
                -float(value["score"]),
                int(value["class_id"]),
                float(value["x1"]),
                float(value["y1"]),
                float(value["x2"]),
                float(value["y2"]),
            ),
        )
        full_mode = (
            "completed_v2:full_bn6_normalized"
            if full_reference_format == "bn6_detections"
            else (
                "completed_v2:full_"
                f"{full_reference_format}_decoded_nms"
            )
        )
        return {
            "available": True,
            "completed_v2_verified": True,
            "completed_task_comparison_endpoint_contract": comparison,
            "completed_task_comparison_endpoint_contract_hash": (
                comparison["endpoint_contract_hash"]
            ),
            "completed_task_comparison_output_endpoint_id": (
                comparison["output_endpoint_id"]
            ),
            "expected_contract_family": "decoded_nms",
            "expected_contract_source": (
                "verified_completed_task_comparison_endpoint_v2"
            ),
            "full_mode": full_mode,
            "full_reference_format": full_reference_format,
            "native_mode": native_mode,
            "reference_detections": reference_detections,
            "native_detections": native_detections,
            "full_reference_result_sha256": (
                _canonical_detection_json_sha256(reference_detections)
            ),
            "native_completed_result_sha256": str(
                native_result.get("detections_sha256")
                or native_result.get("content_sha256")
                or ""
            ),
            **result_binding,
        }
    except Exception as exc:
        return {
            "available": False,
            "completed_v2_verified": False,
            "reason": f"{type(exc).__name__}: {exc}",
        }


def _expected_detection_contract(
    native_output_manifest: Path,
    native_report: Path | None = None,
    boundary_manifest: Path | None = None,
    endpoint_evidence: Mapping[str, Any] | None = None,
) -> tuple[str, str]:
    if _completed_v2_declared(endpoint_evidence):
        try:
            _verified_completed_v2_contract(
                endpoint_evidence or {},
            )
            return (
                "decoded_nms",
                "verified_completed_task_comparison_endpoint_v2",
            )
        except Exception:
            return "unknown", "completed_task_comparison_endpoint_v2_invalid"
    payloads = [_load_json(native_output_manifest) or {}]
    if native_report:
        payloads.append(_load_json(native_report) or {})
    if boundary_manifest:
        payloads.append(_load_json(boundary_manifest) or {})
    manifest_payload = payloads[0] if payloads else {}
    if _decoded_nms_attestation_passed(manifest_payload):
        return "decoded_nms", "explicit_endpoint_declaration_and_runtime_value_attestation"
    joined = " ".join(value.lower() for payload in payloads for value in _contract_text_values(payload))
    if any(token in joined for token in ("raw_head", "raw_yolo", "external_decoder", "requires_external_postprocess")):
        return "raw_head", "explicit_metadata"

    # A shape-only [B,N,6] legacy manifest is deliberately not accepted: a
    # one-class raw head may have the same shape.  Cross-family comparisons
    # remain forbidden and a raw head must be declared by archived metadata.
    return "unknown", "metadata_unavailable"


def _enforce_detection_contract(
    result: dict[str, Any],
    native_output_manifest: Path,
    native_report: Path | None = None,
    boundary_manifest: Path | None = None,
    endpoint_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(result, dict):
        return result
    expected, expected_source = _expected_detection_contract(
        native_output_manifest,
        native_report,
        boundary_manifest,
        endpoint_evidence,
    )
    best = result.get("best") if isinstance(result.get("best"), dict) else {}
    native_mode = str(result.get("decode_mode") or best.get("native_mode") or "")
    full_mode = str(best.get("full_mode") or "")
    native_family = _detection_contract_family(native_mode)
    full_family = _detection_contract_family(full_mode)
    family_match = native_family == full_family and native_family != "unknown"
    metadata_available = expected != "unknown"
    expected_match = metadata_available and native_family == expected
    result["expected_contract_family"] = expected
    result["expected_contract_source"] = expected_source
    result["native_contract_family"] = native_family
    result["full_contract_family"] = full_family
    result["contract_metadata_available"] = metadata_available
    result["contract_family_match"] = bool(family_match and expected_match)
    if not metadata_available:
        result["ok"] = False
        result["semantic_ok"] = False
        result["semantic_available"] = False
        result["diagnosis"] = "detection_contract_metadata_unavailable"
        result["failure_reason"] = (
            f"native={native_family};full={full_family};native_mode={native_mode};"
            f"full_mode={full_mode}; archived output contract does not declare raw_head or decoded_nms"
        )
    elif not family_match or not expected_match:
        result["ok"] = False
        result["semantic_ok"] = False
        result["semantic_available"] = False
        result["diagnosis"] = "detection_contract_family_mismatch"
        result["failure_reason"] = (
            f"expected={expected};native={native_family};full={full_family};"
            f"native_mode={native_mode};full_mode={full_mode}"
        )
    return result


def _self_ref_mode_sanity(mode: str, sat: dict[str, Any], count: int) -> tuple[int, int, int]:
    """Heuristic sanity for choosing a same-decoder Full-ONNX self-reference mode.

    Self-reference can be fooled if both Full and Native are decoded with the same
    wrong [class,score] column order.  Prefer modes where the score column is
    explicitly before class and score saturation is low.
    """
    m = str(mode or '')
    satn = int((sat or {}).get('score_ge_0999') or 0)
    low_sat = int(satn <= max(1, int(0.10 * max(1, count))))
    score_class = int('_score_class' in m or m.endswith(':raw'))
    xyxy = int(':nms:xyxy_' in m)
    raw = int(m.endswith(':raw'))
    class_score_penalty = int('_class_score' in m)
    return (score_class, low_sat, xyxy - raw - class_score_penalty)




def _probe_payload_to_self_reference(
    j: dict[str, Any],
    probe_path: Path | None = None,
    policy: AccuracyGatePolicy | None = None,
) -> dict[str, Any] | None:
    """Convert native_yolo_full_self_reference_probe.json payload to validator self_ref dict."""
    try:
        if not isinstance(j, dict):
            return None
        schema = str(j.get('schema') or '')
        if schema and schema != 'onnx-splitpoint/native-yolo-full-self-reference-probe':
            return None
        if not schema and not (j.get('diagnosis') or j.get('best') or j.get('best_match_ratio') is not None):
            return None
        best = dict(j.get('best') or {})
        if not best and j.get('best_match_ratio') is not None:
            ratio = float(j.get('best_match_ratio') or 0.0)
            best = {
                'match': {'match_ratio': ratio, 'matched': int(round(ratio * 1000)), 'ref_count': 1000},
                'class_agnostic_match': {'match_ratio': float(j.get('best_class_agnostic_ratio') or ratio)},
                'native_mode': j.get('best_native_mode') or '',
                'full_mode': j.get('best_full_mode') or '',
            }
        match = dict(best.get('match') or j.get('match') or {})
        ca = dict(best.get('class_agnostic_match') or j.get('class_agnostic_match') or {})
        diag = str(j.get('diagnosis') or '')
        contract_match = j.get('contract_family_match')
        contract_ok = contract_match is not False and diag != 'detection_contract_family_mismatch'
        similarity = evaluate_detection_similarity(
            match,
            ca,
            policy or AccuracyGatePolicy(),
        )
        sem_ok = bool(
            contract_ok
            and similarity.get('numerical_similarity_pass') is True
        )
        if not contract_ok:
            similarity.update({
                'numerical_similarity_pass': False,
                'numerical_similarity_status': 'failed',
                'numerical_similarity_reason': (
                    'detection_contract_family_mismatch'
                ),
            })
        return {
            'available': True,
            'ok': sem_ok,
            'diagnosis': diag or ('native_semantic_matches_full_self_reference' if sem_ok else 'native_semantic_differs_from_full_self_reference'),
            'reason': str(j.get('evidence_error') or j.get('reason') or ''),
            'source': 'precomputed_native_yolo_full_self_reference_probe' if probe_path else 'native_yolo_full_self_reference_probe',
            'precomputed_probe': str(probe_path) if probe_path else str(j.get('out') or ''),
            'boundary_manifest': str(j.get('boundary_manifest') or ''),
            'native_output_manifest': str(j.get('native_output_manifest') or ''),
            'full_onnx': str(j.get('full_onnx') or ''),
            'best': best,
            'rows': list(j.get('rows') or [])[:30],
            'tensor_compares': list(j.get('tensor_compares') or []),
            'semantic_ok': sem_ok,
            'semantic_available': bool(j.get('semantic_available', True)) and contract_ok,
            'expected_contract_family': str(j.get('expected_contract_family') or ''),
            'expected_contract_source': str(j.get('expected_contract_source') or ''),
            'contract_family_match': bool(contract_ok),
            'match': match,
            'class_agnostic_match': ca,
            'decode_mode': best.get('native_mode') or j.get('best_native_mode') or '',
            'reference_detections': list(j.get('reference_detections') or []),
            'native_detections': list(j.get('native_detections') or []),
            **similarity,
        }
    except Exception:
        return None


def _case_backend_precision_model_hints(path_text: str) -> tuple[str, str, str, str]:
    path_text = str(path_text or '').replace('\\\\', '/')
    m_case = re.search(r'/native_pipeline/([^/]+)/', path_text)
    if not m_case:
        m_case = re.search(r'/(?:case|case_id)=([^/]+)/', path_text)
    case_hint = m_case.group(1) if m_case else ''
    precision_hint = ''
    backend_hint = ''
    m_prec = re.search(r'/(hailo_to_trt|hailo10h_to_trt|deepx_to_trt)/([^/]+)/', path_text)
    if m_prec:
        backend_hint = m_prec.group(1)
        precision_hint = m_prec.group(2)
    else:
        m_backend = re.search(r'/backend=([^/]+)/', path_text)
        backend_hint = m_backend.group(1) if m_backend else ''
        m_precision = re.search(r'/precision=([^/]+)/', path_text)
        precision_hint = m_precision.group(1) if m_precision else ''
    model_hint = ''
    m_model = re.search(r'/model=([^/]+)/', path_text)
    if m_model:
        model_hint = m_model.group(1)
    if not model_hint:
        m_model = re.search(r'/([^/]+)/benchmark_set/', path_text)
        model_hint = m_model.group(1) if m_model else ''
    return case_hint, backend_hint, precision_hint, model_hint


def _native_output_manifest_logical_key(path_value: Any) -> tuple[str, str, str]:
    """Return a rebase-stable producer/model/BenchmarkSet identity."""
    text = str(path_value or '').strip().replace('\\', '/')
    if not text:
        return '', '', ''
    marker = '/benchmark_set/'
    if marker not in text:
        return '', '', ''
    prefix, suffix = text.rsplit(marker, 1)
    model_match = re.search(r'/model=([^/]+)/', '/' + suffix)
    model = (
        model_match.group(1)
        if model_match else prefix.rstrip('/').split('/')[-1]
    )
    producer_match = re.search(r'/native_producers/([^/]+)/', prefix + '/')
    producer = producer_match.group(1) if producer_match else ''
    return producer, model, suffix.strip('/')


def _is_unbound_yolov7_raw_self_reference_target(
    native_output_manifest: Path,
    target_contract_family: str,
) -> bool:
    """Identify the legacy self-reference path forbidden for yolov7_paper.

    Cached and auto-generated legacy self-reference probes predate the v2.75.47
    model-bound decoder contract.  Even an artifact-hash-bound cache can only
    attest that both sides used the same decoder; it cannot prove that the
    decoder used the registered YOLOv7 anchors.  Current ``yolov7_paper`` raw
    heads therefore become claim-capable only through Completed-v2 evidence.
    Other models keep their historical cache/read compatibility.
    """
    if str(target_contract_family or '').strip().lower() != 'raw_head':
        return False
    manifest_path = Path(native_output_manifest).expanduser()
    candidates: list[Any] = [
        _native_output_manifest_logical_key(manifest_path)[1],
        _case_backend_precision_model_hints(str(manifest_path))[3],
    ]
    try:
        payload = _load_json(manifest_path) or {}
        provenance = (
            payload.get('provenance')
            if isinstance(payload.get('provenance'), Mapping) else {}
        )
        candidates.extend((
            payload.get('model'),
            payload.get('model_id'),
            provenance.get('model'),
            provenance.get('model_id'),
        ))
    except Exception:
        pass
    normalized = {
        re.sub(r'[^a-z0-9]+', '_', str(value or '').strip().lower()).strip('_')
        for value in candidates
        if str(value or '').strip()
    }
    return 'yolov7_paper' in normalized


def _unbound_yolov7_self_reference_rejection() -> dict[str, Any]:
    reason = 'yolov7_paper_raw_head_requires_model_bound_decoder_contract'
    return {
        'available': False,
        'ok': False,
        'semantic_ok': False,
        'semantic_available': False,
        'diagnosis': reason,
        'reason': reason,
        'failure_reason': reason,
        'decode_mode': 'rejected:unbound_yolov7_paper_raw_head',
        'expected_contract_family': 'raw_head',
        'contract_family_match': False,
        'numerical_similarity_pass': False,
        'numerical_similarity_status': 'unavailable',
        'numerical_similarity_reason': reason,
    }


def _cached_probe_matches_native_output(
    payload: Mapping[str, Any],
    *,
    probe_path: Path,
    target_manifest: Path,
    target_workdir: Path,
    target_contract_family: str,
) -> bool:
    """Admit cached evidence only for the exact target output artifact."""
    recorded_manifest = str(payload.get('native_output_manifest') or '').strip()
    target_key = _native_output_manifest_logical_key(target_manifest)
    recorded_key = _native_output_manifest_logical_key(recorded_manifest)
    if not all(target_key[1:]) or target_key[1:] != recorded_key[1:]:
        return False
    if target_key[0] and recorded_key[0] and target_key[0] != recorded_key[0]:
        return False
    if _is_unbound_yolov7_raw_self_reference_target(
        target_manifest, target_contract_family,
    ):
        # A cache digest binds bytes, not decoder semantics.  Legacy probes can
        # report a perfect match while decoding both arms with the same Tiny/
        # YOLOv5 anchors, so they are never admissible for current YOLOv7 claims.
        return False
    recorded_family = str(
        payload.get('expected_contract_family') or ''
    ).strip().lower()
    if (
        target_contract_family in {'raw_head', 'decoded_nms'}
        and recorded_family != target_contract_family
    ):
        return False
    try:
        schema_version = int(payload.get('schema_version') or 0)
    except (TypeError, ValueError, OverflowError):
        return False
    recorded_sha = _normalize_sha256(
        payload.get('native_output_manifest_sha256')
    )
    if schema_version >= 4:
        return bool(
            recorded_sha
            and target_manifest.is_file()
            and _sha256_file(target_manifest) == recorded_sha
        )
    # Legacy v3 probes have no artifact digest.  They remain usable only in
    # their exact original workdir; global/root searches may never promote one.
    try:
        return probe_path.resolve().parent == target_workdir.resolve()
    except Exception:
        return False


def _precomputed_full_self_reference_detection(
    native_output_manifest: Path,
    roots: list[Path] | None = None,
    native_report: Path | None = None,
    policy: AccuracyGatePolicy | None = None,
) -> dict[str, Any] | None:
    """Load a precomputed native_yolo_full_self_reference_probe.json near a native output manifest.

    This lets local validation consume a self-reference report that was generated on
    the accelerator host (where onnxruntime and the copied benchmark_set are known
    to exist).  It is intentionally read-only: if no cached probe is present we
    return None and the live ORT fallback can still try to run.
    """
    try:
        p = Path(native_output_manifest).expanduser().resolve()
        work = p.parent.parent
        candidates = [
            work / 'native_yolo_full_self_reference_probe.json',
            work / 'native_fifo_boundary' / 'native_yolo_full_self_reference_probe.json',
            work / 'native_fifo_outputs' / 'native_yolo_full_self_reference_probe.json',
            work / 'native_outputs' / 'native_yolo_full_self_reference_probe.json',
        ]
        try:
            candidates.extend(sorted(work.rglob('native_yolo_full_self_reference_probe.json'))[:32])
        except Exception:
            pass
        # v59eg: copied/remote self-reference probes are sometimes present at the
        # precision workdir level while the validator starts from a rebased
        # native_outputs manifest.  Search a few parents, but keep the search
        # scoped to the same model/case/backend/precision path so we do not pick
        # a stale probe from another row.
        path_text = str(p).replace('\\', '/')
        # v59eh: derive row hints from the manifest path and, if available,
        # from the native report path as well.  Some combined summaries pass a
        # rebased manifest path, so the native report often carries the stable
        # backend/model/case/precision location.
        case_hint, backend_hint, precision_hint, model_hint = _case_backend_precision_model_hints(path_text)
        if native_report:
            nr_case, nr_backend, nr_prec, nr_model = _case_backend_precision_model_hints(str(native_report))
            case_hint = case_hint or nr_case
            backend_hint = backend_hint or nr_backend
            precision_hint = precision_hint or nr_prec
            model_hint = model_hint or nr_model
        try:
            for parent in list(work.parents)[:6]:
                for cand in sorted(parent.rglob('native_yolo_full_self_reference_probe.json'))[:64]:
                    ct = str(cand).replace('\\', '/')
                    if case_hint and f'/native_pipeline/{case_hint}/' not in ct:
                        continue
                    if precision_hint and f'/{precision_hint}/' not in ct:
                        continue
                    if backend_hint and f'/{backend_hint}/' not in ct:
                        continue
                    if model_hint and f'/{model_hint}/' not in ct:
                        continue
                    candidates.append(cand)
        except Exception:
            pass
        # v59eh: also search explicitly supplied roots.  This covers local
        # report generation where native_outputs manifests are copied/rebased
        # but the remote-generated probe lives under the root tree.
        try:
            for root in (roots or []):
                rp = Path(root).expanduser()
                if not rp.exists():
                    continue
                for cand in sorted(rp.rglob('native_yolo_full_self_reference_probe.json'))[:512]:
                    ct = str(cand).replace('\\', '/')
                    if case_hint and f'/native_pipeline/{case_hint}/' not in ct:
                        continue
                    if precision_hint and f'/{precision_hint}/' not in ct:
                        continue
                    if backend_hint and f'/{backend_hint}/' not in ct:
                        continue
                    if model_hint and f'/{model_hint}/' not in ct:
                        continue
                    candidates.append(cand)
        except Exception:
            pass
        # v59ei: do not return the first cached probe blindly.  During
        # bring-up, failed inline/autoprobe JSONs can coexist with a later
        # successful remote probe in the same copied root.  Evaluate all
        # candidates and prefer semantic-ok / highest-ratio probes.
        target_contract_family, _target_contract_source = (
            _expected_detection_contract(
                p, native_report=native_report,
            )
        )
        seen=set()
        reps=[]
        for cand in candidates:
            try:
                cp = Path(cand).expanduser().resolve()
                if str(cp) in seen:
                    continue
                seen.add(str(cp))
                if not cp.is_file():
                    continue
                j = _load_json(cp) or {}
                if not _cached_probe_matches_native_output(
                    j,
                    probe_path=cp,
                    target_manifest=p,
                    target_workdir=work,
                    target_contract_family=target_contract_family,
                ):
                    continue
                rep = _probe_payload_to_self_reference(j, cp, policy)
                if rep is None:
                    continue
                match = rep.get('match') or {}
                best = rep.get('best') or {}
                ratio = float(match.get('match_ratio') or best.get('best_match_ratio') or 0.0)
                matched = int(match.get('matched') or 0)
                ref_count = int(match.get('ref_count') or best.get('full_count') or 0)
                diag_ok = int(str(rep.get('diagnosis') or '') == 'native_semantic_matches_full_self_reference')
                sem_ok = int(bool(rep.get('semantic_ok') or rep.get('ok')) or diag_ok)
                # Prefer exact workdir-local probes, but never over a successful
                # probe with better semantic evidence.
                local_bonus = int(str(cp).startswith(str(work.resolve())))
                reps.append((sem_ok, diag_ok, ratio, matched, -abs(ref_count - max(ref_count, matched)), local_bonus, str(cp), rep))
            except Exception:
                continue
        if reps:
            reps.sort(reverse=True)
            return reps[0][-1]
    except Exception:
        return None
    return None


def _auto_run_yolo_full_self_reference_probe(
    native_output_manifest: Path,
    roots: list[Path] | None = None,
    native_report: Path | None = None,
    policy: AccuracyGatePolicy | None = None,
) -> dict[str, Any] | None:
    """Run the canonical standalone YOLO self-reference probe if no cache exists."""
    try:
        target_contract_family, _target_contract_source = (
            _expected_detection_contract(
                Path(native_output_manifest), native_report=native_report,
            )
        )
        if _is_unbound_yolov7_raw_self_reference_target(
            Path(native_output_manifest), target_contract_family,
        ):
            return None
        boundary_manifest, input_manifest_kind, _reason = (
            _find_self_reference_input_manifest(
                Path(native_output_manifest),
                roots=roots,
                native_report=native_report,
            )
        )
        if not boundary_manifest:
            return None
        bs = None
        for par in Path(boundary_manifest).parents:
            if par.name == 'benchmark_set':
                bs = par
                break
        if bs is None:
            return None
        # Determine case from native_pipeline/<case>/... path.
        case = ''
        m = re.search(r'/native_pipeline/([^/]+)/', str(boundary_manifest).replace('\\\\', '/'))
        if m:
            case = m.group(1)
        if not case and input_manifest_kind == 'native_full_input_manifest':
            case = 'full'
        if not case:
            return None
        script = Path(__file__).with_name('native_yolo_full_self_reference_probe.py')
        if not script.is_file():
            return None
        work = Path(native_output_manifest).expanduser().resolve().parent.parent
        out = work / 'native_yolo_full_self_reference_probe.json'
        effective_policy = policy or AccuracyGatePolicy()
        cmd = [
            sys.executable, str(script),
            '--benchmark-set', str(bs),
            '--case', case,
            '--boundary-manifest', str(boundary_manifest),
            '--native-output-manifest', str(native_output_manifest),
            '--quality-gate-json', json.dumps(
                effective_policy.as_dict(),
                sort_keys=True,
                separators=(',', ':'),
            ),
            '--out', str(out),
        ]
        if native_report is not None:
            cmd.extend(['--native-report', str(native_report)])
        import subprocess
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=180)
        if proc.returncode != 0 or not out.is_file():
            return None
        j = _load_json(out) or {}
        rep = _probe_payload_to_self_reference(j, out, effective_policy)
        if rep is not None:
            rep['source'] = 'auto_native_yolo_full_self_reference_probe'
            rep['autoprobe_cmd'] = cmd
        return rep
    except Exception:
        return None

def _full_onnx_self_reference_detection(
    native_output_manifest: Path,
    eval_root: Path,
    conf: float = 0.25,
    native_report: Path | None = None,
    roots: list[Path] | None = None,
    policy: AccuracyGatePolicy | None = None,
    endpoint_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Decode native output against Full-ONNX run on the same native input dump.

    This is a fallback/override for YOLO split-native validation when the external
    validation_report.json is in a different decoder/coordinate contract.  It is
    intentionally conservative: it only claims semantic OK when a sane same-mode
    decoder matches Full-ONNX well.
    """
    effective_policy = policy or AccuracyGatePolicy()
    input_manifest, input_manifest_kind, input_reason = (
        _find_self_reference_input_manifest(
            native_output_manifest,
            roots=roots,
            native_report=native_report,
            endpoint_evidence=endpoint_evidence,
        )
    )
    is_native_full = input_manifest_kind == 'native_full_input_manifest'
    completed_v2_declared = _completed_v2_declared(endpoint_evidence)
    target_contract_family, _target_contract_source = (
        _expected_detection_contract(
            native_output_manifest,
            native_report=native_report,
            endpoint_evidence=endpoint_evidence,
        )
    )
    if (
        not completed_v2_declared
        and _is_unbound_yolov7_raw_self_reference_target(
            native_output_manifest, target_contract_family,
        )
    ):
        # Do not let cached, auto-probed, or live same-wrong-decoder agreement
        # become semantic evidence.  The current YOLOv7 runtime must present the
        # model-bound Completed-v2 contract instead.
        return _unbound_yolov7_self_reference_rejection()
    cached = (
        None
        if is_native_full or completed_v2_declared
        else _precomputed_full_self_reference_detection(
            native_output_manifest,
            roots=roots,
            native_report=native_report,
            policy=effective_policy,
        )
    )
    # v59ei: a cached failed probe can be stale.  Return successful cached
    # evidence immediately; otherwise run/refresh the canonical standalone probe
    # and keep whichever result has stronger semantic evidence.
    if cached is not None:
        cached = _enforce_detection_contract(
            cached,
            native_output_manifest,
            native_report=native_report,
            endpoint_evidence=endpoint_evidence,
        )
    if cached is not None and bool(cached.get('semantic_ok') or cached.get('ok')):
        return cached
    # v59eh/v59ei: if no cached positive probe exists, run the canonical
    # standalone self-reference probe and consume its JSON. This keeps the
    # summary validator aligned with scripts/native_yolo_full_self_reference_probe.py,
    # which is the canonical YOLO self-reference oracle used during bring-up.
    autoprobe = (
        None
        if is_native_full or completed_v2_declared
        else _auto_run_yolo_full_self_reference_probe(
            native_output_manifest,
            roots=roots,
            native_report=native_report,
            policy=effective_policy,
        )
    )
    if autoprobe is not None:
        autoprobe = _enforce_detection_contract(
            autoprobe,
            native_output_manifest,
            native_report=native_report,
            endpoint_evidence=endpoint_evidence,
        )
        if bool(autoprobe.get('semantic_ok') or autoprobe.get('ok')):
            return autoprobe
        if cached is None:
            return autoprobe
        def _r(x):
            try:
                return float(((x.get('match') or {}).get('match_ratio')) or ((x.get('best') or {}).get('match') or {}).get('match_ratio') or 0.0)
            except Exception:
                return 0.0
        return autoprobe if _r(autoprobe) >= _r(cached) else cached
    if cached is not None:
        return cached
    try:
        import onnxruntime as ort  # type: ignore
    except Exception as exc:
        return {'available': False, 'reason': f'onnxruntime_unavailable:{type(exc).__name__}: {exc}'}
    try:
        boundary_manifest = input_manifest
        if not boundary_manifest:
            return {'available': False, 'reason': input_reason}
        full = _find_full_onnx_for_native_manifest(native_output_manifest, eval_root, roots=roots)
        if not full:
            return {'available': False, 'reason': 'full_onnx_missing'}
        sess = ort.InferenceSession(str(full), providers=['CPUExecutionProvider'])
        inp = sess.get_inputs()[0]
        input_shape = _ort_input_shape(inp.shape)
        dtype = _ort_input_dtype(inp)
        if dtype is None:
            return {'available': False, 'reason': 'onnx_input_dtype_unsupported'}
        reference_input_evidence: dict[str, Any] = {}
        feed = _native_input_dump_feed(
            boundary_manifest,
            input_shape,
            image_scale='native',
            target_dtype=dtype,
            evidence=reference_input_evidence,
        )
        if feed is None:
            return {
                'available': False,
                'reason': 'native_input_dump_unavailable',
                'reference_input_evidence': reference_input_evidence,
            }
        full_outs = [np.asarray(x) for x in sess.run(None, {inp.name: feed})]
        full_names = [o.name for o in sess.get_outputs()]
        full_tensors = {str(full_names[i] if i < len(full_names) else f'output{i}'): np.asarray(o) for i,o in enumerate(full_outs)}
        native_tensors, _meta = load_dump(str(native_output_manifest))
        if _completed_v2_declared(endpoint_evidence):
            completed = _completed_v2_self_reference_detection(
                full_tensors,
                native_tensors,
                endpoint_evidence or {},
                dump_metadata=_load_json(native_output_manifest),
                policy=effective_policy,
                native_report=native_report,
                artifact_root=(
                    _benchmark_set_for_report(native_report)
                    if native_report is not None else None
                ),
            )
            if completed.get("available") is not True:
                return {
                    **completed,
                    "self_reference_input_manifest_kind": (
                        input_manifest_kind
                    ),
                    "reference_input_evidence": reference_input_evidence,
                    "boundary_manifest": str(boundary_manifest),
                    "full_onnx": str(full),
                    "expected_contract_family": "decoded_nms",
                    "expected_contract_source": (
                        "completed_task_comparison_endpoint_v2_invalid"
                    ),
                    "semantic_available": False,
                    "semantic_ok": False,
                }
            reference_detections = list(
                completed.get("reference_detections") or []
            )
            native_detections = list(
                completed.get("native_detections") or []
            )
            match = _match_detections(
                reference_detections,
                native_detections,
                iou_thr=(
                    effective_policy
                    .native_self_reference_iou_threshold
                ),
            )
            class_agnostic_match = (
                _match_detections_class_agnostic(
                    reference_detections,
                    native_detections,
                    iou_thr=(
                        effective_policy
                        .native_self_reference_iou_threshold
                    ),
                )
            )
            similarity = evaluate_detection_similarity(
                match,
                class_agnostic_match,
                effective_policy,
            )
            semantic_available = bool(reference_detections)
            semantic_ok = bool(
                semantic_available
                and similarity.get(
                    "numerical_similarity_pass"
                ) is True
            )
            return {
                **completed,
                "ok": semantic_ok,
                "semantic_ok": semantic_ok,
                "semantic_available": semantic_available,
                "diagnosis": (
                    "native_semantic_matches_full_self_reference"
                    if semantic_ok
                    else "native_semantic_differs_from_full_self_reference"
                ),
                "boundary_manifest": str(boundary_manifest),
                "full_onnx": str(full),
                "self_reference_input_manifest_kind": (
                    input_manifest_kind
                ),
                "reference_input_evidence": reference_input_evidence,
                "contract_metadata_available": True,
                "contract_family_match": True,
                "best": {
                    "full_mode": completed["full_mode"],
                    "native_mode": completed["native_mode"],
                    "full_contract_family": "decoded_nms",
                    "native_contract_family": "decoded_nms",
                    "expected_contract_family": "decoded_nms",
                    "contract_family_match": True,
                    "full_count": len(reference_detections),
                    "native_count": len(native_detections),
                    "match": match,
                    "class_agnostic_match": (
                        class_agnostic_match
                    ),
                },
                "rows": [],
                "rejected_contract_pairs": [],
                "match": match,
                "class_agnostic_match": class_agnostic_match,
                "decode_mode": completed["native_mode"],
                **similarity,
            }
        # Use the model/input dump size for NMS-coordinate decoding.
        man = _load_json(boundary_manifest) or {}
        hwc = man.get('input_shape_hwc') or []
        img_w = int(hwc[1]) if isinstance(hwc, list) and len(hwc) >= 2 else 640
        img_h = int(hwc[0]) if isinstance(hwc, list) and len(hwc) >= 2 else 640
        full_c = []
        for c in _decode_layout_candidates(full_tensors, img_w=img_w, img_h=img_h, conf=conf):
            dets = _nms(c.get('detections') or [])
            full_c.append({'mode': c.get('mode'), 'detections': dets, 'count': len(dets), 'score_saturation': _score_saturation(dets)})
        nat_c = []
        for c in _decode_layout_candidates(native_tensors, img_w=img_w, img_h=img_h, conf=conf):
            dets = _nms(c.get('detections') or [])
            nat_c.append({'mode': c.get('mode'), 'detections': dets, 'count': len(dets), 'score_saturation': _score_saturation(dets)})
        expected_family, expected_source = _expected_detection_contract(
            native_output_manifest,
            native_report,
            boundary_manifest,
            endpoint_evidence,
        )
        rows=[]
        rejected_contract_pairs=[]
        for fc in full_c:
            ref = fc.get('detections') or []
            if not ref:
                continue
            full_family = _detection_contract_family(str(fc.get('mode') or ''))
            for nc in nat_c:
                pred = nc.get('detections') or []
                if not pred:
                    continue
                native_family = _detection_contract_family(str(nc.get('mode') or ''))
                if expected_family == "unknown" or native_family != full_family or native_family != expected_family:
                    rejected_contract_pairs.append({
                        'full_mode': fc.get('mode'), 'native_mode': nc.get('mode'),
                        'full_contract_family': full_family, 'native_contract_family': native_family,
                        'expected_contract_family': expected_family,
                    })
                    continue
                m = _match_detections(
                    ref,
                    pred,
                    iou_thr=effective_policy.native_self_reference_iou_threshold,
                )
                ca = _match_detections_class_agnostic(
                    ref,
                    pred,
                    iou_thr=effective_policy.native_self_reference_iou_threshold,
                )
                same = int(str(fc.get('mode')) == str(nc.get('mode')))
                sane = _self_ref_mode_sanity(str(nc.get('mode')), nc.get('score_saturation') or {}, len(pred))
                rows.append({
                    'full_mode': fc.get('mode'), 'native_mode': nc.get('mode'),
                    'full_contract_family': full_family, 'native_contract_family': native_family,
                    'expected_contract_family': expected_family, 'contract_family_match': True,
                    'full_count': len(ref), 'native_count': len(pred),
                    'match': m, 'class_agnostic_match': ca,
                    'native_score_saturation': nc.get('score_saturation') or {},
                    'full_score_saturation': fc.get('score_saturation') or {},
                    'selection_key': [same, *sane, int(m.get('matched') or 0), float(m.get('match_ratio') or 0.0), float(m.get('mean_iou') or 0.0), -abs(len(pred)-len(ref))]
                })
        rows.sort(key=lambda r: tuple(r.get('selection_key') or []), reverse=True)
        best = rows[0] if rows else {}
        m = best.get('match') or {}
        ca = best.get('class_agnostic_match') or {}
        similarity = evaluate_detection_similarity(
            m,
            ca,
            effective_policy,
        )
        ok = bool(
            expected_family != 'unknown'
            and best
            and similarity.get('numerical_similarity_pass') is True
        )
        if expected_family == 'unknown':
            diag = 'detection_contract_metadata_unavailable'
        else:
            diag = 'native_semantic_matches_full_self_reference' if ok else ('native_boxes_match_full_but_class_score_contract_suspect' if float(ca.get('match_ratio') or 0.0) >= 0.80 else 'native_semantic_differs_from_full_self_reference')
        contract_metadata_available = expected_family != 'unknown'
        semantic_available = bool(contract_metadata_available and best)
        return {
            'available': True, 'ok': bool(ok and semantic_available), 'diagnosis': diag,
            'boundary_manifest': str(boundary_manifest), 'full_onnx': str(full),
            'self_reference_input_manifest_kind': input_manifest_kind,
            'reference_input_evidence': reference_input_evidence,
            'expected_contract_family': expected_family, 'expected_contract_source': expected_source,
            'contract_metadata_available': contract_metadata_available,
            'contract_family_match': bool(best and semantic_available), 'rejected_contract_pairs': rejected_contract_pairs[:100],
            'best': best, 'rows': rows[:30],
            'semantic_ok': bool(ok) if semantic_available else False,
            'semantic_available': semantic_available,
            'match': m, 'class_agnostic_match': ca,
            'decode_mode': best.get('native_mode') or '',
            'reference_detections': (next((c.get('detections') for c in full_c if str(c.get('mode')) == str(best.get('full_mode'))), []) or [])[:300],
            'native_detections': (next((c.get('detections') for c in nat_c if str(c.get('mode')) == str(best.get('native_mode'))), []) or [])[:300],
            **similarity,
        }
    except Exception as exc:
        return {'available': False, 'reason': f'{type(exc).__name__}: {exc}'}

def _reference_topk(report: Path | None) -> list[dict[str, Any]]:
    j = _load_json(report) or {}
    for variant in ('composed','full'):
        try:
            xs = (((j.get('viz') or {}).get(variant) or {}).get('json') or {}).get('topk') or []
            if xs:
                return xs
        except Exception:
            pass
    return []


def _reference_detections(report: Path | None) -> tuple[list[dict[str, Any]], str | None]:
    j = _load_json(report) or {}
    viz = j.get('viz') or {}
    for variant in ('composed','full'):
        try:
            d = ((viz.get(variant) or {}).get('json') or {})
            dets = d.get('detections') or []
            img = (((d.get('provenance') or {}).get('image')) or '')
            if dets:
                return dets, img
        except Exception:
            pass
    return [], None


def _find_image(
    eval_root: Path,
    image_name: str | None,
    *,
    model: str = '',
    manifest: Path | None = None,
    roots: list[Path] | None = None,
    expected_sha256: str = '',
) -> Path | None:
    """Resolve one image inside the exact model scope, never globally."""
    if not image_name:
        return None
    basename = Path(str(image_name)).name
    model_token = str(model or '').strip()
    scopes: list[Path] = []
    if manifest is not None:
        try:
            resolved_manifest = Path(manifest).expanduser().resolve()
            for parent in resolved_manifest.parents:
                if parent.name == 'benchmark_set':
                    scopes.append(parent)
                    break
        except Exception:
            pass
    if model_token:
        scopes.append(Path(eval_root) / 'models' / model_token)
        producer_root = Path(eval_root) / 'native_producers'
        if producer_root.is_dir():
            scopes.extend(
                sorted(producer_root.glob(f'*/{model_token}/benchmark_set'))
            )
        for root in roots or []:
            root_path = Path(root).expanduser()
            if root_path.name == model_token:
                scopes.append(root_path)
            scopes.extend(
                sorted(root_path.glob(f'**/{model_token}/benchmark_set'))
                if root_path.exists() else []
            )
    candidates: list[Path] = []
    seen_scopes: set[str] = set()
    for scope in scopes:
        try:
            resolved_scope = scope.resolve()
            if str(resolved_scope) in seen_scopes or not resolved_scope.exists():
                continue
            seen_scopes.add(str(resolved_scope))
            candidates.extend(
                path.resolve()
                for path in sorted(resolved_scope.rglob(basename))
                if path.is_file()
            )
        except Exception:
            continue
    unique: dict[str, Path] = {str(path): path for path in candidates}
    if not unique:
        return None
    expected = _normalize_sha256(expected_sha256)
    by_sha: dict[str, list[Path]] = {}
    for path in unique.values():
        try:
            by_sha.setdefault(_sha256_file(path), []).append(path)
        except Exception:
            continue
    if expected:
        matches = sorted(by_sha.get(expected) or [])
        return matches[0] if matches else None
    # A basename is sufficient only if every scoped copy has identical bytes.
    if len(by_sha) != 1:
        return None
    return sorted(next(iter(by_sha.values())))[0]


def _norm2d(x: np.ndarray) -> np.ndarray:
    a = np.asarray(x).astype(np.float32)
    if a.size == 0:
        return np.zeros((1,1), dtype=np.uint8)
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = float(np.min(a)), float(np.max(a))
    if hi <= lo:
        return np.zeros(a.shape, dtype=np.uint8)
    return np.clip((a - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x.astype(np.float32), -50, 50)
    return 1.0/(1.0+np.exp(-x))


def _iou(a: dict[str, Any], b: dict[str, Any]) -> float:
    ax1, ay1, ax2, ay2 = float(a['x1']), float(a['y1']), float(a['x2']), float(a['y2'])
    bx1, by1, bx2, by2 = float(b['x1']), float(b['y1']), float(b['x2']), float(b['y2'])
    ix1, iy1 = max(ax1,bx1), max(ay1,by1)
    ix2, iy2 = min(ax2,bx2), min(ay2,by2)
    iw, ih = max(0.0, ix2-ix1), max(0.0, iy2-iy1)
    inter = iw*ih
    aa = max(0.0, ax2-ax1)*max(0.0, ay2-ay1)
    bb = max(0.0, bx2-bx1)*max(0.0, by2-by1)
    return inter / max(1e-9, aa+bb-inter)


def _nms(dets: list[dict[str, Any]], iou_thr: float = 0.45, max_det: int = 300) -> list[dict[str, Any]]:
    dets = sorted(dets, key=lambda d: float(d.get('score',0)), reverse=True)
    keep: list[dict[str, Any]] = []
    for d in dets:
        if len(keep) >= max_det:
            break
        if all((int(k.get('class_id',-1)) != int(d.get('class_id',-2)) or _iou(k,d) < iou_thr) for k in keep):
            keep.append(d)
    return keep


def _clip_box(x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> tuple[float,float,float,float]:
    x1, x2 = sorted((x1,x2)); y1, y2 = sorted((y1,y2))
    return max(0,x1), max(0,y1), min(float(w),x2), min(float(h),y2)


def _prepare_score(v: float) -> float | None:
    """Return a probability-like score or None if the column is not score-like."""
    if not math.isfinite(float(v)):
        return None
    x = float(v)
    if 0.0 <= x <= 1.0001:
        return max(0.0, min(1.0, x))
    # Treat small logits as logits, but never transform coordinate-like values
    # such as 320/640 to 1.0. This was the source of the former bogus boxes.
    if -20.0 <= x <= 20.0:
        return float(_sigmoid(np.array([x], dtype=np.float32))[0])
    return None


def _prepare_class(v: float) -> int | None:
    if not math.isfinite(float(v)):
        return None
    c = int(round(float(v)))
    if 0 <= c < len(COCO80):
        return c
    return None


def _box_from_vals(vals: np.ndarray, cols: tuple[int, int, int, int], fmt: str, img_w: int, img_h: int) -> tuple[float,float,float,float] | None:
    a,b,c,d = [float(vals[i]) for i in cols]
    if fmt == 'xywh':
        # normalized coordinates are accepted; absolute pixels are expected for exported NMS.
        if max(abs(a), abs(b), abs(c), abs(d)) <= 2.0:
            a *= img_w; c *= img_w; b *= img_h; d *= img_h
        x1, y1, x2, y2 = a - c/2.0, b - d/2.0, a + c/2.0, b + d/2.0
    else:
        x1, y1, x2, y2 = a,b,c,d
        if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 2.0:
            x1 *= img_w; x2 *= img_w; y1 *= img_h; y2 *= img_h
    x1,y1,x2,y2 = _clip_box(x1,y1,x2,y2,img_w,img_h)
    if (x2-x1) < 1.0 or (y2-y1) < 1.0:
        return None
    # Reject boxes that are essentially the whole canvas unless the raw values
    # really support it; this avoids treating coordinate columns as scores/classes.
    return x1,y1,x2,y2


def _decode_nms_layout(rows: np.ndarray, spec: dict[str, Any], img_w: int, img_h: int, conf: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dets: list[dict[str, Any]] = []
    valid_score = valid_class = valid_box = 0
    invalid_class_values: list[float] = []
    for r in rows:
        vals = r.astype(np.float32)
        if not np.isfinite(vals).all():
            continue
        score = _prepare_score(float(vals[spec['score']]))
        cls = _prepare_class(float(vals[spec['cls']]))
        if score is not None: valid_score += 1
        if cls is not None: valid_class += 1
        else:
            if len(invalid_class_values) < 10:
                invalid_class_values.append(float(vals[spec['cls']]))
        box = _box_from_vals(vals, tuple(spec['box']), spec.get('box_fmt','xyxy'), img_w, img_h)
        if box is not None: valid_box += 1
        if score is None or cls is None or box is None or score < conf:
            continue
        x1,y1,x2,y2 = box
        dets.append({'x1':x1,'y1':y1,'x2':x2,'y2':y2,'score':float(score),'class_id':int(cls),'label':COCO80[int(cls)]})
    raw_count = len(dets)
    dets = _nms(dets)
    info = {
        'layout': spec['name'],
        'raw_count': raw_count,
        'nms_count': len(dets),
        'valid_score_rows': valid_score,
        'valid_class_rows': valid_class,
        'valid_box_rows': valid_box,
        'invalid_class_examples': invalid_class_values,
    }
    return dets, info


def _nms_layout_specs(ncols: int) -> list[dict[str, Any]]:
    if ncols < 6:
        return []
    specs = [
        {'name':'xyxy_score_class','box':(0,1,2,3),'score':4,'cls':5,'box_fmt':'xyxy'},
        {'name':'xyxy_class_score','box':(0,1,2,3),'score':5,'cls':4,'box_fmt':'xyxy'},
        # Some NMS exports use [y1,x1,y2,x2,...]; these variants keep the
        # tuple in x1,y1,x2,y2 order for the normal xyxy decoder.
        {'name':'yxyx_score_class','box':(1,0,3,2),'score':4,'cls':5,'box_fmt':'xyxy'},
        {'name':'yxyx_class_score','box':(1,0,3,2),'score':5,'cls':4,'box_fmt':'xyxy'},
        {'name':'class_score_xyxy','box':(2,3,4,5),'score':1,'cls':0,'box_fmt':'xyxy'},
        {'name':'score_class_xyxy','box':(2,3,4,5),'score':0,'cls':1,'box_fmt':'xyxy'},
        {'name':'class_score_yxyx','box':(3,2,5,4),'score':1,'cls':0,'box_fmt':'xyxy'},
        {'name':'score_class_yxyx','box':(3,2,5,4),'score':0,'cls':1,'box_fmt':'xyxy'},
        {'name':'class_xyxy_score','box':(1,2,3,4),'score':5,'cls':0,'box_fmt':'xyxy'},
        {'name':'score_xyxy_class','box':(1,2,3,4),'score':0,'cls':5,'box_fmt':'xyxy'},
        {'name':'class_yxyx_score','box':(2,1,4,3),'score':5,'cls':0,'box_fmt':'xyxy'},
        {'name':'score_yxyx_class','box':(2,1,4,3),'score':0,'cls':5,'box_fmt':'xyxy'},
        {'name':'xywh_score_class','box':(0,1,2,3),'score':4,'cls':5,'box_fmt':'xywh'},
        {'name':'xywh_class_score','box':(0,1,2,3),'score':5,'cls':4,'box_fmt':'xywh'},
    ]
    return specs


def _decode_nms_array(a: np.ndarray, img_w: int = 640, img_h: int = 640, conf: float = 0.25, *, return_debug: bool = False):
    """Decode common NMS arrays with conservative layout auto-detection.

    Older versions tried two layouts row-by-row and could accidentally treat a
    coordinate column as class ID and a class/constant column as score, producing
    many boxes with score 1.00 and classes such as 488/604.  This version first
    scores complete layouts and only accepts a layout if score/class/box columns
    are globally plausible.  If no plausible layout exists, it returns no boxes
    and exposes layout diagnostics instead of drawing misleading predictions.
    """
    x = np.asarray(a).squeeze()
    if x.ndim != 2 or x.shape[-1] < 6:
        return ([], {'reason':'not_nx6', 'shape': list(x.shape)}) if return_debug else []
    rows = x.reshape(-1, x.shape[-1]).astype(np.float32)
    cand_infos=[]
    best_dets: list[dict[str, Any]] = []
    best_info: dict[str, Any] | None = None
    for spec in _nms_layout_specs(rows.shape[1]):
        dets, info = _decode_nms_layout(rows, spec, img_w, img_h, conf)
        cand_infos.append(info)
        # Prefer layouts with real COCO-class IDs and score-like columns, then more detections.
        # Avoid selecting a layout if almost no rows have plausible class IDs.
        plausible = info['valid_class_rows'] >= max(1, min(5, int(0.02 * len(rows)))) and info['valid_score_rows'] >= max(1, min(5, int(0.02 * len(rows))))
        if plausible:
            key = (len(dets), info['valid_class_rows'], info['valid_score_rows'], info['valid_box_rows'])
            if best_info is None or key > (len(best_dets), best_info['valid_class_rows'], best_info['valid_score_rows'], best_info['valid_box_rows']):
                best_dets, best_info = dets, info
    if best_info is None:
        dbg={'reason':'no_plausible_nms6_layout', 'shape': list(rows.shape), 'layouts': cand_infos}
        return ([], dbg) if return_debug else []
    dbg={'reason':'ok', 'selected_layout': best_info['layout'], 'shape': list(rows.shape), 'layouts': cand_infos}
    return (best_dets, dbg) if return_debug else best_dets

def _decode_yolo_raw(a: np.ndarray, img_w: int = 640, img_h: int = 640, conf: float = 0.25) -> list[dict[str, Any]]:
    x = np.asarray(a).squeeze().astype(np.float32)
    if x.ndim == 3 and x.shape[0] in (1,):
        x = x[0]
    if x.ndim != 2:
        return []
    # [C,N] -> [N,C]
    if x.shape[0] in (84,85,86,80+4,80+5) and x.shape[1] > x.shape[0]:
        x = x.T
    if x.shape[1] < 6:
        return []
    C = x.shape[1]
    dets=[]
    for r in x:
        if not np.isfinite(r).all():
            continue
        bx = r[:4]
        if C >= 85:  # xywh obj class...
            obj = r[4]
            cls_scores = r[5:85]
            if obj < 0 or obj > 1 or np.max(cls_scores) > 1.5 or np.min(cls_scores) < -0.5:
                obj = float(_sigmoid(np.array([obj]))[0])
                cls_scores = _sigmoid(cls_scores)
            cls = int(np.argmax(cls_scores))
            score = float(obj * cls_scores[cls])
        else:        # xywh class...
            cls_scores = r[4:84]
            if np.max(cls_scores) > 1.5 or np.min(cls_scores) < -0.5:
                cls_scores = _sigmoid(cls_scores)
            cls = int(np.argmax(cls_scores))
            score = float(cls_scores[cls])
        if score < conf:
            continue
        cx,cy,w,h = map(float, bx)
        # normalized coords
        if max(abs(cx),abs(cy),abs(w),abs(h)) <= 2.0:
            cx *= img_w; w *= img_w; cy *= img_h; h *= img_h
        x1,y1,x2,y2 = _clip_box(cx-w/2, cy-h/2, cx+w/2, cy+h/2, img_w, img_h)
        if (x2-x1) < 1 or (y2-y1) < 1:
            continue
        dets.append({'x1':x1,'y1':y1,'x2':x2,'y2':y2,'score':score,'class_id':cls,'label':COCO80[cls] if cls < len(COCO80) else str(cls)})
    return _nms(dets)


def _decode_layout_candidates(tensors: dict[str, np.ndarray], img_w: int = 640, img_h: int = 640, conf: float = 0.25) -> list[dict[str, Any]]:
    """Return every plausible decode candidate instead of committing early.

    v59ct still selected the most numerous plausible Nx6 layout.  That can be
    wrong for tensors such as [1,300,6], where both [xyxy,score,class] and
    [xyxy,class,score] look numerically plausible, but only the semantic match
    to the ORT-CPU/full reference can decide.  This routine keeps all candidates
    so the caller can choose by AP50/IoU proxy when a reference exists.
    """
    out: list[dict[str, Any]] = []
    if (
        _CanonicalYoloHarness is not None
        and _canonical_postprocess_result_to_dict is not None
        and len(tensors) >= 3
    ):
        try:
            canonical_result = _CanonicalYoloHarness(
                conf_thresh=conf,
                iou_thresh=0.45,
                max_det=300,
                class_names=COCO80,
            ).postprocess(
                {str(name): np.asarray(value) for name, value in tensors.items()},
                context={'input_hw': (int(img_h), int(img_w))},
            )
            canonical_payload = _canonical_postprocess_result_to_dict(
                canonical_result
            ).get('json')
            canonical_format = (
                str(canonical_payload.get('format') or '')
                if isinstance(canonical_payload, Mapping) else ''
            )
            if (
                isinstance(canonical_payload, Mapping)
                and canonical_format in {
                    'multiscale_head', 'ultralytics_regcls',
                }
            ):
                canonical_detections = list(
                    canonical_payload.get('detections') or []
                )
                out.append({
                    'output': 'canonical_multiscale',
                    'mode': 'canonical_multiscale:raw',
                    'kind': 'raw_yolo_multiscale',
                    'detections': canonical_detections,
                    'debug': {
                        'layout': 'canonical_yolo_multiscale',
                        'decoder_format': canonical_format,
                        'raw_tensor_count': len(tensors),
                        'nms_count': len(canonical_detections),
                        'plausible_numeric': True,
                        'decoder_provenance': (
                            canonical_payload.get('provenance') or {}
                        ),
                    },
                })
        except Exception as exc:
            if (
                "yolov7_paper_raw_head_requires_model_bound_decoder_contract"
                in str(exc)
            ):
                # Current yolov7_paper evidence is claim-capable only through
                # the verified Completed-v2/model-bound decoder path above.
                # Keep a zero-detection diagnostic marker for archived legacy
                # self-reference inspection; never silently decode with the
                # historical YOLOv5/Tiny anchor default.
                out.append({
                    'output': 'canonical_multiscale',
                    'mode': 'rejected:unbound_yolov7_paper_raw_head',
                    'kind': 'rejected_unbound_yolov7_model_contract',
                    'detections': [],
                    'debug': {
                        'plausible_numeric': False,
                        'claim_capable': False,
                        'reason': (
                            'yolov7_paper_raw_head_requires_'
                            'model_bound_decoder_contract'
                        ),
                    },
                })
            # Individual legacy layouts below remain available.  Absence of a
            # supported candidate is exposed as no_supported_decode.
            pass
    for name, arr in tensors.items():
        a = np.asarray(arr)
        s = np.squeeze(a)
        if s.ndim == 2 and s.shape[-1] >= 6 and s.shape[0] <= 5000:
            rows = s.reshape(-1, s.shape[-1]).astype(np.float32)
            for spec in _nms_layout_specs(rows.shape[1]):
                dets, info = _decode_nms_layout(rows, spec, img_w, img_h, conf)
                # conservative numeric plausibility; semantic choice happens later
                plausible = (
                    info.get('valid_class_rows', 0) >= max(1, min(5, int(0.02 * len(rows)))) and
                    info.get('valid_score_rows', 0) >= max(1, min(5, int(0.02 * len(rows)))) and
                    info.get('valid_box_rows', 0) >= max(1, min(5, int(0.02 * len(rows))))
                )
                info = dict(info)
                info['plausible_numeric'] = bool(plausible)
                if plausible or dets:
                    out.append({'output': name, 'mode': f'{name}:nms:{spec["name"]}', 'kind': 'nms_layout', 'detections': dets, 'debug': info})
        raw = _decode_yolo_raw(a, img_w, img_h, conf)
        if raw:
            out.append({'output': name, 'mode': f'{name}:raw', 'kind': 'raw_yolo', 'detections': raw, 'debug': {'layout': 'raw_yolo', 'raw_count': len(raw), 'nms_count': len(_nms(raw)), 'plausible_numeric': True}})
    return out


def _choose_detection_candidate(
    tensors: dict[str, np.ndarray],
    ref_dets: list[dict[str, Any]] | None = None,
    img_w: int = 640,
    img_h: int = 640,
    conf: float = 0.25,
    *,
    expected_family: str = "unknown",
) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
    """Select a decode candidate without crossing the declared I/O contract.

    Earlier native validation could select ``output0:raw`` merely because its
    decoded boxes happened to score better than an NMS-layout candidate.  That
    is invalid when the archived boundary/output contract declares decoded NMS
    output (and vice versa).  Contract compatibility is therefore a hard gate,
    not a late diagnostic tie-breaker.
    """
    expected = str(expected_family or "unknown").strip().lower()
    if expected not in {"raw_head", "decoded_nms"}:
        expected = "unknown"
    candidates = _decode_layout_candidates(tensors, img_w=img_w, img_h=img_h, conf=conf)
    diag: dict[str, Any] = {
        'candidates': [],
        'expected_contract_family': expected,
        'rejected_contract_candidates': [],
    }
    best = None
    best_key = None
    compatible_count = 0
    for c in candidates:
        dets = _nms(c.get('detections') or [])
        dbg = dict(c.get('debug') or {})
        mode = str(c.get('mode') or '')
        family = _detection_contract_family(mode)
        compatible = expected == "unknown" or family == expected
        entry = {
            'mode': mode,
            'kind': c.get('kind'),
            'contract_family': family,
            'contract_compatible': bool(compatible),
            'count': len(dets),
            'debug': dbg,
        }
        if not compatible:
            entry['rejected_reason'] = 'detection_contract_family_mismatch'
            diag['candidates'].append(entry)
            diag['rejected_contract_candidates'].append({
                'mode': mode,
                'contract_family': family,
                'expected_contract_family': expected,
                'reason': 'detection_contract_family_mismatch',
            })
            continue
        compatible_count += 1
        if ref_dets:
            m = _match_detections(ref_dets, dets)
            ma = _match_detections_class_agnostic(ref_dets, dets)
            sat = _score_saturation(dets)
            entry['match'] = m
            entry['class_agnostic_match'] = ma
            entry['score_saturation'] = sat
            key = (
                int(m.get('matched', 0)),
                float(m.get('match_ratio', 0.0)),
                float(m.get('mean_iou', 0.0)),
                int(ma.get('matched', 0)),
                float(ma.get('match_ratio', 0.0)),
                -int(sat.get('score_ge_0999', 0)),
                -abs(len(dets) - len(ref_dets)),
            )
        else:
            scores = [float(d.get('score', 0.0)) for d in dets[:50]]
            saturated = sum(1 for x in scores if x >= 0.999)
            key = (-max(0, len(dets) - 50), -saturated, len(dets))
        entry['selection_key'] = list(key)
        diag['candidates'].append(entry)
        if best_key is None or key > best_key:
            best_key, best = key, {'dets': dets, 'mode': mode, 'entry': entry}
    diag['compatible_candidate_count'] = compatible_count
    if not best:
        diag['reason'] = 'no_contract_compatible_decode' if candidates and expected != 'unknown' else 'no_supported_decode'
        return [], diag['reason'], diag
    if ref_dets:
        m = best['entry'].get('match') or {}
        if int(m.get('matched', 0)) <= 0:
            diag['reason'] = 'no_layout_matches_reference'
            diag['selected_mode'] = best['mode']
            return [], 'no_semantic_layout_match', diag
    diag['reason'] = 'ok'
    diag['selected_mode'] = best['mode']
    diag['selected_contract_family'] = _detection_contract_family(str(best['mode']))
    return best['dets'], str(best['mode']), diag


def _decode_detections(tensors: dict[str, np.ndarray], img_w: int = 640, img_h: int = 640, conf: float = 0.25) -> tuple[list[dict[str, Any]], str]:
    dets, mode, _diag = _choose_detection_candidate(tensors, None, img_w=img_w, img_h=img_h, conf=conf, expected_family='unknown')
    return _nms(dets), mode


def _match_detections(ref: list[dict[str, Any]], pred: list[dict[str, Any]], iou_thr: float = 0.5) -> dict[str, Any]:
    used=set(); matches=[]
    for i,r in enumerate(ref):
        best=(-1,0.0)
        for j,p in enumerate(pred):
            if j in used: continue
            if int(r.get('class_id',-99)) != int(p.get('class_id',-98)): continue
            v=_iou(r,p)
            if v > best[1]: best=(j,v)
        if best[0] >= 0 and best[1] >= iou_thr:
            used.add(best[0]); matches.append(best[1])
    return {'ref_count':len(ref),'pred_count':len(pred),'matched':len(matches),'match_ratio':(len(matches)/max(1,len(ref))),'mean_iou':(float(np.mean(matches)) if matches else 0.0),'iou_threshold':iou_thr}


def _match_detections_class_agnostic(ref: list[dict[str, Any]], pred: list[dict[str, Any]], iou_thr: float = 0.5) -> dict[str, Any]:
    # IoU-only diagnostic: if this passes while class-aware matching fails,
    # boxes are likely right but class/score mapping is wrong.
    used=set(); matches=[]
    for i,r in enumerate(ref):
        best=(-1,0.0)
        for j,p in enumerate(pred):
            if j in used: continue
            v=_iou(r,p)
            if v > best[1]: best=(j,v)
        if best[0] >= 0 and best[1] >= iou_thr:
            used.add(best[0]); matches.append(best[1])
    return {'ref_count':len(ref),'pred_count':len(pred),'matched':len(matches),'match_ratio':(len(matches)/max(1,len(ref))),'mean_iou':(float(np.mean(matches)) if matches else 0.0),'iou_threshold':iou_thr,'class_agnostic':True}


def _score_saturation(dets: list[dict[str, Any]]) -> dict[str, Any]:
    scores=[float(d.get('score',0.0)) for d in dets]
    if not scores:
        return {'count':0,'score_ge_0999':0,'ratio_ge_0999':0.0}
    n=sum(1 for x in scores if x >= 0.999)
    return {'count':len(scores),'score_ge_0999':n,'ratio_ge_0999':float(n)/max(1,len(scores)), 'max_score':max(scores), 'min_score':min(scores)}


def _draw_boxes(image: Path | None, dets: list[dict[str, Any]], out: Path, title: str = '') -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image, ImageDraw, ImageFont
        if image and image.is_file():
            im = Image.open(image).convert('RGB')
        else:
            im = Image.new('RGB', (640,640), 'white')
        draw = ImageDraw.Draw(im)
        for d in dets[:200]:
            x1,y1,x2,y2 = [float(d[k]) for k in ('x1','y1','x2','y2')]
            draw.rectangle([x1,y1,x2,y2], outline='red', width=2)
            lab = str(d.get('label') or d.get('class_id')) + f" {float(d.get('score',0)):.2f}"
            draw.text((x1, max(0,y1-12)), lab, fill='black')
        if title:
            draw.text((5,5), title, fill='black')
        im.save(out)
    except Exception:
        (out.with_suffix('.json')).write_text(json.dumps({'image':str(image),'detections':dets}, indent=2), encoding='utf-8')


def _validate_tensor_dump(manifest: Path, out_dir: Path) -> dict[str, Any]:
    tensors, meta = load_dump(str(manifest))
    rows = []
    ok = True
    for name, arr in tensors.items():
        s = summarize(np.asarray(arr))
        s['name'] = name
        if not s.get('finite', True): ok = False
        rows.append(s)
    rep = {'schema':'onnx-splitpoint/native-tensor-dump-validation','schema_version':1,'ok':ok,'manifest':str(manifest),'output_count':len(rows),'outputs':rows}
    (out_dir/'tensor_dump_validation.json').write_text(json.dumps(rep, indent=2), encoding='utf-8')
    return rep



def _classification_preprocess_profile(boundary_manifest: Path, full_onnx: Path, eval_root: Path) -> tuple[str, str]:
    man = _load_json(boundary_manifest) or {}
    manifest_source = (
        'native_full_input_manifest'
        if man.get('schema') == 'onnx-splitpoint/native-full-input-dump'
        else 'boundary_manifest'
    )
    prep = man.get('preprocess') if isinstance(man.get('preprocess'), dict) else {}
    explicit = str(prep.get('normalization_profile') or prep.get('image_scale') or '').strip().lower()
    if explicit:
        return explicit, manifest_source
    contract_text = str(prep.get('preprocessing_contract') or '').lower()
    if 'imagenet' in contract_text or 'meanstd' in contract_text:
        return 'imagenet', f'{manifest_source}_contract'
    # BenchmarkSet metadata carries the task-specific preprocessing contract.
    for base in [Path(boundary_manifest).parent, Path(eval_root)]:
        try:
            for cfg in list(base.parents)[:6] + [base]:
                candidate = cfg / 'benchmark_set.json'
                if not candidate.is_file():
                    continue
                payload = _load_json(candidate) or {}
                text = json.dumps(payload, sort_keys=True).lower()
                if 'classification' in text and ('imagenet' in text or 'resnet' in text):
                    return 'imagenet', 'benchmark_set_contract'
        except Exception:
            pass
    path_text = f'{full_onnx} {boundary_manifest}'.lower()
    if 'resnet' in path_text or 'imagenet' in path_text:
        return 'imagenet', 'model_family_contract'
    return str(prep.get('ort_model_scale') or 'norm').lower(), 'legacy_ort_model_scale'


def _full_onnx_self_reference_classification(
    native_output_manifest: Path,
    eval_root: Path,
    topk: int = 5,
    native_report: Path | None = None,
    roots: list[Path] | None = None,
    endpoint_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare native classification logits against Full-ONNX on the exact native input dump.

    This mirrors the YOLO self-reference fallback: it avoids stale/incompatible
    external validation reports and validates the native producer against Full ONNX
    using the exact input_rgb_uint8.bin captured by --native-boundary-debug.
    """
    try:
        import onnxruntime as ort  # type: ignore
    except Exception as exc:
        return {'available': False, 'reason': f'onnxruntime_unavailable:{type(exc).__name__}: {exc}'}
    try:
        boundary_manifest, input_manifest_kind, input_reason = (
            _find_self_reference_input_manifest(
                native_output_manifest,
                roots=roots,
                native_report=native_report,
                endpoint_evidence=endpoint_evidence,
            )
        )
        if not boundary_manifest:
            return {'available': False, 'reason': input_reason}
        full = _find_full_onnx_for_native_manifest(native_output_manifest, eval_root, roots=roots)
        if not full:
            return {'available': False, 'reason': 'full_onnx_missing'}
        sess = ort.InferenceSession(str(full), providers=['CPUExecutionProvider'])
        inp = sess.get_inputs()[0]
        input_shape = _ort_input_shape(inp.shape)
        preprocess_profile, preprocess_source = _classification_preprocess_profile(boundary_manifest, full, eval_root)
        dtype = _ort_input_dtype(inp)
        if dtype is None:
            return {
                'available': False,
                'reason': 'onnx_input_dtype_unsupported',
                'preprocess_profile': preprocess_profile,
                'preprocess_source': preprocess_source,
            }
        reference_input_evidence: dict[str, Any] = {}
        feed = _native_input_dump_feed(
            boundary_manifest,
            input_shape,
            image_scale=preprocess_profile,
            target_dtype=dtype,
            evidence=reference_input_evidence,
        )
        if feed is None:
            return {
                'available': False,
                'reason': 'native_input_dump_unavailable',
                'preprocess_profile': preprocess_profile,
                'preprocess_source': preprocess_source,
                'reference_input_evidence': reference_input_evidence,
            }
        full_outs = [np.asarray(x) for x in sess.run(None, {inp.name: feed})]
        full_names = [o.name for o in sess.get_outputs()]
        full_tensors = {str(full_names[i] if i < len(full_names) else f'output{i}'): np.asarray(o) for i,o in enumerate(full_outs)}
        native_tensors, _meta = load_dump(str(native_output_manifest))
        fn, fa = _select_logits(full_tensors, '')
        nn, na = _select_logits(native_tensors, '')
        full_top = _topk(fa, topk)
        native_top = _topk(na, topk)
        fidx = [int(t.get('index')) for t in full_top if 'index' in t]
        nidx = [int(t.get('index')) for t in native_top if 'index' in t]
        top1 = bool(fidx and nidx and fidx[0] == nidx[0])
        top5ov = len(set(nidx[:5]).intersection(fidx[:5])) if fidx else None
        # v60i: Top-1 is the primary classification contract. Top-5 overlap is
        # diagnostic only and cannot turn a Top-1 mismatch into a claim pass.
        ok = bool(top1)
        return {
            'available': True,
            'ok': ok,
            'semantic_ok': ok,
            'semantic_available': True,
            'diagnosis': 'native_classification_matches_full_self_reference' if ok else 'native_classification_differs_from_full_self_reference',
            'source': 'full_onnx_self_reference',
            'boundary_manifest': str(boundary_manifest),
            'self_reference_input_manifest_kind': input_manifest_kind,
            'full_onnx': str(full),
            'full_output': fn,
            'native_output': nn,
            'preprocess_profile': preprocess_profile,
            'preprocess_source': preprocess_source,
            'reference_input_evidence': reference_input_evidence,
            'reference_topk': full_top,
            'native_topk': native_top,
            'top1_match': top1,
            'top5_overlap': top5ov,
        }
    except Exception as exc:
        return {'available': False, 'reason': f'{type(exc).__name__}: {exc}'}

def _validate_classification(manifest: Path, out_dir: Path, topk: int, ref_report: Path | None) -> dict[str, Any]:
    tensors, _ = load_dump(str(manifest))
    name, arr = _select_logits(tensors, '')
    tops = _topk(arr, topk)
    ref_top = _reference_topk(ref_report)
    cand_idx = [int(t.get('index')) for t in tops if 'index' in t]
    ref_idx = [int(t.get('index')) for t in ref_top if 'index' in t]
    top1_match = bool(cand_idx and ref_idx and cand_idx[0] == ref_idx[0])
    top5_overlap = len(set(cand_idx[:5]).intersection(ref_idx[:5])) if ref_idx else None
    semantic_available = bool(ref_idx)
    # v60i: require Top-1 agreement for classification self-reference.
    # Top-5 overlap remains reported as a diagnostic guardrail.
    semantic_ok = bool(top1_match) if semantic_available else 'unavailable'
    rep = {'schema':'onnx-splitpoint/native-classification-semantic-validation','schema_version':2,'ok':bool(semantic_ok is True),'manifest':str(manifest),'output':name,'topk':tops,'reference_report':str(ref_report) if ref_report else '', 'reference_topk':ref_top, 'top1_match':top1_match, 'top5_overlap':top5_overlap, 'semantic_available':semantic_available, 'semantic_ok':semantic_ok, 'summary':summarize(np.asarray(arr))}
    (out_dir/'classification_validation.json').write_text(json.dumps(rep, indent=2), encoding='utf-8')
    md=['# Native classification semantic validation','',f'Manifest: `{manifest}`','',f'Reference: `{ref_report or ""}`','',f'Semantic: `{semantic_ok}`  Top1 match: `{top1_match}`  Top5 overlap: `{top5_overlap}`','','## Candidate TopK','| rank | class index | score |','|---:|---:|---:|']
    for i,t in enumerate(tops,1): md.append(f"| {i} | {t.get('index')} | {float(t.get('score',0)):.6g} |")
    if ref_top:
        md += ['', '## Reference TopK', '| rank | class index | score/label |','|---:|---:|---|']
        for i,t in enumerate(ref_top,1): md.append(f"| {i} | {t.get('index')} | {t.get('score', t.get('label',''))} |")
    (out_dir/'classification_validation.md').write_text('\n'.join(md)+'\n', encoding='utf-8')
    return rep


def _score_map(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr).squeeze()
    if a.ndim == 5 and a.shape[0] == 1: a = a[0]
    if a.ndim == 4:
        if a.shape[-1] >= 5:
            a = np.max(a[..., 4:], axis=-1); a = np.max(a, axis=0) if a.ndim == 3 else a
        else: a = np.max(a, axis=0)
    elif a.ndim == 3:
        if a.shape[-1] >= 5: a = np.max(a[...,4:], axis=-1)
        elif a.shape[0] <= 256: a = np.max(a, axis=0)
        else: a = np.max(a, axis=-1)
    elif a.ndim == 1:
        side=int(math.ceil(math.sqrt(a.size))); a=np.pad(a,(0,side*side-a.size)).reshape(side,side)
    elif a.ndim != 2: a = a.reshape(1,-1)
    return np.asarray(a, dtype=np.float32)


def _save_png_gray(mat: np.ndarray, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image
        img = Image.fromarray(_norm2d(mat), mode='L')
        scale = max(1, min(8, 512 // max(1, max(img.size))))
        if scale > 1: img = img.resize((img.size[0]*scale, img.size[1]*scale), Image.Resampling.NEAREST)
        img.save(out)
    except Exception:
        np.save(str(out.with_suffix('.npy')), mat)



def _tensor_debug_summary(tensors: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    rows=[]
    for name, arr in tensors.items():
        a=np.asarray(arr)
        flat=a.astype(np.float32).ravel() if a.size else np.asarray([], dtype=np.float32)
        if flat.size:
            flat=np.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)
            qs=np.quantile(flat, [0,0.001,0.01,0.05,0.5,0.95,0.99,0.999,1.0]).tolist()
        else:
            qs=[]
        rows.append({
            'name': name,
            'shape': [int(x) for x in a.shape],
            'dtype': str(a.dtype),
            'size': int(a.size),
            'finite': bool(np.isfinite(a).all()) if a.size else True,
            'quantiles': qs,
            'min': float(np.nanmin(a)) if a.size else None,
            'max': float(np.nanmax(a)) if a.size else None,
            'mean': float(np.nanmean(a)) if a.size else None,
        })
    return rows


def _decode_sweep(
    tensors: dict[str, np.ndarray],
    img_w: int = 640,
    img_h: int = 640,
    ref_dets: list[dict[str, Any]] | None = None,
    *,
    expected_family: str = "unknown",
) -> dict[str, Any]:
    rows=[]
    for conf in (0.05,0.10,0.20,0.25,0.40,0.60):
        dets, mode, diag = _choose_detection_candidate(
            tensors, ref_dets, img_w=img_w, img_h=img_h, conf=conf,
            expected_family=expected_family,
        )
        rows.append({'conf':conf,'mode':mode,'count':len(dets),'top_classes':[d.get('label') for d in dets[:5]],'top_scores':[round(float(d.get('score',0)),4) for d in dets[:5]],'diagnostics':diag})
    return {'schema':'onnx-splitpoint/detection-decode-sweep','schema_version':3,'expected_contract_family':expected_family,'sweep':rows}


def _visualize_detection(
    manifest: Path,
    out_dir: Path,
    max_outputs: int,
    ref_report: Path | None,
    eval_root: Path,
    native_report: Path | None = None,
    roots: list[Path] | None = None,
    policy: AccuracyGatePolicy | None = None,
    endpoint_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    effective_policy = policy or AccuracyGatePolicy()
    tensors, _ = load_dump(str(manifest))
    manifest_json = _load_json(manifest) or {}
    manifest_image = str((manifest_json.get('input_image') or (manifest_json.get('provenance') or {}).get('image') or '')).strip() if isinstance(manifest_json, dict) else ''
    manifest_image_sha = _normalize_sha256(
        manifest_json.get('input_image_sha256')
        or (manifest_json.get('provenance') or {}).get('image_sha256')
    ) if isinstance(manifest_json, dict) else ''
    model_hint = str(
        manifest_json.get('model')
        or manifest_json.get('model_id')
        or _case_backend_precision_model_hints(str(manifest))[3]
        or _case_backend_precision_model_hints(str(native_report or ''))[3]
    ).strip()
    figs=[]
    tensor_debug = _tensor_debug_summary(tensors)
    (out_dir/'detection_tensor_debug.json').write_text(json.dumps({'manifest': str(manifest), 'input_image': manifest_image, 'outputs': tensor_debug}, indent=2), encoding='utf-8')
    for i,(name, arr) in enumerate(tensors.items()):
        if i >= max_outputs: break
        sm = _score_map(np.asarray(arr))
        out = out_dir/(f'detection_scoremap_{i:02d}_{_safe(name)}.png')
        _save_png_gray(sm, out)
        figs.append({'name':name,'figure':str(out),'source_shape':[int(x) for x in np.asarray(arr).shape],'scoremap_shape':[int(x) for x in sm.shape]})
    ref_dets, image_name = _reference_detections(ref_report)
    external_reference_detection_count = len(ref_dets)
    img = _find_image(
        eval_root, image_name, model=model_hint, manifest=manifest, roots=roots,
    )
    input_image_mismatch = False
    if manifest_image and image_name:
        same_basename = (
            Path(manifest_image).name == Path(str(image_name)).name
        )
        same_sha = True
        if manifest_image_sha:
            same_sha = bool(
                img is not None
                and _sha256_file(img) == manifest_image_sha
            )
        input_image_mismatch = not (same_basename and same_sha)
    if input_image_mismatch:
        # The native dump was produced for a different image than the reference report.
        # Do not compute semantic AP/IoU against the wrong target.
        mi = _find_image(
            eval_root,
            Path(manifest_image).name,
            model=model_hint,
            manifest=manifest,
            roots=roots,
            expected_sha256=manifest_image_sha,
        ) or Path(manifest_image)
        if mi and Path(mi).is_file():
            img = Path(mi)
    external_reference_input_mismatch = bool(input_image_mismatch)
    expected_family, expected_family_source = (
        _expected_detection_contract(
            manifest,
            native_report=native_report,
            endpoint_evidence=endpoint_evidence,
        )
    )
    sweep = _decode_sweep(
        tensors,
        ref_dets=[] if input_image_mismatch else ref_dets,
        expected_family=expected_family,
    )
    pred_dets, decode_mode, decode_diag = _choose_detection_candidate(
        tensors,
        [] if input_image_mismatch else ref_dets,
        conf=0.25,
        expected_family=expected_family,
    )
    match = (
        _match_detections(
            ref_dets,
            pred_dets,
            iou_thr=effective_policy.native_self_reference_iou_threshold,
        )
        if (ref_dets and not input_image_mismatch)
        else {
            'ref_count': 0,
            'pred_count': len(pred_dets),
            'matched': 0,
            'match_ratio': None,
            'mean_iou': None,
            'iou_threshold': effective_policy.native_self_reference_iou_threshold,
        }
    )
    class_agnostic_match = (
        _match_detections_class_agnostic(
            ref_dets,
            pred_dets,
            iou_thr=effective_policy.native_self_reference_iou_threshold,
        )
        if (ref_dets and not input_image_mismatch)
        else {
            'ref_count': 0,
            'pred_count': len(pred_dets),
            'matched': 0,
            'match_ratio': None,
            'mean_iou': None,
            'iou_threshold': effective_policy.native_self_reference_iou_threshold,
            'class_agnostic': True,
        }
    )
    semantic_available = bool(ref_dets) and (not input_image_mismatch) and decode_mode not in {'no_supported_decode','no_semantic_layout_match'}
    similarity = evaluate_detection_similarity(
        match,
        class_agnostic_match,
        effective_policy,
    )
    semantic_ok = bool(
        semantic_available
        and similarity.get('numerical_similarity_pass') is True
    )
    semantic_ok_val: Any = semantic_ok if semantic_available else 'unavailable'
    semantic_reference_source = 'external_validation_report' if semantic_available else 'none'
    self_reference_report: dict[str, Any] | None = None

    # v59ds: If the external YOLO reference/decode contract is suspect, use a
    # Full-ONNX self-reference on the exact native input dump.  This avoids
    # false negatives where Full ONNX and ORT split-chain are correct but the
    # stored validation_report.json is in a different decoder/coordinate contract.
    try:
        self_reference_report = _full_onnx_self_reference_detection(
            manifest,
            eval_root,
            conf=effective_policy.native_self_reference_confidence_threshold,
            native_report=native_report,
            roots=roots,
            policy=effective_policy,
            endpoint_evidence=endpoint_evidence,
        )
        if bool((self_reference_report or {}).get('available')):
            match = dict((self_reference_report or {}).get('match') or match)
            class_agnostic_match = dict((self_reference_report or {}).get('class_agnostic_match') or class_agnostic_match)
            pred_dets = list((self_reference_report or {}).get('native_detections') or pred_dets)
            # Keep the self-reference-selected mode/match as the semantic claim,
            # but do not turn missing contract metadata into a semantic failure.
            decode_mode = str((self_reference_report or {}).get('decode_mode') or decode_mode)
            ref_dets = list((self_reference_report or {}).get('reference_detections') or ref_dets)
            semantic_available = bool((self_reference_report or {}).get('semantic_available'))
            similarity = evaluate_detection_similarity(
                match,
                class_agnostic_match,
                effective_policy,
            )
            semantic_ok = bool(
                similarity.get('numerical_similarity_pass') is True
            ) if semantic_available else False
            semantic_ok_val = bool(semantic_ok) if semantic_available else 'unavailable'
            semantic_reference_source = 'full_onnx_self_reference' if semantic_available else 'none'
            if semantic_available:
                # The Full-ONNX probe consumes the exact native boundary input;
                # an unrelated external report image is no longer the semantic
                # reference used for this row.
                input_image_mismatch = False
    except Exception as _self_ref_exc:
        self_reference_report = {'available': False, 'reason': f'{type(_self_ref_exc).__name__}: {_self_ref_exc}'}

    # External validation is admissible only when its model-scoped source image
    # is byte-identical to the native input.  Otherwise keep the technical tensor
    # evidence without inventing a semantic comparison.
    if (
        semantic_reference_source != 'full_onnx_self_reference'
        and (
            semantic_reference_source != 'external_validation_report'
            or external_reference_input_mismatch
        )
    ):
        semantic_available = False
        semantic_ok = False
        semantic_ok_val = 'unavailable'
        semantic_reference_source = 'none'
    if (
        semantic_reference_source == 'none'
        and external_reference_input_mismatch
    ):
        # A known-wrong image is not a zero-quality measurement.  Keep the
        # external row count as diagnostics, but make the scientific same-input
        # comparison explicitly unavailable until the Full-ONNX replay or an
        # exact image identity is available.
        similarity = evaluate_detection_similarity(
            {
                'ref_count': 0,
                'pred_count': len(pred_dets),
                'matched': 0,
                'match_ratio': None,
                'mean_iou': None,
                'iou_threshold': (
                    effective_policy.native_self_reference_iou_threshold
                ),
            },
            {
                'ref_count': 0,
                'pred_count': len(pred_dets),
                'matched': 0,
                'match_ratio': None,
                'mean_iou': None,
                'iou_threshold': (
                    effective_policy.native_self_reference_iou_threshold
                ),
                'class_agnostic': True,
            },
            effective_policy,
        )
        similarity['numerical_similarity_reason'] = (
            'reference_input_identity_mismatch'
        )

    # An excessive number of high-confidence boxes is a decoder/postprocess red flag.
    decoder_warning = None
    if 'decode_diag' in locals() and decode_diag.get('reason') and decode_diag.get('reason') != 'ok':
        decoder_warning = str(decode_diag.get('reason'))
    if len(pred_dets) >= 150 and (match.get('match_ratio') or 0.0) < 0.1:
        decoder_warning = 'too_many_unmatched_boxes_likely_decode_or_postprocess_error'
    if input_image_mismatch:
        decoder_warning = 'native_input_image_mismatch_reference_image'
    elif len(pred_dets) == 0 and ref_dets:
        decoder_warning = 'no_native_detections_but_reference_has_detections'
    elif ref_dets and (match.get('matched',0) == 0) and class_agnostic_match.get('matched',0) > 0:
        decoder_warning = 'boxes_overlap_reference_but_class_or_score_mapping_wrong'
    elif ref_dets and (match.get('matched',0) == 0) and len(pred_dets) > 0:
        decoder_warning = 'decoded_boxes_do_not_overlap_reference_boundary_or_postprocess_suspect'

    # Strong diagnostic for the current native uint8 boundary bridge: if nearly all
    # selected detections have score≈1.0 while semantic and class-agnostic match are
    # very low, the problem is usually not an NMS column order anymore.  It is more
    # likely a boundary contract / quantization / scale-zero-point mismatch feeding
    # the TensorRT part2.  Keep claim_ok false and expose this as a first-class
    # warning so thesis tables do not hide it as a mere visual failure.
    boundary_contract_warning = None
    try:
        selected = None
        for c in (decode_diag or {}).get('candidates', []):
            if str(c.get('mode')) == str(decode_mode):
                selected = c; break
        sat = (selected or {}).get('score_saturation') or {}
        sat_ratio = float(sat.get('ratio_ge_0999') or 0.0)
        if (semantic_reference_source != 'full_onnx_self_reference' and ref_dets and not input_image_mismatch and
            float(match.get('match_ratio') or 0.0) < 0.2 and
            float(class_agnostic_match.get('match_ratio') or 0.0) < 0.2 and
            sat_ratio >= 0.8):
            boundary_contract_warning = 'boundary_quantization_or_score_scale_suspect'
            decoder_warning = boundary_contract_warning
    except Exception:
        boundary_contract_warning = None

    overlay = out_dir/'detection_boxes_overlay.png'
    _draw_boxes(img, pred_dets, overlay, title=f'native decoded {len(pred_dets)} boxes mode={decode_mode}')
    ref_overlay = out_dir/'detection_reference_overlay.png'
    if ref_dets:
        _draw_boxes(img, ref_dets, ref_overlay, title=f'reference {len(ref_dets)} boxes')
    rep = {
        'schema':'onnx-splitpoint/native-detection-semantic-validation',
        'schema_version':5,
        'ok':bool(semantic_ok_val is True),
        # This child artefact evaluates numerical/semantic similarity only.
        # Final ``claim_ok`` is owned by the row-level structural gate below.
        'claim_ok': False,
        'semantic_claim_ok': bool(semantic_ok_val is True),
        'claim_scope': 'semantic_similarity_subcheck_only',
        'manifest':str(manifest),
        'reference_report':str(ref_report) if ref_report else '',
        'semantic_reference_source': semantic_reference_source,
        'self_reference_report': self_reference_report or {},
        'reference_image':str(img) if img else '',
        'native_input_image': manifest_image,
        'input_image_mismatch': bool(input_image_mismatch),
        'external_reference_input_mismatch': (
            external_reference_input_mismatch
        ),
        'external_reference_detection_count': (
            external_reference_detection_count
        ),
        'decode_mode':decode_mode,
        'expected_contract_family': expected_family,
        'expected_contract_source': expected_family_source,
        'selected_contract_family': _detection_contract_family(decode_mode),
        'contract_family_match': bool(expected_family == 'unknown' or _detection_contract_family(decode_mode) == expected_family),
        'decode_warning': decoder_warning,
        'boundary_contract_warning': boundary_contract_warning,
        'decode_diagnostics': decode_diag if 'decode_diag' in locals() else {},
        'decode_sweep': sweep,
        'tensor_debug': tensor_debug,
        'detections':pred_dets[:300],
        'reference_detections':ref_dets[:300],
        'semantic_available':semantic_available,
        'semantic_ok':semantic_ok_val,
        'match':match,
        'class_agnostic_match': class_agnostic_match,
        'box_overlay':str(overlay),
        'reference_overlay':str(ref_overlay) if ref_dets else '',
        'figures':figs
        ,
        **similarity,
    }
    (out_dir/'detection_visual_validation.json').write_text(json.dumps(rep, indent=2), encoding='utf-8')
    md=['# Native detection semantic validation','',f'Manifest: `{manifest}`','',f'Reference: `{ref_report or ""}`','',f'Semantic source: `{semantic_reference_source}`','',f'Decode mode: `{decode_mode}`','',f'Decode warning: `{decoder_warning or ""}`','',f'Boundary warning: `{boundary_contract_warning or ""}`','',f'Semantic: `{semantic_ok_val}`  match_ratio={float(match.get("match_ratio") or 0.0):.3f} matched={match.get("matched")}/{match.get("ref_count")} pred={match.get("pred_count")}','',f'Class-agnostic IoU: match_ratio={float(class_agnostic_match.get("match_ratio") or 0.0):.3f} matched={class_agnostic_match.get("matched")}/{class_agnostic_match.get("ref_count")}','','Artifacts:','',f'- Native box overlay: `{overlay.name}`']
    if ref_dets: md.append(f'- Reference overlay: `{ref_overlay.name}`')
    md += ['', '## Decode sweep', '| conf | mode | count | top classes | top scores |', '|---:|---|---:|---|---|']
    for sw in sweep.get('sweep',[]):
        md.append(f"| {sw.get('conf')} | `{sw.get('mode')}` | {sw.get('count')} | {sw.get('top_classes')} | {sw.get('top_scores')} |")
    md += ['', '## Tensor debug', '| output | shape | dtype | min | max | mean | scoremap |','|---|---|---|---:|---:|---:|---|']
    fig_by_name={f['name']: f for f in figs}
    for td in tensor_debug:
        fn=Path(fig_by_name.get(td['name'],{}).get('figure','')).name if td['name'] in fig_by_name else ''
        md.append(f"| `{td['name']}` | `{td['shape']}` | `{td['dtype']}` | {td.get('min')} | {td.get('max')} | {td.get('mean')} | `{fn}` |")
    (out_dir/'detection_visual_validation.md').write_text('\n'.join(md)+'\n', encoding='utf-8')
    return rep


def _apply_smoke_diagnostic_policy(rows: list[dict[str, Any]]) -> None:
    """Clamp all Smoke rows to technical-only, never claimable evidence."""
    for rec in rows:
        from onnx_splitpoint_tool.native_job_identity import known_build_exclusion
        if known_build_exclusion(rec):
            rec.update(diagnostic_only=True, status='excluded_known_build',
                       claim_eligible=False, claim_ok=False,
                       performance_claim_eligible=False, energy_claim_eligible=False,
                       eligible_for_ranking=False)
            continue
        metric_miss = str(
            rec.get('accuracy_gate_decision') or ''
        ).strip().lower() in {'fail', 'failed'}
        backend = str(rec.get('backend') or '').strip().lower()
        central_binding_required = bool(
            rec.get('native_split_quality_required') is True
            or backend in {'native_full_tensorrt', 'native_full_hailo8', 'native_full_hailo10h', 'native_full_deepx'}
            or rec.get('quality_first_producer_identity_sha256')
        )
        validated_screening_gap = (
            _validated_split_screening_postprocess_gap(rec)
        )
        technical_valid = bool(
            rec.get('tensor_ok') is True
            and rec.get('buildable', True) is not False
            and rec.get('runtime_executable', True) is not False
            and (
                rec.get(
                    'structural_contract_pass',
                    rec.get('contract_consistent', True),
                ) is not False
                or validated_screening_gap
            )
            and (
                not central_binding_required
                or rec.get('central_quality_evidence_verified') is True
            )
            and not rec.get('error')
        )
        rec.update({
            'diagnostic_only': True,
            'claim_eligible': False,
            'claim_ok': False,
            'e2e_claim_eligible': False,
            'performance_claim_eligible': False,
            'energy_claim_eligible': False,
            'scientific_claim_eligible': False,
            'thesis_claim_eligible': False,
            'eligible_for_ranking': False,
            'ranking_eligible': False,
            'performance_eligible': False,
            'energy_eligible': False,
            'pareto_eligible': False,
            'thesis_comparison_eligible': False,
            'thesis_valid': False,
            'accuracy_gate_enforced': False,
            'metric_threshold_miss_warning': metric_miss,
            'claim_exclusion_reason': 'smoke_diagnostic_only',
            'ok': technical_valid,
            'status': (
                'diagnostic_metric_threshold_warning'
                if technical_valid and metric_miss
                else 'diagnostic_technical_pass'
                if technical_valid
                else 'diagnostic_technical_error'
            ),
        })


def _apply_completed_v2_semantic_binding(
    row: dict[str, Any],
    self_reference: Mapping[str, Any] | None,
) -> None:
    """Expose portable replay status and clamp it out of every claim axis."""
    source = (
        self_reference
        if isinstance(self_reference, Mapping)
        else {}
    )
    for field in (
        "semantic_result_binding_status",
        "exact_completed_result_identity_bound",
        "portable_result_hash_mismatch",
        "completed_v2_exact_result_claim_binding",
        "completed_v2_semantic_evidence_tier",
        "performance_hotloop_result_sha256",
        "native_completed_result_sha256",
    ):
        if field in source:
            row[field] = source.get(field)
    if source.get("portable_result_hash_mismatch") is True:
        row.update({
            "claim_eligible": False,
            "e2e_claim_eligible": False,
            "e2e_contract_reason": "portable_result_hash_mismatch",
            "performance_claim_eligible": False,
            "energy_claim_eligible": False,
            "scientific_claim_eligible": False,
            "thesis_claim_eligible": False,
            "eligible_for_ranking": False,
            "ranking_eligible": False,
            "performance_eligible": False,
            "energy_eligible": False,
            "pareto_eligible": False,
            "thesis_comparison_eligible": False,
            "thesis_valid": False,
        })


def _finalize_structural_claim_contract(row: dict[str, Any]) -> None:
    """Finalize legacy claim/status fields after the last accuracy gate.

    ``claim_ok`` remains a technical semantic/contract compatibility field.
    Dataset task quality and ranking eligibility stay on their independent
    axes and are deliberately not folded into this compatibility alias.
    """
    structure = row.get(
        'structural_contract_pass',
        row.get('contract_consistent'),
    )
    source_claim = row.get('claim_ok_source')
    if row.get('claim_ok') is True:
        source_claim = True
        row['claim_ok_source'] = True
    elif not isinstance(source_claim, bool):
        source_claim = row.get('claim_ok') is True
        row['claim_ok_source'] = bool(source_claim)
    structurally_valid = structure is True
    final_claim = bool(
        row.get('claim_ok') is True
        and source_claim is True
        and structurally_valid
    )
    row['claim_structural_gate_pass'] = structurally_valid
    row['claim_ok_structural_clamped'] = bool(
        row.get('claim_ok_structural_clamped') is True
        or (source_claim is True and not structurally_valid)
    )
    row['claim_ok'] = final_claim
    row['ok'] = final_claim
    if (
        not final_claim
        and str(row.get('status') or '').strip().lower() == 'claim_ok'
    ):
        row['status'] = (
            'structural_contract_failed'
            if structure is False
            else 'structural_contract_unavailable'
        )


_SCREENING_RAW_HEAD_SPLIT_BACKENDS = {
    'deepx_to_trt', 'hailo8_to_trt', 'hailo10h_to_trt',
}
_SCREENING_RAW_HEAD_UNRESOLVED_REASON = (
    'raw_head_decoder_postprocess_contract_unresolved'
)


def _validated_split_screening_postprocess_gap(
    row: Mapping[str, Any],
) -> bool:
    """Recognize one narrow, non-technical Split screening evidence gap.

    A successful Split producer may seal and validate its raw accelerator
    endpoint while deliberately omitting a *timed* decoder/NMS completion
    tail.  That makes the row unavailable for completed-task claims, ranking
    and Energy; it does not turn an otherwise verified runtime and semantic
    screening decision into a technical failure.

    This exception is deliberately fail-closed.  It is admitted only for the
    three current Split producers, only under a non-final screening policy,
    and only after the portable Quality-FIRST verifier has rechecked the
    engine/command/boundary/output binding.  Missing or conflicting evidence,
    Native Full rows and every other structural failure remain technical.
    """
    reason = _SCREENING_RAW_HEAD_UNRESOLVED_REASON
    if any(
        str(row.get(field) or '').strip().lower() != reason
        for field in (
            'structural_contract_reason',
            'contract_gate_reason',
            'claim_structural_gate_reason',
        )
    ):
        return False
    if (
        row.get('structural_contract_pass') is not False
        or row.get('contract_consistent') is not False
        or row.get('claim_structural_gate_pass') is not False
        or row.get('claim_ok_structural_clamped') is not True
    ):
        return False

    backend = str(row.get('backend') or '').strip().lower()
    case = str(row.get('case') or row.get('case_id') or '').strip().lower()
    if (
        backend not in _SCREENING_RAW_HEAD_SPLIT_BACKENDS
        or _is_native_full_row(dict(row))
        or re.fullmatch(r'b\d+', case) is None
        or str(row.get('task') or '').strip().lower() != 'detection'
        or str(row.get('stage') or '').strip().lower() != 'raw_head'
        or str(row.get('contract_family') or '').strip().lower() != 'raw_head'
        or str(row.get('accelerator_output_stage') or '').strip().lower()
        != 'raw_head'
        or str(
            row.get('accelerator_output_contract_family') or ''
        ).strip().lower() != 'raw_head'
    ):
        return False

    endpoint_sha, endpoint_sha_valid = _consistent_sha256(
        row.get('endpoint_contract_hash'),
        row.get('accelerator_endpoint_contract_hash'),
        required=True,
    )
    if (
        row.get('endpoint_contract_complete') is not True
        or not endpoint_sha_valid or not endpoint_sha
        or not _normalize_sha256(row.get('output_manifest_sha256'))
        or not _normalize_sha256(row.get('native_command_contract_sha256'))
        or not _normalize_sha256(
            row.get('native_split_quality_binding_sha256')
        )
    ):
        return False
    for field in (
        'output_endpoint_attestation',
        'accelerator_output_endpoint_attestation',
    ):
        attestation = row.get(field)
        if not isinstance(attestation, Mapping) or not (
            attestation.get('attested') is True
            and str(attestation.get('status') or '').strip().lower()
            == 'passed'
            and str(attestation.get('stage') or '').strip().lower()
            == 'raw_head'
            and _normalize_sha256(
                attestation.get('endpoint_contract_hash')
            ) == endpoint_sha
        ):
            return False

    if any(
        row.get(field) is not True
        for field in (
            'native_row_ok', 'report_ok', 'buildable',
            'runtime_executable', 'tensor_ok', 'strict_tensor_ok',
        )
    ) or str(
        row.get('execution_validation_status') or ''
    ).strip().lower() != 'passed':
        return False

    semantic_ok = row.get('semantic_ok')
    if (
        row.get('semantic_available') is not True
        or not isinstance(semantic_ok, bool)
        or row.get('numerical_similarity_pass') is not semantic_ok
        or row.get('self_reference_available') is not True
        or row.get('self_reference_ok') is not semantic_ok
        or str(
            row.get('semantic_validation_status') or ''
        ).strip().lower() != ('passed' if semantic_ok else 'failed')
        or str(
            row.get('numerical_similarity_status') or ''
        ).strip().lower() != ('passed' if semantic_ok else 'failed')
    ):
        return False

    policy = row.get('accuracy_gate_policy')
    quality_gate = row.get('task_quality_gate')
    if not isinstance(policy, Mapping) or not isinstance(
        quality_gate, Mapping,
    ):
        return False
    policy_sha, policy_sha_valid = _consistent_sha256(
        row.get('accuracy_gate_policy_sha256'),
        row.get('task_quality_policy_sha256'),
        row.get('runtime_quality_gate_policy_sha256'),
        quality_gate.get('policy_sha256'),
        required=True,
    )
    decision = str(
        row.get('accuracy_gate_decision') or ''
    ).strip().lower()
    task_status = str(
        row.get('task_quality_status') or ''
    ).strip().lower()
    gate_decision = str(
        quality_gate.get('decision') or ''
    ).strip().lower()
    gate_status = str(
        quality_gate.get('status') or gate_decision
    ).strip().lower()
    quality_binding_axis_valid = bool(
        row.get('precision_quality_binding_verified') is True
        if 'precision_quality_binding_verified' in row
        else row.get('central_quality_evidence_verified') is True
    )
    if decision in {'pass', 'reference_close', 'accuracy_loss', 'not_estimable'}:
        decision_aliases_valid = bool(
            row.get('task_quality_pass') is True
            and row.get('task_valid') is True
            and row.get('accuracy_gate_pass') is True
            and row.get('precision_quality_verified') is True
            and quality_binding_axis_valid
        )
    elif decision == 'fail':
        decision_aliases_valid = bool(
            row.get('task_quality_pass') is False
            and row.get('task_valid') is False
            and row.get('accuracy_gate_pass') is False
            and row.get('precision_quality_verified') in {False, True}
            and quality_binding_axis_valid
        )
    elif decision == 'inconclusive':
        decision_aliases_valid = bool(
            str(row.get('task_quality_pass') or '').strip().lower()
            == 'inconclusive'
            and str(row.get('task_valid') or '').strip().lower()
            == 'inconclusive'
            and row.get('accuracy_gate_pass') is False
            and row.get('precision_quality_verified') in {False, True}
            and quality_binding_axis_valid
        )
    else:
        decision_aliases_valid = False
    if (
        not policy_sha_valid or not policy_sha
        or _canonical_json_sha256(policy) != policy_sha
        or quality_gate.get('policy') != policy
        or str(row.get('accuracy_gate_tier') or '').strip().lower()
        != 'screening'
        or str(quality_gate.get('tier') or '').strip().lower()
        != 'screening'
        or str(policy.get('dataset_tier') or '').strip().lower()
        != 'screening'
        or policy.get('frozen_before_final_campaign') is not False
        or policy.get('screening_eligible_for_ranking') is not False
        or decision not in {'pass', 'fail', 'inconclusive', 'reference_close', 'accuracy_loss', 'not_estimable'}
        or task_status != decision
        or gate_decision != decision
        or gate_status != decision
        or not decision_aliases_valid
        or row.get('accuracy_gate_policy_match') is not True
        or row.get('central_quality_evidence_verified') is not True
        or str(
            row.get('central_quality_binding_status') or ''
        ).strip().lower() != 'exact_identity_match'
        or str(
            row.get('quality_first_binding_status') or ''
        ).strip().lower()
        != 'central_native_exact_engine_command_boundary_match'
        or str(
            row.get('native_split_quality_consumer_status') or ''
        ).strip().lower()
        != 'exact_quality_native_engine_command_and_boundary_match'
        or bool(row.get('quality_first_binding_errors'))
    ):
        return False

    # Re-run the portable verifier instead of trusting any convenience
    # Boolean.  It validates the sealed command, engine and boundary identity,
    # the exact semantic output/boundary hashes and the consumer attestation.
    if bind_quality_to_native_split is None:
        return False
    try:
        verified_binding, binding_status = bind_quality_to_native_split(
            native_row=row,
            quality_binding=row.get('native_split_quality_binding'),
            verification_mode='portable',
        )
    except Exception:
        return False
    if (
        verified_binding is None
        or binding_status
        != 'portable_binding_command_and_consumer_attestation_exact_match'
    ):
        return False

    if backend in _HAILO_TRT_SPLIT_BACKENDS:
        if (
            row.get('interface_contract_pass') is not True
            or row.get('interface_check_pass') is not True
            or str(
                row.get('interface_contract_status') or ''
            ).strip().lower()
            != 'verified_native_command_metadata_boundary_and_bridge'
        ):
            return False
    elif (
        row.get('interface_contract_pass') is False
        or row.get('interface_check_pass') is False
    ):
        return False

    # The gap must be unavailable, never a failed or conflicting completed
    # tail.  A populated completion record with this structural reason is a
    # contradiction and therefore remains technical.
    if (
        row.get('completed_task_endpoint_attested') not in (None, '')
        or str(
            row.get('completed_task_endpoint_attestation_status') or ''
        ).strip()
        or row.get('completed_task_endpoint_attestation') not in (None, {})
        or row.get('completed_task_endpoint_contract') not in (None, {})
        or str(row.get('completed_task_stage') or '').strip()
        or str(row.get('completed_task_contract_family') or '').strip()
        or str(
            row.get('completed_task_endpoint_contract_hash') or ''
        ).strip()
        or str(
            row.get('completed_task_output_endpoint_id') or ''
        ).strip()
        or str(
            row.get('host_postprocessing_evidence_status') or ''
        ).strip().lower() != 'unavailable'
        or str(
            row.get('host_postprocessing_evidence_source') or ''
        ).strip().lower() != 'none'
        or row.get('host_postprocessing_legacy_alias_conflict') is not False
    ):
        return False

    # This classification exception never grants any claim axis.
    diagnostic_technical_ok = bool(
        row.get('diagnostic_only') is True
        and row.get('ok') is True
        and str(row.get('status') or '').strip().lower() in {
            'diagnostic_technical_pass',
            'diagnostic_metric_threshold_warning',
        }
    )
    if (
        row.get('claim_ok') is not False
        or row.get('eligible_for_ranking') is not False
        or (
            row.get('ok') is not False
            and not diagnostic_technical_ok
        )
    ) or any(
        row.get(field) is True
        for field in (
            'claim_eligible', 'performance_claim_eligible',
            'energy_claim_eligible', 'scientific_claim_eligible',
            'thesis_claim_eligible', 'thesis_comparison_eligible',
            'ranking_eligible', 'performance_eligible', 'energy_eligible',
            'pareto_eligible', 'thesis_valid',
        )
    ):
        return False
    return True


def _technical_quality_error(row: Mapping[str, Any]) -> bool:
    """Classify wiring/output failures independently from metric decisions."""
    from onnx_splitpoint_tool.native_job_identity import known_build_exclusion
    if known_build_exclusion(row):
        return False
    # An explicit failed byte/result binding is a processing error. Ordinary
    # numeric similarity/accuracy thresholds remain negative observations.
    binding_reason = str(row.get('self_reference_reason') or '')
    if any(token in binding_reason for token in (
        'fast_oracle_dumped_sentinel_mismatch',
        'fast_oracle_dump_selection_',
        'fast_oracle_dump_repetition_selection_mismatch',
        'fast_oracle_dumped_frame_',
        'semantic_result_hash_mismatch',
        'input_manifest_sha256_mismatch',
    )):
        return True
    if row.get('portable_result_hash_mismatch') is True or row.get('vendor_full_quality_provenance_conflict') is True:
        return True
    if str(row.get('status') or '') == 'diagnostic_technical_error':
        return True
    if row.get('error'):
        return True
    if str(row.get('status') or '').strip().lower() in {
        'native_row_not_ok', 'missing_dump', 'error',
    }:
        return True
    if row.get('tensor_ok') is False:
        return True
    if row.get('buildable') is False or row.get('runtime_executable') is False:
        return True
    if row.get(
        'structural_contract_pass',
        row.get('contract_consistent'),
    ) is False:
        if not _validated_split_screening_postprocess_gap(row):
            return True
    backend = str(row.get('backend') or '').strip().lower()
    central_binding_required = bool(
        row.get('native_split_quality_required') is True
        or backend in {'native_full_tensorrt', 'native_full_hailo8', 'native_full_hailo10h', 'native_full_deepx'}
        or row.get('quality_first_producer_identity_sha256')
    )
    return bool(
        central_binding_required
        and row.get('central_quality_evidence_verified') is not True
    )


def main() -> int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--summary', required=True, help='native_producer_summary.json')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--root', action='append', default=[], help='Additional root(s) to rebase/search for native dumps')
    ap.add_argument('--topk', type=int, default=5)
    ap.add_argument('--max-detection-outputs', type=int, default=6)
    ap.add_argument('--accuracy-gate-policy', default='strict_task_gate_v1',
                    help='Named task-validity gate policy. Default keeps contract-only self-reference rows out of ranking eligibility unless dataset accuracy evidence is present.')
    ap.add_argument('--quality-gate-json', default='',
                    help='Exact Evaluation Profile quality_gate JSON/path. When present it replaces local margin/bootstrap defaults and is hashed in the output.')
    ap.add_argument('--central-quality-summary', default='',
                    help='Central management quality summary. Decisions are accepted only through an exact model/case/run/setup/endpoint/precision join.')
    ap.add_argument('--classification-max-top1-drop', type=float, default=0.01,
                    help='Dataset Top-1 gate: maximum allowed absolute drop versus full/reference. Default: 0.01 = one percentage point.')
    ap.add_argument('--classification-max-top5-drop', type=float, default=0.01,
                    help='Dataset Top-5 gate: maximum allowed absolute drop versus full/reference. Default: 0.01.')
    ap.add_argument('--detection-max-ap50-drop', type=float, default=0.01,
                    help='Dataset AP50 gate: maximum allowed absolute drop versus full/reference. Default: 0.01.')
    ap.add_argument('--detection-max-ap-drop', type=float, default=0.01,
                    help='Dataset AP gate: maximum allowed absolute drop versus full/reference. Default: 0.01.')
    ap.add_argument('--allow-contract-only-ranking', action='store_true',
                    help='Allow Full-ONNX self-reference-only native contract checks to become eligible_for_ranking. Disabled by default.')
    ns=ap.parse_args()
    summary_path=Path(ns.summary).expanduser().resolve()
    eval_root=_eval_root_from_summary(summary_path)
    split_quality_authority = _native_split_authority(eval_root)
    data=_load_json(summary_path) or {}
    roots=[eval_root]
    for r in ns.root: roots.append(Path(r).expanduser().resolve())
    out=Path(ns.out_dir).expanduser().resolve(); out.mkdir(parents=True, exist_ok=True)
    diagnostic_only = False
    quality_gate_source = str(ns.quality_gate_json or '').strip()
    if quality_gate_source:
        try:
            quality_gate_payload = (
                _load_json(Path(quality_gate_source).expanduser().resolve())
                if not quality_gate_source.lstrip().startswith('{')
                else json.loads(quality_gate_source)
            )
            enforcement = (
                quality_gate_payload.get('enforcement')
                if isinstance(quality_gate_payload, Mapping) else {}
            )
            diagnostic_only = bool(
                isinstance(quality_gate_payload, Mapping)
                and quality_gate_payload.get('diagnostic_only') is True
                and isinstance(enforcement, Mapping)
                and str(enforcement.get('technical_quality_error') or '').startswith('partial_continue')
            )
        except Exception:
            diagnostic_only = False
    if quality_gate_source:
        gate_policy = AccuracyGatePolicy.from_mapping(str(ns.quality_gate_json).strip())
        gate_policy = replace(gate_policy, contract_only_eligible_for_ranking=bool(ns.allow_contract_only_ranking))
    else:
        gate_policy = AccuracyGatePolicy(
            classification_max_top1_drop=float(ns.classification_max_top1_drop),
            classification_max_top5_drop=float(ns.classification_max_top5_drop),
            detection_max_ap50_drop=float(ns.detection_max_ap50_drop),
            detection_max_ap_drop=float(ns.detection_max_ap_drop),
            contract_only_eligible_for_ranking=bool(ns.allow_contract_only_ranking),
        )
    central_payload = (
        _load_strict_json_object(
            Path(ns.central_quality_summary).expanduser().resolve()
        )
        if str(ns.central_quality_summary or '').strip() else {}
    )
    central_results = [
        dict(item) for item in list((central_payload or {}).get('results') or [])
        if isinstance(item, dict)
    ] if isinstance(central_payload, dict) else []
    rows=[]
    for idx,row in enumerate(data.get('rows',[]) or []):
        if not _row_ok(row.get('ok')):
            _model = str(row.get('model') or '')
            _failure = str(row.get('failure_reason') or row.get('reason') or row.get('error') or 'native_row_not_ok')
            _rec={
                'row_index':idx, 'backend':row.get('backend'), 'model':_model,
                'case':row.get('case') or row.get('case_id'), 'precision':row.get('precision') or '',
                'setup_id':str(row.get('setup_id') or ''),
                'comparison_backend':str(row.get('comparison_backend') or ''),
                'ok':False, 'claim_ok':False, 'status':'native_row_not_ok',
                'task':_task_for_model(_model), 'dump_status':'skipped',
                'failure_reason':_failure,
                'status_detail':str(row.get('status_detail') or _failure),
                'error':str(row.get('error') or _failure),
                'returncode':row.get('returncode'),
                'timed_out':bool(row.get('timed_out')),
                'stdout_tail':str(row.get('stdout_tail') or ''),
                'stderr_tail':str(row.get('stderr_tail') or ''),
                'unsupported_reason':str(row.get('unsupported_reason') or ''),
                'recommended_action':str(row.get('recommended_action') or ''),
                'steps':row.get('steps') or [],
                'central_quality_evidence_verified':False,
                'precision_quality_verified':False,
                'precision_quality_binding_verified':False,
                'task_quality_observation_valid':False,
            }
            from onnx_splitpoint_tool.native_job_identity import known_build_exclusion
            if known_build_exclusion(row):
                # Carry the original negative evidence, otherwise downstream
                # validation cannot distinguish an exclusion from a failure.
                _rec = {**copy.deepcopy(row), **_rec}
                _rec.update(status='excluded_known_build', quality_applicability='not_applicable_build_excluded')
            if apply_native_split_quality_authority is not None:
                apply_native_split_quality_authority(
                    _rec, split_quality_authority,
                )
            _copy_completed_v2_projection(_rec, row)
            apply_accuracy_gate_to_row(_rec, gate_policy)
            _enforce_historical_split_diagnostic_only(_rec)
            rows.append(_rec)
            continue
        model=str(row.get('model') or '')
        case=str(row.get('case') or row.get('case_id') or '')
        backend=str(row.get('backend') or '')
        precision=str(row.get('precision') or row.get('native_precision') or row.get('trt_precision') or '')
        task=_task_for_model(model)
        # Keep validation artifacts precision-separated.  Without this, two valid
        # Hailo8->TRT variants such as float32_layout_fp16 and uint8_dequant_fp16
        # write into the same hailo8_to_trt__model__case directory and the later
        # report can silently mask the former.
        row_name=f"{_safe(backend)}__{_safe(model)}__{_safe(case)}" + (f"__{_safe(precision)}" if precision else '')
        if _is_native_full_row(row):
            row_name += (
                f"__setup_{_safe(str(row.get('setup_id') or '') or 'unspecified')}"
                f"__comparison_{_safe(str(row.get('comparison_backend') or '') or 'unspecified')}"
            )
        row_dir=out/row_name; row_dir.mkdir(parents=True, exist_ok=True)
        report=_find_report(row, roots)
        dump, dump_status=_find_dump(report, roots, row)
        ref_report=_find_reference_report(eval_root, model, case, task)
        rec={'row_index':idx,'backend':backend,'model':model,'case':case,'precision':precision,'setup_id':str(row.get('setup_id') or ''),'comparison_backend':str(row.get('comparison_backend') or ''),'task':task,'native_report':str(report) if report else '', 'reference_report':str(ref_report) if ref_report else '', 'dump_manifest':str(dump) if dump else '', 'dump_status':dump_status, 'ok':False, 'claim_ok':False, 'status':''}
        rec.update({
            'execution_precision': str(row.get('execution_precision') or ''),
            'full_runtime_precision': str(row.get('full_runtime_precision') or ''),
            'runtime_precision_identity': _runtime_precision_identity(row),
            'native_command_contract': dict(row.get('native_command_contract') or {})
            if isinstance(row.get('native_command_contract'), dict) else {},
            'full_command_contract': dict(row.get('full_command_contract') or {})
            if isinstance(row.get('full_command_contract'), dict) else {},
            'quality_first_producer_identity': dict(row.get('quality_first_producer_identity') or {})
            if isinstance(row.get('quality_first_producer_identity'), dict) else {},
            'quality_first_producer_identity_sha256': str(
                row.get('quality_first_producer_identity_sha256') or ''
            ),
            'native_command_contract_sha256': str(row.get('native_command_contract_sha256') or ''),
            'full_command_contract_sha256': str(row.get('full_command_contract_sha256') or ''),
            'input_image_sha256': str(row.get('input_image_sha256') or ''),
            'model_sha256': str(row.get('model_sha256') or row.get('source_onnx_sha256') or ''),
            'validation_dataset_sha256': str(row.get('validation_dataset_sha256') or row.get('dataset_sha256') or ''),
            'validation_dataset_image_ids_sha256': str(
                row.get('validation_dataset_image_ids_sha256')
                or row.get('validation_image_ids_sha256')
                or row.get('dataset_image_ids_sha256')
                or row.get('image_ids_sha256') or ''
            ),
            'validation_dataset_ground_truth_sha256': str(
                row.get('validation_dataset_ground_truth_sha256')
                or row.get('validation_ground_truth_sha256')
                or row.get('dataset_ground_truth_sha256')
                or row.get('ground_truth_sha256') or ''
            ),
            'source_request_sha256': str(row.get('source_request_sha256') or ''),
            'task_quality_policy_sha256': str(row.get('task_quality_policy_sha256') or ''),
            'runtime_quality_gate_policy_sha256': str(row.get('runtime_quality_gate_policy_sha256') or ''),
            'output_manifest_sha256': _sha256_file(dump),
            'source_e2e_scope': str(row.get('source_e2e_scope') or row.get('e2e_scope') or ''),
            'e2e_scope': str(row.get('e2e_scope') or ''),
            'e2e_claim_eligible': (
                row.get('e2e_claim_eligible')
                if row.get('e2e_claim_eligible') is not None
                else row.get('claim_eligible_e2e')
            ),
            'e2e_contract_reason': str(
                row.get('e2e_contract_reason') or ''
            ),
            'comparison_endpoint_stratum': str(
                row.get('comparison_endpoint_stratum') or ''
            ),
            'measurement_concurrency': row.get(
                'measurement_concurrency'
            ),
            'requires_host_decode_nms': row.get(
                'requires_host_decode_nms'
            ),
            'postprocess_included': row.get('postprocess_included'),
            'postprocess_location': str(
                row.get('postprocess_location') or ''
            ),
            'host_postprocess_frozen': row.get(
                'host_postprocess_frozen'
            ),
            'postprocess_completed_frames': row.get(
                'postprocess_completed_frames'
            ),
            'postprocess_completion_verified': row.get(
                'postprocess_completion_verified'
            ),
            'completed_work_units': row.get(
                'completed_work_units'
            ),
            'completed_frames': row.get('completed_frames'),
            'completion_observation_relation': row.get(
                'completion_observation_relation'
            ),
            'completion_exact_result_claim_bound': row.get(
                'completion_exact_result_claim_bound'
            ),
            'completion_execution_contract': (
                dict(row.get('completion_execution_contract'))
                if isinstance(
                    row.get('completion_execution_contract'), Mapping,
                )
                else None
            ),
            'frozen_host_postprocess_contract': (
                dict(row.get('frozen_host_postprocess_contract'))
                if isinstance(
                    row.get('frozen_host_postprocess_contract'), Mapping,
                )
                else None
            ),
            'frozen_host_postprocess_contract_sha256': str(
                row.get('frozen_host_postprocess_contract_sha256')
                or row.get('frozen_postprocess_contract_sha256')
                or ''
            ),
            'frozen_host_postprocess_result': (
                dict(row.get('frozen_host_postprocess_result'))
                if isinstance(
                    row.get('frozen_host_postprocess_result'), Mapping,
                )
                else None
            ),
        })
        rec['native_row_ok'] = bool(
            _explicit_bool(row, ('ok', 'run_ok', 'runtime_ok')) is True
        )
        native_json=_load_json(report) if report else {}
        native_json = native_json if isinstance(native_json, dict) else {}
        buildable = _explicit_bool(
            row, ('buildable', 'build_ok', 'compile_ok', 'producer_ready'),
        )
        if buildable is None:
            buildable = _explicit_bool(
                native_json,
                ('buildable', 'build_ok', 'compile_ok', 'producer_ready'),
            )
        runtime_executable = _explicit_bool(
            row,
            ('runtime_executable', 'runtime_ok', 'run_ok', 'result_ok', 'ok'),
        )
        if runtime_executable is None:
            runtime_executable = _explicit_bool(
                native_json,
                (
                    'runtime_executable', 'runtime_ok', 'run_ok',
                    'result_ok', 'ok',
                ),
            )
        if buildable is None:
            buildable = rec['native_row_ok']
        if runtime_executable is None:
            runtime_executable = rec['native_row_ok']
        rec.update({
            'buildable': bool(buildable),
            'runtime_executable': bool(runtime_executable),
            'execution_validation_status': (
                'passed'
                if buildable and runtime_executable
                else 'failed'
            ),
        })
        _merge_native_split_quality_evidence(rec, row, native_json)
        _merge_vendor_full_quality_evidence(rec, row, native_json)
        if apply_native_split_quality_authority is not None:
            apply_native_split_quality_authority(
                rec, split_quality_authority,
            )
        for field in (
            'native_command_contract_sha256', 'full_command_contract_sha256',
            'input_image_sha256', 'model_sha256', 'validation_dataset_sha256',
            'validation_dataset_image_ids_sha256',
            'validation_dataset_ground_truth_sha256',
            'source_request_sha256', 'task_quality_policy_sha256',
            'runtime_quality_gate_policy_sha256',
        ):
            if not rec.get(field) and native_json.get(field) not in (None, ''):
                rec[field] = str(native_json.get(field) or '')
        try:
            rec.update(_validate_hailo_trt_interface_contract(
                row, native_json, report=report, roots=roots,
            ))
        except Exception as interface_exc:
            # A verifier bug or malformed nested value may never turn a Hailo
            # boundary into an admitted interface.  Preserve the diagnostic,
            # but keep the row fail-closed.
            if str(backend or '').strip().lower() in _HAILO_TRT_SPLIT_BACKENDS:
                rec.update(_hailo_trt_interface_result(
                    False,
                    'native_hailo_trt_interface_verifier_error',
                    evidence={
                        'error': f'{type(interface_exc).__name__}: {interface_exc}',
                    },
                ))
        dump_payload = _load_json(dump) if dump else {}
        dump_payload = dump_payload if isinstance(dump_payload, dict) else {}
        performance_input_mode, performance_input_mode_status = (
            _project_performance_input_contract_mode(
                row, native_json, dump_payload,
            )
        )
        if (
            backend.strip().lower() == 'native_full_deepx'
            or performance_input_mode != 'missing'
        ):
            rec.update({
                'performance_input_contract_mode': performance_input_mode,
                'performance_input_contract_mode_projection_status': (
                    performance_input_mode_status
                ),
            })
        if backend.strip().lower() == 'native_full_tensorrt':
            # The semantic dump is produced by the same exact engine as the
            # timing report.  Its signed-producer digest is a required link,
            # not an optional diagnostic duplicate.
            dump_producer_sha = dump_payload.get(
                'quality_first_producer_identity_sha256'
            )
            report_producer_sha = native_json.get(
                'quality_first_producer_identity_sha256'
            )
            producer_sha, producer_sha_valid = _consistent_sha256(
                rec.get('quality_first_producer_identity_sha256'),
                dump_producer_sha,
                report_producer_sha,
                required=True,
            )
            rec['quality_first_semantic_dump_binding_valid'] = bool(
                producer_sha_valid and producer_sha
                and _normalize_sha256(dump_producer_sha) == producer_sha
            )
            rec['quality_first_producer_identity_sha256'] = (
                producer_sha
                if rec['quality_first_semantic_dump_binding_valid']
                else 'conflict'
            )
        rec.update({
            'stage': str(dump_payload.get('stage') or ''),
            'endpoint_contract_complete': dump_payload.get('endpoint_contract_complete') is True,
            'endpoint_contract_hash': _normalize_sha256(dump_payload.get('endpoint_contract_hash')),
            'output_endpoint_attestation': dump_payload.get('output_endpoint_attestation')
            if isinstance(dump_payload.get('output_endpoint_attestation'), dict) else {},
            'output_format': str(dump_payload.get('output_format') or ''),
            'contract_family': str(dump_payload.get('contract_family') or ''),
            'accelerator_output_stage': str(
                row.get('accelerator_output_stage')
                or dump_payload.get('stage')
                or ''
            ),
            'accelerator_output_contract_family': str(
                row.get('accelerator_output_contract_family')
                or dump_payload.get('contract_family')
                or ''
            ),
            'accelerator_endpoint_contract_hash': str(
                row.get('accelerator_endpoint_contract_hash')
                or dump_payload.get('endpoint_contract_hash')
                or ''
            ),
            'accelerator_output_endpoint_attestation': (
                row.get('accelerator_output_endpoint_attestation')
                if isinstance(
                    row.get('accelerator_output_endpoint_attestation'), dict
                )
                else dump_payload.get('output_endpoint_attestation')
                if isinstance(
                    dump_payload.get('output_endpoint_attestation'), dict
                )
                else {}
            ),
            'completed_task_stage': str(
                row.get('completed_task_stage') or ''
            ),
            'completed_task_contract_family': str(
                row.get('completed_task_contract_family') or ''
            ),
            'completed_task_endpoint_contract_hash': str(
                row.get('completed_task_endpoint_contract_hash') or ''
            ),
            'completed_task_output_endpoint_id': str(
                row.get('completed_task_output_endpoint_id') or ''
            ),
            'completed_task_comparison_endpoint_contract': (
                dict(row.get(
                    'completed_task_comparison_endpoint_contract'
                ))
                if isinstance(
                    row.get(
                        'completed_task_comparison_endpoint_contract'
                    ),
                    Mapping,
                )
                else None
            ),
            'completed_task_comparison_endpoint_contract_hash': str(
                row.get(
                    'completed_task_comparison_endpoint_contract_hash'
                ) or ''
            ),
            'completed_task_comparison_output_endpoint_id': str(
                row.get(
                    'completed_task_comparison_output_endpoint_id'
                ) or ''
            ),
            'completed_task_completion_mode': str(
                row.get('completed_task_completion_mode') or ''
            ),
            'completed_task_endpoint_attestation': (
                row.get('completed_task_endpoint_attestation')
                if isinstance(
                    row.get('completed_task_endpoint_attestation'), dict
                )
                else {}
            ),
            'completed_task_endpoint_attested': row.get(
                'completed_task_endpoint_attested'
            ),
            'completed_task_endpoint_attestation_status': str(
                row.get('completed_task_endpoint_attestation_status') or ''
            ),
            'completed_task_endpoint_contract': (
                dict(row.get('completed_task_endpoint_contract'))
                if isinstance(
                    row.get('completed_task_endpoint_contract'), Mapping,
                )
                else None
            ),
        })
        _copy_completed_v2_projection(rec, row)
        rec['report_ok']=bool((native_json or {}).get('ok')) if isinstance(native_json, dict) else None
        if not dump:
            rec['status']='missing_dump'
            rec['runtime_executable'] = False
            rec['execution_validation_status'] = 'failed'
            rec['semantic_validation_status'] = 'unavailable'
            rec['semantic_input_binding_status'] = 'unavailable'
            apply_accuracy_gate_to_row(rec, gate_policy)
            rows.append(rec); continue
        try:
            e2e_gate = _native_full_e2e_contract_gate(dump, rec, task)
            rec.update(e2e_gate)
            tensor_rep=_validate_tensor_dump(dump, row_dir)
            rec['tensor_validation']=str(row_dir/'tensor_dump_validation.json')
            rec['tensor_ok']=bool(tensor_rep.get('ok'))
            rec['output_count']=int(tensor_rep.get('output_count') or 0)
            if task=='classification':
                trep=_validate_classification(dump, row_dir, ns.topk, ref_report)
                # Prefer Full-ONNX self-reference when a native boundary/input dump is available.
                # This keeps ResNet-style classification rows thesis-claimable even when
                # the stored external reference report is missing or in a stale contract.
                srep=_full_onnx_self_reference_classification(
                    dump,
                    eval_root,
                    ns.topk,
                    native_report=report,
                    roots=roots,
                    endpoint_evidence=rec,
                )
                if bool((srep or {}).get('available')):
                    trep['self_reference_report']=srep
                    trep['semantic_reference_source']='full_onnx_self_reference'
                    trep['semantic_available']=True
                    trep['semantic_ok']=bool(srep.get('semantic_ok'))
                    trep['ok']=bool(srep.get('ok'))
                    trep['top1_match']=srep.get('top1_match')
                    trep['top5_overlap']=srep.get('top5_overlap')
                    trep['reference_topk']=srep.get('reference_topk')
                    trep['reference_report']='full_onnx_self_reference'
                    # Rewrite the task artifact so the summary and the md agree.
                    (row_dir/'classification_validation.json').write_text(json.dumps(trep, indent=2), encoding='utf-8')
                    md=['# Native classification semantic validation','',f'Manifest: `{dump}`','',f'Reference source: `full_onnx_self_reference`','',f'Self-reference diagnosis: `{srep.get("diagnosis")}`','',f'Semantic: `{trep.get("semantic_ok")}`  Top1 match: `{trep.get("top1_match")}`  Top5 overlap: `{trep.get("top5_overlap")}`','','## Native TopK','| rank | class index | score |','|---:|---:|---:|']
                    for i,t in enumerate(srep.get('native_topk') or [],1): md.append(f"| {i} | {t.get('index')} | {float(t.get('score',0)):.6g} |")
                    md += ['', '## Full-ONNX Self-Reference TopK', '| rank | class index | score |','|---:|---:|---:|']
                    for i,t in enumerate(srep.get('reference_topk') or [],1): md.append(f"| {i} | {t.get('index')} | {float(t.get('score',0)):.6g} |")
                    (row_dir/'classification_validation.md').write_text('\n'.join(md)+'\n', encoding='utf-8')
                else:
                    trep['self_reference_report']=srep
                rec['task_validation']=str(row_dir/'classification_validation.json')
                rec['visual_artifact']=str(row_dir/'classification_validation.md')
                rec['visual_ok']=bool((row_dir/'classification_validation.md').is_file())
                rec['task_ok']=bool(trep.get('ok'))
                rec['semantic_ok']=trep.get('semantic_ok')
                rec['semantic_available']=bool(trep.get('semantic_available'))
                rec['top1_match']=trep.get('top1_match')
                rec['top5_overlap']=trep.get('top5_overlap')
                rec['semantic_reference_source']=trep.get('semantic_reference_source') or ('external_validation_report' if rec['semantic_available'] else '')
                _srr = trep.get('self_reference_report') or {}
                rec['self_reference_available']=_srr.get('available')
                rec['self_reference_ok']=_srr.get('ok')
                rec['self_reference_diagnosis']=_srr.get('diagnosis')
                rec['self_reference_reason']=_srr.get('reason')
                rec['self_reference_input_manifest_kind'] = _srr.get(
                    'self_reference_input_manifest_kind'
                )
                rec['validation_level']='classification_full_self_reference' if rec.get('semantic_reference_source') == 'full_onnx_self_reference' else ('classification_topk_reference' if rec['semantic_available'] else 'classification_topk_visual')
            else:
                trep = _visualize_detection(
                    dump,
                    row_dir,
                    ns.max_detection_outputs,
                    ref_report,
                    eval_root,
                    native_report=report,
                    roots=roots,
                    policy=gate_policy,
                    endpoint_evidence=rec,
                )
                rec['task_validation']=str(row_dir/'detection_visual_validation.json')
                rec['visual_artifact']=str(row_dir/'detection_visual_validation.md')
                rec['visual_ok']=bool((row_dir/'detection_visual_validation.md').is_file())
                rec['task_ok']=bool(trep.get('ok'))
                rec['semantic_ok']=trep.get('semantic_ok')
                rec['semantic_available']=bool(trep.get('semantic_available'))
                rec['decode_mode']=trep.get('decode_mode')
                rec['ap50_proxy']=((trep.get('match') or {}).get('match_ratio'))
                for similarity_field in (
                    'numerical_similarity_pass',
                    'numerical_similarity_status',
                    'numerical_similarity_reason',
                    'numerical_similarity_scope',
                    'numerical_similarity_metric',
                    'numerical_similarity_value',
                    'numerical_similarity_threshold',
                    'numerical_similarity_mean_iou',
                    'numerical_similarity_mean_iou_threshold',
                    'numerical_similarity_policy_id',
                    'numerical_similarity_iou_threshold',
                    'numerical_similarity_confidence_threshold',
                    'numerical_similarity_denominator',
                    'numerical_similarity_class_aware',
                    'numerical_similarity_reference_count',
                    'numerical_similarity_matched_count',
                    'class_agnostic_match_ratio_diagnostic',
                ):
                    if similarity_field in trep:
                        rec[similarity_field] = trep.get(similarity_field)
                rec['self_reference_match'] = dict(trep.get('match') or {})
                rec['class_agnostic_match'] = dict(
                    trep.get('class_agnostic_match') or {}
                )
                rec['semantic_reference_source']=trep.get('semantic_reference_source')
                _srr = trep.get('self_reference_report') or {}
                rec['self_reference_available']=_srr.get('available')
                rec['self_reference_ok']=_srr.get('ok')
                rec['self_reference_diagnosis']=_srr.get('diagnosis')
                rec['self_reference_reason']=_srr.get('reason')
                rec['self_reference_input_manifest_kind'] = _srr.get(
                    'self_reference_input_manifest_kind'
                )
                _apply_completed_v2_semantic_binding(rec, _srr)
                rec['box_overlay']=trep.get('box_overlay')
                rec['validation_level']='detection_yolo_full_self_reference_proxy' if trep.get('semantic_reference_source') == 'full_onnx_self_reference' else ('detection_yolo_decode_nms_proxy' if rec['semantic_available'] else 'detection_scoremap_visual')
            rec['artifact_generated']=bool(rec.get('visual_ok'))
            rec['strict_tensor_ok']=bool(rec.get('tensor_ok'))
            rec['semantic_validation_status'] = (
                'passed'
                if rec.get('semantic_available') is True
                and rec.get('semantic_ok') is True
                else 'failed'
                if rec.get('semantic_available') is True
                and rec.get('semantic_ok') is False
                else 'unavailable'
            )
            rec['semantic_input_binding_status'] = (
                'passed'
                if rec.get('self_reference_input_manifest_kind') in {
                    'native_full_input_manifest',
                    'split_boundary_manifest',
                }
                else 'not_applicable'
                if rec.get('semantic_reference_source')
                != 'full_onnx_self_reference'
                else 'failed'
            )
            # claim/ok must be finalized before v59ek gating, because the
            # generic gate helper uses these fields as build/runtime evidence.
            rec['claim_ok']=bool(
                rec.get('tensor_ok')
                and rec.get('semantic_ok') is True
                and rec.get('e2e_claim_eligible', True)
            )
            rec['ok']=bool(rec.get('claim_ok'))
            if rec['claim_ok']:
                rec['status']='claim_ok'
            elif rec.get('e2e_claim_eligible') is False:
                rec['status']=str(rec.get('e2e_scope') or 'accelerator_only') + '_' + str(rec.get('e2e_contract_reason') or 'not_e2e_claimable')
            elif rec.get('artifact_generated'):
                if rec.get('semantic_ok') == 'unavailable': rec['status']='artifact_generated_semantic_unavailable'
                elif rec.get('semantic_ok') is False: rec['status']='semantic_fail'
                elif not rec.get('tensor_ok'): rec['status']='artifact_generated_tensor_warning'
                else: rec['status']='artifact_generated_claim_warning'
            else:
                rec['status']='failed'
            # v59ek: split thesis eligibility gates after status/claim/ok are
            # finalized. Native self-reference is a contract gate by default;
            # --allow-contract-only-ranking explicitly upgrades it for exploration.
            apply_accuracy_gate_to_row(rec, gate_policy)
        except Exception as e:
            rec.update({'status':'error','error':f'{type(e).__name__}: {e}','visual_ok':False,'artifact_generated':False,'tensor_ok':False,'strict_tensor_ok':False,'claim_ok':False,'ok':False,'semantic_validation_status':'unavailable','semantic_input_binding_status':'failed'})
            apply_accuracy_gate_to_row(rec, gate_policy)
        _bind_central_quality_evidence(
            rec, central_results, gate_policy,
            split_quality_authority=split_quality_authority,
        )
        apply_accuracy_gate_to_row(rec, gate_policy)
        _apply_completed_v2_semantic_binding(rec, rec)
        _enforce_historical_split_diagnostic_only(rec)
        rows.append(rec)
    for rec in rows:
        _finalize_structural_claim_contract(rec)
    # Apply the Smoke policy in one common terminal pass so early rows such as
    # `native_row_not_ok` and `missing_dump` cannot escape the no-claim clamp.
    if diagnostic_only:
        _apply_smoke_diagnostic_policy(rows)
    applied_policy_sha256 = gate_policy.sha256()
    for rec in rows:
        rec['accuracy_gate_policy_sha256'] = applied_policy_sha256
        rec['accuracy_gate_policy_match'] = True
        binding_verified = bool(
            rec.get('central_quality_evidence_verified') is True
            and rec.get('precision_quality_binding_verified') is True
        )
        observation_valid = bool(
            binding_verified
            and rec.get('task_quality_observation_valid') is True
            and isinstance(rec.get('accuracy_gate_pass'), bool)
        )
        rec['precision_quality_verified'] = binding_verified
        rec['precision_quality_binding_verified'] = binding_verified
        rec['task_quality_observation_valid'] = observation_valid
        rec['quality_claim_result_verified'] = bool(
            observation_valid
            and rec.get('claim_ok') is True
            and rec.get('accuracy_gate_pass') is True
            and rec.get('eligible_for_ranking') is True
        )
    jsonp=out/'native_producer_validation_summary.json'; csvp=out/'native_producer_validation_summary.csv'; mdp=out/'native_producer_validation_summary.md'
    empty_output_error = not rows
    for rec in rows:
        rec['technical_quality_error'] = _technical_quality_error(rec)
        if rec['technical_quality_error']:
            rec.update(technical_invalid=True, eligible_for_ranking=False,
                       performance_claim_eligible=False, energy_claim_eligible=False,
                       scientific_claim_eligible=False, claim_ok=False, ok=False)
    row_technical_error_count = sum(1 for r in rows if r['technical_quality_error'])
    measured_technical_error_count = sum(
        1 for r in rows if r['technical_quality_error']
        and r.get('status') not in {'native_row_not_ok', 'excluded_known_build'}
    )
    payload={'schema':'onnx-splitpoint/native-producer-validation-summary','schema_version':10,'status':'complete','summary':str(summary_path),'central_quality_summary':str(ns.central_quality_summary or ''),'native_split_quality_authority':split_quality_authority,'diagnostic_only':diagnostic_only,'claim_eligible':False if diagnostic_only else None,'metric_threshold_policy':'warning_only' if diagnostic_only else 'enforced','technical_chain_complete':not empty_output_error and row_technical_error_count == 0,'empty_output_error':empty_output_error,'technical_error_count':row_technical_error_count + (1 if empty_output_error else 0),'measured_technical_error_count':measured_technical_error_count,'rows':rows,'ok_count':sum(1 for r in rows if r.get('ok')),'claim_ok_count':sum(1 for r in rows if r.get('claim_ok')),'technical_semantic_claim_ok_count':sum(1 for r in rows if r.get('claim_ok') is True),'scientific_claim_eligible_count':sum(1 for r in rows if r.get('quality_claim_result_verified') is True),'scientific_claim_eligible_semantics':'quality_claim_result_verified','claim_decision_count':sum(1 for r in rows if isinstance(r.get('claim_ok'), bool)),'strict_ok_count':sum(1 for r in rows if r.get('strict_tensor_ok')),'artifact_generated_count':sum(1 for r in rows if r.get('artifact_generated') or r.get('visual_ok')),'visual_ok_count':sum(1 for r in rows if r.get('visual_ok')),'tensor_ok_count':sum(1 for r in rows if r.get('tensor_ok')),'semantic_available_count':sum(1 for r in rows if r.get('semantic_available')),'semantic_decision_count':sum(1 for r in rows if r.get('semantic_available') is True and isinstance(r.get('semantic_ok'), bool)),'semantic_unavailable_count':sum(1 for r in rows if r.get('semantic_available') is not True),'semantic_ok_count':sum(1 for r in rows if r.get('semantic_ok') is True),'semantic_fail_count':sum(1 for r in rows if r.get('semantic_available') is True and r.get('semantic_ok') is False),'buildable_count':sum(1 for r in rows if r.get('buildable') is True),'runtime_executable_count':sum(1 for r in rows if r.get('runtime_executable') is True),'contract_consistent_count':sum(1 for r in rows if r.get('contract_consistent') is True),'structural_contract_pass_count':sum(1 for r in rows if r.get('structural_contract_pass') is True),'numerical_similarity_pass_count':sum(1 for r in rows if r.get('numerical_similarity_pass') is True),'numerical_similarity_fail_count':sum(1 for r in rows if r.get('numerical_similarity_pass') is False),'task_quality_pass_count':sum(1 for r in rows if r.get('task_quality_pass') is True),'task_valid_count':sum(1 for r in rows if r.get('task_valid') is True),'eligible_for_ranking_count':sum(1 for r in rows if r.get('eligible_for_ranking')),'accuracy_gate_pass_count':sum(1 for r in rows if r.get('accuracy_gate_pass') is True),'metric_threshold_warning_count':sum(1 for r in rows if r.get('metric_threshold_miss_warning') is True),'central_quality_evidence_verified_count':sum(1 for r in rows if r.get('central_quality_evidence_verified') is True),'precision_quality_verified_count':sum(1 for r in rows if r.get('precision_quality_verified') is True),'precision_quality_binding_verified_count':sum(1 for r in rows if r.get('precision_quality_binding_verified') is True),'task_quality_observation_valid_count':sum(1 for r in rows if r.get('task_quality_observation_valid') is True),'quality_claim_result_verified_count':sum(1 for r in rows if r.get('quality_claim_result_verified') is True),'accuracy_gate_policy':gate_policy.as_dict(),'accuracy_gate_policy_sha256':gate_policy.sha256(),'row_count':len(rows)}
    jsonp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
    fields=['accuracy_assessment','accuracy_class','accuracy_relative_loss','accuracy_absolute_loss_pp','accuracy_relative_loss_ci','accuracy_uncertainty','accuracy_uncertainty_reason','accuracy_gate_semantics','legacy_accuracy_gate','secondary_accuracy_assessments','accuracy_warnings','backend','model','case','precision','execution_precision','full_runtime_precision','runtime_precision_identity','setup_id','comparison_backend','task','stage','contract_family','endpoint_contract_complete','endpoint_contract_hash','accelerator_output_stage','accelerator_output_contract_family','accelerator_endpoint_contract_hash','completed_task_stage','completed_task_contract_family','completed_task_endpoint_contract_hash','completed_task_output_endpoint_id','completed_task_comparison_endpoint_contract_hash','completed_task_comparison_output_endpoint_id','completed_task_completion_mode','completed_work_units','completed_frames','completion_observation_relation','completion_exact_result_claim_bound','semantic_result_binding_status','exact_completed_result_identity_bound','portable_result_hash_mismatch','completed_v2_exact_result_claim_binding','completed_v2_semantic_evidence_tier','performance_hotloop_result_sha256','native_completed_result_sha256','completed_task_endpoint_attested','completed_task_endpoint_attestation_status','interface_contract_pass','interface_contract_status','interface_check_pass','interface_check_status','strict_boundary_numeric_pass','strict_boundary_numeric_status','structural_contract_pass','structural_contract_status','structural_contract_reason','numerical_similarity_pass','numerical_similarity_status','numerical_similarity_reason','numerical_similarity_scope','numerical_similarity_metric','numerical_similarity_value','numerical_similarity_threshold','numerical_similarity_mean_iou','numerical_similarity_mean_iou_threshold','numerical_similarity_policy_id','numerical_similarity_iou_threshold','numerical_similarity_confidence_threshold','numerical_similarity_reference_count','numerical_similarity_matched_count','task_quality_pass','task_quality_status','task_quality_reason','central_quality_evidence_verified','precision_quality_verified','precision_quality_binding_verified','task_quality_observation_valid','quality_claim_result_verified','central_quality_binding_status','central_quality_binding_candidate_count','quality_first_binding_status','quality_first_producer_identity_sha256','quality_request_binding_status','quality_request_binding_sha256','quality_request_binding_set_sha256','vendor_full_central_quality_result_sha256','native_split_quality_binding_sha256','quality_first_semantic_dump_binding_valid','quality_contract_sha256','preprocessing_contract_sha256','decoder_contract_sha256','nms_contract_sha256','output_manifest_sha256','native_command_contract_sha256','full_command_contract_sha256','input_image_sha256','model_sha256','validation_dataset_sha256','validation_dataset_image_ids_sha256','validation_dataset_ground_truth_sha256','source_request_sha256','task_quality_policy_sha256','runtime_quality_gate_policy_sha256','accuracy_gate_policy_sha256','accuracy_gate_policy_match','ok','claim_ok','claim_ok_source','claim_structural_gate_pass','claim_structural_gate_reason','claim_ok_structural_clamped','source_e2e_scope','e2e_claim_eligible','e2e_scope','e2e_contract_reason','comparison_endpoint_stratum','measurement_concurrency','requires_host_decode_nms','postprocess_location','host_postprocess_frozen','host_postprocessing_available','host_tail_available','host_postprocess_required','host_tail_required','host_postprocessing_evidence_status','host_postprocessing_evidence_source','host_postprocessing_legacy_alias_conflict','postprocess_included','postprocess_completed_frames','postprocess_completion_verified','frozen_host_postprocess_contract_sha256','performance_input_contract_mode','performance_input_contract_mode_projection_status','buildable','runtime_executable','execution_validation_status','contract_consistent','task_valid','accuracy_gate_pass','eligible_for_ranking','gate_status','ranking_exclusion_reason','accuracy_gate_reason','accuracy_gate_metric','accuracy_gate_delta','accuracy_gate_threshold','artifact_generated','visual_ok','tensor_ok','strict_tensor_ok','semantic_ok','semantic_available','semantic_validation_status','semantic_input_binding_status','self_reference_input_manifest_kind','validation_level','status','dump_status','ap50_proxy','semantic_reference_source','self_reference_available','self_reference_ok','self_reference_diagnosis','self_reference_reason','top1_match','top5_overlap','decode_mode','dump_manifest','reference_report','tensor_validation','task_validation','visual_artifact','box_overlay','native_report']
    with csvp.open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in rows: w.writerow({k:r.get(k,'') for k in fields})
    md=['# Native producer semantic validation / visual summary','', '`ok` / `claim` means the task/contract validation artifact passed. Structure, numerical similarity and dataset task quality are independent evidence axes. `contract` is the structural axis only; numerical mismatch never rewrites it. Ranking eligibility requires every axis that the versioned policy declares mandatory.','','| backend | model | case | precision | task | ok | claim | eligible | structure | numerical | task quality | accuracy_gate | artifact | tensor | semantic | level | status | ap50/topk | artifact file |','|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|']
    for r in rows:
        visual=Path(str(r.get('visual_artifact') or '')).name if r.get('visual_artifact') else ''
        metric=''
        if r.get('task') == 'detection': metric=f"ap50p={r.get('ap50_proxy','')} mode={r.get('decode_mode','')}"
        elif r.get('task') == 'classification': metric=f"top1={r.get('top1_match','')} top5ov={r.get('top5_overlap','')}"
        md.append(f"| {r.get('backend','')} | {r.get('model','')} | {r.get('case','')} | {r.get('precision','')} | {r.get('task','')} | {r.get('ok',False)} | {r.get('claim_ok','')} | {r.get('eligible_for_ranking','')} | {r.get('structural_contract_pass','')} | {r.get('numerical_similarity_pass','')} | {r.get('task_quality_pass','')} | {r.get('accuracy_gate_pass','')} | {r.get('artifact_generated', r.get('visual_ok',''))} | {r.get('tensor_ok','')} | {r.get('semantic_ok','')} | {r.get('validation_level','')} | {r.get('status','')} | {metric} | {visual} |")
    mdp.write_text('\n'.join(md)+'\n', encoding='utf-8')
    print(json.dumps({'ok':True,'rows':len(rows),'ok_count':payload['ok_count'],'claim_ok_count':payload['claim_ok_count'],'semantic_ok_count':payload['semantic_ok_count'],'eligible_for_ranking_count':payload.get('eligible_for_ranking_count',0),'json':str(jsonp),'csv':str(csvp),'md':str(mdp)}, indent=2))
    return 0

if __name__=='__main__':
    raise SystemExit(main())
