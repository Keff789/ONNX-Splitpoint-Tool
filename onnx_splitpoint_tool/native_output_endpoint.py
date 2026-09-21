"""Fail-closed endpoint contracts for Native accelerator outputs.

Runtime values can establish that a tensor *looks like* decoded detections, but
they cannot establish that non-maximum suppression was executed.  In
particular, a decoded pre-NMS tensor can have the same ``[B, N, 6]`` layout as
an integrated-NMS tensor. The historical ``decoded_nms`` stage token also
supports graph-bound TopK candidates; their explicit selection evidence records
that no NMS was performed. Shape alone never authorizes this contract.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


DECODED_NMS_ATTESTATION_SOURCE = (
    "explicit_endpoint_declaration_plus_runtime_values:v3"
)
DECODED_NMS_ATTESTATION_SCHEMA = (
    "onnx-splitpoint/runtime-output-endpoint-attestation"
)

_FRACTION_THRESHOLD = 0.995
_RANGE_EPSILON = 1.0e-4
_INTEGER_EPSILON = 1.0e-3
_NMS_NAME_TOKENS = (
    "nms", "non_max", "nonmax", "batchednms", "efficientnms",
)
_TOPK_NAME_TOKENS = (
    "topk", "top_k", "indices", "labels", "class_ids", "classid",
)


def inspect_bn6_candidate_graph(path: str | Path) -> dict[str, Any]:
    """Recognize a fixed TopK XYXY/score/class export by its dataflow.

    This is source inspection, never inference. Unrecognized graphs retain the
    strict final-record contract. Names, BN6 shape and exporter metadata alone
    cannot authorize candidate filtering.
    """
    import onnx
    from onnx import helper, numpy_helper

    path = Path(path)
    model = onnx.load(str(path), load_external_data=False)
    graph = model.graph
    if len(graph.output) != 1:
        return {}
    shape = [int(d.dim_value) for d in graph.output[0].type.tensor_type.shape.dim]
    if len(shape) != 3 or shape[0] != 1 or shape[1] <= 0 or shape[2] != 6:
        return {}
    producers = {v: n for n in graph.node for v in n.output}
    constants = {v.name: numpy_helper.to_array(v) for v in graph.initializer}
    for node in graph.node:
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value":
                    constants[node.output[0]] = numpy_helper.to_array(attr.t)
    def node(value, op):
        n = producers[value]
        if n.op_type != op:
            raise ValueError("different graph operation")
        return n
    def attr(n, key, default=None):
        return next((helper.get_attribute_value(a) for a in n.attribute if a.name == key), default)
    def scalar(value):
        a = constants[value]
        if a.size != 1:
            raise ValueError("non-scalar graph parameter")
        return int(a.reshape(-1)[0])
    try:
        final = node(graph.output[0].name, "Concat")
        if len(final.input) != 3 or attr(final, "axis") not in (-1, 2):
            return {}
        boxes = node(final.input[0], "GatherElements")
        scores = node(final.input[1], "Unsqueeze")
        cast = node(final.input[2], "Cast")
        classes = node(cast.input[0], "Unsqueeze")
        top = node(scores.input[0], "TopK")
        mod = node(classes.input[0], "Mod")
        if (mod.input[0] != top.output[1] or scores.input[0] != top.output[0]
                or attr(mod, "fmod", 0) != 0 or attr(cast, "to") != onnx.TensorProto.FLOAT
                or scalar(scores.input[1]) not in (-1, 2) or scalar(classes.input[1]) not in (-1, 2)):
            return {}
        nc, k = scalar(mod.input[1]), scalar(top.input[1])
        if nc <= 0 or k != shape[1] or attr(top, "axis", -1) not in (-1, 1) or attr(top, "largest", 1) != 1:
            return {}
        flat_scores = node(top.input[0], "Flatten")
        selected_scores = node(flat_scores.input[0], "GatherElements")
        split = node(boxes.input[0], "Split")
        if (list(constants[split.input[1]]) != [4, nc] or selected_scores.input[0] != split.output[1]
                or boxes.input[0] != split.output[0] or attr(split, "axis") not in (-1, 2)
                or attr(flat_scores, "axis", 1) != 1):
            return {}
        score_tile = node(selected_scores.input[1], "Tile")
        first_indices = node(score_tile.input[0], "Unsqueeze")
        first = node(first_indices.input[0], "TopK")
        maximum = node(first.input[0], "ReduceMax")
        if (maximum.input[0] != split.output[1] or first_indices.input[0] != first.output[1]
                or scalar(first.input[1]) != k or attr(first, "largest", 1) != 1
                or scalar(maximum.input[1]) not in (-1, 2) or attr(maximum, "keepdims", 1) != 0
                or scalar(first_indices.input[1]) not in (-1, 2) or attr(first, "axis", -1) not in (-1, 1)
                or list(constants[score_tile.input[1]]) != [1, 1, nc]):
            return {}
        box_tile = node(boxes.input[1], "Tile")
        gather = node(box_tile.input[0], "Gather")
        flat_indices = node(gather.input[0], "Flatten")
        quotient = node(gather.input[1], "Div")
        if (flat_indices.input[0] != first_indices.output[0] or attr(flat_indices, "axis", 1) != 2
                or attr(gather, "axis", 0) != 0 or list(constants[box_tile.input[1]]) != [1, 1, 4]
                or quotient.input[0] != top.output[1] or scalar(quotient.input[1]) != nc
                or attr(boxes, "axis") != 1 or attr(selected_scores, "axis") != 1):
            return {}
        # Establish XYXY, not XYWH: anchor - distance and anchor + distance,
        # concatenated before positive stride multiplication. Scores are sigmoid.
        transpose = node(split.input[0], "Transpose")
        decoded = node(transpose.input[0], "Concat")
        scaled = node(decoded.input[0], "Mul")
        corners = node(scaled.input[0], "Concat")
        lower = node(corners.input[0], "Sub")
        upper = node(corners.input[1], "Add")
        node(decoded.input[1], "Sigmoid")
        if (attr(transpose, "perm") != [0, 2, 1]
                or len(decoded.input) != 2 or len(corners.input) != 2
                or attr(decoded, "axis") != 1 or attr(corners, "axis") != 1
                or not np.all(np.isfinite(constants[scaled.input[1]]))
                or not np.all(constants[scaled.input[1]] > 0)):
            return {}
        if lower.input[0] != upper.input[0] and not np.array_equal(constants[lower.input[0]], constants[upper.input[0]]):
            return {}
        # Only this connected tail is authorized. There is no score threshold,
        # NMS or geometry repair between decoded boxes and the graph output.
        tail = []
        pending = [graph.output[0].name]
        seen = set()
        while pending:
            value = pending.pop()
            if value in seen or value in constants or value == split.input[0]:
                continue
            seen.add(value)
            n = producers[value]
            if n.op_type not in {"Concat", "GatherElements", "Gather", "Tile", "Split", "ReduceMax", "TopK", "Flatten", "Unsqueeze", "Div", "Mod", "Cast"}:
                return {}
            if n.name not in tail:
                tail.append(n.name)
            pending.extend(n.input)
    except (KeyError, IndexError, TypeError, ValueError):
        return {}
    return {
        "selection_semantics": "fixed_topk_xyxy_score_class_candidates",
        "source_onnx_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "output_name": graph.output[0].name, "output_shape": shape,
        "candidate_count": k, "class_count": nc,
        "confidence_selection": "consumer_threshold", "host_nms_required": False,
        "graph_tail_nodes": sorted(tail),
        "exporter": model.producer_name, "exporter_version": model.producer_version,
    }


def bn6_candidate_selection(declaration: Mapping[str, Any]) -> dict[str, Any]:
    """Return the graph evidence inside an already bound source declaration."""
    endpoint = declaration.get("source_onnx_detection_endpoint") or {}
    proof = endpoint.get("candidate_selection") or {}
    if not proof:
        return {}
    shape = proof.get("output_shape") or []
    if (proof.get("selection_semantics") != "fixed_topk_xyxy_score_class_candidates"
            or not _is_sha256(proof.get("source_onnx_sha256"))
            or shape != [1, proof.get("candidate_count"), 6]
            or type(proof.get("candidate_count")) is not int or proof["candidate_count"] <= 0
            or type(proof.get("class_count")) is not int or proof["class_count"] <= 0
            or proof.get("confidence_selection") != "consumer_threshold"
            or proof.get("host_nms_required") is not False or not proof.get("graph_tail_nodes")):
        raise ValueError("bn6_candidate_graph_contract_invalid")
    return dict(proof)


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload), ensure_ascii=True, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_contract_backend(value: Any) -> str:
    """Return the backend token used by suite ``output_contracts.json``.

    Native producer names describe the runner (for example
    ``native_full_tensorrt``), while the suite contract describes the logical
    backend that produced the model artifact (``cuda_ort``).  Keep this alias
    table deliberately small and exact so a similarly named backend cannot be
    selected accidentally.
    """
    token = str(value or "").strip().lower().replace("-", "_")
    if token.startswith("native_full_"):
        token = token[len("native_full_"):]
    aliases = {
        "cpu": "cpu_ort",
        "ort_cpu": "cpu_ort",
        "cuda": "cuda_ort",
        "ort_cuda": "cuda_ort",
        "tensorrt": "cuda_ort",
        "trt": "cuda_ort",
        "ort_tensorrt": "cuda_ort",
        "hailo10h": "hailo10",
        "hailo8l": "hailo8",
        "hailo8r": "hailo8",
        "deepx": "deepx_m1",
        "dx_m1": "deepx_m1",
        "dxm1": "deepx_m1",
    }
    return aliases.get(token, token)


def _canonical_declared_stage(value: Any, *, task: str) -> str:
    token = str(value or "").strip().lower().replace("-", "_")
    task_l = str(task or "").strip().lower()
    if token == "decoded":
        if task_l == "classification":
            return "classification_logits"
        if task_l == "detection":
            return "decoded_nms"
        return ""
    aliases = {
        "logits": "classification_logits",
        "classification_scores": "classification_logits",
        "probabilities": "classification_probabilities",
        "softmax": "classification_probabilities",
        "raw_detection_head": "raw_head",
    }
    return aliases.get(token, token)


def _is_sha256(value: Any) -> bool:
    token = str(value or "").strip().lower()
    if token.startswith("sha256:"):
        token = token.split(":", 1)[1]
    return len(token) == 64 and all(char in "0123456789abcdef" for char in token)


def load_authoritative_output_contract(
    suite_root: str | Path,
    *,
    backend: str,
    model_id: str,
    variant: str = "full",
    task: str = "",
) -> dict[str, Any]:
    """Load one fail-closed suite-root output declaration.

    Only the exact ``output_contracts.json`` at ``suite_root`` is considered.
    A missing, malformed, ambiguous, or contradictory contract returns a
    metadata-only mapping without a ``stage``; passing that mapping to
    :func:`runtime_output_contract` therefore cannot attest an endpoint.

    A historical ``endpoint_mode=decoded`` declaration is promoted to
    ``decoded_nms`` only when the workflow recorded it explicitly and also
    recorded that neither host-tail nor other postprocessing is required.
    Runtime tensor shape and values are still checked independently by
    :func:`attest_decoded_nms`.
    """
    root = Path(suite_root).expanduser()
    path = root if root.name == "output_contracts.json" else root / "output_contracts.json"
    base = {
        "contract_resolution_status": "unavailable",
        "contract_resolution_reason": "suite_output_contracts_missing",
        "declaration_source": str(path),
    }
    if not path.is_file():
        return base
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            **base,
            "contract_resolution_reason": "suite_output_contracts_unreadable",
            "contract_resolution_error": f"{type(exc).__name__}: {exc}",
        }
    if not isinstance(payload, Mapping) or not isinstance(payload.get("contracts"), list):
        return {**base, "contract_resolution_reason": "suite_output_contracts_invalid"}

    requested_backend = _canonical_contract_backend(backend)
    requested_model = str(model_id or "").strip()
    requested_variant = str(variant or "full").strip().lower()
    requested_task = str(task or "").strip().lower()
    if requested_task not in {"classification", "detection"}:
        return {
            **base,
            "contract_resolution_reason": "suite_output_contract_task_missing_or_unsupported",
            "requested_task": requested_task,
        }
    payload_task = str(
        payload.get("task") or payload.get("benchmark_task") or ""
    ).strip().lower()
    if payload_task and payload_task != requested_task:
        return {
            **base,
            "contract_resolution_status": "conflict",
            "contract_resolution_reason": "suite_output_contract_task_conflict",
            "requested_task": requested_task,
            "payload_task": payload_task,
        }
    payload_model = str(payload.get("model_id") or "").strip()
    if not requested_model or not payload_model or payload_model != requested_model:
        return {
            **base,
            "contract_resolution_reason": "suite_output_contract_model_mismatch",
            "requested_model_id": requested_model,
            "payload_model_id": payload_model,
        }
    matches: list[dict[str, Any]] = []
    invalid_matches: list[dict[str, Any]] = []
    for raw in payload.get("contracts") or []:
        if not isinstance(raw, Mapping):
            continue
        declared_backend = _canonical_contract_backend(raw.get("backend"))
        declared_variant = str(raw.get("variant") or "full").strip().lower()
        declared_model = str(raw.get("model_id") or "").strip()
        if declared_backend != requested_backend or declared_variant != requested_variant:
            continue
        if not declared_model or declared_model != requested_model:
            continue

        contract = dict(raw)
        row_errors: list[str] = []
        status = str(contract.get("contract_status") or "").strip().lower()
        if status != "recorded":
            row_errors.append("endpoint_contract_not_recorded")

        task_values = {
            str(contract.get(key) or "").strip().lower()
            for key in ("task", "benchmark_task", "model_task")
            if str(contract.get(key) or "").strip()
        }
        if not task_values:
            row_errors.append("endpoint_contract_task_missing")
        elif task_values != {requested_task}:
            row_errors.append("endpoint_contract_task_conflict")

        declared_stages = {
            _canonical_declared_stage(contract.get(key), task=requested_task)
            for key in ("stage", "contract_family", "endpoint_mode")
            if str(contract.get(key) or "").strip()
        }
        if not declared_stages or "" in declared_stages:
            row_errors.append("endpoint_stage_missing_or_unsupported")
        if len(declared_stages) > 1:
            row_errors.append("endpoint_stage_fields_conflict")
        normalized_stage = next(iter(declared_stages), "") if len(declared_stages) == 1 else ""

        host_tail = contract.get("host_tail_required")
        postprocessing = contract.get("postprocessing_required")
        expected_flags = {
            "decoded_nms": (False, False),
            "classification_logits": (False, False),
            "classification_probabilities": (False, False),
            "raw_head": (True, True),
            "decoded_pre_nms": (True, True),
        }.get(normalized_stage)
        if expected_flags is None:
            row_errors.append("endpoint_stage_incompatible_with_task")
        elif (host_tail, postprocessing) != expected_flags:
            row_errors.append("endpoint_postprocessing_flags_conflict")

        if requested_task == "classification" and normalized_stage not in {
            "classification_logits", "classification_probabilities",
        }:
            row_errors.append("endpoint_stage_incompatible_with_task")
        if requested_task == "detection" and normalized_stage not in {
            "decoded_nms", "decoded_pre_nms", "raw_head",
        }:
            row_errors.append("endpoint_stage_incompatible_with_task")

        if requested_task == "detection":
            output_format = str(
                contract.get("output_format") or ""
            ).strip().lower()
            decoded_metadata_fields = (
                "output_record_format",
                "coordinate_format",
                "coordinate_space",
                "source_coordinate_space",
                "score_semantics",
                "class_id_semantics",
            )
            if normalized_stage == "raw_head":
                if output_format and output_format != "raw_detection_tensors":
                    row_errors.append("raw_endpoint_output_format_conflict")
                if any(
                    contract.get(field) not in (None, "")
                    for field in decoded_metadata_fields
                ):
                    row_errors.append(
                        "raw_endpoint_contains_decoded_detection_metadata"
                    )
            elif (
                normalized_stage == "decoded_nms"
                and output_format
                and output_format != "bn6_detections"
            ):
                row_errors.append("decoded_nms_output_format_conflict")
            elif (
                normalized_stage == "decoded_pre_nms"
                and output_format
                and output_format != "ultralytics_decoded"
            ):
                row_errors.append("decoded_pre_nms_output_format_conflict")

        external = contract.get("requires_external_postprocess")
        if external is not None and expected_flags is not None and external is not expected_flags[0]:
            row_errors.append("endpoint_external_postprocess_flag_conflict")
        nested_post = contract.get("postprocessing")
        if isinstance(nested_post, Mapping) and expected_flags is not None:
            for key in ("host_required", "nms_on_host"):
                value = nested_post.get(key)
                if value is not None and value is not expected_flags[0]:
                    row_errors.append("endpoint_nested_postprocess_flag_conflict")

        # A Hailo endpoint may only become ``recorded`` after a successful
        # build or verified reuse has been bound to the exact HEF bytes.
        # Recheck every Hailo generation here so Hailo10 raw-head
        # reconciliation cannot survive a later artifact replacement.
        verify_hailo_artifact = bool(
            requested_backend.startswith("hailo8")
            or (
                requested_backend.startswith("hailo")
                and (
                    str(contract.get("artifact_binding_status") or "").lower()
                    == "verified"
                    or bool(contract.get("contract_reconciliation_status"))
                )
            )
        )
        if verify_hailo_artifact and status == "recorded":
            hailo_error_prefix = requested_backend
            expected_artifact_sha = str(
                contract.get("recorded_artifact_sha256")
                or contract.get("artifact_sha256")
                or ""
            ).strip().lower().removeprefix("sha256:")
            artifact_text = str(
                contract.get("recorded_artifact_path")
                or contract.get("artifact_path")
                or ""
            ).strip()
            artifact_path = Path(artifact_text).expanduser() if artifact_text else None
            if artifact_path is not None and not artifact_path.is_absolute():
                artifact_path = (path.parent / artifact_path).resolve()
            if not _is_sha256(expected_artifact_sha):
                row_errors.append(
                    f"{hailo_error_prefix}_recorded_artifact_sha256_missing_or_invalid"
                )
            elif artifact_path is None or not artifact_path.is_file():
                row_errors.append(
                    f"{hailo_error_prefix}_recorded_artifact_missing"
                )
            else:
                actual_sha = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
                if actual_sha != expected_artifact_sha:
                    row_errors.append(
                        f"{hailo_error_prefix}_recorded_artifact_sha256_mismatch"
                    )
                try:
                    expected_size = int(
                        contract.get("recorded_artifact_size_bytes")
                        or contract.get("artifact_size_bytes")
                        or 0
                    )
                except (TypeError, ValueError):
                    expected_size = 0
                if expected_size <= 0 or expected_size != artifact_path.stat().st_size:
                    row_errors.append(
                        f"{hailo_error_prefix}_recorded_artifact_size_mismatch"
                    )

        if row_errors:
            invalid_matches.append({
                "contract_sha256": _canonical_sha256(dict(raw)),
                "errors": sorted(set(row_errors)),
            })
            continue

        contract.update({
            "stage": normalized_stage,
            "contract_family": normalized_stage,
            "endpoint_mode": normalized_stage,
            "backend": requested_backend,
            "model_id": requested_model,
            "variant": requested_variant,
            "task": requested_task,
            "declaration_source": str(path.resolve()),
            "source_contracts_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "source_contract_sha256": _canonical_sha256(dict(raw)),
            "authoritative_output_contract": True,
            "contract_resolution_status": "attested",
            "contract_resolution_reason": "exact_suite_output_contract",
        })
        matches.append(contract)

    if invalid_matches:
        return {
            **base,
            "contract_resolution_status": "conflict",
            "contract_resolution_reason": (
                "suite_output_contract_exact_match_invalid"
                if len(invalid_matches) == 1 and not matches
                else "suite_output_contract_exact_matches_conflict"
            ),
            "contract_resolution_invalid_matches": invalid_matches,
            "contract_resolution_errors": sorted({
                error for item in invalid_matches for error in item.get("errors", [])
            }),
        }
    if not matches:
        return {**base, "contract_resolution_reason": "suite_output_contract_exact_match_missing"}
    semantic_signatures = {
        (
            str(item.get("stage") or ""),
            item.get("host_tail_required"),
            item.get("postprocessing_required"),
        )
        for item in matches
    }
    contract_hashes = {
        str(item.get("source_contract_sha256") or "") for item in matches
    }
    if len(semantic_signatures) != 1 or len(contract_hashes) != 1:
        return {
            **base,
            "contract_resolution_status": "conflict",
            "contract_resolution_reason": "suite_output_contract_exact_matches_conflict",
            "contract_resolution_signatures": [
                list(value) for value in sorted(semantic_signatures, key=repr)
            ],
            "contract_resolution_contract_sha256": sorted(contract_hashes),
        }
    return dict(matches[0])


_AUTHORITATIVE_IDENTITY_FIELDS = (
    "authoritative_output_contract",
    "contract_resolution_status",
    "contract_resolution_reason",
    "contract_status",
    "model_id",
    "backend",
    "variant",
    "task",
    "stage",
    "contract_family",
    "endpoint_mode",
    "host_tail_required",
    "postprocessing_required",
    "requires_external_postprocess",
    "postprocessing",
    "full_end_node_names",
    "output_format",
    "output_record_format",
    "coordinate_format",
    "coordinate_space",
    "source_coordinate_space",
    "score_semantics",
    "class_id_semantics",
    "compiled_artifact_raw_head",
    "decoder_id",
    "decoder_sha256",
    "nms_implementation",
    "class_aware",
    "iou_threshold",
    "score_threshold",
    "max_detections",
    "artifact_path",
    "artifact_sha256",
    "artifact_size_bytes",
    "recorded_artifact_path",
    "recorded_artifact_sha256",
    "recorded_artifact_size_bytes",
    "artifact_binding_status",
    "artifact_binding_sha256",
    "source_contract_sha256",
    "source_contracts_sha256",
    "source_onnx_detection_endpoint",
)


def _revalidate_authoritative_declaration(
    declared_contract: Mapping[str, Any] | str | None,
    *,
    task: str,
    allowed_stages: set[str],
) -> tuple[dict[str, Any], str]:
    """Reload and compare a loader-issued declaration against its root file.

    Status booleans and plausible SHA strings are not trust anchors.  The
    exact suite container is read again, its bytes and selected raw row are
    re-hashed by :func:`load_authoritative_output_contract`, and every field
    which can affect endpoint semantics is compared with the supplied mapping.
    The freshly resolved mapping is returned so unvalidated extra keys can
    never influence an endpoint hash.
    """
    if not isinstance(declared_contract, Mapping):
        return {}, "authoritative_suite_declaration_required"
    supplied = dict(declared_contract)
    source_text = str(supplied.get("declaration_source") or "").strip()
    if not source_text:
        return {}, "authoritative_declaration_source_missing"
    source = Path(source_text).expanduser()
    if not source.is_absolute() or source.name != "output_contracts.json":
        return {}, "authoritative_declaration_source_not_exact_root_container"
    try:
        source = source.resolve(strict=True)
    except (OSError, RuntimeError):
        return {}, "authoritative_declaration_source_unavailable"
    if not source.is_file():
        return {}, "authoritative_declaration_source_unavailable"

    expected_container_sha = str(
        supplied.get("source_contracts_sha256") or ""
    ).strip().lower().removeprefix("sha256:")
    if not _is_sha256(expected_container_sha):
        return {}, "authoritative_container_sha256_missing_or_invalid"
    actual_container_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    if actual_container_sha != expected_container_sha:
        return {}, "authoritative_container_sha256_mismatch"

    model_id = str(supplied.get("model_id") or "").strip()
    backend = str(supplied.get("backend") or "").strip()
    variant = str(supplied.get("variant") or "").strip().lower()
    task_l = str(task or "").strip().lower()
    if not model_id or not backend or variant != "full":
        return {}, "authoritative_model_backend_variant_binding_missing"
    if str(supplied.get("task") or "").strip().lower() != task_l:
        return {}, "authoritative_task_binding_mismatch"

    fresh = load_authoritative_output_contract(
        source,
        backend=backend,
        model_id=model_id,
        variant=variant,
        task=task_l,
    )
    if (
        fresh.get("contract_resolution_status") != "attested"
        or fresh.get("authoritative_output_contract") is not True
    ):
        return {}, "authoritative_declaration_fresh_resolution_failed"
    fresh_stage = str(fresh.get("stage") or "").strip().lower()
    if fresh_stage not in set(allowed_stages):
        return {}, "authoritative_declaration_stage_not_allowed"

    for key in _AUTHORITATIVE_IDENTITY_FIELDS:
        if supplied.get(key) != fresh.get(key):
            return {}, f"authoritative_declaration_field_mismatch:{key}"
    if str(supplied.get("declaration_source") or "") != str(source):
        return {}, "authoritative_declaration_source_path_mismatch"
    return dict(fresh), ""


def _tensor_signature(outputs: Mapping[str, Any]) -> dict[str, Any]:
    tensors: list[dict[str, Any]] = []
    if isinstance(outputs, Mapping):
        for index, (name, value) in enumerate(outputs.items()):
            try:
                array = np.asarray(value)
                tensors.append({
                    "index": int(index),
                    "name": str(name),
                    "rank": int(array.ndim),
                    "shape": [int(dim) for dim in array.shape],
                    "dtype": str(array.dtype),
                })
            except Exception:
                tensors.append({
                    "index": int(index), "name": str(name),
                    "rank": -1, "shape": [], "dtype": "unavailable",
                })
    return {"tensor_count": len(tensors), "tensors": tensors}


def _comparison_tensor_signature(
    signature: Mapping[str, Any], *, task: str = "", stage: str = "",
) -> dict[str, Any]:
    """Return the semantic/shape signature used by the endpoint hash.

    Numeric precision is deliberately excluded.  Endpoint equality answers
    *what* a producer returns (for example logits, raw heads, or decoded NMS),
    while the independent runtime-precision gate answers *how* it was
    calculated.  Including dtype here made Hailo INT8 logits and TensorRT FP16
    logits look like different logical endpoints and duplicated the precision
    gate in a backend-dependent way.
    """
    tensors = []
    for row in signature.get("tensors") or []:
        if not isinstance(row, Mapping):
            continue
        shape = [int(x) for x in (row.get("shape") or [])]
        rank = int(row.get("rank") or 0)
        # A number of native classifier APIs drop the leading singleton batch
        # axis and return ``[C]`` instead of ONNX's ``[1, C]``.  They represent
        # the same semantic endpoint.  Preserve physical shapes in the evidence
        # manifest, but normalize this optional axis in the semantic hash only.
        if (
            str(task or "").strip().lower() == "classification"
            and str(stage or "").strip().lower().startswith("classification_")
            and len(shape) == 2 and shape[0] == 1
        ):
            shape = shape[1:]
            rank = 1
        tensors.append({
            "index": int(row.get("index") or 0),
            "rank": rank,
            "shape": shape,
        })
    return {"tensor_count": int(signature.get("tensor_count") or 0), "tensors": tensors}


def output_endpoint_identity(
    *, task: str, stage: str, output_format: str,
    signature: Mapping[str, Any], declaration: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Project the existing v3 hash payload for runtime and quality consumers."""
    declaration = dict(declaration or {})
    semantic = {
        key: declaration.get(key)
        for key in (
            "coordinate_format", "coordinate_space", "decoder_id",
            "decoder_sha256", "nms_implementation", "class_aware",
            "iou_threshold", "score_threshold", "max_detections",
            "source_onnx_detection_endpoint",
        )
        if declaration.get(key) not in (None, "")
    }
    return {
        "schema": "onnx-splitpoint/output-endpoint-contract",
        "schema_version": 3,
        "task": str(task),
        "stage": str(stage),
        "output_format": str(output_format),
        "tensor_signature": _comparison_tensor_signature(
            signature, task=task, stage=stage,
        ),
        "semantic": semantic,
    }


def _endpoint_hash(
    *, task: str, stage: str, output_format: str,
    signature: Mapping[str, Any], declaration: Mapping[str, Any] | None = None,
) -> str:
    return _canonical_sha256(output_endpoint_identity(
        task=task, stage=stage, output_format=output_format,
        signature=signature, declaration=declaration,
    ))


def _failed_attestation(reason: str, **evidence: Any) -> dict[str, Any]:
    return {
        "schema": DECODED_NMS_ATTESTATION_SCHEMA,
        "schema_version": 3,
        "endpoint": "decoded_nms",
        "stage": "unknown",
        "attested": False,
        "status": "failed",
        "reason": str(reason),
        "contract_source": DECODED_NMS_ATTESTATION_SOURCE,
        **evidence,
    }


def _declared_nms_contract(
    outputs: Mapping[str, Any],
    declared_contract: Mapping[str, Any] | str | None,
) -> tuple[bool, dict[str, Any], str]:
    declaration: dict[str, Any] = {}
    if isinstance(declared_contract, Mapping):
        declaration = dict(declared_contract)
        stage = str(
            declaration.get("stage")
            or declaration.get("contract_family")
            or declaration.get("endpoint")
            or declaration.get("endpoint_mode")
            or ""
        ).strip().lower()
        source = str(
            declaration.get("declaration_source")
            or declaration.get("contract_source")
            or "producer_export_contract"
        ).strip()
        if stage == "decoded_nms" and source:
            revalidated, reason = _revalidate_authoritative_declaration(
                declaration, task="detection", allowed_stages={"decoded_nms"},
            )
            if revalidated:
                return True, revalidated, str(
                    revalidated.get("declaration_source") or source
                )
            return False, {}, f"authoritative_declaration_revalidation_failed:{reason}"
        return False, declaration, source or "declaration_missing"
    if isinstance(declared_contract, str) and declared_contract.strip():
        return False, {}, "string_endpoint_declaration_not_authoritative"

    # Binding names are producer/export metadata, not an inference from values.
    # Only names explicitly containing an NMS operation count as a declaration;
    # generic names such as ``output0`` or ``detections`` deliberately do not.
    names = [str(name).strip().lower() for name in outputs.keys()]
    declared = bool(names) and any(
        token in name for name in names for token in _NMS_NAME_TOKENS
    )
    if declared:
        return True, {"stage": "decoded_nms"}, "runtime_binding_name_explicit_nms"
    return False, {}, "no_explicit_nms_declaration"


def attest_decoded_nms(
    outputs: Mapping[str, Any],
    declared_contract: Mapping[str, Any] | str | None = None,
) -> dict[str, Any]:
    """Attest a decoded endpoint from declaration and runtime values.

    A plausible ``xyxy, score, class`` tensor without an explicit NMS
    declaration is reported as ``decoded_pre_nms_or_unknown`` and cannot open a
    performance-comparison gate.
    A graph-bound end-to-end TopK source retains the legacy stage token, but
    explicitly requires consumer score selection and has no integrated NMS.
    """
    signature = _tensor_signature(outputs)
    if not isinstance(outputs, Mapping) or len(outputs) != 1:
        return _failed_attestation(
            "exactly_one_runtime_output_required", tensor_signature=signature,
        )
    name, value = next(iter(outputs.items()))
    try:
        array = np.asarray(value)
    except Exception as exc:
        return _failed_attestation(
            "runtime_output_not_array", output_name=str(name),
            tensor_signature=signature, error=f"{type(exc).__name__}: {exc}",
        )
    shape = [int(dim) for dim in array.shape]
    base = {
        "output_count": 1, "output_name": str(name), "output_shape": shape,
        "output_dtype": str(array.dtype), "tensor_signature": signature,
    }
    if array.ndim != 3 or array.shape[-1] != 6:
        return _failed_attestation("exact_bn6_shape_required", **base)
    row_count = int(np.prod(array.shape[:-1], dtype=np.int64))
    if row_count <= 0:
        return _failed_attestation("nonempty_detection_rows_required", **base)
    try:
        rows = np.asarray(array, dtype=np.float64).reshape(row_count, 6)
    except Exception as exc:
        return _failed_attestation(
            "runtime_output_not_numeric", **base,
            error=f"{type(exc).__name__}: {exc}",
        )

    finite_rows = np.isfinite(rows).all(axis=1)
    finite_fraction = float(np.mean(finite_rows))
    if not bool(np.all(finite_rows)):
        return _failed_attestation(
            "nonfinite_runtime_values", **base, row_count=row_count,
            finite_fraction=finite_fraction,
        )
    score = rows[:, 4]
    class_id = rows[:, 5]
    score_valid = (score >= -_RANGE_EPSILON) & (score <= 1.0 + _RANGE_EPSILON)
    class_valid = (
        (class_id >= -_RANGE_EPSILON)
        & (np.abs(class_id - np.rint(class_id)) <= _INTEGER_EPSILON)
    )
    xyxy_valid = (
        (rows[:, 2] + _RANGE_EPSILON >= rows[:, 0])
        & (rows[:, 3] + _RANGE_EPSILON >= rows[:, 1])
    )
    score_fraction = float(np.mean(score_valid))
    class_fraction = float(np.mean(class_valid))
    xyxy_fraction = float(np.mean(xyxy_valid))
    evidence = {
        **base, "row_count": row_count, "finite_fraction": finite_fraction,
        "score_range_fraction": score_fraction,
        "integer_nonnegative_class_fraction": class_fraction,
        "ordered_xyxy_fraction": xyxy_fraction,
        "fraction_threshold": _FRACTION_THRESHOLD,
        "range_epsilon": _RANGE_EPSILON,
        "integer_epsilon": _INTEGER_EPSILON,
    }
    failures = []
    declared, declaration, declaration_source = _declared_nms_contract(outputs, declared_contract)
    try:
        selection = bn6_candidate_selection(declaration) if declared else {}
    except ValueError as exc:
        return _failed_attestation(str(exc), **evidence)
    if selection:
        if shape != selection["output_shape"]:
            return _failed_attestation("candidate_graph_output_shape_mismatch", **evidence)
        if not bool(np.all(score_valid) and np.all(class_valid) and np.all(class_id < selection["class_count"])):
            return _failed_attestation("candidate_score_or_class_invalid", **evidence)
        evidence.update(candidate_selection=selection, raw_invalid_geometry_count=int(np.sum(~xyxy_valid)))
    if score_fraction < _FRACTION_THRESHOLD:
        failures.append("score_column_not_probability_like")
    if class_fraction < _FRACTION_THRESHOLD:
        failures.append("class_column_not_integer_nonnegative")
    if not selection and xyxy_fraction < _FRACTION_THRESHOLD:
        failures.append("coordinates_not_ordered_xyxy")
    if failures:
        return _failed_attestation(";".join(failures), **evidence)

    declared, declaration, declaration_source = _declared_nms_contract(
        outputs, declared_contract,
    )
    if not declared:
        return _failed_attestation(
            "decoded_values_do_not_prove_nms;explicit_nms_declaration_required",
            **evidence, values_decoded_xyxy_score_class=True,
            declaration_attested=False,
            declaration_source=declaration_source,
        )
    endpoint_hash = _endpoint_hash(
        task="detection", stage="decoded_nms", output_format="bn6_detections",
        signature=signature, declaration=declaration,
    )
    return {
        "schema": DECODED_NMS_ATTESTATION_SCHEMA,
        "schema_version": 3,
        "endpoint": "decoded_nms",
        "stage": "decoded_nms",
        "attested": True,
        "status": "passed",
        "reason": ("graph_bound_topk_candidates_require_consumer_selection"
                   if selection else "explicit_nms_declaration_and_runtime_values_verified"),
        "contract_source": DECODED_NMS_ATTESTATION_SOURCE,
        "values_decoded_xyxy_score_class": True,
        "declaration_attested": True,
        "declaration_source": declaration_source,
        "declared_contract": declaration,
        "endpoint_contract_hash": endpoint_hash,
        **evidence,
    }


def _declared_classification_stage(
    declared_contract: Mapping[str, Any] | str | None,
) -> str:
    values: list[str] = []
    if isinstance(declared_contract, str):
        values.append(declared_contract)
    elif isinstance(declared_contract, Mapping):
        for key in ("stage", "contract_family", "output_format", "endpoint_mode"):
            values.append(str(declared_contract.get(key) or ""))
        outputs = declared_contract.get("outputs")
        if isinstance(outputs, list):
            for row in outputs:
                if isinstance(row, Mapping):
                    values.append(str(row.get("name") or ""))
                    values.append(str(row.get("stage") or row.get("output_format") or ""))
    joined = " ".join(values).strip().lower()
    if "probabil" in joined or "softmax" in joined:
        return "classification_probabilities"
    if "logit" in joined or "class_score" in joined:
        return "classification_logits"
    return ""


def _authoritative_classification_declaration_reason(
    declared_contract: Mapping[str, Any] | str | None,
) -> str:
    _fresh, reason = _revalidate_authoritative_declaration(
        declared_contract,
        task="classification",
        allowed_stages={
            "classification_logits", "classification_probabilities",
        },
    )
    return reason


def _classification_contract(
    outputs: Mapping[str, Any],
    declared_contract: Mapping[str, Any] | str | None = None,
) -> dict[str, Any]:
    signature = _tensor_signature(outputs)
    failed = {
        "task": "classification", "output_format": "tensor_outputs",
        "contract_family": "unknown", "stage": "unknown",
        "contract_source": "classification_runtime_attestation_failed",
        "endpoint_contract_complete": False,
        "endpoint_contract_hash": "",
        "tensor_signature": signature,
    }
    revalidated_declaration, declaration_failure = (
        _revalidate_authoritative_declaration(
            declared_contract,
            task="classification",
            allowed_stages={
                "classification_logits", "classification_probabilities",
            },
        )
    )
    if declaration_failure:
        return {**failed, "output_endpoint_attestation": {
            "attested": False, "status": "failed",
            "reason": declaration_failure,
        }}
    declared_contract = revalidated_declaration
    if not isinstance(outputs, Mapping) or len(outputs) != 1:
        return {**failed, "output_endpoint_attestation": {
            "attested": False, "status": "failed",
            "reason": "exactly_one_classification_output_required",
        }}
    name, value = next(iter(outputs.items()))
    low_name = str(name).strip().lower()
    try:
        array = np.asarray(value)
    except Exception:
        return failed
    if any(token in low_name for token in _TOPK_NAME_TOKENS):
        return {**failed, "output_endpoint_attestation": {
            "attested": False, "status": "failed", "reason": "topk_or_label_output",
        }}
    # Revalidation already resolved the exact canonical stage from the three
    # authoritative declaration fields.  Do not infer it again from auxiliary
    # fields such as ``output_format`` or nested output names: those fields may
    # be descriptive producer metadata and a contradictory value must never
    # override the root contract's attested stage.
    declared_stage = str(declared_contract.get("stage") or "").strip().lower()
    declared_or_named_scores = bool(
        declared_stage
        or any(token in low_name for token in ("logit", "score", "probab", "softmax"))
    )
    if (
        array.dtype.kind not in "fc"
        and not (array.dtype.kind in "iu" and declared_or_named_scores)
    ) or array.ndim not in (1, 2):
        return {**failed, "output_endpoint_attestation": {
            "attested": False, "status": "failed",
            "reason": "single_floating_rank1_or_rank2_class_tensor_required",
        }}
    class_count = int(array.shape[-1]) if array.shape else 0
    if class_count < 2 or not bool(np.isfinite(array).all()):
        return {**failed, "output_endpoint_attestation": {
            "attested": False, "status": "failed",
            "reason": "finite_multiclass_tensor_required",
        }}
    rows = np.asarray(array, dtype=np.float64).reshape(-1, class_count)
    probability_like = bool(
        np.all(rows >= -1.0e-4) and np.all(rows <= 1.0 + 1.0e-4)
        and np.allclose(rows.sum(axis=1), 1.0, atol=2.0e-3, rtol=2.0e-3)
    )
    stage = declared_stage
    if stage == "classification_probabilities" and not probability_like:
        return {**failed, "output_endpoint_attestation": {
            "attested": False, "status": "failed",
            "reason": "declared_probabilities_not_probability_like",
        }}
    output_format = stage
    endpoint_hash = _endpoint_hash(
        task="classification", stage=stage, output_format=output_format,
        signature=signature,
        declaration=(
            dict(declared_contract)
            if isinstance(declared_contract, Mapping) else None
        ),
    )
    return {
        "task": "classification", "output_format": output_format,
        "contract_family": stage, "stage": stage,
        "contract_source": "authoritative_suite_contract_plus_runtime_tensor:v4",
        "endpoint_contract_complete": True,
        "endpoint_contract_hash": endpoint_hash,
        "tensor_signature": signature,
        "output_endpoint_attestation": {
            "schema": DECODED_NMS_ATTESTATION_SCHEMA, "schema_version": 3,
            "endpoint": stage, "stage": stage, "attested": True,
            "status": "passed", "reason": "single_finite_multiclass_tensor",
            "class_count": class_count,
            "endpoint_contract_hash": endpoint_hash,
            "declaration_source_contract_sha256": str(
                (declared_contract or {}).get("source_contract_sha256")
                if isinstance(declared_contract, Mapping) else ""
            ),
            "declaration_source_contracts_sha256": str(
                (declared_contract or {}).get("source_contracts_sha256")
                if isinstance(declared_contract, Mapping) else ""
            ),
        },
    }


def _raw_runtime_structure(
    outputs: Mapping[str, Any], *, stage: str,
) -> tuple[bool, dict[str, Any], str]:
    signature = _tensor_signature(outputs)
    evidence = {"tensor_signature": signature, "declared_stage": str(stage)}
    if not isinstance(outputs, Mapping) or not outputs:
        return False, evidence, "nonempty_raw_detection_outputs_required"
    arrays: list[np.ndarray] = []
    try:
        arrays = [np.asarray(value) for value in outputs.values()]
    except Exception:
        return False, evidence, "raw_detection_output_not_array"
    if any(array.ndim < 3 or array.size <= 0 for array in arrays):
        return False, evidence, "raw_detection_rank3_or_higher_required"
    try:
        if any(not bool(np.isfinite(array).all()) for array in arrays):
            return False, evidence, "finite_raw_detection_tensors_required"
    except (TypeError, ValueError):
        return False, evidence, "finite_numeric_raw_detection_tensors_required"
    # [B,N,6] is a decoded/pre-NMS layout, never a raw YOLO head.  A declaration
    # cannot relabel it as ``raw_head`` merely to make an endpoint complete.
    if str(stage) == "raw_head" and any(
        array.ndim == 3 and int(array.shape[-1]) == 6 for array in arrays
    ):
        return False, evidence, "bn6_tensor_is_not_raw_detection_head"
    return True, evidence, "raw_detection_tensor_structure_verified"


def runtime_output_contract(
    task: str,
    outputs: Mapping[str, Any],
    *,
    raw_fallback: bool,
    declared_contract: Mapping[str, Any] | str | None = None,
) -> dict[str, Any]:
    """Return a canonical, fail-closed task/output endpoint contract."""
    normalized_task = str(task or "").strip().lower()
    if normalized_task == "classification":
        return _classification_contract(outputs, declared_contract)
    if normalized_task == "detection":
        attestation = attest_decoded_nms(outputs, declared_contract)
        signature = _tensor_signature(outputs)
        if attestation.get("attested") is True:
            return {
                "task": "detection", "output_format": "bn6_detections",
                "contract_family": "decoded_nms", "stage": "decoded_nms",
                "contract_source": DECODED_NMS_ATTESTATION_SOURCE,
                "endpoint_contract_complete": True,
                "endpoint_contract_hash": str(attestation.get("endpoint_contract_hash") or ""),
                "tensor_signature": signature,
                "output_endpoint_attestation": attestation,
            }
        # Unknown is safer than labelling every unfamiliar tensor a raw head.
        # Explicit raw declarations remain represented as raw_head.
        declaration_stage = ""
        if isinstance(declared_contract, Mapping):
            declaration_stage = str(
                declared_contract.get("stage")
                or declared_contract.get("contract_family")
                or declared_contract.get("endpoint")
                or declared_contract.get("endpoint_mode")
                or ""
            ).strip().lower()
            postprocess = declared_contract.get("postprocessing")
            if (
                declaration_stage != "decoded_pre_nms"
                and isinstance(postprocess, Mapping) and (
                postprocess.get("host_required") is True
                or postprocess.get("nms_on_host") is True
                or "host" in str(postprocess.get("type") or "").lower()
                )
            ):
                declaration_stage = "raw_head"
            if (
                declaration_stage != "decoded_pre_nms"
                and declared_contract.get("requires_external_postprocess") is True
            ):
                declaration_stage = "raw_head"
        elif isinstance(declared_contract, str):
            declaration_stage = declared_contract.strip().lower()
        explicitly_raw = declaration_stage in {
            "raw_head", "raw_detection_head", "decoded_pre_nms",
        }
        family = (
            "raw_head" if declaration_stage == "raw_detection_head"
            else declaration_stage if explicitly_raw else "unknown"
        )
        raw_declaration: dict[str, Any] = {}
        raw_declaration_reason = ""
        if explicitly_raw:
            raw_declaration, raw_declaration_reason = (
                _revalidate_authoritative_declaration(
                    declared_contract,
                    task="detection",
                    allowed_stages={"raw_head", "decoded_pre_nms"},
                )
            )
            if raw_declaration:
                family = str(raw_declaration.get("stage") or "").strip().lower()
                declared_contract = raw_declaration
        declared_output_format = (
            str(declared_contract.get("output_format") or "").strip().lower()
            if isinstance(declared_contract, Mapping) else ""
        )
        output_format = (
            (
                declared_output_format or "ultralytics_decoded"
                if family == "decoded_pre_nms"
                else "raw_detection_tensors"
            )
            if explicitly_raw else "tensor_outputs"
        )
        structure_ok = False
        structure_evidence: dict[str, Any] = {}
        structure_reason = "explicit_raw_declaration_required"
        if explicitly_raw and not raw_declaration_reason:
            structure_ok, structure_evidence, structure_reason = _raw_runtime_structure(
                outputs, stage=family,
            )
        if explicitly_raw and (raw_declaration_reason or not structure_ok):
            return {
                "task": "detection", "output_format": "tensor_outputs",
                "contract_family": "unknown", "stage": "unknown",
                "contract_source": "raw_runtime_structure_attestation_failed",
                "endpoint_contract_complete": False,
                "endpoint_contract_hash": "",
                "tensor_signature": signature,
                "raw_fallback_requested": bool(raw_fallback),
                "output_endpoint_attestation": {
                    "attested": False, "status": "failed",
                    "reason": (
                        f"authoritative_raw_declaration_revalidation_failed:{raw_declaration_reason}"
                        if raw_declaration_reason else structure_reason
                    ),
                    **structure_evidence,
                },
            }
        endpoint_hash = (
            _endpoint_hash(
                task="detection", stage=family, output_format=output_format,
                signature=signature,
                declaration=(
                    dict(declared_contract)
                    if isinstance(declared_contract, Mapping) else None
                ),
            )
            if explicitly_raw else ""
        )
        return {
            "task": "detection", "output_format": output_format,
            "contract_family": family, "stage": family,
            "contract_source": (
                "explicit_producer_export_contract" if explicitly_raw
                else "endpoint_declaration_or_runtime_attestation_failed"
            ),
            "endpoint_contract_complete": bool(explicitly_raw),
            "endpoint_contract_hash": endpoint_hash,
            "tensor_signature": signature,
            "raw_fallback_requested": bool(raw_fallback),
            "output_endpoint_attestation": (
                {
                    "attested": True, "status": "passed",
                    "stage": family, "endpoint": family,
                    "reason": structure_reason,
                    "endpoint_contract_hash": endpoint_hash,
                    **structure_evidence,
                }
                if explicitly_raw else attestation
            ),
        }
    return {
        "task": normalized_task or "unknown",
        "output_format": "tensor_outputs", "contract_family": "unknown",
        "stage": "unknown", "contract_source": "task_unavailable",
        "endpoint_contract_complete": False, "endpoint_contract_hash": "",
        "tensor_signature": _tensor_signature(outputs),
    }


def load_manifest_outputs(
    manifest_path: str | Path,
    manifest: Mapping[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    """Load output tensors described by a standard runner dump manifest."""
    path = Path(manifest_path).expanduser()
    payload = dict(manifest or {})
    rows = payload.get("outputs")
    if not isinstance(rows, list):
        return {}
    outputs: dict[str, np.ndarray] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            return {}
        file_text = str(row.get("file") or "").strip()
        shape_raw = row.get("shape")
        dtype_text = str(row.get("dtype") or "").strip()
        if not file_text or not isinstance(shape_raw, (list, tuple)) or not dtype_text:
            return {}
        file_path = Path(file_text).expanduser()
        if not file_path.is_absolute():
            file_path = path.parent / file_path
        if not file_path.is_file():
            return {}
        try:
            shape = tuple(int(dim) for dim in shape_raw)
            dtype = np.dtype(dtype_text)
            array = np.fromfile(file_path, dtype=dtype)
            if array.size != int(np.prod(shape, dtype=np.int64)):
                return {}
            array = array.reshape(shape)
        except Exception:
            return {}
        outputs[str(row.get("name") or f"output_{index}")] = array
    return outputs
