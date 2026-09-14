"""Strict hand-off from central TensorRT quality to Native Full execution.

The generic benchmark suite builds and evaluates one setup-local TensorRT
engine before the Native stage starts.  This module selects only producers
which were technically completed and cryptographically validated by central
quality, then materialises the per-setup, multi-model input consumed by the
Native Full runner.  It intentionally does not inspect remote paths or try to
recover missing evidence: all scientific identity comes from the signed
producer object.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from onnx_splitpoint_tool.quality_service import (
    _validate_candidate_execution_contract,
)
from onnx_splitpoint_tool.native_split_quality import (
    select_central_native_split_quality_binding,
    validate_central_native_split_quality_selection,
    validate_native_split_quality_binding,
)


PRODUCER_SET_SCHEMA = "onnx-splitpoint/tensorrt-quality-producer-set"
PRODUCER_SCHEMA = "onnx-splitpoint/tensorrt-central-quality-producer-identity"
SPLIT_BINDING_SET_SCHEMA = "onnx-splitpoint/native-split-quality-binding-set"


class TensorRTQualityChainError(RuntimeError):
    """Raised when central quality cannot yield one unambiguous producer."""


def _run_token(value: Any) -> str:
    token = _text(value).lower().replace("-", "_")
    for prefix in ("benchmark_results_", "results_"):
        if token.startswith(prefix):
            token = token[len(prefix):]
    if token.endswith("_auto"):
        token = token[:-5]
    aliases = {
        "hailo8_to_tensorrt": "hailo8_to_trt",
        "hailo10_to_tensorrt": "hailo10h_to_trt",
        "hailo10h_to_tensorrt": "hailo10h_to_trt",
        "hailo10_to_trt": "hailo10h_to_trt",
        "deepx_m1_to_tensorrt": "deepx_to_trt",
        "deepx_m1_to_trt": "deepx_to_trt",
        "deepx_to_tensorrt": "deepx_to_trt",
    }
    return aliases.get(token, token)


def _trt_backend_token(value: Any) -> str:
    """Canonicalize only the logical TensorRT backend mirror.

    Central Quality exposes the logical backend as ``tensorrt`` while the
    signed runtime producer remains ``native_tensorrt``.  Both names describe
    the same TensorRT result, but that alias must not weaken any other signed
    producer field.
    """

    return "tensorrt" if value == "native_tensorrt" else value


def _text(value: Any) -> str:
    return str(value or "").strip()


def _sha256_token(value: Any) -> str:
    """Return one canonical digest while accepting the standard URI prefix."""

    token = _text(value).lower()
    if token.startswith("sha256:"):
        token = token[7:]
    return token if re.fullmatch(r"[0-9a-f]{64}", token) else ""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _strict_split_result_identity(
    result: Mapping[str, Any], identity: Mapping[str, Any],
) -> dict[str, str]:
    """Require every duplicated split-result identity to agree exactly."""

    fields = (
        ("eval_run_id", "eval_run_id", lambda value: _text(value)),
        ("model_id", "model_id", lambda value: _text(value).lower()),
        ("case_id", "case_id", lambda value: _text(value).lower()),
        ("source_run_id", "source_run_id", lambda value: _text(value)),
        ("source_setup_id", "setup_id", lambda value: _text(value)),
        ("task", "task", lambda value: _text(value).lower()),
        ("variant", "variant", lambda value: _text(value).lower()),
        (
            "runtime_precision_identity", "runtime_precision_identity",
            lambda value: _text(value).lower().replace(" ", ""),
        ),
    )
    exact: dict[str, str] = {}
    for result_field, identity_field, normalizer in fields:
        result_value = normalizer(result.get(result_field))
        identity_value = normalizer(identity.get(identity_field))
        canonical_field = (
            "setup_id" if result_field == "source_setup_id" else result_field
        )
        if not result_value or not identity_value:
            raise TensorRTQualityChainError(
                f"split quality {canonical_field} identity mirror is missing"
            )
        if result_value != identity_value:
            raise TensorRTQualityChainError(
                f"split quality {canonical_field} identity mirrors differ"
            )
        exact[canonical_field] = result_value

    # Optional aliases are not allowed to disagree with the required source
    # identity merely because a top-level-first fallback would hide them.
    aliases = (
        (result, "setup_id", exact["setup_id"], _text),
        (result, "run_id", exact["source_run_id"], _text),
        (result, "backend", exact["source_run_id"], _run_token),
        (identity, "backend", exact["source_run_id"], _run_token),
    )
    for container, field, expected, normalizer in aliases:
        if field not in container:
            continue
        observed = normalizer(container.get(field))
        expected_value = normalizer(expected)
        if not observed or observed != expected_value:
            raise TensorRTQualityChainError(
                f"split quality optional {field} identity alias differs"
            )
    return exact


def _producer_flat_identity(producer: Mapping[str, Any]) -> dict[str, Any]:
    """Derive every workflow index field from the signed producer object."""

    source = producer["source_onnx"]
    build = producer["build_onnx"]
    engine = producer["engine"]
    trtexec = producer["trtexec"]
    receipt_binding = producer["engine_build_receipt"]
    receipt = receipt_binding["receipt"]
    inner = dict(receipt)
    inner_sha = _text(inner.pop("receipt_sha256")).lower()
    return {
        "producer_identity_sha256": _text(producer.get("producer_identity_sha256")).lower(),
        "eval_run_id": producer.get("eval_run_id"),
        "model_id": producer.get("model_id"),
        "setup_id": producer.get("setup_id"),
        "source_run_id": producer.get("source_run_id"),
        "originating_plan_run_id": producer.get("originating_plan_run_id", ""),
        "case_id": producer.get("case_id"),
        "execution_role": producer.get("execution_role"),
        "backend": producer.get("backend"),
        "variant": producer.get("variant"),
        "task": producer.get("task"),
        "performance_claims_emitted": producer.get("performance_claims_emitted"),
        "source_onnx_path": source.get("path"),
        "source_onnx_sha256": _text(source.get("sha256")).lower(),
        "source_onnx_size_bytes": source.get("size_bytes"),
        "source_model_sha256": _text(source.get("sha256")).lower(),
        "source_model_size_bytes": source.get("size_bytes"),
        "build_onnx_path": build.get("path"),
        "build_onnx_sha256": _text(build.get("sha256")).lower(),
        "build_onnx_size_bytes": build.get("size_bytes"),
        "engine_path": engine.get("path"),
        "engine_sha256": _text(engine.get("sha256")).lower(),
        "engine_size_bytes": engine.get("size_bytes"),
        "runtime_artifact_sha256": _text(engine.get("sha256")).lower(),
        "runtime_artifact_size_bytes": engine.get("size_bytes"),
        "trtexec_path": trtexec.get("path"),
        "trtexec_sha256": _text(trtexec.get("sha256")).lower(),
        "trtexec_size_bytes": trtexec.get("size_bytes"),
        "engine_build_receipt_path": receipt_binding.get("path"),
        "engine_build_receipt_sha256": _text(receipt_binding.get("sha256")).lower(),
        "engine_build_receipt_file_sha256": _text(
            producer.get("engine_build_receipt_file_sha256")
        ).lower(),
        "engine_build_receipt_size_bytes": receipt_binding.get("size_bytes"),
        "trt_engine_build_receipt_sha256": inner_sha,
        "trt_engine_build_receipt_size_bytes": len(_canonical_bytes(inner)),
    }


def _validated_result_producer(
    result: Mapping[str, Any],
    *,
    eval_run_id: str,
    setup_id: str,
    model_id: str,
) -> tuple[dict[str, Any], tuple[str, str]]:
    """Return the signed producer and exact mirror key for one result."""

    nested = (
        result.get("request_identity")
        if isinstance(result.get("request_identity"), Mapping)
        else {}
    )
    if (
        str(result.get("status") or "") != "completed"
        or str(result.get("technical_status") or "completed") != "completed"
        or result.get("producer_binding_eligible") is not True
        or nested.get("producer_binding_eligible") is not True
        or nested.get("identity_valid") is not True
        or int(nested.get("schema_version") or 0) != 4
    ):
        raise TensorRTQualityChainError(
            f"central quality result is not a completed bindable v4 producer: "
            f"setup={setup_id!r} model={model_id!r}"
        )

    result_producer = result.get("producer_identity")
    nested_producer = nested.get("producer_identity")
    if (
        not isinstance(result_producer, Mapping)
        or not isinstance(nested_producer, Mapping)
        or dict(result_producer) != dict(nested_producer)
    ):
        raise TensorRTQualityChainError(
            f"central quality producer mirrors differ: setup={setup_id!r} "
            f"model={model_id!r}"
        )
    producer = copy.deepcopy(dict(result_producer))
    task = _text(producer.get("task")).lower()
    try:
        validated, producer_sha = _validate_candidate_execution_contract(
            producer, role="central quality Native hand-off", task=task,
        )
    except Exception as exc:
        raise TensorRTQualityChainError(
            f"central quality producer failed strict validation for "
            f"setup={setup_id!r} model={model_id!r}: {type(exc).__name__}: {exc}"
        ) from exc
    producer = copy.deepcopy(dict(validated))

    exact = {
        "schema": PRODUCER_SCHEMA,
        "eval_run_id": eval_run_id,
        "setup_id": setup_id,
        "model_id": model_id,
        "source_run_id": "native_full_tensorrt",
        "case_id": "full",
        "execution_role": "full_quality_only",
        "backend": "native_tensorrt",
        "variant": "full",
    }
    for field_name, expected in exact.items():
        observed = producer.get(field_name)
        if observed != expected:
            raise TensorRTQualityChainError(
                f"central quality producer {field_name} mismatch for "
                f"setup={setup_id!r} model={model_id!r}: "
                f"expected={expected!r} observed={observed!r}"
            )
    if producer.get("performance_claims_emitted") is not False:
        raise TensorRTQualityChainError(
            f"quality-only producer emitted performance claims: "
            f"setup={setup_id!r} model={model_id!r}"
        )

    producer_sha = _text(producer_sha).lower()
    request_sha = _sha256_token(result.get("source_request_sha256"))
    nested_request_sha = _sha256_token(nested.get("source_request_sha256"))
    if not producer_sha or _text(producer.get("producer_identity_sha256")).lower() != producer_sha:
        raise TensorRTQualityChainError(
            f"central quality producer SHA is inconsistent: "
            f"setup={setup_id!r} model={model_id!r}"
        )
    if not request_sha or request_sha != nested_request_sha:
        raise TensorRTQualityChainError(
            f"central quality request SHA is missing or inconsistent: "
            f"setup={setup_id!r} model={model_id!r}"
        )
    expected_flat = _producer_flat_identity(producer)
    for container_name, container in (("result", result), ("request_identity", nested)):
        for field_name, expected in expected_flat.items():
            if field_name not in container:
                raise TensorRTQualityChainError(
                    f"{container_name}.{field_name} is missing: "
                    f"setup={setup_id!r} model={model_id!r}"
                )
            observed = container.get(field_name)
            if field_name.endswith("_sha256"):
                observed = _text(observed).lower()
            elif field_name.endswith("_size_bytes"):
                try:
                    observed = int(observed)
                except (TypeError, ValueError):
                    observed = None
            elif field_name == "backend":
                producer_backend = _text(
                    container.get("producer_backend")
                ).lower()
                if (
                    producer_backend
                    and producer_backend
                    != _text(producer.get("backend")).lower()
                ):
                    raise TensorRTQualityChainError(
                        f"{container_name}.producer_backend differs from signed producer: "
                        f"setup={setup_id!r} model={model_id!r}"
                    )
                observed = _trt_backend_token(observed)
                expected = _trt_backend_token(expected)
            if observed != expected:
                raise TensorRTQualityChainError(
                    f"{container_name}.{field_name} differs from signed producer: "
                    f"setup={setup_id!r} model={model_id!r}"
                )
    return producer, (request_sha, producer_sha)


def producer_set_from_central_quality_summary(
    summary: Mapping[str, Any],
    *,
    eval_run_id: str,
    setup_id: str,
    model_ids: Sequence[str],
) -> dict[str, Any]:
    """Select one exact quality-first producer for every requested model.

    Byte-identical JSON/CSV mirrors are tolerated only when both the portable
    request SHA and signed producer SHA agree.  Any second independent request,
    producer drift, failed result, or missing model makes the hand-off fail
    closed before Native Full starts.
    """

    eval_run_id = _text(eval_run_id)
    setup_id = _text(setup_id)
    requested_models = [_text(item) for item in model_ids]
    if not eval_run_id or not setup_id or not requested_models or any(not item for item in requested_models):
        raise TensorRTQualityChainError("eval_run_id, setup_id and model_ids must be non-empty")
    if len(set(requested_models)) != len(requested_models):
        raise TensorRTQualityChainError("model_ids contain duplicates")
    if (
        summary.get("schema") != "onnx-splitpoint/central-quality-summary"
        or int(summary.get("schema_version") or 0) != 1
        or not isinstance(summary.get("results"), list)
    ):
        raise TensorRTQualityChainError("central quality summary schema is invalid")
    merge = summary.get("merge") if isinstance(summary.get("merge"), Mapping) else {}
    if int(merge.get("summary_only_native_full_quality_conflict_count") or 0) != 0:
        raise TensorRTQualityChainError("central quality summary reports Native Full producer conflicts")

    producers_by_model: dict[str, dict[str, Any]] = {}
    for model_id in requested_models:
        accepted: list[tuple[dict[str, Any], tuple[str, str]]] = []
        candidate_errors: list[str] = []
        for raw_result in summary.get("results") or []:
            if not isinstance(raw_result, Mapping):
                continue
            producer_hint = raw_result.get("producer_identity")
            nested_hint = raw_result.get("request_identity")
            nested_producer = (
                nested_hint.get("producer_identity")
                if isinstance(nested_hint, Mapping)
                and isinstance(nested_hint.get("producer_identity"), Mapping)
                else {}
            )
            hinted = producer_hint if isinstance(producer_hint, Mapping) else nested_producer
            containers = [raw_result]
            if isinstance(nested_hint, Mapping):
                containers.append(nested_hint)
            if isinstance(hinted, Mapping):
                containers.append(hinted)
            # The central summary also contains vendor-Full and composed split
            # results for the same model/setup.  They are not malformed TRT
            # producers and must not poison this exact-role selector.  Once a
            # row declares either half of the target role, however, keep it in
            # scope so the strict validator can reject incomplete/tampered
            # producer records fail-closed.
            declares_trt_full_quality_role = any(
                _text(container.get("source_run_id")) == "native_full_tensorrt"
                or _text(container.get("execution_role")) == "full_quality_only"
                or _text(container.get("schema")) == PRODUCER_SCHEMA
                for container in containers
            )
            if not declares_trt_full_quality_role:
                continue
            belongs_to_requested_scope = any(
                _text(container.get("eval_run_id")) == eval_run_id
                and _text(
                    container.get("setup_id")
                    or container.get("source_setup_id")
                ) == setup_id
                and _text(container.get("model_id")) == model_id
                for container in containers
            )
            if not belongs_to_requested_scope:
                continue
            try:
                accepted.append(_validated_result_producer(
                    raw_result,
                    eval_run_id=eval_run_id,
                    setup_id=setup_id,
                    model_id=model_id,
                ))
            except TensorRTQualityChainError as exc:
                candidate_errors.append(str(exc))
        if candidate_errors:
            raise TensorRTQualityChainError("; ".join(sorted(set(candidate_errors))))
        if not accepted:
            raise TensorRTQualityChainError(
                f"no completed central TensorRT quality producer for "
                f"setup={setup_id!r} model={model_id!r}"
            )
        mirror_keys = {key for _, key in accepted}
        canonical_payloads = {_canonical_bytes(producer) for producer, _ in accepted}
        if len(mirror_keys) != 1 or len(canonical_payloads) != 1:
            raise TensorRTQualityChainError(
                f"ambiguous central TensorRT quality producer for "
                f"setup={setup_id!r} model={model_id!r}"
            )
        producers_by_model[model_id] = accepted[0][0]

    return {
        "schema": PRODUCER_SET_SCHEMA,
        "schema_version": 1,
        "eval_run_id": eval_run_id,
        "setup_id": setup_id,
        "producers_by_model": producers_by_model,
    }


def split_binding_set_from_central_quality_summary(
    summary: Mapping[str, Any],
    *,
    eval_run_id: str,
    setup_id: str,
    selections: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Select exact Quality-FIRST split bindings for later Native execution.

    ``selections`` contains the expected ``model_id``, ``case_id`` and
    ``backend`` (optionally ``task`` and ``precision``).  No backend or
    precision aliases are used after the source-run spelling is normalised.
    Byte-identical collection mirrors collapse only when the request SHA and
    complete sealed binding are identical.
    """

    eval_run_id = _text(eval_run_id)
    setup_id = _text(setup_id)
    requested = [dict(item) for item in selections if isinstance(item, Mapping)]
    if not eval_run_id or not setup_id or not requested or len(requested) != len(selections):
        raise TensorRTQualityChainError(
            "eval_run_id, setup_id and non-empty split selections are required"
        )
    if (
        summary.get("schema") != "onnx-splitpoint/central-quality-summary"
        or int(summary.get("schema_version") or 0) != 1
        or not isinstance(summary.get("results"), list)
    ):
        raise TensorRTQualityChainError("central quality summary schema is invalid")

    bindings: dict[str, dict[str, Any]] = {}
    seen_scope: set[tuple[str, str, str]] = set()
    for selection in requested:
        model_id = _text(selection.get("model_id"))
        case_id = _text(selection.get("case_id")).lower()
        backend = _run_token(selection.get("backend"))
        scope = (model_id, case_id, backend)
        if not all(scope) or backend not in {
            "hailo8_to_trt", "hailo10h_to_trt", "deepx_to_trt",
        }:
            raise TensorRTQualityChainError(
                f"invalid Native split selection: {selection!r}"
            )
        if scope in seen_scope:
            raise TensorRTQualityChainError(
                f"duplicate Native split selection: {scope!r}"
            )
        seen_scope.add(scope)
        accepted: list[
            tuple[dict[str, Any], tuple[str, str, bytes]]
        ] = []
        candidate_errors: list[str] = []
        for raw in summary.get("results") or []:
            if not isinstance(raw, Mapping):
                continue
            identity = raw.get("request_identity")
            identity = identity if isinstance(identity, Mapping) else {}
            possible_scope = bool(
                model_id in {
                    _text(raw.get("model_id")),
                    _text(identity.get("model_id")),
                }
                and case_id in {
                    _text(raw.get("case_id")).lower(),
                    _text(identity.get("case_id")).lower(),
                }
                and backend in {
                    _run_token(raw.get("source_run_id")),
                    _run_token(identity.get("source_run_id")),
                }
                and setup_id in {
                    _text(raw.get("source_setup_id")),
                    _text(raw.get("setup_id")),
                    _text(identity.get("setup_id")),
                }
            )
            if not possible_scope:
                continue
            try:
                exact_identity = _strict_split_result_identity(raw, identity)
            except TensorRTQualityChainError as exc:
                candidate_errors.append(f"{scope!r}: {exc}")
                continue
            observed_scope = (
                exact_identity["model_id"],
                exact_identity["case_id"],
                _run_token(exact_identity["source_run_id"]),
            )
            if observed_scope != scope or exact_identity["setup_id"] != setup_id:
                candidate_errors.append(
                    f"split quality exact scope differs for {scope!r}"
                )
                continue
            if exact_identity["eval_run_id"] != eval_run_id:
                candidate_errors.append(
                    f"split quality eval_run_id mismatch for {scope!r}"
                )
                continue
            expected_task = _text(selection.get("task")).lower()
            expected_precision = (
                _text(selection.get("precision")).lower().replace(" ", "")
            )
            if (
                (expected_task and exact_identity["task"] != expected_task)
                or exact_identity["variant"] != "composed"
                or (
                    expected_precision
                    and exact_identity["runtime_precision_identity"]
                    != expected_precision
                )
            ):
                candidate_errors.append(
                    f"split quality task/variant/precision mismatch for {scope!r}"
                )
                continue
            if (
                str(raw.get("status") or "") != "completed"
                or str(raw.get("technical_status") or "completed") != "completed"
                or identity.get("identity_valid") is not True
                or raw.get("native_split_quality_binding_required") is not True
                or identity.get("native_split_quality_binding_required") is not True
            ):
                candidate_errors.append(
                    f"split quality result is not completed and binding-required for {scope!r}"
                )
                continue
            result_binding = raw.get("native_split_quality_binding")
            identity_binding = identity.get("native_split_quality_binding")
            if (
                not isinstance(result_binding, Mapping)
                or not isinstance(identity_binding, Mapping)
                or dict(result_binding) != dict(identity_binding)
            ):
                candidate_errors.append(
                    f"split quality binding mirrors differ for {scope!r}"
                )
                continue
            expected_identity = {
                "model": model_id,
                "case": case_id,
                "setup_id": setup_id,
                "backend": backend,
                "task": selection.get("task"),
                "precision": selection.get("precision"),
            }
            binding, binding_status = validate_native_split_quality_binding(
                result_binding, expected_identity=expected_identity,
            )
            if binding is None:
                candidate_errors.append(
                    f"split quality binding invalid for {scope!r}: {binding_status}"
                )
                continue
            binding_sha = _text(binding.get("binding_sha256")).lower()
            for container_name, container in (
                ("result", raw), ("request_identity", identity),
            ):
                if (
                    _text(container.get("native_split_quality_binding_sha256")).lower()
                    != binding_sha
                ):
                    candidate_errors.append(
                        f"{container_name} split binding SHA differs for {scope!r}"
                    )
                    binding = None
                    break
            if binding is None:
                continue
            result_request_sha = _sha256_token(raw.get("source_request_sha256"))
            nested_request_sha = _sha256_token(identity.get("source_request_sha256"))
            if (
                not result_request_sha or result_request_sha != nested_request_sha
            ):
                candidate_errors.append(
                    f"split quality request SHA missing or inconsistent for {scope!r}"
                )
                continue
            central_result_sha = _canonical_sha256(raw)
            try:
                selected_binding = select_central_native_split_quality_binding(
                    binding,
                    source_request_sha256=result_request_sha,
                    central_result_sha256=central_result_sha,
                    central_identity=exact_identity,
                )
            except ValueError as exc:
                candidate_errors.append(
                    f"split quality Central selection invalid for {scope!r}: {exc}"
                )
                continue
            selected_receipt, selected_status = (
                validate_central_native_split_quality_selection(
                    selected_binding, required=True,
                )
            )
            if selected_receipt is None:
                candidate_errors.append(
                    f"split quality Central selection invalid for {scope!r}: "
                    f"{selected_status}"
                )
                continue
            accepted.append((copy.deepcopy(selected_binding), (
                result_request_sha, central_result_sha,
                _canonical_bytes(selected_binding),
            )))
        if candidate_errors:
            raise TensorRTQualityChainError("; ".join(sorted(set(candidate_errors))))
        if not accepted:
            raise TensorRTQualityChainError(
                f"no completed Native split quality binding for {scope!r}"
            )
        mirror_keys = {key for _, key in accepted}
        if len(mirror_keys) != 1:
            raise TensorRTQualityChainError(
                f"ambiguous Native split quality binding for {scope!r}"
            )
        bindings["|".join(scope)] = accepted[0][0]

    payload = {
        "schema": SPLIT_BINDING_SET_SCHEMA,
        "schema_version": 2,
        "eval_run_id": eval_run_id,
        "setup_id": setup_id,
        "central_quality_summary_sha256": _canonical_sha256(summary),
        "bindings_by_model_case_backend": bindings,
    }
    payload["binding_set_sha256"] = _canonical_sha256(payload)
    return payload


def load_producer_set_from_central_quality_summary(
    summary_path: str | Path,
    *,
    eval_run_id: str,
    setup_id: str,
    model_ids: Sequence[str],
) -> dict[str, Any]:
    """Load a summary with duplicate-key rejection and build its producer set."""

    path = Path(summary_path)

    def _object_pairs_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise TensorRTQualityChainError(
                    f"duplicate JSON key in central quality summary: {key!r}"
                )
            result[key] = value
        return result

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_object_pairs_no_duplicates,
        )
    except TensorRTQualityChainError:
        raise
    except Exception as exc:
        raise TensorRTQualityChainError(
            f"central quality summary is not valid JSON: {path}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise TensorRTQualityChainError("central quality summary root is not an object")
    return producer_set_from_central_quality_summary(
        payload,
        eval_run_id=eval_run_id,
        setup_id=setup_id,
        model_ids=model_ids,
    )


def load_split_binding_set_from_central_quality_summary(
    summary_path: str | Path,
    *,
    eval_run_id: str,
    setup_id: str,
    selections: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Load a duplicate-key-safe summary and select Native split bindings."""

    path = Path(summary_path)

    def _object_pairs_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise TensorRTQualityChainError(
                    f"duplicate JSON key in central quality summary: {key!r}"
                )
            result[key] = value
        return result

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_object_pairs_no_duplicates,
        )
    except TensorRTQualityChainError:
        raise
    except Exception as exc:
        raise TensorRTQualityChainError(
            f"central quality summary is not valid JSON: {path}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise TensorRTQualityChainError(
            "central quality summary root is not an object"
        )
    return split_binding_set_from_central_quality_summary(
        payload,
        eval_run_id=eval_run_id,
        setup_id=setup_id,
        selections=selections,
    )
