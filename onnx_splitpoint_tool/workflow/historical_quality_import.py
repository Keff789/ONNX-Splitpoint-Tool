"""Explicit, read-only Quality imports for admitted historical Generic FPS.

This is an import relation, never execution admission.  Both frozen contracts
remain original.  Historical postflight evidence is checked by its existing
oracle verifier; it is not presented to, or added to, the execution verifier.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

from .. import generic_only_plan as quality_admission
from ..generic_completion_plan import completion_identity
from ..native_detection_postprocess import (
    _REV3_REPORTING_COMPATIBLE_IMPLEMENTATIONS,
    _implementation_artifacts,
    verify_detection_completion_execution_contract,
)
from ..native_three_stage import verify_fast_completion_attestation
from .central_quality_join import apply_central_quality_result_to_row


RELATION_SCHEMA = "onnx-splitpoint/historical-completed-operator-quality-import"
RELATION_MODE = "historical_completed_operator_quality_import"


@dataclass(frozen=True)
class HistoricalQualityImport:
    """Bound originals and the normal Quality producer's own admission row."""

    fps_result: Mapping[str, Any]
    fps_plan: Mapping[str, Any]
    native_record: Mapping[str, Any]
    quality_config: Mapping[str, Any]
    quality_row: Mapping[str, Any]
    quality_admission: Mapping[str, Any]
    producer_source_authority: Any = None


def _need(condition, message):
    if not condition:
        raise ValueError("historical_quality_import: " + message)


def _json(descriptor):
    value = quality_admission._json(descriptor)
    _need(isinstance(value, Mapping), "bound object missing")
    if "size_bytes" in descriptor:
        path = quality_admission.read_bound(descriptor)
        _need(type(descriptor["size_bytes"]) is int
              and path.stat().st_size == descriptor["size_bytes"], "bound size differs")
    return value


def _sha(value):
    _need(isinstance(value, str) and len(value) == 64
          and all(c in "0123456789abcdef" for c in value), "SHA identity missing")
    return value


def _artifact(value):
    _need(isinstance(value, Mapping), "artifact missing")
    return (_sha(value.get("sha256")), value.get("size_bytes"))


def _same_artifact(left, right, role):
    a, b = _artifact(left), _artifact(right)
    _need(a[0] == b[0] and type(a[1]) is int and a[1] > 0 and a == b,
          "artifact differs: " + role)


def _operator(contract, *, model_sha256):
    """Remove only verified derived seals and explicitly separate context."""
    value = deepcopy(verify_detection_completion_execution_contract(contract))
    _need(value["completion_mode"] == "decoded_nms_attested_materialization_no_second_nms"
          and value["source_endpoint"]["stage"] == "decoded_nms",
          "historical operator family unsupported")
    for key in ("contract_sha256", "hash_layers", "implementation_sha256",
                "schema_sha256", "relation_sha256"):
        value.pop(key)
    source = value["source_endpoint"]
    if "model_sha256" in source:
        _need(source.pop("model_sha256") == model_sha256, "full model differs")
    source.pop("output_endpoint_attestation_sha256")
    processor = value["processor_contract"]
    processor.pop("contract_sha256")
    processor.pop("source_output_endpoint_attestation_sha256")
    # The caller has already admitted the exact existing Rev3 reporting pair.
    processor["implementation_artifacts"]["native_detection_postprocess"]["sha256"] = "reporting-compatible-rev3"
    completed = value["completed_endpoint_contract"]
    for key in ("endpoint_contract_hash", "materialization_contract_sha256", "output_endpoint_id"):
        completed.pop(key)
    return value


def _quality_proof_equal(original, current):
    old, new = deepcopy(original), deepcopy(current)
    # A new controller may live at another path. Its bytes must still match;
    # the original producer observer and every scientific proof remain exact.
    for proof in (old, new):
        consumer = (proof.get("completed_quality_binding") or {}).get("consumer_observer_implementation")
        if consumer is not None:
            _need(isinstance(consumer, dict), "consumer observer context malformed")
            consumer.pop("path", None)
    _need(old == new, "original Quality admission differs from its own normal revalidation")


def _postflight(raw, contract, *, count, prepared_sha=None, source_image_sha=None):
    proof = verify_fast_completion_attestation(raw, execution_contract=contract)
    _need(proof["completed_work_units"] == count, "postflight completion count differs")
    sentinel = proof.get("sentinel_identity") or {}
    _need(sentinel.get("source_snapshot_ownership") == "owned_postflight_after_fifo_drain",
          "owned historical postflight capture missing")
    if prepared_sha is not None:
        _need(sentinel.get("prepared_input_sha256") == prepared_sha, "postflight prepared input differs")
    if source_image_sha is not None:
        _need(sentinel.get("input_image_sha256") == source_image_sha, "Native source image differs")
    return proof


def _admit(target, binding):
    _need(type(binding) is HistoricalQualityImport, "explicit typed historical import required")
    _need(target.get("prior_generic_sourceversion_reuse_admitted") is True
          and target.get("admission_mode") == "reused_prior_generic_sourceversion_h8",
          "target is not an admitted historical Generic FPS row")
    sources = target.get("source_authorities") or {}
    _need(sources.get("reused_original_result") == binding.fps_result
          and sources.get("reused_original_plan") == binding.fps_plan,
          "accepted FPS source authorities differ")
    fps, plan, native, config = (_json(d) for d in (
        binding.fps_result, binding.fps_plan, binding.native_record, binding.quality_config))
    identity = fps.get("identity") or {}
    key = completion_identity(identity)
    _need(completion_identity(target.get("full_product_identity") or {}) == key
          and target.get("full_product_identity") == identity
          and identity.get("direction") == "hailo8_to_trt"
          and target.get("generic_uid") == "|".join(("generic", identity["model_id"],
              "H8", identity["direction"], identity["case_id"]))
          and target.get("model_id") == identity["model_id"]
          and target.get("case_id") == identity["case_id"],
          "target/FPS physical identity differs")
    matches = [r for r in plan.get("rows", []) if isinstance(r, Mapping)
               and r.get("identity") == identity]
    _need(len(matches) == 1, "original FPS plan row absent or ambiguous")
    original = matches[0]
    _need(original.get("result_path") == binding.fps_result.get("path")
          and original.get("native_source") == binding.native_record.get("path"),
          "actual historical backing differs from the original plan")
    runtime = original.get("runtime_config") or {}
    _need(fps.get("status") == "ok" and fps.get("task") == "detection"
          and fps.get("measurement_mode") == "generic_completed_task"
          and fps.get("input_binding") == runtime.get("input_binding")
          and fps.get("measurement_boundary") == runtime.get("measurement_boundary")
          == "prepared_input_before_p1_to_last_completed_task",
          "original FPS measurement/input method differs")
    fps_contract = verify_detection_completion_execution_contract(fps.get("completion_contract"))
    q_contract = verify_detection_completion_execution_contract(config.get("completion_contract"))
    _need(runtime.get("completion_contract") == fps_contract
          and original.get("completion_contract") == fps_contract,
          "original FPS frozen contract differs")
    left_impl = fps_contract["processor_contract"]["implementation_artifacts"]["native_detection_postprocess"]["sha256"]
    right_impl = q_contract["processor_contract"]["implementation_artifacts"]["native_detection_postprocess"]["sha256"]
    _need(left_impl in _REV3_REPORTING_COMPATIBLE_IMPLEMENTATIONS
          and right_impl == _implementation_artifacts()["native_detection_postprocess"]["sha256"],
          "unknown historical reporting implementation pair")
    model_sha = _sha(target.get("model_sha256"))
    _need(original.get("model_sha256") == model_sha
          and fps_contract["model_id"] == identity["model_id"] == q_contract["model_id"]
          and _operator(fps_contract, model_sha256=model_sha) == _operator(q_contract, model_sha256=model_sha),
          "scientific completed operator parameters differ")

    qrow = deepcopy(binding.quality_row)
    _need(qrow.get("completion_contract") == q_contract
          and qrow.get("generic_artifact_binding") == config.get("generic_artifact_binding")
          and all(qrow.get("identity", {}).get(k) == identity[k]
                  and config.get("identity", {}).get(k) == identity[k]
                  for k in ("model_id", "case_id", "setup_id", "comparison_backend", "precision")),
          "Quality own base/config/physical identity differs")
    q_artifacts = (config.get("generic_artifact_binding") or {}).get("artifacts") or {}
    authority = target.get("actual_artifact_authority") or {}
    for role, value in (("source_part1_onnx", authority.get("source_part1_onnx")),
                        ("source_part2_onnx", authority.get("source_part2_onnx")),
                        ("part1_runtime", authority.get("part1")),
                        ("engine", (authority.get("part2_engine_observation") or {}).get("artifact"))):
        _same_artifact(value, q_artifacts.get(role), role)
    old_artifacts = original.get("expected_runtime_artifacts") or {}
    for role in ("part1_runtime", "engine", "source_part2_onnx", "build_part2_onnx", "boundary_metadata"):
        _same_artifact(old_artifacts.get(role), q_artifacts.get(role), role)
    for role in ("part1_runtime", "engine"):
        _need(_artifact(fps.get("expected_runtime_artifacts", {}).get(role))[0]
              == _artifact(q_artifacts.get(role))[0], "actual FPS artifact differs: " + role)

    # This is the unchanged current Generic verifier, applied to Q's OWN base.
    # An historical timing contract is never substituted into this call.
    quality = quality_admission.validate_quality_reference(qrow,
        producer_source_authority=binding.producer_source_authority)
    _quality_proof_equal(_json(binding.quality_admission), quality)
    result = _json(qrow["quality_reference"]["result"])
    _need(result.get("model_sha256") == model_sha, "Quality full model differs")
    representative = (quality["completed_quality_binding"].get("representative_input") or {})
    prepared = (fps.get("input_binding") or {}).get("prepared_input") or {}
    q_prepared = representative.get("prepared_input") or {}
    image_sha = _sha((fps.get("input_binding") or {}).get("source_image_sha256"))
    _need(all(prepared.get(k) is not None and prepared[k] == q_prepared.get(k)
              for k in ("sha256", "shape", "dtype"))
          and representative.get("source_image_sha256") == image_sha
          and representative.get("original_wh") == fps_contract["processor_contract"]["original_wh"],
          "Quality/FPS actual prepared input or geometry differs")
    preprocessing = (fps.get("input_binding") or {}).get("preprocessing_contract_sha256")
    _need(preprocessing and result.get("preprocessing_contract_sha256") == preprocessing,
          "Quality/FPS preprocessing differs")

    _need(native.get("completion_execution_contract") == fps_contract
          and native.get("case") == identity["case_id"]
          and native.get("setup_id") == identity["setup_id"]
          and native.get("precision") == identity["precision"]
          and native.get("task") == "detection" and native.get("ok") is True,
          "actual Native backing identity/contract differs")
    command = native.get("native_command_contract") or {}
    from ..native_command_contract import verify_native_command_contract
    command, command_status = verify_native_command_contract(command)
    _need(command is not None and command_status in {
              "hash_schema_identity_and_artifacts_verified",
              "hash_schema_identity_artifacts_and_quality_first_crosslinks_verified"}
          and command == original.get("native_command_contract")
          and native.get("native_command_contract_sha256") == command.get("contract_sha256")
          and command.get("input_image_sha256") == image_sha
          and all(command.get(k) == identity[v] for k, v in (
              ("model", "model_id"), ("case", "case_id"), ("setup_id", "setup_id"),
              ("precision", "precision"), ("comparison_backend", "comparison_backend"), ("backend", "direction"))),
          "Native original command binding differs")
    for native_role, role in (("hef", "part1_runtime"), ("engine", "engine")):
        _same_artifact(command.get("artifacts", {}).get(native_role), q_artifacts.get(role), native_role)
    feed = native.get("timing_input_binding") or {}
    _need(feed.get("canonical_rgb_sha256") == prepared["sha256"]
          and feed.get("input_image_sha256") == image_sha
          and feed.get("preprocessing_contract_sha256") == preprocessing
          and feed.get("executed_rgb_matches_common_preprocessing") is True,
          "owned Native input/preprocessing backing differs")
    count = runtime.get("task_count")
    warmup = runtime.get("warmup_count")
    _need(type(count) is int and count > 0 and type(warmup) is int and warmup >= 0
          and native.get("completed_work_units") == native.get("requested_frames") == count
          and native.get("frames") == count and native.get("warmup") == warmup
          and native.get("postprocess_completed_frames") == count
          and native.get("postprocess_completion_verified") is True
          and native.get("completed_work_units_status") == "exact_runtime_counter",
          "Native owned capture/count backing differs")
    native_proof = _postflight(native.get("completion_execution_attestation"), fps_contract,
                              count=count, source_image_sha=image_sha)
    reps = fps.get("repetitions")
    _need(isinstance(reps, list) and len(reps) == runtime.get("repetitions") == 3
          and fps.get("process_repetition_policy") == "separate_processes",
          "accepted FPS repetition method differs")
    processes, instances = set(), set()
    source_files = runtime.get("runtime_source_files") or []
    source_map = {item["path"]: _sha(item.get("sha256")) for item in source_files}
    _need(source_map and len(source_map) == len(source_files), "original runtime source closure missing")
    for index, rep in enumerate(reps, 1):
        process = (rep.get("boot_id"), rep.get("pid"), rep.get("process_start_ticks"))
        instance = rep.get("runtime_instance_id")
        _need(rep.get("repetition_index") == index and rep.get("status") == "ok"
              and process[0] and type(process[1]) is int and process[1] > 0
              and type(process[2]) is int and process[2] > 0 and instance
              and process not in processes and instance not in instances
              and rep.get("completed_task_count") == count
              and rep.get("requested_task_count") == count
              and rep.get("requested_warmup_count") == rep.get("warmup_completed_count") == warmup
              and rep.get("processor_count_delta") == count + warmup
              and rep.get("queue_depth") == runtime.get("queue_depth")
              and rep.get("worker_count") == 2
              and rep.get("measurement_boundary") == runtime["measurement_boundary"]
              and rep.get("fill_and_drain_included") is True
              and rep.get("preprocessing_timed") is False and rep.get("handoff_timed") is True
              and rep.get("prepared_input_adapter_timed") is True
              and rep.get("postprocess_timed") is True and rep.get("output_materialization_timed") is True
              and rep.get("completion_runtime_mode") == "fast_oracle_outside_timing",
              "accepted FPS count/process/method differs")
        source_binding = rep.get("runtime_source_binding") or {}
        _need(source_binding.get("unchanged") is True
              and source_binding.get("before") == source_binding.get("after") == source_map,
              "accepted FPS original runtime source closure differs")
        for role in ("part1_runtime", "engine"):
            _need(_artifact((rep.get("runtime_artifacts") or {}).get(role))[0]
                  == _artifact(q_artifacts.get(role))[0], "FPS repetition artifact differs: " + role)
        processes.add(process); instances.add(instance)
        _postflight(rep.get("completion_execution_attestation"), fps_contract,
                    count=count + warmup, prepared_sha=prepared["sha256"])
    _need(math.isfinite(float(fps.get("completed_task_fps")))
          and float(target.get("effective_generic_fps")) == float(fps["completed_task_fps"]),
          "accepted FPS value differs")
    relation = {"schema": RELATION_SCHEMA, "schema_version": 1, "relation": RELATION_MODE,
        "historical_import_only": True, "current_execution_admission": False,
        "original_source_attestation_body_verified": False,
        "fps_completion_contract_sha256": fps_contract["contract_sha256"],
        "quality_completion_contract_sha256": q_contract["contract_sha256"],
        "fps_source_observation_sha256": fps_contract["source_endpoint"]["output_endpoint_attestation_sha256"],
        "quality_source_observation_sha256": q_contract["source_endpoint"]["output_endpoint_attestation_sha256"],
        "historical_observation_relation": native_proof["observation_relation"],
        "historical_capture_ownership": native_proof["sentinel_identity"]["source_snapshot_ownership"],
        "native_oracle_attestation_sha256": native_proof["attestation_sha256"],
        "fps_result": deepcopy(binding.fps_result), "fps_plan": deepcopy(binding.fps_plan),
        "actual_native_backing": deepcopy(binding.native_record),
        "quality_config": deepcopy(binding.quality_config),
        "original_quality_admission": deepcopy(binding.quality_admission),
        "quality_request_sha256": quality["request_sha256"], "quality_result_sha256": quality["result_sha256"],
        "quality_status": quality["status"], "quality_eligible": quality["quality_eligible"]}
    return result, quality, relation


def import_historical_operator_quality(rows: Sequence[Mapping[str, Any]],
                                       imports: Mapping[str, HistoricalQualityImport]):
    """Return Quality-only copies, leaving all FPS/repeats and other rows exact.

    Callers own normal index/merge publication and CURRENT. There is no file
    write, execution, inference or statistics evaluation in this import API.
    """
    result = deepcopy(list(rows))
    keys = [row.get("generic_uid") for row in result]
    _need(isinstance(imports, Mapping) and imports and len(keys) == len(set(keys))
          and all(uid and keys.count(uid) == 1 for uid in imports),
          "target absent, duplicate or ambiguous")
    with quality_admission.validation_scope():
        for uid, binding in imports.items():
            index = keys.index(uid)
            original = deepcopy(result[index])
            value, proof, relation = _admit(original, binding)
            row = result[index]
            identity = deepcopy(value.get("request_identity") or {})
            identity.update(source_request_sha256=proof["request_sha256"],
                            model_id=value["model_id"], case_id=value["case_id"], setup_id=value["setup_id"])
            _need(apply_central_quality_result_to_row(row, value, identity, variant="composed",
                match_mode=RELATION_MODE, match_role="primary", result_cancelled=False, smoke_quality=False),
                "normal Quality projection not completed")
            row.update(quality_status=proof["status"], quality_eligible=proof["quality_eligible"],
                       quality_role=RELATION_MODE, historical_completed_operator_quality_relation=relation)
            for field in ("effective_generic_fps", "effective_repeats", "new_generic_fps", "new_repeats",
                          "new_valid_repeat_count", "fps_status", "measurement_status", "full_product_identity",
                          "reused_runtime", "source_authorities"):
                _need(row.get(field) == original.get(field), "Quality projection changed FPS/provenance: " + field)
            _need((row.get("attempts") or {}).get("fps") == (original.get("attempts") or {}).get("fps"),
                  "Quality projection changed FPS attempts")
    return result
