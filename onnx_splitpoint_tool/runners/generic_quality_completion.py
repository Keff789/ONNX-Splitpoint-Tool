"""Explicit Generic dataset-quality completion using the normal timed factory.

This module owns no device, compilation, CPU reference or metric evaluation. It
materializes one serial composed output per image and carries actual completion
artifacts into the existing central Quality request/candidate transport.
"""
from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import sys

import numpy as np

from . import generic_quality_endpoint as _endpoint

CONFIG_SCHEMA = "onnx-splitpoint/generic-composed-quality-completion-config"
EVIDENCE_SCHEMA = "onnx-splitpoint/generic-composed-quality-completion-evidence"
FIELD = "generic_completed_quality"
ARTIFACT_ROLES = {"part1_runtime", "engine", "source_part1_onnx", "source_part2_onnx",
                  "build_part2_onnx", "engine_build_receipt", "boundary_metadata", "part1_build_receipt"}


def capture_runtime_context(boundary_runtime):
    """Observe the live, already resolved boundary outside all performance timers."""
    from .generic_energy import execution_environment
    executable = Path(sys.executable).absolute()
    value = {"schema": "onnx-splitpoint/generic-quality-runtime-context", "schema_version": 1,
        "boundary_runtime": deepcopy(boundary_runtime),
        "environment": {"executable": str(executable), "executable_resolved": str(executable.resolve()),
            "executable_sha256": _sha(executable.resolve()), "python_version": sys.version,
            "environment": execution_environment(str(executable), os.environ)["environment"]}}
    return validate_runtime_context(value)


def validate_runtime_context(value):
    if (not isinstance(value, Mapping)
            or value.get("schema") != "onnx-splitpoint/generic-quality-runtime-context"
            or type(value.get("schema_version")) is not int or value["schema_version"] != 1):
        raise ValueError("Generic completed Quality observed runtime context invalid")
    boundary = value.get("boundary_runtime")
    if (not isinstance(boundary, Mapping) or set(boundary) != {
            "part1_io", "part2_io", "part2_to_part1", "passthrough_inputs", "transfer_contract"}
            or any(not isinstance(boundary[key], Mapping) for key in boundary)
            or not boundary["part1_io"] or not boundary["part2_io"]
            or not boundary["part2_to_part1"] or not boundary["transfer_contract"]
            or any(not isinstance(key, str) or not key or not isinstance(item, str) or not item
                   for role in ("part2_to_part1", "passthrough_inputs")
                   for key, item in boundary[role].items())):
        raise ValueError("Generic completed Quality actual resolved boundary missing")
    environment = value.get("environment")
    if (not isinstance(environment, Mapping)
            or any(not isinstance(environment.get(key), str) or not Path(environment[key]).is_absolute()
                   for key in ("executable", "executable_resolved"))
            or not _digest(environment.get("executable_sha256"))
            or not isinstance(environment.get("python_version"), str) or not environment["python_version"]
            or not isinstance(environment.get("environment"), Mapping)
            or any(not isinstance(key, str) or not isinstance(item, str)
                   for key, item in environment["environment"].items())):
        raise ValueError("Generic completed Quality actual interpreter/environment missing")
    return deepcopy(dict(value))


def _product():
    if __package__ == "splitpoint_runners":
        from . import native_detection_postprocess
    else:
        from .. import native_detection_postprocess
    return native_detection_postprocess


def _three_stage():
    if __package__ == "splitpoint_runners":
        from . import native_three_stage
    else:
        from .. import native_three_stage
    return native_three_stage


def _sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""):
            value.update(block)
    return value.hexdigest()


def _hash(value):
    return _product().canonical_json_sha256(value)


def _digest(value):
    return isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def _descriptor(value, *, check_files):
    if (not isinstance(value, Mapping) or not _digest(value.get("sha256"))
            or not isinstance(value.get("path"), str) or not Path(value["path"]).is_absolute()):
        raise ValueError("Generic completed Quality file descriptor invalid")
    if "size_bytes" in value and (type(value["size_bytes"]) is not int or value["size_bytes"] < 1):
        raise ValueError("Generic completed Quality artifact size invalid")
    if check_files and (_sha(value["path"]) != value["sha256"] or
            ("size_bytes" in value and Path(value["path"]).stat().st_size != value["size_bytes"])):
        raise ValueError("Generic completed Quality source/artifact bytes changed")
    return dict(value)


def _sources(rows, *, check_files):
    if not isinstance(rows, list) or not rows:
        raise ValueError("Generic completed Quality source closure missing")
    result = {}
    for row in rows:
        checked = _descriptor(row, check_files=check_files)
        if checked["path"] in result:
            raise ValueError("Generic completed Quality source closure duplicated")
        result[checked["path"]] = checked["sha256"]
    return result


def _artifact_identity(artifacts):
    if (not isinstance(artifacts, Mapping) or set(artifacts) != ARTIFACT_ROLES
            or any(not isinstance(value, Mapping) or not _digest(value.get("sha256"))
                   or type(value.get("size_bytes")) is not int or value["size_bytes"] < 1
                   for value in artifacts.values())):
        raise ValueError("Generic completed Quality compiled/source artifact binding missing")
    return {role: {key: value[key] for key in ("sha256", "size_bytes") if key in value}
            for role, value in artifacts.items()}


def _processor_policy(value):
    policy = deepcopy(value)
    for key in ("original_wh", "letterbox_geometry_contract", "letterbox_geometry_contract_sha256",
                "contract_sha256", "invariant_contract_sha256"):
        policy.pop(key, None)
    # Implementation artifacts deliberately remain exact, including nested
    # invariant identity. Only normally derived per-image geometry may vary.
    return policy


def _validate_preprocessing(contract, input_hw):
    expected = {
        "schema": "onnx-splitpoint/image-preprocessing-contract", "schema_version": 2,
        "contract_scope": "prepared_rgb_uint8_semantics", "task": "detection",
        "preprocess_mode": "letterbox", "spatial_transform": "centered_letterbox",
        "target_hw": list(input_hw), "color_space": "RGB", "input_domain": "uint8_0_255",
        "resize_interpolation": "bilinear", "resize_rounding": "python_round_ties_to_even",
        "placement": "floor_top_left_remainder_bottom_right", "pad_value": 114,
        "letterbox_pad_value": 114, "image_scale": "norm", "letterbox": True,
    }
    if contract != expected:
        raise ValueError("Generic completed Quality actual image preprocessing contract differs")
    return deepcopy(contract)


def _present_detections(detections, label_names):
    """Add only normal Quality presentation names to attested numeric records."""
    if (not isinstance(label_names, list)
            or any(not isinstance(name, str) or not name for name in label_names)):
        raise ValueError("Generic completed Quality presentation label map invalid")
    records = []
    for detection in detections:
        class_id = detection.get("class_id")
        if type(class_id) is not int or class_id < 0 or "class_name" in detection:
            raise ValueError("Generic completed Quality attested class ID invalid")
        name = label_names[class_id] if class_id < len(label_names) else str(class_id)
        records.append({**deepcopy(detection), "class_name": name})
    return records


def _verify_contract(value):
    try:
        return _product().verify_detection_completion_execution_contract(value)
    except (RuntimeError, ValueError, KeyError, TypeError) as error:
        raise ValueError("Generic completed Quality completion contract invalid") from error


def _verify_source_projection(base, full_source):
    source = base["source_endpoint"]
    if (not isinstance(full_source, Mapping)
            or full_source.get("endpoint_contract_hash") != source["endpoint_contract_hash"]
            or full_source.get("tensor_signature") != source["tensor_signature"]
            or any(full_source.get(key) != source[key] for key in ("task", "stage", "contract_family"))
            or full_source.get("endpoint_contract_complete") is not True
            or _hash(full_source.get("output_endpoint_attestation")) != source["output_endpoint_attestation_sha256"]):
        raise ValueError("Generic completed Quality full source endpoint missing/different")


def _same_policy(base, candidate):
    product = _product()
    original = _verify_contract(base)
    derived = _verify_contract(candidate)
    for key in ("model_id", "model_family", "completion_mode", "execution_policy",
                "performance_energy_contract_identity", "source_endpoint", "comparison_endpoint_contract",
                "yolov7_head_mapping", "artifact_hash_policy", "content_hash_policy",
                "invocation_hash_policy", "observation_relation_policy"):
        if derived[key] != original[key]:
            raise ValueError("Generic completed Quality completion policy changed: " + key)
    if _processor_policy(derived["processor_contract"]) != _processor_policy(original["processor_contract"]):
        raise ValueError("Generic completed Quality decoder/normalizer implementation changed")
    wh = derived["processor_contract"]["original_wh"]
    if len(wh) != 2 or any(type(value) is not int or value <= 0 for value in wh):
        raise ValueError("Generic completed Quality original image geometry invalid")
    geometry = derived["processor_contract"].get("letterbox_geometry_contract")
    if geometry is not None and geometry != product.build_letterbox_geometry_contract(
            input_hw=derived["processor_contract"]["input_hw"], original_wh=wh,
            preprocess={"mode": "letterbox", "rgb": True, "pad_value": 114}):
        raise ValueError("Generic completed Quality letterbox geometry changed")
    return derived


def load_quality_completion_config(path, *, check_files=True):
    """Validate a preparatory Quality config without requiring admitted Quality."""
    config_path = Path(path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if (not isinstance(config, Mapping) or config.get("schema") != CONFIG_SCHEMA
            or type(config.get("schema_version")) is not int or config["schema_version"] != 1):
        raise ValueError("Generic completed Quality explicit config schema invalid")
    if any(key in config for key in ("native_source", "native_uid", "native_command_contract",
            "native_command_contract_sha256", "native_quality_binding_sha256", "native_split_quality_binding")):
        raise ValueError("Generic completed Quality cannot claim Native authority")
    original = config.get("original_run_cfg") or {}
    if not isinstance(original, Mapping) or original.get("native_split_quality_binding"):
        raise ValueError("Generic completed Quality cannot carry a Native split binding")
    path_root = config.get("path_root")
    if path_root not in (None, "suite_root"):
        raise ValueError("Generic completed Quality relative path root invalid")
    root = config_path.parent.parent if path_root == "suite_root" else config_path.parent

    def resolve(descriptor):
        if not isinstance(descriptor, Mapping):
            raise ValueError("Generic completed Quality descriptor missing")
        resolved = dict(descriptor); relative = Path(str(resolved.get("path") or ""))
        if not relative.is_absolute():
            target = (root/relative).resolve()
            if not relative.parts or ".." in relative.parts or not target.is_relative_to(root):
                raise ValueError("Generic completed Quality relative path escapes suite")
            resolved["path"] = str(target)
        return resolved

    if isinstance(config.get("expected_runtime_artifacts"), Mapping):
        config["expected_runtime_artifacts"] = {key: resolve(value) for key, value in config["expected_runtime_artifacts"].items()}
    if isinstance(config.get("runtime_source_files"), list):
        config["runtime_source_files"] = [resolve(value) for value in config["runtime_source_files"]]
    alias = config.get("generic_artifact_binding")
    if isinstance(alias, Mapping) and isinstance(alias.get("artifacts"), Mapping):
        config["generic_artifact_binding"] = {**alias, "artifacts": {
            key: resolve(value) for key, value in alias["artifacts"].items()}}
    representative = config.get("representative_input")
    if representative is not None:
        if not isinstance(representative, Mapping):
            raise ValueError("Generic completed Quality representative input invalid")
        config["representative_input"] = {**representative, "prepared_input": resolve(representative.get("prepared_input"))}
        _descriptor(config["representative_input"]["prepared_input"], check_files=check_files)
    identity = config.get("identity")
    if (not isinstance(identity, Mapping) or any(not str(identity.get(key) or "").strip()
            for key in ("model_id", "case_id", "setup_id", "precision"))):
        raise ValueError("Generic completed Quality identity missing")
    base = _verify_contract(config.get("completion_contract"))
    if base["model_id"] != identity["model_id"]:
        raise ValueError("Generic completed Quality model differs from completion")
    full_source = config.get("source_endpoint_contract")
    _verify_source_projection(base, full_source)
    if config.get("preprocess") != {"mode": "letterbox", "rgb": True, "pad_value": 114}:
        raise ValueError("Generic completed Quality requires the frozen letterbox input policy")
    artifacts = config.get("expected_runtime_artifacts")
    _artifact_identity(artifacts)
    for descriptor in artifacts.values():
        _descriptor(descriptor, check_files=check_files)
    _endpoint.verify_generic_quality_endpoint_projection(config, check_files=check_files)
    alias = config.get("generic_artifact_binding")
    if alias is not None and (not isinstance(alias, Mapping) or alias.get("artifacts") != artifacts):
        raise ValueError("Generic completed Quality artifact binding aliases differ")
    _sources(config.get("runtime_source_files"), check_files=check_files)
    ids = config.get("expected_image_ids")
    if (not isinstance(ids, list) or not ids or any(not isinstance(value, str) or not value for value in ids)
            or ids != sorted(set(ids))):
        raise ValueError("Generic completed Quality frozen image IDs invalid")
    return dict(config)


def _verify_runtime_producer(config, context):
    if not isinstance(context, Mapping):
        raise ValueError("Generic completed Quality actual composed producer missing")
    for key in ("model_id", "case_id", "setup_id"):
        if context.get(key) != config["identity"][key]:
            raise ValueError("Generic completed Quality actual runtime identity differs: " + key)
    actual = {item["role"]: item for item in context.get("runtime_artifacts", [])}
    expected = config["expected_runtime_artifacts"]
    for actual_role, expected_role in (("stage1_runtime", "part1_runtime"), ("stage2_runtime", "engine")):
        if _sha(actual[actual_role]["path"]) != expected[expected_role]["sha256"]:
            raise ValueError("Generic completed Quality actual runtime artifact differs: " + actual_role)
    for context_key, expected_role in (("part1_model_path", "source_part1_onnx"), ("part2_model_path", "source_part2_onnx")):
        if _sha(context[context_key]) != expected[expected_role]["sha256"]:
            raise ValueError("Generic completed Quality actual split source differs")
    if config.get("source_endpoint_projection") is not None:
        if not context.get("source_model_path"):
            raise ValueError("Generic completed Quality actual full source model missing")
        _endpoint.verify_generic_quality_endpoint_projection(config, check_files=True,
            full_model_path=context["source_model_path"])


def _source_implementation_check(source_map, *, runner_source=None):
    required = [Path(__file__).resolve(), Path(_product().__file__).resolve(),
                Path(_three_stage().__file__).resolve(), Path(_product()._yolo_module.__file__).resolve(),
                Path(_endpoint.__file__).resolve()]
    if runner_source is not None:
        required.append(Path(runner_source).resolve())
    for path in required:
        if source_map.get(str(path)) != _sha(path):
            raise ValueError("Generic completed Quality actual implementation outside frozen source closure: " + path.name)


class GenericQualityCompletion:
    """Consume real serial P2 outputs with one ordinary auto factory per image."""

    def __init__(self, config_path, *, producer_context, runner_source, input_hw,
                 expected_image_ids, runtime_precision_identity, confidence_threshold,
                 iou_threshold, max_detections, letterbox, preprocessing_contract,
                 class_names):
        self.config = load_quality_completion_config(config_path, check_files=True)
        self.config_file = {"path": str(Path(config_path).resolve()), "sha256": _sha(config_path)}
        self.sources_before = _sources(self.config["runtime_source_files"], check_files=True)
        _source_implementation_check(self.sources_before, runner_source=runner_source)
        _verify_runtime_producer(self.config, producer_context)
        self.producer_context = deepcopy(producer_context)
        self.runner_source = str(runner_source)
        self.runner_descriptor = {"path": str(Path(runner_source).resolve()),
            "sha256": _sha(runner_source), "size_bytes": Path(runner_source).stat().st_size}
        self.preprocessing_contract = _validate_preprocessing(preprocessing_contract, input_hw)
        self.class_names = list(class_names)
        _present_detections([], self.class_names)
        processor = self.config["completion_contract"]["processor_contract"]
        if (list(input_hw) != processor["input_hw"] or not letterbox
                or runtime_precision_identity != self.config["identity"]["precision"]
                or float(confidence_threshold) != processor["confidence_threshold"]
                or float(iou_threshold) != processor["iou_threshold"]
                or int(max_detections) != processor["max_detections"]
                or sorted(expected_image_ids) != self.config["expected_image_ids"]):
            raise ValueError("Generic completed Quality actual preprocessing/threshold/dataset policy differs")
        self.entries = {}
        self._runtime_inputs = None
        self._runtime_input_snapshot = None
        self._prepared_input_snapshot = None
        self._boundary_runtime = None
        self._boundary_observations = 0
        self._logical_boundary = None
        self._transfer_observations = 0

    @contextmanager
    def observe_boundary_transfer(self, transfer):
        """Forward the unchanged transfer once, retaining only actual array specs."""
        calls = 0

        def specs(tensors):
            if not isinstance(tensors, Mapping) or not tensors:
                raise ValueError("Generic completed Quality actual boundary arrays missing")
            result = {}
            for name, value in tensors.items():
                if (not isinstance(name, str) or not name or not isinstance(value, np.ndarray)
                        or value.dtype.kind not in "biuf" or not value.size):
                    raise ValueError("Generic completed Quality actual boundary array invalid")
                result[name] = {"shape": list(value.shape), "dtype": str(value.dtype)}
            return result

        def observed(tensors, *args, **kwargs):
            nonlocal calls
            calls += 1
            if calls != 1:
                raise ValueError("Generic completed Quality requires one actual transfer per image")
            part1 = specs(tensors)
            outputs = transfer(tensors, *args, **kwargs)
            boundary = {"part1_outputs": part1, "part2_inputs": specs(outputs)}
            if self._logical_boundary is None:
                self._logical_boundary = boundary
            elif boundary != self._logical_boundary:
                raise ValueError("Generic completed Quality actual boundary shapes/dtypes changed")
            self._transfer_observations += 1
            return outputs

        yield observed
        if calls != 1:
            raise ValueError("Generic completed Quality actual transfer was not observed")

    def observe_boundary_runtime(self, boundary_runtime):
        """Require the same live transfer metadata after every completed image."""
        if self._boundary_runtime is None:
            self._boundary_runtime = deepcopy(boundary_runtime)
        elif boundary_runtime != self._boundary_runtime:
            raise ValueError("Generic completed Quality resolved boundary changed between images")
        self._boundary_observations += 1

    @contextmanager
    def observe_stage1_inputs(self, session, *, prepared_inputs=None):
        """Observe the actual adapter result, forwarding the bound method once.

        Only the explicit serial dataset loop enters this scope. The original
        SDK preparation/infer implementation remains unchanged and is restored
        even when inference or subsequent completion fails.
        """
        original = getattr(session, "infer", None)
        if not callable(original):
            raise ValueError("Generic completed Quality actual stage1 observer unavailable")
        calls = 0
        self._runtime_inputs = None
        self._runtime_input_snapshot = None
        self._prepared_input_snapshot = _input_snapshot(prepared_inputs) if prepared_inputs is not None else None

        def observed(inputs, *args, **kwargs):
            nonlocal calls
            calls += 1
            if calls != 1 or not isinstance(inputs, Mapping) or not inputs:
                raise ValueError("Generic completed Quality requires exactly one actual stage1 inference per image")
            self._runtime_inputs = {str(name): np.array(value, copy=True, order="K")
                                    for name, value in inputs.items()}
            self._runtime_input_snapshot = _input_snapshot(inputs)
            return original(inputs, *args, **kwargs)

        session.infer = observed
        try:
            yield
            if calls != 1:
                raise ValueError("Generic completed Quality stage1 inference was not observed")
        finally:
            session.infer = original

    def process(self, outputs, *, image_path, original_wh, prepared_inputs, runtime_inputs=None):
        product = _product(); image = Path(image_path); image_id = image.name
        if image_id not in self.config["expected_image_ids"] or image_id in self.entries:
            raise ValueError("Generic completed Quality image missing/duplicate/outside frozen dataset")
        from PIL import Image
        with Image.open(image) as source:
            if list(source.size) != list(original_wh):
                raise ValueError("Generic completed Quality geometry differs from the actual source image")
        base = self.config["completion_contract"]; policy = base["processor_contract"]
        names = [item["name"] for item in base["source_endpoint"]["tensor_signature"]["tensors"]]
        if set(outputs) != set(names):
            raise ValueError("Generic completed Quality actual P2 output names changed")
        ordered = {name: np.asarray(outputs[name]) for name in names}
        contract = product.build_detection_completion_execution_contract(model_id=base["model_id"],
            outputs=ordered, input_hw=policy["input_hw"], original_wh=original_wh,
            source_endpoint_contract=self.config["source_endpoint_contract"], preprocess=self.config["preprocess"],
            confidence_threshold=policy["confidence_threshold"], iou_threshold=policy["iou_threshold"],
            max_detections=policy["max_detections"])
        _same_policy(base, contract)
        image_sha = _sha(image)
        runtime = product.create_detection_completion_runtime(contract, source_paths=(self.runner_source, __file__),
            sentinel_context={"image_id": image_id, "source_image_sha256": image_sha})
        runtime.process(ordered)
        if runtime.completed_count != 1:
            raise ValueError("Generic completed Quality image did not complete exactly once")
        attestation = runtime.attestation(completed_work_units=1)
        detections = deepcopy(runtime.last_detections)
        if detections != attestation["last_result"]["detections"]:
            raise ValueError("Generic completed Quality observed result differs from its normal oracle")
        product.detection_completion_runtime_source_check(runtime)
        if runtime_inputs is None:
            runtime_inputs = self._runtime_inputs
            runtime_snapshot = self._runtime_input_snapshot
        else:
            runtime_snapshot = _input_snapshot(runtime_inputs)
        if not isinstance(runtime_inputs, Mapping) or not runtime_inputs:
            raise ValueError("Generic completed Quality actual post-adapter input evidence missing")
        representative = self.config.get("representative_input")
        representative_proof = None
        if representative is not None and representative.get("image_id") == image_id:
            from .task_completion import load_completion_bound_array
            expected = load_completion_bound_array(representative["prepared_input"], file_sha256=_sha)
            name = representative.get("runtime_input_name")
            actual = runtime_inputs.get(name) if name else next(iter(runtime_inputs.values())) if len(runtime_inputs) == 1 else None
            if (actual is None or actual.dtype != expected.dtype or actual.shape != expected.shape
                    or not np.array_equal(actual, expected) or representative.get("source_image_sha256") != image_sha
                    or representative.get("original_wh") != list(original_wh)):
                raise ValueError("Generic completed Quality representative actual runtime input differs from timed prepared bytes")
            representative_proof = {**deepcopy(representative), "actual_runtime_input": runtime_snapshot,
                                    "prepared_tensor": _input_snapshot({"canonical": expected})["canonical"]}
        entry = {"image_id": image_id, "source_image_sha256": image_sha, "original_wh": list(original_wh),
                 "completion_contract": contract, "runtime_mode": runtime.completion_runtime_mode,
                 "attestation": attestation, "detections_sha256": _hash(detections),
                 "prepared_inputs": self._prepared_input_snapshot or _input_snapshot(prepared_inputs), "runtime_inputs": runtime_snapshot,
                 "representative_input_comparison": representative_proof}
        self.entries[image_id] = entry
        return _present_detections(detections, self.class_names)

    def evidence(self, candidate_by_image, *, runtime_context=None):
        ids = self.config["expected_image_ids"]
        if sorted(self.entries) != ids or sorted(candidate_by_image) != ids:
            raise ValueError("Generic completed Quality dataset did not complete every frozen image")
        after = _sources(self.config["runtime_source_files"], check_files=True)
        if after != self.sources_before or _sha(self.config_file["path"]) != self.config_file["sha256"]:
            raise ValueError("Generic completed Quality source/config changed during dataset execution")
        _verify_runtime_producer(self.config, self.producer_context)
        evidence = {"schema": EVIDENCE_SCHEMA, "schema_version": 1,
            "execution": "serial_existing_compiled_parts_no_performance_measurement",
            "completion_factory": "create_detection_completion_runtime:auto",
            "identity": self.config["identity"], "base_completion_contract": self.config["completion_contract"],
            "source_endpoint_contract": self.config["source_endpoint_contract"], "preprocess": self.config["preprocess"],
            "image_preprocessing_contract": self.preprocessing_contract,
            "image_preprocessing_contract_sha256": _hash(self.preprocessing_contract),
            "presentation_labels": {"class_names": self.class_names, "fallback": "decimal_class_id"},
            "runtime_artifacts": self.config["expected_runtime_artifacts"], "config_file": self.config_file,
            "implementation_runner": self.runner_descriptor,
            "runtime_source_files": self.config["runtime_source_files"],
            "source_binding": {"before": self.sources_before, "after": after},
            "image_ids": ids, "image_count": len(ids), "per_image": [self.entries[key] for key in ids],
            "representative_input": self.config.get("representative_input")}
        if runtime_context is not None:
            evidence["runtime_context"] = validate_runtime_context(runtime_context)
            if (runtime_context["boundary_runtime"] != self._boundary_runtime
                    or self._boundary_observations != len(ids)
                    or self._transfer_observations != len(ids)):
                raise ValueError("Generic completed Quality boundary was not observed for every image")
            evidence["runtime_context"]["boundary_observation_count"] = self._boundary_observations
            evidence["runtime_context"]["logical_boundary"] = deepcopy(self._logical_boundary)
        for key in ("source_endpoint_authority", "source_endpoint_projection"):
            if key in self.config:
                evidence[key] = deepcopy(self.config[key])
        evidence["evidence_sha256"] = _hash(evidence)
        # The exporter binds complete records including ground truth separately.
        return evidence


def bind_completed_quality_producer(producer_identity, completion_contract, evidence):
    """Keep the normal Generic producer schema, naming its actual completion."""
    producer = deepcopy(producer_identity); completion = deepcopy(completion_contract)
    if producer.get("execution_role") != "generic_composed_quality_candidate":
        raise ValueError("Generic completed Quality producer role differs")
    completion.pop("contract_sha256", None)
    completion.update(completion_mode="stage1_stage2_then_frozen_detection_completion_per_image",
        frozen_completion_execution_contract_sha256=evidence["base_completion_contract"]["contract_sha256"],
        completed_task_comparison_endpoint_contract_hash=evidence["base_completion_contract"]["comparison_endpoint_contract"]["endpoint_contract_hash"],
        completion_factory="create_detection_completion_runtime:auto")
    completion["contract_sha256"] = _hash(completion)
    producer.pop("producer_identity_sha256", None)
    producer.update(candidate_execution_completion_contract=completion,
        candidate_execution_completion_contract_sha256=completion["contract_sha256"])
    producer["producer_identity_sha256"] = _hash(producer)
    return producer, completion


def completed_quality_export_fields(evidence, candidate_records, producer_identity):
    """Bind actual completed records into both ordinary portable envelopes."""
    proof = deepcopy(evidence)
    _validate_evidence(proof, candidate_records, proof["base_completion_contract"], proof["runtime_artifacts"])
    producer = producer_identity or {}
    _validate_producer(proof, producer)
    proof.pop("evidence_sha256")
    proof["candidate_records_sha256"] = _hash(candidate_records)
    proof["producer_identity_sha256"] = producer["producer_identity_sha256"]
    proof["evidence_sha256"] = _hash(proof)
    return {FIELD: proof}


def _validate_producer(proof, producer):
    runner = proof.get("implementation_runner") or {}
    if (producer.get("implementation_runner_sha256") != runner.get("sha256")
            or producer.get("implementation_runner_size_bytes") != runner.get("size_bytes")):
        raise ValueError("Generic completed Quality exporter runner differs from actual captured source")
    for key in ("model_id", "case_id", "setup_id"):
        if producer.get(key) != proof["identity"][key]:
            raise ValueError("Generic completed Quality exporter producer differs")
    if producer.get("runtime_precision_identity") != proof["identity"]["precision"]:
        raise ValueError("Generic completed Quality exporter precision differs")
    actual = {item["role"]: item for item in producer.get("runtime_artifacts", [])}
    for producer_role, artifact_role in (("stage1_runtime", "part1_runtime"), ("stage2_runtime", "engine")):
        if actual.get(producer_role, {}).get("sha256") != proof["runtime_artifacts"][artifact_role]["sha256"]:
            raise ValueError("Generic completed Quality exporter compiled artifact differs")
    for producer_field, artifact_role in (("part1_onnx_sha256", "source_part1_onnx"), ("part2_onnx_sha256", "source_part2_onnx")):
        if producer.get("model", {}).get(producer_field) != proof["runtime_artifacts"][artifact_role]["sha256"]:
            raise ValueError("Generic completed Quality exporter split source differs")
    projection = proof.get("source_endpoint_projection")
    if (projection is not None
            and producer.get("model", {}).get("source_onnx_sha256") != projection["full_model"]["sha256"]):
        raise ValueError("Generic completed Quality exporter full source differs")
    identity = dict(producer); seal = identity.pop("producer_identity_sha256", "")
    if _hash(identity) != seal:
        raise ValueError("Generic completed Quality producer identity hash differs")


def _input_snapshot(inputs):
    if not isinstance(inputs, Mapping) or not inputs:
        raise ValueError("Generic completed Quality input map missing")
    result = {}
    for name, value in inputs.items():
        array = np.asarray(value)
        if array.dtype.kind not in "biuf" or not array.size:
            raise ValueError("Generic completed Quality actual input dtype/shape invalid")
        result[str(name)] = {"shape": list(array.shape), "dtype": str(array.dtype),
            "strides": list(array.strides), "c_contiguous": bool(array.flags.c_contiguous),
            "tensor_sha256": hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()}
    return result


def _validate_input_snapshot(value):
    if not isinstance(value, Mapping) or not value:
        raise ValueError("Generic completed Quality actual input snapshot missing")
    for record in value.values():
        if (not isinstance(record, Mapping) or not _digest(record.get("tensor_sha256"))
                or not isinstance(record.get("shape"), list) or not record["shape"]
                or any(type(dim) is not int or dim <= 0 for dim in record["shape"])
                or len(record.get("strides") or []) != len(record["shape"])
                or any(type(stride) is not int for stride in record["strides"])
                or type(record.get("c_contiguous")) is not bool
                or not isinstance(record.get("dtype"), str)
                or np.dtype(record.get("dtype")).kind not in "biuf"):
            raise ValueError("Generic completed Quality actual input snapshot invalid")


def _validate_evidence(evidence, records, completion_contract, expected_runtime_artifacts):
    if (not isinstance(evidence, Mapping) or evidence.get("schema") != EVIDENCE_SCHEMA
            or type(evidence.get("schema_version")) is not int or evidence["schema_version"] != 1):
        raise ValueError("Generic completed Quality actual completion evidence missing")
    unhashed = dict(evidence); seal = unhashed.pop("evidence_sha256", None)
    if _hash(unhashed) != seal:
        raise ValueError("Generic completed Quality completion evidence hash differs")
    if "runtime_context" in evidence:
        validate_runtime_context(evidence["runtime_context"])
    base = _verify_contract(completion_contract)
    _endpoint.verify_generic_quality_endpoint_projection({**evidence,
        "expected_runtime_artifacts": evidence.get("runtime_artifacts")}, check_files=False)
    preprocessing = _validate_preprocessing(evidence.get("image_preprocessing_contract"),
                                           base["processor_contract"]["input_hw"])
    presentation = evidence.get("presentation_labels")
    if (evidence.get("image_preprocessing_contract_sha256") != _hash(preprocessing)
            or not isinstance(presentation, Mapping)
            or set(presentation) != {"class_names", "fallback"}
            or presentation.get("fallback") != "decimal_class_id"):
        raise ValueError("Generic completed Quality actual preprocessing/presentation binding invalid")
    _present_detections([], presentation.get("class_names"))
    _verify_source_projection(base, evidence.get("source_endpoint_contract"))
    _descriptor(evidence.get("config_file"), check_files=False)
    if (evidence.get("base_completion_contract") != base
            or evidence.get("completion_factory") != "create_detection_completion_runtime:auto"
            or evidence.get("execution") != "serial_existing_compiled_parts_no_performance_measurement"
            or evidence.get("preprocess") != {"mode": "letterbox", "rgb": True, "pad_value": 114}
            or _artifact_identity(evidence.get("runtime_artifacts")) != _artifact_identity(expected_runtime_artifacts)):
        raise ValueError("Generic completed Quality frozen base policy/artifact binding differs")
    sources = _sources(evidence.get("runtime_source_files"), check_files=False)
    if evidence.get("source_binding") != {"before": sources, "after": sources}:
        raise ValueError("Generic completed Quality source closure was not preserved")
    runner = _descriptor(evidence.get("implementation_runner"), check_files=False)
    if (type(runner.get("size_bytes")) is not int or runner["size_bytes"] < 1
            or sources.get(runner["path"]) != runner["sha256"]):
        raise ValueError("Generic completed Quality actual runner missing from source closure")
    # Controller/source aliases may differ from deployed paths; implementation
    # bytes must still match the actual verifier modules, not just their names.
    for module in (__file__, _product().__file__, _three_stage().__file__, _product()._yolo_module.__file__, _endpoint.__file__):
        if not any(Path(path).name == Path(module).name and digest == _sha(module) for path, digest in sources.items()):
            raise ValueError("Generic completed Quality implementation source differs")
    ids = evidence.get("image_ids"); per_image = evidence.get("per_image")
    if ("runtime_context" in evidence
            and (type(evidence["runtime_context"].get("boundary_observation_count")) is not int
                 or evidence["runtime_context"]["boundary_observation_count"] != evidence.get("image_count")
                 or not isinstance(evidence["runtime_context"].get("logical_boundary"), Mapping))):
        raise ValueError("Generic completed Quality boundary observation count differs")
    if (not isinstance(ids, list) or ids != sorted(set(ids)) or not ids
            or type(evidence.get("image_count")) is not int or evidence["image_count"] != len(ids)
            or not isinstance(per_image, list) or [entry.get("image_id") for entry in per_image] != ids
            or [record.get("image_id") for record in records] != ids):
        raise ValueError("Generic completed Quality frozen image count/order differs")
    default_mode = _product().create_detection_completion_runtime(base).completion_runtime_mode
    for entry, record in zip(per_image, records):
        contract = _same_policy(base, entry.get("completion_contract"))
        if (entry.get("original_wh") != contract["processor_contract"]["original_wh"]
                or not _digest(entry.get("source_image_sha256")) or not entry.get("prepared_inputs")
                or entry.get("runtime_mode") != default_mode):
            raise ValueError("Generic completed Quality image geometry/input/default runtime differs")
        _validate_input_snapshot(entry.get("prepared_inputs"))
        _validate_input_snapshot(entry.get("runtime_inputs"))
        try:
            if default_mode == "fast_oracle_outside_timing":
                attestation = _three_stage().verify_fast_completion_attestation(entry.get("attestation"), execution_contract=contract)
                sentinel = attestation.get("sentinel_identity") or {}
                if any(sentinel.get(key) != entry[key] for key in ("image_id", "source_image_sha256")):
                    raise ValueError("Generic completed Quality image/source association differs")
            else:
                attestation = _product().verify_detection_completion_execution_attestation(
                    entry.get("attestation"), execution_contract=contract, expected_observation_relation="same_hotloop_sentinel")
        except (RuntimeError, ValueError, KeyError, TypeError) as error:
            raise ValueError("Generic completed Quality per-image completion attestation invalid") from error
        detections = attestation["last_result"]["detections"]
        if (attestation["completed_work_units"] != 1
                or record.get("candidate") != _present_detections(detections, presentation["class_names"])
                or entry.get("detections_sha256") != _hash(detections)):
            raise ValueError("Generic completed Quality exported detections differ from actual completed result")
    representative = evidence.get("representative_input")
    if representative is not None:
        matched = [entry for entry in per_image if entry["image_id"] == representative.get("image_id")]
        if len(matched) != 1:
            raise ValueError("Generic completed Quality representative image not completed")
        actual = matched[0]
        comparison = actual.get("representative_input_comparison") or {}
        expected_tensor = comparison.get("prepared_tensor")
        name = representative.get("runtime_input_name")
        observed_tensor = (actual["runtime_inputs"].get(name) if name else
                           next(iter(actual["runtime_inputs"].values())) if len(actual["runtime_inputs"]) == 1 else None)
        if (comparison != {**representative, "actual_runtime_input": actual["runtime_inputs"], "prepared_tensor": expected_tensor}
                or actual["source_image_sha256"] != representative.get("source_image_sha256")
                or actual["original_wh"] != representative.get("original_wh") or observed_tensor is None
                or not isinstance(expected_tensor, Mapping)
                or any(observed_tensor[key] != expected_tensor.get(key) for key in ("shape", "dtype", "tensor_sha256"))):
            raise ValueError("Generic completed Quality representative input proof changed")
    if "candidate_records_sha256" in evidence and evidence["candidate_records_sha256"] != _hash(records):
        raise ValueError("Generic completed Quality complete candidate records changed")
    return dict(evidence)


def validate_completed_quality_binding(request, result, completion_contract, expected_runtime_artifacts,
                                     *, request_path, reference_descriptor):
    """Verify actual normal transported completion evidence before admission."""
    request_path = Path(request_path).resolve(); descriptor = request.get("candidate") or {}
    if json.loads(request_path.read_text(encoding="utf-8")) != request:
        raise ValueError("Generic completed Quality supplied request differs from its file")
    relative = Path(str(descriptor.get("path") or ""))
    candidate_path = (request_path.parent/relative).resolve()
    if (not relative.parts or relative.is_absolute() or not candidate_path.is_relative_to(request_path.parent)
            or _sha(candidate_path) != descriptor.get("sha256")):
        raise ValueError("Generic completed Quality candidate file binding invalid")
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    evidence = request.get(FIELD)
    if evidence != candidate.get(FIELD):
        raise ValueError("Generic completed Quality request/candidate completion evidence differs")
    proof = _validate_evidence(evidence, candidate.get("records") or [], completion_contract, expected_runtime_artifacts)
    producer = request.get("producer_identity") or {}
    _validate_producer(proof, producer)
    completion = producer.get("candidate_execution_completion_contract") or {}
    if (result.get("source_request_sha256", "").removeprefix("sha256:") != _sha(request_path)
            or result.get("status") != "completed" or result.get("technical_status") != "completed"
            or result.get("task") != "detection" or result.get("variant") != "composed"
            or request.get("task") != "detection" or request.get("variant") != "composed"
            or candidate.get("task") != "detection" or candidate.get("variant") != "composed"
            or candidate.get("producer_identity") != producer
            or result.get("producer_identity_sha256") != proof.get("producer_identity_sha256")
            or request.get("expected_image_ids") != proof["image_ids"]
            or request.get("record_count") != proof["image_count"]
            or candidate.get("record_count") != proof["image_count"]
            or reference_descriptor.get("expected_image_ids", proof["image_ids"]) != proof["image_ids"]
            or producer.get("producer_identity_sha256") != proof.get("producer_identity_sha256")
            or completion.get("completion_mode") != "stage1_stage2_then_frozen_detection_completion_per_image"
            or completion.get("frozen_completion_execution_contract_sha256") != completion_contract["contract_sha256"]):
        raise ValueError("Generic completed Quality normal result/reference/image binding differs")
    return {"schema": EVIDENCE_SCHEMA, "evidence_sha256": proof["evidence_sha256"],
            "base_completion_contract_sha256": proof["base_completion_contract"]["contract_sha256"],
            "image_count": proof["image_count"], "completion_factory": proof["completion_factory"],
            "all_actual_completed_records_verified": True,
            "representative_input": deepcopy(proof.get("representative_input")),
            "representative_input_comparison": next((deepcopy(entry["representative_input_comparison"])
                for entry in proof["per_image"] if entry.get("representative_input_comparison") is not None), None)}
