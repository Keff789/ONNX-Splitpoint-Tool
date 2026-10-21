"""Bind optional mixed-input functional controls to a prepared Generic run.

Only existing prepared tensors are loaded. The caller supplies the initialized
product P1/P2 functions and its policy-bound adapter; this module owns no SDK,
transport, process, preprocessing, build, or scientific throughput measurement.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from .task_completion import load_completion_bound_array, normalize_stage1_execution


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _detection_module():
    # Generated bundles vendor this module beside native_detection_postprocess;
    # the repository keeps the canonical module one package level above it.
    if __package__ == "splitpoint_runners":
        from . import native_detection_postprocess
    else:
        from .. import native_detection_postprocess
    return native_detection_postprocess


def _processor_policy(processor):
    value = deepcopy(processor)
    for key in ("original_wh", "letterbox_geometry_contract", "letterbox_geometry_contract_sha256",
                "implementation_artifacts", "contract_sha256", "invariant_contract_sha256"):
        value.pop(key, None)
    if isinstance(value.get("invariant_identity"), Mapping):
        value["invariant_identity"].pop("implementation_artifacts", None)
    return value


def _detection_contract(control, base, metadata):
    product = _detection_module()
    verified = product.verify_detection_completion_execution_contract(control)
    original = product.verify_detection_completion_execution_contract(base)
    for key in ("model_id", "model_family", "completion_mode", "execution_policy",
                "performance_energy_contract_identity", "source_endpoint", "comparison_endpoint_contract",
                "yolov7_head_mapping", "artifact_hash_policy", "content_hash_policy",
                "invocation_hash_policy", "observation_relation_policy"):
        if verified[key] != original[key]:
            raise ValueError("Mixed detection control changed its completion semantics: " + key)
    processor = verified["processor_contract"]
    if _processor_policy(processor) != _processor_policy(original["processor_contract"]):
        raise ValueError("Mixed detection control changed its processor policy")
    original_wh = metadata.get("original_wh")
    if (not isinstance(original_wh, (list, tuple)) or len(original_wh) != 2
            or any(type(value) is not int or value <= 0 for value in original_wh)
            or processor["original_wh"] != list(original_wh)):
        raise ValueError("Mixed detection control geometry differs from its prepared input provenance")
    hw = processor["input_hw"]
    if metadata.get("shape") != [hw[0], hw[1], 3] or metadata.get("dtype") != "uint8":
        raise ValueError("Mixed detection control prepared tensor geometry contract mismatch")
    geometry = product.build_letterbox_geometry_contract(input_hw=hw, original_wh=original_wh,
        preprocess={"mode": "letterbox", "rgb": True, "pad_value": 114})
    if ("letterbox_geometry_contract" in processor
            and processor["letterbox_geometry_contract"] != geometry):
        raise ValueError("Mixed detection control derived geometry mismatch")
    declared = metadata.get("geometry")
    if declared is not None:
        expected = {"mode": "letterbox", "source_hw": [original_wh[1], original_wh[0]],
            "target_hw": hw, "resized_hw": list(reversed(geometry["resized_wh"])),
            "pad_ltrb": [geometry[key] for key in ("pad_left", "pad_top", "pad_right", "pad_bottom")],
            "pad_value": geometry["pad_value"]}
        if not isinstance(declared, Mapping) or any(declared.get(key) != value for key, value in expected.items()):
            raise ValueError("Mixed detection control prepared geometry mismatch")
    return verified


def _descriptor(value):
    return (isinstance(value, Mapping) and set(value) == {"path", "sha256"}
            and isinstance(value["path"], str) and Path(value["path"]).is_absolute()
            and isinstance(value["sha256"], str) and re.fullmatch(r"[a-f0-9]{64}", value["sha256"]) is not None)


def _source_map(config, *, check_files=False):
    sources = config.get("runtime_source_files")
    if not isinstance(sources, list) or not sources or any(not _descriptor(item) for item in sources):
        raise ValueError("Mixed functional control lacks its frozen runtime source closure")
    result = {item["path"]: item["sha256"] for item in sources}
    if len(result) != len(sources):
        raise ValueError("Mixed functional control runtime source paths are duplicated")
    if check_files:
        try:
            changed = any(_sha(path) != digest for path, digest in result.items())
        except OSError as error:
            raise ValueError("Mixed functional control runtime source missing") from error
        if changed:
            raise ValueError("Mixed functional control runtime sources changed")
    return result


def _input_sequence():
    # Both inputs visit every physical slot for the fixed four/eight-slot pools.
    return [(index % 2) ^ (index >= 16) for index in range(32)]


def load_bound_mixed_control(config):
    """Check the optional sidecar and tensor bytes before runtime construction."""
    if not isinstance(config, Mapping):
        raise ValueError("Mixed functional control runtime configuration malformed")
    descriptor = config.get("mixed_input_control")
    if descriptor is None:
        return None
    if config.get("energy_capture"):
        raise ValueError("Mixed functional control is outside energy captures")
    if not _descriptor(descriptor):
        raise ValueError("Mixed functional control requires a hash-bound sidecar")
    source_map = _source_map(config, check_files=True)
    if (source_map.get(descriptor["path"]) != descriptor["sha256"]
            or _sha(descriptor["path"]) != descriptor["sha256"]):
        raise ValueError("Mixed functional control differs from frozen runtime sources")
    payload = json.loads(Path(descriptor["path"]).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or config.get("task") not in {"classification", "detection"}:
        raise ValueError("Mixed functional control sidecar/task malformed")
    if config.get("boundary_observation") is not None or payload.get("boundary_observation") is not None:
        from .generic_mixed_control import normalize_boundary_observation
        contract = normalize_boundary_observation(config.get("boundary_observation"))
        if payload.get("boundary_observation") != contract:
            raise ValueError("Mixed boundary observation sidecar differs from its runtime contract")
    for key in ("identity", "execution_id", "expected_runtime_artifacts"):
        if not config.get(key) or payload.get(key) != config.get(key):
            raise ValueError("Mixed functional control belongs to another runtime: " + key)
    if (payload.get("task_count") != 32 or type(payload.get("task_count")) is not int
            or normalize_stage1_execution(payload.get("stage1_execution"))
            != normalize_stage1_execution(config.get("stage1_execution"))
            or not isinstance(payload.get("inputs"), list) or len(payload["inputs"]) != 2
            or not isinstance(payload.get("provenance"), Mapping) or not payload["provenance"]):
        raise ValueError("Mixed functional control task/policy/provenance contract invalid")
    if (("task_input_indices" in payload and payload["task_input_indices"] != _input_sequence())
            or ("task_sequence_policy" in payload
                and payload["task_sequence_policy"] != "first16_alternate_second16_invert_v1")):
        raise ValueError("Mixed functional control task sequence does not exercise slot reuse")
    expected = config["input_binding"]["prepared_input"]
    if config["task"] == "classification" and config.get("completion_contract") != {
            "task": "classification", "operation": "top1_top5", "label_offset": 0}:
        raise ValueError("Mixed classification control changed Top1/Top5 completion")
    tensors = []
    from .generic_passthrough import passthrough_sources
    canonical = config.get("canonical_full_input_bindings")
    if passthrough_sources(config) and (not isinstance(canonical, list) or len(canonical) != 2):
        raise ValueError("Mixed original-input passthrough lacks two bound canonical full inputs")
    for index, item in enumerate(payload["inputs"]):
        if not isinstance(item, Mapping) or not isinstance(item.get("prepared_input"), Mapping):
            raise ValueError("Mixed functional input descriptor malformed")
        spec = item["prepared_input"]
        if (not isinstance(spec.get("path"), str) or not Path(spec["path"]).is_absolute()
                or any(spec.get(key) != expected.get(key) for key in ("input_name", "shape", "dtype"))
                or source_map.get(spec["path"]) != spec["sha256"]
                or not isinstance(item.get("provenance"), Mapping) or not item["provenance"]):
            raise ValueError("Mixed functional input differs from its prepared contract/source closure")
        if index == 0 and any(spec.get(key) != expected.get(key) for key in
                              ("path", "sha256", "input_name", "shape", "dtype", "format")):
            raise ValueError("Mixed control input zero must preserve the unchanged baseline prepared input")
        tensors.append(load_completion_bound_array(spec, file_sha256=_sha))
        if passthrough_sources(config):
            binding = canonical[index]
            image = binding.get("source_image") or {}
            if (item.get("canonical_full_input_binding") != binding
                    or binding.get("prepared_input_sha256") != spec["sha256"]
                    or source_map.get(image.get("path")) != image.get("sha256")
                    or _sha(image["path"]) != image["sha256"]):
                raise ValueError("Mixed canonical original input differs from its task/source binding")
        if config["task"] == "detection":
            _detection_contract(item["completion_contract"], config["completion_contract"],
                {**spec, "original_wh": item.get("original_wh"), **({"geometry": item["geometry"]} if "geometry" in item else {})})
        elif item.get("completion_contract") not in (None, config["completion_contract"]):
            raise ValueError("Mixed classification control changed Top1/Top5 completion")
    if np.array_equal(tensors[0], tensors[1]):
        raise ValueError("Mixed control requires two distinguishable existing prepared inputs")
    return payload, tensors


def _report_value(value):
    """Retain tensor evidence compactly; no diagnostic FPS is published."""
    if isinstance(value, np.ndarray):
        return {"shape": list(value.shape), "dtype": str(value.dtype),
                "tensor_sha256": hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {key: _report_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_report_value(item) for item in value]
    return value


def _tensor_snapshot(value):
    if (not isinstance(value, Mapping) or set(value) != {"shape", "dtype", "tensor_sha256"}
            or not isinstance(value["shape"], list)
            or any(type(dim) is not int or dim <= 0 for dim in value["shape"])
            or not isinstance(value["dtype"], str)
            or not isinstance(value["tensor_sha256"], str)
            or re.fullmatch(r"[a-f0-9]{64}", value["tensor_sha256"]) is None):
        return False
    try:
        return np.dtype(value["dtype"]).kind in "biuf"
    except TypeError:
        return False


def run_bound_mixed_control(config, output, repetition, *, serial_stage1, transfer,
                            stage2, stage1_async, prepare_async_feed, clone_payload,
                            runner_source, boundary_runtime=None, canonical_full_feeds=None):
    """Run 32 serial references and 32 ordinary pipeline tasks before timing."""
    from .generic_mixed_control import run_mixed_control
    from .harness.classification import ClassificationCompletion

    if type(repetition) is not int or repetition not in (1, 2, 3):
        raise ValueError("Mixed control requires a bound repetition index 1-3")
    loaded = load_bound_mixed_control(config)
    if loaded is None:
        raise ValueError("Mixed control invoked without its explicit bound sidecar")
    payload, tensors = loaded
    boundary_contract = config.get("boundary_observation")
    if (boundary_contract is None) != (boundary_runtime is None):
        raise ValueError("Mixed boundary observation requires its explicit live runtime callback")
    sequence = _input_sequence()
    tasks = [{"task_id": f"task_{index:02d}", "prepared_input": payload["inputs"][sequence[index]]["prepared_input"],
              "feed": {config["input_binding"]["prepared_input"]["input_name"]: tensors[sequence[index]]}}
             for index in range(32)]
    from .generic_passthrough import passthrough_sources, _snapshot
    if passthrough_sources(config):
        if not isinstance(canonical_full_feeds, list) or len(canonical_full_feeds) != 2:
            raise ValueError("Mixed original-input passthrough was not prepared outside timing")
        for index, full in enumerate(canonical_full_feeds):
            binding = config["canonical_full_input_bindings"][index]
            expected = binding["inputs"]
            if (not isinstance(full, Mapping) or set(full) != set(expected)
                    or any(_snapshot(full[name]) != expected[name] for name in full)):
                raise ValueError("Mixed canonical original-input bytes/normalization differ from actual Quality")
            original = _snapshot(tensors[index])
            feed = {config["input_binding"]["prepared_input"]["input_name"]: tensors[index]}
            runtime_feed = prepare_async_feed(feed) if callable(prepare_async_feed) else feed
            if ({name: _snapshot(value) for name, value in runtime_feed.items()} != binding["runtime_inputs"]
                    or _snapshot(tensors[index]) != original):
                raise ValueError("Mixed prepared SDK input differs from the same image's actual normal Quality")
        for index, task in enumerate(tasks):
            task["full_feed"] = canonical_full_feeds[sequence[index]]
    elif canonical_full_feeds is not None:
        raise ValueError("Canonical full feeds require an actual original-input passthrough")

    class DetectionSequence:
        def __init__(self):
            self.completed_count = 0
            self.last_result = {}
            self.processors = [_detection_module().create_detection_completion_runtime(item["completion_contract"],
                source_paths=(runner_source,), sentinel_context={
                    "prepared_input_sha256": item["prepared_input"]["sha256"]}) for item in payload["inputs"]]

        def process(self, outputs):
            processor = self.processors[sequence[self.completed_count]]
            before = processor.completed_count
            processor.process(outputs)
            if processor.completed_count != before + 1:
                raise RuntimeError("Mixed detection control processor completion count mismatch")
            self.last_result = processor.last_result
            self.completed_count += 1
            return self.last_result

    execution = normalize_stage1_execution(config.get("stage1_execution"))
    if ((stage1_async is None) != (execution["policy"] == "synchronous")
            or (stage1_async is not None and normalize_stage1_execution(stage1_async.execution_policy) != execution)):
        raise ValueError("Mixed control adapter differs from its prepared runtime policy")
    if stage1_async is not None and not callable(prepare_async_feed):
        raise ValueError("Mixed control async input adapter missing")
    original_prepare = stage1_async.prepare_inputs if stage1_async is not None else None

    def factory(prepare):
        stage1_async.prepare_inputs = lambda: prepare_async_feed(prepare())
        return stage1_async

    destination = Path(output) / "mixed_input_control" / f"repetition_{repetition}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError("Mixed control result already exists; preserve the previous attempt")
    source_before = {item["path"]: _sha(item["path"]) for item in config["runtime_source_files"]}
    expected_sources = _source_map(config)
    if source_before != expected_sources:
        raise ValueError("Mixed control runtime sources changed before functional execution")
    report = {}
    try:
        report = run_mixed_control(tasks, config=config, original_sources=payload["provenance"],
            serial_stage1=serial_stage1, transfer=transfer, stage2=stage2,
            processor_factory=ClassificationCompletion if config["task"] == "classification" else DetectionSequence,
            stage1_async_factory=factory if stage1_async is not None else None, clone_payload=clone_payload,
            boundary_observation=boundary_contract, boundary_runtime=boundary_runtime)
        submission = _submission_requirements(config, execution)
        if (any(report.get(key) != value for key, value in submission.items())
                or ("stage1_submission_mode" in submission and "stage1_peak_uncollected_jobs" in report)):
            raise RuntimeError("Mixed control did not observe the requested host submission depth")
        source_after = {item["path"]: _sha(item["path"]) for item in config["runtime_source_files"]}
        if source_after != source_before:
            raise RuntimeError("Mixed control runtime sources changed during functional execution")
        report["runtime_source_binding"] = {"before": source_before, "after": source_after, "unchanged": True}
    except BaseException as error:
        report = dict(getattr(error, "partial_report", report) or {})
        report.update(status="failed", error=f"{type(error).__name__}: {error}",
                      control_sidecar=dict(config["mixed_input_control"]))
        with destination.open("x", encoding="utf-8") as stream:
            json.dump(_report_value(report), stream, indent=2, allow_nan=False)
        # The outer throughput handler consumes partial_report as a throughput
        # record. Never leak functional counters or numpy snapshots into it.
        raise RuntimeError("Mixed functional control failed; diagnostic report: " + str(destination)) from error
    finally:
        if stage1_async is not None:
            stage1_async.prepare_inputs = original_prepare
    report["control_sidecar"] = dict(config["mixed_input_control"])
    report["task_input_indices"] = sequence
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(_report_value(report), stream, indent=2, allow_nan=False)
    proof = {"status": "passed", "checked_outside_throughput_timer": True,
            "serial_reference_count": report["serial_reference_count"],
            "mixed_completed_count": report["mixed_completed_count"],
            "drain_completed": report["drain_completed"], "stage1_execution": report["stage1_execution"],
            "control_sidecar": dict(config["mixed_input_control"]),
            "report": {"path": str(destination.relative_to(output)), "sha256": _sha(destination)}}
    proof.update(_submission_requirements(config, execution))
    if boundary_contract is not None:
        proof["boundary_observation"] = deepcopy(report["boundary_observation"])
    return proof


def _validate_boundary_snapshot(value, specs):
    if not isinstance(value, Mapping) or set(value) != set(specs):
        raise ValueError("Generic mixed boundary observation actual tensor names mismatch")
    for name, tensor in value.items():
        spec = specs[name]
        if (not isinstance(tensor, Mapping) or set(tensor) != {
                "shape", "dtype", "tensor_sha256", "strides", "c_contiguous", "f_contiguous",
                "owns_data", "buffer_address", "nbytes"}
                or {key: tensor.get(key) for key in ("shape", "dtype")} != spec
                or not _tensor_snapshot({key: tensor[key] for key in ("shape", "dtype", "tensor_sha256")})
                or not isinstance(tensor["strides"], list) or len(tensor["strides"]) != len(spec["shape"])
                or any(type(stride) is not int for stride in tensor["strides"])
                or any(type(tensor[key]) is not bool for key in ("c_contiguous", "f_contiguous", "owns_data"))
                or type(tensor["buffer_address"]) is not int or tensor["buffer_address"] <= 0
                or type(tensor["nbytes"]) is not int
                or tensor["nbytes"] != int(np.prod(spec["shape"])) * np.dtype(spec["dtype"]).itemsize):
            raise ValueError("Generic mixed boundary observation tensor snapshot malformed")


def _validate_boundary_result(proof, config, report=None):
    from .generic_mixed_control import _boundary_content, normalize_boundary_observation
    expected = config.get("boundary_observation")
    actual = proof.get("boundary_observation")
    if expected is None:
        if actual is not None or (report is not None and (report.get("boundary_observation") is not None
                or any("boundary_observation" in item for item in report.get("observations", [])))):
            raise ValueError("Generic mixed boundary observation was not requested by this runtime")
        return
    contract = normalize_boundary_observation(expected)
    execution = normalize_stage1_execution(config.get("stage1_execution"))
    required = {"schema_version": 1, "contract": contract, "observed_task_count": 32,
                "observed_boundary_tensor_count": len(set(contract["part2_to_part1"].values())),
                "all_boundary_tensors_equal": True, "unchanged_until_inference": True}
    if (not isinstance(actual, Mapping) or set(actual) != set(required) | {"slot_buffer_ownership"}
            or any(actual.get(key) != value or type(actual.get(key)) is not type(value)
                   for key, value in required.items())):
        raise ValueError("Generic mixed boundary observation proof missing or incomplete")
    ownership = actual["slot_buffer_ownership"]
    if execution["policy"] == "synchronous":
        if ownership is not None:
            raise ValueError("Generic mixed boundary synchronous control claims async slot evidence")
    elif (not isinstance(ownership, Mapping) or set(ownership) != {
            "slot_count", "inspected_buffer_count", "slots_disjoint", "slots"}
            or type(ownership["slot_count"]) is not int or ownership["slot_count"] != execution["inflight"]
            or type(ownership["inspected_buffer_count"]) is not int
            or ownership["inspected_buffer_count"] < execution["inflight"]
            or ownership["slots_disjoint"] is not True):
        raise ValueError("Generic mixed boundary actual slot ownership proof missing")
    if execution["policy"] != "synchronous":
        slots = ownership["slots"]
        if not isinstance(slots, list) or len(slots) != execution["inflight"]:
            raise ValueError("Generic mixed boundary actual slot buffer descriptors missing")
        spans, count = [], 0
        for index, slot in enumerate(slots):
            if (not isinstance(slot, Mapping) or set(slot) != {"slot_index", "buffers"}
                    or type(slot["slot_index"]) is not int or slot["slot_index"] != index
                    or not isinstance(slot["buffers"], list) or not slot["buffers"]):
                raise ValueError("Generic mixed boundary actual slot descriptor malformed")
            current = []
            for buffer in slot["buffers"]:
                if (not isinstance(buffer, Mapping) or set(buffer) != {
                        "shape", "dtype", "strides", "buffer_address", "nbytes", "byte_span"}
                        or not isinstance(buffer["shape"], list) or not buffer["shape"]
                        or any(type(dim) is not int or dim <= 0 for dim in buffer["shape"])
                        or not isinstance(buffer["strides"], list)
                        or len(buffer["strides"]) != len(buffer["shape"])
                        or any(type(stride) is not int for stride in buffer["strides"])
                        or type(buffer["buffer_address"]) is not int or buffer["buffer_address"] <= 0
                        or type(buffer["nbytes"]) is not int):
                    raise ValueError("Generic mixed boundary actual SDK buffer descriptor malformed")
                try:
                    dtype = np.dtype(buffer["dtype"])
                except (TypeError, ValueError) as error:
                    raise ValueError("Generic mixed boundary SDK buffer dtype invalid") from error
                offsets = [(dim - 1) * stride for dim, stride in zip(buffer["shape"], buffer["strides"])]
                address = buffer["buffer_address"]
                expected_span = [address + sum(min(0, offset) for offset in offsets),
                                 address + sum(max(0, offset) for offset in offsets) + dtype.itemsize]
                if (dtype.kind not in "biuf" or buffer["byte_span"] != expected_span
                        or buffer["nbytes"] != int(np.prod(buffer["shape"])) * dtype.itemsize):
                    raise ValueError("Generic mixed boundary actual SDK buffer bounds invalid")
                current.append(expected_span)
                count += 1
            if any(max(first[0], second[0]) < min(first[1], second[1])
                   for first in current for previous in spans for second in previous):
                raise ValueError("Generic mixed boundary SDK slots overlap in observed memory")
            spans.append(current)
        if count != ownership["inspected_buffer_count"]:
            raise ValueError("Generic mixed boundary actual SDK buffer count mismatch")
    if report is None:
        return
    if report.get("boundary_observation") != actual:
        raise ValueError("Generic mixed boundary full report differs from its proof")
    first_runtime = None
    for task_index, observation in enumerate(report["observations"]):
        record = observation.get("boundary_observation")
        if not isinstance(record, Mapping) or set(record) != {"serial", "mixed", "passed"} or record["passed"] is not True:
            raise ValueError("Generic mixed boundary task observation missing")
        for phase in ("serial", "mixed"):
            item = record[phase]
            expected_keys = {
                    "part1_at_collection", "part1_before_transfer", "part1_after_transfer",
                    "part2_after_transfer", "part2_before_inference", "part2_after_inference",
                    "no_reusable_sdk_buffer_alias", "no_prepared_input_alias", "runtime", "part2_to_part1_aliases"}
            if contract["passthrough_inputs"]:
                expected_keys.add("passthrough_input_content")
            if not isinstance(item, Mapping) or set(item) != expected_keys:
                raise ValueError("Generic mixed boundary phase evidence malformed")
            for key in ("part1_at_collection", "part1_before_transfer", "part1_after_transfer"):
                _validate_boundary_snapshot(item[key], contract["part1_outputs"])
            for key in ("part2_after_transfer", "part2_before_inference", "part2_after_inference"):
                _validate_boundary_snapshot(item[key], contract["part2_inputs"])
            if contract["passthrough_inputs"]:
                bindings = config.get("canonical_full_input_bindings")
                if not isinstance(bindings, list) or len(bindings) != 2:
                    raise ValueError("Generic mixed canonical original-input binding missing")
                binding = bindings[_input_sequence()[task_index]]
                evidence = item["passthrough_input_content"]
                if (not isinstance(evidence, Mapping) or set(evidence) != set(contract["passthrough_inputs"])
                        or observation.get("input_sha256") != binding["prepared_input_sha256"]):
                    raise ValueError("Generic mixed canonical original-input task/source mismatch")
                _validate_boundary_snapshot(evidence, {name: contract["part2_inputs"][name]
                    for name in contract["passthrough_inputs"]})
                for target, source in contract["passthrough_inputs"].items():
                    observed = {key: evidence[target][key] for key in ("shape", "dtype", "tensor_sha256")}
                    actual_input = {key: item["part2_before_inference"][target][key]
                                    for key in ("shape", "dtype", "tensor_sha256")}
                    if observed != binding["inputs"][source] or actual_input != observed:
                        raise ValueError("Generic mixed passthrough bytes differ from this task's canonical normal Quality input")
            if (_boundary_content(item["part1_at_collection"]) != _boundary_content(item["part1_before_transfer"])
                    or item["part1_before_transfer"] != item["part1_after_transfer"]
                    or item["part2_after_transfer"] != item["part2_before_inference"]
                    or item["part2_before_inference"] != item["part2_after_inference"]
                    or item["no_reusable_sdk_buffer_alias"] is not True
                    or item["no_prepared_input_alias"] is not True):
                raise ValueError("Generic mixed boundary data changed before processing")
            runtime = item["runtime"]
            if (not isinstance(runtime, Mapping) or set(runtime) != {
                    "part1_io", "part2_io", "part2_to_part1", "passthrough_inputs", "transfer_contract"}
                    or any(runtime.get(key) != contract[key] for key in (
                        "part2_to_part1", "passthrough_inputs", "transfer_contract"))
                    or not isinstance(runtime["part1_io"], Mapping) or not runtime["part1_io"]
                    or not isinstance(runtime["part2_io"], Mapping)
                    or runtime["part2_io"].get("inputs") != contract["part2_inputs"]
                    or set(runtime["part1_io"].get("canonical_outputs", [])) != set(contract["part1_outputs"])):
                raise ValueError("Generic mixed boundary live mapping/IO/layout/quantization mismatch")
            if first_runtime is None:
                first_runtime = runtime
            if runtime != first_runtime:
                raise ValueError("Generic mixed boundary live runtime metadata changed")
            aliases = item["part2_to_part1_aliases"]
            if (not isinstance(aliases, Mapping) or set(aliases) != set(contract["part2_inputs"])
                    or any(not isinstance(sources, list) or len(set(sources)) != len(sources)
                           or any(source not in contract["part1_outputs"] for source in sources)
                           for sources in aliases.values())):
                raise ValueError("Generic mixed boundary alias evidence malformed")
        for key in ("part1_at_collection", "part2_before_inference"):
            if _boundary_content(record["serial"][key]) != _boundary_content(record["mixed"][key]):
                raise ValueError("Generic mixed boundary serial/async tensor content differs")


def _submission_requirements(config, execution):
    if (execution["policy"] == "synchronous"
            and config.get("measurement_mode") == "generic_only_completed_task"):
        return {"stage1_submission_mode": "synchronous_stage1_in_two_worker_pipeline"}
    return {"stage1_peak_uncollected_jobs": execution["inflight"]}


def validate_bound_mixed_result(repetition, config, *, result_root=None):
    """Reject foreign/partial controls on normal result import and reentry.

    Frozen runtime paths can be remote. The controller verifies the collected
    report and its embedded source binding without loading remote sidecar paths.
    """
    expected = config.get("mixed_input_control")
    proof = repetition.get("mixed_input_control")
    if expected is None and proof is None:
        return None
    if not _descriptor(expected) or not isinstance(proof, Mapping):
        raise ValueError("Generic mixed functional control proof is missing or foreign")
    sources = _source_map(config)
    if sources.get(expected["path"]) != expected["sha256"]:
        raise ValueError("Generic mixed functional control is outside its frozen source closure")
    execution = normalize_stage1_execution(config.get("stage1_execution"))
    required = {"status": "passed", "checked_outside_throughput_timer": True,
                "serial_reference_count": 32, "mixed_completed_count": 32,
                **_submission_requirements(config, execution), "drain_completed": True,
                "stage1_execution": execution, "control_sidecar": dict(expected)}
    synchronous_generic_only = "stage1_submission_mode" in required
    if (any(proof.get(key) != value or type(proof.get(key)) is not type(value)
            for key, value in required.items())
            or (synchronous_generic_only and "stage1_peak_uncollected_jobs" in proof)
            or normalize_stage1_execution(repetition.get("stage1_execution")) != execution):
        raise ValueError("Generic mixed functional control counts/policy/drain evidence mismatch")
    _validate_boundary_result(proof, config)
    descriptor = proof.get("report")
    if (not isinstance(descriptor, Mapping) or set(descriptor) != {"path", "sha256"}
            or not isinstance(descriptor["path"], str) or not descriptor["path"]
            or Path(descriptor["path"]).is_absolute() or ".." in Path(descriptor["path"]).parts
            or not isinstance(descriptor["sha256"], str)
            or re.fullmatch(r"[a-f0-9]{64}", descriptor["sha256"]) is None):
        raise ValueError("Generic mixed functional control report descriptor invalid")
    if result_root is None:
        return dict(proof)
    root = Path(result_root).resolve()
    path = (root / descriptor["path"]).resolve()
    if not path.is_relative_to(root) or not path.is_file() or _sha(path) != descriptor["sha256"]:
        raise ValueError("Generic mixed functional control report missing, escaped or changed")
    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, Mapping):
        raise ValueError("Generic mixed functional control report malformed")
    report_required = {**{key: value for key, value in required.items() if key != "checked_outside_throughput_timer"},
        "checked_outside_throughput_timer": True, "requested_task_count": 32,
        "scientific_statistics_published": False, "exact_task_order": True,
        "all_declared_tasks_completed": True, "comparison": "exact_outputs_and_completed_results",
        "purpose": "generic_mixed_input_functional_control", "task_input_indices": _input_sequence(),
        "queue_depth": config.get("queue_depth"), "timeout_s": config.get("timeout_s"),
        "handoff_payload_copy": repetition.get("handoff_payload_copy")}
    if (any(report.get(key) != value for key, value in report_required.items())
            or (synchronous_generic_only and "stage1_peak_uncollected_jobs" in report)
            or any(key in report for key in ("completed_task_fps", "measurement_start_ns",
                                              "measured_makespan_seconds"))):
        raise ValueError("Generic mixed functional control full report contract mismatch")
    binding = report.get("runtime_source_binding", {})
    if (binding.get("before") != sources or binding.get("after") != sources
            or binding.get("unchanged") is not True):
        raise ValueError("Generic mixed functional control runtime source binding mismatch")
    runtime_binding = {key: config[key] for key in ("identity", "execution_id", "original_run",
        "expected_runtime_artifacts", "runtime_source_files", "execution_environment") if key in config}
    if report.get("runtime_binding") != runtime_binding:
        raise ValueError("Generic mixed functional control runtime identity/artifacts mismatch")
    observations = report.get("observations")
    if not isinstance(observations, list) or len(observations) != 32:
        raise ValueError("Generic mixed functional control lacks 32 ordered observations")
    input_specs, input_hashes = [], []
    for observation in observations[:2]:
        if not isinstance(observation, Mapping):
            raise ValueError("Generic mixed functional control observation malformed")
        input_specs.append(observation.get("prepared_input"))
        input_hashes.append(observation.get("prepared_tensor_sha256"))
    if (any(not isinstance(spec, Mapping) for spec in input_specs)
            or input_specs[0] == input_specs[1] or input_hashes[0] == input_hashes[1]):
        raise ValueError("Generic mixed functional control inputs are not distinguishable")
    expected_input = config["input_binding"]["prepared_input"]
    if any(input_specs[0].get(key) != expected_input.get(key) for key in
           ("path", "sha256", "input_name", "shape", "dtype", "format")):
        raise ValueError("Generic mixed functional control baseline input mismatch")
    for index, observation in enumerate(observations):
        selected = _input_sequence()[index]
        spec = input_specs[selected]
        if (not isinstance(spec, Mapping) or not isinstance(observation, Mapping)
                or observation.get("task_id") != f"task_{index:02d}"
                or observation.get("prepared_input") != spec
                or observation.get("input_sha256") != spec.get("sha256")
                or sources.get(spec.get("path")) != spec.get("sha256")
                or observation.get("prepared_tensor_sha256") != input_hashes[selected]
                or not isinstance(input_hashes[selected], str)
                or re.fullmatch(r"[a-f0-9]{64}", input_hashes[selected]) is None
                or observation.get("passed") is not True
                or observation.get("metrics") != {"raw_outputs_and_completed_result_exact": True}):
            raise ValueError("Generic mixed functional control task association/comparison mismatch")
        serial, mixed = observation.get("serial"), observation.get("mixed")
        if (not isinstance(serial, Mapping) or not serial.get("outputs")
                or not isinstance(serial["outputs"], Mapping)
                or any(not _tensor_snapshot(value) for value in serial["outputs"].values())
                or not serial.get("completed_result") or serial != mixed):
            raise ValueError("Generic mixed functional control raw/completion snapshots mismatch")
    if observations[0]["serial"]["outputs"] == observations[1]["serial"]["outputs"]:
        raise ValueError("Generic mixed functional control serial outputs cannot distinguish inputs")
    coverage_key = ("stage1_submission_input_coverage" if synchronous_generic_only
                    else "stage1_slot_input_coverage")
    index_key = "submission_stream_index" if synchronous_generic_only else "slot_index"
    if synchronous_generic_only and "stage1_slot_input_coverage" in report:
        raise ValueError("Generic mixed synchronous control cannot claim SDK slot coverage")
    coverage = report.get(coverage_key)
    if not isinstance(coverage, list) or len(coverage) != execution["inflight"]:
        raise ValueError("Generic mixed functional control lacks actual slot input coverage")
    for slot, observed in enumerate(coverage):
        indices = list(range(slot, 32, execution["inflight"]))
        expected_hashes = sorted({observations[index]["prepared_tensor_sha256"] for index in indices})
        if (not isinstance(observed, Mapping) or type(observed.get(index_key)) is not int
                or observed[index_key] != slot
                or observed.get("task_ids") != [f"task_{index:02d}" for index in indices]
                or observed.get("prepared_tensor_sha256") != expected_hashes
                or observed.get("distinct_prepared_tensor_count") != 2
                or observed.get("serial_outputs_discriminating") is not True):
            raise ValueError("Generic mixed functional control slot input/task association mismatch")
    _validate_boundary_result(proof, config, report)
    return dict(proof)
