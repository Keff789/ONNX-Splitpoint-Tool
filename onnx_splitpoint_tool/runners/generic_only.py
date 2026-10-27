"""Explicit Generic-only admission and serial compiled-part controls.

These controls run outside the throughput timer.  They neither supply Native
performance nor replace the separately bound CPU-ONNX dataset quality result.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path

import numpy as np


MODE = "generic_only_completed_task"
ARTIFACT_SCHEMA = "onnx-splitpoint/generic-only-artifact-binding/v1"
ARTIFACT_ROLES = ("part1_runtime", "engine", "source_part1_onnx", "source_part2_onnx",
                  "build_part2_onnx", "engine_build_receipt", "boundary_metadata",
                  "part1_build_receipt")
CLASSIFICATION_CONTRACT = {"task": "classification", "operation": "top1_top5", "label_offset": 0}


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_generic_only(config):
    mode = config.get("measurement_mode") == MODE
    admission = config.get("admission_mode") == "generic_only"
    if mode != admission:
        raise ValueError("Generic-only measurement and admission modes must agree")
    return mode


def _descriptor(value, *, check_files=False):
    if (not isinstance(value, Mapping) or not isinstance(value.get("path"), str)
            or not Path(value["path"]).is_absolute()
            or ".." in Path(value["path"]).parts
            or not isinstance(value.get("sha256"), str)
            or len(value["sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in value["sha256"])):
        raise ValueError("Generic-only artifact descriptor is missing or invalid")
    if check_files and _sha(value["path"]) != value["sha256"]:
        raise ValueError("Generic-only artifact bytes changed: " + value["path"])
    return Path(value["path"])


def validate_generic_only_config(config, *, check_files=False):
    from .task_completion import normalize_stage1_execution

    if not is_generic_only(config):
        raise ValueError("Generic-only configuration requires explicit admission")
    energy = config.get('energy_capture')
    if energy:
        from .generic_energy import validate_config
        validate_config(config)
        reuse = energy.get('generic_only_control_reuse') or {}
        report_descriptor = reuse.get('normal_admission_report') or {}
        performance_descriptor = reuse.get('performance_result') or {}
        sources = {item['path']: item['sha256'] for item in energy['contract']['source_files']}
        def evidence_path(role, descriptor):
            if sources.get(descriptor.get('path')) != descriptor.get('sha256'):
                raise ValueError('Generic-only energy evidence is outside its source seal')
            path = _descriptor(descriptor)
            if check_files or path.is_file():
                return _descriptor(descriptor, check_files=True)
            # Controller imports use only the exact hash-checked local copy of
            # this same role/path. DUT preflight never takes this alias path.
            aliases = [item for item in energy.get('verifier_binding', {}).get('evidence', [])
                       if item.get('role') == role and item.get('runtime_path') == descriptor['path']
                       and item.get('local', {}).get('sha256') == descriptor['sha256']]
            if len(aliases) != 1:
                raise ValueError('Generic-only energy controller evidence alias differs')
            return _descriptor(aliases[0]['local'], check_files=True)
        report_path = evidence_path('normal_admission_report', report_descriptor)
        performance_path = evidence_path('performance_result', performance_descriptor)
        import json
        report = json.loads(report_path.read_text())
        performance = json.loads(performance_path.read_text())
        admission = reuse.get('normal_admission') or {}
        selected = [row for row in report.get('rows', []) if row.get('case_key') == admission.get('case_key')]
        repetitions = performance.get('repetitions') or []
        if (reuse.get('schema') != 'onnx-splitpoint/generic-only-energy-control-reuse/v1'
                or performance_descriptor['sha256'] != energy.get('performance_result_sha256')
                or performance.get('identity') != config['identity']
                or performance.get('execution_id') != config['execution_id']
                or performance.get('process_repetition_policy') != 'separate_processes'
                or len(repetitions) != 3 or selected != [admission]
                or admission.get('admission_mode') != 'generic_only'
                or admission.get('eligible_for_generic_only_execution') is not True
                or admission.get('native_performance_claimed') is not False
                or len(admission.get('completion_attestations') or []) != 3
                or config.get('mixed_input_control')):
            raise ValueError('Generic-only energy requires exact completed normal control evidence')
        for repetition in repetitions:
            proof = repetition.get('mixed_input_control') or {}
            if (proof.get('status') != 'passed' or proof.get('checked_outside_throughput_timer') is not True
                    or proof.get('serial_reference_count') != 32 or proof.get('mixed_completed_count') != 32
                    or proof.get('drain_completed') is not True
                    or proof.get('stage1_execution') != config['stage1_execution']
                    or proof.get('control_sidecar') != reuse.get('original_mixed_input_control')):
                raise ValueError('Generic-only energy completed mixed control differs')
    original = config.get("original_run_cfg") or {}
    provider = str(original.get("stage1_provider") or original.get("provider") or "").lower()
    expected_policy = ({"policy": "hailo_async_slots_v1", "inflight": 8}
                       if provider in {"hailo10h", "hailo10"} else
                       {"policy": "deepx_async_slots_v1", "inflight": 4}
                       if provider == "deepx_m1" else
                       {"policy": "synchronous", "inflight": 1}
                       if provider == "hailo8" else None)
    if (expected_policy is None or not isinstance(config.get("stage1_execution"), Mapping)
            or normalize_stage1_execution(config["stage1_execution"]) != expected_policy):
        raise ValueError("Generic-only execution requires the fixed backend stage1 policy")
    if config.get("task") not in {"classification", "detection"}:
        raise ValueError("Generic-only task is unsupported")
    if (config["task"] == "classification"
            and (config.get("completion_contract") != CLASSIFICATION_CONTRACT
                 or type(config["completion_contract"].get("label_offset")) is not int)):
        raise ValueError("Generic-only classification requires the normal Top1/Top5 contract")
    if (original.get("native_split_quality_binding")
            or config.get("native_source") or config.get("native_command_contract")):
        raise ValueError("Generic-only controls cannot claim a Native binding or energy result")
    if not energy and not config.get("mixed_input_control"):
        raise ValueError("Generic-only execution requires the bound mixed-input control")
    binding = config.get("generic_artifact_binding")
    if not isinstance(binding, Mapping) or binding.get("schema") != ARTIFACT_SCHEMA:
        raise ValueError("Generic-only compiled artifact binding is missing")
    if (not isinstance(binding.get("runtime_precision"), str) or not binding["runtime_precision"]
            or binding["runtime_precision"] != (config.get("identity") or {}).get("precision")):
        raise ValueError("Generic-only identity differs from the producer's runtime precision")
    artifacts = binding.get("artifacts") or {}
    paths = {role: _descriptor(artifacts.get(role), check_files=check_files) for role in ARTIFACT_ROLES}
    if not binding.get("boundary_contract") or not binding.get("quality_reference") or not binding.get("producer_source"):
        raise ValueError("Generic-only boundary, producer and CPU quality bindings are required")
    expected = config.get("expected_runtime_artifacts") or {}
    if set(expected) != {"part1_runtime", "engine"} or any(
            expected[role] != {key: artifacts[role][key] for key in ("path", "sha256")}
            for role in expected):
        raise ValueError("Generic-only exact runtime artifact binding differs")
    reference = config.get("functional_reference") or {}
    prepared = (config.get("input_binding") or {}).get("prepared_input") or {}
    if (reference.get("kind") != "serial_compiled_subnets"
            or reference.get("input_sha256") != prepared.get("sha256")
            or not prepared.get("sha256")):
        raise ValueError("Generic-only serial reference input or kind differs")
    outputs = reference.get("outputs")
    if not isinstance(outputs, Mapping) or not outputs:
        raise ValueError("Generic-only physical output endpoint is missing")
    for name, spec in outputs.items():
        if (not isinstance(name, str) or not name or not isinstance(spec, Mapping)
                or set(spec) != {"shape", "dtype"} or not isinstance(spec["shape"], list)
                or not spec["shape"] or any(type(n) is not int or n <= 0 for n in spec["shape"])
                or spec["dtype"] not in {"float16", "float32", "float64", "int8", "uint8", "int32", "int64"}):
            raise ValueError("Generic-only physical output tensor contract is invalid")
    return paths


def _json_value(value):
    return json.loads(json.dumps(value, default=lambda item:
        item.tolist() if isinstance(item, np.ndarray) else item.item()))


def _completed_value(processor, config):
    value = _json_value(processor.last_result)
    if config["task"] == "detection":
        value.setdefault("host_nms_applied", config["completion_contract"].get("completion_mode")
                         != "decoded_nms_attested_materialization_no_second_nms")
    return value


def _classification_result(outputs):
    from .harness.classification import classification_topk
    top5 = classification_topk(outputs)
    return {"top1": top5[:, 0].tolist(), "top5": top5.tolist()}


def validate_classification_report(report, completed, expected_count):
    """Check the unchanged processor's actual report, with no detector fields."""
    from .harness.classification import ClassificationCompletion
    value = _semantic_result(completed, expected_count, task="classification")
    processor = ClassificationCompletion()
    processor.last_result = {key: np.asarray(item, dtype=np.int64) for key, item in value.items()}
    processor.completed_count = expected_count
    expected = processor.report()
    if (not isinstance(report, Mapping) or dict(report) != expected
            or any(type(report.get(key)) is not type(item) for key, item in expected.items())
            or _semantic_result(report.get("classification_topk"), expected_count,
                                task="classification") != value):
        raise ValueError("Generic-only classification postprocess report/count differs")
    return expected


def capture_serial_reference(config, output, repetition_index, *, stage1, transfer, stage2, processor,
                             attempt_number=1):
    """Run the same compiled callbacks serially once, then own every output byte."""
    validate_generic_only_config(config)
    if type(repetition_index) is not int or repetition_index not in (1, 2, 3):
        raise ValueError("Generic-only serial capture requires one independent process repetition")
    if type(attempt_number) is not int or attempt_number < 1:
        raise ValueError("Generic-only serial capture attempt identity is invalid")
    if processor.completed_count != 0:
        raise ValueError("Generic-only serial reference requires a fresh completion processor")
    root = Path(output)
    destination = root / "serial_reference" / f"repetition_{repetition_index}_attempt_{attempt_number:03d}"
    destination.mkdir(parents=True, exist_ok=False)
    # No asynchronous submission or alternate inference implementation here.
    boundary = stage1()
    inputs = transfer(boundary)
    actual = stage2(inputs)
    specs = config["functional_reference"]["outputs"]
    if not isinstance(actual, Mapping) or set(actual) != set(specs):
        raise ValueError("Generic-only serial output names differ from the physical endpoint")
    outputs, samples = {}, {}
    for ordinal, (name, spec) in enumerate(specs.items()):
        value = np.asarray(actual[name])
        if (list(value.shape) != spec["shape"] or str(value.dtype) != spec["dtype"]
                or not np.isfinite(value).all()):
            raise ValueError("Generic-only serial output shape, dtype or values differ")
        outputs[name] = np.array(value, copy=True, order="C")
        path = destination / f"output_{ordinal}.npy"
        np.save(path, outputs[name], allow_pickle=False)
        samples[name] = {"path": str(path.relative_to(root)), "sha256": _sha(path), **spec}
    processor.process(outputs)
    if processor.completed_count != 1 or not processor.last_result:
        raise ValueError("Generic-only serial reference did not complete the normal task")
    completed = _completed_value(processor, config)
    completed_path = destination / "completed_task.json"
    completed_path.write_text(json.dumps(completed, indent=2) + "\n", encoding="utf-8")
    proof = {"status": "passed", "checked_outside_timer": True, "observed_task_count": 1,
             "relation": "serial_same_compiled_parts_before_measured_tasks",
             "reference_binding": config["functional_reference"],
             "runtime_artifacts": config["expected_runtime_artifacts"],
             "completion_contract": config["completion_contract"],
             "raw_output_samples": samples, "last_completed_result": completed,
             "completed_result_artifact": {"path": str(completed_path.relative_to(root)),
                                           "sha256": _sha(completed_path)}}
    if config["task"] == "classification":
        if completed != _classification_result(outputs):
            raise ValueError("Generic-only serial classification differs from current raw Top1/Top5")
        proof["postprocess_report"] = _json_value(processor.report())
        validate_classification_report(proof["postprocess_report"], completed, 1)
    return outputs, proof


def _semantic_result(value, expected_count, *, task="detection"):
    """Keep every semantic field while checking the separate invocation count.

    Strict completion binds its invocation index/hash; fast completion records
    its completed count. Their full evidence remains saved and the normal
    completion attestation verifier checks the measured execution separately.
    """
    value = _json_value(value)
    if task == "classification":
        if (not isinstance(value, Mapping) or set(value) != {"top1", "top5"}
                or not isinstance(value["top1"], list) or not value["top1"]
                or not isinstance(value["top5"], list) or len(value["top5"]) != len(value["top1"])
                or any(not isinstance(row, list) or not 1 <= len(row) <= 5
                       or any(type(index) is not int or index < 0 for index in row)
                       or len(set(row)) != len(row) or type(first) is not int or first != row[0]
                       for first, row in zip(value["top1"], value["top5"]))):
            raise ValueError("Generic-only classification Top1/Top5 result differs")
        return value
    counters = [name for name in ("completed_count", "invocation_index") if name in value]
    if not counters or any(type(value[name]) is not int or value[name] != expected_count for name in counters):
        raise ValueError("Generic-only serial/measured completion count differs")
    semantic = {key: item for key, item in value.items()
                if key not in {"completed_count", "invocation_index", "invocation_sha256"}}
    if "artifact" in value or "artifact_sha256" in value:
        # Strict completion's transport artifact contains the invocation index,
        # so its hash necessarily differs between task 1 and task 1100. Check
        # the actual artifact before removing only that invocation identity;
        # all of its semantic content remains part of the comparison.
        try:
            from .native_detection_postprocess import canonical_json_sha256
        except ImportError:
            from ..native_detection_postprocess import canonical_json_sha256
        artifact = value.get("artifact")
        if (not isinstance(artifact, Mapping)
                or type(artifact.get("invocation_index")) is not int
                or artifact["invocation_index"] != expected_count
                or value.get("artifact_sha256") != canonical_json_sha256(artifact)):
            raise ValueError("Generic-only completion artifact or invocation count differs")
        semantic["artifact"] = {key: item for key, item in artifact.items()
                                if key != "invocation_index"}
        semantic.pop("artifact_sha256")
    return semantic


def compare_serial_reference(outputs, completed, reference_outputs, proof, *, completed_count=1100):
    if type(completed_count) is not int or completed_count <= 0:
        raise ValueError('Generic-only completed count must be a positive actual count')
    task = (proof.get("completion_contract") or {}).get("task", "detection")
    if (set(outputs) != set(reference_outputs)
            or any(np.asarray(outputs[name]).dtype != reference_outputs[name].dtype
                   or np.asarray(outputs[name]).shape != reference_outputs[name].shape
                   or not np.array_equal(outputs[name], reference_outputs[name])
                   for name in reference_outputs)
            or _semantic_result(completed, completed_count, task=task)
               != _semantic_result(proof["last_completed_result"], 1, task=task)):
        raise ValueError("Generic-only measured raw/completed output differs from its serial reference")
    if task == "classification" and (_json_value(completed) != _classification_result(outputs)
            or proof["last_completed_result"] != _classification_result(reference_outputs)):
        raise ValueError("Generic-only classification result differs from current raw Top1/Top5")


def validate_serial_reference(repetition, config, *, result_root=None):
    """Normal importer verification, including actual saved array equality."""
    validate_generic_only_config(config)
    completed_count = 1100
    if config.get('energy_capture'):
        from .task_completion import validate_split_completion_duration
        validate_split_completion_duration(repetition, duration_s=config['energy_capture']['contract']['duration_s'])
        completed_count = repetition['completed_work_units']
    if (config["stage1_execution"]["policy"] == "synchronous"
            and "stage1_peak_uncollected_jobs" in repetition):
        raise ValueError("Generic-only synchronous completion cannot claim an observed async peak")
    proof = repetition.get("serial_reference") or {}
    expected = {"status": "passed", "checked_outside_timer": True, "observed_task_count": 1,
                "relation": "serial_same_compiled_parts_before_measured_tasks",
                "reference_binding": config["functional_reference"],
                "runtime_artifacts": config["expected_runtime_artifacts"],
                "completion_contract": config["completion_contract"]}
    if any(proof.get(key) != value for key, value in expected.items()):
        raise ValueError("Generic-only serial reference is missing or foreign")
    specs = config["functional_reference"]["outputs"]
    samples = proof.get("raw_output_samples") or {}
    task = config["task"]
    if (set(samples) != set(specs)
            or _semantic_result(proof.get("last_completed_result") or {}, 1, task=task)
               != _semantic_result(repetition.get("last_completed_result") or {}, completed_count, task=task)):
        raise ValueError("Generic-only serial reference endpoint/completed result differs")
    if task == "classification":
        validate_classification_report(proof.get("postprocess_report"), proof["last_completed_result"], 1)
        validate_classification_report(repetition.get("postprocess_report"), repetition["last_completed_result"], completed_count)
    for name, spec in specs.items():
        if any(samples[name].get(key) != value for key, value in spec.items()):
            raise ValueError("Generic-only serial tensor metadata differs")
    if result_root is None:
        return
    root = Path(result_root).resolve()

    def bound(descriptor):
        relative = Path(str(descriptor.get("path") or ""))
        if not descriptor.get("path") or relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Generic-only serial sample path invalid")
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or _sha(path) != descriptor.get("sha256"):
            raise ValueError("Generic-only serial sample missing, escaped or changed")
        return path

    arrays = {}
    actual = {}
    for name, spec in specs.items():
        value = np.load(bound(samples[name]), allow_pickle=False)
        if (list(value.shape) != spec["shape"] or str(value.dtype) != spec["dtype"]
                or not np.isfinite(value).all()):
            raise ValueError("Generic-only saved serial tensor contract differs")
        arrays[name] = value
        actual[name] = np.load(bound(repetition["raw_output_samples"][name]), allow_pickle=False)
    completed = json.loads(bound(proof.get("completed_result_artifact") or {}).read_text(encoding="utf-8"))
    if completed != proof["last_completed_result"]:
        raise ValueError("Generic-only serial completion artifact differs")
    compare_serial_reference(actual, repetition["last_completed_result"], arrays, proof, completed_count=completed_count)
