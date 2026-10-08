"""Measured completion for generic runners, independent of Native reports."""
from __future__ import annotations
import time
import math
import queue
import threading
import itertools
from typing import Mapping

PRODUCERS = {"generic_deepx_full", "generic_hailo_full", "generic_ort_full"}


def load_completion_bound_array(spec, *, file_sha256):
    """Read a bound completion tensor without executing model inference."""
    from pathlib import Path
    import re
    import numpy as np

    path = Path(spec["path"])
    if (not path.is_file() or not re.fullmatch(r"[0-9a-f]{64}", str(spec.get("sha256") or ""))
            or file_sha256(path) != spec["sha256"]):
        raise ValueError(f"Generic completion bound tensor changed: {path}")
    if spec.get("format") in {"raw", "raw_bin", "binary"}:
        value = np.fromfile(path, dtype=np.dtype(spec["dtype"])).reshape(spec["shape"])
    else:
        value = np.load(path, allow_pickle=False)
    if not isinstance(value, np.ndarray) or list(value.shape) != spec["shape"] or str(value.dtype) != spec["dtype"]:
        raise ValueError("Generic completion bound tensor shape/dtype mismatch")
    if value.dtype.kind not in "fiu" or not np.isfinite(value).all():
        raise ValueError("Generic completion bound tensor is not finite numeric data")
    if "c_contiguous" in spec and (type(spec["c_contiguous"]) is not bool
            or bool(value.flags.c_contiguous) != spec["c_contiguous"]):
        raise ValueError("Generic completion bound tensor storage order mismatch")
    return np.ascontiguousarray(value)


class TimedTaskCompletion:
    def __init__(self, producer, task, counter, *, contract=None, clock=time.perf_counter_ns):
        if producer not in PRODUCERS or task not in {"classification", "detection"}:
            raise ValueError("invalid generic completion producer/task")
        self.producer, self.task, self.counter = producer, task, counter
        self.contract, self.clock, self.frames = dict(contract or {}), clock, []

    def run(self, infer, complete):
        before = self.counter()
        start = self.clock()
        outputs = infer()
        runtime_end = self.clock()
        complete(outputs)
        tail_end = self.clock()
        after = self.counter()
        timer_end = self.clock()
        if after - before != 1:
            raise RuntimeError("generic task completion count mismatch")
        self.frames.append(dict(start_ns=start, runtime_end_ns=runtime_end, tail_end_ns=tail_end,
                                timer_end_ns=timer_end, completions=after-before))
        return outputs

    def report(self, warmup, expected):
        frames = self.frames[int(warmup):]
        evidence = dict(producer=self.producer, task=self.task, frames=frames,
                        expected_frames=int(expected), preprocessing_timed=False,
                        scope="prepared_input_to_completed_task", postprocess_contract=self.contract)
        validate_completion(evidence, task=self.task, producer=self.producer)
        return evidence


def validate_completion(evidence, *, task, producer=None):
    if not isinstance(evidence, Mapping) or evidence.get("producer") not in PRODUCERS:
        raise ValueError("missing or foreign generic completion producer")
    if producer is not None and evidence["producer"] != producer:
        raise ValueError("generic completion producer mismatch")
    if evidence.get("task") != task or task not in {"classification", "detection"}:
        raise ValueError("generic completion task mismatch")
    if evidence.get("scope") != "prepared_input_to_completed_task":
        raise ValueError("generic completion timing scope mismatch")
    contract = evidence.get("postprocess_contract") or {}
    if contract.get("source_nms_attested") and contract.get("host_nms_applied"):
        raise ValueError("double NMS")
    frames = evidence.get("frames") or []
    if not frames or len(frames) != evidence.get("expected_frames"):
        raise ValueError("generic completion frame count mismatch")
    for frame in frames:
        stamps = [frame.get(k) for k in ("start_ns", "runtime_end_ns", "tail_end_ns", "timer_end_ns")]
        if any(type(v) is not int or v <= 0 for v in stamps) or stamps != sorted(stamps) or stamps[-1] <= stamps[0] or frame.get("completions") != 1:
            raise ValueError("generic tail missing or outside measurement interval")
    return True


def completion_projection(evidence, *, task, producer=None):
    validate_completion(evidence, task=task, producer=producer)
    frames = evidence["frames"]
    return dict(generic_completion_evidence=dict(evidence), endpoint_contract_complete=True,
                structural_contract_pass=True, structural_contract_reason="measured_generic_task_completion",
                measurement_endpoint="completed_"+task, full_measurement_endpoint="completed_"+task,
                host_tail_available=True, decoder_contract_pass=True, nms_ok=True,
                decoder_id=str((evidence.get("postprocess_contract") or {}).get("decoder_id") or (evidence.get("postprocess_contract") or {}).get("normalizer_id") or "measured_top1_top5"),
                raw_head_contract_status="host_tail_verified", postprocess_included=True, postprocess_completion_verified=True,
                postprocess_completed_frames=len(frames),
                raw_stage_mean_ms=sum(f["runtime_end_ns"]-f["start_ns"] for f in frames)/len(frames)/1e6,
                host_tail_mean_ms=sum(f["tail_end_ns"]-f["runtime_end_ns"] for f in frames)/len(frames)/1e6,
                completed_task_mean_ms=sum(f["timer_end_ns"]-f["start_ns"] for f in frames)/len(frames)/1e6)


def measure_split_completion(stage1, transfer, stage2, processor, *, warmup=100,
                             tasks=1000, queue_depth=2, clone_payload=None,
                             timeout_s=3600.0, clock=time.perf_counter_ns, progress=None,
                             duration_s=None):
    """Time the existing two-worker Generic pipeline through actual completion.

    Backend adapters return materialized host tensors (including their existing
    device wait). No device-wide or per-frame synchronization is added here.
    Warmup drains completely before the measured producer starts. The measured
    interval includes pipeline fill and drain, handoff and task materialization;
    worker creation, input preparation and subsequent validation are outside it.
    """
    duration_mode = duration_s is not None
    if duration_mode and (type(duration_s) not in (int, float)
            or not math.isfinite(duration_s) or duration_s <= 0
            or warmup != 0 or tasks is not None):
        raise ValueError("generic completion duration requires finite positive seconds, zero warmup and no fixed task budget")
    budgets = ((warmup, 0), (queue_depth, 1)) + (() if duration_mode else ((tasks, 1),))
    if any(type(value) is not int or value < minimum for value, minimum in budgets):
        raise ValueError("generic completion invalid work budget")
    if not math.isfinite(timeout_s) or timeout_s <= 0 or (duration_mode and timeout_s <= duration_s):
        raise ValueError("generic completion invalid timeout")
    if not callable(getattr(processor, "process", None)):
        raise ValueError("generic completion processor missing")
    initial_count = processor.completed_count
    if type(initial_count) is not int or initial_count < 0:
        raise ValueError("generic completion processor counter invalid")
    fifo = queue.Queue(maxsize=queue_depth)
    stop = threading.Event()
    warm_done = threading.Event()
    producer_measure_ready = threading.Event()
    consumer_measure_ready = threading.Event()
    measurement_start_gate = threading.Event()
    ready = threading.Barrier(3)
    errors = []
    stamps = {}
    observed = {"warmup": 0, "measured": 0}
    total = None if duration_mode else warmup + tasks
    admitted = {"measured": 0}
    end_of_input = object()

    def concrete_outputs(outputs):
        # A queued future/handle is not an observed task or a tensor map. The
        # Generic runtime adapter owns its native wait and materialization.
        if not isinstance(outputs, Mapping) or not outputs:
            raise RuntimeError("generic completion requires materialized runtime outputs")
        for value in outputs.values():
            if callable(getattr(value, "result", None)) or callable(getattr(value, "wait", None)):
                raise RuntimeError("generic completion runtime output is still asynchronous")
        return outputs

    def producer():
        try:
            ready.wait(timeout=timeout_s)
            for index in (itertools.count() if duration_mode else range(total)):
                if index == warmup:
                    while not warm_done.wait(0.05):
                        if stop.is_set():
                            return
                    producer_measure_ready.set()
                    while not measurement_start_gate.wait(0.05):
                        if stop.is_set():
                            return
                if stop.is_set():
                    return
                if duration_mode and clock() >= stamps["admission_deadline_ns"]:
                    stamps["admission_stopped_ns"] = clock()
                    while not stop.is_set():
                        try:
                            fifo.put(end_of_input, timeout=0.05)
                            return
                        except queue.Full:
                            continue
                    return
                outputs = concrete_outputs(stage1())
                payload = clone_payload(outputs) if clone_payload is not None else outputs
                while not stop.is_set():
                    try:
                        fifo.put((index, payload), timeout=0.05)
                        if duration_mode:
                            admitted["measured"] += 1
                        break
                    except queue.Full:
                        continue
        except BaseException as error:
            errors.append(error)
            stop.set()

    def consumer():
        try:
            ready.wait(timeout=timeout_s)
            if warmup == 0:
                stamps["warmup_end_ns"] = clock()
                warm_done.set()
            for expected_index in (itertools.count() if duration_mode else range(total)):
                if expected_index == warmup:
                    consumer_measure_ready.set()
                    while not measurement_start_gate.wait(0.05):
                        if stop.is_set():
                            return
                while not stop.is_set():
                    try:
                        item = fifo.get(timeout=0.05)
                        break
                    except queue.Empty:
                        continue
                else:
                    return
                if item is end_of_input:
                    stamps["measurement_end_ns"] = clock()
                    fifo.task_done()
                    if progress is not None:
                        progress("measured_completed", observed["measured"])
                    return
                index, payload = item
                if index != expected_index:
                    raise RuntimeError("generic completion task order mismatch")
                outputs = concrete_outputs(stage2(transfer(payload)))
                before = processor.completed_count
                processor.process(outputs)
                if type(processor.completed_count) is not int or processor.completed_count != before + 1:
                    raise RuntimeError("generic completion task counter mismatch")
                phase = "warmup" if index < warmup else "measured"
                observed[phase] += 1
                if index + 1 == warmup:
                    stamps["warmup_end_ns"] = clock()
                    if progress is not None:
                        progress("warmup_completed", warmup)
                    warm_done.set()
                if index + 1 == total:
                    stamps["measurement_end_ns"] = clock()
                if phase == "measured" and progress is not None and observed[phase] == tasks:
                    progress("measured_completed", observed[phase])
                fifo.task_done()
        except BaseException as error:
            errors.append(error)
            stop.set()

    threads = [threading.Thread(target=producer, name="generic-completion-p1", daemon=True),
               threading.Thread(target=consumer, name="generic-completion-p2", daemon=True)]
    for thread in threads:
        thread.start()
    deadline = time.monotonic() + timeout_s
    try:
        ready.wait(timeout=timeout_s)
        while not (producer_measure_ready.is_set() and consumer_measure_ready.is_set()):
            if errors or time.monotonic() >= deadline:
                stop.set()
                break
            stop.wait(0.01)
        if not stop.is_set():
            if progress is not None:
                progress("measured_started", 0)
            # Match the Native FIFO boundary: both workers are ready after the
            # complete warmup drain; include their release and pipeline fill.
            stamps["measurement_start_ns"] = clock()
            if duration_mode:
                stamps["admission_deadline_ns"] = stamps["measurement_start_ns"] + int(duration_s * 1e9)
            measurement_start_gate.set()
        while any(thread.is_alive() for thread in threads):
            if errors or time.monotonic() >= deadline:
                stop.set()
                break
            for thread in threads:
                thread.join(timeout=0.05)
    finally:
        if any(thread.is_alive() for thread in threads):
            stop.set()
            for thread in threads:
                thread.join(timeout=1.0)
    if errors or any(thread.is_alive() for thread in threads) or stop.is_set():
        reason = str(errors[0]) if errors else "workers unfinished"
        failure = RuntimeError(f"generic completion failed after {observed}: {reason}")
        failure.partial_report = {
            "producer": "generic_split", "status": "failed",
            "requested_warmup_count": warmup, "warmup_completed_count": observed["warmup"],
            "requested_task_count": tasks, "completed_task_count": observed["measured"],
            "processor_count_delta": processor.completed_count - initial_count, **stamps,
        }
        raise failure from (errors[0] if errors else None)
    report = {
        "producer": "generic_split", "status": "ok",
        "measurement_boundary": ("prepared_input_before_p1_to_completed_pipeline_drain" if duration_mode
                                 else "prepared_input_before_p1_to_last_completed_task"),
        "clock": "perf_counter_ns", "fill_and_drain_included": True,
        "preprocessing_timed": False, "handoff_timed": True,
        "preprocessing_scope": "image_decode_resize_letterbox_before_prepared_input",
        "prepared_input_adapter_timed": True,
        "postprocess_timed": True, "output_materialization_timed": True,
        "worker_count": 2, "queue_depth": queue_depth,
        "handoff_payload_copy": clone_payload is not None,
        "requested_warmup_count": warmup, "warmup_completed_count": observed["warmup"],
        "requested_task_count": tasks, "completed_task_count": observed["measured"],
        "processor_count_delta": processor.completed_count - initial_count,
        **stamps,
    }
    elapsed_ns = stamps.get("measurement_end_ns", 0) - stamps.get("measurement_start_ns", 0)
    report["measured_makespan_seconds"] = elapsed_ns / 1e9
    report["completed_task_fps"] = observed["measured"] * 1e9 / elapsed_ns if elapsed_ns > 0 else None
    if duration_mode:
        report.update(measurement_mode="generic_completed_task_energy", requested_duration_s=float(duration_s),
                      admitted_task_count=admitted["measured"], completed_work_units=observed["measured"],
                      energy_work_units_source="completed_work_units", energy_work_units_exact=True,
                      admission_stopped_at_deadline=True, accepted_work_drained=True)
        validate_split_completion_duration(report, duration_s=duration_s)
    else:
        validate_split_completion(report, warmup=warmup, tasks=tasks)
    return report


def validate_split_completion_duration(report, *, duration_s):
    """A duration capture counts every admitted task only after successful drain."""
    if (not isinstance(report, Mapping) or report.get("producer") != "generic_split"
            or report.get("status") != "ok" or report.get("measurement_mode") != "generic_completed_task_energy"
            or report.get("measurement_boundary") != "prepared_input_before_p1_to_completed_pipeline_drain"
            or report.get("requested_task_count") is not None
            or report.get("requested_duration_s") != duration_s):
        raise ValueError("generic duration completion scope mismatch")
    count = report.get("completed_work_units")
    if type(count) is not int or count <= 0 or any(report.get(key) != count or type(report.get(key)) is not int
            for key in ("completed_task_count", "admitted_task_count", "processor_count_delta")):
        raise ValueError("generic duration completion count mismatch")
    if any(type(report.get(k)) is not int or report[k] != 0 for k in ("requested_warmup_count", "warmup_completed_count")):
        raise ValueError("generic duration warmup must be zero")
    stamps = [report.get(k) for k in ("warmup_end_ns", "measurement_start_ns", "admission_deadline_ns",
                                    "admission_stopped_ns", "measurement_end_ns")]
    if (any(type(x) is not int or x <= 0 for x in stamps) or stamps != sorted(stamps)
            or stamps[2] != stamps[1] + int(duration_s * 1e9)):
        raise ValueError("generic duration deadline/drain boundary mismatch")
    seconds = (stamps[-1] - stamps[1]) / 1e9
    for key, value in (("measured_makespan_seconds", seconds), ("completed_task_fps", count / seconds)):
        if type(report.get(key)) not in (int, float) or not math.isfinite(report[key]) or not math.isclose(report[key], value, rel_tol=1e-12):
            raise ValueError("generic duration count/time rate mismatch")
    if (report.get("clock") != "perf_counter_ns" or report.get("worker_count") != 2
            or report.get("preprocessing_timed") is not False
            or report.get("energy_work_units_source") != "completed_work_units"
            or any(report.get(k) is not True for k in ("fill_and_drain_included", "handoff_timed", "postprocess_timed",
                "output_materialization_timed", "energy_work_units_exact", "accepted_work_drained", "admission_stopped_at_deadline"))):
        raise ValueError("generic duration operation contract mismatch")
    return True


def validate_split_completion(report, *, warmup=100, tasks=1000):
    """Reject partial, enqueue-only and contradictory completion reports."""
    if not isinstance(report, Mapping) or report.get("producer") != "generic_split" or report.get("status") != "ok":
        raise ValueError("generic split completion evidence missing")
    counts = {"requested_warmup_count": warmup, "warmup_completed_count": warmup,
              "requested_task_count": tasks, "completed_task_count": tasks,
              "processor_count_delta": warmup + tasks}
    if any(type(report.get(key)) is not int or report[key] != expected for key, expected in counts.items()):
        raise ValueError("generic split completion count mismatch")
    stamps = [report.get(key) for key in ("warmup_end_ns", "measurement_start_ns", "measurement_end_ns")]
    if (any(type(value) is not int or value <= 0 for value in stamps)
            or stamps != sorted(stamps) or stamps[-1] <= stamps[-2]):
        raise ValueError("generic split completion timer boundary mismatch")
    seconds = (stamps[-1] - stamps[-2]) / 1e9
    for key, expected in (("measured_makespan_seconds", seconds), ("completed_task_fps", tasks / seconds)):
        value = report.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0 or not math.isclose(value, expected, rel_tol=1e-12):
            raise ValueError("generic split completion rate mismatch")
    if (report.get("measurement_boundary") != "prepared_input_before_p1_to_last_completed_task"
            or report.get("clock") != "perf_counter_ns" or report.get("worker_count") != 2
            or any(report.get(key) is not True for key in ("fill_and_drain_included", "handoff_timed", "postprocess_timed", "output_materialization_timed"))
            or report.get("preprocessing_timed") is not False):
        raise ValueError("generic split completion scope mismatch")
    return True


def valid_split_completion_repetitions(result, config, *, result_root=None):
    """Shared runtime/importer reentry check; preserve valid partial repetitions."""
    if not isinstance(result, Mapping) or not str(result.get("execution_id") or ""):
        raise ValueError("generic split completion execution identity missing")
    if (result.get("schema_version") != 1 or result.get("measurement_mode") != "generic_completed_task"
            or result.get("task") != config.get("task") or result.get("original_run") != config.get("original_run")):
        raise ValueError("generic split completion result scope mismatch")
    for key, expected in (("repetitions", 3), ("warmup_count", 100), ("task_count", 1000)):
        if key in config and (type(config[key]) is not int or config[key] != expected):
            raise ValueError("generic split completion configured work budget mismatch")
    for key in ("identity", "input_binding", "completion_contract", "measurement_boundary",
                "expected_runtime_artifacts", "functional_reference"):
        if not config.get(key) or result.get(key) != config[key]:
            raise ValueError(f"generic split completion {key} mismatch")
    valid = {}
    for repetition in result.get("repetitions", []):
        if not isinstance(repetition, Mapping):
            raise ValueError("generic split repetition malformed")
        if repetition.get("status") != "ok":
            continue
        validate_split_completion(repetition)
        index = repetition.get("repetition_index")
        if type(index) is not int or index not in (1, 2, 3) or not str(repetition.get("execution_id") or ""):
            raise ValueError("generic split repetition identity invalid")
        control = repetition.get("functional_control") or {}
        if (control.get("status") != "passed" or control.get("checked_outside_timer") is not True
                or control.get("reference_binding") != config["functional_reference"]):
            raise ValueError("generic split functional control missing or foreign")
        if repetition.get("runtime_artifacts") != config["expected_runtime_artifacts"]:
            raise ValueError("generic split runtime artifacts mismatch")
        original_cfg = config.get("original_run_cfg") or {}
        expected_depth = config.get("queue_depth", original_cfg.get("throughput_queue_depth", 2))
        if repetition.get("queue_depth") != expected_depth:
            raise ValueError("generic split queue depth mismatch")
        if original_cfg and repetition.get("handoff_payload_copy") != (original_cfg.get("streaming_impl", "auto") == "generic"):
            raise ValueError("generic split handoff copying mismatch")
        if not str(repetition.get("runtime_instance_id") or "") or type(repetition.get("pid")) is not int or repetition["pid"] <= 0:
            raise ValueError("generic split independent runtime identity missing")
        if result_root is not None:
            _validate_split_completion_samples(repetition, config, result_root)
        if index in valid and valid[index] != repetition:
            raise ValueError("generic split conflicting duplicate repetition")
        valid[index] = dict(repetition)
    if result.get("process_repetition_policy") == "separate_processes":
        identities = []
        for row in valid.values():
            ticks, boot = row.get("process_start_ticks"), row.get("boot_id")
            if type(ticks) is not int or ticks <= 0 or not isinstance(boot, str) or not boot:
                raise ValueError("generic split process start identity missing")
            identities.append((boot, row["pid"], ticks))
        if len(set(identities)) != len(identities):
            raise ValueError("generic split repetitions share a process")
    if len({row["runtime_instance_id"] for row in valid.values()}) != len(valid):
        raise ValueError("generic split repetitions share a runtime instance")
    return [valid[index] for index in sorted(valid)]


def _validate_split_completion_samples(repetition, config, result_root):
    """Validate the small persisted last-task samples, without any inference."""
    import hashlib
    import json
    from pathlib import Path
    import numpy as np
    root = Path(result_root).resolve()

    def bound_path(descriptor):
        path = Path(str(descriptor.get("path") or ""))
        if not str(descriptor.get("path") or "") or path.is_absolute() or ".." in path.parts:
            raise ValueError("generic split output sample path invalid")
        resolved = (root / path).resolve()
        if not resolved.is_relative_to(root) or not resolved.is_file():
            raise ValueError("generic split output sample missing or escaped")
        digest = hashlib.sha256()
        with resolved.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != descriptor.get("sha256"):
            raise ValueError("generic split output sample changed")
        return resolved

    if (repetition.get("output_sample_relation") != "last_measured_completed_task"
            or repetition.get("output_samples_written_outside_timer") is not True):
        raise ValueError("generic split output observation relation missing")
    samples = repetition.get("raw_output_samples") or {}
    references = config["functional_reference"].get("outputs") or {}
    if not samples or set(samples) != set(references):
        raise ValueError("generic split output sample endpoint mismatch")
    for name, descriptor in samples.items():
        array = np.load(bound_path(descriptor), allow_pickle=False)
        if (not isinstance(array, np.ndarray) or list(array.shape) != descriptor.get("shape")
                or str(array.dtype) != descriptor.get("dtype")
                or descriptor.get("shape") != references[name].get("shape")
                or descriptor.get("dtype") != references[name].get("dtype")
                or not np.isfinite(array).all()):
            raise ValueError("generic split output sample tensor mismatch")
    completed_path = bound_path(repetition.get("completed_result_artifact") or {})
    if json.loads(completed_path.read_text(encoding="utf-8")) != repetition.get("last_completed_result"):
        raise ValueError("generic split completed result sample mismatch")
