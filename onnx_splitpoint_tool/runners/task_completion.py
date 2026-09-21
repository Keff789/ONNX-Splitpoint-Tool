"""Measured completion for generic runners, independent of Native reports."""
from __future__ import annotations
import time
from typing import Mapping

PRODUCERS = {"generic_deepx_full", "generic_hailo_full", "generic_ort_full"}

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
