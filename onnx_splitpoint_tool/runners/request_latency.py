"""Bounded request pairing. No I/O, device waits or workload work in the recorder.

Request IDs belong to a single measured repeat; warmup never enters this buffer.
All statistics are calculated after the existing workers have drained.
"""
from __future__ import annotations

import math
import os
import statistics
import time
from typing import Any, Mapping

SCHEMA = "onnx-splitpoint/request-latency"
START = "prepared_input_before_first_admission"
COMPLETED = "host_result_after_required_transfer_sync_and_task_postprocess"
HOST_OUTPUT = "host_model_outputs_without_task_postprocess"


def _stats(values: list[float]) -> dict[str, Any]:
    ordered = sorted(values)
    def quantile(q: float) -> float | None:
        if not ordered:
            return None
        position = (len(ordered) - 1) * q
        lo = int(position)
        hi = min(lo + 1, len(ordered) - 1)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)
    return {
        "mean_ms": statistics.fmean(ordered) if ordered else None,
        "p50_ms": quantile(.5), "p95_ms": quantile(.95),
        "min_ms": ordered[0] if ordered else None,
        "max_ms": ordered[-1] if ordered else None,
        "quantile_method": "linear_(n-1)*q",
    }


def validate_latency(value: Any) -> dict[str, Any]:
    """Recompute from preserved pairs; never trust rounded/published quantiles."""
    empty = dict(schema=SCHEMA, schema_version=1, status="unavailable",
                 reason="request_timestamps_missing", unit="ms", count=0,
                 expected_count=0, task_complete=False, **_stats([]))
    if not isinstance(value, Mapping) or not value:
        return empty
    result = {**empty, **value}
    if type(result.get("expected_count")) is not int or result["expected_count"] < 0:
        result["expected_count"] = 0
    result.update(_stats([]))
    result.update(status="invalid", reason="request_contract_invalid", unit="ms", count=0)
    expected = value.get("expected_count")
    if (value.get("schema") != SCHEMA or value.get("schema_version") != 1
            or type(expected) is not int or expected < 0
            or value.get("timestamp_unit") != "ns"
            or value.get("clock") not in {"perf_counter_ns", "steady_clock_ns"}
            or not isinstance(value.get("clock_domain"), str) or not value["clock_domain"]
            or value.get("admission_wait_included") is not True
            or value.get("warmup_included") is not False
            or not isinstance(value.get("start_anchor"), str) or not value["start_anchor"].strip()
            or not isinstance(value.get("end_anchor"), str) or not value["end_anchor"].strip()
            or value.get("start_endpoint") != START
            or value.get("end_endpoint") not in {COMPLETED, HOST_OUTPUT}
            or type(value.get("task_complete")) is not bool
            or value["task_complete"] != (value["end_endpoint"] == COMPLETED)):
        return result
    if value.get("disabled_reason"):
        result.update(status="unavailable", reason=value["disabled_reason"])
        return result
    errors = value.get("errors", {})
    if not isinstance(errors, Mapping) or any(type(v) is not int or v < 0 for v in errors.values()):
        return result
    errors = dict(errors)
    records = value.get("pairs")
    if not isinstance(records, list):
        return result
    seen: set[int] = set()
    values: list[float] = []
    missing_start = missing_end = invalid_pair = duplicate = unknown = 0
    for record in records:
        if not isinstance(record, (list, tuple)) or len(record) != 3:
            invalid_pair += 1
            continue
        identity, start, end = record
        if type(identity) is not int or not 0 <= identity < expected:
            unknown += 1
            continue
        if identity in seen:
            duplicate += 1
            continue
        seen.add(identity)
        missing_start += start is None
        missing_end += end is None
        if start is None or end is None:
            continue
        if type(start) is not int or type(end) is not int or start < 0 or end < start:
            invalid_pair += 1
            continue
        values.append((end - start) / 1_000_000)
    missing_ids = expected - len(seen)
    derived = dict(missing_ids=missing_ids, missing_starts=missing_start,
                   missing_completions=missing_end, invalid_pairs=invalid_pair,
                   duplicate_ids=duplicate, unknown_ids=unknown)
    errors.update({key: max(errors.get(key, 0), count) for key, count in derived.items()})
    complete = expected > 0 and len(values) == expected and not any(errors.values())
    result.update(errors=errors, count=len(values),
                  status="complete" if complete else "invalid",
                  reason="" if complete else "request_pairs_incomplete_or_invalid",
                  semantics=("prepared_input_to_completed_task" if value["task_complete"]
                             else "prepared_input_to_host_outputs"))
    if complete:
        result.update(_stats(values))
    return result


class RequestLatency:
    """One bounded pair of arrays. IDs, not input hashes or slot addresses."""
    def __init__(self, expected_count: int, *, task_complete: bool,
                 start_anchor: str, end_anchor: str, enabled: bool = True):
        if type(expected_count) is not int or expected_count < 0:
            raise ValueError("expected_count must be a nonnegative frame budget")
        self.expected_count = expected_count
        self.enabled = bool(enabled)
        self.starts: list[int | None] = [None] * (expected_count if enabled else 0)
        self.ends: list[int | None] = [None] * (expected_count if enabled else 0)
        self.errors = {"duplicate_starts": 0, "duplicate_completions": 0,
                       "invalid_ids": 0, "invalid_timestamps": 0, "clock_mismatches": 0}
        self.clock_domain = f"perf_counter_ns:process:{os.getpid()}"
        self.task_complete = bool(task_complete)
        self.start_anchor, self.end_anchor = start_anchor, end_anchor

    def _record(self, target: list[int | None], request_id: int, stamp: int | None,
                clock_domain: str | None, duplicate_key: str) -> None:
        if not self.enabled:
            return
        if type(request_id) is not int or not 0 <= request_id < self.expected_count:
            self.errors["invalid_ids"] += 1
            return
        if clock_domain is not None and clock_domain != self.clock_domain:
            self.errors["clock_mismatches"] += 1
            return
        if stamp is None:
            stamp = time.perf_counter_ns()
        if type(stamp) is not int or stamp < 0:
            self.errors["invalid_timestamps"] += 1
            return
        if target[request_id] is not None:
            self.errors[duplicate_key] += 1
            return
        target[request_id] = stamp

    def start(self, request_id: int, stamp: int | None = None, *, clock_domain: str | None = None) -> None:
        self._record(self.starts, request_id, stamp, clock_domain, "duplicate_starts")

    def complete(self, request_id: int, stamp: int | None = None, *, clock_domain: str | None = None) -> None:
        self._record(self.ends, request_id, stamp, clock_domain, "duplicate_completions")

    def report(self) -> dict[str, Any]:
        return validate_latency({
            "schema": SCHEMA, "schema_version": 1, "unit": "ms", "timestamp_unit": "ns",
            "clock": "perf_counter_ns", "clock_domain": self.clock_domain,
            "expected_count": self.expected_count, "task_complete": self.task_complete,
            "start_endpoint": START, "end_endpoint": COMPLETED if self.task_complete else HOST_OUTPUT,
            "start_anchor": self.start_anchor, "end_anchor": self.end_anchor,
            "admission_wait_included": True, "warmup_included": False,
            "errors": dict(self.errors),
            "disabled_reason": "duration_workload_not_instrumented" if not self.enabled else "",
            "pairs": [[i, a, b] for i, (a, b) in enumerate(zip(self.starts, self.ends))],
        })

    def failure(self, message: str) -> RuntimeError:
        error = RuntimeError(message)
        error.request_latency = self.report()
        return error


def aggregate_latency(records: list[Mapping[str, Any]], *, requested: int | None = None) -> dict[str, Any]:
    """Pool complete same-semantic repeats, never average their percentiles."""
    requested = len(records) if requested is None else requested
    repeats = []
    ids = set()
    reason = "" if requested == len(records) and records else "repetition_set_incomplete"
    contracts = set()
    all_values: list[float] = []
    for index, record in enumerate(records):
        identity = str(record.get("repetition_id") or record.get("runtime_instance_id") or "")
        if len(records) > 1 and (not identity or identity in ids):
            reason = "repetition_identity_missing_or_duplicate"
        ids.add(identity)
        raw = validate_latency(record.get("request_latency"))
        repeats.append({"repetition_id": identity, "repetition_index": index + 1, "request_latency": raw})
        contracts.add(tuple(raw.get(key) for key in ("semantics", "start_endpoint", "end_endpoint",
                         "start_anchor", "end_anchor", "admission_wait_included", "warmup_included")))
        if raw["status"] != "complete":
            reason = "repetition_latency_incomplete_or_invalid"
        else:
            all_values.extend((end - start) / 1_000_000 for _, start, end in raw["pairs"])
    if len(contracts) > 1:
        reason = "repetition_latency_semantics_mismatch"
    first = repeats[0]["request_latency"] if repeats else {}
    result = {key: first.get(key) for key in (
        "schema", "schema_version", "semantics", "unit", "start_endpoint", "end_endpoint",
        "task_complete", "admission_wait_included", "warmup_included")}
    result.update(status="invalid" if reason else "complete", reason=reason,
                  aggregation="pooled_complete_request_pairs", repetitions=repeats,
                  repetition_count_requested=requested,
                  count=sum(r["request_latency"]["count"] for r in repeats),
                  expected_count=sum(r["request_latency"]["expected_count"] for r in repeats),
                  **_stats([] if reason else all_values))
    if repeats and all(r["request_latency"]["reason"] == "request_timestamps_missing" for r in repeats):
        result.update(status="unavailable", reason="request_timestamps_missing")
    return result


def latency_fields(payload: Mapping[str, Any]) -> dict[str, Any]:
    endpoints = payload.get("endpoint_results")
    source = (endpoints.get("completed_task") or {}) if isinstance(endpoints, Mapping) and endpoints else payload
    records = source.get("repetition_records") or source.get("repetition_evidence")
    if isinstance(records, list) and records:
        value = aggregate_latency(records, requested=source.get("repetition_count_requested", len(records)))
    else:
        raw = source.get("request_latency")
        if isinstance(raw, Mapping) and raw.get("aggregation") == "pooled_complete_request_pairs":
            value = aggregate_latency(raw.get("repetitions") or [], requested=raw.get("repetition_count_requested", 0))
        else:
            value = validate_latency(raw)
    complete = value["status"] == "complete"
    task = complete and value.get("task_complete") is True
    reason = value.get("reason", "") if not complete else ("" if task else "task_postprocess_not_in_measured_path")
    return {
        "request_latency": value, "request_latency_status": "complete" if task else "unavailable" if complete or value["status"] == "unavailable" else "invalid",
        "request_latency_unavailable_reason": reason,
        "request_latency_semantics": value.get("semantics", ""),
        "request_latency_count": value["count"], "request_latency_expected_count": value["expected_count"],
        **{f"request_latency_{key}": value.get(key) if task else None
           for key in ("mean_ms", "p50_ms", "p95_ms", "min_ms", "max_ms")},
        **{f"host_output_latency_{key}": value.get(key) if complete and not task else None
           for key in ("mean_ms", "p50_ms", "p95_ms")},
        "request_latency_start_endpoint": value.get("start_endpoint", ""),
        "request_latency_end_endpoint": value.get("end_endpoint", ""),
        "request_latency_aggregation": value.get("aggregation", "single_repeat_request_pairs"),
    }
