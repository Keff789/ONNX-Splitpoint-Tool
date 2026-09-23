"""Bounded offline bootstrap execution using the existing owned process pool.

Only JSON descriptors and draw ranges cross the task queue. The registered
PCG64/int64 plan is generated once and mapped read-only by private workers.
Completed absolute reference vectors are admitted before candidate work.
"""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, wait
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np

from .quality_cache import canonical_json, json_fingerprint
from .quality_statistics import WorkerProgress, numerical_environment


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reachable_bytes(value, seen=None):
    """Count resident prepared objects once; mapped plan pages are separate."""
    seen = set() if seen is None else seen
    if id(value) in seen:
        return 0
    seen.add(id(value))
    if isinstance(value, np.memmap):
        return sys.getsizeof(value)
    total = sys.getsizeof(value)
    if isinstance(value, np.ndarray):
        if value.base is not None:
            total += reachable_bytes(value.base, seen)
    elif isinstance(value, dict):
        total += sum(reachable_bytes(k, seen) + reachable_bytes(v, seen) for k, v in value.items())
    elif isinstance(value, (list, tuple, set, frozenset)):
        total += sum(reachable_bytes(v, seen) for v in value)
    elif hasattr(value, "__dict__"):
        total += reachable_bytes(vars(value), seen)
    return total


def spool_payload(root, key, payload):
    from .quality_service import _atomic_write_json
    path = Path(root) / "payloads" / (key + ".json")
    _atomic_write_json(path, payload)
    return {"path": str(path), "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size, "key": key,
            "admission_memory_bytes": 12 * path.stat().st_size + 2 * len(payload["image_ids"]) * int(payload["repetitions"]) * 8,
            "request_id": payload.get("request_id"),
            "_statistics": payload.get("_statistics", {})}


def load_payload(descriptor):
    path = Path(descriptor["path"])
    if path.stat().st_size != descriptor["size_bytes"] or file_sha256(path) != descriptor["sha256"]:
        raise ValueError("statistics payload descriptor integrity mismatch")
    payload = json.loads(path.read_text())
    if payload["evaluation_fingerprint"] != descriptor["key"]:
        raise ValueError("statistics payload descriptor identity mismatch")
    return payload


def prepare_plan(root, payload, *, check_cancelled=lambda: None):
    from .quality_service import deterministic_resample_plan, _atomic_write_json, _reference_store_file_lock
    n, repetitions = len(payload["image_ids"]), int(payload["repetitions"])
    contract = {"schema": "paired-int64-plan-v1", "seed_schema": payload["seed_schema"],
                "n": n, "B": repetitions, "numpy": np.__version__, "dtype": "int64"}
    key = json_fingerprint(contract)
    root = Path(root) / "plans"
    root.mkdir(parents=True, exist_ok=True)
    path, meta = root / (key + ".npy"), root / (key + ".json")
    with _reference_store_file_lock(root, key, check_cancelled=check_cancelled):
        valid = False
        try:
            saved = json.loads(meta.read_text())
            valid = saved["contract"] == contract and saved["sha256"] == file_sha256(path)
            if valid:
                mapped = np.load(path, mmap_mode="r", allow_pickle=False)
                valid = mapped.shape == (repetitions, n) and mapped.dtype == np.dtype("int64")
                del mapped
        except (OSError, ValueError, KeyError, TypeError):
            pass
        if not valid:
            check_cancelled()
            # Retain the exact historical RNG call, including its int64 dtype.
            plan = deterministic_resample_plan(image_count=n, repetitions=repetitions, seed=int(payload["seed"]))
            temporary = root / (key + f".{os.getpid()}.tmp")
            with temporary.open("wb") as stream:
                np.save(stream, plan, allow_pickle=False)
                stream.flush()
                os.fsync(stream.fileno())
            del plan
            check_cancelled()
            os.replace(temporary, path)
            saved = {"contract": contract, "sha256": file_sha256(path)}
            _atomic_write_json(meta, saved)
    return {"path": str(path), "sha256": saved["sha256"], "key": key,
            "shape": [repetitions, n], "dtype": "int64", "reused": valid}


def reference_key(payload, plan):
    return json_fingerprint({"schema": "absolute-reference-vector-v2",
        "reference": payload["reference_predictions_sha256"], "image_ids": payload["image_ids"],
        "annotations": payload["annotations"], "algorithm": payload["algorithm_version"],
        "factory": payload["evaluator_factory"], "plan": plan["sha256"],
        "environment": numerical_environment()})


def standard_family(payload):
    factory = str(payload.get("evaluator_factory", "")).replace(":", ".")
    for name in ("classification", "detection"):
        if factory == f"onnx_splitpoint_tool.quality_metrics.{name}_quality_evaluator":
            return name
    return None


def _absolute(evaluator, family, weights, side):
    if family == "classification":
        return evaluator.evaluate_absolute(weights, side)
    values = evaluator.evaluate_sides(weights)[0]
    return {name: {"value": value} for name, value in zip(
        ("primary", "guardrails.ap50", "guardrails.ap75"), values)}


class _PairFromReference:
    def __init__(self, evaluator, family, reference, identity, config):
        self.evaluator, self.family, self.reference = evaluator, family, reference
        self.identity, self.config, self.position = identity, config, 0
        self.hit = True

    def evaluate(self, weights):
        ref = self.reference["point"] if self.position == 0 else self.reference["values"][self.position - 1]
        cand = ref if self.identity else _absolute(self.evaluator, self.family, weights, "candidate")
        self.position += 1
        from .quality_metrics import _margin
        parts = {}
        for name, item in ref.items():
            metric = ("top1_accuracy" if self.family == "classification" else "coco_ap_50_95") if name == "primary" else name.split(".", 1)[1]
            component = {"metric": metric, "reference": item["value"], "candidate": cand[name]["value"],
                "delta": cand[name]["value"] - item["value"],
                "margin": _margin(self.config, "non_inferiority_margin" if name == "primary" else name + "_margin")}
            if self.family == "classification":
                if item["sample_count"] != cand[name]["sample_count"]:
                    raise ValueError("reference and candidate sample counts differ")
                component.update(reference_hits=item["hits"], candidate_hits=cand[name]["hits"],
                                 sample_count=item["sample_count"])
            parts[name] = component
        return {"primary": parts.pop("primary"), "guardrails": {k.split(".", 1)[1]: v for k, v in parts.items()}}


# One context per owned worker. It retains only this request's bound payload,
# mapped plan and one private side evaluator, never an unbounded request cache.
_CONTEXT = None
_CONTEXTS = {}


def _worker_context(descriptor, phase, reference_descriptor):
    global _CONTEXT
    from .quality_service import _load_evaluator_factory, _evaluate_metric, _prediction_identity_is_bound
    key = (descriptor["sha256"], descriptor["plan"]["sha256"])
    load_started = time.perf_counter()
    loading_s = 0.0
    evictions = 0
    _CONTEXT = _CONTEXTS.pop(key, None)
    if _CONTEXT is None:
        # At most two request-private evaluators share one byte budget.
        if len(_CONTEXTS) >= 2:
            _CONTEXTS.pop(next(iter(_CONTEXTS)))
            evictions += 1
        payload = load_payload(descriptor)
        plan_desc = descriptor["plan"]
        if file_sha256(plan_desc["path"]) != plan_desc["sha256"]:
            raise ValueError("statistics plan integrity mismatch")
        plan = np.load(plan_desc["path"], mmap_mode="r", allow_pickle=False)
        if list(plan.shape) != plan_desc["shape"] or plan.dtype != np.dtype("int64") or plan.flags.writeable:
            raise ValueError("statistics plan shape/dtype/read-only mismatch")
        _CONTEXT = {"key": key, "payload": payload, "plan": plan, "phase": None}
        loading_s = time.perf_counter() - load_started
    ctx = _CONTEXT
    hit = ctx["phase"] == phase
    started = time.perf_counter()
    preparation = {"matching_s": 0.0, "matched_sides": 0, "data_preparation_s": 0.0}
    prepare_s = point_s = reference_load_s = accounting_s = 0.0
    if not hit:
        evictions += int(ctx["phase"] is not None)
        ctx.pop("evaluator", None)
        ctx.pop("point", None)
        payload = ctx["payload"]
        family = standard_family(payload) if descriptor["reuse_reference"] else None
        config = {"metric_gate_config": payload.get("metric_gate_config", {}),
                  "value_field": payload.get("value_field", "value"),
                  "image_ids": payload["image_ids"], "statistics_engine": "optimized_coco_v1"}
        side = "reference" if phase == "reference" else "candidate"
        identity = family and phase == "candidate" and _prediction_identity_is_bound(payload)
        sides = () if identity else (side,)
        if family:
            config.update(coco_sides=sides, metric_sides=sides)
        factory = _load_evaluator_factory(payload["evaluator_factory"])
        evaluator = factory(payload["reference_records"], payload["candidate_records"], payload["annotations"], config)
        prepare_s = time.perf_counter() - started
        preparation.update(getattr(evaluator, "preparation_observation", {}))
        ones = np.ones(len(payload["image_ids"]), dtype=np.float64)
        point_started = time.perf_counter()
        if phase == "reference":
            point = _absolute(evaluator, family, ones, "reference")
        else:
            if family:
                reference_load_started = time.perf_counter()
                reference = _read_reference(reference_descriptor, len(ctx["plan"]), family)
                reference_load_s = time.perf_counter() - reference_load_started
                if reference is None:
                    raise ValueError("completed reference component unavailable")
                evaluator = _PairFromReference(evaluator, family, reference, identity, config)
                point_started = time.perf_counter()
            point = _evaluate_metric(evaluator, ones)
        point_s = time.perf_counter() - point_started
        ctx.update(evaluator=evaluator, point=point, phase=phase, family=family)
        accounting_started = time.perf_counter()
        size = reachable_bytes(ctx)
        accounting_s = time.perf_counter() - accounting_started
        limit = payload["_statistics"]["prepared_cache_limit_mib"] * 1024**2
        if size > limit:
            _CONTEXT = None
            raise MemoryError(f"prepared statistics context {size} exceeds configured {limit} bytes")
        ctx["resident_bytes"] = size
    limit = ctx["payload"]["_statistics"]["prepared_cache_limit_mib"] * 1024**2
    while _CONTEXTS and (len(_CONTEXTS) >= 2 or
            sum(c["resident_bytes"] for c in _CONTEXTS.values()) + ctx["resident_bytes"] > limit):
        _CONTEXTS.pop(next(iter(_CONTEXTS)))
        evictions += 1
    _CONTEXTS[key] = ctx
    return ctx, {"prepared_cache_hit": hit, "prepare_count": int(not hit),
                 "prepared_cache_evictions": evictions, "prepared_cache_bytes": sum(c["resident_bytes"] for c in _CONTEXTS.values()),
                 "prepare_s": prepare_s, "point_s": point_s, "loading_s": loading_s,
                 "reference_load_s": reference_load_s, "cache_accounting_s": accounting_s,
                 **preparation,
                 "mapped_plan_bytes": ctx["plan"].nbytes}


def evaluate_block(descriptor, phase, start, stop, reference_descriptor=None):
    from .quality_service import _evaluate_payload_shard_impl
    block_started = time.perf_counter()
    minimal = {"request_id": descriptor.get("request_id"), "evaluation_fingerprint": descriptor["key"],
               "_statistics": descriptor["_statistics"]}
    with WorkerProgress(minimal, start, stop) as progress:
        progress.emit("loading", component_phase=phase)
        progress.emit("preparing")
        ctx, observation = _worker_context(descriptor, phase, reference_descriptor)
        payload, plan, evaluator = ctx["payload"], ctx["plan"], ctx["evaluator"]
        if phase == "reference":
            progress.emit("accumulating")
            accumulation_started = time.perf_counter()
            values = []
            for indices in plan[start:stop]:
                weights = np.bincount(indices, minlength=plan.shape[1]).astype(np.float64, copy=False)
                values.append(_absolute(evaluator, ctx["family"], weights, "reference"))
                progress.emit("accumulating", len(values))
            result = {"repetition_offset": start, "repetitions": stop-start, "point": ctx["point"], "values": values}
            result["statistics_observation"] = {"accumulation_s": time.perf_counter()-accumulation_started}
            progress.emit("completed", stop-start)
        else:
            if isinstance(evaluator, _PairFromReference):
                evaluator.position = start + 1
            # Canonical absolute values are checkpoint data, independently of
            # diagnostic capture. Legacy delta-only custom factories retain
            # their ordinary evaluation API but cannot create such checkpoints.
            points = [ctx["point"]["primary"], *ctx["point"]["guardrails"].values()]
            absolute_available = all(p.get("reference") is not None and p.get("candidate") is not None for p in points)
            payload = {**payload, "_statistics": {**payload["_statistics"], "capture_draws": absolute_available}}
            result = _evaluate_payload_shard_impl(payload, plan[start:stop], shard_index=start,
                repetition_offset=start, progress=progress, prepared=ctx)
        result["statistics_observation"] = {**result.get("statistics_observation", {}), **observation,
            "worker_pid": os.getpid(), "component_phase": phase, "checkpoint_hit": False,
            "ipc_wait_s": max(0.0, block_started-descriptor.get("submitted_monotonic", block_started)),
            "repetition_start": start, "repetition_stop": stop}
        result["block_identity"] = checkpoint_contract(descriptor, phase, reference_descriptor)
        return result


def _valid_absolute(row, family):
    names = {"primary", "guardrails.top5_accuracy"} if family == "classification" else {"primary", "guardrails.ap50", "guardrails.ap75"}
    if not isinstance(row, dict) or set(row) != names:
        return False
    for item in row.values():
        if not isinstance(item, dict) or type(item.get("value")) not in (int, float) or not math.isfinite(item["value"]):
            return False
        if family == "classification":
            if type(item.get("hits")) is not int or type(item.get("sample_count")) is not int:
                return False
            if not 0 <= item["hits"] <= item["sample_count"] or item["sample_count"] <= 0:
                return False
            if item["value"] != item["hits"] / item["sample_count"]:
                return False
    return True


def _read_reference(descriptor, repetitions, family):
    try:
        saved = json.loads(Path(descriptor["path"]).read_text())
        body = saved["body"]
        if (saved["sha256"] != json_fingerprint(body) or body["schema"] != "absolute-reference-vector-v2"
                or body["key"] != descriptor["key"] or len(body["values"]) != repetitions
                or not all(_valid_absolute(v, family) for v in [body["point"], *body["values"]])):
            return None
        return body
    except (OSError, ValueError, KeyError, TypeError):
        return None


def checkpoint_contract(descriptor, phase, reference_descriptor):
    return {"schema": "paired-statistics-block-v1", "evaluation": descriptor["key"],
            "plan": descriptor["plan"]["sha256"],
            "phase": phase, "reference": (reference_descriptor or {}).get("key"),
            "B": descriptor["plan"]["shape"][0], "n": descriptor["plan"]["shape"][1]}


def write_checkpoint(path, contract, shard):
    from .quality_service import _atomic_write_json
    body = {"contract": contract, "shard": shard,
            "undefined_mask": [v is None for v in shard.get("relative_loss_draws", [])]}
    _atomic_write_json(path, {**body, "sha256": json_fingerprint(body)})


def read_checkpoints(root, contract, payload):
    """Only checksum-valid complete ranges are candidates; overlap is an error."""
    result = []
    for path in sorted(Path(root).glob("*.json")):
        try:
            saved = json.loads(path.read_text())
            body = {k: saved[k] for k in ("contract", "shard", "undefined_mask")}
            if saved["sha256"] != json_fingerprint(body) or body["contract"] != contract:
                continue
            shard = body["shard"]
            if shard.get("block_identity") != contract:
                continue
            start, count = shard["repetition_offset"], shard["repetitions"]
            if type(start) is not int or type(count) is not int or start < 0 or count <= 0 or start + count > contract["B"]:
                continue
            if contract["phase"] == "reference":
                family = standard_family(payload)
                if len(shard["values"]) != count or not all(_valid_absolute(v, family) for v in [shard["point"], *shard["values"]]):
                    continue
            else:
                if body["undefined_mask"] != [v is None for v in shard["relative_loss_draws"]]:
                    continue
                # Reuse the strict merge validator on this locally rebased complete range.
                from .quality_service import _combine_evaluation_shards
                local = {**payload, "repetitions": count}
                local_shard = {k: v for k, v in shard.items() if k != "block_identity"}
                _combine_evaluation_shards(local, [{**local_shard, "repetition_offset": 0}], elapsed_s=0, workers_requested=1)
            shard["statistics_observation"] = {**shard.get("statistics_observation", {}),
                "checkpoint_hit": True, "prepare_count": 0, "prepared_cache_hit": False,
                "historical_worker_pid": shard.get("statistics_observation", {}).get("worker_pid"),
                "worker_pid": None}
            result.append(shard)
        except (OSError, ValueError, TypeError, KeyError, IndexError):
            continue
    result.sort(key=lambda s: s["repetition_offset"])
    end = 0
    for shard in result:
        if shard["repetition_offset"] < end:
            raise ValueError("checkpoint ranges overlap or duplicate")
        end = shard["repetition_offset"] + shard["repetitions"]
    return result


def missing_ranges(completed, repetitions, block):
    cursor = 0
    for start, stop in [(r["repetition_offset"], r["repetition_offset"]+r["repetitions"]) for r in completed] + [(repetitions, repetitions)]:
        while cursor < start:
            following = min(cursor + block, start)
            yield cursor, following
            cursor = following
        cursor = stop


def run_phase(service, group, descriptor, phase, reference_descriptor=None):
    contract = checkpoint_contract(descriptor, phase, reference_descriptor)
    root = service.cache.root / "checkpoints" / group.key
    if phase == "reference":
        root = root / "reference"
    enabled = service.statistics["checkpoint_blocks"]
    check = lambda: service._check_block_cancelled(group)
    with service.pause_gate.activity("checkpoint_load:" + group.key, check_cancelled=check):
        completed = read_checkpoints(root, contract, group.payload) if enabled else []
    reused = sum(r["repetitions"] for r in completed)
    service._statistics_phase(phase, checkpoint_reused_draws=reused, newly_completed_draws=0,
                              requested_draws=contract["B"])
    ranges = iter(missing_ranges(completed, contract["B"], service.statistics["block_repetitions"]))
    pending = {}
    following = next(ranges, None)
    try:
        while pending or following is not None:
            check()
            while following is not None and len(pending) < service.workers:
                # Never wait for a permit while holding a writer lock. A short
                # admission wait lets us publish completed blocks and release
                # their permits, including during a quietness drain.
                token = service.pause_gate.acquire_activity("block:" + group.key,
                    cpu=1, check_cancelled=check, timeout=0.02)
                if token is None:
                    break
                if not service._worker_slots.acquire(blocking=False):
                    service.pause_gate.release_activity(token)
                    break
                start, stop = following
                try:
                    with service._lock:
                        check()
                        future = service._executor.submit(evaluate_block,
                            {**descriptor, "submitted_monotonic": time.perf_counter()}, phase, start, stop, reference_descriptor)
                        with group.lock:
                            group.workers.append(future)
                            group.pending += 1
                    pending[future] = (start, stop, token)
                    following = next(ranges, None)
                except BaseException:
                    service._worker_slots.release()
                    service.pause_gate.release_activity(token)
                    raise
                future.add_done_callback(lambda done: service._block_worker_done(group, done))
            if not pending:
                continue
            done, _ = wait(pending, timeout=0.05, return_when=FIRST_COMPLETED)
            for future in done:
                start, stop, token = pending.pop(future)
                try:
                    shard = dict(future.result())
                    check()
                    skipped = bool(shard.get("skipped_reason")) and not payload_reporting(group.payload)
                    if (shard.get("block_identity") != contract or shard.get("repetition_offset") != start
                            or shard.get("repetitions") != (0 if skipped else stop-start)):
                        raise ValueError("worker returned an incompatible block range")
                    if skipped:
                        following = None
                    if enabled and not skipped:
                        if phase != "reference" and "absolute_bootstrap" not in shard:
                            shard["statistics_observation"]["checkpoint_unavailable_reason"] = "custom_evaluator_has_no_absolute_metrics"
                        else:
                            write_checkpoint(root / f"{start}-{stop}.json", contract, shard)
                    completed.append(shard)
                    service._statistics_phase(phase, newly_completed_draws=sum(r["repetitions"] for r in completed)-reused)
                finally:
                    # The admitted block includes its required checkpoint
                    # flush. A done Future alone is not confirmed quietness.
                    service.pause_gate.release_activity(token)
    finally:
        for future, (_, _, token) in pending.items():
            try:
                future.result()
            except BaseException:
                pass
            finally:
                service.pause_gate.release_activity(token)
    return sorted(completed, key=lambda s: s["repetition_offset"])


def payload_reporting(payload):
    from .quality_service import _reporting
    return bool(_reporting(payload))


def execute_request(service, group, descriptor):
    from .quality_service import _atomic_write_json, _reference_store_file_lock, _combine_evaluation_shards
    from .quality_statistics_config import resource_budget
    payload = group.payload
    service._statistics_phase("checking_resources")
    budget = resource_budget()
    plan_bytes = len(payload["image_ids"]) * int(payload["repetitions"]) * 8
    resident = reachable_bytes(payload)
    # Includes parent preparation/serialization, each private prepared bound,
    # mapped plan pages, RNG creation and worker transport/workspace reserve.
    required = (4 * resident + 2 * plan_bytes + service.workers *
                (service.statistics["prepared_cache_limit_mib"] * 1024**2 + 64 * 1024**2))
    if required > budget["available_memory_bytes"]:
        raise MemoryError(f"statistics admission needs {required} bytes; available {budget['available_memory_bytes']}")
    if shutil.disk_usage(service.cache.root).free < 2 * plan_bytes + descriptor["size_bytes"]:
        raise OSError("insufficient scratch disk for statistics plan and atomic publication")
    check = lambda: service._check_block_cancelled(group)
    service._statistics_phase("loading_plan")
    plan_started = time.perf_counter()
    family = standard_family(payload) if payload_reporting(payload) else None
    with service.pause_gate.activity("quality_plan:" + group.key, check_cancelled=check):
        descriptor = {**descriptor, "plan": prepare_plan(service.cache.root, payload, check_cancelled=check),
                      "reuse_reference": bool(family)}
    plan_prepare_s = time.perf_counter() - plan_started
    reference_desc = None
    reference_hit = False
    reference_observations = []
    reference_started = time.perf_counter()
    if family:
        key = reference_key(payload, descriptor["plan"])
        reference_desc = {"key": key, "path": str(service.cache.root / "reference_vectors" / (key + ".json"))}
        service._statistics_phase("waiting_reference")
        with _reference_store_file_lock(Path(reference_desc["path"]).parent, key, check_cancelled=check):
            with service.pause_gate.activity("reference_load:" + group.key, check_cancelled=check):
                reference = _read_reference(reference_desc, int(payload["repetitions"]), family)
            reference_hit = reference is not None
            if reference is None:
                blocks = run_phase(service, group, descriptor, "reference", reference_desc)
                with service.pause_gate.activity("reference_publish:" + group.key, check_cancelled=check):
                    cursor, values, point = 0, [], blocks[0]["point"]
                    for block in blocks:
                        if block["repetition_offset"] != cursor or canonical_json(block["point"]) != canonical_json(point):
                            raise ValueError("reference ranges/points do not form the registered plan")
                        values.extend(block["values"])
                        cursor += block["repetitions"]
                    if cursor != int(payload["repetitions"]):
                        raise ValueError("incomplete reference vector")
                    reference = {"schema": "absolute-reference-vector-v2", "key": key, "point": point, "values": values}
                    check()
                    _atomic_write_json(Path(reference_desc["path"]), {"body": reference, "sha256": json_fingerprint(reference)})
                    reference_observations = [b["statistics_observation"] for b in blocks]
            del reference
    reference_phase_s = time.perf_counter() - reference_started
    candidate_started = time.perf_counter()
    blocks = run_phase(service, group, descriptor, "candidate", reference_desc)
    candidate_phase_s = time.perf_counter() - candidate_started
    service._statistics_phase("merging")
    with service.pause_gate.activity("quality_merge:" + group.key, check_cancelled=check), service._lock:
        check()
        merge_started = time.perf_counter()
        result = _combine_evaluation_shards(payload, blocks, elapsed_s=time.perf_counter()-group.started,
                                            workers_requested=service.workers_requested)
        if not blocks[0].get("skipped_reason"):
            result["bootstrap_workers_effective"] = min(service.workers, len(blocks))
        result["statistics_observation"] = {"engine": "optimized_coco_v1",
            "merge_s": time.perf_counter()-merge_started,
            "plan_prepare_s": plan_prepare_s, "reference_phase_s": reference_phase_s,
            "candidate_phase_s": candidate_phase_s,
            "shards": [b["statistics_observation"] for b in blocks],
            "reference_shards": reference_observations, "reference_cache_hit": reference_hit,
            "checkpoint_unavailable_reason": next((b["statistics_observation"]["checkpoint_unavailable_reason"]
                for b in blocks if b["statistics_observation"].get("checkpoint_unavailable_reason")), ""),
            "checkpoint_reused_draws": sum(b["repetitions"] for b in blocks if b["statistics_observation"]["checkpoint_hit"]),
            "draws_recomputed": sum(b["repetitions"] for b in blocks if not b["statistics_observation"]["checkpoint_hit"]),
            "resource_budget": budget, "admission_memory_bytes": required, "payload_resident_bytes": resident,
            "plan_bytes": plan_bytes, "plan_reused": descriptor["plan"]["reused"],
            "workers_requested": service.workers_requested, "workers_effective": service.workers}
        if service.statistics.get("capture_draws"):
            _atomic_write_json(service.cache.root / "draws" / (group.key + ".json"),
                {"payload_identity": group.key, "shards": blocks})
        service.cache.put(group.key, result)
        service._complete_block_result(group, result)
        return result
