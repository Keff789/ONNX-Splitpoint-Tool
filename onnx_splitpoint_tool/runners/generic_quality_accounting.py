"""Account explicit Generic quality captures without inventing performance rows.

Capture completion is distinct from the later central Quality decision. This
module starts no process and performs no inference or metric calculation.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def _file(path):
    path = Path(path)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size}


def _stage(value):
    if isinstance(value, Mapping):
        value = value.get("hw_arch") or value.get("provider") or value.get("backend") or value.get("type")
    return str(value or "").lower()


def _binding(row):
    """The profile, rather than a row-local flag, authorizes the exception."""
    cases = row.get("case_ids")
    paths = row.get("generic_quality_completion_configs")
    hashes = row.get("generic_quality_completion_config_sha256")
    sources = row.get("generic_quality_completion_sources")
    if (row.get("quality_source_only") is not True or row.get("variants") != ["composed"]
            or row.get("type") not in {"mixed_backend", "matrix"}
            or _stage(row.get("stage1")) not in {"hailo8", "hailo10", "deepx_m1"}
            or _stage(row.get("stage2")) != "tensorrt"
            or not isinstance(cases, list) or not cases or len(cases) != len(set(cases))
            or any(not isinstance(c, str) or re.fullmatch(r"b[0-9]+", c) is None for c in cases)
            or any(not isinstance(v, Mapping) or set(v) != set(cases) for v in (paths, hashes, sources))
            or not row.get("hardware_setup_id")):
        raise ValueError("generic_quality_accounting_explicit_scope_invalid")
    for case in cases:
        source = sources[case]
        if (paths[case] != f"quality_runtime/{case}.json"
                or not isinstance(hashes[case], str) or re.fullmatch(r"[a-f0-9]{64}", hashes[case]) is None
                or not isinstance(source, Mapping) or source.get("sha256") != hashes[case]
                or not Path(str(source.get("path") or "")).is_absolute()
                or type(source.get("size_bytes")) is not int or source["size_bytes"] < 1):
            raise ValueError("generic_quality_accounting_source_binding_invalid")
    return {"id": row.get("id"), "stage1": _stage(row["stage1"]), "stage2": _stage(row["stage2"]),
            "hardware_setup_id": row["hardware_setup_id"], "case_ids": cases,
            "generic_quality_completion_configs": paths,
            "generic_quality_completion_config_sha256": hashes,
            "generic_quality_completion_sources": sources,
            "validation_images": row.get("validation_images"),
            "validation_max_images": row.get("validation_max_images")}


def authorized_runs(plan, profile):
    quality = (profile or {}).get("quality_gate") or {}
    if (quality.get("statistics") or {}).get("execution_location") != "central_management":
        return []
    authorities = {}
    for row in (profile or {}).get("run_profiles", []):
        if isinstance(row, Mapping) and "generic_quality_completion_configs" in row:
            binding = _binding(row)
            if binding["id"] in authorities:
                raise ValueError("generic_quality_accounting_duplicate_profile_run")
            authorities[binding["id"]] = binding
    result = []
    for row in plan.get("runs", plan.get("planned_runs", [])):
        if isinstance(row, Mapping) and row.get("id") in authorities:
            if _binding(row) != authorities[row["id"]]:
                raise ValueError("generic_quality_accounting_profile_plan_drift")
            if any(existing["id"] == row["id"] for existing in result):
                raise ValueError("generic_quality_accounting_duplicate_plan_run")
            result.append(deepcopy(dict(row)))
    return result


def make_projection(rows):
    if not rows:
        return {}
    value = {"schema": "onnx-splitpoint/generic-quality-only-performance-projection",
             "schema_version": 1, "runs": deepcopy(rows)}
    return {**value, "projection_sha256": _hash(value)}


def projected_runs(plan):
    value = deepcopy(plan.get("generic_quality_only_projection") or {})
    seal = value.pop("projection_sha256", None)
    if not value:
        return []
    if (value.get("schema") != "onnx-splitpoint/generic-quality-only-performance-projection"
            or value.get("schema_version") != 1 or seal != _hash(value)
            or not isinstance(value.get("runs"), list) or not value["runs"]):
        raise ValueError("generic_quality_accounting_projection_invalid")
    if len({row.get("id") for row in value["runs"] if isinstance(row, Mapping)}) != len(value["runs"]):
        raise ValueError("generic_quality_accounting_duplicate_projected_run")
    for row in value["runs"]:
        _binding(row)
    return value["runs"]


def filter_performance_scope(required, rows):
    """Exclude only the exact already-authorized Generic quality identities."""
    keys = {(str(row["hardware_setup_id"]), str(row["id"]), str(case))
            for row in rows for case in row["case_ids"]}
    return [row for row in required if not (
        str(row.get("variant")) in {"split", "composed"}
        and (str(row.get("setup_id") or row.get("expected_setup_id") or ""),
             str(row.get("run_id") or row.get("backend") or ""), str(row.get("case_id") or "")) in keys)]


def verify_capture(request_path, config, *, config_path, eval_run_id=None, source_run_id=None):
    """Verify genuine portable completion bytes before central metric evaluation."""
    from .generic_quality_completion import FIELD, completed_quality_export_fields
    request_path = Path(request_path).resolve()
    request = json.loads(request_path.read_text())
    descriptor = request.get("candidate") or {}
    relative = Path(str(descriptor.get("path") or ""))
    candidate_path = (request_path.parent / relative).resolve()
    if (not relative.parts or relative.is_absolute() or not candidate_path.is_relative_to(request_path.parent)
            or _file(candidate_path)["sha256"] != descriptor.get("sha256")
            or candidate_path.stat().st_size != descriptor.get("size_bytes")):
        raise ValueError("generic_quality_capture_candidate_bytes_invalid")
    candidate = json.loads(candidate_path.read_text())
    proof = request.get(FIELD)
    producer = request.get("producer_identity") or {}
    if (not isinstance(proof, Mapping) or candidate.get(FIELD) != proof
            or candidate.get("producer_identity") != producer
            or completed_quality_export_fields(proof, candidate.get("records") or [], producer).get(FIELD) != proof
            or proof.get("base_completion_contract") != config["completion_contract"]
            or proof.get("identity") != config["identity"]
            or proof.get("image_ids") != config["expected_image_ids"]):
        raise ValueError("generic_quality_capture_completion_binding_invalid")
    if proof.get("config_file", {}).get("sha256") != _file(config_path)["sha256"]:
        raise ValueError("generic_quality_capture_selected_config_changed")
    raw_config = json.loads(Path(config_path).read_text())
    expected_bridge = deepcopy(raw_config.get("existing_part2_bridge"))
    actual_bridge = deepcopy(proof.get("existing_part2_bridge"))
    for bridge in (expected_bridge, actual_bridge):
        if isinstance(bridge, Mapping) and isinstance(bridge.get("binding_source"), Mapping):
            bridge["binding_source"] = {key: bridge["binding_source"].get(key)
                for key in ("sha256", "size_bytes", "pointer")}
    if (actual_bridge != expected_bridge or proof.get("existing_part2_bridge_runtime_policy") != (
            raw_config.get("original_run_cfg") if expected_bridge is not None else None)):
        raise ValueError("generic_quality_capture_existing_bridge_reference_changed")
    observed_sources = proof.get("runtime_source_files") or []
    if len(observed_sources) != len(raw_config.get("runtime_source_files") or []):
        raise ValueError("generic_quality_capture_selected_source_closure_changed")
    for expected_source in raw_config["runtime_source_files"]:
        token = expected_source["path"]
        matches = [source for source in observed_sources if (
            source.get("path") == token if Path(token).is_absolute()
            else str(source.get("path") or "").endswith("/" + token))]
        if (len(matches) != 1 or any(matches[0].get(key) != expected_source.get(key)
                for key in ("sha256", "size_bytes") if key in expected_source)):
            raise ValueError("generic_quality_capture_selected_source_closure_changed")
    expected = config["expected_runtime_artifacts"]
    actual = proof.get("runtime_artifacts") or {}
    if (set(actual) != set(expected) or any(
            any(actual[role].get(k) != expected[role].get(k) for k in ("sha256", "size_bytes"))
            for role in expected)):
        raise ValueError("generic_quality_capture_artifact_binding_invalid")
    for envelope in (request, candidate):
        if (envelope.get("task") != "detection" or envelope.get("variant") != "composed"
                or envelope.get("record_count") != len(config["expected_image_ids"])):
            raise ValueError("generic_quality_capture_envelope_invalid")
    for key in ("model_id", "case_id", "setup_id"):
        if request.get(key) != config["identity"][key] or producer.get(key) != config["identity"][key]:
            raise ValueError("generic_quality_capture_identity_invalid")
    if (request.get("expected_image_ids") != config["expected_image_ids"]
            or request.get("performance_claims_emitted") is not False
            or producer.get("performance_claims_emitted") is not False
            or producer.get("execution_role") != "generic_composed_quality_candidate"
            or (eval_run_id is not None and any(v.get("eval_run_id") != eval_run_id for v in (request, producer)))
            or (source_run_id is not None and any(v.get("source_run_id") != source_run_id for v in (request, producer)))):
        raise ValueError("generic_quality_capture_producer_scope_invalid")
    return {"request": _file(request_path), "candidate": _file(candidate_path),
            "identity": deepcopy(config["identity"]), "image_count": len(config["expected_image_ids"]),
            "completion_evidence_sha256": proof["evidence_sha256"],
            "producer_identity_sha256": producer["producer_identity_sha256"],
            "performance_claims_emitted": False, "central_quality_decision": "pending"}


def _request_descriptor(report):
    source = (report.get("task_quality_input_requests_by_variant") or {}).get("composed") or {}
    return source.get("request") or source


def verify_child_report(report, request_sha256):
    source = _request_descriptor(report)
    if (report.get("quality_source_only") is not True
            or report.get("performance_measurement_executed") is not False
            or report.get("measured_variants") != []
            or (report.get("variant_status") or {}).get("composed") != "ok"
            or str(source.get("sha256") or "").removeprefix("sha256:") != request_sha256):
        raise ValueError("generic_quality_capture_child_measurement_or_request_invalid")
    def nonzero(value):
        if isinstance(value, Mapping):
            return any(nonzero(item) for key, item in value.items()
                       if key not in {"measurement_endpoint", "endpoint_mode"})
        if isinstance(value, list):
            return any(nonzero(item) for item in value)
        if value is None or value == "":
            return False
        return type(value) not in (int, float) or value != 0
    if (nonzero(report.get("timing_ms"))
            or any(nonzero(value) for key, value in report.items()
                   if key.lower() in {"fps", "fps_makespan", "latency_ms", "latency_mean_ms", "throughput_fps"})
            or any(nonzero(value) for key, value in (report.get("throughput") or {}).items()
                   if key in {"fps_makespan", "fps_cycle_est", "makespan_ms", "measured_frames", "frame_count_total", "latency_mean_ms"})):
        raise ValueError("generic_quality_capture_child_performance_claim_present")
    return True


def collect_capture(report_path, config, *, config_path, run_id, eval_run_id):
    report_path = Path(report_path)
    report = json.loads(report_path.read_text())
    request_descriptor = _request_descriptor(report)
    request_path = Path(str(request_descriptor.get("path") or ""))
    if not request_path.is_absolute():
        request_path = report_path.parent / request_path
    receipt = verify_capture(request_path, config, config_path=config_path, eval_run_id=eval_run_id, source_run_id=run_id)
    verify_child_report(report, receipt["request"]["sha256"])
    return {"schema": "onnx-splitpoint/generic-quality-source-capture", "schema_version": 1,
            **receipt, "child_report": _file(report_path), "quality_source_only": True,
            "run_id": run_id, "measured_variants": [], "performance_measurement_executed": False}


def copied_capture_summary(copied, rows, *, suite_dir, model_id, setup_id, eval_run_id):
    """Join copied child reports to exact request bytes, never a status flag."""
    from .generic_quality_completion import load_quality_completion_config
    selected = [r for r in rows if r.get("hardware_setup_id") == setup_id]
    if not selected:
        return {"requested": False, "status": "not_requested", "count": 0, "expected": 0}
    requests = {Path(r["destination"]).resolve() for r in copied
                if r.get("kind") == "central_quality_input" and Path(r["destination"]).name.endswith("_request.json")}
    reports = {Path(r["destination"]).resolve() for r in copied if r.get("kind") == "case_validation_report"}
    expected = sum(len(r["case_ids"]) for r in selected)
    receipts, errors, used = [], [], set()
    for row in selected:
        for case in row["case_ids"]:
            try:
                path = Path(suite_dir) / row["generic_quality_completion_configs"][case]
                if _file(path)["sha256"] != row["generic_quality_completion_config_sha256"][case]:
                    raise ValueError("generic_quality_capture_config_changed")
                # DUT artifact/input paths need not exist on the controller.
                # The DUT loader already checked those bytes before execution;
                # validate their exact bound proof and rehash local shipped code.
                config = load_quality_completion_config(path, check_files=False)
                for source in config["runtime_source_files"]:
                    actual_source = _file(source["path"])
                    if any(actual_source[key] != source[key] for key in ("sha256", "size_bytes") if key in source):
                        raise ValueError("generic_quality_capture_controller_source_changed")
                if config["identity"].get("model_id") != model_id:
                    raise ValueError("generic_quality_capture_model_changed")
                matching = [p for p in requests if json.loads(p.read_text()).get("case_id") == case
                            and json.loads(p.read_text()).get("source_run_id") == row["id"]]
                if len(matching) != 1:
                    raise ValueError("generic_quality_capture_request_cardinality_invalid")
                receipt = verify_capture(matching[0], config, config_path=path, eval_run_id=eval_run_id, source_run_id=row["id"])
                # Diagnostic mirrors are byte-identical representations, not extra captures.
                child = [p for p in reports if str(_request_descriptor(json.loads(p.read_text())).get("sha256") or "").removeprefix("sha256:") == receipt["request"]["sha256"]]
                if not child or len({_file(p)["sha256"] for p in child}) != 1:
                    raise ValueError("generic_quality_capture_child_report_cardinality_invalid")
                verify_child_report(json.loads(child[0].read_text()), receipt["request"]["sha256"])
                used.add(matching[0]); receipts.append({**receipt, "child_report": _file(child[0])})
            except (ValueError, OSError, KeyError, TypeError) as error:
                errors.append(f"{row['id']}/{case}:{type(error).__name__}:{error}")
    if requests - used:
        errors.append("generic_quality_capture_unexpected_request")
    return {"requested": True, "status": "verified_exact" if not errors and len(receipts) == expected else "incomplete",
            "count": len(receipts), "expected": expected, "evidence": receipts, "errors": errors,
            "performance_claims_emitted": False, "central_quality_decision": "pending"}


def normalized_contract(payload):
    """Recognize only the explicit, rowless Generic capture accounting result."""
    if not isinstance(payload, Mapping):
        return False
    try:
        rows = projected_runs(payload)
    except (ValueError, TypeError, KeyError):
        return False
    expected = sum(len(row["case_ids"]) for row in rows)
    return bool(expected > 0 and payload.get("results") == []
        and type(payload.get("result_count")) is int and payload["result_count"] == 0
        and payload.get("status") == "quality_evidence_only_complete"
        and payload.get("performance_matrix_applicable") is False
        and payload.get("quality_evidence_only_complete") is True and payload.get("matrix_complete") is True
        and type(payload.get("expected_generic_quality_count")) is int
        and payload["expected_generic_quality_count"] == expected
        and type(payload.get("quality_evidence_count")) is int
        and payload["quality_evidence_count"] == expected
        and type(payload.get("expected_full_quality_count")) is int
        and payload["expected_full_quality_count"] == 0)


def central_quality_complete(normalized, summary, *, plan, profile, run_root, model_id):
    """Prove completed normal Generic Quality, without claiming NPU performance."""
    from .generic_quality_completion import load_quality_completion_config, validate_completed_quality_binding
    if not normalized_contract(normalized) or not isinstance(summary, Mapping):
        return False
    try:
        from ..accuracy_gates import AccuracyGatePolicy
        policy_sha = AccuracyGatePolicy.from_mapping((profile or {}).get("quality_gate") or {}).sha256()
        declared_reference = (((profile or {}).get("quality_gate") or {}).get("management_reference") or {}).get("existing_reference")
        planned_reference = plan.get("management_cpu_reference")
        if (not isinstance(declared_reference, Mapping) or not isinstance(planned_reference, Mapping)
                or {k:v for k,v in declared_reference.items() if k != "reference_path"}
                != {k:v for k,v in planned_reference.items() if k != "reference_path"}):
            return False
        for reference in (declared_reference, planned_reference):
            reference_file = _file(reference["reference_path"])
            if (reference_file["sha256"] != reference.get("reference_sha256")
                    or reference_file["size_bytes"] != reference.get("reference_size_bytes")):
                return False
        rows = authorized_runs(plan, profile)
        if [_binding(row) for row in rows] != [_binding(row) for row in projected_runs(normalized)]:
            return False
        results = [r for r in summary.get("results", []) if isinstance(r, Mapping) and r.get("model_id") == model_id]
        expected = {(row["hardware_setup_id"], row["id"], case): row for row in rows for case in row["case_ids"]}
        if len(results) != len(expected) or not expected:
            return False
        seen = set()
        for result in results:
            # The normal result reader canonicalizes source_run_id; the signed
            # producer preserves the exact original dispatch identity.
            producer = result.get("producer_identity") or {}
            key = (producer.get("setup_id"), producer.get("source_run_id"), producer.get("case_id"))
            from ..workflow.runner import EvaluationWorkflowRunner
            if (any(result.get(field) != producer.get(field) for field in ("model_id", "case_id", "setup_id", "eval_run_id"))
                    or result.get("execution_role") != "generic_composed_quality_candidate"
                    or EvaluationWorkflowRunner._quality_source_run_token(result.get("source_run_id"))
                    != EvaluationWorkflowRunner._quality_source_run_token(producer.get("source_run_id"))):
                return False
            if key not in expected or key in seen or result.get("performance_claims_emitted") is not False:
                return False
            row = expected[key]; case = key[2]
            source = row["generic_quality_completion_sources"][case]
            config_path = Path(source["path"])
            if any(_file(config_path)[k] != source[k] for k in ("sha256", "size_bytes")):
                return False
            config = load_quality_completion_config(config_path, check_files=False)
            request_path = Path(str(result.get("source_request") or ""))
            request_path = request_path if request_path.is_absolute() else Path(run_root)/request_path
            if not request_path.resolve().is_relative_to(Path(run_root).resolve()):
                return False
            request = json.loads(request_path.read_text())
            token = lambda value: str(value or "").removeprefix("sha256:")
            evaluated_reference = result.get("management_cpu_reference") or {}
            if (any(evaluated_reference.get(key) != planned_reference.get(key)
                    for key in ("reference_sha256", "reference_size_bytes", "quality_contract"))
                    or evaluated_reference.get("semantic_reference_only") is not True
                    or evaluated_reference.get("provider") != "onnxruntime_cpu"
                    or result.get("producer_identity") != request.get("producer_identity")
                    or result.get("runtime_precision_identity") != config["identity"]["precision"]
                    or result.get("runtime_precision_identity") != producer.get("runtime_precision_identity")
                    or type(result.get("n")) is not int or result["n"] != len(config["expected_image_ids"])
                    or token(request.get("policy_sha256")) != policy_sha
                    or token(result.get("policy_sha256")) != token(request.get("policy_sha256"))
                    or token((request.get("reference") or {}).get("quality_contract_sha256"))
                    != token((planned_reference.get("quality_contract") or {}).get("quality_contract_sha256"))
                    or request.get("record_count") != row.get("validation_max_images")
                    or not token(request.get("policy_sha256"))
                    or result.get("validation_image_ids_sha256") != request.get("expected_image_ids_sha256")
                    or str(result.get("quality_decision") or "").lower() not in {
                        "pass", "fail", "inconclusive", "reference_close", "accuracy_loss", "not_estimable"}):
                return False
            # Generic and the old canonical CPU adapter have independently
            # bound decoder implementations. Use the normal strict reader for
            # their shared model/dataset/input/endpoint and separate contracts.
            from ..quality_service import quality_request_from_manifest
            quality_request_from_manifest(request_path, reference_artifact={
                "path": planned_reference["reference_path"],
                "sha256": planned_reference["reference_sha256"],
                "size_bytes": planned_reference["reference_size_bytes"],
            })
            verify_capture(request_path, config, config_path=config_path,
                           eval_run_id=Path(run_root).name, source_run_id=row["id"])
            validate_completed_quality_binding(request, result, config["completion_contract"],
                config["expected_runtime_artifacts"], request_path=request_path,
                reference_descriptor={"expected_image_ids": config["expected_image_ids"]})
            seen.add(key)
        return seen == set(expected)
    except (ValueError, KeyError, TypeError, OSError):
        return False
