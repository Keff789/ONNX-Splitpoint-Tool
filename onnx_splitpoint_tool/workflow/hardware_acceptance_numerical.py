"""Bounded serial/A4 numerical admission through the normal Native owner."""
from __future__ import annotations

from contextlib import ExitStack
import copy
import inspect
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import shutil
import tarfile
import time
from typing import Mapping

from . import hardware_acceptance_native as native


SCHEMA = "onnx-splitpoint/hardware-acceptance-native-numerical/v1"


def execution_lifecycle_safety(raw):
    """Validate explicit Native worker evidence; absence proves nothing.

    This pure interpretation never observes processes or releases a resource.
    A pre-runtime result also needs normal owned transport/resource closure.
    """
    nodes, markers, marker_owners = [], [], []
    def visit(value):
        if isinstance(value, Mapping):
            nodes.append(value)
            if "deepx_execution_lifecycle" in value:
                markers.append(value["deepx_execution_lifecycle"])
                marker_owners.append(value)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    visit(raw)
    result = {"lifecycle_present": bool(markers), "pre_runtime_proven": False,
              "device_completion_proven": False, "unsafe": False, "reason": "explicit_lifecycle_missing",
              "report_contract_complete": None if not markers else True, "evidence_gaps": []}
    if not markers:
        return result
    def reject(reason):
        return {**result, "unsafe": True, "report_contract_complete": False, "reason": reason}
    for node in nodes:
        # A final Full attempt contains two independent worker lifetimes. A
        # surviving semantic marker must not hide a missing measured worker.
        attempted = node.get("repetition_count_attempted")
        composite = ("preparation_evidence" in node and "deepx_prepared_feed_benchmark" in node)
        if node.get("backend") == "native_full_deepx" and (composite or (type(attempted) is int and attempted > 0)):
            for child_key in ("preparation_evidence", "deepx_prepared_feed_benchmark"):
                child = node.get(child_key)
                if not isinstance(child, Mapping) or "deepx_execution_lifecycle" not in child:
                    return reject("full_child_lifecycle_missing:" + child_key)
    flags = ("runtime_construction_started", "runtime_construction_completed", "inference_started")
    counters = ("runtime_constructors_entered", "runtime_constructors_returned", "runtime_calls_entered", "runtime_calls_completed")
    pre, pre_owners = True, []
    # A Full job contains independently completed semantic and measured child
    # workers. Their counters differ; every child must independently be safe.
    for marker, marker_owner in zip(markers, marker_owners):
        if (not isinstance(marker, Mapping) or any(type(marker.get(key)) is not bool for key in flags)
                or any(type(marker.get(key)) is not int or marker[key] < 0 for key in counters)):
            return reject("lifecycle_exact_types_or_counters_invalid")
        ce, cr, entered, completed = (marker[key] for key in counters)
        child_pre = (marker.get("phase") == "pre_runtime_guard" and marker.get("completion") == "not_started"
                     and all(marker[key] is False for key in flags) and (ce, cr, entered, completed) == (0, 0, 0, 0))
        child_drained = (marker.get("phase") == "drained" and marker.get("completion") == "drained"
                         and all(marker[key] is True for key in flags) and ce == cr and ce > 0
                         and entered == completed and entered > 0)
        if not child_pre and not child_drained:
            return reject("runtime_entry_or_completion_unknown")
        if child_pre:
            pre_owners.append(marker_owner)
        pre = pre and child_pre
    unsafe_flags = {"unsafe_runtime_state", "deepx_unsafe_runtime_state", "unsafe_device_state", "serial_completion_unresolved"}
    pending = {"pending", "unreleased", "pending_count", "unresolved_jobs"}
    no_entry_counts = {
        "submitted", "submitted_count", "submit_attempts", "submit_failures", "wait_failures", "wait_completed",
        "accepted_work_units", "warmup_accepted_work_units", "mixed_submitted_requests", "mixed_submitted_count",
        "mixed_submit_attempt_count", "serial_reference_started_count", "serial_reference_count", "serial_reference_requests",
        "mixed_completed_count", "mixed_completions", "smoke_completed_count", "completed_frames", "completed_work_units",
        "warmup_completed_count", "warmup_completed_frames", "postprocess_completed_frames",
        "actual_pending_max", "observed_max_inflight", "deepx_actual_pending_max",
        "setup_deepx_serial_calls", "setup_trt_calls", "serial_sdk_calls_started", "serial_sdk_calls_completed",
        "serial_trt_calls_started", "serial_trt_calls_completed", "serial_runtime_calls_started", "serial_runtime_calls_completed",
        "worker_serial_runtime_calls_started", "worker_serial_runtime_calls_completed", "numerical_total_sdk_calls",
        "serial_reference_calls_started", "serial_reference_calls_completed",
    }
    def pre_counter_contradiction(value):
        if isinstance(value, Mapping):
            for key in no_entry_counts & value.keys():
                if type(value[key]) is not int or value[key] != 0:
                    return key
            for child in value.values():
                problem = pre_counter_contradiction(child)
                if problem:
                    return problem
        elif isinstance(value, list):
            for child in value:
                problem = pre_counter_contradiction(child)
                if problem:
                    return problem
        return ""
    for owner in pre_owners:
        problem = pre_counter_contradiction(owner)
        if problem:
            return reject("lifecycle_pre_child_runtime_counter:" + problem)
    gaps = []
    for node in nodes:
        for key in unsafe_flags & node.keys():
            if type(node[key]) is not bool or node[key] is not False:
                return reject("lifecycle_contradictory_unsafe_flag:" + key)
        for key in (pending | (no_entry_counts if pre else set())) & node.keys():
            if type(node[key]) is not int or node[key] != 0:
                return reject("lifecycle_contradictory_runtime_counter:" + key)
        for started, returned in (("serial_sdk_calls_started", "serial_sdk_calls_completed"),
                                  ("serial_trt_calls_started", "serial_trt_calls_completed"),
                                  ("serial_runtime_calls_started", "serial_runtime_calls_completed"),
                                  ("serial_reference_calls_started", "serial_reference_calls_completed"),
                                  ("worker_serial_runtime_calls_started", "worker_serial_runtime_calls_completed")):
            if started in node or returned in node:
                if (started == "serial_reference_calls_started" and started in node and returned not in node
                        and type(node[started]) is int and node[started] >= 0):
                    gaps.append("serial_reference_calls_completed_missing")
                    continue
                if (type(node.get(started)) is not int or type(node.get(returned)) is not int
                        or node[started] < 0 or node[started] != node[returned]):
                    return reject("lifecycle_serial_completion_contradiction:" + started)
    return {**result, "pre_runtime_proven": pre, "device_completion_proven": not gaps,
            "report_contract_complete": not gaps, "evidence_gaps": sorted(set(gaps)),
            "reason": "lifecycle_report_gap" if gaps else "explicit_pre_runtime_guard" if pre else "explicit_runtime_completion"}


def _independent_missing_reference_completion(raw):
    """Check a missing redundant reference field against its owning SDK proof.

    A sibling worker cannot cover this node. Task success/counts are irrelevant:
    a mapper may fail after the SDK returned. Missing values stay missing.
    """
    covered = []
    def visit(value, owner=None):
        if isinstance(value, Mapping):
            if "deepx_execution_lifecycle" in value:
                owner = value["deepx_execution_lifecycle"]
            if "serial_reference_calls_started" in value and "serial_reference_calls_completed" not in value:
                started = value["serial_reference_calls_started"]
                if not isinstance(owner, Mapping) or type(started) is not int or started < 0:
                    covered.append(False)
                elif owner.get("phase") == "pre_runtime_guard":
                    covered.append(started == 0)
                else:
                    covered.append(type(value.get("serial_sdk_calls_started")) is int
                                   and type(value.get("serial_sdk_calls_completed")) is int
                                   and value["serial_sdk_calls_started"] == value["serial_sdk_calls_completed"] == started
                                   and owner.get("phase") == "drained")
            for child in value.values():
                visit(child, owner)
        elif isinstance(value, list):
            for child in value:
                visit(child, owner)
    visit(raw)
    return bool(covered) and all(covered)


def device_completion_safety(raw):
    """Pure SDK completion proof, independent of report/numerical validity.

    Resource release also requires the caller's exact owned worker and process
    closure. This function performs no observation, recovery or result rewrite.
    """
    reasons, observations, proofs, gaps = [], [], [], []
    def visit(value, path="report"):
        if isinstance(value, Mapping):
            for key in ("unsafe_runtime_state", "deepx_unsafe_runtime_state", "unsafe_device_state", "serial_completion_unresolved"):
                if key in value and (type(value[key]) is not bool or value[key] is True):
                    reasons.append(path + "." + key)
            for prefix in ("serial_sdk_calls", "serial_trt_calls", "serial_runtime_calls",
                           "worker_serial_runtime_calls", "serial_reference_calls"):
                started, returned = value.get(prefix + "_started"), value.get(prefix + "_completed")
                if prefix + "_started" in value or prefix + "_completed" in value:
                    if (prefix == "serial_reference_calls" and prefix + "_completed" not in value
                            and type(started) is int and started >= 0):
                        gaps.append("serial_reference_calls_completed_missing")
                        continue
                    if type(started) is not int or type(returned) is not int or started < 0 or started != returned:
                        reasons.append(path + "." + prefix + "_completion_unproven")
                    elif started > 0:
                        proofs.append(path + "." + prefix + "_returned")
            if "submitted" in value or "wait_completed" in value:
                submitted, waited = value.get("submitted"), value.get("wait_completed")
                if type(submitted) is not int or type(waited) is not int or submitted < 0 or submitted != waited:
                    reasons.append(path + ".submitted_wait_completion_contradiction")
                if "submit_attempts" in value and (type(value["submit_attempts"]) is not int or value["submit_attempts"] != submitted):
                    reasons.append(path + ".submit_attempt_completion_contradiction")
            if "pending" in value or "unreleased" in value:
                pending, retained = value.get("pending"), value.get("unreleased")
                if type(pending) is not int or type(retained) is not int or pending != 0 or retained != 0:
                    reasons.append(path + ".pending_or_unreleased_unproven")
                else:
                    observations.append({"path": path, "pending": pending, "unreleased": retained})
            for key in ("deepx_final_drain_complete", "accepted_work_drained"):
                if value.get(key) is True:
                    proofs.append(path + "." + key)
            for key, child in value.items():
                if key not in {"native_command_contract", "full_command_contract", "source_contract",
                               "historical_full_reference", "quality_provenance", "artifacts"}:
                    visit(child, path + "." + str(key))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, path + "." + str(index))
    visit(raw)
    lifecycle = execution_lifecycle_safety(raw)
    gaps.extend(lifecycle["evidence_gaps"])
    if lifecycle["unsafe"]:
        reasons.append("report.explicit_execution_lifecycle:" + lifecycle["reason"])
    independent = False
    if gaps:
        independent = (set(gaps) == {"serial_reference_calls_completed_missing"}
                       and _independent_missing_reference_completion(raw))
        if independent:
            proofs.append("report.independent_complete_sdk_entries_waits_slots_and_trt_returns")
    elif lifecycle["pre_runtime_proven"]:
        proofs.append("report.explicit_pre_runtime_no_submit")
    elif lifecycle["device_completion_proven"]:
        proofs.append("report.explicit_execution_lifecycle_drained")
    # A queue snapshot or final-drain flag alone never establishes SDK entry
    # and return. Every explicit child lifecycle is validated independently.
    proven = (not reasons and lifecycle["lifecycle_present"]
              and (lifecycle["device_completion_proven"] or independent))
    return {"device_completion_proven": proven, "pre_runtime_proven": lifecycle["pre_runtime_proven"],
            "unsafe_reasons": reasons, "observations": observations, "drain_proofs": proofs,
            "evidence_gaps": sorted(set(gaps)), "report_contract_complete": False if gaps else lifecycle["report_contract_complete"],
            "independent_completion_proven": independent and not reasons, "execution_lifecycle": lifecycle}


def _owned_dispatch_closure(transport, registry, command, returncode, previous):
    """Join this dispatch to its ordinary SSH cleanup diagnostic and journal."""
    directory = getattr(transport, "diagnostics_dir", None)
    active = registry.active_count()
    if (not directory or type(active) is not int or active != 0
            or getattr(registry, "cancelled", True) is not False
            or getattr(registry, "configured_scope", None) is None):
        return {"proven": False, "reason": "owned_process_registry_closure_unproven"}
    matches = []
    for path in Path(directory).glob("ssh_*.json"):
        if path.is_symlink() or not path.is_file() or str(path.resolve()) in previous:
            continue
        try:
            diagnostic = native._read(path)
        except Exception:
            continue
        if diagnostic.get("command") == command:
            matches.append((path, diagnostic))
    if len(matches) != 1:
        return {"proven": False, "reason": "exact_dispatch_cleanup_diagnostic_missing"}
    path, diagnostic = matches[0]
    proven = (type(diagnostic.get("returncode")) is int and diagnostic["returncode"] == returncode
              and diagnostic.get("cleanup_proven") is True
              and diagnostic.get("target_id") == transport.host.id)
    return {"proven": proven, "reason": "owned_dispatch_cleanup_proven" if proven else "owned_dispatch_cleanup_unproven",
            "diagnostic": {"path": str(path), "sha256": native._sha(path)}}


def _owned_preparation_closure(transport, registry, previous):
    """Prove owned staging process closure, never that inference did not start.

    The caller must separately record its pre-worker phase before staging.
    Only ordinary scoped registries and this staging's new SSH receipts count.
    """
    active = registry.active_count()
    if (type(active) is not int or active != 0 or getattr(registry, "cancelled", True) is not False
            or getattr(registry, "configured_scope", None) is None):
        return {"proven": False, "reason": "owned_preparation_registry_closure_unproven"}
    from ..process_control import current_process_registry
    local = current_process_registry()
    if local is not None and not local.is_quiescent(require_nested_session_safety=True):
        return {"proven": False, "reason": "owned_preparation_local_process_closure_unproven"}
    directory = getattr(transport, "diagnostics_dir", None)
    if not directory:
        return {"proven": False, "reason": "owned_preparation_diagnostics_unavailable"}
    receipts = []
    for path in Path(directory).glob("ssh_*.json"):
        if str(path.resolve()) in previous:
            continue
        if path.is_symlink() or not path.is_file():
            return {"proven": False, "reason": "owned_preparation_diagnostic_invalid"}
        try:
            receipt = native._read(path)
        except Exception:
            return {"proven": False, "reason": "owned_preparation_diagnostic_unreadable"}
        if (receipt.get("cleanup_proven") is not True or type(receipt.get("returncode")) is not int
                or receipt.get("target_id") != transport.host.id or not receipt.get("command")):
            return {"proven": False, "reason": "owned_preparation_command_closure_unproven"}
        receipts.append({"path": str(path), "sha256": native._sha(path)})
    return {"proven": True, "reason": "owned_preparation_process_closure_proven", "diagnostics": receipts}


def _admit_numerical_attempt(prepared, context, image_count):
    """Retain same-cell attempt history without a lifetime technical-repair cap.

    A normal invocation still starts only one worker. Its repair rationale is
    reviewed before GUI Start; these original receipts document progress, not
    another retry budget or a replacement for resource/SDK completion rules.
    """
    matrix_path = native._bound_path(prepared["phase_matrix"])
    matrix = native._read(matrix_path)
    schema = matrix.get("schema")
    if schema not in {"deepx-native-a4/finite-release-acceptance/v1", "onnx-splitpoint/native-refresh-selection/v1"}:
        raise ValueError("numerical admission requires the original supported finite matrix")
    campaign = schema == "onnx-splitpoint/native-refresh-selection/v1"
    limits = matrix.get("execution_limits") or {}
    if image_count != (1 if campaign else 32):
        raise ValueError("numerical image count differs from its finite phase")
    identity = prepared["identity"]
    matches = [cell for cell in matrix.get("cells") or [] if isinstance(cell, Mapping)
               and cell.get("model_id") == identity["model"] and cell.get("setup_id") == identity["setup_id"]
               and cell.get("case_id") == identity["case"] and cell.get("backend") == identity["backend"]]
    if len(matches) != 1:
        raise ValueError("numerical cell is not uniquely selected in the original finite matrix")
    root = matrix_path.parent / "hardware_runs" / (identity["model"] + "__" + identity["setup_id"])
    current_run = Path(context["run_dir"]).resolve()
    component = "__".join(identity[key] for key in ("model", "setup_id", "case"))
    output = Path(prepared["output"]).resolve()
    if current_run.parent != root.resolve() or output != current_run / "native" / component:
        raise ValueError("numerical output is outside its original finite phase/cell scope")
    prior = []
    for cell_output in sorted(root.glob("*/native/" + component)):
        if cell_output.resolve() == output:
            continue
        result_path = cell_output / "NUMERICAL_RESULT.json"
        command_path, dispatch_path = cell_output / "numerical/COMMAND.json", cell_output / "numerical/DISPATCH.json"
        if not any(path.exists() for path in (result_path, command_path, dispatch_path)):
            continue
        if cell_output.is_symlink() or not (cell_output / "PREPARED.json").is_file():
            raise ValueError("prior numerical attempt is not bound to its original preparation")
        old_prepared = native._read(cell_output / "PREPARED.json")
        if old_prepared.get("phase_matrix") != prepared["phase_matrix"]:
            raise ValueError("same-cell numerical attempt has a different finite matrix binding")
        if not result_path.is_file() or result_path.is_symlink():
            raise RuntimeError("prior numerical dispatch has no terminal result; normal recovery required")
        old = native._read(result_path)
        if old.get("attempt_admitted") is False:
            if (old.get("fresh_hardware_executed") is not False or command_path.exists() or dispatch_path.exists()):
                raise ValueError("unadmitted numerical result contradicts existing dispatch evidence")
            continue
        if (old.get("schema") != SCHEMA or old.get("identity") != identity
                or old_prepared.get("identity") != identity
                or old.get("phase_matrix") != prepared["phase_matrix"]
                or old.get("source_contract_sha256") != prepared["source_contract"]["contract_sha256"]
                or old.get("deepx_runtime_mode") != "A4"):
            raise ValueError("prior numerical attempt source/identity/phase binding differs")
        if old.get("status") == "numerical_passed":
            raise RuntimeError("prior numerical PASS must be reused through normal Resume, not recaptured")
        origin = cell_output.parent.parent
        manifest = native._read(origin / "run_manifest.json")
        closure = native._read(origin / "reports/artifact_index_closure.json")
        if (old.get("status") != "failed" or old.get("attempt_state") not in {None, "terminal"}
                or manifest.get("run_id") != origin.name or manifest.get("status") not in {"failed", "partial", "cancelled", "ok"}
                or closure.get("status") != "pass" or closure.get("run_id") != origin.name
                or closure.get("workflow_status") != manifest.get("status")):
            raise RuntimeError("prior numerical attempt is not a closed normal workflow")
        record = {"path": str(result_path), "sha256": native._sha(result_path),
                  "worker_dispatched": old.get("fresh_hardware_executed"),
                  "prepared": {"path": str(cell_output / "PREPARED.json"), "sha256": native._sha(cell_output / "PREPARED.json")}}
        deployment = cell_output / "RUNTIME_DEPLOYMENT.json"
        if deployment.is_file():
            if deployment.is_symlink():
                raise ValueError("prior numerical deployment receipt must be its original regular file")
            record["runtime_deployment"] = {"path": str(deployment), "sha256": native._sha(deployment)}
        prior.append(record)
    current_prepared = output / "PREPARED.json"
    current = {"identity": dict(identity), "source_contract_sha256": prepared["source_contract"]["contract_sha256"],
               "runtime_deployment_path": str(output / "RUNTIME_DEPLOYMENT.json")}
    if current_prepared.is_file():
        if current_prepared.is_symlink() or native._read(current_prepared) != prepared:
            raise ValueError("current numerical preparation differs from its original normal checkpoint")
        current["prepared"] = {"path": str(current_prepared), "sha256": native._sha(current_prepared)}
    return {"prior_attempts": prior, "attempt_number": len(prior) + 1,
            "attempt_limit": None, "remaining_attempts_after_admission": None,
            "historical_matrix_attempt_limit": limits.get("numerical_attempts_per_cell_max", 2),
            "technical_repair_policy": "documented repair before normal GUI Start; no lifetime cap; accepted numerical PASS must be reused",
            "technical_repair_after_prior_failure": bool(prior), "current_preparation": current}


def validate_numerical_raw(raw, prepared, expected_images=32):
    """Revalidate actual worker counters, tensor comparisons and SDK drain."""
    identity = prepared["identity"]
    if identity["backend"] not in {"deepx_to_trt", "native_full_deepx"}:
        raise ValueError("numerical acceptance requires Native DeepX")
    if type(expected_images) is not int or not 1 <= expected_images <= 32:
        raise ValueError("numerical acceptance requires one through 32 frozen images")
    if not isinstance(raw, Mapping) or raw.get("ok") is not True:
        raise ValueError("numerical worker did not complete successfully")
    full = identity["backend"] == "native_full_deepx"
    if full:
        rows = raw.get("rows") or []
        if len(rows) != 1 or not isinstance(rows[0], Mapping):
            raise ValueError("numerical Full report requires exactly one row")
        row = rows[0]
        if any(row.get(key) != value for key, value in identity.items()):
            raise ValueError("numerical Full worker identity differs")
        from ..resources.remote_scripts.native_producer_energy_plan import _verify_full_command_contract
        current = row.get("full_command_contract") or {}
        contract, reason = _verify_full_command_contract(current, expected_identity={
            **identity, "remote_root": current.get("root"),
            "remote_tool_dir": prepared.get("remote_tool_dir"),
        })
    else:
        row = raw
        contract, reason = native.verify_native_command_contract(row.get("native_command_contract"), expected_identity=identity)
    if contract is None:
        raise ValueError("numerical actual command contract rejected: " + reason)
    options = contract.get("runtime_options") or {}
    if (row.get("ok") is not True or row.get("deepx_runtime_mode") != "A4"
            or row.get("deepx_max_inflight") != 4
            or options.get("deepx_runtime_mode") != "A4"
            or type(options.get("frames")) is not int or options["frames"] != 64
            or type(options.get("warmup")) is not int or options["warmup"] != 0):
        raise ValueError("numerical worker mode or bounded smoke recipe differs")
    for field in ("completed_frames", "completed_work_units", "postprocess_completed_frames"):
        if field in row and (type(row[field]) is not int or row[field] != 64):
            raise ValueError("numerical smoke completion counters conflict: " + field)
    for field in ("warmup", "warmup_count", "warmup_completed_frames", "warmup_completed_count"):
        if field in row and row[field] is not None and (type(row[field]) is not int or row[field] != 0):
            raise ValueError("numerical smoke warmup must stay zero: " + field)
    source = prepared["source_contract"]
    for role in (("dxnn", "source_onnx") if full else ("dxnn", "engine")):
        previous = (source.get("artifacts") or {}).get(role) or {}
        current = (contract.get("artifacts") or {}).get(role) or {}
        expected = previous.get("sha256") or (source.get("source_model_sha256") if role == "source_onnx" else None)
        if not expected or expected != current.get("sha256"):
            raise ValueError("numerical compiled/source artifact differs: " + role)
    if contract.get("input_image_sha256") != source.get("input_image_sha256"):
        raise ValueError("numerical timing image differs from the original contract")
    numerical = row.get("numerical_validation" if full else "deepx_numerical_preflight") or {}
    serial = numerical.get("serial_reference_count" if full else "serial_reference_requests")
    mixed = numerical.get("mixed_completed_count" if full else "mixed_completions")
    images = numerical.get("images") if full else [request.get("image") for request in numerical.get("requests") or []]
    if (numerical.get("status") != "passed" or type(serial) is not int or serial != expected_images
            or type(mixed) is not int or mixed != expected_images
            or not isinstance(images, list) or len(images) != expected_images
            or any(not isinstance(path, str) or not path for path in images)
            or len(set(images)) != expected_images
            or numerical.get("float_rtol") != 1e-4 or numerical.get("float_atol") != 1e-5):
        raise ValueError("numerical raw image/count/tolerance proof incomplete")
    maximum = numerical.get("actual_pending_max")
    if type(maximum) is not int or not 1 <= maximum <= min(4, expected_images):
        raise ValueError("numerical actual host pending outside the configured A4 limit")
    if full:
        if (numerical.get("all_outputs_compared") is not True
                or numerical.get("completed_endpoints_compared") is not True
                or numerical.get("integer_comparison") != "exact"
                or numerical.get("accepted_work_drained") is not True
                or numerical.get("timed") is not False):
            raise ValueError("numerical Full raw/completed endpoint comparison unproven")
        smoke = row.get("numerical_smoke_completed_count")
    else:
        requests = numerical.get("requests") or []
        if (numerical.get("mode") != "A4" or numerical.get("image_count") != expected_images
                or numerical.get("integer_exact") is not True or numerical.get("nonfinite_rejected") is not True
                or numerical.get("comparison_outside_performance_timing") is not True
                or numerical.get("drained") is not True or numerical.get("queue_depth") != 3
                or [request.get("request_id") for request in requests] != list(range(expected_images))
                or any(request.get("completed_task_equal") is not True for request in requests)):
            raise ValueError("numerical Split ordered raw/completed endpoint comparison unproven")
        smoke = row.get("completed_frames")
    safety = native._deepx_completion_safety(raw)
    if safety.get("report_contract_complete") is False:
        raise ValueError("numerical worker report contract is incomplete or contradictory")
    if (type(smoke) is not int or smoke != 64 or row.get("postprocess_completion_verified") is not True
            or safety["device_completion_proven"] is not True):
        raise ValueError("numerical smoke completion or final SDK drain unproven")
    return {"image_count": expected_images, "serial_reference_count": serial,
            "mixed_completed_count": mixed, "smoke_completed_count": smoke,
            "actual_pending_max": maximum, "device_completion_proven": True,
            "deepx_runtime_mode": "A4", "contract_sha256": contract["contract_sha256"]}


def _set_option(argv, flag, value):
    if argv.count(flag) > 1:
        raise ValueError("duplicate numerical runner option: " + flag)
    if flag in argv:
        argv[argv.index(flag) + 1] = str(value)
    else:
        argv.extend([flag, str(value)])


def _numerical_item(prepared):
    item = copy.deepcopy(prepared["runs"][0])
    old = item["remote_output_root"].rstrip("/")
    remote = str(PurePosixPath(old).parent / "numerical")
    output = remote + "/outputs"
    argv = item["argv"]
    for index, flag in enumerate(argv[:-1]):
        if flag in native._OUTPUT_FLAGS:
            value = argv[index + 1]
            if value == old or value.startswith(old + "/"):
                argv[index + 1] = output + value[len(old):]
    if item.get("native_full") is True:
        root = remote + "/inputs"
        benchmark = root + "/" + prepared["identity"]["model"] + "/benchmark_set"
        _set_option(argv, "--root", root)
        item["full_benchmark_set"] = benchmark
        item["result_json"] = output + "/analysis_tables/native_full_baseline_eval.json"
    else:
        benchmark = remote + "/benchmark_set"
        _set_option(argv, "--benchmark-set", benchmark)
        _set_option(argv, "--queue-depth", 3)
        item["result_json"] = output + "/deepx_native_fifo_e2e_results.json"
    for flag, value in (("--frames", 64), ("--warmup", 0), ("--repetitions", 1),
                        ("--duration-s", 0), ("--deepx-runtime-mode", "A4")):
        _set_option(argv, flag, value)
    if any(flag in argv for flag in ("--energy-workload-only", "--build-missing-engine", "--force-rebuild-engine", "--force-rebuild")):
        raise ValueError("numerical acceptance forbids build and energy options")
    if "--deepx-numerical-only" not in argv:
        argv.append("--deepx-numerical-only")
    item.update(remote_output_root=output, numerical_benchmark_set=benchmark, repeat=1)
    return item


def _observed_counts(raw, prepared):
    full = prepared["identity"]["backend"] == "native_full_deepx"
    rows = raw.get("rows") or []
    row = rows[0] if full and len(rows) == 1 else raw
    numerical = row.get("numerical_validation" if full else "deepx_numerical_preflight") or {}
    values = {
        "serial_reference_count": numerical.get("serial_reference_count" if full else "serial_reference_requests"),
        "mixed_completed_count": numerical.get("mixed_completed_count" if full else "mixed_completions"),
        "smoke_completed_count": row.get("numerical_smoke_completed_count" if full else "completed_frames"),
        "additional_reference_calls": row.get("numerical_setup_reference_calls") if full else numerical.get("setup_deepx_serial_calls"),
        "total_observed_sdk_calls": row.get("numerical_total_sdk_calls"),
        "serial_runtime_calls_started": row.get("worker_serial_runtime_calls_started", row.get("serial_runtime_calls_started")),
        "serial_runtime_calls_completed": row.get("worker_serial_runtime_calls_completed", row.get("serial_runtime_calls_completed")),
        "series_preflight_reference_calls": row.get("numerical_series_preflight_reference_calls"),
    }
    result = {key: value if type(value) is int and value >= 0 else None for key, value in values.items()}
    counts = [result[key] for key in ("serial_reference_count", "mixed_completed_count", "smoke_completed_count", "additional_reference_calls")]
    if raw.get("ok") is True and numerical.get("status") == "passed" and all(type(value) is int for value in counts):
        result["total_observed_sdk_calls"] = sum(counts)
    return result


def _extract_numerical_inputs(benchmark, archive, expected):
    """Actual remotely transmitted standard-library-only bundle extraction."""
    import hashlib
    import json
    from pathlib import Path
    import tarfile
    benchmark, archive = Path(benchmark), Path(archive)
    root = benchmark / "resources/numerical"
    if root.exists() or root.is_symlink() or any(path.is_symlink() for path in benchmark.parents):
        raise ValueError("numerical image destination must be a fresh private directory")
    root.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive, "r:gz") as bundle:
        members = bundle.getmembers()
        if [item.name for item in members] != [row["name"] for row in expected] + ["images.json"]:
            raise ValueError("numerical bundle member list differs")
        for member in members:
            if not member.isfile() or Path(member.name).name != member.name or member.size > 64 * 1024 * 1024:
                raise ValueError("numerical bundle has unsafe or oversized members")
            payload = bundle.extractfile(member).read()
            (root / member.name).write_bytes(payload)
    for row in expected:
        target = root / row["name"]
        if hashlib.sha256(target.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("numerical staged image differs")
    images = json.loads((root / "images.json").read_text())["images"]
    if len(images) != len(expected) or len(set(images)) != len(expected) or any(not Path(path).is_file() for path in images):
        raise ValueError("numerical staged image manifest incomplete")
    return {"image_count": len(expected), "manifest": str(root / "images.json"), "images": images}


def _stage_images(transport, prepared, item, selected, directory):
    sources = selected.get("source_images") or []
    image_count = len(selected.get("images") or [])
    if not 1 <= image_count <= 32 or len(sources) != image_count:
        raise ValueError("numerical staging requires one through 32 frozen local source images")
    benchmark = item["numerical_benchmark_set"]
    bundle_root = directory / "image_bundle"
    bundle_root.mkdir()
    expected, paths = [], []
    for index, source in enumerate(sources):
        local = Path(source["local_path"]).resolve(strict=True)
        sha = source["sha256"]
        if (source.get("index") != index or source.get("original_remote_path") != selected["images"][index]
                or local.stat().st_size > 64 * 1024 * 1024 or native._sha(local) != sha):
            raise ValueError("numerical original image binding differs")
        name = f"{index:02d}_" + local.name
        shutil.copyfile(local, bundle_root / name)
        if native._sha(bundle_root / name) != sha:
            raise ValueError("numerical local image staging changed")
        expected.append({"name": name, "sha256": sha})
        paths.append(benchmark + "/resources/numerical/" + name)
    if len({row["sha256"] for row in expected}) != image_count:
        raise ValueError("numerical source images must have different contents")
    if sources[0]["sha256"] != prepared["source_contract"].get("input_image_sha256"):
        raise ValueError("numerical first image must be the bound timing image")
    if item.get("native_full") is True:
        paths[0] = benchmark + "/resources/acceptance/" + PurePosixPath(prepared["source_contract"]["input_image"]).name
        _set_option(item["argv"], "--image-map", json.dumps({prepared["identity"]["model"]: {"full": paths[0]}}))
    else:
        _set_option(item["argv"], "--image", paths[0])
    native._write(bundle_root / "images.json", {"images": paths})
    bundle_path = directory / "numerical_images.tar.gz"
    with tarfile.open(bundle_path, "w:gz") as bundle:
        for name in [row["name"] for row in expected] + ["images.json"]:
            bundle.add(bundle_root / name, arcname=name, recursive=False)
    remote_bundle = benchmark + "/numerical_images.tar.gz"
    rc, text = transport.scp_upload(str(bundle_path), remote_bundle, timeout=120)
    if rc:
        raise RuntimeError("numerical image bundle upload failed: " + text[-2000:])
    program = ("import json\n" + inspect.getsource(_extract_numerical_inputs) + "\nprint(json.dumps(_extract_numerical_inputs("
               + repr(benchmark) + "," + repr(remote_bundle) + "," + repr(expected) + ")))\n")
    command = shlex.quote(item["argv"][0]) + " -B - <<'NATIVE_NUMERICAL_IMAGES'\n" + program + "NATIVE_NUMERICAL_IMAGES"
    rc, text = transport.run(command, timeout=120)
    native._write(directory / "image_staging.json", {"returncode": rc, "output": text[-12000:], "bundle_sha256": native._sha(bundle_path)})
    if rc:
        raise RuntimeError("numerical image extraction failed: " + text[-2000:])
    proof = json.loads(text)
    if proof.get("images") != paths or proof.get("image_count") != image_count:
        raise ValueError("numerical image staging proof differs")
    _set_option(item["argv"], "--deepx-numerical-images", proof["manifest"])
    return proof


def run_numerical_native(prepared, context, images_binding):
    """Execute one bounded worker using the existing lease, transport and lock."""
    if context.get("execute_allowed") is not True or prepared.get("deepx_runtime_mode") != "A4":
        raise ValueError("numerical Native DeepX A4 hardware dispatch is not enabled")
    selection_path = native._bound_path(images_binding)
    selected = native._read(selection_path)
    image_count = len(selected.get("images") or [])
    if not 1 <= image_count <= 32:
        raise ValueError("numerical acceptance requires one through 32 frozen images")
    for name in ("source_plan", "source_raw_report", "source_validation"):
        if native._sha(prepared[name]) != prepared[name + "_sha256"]:
            raise ValueError("numerical acceptance source changed: " + name)
    if native._verify_current_target(prepared["target"], prepared["identity"], context) != prepared["target_binding"]:
        raise ValueError("numerical acceptance hardware target changed")
    output = Path(prepared["output"])
    result_path = output / "NUMERICAL_RESULT.json"
    if result_path.exists() or result_path.is_symlink():
        raise RuntimeError("existing numerical result must be preserved through normal Resume")
    directory = output / "numerical"
    directory.mkdir(parents=True, exist_ok=False)
    original = directory / "original_images.json"
    shutil.copyfile(selection_path, original)
    result = {"schema": SCHEMA, "status": "failed", "identity": dict(prepared["identity"]),
              "deepx_runtime_mode": "A4", "source_contract_sha256": prepared["source_contract"]["contract_sha256"],
              "phase_matrix": prepared["phase_matrix"], "numerical_images": dict(images_binding),
              "fresh_hardware_executed": False, "new_performance_processes": 0, "new_capture_count": 0,
              "scientific_claim_eligible": False, "scientific_cohort_complete": False,
              "scientific_statistics_published": False, "device_completion_proven": False,
              "attempt_admitted": False, "attempt_state": "pre_admission",
              "evidence_paths": [str(original)], "image_count": image_count,
              "serial_reference_count": 0, "mixed_completed_count": 0, "smoke_completed_count": 0}
    sdk_uncertain = False
    try:
        transport = native._transport(prepared, context)
        owner = {"run_id": Path(context["run_dir"]).name, "purpose": "hardware_acceptance_native_numerical"}
        with ExitStack() as stack:
            physical = stack.enter_context(native.EvaluationRunLock.for_resource(native.physical_dut_key(prepared["target"]), owner=owner))
            def fence_cleanup():
                registry = context["remote_process_registry"]
                active = registry.active_count()
                if sdk_uncertain or type(active) is not int or active != 0 or getattr(registry, "cancelled", False) is True:
                    physical.commit_quarantine_fence({"reason": "hardware_acceptance_native_numerical_completion_unproven", "evidence": str(directory)})
            stack.callback(fence_cleanup)
            with native.bind_workflow_resource_options(controller_setup_id=prepared["identity"]["setup_id"], controller_attempt_dir=str(directory)):
                admission = _admit_numerical_attempt(prepared, context, image_count)
                result.update(attempt_admitted=True, attempt_state="admitted", attempt_admission=admission)
                # Existing result evidence reserves this logical attempt before
                # staging/dispatch. A crash cannot replenish its finite budget.
                with result_path.open("x", encoding="utf-8") as stream:
                    json.dump(result, stream, indent=2)
                    stream.flush()
                    os.fsync(stream.fileno())
                from ..remote.process_lease import RemoteProcessLeaseJournal
                RemoteProcessLeaseJournal._fsync_directory(output, strict=True)
                item = _numerical_item(prepared)
                cancel = context.get("cancel_event")
                if cancel is not None and cancel.is_set():
                    raise RuntimeError("numerical acceptance cancelled before staging")
                diagnostic_dir = getattr(transport, "diagnostics_dir", None)
                staging_diagnostics = {str(path.resolve()) for path in Path(diagnostic_dir).glob("ssh_*.json")} if diagnostic_dir else set()
                result["preparation"] = {"phase": "pre_worker_staging", "worker_dispatched": False}
                native._write(result_path, result)
                # Staging runs owned commands, but this coordinator has not
                # entered the worker. Record that fact before any such command.
                sdk_uncertain = True
                try:
                    native.ensure_current_deepx_runtime(prepared, context)
                    native._stage_benchmark_set(transport, item, directory)
                    _stage_images(transport, prepared, item, selected, directory)
                    native._write(directory / "COMMAND.json", item)
                    command = "cd -- " + shlex.quote(prepared["remote_tool_dir"]) + " && " + shlex.join(item["argv"])
                    if cancel is not None and cancel.is_set():
                        raise RuntimeError("numerical acceptance cancelled before inference")
                except Exception:
                    closure = _owned_preparation_closure(transport, context["remote_process_registry"], staging_diagnostics)
                    result["owned_preparation_closure"] = closure
                    result["preparation"]["phase"] = "pre_worker_failed"
                    result["device_completion_proven"] = True
                    result["resource_completion_proven"] = closure["proven"] is True
                    sdk_uncertain = not result["resource_completion_proven"]
                    raise
                result["preparation"].update(phase="worker_dispatch_entered", worker_dispatched=True)
                native._write(result_path, result)
                result["fresh_hardware_executed"] = True
                result.update(serial_reference_count=None, mixed_completed_count=None, smoke_completed_count=None,
                              additional_reference_calls=None, total_observed_sdk_calls=None)
                sdk_uncertain = True
                diagnostic_dir = getattr(transport, "diagnostics_dir", None)
                previous_diagnostics = {str(path.resolve()) for path in Path(diagnostic_dir).glob("ssh_*.json")} if diagnostic_dir else set()
                rc, text = transport.run(command, timeout=int(item.get("timeout_s") or 900), env=item["env"])
                closure = _owned_dispatch_closure(transport, context["remote_process_registry"], command, rc, previous_diagnostics)
                result["owned_worker_closure"] = closure
                native._write(directory / "DISPATCH.json", {"argv": item["argv"], "returncode": rc, "output": text[-12000:], "completed_at": time.time()})
                report = native._collect_repeat(transport, item, directory)
                result["worker_report"] = {"path": str(report), "sha256": native._sha(report)}
                result["evidence_paths"].extend([str(directory / "COMMAND.json"), str(directory / "DISPATCH.json"), str(report)])
                raw = native._read(report)
                closure = _owned_dispatch_closure(transport, context["remote_process_registry"], command, rc, previous_diagnostics)
                result["owned_worker_closure"] = closure
                safety = native._deepx_completion_safety(raw)
                safety["owned_dispatch_closure"] = closure
                safety["resource_completion_proven"] = safety["device_completion_proven"] is True and closure["proven"] is True
                sdk_uncertain = not safety["resource_completion_proven"]
                result["device_completion_proven"] = safety["device_completion_proven"]
                result["resource_completion_proven"] = safety["resource_completion_proven"]
                native._write(directory / "SDK_COMPLETION.json", safety)
                result["evidence_paths"].append(str(directory / "SDK_COMPLETION.json"))
                result.update(_observed_counts(raw, prepared))
                if sdk_uncertain:
                    raise RuntimeError("numerical SDK or owned worker completion unsafe or unproven; DUT quarantined")
                if rc:
                    raise RuntimeError(f"numerical worker failed with rc {rc}; original report retained")
                result.update(validate_numerical_raw(raw, prepared, expected_images=image_count))
                result.update(status="numerical_passed", numerical_status="passed")
    except Exception as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}", numerical_status="failed")
    result["attempt_state"] = "terminal"
    result["evidence_paths"].append(str(result_path))
    native._write(result_path, result)
    return result
