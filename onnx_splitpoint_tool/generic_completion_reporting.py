"""Consume the selective Generic completion result, without rewriting old runs.

The runtime's completion validator is also the resume authority.  Files and
attempts are not observations: only three bound, functionally checked repetitions
form a new case observation.  Historical raw Generic and measured Native rates
remain separately named and linked.
"""
from __future__ import annotations

import copy
import csv
import io
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

from .generic_completion_plan import completion_identity
from .native_detection_postprocess import FrozenPostprocessError
from .workflow.checkpoints import atomic_write_json, atomic_write_text
from .workflow.cross_runner_reporting import (
    _kendall_tau_b, _pairwise_counts, _rank, _spearman,
)


IDENTITY_FIELDS = (
    "model_id", "case_id", "direction", "precision", "setup_id",
    "comparison_backend", "output_endpoint_id",
)
BOUNDARY_CHECKS = (
    "input_sequence_matches", "batch_matches", "prepared_input_start_matches",
    "completion_endpoint_matches", "fill_and_drain_matches",
    "transfer_and_materialization_scope_matches",
)


def _config(case: Mapping[str, Any]) -> Mapping[str, Any]:
    return case.get("runtime_config", case)


def _identity(case: Mapping[str, Any]) -> tuple[str, ...]:
    identity = _config(case).get("identity")
    if not isinstance(identity, Mapping):
        raise ValueError("generic completion exact identity missing")
    values = tuple(str(identity.get(key) or "") for key in IDENTITY_FIELDS)
    if not all(values):
        raise ValueError("generic completion exact identity incomplete")
    return completion_identity(dict(zip(IDENTITY_FIELDS, values)))


def _read_result(path: str | Path) -> dict[str, Any] | None:
    path = Path(path)
    if not path.exists():
        return None
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate result JSON key: {key}")
            result[key] = value
        return result
    payload = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique)
    if not isinstance(payload, dict):
        raise ValueError("generic completion result is not an object")
    return payload


def load_valid_repetitions(result_path: str | Path, case: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return reusable repetitions; binding conflicts stop reentry explicitly."""
    from .runners.task_completion import valid_split_completion_repetitions
    result = _read_result(result_path)
    if result is None:
        return []
    _identity(case)
    return valid_split_completion_repetitions(result, _config(case), result_root=Path(result_path).parent)


def validate_completion_case(result: Mapping[str, Any] | None, case: Mapping[str, Any], *,
                             result_root: str | Path | None = None) -> dict[str, Any]:
    """Project success/partial/failure honestly, using the actual runtime parser."""
    from .runners.task_completion import valid_split_completion_repetitions
    identity = _identity(case)
    projected = dict(zip(IDENTITY_FIELDS, identity))
    projected.update(case_key=case.get("case_key", "|".join(identity)),
                     status="unstarted", valid_repetition_count=0,
                     completed_task_count=0, pending_repetition_indices=[1, 2, 3],
                     errors=[], repetitions=[], raw_attempts=[])
    if result is None:
        return projected
    projected["raw_attempts"] = copy.deepcopy(result.get("repetitions", []))
    projected["attempt_history"] = copy.deepcopy(result.get("attempt_history", []))
    projected["execution_id"] = result.get("execution_id")
    try:
        repetitions = valid_split_completion_repetitions(result, _config(case), result_root=result_root)
    except (ValueError, TypeError, KeyError) as error:
        projected.update(status="blocked", errors=[str(error)])
        return projected
    indices = {row["repetition_index"] for row in repetitions}
    projected.update(repetitions=repetitions, valid_repetition_count=len(repetitions),
                     completed_task_count=sum(row["completed_task_count"] for row in repetitions),
                     pending_repetition_indices=[i for i in (1, 2, 3) if i not in indices])
    if len(repetitions) == 3:
        rates = [row["completed_task_fps"] for row in repetitions]
        spans = [row["measured_makespan_seconds"] for row in repetitions]
        projected.update(
            status="completed", completed_task_fps=statistics.median(rates),
            completed_task_fps_median=statistics.median(rates),
            completed_task_fps_min=min(rates), completed_task_fps_max=max(rates),
            completed_task_fps_stddev=statistics.stdev(rates),
            measured_makespan_seconds_median=statistics.median(spans),
            measured_makespan_seconds_sum=sum(spans),
            aggregation="median_of_three_independent_measured_repetitions",
            dispersion="sample_standard_deviation_ddof_1",
        )
    elif repetitions:
        projected["status"] = "partial"
    elif result.get("status") in {"failed", "cancelled", "blocked"}:
        projected["status"] = str(result["status"])
    elif result.get("repetitions") or result.get("attempt_history"):
        projected["status"] = "partial"
    # A created/config-only/started JSON with no completed work is not success.
    return projected


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            return {}
    return dict(value) if isinstance(value, Mapping) else {}


def _positive(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value > 0 else None


def _verified_pair_prepared_input(command: Mapping[str, Any], case: Mapping[str, Any],
                                  source: Path, run: Path) -> dict[str, Any]:
    """Resolve the existing Native feed contract and verify its Generic alias."""
    from .generic_completion_plan import _input_binding
    from .runners.task_completion import load_completion_bound_array
    from .workflow.artifacts import sha256_file
    from .resources.remote_scripts.native_producer_validate_visualize import _benchmark_set_for_report

    benchmark_set = _benchmark_set_for_report(source)
    if benchmark_set is None:
        raise ValueError("native_original_prepared_input_benchmark_set_missing")
    config = _config(case)
    binding = _input_binding(command, {
        "source_root": str(benchmark_set.parent.parent), "model": command["model"],
        "task": config["task"], "report": str(source),
    }, run)
    if binding != case.get("input_binding"):
        raise ValueError("native_original_prepared_input_binding_mismatch")
    runtime_binding = _mapping(config.get("input_binding"))
    if any(runtime_binding.get(key) != value for key, value in binding.items()):
        raise ValueError("native_generic_prepared_input_binding_mismatch")
    contract = binding["prepared_input_contract"]
    entries = binding["entries"]
    names = [entry["name"] for entry in entries]
    if (len(entries) != 1 or len(set(names)) != len(names) or not all(names)
            or ("entries" in contract and not contract["entries"])
            or ("entries" in contract and contract.get("slot_order") != names)
            or ("slot_order" in contract and contract["slot_order"] != names)):
        raise ValueError("native_original_prepared_input_slot_order_invalid")
    storage_format = {"raw_runtime_tensor": "raw", "numpy_npy_v1": "npy"}.get(contract.get("format"))
    if storage_format is None:
        raise ValueError("native_original_prepared_input_format_unsupported")
    entry = entries[0]
    prepared = _mapping(runtime_binding.get("prepared_input"))
    expected = {key: entry[key] for key in ("sha256", "shape", "dtype")}
    expected.update(path=entry["remote_path"], format=storage_format)
    if any(prepared.get(key) != value for key, value in expected.items()):
        raise ValueError("native_generic_prepared_input_projection_mismatch")
    source_contract = _mapping(contract.get("source_input_contract"))
    from .native_performance_identity import canonical_native_backend
    backend = canonical_native_backend(command["backend"])
    for key in ("layout", "normalization", "source_domain"):
        if key not in prepared:
            continue
        declarations = [item[key] for item in (contract, source_contract, entry) if item.get(key)]
        if backend == "hailo8" and key != "layout":
            # The existing Generic H8 adapter either preserves UINT8 or performs
            # the paired C++ RGB/255 conversion. Derive this from the bound option.
            runtime_format = str(command.get("runtime_options", {}).get("hailo_format") or "")
            if entry["dtype"] != "uint8" or runtime_format not in {"uint8", "raw_uint8", "float32", "dequant_float32"}:
                raise ValueError("native_generic_prepared_input_domain_binding_missing")
            declarations.append("uint8_0_255" if key == "source_domain" or runtime_format in {"uint8", "raw_uint8"}
                                else "imagenet_mean_std" if config["task"] == "classification" else "unit_float32")
        if not declarations or any(value != prepared[key] for value in declarations):
            raise ValueError("native_generic_prepared_input_descriptor_conflict:" + key)
    array_spec = {**prepared, "path": entry["local_path"]}
    orders = [item["c_contiguous"] for item in (contract, entry) if "c_contiguous" in item]
    if orders:
        if any(type(value) is not bool or value != orders[0] for value in orders):
            raise ValueError("native_original_prepared_input_storage_order_conflict")
        array_spec["c_contiguous"] = orders[0]
    load_completion_bound_array(array_spec,
        file_sha256=lambda path: str(sha256_file(path)).removeprefix("sha256:"))
    return entry


def _completed_endpoint_pair_evidence(observation: Mapping[str, Any], case: Mapping[str, Any]) -> dict[str, Any]:
    """Verify this measured pair independently of other Native backends.

    The ordinary Native cross-backend gate needs two backends in its stratum.
    A first augmentation case instead has one Native observation and its own
    completed Generic observation. Their actual files must prove the same
    execution contract; this does not release any Quality or claim gate.
    """
    evidence = {"status": "unverified", "scope": "native_generic_completed_task_pair"}
    try:
        from .generic_completion_plan import _bound_file, _inside
        from .native_command_contract import verify_native_command_contract
        from .native_job_identity import planned_native_identity
        from .native_split_quality import bind_quality_to_native_split
        from .native_three_stage import NativeThreeStageError, verify_fast_completion_dump_binding
        from .native_rate_endpoints import rate_endpoint_fields
        from .native_detection_postprocess import (
            verify_detection_completion_execution_contract,
            verify_detection_completion_execution_attestation,
        )
        from .resources.remote_scripts.native_producer_final_report import _verify_split_semantic_artifacts
        from .resources.remote_scripts.native_producer_validate_visualize import _verified_execution_completed_result_artifact

        config = _config(case)
        identity = _identity(case)
        if config.get("original_run") != case.get("original_run"):
            raise ValueError("native_generic_original_run_mismatch")
        if config.get("task") != "detection" or observation.get("status") != "completed":
            raise ValueError("completed_detection_pair_required")
        result_path = Path(str(observation.get("result_path") or case.get("result_path") or ""))
        actual = validate_completion_case(_read_result(result_path), case, result_root=result_path.parent)
        for field in (*IDENTITY_FIELDS, "status", "valid_repetition_count", "completed_task_count",
                      "completed_task_fps", "repetitions"):
            if actual.get(field) != observation.get(field):
                raise ValueError("observed_completion_file_projection_mismatch:" + field)
        if actual.get("status") != "completed" or actual.get("valid_repetition_count") != 3:
            raise ValueError("three_bound_generic_repetitions_required")
        run = Path(str(case["original_run"]))
        source = _inside(case["native_source"], run)
        native = _read_result(source) or {}
        if native.get("ok") is not True:
            raise ValueError("native_original_measurement_not_valid")
        command, status = verify_native_command_contract(native.get("native_command_contract"), expected_identity={
            "model": identity[0], "case": identity[1], "setup_id": identity[4],
            "backend": identity[2], "precision": identity[3], "comparison_backend": identity[5],
        })
        if (command is None or command.get("contract_sha256") != case.get("native_command_contract_sha256")
                or command.get("contract_sha256") != native.get("native_command_contract_sha256")
                or command.get("eval_run_id") != run.name
                or command.get("source_request_sha256") != native.get("source_request_sha256")):
            raise ValueError("native_original_command_binding_invalid:" + status)
        # Raw producer results can omit presentation identity fields. As in the
        # normal Native report's portable projection, obtain them from the sealed
        # command already verified against the complete pair identity above.
        # Preserve every existing assertion for the strict consumer, and never
        # mutate the signed raw result or substitute a path-derived identity.
        consumer_native = dict(native)
        command_identity = planned_native_identity(command)
        for field, aliases in (
            ("backend", ("backend", "producer_backend")),
            ("model", ("model", "model_id")),
            ("case", ("case", "case_id", "boundary")),
            ("setup_id", ("setup_id", "measurement_setup_id", "setup")),
            ("precision", ("runtime_precision_identity", "execution_precision", "precision", "contract")),
            ("comparison_backend", ("comparison_backend",)),
        ):
            for alias in aliases:
                if consumer_native.get(alias) not in (None, "") and (
                    planned_native_identity({field: consumer_native[alias]})[field] != command_identity[field]
                ):
                    raise ValueError("native_original_identity_assertion_conflict:" + alias)
            if all(consumer_native.get(alias) in (None, "") for alias in aliases):
                consumer_native[field] = command[field]
        selected, status = bind_quality_to_native_split(native_row=consumer_native,
            quality_binding=native.get("native_split_quality_binding"), verification_mode="portable")
        if selected is None:
            raise ValueError("native_original_consumer_binding_invalid:" + status)
        if selected.get("binding_sha256") != case.get("native_quality_binding_sha256"):
            raise ValueError("native_original_selected_binding_mismatch")
        _bound_file(_inside(case["generic_original_quality_request"], run),
            {"sha256": command["source_request_sha256"]}, "pair_original_quality_request")
        rates = rate_endpoint_fields(native)
        boundary = _mapping(case.get("native_measurement_boundary"))
        if (rates.get("completed_task_work_unit_counts") != boundary.get("observed_completed_counts")
                or rates.get("completed_task_measurement_times_s") != boundary.get("observed_measurement_times_s")):
            raise ValueError("native_original_count_time_projection_mismatch")
        contract = verify_detection_completion_execution_contract(native.get("completion_execution_contract"))
        if (contract != config.get("completion_contract") or contract != case.get("completion_contract")
                or contract != command["runtime_options"].get("completion_execution_contract")
                or contract["source_endpoint"]["output_endpoint_id"] != identity[6]):
            raise ValueError("native_generic_completion_contract_mismatch")
        endpoint = contract["comparison_endpoint_contract"]["output_endpoint_id"]
        if any(boundary.get("completion_output_endpoint_id") != endpoint for boundary in (
                _mapping(case.get("native_measurement_boundary")), _mapping(config.get("measurement_contract")))):
            raise ValueError("native_generic_completed_endpoint_mismatch")
        prepared = _verified_pair_prepared_input(command, case, source, run)
        if config["functional_reference"].get("input_sha256") != prepared["sha256"]:
            raise ValueError("native_generic_prepared_input_binding_mismatch")
        manifest = _inside(case["functional_reference"]["output_manifest"], run)
        manifest_binding = command["artifacts"]["semantic_output_manifest"]
        _bound_file(manifest, manifest_binding, "pair_native_output_manifest")
        if config["functional_reference"].get("source_manifest_sha256") != manifest_binding["sha256"]:
            raise ValueError("native_generic_output_manifest_binding_mismatch")
        proof = _verify_split_semantic_artifacts(result_path=source, manifest_path=manifest,
            manifest_sha256=manifest_binding["sha256"], sources=(native,))
        if (proof.get("native_split_semantic_binding_valid") is not True
                or proof.get("native_split_final_portable_binding_valid") is not True):
            raise ValueError("native_output_payload_or_consumer_invalid")
        tensors = (_read_result(manifest) or {}).get("outputs") or []
        references = config["functional_reference"]["outputs"]
        if set(references) != {item["name"] for item in tensors} or any(
                references[item["name"]].get(field) != item.get(field)
                for item in tensors for field in ("sha256", "shape", "dtype")):
            raise ValueError("native_generic_functional_output_binding_mismatch")
        mode = native.get("completed_task_completion_mode")
        if mode == "native_three_stage_fast_oracle_outside_timing":
            from .resources.remote_scripts.native_producer_energy_plan import _verified_fast_energy_completion
            attestation, comparison = _verified_fast_energy_completion(
                {**consumer_native, **proof}, None, command)
            if comparison != contract["comparison_endpoint_contract"]:
                raise ValueError("native_generic_fast_completed_endpoint_mismatch")
            verify_fast_completion_dump_binding(attestation, _read_result(manifest) or {}, native)
        elif mode == "detection_completion_execution_v1":
            attestation = verify_detection_completion_execution_attestation(
                native.get("completed_task_endpoint_attestation"), execution_contract=contract,
                expected_observation_relation="same_hotloop_sentinel")
        else:
            raise ValueError("native_generic_completion_mode_unsupported")
        _verified_execution_completed_result_artifact(attestation["last_result"], attestation, native,
            native_report=source)
        evidence.update(status="verified", native_source=str(source), generic_source=str(result_path),
            native_command_contract_sha256=command["contract_sha256"],
            completion_contract_sha256=contract["contract_sha256"], completed_output_endpoint_id=endpoint,
            native_output_manifest_sha256=manifest_binding["sha256"], generic_valid_repetitions=3)
    except (ValueError, TypeError, KeyError, OSError, FrozenPostprocessError, NativeThreeStageError) as error:
        evidence["reason"] = str(error)
    return evidence


def completion_pair(observation: Mapping[str, Any], case: Mapping[str, Any]) -> dict[str, Any]:
    """Link all three observations, admitting quotients only for bound limits."""
    historical = _mapping(case.get("historical_pair"))
    identity = _identity(case)
    try:
        historical_identity = completion_identity(historical)
    except ValueError:
        historical_identity = ()
    boundary = _mapping(case.get("measurement_boundary_comparison"))
    boundary_checks = _mapping(boundary.get("checks"))
    sources = boundary.get("sources") or [boundary.get("native_source"), boundary.get("generic_source")]
    if isinstance(sources, Mapping):
        sources = [sources.get("native_source"), sources.get("generic_source")]
    native_fps = _positive(historical.get("native_measured_throughput_fps"))
    generic_fps = _positive(observation.get("completed_task_fps"))
    native_boundary = _mapping(case.get("native_measurement_boundary"))
    native_counts = native_boundary.get("observed_completed_counts")
    native_times = native_boundary.get("observed_measurement_times_s")
    native_count_time_fps = None
    if (native_counts == [1000, 1000, 1000] and isinstance(native_times, list)
            and len(native_times) == 3 and all(_positive(value) is not None for value in native_times)):
        native_count_time_fps = statistics.median(count / float(seconds) for count, seconds in zip(native_counts, native_times))
    environment = _mapping(case.get("runtime_environment_comparison"))
    environment_verified = (environment.get("status") in {"compatible", "verified_compatible"}
                            and bool(environment.get("current_source"))
                            and bool(environment.get("historical_sources")))
    checks = {
        "exact_historical_pair_identity": identity == historical_identity,
        "three_generic_repetitions_valid": observation.get("status") == "completed",
        "native_measured_makespan_present": native_fps is not None
        and historical.get("native_measured_fps_source") == "completed_task_count_time",
        # Same serialization tolerance as native_rate_endpoints: C++ Native
        # reports publish six decimal places, which must not become false gaps.
        "native_original_count_time_matches": native_fps is not None and native_count_time_fps is not None
        and math.isclose(native_fps, native_count_time_fps, rel_tol=1e-6, abs_tol=1e-6),
        "boundary_sources_present": isinstance(sources, (list, tuple)) and len(sources) >= 2 and all(sources),
        **{name: boundary_checks.get(name) is True for name in BOUNDARY_CHECKS},
        "no_known_runtime_environment_conflict": environment.get("status") not in {"conflict", "incompatible", "failed"},
    }
    # The historical checks are not globally released by a new timer.  Only
    # the specific absent Generic-completion observation can now be replaced.
    cohorts = {}
    for tier in ("technical", "quality", "claim"):
        old_checks = _mapping(historical.get(f"{tier}_transfer_eligibility_checks"))
        old_checks["generic_completed_task_measured"] = observation.get("status") == "completed"
        cohorts[tier] = {**old_checks, **checks, "historical_checks_present": bool(
            historical.get(f"{tier}_transfer_eligibility_checks"))}
    cross_backend_endpoint_match = cohorts["technical"].get("native_output_endpoint_match")
    endpoint_pair_evidence = {"status": "not_required"}
    if cross_backend_endpoint_match is False:
        endpoint_pair_evidence = _completed_endpoint_pair_evidence(observation, case)
        # Preserve the Native cross-backend decision in the historical pair,
        # Quality and claim tiers. Only this measured technical pair has its
        # own scope, backed by the normal command/output/completion verifiers.
        cohorts["technical"].pop("native_output_endpoint_match")
        cohorts["technical"]["native_generic_completed_endpoint_match"] = endpoint_pair_evidence["status"] == "verified"
    cohorts["claim"]["runtime_environment_comparability_verified"] = environment_verified
    comparable = generic_fps is not None and all(cohorts["technical"].values())
    row = dict(zip(IDENTITY_FIELDS, identity))
    row.update(
        case_key=case.get("case_key", "|".join(identity)),
        original_run=case.get("original_run"), execution_id=observation.get("execution_id"),
        measurement_mode="generic_completed_task", generic_raw_source=case.get("generic_raw_source"),
        generic_completion_source=observation.get("result_path"), native_source=case.get("native_source"),
        historical_generic_raw_fps=_positive(historical.get("generic_throughput_fps")),
        historical_generic_raw_cycle_ms=_positive(historical.get("generic_cycle_ms")),
        generic_completed_task_fps=generic_fps,
        native_completed_task_fps=native_fps,
        native_original_completed_counts=native_counts,
        native_original_measured_makespans_seconds=native_times,
        native_measured_fps_source=historical.get("native_measured_fps_source"),
        native_over_generic_completed_task_ratio=native_fps / generic_fps if comparable else None,
        comparison_scope="separate_generic_completed_task_supplement",
        native_cross_backend_output_endpoint_match=cross_backend_endpoint_match,
        native_generic_completed_endpoint_evidence=endpoint_pair_evidence,
        measurement_boundaries_comparable=comparable,
        measurement_boundary_comparison=boundary,
        runtime_environment_comparison=environment,
        runtime_environment_comparison_status=environment.get("status", "historical_conditions_not_fully_known"),
        runtime_conditions_comparable=environment_verified,
        throughput_factor_interpretation=("scope_matched_observed_throughput_with_verified_runtime_conditions"
            if environment_verified else "scope_matched_diagnostic_runtime_conditions_not_fully_verified"),
        evaluation_role=case.get("evaluation_role", "unknown"),
        evaluation_role_source=case.get("evaluation_role_source"),
        generic_task_quality_status=historical.get("generic_task_quality_status"),
        native_task_quality_status=historical.get("native_task_quality_status"),
        generic_accuracy_assessment=historical.get("generic_accuracy_assessment"),
        native_accuracy_assessment=historical.get("native_accuracy_assessment"),
        generic_completion_status=observation.get("status"),
        valid_repetition_count=observation.get("valid_repetition_count", 0),
        completed_task_count=observation.get("completed_task_count", 0),
        boundary_class=boundary.get("contract_class", "prepared_input_before_p1_to_last_completed_task"),
    )
    for tier, gates in cohorts.items():
        row[f"{tier}_transfer_eligibility_checks"] = gates
        row[f"eligible_for_{tier}_transfer"] = comparable and all(gates.values())
        row[f"{tier}_exclusion_reasons"] = [key for key, passed in gates.items() if not passed]
    return row


def completion_groups(pairs: list[dict[str, Any]], *, minimum_candidates: int = 3) -> list[dict[str, Any]]:
    """Use the existing Cross-Runner rank/tie rules within each exact stratum."""
    grouped = defaultdict(list)
    fields = ("model_id", "setup_id", "direction", "precision", "comparison_backend", "output_endpoint_id", "boundary_class")
    for pair in pairs:
        grouped[tuple(str(pair.get(key) or "") for key in fields)].append(pair)
    groups = []
    for key, rows in sorted(grouped.items()):
        group = dict(zip(fields, key))
        group.update(planned_candidate_count=len(rows), minimum_candidates=minimum_candidates)
        for tier in ("technical", "quality", "claim"):
            eligible = [r for r in rows if r.get(f"eligible_for_{tier}_transfer")]
            # Same lower-is-better cycle ranking as the original reporter;
            # inverse here is only a rank coordinate of already measured FPS.
            generic = [1000.0 / r["generic_completed_task_fps"] for r in eligible]
            native = [1000.0 / r["native_completed_task_fps"] for r in eligible]
            count = len(eligible)
            enough = count >= minimum_candidates
            rho = _spearman(generic, native) if enough else None
            tau = _kendall_tau_b(generic, native) if enough else None
            concordant, comparable = _pairwise_counts(generic, native)
            group.update({
                f"{tier}_candidate_count": count,
                f"{tier}_spearman_rho": rho,
                f"{tier}_kendall_tau_b": tau,
                f"{tier}_pairwise_concordance": concordant / comparable if enough and comparable else None,
                f"{tier}_pairwise_comparable_count": comparable,
                f"{tier}_status": "insufficient_candidates" if not enough else "constant_ranks" if rho is None else "ok",
            })
            for row, grank, nrank in zip(eligible, _rank(generic), _rank(native)):
                row[f"{tier}_generic_rank"] = grank
                row[f"{tier}_native_rank"] = nrank
        groups.append(group)
    return groups


def _csv(path: Path, rows: list[dict[str, Any]]) -> None:
    stream = io.StringIO(newline="")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value for key, value in row.items()})
    atomic_write_text(path, stream.getvalue())


def case_result_path(case: Mapping[str, Any], results_dir: str | Path) -> Path:
    if case.get("result_path"):
        return Path(str(case["result_path"]))
    relative = case.get("case_output_relative")
    if not relative:
        identity = dict(zip(IDENTITY_FIELDS, _identity(case)))
        relative = f"{identity['setup_id']}/{identity['model_id']}/{identity['case_id']}"
    return Path(results_dir) / str(relative) / "generic_completion_results.json"


def write_completion_reports(plan: Mapping[str, Any], results_dir: str | Path, out_dir: str | Path) -> dict[str, Any]:
    """Idempotent incremental/final accounting from the same bound result files."""
    cases = plan.get("rows", plan.get("cases", []))
    if not isinstance(cases, list) or not cases:
        raise ValueError("generic completion plan rows missing")
    identities = [_identity(case) for case in cases]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate generic completion planned identity")
    paths = [case_result_path(case, results_dir).resolve() for case in cases]
    if len(set(paths)) != len(paths):
        raise ValueError("multiple generic completion identities share result file")
    out = Path(out_dir).resolve()
    for case in cases:
        if case.get("original_run") and out.is_relative_to(Path(case["original_run"]).resolve()):
            raise ValueError("generic completion output must remain outside original run")
    observations = []
    for case, path in zip(cases, paths):
        try:
            observation = validate_completion_case(_read_result(path), case, result_root=path.parent)
        except (ValueError, OSError) as error:
            observation = validate_completion_case(None, case)
            observation.update(status="blocked", errors=[str(error)])
        observation["execution_status"] = case.get("execution_status", "")
        observation["dispatch_attempted"] = case.get("dispatch_attempted", False)
        if case.get("execution_error"):
            observation["execution_error"] = str(case["execution_error"])
            observation["errors"].append(str(case["execution_error"]))
        if observation["status"] == "unstarted" and case.get("execution_status") in {"blocked", "failed", "cancelled"}:
            observation["status"] = case["execution_status"]
        observation["result_path"] = str(path)
        observations.append(observation)
    pairs = [completion_pair(row, case) for row, case in zip(observations, cases)]
    groups = completion_groups(pairs)
    counts = Counter(row["status"] for row in observations)
    summary = {
        "schema": "onnx-splitpoint/generic-completion-supplement", "schema_version": 1,
        "planned_cases": len(cases), "planned_repetitions": len(cases) * 3,
        "status_counts": {key: counts.get(key, 0) for key in ("completed", "partial", "failed", "blocked", "cancelled", "unstarted")},
        "valid_repetitions": sum(row["valid_repetition_count"] for row in observations),
        "completed_task_count": sum(row["completed_task_count"] for row in observations),
        "technical_pairs": sum(row["eligible_for_technical_transfer"] for row in pairs),
        "quality_pairs": sum(row["eligible_for_quality_transfer"] for row in pairs),
        "claim_pairs": sum(row["eligible_for_claim_transfer"] for row in pairs),
        "accuracy_classes": dict(Counter(str(row.get("generic_task_quality_status") or "unknown") for row in pairs)),
        "completed_case_accuracy_classes": dict(Counter(str(row.get("generic_task_quality_status") or "unknown")
            for row in pairs if row.get("generic_completion_status") == "completed")),
        "evaluation_roles": dict(Counter(str(row.get("evaluation_role") or "unknown") for row in pairs)),
        "all_cases_measured": counts.get("completed", 0) == len(cases),
        "analysis_scope": "post_planned_selected_intersection_not_holdout",
        "energy_measurement_performed": False,
    }
    out.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out / "generic_completion_observations.json", {"rows": observations})
    atomic_write_json(out / "generic_completion_cross_runner.json", {"pairs": pairs, "groups": groups, "summary": summary})
    _csv(out / "generic_completion_observations.csv", observations)
    _csv(out / "generic_completion_cross_runner_pairs.csv", pairs)
    _csv(out / "generic_completion_cross_runner_groups.csv", groups)
    raw_repetitions = [{"case_key": row["case_key"], "result_path": row["result_path"], **rep}
                       for row in observations for rep in row["raw_attempts"] if isinstance(rep, Mapping)]
    _csv(out / "generic_completion_raw_repetitions.csv", raw_repetitions)
    atomic_write_json(out / "generic_completion_summary.json", summary)
    return summary


def write_generic_energy_report(plan_path: str | Path, captures_path: str | Path,
                                case: Mapping[str, Any], output_dir: str | Path, *,
                                performance_admission=None) -> dict[str, Any]:
    """Import explicit normal energy captures without changing FPS reports."""
    from .generic_energy_reporting import collect_generic_energy_case
    plan = json.loads(Path(plan_path).read_text(encoding='utf-8'))
    captures = json.loads(Path(captures_path).read_text(encoding='utf-8'))
    if captures.get('schema') != 'onnx-splitpoint/generic-energy-capture-sources/v1':
        raise ValueError('Generic energy capture manifest schema mismatch')
    if captures.get('plan_sha256') != __import__('hashlib').sha256(Path(plan_path).read_bytes()).hexdigest():
        raise ValueError('Generic energy capture plan source changed')
    observation = collect_generic_energy_case(plan, captures.get('sources') or [], case,
        performance_admission=performance_admission)
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    path = out / 'generic_completed_task_energy.json'
    if path.exists():
        raise FileExistsError('Generic energy final report already exists; preserve it and write a new derivation')
    path.write_text(json.dumps(observation, indent=2, sort_keys=True, allow_nan=False)+'\n', encoding='utf-8')
    return observation
