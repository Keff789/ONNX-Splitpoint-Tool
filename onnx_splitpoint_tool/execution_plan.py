from __future__ import annotations

"""Human- and machine-readable effective execution plans.

The plan intentionally exposes the hidden expansion from logical hardware
profiles to actual Generic-Runner dispatches and the separate Native stage.  It
is descriptive only; the workflow executor remains the source of truth.
"""

from typing import Any, Dict, Mapping, Sequence

from .cache_verify_policy import (
    CACHE_VERIFY_ONLY,
    cache_verify_contract_view,
    cache_verify_guard,
)
from .native_execution_contract import resolve_native_execution_contract
from .native_full_quality import resolve_native_split_plan
from .energy.config import resolve_effective_energy_config
from .protocol_freeze import is_confirmatory_holdout
from .workflow.full_only_quality_canary import (
    resolve_full_only_quality_canary,
)
from .workflow.hardware_matrix import matrix_for_runtime
from .workflow.setup_local_trt_dispatch import (
    build_setup_local_tensorrt_quality_dispatch,
)


def _truth(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, Mapping):
        if "enabled" in value:
            return _truth(value.get("enabled"))
        return bool(value)
    return str(value or "").strip().lower() in {"1", "true", "yes", "on", "ok"}


def profile_experiment_purpose(profile: Mapping[str, Any]) -> str:
    """Only explicit metadata selects a scope; free-text legacy purposes stay legacy."""
    purpose = str(profile.get("purpose") or "").strip().lower()
    return purpose if purpose in {"coverage_integration", "ranking_experiment", "deepx_preprocessing_ab"} else "legacy_unspecified"


def _model_rows(profile: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    suite = profile.get("model_suite") if isinstance(profile.get("model_suite"), Mapping) else {}
    return [r for r in (suite.get("primary") or []) if isinstance(r, Mapping) and _truth(r.get("enabled", True))]


def _logical_profiles(profile: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [r for r in (profile.get("run_profiles") or []) if isinstance(r, Mapping) and _truth(r.get("enabled", True))]


def _setup_class(run_id: str) -> str:
    rid = str(run_id or "").lower()
    if "hailo10" in rid:
        return "hailo10h_setup"
    if "hailo8" in rid:
        return "hailo8_setup"
    if "deepx" in rid or "dx_m1" in rid:
        return "deepx_setup"
    # ORT CPU/TensorRT reference rows are normally dispatched on the GPU host,
    # which is the DeepX/Jetson setup in the current hardware registry.
    if rid in {"ort_cpu", "ort_cuda", "ort_tensorrt", "tensorrt_full", "trt_full"} or "tensorrt" in rid:
        return "gpu_reference_setup"
    return "other_setup"


def _setup_local_trt_quality_endpoint_id(setup_group: str) -> str:
    """Return the scheduler-owned Full TRT quality id for a logical setup."""

    return {
        "hailo8_setup": "tensorrt_at_hailo8_full",
        "hailo10h_setup": "tensorrt_at_hailo10h_full",
        "deepx_setup": "tensorrt_at_deepx_m1_full",
    }.get(str(setup_group or ""), "")


def _setup_group_for_trt_quality_producer(producer: str) -> str:
    return {
        "hailo8": "hailo8_setup",
        "hailo10h": "hailo10h_setup",
        "deepx": "deepx_setup",
    }.get(str(producer or "").strip().lower(), "")


def _standard_setup_local_trt_quality_contract(
    profile: Mapping[str, Any],
    *,
    logical_rows: list[Mapping[str, Any]],
    model_ids: list[str],
    requested: bool,
) -> dict[str, Any]:
    """Materialize Standard/Quality TRT companions from frozen hardware.

    Full-only canaries already carry an explicit identity matrix in their
    run-mode contract.  Standard/Quality runs do not, although their remote
    writer emits the same sealed, rowless Native TensorRT Full request.  The
    effective plan is the management-side authority for those requests, so it
    must contain physical setup ids rather than only descriptive setup groups.
    """

    base: dict[str, Any] = {
        "schema": (
            "onnx-splitpoint/"
            "effective-plan-setup-local-tensorrt-quality-companions"
        ),
        "schema_version": 1,
        "mode": "standard_quality",
        "requested": bool(requested),
        "status": "not_requested",
        "identity_authority": "effective_execution_plan",
        "models": list(model_ids),
        "identity_count": 0,
        "setup_ids": [],
        "quality_companion_endpoint_ids": [],
        "physical_setup_ids_by_producer": {},
        "identities": [],
        "errors": [],
    }
    if not requested:
        return base

    dispatch_rows = [dict(row) for row in logical_rows]
    if not any(
        str(row.get("id") or row.get("run_id") or "")
        .strip().lower().replace("-", "_") == "ort_tensorrt"
        for row in dispatch_rows
    ):
        # The scheduler automatically inserts this exact reference recipe when
        # a profile omits it.  Materialize the same executable row here; an
        # existing id-only or CPU-disguised recipe is deliberately not fixed.
        dispatch_rows.append({
            "id": "ort_tensorrt",
            "type": "same_backend_reference",
            "full": "tensorrt",
            "stage1": "tensorrt",
            "stage2": "tensorrt",
        })

    errors: list[str] = []
    try:
        hardware_targets = matrix_for_runtime(profile)
        dispatch = build_setup_local_tensorrt_quality_dispatch(
            profile,
            hardware_targets=hardware_targets,
            plan_rows=dispatch_rows,
        )
    except Exception as exc:
        dispatch = {}
        errors.append(
            "standard_setup_local_tensorrt_contract_resolution_failed:"
            f"{type(exc).__name__}:{exc}"
        )
    errors.extend(str(value) for value in list(dispatch.get("errors") or []))
    if dispatch.get("ok") is not True:
        if not errors:
            errors.append("standard_setup_local_tensorrt_dispatch_not_ready")
        return {
            **base,
            "status": "blocked",
            "errors": list(dict.fromkeys(errors)),
        }

    identities: list[dict[str, Any]] = []
    for raw in list(dispatch.get("setup_dispatches") or []):
        if not isinstance(raw, Mapping):
            errors.append("standard_setup_local_dispatch_not_object")
            continue
        row = dict(raw)
        producer = str(row.get("producer") or "").strip().lower()
        setup_id = str(row.get("setup_id") or "").strip()
        endpoint_id = str(
            row.get("quality_companion_endpoint_id") or ""
        ).strip()
        identity = (
            dict(row.get("quality_companion_identity") or {})
            if isinstance(row.get("quality_companion_identity"), Mapping)
            else {}
        )
        logical_group = _setup_group_for_trt_quality_producer(producer)
        if (
            not producer
            or not logical_group
            or not setup_id
            or not endpoint_id
            or str(identity.get("id") or "").strip() != endpoint_id
            or str(identity.get("setup_id") or "").strip() != setup_id
            or str(identity.get("source_run_id") or "").strip().lower()
            != "native_full_tensorrt"
            or str(identity.get("dispatch_run_id") or "").strip().lower()
            != "ort_tensorrt"
            or str(identity.get("backend") or "").strip().lower()
            != "tensorrt"
            or str(identity.get("variant") or "").strip().lower() != "full"
            or str(identity.get("execution_role") or "").strip().lower()
            != "full_quality_only"
            or identity.get("performance_claims_emitted") is not False
        ):
            errors.append(
                "standard_setup_local_tensorrt_identity_invalid:"
                f"{producer or 'unknown'}@{setup_id or 'missing'}"
            )
            continue
        identities.append({
            **identity,
            "quality_companion_endpoint_id": endpoint_id,
            "producer": producer,
            "logical_setup_group": logical_group,
            "identity_source": (
                "scheduler_materialized_before_remote_dispatch"
            ),
        })

    setup_ids = [str(row.get("setup_id") or "") for row in identities]
    endpoint_ids = [str(row.get("id") or "") for row in identities]
    if len(setup_ids) != len(set(setup_ids)):
        errors.append("standard_setup_local_tensorrt_setup_id_duplicate")
    if len(endpoint_ids) != len(set(endpoint_ids)):
        errors.append("standard_setup_local_tensorrt_endpoint_id_duplicate")
    expected_dispatch_count = len(list(dispatch.get("setup_dispatches") or []))
    if not identities or len(identities) != expected_dispatch_count:
        errors.append(
            "standard_setup_local_tensorrt_identity_count_mismatch:"
            f"{len(identities)}:{expected_dispatch_count}"
        )
    if errors:
        return {
            **base,
            "status": "blocked",
            "physical_setup_ids_by_producer": dict(
                dispatch.get("generic_setup_ids") or {}
            ),
            "errors": list(dict.fromkeys(errors)),
        }
    return {
        **base,
        "status": "ready",
        "identity_count": len(identities),
        "setup_ids": setup_ids,
        "quality_companion_endpoint_ids": endpoint_ids,
        "physical_setup_ids_by_producer": dict(
            dispatch.get("generic_setup_ids") or {}
        ),
        "identities": identities,
        "errors": [],
    }


def _quality_execution(profile: Mapping[str, Any], snapshot: Mapping[str, Any]) -> tuple[str, int]:
    """Return the resolved management-quality placement shown in the plan.

    ``apply_run_mode`` materialises the effective policy at top level.  Reading
    only the immutable run-mode snapshot is wrong when a profile was resolved
    or migrated after that snapshot was selected (the v2.63 Smoke preview did
    exactly that for Native Full and management quality).
    """
    quality_gate = profile.get("quality_gate") if isinstance(profile.get("quality_gate"), Mapping) else {}
    statistics = quality_gate.get("statistics") if isinstance(quality_gate.get("statistics"), Mapping) else {}
    snapshot_quality = snapshot.get("quality") if isinstance(snapshot.get("quality"), Mapping) else {}
    raw_location = str(
        statistics.get("execution_location")
        or quality_gate.get("execution_location")
        or snapshot_quality.get("execution_location")
        or "local"
    ).strip().lower().replace("-", "_")
    if raw_location in {"management", "management_node", "central", "central_cpu"}:
        raw_location = "central_management"
    location = raw_location if raw_location in {"local", "central_management"} else "local"
    raw_workers = statistics.get("workers") or snapshot_quality.get("workers") or (4 if location == "central_management" else 1)
    try:
        workers = max(1, min(64, int(raw_workers)))
    except (TypeError, ValueError):
        workers = 4 if location == "central_management" else 1
    return location, workers


def _resolved_native_full_enabled(
    profile: Mapping[str, Any],
    preset: Mapping[str, Any],
    native_cfg: Mapping[str, Any],
    *,
    native_enabled: bool,
) -> bool:
    """Prefer the materialised Native-Full decision over the mode default."""
    if not native_enabled:
        return False
    effective = preset.get("effective") if isinstance(preset.get("effective"), Mapping) else {}
    if "native_full_baselines_enabled" in effective:
        return _truth(effective.get("native_full_baselines_enabled"))
    producers = profile.get("native_producers") if isinstance(profile.get("native_producers"), Mapping) else {}
    full = producers.get("full_baselines")
    if isinstance(full, Mapping) and "enabled" in full:
        return _truth(full.get("enabled"))
    if full is not None:
        return _truth(full)
    return _truth(native_cfg.get("full_baselines"))


def _resolved_native_performance_repetitions(
    preset: Mapping[str, Any],
    native_cfg: Mapping[str, Any],
) -> int:
    """Resolve timing repeats without consulting the separate energy policy."""
    effective = preset.get("effective") if isinstance(preset.get("effective"), Mapping) else {}
    mode_id = str(preset.get("id") or "standard").strip().lower()
    # Final Quality follows the Standard execution path.  Strict campaign
    # profiles carry an explicit materialised repetition count, so the compact
    # mode-id fallback must not silently restore the retired five-repeat Final.
    fallback = 1 if mode_id == "smoke" else 3
    raw = effective.get("native_performance_repetitions")
    if raw is None:
        raw = native_cfg.get("repetitions")
    try:
        return max(1, int(raw if raw is not None else fallback))
    except (TypeError, ValueError):
        return fallback


def _expected_normalized_rows(
    run_ids: list[str], candidate_count: int,
    run_profiles: Sequence[Mapping[str, Any]] = (),
) -> int:
    """Expand logical run IDs into the rows emitted by normalisation."""
    cases = max(1, int(candidate_count or 1))
    from .workflow.required_run_scope import canonical_run_id, reference_measurement_variants
    recipes = {canonical_run_id(row.get('id')): row for row in run_profiles if isinstance(row, Mapping)}
    total = 0
    for raw in run_ids:
        rid = str(raw or "").strip().lower().replace("-", "_")
        if rid in {"ort_cpu", "cpu_ort", "ort_cuda", "cuda_ort", "ort_tensorrt", "tensorrt", "trt"}:
            # ORT references emit one canonical Full row plus one split-shaped
            # reference row for each selected boundary.
            variants = (reference_measurement_variants(recipes.get(canonical_run_id(rid)))
                        if canonical_run_id(rid) == 'ort_tensorrt' else ('full', 'split'))
            total += int('full' in variants) + cases * int('split' in variants)
        elif "_to_" in rid or "to_tensorrt" in rid or "tensorrt_to" in rid or "trt_to" in rid:
            total += cases
        else:
            total += 1
    return total


from .energy.task_budget import campaign_budget_policy


def build_effective_execution_plan(profile: Mapping[str, Any]) -> Dict[str, Any]:
    models = _model_rows(profile)
    logical = _logical_profiles(profile)
    quality_canary = resolve_full_only_quality_canary(
        profile, plan_rows=logical,
    )
    if quality_canary.get("enabled") and quality_canary.get("ok") is not True:
        raise ValueError(
            "full_only_quality_canary_invalid:"
            + ",".join(
                str(value) for value in quality_canary.get("errors") or []
            )
        )
    selection = profile.get("selection_policy") if isinstance(profile.get("selection_policy"), Mapping) else {}
    workflow = profile.get("workflow") if isinstance(profile.get("workflow"), Mapping) else {}
    generic_runtime_enabled = not _truth(
        workflow.get("skip_runtime_benchmarks")
    )
    preset = profile.get("execution_preset") if isinstance(profile.get("execution_preset"), Mapping) else {}
    snapshot = preset.get("snapshot") if isinstance(preset.get("snapshot"), Mapping) else {}
    overrides = preset.get("overrides") if isinstance(preset.get("overrides"), Mapping) else {}
    runtime = snapshot.get("runtime") if isinstance(snapshot.get("runtime"), Mapping) else {}
    native_cfg = runtime.get("native") if isinstance(runtime.get("native"), Mapping) else {}
    benchmark = runtime.get("benchmark") if isinstance(runtime.get("benchmark"), Mapping) else {}
    quality = snapshot.get("quality") if isinstance(snapshot.get("quality"), Mapping) else {}
    official_coco = (
        profile.get("official_coco_evaluation")
        if isinstance(profile.get("official_coco_evaluation"), Mapping)
        else {}
    )
    official_coco_enabled = _truth(
        official_coco.get(
            "enabled", quality.get("official_coco_enabled", False),
        )
    )
    official_coco_required = _truth(
        official_coco.get(
            "required_for_final",
            quality.get("official_coco_required", False),
        )
    )
    ranking = snapshot.get("ranking") if isinstance(snapshot.get("ranking"), Mapping) else {}
    holdout = snapshot.get("holdout") if isinstance(snapshot.get("holdout"), Mapping) else {}
    top_level_ranking = profile.get("ranking_validation") if isinstance(profile.get("ranking_validation"), Mapping) else {}
    data = snapshot.get("data") if isinstance(snapshot.get("data"), Mapping) else {}
    build = snapshot.get("build") if isinstance(snapshot.get("build"), Mapping) else {}
    hailo = build.get("hailo") if isinstance(build.get("hailo"), Mapping) else {}
    snapshot_deepx = build.get("deepx") if isinstance(build.get("deepx"), Mapping) else {}
    materialized_deepx = profile.get("deepx_build") if isinstance(profile.get("deepx_build"), Mapping) else {}
    deepx_preprocessing_explicit = (
        "classification_preprocessing" in materialized_deepx
        or "classification_preprocessing" in snapshot_deepx
    )
    deepx_classification_preprocessing = str(
        materialized_deepx.get("classification_preprocessing")
        or snapshot_deepx.get("classification_preprocessing")
        or "imagenet_mean_std"
    ).strip().lower()
    native_execution_contract = resolve_native_execution_contract(profile)
    cache_guard = cache_verify_guard(profile)
    cache_verify_view = (
        cache_verify_contract_view(profile) if cache_guard else {}
    )

    ids = [str(r.get("id") or r.get("full") or "").strip() for r in logical]
    ids = [x for x in ids if x]
    automatic_refs: list[str] = []
    if "ort_cpu" not in ids:
        automatic_refs.append("ort_cpu")
    if not any(x in ids for x in ("ort_tensorrt", "tensorrt_full", "trt_full")):
        automatic_refs.append("ort_tensorrt")
    quality_execution_location, quality_workers = _quality_execution(profile, snapshot)
    management_reference_profiles = (
        ["ort_cpu"] if quality_execution_location == "central_management" else []
    )
    # The central CPU reference is a semantic-only management job.  It is not
    # dispatched to an accelerator host and must not inflate Generic latency /
    # FPS row or remote-upload counts.
    remote_automatic_refs = [
        rid for rid in automatic_refs
        if not (rid == "ort_cpu" and quality_execution_location == "central_management")
    ]
    effective_ids = [
        rid for rid in ids
        if not (rid == "ort_cpu" and quality_execution_location == "central_management")
    ] + remote_automatic_refs
    if not generic_runtime_enabled:
        # Preserve the requested logical profiles for provenance, but do not
        # describe rows, uploads or remote invocations that the executor is
        # explicitly forbidden to dispatch.
        effective_ids = []
    if quality_canary.get("enabled"):
        # The selected recipes still execute remotely to produce candidate
        # records, but none of them is a Generic performance observation.
        effective_ids = []
    setup_groups: dict[str, list[str]] = {}
    for rid in effective_ids:
        setup_groups.setdefault(_setup_class(rid), []).append(rid)
    # Reference rows execute on the same physical setup selected by the
    # workflow (prefer the DeepX/Jetson host, otherwise the first setup).  They
    # therefore do not create a fourth upload/dispatch group.
    physical_groups = [key for key in setup_groups if key != "gpu_reference_setup"]
    if "deepx_setup" in physical_groups:
        reference_group = "deepx_setup"
    elif physical_groups:
        reference_group = physical_groups[0]
    else:
        reference_group = "gpu_reference_setup"
    # Pure ORT-TensorRT/CUDA reference rows are assigned by the executor to the
    # DeepX setup when present, otherwise to the first selected physical setup.
    # Apply the same rule for every matrix shape (including Hailo-only profiles)
    # so the preview does not invent a separate upload/dispatch group.
    if reference_group != "gpu_reference_setup" and "gpu_reference_setup" in setup_groups:
        setup_groups.setdefault(reference_group, []).extend(
            setup_groups.pop("gpu_reference_setup")
        )

    # Central quality joins the TensorRT producer by physical setup id.  Show
    # the same setup-local companion expansion used by the executor: every
    # accelerator setup receives ``ort_tensorrt``, but only the preferred
    # DeepX setup owns its latency/FPS row.  These companions do not increase
    # Generic performance-row counts; the Hailo copies are quality-only.
    trt_reference_id = next(
        (
            run_id for run_id in effective_ids
            if str(run_id).strip().lower().replace("-", "_")
            in {"ort_tensorrt", "tensorrt", "trt"}
        ),
        "",
    )
    setup_local_trt_quality_only_groups: list[str] = []
    tensorrt_performance_owner_group = ""
    if (
        quality_execution_location == "central_management"
        and trt_reference_id
        and physical_groups
    ):
        tensorrt_performance_owner_group = reference_group
        for group in physical_groups:
            group_runs = setup_groups.setdefault(group, [])
            if trt_reference_id not in group_runs:
                group_runs.append(trt_reference_id)
            if group != tensorrt_performance_owner_group:
                setup_local_trt_quality_only_groups.append(group)
    if quality_canary.get("enabled"):
        expected_quality = [
            dict(row) for row in list(
                quality_canary.get("expected_full_quality_results") or []
            ) if isinstance(row, Mapping)
        ]
        setup_groups = {}
        for row in expected_quality:
            setup_id = str(row.get("setup_id") or "")
            run_id = str(row.get("run_id") or "")
            if setup_id and run_id:
                runs = setup_groups.setdefault(setup_id, [])
                if run_id not in runs:
                    runs.append(run_id)
        physical_groups = list(setup_groups)
        trt_reference_id = "ort_tensorrt"
        setup_local_trt_quality_only_groups = list(setup_groups)
        tensorrt_performance_owner_group = ""

    standard_setup_local_trt_quality_requested = bool(
        not quality_canary.get("enabled")
        and quality_execution_location == "central_management"
        and trt_reference_id
        and physical_groups
    )
    standard_setup_local_trt_quality_contract = (
        _standard_setup_local_trt_quality_contract(
            profile,
            logical_rows=logical,
            model_ids=[str(row.get("id") or "") for row in models],
            requested=standard_setup_local_trt_quality_requested,
        )
    )
    setup_local_trt_quality_companion_identities: list[dict[str, Any]] = []
    if quality_canary.get("enabled"):
        setup_local_trt_quality_companion_identities = [
            {
                **dict(row),
                "identity_source": "explicit_full_only_quality_canary",
            }
            for row in list(
                quality_canary.get("expected_full_quality_identities") or []
            )
            if isinstance(row, Mapping)
            and str(row.get("backend") or "").strip().lower().replace(
                "-", "_"
            ) in {"tensorrt", "ort_tensorrt", "native_tensorrt"}
        ]
    elif standard_setup_local_trt_quality_contract.get("status") == "ready":
        setup_local_trt_quality_companion_identities = [
            dict(row) for row in list(
                standard_setup_local_trt_quality_contract.get(
                    "identities"
                ) or []
            ) if isinstance(row, Mapping)
        ]

    defaults = snapshot.get("defaults") if isinstance(snapshot.get("defaults"), Mapping) else {}
    native_default = _truth(defaults.get("native_enabled"))
    energy_default = _truth(defaults.get("energy_enabled"))
    native_override_present = "native_enabled" in overrides
    energy_override_present = "energy_enabled" in overrides
    native_enabled = _truth(overrides.get("native_enabled")) if native_override_present else native_default
    native_energy_requested = _truth(overrides.get("energy_enabled")) if energy_override_present else energy_default
    native_energy = native_enabled and native_energy_requested
    effective_energy = resolve_effective_energy_config(profile)
    generic_energy_enabled = bool(
        effective_energy.get("generic_energy_enabled")
        and effective_energy.get("measurement_path")
        in {"generic", "native_and_generic"}
        and not effective_energy.get("configuration_errors")
    )
    energy_measurement_path = str(
        effective_energy.get("measurement_path") or "disabled"
    )
    energy_configuration_errors = [
        str(value)
        for value in list(
            effective_energy.get("configuration_errors") or []
        )
        if str(value)
    ]
    requested_single_part2_input = bool(
        selection.get("require_single_part2_input", False)
    )
    native_split_plan = resolve_native_split_plan(profile)
    native_split_requires_single_part2_input = bool(
        native_enabled and native_split_plan.enabled
    )
    effective_single_part2_input = bool(
        requested_single_part2_input
        or native_split_requires_single_part2_input
    )
    model_count = len(models)
    cases_per_model = max(1, int(selection.get("max_accepted_cases_per_model") or 1))
    selection_strategy = str(
        selection.get("selection_strategy") or "stratified_windows"
    ).strip().lower().replace("-", "_")
    score_independent_audit_requested = bool(
        selection_strategy in {
            "score_independent_audit",
            "ranking_audit",
            "deterministic_audit",
        }
        or _truth(selection.get("score_independent_audit_enabled"))
    )
    experiment_purpose = profile_experiment_purpose(profile)
    ranking_requested = _truth(ranking.get("enabled", False))
    ranking_enabled = ranking_requested and experiment_purpose not in {"coverage_integration", "deepx_preprocessing_ab"}
    ranking_min_candidates = max(2, int(ranking.get("minimum_candidates_for_correlation") or 3))
    holdout_audit_default = max(
        1,
        int(
            holdout.get("audit_size")
            or top_level_ranking.get("audit_size")
            or 20
        ),
    )
    selection_audit_size = max(
        1,
        int(
            selection.get("audit_size")
            or top_level_ranking.get("audit_size")
            or holdout_audit_default
        ),
    )
    selection_audit_minimum_valid = max(
        1,
        int(
            selection.get("minimum_valid_audit_candidates")
            or selection.get("minimum_valid_candidates")
            or top_level_ranking.get("minimum_valid_audit_candidates")
            or top_level_ranking.get("minimum_valid_candidates")
            or 10
        ),
    )
    candidate_counts_by_model: dict[str, int] = {}
    score_independent_audit_counts: dict[str, int] = {}
    score_independent_audit_minimum_valid_counts: dict[str, int] = {}
    development_audit_counts: dict[str, int] = {}
    holdout_audit_counts: dict[str, int] = {}
    deployment_shortlist_upper_bound_by_model: dict[str, int] = {}
    execution_union_candidate_counts_min_by_model: dict[str, int] = {}
    execution_union_candidate_counts_upper_bound_by_model: dict[str, int] = {}
    for model in models:
        model_id = str(model.get("id") or "")
        universe = model.get("candidate_universe") if isinstance(model.get("candidate_universe"), Mapping) else {}
        universe_mode = str(universe.get("mode") or holdout.get("candidate_universe") or "").strip().lower().replace("-", "_")
        holdout_role = is_confirmatory_holdout(model.get("evaluation_role"))
        legacy_holdout_audit = bool(
            holdout_role
            and universe_mode
            in {"all_feasible", "deterministic_audit", "audit_universe"}
        )
        audit_enabled_for_model = bool(
            score_independent_audit_requested or legacy_holdout_audit
        )
        deployment_shortlist_upper_bound_by_model[model_id] = cases_per_model
        if audit_enabled_for_model:
            # The global audit controls are authoritative for the new
            # development/hold-out audit path.  Per-model universe values are
            # retained only for legacy hold-out profiles that do not request a
            # global score-independent audit.
            if score_independent_audit_requested:
                audit_count = selection_audit_size
                minimum_valid = selection_audit_minimum_valid
            else:
                audit_count = max(
                    1,
                    int(universe.get("audit_size") or holdout_audit_default),
                )
                minimum_valid = max(
                    1,
                    int(
                        universe.get("minimum_valid_candidates")
                        or universe.get("minimum_valid_audit_candidates")
                        or holdout.get("minimum_valid_candidates")
                        or top_level_ranking.get(
                            "minimum_valid_audit_candidates"
                        )
                        or 10
                    ),
                )
            score_independent_audit_counts[model_id] = audit_count
            score_independent_audit_minimum_valid_counts[model_id] = (
                minimum_valid
            )
            if holdout_role:
                # Backward-compatible field retained for existing dashboards.
                holdout_audit_counts[model_id] = audit_count
            else:
                development_audit_counts[model_id] = audit_count

            # The executor deduplicates the score-independent audit and the
            # predictor shortlist.  Before selection we cannot know their
            # overlap, so expose the honest configured range and size all
            # resource estimates with its upper bound.  This prevents a
            # 4-audit + 1-shortlist run from being previewed as one candidate.
            execution_union_candidate_counts_min_by_model[model_id] = max(
                audit_count, cases_per_model
            )
            execution_union_candidate_counts_upper_bound_by_model[model_id] = (
                audit_count + cases_per_model
            )
            # Compatibility field describes the candidate population used by
            # ranking gates, not the (possibly larger) execution union.
            candidate_counts_by_model[model_id] = audit_count
        else:
            candidate_counts_by_model[model_id] = cases_per_model
            execution_union_candidate_counts_min_by_model[model_id] = (
                cases_per_model
            )
            execution_union_candidate_counts_upper_bound_by_model[model_id] = (
                cases_per_model
            )
    shortfall_models = {
        model_id: max(0, ranking_min_candidates - count)
        for model_id, count in candidate_counts_by_model.items()
        if count < ranking_min_candidates
    }
    plan_warnings: list[dict[str, Any]] = []
    if energy_configuration_errors:
        plan_warnings.append({
            "id": "energy_configuration_blocked",
            "severity": "error",
            "message": (
                "Energy requested, not started: planning blocked by "
                + "; ".join(energy_configuration_errors)
            ),
            "categories": ["energy_configuration"],
            "errors": energy_configuration_errors,
        })
    if (
        ranking_enabled and shortfall_models and generic_runtime_enabled
        and not quality_canary.get("enabled")
    ):
        plan_warnings.append({
            "id": "ranking_candidate_shortfall",
            "severity": "warning",
            "message": (
                f"Ranking transfer requires at least {ranking_min_candidates} measured candidates per comparable "
                f"model/backend/contract group. Insufficient planned candidate counts: {shortfall_models}. "
                "For score-independent audits, this count comes from the audit universe and is not limited by max_accepted_cases_per_model. "
                "Performance and task-quality rows remain valid; Spearman/Kendall/Top-k transfer will be reported as insufficient_candidates."
            ),
            "selected_cases_per_model": int(cases_per_model),
            "minimum_candidates_for_correlation": int(ranking_min_candidates),
            "candidate_counts_by_model": candidate_counts_by_model,
            "shortfall_by_model": shortfall_models,
        })
    generic_logical_runs_per_model = len(effective_ids)
    expected_generic_result_rows_min_by_model = {
        model_id: _expected_normalized_rows(effective_ids, candidate_count, logical)
        for model_id, candidate_count
        in execution_union_candidate_counts_min_by_model.items()
    }
    expected_generic_result_rows_by_model = {
        model_id: _expected_normalized_rows(effective_ids, candidate_count, logical)
        for model_id, candidate_count
        in execution_union_candidate_counts_upper_bound_by_model.items()
    }
    expected_row_values = sorted(set(expected_generic_result_rows_by_model.values()))
    generic_rows_per_model = expected_row_values[0] if len(expected_row_values) == 1 else None
    generic_rows_total = sum(expected_generic_result_rows_by_model.values())
    generic_rows_min_total = sum(
        expected_generic_result_rows_min_by_model.values()
    )
    # All run IDs mapped to the same physical setup are executed by one
    # setup-local benchmark-suite invocation. Rows remain distinct scientific
    # observations, while upload/extraction and TensorRT caches are shared.
    remote_invocations_per_model = len([k for k, v in setup_groups.items() if v])
    cold_suite_uploads_per_model = remote_invocations_per_model

    validation_items = dict(data.get("validation_items") or {})
    classification_models = sum(1 for row in models if str(row.get("task") or "").strip().lower() == "classification")
    detection_models = sum(1 for row in models if str(row.get("task") or "").strip().lower() == "detection")
    cls_items = max(0, int(validation_items.get("classification") or 0))
    det_items = max(0, int(validation_items.get("detection") or 0))
    # Classification embeds one image per selected item plus one subset
    # manifest. Detection embeds one image and one sidecar per item plus a
    # filtered COCO file and one subset manifest. These are estimates for the
    # pre-run plan; the authoritative values are written by suite bundling.
    estimated_validation_files = (classification_models * (cls_items + (1 if cls_items else 0))) + (detection_models * ((2 * det_items) + (2 if det_items else 0)))

    return {
        "schema": "onnx-splitpoint/effective-execution-plan",
        "schema_version": 1,
        "artifact_policy": CACHE_VERIFY_ONLY if cache_guard else "normal",
        "cache_verify_expected_plan": (
            dict(cache_guard.get("expected_plan") or {}) if cache_guard else {}
        ),
        "cache_verify_actual_plan": cache_verify_view,
        "compiler_dispatch_allowed": False if cache_guard else True,
        "generic_runtime_enabled": generic_runtime_enabled,
        "quality_canary": quality_canary,
        "quality_canary_enabled": bool(quality_canary.get("enabled")),
        "quality_canary_execution_scope": str(
            quality_canary.get("execution_scope") or ""
        ),
        "expected_full_quality_identities": list(
            quality_canary.get("expected_full_quality_identities") or []
        ),
        "expected_full_quality_ids": list(
            quality_canary.get("expected_full_quality_ids") or []
        ),
        "expected_full_quality_setup_ids": list(
            quality_canary.get("expected_full_quality_setup_ids") or []
        ),
        "expected_full_quality_results_per_model": int(
            len(quality_canary.get("expected_full_quality_identities") or [])
        ),
        "expected_full_quality_results_total": int(
            model_count
            * len(quality_canary.get("expected_full_quality_identities") or [])
        ),
        "run_mode": str(preset.get("id") or "standard"),
        "run_mode_label": str(preset.get("label") or preset.get("id") or "Standard"),
        "models": [str(r.get("id") or "") for r in models],
        "model_count": model_count,
        "cases_per_model": cases_per_model,
        "deployment_shortlist_cases_per_model": cases_per_model,
        # Compatibility name preserves the user's explicit selection setting.
        "require_single_part2_input": requested_single_part2_input,
        "requested_require_single_part2_input": (
            requested_single_part2_input
        ),
        "native_split_requires_single_part2_input": (
            native_split_requires_single_part2_input
        ),
        "native_split_backends": list(
            native_split_plan.selected_split_backends
        ),
        "native_split_plan_source": native_split_plan.source,
        "effective_require_single_part2_input": (
            effective_single_part2_input
        ),
        "native_multi_input_policy": (
            "reject_and_backfill_from_frozen_prediction"
            if native_split_requires_single_part2_input
            else "not_applicable"
        ),
        "candidate_counts_by_model": candidate_counts_by_model,
        "candidate_count_semantics": "ranking_candidate_population",
        "score_independent_audit_enabled": bool(
            score_independent_audit_counts
        ),
        "score_independent_audit_requested": (
            score_independent_audit_requested
        ),
        "score_independent_audit_size": selection_audit_size,
        "score_independent_audit_minimum_valid": (
            selection_audit_minimum_valid
        ),
        "score_independent_audit_counts": score_independent_audit_counts,
        "score_independent_audit_minimum_valid_counts": (
            score_independent_audit_minimum_valid_counts
        ),
        "development_audit_counts": development_audit_counts,
        "holdout_audit_size_default": holdout_audit_default,
        "holdout_audit_counts": holdout_audit_counts,
        "deployment_shortlist_upper_bound_by_model": (
            deployment_shortlist_upper_bound_by_model
        ),
        "execution_union_candidate_counts_min_by_model": (
            execution_union_candidate_counts_min_by_model
        ),
        "execution_union_candidate_counts_upper_bound_by_model": (
            execution_union_candidate_counts_upper_bound_by_model
        ),
        "execution_union_candidate_count_min_total": sum(
            execution_union_candidate_counts_min_by_model.values()
        ),
        "execution_union_candidate_count_upper_bound_total": sum(
            execution_union_candidate_counts_upper_bound_by_model.values()
        ),
        "execution_union_count_semantics": (
            "configured_range_before_audit_shortlist_deduplication; "
            "resource_estimates_use_upper_bound"
        ),
        "shortlist": int(selection.get("preferred_shortlist") or cases_per_model),
        "selection_strategy": selection_strategy,
        "logical_run_profiles": ids,
        "automatic_reference_profiles": automatic_refs,
        "management_reference_profiles": management_reference_profiles,
        "quality_execution_location": quality_execution_location,
        "quality_workers": quality_workers,
        "effective_generic_run_ids": effective_ids,
        "generic_logical_runs_per_model": generic_logical_runs_per_model,
        "generic_logical_runs_total": model_count * generic_logical_runs_per_model,
        "expected_generic_result_rows_min_by_model": (
            expected_generic_result_rows_min_by_model
        ),
        "expected_generic_result_rows_by_model": expected_generic_result_rows_by_model,
        "expected_generic_result_rows_min_total": generic_rows_min_total,
        "expected_generic_result_rows_total": generic_rows_total,
        "expected_generic_result_rows_semantics": (
            "configured_minimum_and_upper_bound; legacy total/by_model fields "
            "contain the resource-safe upper bound"
        ),
        "generic_rows_per_model": generic_rows_per_model,
        "generic_rows_total": generic_rows_total,
        "setup_groups": setup_groups,
        "setup_local_tensorrt_quality_companions_requested": bool(
            trt_reference_id and physical_groups
            and quality_execution_location == "central_management"
        ),
        "setup_local_tensorrt_quality_companions": bool(
            setup_local_trt_quality_companion_identities
            and all(
                str(
                    row.get("quality_companion_endpoint_id")
                    or row.get("id") or ""
                ).strip()
                for row in setup_local_trt_quality_companion_identities
            )
        ),
        "setup_local_tensorrt_quality_companion_identities": (
            setup_local_trt_quality_companion_identities
        ),
        "setup_local_tensorrt_quality_companion_contract": (
            standard_setup_local_trt_quality_contract
        ),
        "setup_local_tensorrt_quality_companion_identity_ready": bool(
            setup_local_trt_quality_companion_identities
            and all(
                str(
                    row.get("quality_companion_endpoint_id")
                    or row.get("id") or ""
                ).strip()
                for row in setup_local_trt_quality_companion_identities
            )
        ),
        "setup_local_tensorrt_run_id": trt_reference_id,
        "setup_local_tensorrt_quality_only_groups": (
            setup_local_trt_quality_only_groups
        ),
        "tensorrt_performance_owner_group": (
            tensorrt_performance_owner_group
        ),
        "performance_claims_emitted": (
            False if quality_canary.get("enabled") else None
        ),
        "remote_run_invocations_per_model": remote_invocations_per_model,
        "remote_run_invocations_total": model_count * remote_invocations_per_model,
        "cold_suite_uploads_per_model": cold_suite_uploads_per_model,
        "cold_suite_uploads_total": model_count * cold_suite_uploads_per_model,
        "warm_suite_uploads_total": 0,
        # Backward-compatible field names retained for old dashboards.
        "batched_remote_dispatches_per_model": remote_invocations_per_model,
        "batched_remote_dispatches_total": model_count * remote_invocations_per_model,
        "uploads_per_model_setup": 1,
        "generic_energy_enabled": generic_energy_enabled,
        "native_enabled": native_enabled,
        "native_default": native_default,
        "native_override_present": native_override_present,
        "native_override_active": native_override_present and native_enabled != native_default,
        "native_energy_default": energy_default,
        "native_energy_override_present": energy_override_present,
        "native_energy_override_active": energy_override_present and native_energy_requested != energy_default,
        "native_energy_requested": native_energy_requested,
        "native_full_baselines": _resolved_native_full_enabled(
            profile,
            preset,
            native_cfg,
            native_enabled=native_enabled,
        ),
        "native_energy_enabled": native_energy,
        "native_energy_task_budget": campaign_budget_policy(profile.get("native_producers") or {}),
        "energy_measurement_path": energy_measurement_path,
        "energy_configuration_errors": energy_configuration_errors,
        "energy_plan_blocked": bool(energy_configuration_errors),
        "calibration_items": dict(data.get("calibration_items") or {}),
        "validation_items": validation_items,
        "validation_embedding_policy": "exact_run_mode_subset",
        "estimated_validation_files_total": int(estimated_validation_files),
        "classification_model_count": int(classification_models),
        "detection_model_count": int(detection_models),
        "bootstrap_repetitions": int(quality.get("bootstrap_repetitions") or 0),
        "official_coco_enabled": bool(official_coco_enabled),
        "official_coco_required": bool(official_coco_required),
        "ranking_enabled": bool(ranking_enabled),
        "ranking_requested": bool(ranking_requested),
        "experiment_purpose": experiment_purpose,
        "experiment_scope": {
            "model_coverage_count": model_count,
            "planned_candidate_counts_by_model": candidate_counts_by_model,
            "observed_comparable_valid_candidates": None,
            "ranking_intended": bool(ranking_enabled),
            "ranking_admission_status": (
                "not_requested_for_integration" if experiment_purpose in {"coverage_integration", "deepx_preprocessing_ab"}
                else "insufficient_planned_candidates" if ranking_enabled and shortfall_models
                else "requires_observed_comparable_candidates" if ranking_enabled
                else "not_requested"
            ),
            "scope_note": "Model coverage is distinct from ranking transfer; repetitions and other setups do not add split candidates.",
        },
        "ranking_minimum_candidates": int(ranking_min_candidates),
        "ranking_candidate_shortfall": max(shortfall_models.values(), default=0) if ranking_enabled else 0,
        "warnings": plan_warnings,
        "benchmark_warmup": int(benchmark.get("warmup") or 0),
        "benchmark_runs": int(benchmark.get("runs") or 1),
        "native_frames": int(native_execution_contract["frames"]),
        "native_warmup": int(native_execution_contract["warmup"]),
        "native_performance_repetitions": int(
            native_execution_contract["repetitions"]
        ),
        "native_queue_depth": int(
            native_execution_contract["queue_depth"]
        ),
        "native_inflight": int(native_execution_contract["inflight"]),
        "native_execution_contract": native_execution_contract,
        "native_execution_contract_sha256": str(
            native_execution_contract["contract_sha256"]
        ),
        "hailo_preset": str(hailo.get("preset") or ""),
        "hailo_optimization_level": int(hailo.get("optimization_level") or 0),
        "hailo_calibration_storage": str(hailo.get("calibration_storage") or "memory"),
        "deepx_classification_preprocessing": deepx_classification_preprocessing,
        "deepx_classification_preprocessing_explicit": bool(
            deepx_preprocessing_explicit
        ),
        "deepx_full_cache_contract": (
            "v2_exact_explicit" if deepx_preprocessing_explicit else "legacy_implicit"
        ),
        "deepx_cache_dir": str(
            materialized_deepx.get("cache_dir")
            or snapshot_deepx.get("cache_dir")
            or "~/Models/BackendArtifacts/deepx"
        ),
        "tensorrt_remote_cache": bool(((runtime.get("remote_cache") or {}) if isinstance(runtime.get("remote_cache"), Mapping) else {}).get("reuse_tensorrt_engines", True)),
        "notes": [
            "Generic-Runner energy is disabled by design; task validation and broad candidate screening remain Generic-Runner responsibilities.",
            "Native energy is measured only after Native contract validation and with matching Native-Full baselines.",
            "Run IDs sharing a physical setup execute in one setup-local benchmark invocation and reuse the same content-addressed suite and TensorRT cache.",
        ],
    }


def execution_plan_text(plan: Mapping[str, Any]) -> str:
    groups = plan.get("setup_groups") if isinstance(plan.get("setup_groups"), Mapping) else {}
    union_min = (
        plan.get("execution_union_candidate_counts_min_by_model")
        if isinstance(
            plan.get("execution_union_candidate_counts_min_by_model"),
            Mapping,
        )
        else {}
    )
    union_max = (
        plan.get("execution_union_candidate_counts_upper_bound_by_model")
        if isinstance(
            plan.get("execution_union_candidate_counts_upper_bound_by_model"),
            Mapping,
        )
        else {}
    )
    union_ranges = {
        str(model_id): (
            str(union_max.get(model_id))
            if union_min.get(model_id) == union_max.get(model_id)
            else f"{union_min.get(model_id)}–{union_max.get(model_id)}"
        )
        for model_id in union_max
    }
    generic_rows_min = plan.get(
        "expected_generic_result_rows_min_total",
        plan.get("expected_generic_result_rows_total", plan.get("generic_rows_total")),
    )
    generic_rows_max = plan.get(
        "expected_generic_result_rows_total", plan.get("generic_rows_total")
    )
    generic_rows_label = (
        str(generic_rows_max)
        if generic_rows_min == generic_rows_max
        else f"{generic_rows_min}–{generic_rows_max} (upper-bound sized)"
    )
    lines = [
        f"Run mode: {plan.get('run_mode_label')} ({plan.get('run_mode')})",
        f"Profile purpose: {plan.get('experiment_purpose', 'legacy_unspecified')} · ranking: {(plan.get('experiment_scope') or {}).get('ranking_admission_status', 'unavailable')}",
        "Model coverage and ranking transfer are separate; comparable valid candidate counts require observed results.",
        (
            f"Artifact policy: {plan.get('artifact_policy', 'normal')} · "
            f"compiler dispatch={'allowed' if plan.get('compiler_dispatch_allowed', True) else 'blocked'}"
        ),
        f"Models: {plan.get('model_count')} — {', '.join(plan.get('models') or []) or 'none'}",
        f"Deployment shortlist/model: {plan.get('deployment_shortlist_cases_per_model', plan.get('cases_per_model'))} · shortlist={plan.get('shortlist')} · strategy={plan.get('selection_strategy')}",
        (
            "Part-2 input count = 1: "
            f"requested={'on' if plan.get('requested_require_single_part2_input', plan.get('require_single_part2_input')) else 'off'} · "
            f"effective={'on' if plan.get('effective_require_single_part2_input', plan.get('require_single_part2_input')) else 'off'} "
            + (
                "(Native capability: reject and deterministically backfill; "
                "Generic remains technically multi-input/multi-output capable)"
                if plan.get('native_split_requires_single_part2_input')
                else "(selection only; Generic remains multi-input/multi-output capable)"
            )
        ),
        (
            "Score-independent audit candidates: "
            f"{plan.get('score_independent_audit_counts') or 'none'} "
            f"(development={plan.get('development_audit_counts') or 'none'}; "
            f"hold-out={plan.get('holdout_audit_counts') or 'none'}; "
            f"minimum valid={plan.get('score_independent_audit_minimum_valid_counts') or 'none'})"
        ),
        (
            "Execution union candidates/model: "
            f"{union_ranges or 'none'} "
            "(audit + deployment shortlist, deduplicated; configured range)"
        ),
        (
            f"Generic normalized result rows (planned): {generic_rows_label} "
            f"({plan.get('expected_generic_result_rows_by_model') or str(plan.get('generic_rows_per_model')) + '/model'}; "
            f"logical runs={plan.get('generic_logical_runs_total', plan.get('generic_rows_total'))})"
        ),
        f"Remote benchmark invocations: {plan.get('remote_run_invocations_total')} ({plan.get('remote_run_invocations_per_model')}/model)",
        f"Suite uploads: cold cache <= {plan.get('cold_suite_uploads_total')} ({plan.get('cold_suite_uploads_per_model')}/model); warm cache = {plan.get('warm_suite_uploads_total')}",
        "Setup groups: " + "; ".join(f"{k}=[{', '.join(v)}]" for k, v in groups.items()),
        f"Validation: exact run-mode subsets {plan.get('validation_items')} · estimated embedded files={plan.get('estimated_validation_files_total')}",
        f"Task Quality: bootstrap={plan.get('bootstrap_repetitions')} · benchmark warmup/runs={plan.get('benchmark_warmup')}/{plan.get('benchmark_runs')}",
        (
            f"Quality reference: {plan.get('quality_execution_location')} · "
            f"workers={plan.get('quality_workers')} · "
            f"management profiles={plan.get('management_reference_profiles') or 'none'}"
        ),
        f"Hailo: preset={plan.get('hailo_preset')} · opt={plan.get('hailo_optimization_level')} · calibration storage={plan.get('hailo_calibration_storage')}",
        (
            f"DeepX classification preprocessing: {plan.get('deepx_classification_preprocessing')} · "
            f"cache contract={plan.get('deepx_full_cache_contract')} · "
            f"cache root={plan.get('deepx_cache_dir')}"
        ),
        f"Native: {'on' if plan.get('native_enabled') else 'off'} · full baselines={'on' if plan.get('native_full_baselines') else 'off'} · frames/warmup={plan.get('native_frames')}/{plan.get('native_warmup')} · performance repetitions={plan.get('native_performance_repetitions')} · queue/inflight={plan.get('native_queue_depth')}/{plan.get('native_inflight')} · contract={str(plan.get('native_execution_contract_sha256') or '')[:12]}",
        (
            f"Energy: Generic={'on' if plan.get('generic_energy_enabled') else 'off'} · "
            f"Native={'on' if plan.get('native_energy_enabled') else 'off'} "
            f"(mode default={'on' if plan.get('native_energy_default') else 'off'}; "
            f"{'profile override' if plan.get('native_energy_override_present') else 'mode default'}) · "
            f"path={plan.get('energy_measurement_path')} · "
            f"planning={'blocked: ' + '; '.join(plan.get('energy_configuration_errors') or []) if plan.get('energy_plan_blocked') else 'ready'}"
        ),
        f"Remote TensorRT cache: {'on' if plan.get('tensorrt_remote_cache') else 'off'}",
    ]
    if plan.get("quality_canary_enabled"):
        lines.extend([
            (
                "Quality canary: full_only · performance claims=off · "
                f"expected Full quality={plan.get('expected_full_quality_results_total')} "
                f"({plan.get('expected_full_quality_results_per_model')}/model)"
            ),
            "Full-quality identities: " + "; ".join(
                f"{row.get('id')}={row.get('source_run_id')}@{row.get('setup_id')}/full"
                for row in list(
                    plan.get("expected_full_quality_identities") or []
                ) if isinstance(row, Mapping)
            ),
        ])
    warnings = [w for w in list(plan.get("warnings") or []) if isinstance(w, Mapping)]
    if warnings:
        lines.append("Warnings:")
        for warning in warnings:
            lines.append(f"  - {warning.get('message') or warning.get('id')}")
    return "\n".join(lines)


def execution_plan_markdown(plan: Mapping[str, Any]) -> str:
    return "# Effective execution plan\n\n```text\n" + execution_plan_text(plan) + "\n```\n"
