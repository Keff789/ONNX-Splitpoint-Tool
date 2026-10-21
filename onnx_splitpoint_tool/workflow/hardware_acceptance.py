"""Finite, fresh hardware acceptance through the ordinary profile workflow.

The original campaign supplies immutable artifacts and references. A separate
EvaluationRun owns every new attempt, lease, task budget and result. This path
does not resume the source campaign or invoke its remaining queue.
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import sys
import threading
from typing import Any, Mapping

from .artifacts import now_iso, write_json
from .contracts import StageResult, WorkflowRunResult
from .evidence_status import workflow_completion_projection

SCHEMA = "onnx-splitpoint/hardware-acceptance/v1"
_ENVIRONMENT_LOCK = threading.Lock()


def enabled(profile: Mapping[str, Any]) -> bool:
    config = profile.get("hardware_acceptance") or {}
    if not isinstance(config, Mapping):
        raise ValueError("hardware_acceptance must be a mapping")
    value = config.get("enabled", False)
    if type(value) is not bool:
        raise ValueError("hardware_acceptance.enabled must be boolean")
    return value


def file_binding(value: Any) -> dict[str, Any]:
    declared = dict(value) if isinstance(value, Mapping) else {"path": value}
    path = Path(str(declared.get("path") or "")).expanduser()
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError("regular absolute source evidence required: " + str(path))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if declared.get("sha256") and declared["sha256"] != digest:
        raise ValueError("source evidence changed: " + str(path))
    return {"path": str(path.resolve()), "sha256": digest}


def _variant(value: Any) -> str:
    text = str(value).strip().lower().replace(" ", "_")
    if text not in {"native", "generic_completed"}:
        raise ValueError("unsupported acceptance runtime: " + text)
    return text


def validate_request(profile: Mapping[str, Any], options=None) -> dict[str, Any]:
    """Validate finite selection and unchanged recipe before any dispatch."""
    if not enabled(profile):
        raise ValueError("hardware acceptance is disabled")
    config = copy.deepcopy(dict(profile["hardware_acceptance"]))
    if config.get("schema", SCHEMA) != SCHEMA:
        raise ValueError("unsupported hardware acceptance schema")
    mode = config.get("mode", "scientific")
    if mode not in {"scientific", "functional"}:
        raise ValueError("unsupported hardware acceptance mode")
    phase = config.get("phase", "complete")
    if phase not in {"complete", "numerical"}:
        raise ValueError("unsupported hardware acceptance phase")
    repetitions = 1 if mode == "functional" else 3
    cells = config.get("cells")
    if not isinstance(cells, list) or not cells or len(cells) > 21:
        raise ValueError("hardware acceptance requires 1 through 21 explicit cells")
    models = {str(row.get("id") or row.get("model_id"))
              for row in (profile.get("model_suite") or {}).get("primary", [])}
    setups = {str(value) for value in (profile.get("hardware") or {}).get("selected_setups", [])}
    if not models or not setups:
        raise ValueError("acceptance requires the normal profile model/setup selection")
    native = profile.get("native_producers") or {}
    energy = native.get("energy") or {}
    regular = {"frames": native.get("frames"), "warmup": native.get("warmup"),
               "repetitions": native.get("repetitions"), "duration_s": energy.get("duration_s"),
               "physical_scope": energy.get("physical_scope"), "window_label": energy.get("window_label"),
               "task_budget": copy.deepcopy(energy.get("task_budget") or {})}
    # These are the existing Revision-3 supported runner/collector contracts.
    # No hidden smoke reduction or dataset-quality expansion is permitted.
    if tuple(regular[key] for key in ("frames", "warmup", "repetitions", "duration_s")) != (1000, 100, repetitions, 60):
        raise ValueError(f"acceptance adapters require regular 1000/100/{repetitions} and 60-second budgets for {mode} mode")
    if regular["physical_scope"] != "FS" or regular["window_label"] != "command":
        raise ValueError("acceptance requires the regular FS command measurement boundary")
    from ..energy.task_budget import campaign_budget_policy
    policy = campaign_budget_policy(native)
    if not policy:
        raise ValueError("acceptance requires the regular enabled task budget")
    if options is not None:
        resumable_native = (native.get("deepx_runtime_mode") == "A4"
                            and all(isinstance(cell, Mapping) and cell.get("runtime_variants") == ["native"]
                                    and (cell.get("backend") or cell.get("native_backend")) in {"deepx_to_trt", "native_full_deepx"}
                                    for cell in cells))
        if any((options.resume and not resumable_native, options.dry_run, options.skip_benchmarks, options.no_remote,
                options.rerun_generated_only, options.force_stage, options.stop_after)):
            raise ValueError("acceptance starts a fresh bounded hardware job; shortcut options are invalid")
        if options.execution_mode != "generate_and_run":
            raise ValueError("acceptance requires the normal generate_and_run profile intent")
    seen = set()
    normalized = []
    for value in cells:
        if not isinstance(value, Mapping):
            raise ValueError("acceptance cell must be a mapping")
        cell = copy.deepcopy(dict(value))
        full = str(cell.get("case_id") or "") == "full"
        key = (str(cell.get("model_id") or ""), str(cell.get("setup_id") or ""), full)
        if key in seen:
            raise ValueError("one selected split per model/setup is required")
        seen.add(key)
        if key[0] not in models or key[1] not in setups:
            raise ValueError("acceptance cell is outside the saved profile model/setup selection")
        if not full and not re.fullmatch(r"b[0-9]+", str(cell.get("case_id") or "")):
            raise ValueError("acceptance requires an explicit existing bNNN split identity")
        variants = [_variant(item) for item in cell.get("runtime_variants", ["native", "generic_completed"])]
        if len(variants) != len(set(variants)) or not variants:
            raise ValueError("acceptance runtime variants must be unique and explicit")
        if mode == "functional" and len(variants) != 1:
            raise ValueError("functional acceptance requires one explicitly assigned runtime per cell")
        cell["runtime_variants"] = [v for v in ("native", "generic_completed") if v in variants]
        cell.setdefault("backend", cell.get("native_backend") or cell.get("measurement_direction") or cell.get("profile_split_run_id"))
        if full and (cell["backend"] != "native_full_deepx" or variants != ["native"] or mode != "scientific"):
            raise ValueError("Full acceptance requires an explicit scientific Native DeepX Full cell")
        cell.setdefault("native_source_plan", cell.get("native_performance_plan"))
        cell.setdefault("generic_source_plan", cell.get("generic_performance_plan"))
        cell.setdefault("full_baseline_references", cell.get("baselines", []))
        if phase == "numerical" and (cell["backend"] not in {"native_full_deepx", "deepx_to_trt"}
                                      or variants != ["native"] or native.get("deepx_runtime_mode") != "A4"
                                      or mode != "scientific"):
            raise ValueError("numerical acceptance phase is confined to explicit Native DeepX A4 cells")
        normalized.append(cell)
    report = Path(str(config.get("source_report_root") or "")).expanduser()
    primary = Path(str(config.get("source_primary_run") or "")).expanduser()
    if not report.is_absolute() or not report.is_dir() or not primary.is_absolute() or not primary.is_dir():
        raise ValueError("existing absolute source report and primary run directories are required")
    config.update(cells=normalized, regular_budget=regular, schema=SCHEMA, mode=mode, phase=phase,
                  source_report_root=str(report.resolve()), source_primary_run=str(primary.resolve()))
    return config


def summary_lines(profile: Mapping[str, Any]) -> list[str]:
    config = validate_request(profile)
    cells, budget = config["cells"], config["regular_budget"]
    paths = sum(len(cell["runtime_variants"]) for cell in cells)
    full_count = sum(cell["case_id"] == "full" for cell in cells)
    return [f"Hardware-Abnahme: {len(cells)} feste Zellen, {paths} frische Pfade ({full_count} Full); je Netz/Setup höchstens eine Splitposition.",
            "Numerische Phase: unabhängige serielle Referenzen und gemischte A4-Bilder, danach höchstens 64 Smoke-Tasks, 0 Warmups; keine Energieaufnahme." if config["phase"] == "numerical" else
            "Einzelproben zur Funktionsabnahme; keine neuen wissenschaftlichen Kohorten oder statistischen Claims." if config["mode"] == "functional" else "Reguläre Abnahme mit drei Wiederholungen je Pfad.",
            "Aufnahmen: keine Energie in der numerischen Phase; die regulären Capture-Slots bleiben unbenutzt." if config["phase"] == "numerical" else
            f"Aufnahmen: je Pfad {budget['repetitions']} gültige Energie-Repeats à {budget['duration_s']} s, FS/command; reguläre Retries {budget['task_budget']}.",
            "Numerischer Smoke: warmup=0, tasks=64, Prozesse=1 je Zelle; zusätzlich die expliziten seriellen und gemischten Bildprüfungen." if config["phase"] == "numerical" else
            f"Timing/Task: warmup={budget['warmup']}, tasks={budget['frames']}, Prozesse={budget['repetitions']}.",
            "Vorhandene Artefakte und passende Full/Quality-Referenzen werden geprüft und wiederverwendet; neue Dataset-Quality wird nicht gestartet.",
            "Die alte Restkampagne bleibt pausiert. Prepare und Starts gehören ausschließlich zu diesem neuen EvaluationRun.",
            *[f"  {cell['model_id']} | {cell['setup_id']} | {cell['case_id']} | {','.join(cell['runtime_variants'])}" for cell in cells]]


def _contains_identity(payload, cell) -> bool:
    if isinstance(payload, Mapping):
        row = {**payload, **(payload.get("identity") or {})} if isinstance(payload.get("identity"), Mapping) else payload
        model = row.get("model_id", row.get("model"))
        setup = row.get("setup_id", row.get("setup"))
        cases = {str(row.get(key) or "") for key in ("case_id", "case")}
        if (model, setup) == (cell["model_id"], cell["setup_id"]) and cell["case_id"] in cases:
            return True
        return any(_contains_identity(value, cell) for value in payload.values())
    if isinstance(payload, list):
        return any(_contains_identity(value, cell) for value in payload)
    return False


def _bound_cell(cell, variant):
    value = copy.deepcopy(cell)
    if value.get("blocked_reason"):
        return value
    bindings = {}
    for variant in [variant]:
        key = "native_source_plan" if variant == "native" else "generic_source_plan"
        source = value.get(key)
        if not isinstance(source, Mapping):
            source = {"path": source, "sha256": value.get(key + "_sha256")}
        binding = file_binding(source)
        payload = json.loads(Path(binding["path"]).read_text())
        if not _contains_identity(payload, value):
            raise ValueError("selected cell identity is absent from " + binding["path"])
        if variant == "generic_completed":
            candidates = []
            for item in payload.get("rows", []):
                identity = item.get("identity") or item
                if (isinstance(identity, Mapping)
                        and (identity.get("model_id", identity.get("model")), identity.get("setup_id", identity.get("setup")), identity.get("case_id", identity.get("case")))
                        == (value["model_id"], value["setup_id"], value["case_id"])
                        and (not value.get("generic_case_key") or item.get("case_key") == value["generic_case_key"])
                        and identity.get("comparison_backend")):
                    candidates.append(str(identity["comparison_backend"]))
            if len(candidates) != 1:
                raise ValueError("selected Generic comparison backend must be uniquely bound to its source plan")
            value["comparison_backend"] = candidates[0]
        bindings[key] = binding
        # Existing adapter APIs accept the actual source file path.
        value[key] = binding
    value["source_bindings"] = bindings
    return value


def _serialize(value):
    if isinstance(value, Mapping):
        return {str(key): _serialize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return repr(value)


def _accepted_result(result, context):
    result = dict(result)
    if result.get("status") not in {"ok", "passed", "PASS", "completed"}:
        return result
    if (result.get("fresh_hardware_executed") is not True
            or type(result.get("new_capture_count")) is not int
            or result["new_capture_count"] < context["expected_repetitions"]):
        from .hardware_acceptance_native import verify_same_run_resume_result
        if not verify_same_run_resume_result(result, context):
            raise ValueError("hardware PASS requires a new DUT/collector execution or verified same-Run retained slots")
    repeats = result.get("valid_repeats") or 0
    if isinstance(repeats, Mapping):
        valid = all(repeats.get(key) == context["expected_repetitions"] for key in ("performance", "energy"))
    else:
        valid = int(repeats) == context["expected_repetitions"]
    if not valid:
        raise ValueError("hardware PASS requires all regular valid repeats")
    if context.get("acceptance_mode") == "functional":
        if (result.get("acceptance_mode") != "functional"
                or result.get("evaluation_role") != "functional_acceptance"
                or result.get("functional_acceptance") is not True
                or result.get("diagnostic_only") is not True
                or any(result.get(key) is not False for key in (
                    "scientific_cohort_complete", "scientific_claim_eligible", "scientific_statistics_published"))):
            raise ValueError("functional PASS requires explicit diagnostic-only nonclaimable evidence")
        if (result.get("output_validation", {}).get("status") != "passed"
                or result.get("import_projection", {}).get("normal_report_verified") is not True):
            raise ValueError("functional PASS requires actual numeric oracle validation and normal result import")
    paths = result.get("evidence_paths") or []
    if not paths:
        raise ValueError("hardware PASS requires fresh original evidence paths")
    for raw in paths:
        path = Path(str(raw)).resolve(strict=True)
        if not path.is_file() or not path.is_relative_to(Path(context["run_dir"])):
            raise ValueError("fresh evidence must belong to the new acceptance EvaluationRun")
    result["status"] = "ok"
    return result


def _adapter(variant):
    suffix = "native" if variant == "native" else "generic"
    module = importlib.import_module(".hardware_acceptance_" + suffix, __package__)
    return getattr(module, "prepare_" + suffix), getattr(module, "run_" + suffix)


def _accepted_numerical_result(result, context):
    value = dict(result)
    if value.get("status") != "numerical_passed":
        return value
    retained = value.get("same_run_numerical_retained") is True
    if retained and value != _retained_numerical_result(context["acceptance_cell"], context["prepared_native"], context):
        raise ValueError("numerical Resume projection differs from its unchanged original PASS")
    n = value.get("image_count")
    if (type(n) is not int or not 1 <= n <= 32 or value.get("serial_reference_count") != n
            or value.get("mixed_completed_count") != n or value.get("smoke_completed_count") != 64
            or any(type(value.get(key)) is not int for key in ("serial_reference_count", "mixed_completed_count",
                                                              "smoke_completed_count", "new_capture_count", "new_performance_processes"))
            or value.get("deepx_runtime_mode") != "A4" or value.get("device_completion_proven") is not True
            or value.get("fresh_hardware_executed") is not (False if retained else True) or value.get("new_capture_count") != 0
            or value.get("new_performance_processes") != 0
            or any(value.get(key) is not False for key in ("scientific_claim_eligible", "scientific_cohort_complete",
                                                          "scientific_statistics_published"))):
        raise ValueError("numerical PASS requires actual serial/mixed/smoke completion and safe drain, with no regular performance or capture")
    paths = value.get("evidence_paths") or []
    if not paths:
        raise ValueError("numerical PASS requires fresh original evidence")
    for raw in paths:
        path = Path(str(raw)).resolve(strict=True)
        if not path.is_file() or not path.is_relative_to(Path(context["run_dir"])):
            raise ValueError("numerical evidence must belong to its new normal EvaluationRun")
    return value


def _verify_numerical_reference(cell, prepared, context, *, allow_same_run_resume=False):
    """A4 performance starts only after this same fixed cell's numeric PASS."""
    declared = cell.get("numerical_reference")
    if not isinstance(declared, Mapping) or not re.fullmatch(r"[a-f0-9]{64}", str(declared.get("sha256") or "")):
        raise ValueError("new DeepX A4 acceptance requires its bound numerical PASS reference")
    binding = file_binding(declared)
    payload = json.loads(Path(binding["path"]).read_text())
    matrix = prepared["phase_matrix"]
    phase = Path(matrix["path"]).parent
    path = Path(binding["path"])
    same_run_resume = (allow_same_run_resume is True and context.get("resume") is True
                       and context.get("acceptance_phase") == "numerical"
                       and path == Path(prepared["output"]) / "NUMERICAL_RESULT.json"
                       and json.loads((Path(prepared["output"]) / "PREPARED.json").read_text()) == prepared)
    expected_root = phase / "hardware_runs" / (cell["model_id"] + "__" + cell["setup_id"])
    if (not path.is_relative_to(expected_root)
            or (path.is_relative_to(Path(context["run_dir"])) and context.get("workflow_kind") != "native_refresh" and not same_run_resume)
            or payload.get("schema") != "onnx-splitpoint/hardware-acceptance-native-numerical/v1"
            or payload.get("status") != "numerical_passed" or payload.get("identity") != prepared["identity"]
            or payload.get("phase_matrix") != matrix or payload.get("deepx_runtime_mode") != "A4"
            or payload.get("source_contract_sha256") != prepared["source_contract"].get("contract_sha256")
            or payload.get("device_completion_proven") is not True):
        raise ValueError("numerical PASS is outside this exact current-phase model/artifact/mode binding")
    images = file_binding(cell.get("numerical_images"))
    if payload.get("numerical_images") != images:
        raise ValueError("numerical PASS image selection differs from the saved A4 acceptance cell")
    n = payload.get("image_count")
    if (type(n) is not int or not 1 <= n <= 32 or payload.get("serial_reference_count") != n
            or payload.get("mixed_completed_count") != n or payload.get("smoke_completed_count") != 64
            or any(type(payload.get(key)) is not int for key in ("serial_reference_count", "mixed_completed_count", "smoke_completed_count"))):
        raise ValueError("numerical PASS lacks actual bounded serial/mixed/smoke counts")
    worker = file_binding(payload.get("worker_report"))
    if not Path(worker["path"]).is_relative_to(path.parent):
        raise ValueError("numerical worker report escapes its original result directory")
    from .hardware_acceptance_numerical import validate_numerical_raw
    actual = validate_numerical_raw(json.loads(Path(worker["path"]).read_text()), prepared, expected_images=n)
    for key in ("serial_reference_count", "mixed_completed_count", "smoke_completed_count", "device_completion_proven"):
        if actual.get(key) != payload.get(key):
            raise ValueError("numerical PASS summary differs from its original worker report: " + key)
    return {"reference": binding, "worker_report": worker, "identity": prepared["identity"], "status": "numerical_passed"}


def _retained_numerical_result(cell, prepared, context):
    """Project one unchanged, revalidated same-Run PASS without another worker."""
    if context.get("resume") is not True or context.get("acceptance_phase") != "numerical":
        raise ValueError("retained numerical PASS requires normal same-Run numerical Resume")
    path = Path(prepared["output"]) / "NUMERICAL_RESULT.json"
    if not path.resolve().is_relative_to(Path(context["run_dir"]).resolve()):
        raise ValueError("retained numerical result is outside its selected Run")
    binding = file_binding(path)
    selected = dict(cell, numerical_reference=binding)
    _verify_numerical_reference(selected, prepared, context, allow_same_run_resume=True)
    value = json.loads(path.read_text())
    if value.get("fresh_hardware_executed") is not True:
        raise ValueError("retained numerical original does not record its actual worker start")
    value.update(fresh_hardware_executed=False, same_run_numerical_retained=True,
                 retained_numerical_result=binding, new_numerical_worker_processes=0)
    return value


def _failure_evidence(cell, variant, context, *, invoked, reason):
    """Keep actual ledger starts even if an adapter raises after dispatch."""
    path = Path(context["budget_file"])
    backend = cell["backend"] if variant == "native" else "generic_completed:" + str(cell.get("comparison_backend") or "unbound")
    key = json.dumps([backend, cell["model_id"], cell["case_id"], cell["setup_id"]], separators=(",", ":"))
    try:
        chains = (json.loads(path.read_text()).get("tasks", {}).get(key) or {}).get("chains", [])
    except (OSError, ValueError):
        chains = []
    captures = sum(chain.get("collector_started") is True for chain in chains)
    evidence = [str(Path(chain["run_directory"]) / "energy_summary.json") for chain in chains
                if (Path(chain["run_directory"]) / "energy_summary.json").is_file()]
    return {"status": "failed", "reason": reason, "dispatcher_invoked": invoked,
            "fresh_hardware_executed": True if captures else None if invoked else False,
            "new_capture_count": captures, "valid_repeats": {"performance": None,
                "energy": sum(chain.get("valid") is True for chain in chains)},
            "failed_attempts": {"performance": None,
                "energy": sum(chain.get("collector_started") is True and chain.get("valid") is False for chain in chains)},
            "evidence_paths": evidence, "start_accounting_source": str(path),
            "performance_accounting": "adapter exception; original receipts retained, count unproven" if invoked else "not dispatched"}


def _resource_context(runner, config, context):
    from ..remote.process_lease import ControllerResourceBroker, RemoteProcessLeaseJournal
    from ..quality_service import ResourcePauseGate
    from ..energy.config import energy_setup_from_registry, load_hardware_registry
    from .execution_binding import physical_dut_key
    from .hardware_matrix import normalize_hardware_targets
    targets = {str(target["id"]): target for target in normalize_hardware_targets(runner.profile_payload)}
    registry = load_hardware_registry(context["hardware_registry"])
    resources, source_ids = {}, {}
    for setup in sorted({cell["setup_id"] for cell in config["cells"]}):
        target = targets[setup]
        energy = energy_setup_from_registry(registry, setup, registry_path=context["hardware_registry"])
        source_ids[setup] = str(energy.urecs_address).strip().lower()
        resources[setup] = [physical_dut_key(target), "source:" + source_ids[setup],
                            "controller:cpu", "controller:io", "controller:nic", "controller:capture"]
    environment = {**runner._remote_process_registry.journal_environment(),
                   "ONNX_SPLITPOINT_CONTROLLER_RESOURCES": "required"}
    journal = RemoteProcessLeaseJournal.from_environment(environment, required=True)
    broker = ControllerResourceBroker(journal, ResourcePauseGate(), run_dir=runner.run_dir, setups=resources)
    context.update(targets=targets, environment=environment, resource_broker=broker,
                   resource_tick=broker.tick, source_ids=source_ids)
    return broker


def _initialize_budget(context, config, cell, variant):
    from ..energy.task_budget import EnergyTaskBudget
    path = Path(context["budget_file"])
    path.parent.mkdir(parents=True, exist_ok=True)
    policy = config["regular_budget"]["task_budget"]
    limits = {"max_chains": config["regular_budget"]["repetitions"] * (1 + policy["max_retries"]), "max_retries": policy["max_retries"],
              "max_transport_failures": policy["max_transport_failures"]}
    with os.fdopen(os.open(path, os.O_RDWR | os.O_CREAT, 0o600), "r+", encoding="utf-8") as handle:
        backend = cell["backend"] if variant == "native" else "generic_completed:" + cell["comparison_backend"]
        identity = json.dumps([backend, cell["model_id"], cell["case_id"], cell["setup_id"]], separators=(",", ":"))
        budget = EnergyTaskBudget(handle, limits, runner_cancel(context), campaign_row=identity,
                                  source_id=context["source_ids"][cell["setup_id"]], environment=context["environment"])
        budget.checkpoint_path = path


def runner_cancel(context):
    return context.get("cancel_event")


def run_acceptance(runner, *, config=None, workflow_kind="hardware_acceptance") -> WorkflowRunResult:
    """Run already validated finite requests within the ordinary runner owner."""
    if workflow_kind not in {"hardware_acceptance", "native_refresh"}:
        raise ValueError("unsupported finite Native workflow")
    if config is None:
        config = validate_request(runner.profile_payload, runner.options)
    elif workflow_kind != "native_refresh":
        raise ValueError("acceptance configuration must use its normal validator")
    title = "Native refresh" if workflow_kind == "native_refresh" else "Hardware acceptance"
    job_type = "NativeRefreshJob" if workflow_kind == "native_refresh" else "HardwareAcceptanceJob"
    from .hardware_matrix import default_hardware_setups_file
    registry = runner.options.hardware_setups_file or str(default_hardware_setups_file())
    root = runner.run_dir
    if root.resolve().is_relative_to(Path(config["source_primary_run"])):
        raise ValueError("acceptance output must be outside historical primary evidence")
    context = {"controller_root": str(Path(__file__).resolve().parents[2]),
               "report_root": config["source_report_root"], "source_report_root": config["source_report_root"],
               "source_primary_run": config["source_primary_run"], "primary_run": config["source_primary_run"],
               "acceptance_root": str(root), "run_dir": str(root), "output_root": str(root),
               "python": sys.executable, "profile": runner.profile_payload, "source_profile": runner.profile_payload,
               "options": runner.options, "hardware_registry": registry, "hardware_registry_path": registry,
               "log": runner.log, "progress": runner.progress, "job_event": runner.job_event,
               "cancel_event": runner._cancel_event, "process_registry": runner._process_registry,
               "remote_process_registry": runner._remote_process_registry, "execute_allowed": True,
               "budget_file": str(root / "energy/energy_task_budget.json"),
               "budget_path": str(root / "energy/energy_task_budget.json"),
               "acceptance_mode": config["mode"], "requested_repetitions": config["regular_budget"]["repetitions"],
               "acceptance_phase": config["phase"],
               "workflow_kind": workflow_kind, "resume": bool(runner.options.resume),
               "finite_config": config,
               "finite_request_binding": {"profile_id": getattr(runner, "profile_id", runner.profile_payload.get("name", "")),
                                          "profile_hash": runner.manifest.get("profile_hash"), "schema": config["schema"]},
               "expected_duration_s": 60, "expected_repetitions": config["regular_budget"]["repetitions"], "expected_frames": 1000, "expected_warmup": 100}
    plan_path = write_json(root / "reports" / (workflow_kind + "_plan.json"), config)
    paths = [plan_path]
    rows, errors = [], []
    total = sum(len(cell["runtime_variants"]) for cell in config["cells"])
    runner._progress_total = total + 2
    runner.manifest["workflow_stages"] = [workflow_kind]
    runner.manifest["hardware_acceptance"] = {"schema": SCHEMA, "cell_count": len(config["cells"]),
                                             "runtime_path_count": total, "source_campaign_resumed": False,
                                             "acceptance_phase": config["phase"],
                                             "acceptance_mode": config["mode"], "requested_repetitions": context["expected_repetitions"]}
    if workflow_kind == "native_refresh":
        runner.manifest["native_refresh"] = {**runner.manifest["hardware_acceptance"], "schema": config["schema"]}
    runner.manifest["model_count"] = len({cell["model_id"] for cell in config["cells"]})
    runner._save_manifest("running")
    broker = _resource_context(runner, config, context)
    if config["phase"] == "complete" and (runner.profile_payload.get("native_producers") or {}).get("deepx_runtime_mode") == "A4":
        for selected_cell in config["cells"]:
            for selected_variant in selected_cell["runtime_variants"]:
                _initialize_budget(context, config, selected_cell, selected_variant)
    stop = threading.Event()
    tick_lock = threading.Lock()
    def tick():
        with tick_lock:
            broker.tick()
        if errors:
            raise RuntimeError("controller resource service failed: " + str(errors[0]))
    context["resource_tick"] = tick
    def run_process(argv, *, label="hardware-acceptance", timeout_s=None, cwd=None, env=None):
        from ..native_progress import run_streaming
        environment = {**context["environment"], **dict(env or {})}
        return run_streaming(argv, timeout=timeout_s, cwd=cwd, env=environment, label=label,
                             line_callback=runner.log, cancel_event=runner._cancel_event,
                             process_registry=runner._process_registry, tick_callback=tick,
                             should_cancel=lambda: runner._remote_process_registry.cancelled)
    context["run_process"] = run_process
    def service():
        while not stop.wait(0.1):
            try:
                tick()
            except Exception as exc:
                errors.append(f"{type(exc).__name__}: {exc}")
                runner._cancel_event.set()
                return
    with _ENVIRONMENT_LOCK:
        previous = {key: os.environ.get(key) for key in context["environment"]}
        os.environ.update(context["environment"])
        worker = threading.Thread(target=service, name="hardware-acceptance-resources", daemon=False)
        try:
            worker.start()
            for index, original in enumerate(config["cells"]):
                cell_context = dict(context)
                for variant in original["runtime_variants"]:
                    invoked = False
                    try:
                        cell = _bound_cell(original, variant)
                        selection_error = cell.get("blocked_reason", "")
                    except Exception as exc:
                        cell, selection_error = original, f"{type(exc).__name__}: {exc}"
                    identity = {key: cell[key] for key in ("model_id", "setup_id", "case_id")}
                    identity["runtime_variant"] = variant
                    started = now_iso()
                    job = f"{'refresh' if workflow_kind == 'native_refresh' else 'acceptance'}-{index:02d}-{variant}"
                    runner.job_event(dict(identity, job_id=job, job_type=job_type, stage=workflow_kind,
                                          title=" | ".join(identity.values()), status="running", event="started", run_dir=str(root)))
                    runner.log("[" + workflow_kind + "] selected " + json.dumps(identity, sort_keys=True))
                    try:
                        if runner._cancel_event.is_set():
                            result = {"status": "cancelled", "reason": "normal workflow cancellation", "fresh_hardware_executed": False}
                        elif selection_error:
                            result = {"status": "blocked", "reason": selection_error, "fresh_hardware_executed": False}
                        else:
                            tick()
                            if config["phase"] != "numerical":
                                _initialize_budget(context, config, cell, variant)
                            prepare, execute = _adapter(variant)
                            prepared = prepare(cell, cell_context)
                            if variant == "native":
                                cell_context["prepared_native"] = prepared
                                cell_context["acceptance_cell"] = cell
                            prepared_path = write_json(root / "reports" / "acceptance_prepared" / (job + ".json"), _serialize(prepared))
                            paths.append(prepared_path)
                            runner.log("[hardware-acceptance] normal prepare complete " + str(prepared_path))
                            if runner._cancel_event.is_set():
                                result = {"status": "cancelled", "reason": "cancelled after prepare", "fresh_hardware_executed": False}
                            else:
                                if config["phase"] == "numerical":
                                    from .hardware_acceptance_numerical import run_numerical_native
                                    invoked = True
                                    prior_numerical = Path(prepared["output"]) / "NUMERICAL_RESULT.json"
                                    numerical = (_retained_numerical_result(cell, prepared, cell_context)
                                                 if cell_context.get("resume") is True and prior_numerical.exists()
                                                 else run_numerical_native(prepared, cell_context, file_binding(cell.get("numerical_images"))))
                                    result = _accepted_numerical_result(numerical, cell_context)
                                elif workflow_kind == "native_refresh":
                                    from .native_refresh import execute_point, accepted_result
                                    invoked = True
                                    result = accepted_result(execute_point(cell, prepared, cell_context, execute), cell_context)
                                else:
                                    if prepared.get("deepx_runtime_mode") == "A4":
                                        proof = _verify_numerical_reference(cell, prepared, cell_context)
                                        write_json(Path(prepared["output"]) / "NUMERICAL_ADMISSION.json", proof)
                                    invoked = True
                                    result = _accepted_result(execute(prepared, cell_context), cell_context)
                            if variant == "native":
                                cell_context.update(prepared_native=prepared, native_result=result)
                    except Exception as exc:
                        result = _failure_evidence(cell, variant, context, invoked=invoked,
                                                   reason=f"{type(exc).__name__}: {exc}")
                    if not result.get("reason") and result.get("error"):
                        result["reason"] = str(result["error"])
                    row = dict(identity, **_serialize(result), started_at=started, finished_at=now_iso())
                    rows.append(row)
                    row_path = write_json(root / "reports/acceptance_results" / (job + ".json"), row)
                    paths.append(row_path)
                    passed = row["status"] == ("numerical_passed" if config["phase"] == "numerical" else "ok")
                    status = "ok" if passed else "cancelled" if row["status"] == "cancelled" else "failed"
                    stage = StageResult(workflow_kind, cell["model_id"], status, started, row["finished_at"],
                                        job_id=job, job_type=job_type, execution_mode=variant,
                                        state="completed" if status == "ok" else status, artifacts=[str(row_path)], details=row)
                    runner.stage_results.append(stage.to_dict())
                    runner.manifest.setdefault("models", {}).setdefault(cell["model_id"], {"model_id": cell["model_id"], "stages": {}})["stages"][job] = stage.to_dict()
                    runner.job_event(dict(identity, job_id=job, job_type=job_type, stage=workflow_kind,
                                          title=" | ".join(identity.values()), status=status, event="finished", message=row.get("reason", ""), run_dir=str(root)))
                    runner.progress(len(rows), total + 2, f"{cell['model_id']}/{cell['setup_id']}/{variant}: {row['status']}")
                    write_json(root / "reports" / (workflow_kind + "_results.json"), {"schema": config["schema"], "rows": rows, "complete": False})
        finally:
            stop.set()
            if worker.ident is not None:
                worker.join(timeout=5)
            try:
                with tick_lock:
                    broker.close()
            finally:
                for key, value in previous.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value
    passed_status = "numerical_passed" if config["phase"] == "numerical" else "ok"
    status = "cancelled" if runner._cancel_event.is_set() else "ok" if all(row["status"] == passed_status for row in rows) else "partial"
    completion = workflow_completion_projection(status)
    captures = sum(int(row.get("new_capture_count") or 0) for row in rows)
    final = {"schema": config["schema"], "workflow_kind": workflow_kind, "run_id": runner.run_id, "rows": rows, "status": status,
             "acceptance_mode": config["mode"], "requested_repetitions": context["expected_repetitions"],
             "acceptance_phase": config["phase"], "hardware_acceptance_complete": status == "ok" and config["phase"] == "complete",
             "complete": status == "ok", "matrix_cells": len(config["cells"]), "runtime_paths": total,
             "fresh_pass_paths": sum(row["status"] == "ok" and not row.get("reused_cohort")
                                     and (row.get("new_performance_processes", 1) or row.get("new_capture_count", 0)) > 0 for row in rows),
             "retained_pass_paths": sum(row["status"] == "ok" and not (row.get("new_performance_processes") or row.get("new_capture_count")) for row in rows), "new_capture_count": captures,
             "numerical_pass_paths": sum(row["status"] == "numerical_passed" for row in rows),
             "source_campaign_resumed": False, "new_dataset_quality_started": False, "resource_errors": errors}
    result_path = write_json(root / "reports" / (workflow_kind + "_results.json"), final)
    dashboard_path = write_json(root / "reports/result_dashboard.json", final)
    dashboard_md = root / "reports/result_dashboard.md"
    dashboard_md.write_text(title + ": " + status + "\n\n"
        "Model | Setup | Split | Runtime | Result | New captures | Valid repeats | Limitation\n"
        "--- | --- | --- | --- | --- | --- | --- | ---\n"
        + "\n".join(" | ".join(str(value).replace("|", "/").replace("\n", " ") for value in (
            row["model_id"], row["setup_id"], row["case_id"], row["runtime_variant"], row["status"],
            row.get("new_capture_count", "unproven"), row.get("valid_repeats", "unavailable"), row.get("reason", ""))) for row in rows)
        + "\n\nOnly new physical attempts are counted. Existing compatible Full and dataset-quality references retain their original exclusions.\n"
        + "The source campaign remains paused.\n", encoding="utf-8")
    status_path = write_json(root / "reports/run_status_summary.json", {**final, "completion": completion})
    paths.extend([result_path, dashboard_path, dashboard_md, status_path])
    runner.report_paths.extend(paths)
    runner.outputs.update({workflow_kind + "_results_json": str(result_path), "result_dashboard_json": str(dashboard_path)})
    runner._register_artifacts(paths, kind=workflow_kind, producer_stage=workflow_kind)
    runner._measurement_phase_status = status
    runner._control_state = "cancelled" if status == "cancelled" else "finished"
    runner._write_workflow_control()
    runner._save_manifest(status)
    return WorkflowRunResult(ok=status == "ok", status=status, run_id=runner.run_id, run_dir=str(root),
                             manifest_path=str(runner.manifest_path), artifact_index_path=str(runner.artifact_index_path),
                             report_paths=[str(path) for path in paths], stage_results=list(runner.stage_results),
                             warnings=list(runner.warnings), completed=status != "cancelled",
                             technical_status="ok" if status == "ok" else "incomplete", quality_decision="reused_reference_only",
                             scientific_status="existing_exclusions_preserved", completion=completion,
                             evidence_status={"hardware_acceptance": final})
