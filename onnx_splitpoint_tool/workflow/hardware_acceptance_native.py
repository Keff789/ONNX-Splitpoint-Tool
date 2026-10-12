"""One selected Native split through the ordinary runners and energy consumer.

Saved performance plans are provenance and artifact-resolution inputs.  Only
their normal runner argv are reused; private campaign executors are never
imported.  All outputs belong to the new acceptance EvaluationRun.
"""
from __future__ import annotations

from contextlib import ExitStack
import copy
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import statistics
import time
from typing import Any, Mapping

from ..native_command_contract import verify_native_command_contract
from ..native_energy_reporting import build_native_energy_pairs, collect_native_energy
from ..remote.ssh_transport import HostConfig, SSHTransport
from ..runners.request_latency import validate_latency
from .checkpoints import atomic_write_json
from .execution_binding import physical_dut_key
from .run_control import EvaluationRunLock


_RUNNERS = {
    "hailo8_to_trt": "native_hailo_trt_fifo_from_benchmarkset.py",
    "hailo10h_to_trt": "native_hailo10_trt_e2e_from_benchmarkset.py",
    "deepx_to_trt": "native_deepx_trt_e2e_from_benchmarkset.py",
}
_RAW_NAMES = {
    "hailo8_to_trt": "native_fifo_results.json",
    "hailo10h_to_trt": "hailo10_native_fifo_e2e_results.json",
    "deepx_to_trt": "deepx_native_fifo_e2e_results.json",
}
_STAGE_NAMES = {
    "hailo8_to_trt": "native_fifo_eval_runner_acceptance.json",
    "hailo10h_to_trt": "native_hailo10h_producer_e2e_eval_acceptance.json",
    "deepx_to_trt": "native_deepx_producer_e2e_eval_acceptance.json",
}
_OUTPUT_FLAGS = {"--out-dir", "--result-json", "--config-json", "--output-dir", "--boundary-dir", "--work-dir"}


def _read(path: str | Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Native acceptance evidence must be a JSON object")
    return value


def _sha(path: str | Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, value)


def _bound_path(value: Any) -> Path:
    if isinstance(value, Mapping):
        path = Path(str(value.get("path") or "")).expanduser().resolve(strict=True)
        if value.get("sha256") and _sha(path) != value["sha256"]:
            raise ValueError("Native acceptance source descriptor changed")
        return path
    return Path(str(value)).expanduser().resolve(strict=True)


def _identity(cell: Mapping) -> dict:
    backend = str(cell.get("measurement_direction") or cell.get("backend") or cell.get("profile_split_run_id") or "")
    backend = backend.replace("hailo10_to_tensorrt", "hailo10h_to_trt").replace("hailo10_to_trt", "hailo10h_to_trt")
    result = {"model": str(cell["model_id"]), "case": str(cell["case_id"]),
              "setup_id": str(cell["setup_id"]), "backend": backend}
    if backend not in _RUNNERS or not re.fullmatch(r"b[0-9]{3,}", result["case"]):
        raise ValueError("Native acceptance supports one declared split and known runner")
    if any(not re.fullmatch(r"[A-Za-z0-9_.-]+", text) for text in result.values()):
        raise ValueError("unsafe Native acceptance identity")
    return result


def _argument(argv: list[str], flag: str) -> str:
    if argv.count(flag) != 1 or argv.index(flag) + 1 >= len(argv):
        raise ValueError("Native acceptance requires one argument: " + flag)
    return argv[argv.index(flag) + 1]


def _rewrite_run(item: Mapping, identity: Mapping, remote_root: str) -> dict:
    """Change only writable output destinations of one ordinary runner argv."""
    argv = [str(value) for value in item["argv"]]
    scripts = [value for value in argv if value.endswith(".py")]
    if len(scripts) != 1 or PurePosixPath(scripts[0]).name != _RUNNERS[identity["backend"]]:
        raise ValueError("Native acceptance cannot execute a private or unknown runner")
    if (_argument(argv, "--case") != identity["case"]
            or _argument(argv, "--model-id") != identity["model"]
            or _argument(argv, "--setup-id") != identity["setup_id"]
            or int(_argument(argv, "--frames")) != 1000
            or int(_argument(argv, "--warmup")) != 100
            or int(_argument(argv, "--repetitions")) != 1):
        raise ValueError("saved Native operating point differs from selected cell/profile")
    if "--duration-s" in argv and float(_argument(argv, "--duration-s")) != 0:
        raise ValueError("Native performance is a fixed completed-task run")
    if any(flag in argv for flag in ("--build", "--force-build", "--force-rebuild", "--energy-workload-only")):
        raise ValueError("build or energy mode in Native performance source")
    old_root = str(item["remote_output_root"]).rstrip("/")
    if not old_root.startswith("/") or ".." in PurePosixPath(old_root).parts:
        raise ValueError("saved Native output root is not canonical")
    for index, flag in enumerate(argv[:-1]):
        if flag in _OUTPUT_FLAGS:
            old = argv[index + 1]
            if old != old_root and not old.startswith(old_root + "/"):
                raise ValueError("saved Native writable path escapes its output root")
            argv[index + 1] = remote_root + old[len(old_root):]
    # The archived vendor dependency environment remains ordinary runner
    # configuration; a controller source overlay is never propagated.
    environment = {str(k): str(v) for k, v in dict(item.get("env") or {}).items()
                   if k not in {"PYTHONPATH", "PYTHONHOME"}}
    environment.update(PYTHONDONTWRITEBYTECODE="1", ONNX_SPLITPOINT_NATIVE_TRT_BUILD="0")
    if identity["backend"] == "hailo8_to_trt" and "--no-build" not in argv:
        raise ValueError("Hailo Native acceptance requires explicit no-build")
    result = dict(item, argv=argv, env=environment, remote_output_root=remote_root,
                  result_json=remote_root + "/" + _RAW_NAMES[identity["backend"]])
    result.pop("remote_telemetry_root", None)
    return result


def _verify_baseline_references(cell: Mapping, identity: Mapping, source_contract: Mapping, context: Mapping) -> list[dict]:
    """Re-prove supplied Full references without reassigning their claims."""
    from ..resources.remote_scripts.native_producer_energy_plan import _energy_preprocess_identity, _verify_full_command_contract
    references = cell.get("full_baseline_references", cell.get("baselines", []))
    if not isinstance(references, list) or len(references) != 2:
        raise ValueError("Native acceptance requires two declared setup-local Full references")
    vendor = "native_full_" + str(cell.get("producer_backend") or identity["backend"].removesuffix("_to_trt"))
    if {reference.get("backend") for reference in references} != {vendor, "native_full_tensorrt"}:
        raise ValueError("Native acceptance Full backend scope differs")
    source_input = _energy_preprocess_identity(source_contract)
    result = []
    for reference in references:
        if reference.get("model_id") != identity["model"] or reference.get("setup_id") != identity["setup_id"]:
            raise ValueError("Native Full reference model/setup differs")
        provenance = reference.get("quality_provenance") or {}
        path = _bound_path({"path": provenance["normal_report"], "sha256": provenance["normal_report_sha256"]})
        row = _read(path)["rows"][provenance["normal_row_index"]]
        digest = hashlib.sha256(json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
        if digest != provenance["normal_row_sha256"]:
            raise ValueError("Native Full reference row changed")
        expected = {**identity, "backend": reference["backend"], "case": "full", "comparison_backend": vendor.removeprefix("native_full_")}
        contract, reason = _verify_full_command_contract(row.get("full_command_contract"), expected_identity=expected)
        # Historical Full performance can remain useful when its old command
        # cannot be admitted as a new energy replay.  That is a reference
        # limitation, never permission to fabricate a pair or block the Split.
        full_input = _energy_preprocess_identity(contract or row.get("full_command_contract") or {})
        for field in ("prepared_feed_task", "prepared_feed_preprocess_mode", "prepared_feed_letterbox_pad_value", "prepared_feed_source_image_sha256"):
            if not source_input[field] or full_input[field] != source_input[field]:
                raise ValueError("Native Full reference input/preprocessing mismatch: " + field)
        quality = _bound_path({"path": provenance["quality_summary"], "sha256": provenance["quality_summary_sha256"]})
        result.append({"reference": copy.deepcopy(reference), "normal_row": row, "normal_report": str(path),
                       "normal_report_sha256": _sha(path), "normal_row_sha256": digest,
                       "normal_quality_summary": str(quality), "normal_quality_summary_sha256": _sha(quality),
                       "input_compatibility": source_input, "full_command_contract_sha256": (contract or {}).get("contract_sha256"),
                       "normal_full_replay_status": reason, "normal_full_replay_verified": contract is not None,
                       "quality_and_claim_flags_preserved": True,
                       "historical_normal_energy_eligible": reference.get("normal_energy_eligible") is True,
                       "current_energy_pairing_status": "requires_normal_energy_consumer_and_pair_gate"})
    return result


def _verify_current_target(target: Mapping, identity: Mapping, context: Mapping) -> dict:
    from ..energy.config import load_hardware_registry, energy_setup_from_registry
    from .hardware_matrix import normalize_hardware_targets
    registry_path = str(context.get("hardware_registry_path") or context.get("hardware_registry")
                        or Path.home() / ".onnx_splitpoint_tool/hardware_setups.yaml")
    registry = load_hardware_registry(registry_path)
    current = (context.get("targets") or {}).get(identity["setup_id"])
    if current is None:
        profile = context.get("profile") or {"hardware": {"selected_setups": [identity["setup_id"]], "setups_file": registry_path}}
        current = next((row for row in normalize_hardware_targets(profile) if row["id"] == identity["setup_id"]), None)
    if current is None or physical_dut_key(current) != physical_dut_key(target):
        raise ValueError("current and archived Native selection identify different physical DUTs")
    for key in ("host", "user", "port"):
        default = 22 if key == "port" else ""
        if (current.get("remote") or {}).get(key, default) != (target.get("remote") or {}).get(key, default):
            raise ValueError("current Native target SSH binding differs: " + key)
    if int((current.get("remote") or {}).get("port", 22)) != 22:
        raise ValueError("bounded Native energy adapter supports the normal port-22 binding")
    energy = energy_setup_from_registry(registry, identity["setup_id"], registry_path=registry_path)
    expected_source = str((target.get("energy") or {}).get("urecs_address") or "")
    if not expected_source or str(energy.urecs_address) != expected_source:
        raise ValueError("Native energy source differs from archived DUT binding")
    broker_source = (context.get("source_ids") or {}).get(identity["setup_id"])
    if broker_source and broker_source != expected_source:
        raise ValueError("Native controller broker is bound to a different energy source")
    return {"registry": registry_path, "registry_sha256": _sha(registry_path), "physical_dut_key": physical_dut_key(current),
            "ssh": {key: current["remote"].get(key, 22 if key == "port" else "") for key in ("host", "user", "port")},
            "urecs_address": expected_source}


def _source_references(cell: Mapping, plan: Mapping, source: Path) -> tuple[Path, Path]:
    """Resolve the ordinary saved-profile references and their sealed leaves."""
    quality = cell.get("quality_reference") or {}
    provenance = quality.get("native_provenance") or cell.get("native_quality_provenance") or {}
    validation = _bound_path({"path": cell.get("native_validation_summary") or quality.get("native_validation_summary"),
                              "sha256": provenance.get("quality_summary_sha256")})
    raw = cell.get("native_raw_report")
    if raw is None:
        summary = _bound_path({"path": str(Path(plan["local_output"]) / "summary.json"),
                               "sha256": provenance.get("performance_summary_sha256")})
        payload = _read(summary)
        if (payload.get("plan_sha256") != _sha(source)
                or Path(str(payload.get("plan"))).resolve() != source):
            raise ValueError("saved Native performance summary is not bound to its source plan")
        first = [row for row in payload.get("rows", []) if row.get("repeat") == 1]
        if len(first) != 1:
            raise ValueError("saved Native performance summary has no unique first raw leaf")
        raw = first[0]["raw_report"]
    expected = list(provenance.get("raw_report_sha256s") or [])
    raw_path = _bound_path(raw)
    if expected and _sha(raw_path) not in expected:
        raise ValueError("saved Native raw leaf differs from sealed quality provenance")
    return raw_path, validation


def prepare_native(cell: Mapping, context: Mapping) -> dict:
    """Resolve one saved selection offline; no SSH, build, capture or quality run."""
    identity = _identity(cell)
    if (int(context.get("frames", 1000)) != 1000 or int(context.get("warmup", 100)) != 100
            or int(context.get("repetitions", 3)) != 3
            or float(context.get("expected_duration_s", 60)) != 60):
        raise ValueError("Native acceptance must retain regular 100/1000/3 and 60-second profile")
    declared_source = cell.get("native_source_plan") or cell["native_performance_plan"]
    if not isinstance(declared_source, Mapping):
        declared_source = {"path": declared_source, "sha256": cell.get("native_source_plan_sha256")}
    source = _bound_path(declared_source)
    plan = _read(source)
    if plan.get("build") not in {False, "off", "disabled"}:
        raise ValueError("Native source plan does not prove builds disabled")
    if plan.get("setup_id") != identity["setup_id"] or str(plan.get("case")) != identity["case"]:
        raise ValueError("Native source plan identity differs")
    raw_path, validation = _source_references(cell, plan, source)
    raw = _read(raw_path)
    contract, reason = verify_native_command_contract(raw.get("native_command_contract"), expected_identity=identity)
    if contract is None:
        raise ValueError("Native successful source contract rejected: " + reason)
    root = Path(str(context.get("acceptance_root") or context["run_dir"])).resolve()
    if root == source.parent or root in source.parents:
        raise ValueError("Native acceptance output would overwrite historical evidence")
    component = "__".join(identity[key] for key in ("model", "setup_id", "case"))
    output = root / "native" / component
    remote_tool = str(context.get("remote_tool_dir") or "/home/nx/ONNX-Splitpoint-Tool")
    remote_base = "/home/nx/splitpoint_runs/hardware_acceptance/" + root.name + "/native/" + component
    runs = list(plan.get("runs") or [])
    if len(runs) != 3 or {item.get("repeat") for item in runs} != {1, 2, 3}:
        raise ValueError("Native acceptance needs exactly three source process repetitions")
    prepared_runs = [_rewrite_run(item, identity, remote_base + f"/rep{int(item['repeat']):02d}")
                     for item in sorted(runs, key=lambda item: item["repeat"])]
    declared_primary = cell.get("native_original_run") or context.get("source_primary_run") or cell["original_run"]
    primary = _bound_path(declared_primary)
    manifest = _read(primary / "run_manifest.json")
    manifest_binding = cell.get("native_original_run_manifest")
    if manifest_binding:
        bound_manifest = _bound_path(manifest_binding)
        if bound_manifest != primary / "run_manifest.json":
            raise ValueError("Native original run directory and manifest descriptor differ")
    if isinstance(declared_primary, Mapping) and declared_primary.get("manifest_sha256"):
        if _sha(primary / "run_manifest.json") != declared_primary["manifest_sha256"]:
            raise ValueError("Native original run manifest binding changed")
    if manifest.get("run_id") != contract.get("eval_run_id"):
        raise ValueError("Native source authority must be the exact contract EvaluationRun")
    targets = [target for target in manifest.get("hardware_targets", []) if target.get("id") == identity["setup_id"]]
    if len(targets) != 1:
        raise ValueError("Native selection has no unique saved hardware target")
    target = copy.deepcopy(targets[0])
    remote = target.get("remote") or {}
    if not remote.get("host") or not remote.get("user"):
        raise ValueError("Native selection has no normal SSH binding")
    target_proof = _verify_current_target(target, identity, context)
    baseline_proofs = _verify_baseline_references(cell, identity, contract, context)
    prepared = {"schema": "onnx-splitpoint/hardware-acceptance-native-plan/v1", "identity": identity,
                "source_plan": str(source), "source_plan_sha256": _sha(source),
                "source_raw_report": str(raw_path), "source_raw_report_sha256": _sha(raw_path),
                "source_contract": contract, "source_primary_run": str(primary),
                "source_validation": str(validation), "source_validation_sha256": _sha(validation),
                "output": str(output), "remote_tool_dir": remote_tool, "runs": prepared_runs,
                "target": target, "full_baseline_references": cell.get("full_baseline_references", cell.get("baselines", [])),
                "target_binding": target_proof,
                "full_baseline_proofs": baseline_proofs,
                "performance_recipe": {"warmup_tasks_per_process": 100, "completed_tasks_per_process": 1000, "processes": 3},
                "energy_recipe": {"warmup": 0, "frames_override": 0, "duration_s": 60, "repeats": 3,
                                  "task_count": "duration-controlled measured completed work", "physical_scope": "FS",
                                  "window": "whole_command", "source": "normal_native_energy_parent.command_for_and_split_energy_runtime_argv"},
                "new_performance_processes": 3, "new_selected_energy_repeats": 3,
                "full_recaptures": 0, "quality_dataset_reexecutions": 0, "hardware_executed": False}
    _write(output / "PREPARED.json", prepared)
    return prepared


def _transport(prepared: Mapping, context: Mapping) -> SSHTransport:
    registry = context.get("remote_process_registry")
    scope = getattr(registry, "configured_scope", None)
    if registry is None or scope is None:
        raise ValueError("Native acceptance requires the normal journaled remote ownership scope")
    target = prepared["target"]
    transport = SSHTransport(HostConfig.from_dict({**target["remote"], "id": target["id"], "label": target.get("label", target["id"])}),
                             log=context.get("log"), cancel_event=context.get("cancel_event"),
                             remote_lease_scope=scope, remote_lease_registry=registry)
    transport.diagnostics_dir = Path(prepared["output"]) / "diagnostics"
    return transport


def _run_local(argv: list[str], context: Mapping, *, label: str, timeout: int) -> Any:
    runner = context.get("run_process")
    if not callable(runner):
        raise ValueError("Native acceptance requires the normal controller process runner")
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    environment.update(dict(context.get("environment") or {}))
    if environment.get("PYTHONPATH") or environment.get("PYTHONHOME"):
        raise ValueError("Native controller source redirection is forbidden")
    environment.update(PYTHONDONTWRITEBYTECODE="1", ONNX_SPLITPOINT_NATIVE_TRT_BUILD="0")
    return runner(argv, label=label, timeout_s=timeout, cwd=Path(context["controller_root"]), env=environment)


def _validate_raw(raw: Mapping, identity: Mapping, source_contract: Mapping) -> dict:
    contract, reason = verify_native_command_contract(raw.get("native_command_contract"), expected_identity=identity)
    if contract is None:
        raise ValueError("fresh Native command contract rejected: " + reason)
    options = contract.get("runtime_options") or {}
    latency = validate_latency(raw.get("request_latency"))
    counts = [raw.get(key) for key in ("completed_work_units", "completed_frames", "postprocess_completed_frames")
              if raw.get(key) is not None]
    if (raw.get("ok") is not True or not counts or any(value != 1000 for value in counts)
            or int(raw.get("warmup_count", raw.get("warmup", options.get("warmup", -1)))) != 100
            or latency.get("status") != "complete" or latency.get("count") != 1000
            or latency.get("task_complete") is not True
            or raw.get("postprocess_completion_verified") is not True):
        raise ValueError("fresh Native task counts, latency or completion failed")
    for role in ("hef", "dxnn", "engine"):
        old = (source_contract.get("artifacts") or {}).get(role)
        new = (contract.get("artifacts") or {}).get(role)
        if old and (not new or new.get("sha256") != old.get("sha256")):
            raise ValueError("fresh Native artifact differs: " + role)
    elapsed = float(raw.get("makespan_ms") or 0)
    fps = float(raw.get("fps_makespan") or 0)
    if not math.isfinite(fps) or fps <= 0 or elapsed <= 0 or not math.isclose(fps, 1000000 / elapsed, rel_tol=1e-8):
        raise ValueError("fresh Native measured rate is inconsistent")
    return {"completed_tasks": 1000, "warmup": 100, "fps": fps,
            "latency": latency, "contract_sha256": contract["contract_sha256"], "contract": contract}


def _collect_repeat(transport: SSHTransport, item: Mapping, directory: Path) -> Path:
    destination = directory / "download"
    destination.mkdir(parents=True, exist_ok=False)
    rc, text = transport.scp_download(item["remote_output_root"], str(destination), recursive=True, timeout=600)
    _write(directory / "download.json", {"returncode": rc, "output": text[-4000:]})
    if rc:
        raise RuntimeError("Native result download failed")
    collected = destination / PurePosixPath(item["remote_output_root"]).name
    report = collected / PurePosixPath(item["result_json"]).name
    if not report.is_file():
        raise RuntimeError("fresh Native result publication missing")
    for path in collected.rglob("*"):
        if path.is_symlink():
            raise ValueError("Native result download contains an unsafe symlink")
    return report


def _normal_performance_import(prepared: Mapping, reports: list[Path], context: Mapping) -> Path:
    output = Path(prepared["output"])
    identity = prepared["identity"]
    roots = []
    for index, report in enumerate(reports, 1):
        root = output / f"import_rep{index:02d}"
        raw = _read(report)
        # Normal dispatch metadata points to the original, unmodified leaf.
        row = {"model": identity["model"], "case": identity["case"], "precision": raw["native_command_contract"]["precision"],
               "setup_id": identity["setup_id"], "report": str(report), "ok": True,
               "result_ok": True, "status": "ok", "returncode": 0, "timed_out": False,
               "child_result_fresh": True, "native_fifo_result_sha256": _sha(report),
               "native_fifo_result_size_bytes": report.stat().st_size}
        _write(root / "analysis_tables" / _STAGE_NAMES[identity["backend"]], {"rows": [row]})
        roots.append(root)
    primary = Path(prepared["source_primary_run"])
    # --eval-root already provides the immutable original authority.  Copying
    # its campaign stage into the new report directory would import unrelated
    # capability terminals and change the selected single-case scope.
    reports_dir = output / "reports"
    command = [str(context["python"]), "-I", "-B", "-m", "onnx_splitpoint_tool.resources.remote_scripts.native_producer_final_report",
               "--out-dir", str(reports_dir), "--eval-root", str(primary), "--quality-summary", prepared["source_validation"]]
    for root in roots:
        command += ["--root", str(root)]
    result = _run_local(command, context, label="acceptance-native-import", timeout=600)
    _write(output / "performance_import.json", {"command": command, "returncode": result.returncode,
                                                 "stdout": result.stdout, "stderr": getattr(result, "stderr", "")})
    if result.returncode:
        raise RuntimeError("normal Native performance import failed")
    summary = reports_dir / "native_producer_combined_summary.json"
    payload = _read(summary)
    rows = payload.get("rows") or []
    if len(rows) != 1 or any(rows[0].get(key) != value for key, value in identity.items()):
        raise ValueError("normal Native importer changed selected scope")
    if rows[0].get("repetition_count_valid") != 3:
        raise ValueError("normal Native importer did not admit all three fresh repetitions")
    return summary


def _budget_counts(prepared: Mapping, context: Mapping) -> dict:
    budget = Path(str(context["budget_file"]))
    identity = prepared["identity"]
    key = json.dumps([identity[name] for name in ("backend", "model", "case", "setup_id")], separators=(",", ":"))
    task = (_read(budget).get("tasks") or {}).get(key, {}) if budget.is_file() else {}
    chains = task.get("chains") or []
    valid = {chain.get("logical_repeat") for chain in chains if chain.get("valid") is True and chain.get("finished") is True}
    return {"collector_starts": sum(chain.get("collector_started") is True for chain in chains),
            "workload_starts": sum(chain.get("workload_started") is True for chain in chains),
            "valid_repeats": len(valid),
            "failed_attempts": sum(chain.get("collector_started") is True and chain.get("finished") is True
                                   and chain.get("valid") is not True for chain in chains),
            "reserved_chains": len(chains), "budget_file": str(budget)}


def validate_energy_view(view: Path, source: Path) -> dict:
    """The normal collector's completed-point callback, with no acquisition."""
    view, source = view.resolve(), source.resolve(strict=True)
    expected = view / "reports/native_energy_measurements"
    if source.parent != expected or source.name not in {"native_producer_energy_results.json", "native_producer_energy_results.partial.json"}:
        raise ValueError("Native completed-point source escapes its acceptance view")
    rows = collect_native_energy(view, include_partial=True)
    if (len(rows) != 1 or rows[0].get("ok") is not True or rows[0].get("energy_repeat_valid_n") != 3
            or rows[0].get("energy_calibration_verified") is not True
            or rows[0].get("environment_observation_status") != "pass"):
        raise ValueError("normal Native completed-point consumer rejected energy/calibration/environment")
    return {"status": "normal_completed_point_imported", "source": str(source), "rows": rows}


def _energy_capture(prepared: Mapping, summary: Path, context: Mapping) -> dict:
    output = Path(prepared["output"])
    identity = prepared["identity"]
    energy_view = output / "energy_view"
    measurement = energy_view / "reports/native_energy_measurements"
    validation_out = energy_view / "reports/native_validation/native_producer_validation_summary.json"
    validation_out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(prepared["source_validation"], validation_out)
    remote = prepared["target"]["remote"]
    producer = {"hailo8_to_trt": "hailo8", "hailo10h_to_trt": "hailo10", "deepx_to_trt": "deepx"}[identity["backend"]]
    native = dict(context.get("profile", {}).get("native_producers") or {})
    policy = dict((native.get("energy") or {}).get("task_budget") or {})
    if policy.get("max_retries", 2) != 2 or policy.get("max_transport_failures", 20) != 20:
        raise ValueError("Native acceptance retry policy differs from saved profile")
    command = [str(context["python"]), "-I", "-B", "-u", "-m",
               "onnx_splitpoint_tool.resources.remote_scripts.run_native_producer_energy_from_summary",
               "--summary", str(summary), "--validation-summary", prepared["source_validation"],
               "--out-dir", str(measurement), "--hardware-setups-file", str(context.get("hardware_registry_path") or context["hardware_registry"]),
               "--remote-tool-dir", prepared["remote_tool_dir"], "--remote-root", "/home/nx/native_fifo_evalsets",
               "--duration-s", "60", "--frames", "0", "--warmup", "0", "--runs", "3",
               "--timeout", "900", "--physical-scope", "FS", "--window-label", "command",
               "--require-runtime-work-units", "--require-command-window-alignment", "--allow-unpaired",
               "--campaign-budget-file", str(context["budget_file"]), "--campaign-max-retries", "2",
               "--campaign-max-transport-failures", "20", "--" + producer + "-ssh", remote["user"] + "@" + remote["host"]]
    validator = [str(context["python"]), "-I", "-B", "-m", "onnx_splitpoint_tool.workflow.hardware_acceptance_native",
                 "--validate-energy-view", str(energy_view)]
    command += ["--require-environment-observation", "--environment-run-id", Path(context["run_dir"]).name,
                "--completed-row-validator", shlex.join(validator)]
    # Exact native Split source environments are retained; there is no special
    # source import overlay in this normal collector command.
    remote_environment = dict(prepared["runs"][0]["env"])
    environment_prefix = " ".join(key + "=" + shlex.quote(value) for key, value in remote_environment.items())
    if environment_prefix:
        command += ["--" + producer + "-env", "export " + environment_prefix]
    result = _run_local(command, context, label="acceptance-native-energy", timeout=10800)
    _write(output / "energy_dispatch.json", {"command": command, "returncode": result.returncode,
                                            "stdout": result.stdout, "stderr": getattr(result, "stderr", "")})
    rows = collect_native_energy(energy_view, include_partial=True)
    selected = [row for row in rows if all(row.get(key) == value for key, value in identity.items())]
    _write(output / "NORMAL_ENERGY_IMPORT.json", {"rows": selected, "source": str(measurement), "returncode": result.returncode})
    counts = _budget_counts(prepared, context)
    if (len(selected) != 1 or selected[0].get("ok") is not True or selected[0].get("energy_repeat_valid_n") != 3
            or selected[0].get("energy_calibration_verified") is not True
            or selected[0].get("environment_observation_status") != "pass" or counts["valid_repeats"] != 3):
        raise RuntimeError("normal Native energy consumer did not admit three fresh valid repeats")
    return {"row": selected[0], "new_capture_count": counts["collector_starts"], "valid_repeats": counts["valid_repeats"],
            "failed_attempts": counts["failed_attempts"], "start_accounting": counts,
            "energy_result": str(measurement / "native_producer_energy_results.json"),
            "import": str(output / "NORMAL_ENERGY_IMPORT.json")}


def _reference_energy_pairs(prepared: Mapping, energy_row: Mapping, context: Mapping) -> dict:
    """Read existing Full energy with the ordinary consumer and pair checker."""
    references, missing = [], []
    source_root = Path(str(context.get("source_report_root") or context["report_root"])).resolve()
    cache_root = Path(context["run_dir"]) / "reference_energy"
    for proof in prepared["full_baseline_proofs"]:
        reference = proof["reference"]
        provenance = reference.get("energy_provenance") or {}
        raw = provenance.get("normal_source_report")
        if not raw or not proof["historical_normal_energy_eligible"]:
            missing.append({"backend": reference["backend"], "status": "compatible_full_performance_reused_energy_comparison_unavailable"})
            continue
        source = Path(raw)
        if not source.is_absolute():
            source = source_root / source
        source = _bound_path({"path": str(source), "sha256": provenance["normal_source_report_sha256"]})
        cache = cache_root / (_sha(source) + "_" + proof["normal_quality_summary_sha256"])
        import_path = cache / "NORMAL_REFERENCE_ROWS.json"
        if not import_path.exists():
            destination = cache / "reports/native_energy_measurements/native_producer_energy_results.json"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            validation = cache / "reports/native_validation/native_producer_validation_summary.json"
            validation.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(proof["normal_quality_summary"], validation)
            _write(import_path, {"source": str(source), "source_sha256": _sha(source), "rows": collect_native_energy(cache)})
        selected = [row for row in _read(import_path)["rows"] if row.get("backend") == reference["backend"]
                    and row.get("model") == prepared["identity"]["model"]
                    and row.get("setup_id") == prepared["identity"]["setup_id"] and row.get("case") == "full"]
        if len(selected) != 1:
            raise ValueError("normal Full energy reference importer did not resolve a unique row")
        references.append(selected[0])
    pairs = build_native_energy_pairs([dict(energy_row), *references])
    result = {"normal_consumer": "native_energy_reporting.collect_native_energy_and_build_native_energy_pairs",
              "full_energy_reference_rows": references, "pairs": pairs, "missing_full_energy_comparisons": missing,
              "fresh_full_captures": 0, "scientific_decisions_preserved": True}
    _write(Path(prepared["output"]) / "FULL_REFERENCE_COMPARISONS.json", result)
    return result


def _run_native(prepared: Mapping, context: Mapping) -> dict:
    """Dispatch the finite selection through normal ownership and product APIs."""
    if context.get("execute_allowed") is not True:
        raise ValueError("Native acceptance hardware dispatch is not enabled")
    for name in ("source_plan", "source_raw_report", "source_validation"):
        if _sha(prepared[name]) != prepared[name + "_sha256"]:
            raise ValueError("Native acceptance source changed after preparation: " + name)
    target_binding = _verify_current_target(prepared["target"], prepared["identity"], context)
    if target_binding != prepared["target_binding"]:
        raise ValueError("Native target registry changed after preparation")
    output = Path(prepared["output"])
    transport = _transport(prepared, context)
    reports, normalized, attempts = [], [], []
    started = output / "production-start.json"
    with started.open("x", encoding="utf-8") as stream:
        json.dump({"pid": os.getpid(), "started_at": time.time(), "selection": prepared["identity"]}, stream)
    owner = {"run_id": Path(context["run_dir"]).name, "purpose": "hardware_acceptance_native"}
    with ExitStack() as stack:
        physical = stack.enter_context(EvaluationRunLock.for_resource(physical_dut_key(prepared["target"]), owner=owner))
        def fence_cleanup():
            if context["remote_process_registry"].active_count():
                physical.commit_quarantine_fence({"reason": "hardware_acceptance_native_cleanup_unproven", "evidence": str(output)})
        stack.callback(fence_cleanup)
        for item in prepared["runs"]:
            cancel = context.get("cancel_event")
            if cancel is not None and cancel.is_set():
                raise RuntimeError("Native acceptance cancelled before next repeat")
            directory = output / f"rep{int(item['repeat']):02d}"
            directory.mkdir(parents=True, exist_ok=False)
            command = "cd -- " + shlex.quote(prepared["remote_tool_dir"]) + " && " + shlex.join(item["argv"])
            attempt = {"repeat": item["repeat"], "argv": item["argv"], "environment": item["env"], "started_at": time.time()}
            attempts.append(attempt)
            _write(output / "DISPATCH_ATTEMPTS.json", {"attempts": attempts})
            rc, text = transport.run(command, timeout=int(item.get("timeout_s") or 900), env=item["env"])
            attempt.update(returncode=rc, completed_at=time.time(), console=text[-12000:])
            _write(output / "DISPATCH_ATTEMPTS.json", {"attempts": attempts})
            if rc:
                raise RuntimeError(f"Native acceptance repetition {item['repeat']} failed with rc {rc}")
            report = _collect_repeat(transport, item, directory)
            raw = _read(report)
            proof = _validate_raw(raw, prepared["identity"], prepared["source_contract"])
            proof.update(report=str(report), report_sha256=_sha(report), repeat=item["repeat"],
                         runtime_instance_id=raw.get("runtime_instance_id"))
            _write(directory / "VALIDATED.json", proof)
            normalized.append(proof)
            reports.append(report)
    runtime_ids = [row.get("runtime_instance_id") for row in normalized]
    if not all(runtime_ids) or len(set(runtime_ids)) != 3:
        raise ValueError("Native repetitions did not prove three independent runtime instances")
    summary = _normal_performance_import(prepared, reports, context)
    energy = _energy_capture(prepared, summary, context)
    reference_comparisons = _reference_energy_pairs(prepared, energy["row"], context)
    result = {"status": "completed", "fresh_hardware_executed": True, "new_performance_processes": 3,
              "new_capture_count": energy["new_capture_count"], "valid_repeats": {"performance": 3, "energy": 3},
              "failed_attempts": sum(row["returncode"] != 0 for row in attempts) + energy["failed_attempts"], "attempts": len(attempts),
              "energy_start_accounting": energy["start_accounting"],
              "fps_median": statistics.median(row["fps"] for row in normalized),
              "output_validation": "normal_completed_task_and_output_contract", "performance_summary": str(summary),
              "energy": energy["row"], "normal_projection": True, "full_recaptures": 0,
              "quality_dataset_reexecutions": 0, "full_baseline_references": prepared["full_baseline_references"],
              "full_reference_compatibility": prepared["full_baseline_proofs"],
              "full_reference_comparisons": reference_comparisons,
              "missing_full_energy_comparisons": reference_comparisons["missing_full_energy_comparisons"],
              "evidence_paths": [str(started), str(output / "DISPATCH_ATTEMPTS.json"), str(summary), energy["energy_result"], energy["import"],
                                 str(output / "FULL_REFERENCE_COMPARISONS.json")]}
    _write(output / "RESULT.json", result)
    return result


def run_native(prepared: Mapping, context: Mapping) -> dict:
    """Keep actual starts and valid slots visible after a controlled failure."""
    try:
        return _run_native(prepared, context)
    except Exception as exc:
        output = Path(prepared["output"])
        if not (output / "production-start.json").is_file():
            raise
        attempts_path = output / "DISPATCH_ATTEMPTS.json"
        attempts = _read(attempts_path).get("attempts", []) if attempts_path.is_file() else []
        valid = list(output.glob("rep*/VALIDATED.json"))
        counts = _budget_counts(prepared, context)
        captures = counts["collector_starts"]
        valid_energy = counts["valid_repeats"]
        # A started but unresolved transport call remains an attempted dispatch;
        # it is never reported as proof of a completed hardware run.
        result = {"status": "failed", "error": f"{type(exc).__name__}: {exc}",
                  "fresh_hardware_executed": True if valid or captures else None if attempts else False,
                  "fresh_hardware_execution_unresolved": bool(attempts and not valid),
                  "new_performance_processes": len(attempts), "new_capture_count": captures,
                  "valid_repeats": {"performance": len(valid), "energy": valid_energy},
                  "failed_attempts": sum(item.get("returncode", 1) != 0 for item in attempts)
                                      + counts["failed_attempts"], "energy_start_accounting": counts,
                  "evidence_paths": [str(path) for path in output.rglob("*.json")],
                  "full_recaptures": 0, "quality_dataset_reexecutions": 0}
        _write(output / "RESULT.json", result)
        return result


def main(argv=None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Import one completed Native acceptance energy point; no hardware dispatch")
    parser.add_argument("--validate-energy-view", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    args = parser.parse_args(argv)
    result = validate_energy_view(args.validate_energy_view, args.source)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
