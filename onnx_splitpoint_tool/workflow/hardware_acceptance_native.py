"""One selected Native split through the ordinary runners and energy consumer.

Saved performance plans are provenance and artifact-resolution inputs.  Only
their normal runner argv are reused; private campaign executors are never
imported.  All outputs belong to the new acceptance EvaluationRun.
"""
from __future__ import annotations

from contextlib import ExitStack
import copy
from datetime import datetime
import hashlib
import inspect
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
from ..process_control import bind_workflow_resource_options
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


def request_config(profile: Mapping) -> Mapping:
    """Return the single ordinary finite Native request in a profile."""
    requests = [profile[name] for name in ("hardware_acceptance", "native_refresh")
                if isinstance(profile.get(name), Mapping) and profile[name].get("enabled") is True]
    if len(requests) > 1:
        raise ValueError("hardware acceptance and Native refresh cannot run together")
    return requests[0] if requests else profile.get("hardware_acceptance") or {}


def _functional_mode(value: Mapping) -> bool:
    return (value.get("acceptance_mode") == "functional"
            or request_config(value.get("profile") or {}).get("mode") == "functional")


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
    backend = str(cell.get("measurement_direction") or cell.get("backend") or cell.get("native_backend") or cell.get("profile_split_run_id") or "")
    backend = backend.replace("hailo10_to_tensorrt", "hailo10h_to_trt").replace("hailo10_to_trt", "hailo10h_to_trt")
    result = {"model": str(cell["model_id"]), "case": str(cell["case_id"]),
              "setup_id": str(cell["setup_id"]), "backend": backend}
    full = backend == "native_full_deepx" and result["case"] == "full"
    if not full and (backend not in _RUNNERS or not re.fullmatch(r"b[0-9]{3,}", result["case"])):
        raise ValueError("Native acceptance supports one declared split and known runner")
    if any(not re.fullmatch(r"[A-Za-z0-9_.-]+", text) for text in result.values()):
        raise ValueError("unsafe Native acceptance identity")
    return result


def _argument(argv: list[str], flag: str) -> str:
    if argv.count(flag) != 1 or argv.index(flag) + 1 >= len(argv):
        raise ValueError("Native acceptance requires one argument: " + flag)
    return argv[argv.index(flag) + 1]


def _rewrite_run(item: Mapping, identity: Mapping, remote_root: str, *, deepx_runtime_mode: str | None = None,
                 remote_tool_dir: str | None = None) -> dict:
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
    if any(flag in argv for flag in ("--build", "--force-build", "--force-rebuild", "--build-missing-engine",
                                    "--force-rebuild-engine", "--energy-workload-only")):
        raise ValueError("build or energy mode in Native performance source")
    old_root = str(item["remote_output_root"]).rstrip("/")
    source_benchmark = _argument(argv, "--benchmark-set")
    for path in (old_root, remote_root, source_benchmark):
        if not path.startswith("/") or str(PurePosixPath(path)) != path or ".." in PurePosixPath(path).parts:
            raise ValueError("Native output or benchmark-set path is not canonical")
    if (remote_root == old_root or remote_root.startswith(old_root + "/") or old_root.startswith(remote_root + "/")
            or remote_root == source_benchmark or remote_root.startswith(source_benchmark + "/")
            or source_benchmark.startswith(remote_root + "/")):
        raise ValueError("Native fresh output overlaps historical inputs or outputs")
    # Every supported source uses the ordinary quality-first binding to resolve
    # sealed accelerator artifacts and one explicit image. No dataset tree is copied.
    _argument(argv, "--native-split-quality-binding")
    _argument(argv, "--image")
    for index, flag in enumerate(argv[:-1]):
        if flag in _OUTPUT_FLAGS:
            old = argv[index + 1]
            if old != old_root and not old.startswith(old_root + "/"):
                raise ValueError("saved Native writable path escapes its output root")
            argv[index + 1] = remote_root + old[len(old_root):]
    argv[argv.index("--benchmark-set") + 1] = remote_root + "/benchmark_set"
    if deepx_runtime_mode is not None:
        if identity["backend"] != "deepx_to_trt" or deepx_runtime_mode not in {"A4", "S1"}:
            raise ValueError("explicit DeepX runtime policy is outside the Native DeepX namespace")
        if "--deepx-runtime-mode" in argv:
            argv[argv.index("--deepx-runtime-mode") + 1] = deepx_runtime_mode
        else:
            argv += ["--deepx-runtime-mode", deepx_runtime_mode]
        # Fresh software acceptance runs the installed product, while the
        # original historical command and all model bindings stay preserved.
        if remote_tool_dir is not None:
            argv[argv.index(scripts[0])] = remote_tool_dir.rstrip("/") + "/scripts/" + _RUNNERS[identity["backend"]]
        if deepx_runtime_mode == "A4":
            # The historical contract remains provenance. A fresh software
            # acceptance must bind the worker that normal deployment stages.
            script_name = _RUNNERS[identity["backend"]]
            local_script = Path(__file__).resolve().parents[2] / "scripts" / script_name
            if not local_script.is_file():
                local_script = Path(__file__).resolve().parents[1] / "resources/remote_scripts" / script_name
            current_sha = _sha(local_script)
            if "--expected-runner-sha256" in argv:
                argv[argv.index("--expected-runner-sha256") + 1] = current_sha
            else:
                argv += ["--expected-runner-sha256", current_sha]
    # The archived vendor dependency environment remains ordinary runner
    # configuration; a controller source overlay is never propagated.
    environment = {str(k): str(v) for k, v in dict(item.get("env") or {}).items()
                   if k not in {"PYTHONPATH", "PYTHONHOME"}}
    environment.update(PYTHONDONTWRITEBYTECODE="1", ONNX_SPLITPOINT_NATIVE_TRT_BUILD="0")
    if identity["backend"] == "hailo8_to_trt" and "--no-build" not in argv:
        raise ValueError("Hailo Native acceptance requires explicit no-build")
    result = dict(item, argv=argv, env=environment, remote_output_root=remote_root,
                  source_benchmark_set=source_benchmark,
                  result_json=remote_root + "/" + _RAW_NAMES[identity["backend"]])
    result.pop("remote_telemetry_root", None)
    return result


def _benchmark_case_inputs(contract: Mapping, identity: Mapping) -> list[dict]:
    """Retain the exact optional Hailo10 ONNX canonical-slot input."""
    options = contract.get("runtime_options") or {}
    if options.get("part1_onnx_used") is not True:
        return []
    if identity["backend"] != "hailo10h_to_trt":
        raise ValueError("Native Part1 ONNX helper is not supported for this backend")
    artifact = (contract.get("artifacts") or {}).get("part1_onnx") or {}
    path = str(artifact.get("path") or "")
    name = PurePosixPath(path).name
    digest = str(artifact.get("sha256") or "")
    if (not path.startswith("/") or str(PurePosixPath(path)) != path or ".." in PurePosixPath(path).parts
            or not name.endswith(".onnx") or "part1" not in name or not re.fullmatch(r"[a-f0-9]{64}", digest)
            or not options.get("canonical_input_slot_names") or not options.get("canonical_output_slot_names")):
        raise ValueError("Native Part1 ONNX helper has no exact path/hash/slot binding")
    return [{"path": path, "sha256": digest, "relative_path": identity["case"] + "/" + name}]


def _copy_benchmark_metadata(source, destination, case, case_inputs=()):
    """Create a private writable root from the exact small runner declarations.

    This function is also sent through the ordinary journaled SSHTransport;
    keeping its imports local makes the tested implementation self-contained.
    Artifact and quality-binding paths remain the unchanged sealed inputs.
    """
    import hashlib
    import json
    import re
    from pathlib import Path

    source, destination = Path(source).resolve(strict=True), Path(destination)
    if not source.is_dir() or not destination.is_absolute():
        raise ValueError("Native benchmark metadata source/destination invalid")
    if destination.exists() or destination.is_symlink():
        raise ValueError("Native benchmark metadata destination already exists")
    if any(path.is_symlink() for path in destination.parents):
        raise ValueError("Native benchmark metadata destination has a symlink ancestor")
    canonical = destination.resolve()
    if canonical == source or canonical in source.parents or source in canonical.parents:
        raise ValueError("Native benchmark metadata would overlap its source")
    if not re.fullmatch(r"b[0-9]{3,}", case) or len(case_inputs) > 1:
        raise ValueError("Native benchmark metadata has an invalid case/helper count")
    for item in case_inputs:
        path = Path(str(item.get("path") or ""))
        relative = Path(str(item.get("relative_path") or ""))
        if (not path.is_absolute() or str(path) != str(item.get("path")) or ".." in path.parts
                or path.suffix != ".onnx" or "part1" not in path.name
                or relative.parts != (case, path.name) or not re.fullmatch(r"[a-f0-9]{64}", str(item.get("sha256") or ""))):
            raise ValueError("Native benchmark helper escapes the selected case or lacks an exact hash")
    # Explicit allowlist: no old results, Native pipeline, executable code,
    # other cases or N5000 resources enter the fresh writable tree.
    names = ("benchmark_set.json", "benchmark_plan.json", "output_contracts.json", "split_manifest.json")
    files = []
    for name in names:
        path = source / name
        if not path.is_file():
            if name == "benchmark_set.json":
                raise ValueError("Native benchmark-set declaration missing")
            continue
        if path.stat().st_size > 16 * 1024 * 1024:
            raise ValueError("Native benchmark metadata exceeds bounded size")
        payload = path.read_bytes()
        json.loads(payload)
        digest = hashlib.sha256(payload).hexdigest()
        files.append((name, payload, digest))
    destination.mkdir(parents=True, exist_ok=False)
    copied = []
    for name, payload, digest in files:
        with (destination / name).open("xb") as stream:
            stream.write(payload)
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != digest:
            raise ValueError("Native benchmark metadata changed while staging")
        copied.append({"name": name, "sha256": digest, "size_bytes": len(payload)})
    # The optional graph is only consulted for canonical slot names by the
    # ordinary Hailo10 runner.  It is a copy, never a writable historical link.
    for item in case_inputs:
        path = Path(item["path"])
        target = destination / item["relative_path"]
        if not path.is_file() or path.stat().st_size > 512 * 1024 * 1024:
            raise ValueError("Native benchmark helper missing or exceeds bounded size")
        target.parent.mkdir(parents=True, exist_ok=False)
        digest = hashlib.sha256()
        with path.open("rb") as original, target.open("xb") as staged:
            for block in iter(lambda: original.read(8 * 1024 * 1024), b""):
                staged.write(block)
                digest.update(block)
        if digest.hexdigest() != item["sha256"]:
            raise ValueError("Native benchmark helper differs from its sealed source hash")
        with path.open("rb") as original:
            after = hashlib.sha256()
            for block in iter(lambda: original.read(8 * 1024 * 1024), b""):
                after.update(block)
        if after.hexdigest() != item["sha256"]:
            raise ValueError("Native benchmark helper changed while staging")
        target.chmod(0o444)
        copied.append({"name": item["relative_path"], "sha256": digest.hexdigest(),
                       "size_bytes": target.stat().st_size, "role": "part1_onnx_canonical_slots"})
    return {"source": str(source), "destination": str(destination), "case": case,
            "files": copied, "copied_artifacts": len(case_inputs), "copied_dataset_files": 0,
            "historical_results_copied": False}


def _stage_benchmark_set(transport: SSHTransport, item: Mapping, directory: Path) -> None:
    if item.get("native_full") is True:
        return _stage_full_benchmark_set(transport, item, directory)
    destination = _argument(item["argv"], "--benchmark-set")
    program = (inspect.getsource(_copy_benchmark_metadata) + "\nimport json\nprint(json.dumps(_copy_benchmark_metadata("
               + repr(item["source_benchmark_set"]) + "," + repr(destination) + ","
               + repr(_argument(item["argv"], "--case")) + "," + repr(item.get("benchmark_case_inputs") or [])
               + "),sort_keys=True))\n")
    command = "python3 -B - <<'NATIVE_ACCEPTANCE_METADATA'\n" + program + "NATIVE_ACCEPTANCE_METADATA"
    rc, text = transport.run(command, timeout=120)
    _write(directory / "benchmark_staging.json", {"returncode": rc, "output": text[-12000:]})
    if rc:
        raise RuntimeError("Native private benchmark metadata staging failed: " + text[-2000:])
    proof = json.loads(text)
    if proof.get("destination") != destination or proof.get("historical_results_copied") is not False:
        raise ValueError("Native private benchmark metadata staging proof invalid")


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


def _full_runtime_python(contract: Mapping) -> str:
    """Preserve the sealed invocation, including its virtual environment."""
    artifact = (contract.get("artifacts") or {}).get("runtime_python") or {}
    executable = (artifact.get("interpreter_identity") or {}).get("executable")
    configured = (contract.get("runtime_options") or {}).get("engine_build_python")
    invocation = str(artifact.get("invocation_path") or executable or
                     (configured if configured != "auto" else "") or artifact.get("path") or "")
    if not PurePosixPath(invocation).is_absolute():
        raise ValueError("Native Full requires its sealed absolute Python invocation")
    for value in (artifact.get("invocation_path"), executable, configured):
        if value and value != "auto" and str(value) != invocation:
            raise ValueError("Native Full sealed Python invocation bindings disagree")
    return invocation


def _publish_prepared(prepared: Mapping, context: Mapping) -> dict:
    path = Path(prepared["output"]) / "PREPARED.json"
    if context.get("resume") is True:
        if not path.exists() and not path.parent.exists():
            _write(path, prepared)
            return dict(prepared)
        if not path.is_file() or path.is_symlink():
            raise ValueError("Native Resume requires its existing ordinary PREPARED checkpoint")
        original = _read(path)
        # Debt is an admission observation, not a new artifact/recipe identity.
        for key in set(original) | set(prepared):
            if key != "phase_energy_admission" and original.get(key) != prepared.get(key):
                raise ValueError("Native Resume preparation changed: " + key)
        return original
    _write(path, prepared)
    return dict(prepared)


def _full_benchmark_plan(benchmark, contract):
    """Resolve the existing sealed plan and reject a different Full route."""
    import json
    plan = benchmark.get("plan")
    runs = plan.get("runs") if isinstance(plan, dict) else None
    selected = [row for row in runs or [] if isinstance(row, dict)
                and str(row.get("id") or row.get("name") or "") == "deepx_m1_full"]
    if not isinstance(runs, list) or len(selected) != 1:
        raise ValueError("Full sealed benchmark plan needs exactly one deepx_m1_full run")
    run = selected[0]
    stages = [run.get(key) if isinstance(run.get(key), dict) else {} for key in ("stage1", "stage2")]
    backends = {str(value) for value in (run.get("backend"), run.get("provider"),
                stages[0].get("backend"), stages[1].get("backend")) if value}
    variants = set(run.get("variants") or []) | ({run["variant"]} if run.get("variant") else set())
    case = str(contract.get("input_case") or "")
    declarations = [row for row in benchmark.get("cases") or [] if isinstance(row, dict)
                    and str(row.get("case_id") or row.get("id") or row.get("case") or row.get("folder") or "") == case]
    selected_cases = list(run.get("case_ids") or run.get("cases") or [])
    if run.get("case_id"):
        selected_cases.append(run["case_id"])
    if (run.get("type") not in {"deepx", "deepx_m1", "dx_m1"} or backends != {"deepx_m1"}
            or variants != {"full"} or len(declarations) != 1
            or (selected_cases and case not in selected_cases)
            or run.get("dxnn_path", run.get("artifact")) != "deepx/deepx_m1/full/model.dxnn"
            or run.get("contract_path") != "deepx/deepx_m1/full/output_contract.json"):
        raise ValueError("Full sealed benchmark plan conflicts with its DeepX/full case or artifact route")
    return json.loads(json.dumps(plan))


def _full_case_metadata(primary: Path, contract: Mapping, *, deepx_runtime_mode: str = "S1") -> dict:
    """Bind the declared case in the original, sealed local Full suite."""
    import posixpath
    model, case = str(contract.get("model") or ""), str(contract.get("input_case") or "")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", model) or not re.fullmatch(r"b[0-9]+", case):
        raise ValueError("Full source metadata needs its exact declared model and input case")
    suite = primary / "native_producers/deepx" / model / "benchmark_set"
    expected = (contract.get("artifacts") or {}).get("benchmark_set_manifest") or {}
    manifest = _strict_file_binding({"path": str(suite / "benchmark_set.json"), "sha256": expected.get("sha256")}, suite)
    payload = _read(manifest)
    if deepx_runtime_mode == "A4":
        _full_benchmark_plan(payload, contract)
    declarations = [item for item in payload.get("cases") or [] if isinstance(item, Mapping)
                    and str(item.get("case_id") or item.get("id") or item.get("case") or item.get("folder") or "") == case]
    if (payload.get("model_name") != model or payload.get("model") != "models/" + model + ".onnx"
            or len(declarations) != 1 or any(declarations[0].get(key, case) != case for key in ("case_dir", "folder"))
            or declarations[0].get("manifest", "split_manifest.json") != "split_manifest.json"):
        raise ValueError("Full local source metadata differs from its sealed model/case declaration")
    case_path = suite / case / "split_manifest.json"
    case_path = _strict_file_binding({"path": str(case_path), "sha256": _sha(case_path)}, suite / case)
    split = _read(case_path)
    model_refs = [split.get("full_model"), (split.get("models") or {}).get("full", {}).get("path")]
    if (split.get("schema") != "onnx-splitpoint/split-manifest"
            or (declarations[0].get("boundary") is not None and split.get("boundary") != declarations[0]["boundary"])
            or not any(model_refs) or any(not isinstance(value, str) or posixpath.isabs(value)
                or posixpath.normpath(case + "/" + value) != "models/" + model + ".onnx" for value in model_refs if value is not None)):
        raise ValueError("Full local declared split manifest has conflicting schema/boundary")
    metadata = {"case_id": case,
                "benchmark_manifest": {"path": str(manifest), "sha256": _sha(manifest), "size_bytes": manifest.stat().st_size},
                "case_manifest": {"path": str(case_path), "sha256": _sha(case_path), "size_bytes": case_path.stat().st_size}}
    if deepx_runtime_mode != "A4":
        return metadata
    from ..native_output_endpoint import load_authoritative_output_contract
    attestation = ((contract.get("energy_workload") or {}).get("input_contract") or {}).get("endpoint_semantic_attestation") or {}
    if (attestation.get("status") != "attested" or attestation.get("pass") is not True
            or attestation.get("model_id") != model or attestation.get("backend") != "deepx_m1"
            or attestation.get("variant") != "full"):
        raise ValueError("Full source endpoint lacks its bound authoritative output container")
    outputs = _strict_file_binding({"path": str(suite / "output_contracts.json"),
                                   "sha256": attestation.get("source_file_sha256")}, suite)
    declaration = load_authoritative_output_contract(outputs, backend="deepx_m1", model_id=model,
        variant="full", task=str((attestation.get("authoritative_contract") or {}).get("task") or ""))
    if (declaration.get("authoritative_output_contract") is not True
            or declaration.get("contract_resolution_status") != "attested"
            or declaration.get("source_contract_sha256") != attestation.get("authoritative_contract_sha256")):
        raise ValueError("Full source authoritative output container differs from its sealed endpoint")
    metadata["output_contracts_manifest"] = {"path": str(outputs), "sha256": _sha(outputs), "size_bytes": outputs.stat().st_size}
    return metadata


def _prepare_full(cell: Mapping, context: Mapping, identity: Mapping) -> dict:
    """Bind existing Full artifacts to the ordinary Full runner, offline.

    A historical Full contract supplies model/input evidence. Its old energy
    replay eligibility is preserved separately; it is never replayed as A4.
    """
    reference = cell.get("native_full_reference")
    if reference is None:
        matches = [row for row in cell.get("full_baseline_references", [])
                   if row.get("backend") == identity["backend"]]
        if len(matches) != 1:
            raise ValueError("Full acceptance requires one exact existing Native Full reference")
        reference = matches[0]
    provenance = reference.get("quality_provenance") or {}
    source_report = _bound_path({"path": provenance.get("normal_report"),
                                 "sha256": provenance.get("normal_report_sha256")})
    row = _read(source_report)["rows"][provenance["normal_row_index"]]
    digest = hashlib.sha256(json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    if digest != provenance.get("normal_row_sha256") or any(row.get(k) != v for k, v in identity.items()):
        raise ValueError("Full acceptance original row identity or binding differs")
    contract = copy.deepcopy(row.get("full_command_contract") or {})
    endpoint_attestation = copy.deepcopy((row.get("deepx_prepared_feed_benchmark") or {}).get("source_output_endpoint_attestation"))
    if endpoint_attestation is not None:
        signature = endpoint_attestation.get("tensor_signature") or {}
        tensors = signature.get("tensors") or []
        if (endpoint_attestation.get("attested") is not True or endpoint_attestation.get("status") != "passed"
                or endpoint_attestation.get("endpoint_contract_hash") != (contract.get("energy_workload") or {}).get("source_endpoint_contract_hash")
                or type(signature.get("tensor_count")) is not int or signature["tensor_count"] != len(tensors) or not tensors
                or [tensor.get("index") for tensor in tensors] != list(range(len(tensors)))):
            raise ValueError("Native Full original endpoint signature is not bound to its sealed source")
    unsealed = dict(contract)
    declared = unsealed.pop("contract_sha256", None)
    digest = hashlib.sha256(json.dumps(unsealed, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    if (not declared or digest != declared or any(contract.get(k) != v for k, v in identity.items())
            or not (contract.get("artifacts") or {}).get("dxnn")
            or not contract.get("input_image_sha256") or not contract.get("source_model_sha256")):
        raise ValueError("Full acceptance needs exact sealed model and timing input evidence")
    validation = _bound_path({"path": provenance.get("quality_summary"),
                               "sha256": provenance.get("quality_summary_sha256")})
    source = _bound_path(cell.get("native_source_plan") or {"path": str(source_report), "sha256": _sha(source_report)})
    primary = _bound_path(cell.get("native_original_run") or context["source_primary_run"])
    targets = [target for target in _read(primary / "run_manifest.json").get("hardware_targets", [])
               if target.get("id") == identity["setup_id"]]
    if len(targets) != 1:
        raise ValueError("Full acceptance needs its unique original hardware target")
    target = copy.deepcopy(targets[0])
    target_proof = _verify_current_target(target, identity, context)
    root = Path(str(context.get("acceptance_root") or context["run_dir"])).resolve()
    component = "__".join(identity[key] for key in ("model", "setup_id", "case"))
    output = root / "native" / component
    remote_tool = str(context.get("remote_tool_dir") or "/home/nx/ONNX-Splitpoint-Tool")
    remote_base = "/home/nx/splitpoint_runs/hardware_acceptance/" + root.name + "/native/" + component
    native_options = context.get("profile", {}).get("native_producers") or {}
    mode = native_options.get("deepx_runtime_mode", "S1")
    if mode not in {"S1", "A4"}:
        raise ValueError("Full acceptance DeepX runtime policy is invalid")
    runtime_python = _full_runtime_python(contract)
    case_metadata = _full_case_metadata(primary, contract, deepx_runtime_mode=mode)
    source_benchmark = str(contract.get("benchmark_set") or "")
    image = str(contract.get("input_image") or "")
    if not source_benchmark.startswith("/") or not image.startswith("/"):
        raise ValueError("Full acceptance needs absolute existing benchmark and image paths")
    runs = []
    for repeat in range(1, 4):
        remote_output = remote_base + f"/rep{repeat:02d}"
        remote_input_root = remote_base + f"/rep{repeat:02d}_inputs"
        benchmark = remote_input_root + "/" + identity["model"] + "/benchmark_set"
        timing_image = benchmark + "/resources/acceptance/" + PurePosixPath(image).name
        command = [runtime_python, remote_tool + "/scripts/native_full_baseline_eval_runner.py",
                   "--root", remote_input_root, "--models", identity["model"], "--backends", "deepx",
                   "--frames", "1000", "--warmup", "100", "--repetitions", "1", "--duration-s", "0",
                   "--setup-id", identity["setup_id"], "--comparison-backend", "deepx",
                   "--engine-build-python", runtime_python, "--dump-outputs", "--out-dir", remote_output,
                   "--deepx-runtime-mode", mode, "--image-map", json.dumps({identity["model"]: {"full": timing_image}})]
        runs.append({"repeat": repeat, "native_full": True, "argv": command,
                     "env": {"PYTHONDONTWRITEBYTECODE": "1", "ONNX_SPLITPOINT_NATIVE_TRT_BUILD": "0"},
                     "source_benchmark_set": source_benchmark, "full_benchmark_set": benchmark,
                     "full_source_contract": contract, "remote_tool_dir": remote_tool,
                     "full_source_case_metadata": case_metadata,
                     "full_source_output_endpoint_attestation": endpoint_attestation,
                     "remote_output_root": remote_output,
                     "result_json": remote_output + "/analysis_tables/native_full_baseline_eval.json", "timeout_s": 900})
    prepared = {"schema": "onnx-splitpoint/hardware-acceptance-native-plan/v1", "identity": dict(identity),
                "source_plan": str(source), "source_plan_sha256": _sha(source),
                "source_raw_report": str(source_report), "source_raw_report_sha256": _sha(source_report),
                "source_validation": str(validation), "source_validation_sha256": _sha(validation),
                "source_contract": contract, "source_primary_run": str(primary), "output": str(output),
                "remote_tool_dir": remote_tool, "runs": runs, "target": target, "target_binding": target_proof,
                "full_baseline_proofs": [], "full_baseline_references": [], "deepx_runtime_mode": mode,
                "historical_full_reference": copy.deepcopy(reference),
                "performance_recipe": {"warmup_tasks_per_process": 100, "completed_tasks_per_process": 1000, "processes": 3},
                "energy_recipe": {"warmup": 0, "frames_override": 0, "duration_s": 60, "repeats": 3,
                                  "physical_scope": "FS", "window": "whole_command"},
                "new_performance_processes": 3, "new_selected_energy_repeats": 3,
                "full_recaptures": 3, "quality_dataset_reexecutions": 0, "hardware_executed": False}
    matrix = _bound_path(request_config(context["profile"])["matrix_evidence"])
    prepared["phase_matrix"] = {"path": str(matrix), "sha256": _sha(matrix)}
    reference = _performance_reference(cell)
    if reference is not None:
        retained = _verify_performance_reference(prepared, context, reference)
        prepared.update(fresh_performance_reference=reference, performance_reuse_proof=retained["proof"],
                        new_performance_processes=0, retained_fresh_performance_processes=3)
    continuation = _energy_continuation(cell)
    if continuation is not None:
        if reference is None:
            raise ValueError("Native Full energy continuation requires the retained fresh performance cohort")
        prepared["native_energy_continuation"] = continuation
    prepared["phase_energy_admission"] = _native_phase_debt(prepared, Path(context["run_dir"]).resolve(),
                                                           prepared["phase_matrix"], continuation)
    return _publish_prepared(prepared, context)


def _copy_full_benchmark(source, destination, contract, output_attestation=None, case_metadata=None, deepx_runtime_mode="S1"):
    """Isolate Full publications and refresh only software from the product.

    Copy the existing Full model, compiled artifact, one case declaration and
    timing image. Software is generated on the installed controller and staged
    separately, so this remote helper needs only the Python standard library.
    """
    import hashlib
    import base64
    import json
    import posixpath
    import shutil
    import zlib
    from pathlib import Path

    source, destination = Path(source).resolve(strict=True), Path(destination)
    if (not destination.is_absolute() or destination.exists() or destination.is_symlink()
            or any(parent.is_symlink() for parent in destination.parents)
            or destination.resolve() == source or source in destination.resolve().parents
            or destination.resolve() in source.parents):
        raise ValueError("Full benchmark destination overlaps existing data or is unsafe")
    model = contract["model"]
    if Path(model).name != model:
        raise ValueError("unsafe Full model identity")
    plan_payload = None
    if deepx_runtime_mode == "A4":
        if case_metadata is None:
            raise ValueError("A4 Full staging requires its bound case and endpoint metadata")
        expected_manifest = ((contract.get("artifacts") or {}).get("benchmark_set_manifest") or {}).get("sha256")
        if not expected_manifest or hashlib.sha256((source / "benchmark_set.json").read_bytes()).hexdigest() != expected_manifest:
            raise ValueError("Full handed-off benchmark manifest differs from its sealed metadata")
        benchmark = json.loads((source / "benchmark_set.json").read_text())
        plan = _full_benchmark_plan(benchmark, contract)
        if (source / "benchmark_plan.json").is_file():
            plan_payload = (source / "benchmark_plan.json").read_bytes()
            if json.loads(plan_payload) != plan:
                raise ValueError("Full standalone plan differs from its sealed benchmark plan")
        else:
            plan_payload = (json.dumps(plan, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    embedded = (contract.get("energy_workload") or {}).get("input_contract")
    expected_dxnn = contract["artifacts"]["dxnn"]["sha256"]
    if (not isinstance(embedded, dict) or not isinstance(embedded.get("input"), dict)
            or embedded.get("model_id") != model
            or embedded.get("artifact_sha256", embedded.get("suite_artifact_sha256")) != expected_dxnn):
        raise ValueError("sealed Full input contract does not bind the selected DXNN")
    fresh_contract = json.loads(json.dumps(embedded))
    if output_attestation is not None:
        signature = output_attestation.get("tensor_signature") or {}
        tensors = signature.get("tensors") or []
        if (output_attestation.get("attested") is not True or output_attestation.get("status") != "passed"
                or output_attestation.get("endpoint_contract_hash") != (contract.get("energy_workload") or {}).get("source_endpoint_contract_hash")
                or type(signature.get("tensor_count")) is not int or signature["tensor_count"] != len(tensors) or not tensors
                or [tensor.get("index") for tensor in tensors] != list(range(len(tensors)))
                or len({tensor.get("name") for tensor in tensors}) != len(tensors)):
            raise ValueError("Full output signature differs from its original bound endpoint")
        for tensor in tensors:
            shape = tensor.get("shape")
            if (not isinstance(tensor.get("name"), str) or not tensor["name"] or not isinstance(tensor.get("dtype"), str)
                    or not tensor["dtype"] or not isinstance(shape, list) or not shape
                    or any(type(value) is not int or value <= 0 for value in shape)
                    or tensor.get("rank") != len(shape)):
                raise ValueError("Full bound output tensor signature is incomplete")
        fresh_contract["outputs"] = [{key: tensor[key] for key in ("name", "shape", "dtype")} for tensor in tensors]
        fresh_contract["source_output_endpoint_attestation"] = output_attestation
        fresh_contract["runtime_output_contract_source"] = "original_bound_full_row.source_output_endpoint_attestation"
    declared_case = str(contract.get("input_case") or "")
    case_payload = None
    outputs_payload = None
    if case_metadata is not None:
        benchmark_binding = case_metadata.get("benchmark_manifest") or {}
        case_binding = case_metadata.get("case_manifest") or {}
        expected_benchmark = ((contract.get("artifacts") or {}).get("benchmark_set_manifest") or {}).get("sha256")
        if (case_metadata.get("case_id") != declared_case or not declared_case.startswith("b")
                or not declared_case[1:].isdigit() or Path(declared_case).name != declared_case
                or not expected_benchmark or benchmark_binding.get("sha256") != expected_benchmark
                or hashlib.sha256((source / "benchmark_set.json").read_bytes()).hexdigest() != expected_benchmark
                or Path(str(case_binding.get("path"))).parts[-2:] != (declared_case, "split_manifest.json")):
            raise ValueError("Full handed-off case metadata differs from its sealed benchmark/case binding")
        case_payload = zlib.decompress(base64.b64decode(case_metadata.get("case_manifest_zlib_base64") or "", validate=True))
        if (len(case_payload) != case_binding.get("size_bytes")
                or hashlib.sha256(case_payload).hexdigest() != case_binding.get("sha256")):
            raise ValueError("Full handed-off split manifest bytes differ from their bound source")
        if deepx_runtime_mode == "A4":
            output_binding = case_metadata.get("output_contracts_manifest") or {}
            endpoint = embedded.get("endpoint_semantic_attestation") or {}
            if (Path(str(output_binding.get("path"))).name != "output_contracts.json"
                    or output_binding.get("sha256") != endpoint.get("source_file_sha256")
                    or endpoint.get("status") != "attested" or endpoint.get("pass") is not True
                    or endpoint.get("model_id") != model or endpoint.get("backend") != "deepx_m1"
                    or endpoint.get("variant") != "full"):
                raise ValueError("Full handed-off output container differs from its sealed endpoint binding")
            outputs_payload = zlib.decompress(base64.b64decode(case_metadata.get("output_contracts_zlib_base64") or "", validate=True))
            if (len(outputs_payload) != output_binding.get("size_bytes")
                    or hashlib.sha256(outputs_payload).hexdigest() != output_binding.get("sha256")):
                raise ValueError("Full handed-off output container bytes differ from their sealed source")
        declarations = json.loads((source / "benchmark_set.json").read_text()).get("cases") or []
        selected = [row for row in declarations if isinstance(row, dict)
                    and str(row.get("case_id") or row.get("id") or row.get("case") or row.get("folder") or "") == declared_case]
        split = json.loads(case_payload)
        model_refs = [split.get("full_model"), (split.get("models") or {}).get("full", {}).get("path")]
        if (len(selected) != 1 or split.get("schema") != "onnx-splitpoint/split-manifest"
                or (selected[0].get("boundary") is not None and split.get("boundary") != selected[0]["boundary"])
                or not any(model_refs) or any(not isinstance(value, str) or posixpath.isabs(value)
                    or posixpath.normpath(declared_case + "/" + value) != "models/" + model + ".onnx" for value in model_refs if value is not None)):
            raise ValueError("Full handed-off split manifest is outside its sealed case declaration")
    files = []
    for name in ("benchmark_set.json", "benchmark_plan.json", "output_contracts.json"):
        if name == "benchmark_plan.json" and plan_payload is not None:
            continue
        if name == "output_contracts.json" and outputs_payload is not None:
            continue
        if (source / name).is_file():
            expected = ((contract.get("artifacts") or {}).get("benchmark_set_manifest") or {}).get("sha256") if name == "benchmark_set.json" else None
            files.append((source / name, Path(name), expected))
    files.append((source / "models" / (model + ".onnx"), Path("models") / (model + ".onnx"), contract["source_model_sha256"]))
    full = source / "deepx/deepx_m1/full"
    for path in sorted(full.rglob("*")):
        if path.name == "output_contract.json":
            continue
        if path.is_file() and not path.name.startswith("benchmark_results") and path.suffix not in {".log", ".pyc"}:
            expected = (contract["artifacts"]["dxnn"]["sha256"] if path.name == "model.dxnn" else None)
            files.append((path, path.relative_to(source), expected))
    if not any(relative.as_posix() == "deepx/deepx_m1/full/model.dxnn" for _, relative, _ in files):
        raise ValueError("existing Full DXNN missing; building is disabled")
    if case_payload is None:
        candidates = [source / declared_case / "split_manifest.json", source / "legacy_suite" / declared_case / "split_manifest.json"]
        matching = [path for path in candidates if path.is_file()]
        if not matching:
            raise ValueError("Full benchmark has no existing declared case manifest")
        files.append((matching[0], Path(declared_case) / "split_manifest.json", None))
    image = Path(contract["input_image"])
    files.append((image, Path("resources/acceptance") / image.name, contract["input_image_sha256"]))
    destination.mkdir(parents=True, exist_ok=False)
    copied = []
    for original, relative, expected in files:
        if not original.is_file() or original.stat().st_size > 1024 * 1024 * 1024:
            raise ValueError("Full source file missing or exceeds bounded size: " + str(original))
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        before = hashlib.sha256(original.read_bytes()).hexdigest()
        if expected and before != expected:
            raise ValueError("Full source artifact binding differs: " + str(original))
        shutil.copyfile(original, target)
        if hashlib.sha256(target.read_bytes()).hexdigest() != before or hashlib.sha256(original.read_bytes()).hexdigest() != before:
            raise ValueError("Full source changed during staging")
        copied.append({"path": relative.as_posix(), "sha256": before, "size_bytes": target.stat().st_size})
    if case_payload is not None:
        target = destination / declared_case / "split_manifest.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(case_payload)
        copied.append({"path": target.relative_to(destination).as_posix(), "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                       "size_bytes": target.stat().st_size, "source": case_metadata["case_manifest"]})
    if plan_payload is not None:
        target = destination / "benchmark_plan.json"
        target.write_bytes(plan_payload)
        copied.append({"path": "benchmark_plan.json", "sha256": hashlib.sha256(plan_payload).hexdigest(),
                       "size_bytes": len(plan_payload), "source": "sealed_benchmark_set.json.plan"})
    if outputs_payload is not None:
        target = destination / "output_contracts.json"
        target.write_bytes(outputs_payload)
        copied.append({"path": "output_contracts.json", "sha256": hashlib.sha256(outputs_payload).hexdigest(),
                       "size_bytes": len(outputs_payload), "source": case_metadata["output_contracts_manifest"]})
    # Some original runtime bundles retain the resolved compiler contract only
    # inside their sealed Full command report. Materialize that exact existing
    # payload in this fresh tree; the original report/files stay untouched.
    sidecar = destination / "deepx/deepx_m1/full/output_contract.json"
    sidecar.write_text(json.dumps(fresh_contract, indent=2, ensure_ascii=False) + "\n")
    copied.append({"path": sidecar.relative_to(destination).as_posix(),
                   "sha256": hashlib.sha256(sidecar.read_bytes()).hexdigest(), "size_bytes": sidecar.stat().st_size,
                   "source": "sealed_full_command_contract.energy_workload.input_contract",
                   "output_signature_source": "original_bound_full_row.source_output_endpoint_attestation" if output_attestation else "original_embedded_outputs"})
    return {"source": str(source), "destination": str(destination), "files": copied,
            "case_id": declared_case, "historical_results_copied": False, "model_builds": 0}


def _stage_full_benchmark_set(transport: SSHTransport, item: Mapping, directory: Path) -> None:
    import base64
    import zlib
    case_metadata = copy.deepcopy(item["full_source_case_metadata"])
    runtime_mode = _argument(item["argv"], "--deepx-runtime-mode") or "S1"
    local_case = Path(case_metadata["case_manifest"]["path"])
    local_suite = local_case.parent.parent
    _strict_file_binding(case_metadata["benchmark_manifest"], local_suite)
    _strict_file_binding(case_metadata["case_manifest"], local_suite / case_metadata["case_id"])
    case_metadata["case_manifest_zlib_base64"] = base64.b64encode(zlib.compress(local_case.read_bytes())).decode("ascii")
    if runtime_mode == "A4":
        local_outputs = _strict_file_binding(case_metadata["output_contracts_manifest"], local_suite)
        case_metadata["output_contracts_zlib_base64"] = base64.b64encode(zlib.compress(local_outputs.read_bytes())).decode("ascii")
    program = (inspect.getsource(_full_benchmark_plan) + "\n" + inspect.getsource(_copy_full_benchmark) + "\nprint('FULL_STAGING=' + json.dumps(_copy_full_benchmark("
               + repr(item["source_benchmark_set"]) + "," + repr(item["full_benchmark_set"]) + ","
               + repr(item["full_source_contract"]) + "," + repr(item.get("full_source_output_endpoint_attestation")) + ","
               + repr(case_metadata) + "," + repr(runtime_mode) + "),sort_keys=True))\n")
    # Compress the complete program before ordinary lease/shell quoting: large
    # saved manifests otherwise exceed Linux's limit for one execve argument.
    # Decode the exact existing helpers, metadata and publication import.
    encoded_program = base64.b64encode(zlib.compress(("import json\n" + program).encode("utf-8"))).decode("ascii")
    bootstrap = ("import base64,zlib;exec(compile(zlib.decompress(base64.b64decode("
                 + repr(encoded_program) + ")), '<native-full-staging>', 'exec'))")
    command = shlex.quote(item["argv"][0]) + " -B -c " + shlex.quote(bootstrap)
    rc, text = transport.run(command, timeout=180)
    _write(directory / "benchmark_staging.json", {"returncode": rc, "output": text[-12000:]})
    if rc:
        raise RuntimeError("Native Full benchmark staging failed: " + text[-2000:])
    lines = [line[len("FULL_STAGING="):] for line in text.splitlines() if line.startswith("FULL_STAGING=")]
    proof = json.loads(lines[0]) if len(lines) == 1 else {}
    if (proof.get("destination") != item["full_benchmark_set"] or proof.get("historical_results_copied") is not False
            or proof.get("case_id") != case_metadata["case_id"]):
        raise ValueError("Native Full staging proof invalid")
    staged = {row["path"]: row for row in proof["files"]}
    bindings = [("benchmark_set.json", case_metadata["benchmark_manifest"]),
                (case_metadata["case_id"] + "/split_manifest.json", case_metadata["case_manifest"])]
    if runtime_mode == "A4":
        bindings.append(("output_contracts.json", case_metadata["output_contracts_manifest"]))
    for relative, binding in bindings:
        if (staged.get(relative) or {}).get("sha256") != binding["sha256"]:
            raise ValueError("Native Full staging proof changed its original metadata")
    if runtime_mode == "A4" and (staged.get("benchmark_plan.json") or {}).get("source") != "sealed_benchmark_set.json.plan":
        raise ValueError("Native Full staging proof lacks its sealed benchmark plan")
    # The normal local harness writer has the complete GUI/controller source
    # closure. The selectively deployed Native runtime does not need it. Use
    # the existing suite bundle and transport to deliver only generated code.
    from ..benchmark.suite_refresh import refresh_suite_harness
    from ..remote.bundle import build_suite_bundle
    software = directory / "software"
    software.mkdir(exist_ok=False)
    metadata = ["benchmark_set.json", "benchmark_plan.json", "output_contracts.json",
                str(proof["case_id"]) + "/split_manifest.json"]
    available = {row["path"] for row in proof["files"]}
    for relative in metadata:
        if relative not in available:
            continue
        local = software / relative
        local.parent.mkdir(parents=True, exist_ok=True)
        rc, text = transport.scp_download(item["full_benchmark_set"] + "/" + relative, str(local), timeout=120)
        if rc:
            raise RuntimeError("Full software metadata download failed: " + text[-2000:])
        if _sha(local) != staged[relative]["sha256"] or local.stat().st_size != staged[relative]["size_bytes"]:
            raise ValueError("Full software metadata download differs from its staged binding")
    refresh = refresh_suite_harness(software, benchmark_set_json=software / "benchmark_set.json",
                                    mini_coco_ap50=False, mini_classification_eval=False, software_only=True)
    bundle = directory / "full_software.tar.gz"
    build_suite_bundle(software, bundle, includes=["*.py", "**/*.py"], reuse_if_unchanged=False)
    remote_bundle = item["full_benchmark_set"] + "/full_software.tar.gz"
    rc, text = transport.scp_upload(str(bundle), remote_bundle, timeout=120)
    if rc:
        raise RuntimeError("Full current product software upload failed: " + text[-2000:])
    # Same hash-bound tar delivery as the ordinary suite deployment, merging
    # only the newly generated software into this fresh private input tree.
    extract = ("test " + shlex.quote(_sha(bundle)) + " = \"$(sha256sum -- " + shlex.quote(remote_bundle)
               + " | cut -d ' ' -f 1)\" && tar -xzf " + shlex.quote(remote_bundle)
               + " -C " + shlex.quote(item["full_benchmark_set"]))
    rc, text = transport.run(extract, timeout=120)
    _write(directory / "software_staging.json", {"returncode": rc, "output": text[-4000:],
           "suite_refresh": refresh, "bundle": str(bundle), "bundle_sha256": _sha(bundle)})
    if rc:
        raise RuntimeError("Full current product software extraction failed: " + text[-2000:])


def _validate_full_raw(raw: Mapping, identity: Mapping, source_contract: Mapping, *,
                       remote_root: str | None = None, remote_tool_dir: str | None = None) -> dict:
    from ..resources.remote_scripts.native_producer_energy_plan import _verify_full_command_contract
    rows = raw.get("rows") or []
    if len(rows) != 1 or any(rows[0].get(k) != v for k, v in identity.items()):
        raise ValueError("fresh Native Full scope differs")
    row = rows[0]
    candidate = row.get("full_command_contract") or {}
    expected = {**identity, "remote_root": remote_root or candidate.get("root"),
                "remote_tool_dir": remote_tool_dir or "/home/nx/ONNX-Splitpoint-Tool"}
    contract, reason = _verify_full_command_contract(candidate, expected_identity=expected)
    if contract is None:
        raise ValueError("fresh Native Full command contract rejected: " + reason)
    records = row.get("repetition_records") or []
    if (raw.get("ok") is not True or row.get("ok") is not True or row.get("repetition_count_valid") != 1
            or len(records) != 1 or row.get("outer_makespan_verified") is not True
            or row.get("completed_frames") != 1000
            or int(row.get("warmup_count", row.get("warmup", (contract.get("runtime_options") or {}).get("warmup", -1)))) != 100
            or row.get("postprocess_completion_verified") is not True or not records[0].get("runtime_instance_id")):
        raise ValueError("fresh Native Full count, completion or process identity failed")
    for role in ("dxnn", "source_onnx"):
        old = (source_contract.get("artifacts") or {}).get(role)
        new = (contract.get("artifacts") or {}).get(role)
        expected = old.get("sha256") if old else source_contract.get("source_model_sha256") if role == "source_onnx" else None
        if expected and (not new or new.get("sha256") != expected):
            raise ValueError("fresh Native Full artifact differs: " + role)
    if contract.get("input_image_sha256") != source_contract.get("input_image_sha256"):
        raise ValueError("fresh Native Full timing image differs")
    fps = float(row.get("fps_makespan") or 0)
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fresh Native Full measured rate invalid")
    latency = validate_latency(row.get("request_latency"))
    if latency.get("status") != "complete" or latency.get("count") != 1000 or latency.get("task_complete") is not True:
        raise ValueError("fresh Native Full request latency/task endpoint incomplete")
    return {"completed_tasks": 1000, "warmup": 100, "fps": fps, "latency": latency,
            "runtime_instance_id": records[0]["runtime_instance_id"], "contract_sha256": contract["contract_sha256"], "contract": contract}


def _performance_reference(cell: Mapping) -> dict | None:
    if cell.get("fresh_performance_reference") and cell.get("native_performance_reuse"):
        raise ValueError("ambiguous Native fresh performance reference")
    value = cell.get("fresh_performance_reference") or cell.get("native_performance_reuse")
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("Native fresh performance reference must be explicitly hash-bound")
    if "path" in value:
        if not re.fullmatch(r"[a-f0-9]{64}", str(value.get("sha256") or "")):
            raise ValueError("Native fresh performance descriptor hash missing")
        value = _read(_bound_path(value))
    return copy.deepcopy(dict(value))


def _energy_continuation(cell: Mapping) -> dict | None:
    value = cell.get("native_energy_continuation")
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("Native energy continuation must be explicitly hash-bound")
    if "path" in value:
        if not re.fullmatch(r"[a-f0-9]{64}", str(value.get("sha256") or "")):
            raise ValueError("Native energy continuation descriptor hash missing")
        value = _read(_bound_path(value))
    if (value.get("schema") != "onnx-splitpoint/hardware-acceptance-native-energy-continuation/v1"
            or not isinstance(value.get("prior_budgets"), list) or not value["prior_budgets"]):
        raise ValueError("Native energy continuation needs exact prior normal budgets")
    return copy.deepcopy(dict(value))


def _strict_file_binding(value: Mapping, root: Path) -> Path:
    if not isinstance(value, Mapping) or not re.fullmatch(r"[a-f0-9]{64}", str(value.get("sha256") or "")):
        raise ValueError("Native continuation evidence lacks an explicit hash")
    path = _bound_path(value)
    if (str(path) != str(value.get("path")) or not path.is_file() or path.is_symlink()
            or not path.is_relative_to(root)):
        raise ValueError("Native continuation evidence escapes its normal phase")
    return path


def _verify_native_source_release(continuation: Mapping, chain: Mapping, prepared: Mapping, origin: Path, phase: Path) -> None:
    """Read the exact ordinary recovery; never amend the failed source close."""
    recoveries = continuation.get("source_recovery") or []
    if isinstance(recoveries, Mapping):
        recoveries = [recoveries]
    matches = []
    for descriptor in recoveries:
        request_path = _strict_file_binding(descriptor.get("request"), phase / "hardware_runs")
        reply_path = _strict_file_binding(descriptor.get("reply"), phase / "hardware_runs")
        request = _read(request_path)
        if request.get("attempt_dir") == chain.get("run_directory"):
            matches.append((request_path, reply_path, request, _read(reply_path)))
    if len(matches) != 1:
        raise ValueError("Native unresolved source requires its exact normal recovery receipt")
    request_path, reply_path, request, reply = matches[0]
    recovery = reply.get("recovery") or {}
    identity_fields = ("run_id", "session_id", "operation_id", "token")
    source = prepared["target_binding"]["urecs_address"]
    expected = sorted([physical_dut_key(prepared["target"]), "source:" + source, "controller:capture", "controller:nic"])
    if (not request_path.is_relative_to(origin) or not reply_path.is_relative_to(origin)
            or reply_path.name != request_path.name.removesuffix(".resource-request.json") + ".resource-reply.json"
            or request.get("purpose") != "capture" or request.get("phase") != "FINALIZING"
            or request.get("process_started") is not True or request.get("run_id") != origin.name
            or request.get("setup_id") != prepared["identity"]["setup_id"]
            or request.get("owner_pid") != chain.get("owner_pid")
            or request.get("owner_start_ticks") != chain.get("owner_start_ticks")
            or any(not request.get(key) or reply.get(key) != request[key] for key in identity_fields)
            or any(recovery.get(key) != request[key] for key in identity_fields[:-1])
            or reply.get("state") != "STOP" or reply.get("reason") != "campaign_source_completion_unresolved"
            or recovery.get("status") != "released" or recovery.get("old_attempt_validated") is not False
            or recovery.get("measurements_started") != 0 or (recovery.get("ownership_observation") or {}).get("ok") is not True
            or sorted(recovery.get("resources") or []) != expected):
        raise ValueError("Native recovery does not prove exact normal physical release")


def _native_phase_debt(prepared: Mapping, run_dir: Path, phase_matrix: Mapping, continuation: Mapping | None,
                       *, logical_repeat: str | None = None) -> dict:
    """Count existing normal chains before admission; no second ledger."""
    from ..benchmark.evaluation_profiles import load_evaluation_profile
    matrix = _strict_file_binding(phase_matrix, Path(phase_matrix["path"]).parent)
    phase = matrix.parent
    identity = prepared["identity"]
    cell_directory = phase / "hardware_runs" / (identity["model"] + "__" + identity["setup_id"])
    if run_dir.parent != cell_directory:
        raise ValueError("Native capture admission is outside its fixed hardware phase")
    if Path(prepared["output"]) != run_dir / "native" / "__".join(identity[key] for key in ("model", "setup_id", "case")):
        raise ValueError("Native capture output differs from its selected normal workflow")
    row_id = json.dumps([identity[key] for key in ("backend", "model", "case", "setup_id")], separators=(",", ":"))
    source = prepared["target_binding"]["urecs_address"]
    bound_budgets = {}
    for descriptor in (continuation or {}).get("prior_budgets", []):
        path = _strict_file_binding(descriptor, phase / "hardware_runs")
        if str(path) in bound_budgets:
            raise ValueError("duplicate Native previous budget reference")
        bound_budgets[str(path)] = descriptor
    counts = dict.fromkeys(("repeat:0", "repeat:1", "repeat:2"), 0)
    prior_counts = dict(counts)
    evidence, matched = [], set()
    slot_limits = dict.fromkeys(counts, 3)
    case_limit = 9
    series_command_hash = None
    series_view = None
    for origin in sorted(run_dir.parent.iterdir()):
        path = origin / "energy/energy_task_budget.json"
        if not path.is_file():
            continue
        budget = _read(path)
        if prepared.get("deepx_runtime_mode") == "A4" and origin != run_dir:
            from ..energy.task_budget import campaign_budget_stop_reason
            source_state = (budget.get("sources") or {}).get(source) or {}
            failures = source_state.get("transport_failures")
            if (type(failures) is not int or failures < 0 or failures >= 20
                    or campaign_budget_stop_reason(budget, row_id, source_id=source, parent_run_dir=origin)):
                raise RuntimeError("Native prior phase budget source/task STOP or transport limit requires its original normal recovery")
        task = (budget.get("tasks") or {}).get(row_id)
        if task is None or not task.get("chains"):
            continue
        if origin == run_dir and logical_repeat is not None:
            from ..energy.task_budget import native_resume_capture_series
            series = native_resume_capture_series(budget, row_id, parent_run_dir=run_dir)
            if series is not None:
                if series["stop_reason"]:
                    raise RuntimeError("Native resumed capture series is stopped: " + series["stop_reason"])
                series_command_hash = series["command_contract_sha256"]
                series_view = series["selected_view"]
                original = series["original_chains"]
                for position in series["logical_repeats"]:
                    slot_limits[position] = sum(c["logical_repeat"] == position and c.get("cohort_id") == run_dir.name
                                                for c in original) + series["max_starts"]
                case_limit = max(case_limit, len(original) + series["max_total_starts"])
        if (path.is_symlink() or budget.get("campaign") is not True or budget.get("max_transport_failures") != 20
                or task.get("source_id") != source or task.get("limits") != {"max_chains": 9, "max_retries": 2,
                                                                              "max_transport_failures": 20}):
            raise ValueError("Native phase energy budget/source policy differs")
        if origin != run_dir:
            if str(path) not in bound_budgets:
                raise ValueError("Native prior capture debt lacks its exact budget binding")
            matched.add(str(path))
            manifest = _read(origin / "run_manifest.json")
            closure = _read(origin / "reports/artifact_index_closure.json")
            profile = load_evaluation_profile(origin / "profile.yaml", validate=False).raw_profile
            old_matrix = _bound_path(request_config(profile).get("matrix_evidence"))
            old_prepared = _read(origin / "native" / Path(prepared["output"]).name / "PREPARED.json")
            fixed_fields = ("identity", "source_plan", "source_plan_sha256", "source_raw_report", "source_raw_report_sha256",
                            "source_validation", "source_validation_sha256", "source_contract", "target_binding",
                            "performance_recipe", "energy_recipe", "deepx_runtime_mode")
            matrix_time = datetime.fromisoformat(str(_read(matrix)["recorded_before_hardware_start"]).replace("Z", "+00:00"))
            created = datetime.fromisoformat(str(manifest["created_at"]).replace("Z", "+00:00"))
            if (old_matrix != matrix or manifest.get("run_id") != origin.name
                    or created < matrix_time or any(old_prepared.get(key) != prepared.get(key) for key in fixed_fields)
                    or manifest.get("status") not in {"ok", "partial", "failed", "cancelled"}
                    or (manifest.get("hardware_acceptance") or {}).get("schema") != "onnx-splitpoint/hardware-acceptance/v1"
                    or (manifest.get("hardware_acceptance") or {}).get("source_campaign_resumed") is not False
                    or closure.get("status") != "pass" or closure.get("run_id") != origin.name
                    or closure.get("workflow_status") != manifest.get("status")):
                raise ValueError("Native prior capture is not a closed same-phase acceptance workflow")
        for chain in task["chains"]:
            position = chain.get("logical_repeat")
            if (position not in counts or chain.get("finished") is not True
                    or not Path(str(chain.get("run_directory"))).resolve().is_relative_to(origin)):
                raise ValueError("Native prior capture is active, unknown or outside its normal workflow")
            if chain.get("valid") is True and origin != run_dir:
                raise RuntimeError("Native prior accepted energy slots require normal preserved-slot resume; recapture refused")
            if chain.get("valid") is True and position == logical_repeat:
                raise RuntimeError("Native current accepted energy slot cannot be recaptured")
            if origin != run_dir and chain.get("collector_started") is True and chain.get("source_completion_verified") is not True:
                _verify_native_source_release(continuation or {}, chain, prepared, origin, phase)
            if origin == run_dir and chain.get("collector_started") is True and chain.get("source_completion_verified") is not True:
                raise RuntimeError("Native current source completion is unresolved; normal recovery required")
            counts[position] += 1
            if origin != run_dir:
                prior_counts[position] += 1
        evidence.append({"path": str(path), "sha256": _sha(path), "chains": len(task["chains"]),
                         "physical_collector_starts": sum(chain.get("collector_started") is True for chain in task["chains"])})
    if matched != set(bound_budgets):
        raise ValueError("Native continuation budget is outside selected phase debt")
    if logical_repeat is not None and (logical_repeat not in counts or counts[logical_repeat] >= slot_limits[logical_repeat] or sum(counts.values()) >= case_limit):
        raise RuntimeError("Native cumulative phase retry/chain limit reached for " + str(logical_repeat))
    return {"campaign_row_id": row_id, "campaign_source_id": source, "prior_and_current_chains": counts,
            "prior_chains": prior_counts, "profile_max_retries": 2, "profile_max_chains": 9,
            "budget_evidence": evidence, "slot_chain_limits": slot_limits, "case_chain_limit": case_limit,
            **({"resume_series_command_contract_sha256": series_command_hash, "resume_series_view": series_view}
               if series_command_hash else {})}


def admit_energy_capture(descriptor: Mapping, request: Mapping) -> dict:
    """Normal collector callback before every original or retry reservation."""
    if descriptor.get("schema") != "onnx-splitpoint/hardware-acceptance-native-capture-admission/v1":
        raise ValueError("Native capture admission descriptor schema invalid")
    run_dir = Path(descriptor["run_dir"]).resolve()
    prepared_path = _strict_file_binding(descriptor["prepared"], run_dir)
    prepared = _read(prepared_path)
    identity = prepared["identity"]
    row_id = json.dumps([identity[key] for key in ("backend", "model", "case", "setup_id")], separators=(",", ":"))
    if (str(Path(request["campaign_budget_file"]).resolve()) != str(run_dir / "energy/energy_task_budget.json")
            or request.get("campaign_row_id") != row_id or request.get("setup_id") != identity["setup_id"]
            or request.get("campaign_source_id") != prepared["target_binding"]["urecs_address"]
            or request.get("duration_s") != 60 or request.get("command_contract_sha256") != descriptor["command_contract_sha256"]
            or request.get("require_runtime_work_units") is not True or request.get("require_command_window_alignment") is not True
            or not Path(request["run_directory"]).resolve().is_relative_to(Path(prepared["output"]) / "energy_view")):
        raise ValueError("Native collector request differs from its bound finite job")
    proof = _native_phase_debt(prepared, run_dir, descriptor["phase_matrix"], prepared.get("native_energy_continuation"),
                               logical_repeat=request.get("logical_repeat"))
    if proof.get("resume_series_command_contract_sha256", descriptor["command_contract_sha256"]) != descriptor["command_contract_sha256"]:
        raise ValueError("Native Resume series command differs from its accepted performance")
    if proof.get("resume_series_view") and not Path(request["run_directory"]).resolve().is_relative_to(Path(proof["resume_series_view"])):
        raise ValueError("Native Resume capture escapes its selected normal attempt")
    return {"ok": True, "logical_repeat": request["logical_repeat"], **proof,
            "remaining_chains_for_slot": proof.get("slot_chain_limits", {}).get(request["logical_repeat"], 3) - proof["prior_and_current_chains"][request["logical_repeat"]],
            "remaining_chains_for_case": proof.get("case_chain_limit", 9) - sum(proof["prior_and_current_chains"].values()),
            "normal_recovery_does_not_validate_old_attempt": True}


def _verify_performance_reference(prepared: Mapping, context: Mapping, reference: Mapping) -> dict:
    """Revalidate only a closed, physically completed cohort in this phase.

    Every leaf is supplied with its original hash. No historical campaign raw
    report or cache result can acquire current-phase authority through a name.
    """
    from ..benchmark.evaluation_profiles import load_evaluation_profile

    def bound(value, root=None):
        if not isinstance(value, Mapping) or not re.fullmatch(r"[a-f0-9]{64}", str(value.get("sha256") or "")):
            raise ValueError("Native fresh performance file lacks explicit hash")
        path = _bound_path(value)
        if str(path) != str(value["path"]) or any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError("Native fresh performance evidence is not a canonical regular path")
        if root is not None and not path.is_relative_to(root):
            raise ValueError("Native fresh performance evidence escapes its closed origin")
        return path

    origin_prepared = bound(reference.get("prepared"))
    manifest_path = bound(reference.get("run_manifest"))
    origin = manifest_path.parent
    current = Path(context["run_dir"]).resolve()
    old = _read(origin_prepared)
    manifest = _read(manifest_path)
    closure_path = bound(reference.get("closure"), origin)
    closure = _read(closure_path)
    profile_path = bound(reference.get("profile"), origin)
    old_profile = load_evaluation_profile(profile_path, validate=False).raw_profile
    if (manifest_path.name != "run_manifest.json" or profile_path != origin / "profile.yaml"
            or closure_path != origin / "reports/artifact_index_closure.json"):
        raise ValueError("Native fresh performance terminal bindings have nonstandard paths")
    prior_acceptance = request_config(old_profile)
    current_acceptance = request_config(context.get("profile") or {})
    if (prior_acceptance.get("enabled") is not True or current_acceptance.get("enabled") is not True
            or prior_acceptance.get("schema") != "onnx-splitpoint/hardware-acceptance/v1"
            or (manifest.get("hardware_acceptance") or {}).get("schema") != prior_acceptance.get("schema")
            or (manifest.get("hardware_acceptance") or {}).get("source_campaign_resumed") is not False
            or manifest.get("status") not in {"ok", "partial", "failed", "cancelled"}
            or closure.get("schema") != "onnx-splitpoint/artifact-index-terminal-closure"
            or closure.get("status") != "pass" or closure.get("run_id") != manifest.get("run_id")
            or manifest.get("run_id") != origin.name or closure.get("workflow_status") != manifest.get("status")):
        raise ValueError("Native fresh performance origin is not a closed normal acceptance run")
    matrix = _bound_path(current_acceptance.get("matrix_evidence"))
    old_matrix = _bound_path(prior_acceptance.get("matrix_evidence"))
    if matrix != old_matrix or origin == current or origin.parent != current.parent:
        raise ValueError("Native fresh performance origin is outside the current acceptance phase")
    phase = matrix.parent
    if not origin.is_relative_to(phase / "hardware_runs") or not current.is_relative_to(phase / "hardware_runs"):
        raise ValueError("Native fresh performance origin is outside phase hardware runs")
    matrix_payload = _read(matrix)
    created = datetime.fromisoformat(str(manifest.get("created_at")))
    fixed = datetime.fromisoformat(str(matrix_payload.get("recorded_before_hardware_start")))
    if created < fixed or created > datetime.now(created.tzinfo):
        raise ValueError("Native performance predates the current fixed acceptance phase")
    identity = prepared["identity"]
    if not any(all(row.get(key) == identity[field] for key, field in
                   (("model_id", "model"), ("setup_id", "setup_id"), ("case_id", "case")))
               for row in matrix_payload.get("cells", [])):
        raise ValueError("Native fresh performance selection is not in the fixed phase matrix")
    component = "__".join(identity[key] for key in ("model", "setup_id", "case"))
    old_output = origin / "native" / component
    if origin_prepared != old_output / "PREPARED.json" or Path(str(old.get("output"))) != old_output:
        raise ValueError("Native fresh performance prepared binding has a different origin")
    if old.get("schema") != prepared["schema"] or old.get("identity") != identity:
        raise ValueError("Native fresh performance selection differs")
    for field in ("source_plan", "source_plan_sha256", "source_raw_report_sha256", "source_validation",
                  "source_validation_sha256", "source_primary_run", "source_contract", "target_binding", "performance_recipe",
                  "full_baseline_references", "deepx_runtime_mode"):
        if old.get(field) != prepared.get(field):
            raise ValueError("Native fresh performance source/quality/target differs: " + field)
    for profile in (old_profile, context["profile"]):
        native = profile.get("native_producers") or {}
        energy = native.get("energy") or {}
        task_budget = energy.get("task_budget") or {}
        if (tuple(native.get(key) for key in ("warmup", "frames", "repetitions")) != (100, 1000, 3)
                or tuple(energy.get(key) for key in ("duration_s", "physical_scope", "window_label")) != (60, "FS", "command")
                or task_budget.get("enabled") is not True or task_budget.get("max_retries") != 2
                or task_budget.get("max_transport_failures") != 20):
            raise ValueError("Native fresh performance profile budget differs")
    summary_path = bound(reference.get("summary"), old_output)
    if summary_path != old_output / "reports/native_producer_combined_summary.json":
        raise ValueError("Native fresh performance normal summary path differs")
    summary_rows = _read(summary_path).get("rows") or []
    if (len(summary_rows) != 1 or any(summary_rows[0].get(key) != value for key, value in identity.items())
            or summary_rows[0].get("repetition_count_valid") != 3):
        raise ValueError("Native fresh performance normal import has no complete selected cohort")
    references = reference.get("repeats") or []
    attempts = _read(old_output / "DISPATCH_ATTEMPTS.json").get("attempts") or []
    source_runs = old.get("runs") or []
    if len(references) != 3 or len(attempts) != 3 or len(source_runs) != 3:
        raise ValueError("Native fresh performance requires three original dispatches and raw leaves")
    normalized, reports, leaves, runtime_ids = [], [], [], []
    for repeat, (ref, attempt, source_run) in enumerate(zip(references, attempts, source_runs), 1):
        validation_path = bound(ref.get("validation"), old_output)
        raw_path = bound(ref.get("raw_report"), old_output)
        validation = _read(validation_path)
        raw = _read(raw_path)
        if (validation_path != old_output / f"rep{repeat:02d}/VALIDATED.json"
                or not raw_path.is_relative_to(old_output / f"rep{repeat:02d}/download")
                or validation.get("repeat") != repeat or validation.get("report") != str(raw_path)
                or validation.get("report_sha256") != _sha(raw_path)
                or attempt.get("repeat") != repeat or attempt.get("returncode") != 0
                or attempt.get("argv") != source_run.get("argv")
                or source_run.get("repeat") != repeat or not source_run.get("source_benchmark_set")
                or not source_run.get("remote_output_root", "").startswith(
                    "/home/nx/splitpoint_runs/hardware_acceptance/" + origin.name + "/native/" + component + "/")):
            raise ValueError("Native fresh performance original dispatch/leaf binding invalid")
        full_context = {"remote_root": _argument(source_run["argv"], "--root"),
                        "remote_tool_dir": old.get("remote_tool_dir")} if identity["case"] == "full" else {}
        proof = _validate_raw(raw, identity, prepared["source_contract"], **full_context)
        runtime = proof.get("runtime_instance_id") or raw.get("runtime_instance_id")
        benchmark = source_run.get("full_benchmark_set") if identity["backend"] == "native_full_deepx" else _argument(source_run["argv"], "--benchmark-set")
        if (not runtime or runtime != ref.get("runtime_instance_id") or runtime != validation.get("runtime_instance_id")
                or proof["contract_sha256"] != validation.get("contract_sha256")
                or proof["contract"]["benchmark_set"] != benchmark
                or (prepared.get("deepx_runtime_mode") is not None and
                    (proof["contract"].get("runtime_options") or {}).get("deepx_runtime_mode") != prepared["deepx_runtime_mode"])):
            raise ValueError("Native fresh performance runtime/contract binding differs")
        runtime_ids.append(runtime)
        normalized.append(proof)
        reports.append(str(raw_path))
        leaves.append({"repeat": repeat, "validation": copy.deepcopy(ref["validation"]),
                       "raw_report": copy.deepcopy(ref["raw_report"]), "runtime_instance_id": runtime,
                       "contract_sha256": proof["contract_sha256"]})
    if len(set(runtime_ids)) != 3:
        raise ValueError("Native fresh performance did not prove independent runtime instances")
    proof = {"schema": "onnx-splitpoint/hardware-acceptance-native-performance-reuse/v1", "identity": identity,
             "phase_matrix": {"path": str(matrix), "sha256": _sha(matrix)},
             "origin": {"run_id": origin.name, "run_dir": str(origin),
                        **{key: copy.deepcopy(reference[key]) for key in ("prepared", "summary", "run_manifest", "closure", "profile")}},
             "repeats": leaves, "retained_fresh_performance_processes": 3,
             "new_performance_processes": 0, "old_campaign_or_cache_reused": False}
    return {"proof": proof, "reports": reports, "normalized": normalized}


def prepare_native(cell: Mapping, context: Mapping) -> dict:
    """Resolve one saved selection offline; no SSH, build, capture or quality run."""
    if _functional_mode(context):
        raise ValueError("Fresh Native functional dispatch is unsupported; validate existing captured Native outputs with the normal numeric validators")
    identity = _identity(cell)
    if (int(context.get("frames", 1000)) != 1000 or int(context.get("warmup", 100)) != 100
            or int(context.get("repetitions", 3)) != 3
            or float(context.get("expected_duration_s", 60)) != 60):
        raise ValueError("Native acceptance must retain regular 100/1000/3 and 60-second profile")
    if identity["backend"] == "native_full_deepx":
        return _prepare_full(cell, context, identity)
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
    deepx_mode = (context.get("profile", {}).get("native_producers") or {}).get("deepx_runtime_mode") if identity["backend"] == "deepx_to_trt" else None
    prepared_runs = [_rewrite_run(item, identity, remote_base + f"/rep{int(item['repeat']):02d}",
                                  deepx_runtime_mode=deepx_mode, remote_tool_dir=remote_tool)
                     for item in sorted(runs, key=lambda item: item["repeat"])]
    for item in prepared_runs:
        item["benchmark_case_inputs"] = _benchmark_case_inputs(contract, identity)
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
    if deepx_mode is not None:
        prepared["deepx_runtime_mode"] = deepx_mode
    reference = _performance_reference(cell)
    if reference is not None:
        retained = _verify_performance_reference(prepared, context, reference)
        prepared.update(fresh_performance_reference=reference, performance_reuse_proof=retained["proof"],
                        new_performance_processes=0, retained_fresh_performance_processes=3)
    continuation = _energy_continuation(cell)
    if continuation is not None:
        if reference is None:
            raise ValueError("Native energy continuation requires the retained fresh performance cohort")
        prepared["native_energy_continuation"] = continuation
    matrix = _bound_path(request_config(context["profile"])["matrix_evidence"])
    prepared["phase_matrix"] = {"path": str(matrix), "sha256": _sha(matrix)}
    prepared["phase_energy_admission"] = _native_phase_debt(prepared, Path(context["run_dir"]).resolve(),
                                                            prepared["phase_matrix"], continuation)
    return _publish_prepared(prepared, context)


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


def ensure_current_deepx_runtime(prepared: Mapping, context: Mapping) -> dict:
    """Use the normal Native deployment before a fresh explicit-A4 child.

    The caller owns the physical DUT lock and workflow resource scope. Every
    package byte is staged before imports, as in the ordinary Native workflow.
    These preflights import code and parse CLI options without opening an SDK
    engine or submitting work.
    """
    if prepared.get("deepx_runtime_mode") != "A4":
        return {"status": "not_required"}
    from ..remote_runtime_closure import native_remote_package_closure
    from .runner import (_native_preflight_asset_contract_v264,
                         _native_split_energy_preflight_dependency_spec_v264,
                         _sync_remote_package_asset_v263, _sync_remote_script_v60i,
                         _verify_remote_module_binding_v263)
    runner = context.get("run_process")
    if not callable(runner):
        raise ValueError("Native A4 deployment requires the normal controller process runner")
    target = prepared["target"]["remote"]
    ssh = (str(target.get("user") or "") + "@" if target.get("user") else "") + str(target["host"])
    tool = str(prepared["remote_tool_dir"])
    runtime = str(prepared["runs"][0]["argv"][0])
    if (not PurePosixPath(runtime).is_absolute()
            or re.fullmatch(r"python(?:[0-9]+(?:\.[0-9]+)*)?", PurePosixPath(runtime).name) is None):
        raise ValueError("Native A4 deployment requires its bound runtime interpreter")
    remote_env = "export PATH=" + shlex.quote(str(PurePosixPath(runtime).parent)) + ':"$PATH"'
    proof = {"status": "running", "target": ssh, "remote_tool_dir": tool,
             "runtime_python": runtime, "steps": [], "inference_submitted": False}
    path = Path(prepared["output"]) / "RUNTIME_DEPLOYMENT.json"
    _write(path, proof)
    try:
        proof["local_preflight"] = _native_preflight_asset_contract_v264()
        if proof["local_preflight"].get("ok") is not True:
            raise RuntimeError("Native preflight source/resource contract failed: "
                               + str(proof["local_preflight"].get("errors") or []))
        scripts = (
            ("native_deepx_trt_e2e_from_benchmarkset.py", ("--deepx-runtime-mode", "--deepx-numerical-only", "--native-split-quality-binding")),
            ("native_hailo10_trt_e2e_from_benchmarkset.py", ("class NativeTRT", "_aggregate_repetition_metrics")),
            ("native_trt_from_benchmarkset.py", ("--benchmark-set", "--precision")),
            ("native_full_baseline_eval_runner.py", ("--deepx-runtime-mode", "--deepx-numerical-only", "--remote-contract-preflight")),
            ("native_full_semantic_dump.py", ("--deepx-runtime-mode", "--prepared-input-manifest")),
            ("native_deepx_full_energy_hotloop.py", ("--deepx-runtime-mode", "--expected-dxnn-sha256")),
            ("native_progress.py", ("def stream_command", "class NativeProgressJournal")),
            ("native_host_telemetry.py", ("--compare-root", "--capture-group")),
            ("run_and_report_work_units.py", ("__SPLITPOINT_WORK_UNITS__",)),
            _native_split_energy_preflight_dependency_spec_v264(),
        )
        for name, tokens in scripts:
            proof["steps"].extend(_sync_remote_script_v60i(
                ssh=ssh, remote_tool_dir=tool, script_name=name, required_tokens=tokens,
                timeout=600, process_runner=runner))
        pending = []
        for relative, module, tokens in native_remote_package_closure():
            steps = _sync_remote_package_asset_v263(ssh=ssh, remote_tool_dir=tool,
                relative_path=relative, required_tokens=tokens, timeout=600, process_runner=runner)
            proof["steps"].extend(steps)
            digest = next(str(step["expected_sha256"]) for step in reversed(steps) if step.get("expected_sha256"))
            pending.append((relative, module, digest))
        for relative, module, digest in pending:
            proof["steps"].append(_verify_remote_module_binding_v263(
                ssh=ssh, remote_tool_dir=tool, remote_env=remote_env, module_name=module,
                relative_path=relative, expected_sha256=digest, runtime_python=runtime,
                timeout=180, process_runner=runner))
        prefix = "cd -- " + shlex.quote(tool) + " && " + "PYTHONPATH=" + shlex.quote(tool) + " "
        sdk_sites = "import sys;sites=[];"
        if prepared["identity"]["backend"] == "native_full_deepx":
            sdk_sites += (
                "import runpy;resolver=runpy.run_path("
                + repr(tool.rstrip("/") + "/scripts/native_full_baseline_eval_runner.py")
                + ",run_name='_native_full_metadata');"
                "sites=resolver['_runtime_site_packages']('native_full_deepx');"
                "[sys.path.insert(0,p) for p in reversed(sites) if p not in sys.path];"
            )
        sdk_code = (
            sdk_sites + "import importlib.metadata,inspect,json,dx_engine;"
            "from dx_engine import InferenceEngine;"
            "print(json.dumps({'executable':sys.executable,'dx_engine_module':dx_engine.__file__,"
            "'dx_engine_version':importlib.metadata.version('dx-engine'),"
            "'run_async_signature':str(inspect.signature(InferenceEngine.run_async)),"
            "'wait_signature':str(inspect.signature(InferenceEngine.wait)),"
            "'normal_runtime_sites':sites,'engine_constructed':False}))"
        )
        sdk = runner(["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new", ssh,
                      prefix + shlex.join([runtime, "-B", "-c", sdk_code])],
                     label="native-a4-bound-sdk-metadata", timeout_s=180)
        proof["bound_sdk_metadata"] = {"rc": sdk.returncode, "stdout": str(sdk.stdout or "")[-4000:],
                                       "stderr": str(getattr(sdk, "stderr", "") or "")[-2000:]}
        if sdk.returncode:
            raise RuntimeError("Native A4 bound-interpreter SDK metadata preflight failed")
        checks = (("native_full_baseline_eval_runner.py", ["--root", ".", "--remote-contract-preflight"]),
                  ("native_full_semantic_dump.py", ["--help"]),
                  ("native_deepx_trt_e2e_from_benchmarkset.py", ["--help"]),
                  ("native_deepx_full_energy_hotloop.py", ["--help"]))
        for name, flags in checks:
            command = prefix + shlex.join([runtime, tool.rstrip("/") + "/scripts/" + name, *flags])
            completed = runner(["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new", ssh, command],
                               label="native-a4-import-preflight:" + name, timeout_s=180)
            proof["steps"].append({"name": "import_preflight_" + name, "rc": completed.returncode,
                "stdout_tail": str(completed.stdout or "")[-2000:], "stderr_tail": str(getattr(completed, "stderr", "") or "")[-2000:]})
            if completed.returncode:
                raise RuntimeError("Native A4 remote import preflight failed: " + name)
        proof["status"] = "verified"
    except Exception as exc:
        proof.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        _write(path, proof)
    return proof


def _validate_raw(raw: Mapping, identity: Mapping, source_contract: Mapping, *,
                  remote_root: str | None = None, remote_tool_dir: str | None = None) -> dict:
    if identity["backend"] == "native_full_deepx":
        return _validate_full_raw(raw, identity, source_contract, remote_root=remote_root, remote_tool_dir=remote_tool_dir)
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
    for role in ("hef", "dxnn", "engine", "part1_onnx"):
        old = (source_contract.get("artifacts") or {}).get(role)
        new = (contract.get("artifacts") or {}).get(role)
        if old and (not new or new.get("sha256") != old.get("sha256")):
            raise ValueError("fresh Native artifact differs: " + role)
    if identity["backend"] == "hailo10h_to_trt":
        old_options = source_contract.get("runtime_options") or {}
        for field in ("part1_onnx_used", "canonical_input_slot_names", "canonical_output_slot_names"):
            if options.get(field) != old_options.get(field):
                raise ValueError("fresh Native canonical-slot source differs: " + field)
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
    relative = PurePosixPath(item["result_json"]).relative_to(PurePosixPath(item["remote_output_root"]))
    report = collected / str(relative)
    if not report.is_file():
        raise RuntimeError("fresh Native result publication missing")
    if collected.is_symlink() or any(path.is_symlink() for path in collected.rglob("*")):
        raise ValueError("Native result download contains an unsafe symlink")
    if (item.get("native_full") is True and "--deepx-runtime-mode" in item["argv"]
            and _argument(item["argv"], "--deepx-runtime-mode") == "A4"):
        # A4 replay roles are rooted in the fresh private BenchmarkSet. Keep
        # that portable semantic suffix locally without collecting models or
        # datasets, after the raw row proves publication in the exact scope.
        benchmark = PurePosixPath(item["full_benchmark_set"])
        model = _argument(item["argv"], "--models")
        setup = _argument(item["argv"], "--setup-id") or "unspecified"
        comparison = _argument(item["argv"], "--comparison-backend") or "unspecified"
        semantic_relative = (PurePosixPath("native_full_outputs") / ("model=" + model)
            / "backend=native_full_deepx" / ("setup=" + setup) / ("comparison=" + comparison)
            / "native_full_semantic_dump.json")
        raw = _read(report)
        rows = raw.get("rows") or []
        evidence = rows[0].get("preparation_evidence") if len(rows) == 1 and isinstance(rows[0], Mapping) else None
        if (isinstance(evidence, Mapping)
                and evidence.get("report") == str(benchmark / semantic_relative)):
            authority_binding = (item.get("full_source_case_metadata") or {}).get("output_contracts_manifest") or {}
            if not authority_binding.get("sha256"):
                raise ValueError("Native canonical semantic result lacks its staged output authority binding")
            staged_authority = _strict_file_binding({"path": str(directory / "software/output_contracts.json"),
                "sha256": authority_binding["sha256"]}, directory / "software")
            local_benchmark = collected / model / "benchmark_set"
            destination_semantic = local_benchmark / "native_full_outputs"
            if destination_semantic.exists() or destination_semantic.is_symlink():
                raise ValueError("Native canonical semantic result destination already exists")
            local_benchmark.mkdir(parents=True, exist_ok=True)
            source_semantic = str(benchmark / "native_full_outputs")
            rc, text = transport.scp_download(source_semantic, str(local_benchmark), recursive=True, timeout=600)
            _write(directory / "semantic_download.json", {"source": source_semantic,
                "destination": str(destination_semantic), "publication": evidence["report"],
                "returncode": rc, "output": text[-4000:]})
            if rc or not (local_benchmark / str(semantic_relative)).is_file():
                raise RuntimeError("Native canonical semantic result download failed")
            local_authority = local_benchmark / "output_contracts.json"
            if local_authority.exists() or local_authority.is_symlink():
                raise ValueError("Native canonical output authority destination already exists")
            shutil.copyfile(staged_authority, local_authority)
            if _sha(local_authority) != authority_binding["sha256"] or _sha(staged_authority) != authority_binding["sha256"]:
                raise ValueError("Native canonical output authority changed during collection")
    for path in collected.rglob("*"):
        if path.is_symlink():
            raise ValueError("Native result download contains an unsafe symlink")
    return report


def _performance_summary_proof(prepared: Mapping, summary: Path, reports: list[Path]) -> dict:
    rows = _read(summary).get("rows") or []
    identity = prepared["identity"]
    if (len(rows) != 1 or any(rows[0].get(key) != value for key, value in identity.items())
            or rows[0].get("repetition_count_valid") != 3 or len(reports) != 3):
        raise ValueError("Native performance summary differs from its three accepted leaves")
    name = "full_command_contract" if identity["backend"] == "native_full_deepx" else "native_command_contract"
    contract = rows[0].get(name) or {}
    options = contract.get("runtime_options") or {}
    if (options.get("deepx_runtime_mode") != "A4" or type(options.get("deepx_max_inflight")) is not int
            or options["deepx_max_inflight"] != 4):
        raise ValueError("Native performance summary is not the selected A4 runtime")
    hashes = set()
    for report in reports:
        raw = _read(report)
        leaf = (raw.get("rows") or [{}])[0] if identity["backend"] == "native_full_deepx" else raw
        leaf_contract = leaf.get(name) or {}
        leaf_options = leaf_contract.get("runtime_options") or {}
        if (leaf_options.get("deepx_runtime_mode") != "A4" or type(leaf_options.get("deepx_max_inflight")) is not int
                or leaf_options["deepx_max_inflight"] != 4):
            raise ValueError("Native accepted performance leaf is not A4")
        hashes.add(leaf_contract.get("contract_sha256"))
    command_hash = contract.get("contract_sha256")
    if not re.fullmatch(r"[a-f0-9]{64}", str(command_hash or "")) or command_hash not in hashes:
        raise ValueError("Native summary command contract differs from its accepted performance leaves")
    return {"summary": summary, "row": rows[0], "command_contract_sha256": command_hash}


def verify_performance_import(prepared: Mapping, reports=None) -> dict:
    """Recheck the normal import against immutable accepted worker reports."""
    output = Path(prepared["output"])
    receipt = _read(output / "performance_import.json")
    summary = output / "reports/native_producer_combined_summary.json"
    if reports is None:
        slots = _same_run_performance_slots(prepared)["slots"]
        reports = [Path(slots[index]["report"]) for index in sorted(slots)]
    reports = list(map(Path, reports))
    expected = [{"path": str(path), "sha256": _sha(path)} for path in reports]
    if (receipt.get("returncode") != 0 or receipt.get("source_reports") != expected
            or receipt.get("summary") != {"path": str(summary), "sha256": _sha(summary)}):
        raise ValueError("Native normal performance import binding changed")
    _strict_file_binding(receipt["summary"], output / "reports")
    proof = _performance_summary_proof(prepared, summary, reports)
    if receipt.get("command_contract_sha256") != proof["command_contract_sha256"]:
        raise ValueError("Native performance import command binding changed")
    return proof


def _full_performance_remote_contexts(prepared: Mapping, reports: list[Path]) -> list[dict]:
    """Bind existing reporter contexts to the exact accepted Full dispatches."""
    origin = prepared
    reference = prepared.get("fresh_performance_reference")
    if reference:
        origin = _read(_bound_path(reference["prepared"]))
        if origin.get("identity") != prepared["identity"] or origin.get("remote_tool_dir") != prepared["remote_tool_dir"]:
            raise ValueError("Full performance context origin identity differs")
    output = Path(origin["output"])
    contexts = []
    for report in reports:
        report = _strict_file_binding({"path": str(report), "sha256": _sha(report)}, output)
        matches = []
        for source in origin["runs"]:
            for item, suffix in _performance_attempt_items(source, output):
                relative = PurePosixPath(item["result_json"]).relative_to(PurePosixPath(item["remote_output_root"]))
                expected = (output / (f"rep{source['repeat']:02d}" + suffix) / "download"
                            / PurePosixPath(item["remote_output_root"]).name / str(relative))
                if report == expected.resolve():
                    matches.append(item)
        if len(matches) != 1:
            raise ValueError("Full performance context lacks its exact prepared dispatch leaf")
        item = matches[0]
        root = _argument(item["argv"], "--root")
        proof = _validate_full_raw(_read(report), prepared["identity"], prepared["source_contract"],
            remote_root=root, remote_tool_dir=prepared["remote_tool_dir"])
        if proof["contract"]["benchmark_set"] != item["full_benchmark_set"]:
            raise ValueError("Full performance context benchmark differs from prepared dispatch")
        contexts.append({"schema": "onnx-splitpoint/native-final-report-remote-execution-context",
                         "schema_version": 1, "setup_id": prepared["identity"]["setup_id"],
                         "remote_root": root, "remote_tool_dir": prepared["remote_tool_dir"]})
    if len(contexts) != 3 or len({item["remote_root"] for item in contexts}) != 3:
        raise ValueError("Full performance context requires three independent prepared roots")
    return contexts


def _normal_performance_import(prepared: Mapping, reports: list[Path], context: Mapping) -> Path:
    output = Path(prepared["output"])
    identity = prepared["identity"]
    roots = []
    for index, report in enumerate(reports, 1):
        if identity["backend"] == "native_full_deepx":
            roots.append(report.parent.parent)
            continue
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
    if identity["backend"] == "native_full_deepx" and prepared.get("deepx_runtime_mode") == "A4":
        command += ["--remote-execution-context-json", json.dumps(_full_performance_remote_contexts(prepared, reports), sort_keys=True)]
    receipt = output / "performance_import.json"
    summary = reports_dir / "native_producer_combined_summary.json"
    if context.get("resume") is True and summary.is_file() and receipt.is_file():
        previous = _read(receipt)
        existing = _read(summary).get("rows") or []
        if (previous.get("command") != command or previous.get("returncode") != 0
                or len(existing) != 1 or any(existing[0].get(key) != value for key, value in identity.items())
                or existing[0].get("repetition_count_valid") != 3):
            raise ValueError("Native Resume normal performance import checkpoint changed")
        if prepared.get("deepx_runtime_mode") == "A4":
            verify_performance_import(prepared, reports)
        return summary
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
    if prepared.get("deepx_runtime_mode") == "A4":
        proof = _performance_summary_proof(prepared, summary, reports)
        imported = _read(receipt)
        imported.update(summary={"path": str(summary), "sha256": _sha(summary)},
                        source_reports=[{"path": str(path), "sha256": _sha(path)} for path in reports],
                        command_contract_sha256=proof["command_contract_sha256"])
        _write(receipt, imported)
    return summary


def _budget_counts(prepared: Mapping, context: Mapping) -> dict:
    budget = Path(str(context["budget_file"]))
    identity = prepared["identity"]
    key = json.dumps([identity[name] for name in ("backend", "model", "case", "setup_id")], separators=(",", ":"))
    task = (_read(budget).get("tasks") or {}).get(key, {}) if budget.is_file() else {}
    chains = task.get("chains") or []
    cohort = Path(context["run_dir"]).name if context.get("finite_config") else None
    valid = {chain.get("logical_repeat") for chain in chains if chain.get("valid") is True and chain.get("finished") is True
             and (cohort is None or chain.get("cohort_id") == cohort)}
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


def _verify_a4_energy_row(prepared: Mapping, row: Mapping, command_hash: str) -> None:
    if (row.get("deepx_runtime_mode") != "A4" or type(row.get("deepx_max_inflight")) is not int
            or row["deepx_max_inflight"] != 4):
        raise ValueError("Native energy row is not the selected A4 runtime")
    root = Path(prepared["output"]) / "energy_view/reports/native_energy_measurements"
    source = Path(str(row.get("source") or ""))
    source = _strict_file_binding({"path": str(source), "sha256": _sha(source)}, root)
    matches = [item.get("row") or {} for item in _read(source).get("rows") or []
               if all((item.get("row") or {}).get(key) == value for key, value in prepared["identity"].items())]
    if len(matches) != 1:
        raise ValueError("Native energy row lacks its unique normal source plan")
    plan = matches[0]
    contract_path = _strict_file_binding({"path": plan.get("command_contract_file"),
                                         "sha256": plan.get("command_contract_file_sha256")}, root)
    contract = _read(contract_path)
    options = contract.get("runtime_options") or {}
    if (str(contract_path) != row.get("command_contract_source")
            or plan.get("successful_command_contract_sha256") != command_hash
            or contract.get("contract_sha256") != command_hash
            or plan.get("deepx_runtime_mode") != "A4" or type(plan.get("deepx_max_inflight")) is not int
            or plan["deepx_max_inflight"] != 4 or options.get("deepx_runtime_mode") != "A4"
            or type(options.get("deepx_max_inflight")) is not int or options["deepx_max_inflight"] != 4):
        raise ValueError("Native energy command differs from its accepted A4 performance contract")


def _native_energy_environment_prefix(prepared: Mapping) -> str:
    environment = dict(prepared["runs"][0]["env"])
    keys = list(environment)
    if (prepared.get("deepx_runtime_mode") == "A4"
            and prepared["identity"]["backend"] in {"deepx_to_trt", "native_full_deepx"}):
        # PREPARED.json sorts object keys. Reconstruct the ordinary fresh
        # initializer's fixed prefix so Resume retains the exact managed
        # invocation contract instead of hashing JSON's different key order.
        prefix = ("PYTHONDONTWRITEBYTECODE", "ONNX_SPLITPOINT_NATIVE_TRT_BUILD")
        keys = [key for key in prefix if key in environment]
        keys += sorted(key for key in environment if key not in prefix)
    return " ".join(key + "=" + shlex.quote(environment[key]) for key in keys)


def _energy_capture(prepared: Mapping, summary: Path, context: Mapping) -> dict:
    output = Path(prepared["output"])
    identity = prepared["identity"]
    energy_view = output / "energy_view"
    measurement = energy_view / "reports/native_energy_measurements"
    validation_out = energy_view / "reports/native_validation/native_producer_validation_summary.json"
    validation_out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(prepared["source_validation"], validation_out)
    remote = prepared["target"]["remote"]
    producer = {"hailo8_to_trt": "hailo8", "hailo10h_to_trt": "hailo10", "deepx_to_trt": "deepx", "native_full_deepx": "deepx"}[identity["backend"]]
    native = dict(context.get("profile", {}).get("native_producers") or {})
    policy = dict((native.get("energy") or {}).get("task_budget") or {})
    if policy.get("max_retries", 2) != 2 or policy.get("max_transport_failures", 20) != 20:
        raise ValueError("Native acceptance retry policy differs from saved profile")
    summary_rows = _read(summary).get("rows") or []
    if len(summary_rows) != 1 or any(summary_rows[0].get(key) != value for key, value in identity.items()):
        raise ValueError("Native energy admission summary changed selected scope")
    contract_name = "full_command_contract" if identity["backend"] == "native_full_deepx" else "native_command_contract"
    command_hash = str((summary_rows[0].get(contract_name) or {}).get("contract_sha256") or "")
    if not re.fullmatch(r"[a-f0-9]{64}", command_hash):
        raise ValueError("Native energy admission lacks the normal imported command contract")
    if prepared.get("deepx_runtime_mode") == "A4":
        references = (_verify_performance_reference(prepared, context, prepared["fresh_performance_reference"])["reports"]
                      if prepared.get("fresh_performance_reference") else None)
        if verify_performance_import(prepared, references)["command_contract_sha256"] != command_hash:
            raise ValueError("Native energy admission performance command changed")
    remote_root = "/home/nx/native_fifo_evalsets"
    if identity["backend"] == "native_full_deepx" and prepared.get("deepx_runtime_mode") == "A4":
        receipt = _read(output / "performance_import.json")
        contexts = _full_performance_remote_contexts(prepared, [Path(item["path"]) for item in receipt["source_reports"]])
        contract = summary_rows[0]["full_command_contract"]
        selected = [item for item in contexts if item["remote_root"] == contract.get("root")]
        if len(selected) != 1:
            raise ValueError("Full energy context is not an accepted prepared performance root")
        from ..resources.remote_scripts.native_producer_energy_plan import _verify_full_command_contract
        verified, reason = _verify_full_command_contract(contract, expected_identity={**identity,
            "remote_root": selected[0]["remote_root"], "remote_tool_dir": selected[0]["remote_tool_dir"]})
        if verified is None:
            raise ValueError("Full energy context contract rejected: " + reason)
        remote_root = selected[0]["remote_root"]
    prepared_path = output / "PREPARED.json"
    if _read(prepared_path) != prepared:
        raise ValueError("Native energy admission preparation changed")
    debt = _native_phase_debt(prepared, Path(context["run_dir"]).resolve(), prepared["phase_matrix"],
                              prepared.get("native_energy_continuation"))
    counts_before = _budget_counts(prepared, context)
    cohort = _authorize_energy_cohort(prepared, context) if prepared.get("deepx_runtime_mode") == "A4" and context.get("finite_config") else None
    admission_path = output / "CAPTURE_ADMISSION.json"
    admission = {"schema": "onnx-splitpoint/hardware-acceptance-native-capture-admission/v1",
                           "run_dir": str(Path(context["run_dir"]).resolve()), "phase_matrix": prepared["phase_matrix"],
                           "prepared": {"path": str(prepared_path), "sha256": _sha(prepared_path)},
                           "command_contract_sha256": command_hash, "phase_energy_debt_before_dispatch": debt}
    if context.get("resume") is True and admission_path.is_file():
        previous_admission = _read(admission_path)
        if any(previous_admission.get(key) != value for key, value in admission.items() if key != "phase_energy_debt_before_dispatch"):
            raise ValueError("Native Resume capture admission binding changed")
    else:
        _write(admission_path, admission)
    from .native_energy_reentry import select_energy_view
    selected_view, previous_results = select_energy_view(prepared, context, energy_view, command_hash)
    if selected_view != energy_view:
        energy_view = selected_view
        measurement = energy_view / "reports/native_energy_measurements"
        validation_out = energy_view / "reports/native_validation/native_producer_validation_summary.json"
        validation_out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(prepared["source_validation"], validation_out)
        # Preserve the predecessor's ordinary invocation/import before the
        # selected attempt publishes its own dispatch and normal result.
        for name in ("energy_dispatch.json", "NORMAL_ENERGY_IMPORT.json", "RESULT.json"):
            source = output / name
            saved = energy_view / "resume_history" / name
            if source.is_file() and not source.is_symlink() and not saved.exists():
                saved.parent.mkdir(parents=True, exist_ok=True)
                with saved.open("xb") as stream:
                    stream.write(source.read_bytes())
    command = [str(context["python"]), "-I", "-B", "-u", "-m",
               "onnx_splitpoint_tool.resources.remote_scripts.run_native_producer_energy_from_summary",
               "--summary", str(summary), "--validation-summary", prepared["source_validation"],
               "--out-dir", str(measurement), "--hardware-setups-file", str(context.get("hardware_registry_path") or context["hardware_registry"]),
               "--remote-tool-dir", prepared["remote_tool_dir"], "--remote-root", remote_root,
               "--duration-s", "60", "--frames", "0", "--warmup", "0", "--runs", "3",
               "--timeout", "900", "--physical-scope", "FS", "--window-label", "command",
               "--require-runtime-work-units", "--require-command-window-alignment", "--allow-unpaired",
               "--campaign-budget-file", str(context["budget_file"]), "--campaign-max-retries", "2",
               "--campaign-max-transport-failures", "20", "--" + producer + "-ssh", remote["user"] + "@" + remote["host"]]
    validator = [str(context["python"]), "-I", "-B", "-m", "onnx_splitpoint_tool.workflow.hardware_acceptance_native",
                 "--validate-energy-view", str(energy_view)]
    command += ["--require-environment-observation", "--environment-run-id", Path(context["run_dir"]).name,
                "--completed-row-validator", shlex.join(validator),
                "--capture-admission-file", str(admission_path), "--capture-admission-sha256", _sha(admission_path),
                "--capture-admission-hook", "onnx_splitpoint_tool.workflow.hardware_acceptance_native:admit_energy_capture"]
    if cohort:
        command += ["--campaign-cohort", cohort["cohort_id"]]
    if context.get("resume") is True and (measurement / "stages/native_energy/stage_result.json").is_file():
        command.append("--resume-checkpoint")
    # Exact native Split source environments are retained; there is no special
    # source import overlay in this normal collector command.
    environment_prefix = _native_energy_environment_prefix(prepared)
    if environment_prefix:
        command += ["--" + producer + "-env", "export " + environment_prefix]
    if previous_results is not None:
        # Publish the real selected destination before a child can reserve a
        # physical chain. An interrupted parent must Resume this exact attempt.
        _write(output / "energy_dispatch.json", {"command": command, "returncode": None,
            "pending": True, "stdout": "", "stderr": "", "start_accounting_before": counts_before,
            "pre_capture_predecessor": str(previous_results)})
    result = _run_local(command, context, label="acceptance-native-energy", timeout=10800)
    _write(output / "energy_dispatch.json", {"command": command, "returncode": result.returncode,
                                            "stdout": result.stdout, "stderr": getattr(result, "stderr", ""),
                                            "start_accounting_before": counts_before,
                                            **({"pre_capture_predecessor": str(previous_results)}
                                               if previous_results is not None else {})})
    rows = collect_native_energy(energy_view, include_partial=True)
    selected = [row for row in rows if all(row.get(key) == value for key, value in identity.items())]
    _write(output / "NORMAL_ENERGY_IMPORT.json", {"rows": selected, "source": str(measurement), "returncode": result.returncode})
    counts = _budget_counts(prepared, context)
    if (len(selected) != 1 or selected[0].get("ok") is not True or selected[0].get("energy_repeat_valid_n") != 3
            or selected[0].get("energy_calibration_verified") is not True
            or selected[0].get("environment_observation_status") != "pass" or counts["valid_repeats"] != 3):
        raise RuntimeError("normal Native energy consumer did not admit three fresh valid repeats")
    if prepared.get("deepx_runtime_mode") == "A4":
        _verify_a4_energy_row(prepared, selected[0], command_hash)
    return {"row": selected[0], "new_capture_count": counts["collector_starts"] - counts_before["collector_starts"],
            "retained_valid_repeats": counts_before["valid_repeats"], "valid_repeats": counts["valid_repeats"],
            "failed_attempts": counts["failed_attempts"] - counts_before["failed_attempts"], "start_accounting": counts,
            "phase_energy_admission": debt, "capture_admission": str(admission_path),
            "energy_result": str(measurement / "native_producer_energy_results.json"),
            "import": str(output / "NORMAL_ENERGY_IMPORT.json")}


def _authorize_energy_cohort(prepared: Mapping, context: Mapping) -> dict:
    """Bind the normal final-cohort API to this validated finite GUI request."""
    from ..energy.task_budget import authorize_final_cohort
    config = context["finite_config"]
    if config.get("phase") != "complete" or context.get("acceptance_phase") != "complete":
        raise ValueError("numerical admission cannot authorize an energy cohort")
    source = prepared["target_binding"]["urecs_address"]
    rows = []
    for cell in config["cells"]:
        if context["source_ids"][cell["setup_id"]] != source:
            raise ValueError("finite final cohort spans different physical sources")
        if cell["runtime_variants"] != ["native"] or cell["backend"] not in {"deepx_to_trt", "native_full_deepx"}:
            raise ValueError("Native A4 final cohort contains an unsupported runtime")
        rows.append(json.dumps([cell[key] for key in ("backend", "model_id", "case_id", "setup_id")], separators=(",", ":")))
    root = Path(context["run_dir"]).resolve()
    authorization = {"purpose": "final", "repeats": 3, "row_ids": sorted(rows), "generation": root.name,
                     "explicit_user_request": dict(context["finite_request_binding"])}
    return authorize_final_cohort(context["budget_file"], cohort_id=root.name, parent_run_dir=root,
                                  row_ids=rows, source_id=source, authorization=authorization)


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


def _deepx_completion_safety(raw: Mapping) -> dict:
    """Read shared device completion evidence independently of result validity."""
    from .hardware_acceptance_numerical import device_completion_safety
    return device_completion_safety(raw)


def _performance_retry_item(item: Mapping, retry: int = 1) -> dict:
    """A separately recorded technical repair; never overwrite a prior leaf."""
    if type(retry) is not int or retry < 1:
        raise ValueError("Native technical repair index must be positive")
    result = copy.deepcopy(dict(item))
    suffix = f"_retry{retry:02d}"
    old, fresh = item["remote_output_root"], item["remote_output_root"] + suffix
    replacements = [(old, fresh)]
    if item.get("native_full"):
        root = _argument(item["argv"], "--root")
        replacements.append((root, root + suffix))
    def replace(value):
        if isinstance(value, str):
            for source, destination in replacements:
                if value == source or value.startswith(source + "/"):
                    return destination + value[len(source):]
            return value
        if isinstance(value, list):
            return [replace(child) for child in value]
        if isinstance(value, dict):
            return {key: replace(child) for key, child in value.items()}
        return value
    # Source contracts are immutable provenance, not writable destinations.
    for key in ("argv", "remote_output_root", "result_json", "full_benchmark_set"):
        if key in result:
            result[key] = replace(result[key])
    if item.get("native_full"):
        position = result["argv"].index("--image-map") + 1
        result["argv"][position] = json.dumps(replace(json.loads(item["argv"][position])))
    return result


def _performance_attempt_items(source: Mapping, output: Path) -> list:
    """Find the ordinary attempt leaves, including pre-worker staging failures."""
    base = f"rep{source['repeat']:02d}"
    items = [(source, "")]
    base_path = output / base
    if base_path.is_symlink():
        raise ValueError("Native performance attempt is an unsafe symlink")
    for directory in sorted(output.glob(base + "_retry*")):
        match = re.fullmatch(re.escape(base) + r"_retry([0-9]{2,})", directory.name)
        retry = int(match.group(1)) if match else 0
        if (retry < 1 or directory.name != base + f"_retry{retry:02d}"
                or directory.is_symlink() or not directory.is_dir()):
            raise ValueError("Native performance repair leaf is invalid")
        items.append((_performance_retry_item(source, retry), f"_retry{retry:02d}"))
    return sorted(items, key=lambda pair: int(pair[1][6:]) if pair[1] else 0)


def _same_run_performance_slots(prepared: Mapping, *, recover_completed=False) -> dict:
    """Revalidate ordinary raw/VALIDATED leaves before retaining any slot."""
    output = Path(prepared["output"])
    dispatch = output / "DISPATCH_ATTEMPTS.json"
    state = _read(dispatch) if dispatch.is_file() else {}
    attempts = state.get("attempts", [])
    retained = {}
    for source in prepared["runs"]:
        repeat = source["repeat"]
        items = _performance_attempt_items(source, output)
        for attempt in (row for row in attempts if row.get("repeat") == repeat):
            if sum(attempt.get("argv") == item["argv"] for item, _ in items) != 1:
                raise ValueError("Native original performance dispatch leaf is missing or changed")
        for item, suffix in items:
            directory = output / (f"rep{repeat:02d}" + suffix)
            checkpoint = directory / "VALIDATED.json"
            matching = [attempt for attempt in attempts if attempt.get("repeat") == repeat and attempt.get("argv") == item["argv"]]
            recovering = not checkpoint.is_file()
            if recovering:
                relative = PurePosixPath(item["result_json"]).relative_to(PurePosixPath(item["remote_output_root"]))
                raw_path = directory / "download" / PurePosixPath(item["remote_output_root"]).name / str(relative)
                if not recover_completed or not raw_path.is_file() or len(matching) != 1 or matching[0].get("returncode") != 0:
                    continue
                validation = {"report": str(raw_path), "report_sha256": _sha(raw_path)}
                raw_path = _strict_file_binding({"path": validation["report"], "sha256": validation["report_sha256"]}, directory / "download")
                if prepared.get("deepx_runtime_mode") == "A4":
                    completion = directory / "SDK_COMPLETION.json"
                    safety = _read(completion) if completion.is_file() else {}
                    if (safety.get("device_completion_proven") is not True
                            or (safety.get("owned_dispatch_closure") or {}).get("proven") is not True):
                        continue
            else:
                validation = _read(checkpoint)
                raw_path = _strict_file_binding({"path": validation.get("report"), "sha256": validation.get("report_sha256")}, directory / "download")
            if (len(matching) != 1 or matching[0].get("returncode") != 0
                    or type(matching[0].get("completed_at")) not in {int, float}
                    or matching[0].get("environment") != item["env"]):
                raise ValueError("Native retained performance lacks its exact completed ordinary dispatch")
            raw = _read(raw_path)
            if prepared.get("deepx_runtime_mode") == "A4" and not _deepx_completion_safety(raw)["device_completion_proven"]:
                raise ValueError("Native retained performance SDK completion is unproven")
            try:
                proof = _validate_raw(raw, prepared["identity"], prepared["source_contract"],
                    remote_root=_argument(item["argv"], "--root") if item.get("native_full") else None,
                    remote_tool_dir=prepared["remote_tool_dir"])
            except (ValueError, TypeError, KeyError):
                if recovering:
                    continue  # An unaccepted failed report is not a valid retained slot.
                raise
            proof.update(report=str(raw_path), report_sha256=_sha(raw_path), repeat=repeat,
                         runtime_instance_id=proof.get("runtime_instance_id") or raw.get("runtime_instance_id"))
            if prepared.get("deepx_runtime_mode") == "A4" and (proof["contract"].get("runtime_options") or {}).get("deepx_runtime_mode") != "A4":
                raise ValueError("Native retained performance mode differs")
            if checkpoint.is_file() and validation != proof:
                raise ValueError("Native retained performance checkpoint differs from its original raw report")
            if repeat in retained:
                raise ValueError("Native performance has two accepted leaves for one logical repeat")
            if not checkpoint.is_file():
                _write(checkpoint, proof)
            retained[repeat] = proof
    runtime_ids = [proof["runtime_instance_id"] for proof in retained.values()]
    if not all(runtime_ids) or len(runtime_ids) != len(set(runtime_ids)):
        raise ValueError("Native retained performance runtime instances are not independent")
    return {"slots": retained, "attempts": attempts,
            "preparation_attempts": state.get("preparation_attempts", [])}


def _admit_performance_attempt(prepared: Mapping, context: Mapping, *, replacement=False) -> None:
    """Keep original starts bound to their Run without a repair lifetime cap."""
    _read(_bound_path(prepared["phase_matrix"]))
    root = Path(prepared["phase_matrix"]["path"]).parent / "hardware_runs"
    identity, output = prepared["identity"], Path(prepared["output"])
    for dispatch in root.glob("*/*/native/*/DISPATCH_ATTEMPTS.json"):
        old_path = dispatch.parent / "PREPARED.json"
        if not old_path.is_file() or dispatch.is_symlink():
            raise ValueError("Native phase contains an unbound performance dispatch")
        old = _read(old_path)
        if old.get("phase_matrix") != prepared["phase_matrix"]:
            continue
        records = _read(dispatch).get("attempts") or []
        if old.get("identity") == identity:
            if dispatch.parent != output and records and not prepared.get("fresh_performance_reference"):
                raise ValueError("Native point has previous performance starts; Resume its original Run to preserve slots")


def verify_same_run_resume_result(result: Mapping, context: Mapping) -> bool:
    """Verify retained ordinary slots without claiming new physical starts."""
    prepared = context.get("prepared_native") or {}
    if context.get("resume") is not True or prepared.get("deepx_runtime_mode") != "A4":
        return False
    output = Path(prepared["output"])
    if not output.resolve().is_relative_to(Path(context["run_dir"]).resolve()) or _read(output / "PREPARED.json") != prepared:
        raise ValueError("Native Resume result preparation is not in its selected Run")
    if prepared.get("fresh_performance_reference"):
        performance = _verify_performance_reference(prepared, context, prepared["fresh_performance_reference"])
        valid_performance = len(performance["reports"])
    else:
        performance = _same_run_performance_slots(prepared)
        valid_performance = len(performance["slots"])
    summary = output / "reports/native_producer_combined_summary.json"
    receipt = _read(output / "performance_import.json")
    rows = _read(summary).get("rows") or []
    if (valid_performance != 3 or receipt.get("returncode") != 0 or len(rows) != 1
            or any(rows[0].get(key) != value for key, value in prepared["identity"].items())
            or rows[0].get("repetition_count_valid") != 3 or result.get("performance_summary") != str(summary)):
        raise ValueError("Native Resume lacks its normal three-process performance import")
    reports = performance["reports"] if prepared.get("fresh_performance_reference") else [
        Path(performance["slots"][index]["report"]) for index in sorted(performance["slots"])]
    imported = verify_performance_import(prepared, reports)
    from .native_energy_reentry import selected_energy_view
    energy_rows = collect_native_energy(selected_energy_view(prepared), include_partial=True)
    selected = [row for row in energy_rows if all(row.get(key) == value for key, value in prepared["identity"].items())]
    counts = _budget_counts(prepared, context)
    if (len(selected) != 1 or selected[0].get("ok") is not True or selected[0].get("energy_repeat_valid_n") != 3
            or selected[0].get("energy_calibration_verified") is not True
            or selected[0].get("environment_observation_status") != "pass" or counts["valid_repeats"] != 3):
        raise ValueError("Native Resume lacks its normal three-slot energy import")
    _verify_a4_energy_row(prepared, selected[0], imported["command_contract_sha256"])
    before = _read(output / "energy_dispatch.json").get("start_accounting_before") or {}
    state = _read(output / "PERFORMANCE_RESUME.json")
    attempts = _read(output / "DISPATCH_ATTEMPTS.json").get("attempts") or []
    for value in (before.get("collector_starts"), before.get("valid_repeats"), state.get("dispatch_count_before")):
        if type(value) is not int or value < 0:
            raise ValueError("Native Resume physical-start accounting is missing")
    new_captures = counts["collector_starts"] - before["collector_starts"]
    new_performance = len(attempts) - state["dispatch_count_before"]
    if (new_captures < 0 or new_performance < 0 or before["valid_repeats"] > 3
            or any(type(result.get(name)) is not int for name in (
                "new_capture_count", "new_performance_processes", "retained_valid_energy_repeats"))
            or result.get("new_capture_count") != new_captures
            or result.get("new_performance_processes") != new_performance
            or result.get("retained_valid_energy_repeats") != before["valid_repeats"]
            or result.get("fresh_hardware_executed") is not bool(new_captures or new_performance)):
        raise ValueError("Native Resume result relabels retained slots as new physical starts")
    return True


def _run_native(prepared: Mapping, context: Mapping) -> dict:
    """Dispatch the finite selection through normal ownership and product APIs."""
    if context.get("execute_allowed") is not True:
        raise ValueError("Native acceptance hardware dispatch is not enabled")
    if _functional_mode(prepared) or _functional_mode(context):
        raise ValueError("Fresh Native functional dispatch is unsupported; validate existing captured Native outputs with the normal numeric validators")
    for name in ("source_plan", "source_raw_report", "source_validation"):
        if _sha(prepared[name]) != prepared[name + "_sha256"]:
            raise ValueError("Native acceptance source changed after preparation: " + name)
    target_binding = _verify_current_target(prepared["target"], prepared["identity"], context)
    if target_binding != prepared["target_binding"]:
        raise ValueError("Native target registry changed after preparation")
    output = Path(prepared["output"])
    reports, normalized, attempts = [], [], []
    resumed = context.get("resume") is True and prepared.get("deepx_runtime_mode") == "A4"
    slots = _same_run_performance_slots(prepared, recover_completed=True) if resumed else {"slots": {}, "attempts": []}
    normalized = [slots["slots"][repeat] for repeat in sorted(slots["slots"])]
    reports = [Path(proof["report"]) for proof in normalized]
    attempts = slots["attempts"]
    preparation_attempts = slots.get("preparation_attempts", [])
    previous_dispatch_count = len(attempts)
    _write(output / "PERFORMANCE_RESUME.json", {"resume": resumed, "dispatch_count_before": previous_dispatch_count,
            "retained_logical_repeats": sorted(slots["slots"]), "run_dir": str(Path(context["run_dir"]).resolve())})
    reference = prepared.get("fresh_performance_reference")
    retained = _verify_performance_reference(prepared, context, reference) if reference else None
    started = output / "production-start.json"
    if not resumed or not started.is_file():
        with started.open("x", encoding="utf-8") as stream:
            json.dump({"pid": os.getpid(), "started_at": time.time(), "selection": prepared["identity"]}, stream)
    if retained is not None:
        _write(output / "DISPATCH_ATTEMPTS.json", {"attempts": [], "preparation_attempts": preparation_attempts})
        _write(output / "PERFORMANCE_REUSE.json", retained["proof"])
        reports = [Path(path) for path in retained["reports"]]
        normalized = retained["normalized"]
        for proof, leaf, report in zip(normalized, retained["proof"]["repeats"], reports):
            proof.update(runtime_instance_id=leaf["runtime_instance_id"], repeat=leaf["repeat"], report=str(report))
    elif len(slots["slots"]) < 3:
        from .hardware_acceptance_numerical import _owned_dispatch_closure, _owned_preparation_closure
        transport = _transport(prepared, context)
        diagnostics = Path(getattr(transport, "diagnostics_dir", output / "diagnostics"))
        owner = {"run_id": Path(context["run_dir"]).name, "purpose": "hardware_acceptance_native"}
        with ExitStack() as stack:
            physical = stack.enter_context(EvaluationRunLock.for_resource(physical_dut_key(prepared["target"]), owner=owner))
            sdk_completion_uncertain = False
            def fence_cleanup():
                if context["remote_process_registry"].active_count() or sdk_completion_uncertain:
                    physical.commit_quarantine_fence({"reason": "hardware_acceptance_native_sdk_completion_unproven" if sdk_completion_uncertain
                                                       else "hardware_acceptance_native_cleanup_unproven", "evidence": str(output)})
            stack.callback(fence_cleanup)
            with bind_workflow_resource_options(controller_setup_id=prepared["identity"]["setup_id"],
                                                controller_attempt_dir=str(output)):
                if prepared.get("deepx_runtime_mode") == "A4":
                    previous_diagnostics = {str(path.resolve()) for path in diagnostics.glob("ssh_*.json")}
                    deployment = {"phase": "pre_worker_deployment", "worker_dispatched": False,
                                  "started_at": time.time()}
                    preparation_attempts.append(deployment)
                    _write(output / "DISPATCH_ATTEMPTS.json", {"attempts": attempts, "preparation_attempts": preparation_attempts})
                    try:
                        ensure_current_deepx_runtime(prepared, context)
                    except Exception as exc:
                        closure = _owned_preparation_closure(transport, context["remote_process_registry"], previous_diagnostics)
                        deployment.update(completed_at=time.time(), status="failed", error=f"{type(exc).__name__}: {exc}",
                                          owned_preparation_closure=closure)
                        sdk_completion_uncertain = not closure["proven"]
                        _write(output / "DISPATCH_ATTEMPTS.json", {"attempts": attempts, "preparation_attempts": preparation_attempts})
                        raise
                    deployment.update(completed_at=time.time(), status="completed")
                    _write(output / "DISPATCH_ATTEMPTS.json", {"attempts": attempts, "preparation_attempts": preparation_attempts})
                else:
                    ensure_current_deepx_runtime(prepared, context)
            for source_item in prepared["runs"]:
                repeat = source_item["repeat"]
                if repeat in slots["slots"]:
                    continue
                prior_attempts = [attempt for attempt in attempts if attempt.get("repeat") == repeat]
                prior_items = _performance_attempt_items(source_item, output)
                existing_items = [(item, suffix) for item, suffix in prior_items
                                  if (output / (f"rep{int(repeat):02d}" + suffix)).exists()]
                if (prior_attempts or existing_items) and not resumed:
                    raise ValueError("Native performance attempts already exist; use the normal Resume action")
                if prior_attempts:
                    latest = prior_attempts[-1]
                    matches = [(item, suffix) for item, suffix in prior_items if item["argv"] == latest.get("argv")]
                    if len(matches) != 1:
                        raise ValueError("Native latest performance dispatch differs from its original leaf")
                    failed_directory = output / (f"rep{int(repeat):02d}" + matches[0][1])
                    completion = failed_directory / "SDK_COMPLETION.json"
                    safety = _read(completion) if completion.is_file() else {}
                    if (safety.get("device_completion_proven") is not True
                            or (safety.get("owned_dispatch_closure") or {}).get("proven") is not True):
                        raise RuntimeError("Native failed performance completion is unproven; normal recovery required")
                for old_item, old_suffix in existing_items:
                    if any(attempt.get("argv") == old_item["argv"] for attempt in prior_attempts):
                        continue
                    old_preparation = [record for record in preparation_attempts
                                       if record.get("repeat") == repeat and record.get("argv") == old_item["argv"]]
                    if (len(old_preparation) != 1 or old_preparation[0].get("worker_dispatched") is not False
                            or old_preparation[0].get("phase") != "pre_worker_staging"
                            or (old_preparation[0].get("owned_preparation_closure") or {}).get("proven") is not True):
                        raise RuntimeError("Native previous preparation completion is unproven; normal recovery required")
                retry = max((int(suffix[6:]) if suffix else 0 for _, suffix in existing_items), default=-1) + 1
                replacement = retry > 0
                if prepared.get("deepx_runtime_mode") == "A4":
                    _admit_performance_attempt(prepared, context, replacement=replacement)
                item = _performance_retry_item(source_item, retry) if replacement else source_item
                cancel = context.get("cancel_event")
                if cancel is not None and cancel.is_set():
                    raise RuntimeError("Native acceptance cancelled before next repeat")
                directory = output / (f"rep{int(item['repeat']):02d}" + (f"_retry{retry:02d}" if replacement else ""))
                directory.mkdir(parents=True, exist_ok=False)
                with bind_workflow_resource_options(controller_setup_id=prepared["identity"]["setup_id"],
                                                    controller_attempt_dir=str(directory)):
                    deepx_a4 = prepared.get("deepx_runtime_mode") == "A4"
                    previous_diagnostics = {str(path.resolve()) for path in diagnostics.glob("ssh_*.json")}
                    preparation = {"repeat": item["repeat"], "argv": item["argv"], "environment": item["env"],
                                   "started_at": time.time(), "technical_replacement": replacement,
                                   "phase": "pre_worker_staging", "worker_dispatched": False}
                    preparation_attempts.append(preparation)
                    _write(output / "DISPATCH_ATTEMPTS.json", {"attempts": attempts, "preparation_attempts": preparation_attempts})
                    try:
                        _stage_benchmark_set(transport, item, directory)
                    except Exception as exc:
                        closure = (_owned_preparation_closure(transport, context["remote_process_registry"], previous_diagnostics)
                                   if deepx_a4 else {"proven": not context["remote_process_registry"].active_count()})
                        preparation.update(completed_at=time.time(), status="failed", error=f"{type(exc).__name__}: {exc}",
                                           owned_preparation_closure=closure)
                        sdk_completion_uncertain = not closure["proven"]
                        _write(output / "DISPATCH_ATTEMPTS.json", {"attempts": attempts, "preparation_attempts": preparation_attempts})
                        raise
                    preparation.update(completed_at=time.time(), status="completed")
                    command = "cd -- " + shlex.quote(prepared["remote_tool_dir"]) + " && " + shlex.join(item["argv"])
                    attempt = {"repeat": item["repeat"], "argv": item["argv"], "environment": item["env"], "started_at": time.time(),
                               "technical_replacement": replacement}
                    attempts.append(attempt)
                    preparation["worker_dispatched"] = True
                    _write(output / "DISPATCH_ATTEMPTS.json", {"attempts": attempts, "preparation_attempts": preparation_attempts})
                    if deepx_a4:
                        sdk_completion_uncertain = True
                    previous_diagnostics = {str(path.resolve()) for path in diagnostics.glob("ssh_*.json")}
                    rc, text = transport.run(command, timeout=int(item.get("timeout_s") or 900), env=item["env"])
                    attempt.update(returncode=rc, completed_at=time.time(), console=text[-12000:])
                    _write(output / "DISPATCH_ATTEMPTS.json", {"attempts": attempts, "preparation_attempts": preparation_attempts})
                    if rc and not deepx_a4:
                        raise RuntimeError(f"Native acceptance repetition {item['repeat']} failed with rc {rc}")
                    report = _collect_repeat(transport, item, directory)
                raw = _read(report)
                if deepx_a4:
                    safety = _deepx_completion_safety(raw)
                    safety["owned_dispatch_closure"] = _owned_dispatch_closure(
                        transport, context["remote_process_registry"], command, rc, previous_diagnostics)
                    sdk_completion_uncertain = not (safety["device_completion_proven"] and safety["owned_dispatch_closure"]["proven"])
                    _write(directory / "SDK_COMPLETION.json", safety)
                    if sdk_completion_uncertain:
                        raise RuntimeError("Native DeepX SDK completion is unsafe or unproven; DUT quarantined")
                if rc:
                    raise RuntimeError(f"Native acceptance repetition {item['repeat']} failed with rc {rc}; original report preserved")
                proof = _validate_raw(raw, prepared["identity"], prepared["source_contract"],
                    remote_root=_argument(item["argv"], "--root") if prepared["identity"]["case"] == "full" else None,
                    remote_tool_dir=prepared["remote_tool_dir"])
                proof.update(report=str(report), report_sha256=_sha(report), repeat=item["repeat"],
                             runtime_instance_id=proof.get("runtime_instance_id") or raw.get("runtime_instance_id"))
                if prepared.get("deepx_runtime_mode") is not None:
                    options = (proof.get("contract") or {}).get("runtime_options") or {}
                    if options.get("deepx_runtime_mode") != prepared["deepx_runtime_mode"]:
                        raise ValueError("fresh DeepX runtime mode differs from the saved acceptance profile")
                _write(directory / "VALIDATED.json", proof)
                normalized.append(proof)
                reports.append(report)
    normalized.sort(key=lambda proof: proof["repeat"])
    reports = [Path(proof["report"]) for proof in normalized]
    runtime_ids = [row.get("runtime_instance_id") for row in normalized]
    if not all(runtime_ids) or len(set(runtime_ids)) != 3:
        raise ValueError("Native repetitions did not prove three independent runtime instances")
    summary = _normal_performance_import(prepared, reports, context)
    if retained is not None:
        # Recheck the closed originals after the ordinary reimport and before
        # any new collector admission. Reference changes never trigger replay.
        _verify_performance_reference(prepared, context, reference)
    energy = _energy_capture(prepared, summary, context)
    reference_comparisons = _reference_energy_pairs(prepared, energy["row"], context)
    new_dispatch_count = len(attempts) - previous_dispatch_count
    result = {"status": "completed", "fresh_hardware_executed": bool(new_dispatch_count or energy["new_capture_count"]),
              "new_performance_processes": new_dispatch_count, "cumulative_performance_processes": len(attempts),
              "retained_fresh_performance_processes": 3 if retained else len(slots["slots"]),
              "same_run_slots_preserved": resumed,
              "retained_valid_energy_repeats": energy.get("retained_valid_repeats", 0),
              "performance_reuse_origin": retained["proof"]["origin"] if retained else None,
              "new_capture_count": energy["new_capture_count"], "valid_repeats": {"performance": 3, "energy": 3},
              "failed_attempts": sum(row["returncode"] != 0 for row in attempts[previous_dispatch_count:]) + energy["failed_attempts"],
              "cumulative_failed_attempts": sum(row["returncode"] != 0 for row in attempts) + energy["start_accounting"].get("failed_attempts", 0),
              "attempts": len(attempts),
              "energy_start_accounting": energy["start_accounting"],
              "phase_energy_admission": energy.get("phase_energy_admission", prepared.get("phase_energy_admission")),
              "previous_energy_failed_attempts": sum((prepared.get("phase_energy_admission") or {}).get("prior_chains", {}).values()),
              "fps_median": statistics.median(row["fps"] for row in normalized),
              "output_validation": "normal_completed_task_and_output_contract", "performance_summary": str(summary),
              "energy": energy["row"], "normal_projection": True,
              "full_recaptures": energy["new_capture_count"] if prepared["identity"]["backend"] == "native_full_deepx" else 0,
              "quality_dataset_reexecutions": 0, "full_baseline_references": prepared["full_baseline_references"],
              "full_reference_compatibility": prepared["full_baseline_proofs"],
              "full_reference_comparisons": reference_comparisons,
              "missing_full_energy_comparisons": reference_comparisons["missing_full_energy_comparisons"],
              "evidence_paths": [str(started), str(output / "DISPATCH_ATTEMPTS.json"), str(summary), energy["energy_result"], energy["import"],
                                 str(output / "FULL_REFERENCE_COMPARISONS.json")]}
    if retained is not None:
        result["evidence_paths"].append(str(output / "PERFORMANCE_REUSE.json"))
    if energy.get("capture_admission"):
        result["evidence_paths"].append(energy["capture_admission"])
    _write(output / "RESULT.json", result)
    return result


def run_native(prepared: Mapping, context: Mapping) -> dict:
    """Keep actual starts and valid slots visible after a controlled failure."""
    output = Path(prepared["output"])
    dispatch = output / "DISPATCH_ATTEMPTS.json"
    resumed = context.get("resume") is True and prepared.get("deepx_runtime_mode") == "A4"
    before_dispatches = len(_read(dispatch).get("attempts") or []) if resumed and dispatch.is_file() else 0
    before_energy = _budget_counts(prepared, context) if resumed and context.get("budget_file") else {
        "collector_starts": 0, "failed_attempts": 0}
    try:
        with bind_workflow_resource_options(controller_setup_id=(prepared.get("identity") or {}).get("setup_id"),
                                            controller_attempt_dir=str(prepared["output"])):
            return _run_native(prepared, context)
    except Exception as exc:
        output = Path(prepared["output"])
        if not (output / "production-start.json").is_file():
            raise
        attempts_path = output / "DISPATCH_ATTEMPTS.json"
        attempts = _read(attempts_path).get("attempts", []) if attempts_path.is_file() else []
        if prepared.get("deepx_runtime_mode") == "A4":
            try:
                valid = _same_run_performance_slots(prepared)["slots"]
            except Exception:
                valid = {}
        else:
            valid = list(output.glob("rep*/VALIDATED.json"))
        retained_count, retained_origin = 0, None
        reuse_path = output / "PERFORMANCE_REUSE.json"
        if reuse_path.is_file() and prepared.get("fresh_performance_reference"):
            try:
                proof = _verify_performance_reference(prepared, context, prepared["fresh_performance_reference"])["proof"]
                if _read(reuse_path) == proof:
                    retained_count, retained_origin = 3, proof["origin"]
            except Exception:
                pass  # A changed reference cannot retain accepted completion.
        counts = _budget_counts(prepared, context)
        captures = counts["collector_starts"] - before_energy["collector_starts"]
        valid_energy = counts["valid_repeats"]
        # A started but unresolved transport call remains an attempted dispatch;
        # it is never reported as proof of a completed hardware run.
        result = {"status": "failed", "error": f"{type(exc).__name__}: {exc}",
                  "fresh_hardware_executed": True if valid or captures else None if attempts else False,
                  "fresh_hardware_execution_unresolved": bool(attempts and not valid),
                  "new_performance_processes": len(attempts) - before_dispatches, "new_capture_count": captures,
                  "cumulative_performance_processes": len(attempts),
                  "retained_fresh_performance_processes": retained_count, "performance_reuse_origin": retained_origin,
                  "valid_repeats": {"performance": len(valid) + retained_count, "energy": valid_energy},
                  "failed_attempts": sum(item.get("returncode", 1) != 0 for item in attempts[before_dispatches:])
                                      + counts["failed_attempts"] - before_energy["failed_attempts"], "energy_start_accounting": counts,
                  "phase_energy_admission": prepared.get("phase_energy_admission"),
                  "previous_energy_failed_attempts": sum((prepared.get("phase_energy_admission") or {}).get("prior_chains", {}).values()),
                  "evidence_paths": [str(path) for path in output.rglob("*.json")],
                  "full_recaptures": captures if prepared["identity"]["backend"] == "native_full_deepx" else 0,
                  "quality_dataset_reexecutions": 0}
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
