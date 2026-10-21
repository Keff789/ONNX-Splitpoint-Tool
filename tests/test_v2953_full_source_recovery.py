"""Bounded original Full source/trace release ground, with no worker execution."""
import copy
import hashlib
import io
import json
from pathlib import Path
import shlex
import tarfile
from types import SimpleNamespace
import zipfile

import pytest

from onnx_splitpoint_tool.native_deepx_async import NativeExecutionLifecycle
from onnx_splitpoint_tool.remote import process_lease, resource_recovery as recovery
from onnx_splitpoint_tool.workflow.run_control import WorkflowRunCleanupQuarantineError


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def digest(data):
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def original(tmp_path):
    run = tmp_path / "arbitrary_normal_run"
    base = run / "native/model__setup__full"
    directory = base / "numerical"
    benchmark = "/private/inputs/model/benchmark_set"
    outer = b"def project(step, prepared_feed, changed, result):\n if step.get('timed_out') or not prepared_feed or not changed:\n  result['unsafe_runtime_state'] = True\n"
    suite = b"def main():\n default_plan = root / 'benchmark_plan.json'\n runs = [{'id': f'ort_{args.provider}', 'type': 'onnxruntime', 'provider': args.provider, 'image_scale': args.image_scale}]\n name='__BENCH_JSON__'\n"
    case = b'''PROVENANCE="__GENERATED_SOURCE_FILES__"
RELEASE="__GENERATED_RELEASE_IDENTITY__"
MANIFEST="__MANIFEST_FILENAME__"
ENDPOINT="__VENDORED_ENDPOINT_ATTESTOR_SHA256__"
PROVIDER="__DEFAULT_PROVIDER__"
def _adapt_tensor(value):
 return value
def _pick_providers(provider, available):
 return ['CPUExecutionProvider']
def _create_session(name, path, *options):
 return ort.InferenceSession(path)
def main():
 providers = _pick_providers('cpu', avail)
 sess_p1, info_p1 = _create_session('part1', p1_path, providers, provider_options, sess_options)
 worker = DeepXSession('model')
'''
    adapter = b"def _adapt_tensor(value):\n return value\n"
    endpoint = b"# original endpoint declaration\n"
    release = b"VERSION='fixture'\nBUILD_ID='fixture-build'\n"
    dependency = b"# normal import contains no SDK construction\n"
    archive = tmp_path / "normal_release.zip"
    members = {"scripts/native_full_baseline_eval_runner.py": outer,
        "onnx_splitpoint_tool/resources/templates/benchmark_suite.py.txt": suite,
        "onnx_splitpoint_tool/resources/templates/run_split_onnxruntime.py.txt": case,
        "onnx_splitpoint_tool/runners/backends/hailo_backend.py": adapter,
        "onnx_splitpoint_tool/release_identity.py": release,
        "onnx_splitpoint_tool/native_output_endpoint.py": endpoint,
        "onnx_splitpoint_tool/runners/native_split_quality_runtime.py": dependency}
    with zipfile.ZipFile(archive, "w") as bundle:
        for name, data in members.items(): bundle.writestr("normal-release/" + name, data)
    provenance = {"template": {"path": "onnx_splitpoint_tool/resources/templates/run_split_onnxruntime.py.txt", "source_sha256": digest(case)},
        "package_adapter": {"path": "onnx_splitpoint_tool/runners/backends/hailo_backend.py", "source_sha256": digest(adapter)}}
    generated = case.decode().replace('"__GENERATED_SOURCE_FILES__"', repr(provenance)).replace('"__GENERATED_RELEASE_IDENTITY__"', repr({"version": "fixture", "build_id": "fixture-build"}))
    generated = generated.replace("__MANIFEST_FILENAME__", "split_manifest.json").replace("__VENDORED_ENDPOINT_ATTESTOR_SHA256__", digest(endpoint)).replace("__DEFAULT_PROVIDER__", "auto")
    files = {"benchmark_suite.py": suite.replace(b"__BENCH_JSON__", b"benchmark_set.json"),
        "b004/run_split_onnxruntime.py": generated.encode(), "splitpoint_runners/native_split_quality_runtime.py": dependency}
    for name, data in files.items():
        path = directory / "software" / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data)
    bench = {"cases": [{"case_dir": "b004"}, {"case_dir": "b031"}], "plan": {"runs": [{"id": "deepx_m1_full", "type": "deepx", "backend": "deepx_m1", "provider": "deepx_m1", "variants": ["full"]}]}}
    write(directory / "software/benchmark_set.json", bench)
    copied = {"destination": benchmark, "model_builds": 0, "historical_results_copied": False,
        "files": [{"path": "benchmark_set.json", "sha256": digest((directory / "software/benchmark_set.json").read_bytes())},
                  {"path": "deepx/deepx_m1/full/model.dxnn", "sha256": "b" * 64},
                  {"path": "models/model.onnx", "sha256": "c" * 64}]}
    write(directory / "benchmark_staging.json", {"returncode": 0, "output": "FULL_STAGING=" + json.dumps(copied)})
    cmd = ["/venv/python", "/tool/scripts/run_benchmark_suite_from_set.py", "--benchmark-set", benchmark,
        "--", "--run-id", "deepx_m1_full", "--deepx-runtime-mode", "A4", "--deepx-numerical-only"]
    import ast
    main = next(node for node in ast.parse(generated).body if isinstance(node, ast.FunctionDef) and node.name == "main")
    line = next(node.lineno for node in ast.walk(main) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "_create_session")
    output = ("[suite-wrapper] cmd: " + shlex.join([cmd[0], benchmark + "/benchmark_suite.py", *cmd[cmd.index("--") + 1:]]) + "\n"
        "[ort_auto] [1/2] Running b004 (provider=auto)\n"
        "Using mixed backends: default=auto, full=auto, stage1=auto, stage2=auto (ORT baseline: ['CPUExecutionProvider'])\n"
        "[init] session 'full' ready in 0.1s | providers in use: ['CPUExecutionProvider']\n"
        "Traceback (most recent call last):\n"
        f'  File "{benchmark}/b004/run_split_onnxruntime.py", line {line}, in main\n'
        "    sess_p1, info_p1 = _create_session(\nNoSuchFile: missing part1 ONNX\n"
        "[warn] case failed: b004 (rc=1)\n[ort_auto] [2/2] Running b031 (provider=auto)\n"
        f"[warn] missing runner: {benchmark}/b031/run_split_onnxruntime.py\nDone.\n")
    lifecycle = NativeExecutionLifecycle(); lifecycle.construction_enter(); lifecycle.construction_return(); lifecycle.inference_enter(); lifecycle.inference_return()
    semantic = {"ok": True, "backend": "native_full_deepx", "model": "model", "setup_id": "setup", "image_sha256": "f" * 64,
        "serial_reference_calls_started": 1, "serial_reference_calls_completed": 1, "deepx_execution_lifecycle": lifecycle.report()}
    write(directory / "download/outputs/native_full_outputs/one/native_full_semantic_dump.json", semantic)
    preparation = {**semantic, "step": {"rc": 0, "returncode": 0, "stdout": json.dumps(semantic)}}
    row = {"backend": "native_full_deepx", "deepx_runtime_mode": "A4", "unsafe_runtime_state": True, "unsafe_runtime_reasons": [],
        **{key: {} for key in ("deepx_prepared_feed_benchmark", "deepx_async_observation", "numerical_validation", "failed_phase_counts")},
        **{key: None for key in ("pending", "unreleased", "submitted_count", "completed_frames", "completed_work_units")},
        "repetition_count_attempted": 1, "repetition_count_requested": 1, "repetition_count_valid": 0,
        "steps": [{"cmd": cmd, "rc": 0, "returncode": 0, "stdout": output}], "preparation_evidence": preparation,
        "full_command_contract": {"runner_sha256": digest(outer)}}
    prepared = {"identity": {"backend": "native_full_deepx", "model": "model", "setup_id": "setup", "case": "full"},
        "source_contract": {"input_image_sha256": "f" * 64, "artifacts": {"dxnn": {"sha256": "b" * 64}, "source_onnx": {"sha256": "c" * 64}}}}
    return SimpleNamespace(run=run, base=base, directory=directory, archive=archive, files=files, prepared=prepared, raw={"rows": [row]}, benchmark=benchmark)


def bundle(original, *, duplicate=False):
    path = original.directory / "full_software.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        for name, data in original.files.items():
            item = tarfile.TarInfo(name); item.size = len(data); archive.addfile(item, io.BytesIO(data))
            if duplicate: archive.addfile(item, io.BytesIO(data))
    digest_value = digest(path.read_bytes())
    write(original.directory / "software_staging.json", {"returncode": 0, "bundle": str(path), "bundle_sha256": digest_value})
    remote = original.benchmark + "/full_software.tar.gz"
    command = "test " + digest_value + ' = "$(sha256sum -- ' + remote + " | cut -d ' ' -f 1)\" && tar -xzf " + remote + " -C " + original.benchmark
    diagnostic = original.base / "diagnostics/ssh_software.json"
    write(diagnostic, {"command": command, "returncode": 0, "cleanup_proven": True, "target_id": "setup"})
    (original.run / "evaluation_workflow.log").write_text(f"[scp-upload] {path} -> nx@host:{remote}\n[ssh-result] setup=setup rc=0 · Diagnose: {diagnostic}\n")


def prove(original):
    snapshots = {}
    def read(path): snapshots[path] = path.read_bytes(); return json.loads(snapshots[path])
    return recovery._full_ort_pre_runtime_proof(original.raw, original.prepared, original.directory, original.archive, read, snapshots)


def test_whole_source_positive_trace_keeps_measured_counts_unknown_and_raw_failed(original):
    bundle(original); before = copy.deepcopy(original.raw)
    proof = prove(original)
    assert proof["measured_child_accelerator_entry_proven_absent_by_source_and_trace"] is True
    assert proof["measured_sdk_call_count"] is None and proof["observed_case_count"] == 2
    assert original.raw == before and original.raw["rows"][0]["unsafe_runtime_state"] is True


@pytest.mark.parametrize("fault", ["trace_truncated", "case_missing", "other_provider", "actual_entry", "pending", "unsafe", "unknown_lifecycle", "changed_helper", "extra_startup", "duplicate_tar", "missing_semantic", "changed_semantic", "bad_staging"])
def test_unproven_trace_source_or_sdk_state_never_supplies_a_release_ground(original, fault):
    row = original.raw["rows"][0]
    if fault == "trace_truncated": row["steps"][0]["stdout"] = row["steps"][0]["stdout"][:-6]
    elif fault == "case_missing": row["steps"][0]["stdout"] = row["steps"][0]["stdout"].replace("[ort_auto] [2/2]", "[other] [2/2]")
    elif fault == "other_provider": row["steps"][0]["stdout"] = row["steps"][0]["stdout"].replace("stage1=auto", "stage1=deepx")
    elif fault == "actual_entry": row["serial_sdk_calls_started"] = 1; row["serial_sdk_calls_completed"] = 1
    elif fault == "pending": row["deepx_runtime_observation"] = {"submitted": 1, "wait_completed": 0, "pending": 0, "unreleased": 0}
    elif fault == "unsafe": original.raw["unsafe_device_state"] = True
    elif fault == "unknown_lifecycle": original.raw["deepx_execution_lifecycle"] = {"runtime_constructors_entered": 1, "runtime_constructors_returned": 0}
    elif fault == "changed_helper": original.files["splitpoint_runners/native_split_quality_runtime.py"] += b"engine=InferenceEngine('model')\n"
    elif fault == "extra_startup": original.files["b004/sitecustomize.py"] = b"engine=InferenceEngine('model')\n"
    elif fault == "missing_semantic": next((original.directory / "download").rglob("native_full_semantic_dump.json")).unlink()
    elif fault == "changed_semantic": row["preparation_evidence"]["step"]["stdout"] = "{}"
    for name, data in original.files.items():
        path = original.directory / "software" / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data)
    bundle(original, duplicate=fault == "duplicate_tar")
    if fault == "bad_staging": (original.run / "evaluation_workflow.log").write_text("no original SHA-tested extraction\n")
    with pytest.raises(WorkflowRunCleanupQuarantineError): prove(original)


def test_cli_exposes_optional_bound_source_archive_without_execution(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(recovery, "recover_native_numerical_resource", lambda **kwargs: calls.append(kwargs) or {"status": "reviewed"})
    assert process_lease.process_lease_cli_main(["recover-native-numerical", "--run-dir", "/run", "--prepared", "/run/native/cell/PREPARED.json",
        "--setup-id", "setup", "--operator", "operator", "--evidence-output", "/new/receipt.json", "--source-archive", "/old/release.zip"]) == 0
    assert calls[0]["source_archive"] == "/old/release.zip"
