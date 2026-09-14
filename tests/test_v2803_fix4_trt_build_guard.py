from __future__ import annotations

import ast
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from types import SimpleNamespace

import pytest

from onnx_splitpoint_tool.benchmark.remote_run import _remote_trt_build_guard_shell
from onnx_splitpoint_tool.runners.native_split_quality_runtime import trt_build_forbidden
from onnx_splitpoint_tool.workflow.execution_binding import _remote_args_from_options
from test_v27920_trt_role_cache import _runner, _write_valid_entry


ROOT = Path(__file__).resolve().parents[1]
ENV = "ONNX_SPLITPOINT_TRT_BUILD_GUARD"


def _args(*, cold=None, strict=True, enabled=True, default="warm"):
    policy = {"enabled": enabled, "strict": strict, "default_expectation": default}
    if cold:
        policy["expected_cold"] = cold
    args = _remote_args_from_options(
        SimpleNamespace(), {"artifact_cache_preflight": policy},
        model_id="yolo11l", model_task="detection",
    )
    args.quality_evidence_setup_id = "orin_nx_hailo10_01"
    return args


def _bind(monkeypatch, args=None, case="b062"):
    args = args or _args()
    # Parse the exact payload emitted by the production shell builder.
    shell = _remote_trt_build_guard_shell(args)
    assignment = shlex.split(shell)[1]
    monkeypatch.setenv(ENV, assignment.split("=", 1)[1])
    monkeypatch.setenv("ONNX_SPLITPOINT_TRT_BUILD_CASE", case)
    return args


@pytest.mark.parametrize("role", ["full", "part1", "part2"])
def test_strict_cold_generated_runner_stops_before_compiler_or_engine_load(tmp_path, monkeypatch, role):
    runner = _runner(tmp_path, monkeypatch)
    _bind(monkeypatch)
    model = tmp_path / f"{role}.onnx"
    model.write_bytes(b"diagnostic-model-source")
    calls = []

    def forbidden(*_a, **_k):
        calls.append(True)
        raise AssertionError("compiler/load entry called")

    monkeypatch.setattr(runner.NativeTRTSession, "_build_engine", forbidden)
    monkeypatch.setattr(runner.NativeTRTSession, "_load_engine", forbidden)
    with pytest.raises(RuntimeError, match="cache_miss_blocked:artifact_policy=strict_warm_cache"):
        runner.NativeTRTSession(role, model, cache_root=tmp_path / "cache", allow_build=True)
    assert calls == []
    assert not list((tmp_path / "cache").rglob("*.engine"))


def test_valid_warm_engine_and_receipt_are_reused_unchanged(tmp_path, monkeypatch):
    runner = _runner(tmp_path, monkeypatch)
    _bind(monkeypatch)
    model = tmp_path / "part2.onnx"
    model.write_bytes(b"same-model")
    cache = tmp_path / "cache"
    engine = runner._native_trt_engine_path("part2", model, "fp16", cache)
    _write_valid_entry(runner, model, engine)
    preserved = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in cache.rglob("*") if p.is_file()}
    monkeypatch.setattr(runner.NativeTRTSession, "_build_engine", lambda *_a, **_k: pytest.fail("warm engine compiled"))
    loads = []
    monkeypatch.setattr(runner.NativeTRTSession, "_load_engine", lambda *_a, **_k: loads.append(True))
    session = runner.NativeTRTSession("part2", model, cache_root=cache, allow_build=True)
    assert session.build_info["cache_hit"] is True
    assert session.build_info["build_disabled"] is True
    assert loads == [True]
    assert preserved == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in cache.rglob("*") if p.is_file()}


@pytest.mark.parametrize("strict,enabled,default", [(False, True, "warm"), (True, False, "warm"), (True, True, "unspecified")])
def test_ordinary_build_missing_is_still_reached(tmp_path, monkeypatch, strict, enabled, default):
    runner = _runner(tmp_path, monkeypatch)
    _bind(monkeypatch, _args(strict=strict, enabled=enabled, default=default))
    model = tmp_path / "part2.onnx"
    model.write_bytes(b"missing-engine-normal-policy")
    calls = []

    def compiler(*_a, **_k):
        calls.append(True)
        raise RuntimeError("COMPILER_REACHED")

    monkeypatch.setattr(runner.NativeTRTSession, "_build_engine", compiler)
    with pytest.raises(RuntimeError, match="COMPILER_REACHED"):
        runner.NativeTRTSession("part2", model, cache_root=tmp_path / "cache")
    assert calls == [True]


@pytest.mark.parametrize("declaration,role,case,blocked", [
    ("yolo11l:trt_p2:b062", "part2", "b062", False),
    ("yolo11l:trt_p2:b062", "part1", "b062", True),
    ("yolo11l:trt_p2:b062", "part2", "b063", True),
    ("mobilenet_v3_large:trt_p2:b062", "part2", "b062", True),
    ("yolo11l:trt_p2:orin_nx_hailo10_01/b062", "part2", "b062", False),
    ("yolo11l:trt_p2:orin_nx_hailo8_01/b062", "part2", "b062", True),
    ("yolo11l:trt_full:full", "full", "b062", False),
    ("yolo11l:*:*", "full", "b062", False),
])
def test_expected_cold_exceptions_keep_exact_model_setup_role_case_scope(monkeypatch, declaration, role, case, blocked):
    _bind(monkeypatch, _args(cold=[declaration]), case=case)
    assert trt_build_forbidden(role) is blocked


def test_low_level_guard_survives_constructor_bypass(tmp_path, monkeypatch):
    runner = _runner(tmp_path, monkeypatch)
    _bind(monkeypatch)
    monkeypatch.setattr(runner, "_find_trtexec_local", lambda: pytest.fail("compiler discovery"))
    native = runner.NativeTRTSession.__new__(runner.NativeTRTSession)
    native.kind = "part2"
    # Intentionally no paths: guard must precede every filesystem mutation.
    with pytest.raises(RuntimeError, match="strict_warm_cache"):
        native._build_engine(workspace_mb=1, timeout_s=1)


def test_implicit_ort_compiler_and_native_fallback_are_blocked(tmp_path, monkeypatch):
    runner = _runner(tmp_path, monkeypatch)
    _bind(monkeypatch)
    monkeypatch.setattr(runner.ort, "InferenceSession", lambda *_a, **_k: pytest.fail("implicit ORT compiler"), raising=False)
    with pytest.raises(RuntimeError, match="compiler=onnxruntime_tensorrt_ep"):
        runner._create_session("part2_tensorrt", tmp_path / "missing.onnx", ["TensorrtExecutionProvider"], [], None)
    # Execute the production nested fallback function with a deliberately cold
    # Native engine; it must never enter the ORT fallback after a cache block.
    tree = ast.parse(Path(runner.__file__).read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_get_ort_session")
    ns = dict(vars(runner))
    ns.update(ort_sessions={}, ort_session_infos={}, base_dir=tmp_path,
              args=SimpleNamespace(trt_cache_dir=str(tmp_path / "cache")),
              _want_native_trt=lambda *_a: True,
              _pick_providers=lambda *_a: pytest.fail("Native miss fell back to ORT"))
    exec(compile(ast.Module(body=[node], type_ignores=[]), runner.__file__, "exec"), ns)
    model = tmp_path / "part2.onnx"
    model.write_bytes(b"cold-no-fallback")
    with pytest.raises(RuntimeError, match="strict_warm_cache"):
        ns["_get_ort_session"]("part2", model, "tensorrt")


def test_actual_remote_policy_shell_reaches_generated_child_compiler_fence(tmp_path):
    from onnx_splitpoint_tool.split_export_runners import write_runner_skeleton_onnxruntime
    case = tmp_path / "b062"
    case.mkdir()
    write_runner_skeleton_onnxruntime(str(case), target="cpu")
    child = tmp_path / "child.py"
    child.write_text('''import importlib.util, sys, types
from pathlib import Path
sys.path.insert(0, sys.argv[1])
sys.modules["onnxruntime"] = types.ModuleType("onnxruntime")
spec = importlib.util.spec_from_file_location("generated", sys.argv[2])
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
def compiler(*a, **k):
    print("COMPILER_REACHED")
    raise RuntimeError("unexpected compiler")
module._find_trtexec_local = compiler
module.subprocess.run = compiler
obj = module.NativeTRTSession.__new__(module.NativeTRTSession)
obj.kind = "part2"
try:
    obj._build_engine(workspace_mb=1, timeout_s=1)
except RuntimeError as exc:
    print(str(exc))
    raise SystemExit(0 if "strict_warm_cache" in str(exc) else 9)
raise SystemExit(8)
''')
    shell = _remote_trt_build_guard_shell(_args()) + "\nexport ONNX_SPLITPOINT_TRT_BUILD_CASE=b062\n" + shlex.join([sys.executable, str(child), str(ROOT), str(case / "run_split_onnxruntime.py")])
    cp = subprocess.run(["bash", "-c", shell], text=True, capture_output=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    assert cp.returncode == 0, cp.stdout + cp.stderr
    assert "strict_warm_cache" in cp.stdout
    assert "COMPILER_REACHED" not in cp.stdout


def test_direct_builder_final_fence_blocks_only_build_commands(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("fix4_builder", ROOT / "scripts/native_trt_from_benchmarkset.py")
    builder = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, builder)
    spec.loader.exec_module(builder)
    _bind(monkeypatch)
    calls = []
    monkeypatch.setattr(builder.subprocess, "run", lambda command, **_k: calls.append(command) or SimpleNamespace(returncode=0, stdout=""))
    blocked = builder._run(["trtexec", "--onnx=missing.onnx", "--saveEngine=a.engine"], cwd=tmp_path, log_path=tmp_path / "build.log", dry_run=False, artifact_role="part2", case_id="b062")
    assert blocked["returncode"] == 78 and calls == []
    assert blocked["compiler_dispatched"] is False
    reused = builder._run(["trtexec", "--loadEngine=a.engine"], cwd=tmp_path, log_path=tmp_path / "load.log", dry_run=False)
    assert reused["returncode"] == 0
    assert calls == [["trtexec", "--loadEngine=a.engine"]]


@pytest.mark.parametrize("item", ["b062", "b062:generic", "orin_nx_hailo10_01/b062", "orin_nx_hailo10_01/b062:generic", "orin_nx_hailo8_01/b062:generic"])
@pytest.mark.parametrize("generic", [False, True])
@pytest.mark.parametrize("expectation", ["warm", "cold"])
def test_runtime_generic_and_native_expectations_equal_preflight(monkeypatch, item, generic, expectation):
    from onnx_splitpoint_tool.workflow.artifact_cache_preflight import _expectation_for, resolve_artifact_cache_preflight_policy
    raw = {"enabled": True, "strict": True, "default_expectation": "cold" if expectation == "warm" else "warm",
           "expected_" + expectation: [{"model_id": "yolo11l", "role": "trt_p2", "item_id": item}]}
    profile = {"artifact_cache_preflight": raw}
    args = _remote_args_from_options(SimpleNamespace(), profile, model_id="yolo11l")
    args.quality_evidence_setup_id = "orin_nx_hailo10_01"
    _bind(monkeypatch, args)
    policy = resolve_artifact_cache_preflight_policy(profile)
    report_item = "orin_nx_hailo10_01/b062" + (":generic" if generic else "")
    expected = _expectation_for(policy, model_id="yolo11l", role="trt_p2", item_id=report_item)
    assert trt_build_forbidden("part2", generic_part2=generic) == (expected == "warm")


def test_generic_engine_honors_exact_cold_exception_without_authorizing_native(tmp_path, monkeypatch):
    runner = _runner(tmp_path, monkeypatch)
    _bind(monkeypatch, _args(cold=[{"model_id": "yolo11l", "role": "trt_p2", "item_id": "orin_nx_hailo10_01/b062:generic"}]))
    assert trt_build_forbidden("part2", generic_part2=False) is True
    model = tmp_path / "part2.onnx"
    model.write_bytes(b"explicitly-authorized-generic-cold")
    calls = []
    def compiler(*_a, **_k):
        calls.append(True)
        raise RuntimeError("COMPILER_REACHED")
    monkeypatch.setattr(runner.NativeTRTSession, "_build_engine", compiler)
    with pytest.raises(RuntimeError, match="COMPILER_REACHED"):
        runner.NativeTRTSession("part2", model, cache_root=tmp_path / "cache")
    assert calls == [True]


def test_warm_native_engine_can_seal_new_binding_without_compiling(tmp_path, monkeypatch):
    from onnx_splitpoint_tool.runners import native_split_quality_runtime as runtime
    from test_v269f_native_split_quality_runtime_e2e import test_prepare_native_split_quality_binding_hailo_end_to_end as seed_and_verify_existing_cache
    # Existing end-to-end fixture writes real receipt-bound artifact bytes and
    # exercises ordinary cold+warm behavior. Undo only its intentional final
    # corruption, then remove every old binding to model this run's missing
    # completion evidence while keeping its warm engine and receipt.
    seed_and_verify_existing_cache(tmp_path, monkeypatch)
    cache = tmp_path / "persistent_cache"
    original = tmp_path / "benchmark_set/b038/hailo/hailo8/part1/compiled.hef"
    for part1 in cache.rglob("part1.hef"):
        part1.write_bytes(original.read_bytes())
    for binding in cache.rglob("native_split_quality_binding.json"):
        binding.unlink()
    (tmp_path / "collected/native_split_quality_binding.json").unlink()
    protected = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in cache.rglob("*") if p.is_file() and p.name in {"part2.engine", "engine_build_receipt.json"}}
    args = _args()
    args.trt_build_guard["model_id"] = "yolo26s"
    args.quality_evidence_setup_id = "hailo8_setup"
    _bind(monkeypatch, args, case="b038")
    verifying_builder = runtime.subprocess.run
    commands = []
    def only_verify(command, **kwargs):
        assert "--no-build" in command, "strict warm preparation requested a compiler"
        commands.append(command)
        return verifying_builder(command, **kwargs)
    monkeypatch.setattr(runtime.subprocess, "run", only_verify)
    out = tmp_path / "new-run/native_split_quality_binding.json"
    result = runtime.prepare_native_split_quality_binding(
        benchmark_set=tmp_path / "benchmark_set", case_id="b038", model_id="yolo26s", setup_id="hailo8_setup", backend="hailo8_to_trt",
        eval_run_id="new-strict-warm-run", source_run_id="hailo8_to_trt", cache_root=cache, output_path=out, workspace_mb=2048, timeout_s=90,
    )
    assert len(commands) == 1
    assert out.is_file() and result["binding"]["eval_run_id"] == "new-strict-warm-run"
    assert "--no-build" in result["binding"]["producer_command"]
    assert protected == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in protected}


def test_unresolved_full_source_specific_policy_fails_closed(monkeypatch):
    from onnx_splitpoint_tool.workflow.artifact_cache_preflight import resolve_artifact_cache_preflight_policy
    policy = resolve_artifact_cache_preflight_policy({"artifact_cache_preflight": {
        "enabled": True, "strict": True, "default_expectation": "cold",
        "expected_warm": [{"model_id": "yolo11l", "role": "trt_full", "item_id": "full:012345678abc"}],
    }})
    args = _args()
    args.trt_build_guard["policy"] = policy
    _bind(monkeypatch, args)
    with pytest.raises(RuntimeError, match="full_source_scope_unresolved"):
        trt_build_forbidden("full")
