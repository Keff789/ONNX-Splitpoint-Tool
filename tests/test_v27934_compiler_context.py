"""T34.08–20: exercise real resolver, bounded fake external binaries only."""
import copy
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest

from onnx_splitpoint_tool import hailo_compiler_context as cc


@pytest.fixture
def vendor(tmp_path):
    root = tmp_path / "venv_hailo10"
    (root / "bin").mkdir(parents=True)
    (root / "pyvenv.cfg").write_text("include-system-site-packages = false\n")
    python = root / "bin" / "python"
    python.symlink_to(sys.executable)
    components = root / "lib" / "python3.99" / "site-packages" / "triton" / "backends" / "nvidia"
    (components / "bin").mkdir(parents=True)
    (components / "lib").mkdir()
    libdevice = components / "lib" / "libdevice.10.bc"
    libdevice.write_bytes(b"fixture bitcode")
    ptxas = components / "bin" / "ptxas"
    ptxas.write_text("#!/bin/sh\nexec " + shlex.quote(sys.executable) + " " + shlex.quote(str(components / "ptxas_fixture.py")) + ' "$@"\n')
    ptxas.chmod(0o700)
    (components / "ptxas_fixture.py").write_text('''import os,sys
from pathlib import Path
if '--version' in sys.argv:
    print('fixture ptxas 12.8.93')
    raise SystemExit(0)
if os.environ.get('FIXTURE_UNSUPPORTED'):
    print('Value sm_61 is not defined for option gpu-name',file=sys.stderr)
    raise SystemExit(255)
if not os.environ.get('FIXTURE_EMPTY'):
    Path(sys.argv[sys.argv.index('-o')+1]).write_bytes(b'compiled target object')
''')
    fakebin = tmp_path / "external_bin"
    fakebin.mkdir()
    smi = fakebin / "nvidia-smi"
    smi.write_text("#!/bin/sh\nprintf '%s\\n' \"${FIXTURE_GPU_ROW:-0, GPU-fixture-zero, 6.1}\"\n")
    smi.chmod(0o700)
    env = {"PATH": str(fakebin) + os.pathsep + os.defpath, "LD_LIBRARY_PATH": "/preserved/vendor/libs", "XLA_FLAGS": "--xla_dump_to=/preserved"}
    return {"root": root, "python": python, "components": components, "ptxas": ptxas, "libdevice": libdevice, "env": env}


def resolve(vendor, **kwargs):
    return cc.resolve_hailo_compiler_context(vendor["python"], "hailo10h", job_override="gpu", parent_env=vendor["env"], **kwargs)


@pytest.mark.parametrize("family,expected", [("hailo8", "hailo8"), ("hailo8l", "hailo8"), ("hailo10", "hailo10h"), (" HAILO10H ", "hailo10h")])
def test_family_aliases(family, expected):
    assert cc.normalize_hailo_family(family) == expected


@pytest.mark.parametrize("value", ["auto", "yes", False, 1, [], {}, {"device": False}])
def test_invalid_compute_value(value):
    with pytest.raises(cc.CompilerContextError):
        cc.resolve_compute_selection("hailo10", job_override=value, env={})


def test_alias_conflict_and_equivalent_aliases():
    with pytest.raises(cc.CompilerContextError, match="alias_conflict"):
        cc.normalize_compute_by_family({"hailo10": "cpu", "hailo10h": "gpu"})
    assert cc.normalize_compute_by_family({"hailo10": "cpu", "hailo10h": {"device": "cpu"}}) == {"hailo10h": {"device": "cpu"}}
    assert cc.normalize_compute_by_family(None) == {}


def test_precedence_and_sources():
    family = {"hailo10": {"device": "cpu"}, "hailo8": "cpu"}
    result = cc.resolve_compute_selection("hailo10", job_override="gpu", compute_by_family=family, env={})
    assert (result["device"], result["source"]) == ("gpu", "job_override")
    assert cc.resolve_compute_selection("hailo8", compute_by_family=family, env={})["source"] == "compute_by_family.hailo8"
    assert cc.resolve_compute_selection("hailo10", env={})["source"] == "tool_default"


@pytest.mark.parametrize("env", [{"ONNX_SPLITPOINT_HAILO_COMPUTE": "cpu"}, {"ONNX_SPLITPOINT_HAILO_ALLOW_GPU": "0"}, {"SPLITPOINT_HAILO_ALLOW_GPU": "false"}, {"CUDA_VISIBLE_DEVICES": "-1"}, {"CUDA_VISIBLE_DEVICES": ""}])
def test_explicit_cpu_constraints_are_not_overridden(env):
    with pytest.raises(cc.CompilerContextError, match="conflict"):
        cc.resolve_compute_selection("hailo10", job_override="gpu", env=env)


def test_tool_default_mask_is_not_user_override():
    result = cc.resolve_compute_selection("hailo10h", job_override="gpu", env={"CUDA_VISIBLE_DEVICES": "-1", cc.MASK_SOURCE_ENV: "tool_default"})
    assert result["device"] == "gpu" and result["inherited_cuda_mask_source"] == "tool_default"


def test_global_optin_is_family_local_and_recorded():
    env = {"ONNX_SPLITPOINT_HAILO_ALLOW_GPU": "true"}
    h10 = cc.resolve_compute_selection("hailo10", env=env)
    h8 = cc.resolve_compute_selection("hailo8", env=env)
    assert h10["device"] == "gpu" and "environment:" in h10["source"]
    assert h8["device"] == "cpu" and h8["ignored_legacy"]
    assert cc.resolve_compute_selection("hailo8", job_override="gpu", env=env)["device"] == "gpu"


def test_legacy_conflict_and_invalid_boolean():
    with pytest.raises(cc.CompilerContextError, match="conflict"):
        cc.resolve_compute_selection("hailo10", env={"SPLITPOINT_HAILO_ALLOW_GPU": "0", "ONNX_SPLITPOINT_HAILO_COMPUTE": "gpu"})
    with pytest.raises(cc.CompilerContextError, match="invalid"):
        cc.resolve_compute_selection("hailo10", env={"SPLITPOINT_HAILO_ALLOW_GPU": "maybe"})


def test_environment_mapping_and_job_override():
    env = {"ONNX_SPLITPOINT_HAILO_COMPUTE_BY_FAMILY": '{"hailo10":{"device":"cpu"}}', "ONNX_SPLITPOINT_HAILO_COMPUTE_OVERRIDE": '{"device":"gpu","gpu_selector":"2"}'}
    assert cc.resolve_compute_selection("hailo10", env=env)["gpu_selector"] == "2"


@pytest.mark.parametrize("mask", ["0,1", "MIG-incomplete", "-2", "garbage"])
def test_ambiguous_gpu_mask_rejected(mask):
    with pytest.raises(cc.CompilerContextError, match="selection_invalid"):
        cc.resolve_compute_selection("hailo10", job_override="gpu", env={"CUDA_VISIBLE_DEVICES": mask})


def test_explicit_gpu_different_from_mask_rejected():
    with pytest.raises(cc.CompilerContextError, match="conflict"):
        cc.resolve_compute_selection("hailo10", job_override={"device": "gpu", "gpu_selector": "1"}, env={"CUDA_VISIBLE_DEVICES": "0"})


def test_cpu_does_not_probe_or_discover(vendor, monkeypatch):
    monkeypatch.setattr(cc, "_gpu_target", lambda *a, **k: pytest.fail("CPU launched GPU probe"))
    monkeypatch.setattr(cc, "_locate_components", lambda *a, **k: pytest.fail("CPU looked up components"))
    context = cc.resolve_hailo_compiler_context(vendor["python"], "hailo8", parent_env=vendor["env"])
    assert context["device"] == "cpu"


def test_dynamic_venv_symlink_and_target_probe(vendor):
    context = resolve(vendor)
    assert context["venv_root"] == str(vendor["root"])
    assert context["venv_python"] != str(Path(sys.executable).resolve())
    assert context["target_arch"] == "sm_61"
    assert context["target_probe"]["output_size"] > 0
    assert context["target_probe"]["binary"] == str(vendor["ptxas"])
    assert context["full_cuda_toolkit"] is False


def test_actual_gpu_architecture_is_selected(vendor):
    vendor["env"].update(CUDA_VISIBLE_DEVICES="2", FIXTURE_GPU_ROW="2, GPU-other-device, 8.9")
    context = resolve(vendor)
    assert context["target_arch"] == "sm_89"
    assert ".target sm_50" in context["target_probe"]["source_ptx"]
    assert ".target sm_89" not in context["target_probe"]["source_ptx"]
    assert context["gpu_uuid"] == "GPU-other-device"


def test_different_selected_device_fails(vendor):
    vendor["env"]["CUDA_VISIBLE_DEVICES"] = "2"
    with pytest.raises(cc.CompilerContextError, match="selection_unresolved"):
        resolve(vendor)


@pytest.mark.parametrize("failure", ["FIXTURE_EMPTY", "FIXTURE_UNSUPPORTED"])
def test_target_failures_preserve_original_evidence(vendor, failure):
    vendor["env"][failure] = "1"
    reason = "target_unsupported" if failure == "FIXTURE_UNSUPPORTED" else "target_output_missing"
    with pytest.raises(cc.CompilerContextError, match=reason) as error:
        resolve(vendor)
    assert error.value.details["target_arch"] == "sm_61"
    assert error.value.details["binary"] == str(vendor["ptxas"])
    if failure == "FIXTURE_UNSUPPORTED":
        assert "not defined" in error.value.details["stderr"]


@pytest.mark.parametrize("damage", ["missing_ptxas", "empty_ptxas", "not_executable", "missing_libdevice", "empty_libdevice", "broken_symlink", "foreign_symlink"])
def test_invalid_components_no_system_fallback(vendor, tmp_path, damage):
    ptxas, libdevice = vendor["ptxas"], vendor["libdevice"]
    if damage in {"missing_ptxas", "broken_symlink", "foreign_symlink"}:
        ptxas.unlink()
    if damage == "empty_ptxas":
        ptxas.write_bytes(b"")
    elif damage == "not_executable":
        ptxas.chmod(0o600)
    elif damage == "missing_libdevice":
        libdevice.unlink()
    elif damage == "empty_libdevice":
        libdevice.write_bytes(b"")
    elif damage == "broken_symlink":
        ptxas.symlink_to(tmp_path / "absent")
    elif damage == "foreign_symlink":
        target = tmp_path / "foreign_ptxas"
        target.write_bytes(b"foreign")
        target.chmod(0o700)
        ptxas.symlink_to(target)
    with pytest.raises(cc.CompilerContextError):
        resolve(vendor)


def test_explicit_context_wins_over_ambiguous_discovery(vendor):
    alternative = vendor["root"] / "lib" / "python3.98" / "site-packages" / "triton" / "backends" / "nvidia"
    import shutil
    shutil.copytree(vendor["components"], alternative)
    with pytest.raises(cc.CompilerContextError, match="ambiguous"):
        resolve(vendor)
    context = resolve(vendor, explicit_context={"ptxas_path": str(vendor["ptxas"]), "libdevice_path": str(vendor["libdevice"])})
    assert context["component_source"] == "explicit_context"


def test_explicit_foreign_pair_rejected(vendor, tmp_path):
    lib = tmp_path / "libdevice.10.bc"
    lib.write_bytes(b"foreign")
    with pytest.raises(cc.CompilerContextError, match="foreign_venv"):
        resolve(vendor, explicit_context={"ptxas_path": str(vendor["ptxas"]), "libdevice_path": str(lib)})


@pytest.mark.parametrize("flags", ['--xla_gpu_cuda_data_dir=/old --xla_dump_to=/logs', '--xla_gpu_cuda_data_dir /old --xla_dump_to=/logs', '--xla_gpu_cuda_data_dir=/old --xla_gpu_cuda_data_dir /old --xla_dump_to=/logs'])
def test_exact_xla_root_replaced_preserving_other_flags(flags):
    tokens = shlex.split(cc.replace_xla_cuda_root(flags, "/private/view"))
    assert tokens.count("--xla_gpu_cuda_data_dir=/private/view") == 1
    assert "--xla_dump_to=/logs" in tokens
    assert not any("/old" in token for token in tokens)


@pytest.mark.parametrize("flags", ['--xla_gpu_cuda_data_dir=/a --xla_gpu_cuda_data_dir=/b', '--xla_gpu_cuda_data_dir', '--xla_gpu_cuda_data_dir=', 'garbage', '--xla_gpu_cuda_data_dir="unterminated'])
def test_invalid_xla_tokens_not_swallowed(flags):
    with pytest.raises(cc.CompilerContextError):
        cc.replace_xla_cuda_root(flags, "/view")


def test_prefix_is_not_root_flag():
    result = cc.replace_xla_cuda_root("--xla_gpu_cuda_data_dir_suffix=/keep", "/view")
    assert "--xla_gpu_cuda_data_dir_suffix=/keep" in result


def test_private_view_child_env_lifetime_and_real_forwarder(vendor, tmp_path):
    context = resolve(vendor)
    before = copy.deepcopy(vendor["env"])
    with cc.compiler_child_environment(context, parent_env=vendor["env"], work_dir=tmp_path / "job") as (env, effective):
        view = Path(effective["view_root"])
        assert view.exists()
        assert env["LD_LIBRARY_PATH"] == before["LD_LIBRARY_PATH"]
        assert env["CUDA_VISIBLE_DEVICES"] == "GPU-fixture-zero"
        assert env["TF_NUM_INTEROP_THREADS"] == "2"
        output = tmp_path / "real.cubin"
        result = subprocess.run([str(view / "bin" / "ptxas"), "-arch=sm_61", "-o", str(output)], env=env, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        assert output.read_bytes()
        trace = Path(effective["ptxas_trace_path"])
        records = [json.loads(row) for row in trace.read_text().splitlines()]
        assert len(records) == 1 and records[0]["kind"] == "assembly"
        assert records[0]["binary"] == context["ptxas_path"]
        assert records[0]["target_arches"] == ["sm_61"]
        assert cc.validate_compiler_child_environment(env)["view_root"] == str(view)
    assert not view.exists() and trace.exists()
    assert before == vendor["env"]
    assert "view_root" not in context


def test_private_view_exception_cleanup_and_preserves_user_threads(vendor, tmp_path):
    context = resolve(vendor)
    env = dict(vendor["env"], TF_NUM_INTEROP_THREADS="7", TF_FORCE_GPU_ALLOW_GROWTH="false")
    with pytest.raises(RuntimeError):
        with cc.compiler_child_environment(context, parent_env=env, work_dir=tmp_path / "job") as (child, effective):
            view = Path(effective["view_root"])
            assert " " not in str(view)
            assert child["TF_NUM_INTEROP_THREADS"] == "7"
            assert child["TF_FORCE_GPU_ALLOW_GROWTH"] == "false"
            raise RuntimeError("cancelled")
    assert not view.exists()


def test_late_auto_configure_never_remasks_or_reprobes(vendor, tmp_path, monkeypatch):
    from onnx_splitpoint_tool import cuda_probe
    context = resolve(vendor)
    monkeypatch.setattr(cuda_probe, "probe_cuda_environment", lambda: pytest.fail("late probe"))
    with cc.compiler_child_environment(context, parent_env=vendor["env"], work_dir=tmp_path / "job") as (env, effective):
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        result = cuda_probe.auto_configure_cuda(prefer_gpu=False)
        assert result["mode"] == "gpu_resolved"
        assert os.environ["CUDA_VISIBLE_DEVICES"] == effective["gpu_uuid"]
        assert os.environ["LD_LIBRARY_PATH"] == vendor["env"]["LD_LIBRARY_PATH"]


@pytest.mark.parametrize("changed", ["CUDA_VISIBLE_DEVICES", "CUDA_HOME", "CUDA_PATH", "XLA_FLAGS", "PATH"])
def test_late_context_tampering_fails(vendor, tmp_path, changed):
    with cc.compiler_child_environment(resolve(vendor), parent_env=vendor["env"], work_dir=tmp_path / "job") as (env, _):
        env[changed] = "-1" if changed == "CUDA_VISIBLE_DEVICES" else "/unrelated"
        with pytest.raises(cc.CompilerContextError):
            cc.validate_compiler_child_environment(env)


def test_per_run_probe_cache_component_change_invalidates(vendor, monkeypatch):
    calls = []
    original = cc._run
    def tracked(command, **kwargs):
        calls.append(command)
        return original(command, **kwargs)
    monkeypatch.setattr(cc, "_run", tracked)
    cache = {}
    resolve(vendor, probe_cache=cache)
    resolve(vendor, probe_cache=cache)
    assert sum("-o" in row for row in calls) == 1
    vendor["libdevice"].write_bytes(b"changed local bitcode")
    resolve(vendor, probe_cache=cache)
    assert sum("-o" in row for row in calls) == 2


def test_stdlib_module_does_not_import_frameworks():
    code = "import sys; import onnx_splitpoint_tool.hailo_compiler_context; assert not any(x in sys.modules for x in ('torch','tensorflow','hailo_sdk_client'))"
    result = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_whitespace_job_root_is_actionable_error(vendor, tmp_path):
    with pytest.raises(cc.CompilerContextError, match="without whitespace"):
        with cc.compiler_child_environment(resolve(vendor), parent_env=vendor["env"], work_dir=tmp_path / "path with spaces"):
            pytest.fail("Whitespace root silently accepted")


def test_original_gpu_evidence_and_r2_worker_are_byte_preserved():
    import hashlib
    root = Path(__file__).resolve().parents[1]
    history = root / "tests/fixtures/v27934_gpu_history"
    provenance = json.loads((history / "PROVENANCE.json").read_text())
    for item in provenance["original_members"]:
        assert hashlib.sha256((root / item["fixture"]).read_bytes()).hexdigest() == item["sha256"]
    assert (root / "scripts/hailo_gpu_diagnostics/worker_context_v27934.py").read_bytes() == (history / "R2/source/worker.py").read_bytes()
    r2 = json.loads((history / "R2/compute/gpu_compute_result.json").read_text())
    assert r2["model_compiler_invoked"] is False
    assert r2["status"] == "compute_pass"
    for item in r2["checks"]:
        if item["name"] != "dfc_sdk_import":
            assert item["detail"]["rtol"] == item["detail"]["atol"] == 3e-4
            assert item["detail"]["cpu_fallback_accepted"] is False
    negative = json.loads((history / "FIX1/hailo10h/gpu_compute_result.json").read_text())
    assert any(item["status"] != "pass" for item in negative["checks"])


@pytest.mark.parametrize("cpu_fallback", [False, True])
def test_changed_collector_entrypoint_runs_unchanged_r2_worker(vendor, tmp_path, cpu_fallback):
    """Real controller/context/subprocess; simulated vendor math is NOT GPU evidence."""
    import ast
    import zipfile
    root = Path(__file__).resolve().parents[1]
    previous = ast.parse((root / "tests/test_v27933_hailo_gpu_smoke.py").read_text())
    fake = next(ast.literal_eval(node.value) for node in previous.body if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "FAKE_TF" for target in node.targets))
    fake += "\nC.get_visible_devices = lambda self,t: [GPU()]\n"
    fake += "def forbidden_late_setter(*args): raise RuntimeError('late SDK initialized setter')\n"
    fake += "Th.set_inter_op_parallelism_threads = forbidden_late_setter\nTh.set_intra_op_parallelism_threads = forbidden_late_setter\nExp.set_memory_growth = forbidden_late_setter\nC.set_visible_devices = forbidden_late_setter\n"
    if cpu_fallback:
        fake = fake.replace("self.device='/device:GPU:0'", "self.device='/device:CPU:0'")
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", "--system-site-packages", str(vendor["root"])], capture_output=True, check=True, timeout=20)
    site = vendor["root"] / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    (site / "tensorflow.py").write_text(fake)
    (site / "hailo_sdk_client.py").write_text("import tensorflow\n__version__='TEST_SIMULATED_NO_GPU'\nclass ClientRunner: pass\n")
    collector = root / "scripts/hailo_gpu_diagnostics"
    lock = tmp_path / "test_interlock"
    launch = "import sys;from pathlib import Path;sys.path.insert(0," + repr(str(collector)) + ");import collect;collect.platform_interlock_path=lambda:Path(" + repr(str(lock)) + ");sys.argv=['collect.py']+sys.argv[1:];raise SystemExit(collect.main())"
    env = dict(vendor["env"], HOME=str(tmp_path), PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable, "-I", "-B", "-c", launch, "--compiler-context", "--families", "hailo10h", "--venv-hailo10", str(vendor["root"]), "--output-parent", str(tmp_path)], env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == (2 if cpu_fallback else 0), result.stdout + result.stderr
    archive = next(tmp_path.glob("hailo_gpu_compute_r1_*.zip"))
    with zipfile.ZipFile(archive) as z:
        context = json.loads(z.read("hailo10h/compiler_context.json"))
        data = json.loads(z.read("hailo10h/gpu_compute_result.json"))
        process = json.loads(z.read("hailo10h/process_result.json"))
        assert not Path(context["view_root"]).exists()
        assert data["tensorflow_imported_by_sdk"] is True
        assert data["threading_policy"]["late_thread_setters_called"] is False
        assert data["memory_growth_policy"]["late_memory_growth_setter_called"] is False
        assert data["environment"]["CUDA_VISIBLE_DEVICES"] == context["gpu_uuid"]
        assert data["environment"]["LD_LIBRARY_PATH"] == vendor["env"]["LD_LIBRARY_PATH"]
        assert data["model_compiler_invoked"] is False
        assert process["cleanup_complete"] is True
        assert data["tensorflow"]["version"] == "TEST_SIMULATED_NO_GPU"
        assert len(data["checks"]) == 5
        assert (data["status"] == "compute_pass") is (not cpu_fallback)


def test_hailo10_ignores_unselected_hailo8_dependency_overlay(vendor):
    vendor["env"]["ONNX_SPLITPOINT_HAILO8_DEPENDENCY_MANIFEST"] = "/not/selected/h8/manifest.json"
    assert resolve(vendor)["component_source"] == "selected_venv_triton"


def test_explicit_component_context_not_blocked_by_unused_overlay(vendor):
    env = dict(vendor["env"], ONNX_SPLITPOINT_HAILO8_DEPENDENCY_MANIFEST="/unused/old/context")
    result = cc.resolve_hailo_compiler_context(vendor["python"], "hailo8", job_override="gpu", parent_env=env,
        explicit_context={"ptxas_path":str(vendor["ptxas"]), "libdevice_path":str(vendor["libdevice"])})
    assert result["component_source"] == "explicit_context"


def test_hailo8l_compute_family_preserves_exact_hardware_identity(vendor):
    from onnx_splitpoint_tool.hailo_backend import _normalize_hailo_hw_arch
    selection = cc.resolve_compute_selection("hailo8l", compute_by_family={"hailo8": "cpu"}, env={})
    context = cc.resolve_hailo_compiler_context(vendor["python"], "hailo8l", compute_selection=selection, parent_env=vendor["env"])
    assert context["family"] == "hailo8" and context["device"] == "cpu"
    assert _normalize_hailo_hw_arch("hailo8l") == "hailo8l"


def test_non_target_assembler_failure_not_mislabeled_architecture(vendor):
    source = vendor["components"] / "ptxas_fixture.py"
    source.write_text(source.read_text().replace("if os.environ.get('FIXTURE_UNSUPPORTED'):", "if '--version' not in sys.argv:\n    print('runtime loader failed',file=sys.stderr)\n    raise SystemExit(127)\nif os.environ.get('FIXTURE_UNSUPPORTED'):"))
    with pytest.raises(cc.CompilerContextError, match="target_probe_failed") as exc:
        resolve(vendor)
    assert exc.value.details["returncode"] == 127
    assert "runtime loader failed" in exc.value.details["stderr"]


def test_partial_view_never_fills_missing_nvcc_from_system_path(vendor, tmp_path):
    foreign = Path(vendor["env"]["PATH"].split(os.pathsep)[0]) / "nvcc"
    sentinel = tmp_path / "wrong_system_nvcc_called"
    foreign.write_text("#!/bin/sh\ntouch " + shlex.quote(str(sentinel)) + "\n")
    foreign.chmod(0o700)
    with cc.compiler_child_environment(resolve(vendor), parent_env=vendor["env"], work_dir=tmp_path / "job") as (env, context):
        result = subprocess.run(["nvcc", "--version"], env=env, capture_output=True, text=True)
        assert result.returncode == 127
        assert "hailo_gpu_compiler_component_missing:nvcc" in result.stderr
        assert not sentinel.exists()
        assert context["full_cuda_toolkit"] is False
        assert "nvcc" in context["unprovided_compiler_components"]
