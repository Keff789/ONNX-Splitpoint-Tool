from __future__ import annotations

from pathlib import Path

import pytest

import onnx_splitpoint_tool as package
from onnx_splitpoint_tool import v279_smoke, v27934_smoke
from onnx_splitpoint_tool.release_identity import BUILD_CONTRACT_VERSION, BUILD_ID, VERSION
from onnx_splitpoint_tool.workflow.runner import WORKFLOW_VERSION
from onnx_splitpoint_tool import source_integrity
from scripts import build_source_manifest


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_VERSION = "2.79.34"
EXPECTED_BUILD = "v2.79.34-hailo-compiler-context-force-off"


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_current_identity_and_smoke_alias() -> None:
    assert package.__version__ == package.__release__ == VERSION == EXPECTED_VERSION
    assert package.__development_lineage__ == "v2.79"
    assert BUILD_CONTRACT_VERSION == 2
    assert package.__build_id__ == BUILD_ID == WORKFLOW_VERSION == EXPECTED_BUILD
    assert v27934_smoke.VERSION == v279_smoke.VERSION == EXPECTED_VERSION
    assert v27934_smoke.BUILD_ID == v279_smoke.BUILD_ID == EXPECTED_BUILD


def test_current_entrypoints_retain_historical_aliases() -> None:
    pyproject = _read("pyproject.toml")
    assert 'version = "2.79.34"' in pyproject
    assert 'name = "onnx-splitpoint-tool"\nversion = "2.79.34"' in _read("uv.lock")
    for number in (19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34):
        for alias in (f"v279{number}", f"v2-79-{number}"):
            assert (
                f'onnx-splitpoint-smoke-{alias} = '
                f'"onnx_splitpoint_tool.v279{number}_smoke:main"'
            ) in pyproject
    for path in ("scripts/run_v279_small_acceptance.sh", "scripts/run_local_acceptance.sh"):
        assert "run_v27934_small_acceptance.sh" in _read(path)


def test_updater_exact_identity_and_retained_entrypoints() -> None:
    updater = _read("scripts/update_source_release.sh")
    for marker in (
        'EXPECTED_VERSION = "2.79.34"',
        f'EXPECTED_BUILD_ID = "{EXPECTED_BUILD}"',
        "ONNX-Splitpoint-Tool_v2.79.34",
        "--expected-version 2.79.34",
        "--run-entrypoint onnx-splitpoint-smoke-v27934",
    ):
        assert marker in updater
    for number in (19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34):
        assert f"onnx-splitpoint-smoke-v279{number}=onnx_splitpoint_tool.v279{number}_smoke:main" in updater


def test_updater_supplies_each_entrypoint_as_a_separate_cli_option():
    import shlex
    tokens = shlex.split(_read('scripts/update_source_release.sh').replace('\\\n', ' '), comments=True)
    entrypoints = [(index, token) for index, token in enumerate(tokens)
                   if token.startswith('onnx-splitpoint-') and '=onnx_splitpoint_tool.' in token]
    assert len(entrypoints) >= 26
    assert all(tokens[index - 1] == '--require-entrypoint' for index, _ in entrypoints)


def test_release_docs_are_current_and_historical_docs_are_retained() -> None:
    for path in ("TESTANLEITUNG_2.79.34.md", "VERSION_2.79.34_BUILD_AND_TEST_REPORT.md"):
        contents = _read(path)
        assert EXPECTED_BUILD in contents
        assert "NOT_RUN" in contents
        assert "COMPILE_INFEASIBLE" in contents
        assert "TRANSIENT_INFRASTRUCTURE" in contents
    for historical in (
        "TESTANLEITUNG_2.79.19.md", "VERSION_2.79.19_BUILD_AND_TEST_REPORT.md",
        "TESTANLEITUNG_2.79.20.md", "VERSION_2.79.20_BUILD_AND_TEST_REPORT.md",
        "TESTANLEITUNG_2.79.21.md", "VERSION_2.79.21_BUILD_AND_TEST_REPORT.md",
        "TESTANLEITUNG_2.79.22.md", "VERSION_2.79.22_BUILD_AND_TEST_REPORT.md",
        "TESTANLEITUNG_2.79.23.md", "VERSION_2.79.23_BUILD_AND_TEST_REPORT.md",
        "TESTANLEITUNG_2.79.24.md", "VERSION_2.79.24_BUILD_AND_TEST_REPORT.md",
        "TESTANLEITUNG_2.79.25.md", "VERSION_2.79.25_BUILD_AND_TEST_REPORT.md",
        "TESTANLEITUNG_2.79.26.md", "VERSION_2.79.26_BUILD_AND_TEST_REPORT.md",
    ):
        assert (ROOT / historical).is_file()
        assert historical in _read("scripts/build_source_manifest.py")
        assert historical in _read("scripts/update_source_release.sh")


def test_current_gate_retains_behavioral_regressions_not_old_identity_gate() -> None:
    gate = _read("scripts/run_v27934_small_acceptance.sh")
    assert "tests/test_v27934_release_closure.py" in gate
    for test in (
        "tests/test_v27927_result_endpoints.py",
        "tests/test_v27927_status_reporting.py",
        "tests/test_v27927_deepx_input_transfer.py",
        "tests/test_v27927_deepx_value_diagnostics.py",
        "tests/test_v27927_probe_launcher.py",

        "tests/test_v27926_deepx_full_decoded_pre_nms.py",
        "tests/test_v27926_deepx_full_downstream.py",
        "tests/test_v27926_runtime_debug_pack.py",
        "tests/test_v27926_quality_submission_failure.py",
        "tests/test_v27925_deepx_compiler_overlay.py",
        "tests/test_v27925_hailo_full_bundle_paths.py",
        "tests/test_v27925_energy_calibration_reporting.py",
        "tests/test_v27925_fast_oracle_validation.py",
        "tests/test_v27925_native_energy_raw_debug_pack.py",
        "tests/test_v27924_terminal_hailo_aliases.py",
        "tests/test_v27924_quality_reporting.py",
        "tests/test_v27924_native_error_fixes.py",
        "tests/test_v27924_trt_active_retention.py",
        "tests/test_v27924_hailo_full_portability.py",
        "tests/test_v27924_negative_preflight.py",
        "tests/test_v27924_rank3_boundary_layout.py",
        "tests/test_v27923_trt_legacy_reuse.py",
        "tests/test_v27923_fs_calibration_gate.py",
        "tests/test_v27923_decoded_completion.py",
        "tests/test_v27923_native_failures.py",
        "tests/test_v27923_deepx_preflight.py",
        "tests/test_v27923_workflow_fixes.py",
        "tests/test_v27922_build_evidence_store.py",
        "tests/test_v27922_negative_backend.py",
        "tests/test_v27922_build_evidence_recovery.py",
        "tests/test_v27922_negative_preflight.py",
        "tests/test_v27922_build_context.py",
        "tests/test_v27922_auxiliary_context.py",
        "tests/test_v2783_build_evidence.py",
        "tests/test_v2783_hailo8_first_feasibility.py",
        "tests/test_v2783_hailo8_evidence_binding.py",
        "tests/test_v27921_final_selection_cache_preflight.py",
    ):
        assert test in gate
    for required in ("score_roundoff", "probe_staging", "probe_real_process"):
        assert f"tests/test_v27928_{required}.py" in gate
    assert "tests/test_v27928_release_closure.py" not in gate
    assert "tests/test_v27929_no_opencv.py" in gate
    assert "tests/test_v27929_probe_real_process.py" in gate
    assert "tests/test_v27927_release_closure.py" not in gate
    assert "tests/test_v27926_release_closure.py" not in gate
    assert "tests/test_v27925_release_closure.py" not in gate
    assert "tests/test_v27924_release_closure.py" not in gate
    assert "tests/test_v27923_release_closure.py" not in gate
    assert "tests/test_v27922_release_closure.py" not in gate
    assert "tests/test_v27921_release_closure.py" not in gate
    assert "tests/test_v27920_hailo_deepx_reuse.py" in gate
    assert "tests/test_v27920_trt_builder_cache_reuse.py" in gate
    assert "tests/test_v27920_release_closure.py" not in gate


def test_current_hardware_independent_smoke(capsys) -> None:
    assert v27934_smoke.main() == 0
    assert "PASS v2.79.34 smoke" in capsys.readouterr().out


def _minimal_release(tmp_path: Path) -> Path:
    root = tmp_path / "release"
    files = {
        "pyproject.toml": f'[project]\nname = "onnx-splitpoint-tool"\nversion = "{VERSION}"\n',
        "onnx_splitpoint_tool/release_identity.py": f'VERSION = "{VERSION}"\nBUILD_ID = "{BUILD_ID}"\n',
        "onnx_splitpoint_tool/runtime.py": "VALUE = 1\n",
    }
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    build_source_manifest.build(root)
    return root


@pytest.mark.parametrize("name", ("artifact_store", "build_evidence", "SplitNetworks", "Results", "EnergyMeasurements"))
def test_local_artifacts_and_negative_evidence_are_operational_state(
    tmp_path: Path, name: str,
) -> None:
    root = _minimal_release(tmp_path)
    store = root / name
    store.mkdir()
    evidence = store / "preserved.bin"
    evidence.write_bytes(b"compiled artifact or exact negative evidence")
    (store / "current").symlink_to(evidence.name)

    assert build_source_manifest.verify(root, scope="installed")["ok"]
    assert source_integrity.verify_installed_source_integrity(root)["ok"]
    assert not build_source_manifest.verify(root, scope="release")["ok"]
    assert evidence.read_bytes() == b"compiled artifact or exact negative evidence"

    # The exception is only for these real top-level operational directories.
    (root / "onnx_splitpoint_tool/runtime.py").write_text("VALUE = 2\n")
    assert not build_source_manifest.verify(root, scope="installed")["ok"]
    assert not source_integrity.verify_installed_source_integrity(root)["ok"]


@pytest.mark.parametrize("name", ("artifact_store", "build_evidence", "SplitNetworks", "Results", "EnergyMeasurements"))
@pytest.mark.parametrize("kind", ("symlink", "regular_file"))
def test_operational_root_name_does_not_admit_links_or_files(
    tmp_path: Path, name: str, kind: str,
) -> None:
    root = _minimal_release(tmp_path)
    candidate = root / name
    if kind == "symlink":
        candidate.symlink_to(root / "onnx_splitpoint_tool", target_is_directory=True)
    else:
        candidate.write_bytes(b"not a directory")
    assert not build_source_manifest.verify(root, scope="installed")["ok"]
    assert not source_integrity.verify_installed_source_integrity(root)["ok"]


def test_updater_protects_local_evidence_during_sync_retry_and_drift_check() -> None:
    updater = _read("scripts/update_source_release.sh")
    for name in ("artifact_store", "build_evidence", "SplitNetworks", "Results", "EnergyMeasurements"):
        assert updater.count(f"--exclude='/{name}/'") == 3
        assert name in build_source_manifest.PRESERVED_INSTALL_ROOTS
        assert name in source_integrity._PRESERVED_INSTALL_ROOTS
    assert "--exclude='*.hef'" not in updater


@pytest.mark.parametrize("script", (
    "native_full_baseline_eval_runner.py",
    "native_full_semantic_dump.py",
    "native_producer_final_report.py",
))
def test_R01_dispatched_resource_scripts_match_source(script: str) -> None:
    assert (ROOT / "scripts" / script).read_bytes() == (
        ROOT / "onnx_splitpoint_tool/resources/remote_scripts" / script
    ).read_bytes()


def test_R02_fresh_process_imports_only_the_selected_release(tmp_path: Path) -> None:
    import json
    import os
    import subprocess
    import sys

    code = '''
import importlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1]).resolve(strict=True)
sys.path.insert(0, str(root))
modules = [importlib.import_module(name) for name in (
    "onnx_splitpoint_tool.release_identity",
    "onnx_splitpoint_tool.workflow.artifacts",
    "onnx_splitpoint_tool.workflow.runner",
    "onnx_splitpoint_tool.native_detection_postprocess",
    "onnx_splitpoint_tool.quality_service",
    "onnx_splitpoint_tool.resources.remote_scripts.native_full_semantic_dump",
)]
paths = [str(pathlib.Path(module.__file__).resolve()) for module in modules]
assert all(pathlib.Path(path).is_relative_to(root) for path in paths)
assert modules[0].VERSION == "2.79.34"
print(json.dumps({"version": modules[0].VERSION, "paths": paths}))
'''
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", code, str(ROOT)],
        cwd=tmp_path, text=True, capture_output=True, timeout=45,
        env={key: value for key, value in os.environ.items() if key not in {"PYTHONPATH", "PYTHONHOME"}},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout.strip().splitlines()[-1])["version"] == "2.79.34"


def test_idle_or_paused_gui_blocks_updater_without_workflow_lock(tmp_path: Path) -> None:
    import os
    import signal
    import subprocess
    import sys
    from tests.test_v27910_update_active_workflow_guard import _fixture, _run

    target, archive, env, _ = _fixture(tmp_path)
    # Preserve this test interpreter's venv configuration; resolving its
    # binary symlink loses the standalone runtime's stdlib discovery.
    import shlex
    fixture_python = target / ".venv/bin/python"
    fixture_python.unlink()
    fixture_python.write_text("#!/bin/sh\nexec " + shlex.quote(sys.executable) + ' "$@"\n')
    fixture_python.chmod(0o755)
    gui = target / "analyse_and_split_gui.py"
    gui.write_text("import signal\nprint('GUI_READY', flush=True)\nsignal.pause()\n")
    before = (target / "SOURCE_MANIFEST.json").read_bytes()
    process = subprocess.Popen(
        [sys.executable, str(gui)], cwd=target,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        assert process.stdout.readline().strip() == "GUI_READY"
        for paused in (False, True):
            if paused:
                os.kill(process.pid, signal.SIGSTOP)
                _, status = os.waitpid(process.pid, os.WUNTRACED)
                assert os.WIFSTOPPED(status)
            result = _run(archive, target, env)
            assert result.returncode == 75, result.stdout + result.stderr
            assert "Tool-GUI läuft oder ist pausiert" in result.stderr
            # A test executor may mount host /proc inside a child PID namespace.
            # The diagnostic PID is the observed /proc PID.
            assert "PID=" in result.stderr and str(target) in result.stderr
            if paused:
                assert "Zustand=T" in result.stderr
            assert (target / "SOURCE_MANIFEST.json").read_bytes() == before
    finally:
        os.kill(process.pid, signal.SIGCONT)
        process.terminate()
        process.communicate(timeout=5)


def test_gate_covers_prior_selection_and_all_rev2_regressions() -> None:
    import re

    previous = set(re.findall(r"tests/[^\s\\]+", _read("scripts/run_v27933_small_acceptance.sh")))
    current = set(re.findall(r"tests/[^\s\\]+", _read("scripts/run_v27934_small_acceptance.sh")))
    previous.remove("tests/test_v27933_release_closure.py")
    assert previous <= current
    for test in (
        "test_v269a_deepx_full_central_quality.py", "test_v269d_trt_central_quality_producer.py",
        "test_v269e_quality_export_vendored.py", "test_v60m_runtime_energy_integrity.py",
        "test_v2796_artifact_index_closure.py", "test_v2797_artifact_index_current_path.py",
        "test_v2798_artifact_index_current_path.py", "test_artifact_index_report_cleanup.py",
    ):
        assert "tests/" + test in current


def test_exact_recorded_npz_is_manifested_and_other_binaries_stay_excluded(tmp_path: Path) -> None:
    import ast
    import json
    from pathlib import PurePosixPath

    relative = Path("tests/fixtures/v27930/semantic_probe_outputs.npz")
    root = _minimal_release(tmp_path)
    fixture = root / relative
    fixture.parent.mkdir(parents=True)
    fixture.write_bytes((ROOT / relative).read_bytes())
    image_relative = relative.with_name("semantic_probe_input.jpg")
    (root / image_relative).write_bytes((ROOT / image_relative).read_bytes())
    build_source_manifest.build(root)
    manifest = json.loads((root / "SOURCE_MANIFEST.json").read_text())
    assert str(relative) in {record["path"] for record in manifest["files"]}
    assert str(image_relative) in {record["path"] for record in manifest["files"]}
    assert build_source_manifest.verify(root, scope="release")["ok"]
    assert source_integrity.verify_installed_source_integrity(root)["ok"]

    # Execute the installer's actual stdlib archive path predicate, not a copy
    # of the corrected expression. Its constant data shares the source policy.
    updater = _read("scripts/update_source_release.sh")
    embedded = updater[updater.index('EXPECTED_VERSION = "2.79.34"'):]
    predicate = embedded[embedded.index("def valid_logical_path("):embedded.index("def strict_sha256(")]
    ns = {"PurePosixPath": PurePosixPath, "EXPECTED_VERSION": "2.79.34"}
    for name in (
        "ALLOWED_ROOT_FILES", "RETAINED_RELEASE_DOC_FILES", "ALLOWED_SOURCE_TREES",
        "ALLOWED_DOC_FILES", "EXCLUDED_FILE_NAMES", "EXCLUDED_CASEFOLD_FILE_NAMES",
        "EXCLUDED_FILE_PREFIXES", "EXCLUDED_DIRECTORY_NAMES", "EXCLUDED_FILE_SUFFIXES",
    ):
        ns[name] = getattr(build_source_manifest, name)
    definition = embedded[embedded.index("ALLOWED_BINARY_TEST_FIXTURES = {"):]
    ns["ALLOWED_BINARY_TEST_FIXTURES"] = ast.literal_eval(definition.split("=", 1)[1].split("}", 1)[0] + "}")
    exec(compile(predicate, "updater_archive_source_predicate", "exec"), ns)
    assert ns["allowlisted_source_path"](relative.as_posix())
    for unwanted in (
        "tests/fixtures/v27930/other.npz", "tests/fixtures/v27930/semantic_probe_outputs.NPZ",
        "tests/fixtures/v27930/nested/semantic_probe_outputs.npz", "tests/fixtures/v27930/model.dxnn",
        "tests/fixtures/v27929/semantic_probe_outputs.npz",
    ):
        candidate = Path(unwanted)
        assert not build_source_manifest._source_candidate(root / candidate, root, package_version=VERSION)
        assert not source_integrity._source_candidate(candidate, package_version=VERSION)
        assert not ns["allowlisted_source_path"](unwanted)
    with fixture.open("r+b") as stream:
        stream.write(b"mutated")
    assert not source_integrity.verify_installed_source_integrity(root)["ok"]


@pytest.mark.parametrize("kind, expected_rc", (("pass", 0), ("skipped", 70), ("xfail", 70), ("missing", 70)))
def test_required_acceptance_reports_never_pass_skipped_tests(tmp_path: Path, kind: str, expected_rc: int) -> None:
    import json
    import os
    import subprocess
    import sys

    gate = _read("scripts/run_v27934_small_acceptance.sh")
    report_function = gate.split("write_report() {", 1)[1].split("\nPY\n}", 1)[0]
    report_python = report_function.split('"$PYTHON" -B - <<\'PY\'\n', 1)[1]
    main = tmp_path / "main.xml"
    gui = tmp_path / "gui.xml"
    valid = '<testsuites><testsuite tests="1" failures="0" errors="0" skipped="0"><testcase name="ok"/></testsuite></testsuites>'
    gui.write_text(valid)
    if kind == "pass":
        main.write_text(valid)
    elif kind != "missing":
        skipped_type = "pytest.xfail" if kind == "xfail" else "pytest.skip"
        main.write_text(f'<testsuites><testsuite tests="1" failures="0" errors="0" skipped="1"><testcase name="case"><skipped type="{skipped_type}"/></testcase></testsuite></testsuites>')
    report = tmp_path / "acceptance.json"
    env = {**os.environ, "REPORT": str(report), "STARTED": "fixture", "FINAL_RC": "0",
           "PYTEST_JUNIT_MAIN": str(main), "PYTEST_JUNIT_GUI": str(gui),
           **{key: "PASS" for key in ("SMOKE", "PYTEST", "MANIFEST", "COMPILE", "SHELL")}}
    result = subprocess.run([sys.executable, "-I", "-B", "-c", report_python],
                            env=env, text=True, capture_output=True, timeout=10)
    assert result.returncode == expected_rc, result.stdout + result.stderr
    payload = json.loads(report.read_text())
    assert payload["status"] == ("PASS" if kind == "pass" else "FAIL")
    assert payload["return_code"] == expected_rc
    if kind in {"skipped", "xfail"}:
        assert payload["pytest_totals"]["skipped"] == 1


def test_retained_release_document_policy_matches_runtime_verifier() -> None:
    assert source_integrity._RETAINED_RELEASE_DOC_FILES == build_source_manifest.RETAINED_RELEASE_DOC_FILES
    for logical in build_source_manifest.RETAINED_RELEASE_DOC_FILES:
        assert source_integrity._source_candidate(Path(logical), package_version=VERSION)


def test_T11_3_missing_real_dependencies_blocks_release_environment():
    import json
    import os
    import subprocess
    import sys
    # -S deliberately selects a real dependency-free stdlib environment.
    # No fake import module or patched success predicate is involved.
    result = subprocess.run([sys.executable, '-I', '-S', '-B',
        str(ROOT/'scripts/check_acceptance_environment.py')],
        env={**os.environ,'ORT_DISABLE_TELEMETRY':'1'},
        capture_output=True,text=True,timeout=15)
    assert result.returncode == 78
    report=json.loads(result.stdout)
    assert report['status']=='environment_blocked'
    assert any(row['module']=='onnx' and row['status']=='missing_or_unloadable' for row in report['dependencies'])
    assert report['packages_installed'] is False


def test_T11_4_every_existing_script_resource_mirror_matches():
    remote=ROOT/'onnx_splitpoint_tool/resources/remote_scripts'
    pairs=[(path,ROOT/'scripts'/path.name) for path in remote.iterdir()
           if path.is_file() and (ROOT/'scripts'/path.name).is_file()]
    assert pairs, "release contains no source/resource script pairs"
    for resource,source in pairs:
        assert resource.read_bytes()==source.read_bytes(),source.name


def test_T11_4_new_npz_exception_is_exact():
    relative=Path('tests/fixtures/v27931_classification/original_R1_R2_logits.npz')
    assert relative.is_relative_to('tests/fixtures')
    assert build_source_manifest._source_candidate(ROOT/relative,ROOT,package_version=VERSION)
    assert source_integrity._source_candidate(relative,package_version=VERSION)
    for path in (relative.with_name('model.dxnn'),relative.with_name('other.npz')):
        assert not source_integrity._source_candidate(path,package_version=VERSION)


def test_T32_R05_current_workflow_wrapper_binds_all_source_identifiers(tmp_path: Path) -> None:
    import json
    import os
    import subprocess
    import sys

    code = '''
import importlib.util, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
spec = importlib.util.spec_from_file_location("current_full_smoke", root/"scripts/deepx_full_workflow_smoke_v27934.py")
wrapper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wrapper)
print(json.dumps({key: getattr(wrapper, key) for key in ("SOURCE_VERSION", "SOURCE_TAG", "SOURCE_NAME")} | {"DEFAULT_BUNDLE": wrapper.implementation.probe.DEFAULT_BUNDLE}))
'''
    result = subprocess.run([sys.executable, "-I", "-B", "-c", code, str(ROOT)],
                            cwd=tmp_path, text=True, capture_output=True, timeout=30,
                            env={k: v for k, v in os.environ.items() if k not in {"PYTHONPATH", "PYTHONHOME"}})
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout) == {
        "SOURCE_VERSION": "2.79.34", "SOURCE_TAG": "v27934",
        "SOURCE_NAME": "ONNX-Splitpoint-Tool_v2.79.34_SOURCE.zip",
        "DEFAULT_BUNDLE": "ONNX-Splitpoint-Tool_v2.79.34_COMPLETE_DELIVERY_BUNDLE",
    }


@pytest.mark.parametrize("name", (
    "terminal_closure_smoke", "deepx_full_workflow_smoke", "run_complete_set_replay",
    "hailo10_yolo26_boundary_probe", "classification_input_probe",
))
def test_T32_R09_current_launchers_parse_in_fresh_isolated_process(name: str, tmp_path: Path) -> None:
    import os
    import subprocess
    import sys

    result = subprocess.run([sys.executable, "-I", "-B", str(ROOT / "scripts" / f"{name}_v27934.py"), "--help"],
                            cwd=tmp_path, text=True, capture_output=True, timeout=45,
                            env={k: v for k, v in os.environ.items() if k not in {"PYTHONPATH", "PYTHONHOME"}})
    assert result.returncode == 0, result.stdout + result.stderr
    assert "usage:" in result.stdout.lower()


def test_T32_R09_current_gate_preserves_every_previous_case_and_all_new_suites() -> None:
    import re

    previous = set(re.findall(r"tests/[^\s\\]+", _read("scripts/run_v27933_small_acceptance.sh")))
    current_text = _read("scripts/run_v27934_small_acceptance.sh")
    current = set(re.findall(r"tests/[^\s\\]+", current_text))
    assert previous - {"tests/test_v27933_release_closure.py"} <= current
    assert "tests/test_v27933_release_closure.py" not in current
    assert {"tests/test_v27932_" + suffix + ".py" for suffix in (
        "native_failure_persistence", "full_exception_identity", "compiler_precedence",
        "failure_chain_integration",
    )} <= current
    assert "tests/test_v27934_release_closure.py" in current
    assert current_text.count("-o xfail_strict=true") == 2
    assert "all_selected_tests_required_no_skips_or_xfails" in current_text


def test_T32_R09_scientific_scope_is_separate_from_software_release() -> None:
    docs = _read("TESTANLEITUNG_2.79.34.md")
    assert all(term in docs for term in ("1 s × 3", "30 s × 3", "60 s × 3", "FS/command", "B500", "5.000", "NOT_RUN"))
    assert all(name in build_source_manifest.RETAINED_RELEASE_DOC_FILES for name in (
        "TESTANLEITUNG_2.79.31.md", "VERSION_2.79.31_BUILD_AND_TEST_REPORT.md",
    ))


def _exercise_canonical_workdir_updater(tmp_path: Path, updater_root: Path) -> dict:
    """Real archive/updater/metadata transaction; synthetic package bodies only.

    The tiny stdlib package avoids hardware imports. Neither rsync, the updater,
    archive verification nor the preservation assertions are replaced.
    """
    import hashlib
    import json
    import os
    import re
    import shutil
    import subprocess
    import sys

    update_text = (updater_root / 'scripts/update_source_release.sh').read_text()
    selected_version = re.search(r'^EXPECTED_VERSION = "([^"]+)"', update_text, re.M).group(1)
    selected_build = re.search(r'^EXPECTED_BUILD_ID = "([^"]+)"', update_text, re.M).group(1)
    entries = dict(re.findall(r"'(onnx-splitpoint-[^'=]+)=(onnx_splitpoint_tool\.[^']+:main)'", update_text))
    assert entries
    incoming = tmp_path / 'incoming'
    target = tmp_path / 'home/ONNX-Splitpoint-Tool'
    home = target.parent

    def write_package(root: Path, version: str, build: str) -> None:
        (root / 'onnx_splitpoint_tool/workflow').mkdir(parents=True)
        (root / 'scripts').mkdir()
        (root / 'pyproject.toml').write_text(
            '[project]\nname = "onnx-splitpoint-tool"\nversion = "' + version + '"\n'
            + '[project.scripts]\n' + ''.join(f'{name} = "{value}"\n' for name, value in entries.items()))
        (root / 'onnx_splitpoint_tool/__init__.py').write_text('from .release_identity import VERSION as __version__\n')
        (root / 'onnx_splitpoint_tool/release_identity.py').write_text(f'VERSION = "{version}"\nBUILD_ID = "{build}"\n')
        (root / 'onnx_splitpoint_tool/workflow/runner.py').write_text('from ..release_identity import BUILD_ID as WORKFLOW_VERSION\n')
        for value in set(entries.values()):
            module = root / (value.split(':')[0].replace('.', '/') + '.py')
            module.parent.mkdir(parents=True, exist_ok=True)
            module.write_text('def main():\n    return 0\n')
        for name in ('build_source_manifest.py', 'build_source_release.py', 'refresh_editable_install.py', 'update_source_release.sh'):
            shutil.copyfile(updater_root / 'scripts' / name, root / 'scripts' / name)
        result = subprocess.run([sys.executable, '-I', '-S', '-B', str(root/'scripts/build_source_manifest.py'), '--root', str(root)],
                                text=True, capture_output=True, timeout=30)
        assert result.returncode == 0, result.stdout + result.stderr

    write_package(incoming, selected_version, selected_build)
    write_package(target, '2.79.31', 'v2.79.31-complete-set-quality-integration')
    archive = tmp_path / f'ONNX-Splitpoint-Tool_v{selected_version}_SOURCE.zip'
    def run(command, timeout=60):
        completed = subprocess.run(command, text=True, capture_output=True, timeout=timeout,
                                   env={**os.environ, 'HOME': str(home), 'PYTHONPATH': '', 'PYTHONHOME': '',
                                        'PYTHONDONTWRITEBYTECODE': '1'})
        assert completed.returncode == 0, completed.stdout + completed.stderr
        return completed
    run([sys.executable, '-I', '-S', '-B', str(incoming/'scripts/build_source_release.py'), '--root', str(incoming), '--out', str(archive)])
    run([sys.executable, '-I', '-B', '-m', 'venv', '--without-pip', str(target/'.venv')])
    py = target/'.venv/bin/python'
    run([str(py), '-I', '-B', str(target/'scripts/refresh_editable_install.py'), '--root', str(target), '--expected-version', '2.79.31'])
    payloads = {
        target/'profiles/custom_preserved.yaml': b'name: custom_preserved\n',
        target/'SplitNetworks/synthetic_split/sentinel.bin': b'synthetic split bytes, no model\n',
        target/'Results/example/result.json': b'{"original_result":17}\n',
        target/'EnergyMeasurements/Calibrations/setup/m2_idle_power_calibration.json': b'{"idle_power_w":1.25}\n',
        target/'BenchmarkSets/example/manifest.json': b'{"benchmark_set":"preserved"}\n',
        target/'EvaluationRuns/example/result.json': b'{"historical_run":true}\n',
        target/'artifact_store/synthetic_cache.bin': b'synthetic cache bytes, no model\n',
        target/'build_evidence/negative.json': b'{"status":"COMPILE_INFEASIBLE"}\n',
        home/'.onnx_splitpoint_tool/build_environments.yaml': b'build_environments: []\n',
        home/'Models/BackendArtifacts/deepx/synthetic_cache.bin': b'deepx synthetic cache bytes\n',
        home/'Models/BackendArtifacts/hailo/synthetic_cache.bin': b'hailo synthetic cache bytes\n',
    }
    for path, content in payloads.items():
        path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(content)
    venv_cfg = (target/'.venv/pyvenv.cfg').read_bytes()
    inode = (target/'.venv').stat().st_ino
    output = run(['bash', str(updater_root/'scripts/update_source_release.sh'), str(archive), str(target)], timeout=90)
    observation = {str(path.relative_to(home)): path.exists() and path.read_bytes() == content for path, content in payloads.items()}
    assert (target/'.venv/pyvenv.cfg').read_bytes() == venv_cfg
    assert (target/'.venv').stat().st_ino == inode
    identity = run([str(py), '-I', '-B', '-c',
                    'import onnx_splitpoint_tool as p;from importlib.metadata import version;print(p.__version__);print(version("onnx-splitpoint-tool"))'])
    assert identity.stdout.splitlines() == [selected_version, selected_version]
    return {'preserved': observation, 'updater_return_code': output.returncode,
            'source_version': selected_version, 'venv_preserved': True,
            'source_or_hardware_body_simulation': 'minimal_stdlib_package_only',
            'updater_rsync_manifest_metadata': 'actual_production_implementations'}


def test_T32_R07_real_updater_preserves_canonical_workdir_calibrations_and_results(tmp_path: Path) -> None:
    report = _exercise_canonical_workdir_updater(tmp_path, ROOT)
    assert all(report['preserved'].values()), report


def test_T33_original_audit_archive_exception_is_exact():
    relative = Path("tests/fixtures/v27933_force_origin/force_origin_audit_20260909T095730Z_ni3utrwy.zip")
    assert build_source_manifest._source_candidate(ROOT/relative, ROOT, package_version=VERSION)
    assert source_integrity._source_candidate(relative, package_version=VERSION)
    assert build_source_manifest.ALLOWED_BINARY_TEST_FIXTURES == source_integrity._ALLOWED_BINARY_TEST_FIXTURES
    for other in (relative.with_name("other.zip"), relative.with_suffix(".ZIP"), relative.parent/"nested"/relative.name):
        assert not build_source_manifest._source_candidate(ROOT/other, ROOT, package_version=VERSION)
        assert not source_integrity._source_candidate(other, package_version=VERSION)


def test_T33_gate_includes_every_current_suite_and_previous_behavior():
    gate = _read("scripts/run_v27934_small_acceptance.sh")
    for path in (ROOT/"tests").glob("test_v27934_*.py"):
        assert "tests/" + path.name in gate
    for path in ("tests/test_v60n_run_modes.py", "tests/test_v60p_smoke_warning_fixes.py",
                 "tests/test_v27529_final_quality_standard_path.py"):
        assert path in gate
