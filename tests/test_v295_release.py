from __future__ import annotations

import ast
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tomllib
import zipfile

import pytest

import onnx_splitpoint_tool as package
from onnx_splitpoint_tool import source_integrity
from onnx_splitpoint_tool.remote_runtime_closure import native_remote_package_closure
from onnx_splitpoint_tool.release_identity import BUILD_ID, VERSION
from scripts import build_source_manifest, build_source_release, refresh_editable_install


ROOT = Path(__file__).resolve().parents[1]
CURRENT_RELEASE_DOC = f"docs/RELEASE_{VERSION}.md"
PREVIOUS_RELEASE_DOCS = ("docs/RELEASE_2.95.0.md", "docs/RELEASE_2.95.1.md", "docs/RELEASE_2.95.2.md", "docs/RELEASE_2.95.3.md", "docs/RELEASE_2.95.4.md")
WORKERS = (
    "native_deepx_trt_e2e_from_benchmarkset.py",
    "native_deepx_full_energy_hotloop.py",
    "native_full_baseline_eval_runner.py",
    "native_full_semantic_dump.py",
    "native_producer_energy_plan.py",
    "run_native_producer_energy_from_summary.py",
)
PACKAGE_FILES = (
    "onnx_splitpoint_tool/__init__.py",
    "onnx_splitpoint_tool/release_identity.py",
    "onnx_splitpoint_tool/native_deepx_async.py",
    "onnx_splitpoint_tool/native_command_contract.py",
    "onnx_splitpoint_tool/remote/resource_recovery.py",
    "onnx_splitpoint_tool/remote/resource_recovery_observation.py",
    "onnx_splitpoint_tool/remote/process_lease.py",
    "onnx_splitpoint_tool/remote/process_lease_cli.py",
    "onnx_splitpoint_tool/workflow/run_control.py",
    "onnx_splitpoint_tool/workflow/hardware_acceptance.py",
    "onnx_splitpoint_tool/workflow/hardware_acceptance_native.py",
    "onnx_splitpoint_tool/workflow/hardware_acceptance_numerical.py",
    "onnx_splitpoint_tool/workflow/native_refresh.py",
    "onnx_splitpoint_tool/gui/controller.py",
    "onnx_splitpoint_tool/runners/__init__.py",
    "onnx_splitpoint_tool/runners/_types.py",
    "onnx_splitpoint_tool/runners/request_latency.py",
)


def _run(command, *, cwd):
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(command, cwd=cwd, env=environment,
                               text=True, capture_output=True, timeout=90)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return completed


def test_v295_identity_matches_normal_packaging_updater_and_refresher():
    assert VERSION == package.__version__ == package.__release__ == "2.95.5"
    assert BUILD_ID == package.__build_id__ == "v2.95.5"
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    assert project["project"]["version"] == VERSION
    assert next(row for row in lock["package"] if row["name"] == "onnx-splitpoint-tool")["version"] == VERSION
    assert refresh_editable_install.load_project(ROOT).version == VERSION
    updater = (ROOT / "scripts/update_source_release.sh").read_text()
    for marker in (f'EXPECTED_VERSION = "{VERSION}"', f'EXPECTED_BUILD_ID = "{BUILD_ID}"',
                   f"--expected-version {VERSION}", f"ONNX-Splitpoint-Tool_v{VERSION}",
                   f"onnx-splitpoint-tool=={VERSION}"):
        assert marker in updater
    assert (ROOT / "README.md").read_text().splitlines()[0] == f"# ONNX Splitpoint Tool v{VERSION}"
    assert f"## Current release: {VERSION}" in (ROOT / "docs/VERSIONING.md").read_text()


def _updater_source_policy():
    """Load only the ordinary updater's pure path policy, without running it."""
    updater = (ROOT / "scripts/update_source_release.sh").read_text()
    embedded = updater.split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
    parsed = ast.parse(embedded)
    namespace = {"PurePosixPath": PurePosixPath}
    constants = []
    for node in parsed.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id.isupper():
                constants.append(node)
    functions = [node for node in parsed.body if isinstance(node, ast.FunctionDef)
                 and node.name in {"valid_logical_path", "allowlisted_source_path"}]
    assert len(functions) == 2
    exec(compile(ast.Module(body=constants + functions, type_ignores=[]), "updater_source_policy", "exec"), namespace)
    return namespace["allowlisted_source_path"]


@pytest.mark.parametrize(("relative", "accepted"), [
    ("docs/RELEASE_2.95.5.md", True),
    ("docs/RELEASE_2.95.4.md", True),
    ("docs/RELEASE_2.95.3.md", True),
    ("docs/RELEASE_2.95.2.md", True),
    ("docs/RELEASE_2.95.1.md", True),
    ("docs/RELEASE_2.95.0.md", True),
    ("docs/RELEASE_2.94.0.md", True),
    ("docs/RELEASE_2.95.6.md", False),
    ("docs/RELEASE_2.95.2.md.tmp", False),
    ("onnx_splitpoint_tool/remote/resource_recovery.py", True),
    ("onnx_splitpoint_tool/workflow/hardware_acceptance_numerical.py", True),
    ("onnx_splitpoint_tool/gui/controller.py", True),
    ("logs/NUMERICAL_RESULT.json", False),
    ("reports/NUMERICAL_RESULT.json", False),
])
def test_v2955_release_paths_match_builder_installed_verifier_and_trusted_updater(relative, accepted):
    assert build_source_manifest._source_candidate(ROOT / relative, ROOT, package_version=VERSION) is accepted
    assert source_integrity._source_candidate(Path(relative), package_version=VERSION) is accepted
    assert _updater_source_policy()(relative) is accepted


def test_a4_adapter_and_normal_workers_are_in_existing_release_and_remote_closure():
    closure = {relative for relative, _module, _tokens in native_remote_package_closure()}
    assert "onnx_splitpoint_tool/native_deepx_async.py" in closure
    for relative in (*PACKAGE_FILES, CURRENT_RELEASE_DOC, *PREVIOUS_RELEASE_DOCS):
        assert build_source_manifest._source_candidate(ROOT / relative, ROOT, package_version=VERSION)
        assert source_integrity._source_candidate(Path(relative), package_version=VERSION)
    for name in WORKERS:
        source = ROOT / "scripts" / name
        mirror = ROOT / "onnx_splitpoint_tool/resources/remote_scripts" / name
        assert source.read_bytes() == mirror.read_bytes(), name
        assert build_source_manifest._source_candidate(source, ROOT, package_version=VERSION)
        assert build_source_manifest._source_candidate(mirror, ROOT, package_version=VERSION)


@pytest.fixture
def current_packaging_fixture(tmp_path):
    stage = tmp_path / "source"
    files = (
        "pyproject.toml", "MANIFEST.in", "README.md", CURRENT_RELEASE_DOC, *PREVIOUS_RELEASE_DOCS,
        "scripts/build_source_manifest.py", "scripts/build_source_release.py",
        "scripts/refresh_editable_install.py", *PACKAGE_FILES,
        *(f"scripts/{name}" for name in WORKERS),
        *(f"onnx_splitpoint_tool/resources/remote_scripts/{name}" for name in WORKERS),
    )
    for relative in files:
        destination = stage / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    return stage


def test_ordinary_source_archive_contains_current_adapter_and_exact_workers(current_packaging_fixture, tmp_path):
    stage = current_packaging_fixture
    manifest = build_source_manifest.build(stage)
    assert manifest["package_version"] == VERSION
    assert manifest["workflow_version"] == BUILD_ID
    archive_path = tmp_path / f"ONNX-Splitpoint-Tool_v{VERSION}_source.zip"
    result = build_source_release.build_release(stage, archive_path)
    assert result["ok"] is True
    assert build_source_release.verify_archive(stage, archive_path)["ok"] is True
    prefix = f"ONNX-Splitpoint-Tool_v{VERSION}/"
    with zipfile.ZipFile(archive_path) as archive:
        included = {row["path"] for row in manifest["files"]}
        for relative in (*PACKAGE_FILES, CURRENT_RELEASE_DOC, *PREVIOUS_RELEASE_DOCS):
            assert relative in included
            assert archive.read(prefix + relative) == (stage / relative).read_bytes()
        for name in WORKERS:
            assert archive.read(prefix + "scripts/" + name) == archive.read(prefix + "onnx_splitpoint_tool/resources/remote_scripts/" + name)


def test_normal_wheel_build_contains_adapter_and_imports_it_without_source_overlay(current_packaging_fixture, tmp_path):
    stage = current_packaging_fixture
    wheel_dir = tmp_path / "wheels"
    wheel_dir.mkdir()
    _run([sys.executable, "-I", "-B", "-c",
          "from setuptools.build_meta import build_wheel; "
          f"print(build_wheel({str(wheel_dir)!r}))"], cwd=stage)
    wheels = list(wheel_dir.glob("*.whl"))
    assert len(wheels) == 1
    wheel = wheels[0]
    assert f"-{VERSION}-" in wheel.name
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        for relative in PACKAGE_FILES:
            assert relative in names
            assert archive.read(relative) == (stage / relative).read_bytes()
        metadata_name = next(name for name in names if name.endswith(".dist-info/METADATA"))
        assert f"Version: {VERSION}\n" in archive.read(metadata_name).decode()
        for name in WORKERS:
            target = "onnx_splitpoint_tool/resources/remote_scripts/" + name
            assert target in names
            assert archive.read(target) == (stage / target).read_bytes()
    program = f"""
import json, sys
sys.path.insert(0, {str(wheel)!r})
import onnx_splitpoint_tool as package
import onnx_splitpoint_tool.native_deepx_async as adapter
assert package.__version__ == {VERSION!r}
assert adapter.INFLIGHT_LIMIT == 4 and adapter.FIFO_DEPTH == 3
assert {str(wheel)!r} in adapter.__file__
print(json.dumps({{'version': package.__version__, 'module': adapter.__file__, 'inference_started': False}}))
"""
    result = _run([sys.executable, "-I", "-B", "-c", program], cwd=tmp_path)
    assert json.loads(result.stdout)["inference_started"] is False
