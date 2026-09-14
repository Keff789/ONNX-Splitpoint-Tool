"""AP4/T34.37–42 offline contracts. No GPU/runtime success is simulated."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import pytest

from onnx_splitpoint_tool import hailo_dependency_plan as hp


CUDA_REQUIREMENT = 'nvidia-cuda-runtime-cu12==12.5.82; extra == "and-cuda"'


def metadata(site, name, version, requires=(), extras=()):
    folder = site / (name.replace("-", "_") + "-" + version + ".dist-info")
    folder.mkdir(parents=True)
    path = folder / "METADATA"
    path.write_text("Metadata-Version: 2.1\nName: " + name + "\nVersion: " + version + "\n" +
                    "".join("Requires-Dist: " + x + "\n" for x in requires) +
                    "".join("Provides-Extra: " + x + "\n" for x in extras))
    return path


@pytest.fixture
def h8(tmp_path):
    root = tmp_path / "venv_hailo8"
    (root / "bin").mkdir(parents=True)
    (root / "bin/python").symlink_to(sys.executable)
    (root / "pyvenv.cfg").write_text("include-system-site-packages = false\n")
    site = root / "lib" / ("python" + ".".join(str(x) for x in sys.version_info[:2])) / "site-packages"
    site.mkdir(parents=True)
    metadata(site, "hailo-dataflow-compiler", "3.33.1", ["tensorflow==2.18.0"])
    metadata(site, "tensorflow", "2.18.0", [CUDA_REQUIREMENT], ["and-cuda"])
    return root, site


def inventory(h8):
    return hp.collect_inventory(h8[0] / "bin/python")


def target(h8):
    return h8[0].parent / "hailo8_cuda_reviewed"


def wheel(h8, *, name="nvidia-cuda-runtime-cu12", version="12.5.82", requires=(), extra_files=None,
          suffix="py3-none-any"):
    path = h8[0].parent / (name.replace("-", "_") + "-" + version + "-" + suffix + ".whl")
    dist = name.replace("-", "_") + "-" + version + ".dist-info"
    contents = {dist + "/METADATA": "Metadata-Version: 2.1\nName: " + name + "\nVersion: " + version + "\n" +
                "".join("Requires-Dist: " + r + "\n" for r in requires),
                "nvidia/cuda_runtime/lib/libcudart.so.12": "small synthetic bytes, never loaded",
                "nvidia/cuda_runtime/__init__.py": ""}
    contents.update(extra_files or {})
    with zipfile.ZipFile(path, "w") as z:
        for name, data in contents.items():
            z.writestr(name, data)
    return path


def ready_plan(h8, **kwargs):
    return hp.build_plan(inventory(h8), target(h8), packages=["nvidia-cuda-runtime-cu12"],
                         wheels=[wheel(h8, **kwargs)])


def save_plan(h8, plan):
    path = h8[0].parent / "review.json"
    path.write_text(json.dumps(plan, sort_keys=True))
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def stage(h8, plan=None):
    plan = plan or ready_plan(h8)
    path, digest = save_plan(h8, plan)
    return hp.stage_reviewed_plan(plan, expected_plan_sha256=digest, plan_file=path)


def test_t34_37_full_metadata_fresh_process_no_importhooks(h8):
    marker = h8[0].parent / "IMPORT_HOOK_WAS_RUN"
    (h8[1] / "danger.pth").write_text("import pathlib; pathlib.Path(" + repr(str(marker)) + ").touch()\n")
    (h8[1] / "tensorflow.py").write_text("raise AssertionError('framework must never import')\n")
    before = dict(os.environ)
    result = inventory(h8)
    assert result["python_executable"] == str(h8[0] / "bin/python")
    assert result["selected_venv"] == str(h8[0])
    assert result["distributions"][1]["requires_dist"] == [CUDA_REQUIREMENT]
    assert result["vendor_imports"] is False
    assert not marker.exists()
    assert dict(os.environ) == before


def test_t34_37_markers_dynamic_pins_and_explicit_selection(h8):
    result = inventory(h8)
    tf = next(d for d in result["distributions"] if d["name"] == "tensorflow")
    tf["requires_dist"] += ['nvidia-cudnn-cu12==9.99.1; extra == "and-cuda" and python_version >= "3.8"',
                            'nvidia-curand-cu12==0.0.1; sys_platform == "win32" and extra == "and-cuda"']
    plan = hp.build_plan(result, target(h8))
    assert plan["status"] == "REVIEW_REQUIRED"
    assert [(p["name"], p["version"]) for p in plan["proposals"]] == [
        ("nvidia-cuda-runtime-cu12", "12.5.82"), ("nvidia-cudnn-cu12", "9.99.1")]
    assert plan["selected_packages"] == []
    assert not target(h8).exists()
    assert plan["inventory"] == result


def test_t34_37_historical_q03_preserved_as_proposals_not_install_order(h8):
    fixture = json.loads((Path(__file__).parent / "fixtures/v27934_hailo8_historical_metadata.json").read_text())
    original = copy.deepcopy(fixture)
    result = inventory(h8)
    result["distributions"] = fixture["hailo8"]["packages"]
    plan = hp.build_plan(result, target(h8))
    assert plan["status"] == "REVIEW_REQUIRED"
    assert len(plan["proposals"]) == 12
    assert plan["selected_packages"] == []
    assert all(row["necessity"] == "NOT_PROVEN_BY_METADATA" for row in plan["proposals"])
    assert fixture == original
    # The existing H10 Torch requirement is kept as a negative fixture, never
    # copied into a H8 environment or assumed compatible with H8 TF extras.
    torch = next(row for row in fixture["hailo10h"]["packages"] if row["name"] == "torch")
    result["distributions"] = result["distributions"] + [torch]
    conflict = hp.build_plan(result, target(h8), packages=["nvidia-cuda-runtime-cu12"])
    assert conflict["status"] == "BLOCKED"
    assert any(c["reason"] == "installed_framework_or_sdk_constraint_conflict" and c["source"] == "torch" for c in conflict["conflicts"])


def test_t34_37_unbounded_pin_blocks_review(h8):
    result = inventory(h8)
    result["distributions"][1]["requires_dist"] = ['nvidia-cuda-runtime-cu12>=12; extra == "and-cuda"']
    plan = hp.build_plan(result, target(h8))
    assert plan["status"] == "BLOCKED"
    assert plan["conflicts"][0]["reason"] == "tensorflow_supplement_without_exact_pin"


def test_t34_37_duplicate_distribution_fails(h8):
    metadata(h8[1], "tensorFlow", "2.19.0")
    with pytest.raises(ValueError, match="duplicate_distribution"):
        hp.build_plan(inventory(h8), target(h8))


def test_t34_37_system_site_inventory_rejected(h8):
    (h8[0] / "pyvenv.cfg").write_text("include-system-site-packages = true\n")
    with pytest.raises(ValueError, match="inherits_unbounded_system"):
        inventory(h8)


def test_t34_37_other_python_site_not_misreported_as_active(h8):
    metadata(h8[0] / "lib/python0.0/site-packages", "tensorflow", "0.0.0")
    assert len(inventory(h8)["distributions"]) == 2


def test_t34_38_framework_pin_conflict_stops(h8):
    metadata(h8[1], "torch", "2.9.1", ["nvidia-cuda-runtime-cu12==12.8.90"])
    plan = ready_plan(h8)
    assert plan["status"] == "BLOCKED"
    assert any(r["reason"] == "installed_framework_or_sdk_constraint_conflict" and r["source"] == "torch" for r in plan["conflicts"])
    assert not target(h8).exists()


@pytest.mark.parametrize("replacement", ["tensorflow", "torch", "hailo-dataflow-compiler"])
def test_t34_38_framework_wheels_rejected(h8, replacement):
    plan = hp.build_plan(inventory(h8), target(h8), packages=["nvidia-cuda-runtime-cu12"],
                         wheels=[wheel(h8, name=replacement)])
    assert plan["status"] == "BLOCKED"
    assert any(r["reason"] == "wheel_rejected" for r in plan["conflicts"])


def test_t34_38_solver_added_or_wrong_version_stops(h8):
    plan = ready_plan(h8, version="12.8.90")
    assert plan["status"] == "BLOCKED"
    assert any(r["reason"] == "unreviewed_solver_change_or_wrong_version" for r in plan["conflicts"])


def test_t34_38_dependency_closure_required(h8):
    plan = ready_plan(h8, requires=["nvidia-nvjitlink-cu12==12.5.82"])
    assert plan["status"] == "BLOCKED"
    assert any(r["reason"] == "supplement_dependency_missing_or_conflicting" for r in plan["conflicts"])


def test_t34_38_existing_packages_never_shadowed(h8):
    metadata(h8[1], "nvidia-cuda-runtime-cu12", "12.8.90")
    assert any(r["reason"] == "existing_package_must_not_be_shadowed" for r in ready_plan(h8)["conflicts"])


@pytest.mark.parametrize("suffix", ["cp39-cp39-manylinux_2_17_x86_64", "py3-none-win_amd64", "py3-none-manylinux_999_0_x86_64"])
def test_t34_38_incompatible_wheel_tags_rejected(h8, suffix):
    plan = ready_plan(h8, suffix=suffix)
    assert plan["status"] == "BLOCKED"
    assert any(r["reason"] == "wheel_target_python_or_platform_incompatible" for r in plan["conflicts"])


def test_t34_38_success_only_private_staging_and_framework_unchanged(h8):
    before = {str(p): p.read_bytes() for p in h8[0].rglob("*") if p.is_file() and not p.is_symlink()}
    manifest = stage(h8)
    assert manifest.exists()
    assert {str(p): p.read_bytes() for p in h8[0].rglob("*") if p.is_file() and not p.is_symlink()} == before
    assert json.loads(manifest.read_text())["gpu_readiness"] == "NOT_PROVEN"
    assert json.loads(manifest.read_text())["hardware_execution"] == "NOT_RUN"


def test_t34_38_stale_venv_stops_before_any_write(h8):
    plan = ready_plan(h8)
    path, digest = save_plan(h8, plan)
    metadata(h8[1], "torch", "2.9.1", ["nvidia-cuda-runtime-cu12==12.8.90"])
    with pytest.raises(ValueError, match="metadata_changed_since_review"):
        hp.stage_reviewed_plan(plan, expected_plan_sha256=digest, plan_file=path)
    assert not target(h8).exists()


def test_t34_38_changed_review_bytes_stops(h8):
    plan = ready_plan(h8)
    path, digest = save_plan(h8, plan)
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="reviewed_plan_bytes_changed"):
        hp.stage_reviewed_plan(plan, expected_plan_sha256=digest, plan_file=path)
    assert not target(h8).exists()


def test_t34_38_changed_wheel_stops(h8):
    plan = ready_plan(h8)
    path, digest = save_plan(h8, plan)
    wheel(h8, extra_files={"nvidia/extra.txt": "changed"})
    with pytest.raises(ValueError, match="reviewed_plan_or_wheel_changed"):
        hp.stage_reviewed_plan(plan, expected_plan_sha256=digest, plan_file=path)
    assert not target(h8).exists()


@pytest.mark.parametrize("name", ["evil.pth", "sitecustomize.py", "../../outside", "anything.data/purelib/foo.py", "tensorflow/__init__.py"])
def test_t34_39_hooks_traversal_framework_paths_rejected(h8, name):
    plan = ready_plan(h8, extra_files={name: "unsafe"})
    assert plan["status"] == "BLOCKED"
    assert not target(h8).exists()


def test_t34_39_child_only_library_paths_no_namespace_hooks(h8):
    manifest = stage(h8)
    base = {"LD_LIBRARY_PATH": "/existing/hailo8/lib", "PYTHONPATH": "/tool", "CUDA_VISIBLE_DEVICES": "0"}
    actual = hp.child_library_environment(base, family="hailo8", selected_python=h8[0] / "bin/python", manifest_path=manifest)
    assert actual is not base
    assert base["LD_LIBRARY_PATH"] == "/existing/hailo8/lib"
    assert actual["LD_LIBRARY_PATH"].endswith(":/existing/hailo8/lib")
    assert actual["PYTHONPATH"] == base["PYTHONPATH"]
    assert actual["CUDA_VISIBLE_DEVICES"] == base["CUDA_VISIBLE_DEVICES"]
    assert "hailo8_cuda_reviewed/packages/nvidia/cuda_runtime/lib" in actual["LD_LIBRARY_PATH"]
    assert hp.validated_overlay_components(family="hailo8", selected_python=h8[0] / "bin/python", manifest_path=manifest) is None


def test_t34_39_only_reviewed_complete_nvcc_pair_available(h8):
    tf_metadata = h8[1] / "tensorflow-2.18.0.dist-info/METADATA"
    tf_metadata.write_text(tf_metadata.read_text() + 'Requires-Dist: nvidia-cuda-nvcc-cu12==12.5.82; extra == "and-cuda"\n')
    runtime = wheel(h8)
    nvcc = wheel(h8, name="nvidia-cuda-nvcc-cu12", extra_files={
        "nvidia/cuda_nvcc/bin/ptxas": "#!/bin/sh\nexit 0\n",
        "nvidia/cuda_nvcc/nvvm/libdevice/libdevice.10.bc": "synthetic bitcode, no hardware execution"})
    plan = hp.build_plan(inventory(h8), target(h8),
                         packages=["nvidia-cuda-runtime-cu12", "nvidia-cuda-nvcc-cu12"], wheels=[runtime, nvcc])
    manifest = stage(h8, plan)
    # The synthetic wheel intentionally carries no executable mode initially.
    with pytest.raises(ValueError, match="ptxas_not_executable"):
        hp.validated_overlay_components(family="hailo8", selected_python=h8[0] / "bin/python", manifest_path=manifest)
    ptxas = manifest.parent / "packages/nvidia/cuda_nvcc/bin/ptxas"
    ptxas.chmod(0o700)
    result = hp.validated_overlay_components(family="hailo8", selected_python=h8[0] / "bin/python", manifest_path=manifest)
    assert result["ptxas_path"] == str(ptxas)
    assert result["component_root"] == str(ptxas.parent.parent)
    Path(result["libdevice_path"]).write_bytes(b"")
    with pytest.raises(ValueError, match="component_missing_or_empty"):
        hp.validated_overlay_components(family="hailo8", selected_python=h8[0] / "bin/python", manifest_path=manifest)


def test_t34_39_real_resolver_uses_reviewed_overlay_with_external_assembler_probe(h8):
    from onnx_splitpoint_tool.hailo_compiler_context import resolve_hailo_compiler_context
    tf_metadata = h8[1] / "tensorflow-2.18.0.dist-info/METADATA"
    tf_metadata.write_text(tf_metadata.read_text() + 'Requires-Dist: nvidia-cuda-nvcc-cu12==12.5.82; extra == "and-cuda"\n')
    ptxas_code = ("#!" + sys.executable + "\nimport sys,pathlib\n"
                  "if '--version' in sys.argv: print('external fixture ptxas 12.5.82')\n"
                  "else: pathlib.Path(sys.argv[sys.argv.index('-o')+1]).write_bytes(b'external assembler fixture output')\n")
    nvcc = wheel(h8, name="nvidia-cuda-nvcc-cu12", extra_files={
        "nvidia/cuda_nvcc/bin/ptxas": ptxas_code,
        "nvidia/cuda_nvcc/nvvm/libdevice/libdevice.10.bc": "synthetic target fixture"})
    # The executable mode comes from the reviewed wheel, not a later mutation.
    with zipfile.ZipFile(nvcc) as archive:
        contents = [(info, archive.read(info)) for info in archive.infolist()]
    with zipfile.ZipFile(nvcc, "w") as archive:
        for info, data in contents:
            if info.filename.endswith("/bin/ptxas"):
                info.external_attr = (0o100755 << 16)
            archive.writestr(info, data)
    plan = hp.build_plan(inventory(h8), target(h8), packages=["nvidia-cuda-nvcc-cu12"], wheels=[nvcc])
    manifest = stage(h8, plan)
    commands = h8[0].parent / "external_probe_commands"
    commands.mkdir()
    smi = commands / "nvidia-smi"
    smi.write_text("#!" + sys.executable + "\nprint('0, GPU-EXTERNAL-FIXTURE, 6.1')\n")
    smi.chmod(0o700)
    parent = {"PATH": str(commands), "ONNX_SPLITPOINT_HAILO8_DEPENDENCY_MANIFEST": str(manifest)}
    actual = resolve_hailo_compiler_context(str(h8[0] / "bin/python"), "hailo8",
        job_override={"device": "gpu", "gpu_selector": "0"}, parent_env=parent)
    assert actual["component_source"] == "selected_hailo8_dependency_overlay"
    assert actual["ptxas_path"] == str(manifest.parent / "packages/nvidia/cuda_nvcc/bin/ptxas")
    assert actual["target_probe"]["output_size"] > 0
    assert actual["target_arch"] == "sm_61"
    assert "LD_LIBRARY_PATH" not in parent


@pytest.mark.parametrize("family", ["hailo10", "hailo10h", "deepx"])
def test_t34_39_never_borrow_h8_overlay_from_other_family(h8, family):
    manifest = stage(h8)
    with pytest.raises(ValueError, match="cannot_be_used_by_other_family"):
        hp.child_library_environment({}, family=family, selected_python=h8[0] / "bin/python", manifest_path=manifest)


def test_t34_39_new_framework_metadata_invalidates_selection(h8):
    manifest = stage(h8)
    metadata(h8[1], "torch", "2.9.1")
    with pytest.raises(ValueError, match="distribution_inventory_changed"):
        hp.child_library_environment({}, family="hailo8", selected_python=h8[0] / "bin/python", manifest_path=manifest)


def test_t34_39_modified_overlay_rejected(h8):
    manifest = stage(h8)
    (manifest.parent / "packages/danger.pth").write_text("import bad\n")
    with pytest.raises(ValueError, match="hook_or_symlink_rejected"):
        hp.child_library_environment({}, family="hailo8", selected_python=h8[0] / "bin/python", manifest_path=manifest)


def test_t34_40_and_41_metadata_and_stage_never_release_gpu(h8):
    plan = ready_plan(h8)
    manifest = stage(h8, plan)
    assert plan["hardware_execution"] == "NOT_RUN"
    assert plan["gpu_readiness"] == "NOT_PROVEN"
    assert plan["next_gates"] == ["fresh_hailo8_sdk_import", "hailo8_eager_matmul_and_conv2d", "hailo8_xla",
                                  "hailo8_model_build", "normal_hailo8_reuse_after_restart"]
    assert json.loads(manifest.read_text())["gpu_readiness"] == "NOT_PROVEN"


def test_t34_40_hailo10_metadata_not_hailo8_proof(h8):
    result = inventory(h8)
    result["distributions"][0]["version"] = "5.3.0"
    with pytest.raises(ValueError, match="no_hailo10_borrowing"):
        hp.build_plan(result, target(h8))


def test_t34_42_deselect_and_remove_private_target_preserves_cpu(h8):
    original = inventory(h8)
    manifest = stage(h8)
    base = {"PATH": "/user/bin", "LD_LIBRARY_PATH": "/old/lib"}
    hp.child_library_environment(base, family="hailo8", selected_python=h8[0] / "bin/python", manifest_path=manifest)
    shutil.rmtree(manifest.parent)
    assert base == {"PATH": "/user/bin", "LD_LIBRARY_PATH": "/old/lib"}
    assert inventory(h8) == original


def test_t34_42_no_installer_or_network_entrypoint_in_normal_import(h8):
    script = Path(__file__).resolve().parents[1] / "scripts/hailo8_dependency_plan_v27934.py"
    output = h8[0].parent / "cli_review.json"
    result = subprocess.run([sys.executable, "-I", "-B", str(script), "plan", "--python", str(h8[0] / "bin/python"),
                             "--target", str(target(h8)), "--output", str(output)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "PLAN_STATUS=REVIEW_REQUIRED" in result.stdout
    assert "HAILO8_GPU_READINESS=NOT_PROVEN" in result.stdout
    report = json.loads(output.read_text())
    assert report["network_used"] is False and report["package_manager_used"] is False
    assert report["selected_packages"] == []
    assert not target(h8).exists()


def test_t34_42_existing_target_and_vendor_target_rejected(h8):
    with pytest.raises(ValueError, match="must_be_new_named_sibling"):
        hp.build_plan(inventory(h8), h8[0] / "hailo8_cuda_bad")
    target(h8).mkdir()
    with pytest.raises(ValueError, match="already_exists"):
        hp.build_plan(inventory(h8), target(h8))
