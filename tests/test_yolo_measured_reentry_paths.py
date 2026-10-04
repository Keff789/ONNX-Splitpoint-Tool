"""Published Native variant artifacts keep the existing strict path boundary."""
from pathlib import Path
import shutil
import pytest

from tests.test_v27924_terminal_hailo_aliases import published
from onnx_splitpoint_tool.workflow.runner import WorkflowRunTargetError


def collected(published):
    runner, alias, _ = published
    destination = (runner.run_dir / "native_producers/variants/case0000_hailo8_yolo11l_b120"
                   / "hailo8/yolo11l/benchmark_set/b120/hailo/hailo8/part1")
    destination.parent.mkdir(parents=True)
    shutil.copytree(alias.parent, destination, symlinks=True)
    return runner, destination


def test_normal_published_variant_aliases_are_read_only_resume_inputs(published):
    runner, destination = collected(published)
    links = [p for p in destination.iterdir() if p.is_symlink()]
    assert {p.name for p in links} == {".hailo-current", "compiled.hef", "cache_meta.json", "hailo_hef_build_receipt.json"}
    before = {p: p.readlink() for p in links}
    assert all(runner._resume_read_only_hailo_bundle_alias(p) for p in links)
    runner._validate_resume_target_paths()
    assert {p: p.readlink() for p in links} == before


@pytest.mark.parametrize("damage", ["external_pointer", "external_hef", "unexpected_member", "output", "control", "wrong_layout"])
def test_variant_alias_admission_keeps_output_control_and_bundle_guards(published, tmp_path, damage):
    runner, destination = collected(published)
    target = tmp_path / "external.json"
    target.write_text("{}")
    if damage == "external_pointer":
        p = destination / ".hailo-current"; p.unlink(); p.symlink_to(tmp_path)
    elif damage == "external_hef":
        p = destination / "compiled.hef"; p.unlink(); p.symlink_to(target)
    elif damage == "unexpected_member":
        (destination / "result.json").symlink_to(target)
    elif damage == "output":
        p = destination.parents[3] / "native_pipeline/b120/results.json"
        p.parent.mkdir(parents=True); p.symlink_to(target)
    elif damage == "control":
        p = runner.run_dir / "jobs/workflow_control.json"
        p.parent.mkdir(); p.symlink_to(target)
    else:
        wrong = runner.run_dir / "native_producers/variants/case0000/reports/hailo/hailo8/part1"
        wrong.parent.mkdir(parents=True)
        shutil.copytree(destination, wrong, symlinks=True)
    with pytest.raises(WorkflowRunTargetError, match="symlink/non-regular"):
        runner._validate_resume_target_paths()
