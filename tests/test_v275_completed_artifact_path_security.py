from __future__ import annotations

from pathlib import Path

import pytest

from onnx_splitpoint_tool.native_detection_postprocess import (
    FrozenPostprocessError,
    canonical_json_sha256,
    persist_completed_result_artifact,
)
from scripts import native_trt_full_completed_hotloop as trt_hotloop
from scripts import smoke_hailo10_hef_runner as hailo_runner


def _artifact() -> dict[str, object]:
    return {
        "schema": "test/completed-detection-result",
        "schema_version": 1,
        "detections": [],
    }


def _persist(output_path: Path) -> dict[str, object]:
    artifact = _artifact()
    return persist_completed_result_artifact(
        artifact,
        expected_sha256=canonical_json_sha256(artifact),
        output_path=output_path,
    )


def test_completed_result_writer_persists_exact_regular_file(
    tmp_path: Path,
) -> None:
    output = tmp_path / "results" / "completed.json"

    evidence = _persist(output)

    assert output.is_file()
    assert not output.is_symlink()
    assert evidence["completed_task_result_artifact_path"] == str(output)
    assert evidence["completed_task_result_artifact_saved"] is True
    assert evidence["completed_task_result_artifact_file_sha256"] == (
        canonical_json_sha256(_artifact())
    )


def test_completed_result_writer_rejects_leaf_symlink_without_touching_target(
    tmp_path: Path,
) -> None:
    target = tmp_path / "outside.json"
    target.write_text("sentinel", encoding="utf-8")
    output = tmp_path / "completed.json"
    output.symlink_to(target)

    with pytest.raises(
        FrozenPostprocessError,
        match="completed_result_artifact_output_path_unsafe",
    ):
        _persist(output)

    assert output.is_symlink()
    assert target.read_text(encoding="utf-8") == "sentinel"


def test_completed_result_writer_rejects_symlinked_parent(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    linked_parent = tmp_path / "linked-results"
    linked_parent.symlink_to(outside, target_is_directory=True)

    with pytest.raises(
        FrozenPostprocessError,
        match="completed_result_artifact_output_path_unsafe",
    ):
        _persist(linked_parent / "completed.json")

    assert not (outside / "completed.json").exists()


def test_completed_result_writer_rejects_non_file_destination(
    tmp_path: Path,
) -> None:
    output = tmp_path / "completed.json"
    output.mkdir()

    with pytest.raises(
        FrozenPostprocessError,
        match="completed_result_artifact_output_path_unsafe",
    ):
        _persist(output)


@pytest.mark.parametrize(
    "private_writer",
    [
        trt_hotloop._persist_completed_result_artifact,
        hailo_runner._persist_completed_result_artifact,
    ],
)
def test_private_hotloop_writers_do_not_follow_predictable_temp_symlink(
    tmp_path: Path,
    private_writer: object,
) -> None:
    artifact = _artifact()
    expected_sha256 = canonical_json_sha256(artifact)
    output = tmp_path / "completed.json"
    outside = tmp_path / "outside.json"
    outside.write_text("sentinel", encoding="utf-8")
    predictable_temporary = output.with_name(output.name + ".tmp")
    predictable_temporary.symlink_to(outside)

    evidence = private_writer(  # type: ignore[operator]
        artifact,
        expected_sha256,
        output,
    )

    assert evidence["saved"] is True
    assert evidence["file_sha256"] == expected_sha256
    assert output.is_file()
    assert not output.is_symlink()
    assert predictable_temporary.is_symlink()
    assert outside.read_text(encoding="utf-8") == "sentinel"


def test_private_hotloop_writer_resource_mirrors_are_exact() -> None:
    root = Path(__file__).resolve().parents[1]
    for name in (
        "native_trt_full_completed_hotloop.py",
        "smoke_hailo10_hef_runner.py",
    ):
        assert (root / "scripts" / name).read_bytes() == (
            root / "onnx_splitpoint_tool" / "resources"
            / "remote_scripts" / name
        ).read_bytes()
