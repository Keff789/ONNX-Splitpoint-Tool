from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from onnx_splitpoint_tool.resume_artifact_rehydration import (
    ArtifactRequirement,
    ResumeArtifactResolutionError,
    build_resume_artifact_stage_map,
    requirements_from_contract_artifacts,
)


REMOTE_RUN_ROOT = "/home/nx/native_fifo_evalsets/example_run"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _payload_manifest(
    *,
    payload_remote_path: str,
    payload: bytes,
) -> bytes:
    rows = [
        {
            "role": "outputs[0]",
            "path": payload_remote_path,
            "sha256": _sha256(payload),
            "size_bytes": len(payload),
        }
    ]
    body = {
        "schema": "onnx-splitpoint/runner-output-dump",
        "schema_version": 4,
        "payload_artifacts": rows,
        "payload_artifacts_sha256": _sha256(
            json.dumps(
                rows,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
        ),
    }
    return json.dumps(body, indent=2, sort_keys=True).encode("utf-8")


def test_resolves_run_mirror_and_expands_sealed_manifest_payload(
    tmp_path: Path,
) -> None:
    mirror = tmp_path / "old-run-mirror"
    remote_dir = (
        f"{REMOTE_RUN_ROOT}/yolo26s/benchmark_set/native_pipeline/b038/"
        "hailo10h_to_trt/float32_layout_fp16/native_outputs"
    )
    payload_remote = f"{remote_dir}/output_00_output0.bin"
    manifest_remote = f"{remote_dir}/native_outputs_manifest.json"
    payload = b"exact semantic output bytes"
    manifest = _payload_manifest(
        payload_remote_path=payload_remote,
        payload=payload,
    )

    local_dir = (
        mirror
        / "native_producers"
        / "hailo10h"
        / "yolo26s"
        / "benchmark_set"
        / "native_pipeline"
        / "b038"
        / "hailo10h_to_trt"
        / "float32_layout_fp16"
        / "native_outputs"
    )
    _write(local_dir / "native_outputs_manifest.json", manifest)
    _write(local_dir / "output_00_output0.bin", payload)

    contract = {
        "artifacts": {
            "semantic_output_manifest": {
                "path": manifest_remote,
                "sha256": _sha256(manifest),
                "size_bytes": len(manifest),
            }
        }
    }
    requirements = requirements_from_contract_artifacts(
        contract, roles=("semantic_output_manifest",),
    )
    stage_map = build_resume_artifact_stage_map(
        requirements,
        run_mirror_roots=(mirror,),
        artifact_store_roots=(),
        allowed_remote_roots=(REMOTE_RUN_ROOT,),
    )

    assert stage_map["status"] == "ready"
    assert stage_map["artifact_count"] == 2
    assert stage_map["total_bytes"] == len(manifest) + len(payload)
    assert len(stage_map["stage_map_sha256"]) == 64
    assert stage_map["source_revalidation_required_before_transport"] is True
    repeated = build_resume_artifact_stage_map(
        requirements,
        run_mirror_roots=(mirror,),
        artifact_store_roots=(),
        allowed_remote_roots=(REMOTE_RUN_ROOT,),
    )
    assert repeated["stage_map_sha256"] == stage_map["stage_map_sha256"]
    entries = {row["remote_path"]: row for row in stage_map["entries"]}
    assert set(entries) == {manifest_remote, payload_remote}
    assert entries[manifest_remote]["source_kind"] == "run_mirror"
    assert entries[payload_remote]["source_kind"] == "run_mirror"
    assert entries[payload_remote]["sha256"] == _sha256(payload)
    assert entries[payload_remote]["size_bytes"] == len(payload)
    assert entries[payload_remote]["roles"] == [
        "semantic_output_manifest.payload[0]:outputs[0]"
    ]


def test_uses_exact_content_addressed_store_object_when_mirror_is_stale(
    tmp_path: Path,
) -> None:
    expected = b"old exact HEF bytes"
    digest = _sha256(expected)
    mirror = tmp_path / "mirror"
    _write(mirror / "yolov7_paper" / "full" / "compiled.hef", b"new HEF")
    store = tmp_path / "artifact-store"
    object_path = _write(
        store / "objects" / "sha256" / digest[:2] / digest / "compiled.hef",
        expected,
    )
    remote = f"{REMOTE_RUN_ROOT}/yolov7_paper/benchmark_set/hailo/full/compiled.hef"

    stage_map = build_resume_artifact_stage_map(
        [
            ArtifactRequirement(
                role="hef",
                remote_path=remote,
                sha256=digest,
                # Legacy full contracts bind the HEF hash but may not carry a
                # separate size.  A matching SHA still binds the exact bytes;
                # the observed size is recorded for transport revalidation.
                size_bytes=None,
            )
        ],
        run_mirror_roots=(mirror,),
        artifact_store_roots=(store,),
        allowed_remote_roots=(REMOTE_RUN_ROOT,),
    )

    assert stage_map["artifact_count"] == 1
    entry = stage_map["entries"][0]
    assert entry["source_kind"] == "artifact_store"
    assert Path(entry["source_path"]) == object_path.resolve()
    assert entry["remote_path"] == remote
    assert entry["sha256"] == digest
    assert entry["size_bytes"] == len(expected)
    assert entry["expected_size_bytes"] is None


def test_contract_extraction_is_explicit_and_accepts_legacy_bytes_field() -> None:
    contract = {
        "artifacts": {
            "runtime_input_tensor": {
                "path": f"{REMOTE_RUN_ROOT}/model/runtime_input.bin",
                "sha256": "a" * 64,
                "bytes": 1228800,
            },
            "python_executable": {
                "path": "/usr/bin/python3",
                "sha256": "b" * 64,
            },
        }
    }

    requirements = requirements_from_contract_artifacts(
        contract, roles=("runtime_input_tensor",),
    )

    assert requirements == [
        ArtifactRequirement(
            role="runtime_input_tensor",
            remote_path=f"{REMOTE_RUN_ROOT}/model/runtime_input.bin",
            sha256="a" * 64,
            size_bytes=1228800,
        )
    ]
    with pytest.raises(
        ResumeArtifactResolutionError,
        match="contract_artifact_missing",
    ):
        requirements_from_contract_artifacts(
            contract, roles=("not_present",),
        )


def test_identical_target_requirements_coalesce_roles(tmp_path: Path) -> None:
    payload = b"one immutable object"
    digest = _sha256(payload)
    mirror = tmp_path / "mirror"
    _write(mirror / "model" / "object.bin", payload)
    remote = f"{REMOTE_RUN_ROOT}/model/object.bin"

    result = build_resume_artifact_stage_map(
        [
            ArtifactRequirement("primary", remote, digest, len(payload)),
            ArtifactRequirement("alias", remote, digest, len(payload)),
        ],
        run_mirror_roots=(mirror,),
        artifact_store_roots=(),
        allowed_remote_roots=(REMOTE_RUN_ROOT,),
    )

    assert result["artifact_count"] == 1
    assert result["entries"][0]["roles"] == ["alias", "primary"]


@pytest.mark.parametrize(
    ("expected_size", "source_bytes"),
    [
        (4, b"wrong hash"),
        (99, b"right bytes"),
    ],
)
def test_fails_closed_on_hash_or_size_mismatch(
    tmp_path: Path,
    expected_size: int,
    source_bytes: bytes,
) -> None:
    mirror = tmp_path / "mirror"
    remote = f"{REMOTE_RUN_ROOT}/model/compiled.hef"
    expected = b"right bytes"
    _write(mirror / "model" / "compiled.hef", source_bytes)

    with pytest.raises(
        ResumeArtifactResolutionError,
        match="exact_source_not_found",
    ):
        build_resume_artifact_stage_map(
            [
                ArtifactRequirement(
                    role="hef",
                    remote_path=remote,
                    sha256=_sha256(expected),
                    size_bytes=expected_size,
                )
            ],
            run_mirror_roots=(mirror,),
            artifact_store_roots=(),
            allowed_remote_roots=(REMOTE_RUN_ROOT,),
        )


@pytest.mark.parametrize(
    "remote_path",
    [
        "relative/artifact.hef",
        "/home/nx/native_fifo_evalsets/example_run/../other/artifact.hef",
        "/etc/ld.so.preload",
        REMOTE_RUN_ROOT,
    ],
)
def test_rejects_unsafe_or_out_of_scope_remote_targets(
    tmp_path: Path,
    remote_path: str,
) -> None:
    payload = b"artifact"
    with pytest.raises(ResumeArtifactResolutionError):
        build_resume_artifact_stage_map(
            [
                ArtifactRequirement(
                    role="hef",
                    remote_path=remote_path,
                    sha256=_sha256(payload),
                    size_bytes=len(payload),
                )
            ],
            run_mirror_roots=(tmp_path / "mirror",),
            artifact_store_roots=(),
            allowed_remote_roots=(REMOTE_RUN_ROOT,),
        )


def test_rejects_conflicting_identities_for_one_remote_target(
    tmp_path: Path,
) -> None:
    remote = f"{REMOTE_RUN_ROOT}/model/compiled.hef"
    with pytest.raises(
        ResumeArtifactResolutionError,
        match="conflicting_remote_target",
    ):
        build_resume_artifact_stage_map(
            [
                ArtifactRequirement("hef", remote, "a" * 64, 10),
                ArtifactRequirement("other_hef", remote, "b" * 64, 10),
            ],
            run_mirror_roots=(tmp_path / "mirror",),
            artifact_store_roots=(),
            allowed_remote_roots=(REMOTE_RUN_ROOT,),
        )


def test_rejects_symlink_source_even_when_target_bytes_match(
    tmp_path: Path,
) -> None:
    if not hasattr(os, "symlink"):
        pytest.skip("symlinks are not supported")
    expected = b"exact"
    outside = _write(tmp_path / "outside.hef", expected)
    mirror = tmp_path / "mirror"
    mirror.mkdir()
    link = mirror / "compiled.hef"
    link.symlink_to(outside)
    remote = f"{REMOTE_RUN_ROOT}/compiled.hef"

    with pytest.raises(
        ResumeArtifactResolutionError,
        match="exact_source_not_found",
    ):
        build_resume_artifact_stage_map(
            [ArtifactRequirement("hef", remote, _sha256(expected), len(expected))],
            run_mirror_roots=(mirror,),
            artifact_store_roots=(),
            allowed_remote_roots=(REMOTE_RUN_ROOT,),
        )


def test_rejects_tampered_payload_artifact_set_before_staging(
    tmp_path: Path,
) -> None:
    mirror = tmp_path / "mirror"
    remote_dir = f"{REMOTE_RUN_ROOT}/model/native_outputs"
    payload_remote = f"{remote_dir}/output.bin"
    payload = b"output"
    manifest_obj = json.loads(
        _payload_manifest(
            payload_remote_path=payload_remote,
            payload=payload,
        )
    )
    manifest_obj["payload_artifacts_sha256"] = "0" * 64
    manifest = json.dumps(manifest_obj, sort_keys=True).encode("utf-8")
    _write(mirror / "model" / "native_outputs" / "manifest.json", manifest)
    _write(mirror / "model" / "native_outputs" / "output.bin", payload)

    with pytest.raises(
        ResumeArtifactResolutionError,
        match="payload_artifacts_sha256_mismatch",
    ):
        build_resume_artifact_stage_map(
            [
                ArtifactRequirement(
                    "semantic_output_manifest",
                    f"{remote_dir}/manifest.json",
                    _sha256(manifest),
                    len(manifest),
                )
            ],
            run_mirror_roots=(mirror,),
            artifact_store_roots=(),
            allowed_remote_roots=(REMOTE_RUN_ROOT,),
        )
