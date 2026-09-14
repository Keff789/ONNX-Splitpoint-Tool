from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any

import pytest

import onnx_splitpoint_tool.resume_remote_rehydration as remote
from onnx_splitpoint_tool.resume_artifact_rehydration import (
    ArtifactRequirement,
    build_resume_artifact_stage_map,
)


ROOT = "/home/nx/native_fifo_evalsets/run-20260727"
SSH = "nx@192.168.0.145"


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")).hexdigest()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _source_stat(path: Path) -> dict[str, int]:
    observed = path.stat()
    return {
        "device": observed.st_dev,
        "inode": observed.st_ino,
        "mtime_ns": observed.st_mtime_ns,
        "size_bytes": observed.st_size,
    }


def _stage_map(
    entries: list[dict[str, Any]],
    *,
    allowed_roots: list[str] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": "onnx-splitpoint/resume-artifact-stage-map",
        "schema_version": 1,
        "status": "ready",
        "local_only": True,
        "transport_performed": False,
        "source_revalidation_required_before_transport": True,
        "allowed_remote_roots": allowed_roots or [ROOT],
        "artifact_count": len(entries),
        "total_bytes": sum(int(entry["size_bytes"]) for entry in entries),
        "entries": entries,
    }
    payload["stage_map_sha256"] = _canonical_sha256(payload)
    return payload


def _entry(
    source: Path,
    remote_path: str,
    data: bytes,
    *,
    source_stat: bool = True,
) -> dict[str, Any]:
    return {
        "roles": ["compiled_model"],
        "remote_path": remote_path,
        "source_path": str(source),
        "source_kind": "artifact_store",
        "source_root": str(source.parent),
        "resolution": "content_addressed_sha256_object",
        "sha256": _sha(data),
        "size_bytes": len(data),
        "expected_size_bytes": len(data),
        "source_stat": _source_stat(source) if source_stat else None,
    }


class _Runner:
    def __init__(self, responses: list[tuple[str, str, int]]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def __call__(self, command: list[str], **kwargs: Any) -> Any:
        assert self.responses, f"unexpected command: {command!r}"
        marker, stdout, returncode = self.responses.pop(0)
        assert marker in command[-1]
        stdin = kwargs.get("stdin")
        uploaded = stdin.read() if stdin is not None else None
        self.calls.append({
            "command": command,
            "stdin": uploaded,
            "timeout": kwargs.get("timeout"),
        })
        return subprocess.CompletedProcess(
            command,
            returncode,
            stdout=stdout.encode("utf-8"),
            stderr=b"" if returncode == 0 else b"remote failure",
        )


class _LocalSshRunner:
    """Execute the generated remote Bash command in a local filesystem sandbox."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, command: list[str], **kwargs: Any) -> Any:
        remote_argv = shlex.split(command[-1])
        marker = next(
            (
                line.strip()
                for line in remote_argv[2].splitlines()
                if "ONNX_SPLITPOINT_RESUME_REMOTE_" in line
            ),
            "",
        )
        self.calls.append({
            "command": command,
            "marker": marker,
            "has_stdin": kwargs.get("stdin") is not None,
        })
        return subprocess.run(
            remote_argv,
            stdin=kwargs.get("stdin"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=kwargs.get("timeout"),
        )


def _probe(
    status: str,
    digest: str = "-",
    size: str = "-",
    *,
    storage_device: str = "11",
    storage_inode: str = "12",
    root_device: str = "21",
    root_inode: str = "22",
) -> str:
    if status == "root_missing":
        root_device = "-"
        root_inode = "-"
    return (
        f"OSPRPROBE1\t{status}\t{digest}\t{size}\t"
        f"{storage_device}\t{storage_inode}\t"
        f"{root_device}\t{root_inode}\n"
    )


def _upload(status: str, digest: str = "-", size: str = "-") -> str:
    return f"OSPRUPLOAD1\t{status}\t{digest}\t{size}\n"


def _commit(
    status: str,
    digest: str = "-",
    size: str = "-",
    backup: str = "-",
    old_digest: str = "-",
    old_size: str = "-",
) -> str:
    return (
        f"OSPRCOMMIT1\t{status}\t{digest}\t{size}\t"
        f"{backup}\t{old_digest}\t{old_size}\n"
    )


def test_read_only_probe_filters_exact_missing_and_mismatch() -> None:
    first = b"a"
    second = b"bb"
    third = b"ccc"
    requirements = [
        {
            "remote_path": f"{ROOT}/z.bin",
            "sha256": _sha(third),
            "size_bytes": len(third),
        },
        {
            "remote_path": f"{ROOT}/a.bin",
            "sha256": _sha(first),
            "size_bytes": len(first),
        },
        {
            "remote_path": f"{ROOT}/m.bin",
            "sha256": _sha(second),
            "size_bytes": len(second),
        },
    ]
    runner = _Runner([
        ("ONNX_SPLITPOINT_RESUME_REMOTE_PROBE_V1",
         _probe("regular", _sha(first), str(len(first))), 0),
        ("ONNX_SPLITPOINT_RESUME_REMOTE_PROBE_V1",
         _probe("missing"), 0),
        ("ONNX_SPLITPOINT_RESUME_REMOTE_PROBE_V1",
         _probe("regular", "f" * 64, str(len(third))), 0),
    ])

    report = remote.probe_remote_requirements(
        requirements,
        ssh_target=SSH,
        remote_run_root=ROOT,
        runner=runner,
    )

    assert report["ok"] is True
    assert report["read_only"] is True
    assert report["exact_count"] == 1
    assert report["rehydration_required_count"] == 2
    assert report["all_exact"] is False
    assert [
        entry["remote_status"] for entry in report["entries"]
    ] == ["exact", "missing", "mismatch"]
    assert all(call["stdin"] is None for call in runner.calls)
    assert all(
        call["command"][:7] == [
            "ssh",
            "-T",
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=accept-new",
            SSH,
        ]
        for call in runner.calls
    )
    assert not runner.responses


def test_single_probe_is_public_and_read_only() -> None:
    data = b"manifest"
    path = f"{ROOT}/model/native_outputs/manifest.json"
    runner = _Runner([
        ("ONNX_SPLITPOINT_RESUME_REMOTE_PROBE_V1",
         _probe("regular", _sha(data), str(len(data))), 0),
    ])

    report = remote.probe_remote_artifact(
        {
            "remote_path": path,
            "sha256": _sha(data),
            "size_bytes": len(data),
            "role": "semantic_output_manifest",
        },
        ssh_target=SSH,
        remote_run_root=ROOT,
        runner=runner,
    )

    assert report["exact"] is True
    assert report["remote_status"] == "exact"
    assert report["roles"] == ["semantic_output_manifest"]
    assert report["read_only"] is True


def test_raw_probe_accepts_sha_only_and_uses_size_as_diagnostic() -> None:
    data = b"legacy hef bytes with no size in old contract"
    runner = _Runner([
        ("ONNX_SPLITPOINT_RESUME_REMOTE_PROBE_V1",
         _probe("regular", _sha(data), str(len(data))), 0),
    ])

    report = remote.probe_remote_artifact(
        {
            "remote_path": f"{ROOT}/yolov7/full/compiled.hef",
            "sha256": _sha(data),
        },
        ssh_target=SSH,
        remote_run_root=ROOT,
        runner=runner,
    )

    assert report["exact"] is True
    assert report["expected_size_bytes"] is None
    assert report["remote_size_bytes"] == len(data)


def test_raw_probe_accepts_local_resolver_requirement_objects() -> None:
    data = b"edb441 legacy hef"
    requirement = ArtifactRequirement(
        role="compiled_model",
        remote_path=f"{ROOT}/yolov7/full/compiled.hef",
        sha256=_sha(data),
        size_bytes=None,
    )
    runner = _Runner([
        ("ONNX_SPLITPOINT_RESUME_REMOTE_PROBE_V1",
         _probe("regular", _sha(data), str(len(data))), 0),
    ])

    report = remote.probe_remote_requirements(
        [requirement],
        ssh_target=SSH,
        remote_run_root=ROOT,
        runner=runner,
    )

    assert report["all_exact"] is True
    assert report["entries"][0]["roles"] == ["compiled_model"]
    assert report["entries"][0]["expected_size_bytes"] is None


@pytest.mark.parametrize(
    "ssh_target",
    [
        "-oProxyCommand=bad",
        "nx@host other-command",
        "nx@host:22",
        "nx@host\nbad",
        "",
    ],
)
def test_ssh_target_is_validated_before_subprocess(
    ssh_target: str,
) -> None:
    runner = _Runner([])
    with pytest.raises(
        remote.RemoteArtifactRehydrationError,
        match="invalid_ssh_target",
    ):
        remote.probe_remote_artifact(
            {
                "remote_path": f"{ROOT}/a",
                "sha256": "a" * 64,
                "size_bytes": 1,
            },
            ssh_target=ssh_target,
            remote_run_root=ROOT,
            runner=runner,
        )
    assert runner.calls == []


@pytest.mark.parametrize(
    "path",
    [
        "/home/nx/native_fifo_evalsets/outside.bin",
        f"{ROOT}",
        f"{ROOT}/../escape.bin",
        "relative.bin",
        f"{ROOT}/.energy_resume_backups/forbidden.bin",
    ],
)
def test_only_normalised_strict_children_of_frozen_root_are_allowed(
    path: str,
) -> None:
    runner = _Runner([])
    with pytest.raises(remote.RemoteArtifactRehydrationError):
        remote.probe_remote_artifact(
            {
                "remote_path": path,
                "sha256": "a" * 64,
                "size_bytes": 1,
            },
            ssh_target=SSH,
            remote_run_root=ROOT,
            runner=runner,
        )
    assert runner.calls == []


def test_remote_exact_is_noop_even_when_local_source_is_gone(
    tmp_path: Path,
) -> None:
    data = b"old-smoke-hef"
    missing = tmp_path / "already-removed.hef"
    target = f"{ROOT}/yolov7/full/compiled.hef"
    stage_map = _stage_map([{
        "roles": ["compiled_model"],
        "remote_path": target,
        "source_path": str(missing),
        "sha256": _sha(data),
        "size_bytes": len(data),
        "expected_size_bytes": len(data),
        "source_stat": None,
    }])
    runner = _Runner([
        ("ONNX_SPLITPOINT_RESUME_REMOTE_PROBE_V1",
         _probe("regular", _sha(data), str(len(data))), 0),
        ("ONNX_SPLITPOINT_RESUME_REMOTE_PROBE_V1",
         _probe("regular", _sha(data), str(len(data))), 0),
    ])

    report = remote.rehydrate_remote_stage_map(
        stage_map,
        ssh_target=SSH,
        remote_run_root=ROOT,
        resume_attempt_id="resume-ffc2ef4c",
        runner=runner,
    )

    assert report["ok"] is True
    assert report["status"] == "complete"
    assert report["no_op_count"] == 1
    assert report["rehydrated_count"] == 0
    assert report["transferred_bytes"] == 0
    assert report["entries"][0]["status"] == "no_op_remote_exact"
    assert report["entries"][0]["local_source_accessed"] is False
    assert len(runner.calls) == 2
    assert report["authoritative_final_probe_count"] == 1


def test_missing_run_root_under_real_parent_is_created_and_rehydrated(
    tmp_path: Path,
) -> None:
    storage_root = tmp_path / "native_fifo_evalsets"
    storage_root.mkdir()
    run_root = storage_root / "run-20260727"
    data = b"exact legacy manifest bytes"
    source_root = tmp_path / "sources"
    source_root.mkdir()
    source = source_root / "manifest.json"
    source.write_bytes(data)
    target = run_root / "yolo26s" / "benchmark_set" / "b038" / "manifest.json"
    runner = _LocalSshRunner()

    report = remote.rehydrate_remote_stage_map(
        _stage_map(
            [_entry(source, str(target), data)],
            allowed_roots=[str(run_root)],
        ),
        ssh_target=SSH,
        remote_run_root=str(run_root),
        resume_attempt_id="resume-cleanup-root",
        runner=runner,
    )

    assert report["ok"] is True
    assert report["remote_run_root_created"] is True
    assert report["remote_directory_creation_count"] == 4
    assert report["remote_tree_preparation"]["required_parents"] == [
        str(target.parent)
    ]
    assert report["rehydrated_count"] == 1
    assert report["backup_count"] == 0
    assert run_root.is_dir() and not run_root.is_symlink()
    assert target.parent.is_dir() and not target.parent.is_symlink()
    assert target.read_bytes() == data
    assert [
        call["marker"] for call in runner.calls
    ] == [
        "# ONNX_SPLITPOINT_RESUME_REMOTE_PROBE_V1",
        "# ONNX_SPLITPOINT_RESUME_REMOTE_TREE_V1",
        "# ONNX_SPLITPOINT_RESUME_REMOTE_UPLOAD_V1",
        "# ONNX_SPLITPOINT_RESUME_REMOTE_COMMIT_V1",
        "# ONNX_SPLITPOINT_RESUME_REMOTE_PROBE_V1",
        "# ONNX_SPLITPOINT_RESUME_REMOTE_PROBE_V1",
    ]
    assert [call["has_stdin"] for call in runner.calls] == [
        False, False, True, False, False, False,
    ]


def test_existing_run_root_with_missing_nested_parents_is_rehydrated(
    tmp_path: Path,
) -> None:
    storage_root = tmp_path / "native_fifo_evalsets"
    run_root = storage_root / "run-20260727"
    existing_parent = run_root / "yolo26s"
    existing_parent.mkdir(parents=True)
    target = existing_parent / "benchmark_set" / "b038" / "manifest.json"
    data = b"partial cleanup manifest"
    source_root = tmp_path / "sources"
    source_root.mkdir()
    source = source_root / "manifest.json"
    source.write_bytes(data)
    runner = _LocalSshRunner()

    report = remote.rehydrate_remote_stage_map(
        _stage_map(
            [_entry(source, str(target), data)],
            allowed_roots=[str(run_root)],
        ),
        ssh_target=SSH,
        remote_run_root=str(run_root),
        resume_attempt_id="resume-partial-cleanup",
        runner=runner,
    )

    assert report["ok"] is True
    assert report["remote_run_root_created"] is False
    assert report["remote_directory_creation_count"] == 2
    assert report["remote_tree_preparation"][
        "expected_remote_run_root_state"
    ] == "existing"
    assert target.read_bytes() == data
    assert runner.calls[1]["marker"] == (
        "# ONNX_SPLITPOINT_RESUME_REMOTE_TREE_V1"
    )


def test_storage_root_identity_drift_before_creation_fails_without_run_root(
    tmp_path: Path,
) -> None:
    storage_root = tmp_path / "native_fifo_evalsets"
    storage_root.mkdir()
    old_storage_root = tmp_path / "native_fifo_evalsets.before-drift"
    run_root = storage_root / "run-20260727"
    data = b"verified before remote mutation"
    source = tmp_path / "source.bin"
    source.write_bytes(data)
    base_runner = _LocalSshRunner()
    swapped = False

    def runner(command: list[str], **kwargs: Any) -> Any:
        nonlocal swapped
        remote_argv = shlex.split(command[-1])
        if (
            "ONNX_SPLITPOINT_RESUME_REMOTE_TREE_V1" in remote_argv[2]
            and not swapped
        ):
            storage_root.rename(old_storage_root)
            storage_root.mkdir()
            swapped = True
        return base_runner(command, **kwargs)

    with pytest.raises(
        remote.RemoteArtifactRehydrationError,
        match="remote_tree_storage_root_identity_drift",
    ):
        remote.rehydrate_remote_stage_map(
            _stage_map(
                [_entry(source, str(run_root / "model" / "a.bin"), data)],
                allowed_roots=[str(run_root)],
            ),
            ssh_target=SSH,
            remote_run_root=str(run_root),
            resume_attempt_id="resume-storage-drift",
            runner=runner,
        )

    assert swapped is True
    assert not run_root.exists() and not run_root.is_symlink()
    assert not (old_storage_root / run_root.name).exists()
    assert all(call["has_stdin"] is False for call in base_runner.calls)


def test_created_run_root_identity_drift_before_upload_fails_closed(
    tmp_path: Path,
) -> None:
    storage_root = tmp_path / "native_fifo_evalsets"
    storage_root.mkdir()
    run_root = storage_root / "run-20260727"
    old_run_root = storage_root / "run-20260727.before-drift"
    data = b"never written after root identity drift"
    source = tmp_path / "source.bin"
    source.write_bytes(data)
    target = run_root / "model" / "a.bin"
    base_runner = _LocalSshRunner()
    swapped = False

    def runner(command: list[str], **kwargs: Any) -> Any:
        nonlocal swapped
        remote_argv = shlex.split(command[-1])
        if (
            "ONNX_SPLITPOINT_RESUME_REMOTE_UPLOAD_V1" in remote_argv[2]
            and not swapped
        ):
            run_root.rename(old_run_root)
            run_root.mkdir()
            swapped = True
        return base_runner(command, **kwargs)

    with pytest.raises(
        remote.RemoteArtifactRehydrationError,
        match="remote_upload_root_identity_drift",
    ):
        remote.rehydrate_remote_stage_map(
            _stage_map(
                [_entry(source, str(target), data)],
                allowed_roots=[str(run_root)],
            ),
            ssh_target=SSH,
            remote_run_root=str(run_root),
            resume_attempt_id="resume-root-drift",
            runner=runner,
        )

    assert swapped is True
    assert not target.exists() and not target.is_symlink()
    assert not (old_run_root / "model" / "a.bin").exists()


def test_existing_run_root_identity_drift_before_upload_fails_closed(
    tmp_path: Path,
) -> None:
    storage_root = tmp_path / "native_fifo_evalsets"
    run_root = storage_root / "run-20260727"
    target = run_root / "model" / "a.bin"
    target.parent.mkdir(parents=True)
    old_run_root = storage_root / "run-20260727.before-drift"
    data = b"never written into a replacement existing root"
    source = tmp_path / "source.bin"
    source.write_bytes(data)
    base_runner = _LocalSshRunner()
    swapped = False

    def runner(command: list[str], **kwargs: Any) -> Any:
        nonlocal swapped
        remote_argv = shlex.split(command[-1])
        if (
            "ONNX_SPLITPOINT_RESUME_REMOTE_UPLOAD_V1" in remote_argv[2]
            and not swapped
        ):
            run_root.rename(old_run_root)
            target.parent.mkdir(parents=True)
            swapped = True
        return base_runner(command, **kwargs)

    with pytest.raises(
        remote.RemoteArtifactRehydrationError,
        match="remote_upload_root_identity_drift",
    ):
        remote.rehydrate_remote_stage_map(
            _stage_map(
                [_entry(source, str(target), data)],
                allowed_roots=[str(run_root)],
            ),
            ssh_target=SSH,
            remote_run_root=str(run_root),
            resume_attempt_id="resume-existing-root-drift",
            runner=runner,
        )

    assert swapped is True
    assert not target.exists() and not target.is_symlink()
    assert not (old_run_root / "model" / "a.bin").exists()


def test_storage_identity_drift_after_tree_fails_with_same_run_root_inode(
    tmp_path: Path,
) -> None:
    storage_root = tmp_path / "native_fifo_evalsets"
    storage_root.mkdir()
    old_storage_root = tmp_path / "native_fifo_evalsets.before-drift"
    run_root = storage_root / "run-20260727"
    target = run_root / "model" / "a.bin"
    data = b"storage identity must remain pinned"
    source = tmp_path / "source.bin"
    source.write_bytes(data)
    base_runner = _LocalSshRunner()
    swapped = False

    def runner(command: list[str], **kwargs: Any) -> Any:
        nonlocal swapped
        remote_argv = shlex.split(command[-1])
        if (
            "ONNX_SPLITPOINT_RESUME_REMOTE_UPLOAD_V1" in remote_argv[2]
            and not swapped
        ):
            original_run_identity = (run_root.stat().st_dev, run_root.stat().st_ino)
            storage_root.rename(old_storage_root)
            storage_root.mkdir()
            (old_storage_root / run_root.name).rename(run_root)
            assert (run_root.stat().st_dev, run_root.stat().st_ino) == (
                original_run_identity
            )
            swapped = True
        return base_runner(command, **kwargs)

    with pytest.raises(
        remote.RemoteArtifactRehydrationError,
        match="remote_upload_storage_root_identity_drift",
    ):
        remote.rehydrate_remote_stage_map(
            _stage_map(
                [_entry(source, str(target), data)],
                allowed_roots=[str(run_root)],
            ),
            ssh_target=SSH,
            remote_run_root=str(run_root),
            resume_attempt_id="resume-storage-after-tree-drift",
            runner=runner,
        )

    assert swapped is True
    assert not target.exists() and not target.is_symlink()


def test_root_identity_drift_before_final_probe_fails_with_exact_bytes(
    tmp_path: Path,
) -> None:
    storage_root = tmp_path / "native_fifo_evalsets"
    storage_root.mkdir()
    run_root = storage_root / "run-20260727"
    old_run_root = storage_root / "run-20260727.before-drift"
    target = run_root / "model" / "a.bin"
    data = b"exact bytes in the wrong root inode"
    source = tmp_path / "source.bin"
    source.write_bytes(data)
    base_runner = _LocalSshRunner()
    commit_seen = False
    swapped = False

    def runner(command: list[str], **kwargs: Any) -> Any:
        nonlocal commit_seen, swapped
        remote_argv = shlex.split(command[-1])
        script = remote_argv[2]
        if "ONNX_SPLITPOINT_RESUME_REMOTE_COMMIT_V1" in script:
            result = base_runner(command, **kwargs)
            commit_seen = result.returncode == 0
            return result
        if (
            "ONNX_SPLITPOINT_RESUME_REMOTE_PROBE_V1" in script
            and commit_seen
            and not swapped
        ):
            run_root.rename(old_run_root)
            target.parent.mkdir(parents=True)
            target.write_bytes(data)
            swapped = True
        return base_runner(command, **kwargs)

    with pytest.raises(
        remote.RemoteArtifactRehydrationError,
        match="remote_probe_root_identity_drift",
    ):
        remote.rehydrate_remote_stage_map(
            _stage_map(
                [_entry(source, str(target), data)],
                allowed_roots=[str(run_root)],
            ),
            ssh_target=SSH,
            remote_run_root=str(run_root),
            resume_attempt_id="resume-final-root-drift",
            runner=runner,
        )

    assert commit_seen is True
    assert swapped is True
    assert target.read_bytes() == data
    assert (old_run_root / "model" / "a.bin").read_bytes() == data


def test_existing_real_run_root_remains_noop_without_local_source(
    tmp_path: Path,
) -> None:
    storage_root = tmp_path / "native_fifo_evalsets"
    target = storage_root / "run-20260727" / "model" / "compiled.hef"
    target.parent.mkdir(parents=True)
    data = b"remote exact"
    target.write_bytes(data)
    missing_source = tmp_path / "already-cleaned.hef"
    run_root = storage_root / "run-20260727"
    runner = _LocalSshRunner()

    report = remote.rehydrate_remote_stage_map(
        _stage_map(
            [{
                "roles": ["compiled_model"],
                "remote_path": str(target),
                "source_path": str(missing_source),
                "sha256": _sha(data),
                "size_bytes": len(data),
                "expected_size_bytes": len(data),
                "source_stat": None,
            }],
            allowed_roots=[str(run_root)],
        ),
        ssh_target=SSH,
        remote_run_root=str(run_root),
        resume_attempt_id="resume-existing-root",
        runner=runner,
    )

    assert report["ok"] is True
    assert report["remote_run_root_created"] is False
    assert report["no_op_count"] == 1
    assert report["rehydrated_count"] == 0
    assert target.read_bytes() == data
    assert [call["marker"] for call in runner.calls] == [
        "# ONNX_SPLITPOINT_RESUME_REMOTE_PROBE_V1",
        "# ONNX_SPLITPOINT_RESUME_REMOTE_PROBE_V1",
    ]


def test_initially_exact_target_is_authoritatively_reprobed(
    tmp_path: Path,
) -> None:
    storage_root = tmp_path / "native_fifo_evalsets"
    target = storage_root / "run-20260727" / "model" / "compiled.hef"
    target.parent.mkdir(parents=True)
    data = b"initially exact but deleted before final probe"
    target.write_bytes(data)
    run_root = storage_root / "run-20260727"
    missing_source = tmp_path / "already-cleaned.hef"
    base_runner = _LocalSshRunner()
    probe_count = 0

    def runner(command: list[str], **kwargs: Any) -> Any:
        nonlocal probe_count
        remote_argv = shlex.split(command[-1])
        if "ONNX_SPLITPOINT_RESUME_REMOTE_PROBE_V1" in remote_argv[2]:
            probe_count += 1
            if probe_count == 2:
                target.unlink()
        return base_runner(command, **kwargs)

    with pytest.raises(
        remote.RemoteArtifactRehydrationError,
        match="remote_authoritative_final_probe_identity_mismatch",
    ):
        remote.rehydrate_remote_stage_map(
            _stage_map(
                [{
                    "roles": ["compiled_model"],
                    "remote_path": str(target),
                    "source_path": str(missing_source),
                    "sha256": _sha(data),
                    "size_bytes": len(data),
                    "expected_size_bytes": len(data),
                    "source_stat": None,
                }],
                allowed_roots=[str(run_root)],
            ),
            ssh_target=SSH,
            remote_run_root=str(run_root),
            resume_attempt_id="resume-exact-final-probe",
            runner=runner,
        )

    assert probe_count == 2
    assert not target.exists()


def test_missing_run_root_with_bad_second_source_mutates_nothing(
    tmp_path: Path,
) -> None:
    storage_root = tmp_path / "native_fifo_evalsets"
    storage_root.mkdir()
    run_root = storage_root / "run-20260727"
    source_root = tmp_path / "sources"
    source_root.mkdir()
    first_data = b"first exact source"
    first = source_root / "first.bin"
    first.write_bytes(first_data)
    second = source_root / "second.bin"
    second.write_bytes(b"wrong second bytes")
    expected_second = b"right second bytes"
    second_entry = _entry(
        second, str(run_root / "b" / "second.bin"), b"wrong second bytes",
    )
    second_entry["sha256"] = _sha(expected_second)
    runner = _LocalSshRunner()

    with pytest.raises(
        remote.RemoteArtifactRehydrationError,
        match="local_exact_source_sha256_mismatch",
    ) as raised:
        remote.rehydrate_remote_stage_map(
            _stage_map(
                [
                    _entry(
                        first,
                        str(run_root / "a" / "first.bin"),
                        first_data,
                    ),
                    second_entry,
                ],
                allowed_roots=[str(run_root)],
            ),
            ssh_target=SSH,
            remote_run_root=str(run_root),
            resume_attempt_id="resume-bad-source",
            runner=runner,
        )

    assert raised.value.report is not None
    assert raised.value.report["completed_entry_count"] == 0
    assert not run_root.exists() and not run_root.is_symlink()
    assert list(storage_root.iterdir()) == []
    assert all(
        call["marker"] == "# ONNX_SPLITPOINT_RESUME_REMOTE_PROBE_V1"
        for call in runner.calls
    )
    assert all(call["has_stdin"] is False for call in runner.calls)


@pytest.mark.parametrize(
    "unsafe_kind, expected_code",
    [
        ("storage_symlink", "remote_probe_unsafe_storage_root"),
        ("storage_file", "remote_probe_unsafe_storage_root"),
        ("run_symlink", "remote_probe_unsafe_root"),
        ("run_file", "remote_probe_unsafe_root"),
    ],
)
def test_unsafe_storage_parent_or_run_root_is_rejected_without_mutation(
    tmp_path: Path,
    unsafe_kind: str,
    expected_code: str,
) -> None:
    real_storage = tmp_path / "real-storage"
    real_storage.mkdir()
    if unsafe_kind == "storage_symlink":
        storage_root = tmp_path / "storage-link"
        storage_root.symlink_to(real_storage, target_is_directory=True)
        run_root = storage_root / "run-20260727"
    elif unsafe_kind == "storage_file":
        storage_root = tmp_path / "storage-file"
        storage_root.write_bytes(b"do not replace")
        run_root = storage_root / "run-20260727"
    else:
        storage_root = real_storage
        run_root = storage_root / "run-20260727"
        if unsafe_kind == "run_symlink":
            outside = tmp_path / "outside"
            outside.mkdir()
            run_root.symlink_to(outside, target_is_directory=True)
        else:
            run_root.write_bytes(b"do not replace")
    before = sorted(
        (str(path.relative_to(tmp_path)), path.is_symlink(), path.is_file())
        for path in tmp_path.rglob("*")
    )
    runner = _LocalSshRunner()

    with pytest.raises(
        remote.RemoteArtifactRehydrationError,
        match=expected_code,
    ):
        remote.probe_remote_artifact(
            {
                "remote_path": str(run_root / "model" / "artifact.bin"),
                "sha256": "a" * 64,
                "size_bytes": 1,
            },
            ssh_target=SSH,
            remote_run_root=str(run_root),
            runner=runner,
        )

    after = sorted(
        (str(path.relative_to(tmp_path)), path.is_symlink(), path.is_file())
        for path in tmp_path.rglob("*")
    )
    assert after == before
    assert len(runner.calls) == 1
    assert runner.calls[0]["marker"] == (
        "# ONNX_SPLITPOINT_RESUME_REMOTE_PROBE_V1"
    )
    assert runner.calls[0]["has_stdin"] is False


def test_intermediate_remote_parent_symlink_is_rejected_by_real_guard(
    tmp_path: Path,
) -> None:
    storage_root = tmp_path / "native_fifo_evalsets"
    run_root = storage_root / "run-20260727"
    run_root.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    link = run_root / "model-link"
    link.symlink_to(outside, target_is_directory=True)
    runner = _LocalSshRunner()

    with pytest.raises(
        remote.RemoteArtifactRehydrationError,
        match="remote_probe_unsafe_parent",
    ):
        remote.probe_remote_artifact(
            {
                "remote_path": str(link / "nested" / "artifact.bin"),
                "sha256": "a" * 64,
                "size_bytes": 1,
            },
            ssh_target=SSH,
            remote_run_root=str(run_root),
            runner=runner,
        )

    assert len(runner.calls) == 1
    assert not (outside / "nested").exists()


def test_intermediate_remote_parent_file_is_not_classified_as_missing(
    tmp_path: Path,
) -> None:
    storage_root = tmp_path / "native_fifo_evalsets"
    run_root = storage_root / "run-20260727"
    run_root.mkdir(parents=True)
    blocker = run_root / "model"
    blocker.write_bytes(b"not a directory")
    runner = _LocalSshRunner()

    with pytest.raises(
        remote.RemoteArtifactRehydrationError,
        match="remote_probe_unsafe_parent",
    ):
        remote.probe_remote_artifact(
            {
                "remote_path": str(blocker / "nested" / "artifact.bin"),
                "sha256": "a" * 64,
                "size_bytes": 1,
            },
            ssh_target=SSH,
            remote_run_root=str(run_root),
            runner=runner,
        )

    assert len(runner.calls) == 1
    assert blocker.is_file()


def test_resolver_stage_map_still_noops_after_local_source_disappears(
    tmp_path: Path,
) -> None:
    data = b"legacy-hef-from-content-addressed-source"
    mirror = tmp_path / "old-run-mirror"
    source = mirror / "yolov7" / "full" / "compiled.hef"
    source.parent.mkdir(parents=True)
    source.write_bytes(data)
    target = f"{ROOT}/yolov7/full/compiled.hef"
    stage_map = build_resume_artifact_stage_map(
        [
            ArtifactRequirement(
                role="hef",
                remote_path=target,
                sha256=_sha(data),
                size_bytes=None,
            ),
        ],
        run_mirror_roots=[mirror],
        artifact_store_roots=[],
        allowed_remote_roots=[ROOT],
    )
    source.unlink()
    runner = _Runner([
        ("ONNX_SPLITPOINT_RESUME_REMOTE_PROBE_V1",
         _probe("regular", _sha(data), str(len(data))), 0),
        ("ONNX_SPLITPOINT_RESUME_REMOTE_PROBE_V1",
         _probe("regular", _sha(data), str(len(data))), 0),
    ])

    report = remote.rehydrate_remote_stage_map(
        stage_map,
        ssh_target=SSH,
        remote_run_root=ROOT,
        resume_attempt_id="resume-ffc2ef4c",
        runner=runner,
    )

    assert report["ok"] is True
    assert report["no_op_count"] == 1
    assert report["entries"][0]["local_source_accessed"] is False


def test_mismatch_is_streamed_backed_up_atomically_and_reprobed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = b"exact legacy compiled hef bytes"
    source = tmp_path / "compiled.hef"
    source.write_bytes(data)
    target = f"{ROOT}/yolov7/full/compiled.hef"
    stage_map = _stage_map([_entry(source, target, data)])
    old_sha = "e" * 64
    old_size = "123"
    token = "0123456789abcdef01234567"
    monkeypatch.setattr(remote.secrets, "token_hex", lambda _count: token)
    target_digest = hashlib.sha256(target.encode("utf-8")).hexdigest()
    backup = (
        f"{ROOT}/.energy_resume_backups/resume-ffc2ef4c/"
        f"{target_digest}.before-{token}"
    )
    runner = _Runner([
        ("ONNX_SPLITPOINT_RESUME_REMOTE_PROBE_V1",
         _probe("regular", old_sha, old_size), 0),
        ("ONNX_SPLITPOINT_RESUME_REMOTE_UPLOAD_V1",
         _upload("staged", _sha(data), str(len(data))), 0),
        ("ONNX_SPLITPOINT_RESUME_REMOTE_COMMIT_V1",
         _commit(
             "committed",
             _sha(data),
             str(len(data)),
             backup,
             old_sha,
             old_size,
         ), 0),
        ("ONNX_SPLITPOINT_RESUME_REMOTE_PROBE_V1",
         _probe("regular", _sha(data), str(len(data))), 0),
        ("ONNX_SPLITPOINT_RESUME_REMOTE_PROBE_V1",
         _probe("regular", _sha(data), str(len(data))), 0),
    ])

    report = remote.rehydrate_remote_stage_map(
        stage_map,
        ssh_target=SSH,
        remote_run_root=ROOT,
        resume_attempt_id="resume-ffc2ef4c",
        runner=runner,
    )

    assert report["ok"] is True
    assert report["rehydrated_count"] == 1
    assert report["backup_count"] == 1
    assert report["transferred_bytes"] == len(data)
    restored = report["entries"][0]
    assert restored["action"] == "atomic_replace"
    assert restored["backup_path"] == backup
    assert restored["commit"]["replaced_sha256"] == old_sha
    assert restored["commit"]["replaced_size_bytes"] == int(old_size)
    assert restored["final_probe"]["exact"] is True
    assert runner.calls[1]["stdin"] == data
    assert all(
        call["stdin"] is None
        for index, call in enumerate(runner.calls)
        if index != 1
    )


def test_missing_target_is_installed_without_backup(
    tmp_path: Path,
) -> None:
    data = b"manifest bytes"
    source = tmp_path / "native_outputs_manifest.json"
    source.write_bytes(data)
    target = f"{ROOT}/yolo26/b038/native_outputs/manifest.json"
    runner = _Runner([
        ("ONNX_SPLITPOINT_RESUME_REMOTE_PROBE_V1",
         _probe("missing"), 0),
        ("ONNX_SPLITPOINT_RESUME_REMOTE_UPLOAD_V1",
         _upload("staged", _sha(data), str(len(data))), 0),
        ("ONNX_SPLITPOINT_RESUME_REMOTE_COMMIT_V1",
         _commit("committed", _sha(data), str(len(data))), 0),
        ("ONNX_SPLITPOINT_RESUME_REMOTE_PROBE_V1",
         _probe("regular", _sha(data), str(len(data))), 0),
        ("ONNX_SPLITPOINT_RESUME_REMOTE_PROBE_V1",
         _probe("regular", _sha(data), str(len(data))), 0),
    ])

    report = remote.rehydrate_remote_stage_map(
        _stage_map([_entry(source, target, data)]),
        ssh_target=SSH,
        remote_run_root=ROOT,
        resume_attempt_id="resume-1",
        runner=runner,
    )

    assert report["rehydrated_count"] == 1
    assert report["backup_count"] == 0
    assert report["entries"][0]["backup_path"] is None


@pytest.mark.parametrize("unsafe_status", ["symlink", "non_regular"])
def test_remote_symlink_and_non_regular_target_fail_closed(
    tmp_path: Path,
    unsafe_status: str,
) -> None:
    data = b"safe local bytes"
    source = tmp_path / "artifact.bin"
    source.write_bytes(data)
    runner = _Runner([
        ("ONNX_SPLITPOINT_RESUME_REMOTE_PROBE_V1",
         _probe(unsafe_status), 0),
    ])

    with pytest.raises(
        remote.RemoteArtifactRehydrationError,
        match=f"remote_probe_{unsafe_status}",
    ) as raised:
        remote.rehydrate_remote_stage_map(
            _stage_map([
                _entry(source, f"{ROOT}/artifact.bin", data),
            ]),
            ssh_target=SSH,
            remote_run_root=ROOT,
            resume_attempt_id="resume-1",
            runner=runner,
        )

    assert raised.value.report is not None
    assert raised.value.report["ok"] is False
    assert raised.value.report["failure_code"] == (
        f"remote_probe_{unsafe_status}"
    )
    assert len(runner.calls) == 1


def test_all_local_sources_are_verified_before_first_remote_mutation(
    tmp_path: Path,
) -> None:
    correct = b"expected"
    first = tmp_path / "first.bin"
    first.write_bytes(correct)
    missing = tmp_path / "missing.bin"
    stage_map = _stage_map([
        _entry(first, f"{ROOT}/a.bin", correct),
        {
            "roles": ["payload"],
            "remote_path": f"{ROOT}/b.bin",
            "source_path": str(missing),
            "sha256": _sha(b"second"),
            "size_bytes": len(b"second"),
            "expected_size_bytes": len(b"second"),
            "source_stat": None,
        },
    ])
    runner = _Runner([
        ("ONNX_SPLITPOINT_RESUME_REMOTE_PROBE_V1",
         _probe("missing"), 0),
        ("ONNX_SPLITPOINT_RESUME_REMOTE_PROBE_V1",
         _probe("missing"), 0),
    ])

    with pytest.raises(
        remote.RemoteArtifactRehydrationError,
        match="local_exact_source_missing",
    ) as raised:
        remote.rehydrate_remote_stage_map(
            stage_map,
            ssh_target=SSH,
            remote_run_root=ROOT,
            resume_attempt_id="resume-1",
            runner=runner,
        )

    assert raised.value.report is not None
    assert raised.value.report["completed_entry_count"] == 0
    assert len(runner.calls) == 2
    assert all(
        "ONNX_SPLITPOINT_RESUME_REMOTE_PROBE_V1" in call["command"][-1]
        for call in runner.calls
    )


def test_wrong_local_hash_fails_before_upload(
    tmp_path: Path,
) -> None:
    expected = b"expected"
    source = tmp_path / "wrong.bin"
    source.write_bytes(b"wrong___")
    raw = _entry(source, f"{ROOT}/wrong.bin", b"wrong___")
    raw["sha256"] = _sha(expected)
    stage_map = _stage_map([raw])
    runner = _Runner([
        ("ONNX_SPLITPOINT_RESUME_REMOTE_PROBE_V1",
         _probe("missing"), 0),
    ])

    with pytest.raises(
        remote.RemoteArtifactRehydrationError,
        match="local_exact_source_sha256_mismatch",
    ):
        remote.rehydrate_remote_stage_map(
            stage_map,
            ssh_target=SSH,
            remote_run_root=ROOT,
            resume_attempt_id="resume-1",
            runner=runner,
        )

    assert len(runner.calls) == 1


def test_local_symlink_fails_before_upload(
    tmp_path: Path,
) -> None:
    data = b"exact"
    real_source = tmp_path / "real.bin"
    real_source.write_bytes(data)
    symlink = tmp_path / "link.bin"
    symlink.symlink_to(real_source)
    stage_map = _stage_map([{
        "roles": ["payload"],
        "remote_path": f"{ROOT}/payload.bin",
        "source_path": str(symlink),
        "sha256": _sha(data),
        "size_bytes": len(data),
        "expected_size_bytes": len(data),
        "source_stat": None,
    }])
    runner = _Runner([
        ("ONNX_SPLITPOINT_RESUME_REMOTE_PROBE_V1",
         _probe("missing"), 0),
    ])

    with pytest.raises(
        remote.RemoteArtifactRehydrationError,
        match="local_exact_source_open_failed",
    ):
        remote.rehydrate_remote_stage_map(
            stage_map,
            ssh_target=SSH,
            remote_run_root=ROOT,
            resume_attempt_id="resume-1",
            runner=runner,
        )
    assert len(runner.calls) == 1


def test_remote_temporary_mismatch_never_reaches_commit(
    tmp_path: Path,
) -> None:
    data = b"exact"
    source = tmp_path / "artifact.bin"
    source.write_bytes(data)
    runner = _Runner([
        ("ONNX_SPLITPOINT_RESUME_REMOTE_PROBE_V1",
         _probe("missing"), 0),
        ("ONNX_SPLITPOINT_RESUME_REMOTE_UPLOAD_V1",
         _upload("temporary_identity_mismatch", "f" * 64, str(len(data))), 0),
    ])

    with pytest.raises(
        remote.RemoteArtifactRehydrationError,
        match="remote_upload_temporary_identity_mismatch",
    ):
        remote.rehydrate_remote_stage_map(
            _stage_map([
                _entry(source, f"{ROOT}/artifact.bin", data),
            ]),
            ssh_target=SSH,
            remote_run_root=ROOT,
            resume_attempt_id="resume-1",
            runner=runner,
        )

    assert len(runner.calls) == 2
    assert runner.calls[1]["stdin"] == data


def test_symlink_race_at_commit_fails_closed(
    tmp_path: Path,
) -> None:
    data = b"exact"
    source = tmp_path / "artifact.bin"
    source.write_bytes(data)
    runner = _Runner([
        ("ONNX_SPLITPOINT_RESUME_REMOTE_PROBE_V1",
         _probe("missing"), 0),
        ("ONNX_SPLITPOINT_RESUME_REMOTE_UPLOAD_V1",
         _upload("staged", _sha(data), str(len(data))), 0),
        ("ONNX_SPLITPOINT_RESUME_REMOTE_COMMIT_V1",
         _commit("symlink"), 0),
    ])

    with pytest.raises(
        remote.RemoteArtifactRehydrationError,
        match="remote_commit_symlink",
    ):
        remote.rehydrate_remote_stage_map(
            _stage_map([
                _entry(source, f"{ROOT}/artifact.bin", data),
            ]),
            ssh_target=SSH,
            remote_run_root=ROOT,
            resume_attempt_id="resume-1",
            runner=runner,
        )
    assert len(runner.calls) == 3


def test_final_remote_probe_is_authoritative(
    tmp_path: Path,
) -> None:
    data = b"exact"
    source = tmp_path / "artifact.bin"
    source.write_bytes(data)
    runner = _Runner([
        ("ONNX_SPLITPOINT_RESUME_REMOTE_PROBE_V1",
         _probe("missing"), 0),
        ("ONNX_SPLITPOINT_RESUME_REMOTE_UPLOAD_V1",
         _upload("staged", _sha(data), str(len(data))), 0),
        ("ONNX_SPLITPOINT_RESUME_REMOTE_COMMIT_V1",
         _commit("committed", _sha(data), str(len(data))), 0),
        ("ONNX_SPLITPOINT_RESUME_REMOTE_PROBE_V1",
         _probe("regular", "f" * 64, str(len(data))), 0),
    ])

    with pytest.raises(
        remote.RemoteArtifactRehydrationError,
        match="remote_final_probe_identity_mismatch",
    ) as raised:
        remote.rehydrate_remote_stage_map(
            _stage_map([
                _entry(source, f"{ROOT}/artifact.bin", data),
            ]),
            ssh_target=SSH,
            remote_run_root=ROOT,
            resume_attempt_id="resume-1",
            runner=runner,
        )

    assert raised.value.report is not None
    assert raised.value.report["ok"] is False
    assert len(runner.calls) == 4


def test_initially_exact_entry_is_reprobed_after_other_entry_mutates(
    tmp_path: Path,
) -> None:
    exact_data = b"initially exact"
    restore_data = b"needs restoration"
    missing_exact_source = tmp_path / "cleaned-exact.bin"
    restore_source = tmp_path / "restore.bin"
    restore_source.write_bytes(restore_data)
    exact_path = f"{ROOT}/a-exact.bin"
    restore_path = f"{ROOT}/b-restore.bin"
    runner = _Runner([
        ("ONNX_SPLITPOINT_RESUME_REMOTE_PROBE_V1",
         _probe("regular", _sha(exact_data), str(len(exact_data))), 0),
        ("ONNX_SPLITPOINT_RESUME_REMOTE_PROBE_V1",
         _probe("missing"), 0),
        ("ONNX_SPLITPOINT_RESUME_REMOTE_UPLOAD_V1",
         _upload("staged", _sha(restore_data), str(len(restore_data))), 0),
        ("ONNX_SPLITPOINT_RESUME_REMOTE_COMMIT_V1",
         _commit(
             "committed", _sha(restore_data), str(len(restore_data)),
         ), 0),
        ("ONNX_SPLITPOINT_RESUME_REMOTE_PROBE_V1",
         _probe("regular", _sha(restore_data), str(len(restore_data))), 0),
        ("ONNX_SPLITPOINT_RESUME_REMOTE_PROBE_V1",
         _probe("regular", "f" * 64, str(len(exact_data))), 0),
    ])
    stage_map = _stage_map([
        {
            "roles": ["compiled_model"],
            "remote_path": exact_path,
            "source_path": str(missing_exact_source),
            "sha256": _sha(exact_data),
            "size_bytes": len(exact_data),
            "expected_size_bytes": len(exact_data),
            "source_stat": None,
        },
        _entry(restore_source, restore_path, restore_data),
    ])

    with pytest.raises(
        remote.RemoteArtifactRehydrationError,
        match="remote_authoritative_final_probe_identity_mismatch",
    ) as raised:
        remote.rehydrate_remote_stage_map(
            stage_map,
            ssh_target=SSH,
            remote_run_root=ROOT,
            resume_attempt_id="resume-final-batch",
            runner=runner,
        )

    assert raised.value.report is not None
    assert raised.value.report["ok"] is False
    assert raised.value.report["completed_entry_count"] == 2
    assert raised.value.report["authoritative_final_probe_count"] == 0
    assert not runner.responses


def test_tampered_stage_map_is_rejected_before_remote_probe(
    tmp_path: Path,
) -> None:
    data = b"exact"
    source = tmp_path / "artifact.bin"
    source.write_bytes(data)
    stage_map = _stage_map([
        _entry(source, f"{ROOT}/artifact.bin", data),
    ])
    stage_map["entries"][0]["remote_path"] = f"{ROOT}/changed.bin"
    runner = _Runner([])

    with pytest.raises(
        remote.RemoteArtifactRehydrationError,
        match="stage_map_sha256_mismatch",
    ):
        remote.rehydrate_remote_stage_map(
            stage_map,
            ssh_target=SSH,
            remote_run_root=ROOT,
            resume_attempt_id="resume-1",
            runner=runner,
        )
    assert runner.calls == []


@pytest.mark.parametrize(
    "remote_status",
    ["unsafe_root", "unsafe_storage_root", "unsafe_parent", "path_escape"],
)
def test_unsafe_remote_probe_environment_fails_closed(
    remote_status: str,
) -> None:
    runner = _Runner([
        ("ONNX_SPLITPOINT_RESUME_REMOTE_PROBE_V1",
         _probe(remote_status), 0),
    ])
    with pytest.raises(
        remote.RemoteArtifactRehydrationError,
        match=f"remote_probe_{remote_status}",
    ):
        remote.probe_remote_artifact(
            {
                "remote_path": f"{ROOT}/artifact.bin",
                "sha256": "a" * 64,
                "size_bytes": 1,
            },
            ssh_target=SSH,
            remote_run_root=ROOT,
            runner=runner,
        )


def test_remote_command_failure_is_not_interpreted_as_missing() -> None:
    runner = _Runner([
        ("ONNX_SPLITPOINT_RESUME_REMOTE_PROBE_V1", "", 255),
    ])
    with pytest.raises(
        remote.RemoteArtifactRehydrationError,
        match="ssh_remote_command_failed",
    ):
        remote.probe_remote_artifact(
            {
                "remote_path": f"{ROOT}/artifact.bin",
                "sha256": "a" * 64,
                "size_bytes": 1,
            },
            ssh_target=SSH,
            remote_run_root=ROOT,
            runner=runner,
        )
