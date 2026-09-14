from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from onnx_splitpoint_tool.benchmark import generation_state


def test_write_json_atomic_roundtrip_and_durability_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "new" / "nested" / "generation_state.json"
    events: list[str] = []
    real_fsync = os.fsync
    real_replace = os.replace

    def recording_fsync(fd: int) -> None:
        mode = os.fstat(fd).st_mode
        events.append("fsync_parent" if stat.S_ISDIR(mode) else "fsync_temp")
        real_fsync(fd)

    def recording_replace(
        src: str,
        dst: str,
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
    ) -> None:
        assert src_dir_fd is not None
        assert dst_dir_fd == src_dir_fd
        events.append("replace_dirfd")
        real_replace(
            src,
            dst,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )

    monkeypatch.setattr(generation_state.os, "fsync", recording_fsync)
    monkeypatch.setattr(generation_state.os, "replace", recording_replace)

    payload = {"schema": "test", "counter": 7, "nested": {"ok": True}}
    generation_state.write_json_atomic(path, payload)

    assert json.loads(path.read_text(encoding="utf-8")) == payload
    assert events == ["fsync_temp", "replace_dirfd", "fsync_parent"]


def test_write_json_atomic_does_not_follow_preplanted_temp_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "state"
    parent.mkdir()
    target = parent / "generation_state.json"
    victim = tmp_path / "victim.json"
    victim.write_text("do-not-overwrite", encoding="utf-8")

    planted_token = "11" * 16
    safe_token = "22" * 16
    planted = parent / f".generation-state-{planted_token}.tmp"
    planted.symlink_to(victim)
    tokens = iter((planted_token, safe_token))
    monkeypatch.setattr(
        generation_state.secrets,
        "token_hex",
        lambda _size: next(tokens),
    )

    generation_state.write_json_atomic(target, {"cold_builds_reserved": 1})

    assert json.loads(target.read_text(encoding="utf-8")) == {
        "cold_builds_reserved": 1
    }
    assert victim.read_text(encoding="utf-8") == "do-not-overwrite"
    assert planted.is_symlink()
    assert not (parent / f".generation-state-{safe_token}.tmp").exists()


def test_write_json_atomic_rejects_symlink_destination(tmp_path: Path) -> None:
    victim = tmp_path / "victim.json"
    victim.write_text("stable", encoding="utf-8")
    target = tmp_path / "generation_state.json"
    target.symlink_to(victim)

    with pytest.raises(RuntimeError, match="not a regular file"):
        generation_state.write_json_atomic(target, {"unsafe": True})

    assert target.is_symlink()
    assert victim.read_text(encoding="utf-8") == "stable"


def test_write_json_atomic_rejects_symlink_parent_component(
    tmp_path: Path,
) -> None:
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(OSError):
        generation_state.write_json_atomic(
            linked_parent / "generation_state.json",
            {"unsafe": True},
        )

    assert not (real_parent / "generation_state.json").exists()


def test_write_json_atomic_propagates_fsync_failure_without_replacing_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "generation_state.json"
    target.write_text('{"old": true}', encoding="utf-8")

    def fail_fsync(_fd: int) -> None:
        raise OSError("simulated durability failure")

    monkeypatch.setattr(generation_state.os, "fsync", fail_fsync)

    with pytest.raises(OSError, match="simulated durability failure"):
        generation_state.write_json_atomic(target, {"new": True})

    assert json.loads(target.read_text(encoding="utf-8")) == {"old": True}
    assert not list(tmp_path.glob(".generation-state-*.tmp"))
