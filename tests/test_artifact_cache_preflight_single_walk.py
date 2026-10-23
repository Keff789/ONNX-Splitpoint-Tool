"""One operation-local enumeration retains the original literal glob results."""
import errno
import json
import os
from pathlib import Path

import pytest

from onnx_splitpoint_tool.workflow import artifact_cache_preflight as cache


NAMES = ("hailo_cache_miss.json", "deferred_hailo_build.json",
         "hailo_negative_evidence.json", "deepx_part1_artifact_status.json",
         "deepx_part2_artifact_status.json")


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def literal_paths(root):
    return {name: sorted(root.rglob(name)) for name in NAMES}


@pytest.fixture
def suite(tmp_path):
    root = tmp_path / "run"
    bdir = root / "models/model/benchmark_set"
    suite = bdir / "legacy_suite"
    write(suite / "benchmark_set.json", {"model_id": "model", "cases": [
        {"id": "b001"}, {"id": "b002"}]})
    write(suite / "benchmark_plan.json", {"runs": [{
        "id": "deepx_to_trt", "type": "matrix", "stage1": "deepx_m1",
        "stage2": "tensorrt", "case_ids": ["b001", "b002"], "variants": ["composed"]}, {
        "id": "hailo8_to_trt", "type": "matrix", "stage1": "hailo8",
        "stage2": "tensorrt", "case_ids": ["b001"], "variants": ["composed"]}]})
    hailo = suite / "b001/hailo/hailo8/part1"
    write(hailo / NAMES[0], {"hw_arch": "hailo8", "net_name": "model_part1_b1",
          "cache_key_v3": "retained-key", "reason": "old-sidecar-miss"})
    write(hailo / NAMES[1], {"kwargs": {"hw_arch": "hailo8", "net_name": "model_part1_b1",
          "outdir": str(hailo), "cache_only": True}, "cache_probe_status": "MISS",
          "cache_probe_reason": "fresh-deferred-miss"})
    write(hailo / NAMES[2], {"hw_arch": "hailo8", "net_name": "model_part1_b1",
          "build_evidence": {"status": "HIT", "reusable": True,
          "negative_evidence_hit": True, "state": "PARSER_UNSUPPORTED"}})
    write(suite / "b001/deepx/aaa" / NAMES[3], {"selected": False})
    first = write(suite / "b001/deepx/bbb" / NAMES[3], {
        "cache_lookup": {"outcome": "HIT", "reason": "first-active"}})
    write(suite / "b001/deepx/zzz" / NAMES[3], {
        "cache_lookup": {"outcome": "MISS", "reason": "later-duplicate"}})
    write(suite / "b001/deepx" / NAMES[4], {
        "cache_lookup": {"outcome": "MISS", "reason": "part2-miss"}})
    write(suite / "b999/deepx" / NAMES[3], {"cache_lookup": {"outcome": "HIT"}})
    # An unexpected nested request must still be visited, without expanding scope.
    write(suite / "resources/validation/nested" / NAMES[1], {
        "kwargs": {"hw_arch": "hailo8", "net_name": "model_part1_b2"}})
    write(suite / NAMES[3], {"cases": {"b001": {}}})
    write(bdir / "hailo_artifact_service_plan.json", {"case_hef_requests": [{
        "backend": "hailo8", "case_id": "b001", "stage": "part1",
        "status": "ready_built", "build_attempt": {"compiler_dispatch_count": 1},
        "hef_path": str(hailo / "compiled.hef")}]})
    return root, suite, first


def collect(root):
    rows, roles = cache.collect_model_artifact_cache_probes(
        run_dir=root, model_id="model", targets=["deepx_m1"],
        policy=cache.resolve_artifact_cache_preflight_policy({}))
    return [row.to_dict() for row in rows], roles


def test_exact_literal_pathsets_include_nested_directories_case_and_symlinks(tmp_path):
    root = tmp_path / "suite"
    for index, name in enumerate(NAMES):
        write(root / ".hidden" / str(index) / name, {})
        write(root / "upper" / name.upper(), {})
        write(root / "unrelated" / (name + ".bak"), {})
    (root / "as-directory" / NAMES[0]).mkdir(parents=True)
    write(root / "as-directory" / NAMES[0] / NAMES[1], {})
    (root / "links").mkdir()
    target = write(tmp_path / "outside" / NAMES[3], {})
    (root / "links" / NAMES[0]).symlink_to(target)
    (root / "links" / NAMES[1]).symlink_to(tmp_path / "absent")
    (root / "linked-directory").symlink_to(target.parent, target_is_directory=True)
    (root / "loop").symlink_to(root, target_is_directory=True)
    assert cache._collect_probe_paths(root) == literal_paths(root)
    assert cache._collect_probe_paths(tmp_path / "missing") == literal_paths(tmp_path / "missing")


def test_complete_observations_equal_legacy_with_stale_negative_duplicates_and_missing(suite, monkeypatch):
    root, _suite, _first = suite
    with monkeypatch.context() as patch:
        patch.setattr(cache, "_collect_probe_paths", lambda _: None)
        legacy = collect(root)
    actual = collect(root)
    assert actual == legacy
    rows, _roles = actual
    hailo = next(row for row in rows if row["role"] == cache.ROLE_HAILO8)
    assert hailo["status"] == "MISS" and hailo["reason"] == "fresh-deferred-miss"
    first = next(row for row in rows if row["role"] == cache.ROLE_DEEPX and row["item_id"] == "b001:part1")
    assert first["reason"] == "first-active"
    missing = next(row for row in rows if row["role"] == cache.ROLE_DEEPX and row["item_id"] == "b002:part1")
    assert missing["status"] == "UNKNOWN"


def test_five_recursive_walks_become_one_fresh_walk_for_each_collection(suite, monkeypatch):
    root, suite_dir, first = suite
    rglob_calls, walk_calls = [], []
    original_rglob, original_walk = Path.rglob, os.walk

    def rglob(path, pattern, **kwargs):
        if path == suite_dir:
            rglob_calls.append(pattern)
        yield from original_rglob(path, pattern, **kwargs)

    def walk(path, *args, **kwargs):
        if path == suite_dir:
            walk_calls.append(path)
        yield from original_walk(path, *args, **kwargs)

    monkeypatch.setattr(Path, "rglob", rglob)
    monkeypatch.setattr(os, "walk", walk)
    with monkeypatch.context() as patch:
        patch.setattr(cache, "_collect_probe_paths", lambda _: None)
        legacy = collect(root)
    assert rglob_calls == list(NAMES) and not walk_calls
    rglob_calls.clear()
    assert collect(root) == legacy
    assert rglob_calls == [] and walk_calls == [suite_dir]
    # Both newly created names and changed status payloads are observed next time.
    write(first, {"cache_lookup": {"outcome": "MISS", "reason": "changed-current"}})
    write(suite_dir / "b002/deepx" / NAMES[3], {
        "cache_lookup": {"outcome": "HIT", "reason": "new-current"}})
    rows, _roles = collect(root)
    assert walk_calls == [suite_dir, suite_dir] and rglob_calls == []
    assert any(row["reason"] == "changed-current" for row in rows)
    assert any(row["reason"] == "new-current" for row in rows)


def test_unlistable_parent_preserves_running_pathlib_literal_visibility(tmp_path, monkeypatch):
    root = tmp_path / "suite"
    denied = root / "execute-only"
    known = write(denied / NAMES[0], {"reason": "readable-by-known-path"})
    write(root / "visible" / NAMES[3], {})
    original_scandir, original_glob = os.scandir, Path.glob
    literal_attempts = []

    def scandir(path):
        if Path(path) == denied:
            raise PermissionError(errno.EACCES, "unlistable directory", os.fspath(path))
        return original_scandir(path)

    def glob(path, name, **kwargs):
        if path == denied:
            literal_attempts.append(name)
        yield from original_glob(path, name, **kwargs)

    monkeypatch.setattr(os, "scandir", scandir)
    expected = literal_paths(root)
    assert json.loads(known.read_text())["reason"] == "readable-by-known-path"
    monkeypatch.setattr(Path, "glob", glob)
    assert cache._collect_probe_paths(root) == expected
    # Older precise selectors can see the known file here without listing it;
    # current wildcard selectors cannot. Delegate rather than guess either way.
    assert literal_attempts == list(NAMES)


@pytest.mark.parametrize("fault", ["unknown_walk_error", "literal_lookup_error"])
def test_uncertain_enumeration_falls_back_to_original_five_globs(tmp_path, monkeypatch, fault):
    root = tmp_path / "suite"
    write(root / "nested" / NAMES[0], {})
    expected = literal_paths(root)
    if fault == "unknown_walk_error":
        def failed_walk(path, **kwargs):
            kwargs["onerror"](OSError("directory unavailable without a path"))
            return iter(())
        monkeypatch.setattr(os, "walk", failed_walk)
    else:
        def failed_glob(path, name, **kwargs):
            raise OSError(errno.EIO, "literal lookup unavailable", os.fspath(path))
        monkeypatch.setattr(Path, "glob", failed_glob)
    assert cache._collect_probe_paths(root) == expected
