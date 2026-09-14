from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import threading
import types

import pytest

from onnx_splitpoint_tool.split_export_runners import (
    write_runner_skeleton_onnxruntime,
)


def _runner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    case = tmp_path / "suite" / "b132"
    case.mkdir(parents=True)
    write_runner_skeleton_onnxruntime(str(case), target="cpu")
    monkeypatch.setitem(sys.modules, "onnxruntime", types.ModuleType("onnxruntime"))
    spec = importlib.util.spec_from_file_location(
        f"v27920_case_runner_{id(tmp_path)}",
        case / "run_split_onnxruntime.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def _write_valid_entry(module, source: Path, engine: Path) -> None:
    engine.parent.mkdir(parents=True, exist_ok=True)
    persistent_source = engine.parent / "source.onnx"
    persistent_source.write_bytes(source.read_bytes())
    engine.write_bytes(b"synthetic-compatible-engine")
    module._write_explicit_native_trt_engine_receipt(
        source_onnx=persistent_source,
        engine_path=engine,
        receipt_path=engine.parent / "engine_build_receipt.json",
        trtexec=Path(sys.executable),
        command=[
            str(Path(sys.executable).resolve()),
            f"--onnx={persistent_source}",
            f"--saveEngine={engine}",
            "--fp16",
            "--memPoolSize=workspace:4096",
        ],
        returncode=0,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_uint8_cast_bridge_records_and_verifies_source_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _runner(tmp_path, monkeypatch)
    source = tmp_path / "suite" / "b132" / "source_part2.onnx"
    helper = module.onnx.helper
    tensor = module.onnx.TensorProto
    graph = helper.make_graph(
        [helper.make_node("Identity", ["input"], ["output"])],
        "uint8-bridge-provenance-test",
        [helper.make_tensor_value_info("input", tensor.FLOAT, [1, 3, 4, 4])],
        [helper.make_tensor_value_info("output", tensor.FLOAT, [1, 3, 4, 4])],
    )
    module.onnx.save(helper.make_model(graph), str(source))
    cache_root = tmp_path / "persistent" / "splits" / "b132"

    bridge = module._ensure_uint8_cast_bridge_onnx(
        source,
        "part2",
        "uint8_cast_fp16",
        cache_root=cache_root,
        canonical_role_layout=True,
    )
    meta_path = bridge.parent / "uint8_cast_bridge_meta.json"
    first_meta = json.loads(meta_path.read_text(encoding="utf-8"))

    assert first_meta["source_sha256"] == _sha256(source)
    assert first_meta["source_size_bytes"] == source.stat().st_size
    assert first_meta["bridge_sha256"] == _sha256(bridge)
    assert first_meta["bridge_size_bytes"] == bridge.stat().st_size

    # A valid pair is reused byte-for-byte and without rewriting its metadata.
    first_bridge_bytes = bridge.read_bytes()
    first_meta_bytes = meta_path.read_bytes()
    assert module._ensure_uint8_cast_bridge_onnx(
        source,
        "part2",
        "uint8_cast_fp16",
        cache_root=cache_root,
        canonical_role_layout=True,
    ) == bridge
    assert bridge.read_bytes() == first_bridge_bytes
    assert meta_path.read_bytes() == first_meta_bytes

    # The original Part2 digest is an active reuse condition, not descriptive
    # metadata.  A false source binding is replaced before reuse.
    false_source_meta = dict(first_meta)
    false_source_meta["source_sha256"] = "0" * 64
    meta_path.write_text(
        json.dumps(false_source_meta, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    assert module._ensure_uint8_cast_bridge_onnx(
        source,
        "part2",
        "uint8_cast_fp16",
        cache_root=cache_root,
        canonical_role_layout=True,
    ) == bridge
    repaired_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert repaired_meta["source_sha256"] == _sha256(source)
    assert repaired_meta["bridge_sha256"] == _sha256(bridge)

    # A bridge whose bytes no longer match the recorded digest is regenerated;
    # stale derived bytes can never become an engine-build source.
    bridge.write_bytes(bridge.read_bytes() + b"tampered")
    assert _sha256(bridge) != first_meta["bridge_sha256"]
    assert module._ensure_uint8_cast_bridge_onnx(
        source,
        "part2",
        "uint8_cast_fp16",
        cache_root=cache_root,
        canonical_role_layout=True,
    ) == bridge
    regenerated_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert regenerated_meta["source_sha256"] == _sha256(source)
    assert regenerated_meta["bridge_sha256"] == _sha256(bridge)
    assert regenerated_meta["bridge_size_bytes"] == bridge.stat().st_size
    module.onnx.load(str(bridge), load_external_data=False)


def test_full_engine_is_reused_across_split_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _runner(tmp_path, monkeypatch)
    full = tmp_path / "suite" / "regnet_x_1_6gf.onnx"
    full.write_bytes(b"same-full-model-for-b132-and-b045")
    stable_root = tmp_path / "persistent" / "regnet-model-identity"

    b132_engine = module._native_trt_engine_path(
        "full", full, "fp16", stable_root, canonical_role_layout=True,
    )
    b045_engine = module._native_trt_engine_path(
        "full", full, "fp16", stable_root, canonical_role_layout=True,
    )
    assert b132_engine == b045_engine
    assert b132_engine.relative_to(stable_root).parts[0] == "full"
    assert "b132" not in str(b132_engine)
    assert "b045" not in str(b132_engine)
    _write_valid_entry(module, full, b132_engine)

    monkeypatch.setattr(
        module.NativeTRTSession,
        "_build_engine",
        lambda *_a, **_k: pytest.fail("warm Full cache invoked trtexec"),
    )
    monkeypatch.setattr(
        module.NativeTRTSession, "_load_engine", lambda *_a, **_k: None,
    )
    first = module.NativeTRTSession(
        "full", full, precision="fp16", cache_root=stable_root,
        canonical_role_layout=True,
    )
    second = module.NativeTRTSession(
        "full", full, precision="fp16", cache_root=stable_root,
        canonical_role_layout=True,
    )

    assert first.engine_path == second.engine_path == b132_engine
    assert first.build_info["cache_hit"] is True
    assert second.build_info["cache_hit"] is True
    output = capsys.readouterr().out
    assert output.count("[trt-cache] HIT role=full") == 2
    assert "reason=receipt_verified" in output


def test_split_engines_remain_case_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _runner(tmp_path, monkeypatch)
    part2 = tmp_path / "suite" / "part2.onnx"
    part2.write_bytes(b"split-part2")
    stable_root = tmp_path / "persistent" / "model-identity"
    b132 = module._native_trt_engine_path(
        "part2", part2, "fp16", stable_root / "splits" / "b132",
        canonical_role_layout=True,
    )
    b045 = module._native_trt_engine_path(
        "part2", part2, "fp16", stable_root / "splits" / "b045",
        canonical_role_layout=True,
    )

    assert b132 != b045
    assert b132.relative_to(stable_root).parts[:2] == ("splits", "b132")
    assert b045.relative_to(stable_root).parts[:2] == ("splits", "b045")


def test_invalid_receipt_is_miss_and_rebuilt_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _runner(tmp_path, monkeypatch)
    full = tmp_path / "suite" / "model.onnx"
    full.write_bytes(b"full-model")
    stable_root = tmp_path / "persistent" / "model-identity"
    engine = module._native_trt_engine_path(
        "full", full, "fp16", stable_root, canonical_role_layout=True,
    )
    engine.parent.mkdir(parents=True)
    engine.write_bytes(b"unverified-engine")
    (engine.parent / "engine_build_receipt.json").write_text(
        "{}", encoding="utf-8",
    )
    builds: list[Path] = []

    def fake_build(self, **_kwargs):
        builds.append(self.engine_path)
        self.engine_path.write_bytes(b"rebuilt-engine")
        module._write_explicit_native_trt_engine_receipt(
            source_onnx=self.model_path,
            engine_path=self.engine_path,
            receipt_path=self.receipt_path,
            trtexec=Path(sys.executable),
            command=[
                str(Path(sys.executable).resolve()),
                f"--onnx={self.model_path}",
                f"--saveEngine={self.engine_path}",
                "--fp16",
                "--memPoolSize=workspace:4096",
            ],
            returncode=0,
        )

    monkeypatch.setattr(module.NativeTRTSession, "_build_engine", fake_build)
    monkeypatch.setattr(
        module.NativeTRTSession, "_load_engine", lambda *_a, **_k: None,
    )
    session = module.NativeTRTSession(
        "full", full, precision="fp16", cache_root=stable_root,
        canonical_role_layout=True,
    )

    assert builds == [engine]
    assert session.build_info["cache_hit"] is False
    assert session.build_info["cache_built"] is True
    output = capsys.readouterr().out
    assert "[trt-cache] MISS role=full" in output
    assert "reason=receipt_invalid" in output
    assert "[trt-cache] BUILD role=full" in output


def test_concurrent_miss_cannot_delete_fresh_full_engine(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stale pre-lock MISS must re-check after another process builds.

    The first thread is deliberately paused immediately after its lock-free
    MISS.  A second thread then replaces the invalid pair.  Releasing the
    first thread reproduces the dangerous cross-process ordering: deleting in
    the constructor would remove the fresh pair and invoke trtexec twice.
    """

    module = _runner(tmp_path, monkeypatch)
    full = tmp_path / "suite" / "model.onnx"
    full.write_bytes(b"shared-full-model")
    stable_root = tmp_path / "persistent" / "model-identity"
    engine = module._native_trt_engine_path(
        "full", full, "fp16", stable_root, canonical_role_layout=True,
    )
    engine.parent.mkdir(parents=True)
    engine.write_bytes(b"stale-unverified-engine")
    (engine.parent / "engine_build_receipt.json").write_text(
        "{}", encoding="utf-8",
    )

    paused_miss = threading.Event()
    release_paused_miss = threading.Event()
    original_cache_log = module._native_trt_cache_log

    def coordinated_cache_log(event, **kwargs):
        original_cache_log(event, **kwargs)
        if (
            threading.current_thread().name == "paused-miss"
            and str(event).upper() == "MISS"
        ):
            paused_miss.set()
            assert release_paused_miss.wait(timeout=10)

    build_calls: list[list[str]] = []

    def fake_trtexec(command, **_kwargs):
        argv = [str(value) for value in command]
        build_calls.append(argv)
        save_arg = next(value for value in argv if value.startswith("--saveEngine="))
        Path(save_arg.split("=", 1)[1]).write_bytes(b"fresh-compatible-engine")
        return types.SimpleNamespace(returncode=0, stdout="synthetic build")

    monkeypatch.setattr(module, "_native_trt_cache_log", coordinated_cache_log)
    monkeypatch.setattr(module, "_find_trtexec_local", lambda: sys.executable)
    monkeypatch.setattr(
        module, "_onnx_static_io_info",
        lambda _path: ([("input", [1, 3, 8, 8], False)], []),
    )
    monkeypatch.setattr(module.subprocess, "run", fake_trtexec)
    monkeypatch.setattr(
        module.NativeTRTSession, "_load_engine", lambda *_a, **_k: None,
    )

    sessions: dict[str, object] = {}
    failures: list[BaseException] = []

    def create_session(name: str) -> None:
        try:
            sessions[name] = module.NativeTRTSession(
                "full",
                full,
                precision="fp16",
                cache_root=stable_root,
                canonical_role_layout=True,
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            failures.append(exc)

    paused = threading.Thread(
        target=create_session, args=("paused",), name="paused-miss",
    )
    builder = threading.Thread(
        target=create_session, args=("builder",), name="builder",
    )
    paused.start()
    assert paused_miss.wait(timeout=10)
    builder.start()
    builder.join(timeout=10)
    assert not builder.is_alive()
    release_paused_miss.set()
    paused.join(timeout=10)
    assert not paused.is_alive()

    assert failures == []
    assert len(build_calls) == 1
    assert sessions["builder"].build_info["cache_built"] is True
    assert sessions["paused"].build_info["cache_hit"] is True
    assert sessions["paused"].build_info["cache_hit_after_lock"] is True
    assert sessions["paused"].build_info["cache_built"] is False
    assert sessions["paused"].build_info["cache_resolution"] == (
        "receipt_verified_after_lock"
    )
    module._verify_explicit_native_trt_engine_receipt(
        source_onnx=full,
        engine_path=engine,
        receipt_path=engine.parent / "engine_build_receipt.json",
        expected_precision="fp16",
        expected_workspace_mb=4096,
    )


def test_wrong_dynamic_shape_receipt_is_rejected_before_inference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _runner(tmp_path, monkeypatch)
    full = tmp_path / "suite" / "dynamic_model.onnx"
    full.write_bytes(b"synthetic-dynamic-model")
    stable_root = tmp_path / "persistent" / "model-identity"
    engine = module._native_trt_engine_path(
        "full", full, "fp16", stable_root, canonical_role_layout=True,
    )
    engine.parent.mkdir(parents=True)
    persistent_source = engine.parent / "source.onnx"
    persistent_source.write_bytes(full.read_bytes())
    engine.write_bytes(b"engine-built-for-wrong-profile")
    module._write_explicit_native_trt_engine_receipt(
        source_onnx=persistent_source,
        engine_path=engine,
        receipt_path=engine.parent / "engine_build_receipt.json",
        trtexec=Path(sys.executable),
        command=[
            str(Path(sys.executable).resolve()),
            f"--onnx={persistent_source}",
            f"--saveEngine={engine}",
            "--fp16",
            "--shapes=input:99x3x640x640",
            "--memPoolSize=workspace:4096",
        ],
        returncode=0,
    )
    monkeypatch.setattr(
        module,
        "_onnx_static_io_info",
        lambda _path: ([('input', [1, 3, 640, 640], True)], []),
    )
    monkeypatch.setattr(
        module.NativeTRTSession,
        "_load_engine",
        lambda *_a, **_k: pytest.fail("inference engine loaded before receipt rejection"),
    )
    monkeypatch.setattr(
        module.NativeTRTSession,
        "_build_engine",
        lambda *_a, **_k: pytest.fail("incompatible receipt triggered a hidden rebuild"),
    )

    with pytest.raises(RuntimeError, match="reason=build_contract_mismatch"):
        module.NativeTRTSession(
            "full",
            full,
            precision="fp16",
            workspace_mb=4096,
            allow_build=False,
            cache_root=stable_root,
            canonical_role_layout=True,
        )

    with pytest.raises(RuntimeError, match="input shape differs"):
        module._verify_explicit_native_trt_engine_receipt(
            source_onnx=full,
            engine_path=engine,
            receipt_path=engine.parent / "engine_build_receipt.json",
            expected_precision="fp16",
            expected_workspace_mb=4096,
        )


@pytest.mark.parametrize(
    ("build_flags", "message"),
    [
        (["--int8", "--memPoolSize=workspace:4096"], "precision differs"),
        (["--fp16", "--workspace=2048"], "workspace differs"),
        (
            [
                "--fp16",
                "--memPoolSize=workspace:4096",
                "--builderOptimizationLevel=5",
            ],
            "unrequested build-affecting arguments",
        ),
        (
            [
                "--fp16",
                "--memPoolSize=workspace:4096",
                "--minShapes=input:1x3x640x640",
                "--optShapes=input:1x3x640x640",
                "--maxShapes=input:4x3x640x640",
            ],
            "input profile differs",
        ),
    ],
)
def test_receipt_reuse_binds_all_runtime_engine_build_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    build_flags: list[str],
    message: str,
) -> None:
    module = _runner(tmp_path, monkeypatch)
    source = tmp_path / "suite" / "model.onnx"
    source.write_bytes(b"synthetic-static-model")
    engine = tmp_path / "cache" / "full_fp16.engine"
    engine.parent.mkdir(parents=True)
    persistent_source = engine.parent / "source.onnx"
    persistent_source.write_bytes(source.read_bytes())
    engine.write_bytes(b"incompatible-engine")
    receipt = engine.parent / "engine_build_receipt.json"
    module._write_explicit_native_trt_engine_receipt(
        source_onnx=persistent_source,
        engine_path=engine,
        receipt_path=receipt,
        trtexec=Path(sys.executable),
        command=[
            str(Path(sys.executable).resolve()),
            f"--onnx={persistent_source}",
            f"--saveEngine={engine}",
            *build_flags,
        ],
        returncode=0,
    )
    monkeypatch.setattr(
        module,
        "_onnx_static_io_info",
        lambda _path: ([('input', [1, 3, 640, 640], False)], []),
    )

    with pytest.raises(RuntimeError, match=message):
        module._verify_explicit_native_trt_engine_receipt(
            source_onnx=source,
            engine_path=engine,
            receipt_path=receipt,
            expected_precision="fp16",
            expected_workspace_mb=4096,
        )


def test_ort_trt_fallback_uses_model_scoped_full_cache() -> None:
    """Native-preferred fallback must not put Full below a split cache."""

    template = (
        Path(__file__).parents[1]
        / "onnx_splitpoint_tool/resources/templates/run_split_onnxruntime.py.txt"
    ).read_text(encoding="utf-8")

    assert "providers_local: List[str], *, kind: str" in template
    assert 'if kind == "full" else ""' in template
    assert "popts = _make_provider_options_for(provs, kind=kind)" in template
