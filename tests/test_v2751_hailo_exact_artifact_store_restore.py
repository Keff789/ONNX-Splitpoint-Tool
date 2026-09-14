from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import onnx
import pytest
from onnx import TensorProto, helper

import onnx_splitpoint_tool.hailo_backend as hailo_backend
from onnx_splitpoint_tool.artifact_store import ArtifactStore
from onnx_splitpoint_tool.preprocessing_contract import (
    canonical_image_preprocessing_contract,
    preprocessing_contract_sha256,
)


_SDK_TOKEN = "hailo-dataflow-compiler:3.31.0"


def _write_identity_model(path: Path) -> None:
    graph = helper.make_graph(
        [helper.make_node("Identity", ["images"], ["output"], name="identity")],
        "hailo-cache-fixture",
        [helper.make_tensor_value_info(
            "images", TensorProto.FLOAT, [1, 3, 224, 224]
        )],
        [helper.make_tensor_value_info(
            "output", TensorProto.FLOAT, [1, 3, 224, 224]
        )],
    )
    onnx.save(helper.make_model(graph), path)


def _register_v275_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    metadata_mutator=None,
    contract_mutator=None,
    tamper_object: bool = False,
) -> dict:
    store_root = tmp_path / "artifact-store"
    cache_root = tmp_path / "local-cache"
    monkeypatch.setenv("ONNX_SPLITPOINT_ARTIFACT_STORE_ENABLED", "1")
    monkeypatch.setenv("ONNX_SPLITPOINT_ARTIFACT_STORE_ROOT", str(store_root))
    monkeypatch.setenv("ONNX_SPLITPOINT_HAILO_CACHE_ROOT", str(cache_root))
    monkeypatch.setenv("ONNX_SPLITPOINT_HAILO_CACHE_ENABLED", "1")
    monkeypatch.setenv("ONNX_SPLITPOINT_HAILO_CALIBRATION_STORAGE", "memory")
    monkeypatch.setenv("ONNX_SPLITPOINT_HAILO_CALIB_CAP_MB", "64")
    monkeypatch.setattr(
        hailo_backend, "_hailo_sdk_version_token", lambda: _SDK_TOKEN
    )
    # The compiler-free controller lookup reads selected-venv distribution
    # metadata. Model that external identity with the same SDK token used to
    # publish this fixture; an SDK-import fallback is intentionally forbidden.
    monkeypatch.setattr(
        hailo_backend, "_hailo_sdk_version_token_from_managed_venv",
        lambda **_kwargs: _SDK_TOKEN,
    )

    model = tmp_path / "model.onnx"
    _write_identity_model(model)
    old_output = tmp_path / "v275-output"
    old_output.mkdir()
    old_hef = old_output / "compiled.hef"
    old_hef.write_bytes(b"verified-v2.75-hef-bytes")

    preprocessing = canonical_image_preprocessing_contract(
        "classification", (224, 224)
    )
    storage = hailo_backend._resolve_hailo_calibration_storage(1)
    memory_cap = hailo_backend._calibration_memory_cap_bytes()
    cache_key, cache_payload = hailo_backend._hailo_cache_key(
        model_path=model,
        activation_part1=None,
        hw_arch="hailo8",
        opt_level=1,
        calib_dir=None,
        calib_count=1,
        calib_batch_size=1,
        extra_model_script="",
        start_nodes=None,
        end_nodes=None,
        preprocessing_contract=preprocessing,
        effective_calib_count=1,
        calibration_storage=storage,
        calibration_memory_cap_bytes=memory_cap,
    )
    receipt = hailo_backend._write_hailo_receipt(
        hef_path=old_hef,
        source_onnx=model,
        compiler_onnx=model,
        hw_arch="hailo8",
        net_name="fixture",
        preprocessing_contract=preprocessing,
        preprocessing_sha256=preprocessing_contract_sha256(preprocessing),
        cache_key=cache_key,
        cache_payload=cache_payload,
        calibration_identity=str(cache_payload["calibration_identity"]),
        calibration_count=int(cache_payload["calibration_count"]),
    )
    metadata = {
        "legacy_cache_key": cache_key,
        "preprocessing_contract_sha256": preprocessing_contract_sha256(
            preprocessing
        ),
        "build_receipt": copy.deepcopy(receipt),
        "bridge": "hailo_build_hef",
    }
    if metadata_mutator is not None:
        metadata_mutator(metadata)
    record_contract = {
        "schema": "onnx-splitpoint/hailo-build-contract/v2",
        "tool_version": "2.75.0",
        "source_run": "v275-smoke",
    }
    if contract_mutator is not None:
        contract_mutator(record_contract)
    record = ArtifactStore(store_root).register(
        source_path=old_hef,
        kind="hailo_hef",
        # The exact restore must not depend on this historical/tool contract.
        contract=record_contract,
        metadata=metadata,
        source_run="v275-smoke",
    )
    if tamper_object:
        Path(record.object_path).write_bytes(b"tampered-object")
    return {
        "model": model,
        "preprocessing": preprocessing,
        "cache_key": cache_key,
        "cache_payload": cache_payload,
        "cache_root": cache_root,
        "store_root": store_root,
        "old_hef": old_hef,
    }


def _build_kwargs(fixture: dict, output: Path) -> dict:
    return {
        "backend": "venv",
        "hw_arch": "hailo8",
        "net_name": "fixture",
        "outdir": output,
        "net_input_shapes": {"images": [1, 3, 224, 224]},
        "fixup": False,
        "opt_level": 1,
        "calib_count": 1,
        "calib_batch_size": 1,
        "task": "classification",
        "preprocessing_contract": fixture["preprocessing"],
    }


def _current_v3_key(fixture: dict) -> tuple[str, dict]:
    return hailo_backend._hailo_cache_key(
        model_path=fixture["model"], activation_part1=None,
        hw_arch="hailo8", opt_level=1, calib_dir=None,
        calib_count=1, calib_batch_size=1, extra_model_script="",
        start_nodes=None, end_nodes=None,
        preprocessing_contract=fixture["preprocessing"],
        effective_calib_count=1, calibration_storage="memory",
        calibration_memory_cap_bytes=64 * 1024 * 1024,
        net_name="fixture",
        net_input_shapes={"images": [1, 3, 224, 224]},
        disable_rt_metadata_extraction=True,
    )


def test_v275_artifact_store_record_restores_by_exact_v2_key_without_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _register_v275_record(tmp_path, monkeypatch)
    output = tmp_path / "v2751-output"
    dispatch_calls = []

    def _blocked_dispatch(*args, **kwargs):
        dispatch_calls.append((args, kwargs))
        raise AssertionError("DFC venv dispatcher must not run on an exact restore")

    monkeypatch.setattr(
        hailo_backend, "hailo_build_hef_via_venv", _blocked_dispatch
    )
    monkeypatch.setenv("ONNX_SPLITPOINT_RUN_ID", "v2751-new-run")
    result = hailo_backend.hailo_build_hef_auto(
        fixture["model"], **_build_kwargs(fixture, output)
    )

    assert dispatch_calls == []
    assert result.ok is True
    assert result.skipped is True
    assert result.backend == "artifact_store"
    assert Path(result.hef_path).read_bytes() == fixture["old_hef"].read_bytes()
    assert result.calib_info["cache_source"] == "artifact_store_exact_v2"
    current_key, current_payload = _current_v3_key(fixture)
    assert current_key != fixture["cache_key"]
    assert result.calib_info["cache_key"] == current_key
    assert result.calib_info["cache_backfilled"] is True
    migrated_receipt = hailo_backend._load_valid_hailo_receipt(
        Path(result.hef_path),
        preprocessing_sha256=preprocessing_contract_sha256(
            fixture["preprocessing"]
        ),
        source_onnx_sha256=hailo_backend._bare_file_sha256(fixture["model"]),
        cache_key=current_key,
        cache_payload=current_payload,
        expected_net_name="fixture",
        expected_net_input_shapes={"images": [1, 3, 224, 224]},
        expected_disable_rt_metadata_extraction=True,
    )
    assert migrated_receipt is not None
    assert migrated_receipt["migrated_from_cache_key"] == fixture["cache_key"]

    cache_dir = fixture["cache_root"] / current_key
    assert (cache_dir / "compiled.hef").read_bytes() == fixture[
        "old_hef"
    ].read_bytes()
    cache_meta = json.loads((cache_dir / "cache_meta.json").read_text())
    assert cache_meta["cache_key"] == current_key
    assert cache_meta["source"] == "artifact_store_exact_v2_restore"

    # The backfilled historical exact-key cache remains the first reuse source.
    monkeypatch.setenv("ONNX_SPLITPOINT_ARTIFACT_STORE_ENABLED", "0")
    second_output = tmp_path / "second-output"
    local_hit = hailo_backend.hailo_build_hef(
        fixture["model"],
        **{
            key: value
            for key, value in _build_kwargs(fixture, second_output).items()
            if key != "backend"
        },
    )
    assert local_hit.ok is True
    assert local_hit.skipped is True
    assert local_hit.backend == "local"
    assert local_hit.calib_info["cache_key"] == current_key
    assert Path(local_hit.hef_path).read_bytes() == fixture["old_hef"].read_bytes()


@pytest.mark.parametrize(
    "mutation,tamper_object",
    [
        (lambda metadata: metadata.pop("legacy_cache_key"), False),
        (
            lambda metadata: metadata["build_receipt"].update(
                {"hef_sha256": "0" * 64}
            ),
            False,
        ),
        (
            lambda metadata: metadata["build_receipt"]["cache_payload"].update(
                {"optimization_level": 999}
            ),
            False,
        ),
        (None, True),
    ],
    ids=[
        "missing-legacy-key",
        "receipt-object-hash-mismatch",
        "receipt-payload-mismatch",
        "artifact-object-tampered",
    ],
)
def test_exact_restore_rejects_coarse_or_mismatched_v275_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation,
    tamper_object: bool,
) -> None:
    fixture = _register_v275_record(
        tmp_path,
        monkeypatch,
        metadata_mutator=mutation,
        tamper_object=tamper_object,
    )
    output = tmp_path / "rejected-output"
    result = hailo_backend._hailo_build_hef_legacy(
        fixture["model"],
        **{
            key: value
            for key, value in _build_kwargs(fixture, output).items()
            if key != "backend"
        },
        cache_only=True,
    )

    assert result.ok is False
    assert result.failure_kind == "deferred_cold_full_cache_miss"
    assert not (output / "compiled.hef").exists()
    assert not (fixture["cache_root"] / fixture["cache_key"] / "compiled.hef").exists()


def test_coarse_restore_shim_never_materializes_before_exact_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _register_v275_record(tmp_path, monkeypatch)
    destination = tmp_path / "coarse-output"
    bound = {
        "onnx_path": str(fixture["model"]),
        "outdir": destination,
        "hw_arch": "hailo8",
    }

    assert hailo_backend._v60s_hailo_restore(
        bound, hailo_backend._v60s_hailo_contract(bound)
    ) is None
    assert not (destination / "compiled.hef").exists()


def test_legacy_v2_net_name_mismatch_is_not_a_hit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _register_v275_record(tmp_path, monkeypatch)
    output = tmp_path / "wrong-name-output"
    kwargs = _build_kwargs(fixture, output)
    kwargs["net_name"] = "different-requested-name"
    result = hailo_backend._hailo_build_hef_legacy(
        fixture["model"],
        **{key: value for key, value in kwargs.items() if key != "backend"},
        cache_only=True,
    )

    assert result.ok is False
    assert result.failure_kind == "deferred_cold_full_cache_miss"
    assert not (output / "compiled.hef").exists()


@pytest.mark.parametrize(
    "contract_axis",
    [
        {"net_name": "other"},
        {"net_input_shapes": {"images": [1, 3, 256, 256]}},
        {"disable_rt_metadata_extraction": False},
    ],
    ids=["net-name", "input-shapes", "rt-metadata-flag"],
)
def test_legacy_v2_restore_rejects_explicit_contract_axis_contradictions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    contract_axis: dict,
) -> None:
    fixture = _register_v275_record(
        tmp_path,
        monkeypatch,
        contract_mutator=lambda contract: contract.update(contract_axis),
    )
    output = tmp_path / "contradicting-contract-output"
    result = hailo_backend._hailo_build_hef_legacy(
        fixture["model"],
        **{
            key: value
            for key, value in _build_kwargs(fixture, output).items()
            if key != "backend"
        },
        cache_only=True,
    )

    assert result.ok is False
    assert result.failure_kind == "deferred_cold_full_cache_miss"
    assert not (output / "compiled.hef").exists()


def test_semantic_v3_key_binds_all_translate_axes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _register_v275_record(tmp_path, monkeypatch)
    base = dict(
        model_path=fixture["model"], activation_part1=None,
        hw_arch="hailo8", opt_level=1, calib_dir=None,
        calib_count=1, calib_batch_size=1, extra_model_script="",
        start_nodes=None, end_nodes=None,
        preprocessing_contract=fixture["preprocessing"],
        effective_calib_count=1, calibration_storage="memory",
        calibration_memory_cap_bytes=64 * 1024 * 1024,
        net_name="fixture",
        net_input_shapes={"images": [1, 3, 224, 224]},
        disable_rt_metadata_extraction=True,
    )
    key, payload = hailo_backend._hailo_cache_key(**base)
    assert payload["schema"] == "onnx-splitpoint/hailo-hef-cache-key-v3"
    variants = [
        {"net_name": "other"},
        {"net_input_shapes": {"images": [1, 3, 256, 256]}},
        {"disable_rt_metadata_extraction": False},
    ]
    for change in variants:
        changed_key, _ = hailo_backend._hailo_cache_key(
            **{**base, **change}
        )
        assert changed_key != key


def test_nonsemantic_duplicate_receipt_fields_restore_without_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _register_v275_record(tmp_path, monkeypatch)
    receipt = json.loads(
        hailo_backend._hailo_receipt_path(fixture["old_hef"]).read_text()
    )
    receipt["created_at_unix_s"] = float(receipt["created_at_unix_s"]) + 10
    receipt["migrated_from_cache_key"] = "historical-v2-source-key"
    ArtifactStore(fixture["store_root"]).register(
        source_path=fixture["old_hef"], kind="hailo_hef",
        contract={
            "schema": "onnx-splitpoint/hailo-build-contract/v2",
            "tool_version": "2.75.0", "source_run": "duplicate",
        },
        metadata={
            "legacy_cache_key": fixture["cache_key"],
            "preprocessing_contract_sha256": preprocessing_contract_sha256(
                fixture["preprocessing"]
            ),
            "build_receipt": receipt,
            "bridge": "hailo_build_hef",
        },
        source_run="duplicate",
    )
    calls = []
    monkeypatch.setattr(
        hailo_backend, "hailo_build_hef_via_venv",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    result = hailo_backend.hailo_build_hef_auto(
        fixture["model"],
        **_build_kwargs(fixture, tmp_path / "duplicate-output"),
    )

    assert calls == []
    assert result.ok is True
    assert result.skipped is True


def test_auto_normalizes_net_name_and_forwards_shapes_to_dispatcher(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = tmp_path / "model.onnx"
    _write_identity_model(model)
    preprocessing = canonical_image_preprocessing_contract(
        "classification", (224, 224)
    )
    calls: list[dict] = []

    def fake_venv(_model: Path, **kwargs):
        calls.append(dict(kwargs))
        return hailo_backend.HailoHefBuildResult(
            ok=False,
            elapsed_s=0.0,
            hw_arch=str(kwargs["hw_arch"]),
            net_name=str(kwargs["net_name"]),
            backend="venv",
            failure_kind="launch_error",
            error="blocked test dispatcher",
        )

    monkeypatch.setattr(hailo_backend, "hailo_build_hef_via_venv", fake_venv)
    result = hailo_backend.hailo_build_hef_auto(
        model,
        backend="venv",
        hw_arch="hailo8",
        net_name="  fixture  ",
        outdir=tmp_path / "dispatch-output",
        net_input_shapes={"images": [1, 3, 224, 224]},
        force=False,  # v34 productive API builds an ordinary missing artifact.
        task="classification",
        preprocessing_contract=preprocessing,
    )

    assert result.ok is False
    assert len(calls) == 1
    assert calls[0]["net_name"] == "fixture"
    assert calls[0]["net_input_shapes"] == {
        "images": [1, 3, 224, 224]
    }


def test_managed_venv_serializes_shapes_for_helper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from onnx_splitpoint_tool import paths

    model = tmp_path / "model.onnx"
    _write_identity_model(model)
    preprocessing = canonical_image_preprocessing_contract(
        "classification", (224, 224)
    )
    captured: list[str] = []
    monkeypatch.setattr(
        hailo_backend,
        "_resolve_managed_venv_python",
        lambda **_kwargs: ("fixture-profile", Path(sys.executable), ""),
    )
    monkeypatch.setattr(
        paths, "splitpoint_logs_dir", lambda: tmp_path / "logs"
    )

    def fake_subprocess(command: list[str], **_kwargs):
        captured.extend(command)
        payload = {
            "ok": True,
            "elapsed_s": 0.01,
            "hw_arch": "hailo8",
            "net_name": "fixture",
            "backend": "venv",
            "hef_path": str(tmp_path / "compiled.hef"),
        }
        return hailo_backend._StreamedSubprocessResult(
            returncode=0,
            stdout=(
                hailo_backend._WSL_RESULT_MARKER
                + json.dumps(payload)
            ),
            stderr="",
        )

    monkeypatch.setattr(
        hailo_backend, "_run_streamed_subprocess", fake_subprocess
    )
    result = hailo_backend.hailo_build_hef_via_venv(
        model,
        hw_arch="hailo8",
        net_name=" fixture ",
        outdir=tmp_path / "venv-output",
        net_input_shapes={"images": [1, 3, 224, 224]},
        task="classification",
        preprocessing_contract=preprocessing,
    )

    assert result.ok is True
    assert result.net_name == "fixture"
    shapes_index = captured.index("--net-input-shapes-json")
    assert json.loads(captured[shapes_index + 1]) == {
        "images": [1, 3, 224, 224]
    }
