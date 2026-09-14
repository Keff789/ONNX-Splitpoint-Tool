from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from onnx_splitpoint_tool import hailo_backend as backend
from onnx_splitpoint_tool.hailo_full_contract_promotion import (
    _receipt_path_hints, _validate_hailo_build_receipt,
    promote_verified_hailo_full_contracts,
)
from onnx_splitpoint_tool.preprocessing_contract import (
    canonical_image_preprocessing_contract, preprocessing_contract_sha256,
)
from onnx_splitpoint_tool.workflow.native_transfer import (
    build_native_transfer_inventory, write_rsync_files_from,
)


@pytest.fixture
def suite(tmp_path, monkeypatch):
    monkeypatch.setattr(backend, "_hailo_sdk_version_token", lambda: "hailo-dataflow-compiler:3.31.0")
    root = tmp_path / "suite"
    source = root / "models/mobilenet_v3_large.onnx"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"selected-classification-onnx")
    container = root / "hailo/hailo8/full"
    container.mkdir(parents=True)
    compiler = container / "mobilenet_v3_large_hailo_fixed.onnx"
    compiler.write_bytes(b"fixed-compiler-onnx")
    original = tmp_path / "built.hef"
    original.write_bytes(b"compiled-hef")
    contract = canonical_image_preprocessing_contract("classification", (224, 224))
    key, payload = backend._hailo_cache_key(
        model_path=compiler, activation_part1=None, hw_arch="hailo8", opt_level=1,
        calib_dir=None, calib_count=1, calib_batch_size=1, extra_model_script="",
        start_nodes=None, end_nodes=None, preprocessing_contract=contract,
        effective_calib_count=1, calibration_storage="memory",
        calibration_memory_cap_bytes=64 * 1024 * 1024,
        net_name="mobilenet_v3_large_full", net_input_shapes={"images": [1, 3, 224, 224]},
        disable_rt_metadata_extraction=True,
    )
    receipt = backend._write_hailo_receipt(
        hef_path=original, source_onnx=source, compiler_onnx=compiler,
        hw_arch="hailo8", net_name="mobilenet_v3_large_full",
        preprocessing_contract=contract,
        preprocessing_sha256=preprocessing_contract_sha256(contract),
        cache_key=key, cache_payload=payload,
        calibration_identity=payload["calibration_identity"], calibration_count=1,
    )
    hef = backend._publish_hailo_bundle(
        source_hef=original, destination=container / "compiled.hef", receipt=receipt,
    )
    bench = {"model": "models/mobilenet_v3_large.onnx", "model_source": "/old/host/model.onnx",
             "hailo": {"hefs": {"hailo8": {"full": "hailo/hailo8/full/compiled.hef"}}}}
    (root / "benchmark_set.json").write_text(json.dumps(bench))
    return root, hef, compiler, bench


def _contracts():
    return [{"model_id": "mobilenet_v3_large", "task": "classification",
             "backend": "hailo8", "variant": "full", "requested": True,
             "endpoint_mode": "classification_logits", "contract_status": "pending_build_or_prepare",
             "host_tail_required": False, "postprocessing_required": False}]


def test_pinned_generation_finds_signed_compiler_sibling_and_portable_model(suite):
    root, hef, compiler, bench = suite
    source_hints, compiler_hints = _receipt_path_hints(artifact=hef, suite_bench=bench)
    assert compiler in compiler_hints
    assert root / "models/mobilenet_v3_large.onnx" in source_hints
    evidence = _validate_hailo_build_receipt(
        hef, source_onnx_candidates=source_hints, compiler_onnx_candidates=compiler_hints,
        expected_backend="hailo8", expected_task="classification",
    )
    assert evidence["valid"], evidence["errors"]
    contracts = _contracts()
    promoted = promote_verified_hailo_full_contracts(
        suite_dir=root, model_id="mobilenet_v3_large", task="classification",
        suite_bench=bench, contracts=contracts, copied_verified={},
    )
    assert len(promoted) == 1
    assert contracts[0]["contract_status"] == "recorded"


@pytest.mark.parametrize("changed", ["compiler", "source"])
def test_new_path_hints_never_admit_changed_onnx_bytes(suite, changed):
    root, hef, compiler, bench = suite
    path = compiler if changed == "compiler" else root / "models/mobilenet_v3_large.onnx"
    path.write_bytes(b"different-model")
    assert promote_verified_hailo_full_contracts(
        suite_dir=root, model_id="mobilenet_v3_large", task="classification",
        suite_bench=bench, contracts=_contracts(), copied_verified={},
    ) == []


def test_exact_rsync_inventory_keeps_atomic_aliases_and_full_reader(suite, tmp_path):
    root, hef, compiler, bench = suite
    if shutil.which("rsync") is None:
        pytest.skip("rsync is required for the real local transfer regression")
    inventory = build_native_transfer_inventory(root)
    prefix = "hailo/hailo8/full/"
    assert {prefix + name for name in ["compiled.hef", "hailo_hef_build_receipt.json", "cache_meta.json", ".hailo-current"]} <= set(inventory["relative_paths"])
    paths_file = write_rsync_files_from(inventory, tmp_path / "files.txt")
    destination = tmp_path / "remote"
    destination.mkdir()
    subprocess.run(["rsync", "-a", "--files-from", str(paths_file), "--relative",
                    str(root) + "/", str(destination) + "/"], check=True, capture_output=True)
    alias = destination / prefix / "compiled.hef"
    assert alias.is_symlink() and alias.read_bytes() == hef.read_bytes()
    assert backend._load_valid_hailo_receipt(alias.resolve()) is not None
    script = Path(__file__).parents[1] / "scripts/native_full_baseline_eval_runner.py"
    spec = importlib.util.spec_from_file_location("full_reader_v27924", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module._find_hailo_full_hef(destination, "hailo8") == alias.resolve()
    assert len(promote_verified_hailo_full_contracts(
        suite_dir=destination, model_id="mobilenet_v3_large", task="classification",
        suite_bench=bench, contracts=_contracts(), copied_verified={},
    )) == 1


def test_transfer_rejects_broken_bundle_and_ignores_unrelated_symlink(suite):
    root, hef, _, _ = suite
    (root / "unrelated.onnx").symlink_to("/outside/model.onnx")
    assert "unrelated.onnx" not in build_native_transfer_inventory(root)["relative_paths"]
    hef.unlink()
    with pytest.raises(ValueError, match="native_transfer_invalid_hailo_bundle_alias"):
        build_native_transfer_inventory(root)


def test_excluded_previous_run_alias_cannot_block_current_transfer(suite):
    root, _, _, _ = suite
    stale = root / "native_pipeline/old/compiled.hef"
    stale.parent.mkdir(parents=True)
    stale.symlink_to("/missing/old.hef")
    assert "native_pipeline/old/compiled.hef" not in build_native_transfer_inventory(root)["relative_paths"]


def test_publisher_model_hef_alias_is_transferred(suite, tmp_path):
    _, hef, _, _ = suite
    root = tmp_path / "other_suite"
    alias = root / "hailo/hailo8/full/model.hef"
    backend._publish_hailo_bundle(
        source_hef=hef, destination=alias,
        receipt=backend._load_valid_hailo_receipt(hef),
    )
    inventory = build_native_transfer_inventory(root)
    assert "hailo/hailo8/full/model.hef" in inventory["relative_paths"]
    assert "hailo/hailo8/full/.hailo-current" in inventory["relative_paths"]
