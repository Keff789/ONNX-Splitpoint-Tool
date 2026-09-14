from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tarfile

import pytest

from onnx_splitpoint_tool import hailo_backend as backend
from onnx_splitpoint_tool.hailo_full_contract_promotion import (
    _receipt_path_hints, _validate_hailo_build_receipt,
    promote_verified_hailo_full_contracts,
)
from onnx_splitpoint_tool.native_output_endpoint import load_authoritative_output_contract
from onnx_splitpoint_tool.remote.bundle import build_suite_bundle, remote_minimal_bundle_patterns
from test_v27924_hailo_full_portability import suite, _contracts


def _prepare_contract(root, bench):
    contracts = _contracts()
    assert len(promote_verified_hailo_full_contracts(
        suite_dir=root, model_id="mobilenet_v3_large", task="classification",
        suite_bench=bench, contracts=contracts, copied_verified={},
    )) == 1
    # Matches MobileNet's historical declaration in both uploaded A contracts.
    contracts[0]["endpoint_mode"] = "decoded"
    (root / "output_contracts.json").write_text(json.dumps({
        "model_id": "mobilenet_v3_large", "task": "classification", "contracts": contracts,
    }))
    return contracts[0]


def _load(root):
    return load_authoritative_output_contract(
        root, backend="hailo8", model_id="mobilenet_v3_large",
        variant="full", task="classification",
    )


def _archive(root, destination):
    includes, excludes = remote_minimal_bundle_patterns()
    stats = build_suite_bundle(root, destination, includes=includes, excludes=excludes)
    return stats


def test_tar_roundtrip_keeps_full_contract_paths_and_receipt_without_duplicate_payload(suite, tmp_path):
    root, hef, _, bench = suite
    contract = _prepare_contract(root, bench)
    assert _load(root)["contract_resolution_status"] == "attested"
    before_contract = (root / "output_contracts.json").read_bytes()
    archive = tmp_path / "suite.tar.gz"
    _archive(root, archive)
    remote = tmp_path / "remote"
    with tarfile.open(archive) as tf:
        hef_payloads = [m for m in tf if m.name.endswith(".hef") and m.isfile()]
        assert len(hef_payloads) == 1
        for name in ("compiled.hef", "hailo_hef_build_receipt.json", "cache_meta.json"):
            member = tf.getmember((hef.parent / name).relative_to(root).as_posix())
            assert member.islnk()
            assert member.linkname == "hailo/hailo8/full/" + name
        tf.extractall(remote, filter="data")
    assert (remote / "output_contracts.json").read_bytes() == before_contract
    assert _load(remote)["contract_resolution_status"] == "attested"
    archived_hef = remote / contract["artifact_path"]
    source_hints, compiler_hints = _receipt_path_hints(artifact=archived_hef, suite_bench=bench)
    evidence = _validate_hailo_build_receipt(
        archived_hef, source_onnx_candidates=source_hints, compiler_onnx_candidates=compiler_hints,
        expected_backend="hailo8", expected_task="classification",
    )
    assert evidence["valid"], evidence["errors"]
    assert archived_hef.stat().st_ino == (remote / "hailo/hailo8/full/compiled.hef").stat().st_ino


@pytest.mark.parametrize("changed", ["hef", "compiler", "source", "receipt", "meta"])
def test_roundtrip_still_rejects_changed_bytes(suite, tmp_path, changed):
    root, hef, _, bench = suite
    contract = _prepare_contract(root, bench)
    archive = tmp_path / "suite.tar.gz"
    _archive(root, archive)
    remote = tmp_path / "remote"
    with tarfile.open(archive) as tf:
        tf.extractall(remote, filter="data")
    archived_hef = remote / contract["artifact_path"]
    paths = {
        "hef": archived_hef,
        "compiler": remote / "hailo/hailo8/full/mobilenet_v3_large_hailo_fixed.onnx",
        "source": remote / "models/mobilenet_v3_large.onnx",
        "receipt": archived_hef.parent / "hailo_hef_build_receipt.json",
        "meta": archived_hef.parent / "cache_meta.json",
    }
    paths[changed].write_bytes(b"changed-artifact")
    if changed == "hef":
        assert _load(remote)["contract_resolution_status"] != "attested"
    if changed == "meta":
        assert backend._load_valid_hailo_receipt(archived_hef) is None
    else:
        source_hints, compiler_hints = _receipt_path_hints(artifact=archived_hef, suite_bench=bench)
        checked = _validate_hailo_build_receipt(
            archived_hef, source_onnx_candidates=source_hints, compiler_onnx_candidates=compiler_hints,
            expected_backend="hailo8", expected_task="classification",
        )
        assert not checked["valid"]


@pytest.mark.parametrize("fault", ["wrong_alias", "missing_receipt", "tampered_meta"])
def test_invalid_publisher_cannot_be_bundled_as_verified_generation(suite, tmp_path, fault):
    root, hef, _, _ = suite
    alias = root / "hailo/hailo8/full/compiled.hef"
    if fault == "wrong_alias":
        alias.unlink()
        alias.symlink_to(hef)
    elif fault == "missing_receipt":
        (hef.parent / "hailo_hef_build_receipt.json").unlink()
    else:
        (hef.parent / "cache_meta.json").write_text("{}")
    with pytest.raises(ValueError, match="hailo_bundle_published_generation_invalid"):
        _archive(root, tmp_path / "invalid.tar.gz")


def test_new_generation_invalidates_archive_cache_without_transferring_old_generation(suite, tmp_path):
    root, hef, _, bench = suite
    _prepare_contract(root, bench)
    archive = tmp_path / "suite.tar.gz"
    first = _archive(root, archive)
    assert not first.reused
    assert _archive(root, archive).reused
    new_source = tmp_path / "new.hef"
    new_source.write_bytes(b"new-selected-generation")
    receipt = json.loads((hef.parent / "hailo_hef_build_receipt.json").read_text())
    receipt.update(hef_sha256=hashlib.sha256(new_source.read_bytes()).hexdigest(),
                   hef_size_bytes=new_source.stat().st_size)
    new_hef = backend._publish_hailo_bundle(
        source_hef=new_source, destination=root / "hailo/hailo8/full/compiled.hef", receipt=receipt,
    )
    changed = _archive(root, archive)
    assert not changed.reused
    with tarfile.open(archive) as tf:
        assert hef.relative_to(root).as_posix() not in tf.getnames()
        assert new_hef.relative_to(root).as_posix() in tf.getnames()
        tf.extractall(tmp_path / "changed_remote", filter="data")
    # The old recorded contract is not silently rebound to new HEF bytes.
    assert _load(tmp_path / "changed_remote")["contract_resolution_status"] != "attested"


def test_hef_only_legacy_is_not_promoted(suite):
    root, hef, _, bench = suite
    data = hef.read_bytes()
    alias = root / "hailo/hailo8/full/compiled.hef"
    alias.unlink()
    alias.write_bytes(data)
    (alias.parent / "hailo_hef_build_receipt.json").unlink()
    assert promote_verified_hailo_full_contracts(
        suite_dir=root, model_id="mobilenet_v3_large", task="classification",
        suite_bench=bench, contracts=_contracts(), copied_verified={},
    ) == []
