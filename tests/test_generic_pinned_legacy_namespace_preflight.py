"""Pinned execution needs receipt verification, without namespace migration."""
import pytest

from onnx_splitpoint_tool.benchmark import remote_run
from tests.test_generic_quality_artifact_preflight import _bridge_fixture
from tests.test_v27920_remote_trt_cache_preflight import _builder_abi_sha256, _owner
from tests.test_v27923_trt_legacy_reuse import _fixture as _migration_fixture, _probe


def _legacy_bridge(tmp_path, monkeypatch):
    # Create the existing, fully sealed eight-artifact bridge under an older
    # model namespace. Restore current key calculation before normal preflight.
    with monkeypatch.context() as patch:
        patch.setattr(remote_run, "_stable_trt_engine_cache_key",
                      lambda *args, **kwargs: "yolo26s-1111111111111111")
        fixture = _bridge_fixture(tmp_path, 1)
    return fixture


def test_explicit_pinned_bridge_in_nonmigratable_namespace_is_verified(tmp_path, monkeypatch):
    f = _legacy_bridge(tmp_path, monkeypatch)
    before = {p: p.read_bytes() for p in f["namespace"].rglob("*") if p.is_file()}
    result = f["probe"]()
    assert result["local_errors"] == []
    row = result["observations"][0]
    assert row["status"] == "HIT", row
    assert row["evidence"]["namespace_owner_status"] == "legacy_namespace_not_runtime_migratable"
    assert row["artifact_path"] == str(f["paths"]["engine"])
    assert row["evidence"]["source_binding"] == "generic_quality_existing_part2_bridge"
    assert row["evidence"]["validator_status"] == "local_files_rehashed_and_exact_cross_links_verified"
    assert row["evidence"]["binding_sha256"] == f["binding"]["binding_sha256"]
    assert len(f["transport"].commands) == 1
    assert {p: p.read_bytes() for p in f["namespace"].rglob("*") if p.is_file()} == before


def test_ordinary_nonmigratable_namespace_remains_rejected(tmp_path):
    _, _, _, _, _, calls, payload, _ = _migration_fixture(tmp_path)
    payload["legacy_suite_key"] = "model-another-suite"
    payload["canonical_full_sha256"] = []
    row = _probe(payload)
    assert row["status"] != "HIT"
    assert row["reason"] == "legacy_namespace_not_runtime_migratable"
    assert not calls.exists()


@pytest.mark.parametrize("fault", ["wrong_identity", "missing_owner", "wrong_owner_abi"])
def test_pinned_nonmigratable_namespace_keeps_identity_and_owner_guards(tmp_path, monkeypatch, fault):
    f = _legacy_bridge(tmp_path, monkeypatch)
    if fault == "wrong_identity":
        f["config"]["identity"]["setup_id"] = "wrong-setup"
        f["publish"]()
    elif fault == "missing_owner":
        (f["namespace"] / ".splitpoint_trt_cache_owner.json").unlink()
    else:
        _owner(f["namespace"], builder_abi_sha256="a" * 64)
        assert _builder_abi_sha256(f["abi"]) != "a" * 64
    result = f["probe"]()
    row = result["observations"][0]
    assert row["status"] != "HIT", row
    if fault == "wrong_identity":
        assert result["local_errors"]
        assert f["transport"].commands == []
    else:
        expected = "legacy_owner_missing" if fault == "missing_owner" else "legacy_owner_builder_abi_mismatch"
        assert row["reason"] == expected
