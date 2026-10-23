"""Byte-bound reuse and historical compile staging; scientific drift denies."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from onnx_splitpoint_tool.generic_split_manifest import validate_generic_split_manifest_staging


FIXTURE = Path(__file__).parent / "fixtures/generic_split_manifest"


def bindings(case="b177"):
    args = json.loads((FIXTURE / f"{case}_bindings.json").read_text())
    receipt = FIXTURE / f"{case}_part1_build_receipt.json"
    if receipt.exists():
        args["part1_build_receipt_bytes"] = receipt.read_bytes()
    return (FIXTURE / f"{case}_original.json").read_bytes(), (FIXTURE / f"{case}_staged.json").read_bytes(), args


def rewritten(document, args):
    raw = (json.dumps(document, sort_keys=True, indent=2) + "\n").encode()
    args["staged_source"].update(sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))
    return raw


def rewritten_original(document, args):
    raw = (json.dumps(document, sort_keys=True, indent=2) + "\n").encode()
    descriptor = {"sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)}
    args["original_source"].update(descriptor)
    args["row"]["generic_artifact_binding"]["artifacts"]["boundary_metadata"].update(descriptor)
    return raw


def bind_receipt(raw, args):
    args["part1_build_receipt_bytes"] = raw
    descriptor = {"sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)}
    args["original_artifacts"]["part1_build_receipt"].update(descriptor)
    args["row"]["generic_artifact_binding"]["artifacts"]["part1_build_receipt"].update(descriptor)


def test_actual_normal_staging_preserves_both_file_identities_and_all_bindings():
    original, staged, args = bindings()
    unchanged = deepcopy(args)
    proof = validate_generic_split_manifest_staging(original, staged, **args)
    assert args == unchanged
    assert proof["original_manifest"] == args["original_source"]
    assert proof["staged_manifest"] == args["staged_source"]
    assert proof["original_manifest"]["sha256"] != proof["staged_manifest"]["sha256"]
    assert proof["original_boundary_metadata_artifact"]["sha256"] == args["original_source"]["sha256"]
    assert set(proof["unchanged_artifact_bindings"]) == set(args["original_artifacts"])
    assert proof["prepared_input"]["sha256"] == args["row"]["functional_reference"]["input_sha256"]
    assert proof["scientific_manifest_fields_unchanged"] is True
    paths = {item["pointer"] for item in proof["allowed_differences"]}
    assert "/hailo/hefs/hailo10h" in paths
    assert "/hailo/hefs/hailo8/part1_build/source_onnx_path" in paths
    assert "/deepx/part1_dxnn/cache_lookup/outcome" in paths


@pytest.mark.parametrize("path,value", [
    (("boundary",), 178),
    (("part2",), "another_graph.onnx"),
    (("full_model",), "../models/another_model.onnx"),
    (("cut", "names_part1"), ["different_tensor"]),
    (("io", "part2_external_inputs"), ["different_input"]),
    (("pipeline", "stage2"), "another_stage"),
    (("hailo", "config", "calib_count"), 501),
    (("hailo", "config", "build", "full"), True),
    (("hailo", "hefs", "hailo8", "part1"), "different.hef"),
    (("hailo", "hefs", "hailo8", "part1_build", "source_onnx_path"), "/foreign/b177/yolo11l_part1_b177.onnx"),
    (("hailo", "hefs", "hailo8", "part1_build", "cache_hit"), False),
    (("deepx", "part1_dxnn", "dxnn_sha256"), "f" * 64),
    (("deepx", "part1_dxnn", "cache_contract", "input_shape"), [1, 3, 320, 320]),
    (("deepx", "part1_dxnn", "cache_contract", "calibration_count"), 501),
    (("unexpected_scientific_metadata",), {"source": "unreviewed"}),
])
@pytest.mark.parametrize("case", ["b177", "b239"])
def test_descriptor_valid_scientific_or_unreviewed_metadata_drift_is_rejected(path, value, case):
    original, staged, args = bindings(case)
    document = json.loads(staged)
    parent = document
    for key in path[:-1]:
        parent = parent[key]
    parent[path[-1]] = value
    with pytest.raises(ValueError, match="split-manifest staging"):
        validate_generic_split_manifest_staging(original, rewritten(document, args), **args)


def test_replacement_of_unused_full_contract_is_not_a_removal_exception():
    original, staged, args = bindings()
    document = json.loads(staged)
    document["hailo"]["hefs"]["hailo8"]["full_output_contract"] = {"artifact_sha256": "f" * 64}
    with pytest.raises(ValueError, match="unreviewed metadata or scientific"):
        validate_generic_split_manifest_staging(original, rewritten(document, args), **args)


@pytest.mark.parametrize("fault", [
    lambda a: a["row"]["identity"].update(comparison_backend="hailo10h"),
    lambda a: a["row"].update(admission_mode="paired"),
    lambda a: a["row"]["generic_artifact_binding"]["artifacts"]["boundary_metadata"].update(sha256="f" * 64),
    lambda a: a["row"]["generic_artifact_binding"]["artifacts"]["engine"].update(sha256="f" * 64),
    lambda a: a["row"]["generic_artifact_binding"]["artifacts"]["part1_runtime"].update(size_bytes=1),
    lambda a: a["producer_model"].update(part2_onnx_sha256="f" * 64),
    lambda a: a.update(full_model_sha256="f" * 64),
    lambda a: a["boundary_inputs"][0].update(shape=[1, 1]),
    lambda a: a["row"]["input_binding"]["prepared_input"].update(sha256="f" * 64),
    lambda a: a["row"]["input_binding"].update(runtime_input_format="uint8"),
])
@pytest.mark.parametrize("case", ["b177", "b239"])
def test_actual_artifact_graph_boundary_input_and_scope_bindings_are_required(fault, case):
    original, staged, args = bindings(case)
    fault(args)
    with pytest.raises(ValueError, match="split-manifest staging"):
        validate_generic_split_manifest_staging(original, staged, **args)


@pytest.mark.parametrize("which", ["original_source", "staged_source"])
def test_manifest_byte_descriptor_cannot_be_rewritten_or_omitted(which):
    original, staged, args = bindings()
    args[which]["sha256"] = "f" * 64
    with pytest.raises(ValueError, match="bytes differ from descriptor"):
        validate_generic_split_manifest_staging(original, staged, **args)


def test_actual_historical_successful_build_to_reuse_retains_exact_provenance():
    original, staged, args = bindings("b239")
    unchanged = deepcopy(args)
    proof = validate_generic_split_manifest_staging(original, staged, **args)
    assert args == unchanged
    assert proof["original_manifest"] == args["original_source"]
    assert proof["staged_manifest"] == args["staged_source"]
    assert proof["scientific_manifest_fields_unchanged"] is True
    assert len(proof["allowed_differences"]) == 58
    old = json.loads(original)
    for item in proof["historical_build_reuse"]:
        arch = item["backend_metadata"]
        assert item["historical_observation"] == old["hailo"]["hefs"][arch]["part1_build"]
        assert item["executed_artifact_identity_proved"] is (arch == "hailo8")
        assert item["compiler_source_byte_equality_claimed"] is False
        assert item["source_onnx_sha256"] == args["producer_model"]["part1_onnx_sha256"]
        if arch == "hailo8":
            actual = item["actual_receipt_binding"]
            assert actual["sha256"] == hashlib.sha256(args["part1_build_receipt_bytes"]).hexdigest()
            assert actual["cache_key"] == item["historical_observation"]["cache_key"]
            assert actual["hef_sha256"] == args["original_artifacts"]["part1_runtime"]["sha256"]
        else:
            assert "actual_receipt_binding" not in item
    paths = {change["pointer"] for change in proof["allowed_differences"]}
    assert "/hailo/hefs/hailo8/part1_build/build_evidence" in paths
    assert "/hailo/hefs/hailo8/part1_build/allocation_time_s" in paths
    assert "/hailo/hefs/hailo10/part1_build/context_mode" in paths


def retained_unused_historical_build():
    original, staged, args = bindings("b239")
    document = json.loads(staged)
    document["hailo"]["hefs"]["hailo10"]["part1_build"] = deepcopy(
        json.loads(original)["hailo"]["hefs"]["hailo10"]["part1_build"])
    return original, document, args


def test_unchanged_unused_historical_build_is_provenance_without_a_current_reuse_claim():
    original, document, args = retained_unused_historical_build()
    staged = rewritten(document, args)
    unchanged = deepcopy(args)
    proof = validate_generic_split_manifest_staging(original, staged, **args)
    assert args == unchanged
    assert [item["backend_metadata"] for item in proof["historical_build_reuse"]] == ["hailo8"]
    retained, = proof["retained_historical_build_metadata"]
    assert retained["historical_observation"] == document["hailo"]["hefs"]["hailo10"]["part1_build"]
    assert retained["build_record_unchanged"] is True
    assert retained["executed_artifact_identity_proved"] is False
    assert retained["current_compile_claimed"] is retained["current_reuse_claimed"] is False
    assert "actual_receipt_binding" not in retained
    assert not any(item["pointer"].startswith("/hailo/hefs/hailo10/part1_build")
                   for item in proof["allowed_differences"])
    assert proof["historical_build_reuse"][0]["actual_receipt_binding"]["sha256"] == (
        args["original_artifacts"]["part1_build_receipt"]["sha256"])


@pytest.mark.parametrize("key,value", [
    ("elapsed_s", 1.0),
    ("source_onnx_path", "/foreign/b239/yolo11l_part1_b239.onnx"),
    ("build_evidence", {}),
    ("single_context_failed", 1),
])
def test_unused_historical_record_must_remain_type_exactly_unchanged(key, value):
    original, document, args = retained_unused_historical_build()
    document["hailo"]["hefs"]["hailo10"]["part1_build"][key] = value
    with pytest.raises(ValueError, match="split-manifest staging"):
        validate_generic_split_manifest_staging(original, rewritten(document, args), **args)


@pytest.mark.parametrize("key,value", [
    ("ok", False),
    ("negative_evidence_hit", True),
    ("cache_key", "f" * 64),
    ("context_mode", "unknown_context"),
])
def test_equal_unused_historical_records_still_require_positive_bound_provenance(key, value):
    original, document, args = retained_unused_historical_build()
    old = json.loads(original)
    old["hailo"]["hefs"]["hailo10"]["part1_build"][key] = value
    document["hailo"]["hefs"]["hailo10"]["part1_build"] = deepcopy(
        old["hailo"]["hefs"]["hailo10"]["part1_build"])
    with pytest.raises(ValueError, match="split-manifest staging"):
        validate_generic_split_manifest_staging(
            rewritten_original(old, args), rewritten(document, args), **args)


def test_measured_h8_historical_record_cannot_replace_actual_current_reuse():
    original, document, args = retained_unused_historical_build()
    document["hailo"]["hefs"]["hailo8"]["part1_build"] = deepcopy(
        json.loads(original)["hailo"]["hefs"]["hailo8"]["part1_build"])
    with pytest.raises(ValueError, match="part1 must remain actual artifact reuse"):
        validate_generic_split_manifest_staging(original, rewritten(document, args), **args)


def current_only_terminal_summary(retain_full_metadata):
    original, document, args = retained_unused_historical_build()
    old = json.loads(original)
    if retain_full_metadata:
        for arch in ("hailo8", "hailo10h"):
            retained = document["hailo"]["hefs"].setdefault(arch, {})
            for key, value in old["hailo"]["hefs"][arch].items():
                if key.startswith("full"):
                    retained[key] = deepcopy(value)
            document["hailo"]["case_variant_availability"][arch]["full"] = True
    document["hailo"]["backend_terminal_states"] = [deepcopy(item) for item in
        old["hailo"]["backend_terminal_states"]
        if item["hw_arch"] == "hailo8" and item["variant"] == "part1"]
    return original, document, args


@pytest.mark.parametrize("retain_full_metadata", [False, True])
def test_current_h8_summary_may_omit_only_positive_unexecuted_historical_states(retain_full_metadata):
    original, document, args = current_only_terminal_summary(retain_full_metadata)
    proof = validate_generic_split_manifest_staging(original, rewritten(document, args), **args)
    assert len(proof["omitted_unexecuted_backend_terminal_states"]) == (3 if retain_full_metadata else 1)
    assert all(item["status"] == "ready" and item["available"] is True
               and (item["hw_arch"] != "hailo8" or item["variant"] != "part1")
               for item in proof["omitted_unexecuted_backend_terminal_states"])
    assert document["hailo"]["hefs"]["hailo10"]["part1_build"] == json.loads(original)["hailo"]["hefs"]["hailo10"]["part1_build"]
    assert proof["scientific_manifest_fields_unchanged"] is True


@pytest.mark.parametrize("fault", ["measured_changed", "measured_type", "measured_missing", "unused_failed", "unused_unknown"])
def test_current_summary_projection_cannot_hide_changed_unknown_or_failed_states(fault):
    original, document, args = current_only_terminal_summary(True)
    old = json.loads(original)
    if fault == "measured_changed":
        document["hailo"]["backend_terminal_states"][0]["reason"] = "different"
    elif fault == "measured_type":
        document["hailo"]["backend_terminal_states"][0]["available"] = 1
    elif fault == "measured_missing":
        document["hailo"]["backend_terminal_states"] = []
    else:
        unused = next(item for item in old["hailo"]["backend_terminal_states"]
                      if item["hw_arch"] == "hailo10h" and item["variant"] == "part1")
        if fault == "unused_failed":
            unused.update(status="failed", available=False)
        else:
            unused["unknown_diagnostic"] = True
        original = rewritten_original(old, args)
    with pytest.raises(ValueError, match="split-manifest staging"):
        validate_generic_split_manifest_staging(original, rewritten(document, args), **args)


def demoted_unexecuted_full_claims():
    from onnx_splitpoint_tool.hailo_full_contract_promotion import _demote_hailo_suite_full_claims
    original, document, args = current_only_terminal_summary(True)
    demoted = _demote_hailo_suite_full_claims(document)
    for arch in ("hailo8", "hailo10h"):
        for key, value in demoted["hailo"]["hefs"][arch].items():
            if key.startswith("full"):
                document["hailo"]["hefs"][arch][key] = deepcopy(value)
    return original, document, args


def test_unexecuted_full_claims_may_only_follow_the_existing_normal_demotion():
    original, document, args = demoted_unexecuted_full_claims()
    proof = validate_generic_split_manifest_staging(original, rewritten(document, args), **args)
    assert {item["backend_metadata"] for item in proof["demoted_unexecuted_full_claims"]} == {"hailo8", "hailo10h"}
    assert all(item["current_full_execution_claimed"] is False
               for item in proof["demoted_unexecuted_full_claims"])
    assert len(proof["unchanged_artifact_bindings"]) == 7


@pytest.mark.parametrize("fault", ["promoted", "contract_drift", "receipt_drift"])
def test_full_claim_demotion_cannot_promote_replace_or_change_historical_metadata(fault):
    original, document, args = demoted_unexecuted_full_claims()
    meta = document["hailo"]["hefs"]["hailo8"]
    if fault == "promoted":
        meta["full_receipt_attested"] = True
    elif fault == "contract_drift":
        meta["full_output_contract"]["unreviewed_semantic_contract"] = True
    else:
        meta["full_build_receipt"] = {"sha256": "f" * 64}
    with pytest.raises(ValueError, match="split-manifest staging"):
        validate_generic_split_manifest_staging(original, rewritten(document, args), **args)


def test_historical_relation_has_no_model_or_cut_id_exception():
    from onnx_splitpoint_tool.build_evidence import (
        boundary_endpoint_contract_sha256, canonical_build_key_from_hailo_v3_payload,
        canonical_sha256,
    )

    original, staged, args = bindings("b239")

    def renamed(value):
        if isinstance(value, dict):
            return {key: (517 if key in {"boundary", "boundary_index"} and item == 239
                          else renamed(item)) for key, item in value.items()}
        if isinstance(value, list):
            return [renamed(item) for item in value]
        if isinstance(value, str):
            return value.replace("yolo11l", "another_detector").replace("b239", "b517")
        return value

    document, staged_document, args = renamed(json.loads(original)), renamed(json.loads(staged)), renamed(args)
    for arch in ("hailo8", "hailo10"):
        build = document["hailo"]["hefs"][arch]["part1_build"]
        evidence = build["build_evidence"]
        payload = evidence["cache_payload_v3"]
        build["cache_key"] = evidence["cache_key_v3"] = canonical_sha256(payload)
        evidence["key"] = canonical_build_key_from_hailo_v3_payload(
            payload, builder_source_onnx_sha256=args["producer_model"]["part1_onnx_sha256"],
            full_source_onnx_sha256=args["full_model_sha256"],
            boundary_endpoint_contract_sha256=boundary_endpoint_contract_sha256(
                stage="part1", cache_payload=payload, split_manifest=document),
            expected_cache_key=build["cache_key"])
        evidence["key_sha256"] = canonical_sha256(evidence["key"])
    receipt = renamed(json.loads(args["part1_build_receipt_bytes"]))
    receipt["cache_payload"] = document["hailo"]["hefs"]["hailo8"]["part1_build"]["build_evidence"]["cache_payload_v3"]
    receipt["cache_key"] = document["hailo"]["hefs"]["hailo8"]["part1_build"]["cache_key"]
    bind_receipt(json.dumps(receipt, sort_keys=True).encode(), args)
    proof = validate_generic_split_manifest_staging(
        rewritten_original(document, args), rewritten(staged_document, args), **args)
    assert proof["measured_identity"]["case_id"] == "b517"
    assert proof["measured_identity"]["model_id"] == "another_detector"
    assert len(proof["historical_build_reuse"]) == 2


@pytest.mark.parametrize("arch", ["hailo8", "hailo10"])
@pytest.mark.parametrize("path,value", [
    (("ok",), False),
    (("cache_hit",), True),
    (("skipped",), True),
    (("timed_out",), True),
    (("mapping_failed",), True),
    (("watchdog_expired",), True),
    (("negative_evidence_hit",), True),
    (("context_mode",), "ok_unknown_context"),
    (("single_context_failed",), 1),
    (("context_count",), 1),
    (("partition_time_s",), -1),
    (("expensive_build_reason",), "unbound diagnostic"),
    (("build_evidence",), {}),
    (("build_evidence", "status"), "HIT"),
    (("build_evidence", "negative_evidence_hit"), True),
    (("build_evidence", "key_sha256"), "f" * 64),
    (("build_evidence", "key", "compiler_onnx_sha256"), "f" * 64),
    (("build_evidence", "context", "model_id"), "another_model"),
    (("build_evidence", "context", "split_manifest", "cut_tensors"), ["another_tensor"]),
    (("build_evidence", "cache_payload_v3", "calibration_batch_size"), 99),
    (("build_evidence", "cache_payload_v3", "unexpected_recipe"), "unknown"),
    (("compiler_onnx_path",), "/foreign/b239/hailo/hailo8/part1/yolo11l_part1_b239_hailo_fixed.onnx"),
    (("unknown_build_diagnostic",), True),
])
def test_unproved_failed_mixed_or_unknown_historical_build_denies(arch, path, value):
    original, staged, args = bindings("b239")
    document = json.loads(original)
    parent = document["hailo"]["hefs"][arch]["part1_build"]
    for key in path[:-1]:
        parent = parent[key]
    parent[path[-1]] = value
    with pytest.raises(ValueError, match="split-manifest staging"):
        validate_generic_split_manifest_staging(rewritten_original(document, args), staged, **args)


@pytest.mark.parametrize("key,value", [
    ("context_mode", "multi_context_used"),
    ("multi_context_used", True),
    ("calib_source", "/different/calibration"),
    ("allocation_time_s", 1.0),
    ("build_evidence", {"status": "unknown"}),
    ("unknown_reuse_metadata", True),
])
def test_current_reuse_does_not_hide_build_activity_or_replaced_diagnostics(key, value):
    original, staged, args = bindings("b239")
    document = json.loads(staged)
    document["hailo"]["hefs"]["hailo8"]["part1_build"][key] = value
    with pytest.raises(ValueError, match="split-manifest staging"):
        validate_generic_split_manifest_staging(original, rewritten(document, args), **args)


@pytest.mark.parametrize("role", [
    "part1_runtime", "source_part1_onnx", "source_part2_onnx", "part1_build_receipt",
    "engine", "build_part2_onnx", "engine_build_receipt",
])
def test_historical_compile_transition_requires_every_original_executed_artifact(role):
    original, staged, args = bindings("b239")
    args["row"]["generic_artifact_binding"]["artifacts"][role]["sha256"] = "f" * 64
    with pytest.raises(ValueError, match="original compiled/source artifact differs"):
        validate_generic_split_manifest_staging(original, staged, **args)


@pytest.mark.parametrize("raw", [None, b"", b"{}", "not bytes"])
def test_historical_measured_h8_build_requires_exact_actual_receipt_bytes(raw):
    original, staged, args = bindings("b239")
    args["part1_build_receipt_bytes"] = raw
    with pytest.raises(ValueError, match="receipt bytes"):
        validate_generic_split_manifest_staging(original, staged, **args)


@pytest.mark.parametrize("key,value", [
    ("schema", "unknown-receipt"),
    ("cache_key", "f" * 64),
    ("cache_payload", {}),
    ("source_onnx_sha256", "f" * 64),
    ("compiler_onnx_sha256", "f" * 64),
    ("compiler_onnx_filename", "another_graph.onnx"),
    ("hef_sha256", "f" * 64),
    ("hef_size_bytes", 1),
    ("hw_arch", "hailo10h"),
    ("net_name", "another_model_and_cut"),
    ("hailo_sdk_version", "unknown"),
    ("calibration_count", 1),
    ("preprocessing_contract", {}),
    ("publish_artifacts", False),
    ("diagnostic_only", True),
])
def test_hash_bound_receipt_still_must_couple_old_evidence_and_measured_hef(key, value):
    original, staged, args = bindings("b239")
    receipt = json.loads(args["part1_build_receipt_bytes"])
    receipt[key] = value
    bind_receipt(json.dumps(receipt, sort_keys=True).encode(), args)
    with pytest.raises(ValueError, match="actual measured receipt"):
        validate_generic_split_manifest_staging(original, staged, **args)


def test_self_consistent_old_compile_key_cannot_replace_the_actual_receipt_recipe():
    from onnx_splitpoint_tool.build_evidence import (
        boundary_endpoint_contract_sha256, canonical_build_key_from_hailo_v3_payload,
        canonical_sha256,
    )

    original, staged, args = bindings("b239")
    document = json.loads(original)
    build = document["hailo"]["hefs"]["hailo8"]["part1_build"]
    evidence = build["build_evidence"]
    payload = evidence["cache_payload_v3"]
    payload["extra_model_script"] = "different_compiler_recipe()"
    build["cache_key"] = evidence["cache_key_v3"] = canonical_sha256(payload)
    evidence["key"] = canonical_build_key_from_hailo_v3_payload(
        payload, builder_source_onnx_sha256=args["producer_model"]["part1_onnx_sha256"],
        full_source_onnx_sha256=args["full_model_sha256"],
        boundary_endpoint_contract_sha256=boundary_endpoint_contract_sha256(
            stage="part1", cache_payload=payload, split_manifest=document),
        expected_cache_key=build["cache_key"])
    evidence["key_sha256"] = canonical_sha256(evidence["key"])
    with pytest.raises(ValueError, match="actual measured receipt"):
        validate_generic_split_manifest_staging(rewritten_original(document, args), staged, **args)


@pytest.mark.parametrize("raw", [b"{", b"[]", b'{"schema":1,"schema":2}', b'{"bad":NaN}'])
def test_hash_bound_malformed_or_unknown_receipt_denies(raw):
    original, staged, args = bindings("b239")
    bind_receipt(raw, args)
    with pytest.raises(ValueError, match="split-manifest staging"):
        validate_generic_split_manifest_staging(original, staged, **args)
