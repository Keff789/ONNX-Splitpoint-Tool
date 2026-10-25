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


def historical_without_h8_diagnostics(case="b239", missing=("calib_source", "cache_key")):
    original, staged, args = bindings(case)
    document = json.loads(original)
    build = document["hailo"]["hefs"]["hailo8"]["part1_build"]
    for name in missing:
        del build[name]
    return document, staged, args


def positive_unused_h10_reuse():
    """Attach the observed small H10 reuse summary to a synthetic H8 fixture."""
    original, staged, args = bindings("b017")
    document = json.loads(original)
    historical_source = document["hailo"]["hefs"]["hailo10"]["part1_build"]["source_onnx_path"]
    document["hailo"]["hefs"]["hailo10"]["part1_build"] = {
        "build_receipt_path": str(Path(historical_source).parent / "hailo/hailo10/part1/hailo_hef_build_receipt.json"),
        "cache_hit": True,
        "calib_source": "validated_existing_artifact",
        "compiler_onnx_path": historical_source,
        "context_mode": "skipped",
        "elapsed_s": 1.2098214626312256,
        "mapping_failed": False,
        "multi_context_used": False,
        "ok": True,
        "single_context_failed": False,
        "single_context_used": False,
        "skipped": True,
        "source_onnx_path": historical_source,
        "timed_out": False,
        "watchdog_expired": False,
    }
    return rewritten_original(document, args), staged, args


def test_actual_b189_reuse_to_reuse_composition_preserves_science_and_omitted_states(monkeypatch):
    """Recorded manifest/plan metadata exercise every mapper clause, without payloads."""
    from onnx_splitpoint_tool import build_evidence

    assert build_evidence is not None
    original, staged, args = bindings("b189")
    old, current = json.loads(original), json.loads(staged)
    unchanged = deepcopy(args)

    def no_file_open(*_args, **_kwargs):
        pytest.fail("metadata composition must not open artifacts")

    with monkeypatch.context() as guard:
        guard.setattr("builtins.open", no_file_open)
        guard.setattr(Path, "open", no_file_open)
        proof = validate_generic_split_manifest_staging(original, staged, **args)

    assert args == unchanged
    assert proof["scientific_manifest_fields_unchanged"] is True
    assert len(args["boundary_inputs"]) == 3
    assert proof["historical_build_reuse"] == []  # Both original backends already reused.
    assert current["hailo"]["backend_terminal_states"] == [old["hailo"]["backend_terminal_states"][-1]]
    assert proof["omitted_unexecuted_backend_terminal_states"] == [
        row for row in old["hailo"]["backend_terminal_states"]
        if row["variant"] == "part1" and row["hw_arch"] == "hailo10h"]
    assert proof["unchanged_artifact_bindings"] == args["original_artifacts"]
    omitted, = [row for row in proof["omitted_unexecuted_backend_metadata"]
                if row["backend_metadata"] == "hailo10"]
    assert omitted["historical_observation"] == old["hailo"]["hefs"]["hailo10"]
    assert omitted["validated_historical_build"]["historical_reuse_observation"] is True
    assert omitted["validated_historical_build"]["executed_artifact_identity_proved"] is False
    assert "actual_receipt_binding" not in omitted["validated_historical_build"]
    assert all(omitted[key] is False for key in (
        "current_compile_claimed", "current_reuse_claimed", "current_quality_or_hardware_pass_claimed"))


@pytest.mark.parametrize("fault", [
    "unavailable", "nonterminal", "failed", "unknown_field", "duplicate",
    "wrong_backend", "part2", "changed_selected", "extra_selected",
])
def test_b189_complete_relation_rejects_unproved_terminal_projection(fault):
    original, staged, args = bindings("b189")
    old, current = json.loads(original), json.loads(staged)
    state = old["hailo"]["backend_terminal_states"][1]  # Omitted H10 part1.
    if fault == "unavailable":
        state["available"] = False
    elif fault == "nonterminal":
        state["terminal"] = False
    elif fault == "failed":
        state["status"] = "failed"
    elif fault == "unknown_field":
        state["unknown"] = False
    elif fault == "duplicate":
        old["hailo"]["backend_terminal_states"].append(deepcopy(state))
    elif fault == "wrong_backend":
        state["hw_arch"] = "deepx"
    elif fault == "part2":
        state["variant"] = "part2"
    elif fault == "changed_selected":
        current["hailo"]["backend_terminal_states"][0]["reason"] = "other"
    else:
        current["hailo"]["backend_terminal_states"].append(deepcopy(state))
    with pytest.raises(ValueError, match="backend terminal states"):
        validate_generic_split_manifest_staging(
            rewritten_original(old, args), rewritten(current, args), **args)


@pytest.mark.parametrize("flag", ["mapping_failed", "single_context_used", "timed_out"])
def test_b189_terminal_projection_requires_positive_selected_h8_reuse(flag):
    original, staged, args = bindings("b189")
    old, current = json.loads(original), json.loads(staged)
    for document in (old, current):
        document["hailo"]["hefs"]["hailo8"]["part1_build"][flag] = True
    with pytest.raises(ValueError, match="backend terminal states"):
        validate_generic_split_manifest_staging(
            rewritten_original(old, args), rewritten(current, args), **args)


@pytest.mark.parametrize("fault", ["artifact", "receipt", "full_model", "boundary", "input", "manifest"])
def test_b189_complete_relation_preserves_all_selected_scientific_guards(fault):
    original, staged, args = bindings("b189")
    if fault in {"artifact", "receipt"}:
        role = "part1_runtime" if fault == "artifact" else "part1_build_receipt"
        args["original_artifacts"][role]["sha256"] = "f" * 64
    elif fault == "full_model":
        args["producer_model"]["source_onnx_sha256"] = "f" * 64
    elif fault == "boundary":
        args["boundary_inputs"][0]["dtype"] = "float16"
    elif fault == "input":
        args["row"]["functional_reference"]["input_sha256"] = "f" * 64
    else:
        current = json.loads(staged)
        current["orig_outputs"] = ["changed"]
        staged = rewritten(current, args)
    with pytest.raises(ValueError, match="split-manifest staging"):
        validate_generic_split_manifest_staging(original, staged, **args)


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


@pytest.mark.parametrize("case,missing", [
    ("b239", ("calib_source",)),
    ("b239", ("cache_key",)),
    ("b239", ("calib_source", "cache_key")),
    ("b017", ("calib_source", "cache_key")),
])
def test_historical_h8_absent_diagnostics_preserve_exact_observation_and_receipt(case, missing):
    document, staged, args = historical_without_h8_diagnostics(case, missing)
    original = rewritten_original(document, args)
    unchanged_args = deepcopy(args)

    proof = validate_generic_split_manifest_staging(original, staged, **args)

    assert args == unchanged_args
    assert proof["original_manifest"] == args["original_source"]
    assert proof["scientific_manifest_fields_unchanged"] is True
    measured, = [item for item in proof["historical_build_reuse"]
                 if item["backend_metadata"] == "hailo8"]
    build = document["hailo"]["hefs"]["hailo8"]["part1_build"]
    assert measured["historical_observation"] == build
    assert not set(missing) & measured["historical_observation"].keys()
    receipt, key = measured["actual_receipt_binding"], build["build_evidence"]["key"]
    assert receipt["sha256"] == args["original_artifacts"]["part1_build_receipt"]["sha256"]
    assert receipt["cache_key"] == build["build_evidence"]["cache_key_v3"]
    assert receipt["hef_sha256"] == args["original_artifacts"]["part1_runtime"]["sha256"]
    assert receipt["source_onnx_sha256"] == measured["source_onnx_sha256"] == key["builder_source_onnx_sha256"]
    assert receipt["compiler_onnx_sha256"] == measured["compiler_onnx_sha256"] == key["compiler_onnx_sha256"]
    assert measured["compiler_source_byte_equality_claimed"] is False
    if case == "b017":
        assert receipt["source_onnx_sha256"] != receipt["compiler_onnx_sha256"]


@pytest.mark.parametrize("name,value", [
    ("calib_source", None),
    ("calib_source", "/different/calibration"),
    ("cache_key", None),
    ("cache_key", "f" * 64),
])
def test_historical_h8_absent_diagnostics_cannot_hide_present_contradictions(name, value):
    document, staged, args = historical_without_h8_diagnostics()
    document["hailo"]["hefs"]["hailo8"]["part1_build"][name] = value
    with pytest.raises(ValueError, match="split-manifest staging"):
        validate_generic_split_manifest_staging(rewritten_original(document, args), staged, **args)


@pytest.mark.parametrize("name", ["calib_source", "cache_key"])
def test_historical_h8_absent_diagnostics_do_not_relax_unmeasured_h10(name):
    document, staged, args = historical_without_h8_diagnostics()
    del document["hailo"]["hefs"]["hailo10"]["part1_build"][name]
    with pytest.raises(ValueError, match="split-manifest staging"):
        validate_generic_split_manifest_staging(rewritten_original(document, args), staged, **args)


@pytest.mark.parametrize("path,value", [
    (("build_evidence", "cache_key_v3"), "f" * 64),
    (("build_evidence", "context", "split_manifest", "hailo", "config", "calib_dir"), "/different/calibration"),
    (("build_evidence", "cache_payload_v3", "requested_calibration_count"), 501),
    (("mapping_failed",), True),
])
def test_historical_h8_absent_diagnostics_still_require_bound_positive_compile_evidence(path, value):
    document, staged, args = historical_without_h8_diagnostics()
    parent = document["hailo"]["hefs"]["hailo8"]["part1_build"]
    for name in path[:-1]:
        parent = parent[name]
    parent[path[-1]] = value
    with pytest.raises(ValueError, match="split-manifest staging"):
        validate_generic_split_manifest_staging(rewritten_original(document, args), staged, **args)


@pytest.mark.parametrize("raw", [None, b"modified receipt bytes"])
def test_historical_h8_absent_diagnostics_still_require_exact_receipt_bytes(raw):
    document, staged, args = historical_without_h8_diagnostics()
    args["part1_build_receipt_bytes"] = raw
    with pytest.raises(ValueError, match="receipt bytes"):
        validate_generic_split_manifest_staging(rewritten_original(document, args), staged, **args)


@pytest.mark.parametrize("name,value", [
    ("calibration_identity", "manifest:" + "f" * 64),
    ("prepared_calibration_identity_sha256", "f" * 64),
    ("preprocessing_contract", {}),
    ("hef_sha256", "f" * 64),
])
def test_historical_h8_absent_diagnostics_still_require_measured_receipt_identity(name, value):
    document, staged, args = historical_without_h8_diagnostics()
    receipt = json.loads(args["part1_build_receipt_bytes"])
    receipt[name] = value
    bind_receipt(json.dumps(receipt, sort_keys=True).encode(), args)
    with pytest.raises(ValueError, match="actual measured receipt"):
        validate_generic_split_manifest_staging(rewritten_original(document, args), staged, **args)


def test_historical_h8_absent_diagnostics_cannot_replace_receipt_calibration_with_rekeyed_evidence():
    from onnx_splitpoint_tool.build_evidence import (
        boundary_endpoint_contract_sha256, canonical_build_key_from_hailo_v3_payload,
        canonical_sha256,
    )

    document, staged, args = historical_without_h8_diagnostics()
    evidence = document["hailo"]["hefs"]["hailo8"]["part1_build"]["build_evidence"]
    payload = evidence["cache_payload_v3"]
    payload["calibration_identity"] = "manifest:" + "f" * 64
    payload["prepared_calibration_identity_sha256"] = canonical_sha256({
        "calibration_identity": payload["calibration_identity"],
        "preprocessing_contract_sha256": payload["preprocessing_contract_sha256"],
    })
    evidence["cache_key_v3"] = canonical_sha256(payload)
    evidence["key"] = canonical_build_key_from_hailo_v3_payload(
        payload, builder_source_onnx_sha256=args["producer_model"]["part1_onnx_sha256"],
        full_source_onnx_sha256=args["full_model_sha256"],
        boundary_endpoint_contract_sha256=boundary_endpoint_contract_sha256(
            stage="part1", cache_payload=payload, split_manifest=document),
        expected_cache_key=evidence["cache_key_v3"])
    evidence["key_sha256"] = canonical_sha256(evidence["key"])
    with pytest.raises(ValueError, match="actual measured receipt"):
        validate_generic_split_manifest_staging(rewritten_original(document, args), staged, **args)


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


@pytest.mark.parametrize("case", ["b182", "b017"])
def test_actual_compact_h8_staging_keeps_measured_identity_and_omitted_history(case):
    original, staged, args = bindings(case)
    old, current = json.loads(original), json.loads(staged)
    unchanged = deepcopy(args)

    proof = validate_generic_split_manifest_staging(original, staged, **args)

    assert args == unchanged
    assert set(current["hailo"]["hefs"]) == {"hailo8"}
    assert not {"deepx", "deepx_part1_dxnn", "deepx_part1_output_contract"} & current.keys()
    assert proof["original_manifest"] == args["original_source"]
    assert proof["staged_manifest"] == args["staged_source"]
    assert proof["scientific_manifest_fields_unchanged"] is True
    assert proof["unchanged_artifact_bindings"] == {
        role: {key: value[key] for key in ("sha256", "size_bytes")}
        for role, value in args["original_artifacts"].items()
    }
    measured, = proof["historical_build_reuse"]
    assert measured["backend_metadata"] == "hailo8"
    assert measured["historical_observation"] == old["hailo"]["hefs"]["hailo8"]["part1_build"]
    assert measured["executed_artifact_identity_proved"] is True
    assert measured["compiler_source_byte_equality_claimed"] is False
    receipt = measured["actual_receipt_binding"]
    assert receipt["sha256"] == hashlib.sha256(args["part1_build_receipt_bytes"]).hexdigest()
    assert receipt["hef_sha256"] == args["original_artifacts"]["part1_runtime"]["sha256"]
    omitted = {item["backend_metadata"]: item for item in proof["omitted_unexecuted_backend_metadata"]}
    assert set(omitted) == {"deepx", "hailo10"}
    assert omitted["deepx"]["historical_observation"] == old["deepx"]
    if case == "b017":
        dx = omitted["deepx"]["historical_observation"]["part1_dxnn"]
        assert dx["build_status"] == "ready_built"
        assert dx["cache_lookup"]["outcome"] == "MISS"
        assert dx["cache_lookup"]["reason"] == "model_missing"
        assert dx["cache_lookup"]["artifact"] == ""
    assert omitted["hailo10"]["historical_observation"] == old["hailo"]["hefs"]["hailo10"]
    other_build = omitted["hailo10"]["validated_historical_build"]
    assert other_build["historical_observation"] == old["hailo"]["hefs"]["hailo10"]["part1_build"]
    assert other_build["executed_artifact_identity_proved"] is False
    assert "actual_receipt_binding" not in other_build
    for item in omitted.values():
        assert item["current_compile_claimed"] is False
        assert item["current_reuse_claimed"] is False
        assert item["current_quality_or_hardware_pass_claimed"] is False


def test_omitted_positive_h10_reuse_is_exact_metadata_without_execution_or_reuse_claims(monkeypatch):
    # Load the pure evidence helper before forbidding all file opens in validation.
    from onnx_splitpoint_tool import build_evidence

    assert build_evidence is not None
    original, staged, args = positive_unused_h10_reuse()
    old, current = json.loads(original), json.loads(staged)
    unchanged = deepcopy(args)
    original_bytes = original[:]

    def no_file_open(*_args, **_kwargs):
        pytest.fail("the staging relation must not open historical artifacts")

    with monkeypatch.context() as guard:
        guard.setattr("builtins.open", no_file_open)
        guard.setattr(Path, "open", no_file_open)
        proof = validate_generic_split_manifest_staging(original, staged, **args)

    assert original == original_bytes
    assert json.loads(original) == old
    assert args == unchanged
    assert "hailo10" not in current["hailo"]["hefs"]
    assert proof["original_manifest"] == args["original_source"]
    assert proof["staged_manifest"] == args["staged_source"]
    assert proof["original_boundary_metadata_artifact"] == args["row"]["generic_artifact_binding"]["artifacts"]["boundary_metadata"]
    assert proof["scientific_manifest_fields_unchanged"] is True
    assert proof["unchanged_artifact_bindings"] == {
        role: {key: value[key] for key in ("sha256", "size_bytes")}
        for role, value in args["original_artifacts"].items()
    }
    measured, = proof["historical_build_reuse"]
    assert measured["backend_metadata"] == "hailo8"
    assert measured["executed_artifact_identity_proved"] is True
    assert measured["actual_receipt_binding"]["sha256"] == args["original_artifacts"]["part1_build_receipt"]["sha256"]
    assert measured["actual_receipt_binding"]["hef_sha256"] == args["original_artifacts"]["part1_runtime"]["sha256"]
    omitted, = [item for item in proof["omitted_unexecuted_backend_metadata"]
                if item["backend_metadata"] == "hailo10"]
    assert omitted["historical_observation"] == old["hailo"]["hefs"]["hailo10"]
    reuse = omitted["validated_historical_build"]
    assert reuse["backend_metadata"] == "hailo10"
    assert reuse["historical_observation"] == old["hailo"]["hefs"]["hailo10"]["part1_build"]
    assert reuse["historical_reuse_observation"] is True
    assert reuse["executed_artifact_identity_proved"] is False
    assert reuse["compiler_source_byte_equality_claimed"] is False
    assert "actual_receipt_binding" not in reuse
    assert omitted["current_compile_claimed"] is False
    assert omitted["current_reuse_claimed"] is False
    assert omitted["current_quality_or_hardware_pass_claimed"] is False
    removed, = [item for item in proof["allowed_differences"]
                if item["pointer"] == "/hailo/hefs/hailo10"]
    assert removed["original_present"] is True and removed["staged_present"] is False


@pytest.mark.parametrize("key,value", [
    ("ok", False), ("ok", 1),
    ("cache_hit", False), ("cache_hit", 1),
    ("skipped", False), ("skipped", 1),
    ("calib_source", "unknown_existing_artifact"),
    ("context_mode", "single_context_used"),
    ("mapping_failed", True), ("mapping_failed", 0),
    ("multi_context_used", True), ("single_context_failed", True),
    ("single_context_used", True), ("timed_out", True),
    ("watchdog_expired", True),
    ("elapsed_s", -1), ("elapsed_s", True), ("elapsed_s", "1.2"),
    ("unknown_build_field", True), ("negative_evidence_hit", True),
    ("negative_evidence_hit", False), ("cache_key", "f" * 64),
    ("build_evidence", {}),
])
def test_omitted_positive_h10_reuse_cannot_hide_negative_unknown_or_extra_metadata(key, value):
    original, staged, args = positive_unused_h10_reuse()
    document = json.loads(original)
    document["hailo"]["hefs"]["hailo10"]["part1_build"][key] = value
    with pytest.raises(ValueError, match="split-manifest staging"):
        validate_generic_split_manifest_staging(rewritten_original(document, args), staged, **args)


@pytest.mark.parametrize("key", [
    "build_receipt_path", "cache_hit", "calib_source", "compiler_onnx_path",
    "context_mode", "elapsed_s", "mapping_failed", "multi_context_used", "ok",
    "single_context_failed", "single_context_used", "skipped", "source_onnx_path",
    "timed_out", "watchdog_expired",
])
def test_omitted_positive_h10_reuse_requires_the_complete_observed_summary(key):
    original, staged, args = positive_unused_h10_reuse()
    document = json.loads(original)
    del document["hailo"]["hefs"]["hailo10"]["part1_build"][key]
    with pytest.raises(ValueError, match="split-manifest staging"):
        validate_generic_split_manifest_staging(rewritten_original(document, args), staged, **args)


@pytest.mark.parametrize("fault", [
    "wrong_hef", "extra_backend_metadata", "relative_source", "wrong_source_model",
    "wrong_source_cut", "divergent_compiler", "wrong_receipt_backend",
    "wrong_receipt_name", "divergent_receipt_root", "noncanonical_source",
])
def test_omitted_positive_h10_reuse_requires_coherent_original_model_cut_paths(fault):
    original, staged, args = positive_unused_h10_reuse()
    document = json.loads(original)
    hef = document["hailo"]["hefs"]["hailo10"]
    build = hef["part1_build"]
    source = Path(build["source_onnx_path"])
    if fault == "wrong_hef":
        hef["part1"] = "hailo/hailo8/part1/compiled.hef"
    elif fault == "extra_backend_metadata":
        hef["unknown_backend_field"] = True
    elif fault == "relative_source":
        build["source_onnx_path"] = build["compiler_onnx_path"] = source.name
    elif fault == "wrong_source_model":
        build["source_onnx_path"] = build["compiler_onnx_path"] = str(source.with_name("another_model_part1_b17.onnx"))
    elif fault == "wrong_source_cut":
        build["source_onnx_path"] = build["compiler_onnx_path"] = str(source.with_name("yolov7_paper_part1_b18.onnx"))
    elif fault == "divergent_compiler":
        build["compiler_onnx_path"] = str(source.parent / "hailo/hailo10/part1/yolov7_paper_part1_b17_hailo_fixed.onnx")
    elif fault == "wrong_receipt_backend":
        build["build_receipt_path"] = build["build_receipt_path"].replace("/hailo10/", "/hailo8/")
    elif fault == "wrong_receipt_name":
        build["build_receipt_path"] = str(Path(build["build_receipt_path"]).with_name("another_receipt.json"))
    elif fault == "divergent_receipt_root":
        build["build_receipt_path"] = "/foreign/b017/hailo/hailo10/part1/hailo_hef_build_receipt.json"
    else:
        build["source_onnx_path"] = build["compiler_onnx_path"] = str(source.parent) + "/../b017/" + source.name
    with pytest.raises(ValueError, match="split-manifest staging"):
        validate_generic_split_manifest_staging(rewritten_original(document, args), staged, **args)


@pytest.mark.parametrize("key,value", [
    ("full", True), ("full", 0), ("part1", False), ("part1", 1),
    ("part2", True), ("unknown", False),
])
def test_omitted_positive_h10_reuse_requires_exact_source_only_recipe_flags(key, value):
    original, staged, args = positive_unused_h10_reuse()
    document = json.loads(original)
    document["hailo"]["config"]["build"][key] = value
    with pytest.raises(ValueError, match="split-manifest staging"):
        validate_generic_split_manifest_staging(rewritten_original(document, args), staged, **args)


@pytest.mark.parametrize("role", [
    "part1_runtime", "source_part1_onnx", "source_part2_onnx", "part1_build_receipt",
    "engine", "build_part2_onnx", "engine_build_receipt",
])
def test_omitted_positive_h10_reuse_keeps_all_seven_selected_h8_artifact_bindings(role):
    original, staged, args = positive_unused_h10_reuse()
    args["row"]["generic_artifact_binding"]["artifacts"][role]["sha256"] = "f" * 64
    with pytest.raises(ValueError, match="split-manifest staging"):
        validate_generic_split_manifest_staging(original, staged, **args)


@pytest.mark.parametrize("path,value", [
    (("part1",), "another_part1.onnx"),
    (("part2",), "another_part2.onnx"),
    (("full_model",), "../models/another_model.onnx"),
    (("boundary",), 18),
    (("boundary_index",), 18),
    (("part1_cut_names",), ["unbound_tensor"]),
    (("io", "part2_external_inputs"), ["unbound_input"]),
    (("hailo", "hefs", "hailo8", "part1"), "different.hef"),
    (("hailo", "hefs", "hailo8", "part1_build", "cache_hit"), False),
])
def test_omitted_positive_h10_reuse_cannot_relax_selected_h8_manifest_science(path, value):
    original, staged, args = positive_unused_h10_reuse()
    document = json.loads(staged)
    parent = document
    for key in path[:-1]:
        parent = parent[key]
    parent[path[-1]] = value
    with pytest.raises(ValueError, match="split-manifest staging"):
        validate_generic_split_manifest_staging(original, rewritten(document, args), **args)


@pytest.mark.parametrize("fault", [
    lambda a: a["producer_model"].update(part1_onnx_sha256="f" * 64),
    lambda a: a["producer_model"].update(part2_onnx_sha256="f" * 64),
    lambda a: a.update(full_model_sha256="f" * 64),
    lambda a: a["boundary_inputs"][0].update(shape=[1, 1]),
    lambda a: a["boundary_inputs"][0].update(dtype="float16"),
    lambda a: a["row"]["input_binding"]["prepared_input"].update(sha256="f" * 64),
    lambda a: a["row"]["input_binding"].update(runtime_input_format="uint8"),
    lambda a: a["row"]["identity"].update(comparison_backend="hailo10h"),
    lambda a: a.update(part1_build_receipt_bytes=b"modified receipt bytes"),
])
def test_omitted_positive_h10_reuse_cannot_relax_model_boundary_input_or_h8_receipt(fault):
    original, staged, args = positive_unused_h10_reuse()
    fault(args)
    with pytest.raises(ValueError, match="split-manifest staging"):
        validate_generic_split_manifest_staging(original, staged, **args)


@pytest.mark.parametrize("path", [
    ("hailo", "hefs", "hailo8"),
    ("hailo", "hefs", "hailo8", "part1"),
    ("hailo", "hefs", "hailo8", "part1_build"),
])
def test_compact_staging_cannot_omit_the_measured_h8_record(path):
    original, staged, args = bindings("b182")
    document = json.loads(staged)
    parent = document
    for key in path[:-1]:
        parent = parent[key]
    del parent[path[-1]]
    with pytest.raises(ValueError, match="split-manifest staging"):
        validate_generic_split_manifest_staging(original, rewritten(document, args), **args)


@pytest.mark.parametrize("key,value", [
    ("unknown_current_build_field", True),
    ("cache_hit", 1),
    ("ok", False),
    ("mapping_failed", True),
    ("multi_context_used", True),
    ("calib_source", "unknown_existing_artifact"),
    ("context_mode", "multi_context_used"),
    ("elapsed_s", -1),
    ("source_onnx_path", "/foreign/b182/yolo26s_part1_b182.onnx"),
    ("compiler_onnx_path", "/foreign/b182/yolo26s_part1_b182.onnx"),
    ("build_receipt_path", "/foreign/b182/hailo/hailo8/part1/hailo_hef_build_receipt.json"),
])
def test_compact_measured_h8_summary_requires_ordinary_bound_reuse(key, value):
    original, staged, args = bindings("b182")
    document = json.loads(staged)
    document["hailo"]["hefs"]["hailo8"]["part1_build"][key] = value
    with pytest.raises(ValueError, match="split-manifest staging"):
        validate_generic_split_manifest_staging(original, rewritten(document, args), **args)


@pytest.mark.parametrize("raw", [None, b"{}", b"modified receipt bytes"])
def test_compact_measured_h8_still_requires_exact_receipt_bytes(raw):
    original, staged, args = bindings("b182")
    args["part1_build_receipt_bytes"] = raw
    with pytest.raises(ValueError, match="split-manifest staging"):
        validate_generic_split_manifest_staging(original, staged, **args)


@pytest.mark.parametrize("key,value", [
    ("cache_key", "f" * 64),
    ("compiler_onnx_filename", "different_compiler_graph.onnx"),
    ("hef_sha256", "f" * 64),
])
def test_compact_hash_bound_receipt_cannot_replace_historical_artifact_identity(key, value):
    original, staged, args = bindings("b182")
    receipt = json.loads(args["part1_build_receipt_bytes"])
    receipt[key] = value
    bind_receipt(json.dumps(receipt, sort_keys=True).encode(), args)
    with pytest.raises(ValueError, match="split-manifest staging"):
        validate_generic_split_manifest_staging(original, staged, **args)


@pytest.mark.parametrize("role", [
    "part1_runtime", "source_part1_onnx", "source_part2_onnx", "part1_build_receipt",
    "engine", "build_part2_onnx", "engine_build_receipt",
])
@pytest.mark.parametrize("case", ["b182", "b017"])
def test_compact_staging_still_requires_all_seven_original_artifacts(role, case):
    original, staged, args = bindings(case)
    args["row"]["generic_artifact_binding"]["artifacts"][role]["sha256"] = "f" * 64
    with pytest.raises(ValueError, match="split-manifest staging"):
        validate_generic_split_manifest_staging(original, staged, **args)


@pytest.mark.parametrize("fault", [
    lambda a: a["producer_model"].update(part2_onnx_sha256="f" * 64),
    lambda a: a.update(full_model_sha256="f" * 64),
    lambda a: a["boundary_inputs"][0].update(shape=[1, 1]),
    lambda a: a["row"]["input_binding"]["prepared_input"].update(sha256="f" * 64),
    lambda a: a["row"]["input_binding"].update(runtime_input_format="uint8"),
    lambda a: a["row"]["identity"].update(comparison_backend="hailo10h"),
])
@pytest.mark.parametrize("case", ["b182", "b017"])
def test_compact_staging_still_binds_model_boundary_input_and_execution_scope(fault, case):
    original, staged, args = bindings(case)
    fault(args)
    with pytest.raises(ValueError, match="split-manifest staging"):
        validate_generic_split_manifest_staging(original, staged, **args)


@pytest.mark.parametrize("path,value", [
    (("ok",), False),
    (("status",), "failed"),
    (("build_status",), "ready_reused"),
    (("build_status",), "building"),
    (("source_onnx_sha256",), "f" * 64),
    (("dxnn_sha256",), "not-a-sha256"),
    (("dxnn_size_bytes",), 0),
    (("output_contract_sha256",), "not-a-sha256"),
    (("output_names",), ["unbound_tensor"]),
    (("cache_contract", "input_shape"), [1, 3, 320, 320]),
    (("cache_contract", "case_id"), "b018"),
    (("cache_contract", "calibration_count"), 501),
    (("cache_contract", "compiler_identity"), ""),
    (("cache_contract", "preprocessing_contract"), "unproved-preprocessing"),
    (("cache_lookup", "outcome"), "HIT"),
    (("cache_lookup", "reason"), "compatible"),
    (("cache_lookup", "artifact"), "/fabricated/current/model.dxnn"),
    (("cache_lookup", "artifact"), None),
    (("cache_lookup", "identity"), "deepx_m1_part1_b017_v2_" + "f" * 32),
    (("cache_lookup", "case_id"), "b018"),
    (("cache_lookup", "role"), "full"),
    (("unknown_positive_claim",), True),
])
def test_omitted_historical_deepx_build_requires_original_positive_bound_observation(path, value):
    original, staged, args = bindings("b017")
    document = json.loads(original)
    parent = document["deepx"]["part1_dxnn"]
    for key in path[:-1]:
        parent = parent[key]
    parent[path[-1]] = value
    with pytest.raises(ValueError, match="split-manifest staging"):
        validate_generic_split_manifest_staging(rewritten_original(document, args), staged, **args)


def test_omitted_historical_deepx_build_cannot_claim_current_hit_or_active_deepx():
    original, staged, args = bindings("b017")
    document = json.loads(original)
    dx = document["deepx"]["part1_dxnn"]
    dx["cache_lookup"].update(outcome="HIT", reason="compatible",
                            artifact=f"/fabricated/cache/{dx['cache_key']}/model.dxnn")
    with pytest.raises(ValueError, match="omitted DeepX cache observation"):
        validate_generic_split_manifest_staging(rewritten_original(document, args), staged, **args)

    original, staged, args = bindings("b017")
    args["row"]["identity"].update(comparison_backend="deepx", direction="deepx_to_trt")
    with pytest.raises(ValueError, match="split-manifest staging"):
        validate_generic_split_manifest_staging(original, staged, **args)


@pytest.mark.parametrize("field", [
    "full_end_node_names", "full_endpoint_mode", "full_output_contract", "unknown_full_metadata",
])
def test_compact_staging_cannot_hide_partial_or_unknown_full_endpoint_metadata(field):
    original, staged, args = bindings("b017")
    document = json.loads(original)
    document["hailo"][field] = "unproved"
    with pytest.raises(ValueError, match="unknown compact backend omission scope"):
        validate_generic_split_manifest_staging(rewritten_original(document, args), staged, **args)


@pytest.mark.parametrize("key,value", [("full", 1), ("part1", False), ("part2", True), ("unknown", True)])
def test_compact_historical_build_scope_requires_exact_boolean_source_only_parts(key, value):
    original, staged, args = bindings("b017")
    document = json.loads(original)
    document["hailo"]["config"]["build"][key] = value
    with pytest.raises(ValueError, match="known unexecuted Full scope"):
        validate_generic_split_manifest_staging(rewritten_original(document, args), staged, **args)


@pytest.mark.parametrize("path,value", [
    (("deepx", "unknown_backend_field"), True),
    (("deepx", "part1_dxnn", "ok"), False),
    (("deepx", "part1_dxnn", "status"), "failed"),
    (("deepx", "part1_dxnn", "cache_lookup", "outcome"), "MISS"),
    (("hailo", "hefs", "hailo10", "unknown_backend_field"), True),
    (("hailo", "hefs", "hailo10", "part1_build", "unknown_build_field"), True),
    (("hailo", "hefs", "hailo10", "part1_build", "ok"), False),
    (("hailo", "hefs", "hailo10", "part1_build", "negative_evidence_hit"), True),
    (("hailo", "case_variant_availability", "hailo10h", "part1_failed"), True),
])
def test_compact_staging_cannot_hide_unknown_or_negative_original_backend_metadata(path, value):
    original, staged, args = bindings("b182")
    document = json.loads(original)
    parent = document
    for key in path[:-1]:
        parent = parent[key]
    parent[path[-1]] = value
    with pytest.raises(ValueError, match="split-manifest staging"):
        validate_generic_split_manifest_staging(rewritten_original(document, args), staged, **args)


@pytest.mark.parametrize("context_count", [None, 0, True, 2])
def test_compact_historical_single_context_requires_exact_positive_diagnostic(context_count):
    original, staged, args = bindings("b182")
    document = json.loads(original)
    document["hailo"]["hefs"]["hailo8"]["part1_build"]["context_count"] = context_count
    with pytest.raises(ValueError, match="historical compiler context"):
        validate_generic_split_manifest_staging(rewritten_original(document, args), staged, **args)


@pytest.mark.parametrize("fault", [
    "deepx_alias_only", "deepx_block_only", "h10_availability_only",
    "h10_part1_only", "h10_current_reuse", "h8_full_promotion",
])
@pytest.mark.parametrize("case", ["b182", "b017"])
def test_compact_staging_requires_coherent_omission_and_cannot_promote_unmeasured_backends(fault, case):
    original, staged, args = bindings(case)
    old, document = json.loads(original), json.loads(staged)
    if fault == "deepx_alias_only":
        document["deepx_part1_dxnn"] = old["deepx_part1_dxnn"]
    elif fault == "deepx_block_only":
        document["deepx"] = deepcopy(old["deepx"])
    elif fault == "h10_availability_only":
        document["hailo"]["case_variant_availability"]["hailo10h"] = deepcopy(
            old["hailo"]["case_variant_availability"]["hailo10h"])
    elif fault == "h10_part1_only":
        document["hailo"]["hefs"]["hailo10"] = {
            "part1": old["hailo"]["hefs"]["hailo10"]["part1"]}
    elif fault == "h10_current_reuse":
        document["hailo"]["hefs"]["hailo10"] = {
            "part1": old["hailo"]["hefs"]["hailo10"]["part1"],
            "part1_build": deepcopy(document["hailo"]["hefs"]["hailo8"]["part1_build"]),
        }
    else:
        document["hailo"]["hefs"]["hailo8"]["full"] = old["hailo"]["hefs"]["hailo8"]["full"]
        document["hailo"]["case_variant_availability"]["hailo8"]["full"] = True
    with pytest.raises(ValueError, match="split-manifest staging"):
        validate_generic_split_manifest_staging(original, rewritten(document, args), **args)


def test_actual_b071_staging_binds_explicit_boundary_mirror_and_unchanged_b71_graphs():
    original, staged, args = bindings("b071")
    old, current = json.loads(original), json.loads(staged)
    unchanged = deepcopy(args)
    boundary = args["row"]["generic_artifact_binding"]["artifacts"]["boundary_metadata"]

    proof = validate_generic_split_manifest_staging(original, staged, **args)

    assert args == unchanged
    assert old["boundary"] == current["boundary"] == 71
    assert old["part1"] == current["part1"] == "yolo26s_part1_b71.onnx"
    assert old["part2"] == current["part2"] == "yolo26s_part2_b71.onnx"
    assert proof["measured_identity"]["case_id"] == "b071"
    assert proof["original_manifest"] == args["original_source"]
    assert proof["staged_manifest"] == args["staged_source"]
    assert proof["original_boundary_metadata_artifact"] == boundary
    assert boundary["local_path"] != args["original_source"]["path"]
    assert boundary["source_path_alias"] == args["original_source"]["path"]
    assert boundary["stage_local"] is True
    assert {key: boundary[key] for key in ("sha256", "size_bytes")} == {
        key: args["original_source"][key] for key in ("sha256", "size_bytes")}
    assert proof["unchanged_artifact_bindings"] == {
        role: {key: value[key] for key in ("sha256", "size_bytes")}
        for role, value in args["original_artifacts"].items()
    }
    measured, = proof["historical_build_reuse"]
    assert measured["historical_observation"] == old["hailo"]["hefs"]["hailo8"]["part1_build"]
    assert measured["actual_receipt_binding"]["sha256"] == hashlib.sha256(
        args["part1_build_receipt_bytes"]).hexdigest()
    assert json.loads(args["part1_build_receipt_bytes"])["net_name"] == "yolo26s_part1_b71"
    assert proof["boundary_contract"] == args["row"]["boundary_observation"]
    assert proof["scientific_manifest_fields_unchanged"] is True


@pytest.mark.parametrize("key,value", [
    ("source_path_alias", "/foreign/b071/split_manifest.json"),
    ("source_path_alias", None),
    ("stage_local", False),
    ("stage_local", 1),
    ("local_path", None),
    ("local_path", "bound_assets/b071/boundary_metadata/split_manifest.json"),
    ("local_path", "/bound_assets/b071/../boundary_metadata/split_manifest.json"),
    ("local_path", "/bound_assets/b071/boundary_metadata/another_manifest.json"),
    ("sha256", "f" * 64),
    ("size_bytes", 59943),
])
def test_b071_boundary_mirror_requires_exact_explicit_original_binding(key, value):
    original, staged, args = bindings("b071")
    boundary = args["row"]["generic_artifact_binding"]["artifacts"]["boundary_metadata"]
    if value is None:
        boundary.pop(key)
    else:
        boundary[key] = value
    with pytest.raises(ValueError, match="split-manifest staging"):
        validate_generic_split_manifest_staging(original, staged, **args)


@pytest.mark.parametrize("which", ["original", "staged"])
def test_b071_mirror_relation_cannot_accept_changed_manifest_bytes(which):
    original, staged, args = bindings("b071")
    if which == "original":
        original += b"\n"
    else:
        staged += b"\n"
    with pytest.raises(ValueError, match="bytes differ from descriptor"):
        validate_generic_split_manifest_staging(original, staged, **args)


@pytest.mark.parametrize("which,part", [
    ("original", "part1"), ("original", "part2"),
    ("staged", "part1"), ("staged", "part2"),
])
def test_b071_display_padding_cannot_rewrite_either_actual_graph_name(which, part):
    original, staged, args = bindings("b071")
    document = json.loads(original if which == "original" else staged)
    document[part] = f"yolo26s_{part}_b071.onnx"
    if which == "original":
        original = rewritten_original(document, args)
    else:
        staged = rewritten(document, args)
    with pytest.raises(ValueError, match="split-manifest staging"):
        validate_generic_split_manifest_staging(original, staged, **args)


@pytest.mark.parametrize("role", [
    "part1_runtime", "source_part1_onnx", "source_part2_onnx", "part1_build_receipt",
    "engine", "build_part2_onnx", "engine_build_receipt",
])
def test_b071_boundary_mirror_keeps_all_seven_measured_artifact_identities(role):
    original, staged, args = bindings("b071")
    args["row"]["generic_artifact_binding"]["artifacts"][role]["sha256"] = "f" * 64
    with pytest.raises(ValueError, match="split-manifest staging"):
        validate_generic_split_manifest_staging(original, staged, **args)


@pytest.mark.parametrize("fault", ["cut", "boundary_dtype", "prepared_input", "receipt_net_name"])
def test_b071_name_and_mirror_relation_still_requires_actual_cut_input_and_receipt(fault):
    original, staged, args = bindings("b071")
    if fault == "cut":
        document = json.loads(staged)
        document["boundary"] = document["boundary_index"] = 72
        staged = rewritten(document, args)
    elif fault == "boundary_dtype":
        args["boundary_inputs"][0]["dtype"] = "float16"
    elif fault == "prepared_input":
        args["row"]["input_binding"]["prepared_input"]["sha256"] = "f" * 64
    else:
        receipt = json.loads(args["part1_build_receipt_bytes"])
        receipt["net_name"] = "yolo26s_part1_b071"
        bind_receipt(json.dumps(receipt, sort_keys=True).encode(), args)
    with pytest.raises(ValueError, match="split-manifest staging"):
        validate_generic_split_manifest_staging(original, staged, **args)


def test_normal_read_bound_rejects_tampered_local_boundary_mirror_bytes(tmp_path):
    from onnx_splitpoint_tool.generic_only_plan import read_bound, validation_scope

    original, _, args = bindings("b071")
    mirror = tmp_path / "bound_assets/b071/boundary_metadata/split_manifest.json"
    mirror.parent.mkdir(parents=True)
    mirror.write_bytes(original)
    descriptor = deepcopy(args["row"]["generic_artifact_binding"]["artifacts"]["boundary_metadata"])
    descriptor["local_path"] = str(mirror)

    with pytest.raises(ValueError, match="Generic-only source.*changed"):
        with validation_scope():
            assert read_bound(descriptor) == mirror
            mirror.write_bytes(b"x" + original[1:])
    with pytest.raises(ValueError, match="Generic-only source.*changed"):
        read_bound(descriptor)


@pytest.mark.parametrize("case,graph_case", [("b003", "b3"), ("b071", "b71")])
def test_graph_case_helper_preserves_existing_unpadded_names_for_same_integer_cut(case, graph_case):
    from onnx_splitpoint_tool.generic_split_manifest import _manifest_graph_case

    original = {
        "boundary": int(case[1:]), "boundary_index": int(case[1:]),
        "part1": f"detector_part1_{graph_case}.onnx",
        "part2": f"detector_part2_{graph_case}.onnx",
        "full_model": "../models/detector.onnx",
    }
    assert _manifest_graph_case(original, deepcopy(original), model="detector", case=case) == graph_case


@pytest.mark.parametrize("part,graph_case", [("part1", "b003"), ("part2", "b003"), ("part1", "b4")])
def test_graph_case_helper_cannot_mix_padding_or_change_original_integer_cut(part, graph_case):
    from onnx_splitpoint_tool.generic_split_manifest import _manifest_graph_case

    original = {
        "boundary": 3, "boundary_index": 3,
        "part1": "detector_part1_b3.onnx", "part2": "detector_part2_b3.onnx",
        "full_model": "../models/detector.onnx",
    }
    staged = deepcopy(original)
    staged[part] = f"detector_{part}_{graph_case}.onnx"
    with pytest.raises(ValueError, match="split-manifest staging"):
        _manifest_graph_case(original, staged, model="detector", case="b003")
