from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np

from onnx_splitpoint_tool import hailo_backend as backend
from onnx_splitpoint_tool.build_evidence_store import BuildEvidenceStore
from onnx_splitpoint_tool.preprocessing_contract import canonical_image_preprocessing_contract


@pytest.fixture
def harness(tmp_path, monkeypatch):
    source = tmp_path / "source.onnx"
    source.write_bytes(b"offline-fake-onnx-fixed-input-contract")
    calibration = tmp_path / "calibration"
    calibration.mkdir()
    np.save(calibration / "sample.npy", np.zeros((224, 224, 3), dtype=np.float32))
    contract = canonical_image_preprocessing_contract("classification", (224, 224))
    calls = []
    behavior = {"error": "Mapping Failed: concat22 Agent infeasible"}

    class FakeRunner:
        def __init__(self, **kwargs):
            calls.append("init")

        def translate_onnx_model(self, **kwargs):
            calls.append("translate")
            if behavior.get("parser_error"):
                raise RuntimeError(behavior["parser_error"])

        def get_hn_dict(self):
            return {"layers": {"images": {"type": "input_layer", "output_shape": [1, 224, 224, 3]}}}

        def load_model_script(self, script):
            pass

        def optimize(self, data):
            calls.append("optimize")

        def compile(self):
            calls.append("compile")
            if behavior["error"]:
                raise RuntimeError(behavior["error"])
            return b"fake-compiled-hef-with-sealed-receipt"

    monkeypatch.setitem(sys.modules, "hailo_sdk_client", SimpleNamespace(ClientRunner=FakeRunner))
    token = "hailo-dataflow-compiler:3.31.0"
    monkeypatch.setattr(backend, "_hailo_sdk_version_token", lambda: token)
    monkeypatch.setattr(backend, "_hailo_sdk_version_token_from_controller_metadata", lambda: token)
    monkeypatch.setattr(backend, "_hailo_sdk_version_token_from_managed_venv", lambda **kwargs: token)
    monkeypatch.setattr(backend, "_infer_hailo_image_preprocess", lambda **kwargs: contract["image_scale"])
    monkeypatch.setattr(backend, "hailo_dfc_workspace_preflight", lambda *args, **kwargs: {"status": "passed"})
    for name, value in {
        "ONNX_SPLITPOINT_HAILO_CACHE_ROOT": str(tmp_path / "cache"),
        "ONNX_SPLITPOINT_HAILO_CACHE_ENABLED": "1",
        "ONNX_SPLITPOINT_HAILO_CALIBRATION_STORAGE": "memory",
        "ONNX_SPLITPOINT_HAILO_CALIB_CAP_MB": "64",
        "ONNX_SPLITPOINT_BUILD_EVIDENCE_ROOT": str(tmp_path / "evidence"),
        "ONNX_SPLITPOINT_ARTIFACT_STORE_ROOT": str(tmp_path / "artifacts"),
        "ONNX_SPLITPOINT_ARTIFACT_STORE_ENABLED": "1",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("ONNX_SPLITPOINT_BUILD_EVIDENCE_CONTEXT_JSON", raising=False)
    kwargs = dict(net_name="yolo26s_part1_b364", hw_arch="hailo8",
                  net_input_shapes={"images": [1, 3, 224, 224]}, task="classification",
                  preprocessing_contract=contract, fixup=False, opt_level=1,
                  calib_count=1, calib_batch_size=1, calib_dir=calibration)
    context = {"stage": "part1", "model_id": "yolo26s", "split_manifest": {"boundary": 364},
               "full_source_onnx_path": str(source),
               "full_source_onnx_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
    return SimpleNamespace(source=source, calls=calls, behavior=behavior, kwargs=kwargs,
                           context=context, root=tmp_path)


def _build(h, number, **overrides):
    options = {**h.kwargs, "build_evidence_context": h.context, **overrides}
    return backend.hailo_build_hef_auto(h.source, backend="local", outdir=h.root / f"run-{number}", **options)


@pytest.mark.parametrize("stage", ["full", "part1", "part2"])
@pytest.mark.parametrize("arch", ["hailo8", "hailo10h"])
def test_normal_build_records_exact_failure_and_new_run_skips_compiler(harness, stage, arch):
    h = harness
    h.context["stage"] = stage
    h.kwargs.update(hw_arch=arch, net_name=f"yolo26s_{stage}_b364")
    first = _build(h, 1)
    assert not first.ok
    assert h.calls == ["init", "translate", "optimize", "compile"]
    assert first.details["build_evidence"]["recorded_state"] == "COMPILE_INFEASIBLE"
    before = BuildEvidenceStore().index_path.read_bytes()
    second = _build(h, 2)
    assert not second.ok and second.skipped
    assert second.failure_kind == "known_negative_build_evidence"
    assert second.details["build_evidence"]["state"] == "COMPILE_INFEASIBLE"
    assert len(h.calls) == 4
    assert BuildEvidenceStore().index_path.read_bytes() == before
    receipt = json.loads(next((h.root / "run-1" / "hailo_attempt_receipts").glob("attempt_*.json")).read_text())
    terminals = [json.loads(p.read_text()) for p in (h.root / "run-1" / "hailo_attempt_receipts").glob("attempt_*.json")
                 if not p.name.endswith(".started.json")]
    assert terminals[0]["details"]["build_evidence"]["key"] == first.details["build_evidence"]["key"]


@pytest.mark.parametrize("mode", ["venv", "auto"])
def test_managed_dispatch_is_prevented_before_any_compiler_child(harness, monkeypatch, mode):
    h = harness
    _build(h, 1)
    def forbidden(*args, **kwargs):
        raise AssertionError("known failed build dispatched")
    monkeypatch.setattr(backend, "hailo_build_hef_via_venv", forbidden)
    monkeypatch.setattr(backend, "hailo_build_hef_via_wsl", forbidden)
    # Controller metadata in this test matches the managed target exactly.
    result = backend.hailo_build_hef_auto(h.source, backend=mode, outdir=h.root / "managed",
                                        build_evidence_context=h.context, **h.kwargs)
    assert result.failure_kind == "known_negative_build_evidence"
    assert len(h.calls) == 4


def test_wsl_uses_target_identity_in_guarded_helper_not_controller_guess(harness, monkeypatch):
    h = harness
    _build(h, 1)
    bridges = []
    def bridge(source, **kwargs):
        bridges.append(True)
        allowed = {key: value for key, value in kwargs.items()
                   if key not in {"wsl_distro", "wsl_venv_activate", "wsl_timeout_s", "on_log"}}
        return backend.hailo_build_hef(source, **allowed)
    monkeypatch.setattr(backend, "hailo_build_hef_via_wsl", bridge)
    result = backend.hailo_build_hef_auto(h.source, backend="wsl", outdir=h.root / "wsl",
                                        build_evidence_context=h.context, **h.kwargs)
    assert bridges == [True]
    assert result.failure_kind == "known_negative_build_evidence"
    assert len(h.calls) == 4


def test_parser_failure_is_persistent_before_next_parser_initializes(harness):
    h = harness
    h.behavior["parser_error"] = "UnsupportedShuffleLayerError: unsupported operation at node"
    first = _build(h, 1)
    assert first.details["build_evidence"]["recorded_state"] == "PARSER_UNSUPPORTED"
    second = _build(h, 2)
    assert second.failure_kind == "known_negative_build_evidence"
    assert second.details["build_evidence"]["state"] == "PARSER_UNSUPPORTED"
    assert h.calls == ["init", "translate"]


def test_cache_only_lookup_and_force_retries_do_not_learn_or_repeat(harness):
    h = harness
    first = _build(h, 1, cache_only=True)
    assert first.skipped and not BuildEvidenceStore().root.exists()
    assert not h.calls
    _build(h, 2)
    before = BuildEvidenceStore().index_path.read_bytes()
    result = _build(h, 3, cache_only=True)
    assert result.failure_kind == "known_negative_build_evidence"
    assert BuildEvidenceStore().index_path.read_bytes() == before
    # v34's public API refuses productive Force before any new attempt. The
    # exact negative still survives unchanged and no compiler repeats.
    with pytest.raises(ValueError, match="force_build_disabled_for_productive_jobs"):
        _build(h, 4, force=True)
    assert BuildEvidenceStore().index_path.read_bytes() == before
    assert len(h.calls) == 4


@pytest.mark.parametrize("changed", ["recipe", "boundary", "source", "compiler", "preprocessing"])
def test_changed_exact_identity_is_allowed_to_attempt_again(harness, monkeypatch, changed):
    h = harness
    _build(h, 1)
    if changed == "recipe":
        h.kwargs["opt_level"] = 2
    elif changed == "boundary":
        h.context["split_manifest"]["boundary"] = 365
    elif changed == "source":
        h.source.write_bytes(b"new-source-model")
        h.context["full_source_onnx_sha256"] = hashlib.sha256(h.source.read_bytes()).hexdigest()
    elif changed == "compiler":
        monkeypatch.setattr(backend, "_hailo_sdk_version_token", lambda: "hailo-dataflow-compiler:3.32.0")
    else:
        h.kwargs["preprocessing_contract"] = canonical_image_preprocessing_contract("detection", (224, 224))
        h.kwargs["task"] = "detection"
        monkeypatch.setattr(backend, "_infer_hailo_image_preprocess",
                            lambda **kwargs: h.kwargs["preprocessing_contract"]["image_scale"])
    result = _build(h, 2)
    assert result.failure_kind != "known_negative_build_evidence"
    assert h.calls.count("compile") == 2


@pytest.mark.parametrize("error", ["CUDA memory allocation failed: out of memory",
                                  "Mapping Failed: Agent infeasible; CUDA out of memory",
                                  "compiler timed out", "cancelled by user"])
def test_infrastructure_abort_and_oom_never_block_a_later_run(harness, error):
    h = harness
    h.behavior["error"] = error
    for n in (1, 2):
        result = _build(h, n)
        assert result.failure_kind != "known_negative_build_evidence"
    assert h.calls.count("compile") == 2


def test_positive_verified_hef_wins_over_historical_negative(harness):
    h = harness
    h.behavior["error"] = ""
    first = _build(h, 1)
    assert first.ok, first.error
    # A contradictory historical negative must not hide the actual sealed HEF.
    key = first.details["build_evidence"]["key"]
    BuildEvidenceStore().record(key, "COMPILE_INFEASIBLE", evidence_origin={"source": "historical"})
    second = _build(h, 2)
    assert second.ok and second.skipped
    assert h.calls.count("compile") == 1


def test_missing_full_context_never_guesses_a_negative(harness):
    h = harness
    _build(h, 1)
    result = _build(h, 2, build_evidence_context={})
    assert result.failure_kind != "known_negative_build_evidence"
    assert result.details["build_evidence"]["status"] == "UNAVAILABLE"
    assert "recorded" not in result.details["build_evidence"]
    assert h.calls.count("compile") == 2


def test_changed_full_source_under_retained_context_blocks_stale_split(harness):
    h = harness
    _build(h, 1)
    h.source.write_bytes(b"different-full-source-after-split-selection")
    result = _build(h, 2)
    assert result.failure_kind == "build_evidence_conflict"
    assert result.details["build_evidence"]["status"] == "CONFLICT"
    assert result.details["build_evidence"]["reason"] == "full_source_onnx_identity_changed"
    assert result.details["build_evidence"]["compiler_dispatch_allowed"] is False
    assert h.calls.count("compile") == 1


def test_deferred_windows_context_is_attested_by_actual_linux_helper(harness, monkeypatch):
    h = harness
    # The Windows controller retained the translated actual source path; its
    # unavailable POSIX reader supplied neither a fake digest nor an error.
    deferred = {k: v for k, v in h.context.items() if k != "full_source_onnx_sha256"}
    monkeypatch.setenv("ONNX_SPLITPOINT_BUILD_EVIDENCE_CONTEXT_JSON", json.dumps(deferred))
    first = backend.hailo_build_hef(h.source, outdir=h.root / "child-first", **h.kwargs)
    key = first.details["build_evidence"]["key"]
    assert key["full_source_onnx_sha256"] == hashlib.sha256(h.source.read_bytes()).hexdigest()
    second = backend.hailo_build_hef(h.source, outdir=h.root / "child-second", **h.kwargs)
    assert second.failure_kind == "known_negative_build_evidence"
    assert h.calls.count("compile") == 1


def test_explicit_bad_context_cannot_fall_back_to_full_builder_identity(harness):
    h = harness
    h.context.update(stage="full", identity_error="full_source_onnx_unavailable")
    h.context.pop("full_source_onnx_sha256")
    h.context.pop("full_source_onnx_path")
    result = _build(h, 1)
    assert result.details["build_evidence"]["status"] == "UNAVAILABLE"
    assert result.details["build_evidence"]["reason"] == "full_source_onnx_unavailable"
    assert not BuildEvidenceStore().root.exists()


def test_old_negative_run_diagnostic_does_not_survive_recipe_change(harness):
    h = harness
    _build(h, 1)
    _build(h, 2)
    h.kwargs["opt_level"] = 2
    h.behavior["error"] = ""
    result = _build(h, 2)
    assert result.ok
    diagnostic = json.loads((h.root / "run-2/hailo_negative_evidence.json").read_text())
    assert diagnostic["negative_evidence_hit"] is False


def test_corrupt_index_stops_unknown_repeat_with_visible_reason(harness):
    h = harness
    store = BuildEvidenceStore()
    store.root.mkdir()
    store.index_path.write_text("{broken")
    result = _build(h, 1)
    assert result.failure_kind == "build_evidence_error"
    assert result.details["build_evidence"]["status"] == "ERROR"
    assert not h.calls


def test_direct_local_helper_reads_the_same_context_from_environment(harness, monkeypatch):
    h = harness
    monkeypatch.setenv("ONNX_SPLITPOINT_BUILD_EVIDENCE_CONTEXT_JSON", json.dumps(h.context))
    first = backend.hailo_build_hef(h.source, outdir=h.root / "direct-first", **h.kwargs)
    second = backend.hailo_build_hef(h.source, outdir=h.root / "direct-second", **h.kwargs)
    assert first.details["build_evidence"]["recorded"]
    assert second.failure_kind == "known_negative_build_evidence"
    assert h.calls.count("compile") == 1
