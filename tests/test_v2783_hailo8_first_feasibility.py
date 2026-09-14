from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import jsonschema
import pytest
import yaml

from onnx_splitpoint_tool.benchmark.services import (
    BenchmarkGenerationExecutionCallbacks,
    BenchmarkGenerationExecutionConfig,
    BenchmarkGenerationExecutionService,
    BenchmarkGenerationOrchestrationConfig,
    BenchmarkGenerationOrchestrationService,
    BenchmarkGenerationRuntime,
    BenchmarkGenerationService,
    _hailo_feasibility_file_sha256_v2783,
    _new_hailo_feasibility_state_v2783,
    _revalidate_hailo_feasibility_anchor_v2783,
    _run_hailo8_first_feasibility_v2783,
    normalize_hailo_feasibility_control,
)
from onnx_splitpoint_tool.build_evidence import canonical_sha256


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = (
    ROOT
    / "onnx_splitpoint_tool"
    / "resources"
    / "schemas"
    / "evaluation_profile.schema.json"
)
MATERIALIZED_COMPLETE_SET = Path(
    "/workspace/scratch/514274bf75c1/materialized/"
    "Complete_Set_v2782_canary_b5.yaml"
)


def _control(**updates):
    value = {
        "enabled": True,
        "mode": "hailo8_first_common_anchor",
        "primary_target": "hailo8",
        "gated_target": "hailo10h",
        "required_variant": "part1",
        "max_hailo8_cold_attempts": 3,
        "max_total_cold_builds": 4,
        "max_cold_builds_per_boundary": 2,
        "wall_time_budget_s": 3600,
        "parser_timeout_s": 30,
        "allow_recipe_retries": False,
        "stop_workflow_on_exhaustion": True,
    }
    value.update(updates)
    return value


def _target_outcome(target: str, *, ok: bool, cache_hit: bool = False, error=""):
    target_output = {
        "part1_build": {
            "ok": bool(ok),
            "cache_hit": bool(cache_hit),
            "cache_key": f"{'a' if target == 'hailo8' else 'b'}" * 64,
            "cache_payload_v3": {
                "model_sha256": "c" * 64,
            },
            "failure_kind": "mapping_failed" if error else None,
            "error": error or None,
        }
    }
    if ok:
        target_output["part1"] = f"hailo/{target}/part1/compiled.hef"
    return {
        "hw_arch": target,
        "target_output": target_output,
        "errors": [] if ok else [error or "cache miss"],
        "failure_records": [],
        "first_rejection": None,
        "row_per_cut_hints": [],
        "diagnostics": [],
        "full_metadata": {},
    }


class _Builder:
    def __init__(self, behavior):
        self.behavior = behavior
        self.calls = []

    def __call__(
        self,
        target,
        backend,
        *,
        cache_only_override,
        allow_retries,
        part1_only,
        timeout_override_s=None,
    ):
        self.calls.append((target, bool(cache_only_override)))
        assert allow_retries is False
        assert part1_only is True
        assert timeout_override_s is None or timeout_override_s > 0
        return self.behavior(target, bool(cache_only_override))


def _run(builder, *, state=None, parser=None, evidence=None, control=None):
    return _run_hailo8_first_feasibility_v2783(
        control=control or _control(),
        boundary=52,
        candidate_order=[52, 84, 116],
        targets=["hailo8", "hailo10h"],
        backend="venv",
        builder=builder,
        parser_preflight=parser
        or (
            lambda _target, **_kwargs: SimpleNamespace(
                ok=True, error=None, elapsed_s=0.1
            )
        ),
        state=state,
        evidence_key_base={
            "full_source_onnx_sha256": "1" * 64,
            "builder_source_onnx_sha256": "2" * 64,
            "compiler_onnx_sha256": "",
            "boundary_endpoint_contract_sha256": "3" * 64,
        },
        evidence_lookup=evidence,
    )


def test_hailo10_cold_build_is_never_dispatched_after_hailo8_failure():
    def behavior(target, cache_only):
        if cache_only:
            return _target_outcome(target, ok=False)
        assert target == "hailo8"
        return _target_outcome(target, ok=False, error="mapping failed")

    builder = _Builder(behavior)
    result = _run(builder)

    assert builder.calls == [
        ("hailo8", True),
        ("hailo8", False),
        ("hailo10h", True),
    ]
    assert result["candidate_receipt"]["target_outcomes"] == {
        "hailo8": "COMPILE_INFEASIBLE",
        "hailo10h": "GATED_BY_HAILO8",
    }


def test_hailo8_artifact_pass_unlocks_budgeted_hailo10_cold_build():
    def behavior(target, cache_only):
        if cache_only:
            return _target_outcome(target, ok=False)
        return _target_outcome(target, ok=True)

    builder = _Builder(behavior)
    result = _run(builder)

    assert builder.calls == [
        ("hailo8", True),
        ("hailo8", False),
        ("hailo10h", True),
        ("hailo10h", False),
    ]
    assert result["outcome"] == "ANCHOR_FOUND"
    assert result["state"]["cold_builds"] == 2
    assert result["state"]["hailo8_cold_attempts"] == 1


def test_exact_hailo10_cache_hit_is_preserved_when_hailo8_fails():
    def behavior(target, cache_only):
        if target == "hailo10h" and cache_only:
            return _target_outcome(target, ok=True, cache_hit=True)
        if cache_only:
            return _target_outcome(target, ok=False)
        return _target_outcome(target, ok=False, error="mapping failed")

    builder = _Builder(behavior)
    result = _run(builder)

    assert builder.calls[-1] == ("hailo10h", True)
    assert ("hailo10h", False) not in builder.calls
    assert result["candidate_receipt"]["target_outcomes"]["hailo10h"] == (
        "ARTIFACT_PASS"
    )
    assert result["target_outcomes"][1]["target_output"]["part1_build"][
        "cache_hit"
    ] is True


def test_exact_negative_evidence_skips_parser_and_cold_compiler():
    parser_calls = []

    def evidence(request):
        if request["target"] == "hailo8":
            return {
                "exact": True,
                "outcome": "COMPILE_INFEASIBLE",
                "origin": {"kind": "prior_b5"},
            }
        return {"exact": False, "outcome": "MISS"}

    builder = _Builder(lambda target, cache_only: _target_outcome(target, ok=False))
    result = _run(
        builder,
        parser=lambda target: parser_calls.append(target),
        evidence=evidence,
    )

    assert parser_calls == []
    assert builder.calls == [("hailo8", True), ("hailo10h", True)]
    assert result["state"]["cold_builds"] == 0


def test_unverified_positive_evidence_cannot_create_an_anchor():
    fake_positive = _target_outcome("hailo8", ok=True, cache_hit=True)

    def evidence(request):
        if request["target"] == "hailo8":
            return {
                "exact": True,
                "outcome": "ARTIFACT_PASS",
                "target_outcome": fake_positive,
            }
        return {"exact": False, "outcome": "MISS"}

    builder = _Builder(
        lambda target, cache_only: (
            _target_outcome(target, ok=False)
            if cache_only
            else _target_outcome(target, ok=False, error="mapping failed")
        )
    )
    result = _run(builder, evidence=evidence)

    assert ("hailo8", False) in builder.calls
    assert result["candidate_receipt"]["target_outcomes"]["hailo8"] == (
        "COMPILE_INFEASIBLE"
    )


def test_budget_is_persisted_per_boundary_and_resume_order_is_frozen():
    builder = _Builder(
        lambda target, cache_only: (
            _target_outcome(target, ok=False)
            if cache_only
            else _target_outcome(target, ok=False, error="mapping failed")
        )
    )
    state = {}
    result = _run(
        builder,
        state=state,
        control=_control(max_hailo8_cold_attempts=1, max_total_cold_builds=1),
    )

    assert result["outcome"] == "RUNNING"
    assert result["candidate_receipt"]["cold_budget_exhausted"] is True
    assert state["attempts_by_boundary"]["52"]["cold_builds"] == 1
    assert state["candidate_order"] == [52, 84, 116]
    with pytest.raises(ValueError, match="candidate order changed"):
        _run_hailo8_first_feasibility_v2783(
            control=_control(
                max_hailo8_cold_attempts=1,
                max_total_cold_builds=1,
            ),
            boundary=84,
            candidate_order=[84, 52, 116],
            targets=["hailo8", "hailo10h"],
            backend="venv",
            builder=builder,
            parser_preflight=lambda _target, **_kwargs: SimpleNamespace(ok=True),
            state=state,
        )
    with pytest.raises(ValueError, match="normalized control changed"):
        _run_hailo8_first_feasibility_v2783(
            control=_control(),
            boundary=84,
            candidate_order=[52, 84, 116],
            targets=["hailo8", "hailo10h"],
            backend="venv",
            builder=builder,
            parser_preflight=lambda _target, **_kwargs: SimpleNamespace(ok=True),
            state=state,
        )


def test_control_rejects_unbudgeted_recipe_retry_and_half_bound_ledger():
    with pytest.raises(ValueError, match="recipe retries"):
        normalize_hailo_feasibility_control(
            _control(allow_recipe_retries=True)
        )
    with pytest.raises(ValueError, match="configured together"):
        normalize_hailo_feasibility_control(
            _control(evidence_index_path="/tmp/index.json")
        )


def test_complete_set_profile_accepts_opt_in_gate_a_schema_without_mutation():
    profile_path = (
        MATERIALIZED_COMPLETE_SET
        if MATERIALIZED_COMPLETE_SET.is_file()
        else ROOT / "profiles" / "resnet50_v2772_hailo_parallel_build_canary.yaml"
    )
    if profile_path == MATERIALIZED_COMPLETE_SET:
        assert hashlib.sha256(profile_path.read_bytes()).hexdigest() == (
            "13f486b0e53e335dfb4407bc09f364a116a3cfc774757e3a46bfd3c7c86da91f"
        )
    source = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    source_before = copy.deepcopy(source)
    candidate = copy.deepcopy(source)
    candidate.setdefault("hailo_build", {})["feasibility_control"] = _control()
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))

    jsonschema.validate(candidate, schema)
    assert source == source_before

    invalid = copy.deepcopy(candidate)
    invalid["hailo_build"]["feasibility_control"].pop(
        "allow_recipe_retries"
    )
    invalid["hailo_build"]["feasibility_control"][
        "allow_recipe_retries"
    ] = True
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(invalid, schema)


def test_cold_budget_exhaustion_still_scans_later_exact_cache_anchor():
    control = _control(
        max_hailo8_cold_attempts=1,
        max_total_cold_builds=1,
    )
    state = {}
    first = _Builder(
        lambda target, cache_only: (
            _target_outcome(target, ok=False)
            if cache_only
            else _target_outcome(target, ok=False, error="mapping failed")
        )
    )
    first_result = _run(first, state=state, control=control)
    assert first_result["outcome"] == "RUNNING"
    assert state["cold_builds"] == 1

    second = _Builder(
        lambda target, cache_only: _target_outcome(
            target, ok=True, cache_hit=True
        )
    )
    second_result = _run_hailo8_first_feasibility_v2783(
        control=control,
        boundary=84,
        candidate_order=[52, 84, 116],
        targets=["hailo8", "hailo10h"],
        backend="venv",
        builder=second,
        parser_preflight=lambda _target, **_kwargs: SimpleNamespace(ok=True),
        state=state,
        evidence_key_base={"full_source_onnx_sha256": "1" * 64},
    )
    assert second.calls == [("hailo8", True), ("hailo10h", True)]
    assert second_result["outcome"] == "ANCHOR_FOUND"
    assert state["cold_builds"] == 1


def test_cold_reservation_is_durable_before_crash_and_blocks_retry():
    state = {}
    snapshots = []

    class CrashBuilder(_Builder):
        def __call__(self, target, backend, **kwargs):
            if not bool(kwargs["cache_only_override"]):
                raise KeyboardInterrupt("simulated compiler kill")
            return super().__call__(target, backend, **kwargs)

    crash = CrashBuilder(
        lambda target, cache_only: _target_outcome(target, ok=False)
    )
    with pytest.raises(KeyboardInterrupt):
        _run_hailo8_first_feasibility_v2783(
            control=_control(
                max_hailo8_cold_attempts=1,
                max_total_cold_builds=1,
            ),
            boundary=52,
            candidate_order=[52, 84, 116],
            targets=["hailo8", "hailo10h"],
            backend="venv",
            builder=crash,
            parser_preflight=lambda _target, **_kwargs: SimpleNamespace(ok=True),
            state=state,
            evidence_key_base={"full_source_onnx_sha256": "1" * 64},
            persist_reservation=lambda: snapshots.append(copy.deepcopy(state)),
        )
    assert snapshots[-1]["cold_builds"] == 1
    assert snapshots[-1]["attempts_by_boundary"]["52"]["cold_builds"] == 1

    resumed = _Builder(
        lambda target, cache_only: _target_outcome(target, ok=False)
    )
    result = _run_hailo8_first_feasibility_v2783(
        control=_control(
            max_hailo8_cold_attempts=1,
            max_total_cold_builds=1,
        ),
        boundary=52,
        candidate_order=[52, 84, 116],
        targets=["hailo8", "hailo10h"],
        backend="venv",
        builder=resumed,
        parser_preflight=lambda _target, **_kwargs: SimpleNamespace(ok=True),
        state=state,
        evidence_key_base={"full_source_onnx_sha256": "1" * 64},
    )
    assert resumed.calls == [("hailo8", True), ("hailo10h", True)]
    assert result["outcome"] == "RUNNING"
    assert state["cold_builds"] == 1


def test_wall_budget_caps_dispatch_and_stops_before_hailo10():
    now = [0.0]

    def clock():
        return now[0]

    class TimedBuilder(_Builder):
        def __call__(self, target, backend, **kwargs):
            assert 0 < float(kwargs["timeout_override_s"]) <= 3.0
            result = super().__call__(target, backend, **kwargs)
            now[0] += 4.0
            return result

    builder = TimedBuilder(
        lambda target, cache_only: _target_outcome(target, ok=False)
    )
    result = _run_hailo8_first_feasibility_v2783(
        control=_control(wall_time_budget_s=3),
        boundary=52,
        candidate_order=[52, 84],
        targets=["hailo8", "hailo10h"],
        backend="venv",
        builder=builder,
        parser_preflight=lambda _target, **_kwargs: SimpleNamespace(ok=True),
        evidence_key_base={"full_source_onnx_sha256": "1" * 64},
        clock=clock,
    )
    assert builder.calls == [("hailo8", True)]
    assert result["outcome"] == "CANARY_BUDGET_EXHAUSTED"
    assert result["state"]["exhaustion_reason"] == "wall_time_budget_exhausted"


def test_evidence_exception_is_terminal_conflict_without_cold_build():
    builder = _Builder(
        lambda target, cache_only: _target_outcome(target, ok=False)
    )

    def broken_evidence(_request):
        raise RuntimeError("tampered materialization")

    result = _run(builder, evidence=broken_evidence)
    assert builder.calls == [("hailo8", True)]
    assert result["outcome"] == "EVIDENCE_CONFLICT"
    assert result["state"]["cold_builds"] == 0


def test_forged_terminal_anchor_never_early_returns_without_revalidation():
    builder = _Builder(
        lambda target, cache_only: _target_outcome(target, ok=False)
    )
    state = {}
    _run(builder, state=state)
    state["outcome"] = "ANCHOR_FOUND"
    state["anchor_boundary"] = 52
    result = _run(builder, state=state)
    assert result["outcome"] == "EVIDENCE_CONFLICT"


def test_exact_source_hash_rejects_symlink_and_concurrent_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source = tmp_path / "part1.onnx"
    source.write_bytes(b"a" * (1024 * 1024 + 64))
    alias = tmp_path / "alias.onnx"
    alias.symlink_to(source)
    with pytest.raises(ValueError, match="unsafe|regular"):
        _hailo_feasibility_file_sha256_v2783(alias)

    replacement = tmp_path / "replacement.onnx"
    replacement.write_bytes(b"b" * source.stat().st_size)
    original_read = os.read
    swapped = [False]

    def replacing_read(fd, count):
        block = original_read(fd, count)
        if block and not swapped[0]:
            swapped[0] = True
            os.replace(replacement, source)
        return block

    monkeypatch.setattr(os, "read", replacing_read)
    with pytest.raises(ValueError, match="changed"):
        _hailo_feasibility_file_sha256_v2783(source)


def _write_anchor_artifact(
    case_dir: Path,
    target_folder: str,
    receipt_arch: str,
    builder: Path,
    boundary: int,
) -> str:
    artifact_dir = case_dir / "hailo" / target_folder / "part1"
    artifact_dir.mkdir(parents=True)
    compiler = artifact_dir / f"model_part1_b{boundary}_hailo_fixed.onnx"
    hef = artifact_dir / "compiled.hef"
    compiler.write_bytes(f"compiler-{target_folder}".encode())
    hef.write_bytes(f"hef-{target_folder}".encode())
    preprocessing = {
        "schema": "onnx-splitpoint/image-preprocessing-contract",
        "schema_version": 2,
        "task": "detection",
        "target_hw": [640, 640],
        "image_scale": "norm",
    }
    preprocessing_sha = canonical_sha256(preprocessing)
    calibration_identity = "manifest:" + "9" * 64
    prepared_sha = canonical_sha256(
        {
            "calibration_identity": calibration_identity,
            "preprocessing_contract_sha256": preprocessing_sha,
        }
    )
    payload = {
        "schema": "onnx-splitpoint/hailo-hef-cache-key-v3",
        "model_sha256": hashlib.sha256(compiler.read_bytes()).hexdigest(),
        "activation_part1_sha256": "",
        "hw_arch": receipt_arch,
        "hailo_sdk_version": "hailo-dataflow-compiler:3.33.1",
        "optimization_level": 1,
        "calibration_identity": calibration_identity,
        "prepared_calibration_identity_sha256": prepared_sha,
        "calibration_count": 5,
        "requested_calibration_count": 5,
        "calibration_storage": "memory",
        "calibration_memory_cap_bytes": 1048576,
        "calibration_batch_size": 1,
        "extra_model_script": "",
        "start_nodes": [],
        "end_nodes": ["/head/Conv"],
        "integrity": "strict",
        "preprocessing_contract": preprocessing,
        "preprocessing_contract_sha256": preprocessing_sha,
        "net_name": f"model_part1_b{boundary}",
        "net_input_shapes": [1, 3, 640, 640],
        "disable_rt_metadata_extraction": True,
    }
    cache_key = canonical_sha256(payload)
    receipt = {
        "schema": "onnx-splitpoint/hailo-hef-build-receipt/v2",
        "source_onnx_sha256": hashlib.sha256(builder.read_bytes()).hexdigest(),
        "compiler_onnx_sha256": hashlib.sha256(compiler.read_bytes()).hexdigest(),
        "compiler_onnx_filename": compiler.name,
        "hef_sha256": hashlib.sha256(hef.read_bytes()).hexdigest(),
        "hef_size_bytes": hef.stat().st_size,
        "hw_arch": receipt_arch,
        "net_name": f"model_part1_b{boundary}",
        "hailo_sdk_version": payload["hailo_sdk_version"],
        "calibration_identity": calibration_identity,
        "prepared_calibration_identity_sha256": prepared_sha,
        "calibration_count": 5,
        "requested_calibration_count": 5,
        "calibration_storage": "memory",
        "calibration_memory_cap_bytes": 1048576,
        "preprocessing_contract": preprocessing,
        "preprocessing_contract_sha256": preprocessing_sha,
        "cache_key": cache_key,
        "cache_payload": payload,
    }
    (artifact_dir / "hailo_hef_build_receipt.json").write_text(
        json.dumps(receipt), encoding="utf-8"
    )
    return cache_key


def test_anchor_resume_revalidates_hailo10_alias_and_detects_tamper(
    tmp_path: Path,
):
    root = tmp_path / "suite"
    case = root / "b052"
    models = root / "models"
    case.mkdir(parents=True)
    models.mkdir(parents=True)
    full = models / "model.onnx"
    builder = case / "model_part1_b52.onnx"
    full.write_bytes(b"full-model")
    builder.write_bytes(b"builder-model")
    keys = {
        "hailo8": _write_anchor_artifact(
            case, "hailo8", "hailo8", builder, 52
        ),
        "hailo10": _write_anchor_artifact(
            case, "hailo10", "hailo10h", builder, 52
        ),
    }
    manifest = {
        "schema": "onnx-splitpoint/split-manifest",
        "boundary": 52,
        "full_model": "../models/model.onnx",
        "part1": builder.name,
        "hailo": {
            "hefs": {
                target: {"part1": f"hailo/{target}/part1/compiled.hef"}
                for target in keys
            }
        },
    }
    (case / "split_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    control = normalize_hailo_feasibility_control(_control())
    state = _new_hailo_feasibility_state_v2783(
        control=control,
        candidate_order=[52],
        targets=["hailo8", "hailo10"],
        backend="venv",
        full_source_onnx_sha256=hashlib.sha256(full.read_bytes()).hexdigest(),
    )
    state.update({
        "outcome": "ANCHOR_FOUND",
        "anchor_boundary": 52,
        "candidates": [
            {
                "boundary": 52,
                "anchor": True,
                "target_outcomes": {
                    "hailo8": "ARTIFACT_PASS",
                    "hailo10": "ARTIFACT_PASS",
                },
                "phases": [
                    {
                        "target": target,
                        "cache_key_v3": cache_key,
                    }
                    for target, cache_key in keys.items()
                ],
            }
        ],
    })
    kwargs = {
        "out_dir": root,
        "cases": [
            {"boundary": 52, "case_dir": "b052", "manifest": "split_manifest.json"}
        ],
        "completed_boundaries": {52},
        "accepted_boundaries": {52},
        "targets": ["hailo8", "hailo10"],
    }
    assert _revalidate_hailo_feasibility_anchor_v2783(state, **kwargs)

    runtime = BenchmarkGenerationRuntime(
        out_dir=root,
        bench_log_path=root / "benchmark.log",
        state_path=root / "generation_state.json",
        requested_cases=1,
        ranked_candidates=[52],
        candidate_search_pool=[52],
        model_name="model",
        model_source=str(full),
        hef_full_policy="skip",
        completed_boundaries={52},
        accepted_boundaries={52},
        cases=list(kwargs["cases"]),
        hailo_feasibility_state=copy.deepcopy(state),
    )
    persisted = []
    execution = BenchmarkGenerationExecutionConfig(
        runtime=runtime,
        target_cases=1,
        gap=0,
        ranked_candidates=[52],
        candidate_search_pool=[52],
        out_dir=root,
        base="model",
        pad=3,
        strict_boundary=False,
        model=object(),
        nodes=[],
        order=[],
        analysis_payload={},
        full_model_src=str(full),
        full_model_dst=str(full),
        hef_targets=["hailo8", "hailo10"],
        hef_part1=True,
        hef_part2=False,
        hef_backend="venv",
        hailo_feasibility_control=control,
    )
    callbacks = BenchmarkGenerationExecutionCallbacks(
        log=lambda *_args, **_kwargs: None,
        queue_put=lambda *_args, **_kwargs: None,
        persist_state=lambda **value: persisted.append(dict(value)),
        publish_hailo_diagnostics=lambda *_args, **_kwargs: None,
        predicted_metrics_for_boundary=lambda *_args, **_kwargs: {},
        hailo_parse_entry_for_boundary=lambda *_args, **_kwargs: None,
        hailo_parse_scalar_fields=lambda *_args, **_kwargs: {},
    )
    assert BenchmarkGenerationExecutionService().execute_case_build_loop(
        execution, callbacks
    ) == [52]
    assert runtime.hailo_feasibility_state["resume_anchor_revalidation"] == "PASS"
    assert persisted[-1]["status"] == "complete"

    (case / "hailo/hailo10/part1/compiled.hef").write_bytes(b"tampered")
    assert not _revalidate_hailo_feasibility_anchor_v2783(state, **kwargs)


def test_hailo10_profile_axis_is_preserved_in_outer_outcomes():
    builder = _Builder(
        lambda target, cache_only: _target_outcome(
            target, ok=True, cache_hit=True
        )
    )
    result = _run_hailo8_first_feasibility_v2783(
        control=_control(gated_target="hailo10h"),
        boundary=52,
        candidate_order=[52],
        targets=["hailo8", "hailo10"],
        backend="venv",
        builder=builder,
        parser_preflight=lambda _target, **_kwargs: SimpleNamespace(ok=True),
        evidence_key_base={"full_source_onnx_sha256": "1" * 64},
    )
    assert result["outcome"] == "ANCHOR_FOUND"
    assert [row["hw_arch"] for row in result["target_outcomes"]] == [
        "hailo8",
        "hailo10",
    ]
    assert result["candidate_receipt"]["target_outcomes"] == {
        "hailo8": "ARTIFACT_PASS",
        "hailo10": "ARTIFACT_PASS",
    }


@pytest.mark.parametrize("suffix", [".export.json", ".categories.json"])
def test_gate_resume_never_overwrites_conflicting_portable_sidecar(
    tmp_path: Path, suffix: str
):
    source = tmp_path / "source" / "model.onnx"
    suite = tmp_path / "suite"
    source.parent.mkdir()
    (suite / "models").mkdir(parents=True)
    source.write_bytes(b"model")
    (suite / "models/model.onnx").write_bytes(b"model")
    source.with_suffix(suffix).write_bytes(b"new-sidecar")
    destination_sidecar = (suite / "models/model.onnx").with_suffix(suffix)
    destination_sidecar.write_bytes(b"old-sidecar")
    runtime = SimpleNamespace(
        out_dir=suite,
        errors=[],
        hailo_feasibility_state={"outcome": "ANCHOR_FOUND"},
    )
    with pytest.raises(ValueError, match="different bytes"):
        BenchmarkGenerationService().copy_portable_full_model(runtime, source)
    assert destination_sidecar.read_bytes() == b"old-sidecar"


def test_gate_resume_rejects_stale_destination_only_sidecar(tmp_path: Path):
    source = tmp_path / "source" / "model.onnx"
    suite = tmp_path / "suite"
    source.parent.mkdir()
    (suite / "models").mkdir(parents=True)
    source.write_bytes(b"model")
    (suite / "models/model.onnx").write_bytes(b"model")
    stale = suite / "models/model.categories.json"
    stale.write_bytes(b"stale")
    runtime = SimpleNamespace(
        out_dir=suite,
        errors=[],
        hailo_feasibility_state={"outcome": "ANCHOR_FOUND"},
    )
    with pytest.raises(ValueError, match="stale destination-only"):
        BenchmarkGenerationService().copy_portable_full_model(runtime, source)
    assert stale.read_bytes() == b"stale"


def test_gate_resume_rejects_changed_full_source_identity():
    builder = _Builder(
        lambda target, cache_only: _target_outcome(target, ok=False)
    )
    state = {}
    _run(builder, state=state)
    state["full_source_onnx_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="full source identity changed"):
        _run(builder, state=state)


def test_terminal_gate_persist_failure_propagates_before_return(tmp_path: Path):
    full = tmp_path / "model.onnx"
    full.write_bytes(b"full")
    control = normalize_hailo_feasibility_control(_control())
    state = _new_hailo_feasibility_state_v2783(
        control=control,
        candidate_order=[52],
        targets=["hailo8", "hailo10h"],
        backend="venv",
        full_source_onnx_sha256=hashlib.sha256(full.read_bytes()).hexdigest(),
    )
    state["outcome"] = "CANARY_BUDGET_EXHAUSTED"
    state["exhaustion_reason"] = "candidate_pool_exhausted"
    runtime = BenchmarkGenerationRuntime(
        out_dir=tmp_path,
        bench_log_path=tmp_path / "benchmark.log",
        state_path=tmp_path / "generation_state.json",
        requested_cases=1,
        ranked_candidates=[52],
        candidate_search_pool=[52],
        model_name="model",
        model_source=str(full),
        hef_full_policy="skip",
        hailo_feasibility_state=state,
    )
    cfg = BenchmarkGenerationExecutionConfig(
        runtime=runtime,
        target_cases=1,
        gap=0,
        ranked_candidates=[52],
        candidate_search_pool=[52],
        out_dir=tmp_path,
        base="model",
        pad=3,
        strict_boundary=False,
        model=object(),
        nodes=[],
        order=[],
        analysis_payload={},
        full_model_src=str(full),
        full_model_dst=str(full),
        hef_targets=["hailo8", "hailo10h"],
        hef_part1=True,
        hef_part2=False,
        hef_backend="venv",
        hailo_feasibility_control=control,
    )
    callbacks = BenchmarkGenerationExecutionCallbacks(
        log=lambda *_args, **_kwargs: None,
        queue_put=lambda *_args, **_kwargs: None,
        persist_state=lambda **_kwargs: (_ for _ in ()).throw(
            OSError("fsync failed")
        ),
        publish_hailo_diagnostics=lambda *_args, **_kwargs: None,
        predicted_metrics_for_boundary=lambda *_args, **_kwargs: {},
        hailo_parse_entry_for_boundary=lambda *_args, **_kwargs: None,
        hailo_parse_scalar_fields=lambda *_args, **_kwargs: {},
    )
    with pytest.raises(OSError, match="fsync failed"):
        BenchmarkGenerationExecutionService().execute_case_build_loop(
            cfg, callbacks
        )


def test_evidence_conflict_always_stops_even_if_exhaustion_stop_is_false(
    tmp_path: Path,
):
    full = tmp_path / "model.onnx"
    full.write_bytes(b"full")
    control = normalize_hailo_feasibility_control(
        _control(stop_workflow_on_exhaustion=False)
    )
    state = _new_hailo_feasibility_state_v2783(
        control=control,
        candidate_order=[52],
        targets=["hailo8", "hailo10h"],
        backend="venv",
        full_source_onnx_sha256=hashlib.sha256(full.read_bytes()).hexdigest(),
    )
    state["outcome"] = "EVIDENCE_CONFLICT"
    state["exhaustion_reason"] = "tampered_evidence"
    runtime = BenchmarkGenerationRuntime(
        out_dir=tmp_path,
        bench_log_path=tmp_path / "benchmark.log",
        state_path=tmp_path / "generation_state.json",
        requested_cases=1,
        ranked_candidates=[52],
        candidate_search_pool=[52],
        model_name="model",
        model_source=str(full),
        hef_full_policy="skip",
        hailo_feasibility_state=state,
    )
    cfg = BenchmarkGenerationExecutionConfig(
        runtime=runtime,
        target_cases=1,
        gap=0,
        ranked_candidates=[52],
        candidate_search_pool=[52],
        out_dir=tmp_path,
        base="model",
        pad=3,
        strict_boundary=False,
        model=object(),
        nodes=[],
        order=[],
        analysis_payload={},
        full_model_src=str(full),
        full_model_dst=str(full),
        hef_targets=["hailo8", "hailo10h"],
        hef_part1=True,
        hef_part2=False,
        hef_backend="venv",
        hailo_feasibility_control=control,
    )
    callbacks = BenchmarkGenerationExecutionCallbacks(
        log=lambda *_args, **_kwargs: None,
        queue_put=lambda *_args, **_kwargs: None,
        persist_state=lambda **_kwargs: None,
        publish_hailo_diagnostics=lambda *_args, **_kwargs: None,
        predicted_metrics_for_boundary=lambda *_args, **_kwargs: {},
        hailo_parse_entry_for_boundary=lambda *_args, **_kwargs: None,
        hailo_parse_scalar_fields=lambda *_args, **_kwargs: {},
    )
    assert BenchmarkGenerationExecutionService().execute_case_build_loop(
        cfg, callbacks
    ) == []
    assert runtime.candidate_search_stop == {
        "reason": "EVIDENCE_CONFLICT",
        "detail": "persisted Hailo8-first Gate-A terminal outcome",
        "boundary": None,
        "fallback_allowed": False,
        "stop_workflow": True,
    }


def test_yolo26_gate_a_orchestration_dispatches_no_full_or_part2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    full = tmp_path / "yolo26m.onnx"
    full.write_bytes(b"full")
    runtime = BenchmarkGenerationRuntime(
        out_dir=tmp_path,
        bench_log_path=tmp_path / "benchmark.log",
        state_path=tmp_path / "generation_state.json",
        requested_cases=1,
        ranked_candidates=[],
        candidate_search_pool=[],
        model_name="yolo26m",
        model_source=str(full),
        hef_full_policy="skip",
    )
    control = _control()
    build_calls = []

    def build(*_args, **kwargs):
        build_calls.append(dict(kwargs))
        raise AssertionError("Gate-A must not dispatch empty-pool builds")

    callbacks = BenchmarkGenerationExecutionCallbacks(
        log=lambda *_args, **_kwargs: None,
        queue_put=lambda *_args, **_kwargs: None,
        persist_state=lambda **_kwargs: None,
        publish_hailo_diagnostics=lambda *_args, **_kwargs: None,
        predicted_metrics_for_boundary=lambda *_args, **_kwargs: {},
        hailo_parse_entry_for_boundary=lambda *_args, **_kwargs: None,
        hailo_parse_scalar_fields=lambda *_args, **_kwargs: {},
    )
    execution = BenchmarkGenerationExecutionConfig(
        runtime=runtime,
        target_cases=1,
        gap=0,
        ranked_candidates=[],
        candidate_search_pool=[],
        out_dir=tmp_path,
        base="yolo26m",
        pad=3,
        strict_boundary=False,
        model=object(),
        nodes=[],
        order=[],
        analysis_payload={},
        bench_plan_runs=[],
        full_model_src=str(full),
        full_model_dst=str(full),
        hef_targets=["hailo8", "hailo10h"],
        hef_part1=True,
        hef_part2=False,
        hef_backend="venv",
        hailo_build_hef_fn=build,
        hailo_feasibility_control=control,
    )
    orchestration = BenchmarkGenerationOrchestrationConfig(
        runtime=runtime,
        execution_cfg=execution,
        execution_callbacks=callbacks,
        target_cases=1,
        preferred_shortlist_original=[],
        ranked_candidates=[],
        candidate_search_pool=[],
        out_dir=tmp_path,
        base="yolo26m",
        pad=3,
        full_model_src=str(full),
        full_model_dst=str(full),
        analysis_payload={},
        analysis_params_payload={},
        system_spec_payload=None,
        bench_log_path=str(runtime.bench_log_path),
        bench_plan_runs=[],
        hef_targets=["hailo8", "hailo10h"],
        hef_full=False,
        hef_part1=True,
        hef_part2=False,
        hef_backend="venv",
        hef_fixup=False,
        hef_opt_level=1,
        hef_calib_dir=None,
        hef_calib_count=5,
        hef_calib_bs=1,
        hef_force=False,
        hef_keep=False,
        hef_wsl_distro=None,
        hef_wsl_venv="",
        hef_timeout_s=30,
        full_hef_policy="skip",
        full_model_preflight_policy="skip",
        hailo_build_hef_fn=build,
        hailo_selected=True,
        write_harness_script=lambda suite, _name: str(
            Path(suite) / "benchmark_suite.py"
        ),
    )
    service = BenchmarkGenerationOrchestrationService()
    monkeypatch.setattr(
        service,
        "_force_yolo26_suite_full_baseline_if_needed",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("YOLO26 Full mutation called")
        ),
    )
    monkeypatch.setattr(
        service,
        "_ensure_yolo26_full_hailo_baseline_plan",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("YOLO26 Full planning called")
        ),
    )
    result = service.run(orchestration)
    assert build_calls == []
    assert result.final_status in {"warn", "partial"}
    assert runtime.hailo_feasibility_state["outcome"] == (
        "CANARY_BUDGET_EXHAUSTED"
    )
