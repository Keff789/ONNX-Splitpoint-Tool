from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import onnx_splitpoint_tool.build_evidence as build_evidence
from onnx_splitpoint_tool.workflow.legacy_benchmarkset_binding import (
    _hailo_feasibility_stop_workflow_v2783,
    _make_hailo_feasibility_evidence_lookup_v2783,
    _read_gate_resume_state_v2783,
)


def test_evidence_conflict_stop_policy_ignores_exhaustion_opt_out():
    control = {"stop_workflow_on_exhaustion": False}
    assert _hailo_feasibility_stop_workflow_v2783(
        "EVIDENCE_CONFLICT", control
    ) is True
    assert _hailo_feasibility_stop_workflow_v2783(
        "CANARY_BUDGET_EXHAUSTED", control
    ) is False


def _decision(*, status="HIT", state="ARTIFACT_PASS", reusable=True):
    return SimpleNamespace(
        status=status,
        state=state,
        reusable=reusable,
        evidence_origin={"kind": "prior_b5"},
        as_dict=lambda: {
            "status": status,
            "state": state,
            "reusable": reusable,
        },
    )


def _install_lookup_fakes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    key_arch="hailo10h",
    decision=None,
    materialized_arch="hailo10h",
    materialize_error: Exception | None = None,
):
    monkeypatch.setattr(
        build_evidence, "load_build_evidence_index", lambda _path: {}
    )
    monkeypatch.setattr(
        build_evidence,
        "boundary_endpoint_contract_sha256",
        lambda **_kwargs: "4" * 64,
    )
    monkeypatch.setattr(
        build_evidence,
        "canonical_build_key_from_hailo_v3_payload",
        lambda *_args, **_kwargs: {"hw_arch": key_arch},
    )
    monkeypatch.setattr(
        build_evidence,
        "lookup_build_evidence",
        lambda *_args, **_kwargs: decision or _decision(),
    )

    def materialize(_decision_value, destination, _key, **_kwargs):
        if materialize_error is not None:
            raise materialize_error
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        hef = destination / "compiled.hef"
        hef.write_bytes(b"verified-hef")
        return {
            "hw_arch": materialized_arch,
            "hef_path": str(hef),
            "cache_key": "a" * 64,
            "details": {
                "exact_build_evidence": {
                    "artifact_sha256": "b" * 64,
                    "verified_after_materialization": True,
                }
            },
        }

    monkeypatch.setattr(
        build_evidence, "materialize_verified_artifact", materialize
    )


def _request(case_dir: Path | str, **updates):
    request = {
        "required_variant": "part1",
        "boundary": 52,
        "target": "hailo10",
        "cache_probe": {
            "cache_key_v3": "a" * 64,
            "cache_payload_v3": {"model_sha256": "c" * 64},
        },
        "evidence_context": {
            "builder_source_onnx_sha256": "2" * 64,
            "full_source_onnx_sha256": "1" * 64,
            "split_manifest": {"boundary": 52},
        },
        "materialization": {"case_dir": str(case_dir)},
    }
    request.update(updates)
    return request


def _lookup(
    monkeypatch: pytest.MonkeyPatch,
    suite_root: Path,
    **fake_updates,
):
    _install_lookup_fakes(monkeypatch, **fake_updates)
    callback = _make_hailo_feasibility_evidence_lookup_v2783(
        {
            "evidence_index_path": "/read-only/index.json",
            "evidence_artifact_root": "/read-only/run",
        },
        suite_root=suite_root,
    )
    assert callback is not None
    return callback


def test_materialized_hailo10h_key_preserves_hailo10_profile_axis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    case = tmp_path / "b052"
    case.mkdir()
    result = _lookup(monkeypatch, tmp_path)(_request(case))
    assert result["outcome"] == "ARTIFACT_PASS"
    assert result["target_outcome"]["hw_arch"] == "hailo10"
    assert result["target_outcome"]["target_output"]["part1"] == (
        "hailo/hailo10/part1/compiled.hef"
    )


def test_exact_conflict_is_not_collapsed_to_miss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    case = tmp_path / "b052"
    case.mkdir()
    callback = _lookup(
        monkeypatch,
        tmp_path,
        decision=_decision(status="CONFLICT", state=None, reusable=False),
    )
    result = callback(_request(case))
    assert result["exact"] is True
    assert result["outcome"] == "EVIDENCE_CONFLICT"


@pytest.mark.parametrize(
    "updates",
    [
        {"required_variant": "part2"},
        {"boundary": 53},
        {"evidence_context": {"split_manifest": {"boundary": 51}}},
    ],
)
def test_variant_and_boundary_mismatch_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, updates
):
    case = tmp_path / "b052"
    case.mkdir()
    result = _lookup(monkeypatch, tmp_path)(_request(case, **updates))
    assert result["outcome"] == "EVIDENCE_CONFLICT"


def test_canonical_target_mismatch_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    case = tmp_path / "b052"
    case.mkdir()
    result = _lookup(monkeypatch, tmp_path, key_arch="hailo8")(_request(case))
    assert result["outcome"] == "EVIDENCE_CONFLICT"


@pytest.mark.parametrize("raw_case", ["", "b052"])
def test_empty_or_relative_case_dir_is_rejected(
    raw_case: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    result = _lookup(monkeypatch, tmp_path)(_request(raw_case))
    assert result["outcome"] == "EVIDENCE_CONFLICT"


def test_symlink_component_case_dir_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    real = tmp_path / "real"
    (real / "b052").mkdir(parents=True)
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(Exception, match="symlink|unsafe"):
        _lookup(monkeypatch, link)(_request(link / "b052"))


def test_absolute_b52_outside_active_suite_is_conflict_without_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    suite = tmp_path / "active-suite"
    outside = tmp_path / "outside" / "b052"
    suite.mkdir()
    outside.mkdir(parents=True)
    result = _lookup(monkeypatch, suite)(_request(outside))
    assert result["exact"] is True
    assert result["outcome"] == "EVIDENCE_CONFLICT"
    assert result["reason"] == "materialization_case_outside_active_suite"
    assert not (outside / "hailo").exists()


def test_materialized_target_mismatch_and_tamper_exception_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    case = tmp_path / "b052"
    case.mkdir()
    with pytest.raises(ValueError, match="target mismatch"):
        _lookup(monkeypatch, tmp_path, materialized_arch="hailo8")(_request(case))

    monkeypatch.undo()
    with pytest.raises(RuntimeError, match="tamper"):
        _lookup(
            monkeypatch,
            tmp_path,
            materialize_error=RuntimeError("tamper detected"),
        )(_request(case))


def test_gate_resume_loader_is_strict_nofollow_mapping(tmp_path: Path):
    valid = tmp_path / "generation_state.json"
    valid.write_text(
        '{"hailo_feasibility_state":{"outcome":"RUNNING"}}',
        encoding="utf-8",
    )
    assert _read_gate_resume_state_v2783(valid)[
        "hailo_feasibility_state"
    ]["outcome"] == "RUNNING"

    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{", encoding="utf-8")
    with pytest.raises(Exception, match="invalid_json"):
        _read_gate_resume_state_v2783(corrupt)

    scalar = tmp_path / "scalar.json"
    scalar.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        _read_gate_resume_state_v2783(scalar)

    alias = tmp_path / "alias.json"
    alias.symlink_to(valid)
    with pytest.raises(Exception, match="symlink|regular"):
        _read_gate_resume_state_v2783(alias)
