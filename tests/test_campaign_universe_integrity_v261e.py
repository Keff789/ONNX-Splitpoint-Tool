from __future__ import annotations

import copy
from pathlib import Path

from onnx_splitpoint_tool.campaign import (
    _verify_prediction_freeze_manifest,
    create_candidate_universe_manifest,
    deterministic_audit_cases,
)
from onnx_splitpoint_tool.workflow.artifacts import read_json, sha256_file, write_csv, write_json
from onnx_splitpoint_tool.workflow.scientific_reporting import (
    _load_prediction_freeze,
    _load_suite_prediction,
)


def _candidate(case_id: str, boundary: int) -> dict[str, object]:
    """A tied static candidate carrying deliberately mutable predictor fields."""
    return {
        "case_id": case_id,
        "boundary": boundary,
        "cut_bytes": 4096,
        "cut_mib_val": 0.00390625,
        "imbalance_val": 0.5,
        "n_cut_tensors": 2,
        "strict_ok": True,
        "rank": boundary,
        "score_pred": float(boundary),
        "predicted_total_latency_ms": 10.0 + boundary,
    }


def test_audit_selection_and_universe_are_invariant_to_ties_and_prediction_order(tmp_path: Path) -> None:
    candidates = [
        _candidate("b030", 30),
        _candidate("b010", 10),
        _candidate("b020", 20),
        _candidate("b040", 40),
    ]
    reordered = list(reversed(copy.deepcopy(candidates)))
    for rank, row in enumerate(reordered, start=1):
        # Simulate a different predictor ordering and inverted predictions.  None
        # of these fields may influence a prospective static audit selection.
        row["rank"] = rank
        row["score_pred"] = -float(row["score_pred"])
        row["predicted_total_latency_ms"] = 100.0 - rank

    selected_first = deterministic_audit_cases(candidates, model_id="model-a", size=3, seed=17)
    selected_second = deterministic_audit_cases(reordered, model_id="model-a", size=3, seed=17)
    assert selected_first == selected_second

    common = {
        "model_id": "model-a",
        "mode": "deterministic_audit",
        "audit_size": 3,
        "minimum_valid_candidates": 1,
        "seed": 17,
        # The universe self-hash intentionally remains bound to prediction
        # provenance.  Both calls therefore describe the same frozen source.
        "source_prediction_sha256": "sha256:" + "a" * 64,
        "identity_context": {"model_sha256": "sha256:" + "b" * 64},
        "write_artifacts": False,
    }
    _, _, first = create_candidate_universe_manifest(
        candidates=candidates,
        output_dir=tmp_path / "first",
        **common,
    )
    _, _, second = create_candidate_universe_manifest(
        candidates=reordered,
        output_dir=tmp_path / "second",
        **common,
    )

    assert first["selected_case_ids"] == second["selected_case_ids"]
    assert first["selected_candidate_ids"] == second["selected_candidate_ids"]
    assert first["feasible_candidate_identity_sha256"] == second["feasible_candidate_identity_sha256"]
    assert first["selected_candidate_identity_sha256"] == second["selected_candidate_identity_sha256"]
    assert first["candidates"] == second["candidates"]
    assert first["universe_sha256"] == second["universe_sha256"]

    _, _, changed_provenance = create_candidate_universe_manifest(
        candidates=reordered,
        output_dir=tmp_path / "changed-provenance",
        **{**common, "source_prediction_sha256": "sha256:" + "c" * 64},
    )
    assert first["selected_case_ids"] == changed_provenance["selected_case_ids"]
    assert first["selected_candidate_identity_sha256"] == changed_provenance["selected_candidate_identity_sha256"]
    assert first["universe_sha256"] != changed_provenance["universe_sha256"]


def _freeze_fixture(root: Path) -> tuple[Path, Path, dict[str, object]]:
    candidates = [_candidate("b001", 1), _candidate("b002", 2)]
    prediction = write_json(root / "prediction.json", {"model_id": "model-a", "candidates": candidates})
    prediction_csv = write_csv(root / "predictions_frozen.csv", candidates)
    ranking_csv = write_csv(
        root / "ranking_predictions_frozen.csv",
        [{"case_id": "b001", "method_id": "cut_bytes_only", "predicted_rank": 1}],
    )
    universe_json, universe_csv, universe = create_candidate_universe_manifest(
        model_id="model-a",
        candidates=candidates,
        mode="all_feasible",
        output_dir=root,
        source_prediction_sha256=sha256_file(prediction) or "",
    )
    manifest = {
        "schema": "onnx-splitpoint/prediction-freeze-manifest",
        "schema_version": 1,
        "model_id": "model-a",
        "evaluation_role": "holdout",
        "prospective": True,
        "valid_for_holdout": True,
        "freeze_status": "prospective_frozen",
        "prediction_json": prediction.name,
        "prediction_sha256": sha256_file(prediction),
        "prediction_csv": prediction_csv.name,
        "prediction_csv_sha256": sha256_file(prediction_csv),
        "ranking_prediction_csv": ranking_csv.name,
        "ranking_prediction_csv_sha256": sha256_file(ranking_csv),
        "candidate_universe_manifest": universe_json.name,
        "candidate_universe_sha256": universe["universe_sha256"],
        "candidate_universe_csv": universe_csv.name,
        "candidate_universe_csv_sha256": sha256_file(universe_csv),
    }
    manifest_path = write_json(root / "prediction_freeze_manifest.json", manifest)
    return manifest_path, universe_json, universe


def test_prediction_freeze_verification_rejects_universe_selfhash_tamper(tmp_path: Path) -> None:
    manifest_path, universe_path, universe = _freeze_fixture(tmp_path)
    assert _verify_prediction_freeze_manifest(manifest_path)["ok"] is True

    tampered = copy.deepcopy(universe)
    tampered["selected_case_ids"] = ["b999"]
    # Deliberately retain the old universe_sha256: a verifier must recompute it.
    write_json(universe_path, tampered)

    check = _verify_prediction_freeze_manifest(manifest_path)
    assert check["ok"] is False
    assert any(
        mismatch.get("reason") == "universe_self_sha256_mismatch"
        for mismatch in check["mismatches"]
    )


def test_prediction_freeze_verification_rejects_declared_universe_csv_tamper(tmp_path: Path) -> None:
    manifest_path, _, _ = _freeze_fixture(tmp_path)
    assert _verify_prediction_freeze_manifest(manifest_path)["ok"] is True

    universe_csv = tmp_path / "candidate_universe.csv"
    universe_csv.write_text(universe_csv.read_text(encoding="utf-8") + "tampered\n", encoding="utf-8")

    check = _verify_prediction_freeze_manifest(manifest_path)
    assert check["ok"] is False
    assert any(
        mismatch.get("artifact") == "candidate_universe_csv"
        for mismatch in check["mismatches"]
    )


def _remove_universe_declaration(manifest_path: Path, *, development: bool = False) -> None:
    manifest = dict(read_json(manifest_path, default={}) or {})
    for key in (
        "candidate_universe_manifest",
        "candidate_universe_sha256",
        "candidate_universe_csv",
        "candidate_universe_csv_sha256",
    ):
        manifest.pop(key, None)
    if development:
        manifest["evaluation_role"] = "development"
        manifest["valid_for_holdout"] = False
    write_json(manifest_path, manifest)


def test_evalrun_loader_requires_hash_bound_universe_for_holdout(tmp_path: Path) -> None:
    model_dir = tmp_path / "models" / "model-a"
    analysis_dir = model_dir / "analysis"
    manifest_path, _, _ = _freeze_fixture(analysis_dir)
    prediction_path = analysis_dir / "prediction.json"
    assert _load_prediction_freeze(model_dir, prediction_path)["valid"] is True

    _remove_universe_declaration(manifest_path)
    freeze = _load_prediction_freeze(model_dir, prediction_path)
    assert freeze["valid"] is False
    assert freeze["candidate_universe_required"] is True
    assert freeze["candidate_universe_valid"] is False
    assert freeze["status"] == "candidate_universe_required_missing"


def test_benchmarkset_loader_requires_hash_bound_universe_for_holdout(tmp_path: Path) -> None:
    manifest_path, _, _ = _freeze_fixture(tmp_path)
    plan = {
        "model_id": "model-a",
        "ranking_validation": {"require_complete_candidate_universe": True},
        "model_suite": {
            "primary": [{"id": "model-a", "evaluation_role": "holdout"}],
            "reserve": [],
        },
    }
    rows = [{"model_id": "model-a", "evaluation_role": "holdout"}]
    loaded = _load_suite_prediction(tmp_path, plan, rows)["model-a"]["_prediction_freeze"]
    assert loaded["valid"] is True

    _remove_universe_declaration(manifest_path)
    loaded = _load_suite_prediction(tmp_path, plan, rows)["model-a"]["_prediction_freeze"]
    assert loaded["valid"] is False
    assert loaded["candidate_universe_required"] is True
    assert loaded["candidate_universe_valid"] is False
    assert loaded["status"] == "candidate_universe_required_missing"


def test_evalrun_loader_preserves_development_legacy_without_universe_claim(tmp_path: Path) -> None:
    model_dir = tmp_path / "models" / "model-a"
    analysis_dir = model_dir / "analysis"
    manifest_path, _, _ = _freeze_fixture(analysis_dir)
    _remove_universe_declaration(manifest_path, development=True)

    freeze = _load_prediction_freeze(model_dir, analysis_dir / "prediction.json")
    assert freeze["valid"] is True
    assert freeze["candidate_universe_required"] is False
    assert freeze["candidate_universe_valid"] is False
