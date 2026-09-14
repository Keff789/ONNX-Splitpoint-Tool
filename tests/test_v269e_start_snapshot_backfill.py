from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from onnx_splitpoint_tool.energy.config import EnergyDefaults
from onnx_splitpoint_tool.gui.panels.panel_evaluation_workflow import (
    _commit_profile_summary,
)
from onnx_splitpoint_tool.run_modes import apply_run_mode, default_run_modes_config
from onnx_splitpoint_tool.benchmark.evaluation_profiles import load_evaluation_profile
from onnx_splitpoint_tool.workflow.contracts import WorkflowOptions
from onnx_splitpoint_tool.workflow.legacy_benchmarkset_binding import (
    reconcile_candidate_plan_after_generation,
    write_reconciled_candidate_plan_mirrors,
)
from onnx_splitpoint_tool.workflow.runner import (
    EvaluationWorkflowRunner,
    _native_energy_duration_for_execution_v269e,
    _window_probe_duration_for_execution_v269e,
)
from onnx_splitpoint_tool.workflow.start_snapshot import (
    StartSnapshotConsistencyError,
    build_profile_start_snapshot,
    public_start_snapshot_metadata,
    resolve_runtime_profile_start_snapshot,
    start_snapshot_matches_preview,
    validate_profile_start_snapshot,
)
from onnx_splitpoint_tool.workflow.hardware_matrix import normalize_hardware_targets
from onnx_splitpoint_tool.workflow.artifacts import sha256_json
from onnx_splitpoint_tool.execution_plan import build_effective_execution_plan
from onnx_splitpoint_tool.workflow.run_control import (
    WorkflowRunTargetError,
    build_resume_contract,
)


def test_failed_summary_render_clears_visible_start_snapshot() -> None:
    class FailingSummaryWidget:
        text = "Part-2 inputs=1=off"

        def configure(self, **_kwargs) -> None:
            return None

        def delete(self, *_args) -> None:
            raise RuntimeError("simulated Tcl render failure")

        def insert(self, *_args) -> None:
            self.text = str(_args[-1])

    app = SimpleNamespace(
        _evaluation_workflow_visible_start_snapshot={"snapshot_sha256": "old"}
    )
    widget = FailingSummaryWidget()

    assert _commit_profile_summary(
        app,
        widget,
        "Part-2 inputs=1=on",
        {
            "profile_request": "profile",
            "snapshot_sha256": "new",
        },
    ) is False
    assert widget.text == "Part-2 inputs=1=off"
    assert app._evaluation_workflow_visible_start_snapshot == {}


def _source_profile(*, native: bool = True, energy: bool = True) -> dict:
    config = default_run_modes_config()
    return {
        "name": "requested_smoke_profile",
        "selection_policy": {
            "max_accepted_cases_per_model": 1,
            "preferred_shortlist": 10,
            "selection_strategy": "stratified_windows",
            "min_gap": 1,
            "candidate_search_pool": "auto",
        },
        "model_suite": {
            "primary": [
                {"id": "resnet50", "task": "classification", "enabled": True},
                {"id": "yolo26s", "task": "detection", "enabled": True},
            ]
        },
        "run_profiles": [
            {"id": "hailo8", "enabled": True},
            {"id": "hailo8_to_trt", "enabled": True},
        ],
        "execution_preset": {
            "id": "smoke",
            "follow_tool_config": False,
            "snapshot": copy.deepcopy(config["modes"]["smoke"]),
            "overrides": {
                "native_enabled": native,
                "energy_enabled": energy,
            },
        },
    }


def _start_snapshot(source: dict) -> dict:
    resolved, _ = apply_run_mode(
        source,
        config=default_run_modes_config(),
        follow_tool_config=False,
    )
    _, snapshot = resolve_runtime_profile_start_snapshot(
        profile_request="requested_smoke_profile",
        source_profile=source,
        resolved_profile=resolved,
        profile_id="requested_smoke_profile",
        profile_path="/tmp/requested_smoke_profile.yaml",
        profile_source="file",
    )
    return snapshot


def test_start_snapshot_binds_request_materialization_and_execution_plan() -> None:
    snapshot = _start_snapshot(_source_profile(native=True, energy=True))

    assert snapshot["requested_selection"]["native_enabled"] is True
    assert snapshot["resolved_selection"]["energy_enabled"] is True
    assert snapshot["effective_execution_plan"]["native_enabled"] is True
    assert snapshot["effective_execution_plan"]["native_energy_enabled"] is True
    assert validate_profile_start_snapshot(snapshot)["consistency"]["status"] == "ok"


def test_native_energy_duration_is_frozen_before_start_and_not_reread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "onnx_splitpoint_tool.energy.config.load_energy_defaults",
        lambda: EnergyDefaults(),
    )
    snapshot = _start_snapshot(_source_profile(native=True, energy=True))
    resolved = snapshot["resolved_profile"]
    native_cfg = resolved["native_producers"]
    energy_cfg = native_cfg["energy"]
    probe_cfg = energy_cfg["window_method_validation_probe"]
    frozen_duration = float(energy_cfg["duration_s"])
    assert probe_cfg["max_acquisition_retries"] == 1
    assert probe_cfg["collector_invalid_repeat_retries"] == 0
    assert probe_cfg["acquisition_retry_backoff_s"] == 5.0
    assert snapshot["runtime_bindings"]["window_method_validation_probe"] == probe_cfg

    def _mutable_defaults_must_not_be_read():
        raise AssertionError(
            "EnergyDefaults was reread after the immutable start snapshot"
        )

    monkeypatch.setattr(
        "onnx_splitpoint_tool.energy.config.load_energy_defaults",
        _mutable_defaults_must_not_be_read,
    )
    assert _native_energy_duration_for_execution_v269e(
        snapshot, native_cfg, energy_cfg
    ) == frozen_duration
    assert _window_probe_duration_for_execution_v269e(
        snapshot, native_cfg, energy_cfg, probe_cfg
    ) == frozen_duration
    assert (
        snapshot["runtime_bindings"]["native_energy_duration_s"]
        == frozen_duration
    )


def test_start_snapshot_blocks_requested_override_changed_by_resolution() -> None:
    source = _source_profile(native=True, energy=True)
    resolved, _ = apply_run_mode(
        source,
        config=default_run_modes_config(),
        follow_tool_config=False,
    )
    resolved["execution_preset"]["overrides"]["native_enabled"] = False
    resolved["execution_preset"]["overrides"]["energy_enabled"] = False
    resolved["native_producers"]["enabled"] = False
    resolved["native_producers"]["energy"]["enabled"] = False
    resolved["energy"]["requested_native_energy"] = False

    with pytest.raises(StartSnapshotConsistencyError, match="requested/visible"):
        build_profile_start_snapshot(
            profile_request="requested_smoke_profile",
            source_profile=source,
            resolved_profile=resolved,
            profile_id="requested_smoke_profile",
            profile_path="/tmp/requested_smoke_profile.yaml",
            profile_source="file",
        )


def test_start_snapshot_blocks_internal_materialization_drift() -> None:
    source = _source_profile(native=True, energy=True)
    resolved, _ = apply_run_mode(
        source,
        config=default_run_modes_config(),
        follow_tool_config=False,
    )
    resolved["native_producers"]["enabled"] = False

    with pytest.raises(StartSnapshotConsistencyError, match="materialized.native_producers.enabled"):
        build_profile_start_snapshot(
            profile_request="requested_smoke_profile",
            source_profile=source,
            resolved_profile=resolved,
            profile_id="requested_smoke_profile",
            profile_path="/tmp/requested_smoke_profile.yaml",
            profile_source="file",
        )


def test_worker_uses_exact_gui_snapshot_and_manifest_options_are_redacted(tmp_path: Path) -> None:
    snapshot = _start_snapshot(_source_profile(native=True, energy=True))
    opts = WorkflowOptions(
        profile="profile_that_may_change_after_queue.yaml",
        out=str(tmp_path),
        profile_start_snapshot=snapshot,
    )
    runner = EvaluationWorkflowRunner(opts)
    runner._load_profile()

    assert runner.profile_payload["name"] == "requested_smoke_profile"
    assert runner.profile_payload["execution_preset"]["overrides"] == {
        "native_enabled": True,
        "energy_enabled": True,
    }
    options_dict = opts.to_dict()
    options_record = options_dict["profile_start_snapshot"]
    assert options_record["snapshot_sha256"] == snapshot["snapshot_sha256"]
    assert options_record["resolved_profile"]["name"] == "requested_smoke_profile"
    assert WorkflowOptions(**options_dict).profile_start_snapshot["snapshot_sha256"] == snapshot["snapshot_sha256"]


def test_modified_in_memory_start_snapshot_is_rejected() -> None:
    snapshot = _start_snapshot(_source_profile(native=True, energy=True))
    snapshot["resolved_profile"]["execution_preset"]["overrides"]["native_enabled"] = False

    with pytest.raises(StartSnapshotConsistencyError, match="hash mismatch"):
        validate_profile_start_snapshot(snapshot)


def test_legacy_profile_without_execution_preset_still_loads() -> None:
    loaded = load_evaluation_profile("smoke_regression_v1", validate=True)
    assert loaded is not None
    assert loaded.start_snapshot["consistency"]["status"] == "ok"
    assert loaded.start_snapshot["resolved_selection"]["run_mode"] == ""


def test_legacy_native_profile_without_execution_preset_preserves_native_intent() -> None:
    loaded = load_evaluation_profile("native_resnet_yolo26s_hailo8_smoke_v1", validate=True)
    runtime, snapshot = resolve_runtime_profile_start_snapshot(
        profile_request="native_resnet_yolo26s_hailo8_smoke_v1",
        source_profile=loaded.source_profile,
        resolved_profile=loaded.raw_profile,
        profile_id=loaded.profile_id,
        profile_path=loaded.profile_path,
        profile_source=loaded.source,
    )
    assert runtime["native_producers"]["enabled"] is True
    assert snapshot["resolved_selection"]["native_enabled"] is True
    assert snapshot["resolved_selection"]["energy_enabled"] is True


def test_preview_match_catches_nonplan_profile_and_deep_mode_drift() -> None:
    first = _start_snapshot(_source_profile(native=True, energy=True))
    source_changed = _source_profile(native=True, energy=True)
    source_changed["model_suite"]["primary"][0]["path"] = "/different/resnet50.onnx"
    second = _start_snapshot(source_changed)
    preview = {
        key: first[key]
        for key in (
            "profile_request",
            "source_profile_sha256",
            "resolved_execution_sha256",
            "selection_fingerprint",
        )
    }
    assert not start_snapshot_matches_preview(preview, second, profile_request="requested_smoke_profile")

    deep_changed = copy.deepcopy(first)
    deep_changed["resolved_profile"]["execution_preset"]["snapshot"]["reporting"]["cleanup_legacy"] = not bool(
        deep_changed["resolved_profile"]["execution_preset"]["snapshot"]["reporting"].get("cleanup_legacy")
    )
    deep_changed["resolved_execution_sha256"] = "sha256:" + "0" * 64
    assert not start_snapshot_matches_preview(preview, deep_changed, profile_request="requested_smoke_profile")
    assert not start_snapshot_matches_preview({}, first, profile_request="requested_smoke_profile")


def test_hardware_mapping_is_frozen_and_hash_checked(tmp_path: Path) -> None:
    registry = tmp_path / "hardware.yaml"
    registry.write_text(
        "hardware_setups:\n"
        "  - id: h8\n"
        "    enabled: true\n"
        "    accelerator: hailo8\n"
        "    host: {address: 192.0.2.10, user: nx}\n"
        "build_environments: []\n"
        ,
        encoding="utf-8",
    )
    source = _source_profile(native=True, energy=True)
    source["hardware"] = {"setups_file": str(registry)}
    resolved, _ = apply_run_mode(
        source,
        config=default_run_modes_config(),
        follow_tool_config=False,
    )
    resolved["hardware"] = {"setups_file": str(registry)}
    runtime, _snapshot = resolve_runtime_profile_start_snapshot(
        profile_request="requested_smoke_profile",
        source_profile=source,
        resolved_profile=resolved,
        profile_id="requested_smoke_profile",
        profile_path=str(tmp_path / "profile.yaml"),
        profile_source="file",
    )
    assert normalize_hardware_targets(runtime)[0]["runtime"]["host"] == "192.0.2.10"
    registry.write_text(registry.read_text(encoding="utf-8").replace("192.0.2.10", "192.0.2.99"), encoding="utf-8")
    assert normalize_hardware_targets(runtime)[0]["runtime"]["host"] == "192.0.2.10"
    runtime["hardware"]["resolved_targets"][0]["runtime"]["host"] = "192.0.2.77"
    with pytest.raises(ValueError, match="hash mismatch"):
        normalize_hardware_targets(runtime)


def _write_resume_fixture(
    root: Path,
    *,
    creation_snapshot: dict,
    options: WorkflowOptions,
) -> tuple[Path, Path]:
    run_dir = root / str(options.run_id or "existing_run")
    run_dir.mkdir(parents=True)
    profile_path = run_dir / "profile.yaml"
    profile_path.write_text(
        yaml.safe_dump(
            creation_snapshot["resolved_profile"],
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    (run_dir / "profile_source.yaml").write_text(
        yaml.safe_dump(
            creation_snapshot["source_profile"],
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    snapshot_meta = public_start_snapshot_metadata(creation_snapshot)
    (run_dir / "profile_start_snapshot.json").write_text(
        json.dumps(snapshot_meta, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (run_dir / "run_manifest.json").write_text(
        json.dumps({
            "schema": "onnx-splitpoint/evaluation-run-manifest",
            "schema_version": 1,
            "run_id": run_dir.name,
            "profile_id": "requested_smoke_profile",
            "status": "partial",
            "profile_hash": sha256_json(creation_snapshot["resolved_profile"]),
            "profile_start_snapshot": snapshot_meta,
            "resume_contract": build_resume_contract(
                profile_payload=creation_snapshot["resolved_profile"],
                effective_execution_plan=build_effective_execution_plan(
                    creation_snapshot["resolved_profile"]
                ),
                options=options,
            ),
        }),
        encoding="utf-8",
    )
    return run_dir, profile_path


def test_resume_blocks_snapshot_drift_before_logs_or_profile_write(tmp_path: Path) -> None:
    creation = _start_snapshot(_source_profile(native=True, energy=True))
    current = _start_snapshot(_source_profile(native=False, energy=False))
    options = WorkflowOptions(
        profile="requested_smoke_profile",
        out=str(tmp_path),
        run_id="existing_run",
        resume=True,
        profile_start_snapshot=current,
    )
    run_dir, profile_path = _write_resume_fixture(
        tmp_path,
        creation_snapshot=creation,
        options=options,
    )
    before = profile_path.read_bytes()
    runner = EvaluationWorkflowRunner(options)

    def _load_current() -> None:
        runner.profile_start_snapshot = current
        runner.profile_payload = copy.deepcopy(current["resolved_profile"])
        runner.profile_id = "requested_smoke_profile"

    runner._load_profile = _load_current
    runner._materialize_window_method_probe_profile = lambda: None
    runner._init_run_logs = lambda: pytest.fail("resume drift reached log initialization")
    runner._copy_profile = lambda: pytest.fail("resume drift reached profile copy")

    with pytest.raises(WorkflowRunTargetError, match="contract mismatch"):
        runner.run()
    assert profile_path.read_bytes() == before
    assert not (run_dir / "evaluation_workflow.log").exists()


def test_resume_accepts_identical_snapshot_and_preserves_creation_profile(tmp_path: Path) -> None:
    snapshot = _start_snapshot(_source_profile(native=True, energy=True))
    options = WorkflowOptions(
        profile="requested_smoke_profile",
        out=str(tmp_path),
        run_id="existing_run",
        resume=True,
        profile_start_snapshot=snapshot,
    )
    run_dir, profile_path = _write_resume_fixture(
        tmp_path,
        creation_snapshot=snapshot,
        options=options,
    )
    before = profile_path.read_bytes()
    runner = EvaluationWorkflowRunner(options)
    runner.profile_start_snapshot = snapshot
    runner.profile_payload = copy.deepcopy(snapshot["resolved_profile"])
    runner.profile_id = "requested_smoke_profile"
    runner._open_run_dir()
    runner._validate_resume_profile_snapshot()
    runner.artifact_index = {"schema": "onnx-splitpoint/artifact-index", "artifacts": []}
    runner._copy_profile()

    assert profile_path.read_bytes() == before
    assert runner.outputs["profile_yaml"] == str(profile_path)


@pytest.mark.parametrize(
    ("artifact_name", "corrupt_payload"),
    [
        ("profile.yaml", "not: [valid\n"),
        ("profile_source.yaml", "corrupted-source-profile\n"),
        ("profile_start_snapshot.json", "{not-json\n"),
    ],
)
def test_resume_blocks_corrupted_creation_artifact(
    tmp_path: Path,
    artifact_name: str,
    corrupt_payload: str,
) -> None:
    snapshot = _start_snapshot(_source_profile(native=True, energy=True))
    options = WorkflowOptions(
        profile="requested_smoke_profile",
        out=str(tmp_path),
        run_id="existing_run",
        resume=True,
        profile_start_snapshot=snapshot,
    )
    run_dir, _profile_path = _write_resume_fixture(
        tmp_path,
        creation_snapshot=snapshot,
        options=options,
    )
    (run_dir / artifact_name).write_text(corrupt_payload, encoding="utf-8")
    runner = EvaluationWorkflowRunner(options)
    runner.profile_start_snapshot = snapshot
    runner.profile_payload = copy.deepcopy(snapshot["resolved_profile"])
    runner.profile_id = "requested_smoke_profile"
    runner._open_run_dir()

    with pytest.raises(ValueError, match="Resume blocked"):
        runner._validate_resume_profile_snapshot()


def test_legacy_resume_is_rejected_before_archived_profile_mutation(tmp_path: Path) -> None:
    snapshot = _start_snapshot(_source_profile(native=True, energy=True))
    run_dir = tmp_path / "legacy_run"
    run_dir.mkdir()
    profile_path = run_dir / "profile.yaml"
    profile_path.write_text("name: archived-legacy\n", encoding="utf-8")
    (run_dir / "run_manifest.json").write_text(
        json.dumps({
            "schema": "onnx-splitpoint/evaluation-run-manifest",
            "schema_version": 1,
            "profile_id": "archived_legacy",
            "profile_hash": "sha256:" + "0" * 64,
        }),
        encoding="utf-8",
    )
    options = WorkflowOptions(
        profile="requested_smoke_profile",
        out=str(tmp_path),
        run_id=run_dir.name,
        resume=True,
        profile_start_snapshot=snapshot,
    )
    runner = EvaluationWorkflowRunner(options)
    runner.profile_start_snapshot = snapshot
    runner.profile_payload = copy.deepcopy(snapshot["resolved_profile"])
    runner.profile_id = "requested_smoke_profile"
    with pytest.raises(WorkflowRunTargetError, match="not structurally complete"):
        runner._open_run_dir()
    assert profile_path.read_text(encoding="utf-8") == "name: archived-legacy\n"


def test_legacy_resume_loads_verified_archive_when_source_profile_is_gone(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "legacy_archived_run"
    run_dir.mkdir()
    archived_profile = {
        "name": "archived_legacy",
        "model_suite": {"primary": []},
        "run_profiles": [],
        "native_producers": {"enabled": False, "energy": {"enabled": False}},
    }
    (run_dir / "profile.yaml").write_text(
        yaml.safe_dump(archived_profile, sort_keys=False),
        encoding="utf-8",
    )
    (run_dir / "run_manifest.json").write_text(
        json.dumps({
            "schema": "onnx-splitpoint/evaluation-run-manifest",
            "schema_version": 1,
            "profile_id": "archived_legacy",
            "profile_hash": sha256_json(archived_profile),
        }),
        encoding="utf-8",
    )
    missing_source = tmp_path / "deleted_original_profile.yaml"
    options = WorkflowOptions(
        profile=str(missing_source),
        out=str(tmp_path),
        run_id=run_dir.name,
        resume=True,
    )
    runner = EvaluationWorkflowRunner(options)

    runner._load_profile()
    runner._materialize_window_method_probe_profile()

    assert runner.profile_id == "archived_legacy"
    assert runner.profile_source == "archived_legacy_run"
    assert runner.profile_payload["name"] == "archived_legacy"
    assert runner.profile_start_snapshot["runtime_bindings"][
        "reconstructed_from_legacy_run"
    ] is True
    assert runner.profile_payload["native_producers"]["energy"][
        "window_method_validation_probe"
    ]["enabled"] is False


def _case(case_id: str, *, rank: int) -> dict:
    boundary = int(case_id[1:])
    return {
        "case_id": case_id,
        "boundary": boundary,
        "split_index": boundary,
        "rank": rank,
        "source_rank": rank,
        "origin": "stratified_windows",
        "selection_reason": "pre-generation stratified selection",
    }


def test_generator_backfill_reconciles_authoritative_candidate_plan_and_mirrors(tmp_path: Path) -> None:
    original = {
        "schema": "onnx-splitpoint/final-candidate-plan",
        "schema_version": 3,
        "artifact_id": "candidate_plan_yolo26s_before",
        "model_id": "yolo26s",
        "source_prediction_artifact_id": "prediction_yolo26s",
        "requested_cases": 3,
        "selected_candidates": [
            _case("b038", rank=1),
            _case("b142", rank=39),
            _case("b268", rank=231),
        ],
        "excluded_candidates": [
            {**_case("b036", rank=2), "exclude_reason": "outside_stratified_selection"},
        ],
        "policy_backfills": [],
        "policy_promotions": [],
    }
    prediction = {
        "candidates": [
            _case("b038", rank=1),
            _case("b036", rank=2),
            _case("b142", rank=39),
            _case("b268", rank=231),
        ]
    }
    accepted = [
        {"boundary": 38, "folder": "b038"},
        {"boundary": 142, "folder": "b142"},
        {"boundary": 36, "folder": "b036"},
    ]
    rejected = [{
        "boundary": 268,
        "folder": "b268",
        "reason": "hailo_yolo26_static_part1_guard",
        "detail": "detection-tail boundary is infeasible; backfill earlier split",
    }]

    final, trace = reconcile_candidate_plan_after_generation(
        original,
        prediction=prediction,
        accepted_cases=accepted,
        rejected_cases=rejected,
        generation_summary={"backfilled_cases_count": 1},
    )

    assert [row["case_id"] for row in final["selected_candidates"]] == ["b038", "b142", "b036"]
    assert final["pre_generation_artifact_id"] == "candidate_plan_yolo26s_before"
    assert final["artifact_id"] != original["artifact_id"]
    assert len(final["policy_backfills"]) == 1
    assert final["policy_backfills"][0]["case_id"] == "b036"
    assert final["policy_backfills"][0]["replaced_case_id"] == "b268"
    assert any(
        row["case_id"] == "b268"
        and row["generation_status"] == "rejected_by_benchmark_generator"
        for row in final["excluded_candidates"]
    )
    assert not any(row.get("case_id") == "b036" for row in final["excluded_candidates"])
    assert trace["pre_generation_case_ids"] == ["b038", "b142", "b268"]
    assert trace["final_case_ids"] == ["b038", "b142", "b036"]
    assert trace["backfill_count"] == 1

    paths = write_reconciled_candidate_plan_mirrors(tmp_path / "models" / "yolo26s", final)
    analysis = json.loads(paths["final_candidate_plan_json"].read_text(encoding="utf-8"))
    benchmark = json.loads(paths["benchmark_final_candidate_plan_json"].read_text(encoding="utf-8"))
    assert analysis == benchmark == final


def test_candidate_plan_is_unchanged_when_generator_matches_selection() -> None:
    original = {
        "artifact_id": "same",
        "selected_candidates": [_case("b038", rank=1)],
        "policy_backfills": [],
    }
    final, trace = reconcile_candidate_plan_after_generation(
        original,
        prediction={"candidates": [_case("b038", rank=1)]},
        accepted_cases=[{"boundary": 38, "folder": "b038"}],
        rejected_cases=[],
        generation_summary={"backfilled_cases_count": 0},
    )
    assert final == original
    assert trace["changed"] is False


def test_zero_accepted_cases_preserve_plan_for_direct_fallback() -> None:
    original = {
        "artifact_id": "candidate-plan-for-fallback",
        "selected_candidates": [_case("b038", rank=1)],
        "policy_backfills": [],
    }
    final, trace = reconcile_candidate_plan_after_generation(
        original,
        prediction={"candidates": [_case("b038", rank=1)]},
        accepted_cases=[],
        rejected_cases=[{"folder": "b038", "boundary": 38, "reason": "hailo_build_failed"}],
        generation_summary={"backfilled_cases_count": 0},
    )
    assert final == original
    assert trace["status"] == "no_accepted_cases_pending_direct_fallback"
    assert final["selected_candidates"][0]["case_id"] == "b038"


def test_generator_reconciliation_rejects_case_boundary_contradiction() -> None:
    original = {
        "requested_cases": 1,
        "selected_candidates": [_case("b268", rank=1)],
    }
    with pytest.raises(ValueError, match="identity is contradictory"):
        reconcile_candidate_plan_after_generation(
            original,
            prediction={"candidates": [_case("b036", rank=2)]},
            accepted_cases=[{"folder": "b036", "boundary": 268}],
            rejected_cases=[],
        )


def test_generator_reconciliation_rejects_duplicate_accepted_identity() -> None:
    original = {
        "requested_cases": 2,
        "selected_candidates": [
            _case("b038", rank=1),
            _case("b142", rank=2),
        ],
    }
    with pytest.raises(ValueError, match="duplicate accepted"):
        reconcile_candidate_plan_after_generation(
            original,
            prediction={"candidates": original["selected_candidates"]},
            accepted_cases=[
                {"folder": "b038", "boundary": 38},
                {"folder": "b038", "boundary": 38},
            ],
            rejected_cases=[],
        )


def test_generator_reconciliation_rejects_backfill_outside_prediction_pool() -> None:
    original = {
        "requested_cases": 1,
        "selected_candidates": [_case("b038", rank=1)],
    }
    with pytest.raises(ValueError, match="outside prediction.candidates"):
        reconcile_candidate_plan_after_generation(
            original,
            prediction={"candidates": [_case("b038", rank=1)]},
            accepted_cases=[{"folder": "b999", "boundary": 999}],
            rejected_cases=[{"folder": "b038", "boundary": 38}],
        )


def test_generator_reconciliation_rejects_more_cases_than_requested() -> None:
    original = {
        "requested_cases": 1,
        "selected_candidates": [_case("b038", rank=1)],
    }
    with pytest.raises(ValueError, match="more accepted cases"):
        reconcile_candidate_plan_after_generation(
            original,
            prediction={
                "candidates": [
                    _case("b038", rank=1),
                    _case("b036", rank=2),
                ]
            },
            accepted_cases=[
                {"folder": "b038", "boundary": 38},
                {"folder": "b036", "boundary": 36},
            ],
            rejected_cases=[],
        )


def test_multiple_backfills_do_not_invent_positional_replacement_pairs() -> None:
    original = {
        "artifact_id": "before",
        "requested_cases": 2,
        "selected_candidates": [
            _case("b038", rank=1),
            _case("b142", rank=2),
        ],
        "policy_backfills": [],
    }
    prediction = {
        "candidates": [
            *original["selected_candidates"],
            _case("b036", rank=3),
            _case("b040", rank=4),
        ]
    }
    final, _trace = reconcile_candidate_plan_after_generation(
        original,
        prediction=prediction,
        accepted_cases=[
            {"folder": "b036", "boundary": 36},
            {"folder": "b040", "boundary": 40},
        ],
        rejected_cases=[
            {"folder": "b038", "boundary": 38},
            {"folder": "b142", "boundary": 142},
        ],
    )
    assert len(final["policy_backfills"]) == 2
    assert all(not row["replaced_case_id"] for row in final["policy_backfills"])
    assert all(
        row["replacement_mapping_status"]
        == "unpaired_multiple_backfill_without_generator_causality"
        for row in final["policy_backfills"]
    )
    assert all(
        row["removed_case_ids"] == ["b038", "b142"]
        for row in final["policy_backfills"]
    )
