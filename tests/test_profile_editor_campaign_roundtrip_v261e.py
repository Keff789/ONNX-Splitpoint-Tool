from __future__ import annotations

import inspect
import importlib
import json
import tkinter as tk
import copy
from pathlib import Path
from types import SimpleNamespace

import jsonschema
import pytest
import yaml

import onnx_splitpoint_tool.gui.profile_editor as profile_editor
from onnx_splitpoint_tool.benchmark.evaluation_profiles import (
    profile_model_entries,
    validate_evaluation_profile_payload,
)
from onnx_splitpoint_tool.campaign import build_campaign_readiness
from onnx_splitpoint_tool.gui.panels.panel_evaluation_workflow import (
    score_independent_audit_start_summary,
)
from onnx_splitpoint_tool.run_modes import (
    apply_run_mode as real_apply_run_mode,
    default_run_modes_config,
)


ROOT = Path(__file__).resolve().parents[1]
FINAL_PROFILE = ROOT / "onnx_splitpoint_tool/resources/evaluation_profiles/thesis_final_campaign_v1.yaml"
DEFAULT_RUN_MODES = ROOT / "onnx_splitpoint_tool/resources/run_modes/default_run_modes.yaml"
PROFILE_SCHEMA = ROOT / "onnx_splitpoint_tool/resources/schemas/evaluation_profile.schema.json"


def _import_gui_app_headless(monkeypatch):
    """Import queue orchestration without changing a headless plot backend."""

    import matplotlib

    monkeypatch.setattr(matplotlib, "use", lambda *_args, **_kwargs: None)
    return importlib.import_module("onnx_splitpoint_tool.gui.app")


class _HeadlessMaster:
    master = None
    _w = "."

    def winfo_toplevel(self):
        return self


class _HeadlessTree:
    def get_children(self):
        return []

    def delete(self, *_args):
        return None

    def cget(self, key):
        assert key == "columns"
        return ("id", "task", "usage", "onnx", "family", "shape", "note")

    def insert(self, *_args, **_kwargs):
        return None

    def selection_set(self, *_args):
        return None

    def see(self, *_args):
        return None


def _headless_editor(monkeypatch):
    """Construct the real editor state machine without Tk widgets or a display."""

    interpreter = tk.Tcl()
    master = _HeadlessMaster()
    master.tk = interpreter

    def _toplevel_init(self, master=None, **_kwargs):
        self.master = master
        self.tk = interpreter
        self._w = "."
        self.children = {}
        self._tclCommands = None

    monkeypatch.setattr(tk.Toplevel, "__init__", _toplevel_init)
    for method in ("title", "geometry", "minsize", "resizable", "state", "attributes", "transient"):
        monkeypatch.setattr(tk.Toplevel, method, lambda self, *_args, **_kwargs: None)
    for method in (
        "_build_ui",
        "_try_load_initial_profile",
        "_update_energy_estimate",
        "_update_run_summary",
        "_install_fallback_tooltips",
    ):
        monkeypatch.setattr(
            profile_editor.EvaluationProfileEditor,
            method,
            lambda self, *_args, **_kwargs: None,
        )

    config = default_run_modes_config()
    monkeypatch.setattr(profile_editor, "default_registry_path", lambda: Path("/audit/dataset_registry.json"))
    monkeypatch.setattr(profile_editor, "load_run_modes_config", lambda *_args, **_kwargs: config)
    monkeypatch.setattr(profile_editor, "get_run_mode", lambda mode_id, *_args, **_kwargs: config["modes"][mode_id])
    monkeypatch.setattr(profile_editor, "mode_summary", lambda *_args, **_kwargs: "")

    def _apply(profile, **kwargs):
        forwarded = {key: value for key, value in kwargs.items() if key != "config"}
        return real_apply_run_mode(profile, config=config, **forwarded)

    monkeypatch.setattr(profile_editor, "apply_run_mode", _apply)

    editor = profile_editor.EvaluationProfileEditor(master)
    editor.model_tree = _HeadlessTree()
    editor.status_var = tk.StringVar(editor, value="")
    editor._load_dataset_registry_into_profile = lambda **_kwargs: False
    return editor


def test_final_profile_headless_editor_roundtrip_preserves_campaign_contract(monkeypatch) -> None:
    source = yaml.safe_load(FINAL_PROFILE.read_text(encoding="utf-8"))
    editor = _headless_editor(monkeypatch)

    editor._apply_payload(source, path=str(FINAL_PROFILE))
    result = editor._build_payload()

    assert result["measurement_campaign"] == source["measurement_campaign"]
    assert result["campaign"]["claim_scope"] == source["campaign"]["claim_scope"]
    assert result["campaign"]["prediction_freeze_enabled"] is True
    assert result["campaign"]["require_protocol_freeze"] is True
    assert result["campaign"]["protocol_freeze"] == source["campaign"]["protocol_freeze"]
    assert result["ranking_validation"]["audit_size"] == 20
    assert result["ranking_validation"]["minimum_valid_audit_candidates"] == 10
    assert source["quality_gate"]["statistics"]["execution_location"] == "central_management"
    assert source["quality_gate"]["statistics"]["workers"] == 4
    assert result["quality_gate"]["statistics"]["execution_location"] == "central_management"
    assert result["quality_gate"]["statistics"]["workers"] == 4

    by_id = {row["id"]: row for row in result["model_suite"]["primary"]}
    assert list(by_id) == [
        "yolov7_paper",
        "resnet50",
        "yolo26s",
        "regnet_x_1_6gf",
        "yolo26m",
    ]
    assert [by_id[model_id]["evaluation_role"] for model_id in by_id] == [
        "development",
        "development",
        "development",
        "confirmatory_holdout",
        "confirmatory_holdout",
    ]
    assert by_id["regnet_x_1_6gf"]["family_id"] == "regnet"
    assert by_id["regnet_x_1_6gf"]["generalization_scope"] == "model_family_holdout"
    assert by_id["regnet_x_1_6gf"]["candidate_universe"]["audit_size"] == 20
    assert by_id["regnet_x_1_6gf"]["candidate_universe"]["minimum_valid_candidates"] == 10
    assert by_id["yolo26m"]["family_id"] == "yolo26"
    assert by_id["yolo26m"]["generalization_scope"] == "within_family_transfer"

    native = result["native_producers"]
    assert native["backends"] == ["hailo8", "hailo10h", "deepx"]
    expected_full = {
        "hailo8": {"hailo8", "tensorrt"},
        "hailo10h": {"hailo10h", "tensorrt"},
        "deepx": {"deepx", "tensorrt"},
    }
    assert {
        producer: set(backends)
        for producer, backends in native["full_baselines"]["backends_by_producer"].items()
    } == expected_full

    required_energy_checks = {
        "full_system_power_scope",
        "energy_command_window",
        "energy_repetitions",
        "energy_confidence_interval",
        "energy_run_order",
    }
    readiness = build_campaign_readiness(result, profile_path=FINAL_PROFILE)
    statuses = {row["id"]: row["status"] for row in readiness["checks"] if row["id"] in required_energy_checks}
    assert statuses == {check_id: "pass" for check_id in required_energy_checks}


def test_final_profile_headless_editor_roundtrip_keeps_disabled_structured_reserve(monkeypatch) -> None:
    source = yaml.safe_load(FINAL_PROFILE.read_text(encoding="utf-8"))
    editor = _headless_editor(monkeypatch)

    editor._apply_payload(source, path=str(FINAL_PROFILE))
    result = editor._build_payload()

    assert result["model_suite"]["reserve"] == source["model_suite"]["reserve"]
    reserve = result["model_suite"]["reserve"][0]
    assert reserve["id"] == "yolo26xl"
    assert reserve["enabled"] is False
    assert reserve["family_id"] == "yolo26"
    assert reserve["generalization_scope"] == "stress_test"
    assert "yolo26xl" not in {
        row["id"] for row in profile_model_entries(result, include_reserve=True)
    }


def test_profile_editor_maps_legacy_holdout_role_on_load(monkeypatch) -> None:
    source = yaml.safe_load(FINAL_PROFILE.read_text(encoding="utf-8"))
    source["model_suite"]["primary"][3]["evaluation_role"] = "holdout"
    editor = _headless_editor(monkeypatch)

    editor._apply_payload(source, path=str(FINAL_PROFILE))
    result = editor._build_payload()

    by_id = {row["id"]: row for row in result["model_suite"]["primary"]}
    assert by_id["regnet_x_1_6gf"]["evaluation_role"] == "confirmatory_holdout"


def test_profile_editor_materializes_matrix_scope_for_legacy_profile(monkeypatch) -> None:
    source = yaml.safe_load(FINAL_PROFILE.read_text(encoding="utf-8"))
    source["campaign"].pop("claim_scope", None)
    editor = _headless_editor(monkeypatch)

    editor._apply_payload(source, path=str(FINAL_PROFILE))
    result = editor._build_payload()

    assert result["campaign"]["claim_scope"] == "evaluated_matrix"


def test_packaged_run_modes_are_identical_to_python_defaults() -> None:
    packaged = yaml.safe_load(DEFAULT_RUN_MODES.read_text(encoding="utf-8"))
    assert packaged == default_run_modes_config()


def test_quality_execution_schema_accepts_final_and_rejects_invalid_values() -> None:
    schema = json.loads(PROFILE_SCHEMA.read_text(encoding="utf-8"))
    source = yaml.safe_load(FINAL_PROFILE.read_text(encoding="utf-8"))
    jsonschema.validate(source, schema)

    invalid_location = copy.deepcopy(source)
    invalid_location["quality_gate"]["statistics"]["execution_location"] = "management_gpu"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(invalid_location, schema)

    invalid_workers = copy.deepcopy(source)
    invalid_workers["quality_gate"]["statistics"]["workers"] = 0
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(invalid_workers, schema)


def test_run_mode_quality_execution_defaults_are_central_for_all_modes() -> None:
    config = default_run_modes_config()
    assert config["modes"]["smoke"]["quality"]["execution_location"] == "central_management"
    assert config["modes"]["smoke"]["quality"]["workers"] == 4
    assert config["modes"]["standard"]["quality"]["execution_location"] == "central_management"
    assert config["modes"]["standard"]["quality"]["workers"] == 4
    assert config["modes"]["final"]["quality"]["execution_location"] == "central_management"
    assert config["modes"]["final"]["quality"]["workers"] == 4

    smoke_profile, _ = real_apply_run_mode({}, mode_id="smoke", config=config)
    standard_profile, _ = real_apply_run_mode({}, mode_id="standard", config=config)
    final_profile, _ = real_apply_run_mode({}, mode_id="final", config=config)
    assert smoke_profile["quality_gate"]["statistics"]["execution_location"] == "central_management"
    assert smoke_profile["quality_gate"]["statistics"]["workers"] == 4
    assert standard_profile["quality_gate"]["statistics"]["execution_location"] == "central_management"
    assert standard_profile["quality_gate"]["statistics"]["workers"] == 4
    assert final_profile["quality_gate"]["statistics"]["execution_location"] == "central_management"
    assert final_profile["quality_gate"]["statistics"]["workers"] == 4


def test_profile_editor_infers_compatible_quality_execution_for_legacy_profiles(monkeypatch) -> None:
    source = yaml.safe_load(FINAL_PROFILE.read_text(encoding="utf-8"))
    source["quality_gate"]["statistics"].pop("execution_location")
    source["quality_gate"]["statistics"].pop("workers")
    source["campaign"]["mode"] = "development"
    source["quality_gate"]["dataset_tier"] = "screening"
    editor = _headless_editor(monkeypatch)
    editor._apply_payload(source, path=str(FINAL_PROFILE))
    result = editor._build_payload()
    assert result["quality_gate"]["statistics"]["execution_location"] == "central_management"
    assert result["quality_gate"]["statistics"]["workers"] == 4


def test_active_simplified_model_editor_only_exposes_requested_fields() -> None:
    source = inspect.getsource(profile_editor.EvaluationProfileEditor._build_models_tab_simple)
    for variable in (
        "var_model_id",
        "var_model_task",
        "var_model_usage",
        "var_model_onnx",
        "var_model_family",
        "var_model_shape",
        "var_model_note",
    ):
        assert variable in source
    for hidden_variable in (
        "var_model_family_id",
        "var_model_generalization_scope",
        "var_model_validation_tier",
        "var_model_universe_mode",
        "var_model_audit_size",
        "var_model_min_valid_candidates",
        "var_model_universe_seed",
        "var_model_universe_complete",
    ):
        assert hidden_variable not in source
    assert 'values=[MODEL_USAGE_DEVELOPMENT, MODEL_USAGE_HOLDOUT]' in source


def test_active_simple_targets_exposes_global_score_independent_audit() -> None:
    source = inspect.getsource(profile_editor.EvaluationProfileEditor._build_targets_tab_simple)
    assert '"score_independent_audit"' in source
    assert "var_audit_size" in source
    assert "var_audit_min_valid" in source
    assert "var_audit_seed" in source
    assert "var_model_audit_size" not in source
    assert "var_model_min_valid_candidates" not in source
    assert "var_model_universe_seed" not in source


def test_profile_editor_load_preserves_zero_gap_and_small_audit_budget(monkeypatch) -> None:
    source = yaml.safe_load(FINAL_PROFILE.read_text(encoding="utf-8"))
    source["execution_preset"] = {
        "id": "standard",
        "follow_tool_config": True,
    }
    source["selection_policy"].update({
        "selection_strategy": "score_independent_audit",
        "score_independent_audit_enabled": True,
        "min_gap": 0,
        "audit_size": 4,
        "minimum_valid_audit_candidates": 3,
        "audit_seed": 17,
    })
    editor = _headless_editor(monkeypatch)

    editor._apply_payload(source, path="/profiles/resnet_acceptance.yaml")

    assert int(editor.var_min_gap.get()) == 0
    assert int(editor.var_audit_size.get()) == 4
    assert int(editor.var_audit_min_valid.get()) == 3
    assert int(editor.var_audit_seed.get()) == 17


def test_run_mode_change_does_not_replace_profile_owned_audit_budget(monkeypatch) -> None:
    editor = _headless_editor(monkeypatch)
    editor.var_audit_size.set(4)
    editor.var_audit_min_valid.set(3)
    editor.var_audit_seed.set(17)

    editor.var_run_mode_id.set("smoke")
    editor._on_run_mode_changed()

    assert int(editor.var_audit_size.get()) == 4
    assert int(editor.var_audit_min_valid.get()) == 3
    assert int(editor.var_audit_seed.get()) == 17


def test_all_audit_budget_spinboxes_accept_small_profile_values() -> None:
    simple = inspect.getsource(
        profile_editor.EvaluationProfileEditor._build_targets_tab_simple
    )
    legacy = inspect.getsource(
        profile_editor.EvaluationProfileEditor._build_targets_tab
    )

    assert "textvariable=self.var_audit_size, from_=1" in simple
    assert "textvariable=self.var_audit_min_valid, from_=1" in simple
    assert "from_=1, to=999, textvariable=self.var_audit_size" in legacy
    assert "from_=1, to=999, textvariable=self.var_audit_min_valid" in legacy


@pytest.mark.parametrize(
    ("task", "model_id", "onnx", "family", "shape", "expected"),
    [
        (
            "classification",
            "resnet50",
            "/models/resnet50.onnx",
            "resnet",
            "1x3x224x224",
            ("torchvision", "imagenet_val", "imagenet_val_mini_200"),
        ),
        (
            "detection",
            "yolo26s",
            "/models/yolo26s.onnx",
            "yolo26",
            "1x3x640x640",
            ("ultralytics", "coco2017_val", "coco_50"),
        ),
    ],
)
def test_new_simple_model_keeps_internal_dataset_defaults(
    monkeypatch,
    task,
    model_id,
    onnx,
    family,
    shape,
    expected,
) -> None:
    editor = _headless_editor(monkeypatch)
    editor.var_selection_strategy.set("score_independent_audit")
    editor.var_model_id.set(model_id)
    editor.var_model_task.set(task)
    editor.var_model_usage.set(profile_editor.MODEL_USAGE_DEVELOPMENT)
    editor.var_model_onnx.set(onnx)
    editor.var_model_family.set(family)
    editor.var_model_shape.set(shape)

    row = editor._model_from_fields()

    assert (
        row["source"],
        row["semantic_dataset"],
        row["development_subset"],
    ) == expected
    assert row["family_id"] == family
    assert row["evaluation_role"] == "development"
    assert row["generalization_scope"] == "development"
    assert row["candidate_universe"] == {"mode": "deterministic_audit"}
    assert "candidate_universe_complete" not in row


def test_global_audit_replaces_hidden_legacy_development_overrides(monkeypatch) -> None:
    editor = _headless_editor(monkeypatch)
    editor.models = [
        {
            "id": "resnet50",
            "task": "classification",
            "family": "resnet",
            "family_id": "resnet",
            "evaluation_role": "development",
            "generalization_scope": "development",
            "validation_tier": "screening",
            "candidate_universe_complete": False,
            "candidate_universe": {
                "mode": "deterministic_audit",
                "audit_size": 30,
                "minimum_valid_candidates": 15,
                "seed": 7,
            },
        },
        {
            "id": "regnet_holdout",
            "task": "classification",
            "family": "regnet",
            "family_id": "regnet",
            "evaluation_role": "confirmatory_holdout",
            "generalization_scope": "model_family_holdout",
            "validation_tier": "final",
            "candidate_universe_complete": False,
            "candidate_universe": {
                "mode": "deterministic_audit",
                "audit_size": 40,
                "minimum_valid_candidates": 25,
                "seed": 8,
            },
        },
    ]
    editor.var_selection_strategy.set("score_independent_audit")
    editor.var_audit_size.set(20)
    editor.var_audit_min_valid.set(10)
    editor.var_audit_seed.set(20260710)

    normalized = {
        row["id"]: row for row in editor._normalized_models_for_payload()
    }

    for model_id in ("resnet50", "regnet_holdout"):
        assert normalized[model_id]["candidate_universe"] == {
            "mode": "deterministic_audit"
        }
        assert "candidate_universe_complete" not in normalized[model_id]
    assert int(editor.var_audit_size.get()) == 20
    assert int(editor.var_audit_min_valid.get()) == 10
    assert int(editor.var_audit_seed.get()) == 20260710


def test_holdout_contract_is_derived_without_claiming_universe_complete(monkeypatch) -> None:
    editor = _headless_editor(monkeypatch)
    editor.models = [{
        "id": "yolo26s",
        "task": "detection",
        "family": "yolo26",
        "family_id": "yolo26",
        "evaluation_role": "development",
    }]
    editor._selected_model_index = None
    editor.var_model_id.set("yolo26m")
    editor.var_model_task.set("detection")
    editor.var_model_usage.set(profile_editor.MODEL_USAGE_HOLDOUT)
    editor.var_model_onnx.set("/models/yolo26m.onnx")
    editor.var_model_family.set("yolo26")
    editor.var_model_shape.set("1x3x640x640")

    row = editor._model_from_fields()

    assert row["evaluation_role"] == "confirmatory_holdout"
    assert row["generalization_scope"] == "within_family_transfer"
    assert row["validation_tier"] == "final"
    assert row["candidate_universe"] == {"mode": "deterministic_audit"}
    assert "candidate_universe_complete" not in row


def test_schema_accepts_global_score_independent_audit_contract() -> None:
    schema = json.loads(PROFILE_SCHEMA.read_text(encoding="utf-8"))
    source = yaml.safe_load(FINAL_PROFILE.read_text(encoding="utf-8"))
    source["selection_policy"].update({
        "selection_strategy": "score_independent_audit",
        "score_independent_audit_enabled": True,
        "audit_candidate_universe": "deterministic_audit",
        "audit_size": 20,
        "minimum_valid_audit_candidates": 10,
        "audit_seed": 20260710,
    })
    jsonschema.validate(source, schema)


def test_profile_semantics_reject_impossible_global_audit_minimum() -> None:
    source = yaml.safe_load(FINAL_PROFILE.read_text(encoding="utf-8"))
    source["selection_policy"].update({
        "selection_strategy": "score_independent_audit",
        "score_independent_audit_enabled": True,
        "audit_size": 4,
        "minimum_valid_audit_candidates": 5,
    })

    with pytest.raises(
        ValueError,
        match=(
            r"minimum_valid_audit_candidates \(5\) must be <= "
            r"selection_policy.audit_size \(4\)"
        ),
    ):
        validate_evaluation_profile_payload(source, source="audit-test")


def test_profile_semantics_keep_large_but_consistent_audit_legal() -> None:
    source = yaml.safe_load(FINAL_PROFILE.read_text(encoding="utf-8"))
    source["selection_policy"].update({
        "selection_strategy": "score_independent_audit",
        "score_independent_audit_enabled": True,
        "audit_size": 30,
        "minimum_valid_audit_candidates": 10,
    })

    result = validate_evaluation_profile_payload(source, source="audit-test")

    assert result["selection_policy"]["audit_size"] == 30


@pytest.mark.parametrize("audit_size", [20, 30])
def test_score_independent_audit_has_explicit_gui_start_contract(
    audit_size: int,
) -> None:
    plan = {
        "score_independent_audit_counts": {
            "resnet50": audit_size,
            "yolo26s": audit_size,
            "yolov7_paper": audit_size,
        },
        "execution_union_candidate_count_min_total": 3 * audit_size,
        "execution_union_candidate_count_upper_bound_total": 3 * (audit_size + 1),
        "expected_generic_result_rows_min_total": 12 * audit_size,
        "expected_generic_result_rows_total": 12 * (audit_size + 1),
        "native_enabled": True,
        "native_energy_enabled": False,
    }

    summary = score_independent_audit_start_summary(plan)

    assert summary["confirmation_required"] is True
    assert summary["audit_counts"]["resnet50"] == audit_size
    assert str(3 * audit_size) in summary["confirmation_text"]
    assert "kein kleiner Standardlauf" in summary["confirmation_text"]
    assert "Native=an" in summary["concise"]
    assert "Native Energy=aus" in summary["concise"]


def test_non_audit_has_no_gui_start_confirmation_contract() -> None:
    assert score_independent_audit_start_summary({
        "score_independent_audit_counts": {},
        "native_enabled": True,
    }) == {}


def test_gui_queue_uses_audit_summary_for_fresh_start_confirmation(
    monkeypatch,
) -> None:
    gui_app = _import_gui_app_headless(monkeypatch)

    gui = object.__new__(gui_app.SplitPointAnalyserGUI)
    statuses: list[str] = []
    gui._eval_finalize_partial_options_override = None
    gui.var_eval_workflow_status = SimpleNamespace(set=statuses.append)
    gui._eval_workflow_snapshot_options = lambda **_kwargs: SimpleNamespace(
        resume=False,
        profile_start_snapshot={"resolved_profile": {"name": "audit-profile"}},
    )
    monkeypatch.setattr(
        gui_app,
        "build_effective_execution_plan",
        lambda _profile: {
            "score_independent_audit_counts": {"resnet50": 30},
            "execution_union_candidate_count_min_total": 30,
            "execution_union_candidate_count_upper_bound_total": 31,
            "expected_generic_result_rows_min_total": 120,
            "expected_generic_result_rows_total": 124,
            "native_enabled": True,
            "native_energy_enabled": False,
        },
    )
    prompt: dict[str, object] = {}

    def _decline(title: str, message: str, **kwargs) -> bool:
        prompt.update(title=title, message=message, **kwargs)
        return False

    monkeypatch.setattr(gui_app.messagebox, "askyesno", _decline)

    assert gui._queue_evaluation_workflow() is None
    assert prompt["title"] == "Ranking-Audit starten?"
    assert "resnet50=30" in str(prompt["message"])
    assert prompt["default"] == gui_app.messagebox.NO
    assert statuses == ["Start abgebrochen: Ranking-Audit nicht bestätigt."]


def test_new_profile_defaults_to_imagenet_deepx_preprocessing(monkeypatch) -> None:
    editor = _headless_editor(monkeypatch)

    assert editor.var_deepx_classification_preprocessing.get() == (
        "imagenet_mean_std"
    )


@pytest.mark.parametrize(
    "preprocessing", ["imagenet_mean_std", "current_scale_only"]
)
def test_profile_editor_roundtrip_preserves_explicit_deepx_preprocessing(
    monkeypatch, preprocessing: str,
) -> None:
    source = yaml.safe_load(FINAL_PROFILE.read_text(encoding="utf-8"))
    source["deepx_build"] = {
        "mode": "reuse_and_build_missing",
        "classification_preprocessing": preprocessing,
    }
    editor = _headless_editor(monkeypatch)

    editor._apply_payload(source, path=str(FINAL_PROFILE))
    result = editor._build_payload()

    assert editor.var_deepx_classification_preprocessing.get() == preprocessing
    assert result["deepx_build"]["classification_preprocessing"] == preprocessing


def test_profile_editor_maps_absent_deepx_preprocessing_to_legacy_current(
    monkeypatch,
) -> None:
    source = yaml.safe_load(FINAL_PROFILE.read_text(encoding="utf-8"))
    source.pop("deepx_build", None)
    editor = _headless_editor(monkeypatch)

    editor._apply_payload(source, path=str(FINAL_PROFILE))
    result = editor._build_payload()

    assert editor.var_deepx_classification_preprocessing.get() == (
        "current_scale_only"
    )
    assert result["deepx_build"]["classification_preprocessing"] == (
        "current_scale_only"
    )


def test_score_audit_editor_roundtrip_uses_global_budget_and_drops_false_veto(monkeypatch) -> None:
    source = yaml.safe_load(FINAL_PROFILE.read_text(encoding="utf-8"))
    source["selection_policy"].update({
        "selection_strategy": "score_independent_audit",
        "score_independent_audit_enabled": True,
        "audit_candidate_universe": "deterministic_audit",
        "audit_size": 20,
        "minimum_valid_audit_candidates": 10,
        "audit_seed": 20260710,
    })
    development = source["model_suite"]["primary"][0]
    development["candidate_universe_complete"] = False
    development["candidate_universe"] = {
        "mode": "deterministic_audit",
        "audit_size": 30,
        "minimum_valid_candidates": 15,
        "seed": 7,
    }
    editor = _headless_editor(monkeypatch)

    editor._apply_payload(source, path=str(FINAL_PROFILE))
    result = editor._build_payload()

    assert result["selection_policy"]["audit_size"] == 20
    assert result["selection_policy"]["minimum_valid_audit_candidates"] == 10
    assert result["selection_policy"]["audit_seed"] == 20260710
    development_result = result["model_suite"]["primary"][0]
    assert development_result["candidate_universe"] == {
        "mode": "deterministic_audit"
    }
    assert "candidate_universe_complete" not in development_result
    for row in result["model_suite"]["primary"]:
        universe = row["candidate_universe"]
        assert "audit_size" not in universe
        assert "minimum_valid_candidates" not in universe
        assert "seed" not in universe
