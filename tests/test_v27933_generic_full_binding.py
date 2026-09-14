"""T33.24–26: real Q3 import replay and explicitly synthetic contradictions."""
from __future__ import annotations

import copy
import csv
import hashlib
import json
from pathlib import Path
import sys
import types

import pytest

from onnx_splitpoint_tool.workflow.result_context import (
    bind_benchmark_source_context, load_benchmark_source_contexts,
)
from onnx_splitpoint_tool.workflow.results import normalize_benchmark_files
from onnx_splitpoint_tool.workflow.runner import (
    _bind_results_to_required_scope_v2796, missing_profile_measurements_v60r,
    duplicate_profile_measurements_v269d,
)

FIXTURE = Path(__file__).parent / "fixtures/v27933_generic_full"
MODEL = "mobilenet_v3_large"


def _original():
    data = json.loads((FIXTURE / "q3_mobilenet_projection.json").read_text())
    contexts = load_benchmark_source_contexts(FIXTURE, model_id=MODEL)
    return data["rows"], data["required_results"], contexts


def test_q3_full_rows_bind_actual_dispatch_without_changing_measurements():
    rows, expected, contexts = _original()
    assert len(contexts) == 12
    before = copy.deepcopy(rows)
    recovered = [bind_benchmark_source_context(row, contexts) for row in rows]
    bound, errors = _bind_results_to_required_scope_v2796(expected, recovered)
    assert errors == []
    assert len(rows) == len(bound) == 7
    assert rows == before
    for old, new in zip(rows, bound):
        for key in ("model_id", "backend", "variant", "case_id", "measurement_endpoint",
                    "total_latency_ms", "throughput_primary_fps", "runtime_ok",
                    "validation_ok", "final_pass", "error_class", "quality_identity_errors"):
            assert new.get(key) == old.get(key), key
    full = {row["backend"]: row for row in bound if row["variant"] == "full"}
    for backend in ("hailo8", "hailo10"):
        assert full[backend]["setup_id"] == f"orin_nx_{backend}_01"
        assert full[backend]["measurement_endpoint"] == "classification_logits"
        assert full[backend]["required_scope_binding_status"] == "exact_physical_match"
    blocked = full["deepx_m1"]
    assert blocked["required_scope_binding_status"] == "exact_terminal_failure"
    assert blocked["runtime_ok"] is False and blocked["error_class"] == "missing_artifact"
    assert not blocked["measurement_endpoint"]
    assert blocked["total_latency_ms"] is None and blocked["throughput_primary_fps"] is None
    # The genuinely absent independent Split job remains absent.
    missing = missing_profile_measurements_v60r(expected, bound)
    assert [(row["backend"], row["variant"]) for row in missing] == [("deepx_m1_to_tensorrt", "split")]
    assert duplicate_profile_measurements_v269d(expected, bound) == []


@pytest.mark.parametrize("change", ["setup_conflict", "two_targets", "wrong_model", "wrong_run", "wrong_path", "wrong_endpoint"])
def test_synthetic_conflicts_and_unbound_origins_are_not_repaired(change):
    rows, expected, contexts = _original()
    row = next(row for row in rows if row["backend"] == "hailo8")
    context = next(context for context in contexts if context["source_path"] == row["source_path"])
    if change == "setup_conflict":
        row["setup_id"] = "different_setup"
    elif change == "two_targets":
        contexts.append({**context, "setup_id": "different_setup"})
    elif change == "wrong_model":
        row["model_id"] = "other_model"
    elif change == "wrong_run":
        row["run_id"] = "other_run"
    elif change == "wrong_path":
        row["source_path"] = "/unrelated/" + Path(row["source_path"]).name
    elif change == "wrong_endpoint":
        row["measurement_endpoint"] = "p2_output"
    bound, errors = _bind_results_to_required_scope_v2796(
        expected, [bind_benchmark_source_context(row, contexts)],
    )
    assert errors and bound[0]["required_scope_binding_status"] != "exact_physical_match"
    assert bound[0].get("measurement_endpoint") == row.get("measurement_endpoint")
    if change == "setup_conflict":
        assert bound[0]["setup_id"] == "different_setup"


def test_failed_row_without_exact_context_cannot_satisfy_terminal_scope():
    rows, expected, _ = _original()
    row = next(row for row in rows if row["backend"] == "deepx_m1")
    bound, errors = _bind_results_to_required_scope_v2796(expected, [row])
    assert errors
    assert len(missing_profile_measurements_v60r(expected, bound)) == len(expected)


@pytest.mark.parametrize("context_setup, expected_status", [
    ("my-hailo", "exact_dispatch_copy_match"),
    (" My Hailo ", "exact_dispatch_copy_match"),
    ("other-hailo", "conflict"),
])
def test_synthetic_custom_setup_uses_same_normalization_as_direct_identity(context_setup, expected_status):
    row = {"model_id": MODEL, "run_id": "hailo8", "setup_id": "my_hailo",
           "source_path": "/synthetic/benchmark_results_hailo8_auto.json"}
    context = {**row, "setup_id": context_setup}
    original_context = copy.deepcopy(context)
    bound = bind_benchmark_source_context(row, [context])
    assert bound["source_context_binding_status"] == expected_status
    assert bound["setup_id"] == "my_hailo"
    assert bound["source_context_records"] == [original_context]
    assert context == original_context
    if expected_status == "conflict":
        assert bound["source_context_binding_error"] == "source_context_setup_conflict"


def test_terminal_failure_cannot_satisfy_a_success_required_scope():
    rows, expected, contexts = _original()
    row = next(row for row in rows if row["backend"] == "deepx_m1")
    for identity in expected:
        identity["success_required"] = True
    bound, errors = _bind_results_to_required_scope_v2796(
        expected, [bind_benchmark_source_context(row, contexts)],
    )
    assert errors
    assert bound[0]["runtime_ok"] is False


def test_real_csv_projection_and_synthetic_json_mirror_keep_one_full_row(tmp_path):
    # The classification JSON files were outside Q3's capture size limit.
    # This mirror is explicitly synthetic; its values are the real CSV scalars.
    _, expected, contexts = _original()
    csv_path = tmp_path / "benchmark_results_hailo8_auto.csv"
    csv_path.write_bytes((FIXTURE / csv_path.name).read_bytes())
    raw = list(csv.DictReader(csv_path.open()))
    json_path = csv_path.with_suffix(".json")
    json_path.write_text(json.dumps(raw))
    original_context = next(context for context in contexts if context["source_path"].endswith(csv_path.name))
    # Explicit test-only relocation into a temporary import directory.
    relocated = [{**original_context, "source_path": str(path)} for path in (csv_path, json_path)]
    rows, sources = normalize_benchmark_files(model_id=MODEL, source_paths=[tmp_path], source_contexts=relocated)
    bound, errors = _bind_results_to_required_scope_v2796(expected, rows)
    assert errors == []
    assert len(bound) == 1 and bound[0]["variant"] == "full"
    assert bound[0]["setup_id"] == "orin_nx_hailo8_01"
    assert bound[0]["measurement_endpoint"] == "classification_logits"
    assert next(source for source in sources if source["path"].endswith(".csv"))["suppressed_csv_mirror_count"] == 1


def test_dispatch_and_copy_must_agree_before_context_is_loaded(tmp_path):
    sid = "orin_nx_hailo8_01"
    dispatch = json.loads((FIXTURE / f"remote_benchmark_dispatch_{sid}.json").read_text())
    dispatch["args"]["quality_evidence_setup_id"] = "contradictory_setup"
    (tmp_path / f"remote_benchmark_dispatch_{sid}.json").write_text(json.dumps(dispatch))
    target = tmp_path / "remote_diagnostics" / sid
    target.mkdir(parents=True)
    (target / "result_copy_manifest.json").write_bytes((FIXTURE / "remote_diagnostics" / sid / "result_copy_manifest.json").read_bytes())
    assert load_benchmark_source_contexts(tmp_path, model_id=MODEL) == []


def test_real_q3_detection_json_csv_mirror_is_one_measurement():
    # Both files contain corresponding scalar projections of original Q3
    # JSON and CSV, unlike the explicitly synthetic classification mirror.
    rows, sources = normalize_benchmark_files(
        model_id="yolo11l", source_paths=[FIXTURE / "yolo11l"],
    )
    assert len(rows) == 1 and rows[0]["variant"] == "full"
    assert rows[0]["measurement_endpoint"] == "completed_detection"
    assert rows[0]["total_latency_ms"] == pytest.approx(87.1666431427002)
    assert sum(source.get("suppressed_csv_mirror_count", 0) for source in sources) == 1


@pytest.fixture
def generic_suite(tmp_path, monkeypatch):
    template = Path(__file__).parents[1] / "onnx_splitpoint_tool/resources/templates/benchmark_suite.py.txt"
    module = types.ModuleType("v27933_generic_full_suite")
    module.__file__ = str(tmp_path / "benchmark_suite.py")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    exec(compile(template.read_text().replace("__BENCH_JSON__", "benchmark_set.json"), module.__file__, "exec"), module.__dict__)
    monkeypatch.setattr(module, "_cache_verify_only", lambda: False)
    monkeypatch.setattr(module, "_stage2_accelerator_gate", lambda *a: None)
    monkeypatch.setattr(module, "_provider_unavailable_reason", lambda *a: None)
    monkeypatch.setattr(module, "_select_visual_sample_image", lambda *a, **k: None)
    monkeypatch.setattr(module, "_collect_case_result", lambda *a, **k: {})
    return module


@pytest.mark.parametrize("central", [True, False])
def test_generic_full_producer_forwards_exact_setup_without_quality_only_mode(tmp_path, monkeypatch, generic_suite, central):
    case = tmp_path / "b135"
    case.mkdir()
    (case / "run_split_onnxruntime.py").write_text("# --phase-runs\n")
    commands = []
    monkeypatch.setattr(generic_suite.subprocess, "run", lambda cmd, **kw: commands.append(cmd) or types.SimpleNamespace(returncode=0))
    generic_suite._run_case(case, run_id="hailo8", provider="hailo8", variants=["full"],
        image="default", preset="auto", image_scale="imagenet", warmup=1, runs=1, timeout_s=1,
        quality_evidence_model_id=MODEL, quality_evidence_setup_id="orin_nx_hailo8_01",
        quality_evidence_eval_id="recorded_eval", quality_evidence_source_run_id="hailo8",
        task_quality_gate={"statistics": {"execution_location": "central_management" if central else "local"}})
    assert len(commands) == 1
    command = commands[0]
    assert command[command.index("--quality-evidence-setup-id") + 1] == "orin_nx_hailo8_01"
    assert command[command.index("--quality-evidence-source-run-id") + 1] == "hailo8"
    assert "--quality-evidence-only" not in command
    assert "--full-only-quality-identity-json" not in command


def test_original_evidence_fixtures_remain_byte_unchanged():
    expected = json.loads((FIXTURE / "fixture_checksums.json").read_text())
    for relative, digest in expected.items():
        assert hashlib.sha256((FIXTURE / relative).read_bytes()).hexdigest() == digest
