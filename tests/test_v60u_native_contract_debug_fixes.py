from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

from onnx_splitpoint_tool import __version__
from onnx_splitpoint_tool.run_modes import RUN_MODE_SCHEMA_VERSION, default_run_modes_config
from onnx_splitpoint_tool.workflow.execution_binding import _copy_remote_result_files
from onnx_splitpoint_tool.workflow.zip_utils import write_path_portable
from onnx_splitpoint_tool.native_output_endpoint import DECODED_NMS_ATTESTATION_SOURCE

ROOT = Path(__file__).resolve().parents[1]
_NMS_ATTESTATION_SOURCE = DECODED_NMS_ATTESTATION_SOURCE


def _attested_decoded_contract() -> dict:
    endpoint_hash = "a" * 64
    return {
        "task": "detection",
        "stage": "decoded_nms",
        "output_format": "bn6_detections",
        "contract_family": "decoded_nms",
        "contract_source": _NMS_ATTESTATION_SOURCE,
        "endpoint_contract_complete": True,
        "endpoint_contract_hash": endpoint_hash,
        "output_endpoint_attestation": {
            "attested": True,
            "declaration_attested": True,
            "status": "passed",
            "contract_source": _NMS_ATTESTATION_SOURCE,
            "endpoint_contract_hash": endpoint_hash,
        },
    }


def test_versions_and_cold_build_policy() -> None:
    assert __version__ in {"0.14.20+v60u.nativecontractfix", "0.14.21+v60v.generationnativefix", "0.14.22+v60w.smokedeferralpackfix", "0.14.23+v60x.nativeevidencefix", "0.14.25+v60z.nativefullquality", "0.14.26+v61a.nativefullenergyprogress", "0.14.27+v61b.nativeintegrationfix", "0.14.28+v61c.nativefullpairedenergyfix", "0.14.29+v61d.nativefullsemanticfix", "0.14.30+v61e.standardguifix", "2.61.0+v61e", "2.62.0", "2.63.0", "2.64.0", "2.65.0", "2.66.0", "2.67.0", "2.68.0", "2.69.6", "2.70.6", "2.70.7", "2.70.8", "2.70.9", "2.70.10", "2.71.1", "2.71.2", "2.71.3", "2.71.4", "2.72.0", "2.72.1", "2.72.2", "2.72.3", "2.72.4", "2.72.5", "2.72.6", "2.72.7", "2.73.0", "2.73.6", "2.73.7", "2.75.2", "2.75.3", "2.75.7", "2.75.8", "2.75.9", "2.75.10", "2.75.11", "2.75.12", "2.75.13", "2.75.14", "2.75.15", "2.75.16", "2.75.17", "2.75.20", "2.75.21", "2.75.22", "2.75.24", "2.75.26", "2.75.27", "2.75.28", "2.75.30", "2.75.31", "2.75.32", "2.75.38", "2.75.39", "2.75.40", "2.75.41", "2.75.42", "2.75.45", "2.75.46", "2.75.47", "2.76.0", "2.76.1", "2.76.2", "2.77.0", "2.77.1", "2.77.2", "2.77.3", "2.77.4", "2.77.5", "2.77.6", "2.77.7", "2.77.8", "2.77.9", "2.77.10", "2.77.11", "2.77.12", "2.77.13", "2.77.14", "2.77.15", "2.78.0", "2.78.1", "2.78.2", "2.78.3", "2.78.4"}
    assert RUN_MODE_SCHEMA_VERSION in {5, 7, 8, 9, 10, 11, 12, 13}
    cfg = default_run_modes_config()
    assert cfg["modes"]["smoke"]["build"]["hailo"]["full_baseline_cold_build_policy"] == "cache_or_defer"
    assert cfg["modes"]["standard"]["build"]["hailo"]["full_baseline_cold_build_policy"] == "build_missing"


def test_empty_canonical_file_does_not_count_as_result(tmp_path: Path) -> None:
    remote = tmp_path / "remote" / "results"
    remote.mkdir(parents=True)
    (remote / "benchmark_results_empty.json").write_text("[]", encoding="utf-8")
    (remote / "benchmark_results_empty.error.txt").write_text("No results collected", encoding="utf-8")
    dst = tmp_path / "dst"
    _copy_remote_result_files(remote.parent, dst, flat_prefix="target")
    manifest = json.loads((dst / "remote_diagnostics" / "target" / "result_copy_manifest.json").read_text())
    assert manifest["canonical_file_count"] == 1
    assert manifest["canonical_parseable_file_count"] == 1
    assert manifest["canonical_nonempty_row_count"] == 0
    assert manifest["canonical_run_ids_without_rows"] == ["empty"]
    assert manifest["status"] == "missing_canonical_results"


def test_nonempty_and_empty_canonical_files_produce_partial(tmp_path: Path) -> None:
    remote = tmp_path / "remote" / "results"
    remote.mkdir(parents=True)
    (remote / "benchmark_results_empty.json").write_text("[]", encoding="utf-8")
    (remote / "benchmark_results_ok.json").write_text('[{"fps": 1.0}]', encoding="utf-8")
    dst = tmp_path / "dst"
    _copy_remote_result_files(remote.parent, dst, flat_prefix="target")
    manifest = json.loads((dst / "remote_diagnostics" / "target" / "result_copy_manifest.json").read_text())
    assert manifest["canonical_nonempty_row_count"] == 1
    assert manifest["status"] == "partial"


def test_portable_zip_clamps_pre_1980_timestamp(tmp_path: Path) -> None:
    src = tmp_path / "old.log"
    src.write_text("hello", encoding="utf-8")
    os.utime(src, (0, 0))
    out = tmp_path / "debug.zip"
    with zipfile.ZipFile(out, "w") as zf:
        write_path_portable(zf, src, "native/old.log")
    with zipfile.ZipFile(out) as zf:
        assert zf.getinfo("native/old.log").date_time[0] == 1980
        assert zf.read("native/old.log") == b"hello"


def test_native_diagnostics_and_contract_family_hooks_present() -> None:
    matrix = (ROOT / "scripts/native_fifo_smoke_matrix.py").read_text(encoding="utf-8")
    runner = (ROOT / "scripts/native_fifo_eval_runner.py").read_text(encoding="utf-8")
    final = (ROOT / "scripts/native_producer_final_report.py").read_text(encoding="utf-8")
    validator = (ROOT / "scripts/native_producer_validate_visualize.py").read_text(encoding="utf-8")
    for token in ("failure_reason", "status_detail", "stdout_tail", "stderr_tail"):
        assert token in matrix
        assert token in runner
        assert token in final
    assert "detection_contract_family_mismatch" in validator
    assert "preprocess_profile" in validator


def test_hailo_attempt_provenance_and_cache_only_hooks_present() -> None:
    binding = (ROOT / "onnx_splitpoint_tool/workflow/hailo_remote_binding.py").read_text(encoding="utf-8")
    backend = (ROOT / "onnx_splitpoint_tool/hailo_backend.py").read_text(encoding="utf-8")
    helper = (ROOT / "onnx_splitpoint_tool/wsl_inline_build_hef").read_text(encoding="utf-8")
    assert "hailo_build_attempts.json" in binding
    assert "attempted_timeout" in binding
    assert "deferred_cold_full_cache_miss" in backend
    assert "--cache-only" in helper
    assert "--net-input-shapes-json" in helper
    assert "net_input_shapes=net_input_shapes" in helper


def test_debug_pack_includes_native_analysis_and_safe_zip() -> None:
    app = (ROOT / "onnx_splitpoint_tool/gui/app.py").read_text(encoding="utf-8")
    cli = (ROOT / "scripts/create_evaluation_debug_pack.py").read_text(encoding="utf-8")
    builder = (ROOT / "onnx_splitpoint_tool/workflow/debug_pack.py").read_text(encoding="utf-8")
    assert "from ..workflow.debug_pack import create_evaluation_debug_pack" in app
    assert "create_evaluation_debug_pack" in cli
    assert '"native_producers"' in builder
    assert "reports/native_validation/" in builder
    assert "reports/window_method_validation_probe/" in builder
    assert "zipinfo_for_path" in builder


def _load_native_validator_module():
    path = ROOT / "scripts/native_producer_validate_visualize.py"
    spec = importlib.util.spec_from_file_location("v60u_native_validator_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_detection_contract_gate_requires_archived_contract_metadata(tmp_path: Path) -> None:
    mod = _load_native_validator_module()
    output_manifest = tmp_path / "native_outputs_manifest.json"
    output_manifest.write_text(json.dumps({"outputs": []}), encoding="utf-8")
    result = {
        "ok": True,
        "semantic_ok": True,
        "semantic_available": True,
        "decode_mode": "output0:raw",
        "best": {"native_mode": "output0:raw", "full_mode": "output0:raw"},
    }
    gated = mod._enforce_detection_contract(dict(result), output_manifest)
    assert gated["ok"] is False
    assert gated["semantic_available"] is False
    assert gated["diagnosis"] == "detection_contract_metadata_unavailable"


def test_detection_contract_gate_rejects_raw_decoded_mismatch(tmp_path: Path) -> None:
    mod = _load_native_validator_module()
    output_manifest = tmp_path / "native_outputs_manifest.json"
    output_manifest.write_text(
        json.dumps(_attested_decoded_contract()),
        encoding="utf-8",
    )
    result = {
        "ok": True,
        "semantic_ok": True,
        "semantic_available": True,
        "decode_mode": "output0:raw",
        "best": {"native_mode": "output0:raw", "full_mode": "output0:raw"},
    }
    gated = mod._enforce_detection_contract(dict(result), output_manifest)
    assert gated["ok"] is False
    assert gated["diagnosis"] == "detection_contract_family_mismatch"
    assert gated["expected_contract_family"] == "decoded_nms"


def test_classification_preprocess_profile_is_imagenet_for_resnet(tmp_path: Path) -> None:
    mod = _load_native_validator_module()
    boundary = tmp_path / "resnet50" / "native_boundary_manifest.json"
    boundary.parent.mkdir(parents=True)
    boundary.write_text(json.dumps({"preprocess": {}}), encoding="utf-8")
    full = tmp_path / "resnet50_full.onnx"
    full.write_bytes(b"onnx")
    profile, source = mod._classification_preprocess_profile(boundary, full, tmp_path)
    assert profile == "imagenet"
    assert source in {"model_family_contract", "benchmark_set_contract"}


def test_duplicate_json_csv_run_id_is_complete_when_one_representation_has_rows(tmp_path: Path) -> None:
    remote = tmp_path / "remote" / "results"
    remote.mkdir(parents=True)
    (remote / "benchmark_results_case.json").write_text("[]", encoding="utf-8")
    (remote / "benchmark_results_case.csv").write_text("run_id,fps\ncase,1.0\n", encoding="utf-8")
    dst = tmp_path / "dst"
    _copy_remote_result_files(remote.parent, dst, flat_prefix="target")
    manifest = json.loads((dst / "remote_diagnostics" / "target" / "result_copy_manifest.json").read_text())
    assert manifest["canonical_nonempty_row_count"] == 1
    assert manifest["canonical_run_ids_without_rows"] == []
    assert manifest["status"] == "ok"


def test_hailo_cache_only_miss_is_resolved_before_sdk_compiler_import() -> None:
    source = (ROOT / "onnx_splitpoint_tool/hailo_backend.py").read_text(encoding="utf-8")
    fn_start = source.index("def _hailo_build_hef_legacy")
    fn_end = source.index("def hailo_build_hef_via_wsl", fn_start)
    body = source[fn_start:fn_end]
    assert "deferred_cold_full_cache_miss" in body
    assert body.index("if bool(cache_only)") < body.index("from hailo_sdk_client import ClientRunner")


def test_native_final_report_preserves_hailo10_and_deepx_row_failures(tmp_path: Path) -> None:
    path = ROOT / "scripts/native_producer_final_report.py"
    spec = importlib.util.spec_from_file_location("v60u_native_final_report_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    analysis = tmp_path / "analysis_tables"
    analysis.mkdir()
    (analysis / "native_hailo10h_producer_e2e_eval__x.json").write_text(json.dumps({
        "rows": [{"model": "resnet50", "case": "b052", "ok": False, "failure_reason": "top1_mismatch", "stderr_tail": "hailo10 detail"}]
    }), encoding="utf-8")
    (analysis / "native_deepx_producer_e2e_eval__x.json").write_text(json.dumps({
        "rows": [{"model": "yolo", "case": "b001", "ok": False, "failure_reason": "decoder_contract_mismatch", "stderr_tail": "deepx detail"}]
    }), encoding="utf-8")
    hrows = module._rows_from_hailo10(tmp_path)
    drows = module._rows_from_deepx(tmp_path)
    assert hrows[0]["failure_reason"] == "top1_mismatch"
    assert hrows[0]["stderr_tail"] == "hailo10 detail"
    assert drows[0]["failure_reason"] == "decoder_contract_mismatch"
    assert drows[0]["stderr_tail"] == "deepx detail"


def test_cli_debug_pack_keeps_pre_1980_native_analysis(tmp_path: Path) -> None:
    run = tmp_path / "evaluation_run"
    analysis = run / "native_producers" / "hailo8" / "analysis_tables"
    analysis.mkdir(parents=True)
    diagnostic = analysis / "native_fifo_eval_runner.json"
    diagnostic.write_text('{"failure_reason":"row_error"}', encoding="utf-8")
    os.utime(diagnostic, (0, 0))
    (run / "run_manifest.json").write_text(json.dumps({
        "schema": "onnx-splitpoint/evaluation-run-manifest",
        "schema_version": 1,
        "run_id": run.name,
        "status": "partial",
    }), encoding="utf-8")
    out = tmp_path / "debug.zip"
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts/create_evaluation_debug_pack.py"), "--eval-run-dir", str(run), "--out", str(out)],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout
    with zipfile.ZipFile(out) as zf:
        arc = "native_producers/hailo8/analysis_tables/native_fifo_eval_runner.json"
        assert arc in zf.namelist()
        assert zf.getinfo(arc).date_time[0] == 1980


def _load_script_module(name: str, relpath: str):
    import importlib.util
    path = ROOT / relpath
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_detection_candidate_selection_respects_declared_contract_family(monkeypatch) -> None:
    module = _load_script_module("v60u_native_validator", "scripts/native_producer_validate_visualize.py")
    candidates = [
        {"mode": "output0:raw", "kind": "raw", "detections": [{"box": [0, 0, 10, 10], "score": 1.0, "class_id": 0}]},
        {"mode": "output0:nms:xyxy_score_class", "kind": "decoded", "detections": [{"box": [1, 1, 11, 11], "score": 0.5, "class_id": 0}]},
    ]
    monkeypatch.setattr(module, "_decode_layout_candidates", lambda *a, **k: candidates)
    monkeypatch.setattr(module, "_nms", lambda detections: list(detections))
    monkeypatch.setattr(module, "_match_detections", lambda ref, dets: {"matched": 100 if dets and dets[0]["score"] == 1.0 else 1, "match_ratio": 1.0, "mean_iou": 1.0, "ref_count": 1})
    monkeypatch.setattr(module, "_match_detections_class_agnostic", lambda ref, dets: {"matched": 1, "match_ratio": 1.0, "mean_iou": 1.0, "ref_count": 1})
    monkeypatch.setattr(module, "_score_saturation", lambda dets: {"score_ge_0999": int(bool(dets and dets[0]["score"] >= 0.999))})
    ref = [{"box": [0, 0, 10, 10], "score": 1.0, "class_id": 0}]

    dets, mode, diag = module._choose_detection_candidate({}, ref_dets=ref, expected_family="decoded_nms")
    assert mode.startswith("output0:nms:")
    assert diag["selected_contract_family"] == "decoded_nms"
    assert any(row["contract_family"] == "raw_head" for row in diag["rejected_contract_candidates"])

    dets, mode, diag = module._choose_detection_candidate({}, ref_dets=ref, expected_family="raw_head")
    assert mode == "output0:raw"
    assert diag["selected_contract_family"] == "raw_head"
    assert any(row["contract_family"] == "decoded_nms" for row in diag["rejected_contract_candidates"])


def test_yolo_self_reference_never_compares_cross_contract_families() -> None:
    module = _load_script_module("v60u_yolo_probe", "scripts/native_yolo_full_self_reference_probe.py")
    det = [{"x1": 0.0, "y1": 0.0, "x2": 10.0, "y2": 10.0, "score": 0.9, "class_id": 0}]
    full = [
        {"mode": "full:nms:xyxy_score_class", "detections": det, "score_saturation": {}},
        {"mode": "full:raw", "detections": det, "score_saturation": {}},
    ]
    native = [
        {"mode": "native:raw", "detections": det, "score_saturation": {}},
        {"mode": "native:nms:xyxy_score_class", "detections": det, "score_saturation": {}},
    ]
    best, rows, rejected = module._best_self_match(full, native, expected_family="decoded_nms")
    assert best["contract_family_match"] is True
    assert best["native_contract_family"] == "decoded_nms"
    assert best["full_contract_family"] == "decoded_nms"
    assert rows and rejected
    assert all(row["native_contract_family"] == row["full_contract_family"] == "decoded_nms" for row in rows)
    assert all(row["reason"] == "detection_contract_family_mismatch" for row in rejected)


def test_native_analysis_rc_zero_with_failed_rows_is_partial(tmp_path: Path) -> None:
    from onnx_splitpoint_tool.workflow.runner import _native_analysis_diagnostics_v60u

    table = tmp_path / "analysis_tables"
    table.mkdir(parents=True)
    (table / "native_fifo_eval_runner__smoke.json").write_text(
        json.dumps(
            {
                "rows": [
                    {"backend": "hailo8_to_trt", "model": "resnet50", "case": "b052", "ok": False, "returncode": 0, "failure_reason": "native_result_not_ok", "stderr_tail": "detail"},
                    {"backend": "deepx_to_trt", "model": "resnet50", "case": "b052", "ok": True, "returncode": 0},
                ]
            }
        ),
        encoding="utf-8",
    )
    diag = _native_analysis_diagnostics_v60u(tmp_path)
    assert diag["row_count"] == 2
    assert diag["ok_count"] == 1
    assert diag["failed_count"] == 1
    assert diag["evidence_status"] == "partial"
    assert diag["failure_reasons"] == ["native_result_not_ok"]
    assert diag["failed_rows"][0]["stderr_tail"] == "detail"


def test_cache_only_policy_does_not_change_hailo_artifact_contract(tmp_path: Path) -> None:
    from onnx_splitpoint_tool.hailo_backend import _v60s_hailo_contract

    model = tmp_path / "model.onnx"
    model.write_bytes(b"same-model")
    base = {"onnx_path": model, "hw_arch": "hailo8", "opt_level": 1, "calib_count": 8, "outdir": tmp_path / "out"}
    assert _v60s_hailo_contract({**base, "cache_only": False}) == _v60s_hailo_contract({**base, "cache_only": True})


def test_hailo_timeout_attempt_remains_terminal_in_queue(tmp_path: Path) -> None:
    from onnx_splitpoint_tool.workflow.hailo_remote_binding import (
        _build_queue_from_plan,
        _collect_hailo_build_attempts_binding,
        _merge_attempts_into_queue,
    )

    base = tmp_path / "benchmark_set"
    result = base / "legacy_suite" / "hailo" / "hailo8" / "full" / "hailo_hef_build_result.json"
    result.parent.mkdir(parents=True)
    result.write_text(
        json.dumps(
            {
                "ok": False,
                "timed_out": True,
                "timeout_kind": "hard",
                "last_stage": "partition_search",
                "failure_kind": "timeout",
                "elapsed_s": 900.1,
                "error": "hard timeout",
                "calib_info": {"cache_hit": False, "cache_key": "abc"},
            }
        ),
        encoding="utf-8",
    )
    queue = _build_queue_from_plan(
        {
            "model_id": "yolo26s",
            "full_baseline_requests": [{"backend": "hailo8", "ready": False, "status": "pending_dfc_build_or_preparation"}],
        }
    )
    attempts = _collect_hailo_build_attempts_binding(base)
    _merge_attempts_into_queue(queue, attempts)
    assert queue[0]["status"] == "attempted_timeout"
    assert queue[0]["attempted"] is True
    assert queue[0]["last_stage"] == "partition_search"
    assert queue[0]["elapsed_s"] == 900.1
    assert queue[0]["cache_key"] == "abc"
    assert queue[0]["status"] != "not_dispatched"


def test_debug_pack_cli_includes_native_analysis_and_clamps_old_timestamp(tmp_path: Path) -> None:
    import subprocess
    import sys

    run = tmp_path / "eval"
    native = run / "native_producers" / "hailo8" / "analysis_tables"
    native.mkdir(parents=True)
    diagnostic = native / "native_fifo_eval_runner__b052.json"
    diagnostic.write_text(json.dumps({"rows": [{"ok": False, "failure_reason": "test"}]}), encoding="utf-8")
    os.utime(diagnostic, (0, 0))
    (run / "reports" / "native_validation").mkdir(parents=True)
    (run / "reports" / "native_validation" / "native_producer_validation_summary.json").write_text("{}", encoding="utf-8")
    (run / "run_manifest.json").write_text(json.dumps({
        "schema": "onnx-splitpoint/evaluation-run-manifest",
        "schema_version": 1,
        "run_id": run.name,
        "status": "partial",
    }), encoding="utf-8")
    out = tmp_path / "debug.zip"
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "create_evaluation_debug_pack.py"), "--eval-run-dir", str(run), "--out", str(out)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=ROOT,
    )
    assert proc.returncode == 0, proc.stderr
    with zipfile.ZipFile(out) as zf:
        arc = "native_producers/hailo8/analysis_tables/native_fifo_eval_runner__b052.json"
        assert arc in zf.namelist()
        assert zf.getinfo(arc).date_time[0] == 1980
        manifest = json.loads(zf.read("debug_pack_manifest.json"))
        assert manifest["schema_version"] >= 5
        assert manifest["native_analysis_file_count"] >= 1


def _load_script_module(name: str, path: Path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_detection_contract_metadata_is_a_hard_gate(tmp_path: Path) -> None:
    mod = _load_script_module("native_validate_v60u", ROOT / "scripts/native_producer_validate_visualize.py")
    manifest = tmp_path / "output_manifest.json"
    manifest.write_text(json.dumps(_attested_decoded_contract()), encoding="utf-8")
    good = {
        "ok": True,
        "semantic_ok": True,
        "semantic_available": True,
        "decode_mode": "output0:nms:xyxy_score_class",
        "best": {"native_mode": "output0:nms:xyxy_score_class", "full_mode": "output0:nms:xyxy_score_class"},
    }
    accepted = mod._enforce_detection_contract(dict(good), manifest)
    assert accepted["contract_family_match"] is True
    assert accepted["ok"] is True

    bad = dict(good)
    bad["decode_mode"] = "output0:raw"
    bad["best"] = {"native_mode": "output0:raw", "full_mode": "output0:nms:xyxy_score_class"}
    rejected = mod._enforce_detection_contract(bad, manifest)
    assert rejected["contract_family_match"] is False
    assert rejected["diagnosis"] == "detection_contract_family_mismatch"
    assert rejected["semantic_available"] is False


def test_detection_contract_without_metadata_is_unavailable_not_claimable(tmp_path: Path) -> None:
    mod = _load_script_module("native_validate_v60u_unknown", ROOT / "scripts/native_producer_validate_visualize.py")
    manifest = tmp_path / "output_manifest.json"
    manifest.write_text(json.dumps({"outputs": [{"name": "output0"}]}), encoding="utf-8")
    row = {
        "ok": True,
        "semantic_ok": True,
        "semantic_available": True,
        "decode_mode": "output0:raw",
        "best": {"native_mode": "output0:raw", "full_mode": "output0:raw"},
    }
    result = mod._enforce_detection_contract(row, manifest)
    assert result["diagnosis"] == "detection_contract_metadata_unavailable"
    assert result["semantic_available"] is False
    assert result["ok"] is False


def test_classification_preprocessing_prefers_archived_contract(tmp_path: Path) -> None:
    mod = _load_script_module("native_validate_v60u_cls", ROOT / "scripts/native_producer_validate_visualize.py")
    boundary = tmp_path / "native_fifo_boundary_manifest.json"
    boundary.write_text(json.dumps({"preprocess": {"normalization_profile": "imagenet"}}), encoding="utf-8")
    profile, source = mod._classification_preprocess_profile(boundary, tmp_path / "model.onnx", tmp_path)
    assert profile == "imagenet"
    assert source == "boundary_manifest"


def test_cache_only_hef_miss_defers_before_sdk_import(tmp_path: Path, monkeypatch) -> None:
    from onnx_splitpoint_tool.hailo_backend import _hailo_build_hef_legacy

    model = tmp_path / "model.onnx"
    model.write_bytes(b"not-a-real-onnx-but-stable-for-cache-identity")
    monkeypatch.setenv("ONNX_SPLITPOINT_HAILO_CACHE_ROOT", str(tmp_path / "cache"))
    monkeypatch.setenv("ONNX_SPLITPOINT_HAILO_CACHE_ENABLED", "1")
    result = _hailo_build_hef_legacy(
        model,
        outdir=tmp_path / "out",
        hw_arch="hailo8",
        net_input_shapes=[1, 3, 224, 224],
        fixup=False,
        cache_only=True,
        calib_count=8,
        task="classification",
    )
    assert result.ok is False
    assert result.skipped is True
    assert result.failure_kind == "deferred_cold_full_cache_miss"
    assert result.last_stage == "cache_lookup"


def test_deferred_run_is_not_a_matrix_failure() -> None:
    from onnx_splitpoint_tool.workflow.runner import _selected_run_completeness_v60r

    result = _selected_run_completeness_v60r(
        benchmark_plan={
            "runs": [
                {"id": "hailo8", "required": False, "deferred": True, "deferred_reason": "cold_build_cache_miss"},
                {"id": "ort_cpu", "required": True},
            ]
        },
        source_records=[{"path": "benchmark_results_ort_cpu_auto.json", "tag": "ort_cpu_auto", "row_count": 1}],
    )
    assert result["matrix_complete"] is True
    assert result["deferred_run_count"] == 1
    assert result["missing_selected_run_ids"] == []


def test_targeted_smoke_profile_generator_covers_priority_2(tmp_path: Path) -> None:
    import yaml
    from onnx_splitpoint_tool.v60u_targeted_smokes import SCENARIOS, generate_profiles

    source = tmp_path / "profile.yaml"
    source.write_text(
        yaml.safe_dump({
            "name": "demo",
            "model_suite": {"primary": [{"id": "m", "task": "detection"}]},
            "selection_policy": {"max_accepted_cases_per_model": 1},
            "run_profiles": [{"id": "deepx_m1_to_tensorrt"}],
        }, sort_keys=False),
        encoding="utf-8",
    )
    out = tmp_path / "derived"
    manifest = generate_profiles(source, out, SCENARIOS)
    assert len(manifest["generated"]) == 5
    cross = yaml.safe_load((out / "profile_v60u_cross_runner.yaml").read_text(encoding="utf-8"))
    assert cross["selection_policy"]["max_accepted_cases_per_model"] >= 3
    assert cross["execution_preset"]["snapshot"]["ranking"]["enabled"] is True
    energy = yaml.safe_load((out / "profile_v60u_native_energy.yaml").read_text(encoding="utf-8"))
    assert energy["execution_preset"]["overrides"]["energy_enabled"] is True
    assert energy["execution_preset"]["snapshot"]["energy"]["native_mode"] == "measure"
    cold = yaml.safe_load((out / "profile_v60u_parallel_cold_build.yaml").read_text(encoding="utf-8"))
    assert cold["execution_preset"]["snapshot"]["build"]["scheduler"]["max_workers"] >= 3
    assert cold["execution_preset"]["snapshot"]["build"]["hailo"]["full_baseline_cold_build_policy"] == "build_missing"
