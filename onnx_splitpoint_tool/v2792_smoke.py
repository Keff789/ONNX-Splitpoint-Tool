"""Hardware-independent release smoke for version 2.79.2."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from . import (
    __build_features__, __build_id__, __development_lineage__,
    __release__, __version__,
)
from .build_scheduler import semantic_result_status
from .hailo_backend import _resolve_hef_timeout_policy
from .native_three_stage import (
    ADAPTER_CLASSIFICATION_LOGITS, ADAPTER_YOLO11_DFL16,
    ADAPTER_YOLO26_DECODED, ADAPTER_YOLOV7_SPARSE,
    COMPLETED_DETECTION_ENDPOINT, P2_OUTPUT_ENDPOINT,
)
from .ranking_methods import RANKING_METHOD_IMPLEMENTATION, WORKFLOW_RANKING_METHOD
from .validation.detection_records import (
    DetectionRecordError, parse_detection_record,
)
from .v278_smoke import REQUIRED_FEATURES as V278_REQUIRED_FEATURES
from .workflow.central_quality_join import join_quality_results_by_request_sha
from .workflow.evidence_state_model import (
    project_evidence_state, summarize_evidence_states,
)
from .workflow.logical_measurement import annotate_logical_measurements
from .workflow.phase_eta import PhaseEtaEstimator
from .workflow.required_run_scope import (
    merge_authoritative_runs, seal_required_run_scope,
)
from .workflow.runner import WORKFLOW_VERSION

VERSION = "2.79.2"
LINEAGE = "v2.79"
BUILD_ID = "v2.79.2-native-concurrent-three-stage-b500-evidence-reconciliation"
NEW_FEATURES = {
    "native_three_stage_p1_p2_postprocess",
    "native_p2_output_primary_endpoint",
    "native_completed_detection_application_endpoint",
    "contract_bound_postprocess_adapter_registry",
    "quality_oracle_outside_performance_timing",
    "no_per_frame_crypto_in_native_hotloop",
    "semantic_venv_interpreter_admission",
    "canonical_hailo10_to_tensorrt_profile_id",
    "native_dual_endpoint_quality_binding_both_children",
    "native_yolov7_concurrent_three_stage_normal_runner",
    "native_concurrent_three_stage_single_invocation",
    "immutable_required_run_scope_before_compiler_dispatch",
    "logical_measurement_mirror_provenance",
    "exact_request_sha_central_quality_join",
    "quality_applicability_state_model",
    "deepx_detection_record_schema_strict_parser",
    "hailo_immutable_attempt_receipts",
    "hailo_hard_timeout_explicit_disable",
    "semantic_build_scheduler_outcomes",
    "phase_local_cohort_eta_ranges",
    "atomic_periodic_launcher_status",
    "read_only_existing_run_reconciliation",
}
REQUIRED_FEATURES = set(V278_REQUIRED_FEATURES) | NEW_FEATURES


def _expect_error(fn) -> bool:
    try:
        fn()
    except (DetectionRecordError, RuntimeError, ValueError):
        return True
    return False


def _runtime_checks(root: Path) -> dict[str, bool]:
    request = "a" * 64
    dxnn = "b" * 64
    rows = annotate_logical_measurements([
        {
            "model_id": "yolov7_paper", "case_id": "full",
            "run_id": "deepx_m1_full", "source_run_id": "deepx_m1_full",
            "backend": "deepx_m1", "variant": "full", "task": "detection",
            "setup_id": "orin_nx_deepx_m1_01",
            "source_request_sha256": request, "dxnn_sha256": dxnn,
            "pipeline_fps_selected": 4.0, "source_path": "direct.json",
        },
        {
            "model_id": "yolov7_paper", "case_id": "full",
            "run_id": "deepx_m1_full", "source_run_id": "deepx_m1_full",
            "backend": "deepx_m1", "variant": "full", "task": "detection",
            "setup_id": "", "source_request_sha256": request,
            "dxnn_sha256": dxnn, "source_path": "mirror.json",
        },
    ])
    logical_ids = {row.get("logical_measurement_id") for row in rows}
    join = join_quality_results_by_request_sha(
        rows=rows,
        results=[{
            "model_id": "yolov7_paper", "case_id": "full",
            "source_run_id": "deepx_m1_full", "variant": "full",
            "setup_id": "orin_nx_deepx_m1_01",
            "source_request_sha256": request,
        }],
    )
    direct = rows[join["joins"][0]["row_index"]] if join["joins"] else {}

    p2 = project_evidence_state({
        "presence": "present", "execution_ok": True,
        "measurement_endpoint": "p2_output", "variant": "composed",
    })
    done = project_evidence_state({
        "presence": "present", "execution_ok": True,
        "measurement_endpoint": "completed_detection", "variant": "composed",
        "central_quality_technical_status": "completed",
        "central_quality_decision": "pass",
    })
    state_summary = summarize_evidence_states([p2, done])

    canonical = parse_detection_record({
        "x1": 1, "y1": 2, "x2": 3, "y2": 4,
        "score": 0.8, "class_id": 1,
    })
    legacy = parse_detection_record({
        "box_xyxy": [1, 2, 3, 4], "confidence": 0.8,
        "class_id": 1,
    })

    eta = PhaseEtaEstimator(
        phase="quality", warmup_completions=3, parallelism=4,
    )
    eta.set_baseline(342, now=0.0)
    eta.observe_completion("detection", now=1.0, duration_s=100.0)
    eta.observe_completion("detection", now=2.0, duration_s=120.0)
    eta_warm = eta.estimate(remaining_by_cohort={"detection": 10})
    eta.observe_completion("detection", now=3.0, duration_s=110.0)
    eta_ready = eta.estimate(remaining_by_cohort={"detection": 10})

    projected = merge_authoritative_runs(
        {"planned_runs": [{"id": "hailo10"}]},
        ["hailo8", "hailo10_to_tensorrt"],
    )
    with tempfile.TemporaryDirectory(prefix="osp-v2792-smoke-") as temporary:
        scope_path = Path(temporary) / "required_run_scope.json"
        sealed = seal_required_run_scope(scope_path, {
            "schema": "onnx-splitpoint/required-run-scope",
            "schema_version": 2,
            "scope_level": "run",
            "created_at": "one",
            "requested_run_ids": ["hailo8", "hailo10"],
        })
        resumed = seal_required_run_scope(scope_path, {
            "schema": "onnx-splitpoint/required-run-scope",
            "schema_version": 2,
            "scope_level": "run",
            "created_at": "two",
            "requested_run_ids": ["hailo8", "hailo10"],
        })
        scope_ok = sealed["scope_sha256"] == resumed["scope_sha256"]

    runner = (root / "scripts/native_hailo_trt_fifo_from_benchmarkset.py").read_text(encoding="utf-8")
    concurrent = (root / "scripts/native_hailo_trt_concurrent_three_stage_from_benchmarkset.py").read_text(encoding="utf-8")
    launcher = (root / "scripts/run_v2792_seven_model_long_overnight.sh").read_text(encoding="utf-8")
    updater = (root / "scripts/update_source_release.sh").read_text(encoding="utf-8")
    profile = (root / "profiles/complete_set_7models_v2792_b500_audit20.yaml").read_text(encoding="utf-8")

    hard_off, idle = _resolve_hef_timeout_policy("off")
    hard_zero, _ = _resolve_hef_timeout_policy(0)

    return {
        "mirror_group": len(logical_ids) == 1 and any(
            row.get("representation_role") == "setup_less_mirror" for row in rows
        ),
        "request_sha_join": (
            join["matched_primary_count"] == 1
            and direct.get("setup_id") == "orin_nx_deepx_m1_01"
            and join["unmatched_count"] == 0
            and join["ambiguous_count"] == 0
        ),
        "state_model": (
            state_summary["matrix_required"] == 2
            and state_summary["quality_not_applicable"] == 1
            and state_summary["quality_completed"] == 1
        ),
        "detection_parser": (
            canonical["source_schema"] == "canonical_xyxy_score"
            and legacy["source_schema"] == "legacy_box_xyxy_confidence"
            and _expect_error(lambda: parse_detection_record({
                "x1": 1, "y1": 2, "score": 0.2, "class_id": 1,
            }))
            and _expect_error(lambda: parse_detection_record({
                "box_xyxy": [1, 2, 3, 4], "confidence": float("nan"),
                "class_id": 1,
            }))
        ),
        "scope_projection": (
            {row.get("id") for row in projected.get("runs", [])}
            == {"hailo10", "hailo8", "hailo10_to_tensorrt"}
            and scope_ok
        ),
        "scheduler_semantics": (
            semantic_result_status({"ok": False}) == "failed"
            and semantic_result_status({"timed_out": True}) == "timeout"
            and semantic_result_status({"unsupported_reason": "op"}) == "unsupported"
        ),
        "eta": (
            eta_warm["display"] == "ETA=UNAVAILABLE"
            and eta_ready["status"] == "available"
            and eta_ready["lower_s"] < eta_ready["upper_s"]
        ),
        "hailo_timeout": hard_off == 0 and hard_zero == 0 and idle is None,
        "concurrent_dispatch": (
            "_v2791_use_concurrent_three_stage" in runner
            and "native_hailo_trt_concurrent_three_stage_from_benchmarkset.py" in runner
            and "concurrent_three_stage_single_invocation" in concurrent
            and "three_stage_concurrency_directly_measured" in concurrent
        ),
        "launcher": (
            "V2792_RELEASE_IDENTITY=PASS" in launcher
            and "launcher_status_v2792.py" in launcher
            and "hailo10_to_tensorrt" in launcher
            and '"hailo10_to_trt"' not in launcher
            and "--preflight-only" in launcher
        ),
        "profile": all(marker in profile for marker in (
            "seal_before_first_compiler_dispatch: true",
            "model_local_plan_may_shrink_scope: false",
            "immutable_attempt_receipts: true",
            "terminal_attempt_selection: last_attempt_even_on_failure",
        )),
        "updater": (
            "--expected-version 2.79.2" in updater
            and "onnx-splitpoint-smoke-v2792" in updater
        ),
    }


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    checks = {
        "version": __version__ == VERSION and __release__ == VERSION,
        "lineage": __development_lineage__ == LINEAGE,
        "build": __build_id__ == BUILD_ID and WORKFLOW_VERSION == BUILD_ID,
        "features": REQUIRED_FEATURES.issubset(set(__build_features__)),
        "ranker_frozen": (
            WORKFLOW_RANKING_METHOD == "cut_bytes_only"
            and RANKING_METHOD_IMPLEMENTATION
            == "v277-cut-bytes-only-workflow-freeze-1"
        ),
        "endpoints": (
            P2_OUTPUT_ENDPOINT, COMPLETED_DETECTION_ENDPOINT
        ) == ("p2_output", "completed_detection"),
        "adapters": {
            ADAPTER_CLASSIFICATION_LOGITS, ADAPTER_YOLOV7_SPARSE,
            ADAPTER_YOLO26_DECODED, ADAPTER_YOLO11_DFL16,
        } == {
            "classification_logits_noop", "yolov7_anchor_multiscale_sparse",
            "yolo26_decoded_nms_materialize", "yolo11_regcls_dfl16",
        },
        "entrypoints": all(marker in pyproject for marker in (
            'version = "2.79.2"',
            'onnx-splitpoint-smoke-v2792 = "onnx_splitpoint_tool.v2792_smoke:main"',
            'onnx-splitpoint-smoke-v2-79-2 = "onnx_splitpoint_tool.v2792_smoke:main"',
        )),
        "files": all((root / name).is_file() for name in (
            "scripts/run_v2792_small_acceptance.sh",
            "scripts/run_v2792_seven_model_long_overnight.sh",
            "scripts/launcher_status_v2792.py",
            "scripts/reconcile_existing_run_v2792.py",
            "onnx_splitpoint_tool/workflow/required_run_scope.py",
            "onnx_splitpoint_tool/workflow/logical_measurement.py",
            "onnx_splitpoint_tool/workflow/central_quality_join.py",
            "onnx_splitpoint_tool/workflow/evidence_state_model.py",
            "onnx_splitpoint_tool/hailo_attempt_receipts.py",
            "onnx_splitpoint_tool/validation/detection_records.py",
            "profiles/complete_set_7models_v2792_b500_audit20.yaml",
        )),
    }
    checks.update(_runtime_checks(root))
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        print(json.dumps({"status": "FAIL", "failed": failed, "checks": checks}, indent=2))
        return 1
    print(json.dumps({"status": "PASS", "version": VERSION, "checks": checks}, indent=2))
    print("PASS v2.79.2 smoke")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
