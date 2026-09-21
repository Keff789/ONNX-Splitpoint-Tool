#!/usr/bin/env bash
# Hardware-independent acceptance for v2.83 cold compiler readiness, shared Native input, Hailo10 layout and terminal status.

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  printf '%s\n' \
    'SKIP: run_v282_small_acceptance.sh was sourced; run it with bash.' >&2
  return 0
fi

set -Eeuo pipefail
export ORT_DISABLE_TELEMETRY=1
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTEST_ADDOPTS='' CUDA_VISIBLE_DEVICES=''
SOURCE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
if [[ -n "${PY:-}" ]]; then
  PYTHON="$PY"
elif [[ -x "$SOURCE_ROOT/.venv/bin/python" ]]; then
  PYTHON="$SOURCE_ROOT/.venv/bin/python"
else
  PYTHON=python3
fi

REPORT_REQUEST="${V282_ACCEPTANCE_REPORT:-}"
while (( $# > 0 )); do
  case "$1" in
    --report)
      (( $# >= 2 )) || { echo 'ERROR: --report requires a path.' >&2; exit 64; }
      REPORT_REQUEST="$2"
      shift 2
      ;;
    --help|-h)
      printf '%s\n' \
        'Usage: bash scripts/run_v282_small_acceptance.sh [--report PATH]' \
        '' \
        'Checks runtime fixes, calibration, cache reuse and negative evidence without hardware.'
      exit 0
      ;;
    *)
      printf 'ERROR: unsupported option: %s\n' "$1" >&2
      exit 64
      ;;
  esac
done

REPORT="${REPORT_REQUEST:-${TMPDIR:-/tmp}/onnx-splitpoint-v282-small-acceptance-$$.json}"
REPORT="$($PYTHON -B - "$REPORT" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).expanduser().resolve(strict=False))
PY
)"
case "$REPORT/" in
  "$SOURCE_ROOT/"*)
    echo 'ERROR: acceptance report must be outside the source tree.' >&2
    exit 64
    ;;
esac

PYTEST_JUNIT_MAIN="${REPORT%.json}.pytest.xml"
PYTEST_JUNIT_GUI="${REPORT%.json}.gui_pytest.xml"
mkdir -p -- "$(dirname -- "$REPORT")"
PYCACHE_ROOT="$(/usr/bin/mktemp -d /tmp/onnx-v282-acceptance-pycache.XXXXXXXXXX)"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="$PYCACHE_ROOT"
cd -- "$SOURCE_ROOT"
export PYTHONPATH="$SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
STARTED="$($PYTHON -B - <<'PY'
from datetime import datetime, timezone
print(datetime.now(timezone.utc).isoformat())
PY
)"
DEPENDENCIES=NOT_RUN
SMOKE=NOT_RUN
PYTEST=NOT_RUN
MANIFEST=NOT_RUN
COMPILE=NOT_RUN
SHELL=NOT_RUN

write_report() {
  local final_rc="$1"
  REPORT="$REPORT" STARTED="$STARTED" FINAL_RC="$final_rc" \
  DEPENDENCIES="$DEPENDENCIES" SMOKE="$SMOKE" PYTEST="$PYTEST" MANIFEST="$MANIFEST" \
  COMPILE="$COMPILE" SHELL="$SHELL" \
  PYTEST_JUNIT_MAIN="$PYTEST_JUNIT_MAIN" PYTEST_JUNIT_GUI="$PYTEST_JUNIT_GUI" \
  "$PYTHON" -B - <<'PY'
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import xml.etree.ElementTree as ET

path = Path(os.environ["REPORT"])
checks = {
    "acceptance_environment": os.environ.get("DEPENDENCIES", "PASS"),
    "release_smoke": os.environ["SMOKE"],
    "targeted_pytest": os.environ["PYTEST"],
    "installed_source_manifest": os.environ["MANIFEST"],
    "compileall": os.environ["COMPILE"],
    "shell_syntax": os.environ["SHELL"],
}
required_modules = tuple(sorted(p.stem for p in Path('tests').glob('test_v282_*.py')))
test_runs = []
executed_modules = set()
for name, key in (("main", "PYTEST_JUNIT_MAIN"), ("gui_separate_process", "PYTEST_JUNIT_GUI")):
    xml_path = Path(os.environ[key])
    entry = {"name": name, "junit_path": str(xml_path), "status": "NOT_RUN"}
    if xml_path.is_file():
        try:
            xml_root = ET.parse(xml_path).getroot()
            suites = list(xml_root.iter("testsuite"))
            if name == "main":
                executed_modules.update(row.get("classname", "").split(".")[1] for row in xml_root.iter("testcase") if row.get("classname", "").startswith("tests.test_"))
            counts = {field: sum(int(suite.get(field, "0")) for suite in suites) for field in ("tests", "failures", "errors", "skipped")}
            counts["passed"] = counts["tests"] - counts["failures"] - counts["errors"] - counts["skipped"]
            entry.update(counts, status="PASS" if not counts["failures"] and not counts["errors"] and not counts["skipped"] else "FAIL")
        except Exception as exc:
            entry.update(status="INVALID_REPORT", error=str(exc))
    test_runs.append(entry)
requested_rc = int(os.environ["FINAL_RC"])
missing_required_modules = sorted(set(required_modules) - executed_modules)
required_tests_complete = (all(run["status"] == "PASS" and run.get("tests", 0) > 0 for run in test_runs) and not missing_required_modules)
if not required_tests_complete:
    checks["targeted_pytest"] = "FAIL"
all_pass = all(value == "PASS" for value in checks.values()) and required_tests_complete
rc = requested_rc if requested_rc != 0 else (0 if all_pass else 70)
payload = {
    "schema": "onnx-splitpoint/v282-small-acceptance/v1",
    "schema_version": 1,
    "version": "2.83",
    "build_id": "v2.83-r9b-request-latency",
    "release_scope": "selected_energy_generic_roles_workspace_product_evidence",
    "started_at_utc": os.environ["STARTED"],
    "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    "status": "PASS" if rc == 0 and all_pass else "environment_blocked" if os.environ.get("DEPENDENCIES") == "BLOCKED" else "FAIL",
    "return_code": rc,
    "hardware_execution": "not_run_in_offline_gate",
    "real_seven_model_cache_preflight_status": "NOT_RUN",
    "real_evalrun_status": "NOT_RUN",
    "checks": checks,
    "pytest_runs": test_runs,
    "required_new_modules": list(required_modules),
    "missing_required_new_modules": missing_required_modules,
    "required_test_policy": "all_selected_tests_required_no_skips_or_xfails",
    "pytest_totals": {field: sum(run.get(field, 0) for run in test_runs) for field in ("tests", "passed", "failures", "errors", "skipped")},
}
path.parent.mkdir(parents=True, exist_ok=True)
tmp = path.with_name(f".{path.name}.tmp")
tmp.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
tmp.replace(path)
raise SystemExit(rc)
PY
}

finish() {
  local rc=$?
  local report_rc=0
  local cleanup_rc=0
  local checks_complete=1
  trap - EXIT
  set +e
  for check_state in "$SMOKE" "$PYTEST" "$MANIFEST" "$COMPILE" "$SHELL"; do
    if [[ "$check_state" != PASS ]]; then
      checks_complete=0
      break
    fi
  done
  if [[ "$rc" -eq 0 && "$checks_complete" -ne 1 ]]; then rc=70; fi
  write_report "$rc"
  report_rc=$?
  case "$PYCACHE_ROOT" in
    /tmp/onnx-v282-acceptance-pycache.??????????)
      if [[ -d "$PYCACHE_ROOT" && ! -L "$PYCACHE_ROOT" ]]; then
        /bin/rm -r -- "$PYCACHE_ROOT" || cleanup_rc=$?
      else
        cleanup_rc=70
      fi
      ;;
    *) cleanup_rc=70 ;;
  esac
  printf 'V282_ACCEPTANCE_REPORT=%s\n' "$REPORT"
  if [[ "$rc" -eq 0 && "$report_rc" -ne 0 ]]; then rc="$report_rc"; fi
  if [[ "$rc" -eq 0 && "$cleanup_rc" -ne 0 ]]; then rc="$cleanup_rc"; fi
  exit "$rc"
}
trap finish EXIT

DEPENDENCIES=BLOCKED
"$PYTHON" -B scripts/check_acceptance_environment.py
DEPENDENCIES=PASS

export V27928_REPLAY_NPZ="${V27928_REPLAY_NPZ:-${V282_REPLAY_NPZ:-}}"

SMOKE=FAIL
"$PYTHON" -B -m onnx_splitpoint_tool.v282_smoke
SMOKE=PASS

PYTEST=FAIL
# This GUI-import regression initializes TkAgg before pyplot. Run it in a
# fresh process so headless diagnostic/report imports in other tests cannot
# select a conflicting backend first. Both subprocesses must pass.
"$PYTHON" -B -m pytest -q -ra -o xfail_strict=true -p no:cacheprovider --junitxml="$PYTEST_JUNIT_GUI" \
  tests/test_v27912_energy_setup_claim_admission.py::test_gui_energy_matrix_preflight_blocks_primary_workload
"$PYTHON" -B -m pytest -q -ra -o xfail_strict=true -p no:cacheprovider --junitxml="$PYTEST_JUNIT_MAIN" \
  -k 'not test_gui_energy_matrix_preflight_blocks_primary_workload' \
  tests/test_v2801_cpu_reference_dispatch.py \
  tests/test_v2801_native_remote_closure.py \
  tests/test_v2801_reference_errors.py \
  tests/test_v2801_reference_debug_export.py \
  tests/test_v2801_workflow_integration.py \
  tests/test_v2801_release_scope.py \
  tests/test_v2803_deferred_build_readiness.py \
  tests/test_v2803_native_not_started.py \
  tests/test_v2803_debug_pack_large_sources.py \
  tests/test_v2803_runtime_diagnostic_projection.py \
  tests/test_v2803_reference_workflow_gate.py \
  tests/test_v2803_scope_and_diagnostic_claims.py \
  tests/test_v2803_measurement_configuration.py \
  tests/test_v2803_release_scope.py \
  tests/test_v2804_debug_limits.py \
  tests/test_v2804_gui_overlay.py \
  tests/test_v2804_hailo_claims.py \
  tests/test_v2804_hailo_manifest.py \
  tests/test_v2804_quality_cancel.py \
  tests/test_v2804_reuse_quality.py \
  tests/test_v2804_workflow_starter.py \
  tests/test_v281_compiler_selection.py \
  tests/test_v281_hailo10_boundary_layout.py \
  tests/test_v281_hailo8_shared_input.py \
  tests/test_v281_status_reports.py \
  tests/test_v281_status_review.py \
  tests/test_v281_workflow_acceptance.py \
  tests/test_v282_*.py \
  tests/test_v2803_fix2_deepx_geometry.py \
  tests/test_v2803_fix2_native_reporting.py \
  tests/test_v2803_fix2_reference_prepare.py \
  tests/test_v2803_fix2_trt_preflight.py \
  tests/test_v2803_fix3_hailo_artifact_scope.py \
  tests/test_v2803_fix3_service_debug_export.py \
  tests/test_v2803_fix4_reference_scope.py \
  tests/test_v2803_fix4_trt_build_guard.py \
  tests/test_v2803_fix4_trt_dispatch_scope.py \
  tests/test_v2803_fix5_native_asset_sync.py \
  tests/test_v2802_cpu_reference_binding.py \
  tests/test_v2802_native_quality_blocking.py \
  tests/test_v2802_dashboard_projection.py \
  tests/test_v2802_workflow_integration.py \
  tests/test_v2802_release_scope.py \
  tests/test_v280_hailo8_compute_env.py \
  tests/test_v280_runtime_cleanup.py \
  tests/test_v280_artifact_reuse.py \
  tests/test_v280_release_scope.py \
  tests/test_v27934_model_runtime_worker.py \
  tests/test_v2751_hailo_exact_artifact_store_restore.py \
  tests/test_v2772_hailo_parallel_canary_profile.py \
  tests/test_v27934_preparation_profile.py \
  tests/test_v27934_model_diagnostic.py \
  tests/test_v27934_hailo_backend.py \
  tests/test_v27934_config_workflow.py \
  tests/test_v27934_hailo8_dependency_plan.py \
  tests/test_v27934_force_defaults.py \
  tests/test_v27934_compiler_context.py \
  tests/test_v27933_force_admission.py \
  tests/test_v27933_generic_full_binding.py \
  tests/test_v27933_gui_configuration_start.py \
  tests/test_v27933_hailo_gpu_smoke.py \
  tests/test_v27933_original_force_evidence.py \
  tests/test_v27933_run_mode_config.py \
  tests/test_v27933_runtime_reuse.py \
  tests/test_v60n_run_modes.py \
  tests/test_v60p_smoke_warning_fixes.py \
  tests/test_v27529_final_quality_standard_path.py \
  tests/test_v269e_start_snapshot_backfill.py \
  tests/test_v2777_phase5_hailo_projection.py \
  tests/test_profile_editor_campaign_roundtrip_v261e.py \
  tests/test_v2735_standard_execution_path.py \
  tests/test_v27911_composed_provenance_binding.py \
  tests/test_v268_native_repetition_ui_plan.py \
  tests/test_v27545_large_audit_working_set_admission.py \
  tests/test_v270h_overnight_standard_repairs.py \
  tests/test_v2796_r3_r4_r5_product_path.py \
  tests/test_v27932_native_failure_persistence.py \
  tests/test_v27932_full_exception_identity.py \
  tests/test_v27932_compiler_precedence.py \
  tests/test_v27932_failure_chain_integration.py \
  tests/test_v276_quality_projection.py \
  tests/test_v27522_quality_ap75_replay.py \
  tests/test_v269f_native_split_final_energy_integrity.py \
  tests/test_v269b_deepx_run32_regressions.py \
  tests/test_v27931_replay_provenance.py \
  tests/test_v27931_detection_control.py \
  tests/test_v27931_classification_trt_control.py \
  tests/test_v27931_classification_diagnostic_workflow.py \
  tests/test_v27931_classification_evidence.py \
  tests/test_v27931_classification_routing.py \
  tests/test_v27931_compiler_routing.py \
  tests/test_v27931_energy_accounting.py \
  tests/test_v27931_experiment_scope.py \
  tests/test_v27931_hailo26_boundary_diagnostics.py \
  tests/test_v27931_hailo_manifest_energy.py \
  tests/test_v27931_native_identity_prerequisites.py \
  tests/test_v27931_quality_uncertainty.py \
  tests/test_v27930_deepx_semantic_pre_nms.py \
  tests/test_v27930_deepx_quality_pre_nms.py \
  tests/test_v27930_native_full_semantic_merge.py \
  tests/test_v27930_deepx_workflow_integration.py \
  tests/test_v27930_terminal_hash_uncached.py \
  tests/test_v27930_terminal_index_cost.py \
  tests/test_v27930_terminal_progress.py \
  tests/test_v27930_full_workflow_smoke.py \
  tests/test_v264_execution_plan.py \
  tests/test_v272_evidence_status_contract.py \
  tests/test_v272_evidence_status_reporting.py \
  tests/test_v27530_score_independent_native_ranking_audit.py \
  tests/test_v27916_energy_failure_isolation.py \
  tests/test_v269a_deepx_full_central_quality.py \
  tests/test_v269d_trt_central_quality_producer.py \
  tests/test_v269e_quality_export_vendored.py \
  tests/test_v60m_runtime_energy_integrity.py \
  tests/test_v2796_artifact_index_closure.py \
  tests/test_v2797_artifact_index_current_path.py \
  tests/test_v2798_artifact_index_current_path.py \
  tests/test_artifact_index_report_cleanup.py \
  tests/test_v27910_update_active_workflow_guard.py \
  tests/test_v27928_score_roundoff.py \
  tests/test_v27928_probe_staging.py \
  tests/test_v27928_probe_real_process.py \
  tests/test_v27929_no_opencv.py \
  tests/test_v27929_probe_real_process.py \
  tests/test_v27927_result_endpoints.py \
  tests/test_v27927_status_reporting.py \
  tests/test_v27927_deepx_input_transfer.py \
  tests/test_v27927_deepx_value_diagnostics.py \
  tests/test_v27927_probe_launcher.py \
  tests/test_v27927_probe_process_review.py \
  tests/test_v27926_deepx_full_decoded_pre_nms.py \
  tests/test_v27926_deepx_full_downstream.py \
  tests/test_v2711_direct_bn6_producer_binding.py \
  tests/test_v275_deepx_sealed_runtime_input.py \
  tests/test_v27539_remote_dispatch_propagation.py \
  tests/test_v27926_quality_submission_failure.py \
  tests/test_v27926_runtime_debug_pack.py \
  tests/test_compact_debug_pack.py \
  tests/test_v263_management_quality_service.py \
  tests/test_v27925_deepx_compiler_overlay.py \
  tests/test_v27925_hailo_full_bundle_paths.py \
  tests/test_v27925_energy_calibration_reporting.py \
  tests/test_v27925_fast_oracle_validation.py \
  tests/test_v27925_native_energy_raw_debug_pack.py \
  tests/test_v27924_terminal_hailo_aliases.py \
  tests/test_v27924_quality_reporting.py \
  tests/test_v27924_native_error_fixes.py \
  tests/test_v27924_trt_active_retention.py \
  tests/test_v27924_hailo_full_portability.py \
  tests/test_v27924_negative_preflight.py \
  tests/test_v27924_rank3_boundary_layout.py \
  tests/test_v27923_trt_legacy_reuse.py \
  tests/test_v27923_fs_calibration_gate.py \
  tests/test_v27923_decoded_completion.py \
  tests/test_v27923_native_failures.py \
  tests/test_v27923_deepx_preflight.py \
  tests/test_v27923_workflow_fixes.py \
  tests/test_v272_detection_completion_runtime.py \
  tests/test_v270i_p0_completed_endpoint.py \
  tests/test_v2777_yolo11_native_adapter.py \
  tests/test_v279_native_three_stage.py \
  tests/test_v2751_preprocessing_bridge.py \
  tests/test_v2741_native_energy_quality_projection.py \
  tests/test_v27916_full_system_input_calibration.py \
  tests/test_v27918_full_system_calibration_flow.py \
  tests/test_energy_command_window_binding_v2.py \
  tests/test_v27528_energy_inherited_runtime_binding.py \
  tests/test_v27912_energy_setup_claim_admission.py \
  tests/test_v2799_accelerator_idle_energy_semantics.py \
  tests/test_v270c_run39_native_repairs.py \
  tests/test_v272_detection_hotloop_integration.py \
  tests/test_v27916_hailo10_native_io.py \
  tests/test_v60y_native_storage_input_matrix_fixes.py::test_e2e_wrapper_uses_exact_image_and_clears_success_error \
  tests/test_v2736_variant_nested_ssh.py \
  tests/test_v27922_build_evidence_store.py \
  tests/test_v27922_negative_backend.py \
  tests/test_v27922_build_evidence_recovery.py \
  tests/test_v27922_negative_preflight.py \
  tests/test_v27922_build_context.py \
  tests/test_v27922_auxiliary_context.py \
  tests/test_v2783_build_evidence.py \
  tests/test_v2783_hailo8_first_feasibility.py \
  tests/test_v2783_hailo8_evidence_binding.py \
  tests/test_v27921_editable_rollback.py \
  tests/test_v27921_final_selection_cache_preflight.py \
  tests/test_v27921_remote_trt_part1_preflight.py \
  tests/test_v27921_artifact_store_bundles.py \
  tests/test_v27921_hailo_atomic_bundle.py \
  tests/test_v27921_cache_preflight_completeness.py \
  tests/test_v27921_deferred_builds.py \
  tests/test_v27921_hailo_bundle_integration.py \
  tests/test_v27921_hailo_script_discovery.py \
  tests/test_v27921_prepared_hailo_bundle.py \
  tests/test_v27920_hailo_deepx_reuse.py \
  tests/test_v27920_trt_builder_cache_reuse.py \
  tests/test_v27920_trt_role_cache.py \
  tests/test_v27920_trt_namespace_retention.py \
  tests/test_v27920_artifact_cache_preflight.py \
  tests/test_v27920_artifact_cache_preflight_profile_schema.py \
  tests/test_v27920_artifact_cache_preflight_runner.py \
  tests/test_v27920_remote_trt_cache_preflight.py \
  tests/test_v27920_job_barrier.py \
  tests/test_v2757_remote_cache_policy_closure.py \
  tests/test_v27510_smoke_cache_compatibility.py \
  tests/test_deepx_full_preprocessing_ab_contract.py
PYTEST=PASS

MANIFEST=FAIL
"$PYTHON" -B scripts/build_source_manifest.py \
  --root "$SOURCE_ROOT" --verify --scope installed
MANIFEST=PASS

COMPILE=FAIL
"$PYTHON" -B -m compileall -q onnx_splitpoint_tool scripts
COMPILE=PASS

SHELL=FAIL
for script in \
  scripts/run_v282_small_acceptance.sh \
  scripts/run_v282_short_tests.sh \
  scripts/run_hailo_gpu_compute_v281.sh \
  scripts/hailo_gpu_diagnostics/run_hailo_gpu_smoke.sh \
  scripts/run_classification_input_probe_v281.sh \
  scripts/run_hailo10_yolo26_boundary_probe_v281.sh \
  scripts/run_v27924_small_acceptance.sh \
  scripts/run_v27923_small_acceptance.sh \
  scripts/run_v27922_small_acceptance.sh \
  scripts/run_v27921_small_acceptance.sh \
  scripts/run_v27920_small_acceptance.sh \
  scripts/run_v27919_small_acceptance.sh \
  scripts/run_v279_small_acceptance.sh \
  scripts/run_local_acceptance.sh \
  scripts/update_source_release.sh
do
  bash -n "$script"
done
SHELL=PASS

echo 'PASS v2.83 small acceptance (real cache preflight and hardware EvalRun: NOT_RUN)'
