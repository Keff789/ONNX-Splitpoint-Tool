#!/usr/bin/env bash
# Hardware-independent acceptance for the v2.79.26 DeepX Full and diagnostics fixes.

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  printf '%s\n' \
    'SKIP: run_v27926_small_acceptance.sh was sourced; run it with bash.' >&2
  return 0
fi

set -Eeuo pipefail
SOURCE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
if [[ -n "${PY:-}" ]]; then
  PYTHON="$PY"
elif [[ -x "$SOURCE_ROOT/.venv/bin/python" ]]; then
  PYTHON="$SOURCE_ROOT/.venv/bin/python"
else
  PYTHON=python3
fi

REPORT_REQUEST="${V27926_ACCEPTANCE_REPORT:-}"
while (( $# > 0 )); do
  case "$1" in
    --report)
      (( $# >= 2 )) || { echo 'ERROR: --report requires a path.' >&2; exit 64; }
      REPORT_REQUEST="$2"
      shift 2
      ;;
    --help|-h)
      printf '%s\n' \
        'Usage: bash scripts/run_v27926_small_acceptance.sh [--report PATH]' \
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

REPORT="${REPORT_REQUEST:-${TMPDIR:-/tmp}/onnx-splitpoint-v27926-small-acceptance-$$.json}"
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

PYCACHE_ROOT="$(/usr/bin/mktemp -d /tmp/onnx-v27926-acceptance-pycache.XXXXXXXXXX)"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="$PYCACHE_ROOT"
cd -- "$SOURCE_ROOT"
export PYTHONPATH="$SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
STARTED="$($PYTHON -B - <<'PY'
from datetime import datetime, timezone
print(datetime.now(timezone.utc).isoformat())
PY
)"
SMOKE=NOT_RUN
PYTEST=NOT_RUN
MANIFEST=NOT_RUN
COMPILE=NOT_RUN
SHELL=NOT_RUN

write_report() {
  local final_rc="$1"
  REPORT="$REPORT" STARTED="$STARTED" FINAL_RC="$final_rc" \
  SMOKE="$SMOKE" PYTEST="$PYTEST" MANIFEST="$MANIFEST" \
  COMPILE="$COMPILE" SHELL="$SHELL" \
  "$PYTHON" -B - <<'PY'
from datetime import datetime, timezone
from pathlib import Path
import json
import os

path = Path(os.environ["REPORT"])
checks = {
    "release_smoke": os.environ["SMOKE"],
    "targeted_pytest": os.environ["PYTEST"],
    "installed_source_manifest": os.environ["MANIFEST"],
    "compileall": os.environ["COMPILE"],
    "shell_syntax": os.environ["SHELL"],
}
requested_rc = int(os.environ["FINAL_RC"])
all_pass = all(value == "PASS" for value in checks.values())
rc = requested_rc if requested_rc != 0 else (0 if all_pass else 70)
payload = {
    "schema": "onnx-splitpoint/v27926-small-acceptance/v1",
    "schema_version": 1,
    "version": "2.79.26",
    "build_id": "v2.79.26-deepx-full-diagnostics-fix",
    "release_scope": "deepx_full_diagnostics_fix",
    "started_at_utc": os.environ["STARTED"],
    "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    "status": "PASS" if rc == 0 and all_pass else "FAIL",
    "return_code": rc,
    "hardware_execution": "not_run_in_offline_gate",
    "real_seven_model_cache_preflight_status": "NOT_RUN",
    "real_evalrun_status": "NOT_RUN",
    "checks": checks,
}
path.parent.mkdir(parents=True, exist_ok=True)
tmp = path.with_name(f".{path.name}.tmp")
tmp.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
tmp.replace(path)
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
    /tmp/onnx-v27926-acceptance-pycache.??????????)
      if [[ -d "$PYCACHE_ROOT" && ! -L "$PYCACHE_ROOT" ]]; then
        /bin/rm -r -- "$PYCACHE_ROOT" || cleanup_rc=$?
      else
        cleanup_rc=70
      fi
      ;;
    *) cleanup_rc=70 ;;
  esac
  printf 'V27926_ACCEPTANCE_REPORT=%s\n' "$REPORT"
  if [[ "$rc" -eq 0 && "$report_rc" -ne 0 ]]; then rc="$report_rc"; fi
  if [[ "$rc" -eq 0 && "$cleanup_rc" -ne 0 ]]; then rc="$cleanup_rc"; fi
  exit "$rc"
}
trap finish EXIT

SMOKE=FAIL
"$PYTHON" -B -m onnx_splitpoint_tool.v27926_smoke
SMOKE=PASS

PYTEST=FAIL
# This GUI-import regression initializes TkAgg before pyplot. Run it in a
# fresh process so headless diagnostic/report imports in other tests cannot
# select a conflicting backend first. Both subprocesses must pass.
"$PYTHON" -B -m pytest -q -ra -p no:cacheprovider \
  tests/test_v27912_energy_setup_claim_admission.py::test_gui_energy_matrix_preflight_blocks_primary_workload
"$PYTHON" -B -m pytest -q -ra -p no:cacheprovider \
  -k 'not test_gui_energy_matrix_preflight_blocks_primary_workload' \
  tests/test_v27926_release_closure.py \
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
  scripts/run_v27926_small_acceptance.sh \
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

echo 'PASS v2.79.26 small acceptance (real cache preflight and hardware EvalRun: NOT_RUN)'
