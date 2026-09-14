#!/usr/bin/env bash
# Hardware-independent v2.79.12 release-closure acceptance gate.

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  printf '%s\n' 'SKIP: run_v27912_small_acceptance.sh was sourced; run it with bash.' >&2
  return 0
fi

set -Eeuo pipefail
PYCACHE_ROOT="$(
  /usr/bin/mktemp -d /tmp/onnx-v27912-acceptance-pycache.XXXXXXXXXX
)" || exit 70
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="$PYCACHE_ROOT"
cleanup_pycache() {
  local rc=0
  case "${PYCACHE_ROOT:-}" in
    /tmp/onnx-v27912-acceptance-pycache.??????????)
      if [[ -L "$PYCACHE_ROOT" ]]; then
        /bin/rm -- "$PYCACHE_ROOT" || rc=70
      elif [[ -d "$PYCACHE_ROOT" ]]; then
        /bin/rm -r -- "$PYCACHE_ROOT" || rc=70
      elif [[ -e "$PYCACHE_ROOT" ]]; then
        rc=70
      fi
      ;;
    *) rc=70 ;;
  esac
  return "$rc"
}
cleanup_early_pycache() {
  local rc=$?
  local cleanup_rc=0
  trap - EXIT
  cleanup_pycache || cleanup_rc=$?
  if [[ "$rc" -eq 0 && "$cleanup_rc" -ne 0 ]]; then rc="$cleanup_rc"; fi
  exit "$rc"
}
trap cleanup_early_pycache EXIT
SOURCE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
if [[ -n "${PY:-}" ]]; then
  PYTHON="$PY"
elif [[ -x "$SOURCE_ROOT/.venv/bin/python" ]]; then
  PYTHON="$SOURCE_ROOT/.venv/bin/python"
else
  PYTHON=python3
fi

REPORT_REQUEST="${V27912_ACCEPTANCE_REPORT:-}"
while (( $# > 0 )); do
  case "$1" in
    --report)
      (( $# >= 2 )) || { echo 'ERROR: --report requires a path.' >&2; exit 64; }
      REPORT_REQUEST="$2"; shift 2 ;;
    --help|-h)
      printf '%s\n' \
        'Usage: bash scripts/run_v27912_small_acceptance.sh [--report PATH]' \
        '' \
        'Runs the hardware-independent v2.79.12 identity, platform-power,' \
        'energy-comparison, cancellation-evidence, scheduler, manifest and syntax gates.'
      exit 0 ;;
    *) printf 'ERROR: unsupported option: %s\n' "$1" >&2; exit 64 ;;
  esac
done

REPORT="${REPORT_REQUEST:-${TMPDIR:-/tmp}/onnx-splitpoint-v27912-small-acceptance-$$.json}"
REPORT="$($PYTHON -B - "$REPORT" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).expanduser().resolve(strict=False))
PY
)"
case "$REPORT/" in
  "$SOURCE_ROOT/"*) echo 'ERROR: acceptance report must be outside the source tree.' >&2; exit 64 ;;
esac

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

write_report() {
  local final_rc="$1"
  REPORT="$REPORT" STARTED="$STARTED" FINAL_RC="$final_rc" \
  SMOKE="$SMOKE" PYTEST="$PYTEST" MANIFEST="$MANIFEST" COMPILE="$COMPILE" \
  "$PYTHON" -B - <<'PY'
from datetime import datetime, timezone
from pathlib import Path
import json, os
path = Path(os.environ["REPORT"])
rc = int(os.environ["FINAL_RC"])
payload = {
    "schema": "onnx-splitpoint/v27912-small-acceptance/v1",
    "schema_version": 1,
    "version": "2.79.12",
    "build_id": "v2.79.12-platform-power-calibration-provenance-closure",
    "started_at_utc": os.environ["STARTED"],
    "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    "status": "PASS" if rc == 0 else "FAIL",
    "return_code": rc,
    "hardware_execution": "not_run_in_offline_gate",
    "checks": {
        "release_smoke": os.environ["SMOKE"],
        "targeted_pytest": os.environ["PYTEST"],
        "source_manifest": os.environ["MANIFEST"],
        "compileall": os.environ["COMPILE"],
    },
}
path.parent.mkdir(parents=True, exist_ok=True)
tmp = path.with_name(f".{path.name}.tmp")
tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
tmp.replace(path)
PY
}
finish_acceptance() {
  local rc=$?
  local cleanup_rc=0
  local report_rc=0
  trap - EXIT
  set +e
  write_report "$rc"
  report_rc=$?
  printf "V27912_ACCEPTANCE_REPORT=%s\n" "$REPORT"
  cleanup_pycache || cleanup_rc=$?
  if [[ "$rc" -eq 0 && "$report_rc" -ne 0 ]]; then
    rc="$report_rc"
  fi
  if [[ "$rc" -eq 0 && "$cleanup_rc" -ne 0 ]]; then
    rc="$cleanup_rc"
  fi
  exit "$rc"
}
trap finish_acceptance EXIT

SMOKE=FAIL
"$PYTHON" -B -m onnx_splitpoint_tool.v27912_smoke
SMOKE=PASS

PYTEST=FAIL
"$PYTHON" -B -m pytest -q \
  tests/test_v2799_platform_power.py \
  tests/test_v2799_platform_power_gui.py \
  tests/test_v2799_accelerator_idle_energy_semantics.py \
  tests/test_v27910_registry_atomic_calibration.py \
  tests/test_v27912_registry_m2_migration.py \
  tests/test_v27912_registry_stale_writer_cas.py \
  tests/test_v27912_energy_method_manifest_cli.py \
  tests/test_v27912_platform_power_calibration.py \
  tests/test_v27912_accelerator_idle_binding_v2.py \
  tests/test_v27912_energy_setup_claim_admission.py \
  tests/test_v27912_collector_registry_admission.py \
  tests/test_v27912_accelerator_env_registry_merge.py \
  tests/test_v27912_platform_power_gui_gate_evidence.py \
  tests/test_v27912_platform_power_udp_preflight.py \
  tests/test_v27912_bytecode_isolation.py \
  tests/test_v27912_update_trusted_preflight.py \
  tests/test_v27912_source_integrity.py \
  tests/test_v27911_energy_comparison_closure.py \
  tests/test_v27911_cancelled_diagnostic_collection.py \
  tests/test_v27911_composed_provenance_binding.py \
  tests/test_v27910_update_active_workflow_guard.py \
  tests/test_v27910_yolo11_gate_profile_schema.py \
  tests/test_v27910_yolo11_backend_bound_gate.py \
  tests/test_v27910_yolo11_recovery_manifest.py \
  tests/test_v27910_yolo11_gate_launcher.py \
  tests/test_v27910_yolov7_claim_gate_32.py \
  tests/test_v2796_three_stage_closure.py \
  tests/test_v27542_installed_source_manifest_scope.py::test_current_updater_offline_refresh_is_fail_closed_and_ordered \
  tests/test_v269d_management_trt_quality_summary.py::test_standard_final_missing_companion_postcondition_hard_fails_stage \
  tests/test_v2796_seven_model_overnight_launcher.py::test_v2796_status_helper_is_frozen_and_current_alias_tracks_v27913 \
  tests/test_v2798_seven_model_overnight_launcher.py::test_v2798_status_helper_is_frozen_and_current_alias_tracks_v27913 \
  tests/test_v2798_release_provenance.py \
  tests/test_v2799_release_provenance.py \
  tests/test_v27910_release_provenance.py \
  tests/test_v27912_release_provenance.py \
  tests/test_v27912_seven_model_overnight_launcher.py \
  tests/test_v27912_yolov7_claim_gate_32.py \
  tests/test_v279_release_provenance.py
PYTEST=PASS

MANIFEST=FAIL
# The same acceptance script runs in pristine source trees and in updated
# installations. Installed scope remains byte-exact for release-owned files
# while explicitly allowing preserved user profiles; archive verification in
# the updater stays strict release scope.
"$PYTHON" -B scripts/build_source_manifest.py \
  --root "$SOURCE_ROOT" --verify --scope installed
MANIFEST=PASS

COMPILE=FAIL
"$PYTHON" -B -m compileall -q onnx_splitpoint_tool scripts
COMPILE=PASS

echo 'PASS v2.79.12 small acceptance'
exit 0
