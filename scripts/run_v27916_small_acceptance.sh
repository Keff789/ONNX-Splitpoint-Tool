#!/usr/bin/env bash
# Hardware-independent acceptance for the focused v2.79.16 repair.

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  printf '%s\n' 'SKIP: run_v27916_small_acceptance.sh was sourced; run it with bash.' >&2
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

REPORT_REQUEST="${V27916_ACCEPTANCE_REPORT:-}"
while (( $# > 0 )); do
  case "$1" in
    --report)
      (( $# >= 2 )) || { echo 'ERROR: --report requires a path.' >&2; exit 64; }
      REPORT_REQUEST="$2"
      shift 2
      ;;
    --help|-h)
      printf '%s\n' \
        'Usage: bash scripts/run_v27916_small_acceptance.sh [--report PATH]' \
        '' \
        'Checks the v2.79.16 calibration, Hailo-10 I/O and Energy isolation without hardware.'
      exit 0
      ;;
    *)
      printf 'ERROR: unsupported option: %s\n' "$1" >&2
      exit 64
      ;;
  esac
done

REPORT="${REPORT_REQUEST:-${TMPDIR:-/tmp}/onnx-splitpoint-v27916-small-acceptance-$$.json}"
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

PYCACHE_ROOT="$(/usr/bin/mktemp -d /tmp/onnx-v27916-acceptance-pycache.XXXXXXXXXX)"
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
  SMOKE="$SMOKE" PYTEST="$PYTEST" MANIFEST="$MANIFEST" COMPILE="$COMPILE" SHELL="$SHELL" \
  "$PYTHON" -B - <<'PY'
from datetime import datetime, timezone
from pathlib import Path
import json, os
path = Path(os.environ["REPORT"])
checks = {
    "release_smoke": os.environ["SMOKE"],
    "targeted_pytest": os.environ["PYTEST"],
    "source_manifest": os.environ["MANIFEST"],
    "compileall": os.environ["COMPILE"],
    "shell_syntax": os.environ["SHELL"],
}
requested_rc = int(os.environ["FINAL_RC"])
all_pass = all(value == "PASS" for value in checks.values())
rc = requested_rc if requested_rc != 0 else (0 if all_pass else 70)
payload = {
    "schema": "onnx-splitpoint/v27916-small-acceptance/v1",
    "schema_version": 1,
    "version": "2.79.16",
    "build_id": "v2.79.16-guided-full-system-input-calibration",
    "started_at_utc": os.environ["STARTED"],
    "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    "status": "PASS" if rc == 0 and all_pass else "FAIL",
    "return_code": rc,
    "hardware_execution": "not_run_in_offline_gate",
    "checks": checks,
}
path.parent.mkdir(parents=True, exist_ok=True)
tmp = path.with_name(f".{path.name}.tmp")
tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
  if [[ "$rc" -eq 0 && "$checks_complete" -ne 1 ]]; then
    rc=70
  fi
  write_report "$rc"
  report_rc=$?
  case "$PYCACHE_ROOT" in
    /tmp/onnx-v27916-acceptance-pycache.??????????)
      if [[ -d "$PYCACHE_ROOT" && ! -L "$PYCACHE_ROOT" ]]; then
        /bin/rm -r -- "$PYCACHE_ROOT" || cleanup_rc=$?
      else
        cleanup_rc=70
      fi
      ;;
    *) cleanup_rc=70 ;;
  esac
  printf 'V27916_ACCEPTANCE_REPORT=%s\n' "$REPORT"
  if [[ "$rc" -eq 0 && "$report_rc" -ne 0 ]]; then rc="$report_rc"; fi
  if [[ "$rc" -eq 0 && "$cleanup_rc" -ne 0 ]]; then rc="$cleanup_rc"; fi
  exit "$rc"
}
trap finish EXIT

SMOKE=FAIL
"$PYTHON" -B -m onnx_splitpoint_tool.v27916_smoke
SMOKE=PASS

PYTEST=FAIL
"$PYTHON" -B -m pytest -q -p no:cacheprovider \
  tests/test_v27916_full_system_input_calibration.py \
  tests/test_v27916_hailo10_native_io.py \
  tests/test_v27916_energy_failure_isolation.py \
  tests/test_v27916_energy_row_isolation.py \
  tests/test_v27916_release_provenance.py \
  tests/test_v2621_ab_remote_contract_fixes.py \
  tests/test_v2621_full_hailo_bound_runtime_input.py \
  tests/test_energy_final_contract_regressions.py \
  tests/test_v27914_simple_idle_calibration_comparison.py \
  tests/test_v27910_registry_atomic_calibration.py \
  tests/test_v27912_registry_m2_migration.py \
  tests/test_v27912_platform_power_udp_preflight.py \
  tests/test_v2799_accelerator_idle_energy_semantics.py \
  tests/test_v2621_split_energy_preflight.py \
  tests/test_v269f_hailo_trt_interface_contract.py \
  tests/test_v270d_targeted_console_smokes.py \
  tests/test_v27911_composed_provenance_binding.py
PYTEST=PASS

MANIFEST=FAIL
"$PYTHON" -B scripts/build_source_manifest.py \
  --root "$SOURCE_ROOT" --verify --scope installed
MANIFEST=PASS

COMPILE=FAIL
"$PYTHON" -B -m compileall -q onnx_splitpoint_tool scripts
COMPILE=PASS

SHELL=FAIL
bash -n scripts/run_v27916_small_acceptance.sh \
  scripts/run_v279_small_acceptance.sh \
  scripts/run_local_acceptance.sh \
  scripts/update_source_release.sh
SHELL=PASS

echo 'PASS v2.79.16 small acceptance'
