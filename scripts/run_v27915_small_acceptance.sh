#!/usr/bin/env bash
# Hardware-independent acceptance for the focused v2.79.15 repair.

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  printf '%s\n' 'SKIP: run_v27915_small_acceptance.sh was sourced; run it with bash.' >&2
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

REPORT_REQUEST="${V27915_ACCEPTANCE_REPORT:-}"
while (( $# > 0 )); do
  case "$1" in
    --report)
      (( $# >= 2 )) || { echo 'ERROR: --report requires a path.' >&2; exit 64; }
      REPORT_REQUEST="$2"
      shift 2
      ;;
    --help|-h)
      printf '%s\n' \
        'Usage: bash scripts/run_v27915_small_acceptance.sh [--report PATH]' \
        '' \
        'Checks the FS energy defaults and DeepX Part1 preprocessing without hardware.'
      exit 0
      ;;
    *)
      printf 'ERROR: unsupported option: %s\n' "$1" >&2
      exit 64
      ;;
  esac
done

REPORT="${REPORT_REQUEST:-${TMPDIR:-/tmp}/onnx-splitpoint-v27915-small-acceptance-$$.json}"
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

PYCACHE_ROOT="$(/usr/bin/mktemp -d /tmp/onnx-v27915-acceptance-pycache.XXXXXXXXXX)"
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
    "schema": "onnx-splitpoint/v27915-small-acceptance/v1",
    "schema_version": 1,
    "version": "2.79.15",
    "build_id": "v2.79.15-fs-energy-and-deepx-part1-preprocessing",
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

finish() {
  local rc=$?
  local report_rc=0
  local cleanup_rc=0
  trap - EXIT
  set +e
  write_report "$rc"
  report_rc=$?
  case "$PYCACHE_ROOT" in
    /tmp/onnx-v27915-acceptance-pycache.??????????)
      if [[ -d "$PYCACHE_ROOT" && ! -L "$PYCACHE_ROOT" ]]; then
        /bin/rm -r -- "$PYCACHE_ROOT" || cleanup_rc=$?
      else
        cleanup_rc=70
      fi
      ;;
    *) cleanup_rc=70 ;;
  esac
  printf 'V27915_ACCEPTANCE_REPORT=%s\n' "$REPORT"
  if [[ "$rc" -eq 0 && "$report_rc" -ne 0 ]]; then rc="$report_rc"; fi
  if [[ "$rc" -eq 0 && "$cleanup_rc" -ne 0 ]]; then rc="$cleanup_rc"; fi
  exit "$rc"
}
trap finish EXIT

SMOKE=FAIL
"$PYTHON" -B -m onnx_splitpoint_tool.v27915_smoke
SMOKE=PASS

PYTEST=FAIL
"$PYTHON" -B -m pytest -q \
  tests/test_v27914_simple_m2_idle_calibration.py \
  tests/test_v27914_simple_idle_calibration_comparison.py \
  tests/test_v27915_fs_and_deepx_part1.py \
  tests/test_deepx_full_preprocessing_ab_contract.py \
  tests/test_v2757_cache_verify_only.py::test_manual_deepx_part1_cache_miss_blocks_without_probe_or_compile \
  tests/test_v27915_release_provenance.py \
  tests/test_v27910_registry_atomic_calibration.py \
  tests/test_v27912_registry_m2_migration.py \
  tests/test_v27912_registry_stale_writer_cas.py \
  tests/test_v27912_platform_power_udp_preflight.py \
  tests/test_v2799_platform_power_gui.py \
  tests/test_v2799_accelerator_idle_energy_semantics.py \
  tests/test_energy_final_contract_regressions.py \
  tests/test_energy_command_window_binding_v2.py::test_fake_collector_and_postprocessor_complete_v2_contract
PYTEST=PASS

MANIFEST=FAIL
"$PYTHON" -B scripts/build_source_manifest.py \
  --root "$SOURCE_ROOT" --verify --scope installed
MANIFEST=PASS

COMPILE=FAIL
"$PYTHON" -B -m compileall -q onnx_splitpoint_tool scripts
COMPILE=PASS

echo 'PASS v2.79.15 small acceptance'
