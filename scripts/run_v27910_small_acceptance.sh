#!/usr/bin/env bash
# Hardware-independent v2.79.10 release-closure acceptance gate.

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  printf '%s\n' 'SKIP: run_v27910_small_acceptance.sh was sourced; run it with bash.' >&2
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

REPORT_REQUEST="${V27910_ACCEPTANCE_REPORT:-}"
while (( $# > 0 )); do
  case "$1" in
    --report)
      (( $# >= 2 )) || { echo 'ERROR: --report requires a path.' >&2; exit 64; }
      REPORT_REQUEST="$2"; shift 2 ;;
    --help|-h)
      printf '%s\n' \
        'Usage: bash scripts/run_v27910_small_acceptance.sh [--report PATH]' \
        '' \
        'Runs the hardware-independent v2.79.10 identity, historical provenance,' \
        'platform-power, current launcher, updater-lock, manifest and syntax gates.'
      exit 0 ;;
    *) printf 'ERROR: unsupported option: %s\n' "$1" >&2; exit 64 ;;
  esac
done

REPORT="${REPORT_REQUEST:-${TMPDIR:-/tmp}/onnx-splitpoint-v27910-small-acceptance-$$.json}"
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
export PYTHONDONTWRITEBYTECODE=1
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
    "schema": "onnx-splitpoint/v27910-small-acceptance/v1",
    "schema_version": 1,
    "version": "2.79.10",
    "build_id": "v2.79.10-urecs-platform-power-release-closure",
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
trap 'rc=$?; write_report "$rc"; printf "V27910_ACCEPTANCE_REPORT=%s\n" "$REPORT"; exit "$rc"' EXIT

"$PYTHON" -B -m onnx_splitpoint_tool.v27910_smoke
SMOKE=PASS

"$PYTHON" -B -m pytest -q \
  tests/test_v2799_platform_power.py \
  tests/test_v2799_platform_power_gui.py \
  tests/test_v2799_accelerator_idle_energy_semantics.py \
  tests/test_v27910_registry_atomic_calibration.py \
  tests/test_v2798_release_provenance.py \
  tests/test_v2799_release_provenance.py \
  tests/test_v279_release_provenance.py \
  tests/test_v27910_release_provenance.py \
  tests/test_v27910_seven_model_overnight_launcher.py \
  tests/test_v27910_update_active_workflow_guard.py \
  tests/test_v27910_yolo11_gate_profile_schema.py \
  tests/test_v27910_yolo11_backend_bound_gate.py \
  tests/test_v27910_yolo11_recovery_manifest.py \
  tests/test_v27910_yolo11_gate_launcher.py \
  tests/test_v27910_yolov7_claim_gate_32.py
PYTEST=PASS

"$PYTHON" -B scripts/build_source_manifest.py --root "$SOURCE_ROOT" --verify
MANIFEST=PASS

"$PYTHON" -B -m compileall -q onnx_splitpoint_tool scripts
COMPILE=PASS

echo 'PASS v2.79.10 small acceptance'
exit 0
