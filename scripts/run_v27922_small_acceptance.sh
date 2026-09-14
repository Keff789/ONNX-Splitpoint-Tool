#!/usr/bin/env bash
# Hardware-independent acceptance for the v2.79.22 persistent negative-build-evidence integration.

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  printf '%s\n' \
    'SKIP: run_v27922_small_acceptance.sh was sourced; run it with bash.' >&2
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

REPORT_REQUEST="${V27922_ACCEPTANCE_REPORT:-}"
while (( $# > 0 )); do
  case "$1" in
    --report)
      (( $# >= 2 )) || { echo 'ERROR: --report requires a path.' >&2; exit 64; }
      REPORT_REQUEST="$2"
      shift 2
      ;;
    --help|-h)
      printf '%s\n' \
        'Usage: bash scripts/run_v27922_small_acceptance.sh [--report PATH]' \
        '' \
        'Checks persistent build evidence, recovery, classification and cache reuse without hardware.'
      exit 0
      ;;
    *)
      printf 'ERROR: unsupported option: %s\n' "$1" >&2
      exit 64
      ;;
  esac
done

REPORT="${REPORT_REQUEST:-${TMPDIR:-/tmp}/onnx-splitpoint-v27922-small-acceptance-$$.json}"
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

PYCACHE_ROOT="$(/usr/bin/mktemp -d /tmp/onnx-v27922-acceptance-pycache.XXXXXXXXXX)"
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
    "schema": "onnx-splitpoint/v27922-small-acceptance/v1",
    "schema_version": 1,
    "version": "2.79.22",
    "build_id": "v2.79.22-global-negative-build-evidence",
    "release_scope": "global_negative_build_evidence",
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
    /tmp/onnx-v27922-acceptance-pycache.??????????)
      if [[ -d "$PYCACHE_ROOT" && ! -L "$PYCACHE_ROOT" ]]; then
        /bin/rm -r -- "$PYCACHE_ROOT" || cleanup_rc=$?
      else
        cleanup_rc=70
      fi
      ;;
    *) cleanup_rc=70 ;;
  esac
  printf 'V27922_ACCEPTANCE_REPORT=%s\n' "$REPORT"
  if [[ "$rc" -eq 0 && "$report_rc" -ne 0 ]]; then rc="$report_rc"; fi
  if [[ "$rc" -eq 0 && "$cleanup_rc" -ne 0 ]]; then rc="$cleanup_rc"; fi
  exit "$rc"
}
trap finish EXIT

SMOKE=FAIL
"$PYTHON" -B -m onnx_splitpoint_tool.v27922_smoke
SMOKE=PASS

PYTEST=FAIL
"$PYTHON" -B -m pytest -q -ra -p no:cacheprovider \
  tests/test_v27922_release_closure.py \
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
bash -n scripts/run_v27922_small_acceptance.sh \
  scripts/run_v27921_small_acceptance.sh \
  scripts/run_v27920_small_acceptance.sh \
  scripts/run_v27919_small_acceptance.sh \
  scripts/run_v279_small_acceptance.sh \
  scripts/run_local_acceptance.sh \
  scripts/update_source_release.sh
SHELL=PASS

echo 'PASS v2.79.22 small acceptance (real cache preflight and hardware EvalRun: NOT_RUN)'
