#!/usr/bin/env bash
# Hardware-independent v2.79.6 remaining-changes acceptance gate.

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  printf '%s\n' \
    'SKIP: run_v2796_small_acceptance.sh was sourced; run it with bash.' >&2
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

REPORT_REQUEST="${V2796_ACCEPTANCE_REPORT:-}"
while (( $# > 0 )); do
  case "$1" in
    --report)
      (( $# >= 2 )) || {
        printf '%s\n' 'ERROR: --report requires a path.' >&2
        exit 64
      }
      REPORT_REQUEST="$2"
      shift 2
      ;;
    --help|-h)
      printf '%s\n' \
        'Usage: bash scripts/run_v2796_small_acceptance.sh [--report PATH]' \
        '' \
        'Runs the hardware-independent R1-R7, artifact, and dual-gate tests.' \
        'The JSON report defaults to a unique file below /tmp.'
      exit 0
      ;;
    *)
      printf 'ERROR: unsupported option: %s\n' "$1" >&2
      exit 64
      ;;
  esac
done

if [[ -n "$REPORT_REQUEST" ]]; then
  REPORT="$REPORT_REQUEST"
else
  REPORT="${TMPDIR:-/tmp}/onnx-splitpoint-v2796-small-acceptance-$$.json"
fi

REPORT="$($PYTHON -B - "$REPORT" <<'PY'
from pathlib import Path
import sys

print(Path(sys.argv[1]).expanduser().resolve(strict=False))
PY
)"
case "$REPORT/" in
  "$SOURCE_ROOT/"*)
    printf '%s\n' \
      'ERROR: acceptance report must be outside the source tree.' >&2
    exit 64
    ;;
esac

cd -- "$SOURCE_ROOT"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"

PHASE_ORDER=(
  release_smoke
  R1
  R2
  R3
  R4
  R5
  product_integration
  R6
  R7
  artifact
  gate
  source_integrity
  syntax
)
declare -A PHASE_LABEL PHASE_SELECTION PHASE_STATUS PHASE_RC

PHASE_LABEL[release_smoke]='release identity and contract smoke'
PHASE_SELECTION[release_smoke]='python -m onnx_splitpoint_tool.v2796_smoke'
PHASE_LABEL[R1]='R1 release, aliases, documentation, and historical identity'
PHASE_SELECTION[R1]='test_v2796_release_provenance; test_v2796_documentation_identity; test_v279_release_provenance; historical v2795 identity/documentation'
PHASE_LABEL[R2]='R2 immutable Three-Stage claim invocation'
PHASE_SELECTION[R2]='Three-Stage closure/product adapter; YOLOv7 claim_gate_32 verifier including negatives'
PHASE_LABEL[R3]='R3 logical-primary matrix completeness'
PHASE_SELECTION[R3]='logical measurement identity, primary matrix, mirror, and companion tests'
PHASE_LABEL[R4]='R4 canonical DeepX runtime precision identity'
PHASE_SELECTION[R4]='DeepX precision alias and artifact-proof tests'
PHASE_LABEL[R5]='R5 physical required-scope endpoint binding'
PHASE_SELECTION[R5]='test_v2792_required_scope'
PHASE_LABEL[product_integration]='R3/R4/R5 product-path integration'
PHASE_SELECTION[product_integration]='test_v2796_r3_r4_r5_product_path'
PHASE_LABEL[R6]='R6 exact v2.78.4 legacy reconciliation'
PHASE_SELECTION[R6]='reconciler, evidence-state, and logical-join regressions'
PHASE_LABEL[R7]='R7 unlimited long-run Hailo cold-build admission'
PHASE_SELECTION[R7]='test_v2796_yolo11_r7_r8b_gate'
PHASE_LABEL[artifact]='final artifact-index reseal and verification'
PHASE_SELECTION[artifact]='test_v2796_artifact_index_closure'
PHASE_LABEL[gate]='YOLO11 Full terminal-admission gate'
PHASE_SELECTION[gate]='complete YOLO11 R8B verifier regression file and dual-gate launcher contract'
PHASE_LABEL[source_integrity]='installed source-manifest integrity'
PHASE_SELECTION[source_integrity]='installed-manifest scope regressions; build_source_manifest --verify --scope installed'
PHASE_LABEL[syntax]='release shell and Python syntax'
PHASE_SELECTION[syntax]='bash -n; compileall'

for phase_id in "${PHASE_ORDER[@]}"; do
  PHASE_STATUS[$phase_id]=NOT_RUN
  PHASE_RC[$phase_id]=''
done

CURRENT_PHASE=''
FIRST_FAILED_PHASE=''
ACCEPTANCE_RC=0

write_report() {
  local overall_rc="$1"
  local phase_id
  local -a report_args=(
    "$REPORT" "$SOURCE_ROOT" "$PYTHON" "$overall_rc"
  )
  for phase_id in "${PHASE_ORDER[@]}"; do
    report_args+=(
      "$phase_id"
      "${PHASE_LABEL[$phase_id]}"
      "${PHASE_SELECTION[$phase_id]}"
      "${PHASE_STATUS[$phase_id]}"
      "${PHASE_RC[$phase_id]}"
    )
  done

  "$PYTHON" -B - "${report_args[@]}" <<'PY'
from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
import sys

report = Path(sys.argv[1])
source_root = sys.argv[2]
python = sys.argv[3]
overall_rc = int(sys.argv[4])
fields = sys.argv[5:]
if len(fields) % 5:
    raise SystemExit("invalid acceptance phase payload")

phases = []
for offset in range(0, len(fields), 5):
    phase_id, label, selection, status, rc_text = fields[offset:offset + 5]
    phases.append(
        {
            "id": phase_id,
            "label": label,
            "selection": selection,
            "status": status,
            "return_code": int(rc_text) if rc_text else None,
        }
    )

status_by_phase = {row["id"]: row["status"] for row in phases}


def contract_status(*phase_ids: str) -> str:
    statuses = [status_by_phase[phase_id] for phase_id in phase_ids]
    if any(status == "FAIL" for status in statuses):
        return "fail"
    if all(status == "PASS" for status in statuses):
        return "pass"
    return "not_run"


contracts = {
    "release_identity": contract_status("release_smoke", "R1"),
    "three_stage_product_adapter": contract_status("R2"),
    "logical_mirror_completeness": contract_status("R3"),
    "deepx_precision_identity": contract_status("R4"),
    "physical_required_run_scope": contract_status("R5"),
    "legacy_reconciler_fixture": contract_status("R6"),
    "new_quality_applicability_fixture": contract_status("R6"),
    "source_manifest": contract_status("source_integrity"),
}

passed = overall_rc == 0 and all(row["status"] == "PASS" for row in phases)
payload = {
    "schema": "onnx-splitpoint/v2796-small-acceptance/v1",
    "schema_version": 1,
    "version": "2.79.6",
    "build_id": "v2.79.6-remaining-changes-yolo11-admission-closure",
    "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "source_root": source_root,
    "python": python,
    "status": "PASS" if passed else "FAIL",
    "return_code": overall_rc,
    **contracts,
    "contracts": contracts,
    "hardware_execution": "not_run_in_offline_gate",
    "phases": phases,
}

report.parent.mkdir(parents=True, exist_ok=True)
temporary = report.with_name(f".{report.name}.tmp.{os.getpid()}")
temporary.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
os.replace(temporary, report)
PY
}

finalize() {
  local rc="${1:-99}"
  local report_rc=0
  local phase_id
  trap - EXIT
  set +e

  if [[ -n "$CURRENT_PHASE" && \
        "${PHASE_STATUS[$CURRENT_PHASE]}" == RUNNING ]]; then
    PHASE_STATUS[$CURRENT_PHASE]=FAIL
    PHASE_RC[$CURRENT_PHASE]="$rc"
  fi

  if [[ "$rc" -eq 0 ]]; then
    for phase_id in "${PHASE_ORDER[@]}"; do
      if [[ "${PHASE_STATUS[$phase_id]}" != PASS ]]; then
        rc=1
        break
      fi
    done
  fi

  write_report "$rc"
  report_rc=$?
  if [[ "$report_rc" -ne 0 ]]; then
    printf 'ERROR: v2.79.6 acceptance report write failed (rc=%s).\n' \
      "$report_rc" >&2
    if [[ "$rc" -eq 0 ]]; then
      rc="$report_rc"
    fi
  else
    printf 'V2796_ACCEPTANCE_REPORT=%s\n' "$REPORT"
  fi

  if [[ "$rc" -eq 0 ]]; then
    printf '%s\n' \
      'PASS v2.79.6 small acceptance' \
      'SKIP accelerator compiler and runtime hardware in this local gate'
  else
    printf 'FAIL v2.79.6 small acceptance (rc=%s, phase=%s)\n' \
      "$rc" "${FIRST_FAILED_PHASE:-${CURRENT_PHASE:-before_phases}}" >&2
  fi
  exit "$rc"
}
trap 'finalize $?' EXIT

run_phase() {
  local phase_id="$1"
  shift
  local rc=0

  CURRENT_PHASE="$phase_id"
  PHASE_STATUS[$phase_id]=RUNNING
  printf '\n[%s] %s\n' "$phase_id" "${PHASE_LABEL[$phase_id]}"
  set +e
  "$@"
  rc=$?
  set -e
  PHASE_RC[$phase_id]="$rc"
  if [[ "$rc" -eq 0 ]]; then
    PHASE_STATUS[$phase_id]=PASS
    printf 'PASS_PHASE=%s\n' "$phase_id"
    CURRENT_PHASE=''
    return 0
  fi
  PHASE_STATUS[$phase_id]=FAIL
  printf 'FAIL_PHASE=%s RC=%s\n' "$phase_id" "$rc" >&2
  return "$rc"
}

record_phase() {
  local phase_rc=0
  if run_phase "$@"; then
    return 0
  else
    phase_rc=$?
  fi
  if [[ "$ACCEPTANCE_RC" -eq 0 ]]; then
    ACCEPTANCE_RC="$phase_rc"
    FIRST_FAILED_PHASE="$1"
  fi
  # Keep collecting independent phase results.  The final EXIT trap emits the
  # aggregate nonzero code and a report that distinguishes FAIL from NOT_RUN.
  return 0
}

pytest_phase() {
  "$PYTHON" -B -m pytest -q -p no:cacheprovider \
    --import-mode=importlib --tb=short "$@"
}

source_integrity_phase() {
  pytest_phase tests/test_v27542_installed_source_manifest_scope.py \
  && "$PYTHON" -B scripts/build_source_manifest.py \
    --root "$SOURCE_ROOT" --verify --scope installed
}

syntax_phase() {
  bash -n \
    scripts/run_v2796_small_acceptance.sh \
    scripts/run_v2795_small_acceptance.sh \
    scripts/run_v279_small_acceptance.sh \
    scripts/run_local_acceptance.sh \
    scripts/run_v2796_seven_model_long_overnight.sh \
    scripts/run_v279_seven_model_long_overnight.sh \
    scripts/run_v2796_yolo11_r8b_gate.sh \
    scripts/update_source_release.sh \
  && "$PYTHON" -B -m compileall -q onnx_splitpoint_tool scripts tests
}

record_phase release_smoke \
  "$PYTHON" -B -m onnx_splitpoint_tool.v2796_smoke

record_phase R1 pytest_phase \
  tests/test_v2796_release_provenance.py \
  tests/test_v2796_documentation_identity.py \
  tests/test_v279_release_provenance.py \
  tests/test_v2795_release_provenance.py \
  tests/test_v2795_documentation_identity.py

record_phase R2 pytest_phase \
  tests/test_v2796_three_stage_closure.py \
  tests/test_v2793_productized_three_stage.py \
  tests/test_v2791_concurrent_three_stage_normal_runner.py \
  tests/test_v2796_yolov7_claim_gate_32.py

record_phase R3 pytest_phase \
  tests/test_v2792_logical_join.py::test_direct_setup_and_setup_less_mirror_form_one_logical_measurement \
  tests/test_v2792_logical_join.py::test_distinct_real_setups_are_never_deduplicated \
  tests/test_v2792_logical_join.py::test_exact_request_sha_join_prefers_direct_setup \
  tests/test_v2792_logical_join.py::test_request_sha_join_without_result_setup_still_excludes_mirror_metadata \
  tests/test_v2792_logical_join.py::test_duplicate_looking_rows_without_exact_proof_remain_separate \
  tests/test_v2792_logical_join.py::test_exact_request_sha_join_rejects_conflicting_backend \
  tests/test_v2792_logical_join.py::test_companion_results_never_fill_primary_matrix

record_phase R4 pytest_phase \
  tests/test_v2792_logical_join.py::test_deepx_precision_alias_only_canonicalizes_for_same_artifact \
  tests/test_v2792_logical_join.py::test_deepx_precision_never_uses_shared_trt_engine_as_dxnn_proof

record_phase R5 pytest_phase \
  tests/test_v2792_required_scope.py

record_phase product_integration pytest_phase \
  tests/test_v2796_r3_r4_r5_product_path.py

record_phase R6 pytest_phase \
  tests/test_v2792_reconciler.py \
  tests/test_v2792_evidence_state.py \
  tests/test_v2792_logical_join.py

record_phase R7 pytest_phase \
  tests/test_v2796_yolo11_r7_r8b_gate.py::test_every_documented_hailo_unlimited_token_is_canonical \
  tests/test_v2796_yolo11_r7_r8b_gate.py::test_zero_timeout_means_unlimited \
  tests/test_v2796_yolo11_r7_r8b_gate.py::test_invalid_hailo_timeout_tokens_fail_closed \
  tests/test_v2796_yolo11_r7_r8b_gate.py::test_retry_preserves_previous_timeout_receipt \
  tests/test_v2796_yolo11_r7_r8b_gate.py::test_long_profile_does_not_use_legacy_9000_second_cold_build_cap

record_phase artifact pytest_phase \
  tests/test_v2796_artifact_index_closure.py

record_phase gate pytest_phase \
  tests/test_v2796_yolo11_r7_r8b_gate.py \
  tests/test_v2796_seven_model_overnight_launcher.py

record_phase source_integrity source_integrity_phase

record_phase syntax syntax_phase

exit "$ACCEPTANCE_RC"
