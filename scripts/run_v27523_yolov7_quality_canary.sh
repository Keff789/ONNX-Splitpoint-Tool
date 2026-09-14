#!/usr/bin/env bash
# One-command v2.75.23 YOLOv7 Full-only Quality acceptance and pack verification.

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  printf '%s\n' \
    'SKIP: run_v27523_yolov7_quality_canary.sh was sourced.' \
    'Run it in its own Bash process.' >&2
  return 0
fi

set -Eeuo pipefail

TOOL="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
PY="$TOOL/.venv/bin/python"

if (( $# > 3 )); then
  printf '%s\n' \
    'Usage: bash scripts/run_v27523_yolov7_quality_canary.sh' \
    '       [MODELS_ROOT [RUNS_ROOT [PACK_ROOT]]]' >&2
  exit 64
fi

MODELS_ROOT="${1:-$HOME/Models}"
RUNS_ROOT="${2:-$MODELS_ROOT/EvaluationRuns}"
PACK_ROOT="${3:-$HOME/Downloads}"
PROFILE_ROOT="${ONNX_SPLITPOINT_PROFILE_DIR:-$HOME/.onnx_splitpoint_tool/evaluation_profiles}"
MIN_FREE_GIB="${ONNX_SPLITPOINT_CANARY_MIN_FREE_GIB:-15}"
MIN_FREE_INODES="${ONNX_SPLITPOINT_CANARY_MIN_FREE_INODES:-50000}"
HARD_MIN_FREE_GIB=10
HARD_MIN_FREE_INODES=50000
PACK_MIN_FREE_BYTES=$((2 * 1024 * 1024 * 1024))
PACK_MIN_FREE_INODES=1024

if [[ ! -x "$PY" ]]; then
  printf 'FEHLER: Tool-Venv fehlt: %s\n' "$PY" >&2
  exit 69
fi
MODELS_ROOT="$(realpath -e -- "$MODELS_ROOT")"
mkdir -p -- "$RUNS_ROOT" "$PACK_ROOT" "$PROFILE_ROOT"
RUNS_ROOT="$(realpath -e -- "$RUNS_ROOT")"
PACK_ROOT="$(realpath -e -- "$PACK_ROOT")"
PROFILE_ROOT="$(realpath -e -- "$PROFILE_ROOT")"

case "$MIN_FREE_GIB" in
  ''|*[!0-9]*)
    printf 'FEHLER: ONNX_SPLITPOINT_CANARY_MIN_FREE_GIB muss ganzzahlig sein.\n' >&2
    exit 64
    ;;
esac
case "$MIN_FREE_INODES" in
  ''|*[!0-9]*)
    printf 'FEHLER: ONNX_SPLITPOINT_CANARY_MIN_FREE_INODES muss ganzzahlig sein.\n' >&2
    exit 64
    ;;
esac

MIN_FREE_GIB=$((10#$MIN_FREE_GIB))
MIN_FREE_INODES=$((10#$MIN_FREE_INODES))
if (( MIN_FREE_GIB < HARD_MIN_FREE_GIB )); then
  printf 'FEHLER: Der lokale Canary-Grenzwert darf nicht unter %s GiB liegen.\n' \
    "$HARD_MIN_FREE_GIB" >&2
  exit 64
fi
if (( MIN_FREE_INODES < HARD_MIN_FREE_INODES )); then
  printf 'FEHLER: Der lokale Canary-Grenzwert darf nicht unter %s Inodes liegen.\n' \
    "$HARD_MIN_FREE_INODES" >&2
  exit 64
fi

FREE_BYTES="$(df -B1 --output=avail "$RUNS_ROOT" | awk 'NR==2 {print $1}')"
FREE_INODES="$(df --output=iavail "$RUNS_ROOT" | awk 'NR==2 {print $1}')"
MIN_FREE_BYTES=$((MIN_FREE_GIB * 1024 * 1024 * 1024))
printf 'Lokaler Speicher: %.2f GiB frei, %s freie Inodes\n' \
  "$(awk -v bytes="$FREE_BYTES" 'BEGIN {print bytes / 1073741824}')" \
  "$FREE_INODES"
if (( FREE_BYTES < MIN_FREE_BYTES )); then
  printf 'STOP: Für den 5000-Bilder-Canary werden mindestens %s GiB frei benötigt: %s\n' \
    "$MIN_FREE_GIB" "$RUNS_ROOT" >&2
  exit 70
fi
if (( FREE_INODES < MIN_FREE_INODES )); then
  printf 'STOP: Mindestens %s freie Inodes werden benötigt: %s\n' \
    "$MIN_FREE_INODES" "$RUNS_ROOT" >&2
  exit 71
fi

PACK_FREE_BYTES="$(df -B1 --output=avail "$PACK_ROOT" | awk 'NR==2 {print $1}')"
PACK_FREE_INODES="$(df --output=iavail "$PACK_ROOT" | awk 'NR==2 {print $1}')"
if (( PACK_FREE_BYTES < PACK_MIN_FREE_BYTES )); then
  printf 'STOP: Für den verifizierten Debug-Pack werden mindestens 2 GiB frei benötigt: %s\n' \
    "$PACK_ROOT" >&2
  exit 72
fi
if (( PACK_FREE_INODES < PACK_MIN_FREE_INODES )); then
  printf 'STOP: Für den Debug-Pack werden mindestens %s freie Inodes benötigt: %s\n' \
    "$PACK_MIN_FREE_INODES" "$PACK_ROOT" >&2
  exit 72
fi

cd -- "$TOOL"
unset PYTHONOPTIMIZE
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$TOOL${PYTHONPATH:+:$PYTHONPATH}"

"$PY" -B - <<'PY'
import onnx_splitpoint_tool as tool
from onnx_splitpoint_tool.workflow.runner import WORKFLOW_VERSION

expected_version = "2.75.23"
expected_build = "v2.75.23-full-only-canary-launch-repair"
assert tool.__version__ == expected_version, tool.__version__
assert tool.__release__ == expected_version, tool.__release__
assert tool.__development_lineage__ == "v" + expected_version
assert tool.__build_id__ == expected_build, tool.__build_id__
assert WORKFLOW_VERSION == expected_build, WORKFLOW_VERSION
for feature in (
    "run_mode_full_only_canary_preservation",
    "quality_canary_variant_recipe_validation",
    "yolov7_full_only_canary_launcher",
):
    assert feature in tool.__build_features__, feature
print("PASS v2.75.23 identity")
PY

PY="$PY" bash scripts/run_v27523_small_acceptance.sh

STAMP="$(date +%Y%m%d_%H%M%S)"
PROFILE_ID="yolov7_full_only_quality_canary_v27523_${STAMP}"
PROFILE="$PROFILE_ROOT/${PROFILE_ID}.yaml"
CANARY_LOG="$RUNS_ROOT/${PROFILE_ID}.log"

"$PY" -B scripts/create_v27523_yolov7_quality_canary_profile.py \
  --out "$PROFILE" \
  --models-root "$MODELS_ROOT" \
  --profile-id "$PROFILE_ID"

workflow_rc=0
log_rc=0
set +e
"$PY" -B scripts/run_evaluation_workflow.py \
  --profile "$PROFILE" \
  --models-root "$MODELS_ROOT" \
  --out "$RUNS_ROOT" \
  --profile-driven \
  --require-run-mode standard \
  --require-fresh-run \
  --execution-mode generate_and_run 2>&1 | tee "$CANARY_LOG"
pipeline_status=("${PIPESTATUS[@]}")
workflow_rc=${pipeline_status[0]}
log_rc=${pipeline_status[1]}
set -e
if (( log_rc != 0 )); then
  printf 'FEHLER: Canary-Log konnte nicht vollständig geschrieben werden (tee rc=%s): %s\n' \
    "$log_rc" "$CANARY_LOG" >&2
fi

RUN_DIR="$("$PY" -B - "$RUNS_ROOT" "$PROFILE_ID" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve(strict=True)
prefix = sys.argv[2] + "_"
candidates = []
for path in root.iterdir():
    if (
        path.name.startswith(prefix)
        and path.is_dir()
        and not path.is_symlink()
        and path.resolve(strict=True).parent == root
    ):
        candidates.append(path.resolve(strict=True))
if len(candidates) != 1:
    raise SystemExit(
        f"expected exactly one new canary directory, found {len(candidates)}: "
        + repr([str(path) for path in candidates])
    )
print(candidates[0])
PY
)" || true

pack_rc="not-run"
verify_rc="not-run"
DEBUG_PACK=""
QUALITY_DECISION="not-evaluated"

if [[ -n "$RUN_DIR" && -d "$RUN_DIR" ]]; then
  DEBUG_PACK="$PACK_ROOT/$(basename -- "$RUN_DIR")_debug_pack.zip"
  PACK_FREE_BYTES="$(df -B1 --output=avail "$PACK_ROOT" | awk 'NR==2 {print $1}')"
  PACK_FREE_INODES="$(df --output=iavail "$PACK_ROOT" | awk 'NR==2 {print $1}')"
  if (( PACK_FREE_BYTES < PACK_MIN_FREE_BYTES )); then
    printf 'STOP: Nach dem Canary sind weniger als 2 GiB für den Debug-Pack frei: %s\n' \
      "$PACK_ROOT" >&2
    pack_rc=72
  elif (( PACK_FREE_INODES < PACK_MIN_FREE_INODES )); then
    printf 'STOP: Nach dem Canary sind weniger als %s Inodes für den Debug-Pack frei: %s\n' \
      "$PACK_MIN_FREE_INODES" "$PACK_ROOT" >&2
    pack_rc=72
  else
    set +e
    "$PY" -B scripts/create_evaluation_debug_pack.py \
      --eval-run-dir "$RUN_DIR" \
      --out "$DEBUG_PACK"
    pack_rc=$?
    set -e
  fi
fi

if [[ "$pack_rc" == "0" ]]; then
  set +e
  QUALITY_DECISION="$("$PY" -B - "$DEBUG_PACK" "$DEBUG_PACK.manifest.json" <<'PY'
import hashlib
import json
import sys
import zipfile
from pathlib import Path

pack = Path(sys.argv[1]).resolve(strict=True)
sidecar = Path(sys.argv[2]).resolve(strict=True)
side = json.loads(sidecar.read_text(encoding="utf-8"))
digest = hashlib.sha256()
with pack.open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
archive_sha = digest.hexdigest()
assert side.get("status") == "verified", side.get("status")
assert side.get("archive_sha256") == "sha256:" + archive_sha
assert side.get("archive_size_bytes") == pack.stat().st_size
assert side.get("archive_name") == pack.name

with zipfile.ZipFile(pack) as archive:
    assert archive.testzip() is None
    names = archive.namelist()
    assert len(names) == len(set(names)), "duplicate ZIP member names"
    manifest = json.loads(archive.read("debug_pack_manifest.json"))
    inputs = manifest.get("central_quality_replay_inputs") or {}
    assert inputs.get("hash_valid") is True, inputs
    assert not inputs.get("missing_members"), inputs
    assert not inputs.get("failures"), inputs
    expected_members = set(inputs.get("expected_members") or [])
    present_members = set(inputs.get("present_members") or [])
    assert expected_members and present_members == expected_members, (
        present_members, expected_members
    )
    assert expected_members <= set(names), sorted(expected_members - set(names))
    required_members = set(side.get("required_members") or [])
    assert {"debug_pack_manifest.json", *expected_members} <= required_members
    summary = json.loads(
        archive.read("quality_management/central_quality_summary.json")
    )

rows = summary.get("results") or []
assert summary.get("schema") == "onnx-splitpoint/central-quality-summary"
assert summary.get("schema_version") == 1
assert summary.get("request_count") == 4
assert summary.get("technical_status") == "ok", summary.get("technical_status")
assert summary.get("completed_count") == 4, summary.get("completed_count")
identity_contract = summary.get("quality_acceptance_identity_contract") or {}
postcondition = identity_contract.get("postcondition") or {}
assert identity_contract.get("execution_scope") == "full_only", identity_contract
assert postcondition.get("status") == "verified_exact", postcondition
assert identity_contract.get("identity_key_fields") == [
    "model_id", "source_run_id", "setup_id", "backend", "variant",
    "execution_role", "performance_claims_emitted",
], identity_contract.get("identity_key_fields")
expected_contract = {
    ("yolov7_paper", "hailo8", "orin_nx_hailo8_01", "hailo8",
     "full", "full_quality_only", False),
    ("yolov7_paper", "native_full_tensorrt", "orin_nx_hailo8_01",
     "tensorrt", "full", "full_quality_only", False),
    ("yolov7_paper", "hailo10", "orin_nx_hailo10_01", "hailo10h",
     "full", "full_quality_only", False),
    ("yolov7_paper", "native_full_tensorrt", "orin_nx_hailo10_01",
     "tensorrt", "full", "full_quality_only", False),
}
observed_contract = {
    (
        str(row.get("model_id") or ""),
        str(row.get("source_run_id") or ""),
        str(row.get("setup_id") or ""),
        str(row.get("backend") or ""),
        str(row.get("variant") or ""),
        str(row.get("execution_role") or ""),
        row.get("performance_claims_emitted"),
    )
    for row in identity_contract.get("expected_identities") or []
}
assert observed_contract == expected_contract, observed_contract
expected = {
    ("hailo8", "orin_nx_hailo8_01", "full"),
    ("native_full_tensorrt", "orin_nx_hailo8_01", "full"),
    ("hailo10", "orin_nx_hailo10_01", "full"),
    ("native_full_tensorrt", "orin_nx_hailo10_01", "full"),
}
observed = {
    (
        str(row.get("source_run_id") or ""),
        str(row.get("source_setup_id") or row.get("setup_id") or ""),
        str(row.get("variant") or ""),
    )
    for row in rows
}
assert len(rows) == 4 and observed == expected, (len(rows), observed)
decisions = []
for row in rows:
    assert int(row.get("n") or 0) == 5000, row.get("n")
    assert row.get("configured_guardrails") == ["ap50", "ap75"]
    assert row.get("guardrail_contract_complete") is True
    seed_schema = row.get("seed_schema") or {}
    assert int(seed_schema.get("seed") or 0) == 20260710, seed_schema
    assert int(seed_schema.get("repetitions") or 0) == 5000, seed_schema
    assert int(seed_schema.get("image_count") or 0) == 5000, seed_schema
    assert int(row.get("bootstrap_workers_requested") or 0) == 4, row
    guardrails = row.get("guardrails") or {}
    assert "ap50" in guardrails and "ap75" in guardrails, guardrails
    assert (row.get("primary") or {}).get("metric") == "coco_ap_50_95"
    for component in (row.get("primary"), guardrails["ap50"], guardrails["ap75"]):
        assert component.get("decision") in {"pass", "fail", "inconclusive"}
        assert int(component.get("n") or 0) == 5000, component
        assert float(component.get("margin")) == 0.01, component
        assert int(component.get("bootstrap_repetitions_requested") or 0) == 5000
        effective = int(component.get("bootstrap_repetitions") or 0)
        skipped = str(component.get("bootstrap_skipped_reason") or "")
        assert (effective, skipped) in {
            (5000, ""),
            (0, "candidate_reference_identical"),
            (0, "point_estimate_below_non_inferiority_margin"),
        }, (effective, skipped)
    decision = str(row.get("decision") or "")
    assert decision in {"pass", "fail", "inconclusive"}, decision
    decisions.append(decision)
if "fail" in decisions:
    aggregate = "fail"
elif "inconclusive" in decisions:
    aggregate = "inconclusive"
else:
    aggregate = "pass"
assert summary.get("quality_decision") == aggregate, (
    summary.get("quality_decision"), aggregate
)
print(aggregate)
PY
)"
  verify_rc=$?
  set -e
fi

printf '\nCANARY_RC=%s\n' "$workflow_rc"
printf 'CANARY_LOG_RC=%s\n' "$log_rc"
printf 'PROFILE=%s\n' "$PROFILE"
printf 'CANARY_LOG=%s\n' "$CANARY_LOG"
printf 'RUN_DIR=%s\n' "${RUN_DIR:-NOT_DISCOVERED}"
printf 'PACK_RC=%s\n' "$pack_rc"
printf 'DEBUG_PACK=%s\n' "${DEBUG_PACK:-NOT_CREATED}"
if [[ -n "$DEBUG_PACK" ]]; then
  printf 'DEBUG_PACK_MANIFEST=%s\n' "$DEBUG_PACK.manifest.json"
else
  printf 'DEBUG_PACK_MANIFEST=NOT_CREATED\n'
fi
printf 'PACK_REPLAY_INPUTS_RC=%s\n' "$verify_rc"
if [[ "$verify_rc" == "0" ]]; then
  printf 'OFFLINE_REPLAY=READY_FROM_DEBUG_PACK\n'
else
  printf 'OFFLINE_REPLAY=NOT_READY\n'
fi
printf 'QUALITY_DECISION=%s\n' "$QUALITY_DECISION"

if (( workflow_rc != 0 )); then
  exit "$workflow_rc"
fi
if (( log_rc != 0 )); then
  exit "$log_rc"
fi
if [[ "$pack_rc" != "0" || "$verify_rc" != "0" ]]; then
  exit 1
fi
printf 'PASS v2.75.23 YOLOv7 Full-only Quality canary and replay-ready pack\n'
