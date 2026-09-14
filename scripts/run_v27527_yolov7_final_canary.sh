#!/usr/bin/env bash
# One-model Final-contract canary. It never starts the three-model Final run.

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  printf '%s\n' 'SKIP: run this script in its own Bash process.' >&2
  return 0
fi

set -Eeuo pipefail

TOOL="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
PY="$TOOL/.venv/bin/python"

if (( $# < 1 || $# > 3 )); then
  printf '%s\n' \
    'Usage: bash scripts/run_v27527_yolov7_final_canary.sh SOURCE_PROFILE [RUNS_ROOT [PACK_ROOT]]' >&2
  exit 64
fi

SOURCE_PROFILE="$(realpath -e -- "$1")"
RUNS_ROOT="${2:-$(dirname -- "$SOURCE_PROFILE")/EvaluationRuns}"
PACK_ROOT="${3:-$(dirname -- "$SOURCE_PROFILE")/CanaryPacks}"
MIN_FREE_GIB="${ONNX_SPLITPOINT_CANARY_MIN_FREE_GIB:-10}"
MIN_FREE_INODES="${ONNX_SPLITPOINT_CANARY_MIN_FREE_INODES:-50000}"

if [[ ! -x "$PY" ]]; then
  printf 'FEHLER: Tool-Venv fehlt: %s\n' "$PY" >&2
  exit 69
fi
case "$MIN_FREE_GIB" in ''|*[!0-9]*) printf 'FEHLER: ungültige GiB-Grenze.\n' >&2; exit 64;; esac
case "$MIN_FREE_INODES" in ''|*[!0-9]*) printf 'FEHLER: ungültige Inode-Grenze.\n' >&2; exit 64;; esac
if (( 10#$MIN_FREE_GIB < 10 || 10#$MIN_FREE_INODES < 50000 )); then
  printf 'FEHLER: Canary-Grenzen dürfen 10 GiB / 50000 Inodes nicht unterschreiten.\n' >&2
  exit 64
fi

mkdir -p -- "$RUNS_ROOT" "$PACK_ROOT"
RUNS_ROOT="$(realpath -e -- "$RUNS_ROOT")"
PACK_ROOT="$(realpath -e -- "$PACK_ROOT")"
FREE_BYTES="$(df -B1 --output=avail "$RUNS_ROOT" | awk 'NR==2 {print $1}')"
FREE_INODES="$(df --output=iavail "$RUNS_ROOT" | awk 'NR==2 {print $1}')"
if (( FREE_BYTES < 10#$MIN_FREE_GIB * 1024 * 1024 * 1024 )); then
  printf 'STOP: weniger als %s GiB frei in %s\n' "$MIN_FREE_GIB" "$RUNS_ROOT" >&2
  exit 70
fi
if (( FREE_INODES < 10#$MIN_FREE_INODES )); then
  printf 'STOP: weniger als %s Inodes frei in %s\n' "$MIN_FREE_INODES" "$RUNS_ROOT" >&2
  exit 71
fi

cd -- "$TOOL"
unset PYTHONOPTIMIZE
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$TOOL${PYTHONPATH:+:$PYTHONPATH}"

STAMP="$(date +%Y%m%d_%H%M%S)"
# Stable scientific profile identity is required so a byte-identical second
# invocation can prove a genuine post-fix warm-cache hit.
PROFILE_ID="${ONNX_SPLITPOINT_CANARY_PROFILE_ID:-yolov7_final_contract_canary_v27527}"
# Keep the projection beside its source so relative sealed-manifest references
# retain exactly the same base directory.
PROFILE_DIR="$(dirname -- "$SOURCE_PROFILE")"
PROFILE="$PROFILE_DIR/${PROFILE_ID}.yaml"
PREFLIGHT="$PROFILE_DIR/${PROFILE_ID}_preflight"
LOG="$RUNS_ROOT/${PROFILE_ID}_${STAMP}.log"
BEFORE_RUNS="$(mktemp -p "${TMPDIR:-/tmp}" v27527-canary-before.XXXXXX)"
trap 'rm -f -- "$BEFORE_RUNS"' EXIT
find "$RUNS_ROOT" -mindepth 1 -maxdepth 1 -type d -name "${PROFILE_ID}_*" -printf '%f\n' \
  | sort >"$BEFORE_RUNS"

"$PY" -B scripts/create_v27527_yolov7_final_canary_profile.py \
  --source-profile "$SOURCE_PROFILE" \
  --out "$PROFILE" \
  --preflight-dir "$PREFLIGHT" \
  --profile-id "$PROFILE_ID"

"$PY" -B scripts/run_evaluation_workflow.py \
  --profile "$PROFILE" \
  --out "$RUNS_ROOT" \
  --profile-driven \
  --require-run-mode final \
  --require-fresh-run \
  --execution-mode generate_and_run 2>&1 | tee "$LOG"

RUN_DIR="$("$PY" -B - "$RUNS_ROOT" "$PROFILE_ID" "$BEFORE_RUNS" <<'PY'
import sys
from pathlib import Path
root = Path(sys.argv[1]).resolve(strict=True)
prefix = sys.argv[2] + "_"
before = set(Path(sys.argv[3]).read_text(encoding="utf-8").splitlines())
rows = sorted(
    (p.resolve(strict=True) for p in root.iterdir() if p.is_dir() and p.name.startswith(prefix) and p.name not in before),
    key=lambda p: p.name,
)
if len(rows) != 1:
    raise SystemExit(f"expected one canary run, found {len(rows)}: {rows}")
print(rows[0])
PY
)"

DEBUG_PACK="$PACK_ROOT/$(basename -- "$RUN_DIR")_debug_pack.zip"
"$PY" -B scripts/create_evaluation_debug_pack.py \
  --eval-run-dir "$RUN_DIR" \
  --out "$DEBUG_PACK"

"$PY" -B - "$RUN_DIR" "$DEBUG_PACK" "$LOG" "${ONNX_SPLITPOINT_REQUIRE_WARM_HIT:-0}" <<'PY'
import json
import re
import sys
import zipfile
from pathlib import Path

run = Path(sys.argv[1]).resolve(strict=True)
pack = Path(sys.argv[2]).resolve(strict=True)
log = Path(sys.argv[3]).resolve(strict=True)
require_warm = sys.argv[4].strip().lower() in {"1", "true", "yes"}
readiness = json.loads((run / "campaign" / "campaign_readiness.json").read_text(encoding="utf-8"))
run_status = json.loads((run / "reports" / "run_status_summary.json").read_text(encoding="utf-8"))
native = json.loads((run / "reports" / "native_evidence_status.json").read_text(encoding="utf-8"))
quality = json.loads((run / "quality_management" / "central_quality_summary.json").read_text(encoding="utf-8"))
assert readiness.get("claim_scope") == "evaluated_matrix", readiness.get("claim_scope")
assert readiness.get("final_ready") is True, readiness.get("status")
assert run_status.get("technical_status") == "ok", run_status
assert run_status.get("runtime_complete") is True, run_status
assert int(run_status.get("blocking_reason_count") or 0) == 0, run_status
assert int(quality.get("request_count") or quality.get("requested_count") or 0) > 0, quality
assert int(quality.get("failed_count") or 0) == 0, quality
assert native.get("technical_status") not in {"failed", "error", "blocked"}, native
with zipfile.ZipFile(pack) as archive:
    assert archive.testzip() is None
log_text = log.read_text(encoding="utf-8", errors="replace")
generic_builds = len(re.findall(r"\[native-trt\]\s+building\s+(?:full|part1|part2)\b", log_text))
persistent_cache_marker = "/_onnx_splitpoint_cache/tensorrt_managed_"
expected_generic_sessions = (
    "full:tensorrt",
    "part1:tensorrt",
    "part2:tensorrt",
)
generic_session_counts = {name: 0 for name in expected_generic_sessions}
generic_session_hit_counts = {name: 0 for name in expected_generic_sessions}
generic_session_sources = {name: set() for name in expected_generic_sessions}
for path in sorted(
    run.glob("models/*/benchmark_results/benchmark_results_ort_tensorrt_*.json")
):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        continue
    if isinstance(value, list):
        rows = value
    elif isinstance(value, dict) and isinstance(value.get("rows"), list):
        rows = value["rows"]
    elif isinstance(value, dict) and isinstance(value.get("results"), list):
        rows = value["results"]
    elif isinstance(value, dict):
        rows = [value]
    else:
        rows = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        native_trt = row.get("native_tensorrt")
        sessions = native_trt.get("sessions") if isinstance(native_trt, dict) else None
        if not isinstance(sessions, dict):
            continue
        for name in expected_generic_sessions:
            session = sessions.get(name)
            if not isinstance(session, dict):
                continue
            generic_session_counts[name] += 1
            cache_root = str(session.get("cache_root") or "")
            engine = str(session.get("engine") or "")
            persistent = (
                persistent_cache_marker in cache_root
                or persistent_cache_marker in engine
            )
            if (
                session.get("cache_hit") is True
                and session.get("runtime") == "native_tensorrt"
                and persistent
            ):
                generic_session_hit_counts[name] += 1
                generic_session_sources[name].add(path.relative_to(run).as_posix())

generic_cross_run_hits_complete = all(
    generic_session_counts[name] > 0
    and generic_session_hit_counts[name] == generic_session_counts[name]
    for name in expected_generic_sessions
)

expected_quality_setups = {
    "orin_nx_deepx_m1_01",
    "orin_nx_hailo10_01",
    "orin_nx_hailo8_01",
}
quality_initial_build_ids = set()
quality_split_hit_setups = set()
quality_split_hit_sources = set()
for path in sorted(run.rglob("validation_report.json")):
    relative = path.relative_to(run)
    setup_id = next(
        (part for part in relative.parts if part in expected_quality_setups),
        "",
    )
    canonical_host_report = (
        bool(setup_id)
        and "case_reports" in relative.parts
        and "lean_bundle" not in relative.parts
    )
    if not canonical_host_report:
        continue
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        continue
    native_trt = value.get("native_tensorrt") if isinstance(value, dict) else None
    sessions = native_trt.get("sessions") if isinstance(native_trt, dict) else None
    split_session = sessions.get("part2:tensorrt") if isinstance(sessions, dict) else None
    if isinstance(split_session, dict):
        engine = str(split_session.get("engine") or "")
        if (
            split_session.get("cache_hit") is True
            and split_session.get("build_disabled") is True
            and split_session.get("runtime") == "native_tensorrt"
            and persistent_cache_marker in engine
            and "/native_split_quality/" in engine
        ):
            quality_split_hit_setups.add(setup_id)
            quality_split_hit_sources.add(relative.as_posix())
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            meta = item.get("native_trt_meta")
            if isinstance(meta, dict) and isinstance(meta.get("build"), dict) and str(meta["build"].get("reason") or "") == "initial":
                engine = str(meta.get("engine") or "")
                quality_initial_build_ids.add(engine or relative.as_posix())
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)
quality_initial_builds = len(quality_initial_build_ids)
quality_cross_run_hits_complete = quality_split_hit_setups == expected_quality_setups
zero_physical_builds = generic_builds == 0 and quality_initial_builds == 0
warm_ok = (
    zero_physical_builds
    and generic_cross_run_hits_complete
    and quality_cross_run_hits_complete
)
observation = {
    "schema": "onnx-splitpoint/v27527-yolov7-warm-cache-observation",
    "schema_version": 2,
    "generic_physical_trt_build_count": generic_builds,
    "quality_initial_trt_build_count": quality_initial_builds,
    "zero_physical_trt_builds": zero_physical_builds,
    "expected_generic_sessions": list(expected_generic_sessions),
    "generic_session_observation_counts": generic_session_counts,
    "generic_session_hit_counts": generic_session_hit_counts,
    "generic_session_hit_sources": {
        name: sorted(generic_session_sources[name])
        for name in expected_generic_sessions
    },
    "generic_cross_run_hit_evidence_complete": generic_cross_run_hits_complete,
    "expected_quality_setup_ids": sorted(expected_quality_setups),
    "quality_split_hit_setup_ids": sorted(quality_split_hit_setups),
    "quality_split_hit_sources": sorted(quality_split_hit_sources),
    "quality_cross_run_hit_evidence_complete": quality_cross_run_hits_complete,
    "warm_cache_candidate": warm_ok,
}
(run / "reports" / "v27527_warm_cache_observation.json").write_text(json.dumps(observation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print("CANARY_TECHNICAL_RESULT=PASS")
print(f"CANARY_QUALITY_DECISION={quality.get('quality_decision', 'unknown')}")
print(f"CANARY_SCIENTIFIC_STATUS={native.get('scientific_status', 'unknown')}")
print(f"PHYSICAL_TRT_BUILDS={generic_builds + quality_initial_builds}")
print(
    "GENERIC_TRT_CROSS_RUN_HITS="
    + ",".join(
        f"{name}={generic_session_hit_counts[name]}/{generic_session_counts[name]}"
        for name in expected_generic_sessions
    )
)
print(
    "QUALITY_TRT_CROSS_RUN_HITS="
    f"{len(quality_split_hit_setups)}/{len(expected_quality_setups)}"
)
print(f"WARM_CACHE_CANDIDATE={'PASS' if warm_ok else 'SEED_OR_FAIL'}")
print("FINAL_AUTHORIZATION=NOT_GRANTED_REVIEW_CANARY_FIRST")
if require_warm and not warm_ok:
    raise SystemExit(
        "required warm-cache cross-run evidence failed: "
        f"physical_builds={generic_builds + quality_initial_builds}, "
        f"generic_hits={generic_session_hit_counts}, "
        f"quality_hit_setups={sorted(quality_split_hit_setups)}"
    )
PY

printf 'PROFILE=%s\nRUN_DIR=%s\nDEBUG_PACK=%s\n' "$PROFILE" "$RUN_DIR" "$DEBUG_PACK"
