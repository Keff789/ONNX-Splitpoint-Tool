#!/usr/bin/env bash
# Fresh v2.75.20 field run whose acceptance decision is exactly Full-only 18/18.

v27520_discovery_failure_rc() {
  local workflow_status="${1:?workflow RC required}"
  local discovery_status="${2:?discovery RC required}"
  if (( discovery_status == 0 )); then
    return 0
  fi
  if (( workflow_status > 1 )); then
    return "$workflow_status"
  fi
  return "$discovery_status"
}

v27520_secure_debug_pack_target() {
  local runs_root="${1:?RUNS_ROOT required}"
  local run_name="${2:?run name required}"
  local pack_dir
  pack_dir="$(mktemp -d -p "$runs_root" .v27520-debug-pack.XXXXXXXX)" \
    || return $?
  printf '%s/%s_v27520_debug_pack.zip\n' "$pack_dir" "$run_name"
}

v27520_terminal_wrapper_rc() {
  local workflow_status="${1:?workflow RC required}"
  local manifest_status="${2:?manifest status required}"
  local postcondition_status="${3:?postcondition RC required}"
  local native_status="${4:?native verification RC required}"
  if (( workflow_status > 1 )); then
    return "$workflow_status"
  fi
  if [[ "$manifest_status" == "failed" || "$manifest_status" == "cancelled" ]]; then
    if (( workflow_status != 0 )); then
      return "$workflow_status"
    fi
    return 1
  fi
  if (( postcondition_status != 0 || native_status != 0 )); then
    return 1
  fi
  return 0
}

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  printf '%s\n' \
    "SKIP: scripts/run_v27520_full_only_check.sh was sourced." \
    "Run it in its own process with the Evaluation Profile as argument." >&2
  return 0
fi

set -Eeuo pipefail

SOURCE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
PYTHON="$SOURCE_ROOT/.venv/bin/python"

if (( $# < 1 || $# > 3 )); then
  printf '%s\n' \
    "Usage: bash scripts/run_v27520_full_only_check.sh PROFILE [MODELS_ROOT [RUNS_ROOT]]" >&2
  exit 64
fi

PROFILE="$(realpath -e -- "$1")"
MODELS_ROOT="$(realpath -e -- "${2:-$HOME/Models}")"
RUNS_ROOT="${3:-$MODELS_ROOT/EvaluationRuns}"

if [[ ! -x "$PYTHON" ]]; then
  printf 'FEHLER: Tool-Venv fehlt: %s\n' "$PYTHON" >&2
  exit 69
fi

# Inspect the nearest existing parent without a probe file.  A genuinely
# read-only Models tree is supported when the third argument names a separate
# writable RUNS_ROOT; a read-only output target stops before mkdir/mktemp.
"$PYTHON" -B - "$SOURCE_ROOT" "$RUNS_ROOT" <<'PREFLIGHT_PY'
import sys
from pathlib import Path

source_root = Path(sys.argv[1]).resolve(strict=True)
sys.path.insert(0, str(source_root))
from onnx_splitpoint_tool.filesystem_admission import require_write_target

require_write_target(
    Path(sys.argv[2]).expanduser(),
    operation="v2.75.20 Full-only RUNS_ROOT",
    minimum_free_bytes=64 * 1024 * 1024,
    minimum_free_inodes=256,
)
PREFLIGHT_PY
mkdir -p -- "$RUNS_ROOT"
RUNS_ROOT="$(realpath -e -- "$RUNS_ROOT")"

cd -- "$SOURCE_ROOT"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"

RUN_SNAPSHOT=""
RUN_MARKER=""
RUN_STATUS_RESULT=""
RUN_DIR=""
run_manifest_status="not-discovered"
workflow_rc="not-started"
discovery_rc="not-run"
postcondition_rc="not-run"
native_verify_rc="not-run"
debug_pack_rc="not-run"
DEBUG_PACK_PATH=""

report_wrapper_exit() {
  local wrapper_rc=$?
  set +e
  if [[ -n "$RUN_MARKER" ]]; then
    rm -f -- "$RUN_MARKER"
  fi
  if [[ -n "$RUN_SNAPSHOT" ]]; then
    rm -f -- "$RUN_SNAPSHOT"
  fi
  if [[ -n "$RUN_STATUS_RESULT" ]]; then
    rm -f -- "$RUN_STATUS_RESULT"
  fi
  if (( wrapper_rc != 0 )) && [[ -n "$RUN_DIR" && -d "$RUN_DIR" ]]; then
    DEBUG_PACK_PATH="$(
      v27520_secure_debug_pack_target \
        "$RUNS_ROOT" "$(basename -- "$RUN_DIR")"
    )"
    debug_pack_rc=$?
    if (( debug_pack_rc == 0 )); then
      "$PYTHON" -B scripts/create_evaluation_debug_pack.py \
        --eval-run-dir "$RUN_DIR" \
        --out "$DEBUG_PACK_PATH"
      debug_pack_rc=$?
    else
      DEBUG_PACK_PATH=""
    fi
  fi
  printf 'RUN_DIR: %s\n' "${RUN_DIR:-nicht ermittelt}"
  printf 'Workflow-RC: %s\n' "$workflow_rc"
  printf 'Run-Discovery-RC: %s\n' "$discovery_rc"
  printf 'Run-Manifest-Status: %s\n' "$run_manifest_status"
  printf 'Postcondition-RC: %s\n' "$postcondition_rc"
  printf 'Native-18/18-RC: %s\n' "$native_verify_rc"
  printf 'Debug-Pack-RC: %s\n' "$debug_pack_rc"
  printf 'Wrapper-RC: %d\n' "$wrapper_rc"
  if [[ -n "$RUN_DIR" && -d "$RUN_DIR" ]]; then
    mapfile -t debug_paths < <(
      {
        find "$RUN_DIR" -maxdepth 5 \
          \( -type f -o -type d \) \
          \( -iname '*debug*pack*' -o -iname '*failure*pack*' \) \
          -print 2>/dev/null
        if [[ -n "$DEBUG_PACK_PATH" && -e "$DEBUG_PACK_PATH" ]]; then
          printf '%s\n' "$DEBUG_PACK_PATH"
        fi
      } | sort -u
    )
    if (( ${#debug_paths[@]} > 0 )); then
      printf 'Debug-Pack-Pfade:\n'
      printf '  %s\n' "${debug_paths[@]}"
    else
      printf 'Debug-Pack-Pfade: keine gefunden\n'
    fi
  else
    printf 'Debug-Pack-Pfade: kein Run-Verzeichnis verfügbar\n'
  fi
  return "$wrapper_rc"
}
trap report_wrapper_exit EXIT

PY="$PYTHON" bash scripts/run_v27520_small_acceptance.sh

"$PYTHON" -B - "$PROFILE" <<'PROFILE_PY'
import sys
from pathlib import Path

import yaml

from onnx_splitpoint_tool.native_full_quality import (
    resolve_native_full_plan,
    resolve_native_split_plan,
)

profile_path = Path(sys.argv[1]).resolve(strict=True)
payload = yaml.safe_load(profile_path.read_text(encoding="utf-8")) or {}
if not isinstance(payload, dict):
    raise SystemExit("Evaluation Profile muss ein YAML-Objekt sein")
if "targets" in payload:
    raise SystemExit(
        "Full-only-Feldprofil darf keine top-level targets enthalten"
    )

expected_profiles = {
    "ort_tensorrt": (
        "same_backend_reference", "tensorrt", "tensorrt", "tensorrt",
    ),
    "hailo8": (
        "same_backend_reference", "hailo8", "hailo8", "hailo8",
    ),
    "hailo10": (
        "same_backend_reference", "hailo10", "hailo10", "hailo10",
    ),
    "deepx_m1_full": (
        "same_backend_reference", "deepx_m1", "deepx_m1", "deepx_m1",
    ),
}
expected_models = {
    "resnet50": "classification",
    "yolo26s": "detection",
    "yolov7_paper": "detection",
}

def enabled(row):
    value = row.get("enabled", True)
    if value is False:
        return False
    return str(value).strip().lower() not in {
        "0", "false", "no", "off", "disabled",
    }

def token(value):
    return str(value or "").strip().lower().replace("-", "_")

raw_profiles = payload.get("run_profiles")
if not isinstance(raw_profiles, list):
    raise SystemExit("Full-only-Feldprofil benötigt run_profiles als Liste")
selected_profiles = []
for index, row in enumerate(raw_profiles):
    if not isinstance(row, dict):
        raise SystemExit(f"run_profiles[{index}] muss ein Objekt sein")
    if enabled(row):
        selected_profiles.append(row)
observed_profiles = {}
for row in selected_profiles:
    run_id = token(row.get("id"))
    if not run_id or run_id in observed_profiles:
        raise SystemExit(
            "Full-only-Run-IDs fehlen oder sind doppelt: " + repr(run_id)
        )
    observed_profiles[run_id] = (
        token(row.get("type")),
        token(row.get("full")),
        token(row.get("stage1")),
        token(row.get("stage2")),
    )
if observed_profiles != expected_profiles:
    raise SystemExit(
        "Full-only-Feldprofil muss exakt die vier aktivierten Full-Zeilen "
        "ort_tensorrt,hailo8,hailo10,deepx_m1_full enthalten; beobachtet="
        + repr(observed_profiles)
    )

model_suite = payload.get("model_suite")
primary_models = (
    model_suite.get("primary") if isinstance(model_suite, dict) else None
)
if not isinstance(primary_models, list):
    raise SystemExit("Full-only-Feldprofil benötigt model_suite.primary")
observed_models = {}
for index, row in enumerate(primary_models):
    if not isinstance(row, dict):
        raise SystemExit(f"model_suite.primary[{index}] muss ein Objekt sein")
    if not enabled(row):
        continue
    model_id = token(row.get("id"))
    if not model_id or model_id in observed_models:
        raise SystemExit(
            "Full-only-Modell-IDs fehlen oder sind doppelt: " + repr(model_id)
        )
    observed_models[model_id] = token(row.get("task"))
reserve_models = (
    model_suite.get("reserve") if isinstance(model_suite, dict) else None
)
enabled_reserve = [
    row for row in list(reserve_models or [])
    if isinstance(row, dict) and enabled(row)
]
if observed_models != expected_models or enabled_reserve:
    raise SystemExit(
        "Full-only-Feldprofil muss exakt ResNet50, YOLO26s und YOLOv7 "
        "mit classification/detection/detection enthalten; beobachtet="
        + repr(observed_models)
        + " reserve=" + repr(enabled_reserve)
    )

split = resolve_native_split_plan(payload)
full = resolve_native_full_plan(payload)
expected = {
    "hailo8": ("hailo8", "tensorrt"),
    "hailo10h": ("hailo10h", "tensorrt"),
    "deepx": ("deepx", "tensorrt"),
}
if split.selected_split_backends:
    raise SystemExit(
        "Profil ist nicht Full-only; ausgewählte Native-Splits="
        + repr(list(split.selected_split_backends))
    )
if full.backends_by_producer != expected:
    raise SystemExit(
        "Full-only-Matrix muss exakt 3x2 sein; beobachtet="
        + repr(full.backends_by_producer)
    )
print("PASS Profilvertrag: Split=0, Full-Setups=3x2")
PROFILE_PY

printf '%s\n' \
  "Prüfe die exakte Native-Runtime-Closure auf Hailo-8, Hailo-10H und DeepX."
"$PYTHON" -B scripts/preflight_v27520_native_remotes.py \
  --profile "$PROFILE"

RUN_SNAPSHOT="$(mktemp -p "$RUNS_ROOT" .v27520-full-only-snapshot-XXXXXXXX)"
"$PYTHON" -B - "$RUNS_ROOT" "$RUN_SNAPSHOT" <<'RUN_SNAPSHOT_PY'
import json
import stat
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve(strict=True)
snapshot_path = Path(sys.argv[2]).resolve(strict=True)
children = []
for child in sorted(root.iterdir(), key=lambda path: path.name):
    child_stat = child.stat(follow_symlinks=False)
    children.append({
        "name": child.name,
        "device": int(child_stat.st_dev),
        "inode": int(child_stat.st_ino),
        "is_directory": stat.S_ISDIR(child_stat.st_mode),
        "is_symlink": stat.S_ISLNK(child_stat.st_mode),
    })
snapshot_path.write_text(
    json.dumps({
        "schema": "onnx-splitpoint/full-only-run-directory-snapshot",
        "schema_version": 1,
        "root": str(root),
        "children": children,
    }, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
RUN_SNAPSHOT_PY

RUN_MARKER="$(mktemp -p "$RUNS_ROOT" .v27520-full-only-marker-XXXXXXXX)"
RUN_STATUS_RESULT="$(
  mktemp -p "$RUNS_ROOT" .v27520-full-only-status-XXXXXXXX
)"

printf '%s\n' \
  "Starte frischen Smoke zur Neumaterialisierung der Full-Quality-Verträge." \
  "Die Abnahme entscheidet ausschließlich über die 18 Full-only-Zeilen."

workflow_rc=0
if "$PYTHON" -B -m onnx_splitpoint_tool.workflow.run_evaluation \
    --profile "$PROFILE" \
    --models-root "$MODELS_ROOT" \
    --out "$RUNS_ROOT" \
    --profile-driven \
    --require-run-mode smoke \
    --require-fresh-run \
    --execution-mode generate_and_run; then
  workflow_rc=0
else
  workflow_rc=$?
fi

# A normal profile-driven fresh run must choose its own collision-safe run ID.
# Resolve exactly one newly written authoritative manifest; do not guess from
# a directory prefix or accept an unrelated concurrently updated run.
set +e
RUN_DIR="$(
  "$PYTHON" -B - "$RUN_MARKER" "$RUNS_ROOT" "$PROFILE" "$RUN_SNAPSHOT" "$workflow_rc" "$RUN_STATUS_RESULT" <<'PYTHON'
import json
import sys
from pathlib import Path

marker = Path(sys.argv[1]).resolve(strict=True)
root = Path(sys.argv[2]).resolve(strict=True)
profile = Path(sys.argv[3]).resolve(strict=True)
snapshot_path = Path(sys.argv[4]).resolve(strict=True)
workflow_rc = int(sys.argv[5])
status_result_path = Path(sys.argv[6])
if (
    status_result_path.is_symlink()
    or not status_result_path.is_file()
    or status_result_path.resolve(strict=True).parent != root
):
    raise SystemExit(f"Unsicheres Run-Status-Ziel: {status_result_path}")
cutoff_ns = marker.stat().st_mtime_ns

candidates = []
ignored = []

def reject_duplicate_pairs(pairs):
    payload = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"duplicate JSON key: {key}")
        payload[key] = value
    return payload

def read_object(path):
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicate_pairs,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        return None, f"unreadable:{type(exc).__name__}"
    if not isinstance(payload, dict):
        return None, "not_object"
    return payload, "ok"

snapshot, snapshot_state = read_object(snapshot_path)
if snapshot is None:
    raise SystemExit(f"Run-Verzeichnis-Snapshot {snapshot_state}: {snapshot_path}")
if (
    snapshot.get("schema")
    != "onnx-splitpoint/full-only-run-directory-snapshot"
    or snapshot.get("schema_version") != 1
    or Path(str(snapshot.get("root") or "")).resolve() != root
    or not isinstance(snapshot.get("children"), list)
):
    raise SystemExit(f"Run-Verzeichnis-Snapshot ungültig: {snapshot_path}")
before_names = {
    str(row.get("name") or "")
    for row in snapshot["children"]
    if isinstance(row, dict) and str(row.get("name") or "")
}
before_identities = {
    (int(row.get("device")), int(row.get("inode")))
    for row in snapshot["children"]
    if isinstance(row, dict)
    and row.get("is_directory") is True
    and row.get("is_symlink") is False
}

for manifest in sorted(root.glob("*/run_manifest.json")):
    if manifest.is_symlink() or not manifest.is_file():
        continue
    if manifest.stat().st_mtime_ns < cutoff_ns:
        continue
    run_dir = manifest.parent
    try:
        resolved_run_dir = run_dir.resolve(strict=True)
    except OSError as exc:
        ignored.append((str(run_dir), f"unsafe_path:{type(exc).__name__}"))
        continue
    if run_dir.is_symlink() or resolved_run_dir.parent != root:
        ignored.append((str(run_dir), "unsafe_path"))
        continue
    resolved_stat = resolved_run_dir.stat()
    if resolved_run_dir.name in before_names:
        ignored.append((str(resolved_run_dir), "preexisting_directory_name"))
        continue
    if (int(resolved_stat.st_dev), int(resolved_stat.st_ino)) in before_identities:
        ignored.append((str(resolved_run_dir), "preexisting_directory_identity"))
        continue

    payload, manifest_state = read_object(manifest)
    if payload is None:
        ignored.append((str(resolved_run_dir), f"manifest_{manifest_state}"))
        continue
    manifest_status = str(payload.get("status") or "").strip().lower()
    diagnostic_status = manifest_status in {"failed", "cancelled"}
    summary_path = resolved_run_dir / "reports" / "run_status_summary.json"
    summary, summary_state = read_object(summary_path)
    if summary is None and not diagnostic_status:
        ignored.append((str(resolved_run_dir), f"summary_{summary_state}"))
        continue
    if summary is None:
        summary = {}

    raw_run_dir = str(payload.get("run_dir") or "").strip()
    raw_profile = str(payload.get("profile_path") or "").strip()
    try:
        manifest_run_dir = Path(raw_run_dir).expanduser().resolve(strict=True)
    except (OSError, RuntimeError):
        manifest_run_dir = Path()
    try:
        manifest_profile = Path(raw_profile).expanduser().resolve(strict=True)
    except (OSError, RuntimeError):
        manifest_profile = Path()

    summary_status = str(summary.get("status") or "").strip().lower()
    terminal_statuses = (
        {"ok", "partial"}
        if workflow_rc == 0
        else {"ok", "partial", "failed", "cancelled"}
    )
    checks = {
        "schema": payload.get("schema")
            == "onnx-splitpoint/evaluation-run-manifest",
        "run_id": str(payload.get("run_id") or "") == resolved_run_dir.name,
        "run_dir": manifest_run_dir == resolved_run_dir,
        "profile_path": manifest_profile == profile,
        # A controlled partial may still be decided by the exact Full-only
        # postconditions below. Failed/cancelled rows are retained only so the
        # wrapper can produce diagnostics and are rejected by the final gate.
        "terminal_status": manifest_status in terminal_statuses,
        "tool_version": str(
            payload.get("current_tool_version") or payload.get("tool_version") or ""
        ) == "2.75.20",
        "workflow_version": str(
            payload.get("current_workflow_version")
            or payload.get("workflow_version") or ""
        ) == "v2.75.20-remote-boundary-plan-finalization-repair",
        "fresh": (payload.get("options") or {}).get("require_fresh_run") is True,
        "execution_mode": str(
            (payload.get("options") or {}).get("execution_mode") or ""
        ) == "generate_and_run",
        "summary_schema": diagnostic_status or summary.get("schema")
            == "onnx-splitpoint/run-status-summary",
        "summary_run_id": diagnostic_status or str(summary.get("run_id") or "")
            == resolved_run_dir.name,
        "summary_status": diagnostic_status
            or summary_status == manifest_status,
    }
    failed = sorted(name for name, ok in checks.items() if not ok)
    if failed:
        ignored.append((str(resolved_run_dir), ",".join(failed)))
        continue
    candidates.append((resolved_run_dir, payload))

if len(candidates) != 1:
    names = [str(path) for path, _payload in candidates]
    raise SystemExit(
        "Genau ein neuer vollständiger EvaluationRun wurde erwartet; "
        f"gefunden={len(candidates)}: {names}; ignoriert={ignored}"
    )

run_dir, payload = candidates[0]
status_result_path.write_text(
    str(payload.get("status") or "").strip().lower() + "\n",
    encoding="utf-8",
)
print(run_dir)
PYTHON
)"
discovery_rc=$?
set -e

if (( discovery_rc != 0 )); then
  failure_rc=1
  v27520_discovery_failure_rc "$workflow_rc" "$discovery_rc" \
    || failure_rc=$?
  exit "$failure_rc"
fi

if ! IFS= read -r run_manifest_status < "$RUN_STATUS_RESULT"; then
  printf 'FAIL: Run-Manifest-Status konnte nicht gelesen werden.\n' >&2
  exit 1
fi
case "$run_manifest_status" in
  ok|partial|failed|cancelled) ;;
  *)
    printf 'FAIL: Ungültiger Run-Manifest-Status: %s\n' \
      "$run_manifest_status" >&2
    exit 1
    ;;
esac

set +e
"$PYTHON" -B - "$RUN_DIR" <<'POSTCONDITION_PY'
import json
import sys
from pathlib import Path

from onnx_splitpoint_tool.native_output_endpoint import (
    load_authoritative_output_contract,
)
from onnx_splitpoint_tool.workflow.artifacts import sha256_json

run_dir = Path(sys.argv[1]).resolve(strict=True)
contract_expectations = {
    "resnet50": ("classification", "classification_logits"),
    "yolo26s": ("detection", "decoded_nms"),
    "yolov7_paper": ("detection", "raw_head"),
}
models = tuple(contract_expectations)
setups = ("hailo8", "hailo10h", "deepx")
expected_run_ids = {
    "ort_cpu", "ort_tensorrt", "hailo8", "hailo10", "deepx_m1_full",
}

def load(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"JSON-Objekt erwartet: {path}")
    return payload

def checked_plan(path: Path, *, label: str):
    plan = load(path)
    runs = plan.get("runs")
    planned_runs = plan.get("planned_runs")
    if (
        not isinstance(runs, list)
        or not isinstance(planned_runs, list)
        or any(not isinstance(row, dict) for row in runs)
        or any(not isinstance(row, dict) for row in planned_runs)
    ):
        raise SystemExit(f"Plan-Aliasse fehlen oder sind ungültig {label}: {path}")
    if runs != planned_runs:
        raise SystemExit(f"runs/planned_runs weichen ab {label}: {path}")
    run_ids = [str(row.get("id") or "").strip() for row in runs]
    if (
        len(run_ids) != len(expected_run_ids)
        or len(set(run_ids)) != len(run_ids)
        or set(run_ids) != expected_run_ids
    ):
        raise SystemExit(
            f"Full-only-Runplan muss exakt fünf IDs enthalten {label}: "
            f"beobachtet={run_ids!r} erwartet={sorted(expected_run_ids)!r}"
        )
    cpu_rows = [row for row in runs if row.get("id") == "ort_cpu"]
    if len(cpu_rows) != 1:
        raise SystemExit(f"Exakte ORT-CPU-Rezeptzeile fehlt {label}: {path}")
    cpu = cpu_rows[0]
    cpu_contract = {
        "semantic_reference_only": True,
        "canonical_cpu_reference": True,
        "automatic_reference": True,
        "performance_eligible": False,
        "energy_eligible": False,
        "ranking_eligible": False,
        "pareto_eligible": False,
        "execution_location": "central_management",
    }
    mismatched_cpu = {
        key: (cpu.get(key), expected)
        for key, expected in cpu_contract.items()
        if cpu.get(key) != expected
    }
    if mismatched_cpu:
        raise SystemExit(
            f"ORT-CPU-Rezept ist nicht rein semantisch {label}: "
            f"{mismatched_cpu!r}"
        )
    invariant = plan.get("management_cpu_reference_invariant")
    expected_invariant = {
        "status": "verified",
        "recipe_count": 1,
        "execution_location": "central_management",
        "performance_dispatch_allowed": False,
        "run_plan_sha256": sha256_json(runs),
    }
    if not isinstance(invariant, dict) or any(
        invariant.get(key) != value
        for key, value in expected_invariant.items()
    ):
        raise SystemExit(
            f"ORT-CPU-Planinvariante ungültig {label}: "
            f"beobachtet={invariant!r} erwartet={expected_invariant!r}"
        )
    return runs

def checked_benchmark_set(path: Path, expected_runs, *, label: str):
    contract = load(path)
    planned_runs = contract.get("planned_runs")
    if planned_runs != expected_runs:
        raise SystemExit(
            f"benchmark_set.json spiegelt den Runplan nicht {label}: {path}"
        )
    expected_hash = sha256_json(expected_runs)
    invariant = contract.get("management_cpu_reference_invariant")
    if (
        not isinstance(invariant, dict)
        or invariant.get("run_plan_sha256") != expected_hash
    ):
        raise SystemExit(
            f"benchmark_set.json trägt nicht den finalen Runplan-Hash "
            f"{label}: {path}"
        )
    embedded = contract.get("plan")
    if isinstance(embedded, dict) and (
        embedded.get("runs") != expected_runs
        or embedded.get("planned_runs") != expected_runs
        or (
            embedded.get("management_cpu_reference_invariant") or {}
        ).get("run_plan_sha256") != expected_hash
    ):
        raise SystemExit(
            f"benchmark_set.json Embedded-Plan weicht ab {label}: {path}"
        )
    return contract

def executable_suite_dir(formal_contract, *, model: str):
    raw = str(
        formal_contract.get("legacy_suite_dir")
        or formal_contract.get("suite_dir")
        or formal_contract.get("generated_suite_dir")
        or ""
    ).strip()
    if not raw:
        raise SystemExit(f"Ausführbares BenchmarkSet fehlt im Formalvertrag: {model}")
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = run_dir / candidate
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(run_dir)
    except ValueError as exc:
        raise SystemExit(
            f"Ausführbares BenchmarkSet liegt außerhalb des EvaluationRun: {resolved}"
        ) from exc
    if not resolved.is_dir():
        raise SystemExit(f"Ausführbares BenchmarkSet ist kein Verzeichnis: {resolved}")
    return resolved

stage_path = run_dir / "reports" / "native_producer_stage.json"
stage = load(stage_path)
rows = stage.get("expected_native_rows")
if not isinstance(rows, list):
    raise SystemExit(f"Native-Plan fehlt: {stage_path}")
split_rows = [row for row in rows if row.get("execution_mode") == "native_split"]
full_rows = [
    row for row in rows
    if row.get("execution_mode") == "native_full_baseline"
]
if (len(split_rows), len(full_rows), len(rows)) != (0, 18, 18):
    raise SystemExit(
        "Native-Matrix falsch: "
        f"split={len(split_rows)} full={len(full_rows)} total={len(rows)}"
    )
if stage.get("native_split_requires_single_part2_input") is not False:
    raise SystemExit("Full-only darf keine Native-Split-Single-Input-Pflicht setzen")
if stage.get("split_backends") != []:
    raise SystemExit(f"Split-Backends müssen leer sein: {stage.get('split_backends')!r}")

formal_runs_by_model = {}
for model in models:
    status_path = (
        run_dir / "quality_management" / "references" / model
        / "management_cpu_reference_status.json"
    )
    status = load(status_path)
    if status.get("status") not in {"completed", "cache_hit"}:
        raise SystemExit(
            f"ORT-CPU-Referenz unvollständig {model}: "
            f"status={status.get('status')!r} errors={status.get('errors')!r}"
        )

    formal_dir = run_dir / "models" / model / "benchmark_set"
    formal_runs = checked_plan(
        formal_dir / "benchmark_plan.json",
        label=f"formal:{model}",
    )
    formal_contract = checked_benchmark_set(
        formal_dir / "benchmark_set.json",
        formal_runs,
        label=f"formal:{model}",
    )
    executable_dir = executable_suite_dir(formal_contract, model=model)
    executable_runs = checked_plan(
        executable_dir / "benchmark_plan.json",
        label=f"executable:{model}",
    )
    if executable_runs != formal_runs:
        raise SystemExit(
            f"Formal- und ausführbarer Runplan weichen ab: {model}"
        )
    checked_benchmark_set(
        executable_dir / "benchmark_set.json",
        formal_runs,
        label=f"executable:{model}",
    )
    formal_runs_by_model[model] = formal_runs

for setup in setups:
    for model in models:
        suite = run_dir / "native_producers" / setup / model / "benchmark_set"
        copied_runs = checked_plan(
            suite / "benchmark_plan.json",
            label=f"native-copy:{setup}:{model}",
        )
        if copied_runs != formal_runs_by_model[model]:
            raise SystemExit(
                f"Native BenchmarkSet spiegelt den Formalplan nicht: {suite}"
            )
        checked_benchmark_set(
            suite / "benchmark_set.json",
            copied_runs,
            label=f"native-copy:{setup}:{model}",
        )

        contracts = load(suite / "output_contracts.json")
        cuda = [
            row for row in contracts.get("contracts") or []
            if row.get("backend") == "cuda_ort"
            and row.get("variant") == "full"
            and row.get("contract_status") == "recorded"
        ]
        if len(cuda) != 1:
            raise SystemExit(f"Exakter TensorRT-Full-Vertrag fehlt: {suite}")
        task, expected_stage = contract_expectations[model]
        declaration = load_authoritative_output_contract(
            suite,
            backend="tensorrt",
            model_id=model,
            variant="full",
            task=task,
        )
        if (
            declaration.get("contract_resolution_status") != "attested"
            or declaration.get("authoritative_output_contract") is not True
            or declaration.get("backend") != "cuda_ort"
            or declaration.get("stage") != expected_stage
        ):
            raise SystemExit(
                "TensorRT-Full-Vertrag ist nicht strikt attestiert "
                f"{suite}: status="
                f"{declaration.get('contract_resolution_status')!r} reason="
                f"{declaration.get('contract_resolution_reason')!r} stage="
                f"{declaration.get('stage')!r} expected={expected_stage!r} "
                f"errors={declaration.get('contract_resolution_errors')!r}"
            )

print("PASS Produktionskette: Split=0 Full=18 ORT-CPU=3 TRT-Verträge=9")
POSTCONDITION_PY
postcondition_rc=$?

"$PYTHON" -B scripts/verify_native_full_rows.py \
  --run-dir "$RUN_DIR" \
  --scope all \
  --format table
native_verify_rc=$?
set -e

final_rc=0
v27520_terminal_wrapper_rc \
  "$workflow_rc" "$run_manifest_status" \
  "$postcondition_rc" "$native_verify_rc" \
  || final_rc=$?
if (( final_rc != 0 )); then
  printf 'FAIL Full-only: workflow=%s postcondition=%s native=%s\n' \
    "$workflow_rc" "$postcondition_rc" "$native_verify_rc" >&2
  exit "$final_rc"
fi

printf 'PASS Full-only passed=18/18: %s\n' "$RUN_DIR"
