#!/usr/bin/env bash
# Recover/verify immutable B5 evidence, then run only the YOLO11l Gate-A compiler canary.

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  printf '%s\n' \
    'SKIP: run_v2783_yolo11_hailo8_first_gate.sh was sourced.' \
    'Run it in its own Bash process.' >&2
  return 0
fi

set -Eeuo pipefail
umask 077

die() {
  local rc="$1"
  shift
  printf 'ERROR: %s\n' "$*" >&2
  exit "$rc"
}

SCRIPT_FILE="${BASH_SOURCE[0]}"
[[ -f "$SCRIPT_FILE" && ! -L "$SCRIPT_FILE" ]] ||
  die 66 "Launcher fehlt oder ist ein Symlink: $SCRIPT_FILE"
SCRIPT_DIR="$(cd -- "$(dirname -- "$SCRIPT_FILE")" && pwd -P)"
DEFAULT_TOOL="$(dirname -- "$SCRIPT_DIR")"

TOOL="${TOOL:-$DEFAULT_TOOL}"
SOURCE_RUN=/home/kmika/Models/EvaluationRuns/complete_set_v2782_canary_b5_20260827_193658
WORKFLOW_LOG="$SOURCE_RUN/evaluation_workflow.log"
EVIDENCE_INDEX=/home/kmika/.onnx_splitpoint_tool/build_evidence/v2783_yolo11_gate_a.json
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/kmika/Models/EvaluationRunsCanary}"
MODELS_ROOT="${MODELS_ROOT:-/home/kmika/Models}"
TOOL_PYTHON_REQUEST="${TOOL_PYTHON:-}"
PROFILE_SHA256=dc11012eae5fadbab546d979054df64764d416e632b9b660de0d09558567b283

require_absolute() {
  local label="$1"
  local value="$2"
  [[ "$value" == /* ]] || die 64 "$label muss absolut sein: $value"
}

require_absolute TOOL "$TOOL"
require_absolute SOURCE_RUN "$SOURCE_RUN"
require_absolute WORKFLOW_LOG "$WORKFLOW_LOG"
require_absolute EVIDENCE_INDEX "$EVIDENCE_INDEX"
require_absolute OUTPUT_ROOT "$OUTPUT_ROOT"
require_absolute MODELS_ROOT "$MODELS_ROOT"
[[ -d "$TOOL" && ! -L "$TOOL" ]] ||
  die 66 "Tool-Verzeichnis fehlt oder ist ein Symlink: $TOOL"
TOOL="$(realpath -e -- "$TOOL")"
TOOL_PYTHON="${TOOL_PYTHON_REQUEST:-$TOOL/.venv/bin/python}"
require_absolute TOOL_PYTHON "$TOOL_PYTHON"
[[ -x "$TOOL_PYTHON" ]] || die 69 "Tool-Python fehlt: $TOOL_PYTHON"
[[ -d "$SOURCE_RUN" && ! -L "$SOURCE_RUN" ]] ||
  die 66 "Quelllauf fehlt oder ist ein Symlink: $SOURCE_RUN"
SOURCE_RUN="$(realpath -e -- "$SOURCE_RUN")"
[[ -d "$MODELS_ROOT" && ! -L "$MODELS_ROOT" ]] ||
  die 66 "Modellverzeichnis fehlt oder ist ein Symlink: $MODELS_ROOT"
MODELS_ROOT="$(realpath -e -- "$MODELS_ROOT")"

PROFILE_PATH="$TOOL/profiles/yolo11l_v2783_hailo8_first_b5_gate_a.yaml"
if ! "$TOOL_PYTHON" -B - "$PROFILE_PATH" "$PROFILE_SHA256" <<'PY'
import hashlib
import os
import stat
import sys

path, expected = sys.argv[1:]
flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
fd = os.open(path, flags)
try:
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode):
        raise SystemExit("Gate-A profile is not a regular file")
    digest = hashlib.sha256()
    while True:
        block = os.read(fd, 1024 * 1024)
        if not block:
            break
        digest.update(block)
finally:
    os.close(fd)
if digest.hexdigest() != expected:
    raise SystemExit("Gate-A profile SHA-256 mismatch")
PY
then
  die 66 "Eingefrorenes Gate-A-Profil fehlt, ist unsicher oder bytefremd"
fi

RECOVERY="$TOOL/scripts/recover_b5_build_evidence.py"
WORKFLOW="$TOOL/scripts/run_evaluation_workflow.py"
GATE_VERIFIER="$TOOL/scripts/verify_v2783_yolo11_gate_a_output.py"
for required_file in "$RECOVERY" "$WORKFLOW" "$GATE_VERIFIER"; do
  [[ -f "$required_file" && ! -L "$required_file" ]] ||
    die 66 "Erforderliches Script fehlt oder ist ein Symlink: $required_file"
done

# Create a directory from / one component at a time. Every existing component
# is opened with O_NOFOLLOW before the next mkdirat/openat operation.
secure_directory() {
  "$TOOL_PYTHON" -B - "$1" <<'PY'
import os
import stat
import sys

path = sys.argv[1]
if not os.path.isabs(path) or os.path.normpath(path) != path:
    raise SystemExit(f"unsafe non-canonical absolute directory: {path}")
components = path.split("/")[1:]
if any(value in {"", ".", ".."} for value in components):
    raise SystemExit(f"unsafe directory component: {path}")
flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
parent_fd = os.open("/", flags)
try:
    for component in components:
        try:
            child_fd = os.open(component, flags, dir_fd=parent_fd)
        except FileNotFoundError:
            os.mkdir(component, 0o700, dir_fd=parent_fd)
            child_fd = os.open(component, flags, dir_fd=parent_fd)
        info = os.fstat(child_fd)
        if not stat.S_ISDIR(info.st_mode):
            os.close(child_fd)
            raise SystemExit(f"unsafe non-directory component: {path}")
        os.close(parent_fd)
        parent_fd = child_fd
finally:
    os.close(parent_fd)
print(path)
PY
}

OUTPUT_ROOT="$(secure_directory "$OUTPUT_ROOT")" ||
  die 67 "Canary-Ausgabewurzel enthält einen unsicheren Pfadbestandteil"

EVIDENCE_PARENT="$(dirname -- "$EVIDENCE_INDEX")"
EVIDENCE_PARENT="$(secure_directory "$EVIDENCE_PARENT")" ||
  die 67 "Evidence-Verzeichnis enthält einen unsicheren Pfadbestandteil"
EVIDENCE_INDEX="$EVIDENCE_PARENT/$(basename -- "$EVIDENCE_INDEX")"
[[ ! -L "$EVIDENCE_INDEX" ]] ||
  die 67 "Evidence-Index ist ein Symlink: $EVIDENCE_INDEX"

STAMP="$(date -u +%Y%m%d_%H%M%S_%N)"
GATE_OUTPUT="$OUTPUT_ROOT/v2783_yolo11_hailo8_first_gate_a_${STAMP}_$$"
[[ ! -e "$GATE_OUTPUT" && ! -L "$GATE_OUTPUT" ]] ||
  die 73 "Einmaliges Canary-Ziel existiert bereits: $GATE_OUTPUT"
GATE_OUTPUT="$(secure_directory "$GATE_OUTPUT")" ||
  die 73 "Einmaliges Canary-Ziel konnte nicht sicher erstellt werden"

RECOVERY_RESULT="$GATE_OUTPUT/build_evidence_recovery.json"
RECOVERY_MODE=verify
recovery_rc=0
if [[ -e "$EVIDENCE_INDEX" ]]; then
  [[ -f "$EVIDENCE_INDEX" && ! -L "$EVIDENCE_INDEX" ]] ||
    die 67 "Evidence-Index ist keine reguläre Datei: $EVIDENCE_INDEX"
  set +e
  "$TOOL_PYTHON" -B "$RECOVERY" verify \
    --index "$EVIDENCE_INDEX" \
    --run-dir "$SOURCE_RUN" > "$RECOVERY_RESULT"
  recovery_rc=$?
  set -e
else
  [[ -f "$WORKFLOW_LOG" && ! -L "$WORKFLOW_LOG" ]] ||
    die 66 "Quell-Workflow-Log fehlt oder ist ein Symlink: $WORKFLOW_LOG"
  WORKFLOW_LOG="$(realpath -e -- "$WORKFLOW_LOG")"
  RECOVERY_MODE=create
  set +e
  "$TOOL_PYTHON" -B "$RECOVERY" create \
    --run-dir "$SOURCE_RUN" \
    --workflow-log "$WORKFLOW_LOG" \
    --out "$EVIDENCE_INDEX" > "$RECOVERY_RESULT"
  recovery_rc=$?
  set -e
fi

if (( recovery_rc != 0 )); then
  printf 'BUILD_EVIDENCE_MODE=%s\nBUILD_EVIDENCE_RESULT=%s\n' \
    "$RECOVERY_MODE" "$RECOVERY_RESULT" >&2
  die 2 "Build-Evidence-Recovery/Verifikation fehlgeschlagen (rc=$recovery_rc)"
fi
[[ -f "$EVIDENCE_INDEX" && ! -L "$EVIDENCE_INDEX" ]] ||
  die 67 "Verifizierter Evidence-Index fehlt oder ist unsicher: $EVIDENCE_INDEX"
[[ -f "$RECOVERY_RESULT" && ! -L "$RECOVERY_RESULT" ]] ||
  die 67 "Recovery-Receipt fehlt oder ist unsicher: $RECOVERY_RESULT"

RECOVERY_STATUS="$("$TOOL_PYTHON" -B - "$RECOVERY_RESULT" <<'PY'
import json
import os
import stat
import sys

path = sys.argv[1]
flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
fd = os.open(path, flags)
try:
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode) or info.st_size > 8 * 1024 * 1024:
        raise ValueError("unsafe recovery receipt")
    with os.fdopen(fd, "r", encoding="utf-8") as stream:
        fd = -1
        payload = json.load(stream)
finally:
    if fd >= 0:
        os.close(fd)
if not isinstance(payload, dict) or payload.get("ok") is not True:
    raise SystemExit("recovery receipt is not ok")
if payload.get("status") not in {"PASS", "PARTIAL"}:
    raise SystemExit("unexpected recovery status")
if payload.get("runtime_evidence_included") is not False:
    raise SystemExit("recovery mixed runtime evidence into the build index")
print(payload["status"])
PY
)" || die 2 "Build-Evidence-Receipt ist ungültig: $RECOVERY_RESULT"

cd -- "$TOOL"
unset PYTHONOPTIMIZE
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$TOOL${PYTHONPATH:+:$PYTHONPATH}"

GATE_LOG="$GATE_OUTPUT/gate_a_workflow.log"
set +e
"$TOOL_PYTHON" -B "$WORKFLOW" \
  --profile "$PROFILE_PATH" \
  --out "$GATE_OUTPUT" \
  --profile-driven \
  --require-run-mode standard \
  --execution-mode generate_benchmarksets \
  --models-root "$MODELS_ROOT" 2>&1 | tee "$GATE_LOG"
pipeline_status=("${PIPESTATUS[@]}")
set -e
workflow_rc=${pipeline_status[0]}
tee_rc=${pipeline_status[1]}
(( tee_rc == 0 )) || die 74 "Gate-A-Log konnte nicht geschrieben werden: $GATE_LOG"

printf 'BUILD_EVIDENCE_MODE=%s\n' "$RECOVERY_MODE"
printf 'BUILD_EVIDENCE_STATUS=%s\n' "$RECOVERY_STATUS"
printf 'BUILD_EVIDENCE_INDEX=%s\n' "$EVIDENCE_INDEX"
printf 'PROFILE=%s\n' "$PROFILE_PATH"
printf 'GATE_A_LOG=%s\n' "$GATE_LOG"

# The generic workflow intentionally returns RC=1 for its planned
# stop_after=build_backend_artifacts result.  The dedicated verifier re-opens
# the immutable output and admits that RC only when the run manifest, every
# stage checkpoint and both receipt-bound Part-1 HEFs prove the exact Gate-A
# success contract.  It performs no writes to GATE_OUTPUT.
set +e
"$TOOL_PYTHON" -B "$GATE_VERIFIER" \
  --gate-output "$GATE_OUTPUT" \
  --workflow-rc "$workflow_rc" \
  --expected-profile "$PROFILE_PATH" \
  --format shell
result_rc=$?
set -e
exit "$result_rc"
