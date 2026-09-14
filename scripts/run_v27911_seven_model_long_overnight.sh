#!/usr/bin/env bash
# Verify the retained backend-bound YOLO11 R8B result and start v2.79.11.

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  printf '%s\n' \
    'SKIP: run_v27911_seven_model_long_overnight.sh was sourced.' \
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

usage() {
  printf '%s\n' \
    'Usage:' \
    '  bash scripts/run_v27911_seven_model_long_overnight.sh [options]' \
    '' \
    'Options:' \
    '  --yolo11-r8b-output PATH Retained v2.79.10 YOLO11 R8B output (required).' \
    '  --yolov7-claim-output PATH Retained YOLOv7 b066 claim_gate_32 evidence (required).' \
    '  --out-root PATH          Parent for the new detached long run.' \
    '  --models-root PATH       Directory containing the seven ONNX files.' \
    '  --hardware-setups PATH   Current central hardware registry to freeze.' \
    '  --preflight-only         Verify and freeze inputs, but do not start.' \
    '  --help                   Show this help.' \
    '' \
    'The YOLO11 receipt must bind six successful paths to their exact backend,' \
    'runner, artifact bytes and completion contract. The launcher reruns neither' \
    'hardware gate, never resumes an earlier' \
    'evaluation, and never writes below either supplied evidence directory.'
}

SCRIPT_FILE="${BASH_SOURCE[0]}"
[[ -f "$SCRIPT_FILE" && ! -L "$SCRIPT_FILE" ]] ||
  die 66 "Launcher fehlt oder ist ein Symlink: $SCRIPT_FILE"
SCRIPT_DIR="$(cd -- "$(dirname -- "$SCRIPT_FILE")" && pwd -P)"
DEFAULT_TOOL="$(dirname -- "$SCRIPT_DIR")"

TOOL="${TOOL:-$DEFAULT_TOOL}"
TOOL_PYTHON_REQUEST="${TOOL_PYTHON:-}"
YOLO11_R8B_OUTPUT=""
YOLO11_R8B_OUTPUT_SEEN=0
YOLOV7_CLAIM_OUTPUT=""
YOLOV7_CLAIM_OUTPUT_SEEN=0
OUT_ROOT=/home/kmika/Models/EvaluationRuns
MODELS_ROOT=/home/kmika/Models
HARDWARE_SETUPS=/home/kmika/.onnx_splitpoint_tool/hardware_setups.yaml
PROFILE_EXPECTED_SHA256=15ede1b3462d2a67821f15217f8f8f623d9e86f25ba77b5aa50699076630fd8b
PREFLIGHT_ONLY=0

while (( $# > 0 )); do
  case "$1" in
    --yolo11-r8b-output)
      (( $# >= 2 )) || die 64 '--yolo11-r8b-output requires a path'
      (( YOLO11_R8B_OUTPUT_SEEN == 0 )) ||
        die 64 '--yolo11-r8b-output may be supplied only once'
      YOLO11_R8B_OUTPUT="$2"
      YOLO11_R8B_OUTPUT_SEEN=1
      shift 2
      ;;
    --yolov7-claim-output)
      (( $# >= 2 )) || die 64 '--yolov7-claim-output requires a path'
      (( YOLOV7_CLAIM_OUTPUT_SEEN == 0 )) ||
        die 64 '--yolov7-claim-output may be supplied only once'
      YOLOV7_CLAIM_OUTPUT="$2"
      YOLOV7_CLAIM_OUTPUT_SEEN=1
      shift 2
      ;;
    --out-root)
      (( $# >= 2 )) || die 64 '--out-root requires a path'
      OUT_ROOT="$2"
      shift 2
      ;;
    --models-root)
      (( $# >= 2 )) || die 64 '--models-root requires a path'
      MODELS_ROOT="$2"
      shift 2
      ;;
    --hardware-setups)
      (( $# >= 2 )) || die 64 '--hardware-setups requires a path'
      HARDWARE_SETUPS="$2"
      shift 2
      ;;
    --preflight-only)
      PREFLIGHT_ONLY=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      usage >&2
      die 64 "unsupported option: $1"
      ;;
  esac
done

(( YOLO11_R8B_OUTPUT_SEEN == 1 )) ||
  die 64 '--yolo11-r8b-output is required and has no default'
(( YOLOV7_CLAIM_OUTPUT_SEEN == 1 )) ||
  die 64 '--yolov7-claim-output is required and has no default'

require_absolute() {
  local label="$1"
  local value="$2"
  [[ "$value" == /* ]] || die 64 "$label muss absolut sein: $value"
  [[ "$(realpath -m -- "$value")" == "$value" ]] ||
    die 64 "$label ist nicht kanonisch: $value"
}

for pair in \
  "TOOL:$TOOL" \
  "YOLO11_R8B_OUTPUT:$YOLO11_R8B_OUTPUT" \
  "YOLOV7_CLAIM_OUTPUT:$YOLOV7_CLAIM_OUTPUT" \
  "OUT_ROOT:$OUT_ROOT" \
  "MODELS_ROOT:$MODELS_ROOT" \
  "HARDWARE_SETUPS:$HARDWARE_SETUPS"
do
  require_absolute "${pair%%:*}" "${pair#*:}"
done

# Hold the shared side of the canonical updater/workflow interlock from launch
# admission until this process exits.  FD 7 is inherited by the detached
# worker, which keeps the same open file description (and therefore the same
# shared flock) for the complete workflow lifetime.  An updater already
# holding the exclusive side makes the launcher fail before it reads installed
# release inputs or creates campaign output.
command -v -- flock >/dev/null || die 69 "Erforderliches Kommando fehlt: flock"
NORMALIZED_USER_HOME="$(readlink -f -- "$HOME")"
[[ -d "$NORMALIZED_USER_HOME" ]] ||
  die 73 "Kanonisches Home-Verzeichnis fehlt: $NORMALIZED_USER_HOME"
PLATFORM_LOCK_PARENT="$NORMALIZED_USER_HOME/.onnx_splitpoint_tool/locks"
mkdir -p -- "$PLATFORM_LOCK_PARENT" ||
  die 73 "Plattform-Lockverzeichnis konnte nicht erstellt werden"
[[ "$(readlink -f -- "$PLATFORM_LOCK_PARENT")" == "$PLATFORM_LOCK_PARENT" ]] ||
  die 67 "Plattform-Lockverzeichnis ist nicht kanonisch"
PLATFORM_INTERLOCK="$PLATFORM_LOCK_PARENT/workflow_platform_interlock.lock"
[[ ! -L "$PLATFORM_INTERLOCK" ]] ||
  die 67 "Workflow/Plattform-Interlock ist ein Symlink"
[[ ! -e "$PLATFORM_INTERLOCK" || -f "$PLATFORM_INTERLOCK" ]] ||
  die 67 "Workflow/Plattform-Interlock ist keine regulaere Datei"
exec 7<>"$PLATFORM_INTERLOCK" ||
  die 73 "Workflow/Plattform-Interlock kann nicht geoeffnet werden"
OPENED_PLATFORM_INTERLOCK="$(readlink -f -- "/proc/self/fd/7")" ||
  die 67 "Geoeffneter Workflow/Plattform-Interlock kann nicht validiert werden"
[[ "$OPENED_PLATFORM_INTERLOCK" == "$PLATFORM_INTERLOCK" && \
   -f "$PLATFORM_INTERLOCK" && ! -L "$PLATFORM_INTERLOCK" ]] ||
  die 67 "Workflow/Plattform-Interlock ist nicht die kanonische regulaere Datei"
flock -n -s 7 ||
  die 75 "Quellupdate haelt den exklusiven Workflow/Plattform-Interlock"

[[ -d "$TOOL" && ! -L "$TOOL" ]] ||
  die 66 "Tool-Verzeichnis fehlt oder ist ein Symlink: $TOOL"
TOOL="$(realpath -e -- "$TOOL")"
TOOL_PYTHON="${TOOL_PYTHON_REQUEST:-$TOOL/.venv/bin/python}"
[[ "$TOOL_PYTHON" == /* ]] || die 64 "TOOL_PYTHON muss absolut sein: $TOOL_PYTHON"
[[ -x "$TOOL_PYTHON" ]] || die 69 "Tool-Python fehlt: $TOOL_PYTHON"
# A Python venv normally exposes bin/python as a symlink.  Validate the
# interpreter semantically instead of rejecting that standard layout.
"$TOOL_PYTHON" -B - "$TOOL/.venv" <<'PY'
import os
import sys

expected = os.path.realpath(sys.argv[1])
observed = os.path.realpath(sys.prefix)
if observed != expected:
    raise SystemExit(
        f"TOOL_PYTHON is outside the Tool venv: prefix={observed!r} expected={expected!r}"
    )
print("V279_TOOL_PYTHON_VENV=PASS")
PY

PROFILE="$TOOL/profiles/complete_set_7models_v27911_b500_audit20.yaml"
require_absolute PROFILE "$PROFILE"
[[ -f "$PROFILE" && ! -L "$PROFILE" ]] ||
  die 66 "Eingefrorenes Langprofil fehlt oder ist unsicher: $PROFILE"
[[ -d "$YOLO11_R8B_OUTPUT" && ! -L "$YOLO11_R8B_OUTPUT" ]] ||
  die 66 "Retained R8B-Ausgabe fehlt oder ist unsicher: $YOLO11_R8B_OUTPUT"
R8B_SOURCE_RECEIPT="$YOLO11_R8B_OUTPUT/yolo11_r8b_verification.json"
[[ -f "$R8B_SOURCE_RECEIPT" && ! -L "$R8B_SOURCE_RECEIPT" ]] ||
  die 66 "Retained R8B-Verifikationsreceipt fehlt oder ist unsicher: $R8B_SOURCE_RECEIPT"
[[ -d "$YOLOV7_CLAIM_OUTPUT" && ! -L "$YOLOV7_CLAIM_OUTPUT" ]] ||
  die 66 "Retained YOLOv7-Claim-Evidenz fehlt oder ist unsicher: $YOLOV7_CLAIM_OUTPUT"
YOLOV7_CLAIM_RESULT="$YOLOV7_CLAIM_OUTPUT/native_three_stage_result.json"
YOLOV7_CLAIM_SOURCE_RECEIPT="$YOLOV7_CLAIM_OUTPUT/yolov7_claim_gate_32_verification.json"
for retained_claim_file in "$YOLOV7_CLAIM_RESULT" "$YOLOV7_CLAIM_SOURCE_RECEIPT"; do
  [[ -f "$retained_claim_file" && ! -L "$retained_claim_file" ]] ||
    die 66 "Retained YOLOv7-Claim-Datei fehlt oder ist unsicher: $retained_claim_file"
done
[[ -d "$MODELS_ROOT" && ! -L "$MODELS_ROOT" ]] ||
  die 66 "Modellwurzel fehlt oder ist unsicher: $MODELS_ROOT"
[[ -f "$HARDWARE_SETUPS" && ! -L "$HARDWARE_SETUPS" ]] ||
  die 66 "Hardware-Registry fehlt oder ist unsicher: $HARDWARE_SETUPS"

GATE_VERIFIER="$TOOL/scripts/verify_v27910_yolo11_r8b_gate.py"
GATE_PROFILE="$TOOL/profiles/yolo11l_v27910_r8b_full_b067_gate.yaml"
RECOVERY_HELPER="$TOOL/scripts/prepare_v27910_yolo11_r8b_recovery.py"
YOLOV7_CLAIM_VERIFIER="$TOOL/scripts/verify_v27911_yolov7_claim_gate_32.py"
SOURCE_MANIFEST="$TOOL/SOURCE_MANIFEST.json"
ACCEPTANCE="$TOOL/scripts/run_v27911_small_acceptance.sh"
WORKFLOW_LAUNCHER="$TOOL/scripts/run_fresh_standard_workflow.sh"
STATUS_HELPER="$TOOL/scripts/launcher_status_v27911.py"
for required in \
  "$GATE_VERIFIER" \
  "$GATE_PROFILE" \
  "$RECOVERY_HELPER" \
  "$YOLOV7_CLAIM_VERIFIER" \
  "$SOURCE_MANIFEST" \
  "$ACCEPTANCE" \
  "$WORKFLOW_LAUNCHER" \
  "$STATUS_HELPER"
do
  [[ -f "$required" && ! -L "$required" ]] ||
    die 66 "Erforderliche Release-Datei fehlt oder ist ein Symlink: $required"
done

for command_name in \
  awk chmod cmp cp date flock kill mv nohup realpath setsid sha256sum sleep tail
do
  command -v -- "$command_name" >/dev/null ||
    die 69 "Erforderliches Kommando fehlt: $command_name"
done

PROFILE_OBSERVED_SHA256="$(sha256sum "$PROFILE" | awk '{print $1}')"
[[ "$PROFILE_OBSERVED_SHA256" == "$PROFILE_EXPECTED_SHA256" ]] ||
  die 65 "v2.79.11-Langprofil stimmt nicht mit der Release-Pin überein"
SOURCE_MANIFEST_SHA256="$(sha256sum "$SOURCE_MANIFEST" | awk '{print $1}')"

# This check intentionally precedes every mutable launch artifact.  The release
# acceptance subsequently verifies the complete installed source manifest.
"$TOOL_PYTHON" -B - <<'PY'
import importlib.metadata
import onnx_splitpoint_tool

expected = "2.79.11"
package = str(getattr(onnx_splitpoint_tool, "__version__", ""))
distribution = importlib.metadata.version("onnx-splitpoint-tool")
if package != expected or distribution != expected:
    raise SystemExit(
        f"v2.79.11 required: package={package!r} distribution={distribution!r}"
    )
print("V27911_RELEASE_IDENTITY=PASS")
PY

cd -- "$TOOL"
bash "$ACCEPTANCE"

# Create absolute directories component-by-component without following links.
secure_directory() {
  "$TOOL_PYTHON" -B - "$1" <<'PY'
import os
import stat
import sys

path = sys.argv[1]
if not os.path.isabs(path) or os.path.normpath(path) != path:
    raise SystemExit(f"unsafe directory: {path}")
parts = path.split("/")[1:]
if any(part in {"", ".", ".."} for part in parts):
    raise SystemExit(f"unsafe directory component: {path}")
flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
parent = os.open("/", flags)
try:
    for part in parts:
        try:
            child = os.open(part, flags, dir_fd=parent)
        except FileNotFoundError:
            os.mkdir(part, 0o700, dir_fd=parent)
            child = os.open(part, flags, dir_fd=parent)
        info = os.fstat(child)
        if not stat.S_ISDIR(info.st_mode):
            os.close(child)
            raise SystemExit(f"not a directory: {path}")
        os.close(parent)
        parent = child
finally:
    os.close(parent)
print(path)
PY
}

OUT_ROOT="$(secure_directory "$OUT_ROOT")" ||
  die 73 "Ausgabewurzel konnte nicht sicher angelegt werden"

FROZEN_PARENT="$NORMALIZED_USER_HOME/.onnx_splitpoint_tool/frozen"
LOCK_PARENT="$PLATFORM_LOCK_PARENT"
FROZEN_PARENT="$(secure_directory "$FROZEN_PARENT")" ||
  die 73 "Freeze-Verzeichnis konnte nicht sicher angelegt werden"
LOCK_PARENT="$(secure_directory "$LOCK_PARENT")" ||
  die 73 "Lock-Verzeichnis konnte nicht sicher angelegt werden"
# The frozen profile itself pins this historical registry path.  Bind the
# supplied current registry byte-for-byte to that exact path instead of
# inventing a release-local path that the profile would never consume.
FROZEN_HARDWARE="$FROZEN_PARENT/v2784_7model_hardware_setups.yaml"
GLOBAL_LOCK="$LOCK_PARENT/v27911_7model_b500_audit20.lock"
LEGACY_GLOBAL_LOCK="$LOCK_PARENT/v2792_7model_b500_audit20.lock"
[[ ! -L "$FROZEN_HARDWARE" ]] ||
  die 67 "Eingefrorene Hardware-Registry ist ein Symlink"
[[ ! -L "$GLOBAL_LOCK" ]] || die 67 "Langlauf-Lock ist ein Symlink"
[[ ! -L "$LEGACY_GLOBAL_LOCK" ]] ||
  die 67 "Kompatibilitäts-Langlauf-Lock ist ein Symlink"

if [[ -e "$FROZEN_HARDWARE" ]]; then
  [[ -f "$FROZEN_HARDWARE" && ! -L "$FROZEN_HARDWARE" ]] ||
    die 67 "Freeze-Ziel ist keine reguläre Datei"
  cmp --silent -- "$HARDWARE_SETUPS" "$FROZEN_HARDWARE" ||
    die 78 "Profilgebundener v2.78.4-Hardware-Freeze weicht von der Registry ab"
else
  cp --reflink=auto --preserve=mode,timestamps -- \
    "$HARDWARE_SETUPS" "$FROZEN_HARDWARE"
  chmod 0600 -- "$FROZEN_HARDWARE"
fi

STAMP="$(date -u +%Y%m%d_%H%M%S_%N)"
CONTROL="$OUT_ROOT/v27911_7model_b500_audit20_launch_${STAMP}_$$"
[[ ! -e "$CONTROL" && ! -L "$CONTROL" ]] ||
  die 73 "Einmaliges Launch-Ziel existiert bereits: $CONTROL"
CONTROL="$(secure_directory "$CONTROL")" ||
  die 73 "Launch-Verzeichnis konnte nicht sicher angelegt werden"

# Execute from a private byte-identical profile copy.  A later source update
# cannot silently change the campaign after this admission has completed.
PROFILE_SOURCE="$PROFILE"
PROFILE="$CONTROL/profile.yaml"
cp --reflink=auto --preserve=mode,timestamps -- "$PROFILE_SOURCE" "$PROFILE"
chmod 0600 -- "$PROFILE"
cmp --silent -- "$PROFILE_SOURCE" "$PROFILE" ||
  die 74 "Eingefrorene Profilkopie ist nicht bytegleich"

R8B_SOURCE_BINDING="$("$TOOL_PYTHON" -B - \
  "$R8B_SOURCE_RECEIPT" "$YOLO11_R8B_OUTPUT" <<'PY'
import hashlib
import json
import os
import stat
import sys
from pathlib import Path

path = sys.argv[1]
gate_output = Path(sys.argv[2]).resolve(strict=True)
fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
try:
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode) or info.st_size > 64 * 1024 * 1024:
        raise SystemExit("unsafe retained R8B verdict")
    with os.fdopen(fd, "r", encoding="utf-8") as stream:
        fd = -1
        value = json.load(stream)
finally:
    if fd >= 0:
        os.close(fd)

required = {
    "schema": "onnx-splitpoint/yolo11-r8b-terminal-gate/v2",
    "schema_version": 2,
    "version": "2.79.10",
    "build_id": "v2.79.10-urecs-platform-power-release-closure",
    "workflow_version": "v2.79.10-urecs-platform-power-release-closure",
    "status": "PASS",
    "ok": True,
    "valid_terminal": True,
    "scope_complete": True,
    "runtime_identity_contract": "backend_bound_native_full_v2",
    "required_path_count": 6,
    "fresh_native_full_count": 3,
    "fresh_composed_count": 2,
    "imported_terminal_count": 1,
    "success_count": 6,
    "blocked_count": 0,
    "quality_decision_used_for_terminal": False,
    "model_id": "yolo11l",
    "model_sha256": "f0fcdf56a4ac24d87ec30c627170492ccad9db80486ec5694df6de65c1b3d147",
}
if not isinstance(value, dict):
    raise SystemExit("retained R8B verdict is not an object")
for key, expected in required.items():
    if value.get(key) != expected:
        raise SystemExit(f"retained R8B verdict mismatch:{key}")
expected_paths = {
    "hailo8_full": ("native_full_hailo8", "orin_nx_hailo8_01", "hailo8", "full", "fresh_native_full"),
    "hailo10h_full": ("native_full_hailo10h", "orin_nx_hailo10_01", "hailo10", "full", "fresh_native_full"),
    "deepx_full": ("native_full_deepx", "orin_nx_deepx_m1_01", "deepx_m1_full", "full", "fresh_native_full"),
    "hailo8_b067_composed": ("hailo8_to_tensorrt", "orin_nx_hailo8_01", "hailo8_to_trt", "composed", "read_only_import"),
    "hailo10h_b067_composed": ("hailo10_to_tensorrt", "orin_nx_hailo10_01", "hailo10_to_tensorrt", "composed", "fresh_composed"),
    "deepx_b067_composed": ("deepx_m1_to_tensorrt", "orin_nx_deepx_m1_01", "deepx_m1_to_tensorrt", "composed", "fresh_composed"),
}
paths = value.get("paths")
if not isinstance(paths, dict) or set(paths) != expected_paths:
    raise SystemExit("retained R8B terminal scope mismatch")
for key, expected_identity in expected_paths.items():
    row = paths.get(key)
    if not isinstance(row, dict) or row.get("terminal_status") != "success":
        raise SystemExit(f"retained R8B terminal path invalid:{key}")
    observed_identity = tuple(
        row.get(field)
        for field in ("backend", "setup_id", "run_id", "variant", "evidence_mode")
    )
    if observed_identity != expected_identity:
        raise SystemExit(f"retained R8B runtime identity mismatch:{key}")
    attestation = row.get("runtime_attestation")
    if not isinstance(attestation, dict) or attestation.get("status") != "passed":
        raise SystemExit(f"retained R8B runtime attestation missing:{key}")
    for field in ("provider_bound", "hardware_bound", "variant_bound", "current_artifact_bytes_bound"):
        if attestation.get(field) is not True:
            raise SystemExit(f"retained R8B runtime attestation mismatch:{key}:{field}")
    if row.get("variant") == "full":
        for field in ("makespan_bound", "completion_bound", "endpoint_bound"):
            if attestation.get(field) is not True:
                raise SystemExit(f"retained R8B native-full attestation mismatch:{key}:{field}")
workflow_rc = value.get("workflow_rc")
if isinstance(workflow_rc, bool) or not isinstance(workflow_rc, int):
    raise SystemExit("retained R8B workflow_rc is not an integer")
recovery = value.get("recovery_manifest")
if not isinstance(recovery, dict):
    raise SystemExit("retained R8B recovery manifest identity is missing")
recovery_text = str(recovery.get("path") or "")
if (
    not os.path.isabs(recovery_text)
    or os.path.normpath(recovery_text) != recovery_text
    or "\t" in recovery_text
    or "\n" in recovery_text
):
    raise SystemExit("retained R8B recovery manifest path is unsafe")
recovery_path = Path(recovery_text)
resolved_recovery = recovery_path.resolve(strict=True)
if resolved_recovery != recovery_path:
    raise SystemExit("retained R8B recovery manifest path is not canonical")
if recovery_path.parent != gate_output and gate_output not in recovery_path.parents:
    raise SystemExit("retained R8B recovery manifest is outside gate output")
recovery_info = recovery_path.stat()
if not stat.S_ISREG(recovery_info.st_mode) or recovery_path.is_symlink():
    raise SystemExit("retained R8B recovery manifest is not a safe regular file")
digest = hashlib.sha256()
with recovery_path.open("rb") as handle:
    for block in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(block)
if digest.hexdigest() != str(recovery.get("sha256") or "").lower():
    raise SystemExit("retained R8B recovery manifest SHA-256 mismatch")
if recovery_info.st_size != recovery.get("size_bytes"):
    raise SystemExit("retained R8B recovery manifest size mismatch")
print(f"{workflow_rc}\t{recovery_path}")
PY
)" || die 65 "Retained R8B-Verifikationsreceipt ist nicht launchfähig"
IFS=$'\t' read -r R8B_WORKFLOW_RC R8B_RECOVERY_MANIFEST <<< "$R8B_SOURCE_BINDING"
[[ "$R8B_WORKFLOW_RC" =~ ^[0-9]+$ && -n "$R8B_RECOVERY_MANIFEST" ]] ||
  die 65 "Retained R8B-Workflow/Recovery-Bindung ist unvollständig"

GATE_RECEIPT="$CONTROL/yolo11_r8b_verification.json"
GATE_TMP="$CONTROL/.yolo11_r8b_verification.tmp"
set +e
"$TOOL_PYTHON" -B "$GATE_VERIFIER" \
  --gate-output "$YOLO11_R8B_OUTPUT" \
  --workflow-rc "$R8B_WORKFLOW_RC" \
  --recovery-manifest "$R8B_RECOVERY_MANIFEST" \
  --expected-profile "$GATE_PROFILE" > "$GATE_TMP"
gate_rc=$?
set -e
mv -- "$GATE_TMP" "$GATE_RECEIPT"
(( gate_rc == 0 )) ||
  die 2 "Retained R8B-Ausgabe ist nicht launchfähig (Verifier rc=$gate_rc; $GATE_RECEIPT)"

"$TOOL_PYTHON" -B - "$GATE_RECEIPT" "$YOLO11_R8B_OUTPUT" <<'PY'
import json
import os
import stat
import sys
from pathlib import Path

path = sys.argv[1]
gate_output = Path(sys.argv[2]).resolve(strict=True)
fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
try:
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode) or info.st_size > 64 * 1024 * 1024:
        raise SystemExit("unsafe R8B verdict")
    with os.fdopen(fd, "r", encoding="utf-8") as stream:
        fd = -1
        value = json.load(stream)
finally:
    if fd >= 0:
        os.close(fd)
if not isinstance(value, dict):
    raise SystemExit("R8B verdict is not an object")
required = {
    "schema": "onnx-splitpoint/yolo11-r8b-terminal-gate/v2",
    "schema_version": 2,
    "version": "2.79.10",
    "build_id": "v2.79.10-urecs-platform-power-release-closure",
    "workflow_version": "v2.79.10-urecs-platform-power-release-closure",
    "status": "PASS",
    "ok": True,
    "valid_terminal": True,
    "scope_complete": True,
    "runtime_identity_contract": "backend_bound_native_full_v2",
    "required_path_count": 6,
    "fresh_native_full_count": 3,
    "fresh_composed_count": 2,
    "imported_terminal_count": 1,
    "success_count": 6,
    "blocked_count": 0,
    "quality_decision_used_for_terminal": False,
    "profile_id": "yolo11l_v27910_r8b_full_b067_gate",
    "model_id": "yolo11l",
    "model_sha256": "f0fcdf56a4ac24d87ec30c627170492ccad9db80486ec5694df6de65c1b3d147",
}
for key, expected in required.items():
    if value.get(key) != expected:
        raise SystemExit(f"R8B verdict mismatch:{key}")
expected_paths = {
    "hailo8_full": ("native_full_hailo8", "orin_nx_hailo8_01", "hailo8", "full", "fresh_native_full"),
    "hailo10h_full": ("native_full_hailo10h", "orin_nx_hailo10_01", "hailo10", "full", "fresh_native_full"),
    "deepx_full": ("native_full_deepx", "orin_nx_deepx_m1_01", "deepx_m1_full", "full", "fresh_native_full"),
    "hailo8_b067_composed": ("hailo8_to_tensorrt", "orin_nx_hailo8_01", "hailo8_to_trt", "composed", "read_only_import"),
    "hailo10h_b067_composed": ("hailo10_to_tensorrt", "orin_nx_hailo10_01", "hailo10_to_tensorrt", "composed", "fresh_composed"),
    "deepx_b067_composed": ("deepx_m1_to_tensorrt", "orin_nx_deepx_m1_01", "deepx_m1_to_tensorrt", "composed", "fresh_composed"),
}
paths = value.get("paths")
if not isinstance(paths, dict) or set(paths) != expected_paths:
    raise SystemExit("R8B terminal scope mismatch")
for key, expected_identity in expected_paths.items():
    row = paths.get(key)
    if not isinstance(row, dict) or row.get("terminal_status") != "success":
        raise SystemExit(f"R8B terminal path invalid:{key}")
    observed_identity = tuple(
        row.get(field)
        for field in ("backend", "setup_id", "run_id", "variant", "evidence_mode")
    )
    if observed_identity != expected_identity:
        raise SystemExit(f"R8B runtime identity mismatch:{key}")
    attestation = row.get("runtime_attestation")
    if not isinstance(attestation, dict) or attestation.get("status") != "passed":
        raise SystemExit(f"R8B runtime attestation missing:{key}")
    for field in ("provider_bound", "hardware_bound", "variant_bound", "current_artifact_bytes_bound"):
        if attestation.get(field) is not True:
            raise SystemExit(f"R8B runtime attestation mismatch:{key}:{field}")
    if row.get("variant") == "full":
        for field in ("makespan_bound", "completion_bound", "endpoint_bound"):
            if attestation.get(field) is not True:
                raise SystemExit(f"R8B native-full attestation mismatch:{key}:{field}")
run_dir = Path(str(value.get("run_dir") or "")).resolve(strict=True)
if run_dir != gate_output and gate_output not in run_dir.parents:
    raise SystemExit("R8B run directory is outside the supplied gate output")
print("V27910_YOLO11_FULL_GATE=PASS")
PY

"$TOOL_PYTHON" -B - "$YOLOV7_CLAIM_SOURCE_RECEIPT" <<'PY'
import json
import os
import stat
import sys

path = sys.argv[1]
fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
try:
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode) or info.st_size > 64 * 1024 * 1024:
        raise SystemExit("unsafe retained YOLOv7 claim verdict")
    with os.fdopen(fd, "r", encoding="utf-8") as stream:
        fd = -1
        value = json.load(stream)
finally:
    if fd >= 0:
        os.close(fd)

required = {
    "schema": "onnx-splitpoint/yolov7-claim-gate-32/v1",
    "status": "PASS",
    "ok": True,
    "valid_claim": True,
    "model_id": "yolov7_paper",
    "case_id": "b066",
    "claim_gate": "claim_gate_32",
    "required_item_count": 32,
    "passed_item_count": 32,
    "claim_eligible": True,
}
if not isinstance(value, dict):
    raise SystemExit("retained YOLOv7 claim verdict is not an object")
for key, expected in required.items():
    if value.get(key) != expected:
        raise SystemExit(f"retained YOLOv7 claim verdict mismatch:{key}")
print("V27910_YOLOV7_RETAINED_CLAIM_RECEIPT=PASS")
PY

YOLOV7_CLAIM_RECEIPT="$CONTROL/yolov7_claim_gate_32_verification.json"
YOLOV7_CLAIM_TMP="$CONTROL/.yolov7_claim_gate_32_verification.tmp"
set +e
"$TOOL_PYTHON" -B "$YOLOV7_CLAIM_VERIFIER" \
  --result-json "$YOLOV7_CLAIM_RESULT" \
  --evidence-root "$YOLOV7_CLAIM_OUTPUT" \
  --source-root "$TOOL" \
  --expected-source-manifest-sha256 "$SOURCE_MANIFEST_SHA256" \
  --output "$YOLOV7_CLAIM_TMP"
yolov7_claim_rc=$?
set -e
if [[ -f "$YOLOV7_CLAIM_TMP" && ! -L "$YOLOV7_CLAIM_TMP" ]]; then
  mv -- "$YOLOV7_CLAIM_TMP" "$YOLOV7_CLAIM_RECEIPT"
fi
(( yolov7_claim_rc == 0 )) ||
  die 2 "Retained YOLOv7 claim_gate_32 ist nicht launchfähig (Verifier rc=$yolov7_claim_rc)"
[[ -f "$YOLOV7_CLAIM_RECEIPT" && ! -L "$YOLOV7_CLAIM_RECEIPT" ]] ||
  die 74 "YOLOv7-Claim-Verifier hat kein sicheres Receipt geschrieben"

"$TOOL_PYTHON" -B - "$YOLOV7_CLAIM_RECEIPT" <<'PY'
import json
import os
import stat
import sys

path = sys.argv[1]
fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
try:
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode) or info.st_size > 64 * 1024 * 1024:
        raise SystemExit("unsafe YOLOv7 claim verdict")
    with os.fdopen(fd, "r", encoding="utf-8") as stream:
        fd = -1
        value = json.load(stream)
finally:
    if fd >= 0:
        os.close(fd)

required = {
    "schema": "onnx-splitpoint/yolov7-claim-gate-32/v1",
    "status": "PASS",
    "ok": True,
    "valid_claim": True,
    "model_id": "yolov7_paper",
    "case_id": "b066",
    "claim_gate": "claim_gate_32",
    "required_item_count": 32,
    "passed_item_count": 32,
    "claim_eligible": True,
}
if not isinstance(value, dict):
    raise SystemExit("YOLOv7 claim verdict is not an object")
for key, expected in required.items():
    if value.get(key) != expected:
        raise SystemExit(f"YOLOv7 claim verdict mismatch:{key}")
print("V27911_YOLOV7_CLAIM_GATE_32=PASS")
PY

ADMISSION="$CONTROL/long_run_admission.json"
ADMISSION_TMP="$CONTROL/.long_run_admission.tmp"
"$TOOL_PYTHON" -B - \
  "$PROFILE" "$MODELS_ROOT" "$FROZEN_HARDWARE" \
  "$GATE_RECEIPT" "$YOLOV7_CLAIM_RECEIPT" "$CONTROL" > "$ADMISSION_TMP" <<'PY'
from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any, Mapping

import yaml

from onnx_splitpoint_tool.benchmark.evaluation_profiles import (
    load_evaluation_profile,
)
from onnx_splitpoint_tool.execution_plan import build_effective_execution_plan
from onnx_splitpoint_tool.workflow.hardware_matrix import (
    canon_accelerator,
    normalize_hardware_targets,
)

profile_path = Path(sys.argv[1])
models_root = Path(sys.argv[2]).resolve(strict=True)
hardware_path = Path(sys.argv[3])
gate_receipt_path = Path(sys.argv[4])
yolov7_claim_receipt_path = Path(sys.argv[5])
output_filesystem = Path(sys.argv[6])
expected_models = (
    "resnet50",
    "yolo26s",
    "yolov7_paper",
    "mobilenet_v3_large",
    "regnet_x_1_6gf",
    "yolo26m",
    "yolo11l",
)
expected_run_profiles = {
    "hailo8",
    "hailo8_to_trt",
    "hailo10",
    "hailo10_to_tensorrt",
    "deepx_m1_full",
    "deepx_m1_to_tensorrt",
}


def need(value: Any, reason: str) -> None:
    if not value:
        raise RuntimeError(reason)


def read_regular(path: Path, *, limit: int) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        info = os.fstat(fd)
        need(stat.S_ISREG(info.st_mode), f"not_regular:{path}")
        need(info.st_size <= limit, f"too_large:{path}")
        chunks = []
        while True:
            block = os.read(fd, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        return b"".join(chunks)
    finally:
        os.close(fd)


def file_identity(path: Path, *, limit: int = 8 * 1024 * 1024 * 1024) -> dict[str, Any]:
    digest = hashlib.sha256()
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        info = os.fstat(fd)
        need(stat.S_ISREG(info.st_mode), f"not_regular:{path}")
        need(info.st_size <= limit, f"too_large:{path}")
        while True:
            block = os.read(fd, 1024 * 1024)
            if not block:
                break
            digest.update(block)
    finally:
        os.close(fd)
    return {
        "path": str(path),
        "sha256": digest.hexdigest(),
        "size_bytes": info.st_size,
    }


profile_raw = read_regular(profile_path, limit=8 * 1024 * 1024)
source_profile = yaml.safe_load(profile_raw.decode("utf-8"))
need(isinstance(source_profile, Mapping), "profile_not_mapping")
loaded = load_evaluation_profile(str(profile_path), validate=True)
need(loaded is not None and not isinstance(loaded, tuple), "profile_load_failed")
profile = dict(loaded.raw_profile or {})
need(profile.get("name") == "complete_set_7models_v27911_b500_audit20", "profile_name")

workflow = dict(profile.get("workflow") or {})
need(workflow.get("execution_mode") == "generate_and_run", "execution_mode")
need(workflow.get("skip_runtime_benchmarks") is False, "runtime_disabled")
need(not workflow.get("stop_after"), "partial_stop_configured")
need(workflow.get("parallel_remote_setups") is True, "remote_parallelism_disabled")
need(workflow.get("max_parallel_setups") == 3, "remote_parallelism_scope")
need(workflow.get("max_parallel_uploads") == 1, "parallel_upload_scope")

selection = dict(profile.get("selection_policy") or {})
need(selection.get("selection_strategy") in {"deterministic_audit", "score_independent_audit"}, "selection_strategy")
need(selection.get("score_independent_audit_enabled") is True, "audit_disabled")
need(selection.get("audit_candidate_universe") == "deterministic_audit", "audit_universe")
need(selection.get("audit_size") == 20, "audit_size")
need(selection.get("minimum_valid_audit_candidates") == 10, "audit_minimum")
need(selection.get("audit_seed") == 20260710, "audit_seed")
need(selection.get("forced_cases") == {"yolo11l": ["b067"]}, "yolo11_anchor_binding")

suite = dict(profile.get("model_suite") or {})
rows = list(suite.get("primary") or [])
need(all(isinstance(row, Mapping) for row in rows), "model_row_type")
model_ids = tuple(str(row.get("id") or "") for row in rows)
need(model_ids == expected_models, f"model_scope:{model_ids}")
need(not list(suite.get("reserve") or []), "reserve_models_present")

models = {}
for row in rows:
    row = dict(row)
    model_id = str(row["id"])
    universe = dict(row.get("candidate_universe") or {})
    need(universe.get("mode") == "deterministic_audit", f"candidate_universe:{model_id}")
    path = Path(os.path.expanduser(str(row.get("onnx") or "")))
    need(path.is_absolute(), f"model_path_not_absolute:{model_id}")
    need(path.is_file() and not path.is_symlink(), f"unsafe_model:{model_id}")
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(models_root)
    except ValueError as exc:
        raise RuntimeError(f"model_outside_root:{model_id}") from exc
    identity = file_identity(path)
    declared = str(row.get("model_sha256") or "").lower()
    declared = declared[7:] if declared.startswith("sha256:") else declared
    if declared:
        need(identity["sha256"] == declared, f"model_hash:{model_id}")
    if model_id == "yolov7_paper":
        need(
            identity["sha256"]
            == "7a13e66f91047cce0e251c05f64159646847e842af31d60441c63dcdfad7825d",
            "yolov7_paper_identity",
        )
    models[model_id] = identity

run_profiles = {
    str(row.get("id") or "")
    for row in list(profile.get("run_profiles") or [])
    if isinstance(row, Mapping) and row.get("enabled", True) is not False
}
need(run_profiles == expected_run_profiles, f"run_profile_scope:{sorted(run_profiles)}")

validation = dict(profile.get("validation_execution") or {})
limits = dict(validation.get("max_items") or {})
need(limits.get("classification") == 500, "classification_b500")
need(limits.get("detection") == 500, "detection_b500")
need(dict(profile.get("hailo_build") or {}).get("calib_count") == 500, "hailo_b500")
need(dict(profile.get("deepx_build") or {}).get("calib_count") == 500, "deepx_b500")
hailo_build = dict(profile.get("hailo_build") or {})
need(hailo_build.get("cold_build_timeout_s") == 0, "hailo_cold_timeout_not_unlimited")
need(
    hailo_build.get("hard_timeout_disable_tokens")
    == [0, "off", "none", "unlimited", "disabled"],
    "hailo_disable_tokens",
)
need(hailo_build.get("immutable_attempt_receipts") is True, "hailo_receipts_mutable")
need(
    hailo_build.get("terminal_attempt_selection") == "last_attempt_even_on_failure",
    "hailo_terminal_attempt_selection",
)
need(dict(profile.get("native_producers") or {}).get("enabled") is False, "native_enabled")
energy = dict(profile.get("energy") or {})
need(energy.get("enabled") is False and energy.get("generic_enabled") is False, "energy_enabled")

preset = dict(profile.get("execution_preset") or {})
need(preset.get("id") == "standard", "run_mode")
need(preset.get("follow_tool_config") is False, "run_mode_not_frozen")
preset_hailo = dict(dict(dict(preset.get("snapshot") or {}).get("build") or {}).get("hailo") or {})
need(preset_hailo.get("cold_build_timeout_s") == 0, "preset_hailo_cold_timeout_not_unlimited")

scheduler = dict(profile.get("build_scheduler") or {})
need(scheduler.get("enabled") is True, "build_scheduler_disabled")
need(scheduler.get("max_workers") == 3, "build_scheduler_workers")
need(scheduler.get("cpu_tokens") == 8, "build_scheduler_cpu_tokens")
need(scheduler.get("ram_mb") == 0, "build_scheduler_ram_not_auto")
need(scheduler.get("ram_reserve_mb") == 2048, "build_scheduler_ram_reserve")
need(
    dict(scheduler.get("family_limits") or {})
    == {"hailo8": 1, "hailo10": 1, "deepx": 1},
    "build_scheduler_family_limits",
)

hardware = dict(profile.get("hardware") or {})
profile_hardware = Path(os.path.expanduser(str(hardware.get("setups_file") or "")))
need(profile_hardware.resolve(strict=True) == hardware_path.resolve(strict=True), "hardware_freeze_binding")
hardware_identity = file_identity(hardware_path, limit=16 * 1024 * 1024)
hardware_payload = yaml.safe_load(read_regular(hardware_path, limit=16 * 1024 * 1024).decode("utf-8"))
need(isinstance(hardware_payload, Mapping), "hardware_registry_not_mapping")
need(bool(list(hardware_payload.get("hardware_setups") or [])), "hardware_registry_empty")

expected_targets = {
    "orin_nx_hailo8_01": "hailo8",
    "orin_nx_hailo10_01": "hailo10",
    "orin_nx_deepx_m1_01": "deepx_m1",
}
targets = normalize_hardware_targets(profile)
need(
    {str(row.get("id") or "") for row in targets} == set(expected_targets),
    "hardware_target_scope",
)
hardware_targets = {}
for row in targets:
    setup_id = str(row.get("id") or "")
    expected_accelerator = expected_targets[setup_id]
    accelerator = canon_accelerator(row.get("accelerator"))
    if expected_accelerator == "hailo10":
        need(accelerator.startswith("hailo10"), f"hardware_accelerator:{setup_id}")
    else:
        need(accelerator == expected_accelerator, f"hardware_accelerator:{setup_id}")
    need(row.get("enabled") is True, f"hardware_disabled:{setup_id}")
    remote = dict(row.get("remote") or {})
    need(bool(str(remote.get("host") or "").strip()), f"hardware_host:{setup_id}")
    need(bool(str(remote.get("user") or "").strip()), f"hardware_user:{setup_id}")
    need(int(remote.get("port") or 0) > 0, f"hardware_port:{setup_id}")
    need(
        bool(str(row.get("build_environment_id") or "").strip()),
        f"build_environment:{setup_id}",
    )
    hardware_targets[setup_id] = {
        "accelerator": accelerator,
        "host": str(remote.get("host")),
        "user": str(remote.get("user")),
        "port": int(remote.get("port")),
        "build_environment_id": str(row.get("build_environment_id")),
    }

campaign = dict(profile.get("campaign") or {})
dataset_paths = []
registry = str(campaign.get("dataset_registry") or "")
need(bool(registry), "dataset_registry_missing")
dataset_paths.append(Path(os.path.expanduser(registry)))
manifests = dict(campaign.get("dataset_manifests") or {})
for task in ("classification", "detection"):
    task_rows = dict(manifests.get(task) or {})
    for role in ("calibration", "validation"):
        value = str(task_rows.get(role) or "")
        need(bool(value), f"dataset_manifest_missing:{task}:{role}")
        dataset_paths.append(Path(os.path.expanduser(value)))
datasets = {
    str(path): file_identity(path, limit=256 * 1024 * 1024)
    for path in dataset_paths
}

gate_verdict = json.loads(
    read_regular(gate_receipt_path, limit=8 * 1024 * 1024).decode("utf-8")
)
need(isinstance(gate_verdict, Mapping), "gate_verdict_not_mapping")
need(
    gate_verdict.get("schema")
    == "onnx-splitpoint/yolo11-r8b-terminal-gate/v2",
    "gate_verdict_schema",
)
need(gate_verdict.get("schema_version") == 2, "gate_verdict_schema_version")
need(gate_verdict.get("version") == "2.79.10", "gate_verdict_version")
need(
    gate_verdict.get("build_id")
    == "v2.79.10-urecs-platform-power-release-closure",
    "gate_verdict_build_id",
)
need(
    gate_verdict.get("workflow_version")
    == "v2.79.10-urecs-platform-power-release-closure",
    "gate_verdict_workflow_version",
)
need(
    gate_verdict.get("ok") is True
    and gate_verdict.get("status") == "PASS"
    and gate_verdict.get("valid_terminal") is True
    and gate_verdict.get("scope_complete") is True
    and gate_verdict.get("runtime_identity_contract") == "backend_bound_native_full_v2"
    and gate_verdict.get("required_path_count") == 6
    and gate_verdict.get("fresh_native_full_count") == 3
    and gate_verdict.get("fresh_composed_count") == 2
    and gate_verdict.get("imported_terminal_count") == 1
    and gate_verdict.get("success_count") == 6
    and gate_verdict.get("blocked_count") == 0
    and gate_verdict.get("quality_decision_used_for_terminal") is False,
    "gate_verdict_not_pass",
)
gate_model_sha = str(gate_verdict.get("model_sha256") or "").lower()
need(gate_model_sha == models["yolo11l"]["sha256"], "yolo11_gate_model_identity")
expected_gate_paths = {
    "hailo8_full": {
        "backend": "native_full_hailo8", "setup_id": "orin_nx_hailo8_01",
        "run_id": "hailo8", "variant": "full", "evidence_mode": "fresh_native_full",
    },
    "hailo10h_full": {
        "backend": "native_full_hailo10h", "setup_id": "orin_nx_hailo10_01",
        "run_id": "hailo10", "variant": "full", "evidence_mode": "fresh_native_full",
    },
    "deepx_full": {
        "backend": "native_full_deepx", "setup_id": "orin_nx_deepx_m1_01",
        "run_id": "deepx_m1_full", "variant": "full", "evidence_mode": "fresh_native_full",
    },
    "hailo8_b067_composed": {
        "backend": "hailo8_to_tensorrt", "setup_id": "orin_nx_hailo8_01",
        "run_id": "hailo8_to_trt", "variant": "composed", "evidence_mode": "read_only_import",
    },
    "hailo10h_b067_composed": {
        "backend": "hailo10_to_tensorrt", "setup_id": "orin_nx_hailo10_01",
        "run_id": "hailo10_to_tensorrt", "variant": "composed", "evidence_mode": "fresh_composed",
    },
    "deepx_b067_composed": {
        "backend": "deepx_m1_to_tensorrt", "setup_id": "orin_nx_deepx_m1_01",
        "run_id": "deepx_m1_to_tensorrt", "variant": "composed", "evidence_mode": "fresh_composed",
    },
}
gate_paths = gate_verdict.get("paths")
need(isinstance(gate_paths, Mapping), "gate_terminal_paths_not_mapping")
need(set(gate_paths) == expected_gate_paths, "gate_terminal_scope")
gate_terminal_summary = {}
for key in sorted(expected_gate_paths):
    row = gate_paths.get(key)
    need(isinstance(row, Mapping), f"gate_terminal_row:{key}")
    terminal_status = str(row.get("terminal_status") or "")
    need(terminal_status == "success", f"gate_terminal_status:{key}")
    expected_identity = expected_gate_paths[key]
    need(
        all(row.get(field) == value for field, value in expected_identity.items()),
        f"gate_runtime_identity:{key}",
    )
    attestation = row.get("runtime_attestation")
    need(isinstance(attestation, Mapping), f"gate_runtime_attestation:{key}")
    need(attestation.get("status") == "passed", f"gate_attestation_status:{key}")
    common_attestations = (
        "provider_bound", "hardware_bound", "variant_bound",
        "current_artifact_bytes_bound",
    )
    need(
        all(attestation.get(field) is True for field in common_attestations),
        f"gate_attestation_common:{key}",
    )
    if expected_identity["variant"] == "full":
        need(
            all(
                attestation.get(field) is True
                for field in ("makespan_bound", "completion_bound", "endpoint_bound")
            ),
            f"gate_attestation_native_full:{key}",
        )
    gate_terminal_summary[key] = {
        "terminal_status": terminal_status,
        "technical_classification": str(row.get("technical_classification") or ""),
        **expected_identity,
        "runtime_attestation": dict(attestation),
    }
need(
    int(gate_verdict.get("success_count") or 0) == 6
    and int(gate_verdict.get("blocked_count") or 0) == 0,
    "gate_terminal_count",
)
gate_receipt_identity = file_identity(gate_receipt_path, limit=64 * 1024 * 1024)

yolov7_claim_verdict = json.loads(
    read_regular(
        yolov7_claim_receipt_path,
        limit=64 * 1024 * 1024,
    ).decode("utf-8")
)
need(isinstance(yolov7_claim_verdict, Mapping), "yolov7_claim_verdict_not_mapping")
required_yolov7_claim = {
    "schema": "onnx-splitpoint/yolov7-claim-gate-32/v1",
    "status": "PASS",
    "ok": True,
    "valid_claim": True,
    "model_id": "yolov7_paper",
    "case_id": "b066",
    "claim_gate": "claim_gate_32",
    "required_item_count": 32,
    "passed_item_count": 32,
    "claim_eligible": True,
}
for key, expected in required_yolov7_claim.items():
    need(
        yolov7_claim_verdict.get(key) == expected,
        f"yolov7_claim_verdict:{key}",
    )
yolov7_claim_receipt_identity = file_identity(
    yolov7_claim_receipt_path,
    limit=64 * 1024 * 1024,
)

plan = build_effective_execution_plan(profile)
need(plan.get("run_mode") == "standard", "effective_run_mode")
need(plan.get("models") == list(expected_models), "effective_model_scope")
need(plan.get("model_count") == 7, "effective_model_count")
need(plan.get("generic_runtime_enabled") is True, "effective_runtime_disabled")
need(plan.get("compiler_dispatch_allowed") is True, "compiler_dispatch_blocked")
need(plan.get("score_independent_audit_enabled") is True, "effective_audit_disabled")
need(plan.get("score_independent_audit_size") == 20, "effective_audit_size")
need(plan.get("score_independent_audit_minimum_valid") == 10, "effective_audit_minimum")
need(plan.get("native_enabled") is False, "effective_native_enabled")
need(plan.get("native_energy_requested") is False, "effective_energy_enabled")
need(
    set(dict(plan.get("candidate_counts_by_model") or {})) == set(expected_models),
    "effective_candidate_scope",
)
need(
    all(value == 20 for value in dict(plan.get("candidate_counts_by_model") or {}).values()),
    "effective_candidate_counts",
)
need(int(plan.get("expected_generic_result_rows_min_total") or 0) > 0, "effective_rows_empty")

statvfs = os.statvfs(str(output_filesystem))
free_bytes = int(statvfs.f_bavail) * int(statvfs.f_frsize)
need(free_bytes > 0, "capacity_unavailable")

receipt = {
    "schema": "onnx-splitpoint/v27911-seven-model-long-run-admission/v1",
    "schema_version": 1,
    "status": "PASS",
    "ok": True,
    "profile": {
        "path": str(profile_path),
        "sha256": hashlib.sha256(profile_raw).hexdigest(),
    },
    "model_ids": list(model_ids),
    "models": models,
    "hardware_registry": hardware_identity,
    "hardware_targets": hardware_targets,
    "datasets": datasets,
    "yolo11_anchor": "b067",
    "yolo11_r8b_gate": {
        "receipt": gate_receipt_identity,
        "version": "2.79.10",
        "build_id": "v2.79.10-urecs-platform-power-release-closure",
        "model_sha256": gate_model_sha,
        "runtime_identity_contract": "backend_bound_native_full_v2",
        "required_path_count": 6,
        "fresh_native_full_count": 3,
        "fresh_composed_count": 2,
        "imported_terminal_count": 1,
        "success_count": int(gate_verdict.get("success_count") or 0),
        "blocked_count": int(gate_verdict.get("blocked_count") or 0),
        "terminal_paths": gate_terminal_summary,
        "quality_decision_used_for_terminal": False,
    },
    "yolov7_claim_gate_32": {
        "receipt": yolov7_claim_receipt_identity,
        "model_id": "yolov7_paper",
        "case_id": "b066",
        "required_item_count": 32,
        "passed_item_count": 32,
        "claim_eligible": True,
    },
    "selection_contract": {
        "audit_size": 20,
        "minimum_valid": 10,
        "seed": 20260710,
        "execution_union_order_policy": "forced_deployment_then_audit",
    },
    "effective_plan": plan,
    "capacity_observation": {
        "filesystem": str(output_filesystem),
        "free_bytes_before_launch": free_bytes,
        "authoritative_required_bytes": None,
        "note": "The workflow/bundle admission remains authoritative; no invented fixed GiB threshold is applied.",
    },
    "yolo11_r8b_workflow_rerun": False,
    "yolov7_claim_gate_workflow_rerun": False,
    "native_enabled": False,
    "energy_enabled": False,
}
print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
PY
mv -- "$ADMISSION_TMP" "$ADMISSION"

"$TOOL_PYTHON" -B - "$ADMISSION" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as stream:
    value = json.load(stream)
if value.get("ok") is not True or value.get("status") != "PASS":
    raise SystemExit("long-run admission is not PASS")
if value.get("model_ids") != [
    "resnet50", "yolo26s", "yolov7_paper", "mobilenet_v3_large",
    "regnet_x_1_6gf", "yolo26m", "yolo11l",
]:
    raise SystemExit("long-run scope changed")
if value.get("yolo11_anchor") != "b067":
    raise SystemExit("YOLO11 anchor changed")
gate = value.get("yolo11_r8b_gate")
if not isinstance(gate, dict) or gate.get("required_path_count") != 6:
    raise SystemExit("YOLO11 R8B gate scope changed")
if gate.get("version") != "2.79.10":
    raise SystemExit("YOLO11 R8B gate release changed")
if (
    gate.get("build_id")
    != "v2.79.10-urecs-platform-power-release-closure"
):
    raise SystemExit("YOLO11 R8B gate build changed")
if gate.get("runtime_identity_contract") != "backend_bound_native_full_v2":
    raise SystemExit("YOLO11 R8B runtime identity contract changed")
for field, expected in (
    ("fresh_native_full_count", 3),
    ("fresh_composed_count", 2),
    ("imported_terminal_count", 1),
    ("success_count", 6),
    ("blocked_count", 0),
):
    if gate.get(field) != expected:
        raise SystemExit(f"YOLO11 R8B gate count changed:{field}")
paths = gate.get("terminal_paths")
expected_paths = {
    "hailo8_full", "hailo10h_full", "deepx_full",
    "hailo8_b067_composed", "hailo10h_b067_composed",
    "deepx_b067_composed",
}
if not isinstance(paths, dict) or set(paths) != expected_paths:
    raise SystemExit("YOLO11 R8B terminal path set changed")
if any(
    not isinstance(row, dict)
    or row.get("terminal_status") != "success"
    or not isinstance(row.get("runtime_attestation"), dict)
    or row["runtime_attestation"].get("status") != "passed"
    for row in paths.values()
):
    raise SystemExit("YOLO11 R8B terminal path is not admitted")
yolov7_claim = value.get("yolov7_claim_gate_32")
if not isinstance(yolov7_claim, dict):
    raise SystemExit("YOLOv7 claim_gate_32 admission is missing")
required_yolov7_claim = {
    "model_id": "yolov7_paper",
    "case_id": "b066",
    "required_item_count": 32,
    "passed_item_count": 32,
    "claim_eligible": True,
}
for key, expected in required_yolov7_claim.items():
    if yolov7_claim.get(key) != expected:
        raise SystemExit(f"YOLOv7 claim_gate_32 admission changed:{key}")
print("V27911_SEVEN_MODEL_LONG_ADMISSION=PASS")
PY

PROFILE_SHA256="$(sha256sum "$PROFILE" | awk '{print $1}')"
GATE_RECEIPT_SHA256="$(sha256sum "$GATE_RECEIPT" | awk '{print $1}')"
YOLOV7_CLAIM_RECEIPT_SHA256="$(sha256sum "$YOLOV7_CLAIM_RECEIPT" | awk '{print $1}')"
ADMISSION_SHA256="$(sha256sum "$ADMISSION" | awk '{print $1}')"
HARDWARE_SHA256="$(sha256sum "$FROZEN_HARDWARE" | awk '{print $1}')"

RUN_ROOT="$CONTROL/runs"
RUN_ROOT="$(secure_directory "$RUN_ROOT")" || die 73 "Run-Wurzel konnte nicht erstellt werden"
LOG="$CONTROL/workflow.log"
STATUS="$CONTROL/launcher_status.txt"
PID_FILE="$CONTROL/launcher.pid"

# Probe the global lock in the foreground.  The detached worker acquires it
# again and holds it for the complete workflow lifetime.
exec {probe_fd}>"$GLOBAL_LOCK"
if ! flock -n "$probe_fd"; then
  exec {probe_fd}>&-
  die 75 "Ein anderer v2.79.11-Sieben-Modell-Langlauf hält bereits den Lock"
fi
exec {legacy_probe_fd}>"$LEGACY_GLOBAL_LOCK"
if ! flock -n "$legacy_probe_fd"; then
  exec {legacy_probe_fd}>&-
  flock -u "$probe_fd"
  exec {probe_fd}>&-
  die 75 "Ein älterer v2.79.2-Sieben-Modell-Langlauf hält bereits den Kompatibilitäts-Lock"
fi
flock -u "$legacy_probe_fd"
exec {legacy_probe_fd}>&-
flock -u "$probe_fd"
exec {probe_fd}>&-

if (( PREFLIGHT_ONLY )); then
  printf '%s\n' \
    'V27911_SEVEN_MODEL_LONG_PREFLIGHT=PASS' \
    'V27910_YOLO11_FULL_GATE=PASS' \
    'V27911_YOLOV7_CLAIM_GATE_32=PASS' \
    'LONG_RUN_STARTED=NO' \
    'YOLO11_R8B_WORKFLOW_RERUN=NO' \
    'YOLOV7_CLAIM_GATE_WORKFLOW_RERUN=NO' \
    'YOLO11_ANCHOR_REUSED=b067'
  printf 'LAUNCH_CONTROL=%s\n' "$CONTROL"
  printf 'YOLO11_R8B_OUTPUT=%s\n' "$YOLO11_R8B_OUTPUT"
  printf 'YOLO11_R8B_VERDICT=%s\n' "$GATE_RECEIPT"
  printf 'YOLOV7_CLAIM_OUTPUT=%s\n' "$YOLOV7_CLAIM_OUTPUT"
  printf 'YOLOV7_CLAIM_VERDICT=%s\n' "$YOLOV7_CLAIM_RECEIPT"
  printf 'LONG_RUN_ADMISSION=%s\n' "$ADMISSION"
  exit 0
fi

"$TOOL_PYTHON" -B "$STATUS_HELPER" write \
  --status "$STATUS" \
  --state STARTING \
  --phase detached_worker_admission \
  --log "$LOG" \
  --run-root "$RUN_ROOT" \
  --detail "profile and immutable run scope admitted; detached worker starting"

nohup setsid --fork --wait bash -c '
  set -Eeuo pipefail
  umask 077
  status="$1"
  lock="$2"
  tool="$3"
  profile="$4"
  run_root="$5"
  profile_sha="$6"
  gate_receipt="$7"
  gate_sha="$8"
  yolov7_claim_receipt="$9"
  yolov7_claim_sha="${10}"
  admission="${11}"
  admission_sha="${12}"
  hardware="${13}"
  hardware_sha="${14}"
  log="${15}"
  status_helper="${16}"
  legacy_lock="${17}"

  # FD 7 is the shared canonical workflow/platform interlock inherited from
  # the foreground launcher.  Reassert the nonblocking shared lock so a lost
  # descriptor fails closed instead of allowing an updater/run race.
  if ! flock -n -s 7; then
    "$tool/.venv/bin/python" -B "$status_helper" write \
      --status "$status" --state LOCKED --phase platform_interlock_admission \
      --log "$log" --run-root "$run_root" --workflow-rc 75 \
      --detail "exclusive source updater interlock is active or inherited FD is missing"
    exit 75
  fi

  exec 9>"$lock"
  if ! flock -n 9; then
    "$tool/.venv/bin/python" -B "$status_helper" write \
      --status "$status" --state LOCKED --phase lock_admission \
      --log "$log" --run-root "$run_root" --workflow-rc 75 \
      --detail "global v2.79.11 campaign lock unavailable"
    exit 75
  fi
  exec 8>"$legacy_lock"
  if ! flock -n 8; then
    "$tool/.venv/bin/python" -B "$status_helper" write \
      --status "$status" --state LOCKED --phase lock_admission \
      --log "$log" --run-root "$run_root" --workflow-rc 75 \
      --detail "legacy v2.79.2 campaign compatibility lock unavailable"
    exit 75
  fi

  printf "%s  %s\n" "$profile_sha" "$profile" | sha256sum -c -
  printf "%s  %s\n" "$gate_sha" "$gate_receipt" | sha256sum -c -
  printf "%s  %s\n" "$yolov7_claim_sha" "$yolov7_claim_receipt" | sha256sum -c -
  printf "%s  %s\n" "$admission_sha" "$admission" | sha256sum -c -
  printf "%s  %s\n" "$hardware_sha" "$hardware" | sha256sum -c -

  workflow_pid=""
  status_monitor_pid=""
  finished=0
  on_exit() {
    local rc=$?
    if [[ -n "$status_monitor_pid" ]]; then
      kill "$status_monitor_pid" 2>/dev/null || true
      wait "$status_monitor_pid" 2>/dev/null || true
    fi
    if (( finished == 0 )); then
      "$tool/.venv/bin/python" -B "$status_helper" write \
        --status "$status" --state FAILED --phase detached_worker_exit \
        ${workflow_pid:+--worker-pid "$workflow_pid"} \
        --log "$log" --run-root "$run_root" --workflow-rc "$rc" \
        --detail "detached worker exited before normal finalization" || true
    fi
  }
  trap on_exit EXIT

  set +e
  PY="$tool/.venv/bin/python" \
    bash "$tool/scripts/run_fresh_standard_workflow.sh" \
      --profile "$profile" \
      --out "$run_root" &
  workflow_pid=$!
  set -e

  "$tool/.venv/bin/python" -B "$status_helper" monitor \
    --status "$status" \
    --log "$log" \
    --worker-pid "$workflow_pid" \
    --run-root "$run_root" \
    --interval-s 30 &
  status_monitor_pid=$!

  "$tool/.venv/bin/python" -B "$status_helper" write \
    --status "$status" --state RUNNING --phase workflow_start \
    --worker-pid "$workflow_pid" --monitor-pid "$status_monitor_pid" \
    --log "$log" --run-root "$run_root" \
    --detail "workflow process and periodic status monitor started"

  set +e
  wait "$workflow_pid"
  rc=$?
  set -e
  kill "$status_monitor_pid" 2>/dev/null || true
  wait "$status_monitor_pid" 2>/dev/null || true

  finished=1
  if (( rc == 0 )); then
    state=COMPLETED
  else
    state=FAILED
  fi
  "$tool/.venv/bin/python" -B "$status_helper" write \
    --status "$status" --state "$state" --phase workflow_finished \
    --worker-pid "$workflow_pid" --monitor-pid "$status_monitor_pid" \
    --log "$log" --run-root "$run_root" --workflow-rc "$rc" \
    --detail "workflow process exited with rc=$rc"
  exit "$rc"
' v27911-seven-model-worker \
  "$STATUS" "$GLOBAL_LOCK" "$TOOL" "$PROFILE" "$RUN_ROOT" \
  "$PROFILE_SHA256" "$GATE_RECEIPT" "$GATE_RECEIPT_SHA256" \
  "$YOLOV7_CLAIM_RECEIPT" "$YOLOV7_CLAIM_RECEIPT_SHA256" \
  "$ADMISSION" "$ADMISSION_SHA256" \
  "$FROZEN_HARDWARE" "$HARDWARE_SHA256" \
  "$LOG" "$STATUS_HELPER" "$LEGACY_GLOBAL_LOCK" \
  > "$LOG" 2>&1 < /dev/null &
DETACHED_PID=$!

PID_TMP="$CONTROL/.launcher.pid.tmp"
printf '%s\n' "$DETACHED_PID" > "$PID_TMP"
mv -- "$PID_TMP" "$PID_FILE"

# Detect immediate lock/hash/command failures without waiting for the workflow.
sleep 2
if ! kill -0 "$DETACHED_PID" 2>/dev/null; then
  set +e
  wait "$DETACHED_PID"
  start_rc=$?
  set -e
  tail -n 80 -- "$LOG" >&2 || true
  die "$start_rc" "Abgekoppelter Langlauf ist sofort beendet"
fi

printf '%s\n' \
  'V27911_SEVEN_MODEL_LONG_PREFLIGHT=PASS' \
  'V27910_YOLO11_FULL_GATE=PASS' \
  'V27911_YOLOV7_CLAIM_GATE_32=PASS' \
  'V27911_SEVEN_MODEL_LONG_RUN=STARTED' \
  'DETACHED=YES' \
  'YOLO11_R8B_WORKFLOW_RERUN=NO' \
  'YOLOV7_CLAIM_GATE_WORKFLOW_RERUN=NO' \
  'YOLO11_ANCHOR_REUSED=b067' \
  'NATIVE_RUN=NO' \
  'ENERGY_RUN=NO'
printf 'LAUNCH_CONTROL=%s\n' "$CONTROL"
printf 'YOLO11_R8B_OUTPUT=%s\n' "$YOLO11_R8B_OUTPUT"
printf 'YOLO11_R8B_VERDICT=%s\n' "$GATE_RECEIPT"
printf 'YOLOV7_CLAIM_OUTPUT=%s\n' "$YOLOV7_CLAIM_OUTPUT"
printf 'YOLOV7_CLAIM_VERDICT=%s\n' "$YOLOV7_CLAIM_RECEIPT"
printf 'LONG_RUN_ROOT=%s\n' "$RUN_ROOT"
printf 'LONG_RUN_LOG=%s\n' "$LOG"
printf 'LONG_RUN_STATUS=%s\n' "$STATUS"
printf 'LONG_RUN_PID_FILE=%s\n' "$PID_FILE"
printf 'LONG_RUN_DETACHED_PID=%s\n' "$DETACHED_PID"
