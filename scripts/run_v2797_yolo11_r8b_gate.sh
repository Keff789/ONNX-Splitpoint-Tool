#!/usr/bin/env bash
# Focused YOLO11l Full + b067 technical closure before the seven-model run.

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  printf '%s\n' 'SKIP: run_v2797_yolo11_r8b_gate.sh was sourced; run it in its own Bash process.' >&2
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
[[ -f "$SCRIPT_FILE" && ! -L "$SCRIPT_FILE" ]] || die 66 "Launcher fehlt oder ist ein Symlink: $SCRIPT_FILE"
SCRIPT_DIR="$(cd -- "$(dirname -- "$SCRIPT_FILE")" && pwd -P)"
DEFAULT_TOOL="$(dirname -- "$SCRIPT_DIR")"
TOOL="${TOOL:-$DEFAULT_TOOL}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/kmika/Models/EvaluationRunsCanary}"
MODELS_ROOT="${MODELS_ROOT:-/home/kmika/Models}"
RECOVERY_MANIFEST="${RECOVERY_MANIFEST:-}"
TOOL_PYTHON_REQUEST="${TOOL_PYTHON:-}"
PROFILE_SHA256=69349eb6d84b59be5fc0bd5316b7eb5204d3f672dde2788bb5030702c9ed4ad7
MODEL_SHA256=f0fcdf56a4ac24d87ec30c627170492ccad9db80486ec5694df6de65c1b3d147

usage() {
  printf '%s\n' \
    'Usage: run_v2797_yolo11_r8b_gate.sh --recovery-manifest ABS [--out-root ABS] [--models-root ABS]' \
    '' \
    'Runs exactly YOLO11l Full and b067 composed on Hailo-8, Hailo-10H and' \
    'DeepX-M1. The gate passes only when all six backend-bound paths succeed.' \
    'Exactly the prior Hailo-8 b067 path is admitted through the required' \
    'read-only, hash-bound v2797 recovery manifest.'
}

while (($#)); do
  case "$1" in
    --out-root)
      (($# >= 2)) || die 64 "--out-root erwartet einen absoluten Pfad"
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --models-root)
      (($# >= 2)) || die 64 "--models-root erwartet einen absoluten Pfad"
      MODELS_ROOT="$2"
      shift 2
      ;;
    --recovery-manifest)
      (($# >= 2)) || die 64 "--recovery-manifest erwartet einen absoluten Pfad"
      RECOVERY_MANIFEST="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die 64 "Unbekannte Option: $1"
      ;;
  esac
done

[[ -n "$RECOVERY_MANIFEST" ]] || die 64 "--recovery-manifest ist obligatorisch"

require_absolute() {
  local label="$1"
  local value="$2"
  local normalized
  [[ "$value" == /* ]] || die 64 "$label muss absolut sein: $value"
  normalized="$(realpath -m -s -- "$value")" ||
    die 64 "$label konnte nicht normalisiert werden: $value"
  [[ "$normalized" == "$value" ]] ||
    die 64 "$label muss lexikalisch kanonisch sein: $value"
}

require_absolute TOOL "$TOOL"
require_absolute OUTPUT_ROOT "$OUTPUT_ROOT"
require_absolute MODELS_ROOT "$MODELS_ROOT"
require_absolute RECOVERY_MANIFEST "$RECOVERY_MANIFEST"
[[ -d "$TOOL" && ! -L "$TOOL" ]] || die 66 "Tool-Verzeichnis fehlt oder ist ein Symlink: $TOOL"
TOOL="$(realpath -e -- "$TOOL")"
[[ -d "$MODELS_ROOT" && ! -L "$MODELS_ROOT" ]] || die 66 "Modellverzeichnis fehlt oder ist ein Symlink: $MODELS_ROOT"
MODELS_ROOT="$(realpath -e -- "$MODELS_ROOT")"
[[ -f "$RECOVERY_MANIFEST" && ! -L "$RECOVERY_MANIFEST" ]] ||
  die 66 "Recovery-Manifest fehlt, ist kein reguläres File oder ist ein Symlink: $RECOVERY_MANIFEST"
RECOVERY_MANIFEST_REAL="$(realpath -e -- "$RECOVERY_MANIFEST")" ||
  die 66 "Recovery-Manifest kann nicht kanonisch aufgelöst werden: $RECOVERY_MANIFEST"
[[ "$RECOVERY_MANIFEST_REAL" == "$RECOVERY_MANIFEST" ]] ||
  die 64 "Recovery-Manifest muss auch semantisch kanonisch sein: $RECOVERY_MANIFEST"
RECOVERY_MANIFEST_SOURCE="$RECOVERY_MANIFEST"
RECOVERY_MANIFEST_SOURCE_SHA256="$(sha256sum "$RECOVERY_MANIFEST_SOURCE" | awk '{print $1}')"
RECOVERY_MANIFEST_SOURCE_SIZE="$(stat -c '%s' -- "$RECOVERY_MANIFEST_SOURCE")"
TOOL_PYTHON="${TOOL_PYTHON_REQUEST:-$TOOL/.venv/bin/python}"
require_absolute TOOL_PYTHON "$TOOL_PYTHON"
[[ -x "$TOOL_PYTHON" ]] || die 69 "Tool-Python fehlt: $TOOL_PYTHON"

# A standard venv intentionally uses bin/python -> system-python symlinks.  Do
# not confuse that implementation detail with path traversal: the requested
# path remains lexically canonical, while the interpreter binds itself to the
# exact Tool venv through sys.prefix and sys.executable.
if ! "$TOOL_PYTHON" -B - "$TOOL/.venv" "$TOOL_PYTHON" <<'PY'
import os
import sys

expected_prefix = os.path.realpath(sys.argv[1])
requested_executable = sys.argv[2]
if os.path.realpath(sys.prefix) != expected_prefix:
    raise SystemExit(
        f"unexpected interpreter prefix: {sys.prefix!r}; expected {sys.argv[1]!r}"
    )
if os.path.dirname(requested_executable) != os.path.join(sys.argv[1], "bin"):
    raise SystemExit(
        f"interpreter is outside the selected venv bin directory: {requested_executable!r}"
    )
if os.path.realpath(sys.executable) != os.path.realpath(requested_executable):
    raise SystemExit(
        f"unexpected effective interpreter: {sys.executable!r}; "
        f"requested {requested_executable!r}"
    )
PY
then
  die 69 "Tool-Python gehört nicht zur Tool-venv"
fi

PROFILE="$TOOL/profiles/yolo11l_v2797_r8b_full_b067_gate.yaml"
WORKFLOW="$TOOL/scripts/run_evaluation_workflow.py"
VERIFIER="$TOOL/scripts/verify_v2797_yolo11_r8b_gate.py"
SOURCE_MANIFEST="$TOOL/SOURCE_MANIFEST.json"
SOURCE_MANIFEST_VERIFIER="$TOOL/scripts/build_source_manifest.py"
HARDWARE_SETUPS=/home/kmika/.onnx_splitpoint_tool/hardware_setups.yaml
MODEL="$MODELS_ROOT/yolo11l.onnx"
for file in \
  "$PROFILE" \
  "$WORKFLOW" \
  "$VERIFIER" \
  "$SOURCE_MANIFEST" \
  "$SOURCE_MANIFEST_VERIFIER" \
  "$HARDWARE_SETUPS" \
  "$MODEL"
do
  [[ -f "$file" && ! -L "$file" ]] || die 66 "Erforderliche reguläre Datei fehlt oder ist ein Symlink: $file"
done

if ! "$TOOL_PYTHON" -B - "$TOOL" <<'PY'
import importlib.metadata
import os
import sys

import onnx_splitpoint_tool
from onnx_splitpoint_tool.workflow.runner import WORKFLOW_VERSION

tool = os.path.realpath(sys.argv[1])
package_root = os.path.realpath(
    os.path.dirname(os.path.dirname(onnx_splitpoint_tool.__file__))
)
expected_version = "2.79.7"
expected_build = "v2.79.7-yolo11-six-path-runtime-identity-closure"
observed = {
    "package_root": package_root,
    "version": str(onnx_splitpoint_tool.__version__),
    "release": str(onnx_splitpoint_tool.__release__),
    "build_id": str(onnx_splitpoint_tool.__build_id__),
    "workflow_version": str(WORKFLOW_VERSION),
    "distribution": importlib.metadata.version("onnx-splitpoint-tool"),
}
expected = {
    "package_root": tool,
    "version": expected_version,
    "release": expected_version,
    "build_id": expected_build,
    "workflow_version": expected_build,
    "distribution": expected_version,
}
if observed != expected:
    raise SystemExit(f"release_identity_mismatch:{observed!r}")
print("V2797_R8B_RELEASE_IDENTITY=PASS")
PY
then
  die 65 "Installierte v2.79.7-Package-/Workflow-Identität stimmt nicht"
fi

if ! "$TOOL_PYTHON" -B "$SOURCE_MANIFEST_VERIFIER" \
  --root "$TOOL" \
  --verify \
  --scope installed >/dev/null
then
  die 65 "Installierter v2.79.7-Source-Manifest-Vertrag ist inkonsistent"
fi
SOURCE_MANIFEST_SHA256="$(sha256sum "$SOURCE_MANIFEST" | awk '{print $1}')"
printf 'V2797_R8B_SOURCE_MANIFEST=PASS\n'
printf 'SOURCE_MANIFEST_SHA256=%s\n' "$SOURCE_MANIFEST_SHA256"

if ! "$TOOL_PYTHON" -B - "$PROFILE" "$PROFILE_SHA256" "$MODEL" "$MODEL_SHA256" "$HARDWARE_SETUPS" <<'PY'
import hashlib
import os
import stat
import sys
from pathlib import Path

import yaml

profile_path, profile_expected, model_path, model_expected, hardware_path = sys.argv[1:]

def regular_sha(path: str) -> str:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise SystemExit(f"not a regular file: {path}")
        digest = hashlib.sha256()
        while True:
            block = os.read(fd, 1024 * 1024)
            if not block:
                return digest.hexdigest()
            digest.update(block)
    finally:
        os.close(fd)

if regular_sha(profile_path) != profile_expected:
    raise SystemExit("profile_sha256_mismatch")
if regular_sha(model_path) != model_expected:
    raise SystemExit("model_sha256_mismatch")
payload = yaml.safe_load(Path(hardware_path).read_text(encoding="utf-8"))
rows = payload.get("setups") or payload.get("hardware_setups") or payload.get("targets") or []
if isinstance(rows, dict):
    rows = [{"id": key, **(value if isinstance(value, dict) else {})} for key, value in rows.items()]
actual = {
    str(row.get("id") or row.get("setup_id") or ""): str(row.get("accelerator") or row.get("provider") or "").lower().replace("-", "_")
    for row in rows if isinstance(row, dict) and row.get("enabled", True)
}
expected = {
    "orin_nx_hailo8_01": {"hailo8"},
    "orin_nx_hailo10_01": {"hailo10", "hailo10h"},
    "orin_nx_deepx_m1_01": {"deepx", "deepx_m1"},
}
for setup_id, providers in expected.items():
    if actual.get(setup_id) not in providers:
        raise SystemExit(f"hardware_setup_identity_mismatch:{setup_id}")
PY
then
  die 65 "R8B-Profil-, Modell- oder Hardwareidentität stimmt nicht"
fi

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
fd = os.open("/", flags)
try:
    for part in parts:
        try:
            child = os.open(part, flags, dir_fd=fd)
        except FileNotFoundError:
            os.mkdir(part, 0o700, dir_fd=fd)
            child = os.open(part, flags, dir_fd=fd)
        if not stat.S_ISDIR(os.fstat(child).st_mode):
            os.close(child)
            raise SystemExit(f"unsafe non-directory component: {path}")
        os.close(fd)
        fd = child
finally:
    os.close(fd)
print(path)
PY
}

OUTPUT_ROOT="$(secure_directory "$OUTPUT_ROOT")" || die 67 "Canary-Ausgabewurzel ist unsicher"
STAMP="$(date -u +%Y%m%d_%H%M%S_%N)"
GATE_OUTPUT="$OUTPUT_ROOT/v2797_yolo11_r8b_${STAMP}_$$"
[[ ! -e "$GATE_OUTPUT" && ! -L "$GATE_OUTPUT" ]] || die 73 "Einmaliges Canary-Ziel existiert bereits: $GATE_OUTPUT"
GATE_OUTPUT="$(secure_directory "$GATE_OUTPUT")" || die 73 "Canary-Ziel konnte nicht sicher erstellt werden"

# Materialise the small recovery receipt inside this gate output.  The source
# remains read-only and byte-identical; O_NOFOLLOW/O_EXCL prevent path rebinding
# or accidental overwrite.  This makes the retained v2 verdict self-contained.
RECOVERY_MANIFEST="$GATE_OUTPUT/v2797_recovery_manifest.json"
"$TOOL_PYTHON" -B - \
  "$RECOVERY_MANIFEST_SOURCE" "$RECOVERY_MANIFEST" \
  "$RECOVERY_MANIFEST_SOURCE_SHA256" "$RECOVERY_MANIFEST_SOURCE_SIZE" <<'PY'
import hashlib
import os
import stat
import sys

source, target, expected_sha, expected_size_text = sys.argv[1:]
expected_size = int(expected_size_text)
read_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
write_flags = (
    os.O_WRONLY | os.O_CREAT | os.O_EXCL
    | getattr(os, "O_NOFOLLOW", 0)
)
source_fd = os.open(source, read_flags)
target_fd = -1
digest = hashlib.sha256()
written = 0
try:
    source_info = os.fstat(source_fd)
    if not stat.S_ISREG(source_info.st_mode):
        raise SystemExit("recovery source is not regular")
    target_fd = os.open(target, write_flags, 0o600)
    while True:
        block = os.read(source_fd, 1024 * 1024)
        if not block:
            break
        digest.update(block)
        offset = 0
        while offset < len(block):
            offset += os.write(target_fd, block[offset:])
        written += len(block)
    os.fsync(target_fd)
finally:
    os.close(source_fd)
    if target_fd >= 0:
        os.close(target_fd)
if written != expected_size or digest.hexdigest() != expected_sha:
    raise SystemExit("recovery copy source identity mismatch")
target_fd = os.open(target, read_flags)
try:
    target_info = os.fstat(target_fd)
    if not stat.S_ISREG(target_info.st_mode) or target_info.st_size != expected_size:
        raise SystemExit("recovery copy target identity mismatch")
    target_digest = hashlib.sha256()
    while True:
        block = os.read(target_fd, 1024 * 1024)
        if not block:
            break
        target_digest.update(block)
finally:
    os.close(target_fd)
if target_digest.hexdigest() != expected_sha:
    raise SystemExit("recovery copy target hash mismatch")
PY
[[ "$(sha256sum "$RECOVERY_MANIFEST_SOURCE" | awk '{print $1}')" == "$RECOVERY_MANIFEST_SOURCE_SHA256" ]] ||
  die 65 "Externes Recovery-Manifest wurde während der Übernahme verändert"
[[ "$(stat -c '%s' -- "$RECOVERY_MANIFEST_SOURCE")" == "$RECOVERY_MANIFEST_SOURCE_SIZE" ]] ||
  die 65 "Externes Recovery-Manifest änderte während der Übernahme seine Größe"

cd -- "$TOOL"
unset PYTHONOPTIMIZE
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$TOOL${PYTHONPATH:+:$PYTHONPATH}"

GATE_LOG="$GATE_OUTPUT/yolo11_r8b_workflow.log"
set +e
"$TOOL_PYTHON" -B "$WORKFLOW" \
  --profile "$PROFILE" \
  --out "$GATE_OUTPUT" \
  --profile-driven \
  --require-run-mode standard \
  --execution-mode generate_and_run \
  --models-root "$MODELS_ROOT" 2>&1 | tee "$GATE_LOG"
pipeline_status=("${PIPESTATUS[@]}")
set -e
workflow_rc="${pipeline_status[0]}"
tee_rc="${pipeline_status[1]}"
((tee_rc == 0)) || die 74 "Canary-Log konnte nicht geschrieben werden: $GATE_LOG"

VERDICT="$GATE_OUTPUT/yolo11_r8b_verification.json"
set +e
"$TOOL_PYTHON" -B "$VERIFIER" \
  --gate-output "$GATE_OUTPUT" \
  --workflow-rc "$workflow_rc" \
  --recovery-manifest "$RECOVERY_MANIFEST" \
  --expected-profile "$PROFILE" | tee "$VERDICT"
verify_status=("${PIPESTATUS[@]}")
set -e
verifier_rc="${verify_status[0]}"
verdict_tee_rc="${verify_status[1]}"
((verdict_tee_rc == 0)) || die 74 "Gate-Verifikation konnte nicht geschrieben werden: $VERDICT"

printf 'PROFILE=%s\n' "$PROFILE"
printf 'PROFILE_SHA256=%s\n' "$PROFILE_SHA256"
printf 'RECOVERY_MANIFEST=%s\n' "$RECOVERY_MANIFEST"
printf 'GATE_OUTPUT=%s\n' "$GATE_OUTPUT"
printf 'GATE_LOG=%s\n' "$GATE_LOG"
printf 'WORKFLOW_RC=%s\n' "$workflow_rc"
printf 'VERDICT=%s\n' "$VERDICT"
if ((verifier_rc == 0)); then
  printf 'V2797_YOLO11_R8B_GATE=PASS\n'
else
  printf 'V2797_YOLO11_R8B_GATE=FAIL\n' >&2
fi
exit "$verifier_rc"
