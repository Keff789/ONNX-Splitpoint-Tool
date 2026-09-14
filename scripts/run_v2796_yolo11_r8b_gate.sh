#!/usr/bin/env bash
# Focused YOLO11l Full + b067 technical closure before the seven-model run.

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  printf '%s\n' 'SKIP: run_v2796_yolo11_r8b_gate.sh was sourced; run it in its own Bash process.' >&2
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
TOOL_PYTHON_REQUEST="${TOOL_PYTHON:-}"
PROFILE_SHA256=87c5b102f012b0b9b41de76c7d13980d7e52742d65a2201b01b6e42153ede4eb
MODEL_SHA256=f0fcdf56a4ac24d87ec30c627170492ccad9db80486ec5694df6de65c1b3d147

usage() {
  printf '%s\n' \
    'Usage: run_v2796_yolo11_r8b_gate.sh [--out-root ABS] [--models-root ABS]' \
    '' \
    'Runs exactly YOLO11l Full and b067 composed on Hailo-8, Hailo-10H and' \
    'DeepX-M1. The gate passes only when all six paths end in technical' \
    'success or an exact immutable technical block receipt.'
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
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die 64 "Unbekannte Option: $1"
      ;;
  esac
done

require_absolute() {
  local label="$1"
  local value="$2"
  [[ "$value" == /* && "$(realpath -m -- "$value")" == "$value" ]] ||
    die 64 "$label muss absolut und kanonisch sein: $value"
}

require_absolute TOOL "$TOOL"
require_absolute OUTPUT_ROOT "$OUTPUT_ROOT"
require_absolute MODELS_ROOT "$MODELS_ROOT"
[[ -d "$TOOL" && ! -L "$TOOL" ]] || die 66 "Tool-Verzeichnis fehlt oder ist ein Symlink: $TOOL"
TOOL="$(realpath -e -- "$TOOL")"
[[ -d "$MODELS_ROOT" && ! -L "$MODELS_ROOT" ]] || die 66 "Modellverzeichnis fehlt oder ist ein Symlink: $MODELS_ROOT"
MODELS_ROOT="$(realpath -e -- "$MODELS_ROOT")"
TOOL_PYTHON="${TOOL_PYTHON_REQUEST:-$TOOL/.venv/bin/python}"
require_absolute TOOL_PYTHON "$TOOL_PYTHON"
[[ -x "$TOOL_PYTHON" ]] || die 69 "Tool-Python fehlt: $TOOL_PYTHON"
# bin/python is normally a venv symlink. Bind its effective interpreter to the
# selected Tool installation instead of rejecting the standard venv layout.
"$TOOL_PYTHON" -B - "$TOOL/.venv" <<'PY' || die 69 "Tool-Python gehört nicht zur Tool-venv"
import os
import sys

if os.path.realpath(sys.prefix) != os.path.realpath(sys.argv[1]):
    raise SystemExit(
        f"unexpected interpreter prefix: {sys.prefix!r}; expected {sys.argv[1]!r}"
    )
PY

PROFILE="$TOOL/profiles/yolo11l_v2796_r8b_full_b067_gate.yaml"
WORKFLOW="$TOOL/scripts/run_evaluation_workflow.py"
VERIFIER="$TOOL/scripts/verify_v2796_yolo11_r8b_gate.py"
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

"$TOOL_PYTHON" -B - "$TOOL" <<'PY' ||
  die 65 "Installierte v2.79.6-Package-/Workflow-Identität stimmt nicht"
import importlib.metadata
import os
import sys

import onnx_splitpoint_tool
from onnx_splitpoint_tool.workflow.runner import WORKFLOW_VERSION

tool = os.path.realpath(sys.argv[1])
package_root = os.path.realpath(
    os.path.dirname(os.path.dirname(onnx_splitpoint_tool.__file__))
)
expected_version = "2.79.6"
expected_build = "v2.79.6-remaining-changes-yolo11-admission-closure"
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
print("V2796_R8B_RELEASE_IDENTITY=PASS")
PY

"$TOOL_PYTHON" -B "$SOURCE_MANIFEST_VERIFIER" \
  --root "$TOOL" \
  --verify \
  --scope installed >/dev/null ||
  die 65 "Installierter v2.79.6-Source-Manifest-Vertrag ist inkonsistent"
SOURCE_MANIFEST_SHA256="$(sha256sum "$SOURCE_MANIFEST" | awk '{print $1}')"
printf 'V2796_R8B_SOURCE_MANIFEST=PASS\n'
printf 'SOURCE_MANIFEST_SHA256=%s\n' "$SOURCE_MANIFEST_SHA256"

"$TOOL_PYTHON" -B - "$PROFILE" "$PROFILE_SHA256" "$MODEL" "$MODEL_SHA256" "$HARDWARE_SETUPS" <<'PY' ||
  die 65 "R8B-Profil-, Modell- oder Hardwareidentität stimmt nicht"
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
GATE_OUTPUT="$OUTPUT_ROOT/v2796_yolo11_r8b_${STAMP}_$$"
[[ ! -e "$GATE_OUTPUT" && ! -L "$GATE_OUTPUT" ]] || die 73 "Einmaliges Canary-Ziel existiert bereits: $GATE_OUTPUT"
GATE_OUTPUT="$(secure_directory "$GATE_OUTPUT")" || die 73 "Canary-Ziel konnte nicht sicher erstellt werden"

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
  --expected-profile "$PROFILE" | tee "$VERDICT"
verify_status=("${PIPESTATUS[@]}")
set -e
verifier_rc="${verify_status[0]}"
verdict_tee_rc="${verify_status[1]}"
((verdict_tee_rc == 0)) || die 74 "Gate-Verifikation konnte nicht geschrieben werden: $VERDICT"

printf 'PROFILE=%s\n' "$PROFILE"
printf 'PROFILE_SHA256=%s\n' "$PROFILE_SHA256"
printf 'GATE_OUTPUT=%s\n' "$GATE_OUTPUT"
printf 'GATE_LOG=%s\n' "$GATE_LOG"
printf 'WORKFLOW_RC=%s\n' "$workflow_rc"
printf 'VERDICT=%s\n' "$VERDICT"
if ((verifier_rc == 0)); then
  printf 'V2796_YOLO11_R8B_GATE=PASS\n'
else
  printf 'V2796_YOLO11_R8B_GATE=FAIL\n' >&2
fi
exit "$verifier_rc"
