#!/usr/bin/env bash
# Verify the retained YOLO11 Gate-A result and start the frozen seven-model run.

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  printf '%s\n' \
    'SKIP: run_v2784_seven_model_long_overnight.sh was sourced.' \
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
    '  bash scripts/run_v2784_seven_model_long_overnight.sh [options]' \
    '' \
    'Options:' \
    '  --gate-output PATH       Retained outer v2.78.3 Gate-A directory.' \
    '  --out-root PATH          Parent for the new detached long run.' \
    '  --models-root PATH       Directory containing the seven ONNX files.' \
    '  --hardware-setups PATH   Current central hardware registry to freeze.' \
    '  --profile PATH           Frozen v2.78.4 seven-model profile.' \
    '  --preflight-only         Verify and freeze inputs, but do not start.' \
    '  --help                   Show this help.' \
    '' \
    'The launcher never invokes Gate A, never resumes an earlier evaluation,' \
    'and never writes below the supplied Gate-A directory.'
}

SCRIPT_FILE="${BASH_SOURCE[0]}"
[[ -f "$SCRIPT_FILE" && ! -L "$SCRIPT_FILE" ]] ||
  die 66 "Launcher fehlt oder ist ein Symlink: $SCRIPT_FILE"
SCRIPT_DIR="$(cd -- "$(dirname -- "$SCRIPT_FILE")" && pwd -P)"
DEFAULT_TOOL="$(dirname -- "$SCRIPT_DIR")"

TOOL="${TOOL:-$DEFAULT_TOOL}"
TOOL_PYTHON_REQUEST="${TOOL_PYTHON:-}"
GATE_OUTPUT=/home/kmika/Models/EvaluationRunsCanary/v2783_yolo11_hailo8_first_gate_a_20260828_125840_518433993_1163101
OUT_ROOT=/home/kmika/Models/EvaluationRuns
MODELS_ROOT=/home/kmika/Models
HARDWARE_SETUPS=/home/kmika/.onnx_splitpoint_tool/hardware_setups.yaml
PROFILE_REQUEST=""
PREFLIGHT_ONLY=0
WORKFLOW_RC=1

while (( $# > 0 )); do
  case "$1" in
    --gate-output)
      (( $# >= 2 )) || die 64 '--gate-output requires a path'
      GATE_OUTPUT="$2"
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
    --profile)
      (( $# >= 2 )) || die 64 '--profile requires a path'
      PROFILE_REQUEST="$2"
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

require_absolute() {
  local label="$1"
  local value="$2"
  [[ "$value" == /* ]] || die 64 "$label muss absolut sein: $value"
  [[ "$(realpath -m -- "$value")" == "$value" ]] ||
    die 64 "$label ist nicht kanonisch: $value"
}

for pair in \
  "TOOL:$TOOL" \
  "GATE_OUTPUT:$GATE_OUTPUT" \
  "OUT_ROOT:$OUT_ROOT" \
  "MODELS_ROOT:$MODELS_ROOT" \
  "HARDWARE_SETUPS:$HARDWARE_SETUPS"
do
  require_absolute "${pair%%:*}" "${pair#*:}"
done

[[ -d "$TOOL" && ! -L "$TOOL" ]] ||
  die 66 "Tool-Verzeichnis fehlt oder ist ein Symlink: $TOOL"
TOOL="$(realpath -e -- "$TOOL")"
TOOL_PYTHON="${TOOL_PYTHON_REQUEST:-$TOOL/.venv/bin/python}"
require_absolute TOOL_PYTHON "$TOOL_PYTHON"
[[ -x "$TOOL_PYTHON" ]] || die 69 "Tool-Python fehlt: $TOOL_PYTHON"

PROFILE="${PROFILE_REQUEST:-$TOOL/profiles/complete_set_7models_v2784_b500_audit20.yaml}"
require_absolute PROFILE "$PROFILE"
[[ -f "$PROFILE" && ! -L "$PROFILE" ]] ||
  die 66 "Eingefrorenes Langprofil fehlt oder ist unsicher: $PROFILE"
[[ -d "$GATE_OUTPUT" && ! -L "$GATE_OUTPUT" ]] ||
  die 66 "Retained Gate-A-Ausgabe fehlt oder ist unsicher: $GATE_OUTPUT"
[[ -d "$MODELS_ROOT" && ! -L "$MODELS_ROOT" ]] ||
  die 66 "Modellwurzel fehlt oder ist unsicher: $MODELS_ROOT"
[[ -f "$HARDWARE_SETUPS" && ! -L "$HARDWARE_SETUPS" ]] ||
  die 66 "Hardware-Registry fehlt oder ist unsicher: $HARDWARE_SETUPS"

GATE_VERIFIER="$TOOL/scripts/verify_v2783_yolo11_gate_a_output.py"
GATE_PROFILE="$TOOL/profiles/yolo11l_v2783_hailo8_first_b5_gate_a.yaml"
ACCEPTANCE="$TOOL/scripts/run_v278_small_acceptance.sh"
WORKFLOW_LAUNCHER="$TOOL/scripts/run_fresh_standard_workflow.sh"
for required in \
  "$GATE_VERIFIER" \
  "$GATE_PROFILE" \
  "$ACCEPTANCE" \
  "$WORKFLOW_LAUNCHER"
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

# This check intentionally precedes every mutable launch artifact.  The release
# acceptance subsequently verifies the complete installed source manifest.
"$TOOL_PYTHON" -B - <<'PY'
import importlib.metadata
import onnx_splitpoint_tool

expected = "2.78.4"
package = str(getattr(onnx_splitpoint_tool, "__version__", ""))
distribution = importlib.metadata.version("onnx-splitpoint-tool")
if package != expected or distribution != expected:
    raise SystemExit(
        f"v2.78.4 required: package={package!r} distribution={distribution!r}"
    )
print("V2784_RELEASE_IDENTITY=PASS")
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

FROZEN_PARENT=/home/kmika/.onnx_splitpoint_tool/frozen
LOCK_PARENT=/home/kmika/.onnx_splitpoint_tool/locks
FROZEN_PARENT="$(secure_directory "$FROZEN_PARENT")" ||
  die 73 "Freeze-Verzeichnis konnte nicht sicher angelegt werden"
LOCK_PARENT="$(secure_directory "$LOCK_PARENT")" ||
  die 73 "Lock-Verzeichnis konnte nicht sicher angelegt werden"
FROZEN_HARDWARE="$FROZEN_PARENT/v2784_7model_hardware_setups.yaml"
GLOBAL_LOCK="$LOCK_PARENT/v2784_7model_b500_audit20.lock"
[[ ! -L "$FROZEN_HARDWARE" ]] ||
  die 67 "Eingefrorene Hardware-Registry ist ein Symlink"
[[ ! -L "$GLOBAL_LOCK" ]] || die 67 "Langlauf-Lock ist ein Symlink"

if [[ -e "$FROZEN_HARDWARE" ]]; then
  [[ -f "$FROZEN_HARDWARE" && ! -L "$FROZEN_HARDWARE" ]] ||
    die 67 "Freeze-Ziel ist keine reguläre Datei"
  cmp --silent -- "$HARDWARE_SETUPS" "$FROZEN_HARDWARE" ||
    die 78 "Vorhandener v2.78.4-Hardware-Freeze weicht von der Registry ab"
else
  cp --reflink=auto --preserve=mode,timestamps -- \
    "$HARDWARE_SETUPS" "$FROZEN_HARDWARE"
  chmod 0600 -- "$FROZEN_HARDWARE"
fi

STAMP="$(date -u +%Y%m%d_%H%M%S_%N)"
CONTROL="$OUT_ROOT/v2784_7model_b500_audit20_launch_${STAMP}_$$"
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

GATE_RECEIPT="$CONTROL/gate_a_verification.json"
GATE_TMP="$CONTROL/.gate_a_verification.tmp"
set +e
"$TOOL_PYTHON" -B "$GATE_VERIFIER" \
  --gate-output "$GATE_OUTPUT" \
  --workflow-rc "$WORKFLOW_RC" \
  --expected-profile "$GATE_PROFILE" \
  --format json > "$GATE_TMP"
gate_rc=$?
set -e
mv -- "$GATE_TMP" "$GATE_RECEIPT"
(( gate_rc == 0 )) ||
  die 2 "Retained Gate A ist nicht launchfähig (Verifier rc=$gate_rc; $GATE_RECEIPT)"

"$TOOL_PYTHON" -B - "$GATE_RECEIPT" <<'PY'
import json
import os
import stat
import sys

path = sys.argv[1]
fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
try:
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode) or info.st_size > 8 * 1024 * 1024:
        raise SystemExit("unsafe Gate-A verdict")
    with os.fdopen(fd, "r", encoding="utf-8") as stream:
        fd = -1
        value = json.load(stream)
finally:
    if fd >= 0:
        os.close(fd)
if not isinstance(value, dict):
    raise SystemExit("Gate-A verdict is not an object")
required = {
    "status": "PASS",
    "ok": True,
    "valid_terminal": True,
    "outcome": "ANCHOR_FOUND",
    "anchor_boundary": 67,
    "expected_stop_after": "build_backend_artifacts",
    "expected_stop_after_partial": True,
    "workflow_rc": 1,
    "profile_id": "yolo11l_v2783_hailo8_first_b5_gate_a",
    "model_id": "yolo11l",
}
for key, expected in required.items():
    if value.get(key) != expected:
        raise SystemExit(f"Gate-A verdict mismatch: {key}")
artifacts = value.get("artifacts")
if not isinstance(artifacts, dict) or set(artifacts) != {"hailo8", "hailo10"}:
    raise SystemExit("Gate-A Hailo evidence set mismatch")
for target in ("hailo8", "hailo10"):
    row = artifacts.get(target)
    if not isinstance(row, dict):
        raise SystemExit(f"Gate-A evidence missing: {target}")
    if not row.get("hef_sha256") or not row.get("cache_key"):
        raise SystemExit(f"Gate-A evidence identity missing: {target}")
source_sha = str(value.get("full_source_onnx_sha256") or "").lower()
if len(source_sha) != 64 or any(c not in "0123456789abcdef" for c in source_sha):
    raise SystemExit("Gate-A Full source identity missing")
print("V2784_GATE_A_OFFLINE_ADMISSION=PASS")
PY

ADMISSION="$CONTROL/long_run_admission.json"
ADMISSION_TMP="$CONTROL/.long_run_admission.tmp"
"$TOOL_PYTHON" -B - \
  "$PROFILE" "$MODELS_ROOT" "$FROZEN_HARDWARE" \
  "$GATE_RECEIPT" "$CONTROL" > "$ADMISSION_TMP" <<'PY'
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
from onnx_splitpoint_tool.hailo_backend import (
    _load_valid_hailo_receipt,
)
from onnx_splitpoint_tool.workflow.hardware_matrix import (
    canon_accelerator,
    normalize_hardware_targets,
)

profile_path = Path(sys.argv[1])
models_root = Path(sys.argv[2]).resolve(strict=True)
hardware_path = Path(sys.argv[3])
gate_receipt_path = Path(sys.argv[4])
output_filesystem = Path(sys.argv[5])
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
    "hailo10_to_trt",
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
need(profile.get("name") == "complete_set_7models_v2784_b500_audit20", "profile_name")

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
    declared = str(row.get("model_sha256") or "").lower().removeprefix("sha256:")
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
need(dict(profile.get("native_producers") or {}).get("enabled") is False, "native_enabled")
energy = dict(profile.get("energy") or {})
need(energy.get("enabled") is False and energy.get("generic_enabled") is False, "energy_enabled")

preset = dict(profile.get("execution_preset") or {})
need(preset.get("id") == "standard", "run_mode")
need(preset.get("follow_tool_config") is False, "run_mode_not_frozen")

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
need(gate_verdict.get("ok") is True, "gate_verdict_not_pass")
gate_full_sha = str(gate_verdict.get("full_source_onnx_sha256") or "").lower()
need(gate_full_sha == models["yolo11l"]["sha256"], "yolo11_gate_model_identity")
gate_artifacts = dict(gate_verdict.get("artifacts") or {})
cache_root_requested = Path(
    os.path.expandvars(
        os.path.expanduser(
            str(
                dict(profile.get("hailo_build") or {}).get("cache_root")
                or "~/.cache/onnx_splitpoint/hailo_hef"
            )
        )
    )
)
need(
    cache_root_requested.is_absolute()
    and cache_root_requested.is_dir()
    and not cache_root_requested.is_symlink(),
    "gate_cache_root",
)
cache_root = cache_root_requested.resolve(strict=True)
cache_evidence = {}
for target in ("hailo8", "hailo10"):
    gate_artifact = dict(gate_artifacts.get(target) or {})
    cache_key = str(gate_artifact.get("cache_key") or "").lower()
    need(
        len(cache_key) == 64
        and all(character in "0123456789abcdef" for character in cache_key),
        f"gate_cache_key:{target}",
    )
    cache_dir = cache_root / cache_key
    need(cache_dir.is_dir() and not cache_dir.is_symlink(), f"gate_cache_dir:{target}")
    cache_hef = cache_dir / "compiled.hef"
    cache_receipt = cache_dir / "hailo_hef_build_receipt.json"
    need(cache_hef.is_file() and not cache_hef.is_symlink(), f"gate_cache_hef:{target}")
    need(
        cache_receipt.is_file() and not cache_receipt.is_symlink(),
        f"gate_cache_receipt:{target}",
    )
    cache_identity = file_identity(cache_hef, limit=64 * 1024 * 1024)
    need(
        cache_identity["sha256"] == str(gate_artifact.get("hef_sha256") or ""),
        f"gate_cache_hef_identity:{target}",
    )
    validated_cache = _load_valid_hailo_receipt(
        cache_hef,
        cache_key=cache_key,
    )
    need(validated_cache is not None, f"gate_cache_contract:{target}")
    cache_evidence[target] = {
        "cache_key": cache_key,
        "hef": cache_identity,
        "receipt": file_identity(cache_receipt, limit=8 * 1024 * 1024),
    }

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
    "schema": "onnx-splitpoint/v2784-seven-model-long-run-admission/v1",
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
    "yolo11_gate_full_source_onnx_sha256": gate_full_sha,
    "yolo11_gate_cache": cache_evidence,
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
    "gate_a_rerun": False,
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
print("V2784_SEVEN_MODEL_LONG_ADMISSION=PASS")
PY

PROFILE_SHA256="$(sha256sum "$PROFILE" | awk '{print $1}')"
GATE_RECEIPT_SHA256="$(sha256sum "$GATE_RECEIPT" | awk '{print $1}')"
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
  die 75 "Ein anderer v2.78.4-Sieben-Modell-Langlauf hält bereits den Lock"
fi
flock -u "$probe_fd"
exec {probe_fd}>&-

if (( PREFLIGHT_ONLY )); then
  printf '%s\n' \
    'V2784_SEVEN_MODEL_LONG_PREFLIGHT=PASS' \
    'LONG_RUN_STARTED=NO' \
    'GATE_A_RERUN=NO' \
    'YOLO11_ANCHOR_REUSED=b067'
  printf 'LAUNCH_CONTROL=%s\n' "$CONTROL"
  printf 'GATE_A_VERDICT=%s\n' "$GATE_RECEIPT"
  printf 'LONG_RUN_ADMISSION=%s\n' "$ADMISSION"
  exit 0
fi

{
  printf 'STATE=STARTING\n'
  printf 'WORKFLOW_RC=\n'
  printf 'UPDATED_AT=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'RUN_ROOT=%s\n' "$RUN_ROOT"
  printf 'LOG=%s\n' "$LOG"
} > "$STATUS"

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
  admission="$9"
  admission_sha="${10}"
  hardware="${11}"
  hardware_sha="${12}"

  write_status() {
    local state="$1"
    local rc="$2"
    local temporary="${status}.tmp-${BASHPID}"
    {
      printf "STATE=%s\\n" "$state"
      printf "WORKFLOW_RC=%s\\n" "$rc"
      printf "WORKER_PID=%s\\n" "$BASHPID"
      printf "UPDATED_AT=%s\\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
      printf "RUN_ROOT=%s\\n" "$run_root"
    } > "$temporary"
    mv -- "$temporary" "$status"
  }

  exec 9>"$lock"
  if ! flock -n 9; then
    write_status LOCKED 75
    exit 75
  fi

  printf "%s  %s\\n" "$profile_sha" "$profile" | sha256sum -c -
  printf "%s  %s\\n" "$gate_sha" "$gate_receipt" | sha256sum -c -
  printf "%s  %s\\n" "$admission_sha" "$admission" | sha256sum -c -
  printf "%s  %s\\n" "$hardware_sha" "$hardware" | sha256sum -c -

  finished=0
  on_exit() {
    local rc=$?
    if (( finished == 0 )); then
      write_status FAILED "$rc"
    fi
  }
  trap on_exit EXIT
  write_status RUNNING ""

  set +e
  PY="$tool/.venv/bin/python" \
    bash "$tool/scripts/run_fresh_standard_workflow.sh" \
      --profile "$profile" \
      --out "$run_root"
  rc=$?
  set -e

  finished=1
  if (( rc == 0 )); then
    write_status COMPLETED 0
  else
    write_status FAILED "$rc"
  fi
  exit "$rc"
' v2784-seven-model-worker \
  "$STATUS" "$GLOBAL_LOCK" "$TOOL" "$PROFILE" "$RUN_ROOT" \
  "$PROFILE_SHA256" "$GATE_RECEIPT" "$GATE_RECEIPT_SHA256" \
  "$ADMISSION" "$ADMISSION_SHA256" \
  "$FROZEN_HARDWARE" "$HARDWARE_SHA256" \
  > "$LOG" 2>&1 < /dev/null &
MONITOR_PID=$!

PID_TMP="$CONTROL/.launcher.pid.tmp"
printf '%s\n' "$MONITOR_PID" > "$PID_TMP"
mv -- "$PID_TMP" "$PID_FILE"

# Detect immediate lock/hash/command failures without waiting for the workflow.
sleep 2
if ! kill -0 "$MONITOR_PID" 2>/dev/null; then
  set +e
  wait "$MONITOR_PID"
  start_rc=$?
  set -e
  tail -n 80 -- "$LOG" >&2 || true
  die "$start_rc" "Abgekoppelter Langlauf ist sofort beendet"
fi

printf '%s\n' \
  'V2784_SEVEN_MODEL_LONG_PREFLIGHT=PASS' \
  'V2784_SEVEN_MODEL_LONG_RUN=STARTED' \
  'DETACHED=YES' \
  'GATE_A_RERUN=NO' \
  'YOLO11_ANCHOR_REUSED=b067' \
  'NATIVE_RUN=NO' \
  'ENERGY_RUN=NO'
printf 'LAUNCH_CONTROL=%s\n' "$CONTROL"
printf 'LONG_RUN_ROOT=%s\n' "$RUN_ROOT"
printf 'LONG_RUN_LOG=%s\n' "$LOG"
printf 'LONG_RUN_STATUS=%s\n' "$STATUS"
printf 'LONG_RUN_PID_FILE=%s\n' "$PID_FILE"
printf 'LONG_RUN_MONITOR_PID=%s\n' "$MONITOR_PID"
