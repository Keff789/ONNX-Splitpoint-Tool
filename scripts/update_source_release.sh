#!/usr/bin/env bash
# Synchronize a verified source ZIP into an existing Tool installation.
# The .venv and local state are preserved. Replaced and removed source files
# are copied to a timestamped backup outside the installation.

set -Eeuo pipefail

# Never import source-local bytecode while verifying or refreshing a release.
PYCACHE_ROOT="$(
  /usr/bin/mktemp -d /tmp/onnx-splitpoint-update-pycache.XXXXXXXXXX
)" || exit 70
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="$PYCACHE_ROOT"

cleanup_early_pycache() {
  local exit_status=$?
  trap - EXIT
  case "${PYCACHE_ROOT:-}" in
    /tmp/onnx-splitpoint-update-pycache.??????????)
      if [[ -L "$PYCACHE_ROOT" ]]; then
        /bin/rm -- "$PYCACHE_ROOT" || exit_status=70
      elif [[ -d "$PYCACHE_ROOT" ]]; then
        /bin/rm -r -- "$PYCACHE_ROOT" || exit_status=70
      elif [[ -e "$PYCACHE_ROOT" ]]; then
        exit_status=70
      fi
      ;;
    *) exit_status=70 ;;
  esac
  exit "$exit_status"
}
trap cleanup_early_pycache EXIT

MAINTAINER_ALLOW_ACTIVE_WORKFLOW=0
POSITIONAL_ARGS=()
for argument in "$@"; do
  case "$argument" in
    --maintainer-allow-active-workflow)
      MAINTAINER_ALLOW_ACTIVE_WORKFLOW=1
      ;;
    --*)
      printf 'FEHLER: unbekannte Option: %s\n' "$argument" >&2
      exit 64
      ;;
    *)
      POSITIONAL_ARGS+=("$argument")
      ;;
  esac
done
set -- "${POSITIONAL_ARGS[@]}"

if [[ $# -lt 1 || $# -gt 2 ]]; then
  printf 'Usage: %s RELEASE.zip [TOOL_DIR] [--maintainer-allow-active-workflow]\n' "$0" >&2
  exit 64
fi

ZIP="$(readlink -f -- "$1")"
TOOL_DIR="$(readlink -f -- "${2:-$HOME/ONNX-Splitpoint-Tool}")"
NORMALIZED_USER_HOME="$(readlink -f -- "$HOME")"

if [[ ! -f "$ZIP" ]]; then
  printf 'FEHLER: Release-ZIP fehlt: %s\n' "$ZIP" >&2
  exit 66
fi
if [[ ! -d "$TOOL_DIR" || "$TOOL_DIR" == "/" ]]; then
  printf 'FEHLER: unsicherer oder fehlender Tool-Ordner: %s\n' "$TOOL_DIR" >&2
  exit 64
fi
case "$NORMALIZED_USER_HOME/" in
  "$TOOL_DIR/"*)
    printf 'FEHLER: Tool-Ziel ist Home oder ein übergeordneter Pfad: %s\n' \
      "$TOOL_DIR" >&2
    exit 64
    ;;
esac
case "$ZIP" in
  "$TOOL_DIR/"*)
    printf 'FEHLER: Release-ZIP muss außerhalb des Tool-Ordners liegen.\n' >&2
    exit 64
    ;;
esac
for marker in \
  pyproject.toml \
  SOURCE_MANIFEST.json \
  onnx_splitpoint_tool/__init__.py \
  scripts/build_source_manifest.py
do
  if [[ ! -f "$TOOL_DIR/$marker" ]]; then
    printf 'FEHLER: Ziel ist keine bestehende Tool-Installation: %s fehlt\n' \
      "$TOOL_DIR/$marker" >&2
    exit 64
  fi
done
if [[ ! -x "$TOOL_DIR/.venv/bin/python" ]]; then
  printf 'FEHLER: vorhandene Tool-Venv fehlt: %s/.venv/bin/python\n' \
    "$TOOL_DIR" >&2
  exit 69
fi
for command_name in unzip rsync flock; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    printf 'FEHLER: benötigtes Programm fehlt: %s\n' "$command_name" >&2
    exit 69
  fi
done

# Serialize source replacement against every workflow/platform operation using
# one canonical interlock.  The updater owns the exclusive side for its entire
# process lifetime.  This lock is intentionally stronger than the legacy
# per-workflow scan below and cannot be bypassed by the maintainer recovery
# flag: updating Python source below a newly admitted workflow is never safe.
WORKFLOW_LOCK_DIR="$NORMALIZED_USER_HOME/.onnx_splitpoint_tool/locks"
if ! mkdir -p -- "$WORKFLOW_LOCK_DIR"; then
  printf 'FEHLER: Workflow-Lockverzeichnis kann nicht erstellt werden: %s\n' \
    "$WORKFLOW_LOCK_DIR" >&2
  exit 73
fi
CANONICAL_WORKFLOW_LOCK_DIR="$(readlink -f -- "$WORKFLOW_LOCK_DIR")"
if [[ "$CANONICAL_WORKFLOW_LOCK_DIR" != "$WORKFLOW_LOCK_DIR" ]]; then
  printf 'FEHLER: Workflow-Lockverzeichnis ist nicht kanonisch: %s\n' \
    "$WORKFLOW_LOCK_DIR" >&2
  exit 65
fi
PLATFORM_INTERLOCK="$WORKFLOW_LOCK_DIR/workflow_platform_interlock.lock"
if [[ -L "$PLATFORM_INTERLOCK" || \
      ( -e "$PLATFORM_INTERLOCK" && ! -f "$PLATFORM_INTERLOCK" ) ]]; then
  printf 'FEHLER: unsicherer Plattform-Interlockpfad: %s\n' \
    "$PLATFORM_INTERLOCK" >&2
  exit 65
fi
if ! exec {PLATFORM_INTERLOCK_FD}<>"$PLATFORM_INTERLOCK"; then
  printf 'FEHLER: Plattform-Interlock kann nicht geoeffnet werden: %s\n' \
    "$PLATFORM_INTERLOCK" >&2
  exit 73
fi
OPENED_PLATFORM_INTERLOCK="$(
  readlink -f -- "/proc/self/fd/$PLATFORM_INTERLOCK_FD"
)" || {
  printf 'FEHLER: geoeffneter Plattform-Interlock kann nicht validiert werden.\n' >&2
  exit 65
}
if [[ "$OPENED_PLATFORM_INTERLOCK" != "$PLATFORM_INTERLOCK" || \
      ! -f "$PLATFORM_INTERLOCK" || -L "$PLATFORM_INTERLOCK" ]]; then
  printf 'FEHLER: Plattform-Interlock ist nicht die kanonische regulaere Datei: %s\n' \
    "$PLATFORM_INTERLOCK" >&2
  exit 65
fi
if ! flock -n -x "$PLATFORM_INTERLOCK_FD"; then
  printf '%s\n' \
    'FEHLER: Quellupdate blockiert; Workflow/Plattform-Interlock wird gemeinsam gehalten.' \
    'Der Maintainer-Override kann diesen globalen Interlock nicht umgehen.' >&2
  exit 75
fi

# Lock the canonical target directory inode for the complete update.  Opening
# the directory itself avoids a stale/symlinkable lock-file path.  A concurrent
# updater for the same installation must fail before staging or mutation.
if ! exec {UPDATE_LOCK_FD}<"$TOOL_DIR"; then
  printf 'FEHLER: Tool-Ziel kann nicht für das Update gesperrt werden: %s\n' \
    "$TOOL_DIR" >&2
  exit 73
fi
if ! flock -n "$UPDATE_LOCK_FD"; then
  printf 'FEHLER: Für dieses Tool-Ziel läuft bereits ein Update: %s\n' \
    "$TOOL_DIR" >&2
  exit 75
fi

# A source update changes Python modules beneath running processes.  Refuse it
# while any real workflow lock is held; the mere existence of a stale lock file
# is deliberately not enough.  The override is a maintainer-only recovery flag
# and remains noisy so routine installers cannot silently use it.
HELD_WORKFLOW_LOCKS=()
if [[ -d "$WORKFLOW_LOCK_DIR" ]]; then
  while IFS= read -r -d '' workflow_lock; do
    # This updater already holds the exclusive canonical interlock above.  It
    # must not rediscover its own FD as an unrelated active workflow.
    if [[ "$workflow_lock" == "$PLATFORM_INTERLOCK" ]]; then
      continue
    fi
    if [[ -L "$workflow_lock" || ! -f "$workflow_lock" ]]; then
      printf 'FEHLER: unsicherer Workflow-Lockpfad: %s\n' "$workflow_lock" >&2
      exit 65
    fi
    if ! exec {WORKFLOW_LOCK_FD}<>"$workflow_lock"; then
      printf 'FEHLER: Workflow-Lock kann nicht geöffnet werden: %s\n' \
        "$workflow_lock" >&2
      exit 65
    fi
    if flock -n "$WORKFLOW_LOCK_FD"; then
      flock -u "$WORKFLOW_LOCK_FD"
    else
      HELD_WORKFLOW_LOCKS+=("$workflow_lock")
    fi
    exec {WORKFLOW_LOCK_FD}>&-
  done < <(
    find "$WORKFLOW_LOCK_DIR" -mindepth 1 -maxdepth 1 -name '*.lock' -print0
  )
fi
if (( ${#HELD_WORKFLOW_LOCKS[@]} > 0 )); then
  if (( MAINTAINER_ALLOW_ACTIVE_WORKFLOW == 0 )); then
    printf '%s\n' \
      'FEHLER: Quellupdate blockiert; mindestens ein Workflow-Lock wird gehalten:' >&2
    printf '  %s\n' "${HELD_WORKFLOW_LOCKS[@]}" >&2
    printf '%s\n' \
      'Erst den laufenden Benchmark beenden. Nur Maintainer-Recovery darf --maintainer-allow-active-workflow verwenden.' >&2
    exit 75
  fi
  printf '%s\n' \
    'WARNUNG: Maintainer-Override aktiv; Update trotz gehaltenem Workflow-Lock.' >&2
  printf '  %s\n' "${HELD_WORKFLOW_LOCKS[@]}" >&2
fi

PYTHON="$TOOL_DIR/.venv/bin/python"
# Idle or SIGSTOP-paused GUIs can still import this installation later, even
# without a workflow lock. Refuse source replacement before staging anything.
"$TOOL_DIR/.venv/bin/python" -I -S -B - "$TOOL_DIR" <<'PY_GUI_GUARD'
from pathlib import Path
import os
import sys

target = Path(sys.argv[1]).resolve(strict=True)
def below(path, root):
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False

blocked = []
for process in Path("/proc").iterdir():
    if not process.name.isdigit() or int(process.name) == os.getpid():
        continue
    try:
        if process.stat().st_uid != os.getuid():
            continue
        argv = [part.decode("utf-8", errors="replace") for part in (process / "cmdline").read_bytes().split(b"\0") if part]
        if not argv:
            continue
        cwd = (process / "cwd").resolve(strict=True)
        state = (process / "stat").read_text().rpartition(")")[2].split()[0]
    except (FileNotFoundError, ProcessLookupError):
        continue
    except PermissionError:
        # An unrelated process of this UID may be unreadable. It cannot be
        # classified as a GUI solely because it exists.
        continue
    command = Path(argv[0])
    command = command if command.is_absolute() else cwd / command
    belongs = below(command.absolute(), target / ".venv") or below(cwd, target)
    module_gui = any(argv[i] == "-m" and argv[i + 1] in {"onnx_splitpoint_tool.gui_app", "onnx_splitpoint_tool.gui.app"} for i in range(len(argv) - 1))
    script_gui = False
    for item in argv[:3]:
        if Path(item).name not in {"analyse_and_split_gui.py", "onnx-splitpoint-gui", "gui_app.py"}:
            continue
        candidate = Path(item)
        candidate = candidate if candidate.is_absolute() else cwd / candidate
        if below(candidate.resolve(strict=False), target):
            script_gui = True
    if script_gui or (module_gui and belongs):
        blocked.append((process.name, state))
if blocked:
    print("FEHLER: Quellupdate blockiert; Tool-GUI läuft oder ist pausiert.", file=sys.stderr)
    for pid, state in blocked:
        print(f"  PID={pid} Zustand={state} Installation={target}", file=sys.stderr)
    print("GUI geordnet beenden; eine Pause gibt die Installation nicht frei.", file=sys.stderr)
    raise SystemExit(75)
PY_GUI_GUARD

STAGE_DIR="$(mktemp -d /tmp/onnx-splitpoint-update.XXXXXX)"
STAGED_ZIP="$STAGE_DIR/release-source.zip"
PRESERVED_PROFILE_DIR="$STAGE_DIR/preserved-custom-profiles"
PRESERVED_PROFILE_COUNT=0
PRESERVED_PROFILES_RESTORED=0

# Archive admission must not import or execute anything from the candidate
# release.  This verifier is embedded in the already-running updater and uses
# only the target interpreter's standard library.  -I -S keeps PYTHONPATH,
# editable .pth files, sitecustomize/usercustomize and source-local import
# shadows out of the verifier process; -B prevents cache writes.
trusted_source_preflight() {
  local mode="$1"
  shift
  PYTHONPATH= PYTHONHOME= "$PYTHON" -I -S -B - "$mode" "$@" <<'PY'
from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
import zipfile
from pathlib import Path, PurePosixPath


EXPECTED_VERSION = "2.90.1"
EXPECTED_BUILD_ID = "v2.90.1"
EXPECTED_PREFIX = f"ONNX-Splitpoint-Tool_v{EXPECTED_VERSION}"
MANIFEST_SCHEMA = "onnx-splitpoint/source-manifest-v1"
MANIFEST_NAME = "SOURCE_MANIFEST.json"
SUMS_NAME = "SHA256SUMS.txt"
INDEX_NAMES = (MANIFEST_NAME, SUMS_NAME)
FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)

MAX_ARCHIVE_BYTES = 1024 * 1024 * 1024
MAX_MANIFEST_BYTES = 16 * 1024 * 1024
MAX_SUMS_BYTES = 64 * 1024 * 1024
MAX_MEMBER_BYTES = 512 * 1024 * 1024
MAX_TOTAL_MEMBER_BYTES = 1024 * 1024 * 1024
MAX_MEMBERS = 100_000

ALLOWED_ROOT_FILES = {
    ".gitignore",
    "AGENTS.md",
    "LICENSE",
    "IMPLEMENTIERUNGSABGLEICH_2.80.4.md",
    "IMPLEMENTIERUNGSABGLEICH_2.81.md",
    "IMPLEMENTIERUNGSABGLEICH_2.82.md",
    "IMPLEMENTIERUNGSABGLEICH_2.80.1.md",
    "IMPLEMENTIERUNGSABGLEICH_2.80.3.md",
    "PRUEFBERICHT_2.80.1.md",
    "MANIFEST.in",
    "README.md",
    "V2772_HAILO_PARALLEL_CANARY_README.md",
    "analyse_and_split.py",
    "analyse_and_split_gui.py",
    "download_model_zoo_examples.py",
    "model_zoo_manifest.json",
    "pt_to_onnx.py",
    "pyproject.toml",
    "requirements.txt",
    "start_gui.sh",
    "uv.lock",
}
RETAINED_RELEASE_DOC_FILES = {
    "TESTANLEITUNG_2.81.md",
    "TESTANLEITUNG_2.82.md",
    "VERSION_2.82_BUILD_AND_TEST_REPORT.md",
    "VERSION_2.81_BUILD_AND_TEST_REPORT.md",
    "TESTANLEITUNG_2.80.3.md",
    "TESTANLEITUNG_2.80.4.md",
    "VERSION_2.80.4_BUILD_AND_TEST_REPORT.md",
    "VERSION_2.80.3_BUILD_AND_TEST_REPORT.md",
    "TESTANLEITUNG_2.80.1.md",
    "TESTANLEITUNG_2.80.2.md",
    "VERSION_2.80.2_BUILD_AND_TEST_REPORT.md",
    "VERSION_2.80.1_BUILD_AND_TEST_REPORT.md",
    "TESTANLEITUNG_2.80.md",
    "VERSION_2.80_BUILD_AND_TEST_REPORT.md",
    "TESTANLEITUNG_2.79.31.md",
    "VERSION_2.79.31_BUILD_AND_TEST_REPORT.md",
    "TESTANLEITUNG_2.79.32.md",
    "VERSION_2.79.32_BUILD_AND_TEST_REPORT.md",
    "TESTANLEITUNG_2.79.33.md",
    "VERSION_2.79.33_BUILD_AND_TEST_REPORT.md",
    "TESTANLEITUNG_2.79.34.md",
    "VERSION_2.79.34_BUILD_AND_TEST_REPORT.md",
    "TESTANLEITUNG_2.79.29.md",
    "VERSION_2.79.30_BUILD_AND_TEST_REPORT.md",
    "TESTANLEITUNG_2.79.30.md",
    "VERSION_2.79.29_BUILD_AND_TEST_REPORT.md",
    "TESTANLEITUNG_2.75.42.md",
    "VERSION_2.75.42_BUILD_AND_TEST_REPORT.md",
    "TESTANLEITUNG_2.75.43.md",
    "VERSION_2.75.43_BUILD_AND_TEST_REPORT.md",
    "TESTANLEITUNG_2.75.44.md",
    "VERSION_2.75.44_BUILD_AND_TEST_REPORT.md",
    "TESTANLEITUNG_2.75.45.md",
    "VERSION_2.75.45_BUILD_AND_TEST_REPORT.md",
    "TESTANLEITUNG_2.75.46.md",
    "VERSION_2.75.46_BUILD_AND_TEST_REPORT.md",
    "TESTANLEITUNG_2.75.47.md",
    "VERSION_2.75.47_BUILD_AND_TEST_REPORT.md",
    "TESTANLEITUNG_2.79.19.md",
    "VERSION_2.79.19_BUILD_AND_TEST_REPORT.md",
    "TESTANLEITUNG_2.79.20.md",
    "VERSION_2.79.20_BUILD_AND_TEST_REPORT.md",
    "TESTANLEITUNG_2.79.21.md",
    "VERSION_2.79.21_BUILD_AND_TEST_REPORT.md",
    "TESTANLEITUNG_2.79.22.md",
    "VERSION_2.79.22_BUILD_AND_TEST_REPORT.md",
    "TESTANLEITUNG_2.79.23.md",
    "VERSION_2.79.23_BUILD_AND_TEST_REPORT.md",
    "TESTANLEITUNG_2.79.24.md",
    "VERSION_2.79.24_BUILD_AND_TEST_REPORT.md",
    "TESTANLEITUNG_2.79.25.md",
    "VERSION_2.79.25_BUILD_AND_TEST_REPORT.md",
    "TESTANLEITUNG_2.79.26.md",
    "TESTANLEITUNG_2.79.28.md",
    "VERSION_2.79.28_BUILD_AND_TEST_REPORT.md",
    "TESTANLEITUNG_2.79.27.md",
    "VERSION_2.79.27_BUILD_AND_TEST_REPORT.md",
    "VERSION_2.79.26_BUILD_AND_TEST_REPORT.md",
}
ALLOWED_SOURCE_TREES = {"onnx_splitpoint_tool", "scripts", "tests"}
ALLOWED_DOC_FILES = {
    "docs/V2804_QUALITAET_UND_ABNAHME.md",
    "docs/RELEASE_SCOPE_V281.md",
    "docs/RELEASE_SCOPE_V282.md",
    "docs/V282_EVIDENCE_INDEX.json",
    "docs/ONNX_SPLITPOINT_KnowledgeBase_CANONICAL_v2.82_2026-09-13.md",
    "docs/RELEASE_2.83.md",
    "docs/RELEASE_2.90.0.md",
    "docs/RELEASE_2.90.1.md",
    "docs/ARBEITSSTAND.md",
    "docs/V281_EVIDENCE_INDEX.json",
    "docs/ONNX_SPLITPOINT_KnowledgeBase_CANONICAL_v2.81_2026-09-13.md",
    "docs/RELEASE_SCOPE_V2804.md",
    "docs/V2804_EVIDENCE_INDEX.json",
    "docs/ONNX_SPLITPOINT_KnowledgeBase_CANONICAL_v2.80.4_2026-09-12.md",
    "docs/RELEASE_SCOPE_V2802.md",
    "docs/RELEASE_SCOPE_V2803.md",
    "docs/V2803_DIAGNOSTIC_SCOPE.json",
    "docs/V2803_REFERENCE_AND_DIAGNOSTIC_ACCEPTANCE.md",
    "docs/ARTIFACTS_AND_POSTPROCESSING_CONTRACT.md",
    "docs/V27931_QUALITY_UNCERTAINTY_CONTRACT.md",
    "docs/HAILO10_YOLO26_DIAGNOSE_v27931.md",
    "docs/B500_EVIDENCE_RECONCILIATION_v2792.md",
    "docs/CLEAN_RELEASE.md",
    "docs/ENERGY_WINDOW_METHOD_VALIDATION.md",
    "docs/EVALUATION_PROFILES.md",
    "docs/EVALUATION_WORKFLOW.md",
    "docs/HOLDOUT_ADAPTER_PROTOCOL.md",
    "docs/MODEL_PREPARATION.md",
    "docs/NATIVE_BOUNDARY_INTERFACE_VALIDATOR.md",
    "docs/NATIVE_HAILORT_TRT_FIFO_FASTPATH.md",
    "docs/NATIVE_PRODUCER_EVALRUN_USAGE.md",
    "docs/NATIVE_THREE_STAGE.md",
    "docs/PLATFORM_POWER_CONTROL.md",
    "docs/QUICKSTART.md",
    "docs/README.md",
    "docs/VALIDATION_DATASETS.md",
    "docs/VERSIONING.md",
    "docs/V27933_FORCE_START.md",
    "docs/V27933_NACHTLAUF_UND_GPU.md",
}
EXCLUDED_DIRECTORY_NAMES = {
    ".cache",
    ".coverage",
    ".git",
    ".hypothesis",
    ".idea",
    ".mypy_cache",
    ".nox",
    ".pytest_cache",
    ".ruff_cache",
    ".temp",
    ".tmp",
    ".tox",
    ".venv",
    ".vscode",
    "__macosx",
    "__pycache__",
    "build",
    "console_smokes",
    "dist",
    "htmlcov",
    "temp",
    "tmp",
}
EXCLUDED_FILE_NAMES = {".git", MANIFEST_NAME, SUMS_NAME}
EXCLUDED_CASEFOLD_FILE_NAMES = {
    ".coverage",
    ".directory",
    ".ds_store",
    "coverage.json",
    "coverage.xml",
    "desktop.ini",
    "thumbs.db",
}
EXCLUDED_FILE_PREFIXES = (".#", "._", ".coverage.")
ALLOWED_BINARY_TEST_FIXTURES = {
    'tests/fixtures/v282_h10_generic/models/yolo26m/benchmark_results/remote_diagnostics/orin_nx_hailo10_01/logs/runner.log',
    'tests/fixtures/v282_h10_generic/models/yolo26s/benchmark_results/remote_diagnostics/orin_nx_hailo10_01/logs/runner.log',

    'tests/fixtures/v282_energy_retry_original/yolo26m/measurement/repeat_retry_attempts/repeat_000/attempt_01/run_000/workload_stdout.log',
    'tests/fixtures/v282_energy_retry_original/yolo26m/measurement/run_000/workload_stdout.log',
    'tests/fixtures/v282_energy_retry_original/yolo26m/measurement/run_001/workload_stdout.log',
    'tests/fixtures/v282_energy_retry_original/yolo26m/measurement/run_002/workload_stdout.log',
    'tests/fixtures/v282_energy_retry_original/yolo26s/measurement/repeat_retry_attempts/repeat_000/attempt_01/run_000/workload_stdout.log',
    'tests/fixtures/v282_energy_retry_original/yolo26s/measurement/run_000/workload_stdout.log',
    'tests/fixtures/v282_energy_retry_original/yolo26s/measurement/run_001/workload_stdout.log',
    'tests/fixtures/v282_energy_retry_original/yolo26s/measurement/run_002/workload_stdout.log',

    "tests/fixtures/v27930/semantic_probe_outputs.npz",
    "tests/fixtures/v281_hailo10_nwc/physical_outputs.npz",
    "tests/fixtures/v27931_classification/original_R1_R2_logits.npz",
    "tests/fixtures/v27933_force_origin/force_origin_audit_20260909T095730Z_ni3utrwy.zip",
}
EXCLUDED_FILE_SUFFIXES = {
    ".7z",
    ".bak",
    ".backup",
    ".crdownload",
    ".download",
    ".dxnn",
    ".engine",
    ".gz",
    ".har",
    ".harn",
    ".hef",
    ".log",
    ".npy",
    ".npz",
    ".onnx",
    ".orig",
    ".part",
    ".partial",
    ".parquet",
    ".pyc",
    ".pyo",
    ".rej",
    ".swo",
    ".swp",
    ".tar",
    ".temp",
    ".tgz",
    ".tmp",
    ".whl",
    ".zip",
}

# These roots are intentionally retained by rsync and are outside the release
# source inventory.  Their root entries must still be real directories.
PRESERVED_INSTALL_ROOTS = {
    ".venv",
    ".venv-report",
    ".git",
    ".agents",
    ".codex",
    "logs",
    ".install_logs",
    "EvaluationRuns",
    "RemoteBenchmarkRuns",
    "BenchmarkSets",
    "SplitNetworks",
    "Results",
    "EnergyMeasurements",
    "artifact_store",
    "build_evidence",
}


def fail(message: str) -> None:
    raise ValueError(message)


def strict_json(payload: bytes, *, label: str):
    def no_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                fail(f"{label} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    def no_constant(value: str):
        fail(f"{label} contains non-finite JSON value {value!r}")

    try:
        text = payload.decode("utf-8")
        return json.loads(
            text,
            object_pairs_hook=no_duplicates,
            parse_constant=no_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        fail(f"{label} is not strict UTF-8 JSON: {type(exc).__name__}")


def valid_logical_path(value: object) -> bool:
    if not isinstance(value, str) or not value or "\\" in value:
        return False
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        return False
    logical = PurePosixPath(value)
    return bool(
        not logical.is_absolute()
        and "." not in logical.parts
        and ".." not in logical.parts
        and logical.as_posix() == value
    )


def allowlisted_source_path(value: str) -> bool:
    if not valid_logical_path(value):
        return False
    relative = PurePosixPath(value)
    logical = relative.as_posix()
    if len(relative.parts) == 1:
        allowed = bool(
            logical in ALLOWED_ROOT_FILES
            or logical in RETAINED_RELEASE_DOC_FILES
            or logical
            in {
                f"TESTANLEITUNG_{EXPECTED_VERSION}.md",
                f"VERSION_{EXPECTED_VERSION}_BUILD_AND_TEST_REPORT.md",
            }
        )
    elif relative.parts[0] in ALLOWED_SOURCE_TREES:
        allowed = True
    elif relative.parts[0] == "profiles":
        allowed = bool(
            len(relative.parts) == 2
            and relative.suffix.casefold() in {".yaml", ".yml"}
        )
    elif logical in ALLOWED_DOC_FILES:
        allowed = True
    else:
        allowed = relative.parts[:2] == ("docs", "calibration")
    if not allowed or relative.name in EXCLUDED_FILE_NAMES:
        return False
    casefold_name = relative.name.casefold()
    if casefold_name in EXCLUDED_CASEFOLD_FILE_NAMES:
        return False
    if relative.name.endswith("~") or any(
        casefold_name.startswith(prefix) for prefix in EXCLUDED_FILE_PREFIXES
    ):
        return False
    if any(
        part.casefold() in EXCLUDED_DIRECTORY_NAMES
        or part.casefold().endswith(".egg-info")
        for part in relative.parts[:-1]
    ):
        return False
    return relative.suffix.casefold() not in EXCLUDED_FILE_SUFFIXES or logical in ALLOWED_BINARY_TEST_FIXTURES


def strict_sha256(value: object) -> str:
    if not isinstance(value, str) or len(value) != 64:
        return ""
    if any(character not in "0123456789abcdef" for character in value):
        return ""
    return value


def parse_manifest(payload: bytes) -> tuple[dict, list[dict]]:
    loaded = strict_json(payload, label=MANIFEST_NAME)
    if not isinstance(loaded, dict):
        fail(f"{MANIFEST_NAME} must contain one JSON object")
    required = {
        "schema",
        "package_version",
        "workflow_version",
        "file_count",
        "files",
    }
    if set(loaded) != required:
        fail(f"{MANIFEST_NAME} top-level shape is not exact")
    if loaded.get("schema") != MANIFEST_SCHEMA:
        fail(f"{MANIFEST_NAME} schema mismatch")
    if loaded.get("package_version") != EXPECTED_VERSION:
        fail(f"{MANIFEST_NAME} package version mismatch")
    if loaded.get("workflow_version") != EXPECTED_BUILD_ID:
        fail(f"{MANIFEST_NAME} workflow version mismatch")
    raw_rows = loaded.get("files")
    if not isinstance(raw_rows, list):
        fail(f"{MANIFEST_NAME} files must be a list")
    rows: list[dict] = []
    for index, raw in enumerate(raw_rows):
        if not isinstance(raw, dict) or set(raw) != {"path", "size", "sha256"}:
            fail(f"{MANIFEST_NAME} row {index} shape is invalid")
        path = raw.get("path")
        size = raw.get("size")
        digest = raw.get("sha256")
        if not isinstance(path, str) or not allowlisted_source_path(path):
            fail(f"{MANIFEST_NAME} row {index} path is unsafe or unallowlisted")
        if type(size) is not int or size < 0 or size > MAX_MEMBER_BYTES:
            fail(f"{MANIFEST_NAME} row {index} size is invalid")
        if not strict_sha256(digest):
            fail(f"{MANIFEST_NAME} row {index} SHA-256 is invalid")
        rows.append(dict(raw))
    paths = [row["path"] for row in rows]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        fail(f"{MANIFEST_NAME} paths must be sorted and unique")
    if type(loaded.get("file_count")) is not int or loaded["file_count"] != len(rows):
        fail(f"{MANIFEST_NAME} file_count mismatch")
    if not {
        "pyproject.toml",
        "onnx_splitpoint_tool/release_identity.py",
    }.issubset(paths):
        fail(f"{MANIFEST_NAME} omits required release identity files")
    total_size = sum(row["size"] for row in rows)
    if total_size > MAX_TOTAL_MEMBER_BYTES:
        fail(f"{MANIFEST_NAME} declared source inventory is too large")
    return loaded, rows


def canonical_sums(rows: list[dict]) -> bytes:
    return "".join(
        f"{row['sha256']}  {row['path']}\n" for row in rows
    ).encode("utf-8")


def open_regular(path: Path):
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            fail(f"not a regular file: {path}")
        return descriptor, metadata
    except BaseException:
        os.close(descriptor)
        raise


def read_regular(path: Path, *, maximum: int) -> tuple[bytes, str]:
    descriptor, initial = open_regular(path)
    try:
        chunks: list[bytes] = []
        observed = 0
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            observed += len(chunk)
            if observed > maximum:
                fail(f"regular file exceeds size limit: {path}")
            chunks.append(chunk)
            digest.update(chunk)
        final = os.fstat(descriptor)
        if (
            observed != initial.st_size
            or final.st_size != initial.st_size
            or final.st_mtime_ns != initial.st_mtime_ns
            or final.st_ctime_ns != initial.st_ctime_ns
        ):
            fail(f"regular file changed while it was read: {path}")
        return b"".join(chunks), digest.hexdigest()
    finally:
        os.close(descriptor)


def hash_regular(path: Path, *, expected_size: int | None = None) -> tuple[int, str]:
    descriptor, initial = open_regular(path)
    try:
        if expected_size is not None and initial.st_size != expected_size:
            fail(f"source file size mismatch: {path}")
        if initial.st_size > MAX_MEMBER_BYTES:
            fail(f"source file exceeds size limit: {path}")
        digest = hashlib.sha256()
        observed = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            observed += len(chunk)
            if observed > MAX_MEMBER_BYTES:
                fail(f"source file exceeds size limit: {path}")
            digest.update(chunk)
        final = os.fstat(descriptor)
        if (
            observed != initial.st_size
            or final.st_size != initial.st_size
            or final.st_mtime_ns != initial.st_mtime_ns
            or final.st_ctime_ns != initial.st_ctime_ns
        ):
            fail(f"source file changed while it was hashed: {path}")
        return observed, digest.hexdigest()
    finally:
        os.close(descriptor)


def copy_regular(source: Path, destination: Path) -> str:
    source_descriptor, initial = open_regular(source)
    if initial.st_size > MAX_ARCHIVE_BYTES:
        os.close(source_descriptor)
        fail("source archive exceeds the trusted preflight size limit")
    destination_flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
    )
    destination_descriptor = os.open(destination, destination_flags, 0o600)
    digest = hashlib.sha256()
    observed = 0
    try:
        while True:
            chunk = os.read(source_descriptor, 1024 * 1024)
            if not chunk:
                break
            observed += len(chunk)
            if observed > MAX_ARCHIVE_BYTES:
                fail("source archive exceeds the trusted preflight size limit")
            digest.update(chunk)
            view = memoryview(chunk)
            while view:
                written = os.write(destination_descriptor, view)
                if written <= 0:
                    fail("short write while staging source archive")
                view = view[written:]
        os.fsync(destination_descriptor)
        final = os.fstat(source_descriptor)
        if (
            observed != initial.st_size
            or final.st_size != initial.st_size
            or final.st_mtime_ns != initial.st_mtime_ns
            or final.st_ctime_ns != initial.st_ctime_ns
        ):
            fail("source archive changed while it was staged")
    finally:
        os.close(destination_descriptor)
        os.close(source_descriptor)
    return digest.hexdigest()


def read_zip_member(archive: zipfile.ZipFile, info: zipfile.ZipInfo, *, maximum: int) -> bytes:
    if info.file_size > maximum:
        fail(f"archive member exceeds size limit: {info.filename!r}")
    chunks: list[bytes] = []
    observed = 0
    with archive.open(info, "r") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            observed += len(chunk)
            if observed > maximum:
                fail(f"archive member exceeds size limit: {info.filename!r}")
            chunks.append(chunk)
    if observed != info.file_size:
        fail(f"archive member size changed while reading: {info.filename!r}")
    return b"".join(chunks)


def hash_zip_member(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    *,
    expected_size: int,
) -> str:
    if info.file_size != expected_size or expected_size > MAX_MEMBER_BYTES:
        fail(f"archive member size mismatch: {info.filename!r}")
    digest = hashlib.sha256()
    observed = 0
    with archive.open(info, "r") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            observed += len(chunk)
            if observed > expected_size:
                fail(f"archive member expands beyond its manifest size: {info.filename!r}")
            digest.update(chunk)
    if observed != expected_size:
        fail(f"archive member size mismatch: {info.filename!r}")
    return digest.hexdigest()


def validate_zip_info(info: zipfile.ZipInfo) -> None:
    name = info.filename
    if not valid_logical_path(name) or info.is_dir():
        fail(f"unsafe archive member name: {name!r}")
    mode = int(info.external_attr) >> 16
    if info.create_system != 3 or not stat.S_ISREG(mode):
        fail(f"archive member is not a Unix regular file: {name!r}")
    if mode & ~0o100777 or stat.S_IMODE(mode) not in {0o644, 0o755}:
        fail(f"archive member mode is unsafe: {name!r}")
    if info.flag_bits & 0x1:
        fail(f"encrypted archive member is not permitted: {name!r}")
    if (
        info.date_time != FIXED_ZIP_TIMESTAMP
        or info.compress_type != zipfile.ZIP_DEFLATED
        or info.extra != b""
        or info.comment != b""
    ):
        fail(f"archive member metadata is not deterministic: {name!r}")


def verify_archive(source: Path, snapshot: Path) -> None:
    archive_sha256 = copy_regular(source, snapshot)
    try:
        with zipfile.ZipFile(snapshot, "r") as archive:
            if archive.comment != b"":
                fail("source archive comment must be empty")
            infos = archive.infolist()
            if not infos or len(infos) > MAX_MEMBERS:
                fail("source archive member count is invalid")
            names = [info.filename for info in infos]
            if len(names) != len(set(names)):
                fail("source archive contains duplicate member names")
            for info in infos:
                validate_zip_info(info)
            manifest_names = [
                name
                for name in names
                if PurePosixPath(name).name == MANIFEST_NAME
            ]
            if len(manifest_names) != 1:
                fail("source archive must contain exactly one manifest index")
            manifest_logical = PurePosixPath(manifest_names[0])
            if len(manifest_logical.parts) != 2:
                fail("source archive manifest must be below one top-level prefix")
            prefix = manifest_logical.parts[0]
            if prefix != EXPECTED_PREFIX:
                fail("source archive top-level prefix mismatch")
            info_by_name = {info.filename: info for info in infos}
            manifest_info = info_by_name[f"{prefix}/{MANIFEST_NAME}"]
            manifest_bytes = read_zip_member(
                archive,
                manifest_info,
                maximum=MAX_MANIFEST_BYTES,
            )
            _manifest, rows = parse_manifest(manifest_bytes)
            expected_names = sorted(
                [f"{prefix}/{row['path']}" for row in rows]
                + [f"{prefix}/{name}" for name in INDEX_NAMES]
            )
            if names != expected_names:
                fail("source archive order/member set differs from its strict manifest")
            sums_info = info_by_name[f"{prefix}/{SUMS_NAME}"]
            sums_bytes = read_zip_member(
                archive,
                sums_info,
                maximum=MAX_SUMS_BYTES,
            )
            if sums_bytes != canonical_sums(rows):
                fail(f"{SUMS_NAME} is not the exact manifest projection")
            for row in rows:
                member = f"{prefix}/{row['path']}"
                observed = hash_zip_member(
                    archive,
                    info_by_name[member],
                    expected_size=row["size"],
                )
                if observed != row["sha256"]:
                    fail(f"archive source hash mismatch: {row['path']!r}")
    except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
        fail(f"source archive is unreadable or corrupt: {type(exc).__name__}")
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    print(f"{prefix}\t{manifest_sha256}\t{archive_sha256}")


def expected_directories(files: set[str]) -> set[str]:
    result: set[str] = set()
    for value in files:
        parent = PurePosixPath(value).parent
        while parent.parts:
            result.add(parent.as_posix())
            parent = parent.parent
    return result


def scan_tree(root: Path, *, installed: bool) -> tuple[set[str], set[str]]:
    files: set[str] = set()
    directories: set[str] = set()

    def visit(directory: Path, logical_parent: PurePosixPath | None) -> None:
        with os.scandir(directory) as entries:
            ordered = sorted(entries, key=lambda entry: entry.name)
        for entry in ordered:
            logical = (
                PurePosixPath(entry.name)
                if logical_parent is None
                else logical_parent / entry.name
            )
            value = logical.as_posix()
            if not valid_logical_path(value):
                fail(f"installed source contains an unsafe path: {value!r}")
            metadata = entry.stat(follow_symlinks=False)
            if stat.S_ISLNK(metadata.st_mode):
                fail(f"installed source contains a symlink: {value!r}")
            if (
                installed
                and logical_parent is None
                and entry.name in PRESERVED_INSTALL_ROOTS
            ):
                if not stat.S_ISDIR(metadata.st_mode):
                    fail(f"preserved install root is not a directory: {value!r}")
                continue
            if stat.S_ISDIR(metadata.st_mode):
                directories.add(value)
                visit(Path(entry.path), logical)
            elif stat.S_ISREG(metadata.st_mode):
                files.add(value)
            else:
                fail(f"installed source contains a non-regular entry: {value!r}")

    visit(root, None)
    return files, directories


def verify_tree(
    root: Path,
    snapshot: Path,
    expected_manifest_sha256: str,
    expected_archive_sha256: str,
    *,
    installed: bool,
) -> None:
    root_metadata = os.lstat(root)
    if not stat.S_ISDIR(root_metadata.st_mode) or stat.S_ISLNK(root_metadata.st_mode):
        fail("source tree root must be a real directory")
    archive_size, archive_sha256 = hash_regular(snapshot)
    if archive_size <= 0 or archive_sha256 != expected_archive_sha256:
        fail("private source archive snapshot changed after admission")
    manifest_bytes, manifest_sha256 = read_regular(
        root / MANIFEST_NAME,
        maximum=MAX_MANIFEST_BYTES,
    )
    if manifest_sha256 != expected_manifest_sha256:
        fail("extracted source manifest differs from admitted archive")
    _manifest, rows = parse_manifest(manifest_bytes)
    sums_bytes, _sums_sha256 = read_regular(
        root / SUMS_NAME,
        maximum=MAX_SUMS_BYTES,
    )
    if sums_bytes != canonical_sums(rows):
        fail(f"{SUMS_NAME} is not the exact manifest projection")
    expected_files = {row["path"] for row in rows} | set(INDEX_NAMES)
    actual_files, actual_directories = scan_tree(root, installed=installed)
    allowed_user_profiles: set[str] = set()
    if installed:
        # rsync preserves user-owned, top-level profile YAML files.  They are
        # the only unmanifested source-tree entries allowed in installed mode;
        # arbitrary files and profile subdirectories remain fatal.
        for value in actual_files - expected_files:
            logical = PurePosixPath(value)
            if (
                len(logical.parts) == 2
                and logical.parts[0] == "profiles"
                and logical.suffix.casefold() in {".yaml", ".yml"}
            ):
                allowed_user_profiles.add(value)
        actual_files -= allowed_user_profiles
    if actual_files != expected_files:
        missing = sorted(expected_files - actual_files)[:10]
        extra = sorted(actual_files - expected_files)[:10]
        fail(f"source tree file set mismatch; missing={missing!r}, extra={extra!r}")
    required_directories = expected_directories(expected_files)
    allowed_user_directories = (
        {"profiles"}
        if allowed_user_profiles and "profiles" not in required_directories
        else set()
    )
    admitted_directories = required_directories | allowed_user_directories
    if actual_directories != admitted_directories:
        missing = sorted(admitted_directories - actual_directories)[:10]
        extra = sorted(actual_directories - admitted_directories)[:10]
        fail(f"source tree directory set mismatch; missing={missing!r}, extra={extra!r}")
    for row in rows:
        size, digest = hash_regular(root / row["path"], expected_size=row["size"])
        if size != row["size"] or digest != row["sha256"]:
            fail(f"source tree content mismatch: {row['path']!r}")
    print("verified")


def main() -> None:
    if len(sys.argv) < 2:
        fail("trusted source preflight mode is missing")
    mode = sys.argv[1]
    if mode == "archive":
        if len(sys.argv) != 4:
            fail("trusted archive preflight arguments are invalid")
        verify_archive(Path(sys.argv[2]), Path(sys.argv[3]))
        return
    if mode in {"tree", "installed-tree"}:
        if len(sys.argv) != 6:
            fail("trusted tree preflight arguments are invalid")
        verify_tree(
            Path(sys.argv[2]),
            Path(sys.argv[3]),
            sys.argv[4],
            sys.argv[5],
            installed=mode == "installed-tree",
        )
        return
    fail(f"unsupported trusted source preflight mode: {mode!r}")


try:
    main()
except (OSError, ValueError, zipfile.BadZipFile) as exc:
    print(
        f"FEHLER: vertrauenswürdige Release-Prüfung fehlgeschlagen: {exc}",
        file=sys.stderr,
    )
    raise SystemExit(65)
PY
}

restore_preserved_profiles() {
  if [[ "$PRESERVED_PROFILES_RESTORED" -eq 1 || \
        "$PRESERVED_PROFILE_COUNT" -eq 0 ]]; then
    return 0
  fi
  if ! mkdir -p "$TOOL_DIR/profiles"; then
    return 1
  fi
  if ! cp -a -- "$PRESERVED_PROFILE_DIR/." "$TOOL_DIR/profiles/"; then
    return 1
  fi
  PRESERVED_PROFILES_RESTORED=1
}

cleanup() {
  local exit_status=$?
  local restore_status=0
  trap - EXIT
  set +e
  restore_preserved_profiles
  restore_status=$?
  if [[ -d "$STAGE_DIR" ]]; then
    rm -r -- "$STAGE_DIR"
  fi
  case "${PYCACHE_ROOT:-}" in
    /tmp/onnx-splitpoint-update-pycache.??????????)
      if [[ -L "$PYCACHE_ROOT" ]]; then
        /bin/rm -- "$PYCACHE_ROOT" || exit_status=70
      elif [[ -d "$PYCACHE_ROOT" ]]; then
        /bin/rm -r -- "$PYCACHE_ROOT" || exit_status=70
      elif [[ -e "$PYCACHE_ROOT" ]]; then
        exit_status=70
      fi
      ;;
    *) exit_status=70 ;;
  esac
  if [[ "$restore_status" -ne 0 ]]; then
    printf 'FEHLER: eigene Profile konnten nicht wiederhergestellt werden.\n' \
      >&2
    if [[ "$exit_status" -eq 0 ]]; then
      exit_status=$restore_status
    fi
  fi
  exit "$exit_status"
}
trap cleanup EXIT

BACKUP_ROOT="$HOME/ONNX-Splitpoint-Tool_update_backups"
BACKUP_DIR="$BACKUP_ROOT/$(date +%Y%m%d_%H%M%S)_$$_$(date +%N)"
BACKUP_ROOT="$(readlink -m -- "$BACKUP_ROOT")"
BACKUP_DIR="$(readlink -m -- "$BACKUP_DIR")"
case "$BACKUP_DIR/" in
  "$TOOL_DIR/"*)
    printf 'FEHLER: Backup muss außerhalb des Tool-Ordners liegen: %s\n' \
      "$BACKUP_DIR" >&2
    exit 64
    ;;
esac

if ! TRUSTED_ARCHIVE_REPORT="$(trusted_source_preflight archive "$ZIP" "$STAGED_ZIP" 2>&1)"; then
  printf '%s\n' "$TRUSTED_ARCHIVE_REPORT" >&2
  exit 65
fi
if [[ "$TRUSTED_ARCHIVE_REPORT" == *$'\n'* ]]; then
  printf 'FEHLER: ungültige Ausgabe der vertrauenswürdigen Archivprüfung.\n' >&2
  exit 65
fi
IFS=$'\t' read -r -a TRUSTED_ARCHIVE_FIELDS <<<"$TRUSTED_ARCHIVE_REPORT"
if [[ ${#TRUSTED_ARCHIVE_FIELDS[@]} -ne 3 || \
      "${TRUSTED_ARCHIVE_FIELDS[0]}" != 'ONNX-Splitpoint-Tool_v2.90.1' || \
      ! "${TRUSTED_ARCHIVE_FIELDS[1]}" =~ ^[0-9a-f]{64}$ || \
      ! "${TRUSTED_ARCHIVE_FIELDS[2]}" =~ ^[0-9a-f]{64}$ ]]; then
  printf 'FEHLER: ungültige Ausgabe der vertrauenswürdigen Archivprüfung.\n' >&2
  exit 65
fi
SOURCE_PREFIX="${TRUSTED_ARCHIVE_FIELDS[0]}"
TRUSTED_MANIFEST_SHA256="${TRUSTED_ARCHIVE_FIELDS[1]}"
TRUSTED_ARCHIVE_SHA256="${TRUSTED_ARCHIVE_FIELDS[2]}"

# From here on every consumer uses the private, verified snapshot.  This also
# closes replacement races against the caller-controlled archive path.
ZIP="$STAGED_ZIP"
unzip -q "$ZIP" -d "$STAGE_DIR"
SOURCE_DIR="$STAGE_DIR/$SOURCE_PREFIX"

# unzip is treated only as an extractor.  Its complete output is independently
# checked by the embedded verifier before any candidate-source Python runs.
trusted_source_preflight tree \
  "$SOURCE_DIR" "$ZIP" "$TRUSTED_MANIFEST_SHA256" "$TRUSTED_ARCHIVE_SHA256" \
  >/dev/null

# Top-level profile YAMLs that are not part of the incoming release are user
# input, not obsolete source.  Keep a temporary safety copy so the strict
# --delete synchronization and the subsequent manifest verification can still
# run against the exact release tree.  The EXIT trap restores these files even
# if synchronization or verification fails.
mkdir -p "$PRESERVED_PROFILE_DIR"
if [[ -d "$TOOL_DIR/profiles" ]]; then
  while IFS= read -r -d '' profile_path; do
    profile_name="${profile_path##*/}"
    if [[ ! -e "$SOURCE_DIR/profiles/$profile_name" ]]; then
      cp -a -- "$profile_path" "$PRESERVED_PROFILE_DIR/$profile_name"
      PRESERVED_PROFILE_COUNT=$((PRESERVED_PROFILE_COUNT + 1))
    fi
  done < <(
    find "$TOOL_DIR/profiles" \
      -mindepth 1 -maxdepth 1 -type f \
      \( -iname '*.yaml' -o -iname '*.yml' \) \
      -print0
  )
fi

PYTHONPATH= PYTHONHOME= \
  "$PYTHON" -I -S -B "$SOURCE_DIR/scripts/build_source_manifest.py" \
  --root "$SOURCE_DIR" --verify >/dev/null
PYTHONPATH= PYTHONHOME= \
  "$PYTHON" -I -S -B "$SOURCE_DIR/scripts/build_source_release.py" \
  --root "$SOURCE_DIR" --verify-archive "$ZIP" >/dev/null

mkdir -p "$BACKUP_DIR"
rsync -a \
  --checksum \
  --delete \
  --backup \
  --backup-dir="$BACKUP_DIR" \
  --exclude='/.venv/' \
  --exclude='/.venv-report/' \
  --exclude='/.git/' \
  --exclude='/.agents/' \
  --exclude='/.codex/' \
  --exclude='/logs/' \
  --exclude='/.install_logs/' \
  --exclude='/EvaluationRuns/' \
  --exclude='/RemoteBenchmarkRuns/' \
  --exclude='/BenchmarkSets/' \
  --exclude='/SplitNetworks/' \
  --exclude='/Results/' \
  --exclude='/EnergyMeasurements/' \
  --exclude='/artifact_store/' \
  --exclude='/build_evidence/' \
  "$SOURCE_DIR/" \
  "$TOOL_DIR/"

cd -- "$TOOL_DIR"

verify_installed_release() {
  local attempt="$1"
  local report="$STAGE_DIR/installed-release-verification-${attempt}.log"
  if ! trusted_source_preflight installed-tree \
      "$TOOL_DIR" "$ZIP" \
      "$TRUSTED_MANIFEST_SHA256" "$TRUSTED_ARCHIVE_SHA256" \
      >"$report" 2>&1
  then
    cat -- "$report" >&2
    return 1
  fi
  if PYTHONPATH= PYTHONHOME= \
    "$PYTHON" -I -S -B scripts/build_source_manifest.py \
      --root "$TOOL_DIR" --verify --scope installed \
      >>"$report" 2>&1
  then
    return 0
  fi
  cat -- "$report" >&2
  return 1
}

converge_installed_release() {
  # A bounded retry never overwrites between-pass bytes silently: anything
  # changed after the authoritative first backup is retained separately.
  mkdir -p "$BACKUP_DIR/convergence"
  rsync -a \
    --checksum \
    --delete \
    --delay-updates \
    --backup \
    --backup-dir="$BACKUP_DIR/convergence" \
    --exclude='/.venv/' \
    --exclude='/.venv-report/' \
    --exclude='/.git/' \
    --exclude='/.agents/' \
    --exclude='/.codex/' \
    --exclude='/logs/' \
    --exclude='/.install_logs/' \
    --exclude='/EvaluationRuns/' \
    --exclude='/RemoteBenchmarkRuns/' \
    --exclude='/BenchmarkSets/' \
    --exclude='/SplitNetworks/' \
    --exclude='/Results/' \
    --exclude='/EnergyMeasurements/' \
    --exclude='/artifact_store/' \
    --exclude='/build_evidence/' \
    "$SOURCE_DIR/" \
    "$TOOL_DIR/"
}

installed_source_stable() {
  local attempt="$1"
  local report="$STAGE_DIR/installed-source-stability-${attempt}.rsync"
  local itemized_change
  local drift=0
  if ! rsync -a \
      --checksum \
      --delete \
      --dry-run \
      --itemize-changes \
      --out-format='%i|%n%L' \
      --exclude='/.venv/' \
      --exclude='/.venv-report/' \
      --exclude='/.git/' \
      --exclude='/.agents/' \
      --exclude='/.codex/' \
      --exclude='/logs/' \
      --exclude='/.install_logs/' \
      --exclude='/EvaluationRuns/' \
      --exclude='/RemoteBenchmarkRuns/' \
      --exclude='/BenchmarkSets/' \
      --exclude='/SplitNetworks/' \
      --exclude='/Results/' \
      --exclude='/EnergyMeasurements/' \
      --exclude='/artifact_store/' \
      --exclude='/build_evidence/' \
      "$SOURCE_DIR/" \
      "$TOOL_DIR/" >"$report"
  then
    printf '%s\n' \
      'FEHLER: read-only Quelldriftprüfung konnte nicht vollständig ausgeführt werden.' >&2
    return 2
  fi
  while IFS= read -r itemized_change; do
    case "$itemized_change" in
      '')
        ;;
      '.d..t......|'*)
        # Directory mtimes do not affect the manifest or regular-file payload.
        ;;
      *)
        printf 'FEHLER: Quelldrift nach strikter Verifikation: %s\n' \
          "$itemized_change" >&2
        drift=1
        ;;
    esac
  done <"$report"
  return "$drift"
}

# The archive was already verified strictly against the extracted release.
# Here the trusted archive/tree comparison and Installed-scope manifest check
# verify release-owned bytes while admitting only the updater's exact retained
# operational roots. If either check or the following read-only no-op
# comparison sees drift, exactly one separately backed retry is permitted.
CONVERGENCE_USED=0
if ! verify_installed_release initial; then
  printf '%s\n' \
    'WARNUNG: erster installierter Release-Check driftete; starte einen gesicherten Konvergenzversuch.' >&2
  converge_installed_release
  CONVERGENCE_USED=1
  verify_installed_release convergence || {
    printf '%s\n' 'FEHLER: installierter Release-Stand konvergiert nicht.' >&2
    exit 65
  }
fi

STABILITY_RC=0
installed_source_stable initial || STABILITY_RC=$?
if [[ "$STABILITY_RC" -eq 2 ]]; then
  exit 69
fi
if [[ "$STABILITY_RC" -ne 0 ]]; then
  if [[ "$CONVERGENCE_USED" -ne 0 ]]; then
    printf '%s\n' 'FEHLER: erneute Quelldrift nach dem einzigen Konvergenzversuch.' >&2
    exit 65
  fi
  printf '%s\n' \
    'WARNUNG: Quelldrift nach Release-Check; starte einen gesicherten Konvergenzversuch.' >&2
  converge_installed_release
  CONVERGENCE_USED=1
  verify_installed_release convergence || {
    printf '%s\n' 'FEHLER: installierter Release-Stand konvergiert nicht.' >&2
    exit 65
  }
  STABILITY_RC=0
  installed_source_stable convergence || STABILITY_RC=$?
  if [[ "$STABILITY_RC" -eq 2 ]]; then
    exit 69
  fi
  if [[ "$STABILITY_RC" -ne 0 ]]; then
    printf '%s\n' 'FEHLER: installierter Release-Stand blieb nach Konvergenz instabil.' >&2
    exit 65
  fi
fi

# rsync deliberately preserves the existing venv, including its dependencies,
# metadata and console scripts.  Refresh only this project through the audited
# stdlib-only PEP 376 installer.  It neither imports pip/setuptools nor has a
# package-index/network fallback, and it rolls back its own artifacts on any
# install or fresh-process validation failure.
PYTHONPATH= PYTHONHOME= "$PYTHON" -I -B scripts/refresh_editable_install.py \
  --root "$TOOL_DIR" \
  --expected-version 2.90.1 \
  --require-entrypoint \
    'onnx-splitpoint-smoke-v279=onnx_splitpoint_tool.v279_smoke:main' \
  --require-entrypoint \
    'onnx-splitpoint-smoke-v2791=onnx_splitpoint_tool.v279_smoke:main' \
  --require-entrypoint \
    'onnx-splitpoint-smoke-v2-79-1=onnx_splitpoint_tool.v279_smoke:main' \
  --require-entrypoint \
    'onnx-splitpoint-smoke-v2792=onnx_splitpoint_tool.v2792_smoke:main' \
  --require-entrypoint \
    'onnx-splitpoint-smoke-v2-79-2=onnx_splitpoint_tool.v2792_smoke:main' \
  --require-entrypoint \
    'onnx-splitpoint-smoke-v2793=onnx_splitpoint_tool.v2793_smoke:main' \
  --require-entrypoint \
    'onnx-splitpoint-smoke-v2-79-3=onnx_splitpoint_tool.v2793_smoke:main' \
  --require-entrypoint \
    'onnx-splitpoint-smoke-v2794=onnx_splitpoint_tool.v2794_smoke:main' \
  --require-entrypoint \
    'onnx-splitpoint-smoke-v2-79-4=onnx_splitpoint_tool.v2794_smoke:main' \
  --require-entrypoint \
    'onnx-splitpoint-smoke-v2795=onnx_splitpoint_tool.v2795_smoke:main' \
  --require-entrypoint \
    'onnx-splitpoint-smoke-v2-79-5=onnx_splitpoint_tool.v2795_smoke:main' \
  --require-entrypoint \
    'onnx-splitpoint-smoke-v2796=onnx_splitpoint_tool.v2796_smoke:main' \
  --require-entrypoint \
    'onnx-splitpoint-smoke-v2-79-6=onnx_splitpoint_tool.v2796_smoke:main' \
  --require-entrypoint \
    'onnx-splitpoint-smoke-v2797=onnx_splitpoint_tool.v2797_smoke:main' \
  --require-entrypoint \
    'onnx-splitpoint-smoke-v2-79-7=onnx_splitpoint_tool.v2797_smoke:main' \
  --require-entrypoint \
    'onnx-splitpoint-smoke-v2798=onnx_splitpoint_tool.v2798_smoke:main' \
  --require-entrypoint \
    'onnx-splitpoint-smoke-v2-79-8=onnx_splitpoint_tool.v2798_smoke:main' \
  --require-entrypoint \
    'onnx-splitpoint-smoke-v2799=onnx_splitpoint_tool.v2799_smoke:main' \
  --require-entrypoint \
    'onnx-splitpoint-smoke-v2-79-9=onnx_splitpoint_tool.v2799_smoke:main' \
  --require-entrypoint \
    'onnx-splitpoint-smoke-v27910=onnx_splitpoint_tool.v27910_smoke:main' \
  --require-entrypoint \
    'onnx-splitpoint-smoke-v2-79-10=onnx_splitpoint_tool.v27910_smoke:main' \
  --require-entrypoint \
    'onnx-splitpoint-smoke-v27911=onnx_splitpoint_tool.v27911_smoke:main' \
  --require-entrypoint \
    'onnx-splitpoint-smoke-v2-79-11=onnx_splitpoint_tool.v27911_smoke:main' \
  --require-entrypoint \
    'onnx-splitpoint-smoke-v27912=onnx_splitpoint_tool.v27912_smoke:main' \
  --require-entrypoint \
    'onnx-splitpoint-smoke-v2-79-12=onnx_splitpoint_tool.v27912_smoke:main' \
  --require-entrypoint \
    'onnx-splitpoint-smoke-v27913=onnx_splitpoint_tool.v27913_smoke:main' \
  --require-entrypoint \
    'onnx-splitpoint-smoke-v2-79-13=onnx_splitpoint_tool.v27913_smoke:main' \
  --require-entrypoint \
    'onnx-splitpoint-smoke-v27914=onnx_splitpoint_tool.v27914_smoke:main' \
  --require-entrypoint \
    'onnx-splitpoint-smoke-v2-79-14=onnx_splitpoint_tool.v27914_smoke:main' \
  --require-entrypoint \
    'onnx-splitpoint-smoke-v27915=onnx_splitpoint_tool.v27915_smoke:main' \
  --require-entrypoint \
    'onnx-splitpoint-smoke-v2-79-15=onnx_splitpoint_tool.v27915_smoke:main' \
  --require-entrypoint \
    'onnx-splitpoint-smoke-v27916=onnx_splitpoint_tool.v27916_smoke:main' \
  --require-entrypoint \
    'onnx-splitpoint-smoke-v2-79-16=onnx_splitpoint_tool.v27916_smoke:main' \
  --require-entrypoint \
    'onnx-splitpoint-smoke-v27917=onnx_splitpoint_tool.v27917_smoke:main' \
  --require-entrypoint \
    'onnx-splitpoint-smoke-v2-79-17=onnx_splitpoint_tool.v27917_smoke:main' \
  --require-entrypoint \
    'onnx-splitpoint-smoke-v27918=onnx_splitpoint_tool.v27918_smoke:main' \
  --require-entrypoint \
    'onnx-splitpoint-smoke-v2-79-18=onnx_splitpoint_tool.v27918_smoke:main' \
  --require-entrypoint \
    'onnx-splitpoint-smoke-v27919=onnx_splitpoint_tool.v27919_smoke:main' \
  --require-entrypoint \
    'onnx-splitpoint-smoke-v2-79-19=onnx_splitpoint_tool.v27919_smoke:main' \
  --require-entrypoint \
    'onnx-splitpoint-smoke-v27920=onnx_splitpoint_tool.v27920_smoke:main' \
  --require-entrypoint \
    'onnx-splitpoint-smoke-v2-79-20=onnx_splitpoint_tool.v27920_smoke:main' \
  --require-entrypoint \
    'onnx-splitpoint-smoke-v27921=onnx_splitpoint_tool.v27921_smoke:main' \
  --require-entrypoint \
    'onnx-splitpoint-smoke-v2-79-21=onnx_splitpoint_tool.v27921_smoke:main' \
  --require-entrypoint \
    'onnx-splitpoint-smoke-v27922=onnx_splitpoint_tool.v27922_smoke:main' \
  --require-entrypoint \
    'onnx-splitpoint-smoke-v2-79-22=onnx_splitpoint_tool.v27922_smoke:main' \
  --require-entrypoint \
    'onnx-splitpoint-smoke-v27923=onnx_splitpoint_tool.v27923_smoke:main' \
  --require-entrypoint \
    'onnx-splitpoint-smoke-v2-79-23=onnx_splitpoint_tool.v27923_smoke:main' \
  --require-entrypoint \
    'onnx-splitpoint-smoke-v27924=onnx_splitpoint_tool.v27924_smoke:main' \
  --require-entrypoint \
    'onnx-splitpoint-smoke-v2-79-24=onnx_splitpoint_tool.v27924_smoke:main' \
  --require-entrypoint \
    'onnx-splitpoint-smoke-v27925=onnx_splitpoint_tool.v27925_smoke:main' \
  --require-entrypoint \
    'onnx-splitpoint-smoke-v2-79-25=onnx_splitpoint_tool.v27925_smoke:main' \
  --require-entrypoint \
    'onnx-splitpoint-smoke-v27926=onnx_splitpoint_tool.v27926_smoke:main' \
  --require-entrypoint \
    'onnx-splitpoint-smoke-v2-79-26=onnx_splitpoint_tool.v27926_smoke:main' \
  --require-entrypoint \
    'onnx-splitpoint-smoke-v27927=onnx_splitpoint_tool.v27927_smoke:main' \
  --require-entrypoint \
    'onnx-splitpoint-smoke-v2-79-27=onnx_splitpoint_tool.v27927_smoke:main' \
  --require-entrypoint \
    'onnx-splitpoint-smoke-v27928=onnx_splitpoint_tool.v27928_smoke:main' \
  --require-entrypoint \
    'onnx-splitpoint-smoke-v2-79-28=onnx_splitpoint_tool.v27928_smoke:main' \
  --require-entrypoint \
    'onnx-splitpoint-smoke-v27929=onnx_splitpoint_tool.v27929_smoke:main' \
  --require-entrypoint \
    'onnx-splitpoint-smoke-v2-79-29=onnx_splitpoint_tool.v27929_smoke:main' \
  --require-entrypoint \
    'onnx-splitpoint-smoke-v27932=onnx_splitpoint_tool.v27932_smoke:main' \
  --require-entrypoint \
    'onnx-splitpoint-smoke-v2-79-32=onnx_splitpoint_tool.v27932_smoke:main' \
  --require-entrypoint \
    'onnx-splitpoint-smoke-v27933=onnx_splitpoint_tool.v27933_smoke:main' \
  --require-entrypoint \
    'onnx-splitpoint-smoke-v2-79-33=onnx_splitpoint_tool.v27933_smoke:main' \
  --require-entrypoint \
    'onnx-splitpoint-smoke-v27934=onnx_splitpoint_tool.v27934_smoke:main' \
  --require-entrypoint \
    'onnx-splitpoint-smoke-v2-79-34=onnx_splitpoint_tool.v27934_smoke:main' \
  --require-entrypoint \
    'onnx-splitpoint-smoke-v282=onnx_splitpoint_tool.v282_smoke:main' \
  --require-entrypoint \
    'onnx-splitpoint-smoke-v2-82=onnx_splitpoint_tool.v282_smoke:main' \
  --require-entrypoint \
    'onnx-splitpoint-smoke-v281=onnx_splitpoint_tool.v281_smoke:main' \
  --require-entrypoint \
    'onnx-splitpoint-smoke-v2-81=onnx_splitpoint_tool.v281_smoke:main' \
  --require-entrypoint \
    'onnx-splitpoint-smoke-v2804=onnx_splitpoint_tool.v2804_smoke:main' \
  --require-entrypoint \
    'onnx-splitpoint-smoke-v2803=onnx_splitpoint_tool.v2803_smoke:main' \
  --require-entrypoint \
    'onnx-splitpoint-smoke-v2802=onnx_splitpoint_tool.v2802_smoke:main' \
  --require-entrypoint \
    'onnx-splitpoint-smoke-v2-80-2=onnx_splitpoint_tool.v2802_smoke:main' \
  --require-entrypoint \
    'onnx-splitpoint-smoke-v2-80-4=onnx_splitpoint_tool.v2804_smoke:main' \
  --require-entrypoint \
    'onnx-splitpoint-smoke-v2-80-3=onnx_splitpoint_tool.v2803_smoke:main' \
  --require-entrypoint \
    'onnx-splitpoint-smoke-v2801=onnx_splitpoint_tool.v2801_smoke:main' \
  --require-entrypoint \
    'onnx-splitpoint-smoke-v2-80-1=onnx_splitpoint_tool.v2801_smoke:main' \
  --require-entrypoint \
    'onnx-splitpoint-smoke-v280=onnx_splitpoint_tool.v280_smoke:main' \
  --require-entrypoint \
    'onnx-splitpoint-smoke-v2-80=onnx_splitpoint_tool.v280_smoke:main' \
  --require-entrypoint \
    'onnx-splitpoint-smoke-v27931=onnx_splitpoint_tool.v27931_smoke:main' \
  --require-entrypoint \
    'onnx-splitpoint-smoke-v2-79-31=onnx_splitpoint_tool.v27931_smoke:main' \
  --require-entrypoint \
    'onnx-splitpoint-smoke-v27930=onnx_splitpoint_tool.v27930_smoke:main' \
  --require-entrypoint \
    'onnx-splitpoint-smoke-v2-79-30=onnx_splitpoint_tool.v27930_smoke:main' \
  --require-entrypoint \
    'onnx-splitpoint-platform-power=onnx_splitpoint_tool.platform_power_cli:main' \
  --require-entrypoint \
    'onnx-splitpoint-smoke-v2-79=onnx_splitpoint_tool.v279_smoke:main' \
  --require-entrypoint \
    'onnx-splitpoint-backend-semantic-smoke=onnx_splitpoint_tool.backend_semantic_smoke:main' \
  --require-entrypoint \
    'onnx-splitpoint-smoke-v278=onnx_splitpoint_tool.v278_smoke:main' \
  --require-entrypoint \
    'onnx-splitpoint-smoke-v2-78=onnx_splitpoint_tool.v278_smoke:main' \
  --require-entrypoint \
    'onnx-splitpoint-smoke-v277=onnx_splitpoint_tool.v277_smoke:main' \
  --require-entrypoint \
    'onnx-splitpoint-smoke-v2-77=onnx_splitpoint_tool.v277_smoke:main' \
  --require-entrypoint \
    'onnx-splitpoint-smoke-v276=onnx_splitpoint_tool.v276_smoke:main' \
  --require-entrypoint \
    'onnx-splitpoint-smoke-v2-76=onnx_splitpoint_tool.v276_smoke:main' \
  --require-entrypoint \
    'onnx-splitpoint-smoke-v27550=onnx_splitpoint_tool.v27550_smoke:main' \
  --require-entrypoint \
    'onnx-splitpoint-smoke-v2-75-50=onnx_splitpoint_tool.v27550_smoke:main' \
  --require-entrypoint \
    'onnx-splitpoint-smoke-v27549=onnx_splitpoint_tool.v27549_smoke:main' \
  --require-entrypoint \
    'onnx-splitpoint-smoke-v2-75-49=onnx_splitpoint_tool.v27549_smoke:main' \
  --require-entrypoint \
    'onnx-splitpoint-smoke-v27548=onnx_splitpoint_tool.v27548_smoke:main' \
  --require-entrypoint \
    'onnx-splitpoint-smoke-v2-75-48=onnx_splitpoint_tool.v27548_smoke:main' \
  --require-entrypoint \
    'onnx-splitpoint-smoke-v27547=onnx_splitpoint_tool.v27547_smoke:main' \
  --require-entrypoint \
    'onnx-splitpoint-smoke-v2-75-47=onnx_splitpoint_tool.v27547_smoke:main' \
  --require-entrypoint \
    'onnx-splitpoint-smoke-v27546=onnx_splitpoint_tool.v27546_smoke:main' \
  --require-entrypoint \
    'onnx-splitpoint-smoke-v2-75-46=onnx_splitpoint_tool.v27546_smoke:main' \
  --require-entrypoint \
    'onnx-splitpoint-smoke-v27545=onnx_splitpoint_tool.v27545_smoke:main' \
  --require-entrypoint \
    'onnx-splitpoint-smoke-v2-75-45=onnx_splitpoint_tool.v27545_smoke:main' \
  --require-entrypoint \
    'onnx-splitpoint-smoke-v27544=onnx_splitpoint_tool.v27544_smoke:main' \
  --require-entrypoint \
    'onnx-splitpoint-smoke-v2-75-44=onnx_splitpoint_tool.v27544_smoke:main' \
  --require-entrypoint \
    'onnx-splitpoint-smoke-v27543=onnx_splitpoint_tool.v27543_smoke:main' \
  --require-entrypoint \
    'onnx-splitpoint-smoke-v2-75-43=onnx_splitpoint_tool.v27543_smoke:main' \
  --run-entrypoint onnx-splitpoint-smoke-v282

restore_preserved_profiles

# The synchronized release remains byte-exact while the installation may
# legitimately contain retained top-level operational directories and
# user-owned evaluation profiles. Verify that final hybrid tree explicitly;
# release/archive verification above intentionally remains strict.
trusted_source_preflight installed-tree \
  "$TOOL_DIR" "$ZIP" "$TRUSTED_MANIFEST_SHA256" "$TRUSTED_ARCHIVE_SHA256" \
  >/dev/null
PYTHONPATH= PYTHONHOME= \
  "$PYTHON" -I -S -B scripts/build_source_manifest.py \
  --root "$TOOL_DIR" --verify --scope installed >/dev/null

# The updater intentionally preserves the existing venv and never performs a
# network install. Official COCO remains the accepted YOLOv7 pre-anchor gate, so
# report its readiness explicitly and print the exact local remediation step.
if PYTHONPATH= "$PYTHON" -B -c \
    'import importlib.metadata as m, importlib.util as u, re; required=("onnxruntime","numpy","PIL","pycocotools"); missing=[name for name in required if u.find_spec(name) is None]; assert not missing, missing; v=m.version("pycocotools"); p=re.fullmatch(r"(\d+)\.(\d+)\.(\d+)(?:[.+-].*)?", v); assert p and tuple(map(int, p.groups())) >= (2, 0, 7), v; print("YOLOv7 probe dependencies: available; pycocotools", v)'; then
  YOLOV7_PROBE_DEPS_STATUS="available (pycocotools>=2.0.7)"
else
  YOLOV7_PROBE_DEPS_STATUS="missing/incomplete or pycocotools too old (<2.0.7)"
  printf '%s\n' \
    'WARNUNG: Die YOLOv7-Probe-Abhängigkeiten fehlen oder pycocotools ist älter als 2.0.7.' \
    'Vor dem YOLOv7-Decoder-A/B-Gate ausführen:' >&2
  printf '  cd -- %q\n' "$TOOL_DIR" >&2
  printf '%s\n' \
    '  ./.venv/bin/python -m onnx_splitpoint_tool.dependency_bootstrap --groups yolov7_probe --python ./.venv/bin/python' \
    "  ./.venv/bin/pip install --upgrade 'pycocotools>=2.0.7'" >&2
fi

printf '%s\n' \
  "PASS source release synchronized" \
  "Venv erhalten: $TOOL_DIR/.venv" \
  "Distribution aktualisiert: onnx-splitpoint-tool==2.90.1" \
  "Release-Scope: Force AUS, Artefaktwiederverwendung, Hailo8-Compute-Umgebung und Runtime-Cleanup" \
  "Hardware-Kalibrierung: NOT_RUN (Updater führt keine Hardwareaktion aus)" \
  "YOLOv7-Probe-Abhängigkeiten: $YOLOV7_PROBE_DEPS_STATUS" \
  "Eigene Profile erhalten: $PRESERVED_PROFILE_COUNT" \
  "Backup geänderter/entfernter Dateien: $BACKUP_DIR"
