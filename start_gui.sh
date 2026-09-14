#!/bin/bash
set -e

# Python's -B/write-disable flag alone still permits loading a timestamp-valid
# source-local .pyc.  Route cache lookup to a fresh private directory before
# the first Python process; override any inherited cache path.  Unlike
# /dev/null, the private directory also permits explicit py_compile/compileall
# operations used by diagnostics and release verification.
_SPLITPOINT_PYCACHE_ROOT="$(
    /usr/bin/mktemp -d /tmp/onnx-splitpoint-gui-pycache.XXXXXXXXXX
)" || exit 70
_cleanup_splitpoint_pycache() {
    local rc=$?
    trap - EXIT
    case "${_SPLITPOINT_PYCACHE_ROOT:-}" in
        /tmp/onnx-splitpoint-gui-pycache.??????????)
            if [ -L "$_SPLITPOINT_PYCACHE_ROOT" ]; then
                /bin/rm -- "$_SPLITPOINT_PYCACHE_ROOT" || rc=70
            elif [ -d "$_SPLITPOINT_PYCACHE_ROOT" ]; then
                /bin/rm -r -- "$_SPLITPOINT_PYCACHE_ROOT" || rc=70
            elif [ -e "$_SPLITPOINT_PYCACHE_ROOT" ]; then
                rc=70
            fi
            ;;
        *) rc=70 ;;
    esac
    exit "$rc"
}
trap _cleanup_splitpoint_pycache EXIT
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="$_SPLITPOINT_PYCACHE_ROOT"

_START_TS=$(date +%s)
_log_phase() {
    local now elapsed
    now=$(date +%s)
    elapsed=$((now - _START_TS))
    echo "[startup +${elapsed}s] $*"
}

VENV_DIR=".venv"
REQUIREMENTS="requirements.txt"
PYTHON_EXE=""

ensure_venv() {
    if [ -d "$VENV_DIR" ]; then
        _log_phase "Aktiviere Virtual Environment in $VENV_DIR..."
        # shellcheck disable=SC1091
        source "$VENV_DIR/bin/activate"
        PYTHON_EXE="$PWD/$VENV_DIR/bin/python"
        return
    fi

    _log_phase "Virtual Environment '$VENV_DIR' nicht gefunden. Wird erstellt..."
    python3 -m venv "$VENV_DIR"
    _log_phase "Aktiviere neues Virtual Environment..."
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    PYTHON_EXE="$PWD/$VENV_DIR/bin/python"
    _log_phase "Aktualisiere pip..."
    "$PYTHON_EXE" -m pip install --upgrade pip
    if [ -f "$REQUIREMENTS" ]; then
        _log_phase "Installiere Abhängigkeiten aus $REQUIREMENTS..."
        "$PYTHON_EXE" -m pip install -r "$REQUIREMENTS"
    else
        echo "Warnung: $REQUIREMENTS wurde nicht gefunden. Überspringe Paketinstallation."
    fi
}

ensure_startup_dependencies() {
    # v58t: dependency probing imports heavy modules (onnxruntime, matplotlib).
    # Cache a successful check inside the venv so normal GUI starts do not spend
    # tens of seconds before the first window appears.  Missing packages are still
    # handled by analyse_and_split_gui.py on ModuleNotFoundError.
    local STAMP="$VENV_DIR/.osp_gui_core_deps_ok"
    local FORCE="${ONNX_SPLITPOINT_FORCE_DEPS_CHECK:-0}"
    if [ "$FORCE" != "1" ] && [ -f "$STAMP" ]; then
        _log_phase "GUI-Abhängigkeiten: cached ok (set ONNX_SPLITPOINT_FORCE_DEPS_CHECK=1 to recheck)."
        return
    fi
    _log_phase "Prüfe/ergänze minimale GUI-Abhängigkeiten ..."
    "$PYTHON_EXE" -m onnx_splitpoint_tool.dependency_bootstrap --groups gui_core
    touch "$STAMP" || true
}

ensure_venv
ensure_startup_dependencies

_log_phase "Starte Analyse-GUI..."
export ONNX_SPLITPOINT_STARTUP_SHELL_TS="${_START_TS}"
export PYTHONUNBUFFERED=1
# v58ad: startup tracing is opt-in.  The v58ac visibility trace wrote one line
# per Tk <Map>/<Expose> event and could itself make the first paint take a
# minute on large notebooks.  Enable only when explicitly debugging startup:
#   ONNX_SPLITPOINT_STARTUP_TRACE=1 ./start_gui.sh
if [ "${ONNX_SPLITPOINT_STARTUP_TRACE:-0}" = "1" ]; then
  export ONNX_SPLITPOINT_STARTUP_TRACE_FILE="${ONNX_SPLITPOINT_STARTUP_TRACE_FILE:-$PWD/logs/gui/startup_trace_$(date +%Y%m%d_%H%M%S).log}"
  mkdir -p "$(dirname "$ONNX_SPLITPOINT_STARTUP_TRACE_FILE")" 2>/dev/null || true
  _log_phase "Startup trace: $ONNX_SPLITPOINT_STARTUP_TRACE_FILE"
  {
    echo "[startup-trace shell +$(( $(date +%s) - _START_TS ))s] launching python=$PYTHON_EXE cwd=$PWD"
  } >> "$ONNX_SPLITPOINT_STARTUP_TRACE_FILE" 2>/dev/null || true
else
  unset ONNX_SPLITPOINT_STARTUP_TRACE_FILE
fi
"$PYTHON_EXE" -u analyse_and_split_gui.py

deactivate
