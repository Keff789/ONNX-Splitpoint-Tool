#!/usr/bin/env bash
set -euo pipefail

ROOT="${DEEPX_DX_ALL_SUITE_ROOT:-${DX_ALL_SUITE_ROOT:-$HOME/dx-all-suite}}"
COMP_VENV="${DEEPX_COMPILER_VENV:-$ROOT/dx-compiler/venv-dx-compiler-local}"
RUNTIME_VENV="${DEEPX_RUNTIME_VENV:-$HOME/venvs/deepx-runtime}"
MODE="status"
BRANCH="${DEEPX_DX_ALL_SUITE_REF:-v2.3.2}"
CHECKOUT_IF_MISSING=1
INSTALL_COMPILER=${DEEPX_RUN_COMPILER_INSTALL:-0}
INSTALL_RUNTIME_PY=1
FORCE_COMPILER_RECREATE=${DEEPX_FORCE_COMPILER_RECREATE:-auto}
COMPILER_INSTALL_MODE="${DEEPX_COMPILER_INSTALL_MODE:-direct}"  # direct | upstream | auto
DX_COM_VERSION="${DEEPX_DX_COM_VERSION:-}"
DX_COM_WHEEL_URL="${DEEPX_DX_COM_WHEEL_URL:-}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --repair|--install) MODE="repair"; shift;;
    --status) MODE="status"; shift;;
    --root) ROOT="$2"; shift 2;;
    --compiler-venv) COMP_VENV="$2"; shift 2;;
    --runtime-venv) RUNTIME_VENV="$2"; shift 2;;
    --compiler-python) DEEPX_COMPILER_PYTHON="$2"; shift 2;;
    --runtime-python) DEEPX_RUNTIME_PYTHON="$2"; shift 2;;
    --branch|--ref) BRANCH="$2"; shift 2;;
    --no-checkout) CHECKOUT_IF_MISSING=0; shift;;
    --run-compiler-install|--install-compiler) INSTALL_COMPILER=1; shift;;
    --no-compiler-install) INSTALL_COMPILER=0; shift;;
    --compiler-install-mode) COMPILER_INSTALL_MODE="$2"; shift 2;;
    --direct-compiler-install) COMPILER_INSTALL_MODE="direct"; INSTALL_COMPILER=1; shift;;
    --upstream-compiler-install) COMPILER_INSTALL_MODE="upstream"; INSTALL_COMPILER=1; shift;;
    --no-runtime-python-install) INSTALL_RUNTIME_PY=0; shift;;
    --force-compiler-recreate|--force-recreate-compiler-venv) FORCE_COMPILER_RECREATE=1; shift;;
    *) echo "Unknown argument: $1" >&2; exit 2;;
  esac
done

_expand_path() {
  python3 - <<'PY' "$1"
import os, sys
print(os.path.abspath(os.path.expanduser(os.path.expandvars(sys.argv[1]))))
PY
}
ROOT="$(_expand_path "$ROOT")"
COMP_VENV="$(_expand_path "$COMP_VENV")"
RUNTIME_VENV="$(_expand_path "$RUNTIME_VENV")"

_have_cmd() { command -v "$1" >/dev/null 2>&1; }
_choose_compiler_python() {
  if [[ -n "${DEEPX_COMPILER_PYTHON:-}" ]]; then
    if [[ -x "${DEEPX_COMPILER_PYTHON}" ]]; then echo "${DEEPX_COMPILER_PYTHON}"; return 0; fi
    if _have_cmd "${DEEPX_COMPILER_PYTHON}"; then command -v "${DEEPX_COMPILER_PYTHON}"; return 0; fi
    echo "[warn] requested compiler python not found: ${DEEPX_COMPILER_PYTHON}; falling back" >&2
  fi
  # DX-COM v2.3.x ships cp311 wheels on Ubuntu 24.04.  Prefer 3.11.
  for py in python3.11 python3.10 python3.12 python3; do
    if _have_cmd "$py"; then command -v "$py"; return 0; fi
  done
  echo python3
}
_py_tag() {
  "$1" - <<'PY'
import sys
print(f"cp{sys.version_info.major}{sys.version_info.minor}")
PY
}
_py_ver_minor() {
  "$1" - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
}
_py_ver_full() {
  "$1" - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
PY
}
_runtime_python="${DEEPX_RUNTIME_PYTHON:-python3}"
if [[ -x "$_runtime_python" ]]; then :; elif _have_cmd "$_runtime_python"; then _runtime_python="$(command -v "$_runtime_python")"; else echo "[warn] requested runtime python not found: $_runtime_python; falling back to python3"; _runtime_python="$(command -v python3 || echo python3)"; fi
_compiler_python="$(_choose_compiler_python)"

_find_dx_com_wheels() {
  local roots=()
  [[ -d "$ROOT/dx-compiler/download" ]] && roots+=("$ROOT/dx-compiler/download")
  [[ -d "$ROOT/dx-compiler/dx_com" ]] && roots+=("$ROOT/dx-compiler/dx_com")
  [[ -d "$ROOT/workspace/release/dx_com" ]] && roots+=("$ROOT/workspace/release/dx_com")
  [[ -d "$ROOT/dx-compiler/temp_downloads" ]] && roots+=("$ROOT/dx-compiler/temp_downloads")
  if [[ ${#roots[@]} -gt 0 ]]; then
    find -L "${roots[@]}" -type f -name 'dx_com-*.whl' 2>/dev/null | sort -u
  fi
}
_find_matching_dx_com_wheel() {
  local tag="$1" w
  while IFS= read -r w; do
    [[ -z "$w" ]] && continue
    if [[ "$(basename "$w")" == *"-${tag}-${tag}-"* || "$(basename "$w")" == *"-${tag}-"* ]]; then echo "$w"; return 0; fi
  done < <(_find_dx_com_wheels)
  return 1
}
_first_dx_com_wheel() { _find_dx_com_wheels | head -n 1; }
_guess_dx_com_version() {
  if [[ -n "$DX_COM_VERSION" ]]; then echo "$DX_COM_VERSION"; return 0; fi
  local props="$ROOT/dx-compiler/compiler.properties" v=""
  if [[ -f "$props" ]]; then
    v="$(grep -Eio 'dx[_-]?com[^0-9]{0,40}[0-9]+\.[0-9]+\.[0-9]+' "$props" | grep -Eo '[0-9]+\.[0-9]+\.[0-9]+' | head -n 1 || true)"
    [[ -n "$v" ]] && { echo "$v"; return 0; }
    v="$(grep -Eio 'v[0-9]+\.[0-9]+\.[0-9]+/dx_com-[0-9]+\.[0-9]+\.[0-9]+' "$props" | grep -Eo '[0-9]+\.[0-9]+\.[0-9]+' | tail -n 1 || true)"
    [[ -n "$v" ]] && { echo "$v"; return 0; }
  fi
  # Known mapping from dx-all-suite v2.3.2 compatibility notes/logs.
  case "$BRANCH" in
    v2.3.2|2.3.2|v2.3.*|2.3.*) echo "2.3.0";;
    *) echo "2.3.0";;
  esac
}
_download_dx_com_wheel_direct() {
  local tag="$1" ver url outdir out olddir
  ver="$(_guess_dx_com_version)"
  url="${DX_COM_WHEEL_URL:-https://sdk.deepx.ai/release/dxcom/v${ver}/dx_com-${ver}-${tag}-${tag}-manylinux_2_31_x86_64.whl}"
  outdir="$ROOT/workspace/release/dx_com/download"
  olddir="$ROOT/dx-compiler/download"
  out="$outdir/$(basename "$url")"
  mkdir -p "$outdir" "$olddir"
  if [[ -s "$out" ]]; then
    echo "[info] DX-COM wheel already downloaded: $out"
  else
    echo "[cmd] download DX-COM wheel: $url"
    python3 - <<'PY' "$url" "$out"
import sys, urllib.request, pathlib, time
url, out = sys.argv[1], pathlib.Path(sys.argv[2])
tmp = out.with_suffix(out.suffix + '.part')
print(f"[download] {url}")
with urllib.request.urlopen(url, timeout=120) as r, tmp.open('wb') as f:
    total = int(r.headers.get('content-length') or 0)
    done = 0
    last = 0.0
    while True:
        b = r.read(1024 * 512)
        if not b:
            break
        f.write(b)
        done += len(b)
        now = time.time()
        if now - last > 2:
            if total:
                print(f"[download] {done/1024/1024:.1f}/{total/1024/1024:.1f} MB")
            else:
                print(f"[download] {done/1024/1024:.1f} MB")
            last = now
if total and done < total:
    raise SystemExit(f"download incomplete: {done}/{total}")
tmp.replace(out)
print(f"[download] saved {out} ({done} bytes)")
PY
  fi
  ln -sfn "$out" "$olddir/$(basename "$out")" || true
  echo "$out"
}
_ensure_compiler_venv() {
  local py="$1"
  mkdir -p "$(dirname "$COMP_VENV")"
  if [[ ! -x "$COMP_VENV/bin/python" ]]; then
    echo "[cmd] $py -m venv --system-site-packages $COMP_VENV"
    "$py" -m venv --system-site-packages "$COMP_VENV"
  else
    echo "[info] compiler venv already present: $COMP_VENV ($($COMP_VENV/bin/python -V 2>&1))"
  fi
}
_recreate_compiler_venv() {
  local py="$1"
  if [[ -d "$COMP_VENV" ]]; then
    echo "[info] Recreating compiler venv with $($py -V 2>&1): $COMP_VENV"
    rm -rf "$COMP_VENV"
  fi
  echo "[cmd] $py -m venv --system-site-packages $COMP_VENV"
  "$py" -m venv --system-site-packages "$COMP_VENV"
}
_ensure_compiler_python_tag_matches_wheel() {
  [[ -x "$COMP_VENV/bin/python" ]] || return 0
  local tag first base candidate_py candidate_tag
  tag="$(_py_tag "$COMP_VENV/bin/python")"
  if _find_matching_dx_com_wheel "$tag" >/dev/null 2>&1; then return 0; fi
  first="$(_first_dx_com_wheel || true)"
  [[ -n "$first" ]] || return 0
  base="$(basename "$first")"
  if [[ "$base" == *cp311* ]] && _have_cmd python3.11; then
    candidate_py="$(command -v python3.11)"
    candidate_tag="$(_py_tag "$candidate_py")"
    if [[ "$candidate_tag" == "cp311" ]]; then
      echo "[info] Compiler venv tag $tag does not match $base; switching compiler venv to python3.11."
      _recreate_compiler_venv "$candidate_py"
    fi
  fi
}
_install_base_compiler_pip_tools() {
  echo "[cmd] $COMP_VENV/bin/python -m pip install -U pip 'setuptools<82' wheel packaging"
  "$COMP_VENV/bin/python" -m pip install -U pip 'setuptools<82' wheel packaging || true
}
_install_dx_com_wheel_direct() {
  local py="$COMP_VENV/bin/python" tag wheel
  [[ -x "$py" ]] || return 2
  tag="$(_py_tag "$py")"
  if ! wheel="$(_find_matching_dx_com_wheel "$tag")" || [[ -z "${wheel:-}" ]]; then
    wheel="$(_download_dx_com_wheel_direct "$tag")"
  fi
  echo "[cmd] $py -m pip install -U --force-reinstall $wheel"
  "$py" -m pip install -U --force-reinstall "$wheel"
}
_probe_compiler_import() {
  local py="$COMP_VENV/bin/python"
  [[ -x "$py" ]] || return 2
  "$py" - <<'PY'
import importlib, shutil, sys
ok = False
for mod in ('dx_com', 'dx_compiler'):
    try:
        importlib.import_module(mod)
        print(f"[check] compiler import {mod}: OK")
        ok = True
    except Exception as exc:
        print(f"[warn] compiler import {mod}: {type(exc).__name__}: {exc}")
for exe in ('dxcom', 'dx_com'):
    p = shutil.which(exe)
    if p:
        print(f"[check] compiler CLI {exe}: {p}")
raise SystemExit(0 if ok else 2)
PY
}
_run_upstream_install_best_effort() {
  [[ -x "$ROOT/dx-compiler/install.sh" ]] || { echo "[warn] upstream install.sh not found"; return 1; }
  local minor timeout_s base_py
  base_py="$_compiler_python"
  minor="$(_py_ver_minor "$base_py")"
  timeout_s="${DEEPX_UPSTREAM_INSTALL_TIMEOUT_S:-900}"
  echo "[cmd] upstream dx-compiler/install.sh --python_version=$minor (timeout ${timeout_s}s)"
  echo "[note] Running upstream installer like the manual workflow, not from inside the managed compiler venv."
  echo "[note] The managed compiler venv is removed first so install.sh cannot stall while deleting its own active environment."
  if [[ -d "$COMP_VENV" ]]; then
    echo "[cmd] rm -rf $COMP_VENV"
    rm -rf "$COMP_VENV"
  fi
  (cd "$ROOT" && timeout "$timeout_s" env DEEPX_COMPILER_PYTHON="$base_py" ./dx-compiler/install.sh --python_version="$minor") || {
    echo "[warn] upstream dx-compiler/install.sh returned non-zero or timed out; continuing with direct DX-COM wheel recovery."
    return 1
  }
}

# Header
echo "[info] dx-all-suite root: $ROOT"
echo "[info] dx-all-suite ref : $BRANCH"
echo "[info] compiler python preference: $_compiler_python"
echo "[info] compiler venv: $COMP_VENV"
echo "[info] compiler install mode: $COMPILER_INSTALL_MODE"
echo "[info] runtime python preference: $_runtime_python"
echo "[info] runtime venv : $RUNTIME_VENV"

action_required=0
if [[ ! -d "$ROOT" ]]; then
  if [[ "$MODE" == "repair" && "$CHECKOUT_IF_MISSING" == "1" ]]; then
    parent="$(dirname "$ROOT")"
    echo "[info] dx-all-suite root missing; trying git checkout into $ROOT"
    mkdir -p "$parent"
    if [[ ! -w "$parent" ]]; then
      echo "[error] cannot write to parent directory: $parent" >&2
      echo "[hint] Set DeepX DX-M1 root to a writable path such as ~/dx-all-suite." >&2
      exit 3
    fi
    command -v git >/dev/null 2>&1 || { echo "[error] git is not installed. Install: sudo apt update && sudo apt install -y git" >&2; exit 3; }
    echo "[cmd] git clone -b $BRANCH --recurse-submodules https://github.com/DEEPX-AI/dx-all-suite.git $ROOT"
    git clone -b "$BRANCH" --recurse-submodules https://github.com/DEEPX-AI/dx-all-suite.git "$ROOT"
  else
    echo "[error] dx-all-suite root not found: $ROOT" >&2
    exit 3
  fi
fi

if [[ "$MODE" == "repair" ]]; then
  _ensure_compiler_venv "$_compiler_python"
  if [[ "$FORCE_COMPILER_RECREATE" == "1" ]]; then _recreate_compiler_venv "$_compiler_python"; fi
  _ensure_compiler_python_tag_matches_wheel
  _install_base_compiler_pip_tools

  if [[ "$INSTALL_COMPILER" == "1" ]]; then
    echo "[info] Installing DX-COM compiler module. Default path is direct wheel install; no sudo is required."
    if [[ "$COMPILER_INSTALL_MODE" == "upstream" ]]; then
      _run_upstream_install_best_effort || true
    elif [[ "$COMPILER_INSTALL_MODE" == "auto" ]]; then
      echo "[info] auto mode: trying direct DX-COM wheel install first; upstream install remains a fallback only if explicitly requested."
    fi
    _ensure_compiler_python_tag_matches_wheel
    _install_base_compiler_pip_tools
    if _install_dx_com_wheel_direct; then
      echo "[info] DX-COM wheel installed into compiler venv."
    else
      echo "[warn] direct DX-COM wheel install did not complete. Compiler may remain unavailable."
      action_required=2
    fi
  else
    # Safe repair: install an already available compatible wheel, but do not call upstream/download.
    if [[ -x "$COMP_VENV/bin/python" ]] && _find_matching_dx_com_wheel "$(_py_tag "$COMP_VENV/bin/python")" >/dev/null 2>&1; then
      _install_dx_com_wheel_direct || true
    else
      echo "[info] Safe repair did not run compiler install. Click 'Install compiler' to download/install DX-COM."
    fi
  fi

  if [[ ! -x "$RUNTIME_VENV/bin/python" ]]; then
    mkdir -p "$(dirname "$RUNTIME_VENV")"
    echo "[cmd] $_runtime_python -m venv --system-site-packages $RUNTIME_VENV"
    "$_runtime_python" -m venv --system-site-packages "$RUNTIME_VENV"
  else
    echo "[info] runtime venv already present: $RUNTIME_VENV"
  fi
  if [[ "$INSTALL_RUNTIME_PY" == "1" && -x "$RUNTIME_VENV/bin/python" && -d "$ROOT/dx-runtime/dx_rt/python_package" ]]; then
    echo "[cmd] $RUNTIME_VENV/bin/python -m pip install -U pip setuptools wheel"
    "$RUNTIME_VENV/bin/python" -m pip install -U pip setuptools wheel || true
    echo "[cmd] $RUNTIME_VENV/bin/python -m pip install --no-deps -U $ROOT/dx-runtime/dx_rt/python_package"
    "$RUNTIME_VENV/bin/python" -m pip install --no-deps -U "$ROOT/dx-runtime/dx_rt/python_package" || { echo "[warn] DX-RT Python package install failed; runtime tools may still work if installed system-wide."; action_required=1; }
    # v52f: ORT/TensorRT remote runs also need the ONNX Python package.
    # Keep numpy below 2 to avoid Jetson OpenCV/matplotlib ABI failures.
    echo "[cmd] $RUNTIME_VENV/bin/python -m pip install -U 'numpy<2' 'protobuf<6' 'onnx>=1.17,<1.22'"
    "$RUNTIME_VENV/bin/python" -m pip install -U 'numpy<2' 'protobuf<6' 'onnx>=1.17,<1.22' || echo "[warn] ONNX Python package install failed; TensorRT/ORT runs may fail preflight."
  fi
fi

compiler_ok=0
runtime_ok=0
if [[ -x "$COMP_VENV/bin/python" ]]; then
  echo "[check] compiler python: $($COMP_VENV/bin/python -V 2>&1)"
  if _probe_compiler_import; then compiler_ok=1; else compiler_ok=0; fi
else
  echo "[warn] compiler python missing: $COMP_VENV/bin/python"
fi
if [[ -x "$RUNTIME_VENV/bin/python" ]]; then
  echo "[check] runtime python: $($RUNTIME_VENV/bin/python -V 2>&1)"
  if "$RUNTIME_VENV/bin/python" - <<'PY'; then runtime_ok=1; else runtime_ok=0; fi
import shutil
ok = True
try:
    import dx_engine
    print("[check] runtime import dx_engine: OK")
except Exception as exc:
    print(f"[warn] runtime import dx_engine: {type(exc).__name__}: {exc}")
    ok = False
try:
    import tensorrt as trt
    print(f"[check] runtime import tensorrt: OK {getattr(trt, '__version__', '?')}")
except Exception as exc:
    print(f"[info] runtime import tensorrt: {type(exc).__name__}: {exc}")
try:
    import onnx
    print(f"[check] runtime import onnx: OK {getattr(onnx, '__version__', '?')}")
except Exception as exc:
    print(f"[info] runtime import onnx: {type(exc).__name__}: {exc}")
try:
    import onnxruntime as ort
    print(f"[check] runtime import onnxruntime: OK {getattr(ort, '__version__', '?')} providers={ort.get_available_providers()}")
except Exception as exc:
    print(f"[info] runtime import onnxruntime: {type(exc).__name__}: {exc}")
try:
    import ctypes.util
    print(f"[check] runtime CUDA lib cublas: {ctypes.util.find_library('cublas')}")
    print(f"[check] runtime CUDA lib cublasLt: {ctypes.util.find_library('cublasLt')}")
except Exception as exc:
    print(f"[info] runtime CUDA library check: {type(exc).__name__}: {exc}")
for exe in ("run_model", "parse_model", "dxrt-cli", "trtexec"):
    p = shutil.which(exe)
    print(f"[check] runtime CLI {exe}: {p or '-'}")
raise SystemExit(0 if ok else 1)
PY
else
  echo "[warn] runtime python missing: $RUNTIME_VENV/bin/python"
fi
command -v dxrt-cli >/dev/null 2>&1 && echo "[check] dxrt-cli: $(command -v dxrt-cli)" || echo "[warn] dxrt-cli not in PATH"
command -v parse_model >/dev/null 2>&1 && echo "[check] parse_model: $(command -v parse_model)" || echo "[warn] parse_model not in PATH"
command -v run_model >/dev/null 2>&1 && echo "[check] run_model: $(command -v run_model)" || echo "[warn] run_model not in PATH"
ls /dev/dxrt* >/dev/null 2>&1 && echo "[check] /dev/dxrt*: $(ls /dev/dxrt* | tr '\n' ' ')" || echo "[warn] no /dev/dxrt* device on this host"

if [[ "$compiler_ok" == "1" && "$runtime_ok" == "1" && "$action_required" == "0" ]]; then
  echo "[done] exit_code=0 deepx build/runtime environment ready"
  exit 0
elif [[ "$compiler_ok" == "1" && "$runtime_ok" == "0" ]]; then
  echo "[partial] exit_code=5 deepx compiler ready but runtime is not ready"
  exit 5
elif [[ "$compiler_ok" == "0" && "$runtime_ok" == "1" ]]; then
  echo "[partial] exit_code=4 deepx runtime/repo ready but compiler is not ready"
  echo "[hint] Run Install compiler. If only cp311 DX-COM wheels exist, make sure python3.11 is installed."
  exit 4
else
  echo "[error] exit_code=3 deepx environment is not ready"
  exit 3
fi
