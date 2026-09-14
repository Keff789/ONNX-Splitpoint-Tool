#!/usr/bin/env bash
set -Eeuo pipefail
TEST_TOOL="${ONNX_SPLITPOINT_TOOL_DIR:-$HOME/ONNX-Splitpoint-Tool}"
TEST_PY="$TEST_TOOL/.venv/bin/python"
[[ -x "$TEST_PY" ]] || { echo "STOP: Tool-Venv fehlt: $TEST_PY"; exit 2; }
cd -- "$TEST_TOOL"
mkdir -p -- "$HOME/Downloads"
TEST_OUT="$(mktemp -d "$HOME/Downloads/v27934_short_tests_XXXXXXXX")"

finish() {
  local rc=$?
  trap - EXIT
  set +e
  printf 'SHORT_TESTS_RC=%s\nHARDWARE_EXECUTION=NOT_RUN\n' "$rc" > "$TEST_OUT/status.txt"
  "$TEST_PY" -B - "$TEST_OUT" <<'PY'
from pathlib import Path
import hashlib, sys, zipfile
p = Path(sys.argv[1])
archive = p.parent / (p.name + '.zip')
with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as z:
    for f in sorted(p.rglob('*')):
        if f.is_file() and not f.is_symlink():
            z.write(f, f.relative_to(p.parent))
with zipfile.ZipFile(archive) as z:
    if z.testzip() is not None:
        raise SystemExit('STOP: Evidence-CRC fehlgeschlagen')
print('EVIDENCE_ZIP=' + str(archive))
print('EVIDENCE_SHA256=' + hashlib.sha256(archive.read_bytes()).hexdigest())
PY
  local zip_rc=$?
  [[ "$rc" -ne 0 || "$zip_rc" -eq 0 ]] || rc="$zip_rc"
  printf 'SHORT_TESTS_RC=%s\nREPORT_DIR=%s\n' "$rc" "$TEST_OUT"
  exit "$rc"
}
trap finish EXIT

export PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export PYTEST_ADDOPTS='' PYTHONPATH="$TEST_TOOL" CUDA_VISIBLE_DEVICES=''
export ORT_DISABLE_TELEMETRY=1

"$TEST_PY" -B -c 'import onnxruntime as ort; ort.disable_telemetry_events(); from onnx_splitpoint_tool.v27934_smoke import main; raise SystemExit(main())' \
  2>&1 | tee "$TEST_OUT/release_smoke.log"
"$TEST_PY" -B scripts/build_source_manifest.py --root "$TEST_TOOL" --verify --scope installed \
  > "$TEST_OUT/source_before.json"

TEST_FILES=(
  tests/test_v27933_run_mode_config.py
  tests/test_v27933_force_admission.py
  tests/test_v27933_gui_configuration_start.py
  tests/test_v27933_generic_full_binding.py
  tests/test_v27933_runtime_reuse.py
  tests/test_v27934_force_defaults.py
  tests/test_v27934_compiler_context.py
  tests/test_v27934_config_workflow.py
  tests/test_v27934_hailo_backend.py
  tests/test_v27934_model_diagnostic.py
  tests/test_v27934_model_runtime_worker.py
  tests/test_v27934_hailo8_dependency_plan.py
  tests/test_v27934_preparation_profile.py
)
"$TEST_PY" -B -m pytest -q -ra -o xfail_strict=true -p no:cacheprovider \
  --junitxml="$TEST_OUT/short_tests.xml" "${TEST_FILES[@]}" \
  2>&1 | tee "$TEST_OUT/short_tests.log"
"$TEST_PY" -B scripts/build_source_manifest.py --root "$TEST_TOOL" --verify --scope installed \
  > "$TEST_OUT/source_after.json"
"$TEST_PY" -B - "$TEST_OUT" <<'PY'
from pathlib import Path
import json, sys, xml.etree.ElementTree as ET
p = Path(sys.argv[1])
rows = list(ET.parse(p/'short_tests.xml').iter('testcase'))
bad = [r for r in rows if any(r.find(k) is not None for k in ('failure', 'error', 'skipped'))]
before = json.loads((p/'source_before.json').read_text())
after = json.loads((p/'source_after.json').read_text())
if not rows or bad or not before.get('ok') or not after.get('ok'):
    raise SystemExit(f'STOP: {len(rows)} Tests; {len(bad)} Fehler/Skips oder Sourceprüfung fehlgeschlagen')
if before.get('user_profiles') != after.get('user_profiles'):
    raise SystemExit('STOP: Nutzerprofilbestand während Kurztests geändert')
print(f'SHORT_TESTS=PASS ({len(rows)} Tests; Source und Nutzerprofile geprüft)')
print('HARDWARE_EXECUTION=NOT_RUN')
PY
