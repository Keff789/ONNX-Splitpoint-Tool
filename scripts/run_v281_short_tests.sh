#!/usr/bin/env bash
set -Eeuo pipefail
TEST_TOOL="${ONNX_SPLITPOINT_TOOL_DIR:-$HOME/ONNX-Splitpoint-Tool}"
TEST_PY="$TEST_TOOL/.venv/bin/python"
[[ -x "$TEST_PY" ]] || { echo "STOP: Tool-Venv fehlt: $TEST_PY"; exit 2; }
cd -- "$TEST_TOOL"
mkdir -p -- "$HOME/Downloads"
TEST_OUT="$(mktemp -d "$HOME/Downloads/v281_short_tests_XXXXXXXX")"

TEST_ENV_STATUS=NOT_RUN

finish() {
  local rc=$?
  trap - EXIT
  set +e
  printf 'SHORT_TESTS_RC=%s\nHARDWARE_EXECUTION=NOT_RUN\nACCEPTANCE_ENVIRONMENT=%s\n' "$rc" "$TEST_ENV_STATUS" > "$TEST_OUT/status.txt"
  "$TEST_PY" -B - "$TEST_OUT" <<'PY'
from pathlib import Path
import hashlib, os, sys, tempfile, zipfile
p = Path(sys.argv[1])
archive = p.parent / (p.name + '.zip')
fd, temporary = tempfile.mkstemp(prefix=archive.name + '.', suffix='.part', dir=archive.parent)
os.close(fd)
try:
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as z:
        for f in sorted(p.rglob('*')):
            if f.is_file() and not f.is_symlink():
                z.write(f, f.relative_to(p.parent))
    with zipfile.ZipFile(temporary) as z:
        if z.testzip() is not None:
            raise SystemExit('STOP: Evidence-CRC fehlgeschlagen')
    os.link(temporary, archive)
finally:
    os.unlink(temporary)
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

TEST_ENV_STATUS=environment_blocked
"$TEST_PY" -B scripts/check_acceptance_environment.py | tee "$TEST_OUT/acceptance_environment.json"
TEST_ENV_STATUS=PASS

"$TEST_PY" -B -c 'import onnxruntime as ort; ort.disable_telemetry_events(); from onnx_splitpoint_tool.v281_smoke import main; raise SystemExit(main())' \
  2>&1 | tee "$TEST_OUT/release_smoke.log"
"$TEST_PY" -B scripts/build_source_manifest.py --root "$TEST_TOOL" --verify --scope installed \
  > "$TEST_OUT/source_before.json"

TEST_FILES=(
  tests/test_v280_hailo8_compute_env.py
  tests/test_v280_runtime_cleanup.py
  tests/test_v280_artifact_reuse.py
  tests/test_v280_release_scope.py
  tests/test_v2801_cpu_reference_dispatch.py
  tests/test_v2801_native_remote_closure.py
  tests/test_v2801_reference_errors.py
  tests/test_v2801_reference_debug_export.py
  tests/test_v2801_workflow_integration.py
  tests/test_v2801_release_scope.py
  tests/test_v2803_deferred_build_readiness.py
  tests/test_v2803_native_not_started.py
  tests/test_v2803_debug_pack_large_sources.py
  tests/test_v2803_runtime_diagnostic_projection.py
  tests/test_v2803_reference_workflow_gate.py
  tests/test_v2803_scope_and_diagnostic_claims.py
  tests/test_v2803_measurement_configuration.py
  tests/test_v2803_release_scope.py
  tests/test_v2804_debug_limits.py
  tests/test_v2804_gui_overlay.py
  tests/test_v2804_hailo_claims.py
  tests/test_v2804_hailo_manifest.py
  tests/test_v2804_quality_cancel.py
  tests/test_v2804_reuse_quality.py
  tests/test_v2804_workflow_starter.py
  tests/test_v281_*.py
  tests/test_v2803_fix2_deepx_geometry.py
  tests/test_v2803_fix2_native_reporting.py
  tests/test_v2803_fix2_reference_prepare.py
  tests/test_v2803_fix2_trt_preflight.py
  tests/test_v2803_fix3_hailo_artifact_scope.py
  tests/test_v2803_fix3_service_debug_export.py
  tests/test_v2803_fix4_reference_scope.py
  tests/test_v2803_fix4_trt_build_guard.py
  tests/test_v2803_fix4_trt_dispatch_scope.py
  tests/test_v2803_fix5_native_asset_sync.py
  tests/test_v2802_cpu_reference_binding.py
  tests/test_v2802_native_quality_blocking.py
  tests/test_v2802_dashboard_projection.py
  tests/test_v2802_workflow_integration.py
  tests/test_v2802_release_scope.py
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
required_modules = tuple(sorted(p.stem for p in Path('tests').glob('test_v281_*.py')))
executed_modules = {r.get("classname", "").split(".")[1] for r in rows if r.get("classname", "").startswith("tests.test_")}
missing_modules = sorted(set(required_modules) - executed_modules)
bad = [r for r in rows if any(r.find(k) is not None for k in ('failure', 'error', 'skipped'))]
before = json.loads((p/'source_before.json').read_text())
after = json.loads((p/'source_after.json').read_text())
if not rows or bad or missing_modules or not before.get('ok') or not after.get('ok'):
    raise SystemExit(f'STOP: {len(rows)} Tests; {len(bad)} Fehler/Skips; fehlende Pflichtmodule={missing_modules}; Sourceprüfung beachten')
if before.get('user_profiles') != after.get('user_profiles'):
    raise SystemExit('STOP: Nutzerprofilbestand während Kurztests geändert')
print(f'SHORT_TESTS=PASS ({len(rows)} Tests; Source und Nutzerprofile geprüft)')
print('HARDWARE_EXECUTION=NOT_RUN')
PY
