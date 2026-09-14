#!/usr/bin/env bash
# Fast, local and hardware-free v2.75.21 acceptance. Run; do not source.

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  printf '%s\n' \
    "SKIP: scripts/run_v27521_small_acceptance.sh was sourced." \
    "Run: bash scripts/run_v27521_small_acceptance.sh" >&2
  return 0
fi

set -Eeuo pipefail

SOURCE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
if [[ -n "${PY:-}" ]]; then
  PYTHON="$PY"
elif [[ -x "$SOURCE_ROOT/.venv/bin/python" ]]; then
  PYTHON="$SOURCE_ROOT/.venv/bin/python"
else
  PYTHON="python3"
fi

cd -- "$SOURCE_ROOT"
export PYTHONDONTWRITEBYTECODE=1
# Harness-local only: let an external venv import the freshly extracted source
# without installing it or mutating that venv.
export PYTHONPATH="$SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"

printf '1/5 identity and syntax\n'
"$PYTHON" -B - <<'PYTHON'
import ast
from pathlib import Path

import onnx_splitpoint_tool as tool
from onnx_splitpoint_tool.workflow.runner import WORKFLOW_VERSION

expected = "v2.75.21-setup-local-tensorrt-quality-dispatch-repair"
assert tool.__version__ == "2.75.21", tool.__version__
assert tool.__build_id__ == expected, tool.__build_id__
assert WORKFLOW_VERSION == expected, WORKFLOW_VERSION
for root in (Path("onnx_splitpoint_tool"), Path("scripts")):
    for path in root.rglob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
print("PASS identity/syntax")
PYTHON
bash -n \
  scripts/update_source_release.sh \
  scripts/run_v27521_cache_canary.sh \
  scripts/run_v27521_small_acceptance.sh \
  scripts/run_v27521_full_only_check.sh

printf '2/5 read-only source manifest\n'
"$PYTHON" -B scripts/build_source_manifest.py \
  --root "$SOURCE_ROOT" \
  --verify

printf '3/5 focused exact-cache and Native runtime contracts\n'
"$PYTHON" -B -m pytest -q -p no:cacheprovider --tb=short \
  tests/test_v2757_cache_verify_only.py \
  tests/test_v2757_cache_canary_handoff.py \
  tests/test_v270_generator_stop_state_regressions.py \
  tests/test_v60v_generation_native_runtime_fixes.py \
  tests/test_v2751_native_quality_runtime_repairs.py \
  tests/test_v2721_hailo8_mixed_runtime_launcher.py \
  tests/test_v275_workflow_energy_contract.py \
  tests/test_v27519_full_only_reference_contract.py \
  tests/test_v27519_logical_target_authority.py \
  tests/test_v27519_management_cpu_production_chain.py \
  tests/test_v27519_native_full_only_admission.py \
  tests/test_v27519_trt_contract_production_chain.py \
  tests/test_v27519_trt_contract_serialization.py \
  tests/test_v27520_remote_boundary_and_plan_finalization.py \
  tests/test_v27520_failed_run_offline_fixture.py \
  tests/test_v27521_setup_local_trt_dispatch.py \
  tests/test_v27521_reporting_and_full_only_authority.py \
  tests/test_v27521_release_wrappers.py \
  tests/test_v275_release_contract.py \
  tests/test_v27518_writer_and_debug_pack_closure.py \
  tests/test_v27518_deepx_sealed_chain.py \
  tests/test_v270i_native_full_validation_p0.py \
  tests/test_v27516_read_only_run_admission.py \
  tests/test_v27516_primary_remote_error_cache_policy.py \
  tests/test_v269e_start_snapshot_backfill.py \
  tests/test_v2712_energy_resume.py \
  tests/test_v2727_native_energy_runtime_admission.py \
  tests/test_p03_native_energy_planner_invariant.py \
  tests/test_v275_hailo_receipt_and_calibration.py \
  tests/test_v2751_hailo_exact_artifact_store_restore.py \
  tests/test_v2752_small_acceptance.py \
  tests/test_v2757_remote_cache_policy_closure.py \
  tests/test_v2758_runtime_holdout_repairs.py \
  tests/test_v27510_smoke_cache_compatibility.py \
  tests/test_v27511_cache_verify_semantic_path.py \
  tests/test_v27511_canonical_native_collector.py \
  tests/test_v27511_semantic_cache_canary_verifier.py \
  tests/test_v262_deepx_hwc_contract_fix.py \
  tests/test_v265_energy_pair_input.py \
  tests/test_v268_native_full_level_field.py \
  tests/test_v269a_deepx_full_central_quality.py \
  tests/test_v269a_yolov7_management_reference.py \
  tests/test_v269d_trt_central_quality_producer.py \
  tests/test_v269d_native_provenance.py \
  tests/test_v269e_quality_export_vendored.py \
  tests/test_v269f_vendor_full_detection.py \
  tests/test_energy_final_contract_regressions.py \
  tests/test_v2711_direct_bn6_producer_binding.py \
  tests/test_v270b_hardware_smoke_repair.py \
  tests/test_v27513_hailo_full_contract_overlay.py \
  tests/test_v27513_quality_join_diagnostics.py \
  tests/test_v27513_verify_native_full_rows.py \
  tests/test_v27515_deepx_source_binding.py \
  tests/test_v27515_deepx_reporter_context.py \
  tests/test_v27515_hailo_vendor_repairs.py \
  tests/test_v270i_p0_completed_endpoint.py \
  tests/test_v275_deepx_full_quality_contract.py \
  tests/test_v275_deepx_sealed_runtime_input.py \
  tests/test_completed_attestation_contract_repairs.py \
  tests/test_campaign_readiness_v60.py

printf '4/5 source release smoke\n'
"$PYTHON" -B -m onnx_splitpoint_tool.v273_smoke

printf '5/5 remote-script mirror equality\n'
for name in \
  materialize_cache_verify_native_split_binding.py \
  native_fifo_eval_runner.py \
  native_fifo_smoke_matrix.py \
  native_full_baseline_eval_runner.py \
  native_full_semantic_dump.py \
  smoke_hailo10_hef_runner.py \
  native_hailo_trt_fifo_from_benchmarkset.py \
  native_producer_final_report.py \
  native_producer_validate_visualize.py \
  native_producer_energy_plan.py \
  run_native_producer_energy_from_summary.py \
  run_evalrun_native_producer_variants.py \
  update_evalset_native_producers.py \
  native_trt_from_benchmarkset.py
do
  cmp --silent "scripts/$name" \
    "onnx_splitpoint_tool/resources/remote_scripts/$name"
done

printf '%s\n' \
  "PASS v2.75.21 small acceptance" \
  "SKIP hardware, DFC, DX-COM and trtexec execution"
