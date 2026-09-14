from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from onnx_splitpoint_tool import __version__
from onnx_splitpoint_tool.workflow.runner import WORKFLOW_VERSION, _native_concise_summary_v60w

ROOT=Path(__file__).resolve().parents[1]

def _load_script(name: str):
    path=ROOT/'scripts'/name
    spec=importlib.util.spec_from_file_location('test_v60x_'+name.replace('.py',''), path)
    assert spec and spec.loader
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod

def test_versions_v60x():
    assert __version__ in {'0.14.23+v60x.nativeevidencefix', '0.14.25+v60z.nativefullquality', '0.14.26+v61a.nativefullenergyprogress', '0.14.27+v61b.nativeintegrationfix', '0.14.28+v61c.nativefullpairedenergyfix', '0.14.29+v61d.nativefullsemanticfix', '0.14.30+v61e.standardguifix', '2.61.0+v61e', '2.62.0', '2.63.0', '2.64.0', '2.65.0', '2.66.0', '2.67.0', '2.68.0', '2.69.6', "2.70.6", "2.70.7", "2.70.8", "2.70.9", "2.70.10", "2.71.1", "2.71.2", "2.71.3", "2.71.4", "2.72.0", "2.72.1", "2.72.2", "2.72.3", "2.72.4", "2.72.5", "2.72.6", "2.72.7", "2.73.0", "2.73.6", "2.73.7", "2.75.2", "2.75.3", "2.75.7", "2.75.8", "2.75.9", "2.75.10", "2.75.11", "2.75.12", "2.75.13", "2.75.14", "2.75.15", "2.75.16", "2.75.17", "2.75.20", "2.75.21", "2.75.22", "2.75.24", "2.75.26", "2.75.27", "2.75.28", "2.75.30", "2.75.31", "2.75.32", "2.75.38", "2.75.39", "2.75.40", "2.75.41", "2.75.42", "2.75.46", "2.75.47"}
    assert WORKFLOW_VERSION in {'v60x-native-evidence-contract-fixes', 'v60z-native-full-energy-quality-evidence', 'v61a-native-full-energy-progress-fixes', 'v61b-native-integration-live-energy-fixes', 'v61c-native-full-paired-energy-fixes', 'v61d-native-full-semantic-hailo8-fixes', 'v61e-standard-run-gui-diagnostics-fixes', 'v2.61e-campaign-contract-hardening', 'v2.62-window-validation-native-binding', 'v2.63-campaign-ready', 'v2.64-campaign-ready', 'v2.65-campaign-ready', 'v2.66-campaign-ready', 'v2.67-campaign-ready', 'v2.68-level-playing-field', 'v2.69f-hardware-smoke-native-energy-repair', "v2.70f-legacy-log-callback-repair", "v2.70g-native-validation-bridge-repair", "v2.70h-remote-suite-bootstrap-repair", "v2.70i-native-full-evidence-repair", "v2.70j-native-evidence-reporting-repair", "v2.71.1-live-evidence-binding-repair", "v2.71.2-evidence-status-energy-resume-repair", "v2.71.3-resume-artifact-restage-repair", "v2.71.4-resume-cleanup-root-rehydration", "v2.72.0-completed-detection-evidence-contracts", "v2.72.1-mixed-runtime-energy-contract-repair", "v2.72.2-offline-energy-completed-consumer-repair", "v2.72.3-campaign-gate-quality-binding-repair", "v2.72.4-campaign-gate-quality-contract-mirror-repair", "v2.72.5-campaign-gate-hailo8-resume-contract-repair", "v2.72.6-campaign-gate-hailo8-preflight-artifact-rehydration-repair", "v2.72.7-native-energy-runtime-success-admission-repair", "v2.73.0-completed-quality-evidence-repair", "v2.73.6-single-writer-process-tree-cancellation", "v2.73.7-atomic-stage-energy-checkpoints", "v2.75.2-native-runtime-closure-partial-energy-repair", "v2.75.3-cache-verify-only-gate", "v2.75.7-cache-canary-remote-runtime-closure-repair", "v2.75.8-native-runtime-holdout-contract-repair", "v2.75.9-smoke-cache-reuse-repair", "v2.75.10-historical-cache-multihit-replay-repair", "v2.75.11-canonical-native-result-verification-repair", "v2.75.12-canonical-native-result-status-alias-repair", "v2.75.13-native-full-contract-line-repair", "v2.75.14-vendor-full-nine-row-repair", "v2.75.15-vendor-full-field-contract-repair", "v2.75.16-read-only-run-storage-admission", "v2.75.17-vendor-quality-diagnostic-hailo-alias-repair", "v2.75.20-remote-boundary-plan-finalization-repair", "v2.75.21-setup-local-tensorrt-quality-dispatch-repair", "v2.75.22-quality-replay-reporting-repair", "v2.75.24-standard-hef-preupload-gate", "v2.75.26-trt-quality-join-mean-iou-repair", "v2.75.27-final-claim-scope-cache-key-repair", "v2.75.28-final-campaign-bootstrap-energy-provenance-repair", "v2.75.30-score-independent-native-ranking-audit", "v2.75.31-compact-debug-ranking-gui-repair", "v2.75.32-audit-ranking-e2e-repair", "v2.75.38-deepx-full-quality-dxnn-dispatch-repair", "v2.75.39-pre-mutation-remote-failure-quarantine-repair", "v2.75.40-deepx-preprocessing-ab-and-full-only-quality-export-repair", "v2.75.41-deepx-calibration-500-vs-1000-canary", "v2.75.42-real-evidence-and-quality-companion-repair", "v2.75.46-native-full-onnx-attestation-standard-quality-projection", "v2.75.47-yolov7-decoder-contract-debug-pack-audit-intent"}

def test_native_diagnostics_do_not_mark_success_as_failure():
    mod=_load_script('native_producer_final_report.py')
    d=mod._diagnostic_fields({'ok':True},{'ok':True},fallback='should_not_appear',ok=True)
    assert d['failure_reason']==''
    assert d['status_detail']=='ok'
    assert d['error']==''

def test_concise_summary_clears_stale_failure_on_ok_row(tmp_path: Path):
    reports=tmp_path/'reports'; (reports/'native_validation').mkdir(parents=True)
    (reports/'native_producer_combined_summary.json').write_text(json.dumps({'rows':[{'model':'m','backend':'b','case':'c','ok':True,'failure_reason':'stale'}]}),encoding='utf-8')
    (reports/'native_validation/native_producer_validation_summary.json').write_text(json.dumps({'rows':[]}),encoding='utf-8')
    _, rows=_native_concise_summary_v60w(reports)
    assert rows[0]['runtime_status']=='ok'
    assert rows[0]['failure_reason']==''

def test_nested_classification_subset_selects_exact_manifest_image(tmp_path: Path):
    mod=_load_script('native_fifo_smoke_matrix.py')
    ds=tmp_path/'resources/validation/classification/subset'; img=ds/'images/n123/a.JPEG'
    img.parent.mkdir(parents=True); img.write_bytes(b'x')
    (ds/'manifest.json').write_text(json.dumps({'samples':[{'image':'images/n123/a.JPEG'}]}),encoding='utf-8')
    assert mod._default_image(tmp_path)==img.resolve()

def test_hailo_runner_nested_manifest_helper(tmp_path: Path):
    mod=_load_script('native_hailo_trt_fifo_from_benchmarkset.py')
    ds=tmp_path/'resources/validation/classification/subset'; img=ds/'images/n123/a.JPEG'
    img.parent.mkdir(parents=True); img.write_bytes(b'x')
    (ds/'manifest.json').write_text(json.dumps({'samples':[{'image':'images/n123/a.JPEG'}]}),encoding='utf-8')
    assert mod._default_image(tmp_path)==img.resolve()

def test_detection_contract_shape_only_explicit_metadata_fails_closed(tmp_path: Path):
    mod=_load_script('native_producer_validate_visualize.py')
    p=tmp_path/'native_outputs_manifest.json'
    p.write_text(json.dumps({'task':'detection','output_format':'bn6_detections','contract_family':'decoded_nms','outputs':[{'shape':[1,300,6]}]}),encoding='utf-8')
    assert mod._expected_detection_contract(p)==('unknown','metadata_unavailable')

def test_detection_contract_legacy_task_shape_no_longer_claims_nms(tmp_path: Path):
    mod=_load_script('native_producer_validate_visualize.py')
    p=tmp_path/'native_pipeline/b001/native_outputs/native_outputs_manifest.json'; p.parent.mkdir(parents=True)
    (tmp_path/'benchmark_set.json').write_text(json.dumps({'benchmark_task':'detection'}),encoding='utf-8')
    p.write_text(json.dumps({'outputs':[{'shape':[1,300,6]}]}),encoding='utf-8')
    assert mod._expected_detection_contract(p)==('unknown','metadata_unavailable')

def test_raw_contract_is_not_guessed_from_arbitrary_detection_shape(tmp_path: Path):
    mod=_load_script('native_producer_validate_visualize.py')
    p=tmp_path/'native_pipeline/b001/native_outputs/native_outputs_manifest.json'; p.parent.mkdir(parents=True)
    (tmp_path/'benchmark_set.json').write_text(json.dumps({'benchmark_task':'detection'}),encoding='utf-8')
    p.write_text(json.dumps({'outputs':[{'shape':[1,84,8400]}]}),encoding='utf-8')
    assert mod._expected_detection_contract(p)==('unknown','metadata_unavailable')

def test_output_manifest_writers_persist_contract_metadata():
    for name in ('native_deepx_trt_e2e_from_benchmarkset.py','native_hailo10_trt_e2e_from_benchmarkset.py'):
        text=(ROOT/'scripts'/name).read_text(encoding='utf-8')
        assert 'runtime_output_contract' in text
    text=(ROOT/'scripts/native_hailo_trt_fifo_from_benchmarkset.py').read_text(encoding='utf-8')
    assert '_annotate_output_contract' in text
    assert 'load_manifest_outputs' in text

def test_debug_pack_excludes_native_tensor_dumps_by_default():
    text=(ROOT/'onnx_splitpoint_tool/workflow/debug_pack.py').read_text(encoding='utf-8')
    assert '".bin"' in text
    assert '"tensor .bin and model/runtime binaries"' in text
