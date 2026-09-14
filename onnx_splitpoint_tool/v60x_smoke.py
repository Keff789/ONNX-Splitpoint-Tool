from __future__ import annotations

"""Hardware-free regression smoke for v60x Native evidence fixes."""

import argparse
import importlib.util
import json
import tempfile
from pathlib import Path
from typing import Any, Sequence

from . import __version__
from .v60w_smoke import _run_checks as _base_checks
from .workflow.runner import WORKFLOW_VERSION, _native_concise_summary_v60w


def _load_script(name: str):
    root = Path(__file__).resolve().parents[1]
    path = root / 'scripts' / name
    if not path.is_file():
        path = Path(__file__).resolve().parent / 'resources' / 'remote_scripts' / name
    spec = importlib.util.spec_from_file_location('v60x_' + name.replace('.py',''), path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_checks() -> list[dict[str, Any]]:
    checks = list(_base_checks())
    def run(name: str, fn) -> None:
        try:
            fn(); checks.append({'name': name, 'status': 'pass'})
        except Exception as exc:
            checks.append({'name': name, 'status': 'fail', 'error': f'{type(exc).__name__}: {exc}'})

    run('v60x_version', lambda: (_ for _ in ()).throw(AssertionError(__version__)) if __version__ not in {'0.14.23+v60x.nativeevidencefix', '0.14.25+v60z.nativefullquality', '0.14.26+v61a.nativefullenergyprogress', '0.14.27+v61b.nativeintegrationfix', '0.14.28+v61c.nativefullpairedenergyfix', '0.14.29+v61d.nativefullsemanticfix', '0.14.30+v61e.standardguifix', '2.61.0+v61e', '2.62.0', '2.63.0', '2.64.0', '2.65.0', '2.66.0', '2.67.0', '2.68.0', '2.69.6', '2.70.7', '2.70.8', '2.70.9', '2.70.10', '2.71.1', '2.71.2', '2.71.3', '2.71.4', '2.72.0', '2.72.1', '2.72.2', '2.72.3', '2.72.4', '2.72.5', '2.72.6', '2.73.0', '2.73.6', '2.73.7', '2.75.2', '2.75.3', '2.75.7', '2.75.8', '2.75.9', '2.75.10', '2.75.11', '2.75.12', '2.75.13', '2.75.14', '2.75.16', '2.75.17', '2.75.20', '2.75.21', '2.75.22', '2.75.24', '2.75.26', '2.75.27', '2.75.28', '2.75.30', '2.75.31'} else None)
    run('v60x_workflow', lambda: (_ for _ in ()).throw(AssertionError(WORKFLOW_VERSION)) if WORKFLOW_VERSION not in {'v60x-native-evidence-contract-fixes', 'v60z-native-full-energy-quality-evidence', 'v61a-native-full-energy-progress-fixes', 'v61b-native-integration-live-energy-fixes', 'v61c-native-full-paired-energy-fixes', 'v61d-native-full-semantic-hailo8-fixes', 'v61e-standard-run-gui-diagnostics-fixes', 'v2.61e-campaign-contract-hardening', 'v2.62-window-validation-native-binding', 'v2.63-campaign-ready', 'v2.64-campaign-ready', 'v2.65-campaign-ready', 'v2.66-campaign-ready', 'v2.67-campaign-ready', 'v2.68-level-playing-field', 'v2.69f-hardware-smoke-native-energy-repair', 'v2.70g-native-validation-bridge-repair', 'v2.70h-remote-suite-bootstrap-repair', 'v2.70i-native-full-evidence-repair', 'v2.70j-native-evidence-reporting-repair', 'v2.71.1-live-evidence-binding-repair', 'v2.71.2-evidence-status-energy-resume-repair', 'v2.71.3-resume-artifact-restage-repair', 'v2.72.0-completed-detection-evidence-contracts', 'v2.72.1-mixed-runtime-energy-contract-repair', 'v2.72.2-offline-energy-completed-consumer-repair', 'v2.72.3-campaign-gate-quality-binding-repair', 'v2.72.4-campaign-gate-quality-contract-mirror-repair', 'v2.72.5-campaign-gate-hailo8-resume-contract-repair', 'v2.72.6-campaign-gate-hailo8-preflight-artifact-rehydration-repair', 'v2.73.0-completed-quality-evidence-repair', 'v2.73.6-single-writer-process-tree-cancellation', 'v2.73.7-atomic-stage-energy-checkpoints', 'v2.75.2-native-runtime-closure-partial-energy-repair', 'v2.75.3-cache-verify-only-gate', 'v2.75.7-cache-canary-remote-runtime-closure-repair', 'v2.75.8-native-runtime-holdout-contract-repair', 'v2.75.9-smoke-cache-reuse-repair', 'v2.75.10-historical-cache-multihit-replay-repair', 'v2.75.11-canonical-native-result-verification-repair', 'v2.75.12-canonical-native-result-status-alias-repair', 'v2.75.13-native-full-contract-line-repair', 'v2.75.14-vendor-full-nine-row-repair', 'v2.75.16-read-only-run-storage-admission', 'v2.75.17-vendor-quality-diagnostic-hailo-alias-repair', 'v2.75.20-remote-boundary-plan-finalization-repair', 'v2.75.21-setup-local-tensorrt-quality-dispatch-repair', 'v2.75.22-quality-replay-reporting-repair', 'v2.75.24-standard-hef-preupload-gate', 'v2.75.26-trt-quality-join-mean-iou-repair', 'v2.75.27-final-claim-scope-cache-key-repair', 'v2.75.28-final-campaign-bootstrap-energy-provenance-repair', 'v2.75.30-score-independent-native-ranking-audit', 'v2.75.31-compact-debug-ranking-gui-repair'} else None)

    def successful_row_has_no_failure() -> None:
        with tempfile.TemporaryDirectory() as td:
            reports=Path(td); (reports/'native_validation').mkdir()
            (reports/'native_producer_combined_summary.json').write_text(json.dumps({'rows':[{'model':'m','backend':'b','case':'c','ok':True,'fps_makespan':1.0,'failure_reason':'stale_fallback'}]}),encoding='utf-8')
            (reports/'native_validation/native_producer_validation_summary.json').write_text(json.dumps({'rows':[]}),encoding='utf-8')
            _, rows = _native_concise_summary_v60w(reports)
            assert rows[0]['runtime_status'] == 'ok' and rows[0]['failure_reason'] == ''
    run('successful_native_row_no_false_failure', successful_row_has_no_failure)

    def nested_manifest_image() -> None:
        mod=_load_script('native_fifo_smoke_matrix.py')
        with tempfile.TemporaryDirectory() as td:
            bs=Path(td); ds=bs/'resources/validation/classification/subset'; img=ds/'images/n0001/a.JPEG'
            img.parent.mkdir(parents=True); img.write_bytes(b'x')
            (ds/'manifest.json').write_text(json.dumps({'samples':[{'image':'images/n0001/a.JPEG'}]}),encoding='utf-8')
            assert mod._default_image(bs) == img.resolve()
    run('nested_imagenet_manifest_image', nested_manifest_image)

    def decoded_contract() -> None:
        mod=_load_script('native_producer_validate_visualize.py')
        with tempfile.TemporaryDirectory() as td:
            bs=Path(td); out=bs/'native_pipeline/b001/native_outputs/native_outputs_manifest.json'
            out.parent.mkdir(parents=True)
            (bs/'benchmark_set.json').write_text(json.dumps({'benchmark_task':'detection'}),encoding='utf-8')
            out.write_text(json.dumps({'outputs':[{'shape':[1,300,6]}]}),encoding='utf-8')
            family, source=mod._expected_detection_contract(out)
            assert family == 'decoded_nms' and source in {'benchmark_task_and_runtime_output_shape','explicit_metadata'}
    run('decoded_detection_contract_metadata', decoded_contract)

    return checks


def run_smoke() -> dict[str, Any]:
    checks=_run_checks(); failed=sum(r['status']!='pass' for r in checks)
    return {'schema':'onnx-splitpoint/v60x-smoke','tool_version':__version__,'workflow_version':WORKFLOW_VERSION,'status':'ok' if not failed else 'failed','passed':len(checks)-failed,'failed':failed,'checks':checks}


def main(argv: Sequence[str] | None = None) -> int:
    ap=argparse.ArgumentParser(prog='onnx-splitpoint-smoke-v60x'); ap.add_argument('--json',default=''); ns=ap.parse_args(list(argv) if argv is not None else None)
    payload=run_smoke()
    if ns.json:
        p=Path(ns.json).expanduser().resolve(); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(payload,indent=2)+'\n',encoding='utf-8')
    print(f"v60x smoke: {'ok' if payload['status']=='ok' else 'FAILED'} ({payload['passed']} passed, {payload['failed']} failed)")
    return 0 if payload['status']=='ok' else 1

if __name__ == '__main__':
    raise SystemExit(main())
