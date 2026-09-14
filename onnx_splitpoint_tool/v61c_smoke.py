from __future__ import annotations
import argparse, json, subprocess, sys, tempfile
from pathlib import Path
from . import __version__
from .workflow.runner import WORKFLOW_VERSION, _native_expected_full_rows_v61b, _native_full_backends_by_producer_v61b

EXPECTED_VERSION = "0.14.28+v61c.nativefullpairedenergyfix", "0.14.29+v61d.nativefullsemanticfix", "0.14.30+v61e.standardguifix", "2.61.0+v61e", "2.62.0", "2.63.0", "2.64.0", "2.65.0", "2.66.0", "2.67.0", "2.68.0", "2.69.6", "2.70.6", "2.70.7", "2.70.8", "2.70.9", "2.70.10", "2.71.1", "2.71.2", "2.71.3", "2.71.4", "2.72.0", "2.72.1", "2.72.2", "2.72.3", "2.72.4", "2.72.5", "2.72.6", "2.73.0", "2.73.6", "2.73.7", "2.75.2", "2.75.7", "2.75.8", "2.75.10", "2.75.11", "2.75.12", "2.75.14", "2.75.15", "2.75.16", "2.75.17", "2.75.20", "2.75.21", "2.75.22", "2.75.24", "2.75.26"
EXPECTED_WORKFLOW = "v61c-native-full-paired-energy-fixes", "v61d-native-full-semantic-hailo8-fixes", "v61e-standard-run-gui-diagnostics-fixes", "v2.61e-campaign-contract-hardening", "v2.62-window-validation-native-binding", "v2.63-campaign-ready", "v2.64-campaign-ready", "v2.65-campaign-ready", "v2.66-campaign-ready", "v2.67-campaign-ready", "v2.68-level-playing-field", "v2.69f-hardware-smoke-native-energy-repair", "v2.70f-legacy-log-callback-repair", "v2.70g-native-validation-bridge-repair", "v2.70h-remote-suite-bootstrap-repair", "v2.70i-native-full-evidence-repair", "v2.70j-native-evidence-reporting-repair", "v2.71.1-live-evidence-binding-repair", "v2.71.2-evidence-status-energy-resume-repair", "v2.71.3-resume-artifact-restage-repair", "v2.71.4-resume-cleanup-root-rehydration", "v2.72.0-completed-detection-evidence-contracts", "v2.72.1-mixed-runtime-energy-contract-repair", "v2.72.2-offline-energy-completed-consumer-repair", "v2.72.3-campaign-gate-quality-binding-repair", "v2.72.4-campaign-gate-quality-contract-mirror-repair", "v2.72.5-campaign-gate-hailo8-resume-contract-repair", "v2.72.6-campaign-gate-hailo8-preflight-artifact-rehydration-repair", "v2.73.0-completed-quality-evidence-repair", "v2.73.6-single-writer-process-tree-cancellation", "v2.73.7-atomic-stage-energy-checkpoints", "v2.75.2-native-runtime-closure-partial-energy-repair", "v2.75.3-cache-verify-only-gate", "v2.75.7-cache-canary-remote-runtime-closure-repair", "v2.75.8-native-runtime-holdout-contract-repair", "v2.75.10-historical-cache-multihit-replay-repair", "v2.75.11-canonical-native-result-verification-repair", "v2.75.12-canonical-native-result-status-alias-repair", "v2.75.14-vendor-full-nine-row-repair", "v2.75.15-vendor-full-field-contract-repair", "v2.75.16-read-only-run-storage-admission", "v2.75.17-vendor-quality-diagnostic-hailo-alias-repair", "v2.75.20-remote-boundary-plan-finalization-repair", "v2.75.21-setup-local-tensorrt-quality-dispatch-repair", "v2.75.22-quality-replay-reporting-repair", "v2.75.24-standard-hef-preupload-gate", "v2.75.26-trt-quality-join-mean-iou-repair"

def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument('--json',default=''); ns=ap.parse_args()
    checks=[]
    def check(name,fn):
        try: fn(); checks.append({'name':name,'ok':True})
        except Exception as exc: checks.append({'name':name,'ok':False,'error':f'{type(exc).__name__}: {exc}'})
    check('version', lambda: (_ for _ in ()).throw(AssertionError(__version__)) if __version__ not in EXPECTED_VERSION else None)
    check('workflow', lambda: (_ for _ in ()).throw(AssertionError(WORKFLOW_VERSION)) if WORKFLOW_VERSION not in EXPECTED_WORKFLOW else None)
    def matrix():
        m=_native_full_backends_by_producer_v61b({'enabled':True,'backends_by_producer':{'hailo8':['hailo8','tensorrt'],'hailo10h':['hailo10h','tensorrt'],'deepx':['deepx','tensorrt']}},['hailo8','hailo10h','deepx'])
        r=_native_expected_full_rows_v61b(['resnet50','yolo26s'],m,{'hailo8':'h8','hailo10h':'h10','deepx':'dx'})
        assert len(r)==12 and len({(x['setup_id'],x['model'],x['backend']) for x in r})==12
    check('producer_local_full_matrix',matrix)
    def remote_full_standalone():
        script=Path(__file__).resolve().parent/'resources'/'remote_scripts'/'native_full_baseline_eval_runner.py'
        with tempfile.TemporaryDirectory() as td:
            dst=Path(td)/script.name; dst.write_bytes(script.read_bytes())
            cp=subprocess.run([sys.executable,str(dst),'--help'],cwd=td,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=30)
            assert cp.returncode==0,(cp.stdout,cp.stderr)
    check('remote_full_standalone_import',remote_full_standalone)
    def paired_energy_plan():
        root=Path(__file__).resolve().parents[1]
        script=root/'scripts'/'native_producer_energy_plan.py'
        if not script.is_file(): script=Path(__file__).resolve().parent/'resources'/'remote_scripts'/'native_producer_energy_plan.py'
        with tempfile.TemporaryDirectory() as td:
            d=Path(td); summary=d/'summary.json'; validation=d/'validation.json'; out=d/'out'
            rows=[
              {'ok':True,'backend':'hailo8_to_trt','model':'m','case':'b1','precision':'p','setup_id':'h8','fps_makespan':10},
              {'ok':True,'backend':'native_full_hailo8','model':'m','case':'full','precision':'p','setup_id':'h8','comparison_backend':'hailo8','fps_makespan':8},
              {'ok':True,'backend':'native_full_tensorrt','model':'m','case':'full','precision':'p','setup_id':'h8','comparison_backend':'hailo8','fps_makespan':12},
              {'ok':True,'backend':'deepx_to_trt','model':'m','case':'b2','precision':'p','setup_id':'dx','fps_makespan':5},
            ]
            summary.write_text(json.dumps({'rows':rows}))
            validation.write_text(json.dumps({'rows':[{'backend':'hailo8_to_trt','model':'m','case':'b1','precision':'p','task':'classification','top1_match':True,'contract_consistent':True,'claim_ok':True,'semantic_ok':True},{'backend':'deepx_to_trt','model':'m','case':'b2','precision':'p','task':'classification','top1_match':True,'contract_consistent':True,'claim_ok':True,'semantic_ok':True}]}))
            cp=subprocess.run([sys.executable,str(script),'--summary',str(summary),'--validation-summary',str(validation),'--out-dir',str(out),'--hailo8-ssh','host','--deepx-ssh','host','--runs','1'],text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=30)
            assert cp.returncode==0,(cp.stdout,cp.stderr)
            payload=json.loads((out/'native_producer_energy_plan.json').read_text())
            assert len(payload['rows'])==3,payload
            assert any(x.get('reason')=='pair_baseline_missing' for x in payload['excluded_rows'])
            assert all('--runs 1' in x['measure_command'] for x in payload['rows'])
    check('paired_energy_plan_and_repeat',paired_energy_plan)
    result={'ok':all(x['ok'] for x in checks),'passed':sum(bool(x['ok']) for x in checks),'failed':sum(not bool(x['ok']) for x in checks),'checks':checks}
    if ns.json: Path(ns.json).write_text(json.dumps(result,indent=2))
    print(f"v61c smoke: {'ok' if result['ok'] else 'failed'} ({result['passed']} passed, {result['failed']} failed)")
    return 0 if result['ok'] else 1
if __name__=='__main__': raise SystemExit(main())
