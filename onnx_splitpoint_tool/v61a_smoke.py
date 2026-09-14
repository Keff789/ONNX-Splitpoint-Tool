from __future__ import annotations
import argparse,json,tempfile
from pathlib import Path
from . import __version__
from .native_full_quality import normalise_evaluation_profile
from .native_energy_reporting import write_native_energy_reports

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--json',default=''); ns=ap.parse_args(); rows=[]
    def check(name,fn):
        try: fn(); rows.append({'name':name,'ok':True})
        except Exception as e: rows.append({'name':name,'ok':False,'error':f'{type(e).__name__}: {e}'})
    check('version',lambda: (_ for _ in ()).throw(AssertionError(__version__)) if __version__ not in {'0.14.26+v61a.nativefullenergyprogress','0.14.27+v61b.nativeintegrationfix', '0.14.28+v61c.nativefullpairedenergyfix', '0.14.29+v61d.nativefullsemanticfix', '0.14.30+v61e.standardguifix', '2.61.0+v61e', '2.62.0', '2.63.0', '2.64.0', '2.65.0', '2.66.0', '2.67.0', '2.68.0', '2.69.6', '2.70.7', '2.70.8', '2.70.9', '2.70.10', '2.71.1', '2.71.2', '2.71.3', '2.71.4', '2.72.0', '2.72.1', '2.72.2', '2.72.3', '2.72.4', '2.72.5', '2.72.6', '2.73.0', '2.73.6', '2.73.7', '2.75.2', '2.75.3', '2.75.7', '2.75.8', '2.75.9', '2.75.10', '2.75.11', '2.75.12', '2.75.13', '2.75.14', '2.75.16', '2.75.17', '2.75.20', '2.75.27', '2.75.28', '2.75.30', '2.75.31'} else None)
    def matrix():
        p={'run_profiles':[{'id':'ort_tensorrt','full':'tensorrt'},{'id':'hailo8','full':'hailo8'},{'id':'hailo8_to_trt'},{'id':'hailo10','full':'hailo10'},{'id':'hailo10_to_tensorrt'},{'id':'deepx_m1_full','full':'deepx_m1'},{'id':'deepx_m1_to_tensorrt'}],'native_producers':{'enabled':True,'energy':{'enabled':True}},'execution_preset':{'overrides':{'native_enabled':True,'energy_enabled':True}},'energy':{}}
        normalise_evaluation_profile(p); m=p['native_producers']['full_baselines']['backends_by_producer']; assert m=={'hailo8':['hailo8','tensorrt'],'hailo10h':['hailo10h','tensorrt'],'deepx':['deepx','tensorrt']},m; assert p['native_producers']['energy']['mode']=='measure'; assert p['native_producers']['energy']['include_full_baselines']
    check('producer_specific_full_matrix',matrix)
    def reporting():
        with tempfile.TemporaryDirectory() as td:
            r=Path(td); src=r/'reports/native_energy_measurements'; src.mkdir(parents=True); (src/'native_producer_energy_results.json').write_text(json.dumps({'rows':[]})); o=r/'reports/scientific'; x=write_native_energy_reports(r,o); assert x['row_count']==0; assert (o/'screening_energy_observations.csv').exists()
    check('native_energy_reporting',reporting)
    out={'ok':all(r['ok'] for r in rows),'passed':sum(r['ok'] for r in rows),'failed':sum(not r['ok'] for r in rows),'checks':rows}
    if ns.json: Path(ns.json).write_text(json.dumps(out,indent=2))
    print(f"v61a smoke: {'ok' if out['ok'] else 'failed'} ({out['passed']} passed, {out['failed']} failed)")
    return 0 if out['ok'] else 1
if __name__=='__main__': raise SystemExit(main())
