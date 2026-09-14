from __future__ import annotations
import argparse, json, tempfile
from pathlib import Path

def run_smoke()->dict:
    checks=[]
    def check(name,fn):
        try: fn(); checks.append({'name':name,'ok':True})
        except Exception as exc: checks.append({'name':name,'ok':False,'error':f'{type(exc).__name__}: {exc}'})
    from . import __version__
    from .run_modes import RUN_MODE_SCHEMA_VERSION, default_run_modes_config
    check('version',lambda: (_ for _ in ()).throw(AssertionError(__version__)) if not (__version__.startswith('0.14.18+v60s') or __version__.startswith('0.14.19+v60t') or __version__.startswith('0.14.20+v60u') or __version__.startswith('0.14.21+v60v') or __version__.startswith('0.14.22+v60w') or __version__.startswith('0.14.23+v60x') or __version__.startswith('0.14.24+v60y') or __version__.startswith('2.61.0+v61e') or __version__.startswith('2.62.0') or __version__.startswith('2.63.0') or __version__.startswith('2.64.0') or __version__.startswith('2.65.0') or __version__.startswith('2.66.0') or __version__.startswith('2.67.0') or __version__.startswith('2.68.0') or __version__.startswith('2.69.6') or __version__.startswith('2.70.7') or __version__.startswith('2.70.8') or __version__.startswith('2.70.9') or __version__.startswith('2.70.10') or __version__.startswith('2.71.1') or __version__.startswith('2.71.2') or __version__.startswith('2.71.3') or __version__.startswith('2.71.4') or __version__.startswith(('2.72.0', '2.72.1', '2.72.2', '2.72.3', '2.72.4', '2.72.5', '2.72.6', '2.73.0', '2.73.6', '2.73.7', '2.75.2', '2.75.7', '2.75.8', '2.75.10', '2.75.11', '2.75.12', '2.75.16', '2.75.17', '2.75.20', '2.75.28', '2.75.30', '2.75.31'))) else None)
    check('schema',lambda: (_ for _ in ()).throw(AssertionError(RUN_MODE_SCHEMA_VERSION)) if RUN_MODE_SCHEMA_VERSION not in {4, 5, 7, 8, 9, 10, 11, 12} else None)
    check('run_modes',lambda: default_run_modes_config()['modes']['standard']['build']['artifact_store']['enabled'])
    def artifact_check():
        from .artifact_store import ArtifactStore
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); src=root/'a.hef'; src.write_bytes(b'a'); store=ArtifactStore(root/'store')
            rec=store.register(source_path=src,kind='hailo_hef',contract={'a':1}); assert store.lookup(kind='hailo_hef',contract={'a':1}); store.materialize(rec,root/'b.hef')
    check('artifact_store',artifact_check)
    def scheduler_check():
        from .build_scheduler import BuildScheduler,BuildTaskSpec
        with BuildScheduler(max_workers=2,cpu_tokens=4,family_limits={'a':1,'b':1}) as s:
            assert s.submit(BuildTaskSpec('a','a'),lambda:1).result()==1
    check('build_scheduler',scheduler_check)
    root=Path(__file__).resolve().parents[1]
    check('hailo_bridge',lambda: (_ for _ in ()).throw(AssertionError()) if 'def hailo_build_hef(*args, **kwargs)' not in (root/'onnx_splitpoint_tool/hailo_backend.py').read_text() else None)
    check('deepx_bridge',lambda: (_ for _ in ()).throw(AssertionError()) if 'def _find_cached_dxnn(*args, **kwargs)' not in (root/'onnx_splitpoint_tool/workflow/deepx_build_binding.py').read_text() else None)
    check('parallel_hailo',lambda: (_ for _ in ()).throw(AssertionError()) if '_run_hailo_target_builds_v60s' not in (root/'onnx_splitpoint_tool/benchmark/services.py').read_text() else None)
    check('deepx_prefetch',lambda: (_ for _ in ()).throw(AssertionError()) if '_v60s_start_deepx_prefetch' not in (root/'onnx_splitpoint_tool/workflow/legacy_benchmarkset_binding.py').read_text() else None)
    return {'ok':all(x['ok'] for x in checks),'passed':sum(x['ok'] for x in checks),'failed':sum(not x['ok'] for x in checks),'checks':checks}

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument('--json',default=''); ns=p.parse_args(argv); result=run_smoke()
    print(f"v60s smoke: {'ok' if result['ok'] else 'FAILED'} ({result['passed']} passed, {result['failed']} failed)")
    if ns.json: Path(ns.json).write_text(json.dumps(result,indent=2)+'\n')
    return 0 if result['ok'] else 1
if __name__=='__main__': raise SystemExit(main())
