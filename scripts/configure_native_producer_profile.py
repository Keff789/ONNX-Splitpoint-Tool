#!/usr/bin/env python3
"""Patch an Evaluation Workflow YAML profile with native_producers settings.

This helper is intentionally conservative: it only updates the native_producers
block and leaves all other profile settings untouched. It is useful until the GUI
has first-class controls for the native producer stage.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

def _load_yaml(path: Path):
    try:
        import yaml
    except Exception as e:
        raise SystemExit(f"PyYAML is required for this helper: {e}")
    if path.exists():
        data = yaml.safe_load(path.read_text(encoding='utf-8'))
        if data is None: data = {}
        if not isinstance(data, dict):
            raise SystemExit(f"Profile root must be a mapping: {path}")
        return data, yaml
    return {}, yaml

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--profile', required=True)
    ap.add_argument('--backends', default='hailo8,hailo10h,deepx')
    ap.add_argument('--case-policy', default='all_accepted', choices=['all_accepted','case_map_only','preferred_then_backfill'])
    ap.add_argument('--precision', default='uint8_cast_fp16')
    ap.add_argument('--frames', type=int, default=1000)
    ap.add_argument('--warmup', type=int, default=100)
    ap.add_argument('--repetitions', type=int, default=3, help='Independent performance repetitions. Default: Standard contract (3).')
    ap.add_argument('--queue-depth', type=int, default=3)
    ap.add_argument('--inflight', type=int, default=8)
    ap.add_argument('--hailo-format', default='uint8')
    ap.add_argument('--remote-root', default='/home/nx/native_fifo_evalsets')
    ap.add_argument('--remote-tool-dir', default='/home/nx/ONNX-Splitpoint-Tool')
    ap.add_argument('--hailo8-ssh', default='nx@192.168.0.104')
    ap.add_argument('--hailo10-ssh', default='nx@192.168.0.145')
    ap.add_argument('--deepx-ssh', default='nx@192.168.0.102')
    ap.add_argument('--case-map-json', default='', help='Optional JSON dict, e.g. {"resnet50":["b082"]}')
    ap.add_argument('--native-telemetry-label', default='profile_standard', help='Stable label for non-blocking pre/post host telemetry evidence.')
    ap.add_argument('--disable', action='store_true')
    ap.add_argument('--out', default='', help='Write to another profile path instead of modifying --profile')
    ns = ap.parse_args()
    if ns.repetitions < 1:
        ap.error('--repetitions must be >= 1')
    profile=Path(ns.profile).expanduser().resolve()
    data,yaml=_load_yaml(profile)
    backends=[b.strip() for b in ns.backends.split(',') if b.strip()]
    cfg={
        'enabled': not ns.disable,
        'backends': backends,
        'case_policy': ns.case_policy,
        'precision': ns.precision,
        'frames': ns.frames,
        'warmup': ns.warmup,
        'repetitions': ns.repetitions,
        'performance_aggregation': 'median_ci95',
        'queue_depth': ns.queue_depth,
        'inflight': ns.inflight,
        'hailo_format': ns.hailo_format,
        'remote_root': ns.remote_root,
        'remote_tool_dir': ns.remote_tool_dir,
        'build_missing_engines': True,
        'copy_benchmarksets': True,
        'strict_supported_only': True,
        'telemetry_label': ns.native_telemetry_label,
        'host_telemetry': {'enabled': True, 'mode': 'pre_post_non_blocking'},
        'remotes': {},
    }
    if 'hailo8' in backends:
        cfg['remotes']['hailo8']={'ssh': ns.hailo8_ssh}
    if 'hailo10h' in backends or 'hailo10' in backends:
        cfg['remotes']['hailo10h']={'ssh': ns.hailo10_ssh, 'env': 'export PYTHONNOUSERSITE=1; source ~/venvs/hailo10/bin/activate'}
    if 'deepx' in backends:
        cfg['remotes']['deepx']={'ssh': ns.deepx_ssh, 'env': 'source ~/venvs/deepx-runtime/bin/activate'}
    if ns.case_map_json.strip():
        cfg['case_map']=json.loads(ns.case_map_json)
    data['native_producers']=cfg
    out=Path(ns.out).expanduser().resolve() if ns.out else profile
    out.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding='utf-8')
    print(json.dumps({'ok': True, 'profile': str(out), 'native_producers': cfg}, indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
