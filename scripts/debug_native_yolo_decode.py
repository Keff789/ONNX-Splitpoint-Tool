#!/usr/bin/env python3
"""Debug YOLO/NMS postprocessing for a native output dump.

This tool does not produce a quality claim. It generates tensor statistics,
decode/threshold sweeps, optional overlays, and a markdown report so broken
postprocessing paths can be diagnosed quickly.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from scripts.native_producer_validate_visualize import _visualize_detection, _draw_boxes  # type: ignore

def main() -> int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--reference-report', default='')
    ap.add_argument('--eval-root', default='.')
    ap.add_argument('--image', default='')
    ap.add_argument('--max-detection-outputs', type=int, default=6)
    ns=ap.parse_args()
    manifest=Path(ns.manifest).expanduser().resolve(); out=Path(ns.out_dir).expanduser().resolve(); out.mkdir(parents=True, exist_ok=True)
    ref=Path(ns.reference_report).expanduser().resolve() if ns.reference_report else None
    if ref and not ref.is_file(): ref=None
    rep=_visualize_detection(manifest, out, ns.max_detection_outputs, ref, Path(ns.eval_root).expanduser().resolve())
    if ns.image:
        img=Path(ns.image).expanduser().resolve()
        if img.is_file():
            _draw_boxes(img, rep.get('detections') or [], out/'detection_boxes_overlay_explicit_image.png', title='native decoded explicit image')
            rep['explicit_image_overlay']=str(out/'detection_boxes_overlay_explicit_image.png')
            (out/'detection_visual_validation.json').write_text(json.dumps(rep, indent=2), encoding='utf-8')
    print(json.dumps({'ok':True,'out_dir':str(out),'decode_mode':rep.get('decode_mode'),'semantic_ok':rep.get('semantic_ok'),'boxes':len(rep.get('detections') or []),'debug_md':str(out/'detection_visual_validation.md')}, indent=2))
    return 0
if __name__ == '__main__': raise SystemExit(main())
