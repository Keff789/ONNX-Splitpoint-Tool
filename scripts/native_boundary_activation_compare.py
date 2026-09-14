#!/usr/bin/env python3
"""Compare a raw native boundary dump against the ONNX Part1 activation.

Diagnostic purpose: decide whether a raw uint8 boundary buffer is an affine
quantized version of the ONNX Part1 output and whether a simple layout transform
is missing.  This helps distinguish:
  * wrong dequant scale/zero-point
  * NCHW/NHWC memory layout mismatch
  * wrong producer output tensor / name mapping
  * truly incompatible split contract

The tool is intentionally conservative.  It reports correlations and fitted
affine parameters; it does not claim semantic correctness.
"""
from __future__ import annotations
import argparse, json, math, re
from pathlib import Path
from typing import Any
import numpy as np


def _read_json(p: Path) -> dict[str, Any]:
    return json.loads(p.read_text(encoding='utf-8'))


def _find_part1_onnx(bs: Path, case: str) -> Path:
    c = bs / case
    pats = [f"*_part1_*{case.lstrip('b')}*.onnx", "*part1*.onnx"]
    for pat in pats:
        xs = sorted(c.glob(pat))
        if xs:
            return xs[0]
    raise FileNotFoundError(f"part1 ONNX not found under {c}")


def _onnx_io(path: Path):
    import onnx
    model = onnx.load(str(path))
    def shape(v):
        dims=[]
        for d in v.type.tensor_type.shape.dim:
            dims.append(int(d.dim_value) if d.dim_value else -1)
        return dims
    inputs=[(x.name, shape(x)) for x in model.graph.input]
    outputs=[(x.name, shape(x)) for x in model.graph.output]
    return inputs, outputs


def _summ(x: np.ndarray):
    x=np.asarray(x)
    finite=np.isfinite(x)
    xf=x[finite]
    if xf.size==0:
        return {'shape':list(x.shape),'dtype':str(x.dtype),'finite':False}
    qs=np.quantile(xf,[0,.001,.01,.1,.5,.9,.99,.999,1]).tolist()
    return {'shape':list(x.shape),'dtype':str(x.dtype),'finite':bool(finite.all()),'min':float(xf.min()),'max':float(xf.max()),'mean':float(xf.mean()),'std':float(xf.std()),'quantiles':qs}


def _find_image(bs: Path, image_name: str | None) -> Path | None:
    if not image_name:
        return None
    p=Path(image_name)
    if p.is_file():
        return p.resolve()
    for root in [bs/'resources', bs, bs.parent, bs.parent/'resources']:
        if root.exists():
            hits=list(root.rglob(p.name))
            if hits:
                return hits[0].resolve()
    return None


def _image_from_report(report: Path | None) -> str | None:
    if not report or not report.is_file():
        return None
    # Use the same reference parser as native visual validation first.  The older
    # fallback below can pick the first dataset image, which is not necessarily
    # the image used by the visual/semantic reference row.
    try:
        from native_producer_validate_visualize import _reference_detections  # type: ignore
        _dets, img = _reference_detections(report)
        if img:
            return img
    except Exception:
        pass
    try:
        j=_read_json(report)
    except Exception:
        return None
    # Look in visual/native validation style first.
    for key in ('native_input_image','reference_image','image','input_image'):
        v=j.get(key)
        if isinstance(v,str) and v:
            return v
    # Generic validation_dataset fallback.
    vd=j.get('validation_dataset') if isinstance(j,dict) else None
    if isinstance(vd,dict):
        imgs=vd.get('images')
        if isinstance(imgs,list) and imgs:
            im=imgs[0]
            if isinstance(im,dict) and im.get('image'):
                return im.get('image')
            if isinstance(im,str):
                return im
        variants=vd.get('variants') or {}
        if isinstance(variants,dict):
            for variant in ('full','composed','part2','part1'):
                v=variants.get(variant) or {}
                ims=v.get('images') if isinstance(v,dict) else None
                if isinstance(ims,list) and ims:
                    im=ims[0]
                    if isinstance(im,dict) and im.get('image'):
                        return im.get('image')
                    if isinstance(im,str):
                        return im
    # viz json provenance
    viz=j.get('viz') if isinstance(j,dict) else None
    if isinstance(viz,dict):
        for variant in ('full','composed','part2','part1'):
            node=viz.get(variant) or {}
            js=node.get('json') if isinstance(node,dict) else None
            if isinstance(js,dict):
                prov=js.get('provenance') or {}
                if isinstance(prov,dict) and prov.get('image'):
                    return prov.get('image')
    return None



def _image_from_boundary_manifest(man: dict[str, Any]) -> str | None:
    v = man.get('input_image')
    if isinstance(v, str) and v:
        return v
    prov = man.get('provenance')
    if isinstance(prov, dict):
        v = prov.get('image')
        if isinstance(v, str) and v:
            return v
    return None

def _preprocess_mode_parts(scale: str) -> tuple[str, int, bool]:
    """Return (resize_mode, pad_value, normalize).

    The native Hailo8 FIFO runner does not feed the original JPEG into ONNX. It
    first creates an RGB letterboxed uint8 image. For boundary contract checks we
    must reproduce that exact model input, otherwise the correlation test can
    falsely conclude that the boundary tensor is unrelated.
    """
    mode = str(scale or 'native').strip().lower()
    if mode in {'native', 'native_letterbox', 'native_letterbox_norm', 'letterbox0_norm'}:
        return 'letterbox', 0, True
    if mode in {'native_raw', 'native_letterbox_raw', 'letterbox0_raw'}:
        return 'letterbox', 0, False
    if mode in {'yolo', 'yolo_letterbox', 'yolo_letterbox_norm', 'letterbox114_norm'}:
        return 'letterbox', 114, True
    if mode in {'yolo_raw', 'yolo_letterbox_raw', 'letterbox114_raw'}:
        return 'letterbox', 114, False
    if mode in {'resize_raw', 'raw'}:
        return 'resize', 0, False
    # Historical defaults: norm/auto/imagenet used direct resize + /255.
    return 'resize', 0, True


def _letterbox_image(img, w: int, h: int, pad: int):
    ow, oh = img.size
    if ow <= 0 or oh <= 0:
        raise RuntimeError(f'Invalid image size: {img.size}')
    gain = min(w / float(ow), h / float(oh))
    nw = max(1, int(round(ow * gain)))
    nh = max(1, int(round(oh * gain)))
    from PIL import Image as PILImage
    im_resized = img.resize((nw, nh))
    canvas = PILImage.new('RGB', (w, h), color=(int(pad), int(pad), int(pad)))
    canvas.paste(im_resized, ((w - nw) // 2, (h - nh) // 2))
    return canvas


def _preprocess(img_path: Path, input_shape: list[int], scale: str='native') -> np.ndarray:
    from PIL import Image
    img=Image.open(img_path).convert('RGB')
    resize_mode, pad_value, normalize = _preprocess_mode_parts(scale)
    if len(input_shape)==4 and input_shape[1] in (1,3):
        _,c,h,w=input_shape
        img = _letterbox_image(img, w, h, pad_value) if resize_mode == 'letterbox' else img.resize((w,h))
        arr=np.asarray(img).astype(np.float32)
        if normalize:
            arr=arr/255.0
        arr=np.transpose(arr,(2,0,1))[None]
        return arr.astype(np.float32)
    if len(input_shape)==4 and input_shape[-1] in (1,3):
        _,h,w,c=input_shape
        img = _letterbox_image(img, w, h, pad_value) if resize_mode == 'letterbox' else img.resize((w,h))
        arr=np.asarray(img).astype(np.float32)
        if normalize:
            arr=arr/255.0
        return arr[None].astype(np.float32)
    raise RuntimeError(f'unsupported input shape: {input_shape}')


def _input_dump_feed_from_manifest(man: dict[str, Any], input_shape: list[int], scale: str='native') -> np.ndarray | None:
    """Build the ORT feed from the exact preprocessed input dumped by native FIFO.

    This is the strongest comparison mode: it removes all Python/OpenCV/Pillow
    resize and padding differences from the boundary contract validator.
    """
    p = man.get('input_dump') or man.get('preprocessed_input_file')
    if not isinstance(p, str) or not p:
        return None
    pp = Path(p).expanduser()
    if not pp.is_absolute():
        base = Path(str(man.get('file') or '')).expanduser().parent
        pp = (base / pp).resolve()
    if not pp.is_file():
        return None
    shp = man.get('input_shape_hwc') or (man.get('preprocess') or {}).get('input_shape_hwc')
    if not shp:
        if len(input_shape)==4 and input_shape[1] in (1,3):
            shp = [int(input_shape[2]), int(input_shape[3]), int(input_shape[1])]
        elif len(input_shape)==4 and input_shape[-1] in (1,3):
            shp = [int(input_shape[1]), int(input_shape[2]), int(input_shape[3])]
    if not shp or len(shp) != 3:
        return None
    h,w,c = [int(x) for x in shp]
    raw = np.fromfile(str(pp), dtype=np.uint8)
    if raw.size != h*w*c:
        return None
    arr = raw.reshape((h,w,c)).astype(np.float32)
    mode = str(scale or 'native').strip().lower()
    if mode == 'native':
        prep = man.get('preprocess') if isinstance(man.get('preprocess'), dict) else {}
        mode = str(prep.get('ort_model_scale') or 'norm').lower()
    normalize = mode not in {'raw', 'resize_raw', 'letterbox0_raw', 'letterbox114_raw', 'native_raw', 'native_letterbox_raw', 'yolo_raw', 'yolo_letterbox_raw'}
    if normalize:
        arr /= 255.0
    if len(input_shape)==4 and input_shape[1] in (1,3):
        return np.transpose(arr,(2,0,1))[None].astype(np.float32)
    if len(input_shape)==4 and input_shape[-1] in (1,3):
        return arr[None].astype(np.float32)
    return None


def _candidate_layouts(raw: np.ndarray, target_shape: tuple[int,...]) -> list[tuple[str,np.ndarray]]:
    out=[]
    if tuple(raw.shape)==target_shape:
        out.append(('as_manifest_shape', raw))
    # If target is NCHW, try treating raw memory as NHWC.
    if len(target_shape)==4 and target_shape[1] not in (-1,0) and target_shape[2] not in (-1,0):
        n,c,h,w=target_shape
        if raw.size==int(np.prod(target_shape)):
            try:
                x=raw.reshape((n,h,w,c)).transpose(0,3,1,2)
                out.append(('memory_nhwc_to_nchw', x))
            except Exception:
                pass
            try:
                x=raw.reshape((n,w,h,c)).transpose(0,3,2,1)
                out.append(('memory_nwhc_to_nchw', x))
            except Exception:
                pass
    # If target is NHWC, try NCHW memory.
    if len(target_shape)==4:
        n,h,w,c = target_shape if target_shape[-1] in (1,3,8,16,32,64,128,256,512,1024) else (None,None,None,None)
        if n is not None and raw.size==int(np.prod(target_shape)):
            try:
                x=raw.reshape((n,c,h,w)).transpose(0,2,3,1)
                out.append(('memory_nchw_to_nhwc', x))
            except Exception:
                pass
    # Deduplicate shapes/name maybe not needed.
    return out


def _fit_affine(raw: np.ndarray, ref: np.ndarray, sample: int=500000) -> dict[str,Any]:
    x=raw.astype(np.float64).ravel()
    y=ref.astype(np.float64).ravel()
    mask=np.isfinite(x)&np.isfinite(y)
    x=x[mask]; y=y[mask]
    if x.size>sample:
        idx=np.linspace(0,x.size-1,sample).astype(np.int64)
        x=x[idx]; y=y[idx]
    if x.size<2:
        return {'ok':False,'reason':'too_few_points'}
    xm=x.mean(); ym=y.mean()
    xv=((x-xm)**2).mean(); yv=((y-ym)**2).mean()
    if xv<=0 or yv<=0:
        return {'ok':False,'reason':'zero_variance','raw_std':float(math.sqrt(max(xv,0))), 'ref_std':float(math.sqrt(max(yv,0)))}
    cov=((x-xm)*(y-ym)).mean()
    a=cov/xv
    b=ym-a*xm
    pred=a*x+b
    ss_res=((y-pred)**2).mean()
    r2=1.0-ss_res/yv
    corr=cov/math.sqrt(xv*yv)
    zp = -b/a if abs(a)>1e-12 else None
    mae=np.abs(y-pred).mean()
    return {'ok':True,'scale_fit':float(a),'bias_fit':float(b),'zero_point_fit':(float(zp) if zp is not None else None),'r2':float(r2),'corr':float(corr),'mae':float(mae),'sample_count':int(x.size)}


def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument('--benchmark-set', required=True)
    ap.add_argument('--case', required=True)
    ap.add_argument('--boundary-manifest', required=True)
    ap.add_argument('--image', default='')
    ap.add_argument('--reference-report', default='')
    ap.add_argument('--image-scale', default='native', choices=['native','native_letterbox_norm','native_letterbox_raw','letterbox0_norm','letterbox0_raw','yolo','yolo_letterbox_norm','yolo_letterbox_raw','letterbox114_norm','letterbox114_raw','resize_norm','resize_raw','norm','raw','auto','imagenet'])
    ap.add_argument('--out', default='')
    ap.add_argument('--require-image-match', action='store_true', help='Fail if boundary dump provenance input_image and reference/image argument do not refer to the same file basename.')
    ns=ap.parse_args()
    bs=Path(ns.benchmark_set).expanduser().resolve(); case=ns.case
    man_path=Path(ns.boundary_manifest).expanduser().resolve()
    man=_read_json(man_path)
    binp=Path(man.get('file','')).expanduser()
    if not binp.is_absolute():
        binp=(man_path.parent/binp).resolve()
    raw_shape=man.get('shape') or man.get('boundary_shape')
    dtype_s=str(man.get('dtype') or 'uint8').lower()
    dtype_map={'uint8':np.uint8,'int8':np.int8,'float32':np.float32,'fp32':np.float32,'float16':np.float16,'fp16':np.float16}
    raw=np.fromfile(str(binp), dtype=dtype_map.get(dtype_s, np.uint8))
    if raw_shape:
        raw=raw.reshape([int(x) for x in raw_shape])

    part1=_find_part1_onnx(bs,case)
    inputs, outputs=_onnx_io(part1)
    if not inputs:
        raise RuntimeError(f'no inputs in {part1}')
    inp_name, inp_shape=inputs[0]
    ref_report=Path(ns.reference_report).expanduser().resolve() if ns.reference_report else None
    prov = man.get('provenance') if isinstance(man.get('provenance'), dict) else {}
    boundary_image = man.get('input_image') or man.get('image') or prov.get('image')
    ref_image_name = _image_from_report(ref_report)
    # For activation comparison, the ONNX Part1 activation must be computed for
    # the exact image that produced the raw boundary. Prefer boundary-manifest
    # provenance. --image still overrides for manual debugging.
    img_name = ns.image or boundary_image or ref_image_name
    img_path=_find_image(bs,img_name) if img_name else None
    if not img_path:
        raise FileNotFoundError(f'Could not resolve image {img_name!r}; pass --image explicitly')
    image_alignment = {'boundary_image': boundary_image, 'reference_report_image': ref_image_name, 'used_image': str(img_path), 'boundary_reference_mismatch': False, 'used_image_matches_boundary': False}
    try:
        if boundary_image and ref_image_name and Path(str(boundary_image)).name != Path(str(ref_image_name)).name:
            image_alignment['boundary_reference_mismatch'] = True
    except Exception:
        pass
    try:
        if boundary_image and Path(str(boundary_image)).name == Path(str(img_path)).name:
            image_alignment['used_image_matches_boundary'] = True
    except Exception:
        pass
    if ns.require_image_match and image_alignment.get('boundary_reference_mismatch'):
        out=Path(ns.out).expanduser().resolve() if ns.out else man_path.parent/'native_boundary_activation_compare.json'
        out.parent.mkdir(parents=True,exist_ok=True)
        payload={'schema':'onnx-splitpoint/native-boundary-activation-compare','schema_version':2,'ok':False,'error':'boundary_reference_image_mismatch','image_alignment':image_alignment,'benchmark_set':str(bs),'case':case,'boundary_manifest':str(man_path)}
        out.write_text(json.dumps(payload,indent=2),encoding='utf-8')
        print(json.dumps({'ok':False,'error':'boundary_reference_image_mismatch','out':str(out),'image_alignment':image_alignment},indent=2))
        return 3

    import onnxruntime as ort
    feed=_input_dump_feed_from_manifest(man, inp_shape, ns.image_scale)
    feed_source = 'boundary_manifest_input_dump' if feed is not None else f'image_preprocess:{ns.image_scale}'
    if feed is None:
        feed=_preprocess(img_path, inp_shape, ns.image_scale)
    sess=ort.InferenceSession(str(part1), providers=['CPUExecutionProvider'])
    ort_outs=sess.run(None,{inp_name:feed})
    out_names=[o.name for o in sess.get_outputs()]

    raw_summary=_summ(raw)
    candidates=[]
    for name, ref in zip(out_names, ort_outs):
        ref=np.asarray(ref)
        if ref.size!=raw.size:
            continue
        for lname, lr in _candidate_layouts(raw, tuple(ref.shape)):
            fit=_fit_affine(lr, ref)
            candidates.append({'onnx_output':name,'onnx_shape':list(ref.shape),'layout':lname,'fit':fit,'reference_summary':_summ(ref),'raw_layout_summary':_summ(lr)})
    candidates.sort(key=lambda c: (c.get('fit') or {}).get('r2', -999), reverse=True)
    payload={'schema':'onnx-splitpoint/native-boundary-activation-compare','schema_version':2,'benchmark_set':str(bs),'case':case,'part1_onnx':str(part1),'boundary_manifest':str(man_path),'boundary_file':str(binp),'image':str(img_path),'image_alignment':image_alignment,'input':{'name':inp_name,'shape':inp_shape,'scale':ns.image_scale,'feed_source':feed_source},'raw_summary':raw_summary,'candidates':candidates,'best':(candidates[0] if candidates else None)}
    out=Path(ns.out).expanduser().resolve() if ns.out else man_path.parent/'native_boundary_activation_compare.json'
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(payload,indent=2),encoding='utf-8')
    md=out.with_suffix('.md')
    lines=['# Native boundary activation compare','',f'Benchmark set: `{bs}`',f'Case: `{case}`',f'Image used for ORT Part1: `{img_path}`', f'ORT feed source: `{feed_source}`',f'Boundary manifest image: `{boundary_image}`',f'Reference report primary image: `{ref_image_name}`',f'Used image matches boundary: `{bool(image_alignment.get("used_image_matches_boundary"))}`', f'Boundary/reference-report primary image mismatch: `{bool(image_alignment.get("boundary_reference_mismatch"))}`','',f'Raw summary: min={raw_summary.get("min")} max={raw_summary.get("max")} mean={raw_summary.get("mean")}','', '| rank | onnx output | layout | r2 | corr | scale_fit | zero_point_fit | mae |', '|---:|---|---|---:|---:|---:|---:|---:|']
    for i,c in enumerate(candidates[:20]):
        f=c.get('fit') or {}
        lines.append(f'| {i} | `{c.get("onnx_output")}` | `{c.get("layout")}` | {f.get("r2","")} | {f.get("corr","")} | {f.get("scale_fit","")} | {f.get("zero_point_fit","")} | {f.get("mae","")} |')
    if candidates:
        b=candidates[0]; f=b.get('fit') or {}
        lines += ['', f'Best: output=`{b.get("onnx_output")}` layout=`{b.get("layout")}` r2=`{f.get("r2")}` scale=`{f.get("scale_fit")}` zero_point=`{f.get("zero_point_fit")}`']
    else:
        lines += ['', 'No ONNX Part1 output with the same element count as the boundary dump was found.']
    md.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({'ok':True,'out':str(out),'summary_md':str(md),'candidate_count':len(candidates),'best':payload.get('best')},indent=2))
    return 0

if __name__=='__main__':
    raise SystemExit(main())
