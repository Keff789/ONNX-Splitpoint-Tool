#!/usr/bin/env python3
"""Remote activation-proxy collector for Splitpoint Stage2 accelerator builds.

Runs Part1 ONNX on a remote runtime host (typically the DeepX NX with CUDA/TRT
available), collects the tensors that feed Part2, and writes sample_*.npz plus a
manifest that the build host can download and consume for Stage2 calibration.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def _nname(s: str) -> str:
    import re
    return re.sub(r"[^a-zA-Z0-9]+", "_", str(s or "")).strip("_").lower()


def _providers_for_backend(backend: str) -> List[str]:
    b = str(backend or "ort_cpu").strip().lower().replace("-", "_")
    if b in {"remote_deepx_tensorrt", "tensorrt_ort", "trt", "tensorrt"}:
        return ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
    if b in {"remote_deepx_cuda", "cuda_ort", "cuda"}:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def _provider_name_from_session(providers: List[str]) -> str:
    if "TensorrtExecutionProvider" in providers:
        return "remote_deepx_tensorrt"
    if "CUDAExecutionProvider" in providers:
        return "remote_deepx_cuda"
    return "ort_cpu"


def _try_preload_ort() -> Dict[str, Any]:
    out: Dict[str, Any] = {"attempted": False, "ok": None, "error": ""}
    try:
        import onnxruntime as ort  # type: ignore
        if hasattr(ort, "preload_dlls"):
            out["attempted"] = True
            try:
                ort.preload_dlls(directory="")
                out["ok"] = True
            except Exception as exc:
                out["ok"] = False
                out["error"] = f"{type(exc).__name__}: {exc}"
    except Exception as exc:
        out["error"] = f"onnxruntime import failed before preload: {type(exc).__name__}: {exc}"
    return out


def _load_image(path: Path) -> np.ndarray:
    try:
        import cv2  # type: ignore
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError("cv2.imread returned None")
        if img.ndim == 2:
            img = img[..., None]
        if img.shape[-1] == 4:
            img = img[..., :3]
        if img.shape[-1] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return np.asarray(img)
    except Exception:
        from PIL import Image  # type: ignore
        with Image.open(path) as im:
            if im.mode != "RGB":
                im = im.convert("RGB")
            return np.asarray(im)


def _resize_hwc(x: np.ndarray, h: int | None, w: int | None, c: int | None) -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if c is not None:
        if arr.shape[-1] == c:
            pass
        elif arr.shape[-1] == 1 and c == 3:
            arr = np.repeat(arr, 3, axis=-1)
        elif arr.shape[-1] == 3 and c == 1:
            arr = np.mean(arr, axis=-1, keepdims=True)
    h = int(h or arr.shape[0])
    w = int(w or arr.shape[1])
    if arr.shape[0] != h or arr.shape[1] != w:
        try:
            import cv2  # type: ignore
            arr = cv2.resize(arr, (w, h), interpolation=cv2.INTER_LINEAR)
            if arr.ndim == 2:
                arr = arr[..., None]
        except Exception:
            from PIL import Image  # type: ignore
            im = Image.fromarray(arr.astype(np.uint8) if not np.issubdtype(arr.dtype, np.integer) else arr)
            im = im.resize((w, h), Image.BILINEAR)
            arr = np.asarray(im)
            if arr.ndim == 2:
                arr = arr[..., None]
    return np.ascontiguousarray(arr)


def _input_spec(inp: Any) -> Tuple[int | None, int | None, int | None, str]:
    shape = list(getattr(inp, "shape", []) or [])
    dims: List[int | None] = []
    for d in shape:
        try:
            dims.append(int(d) if d is not None and str(d).isdigit() else None)
        except Exception:
            dims.append(None)
    if len(dims) == 4:
        if dims[1] in (1, 3, 4):
            return dims[2], dims[3], dims[1], "NCHW"
        if dims[3] in (1, 3, 4):
            return dims[1], dims[2], dims[3], "NHWC"
    return None, None, 3, "NCHW"


def _preprocess_hwc(x: np.ndarray, mode: str) -> np.ndarray:
    y = np.asarray(x).astype(np.float32)
    if y.size and float(np.nanmax(y)) > 2.0:
        y = y / 255.0
    m = str(mode or "norm").strip().lower()
    if m in {"imagenet", "meanstd", "torchvision"}:
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        if y.shape[-1] == 3:
            y = (y - mean) / std
    return y


def _prepare_input(path: Path, inp: Any, preprocess: str) -> np.ndarray:
    h, w, c, layout = _input_spec(inp)
    arr = _resize_hwc(_load_image(path), h, w, c)
    exp_type = str(getattr(inp, "type", "") or "").lower()
    if "float" in exp_type:
        arr = _preprocess_hwc(arr, preprocess)
    else:
        if np.issubdtype(arr.dtype, np.floating) and arr.size and float(np.nanmax(arr)) <= 1.5:
            arr = arr * 255.0
        arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
    if layout == "NCHW":
        arr = np.transpose(arr, (2, 0, 1))[None, ...]
    else:
        arr = arr[None, ...]
    return np.ascontiguousarray(arr)


def _iter_calib_items(calib_dir: Path, limit: int) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    items = sorted([p for p in calib_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts])
    return items[: max(1, int(limit))]


def _map_inputs(part1_outputs: List[str], part2_inputs: List[str]) -> Tuple[Dict[str, str], List[str], Dict[str, Any]]:
    mapping: Dict[str, str] = {}
    how: Dict[str, str] = {}
    norm = {_nname(n): n for n in part1_outputs}
    for raw in part2_inputs:
        nn = _nname(raw)
        if raw in part1_outputs:
            mapping[raw] = raw; how[raw] = "exact"
        elif nn in norm:
            mapping[raw] = norm[nn]; how[raw] = "normalized"
    if len(mapping) != len(part2_inputs) and len(part1_outputs) == len(part2_inputs):
        for i, raw in enumerate(part2_inputs):
            if raw not in mapping:
                mapping[raw] = part1_outputs[i]; how[raw] = "positional_fallback"
    missing = [n for n in part2_inputs if n not in mapping]
    return mapping, missing, {"mapping": mapping, "mapping_how": how, "missing_inputs": missing, "part1_outputs": part1_outputs, "part2_inputs": part2_inputs}


def _stats(arrs: List[np.ndarray]) -> Dict[str, Any]:
    try:
        x = np.stack([np.asarray(a, dtype=np.float32) for a in arrs], axis=0)
        return {"min": float(np.min(x)), "max": float(np.max(x)), "mean": float(np.mean(x)), "std": float(np.std(x)), "shape": list(x.shape)}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def _write_manifest(out_dir: Path, args: argparse.Namespace, *, names: List[str], samples: Dict[str, List[np.ndarray]], debug: Dict[str, Any]) -> None:
    tensors = []
    stats = {}
    n = 0
    for name in names:
        vals = samples.get(name) or []
        n = max(n, len(vals))
        if vals:
            sample_shape = list(np.asarray(vals[0]).shape)
            stats[name] = _stats(vals)
        else:
            sample_shape = []
            stats[name] = {"error": "no samples"}
        tensors.append({"name": name, "sample_shape": sample_shape, "dtype": "float32", "layout": "unknown", "stats": stats[name]})
    manifest = {
        "schema_version": 1,
        "cache_kind": "activation_calibration",
        "cache_version": "v54u_remote_proxy",
        "source": debug.get("activation_proxy_source"),
        "calibration_source": debug.get("activation_proxy_source"),
        "requested_backend": debug.get("activation_proxy_requested_backend"),
        "producer_backend": debug.get("activation_proxy_producer_backend"),
        "producer_exact": False,
        "trust_level": "remote_proxy",
        "status": "ready",
        "stage2_backend": args.stage2_backend,
        "part1_onnx": str(args.part1),
        "part2_onnx": str(args.part2),
        "calib_source_dir": str(args.calib_dir),
        "requested_count": int(args.limit),
        "sample_count": int(n),
        "stored_sample_count": int(n),
        "sample_storage": "npz_samples",
        "activation_gen_batch": int(args.batch_size),
        "tensors": tensors,
        "stats": stats,
        "debug": debug,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part1", required=True)
    ap.add_argument("--part2", required=True)
    ap.add_argument("--calib-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--backend", default="remote_deepx_tensorrt")
    ap.add_argument("--preprocess", default="norm")
    ap.add_argument("--stage2-backend", default="deepx_m1")
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()
    part1 = Path(args.part1).resolve(); part2 = Path(args.part2).resolve(); calib = Path(args.calib_dir).resolve(); out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        _preload = _try_preload_ort()
        import onnxruntime as ort  # type: ignore
    except Exception as exc:
        print(json.dumps({"ok": False, "error": f"onnxruntime import failed: {type(exc).__name__}: {exc}"}))
        return 2
    items = _iter_calib_items(calib, args.limit)
    if not items:
        print(json.dumps({"ok": False, "error": f"no calibration images in {calib}"}))
        return 3
    so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
    requested = str(args.backend or "remote_deepx_tensorrt")
    provider_attempts: List[List[str]] = [_providers_for_backend(requested)]
    if requested != "remote_deepx_cuda":
        provider_attempts.append(_providers_for_backend("remote_deepx_cuda"))
    provider_attempts.append(["CPUExecutionProvider"])
    last_exc = ""
    p1_sess = None; used_providers: List[str] = []
    for providers in provider_attempts:
        try:
            sess = ort.InferenceSession(str(part1), sess_options=so, providers=providers)
            got = list(sess.get_providers() or [])
            # If strict, require requested accelerated provider to be active.
            if args.strict and providers[0] not in got:
                raise RuntimeError(f"strict provider {providers[0]} not active, got {got}")
            p1_sess = sess; used_providers = got
            break
        except Exception as exc:
            last_exc = f"{type(exc).__name__}: {exc}"
            if args.strict:
                break
    if p1_sess is None:
        print(json.dumps({"ok": False, "error": f"could not create Part1 session: {last_exc}"}))
        return 4
    p2_sess = ort.InferenceSession(str(part2), providers=["CPUExecutionProvider"])
    p1_in = p1_sess.get_inputs()[0]
    p1_outs = [o.name for o in p1_sess.get_outputs()]
    p2_inputs = [i.name for i in p2_sess.get_inputs()]
    mapping, missing, map_debug = _map_inputs(p1_outs, p2_inputs)
    if missing:
        print(json.dumps({"ok": False, "error": "part2 inputs not produced by part1", "missing": missing, "debug": map_debug}))
        return 5
    req_outs = [mapping[n] for n in p2_inputs]
    samples: Dict[str, List[np.ndarray]] = {n: [] for n in p2_inputs}
    idx = 0
    bs = max(1, int(args.batch_size))
    # Keep batch 1 for static batch-1 networks.
    try:
        if int(p1_in.shape[0]) == 1:
            bs = 1
    except Exception:
        pass
    runtime_fallback_reason = ""
    while idx < len(items):
        batch = items[idx:idx+bs]
        xb = np.concatenate([_prepare_input(p, p1_in, args.preprocess) for p in batch], axis=0) if len(batch) > 1 else _prepare_input(batch[0], p1_in, args.preprocess)
        try:
            outs = p1_sess.run(req_outs, {p1_in.name: xb})
        except Exception as exc:
            # Runtime providers can fail on real model ops even if session creation succeeded.
            if args.strict or used_providers == ["CPUExecutionProvider"]:
                print(json.dumps({"ok": False, "error": f"Part1 run failed: {type(exc).__name__}: {exc}", "providers": used_providers}))
                return 6
            runtime_fallback_reason = f"runtime inference failed on {used_providers}; falling back to CPU: {type(exc).__name__}: {exc}"
            p1_sess = ort.InferenceSession(str(part1), sess_options=so, providers=["CPUExecutionProvider"])
            used_providers = list(p1_sess.get_providers() or [])
            samples = {n: [] for n in p2_inputs}; idx = 0
            continue
        for name, arr in zip(p2_inputs, outs):
            a = np.asarray(arr)
            if a.ndim >= 1 and a.shape[0] == xb.shape[0]:
                for bi in range(int(a.shape[0])):
                    samples[name].append(np.ascontiguousarray(np.asarray(a[bi], dtype=np.float32)))
            else:
                samples[name].append(np.ascontiguousarray(np.asarray(a, dtype=np.float32)))
        idx += bs
    n = min(len(v) for v in samples.values()) if samples else 0
    for i in range(n):
        payload = {name: np.asarray(samples[name][i], dtype=np.float32) for name in p2_inputs}
        np.savez_compressed(str(out_dir / f"sample_{i:06d}.npz"), **payload)
    prod = _provider_name_from_session(used_providers)
    src = f"{prod}_reference_proxy"
    if str(requested).startswith("remote_deepx") and prod.startswith("remote_deepx"):
        src = prod + "_reference_proxy"
    debug = {
        "activation_proxy_requested_backend": requested,
        "activation_proxy_producer_backend": prod,
        "activation_proxy_source": src,
        "activation_proxy_provider_fallback": runtime_fallback_reason,
        "part1_session_providers": used_providers,
        "part2_session_providers": list(p2_sess.get_providers() or []),
        "mapping_part2in_to_part1out": mapping,
        "mapping_debug": map_debug,
        "preload_dlls": _preload,
        "calib_items": [str(p) for p in items[:10]],
        "calib_item_count": len(items),
        "host": os.uname().nodename if hasattr(os, "uname") else "remote",
        "preprocess": args.preprocess,
    }
    _write_manifest(out_dir, args, names=p2_inputs, samples=samples, debug=debug)
    print(json.dumps({"ok": True, "out_dir": str(out_dir), "sample_count": int(n), "producer_backend": prod, "source": src, "providers": used_providers, "fallback_reason": runtime_fallback_reason}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
