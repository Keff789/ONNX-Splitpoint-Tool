from __future__ import annotations

import json
import os
import re
import shutil
import ssl
import tarfile
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

try:  # Pillow is a project dependency, but keep this module import-safe.
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None  # type: ignore

LogFn = Optional[Callable[[str], None]]

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Downloadable public ImageNet-like classification fallback.  This is NOT the
# original ImageNet validation set; it is fast.ai's Imagenette v2 subset with
# ImageNet synsets/classes, useful for small smoke/semantic validation.
IMAGENETTE2_320_URLS = [
    "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz",
    "https://storage.googleapis.com/fastai_data/imagenette2-320.tgz",
]
IMAGENETTE_WNID_TO_IMAGENET_ID: Dict[str, int] = {
    "n01440764": 0,    # tench
    "n02102040": 217,  # English springer
    "n02979186": 482,  # cassette player
    "n03000684": 491,  # chain saw
    "n03028079": 497,  # church
    "n03394916": 566,  # French horn
    "n03417042": 569,  # garbage truck
    "n03425413": 571,  # gas pump
    "n03445777": 574,  # golf ball
    "n03888257": 701,  # parachute
}


def _log(log: LogFn, message: str) -> None:
    if log is not None:
        try:
            log(message)
        except Exception:
            pass


def validation_dataset_default_root() -> Path:
    env = str(os.environ.get("ONNX_SPLITPOINT_TOOL_VALIDATION_DATASETS") or "").strip()
    if env:
        return Path(os.path.expanduser(env)).resolve()
    return (Path.home() / ".onnx_splitpoint_tool" / "validation_datasets").resolve()


def package_resource_root() -> Path:
    return (Path(__file__).resolve().parent.parent / "resources").resolve()


def package_coco50_manifest_path() -> Path:
    return package_resource_root() / "validation" / "coco_50_manifest.json"


def detection_dataset_root() -> Path:
    return validation_dataset_default_root() / "detection"


def coco50_dataset_dir() -> Path:
    return detection_dataset_root() / "coco_50_data"


def classification_dataset_root() -> Path:
    return validation_dataset_default_root() / "classification"


def imagenette_mini_dataset_dir(images: int = 200) -> Path:
    return classification_dataset_root() / f"imagenette_val_mini_{int(images)}"


def imagenette_download_cache_dir() -> Path:
    return validation_dataset_default_root() / "_downloads"


def test_images_dir() -> Path:
    return validation_dataset_default_root() / "test_images"


def _legacy_packaged_coco50_dir() -> Path:
    return package_resource_root() / "validation" / "coco_50_data"


def _legacy_packaged_coco50_zip() -> Path:
    return package_resource_root() / "validation" / "coco_50_data.zip"


def _legacy_packaged_test_images_dir() -> Path:
    return package_resource_root() / "test_images"


def load_coco50_manifest() -> Dict[str, Any]:
    path = package_coco50_manifest_path()
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict) or not isinstance(obj.get("images"), list):
        raise RuntimeError(f"Invalid COCO-50 manifest: {path}")
    return obj


def _looks_like_coco50_dir(path: Path) -> bool:
    try:
        if not path.is_dir():
            return False
        jpgs = list(path.glob("*.jpg")) + list(path.glob("*.jpeg")) + list(path.glob("*.png"))
        jsons = list(path.glob("*.json"))
        return len(jpgs) >= 1 and len(jsons) >= 1
    except Exception:
        return False


def find_coco50_source() -> Optional[Path]:
    local = coco50_dataset_dir()
    if _looks_like_coco50_dir(local):
        return local
    legacy = _legacy_packaged_coco50_dir()
    if _looks_like_coco50_dir(legacy):
        return legacy
    zip_src = _legacy_packaged_coco50_zip()
    if zip_src.is_file():
        return zip_src
    return None


def find_test_images_source() -> Optional[Path]:
    local = test_images_dir()
    if (local / "test_image_coco.png").is_file() and (local / "test_image_imagenet.png").is_file():
        return local
    legacy = _legacy_packaged_test_images_dir()
    if (legacy / "test_image_coco.png").is_file() and (legacy / "test_image_imagenet.png").is_file():
        return legacy
    return None


@dataclass
class ValidationAssetsStatus:
    root: str
    coco50_ready: bool
    coco50_dir: str
    coco50_images: int
    coco50_annotations: int
    test_images_ready: bool
    test_images_dir: str
    manifest_available: bool
    imagenette200_ready: bool
    imagenette200_dir: str
    imagenette200_images: int
    imagenette500_ready: bool
    imagenette500_dir: str
    imagenette500_images: int

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _classification_manifest_image_count(path: Path) -> int:
    try:
        m = path / "manifest.json"
        if not m.is_file():
            return 0
        obj = json.loads(m.read_text(encoding="utf-8"))
        samples = obj.get("samples") if isinstance(obj, dict) else None
        if not isinstance(samples, list):
            return 0
        count = 0
        for s in samples:
            if not isinstance(s, dict):
                continue
            rel = str(s.get("image") or "")
            p = (path / rel).resolve() if rel else None
            if p is not None and p.is_file():
                count += 1
        return count
    except Exception:
        return 0


def validation_assets_status() -> ValidationAssetsStatus:
    cdir = coco50_dataset_dir()
    tdir = test_images_dir()
    img_count = 0
    ann_count = 0
    if cdir.is_dir():
        img_count = len([p for p in cdir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}])
        ann_count = len([p for p in cdir.iterdir() if p.is_file() and p.suffix.lower() == ".json"])
    i200 = imagenette_mini_dataset_dir(200)
    i500 = imagenette_mini_dataset_dir(500)
    i200_count = _classification_manifest_image_count(i200)
    i500_count = _classification_manifest_image_count(i500)
    return ValidationAssetsStatus(
        root=str(validation_dataset_default_root()),
        coco50_ready=img_count >= 50 and ann_count >= 50,
        coco50_dir=str(cdir),
        coco50_images=img_count,
        coco50_annotations=ann_count,
        test_images_ready=(tdir / "test_image_coco.png").is_file() and (tdir / "test_image_imagenet.png").is_file(),
        test_images_dir=str(tdir),
        manifest_available=package_coco50_manifest_path().is_file(),
        imagenette200_ready=i200_count >= 200,
        imagenette200_dir=str(i200),
        imagenette200_images=i200_count,
        imagenette500_ready=i500_count >= 500,
        imagenette500_dir=str(i500),
        imagenette500_images=i500_count,
    )


def _download_file(url: str, dest: Path, *, timeout: float = 60.0, log: LogFn = None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_name = tempfile.mkstemp(prefix=dest.name + ".", suffix=".tmp", dir=str(dest.parent))
    os.close(tmp_fd)
    tmp = Path(tmp_name)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ONNX-Splitpoint-Tool validation-assets"})
        with urllib.request.urlopen(req, timeout=timeout) as resp, tmp.open("wb") as f:
            shutil.copyfileobj(resp, f)
        tmp.replace(dest)
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass


def _download_with_fallbacks(urls: Sequence[str], dest: Path, *, timeout: float = 60.0, log: LogFn = None) -> None:
    errors: List[str] = []
    for idx, url in enumerate(urls, start=1):
        try:
            _log(log, f"[validation-assets] download source {idx}/{len(urls)}: {url}")
            _download_file(url, dest, timeout=timeout, log=log)
            return
        except Exception as exc:
            errors.append(f"{url}: {type(exc).__name__}: {exc}")
            _log(log, f"[validation-assets] download source failed: {type(exc).__name__}: {exc}")
    raise RuntimeError("all download sources failed:\n" + "\n".join(errors))


def _coco_url_candidates(url: str) -> List[str]:
    out: List[str] = []
    raw = str(url or "").strip()
    if raw:
        # Prefer plain HTTP for the COCO image host.  Several Windows/Python
        # setups report a certificate hostname mismatch for the HTTPS endpoint,
        # while the official COCO examples use the HTTP image server.
        if raw.startswith("https://images.cocodataset.org/"):
            out.append("http://" + raw[len("https://"):])
            out.append(raw)
        else:
            out.append(raw)
            if raw.startswith("http://images.cocodataset.org/"):
                out.append("https://" + raw[len("http://"):])
    return list(dict.fromkeys(out))


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def prepare_coco50_dataset(*, overwrite: bool = False, log: LogFn = None, timeout: float = 60.0) -> Path:
    manifest = load_coco50_manifest()
    images = list(manifest.get("images") or [])
    if not images:
        raise RuntimeError("COCO-50 manifest contains no images")
    url_template = str(manifest.get("source_url_template") or "http://images.cocodataset.org/val2017/{file_name}")
    if url_template.startswith("https://images.cocodataset.org/"):
        _log(log, "[validation-assets] COCO host HTTPS may fail on some systems; HTTP fallback is enabled.")
    dest = coco50_dataset_dir()
    dest.mkdir(parents=True, exist_ok=True)
    _log(log, f"[validation-assets] Preparing COCO-50 in {dest}")
    for idx, item in enumerate(images, start=1):
        if not isinstance(item, dict):
            continue
        fname = str(item.get("file_name") or "").strip()
        if not fname:
            continue
        img_dest = dest / fname
        ann_dest = dest / (Path(fname).stem + ".json")
        if overwrite or not ann_dest.exists():
            _write_json(ann_dest, item.get("annotations") or [])
        if overwrite or not img_dest.exists():
            url = url_template.format(file_name=fname)
            _log(log, f"[validation-assets] {idx:02d}/{len(images)} download {fname}")
            _download_with_fallbacks(_coco_url_candidates(url), img_dest, timeout=timeout, log=log)
        else:
            _log(log, f"[validation-assets] {idx:02d}/{len(images)} exists {fname}")
    _write_json(dest / "manifest.json", manifest)
    _log(log, "[validation-assets] COCO-50 ready")
    return dest


def _load_imagenet_labels() -> List[str]:
    p = package_resource_root() / "imagenet_labels.json"
    with p.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, list) or len(obj) != 1000:
        raise RuntimeError(f"Unexpected imagenet_labels.json format: {p}")
    return [str(x) for x in obj]


def _spaced_take(seq: Sequence[Any], target: int) -> List[Any]:
    vals = list(seq)
    if target <= 0 or not vals:
        return []
    if target >= len(vals):
        return vals
    if target == 1:
        return [vals[0]]
    used = set()
    out: List[Any] = []
    last_idx = len(vals) - 1
    for i in range(target):
        pos = int(round((i * last_idx) / float(target - 1)))
        while pos in used and pos < last_idx:
            pos += 1
        if pos in used:
            pos = next(j for j in range(len(vals)) if j not in used)
        used.add(pos)
        out.append(vals[pos])
    return out


def _tar_member_is_imagenette_val_image(name: str) -> Optional[Tuple[str, str]]:
    parts = Path(name).parts
    # Expected: imagenette2-320/val/n01440764/ILSVRC2012_val_...JPEG
    if len(parts) < 4:
        return None
    if parts[-3] != "val":
        return None
    wnid = parts[-2]
    if wnid not in IMAGENETTE_WNID_TO_IMAGENET_ID:
        return None
    if Path(parts[-1]).suffix.lower() not in _IMAGE_EXTS:
        return None
    return wnid, parts[-1]


def _select_imagenette_members(members: Sequence[tarfile.TarInfo], target_images: int) -> List[tarfile.TarInfo]:
    by_class: Dict[str, List[tarfile.TarInfo]] = {}
    for m in members:
        if not m.isfile():
            continue
        parsed = _tar_member_is_imagenette_val_image(m.name)
        if parsed is None:
            continue
        wnid, _ = parsed
        by_class.setdefault(wnid, []).append(m)
    for key in by_class:
        by_class[key] = sorted(by_class[key], key=lambda x: x.name)
    classes = sorted(by_class.keys())
    if not classes:
        raise RuntimeError("No Imagenette validation images were found in the archive")
    selected: List[tarfile.TarInfo] = []
    round_idx = 0
    while len(selected) < target_images:
        progressed = False
        for cls in classes:
            vals = by_class.get(cls) or []
            if round_idx < len(vals):
                selected.append(vals[round_idx])
                progressed = True
                if len(selected) >= target_images:
                    break
        if not progressed:
            break
        round_idx += 1
    if len(selected) < target_images:
        raise RuntimeError(f"Could not select {target_images} Imagenette validation images; got {len(selected)}")
    return selected


def _safe_extract_file_from_tar(tf: tarfile.TarFile, member: tarfile.TarInfo, dest: Path) -> None:
    src = tf.extractfile(member)
    if src is None:
        raise RuntimeError(f"Could not read archive member: {member.name}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with src, dest.open("wb") as f:
        shutil.copyfileobj(src, f)


def prepare_imagenette_mini_dataset(
    *,
    images: int = 200,
    overwrite: bool = False,
    log: LogFn = None,
    timeout: float = 300.0,
) -> Path:
    """Download fast.ai Imagenette2-320 and build a small ImageNet-ID manifest.

    This is a public, downloadable ImageNet-subset fallback.  It is intentionally
    named ``imagenette_*`` to avoid implying that the original ImageNet val set
    is redistributed by the tool.
    """
    images = int(images)
    if images not in {200, 500}:
        raise ValueError("Supported downloadable Imagenette mini sizes are 200 and 500")
    dest = imagenette_mini_dataset_dir(images)
    if dest.exists():
        if not overwrite and _classification_manifest_image_count(dest) >= images:
            _log(log, f"[validation-assets] Imagenette mini-{images} already ready in {dest}")
            return dest
        shutil.rmtree(dest)
    cache = imagenette_download_cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / "imagenette2-320.tgz"
    if overwrite or not archive.is_file():
        _log(log, "[validation-assets] Downloading Imagenette2-320 archive. This is about 326 MB.")
        _download_with_fallbacks(IMAGENETTE2_320_URLS, archive, timeout=timeout, log=log)
    else:
        _log(log, f"[validation-assets] Using cached Imagenette archive: {archive}")

    labels = _load_imagenet_labels()
    images_dir = dest / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    samples: List[Dict[str, Any]] = []
    selected_classes: List[int] = []
    _log(log, f"[validation-assets] Building Imagenette mini-{images} in {dest}")
    with tarfile.open(archive, "r:gz") as tf:
        selected = _select_imagenette_members(tf.getmembers(), images)
        for i, member in enumerate(selected, start=1):
            parsed = _tar_member_is_imagenette_val_image(member.name)
            if parsed is None:
                continue
            wnid, filename = parsed
            label_id = int(IMAGENETTE_WNID_TO_IMAGENET_ID[wnid])
            label_name = labels[label_id]
            safe_name = f"{wnid}_{Path(filename).name}"
            dst = images_dir / safe_name
            _safe_extract_file_from_tar(tf, member, dst)
            selected_classes.append(label_id)
            samples.append(
                {
                    "image": (Path("images") / safe_name).as_posix(),
                    "label_id": label_id,
                    "label_name": label_name,
                    "source_dataset": "imagenette2-320",
                    "source_member": member.name,
                    "wnid": wnid,
                }
            )
            if i % 25 == 0 or i == len(selected):
                _log(log, f"[validation-assets] Imagenette extracted {i}/{len(selected)}")
    manifest = {
        "schema": "onnx-splitpoint/classification-validation-manifest",
        "schema_version": 1,
        "dataset": f"imagenette_val_mini_{images}",
        "source_type": "imagenette2_320_download",
        "source_url": IMAGENETTE2_320_URLS[0],
        "note": "Public fast.ai Imagenette subset; labels map to ImageNet-1k class IDs. This is not the original ImageNet validation set.",
        "strategy": {
            "type": "round_robin_by_imagenette_class",
            "requested_images": images,
            "selected_images": len(samples),
            "unique_classes": len(set(selected_classes)),
        },
        "samples": samples,
    }
    _write_json(dest / "manifest.json", manifest)
    (dest / "README.txt").write_text(
        f"Imagenette validation mini-{images}\n"
        "Built from fast.ai imagenette2-320.tgz.\n"
        "Labels are ImageNet-1k label IDs for the 10 Imagenette classes.\n"
        "This downloadable preset is a public fallback; for final ImageNet-mini evaluation, prefer importing local ImageNet val data.\n",
        encoding="utf-8",
    )
    _log(log, f"[validation-assets] Imagenette mini-{images} ready")
    return dest


def _image_to_png(src: Path, dst: Path) -> bool:
    if Image is None:
        return False
    try:
        with Image.open(src) as im:
            im = im.convert("RGB")
            im.thumbnail((640, 640))
            canvas = Image.new("RGB", (640, 640), (114, 114, 114))
            x = (640 - im.width) // 2
            y = (640 - im.height) // 2
            canvas.paste(im, (x, y))
            dst.parent.mkdir(parents=True, exist_ok=True)
            canvas.save(dst, format="PNG")
        return True
    except Exception:
        return False


def prepare_test_images(*, overwrite: bool = False, log: LogFn = None) -> Path:
    out = test_images_dir()
    out.mkdir(parents=True, exist_ok=True)
    coco_png = out / "test_image_coco.png"
    imagenet_png = out / "test_image_imagenet.png"
    if not overwrite and coco_png.exists() and imagenet_png.exists():
        _log(log, f"[validation-assets] Test images already ready in {out}")
        return out
    src_dir = coco50_dataset_dir()
    candidates = sorted([p for p in src_dir.glob("*.jpg")]) if src_dir.is_dir() else []
    if candidates:
        _log(log, "[validation-assets] Creating runner test images from prepared COCO-50")
        src = candidates[0]
        ok1 = _image_to_png(src, coco_png)
        ok2 = _image_to_png(candidates[min(1, len(candidates)-1)], imagenet_png)
        if ok1 and ok2:
            return out
    c200 = imagenette_mini_dataset_dir(200) / "images"
    cands2 = sorted([p for p in c200.glob("*") if p.suffix.lower() in _IMAGE_EXTS]) if c200.is_dir() else []
    if cands2:
        _log(log, "[validation-assets] Creating ImageNet test image from prepared Imagenette mini")
        _image_to_png(cands2[0], imagenet_png)
    _log(log, "[validation-assets] Could not create all test images; runner placeholder fallback will be used")
    return out


def prepare_all_validation_assets(
    *,
    include_coco50: bool = True,
    include_imagenette200: bool = True,
    include_imagenette500: bool = False,
    include_test_images: bool = True,
    overwrite: bool = False,
    log: LogFn = None,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {"root": str(validation_dataset_default_root()), "prepared": []}
    if include_coco50:
        p = prepare_coco50_dataset(overwrite=overwrite, log=log)
        result["prepared"].append({"name": "coco_50_data", "path": str(p)})
    if include_imagenette200:
        p = prepare_imagenette_mini_dataset(images=200, overwrite=overwrite, log=log)
        result["prepared"].append({"name": "imagenette_val_mini_200", "path": str(p)})
    if include_imagenette500:
        p = prepare_imagenette_mini_dataset(images=500, overwrite=overwrite, log=log)
        result["prepared"].append({"name": "imagenette_val_mini_500", "path": str(p)})
    if include_test_images:
        p = prepare_test_images(overwrite=overwrite, log=log)
        result["prepared"].append({"name": "test_images", "path": str(p)})
    result["status"] = validation_assets_status().as_dict()
    return result


def default_detection_validation_source() -> Optional[Path]:
    return find_coco50_source()


def provision_detection_validation_source_to_suite(suite_dir: Path) -> Optional[str]:
    src = find_coco50_source()
    if src is None:
        return None
    dest_rel = Path("resources") / "validation" / "coco_50_data"
    dest = Path(suite_dir).resolve() / dest_rel
    if dest.is_dir():
        return dest_rel.as_posix()
    dest.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dest, dirs_exist_ok=True)
    elif src.is_file() and src.suffix.lower() == ".zip":
        import zipfile
        with zipfile.ZipFile(src, "r") as zf:
            zf.extractall(dest.parent)
        if not dest.exists() and (dest.parent / src.stem).is_dir():
            maybe = dest.parent / src.stem
            if dest.exists():
                shutil.rmtree(dest)
            maybe.rename(dest)
    return dest_rel.as_posix() if dest.exists() else None


@dataclass
class Coco50Status:
    ready: bool
    path: str
    expected_images: int
    present_images: int
    present_annotations: int
    note: str


def status_coco50() -> Coco50Status:
    st = validation_assets_status()
    return Coco50Status(
        ready=bool(st.coco50_ready),
        path=st.coco50_dir,
        expected_images=50,
        present_images=int(st.coco50_images),
        present_annotations=int(st.coco50_annotations),
        note="ready" if st.coco50_ready else f"{st.coco50_images}/50 images, {st.coco50_annotations}/50 annotations",
    )


def validation_assets_summary() -> Dict[str, Any]:
    return validation_assets_status().as_dict()


def prepare_coco50(*, overwrite: bool = False, progress: LogFn = None) -> Coco50Status:
    def _log_adapter(msg: str) -> None:
        if progress is not None:
            try:
                progress(msg)
            except Exception:
                pass
    prepare_coco50_dataset(overwrite=overwrite, log=_log_adapter)
    prepare_test_images(overwrite=overwrite, log=_log_adapter)
    return status_coco50()
