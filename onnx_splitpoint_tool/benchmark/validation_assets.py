from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
import tempfile
import threading
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None  # type: ignore

LogFn = Optional[Callable[[str], None]]
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_DETECTION_SUBSET_LOCK = threading.Lock()

COCO_ANNOTATIONS_URLS = [
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
    "https://images.cocodataset.org/annotations/annotations_trainval2017.zip",
]

IMAGENETTE2_320_URLS = [
    "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz",
    "https://storage.googleapis.com/fastai_data/imagenette2-320.tgz",
]
IMAGENETTE_WNID_TO_IMAGENET_ID: Dict[str, int] = {
    "n01440764": 0,
    "n02102040": 217,
    "n02979186": 482,
    "n03000684": 491,
    "n03028079": 497,
    "n03394916": 566,
    "n03417042": 569,
    "n03425413": 571,
    "n03445777": 574,
    "n03888257": 701,
}


def _log(log: LogFn, message: str) -> None:
    if log is not None:
        try:
            log(message)
        except Exception:
            pass


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def validation_dataset_default_root() -> Path:
    env = str(os.environ.get("ONNX_SPLITPOINT_TOOL_VALIDATION_DATASETS") or "").strip()
    if env:
        return Path(os.path.expanduser(env)).resolve()
    return (Path.home() / ".onnx_splitpoint_tool" / "validation_datasets").resolve()


def package_resource_root() -> Path:
    return (Path(__file__).resolve().parent.parent / "resources").resolve()


def detection_dataset_root() -> Path:
    return validation_dataset_default_root() / "detection"


def classification_dataset_root() -> Path:
    return validation_dataset_default_root() / "classification"


def coco_dataset_dir(images: int = 50) -> Path:
    return detection_dataset_root() / f"coco_{int(images)}_data"


def coco50_dataset_dir() -> Path:
    return coco_dataset_dir(50)


def coco200_dataset_dir() -> Path:
    return coco_dataset_dir(200)


def imagenette_mini_dataset_dir(images: int = 200) -> Path:
    return classification_dataset_root() / f"imagenette_val_mini_{int(images)}"


def imagenette_download_cache_dir() -> Path:
    return validation_dataset_default_root() / "_downloads"


def test_images_dir() -> Path:
    return validation_dataset_default_root() / "test_images"


def package_coco_manifest_path(images: int = 50) -> Path:
    images = int(images)
    if images == 50:
        return package_resource_root() / "validation" / "coco_50_manifest.json"
    return detection_dataset_root() / f"coco_{images}_manifest.json"


def package_coco50_manifest_path() -> Path:
    return package_coco_manifest_path(50)


def _legacy_packaged_coco50_dir() -> Path:
    return package_resource_root() / "validation" / "coco_50_data"


def _legacy_packaged_coco50_zip() -> Path:
    return package_resource_root() / "validation" / "coco_50_data.zip"


def _legacy_packaged_test_images_dir() -> Path:
    return package_resource_root() / "test_images"


def _normalize_coco_preset_name(value: str | int = "coco_50") -> str:
    raw = str(value or "coco_50").strip().lower().replace("-", "_")
    if raw in {"coco_200", "coco200", "coco_200_data", "200", "coco_200_calib"}:
        return "coco_200"
    return "coco_50"


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
    raw = str(url or "").strip()
    out: List[str] = []
    if raw:
        if raw.startswith("https://images.cocodataset.org/"):
            out.append("http://" + raw[len("https://"):])
            out.append(raw)
        else:
            out.append(raw)
            if raw.startswith("http://images.cocodataset.org/"):
                out.append("https://" + raw[len("http://"):])
    return list(dict.fromkeys(out))


def load_coco_manifest(images: int = 50, *, log: LogFn = None, overwrite: bool = False) -> Dict[str, Any]:
    images = int(images)
    path = package_coco_manifest_path(images)
    if images != 50 and (overwrite or not path.is_file()):
        build_coco_manifest(images=images, overwrite=overwrite, log=log)
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict) or not isinstance(obj.get("images"), list):
        raise RuntimeError(f"Invalid COCO-{images} manifest: {path}")
    return obj


def load_coco50_manifest() -> Dict[str, Any]:
    return load_coco_manifest(50)


def _load_coco_val2017_instances(*, overwrite: bool = False, log: LogFn = None, timeout: float = 300.0) -> Dict[str, Any]:
    cache = imagenette_download_cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / "annotations_trainval2017.zip"
    if overwrite or not archive.is_file():
        _log(log, "[validation-assets] Downloading COCO annotations_trainval2017.zip for COCO-200 manifest generation.")
        _download_with_fallbacks(COCO_ANNOTATIONS_URLS, archive, timeout=timeout, log=log)
    with zipfile.ZipFile(archive, "r") as zf:
        name = "annotations/instances_val2017.json"
        if name not in zf.namelist():
            raise RuntimeError(f"COCO annotations archive does not contain {name}")
        with zf.open(name, "r") as f:
            return json.loads(f.read().decode("utf-8"))


def build_coco_manifest(*, images: int = 200, overwrite: bool = False, log: LogFn = None) -> Path:
    images = int(images)
    if images <= 0:
        raise ValueError("COCO manifest image count must be positive")
    if images == 50:
        return package_coco50_manifest_path()
    out = package_coco_manifest_path(images)
    if out.is_file() and not overwrite:
        return out
    instances = _load_coco_val2017_instances(overwrite=overwrite, log=log)
    anns_by_image: Dict[int, List[Dict[str, Any]]] = {}
    for ann in instances.get("annotations") or []:
        if not isinstance(ann, dict):
            continue
        try:
            iid = int(ann.get("image_id"))
        except Exception:
            continue
        anns_by_image.setdefault(iid, []).append(ann)
    candidates: List[Dict[str, Any]] = []
    for img in instances.get("images") or []:
        if not isinstance(img, dict):
            continue
        try:
            iid = int(img.get("id"))
        except Exception:
            continue
        anns = anns_by_image.get(iid) or []
        if not anns:
            continue
        candidates.append({
            "file_name": str(img.get("file_name") or f"{iid:012d}.jpg"),
            "image_id": iid,
            "annotations": anns,
        })
    if len(candidates) < images:
        raise RuntimeError(f"COCO val2017 has only {len(candidates)} annotated candidates; need {images}")
    selected = _spaced_take(sorted(candidates, key=lambda x: int(x.get("image_id") or 0)), images)
    manifest = {
        "name": f"coco_{images}_data",
        "version": 1,
        "description": f"Automatically generated COCO-{images} detection calibration/validation subset from COCO val2017 annotations.",
        "source_url_template": "http://images.cocodataset.org/val2017/{file_name}",
        "annotations_source": COCO_ANNOTATIONS_URLS[0],
        "selection": {"type": "spaced_annotated_val2017", "requested_images": images, "selected_images": len(selected)},
        "images": selected,
    }
    _write_json(out, manifest)
    _log(log, f"[validation-assets] COCO-{images} manifest ready: {out}")
    return out


def _looks_like_coco_dir(path: Path) -> bool:
    try:
        if not path.is_dir():
            return False
        jpgs = [p for p in path.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
        jsons = [p for p in path.iterdir() if p.is_file() and p.suffix.lower() == ".json" and p.name != "manifest.json"]
        return len(jpgs) >= 1 and len(jsons) >= 1
    except Exception:
        return False


def _looks_like_coco50_dir(path: Path) -> bool:
    return _looks_like_coco_dir(path)


def find_coco_source(images: int = 50) -> Optional[Path]:
    images = int(images)
    local = coco_dataset_dir(images)
    if _looks_like_coco_dir(local):
        return local
    if images == 50:
        legacy = _legacy_packaged_coco50_dir()
        if _looks_like_coco_dir(legacy):
            return legacy
        zip_src = _legacy_packaged_coco50_zip()
        if zip_src.is_file():
            return zip_src
    return None


def find_coco50_source() -> Optional[Path]:
    return find_coco_source(50)


def find_coco200_source() -> Optional[Path]:
    return find_coco_source(200)


def resolve_detection_validation_source(name: str = "coco_50", *, base_dir: Optional[Path] = None) -> Optional[Path]:
    """Resolve a detection dataset alias or explicit directory/manifest path.

    Earlier versions canonicalised every non-COCO string to ``coco_50``.  That
    silently discarded the content-addressed COCO validation root bound by the
    evaluation profile.
    """
    raw = str(name or "").strip()
    low = raw.lower().replace("-", "_")
    if raw:
        try:
            p = Path(os.path.expanduser(raw))
            if not p.is_absolute() and base_dir is not None:
                p = Path(base_dir) / p
            if p.exists():
                return p.resolve()
        except Exception:
            pass
    if low in {"coco_200", "coco200", "coco_200_data", "200", "coco_200_calib"}:
        return find_coco200_source()
    if low in {"", "coco", "coco_50", "coco50", "coco_50_data", "50"}:
        return find_coco50_source()
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
    coco200_ready: bool
    coco200_dir: str
    coco200_images: int
    coco200_annotations: int
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
        for sample in samples:
            if not isinstance(sample, dict):
                continue
            rel = str(sample.get("image") or "")
            p = (path / rel).resolve() if rel else None
            if p is not None and p.is_file():
                count += 1
        return count
    except Exception:
        return 0


def _coco_counts(path: Path) -> Tuple[int, int]:
    if not path.is_dir():
        return 0, 0
    imgs = len([p for p in path.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_EXTS])
    anns = len([p for p in path.iterdir() if p.is_file() and p.suffix.lower() == ".json" and p.name != "manifest.json"])
    return imgs, anns


def validation_assets_status() -> ValidationAssetsStatus:
    c50_img, c50_ann = _coco_counts(coco50_dataset_dir())
    c200_img, c200_ann = _coco_counts(coco200_dataset_dir())
    i200 = imagenette_mini_dataset_dir(200)
    i500 = imagenette_mini_dataset_dir(500)
    i200_count = _classification_manifest_image_count(i200)
    i500_count = _classification_manifest_image_count(i500)
    tdir = test_images_dir()
    return ValidationAssetsStatus(
        root=str(validation_dataset_default_root()),
        coco50_ready=c50_img >= 50 and c50_ann >= 50,
        coco50_dir=str(coco50_dataset_dir()),
        coco50_images=c50_img,
        coco50_annotations=c50_ann,
        coco200_ready=c200_img >= 200 and c200_ann >= 200,
        coco200_dir=str(coco200_dataset_dir()),
        coco200_images=c200_img,
        coco200_annotations=c200_ann,
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


def prepare_coco_dataset(images: int, *, overwrite: bool = False, log: LogFn = None, timeout: float = 60.0) -> Path:
    images = int(images)
    manifest = load_coco_manifest(images, log=log, overwrite=overwrite)
    items = list(manifest.get("images") or [])
    if not items:
        raise RuntimeError(f"COCO-{images} manifest contains no images")
    url_template = str(manifest.get("source_url_template") or "http://images.cocodataset.org/val2017/{file_name}")
    dest = coco_dataset_dir(images)
    dest.mkdir(parents=True, exist_ok=True)
    _log(log, f"[validation-assets] Preparing COCO-{images} in {dest}")
    for idx, item in enumerate(items, start=1):
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
            _log(log, f"[validation-assets] COCO-{images} {idx:03d}/{len(items)} download {fname}")
            _download_with_fallbacks(_coco_url_candidates(url_template.format(file_name=fname)), img_dest, timeout=timeout, log=log)
        elif idx % 25 == 0 or idx == len(items):
            _log(log, f"[validation-assets] COCO-{images} {idx:03d}/{len(items)} exists {fname}")
    _write_json(dest / "manifest.json", manifest)
    _log(log, f"[validation-assets] COCO-{images} ready")
    return dest


def prepare_coco50_dataset(*, overwrite: bool = False, log: LogFn = None, timeout: float = 60.0) -> Path:
    return prepare_coco_dataset(50, overwrite=overwrite, log=log, timeout=timeout)


def prepare_coco200_dataset(*, overwrite: bool = False, log: LogFn = None, timeout: float = 60.0) -> Path:
    return prepare_coco_dataset(200, overwrite=overwrite, log=log, timeout=max(float(timeout), 60.0))


def _load_imagenet_labels() -> List[str]:
    p = package_resource_root() / "imagenet_labels.json"
    with p.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, list) or len(obj) != 1000:
        raise RuntimeError(f"Unexpected imagenet_labels.json format: {p}")
    return [str(x) for x in obj]


def _tar_member_is_imagenette_val_image(name: str) -> Optional[Tuple[str, str]]:
    parts = Path(name).parts
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


def prepare_imagenette_mini_dataset(*, images: int = 200, overwrite: bool = False, log: LogFn = None, timeout: float = 300.0) -> Path:
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
            samples.append({
                "image": (Path("images") / safe_name).as_posix(),
                "label_id": label_id,
                "label_name": label_name,
                "source_dataset": "imagenette2-320",
                "source_member": member.name,
                "wnid": wnid,
            })
            if i % 25 == 0 or i == len(selected):
                _log(log, f"[validation-assets] Imagenette extracted {i}/{len(selected)}")
    manifest = {
        "schema": "onnx-splitpoint/classification-validation-manifest",
        "schema_version": 1,
        "dataset": f"imagenette_val_mini_{images}",
        "source_type": "imagenette2_320_download",
        "source_url": IMAGENETTE2_320_URLS[0],
        "note": "Public fast.ai Imagenette subset; labels map to ImageNet-1k class IDs. This is not the original ImageNet validation set.",
        "strategy": {"type": "round_robin_by_imagenette_class", "requested_images": images, "selected_images": len(samples), "unique_classes": len(set(selected_classes))},
        "samples": samples,
    }
    _write_json(dest / "manifest.json", manifest)
    (dest / "README.txt").write_text(
        f"Imagenette validation mini-{images}\nBuilt from fast.ai imagenette2-320.tgz.\nLabels are ImageNet-1k label IDs for the 10 Imagenette classes.\n",
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
        ok1 = _image_to_png(candidates[0], coco_png)
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


def prepare_all_validation_assets(*, include_coco50: bool = True, include_coco200: bool = True, include_imagenette200: bool = True, include_imagenette500: bool = False, include_test_images: bool = True, overwrite: bool = False, log: LogFn = None) -> Dict[str, Any]:
    result: Dict[str, Any] = {"root": str(validation_dataset_default_root()), "prepared": []}
    if include_coco50:
        p = prepare_coco50_dataset(overwrite=overwrite, log=log)
        result["prepared"].append({"name": "coco_50_data", "path": str(p)})
    if include_coco200:
        p = prepare_coco200_dataset(overwrite=overwrite, log=log)
        result["prepared"].append({"name": "coco_200_data", "path": str(p)})
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
    return resolve_detection_validation_source(os.environ.get("ONNX_SPLITPOINT_DETECTION_VALIDATION_PRESET") or "coco_50")


def _safe_detection_name(value: Any) -> str:
    text = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value or "detection"))
    return text.strip("_") or "detection"


def _json_dict(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None or not Path(path).is_file():
        return None
    try:
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None
    return dict(obj) if isinstance(obj, dict) else None


def _manifest_ref(value: Any, *, manifest_path: Path) -> Optional[Path]:
    raw = str(value or "").strip()
    if not raw:
        return None
    p = Path(os.path.expanduser(raw))
    if not p.is_absolute():
        p = manifest_path.parent / p
    try:
        return p.resolve()
    except Exception:
        return p


def _detection_manifest_path(
    resolved: Path,
    *,
    explicit_manifest: Any = None,
    base_dir: Optional[Path] = None,
) -> Optional[Path]:
    raw = str(explicit_manifest or "").strip()
    if raw:
        p = Path(os.path.expanduser(raw))
        if not p.is_absolute() and base_dir is not None:
            p = Path(base_dir) / p
        try:
            p = p.resolve()
        except Exception:
            pass
        if p.is_file():
            return p
    if resolved.is_file() and resolved.suffix.lower() in {".json", ".jsonl"}:
        return resolved
    if resolved.is_dir() and (resolved / "manifest.json").is_file():
        return (resolved / "manifest.json").resolve()
    return None


def _stable_detection_key(row: Dict[str, Any], seed: int) -> str:
    ident = str(row.get("sample_id") or row.get("image_id") or row.get("src") or "")
    return hashlib.sha256(f"{int(seed)}|{ident}".encode("utf-8")).hexdigest()


def _select_detection_rows(rows: Sequence[Dict[str, Any]], max_images: int, seed: int) -> List[Dict[str, Any]]:
    seq = [dict(row) for row in rows]
    if max_images <= 0 or max_images >= len(seq):
        return seq
    return sorted(seq, key=lambda row: _stable_detection_key(row, seed))[:max_images]


def _hardlink_or_copy_detection(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _detection_rows_from_manifest(
    manifest_path: Path,
    *,
    fallback_root: Path,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    payload = _json_dict(manifest_path) or {}
    rows: List[Dict[str, Any]] = []
    context: Dict[str, Any] = {"payload": payload, "annotations": [], "categories": [], "images": []}

    # Campaign/content-addressed manifest.
    items = payload.get("items")
    if isinstance(items, list):
        root_raw = str(payload.get("root") or "").strip()
        root = _manifest_ref(root_raw, manifest_path=manifest_path) if root_raw else fallback_root
        root = root if root is not None else fallback_root
        ann_meta = payload.get("annotations") if isinstance(payload.get("annotations"), dict) else {}
        ann_path = _manifest_ref((ann_meta or {}).get("path"), manifest_path=manifest_path)
        ann_payload = _json_dict(ann_path) or {}
        context.update({
            "annotations_path": str(ann_path or ""),
            "annotations": list(ann_payload.get("annotations") or []),
            "categories": list(ann_payload.get("categories") or []),
            "images": list(ann_payload.get("images") or []),
        })
        for item in items:
            if not isinstance(item, dict):
                continue
            rel = str(item.get("relative_path") or item.get("image") or item.get("path") or "").strip()
            if not rel:
                continue
            src = Path(rel)
            if not src.is_absolute():
                src = root / src
            try:
                src = src.resolve()
            except Exception:
                pass
            # Campaign manifests are verified upstream.  Avoid one filesystem
            # stat per COCO item before deterministic selection; the chosen
            # source files are checked by the hardlink/copy operation below.
            if src.suffix.lower() not in _IMAGE_EXTS:
                continue
            rows.append({
                "src": src,
                "sample_id": str(item.get("sample_id") or rel),
                "image_id": item.get("image_id", item.get("sample_id")),
                "width": item.get("width"),
                "height": item.get("height"),
                "sha256": str(item.get("sha256") or ""),
                "size_bytes": int(item.get("size_bytes") or 0),
            })
        return rows, context

    # Legacy prepared COCO manifest with image records and inline annotations.
    images = payload.get("images")
    if isinstance(images, list):
        root = fallback_root
        context["categories"] = list(payload.get("categories") or [])
        for item in images:
            if not isinstance(item, dict):
                continue
            rel = str(item.get("file_name") or item.get("image") or item.get("path") or "").strip()
            if not rel:
                continue
            src = Path(rel)
            if not src.is_absolute():
                src = root / src
            try:
                src = src.resolve()
            except Exception:
                pass
            if not src.is_file() or src.suffix.lower() not in _IMAGE_EXTS:
                continue
            rows.append({
                "src": src,
                "sample_id": str(item.get("id") or rel),
                "image_id": item.get("id"),
                "width": item.get("width"),
                "height": item.get("height"),
                "annotations": list(item.get("annotations") or []),
            })
        return rows, context
    return rows, context


def _detection_rows_from_directory(root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not root.is_dir():
        return rows
    for src in sorted(root.rglob("*")):
        if not src.is_file() or src.suffix.lower() not in _IMAGE_EXTS:
            continue
        rel = src.relative_to(root).as_posix()
        sidecar = src.with_suffix(".json")
        annotations: List[Any] = []
        if sidecar.is_file():
            try:
                obj = json.loads(sidecar.read_text(encoding="utf-8"))
                if isinstance(obj, list):
                    annotations = obj
                elif isinstance(obj, dict):
                    annotations = list(obj.get("annotations") or obj.get("objects") or obj.get("instances") or [])
            except Exception:
                annotations = []
        rows.append({
            "src": src.resolve(),
            "sample_id": rel,
            "image_id": None,
            "annotations": annotations,
        })
    return rows


def _detection_subset_is_current(
    dest: Path,
    *,
    resolved: Path,
    manifest_path: Optional[Path],
    selected: Sequence[Dict[str, Any]],
    max_images: int,
    selection_seed: int,
) -> bool:
    payload = _json_dict(dest / "manifest.json")
    if not payload:
        return False
    selection = payload.get("selection") if isinstance(payload.get("selection"), dict) else {}
    samples = payload.get("samples") if isinstance(payload.get("samples"), list) else []
    expected = [str(row.get("sample_id") or row.get("src") or "") for row in selected]
    actual = [str(row.get("sample_id") or "") for row in samples if isinstance(row, dict)]
    if str(payload.get("source") or "") != str(resolved):
        return False
    if str(payload.get("source_manifest") or "") != str(manifest_path or ""):
        return False
    if int(selection.get("seed") or -1) != int(selection_seed):
        return False
    if int(selection.get("requested_images") or -1) != int(max_images):
        return False
    if expected != actual:
        return False
    for sample in samples:
        if not isinstance(sample, dict):
            return False
        rel = str(sample.get("image") or "").strip()
        if not rel or not (dest / rel).is_file() or not (dest / Path(rel).with_suffix(".json")).is_file():
            return False
    return True


def _materialize_detection_subset(
    suite_dir: Path,
    *,
    requested: Any,
    resolved: Path,
    max_images: int,
    manifest_path: Any = None,
    base_dir: Optional[Path] = None,
    selection_seed: int = 20260710,
) -> Optional[str]:
    source_root = resolved if resolved.is_dir() else resolved.parent
    source_manifest = _detection_manifest_path(
        resolved, explicit_manifest=manifest_path, base_dir=base_dir
    )
    rows: List[Dict[str, Any]] = []
    context: Dict[str, Any] = {}
    if source_manifest:
        rows, context = _detection_rows_from_manifest(source_manifest, fallback_root=source_root)
    if not rows:
        rows = _detection_rows_from_directory(source_root)
        context = {}
    selected = _select_detection_rows(rows, int(max_images), int(selection_seed))
    if not selected:
        return None

    raw_name = str(requested or "").strip()
    alias = _normalize_coco_preset_name(raw_name) if raw_name.lower().replace("-", "_") in {
        "", "coco", "coco_50", "coco50", "coco_50_data", "50",
        "coco_200", "coco200", "coco_200_data", "200", "coco_200_calib",
    } else ""
    base_name = alias or _safe_detection_name(source_root.name)
    dest_name = _safe_detection_name(f"{base_name}_n{len(selected)}_s{int(selection_seed)}")
    dest_rel = Path("resources") / "validation" / "detection" / dest_name
    dest = Path(suite_dir).resolve() / dest_rel

    all_annotations = [x for x in list(context.get("annotations") or []) if isinstance(x, dict)]
    all_images = [x for x in list(context.get("images") or []) if isinstance(x, dict)]
    categories = [x for x in list(context.get("categories") or []) if isinstance(x, dict)]
    anns_by_image: Dict[str, List[Dict[str, Any]]] = {}
    for ann in all_annotations:
        anns_by_image.setdefault(str(ann.get("image_id")), []).append(dict(ann))
    image_meta_by_id = {str(row.get("id")): dict(row) for row in all_images}

    with _DETECTION_SUBSET_LOCK:
        if _detection_subset_is_current(
            dest, resolved=resolved, manifest_path=source_manifest, selected=selected,
            max_images=int(max_images), selection_seed=int(selection_seed),
        ):
            return dest_rel.as_posix()

        tmp = dest.with_name(f".{dest.name}.tmp-{os.getpid()}-{threading.get_ident()}")
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True, exist_ok=True)
        samples: List[Dict[str, Any]] = []
        subset_images: List[Dict[str, Any]] = []
        subset_annotations: List[Dict[str, Any]] = []
        try:
            used_names: set[str] = set()
            for idx, row in enumerate(selected):
                src = Path(row["src"])
                name = src.name
                if name in used_names:
                    name = f"{idx:06d}_{name}"
                used_names.add(name)
                dst = tmp / name
                _hardlink_or_copy_detection(src, dst)
                image_id = row.get("image_id")
                annotations = list(row.get("annotations") or [])
                if not annotations and image_id is not None:
                    annotations = list(anns_by_image.get(str(image_id)) or [])
                _write_json(dst.with_suffix(".json"), annotations)

                image_meta = dict(image_meta_by_id.get(str(image_id)) or {})
                image_meta.update({
                    "id": image_id if image_id is not None else row.get("sample_id"),
                    "file_name": name,
                })
                if row.get("width") is not None:
                    image_meta["width"] = row.get("width")
                if row.get("height") is not None:
                    image_meta["height"] = row.get("height")
                subset_images.append(image_meta)
                subset_annotations.extend(dict(a) for a in annotations if isinstance(a, dict))
                sample: Dict[str, Any] = {
                    "image": name,
                    "sample_id": str(row.get("sample_id") or src.name),
                    "image_id": image_id,
                    "annotation_count": len(annotations),
                    "source_image": str(src),
                }
                if row.get("sha256"):
                    sample["source_sha256"] = str(row["sha256"])
                if row.get("size_bytes"):
                    sample["source_size_bytes"] = int(row["size_bytes"])
                samples.append(sample)

            filtered_coco = {
                "images": subset_images,
                "annotations": subset_annotations,
                "categories": categories,
            }
            _write_json(tmp / "instances_subset.json", filtered_coco)
            payload = {
                "schema": "onnx-splitpoint/detection-validation-manifest",
                "schema_version": 2,
                "dataset": dest_name,
                "source_type": "materialized_run_mode_subset",
                "source": str(resolved),
                "source_manifest": str(source_manifest or ""),
                "source_annotations": str(context.get("annotations_path") or ""),
                "selection": {
                    "type": "deterministic_hash",
                    "seed": int(selection_seed),
                    "requested_images": int(max_images),
                    "selected_images": len(samples),
                    "source_population": len(rows),
                },
                "samples": samples,
                "filtered_coco_annotations": "instances_subset.json",
            }
            _write_json(tmp / "manifest.json", payload)
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                shutil.rmtree(dest)
            os.replace(tmp, dest)
        finally:
            if tmp.exists():
                shutil.rmtree(tmp, ignore_errors=True)
    return dest_rel.as_posix()


def provision_detection_validation_source_to_suite(
    suite_dir: Path,
    preset: str | int = "coco_50",
    *,
    base_dir: Optional[Path] = None,
    max_images: int = 0,
    manifest_path: Any = None,
    selection_seed: int = 20260710,
) -> Optional[str]:
    resolved = resolve_detection_validation_source(str(preset), base_dir=base_dir)
    if resolved is None:
        # A manifest can be authoritative even when ``validation_images`` points
        # at a root that is not otherwise recognised as a legacy preset.
        mp = _detection_manifest_path(Path(str(preset or ".")), explicit_manifest=manifest_path, base_dir=base_dir)
        payload = _json_dict(mp) if mp else None
        root = str((payload or {}).get("root") or "").strip()
        if root:
            resolved = resolve_detection_validation_source(root, base_dir=(mp.parent if mp else base_dir))
    if resolved is None:
        return None
    # Generated Evaluation Workflow plans already point at one exact,
    # content-addressed subset inside the suite.  Re-materialising that subset
    # under a new preset-derived folder changes manifest bytes (dataset/source
    # aliases) even though Image IDs and ground truth stay identical.  Return
    # the frozen artefact itself when it is complete and has the requested
    # cardinality.  The downstream bundle and quality contract still hash the
    # exact manifest bytes, so this preserves rather than relaxes integrity.
    if int(max_images or 0) > 0:
        suite_root = Path(suite_dir).expanduser().resolve()
        subset_root = resolved.parent if resolved.is_file() and resolved.name == "manifest.json" else resolved
        try:
            subset_rel = subset_root.resolve().relative_to(suite_root)
        except (OSError, ValueError):
            subset_rel = None
        if subset_rel is not None and subset_root.is_dir():
            frozen_manifest = _json_dict(subset_root / "manifest.json") or {}
            samples = frozen_manifest.get("samples") if isinstance(frozen_manifest.get("samples"), list) else []
            selection = frozen_manifest.get("selection") if isinstance(frozen_manifest.get("selection"), dict) else {}
            frozen_subset = (
                str(frozen_manifest.get("schema") or "") == "onnx-splitpoint/detection-validation-manifest"
                and str(frozen_manifest.get("source_type") or "") == "materialized_run_mode_subset"
                and int(selection.get("requested_images") or 0) == int(max_images)
                and int(selection.get("selected_images") or 0) == int(max_images)
                and len(samples) == int(max_images)
            )
            if frozen_subset:
                for sample in samples:
                    if not isinstance(sample, dict):
                        frozen_subset = False
                        break
                    image_rel = str(sample.get("image") or "").strip()
                    if (
                        not image_rel
                        or not (subset_root / image_rel).is_file()
                        or not (subset_root / Path(image_rel).with_suffix(".json")).is_file()
                    ):
                        frozen_subset = False
                        break
            if frozen_subset:
                return subset_rel.as_posix()
    if int(max_images or 0) > 0 and not (resolved.is_file() and resolved.suffix.lower() == ".zip"):
        subset = _materialize_detection_subset(
            Path(suite_dir), requested=preset, resolved=resolved, max_images=int(max_images),
            manifest_path=manifest_path, base_dir=base_dir, selection_seed=int(selection_seed),
        )
        if subset:
            return subset

    preset_norm = _normalize_coco_preset_name(preset)
    dest_rel = Path("resources") / "validation" / "detection" / ("coco_200_data" if preset_norm == "coco_200" else "coco_50_data")
    dest = Path(suite_dir).resolve() / dest_rel
    if dest.is_dir():
        return dest_rel.as_posix()
    dest.parent.mkdir(parents=True, exist_ok=True)
    if resolved.is_dir():
        shutil.copytree(resolved, dest, dirs_exist_ok=True)
    elif resolved.is_file() and resolved.suffix.lower() == ".zip":
        with zipfile.ZipFile(resolved, "r") as zf:
            zf.extractall(dest.parent)
        if not dest.exists() and (dest.parent / resolved.stem).is_dir():
            maybe = dest.parent / resolved.stem
            if dest.exists():
                shutil.rmtree(dest)
            maybe.rename(dest)
    return dest_rel.as_posix() if dest.exists() else None


@dataclass
class CocoStatus:
    ready: bool
    path: str
    expected_images: int
    present_images: int
    present_annotations: int
    note: str

# Backwards-compatible alias used by older UI code.
Coco50Status = CocoStatus


def status_coco50() -> CocoStatus:
    st = validation_assets_status()
    return CocoStatus(bool(st.coco50_ready), st.coco50_dir, 50, int(st.coco50_images), int(st.coco50_annotations), "ready" if st.coco50_ready else f"{st.coco50_images}/50 images, {st.coco50_annotations}/50 annotations")


def status_coco200() -> CocoStatus:
    st = validation_assets_status()
    return CocoStatus(bool(st.coco200_ready), st.coco200_dir, 200, int(st.coco200_images), int(st.coco200_annotations), "ready" if st.coco200_ready else f"{st.coco200_images}/200 images, {st.coco200_annotations}/200 annotations")


def validation_assets_summary() -> Dict[str, Any]:
    return validation_assets_status().as_dict()


def prepare_coco50(*, overwrite: bool = False, progress: LogFn = None) -> CocoStatus:
    def _log_adapter(msg: str) -> None:
        if progress is not None:
            try:
                progress(msg)
            except Exception:
                pass
    prepare_coco50_dataset(overwrite=overwrite, log=_log_adapter)
    prepare_test_images(overwrite=overwrite, log=_log_adapter)
    return status_coco50()


def prepare_coco200(*, overwrite: bool = False, progress: LogFn = None) -> CocoStatus:
    def _log_adapter(msg: str) -> None:
        if progress is not None:
            try:
                progress(msg)
            except Exception:
                pass
    prepare_coco200_dataset(overwrite=overwrite, log=_log_adapter)
    prepare_test_images(overwrite=overwrite, log=_log_adapter)
    return status_coco200()
