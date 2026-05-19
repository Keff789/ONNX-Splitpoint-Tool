from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

PRESET_SPECS: Dict[str, Dict[str, Any]] = {
    "imagenet_val_mini_200": {
        "aliases": [
            "imagenet-mini-200",
            "imagenet_mini_200",
            "imagenet-mini200",
            "mini-imagenet-200",
            "imagenet200",
        ],
        "images": 200,
        "description": "Deterministic 200-image ImageNet-1k validation subset with broad class spread. Built from local ImageNet val data.",
        "source": "imagenet_val",
    },
    "imagenet_val_mini_500": {
        "aliases": [
            "imagenet-mini-500",
            "imagenet_mini_500",
            "imagenet-mini500",
            "mini-imagenet-500",
            "imagenet500",
        ],
        "images": 500,
        "description": "Deterministic 500-image ImageNet-1k validation subset with broad class spread. Built from local ImageNet val data.",
        "source": "imagenet_val",
    },
    "imagenette_val_mini_200": {
        "aliases": [
            "imagenette-mini-200",
            "imagenette_mini_200",
            "imagenette200",
            "downloadable-imagenet-mini-200",
            "classification-mini-200",
        ],
        "images": 200,
        "description": "Downloadable 200-image fast.ai Imagenette validation subset mapped to ImageNet-1k class IDs.",
        "source": "imagenette2_320_download",
    },
    "imagenette_val_mini_500": {
        "aliases": [
            "imagenette-mini-500",
            "imagenette_mini_500",
            "imagenette500",
            "downloadable-imagenet-mini-500",
            "classification-mini-500",
        ],
        "images": 500,
        "description": "Downloadable 500-image fast.ai Imagenette validation subset mapped to ImageNet-1k class IDs.",
        "source": "imagenette2_320_download",
    },
}


def classification_validation_default_root() -> Path:
    env = str(os.environ.get("ONNX_SPLITPOINT_TOOL_CLASSIFICATION_DATASETS") or "").strip()
    if env:
        return Path(os.path.expanduser(env)).resolve()
    return (Path.home() / ".onnx_splitpoint_tool" / "validation_datasets" / "classification").resolve()


def classification_validation_resource_root() -> Path:
    return (Path(__file__).resolve().parent.parent / "resources" / "validation" / "classification").resolve()


def list_available_presets() -> List[str]:
    return list(PRESET_SPECS.keys())


def normalize_classification_validation_preset(value: Any) -> Optional[str]:
    raw = str(value or "").strip()
    if not raw:
        return None
    raw_l = raw.lower()
    for prefix in ("builtin://", "preset://", "classification://"):
        if raw_l.startswith(prefix):
            raw_l = raw_l[len(prefix):]
            break
    for name, spec in PRESET_SPECS.items():
        if raw_l == name.lower():
            return name
        for alias in list(spec.get("aliases") or []):
            if raw_l == str(alias).strip().lower():
                return name
    return None


def _preset_candidate_paths(name: str, *, base_dir: Optional[Path] = None) -> List[Path]:
    candidates: List[Path] = []
    if base_dir is not None:
        bd = Path(base_dir).resolve()
        candidates.extend(
            [
                bd / "resources" / "validation" / "classification" / name / "manifest.json",
                bd / "resources" / "validation" / "classification" / name,
                bd / "resources" / "validation" / "classification" / f"{name}.json",
            ]
        )
    root = classification_validation_default_root()
    candidates.extend(
        [
            root / name / "manifest.json",
            root / name,
            root / f"{name}.json",
        ]
    )
    res_root = classification_validation_resource_root()
    candidates.extend(
        [
            res_root / name / "manifest.json",
            res_root / name,
            res_root / f"{name}.json",
        ]
    )
    seen = set()
    out: List[Path] = []
    for cand in candidates:
        key = str(cand)
        if key in seen:
            continue
        seen.add(key)
        out.append(cand)
    return out


def _looks_like_classification_dataset_dir(path: Path) -> bool:
    try:
        if not path.is_dir():
            return False
        if (path / "manifest.json").is_file():
            return True
        if (path / "images").is_dir():
            for img in (path / "images").rglob("*"):
                if img.is_file() and img.suffix.lower() in _IMAGE_EXTS:
                    return True
        for child in path.iterdir():
            if child.is_file() and child.suffix.lower() in _IMAGE_EXTS:
                return True
            if child.is_dir():
                # class folders or nested images
                for img in child.rglob("*"):
                    if img.is_file() and img.suffix.lower() in _IMAGE_EXTS:
                        return True
                    break
        return False
    except Exception:
        return False


def resolve_classification_validation_source(value: Any, *, base_dir: Optional[Path] = None) -> Optional[Path]:
    raw = str(value or "").strip()
    if not raw:
        return None
    preset = normalize_classification_validation_preset(raw)
    if preset:
        for cand in _preset_candidate_paths(preset, base_dir=base_dir):
            if cand.is_file() or _looks_like_classification_dataset_dir(cand):
                return cand.resolve()
        return None
    try:
        p = Path(os.path.expanduser(raw))
        if not p.is_absolute() and base_dir is not None:
            p = (Path(base_dir) / p).resolve()
        if p.is_file() or _looks_like_classification_dataset_dir(p):
            return p.resolve()
    except Exception:
        return None
    return None


def default_available_classification_validation_preset(*, base_dir: Optional[Path] = None) -> Optional[str]:
    # Prefer true ImageNet-mini presets imported from local ImageNet val data.
    # Fall back to the downloadable Imagenette mini preset when available.
    for name in ("imagenet_val_mini_200", "imagenet_val_mini_500", "imagenette_val_mini_200", "imagenette_val_mini_500"):
        if resolve_classification_validation_source(name, base_dir=base_dir) is not None:
            return name
    return None


def _copy_or_link(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    if mode == "symlink":
        dst.symlink_to(src)
    else:
        shutil.copy2(src, dst)


def _safe_manifest_name(value: str) -> str:
    txt = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(value or "dataset"))
    txt = txt.strip("_")
    return txt or "dataset"


def _dataset_root_for_source(source: Path) -> Tuple[Path, Path]:
    src = Path(source).resolve()
    if src.is_dir():
        manifest = src / "manifest.json"
        rel = Path("manifest.json") if manifest.is_file() else Path("")
        return src, rel
    if src.suffix.lower() in _IMAGE_EXTS:
        return src, Path("")
    return src.parent, Path(src.name)


def provision_classification_validation_source_to_suite(
    suite_dir: Path,
    requested: Any,
    *,
    base_dir: Optional[Path] = None,
) -> Optional[str]:
    resolved = resolve_classification_validation_source(requested, base_dir=base_dir)
    if resolved is None:
        return None
    source_root, rel_inside = _dataset_root_for_source(resolved)
    preset = normalize_classification_validation_preset(requested)
    dest_name = preset or _safe_manifest_name(source_root.name)
    dest_root_rel = Path("resources") / "validation" / "classification" / dest_name
    dest_root = Path(suite_dir).resolve() / dest_root_rel
    if source_root.is_dir():
        if dest_root.resolve() != source_root.resolve():
            if dest_root.exists():
                shutil.rmtree(dest_root)
            shutil.copytree(source_root, dest_root)
        if rel_inside == Path(""):
            return dest_root_rel.as_posix()
        return (dest_root_rel / rel_inside).as_posix()
    dest_root.mkdir(parents=True, exist_ok=True)
    dst_file = dest_root / source_root.name
    shutil.copy2(source_root, dst_file)
    if rel_inside == Path(""):
        return (dest_root_rel / source_root.name).as_posix()
    return (dest_root_rel / rel_inside).as_posix()


def _load_imagenet_labels() -> List[str]:
    p = Path(__file__).resolve().parent.parent / "resources" / "imagenet_labels.json"
    with p.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, list) or len(obj) != 1000:
        raise RuntimeError(f"Unexpected imagenet_labels.json format: {p}")
    return [str(x) for x in obj]


def _collect_sorted_images(path: Path) -> List[Path]:
    out = [p.resolve() for p in sorted(path.iterdir()) if p.is_file() and p.suffix.lower() in _IMAGE_EXTS]
    if out:
        return out
    out = [p.resolve() for p in sorted(path.rglob("*")) if p.is_file() and p.suffix.lower() in _IMAGE_EXTS]
    return out


def _load_imagenet_ground_truth(path: Path) -> List[int]:
    vals: List[int] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        vals.append(int(line))
    if not vals:
        raise RuntimeError(f"Ground-truth file is empty: {path}")
    vmin = min(vals)
    vmax = max(vals)
    if vmin >= 1 and vmax <= 1000:
        return [int(v) - 1 for v in vals]
    if vmin >= 0 and vmax < 1000:
        return [int(v) for v in vals]
    raise RuntimeError(
        f"Ground-truth labels in {path} are outside the supported ImageNet range (min={vmin}, max={vmax})."
    )


def _spaced_take(seq: Sequence[int], target: int) -> List[int]:
    if target <= 0:
        return []
    if target >= len(seq):
        return list(seq)
    if target == 1:
        return [int(seq[0])]
    used = set()
    out: List[int] = []
    last_idx = len(seq) - 1
    for i in range(target):
        pos = int(round((i * last_idx) / float(target - 1)))
        while pos in used and pos < last_idx:
            pos += 1
        if pos in used:
            pos = next(j for j in range(len(seq)) if j not in used)
        used.add(pos)
        out.append(int(seq[pos]))
    return out


def _build_balanced_selection(files: Sequence[Path], labels: Sequence[int], target_images: int) -> List[int]:
    class_to_indices: Dict[int, List[int]] = {}
    for idx, cls in enumerate(labels):
        class_to_indices.setdefault(int(cls), []).append(int(idx))
    classes = sorted(class_to_indices.keys())
    if not classes:
        return []
    selected: List[int] = []
    if len(classes) >= target_images:
        chosen_classes = _spaced_take(classes, target_images)
        for cls in chosen_classes:
            selected.append(class_to_indices[int(cls)][0])
        return selected

    chosen_classes = list(classes)
    round_idx = 0
    while len(selected) < target_images:
        progressed = False
        for cls in chosen_classes:
            idxs = class_to_indices[int(cls)]
            if round_idx < len(idxs):
                selected.append(idxs[round_idx])
                progressed = True
                if len(selected) >= target_images:
                    break
        if not progressed:
            break
        round_idx += 1
    return selected[:target_images]


def build_imagenet_validation_preset(
    *,
    preset_name: str,
    imagenet_val_dir: Path,
    ground_truth_file: Path,
    output_root: Optional[Path] = None,
    copy_mode: str = "copy",
    overwrite: bool = False,
) -> Path:
    preset = normalize_classification_validation_preset(preset_name) or str(preset_name or "").strip()
    if preset not in PRESET_SPECS:
        raise ValueError(f"Unsupported preset: {preset_name!r}")
    spec = PRESET_SPECS[preset]
    target_images = int(spec.get("images") or 0)
    if target_images <= 0:
        raise ValueError(f"Preset {preset} has no positive image count")
    val_dir = Path(imagenet_val_dir).expanduser().resolve()
    gt_file = Path(ground_truth_file).expanduser().resolve()
    if not val_dir.is_dir():
        raise FileNotFoundError(f"ImageNet val directory not found: {val_dir}")
    if not gt_file.is_file():
        raise FileNotFoundError(f"ImageNet ground-truth file not found: {gt_file}")
    labels = _load_imagenet_labels()
    files = _collect_sorted_images(val_dir)
    gt = _load_imagenet_ground_truth(gt_file)
    if len(files) != len(gt):
        raise RuntimeError(
            f"Image count / ground-truth mismatch for {val_dir}: {len(files)} images vs {len(gt)} labels"
        )
    selected_indices = _build_balanced_selection(files, gt, target_images)
    if len(selected_indices) < target_images:
        raise RuntimeError(f"Could not select {target_images} images; got {len(selected_indices)}")
    out_root = (output_root or classification_validation_default_root())
    out_root = Path(out_root).expanduser().resolve()
    dataset_root = out_root / preset
    if dataset_root.exists():
        if not overwrite:
            raise FileExistsError(f"Destination already exists: {dataset_root}")
        shutil.rmtree(dataset_root)
    images_dir = dataset_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    samples: List[Dict[str, Any]] = []
    selected_classes: List[int] = []
    for src_idx in selected_indices:
        src = files[int(src_idx)]
        label_id = int(gt[int(src_idx)])
        selected_classes.append(label_id)
        dst = images_dir / src.name
        if copy_mode not in {"copy", "symlink", "manifest-only"}:
            raise ValueError(f"Unsupported copy mode: {copy_mode!r}")
        if copy_mode != "manifest-only":
            _copy_or_link(src, dst, copy_mode)
        else:
            dst = src
        samples.append(
            {
                "image": (Path("images") / src.name).as_posix() if copy_mode != "manifest-only" else str(src),
                "label_id": label_id,
                "label_name": labels[label_id],
                "source_image": src.name,
            }
        )
    manifest = {
        "schema": "onnx-splitpoint/classification-validation-manifest",
        "schema_version": 1,
        "dataset": preset,
        "source_type": "imagenet_val",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "strategy": {
            "type": "class_spread_round_robin",
            "requested_images": target_images,
            "selected_images": len(samples),
            "unique_classes": len(set(selected_classes)),
        },
        "samples": samples,
    }
    (dataset_root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    readme = (
        f"{preset}\n"
        f"Generated from ImageNet-1k validation images in {val_dir}.\n"
        f"Ground truth file: {gt_file.name}.\n"
        f"Selection strategy: deterministic class-spread round-robin with {len(set(selected_classes))} unique classes.\n"
        f"Images copied mode: {copy_mode}.\n"
    )
    (dataset_root / "README.txt").write_text(readme, encoding="utf-8")
    return dataset_root


def _cli_build_preset(args: argparse.Namespace) -> int:
    dataset_root = build_imagenet_validation_preset(
        preset_name=args.preset,
        imagenet_val_dir=Path(args.imagenet_val),
        ground_truth_file=Path(args.ground_truth),
        output_root=Path(args.output_root) if getattr(args, "output_root", None) else None,
        copy_mode=str(args.copy_mode or "copy"),
        overwrite=bool(getattr(args, "overwrite", False)),
    )
    print(dataset_root)
    return 0


def _cli_resolve_preset(args: argparse.Namespace) -> int:
    resolved = resolve_classification_validation_source(args.name, base_dir=(Path(args.base_dir) if args.base_dir else None))
    if resolved is None:
        return 1
    print(resolved)
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Build or resolve classification validation presets for the ONNX Splitpoint Tool.",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_build = sub.add_parser("build-imagenet", help="Build an ImageNet-mini validation preset from a local ImageNet val folder.")
    ap_build.add_argument("--preset", choices=list_available_presets(), required=True)
    ap_build.add_argument("--imagenet-val", required=True, help="Path to the local ImageNet-1k validation image directory.")
    ap_build.add_argument("--ground-truth", required=True, help="Path to ILSVRC2012_validation_ground_truth.txt (1-based or 0-based labels).")
    ap_build.add_argument("--output-root", default=str(classification_validation_default_root()), help="Where the generated preset folder should be written.")
    ap_build.add_argument("--copy-mode", choices=["copy", "symlink", "manifest-only"], default="copy")
    ap_build.add_argument("--overwrite", action="store_true")
    ap_build.set_defaults(func=_cli_build_preset)

    ap_resolve = sub.add_parser("resolve", help="Resolve a preset alias or explicit classification dataset path.")
    ap_resolve.add_argument("name")
    ap_resolve.add_argument("--base-dir", default="")
    ap_resolve.set_defaults(func=_cli_resolve_preset)
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = build_arg_parser()
    args = ap.parse_args(list(argv) if argv is not None else None)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
