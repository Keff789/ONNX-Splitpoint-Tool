from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_SUBSET_MATERIALIZE_LOCK = threading.Lock()

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


def _read_json_mapping(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None or not Path(path).is_file():
        return None
    try:
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None
    return dict(obj) if isinstance(obj, dict) else None


def _manifest_ref_path(value: Any, *, manifest_path: Path) -> Optional[Path]:
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


def _classification_manifest_path(
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


def _synset_label_map(path: Optional[Path]) -> Dict[str, Tuple[int, str]]:
    out: Dict[str, Tuple[int, str]] = {}
    if path is None or not path.is_file():
        return out
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return out
    for idx, line in enumerate(lines):
        text = str(line or "").strip()
        if not text:
            continue
        token = text.split(None, 1)[0]
        if token.startswith("n") and token[1:].isdigit():
            label = text.split(None, 1)[1].strip() if len(text.split(None, 1)) > 1 else token
            out[token] = (int(idx), label)
    return out


def _classification_rows_from_manifest(
    manifest_path: Path,
    *,
    fallback_root: Path,
) -> List[Dict[str, Any]]:
    payload = _read_json_mapping(manifest_path)
    if not payload:
        return []

    rows: List[Dict[str, Any]] = []
    labels_meta = payload.get("labels") if isinstance(payload.get("labels"), dict) else {}
    labels_path = _manifest_ref_path((labels_meta or {}).get("path"), manifest_path=manifest_path)
    wnid_map = _synset_label_map(labels_path)

    items = payload.get("items")
    if isinstance(items, list):
        root_raw = str(payload.get("root") or "").strip()
        root = _manifest_ref_path(root_raw, manifest_path=manifest_path) if root_raw else fallback_root
        root = root if root is not None else fallback_root
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
            # The content-addressed manifest was verified before workflow
            # execution.  Do not stat all 50,000 ImageNet files merely to pick a
            # 16-item Smoke subset; selected files are validated when linked.
            if src.suffix.lower() not in _IMAGE_EXTS:
                continue
            class_name = str(item.get("class_name") or src.parent.name or "").strip()
            label_id = item.get("label_id")
            label_name = str(item.get("label_name") or "").strip() or None
            if label_id is None and class_name in wnid_map:
                label_id, mapped = wnid_map[class_name]
                label_name = label_name or mapped
            rows.append({
                "src": src,
                "sample_id": str(item.get("sample_id") or rel),
                "class_name": class_name,
                "label_id": (int(label_id) if label_id is not None else None),
                "label_name": label_name or class_name or None,
                "sha256": str(item.get("sha256") or ""),
                "size_bytes": int(item.get("size_bytes") or 0),
            })
        return rows

    samples = payload.get("samples") or payload.get("images")
    if isinstance(samples, list):
        for item in samples:
            if isinstance(item, str):
                item = {"image": item}
            if not isinstance(item, dict):
                continue
            rel = str(item.get("image") or item.get("path") or item.get("file") or "").strip()
            if not rel:
                continue
            src = Path(rel)
            if not src.is_absolute():
                src = manifest_path.parent / src
            try:
                src = src.resolve()
            except Exception:
                pass
            if not src.is_file() or src.suffix.lower() not in _IMAGE_EXTS:
                continue
            class_name = str(item.get("class_name") or item.get("wnid") or src.parent.name or "").strip()
            label_id = item.get("label_id", item.get("class_id"))
            label_name = str(item.get("label_name") or item.get("label") or "").strip() or None
            if label_id is None and class_name in wnid_map:
                label_id, mapped = wnid_map[class_name]
                label_name = label_name or mapped
            rows.append({
                "src": src,
                "sample_id": str(item.get("sample_id") or item.get("source_image") or rel),
                "class_name": class_name,
                "label_id": (int(label_id) if label_id is not None else None),
                "label_name": label_name or class_name or None,
                "sha256": str(item.get("sha256") or ""),
                "size_bytes": int(item.get("size_bytes") or 0),
            })
    return rows


def _classification_rows_from_directory(source_root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not source_root.is_dir():
        return rows
    direct = [p for p in sorted(source_root.iterdir()) if p.is_file() and p.suffix.lower() in _IMAGE_EXTS]
    if direct:
        for src in direct:
            rows.append({"src": src.resolve(), "sample_id": src.name, "class_name": "", "label_id": None, "label_name": None})
        return rows
    for class_dir in sorted((p for p in source_root.iterdir() if p.is_dir()), key=lambda p: p.name):
        for src in sorted(class_dir.rglob("*")):
            if src.is_file() and src.suffix.lower() in _IMAGE_EXTS:
                rows.append({
                    "src": src.resolve(),
                    "sample_id": src.relative_to(source_root).as_posix(),
                    "class_name": class_dir.name,
                    "label_id": None,
                    "label_name": class_dir.name,
                })
    return rows


def _stable_row_key(row: Dict[str, Any], seed: int) -> str:
    identity = str(row.get("sample_id") or row.get("src") or "")
    return hashlib.sha256(f"{int(seed)}|{identity}".encode("utf-8")).hexdigest()


def _select_classification_rows(rows: Sequence[Dict[str, Any]], max_images: int, seed: int) -> List[Dict[str, Any]]:
    seq = [dict(row) for row in rows]
    if max_images <= 0 or max_images >= len(seq):
        return seq
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in seq:
        key = str(row.get("class_name") or row.get("label_id") or "")
        groups.setdefault(key, []).append(row)
    if len(groups) > 1:
        for values in groups.values():
            values.sort(key=lambda row: _stable_row_key(row, seed))
        class_names = sorted(groups, key=lambda name: hashlib.sha256(f"{seed}|class|{name}".encode("utf-8")).hexdigest())
        out: List[Dict[str, Any]] = []
        cursor = {name: 0 for name in class_names}
        while len(out) < max_images:
            progressed = False
            for name in class_names:
                idx = cursor[name]
                if idx >= len(groups[name]):
                    continue
                out.append(groups[name][idx])
                cursor[name] += 1
                progressed = True
                if len(out) >= max_images:
                    break
            if not progressed:
                break
        return out[:max_images]
    return sorted(seq, key=lambda row: _stable_row_key(row, seed))[:max_images]


def _hardlink_or_copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _materialized_subset_is_current(
    dest_root: Path,
    *,
    resolved: Path,
    manifest_path: Optional[Path],
    selected: Sequence[Dict[str, Any]],
    max_images: int,
    selection_seed: int,
) -> bool:
    manifest = _read_json_mapping(dest_root / "manifest.json")
    if not manifest:
        return False
    selection = manifest.get("selection") if isinstance(manifest.get("selection"), dict) else {}
    expected_ids = [str(row.get("sample_id") or row.get("src") or "") for row in selected]
    samples = manifest.get("samples") if isinstance(manifest.get("samples"), list) else []
    actual_ids = [str(row.get("sample_id") or "") for row in samples if isinstance(row, dict)]
    if str(manifest.get("source") or "") != str(resolved):
        return False
    if str(manifest.get("source_manifest") or "") != str(manifest_path or ""):
        return False
    if int(selection.get("seed") or -1) != int(selection_seed):
        return False
    if int(selection.get("requested_images") or -1) != int(max_images):
        return False
    if actual_ids != expected_ids:
        return False
    for sample in samples:
        if not isinstance(sample, dict):
            return False
        rel = str(sample.get("image") or "").strip()
        if not rel or not (dest_root / rel).is_file():
            return False
    return True


def _existing_suite_subset_root(
    suite_dir: Path,
    resolved: Path,
    *,
    max_images: int,
    selection_seed: int,
) -> Optional[str]:
    """Return an already materialised suite subset without nesting it again.

    Suite refresh is intentionally repeatable and can run once per remote setup.
    On the second refresh ``validation_images`` already points at the subset made
    by the first refresh.  Treating that subset as a new source used to append
    ``_n<k>_s<seed>`` on every pass and eventually left plan rows pointing at a
    path that no longer existed.  The subset manifest is the provenance record;
    the directory is the executable, labelled dataset source consumed by the
    runner.
    """

    source = Path(resolved)
    root = source.parent if source.is_file() and source.name == "manifest.json" else source
    try:
        suite = Path(suite_dir).resolve()
        root = root.resolve()
        rel = root.relative_to(suite)
    except (OSError, ValueError):
        return None
    expected_parent = Path("resources") / "validation" / "classification"
    if rel.parent != expected_parent or not root.is_dir():
        return None
    manifest = _read_json_mapping(root / "manifest.json")
    if not manifest or str(manifest.get("source_type") or "") != "materialized_run_mode_subset":
        return None
    selection = manifest.get("selection") if isinstance(manifest.get("selection"), dict) else {}
    samples = manifest.get("samples") if isinstance(manifest.get("samples"), list) else []
    if (
        int(selection.get("seed") or -1) != int(selection_seed)
        or int(selection.get("requested_images") or -1) != int(max_images)
        or int(selection.get("selected_images") or len(samples)) != len(samples)
        or not samples
    ):
        return None
    for sample in samples:
        if not isinstance(sample, dict):
            return None
        image = str(sample.get("image") or "").strip()
        if not image or not (root / image).is_file():
            return None
        # A materialised classification source is useful only when the label is
        # explicit in the manifest or represented by its class directory.
        if sample.get("label_id") is None and not str(sample.get("class_name") or "").strip():
            return None
    return rel.as_posix()


def _materialize_classification_subset(
    *,
    suite_dir: Path,
    resolved: Path,
    requested: Any,
    max_images: int,
    selection_seed: int,
    explicit_manifest: Any = None,
    base_dir: Optional[Path] = None,
) -> Optional[str]:
    existing_subset = _existing_suite_subset_root(
        suite_dir,
        resolved,
        max_images=int(max_images),
        selection_seed=int(selection_seed),
    )
    if existing_subset:
        return existing_subset

    projection = project_classification_validation_subset(
        resolved=resolved,
        requested=requested,
        max_images=int(max_images),
        selection_seed=int(selection_seed),
        explicit_manifest=explicit_manifest,
        base_dir=base_dir,
    )
    if projection is None:
        return None
    manifest_path = projection["source_manifest"]
    selected = projection["selected"]
    payload = projection["manifest"]
    dest_root_rel = projection["destination_relative"]
    dest_root = Path(suite_dir).resolve() / dest_root_rel

    # Several setup-local remote workers can refresh the same suite at once.
    # Re-materialising the subset in each worker both races destructively and
    # changes mtimes, which defeats the shared bundle cache.  Keep the operation
    # process-atomic and reuse an already complete deterministic subset.
    with _SUBSET_MATERIALIZE_LOCK:
        if _materialized_subset_is_current(
            dest_root,
            resolved=resolved,
            manifest_path=manifest_path,
            selected=selected,
            max_images=int(max_images),
            selection_seed=int(selection_seed),
        ):
            return dest_root_rel.as_posix()

        tmp_root = dest_root.with_name(
            f".{dest_root.name}.tmp-{os.getpid()}-{threading.get_ident()}"
        )
        if tmp_root.exists():
            shutil.rmtree(tmp_root)
        (tmp_root / "images").mkdir(parents=True, exist_ok=True)

        try:
            for row, sample in zip(selected, payload["samples"]):
                src = Path(row["src"])
                dst = tmp_root / str(sample["image"])
                _hardlink_or_copy_file(src, dst)
            out_manifest = tmp_root / "manifest.json"
            out_manifest.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            dest_root.parent.mkdir(parents=True, exist_ok=True)
            if dest_root.exists():
                shutil.rmtree(dest_root)
            os.replace(tmp_root, dest_root)
        finally:
            if tmp_root.exists():
                shutil.rmtree(tmp_root, ignore_errors=True)
    return dest_root_rel.as_posix()


def project_classification_validation_subset(
    *,
    resolved: Path,
    requested: Any,
    max_images: int,
    selection_seed: int,
    explicit_manifest: Any = None,
    base_dir: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    """Project the exact suite-local classification subset without writing it.

    The Evaluation Workflow first selects a deterministic run-mode cohort and
    then materialises the returned manifest and image paths into a portable
    suite.  Keeping the selection and manifest construction in this pure helper
    lets read-only provenance gates prove the future runtime cohort without
    copying images, changing datasets, or maintaining a second implementation
    of the selection algorithm.
    """

    resolved = Path(resolved).resolve()
    manifest_path = _classification_manifest_path(
        resolved,
        explicit_manifest=explicit_manifest,
        base_dir=base_dir,
    )
    source_root, _rel_inside = _dataset_root_for_source(resolved)
    rows = (
        _classification_rows_from_manifest(
            manifest_path, fallback_root=source_root,
        )
        if manifest_path else []
    )
    if not rows:
        rows = _classification_rows_from_directory(source_root)
    selected = _select_classification_rows(
        rows, int(max_images), int(selection_seed),
    )
    if not selected:
        return None

    preset = normalize_classification_validation_preset(requested)
    base_name = preset or _safe_manifest_name(source_root.name)
    suffix = f"_n{len(selected)}_s{int(selection_seed)}"
    dest_name = _safe_manifest_name(base_name + suffix)
    destination_relative = (
        Path("resources") / "validation" / "classification" / dest_name
    )

    samples: List[Dict[str, Any]] = []
    used_images: set[str] = set()
    for idx, row in enumerate(selected):
        src = Path(row["src"])
        class_name = (
            str(row.get("class_name") or "unlabeled").strip()
            or "unlabeled"
        )
        safe_class = _safe_manifest_name(class_name)
        file_name = src.name
        relative_image = (
            Path("images") / safe_class / file_name
        ).as_posix()
        if relative_image in used_images:
            relative_image = (
                Path("images") / safe_class / f"{idx:06d}_{file_name}"
            ).as_posix()
        used_images.add(relative_image)
        sample: Dict[str, Any] = {
            "image": relative_image,
            "source_image": str(src),
            "sample_id": str(row.get("sample_id") or src.name),
            "class_name": class_name,
        }
        if row.get("label_id") is not None:
            sample["label_id"] = int(row["label_id"])
        if row.get("label_name"):
            sample["label_name"] = str(row["label_name"])
        if row.get("sha256"):
            sample["source_sha256"] = str(row["sha256"])
        if row.get("size_bytes"):
            sample["source_size_bytes"] = int(row["size_bytes"])
        samples.append(sample)

    payload = {
        "schema": "onnx-splitpoint/classification-validation-manifest",
        "schema_version": 2,
        "dataset": dest_name,
        "source_type": "materialized_run_mode_subset",
        "source": str(resolved),
        "source_manifest": str(manifest_path or ""),
        "selection": {
            "type": "deterministic_class_stratified",
            "seed": int(selection_seed),
            "requested_images": int(max_images),
            "selected_images": len(samples),
            "source_population": len(rows),
        },
        "samples": samples,
    }
    return {
        "manifest": payload,
        "selected": selected,
        "source_manifest": manifest_path,
        "source_root": source_root,
        "destination_relative": destination_relative,
    }


def provision_classification_validation_source_to_suite(
    suite_dir: Path,
    requested: Any,
    *,
    base_dir: Optional[Path] = None,
    max_images: int = 0,
    manifest_path: Any = None,
    selection_seed: int = 20260710,
) -> Optional[str]:
    """Provision a self-contained classification validation source.

    Positive ``max_images`` values materialise only the effective run-mode
    subset.  This is the important distinction between a registered source
    dataset and the suite-local transport payload: a 16-image Smoke run must
    never copy/package all 50,000 ImageNet validation files.
    """
    resolved = resolve_classification_validation_source(requested, base_dir=base_dir)
    if resolved is None:
        return None
    if int(max_images or 0) > 0:
        subset = _materialize_classification_subset(
            suite_dir=Path(suite_dir),
            resolved=resolved,
            requested=requested,
            max_images=int(max_images),
            selection_seed=int(selection_seed),
            explicit_manifest=manifest_path,
            base_dir=base_dir,
        )
        if subset:
            return subset

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
        # Always hand the executable dataset root to the benchmark runner.  It
        # discovers ``manifest.json`` inside that directory and therefore keeps
        # labels and paths relative to one stable root.  Returning the manifest
        # itself made other consumers treat a JSON file as an image directory.
        return dest_root_rel.as_posix()
    dest_root.mkdir(parents=True, exist_ok=True)
    dst_file = dest_root / source_root.name
    shutil.copy2(source_root, dst_file)
    return dest_root_rel.as_posix()


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
