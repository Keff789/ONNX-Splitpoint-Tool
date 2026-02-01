#!/usr/bin/env python3
"""Download a curated set of ONNX Model Zoo examples.

Why this exists
---------------
For evaluation it is useful to run the analyser / benchmark suite on *many* networks.
Since July 2025, the original onnx/models Git LFS downloads are deprecated; the
models are mirrored in the Hugging Face `onnxmodelzoo/legacy_models` repository.

This helper downloads the model archives (.tar.gz) and extracts them locally.
You can then open the extracted .onnx files in the GUI.

Usage examples
--------------
List available models:
  python download_model_zoo_examples.py --list

Download a few models into ./model_zoo_downloads:
  python download_model_zoo_examples.py --models mobilenetv2-7 resnet50-v1-12 efficientnet-lite4-11

Download everything into a custom folder:
  python download_model_zoo_examples.py --all --out C:\\path\\to\\models

Notes
-----
- The manifest can be extended by editing model_zoo_manifest.json.
- For very large models, download may take a while depending on your network.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


def _load_manifest(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing manifest: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_extract(tar: tarfile.TarFile, path: Path) -> None:
    """Safely extract tar contents into `path` (prevents path traversal)."""
    path = path.resolve()
    for member in tar.getmembers():
        member_path = (path / member.name).resolve()
        if not str(member_path).startswith(str(path) + os.sep) and member_path != path:
            raise RuntimeError(f"Unsafe path in tar archive: {member.name}")
    tar.extractall(path)


def _download(url: str, dst: Path, *, overwrite: bool = False, quiet: bool = False) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists() and not overwrite:
        if not quiet:
            print(f"[skip] {dst.name} already exists")
        return

    # Basic progress printing (works in PowerShell / CMD)
    def _reporthook(blocknum: int, blocksize: int, totalsize: int) -> None:
        if quiet:
            return
        downloaded = blocknum * blocksize
        if totalsize > 0:
            pct = min(100.0, 100.0 * downloaded / totalsize)
            mb = downloaded / (1024 * 1024)
            total_mb = totalsize / (1024 * 1024)
            sys.stdout.write(f"\r[dl] {dst.name}: {pct:6.2f}% ({mb:,.1f} / {total_mb:,.1f} MiB)")
            sys.stdout.flush()
        else:
            mb = downloaded / (1024 * 1024)
            sys.stdout.write(f"\r[dl] {dst.name}: {mb:,.1f} MiB")
            sys.stdout.flush()

    if not quiet:
        print(f"[get] {url}")

    try:
        urllib.request.urlretrieve(url, dst.as_posix(), reporthook=_reporthook)
    except urllib.error.HTTPError as e:
        # Surface the URL so the user can verify it quickly.
        raise RuntimeError(f"HTTP error {e.code} while downloading {url}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"URL error while downloading {url}: {e}") from e
    finally:
        if not quiet:
            sys.stdout.write("\n")


def _find_onnx_files(root: Path) -> List[Path]:
    return sorted([p for p in root.rglob("*.onnx") if p.is_file()])


def _maybe_print_recommendation(entry: Dict) -> None:
    preset = entry.get("suggested_preset")
    if preset:
        print(f"      suggested preset: {preset}")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=str, default=str(Path(__file__).with_name("model_zoo_manifest.json")))
    ap.add_argument("--out", type=str, default="model_zoo_downloads")
    ap.add_argument("--list", action="store_true", help="List available models and exit")
    ap.add_argument("--all", action="store_true", help="Download all models from the manifest")
    ap.add_argument("--models", nargs="*", default=[], help="One or more model IDs from the manifest")
    ap.add_argument("--overwrite", action="store_true", help="Re-download archives even if they already exist")
    ap.add_argument("--keep-archive", action="store_true", help="Keep the downloaded .tar.gz next to the extracted files")
    ap.add_argument("--no-extract", action="store_true", help="Only download the archive; do not extract")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    manifest = _load_manifest(Path(args.manifest))
    base = manifest.get("base", {})
    resolve_url = base.get("resolve_url")
    if not resolve_url:
        raise SystemExit("Manifest missing base.resolve_url")

    entries: List[Dict] = manifest.get("models", [])
    by_id = {e.get("id"): e for e in entries if e.get("id")}

    if args.list:
        print("Available models (from model_zoo_manifest.json):")
        for mid in sorted(by_id.keys()):
            e = by_id[mid]
            task = e.get("task", "?")
            print(f"  - {mid} ({task})")
        return 0

    if args.all:
        selected_ids = list(by_id.keys())
    else:
        selected_ids = list(args.models)

    if not selected_ids:
        print("Nothing selected. Use --list to see IDs, then --models <id>... or --all.")
        return 2

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    # Store the manifest snapshot that was used.
    (out_root / "_manifest_used.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    ok = 0
    for mid in selected_ids:
        entry = by_id.get(mid)
        if not entry:
            print(f"[warn] unknown model id: {mid} (skipping)")
            continue

        rel = entry.get("archive_path")
        if not rel:
            print(f"[warn] model {mid} has no archive_path (skipping)")
            continue

        url = resolve_url.rstrip("/") + "/" + rel.lstrip("/")
        model_dir = out_root / mid
        model_dir.mkdir(parents=True, exist_ok=True)

        archive_name = Path(rel).name
        archive_path = model_dir / archive_name

        try:
            _download(url, archive_path, overwrite=args.overwrite, quiet=args.quiet)
        except Exception as e:
            print(f"[fail] {mid}: {e}")
            continue

        if args.no_extract:
            ok += 1
            continue

        try:
            if not args.quiet:
                print(f"[extract] {archive_path.name} -> {model_dir}")
            with tarfile.open(archive_path, "r:gz") as tar:
                _safe_extract(tar, model_dir)
        except Exception as e:
            print(f"[fail] {mid}: extraction failed: {e}")
            continue

        onnx_files = _find_onnx_files(model_dir)
        if onnx_files:
            # Prefer files that live in a 'model/' directory, otherwise the largest file.
            preferred = [p for p in onnx_files if "model" in {pp.name.lower() for pp in p.parents}]
            if preferred:
                chosen = preferred[0]
            else:
                chosen = max(onnx_files, key=lambda p: p.stat().st_size)
            if not args.quiet:
                print(f"[ok] {mid}: {chosen}")
                _maybe_print_recommendation(entry)
        else:
            print(f"[warn] {mid}: extracted but no .onnx found in {model_dir}")

        if not args.keep_archive:
            try:
                archive_path.unlink(missing_ok=True)  # type: ignore[arg-type]
            except Exception:
                pass

        ok += 1

    if ok == 0:
        print("No models downloaded.")
        return 1

    print(f"Done. Downloaded/extracted {ok} model(s) into: {out_root.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
