"""Helpers for package-bundled resources.

These utilities keep resource access working both from a source checkout and from
an installed wheel. Most call sites only need a temporary filesystem path or the
text content of a bundled template.
"""

from __future__ import annotations

import atexit
import importlib
import shutil
from contextlib import ExitStack, contextmanager
from importlib import resources as importlib_resources
from pathlib import Path
from typing import Iterator


_PACKAGE = "onnx_splitpoint_tool"
_RESOURCE_STACK = ExitStack()
atexit.register(_RESOURCE_STACK.close)


def _resource_ref(*parts: str):
    files_api = getattr(importlib_resources, "files", None)
    if files_api is not None:
        ref = files_api(_PACKAGE)
        for part in parts:
            ref = ref.joinpath(part)
        return ref
    # Python 3.8 has no importlib.resources.files/as_file Traversable API.
    # Normal wheel and source installs expose package resources as real files,
    # so resolve them through the imported package without requiring a new
    # dependency on the remote accelerator image.
    package = importlib.import_module(_PACKAGE)
    package_file = Path(str(getattr(package, "__file__", ""))).resolve()
    if not package_file.is_file():
        raise FileNotFoundError(f"package filesystem root unavailable for {_PACKAGE}")
    return package_file.parent.joinpath(*parts)


@contextmanager
def resource_path(*parts: str) -> Iterator[Path]:
    """Yield a filesystem path for a packaged resource.

    The returned path may be a temporary extracted path when the package is
    imported from a zip/wheel implementation that does not expose the resource
    directly on disk.
    """

    ref = _resource_ref(*parts)
    as_file = getattr(importlib_resources, "as_file", None)
    if isinstance(ref, Path) or as_file is None:
        yield Path(ref)
    else:
        with as_file(ref) as p:
            yield Path(p)



def persistent_resource_path(*parts: str) -> Path:
    """Return a filesystem path that stays valid for the current process."""

    ref = _resource_ref(*parts)
    as_file = getattr(importlib_resources, "as_file", None)
    if isinstance(ref, Path) or as_file is None:
        return Path(ref)
    return Path(_RESOURCE_STACK.enter_context(as_file(ref)))



def read_text(*parts: str, encoding: str = "utf-8") -> str:
    with resource_path(*parts) as p:
        return p.read_text(encoding=encoding)



def copy_resource_tree(*parts: str, dest: Path) -> None:
    """Copy a packaged directory tree to *dest*.

    Existing destinations are removed first.
    """

    with resource_path(*parts) as src:
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(src, dest, ignore=_ignore_noise)



def copy_resource_file(*parts: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with resource_path(*parts) as src:
        shutil.copy2(src, dest)



def _ignore_noise(_dir: str, names: list[str]) -> set[str]:
    ignored = {"__pycache__", ".pytest_cache", ".git"}
    ignored.update({n for n in names if n.endswith((".pyc", ".pyo", ".orig"))})
    return ignored
