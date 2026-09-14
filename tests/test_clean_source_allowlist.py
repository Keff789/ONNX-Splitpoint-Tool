from __future__ import annotations

import importlib.util
import re
from pathlib import Path

from onnx_splitpoint_tool import __version__


ROOT = Path(__file__).resolve().parents[1]


def _manifest_builder():
    path = ROOT / "scripts" / "build_source_manifest.py"
    spec = importlib.util.spec_from_file_location(
        "clean_source_manifest_test",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_clean_source_manifest_uses_strict_current_release_allowlist() -> None:
    builder = _manifest_builder()
    rows = builder._files(ROOT, package_version=__version__)
    included = {row["path"] for row in rows}

    assert {
        "README.md",
        "docs/HOLDOUT_ADAPTER_PROTOCOL.md",
        "pyproject.toml",
        "start_gui.sh",
        "onnx_splitpoint_tool/__init__.py",
        "scripts/build_source_release.py",
        "tests/conftest.py",
    } <= included

    # The current release's exact guide/report pair is explicitly allowlisted;
    # arbitrary historical release notes remain outside the clean inventory.
    assert f"TESTANLEITUNG_{__version__}.md" in included
    assert f"VERSION_{__version__}_BUILD_AND_TEST_REPORT.md" in included

    forbidden = {
        "TESTANLEITUNG_2.70c.md",
        "VERSION_2.69_BUILD_AND_TEST_REPORT.md",
        "OFFLINE_REPLAY_2.72.2.md",
        "profiles/legacy_v58/README.md",
        "onnx_splitpoint_tool.egg-info/PKG-INFO",
    }
    assert included.isdisjoint(forbidden)
    assert not any(path.startswith("docs/RELEASE_") for path in included)
    assert not any(path.startswith("docs/OPEN_ITEMS_") for path in included)
    assert not any("/__pycache__/" in f"/{path}/" for path in included)


def test_allowlisted_markdown_links_stay_inside_the_clean_release() -> None:
    builder = _manifest_builder()
    rows = builder._files(ROOT, package_version=__version__)
    included = {row["path"] for row in rows}
    failures: list[str] = []

    for relative in sorted(
        path for path in included if path.endswith(".md")
    ):
        source = ROOT / relative
        for target in re.findall(
            r"\[[^]]+\]\(([^)]+)\)",
            source.read_text(encoding="utf-8"),
        ):
            if target.startswith(("#", "http://", "https://", "mailto:")):
                continue
            logical = target.split("#", 1)[0]
            if not logical:
                continue
            try:
                resolved = (source.parent / logical).resolve().relative_to(
                    ROOT
                ).as_posix()
            except ValueError:
                failures.append(f"{relative}: path escapes release: {target}")
                continue
            if resolved not in included:
                failures.append(
                    f"{relative}: target not shipped: {target}"
                )

    assert failures == []
