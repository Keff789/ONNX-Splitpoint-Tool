from __future__ import annotations

"""Read-only replay of scientific reports from an existing EvaluationRun.

Only derived presentation artefacts are written.  The source EvaluationRun is
never used as an output root, and an existing destination is never replaced.
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from ..filesystem_admission import (
    require_output_outside_source,
    require_write_target,
)
from .scientific_reporting import build_scientific_reports


def replay_scientific_reports(
    run_dir: str | Path,
    output_dir: str | Path,
    *,
    native_run_dir: str | Path | None = None,
    profile_id: str = "",
    tool_version: str = "",
    workflow_version: str = "",
) -> dict[str, Any]:
    """Reproject one EvaluationRun into a new, separate report directory.

    The destination must not exist.  Report generation happens in a temporary
    sibling directory and is published with one atomic rename only after the
    complete reporter returns successfully.
    """

    source = Path(run_dir).expanduser()
    if source.is_symlink():
        raise RuntimeError(
            f"Offline scientific replay source must not be a symlink: {source}"
        )
    source = source.resolve(strict=True)
    if not source.is_dir():
        raise RuntimeError(
            f"Offline scientific replay source is not a directory: {source}"
        )

    native_source = source
    if native_run_dir is not None:
        native_candidate = Path(native_run_dir).expanduser()
        if native_candidate.is_symlink():
            raise RuntimeError(
                "Offline Native replay source must not be a symlink: "
                f"{native_candidate}"
            )
        native_source = native_candidate.resolve(strict=True)
        if not native_source.is_dir():
            raise RuntimeError(
                "Offline Native replay source is not a directory: "
                f"{native_source}"
            )

    _source, destination = require_output_outside_source(
        source,
        Path(output_dir).expanduser(),
        operation="Offline scientific report replay",
    )
    if native_source != source:
        require_output_outside_source(
            native_source,
            destination,
            operation="Offline Native scientific report replay",
        )
    if destination.exists() or destination.is_symlink():
        raise RuntimeError(
            "Offline scientific report replay destination already exists; "
            f"choose a new directory: {destination}"
        )
    require_write_target(
        destination.parent,
        operation="Offline scientific report replay",
        minimum_free_bytes=16 * 1024 * 1024,
        minimum_free_inodes=32,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)

    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.partial-",
            dir=str(destination.parent),
        )
    )
    try:
        result = build_scientific_reports(
            source,
            profile_id=profile_id,
            tool_version=tool_version,
            workflow_version=workflow_version,
            cleanup_legacy=False,
            output_dir=temporary,
            native_source_dir=native_source,
        )
        # ``build_scientific_reports`` recreates its explicit output subtree.
        # Refuse publication if its mandatory canonical output is absent.
        if not (temporary / "scientific_report.json").is_file():
            raise RuntimeError(
                "Offline scientific report replay produced no "
                "scientific_report.json"
            )
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    remapped_artifacts: dict[str, Path] = {}
    for key, path in dict(result.get("artifacts") or {}).items():
        candidate = Path(path)
        try:
            relative = candidate.relative_to(temporary)
        except ValueError:
            remapped_artifacts[str(key)] = candidate
        else:
            remapped_artifacts[str(key)] = destination / relative

    return {
        **result,
        "artifacts": remapped_artifacts,
        "source_run_dir": str(source),
        "native_source_run_dir": str(native_source),
        "native_source_is_separate": native_source != source,
        "output_dir": str(destination),
        "source_mutated": False,
        "execution_mode": "offline_read_only_report_replay",
    }


__all__ = ["replay_scientific_reports"]
