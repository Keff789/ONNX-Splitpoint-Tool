#!/usr/bin/env python3
"""Refresh one editable distribution without pip or a build backend.

The source updater preserves ``.venv``.  This helper updates only the current
project's PEP 376 metadata, editable path and console launchers using Python's
standard library.  Existing dependencies and unrelated distributions are not
touched.  All replacement artifacts are staged beside their final paths and
rolled back if installation or verification fails.
"""
from __future__ import annotations

import argparse
import ast
import base64
import configparser
import csv
import email.parser
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess
import sys
import sysconfig
import uuid
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple
from urllib.parse import unquote, urlparse


ENTRY_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
ENTRY_VALUE_RE = re.compile(
    r"^(?P<module>[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)"
    r":(?P<attribute>[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)$"
)
BARE_KEY_RE = re.compile(r"^[A-Za-z0-9_-]+$")


class RefreshError(RuntimeError):
    """Raised when a safe stdlib-only refresh cannot be completed."""


@dataclass(frozen=True)
class ProjectMetadata:
    root: Path
    name: str
    version: str
    description: str
    requires_python: str
    dependencies: Tuple[str, ...]
    scripts: Mapping[str, str]
    top_levels: Tuple[str, ...]


def _canonical_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _has_canonical_token(value: str, canonical: str) -> bool:
    return f"-{canonical}-" in f"-{_canonical_name(value)}-"


def _wheel_component(value: str) -> str:
    component = re.sub(r"[^A-Za-z0-9.]+", "_", value)
    if not component or component in {".", ".."}:
        raise RefreshError(f"unsafe distribution component: {value!r}")
    return component


def _without_comment(line: str) -> str:
    quote: Optional[str] = None
    escaped = False
    for index, character in enumerate(line):
        if quote is not None:
            if escaped:
                escaped = False
            elif quote == '"' and character == "\\":
                escaped = True
            elif character == quote:
                quote = None
        elif character in {'"', "'"}:
            quote = character
        elif character == "#":
            return line[:index]
    return line


def _bracket_delta(value: str) -> int:
    quote: Optional[str] = None
    escaped = False
    delta = 0
    for character in value:
        if quote is not None:
            if escaped:
                escaped = False
            elif quote == '"' and character == "\\":
                escaped = True
            elif character == quote:
                quote = None
        elif character in {'"', "'"}:
            quote = character
        elif character == "[":
            delta += 1
        elif character == "]":
            delta -= 1
    return delta


def _split_assignment(line: str) -> Tuple[str, str]:
    quote: Optional[str] = None
    escaped = False
    for index, character in enumerate(line):
        if quote is not None:
            if escaped:
                escaped = False
            elif quote == '"' and character == "\\":
                escaped = True
            elif character == quote:
                quote = None
        elif character in {'"', "'"}:
            quote = character
        elif character == "=":
            return line[:index].strip(), line[index + 1 :].strip()
    raise RefreshError(f"expected TOML assignment: {line!r}")


def _toml_key(raw: str) -> str:
    if BARE_KEY_RE.fullmatch(raw):
        return raw
    return _toml_string(raw)


def _toml_string(raw: str) -> str:
    try:
        value = ast.literal_eval(raw)
    except (SyntaxError, ValueError) as exc:
        raise RefreshError(f"unsupported TOML string: {raw!r}") from exc
    if not isinstance(value, str) or "\x00" in value:
        raise RefreshError(f"expected a safe TOML string: {raw!r}")
    return value


def _table_values(text: str) -> Dict[str, Dict[str, str]]:
    wanted = {"project", "project.scripts"}
    values: Dict[str, Dict[str, str]] = {name: {} for name in wanted}
    section = ""
    pending_section = ""
    pending_key = ""
    pending_parts: List[str] = []
    pending_depth = 0

    def store(table: str, key: str, raw: str) -> None:
        if table not in wanted:
            return
        if key in values[table]:
            raise RefreshError(f"duplicate [{table}] key: {key}")
        values[table][key] = raw.strip()

    for source_line in text.splitlines():
        line = _without_comment(source_line).strip()
        if pending_key:
            if line:
                pending_parts.append(line)
                pending_depth += _bracket_delta(line)
            if pending_depth < 0:
                raise RefreshError(
                    f"unbalanced TOML array for {pending_key!r}"
                )
            if pending_depth == 0:
                store(
                    pending_section,
                    pending_key,
                    " ".join(pending_parts),
                )
                pending_section = ""
                pending_key = ""
                pending_parts = []
            continue
        if not line:
            continue
        if line.startswith("["):
            if not line.endswith("]"):
                raise RefreshError(f"unsupported TOML table header: {line!r}")
            if line.startswith("[["):
                section = ""
            else:
                section = line[1:-1].strip()
            continue
        if section not in wanted:
            continue
        try:
            raw_key, raw_value = _split_assignment(line)
        except RefreshError:
            if section == "project.scripts":
                raise
            continue
        key = _toml_key(raw_key)
        depth = _bracket_delta(raw_value)
        if depth < 0:
            raise RefreshError(f"unbalanced TOML value for {key!r}")
        if depth > 0:
            pending_section = section
            pending_key = key
            pending_parts = [raw_value]
            pending_depth = depth
        else:
            store(section, key, raw_value)
    if pending_key:
        raise RefreshError(f"unterminated TOML array for {pending_key!r}")
    return values


def load_project(root: Path) -> ProjectMetadata:
    root = root.expanduser().resolve()
    if any(character in str(root) for character in "\r\n\x00"):
        raise RefreshError(f"unsafe project root path: {root!s}")
    pyproject = root / "pyproject.toml"
    if not pyproject.is_file() or pyproject.is_symlink():
        raise RefreshError(f"safe pyproject.toml missing: {pyproject}")
    tables = _table_values(pyproject.read_text(encoding="utf-8"))
    project = tables["project"]
    scripts_raw = tables["project.scripts"]
    for required in ("name", "version"):
        if required not in project:
            raise RefreshError(f"[project].{required} is required")
    name = _toml_string(project["name"])
    version = _toml_string(project["version"])
    if any(character in name + version for character in "\r\n\x00"):
        raise RefreshError("distribution name/version contains control data")

    description = (
        _toml_string(project["description"])
        if "description" in project
        else ""
    )
    requires_python = (
        _toml_string(project["requires-python"])
        if "requires-python" in project
        else ""
    )
    dependencies: Tuple[str, ...] = ()
    if "dependencies" in project:
        try:
            parsed_dependencies = ast.literal_eval(project["dependencies"])
        except (SyntaxError, ValueError) as exc:
            raise RefreshError("unsupported [project].dependencies") from exc
        if not isinstance(parsed_dependencies, list) or not all(
            isinstance(item, str) and "\n" not in item and "\r" not in item
            for item in parsed_dependencies
        ):
            raise RefreshError("[project].dependencies must be strings")
        dependencies = tuple(parsed_dependencies)

    scripts: Dict[str, str] = {}
    top_levels: Set[str] = set()
    for raw_name, raw_value in scripts_raw.items():
        entry_name = raw_name
        entry_value = _toml_string(raw_value)
        match = ENTRY_VALUE_RE.fullmatch(entry_value)
        if not ENTRY_NAME_RE.fullmatch(entry_name) or match is None:
            raise RefreshError(
                f"unsafe console entry point: {entry_name}={entry_value!r}"
            )
        module = match.group("module")
        top_level = module.split(".", 1)[0]
        if not (
            (root / f"{top_level}.py").is_file()
            or (root / top_level / "__init__.py").is_file()
        ):
            raise RefreshError(
                f"console entry point module is outside the project: {module}"
            )
        scripts[entry_name] = entry_value
        top_levels.add(top_level)
    if not top_levels:
        guessed = name.replace("-", "_").replace(".", "_")
        if (root / guessed / "__init__.py").is_file():
            top_levels.add(guessed)
        elif (root / f"{guessed}.py").is_file():
            top_levels.add(guessed)
        else:
            raise RefreshError("could not identify an importable project module")

    return ProjectMetadata(
        root=root,
        name=name,
        version=version,
        description=description,
        requires_python=requires_python,
        dependencies=dependencies,
        scripts=dict(sorted(scripts.items())),
        top_levels=tuple(sorted(top_levels)),
    )


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _safe_existing(path: Path) -> None:
    if path.is_symlink():
        raise RefreshError(f"refusing distribution artifact symlink: {path}")
    if not (path.is_file() or path.is_dir()):
        raise RefreshError(f"unsupported distribution artifact: {path}")


def _metadata_name(path: Path) -> Optional[str]:
    metadata_path = path / ("METADATA" if path.suffix == ".dist-info" else "PKG-INFO")
    if path.is_file() and path.suffix == ".egg-info":
        metadata_path = path
    if metadata_path.is_symlink():
        raise RefreshError(f"distribution identity metadata is a symlink: {metadata_path}")
    if not metadata_path.is_file():
        return None
    try:
        text = metadata_path.read_text(encoding="utf-8")
        message = email.parser.Parser().parsestr(text)
    except (OSError, UnicodeError):
        return None
    value = message.get("Name")
    return str(value).strip() if value else None


def _old_entry_points(metadata_path: Path) -> Dict[str, str]:
    path = metadata_path / "entry_points.txt"
    if metadata_path.is_file():
        return {}
    if path.is_symlink():
        raise RefreshError(f"old entry_points.txt is a symlink: {path}")
    if not path.is_file():
        return {}
    parser = configparser.ConfigParser(interpolation=None, strict=True)
    parser.optionxform = str
    try:
        parser.read(path, encoding="utf-8")
    except (configparser.Error, OSError, UnicodeError) as exc:
        raise RefreshError(f"invalid old entry_points.txt: {path}") from exc
    if not parser.has_section("console_scripts"):
        return {}
    result: Dict[str, str] = {}
    for name, value in parser.items("console_scripts"):
        if not ENTRY_NAME_RE.fullmatch(name):
            raise RefreshError(f"unsafe old console entry point: {name!r}")
        result[name] = value.strip()
    return result


def _record_candidates(
    metadata_path: Path,
    *,
    allowed_roots: Sequence[Path],
) -> List[Tuple[str, Path]]:
    record = metadata_path / "RECORD"
    if metadata_path.is_file():
        return []
    if record.is_symlink():
        raise RefreshError(f"old RECORD is a symlink: {record}")
    if not record.is_file():
        return []
    result: List[Tuple[str, Path]] = []
    try:
        with record.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.reader(handle))
    except (csv.Error, OSError, UnicodeError) as exc:
        raise RefreshError(f"invalid old RECORD: {record}") from exc
    for row in rows:
        if len(row) != 3 or not row[0] or "\\" in row[0]:
            raise RefreshError(f"unsafe old RECORD row in {record}: {row!r}")
        logical = PurePosixPath(row[0])
        if logical.is_absolute() or logical.as_posix() != row[0]:
            raise RefreshError(f"absolute old RECORD path: {row[0]!r}")
        candidate = metadata_path.parent.joinpath(*logical.parts).resolve()
        if not any(_is_within(candidate, root) for root in allowed_roots):
            raise RefreshError(
                f"old RECORD path escapes venv/tool roots: {row[0]!r}"
            )
        result.append((logical.as_posix(), candidate))
    return result


def _legacy_v27911_launcher_content(entry_value: str) -> bytes:
    """Return the exact launcher emitted by the v2.79.11 refresher.

    The v2.79.12 launcher gained isolated bytecode handling.  Existing
    v2.79.11 launchers are nevertheless trusted predecessor artefacts when,
    and only when, every byte matches this frozen template for the declared
    entry point and preserved interpreter.
    """

    match = ENTRY_VALUE_RE.fullmatch(entry_value)
    if match is None:
        raise RefreshError(f"invalid launcher entry point: {entry_value!r}")
    if any(character in sys.executable for character in "\r\n\x00"):
        raise RefreshError("unsafe target Python executable path")
    python = shlex.quote(sys.executable)
    module = match.group("module")
    attribute = match.group("attribute")
    content = f"""#!/bin/sh
'''exec' {python} "$0" "$@"
' '''
# -*- coding: utf-8 -*-
import importlib
import sys

target = importlib.import_module({module!r})
for component in {attribute!r}.split('.'):
    target = getattr(target, component)
if not callable(target):
    raise SystemExit('console entry point is not callable: {entry_value}')
raise SystemExit(target())
"""
    compile(content, f"<{module}:{attribute}>", "exec")
    return content.encode("utf-8")


def _launcher_targets_entry_point(path: Path, entry_value: str) -> bool:
    """Recognize this installer or the standard distlib launcher shape."""
    _safe_existing(path)
    if not path.is_file() or path.stat().st_size > 1024 * 1024:
        return False
    payload = path.read_bytes()
    if payload in {
        _launcher_content(entry_value),
        _legacy_v27911_launcher_content(entry_value),
    }:
        return True
    try:
        tree = ast.parse(payload.decode("utf-8"), filename=str(path))
    except (SyntaxError, UnicodeError):
        return False
    match = ENTRY_VALUE_RE.fullmatch(entry_value)
    if match is None or "." in match.group("attribute"):
        return False
    module = match.group("module")
    attribute = match.group("attribute")
    imported = any(
        isinstance(node, ast.ImportFrom)
        and node.module == module
        and any(alias.name == attribute for alias in node.names)
        for node in ast.walk(tree)
    )
    exits_through_target = any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "sys"
        and node.func.attr == "exit"
        and len(node.args) == 1
        and isinstance(node.args[0], ast.Call)
        and isinstance(node.args[0].func, ast.Name)
        and node.args[0].func.id == attribute
        for node in ast.walk(tree)
    )
    return imported and exits_through_target


def _entry_module_is_project_owned(
    project: ProjectMetadata,
    entry_value: str,
) -> bool:
    match = ENTRY_VALUE_RE.fullmatch(entry_value)
    if match is None:
        return False
    module_parts = match.group("module").split(".")
    if module_parts[0] not in set(project.top_levels):
        return False
    module_path = project.root.joinpath(*module_parts)
    return (
        module_path.with_suffix(".py").is_file()
        or (module_path / "__init__.py").is_file()
    )


def _site_and_script_paths() -> Tuple[Tuple[Path, ...], Path, Path]:
    paths = sysconfig.get_paths()
    site_roots = tuple(
        sorted(
            {
                Path(paths[key]).resolve()
                for key in ("purelib", "platlib")
                if paths.get(key)
            },
            key=lambda item: str(item),
        )
    )
    scripts = Path(paths["scripts"]).resolve()
    venv_root = Path(sys.prefix).resolve()
    if sys.prefix == sys.base_prefix:
        raise RefreshError("refresh must run with the preserved target venv")
    for path in tuple(site_roots) + (scripts,):
        if not _is_within(path, venv_root):
            raise RefreshError(f"venv install path escapes sys.prefix: {path}")
        path.mkdir(parents=True, exist_ok=True)
    return site_roots, scripts, venv_root


def discover_old_artifacts(
    project: ProjectMetadata,
    site_roots: Sequence[Path],
    scripts_root: Path,
    venv_root: Path,
) -> Tuple[Set[Path], Set[str]]:
    own = _canonical_name(project.name)
    allowed_roots = tuple(site_roots) + (scripts_root, venv_root, project.root)
    metadata_paths: List[Path] = []
    candidates: List[Path] = []
    for site_root in site_roots:
        candidates.extend(site_root.glob("*.dist-info"))
        candidates.extend(site_root.glob("*.egg-info"))
    candidates.extend(project.root.glob("*.egg-info"))

    for candidate in sorted(set(candidates), key=lambda item: str(item)):
        _safe_existing(candidate)
        declared_name = _metadata_name(candidate)
        filename_matches = _has_canonical_token(candidate.name, own)
        if declared_name and _canonical_name(declared_name) == own:
            metadata_paths.append(candidate.resolve())
        elif filename_matches and candidate.suffix in {".dist-info", ".egg-info"}:
            raise RefreshError(
                f"malformed metadata with own distribution filename: {candidate}"
            )

    artifacts: Set[Path] = set(metadata_paths)
    owned_entry_names: Set[str] = set()
    record_paths: Set[Path] = set()
    for metadata_path in metadata_paths:
        entries = _old_entry_points(metadata_path)
        metadata_record_rows = set(
            _record_candidates(metadata_path, allowed_roots=allowed_roots)
        )
        metadata_record_paths = {
            resolved for _logical, resolved in metadata_record_rows
        }
        record_paths.update(metadata_record_paths)
        for entry_name, entry_value in entries.items():
            if not _entry_module_is_project_owned(project, entry_value):
                raise RefreshError(
                    "canonical old distribution declares a non-project "
                    f"console target: {entry_name}={entry_value!r}"
                )
            for suffix in ("", ".exe", "-script.py", ".cmd"):
                launcher = scripts_root / f"{entry_name}{suffix}"
                resolved_launcher = launcher.resolve()
                expected_logical = _record_name(
                    resolved_launcher,
                    metadata_path.parent,
                )
                if (
                    expected_logical,
                    resolved_launcher,
                ) not in metadata_record_rows:
                    continue
                if not (launcher.exists() or launcher.is_symlink()):
                    continue
                if not _launcher_targets_entry_point(launcher, entry_value):
                    raise RefreshError(
                        "RECORD-owned console launcher does not target its "
                        f"declared project entry point: {launcher}"
                    )
                artifacts.add(resolved_launcher)
                owned_entry_names.add(entry_name)

    for candidate in record_paths:
        name = candidate.name
        canonical_file = _canonical_name(name)
        if any(_is_within(candidate, site_root) for site_root in site_roots):
            if (
                name.endswith(".pth")
                or name.endswith(".egg-link")
                or name.endswith("_finder.py")
            ) and _has_canonical_token(canonical_file, own):
                if candidate.exists() or candidate.is_symlink():
                    _safe_existing(candidate)
                    artifacts.add(candidate)

    for site_root in site_roots:
        for candidate in site_root.iterdir():
            name = candidate.name
            if not name.startswith("__editable__"):
                continue
            if not _has_canonical_token(name, own):
                continue
            if not (
                name.endswith(".pth") or name.endswith("_finder.py")
            ):
                continue
            _safe_existing(candidate)
            artifacts.add(candidate.resolve())

    return artifacts, owned_entry_names


def _hash_record(path: Path) -> Tuple[str, str]:
    payload = path.read_bytes()
    digest = base64.urlsafe_b64encode(hashlib.sha256(payload).digest())
    return "sha256=" + digest.rstrip(b"=").decode("ascii"), str(len(payload))


def _record_name(path: Path, site_root: Path) -> str:
    return Path(os.path.relpath(str(path), str(site_root))).as_posix()


def _launcher_content(entry_value: str) -> bytes:
    match = ENTRY_VALUE_RE.fullmatch(entry_value)
    if match is None:
        raise RefreshError(f"invalid launcher entry point: {entry_value!r}")
    if any(character in sys.executable for character in "\r\n\x00"):
        raise RefreshError("unsafe target Python executable path")
    python = shlex.quote(sys.executable)
    module = match.group("module")
    attribute = match.group("attribute")
    python_source = f"""import importlib
import sys

sys.argv = sys.argv[1:]
if not sys.argv:
    raise SystemExit('console launcher argv is missing')
target = importlib.import_module({module!r})
for component in {attribute!r}.split('.'):
    target = getattr(target, component)
if not callable(target):
    raise SystemExit('console entry point is not callable: {entry_value}')
raise SystemExit(target())
"""
    compile(python_source, f"<{module}:{attribute}>", "exec")
    quoted_python_source = shlex.quote(python_source)
    content = f'''#!/bin/sh
_splitpoint_pycache=$(/usr/bin/mktemp -d /tmp/onnx-splitpoint-console-pycache.XXXXXXXXXX) || exit 70
_cleanup_splitpoint_pycache() {{
    _splitpoint_rc=$?
    trap - 0
    case "${{_splitpoint_pycache:-}}" in
        /tmp/onnx-splitpoint-console-pycache.??????????)
            if [ -L "$_splitpoint_pycache" ]; then
                /bin/rm -- "$_splitpoint_pycache" || _splitpoint_rc=70
            elif [ -d "$_splitpoint_pycache" ]; then
                /bin/rm -r -- "$_splitpoint_pycache" || _splitpoint_rc=70
            elif [ -e "$_splitpoint_pycache" ]; then
                _splitpoint_rc=70
            fi
            ;;
        *) _splitpoint_rc=70 ;;
    esac
    exit "$_splitpoint_rc"
}}
trap _cleanup_splitpoint_pycache 0
PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX="$_splitpoint_pycache" {python} -c {quoted_python_source} "$0" "$@"
exit $?
'''
    return content.encode("utf-8")


def _write_staged_file(final_path: Path, payload: bytes, token: str) -> Path:
    staged = final_path.with_name(f".{final_path.name}.refresh-new-{token}")
    if staged.exists() or staged.is_symlink():
        raise RefreshError(f"staging path already exists: {staged}")
    with staged.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return staged


def stage_install(
    project: ProjectMetadata,
    site_root: Path,
    scripts_root: Path,
    token: str,
) -> Dict[Path, Path]:
    dist_stem = (
        f"{_wheel_component(project.name)}-"
        f"{_wheel_component(project.version)}.dist-info"
    )
    dist_final = site_root / dist_stem
    dist_stage = dist_final.with_name(f".{dist_final.name}.refresh-new-{token}")
    if dist_stage.exists() or dist_stage.is_symlink():
        raise RefreshError(f"staging path already exists: {dist_stage}")
    dist_stage.mkdir()

    metadata_lines = [
        "Metadata-Version: 2.1",
        f"Name: {project.name}",
        f"Version: {project.version}",
    ]
    if project.description:
        metadata_lines.append(f"Summary: {project.description}")
    if project.requires_python:
        metadata_lines.append(f"Requires-Python: {project.requires_python}")
    metadata_lines.extend(
        f"Requires-Dist: {dependency}" for dependency in project.dependencies
    )
    metadata_payload = ("\n".join(metadata_lines) + "\n\n").encode("utf-8")
    wheel_payload = (
        "Wheel-Version: 1.0\n"
        "Generator: onnx-splitpoint-stdlib-editable\n"
        "Root-Is-Purelib: true\n"
        "Tag: py3-none-any\n"
    ).encode("utf-8")
    entry_points_payload = (
        "[console_scripts]\n"
        + "".join(
            f"{name} = {value}\n"
            for name, value in project.scripts.items()
        )
    ).encode("utf-8")
    direct_url_payload = (
        json.dumps(
            {"dir_info": {"editable": True}, "url": project.root.as_uri()},
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    dist_payloads = {
        "INSTALLER": b"onnx-splitpoint-stdlib-editable\n",
        "METADATA": metadata_payload,
        "REQUESTED": b"",
        "WHEEL": wheel_payload,
        "direct_url.json": direct_url_payload,
        "entry_points.txt": entry_points_payload,
        "top_level.txt": (
            "\n".join(project.top_levels) + "\n"
        ).encode("utf-8"),
    }
    for name, payload in dist_payloads.items():
        (dist_stage / name).write_bytes(payload)

    pth_final = site_root / (
        f"__editable__.{_wheel_component(project.name)}-"
        f"{_wheel_component(project.version)}.pth"
    )
    pth_stage = _write_staged_file(
        pth_final,
        (str(project.root) + "\n").encode("utf-8"),
        token,
    )
    staged: Dict[Path, Path] = {pth_final: pth_stage}
    launcher_stages: Dict[Path, Path] = {}
    for name, value in project.scripts.items():
        launcher_final = scripts_root / name
        launcher_stage = _write_staged_file(
            launcher_final,
            _launcher_content(value),
            token,
        )
        launcher_stage.chmod(0o755)
        launcher_stages[launcher_final] = launcher_stage
    staged.update(launcher_stages)

    record_rows: List[Tuple[str, str, str]] = []
    for name in sorted(dist_payloads):
        staged_path = dist_stage / name
        digest, size = _hash_record(staged_path)
        record_rows.append((f"{dist_stem}/{name}", digest, size))
    digest, size = _hash_record(pth_stage)
    record_rows.append((_record_name(pth_final, site_root), digest, size))
    for final_path, staged_path in sorted(
        launcher_stages.items(), key=lambda item: str(item[0])
    ):
        digest, size = _hash_record(staged_path)
        record_rows.append((_record_name(final_path, site_root), digest, size))
    record_rows.append((f"{dist_stem}/RECORD", "", ""))
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerows(record_rows)
    (dist_stage / "RECORD").write_text(output.getvalue(), encoding="utf-8")
    staged[dist_final] = dist_stage
    return staged


def _remove(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(str(path))


def _required_mapping(values: Sequence[str]) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for value in values:
        name, separator, target = value.partition("=")
        if not separator or not ENTRY_NAME_RE.fullmatch(name):
            raise RefreshError(f"invalid required entry point: {value!r}")
        if ENTRY_VALUE_RE.fullmatch(target) is None:
            raise RefreshError(f"invalid required entry target: {target!r}")
        if name in result:
            raise RefreshError(f"duplicate required entry point: {name}")
        result[name] = target
    return result


VERIFY_CODE = r"""
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import sys
from urllib.parse import unquote, urlparse

spec = json.loads(sys.argv[1])
canonical = lambda value: re.sub(r"[-_.]+", "-", value).lower()
matches = [
    dist for dist in importlib.metadata.distributions()
    if canonical(str(dist.metadata.get("Name") or ""))
    == canonical(spec["name"])
]
if len(matches) != 1:
    raise SystemExit("expected one canonical distribution, found " + repr(
        [(dist.metadata.get("Name"), dist.version) for dist in matches]
    ))
dist = matches[0]
if dist.version != spec["version"]:
    raise SystemExit("distribution version mismatch: " + repr(dist.version))
entries = {
    entry.name: entry
    for entry in dist.entry_points
    if entry.group == "console_scripts"
}
if {name: entry.value for name, entry in entries.items()} != spec["scripts"]:
    raise SystemExit("console entry point metadata mismatch")
for name, expected in spec["required"].items():
    entry = entries.get(name)
    if entry is None or entry.value != expected or not callable(entry.load()):
        raise SystemExit("required entry point is not importable: " + name)
for module in spec["top_levels"]:
    imported = importlib.import_module(module)
    source = Path(imported.__file__).resolve()
    source.relative_to(Path(spec["root"]).resolve())
for name in spec["scripts"]:
    launcher = Path(sys.executable).parent / name
    if not launcher.is_file() or not os.access(str(launcher), os.X_OK):
        raise SystemExit("console launcher missing: " + str(launcher))
direct = json.loads(dist.read_text("direct_url.json") or "")
parsed = urlparse(str(direct["url"]))
editable_root = Path(unquote(parsed.path)).resolve()
if (
    parsed.scheme != "file"
    or direct.get("dir_info", {}).get("editable") is not True
    or editable_root != Path(spec["root"]).resolve()
):
    raise SystemExit("editable direct_url mismatch")
print(json.dumps({"ok": True, "name": spec["name"], "version": dist.version}))
"""


def verify_install(
    project: ProjectMetadata,
    required: Mapping[str, str],
    run_entry_points: Sequence[str],
) -> None:
    for name, value in required.items():
        if project.scripts.get(name) != value:
            raise RefreshError(
                f"required project entry point mismatch: {name}={value}"
            )
    for name in run_entry_points:
        if name not in required:
            raise RefreshError(
                f"launcher execution requires a required entry point: {name}"
            )
    spec = {
        "name": project.name,
        "version": project.version,
        "root": str(project.root),
        "scripts": project.scripts,
        "required": dict(required),
        "top_levels": project.top_levels,
    }
    environment = os.environ.copy()
    environment.pop("PYTHONHOME", None)
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [sys.executable, "-I", "-B", "-c", VERIFY_CODE, json.dumps(spec)],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(Path(sys.prefix).resolve()),
        env=environment,
    )
    if completed.returncode != 0:
        raise RefreshError(
            "fresh-process editable verification failed:\n"
            + completed.stdout
            + completed.stderr
        )
    for name in run_entry_points:
        launcher = Path(sysconfig.get_path("scripts")) / name
        executed = subprocess.run(
            [str(launcher)],
            check=False,
            capture_output=True,
            text=True,
            cwd=str(Path(sys.prefix).resolve()),
            env=environment,
        )
        if executed.returncode != 0:
            raise RefreshError(
                f"console launcher failed ({name}, rc={executed.returncode}):\n"
                + executed.stdout
                + executed.stderr
            )


def refresh(
    project: ProjectMetadata,
    *,
    expected_version: str,
    required: Mapping[str, str],
    run_entry_points: Sequence[str],
) -> Dict[str, object]:
    if project.version != expected_version:
        raise RefreshError(
            f"project version {project.version!r} != {expected_version!r}"
        )
    for name, value in required.items():
        if project.scripts.get(name) != value:
            raise RefreshError(f"required entry point is absent: {name}={value}")

    site_roots, scripts_root, venv_root = _site_and_script_paths()
    site_root = site_roots[0]
    old_artifacts, old_entry_names = discover_old_artifacts(
        project,
        site_roots,
        scripts_root,
        venv_root,
    )
    token = uuid.uuid4().hex
    try:
        staged = stage_install(project, site_root, scripts_root, token)
    except BaseException:
        for staging_root in tuple(site_roots) + (scripts_root,):
            for staged_path in staging_root.glob(
                f".*.refresh-new-{token}"
            ):
                if staged_path.exists() or staged_path.is_symlink():
                    _remove(staged_path)
        raise
    new_paths = set(staged)

    for final_path in new_paths:
        if final_path.exists() or final_path.is_symlink():
            if final_path.resolve() not in {path.resolve() for path in old_artifacts}:
                for staged_path in staged.values():
                    if staged_path.exists():
                        _remove(staged_path)
                raise RefreshError(
                    f"refusing to replace an unowned install artifact: {final_path}"
                )
    for name in project.scripts:
        launcher = scripts_root / name
        if (launcher.exists() or launcher.is_symlink()) and name not in old_entry_names:
            for staged_path in staged.values():
                if staged_path.exists():
                    _remove(staged_path)
            raise RefreshError(f"console launcher is owned elsewhere: {launcher}")

    moved: List[Tuple[Path, Path]] = []
    installed: List[Path] = []
    try:
        for old_path in sorted(old_artifacts, key=lambda item: str(item)):
            if not (old_path.exists() or old_path.is_symlink()):
                continue
            backup = old_path.with_name(
                f".{old_path.name}.refresh-backup-{token}"
            )
            if backup.exists() or backup.is_symlink():
                raise RefreshError(f"backup path already exists: {backup}")
            os.replace(str(old_path), str(backup))
            moved.append((old_path, backup))
        for final_path, staged_path in sorted(
            staged.items(),
            key=lambda item: (item[0].suffix == ".dist-info", str(item[0])),
        ):
            os.replace(str(staged_path), str(final_path))
            installed.append(final_path)
        verify_install(project, required, run_entry_points)
    except BaseException as install_error:
        rollback_errors: List[str] = []
        for final_path in reversed(installed):
            try:
                if final_path.exists() or final_path.is_symlink():
                    _remove(final_path)
            except OSError as exc:
                rollback_errors.append(f"remove {final_path}: {exc}")
        for old_path, backup in reversed(moved):
            try:
                if backup.exists() or backup.is_symlink():
                    if old_path.exists() or old_path.is_symlink():
                        raise RefreshError(
                            "rollback destination reappeared; existing path and "
                            f"backup preserved: {old_path}; backup={backup}"
                        )
                    os.replace(str(backup), str(old_path))
            except (OSError, RefreshError) as exc:
                # A failed metadata restore must not prevent restoration of
                # independent launchers and editable paths. Never discard a
                # backup or overwrite a path recreated by another writer.
                rollback_errors.append(f"restore {old_path}: {exc}")
        for staged_path in staged.values():
            try:
                if staged_path.exists() or staged_path.is_symlink():
                    _remove(staged_path)
            except OSError as exc:
                rollback_errors.append(f"remove stage {staged_path}: {exc}")
        if rollback_errors:
            raise RefreshError(
                "original installation failure: "
                f"{type(install_error).__name__}: {install_error}\n"
                "rollback incomplete; retained conflicting data and backups:\n"
                + "\n".join(rollback_errors)
            ) from install_error
        raise
    else:
        for _old_path, backup in moved:
            if backup.exists() or backup.is_symlink():
                _remove(backup)

    return {
        "ok": True,
        "distribution": project.name,
        "version": project.version,
        "entry_point_count": len(project.scripts),
        "replaced_artifact_count": len(moved),
        "dependencies_preserved": True,
        "installer": "stdlib-only-pep376-editable",
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--expected-version", required=True)
    parser.add_argument("--require-entrypoint", action="append", default=[])
    parser.add_argument("--run-entrypoint", action="append", default=[])
    args = parser.parse_args(argv)
    try:
        project = load_project(args.root)
        result = refresh(
            project,
            expected_version=str(args.expected_version),
            required=_required_mapping(args.require_entrypoint),
            run_entry_points=tuple(args.run_entrypoint),
        )
    except (RefreshError, OSError, UnicodeError) as exc:
        print(f"FEHLER: editable distribution refresh failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
