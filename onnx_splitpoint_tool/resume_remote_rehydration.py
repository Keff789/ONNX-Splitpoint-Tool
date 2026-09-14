from __future__ import annotations

"""Fail-closed remote staging for hash-bound resume artifacts.

The local resolver in :mod:`onnx_splitpoint_tool.resume_artifact_rehydration`
produces an immutable stage map.  This module is the deliberately small
transport boundary for that map:

* inspect the remote destination without modifying it;
* treat already exact bytes as a no-op, without touching the local source;
* otherwise re-verify the local regular file and stream it over SSH stdin;
* verify a same-directory temporary file remotely;
* back up an existing regular destination in an attempt-local directory;
* atomically rename the verified temporary file into place; and
* inspect the final destination again.

No command contract, runner, preflight, collector, or hardware operation is
performed here.  A symlink, non-regular file, path escape, ambiguous SSH target,
or byte-identity mismatch aborts before the destination is replaced.
"""

import hashlib
import json
import math
import os
import re
import secrets
import shlex
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence


REMOTE_PROBE_SCHEMA = "onnx-splitpoint/resume-remote-artifact-probe"
REMOTE_PROBE_SCHEMA_VERSION = 1
REMOTE_REHYDRATION_SCHEMA = (
    "onnx-splitpoint/resume-remote-artifact-rehydration"
)
REMOTE_REHYDRATION_SCHEMA_VERSION = 1
STAGE_MAP_SCHEMA = "onnx-splitpoint/resume-artifact-stage-map"
STAGE_MAP_SCHEMA_VERSION = 1

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_ATTEMPT_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_SSH_TARGET_RE = re.compile(
    r"(?:[A-Za-z0-9_][A-Za-z0-9_.-]{0,63}@)?"
    r"(?:"
    r"[A-Za-z0-9](?:[A-Za-z0-9.-]{0,251}[A-Za-z0-9])?"
    r"|\[[0-9A-Fa-f:.]+\]"
    r")"
)
_RESERVED_REMOTE_PREFIX = ".energy_resume_backups"
_OUTPUT_LIMIT = 4096


class RemoteArtifactRehydrationError(RuntimeError):
    """A stable operational or validation failure.

    ``report`` contains the structured partial report when an operation had
    already begun.  Callers can persist that object in the resume-attempt
    directory while still treating the exception as a hard stop.
    """

    def __init__(
        self,
        code: str,
        *,
        detail: str = "",
        report: Optional[dict[str, Any]] = None,
    ) -> None:
        self.code = str(code)
        self.detail = str(detail)
        self.report = report
        message = self.code
        if self.detail:
            message += f":{self.detail}"
        super().__init__(message)


@dataclass(frozen=True)
class RemoteArtifactRequirement:
    """One expected byte identity at one remote path."""

    remote_path: str
    sha256: str
    size_bytes: Optional[int] = None
    roles: tuple[str, ...] = ()


@dataclass(frozen=True)
class _StageEntry:
    remote_path: str
    sha256: str
    size_bytes: int
    roles: tuple[str, ...]
    source_path: Path
    source_root: Optional[Path]
    source_stat: Optional[dict[str, int]]


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _strict_text(value: Any, *, code: str) -> str:
    text = str(value or "")
    if (
        not text
        or "\x00" in text
        or "\n" in text
        or "\r" in text
        or any(ord(character) < 32 or ord(character) == 127 for character in text)
    ):
        raise RemoteArtifactRehydrationError(code)
    return text


def validate_resume_ssh_target(value: Any) -> str:
    """Return one plain ``[user@]host`` target or fail closed.

    SSH options, ports, whitespace, shell syntax, and host aliases beginning
    with ``-`` are intentionally not accepted as part of this field.
    """

    target = _strict_text(value, code="invalid_ssh_target")
    if len(target) > 320 or _SSH_TARGET_RE.fullmatch(target) is None:
        raise RemoteArtifactRehydrationError("invalid_ssh_target")
    return target


def _normalise_sha256(value: Any) -> str:
    digest = str(value or "").strip().lower()
    if digest.startswith("sha256:"):
        digest = digest[7:]
    if _SHA256_RE.fullmatch(digest) is None:
        raise RemoteArtifactRehydrationError("invalid_expected_sha256")
    return digest


def _normalise_size(
    value: Any,
    *,
    required: bool,
) -> Optional[int]:
    if value in (None, ""):
        if required:
            raise RemoteArtifactRehydrationError("invalid_expected_size")
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RemoteArtifactRehydrationError("invalid_expected_size")
    return int(value)


def _normalise_remote_path(value: Any, *, code: str) -> str:
    raw = _strict_text(value, code=code)
    if not raw.startswith("/") or raw.startswith("//"):
        raise RemoteArtifactRehydrationError(code)
    path = PurePosixPath(raw)
    if (
        not path.is_absolute()
        or path == PurePosixPath("/")
        or "." in path.parts
        or ".." in path.parts
        or str(path) != raw
        or not path.name
    ):
        raise RemoteArtifactRehydrationError(code)
    return str(path)


def _normalise_remote_root(value: Any) -> str:
    root = _normalise_remote_path(value, code="invalid_remote_run_root")
    if len(PurePosixPath(root).parts) < 3:
        raise RemoteArtifactRehydrationError(
            "unsafe_remote_run_root", detail=root,
        )
    return root


def _normalise_attempt_id(value: Any) -> str:
    attempt_id = _strict_text(value, code="invalid_resume_attempt_id")
    if (
        attempt_id in {".", ".."}
        or _ATTEMPT_ID_RE.fullmatch(attempt_id) is None
    ):
        raise RemoteArtifactRehydrationError("invalid_resume_attempt_id")
    return attempt_id


def _require_strict_child(target: str, root: str) -> PurePosixPath:
    try:
        relative = PurePosixPath(target).relative_to(PurePosixPath(root))
    except ValueError as exc:
        raise RemoteArtifactRehydrationError(
            "remote_target_outside_frozen_root", detail=target,
        ) from exc
    if (
        not relative.parts
        or relative.parts[0] == _RESERVED_REMOTE_PREFIX
    ):
        raise RemoteArtifactRehydrationError(
            "remote_target_outside_frozen_root", detail=target,
        )
    return relative


def _normalise_roles(value: Any) -> tuple[str, ...]:
    if value in (None, ""):
        return ()
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise RemoteArtifactRehydrationError("invalid_artifact_roles")
    roles: list[str] = []
    for raw in value:
        role = _strict_text(raw, code="invalid_artifact_role").strip()
        if not role or len(role) > 512:
            raise RemoteArtifactRehydrationError("invalid_artifact_role")
        roles.append(role)
    return tuple(sorted(set(roles)))


def _normalise_requirement(
    value: Any,
    *,
    remote_root: str,
    require_size: bool = False,
) -> RemoteArtifactRequirement:
    if isinstance(value, RemoteArtifactRequirement):
        raw_path = value.remote_path
        raw_sha = value.sha256
        raw_size = value.size_bytes
        raw_roles: Any = value.roles
    elif isinstance(value, Mapping):
        raw_path = value.get("remote_path", value.get("path"))
        raw_sha = value.get("sha256")
        raw_size = value.get("size_bytes")
        if raw_size in (None, ""):
            raw_size = value.get("bytes")
        if raw_size in (None, ""):
            raw_size = value.get("size")
        raw_roles = value.get("roles")
        if raw_roles in (None, "") and value.get("role") not in (None, ""):
            raw_roles = [value.get("role")]
    elif (
        hasattr(value, "remote_path")
        and hasattr(value, "sha256")
    ):
        raw_path = getattr(value, "remote_path")
        raw_sha = getattr(value, "sha256")
        raw_size = getattr(
            value,
            "size_bytes",
            getattr(value, "bytes", getattr(value, "size", None)),
        )
        raw_roles = getattr(value, "roles", None)
        if raw_roles in (None, "") and getattr(value, "role", None) not in (
            None, "",
        ):
            raw_roles = [getattr(value, "role")]
    else:
        raise RemoteArtifactRehydrationError(
            "invalid_remote_artifact_requirement",
        )
    path = _normalise_remote_path(
        raw_path, code="invalid_remote_artifact_path",
    )
    _require_strict_child(path, remote_root)
    return RemoteArtifactRequirement(
        remote_path=path,
        sha256=_normalise_sha256(raw_sha),
        size_bytes=_normalise_size(raw_size, required=require_size),
        roles=_normalise_roles(raw_roles),
    )


def _normalise_requirements(
    values: Iterable[RemoteArtifactRequirement | Mapping[str, Any]],
    *,
    remote_root: str,
    require_size: bool = False,
) -> list[RemoteArtifactRequirement]:
    by_path: dict[str, RemoteArtifactRequirement] = {}
    for raw in values:
        item = _normalise_requirement(
            raw,
            remote_root=remote_root,
            require_size=require_size,
        )
        existing = by_path.get(item.remote_path)
        if existing is None:
            by_path[item.remote_path] = item
            continue
        if existing.sha256 != item.sha256 or (
            existing.size_bytes is not None
            and item.size_bytes is not None
            and existing.size_bytes != item.size_bytes
        ):
            raise RemoteArtifactRehydrationError(
                "conflicting_remote_artifact_requirement",
                detail=item.remote_path,
            )
        by_path[item.remote_path] = RemoteArtifactRequirement(
            remote_path=item.remote_path,
            sha256=item.sha256,
            size_bytes=(
                existing.size_bytes
                if existing.size_bytes is not None
                else item.size_bytes
            ),
            roles=tuple(sorted(set(existing.roles).union(item.roles))),
        )
    if not by_path:
        raise RemoteArtifactRehydrationError(
            "remote_artifact_requirements_empty",
        )
    return [by_path[path] for path in sorted(by_path)]


_PROBE_SCRIPT = r"""
# ONNX_SPLITPOINT_RESUME_REMOTE_PROBE_V1
set -euo pipefail
root=$1
target=$2

storage_root=$(dirname -- "$root")
if [ -L "$storage_root" ] || [ ! -d "$storage_root" ]; then
  printf 'OSPRPROBE1\tunsafe_storage_root\t-\t-\t-\t-\t-\t-\n'
  exit 0
fi
storage_root_real=$(realpath -e -- "$storage_root")
if [ "$storage_root_real" != "$storage_root" ]; then
  printf 'OSPRPROBE1\tunsafe_storage_root\t-\t-\t-\t-\t-\t-\n'
  exit 0
fi
storage_device=$(stat -c '%d' -- "$storage_root")
storage_inode=$(stat -c '%i' -- "$storage_root")
case "$root" in
  "$storage_root"/*) ;;
  *)
    printf 'OSPRPROBE1\tunsafe_root\t-\t-\t-\t-\t-\t-\n'
    exit 0
    ;;
esac
root_leaf=${root#"$storage_root"/}
case "$root_leaf" in
  ''|.|..|*/*)
    printf 'OSPRPROBE1\tunsafe_root\t-\t-\t-\t-\t-\t-\n'
    exit 0
    ;;
esac

if [ -L "$root" ] || { [ -e "$root" ] && [ ! -d "$root" ]; }; then
  printf 'OSPRPROBE1\tunsafe_root\t-\t-\t-\t-\t-\t-\n'
  exit 0
fi
if [ ! -e "$root" ]; then
  printf 'OSPRPROBE1\troot_missing\t-\t-\t%s\t%s\t-\t-\n' \
    "$storage_device" "$storage_inode"
  exit 0
fi
root_real=$(realpath -e -- "$root")
if [ "$root_real" != "$root" ]; then
  printf 'OSPRPROBE1\tunsafe_root\t-\t-\t-\t-\t-\t-\n'
  exit 0
fi
root_device=$(stat -c '%d' -- "$root")
root_inode=$(stat -c '%i' -- "$root")
emit_existing() {
  local status=$1
  local value1=$2
  local value2=$3
  local final_storage_real final_storage_device final_storage_inode
  local final_root_real final_root_device final_root_inode
  if [ -L "$storage_root" ] || [ ! -d "$storage_root" ]; then
    printf 'OSPRPROBE1\tunsafe_storage_root\t-\t-\t-\t-\t-\t-\n'
    exit 0
  fi
  final_storage_real=$(realpath -e -- "$storage_root")
  final_storage_device=$(stat -c '%d' -- "$storage_root")
  final_storage_inode=$(stat -c '%i' -- "$storage_root")
  if [ "$final_storage_real" != "$storage_root" ] || \
     [ "$final_storage_device" != "$storage_device" ] || \
     [ "$final_storage_inode" != "$storage_inode" ]; then
    printf 'OSPRPROBE1\tstorage_root_identity_drift\t-\t-\t-\t-\t-\t-\n'
    exit 0
  fi
  if [ -L "$root" ] || [ ! -d "$root" ]; then
    printf 'OSPRPROBE1\tunsafe_root\t-\t-\t-\t-\t-\t-\n'
    exit 0
  fi
  final_root_real=$(realpath -e -- "$root")
  final_root_device=$(stat -c '%d' -- "$root")
  final_root_inode=$(stat -c '%i' -- "$root")
  if [ "$final_root_real" != "$root" ] || \
     [ "$final_root_device" != "$root_device" ] || \
     [ "$final_root_inode" != "$root_inode" ]; then
    printf 'OSPRPROBE1\troot_identity_drift\t-\t-\t%s\t%s\t-\t-\n' \
      "$storage_device" "$storage_inode"
    exit 0
  fi
  printf 'OSPRPROBE1\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$status" "$value1" "$value2" \
    "$storage_device" "$storage_inode" "$root_device" "$root_inode"
}
parent=$(dirname -- "$target")
case "$parent" in
  "$root") relative= ;;
  "$root"/*) relative=${parent#"$root"/} ;;
  *)
    printf 'OSPRPROBE1\tpath_escape\t-\t-\t-\t-\t-\t-\n'
    exit 0
    ;;
esac
current=$root
while [ -n "$relative" ]; do
  component=${relative%%/*}
  if [ "$relative" = "$component" ]; then
    relative=
  else
    relative=${relative#*/}
  fi
  case "$component" in
    ''|.|..)
      printf 'OSPRPROBE1\tpath_escape\t-\t-\t-\t-\t-\t-\n'
      exit 0
      ;;
  esac
  current="$current/$component"
  if [ -L "$current" ]; then
    printf 'OSPRPROBE1\tunsafe_parent\t-\t-\t%s\t%s\t%s\t%s\n' \
      "$storage_device" "$storage_inode" "$root_device" "$root_inode"
    exit 0
  fi
  if [ -e "$current" ] && [ ! -d "$current" ]; then
    printf 'OSPRPROBE1\tunsafe_parent\t-\t-\t%s\t%s\t%s\t%s\n' \
      "$storage_device" "$storage_inode" "$root_device" "$root_inode"
    exit 0
  fi
  if [ ! -e "$current" ]; then
    emit_existing parent_missing - -
    exit 0
  fi
done
parent_real=$(realpath -e -- "$parent")
case "$parent_real" in
  "$root_real"|"$root_real"/*) ;;
  *)
    printf 'OSPRPROBE1\tpath_escape\t-\t-\t%s\t%s\t%s\t%s\n' \
      "$storage_device" "$storage_inode" "$root_device" "$root_inode"
    exit 0
    ;;
esac

if [ -L "$target" ]; then
  emit_existing symlink - -
elif [ -e "$target" ]; then
  if [ ! -f "$target" ]; then
    emit_existing non_regular - -
  else
    digest=$(sha256sum -- "$target" | awk '{print $1}')
    size=$(stat -c '%s' -- "$target")
    emit_existing regular "$digest" "$size"
  fi
else
  emit_existing missing - -
fi
""".strip()


_TREE_SCRIPT = r"""
# ONNX_SPLITPOINT_RESUME_REMOTE_TREE_V1
set -euo pipefail
umask 077
root=$1
expected_root_state=$2
expected_storage_device=$3
expected_storage_inode=$4
expected_root_device=$5
expected_root_inode=$6
root_creation_token=$7
shift 7

created=()
rollback_created() {
  local index
  for ((index=${#created[@]}-1; index>=0; index--)); do
    rmdir -- "${created[$index]}" 2>/dev/null || true
  done
}
fail_tree() {
  local status=$1
  case "$status" in
    storage_root_identity_drift|root_identity_drift)
      ;;
    *)
      rollback_created
      ;;
  esac
  printf 'OSPRTREE1\t%s\tfalse\t0\t-\t-\t-\t-\n' "$status"
  exit 0
}

assert_storage_identity() {
  if [ -L "$storage_root" ] || [ ! -d "$storage_root" ]; then
    fail_tree unsafe_storage_root
  fi
  local observed_device observed_inode observed_real
  observed_real=$(realpath -e -- "$storage_root")
  observed_device=$(stat -c '%d' -- "$storage_root")
  observed_inode=$(stat -c '%i' -- "$storage_root")
  if [ "$observed_real" != "$storage_root" ]; then
    fail_tree unsafe_storage_root
  fi
  if [ "$observed_device" != "$expected_storage_device" ] || \
     [ "$observed_inode" != "$expected_storage_inode" ]; then
    fail_tree storage_root_identity_drift
  fi
}

assert_root_identity() {
  assert_storage_identity
  if [ -L "$root" ] || [ ! -d "$root" ]; then
    fail_tree unsafe_root
  fi
  local observed_device observed_inode observed_real
  observed_real=$(realpath -e -- "$root")
  observed_device=$(stat -c '%d' -- "$root")
  observed_inode=$(stat -c '%i' -- "$root")
  if [ "$observed_real" != "$root" ]; then
    fail_tree unsafe_root
  fi
  if [ "$observed_device" != "$pinned_root_device" ] || \
     [ "$observed_inode" != "$pinned_root_inode" ]; then
    fail_tree root_identity_drift
  fi
}

storage_root=$(dirname -- "$root")
if [ -L "$storage_root" ] || [ ! -d "$storage_root" ]; then
  fail_tree unsafe_storage_root
fi
storage_root_real=$(realpath -e -- "$storage_root")
if [ "$storage_root_real" != "$storage_root" ]; then
  fail_tree unsafe_storage_root
fi
observed_storage_device=$(stat -c '%d' -- "$storage_root")
observed_storage_inode=$(stat -c '%i' -- "$storage_root")
if [ "$observed_storage_device" != "$expected_storage_device" ] || \
   [ "$observed_storage_inode" != "$expected_storage_inode" ]; then
  fail_tree storage_root_identity_drift
fi
case "$root" in
  "$storage_root"/*) ;;
  *) fail_tree unsafe_root ;;
esac
root_leaf=${root#"$storage_root"/}
case "$root_leaf" in
  ''|.|..|*/*) fail_tree unsafe_root ;;
esac

case "$expected_root_state" in
  missing)
    if [ "$expected_root_device" != "-" ] || \
       [ "$expected_root_inode" != "-" ]; then
      fail_tree invalid_expected_root_identity
    fi
    if [ -L "$root" ] || [ -e "$root" ]; then
      fail_tree root_state_changed
    fi
    case "$root_creation_token" in
      ''|*[!0-9a-f]*) fail_tree invalid_creation_token ;;
    esac
    if [ "${#root_creation_token}" -ne 32 ]; then
      fail_tree invalid_creation_token
    fi
    staging_root="$storage_root/.${root_leaf}.resume-root-${root_creation_token}.tmp"
    if [ -L "$staging_root" ] || [ -e "$staging_root" ]; then
      fail_tree root_state_changed
    fi
    if ! mkdir -m 700 -- "$staging_root"; then
      fail_tree root_state_changed
    fi
    created+=("$staging_root")
    if [ -L "$staging_root" ] || [ ! -d "$staging_root" ]; then
      fail_tree root_state_changed
    fi
    staging_real=$(realpath -e -- "$staging_root")
    if [ "$staging_real" != "$staging_root" ]; then
      fail_tree root_state_changed
    fi
    pinned_root_device=$(stat -c '%d' -- "$staging_root")
    pinned_root_inode=$(stat -c '%i' -- "$staging_root")
    assert_storage_identity
    if [ -L "$root" ] || [ -e "$root" ]; then
      fail_tree root_state_changed
    fi
    mv -T -n -- "$staging_root" "$root"
    if [ -e "$staging_root" ] || [ -L "$staging_root" ]; then
      fail_tree root_state_changed
    fi
    created[0]="$root"
    root_created=true
    assert_root_identity
    ;;
  existing)
    if [ -L "$root" ] || [ ! -d "$root" ]; then
      fail_tree unsafe_root
    fi
    observed_root_device=$(stat -c '%d' -- "$root")
    observed_root_inode=$(stat -c '%i' -- "$root")
    if [ "$observed_root_device" != "$expected_root_device" ] || \
       [ "$observed_root_inode" != "$expected_root_inode" ]; then
      fail_tree root_identity_drift
    fi
    pinned_root_device=$expected_root_device
    pinned_root_inode=$expected_root_inode
    root_created=false
    assert_root_identity
    ;;
  *) fail_tree invalid_expected_root_state ;;
esac
root_real=$root

for required_parent in "$@"; do
  assert_root_identity
  case "$required_parent" in
    "$root") relative= ;;
    "$root"/*) relative=${required_parent#"$root"/} ;;
    *) fail_tree path_escape ;;
  esac
  current=$root
  while [ -n "$relative" ]; do
    component=${relative%%/*}
    if [ "$relative" = "$component" ]; then
      relative=
    else
      relative=${relative#*/}
    fi
    case "$component" in
      ''|.|..) fail_tree invalid_component ;;
    esac
    current="$current/$component"
    if [ -L "$current" ]; then
      fail_tree unsafe_parent
    fi
    if [ -e "$current" ]; then
      if [ ! -d "$current" ]; then
        fail_tree unsafe_parent
      fi
    else
      assert_root_identity
      if ! mkdir -m 700 -- "$current"; then
        fail_tree parent_state_changed
      fi
      created+=("$current")
      assert_root_identity
    fi
    if [ -L "$current" ] || [ ! -d "$current" ]; then
      fail_tree unsafe_parent
    fi
    current_real=$(realpath -e -- "$current")
    case "$current_real" in
      "$root_real"|"$root_real"/*) ;;
      *) fail_tree path_escape ;;
    esac
  done
done

if [ "$expected_root_state" = "existing" ] && [ "${#created[@]}" -eq 0 ]; then
  fail_tree parent_state_changed
fi
assert_root_identity
printf 'OSPRTREE1\tcreated\t%s\t%s\t%s\t%s\t%s\t%s\n' \
  "$root_created" "${#created[@]}" \
  "$observed_storage_device" "$observed_storage_inode" \
  "$pinned_root_device" "$pinned_root_inode"
""".strip()


_UPLOAD_SCRIPT = r"""
# ONNX_SPLITPOINT_RESUME_REMOTE_UPLOAD_V1
set -euo pipefail
umask 077
root=$1
target=$2
temporary=$3
expected_sha=$4
expected_size=$5
expected_storage_device=$6
expected_storage_inode=$7
expected_root_device=$8
expected_root_inode=$9

storage_root=$(dirname -- "$root")
if [ -L "$storage_root" ] || [ ! -d "$storage_root" ]; then
  printf 'OSPRUPLOAD1\tunsafe_storage_root\t-\t-\n'
  exit 0
fi
storage_root_real=$(realpath -e -- "$storage_root")
if [ "$storage_root_real" != "$storage_root" ]; then
  printf 'OSPRUPLOAD1\tunsafe_storage_root\t-\t-\n'
  exit 0
fi
observed_storage_device=$(stat -c '%d' -- "$storage_root")
observed_storage_inode=$(stat -c '%i' -- "$storage_root")
if [ "$observed_storage_device" != "$expected_storage_device" ] || \
   [ "$observed_storage_inode" != "$expected_storage_inode" ]; then
  printf 'OSPRUPLOAD1\tstorage_root_identity_drift\t-\t-\n'
  exit 0
fi
if [ ! -d "$root" ] || [ -L "$root" ]; then
  printf 'OSPRUPLOAD1\tunsafe_root\t-\t-\n'
  exit 0
fi
root_real=$(realpath -e -- "$root")
if [ "$root_real" != "$root" ]; then
  printf 'OSPRUPLOAD1\tunsafe_root\t-\t-\n'
  exit 0
fi
observed_root_device=$(stat -c '%d' -- "$root")
observed_root_inode=$(stat -c '%i' -- "$root")
if [ "$observed_root_device" != "$expected_root_device" ] || \
   [ "$observed_root_inode" != "$expected_root_inode" ]; then
  printf 'OSPRUPLOAD1\troot_identity_drift\t-\t-\n'
  exit 0
fi
parent=$(dirname -- "$target")
temporary_parent=$(dirname -- "$temporary")
if [ "$temporary_parent" != "$parent" ]; then
  printf 'OSPRUPLOAD1\tunsafe_parent\t-\t-\n'
  exit 0
fi
case "$parent" in
  "$root") relative= ;;
  "$root"/*) relative=${parent#"$root"/} ;;
  *)
    printf 'OSPRUPLOAD1\tpath_escape\t-\t-\n'
    exit 0
    ;;
esac
current=$root
while [ -n "$relative" ]; do
  component=${relative%%/*}
  if [ "$relative" = "$component" ]; then
    relative=
  else
    relative=${relative#*/}
  fi
  case "$component" in
    ''|.|..)
      printf 'OSPRUPLOAD1\tpath_escape\t-\t-\n'
      exit 0
      ;;
  esac
  current="$current/$component"
  if [ -L "$current" ] || [ ! -d "$current" ]; then
    printf 'OSPRUPLOAD1\tunsafe_parent\t-\t-\n'
    exit 0
  fi
done
parent_real=$(realpath -e -- "$parent")
case "$parent_real" in
  "$root_real"|"$root_real"/*) ;;
  *)
    printf 'OSPRUPLOAD1\tpath_escape\t-\t-\n'
    exit 0
    ;;
esac
if [ -e "$temporary" ] || [ -L "$temporary" ]; then
  printf 'OSPRUPLOAD1\ttemporary_exists\t-\t-\n'
  exit 0
fi

cleanup_temporary() {
  rm -f -- "$temporary"
}
trap cleanup_temporary EXIT
set -C
cat > "$temporary"
set +C
if [ -L "$temporary" ] || [ ! -f "$temporary" ]; then
  printf 'OSPRUPLOAD1\ttemporary_non_regular\t-\t-\n'
  exit 0
fi
digest=$(sha256sum -- "$temporary" | awk '{print $1}')
size=$(stat -c '%s' -- "$temporary")
if [ "$digest" != "$expected_sha" ] || [ "$size" != "$expected_size" ]; then
  printf 'OSPRUPLOAD1\ttemporary_identity_mismatch\t%s\t%s\n' "$digest" "$size"
  exit 0
fi
final_storage_real=$(realpath -e -- "$storage_root")
final_storage_device=$(stat -c '%d' -- "$storage_root")
final_storage_inode=$(stat -c '%i' -- "$storage_root")
if [ "$final_storage_real" != "$storage_root" ] || \
   [ "$final_storage_device" != "$expected_storage_device" ] || \
   [ "$final_storage_inode" != "$expected_storage_inode" ]; then
  printf 'OSPRUPLOAD1\tstorage_root_identity_drift\t-\t-\n'
  exit 0
fi
if [ -L "$root" ] || [ ! -d "$root" ]; then
  printf 'OSPRUPLOAD1\tunsafe_root\t-\t-\n'
  exit 0
fi
final_root_real=$(realpath -e -- "$root")
final_root_device=$(stat -c '%d' -- "$root")
final_root_inode=$(stat -c '%i' -- "$root")
if [ "$final_root_real" != "$root" ] || \
   [ "$final_root_device" != "$expected_root_device" ] || \
   [ "$final_root_inode" != "$expected_root_inode" ]; then
  printf 'OSPRUPLOAD1\troot_identity_drift\t-\t-\n'
  exit 0
fi
trap - EXIT
printf 'OSPRUPLOAD1\tstaged\t%s\t%s\n' "$digest" "$size"
""".strip()


_COMMIT_SCRIPT = r"""
# ONNX_SPLITPOINT_RESUME_REMOTE_COMMIT_V1
set -euo pipefail
umask 077
root=$1
target=$2
temporary=$3
expected_sha=$4
expected_size=$5
attempt_id=$6
backup_path=$7
expected_storage_device=$8
expected_storage_inode=$9
expected_root_device=${10}
expected_root_inode=${11}

storage_root=$(dirname -- "$root")
if [ -L "$storage_root" ] || [ ! -d "$storage_root" ]; then
  printf 'OSPRCOMMIT1\tunsafe_storage_root\t-\t-\t-\t-\t-\n'
  exit 0
fi
storage_root_real=$(realpath -e -- "$storage_root")
if [ "$storage_root_real" != "$storage_root" ]; then
  printf 'OSPRCOMMIT1\tunsafe_storage_root\t-\t-\t-\t-\t-\n'
  exit 0
fi
observed_storage_device=$(stat -c '%d' -- "$storage_root")
observed_storage_inode=$(stat -c '%i' -- "$storage_root")
if [ "$observed_storage_device" != "$expected_storage_device" ] || \
   [ "$observed_storage_inode" != "$expected_storage_inode" ]; then
  printf 'OSPRCOMMIT1\tstorage_root_identity_drift\t-\t-\t-\t-\t-\n'
  exit 0
fi
if [ ! -d "$root" ] || [ -L "$root" ]; then
  printf 'OSPRCOMMIT1\tunsafe_root\t-\t-\t-\t-\t-\n'
  exit 0
fi
root_real=$(realpath -e -- "$root")
if [ "$root_real" != "$root" ]; then
  printf 'OSPRCOMMIT1\tunsafe_root\t-\t-\t-\t-\t-\n'
  exit 0
fi
observed_root_device=$(stat -c '%d' -- "$root")
observed_root_inode=$(stat -c '%i' -- "$root")
if [ "$observed_root_device" != "$expected_root_device" ] || \
   [ "$observed_root_inode" != "$expected_root_inode" ]; then
  printf 'OSPRCOMMIT1\troot_identity_drift\t-\t-\t-\t-\t-\n'
  exit 0
fi
assert_commit_identity() {
  local final_storage_real final_storage_device final_storage_inode
  local final_root_real final_root_device final_root_inode
  if [ -L "$storage_root" ] || [ ! -d "$storage_root" ]; then
    printf 'OSPRCOMMIT1\tunsafe_storage_root\t-\t-\t-\t-\t-\n'
    exit 0
  fi
  final_storage_real=$(realpath -e -- "$storage_root")
  final_storage_device=$(stat -c '%d' -- "$storage_root")
  final_storage_inode=$(stat -c '%i' -- "$storage_root")
  if [ "$final_storage_real" != "$storage_root" ] || \
     [ "$final_storage_device" != "$expected_storage_device" ] || \
     [ "$final_storage_inode" != "$expected_storage_inode" ]; then
    printf 'OSPRCOMMIT1\tstorage_root_identity_drift\t-\t-\t-\t-\t-\n'
    exit 0
  fi
  if [ -L "$root" ] || [ ! -d "$root" ]; then
    printf 'OSPRCOMMIT1\tunsafe_root\t-\t-\t-\t-\t-\n'
    exit 0
  fi
  final_root_real=$(realpath -e -- "$root")
  final_root_device=$(stat -c '%d' -- "$root")
  final_root_inode=$(stat -c '%i' -- "$root")
  if [ "$final_root_real" != "$root" ] || \
     [ "$final_root_device" != "$expected_root_device" ] || \
     [ "$final_root_inode" != "$expected_root_inode" ]; then
    printf 'OSPRCOMMIT1\troot_identity_drift\t-\t-\t-\t-\t-\n'
    exit 0
  fi
}
parent=$(dirname -- "$target")
if [ "$(dirname -- "$temporary")" != "$parent" ]; then
  printf 'OSPRCOMMIT1\tunsafe_parent\t-\t-\t-\t-\t-\n'
  exit 0
fi
case "$parent" in
  "$root") relative= ;;
  "$root"/*) relative=${parent#"$root"/} ;;
  *)
    printf 'OSPRCOMMIT1\tpath_escape\t-\t-\t-\t-\t-\n'
    exit 0
    ;;
esac
current=$root
while [ -n "$relative" ]; do
  component=${relative%%/*}
  if [ "$relative" = "$component" ]; then
    relative=
  else
    relative=${relative#*/}
  fi
  case "$component" in
    ''|.|..)
      printf 'OSPRCOMMIT1\tpath_escape\t-\t-\t-\t-\t-\n'
      exit 0
      ;;
  esac
  current="$current/$component"
  if [ -L "$current" ] || [ ! -d "$current" ]; then
    printf 'OSPRCOMMIT1\tunsafe_parent\t-\t-\t-\t-\t-\n'
    exit 0
  fi
done
parent_real=$(realpath -e -- "$parent")
case "$parent_real" in
  "$root_real"|"$root_real"/*) ;;
  *)
    printf 'OSPRCOMMIT1\tpath_escape\t-\t-\t-\t-\t-\n'
    exit 0
    ;;
esac

if [ -L "$temporary" ] || [ ! -f "$temporary" ]; then
  printf 'OSPRCOMMIT1\ttemporary_non_regular\t-\t-\t-\t-\t-\n'
  exit 0
fi
temporary_sha=$(sha256sum -- "$temporary" | awk '{print $1}')
temporary_size=$(stat -c '%s' -- "$temporary")
if [ "$temporary_sha" != "$expected_sha" ] || [ "$temporary_size" != "$expected_size" ]; then
  printf 'OSPRCOMMIT1\ttemporary_identity_mismatch\t%s\t%s\t-\t-\t-\n' "$temporary_sha" "$temporary_size"
  exit 0
fi

if [ -L "$target" ]; then
  printf 'OSPRCOMMIT1\tsymlink\t-\t-\t-\t-\t-\n'
  exit 0
fi
old_sha=-
old_size=-
if [ -e "$target" ]; then
  if [ ! -f "$target" ]; then
    printf 'OSPRCOMMIT1\tnon_regular\t-\t-\t-\t-\t-\n'
    exit 0
  fi
  old_sha=$(sha256sum -- "$target" | awk '{print $1}')
  old_size=$(stat -c '%s' -- "$target")
  if [ "$old_sha" = "$expected_sha" ] && [ "$old_size" = "$expected_size" ]; then
    assert_commit_identity
    rm -f -- "$temporary"
    assert_commit_identity
    printf 'OSPRCOMMIT1\tnoop_race\t%s\t%s\t-\t%s\t%s\n' "$old_sha" "$old_size" "$old_sha" "$old_size"
    exit 0
  fi

  backup_root="$root/.energy_resume_backups"
  attempt_root="$backup_root/$attempt_id"
  if [ -L "$backup_root" ] || { [ -e "$backup_root" ] && [ ! -d "$backup_root" ]; }; then
    printf 'OSPRCOMMIT1\tunsafe_backup_root\t-\t-\t-\t-\t-\n'
    exit 0
  fi
  if [ ! -e "$backup_root" ]; then
    mkdir -- "$backup_root"
  fi
  if [ -L "$attempt_root" ] || { [ -e "$attempt_root" ] && [ ! -d "$attempt_root" ]; }; then
    printf 'OSPRCOMMIT1\tunsafe_attempt_root\t-\t-\t-\t-\t-\n'
    exit 0
  fi
  if [ ! -e "$attempt_root" ]; then
    mkdir -- "$attempt_root"
  fi
  attempt_real=$(realpath -e -- "$attempt_root")
  case "$attempt_real" in
    "$root_real"/.energy_resume_backups/"$attempt_id") ;;
    *)
      printf 'OSPRCOMMIT1\tbackup_path_escape\t-\t-\t-\t-\t-\n'
      exit 0
      ;;
  esac
  case "$backup_path" in
    "$attempt_root"/*) ;;
    *)
      printf 'OSPRCOMMIT1\tbackup_path_escape\t-\t-\t-\t-\t-\n'
      exit 0
      ;;
  esac
  if [ -e "$backup_path" ] || [ -L "$backup_path" ]; then
    printf 'OSPRCOMMIT1\tbackup_exists\t-\t-\t-\t-\t-\n'
    exit 0
  fi
  cp -p -- "$target" "$backup_path"
  if [ -L "$backup_path" ] || [ ! -f "$backup_path" ]; then
    printf 'OSPRCOMMIT1\tbackup_non_regular\t-\t-\t-\t-\t-\n'
    exit 0
  fi
  backup_sha=$(sha256sum -- "$backup_path" | awk '{print $1}')
  backup_size=$(stat -c '%s' -- "$backup_path")
  if [ "$backup_sha" != "$old_sha" ] || [ "$backup_size" != "$old_size" ]; then
    printf 'OSPRCOMMIT1\tbackup_identity_mismatch\t-\t-\t-\t%s\t%s\n' "$backup_sha" "$backup_size"
    exit 0
  fi
else
  backup_path=-
fi

assert_commit_identity
mv -T -- "$temporary" "$target"
assert_commit_identity
new_sha=$(sha256sum -- "$target" | awk '{print $1}')
new_size=$(stat -c '%s' -- "$target")
if [ "$new_sha" != "$expected_sha" ] || [ "$new_size" != "$expected_size" ]; then
  printf 'OSPRCOMMIT1\tfinal_identity_mismatch\t%s\t%s\t%s\t%s\t%s\n' "$new_sha" "$new_size" "$backup_path" "$old_sha" "$old_size"
  exit 0
fi
assert_commit_identity
printf 'OSPRCOMMIT1\tcommitted\t%s\t%s\t%s\t%s\t%s\n' "$new_sha" "$new_size" "$backup_path" "$old_sha" "$old_size"
""".strip()


def _decode_output(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _normalise_timeout(value: Any) -> float:
    try:
        timeout = float(value)
    except (TypeError, ValueError) as exc:
        raise RemoteArtifactRehydrationError("invalid_ssh_timeout") from exc
    if not math.isfinite(timeout) or timeout <= 0 or timeout > 86400:
        raise RemoteArtifactRehydrationError("invalid_ssh_timeout")
    return timeout


def _trim_output(value: Any) -> str:
    return _decode_output(value).strip()[:_OUTPUT_LIMIT]


def _remote_command(script: str, arguments: Sequence[str]) -> str:
    return (
        "bash -c "
        + shlex.quote(script)
        + " -- "
        + " ".join(shlex.quote(argument) for argument in arguments)
    )


def _invoke_ssh(
    *,
    ssh_target: str,
    remote_command: str,
    timeout_s: float,
    runner: Callable[..., Any],
    stdin: Any = None,
) -> Any:
    command = [
        "ssh",
        "-T",
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
        ssh_target,
        remote_command,
    ]
    try:
        completed = runner(
            command,
            stdin=stdin,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        raise RemoteArtifactRehydrationError(
            "ssh_timeout", detail=ssh_target,
        ) from exc
    except OSError as exc:
        raise RemoteArtifactRehydrationError(
            "ssh_invocation_failed",
            detail=f"{ssh_target}:{type(exc).__name__}",
        ) from exc
    if int(getattr(completed, "returncode", 1)) != 0:
        raise RemoteArtifactRehydrationError(
            "ssh_remote_command_failed",
            detail=(
                f"target={ssh_target};"
                f"rc={getattr(completed, 'returncode', None)};"
                f"stderr={_trim_output(getattr(completed, 'stderr', ''))}"
            ),
        )
    return completed


def _parse_protocol(
    completed: Any,
    *,
    marker: str,
    fields: int,
) -> list[str]:
    output = _decode_output(getattr(completed, "stdout", "")).strip()
    lines = output.splitlines()
    if len(lines) != 1:
        raise RemoteArtifactRehydrationError(
            "remote_protocol_invalid",
            detail=f"marker={marker};stdout={output[:_OUTPUT_LIMIT]}",
        )
    parts = lines[0].split("\t")
    if len(parts) != fields or parts[0] != marker:
        raise RemoteArtifactRehydrationError(
            "remote_protocol_invalid",
            detail=f"marker={marker};stdout={output[:_OUTPUT_LIMIT]}",
        )
    return parts


def _normalise_remote_observation(
    parts: Sequence[str],
    requirement: RemoteArtifactRequirement,
) -> dict[str, Any]:
    raw_status = parts[1]
    root_missing = raw_status == "root_missing"
    parent_missing = raw_status == "parent_missing"
    if raw_status in {
        "symlink",
        "non_regular",
        "unsafe_root",
        "unsafe_storage_root",
        "unsafe_parent",
        "path_escape",
        "storage_root_identity_drift",
        "root_identity_drift",
    }:
        raise RemoteArtifactRehydrationError(
            f"remote_probe_{raw_status}",
            detail=requirement.remote_path,
        )
    if raw_status not in {
        "missing", "root_missing", "parent_missing", "regular",
    }:
        raise RemoteArtifactRehydrationError(
            "remote_protocol_invalid",
            detail=f"unexpected_probe_status={raw_status}",
        )

    try:
        storage_device = int(parts[4])
        storage_inode = int(parts[5])
    except (TypeError, ValueError) as exc:
        raise RemoteArtifactRehydrationError(
            "remote_protocol_invalid",
            detail="remote_storage_root_identity_invalid",
        ) from exc
    if storage_device < 0 or storage_inode <= 0:
        raise RemoteArtifactRehydrationError(
            "remote_protocol_invalid",
            detail="remote_storage_root_identity_invalid",
        )
    if root_missing:
        if parts[6] != "-" or parts[7] != "-":
            raise RemoteArtifactRehydrationError(
                "remote_protocol_invalid",
                detail="remote_missing_run_root_identity_invalid",
            )
        root_device: Optional[int] = None
        root_inode: Optional[int] = None
    else:
        try:
            root_device = int(parts[6])
            root_inode = int(parts[7])
        except (TypeError, ValueError) as exc:
            raise RemoteArtifactRehydrationError(
                "remote_protocol_invalid",
                detail="remote_run_root_identity_invalid",
            ) from exc
        if root_device < 0 or root_inode <= 0:
            raise RemoteArtifactRehydrationError(
                "remote_protocol_invalid",
                detail="remote_run_root_identity_invalid",
            )

    if raw_status in {"missing", "root_missing", "parent_missing"}:
        status = "missing"
        remote_sha: Optional[str] = None
        remote_size: Optional[int] = None
    else:
        try:
            remote_sha = _normalise_sha256(parts[2])
            remote_size = int(parts[3])
        except (RemoteArtifactRehydrationError, TypeError, ValueError) as exc:
            raise RemoteArtifactRehydrationError(
                "remote_protocol_invalid",
                detail="remote_identity_invalid",
            ) from exc
        if remote_size < 0:
            raise RemoteArtifactRehydrationError(
                "remote_protocol_invalid",
                detail="remote_size_invalid",
            )
        status = (
            "exact"
            if (
                remote_sha == requirement.sha256
                and (
                    requirement.size_bytes is None
                    or remote_size == requirement.size_bytes
                )
            )
            else "mismatch"
        )
    return {
        "remote_path": requirement.remote_path,
        "roles": list(requirement.roles),
        "expected_sha256": requirement.sha256,
        "expected_size_bytes": requirement.size_bytes,
        "remote_status": status,
        "remote_sha256": remote_sha,
        "remote_size_bytes": remote_size,
        "remote_run_root_status": (
            "missing" if root_missing else "present"
        ),
        "remote_storage_root_device": storage_device,
        "remote_storage_root_inode": storage_inode,
        "remote_parent_status": (
            "missing" if parent_missing else "present"
        ),
        "remote_run_root_device": root_device,
        "remote_run_root_inode": root_inode,
        "exact": status == "exact",
        "safe_to_rehydrate": status in {"missing", "mismatch"},
    }


def _probe_one(
    requirement: RemoteArtifactRequirement,
    *,
    ssh_target: str,
    remote_root: str,
    timeout_s: float,
    runner: Callable[..., Any],
    expected_storage_identity: Optional[tuple[int, int]] = None,
    expected_root_identity: Optional[tuple[int, int]] = None,
) -> dict[str, Any]:
    completed = _invoke_ssh(
        ssh_target=ssh_target,
        remote_command=_remote_command(
            _PROBE_SCRIPT,
            [remote_root, requirement.remote_path],
        ),
        timeout_s=timeout_s,
        runner=runner,
    )
    parts = _parse_protocol(completed, marker="OSPRPROBE1", fields=8)
    observation = _normalise_remote_observation(parts, requirement)
    if expected_storage_identity is not None and (
        observation["remote_storage_root_device"],
        observation["remote_storage_root_inode"],
    ) != expected_storage_identity:
        raise RemoteArtifactRehydrationError(
            "remote_probe_storage_root_identity_drift",
            detail=requirement.remote_path,
        )
    if expected_root_identity is not None and (
        observation["remote_run_root_device"],
        observation["remote_run_root_inode"],
    ) != expected_root_identity:
        raise RemoteArtifactRehydrationError(
            "remote_probe_root_identity_drift",
            detail=requirement.remote_path,
        )
    return observation


def probe_remote_artifact(
    requirement: Any,
    *,
    ssh_target: str,
    remote_run_root: str,
    timeout_s: float = 30.0,
    runner: Optional[Callable[..., Any]] = None,
) -> dict[str, Any]:
    """Read-only probe for one raw remote byte requirement."""

    target = validate_resume_ssh_target(ssh_target)
    root = _normalise_remote_root(remote_run_root)
    normalised = _normalise_requirement(
        requirement, remote_root=root, require_size=False,
    )
    selected_runner = runner if runner is not None else subprocess.run
    timeout = _normalise_timeout(timeout_s)
    observation = _probe_one(
        normalised,
        ssh_target=target,
        remote_root=root,
        timeout_s=timeout,
        runner=selected_runner,
    )
    return {
        "schema": REMOTE_PROBE_SCHEMA,
        "schema_version": REMOTE_PROBE_SCHEMA_VERSION,
        "ok": True,
        "read_only": True,
        "ssh_target": target,
        "remote_run_root": root,
        **observation,
    }


def probe_remote_requirements(
    requirements: Iterable[Any],
    *,
    ssh_target: str,
    remote_run_root: str,
    timeout_s: float = 30.0,
    runner: Optional[Callable[..., Any]] = None,
) -> dict[str, Any]:
    """Read-only batch probe before local artifact resolution.

    ``missing`` and ``mismatch`` are successful probe outcomes.  They identify
    the subset that needs local resolution.  A correct remote target is exact
    even when no local copy exists.
    """

    target = validate_resume_ssh_target(ssh_target)
    root = _normalise_remote_root(remote_run_root)
    normalised = _normalise_requirements(
        requirements, remote_root=root, require_size=False,
    )
    selected_runner = runner if runner is not None else subprocess.run
    timeout = _normalise_timeout(timeout_s)
    entries = [
        _probe_one(
            requirement,
            ssh_target=target,
            remote_root=root,
            timeout_s=timeout,
            runner=selected_runner,
        )
        for requirement in normalised
    ]
    exact_count = sum(bool(entry["exact"]) for entry in entries)
    return {
        "schema": REMOTE_PROBE_SCHEMA,
        "schema_version": REMOTE_PROBE_SCHEMA_VERSION,
        "ok": True,
        "read_only": True,
        "ssh_target": target,
        "remote_run_root": root,
        "requirement_count": len(entries),
        "exact_count": exact_count,
        "rehydration_required_count": len(entries) - exact_count,
        "all_exact": exact_count == len(entries),
        "entries": entries,
    }


def _normalise_source_stat(value: Any) -> Optional[dict[str, int]]:
    if value in (None, ""):
        return None
    if not isinstance(value, Mapping):
        raise RemoteArtifactRehydrationError("invalid_source_stat")
    result: dict[str, int] = {}
    for field in ("device", "inode", "mtime_ns", "size_bytes"):
        raw = value.get(field)
        if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
            raise RemoteArtifactRehydrationError(
                "invalid_source_stat", detail=field,
            )
        result[field] = int(raw)
    return result


def _validate_stage_map(
    stage_map: Mapping[str, Any],
    *,
    remote_root: str,
) -> list[_StageEntry]:
    if not isinstance(stage_map, Mapping):
        raise RemoteArtifactRehydrationError("invalid_stage_map")
    if (
        stage_map.get("schema") != STAGE_MAP_SCHEMA
        or stage_map.get("schema_version") != STAGE_MAP_SCHEMA_VERSION
        or stage_map.get("status") != "ready"
        or stage_map.get("local_only") is not True
        or stage_map.get("transport_performed") is not False
    ):
        raise RemoteArtifactRehydrationError("invalid_stage_map_contract")
    declared_sha = _normalise_sha256(stage_map.get("stage_map_sha256"))
    body = dict(stage_map)
    body.pop("stage_map_sha256", None)
    if _canonical_sha256(body) != declared_sha:
        raise RemoteArtifactRehydrationError("stage_map_sha256_mismatch")
    allowed_roots_raw = stage_map.get("allowed_remote_roots")
    if not isinstance(allowed_roots_raw, list):
        raise RemoteArtifactRehydrationError(
            "stage_map_allowed_roots_invalid",
        )
    allowed_roots = {
        _normalise_remote_root(value) for value in allowed_roots_raw
    }
    if remote_root not in allowed_roots:
        raise RemoteArtifactRehydrationError(
            "frozen_remote_root_not_authorised_by_stage_map",
        )
    raw_entries = stage_map.get("entries")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise RemoteArtifactRehydrationError("stage_map_entries_empty")
    declared_count = stage_map.get("artifact_count")
    if declared_count != len(raw_entries):
        raise RemoteArtifactRehydrationError(
            "stage_map_artifact_count_mismatch",
        )

    entries: list[_StageEntry] = []
    seen: set[str] = set()
    for raw in raw_entries:
        if not isinstance(raw, Mapping):
            raise RemoteArtifactRehydrationError(
                "invalid_stage_map_entry",
            )
        requirement = _normalise_requirement(
            raw, remote_root=remote_root, require_size=True,
        )
        if requirement.remote_path in seen:
            raise RemoteArtifactRehydrationError(
                "duplicate_stage_map_remote_path",
                detail=requirement.remote_path,
            )
        seen.add(requirement.remote_path)
        source_raw = _strict_text(
            raw.get("source_path"), code="invalid_stage_source_path",
        )
        source = Path(source_raw)
        if (
            not source.is_absolute()
            or str(source) != source_raw
            or source_raw.startswith("//")
            or "." in source.parts
            or ".." in source.parts
        ):
            raise RemoteArtifactRehydrationError(
                "invalid_stage_source_path",
                detail=source_raw,
            )
        source_root_raw = raw.get("source_root")
        source_root: Optional[Path]
        if source_root_raw in (None, ""):
            source_root = None
        else:
            source_root_text = _strict_text(
                source_root_raw, code="invalid_stage_source_root",
            )
            source_root = Path(source_root_text)
            if (
                not source_root.is_absolute()
                or str(source_root) != source_root_text
                or source_root_text.startswith("//")
                or "." in source_root.parts
                or ".." in source_root.parts
            ):
                raise RemoteArtifactRehydrationError(
                    "invalid_stage_source_root",
                    detail=source_root_text,
                )
            try:
                source.relative_to(source_root)
            except ValueError as exc:
                raise RemoteArtifactRehydrationError(
                    "stage_source_outside_source_root",
                    detail=source_raw,
                ) from exc
        expected_size_raw = raw.get("expected_size_bytes")
        if (
            expected_size_raw not in (None, "")
            and _normalise_size(
                expected_size_raw, required=True,
            ) != requirement.size_bytes
        ):
            raise RemoteArtifactRehydrationError(
                "stage_map_expected_size_mismatch",
                detail=requirement.remote_path,
            )
        entries.append(
            _StageEntry(
                remote_path=requirement.remote_path,
                sha256=requirement.sha256,
                size_bytes=int(requirement.size_bytes),
                roles=requirement.roles,
                source_path=source,
                source_root=source_root,
                source_stat=_normalise_source_stat(raw.get("source_stat")),
            )
        )
    if stage_map.get("total_bytes") != sum(
        entry.size_bytes for entry in entries
    ):
        raise RemoteArtifactRehydrationError(
            "stage_map_total_bytes_mismatch",
        )
    return sorted(entries, key=lambda entry: entry.remote_path)


def _hash_open_descriptor(descriptor: int) -> str:
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    while True:
        chunk = os.read(descriptor, 4 * 1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
    os.lseek(descriptor, 0, os.SEEK_SET)
    return digest.hexdigest()


def _open_verified_source(entry: _StageEntry) -> tuple[int, os.stat_result]:
    if entry.source_root is not None:
        cursor = entry.source_root
        try:
            relative = entry.source_path.relative_to(entry.source_root)
        except ValueError as exc:
            raise RemoteArtifactRehydrationError(
                "stage_source_outside_source_root",
                detail=str(entry.source_path),
            ) from exc
        for component in ((),) + tuple(
            relative.parts[:index]
            for index in range(1, len(relative.parts) + 1)
        ):
            candidate = cursor if not component else cursor.joinpath(*component)
            try:
                if candidate.is_symlink():
                    raise RemoteArtifactRehydrationError(
                        "local_exact_source_symlink",
                        detail=str(candidate),
                    )
            except OSError as exc:
                raise RemoteArtifactRehydrationError(
                    "local_exact_source_stat_failed",
                    detail=f"{candidate}:{type(exc).__name__}",
                ) from exc
    flags = os.O_RDONLY
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(str(entry.source_path), flags)
    except FileNotFoundError as exc:
        raise RemoteArtifactRehydrationError(
            "local_exact_source_missing",
            detail=str(entry.source_path),
        ) from exc
    except OSError as exc:
        raise RemoteArtifactRehydrationError(
            "local_exact_source_open_failed",
            detail=f"{entry.source_path}:{type(exc).__name__}",
        ) from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RemoteArtifactRehydrationError(
                "local_exact_source_non_regular",
                detail=str(entry.source_path),
            )
        if int(before.st_size) != entry.size_bytes:
            raise RemoteArtifactRehydrationError(
                "local_exact_source_size_mismatch",
                detail=str(entry.source_path),
            )
        if _hash_open_descriptor(descriptor) != entry.sha256:
            raise RemoteArtifactRehydrationError(
                "local_exact_source_sha256_mismatch",
                detail=str(entry.source_path),
            )
        after = os.fstat(descriptor)
        stable = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) == (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        )
        if not stable:
            raise RemoteArtifactRehydrationError(
                "local_exact_source_changed",
                detail=str(entry.source_path),
            )
        if entry.source_stat is not None:
            expected_stat = entry.source_stat
            observed_stat = {
                "device": int(after.st_dev),
                "inode": int(after.st_ino),
                "mtime_ns": int(after.st_mtime_ns),
                "size_bytes": int(after.st_size),
            }
            if observed_stat != expected_stat:
                raise RemoteArtifactRehydrationError(
                    "local_exact_source_stat_drift",
                    detail=str(entry.source_path),
                )
        return descriptor, after
    except Exception:
        os.close(descriptor)
        raise


def _stage_paths(
    entry: _StageEntry,
    *,
    remote_root: str,
    attempt_id: str,
    token: str,
) -> tuple[str, str]:
    target = PurePosixPath(entry.remote_path)
    temporary = str(
        target.parent
        / f".{target.name}.resume-{attempt_id}-{token}.tmp"
    )
    path_digest = hashlib.sha256(
        entry.remote_path.encode("utf-8")
    ).hexdigest()
    backup = str(
        PurePosixPath(remote_root)
        / _RESERVED_REMOTE_PREFIX
        / attempt_id
        / f"{path_digest}.before-{token}"
    )
    _normalise_remote_path(
        temporary, code="invalid_generated_temporary_path",
    )
    _normalise_remote_path(
        backup, code="invalid_generated_backup_path",
    )
    return temporary, backup


def _prepare_remote_tree(
    entries: Sequence[_StageEntry],
    *,
    ssh_target: str,
    remote_root: str,
    expected_root_state: str,
    expected_storage_identity: tuple[int, int],
    expected_root_identity: Optional[tuple[int, int]],
    timeout_s: float,
    runner: Callable[..., Any],
) -> dict[str, Any]:
    parents = sorted({
        str(PurePosixPath(entry.remote_path).parent)
        for entry in entries
    })
    completed = _invoke_ssh(
        ssh_target=ssh_target,
        remote_command=_remote_command(
            _TREE_SCRIPT,
            [
                remote_root,
                expected_root_state,
                str(expected_storage_identity[0]),
                str(expected_storage_identity[1]),
                (
                    str(expected_root_identity[0])
                    if expected_root_identity is not None
                    else "-"
                ),
                (
                    str(expected_root_identity[1])
                    if expected_root_identity is not None
                    else "-"
                ),
                secrets.token_hex(16),
                *parents,
            ],
        ),
        timeout_s=timeout_s,
        runner=runner,
    )
    parts = _parse_protocol(
        completed, marker="OSPRTREE1", fields=8,
    )
    status = parts[1]
    if status != "created":
        raise RemoteArtifactRehydrationError(
            f"remote_tree_{status}",
            detail=remote_root,
        )
    expected_created = expected_root_state == "missing"
    observed_created = parts[2] == "true"
    if parts[2] not in {"true", "false"} or observed_created != expected_created:
        raise RemoteArtifactRehydrationError(
            "remote_protocol_invalid",
            detail="remote_tree_created_root_invalid",
        )
    try:
        created_directory_count = int(parts[3])
        storage_device = int(parts[4])
        storage_inode = int(parts[5])
        root_device = int(parts[6])
        root_inode = int(parts[7])
    except (TypeError, ValueError) as exc:
        raise RemoteArtifactRehydrationError(
            "remote_protocol_invalid",
            detail="remote_tree_created_count_invalid",
        ) from exc
    if (
        created_directory_count < 1
        or storage_device < 0
        or storage_inode <= 0
        or root_device < 0
        or root_inode <= 0
    ):
        raise RemoteArtifactRehydrationError(
            "remote_protocol_invalid",
            detail="remote_tree_created_count_invalid",
        )
    return {
        "status": status,
        "remote_run_root": remote_root,
        "expected_remote_run_root_state": expected_root_state,
        "remote_run_root_created": observed_created,
        "remote_run_root_device": root_device,
        "remote_run_root_inode": root_inode,
        "remote_storage_root_device": storage_device,
        "remote_storage_root_inode": storage_inode,
        "expected_storage_root_device": expected_storage_identity[0],
        "expected_storage_root_inode": expected_storage_identity[1],
        "expected_run_root_device": (
            expected_root_identity[0]
            if expected_root_identity is not None
            else None
        ),
        "expected_run_root_inode": (
            expected_root_identity[1]
            if expected_root_identity is not None
            else None
        ),
        "created_directory_count": created_directory_count,
        "required_parent_count": len(parents),
        "required_parents": parents,
    }


def _upload_source_to_temporary(
    entry: _StageEntry,
    *,
    descriptor: int,
    ssh_target: str,
    remote_root: str,
    expected_storage_identity: tuple[int, int],
    expected_root_identity: tuple[int, int],
    temporary_path: str,
    timeout_s: float,
    runner: Callable[..., Any],
) -> dict[str, Any]:
    os.lseek(descriptor, 0, os.SEEK_SET)
    with os.fdopen(os.dup(descriptor), "rb") as stream:
        completed = _invoke_ssh(
            ssh_target=ssh_target,
            remote_command=_remote_command(
                _UPLOAD_SCRIPT,
                [
                    remote_root,
                    entry.remote_path,
                    temporary_path,
                    entry.sha256,
                    str(entry.size_bytes),
                    str(expected_storage_identity[0]),
                    str(expected_storage_identity[1]),
                    str(expected_root_identity[0]),
                    str(expected_root_identity[1]),
                ],
            ),
            timeout_s=timeout_s,
            runner=runner,
            stdin=stream,
        )
    parts = _parse_protocol(
        completed, marker="OSPRUPLOAD1", fields=4,
    )
    status = parts[1]
    if status != "staged":
        raise RemoteArtifactRehydrationError(
            f"remote_upload_{status}",
            detail=entry.remote_path,
        )
    try:
        observed_sha = _normalise_sha256(parts[2])
        observed_size = _normalise_size(
            int(parts[3]), required=True,
        )
    except (
        RemoteArtifactRehydrationError,
        TypeError,
        ValueError,
    ) as exc:
        raise RemoteArtifactRehydrationError(
            "remote_protocol_invalid",
            detail="remote_upload_identity_invalid",
        ) from exc
    if (
        observed_sha != entry.sha256
        or observed_size != entry.size_bytes
    ):
        raise RemoteArtifactRehydrationError(
            "remote_upload_identity_mismatch",
            detail=entry.remote_path,
        )
    after = os.fstat(descriptor)
    if (
        int(after.st_size) != entry.size_bytes
        or _hash_open_descriptor(descriptor) != entry.sha256
    ):
        raise RemoteArtifactRehydrationError(
            "local_exact_source_changed_during_upload",
            detail=str(entry.source_path),
        )
    return {
        "status": "staged",
        "temporary_path": temporary_path,
        "sha256": entry.sha256,
        "size_bytes": entry.size_bytes,
    }


def _commit_temporary(
    entry: _StageEntry,
    *,
    ssh_target: str,
    remote_root: str,
    expected_storage_identity: tuple[int, int],
    expected_root_identity: tuple[int, int],
    attempt_id: str,
    temporary_path: str,
    backup_path: str,
    timeout_s: float,
    runner: Callable[..., Any],
) -> dict[str, Any]:
    completed = _invoke_ssh(
        ssh_target=ssh_target,
        remote_command=_remote_command(
            _COMMIT_SCRIPT,
            [
                remote_root,
                entry.remote_path,
                temporary_path,
                entry.sha256,
                str(entry.size_bytes),
                attempt_id,
                backup_path,
                str(expected_storage_identity[0]),
                str(expected_storage_identity[1]),
                str(expected_root_identity[0]),
                str(expected_root_identity[1]),
            ],
        ),
        timeout_s=timeout_s,
        runner=runner,
    )
    parts = _parse_protocol(
        completed, marker="OSPRCOMMIT1", fields=7,
    )
    status = parts[1]
    if status not in {"committed", "noop_race"}:
        raise RemoteArtifactRehydrationError(
            f"remote_commit_{status}",
            detail=entry.remote_path,
        )
    try:
        observed_sha = _normalise_sha256(parts[2])
        observed_size = _normalise_size(
            int(parts[3]), required=True,
        )
    except (
        RemoteArtifactRehydrationError,
        TypeError,
        ValueError,
    ) as exc:
        raise RemoteArtifactRehydrationError(
            "remote_protocol_invalid",
            detail="remote_commit_identity_invalid",
        ) from exc
    if (
        observed_sha != entry.sha256
        or observed_size != entry.size_bytes
    ):
        raise RemoteArtifactRehydrationError(
            "remote_commit_identity_mismatch",
            detail=entry.remote_path,
        )
    backup = None if parts[4] == "-" else parts[4]
    try:
        old_sha = (
            None if parts[5] == "-" else _normalise_sha256(parts[5])
        )
        old_size = None if parts[6] == "-" else int(parts[6])
    except (
        RemoteArtifactRehydrationError,
        TypeError,
        ValueError,
    ) as exc:
        raise RemoteArtifactRehydrationError(
            "remote_protocol_invalid",
            detail="remote_commit_replaced_identity_invalid",
        ) from exc
    if old_size is not None and old_size < 0:
        raise RemoteArtifactRehydrationError(
            "remote_protocol_invalid",
            detail="remote_commit_replaced_size_invalid",
        )
    if status == "noop_race" and backup is not None:
        raise RemoteArtifactRehydrationError(
            "remote_protocol_invalid",
            detail="noop_race_with_backup",
        )
    if backup is not None and backup != backup_path:
        raise RemoteArtifactRehydrationError(
            "remote_protocol_invalid",
            detail="unexpected_backup_path",
        )
    return {
        "status": status,
        "backup_path": backup,
        "replaced_sha256": old_sha,
        "replaced_size_bytes": old_size,
    }


def _base_rehydration_report(
    *,
    ssh_target: str,
    remote_root: str,
    attempt_id: str,
    entry_count: int,
) -> dict[str, Any]:
    return {
        "schema": REMOTE_REHYDRATION_SCHEMA,
        "schema_version": REMOTE_REHYDRATION_SCHEMA_VERSION,
        "ok": False,
        "status": "in_progress",
        "ssh_target": ssh_target,
        "remote_run_root": remote_root,
        "resume_attempt_id": attempt_id,
        "entry_count": entry_count,
        "completed_entry_count": 0,
        "no_op_count": 0,
        "rehydrated_count": 0,
        "backup_count": 0,
        "transferred_bytes": 0,
        "remote_run_root_created": False,
        "remote_directory_creation_count": 0,
        "remote_tree_preparation": None,
        "authoritative_final_probe_count": 0,
        "entries": [],
    }


def _raise_with_report(
    error: RemoteArtifactRehydrationError,
    *,
    report: dict[str, Any],
    remote_path: Optional[str] = None,
) -> None:
    report["ok"] = False
    report["status"] = "failed"
    report["failure_code"] = error.code
    report["failure_detail"] = error.detail
    if remote_path is not None:
        report["failed_remote_path"] = remote_path
    raise RemoteArtifactRehydrationError(
        error.code,
        detail=error.detail,
        report=report,
    ) from error


def rehydrate_remote_stage_map(
    stage_map: Mapping[str, Any],
    *,
    ssh_target: str,
    remote_run_root: str,
    resume_attempt_id: str,
    timeout_s: float = 120.0,
    runner: Optional[Callable[..., Any]] = None,
) -> dict[str, Any]:
    """Safely materialise one already-resolved local stage map.

    The remote probe runs before any local source access.  Consequently, an
    exact remote destination remains a no-op even if its former local source
    has since been deleted.  Local sources are required and hash-verified only
    for ``missing`` or ``mismatch`` destinations.
    """

    target = validate_resume_ssh_target(ssh_target)
    root = _normalise_remote_root(remote_run_root)
    attempt_id = _normalise_attempt_id(resume_attempt_id)
    entries = _validate_stage_map(stage_map, remote_root=root)
    selected_runner = runner if runner is not None else subprocess.run
    timeout = _normalise_timeout(timeout_s)
    report = _base_rehydration_report(
        ssh_target=target,
        remote_root=root,
        attempt_id=attempt_id,
        entry_count=len(entries),
    )

    observations: dict[str, dict[str, Any]] = {}
    try:
        for entry in entries:
            observations[entry.remote_path] = _probe_one(
                RemoteArtifactRequirement(
                    remote_path=entry.remote_path,
                    sha256=entry.sha256,
                    size_bytes=entry.size_bytes,
                    roles=entry.roles,
                ),
                ssh_target=target,
                remote_root=root,
                timeout_s=timeout,
                runner=selected_runner,
            )
    except RemoteArtifactRehydrationError as error:
        _raise_with_report(error, report=report)

    storage_root_identities = {
        (
            int(observation["remote_storage_root_device"]),
            int(observation["remote_storage_root_inode"]),
        )
        for observation in observations.values()
    }
    if len(storage_root_identities) != 1:
        _raise_with_report(
            RemoteArtifactRehydrationError(
                "remote_storage_root_identity_drift_during_probe",
                detail=root,
            ),
            report=report,
        )
    pinned_storage_identity = next(iter(storage_root_identities))

    root_missing_flags = {
        observation.get("remote_run_root_status") == "missing"
        for observation in observations.values()
    }
    if len(root_missing_flags) != 1:
        _raise_with_report(
            RemoteArtifactRehydrationError(
                "remote_run_root_state_changed_during_probe",
                detail=root,
            ),
            report=report,
        )
    remote_root_missing = root_missing_flags == {True}
    tree_expected_root_state: Optional[str] = None
    tree_entries: list[_StageEntry] = []
    pinned_root_identity: Optional[tuple[int, int]] = None
    if remote_root_missing:
        tree_expected_root_state = "missing"
        tree_entries = list(entries)
    else:
        root_identities = {
            (
                int(observation["remote_run_root_device"]),
                int(observation["remote_run_root_inode"]),
            )
            for observation in observations.values()
        }
        if len(root_identities) != 1:
            _raise_with_report(
                RemoteArtifactRehydrationError(
                    "remote_run_root_identity_drift_during_probe",
                    detail=root,
                ),
                report=report,
            )
        pinned_root_identity = next(iter(root_identities))
        parent_missing_paths = {
            path
            for path, observation in observations.items()
            if observation.get("remote_parent_status") == "missing"
        }
        if parent_missing_paths:
            tree_expected_root_state = "existing"
            tree_entries = [
                entry
                for entry in entries
                if entry.remote_path in parent_missing_paths
            ]

    # Verify every required local source before the first remote mutation.
    verified_descriptors: dict[str, int] = {}
    try:
        for entry in entries:
            if observations[entry.remote_path]["exact"]:
                continue
            descriptor, _source_stat = _open_verified_source(entry)
            verified_descriptors[entry.remote_path] = descriptor
    except RemoteArtifactRehydrationError as error:
        for descriptor in verified_descriptors.values():
            os.close(descriptor)
        _raise_with_report(
            error,
            report=report,
            remote_path=entry.remote_path,
        )

    active_remote_path: Optional[str] = None
    try:
        if tree_expected_root_state is not None:
            tree = _prepare_remote_tree(
                tree_entries,
                ssh_target=target,
                remote_root=root,
                expected_root_state=tree_expected_root_state,
                expected_storage_identity=pinned_storage_identity,
                expected_root_identity=pinned_root_identity,
                timeout_s=timeout,
                runner=selected_runner,
            )
            report["remote_tree_preparation"] = tree
            report["remote_run_root_created"] = bool(
                tree["remote_run_root_created"]
            )
            report["remote_directory_creation_count"] = int(
                tree["created_directory_count"]
            )
            pinned_root_identity = (
                int(tree["remote_run_root_device"]),
                int(tree["remote_run_root_inode"]),
            )
            if (
                int(tree["remote_storage_root_device"]),
                int(tree["remote_storage_root_inode"]),
            ) != pinned_storage_identity:
                raise RemoteArtifactRehydrationError(
                    "remote_tree_storage_root_identity_drift",
                    detail=root,
                )
        if pinned_root_identity is None:
            raise RemoteArtifactRehydrationError(
                "remote_run_root_identity_unpinned",
                detail=root,
            )
        for entry in entries:
            active_remote_path = entry.remote_path
            observation = observations[entry.remote_path]
            entry_report: dict[str, Any] = {
                "remote_path": entry.remote_path,
                "roles": list(entry.roles),
                "expected_sha256": entry.sha256,
                "expected_size_bytes": entry.size_bytes,
                "initial_probe": observation,
            }
            if observation["exact"]:
                entry_report.update({
                    "ok": True,
                    "status": "no_op_remote_exact",
                    "action": "none",
                    "local_source_accessed": False,
                    "backup_path": None,
                    "final_probe": observation,
                })
                report["no_op_count"] += 1
            else:
                descriptor = verified_descriptors[entry.remote_path]
                token = secrets.token_hex(12)
                temporary_path, backup_path = _stage_paths(
                    entry,
                    remote_root=root,
                    attempt_id=attempt_id,
                    token=token,
                )
                stage = _upload_source_to_temporary(
                    entry,
                    descriptor=descriptor,
                    ssh_target=target,
                    remote_root=root,
                    expected_storage_identity=pinned_storage_identity,
                    expected_root_identity=pinned_root_identity,
                    temporary_path=temporary_path,
                    timeout_s=timeout,
                    runner=selected_runner,
                )
                commit = _commit_temporary(
                    entry,
                    ssh_target=target,
                    remote_root=root,
                    expected_storage_identity=pinned_storage_identity,
                    expected_root_identity=pinned_root_identity,
                    attempt_id=attempt_id,
                    temporary_path=temporary_path,
                    backup_path=backup_path,
                    timeout_s=timeout,
                    runner=selected_runner,
                )
                final_probe = _probe_one(
                    RemoteArtifactRequirement(
                        remote_path=entry.remote_path,
                        sha256=entry.sha256,
                        size_bytes=entry.size_bytes,
                        roles=entry.roles,
                    ),
                    ssh_target=target,
                    remote_root=root,
                    timeout_s=timeout,
                    runner=selected_runner,
                    expected_storage_identity=pinned_storage_identity,
                    expected_root_identity=pinned_root_identity,
                )
                if not final_probe["exact"]:
                    raise RemoteArtifactRehydrationError(
                        "remote_final_probe_identity_mismatch",
                        detail=entry.remote_path,
                    )
                entry_report.update({
                    "ok": True,
                    "status": (
                        "no_op_remote_became_exact"
                        if commit["status"] == "noop_race"
                        else "rehydrated"
                    ),
                    "action": (
                        "none"
                        if commit["status"] == "noop_race"
                        else "atomic_replace"
                    ),
                    "local_source_accessed": True,
                    "source_path": str(entry.source_path),
                    "stage": stage,
                    "commit": commit,
                    "backup_path": commit["backup_path"],
                    "final_probe": final_probe,
                })
                if commit["status"] == "noop_race":
                    report["no_op_count"] += 1
                else:
                    report["rehydrated_count"] += 1
                    report["transferred_bytes"] += entry.size_bytes
                    if commit["backup_path"] is not None:
                        report["backup_count"] += 1
            report["entries"].append(entry_report)
            report["completed_entry_count"] += 1

        authoritative: dict[str, dict[str, Any]] = {}
        for entry in entries:
            active_remote_path = entry.remote_path
            final_observation = _probe_one(
                RemoteArtifactRequirement(
                    remote_path=entry.remote_path,
                    sha256=entry.sha256,
                    size_bytes=entry.size_bytes,
                    roles=entry.roles,
                ),
                ssh_target=target,
                remote_root=root,
                timeout_s=timeout,
                runner=selected_runner,
                expected_storage_identity=pinned_storage_identity,
                expected_root_identity=pinned_root_identity,
            )
            if final_observation["exact"] is not True:
                raise RemoteArtifactRehydrationError(
                    "remote_authoritative_final_probe_identity_mismatch",
                    detail=entry.remote_path,
                )
            authoritative[entry.remote_path] = final_observation
        for entry_report in report["entries"]:
            entry_report["final_probe"] = authoritative[
                entry_report["remote_path"]
            ]
        report["authoritative_final_probe_count"] = len(authoritative)
    except RemoteArtifactRehydrationError as error:
        _raise_with_report(
            error,
            report=report,
            remote_path=active_remote_path,
        )
    finally:
        for descriptor in verified_descriptors.values():
            os.close(descriptor)

    report["ok"] = True
    report["status"] = "complete"
    report["all_remote_exact"] = True
    return report


__all__ = [
    "REMOTE_PROBE_SCHEMA",
    "REMOTE_PROBE_SCHEMA_VERSION",
    "REMOTE_REHYDRATION_SCHEMA",
    "REMOTE_REHYDRATION_SCHEMA_VERSION",
    "RemoteArtifactRequirement",
    "RemoteArtifactRehydrationError",
    "probe_remote_artifact",
    "probe_remote_requirements",
    "rehydrate_remote_stage_map",
    "validate_resume_ssh_target",
]
