#!/usr/bin/env bash
# Hardware-free AP50:95/AP50/AP75 replay from a v2.75.24 Debug Pack.

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  printf '%s\n' 'SKIP: run_v27524_yolov7_pack_replay.sh was sourced.' >&2
  return 0
fi
set -Eeuo pipefail

TOOL="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
PY="$TOOL/.venv/bin/python"
if (( $# < 1 || $# > 2 )); then
  printf 'Usage: bash scripts/run_v27524_yolov7_pack_replay.sh DEBUG_PACK [OUT_DIR]\n' >&2
  exit 64
fi
if [[ ! -x "$PY" ]]; then
  printf 'FEHLER: Tool-Venv fehlt: %s\n' "$PY" >&2
  exit 69
fi

PACK="$(realpath -e -- "$1")"
SIDECAR="$(realpath -e -- "$PACK.manifest.json")"
OUT="${2:-${PACK%.zip}_offline_replay_v27524}"
mkdir -p -- "$(dirname -- "$OUT")"
OUT_PARENT="$(realpath -e -- "$(dirname -- "$OUT")")"
OUT="$OUT_PARENT/$(basename -- "$OUT")"
if [[ -e "$OUT" ]]; then
  printf 'STOP: Replay-Ziel existiert bereits: %s\n' "$OUT" >&2
  exit 73
fi

FREE_BYTES="$(df -B1 --output=avail "$OUT_PARENT" | awk 'NR==2 {print $1}')"
FREE_INODES="$(df --output=iavail "$OUT_PARENT" | awk 'NR==2 {print $1}')"
MIN_FREE_BYTES=$((2 * 1024 * 1024 * 1024))
MIN_FREE_INODES=10000
if (( FREE_BYTES < MIN_FREE_BYTES )); then
  printf 'STOP: Für Entpacken und Replay werden mindestens 2 GiB freier Speicher benötigt: %s\n' \
    "$OUT_PARENT" >&2
  exit 70
fi
if (( FREE_INODES < MIN_FREE_INODES )); then
  printf 'STOP: Für Entpacken und Replay werden mindestens %s freie Inodes benötigt: %s\n' \
    "$MIN_FREE_INODES" "$OUT_PARENT" >&2
  exit 71
fi

cd -- "$TOOL"
unset PYTHONOPTIMIZE
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$TOOL${PYTHONPATH:+:$PYTHONPATH}"

"$PY" -B - "$PACK" "$SIDECAR" <<'PY'
import hashlib
import json
import sys
import zipfile
from pathlib import Path

pack = Path(sys.argv[1]).resolve(strict=True)
sidecar = Path(sys.argv[2]).resolve(strict=True)
side = json.loads(sidecar.read_text(encoding="utf-8"))
digest = hashlib.sha256()
with pack.open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
sha = digest.hexdigest()
assert side.get("status") == "verified", side.get("status")
assert side.get("archive_sha256") == "sha256:" + sha
assert side.get("archive_size_bytes") == pack.stat().st_size
assert side.get("archive_name") == pack.name
with zipfile.ZipFile(pack) as archive:
    assert archive.testzip() is None
    names = archive.namelist()
    assert len(names) == len(set(names)), "duplicate ZIP member names"
    manifest = json.loads(archive.read("debug_pack_manifest.json"))
    inputs = manifest.get("central_quality_replay_inputs") or {}
    assert inputs.get("hash_valid") is True, inputs
    assert not inputs.get("missing_members"), inputs
    assert not inputs.get("failures"), inputs
    expected = set(inputs.get("expected_members") or [])
    present = set(inputs.get("present_members") or [])
    assert expected and present == expected, (present, expected)
    assert expected <= set(names), sorted(expected - set(names))
    required = set(side.get("required_members") or [])
    assert {"debug_pack_manifest.json", *expected} <= required
print("PASS Debug-Pack replay inputs")
PY

TMP_REPLAY="$(mktemp -d -p "$OUT_PARENT" .v27524-pack-replay.XXXXXXXX)"
cleanup_tmp() {
  case "$TMP_REPLAY" in
    "$OUT_PARENT"/.v27524-pack-replay.*) rm -rf -- "$TMP_REPLAY" ;;
    *) printf 'WARNUNG: temporäres Verzeichnis nicht entfernt: %s\n' "$TMP_REPLAY" >&2 ;;
  esac
}
trap cleanup_tmp EXIT
EXTRACTED="$TMP_REPLAY/evaluation_run"
mkdir -p -- "$EXTRACTED"

"$PY" -B - "$PACK" "$SIDECAR" "$EXTRACTED" <<'PY'
import hashlib
import json
import shutil
import stat
import sys
import zipfile
from pathlib import Path, PurePosixPath

pack = Path(sys.argv[1]).resolve(strict=True)
side = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
root = Path(sys.argv[3]).resolve(strict=True)
max_members = 20_000
max_member_bytes = 128 * 1024 * 1024
max_total_bytes = 1024 * 1024 * 1024

# Hash and extract through one open descriptor so replacing the archive path
# between verification and extraction cannot change the bytes being consumed.
with pack.open("rb") as raw:
    digest = hashlib.sha256()
    for block in iter(lambda: raw.read(1024 * 1024), b""):
        digest.update(block)
    assert side.get("archive_sha256") == "sha256:" + digest.hexdigest()
    assert side.get("archive_size_bytes") == raw.tell()
    raw.seek(0)
    with zipfile.ZipFile(raw) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        assert len(infos) <= max_members, len(infos)
        assert len(names) == len(set(names)), "duplicate ZIP member names"
        assert sum(info.file_size for info in infos) <= max_total_bytes
        for info in infos:
            name = info.filename
            member = PurePosixPath(name)
            assert name and "\\" not in name and "\x00" not in name, name
            assert not member.is_absolute(), name
            assert member.parts and all(part not in {"", ".", ".."} for part in member.parts), name
            assert info.file_size <= max_member_bytes, (name, info.file_size)
            assert not (info.flag_bits & 0x1), f"encrypted ZIP member: {name}"
            mode = (info.external_attr >> 16) & 0xFFFF
            file_type = stat.S_IFMT(mode)
            assert not stat.S_ISLNK(mode), f"symlink ZIP member: {name}"
            assert file_type in {0, stat.S_IFREG, stat.S_IFDIR}, (
                name, oct(file_type)
            )
            target = root.joinpath(*member.parts)
            target.resolve(strict=False).relative_to(root)
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info, "r") as source, target.open("xb") as destination:
                shutil.copyfileobj(source, destination, length=1024 * 1024)
print("PASS bounded safe Debug-Pack extraction")
PY

STAGED_OUT="$TMP_REPLAY/offline_replay"
REPLAY_STDOUT="$("$PY" -B scripts/replay_central_quality.py \
  --eval-run-dir "$EXTRACTED" \
  --out-dir "$STAGED_OUT" \
  --workers 4 \
  --full-only)"
printf '%s\n' "$REPLAY_STDOUT" | awk \
  '!/^REPLAY_JSON=/ && !/^REPLAY_CSV=/'

STAGED_REPLAY_JSON="$STAGED_OUT/central_quality_replay_v27522.json"
STAGED_REPLAY_CSV="$STAGED_OUT/central_quality_replay_v27522.csv"
if [[ ! -s "$STAGED_REPLAY_JSON" || ! -s "$STAGED_REPLAY_CSV" ]]; then
  printf 'FEHLER: Replay hat JSON/CSV nicht vollständig materialisiert.\n' >&2
  exit 74
fi
QUALITY_DECISION="$("$PY" -B - "$STAGED_REPLAY_JSON" "$STAGED_REPLAY_CSV" <<'PY'
import csv
import json
import sys
from pathlib import Path

replay = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
with Path(sys.argv[2]).open("r", encoding="utf-8", newline="") as stream:
    csv_rows = list(csv.DictReader(stream))
rows = replay.get("canonical_full_only_results") or []
expected = {
    ("hailo8", "orin_nx_hailo8_01", "full"),
    ("native_full_tensorrt", "orin_nx_hailo8_01", "full"),
    ("hailo10", "orin_nx_hailo10_01", "full"),
    ("native_full_tensorrt", "orin_nx_hailo10_01", "full"),
}
observed = {
    (
        str(row.get("source_run_id") or ""),
        str(row.get("source_setup_id") or row.get("setup_id") or ""),
        str(row.get("variant") or ""),
    )
    for row in rows
}
csv_observed = {
    (
        str(row.get("source_run_id") or ""),
        str(row.get("source_setup_id") or ""),
        str(row.get("variant") or ""),
    )
    for row in csv_rows
}
assert replay.get("schema") == "onnx-splitpoint/central-quality-offline-replay"
assert replay.get("schema_version") == 1
assert replay.get("status") == "completed"
assert replay.get("technical_status") == "ok"
assert replay.get("full_only") is True
assert replay.get("hardware_executed") is False
assert replay.get("historical_summary_mutated") is False
assert replay.get("request_count") == 4
assert replay.get("canonical_full_only_status") == "complete", replay
assert not replay.get("canonical_full_only_errors"), replay.get(
    "canonical_full_only_errors"
)
assert len(rows) == 4 and observed == expected, (len(rows), observed)
assert len(csv_rows) == 4 and csv_observed == expected, (
    len(csv_rows), csv_observed
)
for row in rows:
    assert int(row.get("n") or 0) == 5000, row.get("n")
    assert row.get("configured_guardrails") == ["ap50", "ap75"]
    assert row.get("guardrail_contract_complete") is True
    seed_schema = row.get("seed_schema") or {}
    assert int(seed_schema.get("seed") or 0) == 20260710, seed_schema
    assert int(seed_schema.get("repetitions") or 0) == 5000, seed_schema
    assert int(seed_schema.get("image_count") or 0) == 5000, seed_schema
    assert int(row.get("bootstrap_workers_requested") or 0) == 4, row
    guardrails = row.get("guardrails") or {}
    assert "ap50" in guardrails and "ap75" in guardrails, guardrails
    assert (row.get("primary") or {}).get("metric") == "coco_ap_50_95"
    for component in (row.get("primary"), guardrails["ap50"], guardrails["ap75"]):
        assert component.get("decision") in {"pass", "fail", "inconclusive"}
        assert int(component.get("n") or 0) == 5000, component
        assert float(component.get("margin")) == 0.01, component
        assert int(component.get("bootstrap_repetitions_requested") or 0) == 5000
        effective = int(component.get("bootstrap_repetitions") or 0)
        skipped = str(component.get("bootstrap_skipped_reason") or "")
        assert (effective, skipped) in {
            (5000, ""),
            (0, "candidate_reference_identical"),
            (0, "point_estimate_below_non_inferiority_margin"),
        }, (effective, skipped)
    assert row.get("decision") in {"pass", "fail", "inconclusive"}
decision = str(replay.get("canonical_full_only_quality_decision") or "")
assert decision in {"pass", "fail", "inconclusive"}, decision
print(decision)
PY
)"

if [[ -e "$OUT" ]]; then
  printf 'STOP: Replay-Ziel wurde während der Auswertung angelegt: %s\n' "$OUT" >&2
  exit 73
fi
mv -T -- "$STAGED_OUT" "$OUT"
REPLAY_JSON="$OUT/central_quality_replay_v27522.json"
REPLAY_CSV="$OUT/central_quality_replay_v27522.csv"

printf 'OFFLINE_REPLAY=PASS\n'
printf 'REPLAY_JSON=%s\n' "$REPLAY_JSON"
printf 'REPLAY_CSV=%s\n' "$REPLAY_CSV"
printf 'QUALITY_DECISION=%s\n' "$QUALITY_DECISION"
