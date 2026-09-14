#!/usr/bin/env python3
"""Run one Native Hailo Full baseline from a generated BenchmarkSet.

Despite the historical filename this wrapper supports both Hailo-8 and
Hailo-10H.  It resolves the exact Full HEF, the canonical full ONNX model, the
benchmark task and (optionally) the exact validation image, then delegates to
``smoke_hailo10_hef_runner.py`` in the currently selected Hailo runtime Python.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _aliases(hw_arch: str) -> list[str]:
    value = str(hw_arch or "").strip().lower()
    out = [value]
    if value == "hailo10":
        out.append("hailo10h")
    elif value == "hailo10h":
        out.append("hailo10")
    elif value in {"hailo8l", "hailo8r"}:
        out.append("hailo8")
    return list(dict.fromkeys(x for x in out if x))


def find_hef(benchmark_set: Path, hw_arch: str) -> Path:
    candidates: list[Path] = []
    for alias in _aliases(hw_arch):
        candidates.extend([
            benchmark_set / "hailo" / alias / "full" / "compiled.hef",
            benchmark_set / "hailo" / alias / "full" / "model.hef",
            benchmark_set / "legacy_suite" / "hailo" / alias / "full" / "compiled.hef",
            benchmark_set / "legacy_suite" / "hailo" / alias / "full" / "model.hef",
        ])
        candidates.extend(sorted(benchmark_set.glob(f"**/hailo/{alias}/full/**/*.hef")))
    seen: set[str] = set()
    for path in candidates:
        if ".hailo-generations" in path.parts:
            continue  # Retained generations must never win artifact discovery.
        try:
            key = str(path.resolve())
        except Exception:
            key = str(path)
        if key in seen:
            continue
        seen.add(key)
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(f"No Full HEF found for {hw_arch} under {benchmark_set}")


def find_full_onnx(benchmark_set: Path) -> Path | None:
    candidates: list[Path] = []
    for root in (benchmark_set / "models", benchmark_set / "legacy_suite" / "models"):
        if root.is_dir():
            candidates.extend(sorted(root.glob("*.onnx")))
    candidates.extend(sorted(benchmark_set.glob("**/models/*.onnx")))
    for path in candidates:
        name = path.name.lower()
        if path.is_file() and "part1" not in name and "part2" not in name and "split" not in name:
            return path.resolve()
    return None


def benchmark_task(benchmark_set: Path, model: str = "") -> str:
    for name in ("benchmark_set.json", "benchmark_plan.json"):
        payload = _load_json(benchmark_set / name)
        if not isinstance(payload, Mapping):
            continue
        for key in ("benchmark_task", "model_task", "task"):
            value = str(payload.get(key) or "").strip().lower()
            if value in {"classification", "detection"}:
                return value
        runs = payload.get("runs") or payload.get("planned_runs")
        if isinstance(runs, list):
            for row in runs:
                if isinstance(row, Mapping):
                    value = str(row.get("benchmark_task") or row.get("task") or "").strip().lower()
                    if value in {"classification", "detection"}:
                        return value
    low = str(model or benchmark_set.parent.name).lower()
    return "detection" if any(token in low for token in ("yolo", "coco", "detect")) else "classification"


def benchmark_model_id(benchmark_set: Path, requested: str = "") -> str:
    value = str(requested or "").strip()
    if value:
        return value
    payload = _load_json(benchmark_set / "benchmark_set.json")
    if isinstance(payload, Mapping):
        value = str(payload.get("model_id") or payload.get("model_name") or "").strip()
        if value:
            return value
    return str(benchmark_set.parent.name)


def resolve_image(benchmark_set: Path, value: str) -> Path | None:
    text = str(value or "").strip()
    if text:
        raw = Path(text).expanduser()
        if raw.is_file():
            return raw.resolve()
        for candidate in (benchmark_set / raw, benchmark_set / "legacy_suite" / raw):
            if candidate.is_file():
                return candidate.resolve()
        if raw.name:
            matches = sorted(path for path in benchmark_set.rglob(raw.name) if path.is_file())
            if matches:
                return matches[0].resolve()
    root = benchmark_set / "resources" / "validation"
    if root.is_dir():
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                return path.resolve()
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-set", required=True)
    parser.add_argument("--hw-arch", default="hailo10h", choices=["hailo10h", "hailo10", "hailo10n", "hailo8", "hailo8l", "hailo8r"])
    parser.add_argument("--model", default="")
    parser.add_argument("--frames", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--inflight", type=int, default=8)
    parser.add_argument("--runtime-api", default="auto", choices=["auto", "infer_model", "vstreams"])
    parser.add_argument("--image", default="")
    parser.add_argument("--task", default="", choices=["", "classification", "detection"])
    parser.add_argument("--preprocess-mode", default="auto", choices=["auto", "resize", "letterbox"])
    parser.add_argument("--letterbox-pad-value", type=int, default=114)
    parser.add_argument("--dump-outputs", action="store_true")
    parser.add_argument("--dump-dir", default="")
    parser.add_argument(
        "--artifacts-dir",
        default="",
        help=(
            "Isolated HailoBackend artifact root forwarded to the HEF runner. "
            "Defaults to that runner's historical /tmp location."
        ),
    )
    parser.add_argument(
        "--declared-output-contract-json",
        default="",
        help=(
            "Optional exact output_contracts.json container. When set, this "
            "takes precedence over the BenchmarkSet's adjacent declaration."
        ),
    )
    parser.add_argument(
        "--diagnostic-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Mark the isolated child report and semantic manifest as "
            "diagnostic and suppress every claim-eligibility field."
        ),
    )
    parser.add_argument("--backend-label", default="")
    parser.add_argument("--setup-id", default="")
    parser.add_argument("--comparison-backend", default="")
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    benchmark_set = Path(args.benchmark_set).expanduser().resolve()
    hef = find_hef(benchmark_set, args.hw_arch)
    onnx = find_full_onnx(benchmark_set)
    model_id = benchmark_model_id(benchmark_set, args.model)
    task = str(args.task or benchmark_task(benchmark_set, model_id)).lower()
    image = resolve_image(benchmark_set, args.image)
    runtime_api = str(args.runtime_api or "auto")
    if runtime_api == "auto":
        runtime_api = "vstreams" if str(args.hw_arch).lower().startswith("hailo8") else "infer_model"
    backend_label = str(args.backend_label or ("native_full_hailo8" if str(args.hw_arch).lower().startswith("hailo8") else "native_full_hailo10h"))

    smoke = Path(__file__).resolve().parent / "smoke_hailo10_hef_runner.py"
    if not smoke.is_file():
        raise FileNotFoundError(f"Missing smoke_hailo10_hef_runner.py next to {__file__}")

    command = [
        sys.executable, str(smoke),
        "--hef", str(hef),
        "--hw-arch", str(args.hw_arch),
        "--runtime-api", runtime_api,
        "--throughput-mode",
        "--copy-outputs",
        "--frames", str(max(1, int(args.frames))),
        "--warmup", str(max(0, int(args.warmup))),
        "--inflight", str(max(1, int(args.inflight))),
        "--task", task,
        "--preprocess-mode", str(args.preprocess_mode),
        "--letterbox-pad-value", str(int(args.letterbox_pad_value)),
        "--backend-label", backend_label,
        "--model", model_id,
        "--setup-id", str(args.setup_id or ""),
        "--comparison-backend", str(args.comparison_backend or ""),
    ]
    if args.artifacts_dir:
        command += [
            "--artifacts-dir",
            str(Path(args.artifacts_dir).expanduser().resolve()),
        ]
    if str(args.hw_arch).lower().startswith("hailo8"):
        command.append("--persistent-activation")
    if onnx is not None:
        command += ["--onnx", str(onnx)]
    # Never let an adjacent single-artifact declaration bypass the exact
    # suite-root model/backend/task resolver in the HEF runner.
    declared_output_contract = (
        Path(args.declared_output_contract_json).expanduser().resolve()
        if str(args.declared_output_contract_json or "").strip()
        else benchmark_set / "output_contracts.json"
    )
    if str(args.declared_output_contract_json or "").strip() and (
        not declared_output_contract.is_file()
        or declared_output_contract.name != "output_contracts.json"
    ):
        raise FileNotFoundError(
            "Explicit declared output contract must be an existing "
            "output_contracts.json"
        )
    if declared_output_contract.is_file():
        command += ["--declared-output-contract-json", str(declared_output_contract)]
    if args.diagnostic_only:
        command.append("--diagnostic-only")
    if image is not None:
        command += ["--image", str(image)]
    if args.dump_outputs:
        if image is None:
            raise FileNotFoundError("--dump-outputs requires a resolvable --image or a materialised validation image")
        command.append("--dump-outputs")
        if args.dump_dir:
            command += ["--dump-dir", str(Path(args.dump_dir).expanduser())]
    if args.json_out:
        command += ["--json-out", str(Path(args.json_out).expanduser())]

    print("[hailo-full-from-bs] hef:", hef, flush=True)
    print("[hailo-full-from-bs] hw_arch:", args.hw_arch, "runtime_api:", runtime_api, flush=True)
    print("[hailo-full-from-bs] task:", task, "image:", image or "<random>", flush=True)
    print("[hailo-full-from-bs] cmd:", " ".join(str(x) for x in command), flush=True)
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
