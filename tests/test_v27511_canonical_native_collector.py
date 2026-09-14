from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from scripts import native_producer_final_report as collector


PRECISION = "float32_layout_fp16"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _result_path(backend_root: Path, *, variant: str = "") -> Path:
    base = backend_root / variant if variant else backend_root
    return (
        base
        / "resnet50/benchmark_set/native_pipeline/b052/hailo_to_trt"
        / PRECISION
        / "native_fifo_results.json"
    )


def _raw_result(runtime_token: str = "one") -> dict[str, Any]:
    record = {
        "repetition_index": 1,
        "repetition_id": f"hailo8:{runtime_token}",
        "runtime_instance_id": f"fresh_process:{runtime_token}",
        "workload_contract_sha256": "c" * 64,
        "repetition_runtime_scope": "fresh_process_per_repetition",
        "ok": True,
        "status": "ok",
        "fps_makespan": 123.0,
        "latency_mean_ms": 1000.0 / 123.0,
    }
    return {
        "ok": True,
        "fps_makespan": 123.0,
        "frames": 10,
        "warmup": 2,
        "task": "classification",
        "repetition_count_requested": 1,
        "repetition_count_attempted": 1,
        "repetition_count_valid": 1,
        "repetition_status": "complete",
        "repetition_aggregation": "median_never_best_of",
        "repetition_records": [record],
        "fps_repetition_samples": [123.0],
    }


def _write_runner_summary(
    backend_root: Path,
    result_path: Path,
    *,
    status: str = "ok",
    result_ok: bool = True,
    returncode: int = 0,
    child_result_fresh: bool = True,
) -> None:
    row = {
        "model": "resnet50",
        "case_id": "b052",
        "precision": PRECISION,
        "status": status,
        "result_ok": result_ok,
        "returncode": returncode,
        "timed_out": False,
        "child_result_fresh": child_result_fresh,
        # Exercise the production rsync projection: remote paths no longer
        # exist locally, so the collector must resolve the staged default.
        "native_fifo_result": "/remote/native_fifo_results.json",
        "native_fifo_result_sha256": hashlib.sha256(
            result_path.read_bytes()
        ).hexdigest(),
        "native_fifo_result_size_bytes": result_path.stat().st_size,
        "native_split_quality_required": True,
        "performance_claims_emitted": False,
        "execution_role": "cache_verify_diagnostic_replay",
    }
    _write_json(
        backend_root / "analysis_tables/native_fifo_eval_runner.json",
        {"precision": PRECISION, "rows": [row]},
    )


def _production_tree(tmp_path: Path) -> tuple[Path, Path, Path]:
    variants = tmp_path / "native_producers/variants"
    backend_root = variants / "v000_cache_verify/hailo8"
    result_path = _result_path(backend_root)
    _write_json(result_path, _raw_result())
    return variants, backend_root, result_path


def test_recursive_variant_collection_keeps_the_runner_envelope(
    tmp_path: Path,
) -> None:
    variants, backend_root, result_path = _production_tree(tmp_path)
    _write_runner_summary(backend_root, result_path)
    roots = collector._find_eval_roots(variants, True)

    normal = collector._aggregate_repetitions(
        collector._rows_from_native_fifo_roots(roots)
    )
    reversed_order = collector._aggregate_repetitions(
        collector._rows_from_native_fifo_roots(reversed(roots))
    )

    assert len(normal) == 1
    assert len(reversed_order) == 1
    for row in (normal[0], reversed_order[0]):
        assert row["ok"] is True
        assert row["returncode"] == 0
        assert row["timed_out"] is False
        assert row["child_result_fresh"] is True
        assert row["report"] == str(result_path)
        assert row["native_fifo_result_sha256"] == hashlib.sha256(
            result_path.read_bytes()
        ).hexdigest()
        assert row["native_fifo_result_size_bytes"] == result_path.stat().st_size
        assert row["performance_claims_emitted"] is False
        assert row["execution_role"] == "cache_verify_diagnostic_replay"


def test_failed_runner_row_cannot_be_laundered_by_ok_raw_result(
    tmp_path: Path,
) -> None:
    variants, backend_root, result_path = _production_tree(tmp_path)
    _write_runner_summary(
        backend_root,
        result_path,
        status="failed",
        result_ok=False,
        returncode=9,
        child_result_fresh=False,
    )

    rows = collector._rows_from_native_fifo_roots(
        collector._find_eval_roots(variants, True)
    )

    assert len(rows) == 1
    assert rows[0]["ok"] is False
    assert rows[0]["status"] == "failed"
    assert rows[0]["returncode"] == 9
    assert rows[0]["child_result_fresh"] is False
    assert rows[0]["analysis_summary"].endswith(
        "analysis_tables/native_fifo_eval_runner.json"
    )


def test_legacy_raw_only_result_is_still_collected(tmp_path: Path) -> None:
    root = tmp_path / "legacy_hailo8"
    result_path = _result_path(root)
    _write_json(result_path, _raw_result("legacy"))

    rows = collector._rows_from_native_fifo_roots(
        collector._find_eval_roots(root, True)
    )

    assert len(rows) == 1
    assert rows[0]["ok"] is True
    assert rows[0]["report"] == str(result_path)
    assert "analysis_summary" not in rows[0]


def test_same_case_independent_raw_runtimes_are_not_coarsely_deduplicated(
    tmp_path: Path,
) -> None:
    root = tmp_path / "legacy_variants"
    first = _result_path(root, variant="variant_a")
    second = _result_path(root, variant="variant_b")
    _write_json(first, _raw_result("first"))
    _write_json(second, _raw_result("second"))

    rows = collector._rows_from_native_fifo_roots([root])

    assert len(rows) == 2
    assert {row["report"] for row in rows} == {str(first), str(second)}
    runtime_ids = {
        row["repetition_records"][0]["runtime_instance_id"] for row in rows
    }
    assert runtime_ids == {"fresh_process:first", "fresh_process:second"}
