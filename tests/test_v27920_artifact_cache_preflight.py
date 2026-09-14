from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from onnx_splitpoint_tool.workflow import artifact_cache_preflight as cache_preflight

from onnx_splitpoint_tool.workflow.artifact_cache_preflight import (
    ROLE_DEEPX,
    ROLE_HAILO10,
    ROLE_HAILO8,
    ROLE_TRT_FULL,
    ROLE_TRT_P2,
    STATUS_HIT,
    STATUS_MISS,
    STATUS_NOT_APPLICABLE,
    STATUS_UNKNOWN,
    build_artifact_cache_preflight,
    collect_model_artifact_cache_probes,
    resolve_artifact_cache_preflight_policy,
    write_artifact_cache_preflight,
)


def _row(
    model: str,
    role: str,
    status: str,
    *,
    item: str = "full",
    reason: str = "compatible_receipt",
    expectation: str = "warm",
) -> dict[str, str]:
    return {
        "model_id": model,
        "role": role,
        "item_id": item,
        "status": status,
        "reason": reason,
        "expectation": expectation,
        "identity": f"existing-{model}-{role}-{item}",
        "artifact_path": f"/cache/{model}/{role}/{item}",
        "receipt_path": f"/cache/{model}/{role}/{item}.json",
        "source": "backend_existing_receipt_probe",
    }


def test_seven_model_warm_matrix_has_no_cold_builds() -> None:
    models = [
        "resnet50",
        "mobilenet_v3_large",
        "regnet_x_1_6gf",
        "yolo11l",
        "yolo26m",
        "yolo26s",
        "yolov7",
    ]
    observations = [
        _row(model, role, STATUS_HIT)
        for model in models
        for role in (
            ROLE_HAILO8,
            ROLE_HAILO10,
            ROLE_DEEPX,
            ROLE_TRT_FULL,
            ROLE_TRT_P2,
        )
    ]
    report = build_artifact_cache_preflight(
        model_ids=models,
        observations=observations,
        default_expectation="warm",
        created_at="2026-09-04T00:00:00Z",
    )

    assert report["status"] == "warm"
    assert report["model_count"] == 7
    assert report["hit_count"] == 35
    assert report["confirmed_miss_count"] == 0
    assert report["expected_cold_builds"] == 0
    assert report["unexpected_cold_builds"] == 0
    assert report["unknown_count"] == 0
    assert report["runtime_dispatch_allowed"] is True
    assert all(
        cell["status"] == STATUS_HIT
        for model in report["matrix"]
        for role, cell in model["cells"].items()
        if role != "trt_p1"
    )
    assert all(model["cells"]["trt_p1"]["status"] == STATUS_NOT_APPLICABLE for model in report["matrix"])


def test_only_confirmed_miss_is_counted_as_cold_build() -> None:
    observations = [
        _row("regnet", ROLE_HAILO8, STATUS_HIT),
        _row(
            "regnet", ROLE_HAILO10, STATUS_UNKNOWN,
            reason="remote_probe_unavailable",
        ),
        _row("regnet", ROLE_DEEPX, STATUS_HIT),
        _row(
            "regnet", ROLE_TRT_FULL, STATUS_MISS,
            reason="not_found", expectation="warm",
        ),
        _row(
            "regnet", ROLE_TRT_P2, STATUS_MISS,
            item="b132", reason="not_found", expectation="cold",
        ),
    ]
    report = build_artifact_cache_preflight(
        model_ids=["regnet"], observations=observations,
        block_on_unexpected_cold_builds=True,
    )

    assert report["status"] == "unexpected_cold_builds"
    assert report["confirmed_miss_count"] == 2
    assert report["cold_builds_required"] == 2
    assert report["expected_cold_builds"] == 1
    assert report["unexpected_cold_builds"] == 1
    assert report["unknown_count"] == 1
    assert report["runtime_dispatch_allowed"] is False

    cells = report["matrix"][0]["cells"]
    assert cells[ROLE_HAILO10]["status"] == STATUS_UNKNOWN
    assert cells[ROLE_TRT_FULL]["status"] == STATUS_MISS
    assert cells[ROLE_TRT_P2]["status"] == STATUS_MISS


def test_unavailable_remote_probe_is_not_promoted_to_hit_or_miss() -> None:
    report = build_artifact_cache_preflight(
        model_ids=["yolo11l"],
        observations=[],
        applicable_roles={"yolo11l": [ROLE_TRT_FULL, ROLE_TRT_P2]},
        default_expectation="warm",
    )
    cells = report["matrix"][0]["cells"]

    assert cells[ROLE_TRT_FULL]["status"] == STATUS_UNKNOWN
    assert cells[ROLE_TRT_P2]["status"] == STATUS_UNKNOWN
    assert cells[ROLE_HAILO8]["status"] == STATUS_NOT_APPLICABLE
    assert report["confirmed_miss_count"] == 0
    assert report["cold_builds_required"] == 0
    assert report["expected_cold_builds"] == 0
    assert report["unexpected_cold_builds"] == 0
    # Each applicable backend without a returned probe is explicitly unknown.
    assert report["unknown_count"] == 2
    assert report["status"] == "probe_incomplete"

    strict = build_artifact_cache_preflight(
        model_ids=["yolo11l"], observations=[],
        applicable_roles={"yolo11l": [ROLE_TRT_FULL]},
        default_expectation="warm",
        block_on_unexpected_cold_builds=True,
    )
    assert strict["unknown_count"] == 1
    assert strict["strict_unknown_blocker_count"] == 1
    assert strict["runtime_dispatch_allowed"] is False


def test_multiple_required_items_roll_up_to_miss() -> None:
    report = build_artifact_cache_preflight(
        model_ids=["yolo26s"],
        observations=[
            _row(
                "yolo26s", ROLE_HAILO8, STATUS_HIT,
                item="b024", expectation="warm",
            ),
            _row(
                "yolo26s", ROLE_HAILO8, STATUS_MISS,
                item="b038", reason="receipt_missing",
                expectation="warm",
            ),
        ],
        applicable_roles={"yolo26s": [ROLE_HAILO8]},
    )
    cell = report["matrix"][0]["cells"][ROLE_HAILO8]

    assert cell["status"] == STATUS_MISS
    assert cell["item_count"] == 2
    assert cell["counts"]["hit"] == 1
    assert cell["counts"]["miss"] == 1
    assert report["unexpected_cold_builds"] == 1


def test_aliases_normalize_trt_probe_payload_without_new_hash() -> None:
    report = build_artifact_cache_preflight(
        model_ids=["regnet"],
        observations=[{
            "model": "regnet",
            "role": "tensorrt_full",
            "case_id": "full",
            "status": "cache_hit",
            "reason": "compatible_receipt",
            "cache_key": "fb-existing-engine-identity",
            "engine": "/remote/full_fp16.engine",
            "receipt": "/remote/native_trt_meta.json",
            "expectation": "expected_hit",
        }],
        applicable_roles={"regnet": [ROLE_TRT_FULL]},
    )
    item = report["observations"][0]

    assert item["status"] == STATUS_HIT
    assert item["role"] == ROLE_TRT_FULL
    assert item["identity"] == "fb-existing-engine-identity"
    assert item["artifact_path"] == "/remote/full_fp16.engine"
    assert item["receipt_path"] == "/remote/native_trt_meta.json"
    assert "sha256" not in report


def test_duplicate_probe_identity_is_rejected() -> None:
    duplicate = _row("resnet50", ROLE_TRT_FULL, STATUS_HIT)
    with pytest.raises(ValueError, match="artifact_cache_probe_duplicate"):
        build_artifact_cache_preflight(
            model_ids=["resnet50"],
            observations=[duplicate, duplicate],
        )


def test_json_csv_and_markdown_views_are_written(tmp_path: Path) -> None:
    report = build_artifact_cache_preflight(
        model_ids=["resnet50"],
        observations=[
            _row("resnet50", ROLE_TRT_FULL, STATUS_HIT),
        ],
        applicable_roles={"resnet50": [ROLE_TRT_FULL]},
        created_at="2026-09-04T00:00:00Z",
    )
    paths = write_artifact_cache_preflight(report, output_dir=tmp_path)

    loaded = json.loads(paths["artifact_cache_preflight_json"].read_text())
    assert loaded["schema"] == "onnx-splitpoint/artifact-cache-preflight"
    with paths["artifact_cache_preflight_csv"].open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["model_id"] == "resnet50"
    assert rows[0][ROLE_TRT_FULL] == STATUS_HIT
    markdown = paths["artifact_cache_preflight_md"].read_text()
    assert "| Model | H8 HEF | H10 HEF | DeepX | TRT Full | TRT P2 |" in markdown
    assert "Expected cold builds: 0" in markdown
    assert "UNKNOWN is not counted as a cache hit or a cold build." in markdown


def test_profile_policy_is_diagnostic_unless_strict_is_explicit() -> None:
    diagnostic = resolve_artifact_cache_preflight_policy({})
    strict = resolve_artifact_cache_preflight_policy({
        "artifact_cache_preflight": {
            "require_warm_cache": True,
            "expected_cold": ["yolo26s:trt_p2:b024"],
        },
    })

    assert diagnostic["enabled"] is True
    assert diagnostic["default_expectation"] == "unspecified"
    assert diagnostic["block_on_unexpected_cold_builds"] is False
    assert strict["default_expectation"] == "warm"
    assert strict["block_on_unexpected_cold_builds"] is True
    assert strict["declarations"] == [{
        "model_id": "yolo26s",
        "role": ROLE_TRT_P2,
        "item_id": "b024",
        "expectation": "cold",
    }]


def test_remote_trt_without_probe_is_unknown_per_required_artifact(
    tmp_path: Path,
) -> None:
    bdir = tmp_path / "models/yolo26s/benchmark_set"
    bdir.mkdir(parents=True)
    (bdir / "benchmark_set.json").write_text(json.dumps({
        "cases": [{"id": "b024"}, {"id": "b038"}],
    }))
    (bdir / "benchmark_plan.json").write_text(json.dumps({
        "runs": [{"id": "hailo8_to_trt"}],
    }))
    policy = resolve_artifact_cache_preflight_policy({})

    observations, roles = collect_model_artifact_cache_probes(
        run_dir=tmp_path,
        model_id="yolo26s",
        targets=["hailo8", "tensorrt"],
        policy=policy,
    )
    trt = [row for row in observations if row.role.startswith("trt_")]

    assert roles >= {ROLE_HAILO8, ROLE_TRT_P2}
    assert ROLE_TRT_FULL not in roles
    assert {(row.role, row.item_id) for row in trt} == {
        (ROLE_TRT_P2, "b024"),
        (ROLE_TRT_P2, "b038"),
    }
    assert all(row.status == STATUS_UNKNOWN for row in trt)
    assert all(
        row.reason == "remote_trt_cache_probe_unavailable" for row in trt
    )


def test_deepx_existing_cache_decisions_are_reused_without_rehashing(
    tmp_path: Path,
) -> None:
    bdir = tmp_path / "models/regnet/benchmark_set"
    full = bdir / "deepx/deepx_artifact_status.json"
    part1 = bdir / "suite/b132/deepx/deepx_m1/part1/deepx_part1_artifact_status.json"
    full.parent.mkdir(parents=True)
    part1.parent.mkdir(parents=True)
    full.write_text(json.dumps({
        "selected": True,
        "build_status": "ready_reused",
        "cache_lookup": {
            "outcome": "HIT", "reason": "artifact_identity_verified",
            "identity": "existing-full-key", "artifact": "/cache/full.dxnn",
        },
    }))
    part1.write_text(json.dumps({
        "build_status": "ready_built",
        "cache_lookup": {
            "outcome": "MISS", "reason": "not_found",
            "identity": "existing-part1-key", "artifact": "",
        },
    }))

    observations, roles = collect_model_artifact_cache_probes(
        run_dir=tmp_path, model_id="regnet", targets=["deepx_m1"],
        policy=resolve_artifact_cache_preflight_policy({}),
    )

    assert roles == {ROLE_DEEPX}
    assert [(row.item_id, row.status, row.reason) for row in observations] == [
        ("full", STATUS_HIT, "artifact_identity_verified"),
        ("b132:part1", STATUS_MISS, "not_found"),
    ]
    assert observations[0].identity == "existing-full-key"


def test_setup_scoped_remote_item_uses_unscoped_policy_declaration(
    tmp_path: Path,
) -> None:
    bdir = tmp_path / "models/model/benchmark_set"
    (bdir / "models").mkdir(parents=True)
    (bdir / "models/model.onnx").write_bytes(b"model")
    (bdir / "benchmark_set.json").write_text(json.dumps({
        "model_id": "model", "model": "models/model.onnx", "cases": [],
    }))
    (bdir / "benchmark_plan.json").write_text(json.dumps({
        "runs": [{
            "id": "ort_tensorrt", "type": "onnxruntime",
            "provider": "tensorrt",
        }],
    }))
    policy = resolve_artifact_cache_preflight_policy({
        "artifact_cache_preflight": {
            "require_warm_cache": True,
            "expected_cold": [{
                "model_id": "model", "role": "trt_full",
                "item_id": "full",
            }],
        },
    })
    observations, _roles = collect_model_artifact_cache_probes(
        run_dir=tmp_path, model_id="model", targets=["tensorrt"],
        policy=policy,
        remote_trt_observations=[{
            "model_id": "model", "role": "trt_full",
            "item_id": "setup_a/full", "status": "MISS",
            "reason": "not_found",
        }],
    )
    row = next(value for value in observations if value.role == ROLE_TRT_FULL)
    assert row.item_id == "setup_a/full"
    assert row.expectation == "cold"


def test_hailo_aliases_and_current_build_remain_visible_as_cold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert cache_preflight._hailo_role("hailo8l") == ROLE_HAILO8
    assert cache_preflight._hailo_role("hailo8r") == ROLE_HAILO8
    assert cache_preflight._hailo_role("hailo10p") == ROLE_HAILO10

    bdir = tmp_path / "models/model/benchmark_set"
    bdir.mkdir(parents=True)
    hef = tmp_path / "cache/model.hef"
    hef.parent.mkdir()
    hef.write_bytes(b"hef")
    (bdir / "hailo_artifact_service_plan.json").write_text(json.dumps({
        "full_baseline_requests": [{
            "backend": "hailo8l", "requested": True,
            "status": "ready_existing_hef", "hef_path": str(hef),
            "build_attempt": {"ok": True, "skipped": False},
        }],
    }))
    import onnx_splitpoint_tool.hailo_backend as hailo_backend
    monkeypatch.setattr(
        hailo_backend, "_load_valid_hailo_receipt",
        lambda _path: {"hw_arch": "hailo8l", "cache_key": "existing"},
    )
    monkeypatch.setattr(
        hailo_backend, "_hailo_receipt_path",
        lambda path: Path(str(path) + ".receipt.json"),
    )
    observations, roles = collect_model_artifact_cache_probes(
        run_dir=tmp_path, model_id="model", targets=["hailo8l"],
        policy=resolve_artifact_cache_preflight_policy({}),
    )
    row = next(value for value in observations if value.role == ROLE_HAILO8)
    assert roles == {ROLE_HAILO8}
    assert row.status == STATUS_MISS
    assert row.reason == "built_during_preparation_artifact_now_ready"
    assert row.evidence["artifact_now_ready"] is True


def test_deepx_identity_unresolved_is_unknown_not_confirmed_miss(
    tmp_path: Path,
) -> None:
    status = (
        tmp_path / "models/model/benchmark_set/deepx"
        / "deepx_artifact_status.json"
    )
    status.parent.mkdir(parents=True)
    status.write_text(json.dumps({
        "selected": True, "build_status": "pending",
        "cache_lookup": {
            "outcome": "MISS", "reason": "backend_identity_unresolved",
        },
    }))
    observations, roles = collect_model_artifact_cache_probes(
        run_dir=tmp_path, model_id="model", targets=["deepx_m1"],
        policy=resolve_artifact_cache_preflight_policy({}),
    )
    row = next(value for value in observations if value.role == ROLE_DEEPX)
    assert roles == {ROLE_DEEPX}
    assert row.status == STATUS_UNKNOWN
    assert row.reason == "backend_identity_unresolved"
