import json
from pathlib import Path

from onnx_splitpoint_tool.workflow.artifact_cache_preflight import (
    ROLE_DEEPX,
    ROLE_HAILO8,
    STATUS_HIT,
    STATUS_UNKNOWN,
    build_artifact_cache_preflight,
    collect_model_artifact_cache_probes,
    resolve_artifact_cache_preflight_policy,
)


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _suite(root: Path, cases: list[str], runs: list[dict]) -> Path:
    suite = root / "models/model/benchmark_set/legacy_suite"
    _write(suite / "benchmark_set.json", {"cases": [{"id": case} for case in cases]})
    _write(suite / "benchmark_plan.json", {"runs": runs})
    return suite


def _collect(root: Path, targets: list[str]):
    return collect_model_artifact_cache_probes(
        run_dir=root, model_id="model", targets=targets,
        policy=resolve_artifact_cache_preflight_policy({"artifact_cache_preflight": {"require_warm_cache": True}}),
    )


def test_missing_selected_case_probe_cannot_hide_behind_another_hit(tmp_path: Path) -> None:
    suite = _suite(tmp_path, ["b132", "b398"], [{
        "id": "deepx_to_trt", "type": "matrix",
        "stage1": "deepx_m1", "stage2": "tensorrt", "variants": ["composed"],
    }])
    _write(suite / "b132/deepx/deepx_m1/part1/deepx_part1_artifact_status.json", {
        "build_status": "ready_reused", "cache_lookup": {"outcome": "HIT", "reason": "verified"},
    })
    observations, roles = _collect(tmp_path, ["deepx_m1", "tensorrt"])
    local = {row.item_id: row for row in observations if row.role == ROLE_DEEPX}
    assert local["b132:part1"].status == STATUS_HIT
    assert local["b398:part1"].status == STATUS_UNKNOWN
    assert local["b398:part1"].reason == "required_artifact_cache_probe_missing"
    report = build_artifact_cache_preflight(
        model_ids=["model"], observations=observations,
        applicable_roles={"model": sorted(roles)}, block_on_unexpected_cold_builds=True,
    )
    assert report["runtime_dispatch_allowed"] is False
    assert report["confirmed_miss_count"] == 0


def test_deepx_part2_and_final_case_filter_exclude_old_suite_and_aggregate(tmp_path: Path) -> None:
    suite = _suite(tmp_path, ["b398"], [{
        "id": "trt_to_deepx_m1", "type": "matrix",
        "stage1": "tensorrt", "stage2": {"backend": "deepx_m1"},
        "variants": ["part2"],
    }])
    hit = {"build_status": "ready_reused", "cache_lookup": {"outcome": "HIT", "reason": "verified"}}
    _write(suite / "b398/deepx/deepx_m1/part2/deepx_part2_artifact_status.json", hit)
    _write(suite / "b132/deepx/deepx_m1/part2/deepx_part2_artifact_status.json", hit)
    _write(suite / "deepx/deepx_m1/part1/deepx_part1_artifact_status.json", {"selected": True, "cases": [{"case_id": "b132"}]})
    _write(suite.parent / "suite/b398/deepx/part1/deepx_part1_artifact_status.json", hit)
    observations, _ = _collect(tmp_path, ["deepx_m1"])
    assert [(row.item_id, row.status) for row in observations if row.role == ROLE_DEEPX] == [("b398:part2", STATUS_HIT)]


def test_missing_hailo_part1_and_part2_are_explicit_and_disabled_full_is_not_required(tmp_path: Path) -> None:
    suite = _suite(tmp_path, ["b398"], [
        {"id": "hailo8_to_trt", "type": "matrix", "stage1": "hailo8", "stage2": "tensorrt", "variants": ["composed"]},
        {"id": "trt_to_hailo8", "type": "matrix", "stage1": "tensorrt", "stage2": "hailo8", "variants": ["part2"]},
        {"id": "hailo8", "type": "full", "backend": "hailo8"},
    ])
    _write(suite.parent / "hailo_artifact_service_plan.json", {
        "full_baseline_requests": [{"backend": "hailo8", "requested": False}],
        "case_hef_requests": [{"backend": "hailo8", "case_id": "b132", "stage": "part1", "status": "pending_dfc_build"}],
    })
    observations, _ = _collect(tmp_path, ["hailo8"])
    hailo = [row for row in observations if row.role == ROLE_HAILO8]
    assert {(row.item_id, row.status) for row in hailo} == {
        ("b398:part1", STATUS_UNKNOWN), ("b398:part2", STATUS_UNKNOWN),
    }


def test_disabled_run_and_unselected_row_cases_do_not_create_requirements(tmp_path: Path) -> None:
    _suite(tmp_path, ["b398"], [
        {"id": "deepx_to_trt", "type": "matrix", "stage1": "deepx", "stage2": "tensorrt", "enabled": False},
        {"id": "hailo8_to_trt", "type": "matrix", "stage1": "hailo8", "stage2": "tensorrt", "case_ids": [132], "variants": ["part1"]},
    ])
    observations, _ = _collect(tmp_path, ["hailo8", "deepx_m1"])
    assert not [row for row in observations if row.role in {ROLE_HAILO8, ROLE_DEEPX}]


def test_reverse_pipeline_requires_explicit_tensorrt_part1_probe_row(tmp_path: Path) -> None:
    from onnx_splitpoint_tool.workflow.artifact_cache_preflight import ROLE_TRT_P1
    _suite(tmp_path, ['b398', 'b420'], [{
        'id': 'trt_to_hailo10h', 'type': 'matrix',
        'stage1': {'provider': 'tensorrt'}, 'stage2': 'hailo10h',
        'variants': ['composed'], 'case_ids': ['b398'],
    }])
    observations, roles = _collect(tmp_path, ['hailo10h', 'tensorrt'])
    rows = [row for row in observations if row.role == ROLE_TRT_P1]
    assert len(rows) == 1
    assert rows[0].boundary == 'b398'
    assert rows[0].artifact_stage == 'part1'
    assert rows[0].backend == 'tensorrt'
    assert rows[0].status == 'UNKNOWN'
    assert rows[0].reason == 'tensorrt_part1_probe_unavailable'
    report = build_artifact_cache_preflight(model_ids=['model'], observations=observations,
        applicable_roles={'model': list(roles)}, block_on_unexpected_cold_builds=True)
    assert report['matrix'][0]['cells']['trt_p1']['status'] == 'UNKNOWN'
    assert report['runtime_dispatch_allowed'] is False
    assert not [row for row in report['cold_build_rows'] if row['role'] == 'trt_p1']


def test_exact_remote_tensorrt_part1_supersedes_missing_probe_placeholder(tmp_path: Path) -> None:
    from onnx_splitpoint_tool.workflow.artifact_cache_preflight import ROLE_TRT_P1
    _suite(tmp_path, ['b398'], [{
        'id': 'trt_to_hailo10h', 'type': 'matrix', 'stage1': 'tensorrt',
        'stage2': 'hailo10h', 'variants': ['composed'],
    }])
    observations, roles = collect_model_artifact_cache_probes(
        run_dir=tmp_path, model_id='model', targets=['hailo10h', 'tensorrt'],
        policy=resolve_artifact_cache_preflight_policy({}),
        remote_trt_observations=[{
            'model_id': 'model', 'role': 'trt_p1', 'item_id': 'orin/b398',
            'status': 'HIT', 'reason': 'existing_builder_and_source_receipt_valid',
            'identity': 'existing-part1-engine-identity',
        }],
    )
    p1 = [row for row in observations if row.role == ROLE_TRT_P1]
    assert len(p1) == 1
    assert p1[0].status == STATUS_HIT
    assert p1[0].item_id == 'orin/b398'
    assert p1[0].boundary == 'b398'
    report = build_artifact_cache_preflight(model_ids=['model'], observations=observations,
        applicable_roles={'model': list(roles)})
    assert report['matrix'][0]['cells']['trt_p1']['status'] == STATUS_HIT
