"""Exact live probes must survive a service plan that omitted unavailable HEFs."""
import json

import pytest

from onnx_splitpoint_tool.workflow.artifact_cache_preflight import (
    build_artifact_cache_preflight,
    collect_model_artifact_cache_probes,
    resolve_artifact_cache_preflight_policy,
)
from onnx_splitpoint_tool.workflow.deferred_hailo_builds import cache_preflight_builder
from test_v27922_negative_preflight import _result, _suite, _write


def _prepare(root):
    model, suite, output, source = _suite(root)
    _write(suite / "benchmark_plan.json", {"runs": []})
    cache_preflight_builder(lambda *a, **k: _result(negative=True))(
        source, outdir=str(output), hw_arch="hailo8", net_name="yolo26s_part1_b364",
    )
    # Before 24, this omission discarded the exact negative cache probe.
    _write(model / "benchmark_set/hailo_artifact_service_plan.json", {
        "case_hef_requests": [],
    })
    return model, suite, output


def _report(root):
    observations, roles = collect_model_artifact_cache_probes(
        run_dir=root, model_id="yolo26s", targets=["hailo8"],
        policy=resolve_artifact_cache_preflight_policy({}),
    )
    return build_artifact_cache_preflight(
        model_ids=["yolo26s"], observations=observations,
        applicable_roles={"yolo26s": list(roles)},
    )


def test_exact_negative_probe_is_visible_when_service_plan_omits_case(tmp_path):
    _prepare(tmp_path)
    report = _report(tmp_path)
    row = next(r for r in report["artifact_matrix"] if r["item_id"] == "b364:part1")
    assert row["status"] == "KNOWN_INFEASIBLE"
    assert row["identity"] == "exact-existing-backend-key"
    assert row["compiler_dispatch_allowed"] is False
    assert report["known_infeasible_count"] == 1
    assert report["cold_builds_required"] == 0


def test_unselected_negative_probe_is_still_excluded(tmp_path):
    _model, suite, _output = _prepare(tmp_path)
    _write(suite / "benchmark_set.json", {"cases": [], "plan": {"runs": []}})
    assert _report(tmp_path)["known_infeasible_count"] == 0


@pytest.mark.parametrize("mutation", ["model", "boundary", "backend"])
def test_mismatched_probe_address_cannot_create_negative_matrix_row(tmp_path, mutation):
    _model, _suite_dir, output = _prepare(tmp_path)
    for path in output.glob("*.json"):
        if path.name not in {"deferred_hailo_build.json", "hailo_negative_evidence.json", "hailo_cache_miss.json"}:
            continue
        value = json.loads(path.read_text())
        coordinates = value.get("kwargs", value)
        if mutation == "model":
            coordinates["net_name"] = "different_model_part1_b364"
        elif mutation == "boundary":
            coordinates["net_name"] = "yolo26s_part1_b365"
        else:
            coordinates["hw_arch"] = "hailo10"
        _write(path, value)
    assert _report(tmp_path)["known_infeasible_count"] == 0


def test_current_miss_supersedes_old_negative_even_without_service_plan_case(tmp_path):
    _model, _suite_dir, output = _prepare(tmp_path)
    cache_preflight_builder(lambda *a, **k: _result())(
        output.parents[2] / "part1.onnx", outdir=str(output),
        hw_arch="hailo8", net_name="yolo26s_part1_b364",
    )
    request = json.loads((output / "deferred_hailo_build.json").read_text())
    request["cache_probe_status"] = "MISS"
    request["cache_probe_reason"] = "new_recipe_exact_miss"
    _write(output / "deferred_hailo_build.json", request)
    report = _report(tmp_path)
    assert report["known_infeasible_count"] == 0
    row = next(r for r in report["artifact_matrix"] if r["item_id"] == "b364:part1")
    assert row["status"] == "MISS"
    assert row["reason"] == "new_recipe_exact_miss"
