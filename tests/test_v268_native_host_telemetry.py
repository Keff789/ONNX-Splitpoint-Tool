from __future__ import annotations

import importlib.util
import hashlib
import json
import zipfile
from pathlib import Path

from onnx_splitpoint_tool.workflow.analysis_pack import create_analysis_pack
from onnx_splitpoint_tool.workflow.run_discovery import build_measurement_set_contract


ROOT = Path(__file__).resolve().parents[1]


def _telemetry_module():
    path = ROOT / "scripts" / "native_host_telemetry.py"
    spec = importlib.util.spec_from_file_location("native_host_telemetry", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _capture(
    phase: str, *, setup_id: str = "jetson_a",
    capture_group: str = "", include_nvpmodel: bool = True,
) -> dict:
    payload = {
        "schema": "onnx-splitpoint/native-host-telemetry",
        "schema_version": 1,
        "captured_at": "2026-07-20T00:00:00+0000",
        "phase": phase,
        "backend": "hailo8",
        "setup_id": setup_id,
        "host": {"hostname": "jetson-a"},
        "nvidia_commands": {
            "nvpmodel_query": {"stdout": "NV Power Mode: MAXN"},
            "jetson_clocks_show": {"stdout": "SOC family:tegra234"},
        },
        "cpu_frequency": [
            {"path": "/cpu0/cpufreq", "scaling_governor": "performance", "scaling_min_freq": "100", "scaling_max_freq": "200"}
        ],
        "device_frequency": [
            {"path": "/sys/class/devfreq/gpu", "governor": "performance", "min_freq": "300", "max_freq": "900"}
        ],
        "thermal_zones": [{"type": "GPU-therm", "temperature_c": 51.0 if phase == "pre" else 58.0}],
    }
    if capture_group:
        payload["capture_group"] = capture_group
    if not include_nvpmodel:
        payload["nvidia_commands"]["nvpmodel_query"]["stdout"] = ""
    return payload


def test_native_host_telemetry_pre_post_comparison(tmp_path: Path) -> None:
    module = _telemetry_module()
    folder = tmp_path / "hailo8" / "host_telemetry"
    folder.mkdir(parents=True)
    (folder / "pre.json").write_text(json.dumps(_capture("pre")), encoding="utf-8")
    (folder / "post.json").write_text(json.dumps(_capture("post")), encoding="utf-8")

    result = module.compare_root(tmp_path)

    assert result["capture_count"] == 2
    assert result["complete_pair_count"] == 1
    assert result["stable_pair_count"] == 1
    assert result["checks"][0]["post_maximum_temperature_c"] == 58.0


def test_native_host_telemetry_pairs_each_capture_group_independently(tmp_path: Path) -> None:
    module = _telemetry_module()
    folder = tmp_path / "hailo8" / "host_telemetry"
    folder.mkdir(parents=True)
    for group in ("raw_head", "integrated_nms"):
        for phase in ("pre", "post"):
            (folder / f"{group}_{phase}.json").write_text(
                json.dumps(_capture(phase, capture_group=group)),
                encoding="utf-8",
            )

    result = module.compare_root(tmp_path)

    assert result["capture_count"] == 4
    assert result["capture_group_count"] == 2
    assert result["complete_pair_count"] == 2
    assert result["stable_pair_count"] == 2
    assert {row["capture_group"] for row in result["checks"]} == {
        "raw_head", "integrated_nms",
    }


def test_cross_setup_equality_fails_closed_when_a_field_is_unobserved(tmp_path: Path) -> None:
    module = _telemetry_module()
    for directory, setup_id, include_nvpmodel in (
        ("hailo8", "jetson_a", True),
        ("deepx", "jetson_b", False),
    ):
        folder = tmp_path / directory / "host_telemetry"
        folder.mkdir(parents=True)
        for phase in ("pre", "post"):
            (folder / f"performance_{phase}.json").write_text(
                json.dumps(_capture(
                    phase,
                    setup_id=setup_id,
                    capture_group="performance",
                    include_nvpmodel=include_nvpmodel,
                )),
                encoding="utf-8",
            )

    result = module.compare_root(tmp_path)
    nvpmodel = next(
        row for row in result["cross_setup_configuration"]
        if row["field"] == "nvpmodel"
    )

    assert result["setup_count"] == 2
    assert nvpmodel["expected_setup_count"] == 2
    assert nvpmodel["observed_setup_count"] == 1
    assert nvpmodel["complete_across_setups_and_capture_groups"] is False
    assert nvpmodel["equal_across_observed_setups"] is False
    assert result["all_observed_setup_configurations_equal"] is False


def test_missing_capture_group_post_is_not_reported_stable(tmp_path: Path) -> None:
    module = _telemetry_module()
    folder = tmp_path / "hailo8" / "host_telemetry"
    folder.mkdir(parents=True)
    (folder / "performance_pre.json").write_text(
        json.dumps(_capture("pre", capture_group="performance")),
        encoding="utf-8",
    )

    result = module.compare_root(tmp_path)

    assert result["complete_pair_count"] == 0
    assert result["stable_pair_count"] == 0
    assert result["all_pairs_complete_and_stable"] is False
    assert result["all_complete_pairs_stable"] is False
    assert result["checks"][0]["pair_complete_and_unambiguous"] is False


def test_analysis_pack_archives_host_telemetry(tmp_path: Path, monkeypatch) -> None:
    run = tmp_path / "run"
    scientific = run / "reports" / "scientific"
    scientific.mkdir(parents=True)
    (run / "models" / "model" / "benchmark_results").mkdir(parents=True)
    (run / "run_manifest.json").write_text(
        json.dumps({
            "schema": "onnx-splitpoint/evaluation-run-manifest",
            "schema_version": 1,
            "run_id": run.name,
            "profile_id": "profile",
            "status": "ok",
            "model_count": 1,
            "models": {"model": {"model_id": "model"}},
        }),
        encoding="utf-8",
    )
    (run / "profile.yaml").write_text("profile_id: profile\n", encoding="utf-8")
    normalized = run / "models" / "model" / "benchmark_results" / "normalized_results.json"
    normalized.write_text(
        json.dumps({
            "schema": "onnx-splitpoint/normalized-benchmark-results",
            "schema_version": 2,
            "evaluation_run_id": run.name,
            "model_id": "model",
            "status": "measured",
            "matrix_complete": True,
            "result_count": 1,
            "missing_measurement_count": 0,
            "missing_required_profile_result_count": 0,
            "duplicate_required_profile_result_count": 0,
            "validation_cardinality_mismatch_count": 0,
            "results": [{"model_id": "model", "backend": "ort_cpu"}],
        }),
        encoding="utf-8",
    )
    measurement_set = build_measurement_set_contract(run)
    assert measurement_set["valid"] is True
    report_path = scientific / "scientific_report.json"
    report_path.write_text(
        json.dumps({
            "schema": "onnx-splitpoint/scientific-report",
            "schema_version": 3,
            "run_id": run.name,
            "measurement_set_sha256": measurement_set["measurement_set_sha256"],
            "measurement_result_count": measurement_set["result_count"],
            "rows": [{"model_id": "model", "backend": "ort_cpu"}],
        }),
        encoding="utf-8",
    )
    for name in (
        "row_eligibility.csv",
        "task_quality.csv",
        "performance_results.csv",
        "energy_results.csv",
    ):
        (scientific / name).write_text(
            "model_id,backend\nmodel,ort_cpu\n", encoding="utf-8"
        )
    report_artifacts = [
        report_path,
        scientific / "row_eligibility.csv",
        scientific / "task_quality.csv",
        scientific / "performance_results.csv",
        scientific / "energy_results.csv",
    ]
    (scientific / "report_manifest.json").write_text(
        json.dumps({
            "schema": "onnx-splitpoint/scientific-report-manifest",
            "schema_version": 2,
            "artifacts": [
                {
                    "path": path.name,
                    "size_bytes": path.stat().st_size,
                    "sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
                }
                for path in report_artifacts
            ],
        }),
        encoding="utf-8",
    )
    (run / "reports" / "run_status_summary.json").write_text(
        json.dumps({
            "schema": "onnx-splitpoint/run-status-summary",
            "schema_version": 1,
            "run_id": run.name,
            "status": "ok",
        }),
        encoding="utf-8",
    )
    (run / "reports" / "results_bundle_manifest.json").write_text(
        json.dumps({
            "schema": "onnx-splitpoint/results-bundle-manifest",
            "schema_version": 2,
            "run_id": run.name,
            "contains_measured_benchmarks": True,
            "measurement_set": measurement_set,
            "contains_scientific_report": True,
            "outputs": {
                "scientific_report_json":
                    "reports/scientific/scientific_report.json"
            },
        }),
        encoding="utf-8",
    )
    (run / "reports" / "native_host_telemetry_summary.json").write_text("{}", encoding="utf-8")
    host = run / "native_producers" / "hailo8" / "host_telemetry"
    host.mkdir(parents=True)
    (host / "pre.json").write_text(json.dumps(_capture("pre")), encoding="utf-8")
    (host / "post.json").write_text(json.dumps(_capture("post")), encoding="utf-8")
    monkeypatch.setattr(
        "onnx_splitpoint_tool.workflow.analysis_pack.CANONICAL_TABLE_FILES",
        (),
    )
    out = tmp_path / "analysis.zip"

    create_analysis_pack(run, out, materialize_missing_report=False)

    with zipfile.ZipFile(out) as archive:
        names = set(archive.namelist())
    assert "99_provenance/reports/native_host_telemetry_summary.json" in names
    assert "99_provenance/host_telemetry/hailo8/pre.json" in names
    assert "99_provenance/host_telemetry/hailo8/post.json" in names
