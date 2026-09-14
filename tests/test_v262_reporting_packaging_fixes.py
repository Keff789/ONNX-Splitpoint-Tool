from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from onnx_splitpoint_tool.validation.accuracy_gates import (
    AccuracyGatePolicy,
    apply_accuracy_gate_to_row,
)
from onnx_splitpoint_tool.workflow.scientific_reporting import (
    _load_evalrun_rows,
    _scientific_row,
    _tex_escape,
    _write_tex_table,
)
from onnx_splitpoint_tool.workflow.zip_utils import (
    publish_verified_zip,
    temporary_zip_path,
    verify_zip_archive,
)
from scripts import build_source_manifest


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    "name",
    (
        "embedded.whl", "nested.zip", "trace.parquet", "model.onnx",
        "part1.hef", "model.dxnn", "part2.engine", "prepared.npy",
        "debug.log",
    ),
)
def test_source_manifest_excludes_binaries_archives_and_measurement_evidence(
    tmp_path: Path, name: str,
) -> None:
    path = tmp_path / name
    path.write_bytes(b"not-source")
    assert build_source_manifest._included(path, tmp_path) is False


def test_verified_zip_is_atomic_and_has_external_final_identity(tmp_path: Path) -> None:
    destination = tmp_path / "debug.zip"
    temporary = temporary_zip_path(destination)
    assert temporary.parent == destination.parent
    assert not temporary.exists()

    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("debug_pack_manifest.json", "{}")
        archive.writestr("payload/report.json", '{"ok": true}')

    published = publish_verified_zip(
        temporary,
        destination,
        required_members=("debug_pack_manifest.json",),
        build_identity={"build_id": "test-build", "package_version": "2.62.0"},
        manifest_extra={"pack_kind": "evaluation_debug_pack"},
    )

    assert destination.is_file()
    assert not temporary.exists()
    assert verify_zip_archive(destination)["status"] == "verified"
    with zipfile.ZipFile(destination) as archive:
        assert archive.testzip() is None

    sidecar = Path(published["verification_manifest"])
    identity = json.loads(sidecar.read_text(encoding="utf-8"))
    assert identity["archive_size_bytes"] == destination.stat().st_size
    assert identity["archive_sha256"] == _sha256(destination)
    assert identity["tool_build"]["build_id"] == "test-build"
    assert identity["structural_check"] == "central_directory_and_all_member_crc"
    assert identity["pack_kind"] == "evaluation_debug_pack"


@pytest.mark.parametrize("failure", ["truncated", "missing_required", "duplicate"])
def test_failed_zip_verification_never_publishes_destination(
    tmp_path: Path, failure: str
) -> None:
    destination = tmp_path / f"{failure}.zip"
    temporary = temporary_zip_path(destination)
    if failure == "truncated":
        temporary.write_bytes(b"PK\x03\x04truncated")
        expected = zipfile.BadZipFile
    elif failure == "missing_required":
        with zipfile.ZipFile(temporary, "w") as archive:
            archive.writestr("some_other_file.json", "{}")
        expected = ValueError
    else:
        with pytest.warns(UserWarning, match="Duplicate name"):
            with zipfile.ZipFile(temporary, "w") as archive:
                archive.writestr("debug_pack_manifest.json", "{}")
                archive.writestr("debug_pack_manifest.json", '{"duplicate": true}')
        expected = ValueError

    with pytest.raises(expected):
        publish_verified_zip(
            temporary,
            destination,
            required_members=("debug_pack_manifest.json",),
        )

    assert not destination.exists()
    assert not destination.with_name(destination.name + ".manifest.json").exists()
    assert not temporary.exists()


def _make_probe_run(tmp_path: Path, *, include_raw_parquet: bool) -> tuple[Path, str, str]:
    run = tmp_path / ("run_raw" if include_raw_parquet else "run_lean")
    probe = run / "reports" / "window_method_validation_probe"
    measurement = probe / "resnet50" / "measurement" / "run_000"
    storage = measurement / "collector_storage"
    storage.mkdir(parents=True)
    (run / "profile.yaml").write_text(
        "native_producers:\n"
        "  energy:\n"
        "    window_method_validation_probe:\n"
        f"      include_raw_parquet: {'true' if include_raw_parquet else 'false'}\n",
        encoding="utf-8",
    )
    (run / "run_manifest.json").write_text(
        json.dumps({
            "schema": "onnx-splitpoint/evaluation-run-manifest",
            "schema_version": 1,
            "run_id": run.name,
            "status": "partial",
        }), encoding="utf-8"
    )
    # Variant-coordinator shape: the resolved values are materialised directly
    # on the top-level stage block (not below ``resolved_config``).  The profile
    # above must still take precedence when it explicitly resolves ``false``.
    (run / "reports" / "native_producer_stage.json").write_text(
        json.dumps(
            {
                "schema": "onnx-splitpoint/native-producer-variant-stage",
                "window_method_validation_probe": {
                    "enabled": True,
                    "include_raw_parquet": True,
                    "requested_repeats": 3,
                    "status": "ok",
                },
            }
        ),
        encoding="utf-8",
    )
    probe_report = json.dumps(
        {"schema": "probe", "padding": "x" * 200}, sort_keys=True
    )
    comparison = json.dumps(
        {"method_a": "duration", "method_b": "edge", "padding": "y" * 200},
        sort_keys=True,
    )
    (probe / "window_method_validation_probe.json").write_text(
        probe_report, encoding="utf-8"
    )
    (measurement / "window_method_comparison.json").write_text(
        comparison, encoding="utf-8"
    )
    (measurement / "energy_command.sh").write_text(
        "#!/bin/sh\nexec native-run --exact-image image.jpg\n", encoding="utf-8"
    )
    (storage / "fast_firmware.parquet").write_bytes(b"PAR1" + b"raw-trace" * 40)
    return run, comparison, probe_report


@pytest.mark.parametrize("include_raw", [False, True])
def test_debug_pack_probe_policy_is_narrow_hashed_and_a_b_is_always_present(
    tmp_path: Path, include_raw: bool
) -> None:
    run, comparison, _ = _make_probe_run(tmp_path, include_raw_parquet=include_raw)
    output = tmp_path / f"cli_{include_raw}.zip"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/create_evaluation_debug_pack.py",
            "--eval-run-dir",
            str(run),
            "--out",
            str(output),
            # Both A/B JSON files deliberately exceed this generic cap.
            "--max-small-file-bytes",
            "32",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)
    assert result["ok"] is True
    assert result["archive_verification"] == "verified"
    assert result["archive_sha256"] == _sha256(output)

    comparison_member = (
        "reports/window_method_validation_probe/resnet50/measurement/run_000/"
        "window_method_comparison.json"
    )
    parquet_member = comparison_member.replace(
        "window_method_comparison.json", "collector_storage/fast_firmware.parquet"
    )
    command_member = comparison_member.replace(
        "window_method_comparison.json", "energy_command.sh"
    )
    with zipfile.ZipFile(output) as archive:
        names = set(archive.namelist())
        assert comparison_member in names
        assert command_member in names
        assert (parquet_member in names) is include_raw
        assert archive.read(comparison_member).decode("utf-8") == comparison
        manifest = json.loads(archive.read("debug_pack_manifest.json"))

    probe = manifest["window_method_validation_probe"]
    assert probe["include_raw_parquet_resolved"] is include_raw
    assert probe["window_method_comparison_file_count"] == 1
    assert probe["command_file_count"] == 1
    assert probe["raw_parquet_count"] == int(include_raw)
    assert probe["all_probe_members_sha256_recorded"] is True
    assert manifest["tool_build"]["build_id"]
    comparison_record = next(
        row for row in manifest["files"] if row["path"] == comparison_member
    )
    assert comparison_record["diagnostic_kind"] == "window_method_A/B_comparison"
    assert comparison_record["sha256"].startswith("sha256:")

    external = json.loads(
        output.with_name(output.name + ".manifest.json").read_text(encoding="utf-8")
    )
    assert external["archive_size_bytes"] == output.stat().st_size
    assert external["archive_sha256"] == _sha256(output)
    assert external["tool_build"]["build_id"] == manifest["tool_build"]["build_id"]


def test_second_cli_debug_pack_uses_same_atomic_probe_contract(tmp_path: Path) -> None:
    run, _, _ = _make_probe_run(tmp_path, include_raw_parquet=True)
    epoch_file = run / "reports" / "pre_1980_timestamp.json"
    epoch_file.write_text('{"portable": true}', encoding="utf-8")
    os.utime(epoch_file, (1, 1))
    output_dir = tmp_path / "second_cli"
    output_dir.mkdir()
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/create_evalrun_pack.py",
            "--eval-run-dir",
            str(run),
            "--kind",
            "debug",
            "--out-dir",
            str(output_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)["artifacts"]["debug"]
    output = Path(result["out_zip"])
    assert result["archive_verification"] == "verified"
    assert result["archive_sha256"] == _sha256(output)
    with zipfile.ZipFile(output) as archive:
        manifest = json.loads(archive.read("debug_pack_manifest.json"))
        assert any(name.endswith("fast_firmware.parquet") for name in archive.namelist())
        assert archive.getinfo("reports/pre_1980_timestamp.json").date_time[0] == 1980
    assert manifest["window_method_validation_probe"]["raw_parquet_count"] == 1
    assert manifest["window_method_validation_probe"]["window_method_comparison_file_count"] == 1
    assert manifest["window_method_validation_probe"]["command_file_count"] == 1
    assert manifest["window_method_validation_probe"]["all_probe_members_sha256_recorded"] is True


def test_both_clis_recognize_direct_variant_stage_probe_shape(tmp_path: Path) -> None:
    run, _, _ = _make_probe_run(tmp_path, include_raw_parquet=False)
    # Exercise the stage fallback rather than the preferred materialized
    # profile.  The stage block intentionally has no ``resolved_config`` child.
    (run / "profile.yaml").unlink()

    first_output = tmp_path / "stage_shape_first.zip"
    subprocess.run(
        [
            sys.executable,
            "scripts/create_evaluation_debug_pack.py",
            "--eval-run-dir",
            str(run),
            "--out",
            str(first_output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    with zipfile.ZipFile(first_output) as archive:
        first_manifest = json.loads(archive.read("debug_pack_manifest.json"))
        assert any(name.endswith("fast_firmware.parquet") for name in archive.namelist())
    assert first_manifest["window_method_validation_probe"][
        "include_raw_parquet_resolved"
    ] is True

    second_output_dir = tmp_path / "stage_shape_second"
    second_output_dir.mkdir()
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/create_evalrun_pack.py",
            "--eval-run-dir",
            str(run),
            "--kind",
            "debug",
            "--out-dir",
            str(second_output_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    second_output = Path(
        json.loads(completed.stdout)["artifacts"]["debug"]["out_zip"]
    )
    with zipfile.ZipFile(second_output) as archive:
        second_manifest = json.loads(archive.read("debug_pack_manifest.json"))
        assert any(name.endswith("fast_firmware.parquet") for name in archive.namelist())
    assert second_manifest["window_method_validation_probe"][
        "include_raw_parquet_resolved"
    ] is True


def test_deepx_full_scientific_row_keeps_source_run_id_and_infers_task(
    tmp_path: Path,
) -> None:
    run = tmp_path / "evaluation_run_01"
    results = run / "models" / "resnet50" / "benchmark_results"
    results.mkdir(parents=True)
    (run / "run_manifest.json").write_text(
        json.dumps({"run_id": "campaign-run-01"}), encoding="utf-8"
    )
    (results / "normalized_results.json").write_text(
        json.dumps(
            {
                "results": [
                    {
                        "model_id": "resnet50",
                        "task": "",
                        "benchmark_task_used": "auto",
                        "backend": "deepx_m1",
                        "source_tag": "deepx_m1_full_run_007",
                        "run_id": None,
                        "variant": "full",
                        "buildable": True,
                        "runtime_executable": True,
                        "contract_consistent": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    profile = {
        "model_suite": {
            "primary": [
                {
                    "id": "resnet50",
                    "task": "classification",
                    "evaluation_role": "development",
                }
            ]
        }
    }
    rows, _, _ = _load_evalrun_rows(run, profile, AccuracyGatePolicy())
    assert len(rows) == 1
    assert rows[0]["run_id"] == "deepx_m1_full_run_007"
    assert rows[0]["task"] == "classification"
    exported = _scientific_row(rows[0])
    assert exported["run_id"] == "deepx_m1_full_run_007"
    assert exported["task"] == "classification"


def test_guardrail_failure_is_reported_when_primary_delta_is_zero() -> None:
    row = {
        "model_id": "resnet50",
        "task": "classification",
        "backend": "hailo10h",
        "variant": "full",
        "buildable": True,
        "runtime_executable": True,
        "contract_consistent": True,
        "task_quality_gate": {
            "status": "fail",
            "decision": "fail",
            "tier": "screening",
            "primary": {
                "metric": "top1_accuracy",
                "candidate": 0.80,
                "reference": 0.80,
                "delta": 0.0,
                "ci_low": 0.0,
                "ci_high": 0.0,
                "margin": 0.01,
                "decision": "pass",
                "bootstrap_skipped_reason": "candidate_identical_to_reference",
            },
            "guardrails": {
                "top5_accuracy": {
                    "metric": "top5_accuracy",
                    "candidate": 0.80,
                    "reference": 0.95,
                    "delta": -0.15,
                    "ci_low": -0.15,
                    "ci_high": -0.15,
                    "margin": 0.01,
                    "decision": "fail",
                    "bootstrap_skipped_reason": (
                        "point_estimate_below_non_inferiority_margin"
                    ),
                }
            },
        },
    }
    apply_accuracy_gate_to_row(row, AccuracyGatePolicy())

    assert row["accuracy_gate_decision"] == "fail"
    assert row["accuracy_gate_delta"] == 0.0
    assert row["ranking_eligible"] is False
    assert row["accuracy_gate_reason"] == "quality_failed"
    assert row["accuracy_gate_trigger_reasons"] == [
        "guardrail:top5_accuracy:top5_accuracy:point_estimate_below_non_inferiority_margin"
    ]
    exported = _scientific_row(row)
    assert "guardrail:top5_accuracy" in exported["task_quality_gate_reason"]


def test_tex_math_headers_and_frozen_selector_caption_are_valid(tmp_path: Path) -> None:
    caption = (
        "Comparison of the frozen Cut Bytes workflow selector with four "
        "pre-registered alternative ranking baselines."
    )
    output = _write_tex_table(
        tmp_path / "ranking.tex",
        [{"method": "selector_a", "rho": 0.5, "tau": 0.4, "mape": 3.0}],
        [
            ("method", "Method", "text"),
            ("mape", "MAPE [percent]", "number"),
            ("rho", "$\\rho$", "number"),
            ("tau", "$\\tau_b$", "number"),
        ],
        caption,
        "tab:ranking-method-comparison",
    )
    text = output.read_text(encoding="utf-8")
    assert "$\\rho$" in text
    assert "$\\tau_b$" in text
    assert r"\textbackslash{}rho" not in text
    assert "MAPE [percent]" in text
    assert "frozen Cut Bytes workflow selector" in text
    assert "four pre-registered alternative" in text
    assert _tex_escape("a_b%") == r"a\_b\%"


def test_standalone_reporter_template_compiles_and_names_frozen_selector() -> None:
    template = Path(
        "onnx_splitpoint_tool/resources/templates/scientific_reporter_v60.py.txt"
    ).read_text(encoding="utf-8")
    compile(template, "scientific_reporter_v60.py", "exec")
    assert "frozen Cut Bytes workflow selector with four pre-registered alternative ranking baselines" in template
    assert "five pre-registered" not in template.lower()
    assert '"$\\\\rho$"' in template
    assert '"$\\\\tau_b$"' in template


def test_gui_delegates_to_shared_compact_policy_without_lean_bundle_mirror() -> None:
    source = Path("onnx_splitpoint_tool/gui/app.py").read_text(encoding="utf-8")
    builder = Path(
        "onnx_splitpoint_tool/workflow/debug_pack.py"
    ).read_text(encoding="utf-8")
    assert "from ..workflow.debug_pack import create_evaluation_debug_pack" in source
    assert "result = create_evaluation_debug_pack(" in source
    assert '"lean_bundle"' in builder
    assert "blocked duplicate/data directory" in builder
