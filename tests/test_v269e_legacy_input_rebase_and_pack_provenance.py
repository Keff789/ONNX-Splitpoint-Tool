from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

from onnx_splitpoint_tool.workflow.analysis_pack import create_analysis_pack


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_PROVENANCE_FILES = {
    "profile_source.yaml",
    "profile_start_snapshot.json",
    "effective_execution_plan.json",
}
ENERGY_PROVENANCE_FILES = {
    "reports/native_energy_model_hash_map.json": (
        "99_provenance/reports/native_energy_model_hash_map.json"
    ),
    "reports/native_energy_measurements/plan/native_producer_energy_plan.json": (
        "99_provenance/reports/native_energy_measurements/plan/"
        "native_producer_energy_plan.json"
    ),
    "reports/native_energy_measurements/plan/native_producer_energy_plan.md": (
        "99_provenance/reports/native_energy_measurements/plan/"
        "native_producer_energy_plan.md"
    ),
}


def _load_script(name: str, relative: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("legacy_key", ("input_dump", "preprocessed_input_file"))
def test_legacy_input_dump_rebases_stale_remote_absolute_path_next_to_manifest(
    tmp_path: Path, legacy_key: str,
) -> None:
    probe = _load_script(
        f"v269e_legacy_input_rebase_{legacy_key}",
        "scripts/native_yolo_full_self_reference_probe.py",
    )
    manifest_path = tmp_path / "native_fifo_boundary_manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    selected = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    local_dump = tmp_path / "legacy_input_rgb_uint8.bin"
    local_dump.write_bytes(selected.tobytes())
    manifest = {
        "schema_version": 2,
        "backend": "hailo8_to_trt",
        legacy_key: f"/home/remote/deleted-suite/{local_dump.name}",
        "input_shape_hwc": [2, 3, 3],
        "_manifest_path": str(manifest_path),
    }

    feed = probe._input_dump_feed_from_manifest(manifest, [1, 3, 2, 3], "raw")

    expected = np.transpose(selected.astype(np.float32), (2, 0, 1))[None]
    assert feed is not None
    assert np.array_equal(feed, expected)


def test_legacy_rebase_is_basename_local_and_selected_input_hash_stays_mandatory(
    tmp_path: Path,
) -> None:
    probe = _load_script(
        "v269e_legacy_input_rebase_scope",
        "scripts/native_yolo_full_self_reference_probe.py",
    )
    manifest_path = tmp_path / "native_fifo_boundary_manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    selected = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)

    # The legacy fallback may use only the stale path's basename beside the
    # extracted manifest; it must not search for an unrelated local dump.
    (tmp_path / "different.bin").write_bytes(selected.tobytes())
    assert probe._input_dump_feed_from_manifest(
        {
            "schema_version": 2,
            "input_dump": "/home/remote/deleted-suite/expected.bin",
            "input_shape_hwc": [2, 3, 3],
            "_manifest_path": str(manifest_path),
        },
        [1, 3, 2, 3],
        "raw",
    ) is None

    # selected_input_dump keeps its existing exact-evidence contract even
    # when its stale absolute path can be rebased beside the manifest.
    selected_path = tmp_path / "selected_input.npy"
    np.save(selected_path, selected, allow_pickle=False)
    exact_manifest = {
        "schema_version": 3,
        "backend": "deepx_to_trt",
        "selected_input_dump": f"/home/remote/deleted-suite/{selected_path.name}",
        "selected_input_dump_sha256": "0" * 64,
        "selected_input_shape": [2, 3, 3],
        "selected_input_dtype": "uint8",
        "_manifest_path": str(manifest_path),
    }
    with pytest.raises(RuntimeError, match="exact_selected_input_dump_sha256_mismatch"):
        probe._input_dump_feed_from_manifest(exact_manifest, [1, 3, 2, 3], "raw")

    exact_manifest["selected_input_dump_sha256"] = hashlib.sha256(
        selected_path.read_bytes()
    ).hexdigest()
    assert probe._input_dump_feed_from_manifest(
        exact_manifest, [1, 3, 2, 3], "raw",
    ) is not None


def test_debug_and_analysis_packs_archive_start_snapshot_provenance(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    scientific = run / "reports" / "scientific"
    scientific.mkdir(parents=True)
    (scientific / "scientific_report.json").write_text(
        json.dumps({"schema": "onnx-splitpoint/scientific-report"}),
        encoding="utf-8",
    )
    (run / "run_manifest.json").write_text(
        json.dumps({
            "schema": "onnx-splitpoint/evaluation-run-manifest",
            "schema_version": 1,
            "run_id": "run",
            "status": "partial",
        }),
        encoding="utf-8",
    )
    (run / "profile.yaml").write_text(
        "model_suite:\n  primary: []\n",
        encoding="utf-8",
    )
    expected = {
        "profile_source.yaml": b"profile_id: source-profile\n",
        "profile_start_snapshot.json": b'{"snapshot": "immutable"}\n',
        "effective_execution_plan.json": b'{"expected_generic_result_rows_total": 48}\n',
    }
    for name, payload in expected.items():
        (run / name).write_bytes(payload)
    energy_expected = {
        source: f"energy provenance: {source}\n".encode("utf-8")
        for source in ENERGY_PROVENANCE_FILES
    }
    for source, payload in energy_expected.items():
        path = run / source
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

    debug_zip = tmp_path / "debug.zip"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "create_evaluation_debug_pack.py"),
            "--eval-run-dir",
            str(run),
            "--out",
            str(debug_zip),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    analysis_zip = tmp_path / "analysis.zip"
    create_analysis_pack(
        run, analysis_zip, tool_version="2.69.5-test",
        materialize_missing_report=True,
    )

    with zipfile.ZipFile(debug_zip) as archive:
        for name, payload in expected.items():
            assert archive.read(name) == payload
        for source, payload in energy_expected.items():
            assert archive.read(source) == payload
    with zipfile.ZipFile(analysis_zip) as archive:
        for name, payload in expected.items():
            assert archive.read(f"99_provenance/{name}") == payload
        for source, payload in energy_expected.items():
            assert archive.read(ENERGY_PROVENANCE_FILES[source]) == payload

    # Keep every supported debug-pack entry point aligned with the GUI pack.
    first_cli = _load_script(
        "v269e_debug_pack_first_cli", "scripts/create_evaluation_debug_pack.py",
    )
    second_cli = _load_script(
        "v269e_debug_pack_second_cli", "scripts/create_evalrun_pack.py",
    )
    assert SNAPSHOT_PROVENANCE_FILES <= set(first_cli.INCLUDE)
    assert SNAPSHOT_PROVENANCE_FILES <= set(second_cli.INCLUDE_ROOT_FILES)
    assert set(ENERGY_PROVENANCE_FILES) <= set(first_cli.INCLUDE)

    gui_source = (ROOT / "onnx_splitpoint_tool" / "gui" / "app.py").read_text(
        encoding="utf-8",
    )
    method_start = gui_source.index("    def _evaluation_workflow_prepare_debug_pack")
    method_end = gui_source.index("\n    def ", method_start + 1)
    gui_debug_method = gui_source[method_start:method_end]
    assert "from ..workflow.debug_pack import create_evaluation_debug_pack" in gui_debug_method
    assert "result = create_evaluation_debug_pack(" in gui_debug_method


def test_remote_self_reference_probe_mirror_matches_primary() -> None:
    assert (
        ROOT / "scripts" / "native_yolo_full_self_reference_probe.py"
    ).read_bytes() == (
        ROOT
        / "onnx_splitpoint_tool"
        / "resources"
        / "remote_scripts"
        / "native_yolo_full_self_reference_probe.py"
    ).read_bytes()
