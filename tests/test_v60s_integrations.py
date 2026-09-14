from __future__ import annotations
from pathlib import Path
import yaml
from onnx_splitpoint_tool import __version__
from onnx_splitpoint_tool.run_modes import RUN_MODE_SCHEMA_VERSION, default_run_modes_config, resolve_run_mode

ROOT=Path(__file__).resolve().parents[1]

def test_versions_and_run_mode_fields():
    assert __version__ in {'0.14.20+v60u.nativecontractfix','0.14.21+v60v.generationnativefix','0.14.22+v60w.smokedeferralpackfix', '0.14.23+v60x.nativeevidencefix', '0.14.25+v60z.nativefullquality', '0.14.26+v61a.nativefullenergyprogress', '0.14.27+v61b.nativeintegrationfix', '0.14.28+v61c.nativefullpairedenergyfix', '0.14.29+v61d.nativefullsemanticfix', '0.14.30+v61e.standardguifix', '2.61.0+v61e', '2.62.0', '2.63.0', '2.64.0', '2.65.0', '2.66.0', '2.67.0', '2.68.0', '2.69.6', "2.70.6", "2.70.7", "2.70.8", "2.70.9", "2.70.10", "2.71.1", "2.71.2", "2.71.3", "2.71.4", "2.72.0", "2.72.1", "2.72.2", "2.72.3", "2.72.4", "2.72.5", "2.72.6", "2.72.7", "2.73.0", "2.73.6", "2.73.7", "2.75.2", "2.75.3", "2.75.7", "2.75.8", "2.75.9", "2.75.10", "2.75.11", "2.75.12", "2.75.13", "2.75.14", "2.75.15", "2.75.16", "2.75.17", "2.75.20", "2.75.21", "2.75.22", "2.75.24", "2.75.26", "2.75.27", "2.75.28", "2.75.30", "2.75.31", "2.75.32", "2.75.38", "2.75.39", "2.75.40", "2.75.41", "2.75.42", "2.75.45", "2.75.46", "2.75.47"}
    assert RUN_MODE_SCHEMA_VERSION in {5, 7, 8, 9, 10, 11, 12, 13}
    cfg=default_run_modes_config()
    for name in ('smoke','standard','final'):
        build=cfg['modes'][name]['build']
        assert build['artifact_store']['enabled'] is True
        assert build['scheduler']['enabled'] is True
        assert build['scheduler']['family_limits']['hailo8']==1


def test_default_yaml_and_entrypoints():
    payload=yaml.safe_load((ROOT/'onnx_splitpoint_tool/resources/run_modes/default_run_modes.yaml').read_text())
    assert payload['schema_version'] in {5, 7, 8, 9, 10, 11, 12, 13}
    pyproject=(ROOT/'pyproject.toml').read_text()
    assert 'onnx-splitpoint-artifacts' in pyproject
    assert 'onnx-splitpoint-smoke-v60s' in pyproject
    assert 'onnx-splitpoint-smoke-v60t' in pyproject
    assert 'onnx-splitpoint-smoke-v60u' in pyproject


def test_hailo_deepx_and_scheduler_hooks_present():
    hailo=(ROOT/'onnx_splitpoint_tool/hailo_backend.py').read_text()
    deepx=(ROOT/'onnx_splitpoint_tool/workflow/deepx_build_binding.py').read_text()
    services=(ROOT/'onnx_splitpoint_tool/benchmark/services.py').read_text()
    binding=(ROOT/'onnx_splitpoint_tool/workflow/legacy_benchmarkset_binding.py').read_text()
    assert 'def hailo_build_hef(*args, **kwargs)' in hailo
    assert 'ArtifactStore' in hailo
    assert 'def _find_cached_dxnn(*args, **kwargs)' in deepx
    assert '_run_hailo_target_builds_v60s' in services
    assert '_v60s_start_deepx_prefetch' in binding
    assert 'register_benchmark_set_artifacts' in binding


def test_artifact_library_panel_imports():
    from onnx_splitpoint_tool.gui.artifact_library_panel import ArtifactLibraryPanel
    assert ArtifactLibraryPanel
