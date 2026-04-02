from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

from onnx_splitpoint_tool.benchmark.services import BenchmarkGenerationService


FORBIDDEN_IMPORT_PATTERNS = {
    'onnx_splitpoint_tool.gui.hailo_backend': 'Nested gui.hailo_backend import path is invalid after the benchmark workflow extraction.',
    'from .gui.controller import': 'benchmark_workflow.py must import controller from its local package, not from a nested gui.gui path.',
    'from .gui.hailo_backend import': 'Nested GUI modules must resolve the Hailo backend from the package root.',
    'from ..gui.hailo_backend import': 'Nested GUI modules must resolve the Hailo backend from the package root.',
}


def test_benchmark_workflow_module_is_directly_importable_for_smoke_tests() -> None:
    mod = importlib.import_module('onnx_splitpoint_tool.gui.benchmark_workflow')
    assert hasattr(mod, 'BenchmarkWorkflowController')
    assert hasattr(mod, 'resolve_hailo_benchmark_helpers')
    assert hasattr(mod, 'resolve_tool_core_version')



def test_resolve_tool_core_version_uses_package_root_api_import_path() -> None:
    mod = importlib.import_module('onnx_splitpoint_tool.gui.benchmark_workflow')
    seen: list[str] = []

    def fake_import(name: str):
        seen.append(name)
        assert name == 'onnx_splitpoint_tool.api'
        return SimpleNamespace(__version__='test-core-version')

    assert mod.resolve_tool_core_version(importer=fake_import) == 'test-core-version'
    assert seen == ['onnx_splitpoint_tool.api']



def test_resolve_hailo_benchmark_helpers_uses_package_root_backend_path() -> None:
    mod = importlib.import_module('onnx_splitpoint_tool.gui.benchmark_workflow')
    seen: list[str] = []
    sentinel_build = object()
    sentinel_precheck = object()
    sentinel_precheck_error = object()
    sentinel_parser_precheck = object()
    sentinel_parser_precheck_error = object()

    def fake_import(name: str):
        seen.append(name)
        assert name == 'onnx_splitpoint_tool.hailo_backend'
        return SimpleNamespace(
            hailo_build_hef_auto=sentinel_build,
            format_hailo_part2_activation_precheck_error=sentinel_precheck_error,
            format_hailo_part2_parser_blocker_error=sentinel_parser_precheck_error,
            hailo_part2_activation_precheck_from_manifest=sentinel_precheck,
            hailo_part2_parser_blocker_precheck_from_model=sentinel_parser_precheck,
        )

    resolved = mod.resolve_hailo_benchmark_helpers(need_build=True, need_part2=True, importer=fake_import)

    assert resolved.hailo_build_hef_fn is sentinel_build
    assert resolved.hailo_part2_precheck_fn is sentinel_precheck
    assert resolved.hailo_part2_precheck_error_fn is sentinel_precheck_error
    assert resolved.hailo_part2_parser_precheck_fn is sentinel_parser_precheck
    assert resolved.hailo_part2_parser_precheck_error_fn is sentinel_parser_precheck_error
    assert resolved.hailo_build_unavailable is None
    assert resolved.hailo_part2_import_error is None
    assert seen == ['onnx_splitpoint_tool.hailo_backend']



def test_resolve_hailo_benchmark_helpers_reports_import_failures_without_wrong_nested_path() -> None:
    mod = importlib.import_module('onnx_splitpoint_tool.gui.benchmark_workflow')
    seen: list[str] = []

    def fake_import(name: str):
        seen.append(name)
        raise ModuleNotFoundError("No module named 'simulated_hailo_backend'")

    resolved = mod.resolve_hailo_benchmark_helpers(need_build=True, need_part2=True, importer=fake_import)

    assert resolved.hailo_build_hef_fn is None
    assert resolved.hailo_build_unavailable == "Hailo HEF build unavailable: No module named 'simulated_hailo_backend'"
    assert resolved.hailo_part2_import_error == "ModuleNotFoundError: No module named 'simulated_hailo_backend'"
    assert seen == ['onnx_splitpoint_tool.hailo_backend']



def test_resume_runtime_does_not_reemit_historical_errors_as_current_warnings(tmp_path: Path) -> None:
    out_dir = tmp_path / 'suite'
    bench_log = out_dir / 'benchmark_generation.log'
    service = BenchmarkGenerationService()
    runtime = service.start_generation_runtime(
        out_dir=out_dir,
        bench_log_path=bench_log,
        requested_cases=10,
        ranked_candidates=[10, 20, 30],
        candidate_search_pool=[10, 20, 30],
        hef_full_policy='end',
        model_name='demo-model',
        model_source='/tmp/demo-model.onnx',
        resume_generation=True,
        resume_state_hint={
            'created_at': '2026-03-27T08:00:00',
            'errors': ["Hailo HEF build unavailable: No module named 'onnx_splitpoint_tool.gui.hailo_backend'"],
            'case_entries': [],
            'discarded_case_entries': [],
        },
    )
    try:
        assert runtime.errors == []
        assert runtime.resumed_previous_errors == [
            "Hailo HEF build unavailable: No module named 'onnx_splitpoint_tool.gui.hailo_backend'"
        ]
        state = json.loads(runtime.state_path.read_text(encoding='utf-8'))
        assert state.get('errors') == []
    finally:
        runtime.close()



def test_no_known_bad_refactor_import_patterns_exist_anywhere_in_package() -> None:
    hits: list[str] = []
    for path in Path('onnx_splitpoint_tool').rglob('*.py'):
        text = path.read_text(encoding='utf-8')
        for pattern, message in FORBIDDEN_IMPORT_PATTERNS.items():
            if pattern in text:
                hits.append(f'{path}: {pattern} :: {message}')
    assert not hits, '\n'.join(hits)
