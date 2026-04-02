from __future__ import annotations

from pathlib import Path
import py_compile


def test_benchmark_suite_template_supports_resume_and_reuses_existing_case_results() -> None:
    src = Path('onnx_splitpoint_tool/resources/templates/benchmark_suite.py.txt').read_text(encoding='utf-8')
    assert 'parser.add_argument("--resume"' in src
    assert 'using existing results from {out_dir_name}/validation_report.json' in src
    assert 'def _collect_case_result(' in src
    assert 'resume=getattr(args, "resume", False)' in src


def test_remote_run_source_auto_reuses_previous_partial_run_and_passes_resume_flag() -> None:
    src = Path('onnx_splitpoint_tool/benchmark/remote_run.py').read_text(encoding='utf-8')
    assert 'def _find_resumable_local_run(' in src
    assert '[resume] Reusing previous partial run:' in src
    assert 'bench_cmd += " --resume"' in src
    assert 'Partial results were collected; rerun the remote benchmark to resume the same run.' in src


def test_ssh_transport_streaming_returns_stable_timeout_and_cancel_codes() -> None:
    src = Path('onnx_splitpoint_tool/remote/ssh_transport.py').read_text(encoding='utf-8')
    assert 'if cancelled:' in src
    assert 'return 130' in src
    assert 'if timed_out:' in src
    assert 'return 124' in src


def test_remote_resume_related_python_sources_compile() -> None:
    py_compile.compile('onnx_splitpoint_tool/benchmark/remote_run.py', doraise=True)
    py_compile.compile('onnx_splitpoint_tool/remote/ssh_transport.py', doraise=True)
    py_compile.compile('onnx_splitpoint_tool/gui/app.py', doraise=True)
