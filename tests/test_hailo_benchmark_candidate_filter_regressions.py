from __future__ import annotations

from pathlib import Path


ROOT = Path("onnx_splitpoint_tool")


def test_hailo_target_defaults_to_part2_in_gui_sources() -> None:
    gui_src = (ROOT / "gui_app.py").read_text(encoding="utf-8")
    assert 'self.var_hailo_target = tk.StringVar(value="part2")' in gui_src

    panel_src = (ROOT / "gui" / "panels" / "panel_hardware.py").read_text(encoding="utf-8")
    assert 'hailo_target_var = _str_var(app, "var_hailo_target", "part2")' in panel_src



def test_benchmark_candidate_pool_source_prefers_selected_ranked_candidates() -> None:
    src = (ROOT / "gui_app.py").read_text(encoding="utf-8")
    start = src.index('    def _benchmark_candidate_pool(self) -> List[int]:')
    end = src.index('    def _analysis_predicted_metrics_for_boundary', start)
    block = src[start:end]
    assert 'sources.append(list(self.current_picks or []))' in block
    assert 'sources.append(list(self.analysis.get("candidate_bounds_selected") or []))' in block
    assert 'sources.append(list(getattr(self.analysis_result, "candidates", []) or []))' in block
    assert 'sources.append(list(self.analysis.get("candidate_bounds") or []))' in block



def test_hailo_backend_exposes_manifest_precheck_helpers() -> None:
    src = (ROOT / "hailo_backend.py").read_text(encoding="utf-8")
    assert 'def hailo_part2_activation_precheck_from_io' in src
    assert 'def hailo_part2_activation_precheck_from_manifest' in src
    assert 'def format_hailo_part2_activation_precheck_error' in src
    assert 'likely_original_inputs' in src
    assert 'Unsupported Hailo Part2 activation-calibration splitpoint' in src



def test_benchmark_export_source_uses_selected_candidates_and_early_hailo_part2_guard() -> None:
    gui_src = (ROOT / "gui_app.py").read_text(encoding="utf-8")
    svc_src = (ROOT / "benchmark" / "services.py").read_text(encoding="utf-8")
    assert 'a["candidate_bounds_selected"] = list(picks)' in gui_src
    assert 'candidate_pool: List[int] = list(self._benchmark_candidate_pool())' in gui_src
    assert 'hailo_part2_precheck_fn' in gui_src
    assert 'HEF(part2,{target_label}) SKIPPED' in svc_src


def test_hailo_backend_exposes_parser_blocker_precheck_helpers() -> None:
    src = (ROOT / 'hailo_backend.py').read_text(encoding='utf-8')
    assert 'def hailo_part2_parser_blocker_precheck_from_model' in src
    assert 'def format_hailo_part2_parser_blocker_error' in src
    assert 'Unsupported Hailo Part2 parser-blocking head detected' in src


def test_benchmark_export_source_uses_hailo_part2_parser_guard() -> None:
    gui_src = (ROOT / 'gui_app.py').read_text(encoding='utf-8')
    svc_src = (ROOT / 'benchmark' / 'services.py').read_text(encoding='utf-8')
    assert 'hailo_part2_parser_precheck_fn' in gui_src
    assert 'hailo_part2_parser_precheck_error_fn' in gui_src
    assert 'skip: Hailo part2 parser precheck' in svc_src
