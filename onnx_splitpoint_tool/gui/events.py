"""Simple GUI event registry for decoupled panel communication."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional


class GuiEvents:
    """Small callback-based event hub used by the Tk GUI."""

    def __init__(self) -> None:
        self._listeners: Dict[str, List[Callable[..., None]]] = {
            "model_loaded": [],
            "analysis_done": [],
            "candidate_selected": [],
            "settings_changed": [],
        }

    def on_model_loaded(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        self._listeners["model_loaded"].append(callback)

    def on_analysis_done(self, callback: Callable[[Any], None]) -> None:
        self._listeners["analysis_done"].append(callback)

    def on_candidate_selected(self, callback: Callable[[Optional[Any]], None]) -> None:
        self._listeners["candidate_selected"].append(callback)

    def on_settings_changed(self, callback: Callable[[], None]) -> None:
        self._listeners["settings_changed"].append(callback)

    def emit_model_loaded(self, model_info: Dict[str, Any]) -> None:
        for callback in self._listeners["model_loaded"]:
            callback(model_info)

    def emit_analysis_done(self, analysis_result: Any) -> None:
        for callback in self._listeners["analysis_done"]:
            callback(analysis_result)

    def emit_candidate_selected(self, candidate: Optional[Any]) -> None:
        for callback in self._listeners["candidate_selected"]:
            callback(candidate)

    def emit_settings_changed(self) -> None:
        for callback in self._listeners["settings_changed"]:
            callback()


__all__ = ["GuiEvents"]
