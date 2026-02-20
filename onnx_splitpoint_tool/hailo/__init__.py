"""Hailo helper subpackage.

This subpackage contains optional helpers to manage multiple Hailo Dataflow
Compiler (DFC) versions (e.g. Hailo-8 vs Hailo-10) without hard-coding the
user's environment.
"""

from .dfc_manager import DfcManager, DfcProfile, ResolvedDfcRuntime, get_dfc_manager

__all__ = [
    "DfcManager",
    "DfcProfile",
    "ResolvedDfcRuntime",
    "get_dfc_manager",
]
