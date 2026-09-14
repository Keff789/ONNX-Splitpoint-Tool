#!/usr/bin/env python3
"""Inspect real release-test dependencies without installing any packages."""
from __future__ import annotations
import importlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
import sys

os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")
REQUIRED = ("pytest", "numpy", "onnx", "onnxruntime", "pycocotools.coco",
            "yaml", "PIL.Image", "jsonschema", "tkinter", "matplotlib.backends.backend_tkagg")


def module_version(name):
    distribution = {"yaml": "PyYAML", "PIL.Image": "Pillow", "pycocotools.coco": "pycocotools",
                    "matplotlib.backends.backend_tkagg": "matplotlib"}.get(name, name)
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "stdlib" if name == "tkinter" else "unknown"


def main():
    observations = []
    for name in REQUIRED:
        try:
            module = importlib.import_module(name)
            if name == "onnxruntime":
                module.disable_telemetry_events()
            observations.append({"module":name,"status":"available",
                "path":str(getattr(module,"__file__","")),"version":module_version(name)})
        except Exception as exc:
            observations.append({"module":name,"status":"missing_or_unloadable",
                                 "error":f"{type(exc).__name__}: {exc}"})
    blocked = any(row["status"] != "available" for row in observations)
    print(json.dumps({"status":"environment_blocked" if blocked else "PASS",
        "python":sys.executable,"dependencies":observations,
        "packages_installed":False,"hardware_executed":False},indent=2))
    return 78 if blocked else 0


if __name__ == "__main__":
    raise SystemExit(main())
