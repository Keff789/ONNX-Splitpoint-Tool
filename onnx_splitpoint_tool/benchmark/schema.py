from __future__ import annotations

"""Schema/version helpers for benchmark artefacts.

This module centralises lightweight schema migration/stamping so GUI code and
analysis code do not each grow their own ad-hoc compatibility logic.
"""

import json
import os
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional

BENCHMARK_SET_SCHEMA = "onnx-splitpoint/benchmark-set"
BENCHMARK_SET_SCHEMA_VERSION = 2
BENCHMARK_ANALYSIS_VERSION = 2
RESULTS_BUNDLE_SCHEMA = "onnx-splitpoint/results-bundle"
RESULTS_BUNDLE_SCHEMA_VERSION = 1


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


_JSON_SAFE_REPR_LIMIT = 240


def _safe_repr(value: Any, *, limit: int = _JSON_SAFE_REPR_LIMIT) -> str:
    try:
        text = str(value)
    except Exception:
        try:
            text = repr(value)
        except Exception:
            text = f"<{type(value).__name__}>"
    text = text.replace("\r", " ").replace("\n", " ")
    if len(text) > limit:
        text = text[: max(0, limit - 3)] + "..."
    return text


def _looks_like_protobuf_message(value: Any) -> bool:
    return callable(getattr(value, "SerializeToString", None)) and callable(getattr(value, "ListFields", None))


def _protobuf_summary(value: Any) -> Dict[str, Any]:
    cls = value.__class__
    summary: Dict[str, Any] = {
        "__non_json__": f"{cls.__module__}.{cls.__name__}",
    }

    graph = getattr(value, "graph", None)
    if graph is not None:
        graph_name = getattr(graph, "name", None)
        if graph_name not in (None, ""):
            summary["graph_name"] = str(graph_name)
        for attr_name, out_name in (
            ("ir_version", "ir_version"),
            ("producer_name", "producer_name"),
            ("producer_version", "producer_version"),
            ("domain", "domain"),
            ("model_version", "model_version"),
        ):
            try:
                raw = getattr(value, attr_name, None)
            except Exception:
                raw = None
            if raw not in (None, ""):
                summary[out_name] = raw
        for attr_name, out_name in (
            ("node", "node_count"),
            ("input", "input_count"),
            ("output", "output_count"),
            ("initializer", "initializer_count"),
        ):
            try:
                summary[out_name] = int(len(getattr(graph, attr_name, []) or []))
            except Exception:
                pass
        try:
            opsets = []
            for imp in list(getattr(value, "opset_import", []) or []):
                try:
                    version = getattr(imp, "version", None)
                    version_val = None if version in (None, "") else int(version)
                except Exception:
                    version_val = None
                opsets.append({
                    "domain": str(getattr(imp, "domain", "") or ""),
                    "version": version_val,
                })
            if opsets:
                summary["opset_imports"] = opsets
        except Exception:
            pass
        return summary

    try:
        blob = value.SerializeToString()
    except Exception:
        blob = None
    if isinstance(blob, (bytes, bytearray)):
        summary["byte_size"] = int(len(blob))
    return summary


def make_json_safe(value: Any, *, _seen: Optional[set[int]] = None) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        try:
            return value.as_posix()
        except Exception:
            return str(value)
    if isinstance(value, (bytes, bytearray)):
        try:
            return bytes(value).decode("utf-8")
        except Exception:
            return {"__non_json__": type(value).__name__, "length": int(len(value))}

    if _seen is None:
        _seen = set()

    if isinstance(value, dict):
        obj_id = id(value)
        if obj_id in _seen:
            return {"__circular_ref__": type(value).__name__}
        _seen.add(obj_id)
        try:
            out: Dict[str, Any] = {}
            for key, item in value.items():
                if isinstance(key, str):
                    out_key = key
                else:
                    out_key = _safe_repr(key)
                out[out_key] = make_json_safe(item, _seen=_seen)
            return out
        finally:
            _seen.discard(obj_id)

    if isinstance(value, (list, tuple)):
        obj_id = id(value)
        if obj_id in _seen:
            return [{"__circular_ref__": type(value).__name__}]
        _seen.add(obj_id)
        try:
            return [make_json_safe(item, _seen=_seen) for item in value]
        finally:
            _seen.discard(obj_id)

    if isinstance(value, (set, frozenset)):
        obj_id = id(value)
        if obj_id in _seen:
            return [{"__circular_ref__": type(value).__name__}]
        _seen.add(obj_id)
        try:
            items = sorted(value, key=_safe_repr)
            return [make_json_safe(item, _seen=_seen) for item in items]
        finally:
            _seen.discard(obj_id)

    if is_dataclass(value):
        try:
            return make_json_safe(asdict(value), _seen=_seen)
        except Exception:
            pass

    item_fn = getattr(value, "item", None)
    if callable(item_fn):
        try:
            scalar = item_fn()
        except Exception:
            scalar = None
        else:
            if scalar is not value:
                return make_json_safe(scalar, _seen=_seen)

    tolist_fn = getattr(value, "tolist", None)
    if callable(tolist_fn):
        try:
            converted = tolist_fn()
        except Exception:
            converted = None
        else:
            if converted is not value:
                return make_json_safe(converted, _seen=_seen)

    if _looks_like_protobuf_message(value):
        return _protobuf_summary(value)

    return {
        "__non_json__": f"{type(value).__module__}.{type(value).__name__}",
        "repr": _safe_repr(value),
    }


def dumps_json_safe(value: Any, **kwargs: Any) -> str:
    return json.dumps(make_json_safe(value), **kwargs)


def _relative_paths(base_dir: Path, paths: Iterable[Path]) -> List[str]:
    out: List[str] = []
    for p in paths:
        try:
            out.append(Path(p).resolve().relative_to(base_dir.resolve()).as_posix())
        except Exception:
            try:
                out.append(Path(p).as_posix())
            except Exception:
                continue
    return sorted(dict.fromkeys(out))


def build_benchmark_artifact_manifest(base_dir: Path, *, extra_paths: Optional[Iterable[Path]] = None) -> Dict[str, Any]:
    """Return a compact artefact manifest for a benchmark suite folder."""

    base_dir = Path(base_dir).expanduser().resolve()
    files: Dict[str, List[str]] = {
        "root": [],
        "analysis_plots": [],
        "analysis_tables": [],
        "schemas": [],
        "models": [],
        "cases": [],
    }
    root_patterns = [
        'benchmark_set.json',
        'benchmark_plan.json',
        'benchmark_suite.py',
        'README_BENCHMARK.txt',
        'benchmark_generation.log',
        'generation_state.json',
    ]
    for pat in root_patterns:
        files['root'].extend(_relative_paths(base_dir, base_dir.glob(pat)))
    for sub in ('analysis_plots', 'analysis_tables', 'schemas', 'models'):
        subdir = base_dir / sub
        if subdir.exists():
            files[sub] = _relative_paths(base_dir, [p for p in subdir.rglob('*') if p.is_file()])
    case_dirs = [p for p in base_dir.glob('b*') if p.is_dir()]
    files['cases'] = sorted(p.name for p in case_dirs)
    if extra_paths:
        files['root'].extend(_relative_paths(base_dir, extra_paths))
        files['root'] = sorted(dict.fromkeys(files['root']))
    return {
        'schema': BENCHMARK_SET_SCHEMA,
        'schema_version': BENCHMARK_SET_SCHEMA_VERSION,
        'analysis_version': BENCHMARK_ANALYSIS_VERSION,
        'generated_at': now_iso(),
        'suite_dir': str(base_dir),
        'files': files,
        'counts': {
            'root': len(files['root']),
            'analysis_plots': len(files['analysis_plots']),
            'analysis_tables': len(files['analysis_tables']),
            'schemas': len(files['schemas']),
            'models': len(files['models']),
            'cases': len(files['cases']),
        },
    }


def stamp_benchmark_set_payload(
    payload: MutableMapping[str, Any],
    *,
    tool_gui_version: Optional[str] = None,
    tool_core_version: Optional[str] = None,
    analysis_version: int = BENCHMARK_ANALYSIS_VERSION,
    suite_dir: Optional[Path] = None,
    artifact_manifest: Optional[Mapping[str, Any]] = None,
    bundle_options: Optional[Mapping[str, Any]] = None,
) -> MutableMapping[str, Any]:
    payload.setdefault('schema', BENCHMARK_SET_SCHEMA)
    payload['schema_version'] = max(int(payload.get('schema_version') or 1), BENCHMARK_SET_SCHEMA_VERSION)
    payload['analysis_version'] = max(int(payload.get('analysis_version') or 1), int(analysis_version))
    payload.setdefault('created_at', now_iso())
    tool = payload.get('tool') if isinstance(payload.get('tool'), dict) else {}
    if tool_gui_version:
        tool['gui'] = str(tool_gui_version)
    if tool_core_version:
        tool['core'] = str(tool_core_version)
    if tool:
        payload['tool'] = tool
    if suite_dir is not None and artifact_manifest is None:
        try:
            artifact_manifest = build_benchmark_artifact_manifest(Path(suite_dir))
        except Exception:
            artifact_manifest = None
    if artifact_manifest is not None:
        payload['artifact_manifest'] = dict(artifact_manifest)
    if bundle_options is not None:
        payload['bundle_options'] = dict(bundle_options)
    return payload


def migrate_benchmark_set_payload(raw: Any) -> Dict[str, Any]:
    payload = dict(raw or {}) if isinstance(raw, Mapping) else {}
    schema = str(payload.get('schema') or BENCHMARK_SET_SCHEMA)
    payload['schema'] = schema or BENCHMARK_SET_SCHEMA
    version = int(payload.get('schema_version') or 1)
    if version < 1:
        version = 1
    if version < 2:
        payload.setdefault('analysis_version', 1)
        payload.setdefault('tool', payload.get('tool') if isinstance(payload.get('tool'), dict) else {})
        payload.setdefault('artifact_manifest', {
            'schema': BENCHMARK_SET_SCHEMA,
            'schema_version': 2,
            'analysis_version': payload.get('analysis_version') or 1,
            'generated_at': payload.get('created_at') or now_iso(),
            'suite_dir': None,
            'files': {},
            'counts': {},
        })
        payload.setdefault('bundle_options', {'results_bundle_modes': ['full', 'lean']})
        version = 2
    payload['schema_version'] = max(version, BENCHMARK_SET_SCHEMA_VERSION)
    payload.setdefault('analysis_version', BENCHMARK_ANALYSIS_VERSION)
    payload.setdefault('cases', [])
    payload.setdefault('summary', {})
    payload.setdefault('errors', [])
    payload.setdefault('discarded_cases', [])
    payload.setdefault('bundle_options', {'results_bundle_modes': ['full', 'lean']})
    return payload


def read_benchmark_set(path: Path) -> Dict[str, Any]:
    return migrate_benchmark_set_payload(json.loads(Path(path).read_text(encoding='utf-8')))


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(dumps_json_safe(dict(payload), indent=2), encoding='utf-8')
    os.replace(tmp, path)
