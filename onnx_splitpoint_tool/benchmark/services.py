from __future__ import annotations

"""Service-layer helpers for benchmark workflows.

These wrappers keep GUI code thinner and make the main operations reusable from
other entry points (tests, future CLI commands, notebooks).
"""

from dataclasses import dataclass, field, replace
from datetime import datetime
import json
import logging
import math
import os
import re
import shutil
import threading
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, IO, List, Mapping, Optional, Sequence, Set, Tuple

from .analysis import (
    BenchmarkAnalysisReport,
    BenchmarkComparisonReport,
    export_benchmark_analysis,
    export_benchmark_comparison,
    load_benchmark_analysis,
    load_benchmark_analysis_comparison,
)
from .hailo_policy import (
    HailoFailureRecord,
    analyze_cut_tensors,
    build_candidate_policy_index,
    build_case_hailo_variant_availability,
    case_has_usable_hailo_variant,
    classify_hailo_build_failure,
    run_variant_hailo_requirements,
    should_skip_from_failure_cluster,
)
from .interleaving_analysis import (
    InterleavingAnalysisReport,
    InterleavingComparisonReport,
    compare_interleaving_reports,
    compute_interleaving_analysis,
    export_metric_audit_comparison,
    export_interleaving_analysis,
    export_interleaving_comparison,
    export_publication_analysis as export_publication_analysis_bundle,
    export_publication_comparison as export_publication_comparison_bundle,
)
from .generation_state import init_state as init_generation_state, update_state as update_generation_state
from .decision_summary import export_decision_summary
from .results_bundle import create_results_bundle_from_suite
from .schema import migrate_benchmark_set_payload, read_benchmark_set, stamp_benchmark_set_payload, write_json_atomic as write_benchmark_json_atomic
from .hailo_scoring import heuristic_for_boundary, rerank_candidates_for_hailo
from .part2_sanity import hailo_part2_concat_sanity_from_model, format_hailo_part2_concat_sanity_error
from .remote_run import RemoteBenchmarkArgs, run_remote_benchmark
from ..benchmark_case_utils import archive_benchmark_case, build_benchmark_case_rejection
from ..remote.ssh_transport import HostConfig as SSHHostConfig, SSHTransport
from .suite_refresh import (
    refresh_suite_harness,
    normalize_benchmark_task,
    normalize_mini_classification_eval,
    normalize_mini_coco_ap50,
    normalize_semantic_validation_request,
    normalize_validation_reference_mode,
)
from .classification_validation_presets import (
    default_available_classification_validation_preset,
    provision_classification_validation_source_to_suite,
)
from .validation_assets import (
    default_detection_validation_source,
    provision_detection_validation_source_to_suite,
)



logger = logging.getLogger(__name__)


def _physical_hailo_hw_arch_for_build(value: Any) -> Optional[str]:
    """Normalize UI/run labels to a physical Hailo DFC architecture.

    Mixed run ids such as ``hailo8_to_trt`` and ``trt_to_hailo8`` are
    benchmark-pipeline labels, not DFC hw_arch values.  DFC must only be
    called with chip names such as ``hailo8``.
    """
    text = str(value or '').strip().lower().replace(' ', '').replace('-', '_')
    if not text:
        return None
    if text in {'hailo8', 'hailo8r', 'hailo8l'}:
        return text
    if 'hailo8r' in text:
        return 'hailo8r'
    if 'hailo8l' in text:
        return 'hailo8l'
    if 'hailo8' in text:
        return 'hailo8'
    if text in {'hailo10', 'hailo10h'}:
        return text
    if 'hailo10h' in text:
        return 'hailo10h'
    if 'hailo10' in text:
        return 'hailo10'
    return None


def _physical_hailo_targets_for_build(values: Sequence[Any]) -> List[str]:
    out: List[str] = []
    for raw in list(values or []):
        hw = _physical_hailo_hw_arch_for_build(raw)
        if hw and hw not in out:
            out.append(hw)
    return out


def _run_stage_kind(run: Mapping[str, Any], key: str) -> str:
    st = run.get(key) if isinstance(run, Mapping) else None
    if isinstance(st, Mapping):
        return str(st.get('type') or st.get('provider') or st.get('id') or '').strip().lower()
    return str(st or '').strip().lower()


def _benchmark_plan_has_hailo_to_trt(runs: Sequence[Mapping[str, Any]]) -> bool:
    for run in list(runs or []):
        if not isinstance(run, Mapping):
            continue
        rid = str(run.get('id') or run.get('name') or '').strip().lower()
        st1 = _run_stage_kind(run, 'stage1')
        st2 = _run_stage_kind(run, 'stage2')
        if ('hailo' in st1 and ('trt' in st2 or 'tensorrt' in st2)) or 'hailo8_to_trt' in rid or 'hailo_to_trt' in rid:
            return True
    return False


def _benchmark_plan_task_set(bench_plan_runs: Sequence[Mapping[str, Any]]) -> Set[str]:
    out: Set[str] = set()
    for run in bench_plan_runs or []:
        try:
            out.add(normalize_benchmark_task((run or {}).get('benchmark_task')))
        except Exception:
            pass
    return out or {'auto'}


def _model_task_sanity_warnings(model_path: str | Path, bench_plan_runs: Sequence[Mapping[str, Any]]) -> List[str]:
    """Cheap ONNX-task sanity hints for benchmark-set generation.

    These are warnings only. They are intentionally conservative and are mainly
    meant to catch cases where a model named like a classifier actually contains
    sequence/decoder ops or has a non-ImageNet-like input signature.
    """
    tasks = _benchmark_plan_task_set(bench_plan_runs)
    if 'classification' not in tasks:
        return []
    warnings: List[str] = []
    try:
        import onnx  # type: ignore
        m = onnx.load(str(model_path))
        nodes = list(getattr(m.graph, 'node', []) or [])
        op_types = {str(getattr(n, 'op_type', '') or '') for n in nodes}
        recurrent = sorted(op for op in op_types if op in {'LSTM', 'GRU', 'RNN', 'Scan'})
        names_blob = ' '.join(str(getattr(n, 'name', '') or '') + ' ' + ' '.join(str(x) for x in list(getattr(n, 'output', []) or [])[:2]) for n in nodes[:3000]).lower()
        if recurrent:
            warnings.append(
                'Classification task selected, but ONNX graph contains recurrent/sequence op(s) ' +
                ', '.join(recurrent) +
                '. This often indicates the model is not a plain ImageNet classifier; Hailo full/part2 parser preflight may fail.'
            )
        if '/decoder' in names_blob or 'decoder' in names_blob:
            warnings.append(
                'Classification task selected, but graph names contain "decoder". Verify that this ONNX is really a pure classification network and not an OCR/sequence model.'
            )
        try:
            dims = []
            for inp in list(getattr(m.graph, 'input', []) or [])[:1]:
                for d in inp.type.tensor_type.shape.dim:
                    val = int(d.dim_value) if getattr(d, 'dim_value', 0) else None
                    dims.append(val)
            fixed_hw = [int(x) for x in dims if isinstance(x, int) and int(x) > 1]
            # Common ImageNet classifier shapes expose 224/256/299-ish dimensions.
            if fixed_hw and not any(192 <= int(x) <= 384 for x in fixed_hw):
                warnings.append(
                    f'Classification task selected, but first input fixed dimensions look non-ImageNet-like: {dims}. ' +
                    'Check preprocessing/image_scale and whether the selected ONNX is the intended classifier.'
                )
        except Exception:
            pass
    except Exception as exc:
        warnings.append(f'Could not inspect ONNX model for classification sanity: {type(exc).__name__}: {exc}')
    return warnings


def _embedded_semantic_validation_dataset_source() -> Optional[Path]:
    """Return a prepared/default detection validation source if available.

    Clean releases do not ship the image-heavy COCO-50 directory inside the
    package. Users can prepare it on demand via the GUI/CLI; development builds
    may still expose a legacy embedded source.
    """

    return default_detection_validation_source()


def _provision_embedded_semantic_validation_dataset(suite_dir: Path) -> Optional[str]:
    """Provision the default detection validation set into a suite.

    Historically this copied an prepared COCO-50 directory. In clean releases it
    copies the user-prepared COCO-50 dataset from
    ``~/.onnx_splitpoint_tool/validation_datasets``.
    """

    return provision_detection_validation_source_to_suite(Path(suite_dir))


def _safe_int(value: Any) -> Optional[int]:
    try:
        if value is None or value == '':
            return None
        return int(value)
    except Exception:
        return None


def normalize_full_hef_policy(value: Any) -> str:
    """Normalize GUI/raw full-model HEF policy values to backend tokens.

    Accepted inputs are both the raw backend values (``start``/``end``/``skip``)
    and the human-readable GUI labels such as ``Build at end (recommended)``.
    Unknown values intentionally fall back to ``end``.
    """

    s = str(value or '').strip().lower()
    if not s:
        return 'end'
    if s == 'start' or s.startswith('build at start'):
        return 'start'
    if s == 'skip' or s.startswith('skip'):
        return 'skip'
    if s == 'end' or s.startswith('build at end'):
        return 'end'
    return 'end'


def normalize_hailo_full_model_preflight_policy(value: Any) -> str:
    """Normalize full-model Hailo parser preflight policy values.

    ``enabled`` keeps the current plan-aware parser preflight. ``skip`` disables
    the suite-level parser preflight so the generator still attempts the actual
    full-model HEF build later.
    """

    if isinstance(value, bool):
        return 'enabled' if value else 'skip'
    s = str(value or '').strip().lower()
    if not s:
        return 'enabled'
    if s in {'enabled', 'enable', 'on', 'true', 'yes', 'check'}:
        return 'enabled'
    if s in {'skip', 'disabled', 'disable', 'off', 'false', 'no'}:
        return 'skip'
    if s.startswith('enabled'):
        return 'enabled'
    if s.startswith('disabled'):
        return 'skip'
    if 'always try' in s or 'without preflight' in s:
        return 'skip'
    return 'enabled'


def _extract_hailo_unsupported_issue_info(text: Any) -> Dict[str, Any]:
    """Pull compact unsupported-op details out of a Hailo parser/build error string.

    The Hailo toolchain tends to emit long free-form messages. For the benchmark
    summary we only need a compact, stable subset: error classes, op types and a
    few representative node names.
    """

    compact = " ".join(str(text or '').split())
    if not compact:
        return {
            'explicit': False,
            'activation': False,
            'ops': [],
            'nodes': [],
            'error_classes': [],
        }

    error_classes: List[str] = []
    for match in re.finditer(r'\b([A-Za-z0-9_]*(?:Unsupported|Unexpected)[A-Za-z0-9_]*Error)\b', compact):
        name = str(match.group(1) or '').strip()
        if name and name not in error_classes:
            error_classes.append(name)

    node_hits: List[str] = []
    for pat in (
        r'\bin op\s+([^:]+):',
        r'\bUnexpected activation at\s+([^,\s]+)',
        r'\bUnexpected node\s+([^\s]+)',
        r'\bat\s+([^,\s]+),\s*op=',
        r'\bnodes?=([^;]+)',
    ):
        for match in re.finditer(pat, compact, flags=re.IGNORECASE):
            raw_node = str(match.group(1) or '').strip()
            if not raw_node:
                continue
            if pat == r'\bnodes?=([^;]+)':
                parts = [part.strip() for part in raw_node.split(',') if part.strip()]
            else:
                parts = [raw_node]
            for part in parts:
                if part and part not in node_hits:
                    node_hits.append(part)

    op_hits: List[str] = []
    for pat in (
        r'\bop=([A-Za-z_][A-Za-z0-9_]*)\b',
        r'\bops=([A-Za-z_][A-Za-z0-9_]*)\b',
        r'\b([A-Za-z_][A-Za-z0-9_]*)\s+operation is unsupported\b',
        r'\bUnexpected activation[^,]*,\s*op=([A-Za-z_][A-Za-z0-9_]*)\b',
    ):
        for match in re.finditer(pat, compact, flags=re.IGNORECASE):
            op = str(match.group(1) or '').strip()
            if op and op not in op_hits:
                op_hits.append(op)

    explicit = bool(error_classes or re.search(r'\b(?:unsupported|unexpected)\b', compact, flags=re.IGNORECASE))
    activation = bool(
        re.search(r'UnsupportedActivationLayerError|Unexpected activation', compact, flags=re.IGNORECASE)
    )
    return {
        'explicit': explicit,
        'activation': activation,
        'ops': op_hits,
        'nodes': node_hits,
        'error_classes': error_classes,
    }


def _format_hailo_unsupported_issue_brief(info: Mapping[str, Any]) -> str:
    ops = [str(x).strip() for x in list(info.get('ops') or []) if str(x).strip()]
    nodes = [str(x).strip() for x in list(info.get('nodes') or []) if str(x).strip()]
    error_classes = [str(x).strip() for x in list(info.get('error_classes') or []) if str(x).strip()]
    pieces: List[str] = []
    if ops:
        pieces.append('ops=' + ', '.join(ops[:4]))
    if error_classes:
        pieces.append('errors=' + ', '.join(error_classes[:3]))
    if nodes:
        preview = ', '.join(nodes[:3])
        if len(nodes) > 3:
            preview += ', …'
        pieces.append('nodes=' + preview)
    if pieces:
        return '; '.join(pieces)
    return 'unsupported op / activation in Hailo parser preflight'


_INPUT_PRELIGHT_PATTERNS = (
    'pixel_values',
    'images',
    'input',
    'embedding',
    'embeddings',
    'patch_embeddings',
    'patch_embed',
    'stem',
    'stage0',
    'stages.0',
)

_OUTPUT_PRELIGHT_PATTERNS = (
    'head',
    'classifier',
    'logits',
    'output',
    'postprocess',
    'nms',
    'stage3',
    'stages.3',
)


def _normalize_hailo_node_token(value: Any) -> str:
    s = str(value or '').strip().strip(',:;')
    return s


def _classify_hailo_preflight_failure_scope(model_path: str, unsupported_nodes: Sequence[Any]) -> Dict[str, Any]:
    """Infer whether unsupported Hailo parser nodes cluster near the input or output side.

    This is intentionally heuristic. We prefer a cheap, conservative signal over a
    brittle exact graph analysis: if unsupported nodes are clearly near the input,
    Hailo-first plans are usually impossible but TRT-to-Hailo may still work; if they
    are near the output side, the reverse often holds.
    """

    node_names = [_normalize_hailo_node_token(x) for x in list(unsupported_nodes or []) if _normalize_hailo_node_token(x)]
    if not node_names:
        return {'scope': 'unknown', 'matched_nodes': 0, 'total_nodes': None, 'fractions': []}

    fractions: List[float] = []
    total_nodes: Optional[int] = None
    try:
        import onnx  # type: ignore

        model = onnx.load(str(model_path), load_external_data=False)
        graph_nodes = list(getattr(getattr(model, 'graph', None), 'node', []) or [])
        total_nodes = int(len(graph_nodes))
        alias_to_index: Dict[str, int] = {}
        for idx, node in enumerate(graph_nodes):
            aliases: Set[str] = set()
            name = _normalize_hailo_node_token(getattr(node, 'name', ''))
            if name:
                aliases.add(name)
                aliases.add(name.lstrip('/'))
            for out in list(getattr(node, 'output', []) or []):
                tok = _normalize_hailo_node_token(out)
                if tok:
                    aliases.add(tok)
                    aliases.add(tok.lstrip('/'))
            for alias in aliases:
                alias_to_index.setdefault(alias, int(idx))
        denom = float(total_nodes - 1) if total_nodes and total_nodes > 1 else 1.0
        for raw in node_names:
            idx = alias_to_index.get(raw)
            if idx is None:
                idx = alias_to_index.get(raw.lstrip('/'))
            if idx is None:
                continue
            fractions.append(max(0.0, min(1.0, float(idx) / denom)))
    except Exception:
        fractions = []

    if fractions:
        min_f = min(fractions)
        max_f = max(fractions)
        if max_f <= 0.35:
            scope = 'input'
        elif min_f >= 0.65:
            scope = 'output'
        else:
            scope = 'mixed'
        return {
            'scope': scope,
            'matched_nodes': int(len(fractions)),
            'total_nodes': total_nodes,
            'fractions': list(fractions),
        }

    lowered = ' '.join(node_names).lower()
    input_hits = any(pat in lowered for pat in _INPUT_PRELIGHT_PATTERNS)
    output_hits = any(pat in lowered for pat in _OUTPUT_PRELIGHT_PATTERNS)
    if input_hits and not output_hits:
        scope = 'input'
    elif output_hits and not input_hits:
        scope = 'output'
    elif input_hits or output_hits:
        scope = 'mixed'
    else:
        scope = 'unknown'
    return {'scope': scope, 'matched_nodes': 0, 'total_nodes': total_nodes, 'fractions': []}



def _blocked_hailo_kinds_from_preflight_scope(scope: str) -> Set[str]:
    scope_key = str(scope or '').strip().lower()
    blocked: Set[str] = {'full'}
    if scope_key == 'input':
        blocked.add('part1')
    elif scope_key == 'output':
        blocked.add('part2')
    return blocked


@dataclass
class BenchmarkGenerationRuntime:
    out_dir: Path
    bench_log_path: Path
    state_path: Path
    requested_cases: int
    ranked_candidates: List[int]
    candidate_search_pool: List[int]
    model_name: str
    model_source: str
    hef_full_policy: str
    suite_hailo_hefs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    generation_state: Dict[str, Any] = field(default_factory=dict)
    completed_boundaries: Set[int] = field(default_factory=set)
    accepted_boundaries: Set[int] = field(default_factory=set)
    discarded_boundaries: Set[int] = field(default_factory=set)
    cases: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    resumed_previous_errors: List[str] = field(default_factory=list)
    discarded_cases: List[Dict[str, Any]] = field(default_factory=list)
    plan_adjustments: List[str] = field(default_factory=list)
    bench_log_fp: Optional[IO[str]] = None
    bench_log_lock: Any = field(default_factory=threading.Lock)

    def log(self, line: str, *, queue_put=None, level: int = logging.INFO) -> None:
        line = str(line or '').rstrip('\n')
        if not line:
            return
        stamped = f"{datetime.now().isoformat(timespec='seconds')} {line}"
        try:
            if self.bench_log_fp is not None:
                with self.bench_log_lock:
                    self.bench_log_fp.write(stamped + "\n")
                    self.bench_log_fp.flush()
        except Exception:
            pass
        try:
            logger.log(level, "[benchmark] %s", line)
        except Exception:
            pass
        if callable(queue_put):
            try:
                queue_put(("log", line))
            except Exception:
                pass

    def persist(self, *, status: str = "running", current_boundary: Optional[int] = None) -> None:
        self.generation_state = update_generation_state(
            self.state_path,
            self.generation_state,
            status=str(status),
            requested_cases=int(self.requested_cases),
            generated_cases=int(len(self.cases)),
            discarded_cases_count=int(len(self.discarded_cases)),
            shortfall=max(0, int(self.requested_cases) - int(len(self.cases))),
            current_boundary=(None if current_boundary is None else int(current_boundary)),
            completed_boundaries=sorted(int(x) for x in self.completed_boundaries),
            accepted_boundaries=sorted(int(x) for x in self.accepted_boundaries),
            discarded_boundaries=sorted(int(x) for x in self.discarded_boundaries),
            case_entries=list(self.cases),
            discarded_case_entries=list(self.discarded_cases),
            errors=list(self.errors),
            suite_full_hefs=dict(self.suite_hailo_hefs),
        )

    def close(self) -> None:
        try:
            if self.bench_log_fp is not None:
                self.bench_log_fp.flush()
                self.bench_log_fp.close()
        except Exception:
            pass
        self.bench_log_fp = None


@dataclass
class BenchmarkGenerationFinalizeResult:
    bench_payload: Dict[str, Any]
    harness_path: Path
    plan_path: Path
    readme_path: Path


@dataclass
class LoadedBenchmarkAnalysis:
    report: BenchmarkAnalysisReport
    interleaving: InterleavingAnalysisReport


@dataclass
class LoadedBenchmarkComparison:
    comparison: BenchmarkComparisonReport
    interleaving_comparison: InterleavingComparisonReport


class BenchmarkAnalysisService:
    def __init__(self, cache_base: Path):
        self.cache_base = Path(cache_base)

    def load_single(self, source: str | Path) -> LoadedBenchmarkAnalysis:
        report = load_benchmark_analysis(source, cache_base=self.cache_base)
        inter = compute_interleaving_analysis(report)
        return LoadedBenchmarkAnalysis(report=report, interleaving=inter)

    def load_comparison(self, left_source: str | Path, right_source: str | Path) -> LoadedBenchmarkComparison:
        comparison = load_benchmark_analysis_comparison(left_source, right_source, cache_base=self.cache_base)
        inter_cmp = compare_interleaving_reports(comparison.left, comparison.right)
        return LoadedBenchmarkComparison(comparison=comparison, interleaving_comparison=inter_cmp)

    def export_single(self, loaded: LoadedBenchmarkAnalysis, output_dir: Path) -> Dict[str, Path]:
        out = export_benchmark_analysis(loaded.report, output_dir)
        out.update(export_interleaving_analysis(loaded.interleaving, output_dir))
        out.update(export_decision_summary(loaded.report, loaded.interleaving, output_dir))
        return out

    def export_comparison(self, loaded: LoadedBenchmarkComparison, output_dir: Path) -> Dict[str, Path]:
        out = export_benchmark_comparison(loaded.comparison, output_dir)
        out.update(export_interleaving_comparison(loaded.interleaving_comparison, output_dir))
        out.update(export_metric_audit_comparison(loaded.comparison.left, compute_interleaving_analysis(loaded.comparison.left), loaded.comparison.right, compute_interleaving_analysis(loaded.comparison.right), output_dir))
        return out

    def export_publication_single(self, loaded: LoadedBenchmarkAnalysis, output_dir: Path, *, use_calibration: bool = True) -> Dict[str, Path]:
        return export_publication_analysis_bundle(loaded.report, loaded.interleaving, output_dir, use_calibration=use_calibration)

    def export_publication_comparison(self, loaded: LoadedBenchmarkComparison, output_dir: Path, *, use_calibration: bool = True) -> Dict[str, Path]:
        return export_publication_comparison_bundle(
            loaded.comparison.left,
            compute_interleaving_analysis(loaded.comparison.left),
            loaded.comparison.right,
            compute_interleaving_analysis(loaded.comparison.right),
            output_dir,
            use_calibration=use_calibration,
        )


class BenchmarkBundleService:
    def create_results_bundle_from_suite(self, suite_dir: Path, tar_gz_path: Path, *, mode: str = 'full') -> Path:
        create_results_bundle_from_suite(Path(suite_dir), Path(tar_gz_path), mode=mode)
        return Path(tar_gz_path)


class BenchmarkSchemaService:
    def load(self, path: Path) -> dict:
        return read_benchmark_set(Path(path))

    def migrate(self, payload: dict) -> dict:
        return migrate_benchmark_set_payload(payload)


@dataclass
class HailoCompileOutlookRow:
    boundary: int
    compile_risk_score: float
    single_context_probability: float
    cut_mib: Optional[float]
    peak_act_right_mib: Optional[float]
    n_cut_tensors: Optional[int]
    flops_right_ratio: Optional[float]
    base_score: Optional[float]
    strict_ok: Optional[bool]
    risk_band: str
    recommendation: str


@dataclass
class HailoCompileOutlookSummary:
    candidate_count: int
    avg_risk_score: Optional[float]
    likely_single_context_count: int
    low_risk_count: int
    medium_risk_count: int
    high_risk_count: int
    top_boundary: Optional[int]
    top_single_context_probability: Optional[float]


@dataclass
class BenchmarkGenerationPlan:
    ranked_candidates: List[int]
    candidate_search_pool: List[int]
    requested_cases: int
    strict_boundary: bool
    hailo_selected: bool
    hailo_compile_rank_meta: Dict[int, Dict[str, Any]]
    hailo_outlook_rows: List[HailoCompileOutlookRow]
    hailo_outlook_summary: Optional[HailoCompileOutlookSummary]


@dataclass
class BenchmarkRunPlan:
    bench_plan_runs: List[Dict[str, Any]]
    hef_targets: List[str]
    hailo_selected: bool
    hef_full: bool
    hef_part1: bool
    hef_part2: bool
    hailo_variants: List[str]
    matrix_variants: List[str]
    image_scale: str
    validation_images: Optional[str] = None
    validation_max_images: int = 0
    validation_reference_mode: str = "auto"
    mini_coco_ap50: bool = False
    benchmark_task: str = "auto"
    mini_classification_eval: bool = False


STREAMING_PRESET_PROFILES: Dict[str, Dict[str, int]] = {
    'disabled': {'frames': 0, 'warmup': 0, 'queue_depth': 1},
    'latency': {'frames': 8, 'warmup': 2, 'queue_depth': 1},
    'default': {'frames': 24, 'warmup': 6, 'queue_depth': 2},
    'throughput': {'frames': 48, 'warmup': 8, 'queue_depth': 3},
    'aggressive': {'frames': 96, 'warmup': 12, 'queue_depth': 4},
}


def _risk_band(score: float) -> str:
    s = float(score)
    if s <= 1.7:
        return 'low'
    if s <= 2.5:
        return 'medium'
    return 'high'


def _recommendation(single_prob: float, risk: float) -> str:
    if single_prob >= 0.80 and risk <= 1.9:
        return 'Low context risk / measure normally'
    if single_prob >= 0.65:
        return 'Likely 1-context / measure'
    if single_prob >= 0.45:
        return 'Borderline; measure, do not discard'
    return 'Likely multi-context; keep if measured FPS/quality are good'


class BenchmarkGenerationService:
    """Prepare benchmark generation plans and Hailo compile outlooks.

    This intentionally keeps GUI code thinner: candidate preparation, strict-boundary
    filtering, Hailo-aware reordering and the user-facing completion summary live
    here instead of being hand-assembled inside :mod:`gui_app`.
    """

    def start_generation_runtime(
        self,
        *,
        out_dir: str | Path,
        bench_log_path: str | Path,
        requested_cases: int,
        ranked_candidates: Sequence[int],
        candidate_search_pool: Sequence[int],
        hef_full_policy: str,
        model_name: str,
        model_source: str,
        resume_generation: bool = False,
        resume_state_hint: Optional[Mapping[str, Any]] = None,
    ) -> BenchmarkGenerationRuntime:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        runtime = BenchmarkGenerationRuntime(
            out_dir=out_dir,
            bench_log_path=Path(bench_log_path),
            state_path=out_dir / 'generation_state.json',
            requested_cases=int(requested_cases),
            ranked_candidates=[int(x) for x in ranked_candidates],
            candidate_search_pool=[int(x) for x in candidate_search_pool],
            model_name=str(model_name),
            model_source=str(model_source),
            hef_full_policy=normalize_full_hef_policy(hef_full_policy),
            generation_state=init_generation_state(
                model_name=str(model_name),
                model_source=str(model_source),
                requested_cases=int(requested_cases),
                ranked_candidates=[int(x) for x in ranked_candidates],
                candidate_search_pool=[int(x) for x in candidate_search_pool],
                hef_full_policy=normalize_full_hef_policy(hef_full_policy),
                run_mode=('resume' if resume_generation else 'new'),
            ),
        )
        runtime.bench_log_path.parent.mkdir(parents=True, exist_ok=True)
        runtime.bench_log_fp = open(runtime.bench_log_path, 'a', encoding='utf-8', buffering=1)
        if resume_generation and isinstance(resume_state_hint, Mapping) and resume_state_hint.get('created_at'):
            runtime.generation_state['created_at'] = resume_state_hint.get('created_at')
        if resume_generation and isinstance(resume_state_hint, Mapping):
            try:
                cases = list(resume_state_hint.get('case_entries') or resume_state_hint.get('cases') or [])
                discarded_cases = list(resume_state_hint.get('discarded_case_entries') or resume_state_hint.get('discarded_cases') or [])
                prev_errors = [str(e) for e in (resume_state_hint.get('errors') or []) if str(e).strip()]
                runtime.resumed_previous_errors.extend(prev_errors)
                runtime.cases.extend(cases)
                runtime.discarded_cases.extend(discarded_cases)
                for rec in cases:
                    runtime.accepted_boundaries.add(int(rec.get('boundary')))
                for rec in discarded_cases:
                    runtime.discarded_boundaries.add(int(rec.get('boundary')))
                runtime.completed_boundaries = set(runtime.accepted_boundaries) | set(runtime.discarded_boundaries)
            except Exception:
                logger.debug('Failed to restore generation resume state', exc_info=True)
        runtime.persist(status='running', current_boundary=None)
        return runtime

    def copy_portable_full_model(self, runtime: BenchmarkGenerationRuntime, full_model_src: str | Path, *, log_cb=None) -> str:
        full_model_src = os.path.abspath(str(full_model_src))
        models_dir = runtime.out_dir / 'models'
        models_dir.mkdir(parents=True, exist_ok=True)
        full_model_dst = models_dir / Path(full_model_src).name
        try:
            if os.path.abspath(str(full_model_dst)) != full_model_src:
                shutil.copy2(full_model_src, full_model_dst)
                for suffix in ('.export.json', '.categories.json'):
                    side_src = Path(full_model_src).with_suffix(suffix)
                    if side_src.is_file():
                        try:
                            shutil.copy2(side_src, full_model_dst.with_suffix(suffix))
                        except Exception:
                            logger.debug('Failed to copy model sidecar %s', side_src, exc_info=True)
        except Exception as exc:
            msg = f"full model copy failed: {type(exc).__name__}: {exc}"
            runtime.errors.append(msg)
            if callable(log_cb):
                log_cb(msg, level=logging.WARNING)
            return full_model_src
        return str(full_model_dst)

    def compact_hailo_build_summary(self, res: Any) -> Dict[str, Any]:
        details = getattr(res, 'details', None) or {}
        if not isinstance(details, dict):
            details = {}
        proc = details.get('process_summary') if isinstance(details.get('process_summary'), dict) else {}
        detected = proc.get('detected') if isinstance(proc.get('detected'), dict) else {}
        context_count = _safe_int(proc.get('context_count'))
        if bool(getattr(res, 'skipped', False)):
            context_mode = 'skipped'
        elif bool(detected.get('single_context_failed')) and bool(detected.get('multi_context_used')):
            context_mode = 'single_context_failed_to_multi'
        elif bool(detected.get('single_context_used')) and not bool(detected.get('multi_context_used')):
            context_mode = 'single_context_used'
        elif bool(detected.get('multi_context_used')):
            context_mode = 'multi_context_used'
        elif context_count == 1:
            context_mode = 'single_context_used'
        elif context_count is not None and context_count > 1:
            context_mode = 'multi_context_used'
        elif bool(getattr(res, 'ok', False)):
            context_mode = 'ok_unknown_context'
        else:
            context_mode = 'failed'

        calib_info = getattr(res, 'calib_info', None) or {}
        if not isinstance(calib_info, dict):
            calib_info = {}

        out = {
            'ok': bool(getattr(res, 'ok', False)),
            'skipped': bool(getattr(res, 'skipped', False)),
            'timed_out': bool(getattr(res, 'timed_out', False)),
            'failure_kind': getattr(res, 'failure_kind', None),
            'unsupported_reason': getattr(res, 'unsupported_reason', None),
            'error': getattr(res, 'error', None),
            'elapsed_s': getattr(res, 'elapsed_s', None),
            'context_count': context_count,
            'context_mode': context_mode,
            'partition_iterations': _safe_int(proc.get('partition_iterations')),
            'partition_time_s': proc.get('partition_time_s'),
            'allocation_time_s': proc.get('allocation_time_s'),
            'compilation_time_s': proc.get('compilation_time_s'),
            'calib_source': calib_info.get('source'),
            'single_context_failed': bool(detected.get('single_context_failed')),
            'single_context_used': bool(detected.get('single_context_used')),
            'multi_context_used': bool(detected.get('multi_context_used')),
            'mapping_failed': bool(detected.get('mapping_failed')),
            'watchdog_expired': bool(detected.get('watchdog_expired')),
        }
        # v45: advisory build-cost flags for expensive Hailo builds.
        try:
            elapsed = float(out.get('elapsed_s') or 0.0)
        except Exception:
            elapsed = 0.0
        ctx_mode = str(out.get('context_mode') or '').strip()
        expensive_reasons: List[str] = []
        if elapsed >= 1800.0:
            expensive_reasons.append('elapsed_s>=1800')
        if ctx_mode in {'multi_context_used', 'single_context_failed_to_multi'}:
            expensive_reasons.append(ctx_mode)
        if expensive_reasons:
            out['expensive_build'] = True
            out['expensive_build_reason'] = ';'.join(expensive_reasons)
        return {k: v for k, v in out.items() if v not in (None, '') or isinstance(v, bool)}

    def hailo_build_cost_note(self, summary: Mapping[str, Any], *, stage: str = '') -> str:
        if not isinstance(summary, Mapping):
            return ''
        notes: List[str] = []
        try:
            elapsed = float(summary.get('elapsed_s') or 0.0)
        except Exception:
            elapsed = 0.0
        if elapsed >= 1800.0:
            notes.append(f'elapsed={elapsed:.1f}s')
        ctx = str(summary.get('context_mode') or '').strip()
        if ctx in {'multi_context_used', 'single_context_failed_to_multi'}:
            notes.append(f'context={ctx}')
        cc = _safe_int(summary.get('context_count'))
        if cc is not None and cc > 1:
            notes.append(f'contexts={cc}')
        if not notes:
            return ''
        label = str(stage or 'Hailo build').strip()
        return f'{label} cost note: ' + ', '.join(notes) + '; cached/reused HEFs are recommended for repeated evaluation runs.'

    def case_hailo_compile_from_hefs(self, hefs_payload: Any) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if not isinstance(hefs_payload, dict):
            return out
        for hw_arch, hw_meta in hefs_payload.items():
            if not isinstance(hw_meta, dict):
                continue
            arch_out: Dict[str, Any] = {}
            for stage in ('part1', 'part2'):
                stage_meta = hw_meta.get(f'{stage}_build')
                if isinstance(stage_meta, dict):
                    arch_out[stage] = dict(stage_meta)
                    if hw_meta.get(stage) and arch_out[stage].get('hef_path') in (None, ''):
                        arch_out[stage]['hef_path'] = hw_meta.get(stage)
                    if hw_meta.get(f'{stage}_error') and arch_out[stage].get('error') in (None, ''):
                        arch_out[stage]['error'] = hw_meta.get(f'{stage}_error')
                elif hw_meta.get(stage) or hw_meta.get(f'{stage}_error'):
                    tmp: Dict[str, Any] = {}
                    if hw_meta.get(stage):
                        tmp['hef_path'] = hw_meta.get(stage)
                    if hw_meta.get(f'{stage}_error'):
                        tmp['error'] = hw_meta.get(f'{stage}_error')
                    arch_out[stage] = tmp
            if arch_out:
                out[str(hw_arch)] = arch_out
        return out

    def summarize_hailo_context_fit(self, case_entries: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
        stats: Dict[str, Dict[str, Any]] = {}
        for rec in case_entries:
            compile_meta = rec.get('hailo_compile') if isinstance(rec, Mapping) and isinstance(rec.get('hailo_compile'), dict) else {}
            for hw_arch, hw_meta in compile_meta.items():
                if not isinstance(hw_meta, dict):
                    continue
                target = stats.setdefault(str(hw_arch), {
                    'cases': 0,
                    'part2_single_context': 0,
                    'both_parts_single_context': 0,
                    'single_to_multi_fallback': 0,
                    'multi_context': 0,
                })
                target['cases'] = int(target.get('cases', 0)) + 1
                p1 = hw_meta.get('part1') if isinstance(hw_meta.get('part1'), dict) else {}
                p2 = hw_meta.get('part2') if isinstance(hw_meta.get('part2'), dict) else {}
                p1_count = _safe_int(p1.get('context_count'))
                p2_count = _safe_int(p2.get('context_count'))
                p1_mode = str(p1.get('context_mode') or '')
                p2_mode = str(p2.get('context_mode') or '')
                p1_single = bool(p1_mode == 'single_context_used' or p1_count == 1)
                p2_single = bool(p2_mode == 'single_context_used' or p2_count == 1)
                if p2_single:
                    target['part2_single_context'] = int(target.get('part2_single_context', 0)) + 1
                if p1_single and p2_single:
                    target['both_parts_single_context'] = int(target.get('both_parts_single_context', 0)) + 1
                if p2_mode == 'single_context_failed_to_multi':
                    target['single_to_multi_fallback'] = int(target.get('single_to_multi_fallback', 0)) + 1
                elif p2_mode == 'multi_context_used':
                    target['multi_context'] = int(target.get('multi_context', 0)) + 1
        return stats

    def format_benchmark_case_label(self, rec: Mapping[str, Any]) -> str:
        try:
            bnd = int(rec.get('boundary'))
        except Exception:
            bnd = rec.get('boundary')
        hw = str(rec.get('hw_arch') or '').strip()
        reason = str(rec.get('reason') or rec.get('stage') or 'rejected')
        likely_orig = list(rec.get('likely_original_inputs') or [])
        missing = list(rec.get('missing_inputs') or [])
        if likely_orig:
            detail = f"needs original input(s): {', '.join(str(x) for x in likely_orig)}"
        elif missing:
            detail = f"missing from Part1 outputs: {', '.join(str(x) for x in missing)}"
        else:
            detail = str(rec.get('detail') or '').splitlines()[0].strip()
        label = f"b{bnd}"
        if hw:
            label += f" ({hw})"
        if detail:
            label += f": {detail}"
        elif reason:
            label += f": {reason}"
        return label

    def finalize_generation_outputs(
        self,
        *,
        out_dir: str | Path,
        base: str,
        full_model_src: str,
        full_model_dst: str,
        analysis_params: Mapping[str, Any],
        system_spec: Optional[Mapping[str, Any]],
        cases: Sequence[Mapping[str, Any]],
        errors: Sequence[str],
        discarded_cases: Sequence[Mapping[str, Any]],
        requested_cases: int,
        preferred_shortlist_original: Sequence[int],
        ranked_candidates: Sequence[int],
        shortlist_prefiltered_boundaries: Sequence[int],
        candidate_search_pool: Sequence[int],
        bench_log_path: str | Path,
        analysis_payload: Mapping[str, Any],
        bench_plan_runs: Sequence[Mapping[str, Any]],
        hef_targets: Sequence[str],
        hef_full: bool,
        hef_part1: bool,
        hef_part2: bool,
        hef_backend: str,
        hef_wsl_distro: Optional[str],
        hef_wsl_venv: str,
        hef_opt_level: int,
        hef_calib_count: int,
        hef_calib_bs: int,
        hef_calib_dir: Optional[str],
        hef_fixup: bool,
        hef_force: bool,
        hef_keep: bool,
        suite_hailo_hefs: Optional[Mapping[str, Any]],
        write_harness_script,
        hailo_full_model_preflight: Optional[Mapping[str, Any]] = None,
        copy_schema_tree=None,
        tool_gui_version: Optional[str] = None,
        tool_core_version: Optional[str] = None,
        benchmark_objective: str = 'latency',
        evaluation_profile: Optional[Mapping[str, Any]] = None,
        full_model_preflight_policy: str = 'enabled',
        hailo_full_end_node_names: Optional[Sequence[str]] = None,
        hailo_full_endpoint_mode: str = '',
    ) -> BenchmarkGenerationFinalizeResult:
        out_dir = Path(out_dir)
        bench_path = out_dir / 'benchmark_set.json'
        plan_path = out_dir / 'benchmark_plan.json'

        # Cases are generated before suite-global Hailo full HEFs are built. Refresh the
        # per-case availability map here so downstream benchmark execution can see that the
        # suite-wide 'full' variant is actually available.
        cases_out: List[Dict[str, Any]] = []
        suite_hailo_hefs_map = dict(suite_hailo_hefs) if isinstance(suite_hailo_hefs, Mapping) else {}
        for raw_case in cases:
            case = dict(raw_case) if isinstance(raw_case, Mapping) else {'value': raw_case}
            availability = case.get('hailo_case_variant_availability')
            avail_map = dict(availability) if isinstance(availability, Mapping) else {}
            if suite_hailo_hefs_map:
                for hw_arch, suite_meta_raw in suite_hailo_hefs_map.items():
                    suite_meta = dict(suite_meta_raw) if isinstance(suite_meta_raw, Mapping) else {}
                    hw_key = str(hw_arch)
                    cur = dict(avail_map.get(hw_key) or {})
                    full_ok = bool(suite_meta.get('full')) and not bool(suite_meta.get('full_error'))
                    if full_ok or cur:
                        cur['full'] = bool(full_ok or cur.get('full'))
                        cur['full_failed'] = bool(suite_meta.get('full_error'))
                        cur.setdefault('part1', bool(cur.get('part1')))
                        cur.setdefault('part2', bool(cur.get('part2')))
                        cur['composed'] = bool(cur.get('part1')) and bool(cur.get('part2'))
                        avail_map[hw_key] = cur
            if avail_map:
                case['hailo_case_variant_availability'] = avail_map
            cases_out.append(case)

        # Provision semantic validation sources once at suite level so remote runs stay
        # self-contained. Detection without an explicit dataset uses the prepared COCO-50.
        # Classification must never silently receive COCO; it either uses an explicit
        # labeled dataset/preset or the first locally imported ImageNet-mini preset.
        embedded_validation_rel = None
        try:
            classification_cache: Dict[str, Optional[str]] = {}
            default_cls_preset = default_available_classification_validation_preset(base_dir=out_dir)

            def _provision_classification_request(req: str) -> Optional[str]:
                req = str(req or '').strip()
                if not req:
                    return None
                if req not in classification_cache:
                    classification_cache[req] = provision_classification_validation_source_to_suite(
                        out_dir,
                        req,
                        base_dir=out_dir,
                    )
                return classification_cache.get(req)

            # Provision COCO only for non-classification runs that do not already have a dataset.
            needs_detection_default = any(
                normalize_benchmark_task((run or {}).get('benchmark_task')) != 'classification'
                and not str((run or {}).get('validation_images') or '').strip()
                for run in bench_plan_runs
            )
            if needs_detection_default:
                embedded_validation_rel = _provision_embedded_semantic_validation_dataset(out_dir)

            for run in bench_plan_runs:
                task = normalize_benchmark_task((run or {}).get('benchmark_task'))
                req = str((run or {}).get('validation_images') or '').strip()
                if task == 'classification':
                    effective_req = req or str(default_cls_preset or '')
                    rel = _provision_classification_request(effective_req) if effective_req else None
                    if rel:
                        run['validation_images'] = str(rel)
                        try:
                            current_max = int(run.get('validation_max_images') or 0)
                        except Exception:
                            current_max = 0
                        if current_max <= 0:
                            run['validation_max_images'] = 500 if effective_req == 'imagenet_val_mini_500' else 200
                    else:
                        # Keep classification validation disabled rather than mixing in COCO labels.
                        run['validation_images'] = ''
                        run['validation_max_images'] = 0
                    continue

                # Detection / auto: explicit datasets remain untouched; missing datasets use COCO-50.
                if not req and embedded_validation_rel:
                    run['validation_images'] = str(embedded_validation_rel)
                    try:
                        current_max = int(run.get('validation_max_images') or 0)
                    except Exception:
                        current_max = 0
                    run['validation_max_images'] = max(current_max, 50)
                elif req and task == 'classification':
                    # Defensive no-op; kept for readability if normalization expands later.
                    pass
        except Exception:
            logger.debug('Failed to provision semantic validation dataset', exc_info=True)

        bench = {
            'schema': 'onnx-splitpoint/benchmark-set',
            'schema_version': 2,
            'tool': {'gui': tool_gui_version or '', 'core': tool_core_version or ''},
            'model': (Path(os.path.relpath(full_model_dst, start=out_dir)).as_posix() if os.path.exists(full_model_dst) else str(full_model_src).replace('\\', '/')),
            'model_source': str(full_model_src).replace('\\', '/'),
            'model_name': str(base),
            'created_at': datetime.now().isoformat(timespec='seconds'),
            'analysis_params': dict(analysis_params or {}),
            'system_spec': (dict(system_spec) if isinstance(system_spec, Mapping) else None),
            'cases': list(cases_out),
            'errors': list(errors),
            'discarded_cases': list(discarded_cases),
            'summary': {
                'requested_cases': int(requested_cases),
                'generated_cases': int(len(cases)),
                'discarded_cases': int(len(discarded_cases)),
                'auto_filtered_candidates': int(sum(1 for rec in discarded_cases if str(rec.get('reason') or '') in {'hailo_part2_prefilter', 'hailo_part2_precheck', 'hailo_part2_auto_filtered', 'hailo_part2_parser_prefilter', 'hailo_part2_parser_auto_filtered', 'hailo_part2_concat_sanity_prefilter', 'hailo_part2_concat_sanity_auto_filtered'})),
                'shortfall': max(0, int(requested_cases) - int(len(cases))),
                'preferred_shortlist_cases': int(len(preferred_shortlist_original)),
                'preferred_shortlist_after_prefilter': int(len(ranked_candidates)),
                'preferred_shortlist_filtered_candidates': int(len(shortlist_prefiltered_boundaries)),
                'candidate_search_pool': int(len(candidate_search_pool)),
                'search_pool_exhausted': bool(int(len(cases)) < int(requested_cases)),
            },
            'generation_log': Path(bench_log_path).name,
            'objective': str(benchmark_objective or 'latency'),
        }
        if isinstance(evaluation_profile, Mapping) and evaluation_profile:
            bench['evaluation_profile'] = dict(evaluation_profile)

        hailo_summary = analysis_payload.get('hailo_check') if isinstance(analysis_payload.get('hailo_check'), dict) else None
        hailo_results = analysis_payload.get('hailo_check_results') if isinstance(analysis_payload.get('hailo_check_results'), dict) else None
        if hailo_summary or hailo_results:
            bench['hailo_parse_check'] = {
                'summary': hailo_summary or {},
                'results_by_boundary': {str(k): v for k, v in (hailo_results or {}).items()},
            }

        bench_plan = {
            'schema': 'onnx-splitpoint/benchmark-plan',
            'schema_version': 1,
            'created_at': datetime.now().isoformat(timespec='seconds'),
            'runs': list(bench_plan_runs),
            'matrix': [],
            'objective': str(benchmark_objective or 'latency'),
        }
        if isinstance(evaluation_profile, Mapping) and evaluation_profile:
            bench_plan['evaluation_profile'] = dict(evaluation_profile)
        bench['plan'] = bench_plan
        write_benchmark_json_atomic(plan_path, bench_plan)

        if hef_targets and (hef_full or hef_part1 or hef_part2):
            bench['hailo'] = {
                'targets': [str(x) for x in hef_targets],
                'build': {'full': bool(hef_full), 'part1': bool(hef_part1), 'part2': bool(hef_part2)},
                'config': {
                    'backend': str(hef_backend),
                    'wsl_distro': hef_wsl_distro,
                    'wsl_venv': str(hef_wsl_venv),
                    'opt_level': int(hef_opt_level),
                    'calib_count': int(hef_calib_count),
                    'calib_batch_size': int(hef_calib_bs),
                    'calib_dir': (str(hef_calib_dir).replace('\\', '/') if hef_calib_dir else None),
                    'fixup': bool(hef_fixup),
                    'force': bool(hef_force),
                    'keep_artifacts': bool(hef_keep),
                    'full_model_preflight_policy': normalize_hailo_full_model_preflight_policy(full_model_preflight_policy),
                },
            }
            full_end_nodes_cfg = [str(x).strip() for x in (hailo_full_end_node_names or []) if str(x).strip()]
            if full_end_nodes_cfg:
                bench['hailo']['config']['full_endpoint_mode'] = str(hailo_full_endpoint_mode or 'custom')
                bench['hailo']['config']['full_end_node_names'] = list(full_end_nodes_cfg)
            suite_contract = None
            if isinstance(suite_hailo_hefs, Mapping):
                for _hw_meta in suite_hailo_hefs.values():
                    if isinstance(_hw_meta, Mapping) and isinstance(_hw_meta.get('full_output_contract'), Mapping):
                        suite_contract = dict(_hw_meta.get('full_output_contract') or {})
                        break
            if suite_contract:
                bench['hailo']['config']['full_output_contract'] = dict(suite_contract)
            if suite_hailo_hefs:
                bench['hailo']['hefs'] = dict(suite_hailo_hefs)
            if isinstance(hailo_full_model_preflight, Mapping):
                bench['hailo']['full_model_preflight'] = dict(hailo_full_model_preflight)
            hailo_context_summary = self.summarize_hailo_context_fit(cases)
            if hailo_context_summary:
                bench['hailo']['context_summary'] = hailo_context_summary

        bench = stamp_benchmark_set_payload(
            bench,
            tool_gui_version=tool_gui_version,
            tool_core_version=tool_core_version,
            suite_dir=Path(out_dir),
            bundle_options={'results_bundle_modes': ['full', 'lean']},
        )
        write_benchmark_json_atomic(bench_path, bench)

        harness_path = Path(write_harness_script(str(out_dir), 'benchmark_set.json'))

        if callable(copy_schema_tree):
            try:
                copy_schema_tree()
            except Exception:
                logger.debug('Failed to copy benchmark schemas', exc_info=True)

        readme_path = out_dir / 'README_BENCHMARK.txt'
        txt = (
            'Benchmark suite generated by the ONNX Split-Point Analyser.\n\n'
            f"Model (portable): {bench.get('model')}\n"
            f"Model (source):   {bench.get('model_source')}\n"
            f"Cases: {len(cases)} (requested: {requested_cases})\n\n"
            'Benchmark plan:\n'
            '  - See benchmark_plan.json (and benchmark_set.json -> plan).\n\n'
            'Next steps:\n'
            '  1) (optional) install deps: pip install onnx onnxruntime numpy pillow matplotlib\n'
            '  2) run ALL runs from the plan: python benchmark_suite.py\n'
            '  3) run a single ORT provider: python benchmark_suite.py --provider cpu\n'
            '     (or: --provider cuda / --provider tensorrt)\n'
            '  4) to also generate human-readable outputs: add --image default --preset auto\n'
            '  5) dataset semantic validation for detection models: by default the prepared COCO-50 set is wired in once at suite level (resources/validation/coco_50_data).\n'
            '     You can override validation_images + validation_max_images in benchmark_plan.json if needed.\n\n'
            'Outputs:\n'
            '  - benchmark_results_<tag>.csv / .json\n'
            '    (tag is typically: <run-id>_<preset>, e.g. ort_cpu_auto)\n'
            '  - benchmark_generation.log\n'
            '    (live generation console stream mirrored for post-mortems)\n'
            '\n'
            'Paper-ready analysis exports (created during benchmark set generation):\n'
            '  - analysis_plots/   (PDF + SVG)\n'
            '  - analysis_tables/  (.tex, .csv, .json)\n'
        )
        if bench.get('hailo'):
            txt += (
                '\nHailo artifacts:\n'
                '  - Suite-level full HEFs (if enabled): hailo/<hw_arch>/full/compiled.hef\n'
                '  - Per-case split HEFs (if enabled):  b*/hailo/<hw_arch>/(part1|part2)/compiled.hef\n'
                "  - Suite-level config is recorded in benchmark_set.json under 'hailo'.\n"
                "  - Per-case Hailo availability is recorded under cases[*].hailo_case_variant_availability.\n"
            )
        if discarded_cases:
            txt += (
                '\nRejected split attempts:\n'
                '  - Candidates that failed required Hailo per-case builds are not part of the suite.\n'
                '  - Preserved artifacts (if any) are archived under _rejected_cases/.\n'
            )
        readme_path.write_text(txt, encoding='utf-8')

        return BenchmarkGenerationFinalizeResult(
            bench_payload=bench,
            harness_path=harness_path,
            plan_path=plan_path,
            readme_path=readme_path,
        )

    def strict_filter_boundaries(self, analysis: Mapping[str, Any], raw_bounds: Sequence[int]) -> List[int]:
        from .. import api as asc

        model = analysis.get('model') if isinstance(analysis, Mapping) else None
        nodes = analysis.get('nodes') if isinstance(analysis, Mapping) else None
        order = analysis.get('order') if isinstance(analysis, Mapping) else None
        if model is None or nodes is None or order is None:
            return []

        filtered: List[int] = []
        strict_ok = analysis.get('strict_ok') if isinstance(analysis, Mapping) else None
        prelim = list(raw_bounds)
        if isinstance(strict_ok, Sequence) and not isinstance(strict_ok, (str, bytes, bytearray)) and len(strict_ok) > 0:
            prelim = [int(b) for b in prelim if 0 <= int(b) < len(strict_ok) and bool(strict_ok[int(b)])]

        for raw_b in prelim:
            try:
                b = int(raw_b)
                cut_tensors = asc.cut_tensors_for_boundary(order, nodes, b)
                extras = asc.strict_boundary_extras(model, cut_tensors)
            except Exception:
                continue
            if len(extras) == 0:
                filtered.append(b)
        return filtered

    def build_hailo_outlook(self, analysis: Mapping[str, Any], boundaries: Sequence[int], *, top_n: int = 12) -> tuple[List[HailoCompileOutlookRow], Optional[HailoCompileOutlookSummary], Dict[int, Dict[str, Any]]]:
        if not isinstance(analysis, Mapping) or not boundaries:
            return [], None, {}
        reranked, meta = rerank_candidates_for_hailo(analysis, boundaries)
        rows: List[HailoCompileOutlookRow] = []
        for b in list(reranked)[:max(1, int(top_n))]:
            h = heuristic_for_boundary(analysis, int(b))
            rows.append(HailoCompileOutlookRow(
                boundary=int(h.boundary),
                compile_risk_score=float(h.compile_risk_score),
                single_context_probability=float(h.single_context_probability),
                cut_mib=h.cut_mib,
                peak_act_right_mib=h.peak_act_right_mib,
                n_cut_tensors=h.n_cut_tensors,
                flops_right_ratio=h.flops_right_ratio,
                base_score=h.base_score,
                strict_ok=h.strict_ok,
                risk_band=_risk_band(h.compile_risk_score),
                recommendation=_recommendation(h.single_context_probability, h.compile_risk_score),
            ))

        all_rows: List[HailoCompileOutlookRow] = []
        for b in reranked:
            h = heuristic_for_boundary(analysis, int(b))
            all_rows.append(HailoCompileOutlookRow(
                boundary=int(h.boundary),
                compile_risk_score=float(h.compile_risk_score),
                single_context_probability=float(h.single_context_probability),
                cut_mib=h.cut_mib,
                peak_act_right_mib=h.peak_act_right_mib,
                n_cut_tensors=h.n_cut_tensors,
                flops_right_ratio=h.flops_right_ratio,
                base_score=h.base_score,
                strict_ok=h.strict_ok,
                risk_band=_risk_band(h.compile_risk_score),
                recommendation=_recommendation(h.single_context_probability, h.compile_risk_score),
            ))

        if not all_rows:
            return rows, None, meta

        risks = [r.compile_risk_score for r in all_rows if math.isfinite(r.compile_risk_score)]
        summary = HailoCompileOutlookSummary(
            candidate_count=len(all_rows),
            avg_risk_score=(sum(risks) / len(risks) if risks else None),
            likely_single_context_count=sum(1 for r in all_rows if r.single_context_probability >= 0.65),
            low_risk_count=sum(1 for r in all_rows if r.risk_band == 'low'),
            medium_risk_count=sum(1 for r in all_rows if r.risk_band == 'medium'),
            high_risk_count=sum(1 for r in all_rows if r.risk_band == 'high'),
            top_boundary=(all_rows[0].boundary if all_rows else None),
            top_single_context_probability=(all_rows[0].single_context_probability if all_rows else None),
        )
        return rows, summary, meta

    def prepare_generation_plan(
        self,
        analysis: Mapping[str, Any],
        ranked_candidates: Sequence[int],
        candidate_search_pool: Sequence[int],
        requested_cases: int,
        *,
        strict_boundary: bool = False,
        hailo_selected: bool = False,
        outlook_top_n: int = 12,
    ) -> BenchmarkGenerationPlan:
        ranked = [int(b) for b in ranked_candidates]
        search_pool = [int(b) for b in candidate_search_pool]
        if strict_boundary:
            ranked = self.strict_filter_boundaries(analysis, ranked)
            search_pool = self.strict_filter_boundaries(analysis, search_pool)
        hailo_meta: Dict[int, Dict[str, Any]] = {}
        outlook_rows: List[HailoCompileOutlookRow] = []
        outlook_summary: Optional[HailoCompileOutlookSummary] = None
        if hailo_selected and search_pool:
            reranked, hailo_meta = rerank_candidates_for_hailo(analysis, search_pool)
            if reranked:
                order = {int(b): idx for idx, b in enumerate(reranked)}
                search_pool = list(reranked)
                ranked = sorted([int(b) for b in ranked], key=lambda b: (order.get(int(b), 10**9), int(b)))
            outlook_rows, outlook_summary, _ = self.build_hailo_outlook(analysis, ranked or search_pool, top_n=outlook_top_n)
        requested = max(1, min(int(requested_cases), len(search_pool) if search_pool else 1))
        return BenchmarkGenerationPlan(
            ranked_candidates=ranked,
            candidate_search_pool=search_pool,
            requested_cases=requested,
            strict_boundary=bool(strict_boundary),
            hailo_selected=bool(hailo_selected),
            hailo_compile_rank_meta=hailo_meta,
            hailo_outlook_rows=outlook_rows,
            hailo_outlook_summary=outlook_summary,
        )

    def streaming_preset_profiles(self) -> Dict[str, Dict[str, int]]:
        return {k: dict(v) for k, v in STREAMING_PRESET_PROFILES.items()}

    def detect_streaming_preset(self, frames: int, warmup: int, queue_depth: int) -> str:
        try:
            triple = (int(frames), int(warmup), int(queue_depth))
        except Exception:
            return 'custom'
        for name, spec in STREAMING_PRESET_PROFILES.items():
            if triple == (int(spec['frames']), int(spec['warmup']), int(spec['queue_depth'])):
                return name
        return 'custom'

    def resolve_streaming_preset(self, preset: str) -> Dict[str, int]:
        key = str(preset or 'default').strip().lower()
        spec = STREAMING_PRESET_PROFILES.get(key) or STREAMING_PRESET_PROFILES['default']
        return dict(spec)

    def build_run_plan(
        self,
        *,
        acc_cpu: bool,
        acc_cuda: bool,
        acc_trt: bool,
        acc_h8: bool,
        acc_h10: bool,
        hailo8_hw: str,
        hailo10_hw: str,
        image_scale: str,
        validation_images: Optional[str] = None,
        validation_max_images: int = 0,
        validation_reference_mode: str = "auto",
        mini_coco_ap50: bool = False,
        benchmark_task: str = "auto",
        mini_classification_eval: bool = False,
        hailo_preset: str = "End-to-end compare",
        hailo_custom_full: bool,
        hailo_custom_composed: bool,
        hailo_custom_part1: bool,
        hailo_custom_part2: bool,
        matrix_trt_to_hailo: bool,
        matrix_hailo_to_trt: bool,
        full_hef_policy: str = 'end',
    ) -> BenchmarkRunPlan:
        plan_image_scale = str(image_scale or 'auto').strip().lower()
        if plan_image_scale not in {'auto', 'norm', 'raw', 'imagenet', 'clip'}:
            plan_image_scale = 'auto'

        plan_benchmark_task = normalize_benchmark_task(benchmark_task)
        plan_validation_images, plan_validation_max_images, _use_embedded_validation = normalize_semantic_validation_request(
            validation_images,
            validation_max_images,
            benchmark_task=plan_benchmark_task,
        )
        plan_validation_reference_mode = normalize_validation_reference_mode(validation_reference_mode)
        plan_mini_coco_ap50 = normalize_mini_coco_ap50(mini_coco_ap50)
        plan_mini_classification_eval = normalize_mini_classification_eval(mini_classification_eval)

        p = str(hailo_preset or '').strip().lower()
        if p.startswith('end'):
            hailo_variants: List[str] = ['full', 'composed']
        elif p.startswith('split'):
            hailo_variants = ['composed', 'part1', 'part2']
        elif p.startswith('every'):
            hailo_variants = ['full', 'composed', 'part1', 'part2']
        else:
            vv: List[str] = []
            if bool(hailo_custom_full):
                vv.append('full')
            if bool(hailo_custom_composed):
                vv.append('composed')
            if bool(hailo_custom_part1):
                vv.append('part1')
            if bool(hailo_custom_part2):
                vv.append('part2')
            hailo_variants = vv or ['full', 'composed']

        _allowed = {'full', 'composed', 'part1', 'part2'}
        _seen: Set[str] = set()
        hailo_variants = [v for v in hailo_variants if v in _allowed and (v not in _seen and not _seen.add(v))]
        full_hef_policy = normalize_full_hef_policy(full_hef_policy)
        if full_hef_policy == 'skip':
            hailo_variants = [v for v in hailo_variants if v != 'full']
            if not hailo_variants:
                hailo_variants = ['composed']

        matrix_variants: List[str] = ['full', 'part1', 'part2', 'composed']

        def _ensure_same_backend_full_reference(variants: List[str]) -> List[str]:
            vv = [str(v).strip().lower() for v in list(variants or []) if str(v).strip()]
            if plan_validation_reference_mode not in {'auto', 'same_backend_full'}:
                return vv
            if 'full' not in vv:
                return ['full'] + vv
            return vv

        bench_plan_runs: List[Dict[str, Any]] = []
        if bool(acc_cpu):
            bench_plan_runs.append({'id': 'ort_cpu', 'type': 'onnxruntime', 'provider': 'cpu', 'image_scale': plan_image_scale, 'validation_images': plan_validation_images, 'validation_max_images': plan_validation_max_images, 'validation_reference_mode': plan_validation_reference_mode, 'mini_coco_ap50': plan_mini_coco_ap50, 'benchmark_task': plan_benchmark_task, 'mini_classification_eval': plan_mini_classification_eval, 'stage1': {'type': 'onnxruntime', 'provider': 'cpu'}, 'stage2': {'type': 'onnxruntime', 'provider': 'cpu'}})
        if bool(acc_cuda):
            bench_plan_runs.append({'id': 'ort_cuda', 'type': 'onnxruntime', 'provider': 'cuda', 'image_scale': plan_image_scale, 'validation_images': plan_validation_images, 'validation_max_images': plan_validation_max_images, 'validation_reference_mode': plan_validation_reference_mode, 'mini_coco_ap50': plan_mini_coco_ap50, 'benchmark_task': plan_benchmark_task, 'mini_classification_eval': plan_mini_classification_eval, 'stage1': {'type': 'onnxruntime', 'provider': 'cuda'}, 'stage2': {'type': 'onnxruntime', 'provider': 'cuda'}})
        if bool(acc_trt):
            bench_plan_runs.append({'id': 'ort_tensorrt', 'type': 'onnxruntime', 'provider': 'tensorrt', 'image_scale': plan_image_scale, 'validation_images': plan_validation_images, 'validation_max_images': plan_validation_max_images, 'validation_reference_mode': plan_validation_reference_mode, 'mini_coco_ap50': plan_mini_coco_ap50, 'benchmark_task': plan_benchmark_task, 'mini_classification_eval': plan_mini_classification_eval, 'stage1': {'type': 'onnxruntime', 'provider': 'tensorrt'}, 'stage2': {'type': 'onnxruntime', 'provider': 'tensorrt'}})
        if bool(acc_h8) and str(hailo8_hw).strip():
            bench_plan_runs.append({'id': str(hailo8_hw).strip(), 'type': 'hailo', 'hw_arch': str(hailo8_hw).strip(), 'variants': list(_ensure_same_backend_full_reference(hailo_variants)), 'image_scale': plan_image_scale, 'validation_images': plan_validation_images, 'validation_max_images': plan_validation_max_images, 'validation_reference_mode': plan_validation_reference_mode, 'mini_coco_ap50': plan_mini_coco_ap50, 'benchmark_task': plan_benchmark_task, 'mini_classification_eval': plan_mini_classification_eval, 'stage1': {'type': 'hailo', 'hw_arch': str(hailo8_hw).strip()}, 'stage2': {'type': 'hailo', 'hw_arch': str(hailo8_hw).strip()}})
        if bool(acc_h10) and str(hailo10_hw).strip():
            bench_plan_runs.append({'id': str(hailo10_hw).strip(), 'type': 'hailo', 'hw_arch': str(hailo10_hw).strip(), 'variants': list(_ensure_same_backend_full_reference(hailo_variants)), 'image_scale': plan_image_scale, 'validation_images': plan_validation_images, 'validation_max_images': plan_validation_max_images, 'validation_reference_mode': plan_validation_reference_mode, 'mini_coco_ap50': plan_mini_coco_ap50, 'benchmark_task': plan_benchmark_task, 'mini_classification_eval': plan_mini_classification_eval, 'stage1': {'type': 'hailo', 'hw_arch': str(hailo10_hw).strip()}, 'stage2': {'type': 'hailo', 'hw_arch': str(hailo10_hw).strip()}})

        if bool(acc_trt) and (bool(matrix_trt_to_hailo) or bool(matrix_hailo_to_trt)):
            hailo_targets_for_matrix: List[str] = []
            if bool(acc_h8) and str(hailo8_hw).strip():
                hailo_targets_for_matrix.append(str(hailo8_hw).strip())
            if bool(acc_h10) and str(hailo10_hw).strip():
                hailo_targets_for_matrix.append(str(hailo10_hw).strip())
            for hw in hailo_targets_for_matrix:
                if bool(matrix_trt_to_hailo):
                    bench_plan_runs.append({'id': f'trt_to_{hw}', 'type': 'matrix', 'provider': 'tensorrt', 'variants': list(matrix_variants), 'image_scale': plan_image_scale, 'validation_images': plan_validation_images, 'validation_max_images': plan_validation_max_images, 'validation_reference_mode': plan_validation_reference_mode, 'mini_coco_ap50': plan_mini_coco_ap50, 'benchmark_task': plan_benchmark_task, 'mini_classification_eval': plan_mini_classification_eval, 'stage1': {'type': 'onnxruntime', 'provider': 'tensorrt'}, 'stage2': {'type': 'hailo', 'hw_arch': hw}})
                if bool(matrix_hailo_to_trt):
                    bench_plan_runs.append({'id': f'{hw}_to_trt', 'type': 'matrix', 'provider': 'tensorrt', 'variants': list(matrix_variants), 'image_scale': plan_image_scale, 'validation_images': plan_validation_images, 'validation_max_images': plan_validation_max_images, 'validation_reference_mode': plan_validation_reference_mode, 'mini_coco_ap50': plan_mini_coco_ap50, 'benchmark_task': plan_benchmark_task, 'mini_classification_eval': plan_mini_classification_eval, 'stage1': {'type': 'hailo', 'hw_arch': hw}, 'stage2': {'type': 'onnxruntime', 'provider': 'tensorrt'}})

        hailo_targets_set: Set[str] = set()
        for run in bench_plan_runs:
            if str(run.get('type') or '').strip().lower() == 'hailo':
                hw = str(run.get('hw_arch') or run.get('id') or '').strip()
                if hw:
                    hailo_targets_set.add(hw)
            for st_key in ('stage1', 'stage2'):
                st = run.get(st_key)
                if not isinstance(st, dict):
                    continue
                if str(st.get('type') or '').strip().lower() != 'hailo':
                    continue
                hw = str(st.get('hw_arch') or st.get('arch') or st.get('id') or '').strip()
                if hw:
                    hailo_targets_set.add(hw)
        hef_targets = sorted(hailo_targets_set)
        hailo_selected = bool(hef_targets)

        need_full = False
        need_part1 = False
        need_part2 = False
        for run in bench_plan_runs:
            vv = run.get('variants')
            if not isinstance(vv, list):
                continue
            vset = {str(x).strip().lower() for x in vv if str(x).strip()}
            st1 = run.get('stage1')
            st2 = run.get('stage2')
            st1_h = isinstance(st1, dict) and str(st1.get('type') or '').strip().lower() == 'hailo'
            st2_h = isinstance(st2, dict) and str(st2.get('type') or '').strip().lower() == 'hailo'
            is_hailo_run = str(run.get('type') or '').strip().lower() == 'hailo'
            if 'full' in vset and (is_hailo_run or st1_h or st2_h):
                need_full = True
            if st1_h and ('part1' in vset or 'composed' in vset):
                need_part1 = True
            if st2_h and ('part2' in vset or 'composed' in vset):
                need_part2 = True

        return BenchmarkRunPlan(
            bench_plan_runs=bench_plan_runs,
            hef_targets=hef_targets,
            hailo_selected=bool(hailo_selected),
            hef_full=bool(hailo_selected and need_full),
            hef_part1=bool(hailo_selected and need_part1),
            hef_part2=bool(hailo_selected and need_part2),
            hailo_variants=list(hailo_variants),
            matrix_variants=list(matrix_variants),
            image_scale=plan_image_scale,
            validation_images=plan_validation_images,
            validation_max_images=plan_validation_max_images,
            validation_reference_mode=plan_validation_reference_mode,
            mini_coco_ap50=bool(plan_mini_coco_ap50),
            benchmark_task=str(plan_benchmark_task),
            mini_classification_eval=bool(plan_mini_classification_eval),
        )

    def _compact_issue_detail(self, text: Any, *, limit: int = 240) -> str:
        s = " ".join(str(text or '').strip().split())
        if len(s) > int(limit):
            return s[: max(0, int(limit) - 1)].rstrip() + '…'
        return s

    def _classify_completion_issue(self, *, reason: str = '', stage: str = '', detail: str = '', kind: str = '') -> Tuple[str, str]:
        reason_l = str(reason or '').strip().lower()
        stage_l = str(stage or '').strip().lower()
        detail_l = str(detail or '').strip().lower()
        blob = ' '.join(x for x in (kind, reason_l, stage_l, detail_l) if str(x).strip())

        unsupported_info = _extract_hailo_unsupported_issue_info(detail)

        if 'hailo_failure_cluster_skip' in reason_l or 'bad boundary neighborhood' in detail_l or 'adaptive skip' in detail_l:
            return 'auto_neighbor_skip', 'Adaptive skip: bad boundary neighborhood'
        if 'unsupported dimensions at concat layer' in detail_l or 'concat layer operates on the feature dim' in detail_l:
            return 'hailo_concat_shape', 'Hailo concat/shape incompatibility'
        if 'part2 inputs are not fully produced by part1 outputs' in detail_l or 'missing from part1 outputs' in detail_l or 'needs original input' in detail_l:
            return 'hailo_part2_input_mismatch', 'Hailo Part2 input mismatch'
        if 'concat/layout sanity' in detail_l or 'concat sanity' in detail_l:
            return 'hailo_part2_concat_sanity', 'Hailo Part2 concat/layout sanity failed'
        if bool(unsupported_info.get('explicit')) and ('parser' in blob or 'translation' in blob or 'preflight' in blob):
            return 'hailo_parser_unsupported_ops', 'Hailo parser blocked by unsupported ops/activations'
        if 'parser' in blob or 'translation' in blob:
            return 'hailo_parser_translation', 'Hailo parser/translation failed'
        if ('no successful assignments' in detail_l or 'agent infeasible' in detail_l or 'mapping failed' in detail_l or 'format_conversion' in detail_l):
            if stage_l == 'part1':
                return 'hailo_part1_mapping', 'Hailo Part1 mapping/allocation failed'
            if stage_l == 'part2':
                return 'hailo_part2_mapping', 'Hailo Part2 mapping/allocation failed'
            return 'hailo_mapping', 'Hailo mapping/allocation failed'
        if reason_l.startswith('hailo_part2_'):
            return 'hailo_part2_prefilter', 'Auto-skip: Hailo Part2 incompatible'
        if 'timeout' in blob or 'timed out' in blob:
            return 'timeout', 'Timeout'
        return 'other', 'Other issues'

    def _group_completion_issue_records(self, records: Sequence[Mapping[str, Any]], *, kind_label: str) -> List[Dict[str, Any]]:
        grouped: Dict[str, Dict[str, Any]] = {}
        for rec in records or []:
            if not isinstance(rec, Mapping):
                continue
            boundary = _safe_int(rec.get('boundary'))
            reason = str(rec.get('reason') or '').strip()
            stage = str(rec.get('stage') or '').strip()
            detail = str(rec.get('detail') or rec.get('error') or '').strip()
            key, title = self._classify_completion_issue(reason=reason, stage=stage, detail=detail, kind=kind_label)
            grp = grouped.setdefault(key, {
                'key': key,
                'kind_label': str(kind_label or ''),
                'title': title,
                'count': 0,
                'examples': [],
                'examples_text': '—',
                'sample_detail': '',
            })
            grp['count'] = int(grp.get('count', 0)) + 1
            example = ''
            if boundary is not None:
                example = f'b{boundary}'
            elif str(rec.get('folder') or '').strip():
                example = str(rec.get('folder') or '').strip()
            if example and example not in grp['examples'] and len(grp['examples']) < 6:
                grp['examples'].append(example)
            if detail and not grp['sample_detail']:
                grp['sample_detail'] = self._compact_issue_detail(detail, limit=360)

        order = {
            'Hailo parser blocked by unsupported ops/activations': 0,
            'Hailo Part1 mapping/allocation failed': 0,
            'Hailo Part2 mapping/allocation failed': 1,
            'Hailo mapping/allocation failed': 2,
            'Hailo concat/shape incompatibility': 3,
            'Hailo Part2 input mismatch': 4,
            'Hailo parser/translation failed': 5,
            'Hailo Part2 concat/layout sanity failed': 6,
            'Adaptive skip: bad boundary neighborhood': 7,
            'Auto-skip: Hailo Part2 incompatible': 8,
            'Timeout': 9,
            'Other issues': 99,
        }

        out: List[Dict[str, Any]] = []
        for grp in grouped.values():
            grp['examples_text'] = ', '.join(str(x) for x in grp.get('examples', []) if str(x).strip()) or '—'
            if not grp.get('sample_detail'):
                grp['sample_detail'] = 'No additional detail recorded.'
            out.append(dict(grp))
        out.sort(key=lambda g: (order.get(str(g.get('title') or ''), 90), -int(g.get('count') or 0), str(g.get('title') or '')))
        return out

    def _group_completion_warning_lines(self, warnings: Sequence[str], *, covered_boundaries: Optional[Set[int]] = None) -> Tuple[List[Dict[str, Any]], List[str]]:
        covered_boundaries = set(int(x) for x in (covered_boundaries or set()))
        grouped: Dict[str, Dict[str, Any]] = {}
        residual_lines: List[str] = []
        for raw in warnings or []:
            line = str(raw or '').strip()
            if not line:
                continue
            m = re.match(r'^b(\d+)\s*:', line, flags=re.IGNORECASE)
            if m is not None:
                try:
                    if int(m.group(1)) in covered_boundaries:
                        continue
                except Exception:
                    pass
            residual_lines.append(line)
            key, title = self._classify_completion_issue(detail=line, kind='warning')
            grp = grouped.setdefault(key, {
                'key': key,
                'kind_label': 'Warning',
                'title': title,
                'count': 0,
                'examples': [],
                'examples_text': '—',
                'sample_detail': '',
            })
            grp['count'] = int(grp.get('count', 0)) + 1
            example = ''
            if m is not None:
                example = f"b{m.group(1)}"
            elif len(grp['examples']) < 4:
                example = self._compact_issue_detail(line, limit=80)
            if example and example not in grp['examples'] and len(grp['examples']) < 6:
                grp['examples'].append(example)
            if not grp['sample_detail']:
                grp['sample_detail'] = self._compact_issue_detail(line, limit=360)

        out: List[Dict[str, Any]] = []
        for grp in grouped.values():
            grp['examples_text'] = ', '.join(str(x) for x in grp.get('examples', []) if str(x).strip()) or '—'
            if not grp.get('sample_detail'):
                grp['sample_detail'] = 'No additional detail recorded.'
            out.append(dict(grp))
        out.sort(key=lambda g: (-int(g.get('count') or 0), str(g.get('title') or '')))
        return out, residual_lines

    def build_completion_summary_data(
        self,
        *,
        out_dir: str | Path,
        harness_path: str | Path,
        bench_log_path: str | Path,
        requested_cases: int,
        accepted_count: int,
        preferred_shortlist_count: int,
        candidate_search_pool_count: int,
        benign_discarded: Sequence[Mapping[str, Any]],
        rejected_discarded: Sequence[Mapping[str, Any]],
        shortlist_prefiltered_count: int = 0,
        backfilled_cases_count: int = 0,
        resume_generation: bool = False,
        resume_lines: Optional[Sequence[str]] = None,
        plan_run_ids: Optional[Sequence[str]] = None,
        errors: Optional[Sequence[str]] = None,
        hailo_selected: bool = False,
        hailo_outlook_summary: Optional[HailoCompileOutlookSummary] = None,
        top_hailo_boundaries: Optional[Sequence[int]] = None,
        hailo_full_model_preflight: Optional[Mapping[str, Any]] = None,
        full_model_preflight_policy: str = 'enabled',
    ) -> Dict[str, Any]:
        out_dir = str(out_dir)
        harness_path = str(harness_path)
        bench_log_path = str(bench_log_path)
        shortfall = max(0, int(requested_cases) - int(accepted_count))
        benign_groups = self._group_completion_issue_records(benign_discarded, kind_label='Auto-skip')
        rejected_groups = self._group_completion_issue_records(rejected_discarded, kind_label='Rejected')
        rejected_boundaries = {int(x) for x in (_safe_int(rec.get('boundary')) for rec in rejected_discarded) if x is not None}
        warning_groups, residual_warning_lines = self._group_completion_warning_lines(list(errors or []), covered_boundaries=rejected_boundaries)

        issue_groups: List[Dict[str, Any]] = []
        issue_groups.extend(rejected_groups)
        issue_groups.extend(benign_groups)
        issue_groups.extend(warning_groups)

        run_ids = [str(x).strip() for x in (plan_run_ids or []) if str(x).strip()]
        top_boundaries = [int(x) for x in (top_hailo_boundaries or []) if _safe_int(x) is not None]

        hailo_outlook: Optional[Dict[str, Any]] = None
        if bool(hailo_selected) and hailo_outlook_summary is not None:
            hailo_outlook = {
                'top_boundary': int(hailo_outlook_summary.top_boundary) if hailo_outlook_summary.top_boundary is not None else None,
                'candidate_count': int(hailo_outlook_summary.candidate_count),
                'likely_single_context_count': int(hailo_outlook_summary.likely_single_context_count),
                'low_risk_count': int(hailo_outlook_summary.low_risk_count),
                'medium_risk_count': int(hailo_outlook_summary.medium_risk_count),
                'high_risk_count': int(hailo_outlook_summary.high_risk_count),
            }

        return {
            'out_dir': out_dir,
            'harness_path': harness_path,
            'bench_log_path': bench_log_path,
            'requested_cases': int(requested_cases),
            'accepted_count': int(accepted_count),
            'shortfall': int(shortfall),
            'preferred_shortlist_count': int(preferred_shortlist_count),
            'candidate_search_pool_count': int(candidate_search_pool_count),
            'benign_count': int(len(list(benign_discarded or []))),
            'rejected_count': int(len(list(rejected_discarded or []))),
            'warning_count': int(len([str(x) for x in (errors or []) if str(x).strip()])),
            'extra_warning_count': int(len(residual_warning_lines)),
            'extra_warning_lines': list(residual_warning_lines),
            'shortlist_prefiltered_count': int(shortlist_prefiltered_count),
            'backfilled_cases_count': int(backfilled_cases_count),
            'resume_generation': bool(resume_generation),
            'resume_lines': [str(x) for x in list(resume_lines or []) if str(x).strip()],
            'plan_run_ids': list(run_ids),
            'hailo_selected': bool(hailo_selected),
            'hailo_outlook': hailo_outlook,
            'top_hailo_boundaries': list(top_boundaries),
            'hailo_full_model_preflight': (dict(hailo_full_model_preflight) if isinstance(hailo_full_model_preflight, Mapping) else None),
            'full_model_preflight_policy': normalize_hailo_full_model_preflight_policy(full_model_preflight_policy),
            'auto_skip_groups': list(benign_groups),
            'rejection_groups': list(rejected_groups),
            'warning_groups': list(warning_groups),
            'issue_groups': list(issue_groups),
        }

    def format_completion_summary_text(self, summary: Mapping[str, Any], *, verbose: bool = False) -> str:
        final_status = str(summary.get('final_status') or 'warn').strip().lower()
        accepted_count = int(summary.get('accepted_count') or 0)
        requested_cases = int(summary.get('requested_cases') or 0)
        shortfall = int(summary.get('shortfall') or 0)
        benign_count = int(summary.get('benign_count') or 0)
        rejected_count = int(summary.get('rejected_count') or 0)
        extra_warning_count = int(summary.get('extra_warning_count') or 0)
        preferred_shortlist_count = int(summary.get('preferred_shortlist_count') or 0)
        candidate_search_pool_count = int(summary.get('candidate_search_pool_count') or 0)
        shortlist_prefiltered_count = int(summary.get('shortlist_prefiltered_count') or 0)
        backfilled_cases_count = int(summary.get('backfilled_cases_count') or 0)
        hailo_selected = bool(summary.get('hailo_selected'))
        hailo_outlook = summary.get('hailo_outlook') if isinstance(summary.get('hailo_outlook'), Mapping) else None
        top_hailo_boundaries = [int(x) for x in (summary.get('top_hailo_boundaries') or []) if _safe_int(x) is not None]
        accepted_hailo_feasible_boundaries = [int(x) for x in (summary.get('accepted_hailo_feasible_boundaries') or []) if _safe_int(x) is not None]
        hailo_full_model_preflight = summary.get('hailo_full_model_preflight') if isinstance(summary.get('hailo_full_model_preflight'), Mapping) else None
        issue_groups = list(summary.get('issue_groups') or [])
        out_dir = str(summary.get('out_dir') or '')
        harness_path = str(summary.get('harness_path') or '')
        bench_log_path = str(summary.get('bench_log_path') or '')
        resume_lines = [str(x) for x in list(summary.get('resume_lines') or []) if str(x).strip()]
        plan_run_ids = [str(x) for x in list(summary.get('plan_run_ids') or []) if str(x).strip()]
        cancellation_reason = str(summary.get('cancellation_reason') or '').strip()
        plan_adjustments = [str(x) for x in list(summary.get('plan_adjustments') or []) if str(x).strip()]
        extra_warning_lines = [str(x) for x in list(summary.get('extra_warning_lines') or []) if str(x).strip()]

        if not verbose:
            if final_status == 'ok':
                lines = [
                    'Benchmark set created successfully',
                    f'Accepted cases: {accepted_count}/{requested_cases}',
                ]
            elif final_status == 'cancelled':
                lines = [
                    'Benchmark-set generation cancelled',
                    f'Accepted cases so far: {accepted_count}/{requested_cases}',
                ]
                if cancellation_reason:
                    lines.append(cancellation_reason)
            else:
                lines = [
                    'Benchmark set created with warnings',
                    f'Accepted cases: {accepted_count}/{requested_cases}',
                ]
            if shortfall > 0:
                lines.append(f'Shortfall: {shortfall}')
            lines.append(f'Auto-skipped: {benign_count}')
            lines.append(f'Rejected: {rejected_count}')
            if extra_warning_count > 0:
                lines.append(f'Other warnings: {extra_warning_count}')
            if hailo_selected and accepted_hailo_feasible_boundaries:
                lines.append('Accepted Hailo-feasible cases: ' + ', '.join(f'b{int(b)}' for b in accepted_hailo_feasible_boundaries[:6]))
            if hailo_selected and hailo_outlook is not None:
                top_boundary = hailo_outlook.get('top_boundary')
                lines.append(
                    'Hailo outlook: '
                    f"top=b{top_boundary if top_boundary is not None else '?'} | "
                    f"risk low/med/high={hailo_outlook.get('low_risk_count', 0)}/"
                    f"{hailo_outlook.get('medium_risk_count', 0)}/"
                    f"{hailo_outlook.get('high_risk_count', 0)} | "
                    f"likely single-context={hailo_outlook.get('likely_single_context_count', 0)}/"
                    f"{hailo_outlook.get('candidate_count', 0)}"
                )
            if hailo_full_model_preflight and bool(hailo_full_model_preflight.get('skipped')):
                lines.append('Hailo parser preflight: skipped (full-model HEF build will be attempted directly)')
            elif hailo_full_model_preflight and bool(hailo_full_model_preflight.get('checked')):
                if bool(hailo_full_model_preflight.get('aborted')):
                    status_txt = 'blocked generation'
                elif bool(hailo_full_model_preflight.get('plan_adjusted')):
                    status_txt = 'adjusted plan and continued'
                else:
                    status_txt = 'completed'
                lines.append(
                    'Hailo parser preflight: '
                    f"{int(hailo_full_model_preflight.get('ok_count') or 0)}/"
                    f"{int(hailo_full_model_preflight.get('result_count') or 0)} targets passed | {status_txt}"
                )
            if out_dir:
                lines.append(f'Benchmark folder: {out_dir}')
            return '\n'.join(lines)

        lines: List[str] = []
        if final_status == 'ok':
            lines.append('Benchmark set created successfully.')
        elif final_status == 'cancelled':
            lines.append('Benchmark-set generation cancelled.')
            if cancellation_reason:
                lines.append(cancellation_reason)
            lines.append('The partial benchmark set was kept on disk and can be resumed later from the Benchmark tab.')
        else:
            lines.append('Benchmark set created with warnings.')
        lines.append('')
        lines.append(f'Accepted cases: {accepted_count}/{requested_cases}')
        lines.append(f'Preferred shortlist: {preferred_shortlist_count}')
        lines.append(f'Candidate search pool: {candidate_search_pool_count}')
        eval_profile = summary.get('evaluation_profile') if isinstance(summary.get('evaluation_profile'), dict) else {}
        if eval_profile:
            prof_id = str(eval_profile.get('profile_id') or eval_profile.get('requested') or '').strip()
            matched = str(eval_profile.get('matched_model_id') or '').strip()
            if prof_id:
                lines.append(f"Evaluation profile: {prof_id}" + (f" -> {matched}" if matched else ''))
        if shortfall > 0:
            lines.append(f'Shortfall: {shortfall}')
        lines.append(f'Auto-skipped candidates: {benign_count}')
        if shortlist_prefiltered_count > 0:
            lines.append(f'  - from preferred shortlist: {shortlist_prefiltered_count}')
        if backfilled_cases_count > 0:
            lines.append(f'  - backfilled from deeper ranked candidates: {backfilled_cases_count}')
        lines.append(f'Rejected splits: {rejected_count}')
        if extra_warning_count > 0:
            lines.append(f'Additional warnings: {extra_warning_count}')

        if hailo_selected and hailo_outlook is not None:
            top_boundary = hailo_outlook.get('top_boundary')
            lines.append('')
            lines.append('Hailo compile outlook:')
            lines.append(f"  - top candidate: b{top_boundary if top_boundary is not None else '?'}")
            lines.append(
                '  - risk bands low/med/high='
                f"{hailo_outlook.get('low_risk_count', 0)}/"
                f"{hailo_outlook.get('medium_risk_count', 0)}/"
                f"{hailo_outlook.get('high_risk_count', 0)}"
            )
            lines.append(
                '  - likely single-context='
                f"{hailo_outlook.get('likely_single_context_count', 0)}/"
                f"{hailo_outlook.get('candidate_count', 0)}"
            )
            if top_hailo_boundaries:
                lines.append('  - top Hailo candidates before endpoint/full-baseline policy: ' + ', '.join(f'b{int(b)}' for b in top_hailo_boundaries[:6]))
            if accepted_hailo_feasible_boundaries:
                lines.append('  - accepted Hailo-feasible cases after policy: ' + ', '.join(f'b{int(b)}' for b in accepted_hailo_feasible_boundaries[:8]))

        if hailo_full_model_preflight and bool(hailo_full_model_preflight.get('skipped')):
            lines.append('')
            lines.append('Hailo parser preflight:')
            lines.append('  - status: skipped (full-model HEF build will be attempted directly)')
        elif hailo_full_model_preflight and bool(hailo_full_model_preflight.get('checked')):
            lines.append('')
            lines.append('Hailo parser preflight:')
            if bool(hailo_full_model_preflight.get('aborted')):
                lines.append('  - status: blocked benchmark generation before the candidate loop')
            elif bool(hailo_full_model_preflight.get('plan_adjusted')):
                lines.append('  - status: warnings (plan-aware preflight adjustment applied; generation continued)')
            elif int(hailo_full_model_preflight.get('failed_count') or 0) > 0:
                lines.append('  - status: warnings (generation continued despite failed Hailo targets)')
            else:
                lines.append('  - status: OK')
            lines.append(
                '  - targets ok/total='
                f"{int(hailo_full_model_preflight.get('ok_count') or 0)}/"
                f"{int(hailo_full_model_preflight.get('result_count') or 0)}"
            )
            for entry in list(hailo_full_model_preflight.get('results') or [])[:6]:
                if not isinstance(entry, Mapping):
                    continue
                hw_arch = str(entry.get('hw_arch') or '?')
                status_txt = 'OK' if bool(entry.get('ok')) else 'FAILED'
                detail_parts: List[str] = []
                unsupported_ops = [str(x).strip() for x in list(entry.get('unsupported_ops') or []) if str(x).strip()]
                unsupported_nodes = [str(x).strip() for x in list(entry.get('unsupported_nodes') or []) if str(x).strip()]
                unsupported_scope = str(entry.get('unsupported_scope') or '').strip()
                if unsupported_scope:
                    detail_parts.append('scope=' + unsupported_scope)
                if unsupported_ops:
                    detail_parts.append('ops=' + ', '.join(unsupported_ops[:4]))
                if unsupported_nodes:
                    preview = ', '.join(unsupported_nodes[:3])
                    if len(unsupported_nodes) > 3:
                        preview += ', …'
                    detail_parts.append('nodes=' + preview)
                error_text = str(entry.get('error') or '').strip()
                if not detail_parts and error_text:
                    detail_parts.append(self._compact_issue_detail(error_text, limit=200))
                detail_txt = f" | {'; '.join(detail_parts)}" if detail_parts else ''
                lines.append(f'  - {hw_arch}: {status_txt}{detail_txt}')

        lines.append('')
        lines.append('Paths:')
        if out_dir:
            lines.append(f'  - Benchmark folder: {out_dir}')
        if harness_path:
            lines.append(f'  - Harness: {harness_path}')
        if bench_log_path:
            lines.append(f'  - Generation log: {bench_log_path}')

        if plan_run_ids:
            lines.append('')
            lines.append('Plan runs: ' + ', '.join(plan_run_ids))

        if resume_lines:
            lines.append('')
            lines.append('Resume info:')
            for line in resume_lines[:8]:
                lines.append(f'  - {line}')

        if plan_adjustments:
            lines.append('')
            lines.append('Plan adjustments:')
            for line in plan_adjustments:
                lines.append(f'  - {line}')

        if issue_groups:
            lines.append('')
            lines.append('Main issues:')
            for grp in issue_groups:
                title = str(grp.get('title') or 'Issue')
                kind_label = str(grp.get('kind_label') or '').strip()
                prefix = f'{kind_label}: ' if kind_label else ''
                lines.append(f"  - {prefix}{title} ({int(grp.get('count') or 0)})")
                examples_text = str(grp.get('examples_text') or '').strip()
                if examples_text and examples_text != '—':
                    lines.append(f'    examples: {examples_text}')
                detail = str(grp.get('sample_detail') or '').strip()
                if detail:
                    lines.append(f'    detail: {detail}')

        if extra_warning_lines:
            lines.append('')
            lines.append('Additional warning lines:')
            for line in extra_warning_lines[:10]:
                lines.append(f"  - {self._compact_issue_detail(line, limit=320)}")
            if len(extra_warning_lines) > 10:
                lines.append(f'  ... and {len(extra_warning_lines) - 10} more')

        return '\n'.join(lines)

    def build_completion_summary(
        self,
        *,
        out_dir: str | Path,
        harness_path: str | Path,
        bench_log_path: str | Path,
        requested_cases: int,
        accepted_count: int,
        preferred_shortlist_count: int,
        candidate_search_pool_count: int,
        benign_discarded: Sequence[Mapping[str, Any]],
        rejected_discarded: Sequence[Mapping[str, Any]],
        shortlist_prefiltered_count: int = 0,
        backfilled_cases_count: int = 0,
        resume_generation: bool = False,
        resume_lines: Optional[Sequence[str]] = None,
        plan_run_ids: Optional[Sequence[str]] = None,
        errors: Optional[Sequence[str]] = None,
        hailo_selected: bool = False,
        hailo_outlook_summary: Optional[HailoCompileOutlookSummary] = None,
        top_hailo_boundaries: Optional[Sequence[int]] = None,
        hailo_full_model_preflight: Optional[Mapping[str, Any]] = None,
    ) -> str:
        summary = self.build_completion_summary_data(
            out_dir=out_dir,
            harness_path=harness_path,
            bench_log_path=bench_log_path,
            requested_cases=requested_cases,
            accepted_count=accepted_count,
            preferred_shortlist_count=preferred_shortlist_count,
            candidate_search_pool_count=candidate_search_pool_count,
            benign_discarded=benign_discarded,
            rejected_discarded=rejected_discarded,
            shortlist_prefiltered_count=shortlist_prefiltered_count,
            backfilled_cases_count=backfilled_cases_count,
            resume_generation=resume_generation,
            resume_lines=resume_lines,
            plan_run_ids=plan_run_ids,
            errors=errors,
            hailo_selected=hailo_selected,
            hailo_outlook_summary=hailo_outlook_summary,
            top_hailo_boundaries=top_hailo_boundaries,
            hailo_full_model_preflight=hailo_full_model_preflight,
        )
        return self.format_completion_summary_text(summary, verbose=False)




@dataclass
class BenchmarkGenerationExecutionConfig:
    runtime: BenchmarkGenerationRuntime
    target_cases: int
    gap: int
    ranked_candidates: List[int]
    candidate_search_pool: List[int]
    out_dir: Path
    base: str
    pad: int
    strict_boundary: bool
    model: Any
    nodes: Any
    order: Any
    analysis_payload: Mapping[str, Any]
    analysis_candidates: Sequence[Mapping[str, Any]] = field(default_factory=list)
    bench_plan_runs: Sequence[Mapping[str, Any]] = field(default_factory=list)
    runner_target: str = "auto"
    do_ctx_full: bool = False
    do_ctx_cutflow: bool = False
    ctx_hops: int = 2
    llm_style: bool = False
    value_bytes_map: Any = None
    full_model_src: str = ""
    full_model_dst: str = ""
    tool_gui_version: str = "?"
    tool_core_version: str = "?"
    hailo_compile_rank_meta: Mapping[int, Dict[str, Any]] = field(default_factory=dict)
    hef_targets: List[str] = field(default_factory=list)
    hef_part1: bool = False
    hef_part2: bool = False
    hef_backend: str = "dataflow_compiler"
    hef_fixup: bool = False
    hef_opt_level: int = 2
    hef_calib_dir: Optional[str] = None
    hef_calib_count: int = 64
    hef_calib_bs: int = 1
    hef_force: bool = False
    hef_keep: bool = False
    hef_wsl_distro: Optional[str] = None
    hef_wsl_venv: str = ""
    hef_timeout_s: int = 0
    hailo_full_end_node_names: Sequence[str] = field(default_factory=list)
    hailo_full_endpoint_mode: str = ''
    hailo_full_output_contract: Optional[Mapping[str, Any]] = None
    hailo_build_hef_fn: Any = None
    hailo_build_unavailable: Optional[str] = None
    hailo_part2_precheck_fn: Any = None
    hailo_part2_precheck_error_fn: Any = None
    hailo_part2_parser_precheck_fn: Any = None
    hailo_part2_parser_precheck_error_fn: Any = None
    hailo_part2_enable_suggested_endnode_fallback: bool = True
    hailo_part2_concat_sanity_enable: bool = True
    hailo_salvage_enable: bool = True
    hailo_salvage_neighbor_radius: int = 48
    evaluation_profile_meta: Optional[Mapping[str, Any]] = None
    should_cancel: Optional[Callable[[], bool]] = None


@dataclass
class BenchmarkGenerationExecutionCallbacks:
    log: Callable[[str], None]
    queue_put: Callable[[tuple], None]
    persist_state: Callable[..., None]
    publish_hailo_diagnostics: Callable[[str, Any, Any], None]
    predicted_metrics_for_boundary: Callable[[Mapping[str, Any], int], Dict[str, Any]]
    hailo_parse_entry_for_boundary: Callable[[Mapping[str, Any], int], Any]
    hailo_parse_scalar_fields: Callable[[Any], Dict[str, Any]]


class BenchmarkGenerationCancelled(RuntimeError):
    """Raised when benchmark-set generation is cancelled by the user."""


class BenchmarkGenerationExecutionService:
    """Execute the per-boundary case-build loop for benchmark-set generation.

    The GUI still owns user interaction and top-level orchestration, but the heavy
    split/build/rejection loop now lives here so it can be tested headlessly and
    reused by future CLI entry points.
    """

    def __init__(self, generation_service: Optional[BenchmarkGenerationService] = None):
        self.generation_service = generation_service or BenchmarkGenerationService()

    def _record_hailo_part2_filter(self, boundary: int, *, pad: int, detail: str, target_label: str, reason: str, info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        rec = build_benchmark_case_rejection(
            boundary=int(boundary),
            folder=f"b{int(boundary):0{pad}d}",
            reason=str(reason),
            stage="part2",
            hw_arch=target_label,
            detail=detail,
        )
        rec["prefiltered"] = (reason == "hailo_part2_prefilter")
        if isinstance(info, dict):
            if info.get("missing_inputs"):
                rec["missing_inputs"] = list(info.get("missing_inputs") or [])
            if info.get("likely_original_inputs"):
                rec["likely_original_inputs"] = list(info.get("likely_original_inputs") or [])
            if info.get("blocked_ops"):
                rec["blocked_ops"] = list(info.get("blocked_ops") or [])
            if info.get("blocked_nodes"):
                rec["blocked_nodes"] = list(info.get("blocked_nodes") or [])
            if info.get("blocked_prefix"):
                rec["blocked_prefix"] = str(info.get("blocked_prefix") or "")
            if info.get("suggested_end_nodes"):
                rec["suggested_end_nodes"] = list(info.get("suggested_end_nodes") or [])
            if info.get("node_name"):
                rec["node_name"] = str(info.get("node_name") or "")
            if info.get("input_shapes"):
                rec["input_shapes"] = list(info.get("input_shapes") or [])
            if info.get("mismatched_dims"):
                rec["mismatched_dims"] = list(info.get("mismatched_dims") or [])
            if info.get("part2_outputs"):
                rec["part2_outputs"] = list(info.get("part2_outputs") or [])
        return rec

    def _maybe_publish_diag(self, callbacks: BenchmarkGenerationExecutionCallbacks, label: str, result: Any, log_cb: Callable[[str], None]) -> None:
        try:
            callbacks.publish_hailo_diagnostics(label, result, log_cb)
        except Exception:
            logger.debug("Could not publish Hailo diagnostics for %s", label, exc_info=True)

    def _extract_row_per_cut_hints(self, result: Any) -> List[str]:
        try:
            details = getattr(result, 'details', None)
            process_summary = details.get('process_summary') if isinstance(details, dict) else None
            hints = [str(x).strip() for x in list((process_summary or {}).get('row_per_cut_hints') or []) if str(x).strip()]
            seen: Set[str] = set()
            out: List[str] = []
            for name in hints:
                if name not in seen:
                    out.append(name)
                    seen.add(name)
            return out
        except Exception:
            return []

    def _extract_validator_failed_nodes(self, result: Any) -> List[str]:
        try:
            details = getattr(result, 'details', None)
            process_summary = details.get('process_summary') if isinstance(details, dict) else None
            names = [str(x).strip() for x in list((process_summary or {}).get('validator_failed_nodes') or []) if str(x).strip()]
            seen: Set[str] = set()
            out: List[str] = []
            for name in names:
                if name not in seen:
                    out.append(name)
                    seen.add(name)
            return out
        except Exception:
            return []

    def _nearest_row_per_cut_donor(self, boundary: int, donors: Mapping[int, Sequence[str]], *, radius: int) -> Tuple[Optional[int], List[str]]:
        best_boundary: Optional[int] = None
        best_distance: Optional[int] = None
        best_hints: List[str] = []
        for raw_b, raw_hints in dict(donors or {}).items():
            try:
                donor_b = int(raw_b)
            except Exception:
                continue
            dist = abs(int(boundary) - donor_b)
            if dist <= 0 or dist > int(max(0, radius)):
                continue
            hints = [str(x).strip() for x in list(raw_hints or []) if str(x).strip()]
            if not hints:
                continue
            if best_distance is None or dist < best_distance or (dist == best_distance and donor_b > int(best_boundary or -1)):
                best_boundary = donor_b
                best_distance = dist
                best_hints = hints
        return best_boundary, best_hints

    def _build_row_per_cut_salvage_script(self, hints: Sequence[str]) -> str:
        # The nearby successful build tells us this boundary family benefits from
        # ROW_PER_CUT-like buffering. We keep the retry conservative and portable: a
        # single model-script performance hint that can help the compiler search a
        # broader mapping space without relying on unsupported per-node script syntax.
        _ = [str(x).strip() for x in list(hints or []) if str(x).strip()]
        return "performance_param(compiler_optimization_level=max)\n"

    def _should_attempt_hailo_salvage(
        self,
        *,
        failure_rec: Optional[HailoFailureRecord],
        validator_nodes: Sequence[str],
        donor_hints: Sequence[str],
        stage: str,
        hw_arch: str,
    ) -> bool:
        if str(stage or '').strip().lower() != 'part1':
            return False
        if str(hw_arch or '').strip().lower() != 'hailo8':
            return False
        if not donor_hints:
            return False
        family = str(getattr(failure_rec, 'family', '') or '').strip().lower()
        if family in {'format_conversion_agent_infeasible', 'validator_concat', 'validator_defuse', 'feature_splitter_agent_infeasible'}:
            return True
        nodes = {str(x).strip().lower() for x in list(validator_nodes or []) if str(x).strip()}
        if nodes.intersection({'concat3', 'dw1_defuse_1x1', 'dw2_defuse_1x1'}):
            return True
        detail = str(getattr(failure_rec, 'detail', '') or '').lower() if failure_rec is not None else ''
        return ('agent infeasible' in detail and ('format_conversion' in detail or 'validator failed on node' in detail))

    def _prune_unused_graph_inputs_for_hailo_prefix(self, model: Any) -> Tuple[Any, List[str]]:
        """Drop graph.input entries that no node actually consumes.

        Part2 accelerator-prefix models can be a backward slice of a larger
        Part2 ONNX.  The generic splitter preserves all original Part2 inputs for
        Part1-like submodels, but after truncating before YOLO DFL/decode only a
        subset may be needed.  ONNX Runtime tolerates these dangling inputs;
        Hailo DFC does not and reports: "Couldn't find inputs from ONNX proto".
        """
        try:
            import onnx  # type: ignore
            copied = onnx.ModelProto()
            copied.CopyFrom(model)
            g = copied.graph
            init_names = {str(getattr(t, 'name', '') or '') for t in getattr(g, 'initializer', [])}
            try:
                init_names.update(str(getattr(t, 'name', '') or '') for t in getattr(g, 'sparse_initializer', []))
            except Exception:
                pass
            consumed: Set[str] = set()
            for node in getattr(g, 'node', []) or []:
                for name in getattr(node, 'input', []) or []:
                    if name:
                        consumed.add(str(name))
            graph_outputs = {str(getattr(o, 'name', '') or '') for o in getattr(g, 'output', []) or []}
            keep = []
            removed: List[str] = []
            for vi in list(getattr(g, 'input', []) or []):
                name = str(getattr(vi, 'name', '') or '')
                if (not name) or name in init_names or name in consumed or name in graph_outputs:
                    keep.append(vi)
                else:
                    removed.append(name)
            if removed:
                del g.input[:]
                g.input.extend(keep)
            return copied, removed
        except Exception:
            logger.debug("Could not prune unused graph inputs for Hailo prefix", exc_info=True)
            return model, []

    def _part2_output_sets_from_suggested_end_nodes(self, nodes: Sequence[Any], parser_info: Optional[Mapping[str, Any]]) -> List[List[str]]:
        suggested_nodes = [str(x).strip() for x in list((parser_info or {}).get('suggested_end_nodes') or []) if str(x).strip()]
        if not suggested_nodes:
            return []
        return self._part2_output_sets_from_end_node_names(nodes, suggested_nodes)

    def _is_yolo26_cfg(self, cfg: Any) -> bool:
        """Best-effort YOLO26 family detection for Hailo endpoint policy.

        YOLO26 exports use an end-to-end / one-to-one head.  Hailo parser
        suggested end nodes such as ``/model.23/Transpose_output_0`` can pass the
        cheap parser precheck, but the real allocator still fails later at
        ``/model.23/Concat_*`` layout/format-conversion nodes.  Keep this helper
        deliberately conservative and based on file/model names and optional
        metadata so it does not affect unrelated detectors.
        """
        raw_values: List[str] = []
        for attr in ('base', 'model_name', 'model_id', 'full_model_src', 'full_model_dst'):
            try:
                value = getattr(cfg, attr, '')
            except Exception:
                value = ''
            if value not in (None, ''):
                raw_values.append(str(value))
        for attr in ('export_metadata', 'model_metadata', 'metadata'):
            try:
                meta = getattr(cfg, attr, None)
            except Exception:
                meta = None
            if isinstance(meta, Mapping):
                for key in ('family', 'model_ref', 'model_name', 'name', 'source_model'):
                    val = meta.get(key)
                    if val not in (None, ''):
                        raw_values.append(str(val))
        joined = ' '.join(raw_values).lower().replace('\\', '/').replace('-', '').replace('_', '')
        return 'yolo26' in joined

    def _raw_detection_head_end_node_names_for_cfg(self, cfg: Any) -> List[str]:
        """Return YOLO raw detection-head Conv node names for a full-model/raw-head deployment.

        Prepared YOLO11 baselines already carry these nodes.  When they are not
        present we try the same lightweight inference helper used by model
        preparation.  Non-YOLO models simply return an empty list.
        """
        contract = (
            dict(getattr(cfg, 'hailo_full_output_contract', None) or {})
            if isinstance(getattr(cfg, 'hailo_full_output_contract', None), Mapping)
            else {}
        )
        explicit: List[str] = []
        for raw in list(getattr(cfg, 'hailo_full_end_node_names', None) or contract.get('end_node_names') or []):
            name = str(raw or '').strip()
            if name and name not in explicit:
                explicit.append(name)
        if explicit:
            return explicit

        cache: Dict[str, List[str]] = getattr(self, '_raw_detection_head_end_node_cache', {}) or {}
        for raw_path in (getattr(cfg, 'full_model_dst', None), getattr(cfg, 'full_model_src', None)):
            model_path = str(raw_path or '').strip()
            if not model_path:
                continue
            if model_path in cache:
                return list(cache.get(model_path) or [])
            try:
                p = Path(model_path).expanduser()
                if not p.is_file():
                    continue
                from .model_preparation import infer_yolo_raw_detection_head_end_nodes
                nodes = [str(x).strip() for x in list(infer_yolo_raw_detection_head_end_nodes(p, '') or []) if str(x).strip()]
                cache[model_path] = list(nodes)
                try:
                    setattr(self, '_raw_detection_head_end_node_cache', cache)
                except Exception:
                    pass
                if nodes:
                    return nodes
            except Exception:
                logger.debug("Could not infer raw detection-head nodes for %s", model_path, exc_info=True)
                cache[model_path] = []
                try:
                    setattr(self, '_raw_detection_head_end_node_cache', cache)
                except Exception:
                    pass
        return []

    def _part2_output_sets_from_end_node_names(self, nodes: Sequence[Any], end_node_names: Sequence[str]) -> List[List[str]]:
        """Map ONNX node names (or tensor names) to candidate part2 output tensor sets."""
        wanted = [str(x).strip() for x in list(end_node_names or []) if str(x).strip()]
        if not wanted:
            return []
        node_outputs: Dict[str, List[str]] = {}
        known_tensors: Set[str] = set()
        for node in list(nodes or []):
            name = str(getattr(node, 'name', '') or '').strip()
            outs = [str(x).strip() for x in list(getattr(node, 'output', []) or []) if str(x).strip()]
            for out_name in outs:
                known_tensors.add(out_name)
            if name and outs:
                node_outputs[name] = outs

        sets: List[List[str]] = []
        seen_keys: Set[Tuple[str, ...]] = set()

        def _add(output_names: Sequence[str]) -> None:
            seq: List[str] = []
            seen_local: Set[str] = set()
            for raw in list(output_names or []):
                name = str(raw or '').strip()
                if name and name not in seen_local:
                    seq.append(name)
                    seen_local.add(name)
            if not seq:
                return
            key = tuple(seq)
            if key in seen_keys:
                return
            seen_keys.add(key)
            sets.append(seq)

        merged: List[str] = []
        for node_or_tensor_name in wanted:
            if node_or_tensor_name in node_outputs:
                merged.extend(node_outputs.get(node_or_tensor_name, []))
            elif node_or_tensor_name in known_tensors:
                merged.append(node_or_tensor_name)
        _add(merged)

        # Keep the single-node variants for parser suggested end nodes.  For YOLO
        # raw-head fallback the merged six-output set is the normal path, but
        # singletons are harmless as last-resort probes and can expose useful logs.
        for node_or_tensor_name in wanted:
            if node_or_tensor_name in node_outputs:
                _add(node_outputs.get(node_or_tensor_name, []))
            elif node_or_tensor_name in known_tensors:
                _add([node_or_tensor_name])
        return sets

    def _part2_output_sets_from_raw_detection_heads(self, cfg: Any) -> Tuple[List[List[str]], List[str]]:
        raw_end_nodes = self._raw_detection_head_end_node_names_for_cfg(cfg)
        if not raw_end_nodes:
            return [], []
        all_sets = self._part2_output_sets_from_end_node_names(getattr(cfg, 'nodes', None) or [], raw_end_nodes)
        if not all_sets:
            return [], raw_end_nodes

        # Raw YOLO head mode needs the complete cv2/cv3 head output set.  A
        # singleton raw-head branch would force the host tail to recompute the
        # remaining branches from the original Part2 inputs, which is technically
        # runnable but not a clean Hailo-prefix + host-tail deployment.
        expected = len([str(x).strip() for x in list(raw_end_nodes or []) if str(x).strip()])
        complete_sets = [list(s) for s in all_sets if len(list(s or [])) >= max(1, expected)]
        if complete_sets:
            return [complete_sets[0]], raw_end_nodes
        return [], raw_end_nodes

    def probe_hailo_part2_support(self, cfg: BenchmarkGenerationExecutionConfig, boundary: int, *, log_cb: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
        from .. import api as asc

        b = int(boundary)
        out: Dict[str, Any] = {
            'inspect_ok': False,
            'compatible': None,
            'boundary': int(b),
            'strategy': 'original',
            'used_suggested_end_nodes': False,
        }
        cut_tensors = asc.cut_tensors_for_boundary(cfg.order, cfg.nodes, b)
        if not cut_tensors:
            out.update({'inspect_ok': True, 'compatible': False, 'reason': 'empty_cut', 'detail': 'No cut tensors'})
            return out

        def _run_split(part2_output_names: Optional[Sequence[str]] = None) -> Tuple[Any, Any, Dict[str, Any], Any, Any, Optional[Dict[str, Any]]]:
            p1_local, p2_local, manifest_local = asc.split_model_on_cut_tensors(
                cfg.model,
                cut_tensors=list(cut_tensors),
                strict_boundary=bool(cfg.strict_boundary),
                part2_output_names=(list(part2_output_names) if part2_output_names is not None else None),
            )
            activation_info_local = None
            if cfg.hailo_part2_precheck_fn is not None and isinstance(manifest_local, dict):
                activation_info_local = cfg.hailo_part2_precheck_fn(manifest_local)
            parser_info_local = None
            if cfg.hailo_part2_parser_precheck_fn is not None:
                try:
                    parser_info_local = cfg.hailo_part2_parser_precheck_fn(p2_local, split_manifest=manifest_local if isinstance(manifest_local, dict) else None)
                except TypeError:
                    parser_info_local = cfg.hailo_part2_parser_precheck_fn(p2_local)
            concat_sanity_local = None
            if bool(getattr(cfg, 'hailo_part2_concat_sanity_enable', True)):
                concat_sanity_local = hailo_part2_concat_sanity_from_model(
                    p2_local,
                    split_manifest=(manifest_local if isinstance(manifest_local, dict) else None),
                )
            return p1_local, p2_local, manifest_local, activation_info_local, parser_info_local, concat_sanity_local

        try:
            p1, p2, split_manifest, activation_info, parser_info, concat_sanity_info = _run_split()
        except Exception as exc:
            out.update({'inspect_ok': False, 'compatible': False, 'reason': 'split_failed', 'detail': f'{type(exc).__name__}: {exc}'})
            return out

        out.update({
            'inspect_ok': True,
            'cut_tensors': list(cut_tensors),
            'p1_model': p1,
            'p2_model': p2,
            'split_manifest': split_manifest,
            'activation_precheck': activation_info,
            'parser_precheck': parser_info,
            'concat_sanity_precheck': concat_sanity_info,
        })

        if isinstance(activation_info, dict) and activation_info.get('inspect_ok') and activation_info.get('compatible') is False:
            detail = (
                cfg.hailo_part2_precheck_error_fn(activation_info)
                if cfg.hailo_part2_precheck_error_fn is not None
                else 'Unsupported Hailo Part2 activation-calibration splitpoint'
            )
            out.update({'compatible': False, 'reason': 'activation', 'detail': detail})
            return out

        parser_incompatible = isinstance(parser_info, dict) and parser_info.get('inspect_ok') and parser_info.get('compatible') is False
        concat_incompatible = isinstance(concat_sanity_info, dict) and concat_sanity_info.get('inspect_ok') and concat_sanity_info.get('compatible') is False
        if not parser_incompatible and not concat_incompatible:
            out.update({'compatible': True})
            return out

        def _try_part2_output_fallback(
            candidate_output_sets: Sequence[Sequence[str]],
            *,
            strategy: str,
            log_label: str,
            extra_payload: Optional[Mapping[str, Any]] = None,
        ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], List[str]]:
            first_concat_failure: Optional[Dict[str, Any]] = None
            first_concat_outputs: List[str] = []
            for output_names in list(candidate_output_sets or []):
                outputs_seq = [str(x).strip() for x in list(output_names or []) if str(x).strip()]
                if not outputs_seq:
                    continue
                try:
                    p1_alt, p2_alt, manifest_alt, activation_alt, parser_alt, concat_alt = _run_split(outputs_seq)
                except Exception as exc:
                    if callable(log_cb):
                        try:
                            log_cb(f"b{b}: {log_label} fallback {outputs_seq} failed ({type(exc).__name__}: {exc})")
                        except Exception:
                            pass
                    continue
                if isinstance(activation_alt, dict) and activation_alt.get('inspect_ok') and activation_alt.get('compatible') is False:
                    continue
                parser_alt_incompatible = isinstance(parser_alt, dict) and parser_alt.get('inspect_ok') and parser_alt.get('compatible') is False
                if parser_alt_incompatible:
                    continue
                concat_alt_incompatible = isinstance(concat_alt, dict) and concat_alt.get('inspect_ok') and concat_alt.get('compatible') is False
                if concat_alt_incompatible:
                    if first_concat_failure is None:
                        first_concat_failure = dict(concat_alt or {})
                        first_concat_outputs = list(outputs_seq)
                    continue
                # The accelerator-safe Part2 model is only the prefix up to the
                # suggested/raw-head outputs.  Keep the normal Part2 ONNX as the
                # semantic CPU tail and materialize an explicit host-tail model, so
                # the runtime can execute: stage2_prefix_on_Hailo -> DFL/decode tail
                # on the Jetson host.  This avoids falsely treating raw-head tensors
                # as decoded model outputs.
                try:
                    p2_prefix_for_hailo, p2_host_tail, tail_manifest = asc.split_model_on_cut_tensors(
                        p2,
                        cut_tensors=list(outputs_seq),
                        strict_boundary=True,
                    )
                    p2_prefix_for_hailo, pruned_prefix_inputs = self._prune_unused_graph_inputs_for_hailo_prefix(p2_prefix_for_hailo)
                    if pruned_prefix_inputs and callable(log_cb):
                        try:
                            log_cb(
                                f"b{b}: pruned unused Hailo Part2-prefix graph inputs "
                                f"{list(pruned_prefix_inputs)} before HEF build"
                            )
                        except Exception:
                            pass
                except Exception as exc:
                    if callable(log_cb):
                        try:
                            log_cb(f"b{b}: {log_label} host-tail split {outputs_seq} failed ({type(exc).__name__}: {exc})")
                        except Exception:
                            pass
                    continue

                payload = {
                    'compatible': True,
                    'strategy': str(strategy),
                    'effective_part2_outputs': list(outputs_seq),
                    'p1_model': p1,
                    'p2_model': p2,
                    'split_manifest': split_manifest,
                    'activation_precheck': activation_alt,
                    'parser_precheck': parser_alt,
                    'concat_sanity_precheck': concat_alt,
                    'p2_hailo_accel_model': p2_prefix_for_hailo,
                    'p2_hailo_accel_manifest': manifest_alt,
                    'p2_hailo_accel_pruned_inputs': list(pruned_prefix_inputs),
                    'p2_host_tail_model': p2_host_tail,
                    'p2_host_tail_manifest': tail_manifest,
                    'part2_host_tail_required': True,
                }
                if extra_payload:
                    payload.update(dict(extra_payload))
                return payload, first_concat_failure, first_concat_outputs
            return None, first_concat_failure, first_concat_outputs

        # YOLO11/YOLO10 part2 usually fails because the right subgraph still
        # contains the DFL/decode tail.  If the suite has a raw-head full-Hailo
        # contract (or if we can infer YOLO raw-head nodes), try the same endpoint
        # discipline for Part2: Hailo executes up to raw cv2/cv3 heads and the
        # decode/NMS tail remains a host-side responsibility.
        raw_candidate_output_sets, raw_end_nodes = self._part2_output_sets_from_raw_detection_heads(cfg)
        raw_payload_extra: Dict[str, Any] = {}
        if raw_candidate_output_sets:
            raw_family = 'yolo26_one2one_raw_head' if any('one2one_cv' in str(x) for x in raw_end_nodes) else 'raw_detection_head'
            raw_desc = (
                'YOLO26 one2one raw detection-head tensors; final transpose/concat/decode/postprocess remains outside the Part2 HEF.'
                if raw_family == 'yolo26_one2one_raw_head'
                else 'YOLO raw cv2/cv3 detection-head tensors; DFL/decode/NMS remains outside the Part2 HEF.'
            )
            raw_payload_extra = {
                'used_raw_detection_head_end_nodes': True,
                'raw_detection_head_end_node_names': list(raw_end_nodes),
                'part2_output_contract': {
                    'mode': raw_family,
                    'requires_external_postprocess': True,
                    'host_tail_required': True,
                    'description': raw_desc,
                    'end_node_names': list(raw_end_nodes),
                },
            }
            payload, raw_concat_failure, raw_concat_outputs = _try_part2_output_fallback(
                raw_candidate_output_sets,
                strategy='raw_detection_head_part2',
                log_label='raw detection-head',
                extra_payload=raw_payload_extra,
            )
            if payload is not None:
                contract = dict(payload.get('part2_output_contract') or {})
                contract['output_names'] = list(payload.get('effective_part2_outputs') or [])
                payload['part2_output_contract'] = contract
                out.update(payload)
                if callable(log_cb):
                    try:
                        log_cb(
                            f"b{b}: Hailo Part2 kept via raw detection-head outputs "
                            f"{list(payload.get('effective_part2_outputs') or [])}; decode/DFL tail stays on host"
                        )
                    except Exception:
                        pass
                return out
        else:
            raw_concat_failure = None
            raw_concat_outputs = []

        # If a YOLO/raw-head contract is active, do not fall back to the Hailo
        # parser's generic suggested end nodes for Part2.  In practice Hailo DFC
        # often suggests /model.23/Concat for YOLO11 tails.  That removes the DFL
        # Reshape blocker, but it still leaves a multi-scale detection-head Concat
        # in the accelerator prefix.  The real compiler then fails with messages
        # like "Unsupported dimensions at concat layer ... translated from
        # /model.23/Concat".  For YOLO deployments the only semantically clean
        # Part2 fallback is the same contract as Full-Hailo: accelerator terminates
        # at the complete raw cv2/cv3 head tensors, and DFL/decode/NMS stays on the
        # host.  Boundaries that cannot expose those raw-head outputs from Part2
        # are therefore rejected and the generator can backfill earlier candidates.
        if raw_end_nodes:
            detail = (
                'YOLO raw-head Hailo Part2 policy active; parser-suggested end-node fallback disabled. '
                'This boundary could not produce the complete raw detection-head output set for Part2, '
                'and using parser-suggested /model.23/Concat/Transpose-style outputs can trigger Hailo concat/layout failures. '
                'Choose/backfill a boundary whose Part2 can terminate at the raw cv2/cv3 or YOLO26 one2one head tensors.'
            )
            if raw_candidate_output_sets:
                detail += ' [raw-head fallback attempted but did not pass prechecks]'
            if raw_concat_failure is not None:
                try:
                    detail += ' [' + format_hailo_part2_concat_sanity_error(raw_concat_failure) + ']'
                except Exception:
                    pass

            # YOLO11 requires a complete raw-head Part2 to make a meaningful
            # Hailo/Hailo case, so we reject and backfill.  YOLO26 is also a
            # prime heterogeneous Hailo->TensorRT target: when the raw one2one
            # Part2 prefix is unavailable, keep the case for Hailo Part1 ->
            # TensorRT/ORT Part2 instead of discarding a potentially useful split.
            if self._is_yolo26_cfg(cfg):
                stage1_detail = (
                    'YOLO26 one2one Part2 raw-head policy did not find a safe Hailo Part2 prefix for this boundary; '
                    'keeping the split as Hailo-Part1 -> TensorRT/ORT-Part2 and skipping Hailo Part2 HEF generation. '
                    + detail
                )
                out.update({
                    'compatible': True,
                    'strategy': 'yolo26_hailo_part1_only_stage1',
                    'used_suggested_end_nodes': False,
                    'effective_part2_outputs': [],
                    'skip_hailo_part2_build': True,
                    'hailo_part2_policy': 'skip_yolo26_raw_head_part2_unavailable',
                    'raw_detection_head_end_node_names': list(raw_end_nodes),
                    'suggested_end_nodes': list((parser_info or {}).get('suggested_end_nodes') or []),
                    'part2_output_contract': {
                        'mode': 'decoded_or_end2end_host_or_trt',
                        'requires_external_postprocess': False,
                        'host_tail_required': False,
                        'hailo_part2_unsupported': True,
                        'raw_head_policy_enforced': True,
                        'end_node_names': list(raw_end_nodes),
                        'description': 'YOLO26 Part2 remains on TensorRT/ORT host/backend; Hailo is used only for Part1 in heterogeneous runs when a safe one2one raw-head Part2 prefix is unavailable.',
                    },
                    'policy_detail': stage1_detail,
                })
                if callable(log_cb):
                    try:
                        log_cb(f"b{b}: {stage1_detail}")
                    except Exception:
                        pass
                return out

            out.update({
                'compatible': False,
                'reason': 'yolo_raw_head_policy',
                'detail': detail,
                'raw_detection_head_end_node_names': list(raw_end_nodes),
                'suggested_end_nodes': list((parser_info or {}).get('suggested_end_nodes') or []),
                'raw_head_policy_enforced': True,
            })
            return out

        if not parser_incompatible and concat_incompatible:
            out.update({
                'compatible': False,
                'reason': 'concat_sanity',
                'detail': format_hailo_part2_concat_sanity_error(concat_sanity_info or {}),
            })
            return out

        # YOLO26 is different from YOLO11: the exported model is normally
        # end-to-end / one-to-one and DFL-free.  The parser often suggests a
        # late Transpose endpoint for Part2, but the real build still fails later
        # in Concat/format-conversion allocation.  For the heterogeneous thesis
        # use case, Hailo only needs to supply stage1; TensorRT/ORT can own the
        # full decoded/end-to-end Part2.  Therefore keep the split as usable, but
        # deliberately skip Hailo-Part2 compilation unless an explicit safe
        # raw-head/end-node policy succeeded above.
        if self._is_yolo26_cfg(cfg):
            suggested_nodes = [str(x).strip() for x in list((parser_info or {}).get('suggested_end_nodes') or []) if str(x).strip()]
            detail = (
                'YOLO26 end-to-end head policy: Hailo Part2 is treated as unsupported for this boundary. '
                'The parser-suggested Transpose/Concat-style endpoint can pass cheap prechecks but has repeatedly '
                'failed during Hailo allocation/layout mapping. The case remains usable for Hailo-Part1 -> TensorRT/ORT-Part2; '
                'Hailo Part2 HEF generation will be skipped.'
            )
            out.update({
                'compatible': True,
                'strategy': 'yolo26_hailo_part1_only_stage1',
                'used_suggested_end_nodes': False,
                'effective_part2_outputs': [],
                'skip_hailo_part2_build': True,
                'hailo_part2_policy': 'skip_yolo26_end2end_head',
                'suggested_end_nodes': suggested_nodes,
                'part2_output_contract': {
                    'mode': 'decoded_or_end2end_host_or_trt',
                    'requires_external_postprocess': False,
                    'host_tail_required': False,
                    'hailo_part2_unsupported': True,
                    'description': 'YOLO26 Part2 remains on TensorRT/ORT host/backend; Hailo is used only for Part1 in heterogeneous runs.',
                },
                'policy_detail': detail,
            })
            if callable(log_cb):
                try:
                    log_cb(f"b{b}: {detail}")
                except Exception:
                    pass
            return out

        if not bool(getattr(cfg, 'hailo_part2_enable_suggested_endnode_fallback', True)):
            detail = (
                cfg.hailo_part2_parser_precheck_error_fn(parser_info)
                if cfg.hailo_part2_parser_precheck_error_fn is not None
                else 'Unsupported Hailo Part2 parser-blocking head'
            )
            suggested_nodes = list((parser_info or {}).get('suggested_end_nodes') or [])
            if suggested_nodes:
                detail = f"{detail} [suggested end-node fallback disabled]"
            if raw_candidate_output_sets:
                detail = f"{detail} [raw-head fallback attempted but did not pass prechecks]"
            out.update({
                'compatible': False,
                'reason': 'parser',
                'detail': detail,
                'suggested_end_nodes': suggested_nodes,
                'raw_detection_head_end_node_names': list(raw_end_nodes),
                'fallback_disabled': True,
            })
            return out

        candidate_output_sets = self._part2_output_sets_from_suggested_end_nodes(cfg.nodes, parser_info)
        suggested_payload_extra = {
            'used_suggested_end_nodes': True,
            'suggested_end_nodes': list((parser_info or {}).get('suggested_end_nodes') or []),
        }
        payload, first_concat_sanity_failure, first_concat_sanity_outputs = _try_part2_output_fallback(
            candidate_output_sets,
            strategy='hailo_parser_suggested_end_nodes',
            log_label='suggested end-node',
            extra_payload=suggested_payload_extra,
        )
        if payload is not None:
            out.update(payload)
            return out

        # Prefer the concat-sanity detail from the fallback that got furthest.
        if first_concat_sanity_failure is None and raw_concat_failure is not None:
            first_concat_sanity_failure = raw_concat_failure
            first_concat_sanity_outputs = list(raw_concat_outputs or [])

        if first_concat_sanity_failure is not None:
            if first_concat_sanity_outputs:
                first_concat_sanity_failure.setdefault('part2_outputs', list(first_concat_sanity_outputs))
            out.update({
                'compatible': False,
                'reason': 'concat_sanity',
                'detail': format_hailo_part2_concat_sanity_error(first_concat_sanity_failure),
                'concat_sanity_precheck': first_concat_sanity_failure,
                'effective_part2_outputs': list(first_concat_sanity_outputs),
                'suggested_end_nodes': list((parser_info or {}).get('suggested_end_nodes') or []),
                'raw_detection_head_end_node_names': list(raw_end_nodes),
            })
            return out

        detail = (
            cfg.hailo_part2_parser_precheck_error_fn(parser_info)
            if cfg.hailo_part2_parser_precheck_error_fn is not None
            else 'Unsupported Hailo Part2 parser-blocking head'
        )
        if raw_candidate_output_sets:
            detail = f"{detail} [raw-head fallback attempted but did not pass prechecks]"
        out.update({
            'compatible': False,
            'reason': 'parser',
            'detail': detail,
            'suggested_end_nodes': list((parser_info or {}).get('suggested_end_nodes') or []),
            'raw_detection_head_end_node_names': list(raw_end_nodes),
        })
        return out

    def _yolo26_part1_static_skip_reason(self, cfg: BenchmarkGenerationExecutionConfig, cut_tensors: Sequence[Any]) -> str:
        """Cheap log-derived guard for YOLO26 Part1 Hailo build attempts.

        Older YOLO26L/YOLO26M generation logs show that boundaries crossing the
        late detection head around ``/model.23/Reshape*`` almost always fail Hailo
        Part1 mapping after minutes of calibration/allocation, typically with
        ``format_conversion* -> Agent infeasible`` or feature-splitter layout
        errors.  Good heterogeneous YOLO26 splits either cut much earlier or cut
        at clean one2one/raw-head Conv tensors without the Reshape/Transpose tail.

        Keep this as a conservative pre-build skip: it only applies to YOLO26 and
        only when a Hailo Part1 HEF is requested.
        """
        try:
            if not bool(cfg.hef_part1) or not cfg.hef_targets or not self._is_yolo26_cfg(cfg):
                return ""
            names = [str(x or "").strip() for x in list(cut_tensors or []) if str(x or "").strip()]
            if not names:
                return ""
            low = [x.lower().replace('\\', '/') for x in names]
            model23_reshape = sum(1 for x in low if x.startswith('/model.23/reshape'))
            model23_late_layout = sum(1 for x in low if x.startswith('/model.23/') and any(tok in x for tok in ('/reshape', '/transpose', '/concat', '/slice')))
            model22_layout = sum(1 for x in low if x.startswith('/model.22/') and any(tok in x for tok in ('/slice', '/split', '/concat', '/reshape')))
            model19_layout = sum(1 for x in low if x.startswith('/model.19/') and any(tok in x for tok in ('/slice', '/split', '/concat', '/reshape')))
            one2one = sum(1 for x in low if 'one2one_' in x)
            crosses_model22_23 = any(x.startswith('/model.22/') for x in low) and any(x.startswith('/model.23/') for x in low)
            crosses_mid_and_head = any(x.startswith('/model.19/') or x.startswith('/model.20/') or x.startswith('/model.21/') for x in low) and any(x.startswith('/model.23/') for x in low)

            if model23_reshape >= 2:
                return (
                    'YOLO26 static Hailo Part1 guard: boundary crosses /model.23/Reshape* detection-head tail; '
                    'old YOLO26L/M logs show this neighborhood repeatedly fails with Hailo format_conversion/agent-infeasible mapping. '
                    'Backfill to a cleaner one2one/raw-head Conv cut or an earlier Hailo-Part1 -> TRT split.'
                )
            if crosses_model22_23 and (model23_late_layout or model22_layout) and len(low) >= 4:
                return (
                    'YOLO26 static Hailo Part1 guard: boundary mixes /model.22 and /model.23 late-head layout tensors; '
                    'this matches prior YOLO26L Hailo mapping failures.'
                )
            if crosses_mid_and_head and model23_late_layout and len(low) >= 4:
                return (
                    'YOLO26 static Hailo Part1 guard: boundary mixes mid-neck tensors with /model.23 layout tensors; '
                    'this matches prior YOLO26 Hailo format-conversion failures.'
                )
            if one2one and (model23_late_layout + model22_layout + model19_layout) >= 3 and len(low) >= 5:
                return (
                    'YOLO26 static Hailo Part1 guard: boundary combines one2one head tensors with late layout tensors; '
                    'prefer a clean one2one Conv cut without Reshape/Slice/Concat crossings.'
                )
        except Exception:
            logger.debug('YOLO26 static Part1 skip guard failed', exc_info=True)
        return ""

    def execute_case_build_loop(self, cfg: BenchmarkGenerationExecutionConfig, cb: BenchmarkGenerationExecutionCallbacks) -> List[int]:
        from .. import api as asc

        runtime = cfg.runtime
        cases = runtime.cases
        errors = runtime.errors
        discarded_cases = runtime.discarded_cases
        accepted_boundaries = runtime.accepted_boundaries
        completed_boundaries = runtime.completed_boundaries
        discarded_boundaries = runtime.discarded_boundaries

        log = cb.log
        qput = cb.queue_put
        chosen: List[int] = sorted(int(x) for x in accepted_boundaries)
        made = int(len(cases))

        def _persist(status: str = "running", current_boundary: Optional[int] = None) -> None:
            try:
                cb.persist_state(status=status, current_boundary=current_boundary)
            except Exception:
                logger.debug("persist_state callback failed", exc_info=True)

        def _cancel_requested() -> bool:
            try:
                return bool(callable(cfg.should_cancel) and cfg.should_cancel())
            except Exception:
                return False

        def _raise_if_cancelled(stage: str = "") -> None:
            if not _cancel_requested():
                return
            detail = f" ({stage})" if str(stage or "").strip() else ""
            raise BenchmarkGenerationCancelled(f"Benchmark-set generation cancelled by user{detail}")

        def _hef_failure_label(res: Any) -> str:
            return "SKIPPED" if bool(getattr(res, "skipped", False)) else "FAILED"

        log(f"min_gap: {cfg.gap}")
        log(f"preferred shortlist size: {len(cfg.ranked_candidates)}")
        log(f"ranked candidates considered: {len(cfg.candidate_search_pool)}")

        semantic_cache: Dict[int, Any] = {}
        for row in list(cfg.analysis_candidates or []):
            try:
                boundary = int(row.get("boundary", -1))
            except Exception:
                continue
            semantic_cache[boundary] = row.get("semantic")

        candidate_policy_index = build_candidate_policy_index(cfg.analysis_candidates or [])
        hailo_failure_records: List[HailoFailureRecord] = []
        row_per_cut_donors: Dict[str, Dict[int, List[str]]] = {}

        target_label = ",".join([str(x).strip() for x in (cfg.hef_targets or []) if str(x).strip()]) or "hailo"

        for raw_boundary in list(cfg.candidate_search_pool):
            _raise_if_cancelled("candidate loop")
            if made >= int(cfg.target_cases):
                break
            b = int(raw_boundary)

            if b in completed_boundaries:
                log(f"b{b}: skip (already completed)")
                qput(("prog", made, f"b{b} (resume-skip)"))
                continue

            if int(cfg.gap) > 0 and any(abs(b - bb) <= int(cfg.gap) for bb in chosen):
                log(f"b{b}: skip (min_gap)")
                continue

            cluster_skip = should_skip_from_failure_cluster(
                b,
                hailo_failure_records,
                candidate_policy=candidate_policy_index.get(int(b)) or {},
                stage="part1",
                hw_archs=cfg.hef_targets,
                radius=12,
                min_failures=2,
            )
            if (not cluster_skip.skip) and bool(cfg.hef_part2):
                cluster_skip = should_skip_from_failure_cluster(
                    b,
                    hailo_failure_records,
                    candidate_policy=candidate_policy_index.get(int(b)) or {},
                    stage="part2",
                    hw_archs=cfg.hef_targets,
                    radius=12,
                    min_failures=2,
                )
            if cluster_skip.skip:
                detail = str(cluster_skip.detail or "nearby Hailo allocator/layout failures")
                log(f"b{b}: skip (adaptive Hailo neighborhood filter) - {detail}")
                discarded_cases.append(
                    build_benchmark_case_rejection(
                        boundary=int(b),
                        folder=f"b{int(b):0{cfg.pad}d}",
                        reason="hailo_failure_cluster_skip",
                        stage="part1",
                        hw_arch=target_label,
                        detail=detail,
                    )
                )
                discarded_boundaries.add(int(b))
                completed_boundaries.add(int(b))
                _persist(status='running', current_boundary=int(b))
                qput(("prog", made, f"b{b} (skip: Hailo fail-cluster)"))
                continue

            qput(("prog", made, f"Splitting b{b} ({made+1}/{cfg.target_cases})..."))
            log(f"--- [{made+1}/{cfg.target_cases}] Split boundary b{b} ---")
            folder = f"b{b:0{cfg.pad}d}"
            case_dir = os.path.join(str(cfg.out_dir), folder)
            os.makedirs(case_dir, exist_ok=True)

            try:
                cut_tensors = asc.cut_tensors_for_boundary(cfg.order, cfg.nodes, b)
            except Exception as exc:
                errors.append(f"b{b}: cut tensor error: {exc}")
                log(f"b{b}: cut tensor error: {exc}")
                qput(("prog", made, f"b{b} (skip)"))
                continue

            if not cut_tensors:
                errors.append(f"b{b}: no cut tensors")
                log(f"b{b}: no cut tensors")
                qput(("prog", made, f"b{b} (skip)"))
                continue

            log(f"b{b}: cut tensors: {len(cut_tensors)}")

            static_part1_skip = self._yolo26_part1_static_skip_reason(cfg, cut_tensors)
            if static_part1_skip:
                log(f"b{b}: skip (YOLO26 static Hailo Part1 guard) - {static_part1_skip}")
                discarded_cases.append(
                    build_benchmark_case_rejection(
                        boundary=int(b),
                        folder=f"b{int(b):0{cfg.pad}d}",
                        reason="hailo_yolo26_static_part1_guard",
                        stage="part1",
                        hw_arch=target_label,
                        detail=static_part1_skip,
                    )
                )
                discarded_boundaries.add(int(b))
                completed_boundaries.add(int(b))
                _persist(status='running', current_boundary=int(b))
                try:
                    shutil.rmtree(case_dir, ignore_errors=True)
                except Exception:
                    pass
                qput(("prog", made, f"b{b} (skip: YOLO26 static Part1 guard)"))
                continue

            p1_path = os.path.join(case_dir, f"{cfg.base}_part1_b{b}.onnx")
            p2_path = os.path.join(case_dir, f"{cfg.base}_part2_b{b}.onnx")
            manifest_path = os.path.join(case_dir, "split_manifest.json")
            _raise_if_cancelled(f"before split b{b}")

            hailo_part2_precheck = None
            hailo_part2_parser_precheck = None
            hailo_part2_concat_sanity_precheck = None
            part2_output_strategy = 'original'
            effective_part2_outputs: List[str] = []
            part2_output_contract: Dict[str, Any] = {}
            p2_hailo_accel_model = None
            p2_hailo_accel_manifest = None
            p2_hailo_accel_pruned_inputs: List[str] = []
            p2_host_tail_model = None
            p2_host_tail_manifest = None
            p2_hailo_accel_path: Optional[str] = None
            p2_host_tail_path: Optional[str] = None
            part2_host_tail_required = False
            skip_hailo_part2_build = False
            hailo_part2_policy_detail = ''
            try:
                if bool(cfg.hef_part2) and (cfg.hailo_part2_precheck_fn is not None or cfg.hailo_part2_parser_precheck_fn is not None):
                    probe = self.probe_hailo_part2_support(cfg, b, log_cb=log)
                    if probe.get('compatible') is False:
                        detail = str(probe.get('detail') or 'Unsupported Hailo Part2 split')
                        probe_reason = str(probe.get('reason') or '').strip()
                        if probe_reason == 'activation':
                            reason_key = 'hailo_part2_auto_filtered'
                            progress_label = f"b{b} (skip: Hailo part2 precheck)"
                            info_payload = probe.get('activation_precheck')
                        elif probe_reason == 'concat_sanity':
                            reason_key = 'hailo_part2_concat_sanity_auto_filtered'
                            progress_label = f"b{b} (skip: Hailo part2 concat sanity)"
                            info_payload = probe.get('concat_sanity_precheck')
                        else:
                            reason_key = 'hailo_part2_parser_auto_filtered'
                            progress_label = f"b{b} (skip: Hailo part2 parser precheck)"
                            info_payload = probe.get('parser_precheck')
                        log(f"b{b}: HEF(part2,{target_label}) SKIPPED: {detail}")
                        rec = self._record_hailo_part2_filter(
                            b,
                            pad=int(cfg.pad),
                            detail=detail,
                            target_label=target_label,
                            reason=reason_key,
                            info=(info_payload if isinstance(info_payload, dict) else None),
                        )
                        if probe.get('effective_part2_outputs'):
                            rec['effective_part2_outputs'] = list(probe.get('effective_part2_outputs') or [])
                        discarded_cases.append(rec)
                        discarded_boundaries.add(int(b))
                        completed_boundaries.add(int(b))
                        _persist(status='running', current_boundary=int(b))
                        try:
                            shutil.rmtree(case_dir, ignore_errors=True)
                        except Exception:
                            pass
                        qput(("prog", made, progress_label))
                        continue
                    p1 = probe.get('p1_model')
                    p2 = probe.get('p2_model')
                    split_manifest = dict(probe.get('split_manifest') or {})
                    hailo_part2_precheck = probe.get('activation_precheck') if isinstance(probe.get('activation_precheck'), dict) else None
                    hailo_part2_parser_precheck = probe.get('parser_precheck') if isinstance(probe.get('parser_precheck'), dict) else None
                    hailo_part2_concat_sanity_precheck = probe.get('concat_sanity_precheck') if isinstance(probe.get('concat_sanity_precheck'), dict) else None
                    part2_output_strategy = str(probe.get('strategy') or 'original')
                    effective_part2_outputs = [str(x).strip() for x in list(probe.get('effective_part2_outputs') or []) if str(x).strip()]
                    part2_output_contract = (dict(probe.get('part2_output_contract') or {}) if isinstance(probe.get('part2_output_contract'), Mapping) else {})
                    p2_hailo_accel_model = probe.get('p2_hailo_accel_model')
                    p2_hailo_accel_manifest = probe.get('p2_hailo_accel_manifest') if isinstance(probe.get('p2_hailo_accel_manifest'), dict) else None
                    p2_hailo_accel_pruned_inputs = [str(x).strip() for x in list(probe.get('p2_hailo_accel_pruned_inputs') or []) if str(x).strip()]
                    p2_host_tail_model = probe.get('p2_host_tail_model')
                    p2_host_tail_manifest = probe.get('p2_host_tail_manifest') if isinstance(probe.get('p2_host_tail_manifest'), dict) else None
                    part2_host_tail_required = bool(probe.get('part2_host_tail_required'))
                    skip_hailo_part2_build = bool(probe.get('skip_hailo_part2_build'))
                    hailo_part2_policy_detail = str(probe.get('policy_detail') or '')
                    if part2_output_strategy != 'original':
                        log(
                            f"b{b}: using alternative Hailo Part2 strategy "
                            f"({part2_output_strategy}, outputs={effective_part2_outputs})"
                        )
                    if skip_hailo_part2_build:
                        log(
                            f"b{b}: Hailo Part2 build disabled by policy "
                            f"({part2_output_strategy}); case remains available for Hailo-Part1 -> TRT/ORT-Part2"
                        )
                else:
                    p1, p2, split_manifest = asc.split_model_on_cut_tensors(
                        cfg.model,
                        cut_tensors=cut_tensors,
                        strict_boundary=bool(cfg.strict_boundary),
                    )
                asc.save_model(p1, p1_path)
                asc.save_model(p2, p2_path)
                if p2_hailo_accel_model is not None:
                    p2_hailo_accel_path = os.path.join(case_dir, f"{cfg.base}_part2_hailo_prefix_b{b}.onnx")
                    asc.save_model(p2_hailo_accel_model, p2_hailo_accel_path)
                if p2_host_tail_model is not None:
                    p2_host_tail_path = os.path.join(case_dir, f"{cfg.base}_part2_host_tail_b{b}.onnx")
                    asc.save_model(p2_host_tail_model, p2_host_tail_path)
            except Exception as exc:
                errors.append(f"b{b}: split failed: {type(exc).__name__}: {exc}")
                log(f"b{b}: split failed: {type(exc).__name__}: {exc}")
                qput(("prog", made, f"b{b} (split failed)"))
                continue

            log(f"b{b}: wrote {os.path.basename(p1_path)}")
            log(f"b{b}: wrote {os.path.basename(p2_path)}")
            if p2_hailo_accel_path:
                log(f"b{b}: wrote {os.path.basename(p2_hailo_accel_path)} (Hailo Part2 accelerator prefix)")
            if p2_host_tail_path:
                log(f"b{b}: wrote {os.path.basename(p2_host_tail_path)} (host-side Part2 tail)")

            pred: Dict[str, Any] = {}
            try:
                pred = dict(cb.predicted_metrics_for_boundary(cfg.analysis_payload, int(b)) or {})
            except Exception:
                pred = {}
            if cfg.hailo_compile_rank_meta:
                try:
                    pred.update(dict(cfg.hailo_compile_rank_meta.get(int(b)) or {}))
                except Exception:
                    pass

            try:
                hailo_parse_entry = cb.hailo_parse_entry_for_boundary(cfg.analysis_payload, int(b))
            except Exception:
                hailo_parse_entry = None
            try:
                hailo_parse_fields = dict(cb.hailo_parse_scalar_fields(hailo_parse_entry) or {})
            except Exception:
                hailo_parse_fields = {}

            manifest_out: Dict[str, Any] = {
                'tool': {'gui': str(cfg.tool_gui_version or '?'), 'core': str(cfg.tool_core_version or getattr(asc, '__version__', '?'))},
                'boundary': int(b),
                'boundary_index': int(b),
                'cut_tensors': list(cut_tensors),
                'strict_boundary': bool(cfg.strict_boundary),
                'predicted': pred,
                'full_model': (
                    Path(os.path.relpath(cfg.full_model_dst, start=case_dir)).as_posix()
                    if cfg.full_model_dst and os.path.exists(cfg.full_model_dst)
                    else str(cfg.full_model_src).replace('\\', '/')
                ),
                'full_model_source': str(cfg.full_model_src).replace('\\', '/'),
                'part1': os.path.basename(p1_path).replace('\\', '/'),
                'part2': os.path.basename(p2_path).replace('\\', '/'),
                'created_at': datetime.now().isoformat(timespec='seconds'),
            }
            if hailo_parse_entry is not None:
                manifest_out['hailo_parse_check'] = hailo_parse_entry
            manifest_out.update(hailo_parse_fields)
            if isinstance(split_manifest, dict):
                manifest_out.update(split_manifest)
            if isinstance(hailo_part2_precheck, dict):
                manifest_out.setdefault('hailo', {})
                manifest_out['hailo']['part2_activation_precheck'] = hailo_part2_precheck
            if isinstance(hailo_part2_parser_precheck, dict):
                manifest_out.setdefault('hailo', {})
                manifest_out['hailo']['part2_parser_precheck'] = hailo_part2_parser_precheck
            if isinstance(hailo_part2_concat_sanity_precheck, dict):
                manifest_out.setdefault('hailo', {})
                manifest_out['hailo']['part2_concat_sanity_precheck'] = hailo_part2_concat_sanity_precheck
            if part2_output_strategy != 'original' or effective_part2_outputs or part2_output_contract:
                manifest_out.setdefault('hailo', {})
                manifest_out['hailo']['part2_output_strategy'] = str(part2_output_strategy)
                if effective_part2_outputs:
                    manifest_out['hailo']['part2_effective_outputs'] = list(effective_part2_outputs)
                if part2_output_contract:
                    _contract = dict(part2_output_contract)
                    if effective_part2_outputs and not _contract.get('output_names'):
                        _contract['output_names'] = list(effective_part2_outputs)
                    if part2_host_tail_required:
                        _contract['host_tail_required'] = True
                    manifest_out['hailo']['part2_output_contract'] = _contract
            if skip_hailo_part2_build:
                manifest_out.setdefault('hailo', {})
                manifest_out['hailo']['part2_build_skipped_by_policy'] = True
                manifest_out['hailo']['part2_policy'] = str(part2_output_strategy or 'policy_skip')
                if hailo_part2_policy_detail:
                    manifest_out['hailo']['part2_policy_detail'] = hailo_part2_policy_detail
            if p2_hailo_accel_path or p2_host_tail_path or part2_host_tail_required:
                manifest_out.setdefault('hailo', {})
                if p2_hailo_accel_path:
                    manifest_out['hailo']['part2_accel_model'] = os.path.basename(p2_hailo_accel_path).replace('\\', '/')
                    manifest_out['hailo']['part2_accelerator_model'] = os.path.basename(p2_hailo_accel_path).replace('\\', '/')
                    if p2_hailo_accel_pruned_inputs:
                        manifest_out['hailo']['part2_accel_pruned_unused_inputs'] = list(p2_hailo_accel_pruned_inputs)
                if p2_host_tail_path:
                    manifest_out['hailo']['part2_host_tail_model'] = os.path.basename(p2_host_tail_path).replace('\\', '/')
                manifest_out['hailo']['part2_host_tail_required'] = bool(part2_host_tail_required or p2_host_tail_path)
                if effective_part2_outputs:
                    manifest_out['hailo']['part2_accelerator_outputs'] = list(effective_part2_outputs)
                if isinstance(p2_hailo_accel_manifest, dict):
                    manifest_out['hailo']['part2_accel_manifest'] = dict(p2_hailo_accel_manifest)
                if isinstance(p2_host_tail_manifest, dict):
                    manifest_out['hailo']['part2_host_tail_manifest'] = dict(p2_host_tail_manifest)
            full_end_nodes_case = [str(x).strip() for x in list(cfg.hailo_full_end_node_names or []) if str(x).strip()]
            full_contract_case = dict(cfg.hailo_full_output_contract or {}) if isinstance(cfg.hailo_full_output_contract, Mapping) else {}
            if full_end_nodes_case or full_contract_case or str(cfg.hailo_full_endpoint_mode or '').strip():
                manifest_out.setdefault('hailo', {})
                manifest_out['hailo']['full_endpoint_mode'] = str(cfg.hailo_full_endpoint_mode or ('custom_end_nodes' if full_end_nodes_case else 'full'))
                if full_end_nodes_case:
                    manifest_out['hailo']['full_end_node_names'] = list(full_end_nodes_case)
                if full_contract_case:
                    manifest_out['hailo']['full_output_contract'] = dict(full_contract_case)

            try:
                manifest_out.setdefault('schema', 'onnx-splitpoint/split-manifest')
                manifest_out.setdefault('schema_version', 1)
                manifest_out['models'] = {
                    'full': {'path': str(manifest_out.get('full_model') or '').replace('\\', '/'), 'source': str(manifest_out.get('full_model_source') or '').replace('\\', '/')},
                    'part1': {'path': str(manifest_out.get('part1_model') or manifest_out.get('part1') or '').replace('\\', '/')},
                    'part2': {'path': str(manifest_out.get('part2_model') or manifest_out.get('part2') or '').replace('\\', '/')},
                }
                if p2_hailo_accel_path:
                    manifest_out['models']['part2_hailo_accel'] = {'path': os.path.basename(p2_hailo_accel_path).replace('\\', '/')}
                if p2_host_tail_path:
                    manifest_out['models']['part2_host_tail'] = {'path': os.path.basename(p2_host_tail_path).replace('\\', '/')}
                cut_full = manifest_out.get('cut_tensors_full') or manifest_out.get('cut_tensors') or []
                cut_p1 = manifest_out.get('part1_cut_names') or []
                cut_p2 = manifest_out.get('part2_cut_names') or []
                manifest_out['cut'] = {
                    'names_full': list(cut_full),
                    'names_part1': list(cut_p1),
                    'names_part2': list(cut_p2),
                    'count': int(len(cut_full) or len(cut_p1) or len(cut_p2) or len(cut_tensors)),
                }
                manifest_out['io'] = {
                    'part1_inputs': list(manifest_out.get('part1_inputs') or []),
                    'part1_outputs': list(manifest_out.get('part1_outputs') or []),
                    'part2_inputs': list(manifest_out.get('part2_inputs') or []),
                    'part2_outputs': list(manifest_out.get('part2_outputs_effective') or manifest_out.get('part2_outputs') or manifest_out.get('orig_outputs') or []),
                    'part1_external_inputs': list(manifest_out.get('part1_external_inputs') or []),
                    'part2_external_inputs': list(manifest_out.get('part2_external_inputs') or []),
                }
                links = []
                if isinstance(cut_p1, list) and isinstance(cut_p2, list) and len(cut_p1) == len(cut_p2):
                    for src_name, dst_name in zip(cut_p1, cut_p2):
                        if src_name and dst_name:
                            links.append({'from': str(src_name), 'to': str(dst_name)})
                manifest_out['pipeline'] = {'stage1': 'part1', 'stage2': 'part2', 'links': links}
            except Exception:
                logger.debug('Could not normalize split manifest for b%s', b, exc_info=True)

            semantic_label = semantic_cache.get(int(b))
            if cfg.do_ctx_full:
                try:
                    ctx = asc.export_boundary_graphviz_context(
                        cfg.model,
                        cfg.order,
                        b,
                        cut_tensors,
                        case_dir,
                        basename=f"split_context_b{b}",
                        render=True,
                        hops=int(cfg.ctx_hops),
                        strict_boundary=bool(cfg.strict_boundary),
                        include_external_inputs=(not bool(cfg.llm_style)),
                        semantic_label=semantic_label,
                        value_bytes_map=cfg.value_bytes_map,
                        force_matplotlib_fallback=bool(cfg.llm_style),
                    )
                    manifest_out['split_context'] = ctx
                except Exception as exc:
                    manifest_out['split_context_error'] = str(exc)

            if cfg.do_ctx_cutflow:
                try:
                    ctx_cf = asc.export_boundary_graphviz_context(
                        cfg.model,
                        cfg.order,
                        b,
                        cut_tensors,
                        case_dir,
                        basename=f"split_context_b{b}_cutflow",
                        render=True,
                        hops=int(cfg.ctx_hops),
                        strict_boundary=bool(cfg.strict_boundary),
                        cut_flow_only=True,
                        include_internal_consumers=False,
                        include_external_inputs=False,
                        semantic_label=semantic_label,
                        value_bytes_map=cfg.value_bytes_map,
                        force_matplotlib_fallback=bool(cfg.llm_style),
                    )
                    manifest_out['split_context_cutflow'] = ctx_cf
                except Exception as exc:
                    manifest_out['split_context_cutflow_error'] = str(exc)

            try:
                runner_path = asc.write_runner_skeleton_onnxruntime(
                    case_dir,
                    manifest_filename=os.path.basename(manifest_path),
                    target=cfg.runner_target,
                )
                manifest_out['runner'] = os.path.basename(runner_path)
            except Exception as exc:
                errors.append(f"b{b}: runner skeleton failed: {exc}")

            case_rejection = None
            case_first_rejection = None
            case_variant_availability: Dict[str, Dict[str, Any]] = {}
            case_suite_full_available = any(
                bool(((runtime.suite_hailo_hefs.get(str(hw).strip()) or {}).get('full')))
                for hw in list(cfg.hef_targets or [])
                if str(hw).strip()
            )
            if cfg.hef_targets and (cfg.hef_part1 or cfg.hef_part2):
                log(
                    f"b{b}: Hailo HEF generation requested (backend={cfg.hef_backend}, targets={cfg.hef_targets}, full={bool(case_suite_full_available)}, part1={cfg.hef_part1}, part2={cfg.hef_part2})"
                )
                if cfg.hailo_build_hef_fn is None:
                    manifest_out['hailo_error'] = str(cfg.hailo_build_unavailable or 'Hailo HEF build unavailable')
                    log(f"b{b}: {manifest_out['hailo_error']}")
                    errors.append(f"b{b}: {manifest_out['hailo_error']}")
                else:
                    manifest_out.setdefault('hailo', {})
                    manifest_out['hailo'].setdefault('hefs', {})
                    manifest_out['hailo']['config'] = {
                        'backend': str(cfg.hef_backend),
                        'wsl_distro': cfg.hef_wsl_distro,
                        'wsl_venv': str(cfg.hef_wsl_venv),
                        'opt_level': int(cfg.hef_opt_level),
                        'calib_count': int(cfg.hef_calib_count),
                        'calib_batch_size': int(cfg.hef_calib_bs),
                        'calib_dir': (str(cfg.hef_calib_dir).replace('\\', '/') if cfg.hef_calib_dir else None),
                        'fixup': bool(cfg.hef_fixup),
                        'force': bool(cfg.hef_force),
                        'keep_artifacts': bool(cfg.hef_keep),
                        'timeout_s': int(cfg.hef_timeout_s),
                        'build': {'full': bool(case_suite_full_available), 'part1': bool(cfg.hef_part1), 'part2': bool(cfg.hef_part2)},
                    }
                    debug_env_key = 'ONNX_SPLITPOINT_HAILO_DEBUG_FILES'
                    old_debug_env = os.environ.get(debug_env_key)
                    if old_debug_env in (None, ''):
                        os.environ[debug_env_key] = '1'
                    try:
                        raw_hef_targets = [str(x).strip() for x in list(cfg.hef_targets or []) if str(x).strip()]
                        physical_hef_targets = _physical_hailo_targets_for_build(raw_hef_targets)
                        if raw_hef_targets and physical_hef_targets != raw_hef_targets:
                            log(
                                f"b{b}: normalizing Hailo build targets {raw_hef_targets} -> physical DFC archs {physical_hef_targets}; "
                                "mixed Hailo<->TRT runs reuse these Hailo artifacts instead of calling DFC with pipeline ids."
                            )
                        for hw_arch in physical_hef_targets:
                            _raise_if_cancelled(f"before Hailo HEF build b{b}")
                            hw_arch = str(hw_arch).strip()
                            if not hw_arch:
                                continue
                            log(f"b{b}: build HEF for hw_arch={hw_arch}")

                            def _on_hef_log(stream: str, line: str, _b: int = b, _hw: str = hw_arch) -> None:
                                msg = f"(b{_b} {_hw}) {line}"
                                log(msg)
                                try:
                                    qput(("hef", stream, msg))
                                except Exception:
                                    pass

                            tgt_out: Dict[str, Any] = {}
                            suite_tgt = runtime.suite_hailo_hefs.get(hw_arch) or {}
                            full_rel = suite_tgt.get('full') if isinstance(suite_tgt, dict) else None
                            if full_rel:
                                abs_full = os.path.join(str(cfg.out_dir), str(full_rel))
                                tgt_out['full'] = os.path.relpath(abs_full, case_dir).replace('\\', '/')
                            if isinstance(suite_tgt, dict):
                                for _full_meta_key in (
                                    'full_endpoint_mode',
                                    'full_end_node_names',
                                    'full_output_contract',
                                    'full_true_full_model',
                                    'full_prepared_baseline',
                                    'full_cached_from_preparation',
                                ):
                                    if _full_meta_key in suite_tgt:
                                        tgt_out[_full_meta_key] = suite_tgt.get(_full_meta_key)
                                if suite_tgt.get('full_endpoint_mode') or suite_tgt.get('full_end_node_names') or suite_tgt.get('full_output_contract'):
                                    manifest_out.setdefault('hailo', {})
                                    if suite_tgt.get('full_endpoint_mode'):
                                        manifest_out['hailo']['full_endpoint_mode'] = suite_tgt.get('full_endpoint_mode')
                                    if suite_tgt.get('full_end_node_names'):
                                        manifest_out['hailo']['full_end_node_names'] = list(suite_tgt.get('full_end_node_names') or [])
                                    if isinstance(suite_tgt.get('full_output_contract'), Mapping):
                                        manifest_out['hailo']['full_output_contract'] = dict(suite_tgt.get('full_output_contract') or {})
                                if suite_tgt.get('full_error'):
                                    tgt_out['full_error'] = suite_tgt.get('full_error')

                            if cfg.hef_part1:
                                out_p1 = os.path.join(case_dir, 'hailo', hw_arch, 'part1')
                                os.makedirs(out_p1, exist_ok=True)
                                r1 = cfg.hailo_build_hef_fn(
                                    p1_path,
                                    backend=cfg.hef_backend,
                                    hw_arch=hw_arch,
                                    net_name=f"{cfg.base}_part1_b{b}",
                                    outdir=out_p1,
                                    fixup=cfg.hef_fixup,
                                    opt_level=int(cfg.hef_opt_level),
                                    calib_dir=cfg.hef_calib_dir,
                                    calib_count=int(cfg.hef_calib_count),
                                    calib_batch_size=int(cfg.hef_calib_bs),
                                    force=cfg.hef_force,
                                    keep_artifacts=cfg.hef_keep,
                                    wsl_distro=cfg.hef_wsl_distro,
                                    wsl_venv_activate=cfg.hef_wsl_venv,
                                    wsl_timeout_s=int(cfg.hef_timeout_s),
                                    on_log=_on_hef_log,
                                )
                                initial_r1 = r1
                                salvage_attempted = False
                                donor_boundary: Optional[int] = None
                                donor_hints: List[str] = []
                                failure_rec_part1: Optional[HailoFailureRecord] = None
                                validator_failed_nodes: List[str] = []
                                if not r1.ok:
                                    failure_rec_part1 = classify_hailo_build_failure(r1, boundary=int(b), stage='part1', hw_arch=hw_arch)
                                    validator_failed_nodes = self._extract_validator_failed_nodes(r1)
                                    donor_boundary, donor_hints = self._nearest_row_per_cut_donor(
                                        int(b),
                                        row_per_cut_donors.get(str(hw_arch).strip()) or {},
                                        radius=int(getattr(cfg, 'hailo_salvage_neighbor_radius', 48) or 48),
                                    )
                                    if bool(getattr(cfg, 'hailo_salvage_enable', True)) and self._should_attempt_hailo_salvage(
                                        failure_rec=failure_rec_part1,
                                        validator_nodes=validator_failed_nodes,
                                        donor_hints=donor_hints,
                                        stage='part1',
                                        hw_arch=hw_arch,
                                    ):
                                        salvage_attempted = True
                                        salvage_script = self._build_row_per_cut_salvage_script(donor_hints)
                                        tgt_out['part1_initial_build'] = self.generation_service.compact_hailo_build_summary(initial_r1)
                                        tgt_out['part1_salvage'] = {
                                            'attempted': True,
                                            'reason_family': str(getattr(failure_rec_part1, 'family', '') or ''),
                                            'donor_boundary': int(donor_boundary) if donor_boundary is not None else None,
                                            'row_per_cut_hints': list(donor_hints),
                                            'validator_failed_nodes': list(validator_failed_nodes),
                                            'strategy': 'nearby_row_per_cut_hint_retry',
                                            'extra_model_script': salvage_script.strip(),
                                        }
                                        log(
                                            f"b{b}: part1 salvage retry for {hw_arch} using nearby ROW_PER_CUT hints "
                                            f"from b{donor_boundary} ({', '.join(list(donor_hints)[:6])})"
                                        )
                                        r1_retry = cfg.hailo_build_hef_fn(
                                            p1_path,
                                            backend=cfg.hef_backend,
                                            hw_arch=hw_arch,
                                            net_name=f"{cfg.base}_part1_b{b}",
                                            outdir=out_p1,
                                            fixup=cfg.hef_fixup,
                                            opt_level=int(cfg.hef_opt_level),
                                            calib_dir=cfg.hef_calib_dir,
                                            calib_count=int(cfg.hef_calib_count),
                                            calib_batch_size=int(cfg.hef_calib_bs),
                                            force=True,
                                            keep_artifacts=True,
                                            wsl_distro=cfg.hef_wsl_distro,
                                            wsl_venv_activate=cfg.hef_wsl_venv,
                                            wsl_timeout_s=int(cfg.hef_timeout_s),
                                            on_log=_on_hef_log,
                                            extra_model_script=salvage_script,
                                        )
                                        tgt_out['part1_salvage_build'] = self.generation_service.compact_hailo_build_summary(r1_retry)
                                        tgt_out['part1_salvage']['ok'] = bool(r1_retry.ok)
                                        if r1_retry.ok:
                                            log(f"b{b}: HEF(part1,{hw_arch}) OK after salvage retry")
                                            r1 = r1_retry
                                        else:
                                            log(f"b{b}: HEF(part1,{hw_arch}) salvage retry {_hef_failure_label(r1_retry)}: {r1_retry.error}")
                                            r1 = r1_retry

                                tgt_out['part1_build'] = self.generation_service.compact_hailo_build_summary(r1)
                                _cost_note_p1 = self.generation_service.hailo_build_cost_note(tgt_out.get('part1_build') or {}, stage=f'HEF(part1,{hw_arch})')
                                if _cost_note_p1:
                                    tgt_out['part1_cost_note'] = _cost_note_p1
                                    log(f"b{b}: {_cost_note_p1}")
                                if r1.ok:
                                    rel = os.path.relpath(r1.hef_path or os.path.join(out_p1, 'compiled.hef'), case_dir)
                                    tgt_out['part1'] = rel.replace('\\', '/')
                                    if not salvage_attempted:
                                        log(f"b{b}: HEF(part1,{hw_arch}) OK")
                                    learned_hints = self._extract_row_per_cut_hints(r1)
                                    if learned_hints:
                                        row_per_cut_donors.setdefault(str(hw_arch).strip(), {})[int(b)] = list(learned_hints)
                                        tgt_out['part1_salvage_hints'] = {'row_per_cut_hints': list(learned_hints)}
                                else:
                                    tgt_out['part1_error'] = r1.error
                                    err_line = f"b{b}: HEF(part1,{hw_arch}) {_hef_failure_label(r1)}: {r1.error}"
                                    errors.append(err_line)
                                    log(err_line)
                                    failure_rec_part1 = classify_hailo_build_failure(r1, boundary=int(b), stage='part1', hw_arch=hw_arch)
                                    if failure_rec_part1.clusterable:
                                        hailo_failure_records.append(failure_rec_part1)
                                    if case_first_rejection is None:
                                        case_first_rejection = build_benchmark_case_rejection(
                                            boundary=int(b),
                                            folder=folder,
                                            reason='hailo_hef_build_failed',
                                            stage='part1',
                                            hw_arch=hw_arch,
                                            detail=r1.error,
                                            hef_result=r1,
                                        )
                                self._maybe_publish_diag(cb, f"benchmark b{b} part1 @ {hw_arch}", r1, log)

                            if cfg.hef_part2:
                                out_p2 = os.path.join(case_dir, 'hailo', hw_arch, 'part2')
                                os.makedirs(out_p2, exist_ok=True)
                                if part2_output_strategy != 'original' or effective_part2_outputs or part2_output_contract:
                                    tgt_out['part2_output_strategy'] = str(part2_output_strategy)
                                    if effective_part2_outputs:
                                        tgt_out['part2_effective_outputs'] = list(effective_part2_outputs)
                                    if part2_output_contract:
                                        _p2_contract = dict(part2_output_contract)
                                        if effective_part2_outputs and not _p2_contract.get('output_names'):
                                            _p2_contract['output_names'] = list(effective_part2_outputs)
                                        if part2_host_tail_required:
                                            _p2_contract['host_tail_required'] = True
                                        tgt_out['part2_output_contract'] = _p2_contract
                                p2_build_path = p2_hailo_accel_path or p2_path
                                if p2_hailo_accel_path:
                                    tgt_out['part2_accel_model'] = os.path.relpath(p2_hailo_accel_path, case_dir).replace('\\', '/')
                                    tgt_out['part2_accelerator_model'] = os.path.relpath(p2_hailo_accel_path, case_dir).replace('\\', '/')
                                    if p2_hailo_accel_pruned_inputs:
                                        tgt_out['part2_accel_pruned_unused_inputs'] = list(p2_hailo_accel_pruned_inputs)
                                    tgt_out['part2_host_tail_required'] = bool(part2_host_tail_required or p2_host_tail_path)
                                    if p2_host_tail_path:
                                        tgt_out['part2_host_tail_model'] = os.path.relpath(p2_host_tail_path, case_dir).replace('\\', '/')
                                    log(f"b{b}: building HEF(part2,{hw_arch}) from accelerator prefix {os.path.basename(p2_build_path)}")
                                if skip_hailo_part2_build:
                                    class _SkippedHailoPart2Build:
                                        ok = False
                                        skipped = True
                                        timed_out = False
                                        hef_path = None
                                        elapsed_s = 0.0
                                        context_count = None
                                        detected = {}
                                        process = {}
                                        result_json_path = None
                                        fixed_onnx_path = None
                                        parsed_har_path = None
                                        quant_har_path = None
                                        calib_info = None
                                        error = 'Hailo Part2 skipped by YOLO26 end-to-end head policy; use TensorRT/ORT for Part2.'
                                    r2 = _SkippedHailoPart2Build()
                                    tgt_out['part2_skipped_by_policy'] = True
                                    tgt_out['part2_policy'] = str(part2_output_strategy or 'policy_skip')
                                    if hailo_part2_policy_detail:
                                        tgt_out['part2_policy_detail'] = hailo_part2_policy_detail
                                    log(f"b{b}: HEF(part2,{hw_arch}) SKIPPED by policy ({part2_output_strategy}); Hailo-Part1 remains usable")
                                else:
                                    r2 = cfg.hailo_build_hef_fn(
                                        p2_build_path,
                                        backend=cfg.hef_backend,
                                        hw_arch=hw_arch,
                                        net_name=(f"{cfg.base}_part2_hailo_prefix_b{b}" if p2_hailo_accel_path else f"{cfg.base}_part2_b{b}"),
                                        outdir=out_p2,
                                        fixup=cfg.hef_fixup,
                                        opt_level=int(cfg.hef_opt_level),
                                        calib_dir=cfg.hef_calib_dir,
                                        calib_count=int(cfg.hef_calib_count),
                                        calib_batch_size=int(cfg.hef_calib_bs),
                                        activation_part1_onnx=p1_path,
                                        activation_gen_batch=int(cfg.hef_calib_bs),
                                        force=cfg.hef_force,
                                        keep_artifacts=cfg.hef_keep,
                                        wsl_distro=cfg.hef_wsl_distro,
                                        wsl_venv_activate=cfg.hef_wsl_venv,
                                        wsl_timeout_s=int(cfg.hef_timeout_s),
                                        on_log=_on_hef_log,
                                    )
                                tgt_out['part2_build'] = self.generation_service.compact_hailo_build_summary(r2)
                                _cost_note_p2 = self.generation_service.hailo_build_cost_note(tgt_out.get('part2_build') or {}, stage=f'HEF(part2,{hw_arch})')
                                if _cost_note_p2:
                                    tgt_out['part2_cost_note'] = _cost_note_p2
                                    log(f"b{b}: {_cost_note_p2}")
                                if r2.ok:
                                    rel = os.path.relpath(r2.hef_path or os.path.join(out_p2, 'compiled.hef'), case_dir)
                                    tgt_out['part2'] = rel.replace('\\', '/')
                                    if str(part2_output_strategy or '').strip() == 'raw_detection_head_part2':
                                        log(f"b{b}: HEF(part2,{hw_arch}) OK (raw detection-head; DFL/decode tail on host)")
                                    else:
                                        log(f"b{b}: HEF(part2,{hw_arch}) OK")
                                else:
                                    if skip_hailo_part2_build and bool(getattr(r2, 'skipped', False)):
                                        # Policy skips are not failures: the case can still be used for
                                        # Hailo-Part1 -> TensorRT/ORT-Part2 pipelines.  Do not create a
                                        # rejection entry or the case will look semantically failed.
                                        tgt_out['part2_error'] = None
                                        tgt_out['part2_skipped_by_policy'] = True
                                    else:
                                        tgt_out['part2_error'] = r2.error
                                        err_line = f"b{b}: HEF(part2,{hw_arch}) {_hef_failure_label(r2)}: {r2.error}"
                                        errors.append(err_line)
                                        log(err_line)
                                        failure_rec = classify_hailo_build_failure(r2, boundary=int(b), stage='part2', hw_arch=hw_arch)
                                        if failure_rec.clusterable:
                                            hailo_failure_records.append(failure_rec)
                                        if case_first_rejection is None:
                                            case_first_rejection = build_benchmark_case_rejection(
                                                boundary=int(b),
                                                folder=folder,
                                                reason='hailo_hef_build_failed',
                                                stage='part2',
                                                hw_arch=hw_arch,
                                                detail=r2.error,
                                                hef_result=r2,
                                            )
                                if not skip_hailo_part2_build:
                                    self._maybe_publish_diag(cb, f"benchmark b{b} part2 @ {hw_arch}", r2, log)

                            _raise_if_cancelled(f"after Hailo HEF build b{b}")

                            if tgt_out:
                                manifest_out['hailo']['hefs'][hw_arch] = dict(tgt_out)
                    finally:
                        if old_debug_env in (None, ''):
                            os.environ.pop(debug_env_key, None)
                        else:
                            os.environ[debug_env_key] = old_debug_env

                    case_hefs_payload = ((manifest_out.get('hailo') or {}).get('hefs') if isinstance(manifest_out.get('hailo'), dict) else None)
                    case_variant_availability = build_case_hailo_variant_availability(runtime.suite_hailo_hefs, case_hefs_payload or {})
                    if case_variant_availability:
                        manifest_out['hailo']['case_variant_availability'] = dict(case_variant_availability)
                    if case_first_rejection is not None:
                        if cfg.bench_plan_runs:
                            case_usable = case_has_usable_hailo_variant(cfg.bench_plan_runs, case_variant_availability)
                        else:
                            case_usable = any(
                                bool(meta.get(kind))
                                for meta in case_variant_availability.values()
                                for kind in ('full', 'part1', 'part2', 'composed')
                            )
                        if case_usable:
                            availability_bits: List[str] = []
                            for hw_arch, meta in sorted(case_variant_availability.items()):
                                enabled = [kind for kind in ('full', 'part1', 'part2', 'composed') if bool(meta.get(kind))]
                                if enabled:
                                    availability_bits.append(f"{hw_arch}: {','.join(enabled)}")
                            keep_msg = (
                                f"b{b}: keeping partial Hailo case; usable Hailo variants remain "
                                f"({'; '.join(availability_bits) if availability_bits else 'partial artifacts available'})"
                            )
                            log(keep_msg)
                            manifest_out['hailo']['partial_keep'] = True
                            manifest_out['hailo']['partial_keep_reason'] = keep_msg
                        else:
                            case_rejection = dict(case_first_rejection)

            if case_rejection is not None:
                manifest_out['benchmark_status'] = 'rejected'
                manifest_out['benchmark_rejection'] = dict(case_rejection)
                write_benchmark_json_atomic(Path(manifest_path), manifest_out)
                archive_info = archive_benchmark_case(case_dir, str(cfg.out_dir), folder=folder)
                if isinstance(archive_info, dict):
                    case_rejection.update({k: v for k, v in archive_info.items() if v not in (None, '')})
                    manifest_out['benchmark_rejection'] = dict(case_rejection)
                    archive_dir = archive_info.get('archive_dir')
                    if archive_dir:
                        try:
                            archived_manifest_path = os.path.join(str(cfg.out_dir), str(archive_dir), os.path.basename(manifest_path))
                            write_benchmark_json_atomic(Path(archived_manifest_path), manifest_out)
                        except Exception:
                            pass
                        log(f"b{b}: rejected case archived to {archive_dir}")
                discarded_cases.append(dict(case_rejection))
                discarded_boundaries.add(int(b))
                completed_boundaries.add(int(b))
                _persist(status='running', current_boundary=int(b))
                qput(("prog", made, f"b{b} (reject: Hailo build failed)"))
                continue

            write_benchmark_json_atomic(Path(manifest_path), manifest_out)

            case_entry = {
                'boundary': int(b),
                'case_dir': folder,
                'folder': folder,
                'manifest': os.path.basename(manifest_path),
                'predicted': pred,
            }
            hailo_compile_meta = self.generation_service.case_hailo_compile_from_hefs(
                ((manifest_out.get('hailo') or {}).get('hefs') if isinstance(manifest_out.get('hailo'), dict) else None)
            )
            if hailo_compile_meta:
                case_entry['hailo_compile'] = hailo_compile_meta
            if case_variant_availability:
                case_entry['hailo_case_variant_availability'] = dict(case_variant_availability)
            if hailo_parse_entry is not None:
                case_entry['hailo_parse_check'] = hailo_parse_entry
            if part2_output_strategy != 'original' or effective_part2_outputs or part2_output_contract:
                case_entry['hailo_part2_output_strategy'] = str(part2_output_strategy)
                if effective_part2_outputs:
                    case_entry['hailo_part2_effective_outputs'] = list(effective_part2_outputs)
                if part2_output_contract:
                    case_entry['hailo_part2_output_contract'] = dict(part2_output_contract)
            case_entry.update(hailo_parse_fields)
            cases.append(case_entry)
            accepted_boundaries.add(int(b))
            completed_boundaries.add(int(b))
            chosen.append(int(b))
            made += 1
            _persist(status='running', current_boundary=int(b))
            qput(("prog", made, f"b{b}"))
            _raise_if_cancelled(f"after case b{b}")

        return chosen


@dataclass
class BenchmarkGenerationOrchestrationConfig:
    runtime: BenchmarkGenerationRuntime
    execution_cfg: BenchmarkGenerationExecutionConfig
    execution_callbacks: BenchmarkGenerationExecutionCallbacks
    target_cases: int
    preferred_shortlist_original: List[int]
    ranked_candidates: List[int]
    candidate_search_pool: List[int]
    out_dir: Path
    base: str
    pad: int
    full_model_src: str
    full_model_dst: str
    analysis_payload: Mapping[str, Any]
    analysis_params_payload: Mapping[str, Any]
    system_spec_payload: Optional[Mapping[str, Any]]
    bench_log_path: str
    bench_plan_runs: Sequence[Mapping[str, Any]]
    hef_targets: List[str]
    hef_full: bool
    hef_part1: bool
    hef_part2: bool
    hef_backend: str
    hef_fixup: bool
    hef_opt_level: int
    hef_calib_dir: Optional[str]
    hef_calib_count: int
    hef_calib_bs: int
    hef_force: bool
    hef_keep: bool
    hef_wsl_distro: Optional[str]
    hef_wsl_venv: str
    hef_timeout_s: int
    full_hef_policy: str
    full_model_preflight_policy: str = 'enabled'
    hailo_full_end_node_names: Sequence[str] = field(default_factory=list)
    hailo_full_endpoint_mode: str = ''
    hailo_full_output_contract: Optional[Mapping[str, Any]] = None
    prepared_full_hailo_baseline: Optional[Mapping[str, Any]] = None
    hailo_build_hef_fn: Any = None
    hailo_parse_check_fn: Any = None
    hailo_build_unavailable: Optional[str] = None
    hailo_part2_precheck_fn: Any = None
    hailo_part2_precheck_error_fn: Any = None
    hailo_part2_parser_precheck_fn: Any = None
    hailo_part2_parser_precheck_error_fn: Any = None
    resume_generation: bool = False
    resume_report_summary_lines: Sequence[str] = field(default_factory=list)
    hailo_selected: bool = False
    hailo_outlook_summary: Any = None
    benign_discard_reasons: Sequence[str] = field(default_factory=lambda: ["hailo_part2_prefilter", "hailo_part2_precheck", "hailo_part2_auto_filtered", "hailo_part2_parser_prefilter", "hailo_part2_parser_auto_filtered", "hailo_part2_concat_sanity_prefilter", "hailo_part2_concat_sanity_auto_filtered", "hailo_failure_cluster_skip"])
    write_harness_script: Any = None
    copy_schema_tree: Any = None
    tool_gui_version: str = "?"
    tool_core_version: str = "?"
    evaluation_profile_meta: Optional[Mapping[str, Any]] = None
    benchmark_objective: str = "latency"
    should_cancel: Optional[Callable[[], bool]] = None


@dataclass
class BenchmarkGenerationOrchestrationResult:
    final_status: str
    final_msg: str
    bench_payload: Dict[str, Any]
    harness_path: str
    ranked_candidates: List[int]
    candidate_search_pool: List[int]
    shortlist_prefiltered_boundaries: List[int] = field(default_factory=list)
    summary_data: Dict[str, Any] = field(default_factory=dict)


class BenchmarkGenerationOrchestrationService:
    """Service-layer orchestration for benchmark-set generation.

    The GUI still owns dialogs and threading, but the top-level benchmark-set
    orchestration (prefilter -> optional suite full HEFs -> case loop ->
    finalize summary) lives here instead of gui_app.py.
    """

    def __init__(self, generation_service: Optional[BenchmarkGenerationService] = None, execution_service: Optional[BenchmarkGenerationExecutionService] = None):
        self.generation_service = generation_service or BenchmarkGenerationService()
        self.execution_service = execution_service or BenchmarkGenerationExecutionService(self.generation_service)

    def _raise_if_cancelled(self, cfg: BenchmarkGenerationOrchestrationConfig, stage: str = "") -> None:
        try:
            requested = bool(callable(cfg.should_cancel) and cfg.should_cancel())
        except Exception:
            requested = False
        if not requested:
            return
        detail = f" ({stage})" if str(stage or "").strip() else ""
        raise BenchmarkGenerationCancelled(f"Benchmark-set generation cancelled by user{detail}")

    def _force_yolo26_suite_full_baseline_if_needed(
        self,
        cfg: BenchmarkGenerationOrchestrationConfig,
        *,
        log: Callable[[str], None],
    ) -> BenchmarkGenerationOrchestrationConfig:
        """Ensure YOLO26 benchmark sets carry an explicit Hailo Full baseline.

        Some user-facing benchmark presets focus on split/matrix runs and can
        arrive here with ``hef_full=False`` even though Hailo is selected. For
        YOLO26 this leaves the generated suite without the single-system Hailo
        baseline needed to compare Hailo->TensorRT throughput against Hailo-only.
        Unless the user explicitly selected the full-HEF policy ``skip``, enable
        the suite-level Full HEF build and let the normal decoded-full ->
        one2one/raw-head fallback decide whether the artifact is feasible.
        """
        try:
            full_policy = normalize_full_hef_policy(cfg.full_hef_policy)
        except Exception:
            full_policy = str(cfg.full_hef_policy or 'end').strip().lower() or 'end'
        try:
            is_yolo26 = bool(self._is_yolo26_orchestration_cfg(cfg))
        except Exception:
            is_yolo26 = False
        effective_targets = self._effective_hailo_targets(cfg)
        if (
            is_yolo26
            and bool(effective_targets)
            and not bool(cfg.hef_full)
        ):
            log(
                'suite: YOLO26 policy: enabling Hailo Full baseline even though the selected run variants did not request a Full HEF; '
                'decoded full will be tried first and one2one raw-head endpoints will be used as fallback if needed.'
            )
            return replace(
                cfg,
                hailo_selected=True,
                hef_targets=list(effective_targets),
                hef_full=True,
                execution_cfg=replace(cfg.execution_cfg, hef_targets=list(effective_targets)),
            )
        return cfg


    def _prefilter_shortlist_for_hailo_part2(self, cfg: BenchmarkGenerationOrchestrationConfig, *, discarded_cases: List[Dict[str, Any]], discarded_boundaries: Set[int], log: Callable[[str], None]) -> Tuple[List[int], List[int], Set[int]]:
        ranked_candidates = list(cfg.ranked_candidates)
        candidate_search_pool = list(cfg.candidate_search_pool)
        shortlist_prefiltered_boundaries: Set[int] = set()
        if not ranked_candidates or not (cfg.hef_targets and cfg.hef_part2):
            return ranked_candidates, candidate_search_pool, shortlist_prefiltered_boundaries

        target_label = ",".join([str(x).strip() for x in cfg.hef_targets if str(x).strip()]) or "hailo"
        log(f"Prefiltering preferred shortlist for Hailo Part2 compatibility ({len(ranked_candidates)} candidates)...")
        filtered: List[int] = []
        for idx, raw_b in enumerate(ranked_candidates, start=1):
            self._raise_if_cancelled(cfg, "shortlist prefilter")
            b = int(raw_b)
            try:
                probe = self.execution_service.probe_hailo_part2_support(cfg.execution_cfg, b, log_cb=log)
            except Exception as exc:
                log(f"b{b}: shortlist prefilter skipped ({type(exc).__name__}: {exc})", level=logging.WARNING)
                filtered.append(b)
                continue

            if probe.get('compatible') is False:
                probe_reason = str(probe.get('reason') or '').strip()
                if probe_reason == 'activation':
                    reason_key = 'hailo_part2_prefilter'
                    info_payload = probe.get('activation_precheck')
                elif probe_reason == 'concat_sanity':
                    reason_key = 'hailo_part2_concat_sanity_prefilter'
                    info_payload = probe.get('concat_sanity_precheck')
                else:
                    reason_key = 'hailo_part2_parser_prefilter'
                    info_payload = probe.get('parser_precheck')
                detail = str(probe.get('detail') or 'Unsupported Hailo Part2 split')
                rec = self.execution_service._record_hailo_part2_filter(
                    b,
                    pad=int(cfg.pad),
                    detail=detail,
                    target_label=target_label,
                    reason=reason_key,
                    info=(info_payload if isinstance(info_payload, dict) else None),
                )
                if probe.get('effective_part2_outputs'):
                    rec['effective_part2_outputs'] = list(probe.get('effective_part2_outputs') or [])
                if isinstance(probe.get('part2_output_contract'), Mapping):
                    rec['part2_output_contract'] = dict(probe.get('part2_output_contract') or {})
                discarded_cases.append(rec)
                discarded_boundaries.add(int(b))
                shortlist_prefiltered_boundaries.add(int(b))
                log(f"b{b}: preferred shortlist auto-skip ({detail})")
            else:
                if bool(probe.get('skip_hailo_part2_build')):
                    log(
                        f"b{b}: preferred shortlist kept for Hailo-Part1 -> TRT/ORT; "
                        f"Hailo Part2 skipped by policy ({probe.get('hailo_part2_policy') or probe.get('strategy')})"
                    )
                elif str(probe.get('strategy') or 'original') != 'original':
                    log(
                        f"b{b}: preferred shortlist kept via alternative Hailo Part2 end nodes "
                        f"{list(probe.get('effective_part2_outputs') or [])}"
                    )
                filtered.append(b)
            if idx % 10 == 0 or idx == len(ranked_candidates):
                log(f"Prefilter progress: {idx}/{len(ranked_candidates)} checked")

        if shortlist_prefiltered_boundaries:
            ranked_candidates = list(filtered)
            candidate_search_pool = [int(x) for x in candidate_search_pool if int(x) not in shortlist_prefiltered_boundaries]
            log(
                f"Preferred shortlist after Hailo Part2 prefilter: {len(ranked_candidates)} kept, "
                f"{len(shortlist_prefiltered_boundaries)} auto-skipped"
            )
        return ranked_candidates, candidate_search_pool, shortlist_prefiltered_boundaries

    def _probe_any_hailo_part2_compatible(self, cfg: BenchmarkGenerationOrchestrationConfig, candidate_search_pool: Sequence[int], *, log: Callable[[str], None]) -> bool:
        try:
            self._last_true_hailo_part2_candidate = None
            self._last_true_hailo_part2_strategy = None
        except Exception:
            pass
        if not (cfg.hef_targets and cfg.hef_part2):
            return True
        checked = 0
        for raw_b in list(candidate_search_pool or []):
            self._raise_if_cancelled(cfg, "global Hailo Part2 probe")
            b = int(raw_b)
            checked += 1
            try:
                probe = self.execution_service.probe_hailo_part2_support(cfg.execution_cfg, b, log_cb=None)
            except Exception as exc:
                log(f"b{b}: global Hailo Part2 probe skipped ({type(exc).__name__}: {exc})", level=logging.WARNING)
                continue
            if probe.get('compatible'):
                if bool(probe.get('skip_hailo_part2_build')):
                    if checked <= 5:
                        log(
                            f"Hailo Part2 global probe: b{b} is usable only as Hailo-Part1 -> host/TRT Part2 "
                            f"({probe.get('hailo_part2_policy') or probe.get('strategy')}); continuing search for a true Hailo-Part2 candidate"
                        )
                    continue
                try:
                    self._last_true_hailo_part2_candidate = int(b)
                    self._last_true_hailo_part2_strategy = str(probe.get('strategy') or 'original')
                except Exception:
                    pass
                if str(probe.get('strategy') or 'original') != 'original':
                    log(
                        f"Hailo Part2 global probe: found compatible candidate b{b} via {probe.get('strategy')} "
                        f"({list(probe.get('effective_part2_outputs') or [])})"
                    )
                else:
                    log(f"Hailo Part2 global probe: found compatible candidate b{b}")
                return True
            if checked % 25 == 0:
                log(f"Hailo Part2 global probe progress: {checked}/{len(candidate_search_pool)} checked")
        return False

    def _recompute_hailo_plan_requirements(self, plan_runs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        hailo_targets_set: Set[str] = set()
        need_full = False
        need_part1 = False
        need_part2 = False
        for run in list(plan_runs or []):
            if not isinstance(run, Mapping):
                continue
            vv = [str(x).strip().lower() for x in list(run.get('variants') or []) if str(x).strip()]
            if not vv:
                continue
            vset = set(vv)
            st1_raw = run.get('stage1')
            st2_raw = run.get('stage2')
            st1 = st1_raw if isinstance(st1_raw, Mapping) else {}
            st2 = st2_raw if isinstance(st2_raw, Mapping) else {}
            st1_hw = str(st1.get('hw_arch') or st1.get('arch') or st1.get('id') or '').strip()
            st2_hw = str(st2.get('hw_arch') or st2.get('arch') or st2.get('id') or '').strip()
            if not st1_hw and not isinstance(st1_raw, Mapping):
                st1_hw = self._normalize_hailo_hw_arch_for_build(st1_raw) or ''
            if not st2_hw and not isinstance(st2_raw, Mapping):
                st2_hw = self._normalize_hailo_hw_arch_for_build(st2_raw) or ''
            st1_h = (str(st1.get('type') or '').strip().lower() == 'hailo') or bool(self._normalize_hailo_hw_arch_for_build(st1_hw))
            st2_h = (str(st2.get('type') or '').strip().lower() == 'hailo') or bool(self._normalize_hailo_hw_arch_for_build(st2_hw))
            is_hailo_run = str(run.get('type') or '').strip().lower() == 'hailo'
            run_hw = str(run.get('hw_arch') or run.get('id') or '').strip()
            if 'full' in vset and (is_hailo_run or st1_h or st2_h):
                need_full = True
                hw = self._normalize_hailo_hw_arch_for_build(run_hw or st1_hw or st2_hw)
                if hw:
                    hailo_targets_set.add(hw)
            if st1_h and ('part1' in vset or 'composed' in vset):
                need_part1 = True
                hw = self._normalize_hailo_hw_arch_for_build(st1_hw)
                if hw:
                    hailo_targets_set.add(hw)
            if st2_h and ('part2' in vset or 'composed' in vset):
                need_part2 = True
                hw = self._normalize_hailo_hw_arch_for_build(st2_hw)
                if hw:
                    hailo_targets_set.add(hw)
        hef_targets = sorted(str(x).strip() for x in hailo_targets_set if str(x).strip())
        return {
            'hef_targets': list(hef_targets),
            'hailo_selected': bool(hef_targets),
            'hef_full': bool(hef_targets and need_full),
            'hef_part1': bool(hef_targets and need_part1),
            'hef_part2': bool(hef_targets and need_part2),
        }

    def _replace_plan_runs(self, cfg: BenchmarkGenerationOrchestrationConfig, plan_runs_out: Sequence[Mapping[str, Any]]) -> BenchmarkGenerationOrchestrationConfig:
        requirements = self._recompute_hailo_plan_requirements(plan_runs_out)
        return replace(
            cfg,
            bench_plan_runs=list(plan_runs_out),
            hef_targets=list(requirements.get('hef_targets') or []),
            hailo_selected=bool(requirements.get('hailo_selected')),
            hef_full=bool(requirements.get('hef_full')),
            hef_part1=bool(requirements.get('hef_part1')),
            hef_part2=bool(requirements.get('hef_part2')),
            execution_cfg=replace(
                cfg.execution_cfg,
                bench_plan_runs=list(plan_runs_out),
                hef_targets=list(requirements.get('hef_targets') or []),
                hef_part1=bool(requirements.get('hef_part1')),
                hef_part2=bool(requirements.get('hef_part2')),
            ),
        )

    def _normalize_hailo_hw_arch_for_build(self, value: Any) -> Optional[str]:
        """Return the physical Hailo DFC hw_arch for a run/profile token.

        Benchmark run ids such as ``hailo8_to_trt`` and ``trt_to_hailo8`` are
        *pipeline labels*, not Hailo DFC architectures.  DFC only accepts real
        chips such as ``hailo8``, ``hailo8r`` or ``hailo8l``.  Keep mixed run
        labels in ``bench_plan_runs`` for runtime/reporting, but normalize all
        HEF build targets to the physical Hailo architecture so generated suites
        do not waste time on invalid ``hw_arch`` values.
        """
        text = str(value or '').strip().lower()
        if not text:
            return None
        # Preserve explicit concrete chip variants.
        if text in {'hailo8', 'hailo8r', 'hailo8l'}:
            return text
        # Mixed run labels / stage labels usually contain the concrete chip name.
        if 'hailo8r' in text:
            return 'hailo8r'
        if 'hailo8l' in text:
            return 'hailo8l'
        if 'hailo8' in text:
            return 'hailo8'
        # Future-proof lightly for Hailo-10 tokens without treating arbitrary
        # pipeline ids as DFC archs.
        if text in {'hailo10', 'hailo10h'}:
            return text
        if 'hailo10h' in text:
            return 'hailo10h'
        if 'hailo10' in text:
            return 'hailo10'
        return None

    def _physical_hailo_targets_from_values(self, values: Sequence[Any]) -> List[str]:
        targets: List[str] = []
        for raw in list(values or []):
            hw = self._normalize_hailo_hw_arch_for_build(raw)
            if hw and hw not in targets:
                targets.append(hw)
        return targets

    def _effective_hailo_targets(self, cfg: BenchmarkGenerationOrchestrationConfig) -> List[str]:
        """Return physical Hailo build targets from orchestration, execution config, and plan runs."""
        targets: List[str] = []

        def _add(value: Any) -> None:
            hw = self._normalize_hailo_hw_arch_for_build(value)
            if hw and hw not in targets:
                targets.append(hw)

        for raw in list(getattr(cfg, 'hef_targets', None) or []):
            _add(raw)
        try:
            for raw in list(getattr(cfg.execution_cfg, 'hef_targets', None) or []):
                _add(raw)
        except Exception:
            pass
        for run in list(getattr(cfg, 'bench_plan_runs', None) or []):
            if not isinstance(run, Mapping):
                continue
            if str(run.get('type') or '').strip().lower() == 'hailo':
                _add(run.get('hw_arch') or run.get('arch') or run.get('id') or 'hailo8')
            for key in ('stage1', 'stage2'):
                st_raw = run.get(key)
                if isinstance(st_raw, Mapping):
                    st = st_raw
                    if str(st.get('type') or '').strip().lower() == 'hailo':
                        _add(st.get('hw_arch') or st.get('arch') or st.get('id') or 'hailo8')
                else:
                    # Some evaluation profiles carry plain strings such as
                    # stage1: hailo8 / stage2: tensorrt.  Treat those as Hailo
                    # only when they actually contain a Hailo chip token.
                    _add(st_raw)
        return targets

    def _copy_run_common_fields(self, source_run: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
        """Copy validation/runtime fields from an existing benchmark run."""
        src = source_run if isinstance(source_run, Mapping) else {}
        out: Dict[str, Any] = {}
        for key in (
            'image_scale', 'validation_images', 'validation_max_images',
            'validation_reference_mode', 'mini_coco_ap50', 'benchmark_task',
            'mini_classification_eval',
        ):
            if key in src:
                out[key] = src.get(key)
        out.setdefault('image_scale', 'auto')
        out.setdefault('validation_reference_mode', 'auto')
        out.setdefault('benchmark_task', 'auto')
        return out

    def _is_yolo26_orchestration_cfg(self, cfg: BenchmarkGenerationOrchestrationConfig) -> bool:
        """Best-effort YOLO26 detection at orchestration level.

        Some GUI paths expose the model name/path on the orchestration config
        while the nested execution config is less descriptive.  Inspect both so
        the early Full-Hailo baseline hook cannot silently miss YOLO26.
        """
        raw_values: List[str] = []
        for attr in ('base', 'full_model_src', 'full_model_dst'):
            try:
                value = getattr(cfg, attr, '')
            except Exception:
                value = ''
            if value not in (None, ''):
                raw_values.append(str(value))
        try:
            exec_cfg = getattr(cfg, 'execution_cfg', None)
        except Exception:
            exec_cfg = None
        if exec_cfg is not None:
            for attr in ('base', 'full_model_src', 'full_model_dst'):
                try:
                    value = getattr(exec_cfg, attr, '')
                except Exception:
                    value = ''
                if value not in (None, ''):
                    raw_values.append(str(value))
            try:
                if bool(self.execution_service._is_yolo26_cfg(exec_cfg)):
                    return True
            except Exception:
                pass
        for payload in (getattr(cfg, 'analysis_payload', None), getattr(cfg, 'analysis_params_payload', None)):
            if isinstance(payload, Mapping):
                stack: List[Any] = [payload]
                seen = 0
                while stack and seen < 80:
                    seen += 1
                    item = stack.pop()
                    if isinstance(item, Mapping):
                        for key, val in item.items():
                            key_l = str(key).lower()
                            if any(tok in key_l for tok in ('model', 'onnx', 'path', 'family', 'name', 'id')) and val not in (None, ''):
                                raw_values.append(str(val))
                            if isinstance(val, (Mapping, list, tuple)):
                                stack.append(val)
                    elif isinstance(item, (list, tuple)):
                        stack.extend(list(item)[:20])
        joined = ' '.join(raw_values).lower().replace('\\', '/').replace('-', '').replace('_', '')
        return 'yolo26' in joined

    def _ensure_yolo26_full_hailo_baseline_plan(self, cfg: BenchmarkGenerationOrchestrationConfig, *, log: Callable[[str], None]) -> BenchmarkGenerationOrchestrationConfig:
        """Ensure YOLO26 suites always attempt a Hailo Full baseline.

        YOLO26 can be useful as Hailo-Part1 -> TensorRT even when Hailo Part2 is
        not available, but the evaluation still needs a Hailo-only full baseline.
        If the selected run matrix did not request a pure Hailo full variant, add
        a full-only Hailo run and set ``hef_full=True``.  The suite full builder
        still tries decoded full first and then one2one raw-head fallback.
        """
        try:
            is_yolo26 = bool(self._is_yolo26_orchestration_cfg(cfg))
        except Exception:
            is_yolo26 = False
        if not is_yolo26:
            return cfg
        original_full_policy = normalize_full_hef_policy(cfg.full_hef_policy)
        targets = self._effective_hailo_targets(cfg)
        if not targets:
            return cfg

        runs: List[Dict[str, Any]] = [dict(r) for r in list(cfg.bench_plan_runs or []) if isinstance(r, Mapping)]
        common_source = runs[0] if runs else {}
        changed = False
        touched: List[str] = []
        for hw in targets:
            found = False
            for run in runs:
                run_type = str(run.get('type') or '').strip().lower()
                run_hw = str(run.get('hw_arch') or run.get('id') or '').strip()
                if run_type == 'hailo' and run_hw == hw:
                    found = True
                    variants = [str(x).strip().lower() for x in list(run.get('variants') or []) if str(x).strip()]
                    if 'full' not in variants:
                        run['variants'] = ['full'] + variants
                        changed = True
                        touched.append(hw)
                    break
            if not found:
                base_run = self._copy_run_common_fields(common_source)
                base_run.update({
                    'id': hw,
                    'type': 'hailo',
                    'hw_arch': hw,
                    'variants': ['full'],
                    'stage1': {'type': 'hailo', 'hw_arch': hw},
                    'stage2': {'type': 'hailo', 'hw_arch': hw},
                    'yolo26_forced_full_baseline': True,
                })
                runs.append(base_run)
                changed = True
                touched.append(hw)

        needs_policy_override = (original_full_policy != 'start')
        if not changed and bool(cfg.hef_full) and bool(cfg.hailo_selected) and not needs_policy_override:
            return cfg
        requirements = self._recompute_hailo_plan_requirements(runs)
        physical_requirements = _physical_hailo_targets_for_build(list(requirements.get('hef_targets') or targets))
        if physical_requirements:
            requirements['hef_targets'] = list(physical_requirements)
        if original_full_policy == 'skip':
            log(
                'suite: YOLO26 policy: overriding full_hef_policy=skip because a Hailo Full baseline is mandatory for YOLO26 Hailo/Jetson comparison.'
            )
        log(
            'suite: YOLO26 policy: ensuring Hailo Full baseline is part of the suite plan '
            f"(targets={touched or targets}); decoded full will be tried first, then one2one raw-head fallback."
        )
        return replace(
            cfg,
            bench_plan_runs=list(runs),
            hef_targets=list(requirements.get('hef_targets') or targets),
            hailo_selected=True,
            hef_full=True,
            full_hef_policy='start',
            hef_part1=bool(requirements.get('hef_part1') or cfg.hef_part1),
            hef_part2=bool(requirements.get('hef_part2') or cfg.hef_part2),
            execution_cfg=replace(
                cfg.execution_cfg,
                bench_plan_runs=list(runs),
                hef_targets=list(requirements.get('hef_targets') or targets),
                hef_part1=bool(requirements.get('hef_part1') or cfg.hef_part1),
                hef_part2=bool(requirements.get('hef_part2') or cfg.hef_part2),
            ),
        )

    def _yolo26_should_build_full_first(self, cfg: BenchmarkGenerationOrchestrationConfig) -> bool:
        """Return True when YOLO26 Hailo full must be attempted before split builds.

        This deliberately does not depend on the user-selected full-build order.
        YOLO26 evaluation needs a single-system Hailo baseline, and previous GUI
        paths could arrive with split-only Hailo variants while still having
        Hailo targets and a build function.
        """
        try:
            is_yolo26 = bool(self._is_yolo26_orchestration_cfg(cfg))
        except Exception:
            is_yolo26 = False
        if not is_yolo26:
            return False
        return bool(self._effective_hailo_targets(cfg) and cfg.hailo_build_hef_fn is not None)

    def _build_suite_full_hefs_first_if_needed(
        self,
        cfg: BenchmarkGenerationOrchestrationConfig,
        *,
        log: Callable[[str], None],
        queue_put: Callable[[tuple], None],
        errors: List[str],
        suite_hailo_hefs: Dict[str, Dict[str, Any]],
    ) -> bool:
        """Build suite Full HEFs before any split candidate work when required.

        Returns True if the full-build stage was invoked.  The underlying builder
        records either an OK artifact or a structured full_error in
        suite_hailo_hefs; candidate split generation continues afterwards.
        """
        full_policy = normalize_full_hef_policy(cfg.full_hef_policy)
        force_yolo26_first = self._yolo26_should_build_full_first(cfg)
        should_build = bool((full_policy == 'start' or force_yolo26_first) and cfg.hef_targets and cfg.hailo_build_hef_fn is not None)
        if not should_build:
            return False
        if force_yolo26_first:
            log(
                'suite: YOLO26 policy: FULL-FIRST baseline stage active; building Hailo Full before shortlist prefilter, '
                'Part2 probes, or split HEFs. Decoded full is tried first; one2one/raw-head endpoints are used as fallback.'
            )
        else:
            log('suite: full-HEF policy=start; building suite Hailo Full baseline before split cases.')
        self._build_suite_full_hefs(
            cfg,
            log=log,
            queue_put=queue_put,
            errors=errors,
            suite_hailo_hefs=suite_hailo_hefs,
            publish_hailo_diagnostics=cfg.execution_callbacks.publish_hailo_diagnostics,
        )
        try:
            cfg.execution_callbacks.persist_state(status='running', current_boundary=None)
        except Exception:
            logger.debug('persist after suite full HEF (full-first) failed', exc_info=True)
        return True

    @staticmethod
    def _promote_boundary(seq: Sequence[int], boundary: Optional[int]) -> List[int]:
        if boundary is None:
            return list(seq or [])
        try:
            b = int(boundary)
        except Exception:
            return list(seq or [])
        out = [int(x) for x in list(seq or []) if _safe_int(x) is not None and int(x) != b]
        return [b] + out

    def _promote_yolo26_true_hailo_part2_candidate(self, cfg: BenchmarkGenerationOrchestrationConfig, ranked_candidates: Sequence[int], candidate_search_pool: Sequence[int], *, log: Callable[[str], None]) -> Tuple[List[int], List[int]]:
        """Prefer a verified YOLO26 one2one Part2-prefix case before stage1-only cases."""
        try:
            if not bool(self.execution_service._is_yolo26_cfg(cfg.execution_cfg)):
                return list(ranked_candidates), list(candidate_search_pool)
        except Exception:
            return list(ranked_candidates), list(candidate_search_pool)
        b = getattr(self, '_last_true_hailo_part2_candidate', None)
        if b is None:
            return list(ranked_candidates), list(candidate_search_pool)
        try:
            b_int = int(b)
        except Exception:
            return list(ranked_candidates), list(candidate_search_pool)
        old_pool = [int(x) for x in list(candidate_search_pool or []) if _safe_int(x) is not None]
        if b_int not in old_pool or (old_pool and old_pool[0] == b_int):
            return list(ranked_candidates), list(candidate_search_pool)
        log(
            f"suite: YOLO26 policy: promoting b{b_int} because the global probe verified a true one2one Hailo-Part2 prefix; "
            "this prioritizes at least one complete Hailo/Hailo candidate before Part1-only candidates in small benchmark sets."
        )
        return self._promote_boundary(ranked_candidates, b_int), self._promote_boundary(candidate_search_pool, b_int)

    def _insert_boundaries_after_prefix(self, seq: Sequence[int], boundaries: Sequence[int], *, prefix_keep: int = 1) -> List[int]:
        base = [int(x) for x in list(seq or []) if _safe_int(x) is not None]
        if not base:
            return base
        prefix = base[:max(0, min(int(prefix_keep), len(base)))]
        rest = [x for x in base if x not in set(prefix)]
        ins: List[int] = []
        rest_set = set(rest)
        for raw in list(boundaries or []):
            try:
                b = int(raw)
            except Exception:
                continue
            if b in rest_set and b not in ins and b not in prefix:
                ins.append(b)
        tail = [x for x in rest if x not in set(ins)]
        return prefix + ins + tail

    def _promote_yolo26_hetero_throughput_candidates(self, cfg: BenchmarkGenerationOrchestrationConfig, ranked_candidates: Sequence[int], candidate_search_pool: Sequence[int], *, log: Callable[[str], None]) -> Tuple[List[int], List[int]]:
        """Reserve YOLO26 early Hailo->TRT candidates for throughput evaluation.

        The v46 full-first policy correctly protects the Full-Hailo baseline and
        at least one complete Hailo/Hailo candidate, but for larger YOLO26 runs
        the thesis question is usually heterogeneous throughput.  Older YOLO26M/L
        runs showed that early Part1-only cuts, especially b041 when available,
        are more informative for Hailo->TensorRT streaming than late one2one-tail
        cuts.  Keep the verified Hailo/Hailo candidate first, then reserve a few
        early candidates without forcing Hailo Part2.
        """
        try:
            is_yolo26 = bool(self._is_yolo26_orchestration_cfg(cfg))
        except Exception:
            is_yolo26 = False
        if not is_yolo26 or not _benchmark_plan_has_hailo_to_trt(cfg.bench_plan_runs):
            return list(ranked_candidates), list(candidate_search_pool)
        try:
            if int(cfg.target_cases) < 3:
                return list(ranked_candidates), list(candidate_search_pool)
        except Exception:
            pass
        # Historical stable early candidates from YOLO26M/L logs, followed by
        # nearby early candidates that are likely to keep Hailo Stage1 cheap.
        preferred = [41, 43, 38, 69, 72, 63, 66, 105, 113, 140, 151, 176, 216, 218, 248, 268]
        pool_set = {int(x) for x in list(candidate_search_pool or []) if _safe_int(x) is not None}
        kept = [b for b in preferred if b in pool_set]
        if not kept:
            return list(ranked_candidates), list(candidate_search_pool)
        log(
            'suite: YOLO26 throughput-first reserve: promoting early Hailo->TensorRT candidates after the verified Hailo/Hailo candidate: '
            + ', '.join(f'b{b}' for b in kept[:8])
        )
        return (
            self._insert_boundaries_after_prefix(ranked_candidates, kept, prefix_keep=1),
            self._insert_boundaries_after_prefix(candidate_search_pool, kept, prefix_keep=1),
        )

    def _adjust_benchmark_plan_from_full_model_preflight(self, cfg: BenchmarkGenerationOrchestrationConfig, preflight: Mapping[str, Any], *, log: Callable[[str], None]) -> Tuple[BenchmarkGenerationOrchestrationConfig, bool]:
        if not isinstance(preflight, Mapping) or not bool(preflight.get('checked')):
            return cfg, False

        blocked_by_hw: Dict[str, Set[str]] = {}
        scope_by_hw: Dict[str, str] = {}
        explicit_failures = 0
        for raw_entry in list(preflight.get('results') or []):
            if not isinstance(raw_entry, Mapping):
                continue
            if bool(raw_entry.get('ok')):
                continue
            if not bool(raw_entry.get('unsupported_explicit')):
                continue
            explicit_failures += 1
            hw_arch = str(raw_entry.get('hw_arch') or '').strip()
            if not hw_arch:
                continue
            scope = str(raw_entry.get('unsupported_scope') or '').strip().lower() or 'unknown'
            scope_by_hw[hw_arch] = scope
            blocked_by_hw.setdefault(hw_arch, set()).update(_blocked_hailo_kinds_from_preflight_scope(scope))

        if not blocked_by_hw:
            return cfg, False

        # YOLO26 decoded/full exports often fail the full parser preflight at the
        # end-to-end TopK/GatherElements/ReduceMax output tail.  That should not
        # drop Hailo full/part2 variants before the raw one2one-head fallback has
        # a chance to run.  The old YOLO26L logs show this premature adjustment
        # removed full and stage2-Hailo runs even though safer head endpoints were
        # available.
        try:
            is_yolo26 = bool(self._is_yolo26_orchestration_cfg(cfg))
        except Exception:
            is_yolo26 = False
        if is_yolo26:
            scopes = {str(x.get('unsupported_scope') or '').strip().lower() for x in list(preflight.get('results') or []) if isinstance(x, Mapping) and not bool(x.get('ok'))}
            raw_nodes = self._infer_suite_raw_head_end_nodes(cfg)
            if raw_nodes and scopes and scopes.issubset({'output', 'unknown'}):
                msg = (
                    'YOLO26 Hailo preflight policy: decoded full parser failed near the output tail, '
                    'but one2one raw-head endpoints are available; keeping Hailo full/part2 plan variants so '
                    'the raw-head fallback or Hailo-Part1 -> TensorRT policy can handle them.'
                )
                log(msg)
                if isinstance(preflight, dict):
                    preflight['plan_adjusted'] = False
                    preflight['raw_head_fallback_prevents_plan_adjustment'] = True
                    preflight['raw_head_fallback_end_node_names'] = list(raw_nodes)
                    preflight['policy_note'] = msg
                return cfg, False

        plan_runs_out: List[Dict[str, Any]] = []
        adjusted_runs: List[str] = []
        dropped_runs: List[str] = []

        for run in list(cfg.bench_plan_runs or []):
            if not isinstance(run, Mapping):
                continue
            run_mut = dict(run)
            run_id = str(run.get('id') or run.get('name') or run.get('hw_arch') or run.get('type') or 'run')
            variants = [str(x).strip().lower() for x in list(run.get('variants') or []) if str(x).strip()]
            if not variants:
                plan_runs_out.append(run_mut)
                continue

            kept_hailo_variants: List[str] = []
            kept_non_hailo_variants: List[str] = []
            for variant in variants:
                req = run_variant_hailo_requirements(run, variant)
                if not req:
                    kept_non_hailo_variants.append(variant)
                    continue
                blocked = False
                for hw, kind in req:
                    if kind in (blocked_by_hw.get(str(hw).strip()) or set()):
                        blocked = True
                        break
                if not blocked:
                    kept_hailo_variants.append(variant)

            if kept_hailo_variants:
                new_variants = [v for v in variants if v in set(kept_hailo_variants) or v in set(kept_non_hailo_variants)]
                if new_variants != variants:
                    adjusted_runs.append(f"{run_id}: {variants} -> {new_variants}")
                run_mut['variants'] = list(new_variants)
                plan_runs_out.append(run_mut)
                continue

            # If no Hailo-dependent variants remain, keep the run only when it never
            # depended on Hailo in the first place. Otherwise it is now redundant.
            stage1 = run.get('stage1') if isinstance(run.get('stage1'), Mapping) else {}
            stage2 = run.get('stage2') if isinstance(run.get('stage2'), Mapping) else {}
            stage1_hw = str(stage1.get('hw_arch') or stage1.get('arch') or stage1.get('id') or '').strip() if str(stage1.get('type') or '').strip().lower() == 'hailo' else ''
            stage2_hw = str(stage2.get('hw_arch') or stage2.get('arch') or stage2.get('id') or '').strip() if str(stage2.get('type') or '').strip().lower() == 'hailo' else ''
            is_hailo_run = str(run.get('type') or '').strip().lower() == 'hailo'
            if is_hailo_run or stage1_hw or stage2_hw:
                dropped_runs.append(run_id)
                continue
            plan_runs_out.append(run_mut)

        if plan_runs_out == list(cfg.bench_plan_runs or []):
            return cfg, False

        msgs: List[str] = []
        for hw_arch, kinds in sorted(blocked_by_hw.items()):
            scope = scope_by_hw.get(hw_arch, 'unknown')
            msgs.append(
                f"{hw_arch}: full-model parser preflight failed near the {scope} side; "
                f"blocking Hailo variants {', '.join(sorted(kinds))} where applicable"
            )
        summary_msg = 'Plan-aware Hailo preflight adjustment: ' + '; '.join(msgs)
        log(summary_msg)
        for line in adjusted_runs[:12]:
            log(f"  adjusted run: {line}")
        if len(adjusted_runs) > 12:
            log(f"  ... and {len(adjusted_runs) - 12} more adjusted runs")
        for run_id in dropped_runs[:12]:
            log(f"  dropped run: {run_id}")
        if len(dropped_runs) > 12:
            log(f"  ... and {len(dropped_runs) - 12} more dropped runs")

        cfg.runtime.plan_adjustments.append(summary_msg)
        adjusted_cfg = self._replace_plan_runs(cfg, plan_runs_out)
        if isinstance(preflight, dict):
            preflight['plan_adjusted'] = True
            preflight['adjusted_runs'] = list(adjusted_runs)
            preflight['dropped_runs'] = list(dropped_runs)
            preflight['blocked_by_hw'] = {str(k): sorted(str(x) for x in v) for k, v in blocked_by_hw.items()}
        return adjusted_cfg, True

    def _downgrade_benchmark_plan_without_hailo_part2(self, cfg: BenchmarkGenerationOrchestrationConfig, *, log: Callable[[str], None]) -> BenchmarkGenerationOrchestrationConfig:
        plan_runs_out: List[Dict[str, Any]] = []
        dropped_runs: List[str] = []
        adjusted_runs: List[str] = []

        for run in list(cfg.bench_plan_runs or []):
            if not isinstance(run, Mapping):
                continue
            run_mut = dict(run)
            variants = [str(x).strip().lower() for x in list(run.get('variants') or []) if str(x).strip()]
            st2 = run.get('stage2') if isinstance(run.get('stage2'), Mapping) else {}
            stage2_hailo = str(st2.get('type') or '').strip().lower() == 'hailo'
            run_type = str(run.get('type') or '').strip().lower()
            run_id = str(run.get('id') or run.get('name') or run.get('hw_arch') or run_type or 'run')
            if not variants or not stage2_hailo:
                plan_runs_out.append(run_mut)
                continue

            if run_type == 'hailo':
                new_variants = ['full', 'part1']
            else:
                new_variants = [v for v in variants if v not in {'part2', 'composed'}]
            if not new_variants:
                dropped_runs.append(run_id)
                continue
            if list(new_variants) != list(variants):
                adjusted_runs.append(f"{run_id}: {variants} -> {new_variants}")
            run_mut['variants'] = list(new_variants)
            plan_runs_out.append(run_mut)

        adjustment_msg = (
            'No Hailo Part2-compatible candidate found; auto-downgrading benchmark plan to skip '
            'Hailo stage2 part2/composed variants and keep only full/part1 where possible.'
        )
        log(adjustment_msg)
        for line in adjusted_runs[:10]:
            log(f"  adjusted run: {line}")
        if len(adjusted_runs) > 10:
            log(f"  ... and {len(adjusted_runs) - 10} more adjusted runs")
        for run_id in dropped_runs[:10]:
            log(f"  dropped run: {run_id}")
        if len(dropped_runs) > 10:
            log(f"  ... and {len(dropped_runs) - 10} more dropped runs")
        cfg.runtime.plan_adjustments.append(adjustment_msg)
        return self._replace_plan_runs(cfg, plan_runs_out)


    def _prepared_full_baseline_targets(self, cfg: BenchmarkGenerationOrchestrationConfig, baseline: Mapping[str, Any]) -> List[str]:
        raw_hw = str(baseline.get('hw_arch') or '').strip()
        targets = [str(x).strip() for x in list(cfg.hef_targets or []) if str(x).strip()]
        if raw_hw:
            if targets and raw_hw not in set(targets):
                return []
            return [raw_hw]
        return targets[:1] if targets else []

    def _copy_optional_prepared_artifact(self, src_value: Any, dst: Path) -> Optional[str]:
        src = Path(str(src_value or '')).expanduser() if str(src_value or '').strip() else None
        if src is None:
            return None
        try:
            if src.is_file():
                dst.parent.mkdir(parents=True, exist_ok=True)
                if str(src.resolve()) != str(dst.resolve()):
                    shutil.copy2(src, dst)
                return dst.name
        except Exception:
            logger.debug('Failed to copy prepared artifact %s to %s', src, dst, exc_info=True)
        return None

    def _materialize_prepared_full_hailo_baseline(
        self,
        cfg: BenchmarkGenerationOrchestrationConfig,
        *,
        log: Callable[[str], None],
        suite_hailo_hefs: Dict[str, Dict[str, Any]],
    ) -> bool:
        """Copy a prepared full-Hailo HEF into the suite and expose it as full.

        Preparation screening may have already compiled a YOLO full deployment with
        raw detection-head endpoints. Reusing that HEF avoids repeatedly trying the
        unsupported decoded YOLO tail during benchmark-set generation.
        """
        baseline = cfg.prepared_full_hailo_baseline if isinstance(cfg.prepared_full_hailo_baseline, Mapping) else {}
        if not bool(baseline.get('ok')):
            return False
        hef_src = Path(str(baseline.get('hef_path') or '')).expanduser()
        if not hef_src.is_file():
            log(f"suite: prepared full-Hailo baseline ignored; HEF missing: {hef_src}", level=logging.WARNING)
            return False
        targets = self._prepared_full_baseline_targets(cfg, baseline)
        if not targets:
            log(
                "suite: prepared full-Hailo baseline ignored; prepared hw_arch "
                f"{baseline.get('hw_arch') or '<unknown>'} does not match requested targets {list(cfg.hef_targets or [])}",
                level=logging.WARNING,
            )
            return False

        used_any = False
        endpoint_mode = str(baseline.get('endpoint_mode') or ('custom_end_nodes' if baseline.get('end_node_names') else 'full')).strip() or 'full'
        end_nodes = [str(x).strip() for x in list(baseline.get('end_node_names') or []) if str(x).strip()]
        output_contract = baseline.get('output_contract') if isinstance(baseline.get('output_contract'), Mapping) else {}
        output_contract = dict(output_contract or {})
        if end_nodes and not output_contract.get('end_node_names'):
            output_contract['end_node_names'] = list(end_nodes)

        for hw_arch in targets:
            self._raise_if_cancelled(cfg, 'copy prepared full Hailo baseline')
            out_full = Path(cfg.out_dir) / 'hailo' / hw_arch / 'full'
            out_full.mkdir(parents=True, exist_ok=True)
            hef_dst = out_full / 'compiled.hef'
            try:
                if str(hef_src.resolve()) != str(hef_dst.resolve()):
                    shutil.copy2(hef_src, hef_dst)
            except Exception as exc:
                log(f"suite: prepared full-Hailo baseline copy failed ({hw_arch}): {type(exc).__name__}: {exc}", level=logging.WARNING)
                continue

            copied_result = self._copy_optional_prepared_artifact(baseline.get('result_json_path'), out_full / 'hailo_hef_build_result.json')
            copied_parsed = self._copy_optional_prepared_artifact(baseline.get('parsed_har_path'), out_full / 'parsed.har')
            copied_quant = self._copy_optional_prepared_artifact(baseline.get('quant_har_path'), out_full / 'quantized.har')
            copied_fixed = self._copy_optional_prepared_artifact(baseline.get('fixed_onnx_path'), out_full / f"{cfg.base}_hailo_fixed.onnx")

            result_json = baseline.get('result_json') if isinstance(baseline.get('result_json'), Mapping) else {}
            context_count = _safe_int(result_json.get('context_count'))
            detected = result_json.get('detected') if isinstance(result_json.get('detected'), Mapping) else {}
            process = result_json.get('process') if isinstance(result_json.get('process'), Mapping) else {}
            full_build = {
                'ok': True,
                'skipped': False,
                'error': None,
                'elapsed_s': baseline.get('elapsed_s') or result_json.get('elapsed_s'),
                'context_count': context_count,
                'context_mode': ('multi_context_used' if (context_count and context_count > 1) else ('single_context_used' if context_count == 1 else 'ok_unknown_context')),
                'partition_iterations': _safe_int(process.get('partition_iterations')),
                'partition_time_s': process.get('partition_time_s'),
                'allocation_time_s': process.get('allocation_time_s'),
                'compilation_time_s': process.get('compilation_time_s'),
                'single_context_failed': bool(detected.get('single_context_failed')),
                'multi_context_used': bool(detected.get('multi_context_used')),
                'calib_source': ((baseline.get('calib_info') or {}).get('source') if isinstance(baseline.get('calib_info'), Mapping) else None),
                'source': 'prepared_model_screening',
            }
            full_build = {k: v for k, v in full_build.items() if v not in (None, '') or isinstance(v, bool)}

            tgt_suite = suite_hailo_hefs.setdefault(str(hw_arch), {})
            tgt_suite['full'] = os.path.relpath(hef_dst, str(cfg.out_dir)).replace('\\', '/')
            tgt_suite['full_build'] = full_build
            tgt_suite['full_prepared_baseline'] = True
            tgt_suite['full_endpoint_mode'] = endpoint_mode
            tgt_suite['full_end_node_names'] = list(end_nodes)
            tgt_suite['full_true_full_model'] = bool(baseline.get('true_full_model', not bool(end_nodes)))
            tgt_suite['full_output_contract'] = dict(output_contract)
            if copied_result:
                tgt_suite['full_result_json'] = os.path.relpath(out_full / copied_result, str(cfg.out_dir)).replace('\\', '/')
            if copied_parsed:
                tgt_suite['full_parsed_har'] = os.path.relpath(out_full / copied_parsed, str(cfg.out_dir)).replace('\\', '/')
            if copied_quant:
                tgt_suite['full_quant_har'] = os.path.relpath(out_full / copied_quant, str(cfg.out_dir)).replace('\\', '/')
            if copied_fixed:
                tgt_suite['full_fixed_onnx'] = os.path.relpath(out_full / copied_fixed, str(cfg.out_dir)).replace('\\', '/')
            tgt_suite['full_preparation'] = {
                'variant_id': baseline.get('variant_id'),
                'variant_label': baseline.get('variant_label'),
                'screening_dir': baseline.get('screening_dir'),
                'source_hef': str(hef_src),
            }
            used_any = True
            log(
                f"suite: using prepared HEF(full,{hw_arch}) baseline "
                f"mode={endpoint_mode} contract={output_contract.get('mode') or 'decoded_full_model'}"
            )
        return used_any


    def _infer_suite_raw_head_end_nodes(self, cfg: BenchmarkGenerationOrchestrationConfig) -> List[str]:
        """Infer YOLO raw detection-head Conv endpoints from the selected full ONNX."""
        try:
            from .model_preparation import infer_yolo_raw_detection_head_end_nodes
        except Exception:
            logger.debug("Could not import YOLO raw-head endpoint inference helper", exc_info=True)
            return []

        candidates: List[str] = []
        for raw in (cfg.full_model_dst, cfg.full_model_src):
            p = str(raw or '').strip()
            if p and p not in candidates:
                candidates.append(p)
        for model_path in candidates:
            try:
                p = Path(model_path).expanduser()
                if p.is_file():
                    nodes = infer_yolo_raw_detection_head_end_nodes(p, '')
                    if nodes:
                        return [str(x).strip() for x in list(nodes) if str(x).strip()]
            except Exception:
                logger.debug("Failed to infer YOLO raw-head endpoints for %s", model_path, exc_info=True)
        return []


    def _build_suite_full_hefs(self, cfg: BenchmarkGenerationOrchestrationConfig, *, log: Callable[[str], None], queue_put: Callable[[tuple], None], errors: List[str], suite_hailo_hefs: Dict[str, Dict[str, Any]], publish_hailo_diagnostics: Callable[[str, Any, Any], None]) -> None:
        if cfg.hailo_build_hef_fn is None or not cfg.hef_targets or not cfg.hef_full:
            return
        raw_suite_targets = [str(x).strip() for x in list(cfg.hef_targets or []) if str(x).strip()]
        physical_suite_targets = _physical_hailo_targets_for_build(raw_suite_targets)
        log(
            f"suite: Hailo HEF generation requested (backend={cfg.hef_backend}, targets={raw_suite_targets}, physical_targets={physical_suite_targets}, full={cfg.hef_full}, part1={cfg.hef_part1}, part2={cfg.hef_part2})"
        )
        if raw_suite_targets and physical_suite_targets != raw_suite_targets:
            log(
                f"suite: normalizing Hailo build targets {raw_suite_targets} -> physical DFC archs {physical_suite_targets}; "
                "mixed Hailo<->TRT run ids are benchmark labels, not DFC hw_arch values."
            )
        for hw_arch in physical_suite_targets:
            self._raise_if_cancelled(cfg, "suite full HEF build")
            hw_arch = str(hw_arch).strip()
            if not hw_arch:
                continue
            existing_meta = suite_hailo_hefs.get(hw_arch) if isinstance(suite_hailo_hefs, dict) else None
            if isinstance(existing_meta, Mapping) and existing_meta.get('full') and not existing_meta.get('full_error'):
                if bool(existing_meta.get('full_prepared_baseline')):
                    log(f"suite: HEF(full,{hw_arch}) already provided by prepared baseline; skipping decoded full build")
                else:
                    log(f"suite: HEF(full,{hw_arch}) already available; skipping duplicate full build")
                continue

            def _on_suite_hef_log(stream: str, line: str, _hw: str = hw_arch) -> None:
                msg = f"(suite {_hw}) {line}"
                log(msg)
                try:
                    queue_put(("hef", stream, msg))
                except Exception:
                    pass

            full_end_nodes = [str(x).strip() for x in (cfg.hailo_full_end_node_names or []) if str(x).strip()]
            endpoint_mode = str(cfg.hailo_full_endpoint_mode or ('custom_end_nodes' if full_end_nodes else 'full')).strip() or 'full'
            cfg_full_contract = getattr(cfg, 'hailo_full_output_contract', None)
            output_contract: Dict[str, Any] = (
                dict(cfg_full_contract or {})
                if isinstance(cfg_full_contract, Mapping)
                else {}
            )
            out_full = os.path.join(str(cfg.out_dir), "hailo", hw_arch, "full")
            os.makedirs(out_full, exist_ok=True)

            def _build_with_end_nodes(nodes: Sequence[str], mode: str, *, force: Optional[bool] = None) -> Any:
                if nodes:
                    log(
                        f"suite: build HEF(full,{hw_arch}) with endpoint override "
                        f"mode={mode} end_nodes={list(nodes)}"
                    )
                else:
                    log(f"suite: build HEF(full,{hw_arch})")
                return cfg.hailo_build_hef_fn(
                    cfg.full_model_src,
                    backend=cfg.hef_backend,
                    hw_arch=hw_arch,
                    net_name=f"{cfg.base}_full",
                    outdir=out_full,
                    fixup=cfg.hef_fixup,
                    opt_level=int(cfg.hef_opt_level),
                    calib_dir=cfg.hef_calib_dir,
                    calib_count=int(cfg.hef_calib_count),
                    calib_batch_size=int(cfg.hef_calib_bs),
                    force=bool(cfg.hef_force if force is None else force),
                    keep_artifacts=cfg.hef_keep,
                    wsl_distro=cfg.hef_wsl_distro,
                    wsl_venv_activate=cfg.hef_wsl_venv,
                    wsl_timeout_s=int(cfg.hef_timeout_s),
                    on_log=_on_suite_hef_log,
                    end_node_names=(list(nodes) if nodes else None),
                )

            yolo26_raw_head_full_first = False
            if not full_end_nodes:
                try:
                    _is_yolo26_for_full = bool(self._is_yolo26_orchestration_cfg(cfg))
                except Exception:
                    _is_yolo26_for_full = False
                if _is_yolo26_for_full:
                    raw_nodes_first = self._infer_suite_raw_head_end_nodes(cfg)
                    if raw_nodes_first and any('one2one_cv' in str(x) for x in raw_nodes_first):
                        endpoint_mode = 'raw_detection_head'
                        full_end_nodes = list(raw_nodes_first)
                        yolo26_raw_head_full_first = True
                        output_contract = {
                            'mode': 'yolo26_one2one_raw_head',
                            'requires_external_postprocess': True,
                            'description': 'YOLO26 one2one raw detection-head tensors; final transpose/concat/decode/postprocess remains outside the HEF.',
                            'end_node_names': list(full_end_nodes),
                        }
                        log(
                            'suite: YOLO26 policy: building one2one raw-head Hailo Full baseline first; '
                            'decoded output tail stays on host to avoid known Gather/TopK/Transpose/Concat tail failures.'
                        )

            r_full = _build_with_end_nodes(full_end_nodes, endpoint_mode)

            # YOLO11/YOLO10 decoded full graphs often fail at the DFL/decode tail.
            # For the final benchmark workflow, do the raw-head retry here as well,
            # so the manual model-preparation screen is no longer a required step.
            # YOLO26 is handled above by trying the one2one raw-head full baseline
            # first; that avoids spending the first generation stage on the known
            # decoded end-to-end tail failure.
            if (not bool(getattr(r_full, 'ok', False))) and (not full_end_nodes) and not yolo26_raw_head_full_first:
                raw_nodes = self._infer_suite_raw_head_end_nodes(cfg)
                if raw_nodes:
                    endpoint_mode = 'raw_detection_head'
                    full_end_nodes = list(raw_nodes)
                    raw_family = 'yolo26_one2one_raw_head' if any('one2one_cv' in str(x) for x in full_end_nodes) else 'raw_detection_head'
                    output_contract = {
                        'mode': raw_family,
                        'requires_external_postprocess': True,
                        'description': (
                            'YOLO26 one2one raw detection-head tensors; final transpose/concat/decode/postprocess remains outside the HEF.'
                            if raw_family == 'yolo26_one2one_raw_head'
                            else 'YOLO raw cv2/cv3 detection-head tensors; decode/NMS remains outside the HEF.'
                        ),
                        'end_node_names': list(full_end_nodes),
                    }
                    log(
                        f"suite: decoded full HEF failed on {hw_arch}; retrying with YOLO raw detection-head endpoints "
                        f"{full_end_nodes}"
                    )
                    r_raw = _build_with_end_nodes(full_end_nodes, endpoint_mode, force=True)
                    if bool(getattr(r_raw, 'ok', False)):
                        r_full = r_raw
                        log(f"suite: HEF(full,{hw_arch}) raw detection-head fallback OK")
                    else:
                        log(f"suite: HEF(full,{hw_arch}) raw detection-head fallback FAILED: {getattr(r_raw, 'error', None)}")
            elif yolo26_raw_head_full_first and bool(getattr(r_full, 'ok', False)):
                log(f"suite: HEF(full,{hw_arch}) YOLO26 one2one raw-head baseline OK")

            tgt_suite = suite_hailo_hefs.setdefault(hw_arch, {})
            if full_end_nodes:
                tgt_suite['full_endpoint_mode'] = endpoint_mode
                tgt_suite['full_end_node_names'] = list(full_end_nodes)
                tgt_suite['full_true_full_model'] = False if endpoint_mode == 'raw_detection_head' else True
                default_contract_mode = (
                    'yolo26_one2one_raw_head'
                    if endpoint_mode == 'raw_detection_head' and any('one2one_cv' in str(x) for x in full_end_nodes)
                    else ('raw_detection_head' if endpoint_mode == 'raw_detection_head' else 'custom_end_nodes')
                )
                tgt_suite['full_output_contract'] = dict(output_contract or {
                    'mode': default_contract_mode,
                    'requires_external_postprocess': bool(endpoint_mode == 'raw_detection_head'),
                    'description': (
                        'YOLO26 one2one raw detection-head tensors; final transpose/concat/decode/postprocess remains outside the HEF.'
                        if default_contract_mode == 'yolo26_one2one_raw_head'
                        else (
                            'YOLO raw cv2/cv3 detection-head tensors; decode/NMS remains outside the HEF.'
                            if endpoint_mode == 'raw_detection_head'
                            else 'Custom full-model Hailo endpoint set.'
                        )
                    ),
                    'end_node_names': list(full_end_nodes),
                })
            tgt_suite['full_build'] = self.generation_service.compact_hailo_build_summary(r_full)
            if getattr(r_full, 'ok', False):
                rel = os.path.relpath(getattr(r_full, 'hef_path', None) or os.path.join(out_full, 'compiled.hef'), str(cfg.out_dir))
                tgt_suite['full'] = rel.replace('\\', '/')
                log(f"suite: HEF(full,{hw_arch}) OK")
            else:
                tgt_suite['full_error'] = getattr(r_full, 'error', None)
                err_line = f"suite: HEF(full,{hw_arch}) {'SKIPPED' if bool(getattr(r_full, 'skipped', False)) else 'FAILED'}: {getattr(r_full, 'error', None)}"
                errors.append(err_line)
                log(err_line)
            try:
                publish_hailo_diagnostics(f"suite full @ {hw_arch}", r_full, log)
            except Exception:
                logger.debug("Could not publish suite Hailo diagnostics", exc_info=True)

    def _run_hailo_full_model_parse_preflight(
        self,
        cfg: BenchmarkGenerationOrchestrationConfig,
        *,
        log: Callable[[str], None],
        errors: List[str],
    ) -> Optional[Dict[str, Any]]:
        """Run a cheap full-model Hailo parser preflight before split generation.

        The result is diagnostic-first: it can still be used to auto-adjust the
        benchmark plan, instead of unconditionally aborting generation. This keeps
        mixed TRT/ORT->Hailo plans viable when unsupported ops are clearly located
        near the model input or output side.
        """

        if cfg.hailo_parse_check_fn is None or not cfg.hef_targets or not bool(cfg.hailo_selected):
            return None

        model_path = str(
            cfg.full_model_dst
            if str(cfg.full_model_dst or '').strip() and os.path.exists(str(cfg.full_model_dst))
            else cfg.full_model_src
        )
        backend = str(cfg.hef_backend or 'auto')
        timeout_s = int(cfg.hef_timeout_s or 0)
        if timeout_s <= 0:
            timeout_s = 180

        full_end_nodes = [str(x).strip() for x in (cfg.hailo_full_end_node_names or []) if str(x).strip()]
        raw_preflight_targets = [str(x).strip() for x in list(cfg.hef_targets or []) if str(x).strip()]
        physical_preflight_targets = _physical_hailo_targets_for_build(raw_preflight_targets)
        if full_end_nodes:
            log(
                f"suite: Hailo full-model parser preflight requested (backend={backend}, targets={raw_preflight_targets}, "
                f"physical_targets={physical_preflight_targets}, endpoint_override={cfg.hailo_full_endpoint_mode or 'custom'}, end_nodes={full_end_nodes})"
            )
        else:
            log(
                f"suite: Hailo full-model parser preflight requested (backend={backend}, targets={raw_preflight_targets}, "
                f"physical_targets={physical_preflight_targets})"
            )
        if raw_preflight_targets and physical_preflight_targets != raw_preflight_targets:
            log(
                f"suite: normalizing Hailo parser-preflight targets {raw_preflight_targets} -> physical DFC archs {physical_preflight_targets}; "
                "mixed Hailo<->TRT run ids are benchmark labels, not DFC hw_arch values."
            )

        results: List[Dict[str, Any]] = []
        failed_entries: List[Dict[str, Any]] = []
        unsupported_failures: List[Dict[str, Any]] = []

        for raw_hw_arch in physical_preflight_targets:
            self._raise_if_cancelled(cfg, f"full-model Hailo parser preflight {raw_hw_arch}")
            hw_arch = str(raw_hw_arch or '').strip()
            if not hw_arch:
                continue
            outdir = os.path.join(str(cfg.out_dir), 'hailo', hw_arch, 'full_parse_preflight')
            os.makedirs(outdir, exist_ok=True)
            try:
                res = cfg.hailo_parse_check_fn(
                    model_path,
                    backend=backend,
                    hw_arch=hw_arch,
                    net_name=f"{cfg.base}_full_parsecheck",
                    outdir=outdir,
                    fixup=bool(cfg.hef_fixup),
                    add_conv_defaults=True,
                    save_har=False,
                    disable_rt_metadata_extraction=True,
                    wsl_distro=cfg.hef_wsl_distro,
                    wsl_venv_activate=str(cfg.hef_wsl_venv or 'auto'),
                    wsl_timeout_s=int(timeout_s),
                    end_node_names=full_end_nodes or None,
                )
                ok = bool(getattr(res, 'ok', False))
                elapsed_s = float(getattr(res, 'elapsed_s', 0.0) or 0.0)
                error_text = str(getattr(res, 'error', '') or '')
                result_backend = str(getattr(res, 'backend', '') or backend)
                fixed_onnx_path = getattr(res, 'fixed_onnx_path', None)
            except Exception as exc:
                ok = False
                elapsed_s = 0.0
                error_text = f"{type(exc).__name__}: {exc}"
                result_backend = backend
                fixed_onnx_path = None

            unsupported_info = _extract_hailo_unsupported_issue_info(error_text)
            scope_info = _classify_hailo_preflight_failure_scope(model_path, unsupported_info.get('nodes') or [])
            entry = {
                'hw_arch': hw_arch,
                'ok': bool(ok),
                'backend': result_backend,
                'elapsed_s': elapsed_s,
                'error': error_text,
                'fixed_onnx_path': (str(fixed_onnx_path) if fixed_onnx_path not in (None, '') else None),
                'endpoint_mode': (str(cfg.hailo_full_endpoint_mode or 'full') if full_end_nodes else 'full'),
                'end_node_names': list(full_end_nodes),
                'unsupported_explicit': bool(unsupported_info.get('explicit')),
                'unsupported_activation': bool(unsupported_info.get('activation')),
                'unsupported_ops': list(unsupported_info.get('ops') or []),
                'unsupported_nodes': list(unsupported_info.get('nodes') or []),
                'error_classes': list(unsupported_info.get('error_classes') or []),
                'unsupported_scope': str(scope_info.get('scope') or 'unknown'),
                'unsupported_scope_matched_nodes': int(scope_info.get('matched_nodes') or 0),
                'unsupported_scope_total_nodes': _safe_int(scope_info.get('total_nodes')),
            }
            results.append(entry)

            if bool(ok):
                log(f"suite: full-model parser preflight OK ({hw_arch})")
                continue

            failed_entries.append(entry)
            if bool(entry.get('unsupported_explicit')):
                unsupported_failures.append(entry)
                scope_txt = str(entry.get('unsupported_scope') or 'unknown')
                err_line = (
                    f"suite: Hailo full-model parser preflight FAILED ({hw_arch}): "
                    f"{_format_hailo_unsupported_issue_brief(unsupported_info)}"
                    f"; scope={scope_txt}"
                )
            else:
                err_line = (
                    f"suite: Hailo full-model parser preflight FAILED ({hw_arch}): "
                    f"{self.generation_service._compact_issue_detail(error_text or 'parse check failed', limit=220)}"
                )
            errors.append(err_line)
            log(err_line)

        checked = bool(results)
        all_failed_explicit = bool(
            checked
            and failed_entries
            and len(failed_entries) == len(results)
            and len(unsupported_failures) == len(results)
        )

        return {
            'checked': checked,
            'aborted': False,
            'all_failed_explicit': bool(all_failed_explicit),
            'backend': backend,
            'model_path': model_path,
            'endpoint_mode': (str(cfg.hailo_full_endpoint_mode or 'full') if full_end_nodes else 'full'),
            'end_node_names': list(full_end_nodes),
            'result_count': int(len(results)),
            'ok_count': int(sum(1 for entry in results if bool(entry.get('ok')))),
            'failed_count': int(len(failed_entries)),
            'unsupported_failure_count': int(len(unsupported_failures)),
            'results': results,
        }

    def run(self, cfg: BenchmarkGenerationOrchestrationConfig) -> BenchmarkGenerationOrchestrationResult:
        runtime = cfg.runtime
        cases = runtime.cases
        errors = runtime.errors
        discarded_cases = runtime.discarded_cases
        suite_hailo_hefs = runtime.suite_hailo_hefs
        discarded_boundaries = runtime.discarded_boundaries

        log = cfg.execution_callbacks.log
        queue_put = cfg.execution_callbacks.queue_put

        log(f"min_gap: {cfg.execution_cfg.gap}")
        log(f"preferred shortlist size: {len(cfg.ranked_candidates)}")
        log(f"ranked candidates considered: {len(cfg.candidate_search_pool)}")
        try:
            for msg in _model_task_sanity_warnings(cfg.full_model_src, cfg.bench_plan_runs):
                warn = f"suite: classification model sanity warning: {msg}"
                log(warn, level=logging.WARNING)
                if warn not in errors:
                    errors.append(warn)
        except Exception:
            logger.debug('classification model sanity check failed', exc_info=True)

        ranked_candidates = list(cfg.ranked_candidates)
        candidate_search_pool = list(cfg.candidate_search_pool)
        shortlist_prefiltered_boundaries: Set[int] = set()
        cancellation_reason: Optional[str] = None
        hailo_full_model_preflight: Optional[Dict[str, Any]] = None
        prepared_full_baseline_used = False
        early_suite_full_hef_attempted = False

        try:
            cfg = self._ensure_yolo26_full_hailo_baseline_plan(cfg, log=log)
            # Normalize top-level Hailo target state for GUI paths that only
            # populated the nested execution config.
            _eff_targets = self._effective_hailo_targets(cfg)
            _raw_targets_for_log = [str(x).strip() for x in list(cfg.hef_targets or []) if str(x).strip()]
            if _eff_targets and (list(cfg.hef_targets or []) != list(_eff_targets) or not cfg.hailo_selected):
                if _raw_targets_for_log and list(_raw_targets_for_log) != list(_eff_targets):
                    log(
                        f"suite: v47 Hailo target normalization: HEF build targets {list(_raw_targets_for_log)} "
                        f"-> physical DFC targets {list(_eff_targets)}; mixed run labels stay in the benchmark plan."
                    )
                cfg = replace(
                    cfg,
                    hef_targets=list(_eff_targets),
                    hailo_selected=True,
                    execution_cfg=replace(cfg.execution_cfg, hef_targets=list(_eff_targets)),
                )
            prepared_full_baseline_used = self._materialize_prepared_full_hailo_baseline(
                cfg,
                log=log,
                suite_hailo_hefs=suite_hailo_hefs,
            )
            if prepared_full_baseline_used:
                cfg = replace(cfg, hef_full=True)
                baseline = cfg.prepared_full_hailo_baseline if isinstance(cfg.prepared_full_hailo_baseline, Mapping) else {}
                hailo_full_model_preflight = {
                    'checked': False,
                    'skipped': True,
                    'policy': 'prepared_full_hailo_baseline',
                    'aborted': False,
                    'all_failed_explicit': False,
                    'backend': str(cfg.hef_backend or 'auto'),
                    'model_path': str(cfg.full_model_dst or cfg.full_model_src or ''),
                    'endpoint_mode': str(baseline.get('endpoint_mode') or cfg.hailo_full_endpoint_mode or 'full'),
                    'end_node_names': list(baseline.get('end_node_names') or cfg.hailo_full_end_node_names or []),
                    'output_contract': dict(baseline.get('output_contract') or {}) if isinstance(baseline.get('output_contract'), Mapping) else {},
                    'prepared_baseline': True,
                    'result_count': 0,
                    'ok_count': 0,
                    'failed_count': 0,
                    'unsupported_failure_count': 0,
                    'results': [],
                }
            cfg = self._force_yolo26_suite_full_baseline_if_needed(cfg, log=log)

            # v46: YOLO26 Full-Hailo baseline must be attempted before any split
            # prefilter, global Part2 probe, or expensive case HEF builds.  The
            # previous v45 wiring could still enter the case loop first when the
            # UI/run profile carried a full-HEF policy of "skip"/"end" or did
            # not request a pure Hailo run.  For YOLO26 thesis comparisons the
            # Hailo-only baseline is a required reference system, so make the
            # first build stage explicit here.
            try:
                _is_yolo26_forced_full = bool(self._is_yolo26_orchestration_cfg(cfg))
            except Exception:
                _is_yolo26_forced_full = False
            if (
                _is_yolo26_forced_full
                and bool(cfg.hailo_selected)
                and bool(cfg.hef_targets)
                and bool(cfg.hef_full)
                and cfg.hailo_build_hef_fn is not None
                and not prepared_full_baseline_used
            ):
                cfg = replace(cfg, hef_full=True)
                self._raise_if_cancelled(cfg, "before mandatory YOLO26 suite full HEF build")
                log(
                    'suite: YOLO26 policy: FORCE early Hailo Full baseline build as the first build stage '
                    'before shortlist prefilter, Part2 probes, or split HEFs.'
                )
                self._build_suite_full_hefs(
                    cfg,
                    log=log,
                    queue_put=queue_put,
                    errors=errors,
                    suite_hailo_hefs=suite_hailo_hefs,
                    publish_hailo_diagnostics=cfg.execution_callbacks.publish_hailo_diagnostics,
                )
                early_suite_full_hef_attempted = True
                try:
                    cfg.execution_callbacks.persist_state(status='running', current_boundary=None)
                except Exception:
                    logger.debug('persist after mandatory YOLO26 suite full HEF failed', exc_info=True)

            preflight_policy = normalize_hailo_full_model_preflight_policy(cfg.full_model_preflight_policy)
            abort_before_candidate_loop = False
            if prepared_full_baseline_used:
                log('suite: Hailo full-model parser preflight skipped; prepared full-Hailo baseline already supplies the suite full HEF.')
            elif early_suite_full_hef_attempted:
                hailo_full_model_preflight = {
                    'checked': False,
                    'skipped': True,
                    'policy': 'forced_early_full_hef_build',
                    'aborted': False,
                    'all_failed_explicit': False,
                    'backend': str(cfg.hef_backend or 'auto'),
                    'model_path': str(cfg.full_model_dst or cfg.full_model_src or ''),
                    'result_count': 0,
                    'ok_count': 0,
                    'failed_count': 0,
                    'unsupported_failure_count': 0,
                    'results': [],
                }
                log('suite: Hailo full-model parser preflight skipped; YOLO26 Full-Hailo build was already attempted as the first suite step.')
            elif bool(cfg.hef_targets or []) and bool(cfg.hailo_selected) and preflight_policy == 'skip':
                hailo_full_model_preflight = {
                    'checked': False,
                    'skipped': True,
                    'policy': 'skip',
                    'aborted': False,
                    'all_failed_explicit': False,
                    'backend': str(cfg.hef_backend or 'auto'),
                    'model_path': str(cfg.full_model_dst or cfg.full_model_src or ''),
                    'result_count': 0,
                    'ok_count': 0,
                    'failed_count': 0,
                    'unsupported_failure_count': 0,
                    'results': [],
                }
                log('suite: Hailo full-model parser preflight disabled; full-model Hailo HEF builds will be attempted directly.')
            else:
                self._raise_if_cancelled(cfg, "before full-model Hailo parser preflight")
                hailo_full_model_preflight = self._run_hailo_full_model_parse_preflight(
                    cfg,
                    log=log,
                    errors=errors,
                )

                if isinstance(hailo_full_model_preflight, dict) and bool(hailo_full_model_preflight.get('all_failed_explicit')):
                    cfg, adjusted = self._adjust_benchmark_plan_from_full_model_preflight(cfg, hailo_full_model_preflight, log=log)
                    if adjusted:
                        log('suite: plan-aware Hailo preflight adjustment applied; generation will continue with the remaining runnable plan.')
                    elif bool(cfg.bench_plan_runs):
                        log(
                            'suite: full-model Hailo parser preflight failed on all requested targets, '
                            'but the unsupported nodes were not clearly input/output-bound; continuing candidate loop '
                            'because later splits may still isolate a Hailo-compatible subgraph.'
                        )
                    if not bool(cfg.bench_plan_runs):
                        abort_before_candidate_loop = True
                        log('suite: aborting benchmark generation before the candidate loop because no runnable benchmark plan remains after Hailo preflight adjustment.')
                    hailo_full_model_preflight['aborted'] = bool(abort_before_candidate_loop)

            if not abort_before_candidate_loop:
                self._raise_if_cancelled(cfg, "before suite full HEF build")
                suite_full_built_first = False
                if not early_suite_full_hef_attempted:
                    suite_full_built_first = self._build_suite_full_hefs_first_if_needed(
                        cfg,
                    log=log,
                    queue_put=queue_put,
                    errors=errors,
                    suite_hailo_hefs=suite_hailo_hefs,
                )

                self._raise_if_cancelled(cfg, "before shortlist prefilter")
                ranked_candidates, candidate_search_pool, shortlist_prefiltered_boundaries = self._prefilter_shortlist_for_hailo_part2(
                    cfg,
                    discarded_cases=discarded_cases,
                    discarded_boundaries=discarded_boundaries,
                    log=log,
                )
                if shortlist_prefiltered_boundaries:
                    try:
                        cfg.execution_callbacks.persist_state(status='running', current_boundary=None)
                    except Exception:
                        logger.debug('persist after shortlist prefilter failed', exc_info=True)

                self._raise_if_cancelled(cfg, "before Hailo Part2 compatibility probe")
                if bool(cfg.hef_part2) and cfg.hef_targets:
                    any_part2_compatible = self._probe_any_hailo_part2_compatible(cfg, candidate_search_pool, log=log)
                    if not any_part2_compatible:
                        cfg = self._downgrade_benchmark_plan_without_hailo_part2(cfg, log=log)
                        try:
                            cfg.execution_callbacks.persist_state(status='running', current_boundary=None)
                        except Exception:
                            logger.debug('persist after Hailo Part2 plan downgrade failed', exc_info=True)
                    else:
                        ranked_candidates, candidate_search_pool = self._promote_yolo26_true_hailo_part2_candidate(
                            cfg, ranked_candidates, candidate_search_pool, log=log
                        )
                        ranked_candidates, candidate_search_pool = self._promote_yolo26_hetero_throughput_candidates(
                            cfg, ranked_candidates, candidate_search_pool, log=log
                        )

                # Suite Full HEFs are now built before shortlist prefilter/probing.
                # Keep this section intentionally empty; final/end builds below still
                # act as a duplicate-safe backstop via _build_suite_full_hefs.

                self._raise_if_cancelled(cfg, "before case build loop")
                exec_cfg = replace(cfg.execution_cfg, ranked_candidates=list(ranked_candidates), candidate_search_pool=list(candidate_search_pool))
                self.execution_service.execute_case_build_loop(exec_cfg, cfg.execution_callbacks)

                self._raise_if_cancelled(cfg, "before final suite full HEF build")
                if normalize_full_hef_policy(cfg.full_hef_policy) == 'end' and not early_suite_full_hef_attempted:
                    self._build_suite_full_hefs(
                        cfg,
                        log=log,
                        queue_put=queue_put,
                        errors=errors,
                        suite_hailo_hefs=suite_hailo_hefs,
                        publish_hailo_diagnostics=cfg.execution_callbacks.publish_hailo_diagnostics,
                    )
                    try:
                        cfg.execution_callbacks.persist_state(status='running', current_boundary=None)
                    except Exception:
                        logger.debug('persist after suite full HEF (end) failed', exc_info=True)
        except BenchmarkGenerationCancelled as exc:
            cancellation_reason = str(exc)
            log(f"[cancel] {cancellation_reason}")

        finalize_result = self.generation_service.finalize_generation_outputs(
            out_dir=cfg.out_dir,
            base=cfg.base,
            full_model_src=cfg.full_model_src,
            full_model_dst=cfg.full_model_dst,
            analysis_params=cfg.analysis_params_payload,
            system_spec=cfg.system_spec_payload,
            cases=cases,
            errors=errors,
            discarded_cases=discarded_cases,
            requested_cases=int(cfg.target_cases),
            preferred_shortlist_original=cfg.preferred_shortlist_original,
            ranked_candidates=ranked_candidates,
            shortlist_prefiltered_boundaries=sorted(int(x) for x in shortlist_prefiltered_boundaries),
            candidate_search_pool=candidate_search_pool,
            bench_log_path=cfg.bench_log_path,
            analysis_payload=cfg.analysis_payload,
            bench_plan_runs=cfg.bench_plan_runs,
            hef_targets=cfg.hef_targets,
            hef_full=cfg.hef_full,
            hef_part1=cfg.hef_part1,
            hef_part2=cfg.hef_part2,
            hef_backend=cfg.hef_backend,
            hef_wsl_distro=cfg.hef_wsl_distro,
            hef_wsl_venv=cfg.hef_wsl_venv,
            hef_opt_level=int(cfg.hef_opt_level),
            hef_calib_count=int(cfg.hef_calib_count),
            hef_calib_bs=int(cfg.hef_calib_bs),
            hef_calib_dir=cfg.hef_calib_dir,
            hef_fixup=bool(cfg.hef_fixup),
            hef_force=bool(cfg.hef_force),
            hef_keep=bool(cfg.hef_keep),
            suite_hailo_hefs=suite_hailo_hefs,
            hailo_full_model_preflight=hailo_full_model_preflight,
            write_harness_script=cfg.write_harness_script,
            copy_schema_tree=cfg.copy_schema_tree,
            tool_gui_version=cfg.tool_gui_version,
            tool_core_version=cfg.tool_core_version,
            benchmark_objective=str(cfg.benchmark_objective or 'latency'),
            evaluation_profile=cfg.evaluation_profile_meta,
            full_model_preflight_policy=str(cfg.full_model_preflight_policy or 'enabled'),
            hailo_full_end_node_names=list(cfg.hailo_full_end_node_names or []),
            hailo_full_endpoint_mode=str(cfg.hailo_full_endpoint_mode or ''),
        )
        try:
            cfg.execution_callbacks.persist_state(
                status=('partial' if cancellation_reason else ('complete' if int(len(cases)) >= int(cfg.target_cases) else 'partial')),
                current_boundary=None,
            )
        except Exception:
            logger.debug('persist after finalize failed', exc_info=True)

        shortfall = max(0, int(cfg.target_cases) - int(len(cases)))
        benign_reasons = {str(x) for x in list(cfg.benign_discard_reasons or [])}
        benign_discarded = [rec for rec in discarded_cases if str(rec.get('reason') or '') in benign_reasons]
        rejected_discarded = [rec for rec in discarded_cases if str(rec.get('reason') or '') not in benign_reasons]
        shortlist_prefiltered = [rec for rec in benign_discarded if int(rec.get('boundary', -1)) in shortlist_prefiltered_boundaries]
        filtered_shortlist_set = {int(x) for x in ranked_candidates}
        backfilled_cases = [rec for rec in cases if int(rec.get('boundary', -1)) not in filtered_shortlist_set]
        run_ids = []
        try:
            run_ids = [str(r.get('id') or r.get('name') or '').strip() for r in cfg.bench_plan_runs if isinstance(r, dict)]
            run_ids = [x for x in run_ids if x]
        except Exception:
            run_ids = []

        errs = [str(e) for e in errors if str(e).strip()]
        final_status = 'cancelled' if cancellation_reason else ('ok' if (not errs and not rejected_discarded and shortfall == 0) else 'warn')
        accepted_hailo_boundaries: List[int] = []
        try:
            for rec in cases:
                if not isinstance(rec, Mapping):
                    continue
                bval = _safe_int(rec.get('boundary'))
                if bval is None:
                    continue
                av = rec.get('hailo_case_variant_availability')
                if isinstance(av, Mapping) and av:
                    accepted_hailo_boundaries.append(int(bval))
        except Exception:
            accepted_hailo_boundaries = []

        summary_data = self.generation_service.build_completion_summary_data(
            out_dir=str(cfg.out_dir),
            harness_path=str(finalize_result.harness_path),
            bench_log_path=str(cfg.bench_log_path),
            requested_cases=int(cfg.target_cases),
            accepted_count=int(len(cases)),
            preferred_shortlist_count=int(len(cfg.preferred_shortlist_original)),
            candidate_search_pool_count=int(len(candidate_search_pool)),
            benign_discarded=benign_discarded,
            rejected_discarded=rejected_discarded,
            shortlist_prefiltered_count=int(len(shortlist_prefiltered)),
            backfilled_cases_count=int(len(backfilled_cases)),
            resume_generation=bool(cfg.resume_generation),
            resume_lines=list(cfg.resume_report_summary_lines or []),
            plan_run_ids=run_ids,
            errors=errors,
            hailo_selected=bool(cfg.hailo_selected),
            hailo_outlook_summary=cfg.hailo_outlook_summary,
            top_hailo_boundaries=list(accepted_hailo_boundaries or candidate_search_pool[:5]),
            hailo_full_model_preflight=hailo_full_model_preflight,
            full_model_preflight_policy=str(cfg.full_model_preflight_policy or 'enabled'),
        )
        if accepted_hailo_boundaries:
            summary_data['accepted_hailo_boundaries'] = list(accepted_hailo_boundaries)
        summary_data['final_status'] = final_status
        try:
            summary_data['accepted_boundaries'] = [
                int(rec.get('boundary')) for rec in list(cases or [])
                if isinstance(rec, Mapping) and _safe_int(rec.get('boundary')) is not None
            ]
            summary_data['accepted_hailo_feasible_boundaries'] = [
                int(rec.get('boundary')) for rec in list(cases or [])
                if isinstance(rec, Mapping)
                and _safe_int(rec.get('boundary')) is not None
                and isinstance((rec.get('hailo') or {}).get('case_variant_availability'), Mapping)
            ]
        except Exception:
            logger.debug('Could not add accepted boundary summary', exc_info=True)
        if cancellation_reason:
            summary_data['cancellation_reason'] = str(cancellation_reason)
        if runtime.plan_adjustments:
            summary_data['plan_adjustments'] = list(runtime.plan_adjustments)
        if cfg.evaluation_profile_meta:
            summary_data['evaluation_profile'] = dict(cfg.evaluation_profile_meta)
        summary_data['raw_text'] = self.generation_service.format_completion_summary_text(summary_data, verbose=True)
        final_msg = self.generation_service.format_completion_summary_text(summary_data, verbose=False)
        return BenchmarkGenerationOrchestrationResult(
            final_status=final_status,
            final_msg=final_msg,
            bench_payload=dict(finalize_result.bench_payload or {}),
            harness_path=str(finalize_result.harness_path),
            ranked_candidates=list(ranked_candidates),
            candidate_search_pool=list(candidate_search_pool),
            shortlist_prefiltered_boundaries=sorted(int(x) for x in shortlist_prefiltered_boundaries),
            summary_data=dict(summary_data),
        )



class RemoteBenchmarkService:
    """Reusable remote-benchmark orchestration helpers.

    GUI code should only own widget interaction / progress display; SSH host parsing,
    bundle maintenance and the actual remote benchmark call live here.
    """

    def host_configs(self, hosts_payload: Sequence[Mapping[str, Any]]) -> List[SSHHostConfig]:
        out: List[SSHHostConfig] = []
        for h in hosts_payload or []:
            if not isinstance(h, Mapping):
                continue
            try:
                out.append(SSHHostConfig(
                    id=str(h.get('id') or h.get('label') or 'host'),
                    label=str(h.get('label') or h.get('id') or 'host'),
                    user=str(h.get('user') or ''),
                    host=str(h.get('host') or ''),
                    port=int(h.get('port') or 22),
                    remote_base_dir=str(h.get('remote_base_dir') or '~/splitpoint_runs'),
                    ssh_extra_args=str(h.get('ssh_extra_args') or ''),
                ))
            except Exception:
                continue
        return out

    def get_selected_host(self, hosts_payload: Sequence[Mapping[str, Any]], selected_id: str) -> Optional[SSHHostConfig]:
        sid = str(selected_id or '').strip()
        if not sid:
            return None
        for host in self.host_configs(hosts_payload):
            if host.id == sid:
                return host
        return None

    def test_connection(self, host: SSHHostConfig, *, timeout_s: int = 10) -> tuple[bool, str]:
        return SSHTransport(host).test_connection(timeout_s=timeout_s)

    def refresh_suite_harness(
        self,
        suite_dir: Path,
        *,
        benchmark_set_json: Optional[Path] = None,
        validation_images: Optional[str] = None,
        validation_max_images: Optional[int] = None,
        validation_reference_mode: Optional[str] = None,
        mini_coco_ap50: Optional[bool] = None,
        benchmark_task: Optional[str] = None,
        mini_classification_eval: Optional[bool] = None,
        log=None,
    ) -> dict[str, Any]:
        return refresh_suite_harness(
            Path(suite_dir),
            benchmark_set_json=(Path(benchmark_set_json) if benchmark_set_json else None),
            validation_images=validation_images,
            validation_max_images=validation_max_images,
            validation_reference_mode=validation_reference_mode,
            mini_coco_ap50=mini_coco_ap50,
            benchmark_task=benchmark_task,
            mini_classification_eval=mini_classification_eval,
            log=log,
        )

    def rebuild_cached_suite_bundle(self, suite_dir: Path) -> tuple[Path, List[str]]:
        suite_dir = Path(suite_dir)
        dist_dir = suite_dir / 'dist'
        targets = [
            dist_dir / 'suite_bundle.tar.gz',
            dist_dir / 'suite_bundle.tar.gz.manifest.json',
            dist_dir / 'suite_bundle.tar.gz.tmp',
            dist_dir / 'suite_bundle.tar.gz.manifest.json.tmp',
        ]
        removed: List[str] = []
        for target in targets:
            if target.exists():
                target.unlink()
                removed.append(target.name)
        return dist_dir, removed

    def run(self, *, host: SSHHostConfig, benchmark_set_json: Path, local_working_dir: Path, run_id: str, args: RemoteBenchmarkArgs, log, progress, cancel_event=None) -> dict[str, Any]:
        return run_remote_benchmark(
            host=host,
            benchmark_set_json=Path(benchmark_set_json),
            repeats_idx='1',
            local_working_dir=Path(local_working_dir),
            run_id=str(run_id),
            args=args,
            log=log,
            progress=progress,
            cancel_event=cancel_event,
        )


@dataclass
class RemoteBenchmarkCallbacks:
    log: Callable[[str], None]
    progress: Callable[[float, str], None]
    finish: Callable[[str], None]
    result: Callable[[str, Dict[str, Any]], None]


class RemoteBenchmarkController:
    """Async controller for remote benchmark runs.

    Keeps the thread worker out of the Tk UI layer; the GUI only provides the
    callbacks that forward updates onto the main loop.
    """

    def __init__(self, service: Optional[RemoteBenchmarkService] = None):
        self.service = service or RemoteBenchmarkService()

    def _emit_matrix(self, out: Mapping[str, Any], callbacks: RemoteBenchmarkCallbacks) -> None:
        try:
            local_dir = Path(str(out.get("local_run_dir") or ""))
            results_dir = local_dir / "results"
            matrix_md = results_dir / "benchmark_suite_status_matrix.md"
            if matrix_md.exists():
                callbacks.log("")
                callbacks.log("--- Status matrix ---")
                for line in matrix_md.read_text(encoding="utf-8").splitlines():
                    callbacks.log(line)
            v42_md = results_dir / "v42_pipeline_summary.md"
            if v42_md.exists():
                callbacks.log("")
                callbacks.log("--- v42 pipeline summary ---")
                for line in v42_md.read_text(encoding="utf-8").splitlines():
                    callbacks.log(line)
        except Exception as exc:
            callbacks.log(f"[warn] Could not read matrix/pipeline summary: {exc}")

    def run(self, *, host: SSHHostConfig, benchmark_set_json: Path, local_working_dir: Path, run_id: str, args: RemoteBenchmarkArgs, cancel_event=None, callbacks: RemoteBenchmarkCallbacks) -> Dict[str, Any]:
        try:
            out = self.service.run(
                host=host,
                benchmark_set_json=Path(benchmark_set_json),
                local_working_dir=Path(local_working_dir),
                run_id=run_id,
                args=args,
                log=callbacks.log,
                progress=callbacks.progress,
                cancel_event=cancel_event,
            )
            status = str(out.get("status") or ("ok" if out.get("ok") else "failed")).strip().lower()
            if status == "ok":
                callbacks.log(f"[ui] DONE: results saved to {out.get('local_run_dir')}")
                callbacks.finish("Done")
                final_kind = "ok"
            elif status == "partial":
                callbacks.log(f"[ui] PARTIAL: {out.get('error')}")
                callbacks.log(f"[ui] Partial results saved to {out.get('local_run_dir')}")
                active_run = out.get("active_run_id")
                requested_run = out.get("requested_run_id")
                if active_run and requested_run and str(active_run) != str(requested_run):
                    callbacks.log(f"[ui] Resumed existing remote run: {active_run}")
                callbacks.finish("Partial")
                final_kind = "partial"
            elif status == "cancelled":
                callbacks.log(f"[ui] CANCELLED: {out.get('error')}")
                callbacks.finish("Cancelled")
                final_kind = "cancelled"
            else:
                callbacks.log(f"[ui] FAILED: {out.get('error')}")
                callbacks.finish("Failed")
                final_kind = "failed"
            self._emit_matrix(out, callbacks)
            callbacks.result(final_kind, dict(out))
            return dict(out)
        except Exception as exc:
            callbacks.log(f"[ui] ERROR: {exc}")
            callbacks.finish("Error")
            out = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
            callbacks.result("error", out)
            return out

    def start_async(self, **kwargs: Any) -> threading.Thread:
        thread = threading.Thread(target=lambda: self.run(**kwargs), daemon=True)
        thread.start()
        return thread
