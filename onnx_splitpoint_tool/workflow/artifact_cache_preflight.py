from __future__ import annotations

"""Read-only artifact-cache readiness report for evaluation campaigns.

The preflight deliberately does not invent another artifact identity or hash
contract.  Backend-specific code remains responsible for validating its
existing receipts, ABI contract and source identity.  This module only
normalizes those probe results and renders a compact campaign matrix before
runtime or energy work starts.

The distinction between ``MISS`` and ``UNKNOWN`` is intentional: a remote
cache that could not be queried is never presented as a hit or counted as a
cold build.  Likewise, an artifact that is present but has not passed its
existing backend receipt validator remains ``UNKNOWN``.
"""

import csv
import json
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .artifacts import now_iso


SCHEMA = "onnx-splitpoint/artifact-cache-preflight"
SCHEMA_VERSION = 1

ROLE_HAILO8 = "hailo8_hef"
ROLE_HAILO10 = "hailo10_hef"
ROLE_DEEPX = "deepx"
ROLE_TRT_FULL = "trt_full"
ROLE_TRT_P2 = "trt_p2"
ROLE_TRT_P1 = "trt_p1"

ROLE_ORDER = (
    ROLE_HAILO8,
    ROLE_HAILO10,
    ROLE_DEEPX,
    ROLE_TRT_FULL,
    ROLE_TRT_P2,
    ROLE_TRT_P1,
)

STATUS_HIT = "HIT"
STATUS_MISS = "MISS"
STATUS_UNKNOWN = "UNKNOWN"
STATUS_KNOWN_INFEASIBLE = "KNOWN_INFEASIBLE"
STATUS_NOT_APPLICABLE = "NOT_APPLICABLE"
STATUSES = {
    STATUS_HIT,
    STATUS_MISS,
    STATUS_UNKNOWN,
    STATUS_KNOWN_INFEASIBLE,
    STATUS_NOT_APPLICABLE,
}

EXPECTATION_WARM = "warm"
EXPECTATION_COLD = "cold"
EXPECTATION_UNSPECIFIED = "unspecified"
EXPECTATIONS = {
    EXPECTATION_WARM,
    EXPECTATION_COLD,
    EXPECTATION_UNSPECIFIED,
}


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def _contains_token(value: Any, *needles: str) -> bool:
    """Search nested plan values without interpreting them as identities."""

    if isinstance(value, Mapping):
        return any(_contains_token(item, *needles) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return any(_contains_token(item, *needles) for item in value)
    token = str(value or "").strip().lower().replace("-", "_")
    return any(needle in token for needle in needles)


def _token(value: Any) -> str:
    return str(value or "").strip()


def known_negative_build_evidence(value: Any) -> dict[str, Any]:
    """Accept only an exact reusable negative decision from the backend.

    Human-readable failure text and historical local receipts are deliberately
    insufficient: only the backend performs the identity and index validation.
    """
    if not isinstance(value, Mapping):
        return {}
    candidates = [value.get("build_evidence")]
    for name in ("details", "calib_info", "cache_probe_details"):
        child = value.get(name)
        if isinstance(child, Mapping):
            candidates.append(child.get("build_evidence"))
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        if (candidate.get("status") == "HIT"
                and candidate.get("reusable") is True
                and candidate.get("negative_evidence_hit") is True
                and candidate.get("state") in {"PARSER_UNSUPPORTED", "COMPILE_INFEASIBLE"}):
            return dict(candidate)
    return {}


def _evidence_origin(decision: Mapping[str, Any]) -> str:
    origin = decision.get("evidence_origin") or decision.get("index_path") or ""
    return json.dumps(origin, sort_keys=True, ensure_ascii=False) if isinstance(origin, (Mapping, list)) else _token(origin)


def _canonical_role(value: Any) -> str:
    token = _token(value).lower().replace("-", "_")
    aliases = {
        "h8": ROLE_HAILO8,
        "hailo8": ROLE_HAILO8,
        "hailo8l": ROLE_HAILO8,
        "hailo8r": ROLE_HAILO8,
        "hailo_8": ROLE_HAILO8,
        "hailo8_hef": ROLE_HAILO8,
        "h10": ROLE_HAILO10,
        "hailo10": ROLE_HAILO10,
        "hailo10h": ROLE_HAILO10,
        "hailo10p": ROLE_HAILO10,
        "hailo_10": ROLE_HAILO10,
        "hailo10_hef": ROLE_HAILO10,
        "deepx": ROLE_DEEPX,
        "deepx_m1": ROLE_DEEPX,
        "dxnn": ROLE_DEEPX,
        "trt_full": ROLE_TRT_FULL,
        "tensorrt_full": ROLE_TRT_FULL,
        "full_trt": ROLE_TRT_FULL,
        "trt_p2": ROLE_TRT_P2,
        "trt_part2": ROLE_TRT_P2,
        "tensorrt_p2": ROLE_TRT_P2,
        "tensorrt_part2": ROLE_TRT_P2,
        "trt_p1": ROLE_TRT_P1,
        "trt_part1": ROLE_TRT_P1,
        "tensorrt_p1": ROLE_TRT_P1,
        "tensorrt_part1": ROLE_TRT_P1,
    }
    if token not in aliases:
        raise ValueError(f"unknown_artifact_cache_role:{value}")
    return aliases[token]


def _canonical_status(value: Any) -> str:
    token = _token(value).upper().replace("-", "_")
    aliases = {
        "READY": STATUS_HIT,
        "COMPATIBLE": STATUS_HIT,
        "CACHE_HIT": STATUS_HIT,
        "HIT": STATUS_HIT,
        "CACHE_MISS": STATUS_MISS,
        "MISSING": STATUS_MISS,
        "MISS": STATUS_MISS,
        "UNAVAILABLE": STATUS_UNKNOWN,
        "UNPROBED": STATUS_UNKNOWN,
        "UNKNOWN": STATUS_UNKNOWN,
        "N/A": STATUS_NOT_APPLICABLE,
        "NA": STATUS_NOT_APPLICABLE,
        "NOT_REQUESTED": STATUS_NOT_APPLICABLE,
        "NOT_APPLICABLE": STATUS_NOT_APPLICABLE,
    }
    status = aliases.get(token, token)
    if status not in STATUSES:
        raise ValueError(f"unknown_artifact_cache_status:{value}")
    return status


def _canonical_expectation(value: Any) -> str:
    token = _token(value).lower().replace("-", "_")
    aliases = {
        "": EXPECTATION_UNSPECIFIED,
        "unknown": EXPECTATION_UNSPECIFIED,
        "unspecified": EXPECTATION_UNSPECIFIED,
        "warm": EXPECTATION_WARM,
        "hit": EXPECTATION_WARM,
        "expected_hit": EXPECTATION_WARM,
        "cold": EXPECTATION_COLD,
        "miss": EXPECTATION_COLD,
        "expected_miss": EXPECTATION_COLD,
    }
    expectation = aliases.get(token, token)
    if expectation not in EXPECTATIONS:
        raise ValueError(f"unknown_artifact_cache_expectation:{value}")
    return expectation


@dataclass(frozen=True)
class CacheProbeObservation:
    """One backend-validated artifact requirement.

    ``identity`` is an already existing backend identity such as a Hailo cache
    key or TensorRT engine identity.  This class never calculates a replacement
    digest.
    """

    model_id: str
    role: str
    item_id: str
    status: str
    reason: str
    expectation: str = EXPECTATION_UNSPECIFIED
    artifact_path: str = ""
    receipt_path: str = ""
    identity: str = ""
    source: str = ""
    source_namespace: str = ""
    evidence: Mapping[str, Any] | None = None
    boundary: str = ""
    backend: str = ""
    artifact_stage: str = ""
    evidence_origin: str = ""

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any], *, default_expectation: str = "",
    ) -> "CacheProbeObservation":
        model_id = _token(value.get("model_id") or value.get("model"))
        if not model_id:
            raise ValueError("artifact_cache_probe_model_id_missing")
        role = _canonical_role(value.get("role"))
        status = _canonical_status(value.get("status"))
        item_id = _token(
            value.get("item_id")
            or value.get("case_id")
            or ("full" if role == ROLE_TRT_FULL else "artifact")
        )
        reason = _token(value.get("reason"))
        if status != STATUS_NOT_APPLICABLE and not reason:
            raise ValueError(
                f"artifact_cache_probe_reason_missing:{model_id}:{role}:{item_id}"
            )
        evidence = value.get("evidence")
        if evidence is not None and not isinstance(evidence, Mapping):
            raise ValueError(
                f"artifact_cache_probe_evidence_not_object:{model_id}:{role}:{item_id}"
            )
        boundary_match = re.search(r"(?:^|[/_:])b(\d+)(?:$|[/_:])", item_id)
        boundary = _token(value.get("boundary")) or (
            f"b{int(boundary_match.group(1)):03d}" if boundary_match
            else "full" if role == ROLE_TRT_FULL or item_id.rsplit("/", 1)[-1] == "full"
            else "unknown"
        )
        artifact_stage = _token(value.get("artifact_stage") or value.get("stage"))
        if not artifact_stage:
            artifact_stage = (
                "part1" if role == ROLE_TRT_P1
                else "full" if role == ROLE_TRT_FULL or boundary == "full"
                else "part2" if role == ROLE_TRT_P2 or item_id.endswith(":part2")
                else "unknown" if boundary == "unknown" else "part1"
            )
        return cls(
            model_id=model_id,
            role=role,
            item_id=item_id,
            status=status,
            reason=reason or "not_requested",
            expectation=_canonical_expectation(
                value.get("expectation") or default_expectation
            ),
            artifact_path=_token(
                value.get("artifact_path")
                or value.get("engine")
                or value.get("path")
            ),
            receipt_path=_token(
                value.get("receipt_path") or value.get("receipt")
            ),
            identity=_token(
                value.get("identity")
                or value.get("engine_identity")
                or value.get("cache_key")
            ),
            source=_token(value.get("source")),
            source_namespace=_token(value.get("source_namespace")),
            evidence=dict(evidence or {}),
            boundary=boundary,
            backend=_token(value.get("backend")) or {
                ROLE_HAILO8: "hailo8", ROLE_HAILO10: "hailo10h",
                ROLE_DEEPX: "deepx", ROLE_TRT_FULL: "tensorrt",
                ROLE_TRT_P2: "tensorrt",
                ROLE_TRT_P1: "tensorrt",
            }[role],
            artifact_stage=artifact_stage,
            evidence_origin=_token(value.get("evidence_origin")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "role": self.role,
            "item_id": self.item_id,
            "status": self.status,
            "reason": self.reason,
            "expectation": self.expectation,
            "artifact_path": self.artifact_path,
            "receipt_path": self.receipt_path,
            "identity": self.identity,
            "source": self.source,
            "source_namespace": self.source_namespace,
            "evidence": dict(self.evidence or {}),
            "boundary": self.boundary,
            "backend": self.backend,
            "artifact_stage": self.artifact_stage,
            "evidence_origin": self.evidence_origin,
            "compiler_dispatch_allowed": self.status == STATUS_MISS and not bool((self.evidence or {}).get("artifact_now_ready")),
            "runtime_artifact_available": self.status == STATUS_HIT or bool((self.evidence or {}).get("artifact_now_ready")),
            "build_label": f"{self.backend}_{self.artifact_stage}",
        }


def unknown_probe(
    *, model_id: str, role: str, item_id: str, reason: str,
    expectation: str = EXPECTATION_UNSPECIFIED, source: str = "",
) -> CacheProbeObservation:
    """Create an explicit UNKNOWN result for an unavailable backend probe."""

    return CacheProbeObservation.from_mapping({
        "model_id": model_id,
        "role": role,
        "item_id": item_id,
        "status": STATUS_UNKNOWN,
        "reason": reason,
        "expectation": expectation,
        "source": source,
    })


def resolve_artifact_cache_preflight_policy(
    profile_payload: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Resolve the optional diagnostic/acceptance policy from a profile.

    The preflight is diagnostic by default.  An explicit strict/warm-cache
    campaign blocks both an unexpected confirmed MISS and an UNKNOWN probe:
    an unreachable host cannot prove the requested warm-cache condition.
    """

    profile = dict(profile_payload or {})
    workflow = (
        profile.get("workflow")
        if isinstance(profile.get("workflow"), Mapping)
        else {}
    )
    raw = profile.get("artifact_cache_preflight")
    if not isinstance(raw, Mapping):
        raw = workflow.get("artifact_cache_preflight")
    cfg = dict(raw) if isinstance(raw, Mapping) else {}
    strict = bool(
        cfg.get("block_on_unexpected_cold_builds")
        or cfg.get("strict")
        or cfg.get("require_warm_cache")
    )
    default = cfg.get("default_expectation") or cfg.get("expectation")
    if not default and cfg.get("require_warm_cache"):
        default = EXPECTATION_WARM
    default_expectation = _canonical_expectation(default)

    declarations: list[dict[str, str]] = []
    for expectation, key in (
        (EXPECTATION_COLD, "expected_cold"),
        (EXPECTATION_WARM, "expected_warm"),
    ):
        for raw_row in cfg.get(key) or []:
            if isinstance(raw_row, Mapping):
                model_id = _token(raw_row.get("model_id") or raw_row.get("model") or "*")
                role = _token(raw_row.get("role") or "*")
                item_id = _token(raw_row.get("item_id") or raw_row.get("case_id") or "*")
            else:
                parts = [part.strip() for part in str(raw_row or "").split(":")]
                model_id = parts[0] if parts and parts[0] else "*"
                role = parts[1] if len(parts) > 1 and parts[1] else "*"
                item_id = parts[2] if len(parts) > 2 and parts[2] else "*"
            if role != "*":
                role = _canonical_role(role)
            declarations.append({
                "model_id": model_id,
                "role": role,
                "item_id": item_id,
                "expectation": expectation,
            })
    return {
        "enabled": cfg.get("enabled") is not False,
        "default_expectation": default_expectation,
        "block_on_unexpected_cold_builds": strict,
        "declarations": declarations,
    }


def _expectation_for(
    policy: Mapping[str, Any], *, model_id: str, role: str, item_id: str,
) -> str:
    for row in reversed(list(policy.get("declarations") or [])):
        if not isinstance(row, Mapping):
            continue
        if _token(row.get("model_id") or "*") not in {"*", model_id}:
            continue
        if _token(row.get("role") or "*") not in {"*", role}:
            continue
        declared_item = _token(row.get("item_id") or "*")
        # Remote TRT observations are deliberately setup-scoped (for example
        # ``orin_nx_hailo8_01/b024``).  Keep existing policy declarations
        # ergonomic and compatible: a declaration for ``b024`` applies to
        # every setup's b024 item, while a setup-qualified declaration still
        # selects exactly one host.
        item_matches = (
            declared_item in {"*", item_id}
            or declared_item in {
                item_id.rsplit("/", 1)[-1],
                item_id.rsplit("/", 1)[-1].split(":", 1)[0],
            }
        )
        if not item_matches:
            continue
        return _canonical_expectation(row.get("expectation"))
    return _canonical_expectation(policy.get("default_expectation"))


def _path_from_run(run_root: Path, value: Any) -> Path | None:
    token = _token(value)
    if not token:
        return None
    path = Path(token).expanduser()
    if not path.is_absolute():
        path = run_root / path
    return path


def _hailo_role(backend: Any) -> str | None:
    token = _token(backend).lower().replace("-", "_")
    if token in {"hailo8", "hailo8l", "hailo8r", "hailo_8"}:
        return ROLE_HAILO8
    if token in {"hailo10", "hailo10h", "hailo10p", "hailo_10"}:
        return ROLE_HAILO10
    return None


def _hailo_observations(
    *, run_root: Path, model_id: str, policy: Mapping[str, Any],
) -> tuple[list[CacheProbeObservation], set[str]]:
    plan_path = (
        run_root / "models" / model_id / "benchmark_set"
        / "hailo_artifact_service_plan.json"
    )
    plan = _read_object(plan_path)

    # Exact backend probes are recorded even when no compiler attempt exists.
    # In particular, a new boundary must not remain merely "pending_dfc_build"
    # after a cache-only generation pass has already proved its cache miss.
    suite_dir = resolve_generated_benchmark_suite_dir(
        run_dir=run_root, model_id=model_id,
    )
    probe_requests: dict[tuple[str, str], dict[str, Any]] = {}

    def register_probe_request(path: Path, payload: Mapping[str, Any]) -> None:
        # A service plan may omit an unavailable/rejected HEF although its
        # selected cache-only request was already probed. Admit that request
        # only at its exact live model/case/backend/stage address. The normal
        # deferred/negative validator below still decides its cache status.
        kwargs = payload.get("kwargs")
        source = kwargs if isinstance(kwargs, Mapping) else payload
        role = _hailo_role(source.get("hw_arch"))
        net_name = _token(source.get("net_name"))
        match = re.fullmatch(re.escape(model_id) + r"_(part[12])_b(\d+)", net_name)
        if role is None:
            return
        if match:
            stage = match.group(1)
            case = f"b{int(match.group(2)):03d}"
            item = f"{case}:{stage}"
            expected_parent = suite_dir / case / "hailo"
        elif net_name == f"{model_id}_full":
            stage, case, item = "full", "", "full"
            expected_parent = suite_dir / "hailo"
        else:
            return
        if (
            path.parent.name != stage
            or path.parent.parent.parent != expected_parent
            or _hailo_role(path.parent.parent.name) != role
        ):
            return
        probe_requests[(role, item)] = {
            "backend": "hailo8" if role == ROLE_HAILO8 else "hailo10",
            "case_id": case,
            "stage": stage,
            "hef_path": str(path.parent / "compiled.hef"),
            "status": "cache_probe_request",
        }

    exact_misses: dict[tuple[str, str], tuple[Path, dict[str, Any]]] = {}
    for miss_path in sorted(suite_dir.rglob("hailo_cache_miss.json")):
        miss = _read_object(miss_path)
        register_probe_request(miss_path, miss)
        miss_role = _hailo_role(miss.get("hw_arch"))
        if not miss_role:
            continue
        net_name = _token(miss.get("net_name"))
        match = re.search(r"_(part[12])_b(\d+)$", net_name)
        miss_item = (
            f"b{int(match.group(2)):03d}:{match.group(1)}"
            if match else "full"
        )
        exact_misses[(miss_role, miss_item)] = (miss_path, miss)
    deferred_requests: dict[tuple[str, str], dict[str, Any]] = {}
    for request_path in sorted(suite_dir.rglob("deferred_hailo_build.json")):
        request = _read_object(request_path)
        register_probe_request(request_path, request)
        kwargs = dict(request.get("kwargs") or {})
        role = _hailo_role(kwargs.get("hw_arch"))
        if role is None:
            continue
        match = re.search(r"_(part[12])_b(\d+)$", _token(kwargs.get("net_name")))
        item = f"b{int(match.group(2)):03d}:{match.group(1)}" if match else "full"
        deferred_requests[(role, item)] = request

    negative_probes: dict[tuple[str, str], dict[str, Any]] = {}
    for negative_path in sorted(suite_dir.rglob("hailo_negative_evidence.json")):
        probe = _read_object(negative_path)
        register_probe_request(negative_path, probe)
        role = _hailo_role(probe.get("hw_arch"))
        if role is None:
            continue
        match = re.search(r"_(part[12])_b(\d+)$", _token(probe.get("net_name")))
        item = f"b{int(match.group(2)):03d}:{match.group(1)}" if match else "full"
        negative_probes[(role, item)] = probe

    observations: list[CacheProbeObservation] = []
    applicable: set[str] = set()
    request_rows: list[tuple[str, Mapping[str, Any]]] = []
    for row in plan.get("full_baseline_requests") or []:
        if isinstance(row, Mapping) and row.get("requested") is not False:
            request_rows.append(("full", row))
    for row in plan.get("case_hef_requests") or []:
        if not isinstance(row, Mapping):
            continue
        case_id = _token(row.get("case_id") or "case")
        stage = _token(row.get("stage") or "part1")
        request_rows.append((f"{case_id}:{stage}", row))

    planned_keys = {
        (_hailo_role(row.get("backend") or row.get("hw_arch")), item)
        for item, row in request_rows
    }
    not_requested_full = {
        (_hailo_role(row.get("backend") or row.get("hw_arch")), "full")
        for row in plan.get("full_baseline_requests") or []
        if isinstance(row, Mapping) and row.get("requested") is False
    }
    for key, row in sorted(probe_requests.items()):
        if key not in planned_keys and key not in not_requested_full:
            request_rows.append((key[1], row))

    for item_id, row in request_rows:
        role = _hailo_role(row.get("backend") or row.get("hw_arch"))
        if role is None:
            continue
        applicable.add(role)
        hef_path = _path_from_run(
            run_root, row.get("hef_path") or row.get("artifact_path")
        )
        attempt = row.get("build_attempt")
        if not isinstance(attempt, Mapping):
            attempt = row.get("attempt") if isinstance(row.get("attempt"), Mapping) else {}
        status_token = _token(row.get("status") or attempt.get("status")).lower()
        identity = _token(attempt.get("cache_key") or row.get("cache_key"))
        receipt_path = ""
        evidence: dict[str, Any] = {
            "service_plan": str(plan_path),
            "service_status": status_token,
        }
        current_cold_build = bool(
            (
                attempt.get("ok") is True
                and attempt.get("skipped") is not True
            )
            or status_token in {
                "ready_built", "built", "build_succeeded", "compiled",
            }
        )
        # An immutable invocation can succeed on cache reuse while its
        # historical ``skipped`` projection remains false. The explicit
        # physical dispatch counter is authoritative, including a real start
        # whose legacy skipped/status projection contradicts the counter.
        dispatch_count = attempt.get("compiler_dispatch_count")
        if type(dispatch_count) is int and dispatch_count >= 0:
            current_cold_build = dispatch_count > 0
            evidence["compiler_dispatch_count"] = dispatch_count
        evidence["current_build"] = current_cold_build
        deferred_request = deferred_requests.get((role, item_id), {})
        if deferred_request:
            # A fresh cache-only observation supersedes historical compiler
            # attempts retained in a resumed suite's immutable build journal.
            current_cold_build = False
            evidence["current_build"] = False
            evidence["cache_probe_ok"] = deferred_request.get("cache_probe_ok")
            evidence["deferred_request"] = str(
                Path(str(dict(deferred_request.get("kwargs") or {}).get("outdir") or ""))
                / "deferred_hailo_build.json")
            workspace_contract = dict(deferred_request.get("cache_probe_details") or {}).get("workspace_contract")
            if isinstance(workspace_contract, Mapping):
                evidence["workspace_contract"] = dict(workspace_contract)
            evidence["compiler_cache_only"] = bool(dict(deferred_request.get("kwargs") or {}).get("cache_only"))
        exact_miss_path, exact_miss = exact_misses.get((role, item_id), (None, {}))
        if hef_path is None and deferred_request:
            output = _token(dict(deferred_request.get("kwargs") or {}).get("outdir"))
            if output:
                hef_path = Path(output) / "compiled.hef"
        if hef_path is None and exact_miss_path is not None:
            hef_path = exact_miss_path.parent / "compiled.hef"
        if exact_miss:
            identity = _token(exact_miss.get("cache_key_v3")) or identity
            evidence["exact_cache_probe"] = str(exact_miss_path)
            # Fresh deferred probe provenance takes precedence over a sidecar.
            if not evidence.get("workspace_contract") and not deferred_request:
                contract = exact_miss.get("workspace_contract")
                if isinstance(contract, Mapping):
                    evidence["workspace_contract"] = dict(contract)
            evidence["probe_outcomes"] = dict(exact_miss.get("probe_outcomes") or {})
            evidence["bundle_statuses"] = dict(exact_miss.get("bundle_statuses") or {})
            evidence["artifact_store_candidates"] = exact_miss.get("artifact_store_candidates") or {}

        # The current deferred probe supersedes stale negative side files on
        # resumed suites. A successful receipt must not be hidden by an old
        # negative decision left by a different recipe or endpoint selection.
        negative = known_negative_build_evidence(deferred_request) if deferred_request else (
            known_negative_build_evidence(attempt) or known_negative_build_evidence(row)
            or known_negative_build_evidence(negative_probes.get((role, item_id), {}))
        )
        if negative:
            observed_status = STATUS_KNOWN_INFEASIBLE
            reason = _token(negative.get("reason")) or _token(negative.get("state"))
            identity = _token(negative.get("cache_key_v3") or negative.get("key")) or identity
            evidence["build_evidence"] = negative
            evidence["negative_evidence_hit"] = True
            evidence["current_build"] = False
        elif (deferred_request.get("status") != "completed"
              and deferred_request.get("cache_probe_status") in {STATUS_MISS, STATUS_UNKNOWN}):
            observed_status = deferred_request["cache_probe_status"]
            reason = _token(deferred_request.get("cache_probe_reason")) or "hailo_cache_probe_not_completed"
        elif hef_path is not None and hef_path.is_file():
            # Reuse the Hailo backend's existing receipt verifier.  It checks
            # the already sealed cache key/source/HEF contract; this report
            # deliberately creates no replacement identity.
            try:
                from ..hailo_backend import (
                    _hailo_receipt_path,
                    _load_valid_hailo_receipt,
                )

                receipt = _load_valid_hailo_receipt(hef_path)
                rp = _hailo_receipt_path(hef_path)
                receipt_path = str(rp)
            except Exception as exc:
                receipt = None
                evidence["validator_error"] = f"{type(exc).__name__}: {exc}"
            observed_arch = _token((receipt or {}).get("hw_arch")).lower()
            arch_ok = observed_arch in (
                {"hailo8", "hailo8l", "hailo8r", "hailo_8"}
                if role == ROLE_HAILO8 else
                {"hailo10", "hailo10h", "hailo10p", "hailo_10"}
            )
            if receipt is not None and arch_ok:
                identity = _token(receipt.get("cache_key")) or identity
                if current_cold_build:
                    # Cache readiness is now true, but the preflight matrix is
                    # also the campaign's cold-build ledger.  Do not let a
                    # build completed during preparation pass a strict warm
                    # policy as if it had been reused.
                    observed_status = STATUS_MISS
                    reason = "built_during_preparation_artifact_now_ready"
                    evidence["artifact_now_ready"] = True
                else:
                    observed_status = STATUS_HIT
                    reason = "existing_hailo_receipt_valid"
            else:
                observed_status = STATUS_MISS
                reason = (
                    "hailo_receipt_hw_arch_mismatch"
                    if receipt is not None else "hailo_receipt_invalid_or_missing"
                )
                if receipt is None:
                    from ..hailo_backend import _hailo_receipt_path
                    if not _hailo_receipt_path(hef_path).is_file():
                        reason = "legacy_unsealed"
                        evidence["cache_classification"] = "legacy_unsealed"
        elif exact_miss:
            outcomes = dict(exact_miss.get("probe_outcomes") or {})
            lookup_error = _token(outcomes.get("cache_lookup_error"))
            observed_status = STATUS_UNKNOWN if lookup_error else STATUS_MISS
            reason = "cache_lookup_error" if lookup_error else (
                _token(exact_miss.get("reason")) or "exact_cache_artifact_missing"
            )
            classifications = [
                _token(outcomes.get(key))
                for key in ("exact_v3_status", "legacy_v2_status", "exact_cache_status", "legacy_cache_status")
            ]
            if not lookup_error and (
                reason == "legacy_unsealed"
                or
                "legacy_unsealed" in classifications
                or any(
                    outcomes.get(f"{prefix}_hef_present") is True
                    and outcomes.get(f"{prefix}_receipt_present") is False
                    for prefix in ("exact_v3", "legacy_v2")
                )
            ):
                reason = "legacy_unsealed"
                evidence["cache_classification"] = "legacy_unsealed"
        elif status_token in {
            "deferred_cache_miss", "cache_miss", "cache_miss_blocked",
            "source_artifact_missing_or_not_copied", "not_found",
        }:
            observed_status = STATUS_MISS
            reason = status_token
        else:
            observed_status = STATUS_UNKNOWN
            reason = status_token or "hailo_cache_probe_not_completed"

        if (
            observed_status not in {STATUS_KNOWN_INFEASIBLE, STATUS_UNKNOWN}
            and deferred_request.get("status") != "completed"
            and dict(deferred_request.get("kwargs") or {}).get("force") is True
        ):
            observed_status = STATUS_MISS
            reason = "force_build_requested"

        observations.append(CacheProbeObservation.from_mapping({
            "model_id": model_id,
            "role": role,
            "item_id": item_id,
            "status": observed_status,
            "reason": reason,
            "expectation": _expectation_for(
                policy, model_id=model_id, role=role, item_id=item_id,
            ),
            "artifact_path": str(hef_path) if hef_path is not None else "",
            "receipt_path": receipt_path,
            "identity": identity,
            "source": "hailo_artifact_service_plan",
            "evidence_origin": _evidence_origin(negative),
            "backend": "hailo10h" if role == ROLE_HAILO10 else "hailo8",
            "evidence": evidence,
        }))
    return observations, applicable


def _deepx_observation_from_status(
    *, status_path: Path, run_root: Path, model_id: str, item_id: str,
    policy: Mapping[str, Any],
) -> CacheProbeObservation:
    payload = _read_object(status_path)
    lookup = payload.get("cache_lookup")
    lookup = dict(lookup) if isinstance(lookup, Mapping) else {}
    outcome = _token(lookup.get("outcome")).upper()
    lookup_reason = _token(lookup.get("reason"))
    build_status = _token(payload.get("build_status") or payload.get("status")).lower()
    if lookup_reason.lower().endswith("identity_unresolved"):
        # A lookup without a resolvable backend identity did not establish
        # artifact absence.  Keep that epistemic distinction visible.
        status = STATUS_UNKNOWN
        reason = lookup_reason
    elif outcome in {STATUS_HIT, STATUS_MISS}:
        status = outcome
        reason = lookup_reason or "deepx_cache_probe"
    elif build_status == "ready_reused":
        status = STATUS_HIT
        reason = "existing_deepx_contract_valid"
    elif build_status == "ready_built":
        status = STATUS_MISS
        reason = "built_after_cache_miss"
    else:
        status = STATUS_UNKNOWN
        reason = build_status or "deepx_cache_probe_not_completed"
    return CacheProbeObservation.from_mapping({
        "model_id": model_id,
        "role": ROLE_DEEPX,
        "item_id": item_id,
        "status": status,
        "reason": reason,
        "expectation": _expectation_for(
            policy, model_id=model_id, role=ROLE_DEEPX, item_id=item_id,
        ),
        "artifact_path": lookup.get("artifact") or payload.get("dxnn_path") or "",
        "receipt_path": payload.get("cache_receipt") or "",
        "identity": lookup.get("identity") or payload.get("cache_key") or "",
        "source": "deepx_artifact_status",
        "evidence": {
            "status_path": str(status_path),
            "current_build": build_status == "ready_built",
            "historical_cache_lookup_outcome": outcome,
            "historical_cache_lookup_reason": lookup_reason,
        },
    })


def _deepx_observations(
    *, run_root: Path, model_id: str, policy: Mapping[str, Any],
) -> tuple[list[CacheProbeObservation], set[str]]:
    bdir = run_root / "models" / model_id / "benchmark_set"
    full_candidates = (
        bdir / "deepx" / "deepx_artifact_status.json",
        bdir / "deepx_artifact_status.json",
    )
    observations: list[CacheProbeObservation] = []
    applicable: set[str] = set()
    for path in full_candidates:
        if not path.is_file():
            continue
        payload = _read_object(path)
        if payload.get("selected") is False or _token(payload.get("status")) == "not_selected":
            break
        applicable.add(ROLE_DEEPX)
        observations.append(_deepx_observation_from_status(
            status_path=path, run_root=run_root, model_id=model_id,
            item_id="full", policy=policy,
        ))
        break
    # Only the selected executable suite may supply current per-case probes.
    # Formal mirrors and prior suite directories can retain stale successes.
    suite_dir = resolve_generated_benchmark_suite_dir(
        run_dir=run_root, model_id=model_id,
    )
    contract = _read_object(suite_dir / "benchmark_set.json")
    accepted = set(_benchmark_case_ids(contract))
    for stage in ("part1", "part2"):
        seen_cases: set[str] = set()
        for path in sorted(suite_dir.rglob(f"deepx_{stage}_artifact_status.json")):
            payload = _read_object(path)
            if payload.get("selected") is False or _token(payload.get("status")) == "not_selected":
                continue
            case_id = next(
                (f"b{int(part[1:]):03d}" for part in reversed(path.relative_to(suite_dir).parts)
                 if part.startswith("b") and part[1:].isdigit()),
                "",
            )
            # The suite-wide aggregate has the same basename; it is not a
            # separate artifact and must never become e.g. "legacy_suite:part1".
            if not case_id or case_id in seen_cases:
                continue
            if "cases" in contract and case_id not in accepted:
                continue
            seen_cases.add(case_id)
            applicable.add(ROLE_DEEPX)
            observations.append(_deepx_observation_from_status(
                status_path=path, run_root=run_root, model_id=model_id,
                item_id=f"{case_id}:{stage}", policy=policy,
            ))
    return observations, applicable


def _benchmark_case_ids(contract: Mapping[str, Any]) -> list[str]:
    case_ids: list[str] = []
    for row in contract.get("cases") or []:
        if isinstance(row, Mapping):
            token = _token(
                row.get("id") or row.get("case_id") or row.get("case")
                or row.get("case_dir") or row.get("folder")
            )
            if not token:
                try:
                    boundary = row.get("boundary")
                    if boundary is None:
                        boundary = row.get("split_index")
                    token = f"b{int(boundary):03d}"
                except (TypeError, ValueError):
                    token = ""
        else:
            token = _token(row)
        if token and token not in case_ids:
            case_ids.append(token)
    return case_ids


def resolve_generated_benchmark_suite_dir(
    *, run_dir: str | Path, model_id: str,
) -> Path:
    """Resolve the generated runtime suite without consulting mutable state."""

    run_root = Path(run_dir).expanduser().resolve()
    bdir = run_root / "models" / model_id / "benchmark_set"
    postcondition = _read_object(bdir / "suite_generation_postcondition.json")
    selected = _token(postcondition.get("selected_suite_dir"))
    candidates: list[Path] = []
    if selected:
        selected_path = Path(selected).expanduser()
        if not selected_path.is_absolute():
            selected_path = run_root / selected_path
        candidates.append(selected_path)
    candidates.extend((bdir / "suite", bdir / "legacy_suite", bdir))
    seen: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve(strict=True)
        except (OSError, RuntimeError):
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        if (
            resolved.is_dir()
            and (resolved / "benchmark_set.json").is_file()
            and (resolved / "benchmark_plan.json").is_file()
        ):
            return resolved
    return bdir


def _trt_observations(
    *, run_root: Path, model_id: str, policy: Mapping[str, Any],
    suite_dir: Path,
    remote_observations: Sequence[Mapping[str, Any]] = (),
) -> tuple[list[CacheProbeObservation], set[str]]:
    try:
        from ..benchmark.remote_run import _trt_preflight_run_requirements

        required = _trt_preflight_run_requirements(suite_dir)
    except Exception:
        required = {
            "full_required": False, "p2_cases": [],
        }
    applicable: set[str] = set()
    if required.get("full_required"):
        applicable.add(ROLE_TRT_FULL)
    case_ids = [str(value) for value in list(required.get("p2_cases") or [])]
    p1_case_ids = [str(value) for value in list(required.get("p1_cases") or [])]
    generic_case_ids = [str(value) for value in list(required.get("generic_p2_cases") or [])]
    if case_ids or generic_case_ids:
        applicable.add(ROLE_TRT_P2)
    if p1_case_ids:
        applicable.add(ROLE_TRT_P1)
    if not applicable:
        return [], set()
    observations: list[CacheProbeObservation] = []
    for raw in remote_observations:
        if not isinstance(raw, Mapping):
            continue
        if _token(raw.get("model_id") or raw.get("model")) != model_id:
            continue
        role = _canonical_role(raw.get("role"))
        item_id = _token(raw.get("item_id"))
        row = dict(raw)
        row["expectation"] = _expectation_for(
            policy, model_id=model_id, role=role, item_id=item_id,
        )
        row.setdefault("source", "remote_trt_cache_preflight")
        observations.append(CacheProbeObservation.from_mapping(row))
    # Never let a partial response (for example Full + Native bridge only)
    # attest a generic split that has different compiler inputs/precision.
    expected = (
        ([(ROLE_TRT_FULL, "full")] if ROLE_TRT_FULL in applicable else [])
        + [(ROLE_TRT_P2, case_id) for case_id in case_ids]
        + [(ROLE_TRT_P1, case_id) for case_id in p1_case_ids]
        + [(ROLE_TRT_P2, f"{case_id}:generic") for case_id in generic_case_ids]
    )
    for role, item_id in expected:
        if any(row.role == role and (
                   row.item_id.rsplit("/", 1)[-1] == item_id
                   or role == ROLE_TRT_FULL and row.item_id.rsplit("/", 1)[-1].startswith("full:"))
               for row in observations):
            continue
        observations.append(unknown_probe(
            model_id=model_id, role=role, item_id=item_id,
            reason=("tensorrt_part1_probe_unavailable" if role == ROLE_TRT_P1
                    else "remote_trt_cache_probe_unavailable"),
            expectation=_expectation_for(
                policy, model_id=model_id, role=role, item_id=item_id,
            ),
            source="artifact_cache_preflight",
        ))
    return observations, applicable


def collect_model_artifact_cache_probes(
    *, run_dir: str | Path, model_id: str, targets: Sequence[str],
    policy: Mapping[str, Any],
    remote_trt_observations: Sequence[Mapping[str, Any]] = (),
) -> tuple[list[CacheProbeObservation], set[str]]:
    """Collect decisions and retain every required final-selection item.

    Missing observations are UNKNOWN: only an actual backend cache probe can
    establish a cold build. Requirements come from the accepted case contract
    and active run directions, never from whichever status files happen to exist.
    """

    run_root = Path(run_dir)
    observations: list[CacheProbeObservation] = []
    applicable: set[str] = set()
    hailo_rows, hailo_roles = _hailo_observations(
        run_root=run_root, model_id=model_id, policy=policy,
    )
    observations.extend(hailo_rows)
    applicable.update(hailo_roles)
    deepx_rows, deepx_roles = _deepx_observations(
        run_root=run_root, model_id=model_id, policy=policy,
    )
    observations.extend(deepx_rows)
    applicable.update(deepx_roles)

    target_tokens = [str(value or "").lower().replace("-", "_") for value in targets]
    if any("hailo8" in value for value in target_tokens):
        applicable.add(ROLE_HAILO8)
    if any("hailo10" in value for value in target_tokens):
        applicable.add(ROLE_HAILO10)
    if any("deepx" in value or "dx_m1" in value for value in target_tokens):
        applicable.add(ROLE_DEEPX)
    suite_dir = resolve_generated_benchmark_suite_dir(
        run_dir=run_root, model_id=model_id,
    )
    contract = _read_object(suite_dir / "benchmark_set.json")
    plan = _read_object(suite_dir / "benchmark_plan.json")
    selected_cases = _benchmark_case_ids(contract)
    selected_set = set(selected_cases)
    from .hailo_artifact_scope import selected_hailo_artifact_stages
    hailo_stage_scope = (
        selected_hailo_artifact_stages(plan, selected_cases)
        if "cases" in contract else None
    )
    scoped_hailo_items = None if hailo_stage_scope is None else {
        (_hailo_role(backend), f"{case_id}:{stage}")
        for backend, case_id, stage in hailo_stage_scope
    }
    if "cases" in contract:
        observations = [
            observation for observation in observations
            if ":part" not in observation.item_id
            or observation.item_id.split(":", 1)[0] in selected_set
            or (observation.evidence or {}).get("current_build") is True
        ]
    if scoped_hailo_items is not None:
        observations = [
            observation for observation in observations
            if observation.role not in {ROLE_HAILO8, ROLE_HAILO10}
            or ":part" not in observation.item_id
            or (observation.role, observation.item_id) in scoped_hailo_items
            # A build that actually happened is still a warm-policy violation,
            # even when its artifact has no consumer in the requested scope.
            or (observation.evidence or {}).get("current_build") is True
        ]

    def local_role(value: Any) -> str | None:
        if isinstance(value, Mapping):
            for key in ("hw_arch", "accelerator", "backend", "provider", "type", "id", "name"):
                role = local_role(value.get(key))
                if role:
                    return role
            return None
        token = _token(value).lower().replace("-", "_")
        if token in {"deepx", "deepx_m1", "dx_m1", "dxm1"}:
            return ROLE_DEEPX
        return _hailo_role(token)

    def canonical_cases(values: Any) -> list[str]:
        if not isinstance(values, (list, tuple)):
            values = [values]
        normalized: list[str] = []
        for value in values:
            if isinstance(value, Mapping):
                value = value.get("id") or value.get("case_id") or value.get("case") or value.get("boundary")
            token = _token(value)
            if token.isdigit():
                token = f"b{int(token):03d}"
            elif token.lower().startswith("b") and token[1:].isdigit():
                token = f"b{int(token[1:]):03d}"
            if token and token not in normalized:
                normalized.append(token)
        return normalized

    required: set[tuple[str, str]] = set()
    service_plan = _read_object(
        run_root / "models" / model_id / "benchmark_set"
        / "hailo_artifact_service_plan.json"
    )
    excluded_full: set[str] = set()
    for row in service_plan.get("full_baseline_requests") or []:
        if not isinstance(row, Mapping):
            continue
        role = local_role(row.get("backend") or row.get("hw_arch"))
        if role:
            if row.get("requested") is False:
                excluded_full.add(role)
            else:
                required.add((role, "full"))
    for row in service_plan.get("case_hef_requests") or []:
        if not isinstance(row, Mapping) or row.get("requested") is False:
            continue
        role = local_role(row.get("backend") or row.get("hw_arch"))
        cases = canonical_cases(row.get("case_id"))
        stage = _token(row.get("stage") or "part1")
        if role and stage in {"part1", "part2"}:
            required.update(
                (role, f"{case_id}:{stage}") for case_id in cases
                if "cases" not in contract or case_id in selected_set
            )
    for row in plan.get("runs") or plan.get("planned_runs") or []:
        if not isinstance(row, Mapping) or row.get("enabled") is False or row.get("deferred"):
            continue
        if _token(row.get("status") or row.get("build_status")).lower() in {"disabled", "deferred", "not_selected"}:
            continue
        run_id = _token(row.get("id") or row.get("run_id")).lower().replace("-", "_")
        run_type = _token(row.get("type")).lower().replace("-", "_")
        variants = {_token(value).lower() for value in row.get("variants") or []}
        full_role = local_role(row.get("full"))
        if not full_role and (run_type not in {"matrix", "split", "mixed_backend"} and "_to_" not in run_id):
            full_role = local_role(row.get("full_provider") or row.get("provider") or row.get("backend") or run_id)
        if full_role and full_role not in excluded_full:
            required.add((full_role, "full"))
        if run_type not in {"matrix", "split", "mixed_backend"} and "_to_" not in run_id:
            continue
        row_cases: list[str] = []
        for key in ("case_id", "case", "cases", "case_ids", "selected_cases"):
            if row.get(key) not in (None, "", [], {}):
                row_cases.extend(canonical_cases(row[key]))
        if not row_cases:
            row_cases = selected_cases
        if "cases" in contract:
            row_cases = [case_id for case_id in row_cases if case_id in selected_set]
        fallback_stages = run_id.split("_to_", 1) if "_to_" in run_id else []
        for index, stage in enumerate(("part1", "part2")):
            if variants and not variants.intersection({stage, "composed", "split"}):
                continue
            role = local_role(row.get(f"stage{index + 1}"))
            if not role and not row.get(f"stage{index + 1}") and fallback_stages:
                role = local_role(fallback_stages[index])
            stage_backend = row.get(f"stage{index + 1}") or (
                fallback_stages[index] if fallback_stages else ""
            )
            if index == 0 and _contains_token(stage_backend, "tensorrt", "trt"):
                role = ROLE_TRT_P1
            if role:
                required.update((role, f"{case_id}:{stage}") for case_id in row_cases)
    if scoped_hailo_items is not None:
        # The active generated plan owns split requirements. Broad historical
        # service requests must not resurrect an unused accelerator Part2.
        required = {
            (role, item_id) for role, item_id in required
            if role not in {ROLE_HAILO8, ROLE_HAILO10} or ":part" not in item_id
        }
        required.update(scoped_hailo_items)
    trt_rows, trt_roles = _trt_observations(
        run_root=run_root, model_id=model_id, policy=policy,
        suite_dir=suite_dir,
        remote_observations=remote_trt_observations,
    )
    observations.extend(trt_rows)
    applicable.update(trt_roles)
    observed_items = {(row.role, row.item_id) for row in observations}
    for role, item_id in sorted(required - observed_items):
        if role == ROLE_TRT_P1 and any(
            row.role == ROLE_TRT_P1 and row.boundary == item_id.split(":", 1)[0]
            for row in observations
        ):
            continue
        applicable.add(role)
        observations.append(unknown_probe(
            model_id=model_id, role=role, item_id=item_id,
            reason=("tensorrt_part1_probe_unavailable" if role == ROLE_TRT_P1
                    else "required_artifact_cache_probe_missing"),
            expectation=_expectation_for(
                policy, model_id=model_id, role=role, item_id=item_id,
            ),
            source="final_selection_artifact_requirements",
        ))
    return observations, applicable


def _cell_status(rows: Sequence[CacheProbeObservation]) -> str:
    applicable = [row for row in rows if row.status != STATUS_NOT_APPLICABLE]
    if not applicable:
        return STATUS_NOT_APPLICABLE
    if any(row.status == STATUS_MISS for row in applicable):
        return STATUS_MISS
    if any(row.status == STATUS_UNKNOWN for row in applicable):
        return STATUS_UNKNOWN
    if any(row.status == STATUS_KNOWN_INFEASIBLE for row in applicable):
        return STATUS_KNOWN_INFEASIBLE
    return STATUS_HIT


def _cell_payload(rows: Sequence[CacheProbeObservation]) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: (row.item_id, row.reason))
    status = _cell_status(ordered)
    counts = {
        candidate.lower(): sum(row.status == candidate for row in ordered)
        for candidate in STATUSES
    }
    reasons = list(dict.fromkeys(row.reason for row in ordered))
    return {
        "status": status,
        "reason": reasons[0] if len(reasons) == 1 else ",".join(reasons),
        "item_count": len(ordered),
        "counts": counts,
        "items": [row.to_dict() for row in ordered],
    }


def build_artifact_cache_preflight(
    *,
    model_ids: Sequence[str],
    observations: Iterable[CacheProbeObservation | Mapping[str, Any]],
    applicable_roles: Mapping[str, Sequence[str]] | None = None,
    default_expectation: str = EXPECTATION_UNSPECIFIED,
    block_on_unexpected_cold_builds: bool = False,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Normalize backend probes into a deterministic per-model matrix.

    Only a confirmed ``MISS`` contributes to a cold-build count.  ``UNKNOWN``
    is kept separate even when a warm cache was expected, because an
    unreachable host or unavailable ABI probe does not prove absence.
    """

    expectation_default = _canonical_expectation(default_expectation)
    models = list(dict.fromkeys(_token(value) for value in model_ids if _token(value)))

    rows: list[CacheProbeObservation] = []
    seen: set[tuple[str, str, str]] = set()
    for raw in observations:
        row = (
            raw
            if isinstance(raw, CacheProbeObservation)
            else CacheProbeObservation.from_mapping(
                raw, default_expectation=expectation_default,
            )
        )
        if row.model_id not in models:
            raise ValueError(
                f"artifact_cache_probe_unknown_model:{row.model_id}"
            )
        key = (row.model_id, row.role, row.item_id)
        if key in seen:
            raise ValueError(
                "artifact_cache_probe_duplicate:"
                + ":".join(key)
            )
        seen.add(key)
        rows.append(row)

    # A Native Part2 recipe binds the bytes of its Part1 producer. Delay
    # only that dependent identity check when the exact producer is an allowed
    # MISS; do not require its own output as a prerequisite for building it.
    # Status remains UNKNOWN and grants no TRT runtime/build permission.
    for index, row in enumerate(rows):
        if row.role != ROLE_TRT_P2 or row.status != STATUS_UNKNOWN or row.reason != "native_part1_identity_unavailable":
            continue
        evidence = dict(row.evidence or {})
        dependency_role = _hailo_role(evidence.get("dependency_backend"))
        if not dependency_role and _token(evidence.get("dependency_backend")).lower() in {"deepx", "dx_m1"}:
            dependency_role = ROLE_DEEPX
        producers = [candidate for candidate in rows
                     if candidate.model_id == row.model_id and candidate.role == dependency_role
                     and candidate.boundary == evidence.get("dependency_boundary")
                     and candidate.artifact_stage == "part1"]
        if len(producers) != 1:
            continue
        producer = producers[0]
        allowed_cold = (producer.status == STATUS_MISS
                        and not (producer.evidence or {}).get("compiler_cache_only")
                        and not (producer.evidence or {}).get("artifact_now_ready")
                        and (not block_on_unexpected_cold_builds or producer.expectation == EXPECTATION_COLD))
        known_exclusion = (producer.status == STATUS_KNOWN_INFEASIBLE
                           and bool(known_negative_build_evidence(producer.evidence or {})))
        if allowed_cold or known_exclusion:
            evidence.update(dependency_resolution=("deferred_until_producer_build" if allowed_cold
                                                   else "upstream_known_infeasible"),
                            producer_item_id=producer.item_id,
                            producer_identity=producer.identity,
                            producer_evidence_origin=producer.evidence_origin,
                            final_identity_check_required=allowed_cold,
                            dependent_runtime_dispatch_allowed=False)
            rows[index] = replace(row, evidence=evidence)

    roles_by_model: dict[str, set[str]] = {}
    for model_id in models:
        raw_roles = (
            applicable_roles.get(model_id, ())
            if isinstance(applicable_roles, Mapping)
            else [role for role in ROLE_ORDER if role != ROLE_TRT_P1]
        )
        roles_by_model[model_id] = {
            _canonical_role(role) for role in raw_roles
        }
        if any(row.model_id == model_id and row.role == ROLE_TRT_P1 for row in rows):
            roles_by_model[model_id].add(ROLE_TRT_P1)

    # Add explicit N/A cells, but never invent HIT/MISS observations.
    grouped: dict[tuple[str, str], list[CacheProbeObservation]] = {}
    for row in rows:
        grouped.setdefault((row.model_id, row.role), []).append(row)

    matrix: list[dict[str, Any]] = []
    synthetic_unknowns: list[CacheProbeObservation] = []
    for model_id in models:
        cells: dict[str, Any] = {}
        for role in ROLE_ORDER:
            role_rows = grouped.get((model_id, role), [])
            if role not in roles_by_model[model_id]:
                cells[role] = _cell_payload([])
            elif not role_rows:
                missing_probe = unknown_probe(
                    model_id=model_id,
                    role=role,
                    item_id="probe",
                    reason="probe_result_missing",
                    expectation=expectation_default,
                    source="artifact_cache_preflight",
                )
                synthetic_unknowns.append(missing_probe)
                cells[role] = _cell_payload([missing_probe])
            else:
                cells[role] = _cell_payload(role_rows)
        matrix.append({"model_id": model_id, "cells": cells})

    confirmed_misses = [row for row in rows if row.status == STATUS_MISS]
    # A preparation build stays in the historical MISS ledger (including
    # strict warm-cache policy), but it is no longer an outstanding build.
    completed_cold = [
        row for row in confirmed_misses
        if bool((row.evidence or {}).get("artifact_now_ready"))
    ]
    pending_cold = [
        row for row in confirmed_misses
        if not bool((row.evidence or {}).get("artifact_now_ready"))
    ]
    expected_cold = [
        row for row in confirmed_misses
        if row.expectation == EXPECTATION_COLD
    ]
    unexpected_cold = [
        row for row in confirmed_misses
        if row.expectation == EXPECTATION_WARM
    ]
    unclassified_cold = [
        row for row in confirmed_misses
        if row.expectation == EXPECTATION_UNSPECIFIED
    ]
    unknown = (
        [row for row in rows if row.status == STATUS_UNKNOWN]
        + synthetic_unknowns
    )
    hits = [row for row in rows if row.status == STATUS_HIT]
    infeasible = [row for row in rows if row.status == STATUS_KNOWN_INFEASIBLE]
    deferred_dependencies = [row for row in unknown if (row.evidence or {}).get("dependency_resolution") in {
        "deferred_until_producer_build", "upstream_known_infeasible"}]
    strict_unknowns = [row for row in unknown if row not in deferred_dependencies]
    strict_probe_blockers = bool(
        block_on_unexpected_cold_builds and (unexpected_cold or strict_unknowns)
    )

    status = (
        "unexpected_cold_builds"
        if unexpected_cold
        else "cold_builds_expected"
        if pending_cold
        else "probe_incomplete"
        if unknown
        else "known_infeasible_artifacts"
        if infeasible
        else "cold_builds_completed"
        if completed_cold
        else "warm"
    )
    artifact_rows = [row.to_dict() for row in sorted(
        rows + synthetic_unknowns,
        key=lambda value: (models.index(value.model_id), value.boundary,
                           ROLE_ORDER.index(value.role), value.item_id),
    )]
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "created_at": created_at or now_iso(),
        "status": status,
        "model_count": len(models),
        "roles": list(ROLE_ORDER),
        "matrix": matrix,
        "artifact_matrix": artifact_rows,
        "cold_build_rows": [row for row in artifact_rows if row["compiler_dispatch_allowed"]],
        "selection_changes": [],
        "selection_phase": "final_selected_cases",
        "observations": [
            row.to_dict()
            for row in sorted(
                rows, key=lambda value: (
                    models.index(value.model_id),
                    ROLE_ORDER.index(value.role),
                    value.item_id,
                ),
            )
        ],
        "hit_count": len(hits),
        "confirmed_miss_count": len(confirmed_misses),
        "unknown_count": len(unknown),
        "known_infeasible_count": len(infeasible),
        "known_infeasible_rows": [row.to_dict() for row in infeasible],
        "strict_unknown_blocker_count": (
            len(strict_unknowns) if block_on_unexpected_cold_builds else 0
        ),
        "deferred_dependency_count": len(deferred_dependencies),
        "deferred_dependency_rows": [row.to_dict() for row in deferred_dependencies],
        "cold_builds_required": len(pending_cold),
        "completed_cold_build_count": len(completed_cold),
        "expected_cold_builds": len(expected_cold),
        "unexpected_cold_builds": len(unexpected_cold),
        "unclassified_cold_builds": len(unclassified_cold),
        "expected_cold_build_rows": [row.to_dict() for row in expected_cold],
        "unexpected_cold_build_rows": [row.to_dict() for row in unexpected_cold],
        "unclassified_cold_build_rows": [row.to_dict() for row in unclassified_cold],
        "unknown_rows": [row.to_dict() for row in unknown],
        "block_on_unexpected_cold_builds": bool(
            block_on_unexpected_cold_builds
        ),
        "runtime_dispatch_allowed": not strict_probe_blockers,
        "note": (
            "Diagnostic aggregation of existing backend cache probes. "
            "No artifact identity or hash is created by this report."
        ),
    }


def collect_selection_changes(
    *, run_dir: str | Path, model_ids: Sequence[str], report: Mapping[str, Any],
    cache_roots: Sequence[str | Path] = (),
) -> list[dict[str, Any]]:
    """Compare selected cases with receipt-validated prior Hailo artifacts.

    The receipt net-name links a cache artifact to its model and boundary;
    neither old run plans nor HEF filenames alone are build evidence.  This
    inventory says which boundaries were built, not that different boundaries
    have an interchangeable source graph or cache identity.
    """
    from ..hailo_backend import _hailo_cache_root, _load_valid_hailo_receipt

    root = Path(run_dir)
    roots = {_hailo_cache_root().expanduser().resolve()}
    roots.update(Path(value).expanduser().resolve() for value in cache_roots if str(value).strip())
    wanted = {_token(mid).lower().replace("-", "_"): mid for mid in model_ids}
    built: dict[str, dict[str, list[str]]] = {mid: {} for mid in model_ids}
    # Public generation pointers only: preserved hidden generations must not
    # pretend that an artifact remains selected by the cache index.
    for hef in sorted({path for cache_root in roots for path in cache_root.glob("*/compiled.hef")}):
        if hef.parent.name.startswith("."):
            continue
        try:
            snapshot = hef.resolve(strict=True)
            receipt = _read_object(snapshot.parent / "hailo_hef_build_receipt.json")
            match = re.fullmatch(r"(.+)_(?:part1|part2)_b(\d+)", _token(receipt.get("net_name")))
            if not match:
                continue
            model_id = wanted.get(match.group(1).lower().replace("-", "_"))
            if model_id is None or _load_valid_hailo_receipt(snapshot) is None:
                continue
            boundary = f"b{int(match.group(2)):03d}"
            built[model_id].setdefault(boundary, []).append(str(hef))
        except (OSError, ValueError, RuntimeError):
            continue

    # Receipt-bearing historical store records need not have been restored to
    # the backend-local cache.  Validate their indexed bytes and embedded
    # receipt without registering/copying anything or updating access times.
    from ..artifact_store import ArtifactStore, default_artifact_store_root
    store_root = default_artifact_store_root()
    if (store_root / "registry.sqlite3").is_file():
        store = ArtifactStore(store_root, read_only=True)
        for record in store.list(kind="hailo_hef", limit=1_000_000):
            receipt = record.metadata.get("build_receipt")
            if not isinstance(receipt, Mapping):
                continue
            match = re.fullmatch(r"(.+)_(?:part1|part2)_b(\d+)", _token(receipt.get("net_name")))
            if not match:
                continue
            model_id = wanted.get(match.group(1).lower().replace("-", "_"))
            if model_id is None:
                continue
            valid, _reason = store.validate_record(record, verify="strict")
            if not valid or _load_valid_hailo_receipt(
                Path(record.object_path), receipt_override=receipt,
                validate_cache_meta=False, allow_legacy_v2=True,
            ) is None:
                continue
            boundary = f"b{int(match.group(2)):03d}"
            paths = built[model_id].setdefault(boundary, [])
            if record.object_path not in paths:
                paths.append(record.object_path)

    changes: list[dict[str, Any]] = []
    for model_id in model_ids:
        suite = resolve_generated_benchmark_suite_dir(run_dir=root, model_id=model_id)
        contract = _read_object(suite / "benchmark_set.json")
        selected = _benchmark_case_ids(contract)
        if not selected:
            selected = sorted({
                _token(row.get("boundary"))
                for row in report.get("artifact_matrix") or []
                if row.get("model_id") == model_id
                and re.fullmatch(r"b\d+", _token(row.get("boundary")))
            })
        previous = sorted(built[model_id])
        for boundary in selected:
            if not previous or boundary in previous:
                continue
            cold = [row for row in report.get("cold_build_rows") or []
                    if row.get("model_id") == model_id and row.get("boundary") == boundary]
            changes.append({
                "event": "selection_changed",
                "model_id": model_id,
                "previously_built_boundaries": previous,
                "currently_selected": boundary,
                "expected_cold_builds": sorted({_token(row.get("build_label")) for row in cold}),
                "cold_build_rows": cold,
                "previous_build_evidence": built[model_id],
                "history_source": "validated_cache_receipts_by_model_net_name",
            })
    return changes


def render_artifact_cache_preflight_log_lines(report: Mapping[str, Any]) -> list[str]:
    """Render every required artifact, including misses and unknown probes."""
    lines = []
    for row in report.get("artifact_matrix") or []:
        status = row.get("status")
        label = ("completed_cold_build" if status == STATUS_MISS and row.get("runtime_artifact_available")
                 else "expected_cold_build" if status == STATUS_MISS
                 else "known_infeasible" if status == STATUS_KNOWN_INFEASIBLE
                 else "cache_probe")
        lines.append(
            f"[artifact-cache-preflight] {label}: model={row.get('model_id')} "
            f"boundary={row.get('boundary')} backend={row.get('backend')} "
            f"artifact={row.get('artifact_stage')} status={status} "
            f"reason={row.get('reason')} item={row.get('item_id')}"
            + (f" evidence_origin={row.get('evidence_origin')} compiler_dispatch_allowed=false"
               if status == STATUS_KNOWN_INFEASIBLE else "")
        )
    for change in report.get("selection_changes") or []:
        lines.append(
            f"[artifact-cache-preflight] model={change.get('model_id')} selection_changed:\n"
            f"previously_built_boundaries={json.dumps(change.get('previously_built_boundaries') or [])}\n"
            f"currently_selected={change.get('currently_selected')}\n"
            f"expected_cold_builds={json.dumps(change.get('expected_cold_builds') or [])}"
        )
    return lines


def _matrix_csv_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in report.get("matrix") or []:
        if not isinstance(model, Mapping):
            continue
        cells = model.get("cells") if isinstance(model.get("cells"), Mapping) else {}
        row: dict[str, Any] = {"model_id": _token(model.get("model_id"))}
        for role in ROLE_ORDER:
            cell = cells.get(role) if isinstance(cells.get(role), Mapping) else {}
            row[role] = _token(cell.get("status") or STATUS_UNKNOWN)
            row[f"{role}_reason"] = _token(cell.get("reason"))
        rows.append(row)
    return rows


def render_artifact_cache_preflight_markdown(report: Mapping[str, Any]) -> str:
    headers = ["Model", "H8 HEF", "H10 HEF", "DeepX", "TRT Full", "TRT P2", "TRT P1"]
    lines = [
        "# Artifact cache preflight",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    labels = {
        ROLE_HAILO8: "H8 HEF",
        ROLE_HAILO10: "H10 HEF",
        ROLE_DEEPX: "DeepX",
        ROLE_TRT_FULL: "TRT Full",
        ROLE_TRT_P2: "TRT P2",
        ROLE_TRT_P1: "TRT P1",
    }
    for row in _matrix_csv_rows(report):
        values = [row["model_id"]]
        for role in ROLE_ORDER:
            value = row[role]
            reason = row.get(f"{role}_reason") or ""
            values.append(f"{value} ({reason})" if reason else value)
        lines.append("| " + " | ".join(values) + " |")
    lines.extend([
        "", "## Final selection: every required artifact", "",
        "| Model | Boundary | Backend | Artifact | Status | Reason | Item / setup | Evidence origin |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ])
    for row in report.get("artifact_matrix") or []:
        values = [str(row.get(key) or "").replace("|", "\\|").replace("\n", " ")
                  for key in ("model_id", "boundary", "backend", "artifact_stage", "status", "reason", "item_id", "evidence_origin")]
        lines.append("| " + " | ".join(values) + " |")
    for change in report.get("selection_changes") or []:
        lines.extend(["", f"### {change.get('model_id')}: selection_changed", "", "```text",
                      f"previously_built_boundaries={json.dumps(change.get('previously_built_boundaries') or [])}",
                      f"currently_selected={change.get('currently_selected')}",
                      f"expected_cold_builds={json.dumps(change.get('expected_cold_builds') or [])}",
                      "```"])
    lines.extend([
        "",
        f"- Confirmed cache misses: {int(report.get('confirmed_miss_count') or 0)}",
        f"- Expected cold builds: {int(report.get('cold_builds_required') or 0)}",
        f"- Already completed cold builds: {int(report.get('completed_cold_build_count') or 0)}",
        f"- Policy-declared cold builds: {int(report.get('expected_cold_builds') or 0)}",
        f"- Unexpected cold builds: {int(report.get('unexpected_cold_builds') or 0)}",
        f"- Unknown probes: {int(report.get('unknown_count') or 0)}",
        f"- Known infeasible artifacts: {int(report.get('known_infeasible_count') or 0)}",
        "",
        "UNKNOWN is not counted as a cache hit or a cold build.",
        "Expected cold builds counts only outstanding builds; policy counts retain historical cache misses.",
        "KNOWN_INFEASIBLE is not counted as a cache hit or a cold build.",
        "KNOWN_INFEASIBLE prevents compiler dispatch for that exact artifact; other selected workloads remain eligible.",
        "",
    ])
    workspace = report.get("hailo_workspace_preflight") or {}
    if workspace.get("jobs"):
        lines.extend(["", "## DFC workspace before cold builds", "",
                      "Snapshot only; each actual dispatch repeats this check. No reservation or new cache identity.", "",
                      "| Model | Boundary | Family / stage | Status | Actual output directory | Free bytes | Required bytes | Reason |",
                      "| --- | --- | --- | --- | --- | --- | --- | --- |"])
        for job in workspace["jobs"]:
            fs = job.get("workspace") or {}
            calculation = job.get("calculation") or {}
            values = [job.get("model_id"), job.get("boundary"),
                      f"{job.get('backend')} / {job.get('artifact_stage')}", job.get("status"),
                      fs.get("requested_path") or (job.get("workspace_contract") or {}).get("out_dir"),
                      fs.get("free_bytes"), calculation.get("required_free_bytes"), job.get("reason")]
            lines.append("| " + " | ".join(str(value if value is not None else "unknown").replace("|", "\\|") for value in values) + " |")
        if workspace.get("blocked_job_count"):
            lines.extend(["", "Compilerkontext und Arbeitsraum sind getrennte Voraussetzungen. "
                          "Nicht gestartet: Arbeitsbereich zu klein oder nicht beschreibbar. "
                          "Unabhängige vorhandene Artefakte bleiben verwendbar.", ""])
    # Keep the labels referenced in one place for simple downstream renderers.
    del labels
    return "\n".join(lines)


def write_artifact_cache_preflight(
    report: Mapping[str, Any], *, output_dir: str | Path,
) -> dict[str, Path]:
    """Write JSON, CSV and Markdown views of one already-built report."""

    if report.get("schema") != SCHEMA:
        raise ValueError("artifact_cache_preflight_schema_invalid")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "artifact_cache_preflight.json"
    csv_path = root / "artifact_cache_preflight.csv"
    md_path = root / "artifact_cache_preflight.md"
    detail_csv_path = root / "artifact_cache_preflight_items.csv"
    json_path.write_text(
        json.dumps(dict(report), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    fieldnames = ["model_id"] + [
        key
        for role in ROLE_ORDER
        for key in (role, f"{role}_reason")
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(_matrix_csv_rows(report))
    detail_fields = ["model_id", "boundary", "backend", "artifact_stage", "status",
                     "reason", "item_id", "expectation", "artifact_path", "receipt_path", "identity", "evidence_origin",
                     "compiler_dispatch_allowed", "runtime_artifact_available"]
    with detail_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=detail_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(report.get("artifact_matrix") or [])
    md_path.write_text(
        render_artifact_cache_preflight_markdown(report), encoding="utf-8",
    )
    return {
        "artifact_cache_preflight_json": json_path,
        "artifact_cache_preflight_csv": csv_path,
        "artifact_cache_preflight_md": md_path,
        "artifact_cache_preflight_items_csv": detail_csv_path,
    }


__all__ = [
    "CacheProbeObservation",
    "EXPECTATION_COLD",
    "EXPECTATION_UNSPECIFIED",
    "EXPECTATION_WARM",
    "ROLE_DEEPX",
    "ROLE_HAILO10",
    "ROLE_HAILO8",
    "ROLE_ORDER",
    "ROLE_TRT_FULL",
    "ROLE_TRT_P2",
    "ROLE_TRT_P1",
    "STATUS_HIT",
    "STATUS_MISS",
    "STATUS_NOT_APPLICABLE",
    "STATUS_UNKNOWN",
    "STATUS_KNOWN_INFEASIBLE",
    "build_artifact_cache_preflight",
    "collect_model_artifact_cache_probes",
    "collect_selection_changes",
    "render_artifact_cache_preflight_log_lines",
    "render_artifact_cache_preflight_markdown",
    "resolve_generated_benchmark_suite_dir",
    "resolve_artifact_cache_preflight_policy",
    "known_negative_build_evidence",
    "unknown_probe",
    "write_artifact_cache_preflight",
]
