from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from ..benchmark.evaluation_profiles import load_evaluation_profile
from ..cache_verify_policy import CACHE_VERIFY_ONLY, cache_verify_guard
from ..config_values import parse_config_bool, validate_profile_config_booleans
from ..hailo_timeout_policy import parse_hailo_timeout_seconds
from .contracts import WorkflowOptions
from .start_snapshot import (
    resolve_runtime_profile_start_snapshot,
    validate_profile_start_snapshot,
)


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return bool(default)
    text = str(value).strip().lower()
    if text in {"", "none", "null"}:
        return bool(default)
    return text in {"1", "true", "yes", "y", "on"}


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(str(value or "").strip())
    except Exception:
        return int(default)


def _string_list(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value or "").replace(";", ",").split(",") if item.strip()]


def _frozen_target_has_remote(target: Any) -> bool:
    if not isinstance(target, Mapping):
        return False
    for remote in (
        target,
        target.get("remote"),
        target.get("runtime"),
    ):
        if not isinstance(remote, Mapping):
            continue
        if not _as_bool(remote.get("enabled"), True):
            continue
        if any(
            str(remote.get(key) or "").strip()
            for key in ("host", "ssh", "address")
        ):
            return True
    return False


def load_runtime_profile_snapshot(
    profile_request: str,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Load one profile exactly as the GUI start preview loads it."""

    loaded = load_evaluation_profile(profile_request, validate=True)
    if loaded is None or isinstance(loaded, tuple):
        raise FileNotFoundError(f"Evaluation profile not found: {profile_request}")
    loaded_payload = dict(getattr(loaded, "raw_profile", {}) or {})
    return resolve_runtime_profile_start_snapshot(
        profile_request=str(profile_request or ""),
        source_profile=dict(
            getattr(loaded, "source_profile", {}) or loaded_payload
        ),
        resolved_profile=loaded_payload,
        profile_id=str(
            getattr(loaded, "profile_id", "")
            or loaded_payload.get("name")
            or profile_request
        ),
        profile_path=str(
            getattr(loaded, "profile_path", "") or profile_request
        ),
        profile_source=str(getattr(loaded, "source", "") or "file"),
        options=None,
    )


def inline_remote_host_payload(remote: Mapping[str, Any]) -> Dict[str, Any]:
    """Resolve only host data embedded in a profile (no GUI state required)."""

    host_id = str(remote.get("host_id") or remote.get("id") or "").split(
        "—", 1
    )[0].strip()
    inline = remote.get("hosts") or remote.get("remote_hosts") or []
    if isinstance(inline, list):
        for item in inline:
            if not isinstance(item, Mapping):
                continue
            item_id = str(item.get("id") or item.get("label") or "").strip()
            if not host_id or item_id == host_id:
                return dict(item)
    raw_json = str(remote.get("host_json") or "").strip()
    if raw_json:
        try:
            payload = json.loads(raw_json)
        except Exception:
            payload = None
        if isinstance(payload, Mapping):
            return dict(payload)
    host = str(remote.get("host") or "").strip()
    if not host:
        return {}
    user = str(remote.get("user") or "").strip()
    if "@" in host and not user:
        user, host = host.split("@", 1)
    return {
        "id": host_id or str(remote.get("label") or host or "workflow_remote"),
        "label": str(remote.get("label") or host_id or host),
        "host": host,
        "user": user,
        "port": _as_int(remote.get("port"), 22),
        "remote_base_dir": str(
            remote.get("remote_base_dir") or "~/splitpoint_runs"
        ),
        "ssh_extra_args": str(remote.get("ssh_extra_args") or ""),
    }


def workflow_options_from_profile_snapshot(
    *,
    profile_request: str,
    out_root: str,
    start_snapshot: Mapping[str, Any],
    models_root: str = "",
    resume: bool = False,
    remote_host_payload: Optional[Mapping[str, Any]] = None,
    remote_working_dir: str = "",
    only_model_override: Optional[str] = None,
    execution_mode_override: str = "",
    run_id_override: Optional[str] = None,
    required_run_mode: str = "",
    require_fresh_run: bool = False,
) -> WorkflowOptions:
    """Build the shared GUI/CLI start options from one frozen profile snapshot.

    This function only resolves configuration. Execution remains exclusively in
    :class:`EvaluationWorkflowRunner` and its existing benchmark/native stages.
    """

    snapshot = validate_profile_start_snapshot(start_snapshot)
    payload = dict(snapshot.get("resolved_profile") or {})
    validate_profile_config_booleans(payload)
    wf = dict(payload.get("workflow") or payload.get("evaluation_workflow") or {})
    bex = dict(payload.get("benchmark_execution") or payload.get("runtime_benchmark") or {})
    remote = dict(payload.get("remote_execution") or payload.get("remote") or {})
    hailo = dict(payload.get("hailo_build") or {})
    guard = cache_verify_guard(payload)
    artifact_policy = CACHE_VERIFY_ONLY if guard else "normal"
    validation = dict(payload.get("validation") or {})
    hw_smoke = dict(payload.get("hardware_smoke") or {})
    hardware_cfg = dict(payload.get("hardware") or {})

    execution_mode = str(
        execution_mode_override
        or wf.get("execution_mode")
        or "generate_benchmarksets"
    ).strip().lower().replace("-", "_").replace(" ", "_")
    if execution_mode not in {
        "contracts_only",
        "generate_benchmarksets",
        "generate_and_run",
        "legacy_benchmarkset",
    }:
        execution_mode = "generate_benchmarksets"
    skip_benchmarks = _as_bool(
        wf.get("skip_runtime_benchmarks", wf.get("skip_benchmarks")),
        execution_mode != "generate_and_run",
    )
    if (
        execution_mode == "generate_and_run"
        and "skip_runtime_benchmarks" not in wf
        and "skip_benchmarks" not in wf
    ):
        skip_benchmarks = False

    selected_setups = _string_list(hardware_cfg.get("selected_setups"))
    resolved_targets = hardware_cfg.get("resolved_targets")
    if not isinstance(resolved_targets, list):
        resolved_targets = []
    frozen_remote_target = any(
        _frozen_target_has_remote(target) for target in resolved_targets
    )
    remote_enabled = bool(
        _as_bool(remote.get("enabled"), False)
        or selected_setups
        or frozen_remote_target
    )
    no_remote = False if selected_setups else not remote_enabled
    if remote_host_payload is None:
        selected_host = (
            inline_remote_host_payload(remote)
            if remote_enabled and not selected_setups
            else {}
        )
    else:
        selected_host = dict(remote_host_payload or {})
    remote_host_json = (
        json.dumps(selected_host, ensure_ascii=False) if selected_host else ""
    )
    remote_host_id = str(
        remote.get("host_id")
        or remote.get("id")
        or selected_host.get("id")
        or ""
    ).strip()

    profile_run_id = str(wf.get("run_id") or "").strip() or None
    if run_id_override is not None:
        run_id = str(run_id_override).strip() or None
        if not run_id:
            raise ValueError("An explicit --run-id must not be empty.")
        if not guard:
            raise ValueError(
                "A profile-driven --run-id override is reserved for an "
                "attested cache_verify_only diagnostic run."
            )
    else:
        run_id = profile_run_id
    profile_only_model = str(wf.get("only_model") or "").strip() or None
    only_model = (
        str(only_model_override).strip() or None
        if only_model_override is not None
        else profile_only_model
    )
    stop_after = str(wf.get("stop_after") or "").strip() or None
    max_models_raw = (
        _as_int(wf.get("max_models"), 0)
        if str(wf.get("max_models") or "").strip()
        else 0
    )
    result_sources = [
        str(item).strip()
        for item in list(wf.get("result_sources") or [])
        if str(item).strip()
    ]
    resolved_models_root = str(
        models_root or payload.get("models_root_hint") or ""
    ).strip()

    benchmark_warmup = max(0, _as_int(bex.get("warmup", 1), 1))
    benchmark_runs = max(
        1, _as_int(bex.get("runs", bex.get("iters", 3)), 3)
    )
    benchmark_backend = str(
        bex.get("backend") or ("remote" if remote_enabled else "auto")
    ).strip().lower()
    if benchmark_backend not in {"auto", "local", "remote"}:
        benchmark_backend = "auto"

    hailo_mode = str(hailo.get("mode") or "reuse_only").strip().lower().replace(
        "-", "_"
    )
    if hailo_mode == "reuse_build_missing":
        hailo_mode = "reuse_and_build_missing"
    if hailo_mode not in {
        "auto",
        "reuse_only",
        "reuse_and_build_missing",
        "request",
        "local",
        "venv",
        "wsl",
        CACHE_VERIFY_ONLY,
    }:
        hailo_mode = "reuse_only"
    if "targets" in hailo:
        # An explicit empty list is the physical no-Hailo contract used by a
        # DeepX-only profile.  Only an absent key receives the compatibility
        # default derived from hw_arch.
        hailo_targets = _string_list(hailo.get("targets"))
    else:
        hailo_targets = [
            str(hailo.get("hw_arch") or "hailo8").strip() or "hailo8"
        ]

    remote_reuse_bundle = _as_bool(remote.get("reuse_bundle"), True)
    remote_resume = _as_bool(remote.get("resume"), True)
    return WorkflowOptions(
        profile=str(profile_request),
        out=str(Path(out_root).expanduser()),
        models_root=resolved_models_root,
        resume=bool(resume),
        dry_run=_as_bool(wf.get("dry_run"), False),
        stop_after=stop_after,
        only_model=only_model,
        include_reserve=_as_bool(wf.get("include_reserve"), False),
        skip_benchmarks=skip_benchmarks,
        no_remote=no_remote,
        run_id=run_id,
        max_models=max_models_raw if max_models_raw > 0 else None,
        force_stage=[],
        no_model_hash=_as_bool(wf.get("no_model_hash"), True),
        skip_analysis=False,
        execution_mode=execution_mode,
        artifact_policy=artifact_policy,
        benchmark_results_root=str(bex.get("results_root") or "").strip(),
        result_sources=result_sources,
        benchmark_provider=str(
            bex.get("provider") or bex.get("benchmark_provider") or ""
        ).strip(),
        benchmark_warmup=benchmark_warmup,
        benchmark_runs=benchmark_runs,
        benchmark_timeout_s=max(0, _as_int(bex.get("timeout_s", 0), 0)),
        benchmark_execution_backend=benchmark_backend,
        benchmark_preset=str(bex.get("preset") or "auto"),
        benchmark_image=str(bex.get("image") or "default"),
        benchmark_extra_args=[
            str(item)
            for item in list(bex.get("extra_args") or [])
            if str(item).strip()
        ],
        hailo_build_mode=hailo_mode,
        hailo_hw_arch=str(
            hailo.get("hw_arch")
            or (hailo_targets[0] if hailo_targets else "hailo8")
            or "hailo8"
        ).strip(),
        hailo_build_targets=hailo_targets,
        hailo_build_backend=str(hailo.get("backend") or "auto").strip(),
        hailo_build_timeout_s=parse_hailo_timeout_seconds(
            hailo.get("timeout_s"),
            default=3600,
            label="hailo_build.timeout_s",
        ),
        hailo_build_full=_as_bool(hailo.get("build_full"), True),
        hailo_build_part1=_as_bool(hailo.get("build_part1"), True),
        hailo_build_part2=_as_bool(hailo.get("build_part2"), True),
        hailo_preset=str(hailo.get("preset") or "quick").strip(),
        hailo_optimization_level=max(
            0, _as_int(hailo.get("optimization_level", 0), 0)
        ),
        hailo_calib_dir=str(hailo.get("calib_dir") or "").strip(),
        hailo_calib_count=max(
            0, _as_int(hailo.get("calib_count", 16), 16)
        ),
        hailo_calib_batch_size=max(
            1, _as_int(hailo.get("calib_batch_size", 8), 8)
        ),
        hailo_force_build=parse_config_bool(hailo.get("force_build", False), field="hailo_build.force_build"),
        hailo_keep_artifacts=_as_bool(hailo.get("keep_artifacts"), True),
        hardware_setups_file=str(
            hardware_cfg.get("setups_file")
            or hardware_cfg.get("config_file")
            or payload.get("hardware_setups_file")
            or ""
        ).strip(),
        hardware_setup_ids=selected_setups,
        hardware_group_ids=_string_list(hardware_cfg.get("selected_groups")),
        validation_mode=str(validation.get("mode") or "summary_only").strip(),
        validation_require_explicit=_as_bool(
            validation.get("require_explicit"), False
        ),
        hardware_smoke_mode=str(
            hw_smoke.get("mode") or "summary_only"
        ).strip(),
        remote_host_json=remote_host_json,
        remote_host_id=remote_host_id,
        remote_host=str(selected_host.get("host") or remote.get("host") or "").strip(),
        remote_user=str(selected_host.get("user") or remote.get("user") or "").strip(),
        remote_port=max(
            1, _as_int(selected_host.get("port") or remote.get("port"), 22)
        ),
        remote_base_dir=str(
            selected_host.get("remote_base_dir")
            or remote.get("remote_base_dir")
            or "~/splitpoint_runs"
        ).strip(),
        remote_ssh_extra_args=str(
            selected_host.get("ssh_extra_args")
            or remote.get("ssh_extra_args")
            or ""
        ).strip(),
        remote_working_dir=str(
            remote_working_dir
            or (Path(out_root).expanduser() / "RemoteBenchmarkRuns")
        ),
        remote_provider=str(remote.get("provider") or "auto").strip(),
        remote_venv=str(
            remote.get("remote_venv") or remote.get("venv") or ""
        ).strip(),
        remote_transfer_mode=str(remote.get("transfer_mode") or "bundle").strip(),
        remote_reuse_bundle=remote_reuse_bundle,
        remote_no_reuse_bundle=not remote_reuse_bundle,
        remote_resume=remote_resume,
        remote_no_resume=not remote_resume,
        remote_repeats=max(1, _as_int(remote.get("repeats", 1), 1)),
        remote_warmup=max(
            0, _as_int(remote.get("warmup", benchmark_warmup), benchmark_warmup)
        ),
        remote_iters=max(
            1, _as_int(remote.get("iters", benchmark_runs), benchmark_runs)
        ),
        remote_timeout_s=max(0, _as_int(remote.get("timeout_s", 0), 0)),
        remote_add_args=str(remote.get("add_args") or "").strip(),
        remote_throughput_frames=max(
            1, _as_int(remote.get("throughput_frames", 24), 24)
        ),
        remote_throughput_warmup_frames=max(
            0, _as_int(remote.get("throughput_warmup_frames", 6), 6)
        ),
        remote_throughput_queue_depth=max(
            1, _as_int(remote.get("throughput_queue_depth", 2), 2)
        ),
        remote_validation_images=str(remote.get("validation_images") or ""),
        remote_validation_max_images=max(
            0, _as_int(remote.get("validation_max_images", 0), 0)
        ),
        remote_validation_reference_mode=str(
            remote.get("validation_reference_mode") or "auto"
        ),
        remote_mini_coco_ap50=_as_bool(remote.get("mini_coco_ap50"), False),
        remote_mini_classification_eval=_as_bool(
            remote.get("mini_classification_eval"), False
        ),
        remote_benchmark_task=str(
            remote.get("benchmark_task") or bex.get("preset") or "auto"
        ).strip(),
        required_run_mode=str(required_run_mode or "").strip(),
        require_fresh_run=bool(require_fresh_run),
        profile_start_snapshot=snapshot,
    )
